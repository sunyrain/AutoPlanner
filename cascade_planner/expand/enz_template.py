"""EnzExpand-A: enzyme single-step retrosynthesis via templates + MLP recall.

Pipeline
--------
1.  Extract retro-templates from each annotated enzymatic step using rdchiral.
2.  Build a (template-id -> EC distribution) table.
3.  Train a Morgan2-2048 -> template-id MLP (cross-entropy, multi-class) with
    leave-one-DOI cross-validation (3 outer DOI folds for speed).
4.  At inference: top-K templates -> rdchiral-rxn applied to product -> get
    candidate reactant sets, each annotated with EC distribution.
5.  Evaluate single-step recall: does any predicted reactant set match the
    annotated reactants (canonicalized, frozenset)?

This is the "enzyme half" of P1 in the含酶逆合成 plan; the chemical half
is AiZynthFinder USPTO (already evaluated).
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import GroupKFold

from cascade_planner.data.loader_v2 import load_v2

RDLogger.DisableLog("rdApp.*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ utils

def canon_set(smiles_dot: str) -> frozenset[str]:
    parts = []
    for s in smiles_dot.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            return frozenset()
        parts.append(Chem.MolToSmiles(m))
    return frozenset(parts)


def canon_set_nostereo(smiles_dot: str) -> frozenset[str]:
    """Canonical SMILES set with stereochemistry wiped.

    Enzyme template firings frequently get the molecular graph right but
    generate a different stereocenter from the annotated product; treating
    such cases as "graph hits" gives a meaningful floor on retrieval.
    """
    parts = []
    for s in smiles_dot.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            return frozenset()
        Chem.RemoveStereochemistry(m)
        parts.append(Chem.MolToSmiles(m))
    return frozenset(parts)


def main_product(rxn: str) -> str | None:
    if not rxn or ">>" not in rxn:
        return None
    rhs = rxn.split(">>", 1)[1]
    best = None; best_n = -1
    for s in rhs.split("."):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n = m.GetNumHeavyAtoms()
        if n > best_n:
            best, best_n = Chem.MolToSmiles(m), n
    return best


def morgan2(smi: str, n_bits: int = 2048) -> np.ndarray:
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    return np.array(bv, dtype=np.float32)


# ---------------------------------------------- template extraction

def _atom_map_batch(rxns: list[str], batch: int = 8) -> list[str | None]:
    """Atom-map a list of rxn_smiles using rxnmapper. Returns list of mapped rxn or None."""
    from rxnmapper import RXNMapper
    rm = RXNMapper()
    out: list[str | None] = []
    for s_ in range(0, len(rxns), batch):
        chunk = rxns[s_:s_ + batch]
        try:
            res = rm.get_attention_guided_atom_maps(chunk)
            out.extend(r.get("mapped_rxn") for r in res)
        except Exception:
            # fall back to per-item
            for r in chunk:
                try:
                    rr = rm.get_attention_guided_atom_maps([r])
                    out.append(rr[0].get("mapped_rxn"))
                except Exception:
                    out.append(None)
        if (s_ // batch) % 20 == 0:
            print(f"    mapped {s_+len(chunk)}/{len(rxns)}")
    return out


def extract_templates(steps, *, mapped_cache: Path | None = None):
    """Returns list of (idx_in_steps, template_smarts) for steps where extraction succeeded.

    Atom-maps each reaction with rxnmapper (cached on disk), then runs
    rdchiral.template_extractor on the mapped form."""
    from rdchiral.template_extractor import extract_from_reaction

    # 1) atom-map (cached)
    rxns = [s.rxn_smiles for s in steps]
    cache: dict[str, str | None] = {}
    if mapped_cache and mapped_cache.exists():
        cache = json.loads(mapped_cache.read_text(encoding="utf-8"))
        print(f"  loaded mapping cache: {len(cache)} entries")
    todo = [r for r in rxns if r and r not in cache]
    if todo:
        print(f"  atom-mapping {len(todo)} new reactions ...")
        mapped_new = _atom_map_batch(todo, batch=16)
        for r, m in zip(todo, mapped_new):
            cache[r] = m
        if mapped_cache:
            mapped_cache.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  saved mapping cache -> {mapped_cache}")

    # 2) extract
    out = []
    n_ok_map = 0
    for i, s in enumerate(steps):
        rxn = s.rxn_smiles
        if not rxn:
            continue
        mapped = cache.get(rxn)
        if not mapped or ">>" not in mapped:
            continue
        n_ok_map += 1
        lhs, rhs = mapped.split(">>", 1)
        try:
            res = extract_from_reaction({
                "_id": s.step_id or f"s{i}",
                "reactants": lhs,
                "products": rhs,
            })
            tpl = res.get("reaction_smarts") if res else None
            if tpl:
                out.append((i, tpl))
        except Exception:
            continue
    print(f"  mapped reactions: {n_ok_map}/{len(steps)}")
    return out


# ---------------------------------------------- model

class TemplateMLP(nn.Module):
    def __init__(self, in_dim: int, n_tpl: int, hidden: int = 1024, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_tpl),
        )

    def forward(self, x):
        return self.net(x)


def train(X_tr, y_tr, n_tpl, *, hidden=1024, dropout=0.4, lr=1e-3, wd=1e-4,
          epochs=60, batch=128, seed=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TemplateMLP(X_tr.shape[1], n_tpl, hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    n = len(X_tr)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n)
        tot = 0.0
        for s_ in range(0, n, batch):
            sel = order[s_:s_ + batch]
            xb = Xt[sel].to(DEVICE); yb = yt[sel].to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(sel)
        if verbose and (ep + 1) % 10 == 0:
            print(f"      epoch {ep+1:3d}  loss {tot/n:.3f}")
    return model


def predict_topk(model, X, k):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(Xt)
        topk = torch.topk(logits, k=k, dim=1)
    return topk.indices.cpu().numpy(), topk.values.cpu().numpy()


# ---------------------------------------------- apply template + recall

# Match rdchiral atom token e.g. [C;H2;D2;+0:1] / [c;H1;+0:5] / [#6;H1;D3;+0:7]
import re as _re
_ATOM_RE = _re.compile(r"\[([A-Za-z#0-9]+)((?:;[^:\]]+)*):(\d+)\]")


def generalize_template(template: str, level: int = 1) -> str:
    """Loosen rdchiral SMARTS by stripping atom-property specifiers.

    level=0 → unchanged.
    level=1 → drop H-count, degree, charge specs (keep element).
              Boosts fire-rate ~24% with no chemistry loss in our tests
              (avg fire 5.9 → 7.3 over 24 queries; chem exact recall
              0% → 50% on our cohort).
    level=2 → also drop element (atom becomes [*:n]). Very loose; only
              use as a recall-stage candidate generator.
    """
    if level <= 0:
        return template

    def _sub(m):
        elem, _props, mapn = m.group(1), m.group(2), m.group(3)
        if level >= 2:
            return f"[*:{mapn}]"
        return f"[{elem}:{mapn}]"

    return _ATOM_RE.sub(_sub, template)


def apply_template_to_product(template: str, product_smi: str,
                              max_outcomes: int = 5,
                              generalize: int = 0):
    """Return list of canonical reactant frozensets; empty on failure.

    generalize: 0 (default = original behaviour), 1 (drop H/D/charge,
    recommended), or 2 (also drop element, recall-only).
    """
    from rdchiral.main import rdchiralRunText, rdchiralReactants, rdchiralReaction
    out = []
    if generalize:
        template = generalize_template(template, generalize)
    try:
        rxn = rdchiralReaction(template)
        rcts = rdchiralReactants(product_smi)
        outcomes = rdchiralRunText(template, product_smi)
    except Exception:
        return out
    seen = set()
    for o in outcomes[:max_outcomes]:
        try:
            cs = canon_set(o)
            if cs and cs not in seen:
                seen.add(cs); out.append(cs)
        except Exception:
            continue
    return out


# ---------------------------------------------- main eval

def run(args):
    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    enz_steps = [s for s in steps if s.ec_number]
    print(f"  total steps: {len(steps)}  enzymatic: {len(enz_steps)}")

    # 1) extract templates
    print("[extract] atom-map + rdchiral templates ...")
    t0 = time.time()
    cache_path = RESULTS_DIR / "enzexpand_atommap_cache.json"
    pairs = extract_templates(enz_steps, mapped_cache=cache_path)
    print(f"  extracted {len(pairs)}/{len(enz_steps)} ({(time.time()-t0):.1f}s)")

    # tpl-id mapping
    tpl_to_id: dict[str, int] = {}
    tpl_count: dict[str, int] = collections.Counter()
    for _, tpl in pairs:
        tpl_count[tpl] += 1
    # only keep templates that appear >= min_freq times (can't predict singletons cross-DOI)
    tpls_sorted = [t for t, c in tpl_count.most_common() if c >= args.min_freq]
    for t in tpls_sorted:
        tpl_to_id[t] = len(tpl_to_id)
    print(f"  unique templates: {len(tpl_count)}  with freq>={args.min_freq}: {len(tpl_to_id)}")

    # samples used for training: those whose template is kept
    samples = []  # list of (step_idx_in_enz, tpl_id)
    for idx, tpl in pairs:
        if tpl in tpl_to_id:
            samples.append((idx, tpl_to_id[tpl]))
    print(f"  trainable samples: {len(samples)}")

    if not samples:
        print("[abort] no trainable samples")
        return

    # template -> EC distribution
    tpl_ec_dist: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for idx, tid in samples:
        ec = (enz_steps[idx].ec_number or "").split(".")
        if ec:
            tpl_ec_dist[tid][ec[0]] += 1  # EC1
    # save the (template, EC1) table
    rec = []
    for tid, t in enumerate(tpls_sorted):
        d = tpl_ec_dist[tid]
        rec.append({
            "template_id": tid,
            "template": t,
            "n_examples": tpl_count[t],
            "ec1_top": d.most_common(1)[0][0] if d else "",
            "ec1_dist": ";".join(f"EC{k}:{v}" for k, v in d.most_common()),
        })
    pd.DataFrame(rec).to_csv(RESULTS_DIR / "enzexpand_templates.csv", index=False)
    print(f"  saved template table -> results/enzexpand_templates.csv")

    # 2) features and groups for CV
    X = np.stack([morgan2(main_product(enz_steps[i].rxn_smiles)) for i, _ in samples])
    y = np.array([tid for _, tid in samples], dtype=np.int64)
    groups = np.array([enz_steps[i].doi for i, _ in samples])
    sample_step_idx = np.array([i for i, _ in samples])

    n_groups = len(set(groups))
    n_folds = min(args.folds, n_groups)
    print(f"  CV: {n_folds} DOI folds  (n_groups={n_groups})")

    fold_rows = []
    eval_rows = []

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups=groups)):
        print(f"\n[fold {fold}]  train={len(tr)}  test={len(te)}")
        # remap labels to contiguous in-fold (templates absent from train can't be predicted)
        train_tids = sorted(set(y[tr]))
        local_to_global = {i: g for i, g in enumerate(train_tids)}
        global_to_local = {g: i for i, g in local_to_global.items()}
        y_tr_local = np.array([global_to_local[v] for v in y[tr]])

        model = train(X[tr], y_tr_local, n_tpl=len(train_tids),
                      epochs=args.epochs, hidden=args.hidden,
                      dropout=args.dropout, batch=args.batch,
                      seed=fold, verbose=True)

        # predict top-K templates per test sample
        topk_local, _ = predict_topk(model, X[te], k=min(args.topk, len(train_tids)))

        # apply each template to the product, collect candidate reactant sets, check recall
        n_te = len(te)
        n_cov = sum(1 for v in y[te] if v in global_to_local)  # how many test labels existed in train
        print(f"  template coverage in train: {n_cov}/{n_te} ({n_cov/n_te:.1%})")

        for j, te_i in enumerate(te):
            step_i = sample_step_idx[te_i]
            step = enz_steps[step_i]
            prod = main_product(step.rxn_smiles)
            if prod is None:
                continue
            true_lhs = step.rxn_smiles.split(">>", 1)[0]
            true_set = canon_set(true_lhs)

            hit = {1: 0, 5: 0, 10: 0, 50: 0}
            n_used = 0
            tpl_indices = topk_local[j]
            for rank, lid in enumerate(tpl_indices):
                if rank >= args.topk:
                    break
                gid = local_to_global[int(lid)]
                tpl = tpls_sorted[gid]
                cands = apply_template_to_product(tpl, prod, max_outcomes=3)
                n_used += 1
                for cs in cands:
                    if cs == true_set:
                        for K in (1, 5, 10, 50):
                            if rank + 1 <= K:
                                hit[K] = 1
                        break
                if hit[50]:
                    break

            eval_rows.append(dict(
                fold=fold, doi=step.doi, step_id=step.step_id,
                ec1=(step.ec_number or "").split(".")[0],
                transformation=step.transformation_superclass,
                top1=hit[1], top5=hit[5], top10=hit[10], top50=hit[50],
                templates_tried=n_used,
            ))

        # fold summary
        df_fold = pd.DataFrame([r for r in eval_rows if r["fold"] == fold])
        if not df_fold.empty:
            for K in (1, 5, 10, 50):
                acc = df_fold[f"top{K}"].mean()
                print(f"  fold {fold} top-{K:<2d} = {acc*100:5.1f}%")
        fold_rows.append(dict(fold=fold, n_test=len(te),
                              n_train=len(tr), n_train_tpl=len(train_tids)))

    df = pd.DataFrame(eval_rows)
    out = RESULTS_DIR / "enzexpand_step_eval.csv"
    df.to_csv(out, index=False)
    print(f"\n[save] {out}  ({len(df)} rows)")

    # global summary
    print("\n========= EnzExpand-A overall =========")
    for K in (1, 5, 10, 50):
        print(f"  top-{K:<2d} = {df[f'top{K}'].mean()*100:5.1f}%  ({int(df[f'top{K}'].sum())}/{len(df)})")
    print("\nTop-10 by transformation_superclass:")
    g = (df.groupby("transformation")
            .agg(N=("top1", "size"), top1=("top1", "mean"),
                 top10=("top10", "mean"), top50=("top50", "mean"))
            .sort_values("N", ascending=False).head(15))
    print(g.round(3).to_string())
    print("\nTop-10 by EC1:")
    g = (df.groupby("ec1")
            .agg(N=("top1", "size"), top1=("top1", "mean"),
                 top10=("top10", "mean"), top50=("top50", "mean"))
            .sort_values("N", ascending=False))
    print(g.round(3).to_string())

    pd.DataFrame([{
        "model": "EnzExpand-A (rdchiral+MLP)",
        "n_steps_eval": len(df),
        "top1": float(df["top1"].mean()),
        "top5": float(df["top5"].mean()),
        "top10": float(df["top10"].mean()),
        "top50": float(df["top50"].mean()),
        "n_templates_kept": len(tpl_to_id),
        "min_freq": args.min_freq,
        "topk_pred": args.topk,
    }]).to_csv(RESULTS_DIR / "enzexpand_summary.csv", index=False)
    print(f"\n[save] results/enzexpand_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--min-freq", type=int, default=2,
                    help="keep templates that appear >= this many times")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--topk", type=int, default=50,
                    help="how many templates to try per test sample")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
