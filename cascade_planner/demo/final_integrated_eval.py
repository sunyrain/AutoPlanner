"""FINAL integrated retrosynthesis demo & honest evaluation.

Stack:
  • Chemistry expander: AiZynthFinder USPTO ONNX (subprocess, .venv_aizynth).
  • Enzyme expander A : EnzExpand-A trained ONLY on our snapshot (≤633 trainable).
  • Enzyme expander B : EnzExpand-EM trained on UNION
                         (our snapshot + EnzymeMap 16k templates),
                         giving ~14k–17k training samples for the MLP.
  • Condition heads : EC1 / transformation / catalyst / solvent logreg + T-by-EC1.

For each held-out DOI:
  1. Drop ALL of that DOI's steps from the training set.
  2. Retrain EnzExpand-A and EnzExpand-EM (cached templates → fast).
  3. Retrain condition heads on the remainder.
  4. Run USPTO (always clean), EnzExpand-A, EnzExpand-EM on the query product.
  5. Record top-K rank of the truth precursor set for each expander.
  6. Predict EC1 / transformation / T on the true rxn (DOI-OOD).

Outputs:
  results/final_loo_eval.csv — per-query rows
  results/final_report.md    — human-readable summary
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.linear_model import LogisticRegression

from cascade_planner.conditions.predict_conditions import _topk_label
from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.expand.enz_template import (apply_template_to_product,
                                                  canon_set, extract_templates,
                                                  main_product, morgan2,
                                                  predict_topk, train as train_mlp)
from cascade_planner.expand.enzymemap_loader import (extract_templates_from_enzymemap,
                                                      load_filtered)
from cascade_planner.training.featurize_v2 import drfp_batch

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
AIZ_PY = ROOT / ".venv_aizynth" / "Scripts" / "python.exe"
AIZ_CONFIG = ROOT / "aizdata" / "config.yml"


def fit_clf(X, y):
    cls = sorted(set(y)); idx = {c: i for i, c in enumerate(cls)}
    yi = np.array([idx[v] for v in y])
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    m.fit(X, yi); return m, cls


def predict_top1(m, cls, x):
    p = m.predict_proba(x.reshape(1, -1))[0]; i = int(np.argmax(p))
    return cls[i], float(p[i])


def fit_reg_by_ec1(values, ec1_arr):
    out = {}
    for e in set(ec1_arr):
        msk = ec1_arr == e
        if msk.sum() > 0:
            out[e] = float(np.mean(values[msk]))
    out["__global__"] = float(np.mean(values))
    return out


def call_uspto(products, topk=50):
    if not AIZ_PY.exists():
        print("[uspto] .venv_aizynth not found"); return [[] for _ in products]
    bridge = ROOT / "cascade_planner" / "demo" / "aizynth_bridge.py"
    payload = json.dumps({"products": products, "topk": topk})
    proc = subprocess.run([str(AIZ_PY), str(bridge), str(AIZ_CONFIG)],
                          input=payload, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        print(f"[uspto] FAIL\n{proc.stderr[-300:]}"); return [[] for _ in products]
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return [r["predictions"] for r in data["results"]]


def build_template_pool(local_steps, enzymemap_rows):
    """Return (X, y, tpls_sorted, tpl_to_id) ready to train TemplateMLP.

    Both sources mapped into one shared template-id space.
    Each sample = (product_morgan2, template_id).
    """
    cache_path = RESULTS / "enzexpand_atommap_cache.json"
    pairs = extract_templates(local_steps, mapped_cache=cache_path)
    local_samples = []
    for i, t in pairs:
        prod = main_product(local_steps[i].rxn_smiles)
        if prod:
            local_samples.append((prod, t))
    em_samples = [(r["product"], r["template"]) for r in enzymemap_rows]

    cnt = collections.Counter(t for _, t in local_samples + em_samples)
    tpls_sorted = [t for t, c in cnt.most_common() if c >= 2]
    tpl_to_id = {t: i for i, t in enumerate(tpls_sorted)}
    samples = [(p, tpl_to_id[t]) for p, t in (local_samples + em_samples)
               if t in tpl_to_id]
    if not samples:
        return None, None, [], {}
    X = np.stack([morgan2(p) for p, _ in samples])
    y = np.array([t for _, t in samples], dtype=np.int64)
    return X, y, tpls_sorted, tpl_to_id


def first_match_rank(pred_iter, truth):
    seen = []
    for cs in pred_iter:
        fs = frozenset(cs)
        if fs not in seen:
            seen.append(fs)
            if fs == truth:
                return len(seen)
    return None


def expand_with_mlp(model, tpls_sorted, prod, k, max_outcomes=3):
    if model is None or not prod:
        return
    x = morgan2(prod).reshape(1, -1)
    tk, _ = predict_topk(model, x, k=k)
    tk = tk[0]
    for lid in tk:
        tpl = tpls_sorted[int(lid)]
        for cs in apply_template_to_product(tpl, prod, max_outcomes=max_outcomes):
            yield cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--topk-tpl", type=int, default=50)
    ap.add_argument("--topk-uspto", type=int, default=50)
    ap.add_argument("--em-cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print(" FINAL integrated retrosynthesis: USPTO + EnzExpand-A + EnzExpand-EM")
    print("=" * 70)
    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    print(f"  total steps={len(steps)}, enz={sum(1 for s in steps if s.ec_number)}")

    print("\n[load] EnzymeMap (cached if available)")
    em_df = load_filtered(min_quality=0.95, only_single=True, ec1_balance=args.em_cap)
    em_rows = extract_templates_from_enzymemap(em_df)

    # Pick queries: 4 enz from each of EC1-EC5 (frequent classes) + 4 chem.
    # Avoid CO2 / single-atom products which are degenerate.
    rng = np.random.default_rng(args.seed)
    by_ec1 = collections.defaultdict(list)
    for s in steps:
        if not s.ec_number:
            continue
        e = s.ec_number.split(".")[0]
        p = main_product(s.rxn_smiles)
        if p and len(p) > 4:
            by_ec1[e].append(s)
    chem = [s for s in steps if not s.ec_number and main_product(s.rxn_smiles) and len(main_product(s.rxn_smiles)) > 4]

    queries = []
    per_ec = max(1, (args.n_queries - 4) // 5)
    for e in ["1", "2", "3", "4", "5"]:
        if not by_ec1[e]:
            continue
        idxs = rng.choice(len(by_ec1[e]), size=min(per_ec, len(by_ec1[e])), replace=False)
        for i in idxs:
            queries.append(("enz_EC" + e, by_ec1[e][int(i)]))
    chem_idxs = rng.choice(len(chem), size=min(4, len(chem)), replace=False)
    for i in chem_idxs:
        queries.append(("chem", chem[int(i)]))
    queries = queries[: args.n_queries]
    print(f"\n[queries] n={len(queries)}")

    products = [main_product(s.rxn_smiles) for _, s in queries]
    print("\n[uspto] one-shot batch USPTO call (always clean)")
    t0 = time.time()
    uspto_per_q = call_uspto(products, topk=args.topk_uspto)
    print(f"  uspto done in {time.time()-t0:.1f}s")

    rows = []
    for q_idx, ((tag, s_q), uspto_preds) in enumerate(zip(queries, uspto_per_q)):
        prod = main_product(s_q.rxn_smiles)
        truth = canon_set(s_q.rxn_smiles.split(">>", 1)[0]) if ">>" in s_q.rxn_smiles else frozenset()
        held = s_q.doi
        train_steps = [s for s in steps if s.doi != held]
        train_enz_local = [s for s in train_steps if s.ec_number]
        print(f"\n[Q{q_idx+1:02d} {tag} DOI={held}] truth EC={s_q.ec_number or 'chem'}")

        # ---- Local-only EnzExpand-A ----
        Xa, ya, tplsA, _ = build_template_pool(train_enz_local, [])
        modelA = train_mlp(Xa, ya, n_tpl=len(tplsA), epochs=30, hidden=1024,
                           dropout=0.4, batch=256, seed=0, verbose=False) if Xa is not None else None

        # ---- Union EnzExpand-EM (local + EnzymeMap) ----
        Xb, yb, tplsB, _ = build_template_pool(train_enz_local, em_rows)
        modelB = train_mlp(Xb, yb, n_tpl=len(tplsB), epochs=20, hidden=1024,
                           dropout=0.4, batch=512, seed=0, verbose=False) if Xb is not None else None
        print(f"  pools: local={Xa.shape[0] if Xa is not None else 0}/{len(tplsA)} tpls, "
              f"union={Xb.shape[0] if Xb is not None else 0}/{len(tplsB)} tpls")

        # ---- Condition heads ----
        rxns = [s.rxn_smiles for s in train_steps]
        Xd = drfp_batch(rxns)
        T_vals = np.array([s.temperature_c if s.temperature_c is not None else np.nan for s in train_steps], dtype=float)
        ec1_all = np.array([s.ec_number.split(".")[0] if s.ec_number else "" for s in train_steps])
        T_by_ec1 = fit_reg_by_ec1(T_vals[~np.isnan(T_vals)], ec1_all[~np.isnan(T_vals)])
        m_ec = np.array([bool(s.ec_number) for s in train_steps])
        ec1_clf, ec1_classes = fit_clf(Xd[m_ec],
                                        np.array([s.ec_number.split(".")[0]
                                                  for s in train_steps if s.ec_number]))
        m_tr = np.array([bool(s.transformation_superclass) for s in train_steps])
        tr_y, _ = _topk_label([s.transformation_superclass for s in train_steps if s.transformation_superclass], 12)
        tr_clf, tr_classes = fit_clf(Xd[m_tr], tr_y)

        # ---- Score ranks ----
        uspto_rank = first_match_rank(
            (frozenset(p["reactants_dot"].split(".")) for p in uspto_preds), truth)
        enz_a_rank = first_match_rank(expand_with_mlp(modelA, tplsA, prod, args.topk_tpl), truth)
        enz_em_rank = first_match_rank(expand_with_mlp(modelB, tplsB, prod, args.topk_tpl), truth)

        x_true = drfp_batch([s_q.rxn_smiles])[0]
        true_ec1 = s_q.ec_number.split(".")[0] if s_q.ec_number else None
        pred_ec1, conf_ec1 = predict_top1(ec1_clf, ec1_classes, x_true)
        true_tr = s_q.transformation_superclass
        pred_tr, conf_tr = predict_top1(tr_clf, tr_classes, x_true)
        T_pred = T_by_ec1.get(pred_ec1, T_by_ec1["__global__"])
        T_err = abs(s_q.temperature_c - T_pred) if s_q.temperature_c is not None else None

        print(f"  ranks  USPTO={uspto_rank}  EnzExpand-A={enz_a_rank}  EnzExpand-EM={enz_em_rank}")
        print(f"  EC1 {true_ec1}->{pred_ec1}({conf_ec1:.2f})  "
              f"transformation {true_tr}->{pred_tr}({conf_tr:.2f})  "
              f"T true={s_q.temperature_c} pred={T_pred:.1f}")

        rows.append(dict(
            tag=tag, doi=held, product=prod,
            truth_precursors=".".join(sorted(truth)), truth_ec=s_q.ec_number,
            uspto_rank=uspto_rank, enzA_rank=enz_a_rank, enzEM_rank=enz_em_rank,
            true_ec1=true_ec1, pred_ec1=pred_ec1, ec1_correct=(pred_ec1 == true_ec1) if true_ec1 else None,
            true_tr=true_tr, pred_tr=pred_tr, tr_correct=(pred_tr == true_tr) if true_tr else None,
            true_T=s_q.temperature_c, pred_T=round(T_pred, 1), T_abs_err=T_err,
        ))

    df = pd.DataFrame(rows)
    out_csv = RESULTS / "final_loo_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")

    # ---------- aggregate report ----------
    enz = df[df["tag"].str.startswith("enz")]
    chem = df[df["tag"] == "chem"]

    def hit(s, k):
        return (s.notna() & (s <= k)).mean()

    md = ["# Final integrated honest evaluation\n",
          f"_data: {args.data}_  ",
          f"_queries: n={len(df)} (enzymatic={len(enz)}, chemical={len(chem)})_  ",
          f"_topk: USPTO={args.topk_uspto}, EnzExpand-*={args.topk_tpl}_  ",
          f"_EnzymeMap pool: {len(em_rows)} templates_\n",
          "## Top-K recall (leave-one-DOI-out)\n",
          "| slice | model | top1 | top5 | top10 | top25 | top50 |",
          "|---|---|---|---|---|---|---|"]
    for label, sub in [("enzymatic", enz), ("chemical", chem)]:
        for col, name in [("uspto_rank", "AiZynth-USPTO"),
                          ("enzA_rank", "EnzExpand-A (local 633)"),
                          ("enzEM_rank", "EnzExpand-EM (local + 16k EnzymeMap)")]:
            row = f"| {label} | {name} | "
            for k in (1, 5, 10, 25, 50):
                row += f"{hit(sub[col], k)*100:.0f}% | "
            md.append(row.rstrip("| ") + " |")

    md += ["\n## Condition heads (DOI-OOD per query)\n",
           f"- EC1 accuracy (enz queries): {enz['ec1_correct'].mean()*100:.1f}%",
           f"- transformation accuracy: {df['tr_correct'].mean()*100:.1f}%",
           f"- T MAE on queries with measured T: "
           f"{df['T_abs_err'].dropna().mean():.2f} °C  (n={df['T_abs_err'].notna().sum()})\n",
           "## Per-query detail\n",
           df.to_markdown(index=False)]
    out_md = RESULTS / "final_report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[save] {out_md}\n")
    print("\n========================== FINAL REPORT ==========================")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
