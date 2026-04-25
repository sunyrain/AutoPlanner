"""End-to-end demo: given a target product, propose retrosynthetic step(s)
and predict per-step conditions.

Pipeline (single-step, can iterate):
  1. Train EnzExpand-A on ALL enzymatic steps (no held-out fold).
  2. Train condition predictors on ALL steps (T mean-by-EC1, EC1 logreg,
     transformation logreg, catalyst_class logreg, solvent logreg).
  3. For each query product (5 random product SMILES from the snapshot,
     held-out from eval just for narrative):
       a. Predict top-K templates → apply with rdchiral → enzymatic candidates
          (each annotated with the template's dominant EC1 and an EC2 guess
          from the EC2 logreg over the produced rxn).
       b. For each top candidate, build the candidate rxn_smiles and run all
          condition predictors on it.
       c. Print a clean per-step proposal: precursor(s), enzyme class,
          T, pH, solvent, with confidences.

Output: results/e2e_demo.json, plus pretty-printed Markdown.
"""
from __future__ import annotations

import argparse
import collections
import json
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from cascade_planner.conditions.predict_conditions import _topk_label
from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.expand.enz_template import (apply_template_to_product,
                                                  extract_templates,
                                                  main_product, morgan2,
                                                  predict_topk, train as train_mlp)
from cascade_planner.training.featurize_v2 import drfp_batch

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


def fit_clf(X, y):
    cls = sorted(set(y)); idx = {c: i for i, c in enumerate(cls)}
    yi = np.array([idx[v] for v in y])
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    m.fit(X, yi)
    return m, cls


def predict_clf_topk(m, cls, x, k=3):
    p = m.predict_proba(x.reshape(1, -1))[0]
    order = np.argsort(-p)[:k]
    return [(cls[i], float(p[i])) for i in order]


def fit_reg_by_ec1(values, ec1_arr):
    """Returns dict ec1 → mean."""
    out = {}
    for e in set(ec1_arr):
        m = ec1_arr == e
        if m.sum() > 0:
            out[e] = float(np.mean(values[m]))
    out["__global__"] = float(np.mean(values))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--topk-tpl", type=int, default=20)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    enz = [s for s in steps if s.ec_number]
    print(f"  steps={len(steps)}  enzymatic={len(enz)}")

    # ---------- 1) EnzExpand-A on full data ----------
    print("\n[train] EnzExpand-A (full enz set, no held-out) ...")
    cache_path = RESULTS / "enzexpand_atommap_cache.json"
    pairs = extract_templates(enz, mapped_cache=cache_path)
    cnt = collections.Counter(t for _, t in pairs)
    tpls_sorted = [t for t, c in cnt.most_common() if c >= args.min_freq]
    tpl_to_id = {t: i for i, t in enumerate(tpls_sorted)}
    print(f"  templates kept (>={args.min_freq}x): {len(tpl_to_id)}")
    samples = [(i, tpl_to_id[t]) for i, t in pairs if t in tpl_to_id]
    # template → dominant EC1
    tpl_ec1 = collections.defaultdict(collections.Counter)
    for i, tid in samples:
        ec1 = enz[i].ec_number.split(".")[0]
        tpl_ec1[tid][ec1] += 1
    tpl_ec1_top = {tid: c.most_common(1)[0][0] for tid, c in tpl_ec1.items()}

    X_tpl = np.stack([morgan2(main_product(enz[i].rxn_smiles)) for i, _ in samples])
    y_tpl = np.array([t for _, t in samples], dtype=np.int64)
    print(f"  trainable: {len(samples)}  unique tpl_ids: {len(set(y_tpl))}")
    model_tpl = train_mlp(X_tpl, y_tpl, n_tpl=len(tpl_to_id),
                          epochs=40, hidden=1024, dropout=0.4, batch=128,
                          seed=0, verbose=True)

    # ---------- 2) Condition predictors on full data ----------
    print("\n[train] condition predictors (DRFP + EC1 features) ...")
    rxns = [s.rxn_smiles for s in steps]
    X_drfp = drfp_batch(rxns)
    print(f"  DRFP features: {X_drfp.shape}")

    def _mask(values):
        return np.array([v is not None and v != "" for v in values])

    # T regression: mean by EC1 (best from CV)
    T_vals = np.array([s.temperature_c if s.temperature_c is not None else np.nan
                       for s in steps], dtype=float)
    ec1_all = np.array([s.ec_number.split(".")[0] if s.ec_number else "" for s in steps])
    mT = ~np.isnan(T_vals)
    T_by_ec1 = fit_reg_by_ec1(T_vals[mT], ec1_all[mT])
    print(f"  T_by_ec1: {T_by_ec1}")

    # pH regression: global mean (CV says rxn-fp doesn't help)
    pH_vals = np.array([s.ph if s.ph is not None else np.nan for s in steps], dtype=float)
    pH_global = float(np.nanmean(pH_vals))
    print(f"  pH global mean: {pH_global:.2f}")

    # EC1 classifier (6-class)
    m_ec = _mask([s.ec_number for s in steps])
    ec1_y = np.array([s.ec_number.split(".")[0] for s in steps if s.ec_number])
    ec1_clf, ec1_classes = fit_clf(X_drfp[m_ec], ec1_y)

    # transformation_superclass (top-12 + other)
    m_tr = _mask([s.transformation_superclass for s in steps])
    tr_raw = [s.transformation_superclass for s in steps if s.transformation_superclass]
    tr_y, _ = _topk_label(tr_raw, 12)
    tr_clf, tr_classes = fit_clf(X_drfp[m_tr], tr_y)

    # catalyst_class (top-8 + other)
    m_cat = _mask([s.catalyst_class for s in steps])
    cat_raw = [s.catalyst_class for s in steps if s.catalyst_class]
    cat_y, _ = _topk_label(cat_raw, 8)
    cat_clf, cat_classes = fit_clf(X_drfp[m_cat], cat_y)

    # solvent (top-12 + other)
    m_sv = _mask([s.solvent_smiles for s in steps])
    sv_raw = [s.solvent_smiles for s in steps if s.solvent_smiles]
    sv_y, _ = _topk_label(sv_raw, 12)
    sv_clf, sv_classes = fit_clf(X_drfp[m_sv], sv_y)

    print("  all condition heads trained.")

    # ---------- 3) Sample N query products ----------
    rng = np.random.default_rng(args.seed)
    # Choose products from steps with EC annotation, picking diverse EC1
    by_ec1 = collections.defaultdict(list)
    for s in enz:
        e = s.ec_number.split(".")[0]
        p = main_product(s.rxn_smiles)
        if p:
            by_ec1[e].append((s, p))
    queries = []
    for e in sorted(by_ec1.keys()):
        bucket = by_ec1[e]
        idx = rng.integers(0, len(bucket))
        s, p = bucket[idx]
        queries.append((s, p))
        if len(queries) >= args.n_queries:
            break
    print(f"\n[demo] {len(queries)} query products (one per EC1 class):")
    for s, p in queries:
        print(f"  EC{s.ec_number}  {s.transformation_superclass:25s}  prod={p}")

    # ---------- 4) Per-query: expand + predict conditions ----------
    out_records = []
    md_lines = ["# End-to-end retrosynthesis demo\n"]

    for q_idx, (s_true, prod) in enumerate(queries):
        md_lines.append(f"\n## Query {q_idx+1}: target = `{prod}`")
        md_lines.append(f"- ground-truth EC = `{s_true.ec_number}`, "
                        f"transformation = `{s_true.transformation_superclass}`")
        md_lines.append(f"- ground-truth precursors = `{s_true.rxn_smiles.split('>>')[0]}`")
        md_lines.append("")

        # expand
        x_prod = morgan2(prod).reshape(1, -1)
        topk_local, scores = predict_topk(model_tpl, x_prod, k=args.topk_tpl)
        topk_local = topk_local[0]; scores = scores[0]
        # softmax for nicer "confidence"
        e = np.exp(scores - scores.max()); probs = e / e.sum()

        proposals = []
        true_set = frozenset()
        if ">>" in s_true.rxn_smiles:
            from cascade_planner.expand.enz_template import canon_set
            true_set = canon_set(s_true.rxn_smiles.split(">>", 1)[0])

        for rank, (lid, sc, pr) in enumerate(zip(topk_local, scores, probs)):
            tpl = tpls_sorted[int(lid)]
            cands = apply_template_to_product(tpl, prod, max_outcomes=2)
            for cs in cands:
                rxn_cand = ".".join(sorted(cs)) + ">>" + prod
                # condition predictions
                xd = drfp_batch([rxn_cand])
                ec1_top = predict_clf_topk(ec1_clf, ec1_classes, xd[0], k=2)
                tr_top = predict_clf_topk(tr_clf, tr_classes, xd[0], k=2)
                cat_top = predict_clf_topk(cat_clf, cat_classes, xd[0], k=2)
                sv_top = predict_clf_topk(sv_clf, sv_classes, xd[0], k=2)
                T_pred = T_by_ec1.get(ec1_top[0][0], T_by_ec1["__global__"])
                proposals.append(dict(
                    rank=rank + 1, tpl_prob=float(pr), tpl_ec1_dom=tpl_ec1_top.get(int(lid), "?"),
                    precursors=sorted(cs), candidate_rxn=rxn_cand,
                    matches_truth=(cs == true_set),
                    pred_ec1=ec1_top, pred_transformation=tr_top,
                    pred_catalyst_class=cat_top, pred_solvent=sv_top,
                    pred_T_C=round(T_pred, 1), pred_pH=round(pH_global, 2),
                ))
            if len(proposals) >= 5:  # cap per query for brevity
                break
        if not proposals:
            md_lines.append("_no template fired on this product_")
            out_records.append(dict(query_idx=q_idx, target=prod, proposals=[]))
            continue

        for p in proposals:
            md_lines.append(
                f"\n### proposal #{p['rank']}  "
                f"(tpl-EC1={p['tpl_ec1_dom']}, prob={p['tpl_prob']:.3f}"
                f"{', **MATCHES TRUTH**' if p['matches_truth'] else ''})"
            )
            md_lines.append(f"- precursors: `{' . '.join(p['precursors'])}`")
            md_lines.append(f"- predicted EC1: " + ", ".join(f"EC{c} ({pr:.2f})" for c, pr in p['pred_ec1']))
            md_lines.append(f"- predicted transformation: " + ", ".join(f"{c} ({pr:.2f})" for c, pr in p['pred_transformation']))
            md_lines.append(f"- predicted catalyst_class: " + ", ".join(f"{c} ({pr:.2f})" for c, pr in p['pred_catalyst_class']))
            md_lines.append(f"- predicted solvent: " + ", ".join(f"`{c}` ({pr:.2f})" for c, pr in p['pred_solvent']))
            md_lines.append(f"- predicted T = {p['pred_T_C']} °C, pH ≈ {p['pred_pH']}")

        out_records.append(dict(query_idx=q_idx, target=prod,
                                truth_ec=s_true.ec_number,
                                truth_transformation=s_true.transformation_superclass,
                                truth_precursors=s_true.rxn_smiles.split(">>", 1)[0],
                                proposals=proposals))

    out_path_json = RESULTS / "e2e_demo.json"
    out_path_md = RESULTS / "e2e_demo.md"
    out_path_json.write_text(json.dumps(out_records, indent=2, ensure_ascii=False), encoding="utf-8")
    out_path_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[save] {out_path_json}\n[save] {out_path_md}")
    print("\n----- markdown preview -----")
    print("\n".join(md_lines[:80]))


if __name__ == "__main__":
    main()
