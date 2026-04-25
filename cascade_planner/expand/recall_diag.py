"""Diagnose why EnzExpand candidate pools miss the GT reactants.

For each test step, classifies the GT-absent case into:
  - template_missing   : GT's training template is not in the MLP top-N
  - template_outside_dict : GT's template has freq < min_freq (dropped entirely)
  - template_fires_fail : template IS in top-N but rdchiral produced no outcome
                          matching GT
  - no_gt_template     : GT's template was never extractable in training
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.paths import results_dir, shared_dir
from cascade_planner.expand.enz_template import (
    apply_template_to_product,
    canon_set,
    canon_set_nostereo,
    extract_templates,
    main_product,
    morgan2,
    predict_topk,
    train as train_mlp,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--top-n-tpl", type=int, default=50)
    ap.add_argument("--max-outcomes", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--tag", default="recall")
    ap.add_argument("--nostereo", action="store_true",
                    help="Use stereo-stripped canonical set when matching GT (graph-level hit)")
    args = ap.parse_args()

    out_dir = results_dir() / "recall_diag" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    steps, _, _ = load_v2(args.data)
    enz_steps = [s for s in steps if s.ec_number]
    print(f"enzymatic steps: {len(enz_steps)}")

    cache_path = shared_dir() / "enzexpand_atommap_cache.json"
    pairs = extract_templates(enz_steps, mapped_cache=cache_path)

    tpl_count = collections.Counter(t for _, t in pairs)
    tpls_sorted = [t for t, c in tpl_count.most_common() if c >= args.min_freq]
    tpl_to_id = {t: i for i, t in enumerate(tpls_sorted)}
    print(f"unique tpl={len(tpl_count)}, kept={len(tpl_to_id)} (mf={args.min_freq})")

    # Build step -> (gt_template_extracted, gt_in_dict) map
    step_gt_tpl = {}
    for idx, tpl in pairs:
        step_gt_tpl[idx] = tpl

    trainable = [(idx, tpl_to_id[tpl]) for idx, tpl in pairs if tpl in tpl_to_id]
    print(f"trainable samples: {len(trainable)}")

    X = np.stack([morgan2(main_product(enz_steps[i].rxn_smiles)) for i, _ in trainable])
    y = np.array([tid for _, tid in trainable], dtype=np.int64)
    groups = np.array([enz_steps[i].doi for i, _ in trainable])

    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups=groups)):
        train_tids = sorted(set(y[tr]))
        local_to_global = dict(enumerate(train_tids))
        global_to_local = {g: i for i, g in local_to_global.items()}
        y_tr_local = np.array([global_to_local[v] for v in y[tr]])
        model = train_mlp(
            X[tr], y_tr_local, n_tpl=len(train_tids),
            epochs=60, hidden=1024, dropout=0.4, batch=128, seed=fold, verbose=False,
        )
        k_use = min(args.top_n_tpl, len(train_tids))
        topk_local_idx, _ = predict_topk(model, X[te], k=k_use)
        for j, te_i in enumerate(te):
            step_i = int(trainable[te_i][0])
            step = enz_steps[step_i]
            gt_tid_global = int(y[te_i])
            gt_tid_in_train = gt_tid_global in train_tids
            topn_global = {local_to_global[int(lid)] for lid in topk_local_idx[j]}
            gt_in_pool = gt_tid_global in topn_global
            # did rdchiral fire GT's template correctly?
            fired_ok = False
            if gt_in_pool:
                prod = main_product(step.rxn_smiles)
                true_set = canon_set(step.rxn_smiles.split(">>", 1)[0])
                if prod and true_set:
                    cands = apply_template_to_product(
                        tpls_sorted[gt_tid_global], prod, max_outcomes=args.max_outcomes
                    )
                    fired_ok = true_set in cands
                    if not fired_ok and args.nostereo:
                        true_ns = canon_set_nostereo(step.rxn_smiles.split(">>", 1)[0])
                        cand_ns = {canon_set_nostereo(".".join(sorted(c))) for c in cands if c}
                        fired_ok = true_ns in cand_ns
            if gt_in_pool and not fired_ok:
                bucket = "template_fires_fail"
            elif gt_in_pool and fired_ok:
                bucket = "pool_hit"
            elif gt_tid_in_train:
                bucket = "template_missing_from_topN"
            else:
                bucket = "template_outside_dict"
            rows.append(dict(
                fold=fold, doi=step.doi, step_id=step.step_id,
                ec1=(step.ec_number or "").split(".")[0],
                transformation=step.transformation_superclass or "other",
                gt_tpl_freq=tpl_count[step_gt_tpl[step_i]],
                bucket=bucket,
            ))
        print(f"[fold {fold}] processed {len(te)} steps")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "recall_per_step.csv", index=False)
    print(f"[save] {out_dir/'recall_per_step.csv'}")

    print("\n=== bucket breakdown ===")
    print(df["bucket"].value_counts(normalize=True).round(4).to_string())
    print("\n=== by EC1 ===")
    ec_tab = (df.groupby(["ec1", "bucket"]).size()
                .unstack(fill_value=0))
    ec_tab["total"] = ec_tab.sum(axis=1)
    ec_tab["pool_hit_pct"] = (ec_tab.get("pool_hit", 0) / ec_tab["total"] * 100).round(1)
    print(ec_tab.sort_values("total", ascending=False).to_string())
    ec_tab.to_csv(out_dir / "recall_by_ec1.csv")

    # top missing-template frequency buckets
    miss = df[df["bucket"] != "pool_hit"]
    print("\n=== freq distribution of GT templates on misses ===")
    print(miss["gt_tpl_freq"].describe().round(2).to_string())

    md = ["# EnzExpand candidate recall diagnosis",
          "",
          f"total steps evaluated: {len(df)}",
          "",
          "## Failure-mode breakdown",
          "",
          "| bucket | count | pct |",
          "|---|---:|---:|"]
    for bucket, cnt in df["bucket"].value_counts().items():
        md.append(f"| {bucket} | {cnt} | {cnt/len(df):.2%} |")
    md += ["",
           "**pool_hit** = GT present in pool AND rdchiral successfully regenerated GT reactants.",
           "**template_fires_fail** = GT template in top-N but rdchiral's reverse synth didn't reproduce the reactant set exactly (regio/stereo/atom-map drift).",
           "**template_missing_from_topN** = GT template is in training dict but MLP didn't rank it in top-N.",
           "**template_outside_dict** = GT template was rare (freq<min_freq) and dropped before training.",
           ""]
    (out_dir / "recall_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[save] {out_dir/'recall_report.md'}")


if __name__ == "__main__":
    main()
