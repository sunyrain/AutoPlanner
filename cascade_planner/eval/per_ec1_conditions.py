"""Per-EC1 breakdown of condition-prediction heads.

Reuses logic from cascade_planner.conditions.predict_conditions but
records *per-test-sample* predictions, then aggregates by EC1 class.

Output: results/conditions_per_ec1.csv with rows like
   task,model,ec1,n,mae,r2,acc,top3
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.training.featurize_v2 import drfp_batch

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
SEEDS = (0, 17, 42)
TOPK_SOLV = 12


def _topk_label(values, k):
    cnt = collections.Counter(v for v in values if v)
    keep = {k_ for k_, _ in cnt.most_common(k)}
    return [v if v in keep else "other" for v in values], sorted(keep) + ["other"]


def _per_ec1_clf(X, y, ec1, groups, name, n_folds=5):
    yi_map = {c: i for i, c in enumerate(sorted(set(y)))}
    yi = np.array([yi_map[v] for v in y])
    n_cls = len(yi_map)
    rows = []
    rng = np.random.default_rng(0)
    ug = np.array(sorted(set(groups))); rng.shuffle(ug)
    order = {g: i for i, g in enumerate(ug)}
    gid = np.array([order[g] for g in groups])
    nf = min(n_folds, len(set(gid)))
    # collect per-sample predictions across folds
    pred_logreg = np.full(len(yi), -1, dtype=int)
    pred_top3 = [None] * len(yi)
    for tr, te in GroupKFold(n_splits=nf).split(X, yi, groups=gid):
        if len(set(yi[tr])) < 2:
            continue
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
        m.fit(X[tr], yi[tr])
        pred_logreg[te] = m.predict(X[te])
        if hasattr(m, "predict_proba") and n_cls >= 3:
            proba = m.predict_proba(X[te])
            top3 = np.argsort(-proba, axis=1)[:, :3]
            for j, k in enumerate(te):
                pred_top3[k] = set(top3[j].tolist())
    # aggregate by EC1
    ec_arr = np.array([e or "?" for e in ec1])
    for ec in sorted(set(ec_arr)):
        mask = (ec_arr == ec) & (pred_logreg >= 0)
        n = int(mask.sum())
        if n < 5:
            continue
        acc = float((pred_logreg[mask] == yi[mask]).mean())
        if pred_top3[0] is not None:
            top3 = float(np.mean([(yi[k] in pred_top3[k])
                                  for k in np.where(mask)[0] if pred_top3[k] is not None]))
        else:
            top3 = np.nan
        rows.append(dict(task=name, model="logreg", ec1=ec, n=n, acc=acc, top3=top3))
    return rows


def _per_ec1_reg(X, y, ec1, groups, name, n_folds=5):
    rows = []
    rng = np.random.default_rng(0)
    ug = np.array(sorted(set(groups))); rng.shuffle(ug)
    order = {g: i for i, g in enumerate(ug)}
    gid = np.array([order[g] for g in groups])
    nf = min(n_folds, len(set(gid)))
    pred = np.full(len(y), np.nan)
    pred_ec_mean = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=nf).split(X, y, groups=gid):
        m = Ridge(alpha=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        # EC1 stratified mean
        tr_ec = np.array(ec1)[tr]; te_ec = np.array(ec1)[te]
        global_mean = float(np.mean(y[tr]))
        ec_mean = {e: float(np.mean(y[tr][tr_ec == e])) if (tr_ec == e).sum() > 0 else global_mean
                   for e in set(tr_ec)}
        pred_ec_mean[te] = np.array([ec_mean.get(e, global_mean) for e in te_ec])
    ec_arr = np.array([e or "?" for e in ec1])
    for ec in sorted(set(ec_arr)):
        mask = (ec_arr == ec) & ~np.isnan(pred)
        n = int(mask.sum())
        if n < 5:
            continue
        for mname, p in [("ridge_drfp", pred), ("mean_by_ec1", pred_ec_mean)]:
            mae = float(mean_absolute_error(y[mask], p[mask]))
            r2 = float(r2_score(y[mask], p[mask])) if len(set(y[mask])) > 1 else float("nan")
            rows.append(dict(task=name, model=mname, ec1=ec, n=n, mae=mae, r2=r2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--out", default="results/conditions_per_ec1.csv")
    args = ap.parse_args()
    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    print(f"  steps: {len(steps)}")
    X_all = drfp_batch([s.rxn_smiles for s in steps])
    groups_all = np.array([s.doi for s in steps])
    out = []

    # T regression
    mask = np.array([s.temperature_c is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y = np.array([s.temperature_c for s in steps if s.temperature_c is not None], dtype=float)
        ec = [s.ec_number.split('.')[0] if s.ec_number else '' for s in steps if s.temperature_c is not None]
        print(f"\n[T reg per-EC1] n={len(y)}")
        out.extend(_per_ec1_reg(X, y, ec, g, "temperature_c"))

    # pH regression
    mask = np.array([s.ph is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y = np.array([s.ph for s in steps if s.ph is not None], dtype=float)
        ec = [s.ec_number.split('.')[0] if s.ec_number else '' for s in steps if s.ph is not None]
        print(f"[pH reg per-EC1] n={len(y)}")
        out.extend(_per_ec1_reg(X, y, ec, g, "ph"))

    # transformation classifier per EC1
    mask = np.array([s.transformation_superclass is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y_raw = [s.transformation_superclass for s in steps if s.transformation_superclass is not None]
        y, _ = _topk_label(y_raw, 12)
        ec = [s.ec_number.split('.')[0] if s.ec_number else '' for s in steps if s.transformation_superclass is not None]
        print(f"[transformation per-EC1] n={len(y)}")
        out.extend(_per_ec1_clf(X, np.array(y), ec, g, "transformation_superclass"))

    # solvent classifier per EC1
    mask = np.array([bool(s.solvent_smiles) for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y_raw = [s.solvent_smiles for s in steps if s.solvent_smiles]
        y, _ = _topk_label(y_raw, TOPK_SOLV)
        ec = [s.ec_number.split('.')[0] if s.ec_number else '' for s in steps if s.solvent_smiles]
        print(f"[solvent per-EC1] n={len(y)}")
        out.extend(_per_ec1_clf(X, np.array(y), ec, g, f"solvent_top{TOPK_SOLV}"))

    df = pd.DataFrame(out)
    out_path = ROOT / args.out
    df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")
    print("\n=========== summary (per task) ===========")
    for task, sub in df.groupby("task"):
        print(f"\n## {task}")
        piv = sub.pivot_table(index="ec1", columns="model",
                              values=["mae", "r2", "acc", "top3"],
                              aggfunc="mean").round(3)
        print(piv)


if __name__ == "__main__":
    main()
