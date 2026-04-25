"""Per-step reaction condition prediction.

Given rxn_smiles (and optional EC label), predict:
  T (regression), pH (regression),
  solvent_topK (multiclass), catalyst_class (multiclass),
  transformation_superclass (multiclass), EC1 (6-class).

Featurization: DRFP-2048 (no condition leakage; same as classifier baselines).
Eval: 5-fold GroupKFold by DOI x 3 seeds.
"""
from __future__ import annotations

import collections
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GroupKFold

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.training.featurize_v2 import drfp_batch

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
SEEDS = [0, 17, 42]
TOPK_SOLV = 12  # most frequent solvents kept (others = "other")


def _topk_label(values, k):
    cnt = collections.Counter(v for v in values if v)
    keep = {k_ for k_, _ in cnt.most_common(k)}
    return [v if v in keep else "other" for v in values], sorted(keep) + ["other"]


def _cv_clf(X, y, groups, name, seed_grid=SEEDS, n_folds=5):
    yi_map = {c: i for i, c in enumerate(sorted(set(y)))}
    yi = np.array([yi_map[v] for v in y])
    n_cls = len(yi_map)
    rows = []
    for seed in seed_grid:
        rng = np.random.default_rng(seed)
        ug = np.array(sorted(set(groups))); rng.shuffle(ug)
        order = {g: i for i, g in enumerate(ug)}
        gid = np.array([order[g] for g in groups])
        nf = min(n_folds, len(set(gid)))
        per = collections.defaultdict(list)
        for tr, te in GroupKFold(n_splits=nf).split(X, yi, groups=gid):
            if len(set(yi[tr])) < 2:
                continue
            for mname, m in {
                "majority": DummyClassifier(strategy="most_frequent"),
                "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=seed),
            }.items():
                m.fit(X[tr], yi[tr]); pred = m.predict(X[te])
                macro = f1_score(yi[te], pred, average="macro", zero_division=0)
                acc = accuracy_score(yi[te], pred)
                top3 = np.nan
                if hasattr(m, "predict_proba") and n_cls >= 3:
                    proba = m.predict_proba(X[te])
                    top3i = np.argsort(-proba, axis=1)[:, :3]
                    top3 = float(np.mean([yi[te][k] in top3i[k] for k in range(len(yi[te]))]))
                per[mname].append((macro, acc, top3))
        for mname, vs in per.items():
            arr = np.array(vs)
            rows.append(dict(task=name, seed=seed, model=mname, n_classes=n_cls,
                             n_samples=len(y),
                             macro_f1=float(arr[:, 0].mean()),
                             accuracy=float(arr[:, 1].mean()),
                             top3_acc=float(arr[:, 2].mean())))
    return rows


def _cv_reg(X, y, groups, name, ec1=None, seed_grid=SEEDS, n_folds=5):
    """Regression. If ec1 (array of '1'..'6'/None) provided, also try stratified
    mean by EC1 and ridge on DRFP ⊕ EC1-onehot."""
    rows = []
    # build EC1 one-hot once if provided
    ec_onehot = None
    if ec1 is not None:
        cls = sorted({e for e in ec1 if e})
        idx = {c: i for i, c in enumerate(cls)}
        eh = np.zeros((len(ec1), len(cls) + 1), dtype=np.float32)
        for i, e in enumerate(ec1):
            eh[i, idx[e] if e in idx else len(cls)] = 1.0
        ec_onehot = eh
        X_aug = np.concatenate([X, ec_onehot * 4.0], axis=1)  # boost weight a bit

    for seed in seed_grid:
        rng = np.random.default_rng(seed)
        ug = np.array(sorted(set(groups))); rng.shuffle(ug)
        order = {g: i for i, g in enumerate(ug)}
        gid = np.array([order[g] for g in groups])
        nf = min(n_folds, len(set(gid)))
        per = collections.defaultdict(list)
        for tr, te in GroupKFold(n_splits=nf).split(X, y, groups=gid):
            models = {
                "mean": DummyRegressor(strategy="mean"),
                "ridge_drfp": Ridge(alpha=1.0, random_state=seed),
            }
            preds = {}
            for mname, m in models.items():
                m.fit(X[tr], y[tr]); preds[mname] = m.predict(X[te])
            # EC1-stratified mean
            if ec_onehot is not None:
                tr_ec = np.array(ec1)[tr]; te_ec = np.array(ec1)[te]
                global_mean = float(np.mean(y[tr]))
                ec_mean = {e: float(np.mean(y[tr][tr_ec == e])) if (tr_ec == e).sum() > 0 else global_mean
                           for e in set(tr_ec)}
                preds["mean_by_ec1"] = np.array([ec_mean.get(e, global_mean) for e in te_ec])
                m = Ridge(alpha=1.0, random_state=seed)
                m.fit(X_aug[tr], y[tr]); preds["ridge_drfp+ec1"] = m.predict(X_aug[te])
            for mname, p in preds.items():
                mae = mean_absolute_error(y[te], p)
                r2 = r2_score(y[te], p) if len(set(y[te])) > 1 else float("nan")
                per[mname].append((mae, r2))
        for mname, vs in per.items():
            arr = np.array(vs)
            rows.append(dict(task=name, seed=seed, model=mname, n_classes=-1,
                             n_samples=len(y),
                             mae=float(arr[:, 0].mean()),
                             r2=float(arr[:, 1].mean())))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--tag", default="", help="Suffix for output csv")
    args = ap.parse_args()
    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    print(f"  steps: {len(steps)}")

    # one shared DRFP featurization (saves time across tasks)
    X_all = drfp_batch([s.rxn_smiles for s in steps])
    groups_all = np.array([s.doi for s in steps])

    out = []

    # T regression
    mask = np.array([s.temperature_c is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; y = np.array([s.temperature_c for s in steps if s.temperature_c is not None], dtype=float)
        g = groups_all[mask]
        ec1 = [s.ec_number.split('.')[0] if s.ec_number else '' for s in steps if s.temperature_c is not None]
        print(f"\n[T regression] n={len(y)}")
        out.extend(_cv_reg(X, y, g, "temperature_c", ec1=ec1))

    # pH regression
    mask = np.array([s.ph is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; y = np.array([s.ph for s in steps if s.ph is not None], dtype=float)
        g = groups_all[mask]
        ec1 = [s.ec_number.split('.')[0] if s.ec_number else '' for s in steps if s.ph is not None]
        print(f"[pH regression] n={len(y)}")
        out.extend(_cv_reg(X, y, g, "ph", ec1=ec1))

    # transformation_superclass
    mask = np.array([s.transformation_superclass is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y_raw = [s.transformation_superclass for s in steps if s.transformation_superclass is not None]
        y, _ = _topk_label(y_raw, 12)
        print(f"[transformation_superclass] n={len(y)} top-12 + other")
        out.extend(_cv_clf(X, np.array(y), g, "transformation_superclass"))

    # EC1 (6-class)
    mask = np.array([s.ec_number is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y = np.array([s.ec_number.split(".")[0] for s in steps if s.ec_number])
        print(f"[EC1] n={len(y)} classes={len(set(y))}")
        out.extend(_cv_clf(X, y, g, "ec1"))

    # catalyst_class (chem vs enz vs none vs metal etc)
    mask = np.array([s.catalyst_class is not None for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y_raw = [s.catalyst_class for s in steps if s.catalyst_class]
        y, _ = _topk_label(y_raw, 8)
        print(f"[catalyst_class] n={len(y)} top-8 + other")
        out.extend(_cv_clf(X, np.array(y), g, "catalyst_class"))

    # solvent (multi-class on top-K)
    mask = np.array([bool(s.solvent_smiles) for s in steps])
    if mask.sum() > 50:
        X = X_all[mask]; g = groups_all[mask]
        y_raw = [s.solvent_smiles for s in steps if s.solvent_smiles]
        y, kept = _topk_label(y_raw, TOPK_SOLV)
        print(f"[solvent top-{TOPK_SOLV}] n={len(y)} kept={len(kept)} (incl 'other')")
        out.extend(_cv_clf(X, np.array(y), g, f"solvent_top{TOPK_SOLV}"))

    df = pd.DataFrame(out)
    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = RESULTS / f"conditions_metrics{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print("\n========= conditions per-step (mean over seeds) =========")
    keep_cols = ["mae", "r2", "macro_f1", "accuracy", "top3_acc"]
    summary = (df.groupby(["task", "model"])[keep_cols]
                 .mean(numeric_only=True).round(3))
    print(summary)
    print(f"\n[save] {out_csv}")


if __name__ == "__main__":
    main()
