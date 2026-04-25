"""v2 baselines: adds new tasks + step-pair features + condition features.

Tasks
-----
A1c transformation_superclass top-7+other (per-step, ~8-class)
A1d EC1 first-digit (per-step, enzyme-only, 6-class)
A1e is_enzymatic (per-step, binary)
F1  pairwise_mode 6-class on step-PAIRS using concat features
F1b pairwise_mode binary (simple/special) on pairs
B+  compatibility_label (cascade) USING condition features
C+  issue_top5 (cascade) USING condition features
D+  mit_top5  (cascade) USING condition features
E1  T regression (per-step) -> RMSE
"""
from __future__ import annotations

import collections
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.training.featurize_v2 import (
    cascade_mean_features,
    drfp_batch,
    pair_features,
    step_features,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_REPEATS_SEEDS = [0, 17, 42]
N_FOLDS = 5
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- model zoos ----------

def models_mc(seed: int, n_classes: int) -> dict:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "kNN-5": KNeighborsClassifier(n_neighbors=5, metric="jaccard"),
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1),
        "RF-300": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed),
        "XGB": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, n_jobs=-1, eval_metric="mlogloss",
            random_state=seed, verbosity=0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
        ),
    }


def models_ml(seed: int) -> dict:
    return {
        "majority": OneVsRestClassifier(DummyClassifier(strategy="most_frequent")),
        "kNN-5": KNeighborsClassifier(n_neighbors=5, metric="jaccard"),
        "logreg": OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "RF-300": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed),
        "XGB": OneVsRestClassifier(XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, n_jobs=-1, eval_metric="logloss",
            random_state=seed, verbosity=0,
        )),
    }


def models_reg(seed: int) -> dict:
    return {
        "mean": DummyRegressor(strategy="mean"),
        "median": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "RF-300": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed),
        "XGB": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, n_jobs=-1, random_state=seed, verbosity=0,
        ),
    }


# ---------- runners ----------

def run_mc(name, X, y, groups):
    rows = []
    classes = sorted(set(y))
    n_classes = len(classes)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_to_idx[v] for v in y])
    n_groups = len(set(groups))
    n_folds = min(N_FOLDS, n_groups)
    for seed in N_REPEATS_SEEDS:
        rng = np.random.default_rng(seed)
        ug = np.array(sorted(set(groups)))
        rng.shuffle(ug)
        order = {g: i for i, g in enumerate(ug)}
        gid = np.array([order[g] for g in groups])
        per: dict[str, list[float]] = collections.defaultdict(list)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y_int, groups=gid):
            if len(set(y_int[tr])) < 2:
                continue
            for mname, m in models_mc(seed, n_classes).items():
                try:
                    if mname == "XGB":
                        m.fit(X[tr], y_int[tr]); pred = m.predict(X[te])
                        f1 = f1_score(y_int[te], pred, average="macro", zero_division=0)
                    else:
                        m.fit(X[tr], y[tr]); pred = m.predict(X[te])
                        f1 = f1_score(y[te], pred, average="macro", zero_division=0, labels=classes)
                    per[mname].append(f1)
                except Exception as e:
                    print(f"  [{name}/{mname}] fold err: {type(e).__name__}: {str(e)[:80]}")
        for mname, sc in per.items():
            if sc:
                rows.append(dict(task=name, model=mname, seed=seed,
                                 macro_f1_mean=float(np.mean(sc)),
                                 macro_f1_std=float(np.std(sc)),
                                 n_folds=len(sc), n_samples=len(y),
                                 n_groups=n_groups, n_classes=n_classes))
    return rows


def run_ml(name, X, y, groups, label_names):
    rows = []
    n_groups = len(set(groups))
    n_folds = min(N_FOLDS, n_groups)
    for seed in N_REPEATS_SEEDS:
        rng = np.random.default_rng(seed)
        ug = np.array(sorted(set(groups)))
        rng.shuffle(ug)
        order = {g: i for i, g in enumerate(ug)}
        gid = np.array([order[g] for g in groups])
        per_mi: dict[str, list[float]] = collections.defaultdict(list)
        per_ma: dict[str, list[float]] = collections.defaultdict(list)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=gid):
            for mname, m in models_ml(seed).items():
                try:
                    m.fit(X[tr], y[tr]); pred = np.asarray(m.predict(X[te]))
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    per_mi[mname].append(f1_score(y[te], pred, average="micro", zero_division=0))
                    per_ma[mname].append(f1_score(y[te], pred, average="macro", zero_division=0))
                except Exception as e:
                    print(f"  [{name}/{mname}] fold err: {type(e).__name__}")
        for mname in models_ml(0):
            if per_mi[mname]:
                rows.append(dict(task=name, model=mname, seed=seed,
                                 micro_f1_mean=float(np.mean(per_mi[mname])),
                                 micro_f1_std=float(np.std(per_mi[mname])),
                                 macro_f1_mean=float(np.mean(per_ma[mname])),
                                 macro_f1_std=float(np.std(per_ma[mname])),
                                 n_folds=len(per_mi[mname]),
                                 n_samples=len(y), n_groups=n_groups,
                                 n_labels=int(y.shape[1]),
                                 label_names=",".join(label_names)))
    return rows


def run_reg(name, X, y, groups):
    rows = []
    n_groups = len(set(groups))
    n_folds = min(N_FOLDS, n_groups)
    for seed in N_REPEATS_SEEDS:
        rng = np.random.default_rng(seed)
        ug = np.array(sorted(set(groups)))
        rng.shuffle(ug)
        order = {g: i for i, g in enumerate(ug)}
        gid = np.array([order[g] for g in groups])
        per: dict[str, list[tuple[float,float]]] = collections.defaultdict(list)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=gid):
            for mname, m in models_reg(seed).items():
                try:
                    m.fit(X[tr], y[tr]); pred = m.predict(X[te])
                    rmse = float(np.sqrt(np.mean((pred - y[te]) ** 2)))
                    mae = float(mean_absolute_error(y[te], pred))
                    per[mname].append((rmse, mae))
                except Exception as e:
                    print(f"  [{name}/{mname}] fold err: {type(e).__name__}")
        for mname, sc in per.items():
            if sc:
                rmses = [a for a,_ in sc]; maes = [b for _,b in sc]
                rows.append(dict(task=name, model=mname, seed=seed,
                                 rmse_mean=float(np.mean(rmses)), rmse_std=float(np.std(rmses)),
                                 mae_mean=float(np.mean(maes)),  mae_std=float(np.std(maes)),
                                 n_folds=len(sc), n_samples=len(y), n_groups=n_groups))
    return rows


# ---------- task assembly ----------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    print(f"Loading: {args.data}")
    steps, pairs, cascades = load_v2(args.data)
    print(f"  steps={len(steps)} pairs={len(pairs)} cascades={len(cascades)} DOIs={len({c.doi for c in cascades})}")

    print("Featurizing...")
    # per-step features (rxn + cond + solvent)
    X_step = np.stack([step_features(s.rxn_smiles, s.temperature_c, s.ph, s.solvent_smiles) for s in steps])
    g_step = np.array([s.doi for s in steps])

    # per-pair features
    X_pair = np.stack([
        pair_features(p.rxn_smiles_a, p.rxn_smiles_b,
                      p.t_a, p.ph_a, p.solv_a, p.t_b, p.ph_b, p.solv_b)
        for p in pairs
    ])
    g_pair = np.array([p.doi for p in pairs])

    # per-cascade features (mean-DRFP + avg cond + first solvent)
    X_casc = np.stack([
        cascade_mean_features(c.rxn_smiles_list, c.avg_temperature_c, c.avg_ph, c.solvent_smiles_first)
        for c in cascades
    ])
    g_casc = np.array([c.doi for c in cascades])

    print(f"  X_step={X_step.shape}  X_pair={X_pair.shape}  X_casc={X_casc.shape}")

    rows_mc, rows_ml, rows_reg = [], [], []

    # ---- A1c transformation_superclass top-7+other ----
    cnt = collections.Counter(s.transformation_superclass for s in steps if s.transformation_superclass)
    top7 = [k for k, _ in cnt.most_common(7)]
    y_tr = []; mask = []
    for i, s in enumerate(steps):
        if not s.transformation_superclass:
            mask.append(False); continue
        mask.append(True)
        y_tr.append(s.transformation_superclass if s.transformation_superclass in top7 else "other")
    mask = np.array(mask)
    print(f"\n=== A1c transformation_superclass top-7+other ({mask.sum()} samples) ===")
    rows_mc += run_mc("transformation_top7", X_step[mask], np.array(y_tr), g_step[mask])

    # ---- A1d EC1 first digit (enzymatic only) ----
    ec_mask = np.array([bool(s.ec_number) for s in steps])
    y_ec = np.array([s.ec_number.split(".")[0] for s in steps if s.ec_number])
    print(f"=== A1d EC1 first-digit ({ec_mask.sum()} samples, {len(set(y_ec))} classes) ===")
    rows_mc += run_mc("ec1_class", X_step[ec_mask], y_ec, g_step[ec_mask])

    # ---- A1e is_enzymatic ----
    y_is = np.array(["enz" if s.ec_number else "chem" for s in steps])
    print(f"=== A1e is_enzymatic (binary, {len(y_is)} samples) ===")
    rows_mc += run_mc("is_enzymatic", X_step, y_is, g_step)

    # ---- F1 step-pair pairwise_mode 6c ----
    y_pm = np.array([p.pairwise_mode for p in pairs])
    print(f"=== F1 pair pairwise_mode 6c ({len(y_pm)} pairs) ===")
    rows_mc += run_mc("pair_pairwise_6c", X_pair, y_pm, g_pair)

    SIMPLE = {"not_applicable", "simultaneous"}
    y_pmb = np.array(["simple" if v in SIMPLE else "special" for v in y_pm])
    print(f"=== F1b pair pairwise_mode binary (simple vs special) ===")
    rows_mc += run_mc("pair_pairwise_bin", X_pair, y_pmb, g_pair)

    # F1c "is there an actual coupling between consecutive steps?"  not_applicable vs rest
    y_pmc = np.array(["coupled" if v != "not_applicable" else "isolated" for v in y_pm])
    print(f"=== F1c pair coupling presence (not_applicable vs rest) ===")
    rows_mc += run_mc("pair_coupling_presence", X_pair, y_pmc, g_pair)

    # ---- B+ compat_label with condition features ----
    keep = [i for i, c in enumerate(cascades) if c.compatibility_label]
    keep_arr = np.array(keep)
    y_cl = np.array([cascades[i].compatibility_label for i in keep])
    print(f"=== B+ compat_label (cond features) ({len(y_cl)}) ===")
    rows_mc += run_mc("compat_label_v2", X_casc[keep_arr], y_cl, g_casc[keep_arr])

    # ---- C+/D+ multi-label with cond ----
    flat = [t for c in cascades for t in c.issue_types]
    top5_i = [k for k, _ in collections.Counter(flat).most_common(5)]
    y_i = np.zeros((len(cascades), len(top5_i)), dtype=int)
    for i, c in enumerate(cascades):
        for j, t in enumerate(top5_i):
            if t in c.issue_types: y_i[i, j] = 1
    print(f"=== C+ issue_top5 (cond features) ===")
    rows_ml += run_ml("issue_top5_v2", X_casc, y_i, g_casc, top5_i)

    flat = [t for c in cascades for t in c.mitigation_strategies]
    top5_m = [k for k, _ in collections.Counter(flat).most_common(5)]
    y_m = np.zeros((len(cascades), len(top5_m)), dtype=int)
    for i, c in enumerate(cascades):
        for j, t in enumerate(top5_m):
            if t in c.mitigation_strategies: y_m[i, j] = 1
    print(f"=== D+ mit_top5 (cond features) ===")
    rows_ml += run_ml("mit_top5_v2", X_casc, y_m, g_casc, top5_m)

    # ---- E1 Temperature regression (drop nulls) ----
    Tmask = np.array([s.temperature_c is not None for s in steps])
    yT = np.array([s.temperature_c for s in steps if s.temperature_c is not None], dtype=np.float32)
    # use rxn-only features (not the T itself which is in conditions block!) — drop the cond cols
    # Actually our step_features INCLUDES cond block which has T... rebuild rxn-only features for fair eval
    X_step_rxnonly = drfp_batch([s.rxn_smiles for s in steps])
    print(f"=== E1 Temperature regression ({Tmask.sum()} samples) ===")
    rows_reg += run_reg("temperature_c", X_step_rxnonly[Tmask], yT, g_step[Tmask])

    pHmask = np.array([s.ph is not None for s in steps])
    ypH = np.array([s.ph for s in steps if s.ph is not None], dtype=np.float32)
    print(f"=== E2 pH regression ({pHmask.sum()} samples) ===")
    rows_reg += run_reg("ph", X_step_rxnonly[pHmask], ypH, g_step[pHmask])

    # ---- save & print ----
    df_mc = pd.DataFrame(rows_mc)
    df_ml = pd.DataFrame(rows_ml)
    df_reg = pd.DataFrame(rows_reg)

    tag = args.tag
    df_mc.to_csv(RESULTS_DIR / f"baselines_multiclass_{tag}.csv", index=False)
    df_ml.to_csv(RESULTS_DIR / f"baselines_multilabel_{tag}.csv", index=False)
    df_reg.to_csv(RESULTS_DIR / f"baselines_regression_{tag}.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY (mean across seeds; per-task model ranking)")
    print("=" * 78)

    if not df_mc.empty:
        s = (df_mc.groupby(["task", "model"])["macro_f1_mean"]
             .agg(["mean", "std"]).reset_index().round(3))
        for t in s["task"].unique():
            sub = s[s["task"] == t].sort_values("mean", ascending=False)
            print(f"\n[{t}]  macro-F1");  print(sub.to_string(index=False))

    if not df_ml.empty:
        s = (df_ml.groupby(["task", "model"])[["micro_f1_mean", "macro_f1_mean"]]
             .agg(["mean", "std"]).reset_index())
        s.columns = ["_".join([c for c in col if c]).strip() for col in s.columns.values]
        for c in s.columns:
            if s[c].dtype.kind == "f":
                s[c] = s[c].round(3)
        for t in s["task"].unique():
            sub = s[s["task"] == t].sort_values("micro_f1_mean_mean", ascending=False)
            print(f"\n[{t}]");  print(sub.to_string(index=False))

    if not df_reg.empty:
        s = (df_reg.groupby(["task", "model"])[["rmse_mean", "mae_mean"]]
             .agg(["mean"]).reset_index())
        s.columns = ["_".join([c for c in col if c]).strip() for col in s.columns.values]
        for c in s.columns:
            if s[c].dtype.kind == "f":
                s[c] = s[c].round(2)
        for t in s["task"].unique():
            sub = s[s["task"] == t].sort_values("rmse_mean_mean")
            print(f"\n[{t}]");  print(sub.to_string(index=False))

    print(f"\nSaved -> results/baselines_*_{tag}.csv")


if __name__ == "__main__":
    main()
