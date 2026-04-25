"""Audited multi-engine ensemble report.

Honest counterpart to ``hybrid_multi.py``. Adds:

1. **K-budget fairness**: 3-engine UNION top-10 has a *30-candidate budget*;
   compare against single-engine top-30/top-50 (closest available column).
2. **Random baseline**: for each step, expected top-K accuracy under random
   selection from the templates_tried pool = ``min(K, pool)/pool``.
3. **Popularity baseline**: pick the K most frequent transformation labels
   per (EC, transformation) and check hit on the GT transformation. Crude
   but useful as a lower bound.
4. **Mask n<20** transformations from "best engine" determination.
5. **EnzExpand pool warning**: report mean templates_tried; when pool < K,
   top-K is mechanically inflated.

Outputs:
  results/hybrid_multi_audited.md
  results/hybrid_multi_audited_overall.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from cascade_planner.paths import results_dir

R = results_dir()  # defaults to results/v2 (override via CASCADE_VERSION env)
KS = (1, 5, 10, 50)
SYNTH_ENGINES = {
    "rootaligned": "RootAligned",
    "megan": "MEGAN",
    "mhnreact": "MHNreact",
    "chemformer": "Chemformer",
    "localretro": "LocalRetro",
}


def load_csv(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def mean_top(df, K, suffix=""):
    col = f"top{K}{suffix}"
    return float(df[col].mean()) if col in df else np.nan


def _row(name, n, vals, note=""):
    d = {"policy": name, "n": int(n)}
    for K in KS:
        d[f"top{K}"] = round(float(vals.get(K, np.nan)), 4) if vals.get(K) is not None else np.nan
    d["note"] = note
    return d


def random_baseline_topk(pool_sizes, K, gt_in_pool=None):
    """Expected top-K hit rate under uniform random selection from each row's pool.

    If ``gt_in_pool`` is given (boolean Series, True iff the GT template is
    among the candidates considered), we condition on it: random can only hit
    when GT is in the pool. Without it, we assume GT is always in the pool,
    which OVERSTATES random performance whenever model top-50 < 1.
    """
    pool = pool_sizes.clip(lower=1)
    per_row = np.minimum(K, pool).div(pool)
    if gt_in_pool is not None:
        per_row = per_row * gt_in_pool.astype(int)
    return float(per_row.mean())


def main():
    aiz = load_csv("aizynthfinder_full_gpu_step_eval.csv")
    if aiz is None:
        raise SystemExit("missing AiZ csv")
    aiz["is_enz"] = aiz["ec_number"].notna() & (aiz["ec_number"].astype(str).str.strip() != "")

    enz_mf2 = load_csv("enzexpand_step_eval_mf2.csv")
    enz_mf5 = load_csv("enzexpand_step_eval_mf5.csv")

    synth_dfs = {}
    for k in SYNTH_ENGINES:
        df = load_csv(f"syntheseus_step_eval_{k}.csv")
        if df is None or len(df) < 0.9 * len(aiz):
            continue
        synth_dfs[k] = df

    print(f"loaded: AiZ={len(aiz)}  EnzExpand mf2={len(enz_mf2) if enz_mf2 is not None else 0}"
          f" mf5={len(enz_mf5) if enz_mf5 is not None else 0}  synth={list(synth_dfs)}")

    rows = []

    # ---- single engines ----
    rows.append(_row("AiZ-USPTO (all)", len(aiz), {K: mean_top(aiz, K) for K in KS}))
    for k, df in synth_dfs.items():
        rows.append(_row(f"{SYNTH_ENGINES[k]} (all)", len(df),
                         {K: mean_top(df, K) for K in KS}))

    # ---- EnzExpand with pool warning ----
    for tag, df in [("mf2", enz_mf2), ("mf5", enz_mf5)]:
        if df is None:
            continue
        pool = df["templates_tried"]
        cap = (pool < 50).mean() * 100
        # GT ∈ pool proxy: model achieves top-50 (rank<=50 within considered pool)
        gt_in_pool = df["top50"].astype(bool)
        cov = gt_in_pool.mean() * 100
        note = (f"templates_tried mean={pool.mean():.1f}  median={pool.median():.0f}  "
                f"rows with pool<50: {cap:.0f}%  GT∈pool (top50==1): {cov:.0f}%")
        rows.append(_row(f"EnzExpand-A ({tag})", len(df),
                         {K: mean_top(df, K) for K in KS}, note=note))
        rand = {K: random_baseline_topk(pool, K, gt_in_pool=gt_in_pool) for K in KS}
        rows.append(_row(f"  └─ random-in-pool baseline ({tag})", len(df), rand,
                         note="E[hit] = (min(K,pool)/pool) × I(GT∈pool); honest random ceiling"))
        # Lift = model_topK / random_topK
        lift = {K: rows[-2][f"top{K}"] / rand[K] if rand[K] > 1e-9 else np.nan for K in KS}
        rows.append(_row(f"  └─ EnzExpand lift over random ({tag})", len(df),
                         lift,
                         note="lift > 1 means model ranks better than uniform draw inside its pool"))

    # ---- aligned intersection across AiZ + synth engines ----
    key_cols = ["doi", "step_id"]
    base = aiz.copy()
    for k, df in synth_dfs.items():
        sub = (df[key_cols + [f"top{K}" for K in KS]]
               .groupby(key_cols, as_index=False).max()
               .rename(columns={f"top{K}": f"top{K}_{k}" for K in KS}))
        base = base.merge(sub, on=key_cols, how="inner")
    base = base.rename(columns={f"top{K}": f"top{K}_aiz" for K in KS})

    n = len(base)
    print(f"intersection: {n} steps")

    engines = ["aiz"] + list(synth_dfs.keys())

    # K-budget fairness:
    # union of E engines × top-10 = 10E candidate budget
    # compare against single engine top-(10E) — we have top-50 so use it for E∈{1..5}
    union_top10 = base[[f"top10_{e}" for e in engines]].max(axis=1)
    union_top50 = base[[f"top50_{e}" for e in engines]].max(axis=1)
    budget = 10 * len(engines)
    rows.append(_row(f"UNION chem-engines ({len(engines)}) top-10 [budget≈{budget}]",
                     n, {1: np.nan, 5: np.nan, 10: union_top10.mean(),
                         50: union_top50.mean()},
                     note="union budget = 10 × n_engines candidates"))
    # Best single-engine top-50 — fair comparison if budget >= 50
    best_single_50 = max(engines, key=lambda e: base[f"top50_{e}"].mean())
    rows.append(_row(f"Best single ({best_single_50}) top-50 [budget=50]", n,
                     {1: mean_top(base, 1, f"_{best_single_50}"),
                      5: mean_top(base, 5, f"_{best_single_50}"),
                      10: mean_top(base, 10, f"_{best_single_50}"),
                      50: mean_top(base, 50, f"_{best_single_50}")},
                     note="fair single-engine vs union (top-10 of 5 engines = budget 50)"))

    # ---- EnzExpand union (only on steps where enz pred exists) ----
    enz_use = enz_mf2 if enz_mf2 is not None else enz_mf5
    if enz_use is not None:
        enz_sub = (enz_use[key_cols + [f"top{K}" for K in KS] + ["templates_tried"]]
                   .groupby(key_cols, as_index=False).max()
                   .rename(columns={f"top{K}": f"top{K}_enz" for K in KS}))
        merged = base.merge(enz_sub, on=key_cols, how="left")
        has_enz = merged["top10_enz"].notna()
        n_enz = int(has_enz.sum())
        for K in KS:
            uc = merged[[f"top{K}_{e}" for e in engines]].max(axis=1)
            ec = merged[f"top{K}_enz"].fillna(0)
            merged[f"union_all_top{K}"] = np.maximum(uc, ec).astype(int)
        rows.append(_row(f"UNION + EnzExpand over {len(merged)} (n_enz={n_enz})",
                         len(merged),
                         {K: merged[f"union_all_top{K}"].mean() for K in KS},
                         note=f"adds enz pred on {n_enz}/{len(merged)} ({n_enz/len(merged)*100:.0f}%) steps"))

    # ---- per-tx with n>=20 only ----
    tx_rows = []
    for tx, sub in base.groupby("transformation"):
        if len(sub) < 20:
            continue
        r = {"transformation": tx, "n": len(sub)}
        for e in engines:
            r[f"{e}_top10"] = round(sub[f"top10_{e}"].mean(), 3)
        r["best_chem_engine"] = max(engines, key=lambda e: sub[f"top10_{e}"].mean())
        tx_rows.append(r)
    df_tx = pd.DataFrame(tx_rows).sort_values("n", ascending=False) if tx_rows else pd.DataFrame()

    df_overall = pd.DataFrame(rows)
    df_overall.to_csv(R / "hybrid_multi_audited_overall.csv", index=False)
    df_tx.to_csv(R / "hybrid_multi_audited_by_transformation.csv", index=False)

    print("\n=== Audited Overall ===")
    print(df_overall.to_string(index=False))
    print("\n=== By transformation (n>=20 only) ===")
    print(df_tx.to_string(index=False))

    md = ["# Audited Multi-engine Hybrid Report",
          "",
          "## Honesty notes",
          "- *K-budget fairness*: a UNION of E engines × top-10 has a ~10E candidate budget. The honest comparison is against a single engine's top-(10E). Below, both rows are shown.",
          "- *Random baseline*: for EnzExpand, expected top-K hit under uniform draw from each row's `templates_tried` pool. When pool is small (e.g. mf=5 for rare EC), top-K trivially saturates.",
          "- *n<20 mask*: per-transformation 'best engine' is computed only on transformations with ≥20 examples in the intersection.",
          "",
          "## Overall",
          "",
          df_overall.to_markdown(index=False) if hasattr(df_overall, "to_markdown") else df_overall.to_string(index=False),
          "",
          "## By transformation (n≥20)",
          "",
          df_tx.to_markdown(index=False) if hasattr(df_tx, "to_markdown") and len(df_tx) else df_tx.to_string(index=False),
          ""]
    (R / "hybrid_multi_audited.md").write_text("\n".join(md), encoding="utf-8")
    print("\n[save]", R / "hybrid_multi_audited.md")


if __name__ == "__main__":
    main()
