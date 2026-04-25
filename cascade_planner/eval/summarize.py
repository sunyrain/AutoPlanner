"""Aggregate all baseline and external-SOTA results into one summary table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results"


def _load(name: str):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def main():
    pd.set_option("display.width", 180)
    pd.set_option("display.max_rows", 300)

    frames_mc, frames_ml, frames_reg = [], [], []
    for tag in ["v2", "full"]:
        fmc = _load(f"baselines_multiclass_{tag}.csv")
        if fmc is not None:
            fmc["source"] = tag
            frames_mc.append(fmc)
        fml = _load(f"baselines_multilabel_{tag}.csv")
        if fml is not None:
            fml["source"] = tag
            frames_ml.append(fml)
    freg = _load("baselines_regression_v2.csv")
    if freg is not None:
        freg["source"] = "v2"
        frames_reg.append(freg)
    fmlp = _load("baselines_multilabel_mlp_asl.csv")
    if fmlp is not None:
        fmlp["source"] = "mlp_asl"
        frames_ml.append(fmlp)

    df_mc = pd.concat(frames_mc, ignore_index=True) if frames_mc else pd.DataFrame()
    df_ml = pd.concat(frames_ml, ignore_index=True) if frames_ml else pd.DataFrame()
    df_reg = pd.concat(frames_reg, ignore_index=True) if frames_reg else pd.DataFrame()

    print("=" * 80)
    print("CASCADE PLANNER — ALL BASELINES (425 cascades, 800 steps, 344 DOIs)")
    print("=" * 80)

    if not df_mc.empty:
        print("\n### Single-label / multiclass  (macro-F1, mean±std over 3 seeds)\n")
        s = (df_mc.groupby(["source", "task", "model"])["macro_f1_mean"]
             .agg(["mean", "std"]).reset_index().round(3))
        s.columns = ["source", "task", "model", "macroF1", "std"]
        # show only best-per-task
        best = s.sort_values("macroF1", ascending=False).groupby(["source", "task"]).head(3)
        print(best.to_string(index=False))

    if not df_ml.empty:
        print("\n### Multi-label  (micro-F1 / macro-F1, mean±std over 3 seeds)\n")
        s = (df_ml.groupby(["source", "task", "model"])[["micro_f1_mean", "macro_f1_mean"]]
             .agg(["mean"]).reset_index())
        s.columns = ["source", "task", "model", "microF1", "macroF1"]
        s[["microF1", "macroF1"]] = s[["microF1", "macroF1"]].round(3)
        best = s.sort_values("microF1", ascending=False).groupby(["source", "task"]).head(3)
        print(best.to_string(index=False))

    if not df_reg.empty:
        print("\n### Regression (per-step conditions)\n")
        s = (df_reg.groupby(["task", "model"])[["rmse_mean", "mae_mean"]]
             .agg(["mean"]).reset_index())
        s.columns = ["task", "model", "RMSE", "MAE"]
        s[["RMSE", "MAE"]] = s[["RMSE", "MAE"]].round(2)
        best = s.sort_values("RMSE").groupby("task").head(3)
        print(best.to_string(index=False))

    # AiZynthFinder external baseline
    aiz = _load("aizynthfinder_step_eval.csv")
    aiz_sum = _load("aizynthfinder_summary.csv")
    if aiz is not None:
        print("\n" + "=" * 80)
        print("### External SOTA — AiZynthFinder (USPTO trained) on cascade steps")
        print("=" * 80)
        print(f"Evaluated steps      : {len(aiz)}")
        for k in [1, 5, 10, 50]:
            c = f"top{k}"
            print(f"  top-{k:<2d} accuracy      : {aiz[c].mean()*100:5.1f}%  ({int(aiz[c].sum())}/{len(aiz)})")
        print("\nBy enzymatic vs chemical step:")
        enz = aiz[aiz["ec_number"].notna() & (aiz["ec_number"].astype(str) != "")]
        chem = aiz[~aiz.index.isin(enz.index)]
        for name, g in [("enzymatic", enz), ("chemical", chem)]:
            if len(g) == 0: continue
            print(f"  {name:9s} N={len(g):3d}  top1={g['top1'].mean()*100:5.1f}%  "
                  f"top10={g['top10'].mean()*100:5.1f}%  top50={g['top50'].mean()*100:5.1f}%")
        print("\nBy transformation_superclass:")
        gg = (aiz.groupby("transformation")
                 .agg(N=("top1","size"), top1=("top1","mean"),
                      top10=("top10","mean"), top50=("top50","mean"))
                 .sort_values("N", ascending=False).head(12))
        print(gg.round(3).to_string())

    # CompatNet (multi-task neural model on step-pairs)
    cn_files = sorted(R.glob("baselines_compatnet_*.csv"))
    if cn_files:
        print("\n" + "=" * 80)
        print("### CompatNet (multi-task MLP on step-pairs) — main task: pair_pairwise_6c")
        print("=" * 80)
        cn = pd.concat([pd.read_csv(p).assign(tag=p.stem.replace("baselines_compatnet_", ""))
                        for p in cn_files], ignore_index=True)
        s = (cn.groupby(["tag", "model"])["macro_f1_mean"].agg(["mean", "std"])
             .reset_index().round(3))
        s.columns = ["variant", "model", "macroF1", "std"]
        print(s.to_string(index=False))
        print("  (Reference: XGB on full 416 pairs = 0.337; XGB on filtered 99 = 0.368)")

    print("\nFiles in results/:")
    for p in sorted(R.glob("*.csv")):
        print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
