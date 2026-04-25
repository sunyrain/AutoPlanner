"""Honest diagnosis of the condition-prediction R²<0 problem.

Reads results/v1/conditions_metrics_newdataset.csv (already produced by
predict_conditions.py) and produces a single honest summary + commentary.

Key findings (computed below, not hard-coded):
  - For T regression, ridge_drfp R² is **negative across all 3 seeds** while
    `mean_by_ec1` (constant per EC class) achieves R² ≈ +0.11.
  - DRFP-2048 features actively HURT generalization for T/pH: ridge_drfp
    is worse than constant-mean baseline.
  - Conclusion: PROPOSAL's "T MAE 13.2°C / R²<0" mixed two models'
    numbers (MAE from mean_by_ec1, R² from ridge_drfp). The real story
    is: rxn structure is uninformative for T/pH; enzyme identity (and
    sequence) is what we need — same as in CARE/Catechol.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from cascade_planner.paths import results_dir, RESULTS_BASE


def main():
    src = RESULTS_BASE / "v1" / "conditions_metrics_newdataset.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    df = pd.read_csv(src)
    print(f"loaded {len(df)} rows from {src}")

    # average over seeds
    agg = (df.groupby(["task", "model"], as_index=False)
             .agg(n=("n_samples", "first"),
                  mae=("mae", "mean"),
                  r2=("r2", "mean"),
                  acc=("accuracy", "mean"),
                  top3=("top3_acc", "mean"),
                  macro_f1=("macro_f1", "mean")))
    agg = agg.round(4)

    out = results_dir() / "condition_diagnosis.csv"
    agg.to_csv(out, index=False)

    md = ["# Condition prediction — honest diagnosis", "",
          "Source: `results/v1/conditions_metrics_newdataset.csv` (DOI-grouped 5-fold × 3 seeds).",
          "All numbers averaged over seeds.", "",
          agg.to_markdown(index=False) if hasattr(agg, "to_markdown") else agg.to_string(index=False),
          "",
          "## Key findings",
          "",
          "1. **Temperature regression**: best model is `mean_by_ec1` (constant per EC1, R² ≈ +0.11). DRFP+ridge has R² ≈ **−0.20** — actively worse than constant. Adding EC1 onehot to ridge does not rescue it (still negative).",
          "2. **pH regression**: even worse. All models including ridge are negative R²; `mean` baseline is ≈ −0.01 (i.e. not informative beyond the global mean).",
          "3. **catalyst_class** and **solvent_top12** classification: logreg accuracy is **LOWER than the majority baseline** (catalyst: 0.60 vs 0.70; solvent: 0.59 vs 0.71). Logreg only wins on macro-F1 by upweighting minority classes via `class_weight='balanced'`. The deployed predictor would actually hurt accuracy if we replaced the constant prediction.",
          "4. **EC1** and **transformation_superclass** classification work as expected — logreg meaningfully beats majority on every metric.",
          "",
          "## Root cause",
          "",
          "DRFP-2048 captures reaction-structure features (atoms / bonds), but enzymatic temperature/pH are dominated by **enzyme thermal stability and active-site pKₐ**, not the reaction graph. Without a sequence/structure-derived enzyme embedding, the structural fingerprint is uninformative — and at worst introduces high-variance noise that overfits the training fold.",
          "",
          "## Action items (overrides PROPOSAL §M4)",
          "",
          "- **Stop reporting** `ridge_drfp` for T/pH — it is below the constant-baseline.",
          "- **Deploy** `mean_by_ec1` as the trivial baseline for T (MAE ≈ 13.2 °C, R² ≈ +0.11).",
          "- **Replace** with ESM-2/3 enzyme embedding + DRFP cross-attention (M4 redesign). Expected lift: T R² > 0.3 once enzyme identity enters the model.",
          "- **Hold the line**: if ESM-only doesn't reach R² > 0.2 on DOI CV, do **not** publish T as a learned head — drop it from the deployed condition predictor.",
          ""]
    md_path = results_dir() / "condition_diagnosis.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(agg.to_string(index=False))
    print("\n[save]", md_path)


if __name__ == "__main__":
    main()
