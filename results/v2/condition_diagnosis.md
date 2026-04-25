# Condition prediction — honest diagnosis

Source: `results/v1/conditions_metrics_newdataset.csv` (DOI-grouped 5-fold × 3 seeds).
All numbers averaged over seeds.

| task                      | model          |    n |      mae |       r2 |      acc |     top3 |   macro_f1 |
|:--------------------------|:---------------|-----:|---------:|---------:|---------:|---------:|-----------:|
| catalyst_class            | logreg         | 2337 | nan      | nan      |   0.6021 |   0.8605 |     0.2306 |
| catalyst_class            | majority       | 2337 | nan      | nan      |   0.6979 |   0.7356 |     0.0931 |
| ec1                       | logreg         | 1739 | nan      | nan      |   0.7244 |   0.9502 |     0.57   |
| ec1                       | majority       | 1739 | nan      | nan      |   0.4646 |   0.8769 |     0.1071 |
| ph                        | mean           | 1364 |   0.7361 |  -0.0136 | nan      | nan      |   nan      |
| ph                        | mean_by_ec1    | 1364 |   0.7393 |  -0.0111 | nan      | nan      |   nan      |
| ph                        | ridge_drfp     | 1364 |   0.9253 |  -0.4595 | nan      | nan      |   nan      |
| ph                        | ridge_drfp+ec1 | 1364 |   0.9447 |  -0.5256 | nan      | nan      |   nan      |
| solvent_top12             | logreg         | 2227 | nan      | nan      |   0.5873 |   0.7893 |     0.1226 |
| solvent_top12             | majority       | 2227 | nan      | nan      |   0.7063 |   0.7185 |     0.0707 |
| temperature_c             | mean           | 2039 |  14.9616 |  -0.0064 | nan      | nan      |   nan      |
| temperature_c             | mean_by_ec1    | 2039 |  13.191  |   0.1067 | nan      | nan      |   nan      |
| temperature_c             | ridge_drfp     | 2039 |  17.4924 |  -0.2456 | nan      | nan      |   nan      |
| temperature_c             | ridge_drfp+ec1 | 2039 |  16.8584 |  -0.1567 | nan      | nan      |   nan      |
| transformation_superclass | logreg         | 2365 | nan      | nan      |   0.5831 |   0.8231 |     0.5163 |
| transformation_superclass | majority       | 2365 | nan      | nan      |   0.2347 |   0.4144 |     0.0318 |

## Key findings

1. **Temperature regression**: best model is `mean_by_ec1` (constant per EC1, R² ≈ +0.11). DRFP+ridge has R² ≈ **−0.20** — actively worse than constant. Adding EC1 onehot to ridge does not rescue it (still negative).
2. **pH regression**: even worse. All models including ridge are negative R²; `mean` baseline is ≈ −0.01 (i.e. not informative beyond the global mean).
3. **catalyst_class** and **solvent_top12** classification: logreg accuracy is **LOWER than the majority baseline** (catalyst: 0.60 vs 0.70; solvent: 0.59 vs 0.71). Logreg only wins on macro-F1 by upweighting minority classes via `class_weight='balanced'`. The deployed predictor would actually hurt accuracy if we replaced the constant prediction.
4. **EC1** and **transformation_superclass** classification work as expected — logreg meaningfully beats majority on every metric.

## Root cause

DRFP-2048 captures reaction-structure features (atoms / bonds), but enzymatic temperature/pH are dominated by **enzyme thermal stability and active-site pKₐ**, not the reaction graph. Without a sequence/structure-derived enzyme embedding, the structural fingerprint is uninformative — and at worst introduces high-variance noise that overfits the training fold.

## Action items (overrides PROPOSAL §M4)

- **Stop reporting** `ridge_drfp` for T/pH — it is below the constant-baseline.
- **Deploy** `mean_by_ec1` as the trivial baseline for T (MAE ≈ 13.2 °C, R² ≈ +0.11).
- **Replace** with ESM-2/3 enzyme embedding + DRFP cross-attention (M4 redesign). Expected lift: T R² > 0.3 once enzyme identity enters the model.
- **Hold the line**: if ESM-only doesn't reach R² > 0.2 on DOI CV, do **not** publish T as a learned head — drop it from the deployed condition predictor.
