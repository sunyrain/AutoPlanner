# Model Strengthening Plan - 2026-05-19

## Core Decision

Do not train a replacement retrosynthesis generator in the next phase.

The current direction is:

```text
ChemEnzy native route pool
  -> product-audit material sanity
  -> route provenance / condition / EC evidence
  -> learned route selector
  -> conservative search-time transition bias
```

ChemEnzy remains the generator. AutoPlanner should learn which generated
routes and transitions are more useful under cascade/product constraints.

## Why The Current Effect Is Not Yet Good Enough

- Student-only controllers did not recover ChemEnzy native search ability.
- Raw ChemEnzy pools contain severe artifacts, including large unexplained atom
  gains.
- Stock closure can reward trivial purchase of complex intermediates.
- GT exact/reagent match is only one reference route, not the sole correct
  answer.
- Previous CCTS/retrieval signals have not yet beaten simple guarded reranking
  with enough evidence to be promoted.

## Phase 1: Route Pool Pack

Build a structured route-level pack from ChemEnzy raw, kept, and rejected routes.

Output:

```text
results/shared/model_strengthening_YYYYMMDD/
  route_pool_pack.jsonl
  route_pool_audit.json
  statin_panel_audit.json
  route_selector_split_manifest.json
```

Each row should contain:

```text
target_id
target_smiles
route_id
native_rank
native_score
n_steps
terminal_reactants
terminal_stock_status
product_audit_class
product_audit_issues
route_plausibility
condition_score_stats
enzyme_confidence_stats
source_model_counts
generic_template_fraction
large_atom_gain_count
v4_evidence_hits
cascade_block_hits
route_diversity_signature
split_id
```

Do not use gold/silver as route preference. Gold/silver can only be evidence
quality metadata.

## Phase 2: RouteSelector-v0

Train a simple pairwise/listwise route selector first.

Preferred first models:

```text
LightGBM ranker
XGBoost ranker
logistic pairwise ranker
```

Training comparisons within the same target:

```text
material-sane route > reject_artifact
reviewable stock-closed route > open-stock weak route
v4-supported cascade route > unsupported generic shortcut
lower-risk route > high-risk route
```

Features:

```text
native_rank
native_score
n_steps
strict_stock_solve
terminal_max_heavy_atoms
terminal_similarity_to_product
route_plausibility_passed
large_atom_gain_count
generic_template_fraction
condition_score_mean
enzyme_confidence_mean
v4_transition_hit_count
v4_block_hit_count
cascade_context_similarity
stock_shortcut_risk
route_diversity_cluster_size
```

Baselines:

```text
A ChemEnzy native rank
B product-audit rule-post
C learned selector only
D learned selector + product-audit guard
```

Implementation note: `train_route_pool_ranker.py` now reports `native_rank`,
`audit_guard`, `pairwise_logistic`, `native_rank_plus_learned`, and
`audit_guard_plus_learned`, which correspond to this A/B/C/D comparison.

Promotion condition:

```text
D > B on held-out split
top1 reject_artifact_rate decreases
top3 reviewable route count increases
stock closure does not collapse
route diversity does not collapse
```

If learned selector cannot beat product-audit rule-post, do not move to a more
complex neural model.

## Phase 3: Strict Splits

Use leakage-resistant splits:

```text
paper_id split
cascade_family split
target scaffold split
statin family held-out panel
```

Main evaluation:

```text
dataset_v4 blind split
statin panel
full100 sanity
PaRoutes / USPTO sanity only
```

Primary metrics:

```text
top1 product-audit class
top1 reject_artifact_rate
top3 has_reviewable_route
top3 has_stock_closed_reviewable_route
route_plausibility_pass_rate
cascade_evidence_recovery
stock_closure_guardrail
route_diversity
runtime
```

## Phase 4: Search-Time Integration

Only after RouteSelector-v0 beats rule-post:

```text
ChemEnzy candidate transition
  -> route-state features
  -> learned cascade/product score
  -> soft in-search bias
```

Rules:

- Do not hard-prune ChemEnzy candidates except severe material-sanity artifacts.
- Start as tie-break / soft priority.
- Sweep learned-score weight: `0.1`, `0.3`, `0.5`, `1.0`.
- Runtime gate can be relaxed to `20-30s/target`.

Critical ablation:

```text
final rerank only
vs
in-search transition scoring
```

If in-search is not better, do not claim planner-level contribution.

## Phase 5: Statin Panel

Use statins as application case studies, not training-specific optimization.

Report:

```text
raw routes
kept routes
rejected routes
top1 / top3 audit class
top3 reviewable routes
main rejection reasons
condition / EC confidence
route plausibility notes
```

The statin panel should answer:

- Does the system avoid chemically absurd atom-gain routes?
- Does it avoid trivial complex-stock shortcuts?
- Does it produce reviewable disconnections?
- Are low-confidence condition/EC predictions clearly marked?

## Immediate Next Actions

1. Implement `build_route_pool_selector_pack.py`. Status: initial version added
   under `cascade_planner/eval/` with focused tests.
2. Build a first route-pool pack from existing ChemEnzy Web artifacts and batch
   route-pool outputs. Status: first Web-artifact smoke pack exists at
   `results/shared/model_strengthening_20260519_web_artifacts/route_pool_pack.jsonl`.
3. Train `RouteSelector-v0` with pairwise/listwise objective. Status: initial
   pairwise wrapper added as `cascade_planner/eval/train_route_selector_v0.py`;
   smoke training report exists at
   `results/shared/model_strengthening_20260519_web_artifacts/route_selector_v0/route_selector_v0_report.json`.
4. Run A/B/C/D selector comparison on held-out route-pool targets.
5. Run the same selector comparison on the statin panel.
6. Promote to search-time integration only if learned selector beats rule-post.

## 2026-05-19 Smoke Result

The current Web-artifact pack is a toolchain smoke, not a promotion benchmark:

```text
pack rows: 194
raw rows before deduplication: 398
targets / selector groups: 3
auto-resplit counts: train 108 / val 60 / test 26
pairwise training rows: 384
selected method: audit_guard
native test MRR: 0.50
audit_guard test MRR: 1.00
pairwise_logistic test MRR: 1.00
```

Interpretation:

- The selector pack builder, grouped split, pairwise training, baselines, blends,
  and reports now run end-to-end.
- `selector_group_id` is now used for pairwise grouping, so filtered/raw/rejected
  artifacts from the same target can form valid positive/negative pairs.
- The dataset is too small and has only one validation/test group each; the
  numbers only prove the pipeline works, not that the learned selector is ready.
- The next required step is a larger ChemEnzy route-pool pack with strict
  scaffold/paper/family splits, then a real A/B/C/D comparison.

## 2026-05-19 V4 Route-Pool Result

Built a larger route-pool selector pack from existing ChemEnzy/CCTS runtime
artifacts:

```text
pack: results/shared/model_strengthening_20260519_v4_ccts_routepool/route_pool_pack.jsonl
rows after deduplication: 32,528
targets: 346
split counts: train 18,998 / val 3,946 / test 9,584
summary: results/shared/model_strengthening_20260519_v4_ccts_routepool/selector_ablation_summary.md
```

Key test results:

```text
native_rank                 MRR 0.8511  R@1 0.7273  R@3 0.8182
audit_guard                 MRR 1.0000  R@1 0.9192  R@3 0.9192
all_pairwise                MRR 1.0000  R@1 0.9192  R@3 0.9192
no_audit_pairwise           MRR 0.8555  R@1 0.7273  R@3 0.8182
no_audit_no_cascade         MRR 0.8538  R@1 0.7273  R@3 0.8182
cascade_only_pairwise       MRR 0.8576  R@1 0.7172  R@3 0.8485
```

Interpretation:

- `all_pairwise` is not a valid model win because it consumes audit-derived
  features that are close to the training label.
- `audit_guard` remains the strongest current production guard for this label
  definition.
- Pure cascade/CCTS features show a weak real signal: `cascade_only_pairwise`
  improves test R@3/R@5/R@10 over native rank, but lowers R@1 and is far behind
  `audit_guard`.
- This is not ready for search-time promotion. The next model step must change
  the label from product-audit artifact labels to transition/block-level cascade
  consistency labels.

## 2026-05-19 Pair-Scorer Result

Built and trained the first usable transition/block-level cascade consistency
model:

```text
pack dir: results/shared/model_strengthening_20260519_pair_scorer/pair_pack
model: results/shared/model_strengthening_20260519_pair_scorer/cascade_pair_scorer.pt
report: results/shared/model_strengthening_20260519_pair_scorer/cascade_pair_scorer_report.md
```

Data:

```text
source: dataset_v4_release/cascade_v4_high_quality.jsonl
full100 benchmark overlap: excluded via data/benchmark_v2_100.json
rows: 9,552
positive adjacent step pairs: 3,184
hard negatives: 6,368
train / val / test: 7,509 / 978 / 1,065
```

Held-out test:

```text
learned pair scorer pairwise group accuracy: 0.998688
rule pair baseline pairwise group accuracy: 0.988189
learned top1 compatibility hit rate: 0.995984
rule top1 compatibility hit rate: 0.959839
```

Trace replay diagnostic:

```text
summary: results/shared/model_strengthening_20260519_pair_scorer/cascade_pair_scorer_phase_summary.md
trace: results/shared/cascade_transition_value/full100_trace_20260509/baseline_trace_merged.jsonl
base top1 stock/no-failure/oracle-child-quality: 0.8452 / 0.8452 / 0.7917
rule_w0p005: 0.8452 / 0.8452 / 0.7917
learned_w0p005: 0.8214 / 0.8214 / 0.7679
learned changed top1 vs base: 0.3274
learned_guarded_w0p005_eps0: 0.8452 / 0.8452 / 0.7917
learned_guarded pair-informative mean child quality: 1.5130 vs base 1.5059
```

Interpretation:

- This is the current strongest model-side result because it uses adjacent-step
  cascade compatibility labels rather than product-audit route labels.
- It still needs a harder ChemEnzy-candidate evaluation. Current hard negatives
  are generated from v4 step substitutions and condition conflicts, not from
  ChemEnzy's actual proposal distribution.
- Directly adding learned pair reward to old search scores is not safe; it
  overrules many base ties and worsens stock/no-failure guardrails.
- Guarded tie-break is the viable search-time form: it preserves the current
  stock/no-failure guardrails in trace replay and gives a small child-quality
  gain on pair-informative events.
- This guarded tie-break is now implemented in live `CascadeProgramSearch` via
  `CascadeSearchConfig(pair_reward_mode="guarded_tie_break")`; focused tests
  cover both safe tie-break and stock-closure preservation.
- The next model step is no longer merely "build hard negatives": the existing
  ChemEnzy runtime hard-negative result must be turned into a guarded live
  search benchmark and a learned-vs-retrieval ablation.

## 2026-05-19 Runtime Hard-Negative CCTS Result

Existing CCTS-v3 runtime artifacts already cover the planned "ChemEnzy
candidate hard-negative" direction:

```text
source: results/shared/cascadebench_strict_20260516/ccts_v3_runtime_candidate_cache/
ranker: results/shared/cascadebench_strict_20260516/ccts_v3_runtime_pairwise_ranker/
summary: results/shared/model_strengthening_20260519_transition_hardneg/transition_hardneg_summary.md
```

Data:

```text
train candidate rows / groups: 173,217 / 3,184
val candidate rows / groups:    31,267 /   617
test candidate rows / groups:   29,564 /   620
test block-supported coverage:  46.5%
test exact coverage:            23.2%
```

Selected method:

```text
chem_plus_runtime_any_plus_runtime_pairwise_block_supported_positive_label__runtime_evidence_only
```

Held-out test:

```text
ChemEnzy original block-supported MRR: 0.391640
selected residual blend block MRR:     0.423058
delta:                                +0.031418

ChemEnzy original exact MRR:           0.335809
selected residual blend exact MRR:     0.375055
delta:                                +0.039246
```

Important caveat:

```text
nonlearned retrieval blend block MRR: 0.422418
```

Interpretation:

- The v4 cascade evidence signal is real on ChemEnzy runtime candidates.
- The current learned blend has not yet clearly beaten retrieval-only evidence.
- Candidate coverage is a hard ceiling: exact candidates appear in only 23.2%
  of held-out test groups.
- This is a valid CCTS-v1 candidate-level scorer direction, but not enough for
  search promotion or a publication claim.

Next required experiment:

```text
A ChemEnzy original rank
B retrieval-only evidence blend
C learned CCTS final rerank
D learned CCTS guarded in-search tie-break
```

The benchmark runner now exposes the guarded in-search mode directly:

```bash
PYTHONPATH=. python -m cascade_planner.eval.run_cascade_search_benchmark \
  --benchmark data/benchmark_v2_100.json \
  --output results/shared/.../guarded_ccts.json \
  --cascade-pair-scorer results/shared/model_strengthening_20260519_pair_scorer/cascade_pair_scorer.pt \
  --cascade-pair-reward-weight 0.005 \
  --cascade-pair-reward-mode guarded_tie_break \
  --cascade-pair-reward-tie-epsilon 0.0
```

The full v4 command-manifest generator also forwards these options:

```bash
PYTHONPATH=. python -m cascade_planner.eval.run_v4_full_training_pipeline \
  --split-dir results/shared/.../splits \
  --output-root results/shared/.../pipeline \
  --cascade-pair-reward-weight 0.005 \
  --cascade-pair-reward-mode guarded_tie_break \
  --cascade-pair-reward-tie-epsilon 0.0
```

A concrete guarded manifest has been generated for the current strict v4
splits:

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/v4_full_training_pipeline_manifest.json
results/shared/model_strengthening_20260519_guarded_v4_pipeline/v4_full_training_commands.sh
```

This manifest has 12 commands using `baseline` train/val traces, no bootstrap
stage, and a locked full100 eval with:

```text
--cascade-pair-reward-weight 0.005
--cascade-pair-reward-mode guarded_tie_break
--cascade-pair-reward-tie-epsilon 0.0
```

The first two model-only commands from this manifest have been executed:

```text
build_pair_pack: passed
train_pair_scorer: passed
report: results/shared/model_strengthening_20260519_guarded_v4_pipeline/PROGRESS.md
model: results/shared/model_strengthening_20260519_guarded_v4_pipeline/models/cascade_pair_scorer.pt
```

This produced a 6,368-row adjacent-step pair pack and a pair scorer with held-out
test pairwise group accuracy 0.998688 against a rule baseline of 0.984252. The
ChemEnzy-dependent trace/action/source/transition/full100 commands are still
pending.

A 5-target live smoke has also been run with that pair scorer:

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/SMOKE_REPORT.md
```

The smoke confirms that guarded pair scoring is active in live search:

```text
cascade_pair_applicable=true: 12
cascade_pair_reward_applied=true: 2
cascade_pair_guard_reason=outside_base_tie_window: 10
```

All 5 smoke targets remained solved and stock closed. This is an integration
check only; it is not a model-quality claim.

A 20-target baseline-vs-guarded smoke has also been run:

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_guarded_pair_limit20_comparison.md
```

Result:

```text
baseline cascade_solved_rate: 0.9
guarded cascade_solved_rate: 0.9
baseline stock_closed_rate: 1.0
guarded stock_closed_rate: 1.0
baseline result_exact_reaction_in_pool: 0.05
guarded result_exact_reaction_in_pool: 0.05
baseline result_gt_reactant_in_pool: 0.35
guarded result_gt_reactant_in_pool: 0.35
guarded pair applicable: 72
guarded pair reward applied: 23
top-route changed targets: 1 / 20
```

Interpretation: the guarded scorer changes live search in at least some cases
without hurting this small sample, but no aggregate quality lift is visible yet.

Tie-epsilon sweep:

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_tie_sweep.md
```

This report is now generated by a reusable comparison utility:

```bash
PYTHONPATH=. python -m cascade_planner.eval.compare_cascade_search_runs \
  --baseline results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_baseline_limit20.json \
  --run guarded_eps0=results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_guarded_pair_limit20.json \
  --run guarded_eps003=results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_guarded_pair_limit20_eps003.json \
  --trace guarded_eps0=results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_guarded_pair_limit20_trace.jsonl \
  --trace guarded_eps003=results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_guarded_pair_limit20_eps003_trace.jsonl \
  --output-json results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_tie_sweep.json \
  --output-md results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_tie_sweep.md
```

```text
baseline       solved 0.9  stock 1.0  result_exact 0.05  result_GT 0.35
guarded eps0   solved 0.9  stock 1.0  result_exact 0.05  result_GT 0.35  applied 23  changed 1/20
guarded eps.03 solved 0.9  stock 1.0  result_exact 0.05  result_GT 0.35  applied 27  changed 1/20
```

Increasing tie epsilon to 0.03 increases the number of applied pair rewards but
does not produce aggregate lift on this small sample.

Weight sweep:

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_weight_tie_sweep.md
```

## 2026-05-19 Route/Block Gate Summary

The current route/block evidence is consolidated here:

```text
results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.json
results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.md
```

Decision:

```text
promote_route_block_scorer: false
status: do_not_promote_yet
```

Key evidence:

```text
route-pool native MRR:                 0.170446
route-pool cascade-only model MRR:     0.645811
route-pool CCTS model-mean MRR:        0.618979
runtime ChemEnzy block MRR:            0.391640
runtime retrieval-only block MRR:      0.422418
runtime learned selected block MRR:    0.423058
guarded live search GT result lift:    0.000000
```

Interpretation:

- Route/block model features beat native rank on the current fixed route pool.
- They do not yet clearly beat retrieval-only evidence controls.
- The learned runtime hard-negative scorer improves over ChemEnzy, but only by
  `0.00064` MRR over the retrieval-only block control.
- Guarded live search is safe and active, but currently has no aggregate quality
  lift.

Therefore the next strengthening step is not search-time promotion. It is:

```text
route_block_value_pack_v1
  -> strict train-only retrieval evidence
  -> learned route/block outcome scorer
  -> explicit learned-vs-retrieval promotion gate
```

The first `route_block_value_pack_v1` has now been built from the current
32,528-route ChemEnzy/CCTS route pool:

```text
results/shared/model_strengthening_20260519_route_block_value_v1/route_block_value_pack.jsonl
results/shared/model_strengthening_20260519_route_block_value_v1/route_block_value_pack_report.json
```

Report:

```text
rows: 32528
targets: 346
split_counts: train 18998 / val 3946 / test 9584
pair_context_evidence_any: 8211
strong_route_evidence: 787
reject_artifact: 4616
reviewable_by_audit: 27912
```

Important schema choice: this pack does **not** create a single weighted
route-value score. It separates `native`, `stock_route`, `product_audit`,
`cascade_retrieval`, `learned_ccts`, `condition_enzyme`, and `route_context`
feature groups, with weak label tasks kept separate. This is intentional so
the next model can prove value over retrieval-only and audit-only controls.

## 2026-05-19 Route/Block Value Model Pilot

Initial trainer:

```text
cascade_planner/eval/train_route_block_value_model.py
```

Pilot outputs:

```text
results/shared/model_strengthening_20260519_route_block_value_v1/models/
  strong_evidence_no_cascade_no_audit/
  strong_evidence_no_retrieval_no_audit/
  strong_evidence_with_retrieval_no_audit/
  reviewable_vs_reject_no_cascade_no_audit/
  reviewable_vs_reject_no_audit/
```

Results:

```text
strong_evidence_no_cascade_no_audit:
  native MRR        0.473143
  retrieval MRR     1.000000
  model MRR         0.653142
  model - retrieval -0.346858

strong_evidence_no_retrieval_no_audit:
  native MRR        0.473143
  retrieval MRR     1.000000
  model MRR         0.966667
  model - retrieval -0.033333
  note              excludes retrieval/audit but still uses learned_ccts features

strong_evidence_with_retrieval_no_audit:
  native MRR        0.473143
  retrieval MRR     1.000000
  model MRR         1.000000
  model - retrieval  0.000000

reviewable_vs_reject_no_audit:
  native MRR        0.851071
  retrieval MRR     0.790797
  audit_guard MRR   1.000000
  model MRR         0.869885
  model - native    0.018814
  model - retrieval 0.079088

reviewable_vs_reject_no_cascade_no_audit:
  native MRR        0.851071
  retrieval MRR     0.790797
  audit_guard MRR   1.000000
  model MRR         0.863744
  model - native    0.012673
  model - retrieval 0.072947
```

Interpretation:

- `strong_route_evidence` is currently too close to retrieval / learned CCTS
  evidence. A model with retrieval features only reproduces the retrieval
  baseline; a model without retrieval but with learned CCTS still does not clear
  retrieval; and a model with both retrieval and learned CCTS removed falls to
  `0.653142` MRR.
- `reviewable_vs_reject_no_audit` gives a modest learned signal after excluding
  direct audit features. The signal remains small when retrieval/learned CCTS
  are also removed, and neither variant beats the audit guard.
- The current value pack now reports
  `evidence_provenance_audit.status=unverifiable_without_source_provenance` for
  all 32,528 retrieval-feature rows. Retrieval-conditioned claims must therefore
  be treated as unverified until evidence source provenance is rebuilt.
- The pack builder now has a hard gate:
  `--require-evidence-provenance` requires both source provenance and an
  explicit train-only marker. The upstream selector-pack builder can now attach
  `evidence_source_split`, `retrieval_corpus_manifest`, and
  `train_only_retrieval`, but the current 32,528-row pack was produced before
  those fields existed and remains unverifiable.
- A strict runtime-only provenance rebuild now exists:

```text
selector pack:
  results/shared/model_strengthening_20260519_v4_ccts_routepool_runtime_train_provenance/route_pool_pack.jsonl

value pack:
  results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl
  evidence_provenance_audit.status: verified_or_no_retrieval_features
  runtime_retrieval_only: true
```

Strict-pack pilot:

```text
strong_evidence_runtime_no_cascade_no_audit:
  model MRR         0.653142
  retrieval MRR     1.000000
  clears retrieval  false

strong_evidence_runtime_with_retrieval_no_audit:
  model MRR         1.000000
  retrieval MRR     1.000000
  clears retrieval  false

reviewable_vs_reject_runtime_no_audit:
  model MRR         0.875724
  native MRR        0.851071
  retrieval MRR     0.790797
  audit_guard MRR   1.000000

reviewable_vs_reject_runtime_no_cascade_no_audit:
  model MRR         0.863744
  native MRR        0.851071
  retrieval MRR     0.790797
  audit_guard MRR   1.000000
```

This fixes the provenance weakness for runtime CCTS features, but it does not
change the promotion decision: learned models still do not beat the simple
retrieval/audit controls.

To make the next real-review run more informative, a strict model-control
disagreement review worklist has been generated:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/
  strict_model_control_disagreement_review.jsonl
  strict_model_control_disagreement_review.csv
  strict_model_control_disagreement_prompts.jsonl
  strict_model_control_disagreement_review_report.json
```

It contains:

```text
rows: 120
targets: 47
source_pool: strict_runtime_train_provenance
selection_basis: strict value-model vs retrieval/audit/native rank disagreement
```

Dry-run validation:

```text
strict_model_dryrun_labels: 120
placeholder_review: 120
usable positives / negatives: 0 / 0
```

This worklist is ready for the real reviewer pipeline once `DEEPSEEK_API_KEY`
is configured. It should be preferred over the generic 150-row expansion when
the goal is to diagnose why the learned value model fails to beat retrieval or
audit controls.

The generation logic is now reusable:

```bash
PYTHONPATH=. python -m cascade_planner.eval.build_strict_model_review_worklist \
  --value-pack results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl \
  --model-pickle results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/models/reviewable_vs_reject_runtime_no_audit/route_block_value_model.pkl \
  --output-jsonl results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.jsonl \
  --output-csv results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.csv \
  --report results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review_report.json
```
- This confirms the next model should not use `strong_route_evidence` as the
  main publication label. It is useful as a control/diagnostic target only.
- The next useful label target must combine route/block outcome evidence that is
  not simply a thresholded retrieval feature: e.g. reviewed route evidence,
  non-artifact route outcome, statin qualitative review tags, or search-time
  recovery of supported blocks under fixed candidate pools.

Available route-pool review labels are currently insufficient as a main
supervised training set:

```text
human CSV review:
  results/shared/cascadebench_strict_20260516/route_pool_evidence_review_batch/
  accepted_rows 0 / csv_rows 51

self-review calibration:
  results/shared/cascadebench_strict_20260516/route_pool_evidence_review_calibration_csv_pipeline_self_review/
  accepted_rows 36
  usable_positive_rows 5
  usable_negative_rows 31
```

The 36 self-review rows are useful for calibration and label-design auditing,
but they are not enough to train or claim a robust route-quality model. The
next data task is to expand calibrated route/block outcome labels or replace
them with stricter automatically verifiable outcome labels.

The available review labels have been normalized into a calibration-only pack:

```text
results/shared/model_strengthening_20260519_route_block_review_labels/route_block_review_label_pack.jsonl
results/shared/model_strengthening_20260519_route_block_review_labels/route_block_review_label_pack_report.json
```

Counts:

```text
rows: 36
targets: 18
review_source_counts: self_review 36
usable_positive_rows: 5
usable_negative_rows: 31
```

Contract: this pack is `calibration_only_until_sufficient_expert_labels`; it is
not route preference training data.

To expand the missing outcome labels, a full 150-row review worklist has been
generated from the existing 20/full100/statin route-pool evidence audits:

```text
results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion.jsonl
results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion.csv
results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion_worklist.csv
results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion_worklist_report.json
```

Coverage:

```text
rows: 150
classes: any_analog_supported 75 / multistep_without_observed_pair 74 / same_pair_analog_supported 1
pools: 20 51 / full100 49 / statin 50
```

Transform-label sanity was also attached:

```text
rows_with_transform_label_warning: 85 / 150
row_label_mismatch_rate: 0.566667
recommended_use: review_triage_only
```

This reinforces that transform labels in the current route-pool evidence are
too noisy to use as direct supervised labels without review.

The 150-row worklist has also been validated through the existing review
pipeline in dry-run mode:

```text
results/shared/model_strengthening_20260519_review_expansion/dryrun_pipeline/
```

Dry-run manifest:

```text
prompt_rows: 150
written_rows: 150
accepted_rows: 150
usable_positive_rows: 0
usable_negative_rows: 0
promotion_gate.ready_for_training: false
```

This only verifies pipeline wiring. It intentionally produces `unclear`
placeholder labels and must not be used for training. A real LLM/expert review
run requires a configured reviewer API key and should write to a non-dry-run
output directory.

The review label pack builder now explicitly detects placeholder reviews:

```text
results/shared/model_strengthening_20260519_review_expansion/dryrun_pipeline/expansion_dryrun_review_label_pack_report.json
placeholder_review: 150
usable_positive_rows: 0
usable_negative_rows: 0
```

This prevents dry-run labels from silently becoming training positives or
negatives.

The current phase completion audit is documented at:

```text
docs/MODEL_STRENGTHENING_COMPLETION_AUDIT_2026-05-19.md
```

Audit conclusion: the plan is not complete. The model/data pipeline is
organized, but promotion is blocked by missing real route/block outcome labels
and lack of learned-vs-retrieval/audit superiority.

Real LLM review can be launched with:

```bash
# either export the key or put DEEPSEEK_API_KEY=... in .env.local or .env
export DEEPSEEK_API_KEY=...
PYTHONPATH=. scripts/run_route_block_review_expansion_real.sh
```

The runner writes to:

```text
results/shared/model_strengthening_20260519_review_expansion/real_review_pipeline/
```

It reads `DEEPSEEK_API_KEY` from the environment, `.env.local`, or `.env`;
then validates the final promotion gate / review-label-pack JSON outputs.

```text
guarded eps.03 weight 0.005: solved 0.9  stock 1.0  result_exact 0.05  result_GT 0.35  changed 1/20
guarded eps.03 weight 0.05:  solved 0.9  stock 1.0  result_exact 0.05  result_GT 0.35  changed 1/20
```

This suggests the missing lift is not simply caused by a too-small pair reward
weight. The learned adjacent-step scorer is safe but too weakly coupled to route
quality under the current search policy.

Decision report:

```text
docs/GUARDED_CCTS_DECISION_2026-05-19.md
```

Decision: keep adjacent-step CCTS as a safe diagnostic/tie-break feature, but do
not promote it as the main model contribution. The next model-strengthening
track should optimize route/block outcome scoring and compare learned scoring
against retrieval-only evidence rank.

Promotion rule:

```text
D must beat B on block/exact recovery while preserving stock/no-failure guardrails.
If C or D only tie B, the model contribution is not learned CCTS yet; it is
retrieval/evidence-conditioned ranking.
```
