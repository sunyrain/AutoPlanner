# Model Strengthening Completion Audit - 2026-05-19

## Objective

当前目标是“全面完成计划”，在本阶段可具体化为：

1. 固定 ChemEnzy 作为 generator。
2. 建立 route/block outcome 数据层，而不是继续训练 student-only 或 adjacent-pair 主模型。
3. 训练并评估 route/block value model。
4. 证明 learned scorer 超过 native、audit-only、retrieval-only controls。
5. 准备可扩展的 route/block review label 流程。
6. 只有在 learned scorer 和 guarded live search 过门槛后，才 promotion。

## Latest No-Human Update

根据后续约束，主训练路径已从 expert/review labels 改为 no-human weak
supervision。专家审阅 packet 和 DEEPSEEK runner 只保留为 fallback/audit 资产，
不再是主线 blocker。

当前 no-human route/block value pack 与 ablation 已完成：

```text
pack rows: 32528
targets: 346
split: train 18998 / val 3946 / test 9584
no_human_consensus_positive: 7765
no_human_consensus_negative: 4616
expert_labels_required: false
```

当前 fixed-pool benchmark：

```text
native rank baseline MRR: 0.707357
retrieval-only baseline MRR: 0.761806
no_human_all_features MRR: 1.000000
no_human_no_audit MRR: 0.797227
no_human_no_audit_no_retrieval MRR: 0.782899
no_human_no_audit_no_cascade MRR: 0.816142
no_human_route_no_audit_no_retrieval MRR: 0.871659
no_human_route_no_audit_no_retrieval minus retrieval: 0.080862
final_rerank_model_minus_native: 0.020588
final_rerank_model_minus_retrieval: 0.080862
final_rerank_model_minus_audit: -0.128341
guarded_eps008_w0005_result_exact: 0.05
guarded_eps008_w0005_result_gt: 0.35
guarded_top_result_exact: 0.00
guarded_top_result_gt: 0.30
additive_w005_top_result_exact: 0.00
additive_w005_top_result_gt: 0.30
route_block_final_rerank_top_result_exact: 0.00
route_block_final_rerank_top_result_gt: 0.25
route_block_final_rerank_changed: 7/20
product_audit_final_rerank_top_result_exact: 0.00
product_audit_final_rerank_top_result_gt: 0.35
product_audit_final_rerank_changed: 4/20
guarded_live_quality_lift: +0.05 via no-label product-audit final rerank
```

结论：no-human 固定池 strict gate 已通过，且无专家标签 product-audit final
rerank 在 20-target live smoke 上给出小幅 top-GT 提升。但 learned route/block
scorer 仍未通过 promotion：runtime hard-negative learned scorer 只比 retrieval-only
control 高 `0.00064`，低于 `+0.03` 门槛；learned final-rerank live 结果也是负向。
剩余 blocker 不是专家标签，而是 learned scorer 必须超过 retrieval-only runtime
control。

追加的 runtime no-human probe 也没有解除该 blocker：

```text
artifact: results/shared/model_strengthening_20260519_transition_hardneg_nohuman_probe/runtime_hardneg_nohuman_probe.json
retrieval_test_block_mrr: 0.422418
best_blend_test_block_mrr: 0.422418
best_delta_vs_retrieval: 0.000000
required_delta_vs_retrieval: 0.030000
decision: fail
```

## Checklist

| Requirement | Evidence | Status |
|---|---|---|
| ChemEnzy fixed as generator | `docs/MODEL_STRENGTHENING_PLAN_2026-05-19.md` states ChemEnzy remains generator | done |
| Adjacent CCTS not promoted as main contribution | `docs/GUARDED_CCTS_DECISION_2026-05-19.md` | done |
| Route/block gate summary created | `results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.json` | done |
| Route/block scorer promotion gate | summary decision: `promote_route_block_scorer=false` | not passed |
| Route/block value pack built | `results/shared/model_strengthening_20260519_route_block_value_v1/route_block_value_pack.jsonl` | done |
| Route/block value pack scale | report: `rows=32528`, `targets=346`, train/val/test present | done |
| Route/block retrieval provenance verified | report: `evidence_provenance_audit.status=unverifiable_without_source_provenance` | failed |
| Provenance hard gate implemented | `build_route_block_value_pack.py --require-evidence-provenance` now requires explicit train-only provenance | done |
| Upstream selector provenance support | `build_route_pool_selector_pack.py` can attach `evidence_source_split`, `retrieval_corpus_manifest`, and `train_only_retrieval` | done |
| Strict runtime train-provenance pack built | `results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack_report.json` has `status=verified_or_no_retrieval_features` | done |
| Strict route/block value ablation summarized | `results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/strict_route_block_value_ablation_summary.md` | done |
| Route/block value trainer implemented | `cascade_planner/eval/train_route_block_value_model.py` | done |
| Strong-evidence model beats retrieval-only | `strong_evidence_*` reports show retrieval MRR `1.0`; model does not exceed it | failed |
| Reviewable-vs-reject model beats native/retrieval | model MRR `0.869885`, native `0.851071`, retrieval `0.790797` | partial |
| Reviewable-vs-reject model beats audit guard | audit_guard MRR `1.0`; model does not beat it | failed |
| Existing review labels normalized | `results/shared/model_strengthening_20260519_route_block_review_labels/route_block_review_label_pack.jsonl` | done |
| Existing review labels sufficient for training | report: `usable_positive_rows=5`, self-review dominated, `sufficient_for_main_training=false` | failed |
| Expanded review worklist created | `results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion_worklist.csv` | done |
| Expanded worklist transform sanity checked | `85/150` rows have transform-label warning | done, high-noise |
| Strict model-control review worklist created | `results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.jsonl` has `120` rows / `47` targets | done |
| Strict model-control worklist builder implemented | `cascade_planner/eval/build_strict_model_review_worklist.py` with `tests/test_build_strict_model_review_worklist.py` | done |
| Strict model-control dry-run validated | `strict_model_dryrun_review_label_pack_report.json` has `placeholder_review=120`, usable positives/negatives `0/0` | done |
| Strict 300-row fallback worklist created | `results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/strict_model_control_disagreement_review_300.jsonl` has `300` rows / `78` targets | done |
| Strict 300-row prompts created | `strict_model_control_disagreement_prompts_300.jsonl` has `300` prompt rows | done |
| Human/external review packets created | `results/shared/model_strengthening_20260519_strict_model_review_packet*/route_pool_evidence_review_calibration_subset_TO_FILL.csv` | done |
| Blank human packet validation | blank CSV pipelines report all rows as unreviewed and `ready_for_training=false` | done |
| Portable external-review packet created | `results/shared/model_strengthening_20260519_strict_review_packets.tar.gz` plus `.sha256` and top-level packet README | done |
| Strict packet CSV validation command aligned | packet READMEs, packet metadata JSON, and packet builder support `--min-evidence-classes 1` because strict disagreement packets use one evidence class | done |
| Human CSV route identity preserved | `ingest_route_pool_evidence_review_csv.py` now emits `target_id`, `route_id`, and rank metadata for merge | done |
| Human CSV one-command continuation prepared | `scripts/run_strict_review_from_filled_csv.sh` validates filled CSVs, merges by `route_id`, refreshes readiness, and trains only if ready | done |
| Human CSV positive-path smoke validated | synthetic filled CSV matched `6/6` route ids and produced split-ready positive/negative labels with `ready_for_expert_training=true` under test thresholds | done |
| Review label-pack decision is merge-aware | `build_route_block_review_label_pack.py` no longer reports human CSV labels as self-review dominated; raw label packs still require merge gate before training | done |
| Review prompt/result route identity preserved | prompt and ingested label rows now preserve `target_id`, `route_id`, and `source_value_pack` | done |
| Review labels can merge back into value pack | `cascade_planner/eval/merge_route_block_review_labels.py` adds `expert_review_positive/negative` tasks | done |
| Dry-run review merge validated | dry-run merge matches `120/120` review route ids and creates `0/0` usable labels because all are placeholders | done |
| Review pipeline wiring validated | dry-run manifest has `prompt_rows=150`, `written_rows=150` | done |
| Dry-run labels protected from training | dry-run review label pack has `placeholder_review=150`, usable positives/negatives `0/0` | done |
| Real review runner prepared | `scripts/run_route_block_review_expansion_real.sh` validates key presence and output JSON | done |
| Real review runner input override | runner now accepts `REVIEW_JSONL`, `TRANSFORM_SANITY_JSON`, and `REVIEW_PREFIX` env overrides | done |
| Strict model review real-run wrapper | `scripts/run_strict_model_review_real.sh` wraps the strict disagreement worklist and currently exits on missing `DEEPSEEK_API_KEY` | done |
| Strict model review 300-row wrapper | `scripts/run_strict_model_review_real_extended.sh` wraps the 300-row fallback worklist and currently exits on missing `DEEPSEEK_API_KEY` | done |
| Real review wrappers auto-merge labels | strict wrappers set `VALUE_PACK`; generic real runner emits `*_merged_route_block_value_pack.jsonl` when review succeeds | done |
| Real review runtime defaults hardened | wrapper defaults to `WORKERS=4`, resumable cache, and `AUTOPLANNRELLM_DEEPSEEK_TIMEOUT_S=300` | done |
| One-command after-key continuation prepared | `scripts/run_strict_review_full_after_key.sh` runs 120-row review, refreshes readiness, trains if ready, and can optionally run 300-row fallback | done |
| Continuation wrappers fail incomplete gates | after-key and filled-CSV wrappers exit `4` when the merged review pack is not ready, unless `ALLOW_NOT_READY_EXIT_ZERO=1` is set for inspection-only runs | done |
| Delivery index created | `docs/DELIVERY_INDEX_2026-05-19.md` centralizes handoff and blockers | done |
| Top-level packet README created | `results/shared/model_strengthening_20260519_strict_review_PACKET_README.md` explains fill order and return path | done |
| Expert-review training wrapper prepared | `scripts/train_strict_model_review_value_models.sh` checks merge readiness then trains expert-review ablations | done |
| Training wrapper refuses placeholder merge | dry-run merged pack exits with `ready_for_expert_training=false` before training | done |
| Trainer accepts merged expert-review tasks | `tests/test_train_route_block_value_model.py` covers `expert_review_positive` / `expert_review_negative` | done |
| Strict review readiness checker added | `cascade_planner/eval/check_strict_review_pipeline_readiness.py` writes `results/shared/model_strengthening_20260519_strict_review_readiness.json` | done |
| Readiness checker validates packet handoff | readiness now checks packet metadata commands, required context columns, archive SHA256, and required archive members, not just packet file existence | done |
| Secret scan before packet handoff | assignment-level scan found `suspicious_deepseek_assignments=0`; only placeholders/docs/tests reference `DEEPSEEK_API_KEY` | done |
| Packet checksum validated | `sha256sum -c results/shared/model_strengthening_20260519_strict_review_packets.tar.gz.sha256` reports OK | done |
| Mock non-dry-run review path validated | `mock_real_pipeline` wrote 120 non-dry-run mock responses, label pack, and merged value pack with 120/120 route matches | done |
| Training wrapper rejects one-sided mock labels | mock merge has 120 positives / 0 negatives and exits before training | done |
| Phase II handoff written | `docs/PHASE2_MODEL_HANDOFF_2026-05-19.md` | done |
| Local key template added | `.env.local.example` contains copy/use comments and a placeholder `DEEPSEEK_API_KEY` only | done |
| Placeholder key guard | readiness, strict review runners, the shared DeepSeek client, and the agent prior/benchmark/CLI paths reject `replace_with_your_deepseek_key` before any API call or mark it unusable, after trimming surrounding whitespace/quotes | done |
| Real LLM/expert review labels produced | no non-dry-run reviewer output exists in this phase | missing |
| Learned scorer promoted into search | promotion gate remains false; no search-time promotion | not done |

## Current Results

Route/block gate summary:

```text
status: do_not_promote_yet
reason:
  runtime hard-negative learned scorer does not clear retrieval-only control
```

Runtime no-human probe:

```text
material_sanity_pairwise_C0.01:
  test block MRR       0.343901
  blend test block MRR 0.422418
  blend alpha          0.0

hgb_block_cls_lr0.1_l2_0.1:
  test block MRR       0.264874
  blend test block MRR 0.422418
  blend alpha          0.0
```

Route/block value model pilot:

```text
strong_evidence_no_cascade_no_audit:
  model MRR         0.653142
  retrieval MRR     1.000000
  clears retrieval  false

strong_evidence_no_retrieval_no_audit:
  model MRR         0.966667
  retrieval MRR     1.000000
  clears retrieval  false
  note              uses learned_ccts features

strong_evidence_with_retrieval_no_audit:
  model MRR         1.000000
  retrieval MRR     1.000000
  clears retrieval  false

reviewable_vs_reject_no_audit:
  model MRR         0.875724
  native MRR        0.851071
  retrieval MRR     0.790797
  audit_guard MRR   1.000000

reviewable_vs_reject_no_cascade_no_audit:
  model MRR         0.863744
  native MRR        0.851071
  retrieval MRR     0.790797
  audit_guard MRR   1.000000
```

Route/block value pack provenance audit:

```text
legacy pack:
evidence_provenance_audit.status: unverifiable_without_source_provenance
missing_retrieval_provenance_rows: 32528
missing_train_only_marker_rows: 0

strict runtime train-provenance pack:
evidence_provenance_audit.status: verified_or_no_retrieval_features
missing_retrieval_provenance_rows: 0
missing_train_only_marker_rows: 0
```

Review label availability:

```text
existing self-review labels:
  rows 36
  usable positive 5
  usable negative 31
  sufficient_for_main_training false

expanded review worklist:
  rows 150
  any_analog_supported 75
  multistep_without_observed_pair 74
  same_pair_analog_supported 1
  transform-label warnings 85

strict model-control review worklist:
  rows 120
  targets 47
  source_pool strict_runtime_train_provenance
  dry-run placeholder_review 120
  usable positive 0
  usable negative 0

strict model-control fallback worklist:
  rows 300
  targets 78
  source_pool strict_runtime_train_provenance
  prompts 300

human/external review packets:
  120-row packet TO_FILL.csv ready
  300-row packet TO_FILL.csv ready
  blank validation accepted rows 0 and ready_for_training false

dry-run review merge:
  matched review route ids 120/120
  matched value rows 120
  usable positive 0
  usable negative 0
  ready_for_expert_training false
  split_ready train/val/test false

mock non-dry-run review smoke:
  output dir results/shared/model_strengthening_20260519_strict_model_review_worklist/mock_real_pipeline
  attempted rows 120
  written rows 120
  error rows 0
  usable positive 120
  usable negative 0
  matched route ids 120/120
  training wrapper refused because negative labels are missing

dry-run pipeline:
  prompt_rows 150
  written_rows 150
  placeholder_review 150
  usable positive 0
  usable negative 0
```

## Verification

Focused tests run successfully:

```text
full project pytest after pytest.ini scoping: 472 passed, 4 warnings
test_build_route_block_review_label_pack.py: OK
test_build_route_block_value_pack.py: OK
test_train_route_block_value_model.py: OK
test_summarize_route_block_strengthening.py: OK
test_build_route_pool_selector_pack.py: OK
test_build_strict_model_review_worklist.py: OK
test_merge_route_block_review_labels.py: OK
test_check_strict_review_pipeline_readiness.py: OK
test_cascadebench_v4_selectors.py: OK
py_compile: OK
JSON validation: OK
bash -n run_route_block_review_expansion_real.sh: OK
bash -n run_strict_model_review_real.sh: OK
```

Latest targeted review/merge/readiness/training matrix:

```text
test_build_route_block_review_label_pack.py: OK
test_merge_route_block_review_labels.py: OK
test_check_strict_review_pipeline_readiness.py: OK
test_strict_human_csv_merge_path.py: OK
test_strict_review_wrapper_exit_codes.py: OK
test_train_route_block_value_model.py: OK
test_cascadebench_v4_selectors.py: OK
```

Formatting check:

```text
git diff --check on strict review / merge / readiness / training changes: OK
```

## Conclusion

The plan is not complete.

The data/model pipeline is now organized and guarded, but the model is not yet
strong enough for promotion. The key blocker is supervised signal quality:
`strong_route_evidence` is retrieval-derived, `reviewable_vs_reject` still loses
to audit guard, and available reviewed labels are too small/self-review-heavy.

## Required Next Step

Run real review on the 120-row strict model-control disagreement worklist:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.jsonl
```

Prepared runner:

```bash
export DEEPSEEK_API_KEY=...
PYTHONPATH=. scripts/run_strict_model_review_real.sh
```

If the 120-row set does not provide enough usable positives/negatives, run the
300-row fallback:

```bash
PYTHONPATH=. scripts/run_strict_model_review_real_extended.sh
```

If no local API key is available, fill the expert columns in:

```text
results/shared/model_strengthening_20260519_strict_model_review_packet/route_pool_evidence_review_calibration_subset_TO_FILL.csv
results/shared/model_strengthening_20260519_strict_model_review_packet_300/route_pool_evidence_review_calibration_subset_TO_FILL.csv
```

Then run the validation command in the corresponding packet `README.md`.

Current environment check:

```text
DEEPSEEK_API_KEY is not set in the shell environment checked during this audit.
The runner can also read DEEPSEEK_API_KEY from .env.local or .env.
```

Latest machine-readable readiness:

```text
results/shared/model_strengthening_20260519_strict_review_readiness.json

strict_120_ready_for_real_review: true
strict_300_fallback_ready: true
external_packet_ready: true
real_review_can_run_now: false
filled_expert_csv_available: false
filled_expert_csv_rows: 0
ready_for_expert_value_training: false
blockers:
  - DEEPSEEK_API_KEY is not configured
  - no filled expert CSV rows are available
  - no merged review value pack has passed expert-training gate
```

`filled_expert_csv_rows` counts rows with at least one filled expert decision
field: `expert_route_plausible`, `expert_block_transform_correct`,
`expert_support_precedent_relevant`, `expert_cascade_coherent`, or
`expert_priority`; comments-only rows do not count.

Continuation check:

```text
PYTHONPATH=. scripts/run_strict_model_review_real.sh
```

failed before any API call with:

```text
DEEPSEEK_API_KEY is not set. Export it or place it in .env.local or .env before running real review.
```

Common shell/project config files, `/root/autodl-tmp` secret/env-like files, and
non-project `/root` env/secret-like files were checked for a loadable
`DEEPSEEK_API_KEY` without printing secret values. No usable key was found; only
`.env.example` files contain the variable name.

After non-placeholder labels exist, rebuild:

```text
route_block_review_label_pack
route/block value model
learned-vs-retrieval gate
guarded live-search comparison
```

Do not promote search-time route/block scoring until those gates pass.
