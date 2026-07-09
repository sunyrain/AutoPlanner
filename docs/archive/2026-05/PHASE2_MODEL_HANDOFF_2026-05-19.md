# Phase II Model Handoff - 2026-05-19

## Current Decision

Do not promote the learned route/block scorer yet.

ChemEnzy remains the fixed generator. The usable Phase II direction is:

```text
ChemEnzy native route pool
  -> strict runtime train-provenance CCTS/value features
  -> no-human route/block weak supervision
  -> learned scorer only if it beats retrieval/audit controls
```

## What Is Ready

Strict runtime train-provenance value pack:

```text
results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/
  route_block_value_pack.jsonl
  route_block_value_pack_report.json
  strict_route_block_value_ablation_summary.md
```

Status:

```text
rows: 32528
targets: 346
evidence provenance: verified_or_no_retrieval_features
runtime_retrieval_only: true
no_human_consensus_positive: 7765
no_human_consensus_negative: 4616
```

No-human route/block value ablations:

```text
results/shared/model_strengthening_20260519_no_human_route_block_value_models/
  no_human_route_block_value_ablation_summary.md
  no_human_route_block_value_ablation_summary.json
```

Strict model-control review worklist:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/
  strict_model_control_disagreement_review.jsonl
  strict_model_control_disagreement_review.csv
  strict_model_control_disagreement_prompts.jsonl
  dryrun_pipeline/
```

Status:

```text
selected rows: 120
targets: 47
dry-run placeholder_review: 120
usable positives / negatives: 0 / 0
```

Expanded fallback review worklist:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/
  strict_model_control_disagreement_review_300.jsonl
  strict_model_control_disagreement_review_300.csv
  strict_model_control_disagreement_prompts_300.jsonl
```

Status:

```text
selected rows: 300
targets: 78
use case: run only if the 120-row strict review does not provide enough usable positives/negatives
```

Human/external-review packets are preserved only as fallback/audit assets:

```text
results/shared/model_strengthening_20260519_strict_model_review_packet/
  route_pool_evidence_review_calibration_subset_TO_FILL.csv
  route_pool_review_calibration_packet.json
  README.md

results/shared/model_strengthening_20260519_strict_model_review_packet_300/
  route_pool_evidence_review_calibration_subset_TO_FILL.csv
  route_pool_review_calibration_packet.json
  README.md
```

These packets are not the main training path. If someone later wants an external
audit sample, fill the expert columns in `*_TO_FILL.csv`, then run the validation
command from that packet's `README.md` or the repo-side continuation wrapper:

The packet CSVs include immutable context columns `target_id`, `route_id`,
`source_value_pack`, and `value_split`. Keep them unchanged and use
`value_split` to balance filled positives/negatives across train, val, and
test.

```bash
PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
PACKET_SIZE=300 PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
```

Portable packet archive:

```text
results/shared/model_strengthening_20260519_strict_review_packets.tar.gz
results/shared/model_strengthening_20260519_strict_review_packets.tar.gz.sha256
```

The archive contains the 120-row and 300-row fillable CSV packets, READMEs,
packet metadata, and prompt/review CSV sources.
Readiness validates the archive checksum and 11 required archive members before
marking `external_packet_ready=true`.
It also includes a top-level packet README:

```text
results/shared/model_strengthening_20260519_strict_review_PACKET_README.md
```

## Key Results

Strict value-model ablation:

```text
strong_evidence_runtime_with_retrieval_no_audit:
  model MRR     1.000000
  retrieval MRR 1.000000
  conclusion    ties retrieval, not a learned win

reviewable_vs_reject_runtime_no_audit:
  model MRR     0.875724
  native MRR    0.851071
  retrieval MRR 0.790797
  audit MRR     1.000000
  conclusion    modest learned signal, still loses to audit guard
```

No-human ablation:

```text
native rank baseline: 0.707357
retrieval-only baseline: 0.761806

no_human_all_features:
  model MRR     1.000000

no_human_no_audit:
  model MRR     0.797227

no_human_no_audit_no_retrieval:
  model MRR     0.782899
  model-native  +0.075542
  model-retrieval +0.021093

no_human_no_audit_no_cascade:
  model MRR     0.816142

no_human_route_no_audit_no_retrieval:
  model MRR       0.871659
  retrieval MRR   0.790797
  model-retrieval +0.080862
```

The no-human fixed-pool strict gate now passes on the no-audit/no-retrieval
control. It still does not authorize search-time promotion.

Final-rerank replay:

```text
route_block_value_model MRR: 0.871659
native rank MRR: 0.851071
retrieval-only MRR: 0.790797
audit-guard MRR: 1.000000
model - retrieval: +0.080862
model - audit: -0.128341
top route changed vs native: 51 / 99 groups
```

Guarded live-search sweep now includes `tie_epsilon=0.08` and an additive probe:

```text
baseline:                top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 0
guarded_eps0_w0005:      top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 23
guarded_eps003_w0005:    top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 27
guarded_eps008_w0005:    top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 27
guarded_eps003_w005:     top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 27
additive_w005:           top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 72
route_block_final_rerank top exact 0.00 / top GT 0.25 / any exact 0.05 / any GT 0.35 / changed 7 / enabled 20
product_audit_final      top exact 0.00 / top GT 0.35 / any exact 0.05 / any GT 0.35 / changed 4 / enabled 20
```

This confirms no guardrail regression and a small live quality lift from the
no-label product-audit conservative final reranker. The learned no-human
route/block final reranker is also wired into live benchmark output, but on the
limit-20 smoke it changed 7/20 top routes and reduced top GT-reactant recovery
from 0.30 to 0.25, so it remains a negative learned-scorer promotion result.

Global gate:

```text
results/shared/model_strengthening_20260519_route_block_summary/
  route_block_strengthening_summary.md

decision: do_not_promote_yet
```

Runtime no-human probe:

```text
results/shared/model_strengthening_20260519_transition_hardneg_nohuman_probe/
  runtime_hardneg_nohuman_probe.json
  runtime_hardneg_nohuman_probe.md

retrieval test block MRR: 0.422418
best no-human blend test block MRR: 0.422418
delta vs retrieval: 0.000000
decision: fail
```

This closes the obvious low-cost fixes for the runtime hard-negative gate:
simple product/material sanity features and HGB runtime rankers do not add
residual signal beyond retrieval-only on the fixed candidate cache.

## Commands

Rebuild the strict no-human value pack:

```bash
PYTHONPATH=. python -m cascade_planner.eval.build_route_block_value_pack \
  --input-jsonl results/shared/model_strengthening_20260519_v4_ccts_routepool_runtime_train_provenance/route_pool_pack.jsonl \
  --output-jsonl results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl \
  --report results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack_report.json \
  --dataset route_block_value_runtime_train_provenance \
  --evidence-contract train_only_runtime_retrieval \
  --runtime-retrieval-only \
  --require-evidence-provenance
```

Train the no-human route/block value ablations:

```bash
PYTHONPATH=. scripts/train_no_human_route_block_value_models.sh
```

Refresh the global summary:

```bash
PYTHONPATH=. python -m cascade_planner.eval.summarize_route_block_strengthening \
  --route-pool-report results/shared/cascadebench_strict_20260516/route_pool_pairwise_ranker_v4_structured_train200_test100/route_pool_pairwise_ranker_report.json \
  --route-block-value-report results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack_report.json \
  --no-human-ablation-summary results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_block_value_ablation_summary.json \
  --ablation-summary results/shared/cascadebench_strict_20260516/route_pool_pairwise_ranker_ablation_v4_structured_train200_test100/ablation_summary.json \
  --bootstrap-stability results/shared/cascadebench_strict_20260516/route_pool_pairwise_ranker_ablation_v4_structured_train200_test100/bootstrap_stability.json \
  --transition-hardneg-summary results/shared/model_strengthening_20260519_transition_hardneg/transition_hardneg_summary.json \
  --guarded-search-comparison results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_weight_tie_sweep.json \
  --output-json results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.json \
  --output-md results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.md
```

Regenerate the optional strict review worklist:

```bash
PYTHONPATH=. python -m cascade_planner.eval.build_strict_model_review_worklist \
  --value-pack results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl \
  --model-pickle results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/models/reviewable_vs_reject_runtime_no_audit/route_block_value_model.pkl \
  --output-jsonl results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.jsonl \
  --output-csv results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.csv \
  --report results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review_report.json
```

Optional fallback: run real review after key is configured:

```bash
export DEEPSEEK_API_KEY=...
PYTHONPATH=. scripts/run_strict_model_review_real.sh
```

The repo includes `.env.local.example` as a placeholder template. Copy it to
`.env.local` and replace the placeholder, or add `DEEPSEEK_API_KEY=...` to
`.env`; real env files remain ignored.

Optional fallback: one-command continuation after key is configured:

```bash
PYTHONPATH=. scripts/run_strict_review_full_after_key.sh
```

To let it run the 300-row fallback automatically if the 120-row review is not
enough:

```bash
RUN_EXTENDED_IF_NOT_READY=1 PYTHONPATH=. scripts/run_strict_review_full_after_key.sh
```

The continuation wrapper exits `4` if the merged review pack is still not ready
for expert training. Set `ALLOW_NOT_READY_EXIT_ZERO=1` only for inspection-only
runs where that incomplete gate should not fail the shell command.

Runtime defaults:

```text
WORKERS=4
AUTOPLANNRELLM_DEEPSEEK_TIMEOUT_S=300
CACHE=<real_review_pipeline>/<prefix>_review_cache.jsonl
```

Default output:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline/
  strict_model_real_labels.jsonl
  strict_model_real_review_label_pack.jsonl
  strict_model_real_review_label_pack_report.json
  strict_model_real_merged_route_block_value_pack.jsonl
  strict_model_real_merged_route_block_value_pack_report.json
  strict_model_real_promotion_gate.json
```

Expanded fallback after key is configured:

```bash
PYTHONPATH=. scripts/run_strict_model_review_real_extended.sh
```

Default output:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/real_review_pipeline/
  strict_model_real_300_labels.jsonl
  strict_model_real_300_review_label_pack.jsonl
  strict_model_real_300_review_label_pack_report.json
  strict_model_real_300_merged_route_block_value_pack.jsonl
  strict_model_real_300_merged_route_block_value_pack_report.json
  strict_model_real_300_promotion_gate.json
```

Run from a filled human/external CSV without an API key:

```bash
PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
PACKET_SIZE=300 PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
```

The filled-CSV wrapper validates expert fields, rejects filled rows missing
`route_id`, builds the route/block review label pack, merges by `route_id`,
refreshes readiness, and trains only if the merge report says
`ready_for_expert_training=true`. It exits `4` if the merge gate is still not
ready, unless `ALLOW_NOT_READY_EXIT_ZERO=1` is set.

Optional fallback: train after a merged review pack passes its gate:

```bash
PYTHONPATH=. scripts/train_strict_model_review_value_models.sh
```

The training wrapper refuses to run unless the merge report says
`ready_for_expert_training=true`, including positive/negative coverage across
`train`, `val`, and `test`.

Check readiness at any time:

```bash
PYTHONPATH=. python -m cascade_planner.eval.check_strict_review_pipeline_readiness \
  --root /root/autodl-tmp/AutoPlanner \
  --output-json results/shared/model_strengthening_20260519_strict_review_readiness.json
```

Current decision from that checker for the fallback review path:

```text
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

Verification:

```text
pytest.ini scoped project tests: 472 passed, 4 warnings
```

## Blocker

The blocker is no longer missing expert labels. The blocker is that learned
route/block scoring has not yet passed search-time promotion gates:

```text
runtime hard-negative learned scorer does not clear retrieval-only control
guarded live search has no aggregate quality lift
```

Until those gates pass, do not:

```text
promote search-time route/block scoring
claim learned CCTS beats retrieval-only
train a publication-facing route-quality model from dry-run labels
```
