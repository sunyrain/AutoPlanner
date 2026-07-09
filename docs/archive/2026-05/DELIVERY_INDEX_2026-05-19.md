# Delivery Index - 2026-05-19

## What Is Ready

- [Current State](CURRENT_STATE_2026-05-19.md)
- [Phase 2 Handoff](PHASE2_MODEL_HANDOFF_2026-05-19.md)
- [Next Model Strengthening Plan](NEXT_MODEL_STRENGTHENING_PLAN_2026-05-19.md)
- [Completion Audit](MODEL_STRENGTHENING_COMPLETION_AUDIT_2026-05-19.md)
- Machine-readable readiness:
  - `results/shared/model_strengthening_20260519_strict_review_readiness.json`
- No-human route/block value results:
  - `results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_block_value_ablation_summary.md`
  - `results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_block_value_ablation_summary.json`
- Global route/block summary:
  - `results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.md`
  - `results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.json`
- Portable external-review packet:
  - `results/shared/model_strengthening_20260519_strict_review_packets.tar.gz`
  - `results/shared/model_strengthening_20260519_strict_review_packets.tar.gz.sha256`
  - `results/shared/model_strengthening_20260519_strict_review_PACKET_README.md`
- Safe key template:
  - `.env.local.example`
- Test collection config:
  - `pytest.ini`

## Core Scripts

- `scripts/train_no_human_route_block_value_models.sh`
- `scripts/run_strict_model_review_real.sh`
- `scripts/run_strict_model_review_real_extended.sh`
- `scripts/run_strict_review_full_after_key.sh`
- `scripts/run_strict_review_from_filled_csv.sh`
- `scripts/train_strict_model_review_value_models.sh`
- `scripts/run_route_block_review_expansion_real.sh`

## Pipeline Components

- `cascade_planner/eval/build_route_block_review_label_pack.py`
- `cascade_planner/eval/build_route_pool_evidence_review_prompts.py`
- `cascade_planner/eval/ingest_route_pool_evidence_review_results.py`
- `cascade_planner/eval/merge_route_block_review_labels.py`
- `cascade_planner/eval/check_strict_review_pipeline_readiness.py`

## Artifact Roots

No-human route/block value model:

```text
results/shared/model_strengthening_20260519_no_human_route_block_value_models/
```

Route/block global summary:

```text
results/shared/model_strengthening_20260519_route_block_summary/
```

Strict 120-row review:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/
```

Strict 300-row fallback:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/
```

External packet directories:

```text
results/shared/model_strengthening_20260519_strict_model_review_packet/
results/shared/model_strengthening_20260519_strict_model_review_packet_300/
```

Mock real-review smoke:

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/mock_real_pipeline/
```

## Blockers

The real blockers remain:

```text
runtime hard-negative learned scorer does not clear retrieval-only control
```

The new live route/block final-rerank smoke is included in the guarded sweep
artifact; it changed 7/20 top routes but reduced top GT-reactant recovery from
0.30 to 0.25, so it is not a promotion path yet.

The new no-label product-audit conservative final-rerank smoke is also included;
it changed 4/20 top routes and improved top GT-reactant recovery from 0.30 to
0.35. This supports no-human guard/rerank work, but it does not promote the
learned route/block scorer.

Expert-review packets remain available as a fallback/audit path, but missing
expert CSV rows are not a blocker for the main no-human training path.
