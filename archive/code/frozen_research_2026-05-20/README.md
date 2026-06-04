# Frozen Research Archive - 2026-05-20

This is a guarded archive manifest for old research lines that should not
influence the current AutoPlanner mainline.

The code is intentionally left at its historical import paths when tests or
report-reproduction helpers still import it. Direct CLI execution is fenced by
`cascade_planner.legacy_guard` and requires:

```bash
AUTOPLANNER_ALLOW_LEGACY_RESEARCH=1
```

## Frozen Lines

- CCTS v0/v1/v2/v3 transition/runtimes:
  - `cascade_planner/eval/train_ccts_v*_*.py`
  - `cascade_planner/eval/build_ccts_v3_*.py`
  - `cascade_planner/eval/replay_ccts_*.py`
  - `cascade_planner/eval/audit_ccts_*.py`
  - `cascade_planner/eval/summarize_ccts_v0_report.py`
- Route-pool LambdaRank/ranker experiments:
  - `cascade_planner/eval/build_route_pool_ranker_pack.py`
  - `cascade_planner/eval/train_route_pool_ranker.py`
  - `cascade_planner/eval/train_route_pool_lambdarank.py`
  - `cascade_planner/eval/replay_route_pool_pairwise_ranker.py`
- Adjacent-step cascade pair scorer:
  - `cascade_planner/eval/train_cascade_pair_scorer.py`
  - `cascade_planner/eval/replay_cascade_pair_scorer.py`
- Block-coherence / block-hard experiments:
  - `cascade_planner/eval/build_cascade_block_coherence_pack.py`
  - `cascade_planner/eval/train_cascade_block_coherence.py`
  - `cascade_planner/eval/build_cascade_block_hard_pack.py`
  - `cascade_planner/eval/replay_block_coherence_on_route_pool.py`
- V4 product-value / action-source / provider-retrieval lineage:
  - `cascade_planner/eval/build_v4_training_splits.py`
  - `cascade_planner/eval/build_v4_cascade_*`
  - `cascade_planner/eval/train_v4_*`
  - `cascade_planner/eval/rerank_native_routes_with_v4_value.py`
  - `cascade_planner/eval/audit_*provider*.py`
  - `cascade_planner/eval/build_provider_injected_route_sketches.py`
  - `cascade_planner/eval/build_cascade_action_value_pack.py`
  - `cascade_planner/eval/train_cascade_action_value.py`
  - `cascade_planner/eval/build_routepool_preference_pack.py`
  - `cascade_planner/eval/run_stage3_baseline_value_training.py`
  - `cascade_planner/eval/run_v4_full_training_pipeline.py`
  - trace/action-value options in
    `cascade_planner/eval/run_v4_heldout_chem_enzy_pool.py`
- CBA v0 / reservoir / controller-v2 lineage:
  - `cascade_planner/eval/*cba_v0*.py`
  - `cascade_planner/eval/*reservoir*.py`
  - `cascade_planner/eval/controller_v2_reports.py`
- Expert/LLM review fallback:
  - `cascade_planner/eval/*route_pool_evidence_review*.py`
  - `cascade_planner/eval/build_strict_model_review_worklist.py`
  - `cascade_planner/eval/check_strict_review_pipeline_readiness.py`
  - `cascade_planner/eval/build_route_block_review_label_pack.py`
  - `cascade_planner/eval/build_route_block_value_pack.py`
  - `cascade_planner/eval/train_route_block_value_model.py`
  - `cascade_planner/eval/replay_route_block_value_model.py`
  - `cascade_planner/eval/merge_route_block_review_labels.py`
  - `cascade_planner/eval/rerank_runtime_ccts_with_product_audit.py`
  - `scripts/run_strict_model_review_real*.sh`
  - `scripts/run_strict_review_*.sh`
  - `scripts/train_no_human_route_block_value_models.sh`
  - `scripts/train_strict_model_review_value_models.sh`

## Mainline Boundary

The current runtime path is:

```text
ChemEnzy native proposal/search
  -> conservative product/material sanity audit
  -> rule cascade verifier gate when explicitly enabled
  -> learned cascade verifier annotation when explicitly enabled
  -> route SVG/static reporting when requested
```

`scripts/run_chem_enzy_plan_for_web.py` now keeps legacy cascade action/source
hooks off by default. They can only be enabled with both a request flag and the
legacy opt-in environment variable above.

The current training path is chosen-only supervised ChemEnzy ONMT adapter
experimentation from verifier preferences. It deliberately excludes rejected
cascades from supervised positives. Direct DPO/LoRA, old CCTS rankers, route
pool rankers, route/block review fallbacks, and strict expert/LLM review lines
are outside the mainline unless a future decision document explicitly
re-promotes them.

## Why Guard Instead Of Delete

Some frozen scripts still provide helper functions used by old report
reproduction tests. Hard-moving every module would create import churn without
improving the current verifier-first plan. The guard prevents accidental direct
use while preserving reproducibility.
