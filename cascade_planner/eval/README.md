# `cascade_planner.eval`

This package contains experiment, audit, training, and batch-report scripts.
Most files here are not imported by the Web runtime directly, but many are
needed to reproduce named reports or are imported by research modules/tests.

## Current Runtime-Adjacent Scripts

- `product_route_feasibility_audit.py`
- `build_route_pool_selector_pack.py`
- `train_route_selector_v0.py`
- `rerank_native_routes_with_product_audit.py`
- `rerank_native_routes_with_v4_value.py`
- `build_routepool_preference_pack.py`

These support product-audit filtering, route-pool triage, and batch reranking.

## Active Research Scripts

Keep these while CCTS / subgoal / cascade-search experiments are still under
review:

- `train_cascade_action_value.py`
- `audit_cascade_subgoal_discovery.py`
- `train_cascade_subgoal_scorer.py`
- `rerank_runtime_ccts_with_product_audit.py`
- `rerank_cascade_only_features_with_product_audit.py`
- `audit_routepool_context_controls.py`
- `audit_selector_regression_cases.py`
- `gate_phase_selector_promotion.py`

They are research-only and should not be presented as promoted runtime unless
a current report explicitly gates them.

## Archived Standalone Experiments

The following unreferenced standalone subgoal runners were moved out of this
package on 2026-05-19:

- `archive/code/research_experiments_2026-05-19/subgoal/run_subgoal_sidecar_same_pool_ablation.py`
- `archive/code/research_experiments_2026-05-19/subgoal/run_subgoal_stitching_smoke.py`

They had no runtime imports, no focused tests, and no current report index
claim. Keep them as reference code only.

## Cleanup Rule

Before archiving a script in this package:

1. Search for imports from `cascade_planner/`, `scripts/`, and `tests/`.
2. Check whether it reproduces a named report in `docs/` or `results/shared/`.
3. Move generated outputs to `archive/`, but keep scripts until their report
   path is superseded or intentionally retired.
