# `cascade_planner.eval`

This package contains experiment, audit, training, and batch-report scripts.
Most files here are not imported by the Web runtime directly, but many are
needed to reproduce named reports or are imported by research modules/tests.

## Current Runtime-Adjacent Scripts

- `product_route_feasibility_audit.py`
- `rerank_native_routes_with_product_audit.py`

These support conservative product/material sanity checks and batch audit
refreshes. They do not promote a learned ranker and should not load retired v4
learned-value artifacts by default.

The current verifier-first / chosen-only adapter mainline mostly lives outside
this package:

- `cascade_planner/cascade_verifier/`
- `scripts/build_cascade_perturbation_pack.py`
- `scripts/train_cascade_verifier_from_pack.py`
- `scripts/build_cascade_verifier_preference_pack.py`
- `scripts/build_supervised_seed_pack_from_verifier_preferences.py`
- `scripts/build_chem_enzy_cascade_onmt_corpus.py`
- `scripts/run_chem_enzy_onmt_adapter_experiment.py`

## Active Research Scripts

These are not promoted runtime or training targets. Keep them only for local
ablation/review while the verifier-first mainline is developed:

- `audit_cascade_subgoal_discovery.py`
- `train_cascade_subgoal_scorer.py`
- `rerank_cascade_only_features_with_product_audit.py`
- `audit_routepool_context_controls.py`
- `audit_selector_regression_cases.py`
- `gate_phase_selector_promotion.py`
- `build_route_pool_selector_pack.py`
- `train_route_selector_v0.py`

They are research-only and should not be presented as promoted runtime or used
as the default next-stage training path unless a current decision document
explicitly re-promotes them.

## Archived Standalone Experiments

The following unreferenced standalone subgoal runners were moved out of this
package on 2026-05-19:

- `archive/code/research_experiments_2026-05-19/subgoal/run_subgoal_sidecar_same_pool_ablation.py`
- `archive/code/research_experiments_2026-05-19/subgoal/run_subgoal_stitching_smoke.py`

They had no runtime imports, no focused tests, and no current report index
claim. Keep them as reference code only.

## Guarded Archive

The 2026-05-20 cleanup moved the following lines out of the mainline contract:

- CCTS v0/v1/v2/v3 transition/runtime rankers and replays.
- CCTS label/rank audits and old report summarizers.
- Route-pool LambdaRank / old route-pool ranker.
- Route-pool selector / phase-selector experiments unless explicitly
  re-promoted by a current decision document.
- Route-block value / no-human review value-model lineage.
- Adjacent-step `cascade_pair_scorer` experiments.
- Block-coherence / block-hard experiments.
- V4 product-value / action-source / provider-retrieval lineage.
- CBA v0 / reservoir / controller-v2 lineage.
- Expert CSV / LLM review fallback workflows.

Their code remains at historical import paths where old tests or report
reproduction helpers still import it, but direct CLI execution is fenced by
`cascade_planner.legacy_guard` and requires
`AUTOPLANNER_ALLOW_LEGACY_RESEARCH=1`.

## Cleanup Rule

Before archiving a script in this package:

1. Search for imports from `cascade_planner/`, `scripts/`, and `tests/`.
2. Check whether it reproduces a named report in `docs/` or `results/shared/`.
3. Move generated outputs to `archive/`, but keep scripts until their report
   path is superseded or intentionally retired.
