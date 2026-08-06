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

## Retained V4 CLI Contracts

The following entrypoints remain active even when they are invoked manually
instead of imported by the runtime. They own current V4 artifact contracts:

- Dataset and replay builders: `build_cascade_program_pack.py`,
  `build_native_route_replay_pack.py`,
  `build_proposal_recall_pack.py`, `build_training_pack.py`,
  `build_vnext_pack.py`, `build_external_step_pairs.py`, and
  `build_external_candidate_pools.py`.
- Skeleton/template supervision: `build_skeleton_prior_splits.py`,
  `build_skeleton_hard_negative_pack.py`,
  `build_template_outcome_supervision_pack.py`, and
  `sample_template_outcome_review_batch.py`.
- Artifact trainers and calibration: `train_candidate_ranker_from_pack.py`,
  `train_cascade_quality_from_pack.py`,
  `train_failure_classifier_from_pack.py`, `train_skeleton_reranker.py`,
  `train_value_function.py`, `train_chemical_template_preselector.py`,
  `train_proposal_rankers.py`, `train_vnext_from_pack.py`, and
  `calibrate_route_tree_value.py`.
- Current benchmark and safety gates: `run_cascade_search_benchmark.py`,
  `run_live_benchmark_parallel.py`,
  `evaluate_paroutes_topk_proxy.py`, `select_safe_union_routes.py`,
  `v4_blind_release_gate.py`, and the USPTO-190 utilities.

`run_cascade_search_benchmark.py` now exposes only the current benchmark,
proposal, verifier, trace, and merge contracts. Frozen learned cascade-value,
transition/action-value, pair-scorer, route-block reranker, and ChemEnzy
cascade-policy options are available only through
`cascade_planner.legacy.eval_runtime.run_cascade_search_benchmark` with the
explicit legacy-research opt-in. The active runner accepts a narrow object
injection contract so the archive can reuse benchmark execution without making
the mainline load or parse historical model adapters.

The active `cascade_planner.cascadeboard.candidate_cache` module is similarly
limited to current cache normalization and merge helpers. EnzExpand cache
construction and dual-tower annotation are historical model adapters under
`cascade_planner.legacy.eval_runtime`.
Cache merging is provider-neutral: candidates are ordered by score and stable
identity fields rather than giving a retired source an implicit priority.

The v3 CascadeBoard learned route scorer is also archived under
`cascade_planner.legacy.eval_runtime`; its only consumer is the frozen
integrated benchmark. It is not part of current route ranking.
The unregistered standalone CascadeBoard CLI, its old transformer trainer, and
its masked/corrupted route-data generator are archived with the same lineage.
The canonical CLI is `python -m cascade_planner`; current focused trainers are
listed above.
The old cache-based particle planner and standalone v20 inpainting planner are
archived with them. `cascadeboard.route_encoder` remains active as the shared
model/tensor contract used by the current skeleton planner.
Their candidate hypergraph, strict cached-graph adapter, lazy expansion helper,
and unused route-preference scorer are archived too. The active
`cascadeboard.candidate_cache` module retains only provider-neutral cache
normalization, merge, and summary behavior.

These files may have no Python caller because they are reproducible operator
commands. Their output schemas are consumed by `cascade_search`, `route_tree`,
`cascadeboard`, or the V4 blind-panel workflow. They must not be moved to the
legacy namespace merely because a text-reference scan finds zero callers.

The historical cascade-oracle payload and pack builders are an exception: they
were coupled to the retired reservoir/ChemEnzy teacher experiment, have no
current artifacts or consumers, and now live under
`cascade_planner.legacy.eval_runtime`. Current native teacher-value rows are
owned by `build_native_route_replay_pack.py`.

## Classic Multistep Blind Panel

`build_classic_multistep_benchmark.py` freezes a deterministic 20-target
PaRoutes set-n1/set-n5 panel. The checked-in manifest contains opaque labels,
canonical SMILES, generic acceptance criteria, and budgets only. Reference
routes and sampling strata are evaluator-only artifacts and must not be passed
to a planner.

Run the full target-only V4 panel with a host-compatible ChemEnzy prefix:

```powershell
python scripts/run_v4_blind_panel.py `
  --manifest benchmarks/paroutes_v4_multistep20.json `
  --output-root results/shared/paroutes_v4_multistep20_v4_full `
  --workers 2 `
  --execution-profile standard `
  --chemenzy-env-prefix D:\conda\envs\py312
```

The benchmark reports runtime completion, route retention, host acceptance,
split-specific PaRoutes stock closure, evidence/condition status, and reference
recovery independently. A low-confidence or partially validated route remains
visible with warnings; it is not relabeled as accepted, but it is also not
erased as though no route were found. Route length is descriptive and a shorter
valid closed route is allowed.

The current verifier-first / chosen-only adapter mainline mostly lives outside
this package:

- `cascade_planner/cascade_verifier/`
- `scripts/build_cascade_perturbation_pack.py`
- `scripts/train_cascade_verifier_from_pack.py`
- `scripts/build_cascade_verifier_preference_pack.py`
- `scripts/build_supervised_seed_pack_from_verifier_preferences.py`
- `scripts/build_chem_enzy_cascade_onmt_corpus.py`
- `scripts/run_chem_enzy_onmt_adapter_experiment.py`

## Research Boundary

Unpromoted research scripts no longer live in this package. Current,
non-authoritative workers belong under `cascade_planner.research`; frozen
report-reproduction and trainer lineages belong under
`cascade_planner.legacy.eval_runtime`. Reusable search contracts remain in
focused `cascade_planner.cascade_search` modules.

## Archived Standalone Experiments

The following unreferenced standalone subgoal runners were moved out of this
package on 2026-05-19:

- `archive/code/research_experiments_2026-05-19/subgoal/run_subgoal_sidecar_same_pool_ablation.py`
- `archive/code/research_experiments_2026-05-19/subgoal/run_subgoal_stitching_smoke.py`

They had no runtime imports, no focused tests, and no current report index
claim. Keep them as reference code only.

## Guarded Archive

The 2026-05-20 cleanup moved the following lines out of the mainline contract:

- CCTS v0/v1/v2/v3 transition/runtime rankers, replays, audits, and report
  summarizers now live under `cascade_planner.legacy.eval_runtime`.
- The two optional CCTS route-tree checkpoint adapters now live under
  `cascade_planner.legacy.route_tree_runtime` and require the legacy-research
  guard before their environment switches are honored.
- Route-pool ranker/LambdaRank and block-coherence/block-hard pack, training,
  replay, and audit modules now live under
  `cascade_planner.legacy.eval_runtime`.
- Route-block value, strict disagreement review, no-human probe, and
  strengthening-summary modules now live under the same legacy runtime.
- Reservoir/controller-v2 comparison, distillation, acceptance, publication,
  calibration, and external-smoke modules now live under the same legacy
  runtime. Current USPTO-190 preparation uses the independent
  `syntharena_uspto190.py` utility.
- CBA v0 and the expert CSV/LLM route-pool review fallback workflow now live
  under the same legacy runtime.
- Adjacent-step cascade-pair pack, training, replay, rule/learned scorers, and
  feature contracts now live under
  `cascade_planner.legacy.cascade_search_runtime` and
  `cascade_planner.legacy.eval_runtime`.
- Route-pool selector, context-control, comparison, and phase-selector
  experiments now live under the same legacy runtime.
- V4 provider-retrieval, heldout route-pool, atom-map, template, and
  transform-pair selector lineage now lives under the same legacy runtime.
- The older results-v2/K2/full100 audit and external-baseline entrypoints are
  also frozen under `cascade_planner.legacy.eval_runtime`; active V4 blind-panel
  and training-pack contracts remain in the current package.
  This includes the historical Syntheseus cascade-step and USPTO-50K K2
  evaluators, candidate/stock-failure diagnostics, and the CascadeBoard CC-A*
  depth report. The aggregate legacy baseline summary and old pipeline-manifest
  executor are archived there too; current USPTO-190 utilities remain active
  here.

The action/source/transition pack, training lineage, checkpoint loaders, and
feature contracts now live under the explicit legacy runtime. The mainline
retains only provider-neutral injection protocols and the current subgoal-hint
action scorer. Closed product-value pack, training, publication-report,
learned reranker, checkpoint loader, and feature contract now live under the
legacy runtime. Current reusable route-pool parsing and product-audit reporting
functions are owned by `native_route_pool_contract.py`. Direct CLI execution
is fenced by
`cascade_planner.legacy.guard` and requires
`AUTOPLANNER_ALLOW_LEGACY_RESEARCH=1`; migrated lines do not retain old-path
wrappers.

No guarded historical module remains in this package. New report-reproduction
entrypoints belong under `cascade_planner.legacy.eval_runtime`, while reusable
current contracts must live in focused active modules.

## Cleanup Rule

Before archiving a script in this package:

1. Search for imports from `cascade_planner/`, `scripts/`, and `tests/`.
2. Check whether it reproduces a named report in `docs/` or `results/shared/`.
3. Move generated outputs to `archive/`, but keep scripts until their report
   path is superseded or intentionally retired.
4. Canonical runtime packages must not import this package. Move reusable
   checkpoint, feature, parsing, or selection contracts to the owning runtime
   package before archiving or retaining a trainer.
