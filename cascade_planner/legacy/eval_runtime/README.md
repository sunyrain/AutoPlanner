# Frozen evaluation runtime

This package contains historical evaluation, training, audit, and replay code
that is excluded from the canonical V4 architecture. Direct execution requires
`AUTOPLANNER_ALLOW_LEGACY_RESEARCH=1` through `cascade_planner.legacy.guard`.

The CCTS v0-v3 lineage, route-pool ranker, block-coherence, and route-block
value/no-human review lineages are retained here only for named-report
reproduction and old route-tree checkpoint replay. New evaluation logic must
use current V4 contracts or the explicit non-authoritative
`cascade_planner.research` layer. The reservoir/controller-v2 acceptance,
comparison, publication, and distilled-controller reports are also frozen here.
CBA v0 and the expert CSV/LLM route-pool review fallback workflow are frozen in
the same namespace, with no compatibility aliases at their former eval paths.
Adjacent-step cascade-pair pack, training, replay, runtime scorers, and feature
contracts are frozen here and under
`cascade_planner.legacy.cascade_search_runtime`.
The provider-retrieval, heldout ChemEnzy route-pool, atom-map, template, and
transform-pair selector experiments are also frozen here and have no old-path
compatibility wrappers.
The action/source/transition pack, training, coverage, ranking, split, and
pipeline lineage is frozen here as well. Its checkpoint loading and feature
contracts live under `cascade_planner.legacy.cascade_search_runtime`; the
active search package retains only generic injection protocols and current
heuristic/verifier value models. The historical `LearnedCascadeValueModel`
adapter lives under `cascade_planner.legacy.cascade_search_runtime.value`.
The closed V4 product-value pack, preference-pack, training, case-study,
inventory, comparison, and publication-report scripts are frozen here. The
learned native-route reranker is frozen here as well; its checkpoint loader,
route encoders, and feature contract live under
`cascade_planner.legacy.cascade_search_runtime.v4_product_value`. The generic
route-pool contract and conservative product feasibility audit remain in the
active eval package.
The unreferenced split augmentation, strict-review readiness, dataset-release
comparison, and route-pool enrichment entrypoints are retained here for old
report reproduction only.
The external reservoir smoke builder is also frozen here; current USPTO-190
target discovery and page parsing now live in the focused active benchmark
utility `cascade_planner.eval.syntharena_uspto190`.
The cascade-oracle payload and pack builders are frozen with that reservoir
lineage. Their optional route-tree advisor runtime lives under
`cascade_planner.legacy.route_tree_runtime`; current native teacher-value rows
use `cascade_planner.eval.build_native_route_replay_pack` instead.
The route-pool selector v0, context-control, regression-case, same-pool
comparison, phase-promotion, and cascade-only guarded-rerank experiments are
frozen here because they depend on retired ranker contracts and were never
promoted to canonical V4.
The cascade subgoal discovery audit and learned scorer trainer are frozen here
as well. Runtime evidence-item, fragment, fingerprint, and feature contracts
remain active in `cascade_planner.cascade_search.subgoal_evidence_contract`.
The closed CascadeBench Phase-II direction, block-readiness, fragment-pack,
fragment-scorer, transition-block evaluation, offline-rerank, decision-summary,
and no-go verification tools are frozen here too. They reproduce historical
decision artifacts only and must not be treated as promoted V4 search components.
The obsolete AiZ MCTS `multistep_solvebench` and `benchmark_v2_100` runner/
summary entrypoints are frozen here as well. Their old `cascade_planner.multistep`
engine is no longer part of the repository; the current benchmark data file is
still consumed by active V4 components and is intentionally retained.
The per-EC1 condition diagnostic, EnzExpand ablation, and chemical-template
pair-ranker trainer are archived because their old `training`/`expand`
dependencies were removed. The active pair-ranker loader and artifact contract
remain under `cascade_planner.cascadeboard`.
The older results-v2/K2/full100 audit line is frozen here as well: external
smoke aggregation, student-route comparison, condition and skeleton audits,
benchmark-overlap checks, v3 gold-smoke/locked-validation builders, old
CascadeProgramSearch and ChemEnzy comparisons, AiZynthFinder and USPTO-50K
baselines, the old Syntheseus cascade-step comparison, generator-recall
diagnostics, candidate-miss and stock-failure audits, the CascadeBoard CC-A*
depth report, the aggregate legacy baseline summary and manifest executor, and
condition-rich/UniProt summaries.
The original v2-100 benchmark freezer is archived here because its
`cascade_dataset_v2.strict.json` source is no longer a current repository
input. The checked-in `data/benchmark_v2_100.json` and CSV remain active
benchmark assets.
The V4 blind panel, current benchmark data, and active training-pack contracts
remain outside this archive.
The old CascadeBoard report card and its condition-diagnosis/hybrid-audit
producers are frozen here too; the active `cascadeboard` package no longer
publishes the `results/v2` report surface.
The v3 learned route scorer is frozen here with the integrated benchmark that
consumes it. It has no current runtime caller or checked-in model artifact;
current route ranking uses the focused candidate, skeleton, proposal, and
route-tree value contracts instead.
The standalone CascadeBoard CLI and its original masked-route/edit-action
training pipeline are frozen here as well. The CLI was never a registered
package entrypoint and defaulted to the cache-based legacy planner. Canonical
execution remains `python -m cascade_planner`; active model loaders stay in the
runtime modules that still consume their checkpoint contracts.
The cache-based particle planner and the unreferenced v20 masked-inpainting
planner are now frozen with those commands. Their shared tensor/model schema
remains active in `cascade_planner.cascadeboard.route_encoder` because the
current skeleton planner still consumes that focused contract.
The original CascadeBoard constraint, counterfactual, policy, ablation,
integrated, strict-cache, data-audit, preference, and candidate-supervision
commands are frozen here as one benchmark lineage. The candidate hypergraph,
strict cached adapter, lazy expansion helper, and old route-preference scorer
are frozen with the planner that consumed them. The active candidate-cache
module now owns only provider-neutral normalization and merge contracts.
The old `results_dir()`/`shared_dir()` compatibility helper now lives at
`cascade_planner.legacy.paths`; the former mainline `cascade_planner.paths`
module is deleted.
The model-specific EnzExpand candidate-cache builder and dual-tower cache
annotator are frozen here as well. The active
`cascade_planner.cascadeboard.candidate_cache` now owns only the small V4
normalization/merge/summary contract used by current supervision and benchmark
code.
Historical condition-rich, locked-validation, candidate-miss, and pairwise
stress benchmark JSON files are stored under
`archive/data/legacy_eval_202605`; active benchmark inputs remain under `data/`.

Frozen CascadeProgramSearch model and ChemEnzy policy switches are owned by
`run_cascade_search_benchmark.py` in this package. It extends the active
benchmark parser, loads archived value/transition/action/pair/reranker
artifacts, and injects them through the mainline's provider-neutral runtime
override contract. `run_v4_full_training_pipeline.py` now emits this legacy
entrypoint directly; it no longer routes frozen options through the active
`cascade_planner.eval` CLI.
