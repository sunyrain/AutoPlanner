# Frozen V3 compatibility

This namespace is the only supported discovery surface for pre-V4 application,
orchestration, provider, and controller-harness APIs. Package-root
aliases have been deleted. The remaining implementations stay here or in their
historical modules until the frozen saved-run corpus has canonical V4 replay
receipts.

Canonical V4 code must not import `cascade_planner.legacy`. Compatibility
exports are deprecated, non-authoritative, and excluded from the default public
`__all__` surfaces.

The combined compatibility UI was retired on 2026-08-29. Historical runs are
reviewed through the canonical V4 workspace and Workbench; this namespace no
longer contains an executable Web application.

The frozen recursive campaign and its scheduling/ledger/portfolio/acceptance
runtime now live under `legacy/orchestration_runtime/` and
`legacy/application_runtime/`. Their former `application.*` and
`orchestration.codex_retrosynthesis` module paths have been deleted.

The blackboard controller, action planners, event state, local tool dispatcher,
legacy runner, and RouteForest compiler live under `legacy/harness_runtime/`.
Only the renderer/layout modules shared by the V4 Workbench remain in
`cascade_planner.harness`.

V3 operator, replay, golden, and RouteForest commands remain under
`scripts/legacy/` only where saved-run replay still requires them. Former Web
application and launcher paths are deleted rather than wrapped.

Blackboard graph reconstruction, external-edge receipts, and the admitted
hyperedge journal live under `legacy/routes_runtime/` and
`legacy/orchestration_runtime/`. The canonical `cascade_planner.routes` package
no longer imports or exports the blackboard rebuild adapter.

The frozen CCTS v0-v3 training, audit, replay, and report lineage now lives
under `legacy/eval_runtime/`. Its two optional route-tree checkpoint scorers
live under `legacy/route_tree_runtime/`; their former `eval.*` and
`route_tree.ccts_*` paths are deleted, and runtime activation requires the
explicit legacy-research guard.

The old route-pool ranker/LambdaRank and block-coherence/block-hard training,
replay, and audit modules are isolated in the same evaluation runtime. Active
research selectors may read their helpers explicitly, but those helpers remain
frozen and cannot become V4 runtime authority.

Route-block value packs, strict review worklists, no-human probes, and
strengthening summaries are also legacy-only and require the same explicit
research opt-in for direct execution.

Reservoir/controller-v2 calibration, comparison, distillation, acceptance,
publication, and statistical reports are frozen in the same runtime. The
external target-cache parser remains outside legacy while current benchmark
preparation still consumes it.

CBA v0 training/audit and the expert CSV/LLM route-pool review fallback
workflow are isolated here as well. Former eval module paths are deleted and
all direct workflow entrypoints require explicit research opt-in.

Adjacent-step cascade-pair pack, training, replay, runtime scorers, and feature
contracts are legacy-only under `cascade_search_runtime` and `eval_runtime`.
The current search controller retains only the provider-neutral pair-scorer
injection protocol.

The unused CascadeBoard chemical-anchor and semisynthesis stock wrappers live
under `cascadeboard_runtime`. Current execution uses the source-supported rescue
providers directly plus the common/vendor/ZINC stock chain; no old-path aliases
are retained.

The V3 saved-run audit regressions live under `scripts/legacy/` and
`tests/legacy/`. They remain part of release compatibility coverage but are no
longer presented as current V4 architecture modules.
