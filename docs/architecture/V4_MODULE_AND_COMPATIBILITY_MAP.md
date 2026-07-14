# V4 module and compatibility map

## Authoritative path

```text
RetrosynthesisCampaignService
  -> RunKernel (events, revision, budgets, recovery)
  -> GlobalCampaignDirector (bounded global hypotheses only)
  -> WorkerRuntime (materialization, evidence, validation, stock facts)
  -> CanonicalHypergraphStore (single scientific graph)
  -> DeficitFrontier (single work projection)
  -> ProofPolicy + route_variants + portfolio_selection
  -> proof_portfolio (publication and explicit closeout)
  -> route_workbench (bounded read model and revision deltas)
```

Blackboards, legacy campaign JSON, legacy frontier queues, and `RouteForest`
are read/write-compatible historical implementations; none is an authority for
a V4 run. New features must enter through the path above.

Working legacy capabilities are migrated according to
[`BLACKBOARD_CAPABILITY_MIGRATION.md`](BLACKBOARD_CAPABILITY_MIGRATION.md).
Freezing the legacy controller does not authorize deleting or duplicating its
validated ChemEnzy, literature, PDF, vision, evidence, stock, or self-evolution
capabilities before their V4 acceptance tests pass.

## Ownership split

| Concern | V4 owner | Frozen compatibility owner |
|---|---|---|
| Operational state, budgets, recovery | `application.run_kernel` | blackboard/controller counters |
| Global multi-route reasoning | `orchestration.global_campaign_director` | action planners and recursive Codex campaign |
| Scheduling/work state | `application.deficit_frontier` + `frontier_runtime` | `frontier_scheduler`, `route_deficit_queue` |
| Scientific identities | `application.canonical_identity` | route/string signatures in several harness modules |
| Graph storage and ingestion | `application.canonical_hypergraph` | blackboard plus route-consensus graph |
| Route traversal | `application.route_variants` | `RouteForest` compiler traversal |
| Ranking/diversity | `application.portfolio_selection` | legacy route portfolio and display ranking |
| Proof and stock closeout | `application.proof_policy` + `proof_portfolio` | legacy acceptance and parent proof |
| Tool registration/execution | `harness.tool_registry` + `tool_execution_policy` | formerly embedded dispatch in `tools.py` |
| Primary patent text evidence | `harness.source_html` + `interfaces.patent_html_evidence` | ad hoc HTML companions |
| Image-only source recovery | `harness.source_ocr` + deterministic literature registry | legacy visual-chain exact claims |
| Optional page-vision hypotheses | `interfaces.visual_evidence` + `RunKernel` budget | legacy unmetered visual retries |
| Presentation projection | `application.route_workbench` + `harness.v4_route_workbench` | `RouteForest` HTML/JSON |

The old `RouteForest` file remains frozen for historical run display. New V4
runs use the bounded workbench projection and a small display-only adapter; they
do not invoke the old compiler. Its V4
responsibilities have been split among artifact storage, canonical identity,
route traversal, portfolio ranking, and proof projection, so it is no longer a
core orchestration dependency.

## Schema ownership

| Schema family | Authority |
|---|---|
| `autoplanner_run_*`, `autoplanner_deficit.v1` | `RunKernel` |
| `global_campaign_plan.v1` and director outcome/config | global director |
| `autoplanner_worker_*` | `WorkerRuntime` |
| `canonical_retrosynthesis_hypergraph.v1` | canonical hypergraph store |
| `deficit_frontier.v1`, `deficit_frontier_item.v1` | single frontier compiler |
| proof policy, edge/leaf proof stitches | `ProofPolicy` |
| proof route, replacement module, portfolio, closeout | proof portfolio modules |
| `retrosynthesis_route_workbench.v1` and its delta | route workbench projection |
| V3 campaign/queue/route-forest schemas | compatibility only; no V4 writes |

Every compatibility path is registered in
`application.compatibility_inventory`. Each entry has a telemetry source,
replacement, owner, and removal milestone. Runtime use is appended to
`.autoplanner/compatibility_usage.jsonl` and carries no scientific authority.

## Dependency and size policy

- `application` may depend on domain/provider/runtime contracts, never web or
  orchestration.
- V4 orchestration may depend on application services; it may not import the
  legacy blackboard controller, recursive campaign, old queue, or RouteForest.
- Web/CLI are adapters and must not implement chemistry, budgets, or closeout.
- New focused modules target 500 lines; scientific compilers have an 800-line
  ceiling. Event-sourced state machines and the canonical ingestion module are
  temporarily grandfathered up to 1,600 lines and may only shrink.
- Frozen compatibility files may exceed the ceiling but may receive only
  security, correctness, telemetry, or deletion changes—no new product logic.
- New functions target cyclomatic complexity below 15; exceptions require a
  characterization test and an explicit follow-up split.

Local architecture tests enforce forbidden dependency edges, focused-module
line budgets, compatibility metadata completeness, registry drift, and
telemetry digest integrity.
