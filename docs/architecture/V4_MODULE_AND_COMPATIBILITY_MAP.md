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
  -> transformation_programs (optional read-only edge→program projection + oracle)
  -> transformation_program_store (explicit append-only shadow admission + replay)
  -> route_program_dual_read (Workbench route:* edge/program equivalence oracle)
  -> program_migration (read-only cross-run migration classification)
  -> candidate_route_observations -> candidate_programs (open-route shadow projection)
  -> candidate_migration (cross-snapshot read-only audit)
  -> candidate_innovation_screen -> route_innovation_discovery (proposal-only enzyme scan)
  -> biocatalytic_programs -> biocatalysis_validation_frontier (drafts + assay plans)
  -> biocatalytic_program_store -> program_innovation_runtime (durable shadow admission + replay)
  -> mechanism_programs -> mechanism_validation_frontier / experiment_feedback
       (one-hop restitch + exact-boundary read-only validation)
  -> route_execution_capabilities -> execution_programs (whole-cell/hybrid read-only restitch)
  -> experimental_claims -> capability_applicability_calibration
       (three-domain exact-boundary observations + read-only dirty hints)
  -> experimental_claim_store (explicit append-only admission + full source reprojection)
  -> experimental_work_frontier -> experiment_execution_contracts / experiment_execution_results
       (canonical-frontier-bound read-only tasks + executor-neutral result audit)
  -> program_route_candidates -> program_route_optimizer (read-only multi-profile Pareto layers)
  -> reported_program_route_candidates (digest-bound full Candidate Program route adapter)
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
| Web service surface | `web.v4_app` | `web.app` combined legacy UI (`--surface combined`) |
| Program migration shadow | `application.transformation_programs` + `transformation_program_store` + `interfaces.program_migration` | no legacy owner; read-only by default, explicit append-only admission only |
| Route/Program UI dual read | `application.route_program_dual_read` + `interfaces.campaign_programs` | overlays secondary Program ids on the current Workbench revision; cannot change route proof, conditions, steps, or acceptance |
| Open reported-route shadow | `application.candidate_route_observations` + `candidate_programs` | legacy Workbench is digest-bound input only; candidate Programs cannot enter canonical proof/closure |
| Candidate migration inventory | `interfaces.candidate_migration` | content-deduplicated Workbench audit; historical acceptance remains diagnostic only |
| Candidate enzyme opportunity scan | `application.candidate_innovation_screen` | temporary screening graph only; matches and zero-match controls grant no scientific write authority |
| Biocatalytic Program draft | `application.biocatalytic_programs` + `biocatalytic_program_contracts` + `orchestration.program_innovation_runtime` | replaces a contiguous baseline Program span only in a read-only candidate; fallback remains explicit and validation cannot close the route |
| Biocatalytic validation frontier | `application.biocatalysis_validation_frontier` | converts every unvalidated exact boundary into a deterministic assay plan; plans grant no validation, admission, proof, or completion |
| Biocatalytic Program shadow store | `application.biocatalytic_program_store*` + `interfaces.campaign_program_innovation_store` | separately persists specialized-validated supersteps with six CAS inputs; append-only, explicit-gate, fallback-preserving, and non-authoritative |
| Program candidate contract + optimizer | `application.program_route_candidate_contracts` + `program_route_candidate_factory` + `program_route_candidates` + `program_route_optimizer` | normalizes baseline, biocatalytic, whole-cell, hybrid, reported full-route and fully restitched mechanism alternatives, then emits all Pareto layers for four eligibility profiles; source kind is not scored and no result can grant proof, completion, or production authority |
| Reported full-route adapter | `application.reported_program_route_candidates` | requires an exact Candidate observation/projection pair and matching target structure; source-bound routes are literature, unbound routes retain a source warning; all remain exploration-only |
| Mechanism full-route restitcher | `application.mechanism_programs` + `mechanism_program_compilation` + `mechanism_program_route_candidates` + shared `program_innovation_contracts` / `program_span_substitutions` | admits only one-hop boundaries that exactly rejoin upstream/downstream states through one replaceable contiguous span; otherwise leaves the proposal in discovery; fallback is always retained |
| Mechanism validation + feedback | `application.mechanism_program_validations` + `mechanism_validation_frontier` + `mechanism_experiment_feedback` + orchestration review facade | exact Program/innovation/state/mechanism-signature binding; accepted success may enable read-only shadow, while net-transform support remains distinct from mechanism proof and failure/inconclusive observations remain visible; no proof/store/canonical mutation |
| Whole-cell/hybrid execution adapter | `application.route_execution_capabilities` + `route_execution_discovery` + `execution_programs` + `execution_program_route_candidates` | data-driven actors/operations/cofactor/carrier contract; reuses the shared contiguous-span restitcher and preserves negative-savings candidates in exploration |
| Execution validation + feedback | `application.execution_program_validations` + `execution_validation_frontier` + `execution_capability_feedback` + orchestration review facades | exact Program/capability/state/operation binding; success may enable read-only shadow, while valid failure/inconclusive records remain exact-boundary feedback; no catalog/store/canonical mutation |
| Unified experimental Claim projection | `application.experimental_claim_*` + `capability_applicability_calibration` + orchestration review facade | normalizes accepted biocatalytic and valid execution/mechanism observations into one exact-boundary Claim set; preserves positive/negative/inconclusive and conflicts; calibration emits read-only dirty hints without catalog mutation |
| Experimental Claim append-only store | `application.experimental_claim_store*` + `interfaces.campaign_experimental_claim_store` + recovery/GC adapters | explicit-gate nonempty admission; six CAS inputs are fully reprojected on replay; observation persistence cannot grant canonical proof, Program admission, completion, acceptance, or catalog authority |
| Experimental work + executor adapter | `application.experimental_work_frontier` + `experiment_execution_contracts` + `experiment_execution_results` + `orchestration.experiment_execution_runtime` | maps three validation frontiers and exact-domain dirty hints into current-canonical-frontier-bound read-only subtasks; result audit may release only an existing-domain validation candidate and cannot publish work, validation, Claim, proof, completion, or catalog writes |
| Bounded experiment dispatch | `providers.experiment` + `orchestration.experiment_dispatch_*` + `interfaces.campaign_experiment_*` | host-trusted policy selects only registered idempotent executors; manual handoff and settlement reuse RunKernel validation tasks, CAS, and rebuildable pointers rather than a second queue; executor/current-frontier/artifact/domain-gate checks remain mandatory |

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
| `chemical_state.v1`, `operation_node.v1`, `transformation_program.v1` | non-authoritative Phase-1 compatibility projection |
| `transformation_program_projection.v1` and oracle | edge/program dual-read migration check |
| `route_program_dual_read.v1` and oracle | current Workbench route-row mapping and UI semantic-equivalence gate |
| Program validation, admission event, store status/replay/oracle | non-authoritative append-only Phase-1 migration store |
| `transformation_program_migration_audit.v1` | read-only cross-run classification; never a scientific authority |
| Candidate route/state/operation/Program projection and oracle | non-authoritative preservation of complete routes with explicit gaps |
| Candidate migration audit and innovation screen | non-authoritative batch classification, positive enzyme proposals and no-applicable-enzyme controls |
| Biocatalytic Program proposal/route/bundle, specialized validation and bundle oracle | non-authoritative superstep replacement contract with explicit boundary, fallback and physical/chemical step accounting |
| Biocatalysis validation plan/frontier | read-only exact-boundary assay planning; literature analogy never becomes exact validation |
| Biocatalytic admission event/validation pack/store status/replay/oracle | append-only shadow persistence for specialized-validated supersteps; cannot change production route authority |
| `route_program_innovation_review.v1` | read-only orchestration review; never performs canonical graph or Program store writes |
| Program execution capability/catalog and execution proposal/route/bundle/oracle | read-only whole-cell/hybrid search priors and exact route restitching; specialized validation is required and no store write path exists |
| Execution validation plan/frontier, result, capability feedback projection and oracle | read-only exact-boundary experiment planning and outcome projection; only accepted success enables shadow eligibility, and no outcome grants persistence, proof, completion, or catalog mutation |
| Mechanism validation plan/frontier, result, feedback projection and oracle | read-only exact-boundary testing of restitched one-hop proposals; result interpretation separately tracks net transformation versus mechanism support; no result inherits anchor authority or creates reaction proof, persistence, completion, or canonical mutation |
| Experimental observation Claim/set/oracle and exact-boundary applicability calibration/oracle | unified read-only observation contract across biocatalytic, execution, and mechanism domains; retains all three polarities and emits exact-domain dirty hints without creating canonical facts |
| Experimental Claim validation pack, admission event/result, store replay/status/oracle | append-only, content-addressed persistence of nonempty exact-boundary Claim sets; full source reprojection and six CAS refs are mandatory; `edge_ids[]` remains production authority |
| Experimental work item/frontier/oracle and experiment execution request/result/audit | executor-neutral, content-addressed I/O bound to the current canonical frontier and exact plan boundary; aborted and negative envelopes stay visible; only a separately re-run domain validation gate can accept a released candidate |
| Program route candidate/set, reported route pack, mechanism/execution bundles and portfolio/oracle | read-only normalized alternatives plus deterministic multi-profile Pareto layers; current adapters are baseline, biocatalytic, whole-cell, hybrid, digest-bound reported full routes and fully restitched one-hop mechanism routes; unrestitched mechanisms remain discovery-only; `edge_ids[]` remains production authority |
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
