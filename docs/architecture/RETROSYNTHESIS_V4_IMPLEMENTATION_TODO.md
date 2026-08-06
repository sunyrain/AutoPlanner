# Retrosynthesis V4 implementation TODO

> Historical delivery record. Its checked boxes mean that a repository
> capability was implemented and tested; they do not prove that the current
> target-only product path is integrated or process-ready. The forward-looking,
> end-to-end acceptance plan is
> [IDEAL_RETROSYNTHESIS_ARCHITECTURE_AND_TODO.md](IDEAL_RETROSYNTHESIS_ARCHITECTURE_AND_TODO.md).

Status: P0-P10.4 and P10.6 implemented; bounded P10.5 real-model A/B remains
an explicit opt-in experiment

Updated: 2026-07-13
North star: Codex keeps a global view of the complete campaign, while only
host-validated evidence, reactions, stock records, and route proofs can become
scientific facts.

## 1. Non-negotiable product contract

- [x] A run has exactly one `RunKernel`, one canonical hypergraph, one frontier,
  one cost ledger, and one acceptance contract.
- [x] Codex can inspect and redesign the complete multi-route campaign; it is
  not reduced to a single-step predictor.
- [x] Codex proposals cannot directly promote chemistry, evidence, stock, or
  route-completion state.
- [x] Exploration hypotheses, materialized reactions, evidence-validated
  reactions, and stock-closed routes remain visibly distinct.
- [x] A completed route means every selected reaction edge and every selected
  leaf satisfies the configured proof and stock boundary.
- [x] Runs are bounded, resumable, replayable, revisioned, and comparable.
- [x] Model-free deterministic replay remains possible for committed golden
  cases.
- [x] No optimization may reduce accepted golden-route counts, proof levels,
  source independence, stock closure, or verifier strictness.

## 2. Target runtime flow

```text
target + acceptance + budgets
          |
          v
       RunKernel
          |
          +--> Campaign Context Compiler
          |          |
          |          v
          |    Codex GlobalCampaignDirector
          |    (initial architecture / event-driven replan / final synthesis)
          |          |
          |          v
          |    structured campaign hypotheses
          |
          +--> evidence / materialization / validation / stock workers
          |          |
          |          v
          +--> canonical evidence ledger + incremental hypergraph
          |          |
          +--> single deficit frontier + portfolio optimizer
          |          |
          +--> proof stitcher + acceptance evaluator
          |
          v
revisioned route portfolio + UI projection + audit bundle
```

The blackboard remains a rebuildable global reasoning projection. It is not an
independent chemistry authority or a second expansion state.

### 2.1 Operating model: global Codex, deterministic execution

Codex is not a single-step reaction oracle. Its unit of work is one bounded
campaign revision: the target, several route families, their shared
intermediates, unresolved proof/stock deficits, rejected chemistry, remaining
budget, and the current Pareto portfolio. One director response may therefore
restructure several routes, nominate a common upstream intermediate, retire a
dominated family, change source priorities, or define a portfolio-level pivot.

The host remains responsible for facts and repetition:

1. `CampaignContextCompiler` creates a bounded whole-campaign context and a
   delta from the previous director revision.
2. `GlobalCampaignDirector` proposes or revises the multi-route architecture;
   it cannot grant proof, inventory, or completion status.
3. The single `DeficitFrontier` turns the plan and canonical graph into cheap,
   typed work: materialize, discover/extract evidence, validate, audit stock,
   resolve conflicts, or diversify a route family.
4. Deterministic workers execute and cache that work. Every result returns
   through canonical ingestion; no worker owns a private route graph.
5. The proof stitcher selects a small route portfolio and evaluates the hard
   acceptance contract at its weakest edge and leaf.
6. Codex is called again only when a material event changes the global decision:
   a critical rejection, new exact evidence, a stock-boundary change, a shared
   bottleneck, a genuinely new family, or measured stagnation.

Default execution is model-free. A production campaign may use at most one
initial architecture call, two evidence/rejection-triggered replans, and one
final synthesis call unless its `RunSpec` explicitly declares a different hard
limit. No edge-level retry may invoke Codex implicitly. Calls, tokens, context
bytes, elapsed time, accepted proposals, and portfolio gain are charged to the
one `RunKernel` ledger. Exhausting that budget produces an explicit unresolved
result; it never relaxes proof or stock requirements.

The expected cost shape is therefore approximately constant in director calls
and proportional to accepted deterministic work, instead of one model call per
candidate edge or search round.

## 3. Delivery sequence

### P0 — Baseline and observability

Purpose: make every later architectural change measurable.

- [x] Add a dependency-free, non-authoritative run metrics recorder.
- [x] Instrument the main blackboard controller and tool execution boundary.
- [x] Bound retained stage rows and aggregate overflow without losing totals.
- [x] Reject/sanitize non-finite metric values and strengthen schema/digest
  validation.
- [x] Read model/token/attempt/accepted-expansion totals from the canonical run
  cost ledger.
- [x] Instrument the deterministic Nirmatrelvir V3 replay by stage.
- [x] Add a model-free benchmark command with warm/cold timings, CPU time,
  artifact bytes, graph size, and cache-hit counters.
- [x] Commit a small baseline manifest; never commit generated run directories.
- [x] Add regression thresholds that tolerate machine variance but reject major
  algorithmic regressions.

Exit gate:

- Nirmatrelvir remains 2 complete routes, 12 unique reaction hyperedges,
  7 stock terminals, at least 2 independent support groups, and 0 model calls.
- Metrics validate on success, failure, and recovery.
- Focused tests, Ruff, and the full local test suite pass.

Observed P0 baseline on Windows/Python 3.12:

- original cold replay: 145.308 s wall, 1.109 s CPU, 291,921 artifact bytes;
- shared run-scoped resolver replay: 134.299 s wall, 1.375 s CPU,
  315,618 artifact bytes, 7 cache hits and 46 misses;
- scientific output in both runs: 2 complete routes, 12 hyperedges, 7 stock
  terminals, 2 independent source groups, and 0 model invocations;
- measured bottleneck: 117.956 s in OPSIN requests, proving external structure
  resolution rather than chemistry/search CPU is the dominant cold-run cost.

### P1 — ArtifactStore, RunIndex, and data boundaries

Purpose: stop copying large mutable JSON state and make runs cheaply queryable.

- [x] Implement a SHA-256 content-addressed `ArtifactStore` with atomic writes.
- [x] Store small compatibility links/manifests in each run directory.
- [x] Implement a rebuildable SQLite `RunIndex` in WAL mode.
- [x] Index run identity, revisions, artifact digests, task status, graph counts,
  proof deficits, stock deficits, and performance metrics.
- [x] Define artifact retention, pinning, garbage-collection dry run, and safe
  deletion rules.
- [x] Add corruption, concurrent writer, recovery, and index-rebuild tests.
- [x] Classify repository/runtime/external data and document their locations.
- [x] Put V4 outputs, caches, vendor corpora, models, and source PDFs behind
  configured paths rather than new Git dependencies; legacy fallbacks remain
  isolated for P7/P9 removal.
- [x] Keep credential files out of repository and make V4 runtime access use
  environment variables or the OS credential store.

Exit gate:

- Repeated identical artifacts deduplicate by digest.
- Removing the index and rebuilding it changes no scientific artifact.
- Interrupted writes leave the previous revision readable.

Observed P1 cold/warm acceptance:

- cold deterministic replay: 123.772 s;
- warm replay from versioned resolver CAS: 0.854 s (144.9x speedup);
- warm resolver state: 46 persistent hits and 0 misses;
- both iterations retained exactly 2 complete routes, 12 hyperedges, 7 stock
  terminals, 2 independent source groups, and 0 model invocations;
- 26 indexed artifact references rebuilt from 2 immutable run manifests with
  SQLite integrity `ok` and no scientific artifact mutation.

### P2 — One RunKernel and durable event state

Purpose: remove competing campaign/round/expansion states.

- [x] Define `RunSpec`, `RunState`, `RunRevision`, `RunEvent`, `Deficit`, and
  `StopDecision` domain contracts.
- [x] Move acceptance, model, attempt, accepted-expansion, evidence, stock, and
  wall-time budgets under one kernel.
- [x] Separate `attempt_budget` from `accepted_expansion_budget` in the V4 path.
- [x] Guarantee one accepted child expansion increments exactly once.
- [x] Make all state transitions idempotent and event-addressed.
- [x] Persist a snapshot plus append-only events; replay must reproduce the same
  canonical state digest.
- [x] Preserve reserved/in-flight tasks deterministically after interruption so
  the scheduler can resume or settle the same task identities.
- [x] Replace round-count completion with acceptance or explicit unresolved
  deficits.
- [x] Expose cancellation, timeout accounting, pause/recovery, terminal states,
  and machine-readable failure reasons.

Exit gate:

- There is no second mutable expansion state in the blackboard, ChemEnzy
  adapter, or Codex campaign tracker.
- Crash/recovery replay produces the same graph and acceptance digest.

Observed P2 V4-path acceptance:

- one real, model-free Nirmatrelvir replay retained 2 complete routes, 12
  hyperedges, 7 stock terminals, and 0 model invocations;
- the run completed through 18 hash-chained events, 6 settled deterministic
  tasks, 0 accepted proposal expansions, and a digest-bound final snapshot;
- attempts and unique accepted-expansion identities are separately enforced,
  including concurrent reservations, prompt-context, visual, token, and
  wall-time admission;
- crash-tail repair, snapshot rebuild, tamper rejection, pause/resume, stale
  acceptance invalidation, and idempotency are covered by focused tests;
- the full local suite passed with 1425 tests, 3 skips, and 2 subtests.

The V4 path no longer creates a second campaign state.  The legacy controller
and Codex campaign tracker remain compatibility implementations until their P7
strangler adapters are complete; they are not treated as V4 authority.

### P3 — Global Campaign Director

Purpose: preserve Codex's distinctive global reasoning while bounding cost.

- [x] Define a versioned `GlobalCampaignPlan` schema containing route families,
  multi-step skeletons, strategic disconnections, shared intermediates,
  critical unknowns, source plan, fallback strategies, frontier priorities,
  pivot conditions, and portfolio rationale.
- [x] Implement the `CampaignContextCompiler` from canonical revisions.
- [x] Include complete campaign topology but compress raw documents, duplicate
  routes, old failures, and unchanged evidence.
- [x] Emit a context digest and delta from the previous director revision.
- [x] Implement `GlobalCampaignDirector` as a direct Codex child-agent boundary.
- [x] Support three bounded modes: initial architecture, event-driven replan,
  and final portfolio synthesis.
- [x] Trigger replanning only after a new actionable host observation: material
  evidence, contract/topology/depth rejection, an open stock-boundary change,
  or a provider/template/source edge that passed host validation. Portfolio
  stagnation alone remains a deficit and does not spend a model call.
- [x] Prevent identical context/config revisions from invoking Codex twice.
- [x] Enforce structured output, route-family count, context bytes, calls,
  tokens, and wall-time at the host boundary.
- [x] Validate every proposed molecule/reaction identity before frontier entry.
- [x] Record accepted, rejected, superseded, and ignored director proposals.
- [x] Ensure the director may prioritize and hypothesize globally but may not
  grant proof or completion authority.
- [x] Provide a deterministic fake/replay director for tests and golden cases.

Exit gate:

- A director plan can coordinate several multi-step route families and shared
  intermediates in one response.
- A verifier rejection causes a targeted global replan rather than blind local
  repetition.
- Default complex campaigns use a small bounded number of director calls.

Observed P3 host-boundary acceptance:

- one structured response coordinated 2 strategically distinct route families,
  2 multi-step skeletons, 3 concrete step hypotheses, a shared intermediate,
  source acquisition, fallback, pivot, stop, and portfolio rationale;
- identical context/config requests, including concurrent requests, invoked the
  runner exactly once and reused the immutable cached plan thereafter;
- a critical-edge rejection triggered an event replan, while unchanged context
  caused no invocation;
- the target-only orchestrator records a separate replan signal gate and audits
  molecule/edge/route-family ID retention after the call, so a replan is a
  canonical-graph union rather than replacement of the initial portfolio;
- invalid molecular identities were rejected as frontier candidates, and any
  attempted solved/proof/stock-closure authority invalidated the whole plan;
- deterministic replay runs with a zero model-call budget and the same plan
  schema, while real child calls remain bounded by the single RunKernel ledger;
- accepted/rejected/superseded/ignored dispositions are immutable audit
  artifacts and never count as accepted chemistry expansions;
- focused runtime/direct-child tests passed, followed by the full local suite:
  1435 tests passed, 3 skipped, and 2 subtests passed.

### P4 — Evidence, materialization, validation, and stock workers

Purpose: turn global hypotheses into auditable facts.

- [x] Define one worker command/result envelope with input revision, idempotency
  key, budget reservation, artifact refs, and status.
- [x] Implement candidate reactant/reagent materialization worker.
- [x] Run identity, parse, element balance, atom-jump, ancestor-cycle, duplicate,
  and impossible-precursor gates before expensive work.
- [x] Implement source discovery separately from exact-row extraction.
- [x] Automatically schedule extraction after a usable source is discovered.
- [x] Resume the campaign when new exact rows arrive.
- [x] Normalize patents, papers/SI, curated registries, image extraction, and
  Codex claims into source bindings with explicit provenance.
- [x] Keep L0 hypothesis, structural/materialized, validated, exact-source, and
  independently supported proof states separate.
- [x] Add conflict representation instead of silently selecting one source.
- [x] Implement versioned inventory adapters and immutable supplier snapshots.
- [x] Audit every selected deep leaf; do not infer availability from commonness.
- [x] Cache workers by normalized input plus dependency revision.
- [x] Add deterministic fixtures for success, rejection, timeout, partial
  extraction, source conflict, and stale stock.

Exit gate:

- A Codex multi-step skeleton can be independently materialized edge by edge.
- Rejected candidates consume attempts but never accepted expansions.
- Exact evidence and stock records are replayable without Codex.

Observed P4 deterministic-worker acceptance:

- one global multi-step plan compiled to unique edge commands while retaining
  every route-family/skeleton/step provenance reference; repeated shared edges
  were executed once and each accepted hyperedge counted exactly once;
- parse, identity, element inventory, large atom jump, self/ancestor cycle,
  surplus precursor, and duplicate gates run before reaction validation, and
  rejected commands consume attempts without creating accepted expansions;
- source discovery schedules exact-row extraction automatically, while exact
  evidence can only be promoted from command-bound, content-addressed artifacts
  in the host-owned `structured_exact_row_extraction` authority scope;
- patent, paper/SI, curated registry, image extraction, and Codex claim records
  share one provenance schema; Codex translations remain advisory and cannot
  claim exact-source authority;
- official patent HTML is frozen before PDF work; publication identity, full
  artifact digest, selected paragraph range, and normalized text digest all
  replay before deterministic product/reactant reconstruction. Only unresolved
  edge IDs descend to PDF/native-text/OCR/optional-vision fallback;
- exact rows emit material events recognized by `GlobalCampaignDirector`, so a
  newly arrived row resumes event-driven global planning rather than another
  blind local expansion;
- image-only pages use allowlisted local OCR with source/PDF/page/image/text
  replay binding; optional Codex page vision is separately budgeted, runs at
  most once per campaign, and contributes only host-normalized L0 hypotheses
  to the next global context;
- proof axes remain separate, independent support requires two host-derived
  source groups, and incompatible structures/conditions remain unresolved
  conflict records with no automatic winner;
- every selected leaf receives a stock audit against a host-trusted, versioned,
  immutable inventory artifact; commonness, untrusted availability claims, and
  stale snapshots cannot close a route;
- normalized input, graph/evidence dependency revisions, artifact digests,
  authority scopes, handler version, and execution policy bind worker caches;
  immutable exact-evidence and stock results replay without a worker or Codex;
- focused tests cover success, multi-step deduplication, cheap rejection,
  validation, automatic extraction, partial/tampered/untrusted extraction,
  source conflict, independent support, fresh/missing/stale/untrusted stock,
  cache reuse, timeout, stale revision, and artifact replay;
- the full local suite passed with 1446 tests, 3 skips, and 2 subtests.

### P5 — Single frontier and incremental hypergraph

Purpose: search for complete portfolios rather than accumulate local branches.

- [x] Define stable molecule, reaction-edge, source-binding, stock-observation,
  route-family, and hypothesis identities.
- [x] Merge Codex, ChemEnzy, templates, literature, and manual imports into one
  canonical hypergraph ingestion path.
- [x] Make Codex precursor/retron proposals real frontier candidates rather than
  UI-only annotations.
- [x] Define one `DeficitFrontier` for missing materialization, evidence,
  validation, stock, diversity, and route closure.
- [x] Rank by expected portfolio gain, distance to closure, evidence gain,
  source independence, route diversity, cost, failure risk, and prior attempts.
- [x] Add dirty-node revisions and incrementally recompute only affected
  ancestors, routes, deficits, and rankings.
- [x] Deduplicate canonical edges while preserving all independent source
  bindings and proposal origins.
- [x] Detect cycles, repeated chemistry, large atom jumps, and dominated route
  families before expansion.
- [x] Add deterministic tie-breaking and replay tests.
- [x] Compare incremental output digests against a full recomputation oracle.

Exit gate:

- No action can bypass canonical ingestion or create a private search graph.
- Incremental and full recomputation yield identical scientific results.
- Large graph updates scale with dirty subgraph size, not total graph size.

Observed P5 canonical-graph acceptance:

- molecule, exact reaction multiset, source binding, stock observation, route
  family, and hypothesis identities are canonical, full-digest IDs; presentation
  labels and precursor order do not change scientific identity;
- `CanonicalHypergraphStore` publishes one immutable graph revision through the
  existing `RunKernel`; blackboard, legacy frontier ledger, and UI remain
  projections and receive no V4 write authority;
- Codex global skeleton steps enter as real `frontier_candidate` hypotheses and
  produce materialization deficits; no reaction edge exists until the proposal
  Worker settles successfully;
- Codex, ChemEnzy, template, literature, and manual origins pass the same
  materialization/admission boundary, merge into one exact edge, retain all
  origin/source records, and count one unique accepted expansion;
- exact literature extraction automatically schedules proposal materialization;
  an evidence result cannot create a private edge or bypass proposal budgets;
- one `DeficitFrontier` ranks materialization, evidence, validation, stock,
  conflict, diversity, and route-closure work using explicit portfolio gain,
  closure distance, evidence/independence/diversity gain, cost, failure risk,
  and prior-attempt components with stable tie-breaking;
- duplicate edges are skipped before worker execution, and parse/element,
  large-atom-jump, ancestor-cycle, canonical graph cycle, and dominated-family
  gates prevent wasteful follow-on expansion;
- edge/source/proof/stock facts are ingested in dependency order independent of
  result arrival order; untrusted or digest-invalid facts are rejected without
  mutating the graph;
- dirty molecule/edge ancestors and affected route families are revisioned and
  recomputed incrementally.  In the 15-edge regression, one local stock update
  recomputed less than 20% of graph entities;
- incremental frontiers and graph projections match full recomputation oracles
  byte-semantically at the scientific projection, including the local-update
  regression;
- focused V4 and legacy-frontier regressions passed, followed by the full local
  suite: 1455 tests passed, 3 skipped, and 2 subtests passed.

### P6 — Proof stitcher and route portfolio

Purpose: make completion strict, diverse, and understandable.

- [x] Define one proof-level policy and remove duplicate/ad-hoc interpretations.
- [x] Require proof for every selected reaction edge and stock audit for every
  selected leaf.
- [x] Stitch literature, generated, and stock segments through canonical IDs.
- [x] Represent proof gaps and conflicting evidence as first-class deficits.
- [x] Compute route edge-set diversity, strategic-disconnection diversity,
  shared bottlenecks, source independence, length, convergence, and risk.
- [x] Select a small Pareto portfolio rather than dozens of nearly identical
  routes.
- [x] Support step alternatives and route-module replacement without duplicating
  the entire route.
- [x] Produce explicit accepted, unresolved, budget-exhausted, and invalid
  closeout decisions.
- [x] Ensure aggregate counts never imply completion by themselves.

Exit gate:

- Every completion claim can be traced through reaction proofs and stock
  observations to immutable artifacts.
- Removing any required proof deterministically reopens the corresponding
  deficit.

Observed P6 proof-portfolio acceptance:

- one versioned `ProofPolicy` now computes L0 through L4 from replayable facts;
  exact literature never substitutes for deterministic reaction validation,
  model claims never substitute for independent sources, and commonness never
  substitutes for a current trusted stock observation;
- selected routes are enumerated over canonical hyperedges, and every complete
  route carries edge proof digests, exact-record/source-binding IDs, leaf stock
  observation IDs, inventory snapshot IDs, and weakest-link proof/stock state;
- variant-level proof, evidence, stock, conflict, closure, and diversity gaps
  use the existing `DeficitFrontier` item schema and taxonomy before projection
  into `RunKernel`; P6 introduces no private queue or completion counter;
- route selection computes edge-set and strategic-disconnection diversity,
  shared bottlenecks/intermediates, source independence, length, convergence,
  and risk, then selects a bounded 2--5 route portfolio with Pareto preference
  and deterministic diversity-aware tie breaking;
- same-product alternatives are represented as canonical replacement modules;
  a module patch replaces one edge ID and reuses the shared subgraph instead of
  copying the full route;
- accepted, unresolved, budget-exhausted, and invalid closeouts are explicit;
  only boolean edge proof plus leaf stock closure can accept, while aggregate
  route/node/edge counts remain diagnostics;
- a deterministic two-route esterification fixture closed with two distinct
  validated edges, two independent exact sources per edge, and all leaves in a
  trusted inventory snapshot.  Removing one reaction proof reopened only that
  validation deficit, removing one shared stock observation reopened only that
  leaf deficit, conflicts blocked acceptance, and digest corruption failed
  closed as invalid;
- 25 focused P4--P6 tests passed, followed by the full local suite: 1460 tests
  passed, 3 skipped, 2 subtests passed, with zero model invocations.

### P7 — Strangle giant modules and compatibility paths

Purpose: make the system maintainable without a risky rewrite.

- [x] Turn the public controller surface into a thin adapter: V4 requests use
  `RunKernel`, while the old behavior remains an explicit frozen compatibility
  engine until P9 consumer migration.
- [x] Extract scheduling, campaign direction, evidence orchestration, graph
  projection, closeout, recovery, and artifact publication into bounded modules.
- [x] Split the tool registry from tool implementations and execution policy.
- [x] Split `RouteForest` storage, identity, traversal, ranking, and projection.
- [x] Remove duplicated Codex campaign state and legacy action-planner ownership.
- [x] Inventory schema versions and compatibility branches.
- [x] Attach telemetry and a removal milestone to every compatibility shim.
- [x] Delete unreachable code and duplicate scripts only after usage/search and
  regression proof.
- [x] Enforce module dependency direction and add architecture tests.
- [x] Establish practical file-size/complexity budgets for new code.

Exit gate:

- Core orchestration can be understood without reading a multi-thousand-line
  controller.
- Legacy adapters contain no scientific logic and can be removed independently.

Observed P7 strangler acceptance:

- `RetrosynthesisCampaignService` is the compact V4 orchestration owner.  It
  coordinates one `RunKernel`, bounded global director, deterministic Workers,
  canonical graph, single frontier, proof portfolio, recovery, and artifact
  publication without importing the blackboard controller, recursive Codex
  campaign, old queues, legacy portfolio, `RouteForest`, or Web;
- graph exploration deficits and proof-closeout deficits now pass through one
  `frontier_runtime` projection into `RunKernel`; V4 creates no second queue,
  campaign-state JSON, or action-planner ownership path;
- the public blackboard controller and recursive Codex campaign functions are
  thin, metadata-bearing compatibility adapters over frozen implementations.
  Package APIs are lazy and expose the V4 service/contracts first, avoiding
  eager initialization of the old orchestration chain;
- stable identity, route enumeration, Pareto selection/metrics, proof policy,
  portfolio publication, and frontier publication are bounded modules;
  `proof_portfolio.py` fell from 829 to 303 lines without output changes;
- the V4 replacement for `RouteForest` separates ArtifactStore persistence,
  canonical identity, route traversal, ranking/diversity, and proof projection.
  The 7k-line legacy display compiler remains only for historical-run rendering
  until P8 migrates its UI consumer;
- local tool names, execution/error policy, and implementations are separate;
  registry drift now fails closed instead of silently exposing or losing a
  tool;
- `compatibility_inventory` records every known legacy owner, its V4
  replacement, telemetry source, and P8/P9/P10 removal milestone.  Instrumented
  uses append digest-bound, explicitly non-scientific events under each run;
- a repository-wide reference search showed the large compatibility modules
  are still used by Web, scripts, and replay tests, so none was falsely called
  unreachable.  They are frozen against new product logic and deletion waits
  for consumer migration plus regression proof;
- architecture tests enforce dependency direction, forbidden legacy imports,
  focused-module line ceilings, compatibility metadata, telemetry integrity,
  and tool registry separation;
- V4 create/materialize/closeout/reopen tests prove recovery from the kernel
  event chain and canonical graph with zero private campaign state and zero
  model calls.  The full local suite passed with 1468 tests, 3 skips, and 2
  subtests.

### P8 — Route UI and rendering performance

Purpose: communicate scientific state instead of branch volume.

- [x] Default to a 2–5 route portfolio, not the entire exploration graph.
- [x] Add explicit views for disconnection hypotheses, expanded graph,
  reaction-validated routes, and stock-closed routes.
- [x] Color encodes proof/confidence; badges encode source/proposal type.
- [x] Show shared intermediates once and expand alternatives on demand.
- [x] Add evidence, stock, rejection, conflict, and provenance inspectors.
- [x] Stream revision deltas rather than replacing the whole graph.
- [x] Stabilize camera/world transforms so pointer drag never moves the render
  layer independently of canvas state.
- [x] Batch pointer movement with animation frames and eliminate drag-time layout
  recomputation.
- [x] Add viewport culling, level of detail, cached molecular depictions, stable
  layout, and move heavy computation off the UI thread (backend projection or
  worker where required).
- [x] Add interaction regressions for drag, zoom, fit, selection, minimap, and
  large graphs.
- [x] Benchmark frame time, dropped frames, DOM/canvas object count, update
  latency, and memory.

Exit gate:

- The accepted portfolio is readable at default zoom.
- Drag/zoom remains stable on benchmark-size graphs without visible flashing.
- UI labels cannot confuse L0 hypotheses with closed routes.

Observed P8 acceptance:

- `route_workbench` projects only the proof portfolio selected by P6 (never more
  than five routes), exposes four explicit scientific views, canonical shared
  intermediates, replacement modules, and proof/evidence/stock/conflict/
  rejection/provenance inspectors. `RetrosynthesisCampaignService.workbench`
  returns both a digest-bound snapshot and entity upsert/removal delta;
- new V4 output reaches the existing offline shell through
  `harness.v4_route_workbench`, a display-only adapter that does not import or
  execute the 7k-line `RouteForest` compiler. Historical artifacts remain
  readable through the frozen compatibility path until P9 migrates Web/CLI;
- the old split camera was removed. Pan and zoom now update one SVG world
  transform, pointer capture begins on pointer-down, motion is latest-value RAF
  batched, layout is cached and never recomputed during drag, and the SVG render
  layer itself stays fixed;
- the client has semantic zoom/LOD, bounded route overview, cached depictions and
  graph models, large-graph viewport culling, and an in-browser performance
  probe. Stable logical layout is computed before rendering, so no heavy layout
  work remains on the interaction thread;
- a real local headless Chromium regression exercised drag, anchor zoom, fit,
  selection, minimap recentering, and culling on a 70-step graph. The observed
  run rendered 281 objects, reported 0 dropped frames, a 7.5 ms graph update,
  and about 10 MB JS heap. These values are environment observations, not fixed
  cross-device guarantees;
- focused projection, delivery, integrity, service recovery, static interaction,
  and browser interaction tests make L0 hypotheses visibly and contractually
  distinct from expanded, reaction-validated, and stock-closed routes. No model
  or network call is used by the P8 test path. The full local suite passed with
  1480 tests, 3 skips, and 2 subtests.

### P9 — Unified interfaces and repository cleanup

Purpose: make the optimized path the obvious path.

- [x] Provide one CLI with `run`, `resume`, `status`, `validate`, `replay`,
  `benchmark`, `export`, and `gc --dry-run` commands.
- [x] Route API and WebUI through the same application services.
- [x] Remove obsolete one-off launchers after mapping their supported use cases.
- [x] Move historical reports out of active documentation navigation.
- [x] Keep active docs short: architecture, runbook, schemas, testing, and data
  policy.
- [x] Tighten ignore rules for runs, caches, credentials, local corpora, and
  generated reports.
- [x] Add a repository audit for tracked size, large blobs in current tree,
  duplicate assets, dead imports, and generated artifacts.
- [x] Do not rewrite published Git history without a separate explicit decision;
  clean the current tree and prevent recurrence first.
- [x] Do not add GitHub Actions or other CI configuration; all quality gates are
  runnable locally.

P9 implementation result:

- `python -m cascade_planner` is the sole campaign/operator entry. Its run path
  fixes model and visual calls at zero and exposes deterministic resume,
  validation, replay, benchmark, export, list, Web serving, repository audit,
  and explicit dry-run-only GC;
- CLI and `/api/v4/runs` share `CampaignGateway` and
  `RetrosynthesisCampaignService`. `/v4` and workbench HTML compile from the
  same digest-bound snapshot; HTTP clients cannot select arbitrary run paths;
- the Waitress and PowerShell wrappers were removed. Remaining scripts are
  classified as frozen V3 saved-run compatibility, P10 golden cases, or
  specialized external-data/source tools in `LEGACY_ENTRYPOINTS.md`;
- generated reports, rendered assets, copied archives, the tracked USPTO corpus,
  and the generated UniProt cache were removed from the current tree without
  changing Git history. The statin summary remains as a 25 KiB test fixture;
- the read-only audit reports 730 tracked files and 17.12 MiB, with zero current
  blobs at or above 1 MiB, generated artifacts, historical copies, duplicate
  assets, credential candidates, GitHub Actions, missing files, or dead-import
  candidates. Ruff independently removed 132 confirmed unused imports;
- focused CLI/gateway/API/audit, compatibility, Web, and fixture regressions run
  without a model or network call. The full local suite is the final P9 gate.

Exit gate:

- A new operator can run and inspect the system without choosing among dozens of
  scripts.
- The tracked current tree contains code, small fixtures, and active docs only.

### P10 — End-to-end acceptance and release

Purpose: prove both scientific and engineering improvement.

#### P10.1 -- Freeze the release contract and replay format

- [x] Define one compact `retrosynthesis_replay_pack.v1` containing the target,
  acceptance and budget contracts, global route plan, exact-row source
  bindings, atom-mapped reactions, versioned stock observations, and expected
  scientific metrics.
- [x] Keep copyrighted PDFs, vendor corpora, generated runs, and caches outside
  Git; committed rows must retain document identity, page/location, artifact
  digest, parser provenance, and authority scope.
- [x] Implement one package-level replay runner used by tests and CLI. Do not
  add another target-specific launcher or alternate campaign state.
- [x] Make replay stages individually idempotent and resumable: create, plan,
  materialize, extract, validate, stock-audit, closeout, workbench, validate.
- [x] Add schema, digest-tamper, source-artifact binding, stale-stock,
  duplicate-edge, and
  expected-metric mismatch tests.

Gate: a replay pack can rebuild a run from an empty runtime directory without a
network or model call, and the second replay reuses immutable worker artifacts.

#### P10.2 -- Nirmatrelvir scientific golden

- [x] Convert the approved Science/SI and WO patent route rows into the replay
  format while preserving two independent source groups.
- [x] Apply both complete route skeletons in one global campaign revision.
- [x] Materialize the 15 source steps into exactly 12 canonical reaction
  hyperedges; shared chemistry must retain both origins without double-counting
  accepted expansions.
- [x] Atom-map and deterministically validate every selected hyperedge. Exact
  literature alone must not masquerade as L2 reaction validation.
- [x] Audit every selected deep leaf against a current, digest-bound inventory
  snapshot and close exactly the intended procurement boundary.
- [x] Require at least 2 distinct complete routes, proof level L3 on every
  selected edge, at least 2 independent source groups, all selected leaves
  stock-closed, and 0 model/visual invocations.
- [x] Interrupt after materialization and after exact-row ingestion, reopen the
  run, validate event replay, and prove identical graph, portfolio, and
  workbench digests. Pause/resume events intentionally change the operational
  event-state digest while leaving scientific projections identical.

Gate: reproduce 2 complete routes, 12 unique reaction hyperedges, 7 stock
terminals, at least 2 independent source groups, zero false closure, and zero
model calls from a clean runtime.

#### P10.3 -- Paclitaxel presentation and scalability case

- [x] Build a bounded, chemically coherent multi-route plan that emphasizes
  convergent route families, shared intermediates, replacements, and unresolved
  alternatives rather than rendering the full exploration graph by default.
- [x] Do not label a route complete unless every displayed selected edge and
  leaf passes the same proof/stock contract used by Nirmatrelvir.
- [x] Verify the four separate UI views: disconnection hypotheses, materialized
  graph, reaction-validated routes, and stock-closed routes.
- [x] Confirm proof-level coloring, source badges, shared-node rendering,
  alternative expansion, inspectors, and explicit unresolved deficits.
- [x] Benchmark stable layout, graph-delta latency, initial object count,
  viewport culling, drag/zoom frame behavior, and memory on the case.

Gate: the default view remains legible and interactive, while the full graph is
available on demand and never confuses branch volume with completion.

#### P10.4 -- Generalization and honest failure panel

- [x] Select at least three structurally and strategically different complex
  targets: one convergent small molecule, one stereochemically dense natural
  product, and one chemoenzymatic or macrocyclic case.
- [x] Include at least one target whose fixture and route are not already in the
  repository at selection time.
- [x] Run the deterministic/local baseline first for every target with identical
  acceptance semantics and target-appropriate bounded budgets.
- [x] Classify outcomes only as accepted, unresolved, budget-exhausted, or
  invalid; store the exact blocking edge/evidence/stock/diversity deficits.
- [x] Reject any proposal that violates identity, element balance, atom-jump,
  ancestor-cycle, or duplicate gates before expensive work.

Gate: unseen targets either close under auditable facts or explain precisely why
they do not; the false-closure count remains zero.

Observed P10.1-P10.4 acceptance:

- the compact Nirmatrelvir pack is 51 KiB and rebuilds a clean run in about
  2.6 s through 29 deterministic tasks: 12 unique accepted expansions, 12/12
  reaction validations, 15 exact rows, 7 audited leaves, 2 distinct complete
  routes, 2 independent sources, and 0 model/visual calls;
- completed replay performs no new task on a second invocation. Interruption
  after 12 materializations or after 16 materialization/evidence tasks resumes
  to the same scientific graph, proof portfolio, and workbench as an
  uninterrupted run;
- the Paclitaxel default was reduced from a 96-branch, 83-node, 122-step,
  roughly 600 MiB local legacy projection with zero proven portfolio routes to
  3 strategically distinct L1 routes. It remains explicitly unresolved with
  evidence, validation, stock, and closure deficits; bounded status/oracle/UI
  projection measured about 41 ms median and 1.28 MiB Python peak allocation;
- Lorlatinib, Trabectedin, and Voclosporin were absent from the repository at
  selection time. Their zero-model, no-fixture baselines terminate as
  `budget_exhausted` with a named diversity deficit, zero routes, zero edges,
  and zero false closure.

#### P10.5 -- Bounded global-Director A/B

- [ ] Run this only after all model-free baselines and golden tests pass.
- [ ] Use one selected unresolved/under-diverse case, not every molecule.
- [ ] Cap the campaign at 1 initial architecture call, 2 material-event replans,
  and 1 final synthesis call; set hard token, context-byte, and wall-time limits.
- [ ] Give Codex the complete compressed campaign topology and portfolio
  deficits so it can change route families and shared strategy globally.
- [ ] Prevent per-edge implicit calls and identical-context reinvocation.
- [ ] Record exact calls, tokens, context bytes, elapsed time, accepted/rejected/
  superseded proposals, new validated edges, route diversity, closed deficits,
  and final portfolio gain.
- [ ] Keep the Codex result only if it improves the scientific portfolio or
  reduces a named deficit without weakening validation; otherwise the measured
  outcome is `no_gain` rather than a forced success claim.

Gate: demonstrate measurable portfolio-level value within the fixed budget, or
retain the cheaper deterministic path as the supported default.

Implementation note: the host now enforces the 1 initial / 2 event-replan / 1
final mode caps from durable task-reservation events, in addition to the total
RunKernel model/token/context/wall-time budget and identical-context cache. The
real-model A/B is intentionally not executed in this release run because the
operator requested no expensive model invocation. Deterministic replay tests
exercise the identical structured boundary without making a model-performance
claim.

#### P10.6 -- Performance, migration, and release

- [x] Compare scientific results, wall/CPU time, peak memory, artifact bytes,
  cache reuse, dirty-graph recomputation, model cost, and UI performance against
  the recorded baseline.
- [x] Run focused replay/recovery/UI/performance tests, Ruff, the full local test
  suite, golden replays, repository audit, and `gc --dry-run`.
- [x] Remove compatibility modules or launchers only when the compatibility
  inventory proves no active consumer remains; retain an explicit adapter when
  saved-run migration is not yet complete.
- [x] Update architecture, schemas, operator runbook, case manifests, migration
  notes, and a concise before/after explanation.
- [x] Confirm the current tree has no generated reports, credentials, local
  corpora, GitHub Actions, or CI configuration.
- [x] Commit intentionally with `[skip ci]`, push directly to `main`, verify the
  remote commit, and leave a clean worktree.

Observed P10.6 acceptance:

- the complete offline suite passes with 1500 tests, 3 expected skips, and 2
  subtests; focused replay/director/architecture/CLI acceptance passes with 29
  tests;
- Ruff passes across `cascade_planner`, `tests`, and `scripts`. Explicit
  per-file exceptions cover only frozen research trees and legacy scripts;
  architecture tests prevent those exceptions from covering V4 or tests;
- repository audit reports a clean 739-file, roughly 18 MB current tree with no
  generated artifacts, tracked credentials, GitHub Actions, missing tracked
  files, Python parse errors, or duplicate assets;
- a clean Nirmatrelvir replay completes in about 2.7 seconds and a second
  invocation schedules zero stages. Artifact GC dry-run is read-only and keeps
  every indexed/pointer-bound artifact pinned.

Exit gate:

- At least one complete, multi-source, stock-closed multi-route portfolio is
  reproduced deterministically.
- New/unseen targets either close validly or report exact unresolved deficits;
  they never fake completion.
- Global Codex planning is structurally bounded to 1 initial / 2 event-replan /
  1 final call and cannot run per edge. A real-model portfolio-gain measurement
  remains an explicit P10.5 experiment and must report `no_gain` honestly when
  it does not improve the portfolio.
- All required local quality gates pass and no CI/Action files are added.

## 4. Cross-phase performance and quality gates

Every phase must report both scientific and engineering deltas.

Scientific invariants:

- accepted complete routes;
- unique validated reaction edges;
- selected leaf stock-closure rate;
- minimum and distribution of edge proof levels;
- independent source groups;
- route-family/edge-set diversity;
- invalid or rejected candidate count by reason;
- false-closure count, which must remain zero.

Engineering invariants:

- wall and CPU time by stage;
- peak memory and artifact bytes;
- full-graph versus dirty-subgraph recomputation;
- cache hit/miss counts;
- attempts versus accepted expansions;
- model calls, input/output tokens, and model wall time;
- recovery time and replay digest equality;
- UI update latency and interaction frame time.

## 5. Implementation discipline

- [x] Make changes through strangler adapters; do not replace the working system
  in one unverified rewrite.
- [x] Add characterization tests before moving legacy behavior.
- [x] Keep chemistry authority separate from operational observability.
- [x] Make deterministic gates run before network/model work.
- [x] Use stable identities and idempotency keys at every boundary.
- [x] Do not add another schema when an existing canonical contract can be
  migrated or extended deliberately.
- [x] Do not optimize by deleting route diversity or weakening validation.
- [x] Do not invoke a model during unit, golden, or performance baseline tests.
- [x] Keep generated artifacts outside Git and publish only compact manifests.
- [x] Complete and verify one phase gate before broadening the migration surface.

## 6. Definition of V4 complete

V4 is complete only when the global director, deterministic workers, canonical
hypergraph, single frontier, proof/stock closure, incremental runtime, and route
portfolio UI operate through one resumable `RunKernel`; the legacy controller is
only a compatibility adapter; complex golden and unseen cases satisfy the local
scientific and performance gates; and the cleaned repository is pushed to
`main` without CI/Action configuration.
