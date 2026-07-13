# Retrosynthesis V4 implementation TODO

Status: active implementation plan

Updated: 2026-07-13
North star: Codex keeps a global view of the complete campaign, while only
host-validated evidence, reactions, stock records, and route proofs can become
scientific facts.

## 1. Non-negotiable product contract

- [ ] A run has exactly one `RunKernel`, one canonical hypergraph, one frontier,
  one cost ledger, and one acceptance contract.
- [ ] Codex can inspect and redesign the complete multi-route campaign; it is
  not reduced to a single-step predictor.
- [ ] Codex proposals cannot directly promote chemistry, evidence, stock, or
  route-completion state.
- [ ] Exploration hypotheses, materialized reactions, evidence-validated
  reactions, and stock-closed routes remain visibly distinct.
- [ ] A completed route means every selected reaction edge and every selected
  leaf satisfies the configured proof and stock boundary.
- [ ] Runs are bounded, resumable, replayable, revisioned, and comparable.
- [ ] Model-free deterministic replay remains possible for committed golden
  cases.
- [ ] No optimization may reduce accepted golden-route counts, proof levels,
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
- [x] Trigger replanning only for material evidence, critical edge rejection,
  portfolio stagnation, new route family, shared bottleneck, or stock-boundary
  changes.
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

- [ ] Define one worker command/result envelope with input revision, idempotency
  key, budget reservation, artifact refs, and status.
- [ ] Implement candidate reactant/reagent materialization worker.
- [ ] Run identity, parse, element balance, atom-jump, ancestor-cycle, duplicate,
  and impossible-precursor gates before expensive work.
- [ ] Implement source discovery separately from exact-row extraction.
- [ ] Automatically schedule extraction after a usable source is discovered.
- [ ] Resume the campaign when new exact rows arrive.
- [ ] Normalize patents, papers/SI, curated registries, image extraction, and
  Codex claims into source bindings with explicit provenance.
- [ ] Keep L0 hypothesis, structural/materialized, validated, exact-source, and
  independently supported proof states separate.
- [ ] Add conflict representation instead of silently selecting one source.
- [ ] Implement versioned inventory adapters and immutable supplier snapshots.
- [ ] Audit every selected deep leaf; do not infer availability from commonness.
- [ ] Cache workers by normalized input plus dependency revision.
- [ ] Add deterministic fixtures for success, rejection, timeout, partial
  extraction, source conflict, and stale stock.

Exit gate:

- A Codex multi-step skeleton can be independently materialized edge by edge.
- Rejected candidates consume attempts but never accepted expansions.
- Exact evidence and stock records are replayable without Codex.

### P5 — Single frontier and incremental hypergraph

Purpose: search for complete portfolios rather than accumulate local branches.

- [ ] Define stable molecule, reaction-edge, source-binding, stock-observation,
  route-family, and hypothesis identities.
- [ ] Merge Codex, ChemEnzy, templates, literature, and manual imports into one
  canonical hypergraph ingestion path.
- [ ] Make Codex precursor/retron proposals real frontier candidates rather than
  UI-only annotations.
- [ ] Define one `DeficitFrontier` for missing materialization, evidence,
  validation, stock, diversity, and route closure.
- [ ] Rank by expected portfolio gain, distance to closure, evidence gain,
  source independence, route diversity, cost, failure risk, and prior attempts.
- [ ] Add dirty-node revisions and incrementally recompute only affected
  ancestors, routes, deficits, and rankings.
- [ ] Deduplicate canonical edges while preserving all independent source
  bindings and proposal origins.
- [ ] Detect cycles, repeated chemistry, large atom jumps, and dominated route
  families before expansion.
- [ ] Add deterministic tie-breaking and replay tests.
- [ ] Compare incremental output digests against a full recomputation oracle.

Exit gate:

- No action can bypass canonical ingestion or create a private search graph.
- Incremental and full recomputation yield identical scientific results.
- Large graph updates scale with dirty subgraph size, not total graph size.

### P6 — Proof stitcher and route portfolio

Purpose: make completion strict, diverse, and understandable.

- [ ] Define one proof-level policy and remove duplicate/ad-hoc interpretations.
- [ ] Require proof for every selected reaction edge and stock audit for every
  selected leaf.
- [ ] Stitch literature, generated, and stock segments through canonical IDs.
- [ ] Represent proof gaps and conflicting evidence as first-class deficits.
- [ ] Compute route edge-set diversity, strategic-disconnection diversity,
  shared bottlenecks, source independence, length, convergence, and risk.
- [ ] Select a small Pareto portfolio rather than dozens of nearly identical
  routes.
- [ ] Support step alternatives and route-module replacement without duplicating
  the entire route.
- [ ] Produce explicit accepted, unresolved, budget-exhausted, and invalid
  closeout decisions.
- [ ] Ensure aggregate counts never imply completion by themselves.

Exit gate:

- Every completion claim can be traced through reaction proofs and stock
  observations to immutable artifacts.
- Removing any required proof deterministically reopens the corresponding
  deficit.

### P7 — Strangle giant modules and compatibility paths

Purpose: make the system maintainable without a risky rewrite.

- [ ] Turn the existing controller into a thin adapter over `RunKernel`.
- [ ] Extract scheduling, campaign direction, evidence orchestration, graph
  projection, closeout, recovery, and artifact publication into bounded modules.
- [ ] Split the tool registry from tool implementations and execution policy.
- [ ] Split `RouteForest` storage, identity, traversal, ranking, and projection.
- [ ] Remove duplicated Codex campaign state and legacy action-planner ownership.
- [ ] Inventory schema versions and compatibility branches.
- [ ] Attach telemetry and a removal milestone to every compatibility shim.
- [ ] Delete unreachable code and duplicate scripts only after usage/search and
  regression proof.
- [ ] Enforce module dependency direction and add architecture tests.
- [ ] Establish practical file-size/complexity budgets for new code.

Exit gate:

- Core orchestration can be understood without reading a multi-thousand-line
  controller.
- Legacy adapters contain no scientific logic and can be removed independently.

### P8 — Route UI and rendering performance

Purpose: communicate scientific state instead of branch volume.

- [ ] Default to a 2–5 route portfolio, not the entire exploration graph.
- [ ] Add explicit views for disconnection hypotheses, expanded graph,
  reaction-validated routes, and stock-closed routes.
- [ ] Color encodes proof/confidence; badges encode source/proposal type.
- [ ] Show shared intermediates once and expand alternatives on demand.
- [ ] Add evidence, stock, rejection, conflict, and provenance inspectors.
- [ ] Stream revision deltas rather than replacing the whole graph.
- [ ] Stabilize camera/world transforms so pointer drag never moves the render
  layer independently of canvas state.
- [ ] Batch pointer movement with animation frames and eliminate drag-time layout
  recomputation.
- [ ] Add viewport culling, level of detail, cached molecular depictions, stable
  layout, and worker-based heavy computation.
- [ ] Add interaction regressions for drag, zoom, fit, selection, minimap, and
  large graphs.
- [ ] Benchmark frame time, dropped frames, DOM/canvas object count, update
  latency, and memory.

Exit gate:

- The accepted portfolio is readable at default zoom.
- Drag/zoom remains stable on benchmark-size graphs without visible flashing.
- UI labels cannot confuse L0 hypotheses with closed routes.

### P9 — Unified interfaces and repository cleanup

Purpose: make the optimized path the obvious path.

- [ ] Provide one CLI with `run`, `resume`, `status`, `validate`, `replay`,
  `benchmark`, `export`, and `gc --dry-run` commands.
- [ ] Route API and WebUI through the same application services.
- [ ] Remove obsolete one-off launchers after mapping their supported use cases.
- [ ] Move historical reports out of active documentation navigation.
- [ ] Keep active docs short: architecture, runbook, schemas, testing, and data
  policy.
- [ ] Tighten ignore rules for runs, caches, credentials, local corpora, and
  generated reports.
- [ ] Add a repository audit for tracked size, large blobs in current tree,
  duplicate assets, dead imports, and generated artifacts.
- [ ] Do not rewrite published Git history without a separate explicit decision;
  clean the current tree and prevent recurrence first.
- [ ] Do not add GitHub Actions or other CI configuration; all quality gates are
  runnable locally.

Exit gate:

- A new operator can run and inspect the system without choosing among dozens of
  scripts.
- The tracked current tree contains code, small fixtures, and active docs only.

### P10 — End-to-end acceptance and release

Purpose: prove both scientific and engineering improvement.

- [ ] Replay Nirmatrelvir as the deterministic evidence-first acceptance case.
- [ ] Replay Paclitaxel as a complex, highly branched presentation case.
- [ ] Select several structurally and strategically different complex targets.
- [ ] Include at least one target without a pre-existing local case fixture.
- [ ] Run model-free baselines before any optional Codex campaign.
- [ ] Run a bounded global-director campaign and record exact calls, tokens,
  elapsed time, accepted proposals, rejected proposals, and portfolio gain.
- [ ] Compare scientific results, wall time, CPU, memory, artifacts, graph update
  work, model cost, and UI performance against the baseline.
- [ ] Verify interruption/resume and repeated-run cache behavior.
- [ ] Run the full local test suite, Ruff, golden replays, repository audit, and
  performance gates.
- [ ] Update architecture, operator runbook, migration notes, and before/after
  explanation.
- [ ] Commit intentionally with `[skip ci]`, push directly to `main`, verify the
  remote head, and leave a clean worktree.

Exit gate:

- At least one complete, multi-source, stock-closed multi-route portfolio is
  reproduced deterministically.
- New/unseen targets either close validly or report exact unresolved deficits;
  they never fake completion.
- Global Codex planning provides measurable portfolio gain without unbounded
  calls or context growth.
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

- [ ] Make changes through strangler adapters; do not replace the working system
  in one unverified rewrite.
- [ ] Add characterization tests before moving legacy behavior.
- [ ] Keep chemistry authority separate from operational observability.
- [ ] Make deterministic gates run before network/model work.
- [ ] Use stable identities and idempotency keys at every boundary.
- [ ] Do not add another schema when an existing canonical contract can be
  migrated or extended deliberately.
- [ ] Do not optimize by deleting route diversity or weakening validation.
- [ ] Do not invoke a model during unit, golden, or performance baseline tests.
- [ ] Keep generated artifacts outside Git and publish only compact manifests.
- [ ] Complete and verify one phase gate before broadening the migration surface.

## 6. Definition of V4 complete

V4 is complete only when the global director, deterministic workers, canonical
hypergraph, single frontier, proof/stock closure, incremental runtime, and route
portfolio UI operate through one resumable `RunKernel`; the legacy controller is
only a compatibility adapter; complex golden and unseen cases satisfy the local
scientific and performance gates; and the cleaned repository is pushed to
`main` without CI/Action configuration.
