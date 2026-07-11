# AutoPlanner Mainline

Last update: 2026-07-12.

The active mainline is the policy-driven agentic blackboard controller. The
normative component boundaries, V1-to-V2 migration rules, proof ladder, and
paclitaxel acceptance contract are in
[Architecture V2](ARCHITECTURE_V2.md).

```text
target input
-> deterministic preflight
-> blackboard state
-> Codex coordinator calls spawn_agent for each specialist role
-> child reports return typed RetrosynthesisProposalReport artifacts
-> provider envelopes bind schemas, versions, hashes, and correlation groups
-> evidence actions acquire 2-3 independent source groups through exact rows
-> route consensus fuses every canonical product intermediate independently
-> stock-first persistent frontiers launch bounded direct Codex child teams
-> frontier neighborhoods assemble into the V2 reaction-hypergraph overlay
-> Codex chooses a compact typed action batch
-> deterministic validator checks safety, budget, binding, and proof boundaries
-> local tools execute approved actions
-> blackboard records typed summaries and artifact refs
-> exact rows retain product, every reactant, source binding, and atom mapping
-> every consensus edge is materialized, mapped where possible, cached per
   exact input, and replayed by conservative current-host transforms
-> a current-host L2 parent proof unlocks its precursor frontier, and the same
   campaign resumes without letting unproved edges spend recursive Agent calls
-> frontier_ledger.v1 projects the complete reachable graph, durable work,
   stock boundaries, exact reaction proof, and dependencies without merging them
-> deterministic verifiers build replayable proof banks for all accepted routes
-> exact edge/stock bindings gate a diverse proof-eligible Top-K portfolio
-> deterministic parent proof emits the final verdict
-> one immutable CAS revision binds consensus, graph/portfolio, frontier ledger,
   parent proof, validated verdict core, forest, HTML, and global dependency view
```

## Why stronger models did not make old runs succeed

The limiting failure was architectural, not token capacity. Earlier runs could
record a plausible disconnection in a caller graph while campaign expansion,
proof stitching, stock closure, and the UI each read a different projection.
Literature discovery did not necessarily materialize an exact reaction row;
Codex precursors could remain prompt text instead of executable frontiers; one
bad sibling could discard several valid hypotheses; failed attempts and
accepted expansions were conflated; and a partially expanded display lane
looked complete. More model calls therefore created more L0 branches without
advancing the weakest reaction edge or leaf.

The mainline fixes this by making one durable canonical graph and frontier
ledger the decision authority. Models propose; the current host materializes,
replays, and promotes exact edges; evidence receipts resume the same campaign;
stock is audited per reachable leaf; attempt and accepted-expansion budgets are
independent; ChemEnzy consumes canonical open frontiers; and every UI stage is
derived from the same queue/proof/stock bindings. Completion is the fixed point
of all selected edges and leaves under the requested objective, never an Agent
return code or a branch count.

The default specialist roles cover target structure, literature, chemoenzymatic
options, and evidence criticism. The coordinator must produce observed
root-thread `spawn_agent` events for every required role. Each spawn prompt has
an explicit role marker; a matching `wait` terminal event and strict child JSON
report are both required. Merely writing role-shaped prose, returning ambiguous
JSON, or completing without an accepted report does not satisfy this contract.

`strict_all` remains the default child policy. For a fresh campaign,
`--codex-agent-team-child-acceptance-mode valid_subset_l0` safely recovers a
host-derived quorum when every role was explicitly spawned but one sibling did
not return a valid final report. It never trusts coordinator-restated
candidates: accepted sibling finals are revalidated and forcibly reduced to
L0/model-only/low, with no authority or solved claim. Timeout, nonzero exit,
tool/runtime/identity failure, missing spawn coverage, and sub-quorum output
still reject the attempt.

Codex can delegate, plan, search, rank, and draft typed artifacts. It cannot
directly mark a case solved, inject raw reactions, write production KB entries,
or promote a child route into a parent solution. All Codex children share one
`codex_model` support group: their agreement raises review coverage, but does
not manufacture independent evidence. DOI/URL/local-document records retain
their own provenance. A fused candidate remains `model_hypothesis` or
`evidence_backed_draft` until a deterministic validator upgrades it; even a
validated candidate is not a solved parent route.

Candidate ranking treats every serialized producer field as observation, not
authority. The producer's channel/evidence/confidence tokens remain visible,
but an unbound candidate is scored as host-derived `model_only`/`low`.
Exact-literature, deterministic computational, and validated ranks appear only
after a trusted source-detail, provider-envelope, or reaction-validation
adapter creates the corresponding private binding.

## Active Entry

```bash
python scripts/run_codex_entry_agentic_blackboard.py \
  --target-name NAME \
  --target-smiles SMILES \
  --codex-agent-team \
  --max-rounds 6 \
  --codex-agent-team-max-depth 6 \
  --codex-agent-team-max-expansions 24 \
  --codex-agent-team-bootstrap-expansions 1 \
  --codex-agent-team-max-expansions-per-invocation 2 \
  --codex-agent-team-max-attempt-runs-per-invocation 4 \
  --codex-agent-team-child-acceptance-mode valid_subset_l0 \
  --codex-action-planner
```

Both Codex switches default to enabled in this launcher. With the agent team
enabled, a rejected Codex action batch fails closed as `stop_unresolved`; no
deterministic scientific planner silently takes ownership. Deterministic code
still performs identity checks, schema validation, stock audit, route
connectivity proof, and final verdict compilation.

Those values are the standard production profile and are now launcher
defaults. `max-expansions=24` is the cumulative count of accepted frontier
expansions; the bootstrap and per-invocation caps prevent the campaign from
running far ahead of evidence, and failed attempts consume the separate
four-attempt cap instead of spending accepted-expansion budget. A later
invocation resumes pending jobs from the same run directory.

Attempts are also bounded across the whole campaign. In the standard profile,
the Python campaign configuration derives `max_attempt_runs=72` from three
times the 24 accepted-expansion ceiling; the four-attempt CLI value remains an
invocation cap. Each Agent call writes an immutable started event before model
work and an immutable terminal event afterward under
`codex_retrosynthesis_team/campaign_attempts/`. Started events count even if a
process exits before writing a terminal event. Accepted-expansion budgets and
attempt budgets live in a monotonic append-only budget-event chain, while
`campaign_state.json` is only a rebuildable compatibility projection.

The campaign runner and proof reconciler hold the same run-directory OS lock
for their complete transaction. This serializes the final accepted-budget
decision, model execution, prepared commit, queue adoption, and proof refresh;
the configurable wait timeout does not permit stale-file lock stealing.

Accepted output uses prepared-result recovery rather than claiming a
cross-file transaction. The team report object and immutable expansion commit
are written before queue completion. On restart, a validated commit can be
adopted by the queue only when campaign, target, job, queue attempt, and lease
digest all match; failed/cancelled terminal jobs stay closed. This prevents the
commit/queue crash window from forcing a duplicate Agent call when the prepared
result is recoverable. The hashes are integrity commitments, not security
signatures or a guarantee against an attacker who controls every local file.

The orchestration team report remains owned by the durable campaign. Controller
refresh and failure diagnostics are written to `controller_projection.json`, so
a late reconciliation failure cannot overwrite an accepted report or erase its
committed expansions. Fused evidence/ChemEnzy projections are combined with
durable commits using a durable-first expansion union; they never become a
second campaign budget or queue.

Controller recovery is also explicit. `blackboard_events/events.jsonl` is
single-writer, expected-head-CAS state with tombstones and an action
`started -> prepared -> committed` outbox. A prepared action result is replayed
without re-running the tool; a started result with unknown outcome is charged
once and blocks automatic retry. Scientific state is removed during recovery
and must be re-established by current-host provider/verifier replay. A single
unterminated crash fragment is preserved in a non-authoritative sidecar and
truncated under lock; terminated corruption, duplicate keys, non-finite JSON,
or identity/digest ambiguity remains fail-closed.

The fused `route_consensus_graph` is now explicitly caller-advisory. Only
`canonical_route_consensus_graph`—fenced Codex commits plus current-host replayed
external admission events—feeds proof state, the frontier ledger, completion,
RouteForest authority stages, and closeout. Newly materialized PDF claims may
enter that graph at L0 through a source/artifact-bound search receipt, but are
reconstructed as `model_only`; L2 still requires reaction replay and L3 still
requires the separately curated precedent registry. This is why a visible
analogy can remain in the suggestion layer without inflating route completion.

External authority is source-replaceable: a host-valid exact-literature or
ChemEnzy admission receipt triggers reconciliation even when the Codex team is
missing or rejected. The campaign may therefore have zero Codex commits while
its canonical graph and ledger contain replayed external edges. Invalid
receipts are quarantined, and a failed reconcile uses an empty identity-bound
authority graph rather than promoting the caller-advisory graph.

By default the launcher also reads `config/trusted_stock_catalogs.json` and
verifies the pinned PaRoutes n1 CSV SHA-256 before execution. This is a
reproducible `benchmark_stock` boundary only. It explicitly makes no commercial
availability, supplier, price, lead-time, or procurement-readiness claim. Use
`--no-codex-agent-team-benchmark-stock` to disable it, or select another
explicitly pinned benchmark with `--benchmark-stock-catalog`.

For a procurement campaign, pass one or more operator-trusted supplier exports
with `--trusted-stock-snapshot PATH`. A file may contain one
`stock_offer_snapshot.v1`, a list, or a `trusted_stock_snapshots.v1` bundle;
every row must include its canonical `snapshot_sha256`. The launcher rejects
`--codex-agent-team-closure-objective procurement` before any Agent call when
no such snapshot is configured. Artifact selection is the trust decision; the
digest is an integrity binding, not a supplier signature or proof that an old
offer remains available.

An exact benchmark hit can terminate a benchmark route, but the scheduler
records proof level 0 and `benchmark_membership_only`; it is never a purchase
claim. Commercial, in-house, and common-commodity boundaries are the only
stock boundary types eligible for level 4 after their own authority replay.

## Active Code Surface

- `scripts/run_codex_entry_agentic_blackboard.py`
- `cascade_planner/harness/agentic_blackboard_controller.py`
- `cascade_planner/harness/agent_action_planner.py`
- `cascade_planner/harness/codex_action_planner.py`
- `cascade_planner/harness/agentic_blackboard.py`
- `cascade_planner/harness/route_objectives.py`
- `cascade_planner/harness/analogical_reaction_templates.py`
- `cascade_planner/harness/parent_route_proof.py`
- `cascade_planner/agent/codex_worker.py`
- `cascade_planner/agent/action_contracts.py`
- `cascade_planner/orchestration/codex_retrosynthesis.py`
- `cascade_planner/harness/codex_edge_verification.py`
- `cascade_planner/baselines/chem_enzy_guidance.py`
- `cascade_planner/routes/consensus.py`
- `cascade_planner/routes/graph.py`
- `cascade_planner/routes/adapters.py`
- `cascade_planner/routes/domain.py`
- `cascade_planner/routes/overlay.py`
- `cascade_planner/providers/`
- `cascade_planner/application/frontier_scheduler.py`
- `cascade_planner/application/frontier_ledger.py`
- `cascade_planner/application/route_portfolio.py`
- `cascade_planner/harness/reaction_step_verifier.py`
- `cascade_planner/runtime/artifact_revision.py`
- `cascade_planner/runtime/`
- `config/trusted_stock_catalogs.json`

## Current Prompt Rule

Codex action planner output must be compact. It should emit action skeletons,
not full downstream policies. Local repair/builders complete:

- source acquisition policy;
- guided ChemEnzy policy;
- child target policy;
- analogical template safety policy;
- stitch/proof payload boundaries.

This avoids structured-output truncation and keeps final authority deterministic.

## Route consensus and display

Equivalent candidates are grouped by stereochemistry-preserving canonical
product and precursor-set identities. The consensus keeps source records,
condition support and conflicts, rejected candidates, and required validation.
Source identity has three host-derived layers. A patent family or canonical
publication identifier defines the independent source group; group plus
content scope defines the logical document; DOI/URL/database and local-PDF
locators are representations of that document. One scholarly source may expose
an article and supporting information as two logical documents in one group,
while a downloaded PDF and its metadata URL remain two representations rather
than two sources. Rendering the article therefore does not mark the SI as
processed, and neither representation nor document count inflates source
independence. Evidence actions can pursue two or three independent groups at
once. Metadata hits trigger acquisition work through HTML/PDF binding,
rendering, visual or deterministic extraction, structure resolution, and exact
reaction rows; later rounds exclude groups already seen.

Numbered compounds are scoped to their extracted source payload. The visual
validator emits `source_compound_label_binding_audit.v1` and rejects every
affected step if one normalized label binds multiple canonical structures in
that payload. It records source group and document ID for diagnosis, but never
assumes that `C16` in two publications means the same molecule.

The route forest consumes this data without copying route-level citations onto
unrelated steps. It chooses `primary_branch_id` from evidence/proof status,
never from the target name, and quarantines proposals from a rejected Codex
team. Every branch declares `solved`, `executable`, `advisory_only`, and
`not_parent_route_proof` explicitly. Its synthesis class (total synthesis,
semisynthesis, biosynthesis, hybrid, or unspecified) is presentation metadata
derived from structured fields and never changes the proof result. Molecule
node IDs are hashes of canonical isomeric SMILES, so equivalent serializations
merge while stereoisomers remain distinct.

Every proof-eligible Top-K portfolio item is projected as an independent
`proof_eligible_portfolio_route` branch and explicit molecule-reaction DAG.
Reaction nodes are branch-specific, while exact canonical molecule nodes remain
shared across branches. Each branch exposes its exact stock leaves, target
alias, weakest proof, correlated support groups, diversity score, and solver or
projection truncation. These branches remain `advisory_only=true`,
`solved=false`, and `not_parent_route_proof=true` until a separate parent proof
is materialized and replayed.

`route_consensus.v1` remains one product's one-step retrosynthetic
neighborhood. `route_consensus_graph.v1` connects independently run
neighborhoods such as `target <- A` and `A <- B`, represents reactions as
precursor hyperedges, records cycles and competing disconnections, and emits
forward-order route hypotheses for display. It never claims stock closure,
solved status, or executability. At closeout, Codex child reports, ChemEnzy
proposals, validated exact-row literature records, templates, and other typed
blackboard channels are adapted back into the canonical consensus. References
remain scoped to the exact step they support.

The V1 graph now carries a content-addressed `route_hypergraph_overlay.v2` and
explicit `route_neighborhood.v2` records. Closeout rebuild groups all accepted
channels by canonical product before fusion, so exact-literature, ChemEnzy, and
Codex records for an intermediate are combined in that intermediate's
neighborhood rather than rejected against the root target. Stable molecule and
hyperedge identities then support AND/OR route closure and safe replacement.
See [Architecture V2](ARCHITECTURE_V2.md#3-per-intermediate-multi-source-reaction-hypergraph-v2)
for the typed records and migration boundary.

## Deterministic proof boundary

The final verdict does not trust a route-shaped summary. A direct parent route
must survive route-verifier replay **and** deterministic validation of every
materialized atom-mapped reaction. Structural graph-and-stock closure is L1
and is no longer sufficient for `solved`; every selected step must reach at
least `L3_precedent_supported` under the current safe parent policy.
Atom-map consistency is emitted as advisory `L2_mapping_consistent`; it cannot
defeat cut/glue counterexamples by itself and is assigned portfolio proof level
zero. The current v3 mapper policy requires complete product-atom provenance
but permits at most twelve reactant heavy atoms to depart the recorded major
product, so ordinary elimination/deprotection is not confused with missing
product atoms or an unlimited atom jump. `L2_reaction_validated` is a distinct
level and may be emitted only when
a trusted deterministic transform is reapplied and its reaction centre
matches; a producer label or a recomputed hash is insufficient. A stitched
route must survive replay of the literature chain and every child closure
embedded in
`proof_inputs` (`stitched_semisynthesis_proof_inputs.v1`). A solved child target,
a high-confidence consensus, or a visually plausible route is therefore still
`child_solved_parent_unresolved` until the complete stock-to-target graph passes
the parent predicate.

An L2-valid, stock-closed stitch remains useful and displayable, but it is not
an L3 parent proof. Literature and every subgoal closure have separate
`precedent_supported` results recomputed from embedded inputs. Missing L3 on
any segment records `reaction_step_precedent_incomplete` and keeps `solved`
false; a validated boolean is never reused as the precedent clause.

Every proof attempt records its missing clauses and open frontiers in
`parent_route_proof_attempt.v1`. This negative proof is retained even when a
structurally plausible parent candidate exists, preventing a child success or
legacy accepted boolean from hiding the exact reaction-proof gap.

Each consensus refresh also writes `codex_edge_verification_report.v1`: every
exact product/precursor edge becomes a reaction candidate, reuses an existing
atom map or requests optional batch mapping, and is replayed by the host verifier.
Mapping consistency alone stays advisory. Only a conservative host transform
whose reaction centre replays may reach `L2_reaction_validated`; generic
cut/glue remains L0/L2-mapping-only. Digest-valid results are persisted in
`reaction_proof_state.json` and supplied to the next bounded campaign resume.
`agent_task_succeeded` and `proof_closed` therefore remain visibly separate.

Recursive expansion is gated by that proof refresh. The campaign root is
eligible immediately; a child precursor is stock-audited and persisted but is
not proposal-expandable until at least one of its exact inbound parent steps
has current-host `L2_reaction_validated` proof. Enabling an existing pending
job is monotonic and requires both the parent step-ID intersection and campaign
identity/root fences. This is an evidence-first spending gate, not a promotion
to L3 parent proof.

The route verifier serializes every accepted materialized route, not only the
best route, into `route_proof_bank.v1`. Each entry binds target identity,
materialized steps, route audit, reaction validation, stock-terminal evidence,
verification policy, and content hash. When several verifier results are
available, the controller treats them as a verifier bundle: every bank entry is
replayed under its own authority before exact product/reactant signatures are
combined. A present but invalid proof bank fails closed rather than falling back
to an unbound legacy best-route summary.

Proof-eligible literature chemistry has a deliberately narrow contract:

- the chain schema is `source_detail_route_chain_audit.v1`;
- every step materializes valid product and reactant structures, declares an
  exact relation, and carries an accepted exact-step validation report;
- materialized evidence is bound to the actual PDF and rendered page by source
  and document identity, manifest/PDF/image SHA-256 values, a real PDF header,
  page number, and a readable page image;
- the reaction digest and that page binding must also appear in an approved
  out-of-band `trusted_literature_step_registry.v1` entry whose authority is a
  human curator or deterministic structure parser.

The registry location comes only from
`AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY`, or from the package default at
`cascade_planner/harness/data/trusted_literature_step_registry.json`. A proof
payload cannot select its own registry. The packaged registry is intentionally
empty, so a PDF manifest or a Codex claim cannot approve arbitrary chemistry.
The populated registry under `tests/fixtures/` is test data and is not a
production trust source.

A strict literature reaction can have several reactant frontiers. The
blackboard promotes all terminal frontiers, and a stitch is attempted only
after every one has an exact matching, independently verified route expansion.
Missing one co-reactant closure rejects the entire stitch. Model-authored
parent-bridge prose, exploratory visual chains, and analogical templates remain
advisory.

The display follows the same boundary. A `stitched_verified_route` is rebuilt
only from an accepted parent proof's revalidated `proof_inputs`; loose
top-level route fields cannot inject it. `direct_verified_route` and
`stitched_verified_route` are the only solved parent branches. Child closures,
proof-eligible portfolio DAGs, consensus DAGs, visual chains, process evidence,
and rejected-team proposals are labelled or quarantined as advisory.

## Stock and route-verifier boundary

Every accepted ChemEnzy route is materialized and reverified against the exact
requested target. The verifier checks connectivity, cycles, duplicate-fragment
padding, hidden non-stock reactants, elemental inventory, and implausible
largest-precursor jumps. A large convergent assembly is exempted only when a
complete, step-bound atom mapping has unique heavy-atom maps, conserves element
identity, and proves a new bond between distinct reactant components;
self-reported convergence flags are ignored.

Stock closure is also recomputed rather than inherited from a backend flag.
The effective catalog names and bindings are taken from the request, resolved
against the ChemEnzy configuration, and checked against the actual catalog
path, size, and SHA-256 before exact canonical-SMILES lookup. A broader Zinc
file cannot silently substitute for a requested PaRoutes catalog. Small common
commodities use the separate, explicitly identified
`autoplanner_common_commodity.v1` supplement.

## Durable frontier and portfolio boundary

Each recursive molecule frontier passes through the persistent stock-first
scheduler before agent work. Jobs are content-identifiable and idempotent, use
leases and heartbeats, recover expired work with bounded retry, and retain
failure reasons. Only a construction-time trusted snapshot whose canonical
content, SHA-256, molecule, boolean availability, and timezone-qualified check
time all replay can close a terminal immediately;
proposal expansion remains proof level zero. The Codex campaign queue persists
`proposal_expansion` work only. Reaction replay and durable proof-state refresh
run between bounded campaign invocations and can guide a resume, while
proof-bank construction, portfolio binding/solving, replacement validation,
parent proof, and CAS publication remain deterministic downstream stages, not
additional successful proposal jobs. Queue exhaustion therefore means only
that no proposal job is currently claimable; it is never route or proof
completion. Depth and cycle boundary leaves are stock-audited before they are
closed as non-expandable.

After every graph/proof refresh, the controller writes
`frontier_ledger.json` (`frontier_ledger.v1`). This is the single completion
projection across the campaign, blackboard, hypergraph, proof refresh, and
frontier queue; it is not a second queue and it does not mutate any source.
Proposal existence, work state, stock authority, reaction proof, and
dependencies remain separate subrecords. The ledger walks every
target-reachable step in `route_consensus_graph.v1`, never the bounded
`route_hypotheses` display list, then solves four provider-replayed fixed
points:

- `any_benchmark_route_closed`: at least one proven alternative closes on the
  search-boundary plane, which may include pinned benchmark membership;
- `all_explored_benchmark_closed`: every reachable alternative and leaf closes
  on that search-boundary plane;
- `any_procurement_route_closed`: at least one proven alternative closes using
  only replayed commercial/in-house/common-commodity leaves at stock level 4;
- `all_explored_procurement_closed`: every reachable alternative and leaf
  closes on that procurement-boundary plane.

The compatibility names `any_route_closed` and
`all_explored_graph_closed` alias the first pair. The procurement pair changes
the leaf authority, not the configured reaction-proof floor, so it does not by
itself assert that all selected reactions are L4.

The same summary reports proposal-pending, expansion-eligible, work-pending,
stock-pending, reaction-proof-pending, and dependency-pending counts
independently. In particular, an unexpanded graph frontier can remain visible
while no Codex proposal job is eligible; this is not a contradiction and must
not trigger an otherwise pointless agent call.

Consumers must validate the ledger schema and `content_sha256`, require valid
graph/queue/reaction-proof inputs, and verify that the root summary matches the
root molecule row. Because a content hash is not a signature, the validator
also reconstructs exact-edge topology and recomputes all four fixed points. A
positive stock leaf additionally requires replay through the construction-time
trusted provider instance; validating the embedded envelope alone is not
enough. If any authority envelope is invalid, positive closure is ignored.
This is how a damaged queue snapshot, rewritten proof record, re-hashed closure boolean,
truncated display projection, or stale blackboard summary fails closed instead
of becoming apparent progress.

The V2 overlay is solved as an AND/OR graph. One disconnection is selected per
product (OR), while every precursor of that reaction must close (AND). Only
stock-closed routes whose selected edges have digest-valid exact proof bindings
and whose leaves have digest-valid exact stock bindings enter the Top-K analysis
portfolio. `L2_mapping_consistent` never passes this gate;
`L2_reaction_validated` passes only through trusted deterministic replay, while
parent authority requires every reaction at L3 or L4. Maximal marginal
relevance favors distinct valid routes and never pads K with advisory
candidates.

Replacement discovery fixes an exact product's selected hyperedge and calls the
backend AND/OR solver again. The replacement may intentionally introduce a
different precursor set; those new precursors must close under the same stock
and proof gates. Accepted and rejected rows are retained in
`route_replacement_catalog.v1`. The UI previews the complete accepted
revalidated branch, never a single-step splice, and keeps rejected candidates
visible with their reasons.

## Closeout and global DAG boundary

The route forest is a complete molecule-reaction bipartite dependency
projection with explicit edges. A closed selected route is a DAG; cycles in the
full explored proposal overlay are shown explicitly. Its trust vector separates
identity, connectivity, source independence, stock, conditions, and forward
feasibility. Colour is the proof tier; width is independent support-group count;
opacity is mean trust; pattern exposes uncertainty. The canonical forest is
untruncated by default, while the initial viewport may collapse lower-value
branches under the recorded display policy. Any explicit producer limit records
omitted counts and displays a warning. The emitted legend maps rejected L0 to
rose, advisory L0 to orange,
materialized L0 to violet, graph-and-stock L1 to amber, mapping-only L2 to blue-grey,
deterministically replayed L2 to blue, precedent-supported L3 to teal, and
procurement-ready L4 to green. The JSON legend, not colour alone, is the
semantic authority.

The presentation path is split into deterministic layers:

```text
explored_route_forest.v1 (complete authority)
-> route_forest_delivery.v1 (canonical source SHA-256 binding)
-> SCC/component/layer and branch-lane logical projections
-> repository-native HTML/CSS/SVG workbench
```

The delivery projection retains every branch, reaction step, explicit edge,
authoritative replacement record, evidence link, and trust semantic. It omits
only browser-unused duplicate molecule SVGs and the quadratic records inside
diagnostics-only interface comparisons; their counts and omission reason remain
visible. History and closeout readers validate both the delivery digest and its
source-forest digest rather than treating the compact payload as a new proof
artifact.

The standalone workbench recomputes the embedded-byte SHA-256 before reporting
the delivery as bound. The `/agent` parent accepts the child ready handshake only
when that browser verification returns `verified`; `pending`, `unavailable`,
unknown, and invalid states all fail closed.

The default route-cluster mode prioritizes a compact Top-K of high-value,
structurally distinct branches and groups or collapses the remainder; the full
forest remains available through expansion and filters. Shared-hypergraph mode
displays canonical sharing, and current-branch mode displays one complete
molecule-reaction DAG. Search, branch kind, proof tier, edge scope, density,
orientation, labels, pan/zoom, fit, reset, and minimap controls change only the
view. Proof colours are never overwritten by selection or task state. The UI
labels task completion, L0 materialization, deterministic L2, stock closure,
portfolio eligibility, and complete parent-route proof independently.
Resizable desktop panes, medium-screen drawers, mobile bottom navigation, and
`embed=1` canvas-first rendering share the same read-only artifact.

The stage switcher is also fail-closed. “Break suggestions” excludes rejected
diagnostics; “fully expanded” requires every nonempty displayed step to bind a
current succeeded proposal-expansion queue job. Partial i/N expansion is shown
as non-authoritative progress and is excluded from that filter. “Reaction
validated” requires every displayed step to bind current-host
L2-R or stronger; and “stock closed” requires current provider replay for every
nonempty synthesis leaf. The stock badge distinguishes benchmark/search closure
from procurement closure. Legacy payloads do not infer these states from proof
colour, and a zero-result filter clears the canvas and inspector instead of
leaving stale content visible.

The operator summary keeps six statements independent: the four ledger fixed
points above, deterministic `L3 parent route solved`, and `L4 procurement route
ready`. An L2 ledger may close either stock plane without granting L3 parent
authority; one L3 parent route can solve without exhausting every advisory
alternative; procurement-capable leaves do not make L2 reactions L4; and an L3
route is not L4 procurement-ready. Projection coverage and branch count are
view diagnostics, not synonyms for chemistry completion.

For L4, the backend audit also rechecks every selected route leaf. A benchmark
catalog binding may support benchmark closure but fails procurement readiness;
each leaf must instead be a validated commercial, in-house, or common-commodity
boundary, and every selected reaction must retain L4 conditions/procurement
authority.

At closeout, consensus, graph/portfolio, frontier ledger, forest, and HTML bytes
are stored in a content-addressed revision with dependency SHA-256 bindings. A
staging manifest is validated before an immutable committed manifest is
written, and only then is the small `latest.json` pointer atomically replaced.
A failed or drifting revision does not replace the previous pointer and its
fixed-name projection is not treated as current truth.

## ChemEnzy Boundary

Simple targets may run immediate baseline Chemenzy. Complex steroid,
polycyclic, or natural-product-like targets require blackboard signal before a
full guided rerun. A first-round complex target probe is allowed only when
explicitly bounded as an initial probe.

Guidance is executable, not decorative: policy precursor hints compile into a
typed guidance contract consumed by the native one-step wrapper, which adjusts
candidate costs/ranking and hard-rejects self-loops, ancestor cycles, obvious
element deficits, implausible heavy-atom jumps, and disallowed terminal
candidates. Trusted stock terminals are preserved. The run emits consumption
counts, cost changes, accepted rows, and exact rejection reasons. These changes
improve proposal quality but do not inject raw reactions or bypass deterministic
reaction/route verification.

Legacy molecule-specific semisynthesis rescue tables are not part of the
generic mainline. `run_chem_enzy_plan_for_web.py` loads them only when the
request explicitly sets `enable_semisynthesis_rescue=true`; the one-step
provider likewise requires
`AUTOPLANNER_ENABLE_SEMISYNTHESIS_RESCUE_PROPOSALS`. Both defaults are off.

## Paclitaxel end-to-end run (2026-07-10)

The current end-to-end replay is
`results/shared/paclitaxel_architecture_v2_20260710`. It is a negative
scientific result and a positive architecture exercise; those two statements
must not be collapsed. The run used the exact stereochemical paclitaxel
identity `RCINICONZNJXQF-MZXODVADSA-N` and five local source documents: the
Holton article and supporting information as distinct documents, Danishefsky's
synthesis, a semisynthesis article, and the Baloglu thesis.

The root coordinator ran on `gpt-5.5`, directly spawned the four required
specialists, and accepted all four completed child reports with a consistent
runtime event trace. Seven planner rounds emitted 14 approved actions. The
first action-planner attempt correctly failed closed after inheriting an
ambient `gpt-5.6-sol` that the installed Codex CLI could not run. The controller
now resolves the action-planner model explicitly from
`AUTOPLANNER_CODEX_ACTION_PLANNER_MODEL`, then `AUTOPLANNER_CODEX_MODEL`, then
the `gpt-5.5` default; rounds 2 through 7 were resumed with `gpt-5.5`. Round 3's
model output requested a disallowed tool and was rejected; the bounded policy
fallback selected only already-authorized actions and gained no scientific or
solved authority.

The migrated durable campaign contains 13 proposal-expansion jobs: 10
succeeded and 3 remain pending, with zero stock-closed jobs and
`proposal_graph_exhausted=false`. A successful proposal job is still L0, so
these counts are execution progress rather than route closure. Evidence work
rendered 145 PDF pages, accepted three PDF-structure evidence artifacts and two
exploratory visual chains, and attempted three named-structure resolutions;
one structure resolved. There were no trusted exact literature rows
(`exact_rows=0`) and the guided verifier materialized no accepted route.

The per-intermediate V2 overlay nevertheless demonstrates the intended
multi-source graph architecture: 85 molecules, 82 reaction hyperedges, 10
route neighborhoods, 24 route variants, 9 alternative sets, and 5 hyperedges
with true independent multi-source support. This is multi-source proposal and
evidence fusion, not a complete multi-source verified route. The untruncated
forest projects 96 advisory branches as 83 canonical molecule nodes, 122
branch-specific reaction nodes, and 337 dependency edges. All observed reaction
tiers remain L0 (22 advisory and 100 materialized); no selected portfolio route
exists from which to materialize a route DAG.

The AND/OR solver therefore returns an honest empty portfolio with reason
`no_stock_closed_reaction_validated_route`. It cannot offer two distinct valid
routes or a backend-revalidated replacement, and the UI does not fabricate
either. Closeout still succeeds operationally: one drift-free immutable CAS
revision binds seven graph, proof-snapshot, verdict, forest, and HTML artifacts.
Its revision is
`sha256:1805d468536eb968acb6d63eee5b985ab146a2675bd8198767300d04923a9faa`.
The refreshed diagnostic chain also passes all 13/13 capability requirements:
the blackboard snapshot, capability audit, and run audit were rebuilt from the
saved actions and tool trace rather than retaining stale closeout diagnostics.

It did **not** solve paclitaxel. The authoritative CAS-bound verdict is:

```json
{
  "verdict": "hypothesis_route_proposed",
  "route_status": "hypothesis_route_execution_partial",
  "solved": false,
  "stock_audit_passed": false,
  "reasons": [
    "hypothesis_only_retrosynthesis_available",
    "no_deterministic_parent_route_proof"
  ]
}
```

`scripts/audit_architecture_v2.py` intentionally separates implementation
surface from materialized contracts and chemistry acceptance:

| Audit surface | Result | Interpretation |
| --- | --- | --- |
| Declared capability surface | 9/9, 100% | All audited architecture mechanisms exist; this is not a completion claim. |
| Executable contracts in this run | 6/10, 60% | Team, scheduler, V2 overlay, dependency graph, CAS, and empty-portfolio contracts replay; proof-dependent outputs are absent or ineligible. |
| Run acceptance | 10/17, 58.8% | Identity, direct child team, source correlation, true multi-source fusion, global projection, and CAS gates pass; chemistry closure does not. |

The seven open run gates are exact: close every selected frontier under proof,
materialize a proof-eligible portfolio, demonstrate acyclic selected route DAGs,
project those portfolio routes, retain at least two distinct valid alternatives,
produce a backend-revalidated full-route replacement, and pass deterministic
parent-route proof. In chemical terms, the known taxane/side-chain literature
bridge still needs exact page-bound structures and reactions in the trusted L3
registry, every convergent precursor needs verified stock closure, and every
selected step must replay at L3 or L4. Until then, the 96-branch display is a
complete view of explored hypotheses, not a completed synthesis route.

Run artifacts under `results/shared/` are intentionally git-ignored and are
available only in the workspace where the replay was executed.

Rebuild and re-audit the current run without rerunning expensive model or
literature tools:

```powershell
python scripts\refresh_agentic_closeout_artifacts.py `
  results\shared\paclitaxel_architecture_v2_20260710
python scripts\evaluate_agentic_run.py `
  results\shared\paclitaxel_architecture_v2_20260710 `
  --output results\shared\paclitaxel_architecture_v2_20260710\evaluation.json `
  --human
python scripts\audit_architecture_v2.py `
  results\shared\paclitaxel_architecture_v2_20260710 `
  --output results\shared\paclitaxel_architecture_v2_20260710\architecture_v2_audit.json `
  --human
```

## Nirmatrelvir recovery status (2026-07-12)

`results/shared/nirmatrelvir_codex_closure_20260712_v5` completed four evidence
rounds and eight Agent attempts. A controller projection bug then overwrote the
fixed-name team report with a failed reconciliation view, making the final
21-node/15-edge graph look as though the accepted campaign work had vanished.
Replaying the five immutable, queue-fenced expansion commits recovers the
durable campaign; unioning those expansions with the source/ChemEnzy controller
projection produces 44 molecules and 38 reaction edges. The recovery does not
invoke a model or consume either budget.

Current-host materialization and mapping cover all 38 recovered edges. Thirteen
reach deterministic L2 reaction validation; 25 remain rejected/open. The
policy-pinned PaRoutes benchmark provider rehydrates with its exact artifact
hash and closes three molecule boundaries as benchmark membership only. The
authoritative ledger still reports every route fixed point false: there is no
complete benchmark route, no procurement route, no L3 parent proof, and no L4
route. Queue state is 5 succeeded, 32 pending, and 3 retry-wait jobs, so the
campaign is resumable and `proposal_graph_exhausted=false`.

The authority-backed recovered workbench renders 70 overlapping branch views:
24 bind at least one actual succeeded expansion, five have every displayed step
at current-host L2 reaction validation or stronger, and zero have every
synthesis leaf stock-closed. These are useful stage views, not 70 independent
routes and not a completion percentage.

The run retained six source records and rendered/extracted patent, Science,
and Nature Communications material, but its stored blackboard has no formally
compiled exact reaction rows. Visual/PDF extraction and model citations are
not substitutes for exact source-detail bindings. The next chemistry work is
therefore precise rather than “use a larger model”: compile the patent and
article/SI endpoint reactions as reaction-complete exact rows, let those rows
unlock only their matching L2 child frontiers, audit every new leaf against a
real procurement snapshot as well as the benchmark catalog, and continue until
the ledger fixed point closes or the explicit 24-expansion/72-attempt limits are
reached. The v5 recovery is strong evidence that durability and control flow now
work; it is deliberately not reported as a solved Nirmatrelvir synthesis.

## Supporting Docs

- [Architecture V2](ARCHITECTURE_V2.md)
- [Complex-molecule showcase policy](SHOWCASE_CASES.md)
- `docs/AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md`
- `docs/CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md`
- `docs/archive/2026-06/legacy_codex_entry_fullflow/`

Older fixed-chain fullflow and SMILES-first documents are historical context
only.

## Minimum Verification

Before pushing controller or prompt edits, run:

```bash
python -m pytest tests/test_agentic_blackboard_controller.py -q
python -m pytest tests/test_codex_entry_harness_contract.py -q
python -m pytest tests/test_codex_retrosynthesis_team.py tests/test_route_consensus.py tests/test_route_consensus_graph.py tests/test_route_source_adapters.py tests/test_route_forest.py tests/test_route_forest_delivery.py tests/test_route_forest_layout.py tests/test_route_forest_history_smoke.py -q
```

When touching proof, stock, objective, or template surfaces, also run:

```bash
python -m pytest tests/test_route_objectives.py tests/test_parent_route_proof.py tests/test_route_verifier.py tests/test_chemenzy_semisynthesis_rescue_policy.py -q
```

For the V2 provider, graph, proof, scheduler, portfolio, artifact, and global
DAG contracts, also run:

```bash
python -m pytest tests/test_provider_registry.py tests/test_builtin_providers.py tests/test_stock_provider.py tests/test_route_source_adapters.py tests/test_reaction_step_verifier.py tests/test_frontier_scheduler.py tests/test_frontier_ledger.py tests/test_route_portfolio.py tests/test_portfolio_controller_integration.py tests/test_artifact_revision.py tests/test_route_forest.py tests/test_route_forest_delivery.py tests/test_route_forest_layout.py tests/test_route_forest_history_smoke.py tests/test_web_app.py tests/test_audit_architecture_v2.py -q
```

The full-suite literature proof fixtures require the fixture registry explicitly;
do not copy it into the package default. PowerShell:

```powershell
$env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY = 'tests/fixtures/trusted_literature_step_registry.json'
python -m pytest -q
Remove-Item Env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY
```

Historical suite counts are not release evidence for the current P0 changes.
Run the complete offline suite again immediately before the direct push and
record its actual result in the handoff. Live retrieval is intentionally
opt-in; set `AUTOPLANNER_LIVE_RETRIEVAL_SMOKE=1` when external
PubChem/Crossref availability is part of the test objective. Also run
`git diff --check` before publishing. This repository intentionally has no
GitHub Actions workflow; local verification is the release gate.

For a completed run, create a machine-readable quality summary with:

```bash
python scripts/evaluate_agentic_run.py PATH_TO_RUN
```

To keep repository capability completion separate from the chemistry result of
one run, also generate the V2 gate report:

```bash
python scripts/audit_architecture_v2.py PATH_TO_RUN --human \
  --output PATH_TO_RUN/architecture_v2_audit.json
```

The current audit compatibility summary prints four independent high-level
facts: benchmark/search any-route, benchmark/search all-explored, L3-parent,
and all-L4 procurement readiness. The route forest and ledger additionally
retain both procurement-boundary fixed points; do not confuse them with the
stricter all-L4 portfolio audit. The JSON also reports ledger schema validity,
canonical digest validity, per-input authority, producer fail-closed behavior,
claimed closure, and effective closure. Missing or invalid
`frontier_ledger.json` always yields false effective closure even when an older
campaign summary says `complete=true`. The L4 audit rejects otherwise valid
routes whose only leaf authority is benchmark membership.
