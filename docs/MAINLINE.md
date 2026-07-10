# AutoPlanner Mainline

Last update: 2026-07-10.

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
-> route consensus fuses every canonical product intermediate independently
-> stock-first persistent frontiers launch bounded direct Codex child teams
-> frontier neighborhoods assemble into the V2 reaction-hypergraph overlay
-> Codex chooses a compact typed action batch
-> deterministic validator checks safety, budget, binding, and proof boundaries
-> local tools execute approved actions
-> blackboard records typed summaries and artifact refs
-> deterministic verifiers build replayable proof banks for all accepted routes
-> exact edge/stock bindings gate a diverse proof-eligible Top-K portfolio
-> deterministic parent proof emits the final verdict
-> one immutable CAS revision binds consensus, graph/portfolio, parent proof,
   validated verdict core, forest, HTML, and global dependency view
```

The default specialist roles cover target structure, literature, chemoenzymatic
options, and evidence criticism. The coordinator must produce observed
root-thread `spawn_agent` events for every required role. Each spawn prompt has
an explicit role marker; a matching `wait` terminal event and strict child JSON
report are both required. Merely writing role-shaped prose, returning ambiguous
JSON, or completing without an accepted report does not satisfy this contract.

Codex can delegate, plan, search, rank, and draft typed artifacts. It cannot
directly mark a case solved, inject raw reactions, write production KB entries,
or promote a child route into a parent solution. All Codex children share one
`codex_model` support group: their agreement raises review coverage, but does
not manufacture independent evidence. DOI/URL/local-document records retain
their own provenance. A fused candidate remains `model_hypothesis` or
`evidence_backed_draft` until a deterministic validator upgrades it; even a
validated candidate is not a solved parent route.

## Active Entry

```bash
python scripts/run_codex_entry_agentic_blackboard.py \
  --target-name NAME \
  --target-smiles SMILES \
  --codex-agent-team \
  --codex-agent-team-max-depth 2 \
  --codex-agent-team-max-expansions 4 \
  --codex-action-planner
```

Both Codex switches default to enabled in this launcher. With the agent team
enabled, a rejected Codex action batch fails closed as `stop_unresolved`; no
deterministic scientific planner silently takes ownership. Deterministic code
still performs identity checks, schema validation, stock audit, route
connectivity proof, and final verdict compilation.

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
- `cascade_planner/routes/consensus.py`
- `cascade_planner/routes/graph.py`
- `cascade_planner/routes/adapters.py`
- `cascade_planner/routes/domain.py`
- `cascade_planner/routes/overlay.py`
- `cascade_planner/providers/`
- `cascade_planner/application/frontier_scheduler.py`
- `cascade_planner/application/route_portfolio.py`
- `cascade_planner/harness/reaction_step_verifier.py`
- `cascade_planner/runtime/artifact_revision.py`
- `cascade_planner/runtime/`

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
One scholarly source may expose several extractable documents (for example an
article and its supporting information). These keep one source identity but
distinct document/PDF identities, so rendering the article does not silently
mark the SI as processed.

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
zero. `L2_reaction_validated` is a distinct level and may be emitted only when
a trusted deterministic transform is reapplied and its reaction centre
matches; a producer label or a recomputed hash is insufficient. A stitched
route must survive replay of the literature chain and every child closure
embedded in
`proof_inputs` (`stitched_semisynthesis_proof_inputs.v1`). A solved child target,
a high-confidence consensus, or a visually plausible route is therefore still
`child_solved_parent_unresolved` until the complete stock-to-target graph passes
the parent predicate.

Every proof attempt records its missing clauses and open frontiers in
`parent_route_proof_attempt.v1`. This negative proof is retained even when a
structurally plausible parent candidate exists, preventing a child success or
legacy accepted boolean from hiding the exact reaction-proof gap.

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
`proposal_expansion` work only. Reaction replay, proof-bank construction,
portfolio binding/solving, replacement validation, parent proof, and CAS
publication are downstream deterministic stages, not additional proof jobs in
the same queue. Queue exhaustion therefore means only that no proposal job is
currently claimable; it is never route or proof completion.

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
opacity is mean trust; pattern exposes uncertainty. The default view is
untruncated, and any explicit limit records omitted counts and displays a
warning. The emitted legend maps rejected L0 to rose, advisory/materialized L0
to orange, graph-and-stock L1 to amber, mapping-only L2 to blue-grey,
deterministically replayed L2 to blue, precedent-supported L3 to teal, and
procurement-ready L4 to green. The JSON legend, not colour alone, is the
semantic authority.

At closeout, consensus, graph/portfolio, forest, and HTML bytes are stored in a
content-addressed revision with dependency SHA-256 bindings. A staging manifest
is validated before an immutable committed manifest is written, and only then
is the small `latest.json` pointer atomically replaced. A failed or drifting
revision does not replace the previous pointer and its fixed-name projection is
not treated as current truth.

## ChemEnzy Boundary

Simple targets may run immediate baseline Chemenzy. Complex steroid,
polycyclic, or natural-product-like targets require blackboard signal before a
full guided rerun. A first-round complex target probe is allowed only when
explicitly bounded as an initial probe.

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
`sha256:3f5e81b19ebc9db5d67fb68c9a679a532d2e4219ad1c5c79f050c53d5d72946e`.
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

## Supporting Docs

- [Architecture V2](ARCHITECTURE_V2.md)
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
python -m pytest tests/test_codex_retrosynthesis_team.py tests/test_route_consensus.py tests/test_route_consensus_graph.py tests/test_route_source_adapters.py tests/test_route_forest.py -q
```

When touching proof, stock, objective, or template surfaces, also run:

```bash
python -m pytest tests/test_route_objectives.py tests/test_parent_route_proof.py tests/test_route_verifier.py tests/test_chemenzy_semisynthesis_rescue_policy.py -q
```

For the V2 provider, graph, proof, scheduler, portfolio, artifact, and global
DAG contracts, also run:

```bash
python -m pytest tests/test_provider_registry.py tests/test_builtin_providers.py tests/test_stock_provider.py tests/test_route_source_adapters.py tests/test_reaction_step_verifier.py tests/test_frontier_scheduler.py tests/test_route_portfolio.py tests/test_portfolio_controller_integration.py tests/test_artifact_revision.py tests/test_route_forest.py tests/test_web_app.py tests/test_audit_architecture_v2.py -q
```

The full-suite literature proof fixtures require the fixture registry explicitly;
do not copy it into the package default. PowerShell:

```powershell
$env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY = 'tests/fixtures/trusted_literature_step_registry.json'
python -m pytest -q
Remove-Item Env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY
```

At the Architecture V2 closeout on 2026-07-10, the complete offline suite
passed with 829 passed and 3 skipped. Live retrieval is intentionally opt-in; set
`AUTOPLANNER_LIVE_RETRIEVAL_SMOKE=1` when external PubChem/Crossref availability
is part of the test objective. Also run `git diff --check` before publishing.

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
