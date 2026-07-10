# AutoPlanner Mainline

Last update: 2026-07-10.

The active mainline is the policy-driven agentic blackboard controller.

```text
target input
-> deterministic preflight
-> blackboard state
-> Codex coordinator calls spawn_agent for each specialist role
-> child reports return typed RetrosynthesisProposalReport artifacts
-> route consensus canonicalizes identity and fuses provenance/conflicts
-> unexpanded precursor frontiers launch another direct Codex specialist team
-> frontier-specific consensuses assemble into route_consensus_graph.v1
-> Codex chooses a compact typed action batch
-> deterministic validator checks safety, budget, binding, and proof boundaries
-> local tools execute approved actions
-> blackboard records typed summaries and artifact refs
-> deterministic parent proof emits the final verdict
-> route forest renders the integrated route and competing hypotheses
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

`route_consensus.v1` remains one product's one-step retrosynthetic
neighborhood. `route_consensus_graph.v1` connects independently run
neighborhoods such as `target <- A` and `A <- B`, represents reactions as
precursor hyperedges, records cycles and competing disconnections, and emits
forward-order route hypotheses for display. It never claims stock closure,
solved status, or executability. At closeout, Codex child reports, ChemEnzy
proposals, validated exact-row literature records, templates, and other typed
blackboard channels are adapted back into the canonical consensus. References
remain scoped to the exact step they support.

## Deterministic proof boundary

The final verdict does not trust a route-shaped summary. A direct parent route
must survive route-verifier replay. A stitched route must survive replay of the
literature chain and every child closure embedded in
`proof_inputs` (`stitched_semisynthesis_proof_inputs.v1`). A solved child target,
a high-confidence consensus, or a visually plausible route is therefore still
`child_solved_parent_unresolved` until the complete stock-to-target graph passes
the parent predicate.

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
consensus DAGs, visual chains, process evidence, and rejected-team proposals are
labelled or quarantined as advisory.

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

The improved run used exact paclitaxel identity
`RCINICONZNJXQF-MZXODVADSA-N`, model `gpt-5.5`, eight planner rounds, a maximum
team depth of two, and five local literature documents supplied as input. The
source set included the Holton article and its supporting information as
distinct documents, Danishefsky's synthesis, a semisynthesis paper, and the
Baloglu thesis. The run exposed and the final code fixes a later merge bug that
could collapse same-DOI article/SI candidates after online scouting; document
path/ID now precedes DOI in blackboard merge and lifecycle identity.

The run materially improved orchestration and evidence coverage over the
baseline: the root Codex team was accepted with all four required child reports,
bounded recursive frontier teams ran, all eight action batches passed
validation, two exploratory visual chains were accepted, and two child-target
searches produced verifier-accepted stock closures. The route forest grew from
13 to 42 branches with explicit truth fields on every branch, and the rejected
baseline team's consensus no longer leaked into the candidate display.

The two accepted closures were strictly scoped child routes: compound 10 had
15 accepted routes out of 93 (best route 3 steps), and compound 6 had 1,474 out
of 2,612 (best route 5 steps). They cover an upstream side-chain sequence, not
the taxane-to-paclitaxel parent route. A post-run deterministic projection replay
also confirmed that internal proposal/template products remain attached to
their own intermediates rather than being drawn as direct arrows to the target.

It did **not** solve paclitaxel. The authoritative outcome in
`results/shared/paclitaxel_codex_improved_20260710/final_verdict.json` is:

```json
{
  "verdict": "unresolved",
  "route_status": "child_solved_parent_unresolved",
  "solved": false,
  "stock_audit_passed": false,
  "reasons": ["child_target_solved_parent_proof_missing"]
}
```

The target-side guided probe returned ten candidates and the verifier accepted
none: all ten violated the large-atom-jump and element-inventory gates. No
strict trusted exact literature row was available (`exact_rows=0`), hence no
complete literature-to-child closure could become a parent proof. The displayed
primary branch is a `subgoal_verified_route` and remains
`advisory_only=true`, `solved=false`; there is no
`stitched_verified_route`, direct verified parent route, or strictly usable
solved branch. This is the intended fail-closed result, not a successful total
or semisynthesis claim.

The remaining literature bridge is concrete. The semisynthesis paper's Scheme
2 shows side-chain acid 11 plus 7-triethylsilylbaccatin III (5) proceeding via
taxane ester 12 to paclitaxel (1), but the accepted exploratory extraction only
captured the earlier 6 -> 7/8 -> 10 -> 11 sequence. The original run then used
a fuzzy target-identity shortcut for the label "paclitaxel derivative 12".
Final code requires an exact target name/explicit alias (optionally with a
terminal compound number), excludes both invalid shortcut records from the
evaluator, and keeps Scheme 2 plus stock closure of precursor 5 as unresolved
work. The final evaluator therefore reports
`invalid_target_identity_shortcut_excluded` and
`route_forest_primary_is_advisory` instead of treating either artifact as
positive evidence.

Run artifacts under `results/shared/` are intentionally git-ignored and are
available only in the workspace where the replay was executed.

Recompute the baseline comparison without rerunning expensive tools:

```powershell
python scripts\evaluate_agentic_run.py `
  results\shared\paclitaxel_codex_baseline_20260710 `
  --compare-to results\shared\paclitaxel_codex_improved_20260710 `
  --output results\shared\paclitaxel_codex_improved_20260710\evaluation_vs_baseline.json `
  --human
```

## Supporting Docs

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

The full-suite literature proof fixtures require the fixture registry explicitly;
do not copy it into the package default. PowerShell:

```powershell
$env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY = 'tests/fixtures/trusted_literature_step_registry.json'
python -m pytest -q
Remove-Item Env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY
```

At the 2026-07-10 closeout, the clean 98-file publication projection completed
with 701 passed and 5 skipped. Live retrieval is intentionally opt-in; set
`AUTOPLANNER_LIVE_RETRIEVAL_SMOKE=1` when external PubChem/Crossref availability
is part of the test objective. Also run `git diff --check` before publishing.

For a completed run, create a machine-readable quality summary with:

```bash
python scripts/evaluate_agentic_run.py PATH_TO_RUN
```
