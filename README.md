# AutoPlanner

AutoPlanner is being refocused around a policy-driven Codex + blackboard route
controller for chemoenzymatic retrosynthesis.

The active architecture is:

```text
target input
-> deterministic preflight
-> agentic blackboard state
-> Codex coordinator directly spawns independent specialist child agents
-> each child report is independently parsed, role-bound, and validated
-> typed Provider SPI envelopes bind schema/version/hash; host policy binds correlation and authority
-> stock-first durable frontiers launch additional direct child-agent teams
-> every intermediate fuses its own multi-source reaction neighborhood
-> typed reaction hyperedges assemble into a provenance-preserving global graph
-> Codex chooses compact typed follow-up actions
-> deterministic validators check safety, budgets, source binding, and proof boundaries
-> local tools execute approved actions
-> deterministic route verifiers preserve every accepted route in a replayable proof bank
-> exact edge/stock bindings gate AND/OR closure and a diverse valid portfolio
-> deterministic parent proof decides final verdict
-> immutable closeout binds the trust-coloured full route graph and alternatives
```

Codex is allowed to delegate, plan, search, classify sources, and draft route
candidates. Multiple Codex roles are one correlated model source, not multiple
independent evidence sources. Codex cannot directly mark a route solved, inject
raw reaction SMILES into production, or promote artifacts to the production KB.
When the multi-agent mainline fails, the run stops unresolved instead of
silently replacing it with a deterministic scientific planner.

The proof boundary is fail-closed. `route_proof_bank.v1` retains every
accepted, materialized verifier route instead of only the best route; a bundle
of verifier results keeps each bank and authority separate while exact
structure signatures are combined. Portfolio admission requires digest-valid
`exact_edge_proof_binding.v1` and `exact_stock_binding.v1` records. Mapping-only
`L2_mapping_consistent` is advisory and never enters the portfolio.
`L2_reaction_validated` is reserved for a trusted deterministic transform
reapply/reaction-centre replay; current parent authority still requires trusted
exact precedent at L3 or L4 for every reaction. A literature stitch
additionally requires exact step chemistry bound to a real
PDF page and an out-of-band trusted registry, plus independently verified stock
closure for every terminal reactant frontier. The packaged registry is empty by
default. Consensus graphs, visual extractions, solved child targets, and
structurally closed but reaction-unvalidated routes remain advisory until those
parent-proof conditions are met.

The durable Codex campaign queue persists stock-first **proposal expansion**
jobs. Reaction replay, proof-bank validation, exact portfolio binding, full
replacement re-solving, parent proof, and CAS publication are downstream
deterministic stages; they are not silently represented as proof jobs in that
same queue. Every proof-eligible Top-K item is rendered as its own closed branch
DAG. Canonical molecule nodes may be shared, but reaction selections are
branch-specific. A replacement preview switches to the complete backend
re-solved route; rejected candidates remain visible, and no UI single-step
splice can establish truth. Portfolio eligibility remains advisory and is not
equivalent to `parent solved`.

The current paclitaxel Architecture V2 replay demonstrates the engineering
path without claiming a scientific solve: the four-child Codex team completed,
the fused overlay contains 85 molecules and 82 reaction hyperedges (including
five independently supported multi-source edges), and a validated CAS revision
binds the complete 96-branch projection of the explored graph. No stock-closed,
reaction-validated route entered the portfolio, so the authoritative verdict
remains `hypothesis_route_proposed`, `solved=false`. The exact run metrics and
remaining gates are recorded in [AutoPlanner mainline](docs/MAINLINE.md#paclitaxel-end-to-end-run-2026-07-10).

The route workbench is a digest-bound view of that full forest, not a second
source of truth. It offers deterministic route-cluster, shared-hypergraph, and
current-branch layouts; branch/source/proof filters; pan, zoom, fit, minimap,
orientation, density, and label controls; and a molecule/reaction/evidence
inspector. Desktop panes are resizable, medium screens use drawers, and mobile
and embedded views start canvas-first. Large diagnostic-only interface matrices
and duplicate graph structure SVGs are omitted from the browser payload while
the complete `explored_route_forest.v1` remains SHA-256-bound and authoritative.
The embedded parent accepts only a browser-verified delivery handshake. Route
replacement previews are complete hidden branches that already passed backend
AND/OR connectivity, stock, and reaction-proof revalidation; pairwise interface
comparisons remain diagnostic and never authorize a single-step splice.

## Current Anchors

- [Docs index](docs/README.md)
- [Architecture V2](docs/ARCHITECTURE_V2.md)
- [AutoPlanner mainline](docs/MAINLINE.md)
- [Agentic blackboard mainline](docs/AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md)
- [Repository surface and hygiene](docs/REPOSITORY_HYGIENE.md)
- [Codex WellAU streaming runbook](docs/CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md)

## Active Repository Surface

| Path | Purpose |
| --- | --- |
| `cascade_planner/agent/` | Codex-facing schemas, controllers, evidence artifacts, route audit, literature workflow, and validation gates. |
| `cascade_planner/orchestration/` | Direct Codex coordinator/child-agent execution and team result collection. |
| `cascade_planner/routes/` | Canonical route identity, multi-source fusion, blackboard adapters, and advisory multi-step graph assembly. |
| `cascade_planner/application/` | Persistent proposal-frontier scheduling plus downstream proof-bound AND/OR route portfolios and full-route replacement validation. |
| `cascade_planner/providers/` | Replaceable typed proposal, evidence, stock, verifier, agent, artifact, and renderer interfaces. |
| `cascade_planner/runtime/` | Persistent agent state, event log, idempotency, and reconciliation contracts. |
| `cascade_planner/baselines/` | ChemEnzy adapter and shared route contracts used as deterministic tools. |
| `cascade_planner/harness/` | Deterministic proof/closeout compilers plus digest-bound route-forest layout and delivery. |
| `cascade_planner/web/` | Local UI and progress/artifact inspection surface. |
| `scripts/` | Active launchers for agentic blackboard runs, Codex/WellAU, ChemEnzy, WebUI, and current replay workflows. |
| `tests/` | Current contract tests for artifacts, route audit, Codex worker control, literature evidence, and WebUI behavior. |
| `data/strategic_disconnections/` | Small curated evidence/disconnection source layer. |
| `results/shared/` | Local run outputs and traces; ignored by git. |
| `docs/archive/` | Historical plans and fixed-chain/fullflow reports kept for provenance. |

## Quick Start

Python 3.12 is the validated development version; Python 3.11 is also
supported. Create an isolated environment and install the application and test
requirements.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

POSIX shell:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest --collect-only -q
```

`requirements.txt` is the application/inference environment;
`requirements-dev.txt` contains test tooling. ChemEnzy model weights, vendor
checkouts, and saved run artifacts are optional local resources and are not
installed by either file.

Run the complete test suite with:

```powershell
$env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY = 'tests/fixtures/trusted_literature_step_registry.json'
python -m pytest -q
Remove-Item Env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY
```

The fixture registry is test-only; never use it as production literature
curation. See [AutoPlanner mainline](docs/MAINLINE.md) for the proof contracts
and the honest paclitaxel end-to-end result.

## Local-Only

Do not commit credentials, `results/shared/`, vendor checkouts, generated
archives, or local run caches. Historical material can be kept locally under an
ignored `archive/harness_prep_*` directory.
