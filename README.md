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
-> unresolved molecular frontiers launch additional direct child-agent teams
-> one-step consensuses assemble into a provenance-preserving multi-step DAG
-> Codex chooses compact typed follow-up actions
-> deterministic validators check safety, budgets, source binding, and proof boundaries
-> local tools execute approved actions
-> deterministic parent proof decides final verdict
-> route forest shows alternatives, support, conflicts, and per-step evidence
```

Codex is allowed to delegate, plan, search, classify sources, and draft route
candidates. Multiple Codex roles are one correlated model source, not multiple
independent evidence sources. Codex cannot directly mark a route solved, inject
raw reaction SMILES into production, or promote artifacts to the production KB.
When the multi-agent mainline fails, the run stops unresolved instead of
silently replacing it with a deterministic scientific planner.

The proof boundary is fail-closed. Every accepted parent route is replayed
against the exact target and identifiable, content-hashed stock catalogs. A
literature stitch additionally requires exact step chemistry bound to a real
PDF page and an out-of-band trusted registry, plus independently verified stock
closure for every terminal reactant frontier. The packaged registry is empty by
default. Consensus graphs, visual extractions, and solved child targets remain
advisory until those parent-proof conditions are met.

## Current Anchors

- [Docs index](docs/README.md)
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
| `cascade_planner/runtime/` | Persistent agent state, event log, idempotency, and reconciliation contracts. |
| `cascade_planner/baselines/` | ChemEnzy adapter and shared route contracts used as deterministic tools. |
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
