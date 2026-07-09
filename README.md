# AutoPlanner

AutoPlanner is being refocused around a policy-driven Codex + blackboard route
controller for chemoenzymatic retrosynthesis.

The current architecture target is:

```text
target input
-> deterministic preflight
-> agentic blackboard state
-> Codex chooses compact typed actions
-> deterministic validators check safety, budgets, source binding, and proof boundaries
-> local tools execute approved actions
-> deterministic parent proof decides final verdict
```

Codex is allowed to plan, search, classify sources, and decide the next bounded
action batch. Codex is not allowed to directly mark a route solved, inject raw
reaction SMILES into production, or promote artifacts to production KB without
local validator approval.

## Current Anchors

- [Docs index](docs/README.md)
- [AutoPlanner mainline](docs/MAINLINE.md)
- [Agentic blackboard mainline](docs/AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md)
- [Codex WellAU streaming runbook](docs/CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md)
- [Optional EnzRetro Transformer integration](docs/ENZRETRO_INTEGRATION.md)
- [GitHub upload checklist](docs/GITHUB_UPLOAD_CHECKLIST.md)
- [Chemical-constraint Transformer benchmark report](docs/CHEM_CONSTRAINT_TRANSFORMER_BENCHMARK_REPORT.md)

## Active Repository Surface

| Path | Purpose |
| --- | --- |
| `cascade_planner/agent/` | Codex-facing schemas, controllers, evidence artifacts, route audit, literature workflow, and validation gates. |
| `cascade_planner/baselines/` | ChemEnzy adapter and shared route contracts used as deterministic tools. |
| `cascade_planner/baselines/enzretro_onestep.py` | Optional EnzRetro Transformer one-step proposal provider, enabled through environment variables. |
| `cascade_planner/web/` | Local UI and progress/artifact inspection surface. |
| `scripts/` | Active launchers for agentic blackboard runs, Codex/WellAU, ChemEnzy, WebUI, and current replay workflows. |
| `tests/` | Current contract tests for artifacts, route audit, Codex worker control, literature evidence, and WebUI behavior. |
| `data/strategic_disconnections/` | Small curated evidence/disconnection source layer. |
| `results/shared/` | Local run outputs and traces; ignored by git. |
| `docs/archive/` | Historical plans and fixed-chain/fullflow reports kept for provenance. |

## Quick Checks

```bash
PYTHONPATH=. pytest --collect-only -q

python - <<'PY'
import ast
from pathlib import Path
for path in sorted(Path("scripts").glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"))
print("scripts parse")
PY
```

## Local-Only

Do not commit credentials, `results/shared/`, vendor checkouts, generated
archives, or local run caches. Historical material can be kept locally under an
ignored `archive/harness_prep_*` directory.

Large external model assets such as ChemEnzy vendor checkouts and EnzRetro
checkpoints are intentionally configured by path rather than stored in this
repository. See `.env.example` for local environment variables.
