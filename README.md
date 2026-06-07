# AutoPlanner

AutoPlanner is being refocused around a Codex-entry route-control harness for
chemoenzymatic retrosynthesis.

The current architecture target is:

```text
target input
-> deterministic preflight
-> Codex controller chooses the workflow
-> local tools execute ChemEnzy, route audit, frontier extraction, structure checks
-> Codex performs open literature/source reasoning
-> deterministic validators decide solved, partial, rejected, or needs_followup
```

Codex is allowed to plan, research, classify sources, and decide which tool to
call next. Codex is not allowed to directly mark a route solved, inject raw
reaction SMILES into production, or promote artifacts to production KB without
local validator approval.

## Current Anchors

- [Docs index](docs/README.md)
- [AutoPlanner mainline](docs/MAINLINE.md)
- [Codex WellAU streaming runbook](docs/CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md)
- [SMILES-first literature workflow](docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md)
- [Literature-to-executable template checklist](docs/LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md)

## Active Repository Surface

| Path | Purpose |
| --- | --- |
| `cascade_planner/agent/` | Codex-facing schemas, controllers, evidence artifacts, route audit, literature workflow, and validation gates. |
| `cascade_planner/baselines/` | ChemEnzy adapter and shared route contracts used as deterministic tools. |
| `cascade_planner/web/` | Local UI and progress/artifact inspection surface. |
| `scripts/` | Small active launchers for Codex/WellAU, ChemEnzy, WebUI, and current replay workflows. |
| `tests/` | Current contract tests for artifacts, route audit, Codex worker control, literature evidence, and WebUI behavior. |
| `data/strategic_disconnections/` | Small curated evidence/disconnection source layer. |
| `results/shared/` | Local run outputs and traces; ignored by git. |
| `archive/harness_prep_*/` | Local-only archive of files moved out of the active harness surface; ignored by git. |

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
