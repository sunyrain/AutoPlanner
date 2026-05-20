# AutoPlanner Web UI

The Web UI is the current local operator surface for the ChemEnzy-backed
AutoPlanner workflow.

## Run

Development server:

```bash
PYTHONPATH=. python -m cascade_planner.web.app --host 127.0.0.1 --port 7860
```

Waitress server used for collaborator testing on this machine:

```bash
PYTHONPATH=. AUTOPLANNER_WEB_HOST=0.0.0.0 AUTOPLANNER_WEB_PORT=7991 \
  CHEMENZY_ENV_PREFIX=/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env \
  python scripts/run_autoplanner_web_waitress.py
```

Open `http://127.0.0.1:7860` or `http://127.0.0.1:7991`.

Monitor the running service from a terminal:

```bash
PYTHONPATH=. python scripts/monitor_autoplanner_web.py --url http://127.0.0.1:7991 --once
```

## Current Features

- ChemEnzy native route search through `scripts/run_chem_enzy_plan_for_web.py`
- queued route jobs with a single worker
- explicit route-search cancel button
- selectable stock modes:
  - commercial / Zinc
  - PaRoutes n1 building-block
  - PaRoutes n5 benchmark
- condition and enzyme annotation display when enabled
- product-audit filtering for severe material-sanity artifacts
- raw sidecar artifact for the unfiltered ChemEnzy output
- rejected sidecar artifact for routes hidden by product-audit
- per-step proposal provenance:
  - source
  - model/type if exported
  - retro/enzyme/condition/confidence scores
  - atom-change screen
  - stock evidence
  - external evidence summary

## Important Interpretation

Predicted conditions, EC numbers, and mechanism text are hypotheses exported
for review. They are not validated experimental protocols.

Routes hidden by product-audit are diagnostic records, not proposed syntheses.
Open the rejected sidecar to inspect why a route was removed.
