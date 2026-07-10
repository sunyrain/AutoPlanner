# AutoPlanner Web UI

The Web UI is the current local operator surface for the ChemEnzy-backed
AutoPlanner workflow.

## Run

Development server:

```bash
PYTHONPATH=. python -m cascade_planner.web.app --host 127.0.0.1 --port 7860
```

Waitress server (loopback-only by default):

```bash
PYTHONPATH=. AUTOPLANNER_WEB_PORT=7991 \
  CHEMENZY_ENV_PREFIX=/path/to/chem_enzy_runtime/envs/retro_planner_env \
  python scripts/run_autoplanner_web_waitress.py
```

The application does not provide user authentication. Do not bind it to
`0.0.0.0` directly. For collaborator access, put it behind an authenticated
reverse proxy and keep Codex provider/key settings in server environment
variables (`AUTOPLANNER_CODEX_BASE_URL`, `AUTOPLANNER_CODEX_KEY_PATH`). HTTP
payloads cannot select a key, provider endpoint, worker auth mode, or sandbox;
web-launched Codex child agents always run read-only.

Open `http://127.0.0.1:7860` or `http://127.0.0.1:7991`.

Monitor the running service from a terminal:

```bash
PYTHONPATH=. python scripts/monitor_autoplanner_web.py --url http://127.0.0.1:7991 --once
```

## Current Features

- independent switches for the Codex action planner and direct child-agent
  retrosynthesis campaign
- bounded Codex frontier depth/expansion controls and a separate Codex team
  model setting; ambient Codex authentication is server-owned
- read-only multi-step `route_consensus_graph.v1` rendering with per-step
  provenance and no solved/executable promotion
- ChemEnzy native route search through `scripts/run_chem_enzy_plan_for_web.py`
- queued route jobs with a single worker
- explicit route-search cancel button
- selectable stock modes:
  - commercial / Zinc
  - PaRoutes n1 building-block
  - PaRoutes n5 benchmark
- condition and enzyme annotation display when enabled
- optional rule cascade-verifier hard gate (`enable_rule_verifier_gate` /
  `cascade_verifier_gate`) for conservative displays
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

The cascade-verifier hard gate is off by default. When enabled, the Web runner
first computes verifier metrics for every ChemEnzy route and then hides routes
with explicit rule-verifier failures. Routes without a `stage_partition` are
interpreted as sequential stepwise syntheses, not one-pot cascades.
