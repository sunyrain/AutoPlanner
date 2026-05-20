# Codebase Status - 2026-05-19

This file is a cleanup guard. It records what should be kept in the current
branch, what is research-only, and what has already been archived.

## Runtime Mainline

These files are part of the current ChemEnzy-backed Web workflow and should not
be archived without a replacement:

| Area | Files |
| --- | --- |
| Web server | `cascade_planner/web/app.py` |
| Web UI | `cascade_planner/web/static/app.js`, `index.html`, `styles.css` |
| Web runner | `scripts/run_chem_enzy_plan_for_web.py`, `scripts/run_autoplanner_web_waitress.py` |
| Web monitor | `scripts/monitor_autoplanner_web.py` |
| ChemEnzy adapter | `cascade_planner/baselines/chem_enzy_adapter.py`, `route_contract.py` |
| Material sanity | `cascade_planner/baselines/route_plausibility.py`, `cascade_planner/eval/product_route_feasibility_audit.py` |
| Route selector pack/training | `cascade_planner/eval/build_route_pool_selector_pack.py`, `cascade_planner/eval/train_route_selector_v0.py` |
| Route selector A/B/C/D report | `cascade_planner/eval/train_route_pool_ranker.py` |
| Web tests | `tests/test_web_app.py`, `tests/test_web_product_audit_filter.py`, `tests/test_chem_enzy_web_payload.py`, `tests/test_route_plausibility.py` |

Current runtime contract:

```text
ChemEnzy native search
  -> Web job queue / cancel
  -> product-audit material sanity filter
  -> main result + raw sidecar + rejected sidecar
  -> UI provenance display
```

## Active Research, Not Default Product Path

These modules are still referenced by current experiments or tests. They are
not promoted as the main system claim, but they should not be deleted blindly.

| Area | Status |
| --- | --- |
| `cascade_planner/cascade_search/` | Active research interface for cascade search, subgoal proposals, action value hooks, and v4 product value. |
| CCTS / runtime rerank scripts in `cascade_planner/eval/` | Research-only. Useful for ablations and held-out replay, not the Web default. |
| `train_cascade_subgoal_scorer.py` and subgoal audit scripts | Research-only, but imported by `cascade_planner/cascade_search/proposals.py`. |
| `train_cascade_action_value.py` | Research-only, but imported by `cascade_planner/cascade_search/action_value.py`. |
| Routepool rerank/product-audit scripts | Useful for batch evaluation and should stay until the statin and blind-split reports are regenerated. |

## Historical / Sidecar Work

| Area | Status |
| --- | --- |
| Phase I reservoir/student-only distillation | Historical research result. Keep docs and artifacts for reproducibility; do not describe it as the current winning model. |
| `AUTOPLANNRELLM/` | Optional LLM side line. Not part of the current default framework. |
| `AI_OS_AutoResearch/` | External nested git checkout. Ignored by this repo. |
| Generated AI_OS patch/bundle files | Archived under `archive/code/generated_patches_2026-05-19/`. |
| Standalone subgoal smoke/ablation runners | Archived under `archive/code/research_experiments_2026-05-19/subgoal/`. |
| `releases/autoplanner_cascade_fixed_20260517/` | Frozen demo/report bundle. Keep as an artifact, not as source of truth. |

## Current Cleanup Rule

Use this order before removing code:

1. Check whether the file is imported by runtime code or tests.
2. Check whether it is needed to reproduce a named report.
3. Move only true generated artifacts to `archive/code/`.
4. For research scripts, prefer marking their status in docs before deletion.
5. Delete only after the corresponding tests and smoke commands still pass.

## Verification Used For This Cleanup

```bash
python -m py_compile cascade_planner/web/app.py scripts/run_chem_enzy_plan_for_web.py
node --check cascade_planner/web/static/app.js
python -m unittest discover -s tests -p 'test_web_product_audit_filter.py' -v
python -m unittest discover -s tests -p 'test_web_app.py' -v
python -m unittest discover -s tests -p 'test_chem_enzy_web_payload.py' -v
python -m unittest discover -s tests -p 'test_route_plausibility.py' -v
python -m unittest discover -s tests -p 'test_monitor_autoplanner_web.py' -v
python -m unittest discover -s tests -p 'test_build_route_pool_selector_pack.py' -v
python -m unittest discover -s tests -p 'test_train_route_selector_v0.py' -v
python -m unittest discover -s tests -p 'test_routepool_context_controls.py' -v
python -m unittest discover -s tests -p 'test_runtime_ccts_product_audit_rerank.py' -v
python -m unittest discover -s tests -p 'test_cascade_only_product_audit_rerank.py' -v
python -m unittest discover -s tests -p 'test_gate_phase_selector_promotion.py' -v
```
