# Cleanup Progress Audit - 2026-05-19

This is a progress audit, not a completion claim.

## Objective Restatement

Current working objective:

```text
全面推进并实时监控；同时清理归档旧代码、更新文档、整理当前系统状态。
```

Concrete deliverables tracked here:

1. Root scratch artifacts should not clutter source status.
2. External integration checkout should not be committed into AutoPlanner.
3. Current Web/ChemEnzy workflow should be documented.
4. Runtime monitor should exist and be tested.
5. Old standalone experiments should be archived only after dependency checks.
6. Remaining research code should have an explicit keep/archive status.
7. Focused tests should cover the retained runtime and research paths.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Archive root AI_OS bundle/patch artifacts | `archive/code/generated_patches_2026-05-19/` | done |
| Keep `AI_OS_AutoResearch/` out of this repo | `.gitignore` entry | done |
| Keep generated release bundles out of source status | `.gitignore` entry for `releases/` | done |
| Document current architecture and next steps | `docs/CURRENT_STATE_2026-05-19.md` | done |
| Document code status and cleanup guard | `docs/CODEBASE_STATUS_2026-05-19.md` | done |
| Document Web runtime and monitor command | `cascade_planner/web/README.md` | done |
| Document scripts entry points | `scripts/README.md` | done |
| Add terminal monitor | `scripts/monitor_autoplanner_web.py` | done |
| Test terminal monitor | `tests/test_monitor_autoplanner_web.py` | done |
| Archive unreferenced standalone subgoal runners | `archive/code/research_experiments_2026-05-19/subgoal/` | done |
| Verify archived runners have no live imports | grep over `cascade_planner`, `tests`, `scripts`, `docs` | done |
| Verify retained subgoal code still works | `test_cascade_subgoal_scorer.py` | done |
| Verify product-audit/Web runtime path | `test_web_product_audit_filter.py`, `test_web_app.py`, `test_chem_enzy_web_payload.py`, `test_route_plausibility.py` | done |
| Verify retained CCTS/routepool side branches | routepool/context, runtime CCTS, cascade-only rerank, promotion gate tests | done |

## Known Remaining Work

- The working tree still contains active research changes under
  `cascade_planner/cascade_search/`, `cascade_planner/eval/`, and related tests.
- These files are intentionally not archived yet because they are imported by
  tests, `cascade_search`, or current rerank/product-audit workflows.
- No global completion claim should be made until the remaining research files
  are either promoted, merged, or explicitly archived.

Current worktree snapshot after this cleanup pass:

| Status | Count | Meaning |
| --- | ---: | --- |
| `M` | 25 | Existing tracked files changed by current Web/product-audit/research cleanup work. |
| `??` | 23 | New files that are runtime support, docs, focused tests, or active research scripts. |

No root-level `.patch`, `.bundle`, `.log`, `tmp`, or `bak` scratch files remain.

## Latest Focused Verification

The following focused checks were run after archiving the standalone subgoal
runners:

| Test file | Result |
| --- | ---: |
| `test_web_product_audit_filter.py` | 4/4 passed |
| `test_web_app.py` | 13/13 passed |
| `test_chem_enzy_web_payload.py` | 6/6 passed |
| `test_route_plausibility.py` | 3/3 passed |
| `test_monitor_autoplanner_web.py` | 2/2 passed |
| `test_build_route_pool_selector_pack.py` | 3/3 passed |
| `test_train_route_selector_v0.py` | 1/1 passed |
| `test_routepool_context_controls.py` | 2/2 passed |
| `test_runtime_ccts_product_audit_rerank.py` | 3/3 passed |
| `test_cascade_only_product_audit_rerank.py` | 6/6 passed |
| `test_gate_phase_selector_promotion.py` | 4/4 passed |

## Current Cleanup Rule

Do not delete a research script just because it is not part of the Web default.
Archive only when all are true:

1. no runtime import,
2. no focused test import,
3. no current report index depends on it,
4. docs record the archive path,
5. focused tests still pass.
