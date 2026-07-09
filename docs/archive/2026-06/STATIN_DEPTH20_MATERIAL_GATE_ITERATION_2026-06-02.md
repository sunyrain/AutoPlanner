# Statin Depth-20 Material-Gate Iteration

Date: 2026-06-02

## Scope

This iteration addressed a specific failure mode observed in the statin reverse-synthesis runs:
top-level chemical disconnections could create implausible material growth, and those bad
chemical starts could later appear as enzyme-containing routes. The goal was to keep real
condition/protecting-group transfer cases while rejecting obvious scaffold/material artifacts.

## Code Changes

- Added a conservative first-step material gate in `cascade_planner/baselines/route_plausibility.py`.
- Integrated the first-step gate into `cascade_planner/eval/product_route_feasibility_audit.py`.
- Preserved condition and enzyme annotations during native-route conversion for product audit.
- Limited product-audit hard rejection to first-step material-gate hard rejects; explainable transfer
  reagent cases are warnings, not artifacts.
- Added regression tests for Boc/Fmoc/TMS transfer reagents, small condition reagent element sources,
  and pyruvate-like unexplained scaffold growth.
- Added `scripts/summarize_statin_depth20_routes.py` to regenerate the formal statin product-audit
  summary from final benchmark rows.

## Formal Benchmark

Command:

```bash
python scripts/run_native_vs_enhanced_route_benchmark.py \
  --target-rows results/shared/statin_enhanced_combo_20260601/statin_target_rows_all9.jsonl \
  --max-targets 0 \
  --output-dir results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/enhanced_all9 \
  --skip-native \
  --enhancement-preset final_clean_fastclosure_material_gate_semisynthesis_chemical_anchor_p16n16 \
  --preset-max-depth-override 20 \
  --preset-expansion-budget-override 480 \
  --preset-n-results-override 20 \
  --preset-route-tree-timeout-s-override 1800 \
  --gpu -1 \
  --checkpoint-every 1
```

Run output:

- Rows: `results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/enhanced_all9/native_vs_enhanced_route_rows.jsonl`
- Report: `results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/enhanced_all9/native_vs_enhanced_route_report.md`
- Strict enzyme audit: `results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/enhanced_all9/product_audit_depth20_budget480_top20_summary_strict_enzyme.json`

## Results

| metric | value |
| --- | ---: |
| targets | 9 |
| targets with routes | 9 |
| targets with solved route | 9 |
| total routes | 84 |
| solved routes | 75 |
| strict enzyme routes | 31 |
| solved strict enzyme routes | 31 |
| strict enzyme route targets | 2 |
| enzyme proposal candidates | 1892 |
| SP-v1 rejections | 563 |
| mean elapsed seconds | 655.848 |

Route-class counts from product audit:

| route class | routes |
| --- | ---: |
| triage_semisynthesis | 35 |
| triage_late_stage | 7 |
| triage_fragment | 3 |
| needs_chemist_review | 33 |
| weak_hint | 6 |
| reject_artifact | 0 |

Strict enzyme routes:

| target | routes | solved | enzyme routes | enzyme route audit |
| --- | ---: | ---: | ---: | --- |
| fluvastatin | 20 | 20 | 13 | 13 needs_chemist_review; first-step gate pass |
| simvastatin | 20 | 20 | 18 | 18 triage_semisynthesis; first-step gate pass |

First-step gate outcome for strict enzyme routes:

| decision | count |
| --- | ---: |
| pass | 31 |
| warn | 0 |
| hard_reject | 0 |

## Route Assessment

- Simvastatin is the strongest enzyme-containing result: 18 strict enzyme routes, all solved,
  all product-audited as semisynthesis triage.
- Lovastatin, mevastatin, pravastatin, and simvastatin give useful natural-statin semisynthesis
  route ideas, usually from advanced natural-core terminals.
- Fluvastatin enzyme routes no longer look like bad material artifacts, but product audit keeps them
  at needs_chemist_review rather than triage_semisynthesis.
- Cerivastatin and rosuvastatin have late-stage route ideas but no strict enzyme routes.
- Pitavastatin has fragment-level hints, but several non-enzyme routes still carry first-step
  material warnings.
- Atorvastatin remains weak: only one needs_chemist_review route and no product-ready target verdict.

## Interpretation

The original bad enzyme-route failure did not reproduce in the formal depth-20 statin run. Under the
strict route-level enzyme flag, all 31 enzyme routes are solved and pass the first-step material gate.
No route is classified as reject_artifact by product audit.

The main remaining limitation is condition quality. All 84 routes carry condition_warning, so these
outputs should be read as route-topology and disconnection evidence, not executable process conditions.

## Verification

- Focused regression: `pytest tests/test_route_plausibility.py tests/test_product_route_feasibility_audit.py`
- Full suite previously passed after the material-gate/product-audit changes: 874 passed, 3 skipped.

## Cleanup Notes

The formal final rows and reports are retained. Duplicate checkpoint rows and an earlier over-broad
enzyme-count summary were removed after the strict enzyme summary was generated.
