# Current State - 2026-05-12

## Status

`未收束，但方向正常`.

The project has moved from a pure route-tree baseline toward a process-aware
AutoPlanner-Cascade controller. The most important recent progress is not a new
global value model; it is that candidate coverage, search access, trace
diagnostics, and open-leaf policy are now explicit and measurable.

## What Changed Recently

### 1. Repository Cleanup And Paper Draft

Completed:

- root `README.md` rewritten around the current AutoPlanner-Cascade direction
- cleanup report added:
  `docs/CLEAN_REPOSITORY_REPORT_2026-05-12.md`
- old root-level reference image/PDF archived under:
  `archive/reference_inputs_2026-05-12/`
- current Nature-style draft added under:
  `paper/nature_autoplanner_cascade/`
- image2-generated Figure 1 background and deterministic labelled Figure 1
  added under:
  `paper/nature_autoplanner_cascade/figures/`
- PDF exported:
  `paper/nature_autoplanner_cascade/build/autoplanner_cascade_nature_draft.pdf`

Root-level `cascade_dataset_v2*.json`, `cascade_dataset_v3.json`,
`templates*.csv.gz`, and `ecreact-1.0.csv` were deliberately kept in place
because older scripts still use those names as default paths.

### 2. Coverage And Search-Access Fixes

Latest full100 coverage-fix artifacts:

- `results/shared/coverage_fix_20260511/coverage_fix_full100_stock.json`
- `results/shared/coverage_fix_20260511/coverage_fix_full100_trace.jsonl`
- `results/shared/coverage_fix_20260511/candidate_miss_audit_coverage_fix_full100_stock.json`
- `results/shared/coverage_fix_20260511/coverage_fix_full100_closure_report.json`

Compared with the 2026-05-09 route-tree baseline:

| Metric | 2026-05-09 baseline | 2026-05-11 current | Change |
| --- | ---: | ---: | ---: |
| `plan_rate` | 0.76 | 0.97 | +0.21 |
| `skeleton_type_GT@1` | 0.58 | 0.70 | +0.12 |
| `skeleton_type_GT@5` | 0.61 | 0.73 | +0.12 |
| `candidate_exact_reaction_in_pool` | 0.24 | 0.40 | +0.16 |
| `candidate_gt_reactant_in_pool` | 0.40 | 0.58 | +0.18 |
| `exact_reaction_in_route_pool` | 0.22 | 0.26 | +0.04 |
| `gt_reactant_in_route_pool` | 0.38 | 0.44 | +0.06 |
| `condition_window_success_any` | 0.68 | 0.76 | +0.08 |
| `cascade_compatibility_success_any` | 0.68 | 0.76 | +0.08 |
| `strict_stock_solve_any` | 0.50 | 0.31 | -0.19 |
| `avg_time_per_target_s` | 17.681 | 14.762 | -2.919 |

Interpretation:

- Candidate access and route-shape recovery improved substantially.
- The selected route frontier improved less than the candidate pool.
- Strict stock closure regressed and is now the most visible controller problem.
- The next model should not be a generic final value model yet; it should first
  improve source scheduling and stock-aware leaf expansion.

### 3. Candidate Miss Audit

Current audit bottlenecks on the stock-checked full100 result:

| Label | Count |
| --- | ---: |
| `intermediate_not_reached` | 126 |
| `queried_budget_too_small` | 76 |
| `selector_missed_candidate` | 12 |
| `generated_ranked_out` | 3 |
| `candidate_recovered` | 32 |

This supports the expert feedback: before stacking more scorers, fix proposal
coverage, source scheduling, and open-leaf expansion policy.

### 4. Learned Open-Leaf Policy

Latest policy artifacts:

- `results/shared/open_leaf_policy_20260511/search_policy.pt`
- `results/shared/open_leaf_policy_20260511/search_policy.json`
- `results/shared/open_leaf_policy_20260511/open_leaf_policy_l3_fixcheck.json`

Training summary:

| Field | Value |
| --- | ---: |
| training rows | 921 |
| train rows | 733 |
| validation rows | 188 |
| node feature dim | 272 |
| validation node top-1 | 0.772727 |
| validation node top-3 | 1.000000 |
| validation top-1 positive | 0.742424 |

This policy is useful but not final. Its next revision must explicitly value
stock closure and source/budget decisions.

## Architecture Position

The current organization should be understood as:

```text
ChemEnzyRetroPlanner native core
  -> mature multi-step route search and proposal machinery

AutoPlanner-Cascade controller
  -> search state, trace collection, typed failure, source scheduling,
     open-leaf policy, cascade/process scoring hooks

Process-aware cascade layer
  -> condition envelopes, stage graph, cofactor ledger, enzyme/evidence modules,
     repair actions
```

ChemEnzy is not merely a one-step provider. Its multi-step search machinery is
valuable. The AutoPlanner contribution is to absorb and steer that machinery
with cascade-aware state, failure labels, source scheduling, and process repair.

## What Is Not Yet Done

- No production-grade `CascadeSourcePolicy` is trained yet.
- No stock-aware open-leaf policy is promoted yet.
- Pairwise/fragment cascade scoring is not a final objective; it is a local
  preference signal and should only apply when adjacent steps actually exist.
- No final state/action value model should be claimed yet.
- Cofactor regeneration, stage split, buffer exchange, and evidence retrieval
  repair actions are not yet active production controls.
- Strict stock solve is below the old baseline and must be fixed before strong
  performance claims.

## Immediate Next Step

Train and evaluate a source/leaf controller, not a broad final value model.

Priority:

1. Build trace supervision for source scheduling:
   `source_not_queried`, `queried_budget_too_small`, `provider_missing`,
   `selector_missed_candidate`, and `stock_dead_end`.
2. Train `CascadeSourcePolicy` to choose source families and budget per state.
3. Update open-leaf utility to include stock closure, candidate yield, adjacent
   pair formation, and dead-end risk.
4. Re-run full100 and require:
   - retain `plan_rate >= 0.95`
   - retain candidate coverage gains
   - recover strict stock solve toward at least `0.45`, then `0.60`
   - avoid runtime regression beyond the 2026-05-11 run

Only after this should the project move into a full state/action value model.
