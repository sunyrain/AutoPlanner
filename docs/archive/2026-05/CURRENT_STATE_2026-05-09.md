# Current State - 2026-05-09

## Status

`未收束`.

The active direction is cascade-native process search. ChemEnzy is used as a
proposal backend; `CascadeProgramSearch` owns the state, failures, repairs,
stage graph, condition envelope, cofactor ledger, and final objective.

## Active Architecture

Keep these components as the current baseline:

- `cascade_planner/cascade_search/state.py`
- `cascade_planner/cascade_search/failure.py`
- `cascade_planner/cascade_search/repair.py`
- `cascade_planner/cascade_search/search.py`
- `cascade_planner/cascade_search/trace.py`
- `cascade_planner/eval/run_cascade_search_benchmark.py`
- `cascade_planner/baselines/chem_enzy_adapter.py`

ChemEnzy remains a proposal provider and baseline, not the owner of cascade
state.

## Invalidated Experiments

Do not use these as project claims or training direction:

- Sparse candidate-label action ranker:
  `exact GT reaction` / `GT reactant overlap` labels only tested whether a
  proposed single step hit a benchmark step. It did not learn process progress.
- `dataset_v4_release` document-quality value model:
  using gold/silver reaction-record quality as the model target tests extraction
  quality, not cascade planning quality.

The corresponding code and generated artifacts were removed from the active
tree.

## Current Full100 Baseline

Current production cascade-search benchmark:

`results/shared/cascade_transition_value/full100_trace_20260509/baseline_merged.json`

| Metric | Value |
| --- | ---: |
| `n_targets` | 100 |
| `chem_enzy_solved_rate` | 1.00 |
| `cascade_solved_rate` | 1.00 |
| `stock_closed_rate` | 1.00 |
| `cofactor_closed_rate` | 1.00 |
| `condition_conflict_free_rate` | 1.00 |
| `enzyme_evidence_sufficient_rate` | 1.00 |
| `candidate_exact_reaction_in_pool` | 0.18 |
| `candidate_gt_reactant_in_pool` | 0.57 |
| `exact_reaction_in_route_pool` | 0.04 |
| `gt_reactant_in_route_pool` | 0.16 |
| `partial_gt_step_overlap_rate` | 0.04 |
| `avg_gt_step_overlap_fraction` | 0.0183 |
| `avg_cascade_search_time_s` | 0.0197 |

Interpretation:

- Proposal recall is nonzero but still weak.
- The selected route frontier remains worse than the candidate pool.
- Next model should learn process-aware state/action value, not single-step
  candidate identity.
- The first transition-value checkpoint is not yet promoted; it improved
  training loss but did not beat baseline full100 route recovery.
