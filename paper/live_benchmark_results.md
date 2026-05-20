# Live Benchmark Results

Source: `results/v2/live_benchmark_full.json`

| Metric | Value |
|---|---:|
| `n_targets` | 100 |
| `plan_rate` | 100.0% |
| `skeleton_type_GT@1` | 58.0% |
| `skeleton_type_GT@5` | 73.0% |
| `filled_route_any` | 100.0% |
| `filled_type_GT@1` | 58.0% |
| `filled_type_GT@5` | 73.0% |
| `terminal_GT_reactant_in_top5` | 10.0% |
| `strict_stock_solve_any` | 46.0% |
| `condition_window_success_any` | 95.0% |
| `cascade_compatibility_success_any` | 95.0% |
| `avg_time_per_target_s` | 3.431 |

## Candidate Sources

| Source | Count |
|---|---:|
| `enzyformer` | 553 |
| `retrochimera` | 329 |
| `v3_retrieval` | 26 |

## Per Domain

| Domain | n | Plan | Filled type GT@5 | Stock | Condition | Compatibility |
|---|---:|---:|---:|---:|---:|---:|
| `all_chemical` | 25 | 100.0% | 92.0% | 60.0% | 96.0% | 96.0% |
| `all_enzymatic` | 42 | 100.0% | 64.3% | 28.6% | 97.6% | 97.6% |
| `chemoenzymatic` | 28 | 100.0% | 67.9% | 57.1% | 89.3% | 89.3% |
| `hybrid_mimetic` | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `whole_cell_biocatalytic` | 4 | 100.0% | 75.0% | 50.0% | 100.0% | 100.0% |
