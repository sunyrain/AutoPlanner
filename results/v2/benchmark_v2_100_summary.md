# Benchmark v2 100 summary (benchmark_v2_100_solvebench.csv)

total rows: 100  ·  unique targets: 93

## Overall by policy
| policy | n | solve | gt@1 | gt@5 | overlap | mean_depth_solved | mean_time_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| uspto | 100 | 66.0% | 14.0% | 20.0% | 0.31 | 2.08 | 27.7 |

## By route_domain × policy
| domain | policy | n | solve | gt@1 | gt@5 | overlap |
|---|---|---:|---:|---:|---:|---:|
| all_chemical | uspto | 25 | 60.0% | 16.0% | 20.0% | 0.34 |
| all_enzymatic | uspto | 42 | 52.4% | 4.8% | 4.8% | 0.26 |
| chemoenzymatic | uspto | 28 | 89.3% | 25.0% | 42.9% | 0.35 |
| hybrid_mimetic | uspto | 1 | 100.0% | 0.0% | 0.0% | 0.20 |
| whole_cell_biocatalytic | uspto | 4 | 75.0% | 25.0% | 25.0% | 0.50 |

## By GT depth × policy
| gt_depth | policy | n | solve | gt@5 |
|---:|---|---:|---:|---:|
| 2 | uspto | 61 | 65.6% | 24.6% |
| 3 | uspto | 32 | 65.6% | 12.5% |
| 4 | uspto | 6 | 66.7% | 16.7% |
| 7 | uspto | 1 | 100.0% | 0.0% |
