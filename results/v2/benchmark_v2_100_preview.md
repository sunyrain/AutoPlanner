# Benchmark v2 100 summary (benchmark_v2_100_solvebench.csv)

total rows: 37  ·  unique targets: 37

## Overall by policy
| policy | n | solve | gt@1 | gt@5 | overlap | mean_depth_solved | mean_time_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| uspto | 37 | 64.9% | 21.6% | 27.0% | 0.35 | 2.04 | 27.9 |

## By route_domain × policy
| domain | policy | n | solve | gt@1 | gt@5 | overlap |
|---|---|---:|---:|---:|---:|---:|
| all_chemical | uspto | 11 | 54.5% | 18.2% | 27.3% | 0.30 |
| all_enzymatic | uspto | 13 | 53.8% | 15.4% | 15.4% | 0.31 |
| chemoenzymatic | uspto | 8 | 87.5% | 37.5% | 50.0% | 0.42 |
| hybrid_mimetic | uspto | 1 | 100.0% | 0.0% | 0.0% | 0.20 |
| whole_cell_biocatalytic | uspto | 4 | 75.0% | 25.0% | 25.0% | 0.50 |

## By GT depth × policy
| gt_depth | policy | n | solve | gt@5 |
|---:|---|---:|---:|---:|
| 2 | uspto | 19 | 63.2% | 31.6% |
| 3 | uspto | 12 | 66.7% | 25.0% |
| 4 | uspto | 5 | 60.0% | 20.0% |
| 7 | uspto | 1 | 100.0% | 0.0% |
