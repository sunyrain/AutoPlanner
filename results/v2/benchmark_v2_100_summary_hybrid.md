# Benchmark v2 100 summary (benchmark_v2_100_solvebench_hybrid.csv)

total rows: 100  ·  unique targets: 93

## Overall by policy
| policy | n | solve | gt@1 | gt@5 | overlap | mean_depth_solved | mean_time_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| uspto+enz | 100 | 72.0% | 13.0% | 21.0% | 0.31 | 2.18 | 37.0 |

## By route_domain × policy
| domain | policy | n | solve | gt@1 | gt@5 | overlap |
|---|---|---:|---:|---:|---:|---:|
| all_chemical | uspto+enz | 25 | 64.0% | 16.0% | 20.0% | 0.34 |
| all_enzymatic | uspto+enz | 42 | 61.9% | 2.4% | 4.8% | 0.26 |
| chemoenzymatic | uspto+enz | 28 | 89.3% | 25.0% | 46.4% | 0.36 |
| hybrid_mimetic | uspto+enz | 1 | 100.0% | 0.0% | 0.0% | 0.20 |
| whole_cell_biocatalytic | uspto+enz | 4 | 100.0% | 25.0% | 25.0% | 0.50 |

## By GT depth × policy
| gt_depth | policy | n | solve | gt@5 |
|---:|---|---:|---:|---:|
| 2 | uspto+enz | 61 | 70.5% | 26.2% |
| 3 | uspto+enz | 32 | 75.0% | 12.5% |
| 4 | uspto+enz | 6 | 66.7% | 16.7% |
| 7 | uspto+enz | 1 | 100.0% | 0.0% |
