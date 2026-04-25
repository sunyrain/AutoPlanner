# 100-target multi-step benchmark — honest evaluation

max_iter=100  max_depth=6  n_routes=5  timeout=180s

`solve-rate`: fraction of targets with ≥1 route whose leaves are all in ZINC stock.
`GT@K`: fraction with any of top-K routes overlapping ≥50% of GT intermediates.

| policy | subset | n | solve-rate | GT@1 | GT@5 | mean GT-overlap | mean depth | mean t(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uspto+enz | ALL | 100 | 72.00% | 13.00% | 21.00% | 0.31 | 2.76 | 37 |
| uspto+enz | all_chemical | 25 | 64.00% | 16.00% | 20.00% | 0.34 | 3.08 | 40 |
| uspto+enz | all_enzymatic | 42 | 61.90% | 2.38% | 4.76% | 0.26 | 2.74 | 41 |
| uspto+enz | chemoenzymatic | 28 | 89.29% | 25.00% | 46.43% | 0.36 | 2.68 | 29 |