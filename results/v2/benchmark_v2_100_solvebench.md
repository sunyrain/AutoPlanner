# 100-target multi-step benchmark — honest evaluation

max_iter=100  max_depth=6  n_routes=5  timeout=180s

`solve-rate`: fraction of targets with ≥1 route whose leaves are all in ZINC stock.
`GT@K`: fraction with any of top-K routes overlapping ≥50% of GT intermediates.

| policy | subset | n | solve-rate | GT@1 | GT@5 | mean GT-overlap | mean depth | mean t(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uspto | ALL | 100 | 66.00% | 14.00% | 20.00% | 0.31 | 2.82 | 28 |
| uspto | all_chemical | 25 | 60.00% | 16.00% | 20.00% | 0.34 | 2.88 | 29 |
| uspto | all_enzymatic | 42 | 52.38% | 4.76% | 4.76% | 0.26 | 3.14 | 31 |
| uspto | chemoenzymatic | 28 | 89.29% | 25.00% | 42.86% | 0.35 | 2.39 | 22 |