# Audited Multi-engine Hybrid Report

## Honesty notes
- *K-budget fairness*: a UNION of E engines × top-10 has a ~10E candidate budget. The honest comparison is against a single engine's top-(10E). Below, both rows are shown.
- *Random baseline*: for EnzExpand, expected top-K hit under uniform draw from each row's `templates_tried` pool. When pool is small (e.g. mf=5 for rare EC), top-K trivially saturates.
- *n<20 mask*: per-transformation 'best engine' is computed only on transformations with ≥20 examples in the intersection.

## Overall

| policy                                    |    n |     top1 |     top5 |   top10 |   top50 | note                                                                                  |
|:------------------------------------------|-----:|---------:|---------:|--------:|--------:|:--------------------------------------------------------------------------------------|
| AiZ-USPTO (all)                           | 3028 |   0.0826 |   0.1291 |  0.1522 |  0.1668 |                                                                                       |
| RootAligned (all)                         | 3028 |   0.0779 |   0.144  |  0.1602 |  0.1991 |                                                                                       |
| MEGAN (all)                               | 3028 |   0.0997 |   0.1456 |  0.1704 |  0.2341 |                                                                                       |
| EnzExpand-A (mf2)                         |  994 |   0.2555 |   0.4064 |  0.4286 |  0.4608 | templates_tried mean=28.6  median=50  rows with pool<50: 46%  GT∈pool (top50==1): 46% |
| └─ random-in-pool baseline (mf2)          |  994 |   0.3224 |   0.4302 |  0.4445 |  0.4608 | E[hit] = (min(K,pool)/pool) × I(GT∈pool); honest random ceiling                       |
| └─ EnzExpand lift over random (mf2)       |  994 |   0.7924 |   0.9448 |  0.9642 |  1.0001 | lift > 1 means model ranks better than uniform draw inside its pool                   |
| EnzExpand-A (mf5)                         |  559 |   0.3918 |   0.5134 |  0.5259 |  0.5349 | templates_tried mean=24.3  median=4  rows with pool<50: 53%  GT∈pool (top50==1): 53%  |
| └─ random-in-pool baseline (mf5)          |  559 |   0.4434 |   0.5253 |  0.5305 |  0.5349 | E[hit] = (min(K,pool)/pool) × I(GT∈pool); honest random ceiling                       |
| └─ EnzExpand lift over random (mf5)       |  559 |   0.8837 |   0.9773 |  0.9914 |  1      | lift > 1 means model ranks better than uniform draw inside its pool                   |
| UNION chem-engines (3) top-10 [budget≈30] | 3028 | nan      | nan      |  0.2193 |  0.3078 | union budget = 10 × n_engines candidates                                              |
| Best single (megan) top-50 [budget=50]    | 3028 |   0.106  |   0.1559 |  0.1836 |  0.2612 | fair single-engine vs union (top-10 of 5 engines = budget 50)                         |
| UNION + EnzExpand over 3028 (n_enz=1064)  | 3028 |   0.182  |   0.248  |  0.2708 |  0.357  | adds enz pred on 1064/3028 (35%) steps                                                |

## By transformation (n≥20)

| transformation                   |   n |   aiz_top10 |   rootaligned_top10 |   megan_top10 | best_chem_engine   |
|:---------------------------------|----:|------------:|--------------------:|--------------:|:-------------------|
| oxidation                        | 718 |       0.205 |               0.24  |         0.252 | megan              |
| reduction                        | 468 |       0.427 |               0.502 |         0.485 | rootaligned        |
| acylation                        | 313 |       0.099 |               0.042 |         0.045 | aiz                |
| racemization                     | 303 |       0.003 |               0.033 |         0.076 | megan              |
| amination                        | 289 |       0.021 |               0.059 |         0.059 | rootaligned        |
| C_C_coupling                     | 257 |       0.113 |               0.097 |         0.101 | aiz                |
| hydrolysis                       | 223 |       0.126 |               0.121 |         0.188 | megan              |
| functional_group_interconversion |  87 |       0.034 |               0.034 |         0.023 | aiz                |
| phosphorylation                  |  75 |       0     |               0     |         0.013 | megan              |
| other                            |  74 |       0.054 |               0.014 |         0.014 | aiz                |
| isomerization                    |  72 |       0.014 |               0.014 |         0.069 | megan              |
| glycosylation                    |  62 |       0     |               0.048 |         0     | rootaligned        |
| esterification                   |  28 |       0.214 |               0.393 |         0.357 | rootaligned        |
| dehalogenation                   |  27 |       0.148 |               0.111 |         0.222 | megan              |
| epoxide_hydrolysis               |  22 |       0     |               0     |         0     | aiz                |
