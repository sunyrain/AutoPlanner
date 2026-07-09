# Postmortem - v4 Data Misread, 2026-05-10

## Correction

The previous statement that the project had only `43` or `87` usable training
targets was wrong.

Those numbers describe only the small ChemEnzy expansion traces that had already
been converted into action-value packs. They are not the size of
`dataset_v4_release`.

## Ground Truth Counts

The current v4 release at `dataset_v4_release/` contains:

| Layer | Count |
| --- | ---: |
| All reaction/cascade rows | 3,810 |
| Gold rows | 2,885 |
| Silver rows | 859 |
| High-quality rows, gold + silver | 3,744 |
| High-quality unique DOI | 2,464 |
| High-quality normalized steps | 8,609 |
| Catalyst components | 9,444 |
| Species rows | 21,225 |
| Substrate-scope entries | 3,458 |

After applying the current retrosynthesis-trace builder filters to
`cascade_v4_high_quality.jsonl`, excluding full100 benchmark overlap:

| Layer | Count |
| --- | ---: |
| Trace-candidate rows | 2,230 |
| Unique trace targets | 1,573 |
| Unique trace DOI | 1,577 |

Skipped rows:

| Reason | Count |
| --- | ---: |
| Missing or multi-target product | 1,024 |
| No usable GT `rxn_smiles` step | 148 |
| Benchmark DOI/cascade overlap | 99 |
| Benchmark target overlap | 243 |

## What Went Wrong

I conflated three different data layers:

1. Raw v4 release data: `dataset_v4_release/`, 3,744 high-quality rows.
2. Trace benchmark selection: `v4_trace_train_l200.json`, a deliberately limited
   200-row balanced subset.
3. Action-value packs: generated from l50/l100 ChemEnzy trace runs, covering only
   a subset of the 200 selected rows.

The bad conclusion came from inspecting layer 3 and speaking as if it described
layer 1. That is a basic accounting error.

The correct interpretation is:

- v4 data volume is not the immediate bottleneck.
- The current bottleneck is that only a small fraction of v4 has been converted
  into ChemEnzy search traces and route-outcome supervision.
- Sparse exact-action positives reflect proposal/search recall under the current
  trace run, not lack of raw cascade data.

## Impact

This error distorted the training discussion in two ways:

- It made the dataset look smaller than it is.
- It pushed the conversation toward "data insufficiency" instead of the real
  issue: full v4 trace generation, clean split design, and process-aware
  supervision extraction.

The prior small-pack results remain useful only as diagnostics. They must not be
treated as evidence that v4 itself is too small.

## Required Rule Going Forward

Before making any claim about data sufficiency, report all four levels:

1. Release size: all rows, high-quality rows, unique DOI.
2. Trace-candidate size after filtering.
3. Actually traced target count.
4. Action/source/value supervision counts and positive rates.

Do not collapse these levels into one number.

## Correct Next Training Framing

Use the 2,230 trace-candidate v4 records as the starting pool, with strict
held-out splits against full100. Generate ChemEnzy traces and route outcomes over
that pool, then train a process-aware state/action or partial-program value
model.

The right statement is:

> We have enough v4 records to start a serious training run, but we have not yet
> converted the full usable v4 pool into production-grade search traces and
> supervision.

