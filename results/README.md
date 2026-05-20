# results/ Layout

Result artifacts are not all comparable. Keep metric families separate.

## Folders

| Folder | Contents |
|---|---|
| `results/v2/` | curated benchmark reports and lightweight CSV summaries |
| `results/shared/` | local caches, checkpoints, traces, and benchmark JSON artifacts; not committed |
| `archive/` | old reports, retired checkpoints, and historical drafts |

## Reporting Rules

- Skeleton-only recovery metrics must be labeled `skeleton_*`.
- Filled-route metrics must be labeled `filled_*`.
- Mock, cached, and live candidate modes must not be mixed in one headline.
- Stock solve must be reported separately from route generation.
- Condition and cascade compatibility must be reported separately from GT route
  overlap.
- Candidate source breakdown is required for live runs.

Required source labels:

- `retrochimera`
- `enzyformer`
- `v3_retrieval`
- `enzexpand`
- `llm_hypothesis` when explicitly enabled

## Active Caveat

`cascade_planner/cascadeboard/integrated_benchmark.py` is currently useful for
development but still includes skeleton-only scoring behavior. Prefer
`cascade_planner/cascadeboard/live_benchmark.py` for live route artifacts, and
still audit its metric definitions before making publication claims.

## Current Curated Reports

- `results/v2/route_tree_v4_depthaligned_full100_20260509.md`
- `results/v2/route_tree_v4_depthaligned_l20_20260509.md`
- `results/v2/candidate_miss_audit_route_tree_v4_depthaligned_full100_20260509.md`
- `results/v2/route_tree_default_reverse_full100_20260508.md`
- `results/v2/route_tree_drop_root_cause_20260508.md`

Heavy JSON artifacts under `results/shared/` are local reproduction artifacts.
Commit concise markdown reports, not full benchmark payloads or checkpoints.
