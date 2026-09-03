# Result-first 20-target V3 invalidation (2026-08-14)

## Decision

`results/shared/synthatlas_result_first20_v3_20260814` was intentionally
stopped after two completed targets. It is retained as a diagnostic root but
is not a benchmark result and must not be resumed or merged with later runs.

## Observed result before stopping

- 2/2 targets reached B1; 0/2 reached B2 or B4.
- The host materialized two and three target-rooted routes respectively.
- Provider integration loss was zero for both targets.
- No evidence, condition, conflict-resolution, or Program work ran before B4,
  so credibility-tail scheduling was not the cause of the miss.

## Invalidating defect

The frozen B4 scorer used
`data_external/retrostar190/retrostar_emolecules_stock.sqlite3` as
`SynthAtlas50Stock` (23,081,629 members; SHA-256
`30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`).
However, the ChemEnzy request used its default `PaRotes_n1-stock`. Search was
therefore optimizing terminal closure under a different inventory from the one
used to score B4. The resulting 0/2 B4 is not an estimate of the configured
system's performance.

## General correction and restart gate

- When a frozen benchmark stock index is present, the CLI and panel runner now
  bind that exact path and logical name to ChemEnzy automatically.
- An explicit provider binding must resolve to the same single path or the run
  fails before model/search work.
- Panel status records both the effective binding and an equality assertion.
- SQLite failure-analysis membership uses an indexed read-only query rather
  than scanning the binary database as CSV.
- A fresh one-target canary must verify the provider request before a new
  20-target root is started.
