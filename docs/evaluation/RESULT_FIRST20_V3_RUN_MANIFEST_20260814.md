# Result-first 20-target V3 run manifest

Status: invalidated; diagnostic only

V3 was stopped after two completed targets. Although provider-to-host route
lineage loss was zero, ChemEnzy searched against its default
`PaRotes_n1-stock` while B4 was scored against the frozen
`SynthAtlas50Stock` SQLite index. Consequently, its 2/2 B1 and 0/2 B4
observations do not measure search performance under the declared benchmark
boundary and must not be combined with a corrected run.

## Frozen identity

- Output root: `results/shared/synthatlas_result_first20_v3_20260814`
- Frozen panel snapshot SHA-256:
  `790f107b7d245f64b1482455fb15799e6f837fee5f9ef483c0bf2e6f7fac1559`
- Frozen stock SHA-256:
  `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`
- Effective provider stock: `PaRotes_n1-stock` (configuration defect; does not
  match the frozen scoring stock above).
- Manifest SHA-256:
  `2354f031098e00e2bfa721820436b3041a7e218a0794d1ab9a08f5f0b15ac6ab`
- Target selection: first 20 cases in manifest order; no V1/V2/canary
  completion is reused.
- One panel worker; `gpt-5.5` low; fast profile; 1,800-second and 128-task
  fixed cutoffs.

## Result-first runtime contract

- Native and guided ChemEnzy retain the 300-second external hard ceiling.
- Native MCTS derives a 210-second internal search deadline, reserving time for
  route extraction, host audit, JSON serialization, and atomic publication.
- A request for 16 output routes uses a 32-route successful reserve rather than
  128. Guided one-route requests retain a four-route minimum reserve.
- Guided hypotheses carry every existing canonical parent route-family ID
  through materialization; providers cannot create canonical parents.
- The panel explicitly selects `--delivery-boundary stock_result`. The action
  loop closes the bounded pass after the first B4 snapshot. Exact evidence,
  conditions, conflicts, and Program work are deferred for a separate resume
  stage and do not occupy the panel worker after stock closure.
- Default CLI/API target solving remains `delivery_boundary=full`; B4 is not a
  universal scientific terminal.

## Real canary evidence

Target 001 under
`synthatlas_result_first_soft_timeout_canary_001c_20260814` changed from the old
V1 observation (B4 false, 920.1 seconds) to:

- B4 true with two stock-closed route skeletons;
- first route at 262.5 seconds and B4 at 549.3 seconds;
- five target-rooted materialized routes;
- two provider routes with final lineage disposition `stock_closed` and zero
  provider-integration-loss targets;
- the native phase wrote a complete 154 KB no-route result at about 222 seconds
  instead of being killed at 300 seconds without an artifact;
- guided frontiers returned complete results in approximately 20 and 11
  seconds.

The full canary continued for 172.5 seconds after B4 and executed 11 evidence,
5 condition, 7 Program discovery, and 2 Program review actions. V3's explicit
stock-result boundary removes that tail from the primary panel.

## Reproduction

```powershell
python scripts/run_v4_blind_panel.py `
  --manifest benchmarks/synthatlas_strategy_closure50_v2.v1.json `
  --output-root results/shared/synthatlas_result_first20_v3_20260814 `
  --model gpt-5.5 --reasoning-effort low --execution-profile fast `
  --fixed-cutoff-wall-time-s 1800 --fixed-cutoff-total-tasks 128 `
  --workers 1 --max-targets 20 --ablation unified-adaptive `
  --benchmark-stock-index data_external/retrostar190/retrostar_emolecules_stock.sqlite3 `
  --benchmark-stock-name SynthAtlas50Stock `
  --chemenzy-env-prefix D:\conda\envs\py312
```

The panel runner injects `--delivery-boundary stock_result` into every target.
Do not resume this root. The corrected runner now binds the benchmark stock
index to both provider search and host scoring and rejects explicit mismatches.
