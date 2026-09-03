# Result-first stranded-action matched canary (frozen 2026-08-14)

Status: `completed`; production preflight and live matched execution both completed 8/8.

## Question

Does registering the already-compiled `recompute_route_closure` action prevent the old false `no_action` termination and let the remaining result-bearing guided frontiers improve root B4?

## Observed matched result

The explicitly authorized live run completed all eight frozen cases under
`results/shared/synthatlas_result_first_stranded_action_canary8_live_20260814`.
The candidate subset moved from baseline B4 `0/8` to `2/8`; targets 012 and
016 were newly stock closed, six targets remained stock open, and there were
no B4 regressions. All candidate cases matched the baseline, all were terminal,
and `v4_matched_panel_comparison.v1` marked the performance claim eligible.

| Target | B4 transition | Closure actions | Guided after closure | Candidate wall (s) |
| --- | --- | ---: | ---: | ---: |
| 006 | still stock open | 4 | 2 | 778.710 |
| 008 | still stock open | 3 | 2 | 698.932 |
| 011 | still stock open | 12 | 2 | 636.607 |
| 012 | newly stock closed | 0 | 0 | 201.381 |
| 013 | still stock open | 0 | 0 | 230.742 |
| 014 | still stock open | 8 | 2 | 717.991 |
| 015 | still stock open | 5 | 2 | 712.131 |
| 016 | newly stock closed | 0 | 0 | 41.525 |

The panel used 18 provider attempts: eight host-admitted successes and ten raw
empty results. It consumed 13 model invocations, 352,627 input tokens, 70,230
output tokens, 18 native-search units, and 4,018.017 summed target wall seconds;
median target wall time was 667.769 seconds. Relative to the same eight frozen
baseline records, wall time increased by 424.537 seconds while B4 increased by
two targets.

The control-path hypothesis is only partly confirmed. Targets
006/008/011/014/015 all executed closure actions and then two guided searches,
so the old stranded-action cutoff was removed. However, locally stock-closed
guided routes did not close every remaining leaf of their root routes. The
materialized command still imposed the frozen two-frontier cap even where the
read-only lower-bound projection exposed four to seven untried frontiers.
Targets 012 and 016 reached B4 directly from the native provider path, so their
gain validates the combined result-first integration fixes rather than the
closure handler alone. Target 013 is an observed provider/runtime failure:
ChemEnzy returned no route and the Codex initial architecture failed after five
streaming timeouts and HTTP fallback; it remains in the denominator and was not
rerun.

This canary is not a new benchmark and cannot rewrite the completed V4 12/20 result. It is a matched diagnostic over **all eight**, and only the eight, frozen V4 B4 failures: `006,008,011,012,013,014,015,016`. The set was frozen before any new outcome is observed; targets may not be dropped after seeing results.

## Why this is result-bearing

All eight old failures ended with `campaign_anytime_core.termination=no_action` while their own final decision selected an eligible `recompute_route_closure`. The target runtime had no handler for that compiled action. A read-only scheduler projection over each frozen graph shows that after excluding the stranded closure action, every target still has untried guided work:

| Target | Minimum untried guided frontiers visible after closure action |
|---|---:|
| 006 | 2 |
| 008 | 1 |
| 011 | 4 |
| 012 | 4 |
| 013 | 7 |
| 014 | 4 |
| 015 | 5 |
| 016 | 5 |

Counts are a lower-bound scheduling projection, not provider success claims.

## Frozen execution contract

- Source manifest: `benchmarks/synthatlas_strategy_closure50_v2.v1.json`.
- Target set: exact case IDs `006,008,011,012,013,014,015,016` from the completed V4 snapshot.
- Stock for search and scoring: `SynthAtlas50Stock` at `data_external/retrostar190/retrostar_emolecules_stock.sqlite3`, SHA-256 `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`.
- Same result-first configuration as V4: unified-adaptive, fast profile, one worker, 1,800 s and 128 tasks per target, `stock_result` delivery boundary, credibility tail deferred.
- Launch strategy: ChemEnzy and Codex concurrent; publish completion progressively; cancel Codex only after root B4.
- Fresh output root only. Do not resume or mutate the completed V4 root.

## Production-path preflight

The exact eight-target selection passed the production runner's provider-free
preflight on 2026-08-14:

- preflight root: `results/shared/synthatlas_result_first_stranded_action_canary8_preflight_20260814`;
- selected targets: 8/8, in source-manifest order;
- preflight receipts accepted: 8/8;
- benchmark/provider stock alignment: true;
- benchmark snapshot SHA-256: `8d687fc7bca496dd7e0ef0a54ca3075622386ada9d22f7ad7a396f56582a5221`;
- manifest file SHA-256: `2354f031098e00e2bfa721820436b3041a7e218a0794d1ab9a08f5f0b15ac6ab`;
- stock SHA-256: `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`.

After matched-baseline binding was added, the exact command received a second
provider-free production preflight at
`results/shared/synthatlas_result_first_stranded_action_canary8_preflight_v2_20260814`:
8/8 cases passed, the selected case set matched the digest-valid V4 baseline,
and stock/search alignment remained accepted. Its fresh benchmark snapshot
SHA-256 is `6e78462a77fbd98f04cdec4becada88d736b692db1cda66d7eeea53d0a7faeac`.

No separate eight-target manifest is introduced. The full frozen manifest is
the single target authority; repeated `--only` arguments are recorded in the
panel snapshot and select the matched diagnostic subset.

The final offline reachability audit also confirms that every action kind able
to change B4 before the deferred credibility tail has a production handler in
the selected configuration: target and frontier ChemEnzy expansion, initial
Codex architecture and event replan, host materialization, reaction
validation, stock audit, and route-closure recomputation. `resolve_conflict`
is outside this result canary because it belongs to the explicitly deferred
credibility/evidence tail.

## Acceptance and stop rules

Report every target, including timeout, empty provider result, and runtime failure.

Primary comparisons against each target's frozen V4 record:

1. root B4 transition;
2. whether `recompute_route_closure` executed;
3. guided frontier calls before and after that action;
4. provider raw/admitted proposal counts;
5. terminal reason (must not be `no_action` while an eligible registered result action exists);
6. wall time, native-search units, Codex calls/tokens, and recovery count.

The production runner now writes `panel-summary.json` and `panel-summary.md`
as part of terminal batch delivery. The summary contract is
`v4_blind_panel_summary.v3` and makes the acceptance list directly inspectable:

- one mutually exclusive terminal disposition for every target, including
  pending, timeout, cancellation, failure, and incomplete states;
- one disposition for every provider attempt: host-admitted success, raw empty,
  raw nonempty but host-filtered, timeout, or provider failure;
- a monotonic selected-route result funnel from raw provider output through
  normalization, host admission, provider rule gate, host portfolio selection,
  canonical binding/materialization, and stock closure;
- a separate all-route canonical provenance first-loss table. It includes
  advisory and quarantined routes and is explicitly not interpreted as a serial
  B4 causal funnel;
- per-target `recompute_route_closure` ordering, guided calls before/after the
  first closure action, termination, observed resources, and recovery/replay
  counts.

A read-only recomputation over the frozen V4 panel validated this delivery
surface without changing its 12/20 score: all 20 targets and all 42 provider
attempts were accounted, with zero unaccounted targets. The historical run
predates replay hardening, so its provider replay count correctly remains zero.

The same read-only result-loss records sharpen the canary hypothesis. Across
the frozen panel, 34 selected provider routes are not stock closed, but most
belong to targets that already have another B4 route. After restricting to
root-B4-open targets, only seven routes on five targets remain actionable:
`011,012,013,014,016`. All seven have canonical materialized routes and all
seven stop at stock closure; none stop at provider filtering or canonical
materialization. Targets `006,008,015` remain in the frozen eight-target set
because it was predeclared before this analysis and provides the zero-result /
host-validation controls.

The runner's `--matched-baseline-summary` option digest-validates and records
the baseline before provider execution. Terminal delivery additionally writes
`matched-comparison.json/.md` with per-target B4 transitions, regressions,
closure/guided ordering, provider-attempt deltas, and resource deltas. Any
pending or incomplete candidate target disables the performance claim.

A provider-free canonical counterfactual also rules out a simpler explanation.
For `011,012,013,014,016`, the frozen graph was loaded read-only and passed
through the same full-recompute oracle used by `recompute_derived`, without
publishing a graph revision. All 5/5 projections were no-ops: scientific
digests, selected routes, closure rates, and B4 remained unchanged. Therefore
the repaired action is not expected to manufacture B4 by refreshing derived
state. Its result-bearing effect is to settle the previously stranded action
and let the scheduler reach the untried guided frontiers behind it. The
machine-readable projection is
`results/shared/synthatlas_result_first20_v4_20260814/route-closure-counterfactual-v1.json`.

The canary is informative after all eight targets finish. It may be stopped early only for a shared infrastructure failure affecting two consecutive targets; such a stop yields no performance claim. No threshold or target set may be changed after launch.

## Historical launch command

The command below is the exact live invocation that produced the completed
result. It used a fresh output root and was not resumed or rerun:

```powershell
python scripts/run_v4_blind_panel.py `
  --manifest benchmarks/synthatlas_strategy_closure50_v2.v1.json `
  --output-root results/shared/synthatlas_result_first_stranded_action_canary8_live_20260814 `
  --matched-baseline-summary results/shared/synthatlas_result_first20_v4_20260814/panel-summary-v3-result-accounting.json `
  --model gpt-5.5 --reasoning-effort low --execution-profile fast `
  --fixed-cutoff-wall-time-s 1800 --fixed-cutoff-total-tasks 128 `
  --workers 1 --ablation unified-adaptive `
  --benchmark-stock-index data_external/retrostar190/retrostar_emolecules_stock.sqlite3 `
  --benchmark-stock-name SynthAtlas50Stock `
  --chemenzy-env-prefix D:\conda\envs\py312 `
  --only "opaque benchmark target 006" `
  --only "opaque benchmark target 008" `
  --only "opaque benchmark target 011" `
  --only "opaque benchmark target 012" `
  --only "opaque benchmark target 013" `
  --only "opaque benchmark target 014" `
  --only "opaque benchmark target 015" `
  --only "opaque benchmark target 016"
```
