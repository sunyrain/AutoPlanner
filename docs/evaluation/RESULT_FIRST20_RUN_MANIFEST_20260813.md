# Result-first 20-target run manifest

Status: invalidated and intentionally stopped after 3 completed targets

This root is diagnostic evidence only. It must not be resumed, merged with a
replacement panel, or used as a success-rate estimate. Two of the three
completed targets contain ChemEnzy routes whose canonical edges were admitted
but whose guided parent route-family binding was lost across the materialization
worker boundary. See `RESULT_FIRST20_V1_INVALIDATION_20260814.md`.

## Frozen run

- Output root: `results/shared/synthatlas_result_first20_v1_20260813`
- Target selection: first 20 cases in
  `benchmarks/synthatlas_strategy_closure50_v2.v1.json`
- Selection mode: manifest order, no post-result selection
- Ablation: `unified-adaptive`
- Execution profile: `fast`
- Panel workers: 1
- Codex model: `gpt-5.5`, reasoning effort `low`
- Case cutoff: 1,800 seconds / 128 settled tasks
- ChemEnzy native: max depth 6, 100 iterations, top-k 50, 300 seconds
- ChemEnzy guided: two frontiers, max depth 6, 100 iterations, top-k 50,
  300 seconds per frontier
- Host route portfolio: 16
- Benchmark stock:
  `data_external/retrostar190/retrostar_emolecules_stock.sqlite3`
- Frozen stock SHA-256:
  `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`
- Branch at launch: `agent/unify-workspace-entrypoints`
- HEAD at launch: `fd606da2ab4fd6ce93177c245da2bb2aef9f3651`
- Worktree state: intentionally dirty; this panel binds the benchmark snapshot
  generated under its output root and is not claimed to be a clean-commit
  publication run.

## Result-first reporting contract

Primary interim fields are B1, B2, B4, target-rooted route count,
host-validated route count, stock-closed route count, time-to-first route,
time-to-B4, ChemEnzy invocations, Codex calls/tokens, and action counts.

C2-C6 credibility reporting remains available but does not control early
interpretation. A target that has not reached B4 must have zero evidence,
condition, conflict, or Program actions in the adaptive action trajectory.

## Reproduction command

```powershell
python scripts/run_v4_blind_panel.py `
  --manifest benchmarks/synthatlas_strategy_closure50_v2.v1.json `
  --output-root results/shared/synthatlas_result_first20_v1_20260813 `
  --model gpt-5.5 --reasoning-effort low --execution-profile fast `
  --fixed-cutoff-wall-time-s 1800 --fixed-cutoff-total-tasks 128 `
  --workers 1 --max-targets 20 --ablation unified-adaptive `
  --benchmark-stock-index data_external/retrostar190/retrostar_emolecules_stock.sqlite3 `
  --benchmark-stock-name SynthAtlas50Stock `
  --chemenzy-env-prefix D:\conda\envs\py312
```

Do not resume this root. A fresh replacement root is required after the
route-family propagation fix.

## Live read-only summary

```powershell
python scripts/summarize_v4_blind_panel.py `
  --panel-root results/shared/synthatlas_result_first20_v1_20260813 `
  --output results/shared/synthatlas_result_first20_v1_20260813/panel-summary.live.json
```

The summary keeps queued, running, failed, timeout, empty, and partial targets
in the full-panel denominator.
