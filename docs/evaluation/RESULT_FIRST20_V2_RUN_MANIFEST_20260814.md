# Result-first 20-target V2 run manifest

Status: invalidated and intentionally stopped before any target completed

The first target reproduced the full 300-second provider hard timeout without
a result artifact. This root is diagnostic only and must not be resumed. See
`RESULT_FIRST20_V2_INVALIDATION_20260814.md`.

## Frozen run

- Output root: `results/shared/synthatlas_result_first20_v2_20260814`
- Target selection: first 20 cases, in manifest order, from
  `benchmarks/synthatlas_strategy_closure50_v2.v1.json`
- No V1 target result is reused.
- Ablation: `unified-adaptive`
- Execution profile: `fast`
- Panel workers: 1
- Codex model: `gpt-5.5`, reasoning effort `low`
- Case cutoff: 1,800 seconds / 128 settled tasks
- ChemEnzy native and guided profile: depth 6, 100 iterations, top-k 50,
  300 seconds; guided search receives two frontiers.
- Host route portfolio: 16
- Benchmark stock:
  `data_external/retrostar190/retrostar_emolecules_stock.sqlite3`
- Frozen stock SHA-256:
  `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`
- Frozen panel snapshot SHA-256:
  `47fcd03018f36d2641a13511b834d0719de60ddea33b7dd6e66cae090a3187dc`

## Replacement boundary

V2 includes the general guided-route integration repair described in
`RESULT_FIRST20_V1_INVALIDATION_20260814.md`. The replacement root is fresh so
that every target uses one implementation. While V2 is active, solver and
canonical-ingestion code is frozen; read-only reporting may continue.

The primary result is B1/B2/B4, route counts, time to first route/B4, provider
status, and resources. Exact evidence, condition completeness, conflicts, and
Program work remain deferred until after B4 and are not early success gates.

## Reproduction command

```powershell
python scripts/run_v4_blind_panel.py `
  --manifest benchmarks/synthatlas_strategy_closure50_v2.v1.json `
  --output-root results/shared/synthatlas_result_first20_v2_20260814 `
  --model gpt-5.5 --reasoning-effort low --execution-profile fast `
  --fixed-cutoff-wall-time-s 1800 --fixed-cutoff-total-tasks 128 `
  --workers 1 --max-targets 20 --ablation unified-adaptive `
  --benchmark-stock-index data_external/retrostar190/retrostar_emolecules_stock.sqlite3 `
  --benchmark-stock-name SynthAtlas50Stock `
  --chemenzy-env-prefix D:\conda\envs\py312
```

Resume is permitted only after an external interruption and only with the same
command plus `--resume`. A code or configuration change requires another fresh
root.

## Read-only summary

```powershell
python scripts/summarize_v4_blind_panel.py `
  --panel-root results/shared/synthatlas_result_first20_v2_20260814 `
  --output results/shared/synthatlas_result_first20_v2_20260814/panel-summary.live.json
```

The summary reports canonical-provider integration loss explicitly and retains
queued, running, failed, partial, and timeout cases in the full-panel
denominator.
