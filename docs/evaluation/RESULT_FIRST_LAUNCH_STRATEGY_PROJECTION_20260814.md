# Result-first launch strategy projection (2026-08-14)

## Decision

Freeze **concurrent start + progressive completion + cancel Codex only after root B4** as the next matched-canary launch strategy. Do not switch to native-first.

This is a read-only counterfactual projection over the frozen V4 20-target telemetry. Route outcomes are held fixed; it is not a new provider run and does not rewrite the official 12/20 B4 result.

## Frozen comparison

| Strategy | B4 | mean B4 (s) | P95 B4 (s) | mean terminal (s) | P95 terminal (s) | initial Codex dispatched / completed |
|---|---:|---:|---:|---:|---:|---:|
| Frozen batch delivery | 12/20 | 253.556 | 291.885 | 331.816 | 581.519 | 20 / 20 |
| Concurrent progressive | 12/20 | 201.190 | 278.631 | 300.388 | 581.519 | 20 / 11 |
| Native-first | 12/20 | 258.896 | 502.081 | 424.751 | 808.895 | 11 / 11 |

Relative to concurrent progressive delivery, native-first adds 124.363 s to mean terminal latency and 227.375 s to P95 terminal latency. On the eight failed targets it adds a mean 224.348 s provider wait before Codex can begin. Avoiding nine Codex dispatches therefore does not compensate for the failed-target latency penalty.

The progressive projection cancels nine already-dispatched Codex tasks after provider-native B4. This is dispatch/compute avoidance, not a claim of billing avoidance: whether a cancelled request incurs provider charges is external to the frozen telemetry.

## Reproduction

```powershell
python scripts/project_result_first_strategies.py `
  --panel-root results/shared/synthatlas_result_first20_v4_20260814 `
  --output results/shared/synthatlas_result_first20_v4_20260814/launch-strategy-projection.json
```

Machine-readable result: `results/shared/synthatlas_result_first20_v4_20260814/launch-strategy-projection.json`.

## Next result-bearing test

When a new live provider run is explicitly authorized, use only a small matched canary: frozen V4 settings versus concurrent-progressive delivery on the same targets. Native-first is not promoted to a live arm unless later telemetry removes the roughly 224 s failed-target provider wait.
