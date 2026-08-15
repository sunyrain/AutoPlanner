# Result-first 20-target V4 run manifest

Status: complete

## Final result at 20/20

- All 20 manifest-ordered targets completed with zero terminal runtime
  failures and without reusing a prior target completion.
- B1 structural route presence: 20/20; B2 host reaction validation: 2/20;
  B4 official benchmark stock closure: 12/20.
- Median end-to-end target time was 269.093 seconds; P95 was 575.941
  seconds.  Median time to B4 among the 12 solved targets was 247.434
  seconds and P95 was 302.208 seconds.
- No target that remained B4-open spent work on the deferred credibility
  tail.  Exact evidence and configured-proof counts are therefore intentionally
  zero in this result-first run, not evidence that the proof layer was run and
  failed.
- Across 42 native and guided provider attempts, 23 had a host-admitted
  solution.  Twenty-two of those 23 already had one in the raw top four and
  all 23 did in the raw top eight.
- The frozen cascade gate quarantined 91 stock-closed routes whose recorded
  verifier reason was atom-balance only.  Read-only root-target analysis finds
  two official B4 failures (targets 012 and 016) with direct stock-closed
  counterfactual routes.  This does not change the official 12/20 score.
- Five B4-open targets had a guided ChemEnzy subtarget solved locally while
  their root route remained stock-open.  These are frontier-continuation
  signals, not additional solved targets.

Machine-readable results are in `panel-summary.json`; the compact report is
in `panel-summary.md` under the frozen output root.

## Interim result at 10/20

- Ten targets completed with zero terminal runtime failures.
- B1 structural route presence: 10/10; B2 host reaction validation: 2/10;
  B4 official benchmark stock closure: 8/10.
- The two B4-open cases are distinct: target 006 had one structural route but
  no stock closure after native plus guided recovery; target 008 had one
  host-validated Codex route whose leaves remained outside the frozen stock.
- No completed target performed credibility-tail work before an unreached B4,
  and no provider-to-host integration loss was detected.
- Across the first 12 host-admitted provider attempts, 11 already contained an
  admitted output in the raw top four and 12/12 did so in the raw top eight.
  This is evidence for a future progressive 4 -> 8 -> 16 reserve, not a change
  to the frozen V4 panel configuration.

These are interim completed-target rates, not the final 20-target estimate.

## Restart gate

The stock-aligned target-001 canary at
`results/shared/synthatlas_result_first_stock_aligned_canary_001_20260814`
passed the restart gate:

- ChemEnzy and B4 both used `SynthAtlas50Stock` at the exact same SQLite path;
- native search returned 32 raw successful routes in 183.3 seconds and stopped
  at its bounded success reserve;
- the rule gate retained three routes and the host materialized three
  target-rooted skeletons;
- two routes were strict stock-closed, with B4 first reached at 258.3 seconds;
- no guided ChemEnzy search or credibility-tail work was needed before closeout.

## Frozen identity

- Output root: `results/shared/synthatlas_result_first20_v4_20260814`
- Manifest: `benchmarks/synthatlas_strategy_closure50_v2.v1.json`
- Target selection: first 20 cases in manifest order; no prior completion reused.
- Benchmark/provider stock: `SynthAtlas50Stock`, both bound to
  `data_external/retrostar190/retrostar_emolecules_stock.sqlite3`.
- Frozen stock SHA-256:
  `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`
- One panel worker; `gpt-5.5` low; fast profile; 1,800-second and 128-task
  fixed cutoffs.
- Delivery boundary: `stock_result`; credibility work remains deferred.

The new preflight fails closed if an explicit ChemEnzy stock path differs from
the benchmark scoring stock. Panel status additionally records the resolved
binding and equality assertion.

## Reproduction

```powershell
python scripts/run_v4_blind_panel.py `
  --manifest benchmarks/synthatlas_strategy_closure50_v2.v1.json `
  --output-root results/shared/synthatlas_result_first20_v4_20260814 `
  --model gpt-5.5 --reasoning-effort low --execution-profile fast `
  --fixed-cutoff-wall-time-s 1800 --fixed-cutoff-total-tasks 128 `
  --workers 1 --max-targets 20 --ablation unified-adaptive `
  --benchmark-stock-index data_external/retrostar190/retrostar_emolecules_stock.sqlite3 `
  --benchmark-stock-name SynthAtlas50Stock `
  --chemenzy-env-prefix D:\conda\envs\py312
```
