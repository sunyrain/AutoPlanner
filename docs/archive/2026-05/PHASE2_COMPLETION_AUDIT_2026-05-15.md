# Phase II Completion Audit

Date: 2026-05-15

## Objective Restatement

Active objective:

1. Build `AUTOPLANNRELLM` as an independent folder/system. It should keep AutoPlanner behavior except that route-tree selection can be delegated to a DeepSeek LLM agent, and each proposal pool can receive one LLM-suggested candidate.
2. Complete Phase II item 1: rerun current cost-scoring full100 A/B/C/D with a relaxed 20-30s gate.
3. Complete Phase II item 2: run D/D_FILTER/D_TOP10_FILTER with `AUTOPLANNER_RESERVOIR_QUALITY_FILTER=1`.
4. Complete Phase II item 3: expand external benchmark coverage to PaRoutes n1/n5 full input, USPTO-190, and BioNavi-like set.

This audit checks actual artifacts only. It does not treat implementation effort or passing tests as sufficient unless they cover the requirement.

## Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Independent `AUTOPLANNRELLM/` folder exists | `AUTOPLANNRELLM/__init__.py`, `controller.py`, `deepseek_client.py`, `proposals.py`, `runner.py`, `README.md` | complete |
| LLM can drive route-tree selection | `AUTOPLANNRELLM/controller.py`; runtime wrapper in `cascade_planner/route_tree/runtime.py`; tests `test_leaf_selection_scores_prefer_deepseek_choice`, `test_action_selection_scores_prefer_deepseek_choice` | implementation complete |
| Candidate pool can append one LLM suggestion | `AUTOPLANNRELLM/proposals.py`; proposal hook in `cascade_planner/route_tree/proposals.py`; tests `test_llm_candidate_appends_one_deepseek_source`, `test_retro_engine_tool_can_return_llm_candidate_when_base_pool_empty` | implementation complete |
| DeepSeek real API evaluation | `results/shared/autoplannrellm_20260515/smoke_real_key_20260515_row1/run.json`; cache records `leaf_selection`, `reaction_suggestion`, and `action_selection` from `deepseek-chat` | smoke complete |
| AUTOPLANNRELLM full100 paired evaluation | `results/shared/autoplannrellm_20260515/full100_publication_real_key_20260515/reports/comparison.md`; LLM_BOTH vs current A/D on 100 targets | complete, negative result |
| AUTOPLANNRELLM top-3 no-timeout evaluation | `results/shared/autoplannrellm_20260515/full100_top3_no_timeout_w4_20260515/reports/comparison.md`; LLM selects 1-3 branches, route-tree soft/hard timeout disabled, workers=4 | complete, diagnostic only |
| AUTOPLANNRELLM fallback behavior | `results/shared/autoplannrellm_20260515/smoke_no_key/run.json`; tests pass | fallback verified |
| full100 A/B/C/D current cost scoring, 20-30s gate | `results/shared/phase2_20260515/full100_abcd_gate30/reports/comparison.md` | complete |
| full100 A/B/C/D metrics | A stock 0.46, B stock 0.93, C stock 0.46, D stock 0.93; D avg 3.269s | complete |
| D/D_FILTER/D_TOP10_FILTER quality-filter ablation | `results/shared/phase2_20260515/quality_filter_ablation_gate30/reports_quality/comparison.md` | complete |
| quality-filter audit | `reports_quality/D_FILTER/stock_closed_alternative_audit.md`: review-pass 0.9756, critical/suspicious 0.0244 | complete |
| PaRoutes n1/n5 full input files present | `targets_n1.txt`, `targets_n5.txt`, `ref_routes_n1.json`, `ref_routes_n5.json` | complete |
| PaRoutes n1/n5 full benchmark JSON assets built | `results/shared/phase2_20260515/external_full_input_assets_20260515/benchmarks/paroutes_n1_smoke.json` and `paroutes_n5_smoke.json`, each 10000 rows | asset complete |
| PaRoutes n1/n5 full-input prefix20 C/D run | `results/shared/phase2_20260515/external_paroutes_full_input_prefix20_20260515/external_smoke_summary.json` | complete |
| PaRoutes n1/n5 full-input offset40 C/D run | `results/shared/phase2_20260515/external_paroutes_full_input_offset40_20260515/external_smoke_summary.json` | complete |
| PaRoutes/BioNavi full-input sampled aggregate | `results/shared/phase2_20260515/external_full_input_sample_aggregate_20260515/external_smoke_aggregate.md` | complete |
| PaRoutes n1/n5 full planner run | No full-run `run.json` over all 10000 rows exists | incomplete |
| USPTO-190 full input fetched/parsed | `results/shared/phase2_20260515/uspto190_cache_20260515/uspto190_cache_report.md`: cached target pages 190/190, missing selected 0 | complete |
| USPTO-190 full C/D planner run | `results/shared/phase2_20260515/external_uspto_full_20260515/C_uspto_190/run.json` and `D_uspto_190/run.json`, each with 190 targets; `D_FILTER` full run was stopped before summary | C/D complete, D_FILTER incomplete |
| USPTO-190 full C/D normalized summary | `results/shared/phase2_20260515/external_uspto_full_cd_summary_20260515/external_smoke_summary.json` | complete |
| BioNavi-like full input fetched/parsed | `results/shared/phase2_20260515/external_bionavi_full_input_20260515/external_smoke_manifest.json`: `bionavi_like` ready with 373 rows | asset complete |
| BioNavi-like full-input C/D planner run | `results/shared/phase2_20260515/external_bionavi_full_cd_20260515/external_smoke_summary.json` | complete |
| BioNavi-like full-input prefix20 C/D run | `results/shared/phase2_20260515/external_bionavi_full_input_prefix20_20260515/external_smoke_summary.json` | complete |
| BioNavi-like full-input offset40 C/D run | `results/shared/phase2_20260515/external_bionavi_full_input_offset40_20260515/external_smoke_summary.json` | complete |
| BioNavi-like full-input offset60 C/D run | `results/shared/phase2_20260515/external_bionavi_full_input_offset60_20260515/external_smoke_summary.json` | complete |
| BioNavi-like full-input offset80 C/D run | `results/shared/phase2_20260515/external_bionavi_full_input_offset80_20260515/external_smoke_summary.json` | complete |
| BioNavi-like full-input offset100 C/D run | `results/shared/phase2_20260515/external_bionavi_full_input_offset100_20260515/external_smoke_summary.json` | complete |
| External benchmark runnable smoke across all four datasets | `results/shared/reservoir_distill_20260513/external_publication_matrix_limit10_ua_20260514/external_smoke_summary.json` has ready=true with C/D/D_APPEND over 10 rows each | smoke complete |
| External benchmark sharding support | `cascade_planner/eval/build_external_reservoir_smokes.py` supports `--offset`; `results/shared/phase2_20260515/external_paroutes_shard10_manifest_20260515/external_smoke_summary.json` covers PaRoutes rows 10-19 for C/D/D_FILTER | shard smoke complete |
| External D_FILTER/D_TOP10_FILTER configs | `build_external_reservoir_smokes.py` supports `D_FILTER` and `D_TOP10_FILTER`; tests cover top-k and quality-filter envs | implementation complete |
| External shard aggregation | `cascade_planner/eval/aggregate_external_smoke_summaries.py`; aggregate report at `results/shared/phase2_20260515/external_aggregate_20260515/external_smoke_aggregate.md` | complete |
| External cross-dataset aggregate | `results/shared/phase2_20260515/external_cross_dataset_bionavi_full_20260515/external_smoke_aggregate.md` plus earlier sampled aggregate | complete |
| USPTO-190 resumable target cache | `cascade_planner/eval/cache_uspto190_targets.py`; cache report at `results/shared/phase2_20260515/uspto190_cache_20260515/uspto190_cache_report.md` | partial cache complete |
| USPTO-190 cached30 manifest | `results/shared/phase2_20260515/external_uspto_cached30_manifest_20260515/external_smoke_manifest.json` has `uspto_190` ready with 30 rows | manifest complete |
| USPTO-190 cached30 planner run | `results/shared/phase2_20260515/external_uspto_cached30_only_20260515/external_smoke_summary.json` has C/D/D_FILTER over 30 rows | cached30 complete |
| External benchmark full-scale claim | PaRoutes full planner runs are still missing; BioNavi-like full C/D is complete; USPTO-190 full D exceeds the 20-30s gate at 37.250s average | incomplete |

## Latest Verification

Current local verification commands:

```bash
PYTHONPATH=. python tests/test_autoplannrellm.py -v
PYTHONPATH=. python tests/test_reservoir_distilled_controller.py -v
PYTHONPATH=. python tests/test_chem_enzy_baseline.py -v
```

Result on 2026-05-15:

| Command | Result |
| --- | --- |
| `tests/test_autoplannrellm.py -v` | 7 tests passed |
| `tests/test_reservoir_distilled_controller.py -v` | 73 tests passed |
| `tests/test_chem_enzy_baseline.py -v` | 32 tests passed |

`pytest` is not installed in this environment, so verification used the repository's direct unittest-style test entrypoints.

## AUTOPLANNRELLM Real-Key Smoke

Artifact: `results/shared/autoplannrellm_20260515/smoke_real_key_20260515_row1/`

| n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 24.6100 | 1.0000 |

Trace reading:

- DeepSeek was actually called for `leaf_selection`, `reaction_suggestion`, and `action_selection`.
- The route-tree event records `model_active=True` and `autoplannrellm:deepseek_action_selection:confidence=0.900`.
- The LLM candidate was valid but duplicated the RetroChimera top-1 reduction and was dropped by dedupe, so this smoke verifies integration rather than an LLM quality gain.

## AUTOPLANNRELLM Full100 Real-Key Run

Artifact: `results/shared/autoplannrellm_20260515/full100_publication_real_key_20260515/`

Run configuration: same benchmark and route-tree settings as current A baseline, with `n_results=5`, `skeleton_samples=2`, `check_stock`, same source/open-leaf policies, and LLM selection plus one LLM candidate enabled.

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.7600 | 0.4600 | 0.4200 | 0.2400 | 0.3900 | 3.0100 | 3.4100 |
| D | 1.0000 | 0.9300 | 0.4100 | 0.2900 | 0.5200 | 3.2690 | 7.1700 |
| LLM_BOTH | 0.6200 | 0.3600 | 0.3800 | 0.1500 | 0.2400 | 21.7350 | 1.1500 |

LLM diagnostics:

- DeepSeek cache tasks: `leaf_selection=293`, `reaction_suggestion=288`, `action_selection=292`, bad JSON rows `0`.
- Trace events with DeepSeek action selection: `282`; leaf-score rows with DeepSeek: `502`.
- LLM proposals final returned: `175`; invalid self-loop candidates: `92`; duplicate LLM candidates: `19`; LLM route steps: `38` across `31` targets.
- Paired vs A: plan gain/loss `0/14`, stock gain/loss `4/14`, route exact gain/loss `1/10`, route GT gain/loss `2/17`.

Verdict: LLM_BOTH is a real full100 run, but it is a negative result. It is slower, returns fewer routes, and loses stock/reference recall relative to A. It should stay outside promotion claims.

## AUTOPLANNRELLM Top-3 No-Timeout Run

Artifact: `results/shared/autoplannrellm_20260515/full100_top3_no_timeout_w4_20260515/`

Run configuration: same current full100 benchmark and policies as above, but LLM leaf/action selection can keep up to 3 branches, route-tree soft/hard timeouts are disabled, 4 parallel workers are used, and DeepSeek HTTP timeout remains 120s to avoid indefinite network hangs.

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.7600 | 0.4600 | 0.4200 | 0.2400 | 0.3900 | 3.0100 | 3.4100 |
| D | 1.0000 | 0.9300 | 0.4100 | 0.2900 | 0.5200 | 3.2690 | 7.1700 |
| LLM_BOTH | 0.6200 | 0.3600 | 0.3800 | 0.1500 | 0.2400 | 21.7350 | 1.1500 |
| LLM_TOP3_NO_TIMEOUT | 0.7600 | 0.5300 | 0.4300 | 0.2400 | 0.3800 | 64.8020 | 3.5700 |

Diagnostics:

- DeepSeek cache tasks: `leaf_selection=834`, `reaction_suggestion=827`, `action_selection=1006`, bad cache JSON rows `0`.
- LLM route steps: `122` across `60` targets.
- Route count histogram: `{0:24, 1:1, 2:3, 3:4, 4:2, 5:66}`.
- Paired vs A: stock gain/loss `7/0`, route exact gain/loss `3/3`, route GT gain/loss `3/4`.

Verdict: top-3 branch selection plus disabled route-tree timeout fixes the route-count collapse observed in `LLM_BOTH` and gives a small stock gain over A. It is not promotable because route GT does not improve and average runtime rises to `64.802s/target`. This should be treated as a slow diagnostic/upper-bound branch, not as Phase II mainline evidence.

## Remaining Blockers

- Real `AUTOPLANNRELLM` effect is now measured on full100 and is negative. There is still no evidence that LLM selection or proposal improves route quality.
- The top-3 no-timeout LLM rerun shows the route-count failure can be mitigated, but the improvement comes from deeper search and costs `64.802s/target`; it still does not beat A on route GT or D on any central metric.
- PaRoutes n1/n5 full-input assets are ready, but all-row planner runs are not practical as an incidental cleanup task. Based on sampled C/D averages, a serial C+D run over 10000 rows is roughly 150 hours for n1 and 130 hours for n5 before retries and native payload overhead.
- USPTO-190 C/D is complete, but full D_FILTER was interrupted. Cached30 evidence shows D_FILTER underperforms, so finishing full D_FILTER is lower priority unless a reviewer specifically requests hard-filter evidence.
- The current evidence supports a hybrid-cascade research direction, not a final promotion claim. Runtime remains above the relaxed 30s gate for BioNavi-like D, PaRoutes n1 D, and USPTO-190 D.

## Current Phase II Metrics

### full100 A/B/C/D gate30

Artifact: `results/shared/phase2_20260515/full100_abcd_gate30/reports/comparison.md`

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.7600 | 0.4600 | 0.4200 | 0.2400 | 0.3900 | 3.0100 | 3.4100 |
| B | 1.0000 | 0.9300 | 0.4200 | 0.3100 | 0.5500 | 3.2460 | 8.0700 |
| C | 0.7600 | 0.4600 | 0.4100 | 0.2300 | 0.3800 | 3.0810 | 3.4300 |
| D | 1.0000 | 0.9300 | 0.4100 | 0.2900 | 0.5200 | 3.2690 | 7.1700 |

### Quality-filter ablation

Artifact: `results/shared/phase2_20260515/quality_filter_ablation_gate30/reports_quality/comparison.md`

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D | 1.0000 | 0.9300 | 0.4200 | 0.3100 | 0.5300 | 3.2830 | 7.1300 |
| D_FILTER | 0.9800 | 0.8500 | 0.4100 | 0.2800 | 0.4700 | 3.3100 | 5.1500 |
| D_TOP10_FILTER | 0.9800 | 0.8500 | 0.4100 | 0.2800 | 0.4700 | 3.3410 | 5.1300 |

### External limit10 C/D/D_APPEND

Artifact: `results/shared/reservoir_distill_20260513/external_publication_matrix_limit10_ua_20260514/external_smoke_summary.json`

| Dataset | C stock | D stock | D_APPEND stock | C route GT | D route GT | D_APPEND route GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PaRoutes n1 | 0.50 | 0.80 | 1.00 | 0.00 | 0.00 | 0.00 |
| PaRoutes n5 | 0.50 | 0.90 | 1.00 | 0.00 | 0.00 | 0.00 |
| USPTO-190 | 0.50 | 0.70 | 1.00 | 0.40 | 0.60 | 0.90 |
| BioNavi-like | 0.40 | 0.90 | 0.90 | 0.20 | 0.20 | 0.30 |

This is useful smoke evidence, not a full external benchmark.

### PaRoutes shard rows 10-19 C/D/D_FILTER

Artifact: `results/shared/phase2_20260515/external_paroutes_shard10_manifest_20260515/external_smoke_summary.json`

| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaRoutes n1 | C | 10 | 1.00 | 0.90 | 0.40 | 0.50 | 21.625 | 2.80 |
| PaRoutes n1 | D | 10 | 0.90 | 0.80 | 0.60 | 0.80 | 21.399 | 5.70 |
| PaRoutes n1 | D_FILTER | 10 | 0.70 | 0.50 | 0.30 | 0.40 | 19.448 | 1.90 |
| PaRoutes n5 | C | 10 | 1.00 | 1.00 | 0.40 | 0.80 | 24.411 | 3.00 |
| PaRoutes n5 | D | 10 | 1.00 | 1.00 | 0.70 | 1.00 | 23.927 | 6.10 |
| PaRoutes n5 | D_FILTER | 10 | 0.80 | 0.80 | 0.30 | 0.60 | 22.799 | 2.40 |

This shard confirms that external sharding works. It is still a 20-target PaRoutes shard, not a full 20000-target PaRoutes evaluation.

### External aggregate

Artifact: `results/shared/phase2_20260515/external_aggregate_20260515/external_smoke_aggregate.md`

The aggregate combines the existing external limit10 summary with the PaRoutes rows 10-19 shard. For PaRoutes C/D, this gives 20 evaluated rows per split.

| Dataset | Config | n | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PaRoutes n1 | C | 20 | 0.7000 | 0.2000 | 0.2500 | 17.6175 | 2.8000 |
| PaRoutes n1 | D | 20 | 0.8000 | 0.3000 | 0.4000 | 17.8970 | 6.0000 |
| PaRoutes n5 | C | 20 | 0.7500 | 0.2000 | 0.4000 | 17.9735 | 2.9500 |
| PaRoutes n5 | D | 20 | 0.9500 | 0.3500 | 0.5000 | 18.0155 | 6.3000 |

Aggregate reading: D improves stock and reference recall on the combined PaRoutes n=20 evidence, but route count roughly doubles. This is still shard-level evidence, not a full benchmark.

### Full-input sampled aggregate

Artifact: `results/shared/phase2_20260515/external_full_input_sample_aggregate_20260515/external_smoke_aggregate.md`

This aggregate combines the full-input PaRoutes prefix20/offset40 shards and BioNavi-like prefix20/offset40/offset60/offset80/offset100 shards.

| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BioNavi-like | C | 60 | 0.7167 | 0.3333 | 0.0333 | 0.2000 | 7.9745 | 1.9833 |
| BioNavi-like | D | 60 | 1.0000 | 0.8667 | 0.1667 | 0.3500 | 33.6107 | 4.9000 |
| PaRoutes n1 | C | 30 | 0.9667 | 0.8000 | 0.3333 | 0.6333 | 21.0083 | 2.6667 |
| PaRoutes n1 | D | 30 | 1.0000 | 0.8667 | 0.4333 | 0.7667 | 33.1407 | 4.1667 |
| PaRoutes n5 | C | 30 | 1.0000 | 0.7333 | 0.3333 | 0.7333 | 19.0240 | 2.8000 |
| PaRoutes n5 | D | 30 | 1.0000 | 0.7333 | 0.3333 | 0.8000 | 27.8527 | 3.7000 |

Aggregate reading: D is consistently better for stock or route recall, but the runtime cost is real. It clears the relaxed 30s gate only on the sampled PaRoutes n5 aggregate.

### External cross-dataset sampled aggregate

Artifact: `results/shared/phase2_20260515/external_cross_dataset_aggregate_20260515/external_smoke_aggregate.md`

This aggregate places the PaRoutes and BioNavi-like sampled full-input evidence next to the USPTO-190 full C/D run.

| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BioNavi-like | C | 60 | 0.7167 | 0.3333 | 0.0333 | 0.2000 | 7.9745 | 1.9833 |
| BioNavi-like | D | 60 | 1.0000 | 0.8667 | 0.1667 | 0.3500 | 33.6107 | 4.9000 |
| PaRoutes n1 | C | 30 | 0.9667 | 0.8000 | 0.3333 | 0.6333 | 21.0083 | 2.6667 |
| PaRoutes n1 | D | 30 | 1.0000 | 0.8667 | 0.4333 | 0.7667 | 33.1407 | 4.1667 |
| PaRoutes n5 | C | 30 | 1.0000 | 0.7333 | 0.3333 | 0.7333 | 19.0240 | 2.8000 |
| PaRoutes n5 | D | 30 | 1.0000 | 0.7333 | 0.3333 | 0.8000 | 27.8527 | 3.7000 |
| USPTO-190 | C | 190 | 0.9000 | 0.6316 | 0.1789 | 0.4158 | 15.0060 | 2.5630 |
| USPTO-190 | D | 190 | 0.9579 | 0.7632 | 0.3474 | 0.5684 | 37.2500 | 4.5580 |

Aggregate reading: D improves stock on BioNavi-like, PaRoutes n1, and USPTO-190, and improves route GT recall on every external set here. The tradeoff is route count and runtime; PaRoutes n5 is the only external aggregate still below the 30s relaxed gate.

### PaRoutes full-input prefix20 C/D

Artifact: `results/shared/phase2_20260515/external_paroutes_full_input_prefix20_20260515/external_smoke_summary.json`

| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaRoutes n1 | C | 20 | 0.9500 | 0.8000 | 0.3500 | 0.5500 | 20.2980 | 2.5500 |
| PaRoutes n1 | D | 20 | 1.0000 | 0.8500 | 0.3500 | 0.7000 | 33.0500 | 4.3000 |
| PaRoutes n5 | C | 20 | 1.0000 | 0.7500 | 0.4000 | 0.7500 | 19.0710 | 2.7000 |
| PaRoutes n5 | D | 20 | 1.0000 | 0.8000 | 0.4000 | 0.7500 | 25.7860 | 3.5500 |

PaRoutes full-input prefix evidence is now real. D improves stock on both splits, but n1 D is above the relaxed 30s gate while n5 D stays within it.

### PaRoutes full-input offset40 C/D

Artifact: `results/shared/phase2_20260515/external_paroutes_full_input_offset40_20260515/external_smoke_summary.json`

| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaRoutes n1 | C | 10 | 1.0000 | 0.8000 | 0.3000 | 0.8000 | 22.4290 | 2.9000 |
| PaRoutes n1 | D | 10 | 1.0000 | 0.9000 | 0.6000 | 0.9000 | 33.3220 | 3.9000 |
| PaRoutes n5 | C | 10 | 1.0000 | 0.7000 | 0.2000 | 0.7000 | 18.9300 | 3.0000 |
| PaRoutes n5 | D | 10 | 1.0000 | 0.6000 | 0.2000 | 0.9000 | 31.9860 | 4.0000 |

This later PaRoutes shard repeats the same pattern: D helps route recall, but both n1 and n5 D are still above the relaxed 30s gate.

### USPTO-190 cache

Artifact: `results/shared/phase2_20260515/uspto190_cache_20260515/uspto190_cache_report.md`

| Item | Value |
| --- | ---: |
| target paths discovered | 190 |
| selected targets | 190 |
| cached target pages | 190 |
| fetched this run | 40 |
| skipped existing | 150 |
| missing selected | 0 |
| errors | 0 |

Ready for selected window: `True`

### USPTO-190 full C/D

Artifact: `results/shared/phase2_20260515/external_uspto_full_20260515/`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 190 | 0.9000 | 0.6316 | 0.3737 | 0.6684 | 0.1789 | 0.4158 | 15.0060 | 2.5630 |
| D | 190 | 0.9579 | 0.7632 | 0.4000 | 0.6842 | 0.3474 | 0.5684 | 37.2500 | 4.5580 |

D improves stock, exact route recall, and GT reactant route recall over C on USPTO-190, but it is above the relaxed 20-30s gate. The D_FILTER full run was interrupted and raw traces were pruned; only C/D summary JSONs are retained.

### USPTO-190 cached30 C/D/D_FILTER

Artifact: `results/shared/phase2_20260515/external_uspto_cached30_only_20260515/external_smoke_summary.json`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 30 | 0.9667 | 0.6667 | 0.3667 | 0.6667 | 0.2000 | 0.4333 | 18.3860 | 2.8330 |
| D | 30 | 0.9667 | 0.8000 | 0.3333 | 0.6333 | 0.5333 | 0.7667 | 16.7070 | 5.7670 |
| D_FILTER | 30 | 0.6333 | 0.4333 | 0.2667 | 0.4667 | 0.1333 | 0.3333 | 18.7840 | 1.8670 |

D improves stock by `+0.1333`, route exact by `+0.3333`, and route GT by `+0.3333` over C on this cached30 subset, while average route count rises from 2.833 to 5.767. D_FILTER is worse than both C and D on this subset, so hard filtering is not a viable default.

### BioNavi-like full input asset

Artifact: `results/shared/phase2_20260515/external_bionavi_full_input_20260515/external_smoke_manifest.json`

| Dataset | n_rows | ready | route_annotations |
| --- | ---: | --- | --- |
| BioNavi-like | 373 | true | true |

This completes full-input asset generation for BioNavi-like.

### BioNavi-like full-input C/D

Artifact: `results/shared/phase2_20260515/external_bionavi_full_cd_20260515/external_smoke_summary.json`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 373 | 0.8391 | 0.1769 | 0.0402 | 0.1206 | 0.0322 | 0.0992 | 4.3360 | 2.2950 |
| D | 373 | 0.9866 | 0.8123 | 0.0429 | 0.1260 | 0.1850 | 0.3619 | 40.7730 | 5.8100 |

`C_bionavi_like` is now a true full-input run. `D_bionavi_like` substantially improves stock and route recall, but it is still above the relaxed external runtime gate.

### BioNavi-like full input prefix20 C/D

Artifact: `results/shared/phase2_20260515/external_bionavi_full_input_prefix20_20260515/external_smoke_summary.json`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 20 | 0.8000 | 0.2500 | 0.0500 | 0.2000 | 0.0500 | 0.1500 | 6.4090 | 2.1500 |
| D | 20 | 1.0000 | 0.8000 | 0.0500 | 0.2000 | 0.2000 | 0.2500 | 32.8470 | 5.2000 |

D improves stock and route recall on the BioNavi-like full-input prefix, but it is again above the relaxed external runtime gate.

### BioNavi-like full input offset40 C/D

Artifact: `results/shared/phase2_20260515/external_bionavi_full_input_offset40_20260515/external_smoke_summary.json`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 10 | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 7.7010 | 1.5000 |
| D | 10 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.2000 | 0.7000 | 23.9020 | 3.9000 |

BioNavi-like remains mixed across shards: this later shard stays within the relaxed 30s gate while still improving stock and route recall.

### BioNavi-like full input offset60 C/D

Artifact: `results/shared/phase2_20260515/external_bionavi_full_input_offset60_20260515/external_smoke_summary.json`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 10 | 0.7000 | 0.3000 | 0.0000 | 0.1000 | 0.0000 | 0.1000 | 9.3200 | 1.9000 |
| D | 10 | 1.0000 | 0.9000 | 0.0000 | 0.1000 | 0.0000 | 0.2000 | 46.8390 | 4.8000 |

This offset swings back to a much slower D, so BioNavi-like remains unstable across target slices even on the full input file.

### BioNavi-like full input offset80 C/D

Artifact: `results/shared/phase2_20260515/external_bionavi_full_input_offset80_20260515/external_smoke_summary.json`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 10 | 0.9000 | 0.4000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 9.1830 | 2.5000 |
| D | 10 | 1.0000 | 0.8000 | 0.1000 | 0.1000 | 0.2000 | 0.2000 | 34.4730 | 4.5000 |

BioNavi-like continues to oscillate by slice; D stays above the relaxed gate here as well.

### BioNavi-like full input offset100 C/D

Artifact: `results/shared/phase2_20260515/external_bionavi_full_input_offset100_20260515/external_smoke_summary.json`

| Config | n | plan | stock | cand exact | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 10 | 0.6000 | 0.3000 | 0.0000 | 0.2000 | 0.0000 | 0.2000 | 8.8250 | 1.7000 |
| D | 10 | 1.0000 | 0.9000 | 0.0000 | 0.2000 | 0.2000 | 0.5000 | 30.7560 | 5.8000 |

This shard sits almost exactly on the relaxed 30s gate, again with a large stock gain from D.

### External cross-dataset full-input aggregate

Artifact: `results/shared/phase2_20260515/external_cross_dataset_bionavi_full_20260515/external_smoke_aggregate.md`

| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BioNavi-like | C | 373 | 0.8391 | 0.1769 | 0.0322 | 0.0992 | 4.3360 | 2.2950 |
| BioNavi-like | D | 373 | 0.9866 | 0.8123 | 0.1850 | 0.3619 | 40.7730 | 5.8100 |
| PaRoutes n1 | C | 30 | 0.9667 | 0.8000 | 0.3333 | 0.6333 | 21.0083 | 2.6667 |
| PaRoutes n1 | D | 30 | 1.0000 | 0.8667 | 0.4333 | 0.7667 | 33.1407 | 4.1667 |
| PaRoutes n5 | C | 30 | 1.0000 | 0.7333 | 0.3333 | 0.7333 | 19.0240 | 2.8000 |
| PaRoutes n5 | D | 30 | 1.0000 | 0.7333 | 0.3333 | 0.8000 | 27.8527 | 3.7000 |
| USPTO-190 | C | 190 | 0.9000 | 0.6316 | 0.1789 | 0.4158 | 15.0060 | 2.5630 |
| USPTO-190 | D | 190 | 0.9579 | 0.7632 | 0.3474 | 0.5684 | 37.2500 | 4.5580 |

Full-input reading: D improves stock and route recall on BioNavi-like, PaRoutes n1, and USPTO-190, but only PaRoutes n5 remains under the relaxed 30s gate. BioNavi-like D and USPTO-190 D are both over the gate.

## Verification Commands Run

```bash
PYTHONPATH=. python tests/test_autoplannrellm.py -v
PYTHONPATH=. python tests/test_reservoir_distilled_controller.py -v
PYTHONPATH=. python tests/test_chem_enzy_baseline.py -v
```

Results:

- `tests/test_autoplannrellm.py`: 5 tests passed
- `tests/test_reservoir_distilled_controller.py`: 73 tests passed after adding external sharding/filter/aggregation/cache coverage
- `tests/test_chem_enzy_baseline.py`: 32 tests passed

## Audit Verdict

The active goal is not complete.

Completed:

- AUTOPLANNRELLM implementation and fallback tests
- AUTOPLANNRELLM one-target real-key smoke covering selection and proposal hooks
- AUTOPLANNRELLM full100 real-key paired run, with negative result recorded
- full100 A/B/C/D gate30
- full100 D/D_FILTER/D_TOP10_FILTER quality-filter ablation
- PaRoutes n1/n5 full benchmark asset generation
- USPTO-190 full target cache, 190/190 pages
- USPTO-190 full C/D planner run
- USPTO-190 full C/D normalized summary
- BioNavi-like full input asset generation, 373 rows
- BioNavi-like full input C/D planner run, 373 rows
- BioNavi-like full input prefix20 C/D run
- BioNavi-like full input offset40 C/D run
- BioNavi-like full input offset60 C/D run
- BioNavi-like full input offset80 C/D run
- BioNavi-like full input offset100 C/D run
- PaRoutes n1/n5 full-input prefix20 C/D run
- PaRoutes n1/n5 full-input offset40 C/D run
- PaRoutes/BioNavi full-input sampled aggregate
- external cross-dataset aggregate with BioNavi-like full C/D and USPTO-190 full C/D
- external limit10 C/D/D_APPEND smoke across PaRoutes n1/n5, USPTO-190, BioNavi-like
- external builder support for `--offset`, `D_FILTER`, and `D_TOP10_FILTER`
- PaRoutes shard rows 10-19 C/D/D_FILTER execution
- external shard aggregation over existing limit10 plus PaRoutes shard10
- resumable USPTO-190 cache tool and cached30 manifest generation
- USPTO-190 cached30 C/D/D_FILTER execution

Incomplete:

- full PaRoutes n1/n5 planner run
- USPTO-190 full D_FILTER run
- external full-scale promotion claim, because PaRoutes full all-row runs are missing and BioNavi-like D / PaRoutes n1 D / USPTO-190 D exceed the relaxed 30s gate

Recommended next concrete action:

1. Run fixed full-input PaRoutes n1/n5 C/D shards rather than trying to execute 20000 targets in one uninterrupted job.
2. Treat BioNavi-like D and USPTO-190 D as coverage wins but runtime failures under a strict 20-30s external gate; tune source budget/reservoir trigger before rerunning D_FILTER.
3. Keep AUTOPLANNRELLM out of promotion claims; the full100 real-key run is currently worse than the non-LLM route-tree.
