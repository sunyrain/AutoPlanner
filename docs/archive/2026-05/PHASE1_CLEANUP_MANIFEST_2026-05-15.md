# Phase I Cleanup Manifest

Date: 2026-05-15

This cleanup removed only temporary, interrupted, or clearly superseded generated artifacts. It did not remove model checkpoints, distillation packs, full matrix results, current full100 gate30 results, quality-filter reports, external benchmark input files, or vendor runtime files.

## Removed

Cache directories:

- all `__pycache__/`
- all `.pytest_cache/`

Superseded phase-I duplicate drafts:

- `docs/PHASE1_RESEARCH_CLOSURE_2026-05-14.md`
- `docs/PHASE1_CLEANUP_MANIFEST_2026-05-14.md`

Interrupted phase2 partial run:

- `results/shared/phase2_20260515/external_expanded_limit20_cd/`
- `results/shared/phase2_20260515/external_bionavi_full_20260515/`
- `results/shared/phase2_20260515/external_paroutes_full_input_offset20_20260515/`
- raw traces, logs, and generated benchmark inputs pruned from `results/shared/phase2_20260515/external_uspto_full_20260515/`; result JSONs and shard summaries retained

AUTOPLANNRELLM temporary smoke artifacts:

- `results/shared/autoplannrellm_20260515/smoke_real_key_20260515/`
- `results/shared/autoplannrellm_20260515/deepseek_client_smoke_cache.jsonl`
- `results/shared/autoplannrellm_20260515/benchmarks/gold_smoke1.json`

The interrupted run had only completed `C_paroutes_n1` and was still running the first `D_paroutes_n1` native payload build when stopped. It is not used as first-stage evidence.
The interrupted BioNavi-like build had only emitted partial PaRoutes smoke assets and a small USPTO source cache before being stopped. It is not used as first-stage evidence.
The interrupted PaRoutes offset20 full-input shard had started but was stopped before completion, so only the first shard traces were briefly produced and then removed.
The USPTO-190 full run was cut back to summary outputs only so the phase-II evidence stays compact.
The first AUTOPLANNRELLM real-key smoke used a target that never entered route-tree expansion, so it was removed to avoid confusing it with LLM-controller evidence. The retained real-key smoke is the later row1 route-tree run.

Controller-v2 debug directories:

- `results/shared/controller_v2_20260512/debug_current_baseline_l20/`
- `results/shared/controller_v2_20260512/debug_default_controller_l20/`
- `results/shared/controller_v2_20260512/debug_single_node_off_l20/`
- `results/shared/controller_v2_20260512/debug_stock_terminal_heuristic_l20/`
- all `results/shared/controller_v2_20260512/full100_matrix_stock_terminal_fix/debug*/`

Approximate generated artifact space removed:

- interrupted phase2 partial: about 123 MB
- interrupted BioNavi-like partial: about 35 MB
- interrupted PaRoutes offset20 partial: small, discarded before summary
- pruned raw USPTO-190 full run: about 1.9 GB reclaimed
- controller-v2 debug directories: about 550 MB
- Python/pytest caches: small, environment-dependent

## Retained

Core code:

- `cascade_planner/route_tree/`
- `cascade_planner/cascade_search/`
- `cascade_planner/eval/`
- `AUTOPLANNRELLM/`
- `scripts/`
- `tests/`

Core evidence artifacts:

- `results/shared/reservoir_distill_20260513/full100_acceptance_real_v2/`
- `results/shared/phase2_20260515/full100_abcd_gate30/`
- `results/shared/phase2_20260515/quality_filter_ablation_gate30/`
- `results/shared/phase2_20260515/external_full_input_assets_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_input_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_cd_20260515/`
- `results/shared/phase2_20260515/external_paroutes_shard10_manifest_20260515/`
- `results/shared/phase2_20260515/external_aggregate_20260515/`
- `results/shared/phase2_20260515/external_uspto_cached30_manifest_20260515/`
- `results/shared/phase2_20260515/external_uspto_cached30_only_20260515/`
- `results/shared/phase2_20260515/external_uspto_cached110_shard30_20260515/`
- `results/shared/phase2_20260515/external_uspto_cached150_shard70_20260515/`
- `results/shared/phase2_20260515/external_uspto_aggregate_20260515/`
- `results/shared/phase2_20260515/external_uspto_full_20260515/` summary JSONs only
- `results/shared/phase2_20260515/external_uspto_full_cd_summary_20260515/`
- `results/shared/phase2_20260515/uspto190_cache_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_input_prefix20_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_input_offset40_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_input_offset60_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_input_offset80_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_input_offset100_20260515/`
- `results/shared/phase2_20260515/external_paroutes_full_input_prefix20_20260515/`
- `results/shared/phase2_20260515/external_paroutes_full_input_offset40_20260515/`
- `results/shared/phase2_20260515/external_full_input_sample_aggregate_20260515/`
- `results/shared/phase2_20260515/external_cross_dataset_bionavi_full_20260515/`
- `results/shared/phase2_20260515/external_cross_dataset_aggregate_20260515/`
- `results/shared/reservoir_distill_20260513/cost_scoring_full100_current_20260514_w2_clean/`
- `results/shared/reservoir_distill_20260513/external_publication_matrix_limit10_ua_20260514/`
- `results/shared/chem_enzy_baseline/`
- `results/shared/autoplannrellm_20260515/smoke_no_key/`
- `results/shared/autoplannrellm_20260515/smoke_real_key_20260515_row1/`
- `results/shared/autoplannrellm_20260515/benchmarks/benchmark_v2_row1.json` as the tiny input file for the retained real-key smoke

External benchmark inputs:

- `targets_n1.txt`
- `targets_n5.txt`
- `ref_routes_n1.json`
- `ref_routes_n5.json`

Runtime/vendor:

- `vendor/`

## Verification

After cleanup:

- no active `build_external_reservoir_smokes`, `run_chem_enzy_smoke`, or `run_live_benchmark_parallel` process remained
- no `__pycache__/` or `.pytest_cache/` directory remained
- `results/shared/phase2_20260515/external_expanded_limit20_cd/` no longer existed
- no `results/shared/controller_v2_20260512/**/debug*` directory remained
- the only retained AUTOPLANNRELLM real-key result artifact is `results/shared/autoplannrellm_20260515/smoke_real_key_20260515_row1/`

## 2026-05-15 Final Repository Sweep

This second sweep removed obsolete training caches, superseded exploratory artifacts, duplicate shard outputs, and old archived snapshots after the first-stage report was completed.

Additional removed high-volume generated artifacts:

- `archive/`
- `results/shared/vnext_feature_cache/`
- `results/shared/vnext_pack/`
- `results/shared/external_candidate_pools/`
- `results/shared/external_step_pairs/`
- `results/shared/proposal_recall/`
- `results/shared/training_pack/`
- `results/shared/route_tree_traces/`
- `results/shared/route_tree_experiment_20260508/`
- `results/shared/route_tree_experiment_20260509/`
- `results/shared/cascade_search_benchmark/`
- `results/shared/coverage_fix_20260511/`
- `results/shared/chem_enzy_internal_cost_20260509/`
- `results/shared/dataset_v4_release/`
- `results/shared/test_runs/`
- `results/shared/enzymemap_training_data/`
- `results/shared/skeleton_hard_negatives/`
- `results/shared/skeleton_prior_split/`
- `results/shared/skeleton_reranker/`
- `results/shared/open_leaf_policy_20260511/`
- old `results/shared/vnext*` model/output directories
- `results/v2/`

Additional removed obsolete top-level weights and cache files:

- `results/shared/enzyformer_retro_finetuned.pt`
- `results/shared/enzyformer_retro_v2.pt`
- `results/shared/enzyformer_retro_v3.pt`
- `results/shared/enzyformer_retro_v5.pt`
- `results/shared/skeleton_inpainter/final.pt`
- `results/shared/dual_tower_v2.pt`
- `results/shared/skeleton_model.pt`
- `results/shared/cascadeboard_candidate_supervision_v1.json`
- `results/shared/enzexpand_v3_expanded_cache.json`

Controller-v2 pruning:

- retained only the current `train_v8/source_policy/` and `train_v8/open_leaf_policy/` artifacts used by Phase II runs
- removed train/val/test trace JSONL, feature cache, source-pack JSONL, ablation policies, early full100 matrices, smoke/debug outputs

Reservoir-distill pruning:

- retained root distillation manifests/reports/controllers, current packs, `full100_acceptance_real_v2/`, `cost_scoring_full100_current_20260514_w2_clean/`, and `external_publication_matrix_limit10_ua_20260514/`
- removed obsolete student-only tuning runs, append-only smokes, old limit10/current smokes, superseded `full100_acceptance_real/`, old external student-only runs, and bounded-reservoir smoke output

Duplicate trace pruning:

- removed `run_trace_shard*.jsonl` and `run_shard*.json` where merged `run_trace.jsonl`/`run.json` are retained
- removed matrix report copies matching `reports*/*/run_trace.jsonl`; original retained run directories keep the merged traces

Code/default-path cleanup:

- updated `cascade_planner/route_tree/runtime.py` default route-tree policy from removed legacy `results/shared/vnext/search_policy.pt` to the retained stock-aware policy
- updated `scripts/run_route_tree_gold_smoke.py` to use the retained stock-aware policy by default
- updated `cascade_planner/route_tree/README.md` so documented runtime defaults match the retained artifacts

Retained runtime-critical artifacts:

- `vendor/ChemEnzyRetroPlanner/`
- `data_external/`
- `workspace/aizdata/`
- `results/shared/zinc_inchikeys.txt`
- `results/shared/skeleton_inpainter/best.pt`
- `results/shared/enzyformer_retro_v4.pt`
- `results/shared/controller_v2_20260512/fullrun/train_v8/source_policy/cascade_source_policy.pt`
- `results/shared/controller_v2_20260512/fullrun/train_v8/open_leaf_policy/stock_aware_search_policy.pt`
- `results/shared/reservoir_distill_20260513/reservoir_distilled_controller.pt`
- `results/shared/chem_enzy_baseline/full100_reservoir_synthesized_rankplusstock_20260512.json`
- Phase II full100/external benchmark summaries and merged traces
- AUTOPLANNRELLM full100 diagnostic runs and reports

Space reclaimed in this final sweep:

- first deletion pass: `39,197,296 KB`
- duplicate shard/report trace pass: `2,771,352 KB`
- total additional reclaimed: about `40.0 GiB`

Post-sweep footprint:

- repository total: about `20G`
- `results/shared`: about `4.4G`
- required vendor runtime: about `7.6G`
- external data/runtime dependencies: about `2.6G`
- git object store: about `4.0G`
