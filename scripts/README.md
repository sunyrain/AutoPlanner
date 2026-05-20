# Scripts

Active repository utilities only.

| Script | Purpose |
|---|---|
| `download_brenda.py` | Download or refresh BRENDA condition data inputs. |
| `monitor_autoplanner_web.py` | Poll the local WebUI service, CUDA status, queued jobs, and latest output/rejected artifacts. |
| `run_autoplanner_web_waitress.py` | Start the Waitress-backed local WebUI service used for collaborator testing. |
| `run_chem_enzy_plan_for_web.py` | Run one ChemEnzy native route search from a WebUI JSON request and emit Web-compatible JSON. |
| `reaudit_route_pool.py` | Refresh product/condition audit metadata for an exported route-pool JSON after audit-rule changes. |
| `render_linear_route_schemes.py` | Render top routes as forward, paper-style synthesis schemes with condition-audit markers. |
| `render_route_trees.py` | Render route topology as continuous Graphviz/RDKit trees. |
| `render_route_figures.py` | Render appendix-style per-step route figures. |
| `run_chem_enzy_smoke.py` | Run or dry-run the external ChemEnzyRetroPlanner core-search baseline and write normalized JSON. |
| `train_no_human_route_block_value_models.sh` | Train the main no-human route/block value-model ablations from weak labels in the strict value pack. |
| `run_route_block_review_expansion_real.sh` | Fallback/audit: run the 150-row route/block review expansion through the real non-dry-run reviewer pipeline after `DEEPSEEK_API_KEY` is set. |
| `run_strict_model_review_real.sh` | Fallback/audit: run the strict model-control disagreement review worklist through the same real reviewer pipeline. |
| `run_strict_model_review_real_extended.sh` | Fallback/audit: run the 300-row strict model-control disagreement review set. |
| `run_strict_review_full_after_key.sh` | Fallback/audit: one-command continuation after a key is configured: run strict review, refresh readiness, train if ready, optionally run the 300-row set. |
| `run_strict_review_from_filled_csv.sh` | Fallback/audit: one-command continuation after a human/external reviewer fills the strict packet CSV. |
| `train_strict_model_review_value_models.sh` | Fallback/audit: train expert-review route/block value-model ablations from a merged strict review value pack after the merge gate passes. |
| `run_route_tree_gold_smoke.py` | Run the current AutoPlanner `route_tree` baseline on the same v3 gold smoke targets used by ChemEnzy. |
| `setup_chem_enzy_runtime.sh` | Download/unpack the ChemEnzy runtime under `/root/autodl-tmp` without filling the root conda envs directory. |
| `setup_chem_enzy_vendor.sh` | Clone/update the ignored ChemEnzyRetroPlanner vendor checkout without running heavy model setup. |
| `setup_enzyformer.sh` | Prepare the external Enzyformer checkout/checkpoints expected by wrappers. |

Common Web commands:

```bash
PYTHONPATH=. AUTOPLANNER_WEB_HOST=0.0.0.0 AUTOPLANNER_WEB_PORT=7991 \
  CHEMENZY_ENV_PREFIX=/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env \
  python scripts/run_autoplanner_web_waitress.py

PYTHONPATH=. python scripts/monitor_autoplanner_web.py --url http://127.0.0.1:7991 --once
```

Main no-human route/block value training:

```bash
PYTHONPATH=. scripts/train_no_human_route_block_value_models.sh
```

This trains consensus-task and route-task no-human ablations, including the
strict `*_no_audit_no_retrieval` controls, then writes:

```text
results/shared/model_strengthening_20260519_no_human_route_block_value_models/
  no_human_route_block_value_ablation_summary.md
  no_human_route_block_value_ablation_summary.json
```

Fixed-pool final rerank replay:

```bash
PYTHONPATH=. python -m cascade_planner.eval.replay_route_block_value_model \
  --pack results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl \
  --model-pickle results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_no_audit_no_retrieval/route_block_value_model.pkl \
  --output-json results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_no_audit_no_retrieval_final_rerank_replay.json \
  --output-md results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_no_audit_no_retrieval_final_rerank_replay.md
```

Live final-rerank smoke, after the fixed-pool replay:

```bash
PYTHONPATH=. python -m cascade_planner.eval.run_cascade_search_benchmark \
  --benchmark data/benchmark_v2_100.json \
  --output results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_route_block_final_rerank_limit20.json \
  --vendor-root vendor/ChemEnzyRetroPlanner \
  --stock Zinc_Fix-stock \
  --iterations 2 \
  --chem-enzy-max-depth 4 \
  --expansion-topk 10 \
  --limit 20 \
  --cascade-expansion-budget 20 \
  --cascade-result-limit 3 \
  --include-route-outcomes \
  --route-block-value-final-reranker results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_no_audit_no_retrieval/route_block_value_model.pkl
```

No-label product-audit conservative final-rerank smoke:

```bash
PYTHONPATH=. python -m cascade_planner.eval.run_cascade_search_benchmark \
  --benchmark data/benchmark_v2_100.json \
  --output results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_product_audit_final_rerank_limit20.json \
  --vendor-root vendor/ChemEnzyRetroPlanner \
  --stock Zinc_Fix-stock \
  --iterations 2 \
  --chem-enzy-max-depth 4 \
  --expansion-topk 10 \
  --limit 20 \
  --cascade-expansion-budget 20 \
  --cascade-result-limit 3 \
  --include-route-outcomes \
  --product-audit-final-reranker
```

Runtime hard-negative no-human probe:

```bash
PYTHONPATH=. python -m cascade_planner.eval.probe_runtime_hardneg_nohuman_controls \
  --cache-dir results/shared/cascadebench_strict_20260516/ccts_v3_runtime_candidate_cache \
  --output-json results/shared/model_strengthening_20260519_transition_hardneg_nohuman_probe/runtime_hardneg_nohuman_probe.json \
  --output-md results/shared/model_strengthening_20260519_transition_hardneg_nohuman_probe/runtime_hardneg_nohuman_probe.md
```

The main path does not require expert labels, a filled expert CSV, or
`DEEPSEEK_API_KEY`.

Fallback/audit route/block review expansion:

```bash
# Either export the key, copy .env.local.example to .env.local and fill it,
# or add DEEPSEEK_API_KEY to .env.
export DEEPSEEK_API_KEY=...
PYTHONPATH=. scripts/run_route_block_review_expansion_real.sh

# Optional: override the review input set without editing the script.
REVIEW_JSONL=results/shared/.../custom_review.jsonl \
TRANSFORM_SANITY_JSON=results/shared/.../custom_transform_sanity.json \
PYTHONPATH=. scripts/run_route_block_review_expansion_real.sh

# Current strict model-control disagreement worklist:
PYTHONPATH=. scripts/run_strict_model_review_real.sh

# Expanded 300-row fallback worklist:
PYTHONPATH=. scripts/run_strict_model_review_real_extended.sh

# After real review succeeds and the merged-pack gate is ready:
PYTHONPATH=. scripts/train_strict_model_review_value_models.sh

# One-command continuation after configuring the key:
PYTHONPATH=. scripts/run_strict_review_full_after_key.sh

# Allow automatic 300-row fallback if the 120-row review is insufficient:
RUN_EXTENDED_IF_NOT_READY=1 PYTHONPATH=. scripts/run_strict_review_full_after_key.sh

# Human/external CSV continuation after the 120-row packet is filled:
PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh

# Human/external CSV continuation for the 300-row fallback packet:
PACKET_SIZE=300 PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
```

This writes non-dry-run review labels, promotion-gate artifacts, and when
`VALUE_PACK` is set a merged `*_merged_route_block_value_pack.jsonl` containing
`expert_review_positive` / `expert_review_negative` tasks.

The real-review wrapper defaults to `WORKERS=4`, writes a resumable cache under
the output directory, and sets `AUTOPLANNRELLM_DEEPSEEK_TIMEOUT_S=300` unless
overridden.

Strict review continuation scripts exit non-zero when the merged review pack is
still not ready for expert training. Set `ALLOW_NOT_READY_EXIT_ZERO=1` only for
inspection-only runs where an incomplete gate should not fail the shell command.

Historical migration, cluster smoke, and packaging helpers are archived under
`archive/cleanup_2026-05-05/scripts/` when present. Generated release bundles
belong under ignored local artifact folders such as `releases/`.
