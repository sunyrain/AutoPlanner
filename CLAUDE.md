# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AutoPlanner is a chemo-enzymatic retrosynthesis planner. It combines enzymatic and chemical single-step models, multi-step tree search, reaction condition prediction, and a learned route optimization system called CascadeBoard++.

Python 3.11. No setup.py/pyproject.toml — modules are run directly via `python -m`.

## Common Commands

```bash
# Data normalization (one-time)
python -m cascade_planner.data.normalize_v2 --in cascade_dataset_v2.json --out cascade_dataset_v2.normalized.json --report cascade_dataset_v2.quality.json
python -m cascade_planner.data.strict_filter_v2

# Single-step evaluation (honest, K-budget-aware)
python -m cascade_planner.eval.hybrid_multi_audited

# Condition prediction evaluation
python -m cascade_planner.eval.condition_diagnosis

# EnzExpand reranker training
python -m cascade_planner.expand.reranker --data cascade_dataset_v2.normalized.json --tag v2_mf2

# Dual-tower contrastive model (requires GPU + ESM embeddings)
python -m cascade_planner.expand.dual_tower --data cascade_dataset_v2.normalized.json --esm-cache results/shared/esm_cache/ --tag v1

# OA-ARM Skeleton Inpainter (new, recommended)
python -m cascade_planner.cascadeboard.skeleton_inpainter train --data cascade_dataset_v3.json --epochs 200
python -m cascade_planner.cascadeboard.skeleton_inpainter predict --target "SMILES" --n-steps 3 --k 5

# Learned Route Scorer
python -m cascade_planner.cascadeboard.learned_scorer --data cascade_dataset_v3.json --epochs 150

# Integrated benchmark (OA-ARM + Scorer, 100 targets)
python -m cascade_planner.cascadeboard.integrated_benchmark

# CascadeBoard++ legacy training
python -m cascade_planner.cascadeboard.train

# CascadeBoard++ planning (CLI)
python -m cascade_planner.cascadeboard.cli --target "SMILES" --n-steps 3
python -m cascade_planner.cascadeboard.cli --target "SMILES" --constraints '{"one_pot": true}' --objective green
python -m cascade_planner.cascadeboard.cli --target "SMILES" --live --n-results 3

# Multi-step benchmark (frozen 100-target)
python -m cascade_planner.eval.run_benchmark_v2_100 --max-iter 100 --max-depth 6
```

## Architecture

### Package layout (`cascade_planner/`)

Six subpackages form a pipeline from raw data to planned routes:

**`data/`** — Dataset loading and normalization. `loader_v2.py` is the canonical loader producing `StepRow` and `StepPairRow` dataclasses from `cascade_dataset_v2.normalized.json`. `strict_filter_v2.py` produces the strict subset. `open_datasets.py` integrates EnzymeMap, ReactZyme, and USPTO-50K.

**`expand/`** — Single-step retrosynthesis models. `enz_template.py` is the template-MLP EnzExpand model (Morgan2-2048 → template-id). `dual_tower.py` is the contrastive DRFP↔ESM-2 enzyme matcher. `reranker.py` / `reranker_v2.py` are LightGBM LambdaRank rerankers. `esm_embedder.py` generates ESM-2 enzyme embeddings. `retrochimera_policy.py` wraps the external RetroChimera chemical model.

**`conditions/`** — Reaction condition prediction. `predict_conditions.py` trains T/pH/solvent/catalyst classifiers (DRFP features, GroupKFold by DOI). `brenda_predictor.py` uses BRENDA enzyme data. `esm_condition_heads.py` adds ESM-2-based condition heads.

**`multistep/`** — Tree search. `aiz_mcts_bridge.py` calls AiZynthFinder MCTS in a separate `.venv_aizynth` virtualenv via subprocess. `plan_route.py` is the end-to-end driver: MCTS → extract steps → predict conditions → output. `two_stage_search.py` combines chemical MCTS with enzymatic expansion.

**`training/`** — `featurize_v2.py` provides DRFP fingerprinting (`drfp_batch`, `drfp_one`). `run_baselines_v2.py` runs baseline models.

### CascadeBoard++ (`cascadeboard/`)

The core learned planning system. Current architecture is a **three-layer independent model pipeline**:

**Layer 1 — Skeleton Generation** (`skeleton_inpainter.py`):
- OA-ARM Transformer (6-layer decoder, d=256, 8 heads, 6.5M params)
- Trained with random permutation order on v3 3,810 routes + augmentation → 50K samples
- Generates K diverse skeletons (reaction types, EC classes, T/pH) conditioned on target FP + domain + constraints
- Val: rtype_acc=92.4%, ec1_acc=94.8%, T MAE=0.74°C

**Layer 2 — Molecular Fill** (`skeleton_planner.py`, `live_retro.py`):
- RetroChimera (chemical) + EnzExpand/Enzyformer (enzymatic) fill concrete reactions into skeleton slots
- Greedy fill with diagnosis-driven refinement

**Layer 3 — Route Scoring** (`learned_scorer.py`):
- 4-layer Transformer encoder (d=128, 4 heads, 0.9M params)
- Multi-task: route_score + compat + opmode + issues + yield/ee
- Ranks filled routes for final selection

**Legacy system** (still functional, used by `cli.py` cache mode):
- `route_encoder.py` — `CascadeBoardTransformer` (v20, 3.92M params): edit policy + inpainting + real-label heads
- `planner.py` — Particle-based refinement with candidate graph
- `energy_api.py` — Rule-based energy scoring

**Supporting modules:**
- `constraint_compiler.py` — Compiles user constraints into masks/factors
- `candidate_graph.py` — AND-OR candidate hypergraph from frozen retro experts
- `train.py` / `training_data.py` — Training loop and data construction
- `cli.py` — End-user CLI interface
- `real_benchmark.py` / `integrated_benchmark.py` — Benchmarking

### Key conventions

- Results go under `results/v2/` (current) or `results/shared/` (cross-version). Controlled by `CASCADE_VERSION` env var and `paths.py`.
- Every single-step metric must report `(model, baseline, lift = model / baseline)`. Reference evaluator: `eval/hybrid_multi_audited.py`.
- Cross-validation uses `GroupKFold` by DOI to prevent data leakage.
- Large weights/models are git-ignored and live outside the repo: `workspace/aizdata/`, `synth_weights/`, `data_external/`.
- AiZynthFinder runs in a separate virtualenv (`.venv_aizynth/`) called via subprocess from `aiz_mcts_bridge.py`.
- The canonical dataset is `cascade_dataset_v2.normalized.json` (8748 steps, 3028 trainable). The v3 dataset (`cascade_dataset_v3.json`, 3810 cascades, 8753 steps) adds compatibility/operation-mode annotations used by skeleton inpainter and scorer training.

### Current performance (100-target benchmark, audited)

Method E: 2-step requires 100% type match, 3-step ≥67%, 4+ step ≥50%

| System | Plan rate | GT@5 | Random baseline | Lift | Fill quality | Avg time |
|--------|-----------|------|-----------------|------|--------------|----------|
| **OA-ARM + Enzyformer v4** | 99% | 75% | 19% | 4.0x | 99% | 0.81s |
| CascadeBoard v20 (legacy) | 100% | 24% | - | - | - | ~10s |
| MCTS-USPTO | 66% | 20% | - | - | - | ~30s |

Statin side-chain cascade: 3/4 variants pass (skeleton 100% match + valid fill + stock-available endpoint)

### Key data structures

- `StepRow` / `StepPairRow` (`data/loader_v2.py`) — Flat row per reaction step / consecutive pair, used by all evaluators.
- `Slot` / `CascadeBoard` (`cascadeboard/__init__.py`) — Board representation: linear chain of slots with molecules, enzyme/catalyst, conditions, candidates, energy terms, and fixed-field constraints.
- `SlotFeatures` / `SkeletonSample` (`cascadeboard/skeleton_inpainter.py`) — Feature vectors for OA-ARM training and inference.
- `SkeletonResult` / `ScoreResult` — Output dataclasses from skeleton generation and route scoring.

## Authoritative docs

- `CASCADEBOARD_TODO.md` — Task tracking and experiment history
- `DATA_TEAM_REQUIREMENTS.md` — Data annotation requirements
