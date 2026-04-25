# AutoPlanner

Chemo-enzymatic retrosynthesis planner — single-step models, multi-step search, condition prediction.

## Layout

```
AutoPlanner/
├── cascade_planner/              active package (39 modules)
│   ├── data/                     v2 loader, normalizer, strict filter, uniprot enrichment
│   ├── expand/                   EnzExpand template model + rxnmapper pipeline
│   ├── conditions/               T/pH/solvent/catalyst predictors + enzyme recommender
│   ├── multistep/                AiZynthFinder MCTS bridge
│   ├── eval/                     honest, K-budget-aware evaluators
│   ├── demo/                     end-to-end demo pipeline
│   ├── training/                 featurizer + baselines for v2
│   └── paths.py                  results-dir helper (v1/v2/shared)
├── cascade_dataset_v2.json               canonical raw (schema 2.0.0, 2491 records)
├── cascade_dataset_v2.normalized.json    loader_v2 input (8748 steps, 3028 trainable)
├── cascade_dataset_v2.strict.json        audit-clean subset (2300 steps, 26% retention)
├── data/
│   ├── benchmark_v2_100.{json,csv}       frozen 100-target multi-step benchmark
│   └── uniprot_cache.json                REST cache
├── results/
│   ├── v1/ v2/ shared/           version-separated eval artefacts
│   └── README.md                 results layout
├── archive/
│   ├── code/                     superseded code (v1 evaluators, v1 schema normalizer)
│   ├── datasets/                 retired v1 snapshots
│   ├── docs/                     early planning docs
│   ├── logs/                     dated run-log dumps
│   ├── migration_2026-04-23/     GPU-cluster migration bundle (zip + manifest)
│   └── results/                  retired v1 result CSVs
├── scripts/
│   ├── archive_2026-04-23/       retired scratch scripts
│   └── build_*.ps1               release packagers
├── PROPOSAL.md                   KPI targets (K1-K7)
└── STATUS_REPORT.md              honest progress vs KPIs (with baselines + lift)
```

## Quickstart

```powershell
# 1. Data (one-time)
python -m cascade_planner.data.normalize_v2 --in cascade_dataset_v2.json --out cascade_dataset_v2.normalized.json --report cascade_dataset_v2.quality.json
python -m cascade_planner.data.strict_filter_v2   # produces cascade_dataset_v2.strict.json

# 2. Open dataset integration (EnzymeMap 33K + ReactZyme + USPTO-50K)
python -m cascade_planner.data.open_datasets --verify

# 3. Honest single-step audit (K-budget + random-in-pool baseline + lift)
python -m cascade_planner.eval.hybrid_multi_audited

# 4. Condition predictor honest diagnosis
python -m cascade_planner.eval.condition_diagnosis

# 5. BRENDA-informed condition prediction (K5/K6)
python -m cascade_planner.conditions.brenda_predictor --data workspace/cascade_dataset_v2.normalized.json --eval

# 6. Freeze 100-target multi-step benchmark
python -m cascade_planner.eval.freeze_benchmark

# 7. EnzExpand reranker (LightGBM LambdaRank)
#    boosts top-1 42.6% -> 49.5% on v2 full.
python -m cascade_planner.expand.reranker --data workspace/cascade_dataset_v2.normalized.json --tag v2_mf2
python -m cascade_planner.expand.reranker_freeze --candidates results/v2/reranker/candidates_v2_mf2.csv

# 8. ESM-2 enzyme embeddings (requires GPU)
python -m cascade_planner.expand.esm_embedder --input data_external/enzyme_sequences/autoplanner_enzymes.tsv --output results/shared/esm_cache/embeddings.npz

# 9. Dual-tower contrastive model (K1/K7)
python -m cascade_planner.expand.dual_tower --data workspace/cascade_dataset_v2.normalized.json --esm-cache results/shared/esm_cache/ --tag v1

# 10. ESM-2 condition heads (K5/K6, requires ESM embeddings)
python -m cascade_planner.conditions.esm_condition_heads --data workspace/cascade_dataset_v2.normalized.json --esm-cache results/shared/esm_cache/ --eval

# 11. Route scoring (cascade compatibility)
python -m cascade_planner.scoring.route_scorer --data workspace/cascade_dataset_v2.normalized.json --eval

# 12. Multi-step solve-rate + GT@K on frozen 100-target benchmark
python -m cascade_planner.eval.run_benchmark_v2_100 --max-iter 100 --max-depth 6

# 13. DESP bidirectional search (requires DESP models, see desp_bridge.py)
python -m cascade_planner.multistep.desp_bridge --targets-file data/benchmark_v2_100.json --output results/v2/desp_benchmark.json
```

Outputs land under `results/v2/` by default (set `CASCADE_VERSION=v1` to target `results/v1/`).

## Reporting discipline (mandatory)

Every single-step metric must carry `(model, baseline, lift = model / baseline)`. The reference evaluator is [cascade_planner/eval/hybrid_multi_audited.py](cascade_planner/eval/hybrid_multi_audited.py) — copy its pattern. Details: [STATUS_REPORT.md](STATUS_REPORT.md).

## Environments

Large weights/envs live outside this repo and are git-ignored:
- `aizdata/` (~767 MB) — AiZynthFinder models + stock
- `synth_weights/` (~1.1 GB) — syntheseus checkpoints
- `.venv_aizynth/`, `ChemEnzyRetroPlanner/` — local envs

For a clean GPU-cluster re-deploy see `archive/migration_2026-04-23/MIGRATION_MANIFEST.md` and `README_CLUSTER.md`.
