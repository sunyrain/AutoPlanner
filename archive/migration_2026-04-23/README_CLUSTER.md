# Cluster setup quickstart

## 1. Unpack
```bash
mkdir -p ~/AutoPlanner && cd ~/AutoPlanner
tar -xzf /path/to/autoplanner_migration_2026-04-23.tar.gz
sha256sum -c autoplanner_migration_2026-04-23.tar.gz.sha256
```

## 2. Two parallel envs

### Env A — main training env (`autoplanner-main`)
```bash
conda create -n autoplanner-main python=3.11 -y
conda activate autoplanner-main
pip install -r requirements.txt
# verify
python -c "import torch, transformers, rdkit, drfp, rxnmapper; print('ok'); print(torch.cuda.is_available())"
```

### Env B — AiZynth multi-step env (`autoplanner-aiz`)
```bash
conda create -n autoplanner-aiz python=3.11 -y
conda activate autoplanner-aiz
pip install -r requirements_aizynth.txt
# verify
python -c "import aizynthfinder, rdchiral, onnxruntime; print('ok')"
```

## 3. Re-fetch large external files (if not in tarball)

```bash
# AiZynth public stock + models (only if zinc_stock.hdf5 was excluded)
conda activate autoplanner-aiz
download_public_data ./aizdata    # writes uspto_*, ringbreaker_*, zinc_stock.hdf5

# ESM-3 small open weights for M5 (CLEAN-2 / ReactZyme target)
huggingface-cli download EvolutionaryScale/esm3-sm-open-v1 --local-dir ./checkpoints/esm3-sm

# DESP for M3 (multi-step search)
git clone https://github.com/coleygroup/desp third_party/desp

# Syntheseus for benchmarking
pip install syntheseus
```

## 4. Smoke tests

```bash
conda activate autoplanner-main
# (a) data loader
python -c "from cascade_planner.data.loader_v2 import load_v2; s,_,_ = load_v2('cascade_dataset.normalized.uniprot.json'); print('steps:', len(s))"

# (b) condition predictor
python -m cascade_planner.conditions.predict_conditions \
    --data cascade_dataset.normalized.uniprot.json --tag cluster_smoke

# (c) end-to-end demo (6 cases)
python -m cascade_planner.demo.demo_pipeline
```

```bash
conda activate autoplanner-aiz
# (d) multi-step on ibuprofen
python -m cascade_planner.multistep.plan_route \
    --product "CC(C)Cc1ccc(C(C)C(=O)O)cc1" \
    --data ../autoplanner-main-mount/cascade_dataset.normalized.uniprot.json \
    --config aizdata/config_hybrid.yml \
    --policies uspto enzexpand --policy-weights 0.3 0.7 \
    --max-iter 200 --max-depth 5 --n-routes 5
```

If all four checks pass, migration is verified — proceed to PROPOSAL.md Phase 1 implementation.

## 5. Project layout on cluster

```
~/AutoPlanner/
├── cascade_planner/                    # source
├── cascade_dataset*.json               # canonical data
├── data/uniprot_cache.json             # speed-up cache
├── aizdata/*                           # ONNX models + configs
├── results/*                           # eval + cache + demos
├── data_external/                      # EnzymeMap etc.
├── checkpoints/                        # NEW — ESM-3, Chemformer, etc.
├── third_party/                        # NEW — desp, syntheseus clones
├── scripts/                            # tarball builder etc.
└── PROPOSAL.md                         # SOTA upgrade plan
```
