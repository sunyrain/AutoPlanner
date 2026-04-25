#!/usr/bin/env bash
# Atom-map all enzymatic rxns in v2; takes minutes on GPU.
set -e
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate synth
cd /root/autodl-tmp/AutoPlanner/workspace
mkdir -p logs results
export PYTHONUNBUFFERED=1
echo "=== map_reactions on v2 ==="
python -m cascade_planner.expand.map_reactions \
    --data cascade_dataset_v2.normalized.json --batch 32 2>&1 | tee logs/map_v2.log | tail -20
echo "=== cache size ==="
ls -la results/enzexpand_atommap_cache.json
