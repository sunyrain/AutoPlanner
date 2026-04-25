#!/usr/bin/env bash
# Backup v1 results then re-run all evals on v2.
set -e
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/autodl-tmp/AutoPlanner/workspace
mkdir -p logs results_v1

# 1) backup
echo "=== backing up v1 results ==="
ls results/ | head -20
cp -n results/*.csv results/*.json results/*.md results_v1/ 2>/dev/null || true
ls results_v1/ | wc -l

# 2) verify aiz env
echo "=== aiz env ==="
conda activate autoplanner-aiz
python -c 'import aizynthfinder, rdchiral, rdkit; print("aiz import OK", "rdkit", rdkit.__version__)'
conda deactivate
echo OK
