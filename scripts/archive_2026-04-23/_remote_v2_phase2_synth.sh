#!/usr/bin/env bash
# Phase 2: syntheseus single-step eval on v2 step set (megan, rootaligned).
# Run on a specific GPU id.
set -e
MODEL=$1
GPU=${2:-0}
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate synth
cd /root/autodl-tmp/AutoPlanner/workspace
mkdir -p logs results
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p /tmp/run_${MODEL}_v2
cd /tmp/run_${MODEL}_v2
PYTHONPATH=/root/autodl-tmp/AutoPlanner/workspace \
  python -m cascade_planner.eval.syntheseus_eval \
    --data cascade_dataset_v2.normalized.json \
    --models $MODEL --limit 0 --num-results 50 \
    > /root/autodl-tmp/AutoPlanner/workspace/logs/syn_${MODEL}_v2.log 2>&1
