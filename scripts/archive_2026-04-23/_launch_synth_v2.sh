#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate synth
MODEL=$1; GPU=$2
mkdir -p /tmp/run_$MODEL; cd /tmp/run_$MODEL
export PYTHONPATH=/root/autodl-tmp/AutoPlanner/workspace
export CUDA_VISIBLE_DEVICES=$GPU
nohup python -u -m cascade_planner.eval.syntheseus_eval --models $MODEL --limit 0 --num-results 50 > run.log 2>&1 &
echo "PID=$!  MODEL=$MODEL  GPU=$GPU  CWD=/tmp/run_$MODEL"
