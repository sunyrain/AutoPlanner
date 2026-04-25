#!/usr/bin/env bash
# Phase 1: train EnzExpand on v2 (mf=2 then mf=5), and AiZ eval on v2.
set -e
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/autodl-tmp/AutoPlanner/workspace
mkdir -p logs results

DATA=cascade_dataset_v2.normalized.json

# --- 1a) EnzExpand mf=2 (synth env) ---
conda activate synth
export PYTHONUNBUFFERED=1
echo "=== [enz mf=2] $(date) ==="
python -m cascade_planner.expand.enz_template \
  --data $DATA --min-freq 2 --folds 3 --epochs 40 --topk 50 \
  > logs/enz_mf2_v2.log 2>&1
cp results/enzexpand_step_eval.csv results/enzexpand_step_eval_mf2.csv
echo "[done mf=2]" $(date)
tail -8 logs/enz_mf2_v2.log

# --- 1b) EnzExpand mf=5 ---
echo "=== [enz mf=5] $(date) ==="
python -m cascade_planner.expand.enz_template \
  --data $DATA --min-freq 5 --folds 3 --epochs 40 --topk 50 \
  > logs/enz_mf5_v2.log 2>&1
cp results/enzexpand_step_eval.csv results/enzexpand_step_eval_mf5.csv
echo "[done mf=5]" $(date)
tail -8 logs/enz_mf5_v2.log
conda deactivate

# --- 1c) AiZ eval (autoplanner-aiz env) ---
conda activate autoplanner-aiz
echo "=== [aiz v2] $(date) ==="
python -m cascade_planner.eval.eval_aizynthfinder \
  --data $DATA --config aizdata/config.yml --max-steps 999999 \
  --out results/aizynthfinder_full_gpu_step_eval.csv \
  > logs/aiz_v2.log 2>&1
echo "[done aiz]" $(date)
tail -10 logs/aiz_v2.log
conda deactivate

echo "=== ALL PHASE1 DONE $(date) ==="
ls -la results/enzexpand_step_eval_mf*.csv results/aizynthfinder_full_gpu_step_eval.csv
