#!/usr/bin/env bash
set -e
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate synth
pip install --no-cache-dir --timeout 120 rxnmapper rdchiral drfp 2>&1 | tail -10
python -c 'import rxnmapper, rdchiral, drfp; print("rxnmapper/rdchiral/drfp OK")'
cd /root/autodl-tmp/AutoPlanner/workspace
python -c 'from cascade_planner.data.loader_v2 import load_v2; s,p,c = load_v2("cascade_dataset_v2.normalized.json"); print("steps", len(s), "with_rxn", sum(1 for x in s if x.rxn_smiles), "with_ec", sum(1 for x in s if x.ec_number), "enz_steps", sum(1 for x in s if x.ec_number))'
