#!/usr/bin/env bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate synth
cd /root/autodl-tmp/AutoPlanner/workspace
python -c 'import rdkit, torch; print("rdkit", rdkit.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())'
python -c 'import rxnmapper; print("rxnmapper OK")' 2>&1 || echo "rxnmapper MISSING"
python -c 'from rdchiral.template_extractor import extract_from_reaction; print("rdchiral OK")' 2>&1 || echo "rdchiral MISSING"
python -c 'from cascade_planner.expand.enz_template import main' 2>&1 | head -3
ls cascade_planner/data/ | head -20
ls -la cascade_dataset_v2*.json
