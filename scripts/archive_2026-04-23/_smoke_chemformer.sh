#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate synth
python -c "import chemformer; print('OK', chemformer.__file__)" 2>&1
echo ---
python -c "from syntheseus.reaction_prediction.inference import ChemformerModel; m = ChemformerModel(); print('LOADED model_dir=', m.model_dir)" 2>&1 | tail -30
