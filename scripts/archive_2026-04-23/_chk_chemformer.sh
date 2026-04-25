#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate synth
pip list 2>/dev/null | grep -iE 'chemformer|pysmiles|pytorch_lightning|lightning'
echo ---
python -c "import chemformer; print(chemformer.__file__)" 2>&1 | tail -10
