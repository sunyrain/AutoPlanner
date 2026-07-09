# GitHub Upload Checklist

Use this checklist before publishing this working tree.

## Include

- `README.md`
- `requirements.txt`
- `.gitignore`
- `.env.example`
- `cascade_planner/`
- `scripts/`
- `tests/`
- `docs/`
- small curated files under `data/` that are required for examples or tests

## Exclude

The current `.gitignore` is configured to exclude the main local-heavy paths:

- `vendor/ChemEnzyRetroPlanner/`
- `results/`
- `results/shared/`
- `workspace/`
- `releases/`
- `archive/results/`
- `archive/datasets/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.npy`, `*.npz`
- `.env`, `.env.*`
- `__pycache__/`, `*.pyc`

## External Model Assets

Do not upload EnzRetro checkpoints or datasets into this repository. Keep them
in a separate storage location and configure paths through environment
variables. See `docs/ENZRETRO_INTEGRATION.md`.

## Preflight Commands

```bash
find . -type f \( -name '*.pyc' -o -name '.DS_Store' \) -print
find . -type f -size +50M -print
PYTHONPATH=. pytest --collect-only -q
python -m py_compile cascade_planner/baselines/enzretro_onestep.py
```

Large files reported by `find` should usually be external assets, not committed
source files.
