# EnzRetro Transformer Integration

This repository keeps EnzRetro as an optional local module. The AutoPlanner code
does not require EnzRetro weights to be committed to GitHub.

## Files In AutoPlanner

- `cascade_planner/baselines/enzretro_onestep.py` exposes EnzRetro predictions as
  AutoPlanner one-step proposal rows.
- `cascade_planner/cascadeboard/live_retro.py` loads the provider when
  `AUTOPLANNER_ENABLE_ENZRETRO_PROPOSALS=1`.
- `.env.example` documents the required local paths and decoding options.

## Required External Files

Keep these outside the GitHub repository, or in ignored local paths:

- EnzRetro code package, for example `enzretro_model_code_package/`
- model checkpoint, for example `outputs/enzretro_ft_filtered_strict_singleprod_v1/best_model.pt`
- tokenizer vocabulary, for example `enzretro/tokenizer/vocab.txt`
- training datasets and evaluation outputs

## Recommended Local Settings

```bash
export AUTOPLANNER_ENABLE_ENZRETRO_PROPOSALS=1
export AUTOPLANNER_ENZRETRO_PACKAGE_ROOT="/absolute/path/to/enzretro_model_code_package"
export AUTOPLANNER_ENZRETRO_MODEL_DIR="/absolute/path/to/enzretro_model_code_package/outputs/enzretro_ft_filtered_strict_singleprod_v1"
export AUTOPLANNER_ENZRETRO_CHECKPOINT="best_model.pt"
export AUTOPLANNER_ENZRETRO_VOCAB_FILE="/absolute/path/to/enzretro_model_code_package/enzretro/tokenizer/vocab.txt"
export AUTOPLANNER_ENZRETRO_DEVICE="cuda"
export AUTOPLANNER_ENZRETRO_BEAM_SIZE=5
export AUTOPLANNER_ENZRETRO_RETURN_TOPK=5
export AUTOPLANNER_ENZRETRO_DEDUPE_SUBSTRATE=1
export AUTOPLANNER_ENZRETRO_LIPID_FILTER=1
export AUTOPLANNER_ENZRETRO_REQUIRE_EXECUTE=1
export AUTOPLANNER_ENZRETRO_CHEMISTRY_CONSTRAINTS=1
```

`AUTOPLANNER_ENZRETRO_CHEMISTRY_CONSTRAINTS` is optional and backwards
compatible. If the external EnzRetro package does not expose the chemistry-aware
rerank argument, AutoPlanner falls back to the older scoring function.

## Smoke Check

```bash
PYTHONPATH=. python - <<'PY'
from cascade_planner.baselines.enzretro_onestep import EnzRetroOneStepProposalProvider

provider = EnzRetroOneStepProposalProvider.from_env()
print({"available": provider.available, "load_error": provider.load_error})
if provider.available:
    rows = provider.predict("N[C@@H](CCCP(=O)(O)O)C(=O)O", top_k=3)
    print(rows[:1])
PY
```

## GitHub Policy

Commit the adapter code and this documentation. Do not commit checkpoints,
datasets, ChemEnzy vendor checkouts, run outputs, or local caches.
