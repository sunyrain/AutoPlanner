#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
IMAGE_GEN="$CODEX_HOME/skills/.system/imagegen/scripts/image_gen.py"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$IMAGE_GEN" generate \
  --model gpt-image-2 \
  --prompt-file "$ROOT_DIR/image2_prompt_architecture.txt" \
  --size 2048x1152 \
  --quality high \
  --out "$ROOT_DIR/figures/figure1_image2_background.png"
