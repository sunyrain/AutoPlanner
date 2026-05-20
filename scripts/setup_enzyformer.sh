#!/usr/bin/env bash
# setup_enzyformer.sh — Clone Enzyformer repo and download pretrained checkpoint.
#
# Enzyformer (Tiantao et al.) is a two-stage pretrained Chemformer-based model
# for enzymatic retrosynthesis using EC-conditioned R-SMILES.
#
# Paper: https://link.springer.com/article/10.1186/s13321-026-01164-y
# Repo:  https://github.com/Tiantao2000/Enzyformer
#
# COMPATIBILITY NOTES:
#   - Enzyformer was developed with Python 3.7, PyTorch 1.9, pytorch-lightning 1.x
#   - Our project uses Python 3.11, PyTorch 2.3, pytorch-lightning 2.6
#   - The wrapper (cascade_planner/expand/enzyformer_wrapper.py) handles compat shims
#   - Key issues: PL 2.x removed LightningModule.load_from_checkpoint positional args,
#     tokenizer API changes, and some deprecated torch ops.
#
# Usage:
#   bash scripts/setup_enzyformer.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/data_external/enzyformer"
CKPT_DIR="${DEST_DIR}/checkpoints"

echo "=== Step 1: Clone Enzyformer repository ==="
if [ -d "${DEST_DIR}/.git" ]; then
    echo "  Already cloned at ${DEST_DIR}, pulling latest..."
    cd "${DEST_DIR}" && git pull
else
    echo "  Cloning to ${DEST_DIR}..."
    git clone https://github.com/Tiantao2000/Enzyformer.git "${DEST_DIR}"
fi

echo ""
echo "=== Step 2: Download pretrained checkpoint ==="
mkdir -p "${CKPT_DIR}"

# The pretrained checkpoint is hosted on Google Drive.
# Link from the Enzyformer README (may require manual download if gdown fails):
#   https://drive.google.com/drive/folders/1VHl3qSFsBNJBgnSbMFkXJPX1sOBmRzCi
#
# We use gdown to download. Install if needed:
pip install gdown -q 2>/dev/null || true

GDRIVE_FOLDER_ID="1VHl3qSFsBNJBgnSbMFkXJPX1sOBmRzCi"

if [ -f "${CKPT_DIR}/enzyformer_retro.ckpt" ]; then
    echo "  Checkpoint already exists at ${CKPT_DIR}/enzyformer_retro.ckpt"
else
    echo "  Downloading from Google Drive folder ${GDRIVE_FOLDER_ID}..."
    echo "  (If this fails, manually download from the Google Drive link above)"
    gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" \
        -O "${CKPT_DIR}" 2>/dev/null || {
        echo ""
        echo "  [WARNING] gdown failed. Please manually download the checkpoint:"
        echo "    1. Visit: https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"
        echo "    2. Download the retrosynthesis .ckpt file"
        echo "    3. Place it at: ${CKPT_DIR}/enzyformer_retro.ckpt"
        echo ""
    }
fi

echo ""
echo "=== Step 3: Install Python dependencies ==="
echo "  Enzyformer requires the Chemformer codebase (MolecularAI/Chemformer)."
echo "  We install it as a local package from the cloned repo's vendored copy,"
echo "  or from the upstream repo if not bundled."

# Check if Chemformer is bundled inside Enzyformer
if [ -d "${DEST_DIR}/Chemformer" ]; then
    echo "  Found bundled Chemformer in Enzyformer repo."
    pip install -e "${DEST_DIR}/Chemformer" -q 2>/dev/null || {
        echo "  [INFO] Chemformer pip install failed — wrapper will use standalone inference."
    }
elif [ -d "${DEST_DIR}/chemformer" ]; then
    echo "  Found bundled chemformer in Enzyformer repo."
    pip install -e "${DEST_DIR}/chemformer" -q 2>/dev/null || {
        echo "  [INFO] chemformer pip install failed — wrapper will use standalone inference."
    }
else
    echo "  No bundled Chemformer found. Cloning MolecularAI/Chemformer..."
    CHEMFORMER_DIR="${REPO_ROOT}/data_external/chemformer"
    if [ ! -d "${CHEMFORMER_DIR}/.git" ]; then
        git clone https://github.com/MolecularAI/Chemformer.git "${CHEMFORMER_DIR}"
    fi
    pip install -e "${CHEMFORMER_DIR}" -q 2>/dev/null || {
        echo "  [INFO] Chemformer pip install failed — wrapper will use standalone inference."
    }
fi

# Additional deps that Chemformer/Enzyformer typically need
pip install pysmilesutils -q 2>/dev/null || true
pip install sentencepiece -q 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo ""
echo "Directory structure:"
echo "  ${DEST_DIR}/          — Enzyformer source code"
echo "  ${CKPT_DIR}/          — Pretrained checkpoints"
echo ""
echo "To verify, run:"
echo "  python -c \"from cascade_planner.expand.enzyformer_wrapper import EnzyformerWrapper; print('OK')\""
echo ""
echo "IMPORTANT: If the checkpoint download failed, see the manual instructions above."
