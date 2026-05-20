#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_ROOT="${CHEMENZY_VENDOR_ROOT:-$ROOT_DIR/vendor/ChemEnzyRetroPlanner}"
RUNTIME_ROOT="${CHEMENZY_RUNTIME_ROOT:-/root/autodl-tmp/chem_enzy_runtime}"
SCOPE="${CHEMENZY_SETUP_SCOPE:-core}"

usage() {
  cat <<'USAGE'
Usage: scripts/setup_chem_enzy_runtime.sh [--scope core|full]

Creates a relocatable ChemEnzy runtime under CHEMENZY_RUNTIME_ROOT
(default: /root/autodl-tmp/chem_enzy_runtime) without writing the packed env
into /root/miniconda3/envs.

Scopes:
  core  Download env + stock + graph + ONMT + value metadata.
  full  Also download condition/enzyme/rxn_filter/easifa/Parrot metadata.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      SCOPE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$SCOPE" != "core" && "$SCOPE" != "full" ]]; then
  echo "--scope must be core or full, got: $SCOPE" >&2
  exit 2
fi

if [[ ! -d "$VENDOR_ROOT" ]]; then
  echo "Vendor checkout missing: $VENDOR_ROOT" >&2
  echo "Run scripts/setup_chem_enzy_vendor.sh first." >&2
  exit 2
fi

if [[ -f /etc/network_turbo ]]; then
  # Academic acceleration for GitHub/HuggingFace access when available.
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

DOWNLOAD_DIR="$RUNTIME_ROOT/downloads"
ENV_PREFIX="$RUNTIME_ROOT/envs/retro_planner_env"
PIP_CACHE_DIR="$RUNTIME_ROOT/cache/pip"
HF_HOME="$RUNTIME_ROOT/cache/huggingface"
TORCH_HOME="$RUNTIME_ROOT/cache/torch"

mkdir -p "$DOWNLOAD_DIR" "$ENV_PREFIX" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

download_hf() {
  local filename="$1"
  local url="https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/${filename}?download=true"
  local output="$DOWNLOAD_DIR/$filename"
  if [[ -s "$output" ]]; then
    echo "[download] exists $filename"
    return
  fi
  echo "[download] $filename"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$output" "$url"
}

core_files=(
  retro_planner_env.tar.gz
  building_block_dataset.zip
  graph_retrosyn_metadata.zip
  onmt_metadata.zip
  value_fun_metadata.zip
)

full_extra_files=(
  condition_predictor_metadata.zip
  easifa_metadata.zip
  enzyme_cls_metadata.zip
  rxn_filter_metadata.zip
  USPTO_condition.mar
)

for file in "${core_files[@]}"; do
  download_hf "$file"
done
if [[ "$SCOPE" == "full" ]]; then
  for file in "${full_extra_files[@]}"; do
    download_hf "$file"
  done
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "[env] extracting retro_planner_env.tar.gz to $ENV_PREFIX"
  tar -xzf "$DOWNLOAD_DIR/retro_planner_env.tar.gz" -C "$ENV_PREFIX"
else
  echo "[env] exists $ENV_PREFIX"
fi

if [[ -x "$ENV_PREFIX/bin/conda-unpack" ]]; then
  echo "[env] conda-unpack"
  "$ENV_PREFIX/bin/conda-unpack"
else
  echo "[env] conda-unpack not found; continuing"
fi

unzip_if_present() {
  local filename="$1"
  if [[ -f "$DOWNLOAD_DIR/$filename" ]]; then
    echo "[metadata] unzip $filename"
    (cd "$VENDOR_ROOT" && unzip -oq "$DOWNLOAD_DIR/$filename")
  fi
}

unzip_if_present building_block_dataset.zip
unzip_if_present graph_retrosyn_metadata.zip
unzip_if_present onmt_metadata.zip
unzip_if_present value_fun_metadata.zip
if [[ "$SCOPE" == "full" ]]; then
  unzip_if_present condition_predictor_metadata.zip
  unzip_if_present easifa_metadata.zip
  unzip_if_present enzyme_cls_metadata.zip
  unzip_if_present rxn_filter_metadata.zip
  if [[ -f "$DOWNLOAD_DIR/USPTO_condition.mar" ]]; then
    mkdir -p "$VENDOR_ROOT/retro_planner/packages/parrot/mars"
    cp -f "$DOWNLOAD_DIR/USPTO_condition.mar" "$VENDOR_ROOT/retro_planner/packages/parrot/mars/USPTO_condition.mar"
  fi
fi

echo "[pip] editable installs"
(
  cd "$VENDOR_ROOT"
  export PATH="$ENV_PREFIX/bin:$PATH"
  export PIP_CACHE_DIR HF_HOME TORCH_HOME
  "$ENV_PREFIX/bin/python" -m pip install -e .
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/mlp_retrosyn/
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/value_function/
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/rxn_filter/
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/onmt/
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/easifa/
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/graph_retrosyn/
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/condition_predictor/
  "$ENV_PREFIX/bin/python" -m pip install -e ./retro_planner/packages/organic_enzyme_rxn_classifier/
)

cat <<MSG
[done] ChemEnzy runtime prepared
  scope:        $SCOPE
  runtime root: $RUNTIME_ROOT
  env prefix:   $ENV_PREFIX
  vendor root:  $VENDOR_ROOT

Use:
  PYTHONPATH=. $ENV_PREFIX/bin/python scripts/run_chem_enzy_smoke.py --limit 1
MSG
