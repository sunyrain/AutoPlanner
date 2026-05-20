#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${CHEMENZY_ENV_PREFIX:-/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "ChemEnzy Python not found: $ENV_PREFIX/bin/python" >&2
  echo "Run scripts/setup_chem_enzy_runtime.sh --scope core first." >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$ENV_PREFIX/bin/python" -m cascade_planner.eval.run_cascade_search_benchmark "$@"
