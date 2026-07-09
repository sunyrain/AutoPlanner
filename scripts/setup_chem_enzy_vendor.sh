#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${CHEMENZY_RETROPLANNER_ROOT:-$ROOT_DIR/vendor/ChemEnzyRetroPlanner}"
REPO_URL="${CHEMENZY_RETROPLANNER_REPO:-https://github.com/wangxr0526/ChemEnzyRetroPlanner.git}"

if [[ -f /etc/network_turbo ]]; then
  # Academic acceleration for GitHub/HuggingFace access when available.
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

if [[ ! -d "$VENDOR_DIR/.git" ]]; then
  mkdir -p "$(dirname "$VENDOR_DIR")"
  git clone --depth 1 "$REPO_URL" "$VENDOR_DIR"
else
  git -C "$VENDOR_DIR" fetch --depth 1 origin
  git -C "$VENDOR_DIR" pull --ff-only
fi

cat <<MSG
ChemEnzyRetroPlanner checkout ready:
  $VENDOR_DIR

Heavy model/environment setup is intentionally not run by this bootstrap.
Use the upstream setup_ChemEnzyRetroPlanner.sh inside an isolated conda env
when you want a real ChemEnzy baseline run.
MSG

