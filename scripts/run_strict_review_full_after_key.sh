#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/AutoPlanner}"
cd "$ROOT"

READINESS_JSON="${READINESS_JSON:-results/shared/model_strengthening_20260519_strict_review_readiness.json}"
RUN_EXTENDED_IF_NOT_READY="${RUN_EXTENDED_IF_NOT_READY:-0}"
ALLOW_NOT_READY_EXIT_ZERO="${ALLOW_NOT_READY_EXIT_ZERO:-0}"

PRIMARY_MERGED_PACK="${PRIMARY_MERGED_PACK:-results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline/strict_model_real_merged_route_block_value_pack.jsonl}"
PRIMARY_MERGE_REPORT="${PRIMARY_MERGE_REPORT:-results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline/strict_model_real_merged_route_block_value_pack_report.json}"
EXTENDED_MERGED_PACK="${EXTENDED_MERGED_PACK:-results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/real_review_pipeline/strict_model_real_300_merged_route_block_value_pack.jsonl}"
EXTENDED_MERGE_REPORT="${EXTENDED_MERGE_REPORT:-results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/real_review_pipeline/strict_model_real_300_merged_route_block_value_pack_report.json}"

refresh_readiness() {
  PYTHONPATH=. python -m cascade_planner.eval.check_strict_review_pipeline_readiness \
    --root "$ROOT" \
    --output-json "$READINESS_JSON"
}

merge_ready() {
  local report="$1"
  python - "$report" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if (payload.get("decision") or {}).get("ready_for_expert_training") else 1)
PY
}

train_from_merge() {
  local pack="$1"
  local report="$2"
  local out_dir="$3"
  MERGED_PACK="$pack" \
  MERGE_REPORT="$report" \
  OUT_DIR="$out_dir" \
  PYTHONPATH=. scripts/train_strict_model_review_value_models.sh
}

echo "Running primary 120-row strict review..."
PYTHONPATH=. scripts/run_strict_model_review_real.sh
refresh_readiness

if merge_ready "$PRIMARY_MERGE_REPORT"; then
  echo "Primary merged pack is ready; training expert value models."
  train_from_merge "$PRIMARY_MERGED_PACK" "$PRIMARY_MERGE_REPORT" \
    "results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline/expert_value_models"
  exit 0
fi

if [[ "$RUN_EXTENDED_IF_NOT_READY" != "1" ]]; then
  echo "Primary merged pack is not ready for expert training."
  echo "Set RUN_EXTENDED_IF_NOT_READY=1 to run the 300-row fallback automatically."
  if [[ "$ALLOW_NOT_READY_EXIT_ZERO" == "1" ]]; then
    exit 0
  fi
  exit 4
fi

echo "Running 300-row fallback strict review..."
PYTHONPATH=. scripts/run_strict_model_review_real_extended.sh
refresh_readiness

if merge_ready "$EXTENDED_MERGE_REPORT"; then
  echo "Extended merged pack is ready; training expert value models."
  train_from_merge "$EXTENDED_MERGED_PACK" "$EXTENDED_MERGE_REPORT" \
    "results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/real_review_pipeline/expert_value_models"
  exit 0
fi

echo "Extended merged pack is still not ready for expert training; inspect $READINESS_JSON."
if [[ "$ALLOW_NOT_READY_EXIT_ZERO" == "1" ]]; then
  exit 0
fi
exit 4
