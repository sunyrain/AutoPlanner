#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/AutoPlanner}"
cd "$ROOT"

PACKET_SIZE="${PACKET_SIZE:-120}"
VALUE_PACK="${VALUE_PACK:-results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl}"
READINESS_JSON="${READINESS_JSON:-results/shared/model_strengthening_20260519_strict_review_readiness.json}"
ALLOW_NOT_READY_EXIT_ZERO="${ALLOW_NOT_READY_EXIT_ZERO:-0}"

case "$PACKET_SIZE" in
  120)
    DEFAULT_CSV="results/shared/model_strengthening_20260519_strict_model_review_packet/route_pool_evidence_review_calibration_subset_TO_FILL.csv"
    DEFAULT_OUT_DIR="results/shared/model_strengthening_20260519_strict_model_review_packet/csv_ingest_pipeline"
    DEFAULT_PREFIX="strict_model_human"
    ;;
  300)
    DEFAULT_CSV="results/shared/model_strengthening_20260519_strict_model_review_packet_300/route_pool_evidence_review_calibration_subset_TO_FILL.csv"
    DEFAULT_OUT_DIR="results/shared/model_strengthening_20260519_strict_model_review_packet_300/csv_ingest_pipeline"
    DEFAULT_PREFIX="strict_model_human_300"
    ;;
  *)
    echo "PACKET_SIZE must be 120 or 300, got: $PACKET_SIZE" >&2
    exit 2
    ;;
esac

FILLED_REVIEW_CSV="${FILLED_REVIEW_CSV:-$DEFAULT_CSV}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
PREFIX="${REVIEW_PREFIX:-$DEFAULT_PREFIX}"
MERGED_PACK="${MERGED_PACK:-${OUT_DIR}/${PREFIX}_merged_route_block_value_pack.jsonl}"
MERGE_REPORT="${MERGE_REPORT:-${OUT_DIR}/${PREFIX}_merged_route_block_value_pack_report.json}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${OUT_DIR}/expert_value_models}"
TRAIN_IF_READY="${TRAIN_IF_READY:-1}"

MIN_ROWS="${MIN_ROWS:-30}"
MIN_USABLE_POSITIVE="${MIN_USABLE_POSITIVE:-5}"
MIN_USABLE_NEGATIVE="${MIN_USABLE_NEGATIVE:-5}"
MAX_UNCLEAR_RATE="${MAX_UNCLEAR_RATE:-0.50}"
MIN_EVIDENCE_CLASSES="${MIN_EVIDENCE_CLASSES:-1}"
MIN_AUC="${MIN_AUC:-0.65}"
MIN_EXPERT_USABLE_POSITIVE="${MIN_EXPERT_USABLE_POSITIVE:-30}"
MIN_EXPERT_USABLE_NEGATIVE="${MIN_EXPERT_USABLE_NEGATIVE:-30}"

if [[ ! -f "$FILLED_REVIEW_CSV" ]]; then
  echo "Filled review CSV does not exist: $FILLED_REVIEW_CSV" >&2
  exit 2
fi
if [[ ! -f "$VALUE_PACK" ]]; then
  echo "Value pack does not exist: $VALUE_PACK" >&2
  exit 2
fi

python - "$FILLED_REVIEW_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
expert_decision_fields = (
    "expert_route_plausible",
    "expert_block_transform_correct",
    "expert_support_precedent_relevant",
    "expert_cascade_coherent",
    "expert_priority",
)
with path.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
filled = sum(
    1
    for row in rows
    if any(str(row.get(field) or "").strip() for field in expert_decision_fields)
)
filled_missing_route_id = sum(
    1
    for row in rows
    if any(str(row.get(field) or "").strip() for field in expert_decision_fields)
    and not str(row.get("route_id") or "").strip()
)
print(f"CSV rows: {len(rows)}; rows with expert decision fields filled: {filled}")
if filled == 0:
    print("No filled expert decision rows found; refusing to create an empty human-review merge.", file=sys.stderr)
    raise SystemExit(3)
if filled_missing_route_id:
    print(
        f"Filled expert decision rows missing route_id: {filled_missing_route_id}; "
        "route_id is required for merge back into the value pack.",
        file=sys.stderr,
    )
    raise SystemExit(3)
PY

mkdir -p "$OUT_DIR"

PYTHONPATH=. python -m cascade_planner.eval.run_route_pool_evidence_review_csv_pipeline \
  --review-csv "$FILLED_REVIEW_CSV" \
  --output-dir "$OUT_DIR" \
  --prefix "$PREFIX" \
  --min-rows "$MIN_ROWS" \
  --min-usable-positive "$MIN_USABLE_POSITIVE" \
  --min-usable-negative "$MIN_USABLE_NEGATIVE" \
  --max-unclear-rate "$MAX_UNCLEAR_RATE" \
  --min-evidence-classes "$MIN_EVIDENCE_CLASSES" \
  --min-auc "$MIN_AUC"

PYTHONPATH=. python -m cascade_planner.eval.build_route_block_review_label_pack \
  --input "${OUT_DIR}/${PREFIX}_labels.jsonl" \
  --output-jsonl "${OUT_DIR}/${PREFIX}_review_label_pack.jsonl" \
  --report "${OUT_DIR}/${PREFIX}_review_label_pack_report.json" \
  --dataset "${PREFIX}_review_labels"

PYTHONPATH=. python -m cascade_planner.eval.merge_route_block_review_labels \
  --value-pack "$VALUE_PACK" \
  --review-label-pack "${OUT_DIR}/${PREFIX}_review_label_pack.jsonl" \
  --output-jsonl "$MERGED_PACK" \
  --report "$MERGE_REPORT" \
  --dataset "${PREFIX}_merged_route_block_value" \
  --min-usable-positive "$MIN_EXPERT_USABLE_POSITIVE" \
  --min-usable-negative "$MIN_EXPERT_USABLE_NEGATIVE"

python -m json.tool "${OUT_DIR}/${PREFIX}_csv_pipeline_manifest.json" >/dev/null
python -m json.tool "${OUT_DIR}/${PREFIX}_review_label_pack_report.json" >/dev/null
python -m json.tool "$MERGE_REPORT" >/dev/null

PYTHONPATH=. python -m cascade_planner.eval.check_strict_review_pipeline_readiness \
  --root "$ROOT" \
  --output-json "$READINESS_JSON" >/dev/null

merge_ready() {
  local report="$1"
  python - "$report" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if (payload.get("decision") or {}).get("ready_for_expert_training") else 1)
PY
}

MERGE_READY=0
if merge_ready "$MERGE_REPORT"; then
  MERGE_READY=1
  echo "Human CSV merged pack is ready for expert value-model training."
  if [[ "$TRAIN_IF_READY" == "1" ]]; then
    MERGED_PACK="$MERGED_PACK" \
    MERGE_REPORT="$MERGE_REPORT" \
    OUT_DIR="$TRAIN_OUT_DIR" \
    PYTHONPATH=. scripts/train_strict_model_review_value_models.sh
  fi
else
  echo "Human CSV merged pack is not ready for expert training; inspect:"
  echo "  $MERGE_REPORT"
fi

echo "Human CSV review pipeline complete:"
echo "  ${OUT_DIR}/${PREFIX}_csv_pipeline_manifest.json"
echo "  ${OUT_DIR}/${PREFIX}_review_label_pack_report.json"
echo "  $MERGE_REPORT"

if [[ "$MERGE_READY" != "1" && "$ALLOW_NOT_READY_EXIT_ZERO" != "1" ]]; then
  exit 4
fi
