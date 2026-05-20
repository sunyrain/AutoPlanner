#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/AutoPlanner}"
cd "$ROOT"

MERGED_PACK="${MERGED_PACK:-results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline/strict_model_real_merged_route_block_value_pack.jsonl}"
MERGE_REPORT="${MERGE_REPORT:-${MERGED_PACK%.jsonl}_report.json}"
OUT_DIR="${OUT_DIR:-results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline/expert_value_models}"

if [[ ! -f "$MERGED_PACK" ]]; then
  echo "Merged value pack does not exist: $MERGED_PACK" >&2
  exit 2
fi
if [[ ! -f "$MERGE_REPORT" ]]; then
  echo "Merge report does not exist: $MERGE_REPORT" >&2
  exit 2
fi

python - "$MERGE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = report.get("decision") or {}
if not decision.get("ready_for_expert_training"):
    print(
        "Merged pack is not ready for expert training: "
        + json.dumps(decision, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
    )
    raise SystemExit(2)
PY

mkdir -p "$OUT_DIR"

run_model() {
  local name="$1"
  shift
  PYTHONPATH=. python -m cascade_planner.eval.train_route_block_value_model \
    --pack "$MERGED_PACK" \
    --output-dir "$OUT_DIR/$name" \
    --positive-task expert_review_positive \
    --negative-task expert_review_negative \
    "$@"
}

run_model expert_all_features
run_model expert_no_audit --exclude-group product_audit
run_model expert_no_audit_no_retrieval \
  --exclude-group product_audit \
  --exclude-group cascade_retrieval \
  --exclude-group route_step_v4_evidence
run_model expert_no_audit_no_cascade \
  --exclude-group product_audit \
  --exclude-group cascade_retrieval \
  --exclude-group route_step_v4_evidence \
  --exclude-group learned_ccts

echo "Expert review value models written under:"
echo "  $OUT_DIR"
