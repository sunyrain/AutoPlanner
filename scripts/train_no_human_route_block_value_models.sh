#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/AutoPlanner}"
cd "$ROOT"

PACK="${PACK:-results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl}"
PACK_REPORT="${PACK_REPORT:-${PACK%.jsonl}_report.json}"
OUT_DIR="${OUT_DIR:-results/shared/model_strengthening_20260519_no_human_route_block_value_models}"
MIN_MRR_DELTA_VS_RETRIEVAL="${MIN_MRR_DELTA_VS_RETRIEVAL:-0.03}"

if [[ ! -f "$PACK" ]]; then
  echo "Route/block value pack does not exist: $PACK" >&2
  exit 2
fi
if [[ ! -f "$PACK_REPORT" ]]; then
  echo "Route/block value pack report does not exist: $PACK_REPORT" >&2
  exit 2
fi

python - "$PACK_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = report.get("weak_label_positive_counts") or {}
required = ["no_human_consensus_positive", "no_human_consensus_negative"]
missing = [name for name in required if int(counts.get(name) or 0) <= 0]
if missing:
    print(
        "Pack does not expose no-human consensus tasks; rebuild the pack first: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    raise SystemExit(2)
PY

mkdir -p "$OUT_DIR"

run_model() {
  local name="$1"
  local positive_task="$2"
  local negative_task="$3"
  shift
  shift
  shift
  PYTHONPATH=. python -m cascade_planner.eval.train_route_block_value_model \
    --pack "$PACK" \
    --output-dir "$OUT_DIR/$name" \
    --positive-task "$positive_task" \
    --negative-task "$negative_task" \
    "$@"
}

run_ablation_set() {
  local prefix="$1"
  local positive_task="$2"
  local negative_task="$3"

  run_model "${prefix}_all_features" "$positive_task" "$negative_task"
  run_model "${prefix}_no_audit" "$positive_task" "$negative_task" --exclude-group product_audit
  run_model "${prefix}_no_audit_no_retrieval" "$positive_task" "$negative_task" \
    --exclude-group product_audit \
    --exclude-group cascade_retrieval \
    --exclude-group route_step_v4_evidence
  run_model "${prefix}_no_audit_no_cascade" "$positive_task" "$negative_task" \
    --exclude-group product_audit \
    --exclude-group cascade_retrieval \
    --exclude-group route_step_v4_evidence \
    --exclude-group learned_ccts
}

run_ablation_set no_human_consensus no_human_consensus_positive no_human_consensus_negative
run_ablation_set no_human_route no_human_route_positive no_human_route_negative

# Compatibility aliases for older reports/docs that used no_human_* to mean the
# stricter consensus task.
run_model no_human_all_features no_human_consensus_positive no_human_consensus_negative
run_model no_human_no_audit no_human_consensus_positive no_human_consensus_negative --exclude-group product_audit
run_model no_human_no_audit_no_retrieval no_human_consensus_positive no_human_consensus_negative \
  --exclude-group product_audit \
  --exclude-group cascade_retrieval \
  --exclude-group route_step_v4_evidence
run_model no_human_no_audit_no_cascade no_human_consensus_positive no_human_consensus_negative \
  --exclude-group product_audit \
  --exclude-group cascade_retrieval \
  --exclude-group route_step_v4_evidence \
  --exclude-group learned_ccts

OUT_DIR="$OUT_DIR" PACK="$PACK" MIN_MRR_DELTA_VS_RETRIEVAL="$MIN_MRR_DELTA_VS_RETRIEVAL" PYTHONPATH=. python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
min_delta = float(os.environ.get("MIN_MRR_DELTA_VS_RETRIEVAL") or 0.03)
rows = []
for report_path in sorted(out_dir.glob("*/route_block_value_model_report.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    interp = report.get("interpretation") or {}
    selection = report.get("selection") or {}
    rows.append(
        {
            "model": report_path.parent.name,
            "positive_task": (report.get("metadata") or {}).get("positive_task"),
            "negative_task": (report.get("metadata") or {}).get("negative_task"),
            "model_mrr": selection.get("model_test_mrr_covered"),
            "native_mrr": selection.get("native_rank_test_mrr_covered"),
            "retrieval_mrr": selection.get("retrieval_only_test_mrr_covered"),
            "selected_method": selection.get("selected_method"),
            "model_minus_native_mrr": interp.get("model_minus_native_mrr"),
            "model_minus_retrieval_mrr": interp.get("model_minus_retrieval_only_mrr"),
            "clears_retrieval_only": interp.get("clears_retrieval_only"),
            "evidence_status": (report.get("evidence_provenance_audit") or {}).get("status"),
        }
    )
control_rows = [
    row
    for row in rows
    if str(row.get("model") or "").endswith("_no_audit_no_retrieval")
]
control = max(control_rows, key=lambda row: float(row.get("model_minus_retrieval_mrr") or 0.0), default={})
control_delta = float(control.get("model_minus_retrieval_mrr") or 0.0)
signal_present = any(bool(row.get("clears_retrieval_only")) for row in rows)
strict_gate_passed = bool(control) and control_delta >= min_delta
summary = {
    "schema_version": "no_human_route_block_value_ablation_summary.v1",
    "pack": str(Path(os.environ.get("PACK", "results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl"))),
    "decision": {
        "expert_labels_required": False,
        "fixed_pool_signal_present": signal_present,
        "fixed_pool_training_gate_passed": strict_gate_passed,
        "strict_fixed_pool_gate_passed": strict_gate_passed,
        "control_model": control.get("model"),
        "control_positive_task": control.get("positive_task"),
        "control_negative_task": control.get("negative_task"),
        "control_model_minus_retrieval_mrr": round(control_delta, 6),
        "required_model_minus_retrieval_mrr": min_delta,
        "promote_search_time": False,
        "reason": (
            "no-human weak-supervision fixed-pool ablation; strict gate requires "
            "the no-audit/no-retrieval control to beat retrieval-only by the configured margin, "
            "and search-time promotion still requires audit control and live-search quality lift"
        ),
    },
    "models": rows,
}
(out_dir / "no_human_route_block_value_ablation_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
lines = [
    "# No-Human Route/Block Value Ablation",
    "",
    "Expert labels required: `False`",
    f"Fixed-pool signal present: `{signal_present}`",
    f"Strict fixed-pool gate passed: `{strict_gate_passed}`",
    f"Control model: `{control.get('model')}`",
    f"Control positive task: `{control.get('positive_task')}`",
    f"Control model-retrieval MRR: `{control_delta:.6f}`",
    f"Required model-retrieval MRR: `{min_delta:.6f}`",
    "Search-time promotion: `False`",
    "",
    "| model | MRR | native | retrieval | model-native | model-retrieval | selected | evidence |",
    "|---|---:|---:|---:|---:|---:|---|---|",
]
for row in rows:
    lines.append(
        f"| `{row['model']}` | {row['model_mrr']} | {row['native_mrr']} | {row['retrieval_mrr']} | "
        f"{row['model_minus_native_mrr']} | {row['model_minus_retrieval_mrr']} | "
        f"`{row['selected_method']}` | `{row['evidence_status']}` |"
    )
lines.extend(
    [
        "",
        "This trains on automatic no_human_consensus and no_human_route labels only. It is not an expert-review model.",
        "Search-time promotion still requires beating retrieval/audit controls and live-search quality lift.",
        "",
    ]
)
(out_dir / "no_human_route_block_value_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary["decision"], indent=2, ensure_ascii=False))
PY

echo "No-human route/block value models written under:"
echo "  $OUT_DIR"
