"""Build a reviewer packet for the route-pool calibration subset."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_review_calibration_packet.v1"

YES_NO_UNCLEAR = "yes | no | unclear"
PRIORITY = "high | medium | low | reject"
RISK_TAGS = [
    "wrong_transform_label",
    "irrelevant_precedent",
    "not_cascade",
    "trivial_stock_closure",
    "missing_reaction_detail",
    "condition_incompatibility",
    "unsupported_selectivity",
    "atom_mapping_or_stoichiometry_issue",
    "other",
]


def build_route_pool_review_calibration_packet(
    *,
    subset_csv: Path,
    subset_report_json: Path,
    output_dir: Path,
    pipeline_command_output_dir: Path | None = None,
    min_evidence_classes: int = 2,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    editable_csv = output_dir / "route_pool_evidence_review_calibration_subset_TO_FILL.csv"
    shutil.copyfile(subset_csv, editable_csv)
    report = _read_json(subset_report_json)
    summary = report.get("summary") or {}
    pipeline_dir = pipeline_command_output_dir or output_dir / "csv_pipeline_result"
    command = (
        "PYTHONPATH=. python -m cascade_planner.eval.run_route_pool_evidence_review_csv_pipeline "
        f"--review-csv {editable_csv} "
        f"--output-dir {pipeline_dir} "
        "--prefix calibration_human_route_pool_evidence_review "
        f"--min-evidence-classes {int(min_evidence_classes)}"
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "subset_csv": str(subset_csv),
            "subset_report_json": str(subset_report_json),
            "output_dir": str(output_dir),
            "editable_csv": str(editable_csv),
            "pipeline_command_output_dir": str(pipeline_dir),
            "min_evidence_classes": int(min_evidence_classes),
        },
        "summary": summary,
        "editable_csv": str(editable_csv),
        "allowed_values": {
            "expert_route_plausible": YES_NO_UNCLEAR,
            "expert_block_transform_correct": YES_NO_UNCLEAR,
            "expert_support_precedent_relevant": YES_NO_UNCLEAR,
            "expert_cascade_coherent": YES_NO_UNCLEAR,
            "expert_priority": PRIORITY,
            "expert_risk_tags": RISK_TAGS,
        },
        "context_columns": ["target_id", "route_id", "source_value_pack", "value_split"],
        "pipeline_command": command,
        "contract": {
            "review_packet_only": True,
            "does_not_create_labels": True,
            "filled_csv_must_be_validated_by_pipeline": True,
            "context_columns_must_remain_unchanged": True,
        },
    }
    (output_dir / "route_pool_review_calibration_packet.json").write_text(
        json.dumps(packet, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(packet), encoding="utf-8")
    return packet


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _readme(packet: dict[str, Any]) -> str:
    summary = packet.get("summary") or {}
    lines = [
        "# Route Pool Evidence Review Calibration Packet",
        "",
        "## What To Fill",
        "",
        f"Editable CSV: `{packet.get('editable_csv')}`",
        "",
        "Fill only these expert columns:",
        "",
        "| Column | Allowed values |",
        "|---|---|",
        f"| `expert_route_plausible` | `{YES_NO_UNCLEAR}` |",
        f"| `expert_block_transform_correct` | `{YES_NO_UNCLEAR}` |",
        f"| `expert_support_precedent_relevant` | `{YES_NO_UNCLEAR}` |",
        f"| `expert_cascade_coherent` | `{YES_NO_UNCLEAR}` |",
        f"| `expert_priority` | `{PRIORITY}` |",
        f"| `expert_risk_tags` | semicolon-separated tags from `{'; '.join(RISK_TAGS)}` |",
        "| `expert_reject_reason` | short text; required when `expert_priority=reject` |",
        "| `expert_comments` | short rationale; required for every reviewed row |",
        "",
        "Leave a row blank if it cannot be reviewed. Blank rows are counted as `unreviewed`, not labels.",
        "Keep identity/context columns such as `target_id`, `route_id`, `source_value_pack`, and `value_split` unchanged.",
        "Use `value_split` only to keep filled positive/negative examples balanced across train, val, and test.",
        "",
        "## Review Rubric",
        "",
        "- Judge the shown route block, not the whole route.",
        "- Do not treat diagnostic evidence class as ground truth.",
        "- Do not trust transform text labels blindly; use `transform_label_warning` and the reaction SMILES.",
        "- Use `wrong_transform_label` when the text label conflicts with the shown chemistry.",
        "- Do not reward a route merely because it is stock-closed.",
        "- Use `unclear` when the reaction block cannot be judged from the provided information.",
        "",
        "## Batch Balance",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Validate After Filling",
        "",
        "Run:",
        "",
        "```bash",
        packet.get("pipeline_command") or "",
        "```",
        "",
        "Training is allowed only if both `signal_calibration.ready_for_proxy_training` and "
        "`promotion_gate.ready_for_training` pass after validation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build route-pool review calibration packet")
    parser.add_argument("--subset-csv", required=True)
    parser.add_argument("--subset-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pipeline-command-output-dir")
    parser.add_argument("--min-evidence-classes", type=int, default=2)
    args = parser.parse_args()
    packet = build_route_pool_review_calibration_packet(
        subset_csv=Path(args.subset_csv),
        subset_report_json=Path(args.subset_report_json),
        output_dir=Path(args.output_dir),
        pipeline_command_output_dir=Path(args.pipeline_command_output_dir) if args.pipeline_command_output_dir else None,
        min_evidence_classes=args.min_evidence_classes,
    )
    print(json.dumps({"summary": packet["summary"], "editable_csv": packet["editable_csv"], "pipeline_command": packet["pipeline_command"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
