#!/usr/bin/env python3
"""Evaluate the rule cascade verifier on a perturbation pack."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cascade_planner.cascade_verifier import verify_cascade_route


def main() -> None:
    args = _parse_args()
    pack = json.loads(args.input.read_text(encoding="utf-8"))
    examples = [row for row in pack.get("examples") or [] if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    confusion: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    expected_reason_hits = 0
    expected_reason_total = 0
    scored = 0
    correct = 0

    for row in examples:
        report = verify_cascade_route(row.get("cascade") or {}, target_smiles=row.get("target_smiles")).to_dict()
        predicted_label = 1 if report.get("feasible") else 0
        label = row.get("label")
        if label in {0, 1}:
            scored += 1
            correct += int(predicted_label == int(label))
            confusion[_confusion_key(int(label), predicted_label)] += 1
        predicted_reasons = set((report.get("reason_counts") or {}).keys())
        for reason, count in (report.get("reason_counts") or {}).items():
            reason_counts[str(reason)] += int(count)
        expected = {str(reason) for reason in row.get("expected_failure_reasons") or []}
        if expected:
            expected_reason_total += 1
            if expected <= predicted_reasons:
                expected_reason_hits += 1
        rows.append(
            {
                "example_id": row.get("example_id"),
                "label": label,
                "predicted_label": predicted_label,
                "perturbation_type": row.get("perturbation_type"),
                "expected_failure_reasons": sorted(expected),
                "predicted_reason_counts": report.get("reason_counts") or {},
                "feasible": bool(report.get("feasible")),
                "score": report.get("score"),
            }
        )

    summary = {
        "schema_version": "cascade_verifier_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(args.input),
        "n_examples": len(examples),
        "n_scored": scored,
        "accuracy": round(correct / scored, 4) if scored else None,
        "confusion": dict(sorted(confusion.items())),
        "expected_reason_coverage": round(expected_reason_hits / expected_reason_total, 4) if expected_reason_total else None,
        "expected_reason_hit_count": expected_reason_hits,
        "expected_reason_total": expected_reason_total,
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "contract": (
            "This evaluates rule-label recovery on perturbations. It is not a measure of expert synthesis feasibility."
        ),
    }
    result = {"summary": summary, "examples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.markdown:
        _write_markdown(summary, args.markdown)
    print(json.dumps(summary, indent=2))


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cascade Verifier Perturbation Evaluation",
        "",
        "## Summary",
        "",
        f"- Examples: `{summary.get('n_examples')}`",
        f"- Scored examples: `{summary.get('n_scored')}`",
        f"- Label accuracy: `{summary.get('accuracy')}`",
        f"- Expected-reason coverage: `{summary.get('expected_reason_coverage')}`",
        f"- Expected-reason hits: `{summary.get('expected_reason_hit_count')}` / `{summary.get('expected_reason_total')}`",
        "",
        "## Confusion",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
    ]
    for key, count in (summary.get("confusion") or {}).items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Reasons", "", "| Reason | Count |", "| --- | ---: |"])
    for reason, count in (summary.get("reason_counts") or {}).items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Contract", "", str(summary.get("contract") or "")])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _confusion_key(label: int, predicted: int) -> str:
    if label == 1 and predicted == 1:
        return "tp_feasible"
    if label == 1 and predicted == 0:
        return "fn_seed_rejected"
    if label == 0 and predicted == 0:
        return "tn_negative_rejected"
    return "fp_negative_accepted"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cascade verifier on perturbation pack")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
