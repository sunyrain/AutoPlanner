"""Gate whether route-pool evidence review labels are ready for training use."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_evidence_review_promotion_gate.v1"


def gate_route_pool_evidence_review_promotion(
    *,
    label_summary_json: Path,
    output_json: Path,
    output_md: Path | None = None,
    min_rows: int = 30,
    min_usable_positive: int = 5,
    min_usable_negative: int = 5,
    max_unclear_rate: float = 0.50,
    min_evidence_classes: int = 2,
) -> dict[str, Any]:
    summary_payload = _read_json(label_summary_json)
    summary = summary_payload.get("summary") or {}
    decision = summary_payload.get("decision") or {}
    class_rows = summary.get("reviewed_classes") or {}
    rows = int(summary.get("rows") or decision.get("review_rows") or 0)
    usable_positive = int(summary.get("usable_positive_rows") or 0)
    usable_negative = int(summary.get("usable_negative_rows") or 0)
    unclear_rate = float(decision.get("unclear_rate") if decision.get("unclear_rate") is not None else _rate(int(summary.get("unclear_rows") or 0), rows))
    covered_classes = sum(1 for value in class_rows.values() if int(value or 0) > 0)
    checks = [
        _check(rows >= min_rows, "minimum reviewed rows", {"actual": rows, "required": min_rows}),
        _check(
            usable_positive >= min_usable_positive,
            "minimum usable positive labels",
            {"actual": usable_positive, "required": min_usable_positive},
        ),
        _check(
            usable_negative >= min_usable_negative,
            "minimum usable negative labels",
            {"actual": usable_negative, "required": min_usable_negative},
        ),
        _check(
            unclear_rate <= max_unclear_rate,
            "maximum unclear rate",
            {"actual": round(unclear_rate, 6), "required_max": max_unclear_rate},
        ),
        _check(
            covered_classes >= min_evidence_classes,
            "minimum evidence-class coverage",
            {"actual": covered_classes, "required": min_evidence_classes, "classes": class_rows},
        ),
    ]
    ready = all(row["ok"] for row in checks)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "label_summary_json": str(label_summary_json),
            "output_json": str(output_json),
            "output_md": str(output_md or output_json.with_suffix(".md")),
            "thresholds": {
                "min_rows": min_rows,
                "min_usable_positive": min_usable_positive,
                "min_usable_negative": min_usable_negative,
                "max_unclear_rate": max_unclear_rate,
                "min_evidence_classes": min_evidence_classes,
            },
        },
        "ready_for_training": ready,
        "recommendation": (
            "labels can be used for preliminary proxy calibration; still keep generator fixed"
            if ready
            else "do not train or promote a scorer from these labels yet"
        ),
        "checks": checks,
        "source_decision": decision,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _check(ok: bool, name: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"ok": bool(ok), "name": name, "details": details}


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Route Pool Evidence Review Promotion Gate",
        "",
        f"- ready_for_training: `{result.get('ready_for_training')}`",
        f"- recommendation: `{result.get('recommendation')}`",
        "",
        "## Checks",
        "",
        "| Check | OK | Details |",
        "|---|---:|---|",
    ]
    for row in result.get("checks") or []:
        lines.append(f"| `{row.get('name')}` | `{row.get('ok')}` | `{json.dumps(row.get('details') or {}, ensure_ascii=False)}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate route-pool evidence review labels for training use")
    parser.add_argument("--label-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-usable-positive", type=int, default=5)
    parser.add_argument("--min-usable-negative", type=int, default=5)
    parser.add_argument("--max-unclear-rate", type=float, default=0.50)
    parser.add_argument("--min-evidence-classes", type=int, default=2)
    args = parser.parse_args()
    result = gate_route_pool_evidence_review_promotion(
        label_summary_json=Path(args.label_summary_json),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        min_rows=int(args.min_rows),
        min_usable_positive=int(args.min_usable_positive),
        min_usable_negative=int(args.min_usable_negative),
        max_unclear_rate=float(args.max_unclear_rate),
        min_evidence_classes=int(args.min_evidence_classes),
    )
    print(json.dumps({"ready_for_training": result["ready_for_training"], "recommendation": result["recommendation"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
