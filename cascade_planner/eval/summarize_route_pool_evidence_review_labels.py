"""Summarize validated route-pool evidence review labels."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_evidence_review_label_summary.v1"


JUDGMENT_FIELDS = (
    "route_plausible",
    "block_transform_correct",
    "support_precedent_relevant",
    "cascade_coherent",
)


def summarize_route_pool_evidence_review_labels(
    *,
    labels_jsonl: Path,
    output_json: Path,
    output_md: Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(labels_jsonl)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "labels_jsonl": str(labels_jsonl),
            "output_json": str(output_json),
            "output_md": str(output_md or output_json.with_suffix(".md")),
        },
        "summary": _summary(rows),
        "by_evidence_class": _group_summary(rows, "evidence_class"),
        "by_source_pool": _group_summary(rows, "source_pool"),
        "by_transform_sanity": _transform_sanity_summary(rows),
        "score_calibration": _score_calibration(rows),
        "decision": _decision(rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "reviewed_classes": dict(Counter(str(row.get("evidence_class") or "") for row in rows)),
        "reviewed_pools": dict(Counter(str(row.get("source_pool") or "") for row in rows)),
        "priority_counts": dict(Counter(_review(row).get("priority") for row in rows)),
        "risk_tag_counts": dict(Counter(tag for row in rows for tag in (_review(row).get("risk_tags") or []))),
        "judgments": {field: dict(Counter(_review(row).get(field) for row in rows)) for field in JUDGMENT_FIELDS},
        "usable_positive_rows": sum(1 for row in rows if _usable_positive(row)),
        "usable_negative_rows": sum(1 for row in rows if _usable_negative(row)),
        "unclear_rows": sum(1 for row in rows if _has_unclear(row)),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "")].append(row)
    return {name: _compact_group_summary(group) for name, group in sorted(grouped.items())}


def _compact_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "rows": total,
        "route_plausible_yes_rate": _rate(sum(1 for row in rows if _review(row).get("route_plausible") == "yes"), total),
        "block_transform_correct_yes_rate": _rate(sum(1 for row in rows if _review(row).get("block_transform_correct") == "yes"), total),
        "support_precedent_relevant_yes_rate": _rate(sum(1 for row in rows if _review(row).get("support_precedent_relevant") == "yes"), total),
        "cascade_coherent_yes_rate": _rate(sum(1 for row in rows if _review(row).get("cascade_coherent") == "yes"), total),
        "usable_positive_rate": _rate(sum(1 for row in rows if _usable_positive(row)), total),
        "usable_negative_rate": _rate(sum(1 for row in rows if _usable_negative(row)), total),
        "unclear_rate": _rate(sum(1 for row in rows if _has_unclear(row)), total),
        "priority_counts": dict(Counter(_review(row).get("priority") for row in rows)),
        "wrong_transform_label_risk_rate": _rate(
            sum(1 for row in rows if "wrong_transform_label" in (_review(row).get("risk_tags") or [])),
            total,
        ),
    }


def _transform_sanity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {
        "label_mismatch_warning": [row for row in rows if _has_transform_label_warning(row)],
        "no_label_mismatch_warning": [row for row in rows if not _has_transform_label_warning(row)],
    }
    return {name: _compact_group_summary(group) for name, group in grouped.items()}


def _score_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("best_any_block_min_sim", "best_same_pair_block_min_sim", "known_transform_step_fraction")
    out = {}
    for field in fields:
        out[field] = {
            "all": _numeric_summary(_score(row, field) for row in rows),
            "usable_positive": _numeric_summary(_score(row, field) for row in rows if _usable_positive(row)),
            "usable_negative": _numeric_summary(_score(row, field) for row in rows if _usable_negative(row)),
            "unclear": _numeric_summary(_score(row, field) for row in rows if _has_unclear(row)),
        }
    return out


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    reviewed_enough = total >= 20
    positive = sum(1 for row in rows if _usable_positive(row))
    negative = sum(1 for row in rows if _usable_negative(row))
    unclear = sum(1 for row in rows if _has_unclear(row))
    return {
        "review_rows": total,
        "reviewed_enough_for_proxy_calibration": reviewed_enough,
        "usable_positive_rate": _rate(positive, total),
        "usable_negative_rate": _rate(negative, total),
        "unclear_rate": _rate(unclear, total),
        "recommendation": (
            "enough labels for preliminary proxy calibration"
            if reviewed_enough
            else "insufficient real review labels; run human/LLM review before training"
        ),
    }


def _review(row: dict[str, Any]) -> dict[str, Any]:
    review = row.get("expert_review") or {}
    return review if isinstance(review, dict) else {}


def _usable_positive(row: dict[str, Any]) -> bool:
    review = _review(row)
    return (
        review.get("route_plausible") == "yes"
        and review.get("block_transform_correct") == "yes"
        and review.get("support_precedent_relevant") == "yes"
        and review.get("cascade_coherent") == "yes"
        and review.get("priority") in {"high", "medium"}
    )


def _usable_negative(row: dict[str, Any]) -> bool:
    review = _review(row)
    return (
        review.get("priority") == "reject"
        or review.get("route_plausible") == "no"
        or review.get("block_transform_correct") == "no"
        or review.get("support_precedent_relevant") == "no"
        or review.get("cascade_coherent") == "no"
    )


def _has_unclear(row: dict[str, Any]) -> bool:
    review = _review(row)
    return any(review.get(field) == "unclear" for field in JUDGMENT_FIELDS)


def _has_transform_label_warning(row: dict[str, Any]) -> bool:
    sanity = row.get("transform_sanity") or {}
    return bool(sanity.get("block_has_label_mismatch"))


def _score(row: dict[str, Any], field: str) -> float | None:
    value = (row.get("diagnostic_scores") or {}).get(field)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_summary(values: Any) -> dict[str, Any]:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Route Pool Evidence Review Label Summary",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(result.get("decision") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (result.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## By Evidence Class", "", "| Class | Rows | Positive Rate | Negative Rate | Unclear Rate |", "|---|---:|---:|---:|---:|"])
    for cls, row in (result.get("by_evidence_class") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    cls,
                    str(row.get("rows")),
                    str(row.get("usable_positive_rate")),
                    str(row.get("usable_negative_rate")),
                    str(row.get("unclear_rate")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## By Transform Sanity", "", "| Group | Rows | Positive Rate | Negative Rate | Wrong-Label Risk Rate |", "|---|---:|---:|---:|---:|"])
    for group, row in (result.get("by_transform_sanity") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(row.get("rows")),
                    str(row.get("usable_positive_rate")),
                    str(row.get("usable_negative_rate")),
                    str(row.get("wrong_transform_label_risk_rate")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validated route-pool evidence review labels")
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args()
    result = summarize_route_pool_evidence_review_labels(
        labels_jsonl=Path(args.labels_jsonl),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
    )
    print(json.dumps({"summary": result["summary"], "decision": result["decision"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
