"""Build a compact route/block review label pack.

These labels are calibration/review evidence, not route preference labels.  The
builder keeps that contract explicit so small self-review batches do not get
accidentally promoted to main supervised training data.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


PACK_SCHEMA_VERSION = "route_block_review_label_pack.v1"


def build_route_block_review_label_pack(
    *,
    inputs: list[Path],
    output_jsonl: Path,
    report_json: Path,
    dataset: str = "route_block_review_labels",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    input_summaries = []
    for path in inputs:
        source_rows = _read_jsonl(path)
        converted = [_convert_row(row, source_path=path, dataset=dataset) for row in source_rows]
        converted = [row for row in converted if row is not None]
        rows.extend(converted)
        input_summaries.append({"path": str(path), "source_rows": len(source_rows), "accepted_rows": len(converted)})
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, rows)
    report = _report(
        rows=rows,
        inputs=inputs,
        input_summaries=input_summaries,
        output_jsonl=output_jsonl,
        report_json=report_json,
        dataset=dataset,
    )
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _convert_row(row: dict[str, Any], *, source_path: Path, dataset: str) -> dict[str, Any] | None:
    review = row.get("expert_review") if isinstance(row.get("expert_review"), dict) else {}
    if not review:
        return None
    priority = _norm(review.get("priority"))
    route_plausible = _norm(review.get("route_plausible"))
    block_transform_correct = _norm(review.get("block_transform_correct"))
    support_relevant = _norm(review.get("support_precedent_relevant"))
    cascade_coherent = _norm(review.get("cascade_coherent"))
    if priority in {"", "unreviewed"}:
        return None
    risk_tags = [str(tag) for tag in review.get("risk_tags") or []]
    rationale = str(review.get("rationale") or "")
    placeholder_review = _is_placeholder_review(
        route_plausible=route_plausible,
        block_transform_correct=block_transform_correct,
        support_relevant=support_relevant,
        cascade_coherent=cascade_coherent,
        risk_tags=risk_tags,
        rationale=rationale,
    )
    labels = {
        "route_plausible_yes": route_plausible == "yes",
        "block_transform_correct_yes": block_transform_correct == "yes",
        "support_precedent_relevant_yes": support_relevant == "yes",
        "cascade_coherent_yes": cascade_coherent == "yes",
        "priority_high_or_medium": priority in {"high", "medium"},
        "priority_reject": priority == "reject",
        "placeholder_review": placeholder_review,
        "usable_positive": (
            not placeholder_review
            and
            priority in {"high", "medium"}
            and route_plausible == "yes"
            and block_transform_correct == "yes"
            and support_relevant == "yes"
            and cascade_coherent == "yes"
        ),
        "usable_negative": (
            not placeholder_review
            and (
                priority == "reject"
                or route_plausible == "no"
                or block_transform_correct == "no"
                or support_relevant == "no"
                or cascade_coherent == "no"
            )
        ),
    }
    route_block = row.get("route_block") if isinstance(row.get("route_block"), dict) else {}
    transform_sanity = row.get("transform_sanity") if isinstance(row.get("transform_sanity"), dict) else {}
    diagnostic_scores = row.get("diagnostic_scores") if isinstance(row.get("diagnostic_scores"), dict) else {}
    diagnostic_labels = row.get("diagnostic_labels") if isinstance(row.get("diagnostic_labels"), dict) else {}
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "dataset": dataset,
        "source_path": str(source_path),
        "review_source": _review_source(source_path),
        "review_id": row.get("review_id"),
        "target_id": row.get("target_id"),
        "target_smiles": row.get("target_smiles"),
        "route_id": row.get("route_id"),
        "value_split": row.get("value_split"),
        "source_pool": str(row.get("source_pool") or ""),
        "native_rank": int(row.get("native_rank") or 0),
        "stock_closed": bool(row.get("stock_closed")),
        "evidence_class": row.get("evidence_class"),
        "route_block": {
            "route_block_index": route_block.get("route_block_index"),
            "transform_pair": route_block.get("transform_pair"),
            "upstream_transform": route_block.get("upstream_transform"),
            "downstream_transform": route_block.get("downstream_transform"),
            "upstream_rxn": route_block.get("upstream_rxn"),
            "downstream_rxn": route_block.get("downstream_rxn"),
            "any_analog_supported": bool(route_block.get("any_analog_supported")),
            "same_pair_analog_supported": bool(route_block.get("same_pair_analog_supported")),
        },
        "review_labels": labels,
        "review_fields": {
            "priority": priority,
            "route_plausible": route_plausible,
            "block_transform_correct": block_transform_correct,
            "support_precedent_relevant": support_relevant,
            "cascade_coherent": cascade_coherent,
            "risk_tags": risk_tags,
            "rationale": rationale,
        },
        "diagnostic_features": {
            "best_any_block_min_sim": _float(diagnostic_scores.get("best_any_block_min_sim")),
            "best_same_pair_block_min_sim": _float(diagnostic_scores.get("best_same_pair_block_min_sim")),
            "has_any_analog_block": float(bool(diagnostic_labels.get("has_any_analog_block"))),
            "has_same_pair_analog_block": float(bool(diagnostic_labels.get("has_same_pair_analog_block"))),
            "has_observed_pair_block": float(bool(diagnostic_labels.get("has_observed_pair_block"))),
            "block_has_label_mismatch": float(bool(transform_sanity.get("block_has_label_mismatch"))),
            "block_label_mismatch_count": float(transform_sanity.get("block_label_mismatch_count") or 0),
            "native_rank": float(row.get("native_rank") or 0),
            "stock_closed": float(bool(row.get("stock_closed"))),
        },
        "training_contract": {
            "use": "calibration_only_until_sufficient_expert_labels",
            "not_route_preference_label": True,
            "self_review_requires_external_validation": _review_source(source_path) == "self_review",
        },
    }


def _report(
    *,
    rows: list[dict[str, Any]],
    inputs: list[Path],
    input_summaries: list[dict[str, Any]],
    output_jsonl: Path,
    report_json: Path,
    dataset: str,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    value_split_counts: Counter[str] = Counter()
    for row in rows:
        source_counts[str(row.get("review_source") or "")] += 1
        class_counts[str(row.get("evidence_class") or "")] += 1
        value_split_counts[str(row.get("value_split") or "")] += 1
        fields = row.get("review_fields") or {}
        priority_counts[str(fields.get("priority") or "")] += 1
        for tag in fields.get("risk_tags") or []:
            risk_counts[str(tag)] += 1
        for key, value in (row.get("review_labels") or {}).items():
            if value:
                label_counts[str(key)] += 1
    usable_positive = int(label_counts.get("usable_positive") or 0)
    usable_negative = int(label_counts.get("usable_negative") or 0)
    decision = _decision(
        rows=rows,
        source_counts=source_counts,
        usable_positive=usable_positive,
        usable_negative=usable_negative,
    )
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset,
        "inputs": [str(path) for path in inputs],
        "outputs": {"pack": str(output_jsonl), "report": str(report_json)},
        "input_summaries": input_summaries,
        "counts": {
            "rows": len(rows),
            "targets": len({row.get("target_smiles") for row in rows}),
            "usable_positive_rows": usable_positive,
            "usable_negative_rows": usable_negative,
        },
        "review_source_counts": dict(sorted(source_counts.items())),
        "evidence_class_counts": dict(sorted(class_counts.items())),
        "value_split_counts": dict(sorted(value_split_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "risk_tag_counts": dict(sorted(risk_counts.items())),
        "label_positive_counts": dict(sorted(label_counts.items())),
        "decision": decision,
    }


def _review_source(path: Path) -> str:
    text = str(path).lower()
    if "self_review" in text:
        return "self_review"
    if "mock" in text:
        return "mock"
    if "human" in text or "csv" in text:
        return "human_csv"
    return "unknown"


def _decision(
    *,
    rows: list[dict[str, Any]],
    source_counts: Counter[str],
    usable_positive: int,
    usable_negative: int,
) -> dict[str, Any]:
    active_sources = {source for source, count in source_counts.items() if count}
    if not rows:
        reason = "no reviewed rows were accepted"
        next_step = "fill expert review rows before attempting route/block label merge"
    elif active_sources and active_sources <= {"self_review"}:
        reason = "review labels are self-review dominated; use for calibration and label-design only"
        next_step = "expand calibrated route/block review labels before treating review judgments as supervised data"
    elif usable_positive == 0 or usable_negative == 0:
        reason = "review labels do not yet contain both usable positives and usable negatives"
        next_step = "collect additional reviewed rows or run the 300-row fallback before training"
    else:
        reason = "review label pack can be merged into the route/block value pack for expert-training gate evaluation"
        next_step = "run merge_route_block_review_labels and require ready_for_expert_training=true before training"
    return {
        "sufficient_for_main_training": False,
        "ready_for_route_block_merge_evaluation": bool(rows) and usable_positive > 0 and usable_negative > 0,
        "route_block_merge_required": True,
        "reason": reason,
        "minimum_next_step": next_step,
    }


def _is_placeholder_review(
    *,
    route_plausible: str,
    block_transform_correct: str,
    support_relevant: str,
    cascade_coherent: str,
    risk_tags: list[str],
    rationale: str,
) -> bool:
    text = rationale.lower()
    if "dry-run placeholder" in text or "no chemistry judgment" in text:
        return True
    return (
        route_plausible == "unclear"
        and block_transform_correct == "unclear"
        and support_relevant == "unclear"
        and cascade_coherent == "unclear"
        and set(risk_tags) <= {"other"}
    )


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a route/block review label calibration pack")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--dataset", default="route_block_review_labels")
    args = ap.parse_args()
    report = build_route_block_review_label_pack(
        inputs=[Path(item) for item in args.input],
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report),
        dataset=args.dataset,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
