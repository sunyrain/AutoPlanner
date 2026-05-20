"""Merge route/block review labels back into route_block_value_pack_v1.

The review labels are not a route preference dataset by themselves.  This tool
only attaches externally reviewed labels to the matching route rows so the
existing pairwise value-model trainer can use explicit expert-review tasks.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_block_review_label_merge.v1"


def merge_route_block_review_labels(
    *,
    value_pack_jsonl: Path,
    review_label_pack_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    dataset: str = "route_block_value_with_review_labels",
    min_usable_positive: int = 30,
    min_usable_negative: int = 30,
) -> dict[str, Any]:
    value_rows = _read_jsonl(value_pack_jsonl)
    review_rows = _read_jsonl(review_label_pack_jsonl)
    reviews_by_route_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    review_rows_without_route_id = 0
    for row in review_rows:
        route_id = str(row.get("route_id") or "")
        if not route_id:
            review_rows_without_route_id += 1
            continue
        reviews_by_route_id[route_id].append(row)

    matched_route_ids: set[str] = set()
    out_rows = []
    for row in value_rows:
        route_id = str(row.get("route_id") or "")
        reviews = reviews_by_route_id.get(route_id, [])
        copied = dict(row)
        copied["dataset"] = dataset
        if reviews:
            matched_route_ids.add(route_id)
            _attach_review_labels(copied, reviews)
        out_rows.append(copied)

    unmatched_review_route_ids = sorted(set(reviews_by_route_id) - matched_route_ids)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, out_rows)
    report = _report(
        value_pack_jsonl=value_pack_jsonl,
        review_label_pack_jsonl=review_label_pack_jsonl,
        output_jsonl=output_jsonl,
        report_json=report_json,
        rows=out_rows,
        review_rows=review_rows,
        reviews_by_route_id=reviews_by_route_id,
        matched_route_ids=matched_route_ids,
        unmatched_review_route_ids=unmatched_review_route_ids,
        review_rows_without_route_id=review_rows_without_route_id,
        dataset=dataset,
        min_usable_positive=min_usable_positive,
        min_usable_negative=min_usable_negative,
    )
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _attach_review_labels(row: dict[str, Any], reviews: list[dict[str, Any]]) -> None:
    labels_list = [
        review.get("review_labels")
        for review in reviews
        if isinstance(review.get("review_labels"), dict)
    ]
    positives = [labels for labels in labels_list if bool(labels.get("usable_positive"))]
    negatives = [labels for labels in labels_list if bool(labels.get("usable_negative"))]
    placeholders = [labels for labels in labels_list if bool(labels.get("placeholder_review"))]
    conflict = bool(positives and negatives)
    positive = bool(positives) and not conflict
    negative = bool(negatives) and not conflict

    tasks = dict(row.get("weak_label_tasks") or {})
    tasks.update(
        {
            "expert_reviewed": True,
            "expert_review_positive": positive,
            "expert_review_negative": negative,
            "expert_review_conflict": conflict,
            "expert_review_placeholder": bool(placeholders) and not positives and not negatives,
            "expert_route_plausible": any(bool(labels.get("route_plausible_yes")) for labels in labels_list),
            "expert_block_transform_correct": any(
                bool(labels.get("block_transform_correct_yes")) for labels in labels_list
            ),
            "expert_support_precedent_relevant": any(
                bool(labels.get("support_precedent_relevant_yes")) for labels in labels_list
            ),
            "expert_cascade_coherent": any(bool(labels.get("cascade_coherent_yes")) for labels in labels_list),
            "expert_priority_reject": any(bool(labels.get("priority_reject")) for labels in labels_list),
        }
    )
    row["weak_label_tasks"] = tasks
    row["review_label_evidence"] = {
        "schema_version": SCHEMA_VERSION,
        "n_reviews": len(reviews),
        "review_ids": [review.get("review_id") for review in reviews],
        "review_sources": sorted({str(review.get("review_source") or "") for review in reviews}),
        "evidence_classes": sorted({str(review.get("evidence_class") or "") for review in reviews}),
        "usable_positive": positive,
        "usable_negative": negative,
        "conflict": conflict,
        "placeholder_only": bool(placeholders) and not positives and not negatives,
        "review_fields": [review.get("review_fields") for review in reviews if isinstance(review.get("review_fields"), dict)],
    }


def _report(
    *,
    value_pack_jsonl: Path,
    review_label_pack_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    reviews_by_route_id: dict[str, list[dict[str, Any]]],
    matched_route_ids: set[str],
    unmatched_review_route_ids: list[str],
    review_rows_without_route_id: int,
    dataset: str,
    min_usable_positive: int,
    min_usable_negative: int,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    matched_rows = 0
    for row in rows:
        split = str(row.get("split") or "")
        split_counts[split] += 1
        evidence = row.get("review_label_evidence") if isinstance(row.get("review_label_evidence"), dict) else {}
        if evidence:
            matched_rows += 1
        tasks = row.get("weak_label_tasks") if isinstance(row.get("weak_label_tasks"), dict) else {}
        for key in (
            "expert_reviewed",
            "expert_review_positive",
            "expert_review_negative",
            "expert_review_conflict",
            "expert_review_placeholder",
            "expert_route_plausible",
            "expert_block_transform_correct",
            "expert_support_precedent_relevant",
            "expert_cascade_coherent",
            "expert_priority_reject",
        ):
            if bool(tasks.get(key)):
                label_counts[key] += 1
                label_split_counts[split][key] += 1
    for row in review_rows:
        source_counts[str(row.get("review_source") or "")] += 1
        class_counts[str(row.get("evidence_class") or "")] += 1

    usable_positive = int(label_counts.get("expert_review_positive") or 0)
    usable_negative = int(label_counts.get("expert_review_negative") or 0)
    conflict = int(label_counts.get("expert_review_conflict") or 0)
    required_splits = ("train", "val", "test")
    split_ready = {
        split: (
            int(label_split_counts[split].get("expert_review_positive") or 0) > 0
            and int(label_split_counts[split].get("expert_review_negative") or 0) > 0
        )
        for split in required_splits
    }
    ready = (
        usable_positive >= int(min_usable_positive)
        and usable_negative >= int(min_usable_negative)
        and conflict == 0
        and all(split_ready.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset,
        "inputs": {
            "value_pack": str(value_pack_jsonl),
            "review_label_pack": str(review_label_pack_jsonl),
        },
        "outputs": {
            "pack": str(output_jsonl),
            "report": str(report_json),
        },
        "counts": {
            "value_rows": len(rows),
            "review_rows": len(review_rows),
            "review_route_ids": len(reviews_by_route_id),
            "matched_route_ids": len(matched_route_ids),
            "matched_value_rows": matched_rows,
            "unmatched_review_route_ids": len(unmatched_review_route_ids),
            "review_rows_without_route_id": review_rows_without_route_id,
            "usable_positive_rows": usable_positive,
            "usable_negative_rows": usable_negative,
            "conflict_rows": conflict,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "expert_task_positive_counts_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(label_split_counts.items())
        },
        "review_source_counts": dict(sorted(source_counts.items())),
        "evidence_class_counts": dict(sorted(class_counts.items())),
        "expert_task_positive_counts": dict(sorted(label_counts.items())),
        "examples": {
            "unmatched_review_route_ids": unmatched_review_route_ids[:10],
        },
        "training_contract": {
            "positive_task": "expert_review_positive",
            "negative_task": "expert_review_negative",
            "do_not_use_placeholder_reviews": True,
            "requires_explicit_review_labels": True,
        },
        "decision": {
            "ready_for_expert_training": ready,
            "reason": (
                "expert review labels are sufficient for a first training run"
                if ready
                else "not enough non-conflicting usable expert positives/negatives yet"
            ),
            "minimum_usable_positive": int(min_usable_positive),
            "minimum_usable_negative": int(min_usable_negative),
            "split_ready": split_ready,
            "required_splits": list(required_splits),
        },
    }


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
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-pack", required=True, type=Path)
    parser.add_argument("--review-label-pack", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dataset", default="route_block_value_with_review_labels")
    parser.add_argument("--min-usable-positive", type=int, default=30)
    parser.add_argument("--min-usable-negative", type=int, default=30)
    args = parser.parse_args()
    report = merge_route_block_review_labels(
        value_pack_jsonl=args.value_pack,
        review_label_pack_jsonl=args.review_label_pack,
        output_jsonl=args.output_jsonl,
        report_json=args.report,
        dataset=args.dataset,
        min_usable_positive=args.min_usable_positive,
        min_usable_negative=args.min_usable_negative,
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "decision": report["decision"],
                "output_jsonl": str(args.output_jsonl),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
