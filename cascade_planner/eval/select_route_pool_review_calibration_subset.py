"""Select a balanced calibration subset from the route-pool review worklist.

The full worklist is small, but calibration should not be dominated by one
evidence class, source pool, or transform-label warning mode.  This selector
creates a deterministic reviewer subset and preserves all empty expert fields.
It does not create labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_review_calibration_subset.v1"


def select_route_pool_review_calibration_subset(
    *,
    worklist_jsonl: Path,
    output_jsonl: Path,
    output_csv: Path,
    report_json: Path,
    size: int = 36,
) -> dict[str, Any]:
    rows = _read_jsonl(worklist_jsonl)
    selected = _select_balanced(rows, size=max(0, int(size)))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_csv(selected, output_csv)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "worklist_jsonl": str(worklist_jsonl),
            "output_jsonl": str(output_jsonl),
            "output_csv": str(output_csv),
            "report_json": str(report_json),
            "requested_size": size,
        },
        "summary": {
            "source_rows": len(rows),
            "selected_rows": len(selected),
            "classes": dict(Counter(str(row.get("evidence_class") or "") for row in selected)),
            "pools": dict(Counter(str(row.get("source_pool") or "") for row in selected)),
            "transform_label_warning": dict(Counter(str(bool(row.get("transform_label_warning"))) for row in selected)),
            "class_warning": {
                _pair_key(key): count
                for key, count in Counter((str(row.get("evidence_class") or ""), str(bool(row.get("transform_label_warning")))) for row in selected).items()
            },
            "pool_warning": {
                _pair_key(key): count
                for key, count in Counter((str(row.get("source_pool") or ""), str(bool(row.get("transform_label_warning")))) for row in selected).items()
            },
        },
        "contract": {
            "review_subset_only": True,
            "not_training_labels": True,
            "deterministic_selection": True,
            "expert_fields_preserved_blank": True,
        },
        "examples": selected[:5],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _select_balanced(rows: list[dict[str, Any]], *, size: int) -> list[dict[str, Any]]:
    if size <= 0 or not rows:
        return []
    pool = sorted(rows, key=_sort_key)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # Always preserve rare same-pair analog examples if present.
    for row in pool:
        if row.get("evidence_class") == "same_pair_analog_supported":
            _append(row, selected, selected_ids, size=size)

    # Cover each class x warning bucket before optimizing broader balance.
    buckets = sorted(
        {(str(row.get("evidence_class") or ""), bool(row.get("transform_label_warning"))) for row in pool},
        key=lambda key: (key[0], str(key[1])),
    )
    for bucket in buckets:
        candidates = [row for row in pool if (str(row.get("evidence_class") or ""), bool(row.get("transform_label_warning"))) == bucket]
        if candidates:
            _append(candidates[0], selected, selected_ids, size=size)

    # Cover each source pool x warning bucket.
    pool_buckets = sorted(
        {(str(row.get("source_pool") or ""), bool(row.get("transform_label_warning"))) for row in pool},
        key=lambda key: (key[0], str(key[1])),
    )
    for bucket in pool_buckets:
        candidates = [row for row in pool if (str(row.get("source_pool") or ""), bool(row.get("transform_label_warning"))) == bucket]
        if candidates:
            _append(candidates[0], selected, selected_ids, size=size)

    while len(selected) < min(size, len(pool)):
        candidates = [row for row in pool if str(row.get("review_id") or "") not in selected_ids]
        if not candidates:
            break
        best = min(candidates, key=lambda row: _balance_cost(selected, row))
        _append(best, selected, selected_ids, size=size)

    return sorted(selected, key=_sort_key)


def _append(row: dict[str, Any], selected: list[dict[str, Any]], selected_ids: set[str], *, size: int) -> None:
    if len(selected) >= size:
        return
    rid = str(row.get("review_id") or "")
    if rid in selected_ids:
        return
    selected.append(dict(row))
    selected_ids.add(rid)


def _balance_cost(selected: list[dict[str, Any]], candidate: dict[str, Any]) -> tuple[int, int, int, str]:
    class_counts = Counter(str(row.get("evidence_class") or "") for row in selected)
    pool_counts = Counter(str(row.get("source_pool") or "") for row in selected)
    warning_counts = Counter(str(bool(row.get("transform_label_warning"))) for row in selected)
    cls = str(candidate.get("evidence_class") or "")
    pool = str(candidate.get("source_pool") or "")
    warn = str(bool(candidate.get("transform_label_warning")))
    return (
        class_counts[cls],
        pool_counts[pool],
        warning_counts[warn],
        str(candidate.get("review_id") or ""),
    )


def _sort_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("evidence_class") or ""),
        str(row.get("source_pool") or ""),
        -int(bool(row.get("transform_label_warning"))),
        str(row.get("review_id") or ""),
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key, value in list(flat.items()):
                if isinstance(value, (list, dict)):
                    flat[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(flat)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _pair_key(key: tuple[str, str]) -> str:
    return "::".join(key)


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Route Pool Review Calibration Subset",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Contract", "", "This is a deterministic review subset, not training labels.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a balanced route-pool review calibration subset")
    parser.add_argument("--worklist-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--size", type=int, default=36)
    args = parser.parse_args()
    report = select_route_pool_review_calibration_subset(
        worklist_jsonl=Path(args.worklist_jsonl),
        output_jsonl=Path(args.output_jsonl),
        output_csv=Path(args.output_csv),
        report_json=Path(args.report_json),
        size=int(args.size),
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
