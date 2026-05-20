"""Select route-pool review prompts matching a calibration subset."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_review_prompt_subset.v1"


def select_route_pool_review_prompt_subset(
    *,
    prompts_jsonl: Path,
    subset_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    preserve_subset_order: bool = True,
) -> dict[str, Any]:
    prompts = _read_jsonl(prompts_jsonl)
    subset = _read_jsonl(subset_jsonl)
    subset_ids = [str(row.get("review_id") or "") for row in subset if row.get("review_id")]
    prompt_by_id = {str(row.get("review_id") or ""): row for row in prompts if row.get("review_id")}
    duplicates = [rid for rid, count in Counter(subset_ids).items() if count > 1]
    missing = [rid for rid in subset_ids if rid not in prompt_by_id]
    if preserve_subset_order:
        selected = [prompt_by_id[rid] for rid in subset_ids if rid in prompt_by_id]
    else:
        subset_id_set = set(subset_ids)
        selected = [row for row in prompts if str(row.get("review_id") or "") in subset_id_set]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "prompts_jsonl": str(prompts_jsonl),
            "subset_jsonl": str(subset_jsonl),
            "output_jsonl": str(output_jsonl),
            "report_json": str(report_json),
            "preserve_subset_order": preserve_subset_order,
        },
        "summary": {
            "prompt_rows": len(prompts),
            "subset_rows": len(subset),
            "subset_ids": len(subset_ids),
            "selected_prompts": len(selected),
            "missing_ids": len(missing),
            "duplicate_subset_ids": len(duplicates),
            "classes": dict(Counter(str(row.get("evidence_class") or "") for row in selected)),
            "pools": dict(Counter(str(row.get("source_pool") or "") for row in selected)),
            "rows_with_transform_sanity": sum(1 for row in selected if row.get("transform_sanity")),
            "rows_with_transform_label_warning": sum(1 for row in selected if (row.get("transform_sanity") or {}).get("block_has_label_mismatch")),
        },
        "missing_review_ids": missing,
        "duplicate_subset_review_ids": duplicates,
        "contract": {
            "review_prompt_subset_only": True,
            "not_training_labels": True,
            "fails_if_missing_or_duplicate": bool(missing or duplicates),
        },
        "examples": selected[:3],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Route Pool Review Prompt Subset",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Contract", "", "This is a prompt subset for review execution, not training labels.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select prompts matching a route-pool review calibration subset")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--subset-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--preserve-prompt-order", action="store_true")
    args = parser.parse_args()
    report = select_route_pool_review_prompt_subset(
        prompts_jsonl=Path(args.prompts_jsonl),
        subset_jsonl=Path(args.subset_jsonl),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        preserve_subset_order=not bool(args.preserve_prompt_order),
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl}, indent=2, ensure_ascii=False))
    if report["summary"]["missing_ids"] or report["summary"]["duplicate_subset_ids"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
