"""Export a flat route-pool evidence review worklist.

This merges the sampled route-pool review batch with optional transform sanity
diagnostics into reviewer-friendly JSONL/CSV rows.  It does not create labels
and must not be treated as training data.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_review_worklist.v1"


def export_route_pool_review_worklist(
    *,
    review_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    output_csv: Path | None = None,
    transform_sanity_json: Path | None = None,
) -> dict[str, Any]:
    reviews = _read_jsonl(review_jsonl)
    sanity_by_id = _load_sanity(transform_sanity_json)
    rows = [_work_row(row, sanity_by_id.get(str(row.get("review_id") or ""))) for row in reviews]
    rows.sort(
        key=lambda row: (
            -int(bool(row.get("transform_label_warning"))),
            str(row.get("evidence_class") or ""),
            str(row.get("source_pool") or ""),
            str(row.get("review_id") or ""),
        )
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if output_csv is not None:
        _write_csv(rows, output_csv)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "review_jsonl": str(review_jsonl),
            "transform_sanity_json": str(transform_sanity_json) if transform_sanity_json else None,
            "output_jsonl": str(output_jsonl),
            "output_csv": str(output_csv) if output_csv else None,
            "report_json": str(report_json),
        },
        "summary": {
            "rows": len(rows),
            "classes": dict(Counter(str(row.get("evidence_class") or "") for row in rows)),
            "pools": dict(Counter(str(row.get("source_pool") or "") for row in rows)),
            "rows_with_transform_sanity": sum(1 for row in rows if row.get("has_transform_sanity")),
            "rows_with_transform_label_warning": sum(1 for row in rows if row.get("transform_label_warning")),
            "warning_reasons": dict(Counter(reason for row in rows for reason in row.get("transform_label_warning_reasons") or [])),
        },
        "contract": {
            "review_worklist_only": True,
            "not_training_labels": True,
            "transform_sanity_is_heuristic_triage": True,
        },
        "examples": rows[:5],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _work_row(row: dict[str, Any], sanity: dict[str, Any] | None) -> dict[str, Any]:
    block = row.get("review_block") or {}
    sanity = sanity or {}
    upstream = sanity.get("upstream") or {}
    downstream = sanity.get("downstream") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": row.get("review_id"),
        "source_pool": row.get("source_pool"),
        "evidence_class": row.get("evidence_class"),
        "target_smiles": row.get("target_smiles"),
        "route_id": row.get("route_id"),
        "native_rank": row.get("native_rank"),
        "stock_closed": row.get("stock_closed"),
        "n_steps": row.get("n_steps"),
        "route_block_index": block.get("route_block_index"),
        "transform_pair": block.get("transform_pair"),
        "upstream_transform": block.get("upstream_transform"),
        "downstream_transform": block.get("downstream_transform"),
        "upstream_rxn": block.get("upstream_rxn"),
        "downstream_rxn": block.get("downstream_rxn"),
        "upstream_main_reactant": block.get("upstream_main_reactant"),
        "upstream_product": block.get("upstream_product"),
        "downstream_main_reactant": block.get("downstream_main_reactant"),
        "downstream_product": block.get("downstream_product"),
        "pair_observed_in_evidence": block.get("pair_observed_in_evidence"),
        "any_analog_supported": block.get("any_analog_supported"),
        "same_pair_analog_supported": block.get("same_pair_analog_supported"),
        "best_any_block_min_sim": block.get("best_any_block_min_sim"),
        "best_same_pair_block_min_sim": block.get("best_same_pair_block_min_sim"),
        "support_any_doi": (row.get("support_any") or {}).get("doi"),
        "support_any_transform_pair": (row.get("support_any") or {}).get("transform_pair"),
        "support_same_pair_doi": (row.get("support_same_pair") or {}).get("doi"),
        "support_same_pair_transform_pair": (row.get("support_same_pair") or {}).get("transform_pair"),
        "has_transform_sanity": bool(sanity),
        "transform_label_warning": bool(sanity.get("block_has_label_mismatch")),
        "transform_label_warning_count": int(sanity.get("block_label_mismatch_count") or 0),
        "transform_label_warning_reasons": sanity.get("block_mismatch_reasons") or [],
        "upstream_inferred_classes": upstream.get("inferred_classes") or [],
        "downstream_inferred_classes": downstream.get("inferred_classes") or [],
        "upstream_label_mismatch_reasons": upstream.get("mismatch_reasons") or [],
        "downstream_label_mismatch_reasons": downstream.get("mismatch_reasons") or [],
        "expert_route_plausible": row.get("expert_route_plausible"),
        "expert_block_transform_correct": row.get("expert_block_transform_correct"),
        "expert_support_precedent_relevant": row.get("expert_support_precedent_relevant"),
        "expert_cascade_coherent": row.get("expert_cascade_coherent"),
        "expert_priority": row.get("expert_priority"),
        "expert_reject_reason": row.get("expert_reject_reason"),
        "expert_risk_tags": row.get("expert_risk_tags"),
        "expert_comments": row.get("expert_comments"),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "source_pool",
        "evidence_class",
        "target_smiles",
        "route_id",
        "native_rank",
        "stock_closed",
        "n_steps",
        "route_block_index",
        "transform_pair",
        "upstream_transform",
        "downstream_transform",
        "transform_label_warning",
        "transform_label_warning_count",
        "transform_label_warning_reasons",
        "upstream_inferred_classes",
        "downstream_inferred_classes",
        "upstream_rxn",
        "downstream_rxn",
        "best_any_block_min_sim",
        "best_same_pair_block_min_sim",
        "support_any_doi",
        "support_any_transform_pair",
        "support_same_pair_doi",
        "support_same_pair_transform_pair",
        "expert_route_plausible",
        "expert_block_transform_correct",
        "expert_support_precedent_relevant",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_risk_tags",
        "expert_comments",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in (
                "transform_label_warning_reasons",
                "upstream_inferred_classes",
                "downstream_inferred_classes",
            ):
                flat[key] = ";".join(str(item) for item in flat.get(key) or [])
            writer.writerow({key: flat.get(key) for key in fieldnames})


def _load_sanity(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else []
    out = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get("review_id"):
            out[str(row.get("review_id"))] = row
    return out


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
        "# Route Pool Evidence Review Worklist",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "This is a reviewer worklist only.  It contains diagnostic triage fields, not training labels.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export flat route-pool review worklist")
    parser.add_argument("--review-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--transform-sanity-json")
    args = parser.parse_args()
    report = export_route_pool_review_worklist(
        review_jsonl=Path(args.review_jsonl),
        transform_sanity_json=Path(args.transform_sanity_json) if args.transform_sanity_json else None,
        output_jsonl=Path(args.output_jsonl),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        report_json=Path(args.report_json),
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
