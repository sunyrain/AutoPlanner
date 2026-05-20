"""Validate human-filled route-pool review worklist CSV files.

The flat worklist is convenient for manual review, but filled CSV files need
the same validation contract as LLM JSON responses before they can be used for
calibration.  This importer accepts only complete enum-valued reviews, records
unreviewed rows separately, and writes labels compatible with
``summarize_route_pool_evidence_review_labels`` and
``calibrate_route_pool_evidence_review_signals``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_evidence_review_csv_results.v1"

YES_NO_UNCLEAR = {"yes", "no", "unclear"}
PRIORITIES = {"high", "medium", "low", "reject"}
RISK_TAGS = {
    "wrong_transform_label",
    "irrelevant_precedent",
    "not_cascade",
    "trivial_stock_closure",
    "missing_reaction_detail",
    "condition_incompatibility",
    "unsupported_selectivity",
    "atom_mapping_or_stoichiometry_issue",
    "other",
}

REVIEW_FIELDS = (
    "expert_route_plausible",
    "expert_block_transform_correct",
    "expert_support_precedent_relevant",
    "expert_cascade_coherent",
    "expert_priority",
)


def ingest_route_pool_evidence_review_csv(
    *,
    review_csv: Path,
    output_jsonl: Path,
    report_json: Path,
    invalid_jsonl: Path | None = None,
    unreviewed_jsonl: Path | None = None,
) -> dict[str, Any]:
    rows = _read_csv(review_csv)
    accepted = []
    invalid = []
    unreviewed = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        review_id = str(row.get("review_id") or "").strip()
        if not review_id:
            invalid.append({"line_index": index, "errors": ["missing_review_id"], "row": row})
            continue
        if review_id in seen:
            invalid.append({"line_index": index, "review_id": review_id, "errors": ["duplicate_review_id"], "row": row})
            continue
        seen.add(review_id)
        if _is_unreviewed(row):
            unreviewed.append({"line_index": index, "review_id": review_id, "row": row})
            continue
        errors = _validate_row(row)
        if errors:
            invalid.append({"line_index": index, "review_id": review_id, "errors": errors, "row": row})
            continue
        accepted.append(_label_row(row))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    invalid_path = invalid_jsonl or report_json.with_name(report_json.stem + "_invalid.jsonl")
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    with invalid_path.open("w", encoding="utf-8") as fh:
        for row in invalid:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    unreviewed_path = unreviewed_jsonl or report_json.with_name(report_json.stem + "_unreviewed.jsonl")
    unreviewed_path.parent.mkdir(parents=True, exist_ok=True)
    with unreviewed_path.open("w", encoding="utf-8") as fh:
        for row in unreviewed:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "review_csv": str(review_csv),
            "output_jsonl": str(output_jsonl),
            "invalid_jsonl": str(invalid_path),
            "unreviewed_jsonl": str(unreviewed_path),
            "report_json": str(report_json),
        },
        "summary": {
            "csv_rows": len(rows),
            "accepted_rows": len(accepted),
            "invalid_rows": len(invalid),
            "unreviewed_rows": len(unreviewed),
            "accepted_classes": dict(Counter(str(row.get("evidence_class") or "") for row in accepted)),
            "accepted_pools": dict(Counter(str(row.get("source_pool") or "") for row in accepted)),
            "value_split_counts": dict(Counter(str(row.get("value_split") or "") for row in accepted)),
            "priority_counts": dict(Counter(str((row.get("expert_review") or {}).get("priority") or "") for row in accepted)),
            "cascade_coherent_counts": dict(Counter(str((row.get("expert_review") or {}).get("cascade_coherent") or "") for row in accepted)),
            "risk_tag_counts": dict(Counter(tag for row in accepted for tag in ((row.get("expert_review") or {}).get("risk_tags") or []))),
        },
        "invalid_error_counts": dict(Counter(error for row in invalid for error in row.get("errors") or [])),
        "contract": {
            "valid_enums_enforced": True,
            "blank_rows_are_unreviewed_not_labels": True,
            "merged_rows_are_review_labels_not_training_preferences": True,
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _label_row(row: dict[str, str]) -> dict[str, Any]:
    review = {
        "route_plausible": _norm(row.get("expert_route_plausible")),
        "block_transform_correct": _norm(row.get("expert_block_transform_correct")),
        "support_precedent_relevant": _norm(row.get("expert_support_precedent_relevant")),
        "cascade_coherent": _norm(row.get("expert_cascade_coherent")),
        "priority": _norm(row.get("expert_priority")),
        "risk_tags": _risk_tags(row),
        "rationale": str(row.get("expert_comments") or row.get("expert_reject_reason") or "").strip(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": row.get("review_id"),
        "target_id": row.get("target_id"),
        "route_id": row.get("route_id"),
        "value_split": row.get("value_split"),
        "source_pool": row.get("source_pool"),
        "evidence_class": row.get("evidence_class"),
        "target_smiles": row.get("target_smiles"),
        "native_rank": _to_int(row.get("native_rank")),
        "n_steps": _to_int(row.get("n_steps")),
        "model_rank": _to_int(row.get("model_rank")),
        "retrieval_rank": _to_int(row.get("retrieval_rank")),
        "audit_rank": _to_int(row.get("audit_rank")),
        "stock_closed": _to_bool(row.get("stock_closed")),
        "route_block": {
            "route_block_index": _to_int(row.get("route_block_index")),
            "upstream_rxn": row.get("upstream_rxn"),
            "downstream_rxn": row.get("downstream_rxn"),
            "upstream_transform": row.get("upstream_transform"),
            "downstream_transform": row.get("downstream_transform"),
            "transform_pair": row.get("transform_pair"),
            "any_analog_supported": _to_bool(row.get("any_analog_supported")),
            "same_pair_analog_supported": _to_bool(row.get("same_pair_analog_supported")),
        },
        "diagnostic_labels": {
            "has_observed_pair_block": _to_bool(row.get("pair_observed_in_evidence")),
            "has_any_analog_block": _to_bool(row.get("any_analog_supported")),
            "has_same_pair_analog_block": _to_bool(row.get("same_pair_analog_supported")),
        },
        "diagnostic_scores": {
            "best_any_block_min_sim": _to_float(row.get("best_any_block_min_sim")),
            "best_same_pair_block_min_sim": _to_float(row.get("best_same_pair_block_min_sim")),
        },
        "transform_sanity": {
            "heuristic_only": True,
            "block_has_label_mismatch": _to_bool(row.get("transform_label_warning")),
            "block_label_mismatch_count": _to_int(row.get("transform_label_warning_count")) or 0,
            "block_mismatch_reasons": _split_tokens(row.get("transform_label_warning_reasons")),
            "upstream": {"inferred_classes": _split_tokens(row.get("upstream_inferred_classes"))},
            "downstream": {"inferred_classes": _split_tokens(row.get("downstream_inferred_classes"))},
        },
        "support_any": {
            "doi": row.get("support_any_doi"),
            "transform_pair": row.get("support_any_transform_pair"),
        },
        "support_same_pair": {
            "doi": row.get("support_same_pair_doi"),
            "transform_pair": row.get("support_same_pair_transform_pair"),
        },
        "expert_review": review,
    }


def _validate_row(row: dict[str, str]) -> list[str]:
    errors = []
    if not str(row.get("route_id") or "").strip():
        errors.append("missing_route_id")
    enum_map = {
        "expert_route_plausible": YES_NO_UNCLEAR,
        "expert_block_transform_correct": YES_NO_UNCLEAR,
        "expert_support_precedent_relevant": YES_NO_UNCLEAR,
        "expert_cascade_coherent": YES_NO_UNCLEAR,
        "expert_priority": PRIORITIES,
    }
    for field, allowed in enum_map.items():
        if _norm(row.get(field)) not in allowed:
            errors.append(f"invalid_{field}")
    tags = _risk_tags(row)
    unknown = [tag for tag in tags if tag not in RISK_TAGS]
    if unknown:
        errors.append("unknown_risk_tag")
    if _norm(row.get("expert_priority")) == "reject" and not str(row.get("expert_reject_reason") or row.get("expert_comments") or "").strip():
        errors.append("missing_reject_reason")
    if not str(row.get("expert_comments") or row.get("expert_reject_reason") or "").strip():
        errors.append("missing_rationale")
    return errors


def _risk_tags(row: dict[str, str]) -> list[str]:
    explicit = _split_tokens(row.get("expert_risk_tags"))
    if explicit:
        return explicit
    return [tag for tag in _split_tokens(row.get("expert_reject_reason")) if tag in RISK_TAGS]


def _is_unreviewed(row: dict[str, str]) -> bool:
    return not any(str(row.get(field) or "").strip() for field in REVIEW_FIELDS)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    return [token.strip() for token in re.split(r"[;,|]", str(value)) if token.strip()]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _to_bool(value: Any) -> bool | None:
    text = _norm(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Route Pool Evidence Human CSV Review Results",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Invalid Error Counts", "", "```json", json.dumps(report.get("invalid_error_counts") or {}, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and ingest human-filled route-pool review CSV")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--invalid-jsonl")
    parser.add_argument("--unreviewed-jsonl")
    args = parser.parse_args()
    report = ingest_route_pool_evidence_review_csv(
        review_csv=Path(args.review_csv),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        invalid_jsonl=Path(args.invalid_jsonl) if args.invalid_jsonl else None,
        unreviewed_jsonl=Path(args.unreviewed_jsonl) if args.unreviewed_jsonl else None,
    )
    print(json.dumps({"summary": report["summary"], "invalid_error_counts": report["invalid_error_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
