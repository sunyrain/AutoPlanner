"""Validate and merge route-pool evidence review responses.

This closes the review loop created by
``build_route_pool_evidence_review_prompts``.  It accepts JSONL reviewer
responses, validates the fixed schema/enums, merges them back onto the prompt
cases, and writes a labeled JSONL plus an audit report.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_evidence_review_results.v1"

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


def ingest_route_pool_evidence_review_results(
    *,
    prompts_jsonl: Path,
    responses_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    invalid_jsonl: Path | None = None,
) -> dict[str, Any]:
    prompts = _read_jsonl(prompts_jsonl)
    prompt_by_id = {str(row.get("review_id") or ""): row for row in prompts if row.get("review_id")}
    raw_responses = _read_jsonl(responses_jsonl)
    accepted = []
    invalid = []
    seen = set()
    for index, raw in enumerate(raw_responses):
        parsed = _extract_response(raw)
        errors = _validate_response(parsed, prompt_by_id=prompt_by_id)
        review_id = str(parsed.get("review_id") or "")
        if review_id in seen:
            errors.append("duplicate_review_id")
        if errors:
            invalid.append({"line_index": index, "errors": errors, "raw": raw, "parsed": parsed})
            continue
        seen.add(review_id)
        accepted.append(_merged_row(prompt_by_id[review_id], parsed))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    invalid_path = invalid_jsonl or report_json.with_name(report_json.stem + "_invalid.jsonl")
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    with invalid_path.open("w", encoding="utf-8") as fh:
        for row in invalid:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "prompts_jsonl": str(prompts_jsonl),
            "responses_jsonl": str(responses_jsonl),
            "output_jsonl": str(output_jsonl),
            "invalid_jsonl": str(invalid_path),
            "report_json": str(report_json),
        },
        "summary": {
            "prompt_rows": len(prompts),
            "response_rows": len(raw_responses),
            "accepted_rows": len(accepted),
            "invalid_rows": len(invalid),
            "unreviewed_prompt_rows": len(set(prompt_by_id) - {row.get("review_id") for row in accepted}),
            "accepted_classes": dict(Counter(str(row.get("evidence_class") or "") for row in accepted)),
            "accepted_pools": dict(Counter(str(row.get("source_pool") or "") for row in accepted)),
            "priority_counts": dict(Counter(str((row.get("expert_review") or {}).get("priority") or "") for row in accepted)),
            "cascade_coherent_counts": dict(Counter(str((row.get("expert_review") or {}).get("cascade_coherent") or "") for row in accepted)),
            "risk_tag_counts": dict(Counter(tag for row in accepted for tag in ((row.get("expert_review") or {}).get("risk_tags") or []))),
        },
        "invalid_error_counts": dict(Counter(error for row in invalid for error in row.get("errors") or [])),
        "contract": {
            "valid_enums_enforced": True,
            "merged_rows_are_review_labels_not_training_preferences": True,
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _extract_response(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("response"), dict):
        return dict(raw["response"])
    if isinstance(raw.get("review"), dict):
        return dict(raw["review"])
    for key in ("output", "content", "text"):
        if isinstance(raw.get(key), str):
            parsed = _parse_json_from_text(raw[key])
            if isinstance(parsed, dict):
                return parsed
    return dict(raw)


def _parse_json_from_text(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _validate_response(row: dict[str, Any], *, prompt_by_id: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    if not isinstance(row, dict):
        return ["response_not_object"]
    review_id = str(row.get("review_id") or "")
    if not review_id:
        errors.append("missing_review_id")
    elif review_id not in prompt_by_id:
        errors.append("unknown_review_id")
    for key in ("route_plausible", "block_transform_correct", "support_precedent_relevant", "cascade_coherent"):
        value = str(row.get(key) or "").lower()
        if value not in YES_NO_UNCLEAR:
            errors.append(f"invalid_{key}")
    priority = str(row.get("priority") or "").lower()
    if priority not in PRIORITIES:
        errors.append("invalid_priority")
    tags = row.get("risk_tags")
    if not isinstance(tags, list):
        errors.append("invalid_risk_tags")
    else:
        for tag in tags:
            if str(tag) not in RISK_TAGS:
                errors.append("unknown_risk_tag")
                break
    rationale = str(row.get("rationale") or "").strip()
    if not rationale:
        errors.append("missing_rationale")
    return errors


def _merged_row(prompt: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    review = {
        "route_plausible": str(response.get("route_plausible") or "").lower(),
        "block_transform_correct": str(response.get("block_transform_correct") or "").lower(),
        "support_precedent_relevant": str(response.get("support_precedent_relevant") or "").lower(),
        "cascade_coherent": str(response.get("cascade_coherent") or "").lower(),
        "priority": str(response.get("priority") or "").lower(),
        "risk_tags": [str(tag) for tag in response.get("risk_tags") or []],
        "rationale": str(response.get("rationale") or "").strip(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": prompt.get("review_id"),
        "source_pool": prompt.get("source_pool"),
        "source_value_pack": prompt.get("source_value_pack"),
        "evidence_class": prompt.get("evidence_class"),
        "target_id": prompt.get("target_id"),
        "target_smiles": prompt.get("target_smiles"),
        "route_id": prompt.get("route_id"),
        "native_rank": prompt.get("native_rank"),
        "native_score": prompt.get("native_score"),
        "n_steps": prompt.get("n_steps"),
        "stock_closed": prompt.get("stock_closed"),
        "route_block": prompt.get("route_block") or {},
        "diagnostic_labels": prompt.get("diagnostic_labels") or {},
        "diagnostic_scores": prompt.get("diagnostic_scores") or {},
        "transform_sanity": prompt.get("transform_sanity") or {},
        "support_any": prompt.get("support_any") or {},
        "support_same_pair": prompt.get("support_same_pair") or {},
        "expert_review": review,
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


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Route Pool Evidence Review Results",
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
    parser = argparse.ArgumentParser(description="Validate and merge route-pool evidence review responses")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--invalid-jsonl")
    args = parser.parse_args()
    report = ingest_route_pool_evidence_review_results(
        prompts_jsonl=Path(args.prompts_jsonl),
        responses_jsonl=Path(args.responses_jsonl),
        output_jsonl=Path(args.output_jsonl),
        invalid_jsonl=Path(args.invalid_jsonl) if args.invalid_jsonl else None,
        report_json=Path(args.report_json),
    )
    print(json.dumps({"summary": report["summary"], "invalid_error_counts": report["invalid_error_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
