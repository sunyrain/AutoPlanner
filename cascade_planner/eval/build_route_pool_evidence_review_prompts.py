"""Build structured human/LLM review prompts from route-pool evidence cases."""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_evidence_review_prompts.v1"


EXPECTED_OUTPUT_SCHEMA = {
    "review_id": "string",
    "route_plausible": "yes|no|unclear",
    "block_transform_correct": "yes|no|unclear",
    "support_precedent_relevant": "yes|no|unclear",
    "cascade_coherent": "yes|no|unclear",
    "priority": "high|medium|low|reject",
    "risk_tags": [
        "wrong_transform_label",
        "irrelevant_precedent",
        "not_cascade",
        "trivial_stock_closure",
        "missing_reaction_detail",
        "condition_incompatibility",
        "unsupported_selectivity",
        "atom_mapping_or_stoichiometry_issue",
        "other",
    ],
    "rationale": "short technical explanation",
}


def build_route_pool_evidence_review_prompts(
    *,
    review_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    output_md: Path | None = None,
    max_rows: int | None = None,
    transform_sanity_json: Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(review_jsonl)
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    sanity_by_id = _load_transform_sanity_by_id(transform_sanity_json)
    prompt_rows = [_prompt_row(row, transform_sanity=sanity_by_id.get(str(row.get("review_id") or ""))) for row in rows]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in prompt_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "review_jsonl": str(review_jsonl),
            "output_jsonl": str(output_jsonl),
            "report_json": str(report_json),
            "output_md": str(output_md or report_json.with_suffix(".md")),
            "max_rows": max_rows,
            "transform_sanity_json": str(transform_sanity_json) if transform_sanity_json else None,
        },
        "summary": {
            "source_rows": len(_read_jsonl(review_jsonl)),
            "prompt_rows": len(prompt_rows),
            "classes": dict(Counter(str(row.get("evidence_class") or "") for row in prompt_rows)),
            "pools": dict(Counter(str(row.get("source_pool") or "") for row in prompt_rows)),
            "rows_with_transform_sanity": sum(1 for row in prompt_rows if row.get("transform_sanity")),
            "rows_with_transform_label_warning": sum(1 for row in prompt_rows if (row.get("transform_sanity") or {}).get("block_has_label_mismatch")),
        },
        "expected_output_schema": EXPECTED_OUTPUT_SCHEMA,
        "contract": {
            "diagnostic_labels_are_not_ground_truth": True,
            "transform_sanity_is_heuristic_triage": True,
            "transform_labels_are_not_supervision": True,
            "reviewer_must_return_json_only": True,
            "unclear_is_allowed": True,
        },
        "examples": prompt_rows[:3],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or report_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(report), encoding="utf-8")
    return report


def _prompt_row(row: dict[str, Any], *, transform_sanity: dict[str, Any] | None = None) -> dict[str, Any]:
    compact_sanity = _compact_transform_sanity(transform_sanity)
    prompt = _prompt_text(row, transform_sanity=compact_sanity)
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": row.get("review_id"),
        "source_pool": row.get("source_pool"),
        "source_value_pack": row.get("source_value_pack"),
        "evidence_class": row.get("evidence_class"),
        "target_id": row.get("target_id"),
        "target_smiles": row.get("target_smiles"),
        "route_id": row.get("route_id"),
        "native_rank": row.get("native_rank"),
        "native_score": row.get("native_score"),
        "n_steps": row.get("n_steps"),
        "stock_closed": row.get("stock_closed"),
        "route_block": row.get("review_block") or {},
        "diagnostic_labels": row.get("diagnostic_labels") or {},
        "diagnostic_scores": row.get("diagnostic_scores") or {},
        "transform_sanity": compact_sanity,
        "support_any": row.get("support_any") or {},
        "support_same_pair": row.get("support_same_pair") or {},
        "expected_output_schema": EXPECTED_OUTPUT_SCHEMA,
        "prompt": prompt,
    }


def _prompt_text(row: dict[str, Any], *, transform_sanity: dict[str, Any] | None = None) -> str:
    block = row.get("review_block") or {}
    transform_sanity = transform_sanity or {}
    payload = {
        "review_id": row.get("review_id"),
        "source_pool": row.get("source_pool"),
        "source_value_pack": row.get("source_value_pack"),
        "evidence_class": row.get("evidence_class"),
        "target_id": row.get("target_id"),
        "target_smiles": row.get("target_smiles"),
        "route_id": row.get("route_id"),
        "native_rank": row.get("native_rank"),
        "native_score": row.get("native_score"),
        "n_steps": row.get("n_steps"),
        "stock_closed": row.get("stock_closed"),
        "route_block_forward_order": {
            "upstream_rxn": block.get("upstream_rxn"),
            "downstream_rxn": block.get("downstream_rxn"),
            "upstream_product": block.get("upstream_product"),
            "downstream_product": block.get("downstream_product"),
            "upstream_main_reactant": block.get("upstream_main_reactant"),
            "downstream_main_reactant": block.get("downstream_main_reactant"),
            "upstream_transform": block.get("upstream_transform"),
            "downstream_transform": block.get("downstream_transform"),
            "transform_pair": block.get("transform_pair"),
        },
        "diagnostic_evidence": {
            "labels": row.get("diagnostic_labels") or {},
            "scores": row.get("diagnostic_scores") or {},
            "transform_sanity": transform_sanity,
            "support_any": row.get("support_any") or {},
            "support_same_pair": row.get("support_same_pair") or {},
        },
    }
    return textwrap.dedent(
        f"""
        You are reviewing a candidate ChemEnzy route block for cascade-planning evidence.

        Important rules:
        - The diagnostic evidence class is not ground truth.
        - Judge only the route block shown here, not the whole route.
        - Mark "unclear" when the structures/reactions are insufficient.
        - Do not reward a route merely because it is stock-closed.
        - Do not trust transform text labels blindly. If transform_sanity flags a mismatch,
          judge the shown reaction SMILES directly and use wrong_transform_label when appropriate.
        - A useful cascade block should have plausible chemistry, coherent adjacent transforms,
          and a support precedent that is actually relevant to the route block.
        - Return JSON only. Do not include prose outside the JSON object.

        Case:
        {json.dumps(payload, ensure_ascii=False, indent=2)}

        Return exactly this JSON shape:
        {json.dumps(EXPECTED_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}
        """
    ).strip()


def _load_transform_sanity_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else []
    out = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get("review_id"):
            out[str(row.get("review_id"))] = row
    return out


def _compact_transform_sanity(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    upstream = row.get("upstream") or {}
    downstream = row.get("downstream") or {}
    return {
        "heuristic_only": True,
        "block_has_label_mismatch": bool(row.get("block_has_label_mismatch")),
        "block_label_mismatch_count": int(row.get("block_label_mismatch_count") or 0),
        "block_mismatch_reasons": row.get("block_mismatch_reasons") or [],
        "upstream": {
            "label": upstream.get("label"),
            "inferred_classes": upstream.get("inferred_classes") or [],
            "mismatch_reasons": upstream.get("mismatch_reasons") or [],
        },
        "downstream": {
            "label": downstream.get("label"),
            "inferred_classes": downstream.get("inferred_classes") or [],
            "mismatch_reasons": downstream.get("mismatch_reasons") or [],
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


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Route Pool Evidence Review Prompts",
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
            "## Review Contract",
            "",
            "- Diagnostic labels are not ground truth.",
            "- Reviewers must return JSON only.",
            "- `unclear` is a valid answer when the chemistry cannot be judged from the provided block.",
            "",
            "## Expected Output Schema",
            "",
            "```json",
            json.dumps(EXPECTED_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build route-pool evidence review prompts")
    parser.add_argument("--review-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--transform-sanity-json")
    args = parser.parse_args()
    report = build_route_pool_evidence_review_prompts(
        review_jsonl=Path(args.review_jsonl),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        output_md=Path(args.output_md) if args.output_md else None,
        max_rows=args.max_rows,
        transform_sanity_json=Path(args.transform_sanity_json) if args.transform_sanity_json else None,
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl, "report_json": args.report_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
