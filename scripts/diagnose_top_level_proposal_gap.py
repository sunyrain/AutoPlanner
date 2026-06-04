#!/usr/bin/env python
"""Diagnose where top-level proposal generation fails against benchmark GT."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "top_level_proposal_gap_diagnostic.v1"


def diagnose_gap(
    *,
    coverage_audit: Path,
    proposal_audit: Path,
    output_json: Path,
    output_md: Path | None = None,
) -> dict[str, Any]:
    coverage = json.loads(Path(coverage_audit).read_text(encoding="utf-8"))
    proposals = json.loads(Path(proposal_audit).read_text(encoding="utf-8"))
    coverage_by_target = {row.get("target_smiles"): row for row in coverage.get("targets") or []}
    rows = [
        _diagnose_target(proposal_row, coverage_by_target.get(proposal_row.get("target_smiles")) or {})
        for proposal_row in proposals.get("targets") or []
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "coverage_audit": str(coverage_audit),
        "proposal_audit": str(proposal_audit),
        "summary": _summary(rows),
        "targets": rows,
        "decision": _decision(rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _diagnose_target(proposal_row: dict[str, Any], coverage_row: dict[str, Any]) -> dict[str, Any]:
    reactions = coverage_row.get("target_step_reactions") or []
    best_pair = max((_as_float((item.get("nearest_pair") or {}).get("combined_similarity")) for item in reactions), default=0.0)
    best_product = max((_as_float((item.get("nearest_product") or {}).get("product_similarity")) for item in reactions), default=0.0)
    exact_reaction_in_corpus = any(bool(item.get("exact_reaction_in_corpus")) for item in reactions)
    exact_product_in_corpus = any(bool(item.get("exact_product_in_corpus")) for item in reactions)
    exact_reactant_side_any_product = any(bool(item.get("exact_reactant_side_any_product")) for item in reactions)
    proposal_exact = bool(proposal_row.get("target_step_gt_reaction_hit") or proposal_row.get("exact_gt_reaction_hit"))
    proposal_reactant = bool(proposal_row.get("gt_reactant_hit"))
    status = _status(
        has_gt_step=bool(reactions),
        exact_reaction_in_corpus=exact_reaction_in_corpus,
        exact_product_in_corpus=exact_product_in_corpus,
        exact_reactant_side_any_product=exact_reactant_side_any_product,
        best_pair=best_pair,
        best_product=best_product,
        proposals_returned=int(proposal_row.get("returned") or 0),
        proposal_exact=proposal_exact,
        proposal_reactant=proposal_reactant,
    )
    return {
        "target_smiles": proposal_row.get("target_smiles"),
        "route_domain": proposal_row.get("route_domain") or coverage_row.get("route_domain"),
        "gt_steps": len(reactions),
        "coverage_label": coverage_row.get("target_coverage_label") or "missing_coverage_row",
        "exact_reaction_in_corpus": exact_reaction_in_corpus,
        "exact_product_in_corpus": exact_product_in_corpus,
        "exact_reactant_side_any_product": exact_reactant_side_any_product,
        "best_nearest_product_similarity": round(best_product, 6),
        "best_nearest_pair_similarity": round(best_pair, 6),
        "proposals_returned": int(proposal_row.get("returned") or 0),
        "proposal_exact_reaction_hit": proposal_exact,
        "proposal_reactant_hit": proposal_reactant,
        "proposal_reactant_best_rank": proposal_row.get("gt_reactant_best_rank"),
        "gap_status": status,
        "recommended_next_data_action": _recommended_action(status),
    }


def _status(
    *,
    has_gt_step: bool,
    exact_reaction_in_corpus: bool,
    exact_product_in_corpus: bool,
    exact_reactant_side_any_product: bool,
    best_pair: float,
    best_product: float,
    proposals_returned: int,
    proposal_exact: bool,
    proposal_reactant: bool,
) -> str:
    if not has_gt_step:
        return "no_target_gt_step"
    if proposal_exact:
        return "proposal_exact_reaction_covered"
    if proposal_reactant:
        return "proposal_partial_reactant_hit"
    if proposals_returned <= 0:
        return "generator_no_output"
    if exact_reaction_in_corpus:
        return "generator_missed_exact_corpus_reaction"
    if exact_reactant_side_any_product:
        return "generator_missed_known_reactant_side"
    if exact_product_in_corpus:
        return "product_seen_reactants_missing"
    if best_pair >= 0.70:
        return "near_pair_seen_but_not_generated"
    if best_product >= 0.70:
        return "near_product_seen_reactants_missing"
    return "target_domain_gap"


def _recommended_action(status: str) -> str:
    return {
        "no_target_gt_step": "Exclude from proposal-generation supervision/audit or add benchmark GT top-level step.",
        "proposal_exact_reaction_covered": "Candidate is already generated; route search/ranking can use it.",
        "proposal_partial_reactant_hit": "Add reaction-completion objective or hard positives that keep the hit precursor and complete missing reactants.",
        "generator_no_output": "Fix provider/runtime coverage before changing training data.",
        "generator_missed_exact_corpus_reaction": "Use retrieval-distilled or target-weighted SFT to force known exact corpus positives into top-k.",
        "generator_missed_known_reactant_side": "Train reactant-set reconstruction/reranking over known reactant sides.",
        "product_seen_reactants_missing": "Add route-targeted positives for the same product with benchmark-compatible reactant sides.",
        "near_pair_seen_but_not_generated": "Upweight near-pair positives and add hard negatives around the nearest pair.",
        "near_product_seen_reactants_missing": "Build synthetic/route-targeted reactant sides for the product family.",
        "target_domain_gap": "Expand corpus with route-targeted or synthetic cascade positives; external single-step data is insufficient.",
    }[status]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row.get("gap_status") for row in rows)
    return {
        "n_targets": len(rows),
        "status_counts": dict(counts),
        "proposal_exact_reaction_hit": sum(1 for row in rows if row.get("proposal_exact_reaction_hit")),
        "proposal_reactant_hit": sum(1 for row in rows if row.get("proposal_reactant_hit")),
        "exact_reaction_in_corpus": sum(1 for row in rows if row.get("exact_reaction_in_corpus")),
        "exact_product_in_corpus": sum(1 for row in rows if row.get("exact_product_in_corpus")),
        "target_domain_gap": counts.get("target_domain_gap", 0),
    }


def _decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(row.get("gap_status") for row in rows)
    if counts.get("proposal_exact_reaction_covered", 0):
        return {
            "status": "route_level_ab_candidate",
            "reason": "At least one target has an exact proposal hit; route search/ranking should be evaluated.",
        }
    if counts.get("proposal_partial_reactant_hit", 0):
        return {
            "status": "partial_candidate_pool_gain",
            "reason": "The model can recover some GT reactants but not complete reactions; improve reaction-completion/proposal objective before promotion.",
        }
    if counts.get("generator_missed_exact_corpus_reaction", 0):
        return {
            "status": "generator_training_miss",
            "reason": "Some exact GT reactions exist in the corpus but are not generated; focus on target-weighted SFT/retrieval distillation.",
        }
    return {
        "status": "data_coverage_gap",
        "reason": "Proposal failures are dominated by missing or only-near training coverage; expand route-targeted/synthetic positives.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Top-level Proposal Gap Diagnostic",
        "",
        f"created_at: `{payload['created_at']}`",
        "",
        "## Decision",
        "",
        f"- status: `{payload['decision']['status']}`",
        f"- reason: {payload['decision']['reason']}",
        "",
        "## Summary",
        "",
        f"- n_targets: {summary['n_targets']}",
        f"- proposal_exact_reaction_hit: {summary['proposal_exact_reaction_hit']}",
        f"- proposal_reactant_hit: {summary['proposal_reactant_hit']}",
        f"- exact_reaction_in_corpus: {summary['exact_reaction_in_corpus']}",
        f"- exact_product_in_corpus: {summary['exact_product_in_corpus']}",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((summary.get("status_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Targets",
        "",
        "| target | coverage | proposal status | best pair | returned | action |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    for row in payload["targets"]:
        lines.append(
            "| `{target}` | `{coverage}` | `{status}` | {pair:.3f} | {returned} | {action} |".format(
                target=row.get("target_smiles"),
                coverage=row.get("coverage_label"),
                status=row.get("gap_status"),
                pair=float(row.get("best_nearest_pair_similarity") or 0.0),
                returned=row.get("proposals_returned"),
                action=row.get("recommended_next_data_action"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage-audit", type=Path, required=True)
    ap.add_argument("--proposal-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    args = ap.parse_args()
    payload = diagnose_gap(
        coverage_audit=args.coverage_audit,
        proposal_audit=args.proposal_audit,
        output_json=args.output,
        output_md=args.markdown_output,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
