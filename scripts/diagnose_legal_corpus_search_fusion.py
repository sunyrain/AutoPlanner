#!/usr/bin/env python
"""Diagnose legal-corpus candidate hits lost during route-level search fusion."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "legal_corpus_search_fusion_diagnostic.v1"


def diagnose_search_fusion(
    *,
    proposal_audit: Path,
    route_run: Path,
    output_json: Path,
    output_md: Path | None = None,
) -> dict[str, Any]:
    proposal_payload = json.loads(Path(proposal_audit).read_text(encoding="utf-8"))
    route_payload = json.loads(Path(route_run).read_text(encoding="utf-8"))
    proposal_rows = _rows_by_occurrence(proposal_payload.get("targets") or [])
    route_rows = {key: (index, row) for key, index, row in _rows_by_occurrence(route_payload.get("targets") or [])}
    rows = []
    seen = set()
    for key, proposal_index, proposal_row in proposal_rows:
        route_index, route_row = route_rows.get(key, (None, {}))
        rows.append(_diagnose_row(key, proposal_index, route_index, proposal_row, route_row))
        seen.add(key)
    for key, (route_index, route_row) in route_rows.items():
        if key not in seen:
            rows.append(_diagnose_row(key, None, route_index, {}, route_row))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "proposal_audit": str(proposal_audit),
        "route_run": str(route_run),
        "route_metadata": {
            "cascade_search": (route_payload.get("metadata") or {}).get("cascade_search"),
            "summary": route_payload.get("summary"),
        },
        "summary": _summary(rows),
        "decision": _decision(rows),
        "targets": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _diagnose_row(
    occurrence_key: str,
    proposal_index: int | None,
    route_index: int | None,
    proposal_row: dict[str, Any],
    route_row: dict[str, Any],
) -> dict[str, Any]:
    recovery = route_row.get("recovery") or {}
    programs = (route_row.get("cascade_search") or {}).get("result_programs") or []
    audit_exact = bool(proposal_row.get("target_step_gt_reaction_hit") or proposal_row.get("exact_gt_reaction_hit"))
    audit_reactant = bool(proposal_row.get("gt_reactant_hit"))
    top_exact = bool(recovery.get("exact_reaction_in_route_pool"))
    top_reactant = bool(recovery.get("gt_reactant_in_route_pool"))
    result_exact = bool(recovery.get("result_exact_reaction_in_pool") or _first_program_rank(programs, "exact_reaction_hit_count"))
    result_reactant = bool(recovery.get("result_gt_reactant_in_pool") or _first_program_rank(programs, "gt_reactant_hit_count"))
    exact_result_rank = _first_program_rank(programs, "exact_reaction_hit_count")
    reactant_result_rank = _first_program_rank(programs, "gt_reactant_hit_count")
    status = _status(
        audit_exact=audit_exact,
        audit_reactant=audit_reactant,
        top_exact=top_exact,
        top_reactant=top_reactant,
        result_exact=result_exact,
        result_reactant=result_reactant,
    )
    return {
        "occurrence_key": occurrence_key,
        "proposal_index": proposal_index,
        "route_index": route_index,
        "target_smiles": proposal_row.get("target_smiles") or route_row.get("target_smiles"),
        "route_domain": proposal_row.get("route_domain") or route_row.get("route_domain"),
        "direct_audit": {
            "returned": int(proposal_row.get("returned") or 0),
            "exact_hit": audit_exact,
            "exact_best_rank": proposal_row.get("target_step_gt_reaction_best_rank")
            or proposal_row.get("exact_gt_reaction_best_rank"),
            "reactant_hit": audit_reactant,
            "reactant_best_rank": proposal_row.get("gt_reactant_best_rank"),
        },
        "route_search": {
            "n_results": (route_row.get("cascade_search") or {}).get("n_results"),
            "top_result_exact_hit": top_exact,
            "top_result_reactant_hit": top_reactant,
            "any_result_exact_hit": result_exact,
            "any_result_reactant_hit": result_reactant,
            "first_exact_result_rank": exact_result_rank,
            "first_reactant_result_rank": reactant_result_rank,
            "failure_categories": (route_row.get("cascade_search") or {}).get("failure_categories") or [],
        },
        "fusion_status": status,
        "recommended_next_action": _recommended_action(status),
    }


def _status(
    *,
    audit_exact: bool,
    audit_reactant: bool,
    top_exact: bool,
    top_reactant: bool,
    result_exact: bool,
    result_reactant: bool,
) -> str:
    if audit_exact:
        if top_exact:
            return "exact_candidate_top_result"
        if result_exact:
            return "exact_candidate_retained_below_top"
        return "exact_candidate_lost_after_direct_audit"
    if audit_reactant:
        if top_reactant:
            return "reactant_candidate_top_result"
        if result_reactant:
            return "reactant_candidate_retained_below_top"
        return "reactant_candidate_lost_after_direct_audit"
    if result_exact:
        return "route_exact_without_direct_audit_hit"
    if result_reactant:
        return "route_reactant_without_direct_audit_hit"
    return "no_direct_candidate_hit"


def _recommended_action(status: str) -> str:
    return {
        "exact_candidate_top_result": "Keep current fusion for this target; exact candidate survives as top route.",
        "exact_candidate_retained_below_top": "Use verifier/value reranking to promote the retained exact candidate.",
        "exact_candidate_lost_after_direct_audit": "Increase sidecar budget or calibrate legal-corpus scores before branch/result cutoff.",
        "reactant_candidate_top_result": "Use reaction-completion objective to turn retained precursor evidence into exact reactions.",
        "reactant_candidate_retained_below_top": "Promote retained precursor evidence, then complete missing reaction details.",
        "reactant_candidate_lost_after_direct_audit": "Increase sidecar/result budget or tune score fusion for precursor-preserving candidates.",
        "route_exact_without_direct_audit_hit": "Inspect alignment; route run found exact evidence not present in direct audit.",
        "route_reactant_without_direct_audit_hit": "Inspect alignment; route run found reactant evidence not present in direct audit.",
        "no_direct_candidate_hit": "Candidate generation/data coverage remains the bottleneck for this target.",
    }[status]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row.get("fusion_status") for row in rows)
    exact_direct = sum(1 for row in rows if row["direct_audit"]["exact_hit"])
    exact_any = sum(1 for row in rows if row["route_search"]["any_result_exact_hit"])
    exact_top = sum(1 for row in rows if row["route_search"]["top_result_exact_hit"])
    react_direct = sum(1 for row in rows if row["direct_audit"]["reactant_hit"])
    react_any = sum(1 for row in rows if row["route_search"]["any_result_reactant_hit"])
    react_top = sum(1 for row in rows if row["route_search"]["top_result_reactant_hit"])
    return {
        "n_targets": len(rows),
        "fusion_status_counts": dict(counts),
        "direct_exact_hit": exact_direct,
        "route_any_exact_hit": exact_any,
        "route_top_exact_hit": exact_top,
        "direct_exact_retention_rate_any_result": _rate(exact_any, exact_direct),
        "direct_exact_retention_rate_top_result": _rate(exact_top, exact_direct),
        "direct_reactant_hit": react_direct,
        "route_any_reactant_hit": react_any,
        "route_top_reactant_hit": react_top,
        "direct_reactant_retention_rate_any_result": _rate(react_any, react_direct),
        "direct_reactant_retention_rate_top_result": _rate(react_top, react_direct),
        "exact_candidate_lost_after_direct_audit": counts.get("exact_candidate_lost_after_direct_audit", 0),
        "reactant_candidate_lost_after_direct_audit": counts.get("reactant_candidate_lost_after_direct_audit", 0),
    }


def _decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(row.get("fusion_status") for row in rows)
    if counts.get("exact_candidate_lost_after_direct_audit", 0):
        return {
            "status": "score_fusion_or_budget_blocks_exact_candidates",
            "reason": "Direct legal-corpus audit finds exact GT reactions that route-level search does not retain in result programs.",
        }
    if counts.get("exact_candidate_retained_below_top", 0):
        return {
            "status": "rerank_can_promote_retained_exact_candidates",
            "reason": "Exact candidates survive route search but are below the top result.",
        }
    if counts.get("reactant_candidate_lost_after_direct_audit", 0):
        return {
            "status": "score_fusion_or_budget_blocks_reactant_candidates",
            "reason": "Direct legal-corpus audit finds GT reactants that route-level search does not retain.",
        }
    return {
        "status": "candidate_generation_or_completion_remains_primary",
        "reason": "Fusion does not appear to be losing direct exact candidates under this run.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Legal Corpus Search Fusion Diagnostic",
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
        f"- direct_exact_hit: {summary['direct_exact_hit']}",
        f"- route_any_exact_hit: {summary['route_any_exact_hit']}",
        f"- route_top_exact_hit: {summary['route_top_exact_hit']}",
        f"- direct_exact_retention_rate_any_result: {summary['direct_exact_retention_rate_any_result']}",
        f"- direct_reactant_hit: {summary['direct_reactant_hit']}",
        f"- route_any_reactant_hit: {summary['route_any_reactant_hit']}",
        f"- route_top_reactant_hit: {summary['route_top_reactant_hit']}",
        f"- direct_reactant_retention_rate_any_result: {summary['direct_reactant_retention_rate_any_result']}",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((summary.get("fusion_status_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Lost/Retained Candidate Targets",
            "",
            "| target | status | audit ranks | result ranks | action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["targets"]:
        status = row.get("fusion_status")
        if status == "no_direct_candidate_hit":
            continue
        audit = row["direct_audit"]
        route = row["route_search"]
        lines.append(
            "| `{target}` | `{status}` | exact={exact_rank}, reactant={react_rank} | exact={exact_result}, reactant={react_result} | {action} |".format(
                target=row.get("target_smiles"),
                status=status,
                exact_rank=audit.get("exact_best_rank"),
                react_rank=audit.get("reactant_best_rank"),
                exact_result=route.get("first_exact_result_rank"),
                react_result=route.get("first_reactant_result_rank"),
                action=row.get("recommended_next_action"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _rows_by_occurrence(rows: list[dict[str, Any]]) -> list[tuple[str, int, dict[str, Any]]]:
    counts: Counter[str] = Counter()
    out: list[tuple[str, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        target = str(row.get("target_smiles") or "")
        occurrence = counts[target]
        counts[target] += 1
        out.append((f"{target}#{occurrence}", index, row))
    return out


def _first_program_rank(programs: list[dict[str, Any]], hit_count_key: str) -> int | None:
    for program in programs:
        if int(program.get(hit_count_key) or 0) > 0:
            return int(program.get("rank") or 0) or None
    return None


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(float(num) / float(den), 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposal-audit", type=Path, required=True)
    ap.add_argument("--route-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    args = ap.parse_args()
    payload = diagnose_search_fusion(
        proposal_audit=args.proposal_audit,
        route_run=args.route_run,
        output_json=args.output,
        output_md=args.markdown_output,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
