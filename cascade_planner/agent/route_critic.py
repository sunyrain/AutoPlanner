"""Route critique over exported route JSON.

The critic only reads route_export payloads. It does not infer new chemistry
facts that are absent from the route artifact.
"""
from __future__ import annotations

from typing import Any

from cascade_planner.agent.schemas import CritiqueFinding, RouteCritique, SearchSuggestion


def _add(finding_list: list[CritiqueFinding], severity: str, kind: str, message: str, path: str) -> None:
    finding_list.append(CritiqueFinding(
        severity=severity,
        kind=kind,
        message=message,
        evidence_path=path,
    ))


def critique_route(route: dict[str, Any], route_id: str = "route_0") -> RouteCritique:
    metrics = route.get("metrics") or {}
    findings: list[CritiqueFinding] = []
    suggestions: list[SearchSuggestion] = []

    if not metrics.get("filled_route"):
        _add(findings, "high", "unfilled_route", "At least one step lacks a filled reaction candidate.", "metrics.filled_route")
        suggestions.append(SearchSuggestion("increase_candidate_budget", "Expand candidates for unfilled slots.", 20))

    stock = metrics.get("strict_stock_solve")
    if stock is False:
        _add(findings, "high", "stock_dead_end", "Terminal reactants are not all in stock.", "metrics.strict_stock_solve")
        suggestions.append(SearchSuggestion("try_alternative_route_mode", "Search for stock-terminating alternatives.", 10))

    cond = metrics.get("condition") or {}
    if cond.get("condition_window_success") is False:
        _add(findings, "medium", "condition_window", "Route has missing or incompatible T/pH window.", "metrics.condition")
        suggestions.append(SearchSuggestion("relax_condition_window", "Allow sequential or telescoped operation mode.", None))

    compat = metrics.get("cascade_compatibility") or {}
    for issue in compat.get("issues") or []:
        _add(findings, "medium", issue, f"Cascade compatibility issue: {issue}.", "metrics.cascade_compatibility.issues")

    enz = metrics.get("enzyme_evidence") or {}
    cov = enz.get("enzyme_evidence_coverage")
    if cov is not None and cov < 1.0:
        _add(findings, "medium", "enzyme_evidence_gap", "Some enzymatic steps lack complete evidence support.", "metrics.enzyme_evidence")
        suggestions.append(SearchSuggestion("request_more_evidence", "Request UniProt/cofactor/precedent evidence for enzymatic steps.", None))

    source_counts = metrics.get("candidate_source_counts") or {}
    if source_counts.get("unknown"):
        _add(findings, "low", "unknown_candidate_source", "One or more steps have unknown candidate provenance.", "metrics.candidate_source_counts")

    if not findings:
        acceptability = "acceptable"
        suggestions.append(SearchSuggestion("accept_route", "No deterministic route-export issues detected.", None))
    elif any(f.severity == "high" for f in findings):
        acceptability = "reject"
    else:
        acceptability = "needs_review"

    return RouteCritique(
        route_id=route_id,
        acceptability=acceptability,
        findings=findings,
        search_suggestions=suggestions,
        hallucinated_claims=[],
        source="deterministic",
    ).normalize()


def critique_route_payload(payload: dict[str, Any]) -> dict[str, Any]:
    critiques = []
    for idx, route in enumerate(payload.get("routes") or []):
        critiques.append(critique_route(route, route_id=f"route_{idx}").to_dict())
    return {
        "target": payload.get("target"),
        "n_routes": len(critiques),
        "critiques": critiques,
    }
