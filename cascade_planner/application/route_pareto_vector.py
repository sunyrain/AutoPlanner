"""Explicit independent objective axes for proof-portfolio Pareto selection."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


ROUTE_PARETO_OBJECTIVE_VECTOR_SCHEMA = "route_pareto_objective_vector.v1"
_AXIS_ORDER = (
    "strategic_value",
    "evidence_maturity",
    "topology_closure",
    "stock_closure",
    "reaction_feasibility",
    "proof_evidence",
    "condition_completeness",
    "route_diversity",
    "cost_length",
    "program_readiness",
)


def compile_route_pareto_objective_vector(
    candidate: Mapping[str, Any],
    *,
    peers: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compile auditable axes without treating missing cost as zero cost."""

    row = dict(candidate)
    peer_rows = [dict(value) for value in peers]
    edge_ids = {str(value) for value in row.get("edge_ids") or [] if str(value)}
    root_ids = {
        str(value) for value in row.get("root_edge_ids") or [] if str(value)
    }
    diversity = min(
        (_edge_set_distance(edge_ids, other.get("edge_ids") or []) for other in peer_rows
         if str(other.get("route_id") or "") != str(row.get("route_id") or "")),
        default=1.0,
    )
    length = max(0, int(row.get("length") or 0))
    explicit_cost = _finite_nonnegative(row.get("estimated_cost"))
    cost_known = explicit_cost is not None
    program_count = (
        int(row.get("biocatalytic_step_count") or 0)
        + int(row.get("mechanism_extrapolation_count") or 0)
    )
    strategy_domain = str(row.get("execution_domain") or "chemical")
    program_applicability = (
        "applicable"
        if program_count
        or strategy_domain in {"enzymatic", "whole_cell", "hybrid", "mechanistic"}
        else "not_applicable"
    )
    program_score = (
        float(not row.get("unvalidated_biocatalytic_edge_ids"))
        if program_count
        else 1.0
    )
    axes = {
        "strategic_value": {
            "score": _unit(row.get("strategic_value_score")),
            "known": bool(row.get("strategy_digest") or row.get("strategic_value")),
            "basis": "strategy_card_and_canonical_topology_only",
            "evidence_independent": True,
        },
        "evidence_maturity": {
            "score": _unit(row.get("evidence_maturity_score")),
            "known": True,
            "basis": "host_proof_and_source_records_only",
            "strategy_wording_independent": True,
        },
        "topology_closure": {
            "score": float(bool(edge_ids and root_ids)),
            "known": True,
            "target_rooted": bool(root_ids),
        },
        "stock_closure": {
            "score": _unit(row.get("stock_closure_rate")),
            "known": True,
        },
        "reaction_feasibility": {
            "score": _unit(row.get("reaction_feasibility_rate")),
            "known": True,
        },
        "proof_evidence": {
            "score": _unit(row.get("exact_evidence_rate")),
            "known": True,
        },
        "condition_completeness": {
            "score": _unit(row.get("condition_completeness_rate")),
            "known": True,
        },
        "route_diversity": {
            "score": round(diversity, 6),
            "known": True,
            "basis": "nearest_peer_edge_set_jaccard_distance",
        },
        "cost_length": {
            "score": round(1.0 / (1.0 + length), 6),
            "known": cost_known,
            "length_known": True,
            "length": length,
            "cost_known": cost_known,
            "estimated_cost": explicit_cost,
            "unknown_cost_is_not_zero": not cost_known,
        },
        "program_readiness": {
            "score": program_score,
            "known": True,
            "applicability": program_applicability,
            "neutral_when_not_applicable": program_applicability == "not_applicable",
            "strategy_execution_domain": strategy_domain,
        },
    }
    result = {
        "schema_version": ROUTE_PARETO_OBJECTIVE_VECTOR_SCHEMA,
        "axes": axes,
        "axis_order": list(_AXIS_ORDER),
        "semantics": {
            "axes_are_independent": True,
            "unknown_cost_is_not_imputed_as_zero": True,
            "conventional_program_not_applicable_is_neutral": True,
            "scalar_utility_is_display_only": True,
            "pareto_coordinates_use_only_axis_scores": True,
            "strategic_value_and_evidence_maturity_are_independent": True,
            "high_strategy_low_evidence_routes_remain_pareto_eligible": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def pareto_coordinates(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    vector = dict(candidate.get("pareto_objective_vector") or {})
    axes = dict(vector.get("axes") or {})
    if vector.get("schema_version") != ROUTE_PARETO_OBJECTIVE_VECTOR_SCHEMA:
        raise ValueError("route_pareto_objective_vector_missing_or_invalid")
    return tuple(_unit(dict(axes.get(name) or {}).get("score")) for name in _AXIS_ORDER)


def _edge_set_distance(left: set[str], right_values: Iterable[Any]) -> float:
    right = {str(value) for value in right_values if str(value)}
    union = left | right
    return 1.0 if not union else 1.0 - len(left & right) / len(union)


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in {float("inf"), float("-inf")}:
        return 0.0
    return max(0.0, min(1.0, number))


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")} or number < 0:
        return None
    return number


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ROUTE_PARETO_OBJECTIVE_VECTOR_SCHEMA",
    "compile_route_pareto_objective_vector",
    "pareto_coordinates",
]
