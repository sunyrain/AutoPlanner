"""Diversity-aware Pareto selection, metrics, and explicit closeout."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)
from cascade_planner.application.pareto import dominates
from cascade_planner.application.route_variants import with_content_digest
from cascade_planner.application.route_pareto_vector import pareto_coordinates


CLOSEOUT_SCHEMA = "retrosynthesis_closeout.v1"


def deduplicate_edge_sets(
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_edges: dict[tuple[str, ...], dict[str, Any]] = {}
    for value in candidates:
        row = dict(value)
        key = tuple(row.get("edge_ids") or [])
        current = by_edges.get(key)
        if current is None or _candidate_sort_key(row) < _candidate_sort_key(current):
            if current:
                row["equivalent_route_family_ids"] = sorted(
                    {
                        str(current["route_family_id"]),
                        *(current.get("equivalent_route_family_ids") or []),
                    }
                )
            by_edges[key] = with_content_digest(row)
        elif current:
            current["equivalent_route_family_ids"] = sorted(
                {
                    str(row["route_family_id"]),
                    *(current.get("equivalent_route_family_ids") or []),
                }
            )
            by_edges[key] = with_content_digest(current)
    return list(by_edges.values())


def pareto_front(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(value) for value in candidates]
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        vector = _objective_vector(row)
        if any(
            dominates(_objective_vector(other), vector)
            for other_index, other in enumerate(rows)
            if other_index != index
        ):
            continue
        out.append(row)
    return sorted(out, key=_candidate_sort_key)


def select_portfolio(
    candidates: Iterable[Mapping[str, Any]],
    *,
    minimum_count: int,
    maximum_count: int,
    require_distinct_edge_sets: bool,
) -> list[dict[str, Any]]:
    remaining = sorted((dict(value) for value in candidates), key=_candidate_sort_key)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < maximum_count:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in remaining:
            if require_distinct_edge_sets and any(
                set(row.get("edge_ids") or []) == set(value.get("edge_ids") or [])
                for value in selected
            ):
                continue
            diversity = (
                min(route_distance(row, value) for value in selected)
                if selected
                else 1.0
            )
            new_strategy = not any(
                row.get("root_edge_ids") == value.get("root_edge_ids")
                for value in selected
            )
            utility = (
                _candidate_utility(row)
                + 35.0 * diversity
                + 8.0 * new_strategy
                + 5.0 * (row.get("pareto_optimal") is True)
            )
            scored.append((-utility, str(row["route_id"]), row))
        if not scored:
            break
        chosen = min(scored)[2]
        selected.append(chosen)
        remaining = [row for row in remaining if row["route_id"] != chosen["route_id"]]
        if len(selected) >= minimum_count and not remaining:
            break
    return selected


def portfolio_metrics(
    selected: Iterable[Mapping[str, Any]],
    *,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    routes = [dict(value) for value in selected]
    pairwise: list[dict[str, Any]] = []
    for index, left in enumerate(routes):
        for right in routes[index + 1 :]:
            pairwise.append(
                {
                    "left_route_id": left["route_id"],
                    "right_route_id": right["route_id"],
                    "edge_set_distance": round(route_distance(left, right), 6),
                    "strategic_disconnection_distinct": (
                        left.get("root_edge_ids") != right.get("root_edge_ids")
                    ),
                }
            )
    edge_routes: dict[str, list[str]] = {}
    molecule_routes: dict[str, list[str]] = {}
    for route in routes:
        for edge_id in route.get("edge_ids") or []:
            edge_routes.setdefault(str(edge_id), []).append(str(route["route_id"]))
            edge = dict(graph.get("edges") or {}).get(str(edge_id)) or {}
            for molecule_id in [
                edge.get("product_molecule_id"),
                *(edge.get("precursor_molecule_ids") or []),
            ]:
                molecule_routes.setdefault(str(molecule_id), []).append(
                    str(route["route_id"])
                )
    return {
        "selected_route_count": len(routes),
        "complete_route_count": sum(value.get("complete") is True for value in routes),
        "distinct_edge_set_count": len(
            {tuple(value.get("edge_ids") or []) for value in routes}
        ),
        "distinct_complete_edge_set_count": len(
            {
                tuple(value.get("edge_ids") or [])
                for value in routes
                if value.get("complete") is True
            }
        ),
        "strategic_disconnection_count": len(
            {tuple(value.get("root_edge_ids") or []) for value in routes}
        ),
        "complete_strategic_disconnection_count": len(
            {
                tuple(value.get("root_edge_ids") or [])
                for value in routes
                if value.get("complete") is True
            }
        ),
        "pairwise_diversity": pairwise,
        "shared_bottleneck_edges": {
            edge_id: sorted(route_ids)
            for edge_id, route_ids in sorted(edge_routes.items())
            if len(set(route_ids)) > 1
        },
        "shared_intermediates": {
            molecule_id: sorted(set(route_ids))
            for molecule_id, route_ids in sorted(molecule_routes.items())
            if molecule_id and len(set(route_ids)) > 1
        },
        "minimum_selected_proof_level": min(
            (int(value.get("minimum_edge_proof_level") or 0) for value in routes),
            default=0,
        ),
        "mean_length": round(
            sum(int(value.get("length") or 0) for value in routes)
            / max(1, len(routes)),
            6,
        ),
        "mean_risk": round(
            sum(float(value.get("risk_score") or 0.0) for value in routes)
            / max(1, len(routes)),
            6,
        ),
    }


def closeout(
    selected: Iterable[Mapping[str, Any]],
    *,
    deficits: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    acceptance: RetrosynthesisAcceptanceSpec,
    graph_reasons: Iterable[str],
    budget_exhausted: bool,
) -> dict[str, Any]:
    routes = [dict(value) for value in selected]
    graph_reason_list = list(graph_reasons)
    reasons = list(graph_reason_list)
    complete = sum(value.get("complete") is True for value in routes)
    if complete < acceptance.minimum_complete_routes:
        reasons.append("minimum_complete_route_count_not_met")
    if acceptance.require_distinct_edge_sets and int(
        metrics.get("distinct_complete_edge_set_count") or 0
    ) < acceptance.minimum_complete_routes:
        reasons.append("distinct_complete_edge_sets_not_met")
    if any(
        value.get("all_edges_proven") is not True
        for value in routes
        if value.get("complete")
    ):
        reasons.append("selected_complete_route_has_unproven_edge")
    if any(
        value.get("all_leaves_stock_closed") is not True
        for value in routes
        if value.get("complete")
    ):
        reasons.append("selected_complete_route_has_open_leaf")
    accepted = not reasons and complete >= acceptance.minimum_complete_routes
    if graph_reason_list:
        decision = "invalid"
    elif accepted:
        decision = "accepted"
    elif budget_exhausted:
        decision = "budget_exhausted"
    else:
        decision = "unresolved"
    row = {
        "schema_version": CLOSEOUT_SCHEMA,
        "decision": decision,
        "accepted": accepted,
        "complete_route_count": complete,
        "selected_route_count": len(routes),
        "deficit_count": len(deficits),
        "reasons": sorted(set(reasons)),
        "semantics": {
            "only_boolean_proof_and_stock_closure_can_accept": True,
            "counts_are_diagnostics_not_authority": True,
            "budget_exhaustion_never_means_success": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def deduplicate_records(
    values: Iterable[Mapping[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    rows = {str(value.get(key) or ""): dict(value) for value in values}
    return [rows[value] for value in sorted(rows) if value]


def route_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_edges = set(left.get("edge_ids") or [])
    right_edges = set(right.get("edge_ids") or [])
    union = left_edges | right_edges
    return 1.0 if not union else 1.0 - len(left_edges & right_edges) / len(union)


def _candidate_utility(value: Mapping[str, Any]) -> float:
    return (
        74.0 * float(value.get("strategic_value_score") or 0.0)
        + 105.0 * (value.get("complete") is True)
        + 9.0 * float(value.get("evidence_maturity_score") or 0.0)
        + 11.0 * int(value.get("minimum_edge_proof_level") or 0)
        + 14.0 * float(value.get("stock_closure_rate") or 0.0)
        + 3.0 * min(3, len(value.get("independent_source_groups") or []))
        + 6.0 * float(value.get("convergence_score") or 0.0)
        - 35.0 * float(value.get("risk_score") or 0.0)
        - 1.5 * int(value.get("length") or 0)
    )


def _candidate_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -(value.get("complete") is True),
        -_candidate_utility(value),
        float(value.get("risk_score") or 0.0),
        int(value.get("length") or 0),
        str(value.get("route_id") or ""),
    )


def _objective_vector(value: Mapping[str, Any]) -> tuple[float, ...]:
    return pareto_coordinates(value)


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
    "CLOSEOUT_SCHEMA",
    "closeout",
    "deduplicate_edge_sets",
    "deduplicate_records",
    "pareto_front",
    "portfolio_metrics",
    "route_distance",
    "select_portfolio",
]
