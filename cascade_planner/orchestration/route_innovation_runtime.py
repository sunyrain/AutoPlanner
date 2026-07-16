"""Application-service adapter for generic route innovation discovery."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)
from cascade_planner.application.proof_portfolio import compile_proof_portfolio


def review_route_innovations(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Select one canonical portfolio route and compile proposal-only options."""

    enriched_graph, route = route_innovation_context(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
    )
    return discover_route_innovations(
        enriched_graph,
        route,
        capabilities=capabilities,
        mechanism_proposals=mechanism_proposals,
    )


def route_innovation_context(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    portfolio = compile_proof_portfolio(graph, acceptance_spec=acceptance_spec)
    edge_proofs = dict(portfolio.get("edge_proofs") or {})
    enriched_graph = {
        **dict(graph),
        "edges": {
            edge_id: {
                **dict(edge),
                "innovation_boundary_proof_level": int(
                    dict(edge_proofs.get(edge_id) or {}).get("achieved_level") or 0
                ),
            }
            for edge_id, edge in dict(graph.get("edges") or {}).items()
        },
    }
    routes = [
        dict(value)
        for value in portfolio.get("route_candidates") or []
        if isinstance(value, Mapping)
    ]
    route = next(
        (value for value in routes if value.get("route_id") == route_id),
        None,
    )
    if route is None:
        raise ValueError(f"route_innovation_route_not_found:{route_id}")
    return enriched_graph, route


__all__ = ["review_route_innovations", "route_innovation_context"]
