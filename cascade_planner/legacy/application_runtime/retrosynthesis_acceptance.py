"""Deterministic acceptance audit for a multi-route retrosynthesis run."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)


RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA = "retrosynthesis_acceptance_report.v1"


def evaluate_retrosynthesis_acceptance(
    *,
    route_portfolio: Mapping[str, Any] | None,
    acceptance_spec: RetrosynthesisAcceptanceSpec | None = None,
) -> dict[str, Any]:
    """Apply the operator contract to replay-verified portfolio output."""

    spec = acceptance_spec or RetrosynthesisAcceptanceSpec()
    portfolio = dict(route_portfolio or {})
    candidates = [
        dict(row)
        for row in portfolio.get("routes") or []
        if isinstance(row, Mapping)
    ]
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, route in enumerate(candidates, start=1):
        route_id = str(route.get("route_id") or f"route:{index}")
        reasons: list[str] = []
        if route.get("complete") is not True:
            reasons.append("route_not_complete")
        try:
            weakest = int(route.get("weakest_proof_level") or 0)
        except (TypeError, ValueError):
            weakest = 0
        if weakest < spec.minimum_edge_proof_level:
            reasons.append(
                f"weakest_edge_proof_level_{weakest}_below_"
                f"{spec.minimum_edge_proof_level}"
            )
        if route.get("reaction_validated") is not True:
            reasons.append("route_reactions_not_validated")
        if spec.require_all_selected_leaves_stock_closed and not _stock_ready(
            route,
            boundary=spec.stock_boundary,
        ):
            reasons.append(f"route_leaves_not_{spec.stock_boundary}_closed")
        edge_ids = _edge_ids(route)
        if not edge_ids:
            reasons.append("route_has_no_reaction_edges")
        row = {
            "route_id": route_id,
            "edge_ids": sorted(edge_ids),
            "independent_support_groups": sorted(
                {
                    str(item)
                    for item in route.get("independent_support_groups") or []
                    if str(item or "").strip()
                }
            ),
            "weakest_proof_level": weakest,
            "accepted": not reasons,
            "reasons": sorted(set(reasons)),
        }
        (eligible if not reasons else rejected).append(row)

    distinct = _distinct_routes(
        eligible,
        require_distinct=spec.require_distinct_edge_sets,
    )
    selected = distinct[: spec.minimum_complete_routes]
    support_groups = sorted(
        {
            group
            for route in selected
            for group in route["independent_support_groups"]
        }
    )
    reasons: list[str] = []
    if len(selected) < spec.minimum_complete_routes:
        reasons.append(
            f"complete_distinct_route_count_{len(selected)}_below_"
            f"{spec.minimum_complete_routes}"
        )
    if len(support_groups) < spec.minimum_independent_source_groups:
        reasons.append(
            f"independent_source_group_count_{len(support_groups)}_below_"
            f"{spec.minimum_independent_source_groups}"
        )

    report: dict[str, Any] = {
        "schema_version": RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA,
        "accepted": not reasons,
        "acceptance_spec": spec.to_dict(),
        "selected_route_ids": [row["route_id"] for row in selected],
        "selected_route_count": len(selected),
        "eligible_route_count": len(eligible),
        "independent_support_groups": support_groups,
        "eligible_routes": eligible,
        "rejected_routes": rejected,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "portfolio_score_is_not_acceptance_authority": True,
            "route_solver_proof_and_stock_bindings_are_reused_not_regranted": True,
            "distinct_routes_require_distinct_edge_sets": (
                spec.require_distinct_edge_sets
            ),
            "source_independence_is_measured_across_selected_routes": True,
        },
    }
    report["content_sha256"] = _digest(report)
    return report


def _stock_ready(route: Mapping[str, Any], *, boundary: str) -> bool:
    if boundary == "procurement":
        return bool(
            route.get("procurement_stock_closed") is True
            or route.get("procurement_ready") is True
        )
    if boundary == "in_house":
        return bool(
            route.get("in_house_stock_closed") is True
            or route.get("in_house_ready") is True
        )
    return bool(
        route.get("benchmark_stock_closed") is True
        or route.get("complete") is True
    )


def _edge_ids(route: Mapping[str, Any]) -> set[str]:
    direct = route.get("hyperedge_ids") or route.get("edge_ids") or []
    edge_ids = {str(item) for item in direct if str(item or "").strip()}
    for raw in route.get("selected_hyperedges") or []:
        if isinstance(raw, Mapping):
            edge_id = str(raw.get("hyperedge_id") or "")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            edge_id = str(raw[1])
        else:
            edge_id = ""
        if edge_id:
            edge_ids.add(edge_id)
    return edge_ids


def _distinct_routes(
    routes: list[dict[str, Any]],
    *,
    require_distinct: bool,
) -> list[dict[str, Any]]:
    if not require_distinct:
        return list(routes)
    selected: list[dict[str, Any]] = []
    edge_sets: set[tuple[str, ...]] = set()
    for route in routes:
        signature = tuple(route["edge_ids"])
        if signature in edge_sets:
            continue
        edge_sets.add(signature)
        selected.append(route)
    return selected


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA",
    "evaluate_retrosynthesis_acceptance",
]
