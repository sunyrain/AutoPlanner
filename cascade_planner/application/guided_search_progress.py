"""Root-route progress accounting for bounded guided native search."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


GUIDED_SEARCH_PROGRESS_SCHEMA = "guided_search_root_stock_progress.v1"


def compile_parent_route_stock_progress(
    portfolio: Mapping[str, Any],
    *,
    parent_route_family_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Measure stock-open leaves only on target-rooted parent routes."""

    requested_families = {
        str(value) for value in parent_route_family_ids if str(value)
    }
    routes = [
        dict(route)
        for route in portfolio.get("route_candidates") or []
        if isinstance(route, Mapping)
        and route.get("root_edge_ids")
        and (
            not requested_families
            or str(route.get("route_family_id") or "") in requested_families
        )
    ]
    route_rows = [
        {
            "route_id": str(route.get("route_id") or ""),
            "route_family_id": str(route.get("route_family_id") or ""),
            "open_leaf_molecule_ids": sorted(
                str(value)
                for value in route.get("open_leaf_molecule_ids") or []
                if str(value)
            ),
            "all_leaves_stock_closed": (
                route.get("all_leaves_stock_closed") is True
            ),
        }
        for route in routes
    ]
    open_counts = [len(row["open_leaf_molecule_ids"]) for row in route_rows]
    payload = {
        "schema_version": GUIDED_SEARCH_PROGRESS_SCHEMA,
        "parent_route_family_ids": sorted(requested_families),
        "target_rooted_route_count": len(route_rows),
        "best_open_leaf_count": min(open_counts) if open_counts else None,
        "root_stock_closed": any(
            row["all_leaves_stock_closed"] for row in route_rows
        ),
        "routes": route_rows,
        "semantics": {
            "only_target_rooted_routes_are_counted": True,
            "subtarget_stock_closure_does_not_grant_root_progress": True,
            "best_parent_route_controls_result_progress": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def evaluate_guided_stock_progress(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    root_b4_reached: bool,
) -> dict[str, Any]:
    """Close one attempted frontier without suppressing distinct frontiers.

    ``root_b4_reached`` is a portfolio milestone, not ownership of the native
    frontier queue.  The caller already records the attempted molecule
    identity and suppresses only that exact occurrence on later scheduler
    passes.  Disabling the whole frontier resource here used to strand other
    target-reachable leaves as soon as any sibling route reached B4.
    """

    before_count = before.get("best_open_leaf_count")
    after_count = after.get("best_open_leaf_count")
    comparable = (
        isinstance(before_count, int)
        and not isinstance(before_count, bool)
        and isinstance(after_count, int)
        and not isinstance(after_count, bool)
    )
    decrease = int(before_count) - int(after_count) if comparable else 0
    progressed = bool(decrease > 0)
    if root_b4_reached:
        reason = "root_b4_stock_boundary_reached"
    elif progressed:
        reason = "parent_route_stock_open_leaf_count_decreased"
    elif comparable:
        reason = "parent_route_stock_open_leaf_count_not_decreased"
    else:
        reason = "target_rooted_parent_route_progress_unavailable"
    payload = {
        "schema_version": GUIDED_SEARCH_PROGRESS_SCHEMA,
        "before": dict(before),
        "after": dict(after),
        "stock_open_leaf_decrease": decrease,
        "root_b4_reached": bool(root_b4_reached),
        "progressed": progressed,
        "retry_same_frontier": False,
        "continue_guided_search": True,
        "reason": reason,
        "semantics": {
            "provider_success_alone_is_not_progress": True,
            "progress_is_measured_after_materialization_and_stock_audit": True,
            "root_b4_is_a_portfolio_milestone_not_a_frontier_queue_stop": True,
            "no_gain_closes_only_the_attempted_frontier": True,
            "distinct_untried_frontiers_remain_scheduler_eligible": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "GUIDED_SEARCH_PROGRESS_SCHEMA",
    "compile_parent_route_stock_progress",
    "evaluate_guided_stock_progress",
]
