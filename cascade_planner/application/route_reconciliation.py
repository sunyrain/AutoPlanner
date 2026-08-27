"""Reconcile provider search projections with the canonical route graph.

The paper-equivalent metric intentionally operates on the canonical graph and
an inventory oracle.  That is the right metric, but it does not explain when a
provider projected a connected route and the host later quarantined one or
more ReactionJSON candidates.  This module keeps those axes explicit without
changing either acceptance policy or the paper metric.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


ROUTE_RECONCILIATION_SCHEMA = "route_search_materialization_reconciliation.v1"


def _route_records(
    records: Iterable[Mapping[str, Any]],
    route_family_id: str,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for origin in record.get("origin_records") or []:
            if not isinstance(origin, Mapping):
                continue
            if str(origin.get("route_family_id") or "") == route_family_id:
                selected.append(record)
                break
    return selected


def _route_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("route_family_id") or "")
        for row in rows
        if isinstance(row, Mapping) and str(row.get("route_family_id") or "")
    }


def _canonical_route_ids(
    records: Iterable[Mapping[str, Any]],
    projected_route_id: str,
) -> set[str]:
    """Resolve one Director alias to lifecycle-owned canonical family ids."""

    canonical: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        matched = False
        for origin in record.get("origin_records") or []:
            if not isinstance(origin, Mapping):
                continue
            if str(origin.get("route_family_id") or "") != projected_route_id:
                continue
            matched = True
            canonical.update(
                str(value)
                for value in origin.get("canonical_route_family_ids") or []
                if str(value)
            )
        if matched:
            canonical.update(
                str(value)
                for value in record.get("route_family_ids") or []
                if str(value)
            )
    return canonical


def compile_route_reconciliation(
    outcomes: Iterable[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    paper_equivalent: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain route progress across search, admission, materialization, and stock.

    ``projected_step_count`` is the provider's connected path projection;
    ``materialized_step_count`` is the count that survived canonical host
    materialization.  They are deliberately not collapsed into one number.
    """

    route_families_by_id: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or str(outcome.get("status") or "") != "accepted":
            continue
        plan = outcome.get("plan")
        if not isinstance(plan, Mapping):
            continue
        for family in plan.get("route_families") or []:
            if isinstance(family, Mapping):
                route_id = str(family.get("route_family_id") or "")
                if route_id:
                    # A bounded replan can repeat a route family.  The last
                    # accepted projection is the authoritative current view;
                    # historical attempts remain in candidate lifecycle.
                    route_families_by_id[route_id] = family

    reached_ids = _route_ids(paper_equivalent.get("reached_routes") or [])
    solved_ids = _route_ids(paper_equivalent.get("solved_routes") or [])
    lifecycle_records = lifecycle.get("records") or []
    routes: list[dict[str, Any]] = []
    for family in route_families_by_id.values():
        route_id = str(family.get("route_family_id") or "")
        if not route_id:
            continue
        search = dict(family.get("aizynthfinder_strategy_search") or {})
        records = _route_records(lifecycle_records, route_id)
        canonical_route_ids = _canonical_route_ids(lifecycle_records, route_id)
        materialized_records = [
            record
            for record in records
            if dict(record.get("materialization") or {}).get("materialized") is True
        ]
        admitted_records = [
            record
            for record in records
            if dict(record.get("admission") or {}).get("accepted") is True
        ]
        quarantined_records = [
            record
            for record in records
            if dict(record.get("materialization") or {}).get("materialized") is not True
        ]
        gap_reasons = sorted(
            {
                str(reason)
                for record in quarantined_records
                for reason in (
                    list(dict(record.get("admission") or {}).get("reasons") or [])
                    + list(dict(record.get("validation") or {}).get("reasons") or [])
                )
                if str(reason).strip()
            }
        )
        projected = int(
            search.get("path_route_step_count")
            or search.get("path_action_count")
            or family.get("route_call_count")
            or 0
        )
        materialized = len(materialized_records)
        comparison_ids = {route_id, *canonical_route_ids}
        reached = bool(comparison_ids.intersection(reached_ids))
        solved = bool(comparison_ids.intersection(solved_ids))
        if quarantined_records:
            classification = "materialization_admission_gap"
        elif solved:
            classification = "paper_equivalent_solved"
        elif reached:
            classification = "stock_closure_open"
        elif projected and materialized == 0:
            classification = "not_materialized"
        else:
            classification = "search_not_reached"
        routes.append(
            {
                "route_family_id": route_id,
                "canonical_route_family_ids": sorted(canonical_route_ids),
                "title": str(family.get("title") or ""),
                "projected_step_count": projected,
                "admitted_step_count": len(admitted_records),
                "materialized_step_count": materialized,
                "quarantined_step_count": len(quarantined_records),
                "materialization_gap_step_count": max(0, projected - materialized),
                "materialization_gap_reasons": gap_reasons,
                "search_selected_open_leaves": int(search.get("selected_open_leaves") or 0),
                "search_selected_solved": bool(search.get("selected_solved")),
                "search_path_projection_complete": bool(
                    search.get("path_route_projection_complete")
                ),
                "paper_reached": reached,
                "paper_equivalent_solved": solved,
                "stock_closure_status": "closed" if solved else "open" if reached else "not_reached",
                "classification": classification,
                "semantics": {
                    "projected_steps_are_not_canonical_steps": True,
                    "paper_metric_is_existential_stock_topology_only": True,
                    "quarantine_precedes_stock_interpretation": True,
                },
            }
        )
    routes.sort(key=lambda row: row["route_family_id"])
    return {
        "schema_version": ROUTE_RECONCILIATION_SCHEMA,
        "routes": routes,
        "route_count": len(routes),
        "materialization_gap_route_count": sum(
            row["classification"] == "materialization_admission_gap" for row in routes
        ),
        "paper_reached_route_count": sum(row["paper_reached"] for row in routes),
        "paper_equivalent_solved_route_count": sum(
            row["paper_equivalent_solved"] for row in routes
        ),
        "semantics": {
            "diagnostic_only": True,
            "does_not_change_paper_equivalent_metric": True,
            "does_not_grant_reaction_validation": True,
            "does_not_grant_scientific_acceptance": True,
        },
    }


__all__ = ["ROUTE_RECONCILIATION_SCHEMA", "compile_route_reconciliation"]
