"""Reconcile provider search projections with the canonical route graph.

The paper-equivalent metric intentionally operates on the canonical graph and
an inventory oracle.  That is the right metric, but it does not explain when a
provider projected a connected route and the host later quarantined one or
more ReactionJSON candidates.  This module keeps those axes explicit without
changing either acceptance policy or the paper metric.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


ROUTE_RECONCILIATION_SCHEMA = "route_search_materialization_reconciliation.v4"


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
        origins = [
            origin for origin in record.get("origin_records") or [] if isinstance(origin, Mapping)
        ]
        matching_origins = [
            origin
            for origin in origins
            if str(origin.get("route_family_id") or "") == projected_route_id
        ]
        if not matching_origins:
            continue
        origin_bindings = [
            {
                str(value)
                for value in origin.get("canonical_route_family_ids") or []
                if str(value)
            }
            for origin in matching_origins
        ]
        origin_bindings = [binding for binding in origin_bindings if binding]
        # Historical materialization copied the record-level OR-union of
        # canonical families onto every origin.  When the matching hypothesis
        # also retains a narrower binding for that same projected alias, the
        # inclusion-minimal binding is the lossless authority.  Independent
        # singleton bindings are all retained; only strict supersets are
        # discarded.  New graph revisions preserve the narrow binding at
        # write time, so this is a deterministic projection migration only.
        narrow_bindings = [
            binding
            for binding in origin_bindings
            if not any(other < binding for other in origin_bindings)
        ]
        origin_canonical_ids = {
            value for binding in narrow_bindings for value in binding
        }
        if origin_canonical_ids:
            canonical.update(origin_canonical_ids)
            continue

        # Older single-route lifecycle records did not carry the explicit
        # origin -> canonical-family binding.  Their record-level families are
        # a safe fallback only when the record contains no other projected
        # route origin.  A shared target/root record may list several
        # canonical families; merging that list into every matching Strategy
        # silently splices otherwise independent routes together.
        projected_origin_ids = {
            str(origin.get("route_family_id") or "")
            for origin in origins
            if str(origin.get("route_family_id") or "")
        }
        if projected_origin_ids <= {projected_route_id}:
            canonical.update(
                str(value) for value in record.get("route_family_ids") or [] if str(value)
            )
    return canonical


def _longest_linear_sequence(
    graph: Mapping[str, Any] | None,
    edge_ids: Iterable[str],
) -> int:
    """Count the longest target-rooted edge path in one canonical route."""

    if not isinstance(graph, Mapping):
        return 0
    edges = {
        str(edge_id): dict(dict(graph.get("edges") or {}).get(str(edge_id)) or {})
        for edge_id in edge_ids
        if str(edge_id) in dict(graph.get("edges") or {})
    }
    target_id = str(graph.get("target_molecule_id") or "")
    by_product: dict[str, list[Mapping[str, Any]]] = {}
    for edge in edges.values():
        product_id = str(edge.get("product_molecule_id") or "")
        if product_id:
            by_product.setdefault(product_id, []).append(edge)

    def visit(molecule_id: str, active: frozenset[str]) -> int:
        if not molecule_id or molecule_id in active:
            return 0
        next_active = active | {molecule_id}
        return max(
            (
                1
                + max(
                    (
                        visit(str(precursor_id), next_active)
                        for precursor_id in edge.get("precursor_molecule_ids") or ()
                    ),
                    default=0,
                )
                for edge in by_product.get(molecule_id, ())
            ),
            default=0,
        )

    return visit(target_id, frozenset())


def _current_route_successors(
    graph: Mapping[str, Any] | None,
    route_family_ids: Iterable[str],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Follow repair lineage to the unique currently selected descendant."""

    origin_ids = {str(value) for value in route_family_ids if str(value)}
    if not isinstance(graph, Mapping):
        return origin_ids, []
    families = {
        str(route_id): dict(route)
        for route_id, route in dict(graph.get("route_families") or {}).items()
        if isinstance(route, Mapping)
    }
    children: dict[str, list[str]] = {}
    for route_id, route in families.items():
        parent_id = str(route.get("supersedes_route_family_id") or "")
        if parent_id:
            children.setdefault(parent_id, []).append(route_id)

    resolved: set[str] = set()
    lineages: list[dict[str, Any]] = []
    for origin_id in sorted(origin_ids):
        pending: list[tuple[str, tuple[str, ...]]] = [(origin_id, (origin_id,))]
        selected_descendants: list[tuple[str, tuple[str, ...]]] = []
        while pending:
            current_id, lineage = pending.pop()
            if current_id in families and families[current_id].get("selected") is not False:
                selected_descendants.append((current_id, lineage))
            for child_id in sorted(children.get(current_id, ()), reverse=True):
                if child_id not in lineage:
                    pending.append((child_id, (*lineage, child_id)))
        deepest = max((len(lineage) for _route_id, lineage in selected_descendants), default=0)
        finalists = [
            (route_id, lineage)
            for route_id, lineage in selected_descendants
            if len(lineage) == deepest
        ]
        if len(finalists) == 1:
            current_id, lineage = finalists[0]
            resolved.add(current_id)
            lineages.append(
                {
                    "origin_route_family_id": origin_id,
                    "current_route_family_id": current_id,
                    "route_family_ids": list(lineage),
                    "ambiguous": False,
                }
            )
        else:
            resolved.add(origin_id)
            lineages.append(
                {
                    "origin_route_family_id": origin_id,
                    "current_route_family_id": origin_id,
                    "route_family_ids": [origin_id],
                    "ambiguous": bool(finalists),
                    "selected_descendant_route_family_ids": sorted(
                        route_id for route_id, _lineage in finalists
                    ),
                }
            )
    return resolved, lineages


def compile_route_reconciliation(
    outcomes: Iterable[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    paper_equivalent: Mapping[str, Any],
    *,
    graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate Builder history from the final canonical route and stock state."""

    route_families_by_id: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or str(outcome.get("status") or "") != "accepted":
            continue
        plan = outcome.get("plan")
        if not isinstance(plan, Mapping):
            continue
        skeleton_steps_by_family: dict[str, list[Mapping[str, Any]]] = {}
        for skeleton in plan.get("multi_step_skeletons") or []:
            if not isinstance(skeleton, Mapping):
                continue
            route_id = str(skeleton.get("route_family_id") or "")
            steps = [
                step
                for step in skeleton.get("steps") or []
                if isinstance(step, Mapping)
            ]
            if route_id and len(steps) > len(skeleton_steps_by_family.get(route_id, [])):
                skeleton_steps_by_family[route_id] = steps
        for family in plan.get("route_families") or []:
            if isinstance(family, Mapping):
                route_id = str(family.get("route_family_id") or "")
                if route_id:
                    # A bounded replan can repeat a route family.  The last
                    # accepted projection is the authoritative current view;
                    # historical attempts remain in candidate lifecycle.
                    current_family = dict(family)
                    if not current_family.get("steps") and skeleton_steps_by_family.get(
                        route_id
                    ):
                        current_family["steps"] = skeleton_steps_by_family[route_id]
                    route_families_by_id[route_id] = current_family

    reached_rows = [
        dict(row)
        for row in paper_equivalent.get("reached_routes") or []
        if isinstance(row, Mapping)
    ]
    solved_rows = [
        dict(row) for row in paper_equivalent.get("solved_routes") or [] if isinstance(row, Mapping)
    ]
    reached_ids = _route_ids(reached_rows)
    solved_ids = _route_ids(solved_rows)
    lifecycle_records = lifecycle.get("records") or []
    routes: list[dict[str, Any]] = []
    for family in route_families_by_id.values():
        route_id = str(family.get("route_family_id") or "")
        if not route_id:
            continue
        search = dict(family.get("aizynthfinder_strategy_search") or {})
        records = _route_records(lifecycle_records, route_id)
        origin_canonical_route_ids = _canonical_route_ids(
            lifecycle_records,
            route_id,
        )
        canonical_route_ids, repair_successor_lineages = _current_route_successors(
            graph,
            origin_canonical_route_ids,
        )
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
        strategy_search_projected = int(
            search.get("path_route_step_count")
            or search.get("path_action_count")
            or family.get("route_call_count")
            or 0
        )
        builder_route_rows = [row for row in family.get("steps") or [] if isinstance(row, Mapping)]
        builder_projected = (
            len(builder_route_rows) if builder_route_rows else strategy_search_projected
        )
        materialized = len(materialized_records)
        comparison_ids = {route_id, *canonical_route_ids}
        reached = bool(comparison_ids.intersection(reached_ids))
        solved = bool(comparison_ids.intersection(solved_ids))
        final_edge_ids = sorted(
            {
                str(edge_id)
                for row in reached_rows
                if str(row.get("route_family_id") or "") in comparison_ids
                for edge_id in row.get("edge_ids") or ()
                if str(edge_id)
            }
        )
        graph_routes = (
            dict(graph.get("route_families") or {})
            if isinstance(graph, Mapping)
            else {}
        )
        final_critic_reviews = [
            {
                "route_family_id": canonical_route_id,
                "status": str(critique.get("status") or ""),
                "review_state": str(critique.get("review_state") or ""),
                "overall_assessment": str(
                    critique.get("overall_assessment") or ""
                ),
                "route_overall_evaluation": str(
                    critique.get("route_overall_evaluation") or ""
                ),
            }
            for canonical_route_id in sorted(canonical_route_ids)
            for critique in (
                dict(
                    dict(graph_routes.get(canonical_route_id) or {}).get(
                        "chemical_critic"
                    )
                    or {}
                ),
            )
            if critique
        ]
        if solved:
            classification = "paper_equivalent_solved"
        elif reached:
            classification = "stock_closure_open"
        elif quarantined_records:
            classification = "builder_candidates_quarantined"
        elif builder_projected and materialized == 0:
            classification = "not_materialized"
        else:
            classification = "search_not_reached"
        routes.append(
            {
                "route_family_id": route_id,
                "canonical_route_family_ids": sorted(canonical_route_ids),
                "origin_canonical_route_family_ids": sorted(
                    origin_canonical_route_ids
                ),
                "repair_successor_lineages": repair_successor_lineages,
                "title": str(family.get("title") or ""),
                "builder_projected_step_count": builder_projected,
                "builder_admitted_step_count": len(admitted_records),
                "host_materialized_candidate_count": materialized,
                "builder_quarantined_candidate_count": len(quarantined_records),
                "builder_quarantine_reasons": gap_reasons,
                "final_canonical_edge_count": len(final_edge_ids),
                "final_longest_linear_sequence": _longest_linear_sequence(
                    graph,
                    final_edge_ids,
                ),
                "final_stock_closed": solved,
                "final_critic_reviews": final_critic_reviews,
                "final_critic_status": (
                    final_critic_reviews[0]["status"]
                    if len(final_critic_reviews) == 1
                    else ""
                ),
                "final_route_overall_evaluation": (
                    final_critic_reviews[0]["route_overall_evaluation"]
                    if len(final_critic_reviews) == 1
                    else ""
                ),
                "strategy_search_projected_step_count": strategy_search_projected,
                "strategy_search_selected_open_leaves": int(
                    search.get("selected_open_leaves") or 0
                ),
                "strategy_search_selected_solved": bool(search.get("selected_solved")),
                "strategy_search_projection_complete": bool(
                    search.get("path_route_projection_complete")
                ),
                "paper_reached": reached,
                "paper_equivalent_solved": solved,
                "stock_closure_status": "closed"
                if solved
                else "open"
                if reached
                else "not_reached",
                "classification": classification,
                "semantics": {
                    "builder_history_is_not_final_route_state": True,
                    "historical_quarantine_does_not_override_final_closure": True,
                    "canonical_edges_own_final_route_length": True,
                    "repair_successor_owns_final_route_state": True,
                    "paper_metric_is_existential_stock_topology_only": True,
                },
            }
        )
    routes.sort(key=lambda row: row["route_family_id"])
    return {
        "schema_version": ROUTE_RECONCILIATION_SCHEMA,
        "routes": routes,
        "route_count": len(routes),
        "builder_quarantined_route_count": sum(
            int(row["builder_quarantined_candidate_count"]) > 0 for row in routes
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
