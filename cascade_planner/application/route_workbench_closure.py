"""Structural route-program closure projection for the Workbench.

This axis answers only whether every step declared in a Director skeleton was
admitted into the canonical graph. It is intentionally independent from
reaction proof, source binding, conditions, and stock/procurement closure.
"""

from __future__ import annotations

from typing import Any, Mapping


def declared_program_closure(graph: Mapping[str, Any]) -> dict[str, Any]:
    edges = dict(graph.get("edges") or {})
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for hypothesis_id, raw in dict(graph.get("hypotheses") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        hypothesis = dict(raw)
        edge_id = f"edge:{str(hypothesis.get('edge_digest') or '')}"
        materialized = edge_id in edges
        for raw_origin in hypothesis.get("origin_records") or []:
            if not isinstance(raw_origin, Mapping):
                continue
            origin = dict(raw_origin)
            skeleton_id = str(origin.get("skeleton_id") or "")
            proposal_id = str(origin.get("proposal_id") or "")
            if not skeleton_id or not proposal_id:
                continue
            key = (
                str(origin.get("origin_ref") or ""),
                str(origin.get("route_family_id") or ""),
                skeleton_id,
            )
            grouped.setdefault(key, {})[proposal_id] = {
                "hypothesis_id": str(hypothesis_id),
                "step_id": proposal_id,
                "edge_id": edge_id if materialized else "",
                "materialized": materialized,
                "status": "materialized"
                if materialized
                else str(hypothesis.get("status") or "frontier_candidate"),
                "reasons": sorted(
                    {
                        str(value)
                        for value in hypothesis.get("admission_reasons") or []
                        if str(value)
                    }
                ),
            }

    programs = []
    for (origin_ref, family_id, skeleton_id), steps_by_id in grouped.items():
        steps = [steps_by_id[key] for key in sorted(steps_by_id)]
        rejected = [step for step in steps if step["status"] == "admission_rejected"]
        gaps = [step for step in steps if not step["materialized"]]
        graph_closed = bool(steps) and not gaps
        programs.append(
            {
                "origin_ref": origin_ref,
                "route_family_id": family_id,
                "skeleton_id": skeleton_id,
                "declared_step_count": len(steps),
                "materialized_step_count": sum(step["materialized"] for step in steps),
                "gap_step_count": len(gaps),
                "admission_rejected_step_count": len(rejected),
                "graph_closed": graph_closed,
                "state": (
                    "declared_route_graph_closed"
                    if graph_closed
                    else "admission_rejected_gap"
                    if rejected
                    else "unmaterialized_gap"
                ),
                "gap_steps": [
                    {
                        "step_id": str(step["step_id"]),
                        "status": str(step["status"]),
                        "reasons": list(step["reasons"]),
                    }
                    for step in gaps
                ],
            }
        )
    programs.sort(
        key=lambda row: (
            row["graph_closed"] is not True,
            -int(row["declared_step_count"]),
            str(row["skeleton_id"]),
            str(row["origin_ref"]),
        )
    )
    closed = [row for row in programs if row["graph_closed"]]
    return {
        "schema_version": "declared_route_program_closure.v1",
        "declared_program_count": len(programs),
        "graph_closed_program_count": len(closed),
        "graph_open_program_count": len(programs) - len(closed),
        "any_declared_route_graph_closed": bool(closed),
        "longest_graph_closed_step_count": max(
            (int(row["declared_step_count"]) for row in closed),
            default=0,
        ),
        "programs": programs,
        "semantics": {
            "graph_closure_requires_every_declared_step_materialized": True,
            "graph_closure_is_not_reaction_validation": True,
            "graph_closure_is_not_literature_grounding": True,
            "graph_closure_is_not_stock_or_procurement_closure": True,
            "open_programs_remain_visible_with_gap_reasons": True,
            "route_length_is_not_an_optimization_target": True,
        },
    }


__all__ = ["declared_program_closure"]
