"""Read-only projection of complete Director skeletons for the Workbench."""

from __future__ import annotations

import re
from typing import Any, Mapping

from cascade_planner.application.route_workbench_route_rows import PROOF_VISUALS
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


MAX_VISIBLE_PLANNED_ROUTES = 8


def planned_route_rows(
    graph: Mapping[str, Any],
    *,
    selected_routes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group Director hypotheses by skeleton without granting route authority."""

    edges = dict(graph.get("edges") or {})
    families = dict(graph.get("route_families") or {})
    family_by_alias = {
        str(alias): str(family_id)
        for family_id, family in families.items()
        if isinstance(family, Mapping)
        for alias in family.get("aliases") or []
        if str(alias)
    }
    selected_edge_sets = {
        frozenset(str(edge_id) for edge_id in route.get("edge_ids") or [])
        for route in selected_routes
    }
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for hypothesis_id, raw in dict(graph.get("hypotheses") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        hypothesis = dict(raw)
        edge_id = f"edge:{str(hypothesis.get('edge_digest') or '')}"
        materialized = edge_id in edges
        origins = [
            dict(value)
            for value in hypothesis.get("origin_records") or []
            if isinstance(value, Mapping)
        ]
        for origin in origins:
            skeleton_id = str(origin.get("skeleton_id") or "")
            proposal_id = str(origin.get("proposal_id") or "")
            if not skeleton_id or not proposal_id:
                continue
            family_id = _family_id(
                origin,
                hypothesis=hypothesis,
                families=families,
                family_by_alias=family_by_alias,
            )
            if not family_id:
                continue
            grouped.setdefault((family_id, skeleton_id), {})[proposal_id] = {
                "hypothesis_id": str(hypothesis_id),
                "step_id": proposal_id,
                "product_smiles": str(hypothesis.get("product_smiles") or ""),
                "precursor_smiles": list(hypothesis.get("precursor_smiles") or []),
                "transformation_hypothesis": str(
                    origin.get("transformation_hypothesis") or ""
                ),
                "status": "materialized" if materialized else str(
                    hypothesis.get("status") or "frontier_candidate"
                ),
                "edge_id": edge_id if materialized else "",
                "admission_reasons": list(
                    hypothesis.get("admission_reasons") or []
                ),
                "origin_kind": str(origin.get("origin_kind") or ""),
            }

    rows = [
        row
        for family_and_skeleton, steps_by_id in grouped.items()
        if (
            row := _planned_route_row(
                family_and_skeleton,
                steps_by_id,
                edges=edges,
                families=families,
                selected_edge_sets=selected_edge_sets,
            )
        )
    ]
    rows.sort(
        key=lambda row: (
            -int(row["declared_step_count"]),
            int(row["admission_rejected_step_count"]),
            str(row["route_id"]),
        )
    )
    return {
        str(row["route_id"]): row
        for row in rows[:MAX_VISIBLE_PLANNED_ROUTES]
    }


def _family_id(
    origin: Mapping[str, Any],
    *,
    hypothesis: Mapping[str, Any],
    families: Mapping[str, Any],
    family_by_alias: Mapping[str, str],
) -> str:
    alias = str(origin.get("route_family_id") or "")
    return family_by_alias.get(alias, "") or next(
        (
            str(value)
            for value in hypothesis.get("route_family_ids") or []
            if str(value) in families
        ),
        "",
    )


def _planned_route_row(
    family_and_skeleton: tuple[str, str],
    steps_by_id: Mapping[str, dict[str, Any]],
    *,
    edges: Mapping[str, Any],
    families: Mapping[str, Any],
    selected_edge_sets: set[frozenset[str]],
) -> dict[str, Any] | None:
    if len(steps_by_id) < 2:
        return None
    family_id, skeleton_id = family_and_skeleton
    steps = [steps_by_id[key] for key in sorted(steps_by_id, key=_natural_step_key)]
    materialized_edge_ids = [
        str(step["edge_id"])
        for step in steps
        if str(step.get("edge_id") or "") in edges
    ]
    unresolved_steps = [step for step in steps if step.get("status") != "materialized"]
    if not unresolved_steps and frozenset(materialized_edge_ids) in selected_edge_sets:
        return None
    admission_rejected_count = sum(
        step.get("status") == "admission_rejected" for step in steps
    )
    identity = {
        "route_family_id": family_id,
        "skeleton_id": skeleton_id,
        "step_ids": [str(step["step_id"]) for step in steps],
    }
    warnings = ["planner_route_is_advisory_only"]
    if unresolved_steps:
        warnings.append("planner_route_contains_unmaterialized_steps")
    if admission_rejected_count:
        warnings.append("planner_route_contains_admission_rejected_steps")
    family = dict(families.get(family_id) or {})
    return {
        "route_id": f"planned-route:{strict_canonical_json_sha256(identity)[:24]}",
        "route_family_id": family_id,
        "skeleton_id": skeleton_id,
        "strategy": str(family.get("strategy") or ""),
        "steps": steps,
        "edge_ids": materialized_edge_ids,
        "declared_step_count": len(steps),
        "materialized_step_count": len(materialized_edge_ids),
        "unmaterialized_step_count": len(unresolved_steps),
        "admission_rejected_step_count": admission_rejected_count,
        "proof_level": 0 if unresolved_steps else 1,
        "proof_color": PROOF_VISUALS[0]["color"],
        "complete": False,
        "listed": True,
        "advisory_only": True,
        "warning_codes": warnings,
        "semantics": {
            "not_proof_portfolio_route": True,
            "never_counts_toward_acceptance": True,
            "retains_rejected_steps_for_review": True,
        },
    }


def _natural_step_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


__all__ = ["MAX_VISIBLE_PLANNED_ROUTES", "planned_route_rows"]
