"""Route-level rows for the bounded retrosynthesis workbench."""
from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.application.product_profiles import (
    PRODUCT_PROFILE_ORDER,
    product_profiles,
)
from cascade_planner.application.route_workbench_inspectors import route_proof_vector


PROOF_VISUALS: dict[int, dict[str, str]] = {
    0: {"name": "L0_hypothesis", "color": "#e76f51", "tone": "proposal"},
    1: {"name": "L1_structural_materialized", "color": "#8b5cf6", "tone": "materialized"},
    2: {"name": "L2_reaction_validated", "color": "#3b82f6", "tone": "validated"},
    3: {"name": "L3_exact_source", "color": "#0f9f8f", "tone": "supported"},
    4: {"name": "L4_procurement_ready", "color": "#16a34a", "tone": "closed"},
}


def closure_profile(*, accepted: bool, stock_boundary: str) -> str:
    if not accepted:
        return "unresolved"
    return {
        "benchmark_search": "exploration_closed",
        "procurement": "procurement_closed",
        "in_house": "in_house_closed",
    }.get(stock_boundary, "configured_boundary_closed")


def route_row(
    route: Mapping[str, Any],
    *,
    edge_rows: Mapping[str, Mapping[str, Any]],
    deficits: list[dict[str, Any]],
    stock_boundary: str,
) -> dict[str, Any]:
    route_id = str(route.get("route_id") or "")
    level = max(0, min(4, int(route.get("minimum_edge_proof_level") or 0)))
    configured_boundary_closed = route.get("complete") is True
    profile = closure_profile(
        accepted=configured_boundary_closed,
        stock_boundary=stock_boundary,
    )
    if configured_boundary_closed:
        stage = "stock_closed"
    elif route.get("all_edges_proven") is True and level >= 2:
        stage = "reaction_validated"
    elif route.get("edge_ids"):
        stage = "expanded"
    else:
        stage = "hypothesis"
    route_deficits = [
        value
        for value in deficits
        if route_id in {
            str(value.get("route_id") or ""),
            *(str(item) for item in value.get("route_ids") or []),
        }
    ]
    badges = [stage.replace("_", "-")]
    source_kinds = sorted(
        {
            str(kind)
            for edge_id in route.get("edge_ids") or []
            for kind in dict(edge_rows.get(str(edge_id)) or {}).get("source_kinds") or []
        }
    )
    badges.extend(f"source:{value}" for value in source_kinds)
    if route.get("pareto_optimal") is True:
        badges.append("pareto")
    if configured_boundary_closed:
        badges.append("configured-boundary-closed")
    reported_in_source = route.get("reported_in_source") is True
    if reported_in_source:
        badges.append("reported-candidate")
    selected_edges = [
        dict(edge_rows.get(str(edge_id)) or {})
        for edge_id in route.get("edge_ids") or []
    ]
    proof_level_counts = {
        str(proof_level): sum(
            int(edge.get("proof_level") or 0) == proof_level
            for edge in selected_edges
        )
        for proof_level in range(5)
        if any(
            int(edge.get("proof_level") or 0) == proof_level
            for edge in selected_edges
        )
    }
    inactive_facts = {
        (str(value.get("subject_kind") or ""), str(value.get("subject_id") or "")): dict(value)
        for edge in selected_edges
        for value in edge.get("inactive_facts") or []
        if isinstance(value, Mapping)
    }
    badges.extend(
        f"fact-{value.get('status') or 'inactive'}" for value in inactive_facts.values()
    )
    proof_vector = route_proof_vector(
        selected_edges,
        independent_source_groups=route.get("independent_source_groups") or [],
        closure_profile=profile,
    )
    acceptance_profiles = product_profiles(proof_vector, closure_profile=profile)
    if proof_vector["conditions"] == "missing":
        badges.append("conditions-missing")
    biocatalytic_step_count = int(route.get("biocatalytic_step_count") or 0)
    biocatalytic_count = int(route.get("biocatalytic_superstep_count") or 0)
    mechanism_count = int(route.get("mechanism_extrapolation_count") or 0)
    if biocatalytic_step_count:
        badges.append("biocatalytic-step")
    if biocatalytic_count:
        badges.append("biocatalytic-superstep")
    if mechanism_count:
        badges.append("mechanism-extrapolation")
    edge_ids = [str(value) for value in route.get("edge_ids") or []]
    return {
        "route_id": route_id,
        "route_family_id": str(route.get("route_family_id") or ""),
        "strategy": str(route.get("strategy") or ""),
        "stage": stage,
        "proof_level": level,
        "proof_name": PROOF_VISUALS[level]["name"],
        "proof_color": PROOF_VISUALS[level]["color"],
        "edge_ids": edge_ids,
        "leaf_molecule_ids": [str(value) for value in route.get("leaf_molecule_ids") or []],
        "root_edge_ids": [str(value) for value in route.get("root_edge_ids") or []],
        "module_selections": dict(route.get("module_selections") or {}),
        "complete": route.get("complete") is True,
        "all_edges_proven": route.get("all_edges_proven") is True,
        "unproven_edge_ids": [str(value) for value in route.get("unproven_edge_ids") or []],
        "proof_level_counts": proof_level_counts,
        "reported_step_count": int(route.get("reported_step_count") or 0),
        "planner_hypothesis_step_count": int(route.get("planner_hypothesis_step_count") or 0),
        "physical_step_count": int(route.get("physical_step_count") or len(edge_ids)),
        "chemical_step_equivalent_count": int(
            route.get("chemical_step_equivalent_count") or len(edge_ids)
        ),
        "net_step_savings": int(route.get("net_step_savings") or 0),
        "biocatalytic_superstep_count": biocatalytic_count,
        "biocatalytic_step_count": biocatalytic_step_count,
        "mechanism_extrapolation_count": mechanism_count,
        "unvalidated_biocatalytic_edge_ids": list(
            route.get("unvalidated_biocatalytic_edge_ids") or []
        ),
        "route_innovation_summary": _copy_json(route.get("route_innovation_summary") or {}),
        "reported_in_source": reported_in_source,
        "reported_source_refs": sorted(
            {str(value) for value in route.get("reported_source_refs") or [] if str(value)}
        ),
        "warning_codes": (
            ["reported_route_contains_unresolved_edges"]
            if reported_in_source and route.get("all_edges_proven") is not True
            else []
        ),
        "configured_boundary_closed": configured_boundary_closed,
        "stock_boundary": stock_boundary,
        "closure_profile": profile,
        "search_closed": acceptance_profiles["exploration_closed"],
        "reaction_validated": acceptance_profiles["reaction_validated"],
        "literature_grounded": acceptance_profiles["literature_grounded"],
        "procurement_closed": acceptance_profiles["procurement_closed"],
        "process_ready": acceptance_profiles["process_ready"],
        "condition_complete": acceptance_profiles["condition_complete"],
        "acceptance_profiles": acceptance_profiles,
        "achieved_profiles": [
            value for value in PRODUCT_PROFILE_ORDER if acceptance_profiles[value]
        ],
        "proof_vector": proof_vector,
        "inactive_fact_count": len(inactive_facts),
        "inactive_facts": [inactive_facts[key] for key in sorted(inactive_facts)],
        "stock_closure_rate": float(route.get("stock_closure_rate") or 0.0),
        "independent_source_groups": list(route.get("independent_source_groups") or []),
        "risk_score": float(route.get("risk_score") or 0.0),
        "convergence_score": float(route.get("convergence_score") or 0.0),
        "deficit_count": len(route_deficits),
        "badges": sorted(set(badges)),
    }


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


__all__ = ["PROOF_VISUALS", "closure_profile", "route_row"]
