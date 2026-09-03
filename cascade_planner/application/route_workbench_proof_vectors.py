"""Compile route-level proof axes for Workbench projections."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.route_workbench_edge_proof_vector import (
    PROOF_VECTOR_SCHEMA,
    edge_proof_vector,
)


def route_proof_vector(
    edges: list[Mapping[str, Any]],
    *,
    independent_source_groups: Any,
    closure_profile: str,
) -> dict[str, Any]:
    vectors = [dict(value.get("proof_vector") or {}) for value in edges]
    conditions = [str(value.get("conditions") or "missing") for value in vectors]
    completeness = [
        str(value.get("condition_completeness") or "missing") for value in vectors
    ]
    reactions = [str(value.get("reaction") or "untested") for value in vectors]
    identities = [str(value.get("identity") or "proposed") for value in vectors]
    source_groups = {str(value) for value in independent_source_groups if str(value)}
    conflicted = any(value.get("sources") == "conflicted" for value in vectors)
    all_source_exact = bool(identities) and all(
        value == "source_exact" for value in identities
    )
    all_reaction_validated = bool(reactions) and all(
        value in {"host_validated", "source_reaction_exact"}
        for value in reactions
    )
    all_conditions_complete = bool(completeness) and all(
        value == "complete" for value in completeness
    )
    all_procedures_bound = bool(vectors) and all(
        value.get("process") == "procedure_bound_candidate" for value in vectors
    )
    stock_state = {
        "exploration_closed": "benchmark_hit",
        "procurement_closed": "offer_verified",
        "in_house_closed": "in_house",
    }.get(closure_profile, "unknown")
    source_state = (
        "conflicted"
        if conflicted
        else "independent_2_plus"
        if len(source_groups) >= 2
        else "single_group"
        if source_groups
        else "none"
    )
    process_ready = bool(
        all_source_exact
        and all_reaction_validated
        and all_conditions_complete
        and all_procedures_bound
        and source_state in {"single_group", "independent_2_plus"}
        and stock_state in {"offer_verified", "in_house"}
    )
    return {
        "schema_version": PROOF_VECTOR_SCHEMA,
        "identity": (
            "all_source_exact"
            if all_source_exact
            else "all_materialized"
            if identities
            and all(value in {"materialized", "source_exact"} for value in identities)
            else "incomplete"
        ),
        "reaction": "all_validated" if all_reaction_validated else "incomplete",
        "conditions": (
            "missing"
            if not conditions or "missing" in conditions
            else "source_exact"
            if all(value == "source_exact" for value in conditions)
            else "mixed_supported"
        ),
        "condition_completeness": (
            "complete"
            if all_conditions_complete
            else "partial"
            if any(value == "partial" for value in completeness)
            else "missing"
        ),
        "condition_missing_required_groups": sorted(
            {
                str(group)
                for value in vectors
                for group in value.get("condition_missing_required_groups") or []
                if str(group)
            }
        ),
        "sources": source_state,
        "stock": stock_state,
        "process": "executable_candidate" if process_ready else "blocked",
        "semantics": {
            "weakest_edge_controls_route_axis": True,
            "configured_boundary_closure_is_not_process_readiness": True,
            "display_projection_grants_no_authority": True,
        },
    }


__all__ = ["PROOF_VECTOR_SCHEMA", "edge_proof_vector", "route_proof_vector"]
