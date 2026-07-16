"""Read-only orchestration from route innovation discovery to Program drafts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.orchestration.program_innovation_materials import (
    compile_route_program_innovation_materials,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def review_route_program_innovations(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    reported_candidate_packs: Iterable[Mapping[str, Any]] = (),
    experience_library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile discoveries into Program drafts without canonical graph writes."""

    materials = compile_route_program_innovation_materials(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        mechanism_proposals=mechanism_proposals,
        validations=validations,
        reported_candidate_packs=reported_candidate_packs,
        experience_library=experience_library,
    )
    result = {
        "schema_version": "route_program_innovation_review.v1",
        "run_id": str(graph.get("run_id") or ""),
        "route_id": str(materials["route"].get("route_id") or ""),
        "discovery": materials["discovery"],
        "program_experience": dict(materials["discovery"].get("program_experience") or {}),
        "program_bundle": materials["bundle"],
        "mechanism_program_bundle": materials["mechanism_bundle"],
        "mechanism_validation_frontier": materials["mechanism_validation_frontier"],
        "mechanism_experiment_feedback": materials["mechanism_experiment_feedback"],
        "mechanism_feedback_oracle": materials["mechanism_feedback_oracle"],
        "execution_program_bundle": materials["execution_bundle"],
        "validation_frontier": materials["validation_frontier"],
        "execution_validation_frontier": materials["execution_validation_frontier"],
        "execution_capability_feedback": materials["execution_capability_feedback"],
        "execution_feedback_oracle": materials["execution_feedback_oracle"],
        "experimental_claims": materials["experimental_claims"],
        "experimental_claims_oracle": materials["experimental_claims_oracle"],
        "capability_calibration": materials["capability_calibration"],
        "capability_calibration_oracle": materials["capability_calibration_oracle"],
        "experimental_work_frontier": materials["experimental_work_frontier"],
        "experimental_work_frontier_oracle": materials[
            "experimental_work_frontier_oracle"
        ],
        "program_route_candidates": materials["program_route_candidates"],
        "program_optimizer": materials["program_optimizer"],
        "program_optimizer_oracle": materials["program_optimizer_oracle"],
        "oracle": materials["oracle"],
        "mechanism_oracle": materials["mechanism_oracle"],
        "execution_oracle": materials["execution_oracle"],
        "semantics": {
            "read_only_review": True,
            "canonical_graph_not_mutated": True,
            "program_drafts_do_not_grant_route_completion": True,
            "reported_candidate_packs_are_review_only": True,
            "mechanism_one_hop_requires_full_route_restitch": True,
            "unrestitched_mechanism_stays_in_discovery": True,
            "restitched_mechanism_requires_exact_validation_for_shadow": True,
            "mechanism_success_does_not_create_canonical_reaction_proof": True,
            "mechanism_failure_and_uncertainty_remain_feedback": True,
            "whole_cell_and_hybrid_are_exploration_only_until_validated": True,
            "execution_validation_enables_only_read_only_shadow": True,
            "experimental_failures_are_retained_as_exact_boundary_feedback": True,
            "feedback_does_not_mutate_or_disable_capability_catalog": True,
            "experimental_claims_are_exact_boundary_observations_only": True,
            "claim_projection_does_not_create_canonical_reaction_proof": True,
            "calibration_dirty_domains_are_read_only_recompute_hints": True,
            "experimental_work_is_bound_to_the_single_canonical_frontier": True,
            "experiment_results_require_explicit_domain_gate_release": True,
            "execution_programs_have_no_store_admission_path": True,
            "self_evolution_memory_is_proposal_ranking_only": True,
        },
    }
    result["content_sha256"] = strict_canonical_json_sha256(result)
    return result


__all__ = ["review_route_program_innovations"]
