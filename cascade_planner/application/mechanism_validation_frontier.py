"""Compile unvalidated mechanism Programs into exact-boundary experiment plans."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.mechanism_program_validations import (
    MECHANISM_PROGRAM_VALIDATION_SCHEMA,
    mechanism_signature_sha256,
)
from cascade_planner.application.mechanism_programs import (
    MECHANISM_PROGRAM_BUNDLE_SCHEMA,
    MechanismProgramError,
)
from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
    with_program_innovation_digest,
)
from cascade_planner.application.program_validation_frontier_contracts import (
    ProgramValidationFrontierError,
    program_validation_state_snapshots,
    validate_program_validation_frontier_inputs,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


MECHANISM_VALIDATION_FRONTIER_SCHEMA = "mechanism_validation_frontier.v1"
MECHANISM_VALIDATION_PLAN_SCHEMA = "mechanism_validation_plan.v1"


def compile_mechanism_validation_frontier(
    graph: Mapping[str, Any],
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Plan experiments for each restitched mechanism Program lacking success."""

    try:
        graph_value = strict_program_innovation_object(graph, "graph")
        discovery_value = strict_program_innovation_object(discovery, "discovery")
        bundle_value = strict_program_innovation_object(bundle, "mechanism_bundle")
        validate_program_validation_frontier_inputs(
            discovery_value,
            bundle_value,
            expected_bundle_schema=MECHANISM_PROGRAM_BUNDLE_SCHEMA,
        )
    except (ProgramInnovationContractError, ProgramValidationFrontierError) as exc:
        raise MechanismProgramError(str(exc)) from exc

    candidates = {
        str(row.get("candidate_id") or ""): dict(row)
        for row in discovery_value.get("candidates") or []
        if isinstance(row, dict) and row.get("candidate_kind") == "mechanism_one_hop"
    }
    plans: dict[str, dict[str, Any]] = {}
    for program_id, raw_proposal in sorted(
        dict(bundle_value.get("program_proposals") or {}).items()
    ):
        proposal = dict(raw_proposal)
        gate = dict(proposal.get("validation_plan") or {})
        if gate.get("accepted") is True:
            continue
        candidate_id = str(proposal.get("source_candidate_id") or "")
        if candidate_id not in candidates:
            raise MechanismProgramError("validation_frontier_candidate_missing")
        required_checks = list(gate.get("required_checks") or [])
        plan_id = (
            "mechanism-plan:"
            + strict_canonical_json_sha256(
                {"program_id": program_id, "candidate_id": candidate_id}
            )[:24]
        )
        plan = {
            "schema_version": MECHANISM_VALIDATION_PLAN_SCHEMA,
            "plan_id": plan_id,
            "program_id": str(program_id),
            "innovation_id": str(proposal.get("source_innovation_id") or ""),
            "candidate_id": candidate_id,
            "canonical_context": {
                "replaced_edge_ids": list(
                    proposal.get("equivalent_reference_span") or []
                ),
            },
            "exact_boundary": {
                "input_states": _state_snapshots(
                    graph_value, proposal.get("input_state_ids") or []
                ),
                "output_states": _state_snapshots(
                    graph_value, proposal.get("output_state_ids") or []
                ),
            },
            "mechanism_hypothesis": {
                "anchor": dict(proposal.get("anchor") or {}),
                "mechanistic_rationale": str(proposal.get("mechanistic_rationale") or ""),
                "elementary_steps": list(proposal.get("elementary_steps") or []),
                "mechanism_signature_sha256": mechanism_signature_sha256(proposal),
            },
            "required_checks": required_checks,
            "evidence_frontier": {
                "anchor_source_refs": list(
                    dict(proposal.get("anchor") or {}).get("source_refs") or []
                ),
                "anchor_reports_extrapolated_reaction": False,
                "missing_gate": "specialized_mechanism_validation",
            },
            "required_output_contract": {
                "schema_version": MECHANISM_PROGRAM_VALIDATION_SCHEMA,
                "required_check_ids": [str(row.get("check_id") or "") for row in required_checks],
                "must_bind_exact_input_and_output_states": True,
                "must_bind_mechanism_signature": True,
                "must_bind_claim_condition_and_analytical_records": True,
                "must_record_success_failure_or_inconclusive": True,
                "net_transform_success_does_not_prove_elementary_mechanism": True,
            },
            "status": "experiment_required",
            "grants_validation": False,
            "eligible_for_shadow_optimizer": False,
            "eligible_for_route_completion": False,
        }
        plans[plan_id] = with_program_innovation_digest(plan)

    return with_program_innovation_digest(
        {
            "schema_version": MECHANISM_VALIDATION_FRONTIER_SCHEMA,
            "run_id": str(graph_value.get("run_id") or ""),
            "route_id": str(discovery_value.get("route_id") or ""),
            "source_discovery_sha256": str(discovery_value["content_sha256"]),
            "source_bundle_sha256": str(bundle_value["content_sha256"]),
            "plans": plans,
            "counts": {
                "experiment_required": len(plans),
                "validation_granted": 0,
            },
            "semantics": {
                "plans_are_read_only": True,
                "anchor_source_does_not_validate_extrapolation": True,
                "net_transform_observation_is_not_full_mechanism_proof": True,
                "plans_do_not_grant_store_admission_or_route_completion": True,
                "target_names_are_not_plan_inputs": True,
            },
        }
    )


def _state_snapshots(graph: Mapping[str, Any], state_ids: list[Any]) -> list[dict[str, str]]:
    try:
        return program_validation_state_snapshots(graph, state_ids)
    except ProgramValidationFrontierError as exc:
        raise MechanismProgramError(str(exc)) from exc


__all__ = [
    "MECHANISM_VALIDATION_FRONTIER_SCHEMA",
    "MECHANISM_VALIDATION_PLAN_SCHEMA",
    "compile_mechanism_validation_frontier",
]
