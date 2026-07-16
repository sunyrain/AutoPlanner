"""Compile whole-cell and hybrid Program proposals into experiment plans."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.execution_program_validations import (
    EXECUTION_PROGRAM_VALIDATION_SCHEMA,
    execution_operation_sequence_sha256,
)
from cascade_planner.application.execution_programs import (
    EXECUTION_PROGRAM_BUNDLE_SCHEMA,
    ExecutionProgramError,
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


EXECUTION_VALIDATION_FRONTIER_SCHEMA = "execution_validation_frontier.v1"
EXECUTION_VALIDATION_PLAN_SCHEMA = "execution_validation_plan.v1"


def compile_execution_validation_frontier(
    graph: Mapping[str, Any],
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Plan exact-boundary experiments for every unvalidated execution Program."""

    try:
        graph_value = strict_program_innovation_object(graph, "graph")
        discovery_value = strict_program_innovation_object(discovery, "discovery")
        bundle_value = strict_program_innovation_object(bundle, "execution_bundle")
        validate_program_validation_frontier_inputs(
            discovery_value,
            bundle_value,
            expected_bundle_schema=EXECUTION_PROGRAM_BUNDLE_SCHEMA,
        )
    except (ProgramInnovationContractError, ProgramValidationFrontierError) as exc:
        raise ExecutionProgramError(str(exc)) from exc

    candidates = {
        str(row.get("candidate_id") or ""): dict(row)
        for row in discovery_value.get("candidates") or []
        if isinstance(row, dict)
        and row.get("candidate_kind") == "program_execution_window"
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
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ExecutionProgramError("validation_frontier_candidate_missing")
        capability = dict(candidate.get("execution_capability") or {})
        boundary = dict(candidate.get("boundary") or {})
        plan_id = "execution-plan:" + strict_canonical_json_sha256(
            {"program_id": program_id, "candidate_id": candidate_id}
        )[:24]
        required_checks = [
            {
                "check_id": str(check_id),
                "required": True,
                "result_contract": "boolean",
            }
            for check_id in gate.get("required_checks") or []
        ]
        plan = {
            "schema_version": EXECUTION_VALIDATION_PLAN_SCHEMA,
            "plan_id": plan_id,
            "program_id": str(program_id),
            "candidate_id": candidate_id,
            "capability_id": str(proposal.get("source_capability_id") or ""),
            "source_capability_sha256": str(
                proposal.get("source_capability_sha256") or ""
            ),
            "execution_domain": str(proposal.get("execution_domain") or ""),
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
                "principal_input_smiles": str(
                    boundary.get("precursor_smiles") or ""
                ),
                "principal_output_smiles": str(boundary.get("product_smiles") or ""),
            },
            "execution_matrix": {
                "actors": dict(proposal.get("actors") or {}),
                "operation_blueprints": list(
                    proposal.get("operation_blueprints") or []
                ),
                "cofactor_and_carrier_ledger": dict(
                    proposal.get("cofactor_and_carrier_ledger") or {}
                ),
            },
            "selectivity_constraints": list(
                proposal.get("selectivity_constraints") or []
            ),
            "required_checks": required_checks,
            "evidence_frontier": {
                "precedent_refs": list(
                    dict(proposal.get("claim_refs") or {}).get("precedent_refs")
                    or []
                ),
                "substrate_scope_basis": str(
                    capability.get("substrate_scope_basis") or ""
                ),
                "exact_substrate_claim_refs": [],
                "missing_gate": "specialized_execution_validation",
            },
            "required_output_contract": {
                "schema_version": EXECUTION_PROGRAM_VALIDATION_SCHEMA,
                "operation_sequence_sha256": execution_operation_sequence_sha256(
                    proposal
                ),
                "required_check_ids": [
                    row["check_id"] for row in required_checks
                ],
                "must_bind_exact_input_and_output_states": True,
                "must_bind_capability_digest_and_execution_domain": True,
                "must_bind_actor_claim_and_condition_refs": True,
                "must_record_success_failure_or_inconclusive": True,
                "must_close_required_cofactor_or_carrier_ledger": True,
            },
            "status": "experiment_required",
            "grants_validation": False,
            "eligible_for_shadow_optimizer": False,
            "eligible_for_route_completion": False,
        }
        plans[plan_id] = with_program_innovation_digest(plan)

    return with_program_innovation_digest(
        {
            "schema_version": EXECUTION_VALIDATION_FRONTIER_SCHEMA,
            "run_id": str(graph_value.get("run_id") or ""),
            "route_id": str(discovery_value.get("route_id") or ""),
            "source_discovery_sha256": str(discovery_value["content_sha256"]),
            "source_bundle_sha256": str(bundle_value["content_sha256"]),
            "plans": plans,
            "counts": {
                "experiment_required": len(plans),
                "whole_cell": sum(
                    row["execution_domain"] == "whole_cell" for row in plans.values()
                ),
                "hybrid": sum(
                    row["execution_domain"] == "hybrid" for row in plans.values()
                ),
                "validation_granted": 0,
            },
            "semantics": {
                "plans_are_read_only": True,
                "exact_boundary_experiments_are_required": True,
                "success_failure_and_inconclusive_outcomes_are_retained": True,
                "plans_do_not_grant_store_admission_or_route_completion": True,
                "target_names_are_not_plan_inputs": True,
            },
        }
    )


def _state_snapshots(
    graph: Mapping[str, Any], state_ids: list[Any]
) -> list[dict[str, str]]:
    try:
        return program_validation_state_snapshots(graph, state_ids)
    except ProgramValidationFrontierError as exc:
        raise ExecutionProgramError(str(exc)) from exc


__all__ = [
    "EXECUTION_VALIDATION_FRONTIER_SCHEMA",
    "EXECUTION_VALIDATION_PLAN_SCHEMA",
    "compile_execution_validation_frontier",
]
