"""Project execution validation outcomes into read-only capability feedback."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.execution_program_validations import (
    execution_validation_gate,
    strict_execution_validations,
)
from cascade_planner.application.execution_programs import (
    EXECUTION_PROGRAM_BUNDLE_SCHEMA,
)
from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
    with_program_innovation_digest,
)
from cascade_planner.application.program_validation_frontier_contracts import (
    ProgramValidationFrontierError,
    validate_program_validation_frontier_inputs,
)
from cascade_planner.application.program_validation_feedback_contracts import (
    ProgramValidationFeedbackError,
    collect_program_validation_feedback,
    validation_feedback_polarity,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


CAPABILITY_APPLICABILITY_FEEDBACK_SCHEMA = "capability_applicability_feedback.v1"
CAPABILITY_FEEDBACK_PROJECTION_SCHEMA = "capability_feedback_projection.v1"
CAPABILITY_FEEDBACK_ORACLE_SCHEMA = "capability_feedback_projection_oracle.v1"
CAPABILITY_FEEDBACK_SEMANTICS = {
    "projection_is_read_only": True,
    "valid_failure_and_inconclusive_records_are_retained": True,
    "feedback_scope_is_exact_boundary_only": True,
    "feedback_does_not_mutate_or_disable_capability_catalog": True,
    "only_accepted_success_can_enable_read_only_shadow": True,
    "feedback_cannot_grant_store_admission_or_route_completion": True,
}


class ExecutionCapabilityFeedbackError(ValueError):
    """Execution results cannot be safely projected into capability feedback."""


def compile_execution_capability_feedback(
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Retain exact-boundary success, failure, and inconclusive observations."""

    try:
        discovery_value = strict_program_innovation_object(discovery, "discovery")
        bundle_value = strict_program_innovation_object(bundle, "execution_bundle")
        validation_rows = strict_execution_validations(validations)
        validate_program_validation_frontier_inputs(
            discovery_value,
            bundle_value,
            expected_bundle_schema=EXECUTION_PROGRAM_BUNDLE_SCHEMA,
        )
    except (
        ProgramInnovationContractError,
        ProgramValidationFrontierError,
        ValueError,
    ) as exc:
        raise ExecutionCapabilityFeedbackError(str(exc)) from exc

    feedback: dict[str, dict[str, Any]] = {}
    try:
        collected = collect_program_validation_feedback(
            bundle_value,
            validation_rows,
            gate_factory=execution_validation_gate,
        )
    except ProgramValidationFeedbackError as exc:
        raise ExecutionCapabilityFeedbackError(str(exc)) from exc
    for observation in collected["observations"]:
        row = _feedback_row(
            observation["proposal"],
            observation["validation"],
            observation["audit"],
        )
        feedback[row["feedback_id"]] = row
    rejected = list(collected["rejected_validations"])
    polarities = [str(row["polarity"]) for row in feedback.values()]
    return with_program_innovation_digest(
        {
            "schema_version": CAPABILITY_FEEDBACK_PROJECTION_SCHEMA,
            "run_id": str(bundle_value.get("run_id") or ""),
            "route_id": str(discovery_value.get("route_id") or ""),
            "source_discovery_sha256": str(discovery_value["content_sha256"]),
            "source_bundle_sha256": str(bundle_value["content_sha256"]),
            "feedback": feedback,
            "rejected_validations": sorted(
                rejected,
                key=lambda row: (
                    str(row["validation_id"]),
                    str(row["program_id"]),
                    tuple(row["reasons"]),
                ),
            ),
            "counts": {
                "feedback_records": len(feedback),
                "positive": polarities.count("positive"),
                "negative": polarities.count("negative"),
                "inconclusive": polarities.count("inconclusive"),
                "rejected_validations": len(rejected),
                "catalog_mutations": 0,
            },
            "semantics": dict(CAPABILITY_FEEDBACK_SEMANTICS),
        }
    )


def execution_capability_feedback_oracle(
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    observed: Mapping[str, Any],
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Recompile a feedback projection and compare all scientific bindings."""

    try:
        rows = [dict(row) for row in validations]
        expected = compile_execution_capability_feedback(discovery, bundle, validations=rows)
        observed_value = strict_program_innovation_object(observed, "feedback")
    except (
        ExecutionCapabilityFeedbackError,
        ProgramInnovationContractError,
        TypeError,
        ValueError,
    ) as exc:
        return _oracle_result(
            False,
            {"inputs_reprojectable": False},
            [f"feedback_inputs_invalid:{type(exc).__name__}"],
            "",
            "",
        )
    material = dict(observed_value)
    observed_digest = str(material.pop("content_sha256", ""))
    checks = {
        "inputs_reprojectable": True,
        "schema_equal": observed_value.get("schema_version")
        == CAPABILITY_FEEDBACK_PROJECTION_SCHEMA,
        "content_digest_valid": observed_digest == strict_canonical_json_sha256(material),
        "projection_equal": observed_value == expected,
        "authority_semantics_equal": observed_value.get("semantics")
        == CAPABILITY_FEEDBACK_SEMANTICS,
    }
    reasons = [key for key, accepted in checks.items() if accepted is not True]
    return _oracle_result(
        not reasons,
        checks,
        reasons,
        str(expected["content_sha256"]),
        observed_digest,
    )


def _feedback_row(
    proposal: Mapping[str, Any],
    validation: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    outcome = str(validation.get("outcome_status") or "")
    polarity = validation_feedback_polarity(outcome, accepted=audit.get("accepted") is True)
    identity = {
        "validation_id": validation.get("validation_id"),
        "validation_sha256": validation.get("content_sha256"),
        "program_id": proposal.get("program_id"),
        "capability_id": proposal.get("source_capability_id"),
    }
    feedback_id = "capability-feedback:" + strict_canonical_json_sha256(identity)[:24]
    return with_program_innovation_digest(
        {
            "schema_version": CAPABILITY_APPLICABILITY_FEEDBACK_SCHEMA,
            "feedback_id": feedback_id,
            "polarity": polarity,
            "outcome_status": outcome,
            "program_id": str(proposal.get("program_id") or ""),
            "capability_id": str(proposal.get("source_capability_id") or ""),
            "source_capability_sha256": str(proposal.get("source_capability_sha256") or ""),
            "execution_domain": str(proposal.get("execution_domain") or ""),
            "validation_id": str(validation.get("validation_id") or ""),
            "source_validation_sha256": str(validation.get("content_sha256") or ""),
            "applicability_scope": {
                "input_state_ids": list(proposal.get("input_state_ids") or []),
                "output_state_ids": list(proposal.get("output_state_ids") or []),
                "operation_sequence_sha256": str(validation.get("operation_sequence_sha256") or ""),
                "generalization_scope": "exact_boundary_only",
            },
            "required_check_results": dict(validation.get("required_check_results") or {}),
            "cofactor_carrier_ledger_closed": bool(
                validation.get("cofactor_carrier_ledger_closed")
            ),
            "evidence_tier": str(validation.get("evidence_tier") or ""),
            "claim_refs": list(validation.get("claim_refs") or []),
            "condition_record_ids": list(validation.get("condition_record_ids") or []),
            "actor_identity_refs": list(validation.get("actor_identity_refs") or []),
            "outcome_metrics": dict(validation.get("outcome_metrics") or {}),
            "grants_validation": audit.get("accepted") is True,
            "candidate_disposition": (
                "read_only_shadow_eligible"
                if audit.get("accepted") is True
                else "exploration_visible"
            ),
            "catalog_mutated": False,
            "capability_disabled": False,
            "eligible_for_route_completion": False,
        }
    )


def _oracle_result(
    accepted: bool,
    checks: Mapping[str, bool],
    reasons: list[str],
    expected_digest: str,
    observed_digest: str,
) -> dict[str, Any]:
    return with_program_innovation_digest(
        {
            "schema_version": CAPABILITY_FEEDBACK_ORACLE_SCHEMA,
            "accepted": accepted,
            "checks": dict(checks),
            "reasons": reasons,
            "expected_feedback_sha256": expected_digest,
            "observed_feedback_sha256": observed_digest,
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_scientific_authority": True,
            },
        }
    )


__all__ = [
    "CAPABILITY_APPLICABILITY_FEEDBACK_SCHEMA",
    "CAPABILITY_FEEDBACK_ORACLE_SCHEMA",
    "CAPABILITY_FEEDBACK_PROJECTION_SCHEMA",
    "CAPABILITY_FEEDBACK_SEMANTICS",
    "ExecutionCapabilityFeedbackError",
    "compile_execution_capability_feedback",
    "execution_capability_feedback_oracle",
]
