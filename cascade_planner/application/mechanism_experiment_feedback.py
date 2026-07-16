"""Project mechanism validation outcomes into read-only experiment feedback."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.mechanism_program_validations import (
    mechanism_validation_gate,
    strict_mechanism_validations,
)
from cascade_planner.application.mechanism_programs import (
    MECHANISM_PROGRAM_BUNDLE_SCHEMA,
)
from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
    with_program_innovation_digest,
)
from cascade_planner.application.program_validation_feedback_contracts import (
    ProgramValidationFeedbackError,
    collect_program_validation_feedback,
    validation_feedback_polarity,
)
from cascade_planner.application.program_validation_frontier_contracts import (
    ProgramValidationFrontierError,
    validate_program_validation_frontier_inputs,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


MECHANISM_EXPERIMENT_FEEDBACK_SCHEMA = "mechanism_experiment_feedback.v1"
MECHANISM_FEEDBACK_PROJECTION_SCHEMA = "mechanism_feedback_projection.v1"
MECHANISM_FEEDBACK_ORACLE_SCHEMA = "mechanism_feedback_projection_oracle.v1"
MECHANISM_FEEDBACK_SEMANTICS = {
    "projection_is_read_only": True,
    "valid_failure_and_inconclusive_records_are_retained": True,
    "feedback_scope_is_exact_boundary_only": True,
    "net_transform_success_does_not_prove_elementary_mechanism": True,
    "anchor_source_does_not_report_the_extrapolated_reaction": True,
    "only_accepted_success_can_enable_read_only_shadow": True,
    "feedback_cannot_create_reaction_proof_store_admission_or_completion": True,
}


class MechanismExperimentFeedbackError(ValueError):
    """Mechanism experiment results cannot be safely projected."""


def compile_mechanism_experiment_feedback(
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Retain exact-boundary mechanism success, failure, and uncertainty."""

    try:
        discovery_value = strict_program_innovation_object(discovery, "discovery")
        bundle_value = strict_program_innovation_object(bundle, "mechanism_bundle")
        validation_rows = strict_mechanism_validations(validations)
        validate_program_validation_frontier_inputs(
            discovery_value,
            bundle_value,
            expected_bundle_schema=MECHANISM_PROGRAM_BUNDLE_SCHEMA,
        )
        collected = collect_program_validation_feedback(
            bundle_value,
            validation_rows,
            gate_factory=mechanism_validation_gate,
        )
    except (
        ProgramInnovationContractError,
        ProgramValidationFeedbackError,
        ProgramValidationFrontierError,
        ValueError,
    ) as exc:
        raise MechanismExperimentFeedbackError(str(exc)) from exc

    feedback: dict[str, dict[str, Any]] = {}
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
            "schema_version": MECHANISM_FEEDBACK_PROJECTION_SCHEMA,
            "run_id": str(bundle_value.get("run_id") or ""),
            "route_id": str(discovery_value.get("route_id") or ""),
            "source_discovery_sha256": str(discovery_value["content_sha256"]),
            "source_bundle_sha256": str(bundle_value["content_sha256"]),
            "feedback": feedback,
            "rejected_validations": rejected,
            "counts": {
                "feedback_records": len(feedback),
                "positive": polarities.count("positive"),
                "negative": polarities.count("negative"),
                "inconclusive": polarities.count("inconclusive"),
                "rejected_validations": len(rejected),
                "reaction_proofs_created": 0,
                "store_mutations": 0,
            },
            "semantics": dict(MECHANISM_FEEDBACK_SEMANTICS),
        }
    )


def mechanism_experiment_feedback_oracle(
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    observed: Mapping[str, Any],
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Recompile mechanism feedback and compare all scientific bindings."""

    try:
        rows = [dict(row) for row in validations]
        expected = compile_mechanism_experiment_feedback(discovery, bundle, validations=rows)
        observed_value = strict_program_innovation_object(observed, "mechanism_feedback")
    except (
        MechanismExperimentFeedbackError,
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
        == MECHANISM_FEEDBACK_PROJECTION_SCHEMA,
        "content_digest_valid": observed_digest == strict_canonical_json_sha256(material),
        "projection_equal": observed_value == expected,
        "authority_semantics_equal": observed_value.get("semantics")
        == MECHANISM_FEEDBACK_SEMANTICS,
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
        "innovation_id": proposal.get("source_innovation_id"),
    }
    feedback_id = "mechanism-feedback:" + strict_canonical_json_sha256(identity)[:24]
    return with_program_innovation_digest(
        {
            "schema_version": MECHANISM_EXPERIMENT_FEEDBACK_SCHEMA,
            "feedback_id": feedback_id,
            "polarity": polarity,
            "outcome_status": outcome,
            "interpretation_status": str(validation.get("interpretation_status") or ""),
            "program_id": str(proposal.get("program_id") or ""),
            "innovation_id": str(proposal.get("source_innovation_id") or ""),
            "validation_id": str(validation.get("validation_id") or ""),
            "source_validation_sha256": str(validation.get("content_sha256") or ""),
            "observation_scope": {
                "input_state_ids": list(proposal.get("input_state_ids") or []),
                "output_state_ids": list(proposal.get("output_state_ids") or []),
                "mechanism_signature_sha256": str(
                    validation.get("mechanism_signature_sha256") or ""
                ),
                "generalization_scope": "exact_boundary_only",
            },
            "required_check_results": dict(validation.get("required_check_results") or {}),
            "evidence_tier": str(validation.get("evidence_tier") or ""),
            "claim_refs": list(validation.get("claim_refs") or []),
            "condition_record_ids": list(validation.get("condition_record_ids") or []),
            "analytical_record_ids": list(validation.get("analytical_record_ids") or []),
            "outcome_metrics": dict(validation.get("outcome_metrics") or {}),
            "grants_validation": audit.get("accepted") is True,
            "candidate_disposition": (
                "read_only_shadow_eligible"
                if audit.get("accepted") is True
                else "exploration_visible"
            ),
            "anchor_source_reports_extrapolated_reaction": False,
            "canonical_reaction_proof_created": False,
            "store_mutated": False,
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
            "schema_version": MECHANISM_FEEDBACK_ORACLE_SCHEMA,
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
    "MECHANISM_EXPERIMENT_FEEDBACK_SCHEMA",
    "MECHANISM_FEEDBACK_ORACLE_SCHEMA",
    "MECHANISM_FEEDBACK_PROJECTION_SCHEMA",
    "MECHANISM_FEEDBACK_SEMANTICS",
    "MechanismExperimentFeedbackError",
    "compile_mechanism_experiment_feedback",
    "mechanism_experiment_feedback_oracle",
]
