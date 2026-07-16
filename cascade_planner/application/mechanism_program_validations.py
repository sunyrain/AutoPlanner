"""Fail-closed validation gates for fully restitched mechanism Programs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
)
from cascade_planner.application.program_validation_contracts import (
    ProgramValidationContractError,
    audit_program_validation_binding,
    string_list,
    with_program_validation_digest,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


MECHANISM_PROGRAM_VALIDATION_SCHEMA = "mechanism_program_validation.v1"
MECHANISM_VALIDATION_TIERS = {
    "exact_reaction_screen",
    "preparative",
    "mechanism_probe",
}
MECHANISM_VALIDATION_OUTCOMES = {"success", "failure", "inconclusive"}
MECHANISM_INTERPRETATIONS = {
    "net_transform_observed",
    "mechanism_consistent",
    "mechanism_discriminated",
    "not_supported",
    "competing_pathway_observed",
    "unresolved",
}
MECHANISM_VALIDATION_FIELDS = {
    "schema_version",
    "validation_id",
    "program_id",
    "innovation_id",
    "outcome_status",
    "evidence_tier",
    "interpretation_status",
    "input_state_ids",
    "output_state_ids",
    "mechanism_signature_sha256",
    "required_check_results",
    "claim_refs",
    "condition_record_ids",
    "analytical_record_ids",
    "outcome_metrics",
    "content_sha256",
}
_BASE_CHECKS = (
    ("exact_input_identity", "Confirm every exact input state before exposure."),
    ("exact_output_identity", "Confirm the exact requested output connectivity."),
    ("mass_balance", "Account for substrate, product, and material side products."),
    ("stereo_and_regiochemistry", "Resolve all requested stereo and regio outcomes."),
    (
        "competing_pathway_assessment",
        "Assess plausible competing products under the bound conditions.",
    ),
)


class MechanismProgramValidationError(ValueError):
    """A mechanism validation record is not strict JSON."""


def with_mechanism_program_validation_digest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return with_program_validation_digest(value)
    except ProgramValidationContractError as exc:
        raise MechanismProgramValidationError(str(exc)) from exc


def mechanism_required_checks(proposal: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = [
        {"check_id": check_id, "objective": objective, "required": True}
        for check_id, objective in _BASE_CHECKS
    ]
    checks.extend(
        {
            "check_id": (
                "falsifiable:"
                + strict_canonical_json_sha256({"check": str(objective)})[:16]
            ),
            "objective": str(objective),
            "required": True,
        }
        for objective in proposal.get("falsifiable_checks") or []
    )
    return checks


def mechanism_signature_sha256(proposal: Mapping[str, Any]) -> str:
    return strict_canonical_json_sha256(
        {
            "program_id": proposal.get("program_id"),
            "innovation_id": proposal.get("source_innovation_id"),
            "input_state_ids": list(proposal.get("input_state_ids") or []),
            "output_state_ids": list(proposal.get("output_state_ids") or []),
            "anchor": dict(proposal.get("anchor") or {}),
            "mechanistic_rationale": proposal.get("mechanistic_rationale"),
            "elementary_steps": list(proposal.get("elementary_steps") or []),
            "falsifiable_checks": list(proposal.get("falsifiable_checks") or []),
        }
    )


def mechanism_validation_gate(
    proposal: Mapping[str, Any],
    validations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    program_id = str(proposal.get("program_id") or "")
    matching = [row for row in validations if row.get("program_id") == program_id]
    id_counts = Counter(str(row.get("validation_id") or "") for row in validations)
    audits = [
        _audit_mechanism_validation(
            proposal,
            row,
            duplicate_id=id_counts[str(row.get("validation_id") or "")] > 1,
        )
        for row in matching
    ]
    accepted_ids = sorted(
        row["validation_id"] for row in audits if row["accepted"] is True
    )
    feedback_ids = sorted(
        row["validation_id"] for row in audits if row["feedback_eligible"] is True
    )
    return {
        "accepted": bool(accepted_ids),
        "validation_ids": sorted(
            str(row.get("validation_id") or "") for row in matching
        ),
        "accepted_validation_ids": accepted_ids,
        "feedback_validation_ids": feedback_ids,
        "negative_validation_ids": sorted(
            row["validation_id"]
            for row in audits
            if row["feedback_eligible"] is True
            and row["outcome_status"] == "failure"
        ),
        "inconclusive_validation_ids": sorted(
            row["validation_id"]
            for row in audits
            if row["feedback_eligible"] is True
            and row["outcome_status"] == "inconclusive"
        ),
        "audits": audits,
        "reasons": [] if accepted_ids else ["mechanism_validation_missing"],
    }


def mechanism_support_state(gate: Mapping[str, Any]) -> str:
    interpretations = {
        str(row.get("interpretation_status") or "")
        for row in gate.get("audits") or []
        if row.get("accepted") is True
    }
    if "mechanism_discriminated" in interpretations:
        return "experimentally_discriminated"
    if "mechanism_consistent" in interpretations:
        return "experimentally_consistent_not_discriminated"
    if "net_transform_observed" in interpretations:
        return "net_transform_observed_mechanism_unresolved"
    return "hypothesis_only"


def strict_mechanism_validations(
    values: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        try:
            rows.append(strict_program_innovation_object(value, "mechanism_validation"))
        except ProgramInnovationContractError as exc:
            raise MechanismProgramValidationError(str(exc)) from exc
    return rows


def _audit_mechanism_validation(
    proposal: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    duplicate_id: bool,
) -> dict[str, Any]:
    binding = audit_program_validation_binding(
        row,
        expected_fields=MECHANISM_VALIDATION_FIELDS,
        expected_schema=MECHANISM_PROGRAM_VALIDATION_SCHEMA,
        expected_program_id=str(proposal.get("program_id") or ""),
        expected_input_state_ids=list(proposal.get("input_state_ids") or []),
        expected_output_state_ids=list(proposal.get("output_state_ids") or []),
        outcome_field="outcome_metrics",
        require_condition_refs=True,
    )
    reasons = list(binding["reasons"])
    if duplicate_id:
        reasons.append("validation_id_duplicate")
    if row.get("innovation_id") != proposal.get("source_innovation_id"):
        reasons.append("validation_innovation_mismatch")
    if row.get("mechanism_signature_sha256") != mechanism_signature_sha256(
        proposal
    ):
        reasons.append("validation_mechanism_signature_mismatch")
    if row.get("evidence_tier") not in MECHANISM_VALIDATION_TIERS:
        reasons.append("validation_evidence_tier_invalid")
    outcome = str(row.get("outcome_status") or "")
    interpretation = str(row.get("interpretation_status") or "")
    if outcome not in MECHANISM_VALIDATION_OUTCOMES:
        reasons.append("validation_outcome_status_invalid")
    if interpretation not in MECHANISM_INTERPRETATIONS:
        reasons.append("validation_interpretation_status_invalid")
    if not _interpretation_matches(outcome, interpretation):
        reasons.append("validation_outcome_interpretation_mismatch")
    required = {
        str(check["check_id"]) for check in mechanism_required_checks(proposal)
    }
    results = row.get("required_check_results")
    if not (
        isinstance(results, dict)
        and set(results) == required
        and all(isinstance(value, bool) for value in results.values())
    ):
        reasons.append("validation_required_checks_invalid")
    if not string_list(row.get("analytical_record_ids"), allow_empty=False):
        reasons.append("validation_analytical_records_missing")
    record_reasons = sorted(set(reasons))
    feedback_eligible = not record_reasons
    acceptance_reasons: list[str] = []
    if feedback_eligible and outcome != "success":
        acceptance_reasons.append("validation_outcome_not_success")
    if feedback_eligible and not all(dict(results).values()):
        acceptance_reasons.append("validation_required_checks_failed")
    accepted = feedback_eligible and not acceptance_reasons
    return {
        **binding,
        "record_valid": feedback_eligible,
        "feedback_eligible": feedback_eligible,
        "accepted": accepted,
        "outcome_status": outcome,
        "interpretation_status": interpretation,
        "reasons": sorted({*record_reasons, *acceptance_reasons}),
    }


def _interpretation_matches(outcome: str, interpretation: str) -> bool:
    allowed = {
        "success": {
            "net_transform_observed",
            "mechanism_consistent",
            "mechanism_discriminated",
        },
        "failure": {"not_supported", "competing_pathway_observed"},
        "inconclusive": {"unresolved"},
    }
    return interpretation in allowed.get(outcome, set())


__all__ = [
    "MECHANISM_INTERPRETATIONS",
    "MECHANISM_PROGRAM_VALIDATION_SCHEMA",
    "MECHANISM_VALIDATION_FIELDS",
    "MECHANISM_VALIDATION_OUTCOMES",
    "MECHANISM_VALIDATION_TIERS",
    "MechanismProgramValidationError",
    "mechanism_required_checks",
    "mechanism_signature_sha256",
    "mechanism_support_state",
    "mechanism_validation_gate",
    "strict_mechanism_validations",
    "with_mechanism_program_validation_digest",
]
