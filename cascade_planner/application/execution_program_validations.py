"""Fail-closed validation gates for whole-cell and hybrid Programs."""

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


EXECUTION_PROGRAM_VALIDATION_SCHEMA = "execution_program_validation.v1"
EXECUTION_VALIDATION_TIERS = {
    "exact_execution_screen",
    "preparative",
    "process_relevant",
}
EXECUTION_VALIDATION_OUTCOMES = {"success", "failure", "inconclusive"}
EXECUTION_VALIDATION_FIELDS = {
    "schema_version",
    "validation_id",
    "program_id",
    "capability_id",
    "source_capability_sha256",
    "execution_domain",
    "outcome_status",
    "evidence_tier",
    "input_state_ids",
    "output_state_ids",
    "operation_sequence_sha256",
    "required_check_results",
    "claim_refs",
    "condition_record_ids",
    "actor_identity_refs",
    "cofactor_carrier_ledger_closed",
    "outcome_metrics",
    "content_sha256",
}


class ExecutionProgramValidationError(ValueError):
    """An execution validation record is not strict JSON."""


def with_execution_program_validation_digest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return with_program_validation_digest(value)
    except ProgramValidationContractError as exc:
        raise ExecutionProgramValidationError(str(exc)) from exc


def execution_operation_sequence_sha256(proposal: Mapping[str, Any]) -> str:
    return strict_canonical_json_sha256(
        list(proposal.get("operation_blueprints") or [])
    )


def execution_validation_gate(
    proposal: Mapping[str, Any],
    validations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    program_id = str(proposal.get("program_id") or "")
    matching = [row for row in validations if row.get("program_id") == program_id]
    id_counts = Counter(str(row.get("validation_id") or "") for row in validations)
    audits = [
        _audit_execution_validation(
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
        "reasons": (
            [] if accepted_ids else ["specialized_execution_validation_missing"]
        ),
    }


def strict_execution_validations(
    values: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        try:
            rows.append(_strict_execution_validation(value))
        except ProgramInnovationContractError as exc:
            raise ExecutionProgramValidationError(str(exc)) from exc
    return rows


def _strict_execution_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Strict-copy a provider record while preserving its observed digest."""

    return strict_program_innovation_object(value, "execution_validation")


def _audit_execution_validation(
    proposal: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    duplicate_id: bool,
) -> dict[str, Any]:
    binding = audit_program_validation_binding(
        row,
        expected_fields=EXECUTION_VALIDATION_FIELDS,
        expected_schema=EXECUTION_PROGRAM_VALIDATION_SCHEMA,
        expected_program_id=str(proposal.get("program_id") or ""),
        expected_input_state_ids=list(proposal.get("input_state_ids") or []),
        expected_output_state_ids=list(proposal.get("output_state_ids") or []),
        outcome_field="outcome_metrics",
        require_condition_refs=True,
    )
    reasons = list(binding["reasons"])
    if duplicate_id:
        reasons.append("validation_id_duplicate")
    if row.get("capability_id") != proposal.get("source_capability_id"):
        reasons.append("validation_capability_mismatch")
    if row.get("source_capability_sha256") != proposal.get(
        "source_capability_sha256"
    ):
        reasons.append("validation_capability_digest_mismatch")
    if row.get("execution_domain") != proposal.get("execution_domain"):
        reasons.append("validation_execution_domain_mismatch")
    if row.get("operation_sequence_sha256") != execution_operation_sequence_sha256(
        proposal
    ):
        reasons.append("validation_operation_sequence_mismatch")
    if row.get("evidence_tier") not in EXECUTION_VALIDATION_TIERS:
        reasons.append("validation_evidence_tier_invalid")
    outcome_status = str(row.get("outcome_status") or "")
    if outcome_status not in EXECUTION_VALIDATION_OUTCOMES:
        reasons.append("validation_outcome_status_invalid")
    required = list(dict(proposal.get("validation_plan") or {}).get("required_checks") or [])
    results = row.get("required_check_results")
    check_results_valid = (
        isinstance(results, dict)
        and set(results) == set(required)
        and all(isinstance(value, bool) for value in results.values())
    )
    if not check_results_valid:
        reasons.append("validation_required_checks_invalid")
    if not string_list(row.get("actor_identity_refs"), allow_empty=False):
        reasons.append("validation_actor_identity_refs_missing")
    if not isinstance(row.get("cofactor_carrier_ledger_closed"), bool):
        reasons.append("validation_cofactor_carrier_ledger_invalid")
    record_reasons = sorted(set(reasons))
    feedback_eligible = not record_reasons
    acceptance_reasons: list[str] = []
    if feedback_eligible and outcome_status != "success":
        acceptance_reasons.append("validation_outcome_not_success")
    if feedback_eligible and not all(dict(results).values()):
        acceptance_reasons.append("validation_required_checks_failed")
    ledger = dict(proposal.get("cofactor_and_carrier_ledger") or {})
    ledger_required = any(dict(ledger.get(key) or {}) for key in ledger)
    if (
        feedback_eligible
        and ledger_required
        and row.get("cofactor_carrier_ledger_closed") is not True
    ):
        acceptance_reasons.append("validation_cofactor_carrier_ledger_open")
    accepted = feedback_eligible and not acceptance_reasons
    return {
        **binding,
        "record_valid": feedback_eligible,
        "feedback_eligible": feedback_eligible,
        "accepted": accepted,
        "outcome_status": outcome_status,
        "reasons": sorted({*record_reasons, *acceptance_reasons}),
    }


__all__ = [
    "EXECUTION_PROGRAM_VALIDATION_SCHEMA",
    "EXECUTION_VALIDATION_FIELDS",
    "EXECUTION_VALIDATION_OUTCOMES",
    "EXECUTION_VALIDATION_TIERS",
    "ExecutionProgramValidationError",
    "execution_operation_sequence_sha256",
    "execution_validation_gate",
    "strict_execution_validations",
    "with_execution_program_validation_digest",
]
