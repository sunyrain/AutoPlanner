"""Execution-domain-neutral binding checks for Program validation records."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


class ProgramValidationContractError(ValueError):
    """A Program validation record is not strict JSON."""


def with_program_validation_digest(
    value: Mapping[str, Any], *, label: str = "validation"
) -> dict[str, Any]:
    try:
        row = strict_program_innovation_object(value, label)
    except ProgramInnovationContractError as exc:
        raise ProgramValidationContractError(str(exc)) from exc
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def audit_program_validation_binding(
    value: Mapping[str, Any],
    *,
    expected_fields: set[str],
    expected_schema: str,
    expected_program_id: str,
    expected_input_state_ids: list[str],
    expected_output_state_ids: list[str],
    outcome_field: str,
    require_condition_refs: bool,
) -> dict[str, Any]:
    row = dict(value)
    material = dict(row)
    observed_digest = str(material.pop("content_sha256", ""))
    validation_id = str(row.get("validation_id") or "")
    reasons: list[str] = []
    if set(row) != expected_fields:
        reasons.append("validation_fields_invalid")
    if row.get("schema_version") != expected_schema:
        reasons.append("validation_schema_invalid")
    if observed_digest != strict_canonical_json_sha256(material):
        reasons.append("validation_digest_invalid")
    if row.get("program_id") != expected_program_id:
        reasons.append("validation_program_mismatch")
    if row.get("input_state_ids") != expected_input_state_ids:
        reasons.append("validation_input_states_mismatch")
    if row.get("output_state_ids") != expected_output_state_ids:
        reasons.append("validation_output_states_mismatch")
    if not string_list(row.get("claim_refs"), allow_empty=False):
        reasons.append("validation_claim_refs_missing")
    if not string_list(
        row.get("condition_record_ids"), allow_empty=not require_condition_refs
    ):
        reasons.append("validation_condition_refs_invalid")
    if not isinstance(row.get(outcome_field), dict) or not row.get(outcome_field):
        reasons.append("validation_outcome_missing")
    if not validation_id:
        reasons.append("validation_id_missing")
    return {
        "validation_id": validation_id,
        "record_valid": not reasons,
        "reasons": sorted(set(reasons)),
        "content_sha256": observed_digest,
    }


def string_list(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item for item in value)
    )


__all__ = [
    "ProgramValidationContractError",
    "audit_program_validation_binding",
    "string_list",
    "with_program_validation_digest",
]
