"""Fail-closed contracts shared by biocatalytic Program proposal compilers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
    validate_program_innovation_inputs,
    with_program_innovation_digest,
)
from cascade_planner.application.program_span_substitutions import (
    ProgramSpanError,
    program_span_boundary,
)
from cascade_planner.application.program_validation_contracts import (
    ProgramValidationContractError,
    audit_program_validation_binding,
    string_list as program_validation_string_list,
    with_program_validation_digest,
)
from cascade_planner.application.route_innovations import (
    BIOCATALYTIC_KINDS,
    BIOCATALYTIC_SUPERSTEP,
    ROUTE_INNOVATION_SCHEMA,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA = "biocatalytic_program_bundle.v1"
BIOCATALYTIC_PROGRAM_PROPOSAL_SCHEMA = "biocatalytic_program_proposal.v1"
BIOCATALYTIC_PROGRAM_ROUTE_SCHEMA = "biocatalytic_program_route_candidate.v1"
BIOCATALYTIC_PROGRAM_ORACLE_SCHEMA = "biocatalytic_program_bundle_oracle.v1"
BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA = "biocatalysis_program_validation.v1"
VALIDATION_TIERS = {"exact_substrate_screen", "preparative", "experiment"}
VALIDATION_KEYS = {
    "schema_version",
    "validation_id",
    "program_id",
    "innovation_id",
    "accepted",
    "evidence_tier",
    "input_state_ids",
    "output_state_ids",
    "claim_refs",
    "condition_record_ids",
    "selectivity_assessed",
    "cofactor_ledger_closed",
    "outcome",
    "content_sha256",
}
BIOCATALYTIC_PROGRAM_SEMANTICS = {
    "read_only_program_proposals": True,
    "supersteps_connect_interval_boundary_states": True,
    "replaced_edge_programs_remain_fallback": True,
    "chemical_equivalent_steps_and_operations_are_distinct": True,
    "analog_capabilities_are_not_reaction_proof": True,
    "specialized_biocatalysis_validation_is_required": True,
    "unvalidated_candidates_remain_visible": True,
    "program_candidates_cannot_grant_route_completion": True,
    "edge_ids_remain_production_route_authority": True,
    "target_names_are_not_compilation_inputs": True,
}


class BiocatalyticProgramError(ValueError):
    """A discovery cannot be projected into a safe program substitution."""


def with_biocatalysis_program_validation_digest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize a provider validation record without deciding its outcome."""

    try:
        return with_program_validation_digest(value)
    except ProgramValidationContractError as exc:
        raise BiocatalyticProgramError(str(exc)) from exc


def validate_biocatalytic_program_inputs(
    graph: dict[str, Any],
    route: dict[str, Any],
    projection: dict[str, Any],
    discovery: dict[str, Any],
) -> None:
    try:
        validate_program_innovation_inputs(graph, route, projection, discovery)
    except ProgramInnovationContractError as exc:
        raise BiocatalyticProgramError(str(exc)) from exc


def validate_biocatalytic_innovation(
    candidate: Mapping[str, Any], innovation: Mapping[str, Any]
) -> None:
    reasons: list[str] = []
    material = dict(innovation)
    observed = str(material.pop("content_sha256", ""))
    if innovation.get("schema_version") != ROUTE_INNOVATION_SCHEMA:
        reasons.append("route_innovation_schema_invalid")
    if observed != strict_canonical_json_sha256(material):
        reasons.append("route_innovation_digest_invalid")
    span = innovation.get("replaced_step_ids")
    if innovation.get("kind") not in BIOCATALYTIC_KINDS:
        reasons.append("route_innovation_not_biocatalytic")
    if not string_list(span) or not span:
        reasons.append("route_innovation_replacement_span_missing")
    elif int(innovation.get("chemical_step_equivalent_count") or 0) != len(span):
        reasons.append("route_innovation_equivalent_count_mismatch")
    elif innovation.get("kind") == BIOCATALYTIC_SUPERSTEP and len(span) < 2:
        reasons.append("route_innovation_superstep_span_too_short")
    if not str(candidate.get("capability_id") or ""):
        reasons.append("route_innovation_capability_missing")
    if not innovation.get("precedent_refs"):
        reasons.append("route_innovation_precedent_missing")
    if reasons:
        raise BiocatalyticProgramError(";".join(sorted(set(reasons))))


def biocatalytic_span_boundary(
    graph: Mapping[str, Any], route: Mapping[str, Any], span: Sequence[str]
) -> dict[str, list[str]]:
    try:
        return program_span_boundary(graph, route, span)
    except ProgramSpanError as exc:
        raise BiocatalyticProgramError(str(exc)) from exc


def biocatalysis_validation_gate(
    proposal_id: str,
    innovation: Mapping[str, Any],
    input_states: list[str],
    output_states: list[str],
    validations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    matching = [row for row in validations if row.get("program_id") == proposal_id]
    audits: list[dict[str, Any]] = []
    accepted_ids: list[str] = []
    for row in matching:
        binding = audit_program_validation_binding(
            row,
            expected_fields=VALIDATION_KEYS,
            expected_schema=BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
            expected_program_id=proposal_id,
            expected_input_state_ids=input_states,
            expected_output_state_ids=output_states,
            outcome_field="outcome",
            require_condition_refs=False,
        )
        reasons = list(binding["reasons"])
        if row.get("accepted") is not True:
            reasons.append("validation_outcome_not_accepted")
        if row.get("evidence_tier") not in VALIDATION_TIERS:
            reasons.append("validation_evidence_tier_invalid")
        if row.get("innovation_id") != innovation.get("innovation_id"):
            reasons.append("validation_innovation_mismatch")
        if row.get("selectivity_assessed") is not True:
            reasons.append("validation_selectivity_missing")
        cofactor_required = bool(innovation.get("cofactor_requirements"))
        if cofactor_required and row.get("cofactor_ledger_closed") is not True:
            reasons.append("validation_cofactor_ledger_open")
        validation_id = str(binding["validation_id"])
        accepted = not reasons
        if accepted:
            accepted_ids.append(validation_id)
        audits.append(
            {
                "validation_id": validation_id,
                "accepted": accepted,
                "reasons": sorted(set(reasons)),
                "content_sha256": str(binding["content_sha256"]),
            }
        )
    return {
        "accepted": bool(accepted_ids),
        "validation_ids": sorted(
            str(row.get("validation_id") or "") for row in matching
        ),
        "accepted_validation_ids": sorted(accepted_ids),
        "audits": audits,
        "reasons": [] if accepted_ids else ["specialized_biocatalysis_validation_missing"],
    }


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def string_list(value: Any) -> bool:
    return program_validation_string_list(value, allow_empty=True)


def strict_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        return strict_program_innovation_object(value, label)
    except ProgramInnovationContractError as exc:
        raise BiocatalyticProgramError(str(exc)) from exc


def with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return with_program_innovation_digest(value)


__all__ = [
    "BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA",
    "BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA",
    "BIOCATALYTIC_PROGRAM_ORACLE_SCHEMA",
    "BIOCATALYTIC_PROGRAM_PROPOSAL_SCHEMA",
    "BIOCATALYTIC_PROGRAM_ROUTE_SCHEMA",
    "BIOCATALYTIC_PROGRAM_SEMANTICS",
    "BiocatalyticProgramError",
    "biocatalysis_validation_gate",
    "biocatalytic_span_boundary",
    "ordered_unique",
    "strict_object",
    "string_list",
    "validate_biocatalytic_innovation",
    "validate_biocatalytic_program_inputs",
    "with_biocatalysis_program_validation_digest",
    "with_digest",
]
