"""Pure candidate checks and operation-boundary binding for execution Programs."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.route_execution_capabilities import (
    EXECUTION_DOMAINS,
    PROGRAM_EXECUTION_CAPABILITY_SCHEMA,
    normalize_program_execution_capability,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def execution_candidate_reasons(
    candidate: Mapping[str, Any], capability: Mapping[str, Any]
) -> list[str]:
    material = dict(capability)
    observed = str(material.pop("content_sha256", ""))
    reasons: list[str] = []
    normalized, normalization_reasons = normalize_program_execution_capability(capability)
    if normalization_reasons or normalized != capability:
        reasons.append("execution_capability_not_normalized")
    if not str(candidate.get("candidate_id") or ""):
        reasons.append("execution_candidate_id_missing")
    if candidate.get("not_program_yet") is not True:
        reasons.append("execution_candidate_program_boundary_invalid")
    if capability.get("schema_version") != PROGRAM_EXECUTION_CAPABILITY_SCHEMA:
        reasons.append("execution_capability_schema_invalid")
    if observed != strict_canonical_json_sha256(material):
        reasons.append("execution_capability_digest_invalid")
    domain = str(capability.get("execution_domain") or "")
    if domain not in EXECUTION_DOMAINS or candidate.get("execution_domain") != domain:
        reasons.append("execution_candidate_domain_mismatch")
    if candidate.get("capability_id") != capability.get("capability_id"):
        reasons.append("execution_candidate_capability_mismatch")
    if dict(candidate.get("match_audit") or {}).get("accepted") is not True:
        reasons.append("execution_candidate_structure_match_not_accepted")
    return sorted(set(reasons))


def bind_execution_operation_boundaries(
    raw_operations: list[dict[str, Any]],
    input_states: list[str],
    output_states: list[str],
) -> list[dict[str, Any]]:
    transform_indexes = [
        index
        for index, row in enumerate(raw_operations)
        if row["contributes_to_net_transform"] is True
    ]
    first, last = transform_indexes[0], transform_indexes[-1]
    return [
        {
            **dict(row),
            "input_state_ids": input_states if index == first else [],
            "output_state_ids": output_states if index == last else [],
            "internal_state_status": (
                "program_boundary"
                if first == last == index
                else "unmaterialized_internal_sequence"
                if index in transform_indexes
                else "not_a_molecular_transformation"
            ),
        }
        for index, row in enumerate(raw_operations)
    ]


__all__ = [
    "bind_execution_operation_boundaries",
    "execution_candidate_reasons",
]
