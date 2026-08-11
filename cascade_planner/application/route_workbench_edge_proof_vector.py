"""Compile one edge's independent proof axes for Workbench projections."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.reaction_condition_records import (
    audit_condition_completeness,
    normalize_source_conditions,
)

PROOF_VECTOR_SCHEMA = "retrosynthesis_proof_vector.v1"


def edge_proof_vector(
    *, edge: Mapping[str, Any], proof: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    records = [
        dict(dict(graph.get("exact_records") or {}).get(str(record_id)) or {})
        for record_id in proof.get("exact_record_ids") or []
    ]
    records = [value for value in records if value]
    procedure_source = (
        proof.get("procedure_record_ids")
        if "procedure_record_ids" in proof
        else edge.get("procedure_record_ids")
    )
    procedure_ids = [str(value) for value in procedure_source or []]
    canonical_procedures = [
        dict(dict(graph.get("procedure_records") or {}).get(value) or {})
        for value in procedure_ids
    ]
    canonical_procedures = [value for value in canonical_procedures if value]
    observation_source = (
        proof.get("source_observation_record_ids")
        if "source_observation_record_ids" in proof
        else edge.get("source_observation_record_ids")
    )
    source_observations = [
        dict(
            dict(graph.get("source_observation_records") or {}).get(str(record_id))
            or {}
        )
        for record_id in observation_source or []
    ]
    source_observations = [value for value in source_observations if value]
    legacy_procedures = [
        value
        for value in records
        if str(
            value.get("procedure_authority_scope")
            or value.get("authority_scope")
            or ""
        )
        in {
            "source_exact_reaction_procedure",
            "source_exact_procedure_observation",
        }
    ]
    exact_procedures = canonical_procedures or legacy_procedures
    condition_records = [
        value
        for value in exact_procedures
        if isinstance(value.get("conditions"), Mapping)
        and bool(value.get("conditions"))
    ]
    unverified_conditions = [
        value
        for value in records
        if value not in legacy_procedures
        and isinstance(value.get("conditions"), Mapping)
        and bool(value.get("conditions"))
    ]
    unverified_conditions.extend(
        value
        for value in source_observations
        if isinstance(value.get("conditions"), Mapping)
        and bool(value.get("conditions"))
    )
    predictions = [
        dict(value)
        for value in (
            edge.get("condition_predictions")
            or dict(edge.get("metadata") or {}).get("condition_predictions")
            or []
        )
        if isinstance(value, Mapping)
    ]
    source_condition_predictions = [
        {
            **value,
            "conditions": (
                dict(value.get("conditions") or {})
                or normalize_source_conditions(value)
            ),
            "condition_completeness": (
                dict(value.get("condition_completeness") or {})
                or audit_condition_completeness(normalize_source_conditions(value))
            ),
        }
        for value in predictions
        if str(value.get("authority_scope") or "")
        == "model_extracted_source_condition_candidate"
        and str(value.get("source_ref") or "")
    ]
    unverified_conditions.extend(source_condition_predictions)
    complete_procedures = [
        value
        for value in exact_procedures
        if dict(value.get("condition_completeness") or {}).get("complete") is True
    ]
    missing_group_sets = [
        sorted(
            str(group)
            for group in dict(value.get("condition_completeness") or {}).get(
                "missing_required_groups"
            )
            or []
            if str(group)
        )
        for value in [*exact_procedures, *source_observations]
    ]
    missing_groups = min(
        missing_group_sets,
        key=lambda values: (len(values), values),
        default=[],
    )
    predicted = bool(predictions)
    condition_state = (
        "source_exact"
        if condition_records
        else "source_recorded_unverified"
        if unverified_conditions or source_condition_predictions
        else "model_predicted"
        if predicted
        else "missing"
    )
    source_groups = {
        str(value)
        for value in proof.get("independent_source_groups") or []
        if str(value)
    }
    conflicted = bool(proof.get("conflict_ids"))
    source_state = (
        "conflicted"
        if conflicted
        else "independent_2_plus"
        if len(source_groups) >= 2
        else "single_group"
        if source_groups
        else "none"
    )
    reaction_state = (
        "source_reaction_exact"
        if exact_procedures and proof.get("reaction_validated") is True
        else "host_validated"
        if proof.get("reaction_validated") is True
        else "invalidated"
        if proof.get("inactive_fact_count") and edge.get("reaction_proofs")
        else "mapped"
        if edge.get("reaction_proofs")
        else "untested"
    )
    process_state = (
        "procedure_bound_candidate"
        if reaction_state in {"host_validated", "source_reaction_exact"}
        and bool(complete_procedures)
        and not conflicted
        else "blocked"
    )
    return {
        "schema_version": PROOF_VECTOR_SCHEMA,
        "identity": "source_exact" if records else "materialized",
        "reaction": reaction_state,
        "conditions": condition_state,
        "sources": source_state,
        "stock": "not_applicable_to_edge",
        "process": process_state,
        "condition_record_count": len(condition_records) + len(unverified_conditions),
        "procedure_record_count": len(exact_procedures),
        "exact_procedure_record_count": len(exact_procedures),
        "complete_procedure_record_count": len(complete_procedures),
        "condition_missing_required_groups": missing_groups,
        "condition_completeness": (
            "complete"
            if complete_procedures
            else "partial"
            if condition_records or unverified_conditions
            else "missing"
        ),
        "semantics": {
            "axes_are_independent": True,
            "exact_structure_does_not_imply_exact_conditions": True,
            "source_observed_conditions_do_not_grant_exact_identity": True,
            "display_projection_grants_no_authority": True,
        },
    }


__all__ = ["PROOF_VECTOR_SCHEMA", "edge_proof_vector"]
