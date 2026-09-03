"""Construct normalized experiment Claim rows from domain feedback."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.application.experimental_claim_contracts import (
    CLAIM_SEMANTICS,
    EXPERIMENTAL_OBSERVATION_CLAIM_SCHEMA,
    with_experimental_claim_digest,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def biocatalytic_claim(
    proposal: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    return _claim(
        domain="biocatalytic",
        polarity="positive",
        outcome_status="success",
        interpretation_status="exact_substrate_biocatalysis_supported",
        program_id=str(proposal.get("program_id") or ""),
        input_state_ids=validation.get("input_state_ids") or [],
        output_state_ids=validation.get("output_state_ids") or [],
        validation=validation,
        evidence_tier=str(validation.get("evidence_tier") or ""),
        claim_refs=validation.get("claim_refs") or [],
        condition_ids=validation.get("condition_record_ids") or [],
        outcome_metrics=dict(validation.get("outcome") or {}),
        grants_validation=True,
        subject_refs={
            "capability_id": str(proposal.get("source_capability_id") or ""),
            "innovation_id": str(proposal.get("source_innovation_id") or ""),
        },
        domain_context={
            "selectivity_assessed": validation.get("selectivity_assessed") is True,
            "cofactor_ledger_closed": validation.get("cofactor_ledger_closed") is True,
        },
    )


def execution_claim(feedback: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    scope = dict(feedback.get("applicability_scope") or {})
    return _claim(
        domain="execution",
        polarity=str(feedback.get("polarity") or ""),
        outcome_status=str(feedback.get("outcome_status") or ""),
        interpretation_status="exact_execution_outcome_observed",
        program_id=str(feedback.get("program_id") or ""),
        input_state_ids=scope.get("input_state_ids") or [],
        output_state_ids=scope.get("output_state_ids") or [],
        validation=validation,
        evidence_tier=str(feedback.get("evidence_tier") or ""),
        claim_refs=feedback.get("claim_refs") or [],
        condition_ids=feedback.get("condition_record_ids") or [],
        outcome_metrics=dict(feedback.get("outcome_metrics") or {}),
        grants_validation=feedback.get("grants_validation") is True,
        subject_refs={
            "capability_id": str(feedback.get("capability_id") or ""),
            "execution_domain": str(feedback.get("execution_domain") or ""),
        },
        domain_context={
            "source_capability_sha256": str(feedback.get("source_capability_sha256") or ""),
            "operation_sequence_sha256": str(scope.get("operation_sequence_sha256") or ""),
            "actor_identity_refs": canonical_strings(feedback.get("actor_identity_refs") or []),
            "cofactor_carrier_ledger_closed": feedback.get("cofactor_carrier_ledger_closed")
            is True,
        },
    )


def mechanism_claim(feedback: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    scope = dict(feedback.get("observation_scope") or {})
    return _claim(
        domain="mechanism",
        polarity=str(feedback.get("polarity") or ""),
        outcome_status=str(feedback.get("outcome_status") or ""),
        interpretation_status=str(feedback.get("interpretation_status") or ""),
        program_id=str(feedback.get("program_id") or ""),
        input_state_ids=scope.get("input_state_ids") or [],
        output_state_ids=scope.get("output_state_ids") or [],
        validation=validation,
        evidence_tier=str(feedback.get("evidence_tier") or ""),
        claim_refs=feedback.get("claim_refs") or [],
        condition_ids=feedback.get("condition_record_ids") or [],
        outcome_metrics=dict(feedback.get("outcome_metrics") or {}),
        grants_validation=feedback.get("grants_validation") is True,
        subject_refs={"innovation_id": str(feedback.get("innovation_id") or "")},
        domain_context={
            "mechanism_signature_sha256": str(scope.get("mechanism_signature_sha256") or ""),
            "analytical_record_ids": canonical_strings(feedback.get("analytical_record_ids") or []),
            "anchor_source_reports_extrapolated_reaction": False,
            "canonical_reaction_proof_created": False,
        },
    )


def _claim(
    *,
    domain: str,
    polarity: str,
    outcome_status: str,
    interpretation_status: str,
    program_id: str,
    input_state_ids: Sequence[Any],
    output_state_ids: Sequence[Any],
    validation: Mapping[str, Any],
    evidence_tier: str,
    claim_refs: Sequence[Any],
    condition_ids: Sequence[Any],
    outcome_metrics: Mapping[str, Any],
    grants_validation: bool,
    subject_refs: Mapping[str, str],
    domain_context: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "domain": domain,
        "validation_schema": validation.get("schema_version"),
        "validation_id": validation.get("validation_id"),
        "validation_sha256": validation.get("content_sha256"),
        "program_id": program_id,
    }
    claim_id = "experimental-claim:" + strict_canonical_json_sha256(identity)[:32]
    return with_experimental_claim_digest(
        {
            "schema_version": EXPERIMENTAL_OBSERVATION_CLAIM_SCHEMA,
            "claim_id": claim_id,
            "claim_kind": "program_validation_observation",
            "domain": domain,
            "polarity": polarity,
            "outcome_status": outcome_status,
            "interpretation_status": interpretation_status,
            "program_id": program_id,
            "subject_refs": {key: value for key, value in sorted(subject_refs.items()) if value},
            "boundary": {
                "input_state_ids": canonical_strings(input_state_ids),
                "output_state_ids": canonical_strings(output_state_ids),
            },
            "source_validation": {
                "schema_version": str(validation.get("schema_version") or ""),
                "validation_id": str(validation.get("validation_id") or ""),
                "content_sha256": str(validation.get("content_sha256") or ""),
            },
            "evidence_tier": evidence_tier,
            "supporting_claim_refs": canonical_strings(claim_refs),
            "condition_record_ids": canonical_strings(condition_ids),
            "outcome_metrics": dict(outcome_metrics),
            "grants_domain_validation": grants_validation,
            "generalization_scope": "exact_boundary_only",
            "authority_scope": "experimental_observation_exact_boundary",
            "domain_context": dict(domain_context),
            "semantics": dict(CLAIM_SEMANTICS),
        }
    )


def canonical_strings(values: Sequence[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


__all__ = [
    "biocatalytic_claim",
    "canonical_strings",
    "execution_claim",
    "mechanism_claim",
]
