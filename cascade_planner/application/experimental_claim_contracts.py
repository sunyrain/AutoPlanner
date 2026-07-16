"""Fail-closed contracts for unified exact-boundary experiment claims."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENTAL_OBSERVATION_CLAIM_SCHEMA = "experimental_observation_claim.v1"
EXPERIMENTAL_CLAIM_SET_SCHEMA = "experimental_observation_claim_set.v1"
EXPERIMENTAL_CLAIM_SET_ORACLE_SCHEMA = "experimental_claim_set_oracle.v1"
EXPERIMENTAL_CLAIM_DOMAINS = {"biocatalytic", "execution", "mechanism"}
EXPERIMENTAL_CLAIM_POLARITIES = {"positive", "negative", "inconclusive"}

CLAIM_SEMANTICS = {
    "immutable_observation": True,
    "authority_is_exact_boundary_only": True,
    "does_not_create_canonical_reaction_proof": True,
    "does_not_inherit_literature_anchor_authority": True,
    "does_not_grant_program_store_admission": True,
    "does_not_grant_route_completion_or_acceptance": True,
    "does_not_mutate_capability_catalog": True,
}
CLAIM_SET_SEMANTICS = {
    "projection_is_read_only": True,
    "domain_validation_semantics_remain_independent": True,
    "valid_positive_negative_and_inconclusive_observations_are_retained": True,
    "biocatalysis_v1_only_represents_accepted_positive_observations": True,
    "claim_set_is_not_a_canonical_fact_store": True,
    "claim_set_cannot_grant_proof_completion_acceptance_or_catalog_mutation": True,
}
_CLAIM_FIELDS = {
    "schema_version",
    "claim_id",
    "claim_kind",
    "domain",
    "polarity",
    "outcome_status",
    "interpretation_status",
    "program_id",
    "subject_refs",
    "boundary",
    "source_validation",
    "evidence_tier",
    "supporting_claim_refs",
    "condition_record_ids",
    "outcome_metrics",
    "grants_domain_validation",
    "generalization_scope",
    "authority_scope",
    "domain_context",
    "semantics",
    "content_sha256",
}
_SET_FIELDS = {
    "schema_version",
    "run_id",
    "route_id",
    "source_artifacts",
    "claims",
    "rejected_validations",
    "counts",
    "semantics",
    "content_sha256",
}
_SOURCE_ARTIFACT_FIELDS = {
    "biocatalytic_bundle_sha256",
    "biocatalytic_oracle_sha256",
    "execution_feedback_sha256",
    "execution_oracle_sha256",
    "mechanism_feedback_sha256",
    "mechanism_oracle_sha256",
    "validation_pack_sha256",
}
_REJECTION_FIELDS = {"domain", "validation_id", "program_id", "reasons"}


class ExperimentalClaimError(ValueError):
    """Experimental observations failed a structural or authority contract."""


def with_experimental_claim_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def experimental_claim_counts(
    claims: Mapping[str, Mapping[str, Any]], rejected: list[Mapping[str, Any]]
) -> dict[str, int]:
    rows = list(claims.values())
    return {
        "claims": len(rows),
        "biocatalytic": sum(row.get("domain") == "biocatalytic" for row in rows),
        "execution": sum(row.get("domain") == "execution" for row in rows),
        "mechanism": sum(row.get("domain") == "mechanism" for row in rows),
        "positive": sum(row.get("polarity") == "positive" for row in rows),
        "negative": sum(row.get("polarity") == "negative" for row in rows),
        "inconclusive": sum(row.get("polarity") == "inconclusive" for row in rows),
        "rejected_validations": len(rejected),
        "canonical_reaction_proofs_created": 0,
        "catalog_mutations": 0,
        "route_completions_granted": 0,
    }


def validate_experimental_claim_set(value: Mapping[str, Any]) -> list[str]:
    row = dict(value)
    reasons: list[str] = []
    if set(row) != _SET_FIELDS:
        reasons.append("experimental_claim_set_fields_invalid")
    if row.get("schema_version") != EXPERIMENTAL_CLAIM_SET_SCHEMA:
        reasons.append("experimental_claim_set_schema_invalid")
    if row.get("semantics") != CLAIM_SET_SEMANTICS:
        reasons.append("experimental_claim_set_semantics_invalid")
    if not _digest_valid(row):
        reasons.append("experimental_claim_set_digest_invalid")
    if not str(row.get("run_id") or "") or not str(row.get("route_id") or ""):
        reasons.append("experimental_claim_set_identity_missing")
    sources = row.get("source_artifacts")
    if not isinstance(sources, dict) or set(sources) != _SOURCE_ARTIFACT_FIELDS:
        reasons.append("experimental_claim_set_sources_invalid")
    elif not all(_sha256(value) for value in sources.values()):
        reasons.append("experimental_claim_set_source_digest_invalid")
    claims = row.get("claims")
    if not isinstance(claims, dict):
        reasons.append("experimental_claim_set_claims_invalid")
        claims = {}
    for claim_id, raw_claim in claims.items():
        reasons.extend(_claim_reasons(str(claim_id), raw_claim))
    rejected = row.get("rejected_validations")
    if not isinstance(rejected, list):
        reasons.append("experimental_claim_set_rejections_invalid")
        rejected = []
    else:
        for item in rejected:
            if not _rejection_valid(item):
                reasons.append("experimental_claim_set_rejection_invalid")
                break
    if row.get("counts") != experimental_claim_counts(claims, rejected):
        reasons.append("experimental_claim_set_counts_invalid")
    return sorted(set(reasons))


def _claim_reasons(claim_id: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"experimental_claim_not_object:{claim_id}"]
    row = dict(value)
    reasons: list[str] = []
    if set(row) != _CLAIM_FIELDS:
        reasons.append(f"experimental_claim_fields_invalid:{claim_id}")
    if row.get("schema_version") != EXPERIMENTAL_OBSERVATION_CLAIM_SCHEMA:
        reasons.append(f"experimental_claim_schema_invalid:{claim_id}")
    if row.get("claim_id") != claim_id or not claim_id:
        reasons.append(f"experimental_claim_identity_invalid:{claim_id}")
    if not _digest_valid(row):
        reasons.append(f"experimental_claim_digest_invalid:{claim_id}")
    if row.get("claim_kind") != "program_validation_observation":
        reasons.append(f"experimental_claim_kind_invalid:{claim_id}")
    if row.get("domain") not in EXPERIMENTAL_CLAIM_DOMAINS:
        reasons.append(f"experimental_claim_domain_invalid:{claim_id}")
    if row.get("polarity") not in EXPERIMENTAL_CLAIM_POLARITIES:
        reasons.append(f"experimental_claim_polarity_invalid:{claim_id}")
    if not str(row.get("outcome_status") or "") or not str(row.get("program_id") or ""):
        reasons.append(f"experimental_claim_subject_invalid:{claim_id}")
    if not _string_mapping(row.get("subject_refs")):
        reasons.append(f"experimental_claim_subject_refs_invalid:{claim_id}")
    boundary = row.get("boundary")
    if not isinstance(boundary, dict) or set(boundary) != {
        "input_state_ids",
        "output_state_ids",
    }:
        reasons.append(f"experimental_claim_boundary_invalid:{claim_id}")
    elif not all(
        _string_list(boundary.get(key), allow_empty=False)
        for key in ("input_state_ids", "output_state_ids")
    ):
        reasons.append(f"experimental_claim_boundary_states_invalid:{claim_id}")
    source = row.get("source_validation")
    if not isinstance(source, dict) or set(source) != {
        "schema_version",
        "validation_id",
        "content_sha256",
    }:
        reasons.append(f"experimental_claim_source_invalid:{claim_id}")
    elif (
        not str(source.get("schema_version") or "")
        or not str(source.get("validation_id") or "")
        or not _sha256(source.get("content_sha256"))
    ):
        reasons.append(f"experimental_claim_source_binding_invalid:{claim_id}")
    for key in ("supporting_claim_refs", "condition_record_ids"):
        if not _string_list(row.get(key), allow_empty=True):
            reasons.append(f"experimental_claim_{key}_invalid:{claim_id}")
    if not isinstance(row.get("outcome_metrics"), dict) or not row.get("outcome_metrics"):
        reasons.append(f"experimental_claim_outcome_invalid:{claim_id}")
    if not isinstance(row.get("grants_domain_validation"), bool):
        reasons.append(f"experimental_claim_grant_invalid:{claim_id}")
    if row.get("generalization_scope") != "exact_boundary_only":
        reasons.append(f"experimental_claim_scope_invalid:{claim_id}")
    if row.get("authority_scope") != "experimental_observation_exact_boundary":
        reasons.append(f"experimental_claim_authority_invalid:{claim_id}")
    if not isinstance(row.get("domain_context"), dict):
        reasons.append(f"experimental_claim_context_invalid:{claim_id}")
    if row.get("semantics") != CLAIM_SEMANTICS:
        reasons.append(f"experimental_claim_semantics_invalid:{claim_id}")
    return reasons


def _rejection_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _REJECTION_FIELDS
        and value.get("domain") in EXPERIMENTAL_CLAIM_DOMAINS
        and isinstance(value.get("validation_id"), str)
        and isinstance(value.get("program_id"), str)
        and _string_list(value.get("reasons"), allow_empty=False)
    )


def _digest_valid(value: Mapping[str, Any]) -> bool:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    try:
        return bool(observed) and observed == strict_canonical_json_sha256(material)
    except (TypeError, ValueError):
        return False


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _string_list(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item for item in value)
        and value == sorted(set(value))
    )


def _string_mapping(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(key, str) and key for key in value)
        and all(isinstance(item, str) for item in value.values())
    )


__all__ = [
    "CLAIM_SEMANTICS",
    "CLAIM_SET_SEMANTICS",
    "EXPERIMENTAL_CLAIM_DOMAINS",
    "EXPERIMENTAL_CLAIM_SET_ORACLE_SCHEMA",
    "EXPERIMENTAL_CLAIM_SET_SCHEMA",
    "EXPERIMENTAL_OBSERVATION_CLAIM_SCHEMA",
    "ExperimentalClaimError",
    "experimental_claim_counts",
    "validate_experimental_claim_set",
    "with_experimental_claim_digest",
]
