"""Target-blind applicability inference from replay-validated Program memory."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.application.route_structure_matching import (
    molecule_similarity,
    structure_transition,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_APPLICABILITY_MODEL_SCHEMA = "program_applicability_model.v1"
STRUCTURAL_ANALOG_SIMILARITY_FLOOR = 0.72
APPLICABILITY_SEMANTICS = {
    "model_is_read_only_and_target_blind": True,
    "source_observations_retain_exact_boundary_authority_only": True,
    "structural_analog_transfer_is_weighted_ranking_evidence_only": True,
    "execution_domains_cannot_share_applicability_evidence": True,
    "current_candidate_requires_its_own_exact_validation": True,
    "model_cannot_grant_validation_proof_completion_or_acceptance": True,
    "model_cannot_mutate_or_disable_capability_catalog": True,
}


def compile_program_applicability_model(
    candidate: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile one similarity-weighted model; return empty when no evidence matches."""

    scope = _candidate_scope(candidate)
    if not scope:
        return {}
    matches = [
        match
        for record in sorted(
            (dict(value) for value in records if isinstance(value, Mapping)),
            key=lambda row: str(row.get("experience_id") or ""),
        )
        if (match := _record_match(scope, record)) is not None
    ]
    if not matches:
        return {}
    raw_counts = {
        polarity: sum(int(row["observation_counts"][polarity]) for row in matches)
        for polarity in ("positive", "negative", "inconclusive")
    }
    weighted = {
        polarity: round(
            sum(
                float(row["transfer_weight"])
                * int(row["observation_counts"][polarity])
                for row in matches
            ),
            6,
        )
        for polarity in ("positive", "negative", "inconclusive")
    }
    total = round(sum(weighted.values()), 6)
    if total <= 0:
        return {}
    positive = weighted["positive"]
    negative = weighted["negative"]
    inconclusive = weighted["inconclusive"]
    informative = positive + negative
    conflict_fraction = (
        min(1.0, 2.0 * min(positive, negative) / informative)
        if positive > 0 and negative > 0 and informative > 0
        else 0.0
    )
    inconclusive_fraction = inconclusive / total
    evidence_strength = total / (total + 0.5)
    net_evidence = (positive - negative) / total
    agreement = 1.0 - 0.75 * conflict_fraction
    applicability_score = round(net_evidence * evidence_strength * agreement, 6)
    confidence = round(
        evidence_strength * agreement * (1.0 - inconclusive_fraction), 6
    )
    uncertainty = round(1.0 - confidence, 6)
    risk = round(min(1.0, (negative + 0.5 * inconclusive) / total), 6)
    priority_adjustment = round(
        applicability_score * (0.12 if applicability_score >= 0 else 0.18), 6
    )
    disposition = (
        "conflicting"
        if positive > 0 and negative > 0
        else "supported"
        if positive > 0
        else "contraindicated"
        if negative > 0
        else "inconclusive"
    )
    exact_observations = sum(
        sum(row["observation_counts"].values())
        for row in matches
        if row["transfer_scope"] == "exact_boundary"
    )
    analog_observations = sum(
        sum(row["observation_counts"].values())
        for row in matches
        if row["transfer_scope"] == "structural_analog"
    )
    reasons = [
        f"disposition:{disposition}",
        f"matched_experience_count:{len(matches)}",
        f"exact_observation_count:{exact_observations}",
        f"structural_analog_observation_count:{analog_observations}",
    ]
    if conflict_fraction > 0:
        reasons.append("positive_negative_evidence_conflict")
    payload = {
        "schema_version": PROGRAM_APPLICABILITY_MODEL_SCHEMA,
        "candidate_scope_sha256": str(scope["scope_sha256"]),
        "domain": str(scope["domain"]),
        "subject_key_sha256": strict_canonical_json_sha256(scope["subject_key"]),
        "matched_experience_ids": [row["experience_id"] for row in matches],
        "source_record_refs": [
            {
                "experience_id": row["experience_id"],
                "content_sha256": row["record_sha256"],
            }
            for row in matches
        ],
        "match_count": len(matches),
        "evidence_counts": raw_counts,
        "weighted_evidence": weighted,
        "exact_observation_count": exact_observations,
        "structural_analog_observation_count": analog_observations,
        "strongest_transfer_scope": (
            "exact_boundary"
            if exact_observations
            else "structural_analog"
        ),
        "maximum_boundary_similarity": max(
            float(row["boundary_similarity"]) for row in matches
        ),
        "applicability_score": applicability_score,
        "evidence_strength": round(evidence_strength, 6),
        "confidence_score": confidence,
        "uncertainty_score": uncertainty,
        "risk_score": risk,
        "priority_adjustment": priority_adjustment,
        "disposition": disposition,
        "matches": matches,
        "reasons": reasons,
        "authority_scope": "proposal_ranking_and_validation_priority_only",
        "semantics": dict(APPLICABILITY_SEMANTICS),
    }
    return _with_digest(payload)


def claim_program_strategy_signature(
    discovery: Mapping[str, Any], domain: str, subject_refs: Mapping[str, Any]
) -> str:
    """Bind mechanism experience to its strategy rather than an innovation label."""

    if domain != "mechanism":
        return ""
    innovation_id = str(subject_refs.get("innovation_id") or "")
    for candidate in discovery.get("candidates") or []:
        innovation = dict(dict(candidate).get("route_innovation") or {})
        if innovation.get("innovation_id") == innovation_id:
            return _mechanism_strategy_signature(innovation)
    return ""


def program_experience_subject_key(
    domain: str, subject_refs: Mapping[str, Any], strategy_signature: str
) -> str:
    """Return the domain-safe identity used by records and applicability matching."""

    if domain == "biocatalytic":
        return str(subject_refs.get("capability_id") or "")
    if domain == "execution":
        return strict_canonical_json_sha256(
            {
                "capability_id": str(subject_refs.get("capability_id") or ""),
                "execution_domain": str(subject_refs.get("execution_domain") or ""),
            }
        )
    return strategy_signature


def _candidate_scope(candidate: Mapping[str, Any]) -> dict[str, Any]:
    domain = _candidate_domain(candidate)
    boundary = dict(candidate.get("boundary") or {})
    transition = structure_transition(
        str(boundary.get("precursor_smiles") or ""),
        str(boundary.get("product_smiles") or ""),
    )
    if not domain or transition.get("valid") is not True:
        return {}
    subject_refs = _candidate_subject_refs(candidate, domain)
    strategy = (
        _mechanism_strategy_signature(dict(candidate.get("route_innovation") or {}))
        if domain == "mechanism"
        else ""
    )
    subject_key = _subject_key(domain, subject_refs, strategy)
    if not subject_key:
        return {}
    identity = {
        "domain": domain,
        "subject_key": subject_key,
        "precursor_smiles": transition["precursor_smiles"],
        "product_smiles": transition["product_smiles"],
        "motif_delta": transition["motif_delta"],
        "element_delta": transition["element_delta"],
    }
    return {
        **identity,
        "scope_sha256": strict_canonical_json_sha256(identity),
        "subject_key": subject_key,
        "transition": transition,
    }


def _record_match(
    scope: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any] | None:
    domain = str(scope["domain"])
    if domain != record.get("domain") or scope["subject_key"] != _subject_key(
        domain,
        dict(record.get("subject_refs") or {}),
        str(record.get("strategy_signature_sha256") or ""),
    ):
        return None
    prior = dict(record.get("exact_boundary") or {})
    prior_inputs = [str(value) for value in prior.get("input_smiles") or [] if str(value)]
    prior_outputs = [str(value) for value in prior.get("output_smiles") or [] if str(value)]
    if not prior_inputs or not prior_outputs:
        return None
    transition = dict(scope["transition"])
    prior_transition = dict(record.get("structural_transition") or {})
    transition_equal = (
        prior_transition.get("valid") is True
        and transition.get("motif_delta") == prior_transition.get("motif_delta")
        and transition.get("element_delta") == prior_transition.get("element_delta")
    )
    if not transition_equal:
        return None
    precursor = str(scope["precursor_smiles"])
    product = str(scope["product_smiles"])
    exact = precursor in prior_inputs and product in prior_outputs
    input_similarity = max(molecule_similarity(precursor, value) for value in prior_inputs)
    output_similarity = max(molecule_similarity(product, value) for value in prior_outputs)
    boundary_similarity = min(input_similarity, output_similarity)
    if not exact and boundary_similarity < STRUCTURAL_ANALOG_SIMILARITY_FLOOR:
        return None
    counts = _observation_counts(record)
    if not sum(counts.values()):
        return None
    normalized_similarity = max(
        0.0,
        min(
            1.0,
            (boundary_similarity - STRUCTURAL_ANALOG_SIMILARITY_FLOOR)
            / (1.0 - STRUCTURAL_ANALOG_SIMILARITY_FLOOR),
        ),
    )
    transfer_weight = 1.0 if exact else 0.35 + 0.5 * normalized_similarity
    return {
        "experience_id": str(record.get("experience_id") or ""),
        "record_sha256": str(record.get("content_sha256") or ""),
        "transfer_scope": "exact_boundary" if exact else "structural_analog",
        "input_similarity": round(input_similarity, 6),
        "output_similarity": round(output_similarity, 6),
        "boundary_similarity": round(boundary_similarity, 6),
        "transfer_weight": round(transfer_weight, 6),
        "observation_counts": counts,
    }


def _observation_counts(record: Mapping[str, Any]) -> dict[str, int]:
    observations = dict(record.get("observations") or {})
    return {
        polarity: sum(
            dict(value).get("polarity") == polarity
            for value in observations.values()
            if isinstance(value, Mapping)
        )
        for polarity in ("positive", "negative", "inconclusive")
    }


def _candidate_domain(candidate: Mapping[str, Any]) -> str:
    return {
        "enzyme_window": "biocatalytic",
        "program_execution_window": "execution",
        "mechanism_one_hop": "mechanism",
    }.get(str(candidate.get("candidate_kind") or ""), "")


def _candidate_subject_refs(
    candidate: Mapping[str, Any], domain: str
) -> dict[str, str]:
    if domain == "biocatalytic":
        return {"capability_id": str(candidate.get("capability_id") or "")}
    if domain == "execution":
        return {
            "capability_id": str(candidate.get("capability_id") or ""),
            "execution_domain": str(candidate.get("execution_domain") or ""),
        }
    return {}


def _subject_key(
    domain: str, subject_refs: Mapping[str, Any], strategy_signature: str
) -> dict[str, str]:
    if domain == "biocatalytic":
        capability_id = str(subject_refs.get("capability_id") or "")
        return {"capability_id": capability_id} if capability_id else {}
    if domain == "execution":
        capability_id = str(subject_refs.get("capability_id") or "")
        execution_domain = str(subject_refs.get("execution_domain") or "")
        return (
            {"capability_id": capability_id, "execution_domain": execution_domain}
            if capability_id and execution_domain
            else {}
        )
    return {"mechanism_strategy_sha256": strategy_signature} if strategy_signature else {}


def _mechanism_strategy_signature(innovation: Mapping[str, Any]) -> str:
    anchor = dict(innovation.get("anchor") or {})
    return strict_canonical_json_sha256(
        {
            "anchor_source_refs": list(anchor.get("source_refs") or []),
            "mechanistic_rationale": str(innovation.get("mechanistic_rationale") or ""),
            "elementary_steps": list(innovation.get("elementary_steps") or []),
            "falsifiable_checks": list(innovation.get("falsifiable_checks") or []),
        }
    )


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "APPLICABILITY_SEMANTICS",
    "PROGRAM_APPLICABILITY_MODEL_SCHEMA",
    "claim_program_strategy_signature",
    "compile_program_applicability_model",
    "program_experience_subject_key",
]
