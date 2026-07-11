"""Canonicalize and fuse retrosynthetic proposals from independent sources.

The blackboard historically accumulated several similar-but-incompatible route
records.  This module defines the narrow domain boundary shared by Codex child
agents, deterministic planners, literature extraction, and presentation.  A
fused proposal is still advisory: only the existing deterministic route proof
may mark a case solved.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable

from rdkit import Chem, RDLogger

from cascade_planner.routes.admission import audit_retrosynthetic_candidate
from cascade_planner.source_locators import (
    canonical_traceable_source_ref,
    source_record_support_group,
    source_ref_sort_key,
)


RDLogger.DisableLog("rdApp.*")

RETROSYNTHESIS_CANDIDATE_SCHEMA = "retrosynthesis_candidate.v1"
RETROSYNTHESIS_PROPOSAL_REPORT_PAYLOAD_SCHEMA = "retrosynthesis_proposal_report.v1"
ROUTE_CONSENSUS_SCHEMA = "route_consensus.v1"

SOURCE_CHANNELS = {
    "codex_strategy",
    "codex_literature",
    "codex_chemoenzymatic",
    "codex_critic",
    "chem_enzy",
    "literature_exact",
    "literature_analogy",
    "template",
    "stock",
    "human",
    "other",
}

EVIDENCE_LEVEL_WEIGHT = {
    "model_only": 0.28,
    "analogy": 0.42,
    "computational": 0.54,
    "literature_exact": 0.72,
    "validated": 0.9,
}
CONFIDENCE_WEIGHT = {"low": 0.25, "medium": 0.5, "medium_high": 0.68, "high": 0.82}


def normalize_route_candidate(
    raw: dict[str, Any],
    *,
    default_source_channel: str = "other",
    report_ref: str = "",
    allow_trusted_validated_evidence: bool = False,
    allow_trusted_literature_exact_evidence: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize one proposal and return ``(candidate, rejection_reasons)``."""
    if not isinstance(raw, dict):
        return None, ["candidate_not_object"]

    reasons: list[str] = []
    schema = str(raw.get("schema_version") or RETROSYNTHESIS_CANDIDATE_SCHEMA)
    if schema != RETROSYNTHESIS_CANDIDATE_SCHEMA:
        reasons.append("invalid_candidate_schema")
    if raw.get("no_solved_claim") is not True:
        reasons.append("missing_no_solved_claim")
    if raw.get("not_parent_route_proof") is not True:
        reasons.append("missing_not_parent_route_proof")
    if _contains_solved_claim(raw):
        reasons.append("direct_solved_claim")
    if _contains_raw_reaction(raw):
        reasons.append("raw_reaction_payload")

    product = _canonical_smiles(raw.get("product_smiles"))
    if not product:
        reasons.append("invalid_product_smiles")
    precursor_values = _precursor_values(raw.get("precursor_smiles"))
    precursors = [_canonical_smiles(value) for value in precursor_values]
    if not precursor_values or any(not value for value in precursors):
        reasons.append("invalid_precursor_smiles")
    # Preserve precursor multiplicity: two identical molecules in a
    # homocoupling are a different reaction from one molecule, even though
    # frontier scheduling may later audit that canonical stock subject once.
    precursors = sorted(value for value in precursors if value)
    if product and precursors and product in precursors and len(precursors) == 1:
        reasons.append("identity_proposal")
    if product and precursors:
        admission_reasons = list(
            audit_retrosynthetic_candidate(product, precursors).get("reasons") or []
        )
        # Preserve the long-standing public reason for a single exact identity
        # while using the shared reason for multi-component self-return cycles.
        if product in precursors and len(precursors) == 1:
            admission_reasons = [
                reason
                for reason in admission_reasons
                if reason != "target_or_current_node_self_loop"
            ]
        reasons.extend(admission_reasons)

    normalization_records: list[dict[str, Any]] = []
    acquisition_hints: list[dict[str, Any]] = []
    host_authority_binding = str(raw.get("_host_authority_binding") or "").strip()
    source_channel, source_channel_record = _normalize_source_channel_with_record(
        raw.get("source_channel") or default_source_channel
    )
    producer_is_codex = source_channel.startswith("codex_")
    trusted_validated_binding = bool(
        not producer_is_codex
        and (
            allow_trusted_validated_evidence
            or host_authority_binding == "deterministic_reaction_validation"
        )
    )
    trusted_literature_binding = bool(
        not producer_is_codex
        and (
            allow_trusted_literature_exact_evidence
            or host_authority_binding == "validated_source_detail_literature_step"
        )
    )
    trusted_computational_binding = bool(
        not producer_is_codex
        and host_authority_binding
        and host_authority_binding
        == {
            "chem_enzy": "deterministic_chemenzy_adapter",
            "template": "deterministic_template_adapter",
            "stock": "deterministic_stock_provider",
        }.get(source_channel, "")
    )
    if source_channel_record:
        normalization_records.append(source_channel_record)
        if source_channel_record.get("reason") == "invalid_enum_value":
            acquisition_hints.append(
                _enum_acquisition_hint(
                    source_channel_record,
                    accepted_values=SOURCE_CHANNELS,
                )
            )

    producer_evidence_value = raw.get(
        "producer_evidence_level",
        raw.get("evidence_level"),
    )
    producer_confidence_value = raw.get(
        "producer_confidence",
        raw.get("confidence"),
    )
    producer_evidence_raw = raw.get(
        "producer_evidence_level_raw",
        producer_evidence_value,
    )
    producer_confidence_raw = raw.get(
        "producer_confidence_raw",
        producer_confidence_value,
    )
    producer_evidence_level, producer_evidence_record = (
        _normalize_evidence_level_with_record(producer_evidence_value)
    )
    producer_confidence, producer_confidence_record = (
        _normalize_confidence_with_record(producer_confidence_value)
    )
    # A consensus replay carries both the already-normalized advisory value
    # and the producer's original token.  Re-evaluate that token so malformed
    # metadata remains visible after blackboard/graph reconstruction.
    if "producer_evidence_level_raw" in raw:
        _, producer_evidence_record = _normalize_evidence_level_with_record(
            producer_evidence_raw
        )
    if "producer_confidence_raw" in raw:
        _, producer_confidence_record = _normalize_confidence_with_record(
            producer_confidence_raw
        )
    for record, accepted_values in (
        (producer_evidence_record, EVIDENCE_LEVEL_WEIGHT),
        (producer_confidence_record, CONFIDENCE_WEIGHT),
    ):
        if not record:
            continue
        normalization_records.append(record)
        if record.get("reason") == "invalid_enum_value":
            acquisition_hints.append(
                _enum_acquisition_hint(record, accepted_values=accepted_values)
            )

    # Producer fields are observations, not permissions.  Every serialized
    # producer is unbound until a host adapter emits a private capability for
    # this exact record.  This applies to legacy/provider strings as well as
    # Codex roles: writing ``chem_enzy`` or ``high`` into a blackboard cannot
    # mint computational authority or an independent support group.
    authority_bound = bool(
        (
            trusted_validated_binding
            and producer_evidence_level == "validated"
        )
        or (
            trusted_literature_binding
            and producer_evidence_level == "literature_exact"
        )
        or (
            trusted_computational_binding
            and producer_evidence_level in {"computational", "analogy"}
        )
    )
    unbound_producer = not authority_bound
    authority_evidence_level = producer_evidence_level
    authority_confidence = producer_confidence
    authority_basis = host_authority_binding or "unbound_producer"
    if (
        authority_evidence_level == "validated"
        and not trusted_validated_binding
        and not producer_is_codex
    ):
        # Validation authority must arrive out-of-band from deterministic code;
        # a non-Codex source field cannot validate itself either.
        reasons.append("untrusted_self_validated_evidence_claim")
    literature_exact_downgraded = bool(
        authority_evidence_level == "literature_exact"
        and (
            not trusted_literature_binding
        )
    )
    if literature_exact_downgraded:
        # Only deterministic source-detail adapters may opt into exact
        # literature authority after document/step binding.
        authority_evidence_level = "analogy"
        authority_basis = "untrusted_literature_claim"
        normalization_records.append(
            _authority_normalization_record(
                field="authority_evidence_level",
                input_value="literature_exact",
                normalized_value="analogy",
                reason="trusted_source_detail_binding_missing",
            )
        )
        acquisition_hints.append(
            {
                "schema_version": "route_candidate_acquisition_hint.v1",
                "hint_type": "trusted_source_detail_binding",
                "reason": "literature_exact_authority_not_host_bound",
                "required_binding": "validated_source_detail_literature_step",
            }
        )
    elif authority_evidence_level == "literature_exact":
        authority_basis = host_authority_binding or "trusted_literature_adapter"
    elif authority_evidence_level == "validated":
        authority_basis = host_authority_binding or "trusted_validation_adapter"
    elif trusted_computational_binding and authority_bound:
        authority_basis = host_authority_binding
    if unbound_producer:
        claimed_evidence_level = authority_evidence_level
        authority_evidence_level = "model_only"
        authority_confidence = "low"
        if claimed_evidence_level != authority_evidence_level:
            normalization_records.append(
                _authority_normalization_record(
                    field="authority_evidence_level",
                    input_value=claimed_evidence_level,
                    normalized_value=authority_evidence_level,
                    reason=(
                        "unbound_codex_producer_cannot_set_evidence_authority"
                        if producer_is_codex
                        else "unbound_producer_cannot_set_evidence_authority"
                    ),
                )
            )
        if producer_confidence != authority_confidence:
            normalization_records.append(
                _authority_normalization_record(
                    field="authority_confidence",
                    input_value=producer_confidence,
                    normalized_value=authority_confidence,
                    reason=(
                        "unbound_codex_producer_cannot_set_confidence_authority"
                        if producer_is_codex
                        else "unbound_producer_cannot_set_confidence_authority"
                    ),
                )
            )
        acquisition_hints.append(
            {
                "schema_version": "route_candidate_acquisition_hint.v1",
                "hint_type": "host_evidence_binding",
                "reason": (
                    "codex_producer_metadata_is_advisory_only"
                    if producer_is_codex
                    else "producer_metadata_is_advisory_only"
                ),
                "required_binding": (
                    "deterministic_reaction_validation_or_trusted_source_detail_step"
                    if producer_is_codex
                    else "deterministic_reaction_validation_or_trusted_source_detail_step_or_provider_envelope"
                ),
            }
        )
    source_refs = _dedupe(_texts(raw.get("source_refs")))
    evidence_refs = _dedupe(_texts(raw.get("evidence_refs")))
    source_aliases = _literature_source_aliases([*source_refs, *evidence_refs])
    if authority_evidence_level == "literature_exact" and not any(
        _traceable_literature_ref(ref) for ref in [*source_refs, *evidence_refs]
    ):
        reasons.append("exact_literature_without_source_ref")

    if reasons:
        return None, sorted(set(reasons))

    candidate_id = str(raw.get("candidate_id") or "").strip() or _stable_id(
        "candidate", product, ".".join(precursors), source_channel
    )
    return {
        "schema_version": RETROSYNTHESIS_CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "precursor_set_smiles": ".".join(precursors),
        "reaction_family": str(raw.get("reaction_family") or "unspecified").strip() or "unspecified",
        "transformation_rationale": str(raw.get("transformation_rationale") or "").strip(),
        "source_channel": source_channel,
        "source_refs": source_refs,
        "evidence_refs": evidence_refs,
        # Compatibility fields intentionally expose the host-derived values;
        # every existing scorer therefore fails closed without needing to know
        # about the additive provenance fields.
        "evidence_level": authority_evidence_level,
        "confidence": authority_confidence,
        "producer_evidence_level_raw": _enum_input_text(producer_evidence_raw),
        "producer_evidence_level": producer_evidence_level,
        "producer_confidence_raw": _enum_input_text(producer_confidence_raw),
        "producer_confidence": producer_confidence,
        "authority_evidence_level": authority_evidence_level,
        "authority_confidence": authority_confidence,
        "authority_basis": authority_basis,
        "authority_bound": authority_bound,
        "normalization_records": _dedupe_records(normalization_records),
        "acquisition_hints": _dedupe_records(acquisition_hints),
        "conditions": _dedupe(_texts(raw.get("conditions"))),
        "catalyst": str(raw.get("catalyst") or "").strip(),
        "enzyme": str(raw.get("enzyme") or "").strip(),
        "limitations": _dedupe(
            [
                *_texts(raw.get("limitations")),
                *(["untrusted_literature_exact_claim_downgraded_to_analogy"] if literature_exact_downgraded else []),
            ]
        ),
        "required_validation": _dedupe(_texts(raw.get("required_validation"))),
        "report_ref": str(report_ref or raw.get("report_ref") or "").strip(),
        "support_group": _support_group(
            source_channel,
            authority_evidence_level,
            source_refs,
            evidence_refs,
            authority_bound=authority_bound,
        ),
        "source_identity_aliases": source_aliases,
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }, []


def fuse_route_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    case_id: str = "",
    target_smiles: str = "",
    allow_trusted_validated_evidence: bool = False,
    allow_trusted_literature_exact_evidence: bool = False,
) -> dict[str, Any]:
    """Fuse equivalent product/precursor proposals with preserved provenance."""
    target = _canonical_smiles(target_smiles) if target_smiles else ""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        raw_product = (
            _canonical_smiles(raw.get("product_smiles"))
            if isinstance(raw, dict)
            else ""
        )
        if target and raw_product and raw_product != target:
            rejected.append(
                {
                    "index": index,
                    "candidate_id": str(raw.get("candidate_id") or ""),
                    "reasons": ["candidate_product_does_not_match_requested_target"],
                }
            )
            continue
        report_ref = str(raw.get("report_ref") or "") if isinstance(raw, dict) else ""
        normalized, reasons = normalize_route_candidate(
            raw,
            report_ref=report_ref,
            allow_trusted_validated_evidence=allow_trusted_validated_evidence,
            allow_trusted_literature_exact_evidence=allow_trusted_literature_exact_evidence,
        )
        if normalized is None:
            rejected.append({"index": index, "candidate_id": str((raw or {}).get("candidate_id") or "") if isinstance(raw, dict) else "", "reasons": reasons})
        elif target and normalized["product_smiles"] != target:
            rejected.append({
                "index": index,
                "candidate_id": normalized["candidate_id"],
                "reasons": ["candidate_product_does_not_match_requested_target"],
            })
        else:
            accepted.append(normalized)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        signature = _proposal_signature(row["product_smiles"], row["precursor_smiles"])
        grouped[signature].append(row)

    proposals = [_fuse_group(signature, rows, target=target) for signature, rows in grouped.items()]
    proposals.sort(key=lambda row: (-float(row["rank_score"]), row["consensus_id"]))
    for rank, row in enumerate(proposals, start=1):
        row["rank"] = rank

    channel_counts: dict[str, int] = defaultdict(int)
    for row in accepted:
        channel_counts[row["source_channel"]] += 1
    return {
        "schema_version": ROUTE_CONSENSUS_SCHEMA,
        "case_id": str(case_id),
        "target_smiles": target or str(target_smiles or ""),
        "accepted": bool(proposals),
        "proposals": proposals,
        "rejected_candidates": rejected,
        "source_summary": {
            "candidate_count": len(accepted),
            "rejected_count": len(rejected),
            "proposal_count": len(proposals),
            "channel_counts": dict(sorted(channel_counts.items())),
            "multi_source_proposals": sum(1 for row in proposals if row["source_diversity"] > 1),
            "authority_capped_candidate_count": sum(
                1 for row in accepted if row.get("authority_bound") is False
            ),
            "normalization_record_count": sum(
                len(row.get("normalization_records") or []) for row in accepted
            ),
        },
        "semantics": {
            "advisory_only": True,
            "deterministic_parent_proof_required": True,
            "no_solved_claim": True,
            "authority_ranking": "host_derived",
            "producer_evidence_and_confidence": "advisory_only",
            "unbound_producer_authority_evidence_level": "model_only",
            "unbound_producer_authority_confidence": "low",
        },
    }


def consensus_to_blackboard_proposals(consensus: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt canonical consensus records to the legacy blackboard proposal bus."""
    rows: list[dict[str, Any]] = []
    for proposal in consensus.get("proposals") or []:
        if not isinstance(proposal, dict):
            continue
        status = str(proposal.get("status") or "model_hypothesis")
        independent_source_count = len(_proposal_independent_support_groups(proposal))
        consensus_scope = (
            "multi_source" if independent_source_count > 1 else "correlated_single_source"
        )
        rows.append({
            "schema_version": "retrosynthetic_proposal.v1",
            "proposal_id": f"consensus:{proposal.get('consensus_id')}",
            "proposal_type": "evidence_backed_advisory" if status == "evidence_backed_draft" else "strategic",
            "source_type": (
                "multi_source_consensus"
                if consensus_scope == "multi_source"
                else "correlated_consensus"
            ),
            "consensus_scope": consensus_scope,
            "independent_source_count": independent_source_count,
            "multi_source": consensus_scope == "multi_source",
            "source_channels": list(proposal.get("source_channels") or []),
            "source_support": list(proposal.get("source_records") or []),
            "proposal_label": str(proposal.get("reaction_family") or "consensus proposal"),
            "target_smiles": str(proposal.get("product_smiles") or ""),
            "precursor_smiles": str(proposal.get("precursor_set_smiles") or ""),
            "precursor_component_count": len(proposal.get("precursor_smiles") or []),
            "multi_component_precursor_set": len(proposal.get("precursor_smiles") or []) > 1,
            "transformation_idea": " | ".join(proposal.get("rationales") or []),
            "confidence": str(proposal.get("confidence") or "low"),
            "authority_evidence_level": str(
                proposal.get("authority_evidence_level")
                or proposal.get("evidence_level")
                or "model_only"
            ),
            "authority_policy": str(proposal.get("authority_policy") or "host_derived"),
            "producer_evidence_levels": list(proposal.get("producer_evidence_levels") or []),
            "producer_confidences": list(proposal.get("producer_confidences") or []),
            "consensus_score": float(proposal.get("rank_score") or 0.0),
            "consensus_status": status,
            "recursive_expandable": bool(proposal.get("precursor_smiles")),
            # Canonical consensus is an advisory evidence layer. Executability
            # can only be granted by the separate deterministic template/route
            # validators, never by a source-provided evidence_level string.
            "executable": False,
            "evidence_refs": list(proposal.get("evidence_refs") or []),
            "source_refs": list(proposal.get("source_refs") or []),
            "risk_flags": list(proposal.get("limitations") or []),
            "required_verification": list(proposal.get("required_validation") or []),
            "normalization_records": list(proposal.get("normalization_records") or []),
            "acquisition_hints": list(proposal.get("acquisition_hints") or []),
            "not_exact_literature_segment": status != "evidence_backed_draft",
            "not_parent_route_proof": True,
            "no_solved_claim": True,
        })
    return rows


def _proposal_independent_support_groups(proposal: dict[str, Any]) -> list[str]:
    groups = _dedupe(
        _support_group(
            _normalize_source_channel(record.get("source_channel")),
            _normalize_evidence_level(record.get("evidence_level")),
            _texts(record.get("source_refs")),
            _texts(record.get("evidence_refs")),
            authority_bound=record.get("authority_bound") is True,
        )
        for record in proposal.get("source_records") or []
        if isinstance(record, dict)
    )
    if groups:
        return groups
    legacy_channels = [
        str(channel or "").strip().lower()
        for channel in proposal.get("source_channels") or []
        if str(channel or "").strip()
    ]
    legacy_groups = [
        str(group or "").strip().lower()
        for group in proposal.get("independent_support_groups") or []
        if str(group or "").strip()
    ]
    if any(value.startswith("codex") for value in [*legacy_channels, *legacy_groups]):
        return ["codex_model"]
    if legacy_channels or legacy_groups or proposal.get("source_refs") or proposal.get("evidence_refs"):
        return ["legacy_unverified_support"]
    return []


def validate_retrosynthesis_report_payload(payload: Any) -> list[str]:
    """Validate a child-agent proposal report; returns stable reason codes."""
    if not isinstance(payload, dict):
        return ["proposal_report_payload_not_object"]
    reasons: list[str] = []
    if payload.get("schema_version") != RETROSYNTHESIS_PROPOSAL_REPORT_PAYLOAD_SCHEMA:
        reasons.append("invalid_proposal_report_schema")
    if not str(payload.get("case_id") or "").strip():
        reasons.append("missing_proposal_report_case_id")
    if not str(payload.get("agent_role") or "").strip():
        reasons.append("missing_proposal_report_agent_role")
    if payload.get("no_solved_claim") is not True or _contains_solved_claim(payload):
        reasons.append("proposal_report_direct_solved_claim")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        reasons.append("proposal_report_candidates_not_list")
        candidates = []
    if len(candidates) > 24:
        reasons.append("proposal_report_candidate_limit_exceeded")
    for index, raw in enumerate(candidates):
        _, candidate_reasons = normalize_route_candidate(
            raw if isinstance(raw, dict) else {},
            default_source_channel=_role_source_channel(payload.get("agent_role")),
        )
        reasons.extend(f"proposal_report_candidate:{index}:{reason}" for reason in candidate_reasons)
    return sorted(set(reasons))


def _fuse_group(signature: str, rows: list[dict[str, Any]], *, target: str) -> dict[str, Any]:
    first = rows[0]
    _reconcile_literature_support_groups(rows)
    channels = sorted({row["source_channel"] for row in rows})
    support_groups = sorted({row["support_group"] for row in rows})
    evidence_levels = [row["authority_evidence_level"] for row in rows]
    source_records = [
        {
            "candidate_id": row["candidate_id"],
            "source_channel": row["source_channel"],
            "evidence_level": row["authority_evidence_level"],
            "confidence": row["authority_confidence"],
            "producer_evidence_level_raw": row["producer_evidence_level_raw"],
            "producer_evidence_level": row["producer_evidence_level"],
            "producer_confidence_raw": row["producer_confidence_raw"],
            "producer_confidence": row["producer_confidence"],
            "authority_evidence_level": row["authority_evidence_level"],
            "authority_confidence": row["authority_confidence"],
            "authority_basis": row["authority_basis"],
            "authority_bound": row["authority_bound"],
            "normalization_records": list(row["normalization_records"]),
            "acquisition_hints": list(row["acquisition_hints"]),
            "source_refs": row["source_refs"],
            "evidence_refs": row["evidence_refs"],
            "report_ref": row["report_ref"],
            "support_group": row["support_group"],
            "source_identity_aliases": list(row.get("source_identity_aliases") or []),
        }
        for row in rows
    ]
    confidence_score = _combined_confidence(rows)
    diversity = len(support_groups)
    cross_source_bonus = min(0.12, max(0, diversity - 1) * 0.045)
    target_bonus = 0.03 if target and first["product_smiles"] == target else 0.0
    rank_score = min(0.99, confidence_score + cross_source_bonus + target_bonus)
    status = _proposal_status(evidence_levels, support_groups)
    condition_support = [
        {
            "candidate_id": row["candidate_id"],
            "support_group": row["support_group"],
            "conditions": list(row["conditions"]),
            "catalyst": row["catalyst"],
            "enzyme": row["enzyme"],
            "source_refs": list(row["source_refs"]),
            "evidence_refs": list(row["evidence_refs"]),
        }
        for row in rows
        if row["conditions"] or row["catalyst"] or row["enzyme"]
    ]
    return {
        "schema_version": "route_consensus_proposal.v1",
        "consensus_id": _stable_id("consensus", signature),
        "signature": signature,
        "product_smiles": first["product_smiles"],
        "precursor_smiles": list(first["precursor_smiles"]),
        "precursor_set_smiles": first["precursor_set_smiles"],
        "reaction_family": _best_reaction_family(rows),
        "reaction_families": _dedupe(row["reaction_family"] for row in rows),
        "rationales": _dedupe(row["transformation_rationale"] for row in rows if row["transformation_rationale"]),
        "source_channels": channels,
        "source_channel_count": len(channels),
        "independent_support_groups": support_groups,
        "source_diversity": diversity,
        "support_count": len(rows),
        "source_records": source_records,
        "source_refs": _dedupe(ref for row in rows for ref in row["source_refs"]),
        "evidence_refs": _dedupe(ref for row in rows for ref in row["evidence_refs"]),
        "evidence_level": max(evidence_levels, key=lambda value: EVIDENCE_LEVEL_WEIGHT[value]),
        "authority_evidence_level": max(
            evidence_levels,
            key=lambda value: EVIDENCE_LEVEL_WEIGHT[value],
        ),
        "authority_policy": "host_derived",
        "producer_evidence_levels": _dedupe(
            row["producer_evidence_level"] for row in rows
        ),
        "producer_confidences": _dedupe(row["producer_confidence"] for row in rows),
        "confidence": _confidence_label(rank_score),
        "confidence_score": round(confidence_score, 4),
        "rank_score": round(rank_score, 4),
        "status": status,
        "conditions": _dedupe(value for row in rows for value in row["conditions"]),
        "catalysts": _dedupe(row["catalyst"] for row in rows if row["catalyst"]),
        "enzymes": _dedupe(row["enzyme"] for row in rows if row["enzyme"]),
        "condition_support": condition_support,
        "condition_conflicts": _condition_conflicts(condition_support),
        "limitations": _dedupe(value for row in rows for value in row["limitations"]),
        "required_validation": _dedupe(value for row in rows for value in row["required_validation"]),
        "normalization_records": _dedupe_records(
            record
            for row in rows
            for record in row["normalization_records"]
        ),
        "acquisition_hints": _dedupe_records(
            hint
            for row in rows
            for hint in row["acquisition_hints"]
        ),
        "target_match": bool(target and first["product_smiles"] == target),
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }


def _combined_confidence(rows: list[dict[str, Any]]) -> float:
    # Evidence from duplicate records in one channel is correlated.  Keep only
    # the strongest record per channel before combining support.
    strongest: dict[str, float] = {}
    for row in rows:
        weight = math.sqrt(
            EVIDENCE_LEVEL_WEIGHT[row["authority_evidence_level"]]
            * CONFIDENCE_WEIGHT[row["authority_confidence"]]
        )
        group = row["support_group"]
        strongest[group] = max(strongest.get(group, 0.0), weight)
    residual = 1.0
    for weight in strongest.values():
        residual *= 1.0 - min(0.92, weight)
    return min(0.96, 1.0 - residual)


def _proposal_status(evidence_levels: list[str], support_groups: list[str]) -> str:
    if "validated" in evidence_levels:
        return "validation_claimed_draft"
    if "literature_exact" in evidence_levels:
        return "evidence_backed_draft"
    non_model_groups = [group for group in support_groups if group != "codex_model"]
    if len(non_model_groups) >= 2:
        return "consensus_candidate"
    return "model_hypothesis"


def _support_group(
    source_channel: str,
    evidence_level: str,
    source_refs: list[str],
    evidence_refs: list[str],
    *,
    authority_bound: bool,
) -> str:
    if not authority_bound and source_channel in {"chem_enzy", "template", "stock"}:
        return "codex_model"
    return source_record_support_group(
        source_channel,
        evidence_level,
        source_refs,
        evidence_refs,
    )


def _traceable_literature_ref(value: Any) -> bool:
    return bool(_canonical_literature_ref(value))


def _canonical_literature_ref(value: Any) -> str:
    """Return one syntactically validated article/source alias."""
    return canonical_traceable_source_ref(value)


def _literature_source_aliases(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            alias
            for value in values
            if (alias := _canonical_literature_ref(value))
        },
        key=_source_alias_sort_key,
    )


def _source_alias_sort_key(value: str) -> tuple[int, str]:
    return source_ref_sort_key(value)


def _preferred_source_alias(values: Iterable[str]) -> str:
    aliases = sorted({str(value) for value in values if str(value)}, key=_source_alias_sort_key)
    return aliases[0] if aliases else ""


def _reconcile_literature_support_groups(rows: list[dict[str, Any]]) -> None:
    """Collapse DOI/PMID/PMC/URL/local aliases joined by provenance records.

    A record containing both a DOI and a local/SI reference acts as an explicit
    bridge.  Transitive bridges are resolved within the fused reaction group,
    so article HTML, supporting information and cached copies cannot inflate
    independent support.
    """
    eligible = [
        index
        for index, row in enumerate(rows)
        if row.get("evidence_level") in {"literature_exact", "validated"}
        and row.get("source_channel") == "literature_exact"
        and row.get("source_identity_aliases")
    ]
    parents = {index: index for index in eligible}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for position, left in enumerate(eligible):
        left_aliases = set(rows[left].get("source_identity_aliases") or [])
        for right in eligible[position + 1 :]:
            if not left_aliases.intersection(rows[right].get("source_identity_aliases") or []):
                continue
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root
    components: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        components[find(index)].append(index)
    for indexes in components.values():
        aliases = {
            alias
            for index in indexes
            for alias in rows[index].get("source_identity_aliases") or []
        }
        primary = _preferred_source_alias(aliases)
        if not primary:
            continue
        for index in indexes:
            prefix = "validated" if rows[index].get("evidence_level") == "validated" else "literature"
            rows[index]["support_group"] = f"{prefix}:{primary}"


def _condition_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for field in ("catalyst", "enzyme"):
        values = sorted({str(row.get(field) or "").strip() for row in rows if str(row.get(field) or "").strip()})
        if len(values) > 1:
            conflicts.append({"field": field, "values": values, "requires_review": True})
    condition_sets = sorted({tuple(row.get("conditions") or []) for row in rows if row.get("conditions")})
    if len(condition_sets) > 1:
        conflicts.append({
            "field": "conditions",
            "values": [list(values) for values in condition_sets],
            "requires_review": True,
        })
    return conflicts


def _proposal_signature(product: str, precursors: list[str]) -> str:
    return f"{product}<-{'.'.join(sorted(precursors))}"


def _best_reaction_family(rows: list[dict[str, Any]]) -> str:
    values = [row["reaction_family"] for row in rows if row["reaction_family"] != "unspecified"]
    if not values:
        return "unspecified"
    counts = {value: values.count(value) for value in set(values)}
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _role_source_channel(role: Any) -> str:
    value = str(role or "").lower()
    if "literature" in value:
        return "codex_literature"
    if "enzyme" in value or "chemo" in value:
        return "codex_chemoenzymatic"
    if "critic" in value:
        return "codex_critic"
    return "codex_strategy"


def _normalize_source_channel(value: Any) -> str:
    normalized, _ = _normalize_source_channel_with_record(value)
    return normalized


def _normalize_source_channel_with_record(
    value: Any,
) -> tuple[str, dict[str, Any] | None]:
    text = str(value or "other").strip().lower().replace("-", "_")
    aliases = {
        "chemenzy": "chem_enzy",
        "exact_literature": "literature_exact",
        "analog_literature": "literature_analogy",
        "codex_enzyme": "codex_chemoenzymatic",
    }
    normalized = aliases.get(text, text)
    if normalized not in SOURCE_CHANNELS:
        return "other", _enum_normalization_record(
            field="source_channel",
            input_value=value,
            normalized_value="other",
            reason="invalid_enum_value",
        )
    if normalized != text:
        return normalized, _enum_normalization_record(
            field="source_channel",
            input_value=value,
            normalized_value=normalized,
            reason="enum_alias_canonicalized",
        )
    return normalized, None


def _normalize_evidence_level(value: Any) -> str:
    normalized, _ = _normalize_evidence_level_with_record(value)
    return normalized


def _normalize_evidence_level_with_record(
    value: Any,
) -> tuple[str, dict[str, Any] | None]:
    text = str(value or "model_only").strip().lower().replace("-", "_")
    aliases = {"exact": "literature_exact", "model": "model_only", "verified": "validated"}
    normalized = aliases.get(text, text)
    if normalized not in EVIDENCE_LEVEL_WEIGHT:
        return "model_only", _enum_normalization_record(
            field="producer_evidence_level",
            input_value=value,
            normalized_value="model_only",
            reason="invalid_enum_value",
        )
    if normalized != text:
        return normalized, _enum_normalization_record(
            field="producer_evidence_level",
            input_value=value,
            normalized_value=normalized,
            reason="enum_alias_canonicalized",
        )
    return normalized, None


def _normalize_confidence(value: Any) -> str:
    normalized, _ = _normalize_confidence_with_record(value)
    return normalized


def _normalize_confidence_with_record(
    value: Any,
) -> tuple[str, dict[str, Any] | None]:
    text = str(value or "low").strip().lower().replace("-", "_")
    if text not in CONFIDENCE_WEIGHT:
        return "low", _enum_normalization_record(
            field="producer_confidence",
            input_value=value,
            normalized_value="low",
            reason="invalid_enum_value",
        )
    return text, None


def _enum_normalization_record(
    *,
    field: str,
    input_value: Any,
    normalized_value: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "route_candidate_normalization.v1",
        "field": field,
        "input_value": _enum_input_text(input_value),
        "normalized_value": normalized_value,
        "reason": reason,
        "authority_effect": "conservative_default",
    }


def _authority_normalization_record(
    *,
    field: str,
    input_value: Any,
    normalized_value: str,
    reason: str,
) -> dict[str, Any]:
    record = _enum_normalization_record(
        field=field,
        input_value=input_value,
        normalized_value=normalized_value,
        reason=reason,
    )
    record["authority_effect"] = "authority_capped"
    return record


def _enum_acquisition_hint(
    record: dict[str, Any],
    *,
    accepted_values: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema_version": "route_candidate_acquisition_hint.v1",
        "hint_type": "producer_enum_correction",
        "field": str(record.get("field") or ""),
        "reason": str(record.get("reason") or "invalid_enum_value"),
        "accepted_values": sorted(str(value) for value in accepted_values),
    }


def _enum_input_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _confidence_label(score: float) -> str:
    if score >= 0.78:
        return "high"
    if score >= 0.6:
        return "medium_high"
    if score >= 0.4:
        return "medium"
    return "low"


def _canonical_smiles(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _precursor_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(".") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_precursor_values(item))
        return out
    return []


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _contains_raw_reaction(value: Any) -> bool:
    forbidden_keys = {"rxn", "rxn_smiles", "reaction_smiles", "raw_reaction", "reaction_candidates"}
    if isinstance(value, dict):
        return any(str(key).lower() in forbidden_keys or _contains_raw_reaction(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_reaction(item) for item in value)
    return isinstance(value, str) and ">>" in value


def _contains_solved_claim(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "solved" and item is True:
                return True
            if lowered in {"status", "route_status", "verdict"} and str(item).strip().lower() == "solved":
                return True
            if _contains_solved_claim(item):
                return True
    if isinstance(value, (list, tuple)):
        return any(_contains_solved_claim(item) for item in value)
    return False


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_records(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(value))
    return rows
