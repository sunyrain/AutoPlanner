"""Strict request parsing for Program innovation review and admission routes."""

from __future__ import annotations

from typing import Any, Mapping


def program_innovation_payload(
    value: Mapping[str, Any], *, allow_reported_candidates: bool = False
) -> dict[str, Any]:
    route_id = str(value.get("route_id") or "").strip()
    capabilities = value.get("capabilities")
    mechanism_proposals = value.get("mechanism_proposals") or []
    validations = value.get("validations") or []
    reported_candidate_packs = value.get("reported_candidate_packs") or []
    if not route_id:
        raise ValueError("route_id_required")
    if not isinstance(capabilities, (dict, list)):
        raise ValueError("capabilities_must_be_an_object_or_list")
    if not isinstance(mechanism_proposals, list):
        raise ValueError("mechanism_proposals_must_be_a_list")
    if not isinstance(validations, list):
        raise ValueError("validations_must_be_a_list")
    if not isinstance(reported_candidate_packs, list):
        raise ValueError("reported_candidate_packs_must_be_a_list")
    if reported_candidate_packs and not allow_reported_candidates:
        raise ValueError("reported_candidate_packs_are_review_only")
    result = {
        "route_id": route_id,
        "capabilities": capabilities,
        "mechanism_proposals": mechanism_proposals,
        "validations": validations,
    }
    if allow_reported_candidates:
        result["reported_candidate_packs"] = reported_candidate_packs
    return result


__all__ = ["program_innovation_payload"]
