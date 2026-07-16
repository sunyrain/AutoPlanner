"""Adapt fully restitched mechanism Programs into the common route space."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.program_route_candidate_contracts import (
    ProgramRouteCandidateError,
)
from cascade_planner.application.program_route_candidate_factory import (
    build_program_route_candidate,
    canonical_route_authority_snapshot,
    canonical_route_metrics,
    normalize_strings,
    program_execution_domains,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def compile_mechanism_program_route_candidates(
    route: Mapping[str, Any],
    programs: Mapping[str, Any],
    mechanism_bundle: Mapping[str, Any],
    *,
    source_route_sha256: str,
    source_projection_sha256: str,
    source_discovery_sha256: str,
) -> dict[str, dict[str, Any]]:
    proposals = dict(mechanism_bundle.get("program_proposals") or {})
    candidates: dict[str, dict[str, Any]] = {}
    for route_candidate_id, raw_variant in sorted(
        dict(mechanism_bundle.get("route_candidates") or {}).items()
    ):
        variant = dict(raw_variant)
        proposal_id = str(variant.get("mechanism_program_id") or "")
        proposal = dict(proposals.get(proposal_id) or {})
        if not proposal:
            raise ProgramRouteCandidateError(
                f"program_candidate_mechanism_missing:{route_candidate_id}"
            )
        candidate = _candidate(
            dict(route),
            dict(programs),
            route_candidate_id=str(route_candidate_id),
            variant=variant,
            proposal=proposal,
            source_route_sha256=source_route_sha256,
            source_projection_sha256=source_projection_sha256,
            source_discovery_sha256=source_discovery_sha256,
            source_bundle_sha256=str(mechanism_bundle.get("content_sha256") or ""),
        )
        candidates[candidate["candidate_id"]] = candidate
    return candidates


def _candidate(
    route: dict[str, Any],
    programs: dict[str, Any],
    *,
    route_candidate_id: str,
    variant: dict[str, Any],
    proposal: dict[str, Any],
    source_route_sha256: str,
    source_projection_sha256: str,
    source_discovery_sha256: str,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    selected = [str(value) for value in variant.get("selected_program_ids") or []]
    fallback = [str(value) for value in variant.get("fallback_program_ids") or []]
    proposal_id = str(proposal.get("program_id") or "")
    validation_gate = dict(proposal.get("validation_plan") or {})
    validated = validation_gate.get("accepted") is True
    mechanism_support = str(dict(proposal.get("validation_vector") or {}).get("mechanism") or "")
    if (
        variant.get("full_candidate_route_restitched") is not True
        or not selected
        or proposal_id not in selected
        or any(value not in programs and value != proposal_id for value in selected)
        or variant.get("eligible_for_program_optimizer") is not validated
    ):
        raise ProgramRouteCandidateError(
            f"program_candidate_mechanism_mapping_invalid:{route_candidate_id}"
        )
    metrics = canonical_route_metrics(
        route,
        physical=int(variant.get("physical_step_count") or 0),
        chemical=int(variant.get("chemical_step_equivalent_count") or 0),
        replaced_edges=[str(value) for value in variant.get("replaced_edge_ids") or []],
        substitution_validated=validated,
        specialized_validation_deficit=0 if validated else 1,
    )
    metrics["minimum_proof_level"] = 0
    if validated:
        metrics["reaction_validation_deficit_count"] = 0
        metrics["condition_deficit_count"] = 0
    else:
        metrics["reaction_validation_deficit_count"] = max(
            1, int(metrics["reaction_validation_deficit_count"])
        )
        metrics["condition_deficit_count"] = max(1, int(metrics["condition_deficit_count"]))
    metrics["source_deficit_count"] = max(1, int(metrics["source_deficit_count"]))
    metrics["risk_data_deficit_count"] = max(1, int(metrics["risk_data_deficit_count"]))
    authority_snapshot = {
        "canonical_route": canonical_route_authority_snapshot(route),
        "mechanism_program_id": proposal_id,
        "source_innovation_id": str(proposal.get("source_innovation_id") or ""),
        "mechanism_support": str(
            dict(proposal.get("validation_vector") or {}).get("mechanism") or ""
        ),
        "replaced_edge_ids": [str(value) for value in variant.get("replaced_edge_ids") or []],
        "full_candidate_route_restitched": True,
    }
    return build_program_route_candidate(
        candidate_id=(
            "program-route:mechanism:"
            + strict_canonical_json_sha256(
                {
                    "route_candidate_id": route_candidate_id,
                    "program_id": proposal_id,
                }
            )[:24]
        ),
        source_kind="mechanism",
        source_route_id=str(route.get("route_id") or ""),
        program_ids=selected,
        fallback_program_ids=fallback,
        substitution_program_ids=[proposal_id],
        execution_domains=sorted(
            {
                *program_execution_domains(
                    [value for value in selected if value in programs], programs
                ),
                "chemical",
            }
        ),
        metrics=metrics,
        shadow_optimizer=validated,
        specialized_validation_ids=[
            str(value) for value in validation_gate.get("accepted_validation_ids") or []
        ],
        source_refs=normalize_strings(dict(proposal.get("anchor") or {}).get("source_refs")),
        source_artifact_sha256s=[
            source_route_sha256,
            source_projection_sha256,
            source_discovery_sha256,
            source_bundle_sha256,
        ],
        warning_codes=sorted(
            {
                *normalize_strings(proposal.get("warning_codes")),
                "FULL_CANDIDATE_ROUTE_RESTITCHED",
                f"MECHANISM_SUPPORT_{mechanism_support.upper()}",
                *(
                    ["MECHANISM_VALIDATION_BOUND"]
                    if validated
                    else ["MECHANISM_REACTION_PROOF_REQUIRED"]
                ),
            }
        ),
        authority_snapshot=authority_snapshot,
    )


__all__ = ["compile_mechanism_program_route_candidates"]
