"""Contracts for durable, non-authoritative mechanism Program admissions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


MECHANISM_PROGRAM_ADMISSION_POLICY = "validated_one_hop_mechanism_program_shadow_admission.v1"
MECHANISM_PROGRAM_ADMISSION_EVENT_SCHEMA = "mechanism_program_admission_event.v1"
MECHANISM_PROGRAM_ADMISSION_RESULT_SCHEMA = "mechanism_program_admission_result.v1"
MECHANISM_PROGRAM_STORE_REPLAY_SCHEMA = "mechanism_program_store_replay.v1"
MECHANISM_PROGRAM_STORE_STATUS_SCHEMA = "mechanism_program_store_status.v1"
MECHANISM_PROGRAM_STORE_ORACLE_SCHEMA = "mechanism_program_store_oracle.v1"
MECHANISM_VALIDATION_PACK_SCHEMA = "mechanism_program_validation_pack.v1"
VALIDATION_PACK_KEYS = {
    "schema_version",
    "run_id",
    "route_id",
    "validations",
    "semantics",
    "content_sha256",
}
VALIDATION_PACK_SEMANTICS = {
    "records_remain_exact_program_innovation_and_boundary_bound": True,
    "success_does_not_promote_anchor_evidence": True,
    "pack_does_not_grant_reaction_proof_or_route_completion": True,
}
EVENT_KEYS = {
    "schema_version",
    "event_id",
    "event_identity_sha256",
    "run_id",
    "source_graph_revision",
    "source_graph_scientific_sha256",
    "source_graph_ref",
    "source_route_id",
    "source_route_sha256",
    "source_route_ref",
    "baseline_projection_sha256",
    "baseline_projection_ref",
    "discovery_sha256",
    "discovery_ref",
    "bundle_sha256",
    "bundle_ref",
    "validation_pack_sha256",
    "validation_pack_ref",
    "admitted_program_ids",
    "admitted_route_candidate_ids",
    "counts",
    "bundle_oracle",
    "admission_policy",
    "semantics",
    "content_sha256",
}
EVENT_SEMANTICS = {
    "append_only_content_addressed_event": True,
    "validated_one_hop_mechanism_programs_only": True,
    "exact_route_restitch_required": True,
    "anchor_evidence_does_not_report_or_support_the_extrapolated_reaction": True,
    "baseline_program_route_retained_as_fallback": True,
    "shadow_admission_only": True,
    "cannot_mutate_canonical_graph": True,
    "cannot_grant_reaction_proof_route_completion_or_acceptance": True,
    "edge_ids_remain_production_route_authority": True,
}


class MechanismProgramStoreError(RuntimeError):
    """Base error for mechanism Program shadow admission or replay."""


class MechanismProgramAdmissionDisabled(MechanismProgramStoreError):
    """A write was attempted without explicit mechanism admission."""


class MechanismProgramStoreCorruption(MechanismProgramStoreError):
    """A mechanism event or referenced CAS object failed replay."""


def validation_pack(
    *, run_id: str, route_id: str, validations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = sorted(
        (dict(value) for value in validations),
        key=lambda row: (
            str(row.get("validation_id") or ""),
            str(row.get("content_sha256") or ""),
        ),
    )
    return with_digest(
        {
            "schema_version": MECHANISM_VALIDATION_PACK_SCHEMA,
            "run_id": run_id,
            "route_id": route_id,
            "validations": rows,
            "semantics": dict(VALIDATION_PACK_SEMANTICS),
        }
    )


def admitted_entities(bundle: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    proposals = dict(bundle.get("program_proposals") or {})
    program_ids = sorted(
        program_id
        for program_id, value in proposals.items()
        if isinstance(value, Mapping)
        and value.get("eligible_for_shadow_optimizer") is True
        and value.get("status") == "shadow_ready"
        and dict(value.get("validation_plan") or {}).get("accepted") is True
        and value.get("proposal_kind") == "mechanism_extrapolation"
    )
    selected = set(program_ids)
    route_ids = sorted(
        route_id
        for route_id, value in dict(bundle.get("route_candidates") or {}).items()
        if isinstance(value, Mapping)
        and value.get("mechanism_program_id") in selected
        and value.get("full_candidate_route_restitched") is True
        and value.get("eligible_for_program_optimizer") is True
        and value.get("eligible_for_route_completion") is False
        and value.get("fallback_program_ids")
    )
    return program_ids, route_ids


def admission_counts(
    bundle: Mapping[str, Any], program_ids: Sequence[str], route_ids: Sequence[str]
) -> dict[str, int]:
    return {
        "admitted_programs": len(program_ids),
        "admitted_route_candidates": len(route_ids),
        "bundle_program_proposals": len(dict(bundle.get("program_proposals") or {})),
        "bundle_route_candidates": len(dict(bundle.get("route_candidates") or {})),
    }


def admission_event(
    *,
    run_id: str,
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    pack: Mapping[str, Any],
    refs: Mapping[str, Mapping[str, Any]],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    program_ids, route_ids = admitted_entities(bundle)
    event: dict[str, Any] = {
        "schema_version": MECHANISM_PROGRAM_ADMISSION_EVENT_SCHEMA,
        "run_id": run_id,
        "source_graph_revision": int(graph.get("revision") or 0),
        "source_graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "source_graph_ref": dict(refs["graph"]),
        "source_route_id": str(route.get("route_id") or ""),
        "source_route_sha256": strict_canonical_json_sha256(route),
        "source_route_ref": dict(refs["route"]),
        "baseline_projection_sha256": str(projection.get("content_sha256") or ""),
        "baseline_projection_ref": dict(refs["projection"]),
        "discovery_sha256": str(discovery.get("content_sha256") or ""),
        "discovery_ref": dict(refs["discovery"]),
        "bundle_sha256": str(bundle.get("content_sha256") or ""),
        "bundle_ref": dict(refs["bundle"]),
        "validation_pack_sha256": str(pack.get("content_sha256") or ""),
        "validation_pack_ref": dict(refs["validation_pack"]),
        "admitted_program_ids": program_ids,
        "admitted_route_candidate_ids": route_ids,
        "counts": admission_counts(bundle, program_ids, route_ids),
        "bundle_oracle": dict(oracle),
        "admission_policy": MECHANISM_PROGRAM_ADMISSION_POLICY,
        "semantics": dict(EVENT_SEMANTICS),
    }
    identity = event_identity(event)
    event["event_identity_sha256"] = identity
    event["event_id"] = f"mechanism-program-admission:sha256:{identity}"
    return with_digest(event)


def event_identity(event: Mapping[str, Any]) -> str:
    return strict_canonical_json_sha256(
        {
            "schema_version": MECHANISM_PROGRAM_ADMISSION_EVENT_SCHEMA,
            "run_id": event.get("run_id"),
            "source_graph_revision": event.get("source_graph_revision"),
            "source_graph_scientific_sha256": event.get("source_graph_scientific_sha256"),
            "source_route_sha256": event.get("source_route_sha256"),
            "baseline_projection_sha256": event.get("baseline_projection_sha256"),
            "discovery_sha256": event.get("discovery_sha256"),
            "bundle_sha256": event.get("bundle_sha256"),
            "validation_pack_sha256": event.get("validation_pack_sha256"),
            "artifact_sha256s": {
                key: dict(event.get(f"{key}_ref") or {}).get("sha256")
                for key in (
                    "source_graph",
                    "source_route",
                    "baseline_projection",
                    "discovery",
                    "bundle",
                    "validation_pack",
                )
            },
            "admission_policy": MECHANISM_PROGRAM_ADMISSION_POLICY,
        }
    )


def with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "EVENT_KEYS",
    "EVENT_SEMANTICS",
    "MECHANISM_PROGRAM_ADMISSION_EVENT_SCHEMA",
    "MECHANISM_PROGRAM_ADMISSION_POLICY",
    "MECHANISM_PROGRAM_ADMISSION_RESULT_SCHEMA",
    "MECHANISM_PROGRAM_STORE_ORACLE_SCHEMA",
    "MECHANISM_PROGRAM_STORE_REPLAY_SCHEMA",
    "MECHANISM_PROGRAM_STORE_STATUS_SCHEMA",
    "MECHANISM_VALIDATION_PACK_SCHEMA",
    "MechanismProgramAdmissionDisabled",
    "MechanismProgramStoreCorruption",
    "MechanismProgramStoreError",
    "VALIDATION_PACK_KEYS",
    "VALIDATION_PACK_SEMANTICS",
    "admission_counts",
    "admission_event",
    "admitted_entities",
    "event_identity",
    "validation_pack",
    "with_digest",
]
