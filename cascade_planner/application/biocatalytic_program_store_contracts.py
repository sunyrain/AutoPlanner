"""Contracts for durable, non-authoritative biocatalytic Program admissions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


BIOCATALYTIC_PROGRAM_ADMISSION_POLICY = (
    "validated_biocatalytic_program_shadow_admission.v1"
)
BIOCATALYTIC_PROGRAM_ADMISSION_EVENT_SCHEMA = (
    "biocatalytic_program_admission_event.v1"
)
BIOCATALYTIC_PROGRAM_ADMISSION_RESULT_SCHEMA = (
    "biocatalytic_program_admission_result.v1"
)
BIOCATALYTIC_PROGRAM_STORE_REPLAY_SCHEMA = "biocatalytic_program_store_replay.v1"
BIOCATALYTIC_PROGRAM_STORE_STATUS_SCHEMA = "biocatalytic_program_store_status.v1"
BIOCATALYTIC_PROGRAM_STORE_ORACLE_SCHEMA = "biocatalytic_program_store_oracle.v1"
BIOCATALYSIS_VALIDATION_PACK_SCHEMA = "biocatalysis_program_validation_pack.v1"
VALIDATION_PACK_KEYS = {
    "schema_version",
    "run_id",
    "route_id",
    "validations",
    "semantics",
    "content_sha256",
}
VALIDATION_PACK_SEMANTICS = {
    "records_remain_claim_bound": True,
    "pack_does_not_grant_route_completion": True,
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
    "validated_biocatalytic_programs_only": True,
    "baseline_program_route_retained_as_fallback": True,
    "shadow_admission_only": True,
    "cannot_mutate_canonical_graph": True,
    "cannot_grant_reaction_proof": True,
    "cannot_grant_route_completion": True,
    "edge_ids_remain_production_route_authority": True,
}


class BiocatalyticProgramStoreError(RuntimeError):
    """Base error for a durable biocatalytic Program shadow store."""


class BiocatalyticProgramAdmissionDisabled(BiocatalyticProgramStoreError):
    """A caller attempted a write without the explicit admission gate."""


class BiocatalyticProgramStoreCorruption(BiocatalyticProgramStoreError):
    """An immutable event or referenced CAS object failed replay."""


def validation_pack(
    *, run_id: str, route_id: str, validations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return with_digest(
        {
            "schema_version": BIOCATALYSIS_VALIDATION_PACK_SCHEMA,
            "run_id": run_id,
            "route_id": route_id,
            "validations": [dict(value) for value in validations],
            "semantics": dict(VALIDATION_PACK_SEMANTICS),
        }
    )


def admitted_entities(bundle: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    proposals = dict(bundle.get("program_proposals") or {})
    program_ids = sorted(
        key
        for key, row in proposals.items()
        if isinstance(row, Mapping)
        and row.get("eligible_for_shadow_admission") is True
        and row.get("status") == "admission_ready"
    )
    program_set = set(program_ids)
    routes = dict(bundle.get("route_candidates") or {})
    route_ids = sorted(
        key
        for key, row in routes.items()
        if isinstance(row, Mapping)
        and row.get("superstep_program_id") in program_set
        and row.get("substitution_validated") is True
        and row.get("eligible_for_program_optimizer") is True
        and row.get("eligible_for_route_completion") is False
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
        "schema_version": BIOCATALYTIC_PROGRAM_ADMISSION_EVENT_SCHEMA,
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
        "admission_policy": BIOCATALYTIC_PROGRAM_ADMISSION_POLICY,
        "semantics": dict(EVENT_SEMANTICS),
    }
    identity = event_identity(event)
    event["event_identity_sha256"] = identity
    event["event_id"] = f"biocatalytic-program-admission:sha256:{identity}"
    return with_digest(event)


def event_identity(event: Mapping[str, Any]) -> str:
    return strict_canonical_json_sha256(
        {
            "schema_version": BIOCATALYTIC_PROGRAM_ADMISSION_EVENT_SCHEMA,
            "run_id": event.get("run_id"),
            "source_graph_revision": event.get("source_graph_revision"),
            "source_graph_scientific_sha256": event.get(
                "source_graph_scientific_sha256"
            ),
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
            "admission_policy": BIOCATALYTIC_PROGRAM_ADMISSION_POLICY,
        }
    )


def with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "BIOCATALYSIS_VALIDATION_PACK_SCHEMA",
    "BIOCATALYTIC_PROGRAM_ADMISSION_EVENT_SCHEMA",
    "BIOCATALYTIC_PROGRAM_ADMISSION_POLICY",
    "BIOCATALYTIC_PROGRAM_ADMISSION_RESULT_SCHEMA",
    "BIOCATALYTIC_PROGRAM_STORE_ORACLE_SCHEMA",
    "BIOCATALYTIC_PROGRAM_STORE_REPLAY_SCHEMA",
    "BIOCATALYTIC_PROGRAM_STORE_STATUS_SCHEMA",
    "BiocatalyticProgramAdmissionDisabled",
    "BiocatalyticProgramStoreCorruption",
    "BiocatalyticProgramStoreError",
    "EVENT_KEYS",
    "EVENT_SEMANTICS",
    "VALIDATION_PACK_KEYS",
    "VALIDATION_PACK_SEMANTICS",
    "admission_counts",
    "admission_event",
    "admitted_entities",
    "event_identity",
    "validation_pack",
    "with_digest",
]
