"""Contracts for durable exact-boundary experimental observation Claims."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENTAL_CLAIM_ADMISSION_POLICY = "exact_boundary_experimental_claim_admission.v1"
EXPERIMENTAL_CLAIM_ADMISSION_EVENT_SCHEMA = "experimental_claim_admission_event.v1"
EXPERIMENTAL_CLAIM_ADMISSION_RESULT_SCHEMA = "experimental_claim_admission_result.v1"
EXPERIMENTAL_CLAIM_STORE_REPLAY_SCHEMA = "experimental_claim_store_replay.v1"
EXPERIMENTAL_CLAIM_STORE_STATUS_SCHEMA = "experimental_claim_store_status.v1"
EXPERIMENTAL_CLAIM_STORE_ORACLE_SCHEMA = "experimental_claim_store_oracle.v1"
EXPERIMENTAL_VALIDATION_PACK_SCHEMA = "experimental_claim_validation_pack.v1"

VALIDATION_PACK_KEYS = {
    "schema_version",
    "run_id",
    "route_id",
    "validations",
    "semantics",
    "content_sha256",
}
VALIDATION_PACK_SEMANTICS = {
    "input_order_is_not_authority": True,
    "records_remain_domain_and_claim_bound": True,
    "pack_does_not_grant_proof_completion_or_acceptance": True,
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
    "source_projection_sha256",
    "source_projection_ref",
    "source_discovery_sha256",
    "source_discovery_ref",
    "validation_pack_sha256",
    "validation_pack_ref",
    "claim_set_sha256",
    "claim_set_ref",
    "claim_ids",
    "counts",
    "claim_set_oracle",
    "admission_policy",
    "semantics",
    "content_sha256",
}
EVENT_SEMANTICS = {
    "append_only_content_addressed_event": True,
    "exact_boundary_observations_only": True,
    "positive_negative_and_inconclusive_claims_are_persistable": True,
    "cannot_mutate_canonical_graph": True,
    "cannot_create_canonical_reaction_proof": True,
    "cannot_grant_program_admission_route_completion_or_acceptance": True,
    "cannot_mutate_or_disable_capability_catalog": True,
    "edge_ids_remain_production_route_authority": True,
}


class ExperimentalClaimStoreError(RuntimeError):
    """Base error for durable experimental Claim admission or replay."""


class ExperimentalClaimAdmissionDisabled(ExperimentalClaimStoreError):
    """A write was attempted without the explicit Claim admission gate."""


class ExperimentalClaimStoreCorruption(ExperimentalClaimStoreError):
    """An immutable Claim event or referenced object failed replay."""


def validation_pack(
    *, run_id: str, route_id: str, validations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = sorted(
        (dict(value) for value in validations),
        key=lambda row: (
            str(row.get("schema_version") or ""),
            str(row.get("validation_id") or ""),
            str(row.get("content_sha256") or ""),
        ),
    )
    return with_digest(
        {
            "schema_version": EXPERIMENTAL_VALIDATION_PACK_SCHEMA,
            "run_id": run_id,
            "route_id": route_id,
            "validations": rows,
            "semantics": dict(VALIDATION_PACK_SEMANTICS),
        }
    )


def admission_event(
    *,
    run_id: str,
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    pack: Mapping[str, Any],
    claim_set: Mapping[str, Any],
    refs: Mapping[str, Mapping[str, Any]],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": EXPERIMENTAL_CLAIM_ADMISSION_EVENT_SCHEMA,
        "run_id": run_id,
        "source_graph_revision": int(graph.get("revision") or 0),
        "source_graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "source_graph_ref": dict(refs["graph"]),
        "source_route_id": str(route.get("route_id") or ""),
        "source_route_sha256": strict_canonical_json_sha256(route),
        "source_route_ref": dict(refs["route"]),
        "source_projection_sha256": str(projection.get("content_sha256") or ""),
        "source_projection_ref": dict(refs["projection"]),
        "source_discovery_sha256": str(discovery.get("content_sha256") or ""),
        "source_discovery_ref": dict(refs["discovery"]),
        "validation_pack_sha256": str(pack.get("content_sha256") or ""),
        "validation_pack_ref": dict(refs["validation_pack"]),
        "claim_set_sha256": str(claim_set.get("content_sha256") or ""),
        "claim_set_ref": dict(refs["claim_set"]),
        "claim_ids": sorted(dict(claim_set.get("claims") or {})),
        "counts": dict(claim_set.get("counts") or {}),
        "claim_set_oracle": dict(oracle),
        "admission_policy": EXPERIMENTAL_CLAIM_ADMISSION_POLICY,
        "semantics": dict(EVENT_SEMANTICS),
    }
    identity = event_identity(event)
    event["event_identity_sha256"] = identity
    event["event_id"] = f"experimental-claim-admission:sha256:{identity}"
    return with_digest(event)


def event_identity(event: Mapping[str, Any]) -> str:
    return strict_canonical_json_sha256(
        {
            "schema_version": EXPERIMENTAL_CLAIM_ADMISSION_EVENT_SCHEMA,
            "run_id": event.get("run_id"),
            "source_graph_revision": event.get("source_graph_revision"),
            "source_graph_scientific_sha256": event.get(
                "source_graph_scientific_sha256"
            ),
            "source_route_sha256": event.get("source_route_sha256"),
            "source_projection_sha256": event.get("source_projection_sha256"),
            "source_discovery_sha256": event.get("source_discovery_sha256"),
            "validation_pack_sha256": event.get("validation_pack_sha256"),
            "claim_set_sha256": event.get("claim_set_sha256"),
            "artifact_sha256s": {
                key: dict(event.get(f"{key}_ref") or {}).get("sha256")
                for key in (
                    "source_graph",
                    "source_route",
                    "source_projection",
                    "source_discovery",
                    "validation_pack",
                    "claim_set",
                )
            },
            "admission_policy": EXPERIMENTAL_CLAIM_ADMISSION_POLICY,
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
    "EXPERIMENTAL_CLAIM_ADMISSION_EVENT_SCHEMA",
    "EXPERIMENTAL_CLAIM_ADMISSION_POLICY",
    "EXPERIMENTAL_CLAIM_ADMISSION_RESULT_SCHEMA",
    "EXPERIMENTAL_CLAIM_STORE_ORACLE_SCHEMA",
    "EXPERIMENTAL_CLAIM_STORE_REPLAY_SCHEMA",
    "EXPERIMENTAL_CLAIM_STORE_STATUS_SCHEMA",
    "EXPERIMENTAL_VALIDATION_PACK_SCHEMA",
    "ExperimentalClaimAdmissionDisabled",
    "ExperimentalClaimStoreCorruption",
    "ExperimentalClaimStoreError",
    "VALIDATION_PACK_KEYS",
    "VALIDATION_PACK_SEMANTICS",
    "admission_event",
    "event_identity",
    "validation_pack",
    "with_digest",
]
