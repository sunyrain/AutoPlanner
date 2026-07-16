"""Replay validation for immutable experimental Claim admission events."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.application.experimental_claim_contracts import (
    validate_experimental_claim_set,
)
from cascade_planner.application.experimental_claim_store_contracts import (
    EVENT_KEYS,
    EVENT_SEMANTICS,
    EXPERIMENTAL_CLAIM_ADMISSION_EVENT_SCHEMA,
    EXPERIMENTAL_CLAIM_ADMISSION_POLICY,
    EXPERIMENTAL_VALIDATION_PACK_SCHEMA,
    VALIDATION_PACK_KEYS,
    VALIDATION_PACK_SEMANTICS,
    ExperimentalClaimStoreCorruption,
    event_identity,
)
from cascade_planner.application.transformation_programs import (
    program_projection_oracle,
)
from cascade_planner.runtime.artifact_store import ArtifactRef, ArtifactStore, ArtifactStoreError
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256
from cascade_planner.application.experimental_claim_store_projection import (
    reproject_experimental_claim_materials,
)


_REF_NAMES = (
    "source_graph",
    "source_route",
    "source_projection",
    "source_discovery",
    "validation_pack",
    "claim_set",
)


@dataclass(frozen=True, slots=True)
class ExperimentalClaimAdmissionRecord:
    event: dict[str, Any]
    graph: dict[str, Any]
    route: dict[str, Any]
    projection: dict[str, Any]
    discovery: dict[str, Any]
    validation_pack: dict[str, Any]
    claim_set: dict[str, Any]


def load_experimental_claim_admission_record(
    path: Path, *, run_id: str, artifacts: ArtifactStore
) -> ExperimentalClaimAdmissionRecord:
    event = _read_event(path)
    reasons = _event_envelope_reasons(event, path=path, run_id=run_id)
    refs, payloads = _read_payloads(event, artifacts)
    graph, route, projection, discovery, pack, claim_set = (
        payloads[key] for key in _REF_NAMES
    )
    reasons.extend(_ref_reasons(event, refs))
    try:
        reasons.extend(
            _source_binding_reasons(
                event, graph, route, projection, discovery, pack, claim_set
            )
        )
        projected = reproject_experimental_claim_materials(
            graph,
            route,
            projection,
            discovery,
            list(pack.get("validations") or []),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentalClaimStoreCorruption(
            f"experimental_claim_reprojection_failed:{type(exc).__name__}"
        ) from exc
    expected_claim_set = projected["experimental_claims"]
    expected_oracle = projected["experimental_claims_oracle"]
    if claim_set != expected_claim_set:
        reasons.append("claim_set_reprojection_mismatch")
    if event.get("claim_set_oracle") != expected_oracle:
        reasons.append("claim_set_oracle_replay_mismatch")
    if event.get("claim_ids") != sorted(expected_claim_set["claims"]):
        reasons.append("claim_ids_invalid")
    if event.get("counts") != expected_claim_set.get("counts"):
        reasons.append("claim_counts_invalid")
    if not expected_claim_set["claims"]:
        reasons.append("claim_admission_empty")
    if reasons:
        raise ExperimentalClaimStoreCorruption(
            "experimental_claim_event_invalid:" + ",".join(sorted(set(reasons)))
        )
    return ExperimentalClaimAdmissionRecord(
        event=event,
        graph=graph,
        route=route,
        projection=projection,
        discovery=discovery,
        validation_pack=pack,
        claim_set=claim_set,
    )


def _read_event(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentalClaimStoreCorruption(
            f"experimental_claim_event_unreadable:{type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise ExperimentalClaimStoreCorruption("experimental_claim_event_not_object")
    return dict(value)


def _event_envelope_reasons(
    event: Mapping[str, Any], *, path: Path, run_id: str
) -> list[str]:
    reasons: list[str] = []
    material = dict(event)
    content_sha256 = str(material.pop("content_sha256", ""))
    if set(event) != EVENT_KEYS:
        reasons.append("event_fields_invalid")
    if content_sha256 != strict_canonical_json_sha256(material):
        reasons.append("event_content_digest_invalid")
    if path.stem != content_sha256 or path.parent.name != content_sha256[:2]:
        reasons.append("event_path_binding_invalid")
    if event.get("schema_version") != EXPERIMENTAL_CLAIM_ADMISSION_EVENT_SCHEMA:
        reasons.append("event_schema_invalid")
    if event.get("run_id") != run_id:
        reasons.append("event_run_id_mismatch")
    if event.get("admission_policy") != EXPERIMENTAL_CLAIM_ADMISSION_POLICY:
        reasons.append("event_admission_policy_invalid")
    if event.get("semantics") != EVENT_SEMANTICS:
        reasons.append("event_authority_semantics_invalid")
    identity = event_identity(event)
    if event.get("event_identity_sha256") != identity or event.get("event_id") != (
        f"experimental-claim-admission:sha256:{identity}"
    ):
        reasons.append("event_identity_invalid")
    return reasons


def _read_payloads(
    event: Mapping[str, Any], artifacts: ArtifactStore
) -> tuple[dict[str, ArtifactRef], dict[str, dict[str, Any]]]:
    refs: dict[str, ArtifactRef] = {}
    payloads: dict[str, dict[str, Any]] = {}
    try:
        for key in _REF_NAMES:
            ref = ArtifactRef.from_dict(dict(event.get(f"{key}_ref") or {}))
            value = artifacts.read_json(ref)
            if not isinstance(value, dict):
                raise TypeError(f"{key}_payload_not_object")
            refs[key] = ref
            payloads[key] = dict(value)
    except (ArtifactStoreError, TypeError, ValueError) as exc:
        raise ExperimentalClaimStoreCorruption(
            f"experimental_claim_artifact_replay_failed:{type(exc).__name__}"
        ) from exc
    return refs, payloads


def _ref_reasons(
    event: Mapping[str, Any], refs: Mapping[str, ArtifactRef]
) -> list[str]:
    return [
        f"{key}_ref_contract_invalid"
        for key, ref in refs.items()
        if event.get(f"{key}_ref") != ref.to_dict()
    ]


def _source_binding_reasons(
    event: Mapping[str, Any],
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    pack: Mapping[str, Any],
    claim_set: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if (
        graph.get("run_id") != event.get("run_id")
        or graph.get("revision") != event.get("source_graph_revision")
        or graph.get("scientific_sha256")
        != event.get("source_graph_scientific_sha256")
    ):
        reasons.append("source_graph_binding_invalid")
    if route.get("route_id") != event.get("source_route_id") or (
        strict_canonical_json_sha256(route) != event.get("source_route_sha256")
    ):
        reasons.append("source_route_binding_invalid")
    if (
        projection.get("content_sha256") != event.get("source_projection_sha256")
        or program_projection_oracle(graph, projection).get("accepted") is not True
    ):
        reasons.append("source_projection_binding_invalid")
    if (
        discovery.get("content_sha256") != event.get("source_discovery_sha256")
        or discovery.get("route_id") != event.get("source_route_id")
    ):
        reasons.append("source_discovery_binding_invalid")
    pack_material = dict(pack)
    pack_digest = str(pack_material.pop("content_sha256", ""))
    if (
        set(pack) != VALIDATION_PACK_KEYS
        or pack.get("schema_version") != EXPERIMENTAL_VALIDATION_PACK_SCHEMA
        or pack.get("semantics") != VALIDATION_PACK_SEMANTICS
        or pack_digest != strict_canonical_json_sha256(pack_material)
        or pack_digest != event.get("validation_pack_sha256")
        or pack.get("run_id") != event.get("run_id")
        or pack.get("route_id") != event.get("source_route_id")
        or not isinstance(pack.get("validations"), list)
    ):
        reasons.append("validation_pack_binding_invalid")
    if (
        claim_set.get("content_sha256") != event.get("claim_set_sha256")
        or claim_set.get("run_id") != event.get("run_id")
        or claim_set.get("route_id") != event.get("source_route_id")
        or validate_experimental_claim_set(claim_set)
    ):
        reasons.append("claim_set_binding_invalid")
    return reasons


__all__ = [
    "ExperimentalClaimAdmissionRecord",
    "load_experimental_claim_admission_record",
]
