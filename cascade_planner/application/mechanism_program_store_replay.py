"""Replay validation for immutable mechanism Program admission events."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.application.mechanism_program_store_contracts import (
    EVENT_KEYS,
    EVENT_SEMANTICS,
    MECHANISM_PROGRAM_ADMISSION_EVENT_SCHEMA,
    MECHANISM_PROGRAM_ADMISSION_POLICY,
    MECHANISM_VALIDATION_PACK_SCHEMA,
    VALIDATION_PACK_KEYS,
    VALIDATION_PACK_SEMANTICS,
    MechanismProgramStoreCorruption,
    admission_counts,
    admitted_entities,
    event_identity,
)
from cascade_planner.application.mechanism_programs import mechanism_program_bundle_oracle
from cascade_planner.application.transformation_program_validation import (
    validate_program_projection,
)
from cascade_planner.application.transformation_programs import program_projection_oracle
from cascade_planner.runtime.artifact_store import ArtifactRef, ArtifactStore, ArtifactStoreError
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


_REF_NAMES = (
    "source_graph",
    "source_route",
    "baseline_projection",
    "discovery",
    "bundle",
    "validation_pack",
)


@dataclass(frozen=True, slots=True)
class MechanismAdmissionRecord:
    event: dict[str, Any]
    graph: dict[str, Any]
    route: dict[str, Any]
    projection: dict[str, Any]
    discovery: dict[str, Any]
    bundle: dict[str, Any]
    validation_pack: dict[str, Any]


def load_mechanism_admission_record(
    path: Path, *, run_id: str, artifacts: ArtifactStore
) -> MechanismAdmissionRecord:
    event = _read_event(path)
    reasons = _event_envelope_reasons(event, path=path, run_id=run_id)
    refs, payloads = _read_payloads(event, artifacts)
    graph, route, projection, discovery, bundle, pack = (
        payloads[key] for key in _REF_NAMES
    )
    try:
        reasons.extend(
            _binding_reasons(event, graph, route, projection, discovery, bundle, pack)
        )
        oracle = mechanism_program_bundle_oracle(
            graph,
            route,
            projection,
            discovery,
            bundle,
            validations=list(pack.get("validations") or []),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MechanismProgramStoreCorruption(
            f"mechanism_program_artifact_binding_failed:{type(exc).__name__}"
        ) from exc
    reasons.extend(_ref_reasons(event, refs))
    if oracle.get("accepted") is not True or event.get("bundle_oracle") != oracle:
        reasons.append("bundle_oracle_replay_mismatch")
    program_ids, route_ids = admitted_entities(bundle)
    if not program_ids or not route_ids:
        reasons.append("admission_contains_no_validated_program")
    if event.get("admitted_program_ids") != program_ids:
        reasons.append("admitted_program_ids_invalid")
    if event.get("admitted_route_candidate_ids") != route_ids:
        reasons.append("admitted_route_candidate_ids_invalid")
    if event.get("counts") != admission_counts(bundle, program_ids, route_ids):
        reasons.append("admission_counts_invalid")
    if reasons:
        raise MechanismProgramStoreCorruption(
            "mechanism_program_event_invalid:" + ",".join(sorted(set(reasons)))
        )
    return MechanismAdmissionRecord(
        event=event,
        graph=graph,
        route=route,
        projection=projection,
        discovery=discovery,
        bundle=bundle,
        validation_pack=pack,
    )


def _read_event(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MechanismProgramStoreCorruption(
            f"mechanism_program_event_unreadable:{type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise MechanismProgramStoreCorruption("mechanism_program_event_not_object")
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
    if event.get("schema_version") != MECHANISM_PROGRAM_ADMISSION_EVENT_SCHEMA:
        reasons.append("event_schema_invalid")
    if event.get("run_id") != run_id:
        reasons.append("event_run_id_mismatch")
    if event.get("admission_policy") != MECHANISM_PROGRAM_ADMISSION_POLICY:
        reasons.append("event_admission_policy_invalid")
    if event.get("semantics") != EVENT_SEMANTICS:
        reasons.append("event_authority_semantics_invalid")
    identity = event_identity(event)
    if event.get("event_identity_sha256") != identity or event.get("event_id") != (
        f"mechanism-program-admission:sha256:{identity}"
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
        raise MechanismProgramStoreCorruption(
            f"mechanism_program_artifact_replay_failed:{type(exc).__name__}"
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


def _binding_reasons(
    event: Mapping[str, Any],
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if (
        graph.get("run_id") != event.get("run_id")
        or graph.get("revision") != event.get("source_graph_revision")
        or graph.get("scientific_sha256") != event.get("source_graph_scientific_sha256")
    ):
        reasons.append("source_graph_binding_invalid")
    if route.get("route_id") != event.get("source_route_id") or (
        strict_canonical_json_sha256(route) != event.get("source_route_sha256")
    ):
        reasons.append("source_route_binding_invalid")
    if validate_program_projection(
        projection, expected_run_id=str(event.get("run_id") or "")
    ).get("accepted") is not True:
        reasons.append("baseline_projection_contract_invalid")
    if program_projection_oracle(graph, projection).get("accepted") is not True:
        reasons.append("baseline_projection_oracle_invalid")
    for label, value, event_key in (
        ("baseline_projection", projection, "baseline_projection_sha256"),
        ("discovery", discovery, "discovery_sha256"),
        ("bundle", bundle, "bundle_sha256"),
        ("validation_pack", pack, "validation_pack_sha256"),
    ):
        if value.get("content_sha256") != event.get(event_key):
            reasons.append(f"{label}_binding_invalid")
    material = dict(pack)
    observed = str(material.pop("content_sha256", ""))
    if (
        set(pack) != VALIDATION_PACK_KEYS
        or pack.get("semantics") != VALIDATION_PACK_SEMANTICS
        or pack.get("schema_version") != MECHANISM_VALIDATION_PACK_SCHEMA
        or observed != strict_canonical_json_sha256(material)
        or pack.get("run_id") != event.get("run_id")
        or pack.get("route_id") != event.get("source_route_id")
        or not isinstance(pack.get("validations"), list)
    ):
        reasons.append("validation_pack_contract_invalid")
    return reasons


__all__ = ["MechanismAdmissionRecord", "load_mechanism_admission_record"]
