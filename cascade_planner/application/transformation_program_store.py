"""Append-only, replayable admission store for shadow TransformationPrograms."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.application.transformation_program_validation import (
    validate_program_projection,
)
from cascade_planner.application.transformation_programs import (
    PROGRAM_PROJECTION_SCHEMA,
    program_projection_oracle,
    project_canonical_graph_to_programs,
)
from cascade_planner.runtime.artifact_store import (
    ArtifactRef,
    ArtifactStore,
    ArtifactStoreError,
)
from cascade_planner.runtime.canonical_json import (
    strict_canonical_json_sha256,
)
from cascade_planner.runtime.immutable_event_store import (
    load_replayable_event_records,
    publish_replayable_event,
)
from cascade_planner.runtime.run_index import RunIndex


PROGRAM_ADMISSION_POLICY = "canonical_edge_program_projection_admission.v1"
PROGRAM_ADMISSION_EVENT_SCHEMA = "transformation_program_admission_event.v1"
PROGRAM_ADMISSION_RESULT_SCHEMA = "transformation_program_admission_result.v1"
PROGRAM_STORE_REPLAY_SCHEMA = "transformation_program_store_replay.v1"
PROGRAM_STORE_STATUS_SCHEMA = "transformation_program_store_status.v1"
PROGRAM_STORE_ORACLE_SCHEMA = "transformation_program_store_oracle.v1"

_EVENT_KEYS = {
    "schema_version",
    "event_id",
    "event_identity_sha256",
    "run_id",
    "source_graph_revision",
    "source_graph_scientific_sha256",
    "source_graph_ref",
    "projection_sha256",
    "projection_ref",
    "entity_ids",
    "counts",
    "projection_validation",
    "admission_oracle",
    "admission_policy",
    "semantics",
    "content_sha256",
}
_EVENT_SEMANTICS = {
    "append_only_content_addressed_event": True,
    "shadow_program_admission_only": True,
    "cannot_mutate_canonical_graph": True,
    "cannot_grant_reaction_proof": True,
    "cannot_grant_route_completion": True,
    "edge_ids_remain_production_route_authority": True,
}


class TransformationProgramStoreError(RuntimeError):
    """Base error for Program admission or replay."""


class TransformationProgramAdmissionDisabled(TransformationProgramStoreError):
    """A caller attempted a Program write without the explicit gate."""


class TransformationProgramStoreCorruption(TransformationProgramStoreError):
    """An immutable admission event or referenced object failed replay."""


@dataclass(frozen=True, slots=True)
class _AdmissionRecord:
    event: dict[str, Any]
    graph: dict[str, Any]
    projection: dict[str, Any]


class TransformationProgramStore:
    """Persist shadow Program projections without changing production routes."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: str | os.PathLike[str],
        artifacts: ArtifactStore,
        index: RunIndex | None = None,
    ) -> None:
        identity = str(run_id or "").strip()
        if not identity:
            raise ValueError("program_store_run_id_required")
        directory = Path(run_dir).expanduser().resolve()
        self.run_id = identity
        self.run_dir = directory
        self.root = (directory / ".autoplanner" / "program_store").resolve()
        self.event_root = self.root / "events" / "sha256"
        try:
            self.root.relative_to(directory)
        except ValueError as exc:
            raise ValueError("program_store_path_escape") from exc
        self.artifacts = artifacts
        self.index = index

    def admit(
        self,
        graph: Mapping[str, Any],
        *,
        enable_program_admission: bool = False,
    ) -> dict[str, Any]:
        """Admit the exact current projection only after explicit enablement."""

        if enable_program_admission is not True:
            raise TransformationProgramAdmissionDisabled(
                "program_admission_disabled:explicit_enable_required"
            )
        projection = project_canonical_graph_to_programs(graph)
        validation = validate_program_projection(projection, expected_run_id=self.run_id)
        oracle = program_projection_oracle(graph, projection)
        if validation.get("accepted") is not True or oracle.get("accepted") is not True:
            raise TransformationProgramStoreError("program_projection_failed_host_admission")
        graph_ref = self.artifacts.put_json(
            graph,
            logical_name="canonical_hypergraph.program_admission.json",
            producer="autoplanner.transformation_program_store",
        )
        projection_ref = self.artifacts.put_json(
            projection,
            logical_name="transformation_program_projection.json",
            producer="autoplanner.transformation_program_store",
        )
        event = _admission_event(
            run_id=self.run_id,
            graph=graph,
            projection=projection,
            graph_ref=graph_ref,
            projection_ref=projection_ref,
            validation=validation,
            oracle=oracle,
        )
        self._pin_artifacts(event, graph_ref=graph_ref, projection_ref=projection_ref)
        path, created = self._publish(event)
        record = self._load(path)
        status = self.status(graph)
        result = {
            "schema_version": PROGRAM_ADMISSION_RESULT_SCHEMA,
            "run_id": self.run_id,
            "admitted": True,
            "created": created,
            "event": record.event,
            "store": status,
            "semantics": {
                "explicit_enablement_observed": True,
                "admission_is_idempotent": True,
                "production_route_authority_unchanged": True,
            },
        }
        return _with_digest(result)

    def replay(self) -> dict[str, Any]:
        """Replay every immutable event and both referenced CAS objects."""

        records = self._records()
        report = {
            "schema_version": PROGRAM_STORE_REPLAY_SCHEMA,
            "run_id": self.run_id,
            "event_count": len(records),
            "events": [record.event for record in records],
            "projection_sha256s": sorted(
                str(record.event["projection_sha256"]) for record in records
            ),
            "semantics": {
                "all_events_replayed": True,
                "source_graph_and_projection_objects_verified": True,
                "replay_grants_no_production_authority": True,
            },
        }
        return _with_digest(report)

    def status(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        """Compare the durable store with a fresh read of the canonical graph."""

        records = self._records()
        projection = project_canonical_graph_to_programs(graph)
        expected_sha256 = str(projection["content_sha256"])
        graph_revision = int(graph.get("revision") or 0)
        graph_sha256 = str(graph.get("scientific_sha256") or "")
        matching = [
            record
            for record in records
            if record.event.get("projection_sha256") == expected_sha256
            and record.event.get("source_graph_revision") == graph_revision
            and record.event.get("source_graph_scientific_sha256") == graph_sha256
        ]
        current = max(
            matching,
            key=lambda row: str(row.event.get("content_sha256") or ""),
            default=None,
        )
        checks = {
            "event_replay_valid": True,
            "current_projection_event_present": current is not None,
            "current_projection_oracle_equal": (
                current is not None
                and program_projection_oracle(graph, current.projection).get("accepted") is True
            ),
        }
        oracle = _with_digest(
            {
                "schema_version": PROGRAM_STORE_ORACLE_SCHEMA,
                "accepted": all(checks.values()),
                "checks": checks,
                "reasons": sorted(key for key, accepted in checks.items() if not accepted),
                "expected_projection_sha256": expected_sha256,
                "observed_projection_sha256": (
                    str(current.event["projection_sha256"]) if current else ""
                ),
                "semantics": {
                    "dual_read_only": True,
                    "cannot_switch_route_authority": True,
                },
            }
        )
        latest = max(
            records,
            key=lambda row: (
                int(row.event.get("source_graph_revision") or 0),
                str(row.event.get("content_sha256") or ""),
            ),
            default=None,
        )
        status = {
            "schema_version": PROGRAM_STORE_STATUS_SCHEMA,
            "run_id": self.run_id,
            "initialized": bool(records),
            "event_count": len(records),
            "current_projection_admitted": current is not None,
            "current_graph_revision": graph_revision,
            "current_graph_scientific_sha256": graph_sha256,
            "current_projection_sha256": expected_sha256,
            "latest_event": _event_summary(latest.event) if latest else {},
            "oracle": oracle,
            "semantics": {
                "query_is_read_only": True,
                "store_is_append_only": True,
                "admission_default_enabled": False,
                "edge_ids_remain_production_route_authority": True,
            },
        }
        return _with_digest(status)

    def _pin_artifacts(
        self,
        event: Mapping[str, Any],
        *,
        graph_ref: ArtifactRef,
        projection_ref: ArtifactRef,
    ) -> None:
        if self.index is None:
            return
        identity = str(event.get("event_identity_sha256") or "")[:24]
        revision = int(event.get("source_graph_revision") or 0)
        for artifact_id, ref, scope in (
            (
                f"program_admission_source_graph:{identity}",
                graph_ref,
                "shadow_program_admission_source_graph",
            ),
            (
                f"program_admission_projection:{identity}",
                projection_ref,
                "shadow_program_admission_projection",
            ),
        ):
            self.index.index_artifact(
                run_id=self.run_id,
                artifact_id=artifact_id,
                ref=ref,
                revision=revision,
                authority_scope=scope,
            )

    def _records(self) -> list[_AdmissionRecord]:
        return load_replayable_event_records(
            self.event_root,
            load=self._load,
            event_id=lambda record: str(record.event.get("event_id") or ""),
            corruption=TransformationProgramStoreCorruption,
            root_not_directory="program_store_event_root_not_directory",
            duplicate_identity="program_store_duplicate_event_identity",
        )

    def _publish(self, event: Mapping[str, Any]) -> tuple[Path, bool]:
        return publish_replayable_event(
            self.event_root,
            event,
            load=self._load,
        )

    def _load(self, path: Path) -> _AdmissionRecord:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransformationProgramStoreCorruption(
                f"program_store_event_unreadable:{type(exc).__name__}"
            ) from exc
        if not isinstance(raw, dict):
            raise TransformationProgramStoreCorruption("program_store_event_not_object")
        event = dict(raw)
        reasons: list[str] = []
        if set(event) != _EVENT_KEYS:
            reasons.append("event_fields_invalid")
        material = dict(event)
        content_sha256 = str(material.pop("content_sha256", ""))
        if content_sha256 != strict_canonical_json_sha256(material):
            reasons.append("event_content_digest_invalid")
        if path.stem != content_sha256 or path.parent.name != content_sha256[:2]:
            reasons.append("event_path_binding_invalid")
        if event.get("schema_version") != PROGRAM_ADMISSION_EVENT_SCHEMA:
            reasons.append("event_schema_invalid")
        if event.get("run_id") != self.run_id:
            reasons.append("event_run_id_mismatch")
        if event.get("admission_policy") != PROGRAM_ADMISSION_POLICY:
            reasons.append("event_admission_policy_invalid")
        if event.get("semantics") != _EVENT_SEMANTICS:
            reasons.append("event_authority_semantics_invalid")
        expected_identity = _event_identity(event)
        if (
            event.get("event_identity_sha256") != expected_identity
            or event.get("event_id") != f"program-admission:sha256:{expected_identity}"
        ):
            reasons.append("event_identity_invalid")
        try:
            graph_ref = ArtifactRef.from_dict(dict(event.get("source_graph_ref") or {}))
            projection_ref = ArtifactRef.from_dict(dict(event.get("projection_ref") or {}))
            if event.get("source_graph_ref") != graph_ref.to_dict():
                reasons.append("source_graph_ref_contract_invalid")
            if event.get("projection_ref") != projection_ref.to_dict():
                reasons.append("projection_ref_contract_invalid")
            graph = self.artifacts.read_json(graph_ref)
            projection = self.artifacts.read_json(projection_ref)
        except (ArtifactStoreError, TypeError, ValueError) as exc:
            raise TransformationProgramStoreCorruption(
                f"program_store_artifact_replay_failed:{type(exc).__name__}"
            ) from exc
        if not isinstance(graph, dict) or not isinstance(projection, dict):
            reasons.append("event_artifact_payload_not_object")
            graph = dict(graph) if isinstance(graph, Mapping) else {}
            projection = dict(projection) if isinstance(projection, Mapping) else {}
        if (
            graph.get("run_id") != self.run_id
            or graph.get("revision") != event.get("source_graph_revision")
            or graph.get("scientific_sha256") != event.get("source_graph_scientific_sha256")
        ):
            reasons.append("source_graph_binding_invalid")
        validation = validate_program_projection(projection, expected_run_id=self.run_id)
        if validation.get("accepted") is not True:
            reasons.append("projection_contract_replay_failed")
        if event.get("projection_validation") != validation:
            reasons.append("projection_validation_replay_mismatch")
        if projection.get("schema_version") != PROGRAM_PROJECTION_SCHEMA or projection.get(
            "content_sha256"
        ) != event.get("projection_sha256"):
            reasons.append("projection_binding_invalid")
        try:
            oracle = program_projection_oracle(graph, projection)
        except (TypeError, ValueError) as exc:
            raise TransformationProgramStoreCorruption(
                f"program_store_oracle_replay_failed:{type(exc).__name__}"
            ) from exc
        if oracle.get("accepted") is not True or event.get("admission_oracle") != oracle:
            reasons.append("admission_oracle_replay_mismatch")
        entity_ids = event.get("entity_ids")
        expected_ids = _entity_ids(projection)
        if entity_ids != expected_ids:
            reasons.append("event_entity_ids_invalid")
        if event.get("counts") != projection.get("counts"):
            reasons.append("event_counts_invalid")
        if reasons:
            raise TransformationProgramStoreCorruption(
                "program_store_event_invalid:" + ",".join(sorted(set(reasons)))
            )
        return _AdmissionRecord(event=event, graph=graph, projection=projection)


def _admission_event(
    *,
    run_id: str,
    graph: Mapping[str, Any],
    projection: Mapping[str, Any],
    graph_ref: ArtifactRef,
    projection_ref: ArtifactRef,
    validation: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": PROGRAM_ADMISSION_EVENT_SCHEMA,
        "run_id": run_id,
        "source_graph_revision": int(graph.get("revision") or 0),
        "source_graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "source_graph_ref": graph_ref.to_dict(),
        "projection_sha256": str(projection.get("content_sha256") or ""),
        "projection_ref": projection_ref.to_dict(),
        "entity_ids": _entity_ids(projection),
        "counts": dict(projection.get("counts") or {}),
        "projection_validation": dict(validation),
        "admission_oracle": dict(oracle),
        "admission_policy": PROGRAM_ADMISSION_POLICY,
        "semantics": dict(_EVENT_SEMANTICS),
    }
    identity = _event_identity(event)
    event["event_identity_sha256"] = identity
    event["event_id"] = f"program-admission:sha256:{identity}"
    return _with_digest(event)


def _event_identity(event: Mapping[str, Any]) -> str:
    return strict_canonical_json_sha256(
        {
            "schema_version": PROGRAM_ADMISSION_EVENT_SCHEMA,
            "run_id": event.get("run_id"),
            "source_graph_revision": event.get("source_graph_revision"),
            "source_graph_scientific_sha256": event.get("source_graph_scientific_sha256"),
            "source_graph_artifact_sha256": dict(event.get("source_graph_ref") or {}).get("sha256"),
            "projection_sha256": event.get("projection_sha256"),
            "projection_artifact_sha256": dict(event.get("projection_ref") or {}).get("sha256"),
            "admission_policy": PROGRAM_ADMISSION_POLICY,
        }
    )


def _entity_ids(projection: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        "chemical_state_ids": sorted(dict(projection.get("chemical_states") or {})),
        "operation_node_ids": sorted(dict(projection.get("operation_nodes") or {})),
        "program_ids": sorted(dict(projection.get("programs") or {})),
        "route_ids": sorted(dict(projection.get("routes") or {})),
    }


def _event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(event.get("event_id") or ""),
        "content_sha256": str(event.get("content_sha256") or ""),
        "source_graph_revision": int(event.get("source_graph_revision") or 0),
        "source_graph_scientific_sha256": str(event.get("source_graph_scientific_sha256") or ""),
        "projection_sha256": str(event.get("projection_sha256") or ""),
        "counts": dict(event.get("counts") or {}),
    }


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "PROGRAM_ADMISSION_EVENT_SCHEMA",
    "PROGRAM_ADMISSION_POLICY",
    "PROGRAM_ADMISSION_RESULT_SCHEMA",
    "PROGRAM_STORE_ORACLE_SCHEMA",
    "PROGRAM_STORE_REPLAY_SCHEMA",
    "PROGRAM_STORE_STATUS_SCHEMA",
    "TransformationProgramAdmissionDisabled",
    "TransformationProgramStore",
    "TransformationProgramStoreCorruption",
    "TransformationProgramStoreError",
]
