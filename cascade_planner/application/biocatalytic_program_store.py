"""Append-only shadow store for validated biocatalytic Program candidates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from cascade_planner.application.biocatalytic_program_store_contracts import (
    BIOCATALYTIC_PROGRAM_ADMISSION_RESULT_SCHEMA,
    BIOCATALYTIC_PROGRAM_STORE_ORACLE_SCHEMA,
    BIOCATALYTIC_PROGRAM_STORE_REPLAY_SCHEMA,
    BIOCATALYTIC_PROGRAM_STORE_STATUS_SCHEMA,
    BiocatalyticProgramAdmissionDisabled,
    BiocatalyticProgramStoreCorruption,
    BiocatalyticProgramStoreError,
    admission_event,
    admitted_entities,
    validation_pack,
    with_digest,
)
from cascade_planner.application.biocatalytic_program_store_replay import (
    BiocatalyticAdmissionRecord,
    load_biocatalytic_admission_record,
)
from cascade_planner.application.biocatalytic_programs import (
    biocatalytic_program_bundle_oracle,
)
from cascade_planner.runtime.artifact_store import ArtifactRef, ArtifactStore
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256
from cascade_planner.runtime.immutable_event_store import (
    load_replayable_event_records,
    publish_replayable_event,
)
from cascade_planner.runtime.run_index import RunIndex


_ARTIFACTS = {
    "graph": (
        "canonical_hypergraph.biocatalytic_program_admission.json",
        "shadow_biocatalytic_program_source_graph",
    ),
    "route": (
        "source_route.biocatalytic_program_admission.json",
        "shadow_biocatalytic_program_source_route",
    ),
    "projection": (
        "baseline_program_projection.biocatalytic_admission.json",
        "shadow_biocatalytic_program_baseline_projection",
    ),
    "discovery": (
        "route_innovation_discovery.biocatalytic_admission.json",
        "shadow_biocatalytic_program_discovery",
    ),
    "bundle": (
        "biocatalytic_program_bundle.admission.json",
        "shadow_biocatalytic_program_bundle",
    ),
    "validation_pack": (
        "biocatalysis_program_validation_pack.json",
        "shadow_biocatalytic_program_validations",
    ),
}


class BiocatalyticProgramStore:
    """Persist validated Program alternatives without changing route authority."""

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
            raise ValueError("biocatalytic_program_store_run_id_required")
        directory = Path(run_dir).expanduser().resolve()
        self.run_id = identity
        self.run_dir = directory
        self.root = (directory / ".autoplanner" / "bio_programs").resolve()
        self.event_root = self.root / "e"
        try:
            self.root.relative_to(directory)
        except ValueError as exc:
            raise ValueError("biocatalytic_program_store_path_escape") from exc
        self.artifacts = artifacts
        self.index = index

    def admit(
        self,
        *,
        graph: Mapping[str, Any],
        route: Mapping[str, Any],
        projection: Mapping[str, Any],
        discovery: Mapping[str, Any],
        bundle: Mapping[str, Any],
        validations: Sequence[Mapping[str, Any]],
        enable_biocatalytic_program_admission: bool = False,
    ) -> dict[str, Any]:
        if enable_biocatalytic_program_admission is not True:
            raise BiocatalyticProgramAdmissionDisabled(
                "biocatalytic_program_admission_disabled:explicit_enable_required"
            )
        rows = [dict(value) for value in validations]
        oracle = biocatalytic_program_bundle_oracle(
            graph,
            route,
            projection,
            discovery,
            bundle,
            validations=rows,
        )
        program_ids, route_ids = admitted_entities(bundle)
        if oracle.get("accepted") is not True:
            raise BiocatalyticProgramStoreError("biocatalytic_program_bundle_oracle_failed")
        if not program_ids or not route_ids:
            raise BiocatalyticProgramStoreError(
                "biocatalytic_program_admission_requires_validated_candidate"
            )
        pack = validation_pack(
            run_id=self.run_id,
            route_id=str(route.get("route_id") or ""),
            validations=rows,
        )
        values = {
            "graph": dict(graph),
            "route": dict(route),
            "projection": dict(projection),
            "discovery": dict(discovery),
            "bundle": dict(bundle),
            "validation_pack": pack,
        }
        refs = self._put_artifacts(values)
        event = admission_event(
            run_id=self.run_id,
            graph=graph,
            route=route,
            projection=projection,
            discovery=discovery,
            bundle=bundle,
            pack=pack,
            refs={key: value.to_dict() for key, value in refs.items()},
            oracle=oracle,
        )
        path, created = self._publish(event)
        self._pin_artifacts(event, refs)
        record = self._load(path)
        result = {
            "schema_version": BIOCATALYTIC_PROGRAM_ADMISSION_RESULT_SCHEMA,
            "run_id": self.run_id,
            "admitted": True,
            "created": created,
            "event": record.event,
            "store": self.status(
                graph=graph,
                route=route,
                projection=projection,
                discovery=discovery,
                bundle=bundle,
                validations=rows,
            ),
            "semantics": {
                "explicit_enablement_observed": True,
                "admission_is_idempotent": True,
                "baseline_fallback_retained": True,
                "production_route_authority_unchanged": True,
            },
        }
        return with_digest(result)

    def replay(self) -> dict[str, Any]:
        records = self._records()
        report = {
            "schema_version": BIOCATALYTIC_PROGRAM_STORE_REPLAY_SCHEMA,
            "run_id": self.run_id,
            "event_count": len(records),
            "events": [record.event for record in records],
            "admitted_program_ids": sorted(
                {
                    value
                    for record in records
                    for value in record.event["admitted_program_ids"]
                }
            ),
            "semantics": {
                "all_events_and_cas_objects_replayed": True,
                "replay_grants_no_production_authority": True,
            },
        }
        return with_digest(report)

    def status(
        self,
        *,
        graph: Mapping[str, Any],
        route: Mapping[str, Any],
        projection: Mapping[str, Any],
        discovery: Mapping[str, Any],
        bundle: Mapping[str, Any],
        validations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        records = self._records()
        pack = validation_pack(
            run_id=self.run_id,
            route_id=str(route.get("route_id") or ""),
            validations=validations,
        )
        route_sha256 = strict_canonical_json_sha256(route)
        matching = [
            record
            for record in records
            if record.event.get("source_graph_revision") == graph.get("revision")
            and record.event.get("source_graph_scientific_sha256")
            == graph.get("scientific_sha256")
            and record.event.get("source_route_sha256") == route_sha256
            and record.event.get("baseline_projection_sha256")
            == projection.get("content_sha256")
            and record.event.get("discovery_sha256") == discovery.get("content_sha256")
            and record.event.get("bundle_sha256") == bundle.get("content_sha256")
            and record.event.get("validation_pack_sha256") == pack.get("content_sha256")
        ]
        current = max(
            matching,
            key=lambda row: str(row.event.get("content_sha256") or ""),
            default=None,
        )
        current_oracle = biocatalytic_program_bundle_oracle(
            graph,
            route,
            projection,
            discovery,
            bundle,
            validations=validations,
        )
        checks = {
            "event_replay_valid": True,
            "current_bundle_event_present": current is not None,
            "current_bundle_oracle_equal": (
                current is not None
                and current_oracle.get("accepted") is True
                and current.bundle == bundle
            ),
        }
        oracle = with_digest(
            {
                "schema_version": BIOCATALYTIC_PROGRAM_STORE_ORACLE_SCHEMA,
                "accepted": all(checks.values()),
                "checks": checks,
                "reasons": sorted(key for key, value in checks.items() if not value),
                "expected_bundle_sha256": str(bundle.get("content_sha256") or ""),
                "observed_bundle_sha256": (
                    str(current.event.get("bundle_sha256") or "") if current else ""
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
        return with_digest(
            {
                "schema_version": BIOCATALYTIC_PROGRAM_STORE_STATUS_SCHEMA,
                "run_id": self.run_id,
                "initialized": bool(records),
                "event_count": len(records),
                "current_bundle_admitted": current is not None,
                "latest_event": _event_summary(latest.event) if latest else {},
                "oracle": oracle,
                "semantics": {
                    "query_is_read_only": True,
                    "store_is_append_only": True,
                    "admission_default_enabled": False,
                    "edge_ids_remain_production_route_authority": True,
                },
            }
        )

    def _put_artifacts(
        self, values: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, ArtifactRef]:
        return {
            key: self.artifacts.put_json(
                values[key],
                logical_name=logical_name,
                producer="autoplanner.biocatalytic_program_store",
            )
            for key, (logical_name, _scope) in _ARTIFACTS.items()
        }

    def _pin_artifacts(
        self, event: Mapping[str, Any], refs: Mapping[str, ArtifactRef]
    ) -> None:
        if self.index is None:
            return
        identity = str(event.get("event_identity_sha256") or "")[:24]
        revision = int(event.get("source_graph_revision") or 0)
        for key, (_logical_name, scope) in _ARTIFACTS.items():
            self.index.index_artifact(
                run_id=self.run_id,
                artifact_id=f"biocatalytic_program_admission_{key}:{identity}",
                ref=refs[key],
                revision=revision,
                authority_scope=scope,
            )

    def _records(self) -> list[BiocatalyticAdmissionRecord]:
        return load_replayable_event_records(
            self.event_root,
            load=self._load,
            event_id=lambda record: str(record.event.get("event_id") or ""),
            corruption=BiocatalyticProgramStoreCorruption,
            root_not_directory="biocatalytic_program_event_root_not_directory",
            duplicate_identity="biocatalytic_program_store_duplicate_event_identity",
        )

    def _publish(self, event: Mapping[str, Any]) -> tuple[Path, bool]:
        return publish_replayable_event(
            self.event_root,
            event,
            load=self._load,
        )

    def _load(self, path: Path) -> BiocatalyticAdmissionRecord:
        return load_biocatalytic_admission_record(
            path,
            run_id=self.run_id,
            artifacts=self.artifacts,
        )


def _event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(event.get("event_id") or ""),
        "content_sha256": str(event.get("content_sha256") or ""),
        "source_graph_revision": int(event.get("source_graph_revision") or 0),
        "source_route_id": str(event.get("source_route_id") or ""),
        "bundle_sha256": str(event.get("bundle_sha256") or ""),
        "admitted_program_ids": list(event.get("admitted_program_ids") or []),
        "counts": dict(event.get("counts") or {}),
    }


__all__ = [
    "BiocatalyticProgramAdmissionDisabled",
    "BiocatalyticProgramStore",
    "BiocatalyticProgramStoreCorruption",
    "BiocatalyticProgramStoreError",
]
