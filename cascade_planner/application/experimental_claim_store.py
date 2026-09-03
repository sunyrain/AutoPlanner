"""Append-only store for exact-boundary experimental observation Claims."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from cascade_planner.application.experimental_claim_store_contracts import (
    EXPERIMENTAL_CLAIM_ADMISSION_RESULT_SCHEMA,
    EXPERIMENTAL_CLAIM_STORE_ORACLE_SCHEMA,
    EXPERIMENTAL_CLAIM_STORE_REPLAY_SCHEMA,
    EXPERIMENTAL_CLAIM_STORE_STATUS_SCHEMA,
    ExperimentalClaimAdmissionDisabled,
    ExperimentalClaimStoreCorruption,
    ExperimentalClaimStoreError,
    admission_event,
    validation_pack,
    with_digest,
)
from cascade_planner.application.experimental_claim_store_replay import (
    ExperimentalClaimAdmissionRecord,
    load_experimental_claim_admission_record,
)
from cascade_planner.runtime.artifact_store import ArtifactRef, ArtifactStore
from cascade_planner.runtime.immutable_event_store import (
    load_replayable_event_records,
    publish_replayable_event,
)
from cascade_planner.runtime.run_index import RunIndex
from cascade_planner.application.experimental_claim_store_projection import (
    reproject_experimental_claim_materials,
)


_ARTIFACTS = {
    "graph": (
        "canonical_hypergraph.experimental_claim_admission.json",
        "experimental_claim_source_graph",
    ),
    "route": (
        "source_route.experimental_claim_admission.json",
        "experimental_claim_source_route",
    ),
    "projection": (
        "program_projection.experimental_claim_admission.json",
        "experimental_claim_source_projection",
    ),
    "discovery": (
        "route_innovation_discovery.experimental_claim_admission.json",
        "experimental_claim_source_discovery",
    ),
    "validation_pack": (
        "experimental_claim_validation_pack.json",
        "experimental_claim_source_validations",
    ),
    "claim_set": (
        "experimental_observation_claim_set.admission.json",
        "experimental_claim_exact_boundary_observations",
    ),
}


class ExperimentalClaimStore:
    """Persist observations without promoting reaction or route authority."""

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
            raise ValueError("experimental_claim_store_run_id_required")
        directory = Path(run_dir).expanduser().resolve()
        self.run_id = identity
        self.run_dir = directory
        self.root = (directory / ".autoplanner" / "experimental_claims").resolve()
        self.event_root = self.root / "events" / "sha256"
        try:
            self.root.relative_to(directory)
        except ValueError as exc:
            raise ValueError("experimental_claim_store_path_escape") from exc
        self.artifacts = artifacts
        self.index = index

    def admit(
        self,
        *,
        graph: Mapping[str, Any],
        route: Mapping[str, Any],
        projection: Mapping[str, Any],
        discovery: Mapping[str, Any],
        validations: Sequence[Mapping[str, Any]],
        enable_experimental_claim_admission: bool = False,
    ) -> dict[str, Any]:
        if enable_experimental_claim_admission is not True:
            raise ExperimentalClaimAdmissionDisabled(
                "experimental_claim_admission_disabled:explicit_enable_required"
            )
        rows = [dict(value) for value in validations]
        pack = validation_pack(
            run_id=self.run_id,
            route_id=str(route.get("route_id") or ""),
            validations=rows,
        )
        canonical_rows = list(pack["validations"])
        projected = reproject_experimental_claim_materials(
            graph, route, projection, discovery, canonical_rows
        )
        claim_set = projected["experimental_claims"]
        oracle = projected["experimental_claims_oracle"]
        if oracle.get("accepted") is not True:
            raise ExperimentalClaimStoreError("experimental_claim_set_oracle_failed")
        if (
            claim_set.get("run_id") != self.run_id
            or claim_set.get("route_id") != route.get("route_id")
        ):
            raise ExperimentalClaimStoreError("experimental_claim_store_identity_mismatch")
        if not claim_set.get("claims"):
            raise ExperimentalClaimStoreError(
                "experimental_claim_admission_requires_observation"
            )
        values = {
            "graph": dict(graph),
            "route": dict(route),
            "projection": dict(projection),
            "discovery": dict(discovery),
            "validation_pack": pack,
            "claim_set": claim_set,
        }
        refs = self._put_artifacts(values)
        event = admission_event(
            run_id=self.run_id,
            graph=graph,
            route=route,
            projection=projection,
            discovery=discovery,
            pack=pack,
            claim_set=claim_set,
            refs={key: ref.to_dict() for key, ref in refs.items()},
            oracle=oracle,
        )
        path, created = self._publish(event)
        self._pin_artifacts(event, refs)
        record = self._load(path)
        return with_digest(
            {
                "schema_version": EXPERIMENTAL_CLAIM_ADMISSION_RESULT_SCHEMA,
                "run_id": self.run_id,
                "admitted": True,
                "created": created,
                "event": record.event,
                "store": self.status(
                    graph=graph,
                    route=route,
                    projection=projection,
                    discovery=discovery,
                    validations=rows,
                ),
                "semantics": {
                    "explicit_enablement_observed": True,
                    "admission_is_idempotent": True,
                    "positive_negative_and_inconclusive_claims_are_equal_store_inputs": True,
                    "production_route_and_canonical_fact_authority_unchanged": True,
                },
            }
        )

    def replay(self) -> dict[str, Any]:
        records = self._records()
        return with_digest(
            {
                "schema_version": EXPERIMENTAL_CLAIM_STORE_REPLAY_SCHEMA,
                "run_id": self.run_id,
                "event_count": len(records),
                "events": [record.event for record in records],
                "claim_ids": sorted(
                    {
                        claim_id
                        for record in records
                        for claim_id in record.event["claim_ids"]
                    }
                ),
                "semantics": {
                    "all_events_and_cas_objects_replayed": True,
                    "all_claim_sets_reprojected_from_source_inputs": True,
                    "replay_grants_no_proof_completion_acceptance_or_catalog_authority": True,
                },
            }
        )

    def experience_sources(self) -> list[dict[str, Any]]:
        """Expose only fully replayed source triples for external memory learning."""

        return [
            {
                "graph": record.graph,
                "discovery": record.discovery,
                "claim_set": record.claim_set,
            }
            for record in self._records()
        ]

    def status(
        self,
        *,
        graph: Mapping[str, Any],
        route: Mapping[str, Any],
        projection: Mapping[str, Any],
        discovery: Mapping[str, Any],
        validations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        records = self._records()
        rows = [dict(value) for value in validations]
        pack = validation_pack(
            run_id=self.run_id,
            route_id=str(route.get("route_id") or ""),
            validations=rows,
        )
        projected = reproject_experimental_claim_materials(
            graph, route, projection, discovery, list(pack["validations"])
        )
        claim_set = projected["experimental_claims"]
        matching = [
            record
            for record in records
            if record.event.get("source_graph_revision") == graph.get("revision")
            and record.event.get("source_graph_scientific_sha256")
            == graph.get("scientific_sha256")
            and record.event.get("source_route_id") == route.get("route_id")
            and record.event.get("source_projection_sha256")
            == projection.get("content_sha256")
            and record.event.get("source_discovery_sha256")
            == discovery.get("content_sha256")
            and record.event.get("claim_set_sha256")
            == claim_set.get("content_sha256")
        ]
        current = max(
            matching,
            key=lambda row: str(row.event.get("content_sha256") or ""),
            default=None,
        )
        checks = {
            "event_replay_valid": True,
            "current_claim_set_event_present": current is not None,
            "current_claim_set_oracle_equal": (
                current is not None
                and projected["experimental_claims_oracle"].get("accepted") is True
                and current.claim_set == claim_set
            ),
        }
        oracle = with_digest(
            {
                "schema_version": EXPERIMENTAL_CLAIM_STORE_ORACLE_SCHEMA,
                "accepted": all(checks.values()),
                "checks": checks,
                "reasons": sorted(key for key, value in checks.items() if not value),
                "expected_claim_set_sha256": str(claim_set.get("content_sha256") or ""),
                "observed_claim_set_sha256": (
                    str(current.event.get("claim_set_sha256") or "") if current else ""
                ),
                "semantics": {
                    "dual_read_only": True,
                    "cannot_switch_route_or_fact_authority": True,
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
                "schema_version": EXPERIMENTAL_CLAIM_STORE_STATUS_SCHEMA,
                "run_id": self.run_id,
                "initialized": bool(records),
                "event_count": len(records),
                "current_claim_set_admitted": current is not None,
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
                producer="autoplanner.experimental_claim_store",
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
                artifact_id=f"experimental_claim_admission_{key}:{identity}",
                ref=refs[key],
                revision=revision,
                authority_scope=scope,
            )

    def _records(self) -> list[ExperimentalClaimAdmissionRecord]:
        return load_replayable_event_records(
            self.event_root,
            load=self._load,
            event_id=lambda record: str(record.event.get("event_id") or ""),
            corruption=ExperimentalClaimStoreCorruption,
            root_not_directory="experimental_claim_event_root_not_directory",
            duplicate_identity="experimental_claim_store_duplicate_event_identity",
        )

    def _publish(self, event: Mapping[str, Any]) -> tuple[Path, bool]:
        return publish_replayable_event(self.event_root, event, load=self._load)

    def _load(self, path: Path) -> ExperimentalClaimAdmissionRecord:
        return load_experimental_claim_admission_record(
            path, run_id=self.run_id, artifacts=self.artifacts
        )


def _event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(event.get("event_id") or ""),
        "content_sha256": str(event.get("content_sha256") or ""),
        "source_graph_revision": int(event.get("source_graph_revision") or 0),
        "source_route_id": str(event.get("source_route_id") or ""),
        "claim_set_sha256": str(event.get("claim_set_sha256") or ""),
        "claim_ids": list(event.get("claim_ids") or []),
        "counts": dict(event.get("counts") or {}),
    }


__all__ = [
    "ExperimentalClaimAdmissionDisabled",
    "ExperimentalClaimStore",
    "ExperimentalClaimStoreCorruption",
    "ExperimentalClaimStoreError",
]
