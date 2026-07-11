"""Immutable, content-addressed closeout revisions.

Compatibility artifacts keep their historical filenames so existing readers
continue to work.  A committed revision additionally snapshots those bytes in
an immutable CAS, pins every dependency by artifact id and SHA-256, and only
then atomically replaces the small ``latest.json`` pointer.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from cascade_planner.harness.schemas import (
    ArtifactDigestDependency,
    CLOSEOUT_LATEST_POINTER_SCHEMA,
    CLOSEOUT_REVISION_MANIFEST_SCHEMA,
    CONTENT_ADDRESSED_ARTIFACT_SCHEMA,
    CloseoutLatestPointer,
    CloseoutRevisionManifest,
    ContentAddressedArtifact,
)


CLOSEOUT_VALIDATION_SCHEMA = "closeout_revision_validation.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "route_consensus_graph": ("route_consensus",),
    "canonical_route_consensus_graph": (
        "codex_campaign_proof_reconciliation",
    ),
    "final_verdict_core": ("parent_route_proof_snapshot",),
    "explored_route_forest": (
        "route_consensus",
        "route_consensus_graph",
        "parent_route_proof_snapshot",
        "final_verdict_core",
    ),
    "route_forest_html": ("explored_route_forest",),
}


class ArtifactRevisionError(RuntimeError):
    """Raised when a closeout revision cannot be safely committed."""


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_closeout_revision(
    run_dir: str | Path,
    *,
    artifacts: Mapping[str, str | Path],
    dependencies: Mapping[str, Sequence[str]] | None = None,
    producer: str,
    case_id: str = "",
    expected_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Stage, validate, and atomically activate one closeout revision.

    ``expected_digests`` binds publication to bytes captured by the producer.
    If a compatibility file changes between rendering and closeout, no latest
    pointer is switched.  Re-publishing identical bytes is idempotent.
    """
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dependency_map = {
        str(key): tuple(str(item) for item in value)
        for key, value in (dependencies or {}).items()
    }
    expected = {str(key): str(value).lower() for key, value in (expected_digests or {}).items()}
    if not artifacts:
        raise ArtifactRevisionError("closeout_revision_requires_artifacts")

    source_rows: dict[str, dict[str, Any]] = {}
    for raw_artifact_id, raw_path in artifacts.items():
        artifact_id = str(raw_artifact_id).strip()
        if not artifact_id:
            raise ArtifactRevisionError("closeout_artifact_id_missing")
        if artifact_id in source_rows:
            raise ArtifactRevisionError(f"duplicate_closeout_artifact_id:{artifact_id}")
        source_path = _resolve_under_root(root, raw_path)
        if not source_path.is_file():
            raise ArtifactRevisionError(f"closeout_artifact_missing:{artifact_id}:{source_path}")
        digest = sha256_file(source_path)
        expected_digest = expected.get(artifact_id, "")
        if expected_digest and expected_digest != digest:
            raise ArtifactRevisionError(
                f"closeout_artifact_changed_before_commit:{artifact_id}:"
                f"expected={expected_digest}:actual={digest}"
            )
        source_rows[artifact_id] = {
            "path": source_path,
            "relative_path": source_path.relative_to(root).as_posix(),
            "sha256": digest,
            "size_bytes": source_path.stat().st_size,
            "artifact_schema_version": _artifact_schema_version(source_path),
        }

    unknown_dependencies = sorted(
        {
            dependency_id
            for artifact_id, dependency_ids in dependency_map.items()
            for dependency_id in dependency_ids
            if artifact_id in source_rows and dependency_id not in source_rows
        }
    )
    if unknown_dependencies:
        raise ArtifactRevisionError(
            "closeout_dependency_missing:" + ",".join(unknown_dependencies)
        )
    unknown_dependency_owners = sorted(set(dependency_map).difference(source_rows))
    if unknown_dependency_owners:
        raise ArtifactRevisionError(
            "closeout_dependency_owner_missing:" + ",".join(unknown_dependency_owners)
        )

    records: list[ContentAddressedArtifact] = []
    for artifact_id in sorted(source_rows):
        row = source_rows[artifact_id]
        source_path = Path(row["path"])
        digest = str(row["sha256"])
        content_path = _content_path(root, artifact_id, source_path.suffix, digest)
        _write_immutable_bytes(content_path, source_path.read_bytes())
        dependency_records = tuple(
            ArtifactDigestDependency(
                artifact_id=dependency_id,
                sha256=str(source_rows[dependency_id]["sha256"]),
            )
            for dependency_id in sorted(set(dependency_map.get(artifact_id, ())))
        )
        records.append(
            ContentAddressedArtifact(
                artifact_id=artifact_id,
                path=str(row["relative_path"]),
                content_path=content_path.relative_to(root).as_posix(),
                sha256=digest,
                size_bytes=int(row["size_bytes"]),
                producer=str(producer),
                artifact_schema_version=str(row["artifact_schema_version"]),
                dependencies=dependency_records,
            )
        )

    revision_id = _revision_id(case_id=str(case_id), producer=str(producer), artifacts=records)
    closeout_root = root / ".autoplanner" / "closeout"
    staging_path = closeout_root / "staging" / f"{_revision_digest(revision_id)}.json"
    revision_path = closeout_root / "revisions" / f"{_revision_digest(revision_id)}.json"
    latest_path = closeout_root / "latest.json"

    existing_manifest = _load_json_object(revision_path) if revision_path.is_file() else {}
    created_at = str(existing_manifest.get("created_at") or _utc_now())
    staging_manifest = CloseoutRevisionManifest(
        revision_id=revision_id,
        case_id=str(case_id),
        producer=str(producer),
        artifacts=tuple(records),
        status="staging",
        created_at=created_at,
    )
    _atomic_write_json(staging_path, staging_manifest.to_dict())
    staging_validation = validate_closeout_manifest(root, staging_path)
    if staging_validation.get("accepted") is not True:
        raise ArtifactRevisionError(
            "closeout_staging_validation_failed:"
            + ",".join(str(item) for item in staging_validation.get("reasons") or [])
        )

    committed_manifest = replace(staging_manifest, status="committed")
    _write_immutable_json(revision_path, committed_manifest.to_dict())
    committed_validation = validate_closeout_manifest(root, revision_path)
    if committed_validation.get("accepted") is not True:
        raise ArtifactRevisionError(
            "closeout_committed_validation_failed:"
            + ",".join(str(item) for item in committed_validation.get("reasons") or [])
        )

    pointer = CloseoutLatestPointer(
        revision_id=revision_id,
        manifest_path=revision_path.relative_to(root).as_posix(),
        manifest_sha256=sha256_file(revision_path),
        activated_at=_utc_now(),
    )
    _atomic_write_json(latest_path, pointer.to_dict())
    latest_validation = validate_latest_closeout_revision(root)
    if latest_validation.get("accepted") is not True:
        raise ArtifactRevisionError(
            "closeout_latest_validation_failed:"
            + ",".join(str(item) for item in latest_validation.get("reasons") or [])
        )
    return {
        "schema_version": "closeout_revision_publish_result.v1",
        "accepted": True,
        "revision_id": revision_id,
        "manifest": committed_manifest.to_dict(),
        "manifest_path": str(revision_path),
        "manifest_sha256": pointer.manifest_sha256,
        "latest_pointer": pointer.to_dict(),
        "latest_pointer_path": str(latest_path),
        "staging_manifest_path": str(staging_path),
        "validation": latest_validation,
    }


def validate_closeout_manifest(
    run_dir: str | Path,
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    require_compatibility_paths: bool = True,
) -> dict[str, Any]:
    """Validate schema, revision identity, content and dependency hashes."""
    root = Path(run_dir).expanduser().resolve()
    reasons: list[str] = []
    manifest_path = ""
    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
    else:
        try:
            path = _resolve_under_root(root, manifest_or_path)
        except ArtifactRevisionError as exc:
            return _validation_report(False, [str(exc)], "", "", 0)
        manifest_path = str(path)
        try:
            manifest = _load_json_object(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ArtifactRevisionError) as exc:
            return _validation_report(
                False,
                [f"closeout_manifest_invalid_json:{type(exc).__name__}:{exc}"],
                "",
                manifest_path,
                0,
            )

    if manifest.get("schema_version") != CLOSEOUT_REVISION_MANIFEST_SCHEMA:
        reasons.append("closeout_manifest_schema_invalid")
    if manifest.get("status") not in {"staging", "committed"}:
        reasons.append("closeout_manifest_status_invalid")
    manifest_producer = str(manifest.get("producer") or "")
    if not manifest_producer:
        reasons.append("closeout_manifest_producer_missing")
    revision_id = str(manifest.get("revision_id") or "")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        reasons.append("closeout_manifest_artifacts_missing")
        rows = []

    index: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    reconstructed: list[ContentAddressedArtifact] = []
    for index_number, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            reasons.append(f"closeout_artifact_record_invalid:{index_number}")
            continue
        row = dict(raw_row)
        artifact_id = str(row.get("artifact_id") or "")
        if not artifact_id:
            reasons.append(f"closeout_artifact_id_missing:{index_number}")
            continue
        if artifact_id in index:
            reasons.append(f"closeout_artifact_id_duplicate:{artifact_id}")
            continue
        index[artifact_id] = row
        digest = str(row.get("sha256") or "").lower()
        if not _SHA256_RE.fullmatch(digest):
            reasons.append(f"closeout_artifact_sha256_invalid:{artifact_id}")
        if not str(row.get("artifact_schema_version") or ""):
            reasons.append(f"closeout_artifact_payload_schema_missing:{artifact_id}")
        row_producer = str(row.get("producer") or "")
        if not row_producer:
            reasons.append(f"closeout_artifact_producer_missing:{artifact_id}")
        elif row_producer != manifest_producer:
            reasons.append(f"closeout_artifact_producer_mismatch:{artifact_id}")
        if _safe_int(row.get("size_bytes"), default=-1) < 0:
            reasons.append(f"closeout_artifact_size_invalid:{artifact_id}")
        dependencies: list[ArtifactDigestDependency] = []
        seen_dependency_ids: set[str] = set()
        dependency_rows = row.get("dependencies")
        if not isinstance(dependency_rows, list):
            reasons.append(f"closeout_artifact_dependencies_invalid:{artifact_id}")
            dependency_rows = []
        for dependency_number, raw_dependency in enumerate(dependency_rows):
            if not isinstance(raw_dependency, dict):
                reasons.append(
                    f"closeout_dependency_record_invalid:{artifact_id}:{dependency_number}"
                )
                continue
            dependency_id = str(raw_dependency.get("artifact_id") or "")
            dependency_sha = str(raw_dependency.get("sha256") or "").lower()
            if not dependency_id or not _SHA256_RE.fullmatch(dependency_sha):
                reasons.append(
                    f"closeout_dependency_identity_invalid:{artifact_id}:{dependency_number}"
                )
                continue
            if dependency_id == artifact_id:
                reasons.append(f"closeout_dependency_self_reference:{artifact_id}")
            if dependency_id in seen_dependency_ids:
                reasons.append(f"closeout_dependency_duplicate:{artifact_id}:{dependency_id}")
            seen_dependency_ids.add(dependency_id)
            dependencies.append(
                ArtifactDigestDependency(artifact_id=dependency_id, sha256=dependency_sha)
            )
        reconstructed.append(
            ContentAddressedArtifact(
                artifact_id=artifact_id,
                path=str(row.get("path") or ""),
                content_path=str(row.get("content_path") or ""),
                sha256=digest,
                size_bytes=_safe_int(row.get("size_bytes"), default=-1),
                producer=str(row.get("producer") or ""),
                artifact_schema_version=str(row.get("artifact_schema_version") or ""),
                dependencies=tuple(dependencies),
                schema_version=str(row.get("schema_version") or ""),
            )
        )
        if str(row.get("schema_version") or "") != CONTENT_ADDRESSED_ARTIFACT_SCHEMA:
            reasons.append(f"closeout_artifact_schema_invalid:{artifact_id}")
        for path_field, required in (
            ("path", require_compatibility_paths),
            ("content_path", True),
        ):
            if path_field == "path" and not require_compatibility_paths:
                continue
            path_value = str(row.get(path_field) or "")
            if not path_value:
                reasons.append(f"closeout_artifact_{path_field}_missing:{artifact_id}")
                continue
            try:
                artifact_path = _resolve_under_root(root, path_value)
            except ArtifactRevisionError:
                reasons.append(f"closeout_artifact_{path_field}_outside_run:{artifact_id}")
                continue
            if path_field == "content_path":
                expected_parts = (
                    ".autoplanner",
                    "closeout",
                    "objects",
                    "sha256",
                    digest[:2],
                    digest,
                )
                relative_parts = artifact_path.relative_to(root).parts
                if tuple(relative_parts[:6]) != expected_parts:
                    reasons.append(f"closeout_artifact_content_path_not_addressed:{artifact_id}")
            normalized = artifact_path.as_posix().lower()
            if path_field == "path" and normalized in seen_paths:
                reasons.append(f"closeout_artifact_path_duplicate:{artifact_id}")
            if path_field == "path":
                seen_paths.add(normalized)
            if not artifact_path.is_file():
                if required:
                    reasons.append(f"closeout_artifact_{path_field}_missing:{artifact_id}")
                continue
            actual_digest = sha256_file(artifact_path)
            if actual_digest != digest:
                suffix = "content_drift" if path_field == "path" else "cas_corrupt"
                reasons.append(f"closeout_artifact_{suffix}:{artifact_id}")
            actual_size = artifact_path.stat().st_size
            if _safe_int(row.get("size_bytes"), default=-1) != actual_size:
                reasons.append(f"closeout_artifact_size_mismatch:{artifact_id}:{path_field}")

    for artifact_id, row in index.items():
        dependencies = [item for item in row.get("dependencies") or [] if isinstance(item, dict)]
        dependency_ids = {str(item.get("artifact_id") or "") for item in dependencies}
        for dependency in dependencies:
            dependency_id = str(dependency.get("artifact_id") or "")
            upstream = index.get(dependency_id)
            if upstream is None:
                reasons.append(f"closeout_dependency_missing:{artifact_id}:{dependency_id}")
                continue
            if str(dependency.get("sha256") or "").lower() != str(upstream.get("sha256") or "").lower():
                reasons.append(f"closeout_dependency_stale:{artifact_id}:{dependency_id}")
        for required_dependency in _REQUIRED_DEPENDENCIES.get(artifact_id, ()):
            if required_dependency in index and required_dependency not in dependency_ids:
                reasons.append(
                    f"closeout_required_dependency_missing:{artifact_id}:{required_dependency}"
                )
        if artifact_id in {"frontier_ledger", "parent_route_proof_snapshot"}:
            authority_graph_id = (
                "canonical_route_consensus_graph"
                if "canonical_route_consensus_graph" in index
                else "route_consensus_graph"
            )
            if (
                authority_graph_id in index
                and authority_graph_id not in dependency_ids
            ):
                reasons.append(
                    "closeout_required_dependency_missing:"
                    f"{artifact_id}:{authority_graph_id}"
                )
        if (
            artifact_id == "explored_route_forest"
            and "canonical_route_consensus_graph" in index
            and "canonical_route_consensus_graph" not in dependency_ids
        ):
            reasons.append(
                "closeout_required_dependency_missing:"
                "explored_route_forest:canonical_route_consensus_graph"
            )
    reasons.extend(_dependency_cycle_reasons(index))
    reasons.extend(_decision_semantic_reasons(root, index))

    expected_revision_id = _revision_id(
        case_id=str(manifest.get("case_id") or ""),
        producer=str(manifest.get("producer") or ""),
        artifacts=sorted(reconstructed, key=lambda item: item.artifact_id),
    ) if reconstructed else ""
    if revision_id != expected_revision_id:
        reasons.append("closeout_revision_id_mismatch")
    return _validation_report(
        not reasons,
        reasons,
        revision_id,
        manifest_path,
        len(index),
    )


def validate_latest_closeout_revision(run_dir: str | Path) -> dict[str, Any]:
    """Validate the atomic latest pointer and its referenced manifest."""
    root = Path(run_dir).expanduser().resolve()
    pointer_path = root / ".autoplanner" / "closeout" / "latest.json"
    if not pointer_path.is_file():
        return {
            **_validation_report(
                False,
                ["closeout_latest_pointer_missing"],
                "",
                "",
                0,
            ),
            "present": False,
            "pointer_path": str(pointer_path),
        }
    reasons: list[str] = []
    compatibility_validation = _validation_report(
        False,
        ["closeout_compatibility_projection_unavailable"],
        "",
        "",
        0,
    )
    try:
        pointer = _load_json_object(pointer_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ArtifactRevisionError) as exc:
        return {
            **_validation_report(
                False,
                [f"closeout_latest_pointer_invalid:{type(exc).__name__}:{exc}"],
                "",
                "",
                0,
            ),
            "present": True,
            "pointer_path": str(pointer_path),
        }
    if pointer.get("schema_version") != CLOSEOUT_LATEST_POINTER_SCHEMA:
        reasons.append("closeout_latest_pointer_schema_invalid")
    try:
        manifest_path = _resolve_under_root(root, str(pointer.get("manifest_path") or ""))
    except ArtifactRevisionError:
        manifest_path = root / "__invalid_closeout_manifest_path__"
        reasons.append("closeout_latest_manifest_path_invalid")
    if not manifest_path.is_file():
        reasons.append("closeout_latest_manifest_missing")
        manifest_validation = _validation_report(False, [], "", str(manifest_path), 0)
    else:
        expected_manifest_hash = str(pointer.get("manifest_sha256") or "").lower()
        if not _SHA256_RE.fullmatch(expected_manifest_hash):
            reasons.append("closeout_latest_manifest_sha256_invalid")
        elif sha256_file(manifest_path) != expected_manifest_hash:
            reasons.append("closeout_latest_manifest_drift")
        manifest_validation = validate_closeout_manifest(
            root,
            manifest_path,
            require_compatibility_paths=False,
        )
        compatibility_validation = validate_closeout_manifest(
            root,
            manifest_path,
            require_compatibility_paths=True,
        )
        reasons.extend(str(item) for item in manifest_validation.get("reasons") or [])
        try:
            pointed_manifest = _load_json_object(manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ArtifactRevisionError):
            pointed_manifest = {}
        if pointed_manifest.get("status") != "committed":
            reasons.append("closeout_latest_manifest_not_committed")
    pointer_revision = str(pointer.get("revision_id") or "")
    manifest_revision = str(manifest_validation.get("revision_id") or "")
    if pointer_revision != manifest_revision:
        reasons.append("closeout_latest_revision_mismatch")
    if pointer_revision.startswith("sha256:"):
        expected_name = f"{pointer_revision[7:]}.json"
        if manifest_path.name != expected_name:
            reasons.append("closeout_latest_manifest_path_revision_mismatch")
    report = _validation_report(
        not reasons,
        reasons,
        pointer_revision,
        str(manifest_path),
        _safe_int(manifest_validation.get("artifact_count")),
    )
    report.update(
        {
            "present": True,
            "pointer_path": str(pointer_path),
            "manifest_sha256": str(pointer.get("manifest_sha256") or ""),
            "compatibility_projection_validation": compatibility_validation,
            "compatibility_projection_drift": compatibility_validation.get("accepted")
            is not True,
        }
    )
    return report


def load_latest_closeout_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Return the validated latest manifest, or raise fail-closed."""
    root = Path(run_dir).expanduser().resolve()
    validation = validate_latest_closeout_revision(root)
    if validation.get("accepted") is not True:
        raise ArtifactRevisionError(
            "closeout_latest_invalid:"
            + ",".join(str(item) for item in validation.get("reasons") or [])
        )
    return _load_json_object(Path(str(validation["manifest_path"])))


def load_latest_closeout_artifact(
    run_dir: str | Path,
    artifact_id: str,
) -> dict[str, Any]:
    """Load one JSON artifact from the authoritative CAS object, never its view."""

    root = Path(run_dir).expanduser().resolve()
    manifest = load_latest_closeout_manifest(root)
    row = next(
        (
            dict(item)
            for item in manifest.get("artifacts") or []
            if isinstance(item, Mapping) and item.get("artifact_id") == artifact_id
        ),
        None,
    )
    if row is None:
        raise ArtifactRevisionError(f"closeout_artifact_not_found:{artifact_id}")
    path = _resolve_under_root(root, str(row.get("content_path") or ""))
    if path.suffix.lower() != ".json":
        raise ArtifactRevisionError(f"closeout_artifact_not_json:{artifact_id}")
    value = _load_json_object(path)
    if sha256_file(path) != str(row.get("sha256") or ""):
        raise ArtifactRevisionError(f"closeout_artifact_cas_corrupt:{artifact_id}")
    return value


def load_latest_closeout_decision(run_dir: str | Path) -> dict[str, Any]:
    """Return the proof and verdict that are authoritative for a committed run."""

    validation = validate_latest_closeout_revision(run_dir)
    if validation.get("accepted") is not True:
        raise ArtifactRevisionError(
            "closeout_latest_invalid:"
            + ",".join(str(item) for item in validation.get("reasons") or [])
        )
    proof_snapshot = load_latest_closeout_artifact(
        run_dir,
        "parent_route_proof_snapshot",
    )
    verdict_core = load_latest_closeout_artifact(run_dir, "final_verdict_core")
    return {
        "schema_version": "closeout_decision.v1",
        "revision_id": str(validation.get("revision_id") or ""),
        "proof_snapshot": proof_snapshot,
        "parent_route_proof": dict(proof_snapshot.get("proof") or {}),
        "final_verdict_core": verdict_core,
        "final_verdict": dict(verdict_core.get("verdict") or {}),
        "compatibility_projection_validation": dict(
            validation.get("compatibility_projection_validation") or {}
        ),
        "compatibility_projection_drift": validation.get("compatibility_projection_drift")
        is True,
    }


def _decision_semantic_reasons(
    root: Path,
    index: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    proof_row = index.get("parent_route_proof_snapshot")
    verdict_row = index.get("final_verdict_core")
    if proof_row is None and verdict_row is None:
        return []
    reasons: list[str] = []
    if proof_row is None:
        return ["closeout_parent_route_proof_snapshot_missing"]
    if verdict_row is None:
        return ["closeout_final_verdict_core_missing"]
    try:
        proof_snapshot = _load_json_object(
            _resolve_under_root(root, str(proof_row.get("content_path") or ""))
        )
        verdict_core = _load_json_object(
            _resolve_under_root(root, str(verdict_row.get("content_path") or ""))
        )
    except (ArtifactRevisionError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"closeout_decision_artifact_invalid:{type(exc).__name__}:{exc}"]
    if proof_snapshot.get("schema_version") != "parent_route_proof_snapshot.v1":
        reasons.append("closeout_parent_route_proof_snapshot_schema_invalid")
    if verdict_core.get("schema_version") != "final_verdict_core.v1":
        reasons.append("closeout_final_verdict_core_schema_invalid")
    proof = dict(proof_snapshot.get("proof") or {})
    expected_target = str(proof_snapshot.get("target_smiles") or "")
    try:
        from cascade_planner.harness.parent_route_proof import (
            is_solved_parent_route_proof,
        )

        replayed_solved = is_solved_parent_route_proof(
            proof,
            expected_target_smiles=expected_target,
        )
    except Exception as exc:  # pragma: no cover - defensive import boundary
        reasons.append(f"closeout_parent_route_proof_replay_error:{type(exc).__name__}:{exc}")
        replayed_solved = False
    if proof_snapshot.get("solved") is not replayed_solved:
        reasons.append("closeout_parent_route_proof_snapshot_solved_mismatch")
    if verdict_core.get("parent_route_proof_solved") is not replayed_solved:
        reasons.append("closeout_final_verdict_parent_proof_mismatch")
    verdict = dict(verdict_core.get("verdict") or {})
    verdict_solved = verdict.get("solved") is True
    if verdict_solved is not replayed_solved:
        reasons.append("closeout_final_verdict_solved_mismatch")
    if (str(verdict.get("verdict") or "") == "solved") is not verdict_solved:
        reasons.append("closeout_final_verdict_label_mismatch")
    if (str(verdict.get("route_status") or "") == "solved") is not verdict_solved:
        reasons.append("closeout_final_route_status_mismatch")
    validation = dict(verdict_core.get("validation") or {})
    if validation.get("accepted") is not True:
        reasons.append("closeout_final_verdict_core_validation_rejected")
    if str(proof_snapshot.get("case_id") or "") != str(verdict_core.get("case_id") or ""):
        reasons.append("closeout_decision_case_id_mismatch")
    forest_row = index.get("explored_route_forest")
    if forest_row is not None:
        try:
            forest = _load_json_object(
                _resolve_under_root(root, str(forest_row.get("content_path") or ""))
            )
        except (ArtifactRevisionError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            reasons.append(
                f"closeout_route_forest_semantic_read_error:{type(exc).__name__}:{exc}"
            )
            forest = {}
        branches = [
            dict(row)
            for row in forest.get("branches") or []
            if isinstance(row, Mapping)
        ]
        authoritative_branches = [
            row
            for row in branches
            if row.get("solved") is True
            or row.get("executable") is True
            or row.get("advisory_only") is False
            or row.get("not_parent_route_proof") is False
        ]
        if not replayed_solved and authoritative_branches:
            reasons.append("closeout_route_forest_false_authoritative_branch")
        if replayed_solved and not any(
            row.get("solved") is True
            and row.get("executable") is True
            and row.get("advisory_only") is False
            and row.get("not_parent_route_proof") is False
            for row in branches
        ):
            reasons.append("closeout_route_forest_solved_branch_missing")
        primary_status = str((forest.get("primary_selection") or {}).get("status") or "")
        if (primary_status == "deterministically_verified") is not replayed_solved:
            reasons.append("closeout_route_forest_primary_status_mismatch")
        forest_case_id = str(forest.get("case_id") or "")
        if forest_case_id and forest_case_id != str(proof_snapshot.get("case_id") or ""):
            reasons.append("closeout_route_forest_case_id_mismatch")
    return reasons


def _dependency_cycle_reasons(index: Mapping[str, Mapping[str, Any]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_nodes: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in visited:
            return
        if artifact_id in visiting:
            cycle_nodes.add(artifact_id)
            return
        visiting.add(artifact_id)
        row = index.get(artifact_id) or {}
        for dependency in row.get("dependencies") or []:
            if not isinstance(dependency, Mapping):
                continue
            dependency_id = str(dependency.get("artifact_id") or "")
            if dependency_id in index:
                visit(dependency_id)
                if dependency_id in cycle_nodes:
                    cycle_nodes.add(artifact_id)
        visiting.discard(artifact_id)
        visited.add(artifact_id)

    for artifact_id in index:
        visit(artifact_id)
    return [
        f"closeout_dependency_cycle:{artifact_id}"
        for artifact_id in sorted(cycle_nodes)
    ]


def _revision_id(
    *,
    case_id: str,
    producer: str,
    artifacts: Sequence[ContentAddressedArtifact],
) -> str:
    identity = {
        "schema_version": CLOSEOUT_REVISION_MANIFEST_SCHEMA,
        "case_id": case_id,
        "producer": producer,
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }
    digest = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return f"sha256:{digest}"


def _revision_digest(revision_id: str) -> str:
    if not revision_id.startswith("sha256:") or not _SHA256_RE.fullmatch(revision_id[7:]):
        raise ArtifactRevisionError(f"invalid_closeout_revision_id:{revision_id}")
    return revision_id[7:]


def _artifact_schema_version(path: Path) -> str:
    if path.suffix.lower() != ".json":
        return "text/html" if path.suffix.lower() in {".html", ".htm"} else "binary"
    try:
        value = _load_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ArtifactRevisionError):
        return "invalid_json"
    return str(value.get("schema_version") or "unversioned_json")


def _content_path(root: Path, artifact_id: str, suffix: str, digest: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", artifact_id).strip("._") or "artifact"
    safe_suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix or "") else ".bin"
    return (
        root
        / ".autoplanner"
        / "closeout"
        / "objects"
        / "sha256"
        / digest[:2]
        / digest
        / f"{safe_id}{safe_suffix}"
    )


def _resolve_under_root(root: Path, path: str | Path) -> Path:
    raw = Path(path).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactRevisionError(f"closeout_path_outside_run:{candidate}") from exc
    return candidate


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ArtifactRevisionError(f"immutable_closeout_content_conflict:{path}")
        return
    _atomic_write_bytes(path, payload)
    if path.read_bytes() != payload:
        raise ArtifactRevisionError(f"immutable_closeout_content_write_mismatch:{path}")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _pretty_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ArtifactRevisionError(f"immutable_closeout_manifest_conflict:{path}")
        return
    _write_immutable_bytes(path, payload)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _pretty_json_bytes(value))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactRevisionError(f"closeout_json_not_object:{path}")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _validation_report(
    accepted: bool,
    reasons: Sequence[str],
    revision_id: str,
    manifest_path: str,
    artifact_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": CLOSEOUT_VALIDATION_SCHEMA,
        "accepted": bool(accepted),
        "reasons": sorted(set(str(item) for item in reasons if str(item))),
        "revision_id": revision_id,
        "manifest_path": manifest_path,
        "artifact_count": int(artifact_count),
    }


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ArtifactRevisionError",
    "CLOSEOUT_VALIDATION_SCHEMA",
    "load_latest_closeout_manifest",
    "load_latest_closeout_artifact",
    "load_latest_closeout_decision",
    "publish_closeout_revision",
    "sha256_file",
    "validate_closeout_manifest",
    "validate_latest_closeout_revision",
]
