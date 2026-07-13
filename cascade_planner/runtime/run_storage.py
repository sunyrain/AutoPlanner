"""Publish immutable run artifacts and rebuildable operational projections."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .artifact_store import ArtifactRef, ArtifactStore
from .run_index import RUN_MANIFEST_SCHEMA, RunIndex


RUN_STORAGE_PUBLISH_SCHEMA = "autoplanner_run_storage_publish.v1"
RUN_STORAGE_REBUILD_SCHEMA = "autoplanner_run_storage_rebuild.v1"


def publish_run_projection(
    runtime_root: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, str | os.PathLike[str]],
    authority_scopes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Publish bytes first, then a manifest pointer and rebuildable index rows."""

    root = Path(runtime_root).expanduser().resolve()
    store = ArtifactStore(root / "artifacts")
    index = RunIndex(root / "run_index.sqlite3")
    row = dict(manifest)
    if row.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("run_storage_manifest_schema_invalid")
    run_id = str(row.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_storage_manifest_id_missing")
    scopes = {str(key): str(value) for key, value in (authority_scopes or {}).items()}
    indexed_artifacts: list[dict[str, Any]] = []
    for artifact_id, raw_path in sorted(artifacts.items()):
        path = Path(raw_path).expanduser().resolve()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        ref = store.put_file(
            path,
            media_type=media_type,
            logical_name=path.name,
            producer=str(row.get("producer") or "autoplanner"),
        )
        indexed_artifacts.append(
            {
                "artifact_id": str(artifact_id),
                "revision": max(0, int(row.get("revision") or 0)),
                "authority_scope": scopes.get(
                    str(artifact_id),
                    "operational_projection",
                ),
                "ref": ref.to_dict(),
            }
        )
    row["indexed_artifacts"] = indexed_artifacts
    row.setdefault(
        "semantics",
        {
            "manifest_is_replay_input_for_operational_index": True,
            "manifest_does_not_grant_scientific_authority": True,
        },
    )
    manifest_ref = store.put_json(
        row,
        logical_name="run_manifest.json",
        producer=str(row.get("producer") or "autoplanner"),
    )
    pointer_name = f"runs/{hashlib.sha256(run_id.encode('utf-8')).hexdigest()}/latest"
    pointer_path = store.write_pointer(
        pointer_name,
        manifest_ref,
        metadata={
            "run_id": run_id,
            "revision": max(0, int(row.get("revision") or 0)),
        },
    )
    index.upsert_run(row)
    for artifact in indexed_artifacts:
        index.index_artifact(
            run_id=run_id,
            artifact_id=str(artifact["artifact_id"]),
            revision=int(artifact["revision"]),
            authority_scope=str(artifact["authority_scope"]),
            ref=ArtifactRef.from_dict(artifact["ref"]),
        )
    return {
        "schema_version": RUN_STORAGE_PUBLISH_SCHEMA,
        "run_id": run_id,
        "revision": max(0, int(row.get("revision") or 0)),
        "manifest_ref": manifest_ref.to_dict(),
        "manifest": row,
        "manifest_pointer_path": str(pointer_path),
        "artifact_count": len(indexed_artifacts),
        "index_health": index.health(),
        "semantics": {
            "objects_are_immutable": True,
            "index_is_rebuildable": True,
            "storage_grants_no_scientific_authority": True,
        },
    }


def rebuild_run_index(
    runtime_root: str | os.PathLike[str],
    *,
    index_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Rebuild an index solely from manifest objects pinned by run pointers."""

    root = Path(runtime_root).expanduser().resolve()
    store = ArtifactStore(root / "artifacts")
    manifests: list[dict[str, Any]] = []
    pointer_paths = sorted(store.pointers_root.glob("runs/*/latest.json"))
    for pointer_path in pointer_paths:
        relative = pointer_path.relative_to(store.pointers_root).as_posix()
        pointer_name = relative.removesuffix(".json")
        ref, pointer = store.load_pointer(pointer_name)
        manifest = store.read_json(ref)
        if not isinstance(manifest, dict):
            raise ValueError("run_storage_manifest_not_object")
        metadata = dict(pointer.get("metadata") or {})
        if str(manifest.get("run_id") or "") != str(metadata.get("run_id") or ""):
            raise ValueError("run_storage_pointer_run_id_mismatch")
        if int(manifest.get("revision") or 0) != int(metadata.get("revision") or 0):
            raise ValueError("run_storage_pointer_revision_mismatch")
        manifests.append(manifest)
    resolved_index_path = (
        Path(index_path).expanduser().resolve()
        if index_path is not None
        else root / "run_index.sqlite3"
    )
    index = RunIndex(resolved_index_path)
    report = index.rebuild(manifests)
    return {
        "schema_version": RUN_STORAGE_REBUILD_SCHEMA,
        "manifest_count": len(manifests),
        "pointer_count": len(pointer_paths),
        "index_path": str(resolved_index_path),
        "index_report": report,
        "semantics": {
            "rebuilt_from_immutable_manifest_objects": True,
            "scientific_artifacts_were_not_modified": True,
        },
    }


def run_storage_object_stats(
    runtime_root: str | os.PathLike[str],
) -> dict[str, int]:
    root = Path(runtime_root).expanduser().resolve()
    store = ArtifactStore(root / "artifacts")
    paths = store.objects_root.glob("*/*")
    sizes = [path.stat().st_size for path in paths if path.is_file()]
    index = RunIndex(root / "run_index.sqlite3")
    health = index.health()
    return {
        "object_count": len(sizes),
        "object_bytes": sum(sizes),
        "indexed_run_count": int(health["run_count"]),
        "indexed_artifact_count": int(health["artifact_count"]),
        "indexed_task_count": int(health["task_count"]),
    }


def write_run_manifest_compatibility(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> Path:
    """Write a small compatibility manifest; immutable storage remains primary."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


__all__ = [
    "RUN_STORAGE_PUBLISH_SCHEMA",
    "RUN_STORAGE_REBUILD_SCHEMA",
    "publish_run_projection",
    "rebuild_run_index",
    "run_storage_object_stats",
    "write_run_manifest_compatibility",
]
