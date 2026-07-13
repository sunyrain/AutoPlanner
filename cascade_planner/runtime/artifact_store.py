"""Shared immutable content-addressed artifact storage.

The store owns bytes, not scientific meaning.  A digest can prove that two
consumers read the same bytes, but it cannot promote evidence, chemistry,
stock, or route completion.  Mutable compatibility files and indexes may point
at objects and can always be rebuilt.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from typing import Any, BinaryIO, Iterable, Mapping


ARTIFACT_REF_SCHEMA = "autoplanner_artifact_ref.v1"
ARTIFACT_POINTER_SCHEMA = "autoplanner_artifact_pointer.v1"
ARTIFACT_GC_PLAN_SCHEMA = "autoplanner_artifact_gc_plan.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_POINTER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,239}$")


class ArtifactStoreError(RuntimeError):
    """Base error for immutable artifact storage."""


class ArtifactCorruptionError(ArtifactStoreError):
    """Raised when object bytes disagree with their content address."""


class ArtifactReferenceError(ArtifactStoreError):
    """Raised when a reference or pointer violates the store contract."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    logical_name: str = ""
    producer: str = ""
    schema_version: str = ARTIFACT_REF_SCHEMA

    def __post_init__(self) -> None:
        digest = str(self.sha256).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ArtifactReferenceError("artifact_ref_sha256_invalid")
        if int(self.size_bytes) < 0:
            raise ArtifactReferenceError("artifact_ref_size_invalid")
        if not str(self.media_type or "").strip():
            raise ArtifactReferenceError("artifact_ref_media_type_missing")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "size_bytes", int(self.size_bytes))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        row = dict(value)
        if row.get("schema_version") != ARTIFACT_REF_SCHEMA:
            raise ArtifactReferenceError("artifact_ref_schema_invalid")
        return cls(
            sha256=str(row.get("sha256") or ""),
            size_bytes=int(row.get("size_bytes") or 0),
            media_type=str(row.get("media_type") or ""),
            logical_name=str(row.get("logical_name") or ""),
            producer=str(row.get("producer") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "logical_name": self.logical_name,
            "producer": self.producer,
            "object_path": f"objects/sha256/{self.sha256[:2]}/{self.sha256}",
            "semantics": {
                "content_identity_only": True,
                "grants_no_scientific_authority": True,
            },
        }


class ArtifactStore:
    """Atomic SHA-256 store safe for concurrent identical writers."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects_root = self.root / "objects" / "sha256"
        self.pointers_root = self.root / "pointers"
        self.temp_root = self.root / ".tmp"
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.pointers_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()

    def put_bytes(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
        logical_name: str = "",
        producer: str = "",
    ) -> ArtifactRef:
        data = bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        path = self.object_path(digest)
        with self._write_lock:
            if path.is_file():
                self._verify_object(path, digest=digest, size_bytes=len(data))
            else:
                self._write_object_bytes(path, data)
        return ArtifactRef(
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
            logical_name=logical_name,
            producer=producer,
        )

    def put_json(
        self,
        value: Any,
        *,
        logical_name: str = "",
        producer: str = "",
    ) -> ArtifactRef:
        try:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreError(
                f"artifact_json_not_canonicalizable:{type(exc).__name__}"
            ) from exc
        return self.put_bytes(
            payload,
            media_type="application/json",
            logical_name=logical_name,
            producer=producer,
        )

    def put_file(
        self,
        source: str | os.PathLike[str],
        *,
        media_type: str = "application/octet-stream",
        logical_name: str = "",
        producer: str = "",
    ) -> ArtifactRef:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise ArtifactStoreError(f"artifact_source_missing:{source_path}")
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="artifact.",
            suffix=".tmp",
            dir=self.temp_root,
            delete=False,
        )
        temporary = Path(handle.name)
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with source_path.open("rb") as source_handle, handle:
                for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                    handle.write(block)
                    digest.update(block)
                    size_bytes += len(block)
                handle.flush()
                os.fsync(handle.fileno())
            digest_value = digest.hexdigest()
            object_path = self.object_path(digest_value)
            with self._write_lock:
                if object_path.is_file():
                    self._verify_object(
                        object_path,
                        digest=digest_value,
                        size_bytes=size_bytes,
                    )
                else:
                    object_path.parent.mkdir(parents=True, exist_ok=True)
                    self._publish_temporary(
                        temporary,
                        object_path,
                        digest=digest_value,
                        size_bytes=size_bytes,
                    )
                self._verify_object(
                    object_path,
                    digest=digest_value,
                    size_bytes=size_bytes,
                )
            return ArtifactRef(
                sha256=digest_value,
                size_bytes=size_bytes,
                media_type=media_type,
                logical_name=logical_name or source_path.name,
                producer=producer,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def object_path(self, digest: str) -> Path:
        normalized = str(digest or "").lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ArtifactReferenceError("artifact_digest_invalid")
        return self.objects_root / normalized[:2] / normalized

    def contains(self, ref_or_digest: ArtifactRef | str) -> bool:
        digest = (
            ref_or_digest.sha256
            if isinstance(ref_or_digest, ArtifactRef)
            else str(ref_or_digest)
        )
        path = self.object_path(digest)
        return path.is_file()

    def verify(self, ref_or_digest: ArtifactRef | str) -> bool:
        if isinstance(ref_or_digest, ArtifactRef):
            digest = ref_or_digest.sha256
            expected_size: int | None = ref_or_digest.size_bytes
        else:
            digest = str(ref_or_digest)
            expected_size = None
        self._verify_object(
            self.object_path(digest),
            digest=digest,
            size_bytes=expected_size,
        )
        return True

    def open(self, ref_or_digest: ArtifactRef | str) -> BinaryIO:
        self.verify(ref_or_digest)
        digest = (
            ref_or_digest.sha256
            if isinstance(ref_or_digest, ArtifactRef)
            else str(ref_or_digest)
        )
        return self.object_path(digest).open("rb")

    def read_bytes(self, ref_or_digest: ArtifactRef | str) -> bytes:
        with self.open(ref_or_digest) as handle:
            return handle.read()

    def read_json(self, ref_or_digest: ArtifactRef | str) -> Any:
        try:
            return json.loads(self.read_bytes(ref_or_digest).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactCorruptionError(
                f"artifact_json_invalid:{type(exc).__name__}"
            ) from exc

    def materialize(
        self,
        ref_or_digest: ArtifactRef | str,
        destination: str | os.PathLike[str],
    ) -> Path:
        self.verify(ref_or_digest)
        digest = (
            ref_or_digest.sha256
            if isinstance(ref_or_digest, ArtifactRef)
            else str(ref_or_digest)
        )
        source = self.object_path(digest)
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with source.open("rb") as source_handle, handle:
                shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def write_pointer(
        self,
        name: str,
        ref: ArtifactRef,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        pointer_path = self._pointer_path(name)
        payload: dict[str, Any] = {
            "schema_version": ARTIFACT_POINTER_SCHEMA,
            "name": name,
            "artifact": ref.to_dict(),
            "metadata": dict(metadata or {}),
            "updated_at": _utc_now(),
            "semantics": {
                "pointer_is_mutable_projection": True,
                "artifact_bytes_are_immutable": True,
                "pointer_grants_no_scientific_authority": True,
            },
        }
        self._atomic_write_json(pointer_path, payload)
        return pointer_path

    def load_pointer(self, name: str) -> tuple[ArtifactRef, dict[str, Any]]:
        path = self._pointer_path(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactReferenceError(
                f"artifact_pointer_invalid:{type(exc).__name__}"
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != ARTIFACT_POINTER_SCHEMA
            or payload.get("name") != name
            or not isinstance(payload.get("artifact"), dict)
        ):
            raise ArtifactReferenceError("artifact_pointer_contract_invalid")
        ref = ArtifactRef.from_dict(payload["artifact"])
        self.verify(ref)
        return ref, dict(payload)

    def garbage_collection_plan(
        self,
        *,
        pinned_digests: Iterable[str] = (),
        minimum_age_s: float = 86_400.0,
    ) -> dict[str, Any]:
        pinned = {str(item).lower() for item in pinned_digests}
        pinned.update(self._pointer_digests())
        now = time.time()
        candidates: list[dict[str, Any]] = []
        retained_count = 0
        for path in self._object_paths():
            digest = path.name.lower()
            age_s = max(0.0, now - path.stat().st_mtime)
            if digest in pinned or age_s < max(0.0, float(minimum_age_s)):
                retained_count += 1
                continue
            candidates.append(
                {
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "age_s": round(age_s, 3),
                }
            )
        return {
            "schema_version": ARTIFACT_GC_PLAN_SCHEMA,
            "dry_run": True,
            "candidate_count": len(candidates),
            "candidate_bytes": sum(row["size_bytes"] for row in candidates),
            "retained_count": retained_count,
            "pinned_digest_count": len(pinned),
            "candidates": sorted(candidates, key=lambda row: row["sha256"]),
            "semantics": {
                "plan_does_not_delete": True,
                "pointers_are_automatically_pinned": True,
            },
        }

    def collect_garbage(
        self,
        plan: Mapping[str, Any],
        *,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if not confirm:
            raise ArtifactStoreError("artifact_gc_requires_explicit_confirmation")
        if plan.get("schema_version") != ARTIFACT_GC_PLAN_SCHEMA:
            raise ArtifactStoreError("artifact_gc_plan_schema_invalid")
        removed: list[str] = []
        skipped: list[str] = []
        pinned = self._pointer_digests()
        for row in plan.get("candidates") or []:
            digest = str((row or {}).get("sha256") or "").lower()
            path = self.object_path(digest)
            if digest in pinned or not path.is_file():
                skipped.append(digest)
                continue
            self._verify_object(
                path,
                digest=digest,
                size_bytes=int((row or {}).get("size_bytes") or 0),
            )
            path.unlink()
            removed.append(digest)
        return {
            "schema_version": "autoplanner_artifact_gc_result.v1",
            "removed": sorted(removed),
            "skipped": sorted(skipped),
        }

    def _write_object_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="artifact.",
            suffix=".tmp",
            dir=self.temp_root,
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_file():
                self._verify_object(
                    path,
                    digest=path.name,
                    size_bytes=len(payload),
                )
            else:
                self._publish_temporary(
                    temporary,
                    path,
                    digest=path.name,
                    size_bytes=len(payload),
                )
        finally:
            temporary.unlink(missing_ok=True)

    def _verify_object(
        self,
        path: Path,
        *,
        digest: str,
        size_bytes: int | None,
    ) -> None:
        if not path.is_file():
            raise ArtifactCorruptionError(f"artifact_object_missing:{digest}")
        if size_bytes is not None and path.stat().st_size != int(size_bytes):
            raise ArtifactCorruptionError(f"artifact_object_size_mismatch:{digest}")
        observed = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                observed.update(block)
        if observed.hexdigest() != str(digest).lower():
            raise ArtifactCorruptionError(f"artifact_object_digest_mismatch:{digest}")

    def _publish_temporary(
        self,
        temporary: Path,
        target: Path,
        *,
        digest: str,
        size_bytes: int,
    ) -> None:
        try:
            os.replace(temporary, target)
        except OSError:
            # Another process may have published the same digest between the
            # existence check and replace.  Reuse it only after full validation.
            if not target.is_file():
                raise
            self._verify_object(
                target,
                digest=digest,
                size_bytes=size_bytes,
            )
        else:
            self._fsync_directory(target.parent)

    def _pointer_path(self, name: str) -> Path:
        normalized = str(name or "").replace("\\", "/")
        if (
            not _SAFE_POINTER_RE.fullmatch(normalized)
            or ".." in normalized.split("/")
        ):
            raise ArtifactReferenceError("artifact_pointer_name_invalid")
        path = (self.pointers_root / f"{normalized}.json").resolve()
        try:
            path.relative_to(self.pointers_root)
        except ValueError as exc:
            raise ArtifactReferenceError("artifact_pointer_path_escape") from exc
        return path

    def _pointer_digests(self) -> set[str]:
        digests: set[str] = set()
        for path in self.pointers_root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                digest = str((payload.get("artifact") or {}).get("sha256") or "")
            except (OSError, AttributeError, json.JSONDecodeError):
                continue
            if _SHA256_RE.fullmatch(digest):
                digests.add(digest)
        return digests

    def _object_paths(self) -> list[Path]:
        return [
            path
            for path in self.objects_root.glob("*/*")
            if path.is_file() and _SHA256_RE.fullmatch(path.name)
        ]

    @staticmethod
    def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            ArtifactStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "ARTIFACT_GC_PLAN_SCHEMA",
    "ARTIFACT_POINTER_SCHEMA",
    "ARTIFACT_REF_SCHEMA",
    "ArtifactCorruptionError",
    "ArtifactRef",
    "ArtifactReferenceError",
    "ArtifactStore",
    "ArtifactStoreError",
]
