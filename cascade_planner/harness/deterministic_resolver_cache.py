"""CAS-backed, non-authoritative cache for deterministic name resolution."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from cascade_planner.runtime.artifact_store import (
    ArtifactReferenceError,
    ArtifactStore,
)


RESOLVER_CACHE_SCHEMA = "deterministic_literature_resolver_cache.v1"


class DeterministicResolverCache:
    """Versioned cache whose contents can accelerate but never prove chemistry."""

    def __init__(
        self,
        store_root: str | Path,
        *,
        authority_id: str,
        opsin_base_url: str,
        pubchem_base_url: str,
        failure_ttl_s: float = 86_400.0,
    ) -> None:
        self.store = ArtifactStore(store_root)
        self.namespace = {
            "authority_id": str(authority_id),
            "opsin_base_url": str(opsin_base_url).rstrip("/"),
            "pubchem_base_url": str(pubchem_base_url).rstrip("/"),
        }
        namespace_sha = _digest(self.namespace)
        self.pointer_name = f"caches/deterministic-resolver/{namespace_sha}/latest"
        self.failure_ttl_s = max(0.0, float(failure_ttl_s))
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._load_latest()

    def get(self, kind: str, query: str) -> tuple[bool, Any]:
        key = _entry_key(kind, query)
        entry = self._entries.get(key)
        if not isinstance(entry, dict):
            return False, None
        if entry.get("kind") != kind:
            return False, None
        status = str(entry.get("status") or "")
        if status == "success":
            return True, entry.get("value")
        if status != "failure":
            return False, None
        observed_epoch_s = float(entry.get("observed_epoch_s") or 0.0)
        if time.time() - observed_epoch_s > self.failure_ttl_s:
            return False, None
        return True, entry.get("value")

    def put(self, kind: str, query: str, value: Any, *, success: bool) -> None:
        self._entries[_entry_key(kind, query)] = {
            "kind": str(kind),
            "query_sha256": hashlib.sha256(
                str(query).encode("utf-8")
            ).hexdigest(),
            "status": "success" if success else "failure",
            "value": value,
            "observed_at": _utc_now(),
            "observed_epoch_s": round(time.time(), 3),
        }
        self._dirty = True

    def flush(self) -> dict[str, Any]:
        if not self._dirty:
            return {
                "flushed": False,
                "entry_count": len(self._entries),
            }
        merged = dict(self._entries)
        latest = self._read_latest_payload()
        for key, raw in dict(latest.get("entries") or {}).items():
            if not isinstance(raw, Mapping):
                continue
            existing = dict(merged.get(str(key)) or {})
            if float(raw.get("observed_epoch_s") or 0.0) > float(
                existing.get("observed_epoch_s") or 0.0
            ):
                merged[str(key)] = dict(raw)
        payload = {
            "schema_version": RESOLVER_CACHE_SCHEMA,
            "namespace": dict(self.namespace),
            "entries": dict(sorted(merged.items())),
            "updated_at": _utc_now(),
            "semantics": {
                "cache_is_non_authoritative": True,
                "cache_cannot_approve_reaction_or_route": True,
                "failed_entries_expire": True,
                "namespace_change_invalidates_cache": True,
            },
        }
        ref = self.store.put_json(
            payload,
            logical_name="deterministic_resolver_cache.json",
            producer="autoplanner.deterministic_literature_parser",
        )
        self.store.write_pointer(
            self.pointer_name,
            ref,
            metadata={
                "entry_count": len(merged),
                "namespace_sha256": _digest(self.namespace),
            },
        )
        self._entries = merged
        self._dirty = False
        return {
            "flushed": True,
            "entry_count": len(merged),
            "artifact_ref": ref.to_dict(),
        }

    def _load_latest(self) -> None:
        payload = self._read_latest_payload()
        self._entries = {
            str(key): dict(value)
            for key, value in dict(payload.get("entries") or {}).items()
            if isinstance(value, Mapping)
        }

    def _read_latest_payload(self) -> dict[str, Any]:
        try:
            ref, _ = self.store.load_pointer(self.pointer_name)
        except ArtifactReferenceError:
            return {}
        payload = self.store.read_json(ref)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != RESOLVER_CACHE_SCHEMA
            or payload.get("namespace") != self.namespace
            or not isinstance(payload.get("entries"), dict)
        ):
            return {}
        return payload


def _entry_key(kind: str, query: str) -> str:
    return f"{kind}:{hashlib.sha256(str(query).encode('utf-8')).hexdigest()}"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["RESOLVER_CACHE_SCHEMA", "DeterministicResolverCache"]
