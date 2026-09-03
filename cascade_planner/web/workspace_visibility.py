"""Durable, recoverable visibility controls for the unified workspace.

Deleting a workspace row removes only its rebuildable presentation projection.
Canonical run directories, graph events and scientific artifacts remain intact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Mapping


WORKSPACE_VISIBILITY_SCHEMA = "autoplanner.workspace_visibility.v1"
_LOCK = RLock()


class WorkspaceVisibilityError(RuntimeError):
    """Raised when the workspace visibility registry is invalid."""


class WorkspaceVisibilityStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def snapshot(self) -> dict[str, Any]:
        with _LOCK:
            return self._read()

    def hide_route(self, route_id: str) -> dict[str, Any]:
        identity = _identity(route_id, kind="route")
        with _LOCK:
            value = self._read()
            value["hidden_routes"][identity] = {"deleted_at": _utc_now()}
            return self._write(value, operation="delete_route", identity=identity)

    def hide_queue_run(self, run_id: str) -> dict[str, Any]:
        identity = _identity(run_id, kind="run")
        with _LOCK:
            value = self._read()
            value["hidden_queue_runs"][identity] = {"deleted_at": _utc_now()}
            return self._write(value, operation="delete_queue_run", identity=identity)

    def restore(self, *, scope: str, identity: str = "") -> dict[str, Any]:
        normalized_scope = str(scope or "").strip().lower()
        if normalized_scope not in {"routes", "queue", "all"}:
            raise WorkspaceVisibilityError("workspace_restore_scope_invalid")
        with _LOCK:
            value = self._read()
            restored = 0
            targets = []
            if normalized_scope in {"routes", "all"}:
                targets.append(("hidden_routes", "route"))
            if normalized_scope in {"queue", "all"}:
                targets.append(("hidden_queue_runs", "run"))
            for key, kind in targets:
                if identity:
                    normalized = _identity(identity, kind=kind)
                    restored += int(value[key].pop(normalized, None) is not None)
                else:
                    restored += len(value[key])
                    value[key] = {}
            result = self._write(
                value,
                operation=f"restore_{normalized_scope}",
                identity=str(identity or ""),
            )
            result["restored_count"] = restored
            return result

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return _empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkspaceVisibilityError("workspace_visibility_registry_unreadable") from exc
        if not isinstance(raw, Mapping) or raw.get("schema_version") != WORKSPACE_VISIBILITY_SCHEMA:
            raise WorkspaceVisibilityError("workspace_visibility_registry_schema_invalid")
        return {
            "schema_version": WORKSPACE_VISIBILITY_SCHEMA,
            "revision": max(0, int(raw.get("revision") or 0)),
            "updated_at": str(raw.get("updated_at") or ""),
            "hidden_routes": _records(raw.get("hidden_routes")),
            "hidden_queue_runs": _records(raw.get("hidden_queue_runs")),
        }

    def _write(
        self,
        value: Mapping[str, Any],
        *,
        operation: str,
        identity: str,
    ) -> dict[str, Any]:
        row = {
            "schema_version": WORKSPACE_VISIBILITY_SCHEMA,
            "revision": max(0, int(value.get("revision") or 0)) + 1,
            "updated_at": _utc_now(),
            "hidden_routes": _records(value.get("hidden_routes")),
            "hidden_queue_runs": _records(value.get("hidden_queue_runs")),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        return {
            **row,
            "operation": operation,
            "identity": identity,
            "removed_from_workspace": operation.startswith("delete_"),
            "scientific_artifacts_preserved": True,
            "recoverable": True,
        }


def workspace_visibility_store(gateway: Any) -> WorkspaceVisibilityStore:
    runtime_root = getattr(getattr(gateway, "paths", None), "runtime_root", None)
    if runtime_root is None:
        raise WorkspaceVisibilityError("workspace_visibility_runtime_root_unavailable")
    runtime_root = Path(runtime_root)
    return WorkspaceVisibilityStore(runtime_root / "workspace_visibility.json")


def _empty() -> dict[str, Any]:
    return {
        "schema_version": WORKSPACE_VISIBILITY_SCHEMA,
        "revision": 0,
        "updated_at": "",
        "hidden_routes": {},
        "hidden_queue_runs": {},
    }


def _records(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): {"deleted_at": str(dict(record).get("deleted_at") or "")}
        for key, record in value.items()
        if str(key).strip() and isinstance(record, Mapping)
    }


def _identity(value: Any, *, kind: str) -> str:
    identity = str(value or "").strip()
    if not identity or len(identity) > 512 or any(ord(char) < 32 for char in identity):
        raise WorkspaceVisibilityError(f"workspace_{kind}_identity_invalid")
    return identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "WORKSPACE_VISIBILITY_SCHEMA",
    "WorkspaceVisibilityError",
    "WorkspaceVisibilityStore",
    "workspace_visibility_store",
]
