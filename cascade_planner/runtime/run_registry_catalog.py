"""Explicit discovery catalog for independently rooted AutoPlanner run indexes.

The catalog is an operational projection only.  It stores where a registry
lives and which project it belongs to; run lifecycle and scientific state stay
owned by that registry's RunIndex/RunKernel artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator

from cascade_planner.runtime.paths import RuntimePaths


RUN_REGISTRY_CATALOG_SCHEMA = "autoplanner_run_registry_catalog.v1"
RUN_REGISTRY_BINDING_SCHEMA = "autoplanner_run_registry_binding.v1"
_REGISTRY_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


class RunRegistryCatalogError(RuntimeError):
    """Raised when registry discovery metadata violates its contract."""


@dataclass(frozen=True, slots=True)
class RunRegistryBinding:
    registry_id: str
    registry_label: str
    project_id: str
    project_label: str
    case_id: str
    repository_root: Path
    runtime_root: Path
    runs_root: Path
    artifact_store_root: Path
    run_index_path: Path
    external_data_root: Path
    vendor_root: Path
    source: str = "explicit"
    read_only: bool = True
    display_order: int = 0
    registered_at: str = ""
    updated_at: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("registry_id", "project_id"):
            value = str(getattr(self, name) or "").strip().casefold()
            if not _REGISTRY_ID.fullmatch(value):
                raise RunRegistryCatalogError(f"run_registry_{name}_invalid")
            object.__setattr__(self, name, value)
        for name in ("registry_label", "project_label"):
            value = str(getattr(self, name) or "").strip()
            if not value or len(value) > 200:
                raise RunRegistryCatalogError(f"run_registry_{name}_invalid")
            object.__setattr__(self, name, value)
        case_id = str(self.case_id or "").strip()
        if len(case_id) > 200 or any(ord(char) < 32 for char in case_id):
            raise RunRegistryCatalogError("run_registry_case_id_invalid")
        object.__setattr__(self, "case_id", case_id)
        for name in (
            "repository_root",
            "runtime_root",
            "runs_root",
            "artifact_store_root",
            "run_index_path",
            "external_data_root",
            "vendor_root",
        ):
            object.__setattr__(
                self,
                name,
                Path(getattr(self, name)).expanduser().resolve(),
            )
        source = str(self.source or "explicit").strip()
        if not source or len(source) > 100:
            raise RunRegistryCatalogError("run_registry_source_invalid")
        object.__setattr__(self, "source", source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_REGISTRY_BINDING_SCHEMA,
            "registry_id": self.registry_id,
            "registry_label": self.registry_label,
            "project_id": self.project_id,
            "project_label": self.project_label,
            "case_id": self.case_id,
            "repository_root": str(self.repository_root),
            "runtime_root": str(self.runtime_root),
            "runs_root": str(self.runs_root),
            "artifact_store_root": str(self.artifact_store_root),
            "run_index_path": str(self.run_index_path),
            "external_data_root": str(self.external_data_root),
            "vendor_root": str(self.vendor_root),
            "source": self.source,
            "read_only": self.read_only,
            "display_order": self.display_order,
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
            "enabled": self.enabled,
            "semantics": {
                "catalog_is_discovery_only": True,
                "registry_retains_run_lifecycle_authority": True,
                "catalog_grants_no_scientific_authority": True,
            },
        }

    def runtime_paths(self) -> RuntimePaths:
        return RuntimePaths(
            repository_root=self.repository_root,
            runtime_root=self.runtime_root,
            runs_root=self.runs_root,
            artifact_store_root=self.artifact_store_root,
            run_index_path=self.run_index_path,
            cache_root=self.runtime_root / "cache",
            source_root=self.runtime_root / "sources",
            external_data_root=self.external_data_root,
            model_root=self.external_data_root / "models",
            vendor_root=self.vendor_root,
        )


class RunRegistryCatalog:
    """WAL-backed registry-of-registries with no run-status copies."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register(self, binding: RunRegistryBinding) -> dict[str, Any]:
        now = _utc_now()
        row = binding.to_dict()
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                INSERT INTO registries (
                    registry_id, registry_label, project_id, project_label,
                    case_id, repository_root, runtime_root, runs_root,
                    artifact_store_root, run_index_path, external_data_root,
                    vendor_root, source, read_only, display_order,
                    registered_at, updated_at, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registry_id) DO UPDATE SET
                    registry_label = excluded.registry_label,
                    project_id = excluded.project_id,
                    project_label = excluded.project_label,
                    case_id = excluded.case_id,
                    repository_root = excluded.repository_root,
                    runtime_root = excluded.runtime_root,
                    runs_root = excluded.runs_root,
                    artifact_store_root = excluded.artifact_store_root,
                    run_index_path = excluded.run_index_path,
                    external_data_root = excluded.external_data_root,
                    vendor_root = excluded.vendor_root,
                    source = excluded.source,
                    read_only = excluded.read_only,
                    display_order = excluded.display_order,
                    updated_at = excluded.updated_at,
                    enabled = excluded.enabled
                """,
                    (
                        row["registry_id"],
                        row["registry_label"],
                        row["project_id"],
                        row["project_label"],
                        row["case_id"],
                        row["repository_root"],
                        row["runtime_root"],
                        row["runs_root"],
                        row["artifact_store_root"],
                        row["run_index_path"],
                        row["external_data_root"],
                        row["vendor_root"],
                        row["source"],
                        int(row["read_only"]),
                        int(row["display_order"]),
                        now,
                        now,
                        int(row["enabled"]),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "run_index_path" in str(exc):
                raise RunRegistryCatalogError("run_registry_path_already_registered") from exc
            raise RunRegistryCatalogError("run_registry_registration_conflict") from exc
        registered = self.get(binding.registry_id)
        if registered is None:
            raise RunRegistryCatalogError("run_registry_registration_missing")
        return {
            "schema_version": RUN_REGISTRY_CATALOG_SCHEMA,
            "operation": "register",
            "registry": registered.to_dict(),
        }

    def get(self, registry_id: str) -> RunRegistryBinding | None:
        identity = _normalize_id(registry_id, field="registry_id")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM registries WHERE registry_id = ?",
                (identity,),
            ).fetchone()
        return _binding_from_row(row) if row is not None else None

    def list_registries(self, *, enabled_only: bool = True) -> list[RunRegistryBinding]:
        query = "SELECT * FROM registries"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY display_order ASC, project_id ASC, registry_id ASC"
        with self._connection() as connection:
            rows = connection.execute(query).fetchall()
        return [_binding_from_row(row) for row in rows]

    def set_enabled(self, registry_id: str, *, enabled: bool) -> dict[str, Any]:
        identity = _normalize_id(registry_id, field="registry_id")
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE registries SET enabled = ?, updated_at = ? WHERE registry_id = ?",
                (int(enabled), _utc_now(), identity),
            ).rowcount
        if not changed:
            raise RunRegistryCatalogError("run_registry_not_found")
        return {
            "schema_version": RUN_REGISTRY_CATALOG_SCHEMA,
            "operation": "set_enabled",
            "registry_id": identity,
            "enabled": bool(enabled),
        }

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registries (
                    registry_id TEXT PRIMARY KEY,
                    registry_label TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    project_label TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    repository_root TEXT NOT NULL,
                    runtime_root TEXT NOT NULL,
                    runs_root TEXT NOT NULL,
                    artifact_store_root TEXT NOT NULL,
                    run_index_path TEXT NOT NULL UNIQUE,
                    external_data_root TEXT NOT NULL,
                    vendor_root TEXT NOT NULL,
                    source TEXT NOT NULL,
                    read_only INTEGER NOT NULL,
                    display_order INTEGER NOT NULL,
                    registered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    enabled INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS registries_project
                    ON registries(project_id, display_order, registry_id);
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


def registry_catalog_path(paths: RuntimePaths) -> Path:
    configured = str(os.environ.get("AUTOPLANNER_RUN_REGISTRY_CATALOG_PATH") or "")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (paths.runtime_root / "run_registry_catalog.sqlite3")
    )


def binding_from_paths(
    paths: RuntimePaths,
    *,
    registry_id: str,
    registry_label: str,
    project_id: str,
    project_label: str,
    case_id: str = "",
    source: str = "explicit",
    read_only: bool = True,
    display_order: int = 0,
) -> RunRegistryBinding:
    return RunRegistryBinding(
        registry_id=registry_id,
        registry_label=registry_label,
        project_id=project_id,
        project_label=project_label,
        case_id=case_id,
        repository_root=paths.repository_root,
        runtime_root=paths.runtime_root,
        runs_root=paths.runs_root,
        artifact_store_root=paths.artifact_store_root,
        run_index_path=paths.run_index_path,
        external_data_root=paths.external_data_root,
        vendor_root=paths.vendor_root,
        source=source,
        read_only=read_only,
        display_order=display_order,
    )


def binding_from_registry_root(
    registry_root: str | os.PathLike[str],
    *,
    registry_id: str = "",
    registry_label: str = "",
    project_id: str,
    project_label: str,
    case_id: str = "",
    repository_root: str | os.PathLike[str] | None = None,
    source: str = "panel",
    display_order: int = 0,
) -> RunRegistryBinding:
    root = Path(registry_root).expanduser().resolve()
    repository = Path(repository_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    runtime = root / "runtime"
    identity = registry_id or registry_id_for_path(runtime / "run_index.sqlite3")
    return RunRegistryBinding(
        registry_id=identity,
        registry_label=registry_label or root.name,
        project_id=project_id,
        project_label=project_label,
        case_id=case_id,
        repository_root=repository,
        runtime_root=runtime,
        runs_root=root / "runs",
        artifact_store_root=root / "artifacts",
        run_index_path=runtime / "run_index.sqlite3",
        external_data_root=root / "external",
        vendor_root=repository / "vendor",
        source=source,
        read_only=True,
        display_order=display_order,
    )


def registry_id_for_path(path: str | os.PathLike[str]) -> str:
    resolved = str(Path(path).expanduser().resolve()).casefold()
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"registry-{digest}"


def _binding_from_row(row: sqlite3.Row) -> RunRegistryBinding:
    return RunRegistryBinding(
        registry_id=row["registry_id"],
        registry_label=row["registry_label"],
        project_id=row["project_id"],
        project_label=row["project_label"],
        case_id=row["case_id"],
        repository_root=Path(row["repository_root"]),
        runtime_root=Path(row["runtime_root"]),
        runs_root=Path(row["runs_root"]),
        artifact_store_root=Path(row["artifact_store_root"]),
        run_index_path=Path(row["run_index_path"]),
        external_data_root=Path(row["external_data_root"]),
        vendor_root=Path(row["vendor_root"]),
        source=row["source"],
        read_only=bool(row["read_only"]),
        display_order=int(row["display_order"]),
        registered_at=row["registered_at"],
        updated_at=row["updated_at"],
        enabled=bool(row["enabled"]),
    )


def _normalize_id(value: Any, *, field: str) -> str:
    identity = str(value or "").strip().casefold()
    if not _REGISTRY_ID.fullmatch(identity):
        raise RunRegistryCatalogError(f"run_registry_{field}_invalid")
    return identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage explicit AutoPlanner run registries")
    parser.add_argument("--catalog-path", default="")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--registry-root", required=True)
    register.add_argument("--registry-id", default="")
    register.add_argument("--registry-label", default="")
    register.add_argument("--project-id", required=True)
    register.add_argument("--project-label", required=True)
    register.add_argument("--case-id", default="")
    register.add_argument("--display-order", type=int, default=0)
    subparsers.add_parser("list")
    args = parser.parse_args(argv)
    primary_paths = RuntimePaths.discover()
    catalog = RunRegistryCatalog(
        Path(args.catalog_path).expanduser().resolve()
        if args.catalog_path
        else registry_catalog_path(primary_paths)
    )
    if args.operation == "register":
        result = catalog.register(
            binding_from_registry_root(
                args.registry_root,
                registry_id=args.registry_id,
                registry_label=args.registry_label,
                project_id=args.project_id,
                project_label=args.project_label,
                case_id=args.case_id,
                repository_root=primary_paths.repository_root,
                display_order=args.display_order,
            )
        )
    else:
        result = {
            "schema_version": RUN_REGISTRY_CATALOG_SCHEMA,
            "operation": "list",
            "registries": [binding.to_dict() for binding in catalog.list_registries()],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUN_REGISTRY_BINDING_SCHEMA",
    "RUN_REGISTRY_CATALOG_SCHEMA",
    "RunRegistryBinding",
    "RunRegistryCatalog",
    "RunRegistryCatalogError",
    "binding_from_paths",
    "binding_from_registry_root",
    "registry_catalog_path",
    "registry_id_for_path",
]
