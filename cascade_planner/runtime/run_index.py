"""Rebuildable SQLite projection for retrosynthesis runs.

The index is deliberately non-authoritative.  It accelerates status pages,
resume discovery, and operational queries; deleting it and replaying manifests
must reproduce the same rows without changing any scientific artifact.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable, Iterator, Mapping

from .artifact_store import ArtifactRef


RUN_INDEX_SCHEMA = "autoplanner_run_index.v1"
RUN_MANIFEST_SCHEMA = "autoplanner_run_manifest.v1"
_SCHEMA_VERSION = 1


class RunIndexError(RuntimeError):
    """Raised when a run projection cannot satisfy its contract."""


class RunIndex:
    """Small WAL-backed projection with one transaction per public mutation."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        timeout_s: float = 30.0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_s = max(0.1, float(timeout_s))
        self._initialize_lock = threading.Lock()
        self._initialize()

    def upsert_run(self, manifest: Mapping[str, Any]) -> None:
        row = _normalize_run_manifest(manifest)
        with self._transaction() as connection:
            self._upsert_run(connection, row)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_runs(
        self,
        *,
        status: str | None = None,
        case_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(str(status))
        if case_id is not None:
            clauses.append("case_id = ?")
            values.append(str(case_id))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(10_000, int(limit))))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT manifest_json FROM runs"
                + where
                + " ORDER BY updated_at DESC, run_id ASC LIMIT ?",
                values,
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def index_artifact(
        self,
        *,
        run_id: str,
        artifact_id: str,
        ref: ArtifactRef | Mapping[str, Any],
        revision: int = 0,
        authority_scope: str = "operational_projection",
    ) -> None:
        artifact_ref = (
            ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_dict(ref)
        )
        if not str(run_id or "").strip() or not str(artifact_id or "").strip():
            raise RunIndexError("run_index_artifact_identity_missing")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    run_id, artifact_id, revision, sha256, size_bytes,
                    media_type, logical_name, producer, authority_scope,
                    ref_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, artifact_id, revision) DO UPDATE SET
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    media_type = excluded.media_type,
                    logical_name = excluded.logical_name,
                    producer = excluded.producer,
                    authority_scope = excluded.authority_scope,
                    ref_json = excluded.ref_json
                """,
                (
                    str(run_id),
                    str(artifact_id),
                    max(0, int(revision)),
                    artifact_ref.sha256,
                    artifact_ref.size_bytes,
                    artifact_ref.media_type,
                    artifact_ref.logical_name,
                    artifact_ref.producer,
                    str(authority_scope),
                    _canonical_json(artifact_ref.to_dict()),
                ),
            )

    def artifacts_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT artifact_id, revision, authority_scope, ref_json
                FROM artifacts
                WHERE run_id = ?
                ORDER BY revision ASC, artifact_id ASC
                """,
                (str(run_id),),
            ).fetchall()
        return [
            {
                "artifact_id": row[0],
                "revision": int(row[1]),
                "authority_scope": row[2],
                "ref": json.loads(row[3]),
            }
            for row in rows
        ]

    def upsert_task(self, task: Mapping[str, Any]) -> None:
        row = dict(task)
        required = ("run_id", "task_id", "kind", "status", "idempotency_key")
        if any(not str(row.get(key) or "").strip() for key in required):
            raise RunIndexError("run_index_task_identity_missing")
        updated_at = str(row.get("updated_at") or _utc_now())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    run_id, task_id, kind, status, idempotency_key,
                    input_revision, output_sha256, updated_at, task_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, task_id) DO UPDATE SET
                    kind = excluded.kind,
                    status = excluded.status,
                    idempotency_key = excluded.idempotency_key,
                    input_revision = excluded.input_revision,
                    output_sha256 = excluded.output_sha256,
                    updated_at = excluded.updated_at,
                    task_json = excluded.task_json
                """,
                (
                    str(row["run_id"]),
                    str(row["task_id"]),
                    str(row["kind"]),
                    str(row["status"]),
                    str(row["idempotency_key"]),
                    max(0, int(row.get("input_revision") or 0)),
                    str(row.get("output_sha256") or ""),
                    updated_at,
                    _canonical_json(row),
                ),
            )

    def tasks_for_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT task_json FROM tasks WHERE run_id = ?"
        values: list[Any] = [str(run_id)]
        if status is not None:
            query += " AND status = ?"
            values.append(str(status))
        query += " ORDER BY updated_at ASC, task_id ASC"
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [json.loads(row[0]) for row in rows]

    def rebuild(self, manifests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        normalized = [_normalize_run_manifest(row) for row in manifests]
        with self._transaction() as connection:
            connection.execute("DELETE FROM artifacts")
            connection.execute("DELETE FROM tasks")
            connection.execute("DELETE FROM runs")
            for manifest in normalized:
                self._upsert_run(connection, manifest)
                run_id = str(manifest["run_id"])
                for raw in manifest.get("indexed_artifacts") or []:
                    if not isinstance(raw, Mapping):
                        continue
                    artifact = dict(raw)
                    ref = ArtifactRef.from_dict(dict(artifact.get("ref") or {}))
                    connection.execute(
                        """
                        INSERT INTO artifacts (
                            run_id, artifact_id, revision, sha256, size_bytes,
                            media_type, logical_name, producer, authority_scope,
                            ref_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            str(artifact.get("artifact_id") or ref.logical_name),
                            max(0, int(artifact.get("revision") or 0)),
                            ref.sha256,
                            ref.size_bytes,
                            ref.media_type,
                            ref.logical_name,
                            ref.producer,
                            str(
                                artifact.get("authority_scope")
                                or "operational_projection"
                            ),
                            _canonical_json(ref.to_dict()),
                        ),
                    )
                for raw_task in manifest.get("indexed_tasks") or []:
                    if not isinstance(raw_task, Mapping):
                        continue
                    task = dict(raw_task)
                    required = (
                        "task_id",
                        "kind",
                        "status",
                        "idempotency_key",
                    )
                    if any(
                        not str(task.get(key) or "").strip() for key in required
                    ):
                        raise RunIndexError("run_index_task_identity_missing")
                    task["run_id"] = run_id
                    updated_at = str(task.get("updated_at") or _utc_now())
                    connection.execute(
                        """
                        INSERT INTO tasks (
                            run_id, task_id, kind, status, idempotency_key,
                            input_revision, output_sha256, updated_at, task_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            str(task["task_id"]),
                            str(task["kind"]),
                            str(task["status"]),
                            str(task["idempotency_key"]),
                            max(0, int(task.get("input_revision") or 0)),
                            str(task.get("output_sha256") or ""),
                            updated_at,
                            _canonical_json(task),
                        ),
                    )
        return {
            "schema_version": "autoplanner_run_index_rebuild.v1",
            "run_count": len(normalized),
            "health": self.health(),
            "semantics": {
                "index_was_rebuilt_from_manifests": True,
                "scientific_artifacts_were_not_modified": True,
            },
        }

    def rebuild_from_manifest_paths(
        self,
        paths: Iterable[str | os.PathLike[str]],
    ) -> dict[str, Any]:
        manifests: list[dict[str, Any]] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RunIndexError(
                    f"run_manifest_unreadable:{path}:{type(exc).__name__}"
                ) from exc
            if not isinstance(value, dict):
                raise RunIndexError(f"run_manifest_not_object:{path}")
            manifests.append(value)
        return self.rebuild(manifests)

    def health(self) -> dict[str, Any]:
        with self._connection() as connection:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
            run_count = int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
            artifact_count = int(
                connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            )
            task_count = int(
                connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            )
        return {
            "schema_version": RUN_INDEX_SCHEMA,
            "accepted": integrity == "ok" and journal_mode.casefold() == "wal",
            "integrity": integrity,
            "journal_mode": journal_mode,
            "run_count": run_count,
            "artifact_count": artifact_count,
            "task_count": task_count,
            "semantics": {
                "index_is_rebuildable_projection": True,
                "index_grants_no_scientific_authority": True,
            },
        }

    def _initialize(self) -> None:
        with self._initialize_lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    accepted INTEGER,
                    model_invocations INTEGER NOT NULL,
                    attempt_runs INTEGER NOT NULL,
                    accepted_expansions INTEGER NOT NULL,
                    molecule_count INTEGER NOT NULL,
                    hyperedge_count INTEGER NOT NULL,
                    complete_route_count INTEGER NOT NULL,
                    proof_deficit_count INTEGER NOT NULL,
                    stock_deficit_count INTEGER NOT NULL,
                    metrics_sha256 TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_case_updated
                    ON runs(case_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS runs_status_updated
                    ON runs(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    logical_name TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    authority_scope TEXT NOT NULL,
                    ref_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, artifact_id, revision),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS artifacts_digest
                    ON artifacts(sha256);
                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    input_revision INTEGER NOT NULL,
                    output_sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, task_id),
                    UNIQUE (run_id, idempotency_key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS tasks_run_status
                    ON tasks(run_id, status, updated_at);
                """
            )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_s,
            isolation_level=None,
        )
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_s * 1000)}")
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

    @staticmethod
    def _upsert_run(
        connection: sqlite3.Connection,
        manifest: Mapping[str, Any],
    ) -> None:
        metrics = dict(manifest.get("metrics") or {})
        graph = dict(manifest.get("graph") or {})
        deficits = dict(manifest.get("deficits") or {})
        budgets = dict(manifest.get("cost_totals") or {})
        accepted = manifest.get("accepted")
        connection.execute(
            """
            INSERT INTO runs (
                run_id, case_id, target_name, status, revision, updated_at,
                run_dir, state_sha256, accepted, model_invocations,
                attempt_runs, accepted_expansions, molecule_count,
                hyperedge_count, complete_route_count, proof_deficit_count,
                stock_deficit_count, metrics_sha256, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                case_id = excluded.case_id,
                target_name = excluded.target_name,
                status = excluded.status,
                revision = excluded.revision,
                updated_at = excluded.updated_at,
                run_dir = excluded.run_dir,
                state_sha256 = excluded.state_sha256,
                accepted = excluded.accepted,
                model_invocations = excluded.model_invocations,
                attempt_runs = excluded.attempt_runs,
                accepted_expansions = excluded.accepted_expansions,
                molecule_count = excluded.molecule_count,
                hyperedge_count = excluded.hyperedge_count,
                complete_route_count = excluded.complete_route_count,
                proof_deficit_count = excluded.proof_deficit_count,
                stock_deficit_count = excluded.stock_deficit_count,
                metrics_sha256 = excluded.metrics_sha256,
                manifest_json = excluded.manifest_json
            WHERE excluded.revision >= runs.revision
            """,
            (
                str(manifest["run_id"]),
                str(manifest.get("case_id") or ""),
                str(manifest.get("target_name") or ""),
                str(manifest.get("status") or "unknown"),
                max(0, int(manifest.get("revision") or 0)),
                str(manifest.get("updated_at") or _utc_now()),
                str(manifest.get("run_dir") or ""),
                str(manifest.get("state_sha256") or ""),
                None if accepted is None else int(accepted is True),
                max(0, int(budgets.get("model_invocations") or 0)),
                max(0, int(budgets.get("attempt_runs") or 0)),
                max(0, int(budgets.get("accepted_expansions") or 0)),
                max(0, int(graph.get("molecule_count") or 0)),
                max(0, int(graph.get("hyperedge_count") or 0)),
                max(0, int(graph.get("complete_route_count") or 0)),
                max(0, int(deficits.get("proof") or 0)),
                max(0, int(deficits.get("stock") or 0)),
                str(metrics.get("sha256") or ""),
                _canonical_json(manifest),
            ),
        )


def _normalize_run_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    if row.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise RunIndexError("run_manifest_schema_invalid")
    if not str(row.get("run_id") or "").strip():
        raise RunIndexError("run_manifest_id_missing")
    revision = row.get("revision")
    if isinstance(revision, bool):
        raise RunIndexError("run_manifest_revision_invalid")
    try:
        normalized_revision = int(revision or 0)
    except (TypeError, ValueError) as exc:
        raise RunIndexError("run_manifest_revision_invalid") from exc
    if normalized_revision < 0:
        raise RunIndexError("run_manifest_revision_invalid")
    row["revision"] = normalized_revision
    row.setdefault("updated_at", _utc_now())
    row.setdefault(
        "semantics",
        {
            "manifest_is_replay_input_for_operational_index": True,
            "manifest_does_not_grant_scientific_authority": True,
        },
    )
    _canonical_json(row)
    return row


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RunIndexError(
            f"run_index_value_not_canonicalizable:{type(exc).__name__}"
        ) from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "RUN_INDEX_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "RunIndex",
    "RunIndexError",
]
