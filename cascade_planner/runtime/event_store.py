"""Durable append-only storage for child-agent runtime contracts.

Events are authoritative history. State and result JSON files are replaceable
snapshots for fast restart/reconciliation. The implementation uses only the
standard library and does not start or manage agent processes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .contracts import (
    AgentContractError,
    AgentEvent,
    AgentHandle,
    AgentResult,
    AgentState,
    coerce_agent_state,
    utc_now,
)


class EventStoreError(RuntimeError):
    """Base class for durable runtime-store failures."""


class EventStoreCorruptionError(EventStoreError):
    """Raised when durable data violates the runtime contract."""


class EventStoreLockTimeout(EventStoreError, TimeoutError):
    """Raised when another writer holds a run log lock too long."""


class IdempotencyConflictError(EventStoreError):
    """Raised when one idempotency key is reused for different content."""


class EventSequenceConflictError(EventStoreError):
    """Raised when a transition does not continue an agent's event history."""


class SnapshotConflictError(EventStoreError):
    """Raised when a snapshot would overwrite a different attempt result."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentReconciliation:
    """Read-only comparison of event history and fast snapshots."""

    run_id: str
    agent_id: str
    attempt: int
    snapshot_state: AgentState | None
    event_state: AgentState | None
    result_state: AgentState | None
    authoritative_state: AgentState | None
    latest_cursor: int
    issues: tuple[str, ...] = ()

    @property
    def consistent(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "attempt": self.attempt,
            "snapshot_state": self.snapshot_state.value if self.snapshot_state else None,
            "event_state": self.event_state.value if self.event_state else None,
            "result_state": self.result_state.value if self.result_state else None,
            "authoritative_state": (
                self.authoritative_state.value if self.authoritative_state else None
            ),
            "latest_cursor": self.latest_cursor,
            "consistent": self.consistent,
            "issues": list(self.issues),
        }


class EventStore:
    """Filesystem event store partitioned by ``run_id`` and agent attempt.

    Public identifiers are never used directly as path components. A stable
    digest prevents path traversal and keeps Windows-incompatible characters
    out of runtime paths; the original ids remain inside every durable record.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        lock_timeout_s: float = 10.0,
        stale_lock_s: float = 120.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.lock_timeout_s = float(lock_timeout_s)
        self.stale_lock_s = float(stale_lock_s)
        if self.lock_timeout_s <= 0:
            raise ValueError("lock_timeout_s must be > 0")
        if self.stale_lock_s <= 0:
            raise ValueError("stale_lock_s must be > 0")
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Append-only event API
    # ------------------------------------------------------------------
    def append(self, event: AgentEvent) -> AgentEvent:
        """Alias for :meth:`append_event`."""

        return self.append_event(event)

    def append_event(self, event: AgentEvent) -> AgentEvent:
        """Append one event, assigning a monotonically increasing run cursor.

        Replaying the same ``(agent_id, attempt, idempotency_key)`` returns the
        first durable event. Reusing that key for different semantic content is
        rejected instead of silently corrupting the history.
        """

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        run_dir = self._run_dir(event.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        with self._run_write_lock(event.run_id):
            self._repair_event_log_tail(event.run_id)
            existing_events = self._load_run_events(event.run_id)
            for existing in existing_events:
                if existing.event_id == event.event_id:
                    if existing.semantic_dict() != event.semantic_dict():
                        raise IdempotencyConflictError(
                            f"event_id {event.event_id!r} already has different content"
                        )
                    return existing
                if existing.idempotency_scope() == event.idempotency_scope():
                    if existing.semantic_dict() != event.semantic_dict():
                        raise IdempotencyConflictError(
                            "idempotency key reused for different event content: "
                            f"agent={event.agent_id!r}, attempt={event.attempt}, "
                            f"key={event.idempotency_key!r}"
                        )
                    return existing

            attempt_events = [
                existing
                for existing in existing_events
                if existing.agent_id == event.agent_id and existing.attempt == event.attempt
            ]
            context_values = {existing.context_hash for existing in attempt_events}
            if context_values and event.context_hash not in context_values:
                raise EventSequenceConflictError(
                    "context_hash changed within one agent attempt"
                )
            if event.from_state is not None:
                previous_state = next(
                    (
                        existing.to_state
                        for existing in reversed(attempt_events)
                        if existing.to_state is not None
                    ),
                    AgentState.PENDING,
                )
                if previous_state != event.from_state:
                    raise EventSequenceConflictError(
                        "transition does not continue event history: "
                        f"expected from_state={previous_state.value}, "
                        f"got {event.from_state.value}"
                    )

            next_cursor = existing_events[-1].cursor + 1 if existing_events else 1
            if event.cursor is not None and event.cursor != next_cursor:
                raise AgentContractError(
                    f"new event cursor must be {next_cursor}, got {event.cursor}"
                )
            persisted = event.persisted(cursor=next_cursor, recorded_at=utc_now())
            self._append_json_line(self._events_path(event.run_id), persisted.to_dict())
            return persisted

    def read_events(
        self,
        run_id: str,
        *,
        agent_id: str | None = None,
        attempt: int | None = None,
        after_cursor: int = 0,
        limit: int | None = None,
        kinds: set[str] | frozenset[str] | None = None,
    ) -> list[AgentEvent]:
        """Read events in cursor order, optionally filtering one agent."""

        cursor = int(after_cursor)
        if cursor < 0:
            raise ValueError("after_cursor must be >= 0")
        if attempt is not None and int(attempt) < 1:
            raise ValueError("attempt must be >= 1")
        if limit is not None and int(limit) < 0:
            raise ValueError("limit must be >= 0")
        if limit == 0:
            return []
        accepted_kinds = set(kinds) if kinds is not None else None
        result: list[AgentEvent] = []
        for event in self._load_run_events(run_id):
            if event.cursor is None or event.cursor <= cursor:
                continue
            if agent_id is not None and event.agent_id != agent_id:
                continue
            if attempt is not None and event.attempt != int(attempt):
                continue
            if accepted_kinds is not None and event.kind not in accepted_kinds:
                continue
            result.append(event)
            if limit is not None and len(result) >= int(limit):
                break
        return result

    def iter_events(self, run_id: str, **kwargs: Any) -> Iterator[AgentEvent]:
        return iter(self.read_events(run_id, **kwargs))

    def read_agent_events(
        self,
        run_id: str,
        agent_id: str,
        *,
        attempt: int | None = None,
        after_cursor: int = 0,
        limit: int | None = None,
    ) -> list[AgentEvent]:
        return self.read_events(
            run_id,
            agent_id=agent_id,
            attempt=attempt,
            after_cursor=after_cursor,
            limit=limit,
        )

    def last_cursor(self, run_id: str) -> int:
        events = self._load_run_events(run_id)
        return int(events[-1].cursor or 0) if events else 0

    def find_idempotent_event(
        self,
        run_id: str,
        agent_id: str,
        *,
        attempt: int,
        idempotency_key: str,
    ) -> AgentEvent | None:
        scope = (agent_id, int(attempt), str(idempotency_key))
        for event in self._load_run_events(run_id):
            if event.idempotency_scope() == scope:
                return event
        return None

    # ------------------------------------------------------------------
    # Atomic state/result snapshot API
    # ------------------------------------------------------------------
    def write_state(self, handle: AgentHandle) -> AgentHandle:
        if not isinstance(handle, AgentHandle):
            raise TypeError("handle must be an AgentHandle")
        path = self._state_path(handle.run_id, handle.agent_id, handle.attempt)
        with self._run_write_lock(handle.run_id):
            existing = self.read_state(handle.run_id, handle.agent_id, attempt=handle.attempt)
            if existing is not None:
                self._require_same_attempt_identity(existing, handle)
                if handle.last_event_cursor < existing.last_event_cursor:
                    raise SnapshotConflictError("state snapshot cursor cannot move backwards")
                if handle.last_event_cursor == existing.last_event_cursor:
                    if _canonical_json(handle.to_dict()) != _canonical_json(existing.to_dict()):
                        raise SnapshotConflictError(
                            "state snapshot changed without a newer event cursor"
                        )
                    return existing
                if existing.state.is_terminal and handle.state != existing.state:
                    raise SnapshotConflictError("terminal state snapshot cannot be replaced")
            self._atomic_write_json(path, handle.to_dict())
        return handle

    def read_state(
        self,
        run_id: str,
        agent_id: str,
        *,
        attempt: int | None = None,
    ) -> AgentHandle | None:
        selected_attempt = self._select_attempt(run_id, agent_id, attempt, "state.json")
        if selected_attempt is None:
            return None
        path = self._snapshot_read_path(run_id, agent_id, selected_attempt, "state.json")
        raw = self._read_snapshot(path, expected_schema=AgentHandle.schema_version)
        if raw is None:
            return None
        try:
            handle = AgentHandle.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise EventStoreCorruptionError(f"invalid state snapshot {path}: {exc}") from exc
        self._require_requested_identity(run_id, agent_id, selected_attempt, handle, path)
        return handle

    def write_result(self, result: AgentResult) -> AgentResult:
        if not isinstance(result, AgentResult):
            raise TypeError("result must be an AgentResult")
        path = self._result_path(result.run_id, result.agent_id, result.attempt)
        with self._run_write_lock(result.run_id):
            existing = self.read_result(result.run_id, result.agent_id, attempt=result.attempt)
            if existing is not None:
                if existing.idempotency_key == result.idempotency_key:
                    if _canonical_json(existing.semantic_dict()) != _canonical_json(
                        result.semantic_dict()
                    ):
                        raise IdempotencyConflictError(
                            "result idempotency key reused for different content"
                        )
                    return existing
                raise SnapshotConflictError(
                    f"agent attempt already has a terminal result: {result.agent_id!r} "
                    f"attempt {result.attempt}"
                )
            self._atomic_write_json(path, result.to_dict())
        return result

    def read_result(
        self,
        run_id: str,
        agent_id: str,
        *,
        attempt: int | None = None,
    ) -> AgentResult | None:
        selected_attempt = self._select_attempt(run_id, agent_id, attempt, "result.json")
        if selected_attempt is None:
            return None
        path = self._snapshot_read_path(run_id, agent_id, selected_attempt, "result.json")
        raw = self._read_snapshot(path, expected_schema=AgentResult.schema_version)
        if raw is None:
            return None
        try:
            result = AgentResult.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise EventStoreCorruptionError(f"invalid result snapshot {path}: {exc}") from exc
        self._require_requested_identity(run_id, agent_id, selected_attempt, result, path)
        return result

    # ------------------------------------------------------------------
    # Lifecycle and reconciliation conveniences
    # ------------------------------------------------------------------
    def transition(
        self,
        handle: AgentHandle,
        target: AgentState | str,
        *,
        idempotency_key: str,
        kind: str = "agent.state_changed",
        child_agent_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> tuple[AgentHandle, AgentEvent]:
        """Append a transition event and refresh the derived state snapshot.

        A crash between these writes is detectable with :meth:`reconcile_agent`;
        history is written first so no transition can exist only in a snapshot.
        """

        target_state = coerce_agent_state(target)
        if handle.state == target_state:
            existing = self.find_idempotent_event(
                handle.run_id,
                handle.agent_id,
                attempt=handle.attempt,
                idempotency_key=idempotency_key,
            )
            if existing is None or existing.to_state != target_state:
                handle.state.require_transition_to(target_state)
            if (
                existing.kind != kind
                or existing.child_agent_id != child_agent_id
                or existing.payload != dict(payload or {})
            ):
                raise IdempotencyConflictError(
                    "transition idempotency key reused for different content"
                )
            refreshed = handle.with_event_cursor(existing.cursor or handle.last_event_cursor)
            self.write_state(refreshed)
            return refreshed, existing

        event = AgentEvent.for_transition(
            handle,
            target_state,
            idempotency_key=idempotency_key,
            kind=kind,
            child_agent_id=child_agent_id,
            payload=payload,
            event_id=event_id,
        )
        persisted = self.append_event(event)
        transitioned = handle.transition(
            target_state,
            event_cursor=persisted.cursor,
            updated_at=persisted.occurred_at,
        )
        self.write_state(transitioned)
        return transitioned, persisted

    def latest_event_state(
        self,
        run_id: str,
        agent_id: str,
        *,
        attempt: int | None = None,
    ) -> AgentState | None:
        state: AgentState | None = None
        for event in self.read_agent_events(run_id, agent_id, attempt=attempt):
            if event.to_state is not None:
                state = event.to_state
        return state

    def list_agent_ids(self, run_id: str) -> list[str]:
        agent_ids = {event.agent_id for event in self._load_run_events(run_id)}
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            for pattern in ("s-*.json", "o-*.json"):
                for path in run_dir.glob(pattern):
                    try:
                        raw = self._read_snapshot(path, expected_schema=None)
                    except EventStoreCorruptionError:
                        continue
                    if raw and raw.get("run_id") == run_id and raw.get("agent_id"):
                        agent_ids.add(str(raw["agent_id"]))
        # Read pre-flattening stores as well so an interrupted run can be
        # resumed after upgrading the runtime layout.
        legacy_agents_dir = self._legacy_run_dir(run_id) / "agents"
        if legacy_agents_dir.exists():
            for filename in ("state.json", "result.json"):
                for path in legacy_agents_dir.glob(f"*/attempts/*/{filename}"):
                    try:
                        raw = self._read_snapshot(path, expected_schema=None)
                    except EventStoreCorruptionError:
                        continue
                    if raw and raw.get("run_id") == run_id and raw.get("agent_id"):
                        agent_ids.add(str(raw["agent_id"]))
        return sorted(agent_ids)

    def reconcile_agent(
        self,
        run_id: str,
        agent_id: str,
        *,
        attempt: int | None = None,
    ) -> AgentReconciliation:
        selected_attempt = self._latest_known_attempt(run_id, agent_id) if attempt is None else int(attempt)
        if selected_attempt < 1:
            selected_attempt = 1
        handle = self.read_state(run_id, agent_id, attempt=selected_attempt)
        result = self.read_result(run_id, agent_id, attempt=selected_attempt)
        events = self.read_agent_events(run_id, agent_id, attempt=selected_attempt)
        event_state: AgentState | None = None
        for event in events:
            if event.to_state is not None:
                event_state = event.to_state
        snapshot_state = handle.state if handle else None
        result_state = result.state if result else None
        authoritative = event_state or result_state or snapshot_state
        latest_cursor = max((int(event.cursor or 0) for event in events), default=0)
        issues: list[str] = []
        if handle is None and (events or result is not None):
            issues.append("missing_state_snapshot")
        if handle is None and not events and result is None:
            issues.append("agent_attempt_not_found")
        if handle is not None and event_state is not None and handle.state != event_state:
            issues.append("state_snapshot_event_mismatch")
        if result is not None and event_state is None:
            issues.append("result_without_terminal_event")
        if result is not None and event_state is not None and result.state != event_state:
            issues.append("event_result_mismatch")
        if result is not None and handle is not None and result.state != handle.state:
            issues.append("state_snapshot_result_mismatch")
        if result is None and authoritative is not None and authoritative.is_terminal:
            issues.append("terminal_state_missing_result")
        if handle is not None and handle.last_event_cursor < latest_cursor:
            issues.append("state_snapshot_cursor_behind")
        if handle is not None and handle.last_event_cursor > latest_cursor:
            issues.append("state_snapshot_cursor_ahead")
        context_values = {
            item.context_hash
            for item in [handle, result, *events]
            if item is not None
        }
        if len(context_values) > 1:
            issues.append("context_hash_mismatch")
        parent_values = {
            item.parent_agent_id
            for item in [handle, result, *events]
            if item is not None and item.parent_agent_id is not None
        }
        if len(parent_values) > 1:
            issues.append("parent_agent_id_mismatch")
        return AgentReconciliation(
            run_id=run_id,
            agent_id=agent_id,
            attempt=selected_attempt,
            snapshot_state=snapshot_state,
            event_state=event_state,
            result_state=result_state,
            authoritative_state=authoritative,
            latest_cursor=latest_cursor,
            issues=tuple(dict.fromkeys(issues)),
        )

    def reconcile_run(self, run_id: str) -> list[AgentReconciliation]:
        return [self.reconcile_agent(run_id, agent_id) for agent_id in self.list_agent_ids(run_id)]

    # ------------------------------------------------------------------
    # Paths, parsing, and atomic I/O
    # ------------------------------------------------------------------
    def _run_dir(self, run_id: str) -> Path:
        # Runtime roots are commonly nested below a frontier run directory.
        # Keep every additional component compact so atomic temporary snapshot
        # names remain below the legacy Windows MAX_PATH boundary.
        return self.root / f"r-{_id_digest('run', run_id)}"

    def _events_path(self, run_id: str) -> Path:
        current = self._run_dir(run_id) / "events.jsonl"
        legacy = self._legacy_run_dir(run_id) / "events.jsonl"
        return legacy if not current.exists() and legacy.exists() else current

    def _state_path(self, run_id: str, agent_id: str, attempt: int) -> Path:
        return self._snapshot_path(run_id, agent_id, attempt, kind="s")

    def _result_path(self, run_id: str, agent_id: str, attempt: int) -> Path:
        return self._snapshot_path(run_id, agent_id, attempt, kind="o")

    def _snapshot_path(self, run_id: str, agent_id: str, attempt: int, *, kind: str) -> Path:
        value = int(attempt)
        if value < 1:
            raise ValueError("attempt must be >= 1")
        return self._run_dir(run_id) / f"{kind}-{_id_digest('agent', agent_id)}-{value}.json"

    def _legacy_run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / _id_digest("run", run_id)

    def _legacy_attempt_dir(self, run_id: str, agent_id: str, attempt: int) -> Path:
        return (
            self._legacy_run_dir(run_id)
            / "agents"
            / _id_digest("agent", agent_id)
            / "attempts"
            / str(int(attempt))
        )

    def _snapshot_read_path(
        self,
        run_id: str,
        agent_id: str,
        attempt: int,
        filename: str,
    ) -> Path:
        current = (
            self._state_path(run_id, agent_id, attempt)
            if filename == "state.json"
            else self._result_path(run_id, agent_id, attempt)
        )
        if current.exists():
            return current
        return self._legacy_attempt_dir(run_id, agent_id, attempt) / filename

    def _select_attempt(
        self,
        run_id: str,
        agent_id: str,
        attempt: int | None,
        filename: str,
    ) -> int | None:
        kind = "s" if filename == "state.json" else "o"
        if attempt is not None:
            value = int(attempt)
            if value < 1:
                raise ValueError("attempt must be >= 1")
            current = self._snapshot_path(run_id, agent_id, value, kind=kind)
            legacy = self._legacy_attempt_dir(run_id, agent_id, value) / filename
            return value if current.exists() or legacy.exists() else None
        digest = _id_digest("agent", agent_id)
        prefix = f"{kind}-{digest}-"
        values: list[int] = []
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            for path in run_dir.glob(f"{prefix}*.json"):
                token = path.name.removeprefix(prefix).removesuffix(".json")
                if token.isdigit():
                    values.append(int(token))
        attempts_dir = self._legacy_attempt_dir(run_id, agent_id, 1).parent
        if attempts_dir.exists():
            values.extend(
                int(path.name)
                for path in attempts_dir.iterdir()
                if path.is_dir() and path.name.isdigit() and (path / filename).exists()
            )
        return max(values) if values else None

    def _latest_known_attempt(self, run_id: str, agent_id: str) -> int:
        attempts: set[int] = {
            event.attempt for event in self.read_agent_events(run_id, agent_id)
        }
        for filename in ("state.json", "result.json"):
            value = self._select_attempt(run_id, agent_id, None, filename)
            if value is not None:
                attempts.add(value)
        return max(attempts) if attempts else 1

    def _load_run_events(self, run_id: str) -> list[AgentEvent]:
        path = self._events_path(run_id)
        if not path.exists():
            return []
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise EventStoreError(f"cannot read event log {path}: {exc}") from exc
        lines = raw_bytes.splitlines(keepends=True)
        events: list[AgentEvent] = []
        expected_cursor = 1
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                is_truncated_tail = index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
                if is_truncated_tail:
                    break
                raise EventStoreCorruptionError(
                    f"invalid event JSON at {path}:{index + 1}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise EventStoreCorruptionError(
                    f"event at {path}:{index + 1} is not a JSON object"
                )
            if data.get("schema_version") != AgentEvent.schema_version:
                raise EventStoreCorruptionError(
                    f"unsupported event schema at {path}:{index + 1}: "
                    f"{data.get('schema_version')!r}"
                )
            try:
                event = AgentEvent.from_dict(data)
            except (TypeError, ValueError) as exc:
                raise EventStoreCorruptionError(
                    f"invalid event contract at {path}:{index + 1}: {exc}"
                ) from exc
            if event.run_id != run_id:
                raise EventStoreCorruptionError(
                    f"event run_id mismatch at {path}:{index + 1}"
                )
            if event.cursor != expected_cursor:
                raise EventStoreCorruptionError(
                    f"non-contiguous event cursor at {path}:{index + 1}: "
                    f"expected {expected_cursor}, got {event.cursor}"
                )
            events.append(event)
            expected_cursor += 1
        return events

    def _repair_event_log_tail(self, run_id: str) -> None:
        """Repair only a crash-truncated final line while the run lock is held."""

        path = self._events_path(run_id)
        if not path.exists():
            return
        raw = path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        boundary = raw.rfind(b"\n") + 1
        tail = raw[boundary:]
        try:
            decoded = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            repaired = raw[:boundary]
        else:
            if not isinstance(decoded, dict):
                raise EventStoreCorruptionError(
                    f"event log tail in {path} is not a JSON object"
                )
            repaired = raw + b"\n"
        with path.open("r+b") as stream:
            stream.seek(0)
            stream.write(repaired)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def _run_write_lock(self, run_id: str) -> Iterator[None]:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / ".events.lock"
        token = f"{os.getpid()}:{uuid4().hex}:{time.time_ns()}".encode("ascii")
        deadline = time.monotonic() + self.lock_timeout_s
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age_s = max(0.0, time.time() - path.stat().st_mtime)
                    if age_s > self.stale_lock_s and not _lock_owner_alive(path):
                        path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise EventStoreLockTimeout(f"timed out waiting for event log lock {path}")
                time.sleep(0.02)
                continue
            try:
                os.write(fd, token)
                os.fsync(fd)
            finally:
                os.close(fd)
            break
        try:
            yield
        finally:
            try:
                if path.read_bytes() == token:
                    path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

    @staticmethod
    def _append_json_line(path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_json(data).encode("utf-8") + b"\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(path, flags, 0o600)
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(fd, payload[offset:])
                    if written <= 0:
                        raise OSError("short write while appending event")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise EventStoreError(f"cannot append event log {path}: {exc}") from exc

    @staticmethod
    def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (_canonical_json(data) + "\n").encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_snapshot(
        path: Path,
        *,
        expected_schema: str | None,
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EventStoreCorruptionError(f"invalid JSON snapshot {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise EventStoreCorruptionError(f"snapshot {path} is not a JSON object")
        if expected_schema is not None and data.get("schema_version") != expected_schema:
            raise EventStoreCorruptionError(
                f"unsupported snapshot schema in {path}: {data.get('schema_version')!r}"
            )
        return data

    @staticmethod
    def _require_requested_identity(
        run_id: str,
        agent_id: str,
        attempt: int,
        record: AgentHandle | AgentResult,
        path: Path,
    ) -> None:
        if (
            record.run_id != run_id
            or record.agent_id != agent_id
            or record.attempt != int(attempt)
        ):
            raise EventStoreCorruptionError(f"snapshot identity mismatch in {path}")

    @staticmethod
    def _require_same_attempt_identity(
        existing: AgentHandle,
        incoming: AgentHandle,
    ) -> None:
        if (
            existing.run_id != incoming.run_id
            or existing.agent_id != incoming.agent_id
            or existing.attempt != incoming.attempt
            or existing.idempotency_key != incoming.idempotency_key
            or existing.context_hash != incoming.context_hash
            or existing.parent_agent_id != incoming.parent_agent_id
            or existing.role != incoming.role
            or existing.capabilities != incoming.capabilities
            or existing.write_scope != incoming.write_scope
            or existing.budget != incoming.budget
            or existing.created_at != incoming.created_at
        ):
            raise SnapshotConflictError("state snapshot identity/context mismatch")


def _id_digest(namespace: str, value: str) -> str:
    encoded = f"{namespace}\0{value}".encode("utf-8")
    # A 96-bit digest remains collision-resistant for local runtime ids while
    # keeping nested snapshot paths below the legacy Windows MAX_PATH limit.
    return hashlib.sha256(encoded).digest()[:12].hex()


def _lock_owner_alive(path: Path) -> bool:
    try:
        owner_pid = int(path.read_text(encoding="ascii").split(":", 1)[0])
    except (OSError, ValueError):
        return False
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, PermissionError):
        return True
    return True


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


JsonlEventStore = EventStore


__all__ = [
    "AgentReconciliation",
    "EventStore",
    "EventStoreCorruptionError",
    "EventStoreError",
    "EventStoreLockTimeout",
    "EventSequenceConflictError",
    "IdempotencyConflictError",
    "JsonlEventStore",
    "SnapshotConflictError",
]
