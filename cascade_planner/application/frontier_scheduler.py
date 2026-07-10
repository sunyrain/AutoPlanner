"""Persistent, stock-first scheduling for retrosynthesis frontiers.

The queue is an application-layer authority: providers may propose work, but
only this service owns leases, retries and the definition of campaign closure.
An empty queue is deliberately *not* treated as proof that a route is closed.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Awaitable, Callable, ClassVar, Iterator, Mapping, Sequence
from uuid import uuid4

from rdkit import Chem, RDLogger

from cascade_planner.providers.contracts import ProviderContext, StockProvider


RDLogger.DisableLog("rdApp.*")


class FrontierQueueError(RuntimeError):
    """Base class for durable frontier queue failures."""


class FrontierIdempotencyConflict(FrontierQueueError):
    """Raised when a stable job identity is reused for different work."""


class FrontierLeaseError(FrontierQueueError):
    """Raised when a mutation is attempted without the current lease."""


class FrontierJobState(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            FrontierJobState.SUCCEEDED,
            FrontierJobState.FAILED,
            FrontierJobState.CANCELLED,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FrontierJob:
    """One durable molecule/proof frontier.

    ``proof_deficit`` is the number of levels missing from the required proof
    threshold. Higher deficits and closure probability are scheduled first;
    estimated cost is a bounded penalty, never a reason to hide a frontier.
    """

    run_id: str
    job_id: str
    idempotency_key: str
    frontier_smiles: str
    frontier_node_id: str
    required_proof_level: int = 2
    proof_deficit: int = 2
    closure_probability: float = 0.5
    diversity_gain: float = 0.0
    estimated_cost_units: float = 0.0
    dependency_ids: tuple[str, ...] = ()
    state: FrontierJobState = FrontierJobState.PENDING
    attempt: int = 0
    max_attempts: int = 3
    available_at: str = ""
    lease_owner: str = ""
    lease_token: str = ""
    lease_expires_at: str = ""
    heartbeat_at: str = ""
    closure_kind: str = ""
    achieved_proof_level: int = 0
    result_ref: str = ""
    failure_reasons: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: ClassVar[str] = "frontier_job.v1"

    def __post_init__(self) -> None:
        canonical = _canonical_smiles(self.frontier_smiles)
        if not self.run_id or not self.job_id or not self.idempotency_key:
            raise ValueError("frontier job identity is required")
        if not canonical or canonical != self.frontier_smiles:
            raise ValueError("frontier_smiles must be canonical isomeric SMILES")
        if not self.frontier_node_id:
            raise ValueError("frontier_node_id is required")
        if not 0 <= self.required_proof_level <= 4:
            raise ValueError("required_proof_level must be in [0, 4]")
        if not 0 <= self.proof_deficit <= 4:
            raise ValueError("proof_deficit must be in [0, 4]")
        for name, value in (
            ("closure_probability", self.closure_probability),
            ("diversity_gain", self.diversity_gain),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.estimated_cost_units) or self.estimated_cost_units < 0:
            raise ValueError("estimated_cost_units must be finite and nonnegative")
        if self.attempt < 0 or self.max_attempts < 1 or self.attempt > self.max_attempts:
            raise ValueError("invalid frontier attempt bounds")
        if self.achieved_proof_level < 0 or self.achieved_proof_level > 4:
            raise ValueError("achieved_proof_level must be in [0, 4]")
        if self.state == FrontierJobState.LEASED and not (
            self.lease_owner and self.lease_token and self.lease_expires_at
        ):
            raise ValueError("leased frontier requires owner, token and expiry")

    @property
    def priority_score(self) -> float:
        cost_penalty = min(math.log1p(self.estimated_cost_units), 5.0)
        return round(
            4.0 * self.proof_deficit
            + 3.0 * self.closure_probability
            + 2.0 * self.diversity_gain
            - cost_penalty,
            8,
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["state"] = self.state.value
        row["dependency_ids"] = list(self.dependency_ids)
        row["failure_reasons"] = list(self.failure_reasons)
        row["metadata"] = dict(self.metadata)
        row["schema_version"] = self.schema_version
        row["priority_score"] = self.priority_score
        return row

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrontierJob":
        row = dict(value)
        row.pop("schema_version", None)
        row.pop("priority_score", None)
        row["state"] = FrontierJobState(str(row.get("state") or "pending"))
        row["dependency_ids"] = tuple(str(item) for item in row.get("dependency_ids") or [])
        row["failure_reasons"] = tuple(str(item) for item in row.get("failure_reasons") or [])
        row["metadata"] = dict(row.get("metadata") or {})
        return cls(**row)


@dataclass(frozen=True, slots=True, kw_only=True)
class FrontierCompletenessReport:
    complete: bool
    terminal_count: int
    closed_count: int
    stock_closed_count: int
    reaction_closed_count: int
    unresolved_frontiers: tuple[Mapping[str, Any], ...]
    schema_version: ClassVar[str] = "frontier_completeness.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "complete": self.complete,
            "terminal_count": self.terminal_count,
            "closed_count": self.closed_count,
            "stock_closed_count": self.stock_closed_count,
            "reaction_closed_count": self.reaction_closed_count,
            "unresolved_frontiers": [dict(row) for row in self.unresolved_frontiers],
            "completion_rule": (
                "every_terminal_is_stock_or_reaction_validated_and_no_open_proof_frontier"
            ),
            "queue_empty_is_not_completion": True,
        }


class PersistentFrontierQueue:
    """Atomic JSON snapshot queue with leases and crash recovery."""

    schema_version = "frontier_queue.v1"

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        lock_timeout_s: float = 5.0,
        stale_lock_s: float = 120.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_s = float(lock_timeout_s)
        self.stale_lock_s = float(stale_lock_s)
        if self.lock_timeout_s <= 0 or self.stale_lock_s <= 0:
            raise ValueError("queue lock timeouts must be positive")

    def enqueue(self, job: FrontierJob) -> FrontierJob:
        with self._locked(job.run_id):
            state = self._read(job.run_id)
            by_id = {row.job_id: row for row in state["jobs"]}
            by_key = {row.idempotency_key: row for row in state["jobs"]}
            existing = by_id.get(job.job_id) or by_key.get(job.idempotency_key)
            if existing is not None:
                if _job_semantics(existing) != _job_semantics(job):
                    raise FrontierIdempotencyConflict(
                        "frontier job identity reused for different semantic work"
                    )
                return existing
            if any(dep == job.job_id for dep in job.dependency_ids):
                raise ValueError("frontier job cannot depend on itself")
            state["jobs"].append(job)
            self._write(job.run_id, state)
        return job

    def get(self, run_id: str, job_id: str) -> FrontierJob | None:
        return next((row for row in self._read(run_id)["jobs"] if row.job_id == job_id), None)

    def list_jobs(
        self,
        run_id: str,
        *,
        states: set[FrontierJobState] | None = None,
    ) -> list[FrontierJob]:
        rows = self._read(run_id)["jobs"]
        if states is not None:
            rows = [row for row in rows if row.state in states]
        return sorted(rows, key=lambda row: (row.created_at, row.job_id))

    def claim(
        self,
        run_id: str,
        *,
        worker_id: str,
        limit: int = 1,
        lease_seconds: float = 300.0,
        now: str | None = None,
    ) -> list[FrontierJob]:
        if not worker_id or limit < 1 or lease_seconds <= 0:
            raise ValueError("valid worker_id, limit and lease_seconds are required")
        clock = _coerce_time(now)
        with self._locked(run_id):
            state = self._read(run_id)
            state["jobs"] = self._recover_rows(
                state["jobs"], now=clock, retry_base_seconds=0.0
            )
            by_id = {row.job_id: row for row in state["jobs"]}
            ready = [
                row
                for row in state["jobs"]
                if row.state in {FrontierJobState.PENDING, FrontierJobState.RETRY_WAIT}
                and (not row.available_at or _parse_time(row.available_at) <= clock)
                and all(
                    by_id.get(dep) is not None
                    and by_id[dep].state == FrontierJobState.SUCCEEDED
                    for dep in row.dependency_ids
                )
            ]
            ready.sort(
                key=lambda row: (
                    -row.priority_score,
                    row.attempt,
                    row.created_at,
                    row.job_id,
                )
            )
            selected_ids = {row.job_id for row in ready[:limit]}
            claimed: list[FrontierJob] = []
            lease_expiry = _format_time(clock + timedelta(seconds=float(lease_seconds)))
            updated: list[FrontierJob] = []
            for row in state["jobs"]:
                if row.job_id not in selected_ids:
                    updated.append(row)
                    continue
                leased = replace(
                    row,
                    state=FrontierJobState.LEASED,
                    attempt=row.attempt + 1,
                    lease_owner=worker_id,
                    lease_token=uuid4().hex,
                    lease_expires_at=lease_expiry,
                    heartbeat_at=_format_time(clock),
                    updated_at=_format_time(clock),
                )
                updated.append(leased)
                claimed.append(leased)
            state["jobs"] = updated
            self._write(run_id, state)
        claimed.sort(key=lambda row: ready.index(next(item for item in ready if item.job_id == row.job_id)))
        return claimed

    def heartbeat(
        self,
        run_id: str,
        job_id: str,
        *,
        lease_token: str,
        extend_seconds: float = 300.0,
        now: str | None = None,
    ) -> FrontierJob:
        if extend_seconds <= 0:
            raise ValueError("extend_seconds must be positive")
        clock = _coerce_time(now)
        return self._mutate_leased(
            run_id,
            job_id,
            lease_token=lease_token,
            transform=lambda row: replace(
                row,
                heartbeat_at=_format_time(clock),
                lease_expires_at=_format_time(clock + timedelta(seconds=extend_seconds)),
                updated_at=_format_time(clock),
            ),
        )

    def complete(
        self,
        run_id: str,
        job_id: str,
        *,
        lease_token: str,
        result_ref: str,
        closure_kind: str = "reaction_route",
        achieved_proof_level: int = 2,
        now: str | None = None,
    ) -> FrontierJob:
        if not result_ref:
            raise ValueError("result_ref is required")
        if closure_kind not in {
            "proposal_expansion",
            "reaction_route",
            "stock_boundary",
            "verified_precedent",
        }:
            raise ValueError("unsupported closure_kind")
        clock = _coerce_time(now)

        def transform(row: FrontierJob) -> FrontierJob:
            return replace(
                row,
                state=FrontierJobState.SUCCEEDED,
                lease_owner="",
                lease_token="",
                lease_expires_at="",
                heartbeat_at="",
                closure_kind=closure_kind,
                achieved_proof_level=int(achieved_proof_level),
                result_ref=result_ref,
                updated_at=_format_time(clock),
            )

        return self._mutate_leased(
            run_id,
            job_id,
            lease_token=lease_token,
            transform=transform,
            idempotent_terminal=(result_ref, closure_kind, int(achieved_proof_level)),
        )

    def fail(
        self,
        run_id: str,
        job_id: str,
        *,
        lease_token: str,
        reason: str,
        retryable: bool = True,
        retry_base_seconds: float = 30.0,
        retry_max_seconds: float = 3600.0,
        now: str | None = None,
    ) -> FrontierJob:
        if not reason or retry_base_seconds < 0 or retry_max_seconds < retry_base_seconds:
            raise ValueError("valid failure reason and retry bounds are required")
        clock = _coerce_time(now)

        def transform(row: FrontierJob) -> FrontierJob:
            retry = retryable and row.attempt < row.max_attempts
            delay = min(retry_base_seconds * (2 ** max(row.attempt - 1, 0)), retry_max_seconds)
            return replace(
                row,
                state=FrontierJobState.RETRY_WAIT if retry else FrontierJobState.FAILED,
                available_at=_format_time(clock + timedelta(seconds=delay)) if retry else "",
                lease_owner="",
                lease_token="",
                lease_expires_at="",
                heartbeat_at="",
                failure_reasons=tuple([*row.failure_reasons, reason]),
                updated_at=_format_time(clock),
            )

        return self._mutate_leased(
            run_id,
            job_id,
            lease_token=lease_token,
            transform=transform,
        )

    def recover_expired(
        self,
        run_id: str,
        *,
        retry_base_seconds: float = 30.0,
        now: str | None = None,
    ) -> list[FrontierJob]:
        clock = _coerce_time(now)
        with self._locked(run_id):
            state = self._read(run_id)
            before = {row.job_id: row for row in state["jobs"]}
            state["jobs"] = self._recover_rows(
                state["jobs"], now=clock, retry_base_seconds=retry_base_seconds
            )
            changed = [
                row for row in state["jobs"] if before.get(row.job_id) != row
            ]
            if changed:
                self._write(run_id, state)
            return changed

    def invalidate_succeeded_result(
        self,
        run_id: str,
        job_id: str,
        *,
        expected_result_ref: str,
        reason: str,
        now: str | None = None,
    ) -> FrontierJob:
        """Fail closed when a terminal result artifact cannot be replayed."""

        if not expected_result_ref or not reason:
            raise ValueError("expected_result_ref and reason are required")
        clock = _coerce_time(now)
        with self._locked(run_id):
            state = self._read(run_id)
            found: FrontierJob | None = None
            updated: list[FrontierJob] = []
            for row in state["jobs"]:
                if row.job_id != job_id:
                    updated.append(row)
                    continue
                if row.state != FrontierJobState.SUCCEEDED:
                    raise FrontierQueueError("only a succeeded frontier result can be invalidated")
                if row.result_ref != expected_result_ref:
                    raise FrontierQueueError("frontier result changed before invalidation")
                retry = row.attempt < row.max_attempts
                found = replace(
                    row,
                    state=FrontierJobState.RETRY_WAIT if retry else FrontierJobState.FAILED,
                    available_at=_format_time(clock) if retry else "",
                    closure_kind="",
                    achieved_proof_level=0,
                    result_ref="",
                    failure_reasons=tuple([*row.failure_reasons, reason]),
                    updated_at=_format_time(clock),
                )
                updated.append(found)
            if found is None:
                raise KeyError(f"unknown frontier job: {job_id}")
            state["jobs"] = updated
            self._write(run_id, state)
            return found

    def rebind_succeeded_result(
        self,
        run_id: str,
        job_id: str,
        *,
        expected_result_ref: str,
        result_ref: str,
        metadata_updates: Mapping[str, Any] | None = None,
        now: str | None = None,
    ) -> FrontierJob:
        """Migrate a valid terminal result to an immutable commit reference."""

        if not expected_result_ref or not result_ref:
            raise ValueError("old and new result references are required")
        clock = _coerce_time(now)
        with self._locked(run_id):
            state = self._read(run_id)
            found: FrontierJob | None = None
            updated: list[FrontierJob] = []
            for row in state["jobs"]:
                if row.job_id != job_id:
                    updated.append(row)
                    continue
                if row.state != FrontierJobState.SUCCEEDED:
                    raise FrontierQueueError("only a succeeded frontier result can be rebound")
                if row.result_ref == result_ref:
                    return row
                if row.result_ref != expected_result_ref:
                    raise FrontierQueueError("frontier result changed before rebind")
                found = replace(
                    row,
                    result_ref=result_ref,
                    metadata={**dict(row.metadata), **_json_value(dict(metadata_updates or {}))},
                    updated_at=_format_time(clock),
                )
                updated.append(found)
            if found is None:
                raise KeyError(f"unknown frontier job: {job_id}")
            state["jobs"] = updated
            self._write(run_id, state)
            return found

    def snapshot(self, run_id: str) -> dict[str, Any]:
        state = self._read(run_id)
        rows = [job.to_dict() for job in state["jobs"]]
        payload = {
            "schema_version": self.schema_version,
            "run_id": run_id,
            "revision": state["revision"],
            "jobs": rows,
        }
        payload["content_sha256"] = _digest(payload)
        return payload

    def _mutate_leased(
        self,
        run_id: str,
        job_id: str,
        *,
        lease_token: str,
        transform: Callable[[FrontierJob], FrontierJob],
        idempotent_terminal: tuple[str, str, int] | None = None,
    ) -> FrontierJob:
        with self._locked(run_id):
            state = self._read(run_id)
            found: FrontierJob | None = None
            updated: list[FrontierJob] = []
            for row in state["jobs"]:
                if row.job_id != job_id:
                    updated.append(row)
                    continue
                if idempotent_terminal and row.state == FrontierJobState.SUCCEEDED:
                    expected = (row.result_ref, row.closure_kind, row.achieved_proof_level)
                    if expected == idempotent_terminal:
                        return row
                if row.state != FrontierJobState.LEASED or row.lease_token != lease_token:
                    raise FrontierLeaseError("frontier lease is absent, stale or owned elsewhere")
                found = transform(row)
                updated.append(found)
            if found is None:
                raise KeyError(f"unknown frontier job: {job_id}")
            state["jobs"] = updated
            self._write(run_id, state)
            return found

    def _recover_rows(
        self,
        rows: Sequence[FrontierJob],
        *,
        now: datetime,
        retry_base_seconds: float,
    ) -> list[FrontierJob]:
        result: list[FrontierJob] = []
        for row in rows:
            if (
                row.state != FrontierJobState.LEASED
                or not row.lease_expires_at
                or _parse_time(row.lease_expires_at) > now
            ):
                result.append(row)
                continue
            retry = row.attempt < row.max_attempts
            result.append(
                replace(
                    row,
                    state=FrontierJobState.RETRY_WAIT if retry else FrontierJobState.FAILED,
                    available_at=(
                        _format_time(now + timedelta(seconds=retry_base_seconds))
                        if retry
                        else ""
                    ),
                    lease_owner="",
                    lease_token="",
                    lease_expires_at="",
                    heartbeat_at="",
                    failure_reasons=tuple([*row.failure_reasons, "lease_expired"]),
                    updated_at=_format_time(now),
                )
            )
        return result

    def _read(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id)
        if not path.exists():
            return {"revision": 0, "jobs": []}
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrontierQueueError(f"invalid frontier queue snapshot: {path}") from exc
        expected_digest = str(row.get("content_sha256") or "")
        digest_payload = dict(row)
        digest_payload.pop("content_sha256", None)
        if not expected_digest or expected_digest != _digest(digest_payload):
            raise FrontierQueueError("frontier queue content digest mismatch")
        if row.get("schema_version") != self.schema_version or row.get("run_id") != run_id:
            raise FrontierQueueError("frontier queue identity or schema mismatch")
        return {
            "revision": int(row.get("revision") or 0),
            "jobs": [FrontierJob.from_dict(item) for item in row.get("jobs") or []],
        }

    def _write(self, run_id: str, state: Mapping[str, Any]) -> None:
        path = self._path(run_id)
        revision = int(state.get("revision") or 0) + 1
        payload = {
            "schema_version": self.schema_version,
            "run_id": run_id,
            "revision": revision,
            "jobs": [row.to_dict() for row in state.get("jobs") or []],
        }
        payload["content_sha256"] = _digest(payload)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @contextmanager
    def _locked(self, run_id: str) -> Iterator[None]:
        lock = self._path(run_id).with_suffix(".lock")
        deadline = time.monotonic() + self.lock_timeout_s
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            except FileExistsError:
                try:
                    stale = time.time() - lock.stat().st_mtime > self.stale_lock_s
                except FileNotFoundError:
                    continue
                if stale:
                    try:
                        lock.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise FrontierQueueError(f"frontier queue lock timed out: {lock}")
                time.sleep(0.01)
        try:
            yield
        finally:
            os.close(fd)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def _path(self, run_id: str) -> Path:
        if not str(run_id or "").strip():
            raise ValueError("run_id is required")
        return self.root / f"frontiers-{hashlib.sha256(run_id.encode()).hexdigest()[:24]}.json"


class FrontierScheduler:
    """Submit frontiers through a stock-first terminal audit."""

    def __init__(self, queue: PersistentFrontierQueue, stock_provider: StockProvider) -> None:
        self.queue = queue
        self.stock_provider = stock_provider

    def submit(
        self,
        *,
        run_id: str,
        case_id: str,
        frontier_smiles: str,
        frontier_node_id: str,
        idempotency_key: str,
        stock_request: Mapping[str, Any] | None = None,
        required_proof_level: int = 2,
        proof_deficit: int | None = None,
        closure_probability: float = 0.5,
        diversity_gain: float = 0.0,
        estimated_cost_units: float = 0.0,
        dependency_ids: Sequence[str] = (),
        max_attempts: int = 3,
        metadata: Mapping[str, Any] | None = None,
        now: str | None = None,
    ) -> FrontierJob:
        canonical = _canonical_smiles(frontier_smiles)
        if not canonical:
            raise ValueError("frontier_smiles is invalid")
        clock = _format_time(_coerce_time(now))
        context = ProviderContext(
            run_id=run_id,
            case_id=case_id,
            target_smiles=canonical,
        )
        stock_result = self.stock_provider.invoke(
            {"smiles": canonical, **dict(stock_request or {})},
            context=context,
        )
        stock_closed = bool(stock_result.accepted and stock_result.payload.get("accepted") is True)
        semantic = {
            "run_id": run_id,
            "frontier_smiles": canonical,
            "frontier_node_id": frontier_node_id,
            "required_proof_level": required_proof_level,
            "dependency_ids": sorted(str(item) for item in dependency_ids),
        }
        job = FrontierJob(
            run_id=run_id,
            job_id=f"frontier:{_digest(semantic)[:24]}",
            idempotency_key=idempotency_key,
            frontier_smiles=canonical,
            frontier_node_id=frontier_node_id,
            required_proof_level=required_proof_level,
            proof_deficit=(
                max(required_proof_level, 0) if proof_deficit is None else proof_deficit
            ),
            closure_probability=closure_probability,
            diversity_gain=diversity_gain,
            estimated_cost_units=estimated_cost_units,
            dependency_ids=tuple(sorted(set(str(item) for item in dependency_ids))),
            state=FrontierJobState.SUCCEEDED if stock_closed else FrontierJobState.PENDING,
            max_attempts=max_attempts,
            closure_kind="stock_boundary" if stock_closed else "",
            achieved_proof_level=4 if stock_closed else 0,
            result_ref=(
                f"provider-result:sha256:{stock_result.content_hash}" if stock_closed else ""
            ),
            failure_reasons=tuple(stock_result.reasons) if not stock_closed else (),
            created_at=clock,
            updated_at=clock,
            metadata={
                "stock_audit": _json_value(stock_result.to_dict()),
                "stock_audit_preceded_agent_work": True,
                **_json_value(dict(metadata or {})),
            },
        )
        return self.queue.enqueue(job)


FrontierHandler = Callable[[FrontierJob], Awaitable[Mapping[str, Any]]]


class FrontierExecutor:
    """Bounded async executor over durable queue leases."""

    def __init__(
        self,
        queue: PersistentFrontierQueue,
        *,
        worker_id: str,
        max_concurrency: int = 4,
        lease_seconds: float = 300.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.queue = queue
        self.worker_id = worker_id
        self.max_concurrency = int(max_concurrency)
        self.lease_seconds = float(lease_seconds)

    async def run_ready(self, run_id: str, handler: FrontierHandler) -> list[FrontierJob]:
        claimed = self.queue.claim(
            run_id,
            worker_id=self.worker_id,
            limit=self.max_concurrency,
            lease_seconds=self.lease_seconds,
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute(job: FrontierJob) -> FrontierJob:
            async with semaphore:
                stop_heartbeat = asyncio.Event()
                lease_lost = asyncio.Event()

                async def heartbeat_loop() -> None:
                    interval = max(0.02, min(self.lease_seconds / 3.0, 60.0))
                    while True:
                        try:
                            await asyncio.wait_for(stop_heartbeat.wait(), timeout=interval)
                            return
                        except TimeoutError:
                            pass
                        try:
                            await asyncio.to_thread(
                                self.queue.heartbeat,
                                run_id,
                                job.job_id,
                                lease_token=job.lease_token,
                                extend_seconds=self.lease_seconds,
                            )
                        except (FrontierLeaseError, KeyError, FrontierQueueError):
                            lease_lost.set()
                            return

                heartbeat_task = asyncio.create_task(heartbeat_loop())
                try:
                    result = dict(await handler(job))
                    if lease_lost.is_set():
                        current = self.queue.get(run_id, job.job_id)
                        if current is None:
                            raise FrontierLeaseError("frontier disappeared after lease loss")
                        return current
                    return self.queue.complete(
                        run_id,
                        job.job_id,
                        lease_token=job.lease_token,
                        result_ref=str(result.get("result_ref") or ""),
                        closure_kind=str(result.get("closure_kind") or "reaction_route"),
                        achieved_proof_level=int(result.get("achieved_proof_level") or 0),
                    )
                except FrontierLeaseError:
                    current = self.queue.get(run_id, job.job_id)
                    if current is None:
                        raise
                    return current
                except Exception as exc:  # noqa: BLE001 - failure must become durable state
                    if lease_lost.is_set():
                        current = self.queue.get(run_id, job.job_id)
                        if current is None:
                            raise FrontierLeaseError(
                                "frontier disappeared after lease loss"
                            ) from exc
                        return current
                    try:
                        return self.queue.fail(
                            run_id,
                            job.job_id,
                            lease_token=job.lease_token,
                            reason=f"{type(exc).__name__}:{exc}",
                            retryable=True,
                        )
                    except FrontierLeaseError:
                        current = self.queue.get(run_id, job.job_id)
                        if current is None:
                            raise
                        return current
                finally:
                    stop_heartbeat.set()
                    await heartbeat_task

        return list(await asyncio.gather(*(execute(job) for job in claimed)))


def assess_frontier_completeness(
    terminal_smiles: Sequence[str],
    jobs: Sequence[FrontierJob],
    *,
    open_proof_frontiers: Sequence[Mapping[str, Any] | str] = (),
    required_proof_level: int = 2,
) -> FrontierCompletenessReport:
    """Assess graph closure independently of queue occupancy."""

    terminals = list(dict.fromkeys(_canonical_smiles(item) for item in terminal_smiles))
    terminals = [item for item in terminals if item]
    by_smiles: dict[str, list[FrontierJob]] = {}
    for job in jobs:
        by_smiles.setdefault(job.frontier_smiles, []).append(job)
    unresolved: list[Mapping[str, Any]] = []
    stock_closed = 0
    reaction_closed = 0
    for smiles in terminals:
        candidates = by_smiles.get(smiles, [])
        stock = any(
            row.state == FrontierJobState.SUCCEEDED and row.closure_kind == "stock_boundary"
            for row in candidates
        )
        reaction = any(
            row.state == FrontierJobState.SUCCEEDED
            and row.closure_kind in {"reaction_route", "verified_precedent"}
            and row.achieved_proof_level >= required_proof_level
            for row in candidates
        )
        if stock:
            stock_closed += 1
        elif reaction:
            reaction_closed += 1
        else:
            unresolved.append(
                {
                    "canonical_smiles": smiles,
                    "reason": (
                        "no_frontier_job" if not candidates else "terminal_not_proof_closed"
                    ),
                    "job_states": sorted({row.state.value for row in candidates}),
                    "best_proof_level": max(
                        (row.achieved_proof_level for row in candidates), default=0
                    ),
                }
            )
    for frontier in open_proof_frontiers:
        row = dict(frontier) if isinstance(frontier, Mapping) else {"frontier": str(frontier)}
        row.setdefault("reason", "open_proof_frontier")
        unresolved.append(row)
    closed = stock_closed + reaction_closed
    return FrontierCompletenessReport(
        complete=bool(terminals) and closed == len(terminals) and not unresolved,
        terminal_count=len(terminals),
        closed_count=closed,
        stock_closed_count=stock_closed,
        reaction_closed_count=reaction_closed,
        unresolved_frontiers=tuple(unresolved),
    )


def _job_semantics(job: FrontierJob) -> dict[str, Any]:
    return {
        "run_id": job.run_id,
        "job_id": job.job_id,
        "idempotency_key": job.idempotency_key,
        "frontier_smiles": job.frontier_smiles,
        "frontier_node_id": job.frontier_node_id,
        "required_proof_level": job.required_proof_level,
        "proof_deficit": job.proof_deficit,
        "closure_probability": job.closure_probability,
        "diversity_gain": job.diversity_gain,
        "estimated_cost_units": job.estimated_cost_units,
        "dependency_ids": list(job.dependency_ids),
        "max_attempts": job.max_attempts,
    }


def _canonical_smiles(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    """Return the canonical JSON representation used by durable snapshots."""

    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    )


def _coerce_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return _parse_time(value)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
