"""Deterministic worker boundary for one :class:`RunKernel` campaign.

Workers are deliberately stateless.  A command identifies immutable input and
dependency revisions; an outcome contains proposed facts, never search state.
The run kernel remains the only owner of reservations, attempts, budgets, and
accepted-expansion accounting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import threading
import time
from typing import Any, Callable, Mapping

from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.runtime.artifact_store import ArtifactRef, ArtifactStore


WORKER_COMMAND_SCHEMA = "autoplanner_worker_command.v1"
WORKER_BUDGET_SCHEMA = "autoplanner_worker_budget.v1"
WORKER_OUTCOME_SCHEMA = "autoplanner_worker_outcome.v1"
WORKER_RESULT_SCHEMA = "autoplanner_worker_result.v1"
WORKER_BATCH_SCHEMA = "autoplanner_worker_batch.v1"
_TASK_KINDS = {"model", "evidence", "stock", "validation", "proposal", "other"}
_OUTCOME_STATUSES = {
    "completed",
    "rejected",
    "partial",
    "failed",
    "timed_out",
    "stale",
}


class WorkerRuntimeError(RuntimeError):
    """The worker envelope or execution contract is invalid."""


@dataclass(frozen=True, slots=True)
class WorkerBudget:
    task_kind: str
    timeout_s: float = 60.0
    uses_model: bool = False
    max_output_bytes: int = 2_000_000
    schema_version: str = WORKER_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        if self.task_kind not in _TASK_KINDS:
            raise ValueError(f"unsupported worker task kind:{self.task_kind}")
        if not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("worker timeout must be positive and finite")
        if int(self.max_output_bytes) <= 0:
            raise ValueError("worker output byte limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkerBudget":
        row = dict(value)
        if row.get("schema_version", WORKER_BUDGET_SCHEMA) != WORKER_BUDGET_SCHEMA:
            raise WorkerRuntimeError("worker_budget_schema_invalid")
        return cls(
            task_kind=str(row.get("task_kind") or ""),
            timeout_s=float(row.get("timeout_s") or 0.0),
            uses_model=row.get("uses_model") is True,
            max_output_bytes=int(row.get("max_output_bytes") or 0),
        )


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    command_id: str
    run_id: str
    worker_type: str
    input_revision: int
    idempotency_key: str
    payload: Mapping[str, Any]
    budget: WorkerBudget
    dependency_revisions: Mapping[str, str | int] = field(default_factory=dict)
    artifact_refs: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = WORKER_COMMAND_SCHEMA

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (
                self.command_id,
                self.run_id,
                self.worker_type,
                self.idempotency_key,
            )
        ):
            raise ValueError("worker command identity fields are required")
        if int(self.input_revision) < 0:
            raise ValueError("worker command input revision cannot be negative")
        _canonical_bytes(dict(self.payload))
        _canonical_bytes(dict(self.dependency_revisions))
        for value in self.artifact_refs:
            ArtifactRef.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "command_id": self.command_id,
            "run_id": self.run_id,
            "worker_type": self.worker_type,
            "input_revision": int(self.input_revision),
            "idempotency_key": self.idempotency_key,
            "payload": _json_value(dict(self.payload)),
            "budget": self.budget.to_dict(),
            "dependency_revisions": _json_value(dict(self.dependency_revisions)),
            "artifact_refs": [_json_value(dict(value)) for value in self.artifact_refs],
            "semantics": {
                "immutable_input": True,
                "worker_cannot_mutate_run_state": True,
                "kernel_owns_budget": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row

    @property
    def content_sha256(self) -> str:
        return str(self.to_dict()["content_sha256"])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkerCommand":
        row = dict(value)
        supplied = str(row.pop("content_sha256", ""))
        if row.get("schema_version") != WORKER_COMMAND_SCHEMA:
            raise WorkerRuntimeError("worker_command_schema_invalid")
        if not supplied or supplied != _digest(row):
            raise WorkerRuntimeError("worker_command_digest_invalid")
        return cls(
            command_id=str(row.get("command_id") or ""),
            run_id=str(row.get("run_id") or ""),
            worker_type=str(row.get("worker_type") or ""),
            input_revision=int(row.get("input_revision") or 0),
            idempotency_key=str(row.get("idempotency_key") or ""),
            payload=dict(row.get("payload") or {}),
            budget=WorkerBudget.from_dict(dict(row.get("budget") or {})),
            dependency_revisions=dict(row.get("dependency_revisions") or {}),
            artifact_refs=tuple(
                dict(item)
                for item in row.get("artifact_refs") or []
                if isinstance(item, Mapping)
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkerHandlerSpec:
    worker_type: str
    version: str
    task_kind: str
    handler: Callable[[WorkerCommand, "WorkerArtifactReader"], Mapping[str, Any]]
    model_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.worker_type or not self.version or self.task_kind not in _TASK_KINDS:
            raise ValueError("worker handler specification is invalid")


@dataclass(frozen=True, slots=True)
class WorkerArtifactReader:
    """Read-only, capability-scoped access to command-bound artifacts."""

    store: ArtifactStore
    allowed_refs: Mapping[str, ArtifactRef]
    authority_scopes: Mapping[str, str]

    @classmethod
    def for_command(
        cls,
        store: ArtifactStore,
        command: WorkerCommand,
        artifact_authorities: Mapping[str, str],
    ) -> "WorkerArtifactReader":
        refs = {
            ref.sha256: ref
            for value in command.artifact_refs
            if (ref := ArtifactRef.from_dict(value))
        }
        return cls(
            store=store,
            allowed_refs=refs,
            authority_scopes={
                digest: str(artifact_authorities.get(digest) or "")
                for digest in refs
            },
        )

    def read_json(
        self,
        sha256: str,
        *,
        required_authority_scope: str = "",
    ) -> Any:
        digest = str(sha256 or "").lower()
        ref = self.allowed_refs.get(digest)
        if ref is None:
            raise WorkerRuntimeError("worker_artifact_not_bound_to_command")
        if required_authority_scope and self.authority_scopes.get(digest) != str(
            required_authority_scope
        ):
            raise WorkerRuntimeError("worker_artifact_authority_scope_missing")
        if ref.media_type != "application/json":
            raise WorkerRuntimeError("worker_artifact_media_type_not_json")
        return self.store.read_json(ref)


@dataclass(frozen=True, slots=True)
class WorkerResult:
    command_id: str
    command_sha256: str
    worker_type: str
    status: str
    cache_hit: bool
    outcome_ref: Mapping[str, Any]
    payload: Mapping[str, Any]
    failure_reasons: tuple[str, ...] = ()
    scheduled_commands: tuple[Mapping[str, Any], ...] = ()
    material_events: tuple[str, ...] = ()
    schema_version: str = WORKER_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in _OUTCOME_STATUSES:
            raise ValueError(f"worker result status invalid:{self.status}")
        ArtifactRef.from_dict(self.outcome_ref)

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "command_id": self.command_id,
            "command_sha256": self.command_sha256,
            "worker_type": self.worker_type,
            "status": self.status,
            "cache_hit": self.cache_hit,
            "outcome_ref": dict(self.outcome_ref),
            "payload": _json_value(dict(self.payload)),
            "failure_reasons": list(self.failure_reasons),
            "scheduled_commands": [dict(value) for value in self.scheduled_commands],
            "material_events": list(self.material_events),
            "semantics": {
                "result_is_execution_receipt": True,
                "outcome_artifact_is_immutable": True,
                "cache_hits_do_not_consume_attempts": True,
                "grants_no_scientific_authority": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkerResult":
        row = dict(value)
        supplied = str(row.pop("content_sha256", ""))
        if row.get("schema_version") != WORKER_RESULT_SCHEMA:
            raise WorkerRuntimeError("worker_result_schema_invalid")
        if not supplied or supplied != _digest(row):
            raise WorkerRuntimeError("worker_result_digest_invalid")
        return cls(
            command_id=str(row.get("command_id") or ""),
            command_sha256=str(row.get("command_sha256") or ""),
            worker_type=str(row.get("worker_type") or ""),
            status=str(row.get("status") or ""),
            cache_hit=row.get("cache_hit") is True,
            outcome_ref=dict(row.get("outcome_ref") or {}),
            payload=dict(row.get("payload") or {}),
            failure_reasons=tuple(str(value) for value in row.get("failure_reasons") or []),
            scheduled_commands=tuple(
                dict(value)
                for value in row.get("scheduled_commands") or []
                if isinstance(value, Mapping)
            ),
            material_events=tuple(str(value) for value in row.get("material_events") or []),
        )


@dataclass(frozen=True, slots=True)
class WorkerBatchResult:
    results: tuple[WorkerResult, ...]
    material_events: tuple[str, ...]
    resume_campaign: bool
    schema_version: str = WORKER_BATCH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "results": [result.to_dict() for result in self.results],
            "material_events": list(self.material_events),
            "resume_campaign": self.resume_campaign,
            "semantics": {
                "scheduled_commands_executed_breadth_first": True,
                "resume_is_signal_not_director_authority": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row


class WorkerRuntime:
    """Execute registered workers through RunKernel and immutable artifacts."""

    def __init__(
        self,
        kernel: RunKernel,
        handlers: Mapping[str, WorkerHandlerSpec],
        *,
        artifact_authorities: Mapping[str, str] | None = None,
    ) -> None:
        self.kernel = kernel
        self.handlers = dict(handlers)
        if any(key != spec.worker_type for key, spec in self.handlers.items()):
            raise ValueError("worker handler registry key mismatch")
        self.artifact_authorities = {
            str(digest).lower(): str(scope)
            for digest, scope in dict(artifact_authorities or {}).items()
            if str(digest).strip() and str(scope).strip()
        }
        self._cache_locks_guard = threading.RLock()
        self._cache_locks: dict[str, threading.RLock] = {}

    def execute(self, command: WorkerCommand) -> WorkerResult:
        spec = self._validate_command(command)
        cache_key = self._cache_key(command, spec)
        with self._cache_lock(cache_key):
            cached = self._load_cached_outcome(cache_key, spec=spec)
            if cached is not None:
                self._settle_recovered_reservation(command, cached)
                return self._receipt(command, cached, cache_hit=True)

            stale_reasons = self._stale_reasons(command)
            if stale_reasons:
                outcome, ref = self._persist_outcome(
                    command,
                    spec,
                    cache_key=cache_key,
                    status="stale",
                    payload={},
                    failure_reasons=stale_reasons,
                    scheduled_commands=(),
                    material_events=(),
                    accepted_expansion_ids=(),
                    elapsed_s=0.0,
                    publish_cache=False,
                )
                return self._receipt(command, outcome, ref=ref, cache_hit=False)

            task_id = f"worker:{hashlib.sha256(command.command_id.encode('utf-8')).hexdigest()[:24]}"
            self.kernel.reserve_task(
                task_id=task_id,
                kind=command.budget.task_kind,
                idempotency_key=f"worker:reserve:{command.idempotency_key}",
                input_revision=command.input_revision,
                uses_model=command.budget.uses_model,
                metadata={
                    "command_id": command.command_id,
                    "command_sha256": command.content_sha256,
                    "worker_type": command.worker_type,
                    "handler_version": spec.version,
                    "cache_key": cache_key,
                },
            )
            started = time.perf_counter()
            try:
                reader = WorkerArtifactReader.for_command(
                    self.kernel.artifacts,
                    command,
                    self.artifact_authorities,
                )
                raw = dict(spec.handler(command, reader) or {})
                measured = time.perf_counter() - started
                elapsed = max(measured, _nonnegative_float(raw.pop("elapsed_s", 0.0)))
                normalized = self._normalize_handler_result(raw)
                if elapsed > command.budget.timeout_s:
                    normalized = {
                        "status": "timed_out",
                        "payload": {},
                        "failure_reasons": ("worker_timeout_exceeded",),
                        "scheduled_commands": (),
                        "material_events": (),
                        "accepted_expansion_ids": (),
                    }
            except Exception as exc:  # worker failures become replayable outcomes
                elapsed = time.perf_counter() - started
                normalized = {
                    "status": "failed",
                    "payload": {},
                    "failure_reasons": (f"worker_handler_error:{type(exc).__name__}",),
                    "scheduled_commands": (),
                    "material_events": (),
                    "accepted_expansion_ids": (),
                }
            outcome, ref = self._persist_outcome(
                command,
                spec,
                cache_key=cache_key,
                elapsed_s=elapsed,
                publish_cache=True,
                **normalized,
            )
            self.kernel.settle_task(
                task_id=task_id,
                idempotency_key=f"worker:settle:{command.idempotency_key}",
                status=str(outcome["status"]),
                accepted_expansion_ids=outcome["accepted_expansion_ids"],
                output_sha256=ref.sha256,
                failure_reasons=outcome["failure_reasons"],
                elapsed_s=float(outcome["elapsed_s"]),
            )
            return self._receipt(command, outcome, ref=ref, cache_hit=False)

    def execute_pipeline(
        self,
        command: WorkerCommand,
        *,
        max_commands: int = 128,
    ) -> WorkerBatchResult:
        """Execute a root command and every explicitly scheduled child command."""
        queue = [command]
        results: list[WorkerResult] = []
        while queue:
            if len(results) >= max(1, int(max_commands)):
                raise WorkerRuntimeError("worker_pipeline_command_limit_exceeded")
            result = self.execute(queue.pop(0))
            results.append(result)
            queue.extend(WorkerCommand.from_dict(row) for row in result.scheduled_commands)
        material_events = tuple(
            sorted({event for result in results for event in result.material_events})
        )
        return WorkerBatchResult(
            results=tuple(results),
            material_events=material_events,
            resume_campaign=any(
                event in {"exact_rows_added", "source_conflict_added", "stock_records_added"}
                for event in material_events
            ),
        )

    def replay_result(self, value: WorkerResult | Mapping[str, Any]) -> WorkerResult:
        """Replay a result and its immutable outcome without executing a worker."""
        result = value if isinstance(value, WorkerResult) else WorkerResult.from_dict(value)
        ref = ArtifactRef.from_dict(result.outcome_ref)
        self.kernel.artifacts.verify(ref)
        raw = self.kernel.artifacts.read_json(ref)
        outcome = self._validated_outcome(
            raw,
            expected_worker_type=result.worker_type,
        )
        if (
            result.status != outcome["status"]
            or dict(result.payload) != dict(outcome["payload"])
            or result.failure_reasons != tuple(outcome["failure_reasons"])
            or result.scheduled_commands != tuple(outcome["scheduled_commands"])
            or result.material_events != tuple(outcome["material_events"])
        ):
            raise WorkerRuntimeError("worker_result_outcome_mismatch")
        return result

    def _validate_command(self, command: WorkerCommand) -> WorkerHandlerSpec:
        if command.run_id != self.kernel.spec.run_id:
            raise WorkerRuntimeError("worker_command_run_mismatch")
        spec = self.handlers.get(command.worker_type)
        if spec is None:
            raise WorkerRuntimeError(f"worker_type_unregistered:{command.worker_type}")
        if command.budget.task_kind != spec.task_kind:
            raise WorkerRuntimeError("worker_budget_task_kind_mismatch")
        if command.budget.uses_model and not spec.model_allowed:
            raise WorkerRuntimeError("worker_model_use_not_allowed")
        for value in command.artifact_refs:
            self.kernel.artifacts.verify(ArtifactRef.from_dict(value))
        return spec

    def _stale_reasons(self, command: WorkerCommand) -> tuple[str, ...]:
        revision = self.kernel.revision
        reasons: list[str] = []
        if command.input_revision != revision.graph_revision:
            reasons.append("worker_input_graph_revision_stale")
        expected = {
            "graph_revision": revision.graph_revision,
            "evidence_revision": revision.evidence_revision,
        }
        for key, current in expected.items():
            if key in command.dependency_revisions:
                try:
                    supplied = int(command.dependency_revisions[key])
                except (TypeError, ValueError):
                    reasons.append(f"worker_dependency_revision_invalid:{key}")
                else:
                    if supplied != current:
                        reasons.append(f"worker_dependency_revision_stale:{key}")
        return tuple(sorted(set(reasons)))

    def _cache_key(self, command: WorkerCommand, spec: WorkerHandlerSpec) -> str:
        return _digest(
            {
                "schema_version": "autoplanner_worker_cache_key.v1",
                # Cached outcomes are reusable within one campaign.  A new
                # run must still account for accepting the expansion in its
                # own kernel ledger.
                "run_id": command.run_id,
                "worker_type": command.worker_type,
                "handler_version": spec.version,
                "input_revision": command.input_revision,
                "payload": _json_value(dict(command.payload)),
                "dependency_revisions": _json_value(dict(command.dependency_revisions)),
                "artifact_sha256": sorted(
                    str(value.get("sha256") or "") for value in command.artifact_refs
                ),
                "artifact_authorities": {
                    str(value.get("sha256") or ""): self.artifact_authorities.get(
                        str(value.get("sha256") or "").lower(),
                        "",
                    )
                    for value in command.artifact_refs
                },
                "execution_policy": {
                    "task_kind": command.budget.task_kind,
                    "uses_model": command.budget.uses_model,
                    "timeout_s": command.budget.timeout_s,
                    "max_output_bytes": command.budget.max_output_bytes,
                },
            }
        )

    def _persist_outcome(
        self,
        command: WorkerCommand,
        spec: WorkerHandlerSpec,
        *,
        cache_key: str,
        status: str,
        payload: Mapping[str, Any],
        failure_reasons: tuple[str, ...],
        scheduled_commands: tuple[Mapping[str, Any], ...],
        material_events: tuple[str, ...],
        accepted_expansion_ids: tuple[str, ...],
        elapsed_s: float,
        publish_cache: bool,
    ) -> tuple[dict[str, Any], ArtifactRef]:
        outcome = {
            "schema_version": WORKER_OUTCOME_SCHEMA,
            "worker_type": command.worker_type,
            "handler_version": spec.version,
            "cache_key": cache_key,
            "status": status,
            "payload": _json_value(dict(payload)),
            "failure_reasons": list(failure_reasons),
            "scheduled_commands": [dict(value) for value in scheduled_commands],
            "material_events": list(material_events),
            "accepted_expansion_ids": list(accepted_expansion_ids),
            "elapsed_s": round(max(0.0, float(elapsed_s)), 6),
            "semantics": {
                "worker_output_is_proposal_or_observation": True,
                "kernel_owns_operational_authority": True,
                "host_validators_own_scientific_authority": True,
            },
        }
        serialized = _canonical_bytes(outcome)
        if len(serialized) > command.budget.max_output_bytes:
            outcome.update(
                {
                    "status": "failed",
                    "payload": {},
                    "failure_reasons": ["worker_output_byte_limit_exceeded"],
                    "scheduled_commands": [],
                    "material_events": [],
                    "accepted_expansion_ids": [],
                }
            )
        ref = self.kernel.artifacts.put_json(
            outcome,
            logical_name=f"{command.worker_type}_outcome.json",
            producer=f"autoplanner.worker.{command.worker_type}",
        )
        if publish_cache:
            self.kernel.artifacts.write_pointer(
                f"w/{cache_key[:2]}/{cache_key[:24]}",
                ref,
                metadata={
                    "worker_type": command.worker_type,
                    "handler_version": spec.version,
                    "cache_key": cache_key,
                },
            )
        return outcome, ref

    def _load_cached_outcome(
        self,
        cache_key: str,
        *,
        spec: WorkerHandlerSpec,
    ) -> tuple[dict[str, Any], ArtifactRef] | None:
        pointer_name = f"w/{cache_key[:2]}/{cache_key[:24]}"
        pointer_path = self.kernel.artifacts.pointers_root / f"{pointer_name}.json"
        if not pointer_path.is_file():
            return None
        try:
            ref, pointer = self.kernel.artifacts.load_pointer(
                pointer_name
            )
        except Exception as exc:
            raise WorkerRuntimeError("worker_cache_pointer_invalid") from exc
        row = self._validated_outcome(
            self.kernel.artifacts.read_json(ref),
            expected_worker_type=spec.worker_type,
        )
        metadata = dict(pointer.get("metadata") or {})
        if (
            row.get("schema_version") != WORKER_OUTCOME_SCHEMA
            or row.get("cache_key") != cache_key
            or row.get("worker_type") != spec.worker_type
            or row.get("handler_version") != spec.version
            or metadata.get("cache_key") != cache_key
        ):
            raise WorkerRuntimeError("worker_cached_outcome_binding_invalid")
        return row, ref

    def _validated_outcome(
        self,
        value: Any,
        *,
        expected_worker_type: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise WorkerRuntimeError("worker_outcome_not_object")
        row = dict(value)
        if (
            row.get("schema_version") != WORKER_OUTCOME_SCHEMA
            or row.get("worker_type") != expected_worker_type
            or row.get("status") not in _OUTCOME_STATUSES
            or not isinstance(row.get("payload"), Mapping)
            or not isinstance(row.get("failure_reasons"), list)
            or not isinstance(row.get("scheduled_commands"), list)
            or not isinstance(row.get("material_events"), list)
            or not isinstance(row.get("accepted_expansion_ids"), list)
        ):
            raise WorkerRuntimeError("worker_outcome_contract_invalid")
        if row.get("accepted_expansion_ids") and self.handlers[
            expected_worker_type
        ].task_kind not in {"proposal", "model"}:
            raise WorkerRuntimeError("worker_outcome_expansion_authority_invalid")
        for command in row["scheduled_commands"]:
            WorkerCommand.from_dict(command)
        return row

    def _settle_recovered_reservation(
        self,
        command: WorkerCommand,
        cached: tuple[dict[str, Any], ArtifactRef],
    ) -> None:
        outcome, ref = cached
        task_id = f"worker:{hashlib.sha256(command.command_id.encode('utf-8')).hexdigest()[:24]}"
        if task_id not in self.kernel.state.in_flight_tasks:
            return
        self.kernel.settle_task(
            task_id=task_id,
            idempotency_key=f"worker:settle:{command.idempotency_key}",
            status=str(outcome.get("status") or "failed"),
            accepted_expansion_ids=outcome.get("accepted_expansion_ids") or [],
            output_sha256=ref.sha256,
            failure_reasons=outcome.get("failure_reasons") or [],
            elapsed_s=float(outcome.get("elapsed_s") or 0.0),
        )

    def _receipt(
        self,
        command: WorkerCommand,
        cached: tuple[dict[str, Any], ArtifactRef] | dict[str, Any],
        *,
        ref: ArtifactRef | None = None,
        cache_hit: bool,
    ) -> WorkerResult:
        if isinstance(cached, tuple):
            outcome, resolved_ref = cached
        else:
            outcome, resolved_ref = cached, ref
        if resolved_ref is None:
            raise WorkerRuntimeError("worker_outcome_ref_missing")
        return WorkerResult(
            command_id=command.command_id,
            command_sha256=command.content_sha256,
            worker_type=command.worker_type,
            status=str(outcome.get("status") or "failed"),
            cache_hit=cache_hit,
            outcome_ref=resolved_ref.to_dict(),
            payload=dict(outcome.get("payload") or {}),
            failure_reasons=tuple(str(value) for value in outcome.get("failure_reasons") or []),
            scheduled_commands=tuple(
                dict(value)
                for value in outcome.get("scheduled_commands") or []
                if isinstance(value, Mapping)
            ),
            material_events=tuple(
                str(value) for value in outcome.get("material_events") or []
            ),
        )

    def _normalize_handler_result(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        status = str(raw.get("status") or "completed")
        if status not in _OUTCOME_STATUSES - {"stale"}:
            raise WorkerRuntimeError(f"worker_result_status_invalid:{status}")
        scheduled: list[dict[str, Any]] = []
        for value in raw.get("scheduled_commands") or []:
            command = value if isinstance(value, WorkerCommand) else WorkerCommand.from_dict(value)
            scheduled.append(command.to_dict())
        return {
            "status": status,
            "payload": dict(raw.get("payload") or {}),
            "failure_reasons": tuple(
                sorted({str(value) for value in raw.get("failure_reasons") or []})
            ),
            "scheduled_commands": tuple(scheduled),
            "material_events": tuple(
                sorted({str(value) for value in raw.get("material_events") or []})
            ),
            "accepted_expansion_ids": tuple(
                sorted({str(value) for value in raw.get("accepted_expansion_ids") or []})
            ),
        }

    def _cache_lock(self, key: str) -> threading.RLock:
        with self._cache_locks_guard:
            return self._cache_locks.setdefault(key, threading.RLock())


def _nonnegative_float(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0 else 0.0


def _json_value(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkerRuntimeError(
            f"worker_value_not_canonicalizable:{type(exc).__name__}"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()
