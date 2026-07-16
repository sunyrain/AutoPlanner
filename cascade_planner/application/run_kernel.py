"""Single durable state machine for one retrosynthesis campaign.

Workers propose facts and report observed work.  The kernel alone owns run
state, revision order, reservations, budgets, accepted-expansion identity, and
stop decisions.  Its append-only hash chain is operational authority; it still
cannot approve chemistry, evidence, stock, or route completion without the
configured acceptance report.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.runtime.artifact_store import ArtifactRef, ArtifactStore
from cascade_planner.runtime.run_index import (
    RUN_MANIFEST_SCHEMA,
    RunIndex,
)


RUN_SPEC_SCHEMA = "autoplanner_run_spec.v1"
RUN_LIMITS_SCHEMA = "autoplanner_run_limits.v1"
RUN_EVENT_SCHEMA = "autoplanner_run_event.v1"
RUN_STATE_SCHEMA = "autoplanner_run_state.v1"
RUN_REVISION_SCHEMA = "autoplanner_run_revision.v1"
DEFICIT_SCHEMA = "autoplanner_deficit.v1"
STOP_DECISION_SCHEMA = "autoplanner_stop_decision.v1"
_TASK_KINDS = {"model", "evidence", "stock", "validation", "proposal", "other"}
_TERMINAL_STATUSES = {"completed", "unresolved", "budget_exhausted", "cancelled", "failed"}
_STATUS_TRANSITIONS = {
    "created": {"running", "cancelled", "failed"},
    "running": _TERMINAL_STATUSES | {"paused"},
    "paused": {"running", "cancelled", "failed"},
    "completed": set(),
    "unresolved": set(),
    "budget_exhausted": set(),
    "cancelled": set(),
    "failed": set(),
}


class RunKernelError(RuntimeError):
    """Base error for the run state machine."""


class RunKernelCorruptionError(RunKernelError):
    """Raised when the event chain or snapshot is inconsistent."""


class RunKernelIdempotencyConflict(RunKernelError):
    """Raised when an idempotency key is reused for different work."""


class RunKernelBudgetError(RunKernelError):
    """Raised when new work exceeds the operator-owned run limits."""


@dataclass(frozen=True, slots=True)
class Deficit:
    """One revision-bound unit of unfinished, schedulable campaign work.

    A deficit is operational search state, never evidence that chemistry is
    valid.  ``deterministic`` means it can still be worked after model budgets
    are exhausted; ``model_allowed`` only controls admissible executors.
    """

    deficit_id: str
    kind: str
    source_revision: int
    priority: float = 0.0
    deterministic: bool = False
    model_allowed: bool = True
    entity_refs: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DEFICIT_SCHEMA

    def __post_init__(self) -> None:
        if not str(self.deficit_id).strip() or not str(self.kind).strip():
            raise ValueError("deficit identity and kind are required")
        if self.source_revision < 0:
            raise ValueError("deficit source_revision cannot be negative")
        if not math.isfinite(float(self.priority)):
            raise ValueError("deficit priority must be finite")
        object.__setattr__(
            self,
            "entity_refs",
            tuple(sorted({str(item) for item in self.entity_refs if str(item).strip()})),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(sorted({str(item) for item in self.reasons if str(item).strip()})),
        )
        _canonical_bytes(dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["entity_refs"] = list(self.entity_refs)
        row["reasons"] = list(self.reasons)
        row["metadata"] = dict(self.metadata)
        row["content_sha256"] = _digest(row)
        return row

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        source_revision: int | None = None,
    ) -> "Deficit":
        row = dict(value)
        supplied = str(row.pop("content_sha256", ""))
        schema = str(row.pop("schema_version", DEFICIT_SCHEMA))
        if schema != DEFICIT_SCHEMA:
            raise ValueError("deficit schema is unsupported")
        if supplied:
            check = {"schema_version": schema, **row}
            if supplied != _digest(check):
                raise RunKernelCorruptionError("deficit_digest_invalid")
        resolved_revision = (
            int(row.get("source_revision") or 0)
            if source_revision is None
            else int(source_revision)
        )
        known = {
            "deficit_id",
            "kind",
            "source_revision",
            "priority",
            "deterministic",
            "model_allowed",
            "entity_refs",
            "reasons",
            "metadata",
        }
        metadata = dict(row.get("metadata") or {})
        metadata.update({key: item for key, item in row.items() if key not in known})
        return cls(
            deficit_id=str(row.get("deficit_id") or ""),
            kind=str(row.get("kind") or ""),
            source_revision=resolved_revision,
            priority=float(row.get("priority") or 0.0),
            deterministic=row.get("deterministic") is True,
            model_allowed=row.get("model_allowed") is not False,
            entity_refs=tuple(str(item) for item in row.get("entity_refs") or ()),
            reasons=tuple(str(item) for item in row.get("reasons") or ()),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class RunLimits:
    model: RetrosynthesisRunBudget = field(default_factory=RetrosynthesisRunBudget)
    max_total_tasks: int = 256
    max_evidence_tasks: int = 64
    max_stock_tasks: int = 128
    max_validation_tasks: int = 128
    max_run_wall_time_s: float = 7_200.0
    schema_version: str = RUN_LIMITS_SCHEMA

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.max_total_tasks,
                self.max_evidence_tasks,
                self.max_stock_tasks,
                self.max_validation_tasks,
            )
        ):
            raise ValueError("run task limits cannot be negative")
        if not math.isfinite(float(self.max_run_wall_time_s)) or (
            self.max_run_wall_time_s < 0
        ):
            raise ValueError("max_run_wall_time_s must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "max_total_tasks": self.max_total_tasks,
            "max_evidence_tasks": self.max_evidence_tasks,
            "max_stock_tasks": self.max_stock_tasks,
            "max_validation_tasks": self.max_validation_tasks,
            "max_run_wall_time_s": self.max_run_wall_time_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunLimits":
        row = dict(value)
        model_row = dict(row.get("model") or {})
        model_row.pop("schema_version", None)
        model_row.pop("content_sha256", None)
        return cls(
            model=RetrosynthesisRunBudget(**model_row),
            max_total_tasks=int(row.get("max_total_tasks") or 0),
            max_evidence_tasks=int(row.get("max_evidence_tasks") or 0),
            max_stock_tasks=int(row.get("max_stock_tasks") or 0),
            max_validation_tasks=int(row.get("max_validation_tasks") or 0),
            max_run_wall_time_s=float(row.get("max_run_wall_time_s") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    target_name: str
    target_smiles: str
    acceptance: RetrosynthesisAcceptanceSpec = field(
        default_factory=RetrosynthesisAcceptanceSpec
    )
    limits: RunLimits = field(default_factory=RunLimits)
    producer: str = "autoplanner"
    created_at: str = ""
    schema_version: str = RUN_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if not self.run_id or not self.target_name or not self.target_smiles:
            raise ValueError("run spec identity and target are required")

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "target_name": self.target_name,
            "target_smiles": self.target_smiles,
            "acceptance": self.acceptance.to_dict(),
            "limits": self.limits.to_dict(),
            "producer": self.producer,
            "created_at": self.created_at,
            "semantics": {
                "one_kernel_per_run": True,
                "workers_cannot_extend_limits": True,
                "acceptance_is_scientific_completion_authority": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSpec":
        row = dict(value)
        if row.get("schema_version") != RUN_SPEC_SCHEMA:
            raise RunKernelCorruptionError("run_spec_schema_invalid")
        supplied_digest = str(row.pop("content_sha256", ""))
        if supplied_digest != _digest(row):
            raise RunKernelCorruptionError("run_spec_digest_invalid")
        acceptance_row = dict(row.get("acceptance") or {})
        acceptance_row.pop("schema_version", None)
        acceptance_row.pop("content_sha256", None)
        return cls(
            run_id=str(row.get("run_id") or ""),
            target_name=str(row.get("target_name") or ""),
            target_smiles=str(row.get("target_smiles") or ""),
            acceptance=RetrosynthesisAcceptanceSpec(**acceptance_row),
            limits=RunLimits.from_dict(dict(row.get("limits") or {})),
            producer=str(row.get("producer") or "autoplanner"),
            created_at=str(row.get("created_at") or ""),
        )


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    event_id: str
    event_type: str
    idempotency_key: str
    created_at: str
    payload: Mapping[str, Any]
    previous_event_sha256: str
    event_sha256: str
    schema_version: str = RUN_EVENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at,
            "payload": dict(self.payload),
            "previous_event_sha256": self.previous_event_sha256,
            "event_sha256": self.event_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunEvent":
        row = dict(value)
        if row.get("schema_version") != RUN_EVENT_SCHEMA:
            raise RunKernelCorruptionError("run_event_schema_invalid")
        return cls(
            run_id=str(row.get("run_id") or ""),
            sequence=int(row.get("sequence") or 0),
            event_id=str(row.get("event_id") or ""),
            event_type=str(row.get("event_type") or ""),
            idempotency_key=str(row.get("idempotency_key") or ""),
            created_at=str(row.get("created_at") or ""),
            payload=dict(row.get("payload") or {}),
            previous_event_sha256=str(row.get("previous_event_sha256") or ""),
            event_sha256=str(row.get("event_sha256") or ""),
        )


@dataclass(frozen=True, slots=True)
class RunState:
    run_id: str
    status: str = "created"
    revision: int = 0
    event_count: int = 0
    attempt_count: int = 0
    settled_task_count: int = 0
    task_wall_time_s: float = 0.0
    task_counts: Mapping[str, int] = field(default_factory=dict)
    in_flight_tasks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    accepted_expansion_ids: tuple[str, ...] = ()
    model_totals: Mapping[str, int | float] = field(default_factory=dict)
    graph_revision: int = 0
    evidence_revision: int = 0
    deficits: tuple[Mapping[str, Any], ...] = ()
    acceptance_report: Mapping[str, Any] = field(default_factory=dict)
    failure_reasons: tuple[str, ...] = ()
    updated_at: str = ""
    schema_version: str = RUN_STATE_SCHEMA

    @property
    def accepted_expansion_count(self) -> int:
        return len(self.accepted_expansion_ids)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["task_counts"] = dict(self.task_counts)
        row["in_flight_tasks"] = {
            key: dict(value) for key, value in self.in_flight_tasks.items()
        }
        row["accepted_expansion_ids"] = list(self.accepted_expansion_ids)
        row["accepted_expansion_count"] = self.accepted_expansion_count
        row["model_totals"] = dict(self.model_totals)
        row["deficits"] = [dict(value) for value in self.deficits]
        row["acceptance_report"] = dict(self.acceptance_report)
        row["failure_reasons"] = list(self.failure_reasons)
        row["terminal"] = self.terminal
        row["semantics"] = {
            "state_is_rebuilt_from_events": True,
            "accepted_expansions_are_unique_ids": True,
            "attempt_count_is_settled_proposal_tasks": True,
            "settled_task_count_includes_all_task_kinds": True,
            "queue_empty_is_not_completion": True,
        }
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True, slots=True)
class RunRevision:
    """Immutable public projection of one kernel revision."""

    run_id: str
    revision: int
    state_sha256: str
    graph_revision: int
    evidence_revision: int
    deficit_sha256: str
    acceptance_sha256: str
    status: str
    updated_at: str
    schema_version: str = RUN_REVISION_SCHEMA

    @classmethod
    def from_state(cls, state: RunState) -> "RunRevision":
        state_row = state.to_dict()
        return cls(
            run_id=state.run_id,
            revision=state.revision,
            state_sha256=str(state_row["content_sha256"]),
            graph_revision=state.graph_revision,
            evidence_revision=state.evidence_revision,
            deficit_sha256=_digest([dict(row) for row in state.deficits]),
            acceptance_sha256=_digest(dict(state.acceptance_report)),
            status=state.status,
            updated_at=state.updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True, slots=True)
class StopDecision:
    decision: str
    terminal: bool
    reasons: tuple[str, ...] = ()
    next_deficit_id: str = ""
    schema_version: str = STOP_DECISION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "terminal": self.terminal,
            "reasons": list(self.reasons),
            "next_deficit_id": self.next_deficit_id,
            "semantics": {
                "only_acceptance_can_decide_completed": True,
                "empty_deficit_queue_is_not_completion": True,
            },
        }


class RunKernel:
    """Event-sourced owner of one run's operational state and budgets."""

    def __init__(
        self,
        runtime_root: str | os.PathLike[str],
        run_dir: str | os.PathLike[str],
        *,
        spec: RunSpec | None = None,
        artifact_store_root: str | os.PathLike[str] | None = None,
        run_index_path: str | os.PathLike[str] | None = None,
        lock_timeout_s: float = 10.0,
        stale_lock_s: float = 120.0,
    ) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.kernel_dir = self.run_dir / ".autoplanner" / "kernel"
        self.kernel_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.kernel_dir / "events.jsonl"
        self.snapshot_path = self.kernel_dir / "state.json"
        self.spec_path = self.kernel_dir / "run_spec.json"
        self.lock_path = self.kernel_dir / "writer.lock"
        self.lock_timeout_s = max(0.1, float(lock_timeout_s))
        self.stale_lock_s = max(1.0, float(stale_lock_s))
        self.artifacts = ArtifactStore(
            artifact_store_root or self.runtime_root / "artifacts"
        )
        self.index = RunIndex(run_index_path or self.runtime_root / "run_index.sqlite3")
        if self.spec_path.is_file():
            stored = RunSpec.from_dict(_read_json_object(self.spec_path))
            if spec is not None and spec.to_dict()["content_sha256"] != stored.to_dict()[
                "content_sha256"
            ]:
                raise RunKernelError("run_spec_conflict")
            self.spec = stored
        else:
            if spec is None:
                raise RunKernelError("run_spec_required_for_new_kernel")
            self.spec = spec
            _atomic_write_json(self.spec_path, spec.to_dict())
        if not self.events_path.is_file() or self.events_path.stat().st_size == 0:
            self._append(
                "run_created",
                "run:create",
                {"spec_sha256": self.spec.to_dict()["content_sha256"]},
            )
        self.recover()

    @property
    def state(self) -> RunState:
        with self._locked():
            return _state_from_dict(_read_json_object(self.snapshot_path))

    @property
    def revision(self) -> RunRevision:
        return RunRevision.from_state(self.state)

    def count_task_reservations(
        self,
        *,
        kind: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count durable task admissions matching an optional metadata subset.

        This reads the event authority rather than transient in-flight state,
        so failed, interrupted, and completed attempts all consume the same
        campaign-level call allowance.
        """
        expected = dict(metadata or {})
        with self._locked():
            return sum(
                1
                for event in self._read_events()
                if event.event_type == "task_reserved"
                and (not kind or str(event.payload.get("kind") or "") == kind)
                and all(
                    dict(event.payload.get("metadata") or {}).get(key) == value
                    for key, value in expected.items()
                )
            )

    def task_lifecycle(self, task_id: str) -> dict[str, Any]:
        """Read one task's durable reservation/settlement without new authority."""

        identity = str(task_id or "").strip()
        if not identity:
            raise ValueError("task_id is required")
        reservation: RunEvent | None = None
        settlement: RunEvent | None = None
        with self._locked():
            for event in self._read_events():
                if str(event.payload.get("task_id") or "") != identity:
                    continue
                if event.event_type == "task_reserved":
                    reservation = event
                elif event.event_type == "task_settled":
                    settlement = event
        return {
            "schema_version": "autoplanner_task_lifecycle.v1",
            "run_id": self.spec.run_id,
            "task_id": identity,
            "status": (
                "settled" if settlement is not None
                else "in_flight" if reservation is not None
                else "absent"
            ),
            "reservation": reservation.to_dict() if reservation else {},
            "settlement": settlement.to_dict() if settlement else {},
            "semantics": {
                "event_log_is_operational_authority": True,
                "projection_grants_no_scientific_authority": True,
            },
        }

    def start(self) -> RunEvent:
        return self.transition("running", idempotency_key="run:start")

    def pause(self, *, idempotency_key: str = "run:pause") -> RunEvent:
        return self.transition("paused", idempotency_key=idempotency_key)

    def resume(self, *, idempotency_key: str = "run:resume") -> RunEvent:
        return self.transition("running", idempotency_key=idempotency_key)

    def cancel(
        self,
        *,
        idempotency_key: str = "run:cancel",
        reasons: Iterable[str] = (),
    ) -> RunEvent:
        return self.transition(
            "cancelled",
            idempotency_key=idempotency_key,
            reasons=reasons,
        )

    def transition(
        self,
        status: str,
        *,
        idempotency_key: str,
        reasons: Iterable[str] = (),
    ) -> RunEvent:
        current = self.state
        target = str(status)
        if target == current.status:
            existing = self._event_by_key(idempotency_key)
            if existing is not None:
                return existing
        if target not in _STATUS_TRANSITIONS.get(current.status, set()):
            raise RunKernelError(
                f"run_status_transition_invalid:{current.status}->{target}"
            )
        return self._append(
            "state_transition",
            idempotency_key,
            {"status": target, "reasons": sorted(set(str(item) for item in reasons))},
        )

    def reserve_task(
        self,
        *,
        task_id: str,
        kind: str,
        idempotency_key: str,
        input_revision: int,
        uses_model: bool = False,
        visual: bool = False,
        prompt_context_bytes: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        normalized_kind = str(kind)
        if normalized_kind not in _TASK_KINDS:
            raise ValueError(f"unsupported run task kind:{normalized_kind}")
        payload = {
            "task_id": str(task_id),
            "kind": normalized_kind,
            "input_revision": max(0, int(input_revision)),
            "uses_model": bool(uses_model),
            "visual": bool(visual),
            "prompt_context_bytes": max(0, int(prompt_context_bytes)),
            "metadata": _safe_mapping(metadata),
        }
        if visual and not uses_model:
            raise ValueError("visual task must use a model")
        if int(prompt_context_bytes) < 0:
            raise ValueError("prompt_context_bytes cannot be negative")
        existing = self._event_by_key(idempotency_key)
        if existing is not None:
            return self._assert_idempotent_event(
                existing,
                event_type="task_reserved",
                payload=payload,
            )
        state = self.state
        if state.status != "running":
            raise RunKernelError("non_running_run_cannot_reserve_task")
        self._assert_task_budget(
            state,
            kind=normalized_kind,
            uses_model=uses_model,
            visual=visual,
            prompt_context_bytes=prompt_context_bytes,
        )
        return self._append(
            "task_reserved",
            idempotency_key,
            payload,
        )

    def settle_task(
        self,
        *,
        task_id: str,
        idempotency_key: str,
        status: str,
        accepted_expansion_ids: Iterable[str] = (),
        output_sha256: str = "",
        failure_reasons: Iterable[str] = (),
        model_usage: Mapping[str, Any] | None = None,
        elapsed_s: float = 0.0,
    ) -> RunEvent:
        state = self.state
        reservation = dict(state.in_flight_tasks.get(str(task_id)) or {})
        if not reservation:
            existing = self._event_by_key(idempotency_key)
            if existing is not None:
                return self._assert_idempotent_event(
                    existing,
                    event_type="task_settled",
                    payload={
                        "task_id": str(task_id),
                        "status": str(status),
                        "accepted_expansion_ids": sorted(
                            set(
                                str(item)
                                for item in accepted_expansion_ids
                                if str(item).strip()
                            )
                        ),
                        "output_sha256": str(output_sha256),
                        "failure_reasons": sorted(
                            set(str(item) for item in failure_reasons)
                        ),
                        "model_usage": _normalized_model_usage(model_usage),
                        "elapsed_s": _finite_nonnegative_float(
                            elapsed_s,
                            field_name="elapsed_s",
                        ),
                    },
                )
            raise RunKernelError(f"task_not_reserved:{task_id}")
        payload = {
            "task_id": str(task_id),
            "status": str(status),
            "accepted_expansion_ids": sorted(
                set(
                    str(item)
                    for item in accepted_expansion_ids
                    if str(item).strip()
                )
            ),
            "output_sha256": str(output_sha256),
            "failure_reasons": sorted(set(str(item) for item in failure_reasons)),
            "model_usage": _normalized_model_usage(model_usage),
            "elapsed_s": _finite_nonnegative_float(
                elapsed_s,
                field_name="elapsed_s",
            ),
        }
        return self._append("task_settled", idempotency_key, payload)

    def publish_graph_revision(
        self,
        revision: int,
        *,
        graph_sha256: str,
        evidence_revision: int,
        idempotency_key: str,
    ) -> RunEvent:
        state = self.state
        value = int(revision)
        if value < state.graph_revision:
            raise RunKernelError("graph_revision_cannot_decrease")
        return self._append(
            "graph_revision_published",
            idempotency_key,
            {
                "graph_revision": value,
                "graph_sha256": str(graph_sha256),
                "evidence_revision": max(0, int(evidence_revision)),
            },
        )

    def replace_deficits(
        self,
        deficits: Iterable[Mapping[str, Any] | Deficit],
        *,
        source_revision: int,
        idempotency_key: str,
    ) -> RunEvent:
        rows: list[dict[str, Any]] = []
        for value in deficits:
            deficit = (
                value
                if isinstance(value, Deficit)
                else Deficit.from_dict(value, source_revision=source_revision)
            )
            if deficit.source_revision != int(source_revision):
                raise ValueError("kernel_deficit_source_revision_mismatch")
            row = deficit.to_dict()
            row.pop("content_sha256", None)
            rows.append(row)
        rows.sort(key=lambda row: (-float(row.get("priority") or 0.0), str(row.get("deficit_id") or "")))
        if any(not str(row.get("deficit_id") or "").strip() for row in rows):
            raise ValueError("kernel_deficit_id_missing")
        return self._append(
            "deficits_replaced",
            idempotency_key,
            {
                "source_revision": max(0, int(source_revision)),
                "deficits": rows,
                "deficits_sha256": _digest(rows),
            },
        )

    def record_acceptance(
        self,
        report: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> RunEvent:
        row = dict(report)
        if row.get("accepted") not in {True, False}:
            raise ValueError("kernel_acceptance_report_missing_boolean")
        return self._append(
            "acceptance_evaluated",
            idempotency_key,
            {"report": row},
        )

    def decide_stop(self) -> StopDecision:
        state = self.state
        if state.status in _TERMINAL_STATUSES:
            return StopDecision(
                decision=state.status,
                terminal=True,
                reasons=state.failure_reasons,
            )
        if state.status == "paused":
            return StopDecision(
                decision="paused",
                terminal=False,
                reasons=("operator_paused",),
            )
        if state.in_flight_tasks:
            return StopDecision(decision="continue", terminal=False)
        violation_reasons = self._budget_violation_reasons(state)
        if violation_reasons:
            return StopDecision(
                decision="budget_exhausted",
                terminal=True,
                reasons=tuple(violation_reasons),
            )
        if state.acceptance_report.get("accepted") is True:
            return StopDecision(decision="completed", terminal=True)
        if not state.deficits:
            return StopDecision(
                decision="unresolved",
                terminal=True,
                reasons=("acceptance_not_met_and_no_actionable_deficits",),
            )
        budget_reasons = self._budget_reasons(state)
        if budget_reasons:
            return StopDecision(
                decision="budget_exhausted",
                terminal=True,
                reasons=tuple(budget_reasons),
            )
        return StopDecision(
            decision="continue",
            terminal=False,
            next_deficit_id=str(state.deficits[0].get("deficit_id") or ""),
        )

    def apply_stop_decision(self, *, idempotency_key: str) -> StopDecision:
        decision = self.decide_stop()
        if decision.terminal and not self.state.terminal:
            self.transition(
                decision.decision,
                idempotency_key=idempotency_key,
                reasons=decision.reasons,
            )
        return decision

    def recover(self) -> dict[str, Any]:
        with self._locked():
            repaired_bytes = self._repair_event_tail()
            events = self._read_events()
            state = _replay(self.spec, events)
            prior_digest = ""
            if self.snapshot_path.is_file():
                try:
                    prior_digest = str(
                        _read_json_object(self.snapshot_path).get("content_sha256") or ""
                    )
                except (OSError, UnicodeError, json.JSONDecodeError):
                    prior_digest = ""
            ref = self._persist_state(state)
        return {
            "schema_version": "autoplanner_run_kernel_recovery.v1",
            "run_id": self.spec.run_id,
            "event_count": len(events),
            "repaired_tail_bytes": repaired_bytes,
            "prior_snapshot_sha256": prior_digest,
            "replayed_state_sha256": state.to_dict()["content_sha256"],
            "state_ref": ref.to_dict(),
            "in_flight_task_count": len(state.in_flight_tasks),
            "semantics": {
                "events_are_operational_authority": True,
                "snapshot_was_rebuilt": True,
                "in_flight_tasks_are_recoverable": True,
            },
        }

    def _append(
        self,
        event_type: str,
        idempotency_key: str,
        payload: Mapping[str, Any],
    ) -> RunEvent:
        key = str(idempotency_key or "").strip()
        if not key:
            raise ValueError("run_event_idempotency_key_required")
        canonical_payload = _safe_mapping(payload)
        with self._locked():
            self._repair_event_tail()
            events = self._read_events()
            for event in events:
                if event.idempotency_key == key:
                    return self._assert_idempotent_event(
                        event,
                        event_type=event_type,
                        payload=canonical_payload,
                    )
            current_state = _replay(self.spec, events)
            self._validate_next_event(
                current_state,
                event_type=event_type,
                payload=canonical_payload,
            )
            previous = events[-1].event_sha256 if events else ""
            sequence = len(events) + 1
            semantic = {
                "run_id": self.spec.run_id,
                "event_type": str(event_type),
                "idempotency_key": key,
                "payload": canonical_payload,
            }
            event_id = f"event:{hashlib.sha256(_canonical_bytes(semantic)).hexdigest()[:24]}"
            body = {
                "schema_version": RUN_EVENT_SCHEMA,
                "run_id": self.spec.run_id,
                "sequence": sequence,
                "event_id": event_id,
                "event_type": str(event_type),
                "idempotency_key": key,
                "created_at": _utc_now(),
                "payload": canonical_payload,
                "previous_event_sha256": previous,
            }
            body["event_sha256"] = _digest(body)
            event = RunEvent.from_dict(body)
            self._append_event_bytes(event)
            events.append(event)
            state = _replay(self.spec, events)
            self._persist_state(state)
            return event

    def _validate_next_event(
        self,
        state: RunState,
        *,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if event_type == "task_reserved":
            task_id = str(payload.get("task_id") or "")
            if not task_id or task_id in state.in_flight_tasks:
                raise RunKernelError(f"task_already_reserved:{task_id}")
            self._assert_task_budget(
                state,
                kind=str(payload.get("kind") or "other"),
                uses_model=payload.get("uses_model") is True,
                visual=payload.get("visual") is True,
                prompt_context_bytes=int(payload.get("prompt_context_bytes") or 0),
            )
        elif event_type == "task_settled":
            task_id = str(payload.get("task_id") or "")
            if task_id not in state.in_flight_tasks:
                raise RunKernelError(f"task_not_reserved:{task_id}")
            reservation = dict(state.in_flight_tasks[task_id])
            usage = dict(payload.get("model_usage") or {})
            if (
                int(usage.get("model_invocations") or 0) > 0
                and reservation.get("uses_model") is not True
            ):
                raise RunKernelError("non_model_task_reported_model_usage")
            if payload.get("accepted_expansion_ids") and str(
                reservation.get("kind") or ""
            ) not in {"proposal", "model"}:
                raise RunKernelError("non_proposal_task_reported_accepted_expansion")
            accepted = set(state.accepted_expansion_ids)
            accepted.update(
                str(item)
                for item in payload.get("accepted_expansion_ids") or []
                if str(item).strip()
            )
            if len(accepted) > self.spec.limits.model.max_accepted_expansions:
                raise RunKernelBudgetError(
                    "run_accepted_expansion_budget_exceeded"
                )
        elif event_type == "graph_revision_published":
            if int(payload.get("graph_revision") or 0) < state.graph_revision:
                raise RunKernelError("graph_revision_cannot_decrease")
        elif event_type == "deficits_replaced":
            if int(payload.get("source_revision") or 0) != state.graph_revision:
                raise RunKernelError("deficit_source_revision_mismatch")
        elif event_type == "acceptance_evaluated":
            report = dict(payload.get("report") or {})
            if int(report.get("graph_revision") or 0) != state.graph_revision:
                raise RunKernelError("acceptance_graph_revision_mismatch")
        elif event_type == "state_transition":
            target = str(payload.get("status") or "")
            if target not in _STATUS_TRANSITIONS.get(state.status, set()):
                raise RunKernelError(
                    f"run_status_transition_invalid:{state.status}->{target}"
                )
        elif event_type != "run_created":
            raise RunKernelError(f"run_event_type_unsupported:{event_type}")

    def _assert_idempotent_event(
        self,
        event: RunEvent,
        *,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> RunEvent:
        if event.event_type != event_type or dict(event.payload) != dict(payload):
            raise RunKernelIdempotencyConflict(
                f"run_event_idempotency_conflict:{event.idempotency_key}"
            )
        return event

    def _event_by_key(self, key: str) -> RunEvent | None:
        with self._locked():
            return next(
                (event for event in self._read_events() if event.idempotency_key == key),
                None,
            )

    def _assert_task_budget(
        self,
        state: RunState,
        *,
        kind: str,
        uses_model: bool,
        visual: bool = False,
        prompt_context_bytes: int = 0,
    ) -> None:
        reasons: list[str] = []
        pending_count = len(state.in_flight_tasks)
        pending_proposal_attempts = sum(
            1
            for row in state.in_flight_tasks.values()
            if str(row.get("kind") or "") == "proposal"
        )
        if kind == "proposal" and (
            state.attempt_count + pending_proposal_attempts
            >= self.spec.limits.model.max_attempt_runs
        ):
            reasons.append("run_attempt_budget_exhausted")
        if state.settled_task_count + pending_count >= self.spec.limits.max_total_tasks:
            reasons.append("run_total_task_budget_exhausted")
        if state.task_wall_time_s >= self.spec.limits.max_run_wall_time_s:
            reasons.append("run_wall_time_budget_exhausted")
        kind_count = int(state.task_counts.get(kind) or 0) + sum(
            1
            for row in state.in_flight_tasks.values()
            if str(row.get("kind") or "") == kind
        )
        kind_limits = {
            "evidence": self.spec.limits.max_evidence_tasks,
            "stock": self.spec.limits.max_stock_tasks,
            "validation": self.spec.limits.max_validation_tasks,
        }
        if kind in kind_limits and kind_count >= kind_limits[kind]:
            reasons.append(f"run_{kind}_task_budget_exhausted")
        if (
            kind == "proposal"
            and state.accepted_expansion_count
            >= self.spec.limits.model.max_accepted_expansions
        ):
            reasons.append("run_accepted_expansion_budget_exhausted")
        if uses_model:
            pending_models = sum(
                1
                for row in state.in_flight_tasks.values()
                if row.get("uses_model") is True
            )
            if (
                int(state.model_totals.get("model_invocations") or 0)
                + pending_models
                >= self.spec.limits.model.max_model_invocations
            ):
                reasons.append("run_model_invocation_budget_exhausted")
            if (
                int(state.model_totals.get("input_tokens") or 0)
                >= self.spec.limits.model.max_total_input_tokens
            ):
                reasons.append("run_input_token_budget_exhausted")
            if (
                int(state.model_totals.get("output_tokens") or 0)
                >= self.spec.limits.model.max_total_output_tokens
            ):
                reasons.append("run_output_token_budget_exhausted")
            if (
                float(state.model_totals.get("wall_time_s") or 0.0)
                >= self.spec.limits.model.max_total_wall_time_s
            ):
                reasons.append("run_model_wall_time_budget_exhausted")
            pending_visual = sum(
                1
                for row in state.in_flight_tasks.values()
                if row.get("visual") is True
            )
            if visual and (
                int(state.model_totals.get("visual_invocations") or 0)
                + pending_visual
                >= self.spec.limits.model.max_visual_invocations
            ):
                reasons.append("run_visual_invocation_budget_exhausted")
            if prompt_context_bytes > self.spec.limits.model.max_prompt_context_bytes:
                reasons.append("prompt_context_byte_budget_exceeded")
        if reasons:
            raise RunKernelBudgetError(";".join(sorted(set(reasons))))

    def _budget_reasons(
        self,
        state: RunState,
    ) -> list[str]:
        budget = self.spec.limits.model
        totals = state.model_totals
        reasons: list[str] = []
        deficit_kinds = {
            str(row.get("kind") or "")
            for row in state.deficits
            if str(row.get("kind") or "")
        }
        proposal_only_kinds = {"materialization", "expansion", "diversity"}
        non_proposal_kinds = deficit_kinds - proposal_only_kinds - {"route_closure"}
        if (
            state.attempt_count >= budget.max_attempt_runs
            and bool(deficit_kinds & proposal_only_kinds)
            and not non_proposal_kinds
        ):
            reasons.append("run_attempt_budget_exhausted")
        if state.settled_task_count >= self.spec.limits.max_total_tasks:
            reasons.append("run_total_task_budget_exhausted")
        if state.task_wall_time_s >= self.spec.limits.max_run_wall_time_s:
            reasons.append("run_wall_time_budget_exhausted")
        deterministic_work = any(
            row.get("deterministic") is True
            or row.get("model_allowed") is not True
            for row in state.deficits
        )
        if not deterministic_work:
            if state.accepted_expansion_count >= budget.max_accepted_expansions:
                reasons.append("run_accepted_expansion_budget_exhausted")
            if int(totals.get("model_invocations") or 0) >= budget.max_model_invocations:
                reasons.append("run_model_invocation_budget_exhausted")
            if int(totals.get("input_tokens") or 0) >= budget.max_total_input_tokens:
                reasons.append("run_input_token_budget_exhausted")
            if int(totals.get("output_tokens") or 0) >= budget.max_total_output_tokens:
                reasons.append("run_output_token_budget_exhausted")
            if float(totals.get("wall_time_s") or 0.0) >= budget.max_total_wall_time_s:
                reasons.append("run_model_wall_time_budget_exhausted")
        return reasons

    def _budget_violation_reasons(self, state: RunState) -> list[str]:
        """Report observed overrun; reaching an allowed cap is still compliant."""

        budget = self.spec.limits.model
        totals = state.model_totals
        checks = (
            (state.attempt_count, budget.max_attempt_runs, "run_attempt_budget_violated"),
            (
                state.settled_task_count,
                self.spec.limits.max_total_tasks,
                "run_total_task_budget_violated",
            ),
            (
                state.task_wall_time_s,
                self.spec.limits.max_run_wall_time_s,
                "run_wall_time_budget_violated",
            ),
            (
                state.accepted_expansion_count,
                budget.max_accepted_expansions,
                "run_accepted_expansion_budget_violated",
            ),
            (
                int(totals.get("model_invocations") or 0),
                budget.max_model_invocations,
                "run_model_invocation_budget_violated",
            ),
            (
                int(totals.get("input_tokens") or 0),
                budget.max_total_input_tokens,
                "run_input_token_budget_violated",
            ),
            (
                int(totals.get("output_tokens") or 0),
                budget.max_total_output_tokens,
                "run_output_token_budget_violated",
            ),
            (
                float(totals.get("wall_time_s") or 0.0),
                budget.max_total_wall_time_s,
                "run_model_wall_time_budget_violated",
            ),
            (
                int(totals.get("visual_invocations") or 0),
                budget.max_visual_invocations,
                "run_visual_invocation_budget_violated",
            ),
        )
        return sorted(reason for observed, limit, reason in checks if observed > limit)

    def _persist_state(self, state: RunState) -> ArtifactRef:
        payload = state.to_dict()
        _atomic_write_json(self.snapshot_path, payload)
        ref = self.artifacts.put_json(
            payload,
            logical_name="run_state.json",
            producer="autoplanner.run_kernel",
        )
        pointer = (
            "runs/"
            + hashlib.sha256(self.spec.run_id.encode("utf-8")).hexdigest()
            + "/kernel/latest"
        )
        self.artifacts.write_pointer(
            pointer,
            ref,
            metadata={"run_id": self.spec.run_id, "revision": state.revision},
        )
        proof_deficits = sum(
            1
            for row in state.deficits
            if str(row.get("kind") or "")
            in {"exact_evidence", "reaction_validation"}
        )
        stock_deficits = sum(
            1 for row in state.deficits if str(row.get("kind") or "") == "stock_audit"
        )
        self.index.upsert_run(
            {
                "schema_version": RUN_MANIFEST_SCHEMA,
                "run_id": self.spec.run_id,
                "case_id": self.spec.run_id,
                "target_name": self.spec.target_name,
                "status": state.status,
                "revision": state.revision,
                "updated_at": state.updated_at,
                "run_dir": str(self.run_dir),
                "state_sha256": str(payload["content_sha256"]),
                "accepted": state.acceptance_report.get("accepted"),
                "cost_totals": {
                    **dict(state.model_totals),
                    "attempt_runs": state.attempt_count,
                    "accepted_expansions": state.accepted_expansion_count,
                    "task_wall_time_s": state.task_wall_time_s,
                },
                "graph": {
                    "molecule_count": int(
                        state.acceptance_report.get("molecule_count") or 0
                    ),
                    "hyperedge_count": int(
                        state.acceptance_report.get("hyperedge_count") or 0
                    ),
                    "complete_route_count": int(
                        state.acceptance_report.get("complete_route_count") or 0
                    ),
                },
                "deficits": {"proof": proof_deficits, "stock": stock_deficits},
                "metrics": {},
            }
        )
        self.index.index_artifact(
            run_id=self.spec.run_id,
            artifact_id="run_kernel_state",
            ref=ref,
            revision=state.revision,
            authority_scope="operational_state_projection",
        )
        return ref

    def _append_event_bytes(self, event: RunEvent) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_bytes(event.to_dict()) + b"\n"
        with self.events_path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_events(self) -> list[RunEvent]:
        if not self.events_path.is_file():
            return []
        events: list[RunEvent] = []
        previous = ""
        for line_number, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunKernelCorruptionError(
                    f"run_event_json_invalid:{line_number}"
                ) from exc
            event = RunEvent.from_dict(raw)
            body = event.to_dict()
            supplied = str(body.pop("event_sha256") or "")
            if (
                event.run_id != self.spec.run_id
                or event.sequence != len(events) + 1
                or event.previous_event_sha256 != previous
                or supplied != _digest(body)
            ):
                raise RunKernelCorruptionError(
                    f"run_event_chain_invalid:{line_number}"
                )
            previous = supplied
            events.append(event)
        return events

    def _repair_event_tail(self) -> int:
        if not self.events_path.is_file():
            return 0
        payload = self.events_path.read_bytes()
        if not payload or payload.endswith(b"\n"):
            return 0
        last_newline = payload.rfind(b"\n")
        tail = payload[last_newline + 1 :]
        try:
            value = json.loads(tail.decode("utf-8"))
            RunEvent.from_dict(value)
        except (UnicodeDecodeError, json.JSONDecodeError, RunKernelError, ValueError):
            repaired = len(tail)
            with self.events_path.open("r+b") as handle:
                handle.truncate(last_newline + 1)
                handle.flush()
                os.fsync(handle.fileno())
            return repaired
        with self.events_path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return 0

    @contextmanager
    def _locked(self) -> Iterator[None]:
        deadline = time.monotonic() + self.lock_timeout_s
        last_permission_error: PermissionError | None = None
        observed_contention = False
        while True:
            try:
                self.lock_path.mkdir()
                break
            except FileExistsError:
                observed_contention = True
            except PermissionError as exc:
                # Windows can report Access denied, instead of FileExistsError,
                # while another thread is creating or removing the directory.
                # Retry that bounded race, but preserve a genuine permission
                # failure when no competing lock was ever observable.
                last_permission_error = exc
                try:
                    self.kernel_dir.stat()
                except OSError:
                    raise
                if self.lock_path.exists():
                    observed_contention = True
            try:
                age = time.time() - self.lock_path.stat().st_mtime
                if age > self.stale_lock_s:
                    try:
                        self.lock_path.rmdir()
                    except FileNotFoundError:
                        continue
                    except OSError:
                        pass
                    else:
                        continue
            except (FileNotFoundError, PermissionError, OSError):
                pass
            if time.monotonic() >= deadline:
                if last_permission_error is not None and not observed_contention:
                    raise last_permission_error
                raise RunKernelError("run_kernel_writer_lock_timeout")
            time.sleep(0.01)
        try:
            yield
        finally:
            for attempt in range(8):
                try:
                    self.lock_path.rmdir()
                    break
                except FileNotFoundError:
                    break
                except PermissionError:
                    if attempt == 7:
                        raise RunKernelError("run_kernel_writer_lock_release_failed")
                    time.sleep(min(0.1, 0.005 * (2**attempt)))


def _replay(spec: RunSpec, events: Iterable[RunEvent]) -> RunState:
    state: dict[str, Any] = {
        "run_id": spec.run_id,
        "status": "created",
        "revision": 0,
        "event_count": 0,
        "attempt_count": 0,
        "settled_task_count": 0,
        "task_wall_time_s": 0.0,
        "task_counts": {},
        "in_flight_tasks": {},
        "accepted_expansion_ids": set(),
        "model_totals": {
            "model_invocations": 0,
            "visual_invocations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_time_s": 0.0,
        },
        "graph_revision": 0,
        "evidence_revision": 0,
        "deficits": [],
        "acceptance_report": {},
        "failure_reasons": [],
        "updated_at": spec.created_at,
    }
    for event in events:
        payload = dict(event.payload)
        if event.event_type == "task_reserved":
            task_id = str(payload.get("task_id") or "")
            if task_id in state["in_flight_tasks"]:
                raise RunKernelCorruptionError(f"task_reserved_twice:{task_id}")
            state["in_flight_tasks"][task_id] = payload
        elif event.event_type == "task_settled":
            task_id = str(payload.get("task_id") or "")
            reservation = state["in_flight_tasks"].pop(task_id, None)
            if reservation is None:
                raise RunKernelCorruptionError(f"task_settled_without_reservation:{task_id}")
            kind = str(reservation.get("kind") or "other")
            if kind == "proposal":
                state["attempt_count"] += 1
            state["settled_task_count"] += 1
            state["task_wall_time_s"] = round(
                float(state["task_wall_time_s"])
                + max(0.0, float(payload.get("elapsed_s") or 0.0)),
                6,
            )
            state["task_counts"][kind] = int(state["task_counts"].get(kind) or 0) + 1
            state["accepted_expansion_ids"].update(
                str(item)
                for item in payload.get("accepted_expansion_ids") or []
                if str(item).strip()
            )
            usage = dict(payload.get("model_usage") or {})
            for key in ("model_invocations", "input_tokens", "output_tokens"):
                state["model_totals"][key] += int(usage.get(key) or 0)
            state["model_totals"]["visual_invocations"] += int(
                usage.get("visual_invocations") or 0
            )
            state["model_totals"]["wall_time_s"] = round(
                float(state["model_totals"].get("wall_time_s") or 0.0)
                + float(usage.get("wall_time_s") or 0.0),
                6,
            )
        elif event.event_type == "graph_revision_published":
            graph_revision = int(payload.get("graph_revision") or 0)
            evidence_revision = int(payload.get("evidence_revision") or 0)
            if (
                graph_revision < state["graph_revision"]
                or evidence_revision < state["evidence_revision"]
            ):
                raise RunKernelCorruptionError("replayed_revision_decreased")
            if graph_revision != state["graph_revision"]:
                state["acceptance_report"] = {}
            state["graph_revision"] = graph_revision
            state["evidence_revision"] = evidence_revision
        elif event.event_type == "deficits_replaced":
            if int(payload.get("source_revision") or 0) != state["graph_revision"]:
                raise RunKernelCorruptionError(
                    "replayed_deficit_source_revision_mismatch"
                )
            state["deficits"] = [
                dict(row)
                for row in payload.get("deficits") or []
                if isinstance(row, Mapping)
            ]
        elif event.event_type == "acceptance_evaluated":
            report = dict(payload.get("report") or {})
            if int(report.get("graph_revision") or 0) != state["graph_revision"]:
                raise RunKernelCorruptionError(
                    "replayed_acceptance_graph_revision_mismatch"
                )
            state["acceptance_report"] = report
        elif event.event_type == "state_transition":
            target = str(payload.get("status") or "")
            if target not in _STATUS_TRANSITIONS.get(state["status"], set()):
                raise RunKernelCorruptionError(
                    f"replayed_status_transition_invalid:{state['status']}->{target}"
                )
            state["status"] = target
            state["failure_reasons"] = list(payload.get("reasons") or [])
        elif event.event_type == "run_created":
            if event.sequence != 1:
                raise RunKernelCorruptionError("run_created_event_not_first")
        else:
            raise RunKernelCorruptionError(
                f"run_event_type_unsupported:{event.event_type}"
            )
        state["revision"] = event.sequence
        state["event_count"] = event.sequence
        state["updated_at"] = event.created_at
    return RunState(
        run_id=state["run_id"],
        status=state["status"],
        revision=state["revision"],
        event_count=state["event_count"],
        attempt_count=state["attempt_count"],
        settled_task_count=state["settled_task_count"],
        task_wall_time_s=state["task_wall_time_s"],
        task_counts=dict(state["task_counts"]),
        in_flight_tasks={
            key: dict(value) for key, value in state["in_flight_tasks"].items()
        },
        accepted_expansion_ids=tuple(sorted(state["accepted_expansion_ids"])),
        model_totals=dict(state["model_totals"]),
        graph_revision=state["graph_revision"],
        evidence_revision=state["evidence_revision"],
        deficits=tuple(dict(row) for row in state["deficits"]),
        acceptance_report=dict(state["acceptance_report"]),
        failure_reasons=tuple(state["failure_reasons"]),
        updated_at=state["updated_at"],
    )


def _state_from_dict(value: Mapping[str, Any]) -> RunState:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    if value.get("schema_version") != RUN_STATE_SCHEMA or supplied != _digest(row):
        raise RunKernelCorruptionError("run_state_snapshot_digest_invalid")
    return RunState(
        run_id=str(row.get("run_id") or ""),
        status=str(row.get("status") or "created"),
        revision=int(row.get("revision") or 0),
        event_count=int(row.get("event_count") or 0),
        attempt_count=int(row.get("attempt_count") or 0),
        settled_task_count=int(row.get("settled_task_count") or 0),
        task_wall_time_s=float(row.get("task_wall_time_s") or 0.0),
        task_counts=dict(row.get("task_counts") or {}),
        in_flight_tasks={
            str(key): dict(item)
            for key, item in dict(row.get("in_flight_tasks") or {}).items()
        },
        accepted_expansion_ids=tuple(
            str(item) for item in row.get("accepted_expansion_ids") or []
        ),
        model_totals=dict(row.get("model_totals") or {}),
        graph_revision=int(row.get("graph_revision") or 0),
        evidence_revision=int(row.get("evidence_revision") or 0),
        deficits=tuple(
            dict(item)
            for item in row.get("deficits") or []
            if isinstance(item, Mapping)
        ),
        acceptance_report=dict(row.get("acceptance_report") or {}),
        failure_reasons=tuple(
            str(item) for item in row.get("failure_reasons") or []
        ),
        updated_at=str(row.get("updated_at") or ""),
    )


def _normalized_model_usage(value: Mapping[str, Any] | None) -> dict[str, int | float]:
    row = dict(value or {})
    wall_time_s = _finite_nonnegative_float(
        row.get("wall_time_s"),
        field_name="model_usage.wall_time_s",
    )
    return {
        "model_invocations": max(0, int(row.get("model_invocations") or 0)),
        "visual_invocations": max(0, int(row.get("visual_invocations") or 0)),
        "input_tokens": max(0, int(row.get("input_tokens") or 0)),
        "output_tokens": max(0, int(row.get("output_tokens") or 0)),
        "wall_time_s": wall_time_s,
    }


def _finite_nonnegative_float(value: Any, *, field_name: str) -> float:
    resolved = float(value or 0.0)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return resolved


def _safe_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    row = dict(value or {})
    _canonical_bytes(row)
    return row


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
        raise RunKernelError(
            f"run_kernel_value_not_canonicalizable:{type(exc).__name__}"
        ) from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunKernelCorruptionError(f"run_kernel_json_not_object:{path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
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
        _replace_with_bounded_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_with_bounded_retry(source: Path, destination: Path) -> None:
    """Survive short Windows reader/AV locks without weakening atomicity."""

    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.4, 0.025 * (2**attempt)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "RUN_EVENT_SCHEMA",
    "RUN_LIMITS_SCHEMA",
    "RUN_SPEC_SCHEMA",
    "RUN_STATE_SCHEMA",
    "STOP_DECISION_SCHEMA",
    "RunEvent",
    "RunKernel",
    "RunKernelBudgetError",
    "RunKernelCorruptionError",
    "RunKernelError",
    "RunKernelIdempotencyConflict",
    "RunLimits",
    "RunSpec",
    "RunState",
    "StopDecision",
]
