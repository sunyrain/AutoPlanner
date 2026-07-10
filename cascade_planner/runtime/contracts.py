"""Dependency-free contracts for direct child-agent execution.

The runtime package deliberately models *execution*, not orchestration policy.
Adapters may use Codex, another agent backend, or a deterministic test double as
long as they preserve these identifiers and lifecycle rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, ClassVar, Iterable, Mapping
from uuid import uuid4


def utc_now() -> str:
    """Return a stable, JSON-friendly UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def context_hash(context: Any) -> str:
    """Hash JSON-compatible context using canonical UTF-8 JSON.

    Context hashes are carried through every lifecycle artifact so a controller
    can reject a stale result even when an agent id was accidentally reused.
    """

    encoded = json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AgentContractError(ValueError):
    """Base class for invalid child-agent runtime data."""


class InvalidStateTransition(AgentContractError):
    """Raised when a lifecycle transition is not permitted."""

    def __init__(self, source: "AgentState", target: "AgentState") -> None:
        super().__init__(f"invalid agent state transition: {source.value} -> {target.value}")
        self.source = source
        self.target = target


class AgentState(str, Enum):
    """Backend-neutral child-agent lifecycle state.

    ``WAITING`` means that an agent remains live but is waiting for a child or
    another bounded dependency. Terminal states are intentionally immutable;
    a retry is a new ``attempt`` beginning in ``PENDING``.
    """

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    LOST = "lost"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_AGENT_STATES

    @property
    def is_active(self) -> bool:
        return self in ACTIVE_AGENT_STATES

    def can_transition_to(self, target: "AgentState | str") -> bool:
        target_state = coerce_agent_state(target)
        return target_state in ALLOWED_AGENT_STATE_TRANSITIONS[self]

    def require_transition_to(self, target: "AgentState | str") -> "AgentState":
        target_state = coerce_agent_state(target)
        if not self.can_transition_to(target_state):
            raise InvalidStateTransition(self, target_state)
        return target_state


TERMINAL_AGENT_STATES = frozenset(
    {
        AgentState.SUCCEEDED,
        AgentState.FAILED,
        AgentState.CANCELLED,
        AgentState.TIMED_OUT,
        AgentState.LOST,
    }
)
ACTIVE_AGENT_STATES = frozenset(
    {
        AgentState.PENDING,
        AgentState.STARTING,
        AgentState.RUNNING,
        AgentState.WAITING,
    }
)
ALLOWED_AGENT_STATE_TRANSITIONS: Mapping[AgentState, frozenset[AgentState]] = {
    AgentState.PENDING: frozenset({AgentState.STARTING, AgentState.CANCELLED}),
    AgentState.STARTING: frozenset(
        {
            AgentState.RUNNING,
            AgentState.FAILED,
            AgentState.CANCELLED,
            AgentState.TIMED_OUT,
            AgentState.LOST,
        }
    ),
    AgentState.RUNNING: frozenset(
        {
            AgentState.WAITING,
            AgentState.SUCCEEDED,
            AgentState.FAILED,
            AgentState.CANCELLED,
            AgentState.TIMED_OUT,
            AgentState.LOST,
        }
    ),
    AgentState.WAITING: frozenset(
        {
            AgentState.RUNNING,
            AgentState.SUCCEEDED,
            AgentState.FAILED,
            AgentState.CANCELLED,
            AgentState.TIMED_OUT,
            AgentState.LOST,
        }
    ),
    AgentState.SUCCEEDED: frozenset(),
    AgentState.FAILED: frozenset(),
    AgentState.CANCELLED: frozenset(),
    AgentState.TIMED_OUT: frozenset(),
    AgentState.LOST: frozenset(),
}


def coerce_agent_state(value: AgentState | str) -> AgentState:
    if isinstance(value, AgentState):
        return value
    try:
        return AgentState(str(value))
    except ValueError as exc:
        raise AgentContractError(f"unknown agent state: {value!r}") from exc


def require_agent_state_transition(
    source: AgentState | str,
    target: AgentState | str,
) -> AgentState:
    return coerce_agent_state(source).require_transition_to(target)


def _require_text(value: str, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise AgentContractError(f"{name} must be non-empty")
    return normalized


def _string_tuple(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise AgentContractError(f"{name} must be a sequence of strings, not a string")
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            raise AgentContractError(f"{name} entries must be non-empty")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _validate_identity(
    *,
    run_id: str,
    agent_id: str,
    parent_agent_id: str | None,
    child_agent_ids: tuple[str, ...],
    attempt: int,
    idempotency_key: str,
    context_hash_value: str,
) -> None:
    _require_text(run_id, "run_id")
    _require_text(agent_id, "agent_id")
    _require_text(idempotency_key, "idempotency_key")
    _require_text(context_hash_value, "context_hash")
    if int(attempt) < 1:
        raise AgentContractError("attempt must be >= 1")
    if parent_agent_id is not None:
        _require_text(parent_agent_id, "parent_agent_id")
        if parent_agent_id == agent_id:
            raise AgentContractError("an agent cannot be its own parent")
    if agent_id in child_agent_ids:
        raise AgentContractError("an agent cannot be its own child")
    if parent_agent_id is not None and parent_agent_id in child_agent_ids:
        raise AgentContractError("parent_agent_id cannot also be a child_agent_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class Budget:
    """Hard execution limits understood by all child-agent adapters.

    ``None`` means that a particular dimension is not constrained by the
    controller. Adapters must never silently increase a non-``None`` limit.
    """

    max_wall_time_s: float | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_output_bytes: int | None = None
    max_children: int | None = None

    schema_version: ClassVar[str] = "child_agent_budget.v1"

    def __post_init__(self) -> None:
        if self.max_wall_time_s is not None:
            wall_time = float(self.max_wall_time_s)
            if not math.isfinite(wall_time) or wall_time <= 0:
                raise AgentContractError("max_wall_time_s must be finite and > 0 when set")
            object.__setattr__(self, "max_wall_time_s", wall_time)
        for name in (
            "max_turns",
            "max_tool_calls",
            "max_tokens",
            "max_output_bytes",
            "max_children",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            integer = int(value)
            if isinstance(value, float) and not value.is_integer():
                raise AgentContractError(f"{name} must be an integer when set")
            if integer < 0:
                raise AgentContractError(f"{name} must be >= 0 when set")
            object.__setattr__(self, name, integer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_wall_time_s": self.max_wall_time_s,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
            "max_output_bytes": self.max_output_bytes,
            "max_children": self.max_children,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Budget":
        raw = dict(data or {})
        return cls(
            max_wall_time_s=(
                float(raw["max_wall_time_s"])
                if raw.get("max_wall_time_s") is not None
                else None
            ),
            max_turns=int(raw["max_turns"]) if raw.get("max_turns") is not None else None,
            max_tool_calls=(
                int(raw["max_tool_calls"])
                if raw.get("max_tool_calls") is not None
                else None
            ),
            max_tokens=int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None,
            max_output_bytes=(
                int(raw["max_output_bytes"])
                if raw.get("max_output_bytes") is not None
                else None
            ),
            max_children=(
                int(raw["max_children"])
                if raw.get("max_children") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSpec:
    """Immutable request to start one direct child agent."""

    run_id: str
    agent_id: str
    role: str
    objective: str
    idempotency_key: str
    context_hash: str
    parent_agent_id: str | None = None
    child_agent_ids: tuple[str, ...] = ()
    attempt: int = 1
    capabilities: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()
    budget: Budget = field(default_factory=Budget)
    context_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = "child_agent_spec.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(self, "agent_id", _require_text(self.agent_id, "agent_id"))
        object.__setattr__(self, "role", _require_text(self.role, "role"))
        object.__setattr__(self, "objective", _require_text(self.objective, "objective"))
        parent = str(self.parent_agent_id).strip() if self.parent_agent_id is not None else None
        children = _string_tuple(self.child_agent_ids, "child_agent_ids")
        object.__setattr__(self, "parent_agent_id", parent)
        object.__setattr__(self, "child_agent_ids", children)
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(self, "capabilities", _string_tuple(self.capabilities, "capabilities"))
        object.__setattr__(self, "write_scope", _string_tuple(self.write_scope, "write_scope"))
        object.__setattr__(self, "context_refs", _string_tuple(self.context_refs, "context_refs"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if not isinstance(self.budget, Budget):
            raise AgentContractError("budget must be a Budget")
        _validate_identity(
            run_id=self.run_id,
            agent_id=self.agent_id,
            parent_agent_id=parent,
            child_agent_ids=children,
            attempt=self.attempt,
            idempotency_key=self.idempotency_key,
            context_hash_value=self.context_hash,
        )

    @classmethod
    def from_context(
        cls,
        *,
        run_id: str,
        agent_id: str,
        role: str,
        objective: str,
        context: Any,
        idempotency_key: str,
        **kwargs: Any,
    ) -> "AgentSpec":
        return cls(
            run_id=run_id,
            agent_id=agent_id,
            role=role,
            objective=objective,
            idempotency_key=idempotency_key,
            context_hash=context_hash(context),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "objective": self.objective,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_ids": list(self.child_agent_ids),
            "attempt": self.attempt,
            "idempotency_key": self.idempotency_key,
            "context_hash": self.context_hash,
            "capabilities": list(self.capabilities),
            "write_scope": list(self.write_scope),
            "budget": self.budget.to_dict(),
            "context_refs": list(self.context_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentSpec":
        raw = dict(data)
        return cls(
            run_id=str(raw.get("run_id") or ""),
            agent_id=str(raw.get("agent_id") or ""),
            role=str(raw.get("role") or ""),
            objective=str(raw.get("objective") or ""),
            parent_agent_id=(
                str(raw["parent_agent_id"])
                if raw.get("parent_agent_id") is not None
                else None
            ),
            child_agent_ids=tuple(raw.get("child_agent_ids") or ()),
            attempt=int(raw.get("attempt") or 1),
            idempotency_key=str(raw.get("idempotency_key") or ""),
            context_hash=str(raw.get("context_hash") or ""),
            capabilities=tuple(raw.get("capabilities") or ()),
            write_scope=tuple(raw.get("write_scope") or ()),
            budget=Budget.from_dict(raw.get("budget")),
            context_refs=tuple(raw.get("context_refs") or ()),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentHandle:
    """Durable controller view of a child-agent attempt."""

    run_id: str
    agent_id: str
    state: AgentState
    idempotency_key: str
    context_hash: str
    parent_agent_id: str | None = None
    child_agent_ids: tuple[str, ...] = ()
    attempt: int = 1
    role: str = "agent"
    capabilities: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()
    budget: Budget = field(default_factory=Budget)
    backend: str = ""
    backend_handle: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_event_cursor: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = "child_agent_handle.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(self, "agent_id", _require_text(self.agent_id, "agent_id"))
        object.__setattr__(self, "state", coerce_agent_state(self.state))
        parent = str(self.parent_agent_id).strip() if self.parent_agent_id is not None else None
        children = _string_tuple(self.child_agent_ids, "child_agent_ids")
        object.__setattr__(self, "parent_agent_id", parent)
        object.__setattr__(self, "child_agent_ids", children)
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(self, "role", _require_text(self.role, "role"))
        object.__setattr__(self, "capabilities", _string_tuple(self.capabilities, "capabilities"))
        object.__setattr__(self, "write_scope", _string_tuple(self.write_scope, "write_scope"))
        object.__setattr__(self, "last_event_cursor", int(self.last_event_cursor))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if not isinstance(self.budget, Budget):
            raise AgentContractError("budget must be a Budget")
        if self.last_event_cursor < 0:
            raise AgentContractError("last_event_cursor must be >= 0")
        _validate_identity(
            run_id=self.run_id,
            agent_id=self.agent_id,
            parent_agent_id=parent,
            child_agent_ids=children,
            attempt=self.attempt,
            idempotency_key=self.idempotency_key,
            context_hash_value=self.context_hash,
        )

    @classmethod
    def from_spec(
        cls,
        spec: AgentSpec,
        *,
        backend: str = "",
        backend_handle: str = "",
        now: str | None = None,
    ) -> "AgentHandle":
        timestamp = now or utc_now()
        return cls(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            child_agent_ids=spec.child_agent_ids,
            attempt=spec.attempt,
            idempotency_key=spec.idempotency_key,
            context_hash=spec.context_hash,
            role=spec.role,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.PENDING,
            backend=backend,
            backend_handle=backend_handle,
            created_at=timestamp,
            updated_at=timestamp,
            metadata=dict(spec.metadata),
        )

    def transition(
        self,
        target: AgentState | str,
        *,
        event_cursor: int | None = None,
        updated_at: str | None = None,
        backend_handle: str | None = None,
        child_agent_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AgentHandle":
        next_state = self.state.require_transition_to(target)
        cursor = self.last_event_cursor if event_cursor is None else int(event_cursor)
        if cursor < self.last_event_cursor:
            raise AgentContractError("event_cursor cannot move backwards")
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return replace(
            self,
            state=next_state,
            last_event_cursor=cursor,
            updated_at=updated_at or utc_now(),
            backend_handle=(self.backend_handle if backend_handle is None else backend_handle),
            child_agent_ids=(
                self.child_agent_ids
                if child_agent_ids is None
                else _string_tuple(child_agent_ids, "child_agent_ids")
            ),
            metadata=merged_metadata,
        )

    def with_event_cursor(self, cursor: int, *, updated_at: str | None = None) -> "AgentHandle":
        value = int(cursor)
        if value < self.last_event_cursor:
            raise AgentContractError("event cursor cannot move backwards")
        return replace(self, last_event_cursor=value, updated_at=updated_at or self.updated_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_ids": list(self.child_agent_ids),
            "attempt": self.attempt,
            "idempotency_key": self.idempotency_key,
            "context_hash": self.context_hash,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "write_scope": list(self.write_scope),
            "budget": self.budget.to_dict(),
            "state": self.state.value,
            "backend": self.backend,
            "backend_handle": self.backend_handle,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_event_cursor": self.last_event_cursor,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentHandle":
        raw = dict(data)
        return cls(
            run_id=str(raw.get("run_id") or ""),
            agent_id=str(raw.get("agent_id") or ""),
            parent_agent_id=(
                str(raw["parent_agent_id"])
                if raw.get("parent_agent_id") is not None
                else None
            ),
            child_agent_ids=tuple(raw.get("child_agent_ids") or ()),
            attempt=int(raw.get("attempt") or 1),
            idempotency_key=str(raw.get("idempotency_key") or ""),
            context_hash=str(raw.get("context_hash") or ""),
            role=str(raw.get("role") or "agent"),
            capabilities=tuple(raw.get("capabilities") or ()),
            write_scope=tuple(raw.get("write_scope") or ()),
            budget=Budget.from_dict(raw.get("budget")),
            state=coerce_agent_state(str(raw.get("state") or "")),
            backend=str(raw.get("backend") or ""),
            backend_handle=str(raw.get("backend_handle") or ""),
            created_at=str(raw.get("created_at") or utc_now()),
            updated_at=str(raw.get("updated_at") or utc_now()),
            last_event_cursor=int(raw.get("last_event_cursor") or 0),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentEvent:
    """One append-only lifecycle or domain event for an agent attempt."""

    run_id: str
    agent_id: str
    kind: str
    idempotency_key: str
    context_hash: str
    attempt: int = 1
    parent_agent_id: str | None = None
    child_agent_id: str | None = None
    event_id: str = field(default_factory=lambda: uuid4().hex)
    cursor: int | None = None
    from_state: AgentState | None = None
    to_state: AgentState | None = None
    occurred_at: str = field(default_factory=utc_now)
    recorded_at: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = "child_agent_event.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(self, "agent_id", _require_text(self.agent_id, "agent_id"))
        object.__setattr__(self, "kind", _require_text(self.kind, "kind"))
        object.__setattr__(self, "event_id", _require_text(self.event_id, "event_id"))
        object.__setattr__(self, "attempt", int(self.attempt))
        parent = str(self.parent_agent_id).strip() if self.parent_agent_id is not None else None
        child = str(self.child_agent_id).strip() if self.child_agent_id is not None else None
        object.__setattr__(self, "parent_agent_id", parent)
        object.__setattr__(self, "child_agent_id", child)
        if child is not None:
            _require_text(child, "child_agent_id")
            if child == self.agent_id:
                raise AgentContractError("an event child_agent_id cannot equal agent_id")
        if self.cursor is not None:
            object.__setattr__(self, "cursor", int(self.cursor))
            if self.cursor < 1:
                raise AgentContractError("event cursor must be >= 1 when set")
        from_state = coerce_agent_state(self.from_state) if self.from_state is not None else None
        to_state = coerce_agent_state(self.to_state) if self.to_state is not None else None
        object.__setattr__(self, "from_state", from_state)
        object.__setattr__(self, "to_state", to_state)
        if (from_state is None) != (to_state is None):
            raise AgentContractError("from_state and to_state must be set together")
        if from_state is not None and to_state is not None:
            from_state.require_transition_to(to_state)
        object.__setattr__(self, "payload", dict(self.payload or {}))
        _validate_identity(
            run_id=self.run_id,
            agent_id=self.agent_id,
            parent_agent_id=parent,
            child_agent_ids=tuple([child] if child else []),
            attempt=self.attempt,
            idempotency_key=self.idempotency_key,
            context_hash_value=self.context_hash,
        )

    @classmethod
    def for_transition(
        cls,
        handle: AgentHandle,
        target: AgentState | str,
        *,
        idempotency_key: str,
        kind: str = "agent.state_changed",
        child_agent_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        occurred_at: str | None = None,
        event_id: str | None = None,
    ) -> "AgentEvent":
        next_state = handle.state.require_transition_to(target)
        kwargs: dict[str, Any] = {}
        if event_id is not None:
            kwargs["event_id"] = event_id
        return cls(
            run_id=handle.run_id,
            agent_id=handle.agent_id,
            parent_agent_id=handle.parent_agent_id,
            child_agent_id=child_agent_id,
            attempt=handle.attempt,
            idempotency_key=idempotency_key,
            context_hash=handle.context_hash,
            kind=kind,
            from_state=handle.state,
            to_state=next_state,
            occurred_at=occurred_at or utc_now(),
            payload=dict(payload or {}),
            **kwargs,
        )

    def persisted(self, *, cursor: int, recorded_at: str | None = None) -> "AgentEvent":
        if self.cursor is not None and self.cursor != int(cursor):
            raise AgentContractError("cannot change an assigned event cursor")
        return replace(self, cursor=int(cursor), recorded_at=recorded_at or self.recorded_at or utc_now())

    def idempotency_scope(self) -> tuple[str, int, str]:
        return (self.agent_id, self.attempt, self.idempotency_key)

    def semantic_dict(self) -> dict[str, Any]:
        """Payload compared when an idempotency key is replayed.

        Transport-generated identity, cursor, and timestamps are excluded so a
        retried request can safely return the first durable event.
        """

        data = self.to_dict()
        for key in ("event_id", "cursor", "occurred_at", "recorded_at"):
            data.pop(key, None)
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "cursor": self.cursor,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_id": self.child_agent_id,
            "attempt": self.attempt,
            "idempotency_key": self.idempotency_key,
            "context_hash": self.context_hash,
            "kind": self.kind,
            "from_state": self.from_state.value if self.from_state is not None else None,
            "to_state": self.to_state.value if self.to_state is not None else None,
            "occurred_at": self.occurred_at,
            "recorded_at": self.recorded_at,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentEvent":
        raw = dict(data)
        return cls(
            event_id=str(raw.get("event_id") or ""),
            cursor=int(raw["cursor"]) if raw.get("cursor") is not None else None,
            run_id=str(raw.get("run_id") or ""),
            agent_id=str(raw.get("agent_id") or ""),
            parent_agent_id=(
                str(raw["parent_agent_id"])
                if raw.get("parent_agent_id") is not None
                else None
            ),
            child_agent_id=(
                str(raw["child_agent_id"])
                if raw.get("child_agent_id") is not None
                else None
            ),
            attempt=int(raw.get("attempt") or 1),
            idempotency_key=str(raw.get("idempotency_key") or ""),
            context_hash=str(raw.get("context_hash") or ""),
            kind=str(raw.get("kind") or ""),
            from_state=(
                coerce_agent_state(str(raw["from_state"]))
                if raw.get("from_state") is not None
                else None
            ),
            to_state=(
                coerce_agent_state(str(raw["to_state"]))
                if raw.get("to_state") is not None
                else None
            ),
            occurred_at=str(raw.get("occurred_at") or utc_now()),
            recorded_at=(
                str(raw["recorded_at"])
                if raw.get("recorded_at") is not None
                else None
            ),
            payload=dict(raw.get("payload") or {}),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentResult:
    """Durable terminal result for exactly one child-agent attempt."""

    run_id: str
    agent_id: str
    state: AgentState
    idempotency_key: str
    context_hash: str
    attempt: int = 1
    parent_agent_id: str | None = None
    child_agent_ids: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()
    budget: Budget = field(default_factory=Budget)
    output: Any = None
    error: str = ""
    artifacts: tuple[str, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = "child_agent_result.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(self, "agent_id", _require_text(self.agent_id, "agent_id"))
        state = coerce_agent_state(self.state)
        object.__setattr__(self, "state", state)
        if not state.is_terminal:
            raise AgentContractError("AgentResult state must be terminal")
        parent = str(self.parent_agent_id).strip() if self.parent_agent_id is not None else None
        children = _string_tuple(self.child_agent_ids, "child_agent_ids")
        object.__setattr__(self, "parent_agent_id", parent)
        object.__setattr__(self, "child_agent_ids", children)
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(self, "capabilities", _string_tuple(self.capabilities, "capabilities"))
        object.__setattr__(self, "write_scope", _string_tuple(self.write_scope, "write_scope"))
        object.__setattr__(self, "artifacts", _string_tuple(self.artifacts, "artifacts"))
        object.__setattr__(self, "usage", dict(self.usage or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if not isinstance(self.budget, Budget):
            raise AgentContractError("budget must be a Budget")
        _validate_identity(
            run_id=self.run_id,
            agent_id=self.agent_id,
            parent_agent_id=parent,
            child_agent_ids=children,
            attempt=self.attempt,
            idempotency_key=self.idempotency_key,
            context_hash_value=self.context_hash,
        )

    @property
    def succeeded(self) -> bool:
        return self.state is AgentState.SUCCEEDED

    @classmethod
    def from_handle(
        cls,
        handle: AgentHandle,
        *,
        state: AgentState | str,
        output: Any = None,
        error: str = "",
        artifacts: Iterable[str] = (),
        usage: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        finished_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AgentResult":
        terminal_state = coerce_agent_state(state)
        if not terminal_state.is_terminal:
            raise AgentContractError("AgentResult state must be terminal")
        if not handle.state.is_terminal:
            raise AgentContractError("AgentResult requires a terminal AgentHandle")
        if terminal_state != handle.state:
            raise AgentContractError("AgentResult state must match AgentHandle state")
        return cls(
            run_id=handle.run_id,
            agent_id=handle.agent_id,
            parent_agent_id=handle.parent_agent_id,
            child_agent_ids=handle.child_agent_ids,
            attempt=handle.attempt,
            idempotency_key=idempotency_key or f"{handle.idempotency_key}:result",
            context_hash=handle.context_hash,
            capabilities=handle.capabilities,
            write_scope=handle.write_scope,
            budget=handle.budget,
            state=terminal_state,
            output=output,
            error=error,
            artifacts=tuple(artifacts),
            usage=dict(usage or {}),
            started_at=handle.created_at,
            finished_at=finished_at or utc_now(),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_ids": list(self.child_agent_ids),
            "attempt": self.attempt,
            "idempotency_key": self.idempotency_key,
            "context_hash": self.context_hash,
            "capabilities": list(self.capabilities),
            "write_scope": list(self.write_scope),
            "budget": self.budget.to_dict(),
            "state": self.state.value,
            "output": self.output,
            "error": self.error,
            "artifacts": list(self.artifacts),
            "usage": dict(self.usage),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
        }

    def semantic_dict(self) -> dict[str, Any]:
        """Content compared when a terminal-result request is retried."""

        data = self.to_dict()
        data.pop("finished_at", None)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentResult":
        raw = dict(data)
        return cls(
            run_id=str(raw.get("run_id") or ""),
            agent_id=str(raw.get("agent_id") or ""),
            parent_agent_id=(
                str(raw["parent_agent_id"])
                if raw.get("parent_agent_id") is not None
                else None
            ),
            child_agent_ids=tuple(raw.get("child_agent_ids") or ()),
            attempt=int(raw.get("attempt") or 1),
            idempotency_key=str(raw.get("idempotency_key") or ""),
            context_hash=str(raw.get("context_hash") or ""),
            capabilities=tuple(raw.get("capabilities") or ()),
            write_scope=tuple(raw.get("write_scope") or ()),
            budget=Budget.from_dict(raw.get("budget")),
            state=coerce_agent_state(str(raw.get("state") or "")),
            output=raw.get("output"),
            error=str(raw.get("error") or ""),
            artifacts=tuple(raw.get("artifacts") or ()),
            usage=dict(raw.get("usage") or {}),
            started_at=(str(raw["started_at"]) if raw.get("started_at") is not None else None),
            finished_at=str(raw.get("finished_at") or utc_now()),
            metadata=dict(raw.get("metadata") or {}),
        )


__all__ = [
    "ACTIVE_AGENT_STATES",
    "ALLOWED_AGENT_STATE_TRANSITIONS",
    "AgentContractError",
    "AgentEvent",
    "AgentHandle",
    "AgentResult",
    "AgentSpec",
    "AgentState",
    "Budget",
    "InvalidStateTransition",
    "TERMINAL_AGENT_STATES",
    "coerce_agent_state",
    "context_hash",
    "require_agent_state_transition",
    "utc_now",
]
