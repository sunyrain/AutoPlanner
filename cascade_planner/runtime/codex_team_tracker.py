"""Durable lifecycle tracking for one Codex coordinator and its child roles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import AgentHandle, AgentResult, AgentSpec, AgentState, Budget
from .event_store import EventStore


@dataclass(slots=True)
class CodexTeamRuntimeTracker:
    """Persist backend-neutral state around a Codex multi-agent invocation.

    Codex currently returns child spawn/completion events after the CLI process
    exits. Child transitions are therefore recorded as observed lifecycle facts,
    not as a claim that the local controller scheduled each child itself.
    """

    store: EventStore
    run_id: str
    attempt: int
    coordinator: AgentHandle
    child_roles: tuple[str, ...]
    context: dict[str, Any]
    context_ref: str
    capabilities: tuple[str, ...]
    budget: Budget

    schema_version = "codex_team_runtime_tracker.v1"

    @classmethod
    def start(
        cls,
        *,
        root: str | Path,
        run_id: str,
        coordinator_agent_id: str,
        child_roles: Iterable[str],
        objective: str,
        context: Mapping[str, Any],
        context_ref: str,
        capabilities: Iterable[str],
        budget: Budget,
    ) -> "CodexTeamRuntimeTracker":
        store = EventStore(root)
        attempt = _next_attempt(store, run_id=run_id, agent_id=coordinator_agent_id)
        roles = tuple(str(role) for role in child_roles)
        child_ids = tuple(_child_agent_id(coordinator_agent_id, role) for role in roles)
        spec = AgentSpec.from_context(
            run_id=run_id,
            agent_id=coordinator_agent_id,
            role="retrosynthesis_team_coordinator",
            objective=objective,
            context=dict(context),
            parent_agent_id=None,
            child_agent_ids=child_ids,
            attempt=attempt,
            idempotency_key=f"{run_id}:coordinator:{attempt}",
            capabilities=tuple(capabilities),
            write_scope=(),
            budget=budget,
            context_refs=(str(context_ref),),
            metadata={
                "backend": "codex_cli",
                "local_artifact_writer": "autoplanner_orchestrator",
            },
        )
        handle = AgentHandle.from_spec(spec, backend="codex_cli")
        store.write_state(handle)
        handle, _ = store.transition(
            handle,
            AgentState.STARTING,
            idempotency_key=f"{run_id}:coordinator:{attempt}:starting",
            kind="coordinator.starting",
            payload={"required_child_roles": list(roles)},
        )
        handle, _ = store.transition(
            handle,
            AgentState.RUNNING,
            idempotency_key=f"{run_id}:coordinator:{attempt}:running",
            kind="coordinator.running",
            payload={"backend": "codex_cli"},
        )
        return cls(
            store=store,
            run_id=run_id,
            attempt=attempt,
            coordinator=handle,
            child_roles=roles,
            context=dict(context),
            context_ref=str(context_ref),
            capabilities=tuple(capabilities),
            budget=budget,
        )

    def complete(self, record: Any, *, artifacts: Iterable[str] = ()) -> dict[str, Any]:
        """Record observed children and the coordinator's terminal result."""
        metadata = dict(getattr(record, "metadata", None) or {})
        observed = [dict(row) for row in metadata.get("child_agents") or [] if isinstance(row, dict)]
        child_summaries = self._record_children(observed)
        record_status = str(getattr(record, "status", "") or "")
        terminal = AgentState.SUCCEEDED if record_status == "accepted_draft" else _record_terminal_state(record_status)
        self.coordinator, _ = self.store.transition(
            self.coordinator,
            terminal,
            idempotency_key=f"{self.run_id}:coordinator:{self.attempt}:terminal",
            kind="coordinator.completed",
            payload={
                "record_status": record_status,
                "observed_child_count": len(observed),
                "required_child_count": len(self.child_roles),
            },
        )
        result = AgentResult.from_handle(
            self.coordinator,
            state=terminal,
            output={
                "record_status": record_status,
                "backend": str(getattr(record, "backend", "") or ""),
                "session_id": str(metadata.get("session_id") or ""),
                "child_agents": observed,
            },
            error="" if terminal is AgentState.SUCCEEDED else _record_error(record),
            artifacts=tuple(str(item) for item in artifacts if str(item or "").strip()),
            usage=dict(getattr(record, "usage", None) or {}),
            metadata={"event_summary": dict(metadata.get("event_summary") or {})},
        )
        self.store.write_result(result)
        return self.summary(child_summaries=child_summaries)

    def fail(self, exc: BaseException) -> dict[str, Any]:
        """Persist a raised backend failure before the caller propagates it."""
        child_summaries = self._record_children([])
        terminal = AgentState.TIMED_OUT if isinstance(exc, TimeoutError) else AgentState.FAILED
        self.coordinator, _ = self.store.transition(
            self.coordinator,
            terminal,
            idempotency_key=f"{self.run_id}:coordinator:{self.attempt}:terminal",
            kind="coordinator.failed",
            payload={"error_type": type(exc).__name__},
        )
        self.store.write_result(
            AgentResult.from_handle(
                self.coordinator,
                state=terminal,
                error=f"{type(exc).__name__}:{exc}",
            )
        )
        return self.summary(child_summaries=child_summaries)

    def summary(self, *, child_summaries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        reconciliations = self.store.reconcile_run(self.run_id)
        return {
            "schema_version": "codex_team_runtime_summary.v1",
            "run_id": self.run_id,
            "attempt": self.attempt,
            "event_store_root": str(self.store.root),
            "last_event_cursor": self.store.last_cursor(self.run_id),
            "consistent": all(row.consistent for row in reconciliations),
            "agents": [row.to_dict() for row in reconciliations],
            "children": list(child_summaries or []),
        }

    def _record_children(self, observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        observations = [dict(row) for row in observed if isinstance(row, dict)]
        claimed: set[int] = set()
        for index, role in enumerate(self.child_roles):
            matched_index = next(
                (
                    observed_index
                    for observed_index, row in enumerate(observations)
                    if observed_index not in claimed and str(row.get("role") or "") == role
                ),
                None,
            )
            if matched_index is None:
                # Compatibility for injected/test runners that predate role
                # annotations. Never use this fallback for an explicitly
                # labelled but wrong-role observation.
                matched_index = next(
                    (
                        observed_index
                        for observed_index, row in enumerate(observations)
                        if observed_index not in claimed and not str(row.get("role") or "").strip()
                    ),
                    None,
                )
            observation = observations[matched_index] if matched_index is not None else {}
            if matched_index is not None:
                claimed.add(matched_index)
            agent_id = _child_agent_id(self.coordinator.agent_id, role)
            backend_handle = str(observation.get("agent_id") or observation.get("call_id") or "")
            spec = AgentSpec.from_context(
                run_id=self.run_id,
                agent_id=agent_id,
                role=role,
                objective=f"Independent {role} review for the coordinator",
                context={**self.context, "child_role": role},
                parent_agent_id=self.coordinator.agent_id,
                child_agent_ids=(),
                attempt=self.attempt,
                idempotency_key=f"{self.run_id}:{agent_id}:{self.attempt}",
                capabilities=self.capabilities,
                write_scope=(),
                budget=Budget(
                    max_wall_time_s=self.budget.max_wall_time_s,
                    max_tool_calls=self.budget.max_tool_calls,
                    max_tokens=self.budget.max_tokens,
                    max_output_bytes=self.budget.max_output_bytes,
                    max_children=0,
                ),
                context_refs=(self.context_ref,),
                metadata={"observation_index": index, "backend": "codex_child_agent"},
            )
            handle = AgentHandle.from_spec(
                spec,
                backend="codex_child_agent",
                backend_handle=backend_handle,
            )
            self.store.write_state(handle)
            for state, suffix, kind in (
                (AgentState.STARTING, "starting", "child.spawn_observed"),
                (AgentState.RUNNING, "running", "child.running_observed"),
            ):
                handle, _ = self.store.transition(
                    handle,
                    state,
                    idempotency_key=f"{self.run_id}:{agent_id}:{self.attempt}:{suffix}",
                    kind=kind,
                    payload={"role": role, "backend_handle": backend_handle},
                )
            terminal = _child_terminal_state(observation)
            handle, _ = self.store.transition(
                handle,
                terminal,
                idempotency_key=f"{self.run_id}:{agent_id}:{self.attempt}:terminal",
                kind="child.completion_observed",
                payload={"role": role, "observation": observation},
            )
            error = "" if terminal is AgentState.SUCCEEDED else (
                "required_child_agent_not_observed"
                if not observation
                else "child_report_not_accepted"
                if observation.get("report_accepted") is False
                else f"child_status:{observation.get('status')}"
            )
            self.store.write_result(
                AgentResult.from_handle(
                    handle,
                    state=terminal,
                    output={"role": role, "observation": observation},
                    error=error,
                    metadata={"observed_after_coordinator_exit": True},
                )
            )
            summaries.append(
                {
                    "agent_id": agent_id,
                    "role": role,
                    "state": terminal.value,
                    "backend_handle": backend_handle,
                }
            )
        return summaries


def _next_attempt(store: EventStore, *, run_id: str, agent_id: str) -> int:
    attempt = 1
    while store.read_state(run_id, agent_id, attempt=attempt) is not None:
        attempt += 1
    return attempt


def _child_agent_id(coordinator_agent_id: str, role: str) -> str:
    normalized = "_".join(part for part in str(role).lower().replace("-", "_").split("_") if part)
    return f"{coordinator_agent_id}:child:{normalized or 'agent'}"


def _child_terminal_state(observation: Mapping[str, Any]) -> AgentState:
    if not observation:
        return AgentState.LOST
    status = str(observation.get("status") or "").strip().lower()
    if status in {"completed", "succeeded", "success", "accepted"}:
        return AgentState.SUCCEEDED if observation.get("report_accepted") is not False else AgentState.FAILED
    if status in {"cancelled", "canceled"}:
        return AgentState.CANCELLED
    if status in {"timed_out", "timeout"}:
        return AgentState.TIMED_OUT
    return AgentState.FAILED


def _record_terminal_state(status: str) -> AgentState:
    value = str(status or "").strip().lower()
    if value in {"timeout", "timed_out"}:
        return AgentState.TIMED_OUT
    if value in {"cancelled", "canceled"}:
        return AgentState.CANCELLED
    return AgentState.FAILED


def _record_error(record: Any) -> str:
    error = getattr(record, "error", None)
    if error:
        return str(error)
    validation = dict(getattr(record, "output_validation", None) or {})
    reasons = [str(item) for item in validation.get("reasons") or []]
    return ";".join(reasons) or f"worker_status:{getattr(record, 'status', '')}"


__all__ = ["CodexTeamRuntimeTracker"]
