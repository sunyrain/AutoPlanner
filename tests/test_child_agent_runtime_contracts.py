from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cascade_planner.runtime import (
    AgentContractError,
    AgentEvent,
    AgentHandle,
    AgentResult,
    AgentSpec,
    AgentState,
    Budget,
    EventStore,
    IdempotencyConflictError,
    InvalidStateTransition,
    context_hash,
)


def _spec(
    *,
    agent_id: str = "child-1",
    idempotency_key: str | None = None,
) -> AgentSpec:
    return AgentSpec.from_context(
        run_id="run-1",
        agent_id=agent_id,
        parent_agent_id="root",
        role="route_researcher",
        objective="Find a source-backed route segment",
        context={"target_smiles": "CCO", "sources": ["literature", "baseline"]},
        idempotency_key=idempotency_key or f"spawn:{agent_id}:1",
        capabilities=("web_search", "route_draft"),
        write_scope=(f"workspace/agents/{agent_id}",),
        budget=Budget(
            max_wall_time_s=120,
            max_turns=8,
            max_tool_calls=12,
            max_children=0,
        ),
        context_refs=("case/target.json",),
    )


def test_agent_state_machine_accepts_lifecycle_and_rejects_terminal_restart() -> None:
    handle = AgentHandle.from_spec(_spec(), now="2026-07-10T00:00:00Z")

    assert handle.state is AgentState.PENDING
    starting = handle.transition(AgentState.STARTING, updated_at="2026-07-10T00:00:01Z")
    running = starting.transition("running", updated_at="2026-07-10T00:00:02Z")
    waiting = running.transition(AgentState.WAITING)
    resumed = waiting.transition(AgentState.RUNNING)
    finished = resumed.transition(AgentState.SUCCEEDED)

    assert finished.state.is_terminal
    assert not finished.state.is_active
    with pytest.raises(InvalidStateTransition):
        finished.transition(AgentState.RUNNING)
    with pytest.raises(InvalidStateTransition):
        handle.transition(AgentState.SUCCEEDED)
    with pytest.raises(InvalidStateTransition):
        running.transition(AgentState.RUNNING)


def test_contract_roundtrip_preserves_identity_budget_capability_and_scope() -> None:
    spec = _spec()

    assert AgentSpec.from_dict(spec.to_dict()) == spec
    assert Budget.from_dict(spec.budget.to_dict()) == spec.budget
    assert spec.parent_agent_id == "root"
    assert spec.capabilities == ("web_search", "route_draft")
    assert spec.write_scope == ("workspace/agents/child-1",)
    assert context_hash({"b": 2, "a": 1}) == context_hash({"a": 1, "b": 2})

    with pytest.raises(AgentContractError):
        AgentResult.from_handle(
            AgentHandle.from_spec(spec),
            state=AgentState.RUNNING,
        )


def test_event_cursor_is_run_global_filterable_and_idempotent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "runtime")
    first_handle = AgentHandle.from_spec(_spec())
    second_handle = AgentHandle.from_spec(_spec(agent_id="child-2"))

    first_request = AgentEvent.for_transition(
        first_handle,
        AgentState.STARTING,
        idempotency_key="child-1:starting",
        event_id="event-1",
        occurred_at="2026-07-10T00:00:01Z",
    )
    first = store.append_event(first_request)
    second = store.append_event(
        AgentEvent.for_transition(
            second_handle,
            AgentState.STARTING,
            idempotency_key="child-2:starting",
            event_id="event-2",
            occurred_at="2026-07-10T00:00:02Z",
        )
    )
    starting = first_handle.transition(AgentState.STARTING, event_cursor=first.cursor)
    third = store.append_event(
        AgentEvent.for_transition(
            starting,
            AgentState.RUNNING,
            idempotency_key="child-1:running",
            event_id="event-3",
            occurred_at="2026-07-10T00:00:03Z",
        )
    )

    assert first_request.cursor is None
    assert [first.cursor, second.cursor, third.cursor] == [1, 2, 3]
    assert store.last_cursor("run-1") == 3
    assert [event.cursor for event in store.read_events("run-1", after_cursor=1)] == [2, 3]
    assert [
        event.cursor
        for event in store.read_agent_events("run-1", "child-1", after_cursor=1)
    ] == [3]

    replay = store.append_event(
        AgentEvent.for_transition(
            first_handle,
            AgentState.STARTING,
            idempotency_key="child-1:starting",
        )
    )
    assert replay == first
    assert store.last_cursor("run-1") == 3

    with pytest.raises(IdempotencyConflictError):
        store.append_event(
            AgentEvent.for_transition(
                first_handle,
                AgentState.STARTING,
                idempotency_key="child-1:starting",
                kind="agent.backend_started",
            )
        )


def test_append_repairs_a_crash_truncated_jsonl_tail(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "runtime")
    pending = AgentHandle.from_spec(_spec())
    first = store.append_event(
        AgentEvent.for_transition(
            pending,
            AgentState.STARTING,
            idempotency_key="tail:starting",
        )
    )
    event_path = next((tmp_path / "runtime").rglob("events.jsonl"))
    with event_path.open("ab") as stream:
        stream.write(b'{"crash_truncated":')

    starting = pending.transition(AgentState.STARTING, event_cursor=first.cursor)
    second = store.append_event(
        AgentEvent.for_transition(
            starting,
            AgentState.RUNNING,
            idempotency_key="tail:running",
        )
    )

    assert second.cursor == 2
    assert [event.cursor for event in store.read_events("run-1")] == [1, 2]


def test_state_event_requires_both_sides_of_a_transition() -> None:
    with pytest.raises(AgentContractError, match="must be set together"):
        AgentEvent(
            run_id="run-1",
            agent_id="child-1",
            kind="agent.state_changed",
            idempotency_key="bad:terminal",
            context_hash=context_hash({"target": "CCO"}),
            to_state=AgentState.SUCCEEDED,
        )


def test_atomic_state_and_result_snapshots_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = EventStore(tmp_path / "runtime")
    pending = AgentHandle.from_spec(_spec(), backend="codex_child_agent")
    store.write_state(pending)

    assert store.read_state("run-1", "child-1") == pending

    starting = pending.transition(AgentState.STARTING, event_cursor=1)

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr("cascade_planner.runtime.event_store.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated interrupted replace"):
        store.write_state(starting)

    assert store.read_state("run-1", "child-1") == pending
    assert not list((tmp_path / "runtime").rglob("*.tmp"))

    monkeypatch.undo()
    running = starting.transition(AgentState.RUNNING, event_cursor=2)
    succeeded = running.transition(AgentState.SUCCEEDED, event_cursor=3)
    store.write_state(succeeded)
    result = AgentResult.from_handle(
        succeeded,
        state=AgentState.SUCCEEDED,
        output={"summary": "source-backed route segment", "route_ids": ["route-1"]},
        artifacts=("workspace/agents/child-1/result.json",),
        usage={"turns": 4, "tool_calls": 7},
        finished_at="2026-07-10T00:01:00Z",
    )

    assert store.write_result(result) == result
    assert store.write_result(result) == result
    assert store.write_result(replace(result, finished_at="2026-07-10T00:02:00Z")) == result
    assert store.read_result("run-1", "child-1") == result
    with pytest.raises(IdempotencyConflictError):
        store.write_result(replace(result, output={"summary": "different"}))


def test_flat_snapshot_layout_survives_when_legacy_temp_path_exceeds_windows_max_path(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    index = 0
    while len(str(runtime_root)) < 174:
        runtime_root /= f"f{index:02d}"
        index += 1

    store = EventStore(runtime_root)
    pending = AgentHandle.from_spec(_spec(), backend="codex_child_agent")
    run_digest = "r" * 24
    agent_digest = "a" * 24
    legacy_temp_path = (
        runtime_root
        / "runs"
        / run_digest
        / "agents"
        / agent_digest
        / "attempts"
        / "1"
        / ".state.json.12345678.tmp"
    )
    current_state_path = store._state_path(pending.run_id, pending.agent_id, pending.attempt)
    current_temp_path = current_state_path.parent / f".{current_state_path.name}.12345678.tmp"

    assert len(str(legacy_temp_path)) > 260
    assert len(str(current_temp_path)) < 260
    assert len(current_state_path.relative_to(runtime_root).parts) == 2

    store.write_state(pending)
    running, _ = store.transition(
        pending,
        AgentState.STARTING,
        idempotency_key="long-path:starting",
    )

    assert store.read_state("run-1", "child-1") == running
    assert store.list_agent_ids("run-1") == ["child-1"]


def test_transition_convenience_and_reconciliation_agree(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "runtime")
    handle = AgentHandle.from_spec(_spec())
    store.write_state(handle)

    handle, _ = store.transition(
        handle,
        AgentState.STARTING,
        idempotency_key="transition:starting",
    )
    handle, _ = store.transition(
        handle,
        AgentState.RUNNING,
        idempotency_key="transition:running",
    )
    handle, terminal_event = store.transition(
        handle,
        AgentState.SUCCEEDED,
        idempotency_key="transition:succeeded",
    )
    result = AgentResult.from_handle(
        handle,
        state=AgentState.SUCCEEDED,
        output={"ok": True},
    )
    store.write_result(result)

    reconciliation = store.reconcile_agent("run-1", "child-1")
    assert terminal_event.cursor == 3
    assert reconciliation.consistent
    assert reconciliation.authoritative_state is AgentState.SUCCEEDED
    assert reconciliation.latest_cursor == 3
    assert store.list_agent_ids("run-1") == ["child-1"]
