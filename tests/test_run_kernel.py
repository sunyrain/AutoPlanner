from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pytest

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import (
    Deficit,
    RunKernel,
    RunKernelBudgetError,
    RunKernelCorruptionError,
    RunKernelIdempotencyConflict,
    RunKernelError,
    RunLimits,
    RunRevision,
    RunSpec,
)


def _spec(
    *,
    run_id: str = "run-1",
    attempts: int = 12,
    accepted_expansions: int = 8,
    model_invocations: int = 3,
    output_tokens: int = 200_000,
    total_tasks: int = 256,
) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        target_name="target",
        target_smiles="CCO",
        created_at="2026-07-13T00:00:00Z",
        limits=RunLimits(
            model=RetrosynthesisRunBudget(
                max_model_invocations=model_invocations,
                max_total_output_tokens=output_tokens,
                max_accepted_expansions=accepted_expansions,
                max_attempt_runs=attempts,
            ),
            max_total_tasks=total_tasks,
        ),
    )


def _kernel(tmp_path: Path, **kwargs) -> RunKernel:
    return RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=_spec(**kwargs),
    )


def test_kernel_retries_transient_windows_atomic_replace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise PermissionError("transient reader lock")
        real_replace(source, destination)

    monkeypatch.setattr(
        "cascade_planner.application.run_kernel.os.replace",
        flaky_replace,
    )
    kernel = _kernel(tmp_path)

    assert kernel.state.run_id == "run-1"
    assert attempts >= 3


def test_kernel_creates_one_event_chain_snapshot_and_index(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()

    state = kernel.state
    recovery = kernel.recover()

    assert state.status == "running"
    assert state.event_count == 2
    assert recovery["replayed_state_sha256"] == kernel.state.to_dict()[
        "content_sha256"
    ]
    assert recovery["semantics"]["events_are_operational_authority"] is True
    assert kernel.index.health()["run_count"] == 1
    assert kernel.index.artifacts_for_run("run-1")[-1]["revision"] == 2


def test_accepted_expansions_are_unique_and_attempts_are_independent(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path, attempts=4, accepted_expansions=2)
    kernel.start()
    for number in (1, 2):
        kernel.reserve_task(
            task_id=f"proposal-{number}",
            kind="proposal",
            idempotency_key=f"reserve-{number}",
            input_revision=0,
        )
        kernel.settle_task(
            task_id=f"proposal-{number}",
            idempotency_key=f"settle-{number}",
            status="accepted",
            accepted_expansion_ids=["child:shared", "child:shared"],
        )

    state = kernel.state

    assert state.attempt_count == 2
    assert state.accepted_expansion_count == 1
    assert state.accepted_expansion_ids == ("child:shared",)


def test_accepted_expansion_cap_does_not_block_validation(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path, accepted_expansions=1)
    kernel.start()
    kernel.reserve_task(
        task_id="proposal",
        kind="proposal",
        idempotency_key="reserve-proposal",
        input_revision=0,
    )
    kernel.settle_task(
        task_id="proposal",
        idempotency_key="settle-proposal",
        status="accepted",
        accepted_expansion_ids=["child:one"],
    )

    kernel.reserve_task(
        task_id="validate",
        kind="validation",
        idempotency_key="reserve-validation",
        input_revision=0,
    )
    with pytest.raises(
        RunKernelBudgetError,
        match="accepted_expansion_budget_exhausted",
    ):
        kernel.reserve_task(
            task_id="proposal-2",
            kind="proposal",
            idempotency_key="reserve-proposal-2",
            input_revision=0,
        )


def test_event_idempotency_prevents_double_count_and_conflicts(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    first = kernel.reserve_task(
        task_id="evidence",
        kind="evidence",
        idempotency_key="reserve-evidence",
        input_revision=0,
    )
    repeated = kernel.reserve_task(
        task_id="evidence",
        kind="evidence",
        idempotency_key="reserve-evidence",
        input_revision=0,
    )

    assert repeated.event_id == first.event_id
    with pytest.raises(RunKernelIdempotencyConflict):
        kernel.reserve_task(
            task_id="different",
            kind="evidence",
            idempotency_key="reserve-evidence",
            input_revision=0,
        )

    settled = kernel.settle_task(
        task_id="evidence",
        idempotency_key="settle-evidence",
        status="completed",
    )
    replayed = kernel.settle_task(
        task_id="evidence",
        idempotency_key="settle-evidence",
        status="completed",
    )
    assert replayed.event_id == settled.event_id
    assert kernel.state.attempt_count == 1


def test_stop_decision_requires_bound_acceptance_report(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, model_invocations=0)
    kernel.start()
    unresolved = kernel.decide_stop()
    assert unresolved.decision == "unresolved"

    kernel.replace_deficits(
        [
            {
                "deficit_id": "evidence:1",
                "kind": "exact_evidence",
                "priority": 10,
                "deterministic": True,
                "model_allowed": False,
            }
        ],
        source_revision=0,
        idempotency_key="deficits-0",
    )
    assert kernel.decide_stop().decision == "continue"
    kernel.record_acceptance(
        {"accepted": True, "graph_revision": 0, "complete_route_count": 2},
        idempotency_key="acceptance-0",
    )

    decision = kernel.apply_stop_decision(idempotency_key="stop-complete")

    assert decision.decision == "completed"
    assert kernel.state.status == "completed"
    assert decision.to_dict()["semantics"][
        "only_acceptance_can_decide_completed"
    ] is True


def test_acceptance_cannot_hide_observed_model_budget_overrun(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, model_invocations=1, output_tokens=5)
    kernel.start()
    kernel.reserve_task(
        task_id="model-overrun",
        kind="model",
        idempotency_key="reserve-model-overrun",
        input_revision=0,
        uses_model=True,
    )
    kernel.settle_task(
        task_id="model-overrun",
        idempotency_key="settle-model-overrun",
        status="completed",
        model_usage={"model_invocations": 1, "output_tokens": 6},
    )
    kernel.record_acceptance(
        {"accepted": True, "graph_revision": 0},
        idempotency_key="accepted-despite-overrun",
    )

    decision = kernel.decide_stop()

    assert decision.decision == "budget_exhausted"
    assert decision.reasons == ("run_output_token_budget_violated",)


def test_graph_revision_invalidates_stale_acceptance(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    kernel.record_acceptance(
        {"accepted": True, "graph_revision": 0},
        idempotency_key="acceptance-0",
    )
    kernel.publish_graph_revision(
        1,
        graph_sha256="graph-1",
        evidence_revision=1,
        idempotency_key="graph-1",
    )

    assert kernel.state.acceptance_report == {}
    with pytest.raises(RunKernelError, match="acceptance_graph_revision_mismatch"):
        kernel.record_acceptance(
            {"accepted": True, "graph_revision": 0},
            idempotency_key="stale-acceptance",
        )


def test_recovery_repairs_partial_tail_and_corrupt_snapshot(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    kernel.reserve_task(
        task_id="in-flight",
        kind="evidence",
        idempotency_key="reserve-in-flight",
        input_revision=0,
    )
    expected_digest = kernel.state.to_dict()["content_sha256"]
    with kernel.events_path.open("ab") as handle:
        handle.write(b'{"partial":')
    kernel.snapshot_path.write_text("not json", encoding="utf-8")

    recovery = kernel.recover()

    assert recovery["repaired_tail_bytes"] > 0
    assert recovery["replayed_state_sha256"] == expected_digest
    assert "in-flight" in kernel.state.in_flight_tasks


def test_tampered_middle_event_fails_closed(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    lines = kernel.events_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["payload"]["spec_sha256"] = "tampered"
    lines[0] = json.dumps(event, sort_keys=True)
    kernel.events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RunKernelCorruptionError, match="event_chain_invalid"):
        kernel.recover()


def test_concurrent_reservations_cannot_exceed_total_budget(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path, attempts=5, total_tasks=5)
    kernel.start()

    def reserve(number: int) -> str:
        try:
            kernel.reserve_task(
                task_id=f"task-{number}",
                kind="evidence",
                idempotency_key=f"reserve-{number}",
                input_revision=0,
            )
            return "accepted"
        except RunKernelBudgetError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(20)))

    assert outcomes.count("accepted") == 5
    assert len(kernel.state.in_flight_tasks) == 5
    assert kernel.recover()["event_count"] == 7


def test_pause_resume_and_wall_budget_are_kernel_owned(tmp_path: Path) -> None:
    spec = _spec(run_id="paused-run")
    spec = RunSpec(
        run_id=spec.run_id,
        target_name=spec.target_name,
        target_smiles=spec.target_smiles,
        created_at=spec.created_at,
        limits=RunLimits(
            model=spec.limits.model,
            max_total_tasks=spec.limits.max_total_tasks,
            max_run_wall_time_s=1.0,
        ),
    )
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=spec,
    )
    kernel.start()
    kernel.pause()
    assert kernel.decide_stop().decision == "paused"
    with pytest.raises(RunKernelError, match="non_running"):
        kernel.reserve_task(
            task_id="paused-task",
            kind="evidence",
            idempotency_key="reserve-paused",
            input_revision=0,
        )
    kernel.resume()
    kernel.replace_deficits(
        [
            {
                "deficit_id": "proof:1",
                "kind": "exact_evidence",
                "priority": 1,
                "deterministic": True,
            }
        ],
        source_revision=0,
        idempotency_key="deficits",
    )
    kernel.reserve_task(
        task_id="timed-task",
        kind="evidence",
        idempotency_key="reserve-timed",
        input_revision=0,
    )
    kernel.settle_task(
        task_id="timed-task",
        idempotency_key="settle-timed",
        status="completed",
        elapsed_s=1.0,
    )

    assert kernel.state.task_wall_time_s == 1.0
    assert kernel.decide_stop().decision == "budget_exhausted"


def test_revision_and_deficit_contracts_are_digest_bound(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    deficit = Deficit(
        deficit_id="evidence:edge-1",
        kind="exact_evidence",
        source_revision=0,
        priority=7.5,
        deterministic=True,
        model_allowed=False,
        entity_refs=("edge:1", "edge:1"),
        metadata={"source_refs": ["patent:WO1"]},
    )
    kernel.replace_deficits(
        [deficit],
        source_revision=0,
        idempotency_key="deficit-contract",
    )

    revision = kernel.revision

    assert isinstance(revision, RunRevision)
    assert revision.state_sha256 == kernel.state.to_dict()["content_sha256"]
    assert revision.to_dict()["schema_version"] == "autoplanner_run_revision.v1"
    assert kernel.state.deficits[0]["source_revision"] == 0
    assert kernel.state.deficits[0]["metadata"]["source_refs"] == ["patent:WO1"]


def test_model_boundary_enforces_context_visual_and_finite_usage(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, model_invocations=2)
    kernel.start()
    with pytest.raises(RunKernelBudgetError, match="prompt_context_byte"):
        kernel.reserve_task(
            task_id="oversized",
            kind="model",
            idempotency_key="reserve-oversized",
            input_revision=0,
            uses_model=True,
            prompt_context_bytes=100_000,
        )
    kernel.reserve_task(
        task_id="director",
        kind="model",
        idempotency_key="reserve-director",
        input_revision=0,
        uses_model=True,
        visual=True,
        prompt_context_bytes=1_024,
    )
    with pytest.raises(ValueError, match="finite"):
        kernel.settle_task(
            task_id="director",
            idempotency_key="settle-director",
            status="failed",
            model_usage={"model_invocations": 1, "wall_time_s": float("nan")},
        )
