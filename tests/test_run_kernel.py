from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
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
from cascade_planner.application.unified_campaign_spec import (
    CampaignResourceBudget,
    StockOracleReference,
    UnifiedCampaignSpec,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _spec(
    *,
    run_id: str = "run-1",
    attempts: int = 12,
    accepted_expansions: int = 8,
    model_invocations: int = 3,
    output_tokens: int = 200_000,
    total_tasks: int = 256,
    native_total: int | None = None,
    target_native_minimum: int | None = None,
    frontier_native_limit: int | None = None,
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
                max_native_search_invocations=native_total,
                min_target_native_search_invocations=target_native_minimum,
                max_frontier_native_search_invocations=frontier_native_limit,
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


def test_new_run_spec_embeds_unified_campaign_contract() -> None:
    spec = _spec()

    row = spec.to_dict()

    assert row["schema_version"] == "autoplanner_run_spec.v2"
    assert row["campaign_spec"]["target"] == {"canonical_smiles": "CCO"}
    assert "target_name" not in row["campaign_spec"]
    assert "acceptance" not in row["campaign_spec"]
    assert RunSpec.from_dict(row).to_dict() == row


def test_program_and_experiment_task_budgets_are_independent_and_replayable(
    tmp_path: Path,
) -> None:
    limits = RunLimits()
    campaign_budget = replace(
        CampaignResourceBudget.from_dict(limits.to_dict()),
        max_program_tasks=1,
        max_experiment_tasks=1,
    )
    spec = RunSpec(
        run_id="task-dimensions",
        target_name="target",
        target_smiles="CCO",
        limits=limits,
        campaign_spec=UnifiedCampaignSpec(
            target_smiles="CCO",
            stock_oracle=StockOracleReference.compatibility_unbound(
                boundary="procurement"
            ),
            resource_budget=campaign_budget,
        ),
    )
    kernel = RunKernel(tmp_path / "runtime", tmp_path / "run", spec=spec)
    kernel.start()

    for kind in ("program", "experiment"):
        kernel.reserve_task(
            task_id=f"{kind}:1",
            kind=kind,
            idempotency_key=f"{kind}:reserve:1",
            input_revision=0,
        )
        kernel.settle_task(
            task_id=f"{kind}:1",
            idempotency_key=f"{kind}:settle:1",
            status="completed",
        )
        with pytest.raises(
            RunKernelBudgetError,
            match=f"run_{kind}_task_budget_exhausted",
        ):
            kernel.reserve_task(
                task_id=f"{kind}:2",
                kind=kind,
                idempotency_key=f"{kind}:reserve:2",
                input_revision=0,
            )

    projection = kernel.task_budget()["dimensions"]
    assert projection["program"]["settled"] == 1
    assert projection["program"]["available"] is False
    assert projection["experiment"]["settled"] == 1
    assert projection["experiment"]["available"] is False
    reopened = RunKernel(tmp_path / "runtime", tmp_path / "run")
    assert reopened.task_budget() == kernel.task_budget()


def test_legacy_run_spec_v1_digest_and_projection_remain_readable() -> None:
    current = _spec()
    row = {
        "schema_version": "autoplanner_run_spec.v1",
        "run_id": current.run_id,
        "target_name": current.target_name,
        "target_smiles": current.target_smiles,
        "acceptance": current.acceptance.to_dict(),
        "limits": current.limits.to_dict(),
        "producer": current.producer,
        "created_at": current.created_at,
        "semantics": {
            "one_kernel_per_run": True,
            "workers_cannot_extend_limits": True,
            "acceptance_is_scientific_completion_authority": True,
        },
    }
    row["content_sha256"] = _digest(row)

    restored = RunSpec.from_dict(row)

    assert restored.to_dict() == row
    assert restored.campaign_spec.target_smiles == "CCO"
    assert restored.campaign_spec.stock_oracle.binding["positive_authority"] is False


def test_explicit_model_budget_extension_is_durable_and_preserves_usage(
    tmp_path: Path,
) -> None:
    spec = _spec(model_invocations=1)
    spec = replace(
        spec,
        campaign_spec=replace(
            spec.campaign_spec,
            resource_budget=replace(
                spec.campaign_spec.resource_budget,
                max_program_tasks=7,
                max_experiment_tasks=3,
            ),
        ),
    )
    kernel = RunKernel(tmp_path / "runtime", tmp_path / "run", spec=spec)
    kernel.start()
    kernel.reserve_task(
        task_id="model-1",
        kind="model",
        idempotency_key="reserve-model-1",
        input_revision=0,
        uses_model=True,
    )
    kernel.settle_task(
        task_id="model-1",
        idempotency_key="settle-model-1",
        status="completed",
        model_usage={"model_invocations": 1, "input_tokens": 50},
    )
    with pytest.raises(
        RunKernelBudgetError,
        match="run_model_invocation_budget_exhausted",
    ):
        kernel.reserve_task(
            task_id="model-2",
            kind="model",
            idempotency_key="reserve-model-2-before-extension",
            input_revision=0,
            uses_model=True,
        )

    extended = replace(kernel.spec.limits.model, max_model_invocations=2)
    event = kernel.extend_model_budget(
        extended,
        idempotency_key="operator-extends-model-budget-to-2",
    )

    assert event is not None
    assert event.event_type == "model_budget_extended"
    assert kernel.spec.limits.model.max_model_invocations == 2
    assert kernel.state.model_totals["model_invocations"] == 1
    kernel.reserve_task(
        task_id="model-2",
        kind="model",
        idempotency_key="reserve-model-2",
        input_revision=0,
        uses_model=True,
    )

    reopened = RunKernel(tmp_path / "runtime", tmp_path / "run")
    assert reopened.spec.limits.model.max_model_invocations == 2
    assert reopened.spec.campaign_spec.resource_budget.max_program_tasks == 7
    assert reopened.spec.campaign_spec.resource_budget.max_experiment_tasks == 3
    assert reopened.state.model_totals["model_invocations"] == 1
    assert reopened.task_lifecycle("model-2")["status"] == "in_flight"


def test_model_budget_extension_cannot_reduce_existing_limits(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path, model_invocations=2)

    with pytest.raises(
        RunKernelBudgetError,
        match="run_model_budget_extension_cannot_decrease_or_change_policy",
    ):
        kernel.extend_model_budget(
            replace(kernel.spec.limits.model, max_model_invocations=1),
            idempotency_key="invalid-budget-reduction",
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


def test_kernel_retries_transient_windows_writer_lock_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    real_mkdir = Path.mkdir
    attempts = 0

    def flaky_mkdir(path: Path, *args, **kwargs) -> None:
        nonlocal attempts
        if path == kernel.lock_path and attempts < 2:
            attempts += 1
            raise PermissionError("transient writer lock race")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)

    kernel.reserve_task(
        task_id="windows-contention",
        kind="evidence",
        idempotency_key="reserve-windows-contention",
        input_revision=0,
    )

    assert attempts == 2
    assert kernel.task_lifecycle("windows-contention")["status"] == "in_flight"


def test_kernel_preserves_genuine_writer_lock_permission_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=_spec(),
        lock_timeout_s=0.1,
    )
    kernel.start()
    real_mkdir = Path.mkdir

    def denied_mkdir(path: Path, *args, **kwargs) -> None:
        if path == kernel.lock_path:
            raise PermissionError("writer lock ACL denied")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied_mkdir)

    with pytest.raises(PermissionError, match="writer lock ACL denied"):
        kernel.reserve_task(
            task_id="permission-denied",
            kind="evidence",
            idempotency_key="reserve-permission-denied",
            input_revision=0,
        )


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


def test_terminal_run_reopens_only_for_explicitly_bound_new_work(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    kernel.transition(
        "completed",
        idempotency_key="complete-before-external-feedback",
    )
    before = kernel.state

    event = kernel.reopen_for_new_work(
        work_fingerprint="feedback-sha256",
        reasons=("new_program_validation_feedback",),
        idempotency_key="reopen-for-feedback-sha256",
    )

    assert event.event_type == "run_reopened"
    assert event.payload["from_status"] == "completed"
    assert kernel.state.status == "running"
    assert kernel.state.attempt_count == before.attempt_count
    assert kernel.state.accepted_expansion_ids == before.accepted_expansion_ids
    replayed_event = kernel.reopen_for_new_work(
        work_fingerprint="feedback-sha256",
        reasons=("new_program_validation_feedback",),
        idempotency_key="reopen-for-feedback-sha256",
    )
    assert replayed_event.event_id == event.event_id

    replayed = RunKernel(tmp_path / "runtime", tmp_path / "run")
    assert replayed.state.status == "running"
    with pytest.raises(RunKernelError, match="run_status_reopen_invalid"):
        replayed.reopen_for_new_work(
            work_fingerprint="other-work",
            idempotency_key="cannot-reopen-running-run",
        )


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
    assert kernel.task_lifecycle("evidence")["status"] == "absent"
    first = kernel.reserve_task(
        task_id="evidence",
        kind="evidence",
        idempotency_key="reserve-evidence",
        input_revision=0,
    )
    lifecycle = kernel.task_lifecycle("evidence")
    assert lifecycle["status"] == "in_flight"
    assert lifecycle["reservation"]["event_id"] == first.event_id
    assert lifecycle["semantics"]["projection_grants_no_scientific_authority"] is True
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
    lifecycle = kernel.task_lifecycle("evidence")
    assert lifecycle["status"] == "settled"
    assert lifecycle["settlement"]["event_id"] == settled.event_id
    assert kernel.state.attempt_count == 0


def test_task_checkpoints_are_cas_bound_ordered_and_replayable(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    kernel.reserve_task(
        task_id="experiment", kind="experiment",
        idempotency_key="reserve-experiment", input_revision=0,
    )
    first_ref = kernel.artifacts.put_json(
        {"status": "submitted"}, logical_name="submitted.json", producer="test"
    )

    def record_first():
        return kernel.record_task_checkpoint(
            task_id="experiment", checkpoint_kind="external_job",
            artifact_ref=first_ref, predecessor_checkpoint_sha256="",
            operational_status="submitted", idempotency_key="checkpoint:first",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        first_events = list(pool.map(lambda _: record_first(), range(4)))
    assert len({event.event_id for event in first_events}) == 1
    second_ref = kernel.artifacts.put_json(
        {"status": "running"}, logical_name="running.json", producer="test"
    )
    with pytest.raises(RunKernelError, match="task_checkpoint_binding_invalid"):
        kernel.record_task_checkpoint(
            task_id="experiment", checkpoint_kind="external_job",
            artifact_ref=second_ref, predecessor_checkpoint_sha256="",
            operational_status="running", idempotency_key="checkpoint:wrong",
        )
    second = kernel.record_task_checkpoint(
        task_id="experiment", checkpoint_kind="external_job",
        artifact_ref=second_ref, predecessor_checkpoint_sha256=first_ref.sha256,
        operational_status="running", idempotency_key="checkpoint:second",
    )
    lifecycle = kernel.task_lifecycle("experiment")
    assert [
        row["payload"]["artifact_sha256"] for row in lifecycle["checkpoints"]
    ] == [first_ref.sha256, second_ref.sha256]
    assert lifecycle["semantics"][
        "checkpoints_are_operational_observations_only"
    ] is True
    assert kernel.state.graph_revision == 0
    assert kernel.state.task_checkpoints["experiment"][-1][
        "artifact_sha256"
    ] == second_ref.sha256
    reopened = RunKernel(tmp_path / "runtime", tmp_path / "run")
    assert reopened.task_lifecycle("experiment")["checkpoints"][-1][
        "event_id"
    ] == second.event_id
    reopened.settle_task(
        task_id="experiment", idempotency_key="settle-experiment",
        status="completed",
    )
    with pytest.raises(RunKernelError, match="task_not_reserved"):
        reopened.record_task_checkpoint(
            task_id="experiment", checkpoint_kind="external_job",
            artifact_ref=second_ref, predecessor_checkpoint_sha256=second_ref.sha256,
            operational_status="completed", idempotency_key="checkpoint:late",
        )


def test_concurrent_different_task_checkpoint_payloads_fail_closed(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    kernel.start()
    kernel.reserve_task(
        task_id="experiment-race", kind="experiment",
        idempotency_key="reserve-experiment-race", input_revision=0,
    )
    refs = [
        kernel.artifacts.put_json(
            {"status": status}, logical_name=f"{status}.json", producer="test"
        )
        for status in ("running", "failed")
    ]

    def record(index: int):
        try:
            return kernel.record_task_checkpoint(
                task_id="experiment-race", checkpoint_kind="external_job",
                artifact_ref=refs[index], predecessor_checkpoint_sha256="",
                operational_status=("running", "failed")[index],
                idempotency_key="checkpoint:race",
            )
        except RunKernelIdempotencyConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(record, range(2)))
    assert sum(isinstance(value, RunKernelIdempotencyConflict) for value in outcomes) == 1
    assert len(kernel.task_lifecycle("experiment-race")["checkpoints"]) == 1


def test_only_proposal_tasks_consume_attempt_budget(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, attempts=1)
    kernel.start()
    for kind in ("model", "evidence", "validation", "stock", "other"):
        kernel.reserve_task(
            task_id=kind,
            kind=kind,
            idempotency_key=f"reserve-{kind}",
            input_revision=0,
            uses_model=kind == "model",
        )
        kernel.settle_task(
            task_id=kind,
            idempotency_key=f"settle-{kind}",
            status="completed",
        )

    assert kernel.state.attempt_count == 0
    assert kernel.state.settled_task_count == 5

    kernel.reserve_task(
        task_id="proposal-1",
        kind="proposal",
        idempotency_key="reserve-proposal-1",
        input_revision=0,
    )
    kernel.settle_task(
        task_id="proposal-1",
        idempotency_key="settle-proposal-1",
        status="rejected",
    )
    assert kernel.state.attempt_count == 1

    kernel.reserve_task(
        task_id="evidence-after-proposal-cap",
        kind="evidence",
        idempotency_key="reserve-evidence-after-proposal-cap",
        input_revision=0,
    )
    with pytest.raises(RunKernelBudgetError, match="run_attempt_budget_exhausted"):
        kernel.reserve_task(
            task_id="proposal-2",
            kind="proposal",
            idempotency_key="reserve-proposal-2",
            input_revision=0,
        )


def test_attempt_cap_does_not_stop_remaining_non_proposal_work(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, attempts=1)
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
        status="rejected",
    )
    kernel.replace_deficits(
        [
            Deficit(
                deficit_id="validation:edge",
                kind="validation",
                source_revision=0,
                deterministic=True,
                model_allowed=False,
            ),
            Deficit(
                deficit_id="expansion:leaf",
                kind="expansion",
                source_revision=0,
                deterministic=False,
                model_allowed=True,
            ),
        ],
        source_revision=0,
        idempotency_key="replace-mixed-deficits",
    )

    decision = kernel.decide_stop()

    assert decision.decision == "continue"
    assert decision.terminal is False


def test_native_target_reserve_is_protected_released_and_replayable(
    tmp_path: Path,
) -> None:
    kernel = _kernel(
        tmp_path,
        native_total=3,
        target_native_minimum=2,
        frontier_native_limit=1,
    )
    kernel.start()
    kernel.reserve_task(
        task_id="frontier-1",
        kind="other",
        idempotency_key="reserve-frontier-1",
        input_revision=0,
        resource_class="native_search_frontier",
        resource_units=1,
    )
    kernel.settle_task(
        task_id="frontier-1",
        idempotency_key="settle-frontier-1",
        status="completed",
    )
    with pytest.raises(
        RunKernelBudgetError,
        match="run_native_target_reserve_protected",
    ):
        kernel.reserve_task(
            task_id="frontier-protected",
            kind="other",
            idempotency_key="reserve-frontier-protected",
            input_revision=0,
            resource_class="native_search_frontier",
            resource_units=1,
        )
    kernel.reserve_task(
        task_id="target-1",
        kind="other",
        idempotency_key="reserve-target-1",
        input_revision=0,
        resource_class="native_search_target",
        resource_units=1,
    )
    kernel.settle_task(
        task_id="target-1",
        idempotency_key="settle-target-1",
        status="completed",
    )
    kernel.release_native_target_reserve(
        units=1,
        reason="target_native_search_terminal",
        idempotency_key="release-unused-target-native",
    )
    borrowed = kernel.reserve_task(
        task_id="frontier-borrowed",
        kind="other",
        idempotency_key="reserve-frontier-borrowed",
        input_revision=0,
        resource_class="native_search_frontier",
        resource_units=1,
    )
    assert borrowed.payload["resource_reservation"]["decision"] == "borrow_granted"
    assert borrowed.payload["resource_reservation"]["borrowed_units"] == 1
    kernel.settle_task(
        task_id="frontier-borrowed",
        idempotency_key="settle-frontier-borrowed",
        status="completed",
    )

    projection = kernel.native_search_budget()
    assert projection["committed_total"] == 3
    assert projection["hard_remaining"] == 0
    assert projection["target"]["minimum_service_satisfied"] is True
    assert projection["frontier"]["borrowed_total"] == 1

    reopened = RunKernel(tmp_path / "runtime", tmp_path / "run")
    assert reopened.native_search_budget() == projection


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
