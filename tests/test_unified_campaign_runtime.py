from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event

import pytest

from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.campaign_actions import (
    CampaignActionKind,
    bind_scheduled_action,
    compile_action_opportunities,
    legacy_campaign_action_sha256,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.application.worker_runtime import WorkerBudget, WorkerCommand
from cascade_planner.interfaces.target_solver_stages import (
    discover_director_source_hints,
)
from cascade_planner.interfaces.target_solver import (
    _transition_unresolved_if_active,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.orchestration.unified_campaign_runtime import (
    CampaignActionRuntime,
    CampaignActionRuntimeError,
)


def _kernel(tmp_path: Path, *, max_total_tasks: int = 256) -> RunKernel:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="campaign-action-test",
            target_name="target",
            target_smiles="CCO",
            created_at="2026-08-06T00:00:00Z",
            limits=RunLimits(max_total_tasks=max_total_tasks),
        ),
    )
    kernel.start()
    return kernel


def _opportunity_set(*, kind: str = "materialization") -> dict:
    return compile_action_opportunities(
        {
            "content_sha256": "frontier-sha",
            "items": [
                {
                    "deficit_id": f"deficit:{kind}:1",
                    "kind": kind,
                    "object_id": "hypothesis:1" if kind == "materialization" else "edge:1",
                    "entity_ids": [
                        "hypothesis:1" if kind == "materialization" else "edge:1"
                    ],
                    "route_family_ids": ["route-family:1"],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 500.0,
                    "reason": f"test_{kind}",
                    "score": {
                        "expected_portfolio_gain": 0.8,
                        "distance_to_closure": 0.8,
                        "evidence_gain": 0.2,
                        "route_diversity_gain": 0.1,
                        "cost_penalty": 0.1,
                        "failure_risk_penalty": 0.0,
                    },
                }
            ],
        }
    )


def _decision(*, kind: str = "materialization") -> dict:
    return schedule_next_action(
        _opportunity_set(kind=kind),
        milestones={},
        resource_availability={"deterministic": True, "validation": True},
    )


def test_action_binding_is_stable_and_revision_bound() -> None:
    decision = _decision()

    first = bind_scheduled_action(decision, input_revision=3)
    repeated = bind_scheduled_action(decision, input_revision=3)
    next_revision = bind_scheduled_action(decision, input_revision=4)

    assert first == repeated
    assert first.execution_id != next_revision.execution_id
    assert first.task_id != next_revision.task_id
    assert first.idempotency_key != next_revision.idempotency_key
    assert first.to_dict()["schema_version"] == "campaign_action.v2"
    assert first.estimate["schema_version"] == "campaign_action_estimate.v1"
    assert first.estimate["success_probability"] == {
        "low": 0.0,
        "high": 1.0,
        "assessed": False,
    }
    assert first.estimate["expected_gain"]["dependency_unblock_count"] == 0


def test_action_estimate_preserves_assessed_interval_gain_and_uncertainty() -> None:
    opportunities = compile_action_opportunities(
        {
            "content_sha256": "estimate-frontier",
            "items": [
                {
                    "deficit_id": "deficit:evidence:estimate",
                    "kind": "evidence",
                    "object_id": "edge:estimate",
                    "entity_ids": ["edge:estimate"],
                    "route_family_ids": ["route:estimate"],
                    "dependency_ids": ["dep:1", "dep:2"],
                    "deterministic": False,
                    "model_allowed": False,
                    "priority": 700.0,
                    "reason": "estimate_contract",
                    "score": {
                        "success_probability_interval": [0.25, 0.75],
                        "expected_portfolio_gain": 0.4,
                        "evidence_gain": 0.8,
                        "route_diversity_gain": 0.2,
                        "dependency_unblock_count": 2,
                        "novelty_gain": 0.6,
                        "cost_penalty": 0.3,
                        "uncertainty": {"source": "calibrated_holdout"},
                    },
                    "metadata": {},
                }
            ],
        }
    )
    decision = schedule_next_action(
        opportunities,
        milestones={},
        resource_availability={"evidence": True},
    )

    action = bind_scheduled_action(decision, input_revision=5)

    assert action.estimate["success_probability"] == {
        "low": 0.25,
        "high": 0.75,
        "assessed": True,
    }
    assert action.estimate["expected_gain"] == {
        "route": 0.4,
        "proof": 0.8,
        "diversity": 0.2,
        "dependency_unblock_count": 2,
        "novelty": 0.6,
    }
    assert action.estimate["cost"]["penalty"] == 0.3
    assert action.estimate["uncertainty"]["source"] == "calibrated_holdout"
    assert action.estimate["uncertainty"]["success_probability"] == "assessed"


def test_runtime_reserves_settles_and_replays_without_double_counting(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    calls = 0

    def handle(_action) -> dict:
        nonlocal calls
        calls += 1
        return {"status": "completed", "changed": False}

    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: handle},
    )
    decision = _decision()
    action = bind_scheduled_action(decision, input_revision=kernel.state.graph_revision)

    first = runtime.execute(action, decision=decision)
    replay = runtime.execute(action, decision=decision)

    assert first["status"] == "completed"
    assert first["cache_hit"] is False
    assert replay["cache_hit"] is True
    assert calls == 1
    assert first["outcome"]["schema_version"] == "campaign_action_result.v1"
    assert len(first["outcome_ref"]["sha256"]) == 64
    assert first["outcome"]["failure_type"] == ""
    assert first["outcome"]["candidate_delta"] == {
        "proposal_count": 0,
        "candidate_count": 0,
        "accepted_count": 0,
    }
    assert first["outcome"]["fact_delta"] == {
        "graph_revision_delta": 0,
        "changed": False,
        "handler_reported_changed": False,
        "authority": "run_kernel_canonical_graph_revision",
    }
    lifecycle = kernel.task_lifecycle(action.task_id)
    assert lifecycle["reservation"]["payload"]["kind"] == "other"
    expected = lifecycle["reservation"]["payload"]["metadata"][
        "expected_resources"
    ]
    assert expected["estimated"]["task_counts"] == {"other": 1}
    assert lifecycle["status"] == "settled"
    assert lifecycle["settlement"]["payload"]["status"] == "completed"
    accounting = first["outcome"]["resource_accounting"]
    assert accounting["actual"]["task_counts"] == {"other": 1}
    assert accounting["variance"]["total_tasks"] == 0
    assert lifecycle["settlement"]["payload"]["resource_usage"] == accounting[
        "actual"
    ]
    assert kernel.state.attempt_count == 0
    assert kernel.state.model_totals["model_invocations"] == 0


def test_handler_change_claim_does_not_create_canonical_fact_delta(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": True,
            }
        },
    )
    action = bind_scheduled_action(_decision(), input_revision=0)

    result = runtime.execute(action)

    assert result["outcome"]["fact_delta"] == {
        "graph_revision_delta": 0,
        "changed": False,
        "handler_reported_changed": True,
        "authority": "run_kernel_canonical_graph_revision",
    }


@pytest.mark.parametrize(
    ("status", "failure_type", "settled_status"),
    [
        ("cancelled", "cancelled", "cancelled"),
        ("partial", "partial_failure", "partial"),
        ("timeout", "timeout", "timeout"),
        ("failed", "handler_failure", "failed"),
    ],
)
def test_failed_action_dispositions_release_owned_child_reservations(
    tmp_path: Path,
    status: str,
    failure_type: str,
    settled_status: str,
) -> None:
    kernel = _kernel(tmp_path)
    action = bind_scheduled_action(_decision(kind="evidence"), input_revision=0)
    child_task_id = f"child:{status}"

    def handle(_action) -> dict:
        kernel.reserve_task(
            task_id=child_task_id,
            kind="evidence",
            idempotency_key=f"{child_task_id}:reserve",
            input_revision=0,
        )
        return {"status": status, "reasons": [f"test_{status}"]}

    result = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.ACQUIRE_EVIDENCE: handle},
    ).execute(action)

    assert result["outcome"]["failure_type"] == failure_type
    assert result["outcome"]["failure_reasons"] == [f"test_{status}"]
    assert kernel.task_lifecycle(child_task_id)["status"] == "settled"
    assert kernel.task_lifecycle(child_task_id)["settlement"]["payload"][
        "status"
    ] == settled_status
    assert kernel.state.in_flight_tasks == {}


def test_action_child_tasks_inherit_identity_and_settle_actual_resources(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    decision = _decision(kind="evidence")
    action = bind_scheduled_action(decision, input_revision=0)
    wrapper_metadata: dict = {}

    def handle(_action) -> dict:
        wrapper_metadata.update(
            dict(kernel.state.in_flight_tasks[action.task_id].get("metadata") or {})
        )
        kernel.reserve_task(
            task_id="evidence-child",
            kind="evidence",
            idempotency_key="evidence-child:reserve",
            input_revision=0,
        )
        kernel.settle_task(
            task_id="evidence-child",
            idempotency_key="evidence-child:settle",
            status="completed",
            elapsed_s=0.25,
        )
        return {
            "status": "completed",
            "material_events": ["exact_source_added"],
            "proposals": [{"proposal_id": "proposal:1"}],
            "artifact_ref": {
                "sha256": "a" * 64,
                "size_bytes": 12,
                "media_type": "application/json",
                "logical_name": "evidence-child.json",
                "producer": "test",
            },
        }

    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.ACQUIRE_EVIDENCE: handle},
    )
    result = runtime.execute(action, decision=decision)

    child = kernel.task_lifecycle("evidence-child")
    child_metadata = child["reservation"]["payload"]["metadata"]
    assert wrapper_metadata["campaign_action_kind"] == "acquire_exact_evidence"
    assert child_metadata["campaign_action_execution_id"] == action.execution_id
    assert child_metadata["campaign_action_expected_resources_sha256"] == (
        action.expected_resources["content_sha256"]
    )
    accounting = result["outcome"]["resource_accounting"]
    assert accounting["expected"]["estimated"]["task_counts"] == {
        "evidence": 1,
        "other": 1,
    }
    assert accounting["actual"]["task_counts"] == {
        "evidence": 1,
        "other": 1,
    }
    assert accounting["actual"]["settled_task_count"] == 2
    assert accounting["actual"]["in_flight_task_count"] == 0
    assert accounting["variance"]["total_tasks"] == 0
    assert result["outcome"]["actual_resources"] == accounting["actual"]
    assert result["outcome"]["material_events"] == ["exact_source_added"]
    assert result["outcome"]["candidate_delta"]["proposal_count"] == 1
    assert result["outcome"]["immutable_artifact_refs"] == [
        {
            "sha256": "a" * 64,
            "size_bytes": 12,
            "media_type": "application/json",
            "logical_name": "evidence-child.json",
            "producer": "test",
        }
    ]


def test_runtime_accounts_target_native_search_without_cache_double_count(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunities = compile_action_opportunities(
        {
            "content_sha256": "target-native-frontier",
            "items": [
                {
                    "deficit_id": "deficit:target-native:1",
                    "kind": "expansion",
                    "object_id": "target:1",
                    "entity_ids": ["target:1"],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": False,
                    "model_allowed": False,
                    "priority": 900.0,
                    "reason": "target_requires_native_multi_step_search",
                    "score": {"expected_portfolio_gain": 1.0},
                    "metadata": {
                        "provider_preferences": ["chemenzy"],
                        "target_level_native_search": True,
                        "native_budget_reservation": "target_level",
                    },
                }
            ],
        }
    )
    decision = schedule_next_action(
        opportunities,
        milestones={},
        resource_availability={"native_search_target": True},
    )
    action = bind_scheduled_action(
        decision,
        input_revision=kernel.state.graph_revision,
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.CHEMENZY_TARGET_EXPAND: lambda _action: {
                "status": "completed",
                "proposal_count": 2,
            }
        },
    )

    first = runtime.execute(action, decision=decision)
    replay = runtime.execute(action, decision=decision)

    reservation = first["outcome"]["resource_reservation"]
    assert reservation["resource_class"] == "native_search_target"
    assert reservation["decision"] == "target_service_reserved"
    assert kernel.native_search_budget()["target"]["settled"] == 1
    assert replay["cache_hit"] is True
    assert kernel.native_search_budget()["target"]["settled"] == 1


def test_runtime_rejects_stale_revision_without_reservation(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    decision = _decision()
    action = bind_scheduled_action(decision, input_revision=kernel.state.graph_revision)
    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: lambda _action: {"status": "completed"}},
    )
    kernel.publish_graph_revision(
        1,
        graph_sha256="graph-1",
        evidence_revision=0,
        idempotency_key="publish-graph-1",
    )

    result = runtime.execute(action, decision=decision)

    assert result["status"] == "stale"
    assert result["cache_hit"] is False
    assert kernel.task_lifecycle(action.task_id)["status"] == "absent"


def test_runtime_blocks_actions_without_registered_handler(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    runtime = CampaignActionRuntime(kernel, {})

    result = runtime.schedule_and_execute(
        _opportunity_set(kind="validation"),
        milestones={},
        resource_availability={"validation": True},
    )

    assert result["status"] == "no_action"
    candidate = result["decision"]["candidates"][0]
    assert candidate["eligible"] is False
    assert candidate["blocked_reasons"] == [
        "handler_unavailable:reaction_validate"
    ]
    assert kernel.state.settled_task_count == 0


def test_anytime_loop_converges_after_bounded_no_gain_actions(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunity_set = compile_action_opportunities(
        {
            "content_sha256": "three-no-gain-actions",
            "items": [
                {
                    "deficit_id": f"deficit:materialization:{index}",
                    "kind": "materialization",
                    "object_id": f"hypothesis:{index}",
                    "entity_ids": [f"hypothesis:{index}"],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 500.0 - index,
                    "reason": "test_no_gain_materialization",
                    "score": {},
                }
                for index in range(3)
            ],
        }
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": False,
            }
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=5,
        max_consecutive_no_gain=2,
    )

    assert result["termination"] == "converged_low_marginal_gain"
    assert result["execution_count"] == 2
    assert result["unexecuted_actions"]["action_count"] == 1
    assert result["unexecuted_actions"]["actions"][0]["reasons"] == [
        "low_marginal_gain_convergence"
    ]


def test_anytime_loop_honors_explicit_kernel_cancellation_without_dispatch(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    kernel.cancel(
        idempotency_key="operator-cancel",
        reasons=("operator_requested_stop",),
    )
    calls = 0

    def handle(_action) -> dict:
        nonlocal calls
        calls += 1
        return {"status": "completed"}

    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: handle},
    )

    result = runtime.run_anytime(
        opportunity_provider=_opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=4,
    )

    assert result["termination"] == "user_cancelled"
    assert result["termination_reasons"] == ["operator_requested_stop"]
    assert result["execution_count"] == 0
    assert result["kernel_stop_decision"]["decision"] == "cancelled"
    assert result["unexecuted_actions"]["actions"][0]["reasons"] == [
        "explicit_user_cancelled",
        "operator_requested_stop",
    ]
    assert calls == 0
    assert result["semantics"]["single_scheduler_loop"] is True


def test_configured_acceptance_is_a_snapshot_and_does_not_stop_action_loop(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    calls = 0

    def handle(_action) -> dict:
        nonlocal calls
        calls += 1
        return {"status": "completed", "changed": True}

    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: handle},
    )

    result = runtime.run_anytime(
        opportunity_provider=_opportunity_set,
        milestones_provider=lambda: {
            "B4_stock_boundary": True,
            "B5_configured_portfolio_acceptance": True,
        },
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
    )

    assert calls == 1
    assert result["execution_count"] == 1
    assert result["termination"] == "action_limit"
    assert result["semantics"]["B4_and_B5_do_not_stop_the_loop"] is True


def test_anytime_loop_treats_total_task_budget_as_normal_terminal(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path, max_total_tasks=1)
    kernel.reserve_task(
        task_id="prefill",
        kind="other",
        idempotency_key="prefill:reserve",
        input_revision=0,
    )
    kernel.settle_task(
        task_id="prefill",
        idempotency_key="prefill:settle",
        status="completed",
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": True,
            }
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=_opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=3,
    )

    assert result["termination"] == "budget_exhausted"
    assert result["termination_reasons"] == [
        "run_total_task_budget_exhausted"
    ]
    assert result["execution_count"] == 0
    assert result["unexecuted_actions"]["actions"][0]["reasons"] == [
        "run_total_task_budget_exhausted"
    ]
    assert result["semantics"]["global_budget_exhaustion_is_a_normal_terminal"]
    assert result["semantics"][
        "global_budget_terminal_is_persisted_before_return"
    ]
    assert result["kernel_stop_decision"]["decision"] == "budget_exhausted"
    assert kernel.state.status == "budget_exhausted"
    assert kernel.state.failure_reasons == ("run_total_task_budget_exhausted",)
    assert kernel.state.settled_task_count == 1
    assert kernel.state.in_flight_tasks == {}


def test_budget_terminal_blocks_post_loop_source_task_and_keeps_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(tmp_path, max_total_tasks=1)
    kernel.reserve_task(
        task_id="prefill",
        kind="other",
        idempotency_key="prefill:reserve",
        input_revision=0,
    )
    kernel.settle_task(
        task_id="prefill",
        idempotency_key="prefill:settle",
        status="completed",
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": True,
            }
        },
    )
    terminal = runtime.run_anytime(
        opportunity_provider=_opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
    )
    assert terminal["termination"] == "budget_exhausted"

    service = RetrosynthesisCampaignService(kernel)

    def fail_if_executed(_command: WorkerCommand):
        raise AssertionError("post-loop worker command must not execute")

    monkeypatch.setattr(service.workers, "execute", fail_if_executed)
    command = WorkerCommand(
        command_id="post-loop-source",
        run_id=kernel.spec.run_id,
        worker_type="discover_sources",
        input_revision=kernel.state.graph_revision,
        idempotency_key="post-loop-source",
        payload={"sources": []},
        budget=WorkerBudget(task_kind="evidence"),
    )
    blocked = service.execute_commands(
        (command,),
        idempotency_key="post-loop-source-batch",
        include_scheduled=False,
    )
    assert blocked["status"] == "budget_exhausted"
    assert blocked["executed_command_count"] == 0
    assert blocked["skipped_command_count"] == 1
    assert kernel.count_task_reservations(kind="evidence") == 0

    source_stage = discover_director_source_hints(
        service,
        (
            {
                "plan": {
                    "multi_step_skeletons": [
                        {
                            "skeleton_id": "skeleton:1",
                            "steps": [
                                {
                                    "step_id": "step:1",
                                    "product_smiles": kernel.spec.target_smiles,
                                    "source_hints": ["doi:10.1000/post-loop"],
                                }
                            ],
                        }
                    ]
                }
            },
        ),
    )
    assert source_stage["status"] == "budget_exhausted"
    assert source_stage["execution"]["executed_command_count"] == 0
    assert kernel.count_task_reservations(kind="evidence") == 0

    closeout = service.closeout(
        idempotency_key="post-loop-budget-closeout",
        budget_exhausted=True,
    )
    assert closeout["portfolio"]["closeout"]["decision"] == "budget_exhausted"
    assert kernel.state.status == "budget_exhausted"

    late_kernel = _kernel(tmp_path / "late", max_total_tasks=1)
    late_kernel.reserve_task(
        task_id="late-prefill",
        kind="other",
        idempotency_key="late-prefill:reserve",
        input_revision=0,
    )
    late_kernel.settle_task(
        task_id="late-prefill",
        idempotency_key="late-prefill:settle",
        status="completed",
    )
    late_service = RetrosynthesisCampaignService(late_kernel)
    monkeypatch.setattr(late_service.workers, "execute", fail_if_executed)
    late_blocked = late_service.execute_commands(
        (command,),
        idempotency_key="late-post-loop-source-batch",
        include_scheduled=False,
    )
    assert late_blocked["status"] == "budget_exhausted"
    assert late_blocked["executed_command_count"] == 0
    assert late_kernel.state.status == "budget_exhausted"
    assert late_kernel.state.failure_reasons == (
        "run_total_task_budget_exhausted",
    )
    assert late_kernel.count_task_reservations(kind="evidence") == 0


def test_budget_terminal_is_not_overwritten_by_unresolved_disposition(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path, max_total_tasks=1)
    kernel.reserve_task(
        task_id="prefill",
        kind="other",
        idempotency_key="prefill:reserve",
        input_revision=0,
    )
    kernel.settle_task(
        task_id="prefill",
        idempotency_key="prefill:settle",
        status="completed",
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": True,
            }
        },
    )
    terminal = runtime.run_anytime(
        opportunity_provider=_opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
    )
    assert terminal["termination"] == "budget_exhausted"

    transitioned = _transition_unresolved_if_active(
        kernel,
        idempotency_key="late-unresolved-disposition",
        reasons=("director_outcome_limit_exhausted",),
    )

    assert transitioned is False
    assert kernel.state.status == "budget_exhausted"
    assert kernel.state.failure_reasons == (
        "run_total_task_budget_exhausted",
    )

    late_kernel = _kernel(tmp_path / "pre-disposition", max_total_tasks=1)
    late_kernel.reserve_task(
        task_id="prefill",
        kind="other",
        idempotency_key="prefill:reserve",
        input_revision=0,
    )
    late_kernel.settle_task(
        task_id="prefill",
        idempotency_key="prefill:settle",
        status="completed",
    )
    late_service = RetrosynthesisCampaignService(late_kernel)
    assert late_service.terminalize_global_budget_if_reached(
        idempotency_key="pre-disposition-global-budget",
    )
    assert not _transition_unresolved_if_active(
        late_kernel,
        idempotency_key="late-unresolved-disposition",
        reasons=("director_outcome_limit_exhausted",),
    )
    assert late_kernel.state.status == "budget_exhausted"
    assert late_kernel.state.failure_reasons == (
        "run_total_task_budget_exhausted",
    )


def test_chemenzy_timeout_keeps_codex_peer_and_names_unstarted_validation(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    chemenzy_started = Event()
    codex_started = Event()

    def handle_chemenzy(_action) -> dict:
        chemenzy_started.set()
        assert codex_started.wait(timeout=2.0)
        kernel.reserve_task(
            task_id="chemenzy-child",
            kind="stock",
            idempotency_key="chemenzy-child:reserve",
            input_revision=0,
        )
        kernel.settle_task(
            task_id="chemenzy-child",
            idempotency_key="chemenzy-child:settle",
            status="completed",
        )
        return {
            "status": "timeout",
            "failure_reasons": ["chemenzy_provider_timeout"],
        }

    def handle_codex(_action) -> dict:
        codex_started.set()
        assert chemenzy_started.wait(timeout=2.0)
        kernel.reserve_task(
            task_id="codex-child",
            kind="evidence",
            idempotency_key="codex-child:reserve",
            input_revision=0,
        )
        kernel.settle_task(
            task_id="codex-child",
            idempotency_key="codex-child:settle",
            status="completed",
        )
        return {"status": "completed", "plan": {"route_families": ["route:1"]}}

    opportunities = compile_action_opportunities(
        {
            "content_sha256": "same-revision-start-cohort",
            "items": [
                {
                    "deficit_id": "deficit:target-native:start",
                    "kind": "expansion",
                    "object_id": "target:1",
                    "entity_ids": ["target:1"],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": False,
                    "model_allowed": False,
                    "priority": 900.0,
                    "reason": "target_requires_native_multi_step_search",
                    "score": {"expected_portfolio_gain": 1.0},
                    "metadata": {
                        "provider_preferences": ["chemenzy"],
                        "target_level_native_search": True,
                        "native_budget_reservation": "target_level",
                    },
                },
                {
                    "deficit_id": "deficit:architecture:start",
                    "kind": "architecture",
                    "object_id": "target:1",
                    "entity_ids": ["target:1"],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": False,
                    "model_allowed": True,
                    "priority": 880.0,
                    "reason": "target_requires_global_architecture",
                    "score": {"expected_portfolio_gain": 1.0},
                    "metadata": {"global_architecture": True},
                },
                {
                    "deficit_id": "deficit:validation:start",
                    "kind": "validation",
                    "object_id": "edge:1",
                    "entity_ids": ["edge:1"],
                    "route_family_ids": ["route:1"],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 700.0,
                    "reason": "candidate_requires_host_validation",
                    "score": {"evidence_gain": 0.8},
                    "metadata": {},
                },
            ],
        }
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.CHEMENZY_TARGET_EXPAND: handle_chemenzy,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE: handle_codex,
            CampaignActionKind.REACTION_VALIDATE: lambda _action: {
                "status": "completed"
            },
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {
            "native_search_target": True,
            "model": True,
            "validation": True,
        },
        max_actions=2,
        concurrent_start_kinds=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        ),
    )

    cohort = result["start_cohort"]
    executions = result["executions"]
    assert cohort["status"] == "completed"
    assert cohort["max_in_flight_action_count"] == 2
    assert [row["action"]["kind"] for row in executions] == [
        CampaignActionKind.CHEMENZY_TARGET_EXPAND.value,
        CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE.value,
    ]
    assert {row["action"]["input_revision"] for row in executions} == {0}
    assert [row["status"] for row in executions] == ["timeout", "completed"]
    assert executions[0]["outcome"]["failure_reasons"] == [
        "chemenzy_provider_timeout"
    ]
    assert executions[0]["outcome"]["failure_type"] == "timeout"
    assert executions[0]["outcome"]["resource_accounting"]["actual"][
        "task_counts"
    ] == {"other": 1, "stock": 1}
    assert executions[1]["outcome"]["resource_accounting"]["actual"][
        "task_counts"
    ] == {"evidence": 1, "other": 1}
    assert kernel.state.settled_task_count == 4
    assert kernel.state.in_flight_tasks == {}
    backlog = result["unexecuted_actions"]
    assert backlog["action_count"] == 1
    assert backlog["actions"][0]["kind"] == CampaignActionKind.REACTION_VALIDATE.value
    assert backlog["actions"][0]["reasons"] == ["action_limit_reached"]

    replay = runtime.execute_concurrent_cohort(
        opportunities,
        action_kinds=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        ),
        milestones={},
        resource_availability={
            "native_search_target": True,
            "model": True,
            "validation": True,
        },
    )

    assert replay["cohort_id"] == cohort["cohort_id"]
    assert replay["action_execution_ids"] == cohort["action_execution_ids"]
    assert replay["max_in_flight_action_count"] == 0
    assert all(row["cache_hit"] is True for row in replay["executions"])
    assert kernel.state.settled_task_count == 4


def test_program_validation_and_feedback_use_independent_action_accounting(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunities = compile_action_opportunities(
        {
            "content_sha256": "program-validation-actions",
            "items": [
                {
                    "deficit_id": "signal:program-validation:1",
                    "kind": "program_validation",
                    "object_id": "experimental-work:1",
                    "entity_ids": ["program:1"],
                    "route_family_ids": ["route:1"],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 600.0,
                    "reason": "program_candidate_requires_specialized_validation",
                    "score": {"evidence_gain": 0.5},
                    "metadata": {"program_validation": True},
                },
                {
                    "deficit_id": "signal:experiment-feedback:1",
                    "kind": "experiment_feedback",
                    "object_id": "validation:1",
                    "entity_ids": ["program:1"],
                    "route_family_ids": ["route:1"],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 590.0,
                    "reason": "external_program_validation_feedback_available",
                    "score": {"evidence_gain": 0.8},
                    "metadata": {"experiment_feedback": True},
                },
            ],
        }
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.PROGRAM_VALIDATE: lambda _action: {
                "status": "awaiting_external_result",
                "changed": False,
                "semantics": {"grants_no_validation_claim": True},
            },
            CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST: lambda _action: {
                "status": "completed",
                "changed": False,
                "semantics": {"canonical_graph_not_mutated": True},
            },
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {
            "program": True,
            "experiment": True,
        },
        max_actions=2,
        max_consecutive_no_gain=3,
    )

    assert [row["action"]["kind"] for row in result["executions"]] == [
        CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST.value,
        CampaignActionKind.PROGRAM_VALIDATE.value,
    ]
    assert [row["status"] for row in result["executions"]] == [
        "completed",
        "awaiting_external_result",
    ]
    assert [
        kernel.task_lifecycle(row["action"]["task_id"])["reservation"][
            "payload"
        ]["kind"]
        for row in result["executions"]
    ] == ["experiment", "program"]
    assert [
        kernel.task_lifecycle(row["action"]["task_id"])["reservation"][
            "payload"
        ]["metadata"]["delegated_resource_class"]
        for row in result["executions"]
    ] == ["experiment", "program"]
    assert kernel.state.model_totals["model_invocations"] == 0
    assert kernel.state.attempt_count == 0
    assert kernel.state.in_flight_tasks == {}


def test_runtime_converts_handler_exception_to_replayable_failure(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)

    def fail(_action) -> dict:
        raise RuntimeError("handler exploded")

    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: fail},
    )
    decision = _decision()
    action = bind_scheduled_action(decision, input_revision=kernel.state.graph_revision)

    first = runtime.execute(action, decision=decision)
    replay = runtime.execute(action, decision=decision)

    assert first["status"] == "failed"
    assert first["outcome"]["failure_reasons"] == [
        "campaign_action_handler_error:RuntimeError:handler exploded"
    ]
    assert replay["cache_hit"] is True
    assert replay["status"] == "failed"


def test_runtime_reads_strictly_bound_legacy_action_receipt(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: lambda _action: {"status": "completed"}},
    )
    decision = _decision()
    action = bind_scheduled_action(decision, input_revision=0)
    current = runtime.execute(action, decision=decision)
    legacy_action_sha256 = legacy_campaign_action_sha256(action)
    legacy_outcome = dict(current["outcome"])
    legacy_outcome["schema_version"] = "campaign_action_outcome.v1"
    legacy_outcome["action_sha256"] = legacy_action_sha256
    legacy_outcome.pop("resource_accounting", None)
    legacy_outcome["content_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in legacy_outcome.items()
                if key != "content_sha256"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    legacy_ref = kernel.artifacts.put_json(
        legacy_outcome,
        logical_name="legacy-campaign-action.json",
        producer="test",
    )
    kernel.artifacts.write_pointer(
        runtime._pointer_name(action),
        legacy_ref,
        metadata={
            "action_execution_id": action.execution_id,
            "action_sha256": legacy_action_sha256,
            "input_revision": action.input_revision,
            "output_revision": kernel.state.graph_revision,
        },
    )

    replay = runtime.execute(action, decision=decision)

    assert replay["cache_hit"] is True
    assert replay["outcome"]["action_sha256"] == legacy_action_sha256
    assert "resource_accounting" not in replay["outcome"]


def test_runtime_resumes_strictly_bound_legacy_in_flight_action(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    decision = _decision()
    action = bind_scheduled_action(decision, input_revision=0)
    legacy_sha256 = legacy_campaign_action_sha256(action)
    kernel.reserve_task(
        task_id=action.task_id,
        kind="other",
        idempotency_key=f"{action.idempotency_key}:reserve",
        input_revision=0,
        metadata={
            "campaign_action_id": action.action_id,
            "campaign_action_execution_id": action.execution_id,
            "campaign_action_sha256": legacy_sha256,
            "delegated_resource_class": action.resource_class,
            "producer": action.producer,
        },
    )
    calls = 0

    def handle(_action) -> dict:
        nonlocal calls
        calls += 1
        return {"status": "completed"}

    reopened_kernel = _kernel(tmp_path)
    runtime = CampaignActionRuntime(
        reopened_kernel,
        {CampaignActionKind.MATERIALIZE: handle},
    )

    result = runtime.execute(action, decision=decision)

    assert result["status"] == "completed"
    assert calls == 1
    assert reopened_kernel.task_lifecycle(action.task_id)["status"] == "settled"
    assert result["outcome"]["resource_accounting"]["actual"][
        "settled_task_count"
    ] == 1


def test_runtime_fails_closed_on_outcome_digest_tamper(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": False,
            }
        },
    )
    decision = _decision()
    action = bind_scheduled_action(decision, input_revision=kernel.state.graph_revision)
    first = runtime.execute(action, decision=decision)
    tampered = dict(first["outcome"])
    tampered["handler_result"] = {"status": "completed", "changed": True}
    tampered_ref = kernel.artifacts.put_json(
        tampered,
        logical_name="tampered-campaign-action.json",
        producer="test",
    )
    action_row = action.to_dict()
    kernel.artifacts.write_pointer(
        runtime._pointer_name(action),
        tampered_ref,
        metadata={
            "action_execution_id": action.execution_id,
            "action_sha256": action_row["content_sha256"],
            "input_revision": action.input_revision,
            "output_revision": kernel.state.graph_revision,
        },
    )

    with pytest.raises(
        CampaignActionRuntimeError,
        match="campaign_action_outcome_digest_invalid",
    ):
        runtime.execute(action, decision=decision)
