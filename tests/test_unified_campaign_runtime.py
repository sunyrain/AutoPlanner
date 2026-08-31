from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Barrier, Event
from time import sleep

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
    CampaignActionDeferredHandler,
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


def _no_gain_opportunity_set(*, count: int = 3) -> dict:
    return compile_action_opportunities(
        {
            "content_sha256": f"{count}-no-gain-actions",
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
                for index in range(count)
            ],
        }
    )


def _concurrent_opportunity_set() -> dict:
    shared = {
        "dependency_ids": [],
        "deterministic": False,
        "model_allowed": True,
        "priority": 700.0,
        "score": {
            "expected_portfolio_gain": 0.7,
            "evidence_gain": 0.7,
        },
    }
    return compile_action_opportunities(
        {
            "content_sha256": "bounded-concurrent-actions",
            "items": [
                {
                    **shared,
                    "deficit_id": "deficit:expansion:concurrent",
                    "kind": "expansion",
                    "object_id": "mol:frontier",
                    "entity_ids": ["mol:frontier"],
                    "route_family_ids": ["route:search"],
                    "reason": "frontier_requires_local_generation",
                    "metadata": {
                        "frontier_smiles": "CCO",
                        "provider_preferences": ["chemenzy"],
                    },
                },
                {
                    **shared,
                    "deficit_id": "deficit:replan:concurrent",
                    "kind": "replan",
                    "object_id": "target:1",
                    "entity_ids": ["target:1"],
                    "route_family_ids": [],
                    "reason": "material_event_requires_global_replan",
                    "metadata": {"global_replan": True},
                },
                {
                    **shared,
                    "deficit_id": "deficit:evidence:concurrent",
                    "kind": "evidence",
                    "object_id": "edge:evidence",
                    "entity_ids": ["edge:evidence"],
                    "route_family_ids": ["route:proof"],
                    "reason": "edge_requires_exact_source_acquisition",
                    "metadata": {},
                },
                {
                    **shared,
                    "deficit_id": "deficit:validation:concurrent",
                    "kind": "validation",
                    "object_id": "edge:validation",
                    "entity_ids": ["edge:validation"],
                    "route_family_ids": ["route:proof"],
                    "reason": "materialized_edge_requires_reaction_validation",
                    "metadata": {},
                },
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


def test_action_cache_reuses_same_execution_across_scheduler_diagnostic_drift(
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
    first_decision = _decision()
    first_action = bind_scheduled_action(first_decision, input_revision=0)
    runtime.execute(first_action, decision=first_decision)

    rebound_decision = {
        **first_decision,
        "scheduler_policy": "rebound-diagnostic-policy",
        "round_robin_cursor": 17,
    }
    rebound_action = bind_scheduled_action(rebound_decision, input_revision=0)

    assert rebound_action.execution_id == first_action.execution_id
    assert (
        rebound_action.to_dict()["content_sha256"]
        != first_action.to_dict()["content_sha256"]
    )
    replay = runtime.execute(rebound_action, decision=rebound_decision)

    assert replay["cache_hit"] is True
    assert calls == 1


def test_native_handler_checkpoint_resumes_without_second_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(tmp_path)
    decision = schedule_next_action(
        _concurrent_opportunity_set(),
        milestones={},
        resource_availability={"native_search_frontier": True},
        available_action_kinds=(
            CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND.value,
        ),
    )
    action = bind_scheduled_action(decision, input_revision=0)
    calls = 0

    def handle(_action) -> dict:
        nonlocal calls
        calls += 1
        return {
            "status": "completed",
            "frontier_smiles": ["CCO"],
            "proposal_count": 1,
            "provider_invocation_count": 1,
            "provider_result_replay_count": 1,
        }

    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND: handle},
    )
    original_finalize = runtime._finalize_reserved

    def interrupt_after_checkpoint(*_args, **_kwargs):
        raise RuntimeError("simulated_host_interrupt_after_provider_return")

    monkeypatch.setattr(runtime, "_finalize_reserved", interrupt_after_checkpoint)
    with pytest.raises(
        RuntimeError,
        match="simulated_host_interrupt_after_provider_return",
    ):
        runtime.execute(action, decision=decision)

    lifecycle = kernel.task_lifecycle(action.task_id)
    assert lifecycle["status"] == "in_flight"
    assert lifecycle["checkpoints"][-1]["payload"]["checkpoint_kind"] == (
        "campaign_action_handler_result"
    )
    monkeypatch.setattr(runtime, "_finalize_reserved", original_finalize)

    resumed = CampaignActionRuntime(
        _kernel(tmp_path),
        {CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND: handle},
    ).execute(action, decision=decision)

    assert resumed["status"] == "completed"
    assert resumed["handler_checkpoint_replayed"] is True
    assert resumed["outcome"]["handler_result"]["frontier_smiles"] == [
        "CCO"
    ]
    assert resumed["outcome"]["handler_result"][
        "provider_result_replay_count"
    ] == 1
    assert calls == 1


def test_action_history_exposes_guided_frontier_from_durable_outcome(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    decision = schedule_next_action(
        _concurrent_opportunity_set(),
        milestones={},
        resource_availability={"native_search_frontier": True},
        available_action_kinds=(
            CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND.value,
        ),
    )
    action = bind_scheduled_action(decision, input_revision=0)
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND: lambda _action: {
                "status": "unresolved",
                "frontier_smiles": ["CCO"],
                "proposal_count": 0,
                "provider_invocation_count": 1,
                "provider_result_replay_count": 1,
            }
        },
    )

    runtime.execute(action, decision=decision)
    history = CampaignActionRuntime(
        _kernel(tmp_path),
        runtime.handlers,
    ).action_execution_history()

    assert history[-1]["settled"] is True
    assert history[-1]["handler_result"]["frontier_smiles"] == ["CCO"]
    assert history[-1]["handler_result"]["provider_invocation_count"] == 1
    assert history[-1]["handler_result"]["provider_result_replay_count"] == 1


def test_action_class_service_history_replays_across_runtime_reopen(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": False,
            },
            CampaignActionKind.REACTION_VALIDATE: lambda _action: {
                "status": "completed",
                "changed": False,
            },
        },
    )
    first = runtime.schedule_and_execute(
        _opportunity_set(kind="materialization"),
        milestones={},
        resource_availability={"deterministic": True},
    )
    first_lifecycle = kernel.task_lifecycle(first["action"]["task_id"])
    first_metadata = first_lifecycle["reservation"]["payload"]["metadata"]

    assert runtime.action_service_history() == ("host_materialize",)
    assert first_metadata["campaign_action_class"] == "deterministic_closure"
    assert first_metadata["action_class_service_ordinal"] == 1
    assert first_metadata["action_class_service_sha256"] == first["decision"][
        "action_class_service"
    ]["content_sha256"]

    expected_next = schedule_next_action(
        _opportunity_set(kind="validation"),
        prior_action_kinds=runtime.action_service_history(),
    )
    reopened_kernel = _kernel(tmp_path)
    reopened = CampaignActionRuntime(
        reopened_kernel,
        runtime.handlers,
    )
    replayed_next = schedule_next_action(
        _opportunity_set(kind="validation"),
        prior_action_kinds=reopened.action_service_history(),
    )

    assert reopened.action_service_history() == runtime.action_service_history()
    assert replayed_next["selected_action_id"] == expected_next["selected_action_id"]
    assert replayed_next["action_class_service"] == expected_next[
        "action_class_service"
    ]

    second = reopened.schedule_and_execute(
        _opportunity_set(kind="validation"),
        milestones={},
        resource_availability={"validation": True},
    )
    assert second["action"]["kind"] == "reaction_validate"
    assert reopened.action_service_history() == (
        "host_materialize",
        "reaction_validate",
    )
    assert len(reopened_kernel.task_reservation_history()) == 2


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
    opportunity_set = _no_gain_opportunity_set()
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


def test_failed_or_rejected_action_is_not_recorded_as_no_gain(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunity_set = _no_gain_opportunity_set(count=1)
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "rejected",
                "reasons": ["contract_blocked"],
                "changed": False,
            }
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=1,
    )

    assert result["consecutive_no_gain"] == 0
    assert result["convergence_ledger"]["no_gain_bindings"] == []


def test_provider_runtime_pause_preserves_action_for_exact_resume(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunities = _opportunity_set()
    calls: list[str] = []

    def handle(_action) -> dict:
        calls.append(_action.execution_id)
        if len(calls) == 1:
            return {
                "status": "runtime_unavailable",
                "runtime_unavailable": True,
                "runtime_pause": True,
                "reason": "provider_auth_unavailable",
                "changed": False,
                "model_invocations": 0,
            }
        return {"status": "completed", "changed": False}

    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: handle},
    )
    first = runtime.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=3,
        max_consecutive_no_gain=3,
    )

    first_action = first["executions"][0]["action"]
    assert first["termination"] == "runtime_unavailable"
    assert first["termination_reasons"] == ["provider_auth_unavailable"]
    assert first["consecutive_no_gain"] == 0
    assert first["convergence_ledger"][
        "attempted_action_ids_at_current_revision"
    ] == []
    assert kernel.state.status == "paused"
    assert kernel.task_lifecycle(first_action["task_id"])["status"] == "in_flight"

    kernel.resume(idempotency_key="test:provider-recovered")
    second = runtime.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=3,
    )

    second_action = second["executions"][0]["action"]
    assert calls == [first_action["execution_id"], first_action["execution_id"]]
    assert second_action["execution_id"] == first_action["execution_id"]
    assert kernel.task_lifecycle(first_action["task_id"])["status"] == "settled"
    assert second["convergence_ledger"]["settled_execution_count"] == 1


def test_retryable_prerequisite_is_not_recorded_as_no_gain(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunity_set = _no_gain_opportunity_set(count=1)
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": False,
                "retryable_after_graph_revision": True,
            }
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=1,
    )

    assert result["consecutive_no_gain"] == 0
    assert result["convergence_ledger"]["failed_or_rejected_count"] == 1
    assert result["convergence_ledger"]["no_gain_bindings"] == []


def test_anytime_slice_uses_one_scheduler_with_bounded_action_kinds(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunity_set = compile_action_opportunities(
        {
            "content_sha256": "bounded-anytime-action-family",
            "items": [
                {
                    "deficit_id": "deficit:materialization:1",
                    "kind": "materialization",
                    "object_id": "hypothesis:1",
                    "entity_ids": ["hypothesis:1"],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 900.0,
                    "reason": "materialization_outside_recovery_slice",
                    "score": {},
                },
                {
                    "deficit_id": "deficit:validation:1",
                    "kind": "validation",
                    "object_id": "edge:1",
                    "entity_ids": ["edge:1"],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 100.0,
                    "reason": "validation_inside_recovery_slice",
                    "score": {},
                },
            ],
        }
    )
    executed: list[str] = []
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda action: executed.append(
                action.kind.value
            )
            or {"status": "completed", "changed": False},
            CampaignActionKind.REACTION_VALIDATE: lambda action: executed.append(
                action.kind.value
            )
            or {"status": "completed", "changed": False},
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=2,
        available_action_kinds=(CampaignActionKind.REACTION_VALIDATE,),
    )

    assert result["execution_count"] == 1
    assert executed == [CampaignActionKind.REACTION_VALIDATE.value]
    assert result["executions"][0]["action"]["kind"] == (
        CampaignActionKind.REACTION_VALIDATE.value
    )


def test_canonical_rejection_report_is_not_persisted_as_no_gain(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunity_set = _no_gain_opportunity_set(count=1)
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": False,
                "rejected": [
                    {
                        "kind": "reaction_edge",
                        "reasons": ["ancestor_or_target_cycle"],
                    }
                ],
            }
        },
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunity_set,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=1,
    )

    assert result["execution_count"] == 1
    assert result["consecutive_no_gain"] == 0
    assert result["convergence_ledger"]["failed_or_rejected_count"] == 1
    assert result["convergence_ledger"]["no_gain_bindings"] == []
    reasons = result["unexecuted_actions"]["actions"]
    assert reasons == []


def test_anytime_convergence_replays_across_slices_and_runtime_reopen(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    calls: list[str] = []

    def handle(action) -> dict:
        calls.append(action.action_id)
        return {"status": "completed", "changed": False}

    opportunities = _no_gain_opportunity_set()
    runtime = CampaignActionRuntime(
        kernel,
        {CampaignActionKind.MATERIALIZE: handle},
    )
    first = runtime.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=2,
    )

    assert first["termination"] == "action_limit"
    assert first["consecutive_no_gain"] == 1
    assert first["convergence_ledger"]["settled_execution_count"] == 1
    first_action_id = first["executions"][0]["action"]["action_id"]

    reopened_kernel = _kernel(tmp_path)
    reopened = CampaignActionRuntime(
        reopened_kernel,
        {CampaignActionKind.MATERIALIZE: handle},
    )
    second = reopened.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=2,
        max_consecutive_no_gain=2,
    )

    assert second["termination"] == "converged_low_marginal_gain"
    assert second["execution_count"] == 1
    assert second["executions"][0]["cache_hit"] is False
    assert second["executions"][0]["action"]["action_id"] != first_action_id
    assert second["consecutive_no_gain"] == 2
    assert second["semantics"]["convergence_resumed_from_history"] is True
    assert reopened.action_convergence_ledger() == second["convergence_ledger"]

    final_reopen = CampaignActionRuntime(
        _kernel(tmp_path),
        {CampaignActionKind.MATERIALIZE: handle},
    )
    third = final_reopen.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=2,
        max_consecutive_no_gain=2,
    )

    assert third["termination"] == "converged_low_marginal_gain"
    assert third["execution_count"] == 0
    assert third["consecutive_no_gain"] == 2
    assert len(calls) == 2
    assert len(final_reopen.action_execution_history()) == 2


def test_external_graph_progress_breaks_resumed_no_gain_streak(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunities = _no_gain_opportunity_set()
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.MATERIALIZE: lambda _action: {
                "status": "completed",
                "changed": False,
            }
        },
    )
    first = runtime.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=2,
    )
    assert first["consecutive_no_gain"] == 1
    kernel.publish_graph_revision(
        1,
        graph_sha256="external-graph-progress",
        evidence_revision=0,
        idempotency_key="external-graph-progress",
    )

    reopened = CampaignActionRuntime(
        _kernel(tmp_path),
        runtime.handlers,
    )
    reset = reopened.action_convergence_ledger()

    assert reset["consecutive_no_gain"] == 0
    assert reset["revision_advanced_outside_history"] is True
    resumed = reopened.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=1,
        max_consecutive_no_gain=2,
    )
    assert resumed["termination"] == "action_limit"
    assert resumed["execution_count"] == 1
    assert resumed["consecutive_no_gain"] == 1
    assert resumed["convergence_ledger"]["revision_discontinuity_count"] == 1


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


def test_short_tail_closure_and_later_leaf_stay_in_one_anytime_loop(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    state = {"phase": "first_tail", "stock_closed": False}

    def opportunities() -> dict:
        phase = state["phase"]
        common = {
            "object_id": "target:1",
            "route_family_ids": ["route:1"],
            "dependency_ids": [],
            "score": {"expected_portfolio_gain": 1.0},
        }
        if phase in {"first_tail", "second_tail"}:
            suffix = "1" if phase == "first_tail" else "2"
            item = {
                **common,
                "deficit_id": f"deficit:tail:{suffix}",
                "kind": "expansion",
                "entity_ids": [f"mol:leaf:{suffix}"],
                "deterministic": False,
                "model_allowed": False,
                "reason": "stock_rejected_target_leaf_requires_short_tail",
                "metadata": {
                    "provider_preferences": ["native_short_tail"],
                    "frontier_smiles": "CCO" if suffix == "1" else "CCN",
                    "paper_short_tail_eligible": True,
                },
            }
        elif phase == "materialize":
            item = {
                **common,
                "deficit_id": "deficit:materialize:1",
                "kind": "materialization",
                "entity_ids": ["hypothesis:1"],
                "deterministic": True,
                "model_allowed": False,
                "reason": "accepted_hypothesis_requires_materialization",
            }
        elif phase == "stock":
            item = {
                **common,
                "deficit_id": "deficit:stock:1",
                "kind": "stock",
                "entity_ids": ["mol:leaf:closed"],
                "deterministic": True,
                "model_allowed": False,
                "reason": "selected_leaf_requires_trusted_stock_audit",
            }
        else:
            return compile_action_opportunities(
                {"content_sha256": "single-anytime-done", "items": []}
            )
        return compile_action_opportunities(
            {"content_sha256": f"single-anytime-{phase}", "items": [item]}
        )

    def handle_short_tail(_action) -> dict:
        state["phase"] = (
            "materialize"
            if state["phase"] == "first_tail"
            else "done"
        )
        return {"status": "completed", "changed": True, "proposal_count": 1}

    def handle_materialize(_action) -> dict:
        state["phase"] = "stock"
        return {"status": "completed", "changed": True}

    def handle_stock(_action) -> dict:
        state["stock_closed"] = True
        state["phase"] = "second_tail"
        return {"status": "completed", "changed": True}

    result = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND: handle_short_tail,
            CampaignActionKind.MATERIALIZE: handle_materialize,
            CampaignActionKind.STOCK_AUDIT: handle_stock,
        },
    ).run_anytime(
        opportunity_provider=opportunities,
        milestones_provider=lambda: {
            "B4_stock_boundary": state["stock_closed"]
        },
        resource_availability_provider=lambda: {
            "native_search_frontier": True,
            "deterministic": True,
            "stock": True,
        },
        max_actions=4,
        max_consecutive_no_gain=5,
    )

    assert [row["action"]["kind"] for row in result["executions"]] == [
        CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND.value,
        CampaignActionKind.MATERIALIZE.value,
        CampaignActionKind.STOCK_AUDIT.value,
        CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND.value,
    ]
    assert result["termination"] == "action_limit"
    assert result["semantics"]["single_scheduler_loop"] is True
    assert result["semantics"]["B4_and_B5_do_not_stop_the_loop"] is True


def test_explicit_stock_delivery_boundary_stops_after_b4_snapshot(
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
            "B4_stock_boundary": calls >= 1,
        },
        resource_availability_provider=lambda: {"deterministic": True},
        max_actions=4,
        stop_milestone="B4_stock_boundary",
    )

    assert calls == 1
    assert result["execution_count"] == 1
    assert result["termination"] == "milestone_reached"
    assert result["termination_reasons"] == [
        "delivery_milestone_reached:B4_stock_boundary"
    ]
    assert result["semantics"]["B4_and_B5_do_not_stop_the_loop"] is False
    assert result["semantics"]["configured_stop_milestone"] == (
        "B4_stock_boundary"
    )


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
    assert runtime.action_service_history() == ()


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


def test_anytime_loop_runs_bounded_cross_class_cohorts_and_replays(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunities = _concurrent_opportunity_set()
    rendezvous = Barrier(4)
    started: list[str] = []
    observed: list[tuple[int, str]] = []

    def handle(action) -> dict:
        started.append(action.kind.value)
        rendezvous.wait(timeout=2.0)
        return {"status": "completed", "changed": False}

    concurrent_kinds = (
        CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND,
        CampaignActionKind.CODEX_REPLAN,
        CampaignActionKind.ACQUIRE_EVIDENCE,
        CampaignActionKind.REACTION_VALIDATE,
    )
    runtime = CampaignActionRuntime(
        kernel,
        {kind: handle for kind in concurrent_kinds},
    )

    result = runtime.run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {
            "native_search_frontier": True,
            "model": True,
            "evidence": True,
            "validation": True,
        },
        max_actions=4,
        max_consecutive_no_gain=5,
        concurrent_action_kinds=concurrent_kinds,
        max_concurrent_actions=99,
        on_execution=lambda index, execution: observed.append(
            (index, str(dict(execution.get("action") or {}).get("kind") or ""))
        ),
    )

    assert set(started) == {kind.value for kind in concurrent_kinds}
    assert [row["action"]["kind"] for row in result["executions"]] == [
        kind.value for kind in concurrent_kinds
    ]
    assert observed == [
        (index, kind.value)
        for index, kind in enumerate(concurrent_kinds, start=1)
    ]
    assert result["execution_count"] == 4
    assert result["concurrent_worker_limit"] == 4
    assert len(result["concurrent_cohorts"]) == 1
    cohort = result["concurrent_cohorts"][0]
    assert cohort["worker_limit"] == 4
    assert cohort["max_in_flight_action_count"] == 4
    assert cohort["semantics"][
        "worker_pool_is_runtime_owned_and_hard_bounded"
    ] is True
    assert result["semantics"][
        "all_concurrency_is_owned_by_this_action_loop"
    ] is True
    assert kernel.state.in_flight_tasks == {}

    replay = runtime.execute_concurrent_cohort(
        opportunities,
        action_kinds=concurrent_kinds,
        milestones={},
        resource_availability={
            "native_search_frontier": True,
            "model": True,
            "evidence": True,
            "validation": True,
        },
        max_actions=4,
    )

    assert replay["cohort_id"] == cohort["cohort_id"]
    assert replay["max_in_flight_action_count"] == 0
    assert all(row["cache_hit"] is True for row in replay["executions"])


def test_concurrent_cohort_excludes_same_resource_and_fits_wrapper_budget(
    tmp_path: Path,
) -> None:
    opportunities = _concurrent_opportunity_set()
    acquire = next(
        row
        for row in opportunities["actions"]
        if row["kind"] == CampaignActionKind.ACQUIRE_EVIDENCE.value
    )
    validation = next(
        row
        for row in opportunities["actions"]
        if row["kind"] == CampaignActionKind.REACTION_VALIDATE.value
    )
    bind = {
        **acquire,
        "action_id": "action:bind_exact_evidence:resource-collision",
        "kind": CampaignActionKind.BIND_EVIDENCE.value,
        "deficit_id": "deficit:evidence:binding",
        "content_sha256": "bind-resource-collision",
    }
    proof_opportunities = {
        "content_sha256": "proof-resource-collision",
        "actions": [acquire, bind, validation],
    }
    rendezvous = Barrier(2)
    calls: list[str] = []

    def handle(action) -> dict:
        calls.append(action.kind.value)
        rendezvous.wait(timeout=2.0)
        return {"status": "completed", "changed": False}

    handlers = {
        CampaignActionKind.ACQUIRE_EVIDENCE: handle,
        CampaignActionKind.BIND_EVIDENCE: handle,
        CampaignActionKind.REACTION_VALIDATE: handle,
    }
    runtime = CampaignActionRuntime(_kernel(tmp_path / "resource"), handlers)
    cohort = runtime.execute_concurrent_cohort(
        proof_opportunities,
        action_kinds=tuple(handlers),
        milestones={},
        resource_availability={"evidence": True, "validation": True},
        max_actions=4,
    )

    assert cohort["status"] == "completed"
    assert cohort["resource_collision_kinds"] == [
        CampaignActionKind.BIND_EVIDENCE.value
    ]
    assert set(calls) == {
        CampaignActionKind.ACQUIRE_EVIDENCE.value,
        CampaignActionKind.REACTION_VALIDATE.value,
    }
    assert cohort["max_in_flight_action_count"] == 2

    limited_kernel = _kernel(tmp_path / "limited", max_total_tasks=1)
    limited = CampaignActionRuntime(
        limited_kernel,
        {
            CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND: handle,
            CampaignActionKind.CODEX_REPLAN: handle,
            CampaignActionKind.ACQUIRE_EVIDENCE: handle,
            CampaignActionKind.REACTION_VALIDATE: handle,
        },
    ).execute_concurrent_cohort(
        opportunities,
        action_kinds=(
            CampaignActionKind.NATIVE_SHORT_TAIL_EXPAND,
            CampaignActionKind.CODEX_REPLAN,
            CampaignActionKind.ACQUIRE_EVIDENCE,
            CampaignActionKind.REACTION_VALIDATE,
        ),
        milestones={},
        resource_availability={
            "native_search_frontier": True,
            "model": True,
            "evidence": True,
            "validation": True,
        },
        max_actions=4,
    )

    assert limited["status"] == "not_launched"
    assert len(limited["selected_action_ids"]) == 1
    assert len(limited["omitted_for_wrapper_budget"]) == 3
    assert limited_kernel.state.in_flight_tasks == {}


def test_deferred_cohort_commits_in_action_order_after_reversed_prepare_completion(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    opportunities = _concurrent_opportunity_set()
    validation_prepared = Event()
    commit_order: list[str] = []

    def prepare_evidence(_action) -> dict:
        assert validation_prepared.wait(timeout=2.0)
        return {"prepared": "evidence"}

    def prepare_validation(_action) -> dict:
        validation_prepared.set()
        return {"prepared": "validation"}

    def commit(action, prepared) -> dict:
        commit_order.append(action.kind.value)
        return {
            "status": "completed",
            "prepared": str(prepared.get("prepared") or ""),
        }

    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.ACQUIRE_EVIDENCE: CampaignActionDeferredHandler(
                prepare=prepare_evidence,
                commit=commit,
            ),
            CampaignActionKind.REACTION_VALIDATE: CampaignActionDeferredHandler(
                prepare=prepare_validation,
                commit=commit,
            ),
        },
    )

    cohort = runtime.execute_concurrent_cohort(
        opportunities,
        action_kinds=(
            CampaignActionKind.ACQUIRE_EVIDENCE,
            CampaignActionKind.REACTION_VALIDATE,
        ),
        milestones={},
        resource_availability={"evidence": True, "validation": True},
        max_actions=2,
    )

    assert [row["status"] for row in cohort["executions"]] == [
        "completed",
        "completed",
    ]
    assert commit_order == [
        CampaignActionKind.ACQUIRE_EVIDENCE.value,
        CampaignActionKind.REACTION_VALIDATE.value,
    ]
    assert cohort["semantics"]["deferred_commits_follow_stable_action_order"] is True
    assert kernel.state.in_flight_tasks == {}


def test_deferred_prepare_failure_does_not_cancel_peer_and_commit_failure_replays(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path / "prepare")
    opportunities = _concurrent_opportunity_set()
    rendezvous = Barrier(2)
    validation_commits = 0

    def fail_prepare(_action) -> dict:
        rendezvous.wait(timeout=2.0)
        raise RuntimeError("prepare boom")

    def prepare_validation(_action) -> dict:
        rendezvous.wait(timeout=2.0)
        return {"prepared": True}

    def commit_validation(_action, _prepared) -> dict:
        nonlocal validation_commits
        validation_commits += 1
        return {"status": "completed"}

    cohort = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.ACQUIRE_EVIDENCE: CampaignActionDeferredHandler(
                prepare=fail_prepare,
                commit=lambda _action, _prepared: {"status": "completed"},
            ),
            CampaignActionKind.REACTION_VALIDATE: CampaignActionDeferredHandler(
                prepare=prepare_validation,
                commit=commit_validation,
            ),
        },
    ).execute_concurrent_cohort(
        opportunities,
        action_kinds=(
            CampaignActionKind.ACQUIRE_EVIDENCE,
            CampaignActionKind.REACTION_VALIDATE,
        ),
        milestones={},
        resource_availability={"evidence": True, "validation": True},
        max_actions=2,
    )

    assert [row["status"] for row in cohort["executions"]] == [
        "failed",
        "completed",
    ]
    assert cohort["executions"][0]["outcome"]["failure_reasons"] == [
        "campaign_action_prepare_error:RuntimeError:prepare boom"
    ]
    assert validation_commits == 1
    assert kernel.state.in_flight_tasks == {}

    replay_kernel = _kernel(tmp_path / "commit")
    decision = _decision(kind="validation")
    action = bind_scheduled_action(
        decision,
        input_revision=replay_kernel.state.graph_revision,
    )
    calls = {"prepare": 0, "commit": 0}

    def prepare_once(_action) -> dict:
        calls["prepare"] += 1
        return {"prepared": True}

    def fail_commit(_action, _prepared) -> dict:
        calls["commit"] += 1
        raise RuntimeError("commit boom")

    replay_runtime = CampaignActionRuntime(
        replay_kernel,
        {
            CampaignActionKind.REACTION_VALIDATE: CampaignActionDeferredHandler(
                prepare=prepare_once,
                commit=fail_commit,
            )
        },
    )
    first = replay_runtime.execute(action, decision=decision)
    replay = replay_runtime.execute(action, decision=decision)

    assert first["status"] == "failed"
    assert first["outcome"]["failure_reasons"] == [
        "campaign_action_commit_error:RuntimeError:commit boom"
    ]
    assert replay["cache_hit"] is True
    assert replay["outcome"] == first["outcome"]
    assert calls == {"prepare": 1, "commit": 1}


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


@pytest.mark.parametrize(
    ("codex_status", "failure_reason"),
    [
        ("failed", "codex_provider_failed"),
        ("timeout", "codex_provider_timeout"),
        ("failed", "codex_contract_rejected"),
    ],
)
def test_codex_failure_timeout_or_contract_rejection_keeps_chemenzy_peer(
    tmp_path: Path,
    codex_status: str,
    failure_reason: str,
) -> None:
    kernel = _kernel(tmp_path / failure_reason)
    rendezvous = Barrier(2)

    def handle_chemenzy(_action) -> dict:
        rendezvous.wait(timeout=2.0)
        return {
            "status": "completed",
            "routes": [{"route_trace_id": "route:chem-enzy:1"}],
        }

    def handle_codex(_action) -> dict:
        rendezvous.wait(timeout=2.0)
        return {
            "status": codex_status,
            "failure_reasons": [failure_reason],
        }

    opportunities = compile_action_opportunities(
        {
            "content_sha256": f"codex-peer-isolation:{failure_reason}",
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
            ],
        }
    )
    result = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.CHEMENZY_TARGET_EXPAND: handle_chemenzy,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE: handle_codex,
        },
    ).run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {
            "native_search_target": True,
            "model": True,
        },
        max_actions=2,
        concurrent_start_kinds=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        ),
    )

    assert result["start_cohort"]["max_in_flight_action_count"] == 2
    executions = result["executions"]
    assert [row["action"]["kind"] for row in executions] == [
        CampaignActionKind.CHEMENZY_TARGET_EXPAND.value,
        CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE.value,
    ]
    assert [row["status"] for row in executions] == ["completed", codex_status]
    assert executions[0]["outcome"]["handler_result"]["routes"] == [
        {"route_trace_id": "route:chem-enzy:1"}
    ]
    assert executions[1]["outcome"]["failure_reasons"] == [failure_reason]
    assert kernel.state.settled_task_count == 2
    assert kernel.state.in_flight_tasks == {}


def test_slow_codex_peer_does_not_inflate_chemenzy_first_proposal_timing(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path / "first-proposal-latency")
    rendezvous = Barrier(2)
    chemenzy_returning = Event()

    def handle_chemenzy(_action) -> dict:
        rendezvous.wait(timeout=2.0)
        chemenzy_returning.set()
        return {
            "status": "completed",
            "routes": [{"route_trace_id": "route:chem-enzy:first"}],
        }

    def handle_codex(_action) -> dict:
        rendezvous.wait(timeout=2.0)
        assert chemenzy_returning.wait(timeout=2.0)
        sleep(0.05)
        return {"status": "completed", "plan": {"proposal_count": 1}}

    opportunities = compile_action_opportunities(
        {
            "content_sha256": "first-proposal-latency",
            "items": [
                {
                    "deficit_id": "deficit:target-native:first-proposal",
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
                    "deficit_id": "deficit:architecture:first-proposal",
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
            ],
        }
    )
    result = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.CHEMENZY_TARGET_EXPAND: handle_chemenzy,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE: handle_codex,
        },
    ).run_anytime(
        opportunity_provider=lambda: opportunities,
        milestones_provider=lambda: {},
        resource_availability_provider=lambda: {
            "native_search_target": True,
            "model": True,
        },
        max_actions=2,
        concurrent_start_kinds=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        ),
    )

    audit = result["start_cohort"]["latency_audit"]
    first = audit["chemenzy_first_proposal"]
    assert audit["applicable"] is True
    assert audit["accepted"] is True
    assert audit[
        "both_initial_providers_submitted_before_either_completed"
    ] is True
    assert audit["completion_order_action_kinds"][0] == (
        CampaignActionKind.CHEMENZY_TARGET_EXPAND.value
    )
    assert first["nonempty_raw_proposal_observed"] is True
    assert first["codex_peer_in_flight_at_chemenzy_completion"] is True
    assert first["peer_wait_excluded_s"] > 0
    assert result["first_result_timing"] == first
    assert result["semantics"][
        "first_proposal_timing_excludes_codex_peer_wait"
    ] is True


def test_result_delivery_materializes_and_stocks_before_slow_codex_peer(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path / "progressive-delivery")
    phase = {"value": "start"}
    delivery_cancelled = Event()
    codex_started = Event()

    def opportunities() -> dict:
        if phase["value"] == "start":
            items = [
                {
                    "deficit_id": "deficit:target-native:progressive",
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
                    "deficit_id": "deficit:architecture:progressive",
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
            ]
        elif phase["value"] == "materialize":
            items = [
                {
                    "deficit_id": "deficit:materialization:progressive",
                    "kind": "materialization",
                    "object_id": "hypothesis:1",
                    "entity_ids": ["hypothesis:1"],
                    "route_family_ids": ["route:1"],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 800.0,
                    "reason": "provider_route_requires_materialization",
                    "score": {"expected_portfolio_gain": 1.0},
                }
            ]
        elif phase["value"] == "stock":
            items = [
                {
                    "deficit_id": "deficit:stock:progressive",
                    "kind": "stock",
                    "object_id": "mol:leaf",
                    "entity_ids": ["mol:leaf"],
                    "route_family_ids": ["route:1"],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "priority": 700.0,
                    "reason": "selected_leaf_requires_trusted_stock_audit",
                    "score": {"expected_portfolio_gain": 1.0},
                }
            ]
        else:
            items = []
        return compile_action_opportunities(
            {"content_sha256": f"progressive-{phase['value']}", "items": items}
        )

    def handle_chemenzy(_action) -> dict:
        assert codex_started.wait(timeout=2.0)
        phase["value"] = "materialize"
        return {
            "status": "completed",
            "routes": [{"route_trace_id": "route:progressive:1"}],
        }

    def handle_codex(_action) -> dict:
        codex_started.set()
        assert delivery_cancelled.wait(timeout=2.0)
        return {
            "status": "cancelled",
            "reasons": ["delivery_milestone_reached"],
        }

    def handle_materialize(_action) -> dict:
        phase["value"] = "stock"
        return {"status": "completed", "changed": True}

    def handle_stock(_action) -> dict:
        phase["value"] = "closed"
        return {"status": "completed", "changed": True}

    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.CHEMENZY_TARGET_EXPAND: handle_chemenzy,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE: handle_codex,
            CampaignActionKind.MATERIALIZE: handle_materialize,
            CampaignActionKind.STOCK_AUDIT: handle_stock,
        },
    )
    result = runtime.run_anytime(
        opportunity_provider=opportunities,
        milestones_provider=lambda: {
            "B4_stock_boundary": phase["value"] == "closed"
        },
        resource_availability_provider=lambda: {
            "native_search_target": True,
            "model": True,
            "deterministic": True,
            "stock": True,
        },
        max_actions=4,
        concurrent_start_kinds=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        ),
        stop_milestone="B4_stock_boundary",
        progressive_start_kind=CampaignActionKind.CHEMENZY_TARGET_EXPAND,
        progressive_delivery_action_kinds=(
            CampaignActionKind.MATERIALIZE,
            CampaignActionKind.STOCK_AUDIT,
        ),
        on_delivery_milestone=delivery_cancelled.set,
    )

    assert [row["action"]["kind"] for row in result["executions"]] == [
        CampaignActionKind.CHEMENZY_TARGET_EXPAND.value,
        CampaignActionKind.MATERIALIZE.value,
        CampaignActionKind.STOCK_AUDIT.value,
        CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE.value,
    ]
    assert result["executions"][-1]["status"] == "cancelled"
    assert result["termination"] == "milestone_reached"
    assert delivery_cancelled.is_set()
    assert kernel.state.in_flight_tasks == {}
    assert result["start_cohort"]["semantics"][
        "completed_nondeferred_action_can_publish_before_peer_barrier"
    ] is True


def test_progressive_delivery_does_not_cancel_peer_before_milestone(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path / "progressive-no-delivery")
    codex_started = Event()
    chemenzy_finished = Event()
    delivery_cancelled = Event()

    def opportunities() -> dict:
        if chemenzy_finished.is_set():
            items = []
        else:
            items = [
                {
                    "deficit_id": "deficit:target-native:no-delivery",
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
                    "deficit_id": "deficit:architecture:no-delivery",
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
            ]
        return compile_action_opportunities(
            {"content_sha256": "progressive-no-delivery", "items": items}
        )

    def handle_chemenzy(_action) -> dict:
        assert codex_started.wait(timeout=2.0)
        chemenzy_finished.set()
        return {"status": "completed", "routes": []}

    def handle_codex(_action) -> dict:
        codex_started.set()
        assert chemenzy_finished.wait(timeout=2.0)
        return {"status": "completed", "routes": [{"route_trace_id": "codex:1"}]}

    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.CHEMENZY_TARGET_EXPAND: handle_chemenzy,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE: handle_codex,
        },
    )
    result = runtime.run_anytime(
        opportunity_provider=opportunities,
        milestones_provider=lambda: {"B4_stock_boundary": False},
        resource_availability_provider=lambda: {
            "native_search_target": True,
            "model": True,
        },
        max_actions=2,
        concurrent_start_kinds=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        ),
        stop_milestone="B4_stock_boundary",
        progressive_start_kind=CampaignActionKind.CHEMENZY_TARGET_EXPAND,
        progressive_delivery_action_kinds=(
            CampaignActionKind.MATERIALIZE,
            CampaignActionKind.STOCK_AUDIT,
        ),
        on_delivery_milestone=delivery_cancelled.set,
    )

    assert [row["status"] for row in result["executions"]] == [
        "completed",
        "completed",
    ]
    assert result["termination"] != "milestone_reached"
    assert delivery_cancelled.is_set() is False
    assert kernel.state.in_flight_tasks == {}


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
