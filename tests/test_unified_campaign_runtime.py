from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.campaign_actions import (
    CampaignActionKind,
    bind_scheduled_action,
    compile_action_opportunities,
)
from cascade_planner.application.run_kernel import RunKernel, RunSpec
from cascade_planner.orchestration.unified_campaign_runtime import (
    CampaignActionRuntime,
    CampaignActionRuntimeError,
)


def _kernel(tmp_path: Path) -> RunKernel:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="campaign-action-test",
            target_name="target",
            target_smiles="CCO",
            created_at="2026-08-06T00:00:00Z",
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
    lifecycle = kernel.task_lifecycle(action.task_id)
    assert lifecycle["reservation"]["payload"]["kind"] == "other"
    assert lifecycle["status"] == "settled"
    assert lifecycle["settlement"]["payload"]["status"] == "completed"
    assert kernel.state.attempt_count == 0
    assert kernel.state.model_totals["model_invocations"] == 0


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
    assert result["semantics"]["single_scheduler_loop"] is True


def test_same_revision_start_cohort_runs_peers_without_failure_cancellation(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    chemenzy_started = Event()
    codex_started = Event()

    def handle_chemenzy(_action) -> dict:
        chemenzy_started.set()
        assert codex_started.wait(timeout=2.0)
        return {"status": "completed", "proposal_count": 2}

    def handle_codex(_action) -> dict:
        codex_started.set()
        assert chemenzy_started.wait(timeout=2.0)
        raise RuntimeError("codex failed after peer launch")

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
            ],
        }
    )
    runtime = CampaignActionRuntime(
        kernel,
        {
            CampaignActionKind.CHEMENZY_TARGET_EXPAND: handle_chemenzy,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE: handle_codex,
        },
    )

    result = runtime.run_anytime(
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

    cohort = result["start_cohort"]
    executions = result["executions"]
    assert cohort["status"] == "completed"
    assert cohort["max_in_flight_action_count"] == 2
    assert [row["action"]["kind"] for row in executions] == [
        CampaignActionKind.CHEMENZY_TARGET_EXPAND.value,
        CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE.value,
    ]
    assert {row["action"]["input_revision"] for row in executions} == {0}
    assert [row["status"] for row in executions] == ["completed", "failed"]
    assert executions[1]["outcome"]["failure_reasons"] == [
        "campaign_action_handler_error:RuntimeError:codex failed after peer launch"
    ]
    assert kernel.state.settled_task_count == 2
    assert kernel.state.in_flight_tasks == {}

    replay = runtime.execute_concurrent_cohort(
        opportunities,
        action_kinds=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND,
            CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        ),
        milestones={},
        resource_availability={"native_search_target": True, "model": True},
    )

    assert replay["cohort_id"] == cohort["cohort_id"]
    assert replay["action_execution_ids"] == cohort["action_execution_ids"]
    assert replay["max_in_flight_action_count"] == 0
    assert all(row["cache_hit"] is True for row in replay["executions"])
    assert kernel.state.settled_task_count == 2


def test_program_validation_and_feedback_share_validation_action_accounting(
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
        resource_availability_provider=lambda: {"validation": True},
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
    assert all(
        kernel.task_lifecycle(row["action"]["task_id"])["reservation"][
            "payload"
        ]["metadata"]["delegated_resource_class"]
        == "validation"
        for row in result["executions"]
    )
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
