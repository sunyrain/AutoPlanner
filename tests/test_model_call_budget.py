from dataclasses import replace
import time

from cascade_planner.agent.codex_worker import WorkerRunRecord, WorkerTask
from cascade_planner.orchestration.model_call_budget import (
    NodeCallBudget,
    SharedModelCallLedger,
    budget_exposure,
    call_token_reserve,
)
from cascade_planner.orchestration.sequential_strategy_director import _node_budget_block_reason


def record(
    task_id,
    usage,
    *,
    task_type="paper_matched_route_step",
    model="model-a",
    status="accepted_draft",
):
    return WorkerRunRecord(
        run_id=task_id,
        task_id=task_id,
        case_id="test",
        status=status,
        usage=usage,
        metadata={"task_type": task_type, "model": model},
    )


def test_adaptive_reserve_preserves_final_reviews_and_separates_models():
    rows = [
        record("builder", {"input_tokens": 60_000, "output_tokens": 10_000}),
        record(
            "critic",
            {"input_tokens": 115_000, "output_tokens": 18_000},
            task_type="paper_matched_key_event_critic",
        ),
        record("other", {"input_tokens": 900_000, "output_tokens": 1}, model="model-b"),
    ]
    assert call_token_reserve(rows, "builder", "input_tokens", model="model-a") == 60_000
    assert call_token_reserve(rows, "critic", "input_tokens", model="model-a") == 115_000
    quota = NodeCallBudget(8, 500_000, 200_000, 60)
    ledger = SharedModelCallLedger(
        quota,
        rows[:2],
        protected_model_invocations=2,
        protected_input_tokens=48_000,
        protected_output_tokens=32_000,
    )
    task = WorkerTask(
        "next", "test", "paper_matched_route_step", "RetrosynthesisProposalReport", model="model-a"
    )
    admitted, _ = ledger.reserve(task=task)
    assert admitted.input_tokens == 60_000
    assert ledger.snapshot()["protected_final_critics"]["input_tokens"] == 230_000
    assert ledger.reserve(task=task) == (None, "input_token_allocation_exhausted")


def test_timeout_hold_survives_resume_and_outer_budget_checks_without_inventing_usage():
    quota = NodeCallBudget(8, 80_000, 80_000, 60)
    ledger = SharedModelCallLedger(quota, [])
    reservation, _ = ledger.reserve(input_tokens=60_000, output_tokens=20_000)
    failed = record("timeout", {}, status="timeout")
    ledger.settle(reservation, failed)
    assert failed.usage == {}
    assert ledger.snapshot()["committed"]["input_tokens"] == 0
    exposure = ledger.snapshot()["budget_exposure"]
    assert exposure["unknown_input_tokens_held"] == 60_000
    restored = WorkerRunRecord(**failed.to_dict())
    resumed = SharedModelCallLedger(quota, [restored])
    assert resumed.snapshot()["budget_exposure"] == exposure
    assert resumed.reserve(input_tokens=24_000, output_tokens=16_000)[0] is None
    assert (
        _node_budget_block_reason(
            [restored], quota=replace(quota, input_tokens=60_000), started=time.monotonic()
        )
        == "input_token_allocation_exhausted"
    )


def test_partial_and_legacy_unknown_use_estimates_but_real_zero_and_aliases_stay_known():
    known = record("known", {"input_tokens": 60_000, "output_tokens": 20_000})
    partial = record("partial", {"input_tokens": 0, "prompt_tokens": 99}, status="timeout")
    exposure = budget_exposure([known, partial])
    assert exposure["input_tokens"] == 60_000
    assert exposure["output_tokens"] == 40_000
    assert exposure["unknown_input_tokens_held"] == 0
    assert exposure["unknown_output_tokens_held"] == 20_000
    assert (
        budget_exposure([record("zero", {"prompt_tokens": 0, "completion_tokens": 0})])[
            "usage_incomplete_calls"
        ]
        == 0
    )


def test_provider_failure_preserves_reported_spend_without_consuming_semantic_turn():
    failure = record("auth", {}, status="provider_error")
    failure.metadata["provider_failure_reason"] = "provider_auth_unavailable"
    assert budget_exposure([failure])["input_tokens"] == 0
    failure.usage = {"input_tokens": 900, "output_tokens": 100}
    assert budget_exposure([failure])["input_tokens"] == 900
    assert budget_exposure([failure])["model_invocations"] == 0


def test_outer_settlement_and_director_normalization_preserve_unknown_holds():
    from cascade_planner.interfaces.target_solver import _stage_model_usage
    from cascade_planner.orchestration.global_campaign_director import normalize_director_usage
    from cascade_planner.orchestration.sequential_strategy_director import _aggregate_usage

    estimate = {"input_tokens": 60_000, "output_tokens": 20_000}
    timeout = record("timeout", {"input_tokens": 123}, status="timeout")
    timeout.metadata["budget_reservation"] = estimate
    for usage in (
        _stage_model_usage(timeout, estimate),
        normalize_director_usage(_aggregate_usage([timeout], elapsed_s=1)),
    ):
        assert usage["input_tokens"] == 123
        assert usage["output_tokens"] == 0
        assert usage["unknown_output_tokens_held"] == 20_000
        assert usage.get("unknown_input_tokens_held", 0) == 0
    assert _stage_model_usage(None, estimate)["unknown_input_tokens_held"] == 60_000
    auth = record("auth", {}, status="provider_error")
    auth.metadata["provider_failure_reason"] = "provider_auth_unavailable"
    assert _stage_model_usage(auth, estimate).get("unknown_input_tokens_held", 0) == 0
