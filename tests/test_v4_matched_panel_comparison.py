from __future__ import annotations

import pytest

from scripts.compare_v4_matched_panels import compare_matched_panels


def _row(case_id: str, *, b4: bool, terminal: str) -> dict:
    return {
        "case_id": case_id,
        "target_name": f"target {case_id}",
        "status": "completed",
        "terminal_disposition": terminal,
        "gate_summary": {"B1": True, "B2": False, "B4": b4},
        "provider_search_attempts": [
            {
                "raw_solved": True,
                "host_admitted_solved": True,
                "native_raw_route_count": 2,
                "output_route_count": 1,
            }
        ],
        "resource_observed": {
            "input_tokens": 10,
            "native_search_committed": 1,
        },
        "runtime_recovery": {"provider_result_replay_count": 0},
        "result_action_trace": {
            "recompute_route_closure_count": 0,
            "guided_before_first_route_closure": 1,
            "guided_after_first_route_closure": 0,
            "campaign_termination": "no_action",
        },
    }


def test_matched_comparison_reports_transitions_actions_and_costs() -> None:
    baseline = {
        "content_sha256": "a" * 64,
        "per_target": [
            _row("one", b4=False, terminal="completed_b4_open"),
            _row("two", b4=False, terminal="completed_b4_open"),
            _row("baseline-only", b4=True, terminal="completed_b4_stock_closed"),
        ],
    }
    solved = _row("one", b4=True, terminal="completed_b4_stock_closed")
    solved["resource_observed"]["input_tokens"] = 14
    solved["result_action_trace"] = {
        "recompute_route_closure_count": 1,
        "guided_before_first_route_closure": 1,
        "guided_after_first_route_closure": 2,
        "campaign_termination": "milestone_reached",
    }
    candidate = {
        "content_sha256": "b" * 64,
        "per_target": [
            solved,
            _row("two", b4=False, terminal="completed_b4_open"),
        ],
    }

    result = compare_matched_panels(baseline, candidate)

    assert result["case_count"] == 2
    assert result["performance_claim_eligible"] is True
    assert result["b4"] == {
        "baseline_count": 0,
        "candidate_count": 1,
        "count_delta": 1,
        "transition_counts": {
            "newly_stock_closed": 1,
            "stock_closed_preserved": 0,
            "still_stock_open": 1,
            "stock_closed_regressed": 0,
        },
    }
    first = result["per_target"][0]
    assert first["case_id"] == "one"
    assert first["b4_transition"] == "newly_stock_closed"
    assert first["candidate"]["recompute_route_closure_count"] == 1
    assert first["candidate"]["guided_after_first_route_closure"] == 2
    assert first["resource_delta"]["input_tokens"] == 4


def test_pending_candidate_disables_performance_claim() -> None:
    old = _row("one", b4=False, terminal="completed_b4_open")
    pending = _row("one", b4=False, terminal="pending_running")
    pending["status"] = "running"

    result = compare_matched_panels(
        {"per_target": [old]},
        {"per_target": [pending]},
    )

    assert result["all_candidate_cases_terminal"] is False
    assert result["performance_claim_eligible"] is False


def test_candidate_case_missing_from_baseline_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="candidate_cases_missing_from_baseline:new",
    ):
        compare_matched_panels(
            {"per_target": [_row("old", b4=False, terminal="open")]},
            {"per_target": [_row("new", b4=True, terminal="closed")]},
        )
