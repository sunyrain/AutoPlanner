from __future__ import annotations

from scripts.summarize_v4_blind_panel import summarize_panel


def test_panel_summary_keeps_stock_solved_separate_from_proof_acceptance() -> None:
    summary = summarize_panel(
        {
            "target_count": 2,
            "targets": {
                "target 1": {
                    "status": "completed",
                    "case_id": "one",
                    "accepted_under_configured_policy": False,
                    "within_resource_budget": True,
                    "route_counts": {
                        "target_rooted_distinct_skeletons": 1,
                        "materialized_skeletons": 1,
                        "reaction_validated_skeletons": 0,
                        "stock_closed_skeletons": 1,
                        "evidence_closed_skeletons": 0,
                    },
                    "model_cost": {"model_invocations": 2, "input_tokens": 100},
                    "elapsed_s": 10,
                },
                "target 2": {
                    "status": "failed",
                    "case_id": "two",
                    "error": "provider timeout",
                },
            },
        }
    )

    assert summary["metrics"]["official_benchmark_stock_closed"] == {
        "count": 1,
        "rate_over_full_panel": 0.5,
        "rate_over_completed": 1.0,
    }
    assert summary["metrics"]["configured_proof_policy_accepted"]["count"] == 0
    assert summary["resource_totals"]["model_invocations"] == 2
    assert summary["counts"]["failed_or_incomplete"] == 1
