from __future__ import annotations

from cascade_planner.interfaces.target_cli import _compact_target_result


def test_compact_target_result_omits_route_and_stage_payloads() -> None:
    summary = _compact_target_result(
        {
            "run_id": "blind-one",
            "target": {"name": "one", "canonical_smiles": "CCO"},
            "gates": {
                "gates": {"B0_blind_input": True},
                "highest_contiguous_gate": "B0",
                "counts": {"target_rooted_distinct_skeletons": 3},
            },
            "claim": {"accepted_under_configured_policy": False},
            "current_disposition": {"state": "unresolved"},
            "model_cost": {"model_invocations": 1},
            "resource_envelope": {
                "within_budget": True,
                "observed": {"input_tokens": 10},
                "violations": [],
            },
            "attempt_count": 4,
            "accepted_expansion_count": 3,
            "stop_decision": {"decision": "continue"},
            "report_path": "target-only-solve-report.json",
            "content_sha256": "a" * 64,
            "director_outcomes": [{"plan": {"multi_step_skeletons": ["large"]}}],
            "stages": [{"detail": {"large": True}}],
        }
    )

    assert summary["schema_version"] == "target_solve_cli_summary.v1"
    assert summary["highest_contiguous_gate"] == "B0"
    assert summary["report_sha256"] == "a" * 64
    assert "director_outcomes" not in summary
    assert "stages" not in summary
