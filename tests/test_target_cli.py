from __future__ import annotations

import argparse

from cascade_planner.interfaces.target_cli import (
    _compact_target_result,
    add_target_commands,
)


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


def test_target_cli_visual_evidence_is_explicitly_opt_in_and_bounded() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)

    default = parser.parse_args(
        ["solve-target", "--target-smiles", "CCOC(C)=O"]
    )
    opted_in = parser.parse_args(
        [
            "solve-target",
            "--target-smiles",
            "CCOC(C)=O",
            "--max-visual-invocations",
            "1",
            "--max-visual-pages",
            "2",
        ]
    )

    assert default.max_visual_invocations == 0
    assert default.max_visual_pages == 6
    assert default.target_name == ""
    assert opted_in.max_visual_invocations == 1
    assert opted_in.max_visual_pages == 2


def test_target_cli_exposes_bounded_chemenzy_runtime_controls() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)

    default = parser.parse_args(["solve-target", "--target-smiles", "CCO"])
    configured = parser.parse_args(
        [
            "solve-target",
            "--target-smiles",
            "CCO",
            "--chemenzy-env-prefix",
            "D:/isolated/chemenzy",
            "--chemenzy-max-routes",
            "3",
            "--chemenzy-iterations",
            "7",
            "--chemenzy-expansion-topk",
            "12",
            "--chemenzy-timeout-s",
            "45",
        ]
    )

    assert default.no_chemenzy is False
    assert default.chemenzy_iterations == 10
    assert configured.chemenzy_env_prefix == "D:/isolated/chemenzy"
    assert configured.chemenzy_max_routes == 3
    assert configured.chemenzy_iterations == 7
    assert configured.chemenzy_expansion_topk == 12
    assert configured.chemenzy_timeout_s == 45.0
