from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.application.blind_benchmark_contract import BlindCase
from cascade_planner.application.campaign_trajectory import (
    compile_campaign_snapshot,
    compile_campaign_trajectory,
)
from scripts.run_v4_blind_panel import (
    _ablation_cli_args,
    _acceptance_cli_args,
    _prepare_panel_snapshot,
    _run_id_for_case,
    _select_cases,
    _summarize_report,
)


def test_blind_panel_run_id_uses_manifest_identity_without_statin_hardcoding() -> None:
    case = BlindCase(
        case_id="complex-target-v4-blind-01",
        target_name="unrelated target family",
        target_smiles="CCO",
    )

    assert _run_id_for_case(case) == "complex-target-v4-blind-01"
    assert "statin" not in _run_id_for_case(case)


def test_blind_panel_passes_manifest_acceptance_without_target_hardcoding() -> None:
    case = BlindCase(
        case_id="complex-target-v4-blind-02",
        target_name="another family",
        target_smiles="CCO",
        acceptance={
            "minimum_complete_routes": 3,
            "minimum_edge_proof_level": 1,
            "minimum_independent_source_groups": 4,
            "minimum_planning_route_steps": 20,
            "stock_boundary": "procurement",
        },
    )

    args = _acceptance_cli_args(case)

    assert args == [
        "--minimum-complete-routes",
        "3",
        "--minimum-edge-proof-level",
        "1",
        "--minimum-source-groups",
        "4",
        "--minimum-planning-route-steps",
        "20",
        "--stock-boundary",
        "procurement",
    ]


def test_blind_panel_ablation_changes_only_the_declared_subsystem() -> None:
    assert _ablation_cli_args("baseline") == []
    assert _ablation_cli_args("no-chemenzy") == [
        "--no-chemenzy",
        "--no-guided-chemenzy",
    ]
    assert _ablation_cli_args("no-self-evo") == ["--no-patent-self-evo"]
    assert _ablation_cli_args("no-replan") == ["--no-replan"]
    assert _ablation_cli_args("chemenzy-only") == ["--no-codex"]
    assert _ablation_cli_args("codex-only") == [
        "--no-chemenzy",
        "--no-guided-chemenzy",
    ]
    assert _ablation_cli_args("unified-round-robin") == [
        "--action-scheduler",
        "round_robin",
    ]
    assert _ablation_cli_args("unified-adaptive") == [
        "--action-scheduler",
        "adaptive",
    ]


def test_blind_panel_freezes_a_manifest_ordered_target_pilot() -> None:
    cases = [
        BlindCase(
            case_id=f"blind-{index:03d}",
            target_name=f"opaque target {index:03d}",
            target_smiles="CCO",
        )
        for index in range(1, 36)
    ]

    selected = _select_cases(cases, max_targets=30)

    assert len(selected) == 30
    assert [case.case_id for case in selected[:2]] == ["blind-001", "blind-002"]
    assert selected[-1].case_id == "blind-030"


def test_blind_panel_rejects_a_nonpositive_target_pilot_limit() -> None:
    case = BlindCase(
        case_id="blind-001",
        target_name="opaque target 001",
        target_smiles="CCO",
    )

    try:
        _select_cases([case], max_targets=0)
    except ValueError as exc:
        assert str(exc) == "max_targets must be a positive integer"
    else:
        raise AssertionError("expected max_targets validation")


def test_panel_snapshot_keeps_ablation_out_of_base_environment_but_binds_workers(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    case = BlindCase(case_id="blind-01", target_name="opaque target", target_smiles="CCO")

    def freeze(name: str, *, ablation: str, workers: int) -> dict:
        return _prepare_panel_snapshot(
            output_root=tmp_path / name,
            manifest=manifest,
            cases=[case],
            model="gpt-5.5",
            reasoning_effort="low",
            execution_profile="standard",
            worker_count=workers,
            visual=False,
            ablation=ablation,
            chemenzy_env_prefix=None,
            self_evo_library_seed=None,
            inventory_snapshot=None,
            leakage_audit_pack=None,
            resume=False,
        )

    baseline = freeze("baseline", ablation="baseline", workers=1)
    no_replan = freeze("no-replan", ablation="no-replan", workers=1)
    two_workers = freeze("two-workers", ablation="baseline", workers=2)

    assert baseline["base_environment_sha256"] == no_replan["base_environment_sha256"]
    assert baseline["content_sha256"] != no_replan["content_sha256"]
    assert baseline["base_environment_sha256"] != two_workers["base_environment_sha256"]


def test_panel_summary_scores_only_the_fixed_cutoff_trajectory_projection(
    tmp_path: Path,
) -> None:
    snapshot = compile_campaign_snapshot(
        phase="fixed-cutoff-observation",
        observed_at="2026-08-10T00:00:00Z",
        event_sequence=3,
        graph_revision=2,
        wall_time_s=10.0,
        gates={
            "gates": {
                "B0_blind_input": True,
                "B4_stock_boundary": True,
                "B5_configured_portfolio_acceptance": False,
            },
            "counts": {"stock_closed_skeletons": 3},
        },
        resource_usage={
            "model": {
                "model_invocations": 1,
                "visual_invocations": 0,
                "input_tokens": 100,
                "output_tokens": 20,
                "wall_time_s": 2.0,
            },
            "tasks": {"dimensions": {"total": {"settled": 3}}},
            "attempt_count": 3,
            "accepted_expansion_count": 2,
            "settled_task_count": 3,
        },
        route_counts={"stock_closed_route_count": 3},
    )
    trajectory = compile_campaign_trajectory([snapshot])
    report_path = tmp_path / "target-only-solve-report.json"
    report_path.write_text(
        json.dumps(
            {
                "run_id": "benchmark-search-only",
                "run_dir": str(tmp_path),
                "preflight": {"case": {"case_id": "case-1"}},
                "claim": {
                    "achieved_profile": "exploration_closed",
                    "objective_mode": "benchmark_search",
                    "objective_achieved": True,
                    "benchmark_search_completed": True,
                    "accepted_under_configured_policy": False,
                },
                "gates": {
                    "gates": {
                        "B0_blind_input": True,
                        "B4_stock_boundary": True,
                        "B5_configured_portfolio_acceptance": False,
                    },
                    "counts": {"stock_closed_skeletons": 3},
                },
                "stages": [
                    {
                        "stage": "chemenzy_baseline",
                        "status": "completed",
                        "detail": {
                            "status": "completed",
                            "proposal_count": 17,
                        },
                    }
                ],
                "resource_envelope": {"within_budget": True},
                "trajectory": trajectory,
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report(
        report_path,
        elapsed_s=12.5,
        reused=False,
        cutoff={"wall_time_s": 20.0, "settled_task_count": 8},
    )

    assert summary["claim"] == "fixed_cutoff_stock_closed"
    assert "objective_mode" not in summary
    assert summary["accepted_under_configured_policy"] is False
    assert summary["route_counts"]["stock_closed_skeletons"] == 3
    assert summary["model_cost"]["model_invocations"] == 1
    assert summary["elapsed_s"] == 10.0
    assert summary["runner_elapsed_s"] == 12.5
    assert summary["fixed_cutoff_projection"]["available"] is True
    assert summary["final_state"]["claim"]["objective_mode"] == "benchmark_search"
    assert summary["chemenzy"]["status"] == "completed"
    assert summary["chemenzy"]["proposal_count"] == 17
