from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade_planner.application.blind_benchmark_contract import (
    BlindBenchmarkError,
    BlindCase,
)
from cascade_planner.application.campaign_trajectory import (
    compile_campaign_snapshot,
    compile_campaign_trajectory,
)
from scripts.run_v4_blind_panel import (
    _ablation_cli_args,
    _acceptance_cli_args,
    _guided_chemenzy_cli_args,
    _prior_target_manifest_files,
    _prepare_panel_snapshot,
    _resolve_panel_chemenzy_stock_binding,
    _resolved_model_wall_time_s,
    _resume_completed_targets,
    _run_id_for_case,
    _select_cases,
    _summarize_report,
    _validate_matched_baseline,
    _write_matched_comparison,
    _write_result_summary,
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


def test_guided_chemenzy_uses_the_same_matched_short_tail_for_every_profile() -> None:
    fast = _guided_chemenzy_cli_args("fast")
    standard = _guided_chemenzy_cli_args("standard")

    assert fast == [
        "--guided-chemenzy-iterations",
        "500",
        "--guided-chemenzy-timeout-s",
        "1200.0",
    ]
    assert standard == [
        "--guided-chemenzy-iterations",
        "500",
        "--guided-chemenzy-timeout-s",
        "1200.0",
    ]
    assert "--guided-chemenzy-frontiers" not in fast
    assert "--guided-chemenzy-frontiers" not in standard


def test_fixed_cutoff_caps_an_older_larger_manifest_model_budget() -> None:
    assert _resolved_model_wall_time_s(
        case_budget={"max_total_wall_time_s": 7_200},
        fixed_cutoff_wall_time_s=1_800,
    ) == 1_800


def test_larger_cutoff_preserves_the_matched_model_budget_floor() -> None:
    assert _resolved_model_wall_time_s(
        case_budget={"max_total_wall_time_s": 900},
        fixed_cutoff_wall_time_s=7_200,
    ) == 1_800


def test_panel_binds_benchmark_stock_into_chemenzy_by_default(tmp_path: Path) -> None:
    stock = tmp_path / "benchmark.sqlite3"
    stock.write_bytes(b"fixture")

    names, paths = _resolve_panel_chemenzy_stock_binding(
        benchmark_stock_index=str(stock),
        benchmark_stock_name="FrozenBenchmarkStock",
        stock_names=(),
        stock_paths=(),
    )

    assert names == ("FrozenBenchmarkStock",)
    assert paths == (f"FrozenBenchmarkStock={stock.resolve()}",)


def test_panel_rejects_a_provider_stock_different_from_scoring_stock(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.sqlite3"
    provider = tmp_path / "provider.csv"
    benchmark.write_bytes(b"benchmark")
    provider.write_text("smiles\nCCO\n", encoding="utf-8")

    try:
        _resolve_panel_chemenzy_stock_binding(
            benchmark_stock_index=str(benchmark),
            benchmark_stock_name="benchmark",
            stock_names=("provider",),
            stock_paths=(f"provider={provider}",),
        )
    except ValueError as exc:
        assert str(exc) == "benchmark_and_chemenzy_stock_paths_differ"
    else:
        raise AssertionError("expected provider/scoring stock mismatch rejection")


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


def test_known_target_reproduction_binds_only_valid_prior_target_manifests(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior-targets.json"
    prior.write_text(
        json.dumps(
            {
                "schema_version": "blind_retrosynthesis_manifest.v1",
                "cases": [
                    BlindCase(
                        case_id="prior-001",
                        target_name="opaque prior target 001",
                        target_smiles="CCO",
                    ).to_dict()
                ],
            }
        ),
        encoding="utf-8",
    )

    paths = _prior_target_manifest_files([str(prior)])

    assert paths == (prior.resolve(),)


def test_known_target_reproduction_rejects_non_manifest_prior_artifact(
    tmp_path: Path,
) -> None:
    route_answer = tmp_path / "route-answer.json"
    route_answer.write_text('{"route": "CCO>>CC"}', encoding="utf-8")

    with pytest.raises(BlindBenchmarkError):
        _prior_target_manifest_files([str(route_answer)])


def test_completed_panel_immediately_publishes_result_first_summary(
    tmp_path: Path,
) -> None:
    state = {
        "schema_version": "v4_blind_panel_status.v1",
        "target_count": 2,
        "complete": True,
        "targets": {
            "solved": {
                "status": "completed",
                "case_id": "solved",
                "gate_summary": {"B1": True, "B2": False, "B4": True},
                "route_counts": {"stock_closed_skeletons": 1},
            },
            "failed": {
                "status": "failed",
                "case_id": "failed",
                "error": "provider timeout",
            },
        },
    }

    path = _write_result_summary(tmp_path, state)

    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "v4_blind_panel_summary.v3"
    assert summary["outcome_accounting"]["accounted_target_count"] == 2
    assert summary["outcome_accounting"]["unaccounted_target_count"] == 0
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Outcome accounting" in markdown
    assert "| terminal_failed | 1 |" in markdown


def test_panel_validates_and_publishes_matched_comparison(tmp_path: Path) -> None:
    case = BlindCase(
        case_id="case-1",
        target_name="target",
        target_smiles="CCO",
    )
    baseline = {
        "content_sha256": "a" * 64,
        "per_target": [
            {
                "case_id": "case-1",
                "target_name": "target",
                "status": "completed",
                "terminal_disposition": "completed_b4_open",
                "gate_summary": {"B4": False},
            }
        ],
    }
    candidate = {
        "content_sha256": "b" * 64,
        "per_target": [
            {
                "case_id": "case-1",
                "target_name": "target",
                "status": "completed",
                "terminal_disposition": "completed_b4_stock_closed",
                "gate_summary": {"B4": True},
            }
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "panel-summary.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    _validate_matched_baseline(baseline, cases=[case])
    path = _write_matched_comparison(
        tmp_path,
        baseline=baseline,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["b4"]["count_delta"] == 1
    assert result["performance_claim_eligible"] is True
    assert "Root B4: 0 -> 1" in path.with_suffix(".md").read_text(
        encoding="utf-8"
    )


def test_panel_rejects_a_tampered_v3_matched_baseline() -> None:
    case = BlindCase(
        case_id="case-1",
        target_name="target",
        target_smiles="CCO",
    )
    with pytest.raises(ValueError, match="matched_baseline_summary_digest_invalid"):
        _validate_matched_baseline(
            {
                "schema_version": "v4_blind_panel_summary.v3",
                "content_sha256": "0" * 64,
                "per_target": [{"case_id": "case-1"}],
            },
            cases=[case],
        )


def test_resume_reuses_only_completed_report_backed_targets(tmp_path: Path) -> None:
    cases = [
        BlindCase(case_id=f"case-{index}", target_name=f"target-{index}", target_smiles="CCO")
        for index in range(2)
    ]
    snapshot = {"content_sha256": "a" * 64}
    report = tmp_path / "runs" / "target-0" / "target-only-solve-report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}\n", encoding="utf-8")
    previous = {
        "schema_version": "v4_blind_panel_status.v1",
        "ablation": "codex-only",
        "selection": {"selected_case_ids": ["case-0", "case-1"]},
        "frozen_snapshot": {"content_sha256": "a" * 64},
        "targets": {
            "target-0": {"status": "completed", "case_id": "case-0"},
            "target-1": {"status": "running", "case_id": "case-1"},
        },
    }

    resumed = _resume_completed_targets(
        previous,
        cases=cases,
        output_root=tmp_path,
        snapshot=snapshot,
        ablation="codex-only",
    )

    assert list(resumed) == ["target-0"]
    assert resumed["target-0"]["resume_reused_completed_report"] is True


def test_panel_snapshot_keeps_ablation_out_of_base_environment_but_binds_workers(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    case = BlindCase(
        case_id="blind-01", target_name="opaque target", target_smiles="CCO"
    )

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
        route_counts={
            "stock_closed_route_count": 3,
            "condition_complete_route_count": 2,
        },
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
    assert summary["route_counts"]["stock_closed_route_count"] == 3
    assert summary["route_counts"]["condition_complete_route_count"] == 2
    assert summary["model_cost"]["model_invocations"] == 1
    assert summary["elapsed_s"] == 10.0
    assert summary["runner_elapsed_s"] == 12.5
    assert summary["fixed_cutoff_projection"]["available"] is True
    assert summary["final_state"]["claim"]["objective_mode"] == "benchmark_search"
    assert summary["chemenzy"]["status"] == "completed"
    assert summary["chemenzy"]["proposal_count"] == 17


def test_panel_does_not_mark_nonaccepted_terminal_report_as_completed(
    tmp_path: Path,
) -> None:
    snapshot = compile_campaign_snapshot(
        phase="budget-exhausted",
        observed_at="2026-08-10T00:00:00Z",
        event_sequence=1,
        graph_revision=0,
        wall_time_s=1.0,
        gates={"gates": {}, "counts": {}},
        resource_usage={"tasks": {"dimensions": {"total": {"settled": 1}}}},
        route_counts={},
    )
    report_path = tmp_path / "target-only-solve-report.json"
    report_path.write_text(
        json.dumps(
            {
                "run_id": "case-budget",
                "preflight": {"case": {"case_id": "case-budget"}},
                "claim": {
                    "accepted_under_configured_policy": False,
                    "achieved_profile": "unresolved",
                },
                "stop_decision": {
                    "decision": "budget_exhausted",
                    "terminal": True,
                },
                "current_disposition": {
                    "state": "budget_exhausted",
                    "scientifically_accepted": False,
                },
                "gates": {},
                "trajectory": compile_campaign_trajectory([snapshot]),
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report(
        report_path,
        elapsed_s=1.0,
        reused=False,
        cutoff={"wall_time_s": 2.0, "settled_task_count": 2},
    )

    assert summary["fixed_cutoff_projection"]["available"] is True
    assert summary["status"] == "incomplete"


def test_panel_summary_retains_stage_failure_reasons(tmp_path: Path) -> None:
    snapshot = compile_campaign_snapshot(
        phase="failed-observation",
        observed_at="2026-08-10T00:00:00Z",
        event_sequence=1,
        graph_revision=0,
        wall_time_s=1.0,
        gates={"gates": {}, "counts": {}},
        resource_usage={"tasks": {"dimensions": {"total": {"settled": 1}}}},
        route_counts={},
    )
    report_path = tmp_path / "target-only-solve-report.json"
    report_path.write_text(
        json.dumps(
            {
                "run_id": "case-1",
                "preflight": {"case": {"case_id": "case-1"}},
                "gates": {},
                "stages": [
                    {
                        "stage": "global_campaign",
                        "status": "failed",
                        "detail": {
                            "status": "failed",
                            "reasons": [
                                "GlobalCampaignPlanValidationError",
                                "plan_context_sha256_mismatch",
                            ],
                        },
                    }
                ],
                "trajectory": compile_campaign_trajectory([snapshot]),
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report(
        report_path,
        elapsed_s=1.0,
        reused=False,
        cutoff={"wall_time_s": 10.0, "settled_task_count": 10},
    )

    assert summary["failure_events"] == [
        {
            "stage": "global_campaign",
            "status": "failed",
            "reasons": [
                "GlobalCampaignPlanValidationError",
                "plan_context_sha256_mismatch",
            ],
        }
    ]
