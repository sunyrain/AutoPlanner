from __future__ import annotations

import argparse

import pytest

from cascade_planner.interfaces.target_cli import (
    _compact_target_result,
    _parse_safety_limits,
    _resolve_chemenzy_stock_binding,
    _resolve_objective_compatibility_view,
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
            "quality_state": {
                "schema_version": "campaign_quality_state.v1",
                "axes": {"topology": {"state": "satisfied"}},
            },
            "model_cost": {"model_invocations": 1},
            "resource_envelope": {
                "within_budget": True,
                "observed": {"input_tokens": 10},
                "task_budget": {
                    "schema_version": "campaign_task_budget.v1",
                    "dimensions": {"program": {"limit": 7}},
                },
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
    assert summary["quality_state"]["axes"]["topology"]["state"] == "satisfied"
    assert summary["resource_envelope"]["task_budget"]["dimensions"]["program"] == {
        "limit": 7
    }
    assert "director_outcomes" not in summary
    assert "stages" not in summary


def test_target_cli_visual_evidence_is_explicitly_opt_in_and_bounded() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)

    default = parser.parse_args(["solve-target", "--target-smiles", "CCOC(C)=O"])
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


def test_target_cli_objective_mode_is_deprecated_compatibility_only() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)

    default = parser.parse_args(["solve-target", "--target-smiles", "CCO"])
    legacy = parser.parse_args(
        [
            "solve-target",
            "--target-smiles",
            "CCO",
            "--objective-mode",
            "benchmark_search",
        ]
    )

    assert default.delivery_boundary == "stock_result"
    assert default.objective_mode is None
    assert _resolve_objective_compatibility_view(default.objective_mode) == (
        "scientific_proof"
    )
    with pytest.warns(FutureWarning, match="compatibility metadata"):
        assert _resolve_objective_compatibility_view(legacy.objective_mode) == (
            "benchmark_search"
        )


def test_target_cli_exposes_planning_depth_and_prompt_budget() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)

    args = parser.parse_args(
        [
            "solve-target",
            "--target-smiles",
            "CCO",
            "--minimum-planning-route-steps",
            "20",
            "--max-prompt-context-bytes",
            "256000",
            "--max-total-tasks",
            "300",
            "--max-evidence-tasks",
            "20",
            "--max-stock-tasks",
            "30",
            "--max-validation-tasks",
            "40",
            "--max-program-tasks",
            "12",
            "--max-experiment-tasks",
            "5",
            "--max-run-wall-time-s",
            "1800",
        ]
    )

    assert args.minimum_planning_route_steps == 20
    assert args.max_prompt_context_bytes == 256000
    assert args.max_total_tasks == 300
    assert args.max_evidence_tasks == 20
    assert args.max_stock_tasks == 30
    assert args.max_validation_tasks == 40
    assert args.max_program_tasks == 12
    assert args.max_experiment_tasks == 5
    assert args.max_run_wall_time_s == 1800


def test_target_cli_exposes_dataset_blind_campaign_constraints() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)

    args = parser.parse_args(
        [
            "solve-target",
            "--target-smiles",
            "CCO",
            "--forbidden-reagent",
            "benzene",
            "--max-route-steps",
            "8",
            "--allowed-execution-domain",
            "chemical",
            "--allowed-execution-domain",
            "hybrid",
            "--safety-limit",
            "max_temperature_c=120",
            "--safety-limit",
            "allow_explosive_intermediates=false",
            "--stock-source-id",
            "inventory-v2",
        ]
    )

    assert args.forbidden_reagent == ["benzene"]
    assert args.max_route_steps == 8
    assert args.allowed_execution_domain == ["chemical", "hybrid"]
    assert args.stock_source_id == ["inventory-v2"]
    assert _parse_safety_limits(args.safety_limit) == {
        "max_temperature_c": 120,
        "allow_explosive_intermediates": False,
    }


def test_target_cli_exposes_frozen_benchmark_stock_index() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)

    args = parser.parse_args(
        [
            "solve-target",
            "--target-smiles",
            "CCO",
            "--benchmark-stock-index",
            "D:/bench/stock.sqlite3",
            "--benchmark-stock-index-sha256",
            "a" * 64,
            "--benchmark-stock-name",
            "retrostar-emolecules-23m",
        ]
    )

    assert args.benchmark_stock_index == "D:/bench/stock.sqlite3"
    assert args.benchmark_stock_index_sha256 == "a" * 64
    assert args.benchmark_stock_name == "retrostar-emolecules-23m"


def test_benchmark_stock_defaults_to_the_same_chemenzy_search_boundary(
    tmp_path,
) -> None:
    stock = tmp_path / "benchmark.sqlite3"
    stock.write_bytes(b"fixture")

    names, paths = _resolve_chemenzy_stock_binding(
        stock_names=(),
        stock_paths=(),
        benchmark_stock_index=str(stock),
        benchmark_stock_name="FrozenBenchmarkStock",
        chemenzy_enabled=True,
    )

    assert names == ("FrozenBenchmarkStock",)
    assert paths == (("FrozenBenchmarkStock", str(stock.resolve())),)


def test_benchmark_stock_rejects_a_different_chemenzy_boundary(tmp_path) -> None:
    benchmark = tmp_path / "benchmark.sqlite3"
    provider = tmp_path / "provider.csv"
    benchmark.write_bytes(b"benchmark")
    provider.write_text("smiles\nCCO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark_and_chemenzy_stock_paths_differ"):
        _resolve_chemenzy_stock_binding(
            stock_names=("provider",),
            stock_paths=(("provider", str(provider)),),
            benchmark_stock_index=str(benchmark),
            benchmark_stock_name="benchmark",
            chemenzy_enabled=True,
        )


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
            "--chemenzy-stock-name",
            "RetroStar-stock",
            "--chemenzy-stock-path",
            "RetroStar-stock=D:/bench/origin_dict.csv",
            "--chemenzy-max-routes",
            "3",
            "--chemenzy-iterations",
            "7",
            "--chemenzy-expansion-topk",
            "12",
            "--chemenzy-timeout-s",
            "45",
            "--chemenzy-seed",
            "17",
            "--aizynthfinder-python-executable",
            "D:/aiz/.venv/Scripts/python.exe",
            "--aizynthfinder-config-path",
            "D:/aiz/config/paper.yml",
            "--aizynthfinder-runtime-root",
            "D:/aiz",
        ]
    )
    disabled = parser.parse_args(
        [
            "solve-target",
            "--target-smiles",
            "CCO",
            "--no-target-chemenzy-baseline",
        ]
    )

    assert default.no_chemenzy is False
    assert default.target_chemenzy_baseline is False
    assert disabled.target_chemenzy_baseline is False
    assert default.chemenzy_iterations == 500
    assert default.chemenzy_timeout_s == 1_200.0
    assert default.guided_chemenzy_iterations == 500
    assert default.guided_chemenzy_timeout_s == 1_200.0
    assert configured.chemenzy_env_prefix == "D:/isolated/chemenzy"
    assert configured.chemenzy_stock_name == ["RetroStar-stock"]
    assert configured.chemenzy_stock_path == [
        "RetroStar-stock=D:/bench/origin_dict.csv"
    ]
    assert configured.chemenzy_max_routes == 3
    assert configured.chemenzy_iterations == 7
    assert configured.chemenzy_expansion_topk == 12
    assert configured.chemenzy_timeout_s == 45.0
    assert configured.chemenzy_seed == 17
    assert configured.aizynthfinder_python_executable == (
        "D:/aiz/.venv/Scripts/python.exe"
    )
    assert configured.aizynthfinder_config_path == "D:/aiz/config/paper.yml"
    assert configured.aizynthfinder_runtime_root == "D:/aiz"


def test_validation_fork_supports_parallel_patent_and_literature_sources() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)
    args = parser.parse_args(
        [
            "fork-validation",
            "source-run",
            "--patent-publication",
            "EP2486129B1",
            "--literature-doi",
            "10.1128/AEM.02820-06",
            "--max-patent-sources",
            "1",
            "--max-literature-sources",
            "2",
        ]
    )

    assert args.source_run_id == "source-run"
    assert args.patent_publication == ["EP2486129B1"]
    assert args.literature_doi == ["10.1128/AEM.02820-06"]
    assert args.max_patent_sources == 1
    assert args.max_literature_sources == 2
    assert args.no_auto_patent_evidence is False
    assert args.no_auto_literature_evidence is False


def test_validation_fork_can_reuse_the_frozen_benchmark_stock() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    add_target_commands(commands)
    args = parser.parse_args(
        [
            "fork-validation",
            "source-run",
            "--benchmark-stock-index",
            "D:/bench/stock.sqlite3",
            "--benchmark-stock-index-sha256",
            "a" * 64,
            "--benchmark-stock-name",
            "emolecules-frozen",
        ]
    )

    assert args.benchmark_stock_index == "D:/bench/stock.sqlite3"
    assert args.benchmark_stock_index_sha256 == "a" * 64
    assert args.benchmark_stock_name == "emolecules-frozen"
