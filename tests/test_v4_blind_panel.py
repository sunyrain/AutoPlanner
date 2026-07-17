from __future__ import annotations

from pathlib import Path

from cascade_planner.application.blind_benchmark_contract import BlindCase
from scripts.run_v4_blind_panel import (
    _ablation_cli_args,
    _acceptance_cli_args,
    _prepare_panel_snapshot,
    _run_id_for_case,
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
