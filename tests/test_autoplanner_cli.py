from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

from cascade_planner.application.biocatalytic_programs import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    with_biocatalysis_program_validation_digest,
)
from cascade_planner.application.experiment_execution_results import (
    build_experiment_execution_result,
)
from cascade_planner.cli import _emit, build_parser, main
from cascade_planner.web.server import serve_web


def _storage_args(tmp_path: Path) -> list[str]:
    return [
        "--repository-root",
        str(tmp_path),
        "--runtime-root",
        str(tmp_path / "runtime"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--artifact-store-root",
        str(tmp_path / "cas"),
        "--run-index-path",
        str(tmp_path / "index.sqlite3"),
    ]


def test_emit_escapes_unicode_for_non_utf8_windows_stream() -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="gbk")

    _emit({"title": "RÉCEPTEURS"}, stream=stream)
    stream.flush()

    assert json.loads(buffer.getvalue().decode("gbk"))["title"] == "RÉCEPTEURS"


def test_cli_run_status_validate_replay_benchmark_export_and_gc(
    tmp_path: Path,
    capsys,
) -> None:
    base = _storage_args(tmp_path)
    assert (
        main(
            [
                *base,
                "run",
                "--run-id",
                "cli-example",
                "--target-name",
                "ethanol",
                "--target-smiles",
                "CCO",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["status"]["model_totals"]["model_invocations"] == 0

    for command in ("status", "validate", "replay"):
        assert main([*base, command, "cli-example"]) == 0
        value = json.loads(capsys.readouterr().out)
        assert value["run_id"] == "cli-example"

    assert main([*base, "programs", "cli-example"]) == 0
    programs = json.loads(capsys.readouterr().out)
    assert programs["oracle"]["accepted"] is True
    assert main([*base, "program-routes", "cli-example"]) == 0
    program_routes = json.loads(capsys.readouterr().out)
    assert program_routes["oracle"]["accepted"] is True
    assert program_routes["overlay"]["counts"]["displayed_routes"] == 0
    assert main([*base, "program-store", "cli-example"]) == 0
    empty_program_store = json.loads(capsys.readouterr().out)
    assert empty_program_store["status"]["event_count"] == 0
    assert main([*base, "program-innovation-store", "cli-example"]) == 0
    empty_innovation_store = json.loads(capsys.readouterr().out)
    assert empty_innovation_store["replay"]["event_count"] == 0
    assert main([*base, "experimental-claim-store", "cli-example"]) == 0
    empty_claim_store = json.loads(capsys.readouterr().out)
    assert empty_claim_store["replay"]["event_count"] == 0
    assert (
        main(
            [
                *base,
                "admit-programs",
                "cli-example",
                "--enable-program-admission",
            ]
        )
        == 0
    )
    admitted = json.loads(capsys.readouterr().out)
    assert admitted["created"] is True
    assert main([*base, "program-store", "cli-example"]) == 0
    durable_program_store = json.loads(capsys.readouterr().out)
    assert durable_program_store["status"]["oracle"]["accepted"] is True
    assert main([*base, "audit-programs", "--run-id", "cli-example", "--limit", "10"]) == 0
    program_audit = json.loads(capsys.readouterr().out)
    assert program_audit["run_count"] == 1
    assert program_audit["accepted_run_count"] == 1
    assert program_audit["semantics"]["program_admission_performed"] is False

    assert main([*base, "benchmark", "cli-example", "--iterations", "1"]) == 0
    benchmark = json.loads(capsys.readouterr().out)
    assert benchmark["model_invocations"] == 0

    destination = tmp_path / "offline"
    assert main([*base, "export", "cli-example", "--output-dir", str(destination)]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert Path(exported["files"]["html"]).is_file()
    assert Path(exported["files"]["review_bundle"]).is_file()
    assert Path(exported["files"]["action_trace"]).is_file()
    assert len(exported["review_bundle_sha256"]) == 64

    assert main([*base, "gc", "--dry-run", "--minimum-age-hours", "0"]) == 0
    gc = json.loads(capsys.readouterr().out)
    assert gc["dry_run"] is True


def test_cli_gc_refuses_an_implicit_destructive_mode(
    tmp_path: Path,
    capsys,
) -> None:
    assert main([*_storage_args(tmp_path), "gc"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["reason"] == "gc_requires_explicit_--dry-run"


def test_cli_exposes_bounded_replay_case_recovery_stages() -> None:
    args = build_parser().parse_args(
        [
            "replay-case",
            "--pack",
            "case.json",
            "--stop-after",
            "evidence",
        ]
    )

    assert args.command == "replay-case"
    assert args.pack == Path("case.json")
    assert args.stop_after == "evidence"


def test_cli_exposes_case_compile_and_one_command_dossier_replay() -> None:
    compile_args = build_parser().parse_args(
        ["compile-case", "--dossier", "case.json", "--output", "pack.json"]
    )
    replay_args = build_parser().parse_args(
        ["replay-dossier", "--dossier", "case.json", "--output-dir", "showcase"]
    )

    assert compile_args.dossier == Path("case.json")
    assert compile_args.output == Path("pack.json")
    assert replay_args.command == "replay-dossier"
    assert replay_args.output_dir == "showcase"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["solve-case", "--dossier", "case.json", "--output-dir", "showcase"]
        )


def test_cli_serves_only_the_isolated_v4_surface() -> None:
    mainline = build_parser().parse_args(["serve"])

    assert mainline.server == "auto"
    assert not hasattr(mainline, "surface")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--surface", "combined"])


def test_auto_web_server_falls_back_to_flask_when_waitress_is_unavailable() -> None:
    app = Mock()
    with (
        patch("cascade_planner.web.v4_app.create_v4_app", return_value=app),
        patch.dict(sys.modules, {"waitress": None}),
    ):
        serve_web(
            host="127.0.0.1",
            port=8899,
            server="auto",
            threads=2,
            debug=False,
        )

    app.run.assert_called_once_with(
        host="127.0.0.1", port=8899, debug=False, threaded=True
    )


def test_cli_program_admission_command_requires_explicit_enable_switch() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["admit-programs", "example"])

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "admit-program-innovation",
                "example",
                "--route-id",
                "route:example",
                "--capabilities-json",
                "capabilities.json",
            ]
        )

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "admit-experimental-claims",
                "example",
                "--route-id",
                "route:example",
                "--capabilities-json",
                "capabilities.json",
            ]
        )


def test_cli_program_innovations_exposes_read_only_pareto_portfolio(
    tmp_path: Path,
    capsys,
    reported_ethanol_program_pack: dict,
) -> None:
    base = _storage_args(tmp_path)
    plan_path = tmp_path / "reduction-plan.json"
    capability_path = tmp_path / "reduction-capabilities.json"
    reported_path = tmp_path / "reported-program-routes.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "global_campaign_plan.v1",
                "route_families": [
                    {
                        "route_family_id": "family:reduction",
                        "strategic_disconnection": "carbonyl reduction",
                    }
                ],
                "multi_step_skeletons": [
                    {
                        "skeleton_id": "skeleton:reduction",
                        "route_family_id": "family:reduction",
                        "steps": [
                            {
                                "step_id": "step:reduction",
                                "product_smiles": "CCO",
                                "precursor_smiles": ["CC=O"],
                                "transformation_hypothesis": "carbonyl reduction",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    capability_path.write_text(
        json.dumps(
            [
                {
                    "capability_id": "fixture:cli-carbonyl-reduction",
                    "enzyme": {"classes": ["alcohol dehydrogenase"]},
                    "match": {
                        "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
                        "element_delta": {"C": 0, "O": 0},
                        "min_scaffold_similarity": 0.05,
                        "max_abs_heavy_atom_delta": 0,
                        "min_substrate_carbons": 2,
                        "min_window_steps": 1,
                        "max_window_steps": 1,
                        "reject_unlisted_motif_changes": True,
                    },
                    "selectivity_objective": "Reduce carbonyl without carbon loss.",
                    "substrate_scope_basis": "generic CLI fixture",
                    "precedent_refs": ["doi:10.1000/cli-reduction-fixture"],
                }
            ]
        ),
        encoding="utf-8",
    )
    reported_path.write_text(
        json.dumps([reported_ethanol_program_pack]),
        encoding="utf-8",
    )

    assert (
        main(
            [
                *base,
                "run",
                "--run-id",
                "cli-program-optimizer",
                "--target-name",
                "ethanol",
                "--target-smiles",
                "CCO",
                "--plan",
                str(plan_path),
                "--materialize",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main([*base, "program-routes", "cli-program-optimizer"]) == 0
    routes = json.loads(capsys.readouterr().out)
    route_id = routes["overlay"]["display_route_ids"][0]

    assert (
        main(
            [
                *base,
                "program-innovations",
                "cli-program-optimizer",
                "--route-id",
                route_id,
                "--capabilities-json",
                str(capability_path),
                "--reported-candidates-json",
                str(reported_path),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["program_optimizer_oracle"]["accepted"] is True
    assert result["program_route_candidates"]["counts"]["candidates"] == 3
    assert result["program_route_candidates"]["counts"]["literature"] == 1
    assert result["program_optimizer"]["profiles"]["exploration"][
        "eligible_candidate_ids"
    ]
    assert result["program_optimizer"]["semantics"][
        "portfolio_cannot_grant_proof_completion_or_production_authority"
    ] is True
    assert result["experimental_work_frontier_oracle"]["accepted"] is True

    proposal = next(iter(result["program_bundle"]["program_proposals"].values()))
    request = next(
        iter(result["experimental_work_frontier"]["work_items"].values())
    )["execution_request"]
    validation = with_biocatalysis_program_validation_digest(
        {
            "schema_version": BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
            "validation_id": "validation:cli-experiment-result",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "accepted": True,
            "evidence_tier": "exact_substrate_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "claim_refs": ["claim:cli-experiment-result"],
            "condition_record_ids": [],
            "selectivity_assessed": True,
            "cofactor_ledger_closed": True,
            "outcome": {"conversion_fraction": 0.88},
        }
    )
    executor_result = build_experiment_execution_result(
        request,
        result_id="experiment-result:cli",
        executor_id="cli-fixture-lab",
        executor_version="1",
        status="success",
        artifact_refs=[
            {"sha256": "d" * 64, "media_type": "application/json", "role": "raw_record"}
        ],
        domain_validation_candidate=validation,
    )
    result_path = tmp_path / "executor-result.json"
    result_path.write_text(json.dumps(executor_result), encoding="utf-8")
    assert (
        main(
            [
                *base,
                "audit-experiment-result",
                "cli-program-optimizer",
                "--route-id",
                route_id,
                "--capabilities-json",
                str(capability_path),
                "--result-json",
                str(result_path),
            ]
        )
        == 0
    )
    audited = json.loads(capsys.readouterr().out)
    assert audited["result_audit"]["accepted_for_domain_gate"] is True
    assert audited["domain_validation_candidate"] == validation

    raw_path = tmp_path / "experiment-raw.json"
    raw_path.write_text(json.dumps({"conversion_fraction": 0.88}), encoding="utf-8")
    assert (
        main(
            [
                *base,
                "stage-experiment-artifact",
                "cli-program-optimizer",
                "--artifact-json",
                str(raw_path),
                "--logical-name",
                "cli-experiment-raw.json",
                "--enable-experiment-artifact-staging",
            ]
        )
        == 0
    )
    staged = json.loads(capsys.readouterr().out)
    policy_path = tmp_path / "experiment-provider-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "experiment_executor_policy.v1",
                "enabled": True,
                "allowed_provider_ids": ["autoplanner.manual_experiment_executor"],
                "preferred_provider_ids": ["autoplanner.manual_experiment_executor"],
                "allowed_domains": ["biocatalytic"],
                "allow_network_access": False,
                "max_estimated_cost_units": 0,
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                *base,
                "dispatch-experiment",
                "cli-program-optimizer",
                "--route-id",
                route_id,
                "--capabilities-json",
                str(capability_path),
                "--request-id",
                request["request_id"],
                "--provider-policy-json",
                str(policy_path),
                "--enable-experiment-dispatch",
            ]
        )
        == 0
    )
    dispatch = json.loads(capsys.readouterr().out)["dispatch"]
    manual_result = build_experiment_execution_result(
        request,
        result_id="experiment-result:cli-manual-dispatch",
        executor_id="autoplanner.manual_experiment_executor",
        executor_version="1.0.0",
        status="success",
        artifact_refs=[
            {
                "sha256": staged["artifact"]["sha256"],
                "media_type": "application/json",
                "role": "raw_record",
            }
        ],
        domain_validation_candidate=validation,
    )
    manual_result_path = tmp_path / "manual-executor-result.json"
    manual_result_path.write_text(json.dumps(manual_result), encoding="utf-8")
    assert (
        main(
            [
                *base,
                "settle-experiment-dispatch",
                "cli-program-optimizer",
                "--route-id",
                route_id,
                "--capabilities-json",
                str(capability_path),
                "--dispatch-id",
                dispatch["dispatch_id"],
                "--result-json",
                str(manual_result_path),
                "--enable-experiment-settlement",
            ]
        )
        == 0
    )
    settled = json.loads(capsys.readouterr().out)["dispatch"]
    assert settled["status"] == "settled"
    assert settled["domain_validation_candidate"] == validation
