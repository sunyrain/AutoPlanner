from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir()
    runtime = tmp_path / "runtime"
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(runtime),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index" / "runs.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(tmp_path / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(tmp_path / "vendor"),
        },
    )


def _plan() -> dict:
    return {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:ester",
                "strategic_disconnection": "late acyl substitution",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:ester",
                "route_family_id": "family:ester",
                "steps": [
                    {
                        "step_id": "step:ester",
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)Cl"],
                        "transformation_hypothesis": "acyl substitution",
                    }
                ],
            }
        ],
    }


def test_gateway_runs_every_operator_operation_without_model_calls(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    gateway = CampaignGateway(paths)

    created = gateway.create_run(
        run_id="gateway-example",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
        global_plan=_plan(),
        materialize=True,
    )

    assert created["status"]["graph_revision"] == 2
    assert created["status"]["model_totals"]["model_invocations"] == 0
    assert Path(created["run_dir"]).is_relative_to(paths.runs_root)
    assert paths.artifact_store_root.is_dir()
    assert paths.run_index_path.is_file()
    assert not (paths.runtime_root / "artifacts").exists()

    assert gateway.status("gateway-example")["status"]["graph_revision"] == 2
    assert gateway.validate("gateway-example")["accepted"] is True
    assert gateway.replay("gateway-example")["accepted"] is True
    benchmark = gateway.benchmark("gateway-example", iterations=1)
    assert benchmark["model_invocations"] == 0
    assert benchmark["semantics"]["network_free"] is True

    exported = gateway.export(
        "gateway-example",
        output_dir=tmp_path / "export",
    )
    for path in exported["files"].values():
        assert Path(path).is_file()
    snapshot = json.loads(Path(exported["files"]["snapshot"]).read_text("utf-8"))
    assert snapshot["schema_version"] == "retrosynthesis_route_workbench.v1"

    gc = gateway.gc_plan(minimum_age_s=0)
    assert gc["dry_run"] is True
    assert gc["plan"]["dry_run"] is True
    assert gc["indexed_artifact_pin_count"] > 0
    assert gateway.list_runs()["run_count"] == 1


def test_gateway_applies_a_later_global_plan_through_same_graph(tmp_path: Path) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        run_id="later-plan",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
    )

    applied = gateway.apply_plan(
        "later-plan",
        _plan(),
        materialize=True,
    )

    assert applied["operation"] == "apply-plan"
    assert applied["status"]["graph_revision"] == 2
    assert applied["status"]["accepted_expansion_count"] == 1
    assert applied["status"]["model_totals"]["model_invocations"] == 0
