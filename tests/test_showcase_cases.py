from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths


_ROOT = Path(__file__).resolve().parents[1]


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(tmp_path / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(tmp_path / "vendor"),
        },
    )


def test_paclitaxel_showcase_is_bounded_diverse_and_explicitly_unresolved(
    tmp_path: Path,
) -> None:
    plan = json.loads(
        (_ROOT / "config/examples/paclitaxel_v4_bounded_showcase_plan.json")
        .read_text(encoding="utf-8")
    )
    gateway = CampaignGateway(_paths(tmp_path))

    result = gateway.create_run(
        run_id="paclitaxel-showcase",
        target_name=plan["target"]["name"],
        target_smiles=plan["target"]["smiles"],
        global_plan=plan,
        materialize=True,
        closeout=True,
        budget=RetrosynthesisRunBudget(
            max_model_invocations=0,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=12,
        ),
    )
    status = result["status"]
    workbench = gateway.workbench("paclitaxel-showcase")["snapshot"]

    assert status["accepted_expansion_count"] == 3
    assert status["attempt_count"] == 3
    assert status["model_totals"]["model_invocations"] == 0
    assert status["portfolio"]["accepted"] is False
    assert status["portfolio"]["metrics"]["selected_route_count"] == 3
    assert status["portfolio"]["metrics"]["distinct_edge_set_count"] == 3
    assert status["portfolio"]["metrics"]["complete_route_count"] == 0
    assert status["portfolio"]["metrics"]["minimum_selected_proof_level"] == 1
    assert {row["kind"] for row in status["frontier"]} >= {
        "evidence",
        "validation",
        "stock",
        "route_closure",
    }
    assert workbench["views"]["expanded"]["count"] == 3
    assert workbench["views"]["reaction_validated"]["count"] == 0
    assert workbench["views"]["stock_closed"]["count"] == 0


def test_unseen_panel_fails_closed_with_named_deficits_and_zero_model_calls(
    tmp_path: Path,
) -> None:
    panel = json.loads(
        (_ROOT / "config/examples/unseen_v4_baseline_panel.json").read_text(
            encoding="utf-8"
        )
    )
    gateway = CampaignGateway(_paths(tmp_path))
    acceptance = RetrosynthesisAcceptanceSpec(**panel["acceptance"])
    budget = RetrosynthesisRunBudget(**panel["budget"])

    for target in panel["targets"]:
        assert Chem.MolFromSmiles(target["smiles"]) is not None
        result = gateway.create_run(
            run_id=f"unseen-{target['name']}",
            target_name=target["name"],
            target_smiles=target["smiles"],
            acceptance=acceptance,
            budget=budget,
            closeout=True,
        )
        status = result["status"]

        assert status["portfolio"]["accepted"] is False
        assert status["portfolio"]["metrics"]["complete_route_count"] == 0
        assert status["model_totals"]["model_invocations"] == 0
        assert status["stop_decision"]["decision"] == target["expected_outcome"]
        assert {row["kind"] for row in status["frontier"]} == {
            target["expected_blocking_deficit"]
        }
