import json
from pathlib import Path

import pytest

from scripts.build_route_pool_native_replay_pack import build_route_pool_native_replay_pack


def test_build_route_pool_native_replay_pack_keeps_chemical_steps(tmp_path: Path):
    source = tmp_path / "train_routes.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "target_smiles": "CCO",
                        "target_id": "t1",
                        "route_id": "r1",
                        "native_rank": 2,
                        "native_score": 0.7,
                        "stock_closed": True,
                        "steps": [
                            {
                                "product_smiles": "CCO",
                                "reactants": ["CC=O"],
                                "rxn_smiles": "CC=O>>CCO",
                                "native_step_score": 0.8,
                                "source_model": "ChemEnzyRetroPlanner",
                                "transformation_superclass": "reduction",
                            },
                            {
                                "product_smiles": "CO",
                                "reactants": ["C=O"],
                                "rxn_smiles": "C=O>>CO",
                                "native_step_score": 0.4,
                                "catalyst_classes": ["enzyme"],
                            },
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "native_replay.jsonl"
    report = tmp_path / "report.json"

    summary = build_route_pool_native_replay_pack(
        input_paths=[source],
        output_pack=output,
        report_path=report,
        split="train",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["rows"] == 1
    assert summary["counters"]["steps_skipped_enzymatic"] == 1
    assert rows[0]["leaf"] == "CCO"
    assert rows[0]["candidate_reaction"] == "CC=O>>CCO"
    assert rows[0]["reactants"] == ["CC=O"]
    assert rows[0]["source_policy_group"] == "template"
    assert rows[0]["eval_only"] is False


def test_build_route_pool_native_replay_pack_eval_marks_rows(tmp_path: Path):
    source = tmp_path / "test_routes.jsonl"
    source.write_text(
        json.dumps(
            {
                "target_smiles": "CCO",
                "steps": [{"rxn_smiles": "CC=O>>CCO", "reactants": ["CC=O"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "native_replay.jsonl"
    report = tmp_path / "report.json"

    build_route_pool_native_replay_pack(
        input_paths=[source],
        output_pack=output,
        report_path=report,
        split="eval",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["eval_only"] is True


def test_build_route_pool_native_replay_pack_refuses_eval_name_for_train(tmp_path: Path):
    source = tmp_path / "test_routes.jsonl"
    source.write_text('{"target_smiles":"CCO","steps":[]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="eval-looking"):
        build_route_pool_native_replay_pack(
            input_paths=[source],
            output_pack=tmp_path / "native_replay.jsonl",
            report_path=tmp_path / "report.json",
            split="train",
        )
