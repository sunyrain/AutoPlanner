from __future__ import annotations

import json

from cascade_planner.harness.parent_route_proof import compile_stitched_parent_route_proof
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from scripts.run_fresh_agentic_smoke import run_fresh_aspirin_smoke


def test_fresh_agentic_smoke_validate_only_accepts_solved_aspirin_run(tmp_path) -> None:
    run_dir = tmp_path / "fresh_aspirin"
    run_dir.mkdir()
    target_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    reactants = ["CC"] * 4 + ["C"] + ["O"] * 4
    terminals = list(dict.fromkeys(reactants))
    verifier = verify_chemenzy_raw_routes(
        {
            "target": target_smiles,
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": terminals,
                        "terminal_stock_status": {item: True for item in terminals},
                    },
                    "steps": [
                        {
                            "product": target_smiles,
                            "reactant_smiles": reactants,
                            "stock_status": {item: True for item in terminals},
                            "reaction_type": "materialized aspirin smoke route",
                        }
                    ],
                }
            ],
        },
        target_smiles=target_smiles,
    )
    proof = compile_stitched_parent_route_proof(
        target_smiles=target_smiles,
        target_name="aspirin",
        case_id="aspirin",
        parent_verifier=verifier,
    )
    assert proof["accepted"], proof["reasons"]
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "aspirin",
                    "target_profile": {
                        "target_name": "aspirin",
                        "target_smiles": target_smiles,
                    },
                    "parent_route_proof": proof,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "action_batch_round_1.json").write_text(
        json.dumps({"actions": [{"action_type": "run_guided_chemenzy"}]}),
        encoding="utf-8",
    )
    (run_dir / "action_batch_round_2.json").write_text(
        json.dumps({"actions": [{"action_type": "stitch_parent_route"}]}),
        encoding="utf-8",
    )
    (run_dir / "final_verdict.json").write_text(
        json.dumps(
            {
                "verdict": "solved",
                "route_status": "solved",
                "solved": True,
                "reasons": [],
            }
        ),
        encoding="utf-8",
    )

    summary = run_fresh_aspirin_smoke(output_dir=run_dir, validate_only=True)

    assert summary["accepted"], summary["validation"]["reasons"]
    assert summary["validate_only"] is True
    assert summary["action_types"] == ["run_guided_chemenzy", "stitch_parent_route"]
    assert summary["final_verdict"]["verdict"] == "solved"
    assert (run_dir / "fresh_agentic_smoke_summary.json").exists()
    assert (run_dir / "route_forest.html").exists()
