from __future__ import annotations

from cascade_planner.harness.hypothesis_execution_report import (
    compile_hypothesis_execution_report,
)


def _hypothesis(smiles: str) -> dict:
    return {
        "payload": {
            "candidate_precursors": [
                {
                    "candidate_id": "candidate-1",
                    "precursor_role": "advanced intermediate",
                    "precursor_smiles": smiles,
                }
            ]
        }
    }


def test_codex_campaign_expansion_counts_as_executed_but_not_proof_closed() -> None:
    smiles = "CC=O"
    artifacts = {
        "codex_retrosynthesis_team": {
            "campaign": {
                "runs": [
                    {
                        "accepted": True,
                        "target_smiles": smiles,
                        "frontier_job_id": "frontier:1",
                        "team_report_ref": "team.json",
                    }
                ],
                "frontier_queue": {
                    "jobs": [
                        {
                            "job_id": "frontier:1",
                            "frontier_smiles": smiles,
                            "state": "succeeded",
                            "closure_kind": "proposal_expansion",
                            "achieved_proof_level": 0,
                            "required_proof_level": 2,
                            "metadata": {},
                        }
                    ]
                },
            }
        }
    }

    report = compile_hypothesis_execution_report(
        blackboard={},
        hypothesis_report=_hypothesis(smiles),
        artifacts=artifacts,
    )

    assert report["executed_candidate_count"] == 1
    assert report["pending_candidate_count"] == 0
    assert report["verified_child_route_count"] == 0
    row = report["candidate_executions"][0]
    assert row["execution_status"] == "executed_advisory_frontier_expansion"
    assert row["agent_task_completed"] is True
    assert row["frontier_proof_closed"] is False
    assert row["verifier_accepted"] is False


def test_codex_campaign_proof_closed_is_distinct_from_agent_success() -> None:
    smiles = "CC=O"
    artifacts = {
        "codex_retrosynthesis_team": {
            "campaign": {
                "runs": [
                    {
                        "accepted": True,
                        "target_smiles": smiles,
                        "frontier_job_id": "frontier:1",
                    }
                ],
                "frontier_queue": {
                    "jobs": [
                        {
                            "job_id": "frontier:1",
                            "frontier_smiles": smiles,
                            "state": "succeeded",
                            "closure_kind": "reaction_validated",
                            "achieved_proof_level": 2,
                            "required_proof_level": 2,
                            "metadata": {},
                        }
                    ]
                },
            }
        }
    }

    report = compile_hypothesis_execution_report(
        blackboard={},
        hypothesis_report=_hypothesis(smiles),
        artifacts=artifacts,
    )

    row = report["candidate_executions"][0]
    assert row["execution_status"] == "executed_verified_child_route"
    assert row["frontier_proof_closed"] is True
    assert row["verifier_accepted"] is True
