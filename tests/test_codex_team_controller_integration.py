from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.harness.agentic_blackboard_controller import run_agentic_blackboard_controller


def _team_artifact(case_id: str, *, target_smiles: str = "CCO", precursor_smiles: str = "CC=O") -> dict:
    return {
        "schema_version": "retrosynthesis_proposal_report_artifact.v1",
        "artifact_id": f"{case_id}:team_report",
        "artifact_type": "RetrosynthesisProposalReport",
        "case_id": case_id,
        "source": "codex_cli",
        "input_refs": ["context_snapshot.json"],
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "direct child-agent draft",
        "payload": {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": case_id,
            "agent_role": "retrosynthesis_team_coordinator",
            "target_smiles": target_smiles,
            "candidates": [
                {
                    "schema_version": "retrosynthesis_candidate.v1",
                    "candidate_id": "ethanol_from_acetaldehyde",
                    "product_smiles": target_smiles,
                    "precursor_smiles": [precursor_smiles],
                    "reaction_family": "carbonyl reduction",
                    "transformation_rationale": "Reduce acetaldehyde after validation.",
                    "source_channel": "codex_strategy",
                    "source_refs": ["child:target_structure_strategist"],
                    "evidence_refs": [],
                    "evidence_level": "model_only",
                    "confidence": "medium",
                    "conditions": [],
                    "catalyst": "",
                    "enzyme": "",
                    "limitations": ["model hypothesis only"],
                    "required_validation": ["forward reconstruction", "stock audit"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                }
            ],
            "evidence_refs": [],
            "limitations": ["No parent proof."],
            "no_solved_claim": True,
        },
    }


def _team_runner(task) -> WorkerRunRecord:
    context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
    target_smiles = str(context["target"]["smiles"])
    precursor_smiles = "CC=O" if target_smiles in {"CCO", "OCC"} else "C"
    artifact = _team_artifact(
        task.case_id,
        target_smiles=target_smiles,
        precursor_smiles=precursor_smiles,
    )
    coordinator_payload = artifact["payload"]
    return WorkerRunRecord(
        run_id=f"{task.task_id}:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        backend="codex_cli",
        output_artifact=artifact,
        output_validation={"accepted": True, "reasons": []},
        metadata={
            "session_id": "controller-integration-thread",
            "event_summary": {"child_agent_spawn_count": len(task.child_roles)},
            "child_agents": [
                {
                    "agent_id": f"child-{index}",
                    "role": role,
                    "role_binding_method": "explicit_spawn_contract",
                    "wait_call_id": f"wait-{index}",
                    "status": "completed",
                    "message": json.dumps(
                        {
                            **coordinator_payload,
                            "agent_role": role,
                            "candidates": list(coordinator_payload["candidates"]) if index == 0 else [],
                        },
                        sort_keys=True,
                    ),
                }
                for index, role in enumerate(task.child_roles)
            ],
        },
    )


def _stop_planner(*, blackboard, round_index: int, run_dir: Path) -> dict:
    del run_dir
    return {
        "schema_version": "agent_action_batch.v1",
        "case_id": blackboard["case_id"],
        "round_index": round_index,
        "mode": "integration_test_stop",
        "actions": [
            {
                "schema_version": "agent_action.v1",
                "action_id": f"r{round_index}:stop",
                "action_type": "stop_unresolved",
                "rationale": "Do not promote the team draft without deterministic proof.",
                "expected_artifact": "unresolved stop marker",
                "success_condition": "run remains unresolved",
                "payload": {"no_solved_claim": True},
            }
        ],
        "semantics": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "deterministic_validator_required": True,
        },
    }


def test_controller_merges_direct_team_consensus_and_renders_it_advisory(tmp_path: Path) -> None:
    result = run_agentic_blackboard_controller(
        target_name="ethanol",
        target_smiles="CCO",
        output_dir=tmp_path,
        max_rounds=1,
        use_codex_agent_team=True,
        codex_agent_team_runner=_team_runner,
        action_planner=_stop_planner,
    )

    board = result["agent_blackboard"]
    assert board["codex_agent_team"]["accepted"] is True
    assert board["codex_agent_team"]["runtime_summary"]["consistent"] is True
    assert board["route_consensus"]["accepted"] is True
    assert board["route_consensus_graph"]["has_hypotheses"] is True
    assert len(board["route_consensus_graph"]["steps"]) == 2
    assert board["codex_agent_team"]["campaign"]["expansion_run_count"] == 2
    assert board["codex_agent_team"]["campaign"]["graph_complete"] is True
    assert board["retrosynthetic_proposals"]
    assert all(row["executable"] is False for row in board["retrosynthetic_proposals"])
    assert result["final_verdict"]["solved"] is False

    forest_path = Path(result["artifacts"]["explored_route_forest"])
    forest = json.loads(forest_path.read_text(encoding="utf-8"))
    assert forest["route_consensus"]["available"] is True
    branches = [row for row in forest["branches"] if row.get("kind") == "route_consensus"]
    assert len(branches) == 1
    assert branches[0]["advisory_only"] is True
    assert branches[0]["solved"] is False
    assert branches[0]["executable"] is False
    graph_branches = [row for row in forest["branches"] if row.get("kind") == "route_consensus_graph"]
    assert len(graph_branches) == 1
    assert len(graph_branches[0]["step_ids"]) == 2
    assert graph_branches[0]["advisory_only"] is True
    assert forest["route_consensus_graph"]["route_count"] == 1
