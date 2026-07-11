from __future__ import annotations

import json
from pathlib import Path

import cascade_planner.harness.agentic_blackboard_controller as controller_module
from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.harness.agentic_blackboard_controller import (
    _controller_codex_search_should_stop,
    _controller_evidence_stop_preserves_campaign,
    _codex_team_has_remaining_campaign_work,
    _run_and_merge_codex_agent_team,
    run_agentic_blackboard_controller,
)
from cascade_planner.harness.tools import ToolExecutionState
from cascade_planner.orchestration.codex_retrosynthesis import RetrosynthesisTeamConfig
from cascade_planner.routes import rebuild_consensus_graph_from_blackboard


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
    return _team_run_record(
        task,
        artifact=artifact,
        session_id="controller-integration-thread",
    )


def _chain_team_runner(task) -> WorkerRunRecord:
    context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
    target_smiles = str(context["target"]["smiles"])
    precursor_smiles = {
        "CCO": "CC=O",
        "OCC": "CC=O",
        "CC=O": "CC(O)O",
        "CC": "C",
    }.get(target_smiles, "C")
    artifact = _team_artifact(
        task.case_id,
        target_smiles=target_smiles,
        precursor_smiles=precursor_smiles,
    )
    return _team_run_record(
        task,
        artifact=artifact,
        session_id="controller-drain-thread",
    )


def _team_run_record(task, *, artifact: dict, session_id: str) -> WorkerRunRecord:
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
            "session_id": session_id,
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
                            "candidates": list(coordinator_payload["candidates"])
                            if index == 0
                            else [],
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


def _continue_planner(*, blackboard, round_index: int, run_dir: Path) -> dict:
    del run_dir
    return {
        "schema_version": "agent_action_batch.v1",
        "case_id": blackboard["case_id"],
        "round_index": round_index,
        "mode": "integration_test_continue",
        "actions": [
            {
                "schema_version": "agent_action.v1",
                "action_id": f"r{round_index}:disconnect",
                "action_type": "generate_disconnection_hypotheses",
                "rationale": "Keep the evidence round open for campaign draining.",
                "expected_artifact": "target_side_disconnection_hypotheses.v1",
                "success_condition": "advisory hypotheses are emitted",
                "payload": {},
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
    assert len(board["route_consensus_graph"]["steps"]) == 1
    assert board["codex_agent_team"]["campaign"]["expansion_run_count"] == 1
    durable_team = json.loads(
        (tmp_path / "codex_retrosynthesis_team" / "team_report.json").read_text(
            encoding="utf-8"
        )
    )
    durable_expansion_ids = {
        row["expansion_id"] for row in durable_team["route_consensus_expansions"]
    }
    projected_expansion_ids = {
        row["expansion_id"]
        for row in board["codex_agent_team"]["route_consensus_expansions"]
    }
    assert durable_expansion_ids <= projected_expansion_ids
    expansion_projection = board["codex_agent_team"][
        "route_consensus_expansion_projection"
    ]
    assert set(expansion_projection["campaign_expansion_ids"]) == (
        durable_expansion_ids
    )
    assert expansion_projection["semantics"][
        "fused_expansions_are_graph_projection_only"
    ] is True
    campaign = board["codex_agent_team"]["campaign"]
    assert campaign["proposal_graph_exhausted"] is False
    assert campaign["resumable"] is True
    assert campaign["graph_complete"] is False
    assert campaign["closure_objective"] == "benchmark_search"
    assert campaign["exploration_mode"] == "exhaustive"
    assert campaign["route_solved"] is False
    assert campaign["campaign_search_complete"] is False
    assert campaign["graph_complete"] is campaign["campaign_search_complete"]
    for field in (
        "all_reaction_edges_closed",
        "all_benchmark_leaves_closed",
        "all_procurement_leaves_closed",
    ):
        assert field in campaign
    assert campaign["frontier_completeness"]["complete"] is False
    assert campaign["semantics"]["queue_exhaustion_is_not_route_completion"] is True
    assert board["retrosynthetic_proposals"]
    assert all(row["executable"] is False for row in board["retrosynthetic_proposals"])
    injected = board["codex_precursor_frontier_injection"]
    assert injected["new_frontier_count"] == 1
    assert injected["semantics"]["raw_reaction_injection"] is False
    codex_frontier = next(
        row
        for row in board["recursive_hypothesis_tasks"]
        if row.get("frontier_origin") == "codex_consensus_precursor"
    )
    assert codex_frontier["precursor_smiles"] == "CC=O"
    assert codex_frontier["parent_edge_requires_independent_l2_validation"] is True
    assert "expand_child_target" in board["current_belief"]["next_action_bias"]
    assert result["final_verdict"]["solved"] is False

    forest_path = Path(result["artifacts"]["explored_route_forest"])
    forest = json.loads(forest_path.read_text(encoding="utf-8"))
    assert forest["route_consensus"]["available"] is True
    # The one-step consensus is already the graph edge, so it is merged into
    # that canonical projection instead of being shown as a duplicate branch.
    branches = [row for row in forest["branches"] if row.get("kind") == "route_consensus"]
    assert branches == []
    graph_branches = [row for row in forest["branches"] if row.get("kind") == "route_consensus_graph"]
    assert len(graph_branches) == 1
    assert len(graph_branches[0]["step_ids"]) == 1
    assert graph_branches[0]["advisory_only"] is True
    assert graph_branches[0]["merged_consensus_ids"]
    assert forest["projection_coverage"]["categories"]["route_consensus"]["merged_into_graph_count"] == 1
    assert forest["semantic_summary"]["agent_tasks"]["completed"] == 4
    assert forest["semantic_summary"]["agent_tasks"]["total"] == 4
    assert forest["counts"]["l0_advisory_branches"] == 1
    assert forest["counts"]["complete_portfolio_routes"] == 0
    assert forest["route_consensus_graph"]["route_count"] == 1
    ledger_path = Path(board["artifact_refs"]["frontier_ledger"])
    assert ledger_path.is_file()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert board["frontier_ledger"] == ledger
    assert (
        board["frontier_ledger_summary"]["frontier_ledger_content_sha256"]
        == ledger["content_sha256"]
    )
    assert "frontier_ledger" in board["artifact_digest_refs"]


def test_controller_refresh_uses_projection_without_rewriting_durable_team_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    durable_bytes_seen: list[bytes] = []
    original_reconcile = controller_module.reconcile_codex_campaign_proof_state

    def observed_reconcile(*args, **kwargs):
        durable_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
        durable_bytes_seen.append(durable_path.read_bytes())
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        controller_module,
        "reconcile_codex_campaign_proof_state",
        observed_reconcile,
    )
    result = run_agentic_blackboard_controller(
        target_name="ethanol",
        target_smiles="CCO",
        output_dir=tmp_path,
        max_rounds=1,
        use_codex_agent_team=True,
        codex_agent_team_runner=_team_runner,
        action_planner=_stop_planner,
    )

    durable_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
    projection_path = (
        tmp_path / "codex_retrosynthesis_team" / "controller_projection.json"
    )
    assert durable_bytes_seen
    assert durable_path.read_bytes() == durable_bytes_seen[-1]
    durable = json.loads(durable_path.read_text(encoding="utf-8"))
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    assert "frontier_ledger_ref" not in durable
    assert projection["schema_version"] == (
        "codex_retrosynthesis_controller_projection.v1"
    )
    assert projection["durable_team_report_sha256"]
    assert projection["team_projection"]["frontier_ledger_ref"]
    assert projection["semantics"][
        "controller_never_rewrites_durable_team_report"
    ] is True
    assert result["agent_blackboard"]["artifact_refs"][
        "codex_retrosynthesis_controller_projection"
    ] == str(projection_path)


def test_resume_failure_does_not_restore_unreplayed_prior_campaign(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = run_agentic_blackboard_controller(
        target_name="ethanol",
        target_smiles="CCO",
        output_dir=tmp_path,
        max_rounds=1,
        use_codex_agent_team=True,
        codex_agent_team_runner=_team_runner,
        action_planner=_stop_planner,
    )
    prior_board = first["agent_blackboard"]
    prior_team = dict(prior_board["codex_agent_team"])
    prior_expansions = list(prior_team.get("route_consensus_expansions") or [])
    assert prior_expansions
    durable_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
    durable_before = durable_path.read_bytes()

    class BrokenRegistry:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("injected controller resume failure")

    monkeypatch.setattr(
        controller_module,
        "build_default_provider_registry",
        lambda **kwargs: BrokenRegistry(),
    )
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"name": "ethanol", "smiles": "CCO"},
        preflight={"case_id": prior_board["case_id"]},
    )
    resumed_board = _run_and_merge_codex_agent_team(
        blackboard=prior_board,
        state=state,
        target_name="ethanol",
        target_smiles="CCO",
        literature_sources=[],
        config=RetrosynthesisTeamConfig(max_depth=2, max_expansions=4),
        runner=_team_runner,
    )

    resumed_team = resumed_board["codex_agent_team"]
    projection = json.loads(
        (
            tmp_path
            / "codex_retrosynthesis_team"
            / "controller_projection.json"
        ).read_text(encoding="utf-8")
    )
    assert durable_path.read_bytes() == durable_before
    assert resumed_team["accepted"] is False
    assert "route_consensus" not in resumed_board
    assert "route_consensus_graph" not in resumed_board
    assert "codex_precursor_frontier_injection" not in resumed_board
    assert "codex_retrosynthesis_team" not in resumed_board["artifact_refs"]
    assert projection["prior_accepted_team_preserved"] is False
    assert "injected controller resume failure" in json.dumps(
        projection["latest_failure"],
        ensure_ascii=False,
    )
    assert resumed_board["agent_team_history"][-1][
        "prior_accepted_team_preserved"
    ] is False
    assert projection["team_projection"]["accepted"] is False
    rebuilt = rebuild_consensus_graph_from_blackboard(resumed_board, max_depth=2)
    assert not {
        row["expansion_id"] for row in rebuilt.get("expansions") or []
    }.intersection({row["expansion_id"] for row in prior_expansions})

    class RejectedEnvelope:
        payload = {
            "schema_version": "codex_retrosynthesis_team_run.v1",
            "accepted": False,
            "case_id": prior_board["case_id"],
            "reasons": ["encoded_resume_runtime_failure"],
        }

        def to_dict(self):
            return {"accepted": False, "payload": self.payload}

    class RejectedRegistry:
        def invoke(self, *args, **kwargs):
            return RejectedEnvelope()

    monkeypatch.setattr(
        controller_module,
        "build_default_provider_registry",
        lambda **kwargs: RejectedRegistry(),
    )
    rejected_board = _run_and_merge_codex_agent_team(
        blackboard=resumed_board,
        state=state,
        target_name="ethanol",
        target_smiles="CCO",
        literature_sources=[],
        config=RetrosynthesisTeamConfig(max_depth=2, max_expansions=4),
        runner=_team_runner,
    )
    assert rejected_board["codex_agent_team"]["accepted"] is False
    assert "route_consensus" not in rejected_board
    assert "route_consensus_graph" not in rejected_board
    assert durable_path.read_bytes() == durable_before


def test_reconciliation_exception_is_projected_without_erasing_accepted_campaign(
    tmp_path: Path,
    monkeypatch,
) -> None:
    durable_bytes_seen: list[bytes] = []

    def failed_reconcile(*args, **kwargs):
        durable_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
        durable_bytes_seen.append(durable_path.read_bytes())
        raise RuntimeError("injected reconciliation failure")

    monkeypatch.setattr(
        controller_module,
        "reconcile_codex_campaign_proof_state",
        failed_reconcile,
    )
    result = run_agentic_blackboard_controller(
        target_name="ethanol",
        target_smiles="CCO",
        output_dir=tmp_path,
        max_rounds=1,
        use_codex_agent_team=True,
        codex_agent_team_runner=_team_runner,
        action_planner=_stop_planner,
    )

    durable_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
    durable = json.loads(durable_path.read_text(encoding="utf-8"))
    board_team = result["agent_blackboard"]["codex_agent_team"]
    projection = json.loads(
        (
            tmp_path
            / "codex_retrosynthesis_team"
            / "controller_projection.json"
        ).read_text(encoding="utf-8")
    )
    failure_events = [
        row
        for row in projection["events"]
        if row.get("stage") == "proof_reconciliation_failed"
    ]
    assert durable_bytes_seen
    assert durable_path.read_bytes() == durable_bytes_seen[-1]
    assert board_team["accepted"] is True
    assert board_team["route_consensus_expansions"] == durable[
        "route_consensus_expansions"
    ]
    assert board_team["campaign"]["accepted_expansion_count"] == durable["campaign"][
        "accepted_expansion_count"
    ]
    assert "proof_reconciliation_failed" not in board_team["campaign"]
    assert failure_events
    assert failure_events[-1]["prior_accepted_team_preserved"] is True
    assert "injected reconciliation failure" in json.dumps(
        failure_events[-1]["failure"],
        ensure_ascii=False,
    )


def test_initial_controller_failure_creates_projection_not_fake_team_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class BrokenRegistry:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("initial provider failure")

    monkeypatch.setattr(
        controller_module,
        "build_default_provider_registry",
        lambda **kwargs: BrokenRegistry(),
    )
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"name": "ethanol", "smiles": "CCO"},
        preflight={"case_id": "initial-controller-failure"},
    )
    board = _run_and_merge_codex_agent_team(
        blackboard={"case_id": "initial-controller-failure"},
        state=state,
        target_name="ethanol",
        target_smiles="CCO",
        literature_sources=[],
        config=RetrosynthesisTeamConfig(max_depth=2, max_expansions=2),
        runner=_team_runner,
    )

    durable_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
    projection_path = (
        tmp_path / "codex_retrosynthesis_team" / "controller_projection.json"
    )
    assert durable_path.exists() is False
    assert projection_path.is_file()
    assert board["codex_agent_team"]["accepted"] is False
    assert "codex_retrosynthesis_team" not in board["artifact_refs"]
    assert board["artifact_refs"][
        "codex_retrosynthesis_controller_projection"
    ] == str(projection_path)


def test_controller_drains_campaign_after_evidence_round_budget(tmp_path: Path) -> None:
    result = run_agentic_blackboard_controller(
        target_name="ethanol",
        target_smiles="CCO",
        output_dir=tmp_path,
        max_rounds=1,
        use_codex_agent_team=True,
        codex_agent_team_max_depth=4,
        codex_agent_team_max_expansions=3,
        codex_agent_team_max_attempt_runs=9,
        codex_agent_team_bootstrap_expansions=1,
        codex_agent_team_max_expansions_per_invocation=1,
        codex_agent_team_max_attempt_runs_per_invocation=2,
        codex_agent_team_runner=_chain_team_runner,
        action_planner=_continue_planner,
    )

    board = result["agent_blackboard"]
    campaign = board["codex_agent_team"]["campaign"]
    drain = json.loads((tmp_path / "codex_campaign_drain.json").read_text(encoding="utf-8"))
    # The second balanced expansion reaches the graph, then the separately
    # bounded drain explores its child. The child's impossible proposal is
    # rejected by admission, consuming attempts but never accepted budget.
    assert campaign["accepted_expansion_count"] == 2
    assert drain["invocation_count"] == 2
    assert drain["stop_reason"] == "no_resumable_campaign_work"
    assert campaign["attempt_run_count"] == 5
    assert all(
        row["accepted_expansion_count"] == 2
        and row["invocation_accepted_expansion_count"] == 0
        for row in drain["invocations"]
    )
    assert campaign["max_attempt_runs"] == 9
    assert drain["semantics"]["max_rounds_is_evidence_budget_not_closure_claim"] is True
    assert result["final_verdict"]["solved"] is False


def test_codex_campaign_resume_requires_open_durable_queue_work() -> None:
    proof_only_board = {
        "codex_agent_team": {
            "accepted": True,
            "campaign": {
                "graph_complete": False,
                "accepted_expansion_count": 1,
                "max_expansions": 10,
                "remaining_frontier": [{"reason": "open_proof:step:1"}],
                "open_reaction_proofs": [{"step_id": "step:1"}],
                "proposal_graph_exhausted": False,
                "frontier_queue": {
                    "jobs": [
                        {
                            "job_id": "done",
                            "state": "succeeded",
                        }
                    ]
                },
            },
        }
    }
    assert _codex_team_has_remaining_campaign_work(proof_only_board) is False

    proposal_board = json.loads(json.dumps(proof_only_board))
    proposal_board["codex_agent_team"]["campaign"]["frontier_queue"]["jobs"].append(
        {"job_id": "ready", "state": "pending"}
    )
    assert _codex_team_has_remaining_campaign_work(proposal_board) is True


def test_controller_does_not_stop_exhaustive_search_at_one_solved_route() -> None:
    board = {
        "codex_agent_team": {
            "accepted": True,
            "campaign": {
                "exploration_mode": "exhaustive",
                "route_solved": True,
                "campaign_search_complete": False,
                # A stale compatibility alias cannot override the explicit
                # objective-aware completion field.
                "graph_complete": True,
            },
        }
    }

    assert _controller_codex_search_should_stop(board) is False
    assert _controller_evidence_stop_preserves_campaign(board) is True
    board["codex_agent_team"]["campaign"]["exploration_mode"] = "first_solved"
    assert _controller_codex_search_should_stop(board) is True
    assert _controller_evidence_stop_preserves_campaign(board) is False
    board["codex_agent_team"]["campaign"].update(
        {
            "exploration_mode": "exhaustive",
            "campaign_search_complete": True,
        }
    )
    assert _controller_codex_search_should_stop(board) is True
    assert _controller_evidence_stop_preserves_campaign(board) is False
