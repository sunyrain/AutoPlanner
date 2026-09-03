from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.legacy.harness_runtime.agentic_blackboard import initialize_agent_blackboard
from cascade_planner.legacy.harness_runtime.agentic_blackboard_controller import (
    run_agentic_blackboard_controller,
)
from cascade_planner.legacy.harness_runtime.blackboard_events import (
    BlackboardJournalError,
    append_blackboard_checkpoint,
    begin_blackboard_action,
    blackboard_controller_lock,
    blackboard_event_journal_path,
    commit_prepared_blackboard_action,
    prepare_blackboard_action_result,
    rehydrate_blackboard_from_events,
)
from cascade_planner.legacy.harness_runtime.preflight import run_preflight
from cascade_planner.legacy.harness_runtime.schemas import TargetInput, write_json


def _fresh_board(*, target_name: str = "journal ethanol", max_rounds: int = 2) -> dict:
    target = TargetInput(target_name=target_name, target_smiles="CCO")
    preflight = run_preflight(target)
    return initialize_agent_blackboard(
        target_input=target.to_dict(),
        preflight=preflight,
        max_rounds=max_rounds,
    )


def _team_artifact(case_id: str, *, target_smiles: str) -> dict:
    return {
        "schema_version": "retrosynthesis_proposal_report_artifact.v1",
        "artifact_id": f"{case_id}:journal-team",
        "artifact_type": "RetrosynthesisProposalReport",
        "case_id": case_id,
        "source": "codex_cli",
        "input_refs": ["context_snapshot.json"],
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "journal recovery integration draft",
        "payload": {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": case_id,
            "agent_role": "retrosynthesis_team_coordinator",
            "target_smiles": target_smiles,
            "candidates": [
                {
                    "schema_version": "retrosynthesis_candidate.v1",
                    "candidate_id": "journal-ethanol-from-acetaldehyde",
                    "product_smiles": target_smiles,
                    "precursor_smiles": ["CC=O"],
                    "reaction_family": "carbonyl reduction",
                    "product_retron_type": "carbonyl interconversion",
                    "transformation_rationale": "Recovery-only proposal fixture.",
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
    artifact = _team_artifact(
        task.case_id,
        target_smiles=str(context["target"]["smiles"]),
    )
    coordinator_payload = artifact["payload"]
    return WorkerRunRecord(
        run_id=f"{task.task_id}:journal-run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        backend="codex_cli",
        output_artifact=artifact,
        output_validation={"accepted": True, "reasons": []},
        metadata={
            "session_id": "blackboard-journal-integration",
            "event_summary": {"child_agent_spawn_count": len(task.child_roles)},
            "child_agents": [
                {
                    "agent_id": f"journal-child-{index}",
                    "role": role,
                    "role_binding_method": "explicit_spawn_contract",
                    "wait_call_id": f"journal-wait-{index}",
                    "status": "completed",
                    "message": json.dumps(
                        {
                            **coordinator_payload,
                            "agent_role": role,
                            "candidates": (
                                list(coordinator_payload["candidates"])
                                if index == 0
                                else []
                            ),
                        },
                        sort_keys=True,
                    ),
                }
                for index, role in enumerate(task.child_roles)
            ],
        },
    )


def _search_then_stop_planner(*, blackboard, round_index: int, run_dir: Path) -> dict:
    del run_dir
    return {
        "schema_version": "agent_action_batch.v1",
        "case_id": blackboard["case_id"],
        "round_index": round_index,
        "actions": [
            {
                "schema_version": "agent_action.v1",
                "action_id": f"journal-search:{round_index}",
                "action_type": "search_literature",
                "rationale": "Create durable action and budget history.",
                "expected_artifact": "literature_scout_report.v1",
                "success_condition": "one traceable source candidate",
                "payload": {
                    "schema_version": "agentic_literature_search_payload.v1",
                    "search_intent": "target_proximal_source_discovery",
                    "query": "ethanol synthesis journal recovery",
                    "queries": ["ethanol synthesis journal recovery"],
                    "search_queries": ["ethanol synthesis journal recovery"],
                    "max_sources": 1,
                    "source_acquisition_policy": {
                        "schema_version": "agentic_source_acquisition_policy.v1",
                        "codex_online_first": True,
                        "local_pdf_fallback_allowed": True,
                        "placeholder_allowed_after_failures": True,
                        "auto_local_pdf_requires_agent_discovered_metadata": True,
                        "fallback_order": ["codex_online", "local_pdf", "placeholder"],
                        "no_solved_claim": True,
                    },
                    "no_solved_claim": True,
                },
            },
            {
                "schema_version": "agent_action.v1",
                "action_id": f"journal-stop:{round_index}",
                "action_type": "stop_unresolved",
                "rationale": "Keep the integration case unresolved.",
                "expected_artifact": "unresolved stop marker",
                "success_condition": "no solved claim",
                "payload": {"no_solved_claim": True},
            },
        ],
    }


def _scout_result() -> dict:
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": True,
        "source_candidates": [
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "journal-source",
                "source_ref": "doi:10.1000/journal-recovery",
                "doi": "10.1000/journal-recovery",
                "url": "https://doi.org/10.1000/journal-recovery",
                "title": "Journal recovery source",
                "access_status": "metadata_only",
                "no_solved_claim": True,
            }
        ],
        "source_refs": ["doi:10.1000/journal-recovery"],
        "reasons": [],
        "limitations": [],
        "no_solved_claim": True,
    }


def test_blackboard_event_reducer_rejects_target_mismatch_and_tampering(
    tmp_path: Path,
) -> None:
    board = _fresh_board()
    board["budget_state"]["scout_calls"] = 1
    proof_banks = [
        {
            "schema_version": "blackboard_chemenzy_route_proof_bank.v1",
            "bank_id": "chemenzy-proof-bank:journal-fixture",
            "artifact_ref": "artifacts/guided_chemenzy_result.json",
            "route_proof_bank": {
                "schema_version": "route_proof_bank.v1",
                "target_smiles": "CCO",
                "entry_count": 0,
                "entries": [],
                "content_hash": "journal-fixture",
            },
            "no_solved_claim": True,
            "requires_current_host_replay": True,
        }
    ]
    board["chemenzy_route_proof_banks"] = deepcopy(proof_banks)
    board, _ = append_blackboard_checkpoint(
        tmp_path,
        board,
        stage="agent_action_merged",
    )

    recovered, report = rehydrate_blackboard_from_events(board, run_dir=tmp_path)
    assert recovered["chemenzy_route_proof_banks"] == proof_banks
    assert recovered["chemenzy_route_proof_banks"][0][
        "requires_current_host_replay"
    ] is True
    assert report["final_or_closeout_authority_restored"] is False

    mismatched = _fresh_board(target_name="different target label")
    mismatched["case_id"] = board["case_id"]
    with pytest.raises(BlackboardJournalError, match="target_identity_mismatch"):
        rehydrate_blackboard_from_events(mismatched, run_dir=tmp_path)

    case_mismatched = _fresh_board()
    case_mismatched["case_id"] = "different-case-id"
    with pytest.raises(BlackboardJournalError, match="case_id_mismatch"):
        rehydrate_blackboard_from_events(case_mismatched, run_dir=tmp_path)

    journal_path = blackboard_event_journal_path(tmp_path)
    event = json.loads(journal_path.read_text(encoding="utf-8").splitlines()[0])
    event["checkpoint"]["recoverable_blackboard"]["budget_state"][
        "scout_calls"
    ] = 99
    journal_path.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(BlackboardJournalError, match="checkpoint_digest_mismatch"):
        rehydrate_blackboard_from_events(board, run_dir=tmp_path)


def test_controller_blackboard_rehydrates_after_process_restart(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    first = run_agentic_blackboard_controller(
        target_name="journal recovery ethanol",
        target_smiles="CCO",
        output_dir=run_dir,
        max_rounds=1,
        action_planner=_search_then_stop_planner,
        mock_tool_results={"codex_literature_scout": _scout_result()},
        use_codex_agent_team=True,
        codex_agent_team_max_expansions=1,
        codex_agent_team_max_attempt_runs=2,
        codex_agent_team_bootstrap_expansions=1,
        codex_agent_team_max_expansions_per_invocation=1,
        codex_agent_team_max_attempt_runs_per_invocation=1,
        codex_agent_team_runner=_team_runner,
    )
    first_board = first["agent_blackboard"]
    assert first_board["codex_agent_team"]["accepted"] is True
    assert first_board["recursive_hypothesis_tasks"]
    assert first_board["budget_state"]["scout_calls"] == 1
    assert len(first_board["action_history"]) == 2

    late_board = deepcopy(first_board)
    late_board["literature_evidence"]["exact_rows"] = [
        {
            "schema_version": "literature_exact_step.v1",
            "row_id": "late-exact-row:ethanol",
            "step_id": "late-exact-step:ethanol",
            "source_ref": "doi:10.1000/journal-recovery",
            "product_smiles": "CCO",
            "reactant_smiles": ["CC=O"],
            "reaction_smiles": "CC=O>>CCO",
            "source_detail_exact_step": True,
            "no_solved_claim": True,
        }
    ]
    late_board["route_expansion_subgoals"] = [
        {
            "schema_version": "route_expansion_subgoal_summary.v1",
            "canonical_smiles": "CC=O",
            "accepted": False,
            "status": "pending_reaction_proof",
            "no_solved_claim": True,
        }
    ]
    late_board["final_verdict"] = {"verdict": "solved", "solved": True}
    late_board["parent_route_proof"] = {"accepted": True, "solved": True}
    append_blackboard_checkpoint(
        run_dir,
        late_board,
        stage="late_exact_rows_worker",
        metadata={"late_exact_row_count": 1},
    )

    # The mutable projection is intentionally corrupted. Recovery must ignore
    # it and reduce only the digest-chained journal.
    corrupted_projection = deepcopy(first_board)
    corrupted_projection["literature_evidence"]["exact_rows"] = []
    corrupted_projection["recursive_hypothesis_tasks"] = []
    corrupted_projection["route_expansion_subgoals"] = []
    corrupted_projection["codex_agent_team"] = {"accepted": False}
    corrupted_projection["action_history"] = []
    corrupted_projection["budget_state"]["scout_calls"] = 0
    corrupted_projection["final_verdict"] = {
        "verdict": "solved",
        "solved": True,
    }
    corrupted_projection["parent_route_proof"] = {
        "accepted": True,
        "solved": True,
    }
    write_json(run_dir / "agent_blackboard.json", corrupted_projection)

    second = run_agentic_blackboard_controller(
        target_name="journal recovery ethanol",
        target_smiles="CCO",
        output_dir=run_dir,
        max_rounds=1,
        use_codex_action_planner=False,
        use_codex_agent_team=False,
    )
    second_board = second["agent_blackboard"]

    assert second_board["blackboard_rehydration"]["recovered"] is True
    assert second_board["blackboard_rehydration"]["projection_source_used"] is False
    assert second_board["blackboard_event_journal"]["rehydrated"] is True
    assert [
        row["row_id"]
        for row in second_board["literature_evidence"]["exact_rows"]
    ] == ["late-exact-row:ethanol"]
    assert second_board["recursive_hypothesis_tasks"] == first_board[
        "recursive_hypothesis_tasks"
    ]
    assert second_board["route_expansion_subgoals"] == late_board[
        "route_expansion_subgoals"
    ]
    # Scientific projections are not restored from the blackboard journal.
    # The already-created durable campaign is instead reconstructed from its
    # immutable campaign authority, even when this invocation does not request
    # another Agent expansion.
    assert second_board["codex_agent_team"]["accepted"] is True
    campaign_authority = second_board["codex_campaign_authority_projection"]
    assert campaign_authority["accepted"] is True
    assert campaign_authority["reconciliation_trigger"] == (
        "durable_campaign_recovery"
    )
    assert campaign_authority["proposal_runner_invoked"] is False
    assert campaign_authority["durable_accepted_expansion_count"] == 1
    assert len(second_board["action_history"]) == 2
    assert len(second_board["controller_action_batches"]) == 1
    assert len(second_board["controller_action_batch_validations"]) == 1
    assert second_board["budget_state"]["scout_calls"] == 1
    assert second_board["budget_state"]["rounds_completed"] == 1
    assert second_board["parent_route_proof"] == {}
    assert second["final_verdict"]["solved"] is False
    capability_checks = {
        row["requirement_id"]: row
        for row in second["artifact_bundle"]["artifacts"][
            "agentic_capability_audit"
        ]["payload"]["requirement_checks"]
    }
    assert capability_checks["blackboard_single_state_source"]["accepted"] is True
    assert (
        capability_checks["deterministic_action_batch_validation_gate"]["accepted"]
        is True
    )

    events = [
        json.loads(line)
        for line in blackboard_event_journal_path(run_dir)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stages = [str(row.get("stage") or "") for row in events]
    assert "action_batch_merged" in stages
    assert "agent_action_started" in stages
    assert "agent_action_result_prepared" in stages
    assert "agent_action_committed" in stages
    assert "codex_agent_team_accepted" in stages
    assert "consensus_refresh_checkpoint" in stages
    assert "late_exact_rows_worker" in stages
    assert all(row["sequence"] == index for index, row in enumerate(events, start=1))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rehash_event_chain(events: list[dict]) -> list[dict]:
    previous = ""
    for event in events:
        event["sequence"] = events.index(event) + 1
        event["previous_event_sha256"] = previous
        if event.get("event_type") == "blackboard_checkpoint":
            event["checkpoint_sha256"] = _canonical_sha256(event["checkpoint"])
        event.pop("event_sha256", None)
        event["event_sha256"] = _canonical_sha256(event)
        previous = event["event_sha256"]
    return events


def test_checkpoint_cas_and_tombstone_prevent_stale_field_resurrection(
    tmp_path: Path,
) -> None:
    initial = _fresh_board()
    initial["analogical_hypotheses"] = [{"hypothesis_id": "must-disappear"}]
    first, _ = append_blackboard_checkpoint(
        tmp_path,
        initial,
        stage="first",
    )
    stale = deepcopy(first)
    deleted = deepcopy(first)
    deleted.pop("analogical_hypotheses")
    current, _ = append_blackboard_checkpoint(
        tmp_path,
        deleted,
        stage="delete_field",
    )

    with pytest.raises(BlackboardJournalError, match="stale_head"):
        append_blackboard_checkpoint(
            tmp_path,
            stale,
            stage="stale_writer",
        )

    fresh_with_default = _fresh_board()
    fresh_with_default["analogical_hypotheses"] = [
        {"hypothesis_id": "default-must-not-resurrect"}
    ]
    recovered, report = rehydrate_blackboard_from_events(
        fresh_with_default,
        run_dir=tmp_path,
    )
    assert "analogical_hypotheses" not in recovered
    assert "analogical_hypotheses" in report["tombstoned_fields"]
    assert current["blackboard_event_journal"]["event_count"] == 2


def test_large_checkpoint_uses_immutable_content_addressed_object(
    tmp_path: Path,
) -> None:
    board = _fresh_board()
    board["route_failures"] = [
        {
            "failure_id": "large-checkpoint-fixture",
            "diagnostic": "x" * (256 * 1024),
        }
    ]

    board, event = append_blackboard_checkpoint(
        tmp_path,
        board,
        stage="large_checkpoint",
    )

    assert "checkpoint" not in event
    checkpoint_ref = event["checkpoint_ref"]
    assert checkpoint_ref["storage"] == "immutable_content_addressed_json"
    object_path = (
        blackboard_event_journal_path(tmp_path).parent
        / checkpoint_ref["relative_path"]
    )
    assert object_path.is_file()
    assert object_path.stat().st_size == checkpoint_ref["byte_count"]
    assert blackboard_event_journal_path(tmp_path).stat().st_size < 16 * 1024

    recovered, report = rehydrate_blackboard_from_events(
        _fresh_board(),
        run_dir=tmp_path,
    )
    assert recovered["route_failures"] == board["route_failures"]
    assert report["event_count"] == 1

    object_path.write_bytes(object_path.read_bytes() + b" ")
    with pytest.raises(
        BlackboardJournalError,
        match="checkpoint_object_digest_mismatch",
    ):
        rehydrate_blackboard_from_events(_fresh_board(), run_dir=tmp_path)


def test_rehashed_journal_cannot_restore_scientific_authority(
    tmp_path: Path,
) -> None:
    board = _fresh_board()
    board["current_belief"] = {"open_questions": ["safe operational question"]}
    board, _ = append_blackboard_checkpoint(tmp_path, board, stage="one")
    board, _ = append_blackboard_checkpoint(tmp_path, board, stage="two")
    path = blackboard_event_journal_path(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    projection = events[0]["checkpoint"]["recoverable_blackboard"]
    projection["codex_agent_team"] = {"accepted": True, "solved": True}
    projection["codex_precursor_frontier_injection"] = {"accepted": True}
    projection["route_consensus"] = {"accepted": True}
    projection["route_consensus_graph"] = {
        "accepted": True,
        "graph_complete": True,
    }
    projection["current_belief"] = {
        "parent_route_verifier": {"accepted": True, "solved": True},
        "child_route_solved": True,
        "route_solved": True,
        "closeout_hint": "publish solved route",
        "open_questions": ["safe operational question", "route solved"],
    }
    events = _rehash_event_chain(events)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    recovered, report = rehydrate_blackboard_from_events(
        _fresh_board(),
        run_dir=tmp_path,
    )
    assert "codex_agent_team" not in recovered
    assert "codex_precursor_frontier_injection" not in recovered
    assert "route_consensus" not in recovered
    assert "route_consensus_graph" not in recovered
    belief = recovered["current_belief"]
    assert "parent_route_verifier" not in belief
    assert "child_route_solved" not in belief
    assert "route_solved" not in belief
    assert "closeout_hint" not in belief
    assert belief["open_questions"] == ["safe operational question"]
    assert report["final_or_closeout_authority_restored"] is False


def test_controller_decorator_rejects_concurrent_process(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "locked-run"
    code = "\n".join(
        [
            "from cascade_planner.legacy.harness_runtime.agentic_blackboard_controller import run_agentic_blackboard_controller",
            "from cascade_planner.legacy.harness_runtime.blackboard_events import BlackboardJournalError",
            "from pathlib import Path",
            f"run_dir = Path({str(run_dir)!r})",
            "try:",
            "    run_agentic_blackboard_controller(target_name='locked', target_smiles='CCO', output_dir=run_dir, max_rounds=0, use_codex_action_planner=False)",
            "except BlackboardJournalError as exc:",
            "    print(str(exc))",
            "else:",
            "    print('unexpected-controller-entry')",
        ]
    )
    with blackboard_controller_lock(run_dir, timeout_seconds=1.0):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
            env={
                **os.environ,
                "AUTOPLANNER_CONTROLLER_LOCK_TIMEOUT_SECONDS": "0.2",
            },
        )
    assert completed.returncode == 0
    assert "blackboard_controller_lock_timeout" in completed.stdout
    assert "unexpected-controller-entry" not in completed.stdout
    assert not (run_dir / "target_input.json").exists()


def test_prepared_action_result_replays_without_second_tool_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cascade_planner.legacy.harness_runtime.agentic_blackboard_controller as controller

    run_dir = tmp_path / "prepared-replay"
    execute_calls: list[str] = []
    original_execute = controller._execute_agent_action
    original_commit = controller.commit_prepared_blackboard_action

    def counted_execute(*args, **kwargs):
        action = kwargs["action"]
        execute_calls.append(str(action.get("action_type") or ""))
        return original_execute(*args, **kwargs)

    def crash_before_commit(*args, **kwargs):
        raise RuntimeError("injected crash after prepared result")

    monkeypatch.setattr(controller, "_execute_agent_action", counted_execute)
    monkeypatch.setattr(
        controller,
        "commit_prepared_blackboard_action",
        crash_before_commit,
    )
    with pytest.raises(RuntimeError, match="after prepared result"):
        run_agentic_blackboard_controller(
            target_name="prepared replay ethanol",
            target_smiles="CCO",
            output_dir=run_dir,
            max_rounds=1,
            action_planner=_search_then_stop_planner,
            mock_tool_results={"codex_literature_scout": _scout_result()},
            use_codex_action_planner=False,
        )
    assert execute_calls == ["search_literature"]

    monkeypatch.setattr(
        controller,
        "commit_prepared_blackboard_action",
        original_commit,
    )
    resumed = run_agentic_blackboard_controller(
        target_name="prepared replay ethanol",
        target_smiles="CCO",
        output_dir=run_dir,
        max_rounds=1,
        action_planner=_search_then_stop_planner,
        mock_tool_results={"codex_literature_scout": _scout_result()},
        use_codex_action_planner=False,
    )
    assert execute_calls.count("search_literature") == 1
    assert execute_calls.count("stop_unresolved") == 1
    assert resumed["agent_blackboard"]["budget_state"]["scout_calls"] == 1
    assert len(resumed["agent_blackboard"]["action_history"]) == 2
    assert resumed["agent_blackboard"]["literature_evidence"][
        "source_candidates"
    ]
    events = [
        json.loads(line)
        for line in blackboard_event_journal_path(run_dir)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    assert "action_result_prepared" in event_types
    assert "action_committed" in event_types


def test_started_without_prepared_is_charged_and_never_auto_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cascade_planner.legacy.harness_runtime.agentic_blackboard_controller as controller

    run_dir = tmp_path / "indeterminate-start"
    execute_calls: list[str] = []
    original_execute = controller._execute_agent_action
    original_prepare = controller.prepare_blackboard_action_result

    def counted_execute(*args, **kwargs):
        execute_calls.append(str(kwargs["action"].get("action_type") or ""))
        return original_execute(*args, **kwargs)

    def crash_before_prepare(*args, **kwargs):
        raise RuntimeError("injected crash before prepared result")

    monkeypatch.setattr(controller, "_execute_agent_action", counted_execute)
    monkeypatch.setattr(
        controller,
        "prepare_blackboard_action_result",
        crash_before_prepare,
    )
    with pytest.raises(RuntimeError, match="before prepared result"):
        run_agentic_blackboard_controller(
            target_name="indeterminate ethanol",
            target_smiles="CCO",
            output_dir=run_dir,
            max_rounds=1,
            action_planner=_search_then_stop_planner,
            mock_tool_results={"codex_literature_scout": _scout_result()},
            use_codex_action_planner=False,
        )
    assert execute_calls == ["search_literature"]

    monkeypatch.setattr(
        controller,
        "prepare_blackboard_action_result",
        original_prepare,
    )
    resumed = run_agentic_blackboard_controller(
        target_name="indeterminate ethanol",
        target_smiles="CCO",
        output_dir=run_dir,
        max_rounds=1,
        action_planner=_search_then_stop_planner,
        mock_tool_results={"codex_literature_scout": _scout_result()},
        use_codex_action_planner=False,
    )
    board = resumed["agent_blackboard"]
    assert execute_calls == ["search_literature"]
    assert board["budget_state"]["scout_calls"] == 1
    assert any(
        "blackboard_action_recovery_blocked" in str(flag)
        for flag in board["safety_flags"]
    )
    failure = next(
        row
        for row in board["route_failures"]
        if row.get("schema_version") == "agent_action_recovery_failure.v1"
    )
    assert failure["automatic_retry_allowed"] is False
    assert failure["charged_attempt_count"] == 1


def test_explicit_idempotent_recovery_retry_reuses_original_reservation(
    tmp_path: Path,
) -> None:
    board = _fresh_board(max_rounds=1)
    action = {
        "schema_version": "agent_action.v1",
        "action_id": "r1:extract_visual_literature_chain",
        "action_type": "extract_visual_literature_chain",
        "rationale": "retry a local evidence extraction",
        "expected_artifact": "visual_literature_chain_extraction_result.v1",
        "success_condition": "source-bound visual result is recorded",
        "payload": {"source_ref": "doi:10.example/retry"},
    }
    reserved = deepcopy(board["budget_state"])
    reserved["visual_calls"] = 1
    _, first = begin_blackboard_action(
        tmp_path,
        board,
        action=action,
        round_index=1,
        reserved_budget_state=reserved,
    )
    recovered, _ = rehydrate_blackboard_from_events(
        board,
        run_dir=tmp_path,
    )
    incorrectly_double_charged = deepcopy(recovered["budget_state"])
    incorrectly_double_charged["visual_calls"] = 2

    retried_board, retry = begin_blackboard_action(
        tmp_path,
        recovered,
        action=action,
        round_index=1,
        reserved_budget_state=incorrectly_double_charged,
        allow_idempotent_retry=True,
    )

    execution = retry["started_event"]["action_execution"]
    assert retry["status"] == "started"
    assert retry["attempt_index"] == 2
    assert retry["charged_retry"] is False
    assert retry["idempotent_recovery_retry"] is True
    assert execution["retry_of_event_id"] == first["started_event"]["event_id"]
    assert execution["retry_reason"] == "idempotent_local_action_recovery"
    assert execution["budget_pre_state"]["visual_calls"] == 1
    assert execution["budget_after_reservation"]["visual_calls"] == 1
    assert retried_board["budget_state"]["visual_calls"] == 1


def test_raw_journal_binding_survives_host_capability_payload_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cascade_planner.legacy.harness_runtime.agentic_blackboard_controller as controller

    run_dir = tmp_path / "raw-binding-effective-execution"
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\nsource\n")
    page_image = tmp_path / "page.png"
    page_image.write_bytes(b"materialized-page")
    execute_payloads: list[dict] = []
    original_execute = controller._execute_agent_action
    original_prepare = controller.prepare_blackboard_action_result

    def planner(**kwargs):
        return {
            "schema_version": "agent_action_batch.v1",
            "case_id": "normalized journal binding",
            "round_index": kwargs["round_index"],
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "pdf:normalized",
                    "action_type": "extract_pdf_literature_structures",
                    "rationale": "render the only current PDF",
                    "expected_artifact": "literature_pdf_structure_evidence.v1",
                    "success_condition": "rendered evidence is recorded",
                    "payload": {},
                }
            ],
        }

    def counted_execute(*args, **kwargs):
        execute_payloads.append(dict(kwargs["action"].get("payload") or {}))
        return original_execute(*args, **kwargs)

    def crash_before_prepare(*args, **kwargs):
        raise RuntimeError("injected crash after normalized execution")

    mock_result = {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "accepted": True,
        "source_ref": "doi:10.1000/journal-binding",
        "source_pdf_path": str(source_pdf),
        "rendered_page_count": 1,
        "rendered_pages": [{"page_number": 1, "image_path": str(page_image)}],
        "reasons": [],
    }
    monkeypatch.setattr(controller, "_execute_agent_action", counted_execute)
    monkeypatch.setattr(
        controller,
        "prepare_blackboard_action_result",
        crash_before_prepare,
    )
    with pytest.raises(RuntimeError, match="normalized execution"):
        run_agentic_blackboard_controller(
            target_name="normalized journal binding",
            target_smiles="CCO",
            output_dir=run_dir,
            literature_pdf_path=source_pdf,
            literature_pdf_source_ref="doi:10.1000/journal-binding",
            auto_discover_local_pdfs=False,
            max_rounds=1,
            action_planner=planner,
            mock_tool_results={"extract_pdf_literature_structures": mock_result},
            use_codex_action_planner=False,
        )
    assert len(execute_payloads) == 1
    assert execute_payloads[0]["source_capability_id"].startswith(
        "source-capability:sha256:"
    )

    monkeypatch.setattr(
        controller,
        "prepare_blackboard_action_result",
        original_prepare,
    )
    resumed = run_agentic_blackboard_controller(
        target_name="normalized journal binding",
        target_smiles="CCO",
        output_dir=run_dir,
        literature_pdf_path=source_pdf,
        literature_pdf_source_ref="doi:10.1000/journal-binding",
        auto_discover_local_pdfs=False,
        max_rounds=1,
        action_planner=planner,
        mock_tool_results={"extract_pdf_literature_structures": mock_result},
        use_codex_action_planner=False,
    )

    assert len(execute_payloads) == 2
    assert execute_payloads[1]["source_capability_id"] == (
        execute_payloads[0]["source_capability_id"]
    )
    events = [
        json.loads(line)
        for line in blackboard_event_journal_path(run_dir)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    retries = [
        event
        for event in events
        if event.get("stage") == "agent_action_idempotent_retry_started"
    ]
    assert len(retries) == 1
    assert retries[0]["action_execution"]["retry_reason"] == (
        "idempotent_local_action_recovery"
    )
    assert not any(
        "blackboard_action_recovery_blocked" in str(flag)
        for flag in resumed["agent_blackboard"].get("safety_flags") or []
    )


def _torn_tail_sidecars(journal_path: Path) -> list[Path]:
    return sorted(
        journal_path.parent.glob(f"{journal_path.stem}.torn-tail.*.json")
    )


def test_unterminated_invalid_tail_is_preserved_then_truncated(
    tmp_path: Path,
) -> None:
    initial = _fresh_board()
    board, event = append_blackboard_checkpoint(
        tmp_path,
        initial,
        stage="durable_prefix",
    )
    journal_path = blackboard_event_journal_path(tmp_path)
    valid_prefix = journal_path.read_bytes()
    fragment = b'{"schema_version":"agent_blackboard_event.v1","sequence":'
    journal_path.write_bytes(valid_prefix + fragment)

    recovered, report = rehydrate_blackboard_from_events(
        initial,
        run_dir=tmp_path,
    )

    assert report["event_count"] == 1
    assert report["last_event_sha256"] == event["event_sha256"]
    assert recovered["blackboard_event_journal"]["event_count"] == 1
    assert journal_path.read_bytes() == valid_prefix
    sidecars = _torn_tail_sidecars(journal_path)
    assert len(sidecars) == 1
    evidence = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "agent_blackboard_torn_tail_quarantine.v1"
    )
    assert base64.b64decode(evidence["fragment_base64"]) == fragment
    assert evidence["fragment_sha256"] == hashlib.sha256(fragment).hexdigest()
    assert evidence["valid_prefix_length"] == len(valid_prefix)
    assert evidence["scientific_authority"] is False
    assert evidence["action_authority"] is False
    assert evidence["replay_eligible"] is False
    assert evidence["quarantined_under_exclusive_journal_lock"] is True


def test_valid_unterminated_event_is_retained_and_next_append_separates_it(
    tmp_path: Path,
) -> None:
    initial = _fresh_board()
    board, first_event = append_blackboard_checkpoint(
        tmp_path,
        initial,
        stage="first",
    )
    journal_path = blackboard_event_journal_path(tmp_path)
    without_newline = journal_path.read_bytes().removesuffix(b"\n")
    journal_path.write_bytes(without_newline)

    recovered, report = rehydrate_blackboard_from_events(
        initial,
        run_dir=tmp_path,
    )

    assert report["last_event_sha256"] == first_event["event_sha256"]
    assert journal_path.read_bytes() == without_newline
    assert _torn_tail_sidecars(journal_path) == []

    recovered, second_event = append_blackboard_checkpoint(
        tmp_path,
        recovered,
        stage="second",
    )
    records = journal_path.read_bytes().splitlines()
    assert journal_path.read_bytes().endswith(b"\n")
    assert len(records) == 2
    assert [json.loads(record)["sequence"] for record in records] == [1, 2]
    assert second_event["sequence"] == 2
    assert recovered["blackboard_event_journal"]["event_count"] == 2


def test_newline_terminated_bad_json_fails_closed_without_truncation(
    tmp_path: Path,
) -> None:
    initial = _fresh_board()
    append_blackboard_checkpoint(tmp_path, initial, stage="valid")
    journal_path = blackboard_event_journal_path(tmp_path)
    corrupted = journal_path.read_bytes() + b'{"schema_version":\n'
    journal_path.write_bytes(corrupted)

    with pytest.raises(BlackboardJournalError, match="json_invalid:2"):
        rehydrate_blackboard_from_events(initial, run_dir=tmp_path)

    assert journal_path.read_bytes() == corrupted
    assert _torn_tail_sidecars(journal_path) == []


@pytest.mark.parametrize(
    ("fragment", "reason"),
    [
        (
            b'{"schema_version":"first","schema_version":"second"}',
            "json_duplicate_key:2",
        ),
        (b'{"schema_version":NaN}', "json_non_finite_number:2"),
        (b'{"schema_version":Infinity}', "json_non_finite_number:2"),
        (b'{"schema_version":1e999}', "json_non_finite_number:2"),
        (
            b'{"overflow":1e999,"schema_version":',
            "json_non_finite_number:2",
        ),
        (
            b'{"nested":{"key":1,"key":2},"schema_version":',
            "json_duplicate_key:2",
        ),
    ],
)
def test_ambiguous_or_non_finite_unterminated_json_fails_closed(
    tmp_path: Path,
    fragment: bytes,
    reason: str,
) -> None:
    initial = _fresh_board()
    append_blackboard_checkpoint(tmp_path, initial, stage="valid")
    journal_path = blackboard_event_journal_path(tmp_path)
    corrupted = journal_path.read_bytes() + fragment
    journal_path.write_bytes(corrupted)

    with pytest.raises(BlackboardJournalError, match=reason):
        rehydrate_blackboard_from_events(initial, run_dir=tmp_path)

    assert journal_path.read_bytes() == corrupted
    assert _torn_tail_sidecars(journal_path) == []


@pytest.mark.parametrize(
    ("corruption", "reason"),
    [
        ("chain", "previous_digest_mismatch"),
        ("identity", "case_id_mismatch"),
        ("digest", "event_digest_mismatch"),
    ],
)
def test_unterminated_chain_identity_and_digest_errors_fail_closed(
    tmp_path: Path,
    corruption: str,
    reason: str,
) -> None:
    initial = _fresh_board()
    board, _ = append_blackboard_checkpoint(tmp_path, initial, stage="first")
    append_blackboard_checkpoint(tmp_path, board, stage="second")
    journal_path = blackboard_event_journal_path(tmp_path)
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    final_event = events[-1]
    if corruption == "chain":
        final_event["previous_event_sha256"] = "0" * 64
    elif corruption == "identity":
        final_event["case_id"] = "different-case"
    else:
        final_event["event_sha256"] = "0" * 64
    if corruption != "digest":
        final_event.pop("event_sha256")
        final_event["event_sha256"] = _canonical_sha256(final_event)
    corrupted = (
        json.dumps(events[0], sort_keys=True).encode("utf-8")
        + b"\n"
        + json.dumps(final_event, sort_keys=True).encode("utf-8")
    )
    journal_path.write_bytes(corrupted)

    with pytest.raises(BlackboardJournalError, match=reason):
        rehydrate_blackboard_from_events(initial, run_dir=tmp_path)

    assert journal_path.read_bytes() == corrupted
    assert _torn_tail_sidecars(journal_path) == []


@pytest.mark.parametrize(
    ("tail_stage", "expected_status", "expect_quarantine"),
    [
        ("started_complete_without_newline", "indeterminate", False),
        ("prepared_torn", "indeterminate", True),
        ("committed_torn", "prepared", True),
    ],
)
def test_torn_action_lifecycle_never_allocates_a_second_tool_attempt(
    tmp_path: Path,
    tail_stage: str,
    expected_status: str,
    expect_quarantine: bool,
) -> None:
    run_dir = tmp_path / tail_stage
    initial = _fresh_board()
    action = _search_then_stop_planner(
        blackboard=initial,
        round_index=1,
        run_dir=run_dir,
    )["actions"][0]
    reserved_budget = deepcopy(initial["budget_state"])
    reserved_budget["scout_calls"] = 1
    board, started = begin_blackboard_action(
        run_dir,
        initial,
        action=action,
        round_index=1,
        reserved_budget_state=reserved_budget,
    )
    started_event = started["started_event"]
    journal_path = blackboard_event_journal_path(run_dir)

    if tail_stage == "started_complete_without_newline":
        journal_path.write_bytes(journal_path.read_bytes().removesuffix(b"\n"))
    else:
        board, prepared_event = prepare_blackboard_action_result(
            run_dir,
            board,
            action=action,
            round_index=1,
            started_event=started_event,
            action_result=_scout_result(),
            tool_records=[],
            artifact_updates={},
        )
        if tail_stage == "committed_torn":
            board, _ = commit_prepared_blackboard_action(
                run_dir,
                board,
                action=action,
                round_index=1,
                prepared_event=prepared_event,
            )
        journal = journal_path.read_bytes()
        final_start = journal.rfind(b"\n", 0, len(journal) - 1) + 1
        final_record = journal[final_start:-1]
        journal_path.write_bytes(
            journal[:final_start] + final_record[: len(final_record) // 2]
        )

    recovered, _ = rehydrate_blackboard_from_events(
        initial,
        run_dir=run_dir,
    )
    before_resume = journal_path.read_bytes()
    _, lifecycle = begin_blackboard_action(
        run_dir,
        recovered,
        action=action,
        round_index=1,
        reserved_budget_state=reserved_budget,
    )

    assert lifecycle["status"] == expected_status
    assert journal_path.read_bytes() == before_resume
    assert sum(
        json.loads(line)["event_type"] == "action_started"
        for line in journal_path.read_bytes().splitlines()
    ) == 1
    assert bool(_torn_tail_sidecars(journal_path)) is expect_quarantine
    if expected_status == "indeterminate":
        assert lifecycle["automatic_retry_allowed"] is False
