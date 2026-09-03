from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade_planner.legacy.harness_runtime import codex_action_planner
from cascade_planner.agent.codex_worker import (
    WorkerProcessResult,
    run_codex_worker,
)
from cascade_planner.legacy.harness_runtime.agent_action_planner import validate_action_batch
from cascade_planner.legacy.harness_runtime.codex_action_planner import (
    _bounded_planner_prompt_payload,
    _codex_action_planner_repair_task,
    _codex_action_planner_task,
    _normalize_codex_batch,
    _planner_json_bytes,
    _planner_json_sha256,
    _write_codex_blackboard_snapshot,
)


def _planner_board(tmp_path: Path) -> dict:
    pdf_path = tmp_path / "source-bound-process.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nplanner prompt fixture\n")
    ledger_digest = "d" * 64
    return {
        "case_id": "embedded-planner-snapshot",
        "target_profile": {
            "target_name": "fixture target",
            "target_smiles": "CCO",
            "canonical_smiles": "CCO",
            "valid": True,
        },
        "current_belief": {
            "next_action_bias": ["continue_source_bound_evidence"],
            "blocked_directions": [],
            "constraints": {},
        },
        "literature_evidence": {
            "source_candidates": [
                {
                    "schema_version": "literature_source_candidate.v1",
                    "candidate_id": "source:process",
                    "source_ref": "doi:10.1000/embedded.snapshot",
                    "doi": "10.1000/embedded.snapshot",
                    "title": "Source-bound process route",
                    "local_pdf": str(pdf_path),
                    "expected_scheme_or_compound_labels": ["compound 7"],
                    "access_status": "local_pdf_available",
                    "no_solved_claim": True,
                }
            ],
            "source_lifecycle": [
                {
                    "source_ref": "doi:10.1000/embedded.snapshot",
                    "stage": "local_pdf_available",
                    "local_pdf": str(pdf_path),
                }
            ],
            "pdf_structure_evidence": [],
            "visual_chains": [],
            "exact_rows": [],
            "process_evidence_rows": [],
            "structure_resolution_tasks": [
                {
                    "schema_version": "agent_structure_resolution_task.v1",
                    "task_id": "resolve:compound-7",
                    "label": "compound 7",
                    "source_ref": "doi:10.1000/embedded.snapshot",
                    "status": "pending",
                    "no_solved_claim": True,
                }
            ],
        },
        "frontier_ledger": {
            "schema_version": "frontier_ledger.v1",
            "content_sha256": ledger_digest,
            "root": {"canonical_smiles": "CCO", "closure": {}},
            "summary": {
                "proposal_pending_molecule_count": 1,
                "proposal_expansion_eligible_molecule_count": 1,
                "stock_pending_leaf_count": 1,
            },
            "molecules": {
                "CCO": {
                    "proposal": {"state": "frontier"},
                    "work": {
                        "open": True,
                        "states": ["queued"],
                        "proposal_expansion_allowed": True,
                    },
                    "stock": {"closed": False},
                }
            },
        },
        "frontier_ledger_summary": {
            "schema_version": "frontier_ledger_summary.v1",
            "frontier_ledger_content_sha256": ledger_digest,
            "input_valid": True,
            "ledger_validation_accepted": True,
        },
        "retrosynthetic_proposals": [],
        "recursive_hypothesis_tasks": [],
        "bridge_tasks": [],
        "route_failures": [],
        "proposal_failure_feedback": [],
        "plugin_runtime_diagnostics": [],
        "action_history": [],
        "budget_state": {
            "scout_calls": 0,
            "max_scout_calls": 4,
            "visual_calls": 0,
            "max_visual_calls": 4,
            "chemenzy_runs": 0,
            "max_chemenzy_runs": 2,
            "child_target_runs": 0,
            "max_child_target_runs": 2,
        },
    }


def _valid_action_artifact(task, *, pdf_path: str) -> dict:
    return {
        "schema_version": "agent_action_batch_artifact.v1",
        "artifact_id": f"{task.task_id}:AgentActionBatch",
        "artifact_type": "AgentActionBatch",
        "case_id": task.case_id,
        "source": "codex_test_runner",
        "input_refs": list(task.input_refs),
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "source-bound action without filesystem reads",
        "payload": {
            "schema_version": "agent_action_batch.v1",
            "case_id": task.case_id,
            "round_index": 2,
            "mode": "codex_blackboard_planner",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "r2:extract-pdf",
                    "action_type": "extract_pdf_literature_structures",
                    "rationale": "Materialize the pending source-bound PDF.",
                    "expected_artifact": "PDF structure evidence",
                    "success_condition": "Source pages become available.",
                    "payload": {
                        "source_ref": "doi:10.1000/embedded.snapshot",
                        "pdf_path": pdf_path,
                        "no_solved_claim": True,
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        },
    }


def test_prompt_embeds_digest_bound_pending_decision_state_without_shell(
    tmp_path: Path,
) -> None:
    board = _planner_board(tmp_path)
    snapshot_path = _write_codex_blackboard_snapshot(
        board,
        run_dir=tmp_path,
        round_index=2,
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    task = _codex_action_planner_task(
        blackboard=board,
        round_index=2,
        run_dir=tmp_path,
        snapshot_path=snapshot_path,
    )

    content_payload = dict(snapshot)
    supplied_content_digest = content_payload.pop("content_sha256")
    assert supplied_content_digest == _planner_json_sha256(content_payload)
    assert snapshot["prompt_payload_sha256"] == _planner_json_sha256(
        snapshot["prompt_payload"]
    )
    assert snapshot["prompt_payload_bounds"]["within_bound"] is True
    assert snapshot["prompt_payload_bounds"]["embedded_bytes"] <= snapshot[
        "prompt_payload_bounds"
    ]["max_bytes"]
    assert task.input_refs == [str(snapshot_path)]
    assert task.allowed_tools == ["web_search", "browser", "literature_search"]
    assert "codex_action_planner_embedded_snapshot.v1" in task.objective
    assert snapshot["prompt_payload_sha256"] in task.objective
    assert "doi:10.1000/embedded.snapshot" in task.objective
    assert "source-bound-process.pdf" in task.objective
    assert "resolve:compound-7" in task.objective
    assert "d" * 64 in task.objective
    assert "Do not call shell" in task.objective
    assert "Input refs are audit locators only" in task.objective
    assert '"shell_allowed": false' in task.objective
    assert "agent_blackboard.json" not in task.objective


def test_valid_worker_batch_needs_no_shell_and_shell_remains_rejected(
    tmp_path: Path,
) -> None:
    board = _planner_board(tmp_path)
    snapshot_path = _write_codex_blackboard_snapshot(
        board,
        run_dir=tmp_path,
        round_index=2,
    )
    task = _codex_action_planner_task(
        blackboard=board,
        round_index=2,
        run_dir=tmp_path,
        snapshot_path=snapshot_path,
    )
    artifact = _valid_action_artifact(
        task,
        pdf_path=str(tmp_path / "source-bound-process.pdf"),
    )
    normalized = _normalize_codex_batch(
        artifact["payload"],
        blackboard=board,
        round_index=2,
    )
    assert validate_action_batch(normalized, blackboard=board)["accepted"] is True

    without_shell = run_codex_worker(
        task,
        runner=lambda _: WorkerProcessResult(
            stdout=json.dumps(artifact),
            exit_code=0,
            tool_calls=[],
        ),
    )
    with_shell = run_codex_worker(
        task,
        runner=lambda _: WorkerProcessResult(
            stdout=json.dumps(artifact),
            exit_code=0,
            tool_calls=[{"tool": "shell"}],
        ),
    )

    assert without_shell.status == "accepted_draft"
    assert without_shell.output_validation["accepted"] is True
    assert with_shell.status == "rejected_output"
    assert with_shell.output_validation["accepted"] is False
    assert "tool_not_allowed" in with_shell.output_validation["reasons"]


def test_oversized_handoff_is_bounded_without_losing_core_pending_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = _planner_board(tmp_path)
    snapshot_path = _write_codex_blackboard_snapshot(
        board,
        run_dir=tmp_path,
        round_index=2,
    )
    handoff = json.loads(snapshot_path.read_text(encoding="utf-8"))["blackboard"]
    handoff["evidence_board"]["source_candidates"].extend(
        {
            "source_ref": f"doi:10.1000/oversized.{index}",
            "title": "oversized source " + ("x" * 2_000),
            "local_pdf": str(tmp_path / f"oversized-{index}.pdf"),
        }
        for index in range(100)
    )
    handoff["route_board"]["route_anchor_opportunities"] = {
        "opportunities": [
            {"anchor_id": f"anchor:{index}", "detail": "y" * 2_000}
            for index in range(100)
        ]
    }
    handoff["route_board"]["frontier_ledger"].update(
        {
            f"hostile-ledger-key-{index}-{'k' * 500}": {
                f"hostile-nested-key-{'n' * 500}": "z" * 3_000,
            }
            for index in range(550)
        }
    )
    monkeypatch.setenv(
        "AUTOPLANNER_CODEX_ACTION_PLANNER_PROMPT_SNAPSHOT_MAX_BYTES",
        "12000",
    )

    payload, bounds = _bounded_planner_prompt_payload(handoff, round_index=2)
    compact = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert bounds["within_bound"] is True
    assert bounds["embedded_bytes"] <= 12_000
    assert len(_planner_json_bytes(payload)) == bounds["embedded_bytes"]
    assert bounds["compaction"] != "bounded_handoff"
    assert "doi:10.1000/embedded.snapshot" in compact
    assert "source-bound-process.pdf" in compact
    assert "resolve:compound-7" in compact
    assert "d" * 64 in compact
    assert "k" * 500 not in compact
    assert compact.count("hostile-ledger-key") <= 16

    monkeypatch.setattr(
        codex_action_planner,
        "_planner_prompt_snapshot_max_bytes",
        lambda: 7_000,
    )
    fixed_payload, fixed_bounds = _bounded_planner_prompt_payload(
        handoff,
        round_index=2,
    )
    fixed_compact = json.dumps(fixed_payload, ensure_ascii=False, sort_keys=True)
    assert fixed_bounds["compaction"] == "absolute_minimum_decision_projection"
    assert fixed_bounds["embedded_bytes"] <= 7_000
    assert "hostile-ledger-key" not in fixed_compact
    assert "doi:10.1000/embedded.snapshot" in fixed_compact
    assert "source-bound-process.pdf" in fixed_compact
    assert "resolve:compound-7" in fixed_compact
    assert "d" * 64 in fixed_compact
    fixed_handoff = fixed_payload["blackboard_handoff"]
    assert fixed_handoff["action_requirements"]["source_sensitive_actions"][
        "extract_pdf_literature_structures"
    ]["currently_required"] is True
    assert "source_capability_id" in fixed_handoff["action_requirements"][
        "source_sensitive_actions"
    ]["extract_pdf_literature_structures"]["accepted_payload_fields"]
    assert fixed_payload["no_file_read_required"] is True
    assert fixed_payload["shell_read_allowed"] is False


def test_handoff_hard_bound_handles_hostile_utf8_numbers_and_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = _planner_board(tmp_path)
    snapshot_path = _write_codex_blackboard_snapshot(
        board,
        run_dir=tmp_path,
        round_index=2,
    )
    handoff = json.loads(snapshot_path.read_text(encoding="utf-8"))["blackboard"]
    handoff["case_id"] = "🧪" * 20_000
    handoff["target_profile"]["target_name"] = "🧬" * 20_000
    handoff["state_counts"]["source_candidates"] = 10**20_000
    handoff["route_board"]["frontier_ledger"]["hostile_nonfinite"] = float(
        "nan"
    )
    handoff["route_board"]["frontier_ledger"].update(
        {
            f"hostile-{'🔬' * 400}-{index}": {
                f"nested-{'🧫' * 400}": "⚗️" * 4_000,
            }
            for index in range(100)
        }
    )
    handoff["action_requirements"][f"hostile-{'🧯' * 4_000}"] = {
        "payload": "💥" * 20_000,
    }
    monkeypatch.setattr(
        codex_action_planner,
        "_planner_prompt_snapshot_max_bytes",
        lambda: 7_000,
    )

    payload, bounds = _bounded_planner_prompt_payload(handoff, round_index=2)
    compact = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert bounds["within_bound"] is True
    assert bounds["embedded_bytes"] <= 7_000
    assert len(_planner_json_bytes(payload)) == bounds["embedded_bytes"]
    assert "doi:10.1000/embedded.snapshot" in compact
    assert "source-bound-process.pdf" in compact
    assert "resolve:compound-7" in compact
    assert "d" * 64 in compact
    assert "source_capability_id" in compact
    assert payload["no_file_read_required"] is True
    assert payload["input_refs_are_audit_only"] is True
    assert payload["shell_read_allowed"] is False


def test_main_and_repair_share_verified_bounded_snapshot(tmp_path: Path) -> None:
    board = _planner_board(tmp_path)
    snapshot_path = _write_codex_blackboard_snapshot(
        board,
        run_dir=tmp_path,
        round_index=2,
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    main_task = _codex_action_planner_task(
        blackboard=board,
        round_index=2,
        run_dir=tmp_path,
        snapshot_path=snapshot_path,
    )
    repair_task = _codex_action_planner_repair_task(
        blackboard=board,
        round_index=2,
        run_dir=tmp_path,
        snapshot_path=snapshot_path,
        invalid_batch={"schema_version": "agent_action_batch.v1", "actions": []},
        initial_validation={"accepted": False, "reasons": ["empty_action_batch"]},
    )

    for task in (main_task, repair_task):
        assert task.input_refs == [str(snapshot_path)]
        assert snapshot["prompt_payload_sha256"] in task.objective
        assert "doi:10.1000/embedded.snapshot" in task.objective
        assert "source-bound-process.pdf" in task.objective
        assert "resolve:compound-7" in task.objective
        assert "d" * 64 in task.objective
        assert "agent_blackboard.json" not in task.objective


def test_prompt_payload_fails_closed_when_fixed_schema_cannot_fit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = _planner_board(tmp_path)
    snapshot_path = _write_codex_blackboard_snapshot(
        board,
        run_dir=tmp_path,
        round_index=2,
    )
    handoff = json.loads(snapshot_path.read_text(encoding="utf-8"))["blackboard"]
    monkeypatch.setattr(
        codex_action_planner,
        "_planner_prompt_snapshot_max_bytes",
        lambda: 128,
    )

    with pytest.raises(ValueError, match="hard byte bound"):
        _bounded_planner_prompt_payload(handoff, round_index=2)
