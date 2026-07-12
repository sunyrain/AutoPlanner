from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from cascade_planner.harness.agent_action_planner import (
    plan_literature_evidence_followup_actions,
    validate_action_batch,
)
from cascade_planner.harness.agentic_blackboard import _build_source_lifecycle
from cascade_planner.harness.codex_action_planner import (
    _codex_action_planner_task,
    _locally_repair_invalid_codex_batch,
    _planner_context_summary,
    _write_codex_blackboard_snapshot,
)
from cascade_planner.harness.source_capabilities import (
    build_source_capability_queue,
    eligible_source_capabilities,
)
from cascade_planner.harness.tools import (
    ToolExecutionState,
    _validate_local_pdf_source_binding,
)


SCIENCE_REF = "doi:10.1126/science.abf9559"
PATENT_REF = "patent:WO2021250648A1"
PATENT_URL = "https://patents.google.com/patent/WO2021250648A1/en"


def _source_board(tmp_path: Path) -> dict:
    science_pdf = tmp_path / "science_si.pdf"
    patent_pdf = tmp_path / "patent.pdf"
    rendered_page = tmp_path / "science_page_1.png"
    science_pdf.write_bytes(b"%PDF-1.4\nscience\n")
    patent_pdf.write_bytes(b"%PDF-1.4\npatent\n")
    rendered_page.write_bytes(b"not-a-real-png-but-materialized")
    return {
        "case_id": "nirmatrelvir_v7_regression",
        "target_profile": {
            "valid": True,
            "target_name": "nirmatrelvir",
            "target_smiles": "CCO",
            "canonical_smiles": "CCO",
        },
        "budget_state": {
            "rounds_completed": 2,
            "visual_calls": 0,
            "max_visual_calls": 4,
            "scout_calls": 0,
            "max_scout_calls": 4,
            "child_target_runs": 0,
            "max_child_target_runs": 4,
        },
        "literature_evidence": {
            "source_candidates": [
                {
                    "candidate_id": "science-si",
                    "source_ref": SCIENCE_REF,
                    "title": "Science supporting information",
                    "content_scope": "supplementary_information",
                    "local_pdf": str(science_pdf),
                },
                {
                    "candidate_id": "patent",
                    "source_ref": PATENT_REF,
                    "title": "Nirmatrelvir patent",
                    "local_pdf": str(patent_pdf),
                },
            ],
            "pdf_structure_evidence": [
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "accepted": True,
                    "source_ref": SCIENCE_REF,
                    "content_scope": "supplementary_information",
                    "source_pdf_path": str(science_pdf),
                    "rendered_page_count": 1,
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(rendered_page)}
                    ],
                    "reasons": [],
                },
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "accepted": False,
                    "source_ref": PATENT_URL,
                    "source_pdf_path": str(patent_pdf),
                    "rendered_page_count": 0,
                    "rendered_pages": [],
                    "reasons": ["local_pdf_source_ref_mismatch"],
                },
            ],
            "visual_chains": [],
            "exact_rows": [],
            "structure_resolution_tasks": [],
        },
        "action_history": [],
        "current_belief": {},
        "retrosynthetic_proposals": [],
        "recursive_hypothesis_tasks": [],
        "bridge_tasks": [],
    }


def _action(
    action_id: str,
    action_type: str,
    payload: dict,
) -> dict:
    return {
        "schema_version": "agent_action.v1",
        "action_id": action_id,
        "action_type": action_type,
        "rationale": f"execute {action_type}",
        "expected_artifact": "typed artifact",
        "success_condition": "artifact or explicit rejection",
        "payload": payload,
    }


def _child_action() -> dict:
    return _action(
        "child:C42",
        "expand_child_target",
        {
            "subgoal_targets": [
                {
                    "smiles": "CCN",
                    "name": "C42 upstream child",
                    "target_equivalence_audit_required": True,
                    "exact_target_override": True,
                    "no_solved_claim": True,
                    "child_route_cannot_promote_parent": True,
                    "policy_runtime_rebuild": True,
                }
            ],
            "child_policy_runtime_rebuild": True,
            "no_solved_claim": True,
        },
    )


def test_google_patent_url_and_patent_locator_bind_same_local_pdf(tmp_path: Path) -> None:
    patent_pdf = tmp_path / "patent.pdf"
    patent_pdf.write_bytes(b"%PDF-1.4\npatent\n")
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={
            "target_name": "nirmatrelvir",
            "target_smiles": "CCO",
            "local_literature_cache": [
                {
                    "candidate_id": "patent-cache",
                    "source_ref": PATENT_REF,
                    "local_pdf": str(patent_pdf),
                }
            ],
        },
        preflight={"case_id": "nirmatrelvir", "accepted": True},
    )

    binding = _validate_local_pdf_source_binding(
        state,
        {"source_ref": PATENT_URL, "pdf_path": str(patent_pdf)},
        pdf_path=patent_pdf,
    )

    assert binding["accepted"] is True
    assert binding["payload"]["source_ref"] == PATENT_REF


def test_queue_and_lifecycle_require_real_accepted_pdf_render(tmp_path: Path) -> None:
    board = _source_board(tmp_path)

    queue = build_source_capability_queue(board, round_index=3)
    visual = eligible_source_capabilities(
        queue,
        "extract_visual_literature_chain",
    )
    pdf = eligible_source_capabilities(
        queue,
        "extract_pdf_literature_structures",
    )
    lifecycle = _build_source_lifecycle(board["literature_evidence"])
    lifecycle_by_ref = {row["source_ref"]: row for row in lifecycle}

    assert [row["source_ref"] for row in visual] == [SCIENCE_REF]
    assert [row["source_ref"] for row in pdf] == [PATENT_REF]
    assert lifecycle_by_ref[SCIENCE_REF]["stage"] == "pdf_rendered"
    assert lifecycle_by_ref[PATENT_REF]["stage"] == "local_pdf_available"
    assert queue["budget"]["literature_source_units_max_this_round"] == 3
    assert visual[0]["cost"]["literature_source_units"] == 1
    assert visual[0]["cost"]["visual_calls"] == 1


def test_queue_is_order_invariant_and_drives_codex_and_deterministic_views(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    science_only = deepcopy(board)
    science_only["literature_evidence"]["source_candidates"] = science_only[
        "literature_evidence"
    ]["source_candidates"][:1]
    science_only["literature_evidence"]["pdf_structure_evidence"] = science_only[
        "literature_evidence"
    ]["pdf_structure_evidence"][:1]

    forward = build_source_capability_queue(board, round_index=3)
    reversed_board = deepcopy(board)
    reversed_board["literature_evidence"]["source_candidates"].reverse()
    reversed_board["literature_evidence"]["pdf_structure_evidence"].reverse()
    reverse = build_source_capability_queue(reversed_board, round_index=3)
    context = _planner_context_summary(board, round_index=3)
    deterministic = plan_literature_evidence_followup_actions(
        science_only,
        round_index=3,
    )
    snapshot_path = _write_codex_blackboard_snapshot(
        board,
        run_dir=tmp_path,
        round_index=3,
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    task = _codex_action_planner_task(
        blackboard=board,
        round_index=3,
        run_dir=tmp_path,
        snapshot_path=snapshot_path,
    )

    assert forward["content_sha256"] == reverse["content_sha256"]
    assert forward["capabilities"] == reverse["capabilities"]
    assert context["literature_processing"]["source_capability_queue"][
        "content_sha256"
    ] == forward["content_sha256"]
    assert context["literature_processing"]["pending_visual_extraction_sources"][
        0
    ]["source_ref"] == SCIENCE_REF
    assert context["budget_remaining"][
        "literature_source_units_max_this_round"
    ] == 3
    requirements = context["action_payload_requirements"][
        "source_sensitive_actions"
    ]
    assert requirements["extract_visual_literature_chain"]["binding_candidates"][
        0
    ]["source_ref"] == SCIENCE_REF
    assert requirements["extract_pdf_literature_structures"]["binding_candidates"][
        0
    ]["source_ref"] == PATENT_REF
    assert "literature_source_units_max_this_round" in task.objective
    assert "source_capability_queue" in task.objective
    assert snapshot["prompt_payload"]["blackboard_handoff"]["evidence_board"][
        "source_capability_queue"
    ]["capabilities"]
    assert deterministic[0]["action_type"] == "extract_visual_literature_chain"
    assert deterministic[0]["payload"]["source_ref"] == SCIENCE_REF
    assert deterministic[0]["payload"]["source_capability_id"].startswith(
        "source-capability:sha256:"
    )


def test_validator_reports_action_local_failure_and_salvage_keeps_siblings(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    science_pdf = board["literature_evidence"]["source_candidates"][0]["local_pdf"]
    patent_pdf = board["literature_evidence"]["source_candidates"][1]["local_pdf"]
    batch = {
        "schema_version": "agent_action_batch.v1",
        "case_id": board["case_id"],
        "round_index": 3,
        "actions": [
            _action(
                "science:visual",
                "extract_visual_literature_chain",
                {"source_ref": SCIENCE_REF, "pdf_path": science_pdf},
            ),
            _action(
                "patent:visual",
                "extract_visual_literature_chain",
                {"source_ref": PATENT_URL, "pdf_path": patent_pdf},
            ),
            _child_action(),
        ],
        "semantics": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "deterministic_validator_required": True,
        },
    }

    validation = validate_action_batch(batch, blackboard=board)
    repaired = _locally_repair_invalid_codex_batch(
        batch,
        validation=validation,
        blackboard=board,
    )

    assert validation["accepted"] is False
    assert validation["batch_reasons"] == []
    assert [row["accepted"] for row in validation["action_validations"]] == [
        True,
        False,
        True,
    ]
    assert validation["action_validations"][1]["reasons"] == [
        "extract_visual_literature_chain_requires_rendered_pdf_evidence:1"
    ]
    assert validation["salvage_allowed"] is True
    assert repaired is not None
    assert [row["action_id"] for row in repaired["actions"]] == [
        "science:visual",
        "child:C42",
    ]
    repaired_validation = validate_action_batch(repaired, blackboard=board)
    assert repaired_validation["accepted"] is True, repaired_validation["reasons"]
    assert repaired["repair_audit"]["dropped_action_ids"] == ["patent:visual"]


def test_unsafe_reason_remains_batch_global_and_cannot_be_salvaged(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    batch = {
        "schema_version": "agent_action_batch.v1",
        "case_id": board["case_id"],
        "round_index": 3,
        "actions": [
            _action(
                "safe",
                "generate_disconnection_hypotheses",
                {"no_solved_claim": True},
            ),
            {
                **_action(
                    "unsafe",
                    "generate_disconnection_hypotheses",
                    {"no_solved_claim": True},
                ),
                "reaction_smiles": "CC>>C",
            },
        ],
    }

    validation = validate_action_batch(batch, blackboard=board)
    repaired = _locally_repair_invalid_codex_batch(
        batch,
        validation=validation,
        blackboard=board,
    )

    assert "raw_reaction_injection" in validation["batch_reasons"]
    assert validation["salvage_allowed"] is False
    assert repaired is None


def test_round_three_salvage_drops_invalid_visual_and_refits_source_budget(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    patent_pdf = board["literature_evidence"]["source_candidates"][1]["local_pdf"]
    batch = {
        "schema_version": "agent_action_batch.v1",
        "case_id": board["case_id"],
        "round_index": 3,
        "actions": [
            _action(
                "patent:visual",
                "extract_visual_literature_chain",
                {"source_ref": PATENT_URL, "pdf_path": patent_pdf},
            ),
            _child_action(),
            _action(
                "discover:independent",
                "search_literature",
                {
                    "search_intent": "find an independent process source",
                    "queries": ["nirmatrelvir independent process synthesis"],
                    "max_sources": 3,
                },
            ),
        ],
    }

    validation = validate_action_batch(batch, blackboard=board)
    repaired = _locally_repair_invalid_codex_batch(
        batch,
        validation=validation,
        blackboard=board,
    )

    assert "literature_source_round_budget_exceeded" in validation["batch_reasons"]
    assert repaired is not None
    assert [row["action_id"] for row in repaired["actions"]] == [
        "child:C42",
        "discover:independent",
    ]
    search = repaired["actions"][1]
    assert search["payload"]["max_sources"] == 3
    final_validation = validate_action_batch(repaired, blackboard=board)
    assert final_validation["accepted"] is True, final_validation["reasons"]
