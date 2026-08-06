from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from cascade_planner.legacy.harness_runtime.agent_action_planner import (
    _action_signature,
    plan_literature_evidence_followup_actions,
    validate_action_batch,
)
from cascade_planner.legacy.harness_runtime.agentic_blackboard import (
    _build_source_lifecycle,
    update_blackboard_from_action,
    update_budget_for_action,
)
from cascade_planner.legacy.harness_runtime.codex_action_planner import (
    _codex_action_planner_task,
    _locally_repair_invalid_codex_batch,
    _planner_context_summary,
    _write_codex_blackboard_snapshot,
)
from cascade_planner.harness.literature_pdf_extraction import (
    PAGE_FOCUS_ALGORITHM_VERSION,
)
from cascade_planner.harness.source_capabilities import (
    build_source_capability_queue,
    eligible_source_capabilities,
    matching_source_capabilities,
)
from cascade_planner.legacy.harness_runtime.tools import (
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


def test_transient_visual_failure_remains_retryable(tmp_path: Path) -> None:
    board = _source_board(tmp_path)
    board["literature_evidence"]["visual_chains"] = [
        {
            "schema_version": "agent_visual_chain_summary.v1",
            "accepted": False,
            "source_ref": SCIENCE_REF,
            "source_pdf_path": board["literature_evidence"]["source_candidates"][0][
                "local_pdf"
            ],
            "candidate_step_count": 0,
            "step_count": 0,
            "steps": [],
            "reasons": ["visual_direct_api_failed"],
        }
    ]

    queue = build_source_capability_queue(board, round_index=3)
    visual = eligible_source_capabilities(
        queue,
        "extract_visual_literature_chain",
    )

    assert [row["source_ref"] for row in visual] == [SCIENCE_REF]
    assert visual[0]["stage_from"] == "pdf_rendered"
    assert visual[0]["stage_to"] == "visual_extracted"


def test_explicit_empty_visual_terminal_does_not_repeat_but_runtime_failures_do(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    science = board["literature_evidence"]["source_candidates"][0]

    def visual_capabilities(reasons: list[str]) -> list[dict]:
        candidate = deepcopy(board)
        candidate["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "accepted": False,
                "source_ref": SCIENCE_REF,
                "source_pdf_path": science["local_pdf"],
                "candidate_step_count": 0,
                "steps": [],
                "reasons": reasons,
            }
        ]
        return eligible_source_capabilities(
            build_source_capability_queue(candidate, round_index=3),
            "extract_visual_literature_chain",
        )

    assert visual_capabilities(["no_relevant_steps"]) == []
    for retryable_reason in (
        "visual_direct_api_failed",
        "visual_literature_chain_timeout",
        "visual_api_auth_failed",
        "visual_input_images_missing",
    ):
        assert [row["source_ref"] for row in visual_capabilities([retryable_reason])] == [
            SCIENCE_REF
        ]
    assert [
        row["source_ref"]
        for row in visual_capabilities(
            ["no_relevant_steps", "visual_literature_chain_timeout"]
        )
    ] == [SCIENCE_REF]


def test_source_ref_only_pdf_evidence_is_ambiguous_across_article_and_si(
    tmp_path: Path,
) -> None:
    doi = "doi:10.1000/article-si"
    article_pdf = tmp_path / "article.pdf"
    si_pdf = tmp_path / "article_si.pdf"
    page = tmp_path / "page.png"
    for path in (article_pdf, si_pdf):
        path.write_bytes(b"%PDF-1.4\nsource\n")
    page.write_bytes(b"materialized-page")
    board = {
        "literature_evidence": {
            "source_candidates": [
                {
                    "source_ref": doi,
                    "content_scope": "article",
                    "local_pdf": str(article_pdf),
                },
                {
                    "source_ref": doi,
                    "content_scope": "supplementary_information",
                    "local_pdf": str(si_pdf),
                },
            ],
            "pdf_structure_evidence": [
                {
                    "accepted": True,
                    "source_ref": doi,
                    "rendered_page_count": 1,
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(page)}
                    ],
                    "reasons": [],
                }
            ],
            "visual_chains": [],
            "exact_rows": [],
            "structure_resolution_tasks": [],
        },
        "budget_state": {"visual_calls": 0, "max_visual_calls": 4},
    }

    ambiguous = build_source_capability_queue(board, round_index=2)
    assert eligible_source_capabilities(
        ambiguous,
        "extract_visual_literature_chain",
    ) == []
    pending_pdf = eligible_source_capabilities(
        ambiguous,
        "extract_pdf_literature_structures",
    )
    assert {row["payload_binding"]["pdf_path"] for row in pending_pdf} == {
        str(article_pdf),
        str(si_pdf),
    }

    scoped = deepcopy(board)
    scoped["literature_evidence"]["pdf_structure_evidence"][0][
        "content_scope"
    ] = "supplementary_information"
    resolved = build_source_capability_queue(scoped, round_index=2)
    visual = eligible_source_capabilities(
        resolved,
        "extract_visual_literature_chain",
    )
    assert [row["payload_binding"]["pdf_path"] for row in visual] == [
        str(si_pdf)
    ]


def test_zero_step_structure_gap_hands_off_to_resolution_capability(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    evidence = board["literature_evidence"]
    science = evidence["source_candidates"][0]
    evidence["visual_chains"] = [
        {
            "schema_version": "agent_visual_chain_summary.v1",
            "accepted": False,
            "source_ref": SCIENCE_REF,
            "source_pdf_path": science["local_pdf"],
            "candidate_step_count": 0,
            "structure_resolution_task_count": 1,
            "extraction_gaps": [
                {"label": "C42", "gap_type": "structure_gap"}
            ],
        }
    ]
    evidence["structure_resolution_tasks"] = [
        {
            "task_id": "resolve:C42",
            "label": "C42",
            "source_ref": SCIENCE_REF,
            "status": "open",
        }
    ]

    queue = build_source_capability_queue(board, round_index=3)

    assert eligible_source_capabilities(
        queue,
        "extract_visual_literature_chain",
    ) == []
    resolution = eligible_source_capabilities(
        queue,
        "resolve_literature_structure_task",
    )
    assert [row["payload_binding"]["task_id"] for row in resolution] == [
        "resolve:C42"
    ]


def test_stale_focus_zero_step_chain_retries_visual_before_resolution(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    evidence = board["literature_evidence"]
    science = evidence["source_candidates"][0]
    pdf_row = evidence["pdf_structure_evidence"][0]
    artifact = tmp_path / "legacy_pdf_evidence.json"
    artifact.write_text(
        json.dumps(
            {
                "accepted": True,
                "result": {
                    "focus_audit": {
                        "schema_version": "literature_pdf_page_focus_audit.v1",
                        "algorithm_version": "literature_pdf_page_focus.v1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pdf_row["artifact_ref"] = str(artifact)
    evidence["visual_chains"] = [
        {
            "schema_version": "agent_visual_chain_summary.v1",
            "accepted": False,
            "source_ref": SCIENCE_REF,
            "source_pdf_path": science["local_pdf"],
            "candidate_step_count": 0,
            "structure_resolution_task_count": 1,
            "extraction_gaps": [
                {"label": "C42", "gap_type": "structure_gap"}
            ],
        }
    ]
    evidence["structure_resolution_tasks"] = [
        {
            "task_id": "resolve:C42",
            "label": "C42",
            "source_ref": SCIENCE_REF,
            "status": "open",
        }
    ]

    stale_queue = build_source_capability_queue(board, round_index=3)
    visual = eligible_source_capabilities(
        stale_queue, "extract_visual_literature_chain"
    )
    assert [row["source_ref"] for row in visual] == [SCIENCE_REF]

    refreshed = deepcopy(board)
    refreshed["literature_evidence"]["visual_chains"][0][
        "page_focus_refresh_audit"
    ] = {
        "accepted": True,
        "current_algorithm_version": PAGE_FOCUS_ALGORITHM_VERSION,
    }
    current_queue = build_source_capability_queue(refreshed, round_index=3)
    assert eligible_source_capabilities(
        current_queue, "extract_visual_literature_chain"
    ) == []
    assert [
        row["payload_binding"]["task_id"]
        for row in eligible_source_capabilities(
            current_queue, "resolve_literature_structure_task"
        )
    ] == ["resolve:C42"]


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


def test_validator_rejects_source_action_when_capability_queue_has_no_match(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    evidence = board["literature_evidence"]
    evidence["source_candidates"] = evidence["source_candidates"][:1]
    evidence["pdf_structure_evidence"] = evidence["pdf_structure_evidence"][:1]
    science_pdf = evidence["source_candidates"][0]["local_pdf"]
    batch = {
        "schema_version": "agent_action_batch.v1",
        "case_id": board["case_id"],
        "round_index": 3,
        "actions": [
            _action(
                "science:pdf:duplicate",
                "extract_pdf_literature_structures",
                {"source_ref": SCIENCE_REF, "pdf_path": science_pdf},
            )
        ],
        "semantics": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "deterministic_validator_required": True,
        },
    }

    queue = build_source_capability_queue(board, round_index=3)
    validation = validate_action_batch(batch, blackboard=board)

    assert eligible_source_capabilities(
        queue,
        "extract_pdf_literature_structures",
    ) == []
    assert validation["accepted"] is False
    assert validation["batch_reasons"] == []
    assert validation["action_validations"][0]["reasons"] == [
        "source_capability_not_eligible:0:extract_pdf_literature_structures"
    ]


def test_host_bound_empty_payload_matches_effective_stale_history(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    evidence = board["literature_evidence"]
    evidence["source_candidates"] = evidence["source_candidates"][1:]
    evidence["pdf_structure_evidence"] = []
    action = _action(
        "patent:pdf",
        "extract_pdf_literature_structures",
        {},
    )
    batch = {
        "schema_version": "agent_action_batch.v1",
        "case_id": board["case_id"],
        "round_index": 3,
        "actions": [action],
    }
    first = validate_action_batch(batch, blackboard=board)
    assert first["accepted"] is True, first["reasons"]
    effective_action = deepcopy(action)
    effective_action["payload"] = first["action_validations"][0][
        "effective_payload"
    ]
    signature = _action_signature(effective_action)
    board["action_history"] = [
        {"round_index": 1, "stale": True, "action_signature": signature},
        {"round_index": 2, "stale": True, "action_signature": signature},
    ]

    repeated = validate_action_batch(batch, blackboard=board)

    assert repeated["accepted"] is False
    assert "stale_action_repeated:0:extract_pdf_literature_structures" in repeated[
        "reasons"
    ]


def test_focused_visual_repair_exception_revalidates_gap_render_and_budget(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    science = board["literature_evidence"]["source_candidates"][0]
    board["literature_evidence"]["visual_chains"] = [
        {
            "schema_version": "agent_visual_chain_summary.v1",
            "accepted": False,
            "source_ref": SCIENCE_REF,
            "source_pdf_path": science["local_pdf"],
            "candidate_step_count": 1,
            "step_count": 1,
            "condition_gap_labels": ["C42"],
            "reasons": ["visual_literature_chain_condition_gaps"],
        }
    ]
    action = _action(
        "science:visual:repair",
        "extract_visual_literature_chain",
        {
            "source_ref": SCIENCE_REF,
            "pdf_path": science["local_pdf"],
            "focused_gap_repair": True,
        },
    )
    batch = {
        "schema_version": "agent_action_batch.v1",
        "case_id": board["case_id"],
        "round_index": 3,
        "actions": [action],
    }

    accepted = validate_action_batch(batch, blackboard=board)
    exhausted_board = deepcopy(board)
    exhausted_board["budget_state"]["visual_calls"] = exhausted_board[
        "budget_state"
    ]["max_visual_calls"]
    exhausted = validate_action_batch(batch, blackboard=exhausted_board)
    closed_board = deepcopy(board)
    closed_board["literature_evidence"]["visual_chains"][0][
        "condition_gap_labels"
    ] = []
    closed = validate_action_batch(batch, blackboard=closed_board)

    assert accepted["accepted"] is True, accepted["reasons"]
    assert exhausted["accepted"] is False
    assert "visual_total_budget_exceeded" in exhausted["reasons"]
    assert closed["accepted"] is False
    assert (
        "extract_visual_literature_chain_requires_rendered_pdf_evidence:0"
        in closed["reasons"]
    )


def test_focused_visual_repair_binds_the_unique_gap_document_with_multiple_pdfs(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    evidence = board["literature_evidence"]
    science, patent = evidence["source_candidates"]
    patent_page = tmp_path / "patent_page_1.png"
    patent_page.write_bytes(b"materialized-patent-page")
    evidence["pdf_structure_evidence"][1] = {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "accepted": True,
        "source_ref": PATENT_REF,
        "source_pdf_path": patent["local_pdf"],
        "rendered_page_count": 1,
        "rendered_pages": [
            {"page_number": 1, "image_path": str(patent_page)}
        ],
        "reasons": [],
    }
    evidence["visual_chains"] = [
        {
            "schema_version": "agent_visual_chain_summary.v1",
            "accepted": False,
            "source_ref": SCIENCE_REF,
            "source_pdf_path": science["local_pdf"],
            "candidate_step_count": 1,
            "condition_gap_labels": ["C42"],
        }
    ]

    def validation_for(source: dict) -> dict:
        return validate_action_batch(
            {
                "schema_version": "agent_action_batch.v1",
                "case_id": board["case_id"],
                "round_index": 3,
                "actions": [
                    _action(
                        "visual:repair",
                        "extract_visual_literature_chain",
                        {
                            "source_ref": source["source_ref"],
                            "pdf_path": source["local_pdf"],
                            "focused_gap_repair": True,
                        },
                    )
                ],
            },
            blackboard=board,
        )

    correct = validation_for(science)
    wrong = validation_for(patent)

    assert correct["accepted"] is True, correct["reasons"]
    assert wrong["accepted"] is False
    assert (
        "extract_visual_literature_chain_requires_rendered_pdf_evidence:0"
        in wrong["reasons"]
    )


def test_strong_source_selectors_reject_conflicting_authority_fields(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    evidence = board["literature_evidence"]
    science, patent = evidence["source_candidates"]
    queue = build_source_capability_queue(board, round_index=3)
    visual = eligible_source_capabilities(
        queue,
        "extract_visual_literature_chain",
    )[0]
    assert matching_source_capabilities(
        queue,
        action_type="extract_visual_literature_chain",
        payload={
            "source_capability_id": visual["capability_id"],
            "source_ref": patent["source_ref"],
            "pdf_path": patent["local_pdf"],
        },
    ) == []
    for conflicting_field, conflicting_value in (
        ("doi", "10.9999/wrong"),
        ("pii", "S123456789012"),
        ("url", "https://example.com/wrong"),
        ("patent", "WO2024000001A1"),
        ("patent_publication", "WO2024000001A1"),
        ("artifact_ref", "wrong-artifact.json"),
    ):
        assert matching_source_capabilities(
            queue,
            action_type="extract_visual_literature_chain",
            payload={
                "source_capability_id": visual["capability_id"],
                conflicting_field: conflicting_value,
            },
        ) == []
    valid_artifact_ref = visual["prerequisite_evidence_refs"][0]
    assert len(
        matching_source_capabilities(
            queue,
            action_type="extract_visual_literature_chain",
            payload={
                "source_capability_id": visual["capability_id"],
                "artifact_ref": valid_artifact_ref,
            },
        )
    ) == 1
    assert len(
        matching_source_capabilities(
            queue,
            action_type="extract_visual_literature_chain",
            payload={
                "source_ref": science["source_ref"],
                "pdf_path": science["local_pdf"],
            },
        )
    ) == 1

    evidence["structure_resolution_tasks"] = [
        {
            "task_id": "resolve:C42",
            "label": "C42",
            "source_ref": science["source_ref"],
            "status": "open",
        }
    ]
    task_queue = build_source_capability_queue(board, round_index=3)
    assert matching_source_capabilities(
        task_queue,
        action_type="resolve_literature_structure_task",
        payload={
            "task_id": "resolve:C42",
            "source_ref": patent["source_ref"],
            "pdf_path": patent["local_pdf"],
        },
    ) == []

    evidence["structure_resolution_tasks"] = []
    evidence["visual_chains"] = [
        {
            "schema_version": "agent_visual_chain_summary.v1",
            "accepted": True,
            "chain_id": "science:chain",
            "source_ref": science["source_ref"],
            "source_pdf_path": science["local_pdf"],
            "candidate_step_count": 1,
            "steps": [{"step_id": "science:1"}],
        }
    ]
    chain_queue = build_source_capability_queue(board, round_index=3)
    assert matching_source_capabilities(
        chain_queue,
        action_type="compile_exact_literature_rows",
        payload={
            "chain_id": "science:chain",
            "source_ref": patent["source_ref"],
            "pdf_path": patent["local_pdf"],
        },
    ) == []
    assert len(
        matching_source_capabilities(
            chain_queue,
            action_type="compile_exact_literature_rows",
            payload={
                "visual_chain_id": "science:chain",
                "source_ref": science["source_ref"],
            },
        )
    ) == 1


def test_resolve_capability_cost_and_target_identity_shortcut_are_host_derived(
    tmp_path: Path,
) -> None:
    board = _source_board(tmp_path)
    evidence = board["literature_evidence"]
    evidence["source_candidates"] = evidence["source_candidates"][:1]
    evidence["pdf_structure_evidence"] = evidence["pdf_structure_evidence"][:1]
    board["budget_state"]["visual_calls"] = 0
    board["budget_state"]["max_visual_calls"] = 1
    evidence["structure_resolution_tasks"] = [
        {
            "task_id": "resolve:C41",
            "label": "C41",
            "source_ref": SCIENCE_REF,
            "status": "open",
        },
        {
            "task_id": "resolve:C42",
            "label": "C42",
            "source_ref": SCIENCE_REF,
            "status": "open",
        },
    ]
    batch = {
        "schema_version": "agent_action_batch.v1",
        "case_id": board["case_id"],
        "round_index": 3,
        "actions": [
            _action(
                "resolve:C41",
                "resolve_literature_structure_task",
                {"task_id": "resolve:C41"},
            ),
            _action(
                "resolve:C42",
                "resolve_literature_structure_task",
                {"task_id": "resolve:C42"},
            ),
        ],
    }
    validation = validate_action_batch(batch, blackboard=board)

    assert validation["accepted"] is False
    assert "visual_total_budget_exceeded" in validation["reasons"]
    assert [row["cost"]["visual_calls"] for row in validation["action_validations"]] == [
        1,
        1,
    ]
    assert all(
        row["effective_payload"]["run_visual"] is True
        for row in validation["action_validations"]
    )

    for bad_value in (None, "", False):
        invalid = deepcopy(batch)
        invalid["actions"] = invalid["actions"][:1]
        invalid["actions"][0]["payload"]["run_visual"] = bad_value
        invalid_validation = validate_action_batch(invalid, blackboard=board)
        assert invalid_validation["accepted"] is False
        assert any(
            "source_capability_not_eligible" in reason
            for reason in invalid_validation["reasons"]
        )

    target_board = deepcopy(board)
    target_board["target_profile"].update(
        {"target_name": "nirmatrelvir", "target_smiles": "CCO"}
    )
    target_board["budget_state"]["max_visual_calls"] = 0
    target_board["literature_evidence"]["structure_resolution_tasks"] = [
        {
            "task_id": "resolve:target",
            "label": "nirmatrelvir",
            "source_ref": SCIENCE_REF,
            "status": "open",
        }
    ]
    target_batch = {
        **batch,
        "actions": [
            _action(
                "resolve:target",
                "resolve_literature_structure_task",
                {"task_id": "resolve:target"},
            )
        ],
    }
    target_validation = validate_action_batch(target_batch, blackboard=target_board)

    assert target_validation["accepted"] is True, target_validation["reasons"]
    target_row = target_validation["action_validations"][0]
    assert target_row["cost"]["visual_calls"] == 0
    assert target_row["effective_payload"]["run_visual"] is False
    assert target_row["effective_payload"]["target_identity_shortcut"] is True
    reserved = update_budget_for_action(
        target_board,
        "resolve_literature_structure_task",
        payload=target_row["effective_payload"],
        resource_cost=target_row["cost"],
    )
    assert reserved["budget_state"]["visual_calls"] == 0
    effective_action = deepcopy(target_batch["actions"][0])
    effective_action["payload"] = target_row["effective_payload"]
    effective_action["_host_resource_cost"] = target_row["cost"]
    recorded = update_blackboard_from_action(
        target_board,
        action=effective_action,
        action_result={"accepted": True, "result": {}},
        round_index=3,
        run_dir=tmp_path,
    )
    assert recorded["action_history"][-1]["resource_cost"] == target_row["cost"]

    forged = deepcopy(target_batch)
    forged["actions"][0]["payload"] = {
        "task_id": "resolve:C41",
        "run_visual": False,
        "target_identity_shortcut": True,
    }
    forged_validation = validate_action_batch(forged, blackboard=board)
    assert forged_validation["accepted"] is False


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
