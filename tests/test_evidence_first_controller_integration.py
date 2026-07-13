from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cascade_planner.harness.agentic_blackboard_controller as controller_module
from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.harness.agentic_blackboard_controller import (
    run_agentic_blackboard_controller,
)


_FIXTURES = Path(__file__).parent / "fixtures"
_SOURCE_PDF = (_FIXTURES / "source_evidence_stub.pdf").resolve()
_SOURCE_PAGE = (_FIXTURES / "source_page.ppm").resolve()
_SOURCE_REF = "doi:10.1000/evidence-first-controller"


def test_recovered_campaign_defers_bootstrap_for_bound_exact_compile(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    page = tmp_path / "page.png"
    pdf.write_bytes(b"%PDF-1.4\n")
    page.write_bytes(b"materialized-page")
    board = {
        "case_id": "evidence-first",
        "target_profile": {
            "valid": True,
            "target_name": "acetaldehyde",
            "target_smiles": "CC=O",
        },
        "budget_state": {
            "rounds_completed": 2,
            "visual_calls": 1,
            "max_visual_calls": 4,
            "scout_calls": 0,
            "max_scout_calls": 4,
        },
        "literature_evidence": {
            "source_candidates": [
                {
                    "source_ref": _SOURCE_REF,
                    "local_pdf": str(pdf),
                }
            ],
            "pdf_structure_evidence": [
                {
                    "accepted": True,
                    "source_ref": _SOURCE_REF,
                    "source_pdf_path": str(pdf),
                    "rendered_page_count": 1,
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(page)}
                    ],
                    "reasons": [],
                }
            ],
            "visual_chains": [
                {
                    "schema_version": "agent_visual_chain_summary.v1",
                    "accepted": True,
                    "chain_id": "visual:exact",
                    "source_ref": _SOURCE_REF,
                    "source_pdf_path": str(pdf),
                    "candidate_step_count": 1,
                    "steps": [
                        {
                            "step_id": "oxidation",
                            "product_smiles": "CC=O",
                            "reactant_smiles": ["CCO"],
                            "allowed_use": "exact_candidate",
                        }
                    ],
                }
            ],
            "exact_rows": [],
            "structure_resolution_tasks": [],
        },
    }

    assert controller_module._controller_evidence_first_work_pending(board)

    board["literature_evidence"]["visual_chains"] = []
    board["literature_evidence"]["pdf_structure_evidence"] = []
    board["literature_evidence"]["source_candidates"] = []
    assert not controller_module._controller_evidence_first_work_pending(board)


def test_source_bound_evidence_queue_bypasses_redundant_codex_planner(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    page = tmp_path / "page.png"
    pdf.write_bytes(b"%PDF-1.4\n")
    page.write_bytes(b"materialized-page")
    board = {
        "case_id": "evidence-first",
        "target_profile": {
            "valid": True,
            "target_name": "acetaldehyde",
            "target_smiles": "CC=O",
        },
        "budget_state": {
            "rounds_completed": 2,
            "visual_calls": 1,
            "max_visual_calls": 4,
            "scout_calls": 0,
            "max_scout_calls": 4,
        },
        "literature_evidence": {
            "source_candidates": [
                {"source_ref": _SOURCE_REF, "local_pdf": str(pdf)}
            ],
            "pdf_structure_evidence": [
                {
                    "accepted": True,
                    "source_ref": _SOURCE_REF,
                    "source_pdf_path": str(pdf),
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
    }

    with patch.object(
        controller_module,
        "plan_action_batch_with_codex",
        side_effect=AssertionError("Codex planner must not be invoked"),
    ):
        batch = controller_module._obtain_action_batch(
            blackboard=board,
            round_index=3,
            run_dir=tmp_path,
            state=None,
            action_planner=None,
            exhaust_round_budget=True,
            use_codex_action_planner=True,
            allow_deterministic_fallback=True,
        )

    assert batch["mode"] == "deterministic_evidence_first_scheduler"
    assert batch["planner_audit"]["codex_action_planner_invoked"] is False
    assert batch["actions"][0]["action_type"] == (
        "extract_visual_literature_chain"
    )


def _team_artifact(
    case_id: str,
    *,
    target_smiles: str,
    precursor_smiles: str,
) -> dict[str, Any]:
    return {
        "schema_version": "retrosynthesis_proposal_report_artifact.v1",
        "artifact_id": f"{case_id}:team_report:{target_smiles}",
        "artifact_type": "RetrosynthesisProposalReport",
        "case_id": case_id,
        "source": "codex_cli",
        "input_refs": ["context_snapshot.json"],
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "offline controller integration draft",
        "payload": {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": case_id,
            "agent_role": "retrosynthesis_team_coordinator",
            "target_smiles": target_smiles,
            "candidates": [
                {
                    "schema_version": "retrosynthesis_candidate.v1",
                    "candidate_id": f"candidate:{target_smiles}:{precursor_smiles}",
                    "product_smiles": target_smiles,
                    "precursor_smiles": [precursor_smiles],
                    "reaction_family": "evidence-gated test transformation",
                    "product_retron_type": "carbonyl interconversion",
                    "transformation_rationale": (
                        "Keep this proposal advisory until the host verifier accepts it."
                    ),
                    "source_channel": "codex_strategy",
                    "source_refs": ["child:target_structure_strategist"],
                    "evidence_refs": [],
                    "evidence_level": "model_only",
                    "confidence": "medium",
                    "conditions": [],
                    "catalyst": "",
                    "enzyme": "",
                    "limitations": ["model hypothesis only"],
                    "required_validation": [
                        "forward reconstruction",
                        "stock audit",
                    ],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                }
            ],
            "evidence_refs": [],
            "limitations": ["No parent proof."],
            "no_solved_claim": True,
        },
    }


def _team_run_record(task: Any, artifact: dict[str, Any]) -> WorkerRunRecord:
    payload = dict(artifact["payload"])
    return WorkerRunRecord(
        run_id=f"{task.task_id}:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        backend="codex_cli",
        output_artifact=artifact,
        output_validation={"accepted": True, "reasons": []},
        usage={"input_tokens": 100, "output_tokens": 20},
        metadata={
            "session_id": "evidence-first-controller-integration",
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
                            **payload,
                            "agent_role": role,
                            "candidates": list(payload["candidates"])
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


def _action(
    *,
    round_index: int,
    action_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "agent_action.v1",
        "action_id": f"r{round_index}:{action_type}",
        "action_type": action_type,
        "rationale": "Advance one deterministic source-lifecycle stage.",
        "expected_artifact": f"{action_type}.v1",
        "success_condition": "The requested evidence stage is recorded.",
        "payload": payload,
    }


def _batch(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "agent_action_batch.v1",
        "case_id": blackboard["case_id"],
        "round_index": round_index,
        "mode": "offline_evidence_first_controller_integration",
        "actions": [action],
        "semantics": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "deterministic_validator_required": True,
        },
    }


def test_late_exact_row_unlocks_one_child_frontier_and_resumes_campaign_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise discovery -> extraction -> proof wake-up through the controller.

    The root proposal creates a non-root frontier immediately, but its proposal
    gate must remain closed until the late exact-row phase makes a host-replayed
    L2 proof available.  Proof reconciliation itself must never invoke the
    proposal runner or consume either campaign budget.
    """

    assert _SOURCE_PDF.is_file()
    assert _SOURCE_PAGE.is_file()
    runner_targets: list[str] = []
    phase = {"exact_row_arrived": False}
    reconciliation_observations: list[dict[str, Any]] = []

    def runner(task: Any) -> WorkerRunRecord:
        context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
        target = str(context["target"]["smiles"])
        runner_targets.append(target)
        precursor = {
            "CCO": "CC=O",
            # Keep the resumed proposal element-conserving so admission accepts
            # it, but leave it unmapped so its own child remains proof-blocked.
            "CC=O": "CC(O)O",
        }.get(target)
        if precursor is None:
            raise AssertionError(f"unexpected duplicate campaign resume for {target}")
        return _team_run_record(
            task,
            _team_artifact(
                task.case_id,
                target_smiles=target,
                precursor_smiles=precursor,
            ),
        )

    original_verify = controller_module.verify_codex_consensus_graph

    def verify_without_optional_mapper(
        graph: dict[str, Any],
        *,
        exact_rows=(),
        stock_closed_smiles=(),
        **kwargs: Any,
    ) -> dict[str, Any]:
        rows = list(exact_rows)
        # Never import or initialize the optional mapper.  The late exact row
        # must retain its own mapped reaction through blackboard compaction.
        kwargs.pop("enable_optional_rxnmapper", None)
        return original_verify(
            graph,
            exact_rows=rows,
            stock_closed_smiles=stock_closed_smiles,
            enable_optional_rxnmapper=False,
            **kwargs,
        )

    monkeypatch.setattr(
        controller_module,
        "verify_codex_consensus_graph",
        verify_without_optional_mapper,
    )

    original_reconcile = controller_module.reconcile_codex_campaign_proof_state

    def observed_reconcile(*args: Any, **kwargs: Any) -> dict[str, Any]:
        durable_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
        durable = json.loads(durable_path.read_text(encoding="utf-8"))
        result = original_reconcile(*args, **kwargs)
        leaf = next(
            (
                row
                for row in (result.get("frontier_queue") or {}).get("jobs") or []
                if row.get("frontier_smiles") == "CC=O"
            ),
            {},
        )
        reconciliation_observations.append(
            {
                "exact_row_arrived": phase["exact_row_arrived"],
                "accepted_expansion_count": int(
                    (durable.get("campaign") or {}).get("accepted_expansion_count")
                    or 0
                ),
                "attempt_run_count": int(
                    (durable.get("campaign") or {}).get("attempt_run_count") or 0
                ),
                "proposal_runner_invoked": result.get("proposal_runner_invoked"),
                "expansion_budget_consumed": result.get(
                    "expansion_budget_consumed"
                ),
                "enabled_job_count": int(
                    (result.get("frontier_sync") or {}).get("enabled_job_count")
                    or 0
                ),
                "validated_proof_count": int(
                    ((result.get("reaction_proof_state") or {}).get("summary") or {}).get(
                        "validated"
                    )
                    or 0
                ),
                "leaf_state": str(leaf.get("state") or ""),
                "leaf_proposal_allowed": (
                    (leaf.get("metadata") or {}).get("proposal_expansion_allowed")
                ),
                "leaf_gate_status": str(
                    (
                        (leaf.get("metadata") or {}).get(
                            "proposal_expansion_gate"
                        )
                        or {}
                    ).get("status")
                    or ""
                ),
            }
        )
        return result

    monkeypatch.setattr(
        controller_module,
        "reconcile_codex_campaign_proof_state",
        observed_reconcile,
    )

    def planner(
        *,
        blackboard: dict[str, Any],
        round_index: int,
        run_dir: Path,
    ) -> dict[str, Any]:
        del run_dir
        if round_index == 1:
            action = _action(
                round_index=round_index,
                action_type="search_literature",
                payload={
                    "schema_version": "agentic_literature_search_payload.v1",
                    "search_intent": "target_proximal_source_discovery",
                    "query": "ethanol reduction exact reaction",
                    "queries": ["ethanol reduction exact reaction"],
                    "search_queries": ["ethanol reduction exact reaction"],
                    "max_sources": 1,
                    "source_acquisition_policy": {
                        "schema_version": "agentic_source_acquisition_policy.v1",
                        "codex_online_first": True,
                        "local_pdf_fallback_allowed": True,
                        "placeholder_allowed_after_failures": True,
                        "auto_local_pdf_requires_agent_discovered_metadata": True,
                        "fallback_order": [
                            "codex_online",
                            "local_pdf",
                            "placeholder",
                        ],
                        "no_solved_claim": True,
                    },
                    "no_solved_claim": True,
                },
            )
        elif round_index == 2:
            action = _action(
                round_index=round_index,
                action_type="extract_pdf_literature_structures",
                payload={
                    "source_ref": _SOURCE_REF,
                    "pdf_path": str(_SOURCE_PDF),
                },
            )
        elif round_index == 3:
            action = _action(
                round_index=round_index,
                action_type="extract_visual_literature_chain",
                payload={
                    "source_ref": _SOURCE_REF,
                    "pdf_path": str(_SOURCE_PDF),
                    "image_paths": [str(_SOURCE_PAGE)],
                },
            )
        else:
            action = _action(
                round_index=round_index,
                action_type="compile_exact_literature_rows",
                payload={
                    "source_ref": _SOURCE_REF,
                    "pdf_path": str(_SOURCE_PDF),
                    "chain_id": "visual:late-root-edge",
                    "compile_attempt": 1,
                },
            )
        return _batch(
            blackboard=blackboard,
            round_index=round_index,
            action=action,
        )

    def mock_search(*_: Any) -> dict[str, Any]:
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": True,
            "source_candidates": [
                {
                    "schema_version": "literature_source_candidate.v1",
                    "source_ref": _SOURCE_REF,
                    "doi": "10.1000/evidence-first-controller",
                    "local_pdf": str(_SOURCE_PDF),
                    "access_status": "local_pdf_available",
                    "source_type": "literature_metadata+local_pdf",
                }
            ],
            "source_refs": [_SOURCE_REF],
            "source_discovery_mode": "codex_online",
            "reasons": [],
            "no_solved_claim": True,
        }

    def mock_pdf(*_: Any) -> dict[str, Any]:
        return {
            "schema_version": "literature_pdf_structure_evidence.v1",
            "accepted": True,
            "evidence_id": "pdf:late-root-edge",
            "source_ref": _SOURCE_REF,
            "source_pdf_path": str(_SOURCE_PDF),
            "summary": {
                "rendered_page_count": 1,
                "indexed_image_count": 1,
                "scheme_crop_count": 1,
                "compound_text_snippet_count": 0,
            },
            "rendered_pages": [{"image_path": str(_SOURCE_PAGE)}],
            "indexed_images": [{"image_path": str(_SOURCE_PAGE)}],
            "reasons": [],
            "no_solved_claim": True,
        }

    def mock_visual(*_: Any) -> dict[str, Any]:
        return {
            "schema_version": "visual_literature_chain_extraction_result.v1",
            "accepted": True,
            "chain_id": "visual:late-root-edge",
            "source_ref": _SOURCE_REF,
            "source_pdf_path": str(_SOURCE_PDF),
            "candidate_chain": {
                "schema_version": "visual_structure_candidate_chain.v1",
                "source_ref": _SOURCE_REF,
                "steps": [
                    {
                        "step_id": "visual:acetaldehyde_to_ethanol",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC=O"],
                        "main_reactant_smiles": "CC=O",
                        "confidence": "high",
                        "allowed_use": "exact_candidate",
                        "not_exact_literature_segment": False,
                    }
                ],
            },
            "candidate_quality": {
                "acceptance_level": "source_detail_exact_candidate",
                "exact_ready": True,
                "exploratory_accepted": True,
            },
            "reasons": [],
            "no_solved_claim": True,
        }

    def mock_exact(*_: Any) -> dict[str, Any]:
        phase["exact_row_arrived"] = True
        return {
            "schema_version": "source_detail_chain_route.v1",
            "accepted": True,
            "exact_rows": [
                {
                    "row_id": "source_detail_exact_step:acetaldehyde_to_ethanol",
                    "source_ref": _SOURCE_REF,
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CC=O"],
                    "atom_mapped_reaction_smiles": (
                        "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
                    ),
                    "source_detail_exact_step": True,
                    "relation_type": "exact",
                }
            ],
            "reasons": [],
            "no_solved_claim": True,
        }

    result = run_agentic_blackboard_controller(
        target_name="ethanol",
        target_smiles="CCO",
        output_dir=tmp_path,
        max_rounds=4,
        auto_discover_local_pdfs=False,
        use_codex_agent_team=True,
        codex_agent_team_max_depth=3,
        codex_agent_team_max_expansions=3,
        codex_agent_team_max_attempt_runs=6,
        codex_agent_team_bootstrap_expansions=1,
        codex_agent_team_max_expansions_per_invocation=1,
        codex_agent_team_max_attempt_runs_per_invocation=1,
        codex_agent_team_frontier_batch_size=1,
        codex_agent_team_auto_resume=True,
        codex_agent_team_runner=runner,
        action_planner=planner,
        mock_tool_results={
            "search_literature": mock_search,
            "extract_pdf_literature_structures": mock_pdf,
            "extract_visual_literature_chain": mock_visual,
            "compile_exact_literature_rows": mock_exact,
        },
    )

    board = result["agent_blackboard"]
    campaign = board["codex_agent_team"]["campaign"]
    before_exact = [
        row
        for row in reconciliation_observations
        if row["exact_row_arrived"] is False and row["leaf_state"]
    ]
    unlock = next(
        row
        for row in reconciliation_observations
        if row["exact_row_arrived"] is True and row["enabled_job_count"] == 1
    )

    assert before_exact
    assert all(row["accepted_expansion_count"] == 1 for row in before_exact)
    assert all(row["attempt_run_count"] == 1 for row in before_exact)
    assert all(row["leaf_state"] == "pending" for row in before_exact)
    assert all(row["leaf_proposal_allowed"] is False for row in before_exact)
    assert all(
        row["leaf_gate_status"]
        == "blocked_pending_current_host_l2_parent_proof"
        for row in before_exact
    )

    assert unlock["accepted_expansion_count"] == 1
    assert unlock["attempt_run_count"] == 1
    assert unlock["validated_proof_count"] >= 1
    assert unlock["leaf_state"] == "pending"
    assert unlock["leaf_proposal_allowed"] is True
    assert "enabled_by_current_host_l2_parent_proof" in unlock[
        "leaf_gate_status"
    ]
    assert all(
        row["proposal_runner_invoked"] is False
        and int(row["expansion_budget_consumed"] or 0) == 0
        for row in reconciliation_observations
    )

    assert runner_targets == ["CCO", "CC=O"]
    assert campaign["accepted_expansion_count"] == 2
    assert campaign["attempt_run_count"] == 2
    assert campaign["expansion_run_count"] == 2
    exact_rows = (board.get("literature_evidence") or {}).get("exact_rows") or []
    assert len(exact_rows) == 1
    assert exact_rows[0]["reactant_smiles"] == ["CC=O"]
    assert exact_rows[0]["atom_mapped_reaction_smiles"] == (
        "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
    )
    assert any(
        record.get("candidate_id")
        == "source_detail_exact_step:acetaldehyde_to_ethanol"
        for proposal in (board.get("route_consensus") or {}).get("proposals") or []
        for record in proposal.get("source_records") or []
    )

    decisions = [
        json.loads(line)
        for line in (tmp_path / "decision_trace.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    resumes = [row for row in decisions if row.get("stage") == "codex_agent_team_resume"]
    assert len(resumes) == 1
    assert resumes[0]["prior_accepted_expansion_count"] == 1
    assert resumes[0]["accepted_expansion_count"] == 2
    assert resumes[0]["invocation_accepted_expansion_count"] == 1

    drain = json.loads(
        (tmp_path / "codex_campaign_drain.json").read_text(encoding="utf-8")
    )
    assert drain["invocation_count"] == 0
    assert drain["stop_reason"] == "awaiting_reaction_proof_materialization"
