from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cascade_planner.harness import agentic_blackboard_controller as controller_module
from cascade_planner.harness.agentic_blackboard_controller import (
    _portfolio_verifier_bundle,
    _refresh_multisource_route_consensus,
)
from cascade_planner.harness.tools import ToolExecutionState
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.application.frontier_ledger import exact_edge_signature
from cascade_planner.orchestration.codex_retrosynthesis import (
    RetrosynthesisTeamConfig,
)


_FIXTURES = Path(__file__).parent / "fixtures"
_SOURCE_PDF = _FIXTURES / "source_evidence_stub.pdf"
_SOURCE_PAGE = _FIXTURES / "source_page.ppm"
_SOURCE_MANIFEST = _FIXTURES / "source_evidence_manifest.json"
_TRUSTED_REGISTRY = _FIXTURES / "trusted_literature_step_registry.json"


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validated_exact_row() -> dict:
    template_id = "source_detail_exact_step:ethanol_hydration"
    return {
        "row_id": template_id,
        "step_id": "ethanol_hydration",
        "accepted": True,
        "product_smiles": "CCO",
        "reactant_smiles": ["CC", "O"],
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "source_ref": "doi:10.1000/revalidatable-stitch",
        "exact_step_validation": {
            "schema_version": "template_validation_report.v1",
            "accepted": True,
            "allowed_for_one_step_source": True,
            "source_template_id": template_id,
            "reasons": [],
        },
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "document_id": "fixture:revalidatable-stitch",
                "manifest_path": str(_SOURCE_MANIFEST.resolve()),
                "manifest_sha256": hashlib.sha256(
                    _SOURCE_MANIFEST.read_bytes()
                ).hexdigest(),
                "source_pdf_path": str(_SOURCE_PDF.resolve()),
                "source_pdf_sha256": hashlib.sha256(
                    _SOURCE_PDF.read_bytes()
                ).hexdigest(),
                "page_number": 1,
                "image_path": str(_SOURCE_PAGE.resolve()),
                "image_sha256": hashlib.sha256(
                    _SOURCE_PAGE.read_bytes()
                ).hexdigest(),
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }


def _chemenzy_proof_bank() -> dict:
    report = verify_chemenzy_raw_routes(
        {
            "target": "CCO",
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": ["CC", "O"],
                        "terminal_stock_status": {"CC": True, "O": True},
                    },
                    "steps": [
                        {
                            "index": 0,
                            "product": "CCO",
                            "reactant_smiles": ["CC", "O"],
                            "stock_status": {"CC": True, "O": True},
                        }
                    ],
                }
            ],
        },
        target_smiles="CCO",
        case_id="case",
    )
    assert report["accepted"] is True
    return {
        "artifact_ref": "guided_chemenzy_result.json",
        "route_proof_bank": report["route_proof_bank"],
    }


def _untrusted_caller_advisory_proposal() -> dict:
    return {
        "proposal_id": "caller-advisory-only",
        "target_smiles": "CCO",
        "precursor_smiles": ["C", "CO"],
        "proposal_label": "unbound advisory split",
        "proposal_type": "strategic",
        "source_type": "model",
        "confidence": "high",
        "evidence_level": "validated",
        "source_refs": [],
        "evidence_refs": [],
    }


def _refresh_with_real_external_admission(
    *,
    state: ToolExecutionState,
    blackboard: dict,
) -> dict:
    bindings = {
        "schema_version": "route_portfolio_bindings.v1",
        "stock_molecule_ids": [],
        "stock_bindings": {},
        "edge_proof_levels": {},
        "content_sha256": "b" * 64,
    }
    with (
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "verify_codex_consensus_graph",
            return_value={"edge_verifications": [], "content_sha256": "d" * 64},
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "derive_portfolio_bindings",
            return_value=bindings,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "solve_diverse_routes",
            return_value=SimpleNamespace(
                to_dict=lambda: {
                    "schema_version": "route_portfolio.v1",
                    "routes": [],
                    "content_sha256": "a" * 64,
                }
            ),
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "validate_portfolio_replacements",
            return_value={"schema_version": "route_replacement_catalog.v1"},
        ),
    ):
        return _refresh_multisource_route_consensus(
            state=state,
            blackboard=blackboard,
            codex_campaign_config=RetrosynthesisTeamConfig(max_depth=2),
        )


def _verifier(marker: str, target: str) -> dict:
    return {
        "schema_version": "harness_route_verifier_report.v1",
        "marker": marker,
        "target_equivalence_audit": {
            "request_canonical_isomeric_smiles": target,
        },
    }


def test_controller_collects_parent_and_child_verifiers_into_deduplicated_bundle() -> None:
    parent = _verifier("parent", "CCO")
    child = _verifier("child", "CC")

    bundle = _portfolio_verifier_bundle(
        artifacts={
            "guided_chemenzy": {"raw_route_verifier": parent},
            "route_expansion_subgoal_search": {
                "subgoals": [{"verifier": child}, {"verifier": child}]
            },
        },
        parent_proof={
            "proof_evidence": {"parent_verifier_attempt": parent},
        },
        solved_parent_verifier={},
    )

    assert bundle["schema_version"] == "route_verifier_bundle.v1"
    assert bundle["input_report_count"] == 4
    assert bundle["report_count"] == 2
    assert bundle["duplicate_report_count"] == 2
    payload = dict(bundle)
    content_sha256 = payload.pop("content_sha256")
    assert content_sha256 == _digest(payload)


def test_consensus_refresh_keeps_hashed_portfolio_and_runtime_bindings_as_siblings(
    tmp_path,
) -> None:
    portfolio = {
        "schema_version": "route_portfolio.v1",
        "routes": [],
        "content_sha256": "a" * 64,
    }
    bindings = {
        "schema_version": "route_portfolio_bindings.v1",
        "stock_molecule_ids": [],
        "stock_bindings": {},
        "edge_proof_levels": {},
        "content_sha256": "b" * 64,
    }
    replacement_catalog = {
        "schema_version": "route_replacement_catalog.v1",
        "candidates": [],
        "content_sha256": "c" * 64,
    }
    rebuild = {
        "accepted": True,
        "consensus": {"schema_version": "route_consensus.v1", "proposals": []},
        "graph": {
            "schema_version": "route_consensus_graph.v1",
            "v2_overlay": {
                "schema_version": "route_hypergraph_overlay.v2",
                "root_molecule_id": "target",
                "validation": {"valid": True, "errors": []},
                "molecules": [],
                "reaction_hyperedges": [],
            },
        },
    }
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )
    with (
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "rebuild_consensus_graph_from_blackboard",
            return_value=rebuild,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "derive_portfolio_bindings",
            return_value=bindings,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller.solve_diverse_routes",
            return_value=SimpleNamespace(to_dict=lambda: dict(portfolio)),
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "validate_portfolio_replacements",
            return_value=replacement_catalog,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "consensus_to_blackboard_proposals",
            return_value=[],
        ),
    ):
        board = _refresh_multisource_route_consensus(
            state=state,
            blackboard={
                "schema_version": "agent_blackboard.v1",
                "case_id": "case",
                "target_profile": {"target_smiles": "CCO"},
            },
        )

    graph = board["route_consensus_graph"]
    assert graph["route_portfolio"] == portfolio
    assert "bindings" not in graph["route_portfolio"]
    assert graph["route_portfolio_bindings"] == bindings
    assert graph["route_replacement_catalog"] == replacement_catalog


def test_consensus_refresh_reconciles_campaign_proofs_without_spending_budget(
    tmp_path,
) -> None:
    graph = {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "case",
        "target_smiles": "CCO",
        "steps": [],
        "v2_overlay": {
            "schema_version": "route_hypergraph_overlay.v2",
            "root_molecule_id": "target",
            "validation": {"valid": True, "errors": []},
            "molecules": [],
            "reaction_hyperedges": [],
        },
    }
    rebuild = {
        "accepted": True,
        "consensus": {"schema_version": "route_consensus.v1", "proposals": []},
        "graph": {**graph, "projection_marker": "caller_advisory"},
        "admission_receipts": {
            "edge:sha256:" + "e" * 64: [
                {"schema_version": "fixture_receipt_material.v1"}
            ]
        },
    }
    canonical_graph = {**graph, "projection_marker": "canonical_durable_union"}
    bindings = {
        "schema_version": "route_portfolio_bindings.v1",
        "stock_molecule_ids": [],
        "stock_bindings": {},
        "edge_proof_levels": {},
        "content_sha256": "b" * 64,
    }
    reconciliation = {
        "schema_version": "codex_campaign_proof_reconciliation.v1",
        "accepted": True,
        "graph_complete": True,
        "reaction_proof_state": {"records": []},
        "reaction_proof_state_ref": str(tmp_path / "reaction_proof_state.json"),
        "open_reaction_proofs": [],
        "frontier_completeness": {"complete": True},
        "proposal_runner_invoked": False,
        "expansion_budget_consumed": 0,
        "canonical_route_consensus_graph": canonical_graph,
    }
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )
    projected_graphs: list[dict] = []
    original_project_frontier_ledger = controller_module.project_frontier_ledger

    def observed_project_frontier_ledger(route_graph, *args, **kwargs):
        projected_graphs.append(dict(route_graph))
        return original_project_frontier_ledger(route_graph, *args, **kwargs)

    with (
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "rebuild_consensus_graph_from_blackboard",
            return_value=rebuild,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "verify_codex_consensus_graph",
            return_value={"edge_verifications": [], "content_sha256": "d" * 64},
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "derive_portfolio_bindings",
            return_value=bindings,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller.solve_diverse_routes",
            return_value=SimpleNamespace(
                to_dict=lambda: {
                    "schema_version": "route_portfolio.v1",
                    "routes": [],
                    "content_sha256": "a" * 64,
                }
            ),
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "validate_portfolio_replacements",
            return_value={"schema_version": "route_replacement_catalog.v1"},
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "consensus_to_blackboard_proposals",
            return_value=[],
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "reconcile_codex_campaign_proof_state",
            return_value=reconciliation,
        ) as reconcile,
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "project_frontier_ledger",
            side_effect=observed_project_frontier_ledger,
        ),
    ):
        board = _refresh_multisource_route_consensus(
            state=state,
            blackboard={
                "schema_version": "agent_blackboard.v1",
                "case_id": "case",
                "target_profile": {"target_smiles": "CCO"},
                "codex_agent_team": {
                    "accepted": True,
                    "campaign": {
                        "accepted_expansion_count": 24,
                        "max_expansions": 24,
                    },
                },
            },
        )

    reconcile.assert_called_once()
    assert reconcile.call_args.kwargs[
        "external_hyperedge_admission_receipts"
    ] == rebuild["admission_receipts"]
    assert board["route_consensus_graph"]["projection_marker"] == (
        "caller_advisory"
    )
    assert board["caller_advisory_route_consensus_graph"][
        "projection_marker"
    ] == "caller_advisory"
    assert board["canonical_route_consensus_graph"]["projection_marker"] == (
        "canonical_durable_union"
    )
    assert state.artifacts["canonical_route_consensus_graph"] == (
        board["canonical_route_consensus_graph"]
    )
    assert projected_graphs[-1]["projection_marker"] == (
        "canonical_durable_union"
    )
    campaign = board["codex_agent_team"]["campaign"]
    assert campaign["accepted_expansion_count"] == 24
    # The reconciliation's legacy boolean is retained for diagnostics, but an
    # unbound proof state/queue cannot authorize complete-graph closure.
    assert campaign["reconciliation_graph_complete"] is True
    assert campaign["graph_complete"] is False
    assert board["codex_agent_team"]["proof_closed"] is False
    assert board["frontier_ledger_summary"]["input_valid"] is False
    assert board["frontier_ledger_summary"]["all_explored_graph_closed"] is False
    assert state.artifacts["codex_campaign_proof_reconciliation"][
        "proposal_runner_invoked"
    ] is False
    assert state.artifacts["codex_campaign_proof_reconciliation"][
        "expansion_budget_consumed"
    ] == 0


def test_consensus_refresh_ledger_projects_all_steps_not_bounded_route_hypotheses(
    tmp_path,
) -> None:
    steps = [
        {
            "schema_version": "route_consensus_step.v1",
            "step_id": f"step:{index}",
            "signature": f"CCO<-{'C' * index}",
            "product_smiles": "CCO",
            "precursor_smiles": ["C" * index],
            "product_node_id": "target",
            "precursor_node_ids": [f"leaf:{index}"],
            "proposal_ids": [f"proposal:{index}"],
            "source_refs": [],
            "evidence_refs": [],
        }
        for index in range(1, 31)
    ]
    graph = {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "case",
        "target_smiles": "CCO",
        "nodes": [],
        "steps": steps,
        "route_hypotheses": [
            {"retrosynthetic_step_ids": [f"step:{index}"]}
            for index in range(1, 25)
        ],
        "truncation": {"route_hypotheses_truncated": True},
        "v2_overlay": {
            "schema_version": "route_hypergraph_overlay.v2",
            "root_molecule_id": "target",
            "validation": {"valid": True, "errors": []},
            "molecules": [],
            "reaction_hyperedges": [],
        },
    }
    queue = {
        "schema_version": "frontier_queue.v1",
        "run_id": "case",
        "revision": 1,
        "jobs": [],
    }
    queue["content_sha256"] = _digest(queue)
    graph_identity = {
        "schema_version": graph["schema_version"],
        "case_id": graph["case_id"],
        "target_smiles": graph["target_smiles"],
        "steps": sorted(
            [
                {
                    "step_id": step["step_id"],
                    "signature": step["signature"],
                    "product_smiles": step["product_smiles"],
                    "precursor_smiles": sorted(step["precursor_smiles"]),
                }
                for step in steps
            ],
            key=lambda row: (row["step_id"], row["signature"]),
        ),
    }
    proof_state = {
        "schema_version": "codex_retrosynthesis_reaction_proof_state.v1",
        "graph_identity_sha256": _digest(graph_identity),
        "records": [],
    }
    proof_state["content_sha256"] = _digest(proof_state)
    reconciliation = {
        "schema_version": "codex_campaign_proof_reconciliation.v1",
        "accepted": True,
        # Deliberately stale/incorrect: the ledger fixed point must win.
        "graph_complete": True,
        "frontier_queue": queue,
        "reaction_proof_state": proof_state,
        "open_reaction_proofs": [],
        "frontier_completeness": {"complete": True},
        "proposal_runner_invoked": False,
        "expansion_budget_consumed": 0,
        "canonical_route_consensus_graph": graph,
    }
    rebuild = {
        "accepted": True,
        "consensus": {"schema_version": "route_consensus.v1", "proposals": []},
        "graph": graph,
    }
    bindings = {
        "schema_version": "route_portfolio_bindings.v1",
        "stock_molecule_ids": [],
        "stock_bindings": {},
        "edge_proof_levels": {},
        "content_sha256": "b" * 64,
    }
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )

    with (
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "rebuild_consensus_graph_from_blackboard",
            return_value=rebuild,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "verify_codex_consensus_graph",
            return_value={"edge_verifications": [], "content_sha256": "d" * 64},
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "derive_portfolio_bindings",
            return_value=bindings,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller.solve_diverse_routes",
            return_value=SimpleNamespace(
                to_dict=lambda: {
                    "schema_version": "route_portfolio.v1",
                    "routes": [],
                    "content_sha256": "a" * 64,
                }
            ),
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "validate_portfolio_replacements",
            return_value={"schema_version": "route_replacement_catalog.v1"},
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "consensus_to_blackboard_proposals",
            return_value=[],
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "reconcile_codex_campaign_proof_state",
            return_value=reconciliation,
        ),
    ):
        board = _refresh_multisource_route_consensus(
            state=state,
            blackboard={
                "schema_version": "agent_blackboard.v1",
                "case_id": "case",
                "target_profile": {"target_smiles": "CCO"},
                "codex_agent_team": {
                    "accepted": True,
                    "campaign": {
                        "accepted_expansion_count": 1,
                        "max_expansions": 40,
                    },
                },
            },
        )

    ledger_path = tmp_path / "frontier_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    summary = board["frontier_ledger_summary"]
    assert len(graph["route_hypotheses"]) == 24
    assert ledger["summary"]["reachable_edge_count"] == 30
    assert summary["summary"]["reaction_proof_pending_edge_count"] == 30
    # All 30 unexpanded graph frontiers remain visible, while the independent
    # scheduler-eligibility count correctly says the empty durable queue may
    # not launch a proposal worker for any of them.
    assert summary["summary"]["proposal_pending_molecule_count"] == 30
    assert summary["summary"]["proposal_expansion_eligible_molecule_count"] == 0
    assert summary["summary"]["stock_pending_leaf_count"] == 30
    assert summary["summary"]["dependency_pending_edge_count"] == 30
    assert summary["input_valid"] is True
    assert summary["content_sha256"] == ledger["content_sha256"]
    assert summary["frontier_ledger_content_sha256"] == ledger["content_sha256"]
    assert board["frontier_ledger"] == ledger
    assert board["artifact_refs"]["frontier_ledger"] == str(ledger_path)
    assert "frontier_ledger_summary" not in board["route_consensus_graph"]
    team = board["codex_agent_team"]
    assert team["frontier_ledger_ref"] == str(ledger_path)
    assert team["frontier_ledger_summary"] == summary
    assert team["campaign"]["reconciliation_graph_complete"] is True
    assert team["campaign"]["graph_complete"] is False
    assert team["proof_closed"] is False


def test_rejected_codex_team_still_reconciles_valid_exact_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY.resolve()),
    )
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )
    board = _refresh_with_real_external_admission(
        state=state,
        blackboard={
            "schema_version": "agent_blackboard.v1",
            "case_id": "case",
            "target_profile": {"target_smiles": "CCO"},
            "literature_evidence": {"exact_rows": [_validated_exact_row()]},
            "retrosynthetic_proposals": [
                _untrusted_caller_advisory_proposal()
            ],
            "codex_agent_team": {
                "accepted": False,
                "reasons": ["required_child_reports_not_valid"],
            },
        },
    )

    reconciliation = state.artifacts["codex_campaign_proof_reconciliation"]
    canonical = board["canonical_route_consensus_graph"]
    caller = board["caller_advisory_route_consensus_graph"]
    assert reconciliation["accepted"] is True
    assert reconciliation["proposal_runner_invoked"] is False
    assert reconciliation["expansion_budget_consumed"] == 0
    assert len(canonical["steps"]) == 1
    assert canonical["steps"][0]["precursor_smiles"] == ["CC", "O"]
    assert len(caller["steps"]) == 2
    assert not any(
        step["precursor_smiles"] == ["C", "CO"]
        for step in canonical["steps"]
    )
    journal = reconciliation["admitted_hyperedge_journal"]
    assert journal["new_event_count"] == 1
    assert journal["quarantined_edge_count"] == 1
    assert board["codex_agent_team"]["accepted"] is False
    projection = board["codex_campaign_authority_projection"]
    assert projection["reconciliation_trigger"] == (
        "host_external_admission_material"
    )
    assert projection["codex_team_present"] is True
    assert projection["codex_team_accepted"] is False


def test_missing_codex_team_still_reconciles_current_host_chemenzy_bank(
    tmp_path,
) -> None:
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )
    board = _refresh_with_real_external_admission(
        state=state,
        blackboard={
            "schema_version": "agent_blackboard.v1",
            "case_id": "case",
            "target_profile": {"target_smiles": "CCO"},
            "chemenzy_route_proof_banks": [_chemenzy_proof_bank()],
            "retrosynthetic_proposals": [
                _untrusted_caller_advisory_proposal()
            ],
        },
    )

    reconciliation = state.artifacts["codex_campaign_proof_reconciliation"]
    canonical = board["canonical_route_consensus_graph"]
    caller = board["caller_advisory_route_consensus_graph"]
    assert reconciliation["accepted"] is True
    assert reconciliation["proposal_runner_invoked"] is False
    assert reconciliation["expansion_budget_consumed"] == 0
    assert len(canonical["steps"]) == 1
    assert canonical["steps"][0]["precursor_smiles"] == ["CC", "O"]
    assert len(caller["steps"]) == 2
    assert not any(
        step["precursor_smiles"] == ["C", "CO"]
        for step in canonical["steps"]
    )
    journal = reconciliation["admitted_hyperedge_journal"]
    assert journal["new_event_count"] == 1
    assert journal["quarantined_edge_count"] == 1
    event_path = Path(journal["event_refs"][0])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["provenance_receipt"]["source_kind"] == (
        "current_host_replayed_chemenzy_bank"
    )
    projection = board["codex_campaign_authority_projection"]
    assert projection["reconciliation_trigger"] == (
        "host_external_admission_material"
    )
    assert projection["codex_team_present"] is False
    assert projection["codex_team_accepted"] is False
    assert len(projection["campaign_identity_sha256"]) == 64
    assert len(projection["campaign_policy_sha256"]) == 64
    authority_root = tmp_path / "codex_retrosynthesis_team"
    assert (authority_root / "campaign_identity.json").is_file()
    assert (authority_root / "campaign_policy.json").is_file()
    assert (authority_root / ".campaign-authority.lock").is_file()
    assert (authority_root / "frontier_ledger.json").is_file()
    assert list((authority_root / "campaign_commits").glob("*.json")) == []


def test_invalid_external_material_cannot_enter_canonical_or_ledger(
    tmp_path,
) -> None:
    input_board = {
        "schema_version": "agent_blackboard.v1",
        "case_id": "case",
        "target_profile": {"target_smiles": "CCO"},
        "retrosynthetic_proposals": [_untrusted_caller_advisory_proposal()],
    }
    rebuild = controller_module.rebuild_consensus_graph_from_blackboard(
        input_board,
        max_depth=2,
    )
    edge = rebuild["graph"]["steps"][0]
    edge_key = exact_edge_signature(
        edge["product_smiles"],
        edge["precursor_smiles"],
    )
    rebuild["admission_receipts"] = {
        edge_key: [{"schema_version": "forged_admission_material.v1"}]
    }
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )
    with patch(
        "cascade_planner.harness.agentic_blackboard_controller."
        "rebuild_consensus_graph_from_blackboard",
        return_value=rebuild,
    ):
        board = _refresh_with_real_external_admission(
            state=state,
            blackboard=input_board,
        )

    reconciliation = state.artifacts["codex_campaign_proof_reconciliation"]
    assert reconciliation["accepted"] is True
    assert reconciliation["admitted_hyperedge_journal"][
        "new_event_count"
    ] == 0
    assert reconciliation["admitted_hyperedge_journal"][
        "quarantined_edge_count"
    ] == 1
    assert board["caller_advisory_route_consensus_graph"]["steps"]
    assert board["canonical_route_consensus_graph"]["steps"] == []
    assert board["frontier_ledger"]["edges"] == {}


def test_reconciliation_error_fails_closed_without_caller_ledger_authority(
    tmp_path,
) -> None:
    input_board = {
        "schema_version": "agent_blackboard.v1",
        "case_id": "case",
        "target_profile": {"target_smiles": "CCO"},
        "retrosynthetic_proposals": [_untrusted_caller_advisory_proposal()],
    }
    rebuild = controller_module.rebuild_consensus_graph_from_blackboard(
        input_board,
        max_depth=2,
    )
    rebuild["admission_receipts"] = {
        "edge:sha256:" + "e" * 64: [
            {"schema_version": "forged_admission_material.v1"}
        ]
    }
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "case"},
    )
    with (
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "rebuild_consensus_graph_from_blackboard",
            return_value=rebuild,
        ),
        patch(
            "cascade_planner.harness.agentic_blackboard_controller."
            "reconcile_codex_campaign_proof_state",
            side_effect=ValueError("forced reconciliation failure"),
        ),
    ):
        board = _refresh_with_real_external_admission(
            state=state,
            blackboard=input_board,
        )

    reconciliation = state.artifacts["codex_campaign_proof_reconciliation"]
    assert reconciliation["accepted"] is False
    assert board["caller_advisory_route_consensus_graph"]["steps"]
    assert board["frontier_ledger"]["edges"] == {}
    assert board["frontier_ledger_summary"]["input_valid"] is False
    assert "canonical_route_consensus_graph" not in board
    assert "codex_campaign_authority_projection" not in board
