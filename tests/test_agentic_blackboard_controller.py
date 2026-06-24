import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from cascade_planner.agent.codex_worker import _task_allows_cli_search
from cascade_planner.agent.chem_enzy_policy import apply_chem_enzy_search_policy, validate_chem_enzy_search_policy
from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.harness.analogical_reaction_templates import (
    apply_analogical_templates_to_target,
    extract_analogical_reaction_templates_from_blackboard,
    rank_analogical_reaction_templates_from_blackboard,
    validate_analogical_reaction_template,
)
from cascade_planner.harness.agent_action_planner import plan_action_batch, validate_action_batch
from cascade_planner.harness.agentic_blackboard import (
    build_agentic_guided_payload,
    initialize_agent_blackboard,
    update_blackboard_from_action_batch,
    update_blackboard_from_action,
    update_budget_for_action,
)
from cascade_planner.harness.agentic_blackboard_controller import (
    _capability_check_planner_history,
    _capability_check_source_acquisition,
    _codex_literature_scout_task,
    _inject_pdf_defaults,
    _local_pdf_cache_match_report,
    _validate_agentic_final_verdict,
    emit_agentic_final_verdict,
    run_agentic_blackboard_controller,
)
from cascade_planner.harness.hypothetical_retrosynthesis_report import (
    compile_hypothesis_only_retrosynthesis_report,
)
from cascade_planner.harness.hypothesis_execution_report import (
    compile_hypothesis_execution_report,
)
from cascade_planner.harness.codex_action_planner import (
    _codex_action_planner_task,
    _planner_context_summary,
    _write_codex_blackboard_snapshot,
    plan_action_batch_with_codex,
)
from cascade_planner.harness.local_pdf_proxy import (
    load_pdf_requests,
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_request_queue_path,
)
from cascade_planner.harness.open_research_experience import audit_local_pdf_proxy_fallback
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.schemas import TargetInput
from cascade_planner.harness.target_side_strategy import build_target_side_disconnection_hypotheses
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    _pdf_evidence_from_payload_or_artifacts,
    _route_expansion_child_targets,
    _visual_chain_image_paths,
    execute_local_tool,
    run_guided_chemenzy_rerun,
)
from cascade_planner.harness.visual_literature_chain_agent import (
    _bufotalin_tet2025_label_anchor_chain,
    _candidate_chain_from_parsed,
    _candidate_quality,
    _run_codex_visual_prompt,
    _run_direct_visual_prompt,
)
from cascade_planner.harness.visual_structure_extraction import validate_visual_structure_chain


MLA_LIKE_SMILES = "CN1CC2CCC1CC2OC(=O)c3ccccc3N4C(=O)CCC4=O"
BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H]"
    "(CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
BUFOTALIN_ACHIRAL_SMILES = "CC(=O)OC1CC2(O)C3CCC4CC(O)CCC4(C)C3CCC2(C)C1c1ccc(=O)oc1"
C22_9OH_4HP_SMILES = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"


def _test_search_payload(query: str = "target proximal synthesis", **overrides):
    payload = {
        "schema_version": "agentic_literature_search_payload.v1",
        "search_intent": "target_proximal_source_discovery",
        "query": query,
        "queries": [query],
        "search_queries": [query],
        "max_sources": 3,
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
    }
    payload.update(overrides)
    return payload


def _test_analogical_template_payload(action_type: str = "extract_analogical_reaction_templates", **overrides):
    payload = {
        "max_templates": 4,
        "max_applications": 3,
        "template_radius_policy": "auto",
        "analog_template_confidence_threshold": "low",
        "analogical_template_policy": {
            "schema_version": "agentic_analogical_template_action_policy.v1",
            "action_type": action_type,
            "analogy_is_advisory_only": True,
            "no_solved_claim": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "production_write_blocked": True,
            "raw_reaction_output_allowed": False,
            "final_verdict_authority": "deterministic_parent_route_proof",
            "allowed_use": ["planner_priority", "guided_policy_hint", "template_candidate_validation"],
            "deterministic_template_validation_required": True,
        },
    }
    payload.update(overrides)
    return payload


def _test_search_requirements():
    return {
        "search_literature": {
            "currently_required_when_selected": True,
            "accepted_payload_fields": ["search_intent", "query", "queries", "search_queries", "source_acquisition_policy"],
        }
    }


def _test_source_sensitive_requirements():
    return {
        action_type: {
            "currently_required": False,
            "accepted_payload_fields": ["source_ref", "task_id", "label"],
            "binding_candidates": [],
        }
        for action_type in (
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "resolve_literature_structure_task",
            "compile_exact_literature_rows",
        )
    }


def _test_analogical_template_requirements():
    return {
        action_type: {
            "currently_required_when_selected": True,
            "accepted_payload_fields": [
                "analogical_template_policy",
                "max_templates",
                "max_applications",
                "template_radius_policy",
                "analog_template_confidence_threshold",
            ],
        }
        for action_type in (
            "extract_analogical_reaction_templates",
            "rank_analogical_reaction_templates",
            "apply_analogical_template_to_target",
            "validate_template_application",
        )
    }


class AgenticBlackboardControllerTest(unittest.TestCase):
    def test_blackboard_initialization_writes_target_profile(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)

        self.assertEqual(board["schema_version"], "agent_blackboard.v1")
        self.assertEqual(board["target_profile"]["target_smiles"], "CCO")
        self.assertEqual(board["budget_state"]["max_rounds"], 3)
        self.assertEqual(board["planner_history"], [])
        self.assertEqual(board["budget_state"]["codex_action_planner_runs"], 0)

    def test_hypothesis_only_report_emits_achiral_connectivity_candidates(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["template_applications"] = [
            {
                "application_id": "apply:analog_template:test",
                "template_id": "analog_template:test",
                "evidence_refs": ["local_pdf:test"],
                "hypothetical_route_hypothesis": {
                    "reaction_center_idea": "late same-core alcohol protection or redox adjustment",
                    "template_application": "search same-core protected alcohol precursor",
                    "risk_flags": ["broad_template_scope", "selectivity_not_proven"],
                },
                "hypothetical_precursor_hints": [
                    {
                        "target_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "precursor_smiles": "CC(=O)OC1CC2(O)C3CCC4CC(O)CCC4(C)C3CCC2(C)C1C1=CC(=O)OC=C1",
                        "precursor_role": "same_core_enone_or_protected_alcohol_precursor",
                        "derived_from_retron": "steroid_alcohol_protection_redox_adjustment",
                        "risk_flags": ["hypothesis_only_not_literature_exact"],
                    }
                ],
            }
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "artifact_ref": "/tmp/visual.json",
                "source_ref": "local_pdf:test",
                "exploratory_accepted": True,
                "steps": [
                    {
                        "product_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "main_reactant_smiles": "CC12CCC(=O)C=C1CCC3C2CCC4(C3CCC4(C(=O)CO)O)C",
                        "reactant_labels": ["prednisone"],
                        "confidence": "low",
                        "stereochemistry_status": "unspecified_or_partial",
                        "risk_flags": ["visual_connectivity_approximation"],
                        "source_locator": "page 2 scheme",
                    }
                ],
            }
        ]

        report = compile_hypothesis_only_retrosynthesis_report(blackboard=board)

        self.assertTrue(report["accepted"])
        self.assertFalse(report["solved"])
        self.assertTrue(report["no_solved_claim"])
        self.assertEqual(report["final_verdict_authority"], "none")
        self.assertGreaterEqual(report["candidate_precursor_count"], 2)
        self.assertTrue(report["stereochemistry_policy"]["achiral_connectivity_candidates_allowed"])
        self.assertTrue(
            any(row["source_type"] == "visual_connectivity_candidate" for row in report["candidate_precursors"])
        )
        self.assertTrue(
            all(row["allowed_use"] == "guided_search_seed_only" for row in report["candidate_precursors"])
        )
        artifact = {
            "schema_version": "hypothesis_only_retrosynthesis_report_artifact.v1",
            "artifact_type": "HypothesisOnlyRetrosynthesisReport",
            "artifact_id": "target1_steroid:hypothesis_only_retrosynthesis_report",
            "case_id": "target1_steroid",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["local_pdf:test"],
            "validation_status": "accepted",
            "payload": report,
        }
        validation = validate_typed_artifact(artifact)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_exhaustive_policy_stops_instead_of_repeating_stale_failure_critic(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=60)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "existing"}]}
        board["route_failures"] = [{"schema_version": "agent_route_failure.v1", "reason": "large_atom_jump"}]
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "known_source", "doi": "10.0000/example", "title": "known source"}
        ]
        board["current_belief"]["next_action_bias"] = []
        board["current_belief"]["template_policy"]["enabled"] = False
        board["analogical_hypothesis_ranking"] = {"ranked_hypotheses": []}
        board["action_history"] = [
            {"round_index": 1, "action_type": "run_guided_chemenzy", "useful_artifact": True, "reasons": ["large_atom_jump"]},
            {"round_index": 2, "action_type": "build_failure_critic_report", "useful_artifact": True, "reasons": []},
            {"round_index": 2, "action_type": "stitch_parent_route", "useful_artifact": True, "reasons": ["parent_route_verifier_not_accepted"]},
        ]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)

        self.assertEqual([row["action_type"] for row in batch["actions"]], ["stop_unresolved"])
        self.assertIn("no non-stale action", batch["actions"][0]["rationale"])

    def test_hypothesis_report_relaxes_final_verdict_without_solved_claim(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["parent_route_proof"] = {
            "accepted": False,
            "solved": False,
            "route_status": "partial_anchor_only_not_solved",
            "reasons": ["parent_route_verifier_not_accepted"],
        }
        report = {
            "schema_version": "hypothesis_only_retrosynthesis_report.v1",
            "accepted": True,
            "candidate_precursor_count": 1,
            "no_solved_claim": True,
        }
        final = emit_agentic_final_verdict(
            blackboard=board,
            artifacts={
                "hypothesis_only_retrosynthesis_report": {
                    "artifact_type": "HypothesisOnlyRetrosynthesisReport",
                    "payload": report,
                }
            },
            bundle={"case_id": "target1_steroid"},
        ).to_dict()

        self.assertEqual(final["verdict"], "hypothesis_route_proposed")
        self.assertEqual(final["route_status"], "hypothesis_route_proposed")
        self.assertFalse(final["solved"])
        self.assertIn("hypothesis_only_retrosynthesis_available", final["reasons"])
        validation = _validate_agentic_final_verdict(final, blackboard=board, validations=[])
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_hypothesis_execution_report_tracks_rejected_route_expansion(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        hypothesis_report = {
            "schema_version": "hypothesis_only_retrosynthesis_report_artifact.v1",
            "artifact_type": "HypothesisOnlyRetrosynthesisReport",
            "artifact_id": "target1_steroid:hypothesis_only_retrosynthesis_report",
            "case_id": "target1_steroid",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["local_pdf:test"],
            "validation_status": "accepted",
            "payload": {
                "schema_version": "hypothesis_only_retrosynthesis_report.v1",
                "accepted": True,
                "solved": False,
                "no_solved_claim": True,
                "candidate_precursor_count": 1,
                "candidate_precursors": [
                    {
                        "schema_version": "hypothesis_precursor_candidate.v1",
                        "candidate_id": "hypothesis:protected_alcohol",
                        "precursor_role": "same_core_protected_alcohol",
                        "precursor_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "allowed_use": "guided_search_seed_only",
                    }
                ],
            },
        }
        route_expansion = {
            "schema_version": "route_expansion_subgoal_search_result.v1",
            "result": {
                "subgoal_count": 1,
                "accepted_subgoal_count": 0,
                "rejected_subgoal_count": 1,
                "subgoals": [
                    {
                        "subgoal": {
                            "name": "same_core_protected_alcohol",
                            "smiles": BUFOTALIN_ACHIRAL_SMILES,
                        },
                        "route_count": 12,
                        "accepted": False,
                        "solved": False,
                        "verifier": {
                            "accepted": False,
                            "route_status": "fake_closed_rejected",
                            "reasons": ["large_atom_jump", "no_verifier_accepted_stock_closed_route"],
                            "accepted_route_count": 0,
                            "rejected_route_count": 12,
                        },
                    }
                ],
            },
        }

        payload = compile_hypothesis_execution_report(
            blackboard=board,
            hypothesis_report=hypothesis_report,
            route_expansion_results=[route_expansion],
        )

        self.assertEqual(payload["route_status"], "hypothesis_routes_executed_rejected")
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["executed_candidate_count"], 1)
        self.assertEqual(payload["rejected_candidate_count"], 1)
        self.assertEqual(payload["pending_candidate_count"], 0)
        row = payload["candidate_executions"][0]
        self.assertEqual(row["execution_status"], "executed_rejected")
        self.assertEqual(row["route_count"], 12)
        self.assertIn("large_atom_jump", row["reasons"])
        artifact = {
            "schema_version": "hypothesis_execution_report_artifact.v1",
            "artifact_type": "HypothesisExecutionReport",
            "artifact_id": "target1_steroid:hypothesis_execution_report",
            "case_id": "target1_steroid",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["route_expansion_subgoal_search_result.json"],
            "validation_status": "accepted",
            "payload": payload,
        }
        validation = validate_typed_artifact(artifact)
        self.assertTrue(validation["accepted"], validation["reasons"])

        final = emit_agentic_final_verdict(
            blackboard=board,
            artifacts={
                "hypothesis_only_retrosynthesis_report": hypothesis_report,
                "hypothesis_execution_report": artifact,
            },
            bundle={"case_id": "target1_steroid"},
        ).to_dict()
        self.assertEqual(final["verdict"], "hypothesis_route_proposed")
        self.assertEqual(final["route_status"], "hypothesis_routes_executed_rejected")
        self.assertFalse(final["solved"])

    def test_rejected_hypothesis_subgoal_creates_recursive_followup_tasks(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=6)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "expand:hypothesis",
            "action_type": "expand_child_target",
            "rationale": "execute first-level hypothesis precursor",
            "expected_artifact": "route_expansion_subgoal_search_result.v1",
            "success_condition": "verifier result is recorded",
            "payload": {
                "subgoal_targets": [
                    {
                        "name": "same_core_alcohol_precursor",
                        "smiles": "CCO",
                        "source": "analogical_hypothesis_precursor_hint",
                        "hypothesis_only_not_solved": True,
                    }
                ]
            },
        }
        action_result = {
            "accepted": True,
            "result": {
                "schema_version": "route_expansion_subgoal_search_result.v1",
                "accepted": False,
                "solved": False,
                "subgoal_count": 1,
                "accepted_subgoal_count": 0,
                "rejected_subgoal_count": 1,
                "reasons": ["no_route_expansion_subgoal_verified_solved"],
                "subgoals": [
                    {
                        "accepted": False,
                        "solved": False,
                        "route_count": 3,
                        "subgoal": {
                            "name": "same_core_alcohol_precursor",
                            "smiles": "CCO",
                            "source": "analogical_hypothesis_precursor_hint",
                            "hypothesis_only_not_solved": True,
                            "policy": {
                                "compiler_metadata": {
                                    "hypothesis_only_not_solved": True,
                                    "no_solved_claim": True,
                                }
                            },
                        },
                        "verifier": {
                            "accepted": False,
                            "route_status": "fake_closed_rejected",
                            "reasons": ["large_atom_jump"],
                        },
                    }
                ],
            },
            "reasons": ["no_route_expansion_subgoal_verified_solved"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action=action,
                action_result=action_result,
                round_index=2,
                run_dir=tmp,
            )

        tasks = board["recursive_hypothesis_tasks"]
        self.assertGreaterEqual(len(tasks), 1)
        self.assertTrue(all(row["schema_version"] == "recursive_hypothesis_task.v1" for row in tasks))
        self.assertTrue(all(row["recursive_depth"] == 1 for row in tasks))
        self.assertTrue(all(row["no_solved_claim"] for row in tasks))
        self.assertTrue(all(row["precursor_smiles"] != "CCO" for row in tasks))
        self.assertIn("expand_child_target", board["current_belief"]["next_action_bias"])
        self.assertFalse(board["current_belief"]["child_route_solved"])

    def test_planner_expands_recursive_hypothesis_tasks(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_route_expansion_subgoal_runs": 4},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "doi:10.0000/example", "doi": "10.0000/example", "title": "Example source"}
        ]
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:test",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "rejected_hypothesis_precursor",
                "parent_smiles": "CCO",
                "precursor_smiles": "CC=O",
                "name": "recursive_primary_alcohol_to_aldehyde_precursor",
                "recursive_depth": 1,
                "operation_idea": "continue through aldehyde-level oxidation state",
                "variant_type": "primary_alcohol_to_aldehyde_precursor",
                "failure_reasons": ["large_atom_jump"],
                "allowed_use": "route_expansion_subgoal_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "child_route_cannot_promote_parent": True,
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        expand = [row for row in batch["actions"] if row["action_type"] == "expand_child_target"]

        self.assertTrue(expand)
        target_payload = expand[0]["payload"]["subgoal_targets"][0]
        self.assertEqual(target_payload["smiles"], "CC=O")
        self.assertEqual(target_payload["source"], "recursive_hypothesis_task")
        self.assertEqual(target_payload["recursive_depth"], 1)
        policy = target_payload["chem_enzy_search_policy"]
        self.assertTrue(policy["source_budget"]["recursive_hypothesis_frontier"])
        self.assertIn("recursive_failed_hypothesis_frontier_expansion", policy["source_budget"]["preferred_reaction_classes"])

    def test_invalid_input_still_emits_agentic_closing_audit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="invalid_case",
                target_smiles="not_a_smiles",
                output_dir=tmp,
                max_rounds=3,
            )

        self.assertEqual(result["final_verdict"]["verdict"], "invalid_input")
        self.assertFalse(result["preflight"]["accepted"])
        self.assertEqual(result["action_batches"], [])
        artifacts = result["artifact_bundle"]["artifacts"]
        self.assertEqual(artifacts["agent_blackboard_snapshot"]["artifact_type"], "AgentBlackboardSnapshot")
        self.assertEqual(artifacts["agentic_capability_audit"]["artifact_type"], "AgenticCapabilityAudit")
        self.assertEqual(artifacts["agentic_final_verdict_validation"]["artifact_type"], "AgenticFinalVerdictValidation")
        self.assertEqual(artifacts["hypothesis_only_retrosynthesis_report"]["artifact_type"], "HypothesisOnlyRetrosynthesisReport")
        self.assertEqual(artifacts["agentic_run_audit"]["artifact_type"], "AgenticRunAudit")
        refs = result["final_verdict"]["artifact_refs"]
        self.assertIn("agent_blackboard_snapshot", refs)
        self.assertIn("agentic_capability_audit", refs)
        self.assertIn("agentic_final_verdict_validation", refs)
        self.assertIn("hypothesis_only_retrosynthesis_report", refs)
        self.assertIn("agentic_run_audit", refs)
        capability_payload = artifacts["agentic_capability_audit"]["payload"]
        self.assertTrue(capability_payload["accepted"], capability_payload["failed_requirements"])
        capability_checks = {
            row["requirement_id"]: row
            for row in capability_payload["requirement_checks"]
        }
        self.assertTrue(
            capability_checks["artifact_refs_and_typed_validation_integrity"]["accepted"],
            capability_checks["artifact_refs_and_typed_validation_integrity"]["reasons"],
        )
        preflight_statuses = {
            row["requirement_id"]: row["status"]
            for row in capability_payload["requirement_checks"]
            if row["requirement_id"] in {
                "policy_driven_typed_action_batches",
                "deterministic_action_batch_validation_gate",
                "planner_decision_history_audited",
            }
        }
        self.assertEqual(preflight_statuses["policy_driven_typed_action_batches"], "preflight_rejected")
        self.assertEqual(preflight_statuses["deterministic_action_batch_validation_gate"], "preflight_rejected")
        self.assertEqual(preflight_statuses["planner_decision_history_audited"], "preflight_rejected")
        validation_keys = {
            row.get("artifact_key")
            for row in result["artifact_bundle"]["validations"]
            if row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
            and row.get("accepted")
        }
        self.assertIn("agent_blackboard_snapshot", validation_keys)
        self.assertIn("agentic_capability_audit", validation_keys)
        self.assertIn("agentic_final_verdict_validation", validation_keys)
        self.assertIn("agentic_run_audit", validation_keys)

    def test_action_batch_validation_rejects_unknown_solved_and_raw_payloads(self):
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "bad",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "bad:1",
                    "action_type": "unknown",
                    "rationale": "x",
                    "expected_artifact": "x",
                    "success_condition": "x",
                    "payload": {"rxn_smiles": "CCO>>CC=O"},
                    "route_status": "solved",
                }
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("unknown_action:0:unknown", validation["reasons"])
        self.assertIn("raw_reaction_injection", validation["reasons"])
        self.assertIn("planner_direct_solved_claim", validation["reasons"])

    def test_action_batch_validation_rejects_bad_semantics_and_hidden_reaction_string(self):
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "bad_semantics",
            "round_index": 1,
            "route_status": "solved",
            "notes": "Do not allow hidden reaction strings like CCO>>CC=O.",
            "semantics": {
                "planner_can_emit_solved": True,
                "raw_reaction_output_allowed": True,
                "deterministic_validator_required": True,
            },
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "stop",
                    "action_type": "stop_unresolved",
                    "rationale": "stop",
                    "expected_artifact": "stop marker",
                    "success_condition": "stop selected",
                    "payload": {},
                }
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("planner_direct_solved_claim", validation["reasons"])
        self.assertIn("planner_semantics_allow_solved_claim", validation["reasons"])
        self.assertIn("planner_semantics_allow_raw_reaction_output", validation["reasons"])
        self.assertIn("raw_reaction_injection", validation["reasons"])

    def test_action_batch_validation_requires_direction_change_after_two_unproductive_rounds(self):
        repeated_action = {
            "schema_version": "agent_action.v1",
            "action_id": "search:same",
            "action_type": "search_literature",
            "rationale": "repeat same search",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidate",
            "payload": _test_search_payload("same"),
        }
        repeated_signature = json.dumps(
            {"action_type": "search_literature", "payload": _test_search_payload("same")},
            sort_keys=True,
        )
        board = {
            "action_history": [
                {
                    "round_index": 1,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "action_signature": repeated_signature,
                },
                {
                    "round_index": 2,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "action_signature": repeated_signature,
                },
            ]
        }
        repeated_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "stuck",
            "round_index": 3,
            "actions": [repeated_action],
        }
        changed_batch = {
            **repeated_batch,
            "actions": [{**repeated_action, "action_id": "search:new", "payload": _test_search_payload("new")}],
        }
        stop_batch = {
            **repeated_batch,
            "actions": [
                {
                    **repeated_action,
                    "action_id": "stop",
                    "action_type": "stop_unresolved",
                    "payload": {},
                    "rationale": "stop after repeated unproductive rounds",
                    "expected_artifact": "stop marker",
                    "success_condition": "stop selected",
                }
            ],
        }

        repeated_validation = validate_action_batch(repeated_batch, blackboard=board)
        changed_validation = validate_action_batch(changed_batch, blackboard=board)
        stop_validation = validate_action_batch(stop_batch, blackboard=board)

        self.assertFalse(repeated_validation["accepted"])
        self.assertIn(
            "planner_must_stop_or_change_direction_after_two_unproductive_rounds",
            repeated_validation["reasons"],
        )
        self.assertTrue(changed_validation["accepted"], changed_validation["reasons"])
        self.assertTrue(stop_validation["accepted"], stop_validation["reasons"])

    def test_repeated_empty_literature_search_is_not_planned_from_bias(self):
        board = {
            "case_id": "empty_search",
            "target_profile": {"valid": True, "target_name": "steroid", "family_hint": "steroid"},
            "target_side_disconnection_hypotheses": {"hypotheses": [{"hypothesis_id": "h1"}]},
            "analogical_hypothesis_ranking": {"selected_hypotheses": [{"hypothesis_id": "h1"}]},
            "bridge_tasks": [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}],
            "literature_evidence": {
                "source_candidates": [
                    {
                        "source_ref": "doi:10.1000/local",
                        "doi": "10.1000/local",
                        "local_pdf": "/tmp/source.pdf",
                        "access_status": "local_pdf_available",
                    }
                ]
            },
            "current_belief": {"next_action_bias": ["search_literature"]},
            "budget_state": {
                "scout_calls": 2,
                "max_scout_calls": 5,
                "visual_calls": 0,
                "max_visual_calls": 0,
                "chemenzy_runs": 0,
                "max_chemenzy_runs": 1,
                "child_target_runs": 0,
                "max_child_target_runs": 0,
            },
            "action_history": [
                {
                    "round_index": 3,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "reasons": ["no_source_candidates"],
                    "blackboard_delta": {},
                    "action_signature": json.dumps({"action_type": "search_literature", "payload": {"query": "first"}}),
                },
                {
                    "round_index": 4,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "reasons": ["no_source_candidates"],
                    "blackboard_delta": {},
                    "action_signature": json.dumps({"action_type": "search_literature", "payload": {"query": "second"}}),
                },
            ],
        }

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("search_literature", action_types)

    def test_repeated_literature_source_is_stale_not_useful(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO", family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "search",
            "action_type": "search_literature",
            "rationale": "search",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidate",
            "payload": _test_search_payload("steroid synthesis"),
        }
        result = {
            "accepted": True,
            "result": {
                "schema_version": "literature_scout_report.v1",
                "source_candidates": [
                    {
                        "source_ref": "doi:10.1000/source",
                        "doi": "10.1000/source",
                        "title": "Source",
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(board, action=action, action_result=result, round_index=1, run_dir=tmp)
            board = update_blackboard_from_action(board, action=action, action_result=result, round_index=2, run_dir=tmp)

        self.assertTrue(board["action_history"][0]["useful_artifact"])
        self.assertFalse(board["action_history"][1]["useful_artifact"])
        self.assertTrue(board["action_history"][1]["stale"])

    def test_action_batch_validation_rejects_round_budget_overrun(self):
        action = {
            "schema_version": "agent_action.v1",
            "rationale": "x",
            "expected_artifact": "x",
            "success_condition": "x",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "budget",
            "round_index": 1,
            "actions": [
                {**action, "action_id": "a", "action_type": "run_guided_chemenzy"},
                {**action, "action_id": "b", "action_type": "run_guided_chemenzy"},
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("guided_chemenzy_round_budget_exceeded", validation["reasons"])

    def test_action_batch_validation_rejects_total_budget_overrun(self):
        action = {
            "schema_version": "agent_action.v1",
            "rationale": "x",
            "expected_artifact": "x",
            "success_condition": "x",
            "payload": {},
        }
        board = {
            "budget_state": {
                "scout_calls": 3,
                "max_scout_calls": 3,
                "visual_calls": 2,
                "max_visual_calls": 2,
                "chemenzy_runs": 1,
                "max_chemenzy_runs": 1,
                "child_target_runs": 2,
                "max_child_target_runs": 2,
                "template_application_actions": 3,
                "max_template_application_actions": 3,
            }
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "budget",
            "round_index": 9,
            "actions": [
                {**action, "action_id": "search", "action_type": "search_literature"},
                {**action, "action_id": "visual", "action_type": "extract_visual_literature_chain"},
                {**action, "action_id": "chemenzy", "action_type": "run_guided_chemenzy"},
            ],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertFalse(validation["accepted"])
        self.assertIn("scout_total_budget_exceeded", validation["reasons"])
        self.assertIn("visual_total_budget_exceeded", validation["reasons"])
        self.assertIn("guided_chemenzy_total_budget_exceeded", validation["reasons"])

        child_batch = {**batch, "actions": [{**action, "action_id": "child", "action_type": "expand_child_target"}]}
        template_batch = {
            **batch,
            "actions": [{**action, "action_id": "template", "action_type": "apply_analogical_template_to_target"}],
        }

        self.assertIn("child_expansion_total_budget_exceeded", validate_action_batch(child_batch, blackboard=board)["reasons"])
        self.assertIn(
            "template_application_total_budget_exceeded",
            validate_action_batch(template_batch, blackboard=board)["reasons"],
        )

    def test_action_batch_validation_does_not_count_pdf_structure_as_visual_budget(self):
        board = {"budget_state": {"visual_calls": 2, "max_visual_calls": 2}}
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "pdf",
            "action_type": "extract_pdf_literature_structures",
            "rationale": "read local PDF structures",
            "expected_artifact": "literature_pdf_structure_evidence.v1",
            "success_condition": "PDF structure evidence is recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "pdf_budget",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)
        after_pdf = update_budget_for_action(board, "extract_pdf_literature_structures", payload={})
        after_visual = update_budget_for_action(board, "extract_visual_literature_chain", payload={})

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(after_pdf["budget_state"]["visual_calls"], 2)
        self.assertEqual(after_visual["budget_state"]["visual_calls"], 3)

    def test_action_batch_validation_does_not_count_template_actions_as_literature_sources(self):
        action = {
            "schema_version": "agent_action.v1",
            "rationale": "x",
            "expected_artifact": "x",
            "success_condition": "x",
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "template_parallel",
            "round_index": 5,
            "actions": [
                {
                    **action,
                    "action_id": "a",
                    "action_type": "search_literature",
                    "payload": _test_search_payload("template parallel literature", max_sources=3),
                },
                {
                    **action,
                    "action_id": "b",
                    "action_type": "extract_analogical_reaction_templates",
                    "payload": _test_analogical_template_payload(
                        "extract_analogical_reaction_templates",
                        max_templates=10,
                    ),
                },
            ],
        }

        validation = validate_action_batch(batch)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_guided_chemenzy_search_policy(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "try guided search",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "guided_missing_policy",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("guided_chemenzy_payload:0:missing_search_policy", validation["reasons"])

    def test_action_batch_validation_accepts_blackboard_guided_chemenzy_policy(self):
        target = TargetInput(target_name="guided_valid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_refs"] = ["doi:10.0000/source"]
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "try guided search with auditable policy",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": build_agentic_guided_payload(board),
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_allows_simple_direct_chemenzy_baseline(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "simple target direct baseline",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": build_agentic_guided_payload(board),
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_rejects_complex_guided_without_prior_signal(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "complex target premature guided search",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": build_agentic_guided_payload(board),
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertFalse(validation["accepted"])
        self.assertIn(
            "guided_chemenzy_payload:0:guided_chemenzy_missing_prior_signal_for_complex_target",
            validation["reasons"],
        )

    def test_action_batch_validation_allows_bounded_complex_initial_probe(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        payload = build_agentic_guided_payload(board)
        policy = payload["search_policy"]
        policy["mode"] = "guided"
        policy["search_mode"] = "initial_probe"
        policy["source_budget"]["initial_scan_allowed"] = True
        policy["source_budget"]["max_candidates"] = 3
        policy["compiler_metadata"]["initial_scan_probe"] = True
        payload.update(
            {
                "initial_probe": True,
                "max_steps": 4,
                "chem_enzy_iterations": 6,
                "chem_enzy_expansion_topk": 12,
                "timeout_s": 60,
            }
        )
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "complex target cheap initial probe",
            "expected_artifact": "guided_chemenzy_probe_result.v1",
            "success_condition": "bounded verifier feedback is recorded",
            "payload": payload,
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_explicit_child_subgoal_targets(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "child",
            "action_type": "expand_child_target",
            "rationale": "expand a child target",
            "expected_artifact": "route_expansion_subgoal_search_result.v1",
            "success_condition": "child verifier feedback is recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "child_missing_target",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("child_expansion_payload:0:missing_subgoal_targets", validation["reasons"])

    def test_action_batch_validation_requires_stitch_parent_route_binding(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "stitch",
            "action_type": "stitch_parent_route",
            "rationale": "prove parent route connectivity",
            "expected_artifact": "stitched_parent_route_proof.v1",
            "success_condition": "parent proof clauses are recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "stitch_missing_binding",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("stitch_parent_route_payload:0:missing_proof_binding", validation["reasons"])

    def test_action_batch_validation_requires_literature_search_policy(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "search",
            "action_type": "search_literature",
            "rationale": "search for literature",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidates are recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "search_missing_policy",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("search_literature_payload:0:missing_search_intent_or_queries", validation["reasons"])
        self.assertIn("search_literature_payload:0:missing_source_acquisition_policy", validation["reasons"])

    def test_action_batch_validation_accepts_source_acquisition_policy_schema_alias(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "search",
            "action_type": "search_literature",
            "rationale": "search for source metadata",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidates are recorded",
            "payload": {
                "search_intent": "find source metadata",
                "source_acquisition_policy": {
                    "schema_version": "source_acquisition_policy.v1",
                    "codex_online_first": True,
                    "local_pdf_fallback_allowed": True,
                    "placeholder_allowed_after_failures": True,
                    "auto_local_pdf_requires_agent_discovered_metadata": True,
                    "fallback_order": ["codex_online", "local_pdf", "placeholder"],
                    "no_solved_claim": True,
                },
            },
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "search_policy_alias",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_rejects_invalid_planner_source_hints(self):
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "bad_hints",
            "round_index": 1,
            "actions": [],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
            "planner_source_hints": [
                {
                    "schema_version": "planner_source_hint.v1",
                    "hint_id": "bad",
                    "source_ref": "doi:10.1000/bad",
                    "title": "bad",
                    "doi": "10.1000/bad",
                    "evidence_class": "planner_source_hint",
                    "allowed_use": "parent_route_proof",
                    "no_solved_claim": True,
                }
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("planner_source_hint_invalid_allowed_use:0", validation["reasons"])

    def test_action_batch_validation_requires_analogical_template_policy(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "template",
            "action_type": "extract_analogical_reaction_templates",
            "rationale": "extract guarded analogical templates",
            "expected_artifact": "analogical_reaction_template_report.v1",
            "success_condition": "advisory templates are recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "template_missing_policy",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("analogical_template_payload:0:missing_analogical_template_policy", validation["reasons"])

    def test_action_batch_validation_allows_analogical_bridge_task_triage(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "template",
            "action_type": "rank_analogical_reaction_templates",
            "rationale": "rank guarded analogical templates",
            "expected_artifact": "analogical_reaction_template_ranking.v1",
            "success_condition": "advisory templates are ranked",
            "payload": {
                "analogical_template_policy": {
                    "schema_version": "agentic_analogical_template_action_policy.v1",
                    "action_type": "rank_analogical_reaction_templates",
                    "analogy_is_advisory_only": True,
                    "no_solved_claim": True,
                    "requires_verifier": True,
                    "requires_parent_route_proof": True,
                    "production_write_blocked": True,
                    "raw_reaction_output_allowed": False,
                    "final_verdict_authority": "deterministic_parent_route_proof",
                    "allowed_use": ["planner_priority", "bridge_task_triage"],
                    "deterministic_template_validation_required": True,
                }
            },
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "template_bridge_triage",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_search_literature_action_includes_source_acquisition_policy(self):
        target = TargetInput(target_name="policy_search", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]

        batch = plan_action_batch(board, round_index=1, exhaust_round_budget=True)
        search_action = next(row for row in batch["actions"] if row["action_type"] == "search_literature")

        self.assertEqual(search_action["payload"]["source_acquisition_policy"]["fallback_order"], ["codex_online", "local_pdf", "placeholder"])
        self.assertTrue(search_action["payload"]["source_acquisition_policy"]["auto_local_pdf_requires_agent_discovered_metadata"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_search_literature_payload_includes_planner_source_hints(self):
        target = TargetInput(target_name="hinted_search", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["planner_source_hints"] = [
            {
                "schema_version": "planner_source_hint.v1",
                "hint_id": "hint1",
                "hint_key": "10.4242/plannerhint2026",
                "source_ref": "doi:10.4242/plannerhint2026",
                "title": "Planner hinted steroid synthesis",
                "doi": "10.4242/plannerhint2026",
                "pii": "",
                "url": "https://doi.org/10.4242/plannerhint2026",
                "local_pdf": "",
                "local_ref": "",
                "source_type": "planner_discovered_literature_metadata",
                "relevance_rationale": "planner found DOI during search",
                "expected_scheme_or_compound_labels": ["1", "2"],
                "extraction_task_recommendations": [],
                "evidence_class": "planner_source_hint",
                "allowed_use": "source_acquisition_hint_only",
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=1, exhaust_round_budget=True)
        search_action = next(row for row in batch["actions"] if row["action_type"] == "search_literature")

        self.assertEqual(search_action["payload"]["planner_source_hints"][0]["doi"], "10.4242/plannerhint2026")
        self.assertIn("10.4242/plannerhint2026", search_action["payload"]["queries"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_source_binding_for_multi_source_extraction(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "extract",
            "action_type": "extract_visual_literature_chain",
            "rationale": "extract a specific source",
            "expected_artifact": "visual_literature_chain.v1",
            "success_condition": "one source is extracted",
            "payload": {},
        }
        board = {
            "literature_evidence": {
                "source_candidates": [
                    {"source_ref": "doi:first", "doi": "10.1/first", "local_pdf": "/tmp/first.pdf"},
                    {"source_ref": "doi:second", "doi": "10.1/second", "local_pdf": "/tmp/second.pdf"},
                ]
            }
        }
        unbound_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "multi_source",
            "round_index": 1,
            "actions": [action],
        }
        bound_batch = {
            **unbound_batch,
            "actions": [{**action, "payload": {"source_ref": "doi:first"}}],
        }

        unbound_validation = validate_action_batch(unbound_batch, blackboard=board)
        bound_validation = validate_action_batch(bound_batch, blackboard=board)

        self.assertFalse(unbound_validation["accepted"])
        self.assertIn(
            "source_sensitive_action_missing_source_binding:0:extract_visual_literature_chain",
            unbound_validation["reasons"],
        )
        self.assertTrue(bound_validation["accepted"], bound_validation["reasons"])

    def test_action_batch_validation_allows_single_source_extraction_default(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "extract",
            "action_type": "extract_pdf_literature_structures",
            "rationale": "extract the only available source",
            "expected_artifact": "literature_pdf_structure_evidence.v1",
            "success_condition": "single source is rendered",
            "payload": {},
        }
        board = {
            "literature_evidence": {
                "source_candidates": [
                    {"source_ref": "doi:only", "doi": "10.1/only", "local_pdf": "/tmp/only.pdf"},
                ]
            }
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "single_source",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_chain_binding_for_multi_chain_compile(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "compile",
            "action_type": "compile_exact_literature_rows",
            "rationale": "compile a specific visual chain",
            "expected_artifact": "literature_exact_rows.v1",
            "success_condition": "one visual chain is compiled",
            "payload": {},
        }
        board = {
            "literature_evidence": {
                "visual_chains": [
                    {"chain_id": "visual:first", "source_ref": "doi:first", "source_pdf_path": "/tmp/first.pdf"},
                    {"chain_id": "visual:second", "source_ref": "doi:second", "source_pdf_path": "/tmp/second.pdf"},
                ]
            }
        }
        unbound_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "multi_chain",
            "round_index": 1,
            "actions": [action],
        }
        bound_batch = {
            **unbound_batch,
            "actions": [{**action, "payload": {"chain_id": "visual:first"}}],
        }

        unbound_validation = validate_action_batch(unbound_batch, blackboard=board)
        bound_validation = validate_action_batch(bound_batch, blackboard=board)

        self.assertFalse(unbound_validation["accepted"])
        self.assertIn(
            "source_sensitive_action_missing_source_binding:0:compile_exact_literature_rows",
            unbound_validation["reasons"],
        )
        self.assertTrue(bound_validation["accepted"], bound_validation["reasons"])

    def test_codex_action_planner_batch_is_used_when_enabled(self):
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "mode": "codex_test",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:disconnection",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "Codex chose to inspect target handles before any rerun.",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "advisory hypotheses are recorded",
                    "payload": {},
                }
            ],
            "planner_source_hints": [
                {
                    "schema_version": "planner_source_hint.v1",
                    "hint_id": "planner_hint_1",
                    "source_ref": "doi:10.4242/plannerhint2026",
                    "title": "Planner hinted target-proximal synthesis",
                    "doi": "10.4242/plannerhint2026",
                    "pii": "",
                    "url": "https://doi.org/10.4242/plannerhint2026",
                    "local_pdf": "",
                    "local_ref": "",
                    "source_type": "planner_discovered_literature_metadata",
                    "relevance_rationale": "Codex planner found a traceable source lead while choosing actions.",
                    "expected_scheme_or_compound_labels": ["1", "2"],
                    "extraction_task_recommendations": ["search_literature"],
                    "evidence_class": "planner_source_hint",
                    "allowed_use": "source_acquisition_hint_only",
                    "no_solved_claim": True,
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        expected_snapshot_ref = ""
        snapshot_context_schema = ""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            expected_snapshot_ref = str(run_dir / "codex_action_planner_blackboard_round_1.json")
            result = run_agentic_blackboard_controller(
                target_name="codex_plan",
                target_smiles="CCO",
                output_dir=run_dir,
                max_rounds=1,
                mock_tool_results={"codex_action_planner": codex_batch},
            )
            self.assertTrue((run_dir / "action_batch_round_1.json").exists())
            self.assertTrue((run_dir / "action_batch_validation_round_1.json").exists())
            bundle_artifacts = result["artifact_bundle"]["artifacts"]
            self.assertIn("agent_action_batch_round_1", bundle_artifacts)
            self.assertIn("agent_action_batch_validation_round_1", bundle_artifacts)
            self.assertEqual(bundle_artifacts["agent_action_batch_round_1"]["artifact_type"], "AgentActionBatch")
            self.assertEqual(
                bundle_artifacts["agent_action_batch_round_1"]["validation_ref"],
                str(run_dir / "action_batch_validation_round_1.json"),
            )
            validation_keys = {
                row.get("artifact_key")
                for row in result["artifact_bundle"]["validations"]
                if row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
            }
            self.assertIn("agent_action_batch_round_1", validation_keys)
            self.assertIn("agent_action_batch_validation_round_1", validation_keys)
            snapshot = json.loads(Path(expected_snapshot_ref).read_text(encoding="utf-8"))
            snapshot_context_schema = str((snapshot.get("planner_context") or {}).get("schema_version") or "")

        batch = result["action_batches"][0]
        self.assertEqual(batch["mode"], "codex_xhigh_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual(
            batch["codex_action_planner"]["blackboard_snapshot_ref"],
            expected_snapshot_ref,
        )
        self.assertEqual(snapshot_context_schema, "codex_action_planner_context.v1")
        self.assertEqual(batch["actions"][0]["action_type"], "generate_disconnection_hypotheses")
        self.assertIn("web_search", batch["codex_action_planner"]["tool_policy"]["allowed_tools"])
        self.assertGreater(batch["codex_action_planner"]["tool_policy"]["max_tool_calls"], 0)
        self.assertTrue(batch["codex_action_planner"]["tool_policy"]["cli_search_enabled"])
        self.assertEqual(batch["planner_source_hints"][0]["allowed_use"], "source_acquisition_hint_only")
        self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])
        self.assertEqual(
            result["agent_blackboard"]["literature_evidence"]["planner_source_hints"][0]["doi"],
            "10.4242/plannerhint2026",
        )
        self.assertEqual(result["agent_blackboard"]["literature_evidence"]["source_candidates"], [])
        self.assertEqual(result["agent_blackboard"]["literature_evidence"]["source_lifecycle"][0]["stage"], "planner_hint")
        self.assertEqual(
            result["agent_blackboard"]["literature_evidence"]["source_lifecycle"][0]["next_recommended_stage"],
            "search_literature",
        )
        planner_history = result["agent_blackboard"]["planner_history"]
        self.assertEqual(len(planner_history), 1)
        self.assertEqual(planner_history[0]["mode"], "codex_xhigh_blackboard_planner")
        self.assertEqual(planner_history[0]["planner_source_hint_count"], 1)
        self.assertTrue(planner_history[0]["codex_action_planner"]["attempted"])
        self.assertFalse(planner_history[0]["codex_action_planner"]["fallback_used"])
        self.assertIn("web_search", planner_history[0]["codex_action_planner"]["tool_policy"]["allowed_tools"])
        self.assertEqual(
            planner_history[0]["codex_action_planner"]["blackboard_snapshot_ref"],
            expected_snapshot_ref,
        )
        self.assertEqual(result["agent_blackboard"]["budget_state"]["codex_action_planner_runs"], 1)
        self.assertIn("codex_action_planner_round_1", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("codex_action_planner_blackboard_snapshot_round_1", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("agent_action_batch_round_1", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("agent_action_batch_validation_round_1", result["agent_blackboard"]["artifact_refs"])
        capability_checks = {
            row["requirement_id"]: row
            for row in result["artifact_bundle"]["artifacts"]["agentic_capability_audit"]["payload"]["requirement_checks"]
        }
        planner_check = capability_checks["planner_decision_history_audited"]
        self.assertTrue(planner_check["accepted"], planner_check["reasons"])
        self.assertIn("codex_snapshot_context_count:1", planner_check["evidence"])
        self.assertIn("codex_snapshot_payload_requirement_count:1", planner_check["evidence"])
        self.assertIn("codex_snapshot_tool_policy_count:1", planner_check["evidence"])
        self.assertIn("codex_history_tool_policy_count:1", planner_check["evidence"])

    def test_codex_planner_snapshot_includes_derived_context_for_pending_sources_and_transitions(self):
        target = TargetInput(target_name="planner_context", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["source_discovery_mode"] = "codex_online+local_pdf_cache"
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:first",
                "doi": "10.1000/first",
                "local_pdf": "/tmp/first.pdf",
                "source_discovery_mode": "codex_online+local_pdf_cache",
                "local_pdf_match": {"match_basis": "doi", "agent_discovered_doi": "10.1000/first"},
                "local_pdf_index": {"match_policy": "agent_discovered_metadata_required"},
            },
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:second",
                "doi": "10.1000/second",
                "local_pdf": "/tmp/second.pdf",
                "source_discovery_mode": "codex_online+local_pdf_cache",
                "local_pdf_match": {"match_basis": "doi", "agent_discovered_doi": "10.1000/second"},
                "local_pdf_index": {"match_policy": "agent_discovered_metadata_required"},
            },
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:third",
                "doi": "10.1000/third",
                "url": "https://doi.org/10.1000/third",
                "local_pdf": "",
                "source_discovery_mode": "codex_online",
                "access_status": "metadata_only",
            },
        ]
        board["literature_evidence"]["local_pdf_proxy_requests"] = [
            {
                "schema_version": "local_pdf_proxy_request.v1",
                "request_id": "req-third",
                "source_ref": "doi:third",
                "doi": "10.1000/third",
                "url": "https://doi.org/10.1000/third",
                "title": "Third",
                "content_scope": "article",
            }
        ]
        board["literature_evidence"]["source_lifecycle"] = [
            {
                "schema_version": "agent_source_lifecycle.v1",
                "source_key": "doi:10.1000/first",
                "source_ref": "doi:first",
                "title": "First",
                "doi": "10.1000/first",
                "local_pdf": "/tmp/first.pdf",
                "stage": "pdf_rendered",
                "next_recommended_stage": "extract_visual_literature_chain",
                "stage_flags": {"pdf_rendered": True, "local_pdf_available": True},
                "counts": {"source_candidates": 1, "pdf_structure_evidence": 1},
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_source_lifecycle.v1",
                "source_key": "doi:10.1000/second",
                "source_ref": "doi:second",
                "title": "Second",
                "doi": "10.1000/second",
                "local_pdf": "/tmp/second.pdf",
                "stage": "local_pdf_available",
                "next_recommended_stage": "extract_pdf_literature_structures",
                "stage_flags": {"local_pdf_available": True},
                "counts": {"source_candidates": 1},
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_source_lifecycle.v1",
                "source_key": "doi:10.1000/third",
                "source_ref": "doi:third",
                "title": "Third",
                "doi": "10.1000/third",
                "local_pdf": "",
                "stage": "local_pdf_proxy_requested",
                "next_recommended_stage": "await_local_pdf_proxy_download",
                "stage_flags": {"source_candidate": True, "local_pdf_proxy_requested": True},
                "counts": {"source_candidates": 1, "local_pdf_proxy_requests": 1},
                "no_solved_claim": True,
            },
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:first", "accepted": True}
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "useful_artifact": True,
                "stale": False,
                "blackboard_delta": {"source_candidates": 2},
                "changed_blackboard_fields": ["source_candidates"],
            },
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 2,
                "action_type": "extract_pdf_literature_structures",
                "useful_artifact": True,
                "stale": False,
                "blackboard_delta": {"pdf_structure_evidence": 1},
                "changed_blackboard_fields": ["pdf_structure_evidence"],
            },
        ]

        context = _planner_context_summary(board)

        self.assertEqual(context["source_acquisition"]["auto_local_pdf_cache_match_count"], 2)
        self.assertEqual(context["source_acquisition"]["source_lifecycle_count"], 3)
        self.assertEqual(context["source_acquisition"]["source_lifecycle_stage_counts"]["pdf_rendered"], 1)
        self.assertEqual(context["source_acquisition"]["source_lifecycle_stage_counts"]["local_pdf_proxy_requested"], 1)
        self.assertEqual(context["source_acquisition"]["awaiting_local_pdf_proxy_count"], 1)
        self.assertEqual(context["source_acquisition"]["local_pdf_proxy_request_count"], 1)
        self.assertEqual(context["literature_processing"]["source_lifecycle"][0]["stage"], "pdf_rendered")
        self.assertEqual(
            context["literature_processing"]["pending_local_pdf_proxy_sources"][0]["source_ref"],
            "doi:third",
        )
        self.assertEqual(context["literature_processing"]["pending_pdf_extraction_sources"][0]["source_ref"], "doi:second")
        self.assertEqual(context["literature_processing"]["pending_visual_extraction_sources"][0]["source_ref"], "doi:first")
        search_requirements = context["action_payload_requirements"]["search_actions"]["search_literature"]
        self.assertIn("source_acquisition_policy", search_requirements["accepted_payload_fields"])
        self.assertTrue(search_requirements["blackboard_guidance"]["auto_local_pdf_requires_agent_discovered_metadata"])
        requirements = context["action_payload_requirements"]["source_sensitive_actions"]
        self.assertTrue(requirements["extract_pdf_literature_structures"]["currently_required"])
        self.assertTrue(requirements["extract_visual_literature_chain"]["currently_required"])
        self.assertTrue(requirements["compile_exact_literature_rows"]["currently_required"])
        self.assertIn("source_ref", requirements["extract_visual_literature_chain"]["accepted_payload_fields"])
        self.assertIn("pdf_path", requirements["extract_pdf_literature_structures"]["accepted_payload_fields"])
        self.assertIn("chain_id", requirements["compile_exact_literature_rows"]["accepted_payload_fields"])
        self.assertEqual(
            requirements["extract_pdf_literature_structures"]["binding_candidates"][0]["source_ref"],
            "doi:first",
        )
        guided_requirements = context["action_payload_requirements"]["guided_actions"]["run_guided_chemenzy"]
        self.assertIn("search_policy", guided_requirements["accepted_payload_fields"])
        self.assertIn("compiler_metadata.requires_verifier", guided_requirements["required_policy_safety_fields"])
        self.assertEqual(guided_requirements["blackboard_guidance"]["exact_row_count"], 0)
        child_requirements = context["action_payload_requirements"]["child_expansion_actions"]["expand_child_target"]
        self.assertIn("subgoal_targets", child_requirements["accepted_payload_fields"])
        self.assertIn("child_route_cannot_promote_parent", child_requirements["required_target_fields"])
        self.assertTrue(child_requirements["blackboard_guidance"]["parent_proof_required_after_child_run"])
        stitch_requirements = context["action_payload_requirements"]["stitch_actions"]["stitch_parent_route"]
        self.assertIn("proof_binding", stitch_requirements["accepted_payload_fields"])
        self.assertIn("exact_literature_row_ids", stitch_requirements["required_binding_fields"])
        self.assertEqual(stitch_requirements["blackboard_guidance"]["final_verdict_authority"], "deterministic_parent_route_proof")
        template_requirements = context["action_payload_requirements"]["analogical_template_actions"][
            "extract_analogical_reaction_templates"
        ]
        self.assertIn("analogical_template_policy", template_requirements["accepted_payload_fields"])
        self.assertIn("requires_parent_route_proof", template_requirements["required_policy_fields"])
        self.assertTrue(template_requirements["blackboard_guidance"]["analogy_is_advisory_only"])
        self.assertEqual(context["recent_blackboard_transitions"][-1]["blackboard_delta"]["pdf_structure_evidence"], 1)
        self.assertFalse(context["safety_boundaries"]["planner_can_emit_solved"])

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = _write_codex_blackboard_snapshot(board, run_dir=Path(tmp), round_index=3)
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            task = _codex_action_planner_task(
                blackboard=board,
                round_index=3,
                run_dir=Path(tmp),
                snapshot_path=snapshot_path,
            )

        self.assertEqual(snapshot["planner_context"]["schema_version"], "codex_action_planner_context.v1")
        self.assertEqual(
            snapshot["planner_context"]["action_payload_requirements"]["schema_version"],
            "codex_action_payload_requirements.v1",
        )
        self.assertEqual(
            snapshot["planner_context"]["planner_tool_policy"]["schema_version"],
            "codex_action_planner_tool_policy.v1",
        )
        self.assertIn("planner_context", snapshot)
        self.assertIn("action_payload_requirements", task.objective)
        self.assertIn("source_ref", task.objective)
        self.assertIn("chain_id", task.objective)
        self.assertIn("analogical_template_policy", task.objective)
        self.assertIn("Planner tool policy", task.objective)
        self.assertIn("web_search", task.allowed_tools)
        self.assertGreater(task.budget.max_tool_calls, 0)
        self.assertTrue(_task_allows_cli_search(task))

    def test_fallback_planner_stops_when_only_waiting_for_local_pdf_proxy(self):
        target = TargetInput(target_name="proxy_wait", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["target_side_disconnection_hypotheses"] = {
            "schema_version": "target_side_disconnection_hypotheses.v1",
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "target_handle": "source_bridge",
                    "no_solved_claim": True,
                }
            ],
            "no_solved_claim": True,
        }
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [
            {"mode": "codex_online", "attempted": True, "accepted": True}
        ]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.5555/proxy.wait",
                "doi": "10.5555/proxy.wait",
                "url": "https://doi.org/10.5555/proxy.wait",
                "local_pdf": "",
                "access_status": "metadata_only",
                "no_solved_claim": True,
            }
        ]
        board["literature_evidence"]["local_pdf_proxy_requests"] = [
            {
                "schema_version": "local_pdf_proxy_request.v1",
                "request_id": "proxy-wait",
                "source_ref": "doi:10.5555/proxy.wait",
                "doi": "10.5555/proxy.wait",
                "url": "https://doi.org/10.5555/proxy.wait",
                "content_scope": "article",
            }
        ]

        batch = plan_action_batch(board, round_index=2)
        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual([row["action_type"] for row in batch["actions"]], ["stop_unresolved"])
        self.assertEqual(batch["actions"][0]["payload"]["wait_state"], "local_pdf_proxy_requested")

    def test_codex_action_planner_tool_budget_can_be_disabled(self):
        target = TargetInput(target_name="tool_disabled", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.dict(
                "os.environ",
                {
                    "AUTOPLANNER_CODEX_ACTION_PLANNER_ALLOWED_TOOLS": "none",
                    "AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_TOOL_CALLS": "8",
                },
            ):
                snapshot_path = _write_codex_blackboard_snapshot(board, run_dir=run_dir, round_index=1)
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                task = _codex_action_planner_task(
                    blackboard=board,
                    round_index=1,
                    run_dir=run_dir,
                    snapshot_path=snapshot_path,
                )

        policy = snapshot["planner_context"]["planner_tool_policy"]
        self.assertEqual(policy["allowed_tools"], [])
        self.assertEqual(policy["max_tool_calls"], 0)
        self.assertFalse(policy["cli_search_enabled"])
        self.assertEqual(task.allowed_tools, [])
        self.assertEqual(task.budget.max_tool_calls, 0)
        self.assertFalse(_task_allows_cli_search(task))
        self.assertEqual(task.budget.max_tool_calls, 0)
        self.assertFalse(_task_allows_cli_search(task))

    def test_blackboard_records_codex_planner_fallback_as_planner_note(self):
        target = TargetInput(target_name="planner_fallback", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "mode": "deterministic_policy_fallback_after_codex_planner",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "fallback:stop",
                    "action_type": "stop_unresolved",
                    "rationale": "fallback",
                    "expected_artifact": "stop",
                    "success_condition": "stop",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
            "codex_action_planner": {
                "schema_version": "codex_action_planner_metadata.v1",
                "fallback_used": True,
                "fallback_reason": "codex_action_planner_batch_invalid",
                "record_status": "accepted_draft",
                "record_backend": "mock_output",
                "record_ref": "/tmp/codex_action_planner_run_record_round_1.json",
                "batch_validation": {"accepted": False, "reasons": ["raw_reaction_injection"]},
            },
        }
        validation = validate_action_batch(batch, blackboard=board)

        updated = update_blackboard_from_action_batch(
            board,
            action_batch=batch,
            validation=validation,
            round_index=1,
        )

        self.assertTrue(updated["planner_history"][0]["codex_action_planner"]["attempted"])
        self.assertTrue(updated["planner_history"][0]["codex_action_planner"]["fallback_used"])
        self.assertEqual(updated["budget_state"]["codex_action_planner_runs"], 1)
        self.assertEqual(updated["current_belief"]["planner_notes"][0]["reason"], "codex_action_planner_batch_invalid")
        self.assertEqual(
            updated["artifact_refs"]["codex_action_planner_round_1"],
            "/tmp/codex_action_planner_run_record_round_1.json",
        )

    def test_capability_audit_rejects_codex_snapshot_without_payload_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_payload_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_tool_policy(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["search_policy", "chem_enzy_search_policy"],
            }
        }
        child_actions = {
            "expand_child_target": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["subgoal_targets", "child_targets"],
            }
        }
        stitch_actions = {
            "stitch_parent_route": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["proof_binding", "proof_policy", "analogy_refs"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                                "child_expansion_actions": child_actions,
                                "stitch_actions": stitch_actions,
                                "analogical_template_actions": _test_analogical_template_requirements(),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_tool_policy:0", check["reasons"])
        self.assertIn("codex_planner_history_missing_tool_policy:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_search_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_search_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_guided_payload_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_guided_action_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_child_expansion_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["search_policy", "chem_enzy_search_policy"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_child_expansion_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_stitch_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["search_policy", "chem_enzy_search_policy"],
            }
        }
        child_actions = {
            "expand_child_target": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["subgoal_targets", "child_targets"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                                "child_expansion_actions": child_actions,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_stitch_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_analogical_template_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["search_policy", "chem_enzy_search_policy"],
            }
        }
        child_actions = {
            "expand_child_target": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["subgoal_targets", "child_targets"],
            }
        }
        stitch_actions = {
            "stitch_parent_route": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["proof_binding", "proof_policy", "analogy_refs"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                                "child_expansion_actions": child_actions,
                                "stitch_actions": stitch_actions,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_analogical_template_requirements:0", check["reasons"])

    def test_codex_action_planner_rejects_solved_or_raw_reaction_and_falls_back(self):
        target = TargetInput(target_name="bad_codex_plan", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        bad_codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "bad:route",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "bad",
                    "expected_artifact": "bad",
                    "success_condition": "bad",
                    "route_status": "solved",
                    "payload": {"rxn_smiles": "CCO>>CC=O"},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        snapshot_exists = False
        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=bad_codex_batch,
            )
            snapshot_exists = Path(batch["codex_action_planner"]["blackboard_snapshot_ref"]).is_file()

        self.assertEqual(batch["mode"], "deterministic_policy_fallback_after_codex_planner")
        self.assertTrue(batch["codex_action_planner"]["fallback_used"])
        self.assertTrue(snapshot_exists)
        self.assertIn(
            batch["codex_action_planner"]["fallback_reason"],
            {"codex_action_planner_worker_rejected", "codex_action_planner_batch_invalid"},
        )
        self.assertNotEqual(batch["actions"][0]["action_type"], "run_guided_chemenzy")

    def test_codex_action_planner_requires_source_binding_for_multi_source_actions(self):
        target = TargetInput(target_name="codex_multi_source", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "doi:first", "doi": "10.1/first", "local_pdf": "/tmp/first.pdf"},
            {"source_ref": "doi:second", "doi": "10.1/second", "local_pdf": "/tmp/second.pdf"},
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:visual",
                    "action_type": "extract_visual_literature_chain",
                    "rationale": "extract the useful source",
                    "expected_artifact": "visual_literature_chain.v1",
                    "success_condition": "a chain is extracted",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "deterministic_policy_fallback_after_codex_planner")
        self.assertTrue(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual(batch["codex_action_planner"]["fallback_reason"], "codex_action_planner_batch_invalid")
        self.assertIn(
            "source_sensitive_action_missing_source_binding:0:extract_visual_literature_chain",
            batch["codex_action_planner"]["batch_validation"]["reasons"],
        )
        self.assertNotEqual(batch["actions"][0]["action_type"], "extract_visual_literature_chain")

    def test_codex_action_planner_repairs_literature_search_policy_from_blackboard(self):
        target = TargetInput(target_name="codex_search_policy", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:search",
                    "action_type": "search_literature",
                    "rationale": "search literature",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "source candidates are recorded",
                    "payload": {
                        "queries": [
                            {"query_id": "q1", "query": "ethanol synthesis DOI"},
                            "{'query_id': 'q2', 'query': 'ethanol retrosynthesis source'}",
                        ]
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_xhigh_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["queries"], ["ethanol synthesis DOI", "ethanol retrosynthesis source"])
        self.assertTrue(payload["source_acquisition_policy"]["codex_online_first"])
        self.assertEqual(payload["source_acquisition_policy"]["fallback_order"], ["codex_online", "local_pdf", "placeholder"])
        self.assertTrue(payload["source_acquisition_policy"]["no_solved_claim"])
        self.assertTrue(payload["codex_payload_repair"]["completed_from_blackboard"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_drops_premature_complex_guided_chemenzy(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:search",
                    "action_type": "search_literature",
                    "rationale": "search literature",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "source candidates are recorded",
                    "payload": {"query": "steroid target proximal synthesis"},
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:hypotheses",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "generate target-side hypotheses",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "hypotheses are recorded",
                    "payload": {},
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:guided",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "premature complex guided search",
                    "expected_artifact": "guided_chemenzy_result.v1",
                    "success_condition": "verifier feedback is recorded",
                    "payload": build_agentic_guided_payload(board),
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertEqual(batch["mode"], "codex_xhigh_blackboard_planner_repaired")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertIn("search_literature", action_types)
        self.assertIn("generate_disconnection_hypotheses", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        self.assertIn(
            "guided_chemenzy_payload:2:guided_chemenzy_missing_prior_signal_for_complex_target",
            batch["codex_action_planner"]["initial_validation"]["reasons"],
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_bounded_complex_guided_probe_skeleton(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:probe",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "cheap complex-target initial probe",
                    "expected_artifact": "guided_chemenzy_probe_result.v1",
                    "success_condition": "bounded verifier feedback is recorded",
                    "payload": {
                        "initial_probe": True,
                        "search_mode": "initial_probe",
                        "max_steps": 4,
                        "chem_enzy_iterations": 6,
                        "chem_enzy_expansion_topk": 12,
                        "timeout_s": 60,
                        "max_candidates": 3,
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        payload = batch["actions"][0]["payload"]
        policy = payload["search_policy"]
        self.assertEqual(batch["mode"], "codex_xhigh_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertTrue(policy["source_budget"]["initial_scan_allowed"])
        self.assertEqual(policy["source_budget"]["max_candidates"], 3)
        self.assertTrue(policy["compiler_metadata"]["initial_scan_probe"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_guided_chemenzy_policy_from_blackboard(self):
        target = TargetInput(target_name="codex_guided_policy", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:guided",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "try guided search",
                    "expected_artifact": "guided_chemenzy_result.v1",
                    "success_condition": "verifier feedback is recorded",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_xhigh_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        payload = batch["actions"][0]["payload"]
        self.assertTrue(payload["search_policy"]["compiler_metadata"]["requires_verifier"])
        self.assertTrue(payload["search_policy"]["source_budget"]["require_target_core_retention"])
        self.assertTrue(payload["codex_payload_repair"]["completed_from_blackboard"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_child_target_policy(self):
        target = TargetInput(target_name="codex_child_target", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "literature_terminal_child:target", "task_type": "upstream_terminal_synthesis"}]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:child",
                    "action_type": "expand_child_target",
                    "rationale": "expand the upstream child target",
                    "expected_artifact": "route_expansion_subgoal_search_result.v1",
                    "success_condition": "child verifier feedback is recorded",
                    "payload": {"child_targets": [{"name": "ethanol child", "target_smiles": "CCO"}]},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_xhigh_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        child = batch["actions"][0]["payload"]["subgoal_targets"][0]
        self.assertEqual(child["smiles"], "CCO")
        self.assertTrue(child["target_equivalence_audit_required"])
        self.assertTrue(child["child_route_cannot_promote_parent"])
        self.assertTrue(child["chem_enzy_search_policy"]["compiler_metadata"]["requires_verifier"])
        self.assertTrue(child["chem_enzy_search_policy"]["compiler_metadata"]["child_route_cannot_promote_parent"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_requires_stitch_parent_route_binding(self):
        target = TargetInput(target_name="codex_stitch", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:ethanol"}]
        board["action_history"] = [
            {"round_index": 1, "action_type": "expand_child_target", "useful_artifact": True, "stale": False}
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:stitch",
                    "action_type": "stitch_parent_route",
                    "rationale": "prove parent route",
                    "expected_artifact": "stitched_parent_route_proof.v1",
                    "success_condition": "parent proof clauses are recorded",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "deterministic_policy_fallback_after_codex_planner")
        self.assertTrue(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual(batch["codex_action_planner"]["fallback_reason"], "codex_action_planner_batch_invalid")
        self.assertIn(
            "stitch_parent_route_payload:0:missing_proof_binding",
            batch["codex_action_planner"]["batch_validation"]["reasons"],
        )

    def test_codex_action_planner_repairs_analogical_template_policy(self):
        target = TargetInput(target_name="codex_template_policy", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_refs"] = ["doi:analog"]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:template",
                    "action_type": "extract_analogical_reaction_templates",
                    "rationale": "extract analogical templates",
                    "expected_artifact": "analogical_reaction_template_report.v1",
                    "success_condition": "advisory templates are recorded",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_xhigh_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        policy = batch["actions"][0]["payload"]["analogical_template_policy"]
        self.assertTrue(policy["analogy_is_advisory_only"])
        self.assertTrue(policy["no_solved_claim"])
        self.assertIn("bridge_task_triage", policy["allowed_use"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_explicit_action_planner_overrides_codex_action_planner(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "override",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "override:stop",
                        "action_type": "stop_unresolved",
                        "rationale": "explicit planner override",
                        "expected_artifact": "stop marker",
                        "success_condition": "stop selected",
                        "payload": _test_search_payload("online case target proximal literature"),
                    }
                ],
            }

        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:disconnection",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "should not run",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "should not run",
                    "payload": {},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="override",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                use_codex_action_planner=True,
                action_planner=planner,
                mock_tool_results={"codex_action_planner": codex_batch},
            )

        self.assertEqual(result["action_batches"][0]["actions"][0]["action_type"], "stop_unresolved")
        self.assertNotIn("codex_action_planner", result["action_batches"][0])

    def test_target_side_strategy_for_mla_like_target_is_advisory(self):
        result = build_target_side_disconnection_hypotheses(
            target_smiles=MLA_LIKE_SMILES,
            target_name="MLA analog",
            family_hint="MLA alkaloid",
        )
        handles = {row["target_handle"] for row in result["hypotheses"]}
        payload = json.dumps(result, sort_keys=True)

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertIn("aryl_ester_or_anthranilate_sidechain", handles)
        self.assertIn("imide_or_succinimide_fragment", handles)
        self.assertIn("polycyclic_cage_core", handles)
        self.assertIn("tertiary_amine", handles)
        self.assertTrue(result["no_solved_claim"])
        self.assertNotIn("rxn_smiles", payload)
        self.assertNotIn("reaction_smiles", payload)

    def test_target_side_strategy_for_bufotalin_recognizes_c17_pyrone_handle(self):
        result = build_target_side_disconnection_hypotheses(
            target_smiles=BUFOTALIN_SMILES,
            target_name="bufotalin",
            family_hint="bufadienolide steroid C17 pyrone",
        )
        handles = {row["target_handle"] for row in result["hypotheses"]}
        payload = json.dumps(result, sort_keys=True)

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertIn("bufadienolide_c17_pyrone_sidechain", handles)
        self.assertTrue(result["no_solved_claim"])
        self.assertNotIn("rxn_smiles", payload)
        self.assertNotIn("reaction_smiles", payload)

    def test_target_side_strategy_for_9oh4hp_prefers_generic_objective_endpoints(self):
        result = build_target_side_disconnection_hypotheses(
            target_smiles=C22_9OH_4HP_SMILES,
            target_name="9-OH-4-HP",
            family_hint="9,21-dihydroxy-20-methyl-pregna-4-en-3-one steroid",
            case_id="target1_steroid",
        )
        handles = {row["target_handle"] for row in result["hypotheses"]}
        anchors = {row["anchor_id"]: row for row in result["semisynthesis_anchors"]}
        selected_objectives = {
            row["objective_type"]
            for row in result["route_objective_summary"]["selected_objectives"]
        }
        payload = json.dumps(result, sort_keys=True)

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertIn("semisynthesis_or_biotransformation_anchor", handles)
        self.assertTrue(result["route_scope"]["de_novo_core_construction_deprioritized"])
        self.assertTrue(result["route_scope"]["objective_evidence_validation_required"])
        self.assertIn("semisynthesis_from_natural_product", selected_objectives)
        self.assertIn("biotransformation_endpoint", selected_objectives)
        self.assertIn(
            "route_objective_anchor:semisynthesis_from_natural_product:natural_product_or_feedstock_same_scaffold_pool",
            anchors,
        )
        self.assertIn(
            "route_objective_anchor:biotransformation_endpoint:same_core_biotransformation_substrate",
            anchors,
        )
        self.assertEqual(result["source_candidates"], [])
        self.assertNotIn("10.1186/s12934-021-01717-w", payload)
        self.assertTrue(result["no_solved_claim"])
        self.assertNotIn("rxn_smiles", payload)
        self.assertNotIn("reaction_smiles", payload)

    def test_planner_validates_semisynthesis_anchor_before_recursive_small_molecule_expansion(self):
        target = TargetInput(target_name="9-OH-4-HP", target_smiles=C22_9OH_4HP_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_route_expansion_subgoal_runs": 4},
        )
        result = build_target_side_disconnection_hypotheses(
            target_smiles=C22_9OH_4HP_SMILES,
            target_name="9-OH-4-HP",
            family_hint="9,21-dihydroxy-20-methyl-pregna-4-en-3-one steroid",
            case_id="target1_steroid",
        )
        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={
                    "schema_version": "agent_action.v1",
                    "action_id": "generate:semisynthesis",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "classify target route scope",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "semisynthesis anchors are recorded",
                    "payload": {},
                },
                action_result={"accepted": True, "result": result, "reasons": []},
                round_index=1,
                run_dir=tmp,
            )
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:should_wait",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "parent_smiles": C22_9OH_4HP_SMILES,
                "precursor_smiles": "CC=O",
                "recursive_depth": 1,
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types[0], "search_literature")
        self.assertEqual(batch["actions"][0]["payload"]["search_intent"], "route_objective_endpoint_validation")
        self.assertIn("biotransformation endpoint", " ".join(batch["actions"][0]["payload"]["search_queries"]))
        self.assertNotIn("10.1186/s12934-021-01717-w", " ".join(batch["actions"][0]["payload"]["search_queries"]))
        self.assertNotIn("expand_child_target", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        self.assertTrue(board["current_belief"]["constraints"]["objective_evidence_validation_required"])

    def test_failed_run_replay_generates_bridge_tasks_and_no_solved_verdict(self):
        prior_artifacts = {
            "route_verifier": {
                "schema_version": "harness_route_verifier_report.v1",
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump"}],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="MLA analog",
                target_smiles=MLA_LIKE_SMILES,
                family_hint="MLA alkaloid",
                output_dir=tmp,
                max_rounds=1,
                use_codex_action_planner=False,
                prior_artifacts=prior_artifacts,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "mla_analog",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_no_online_sources"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )
            board = json.loads((Path(tmp) / "agent_blackboard.json").read_text(encoding="utf-8"))
            audit = json.loads((Path(tmp) / "agentic_run_audit.json").read_text(encoding="utf-8"))
            bundle = json.loads((Path(tmp) / "artifact_bundle.json").read_text(encoding="utf-8"))

        action_types = [row["action_type"] for row in result["action_batches"][0]["actions"]]
        task_types = {row["task_type"] for row in board["bridge_tasks"]}
        self.assertIn("generate_disconnection_hypotheses", action_types)
        self.assertIn("build_failure_critic_report", action_types)
        self.assertIn("search_literature", action_types)
        self.assertIn("target_proximal_bridge_required", task_types)
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertNotEqual(result["final_verdict"]["verdict"], "solved")
        self.assertEqual(audit["artifact_type"], "AgenticRunAudit")
        self.assertEqual(audit["payload"]["schema_version"], "agentic_blackboard_run_audit.v1")
        self.assertEqual(audit["payload"]["final_verdict"]["verdict"], result["final_verdict"]["verdict"])
        self.assertTrue(audit["payload"]["safety_invariants"]["parent_proof_required_for_solved"])
        self.assertIn("no_deterministic_parent_route_proof", audit["payload"]["unresolved_reasons"])
        self.assertEqual(audit["payload"]["source_acquisition_summary"]["fallback_order"], ["codex_online", "local_pdf", "placeholder"])
        self.assertTrue(audit["payload"]["source_acquisition_summary"]["codex_online_attempted"])
        self.assertTrue(audit["payload"]["source_acquisition_summary"]["placeholder_used"])
        self.assertEqual(audit["payload"]["source_acquisition_summary"]["real_source_count"], 0)
        followup_types = {row["task_type"] for row in audit["payload"]["followup_tasks"]}
        self.assertIn("continue_bridge_task", followup_types)
        transition = audit["payload"]["blackboard_transition_summary"]
        self.assertEqual(transition["schema_version"], "agent_blackboard_transition_summary.v1")
        self.assertEqual(transition["action_transition_count"], len(board["action_history"]))
        self.assertGreaterEqual(transition["changed_transition_count"], 1)
        self.assertTrue(transition["no_solved_claim"])
        self.assertIn("bridge_tasks", transition["changed_blackboard_fields"])
        self.assertTrue(audit["payload"]["round_summaries"][0]["changed_blackboard_fields"])
        self.assertIn(
            "literature_scout_report",
            audit["payload"]["typed_artifact_validation_summary"]["accepted_artifact_keys"],
        )
        self.assertIn("agentic_capability_audit", bundle["artifacts"])
        self.assertEqual(bundle["artifacts"]["agentic_capability_audit"]["artifact_type"], "AgenticCapabilityAudit")
        self.assertTrue(bundle["artifacts"]["agentic_capability_audit"]["payload"]["accepted"])
        capability_checks = {
            row["requirement_id"]: row
            for row in bundle["artifacts"]["agentic_capability_audit"]["payload"]["requirement_checks"]
        }
        self.assertTrue(
            capability_checks["artifact_refs_and_typed_validation_integrity"]["accepted"],
            capability_checks["artifact_refs_and_typed_validation_integrity"]["reasons"],
        )
        self.assertTrue(
            capability_checks["blackboard_transition_history_audited"]["accepted"],
            capability_checks["blackboard_transition_history_audited"]["reasons"],
        )
        self.assertIn("agent_blackboard_snapshot", bundle["artifacts"])
        self.assertEqual(bundle["artifacts"]["agent_blackboard_snapshot"]["artifact_type"], "AgentBlackboardSnapshot")
        self.assertEqual(bundle["artifacts"]["agent_blackboard_snapshot"]["payload"]["schema_version"], "agent_blackboard.v1")
        self.assertIn("agent_blackboard_snapshot", result["final_verdict"]["artifact_refs"])
        self.assertIn("agentic_capability_audit", result["final_verdict"]["artifact_refs"])
        self.assertIn("agentic_run_audit", bundle["artifacts"])
        self.assertEqual(bundle["artifacts"]["agentic_run_audit"]["artifact_type"], "AgenticRunAudit")
        blackboard_validations = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agent_blackboard_snapshot"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(blackboard_validations)
        self.assertTrue(blackboard_validations[0]["accepted"], blackboard_validations[0]["reasons"])
        capability_validations = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agentic_capability_audit"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(capability_validations)
        self.assertTrue(capability_validations[0]["accepted"], capability_validations[0]["reasons"])
        audit_validations = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agentic_run_audit"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(audit_validations)
        self.assertTrue(audit_validations[0]["accepted"], audit_validations[0]["reasons"])
        final_validations = [
            row for row in bundle["validations"] if row.get("schema_version") == "agentic_final_verdict_validation.v1"
        ]
        self.assertTrue(final_validations)
        self.assertTrue(final_validations[-1]["accepted"], final_validations[-1]["reasons"])
        self.assertEqual(
            bundle["artifacts"]["agentic_final_verdict_validation"]["artifact_type"],
            "AgenticFinalVerdictValidation",
        )
        final_validation_artifact_checks = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agentic_final_verdict_validation"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(final_validation_artifact_checks)
        self.assertTrue(final_validation_artifact_checks[0]["accepted"], final_validation_artifact_checks[0]["reasons"])
        self.assertIn("agentic_run_audit", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("agentic_final_verdict_validation", result["agent_blackboard"]["artifact_refs"])

    def test_failure_critic_bias_enters_blackboard_and_duplicate_critic_is_stale(self):
        target = TargetInput(target_name="critic_target", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        report = {
            "schema_version": "failure_critic_report.v1",
            "accepted": True,
            "case_id": "critic_target",
            "source_reasons": ["large_atom_jump"],
            "route_failures": [
                {
                    "schema_version": "agent_route_failure.v1",
                    "reason": "large_atom_jump",
                    "summary": "jump",
                }
            ],
            "bridge_tasks": [
                {
                    "schema_version": "agent_bridge_task.v1",
                    "task_id": "target_proximal_bridge_required:critic_target",
                    "task_type": "target_proximal_bridge_required",
                    "status": "open",
                }
            ],
            "terminal_blacklist": [],
            "blocked_directions": [
                {
                    "schema_version": "agent_blocked_direction.v1",
                    "direction": "current_route_family_without_core_bridge",
                    "reason": "large_atom_jump",
                }
            ],
            "next_action_bias": ["generate_disconnection_hypotheses", "search_literature"],
            "constraints": {"target_core_retention_required": True, "max_unexplained_heavy_atom_jump": 12},
            "no_solved_claim": True,
        }
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "critic:1",
            "action_type": "build_failure_critic_report",
            "rationale": "normalize failure",
            "expected_artifact": "failure_critic_report.v1",
            "success_condition": "bridge task",
            "payload": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action=action,
                action_result={"accepted": True, "result": report},
                round_index=1,
                run_dir=tmp,
            )
            board = update_blackboard_from_action(
                board,
                action={**action, "action_id": "critic:2"},
                action_result={"accepted": True, "result": report},
                round_index=2,
                run_dir=tmp,
            )

        belief = board["current_belief"]
        self.assertIn("generate_disconnection_hypotheses", belief["next_action_bias"])
        self.assertIn("search_literature", belief["next_action_bias"])
        self.assertEqual(belief["constraints"]["max_unexplained_heavy_atom_jump"], 12)
        self.assertTrue(board["action_history"][0]["useful_artifact"])
        self.assertFalse(board["action_history"][1]["useful_artifact"])
        self.assertTrue(board["action_history"][1]["stale"])

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertIn("search_literature", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_literature_scout_default_timeout_is_action_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = TargetInput(target_name="steroid", target_smiles="CCO")
            preflight = run_preflight(target)
            board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(open_research_timeout_s=900.0),
            )

            task = _codex_literature_scout_task(
                blackboard=board,
                state=state,
                payload={},
                max_sources=3,
            )

        self.assertEqual(task.budget.timeout_s, 180.0)
        self.assertIn("web_search", task.allowed_tools)
        self.assertIn("browser", task.allowed_tools)
        self.assertGreater(task.budget.max_tool_calls, 0)
        self.assertTrue(_task_allows_cli_search(task))

    def test_local_pdf_cache_match_prefers_exact_doi_over_same_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wrong_pdf = tmp_path / "Angew Total Synthesis of Ouabagenin and Ouabain.pdf"
            right_pdf = tmp_path / "Asian Journal Total Synthesis of Ouabagenin and Ouabain.pdf"
            wrong_pdf.write_bytes(b"%PDF-1.4\nDOI: 10.1002/anie.200704959\n")
            right_pdf.write_bytes(b"%PDF-1.4\nDOI: 10.1002/asia.200800429\n")
            target = TargetInput(target_name="same_title_doi_binding", target_smiles="CCO")
            target_input = target.to_dict()
            target_input["local_literature_cache"] = [
                {
                    "source_ref": "doi:10.1002/anie.200704959",
                    "doi": "10.1002/anie.200704959",
                    "title": "Total Synthesis of Ouabagenin and Ouabain",
                    "local_pdf": str(wrong_pdf),
                    "source_role": "auto_local_pdf_cache",
                },
                {
                    "source_ref": "doi:10.1002/asia.200800429",
                    "doi": "10.1002/asia.200800429",
                    "title": "Total Synthesis of Ouabagenin and Ouabain",
                    "local_pdf": str(right_pdf),
                    "source_role": "auto_local_pdf_cache",
                },
            ]
            preflight = run_preflight(target)
            state = ToolExecutionState(
                run_dir=tmp_path,
                target_input=target_input,
                preflight=preflight,
            )

            report = _local_pdf_cache_match_report(
                codex_report={
                    "source_candidates": [
                        {
                            "source_ref": "doi:10.1002/asia.200800429",
                            "doi": "10.1002/asia.200800429",
                            "title": "Total Synthesis of Ouabagenin and Ouabain",
                            "url": "https://doi.org/10.1002/asia.200800429",
                        }
                    ]
                },
                state=state,
                payload={},
                max_sources=3,
            )

        self.assertTrue(report["accepted"], report["reasons"])
        self.assertEqual(len(report["source_candidates"]), 1)
        candidate = report["source_candidates"][0]
        self.assertEqual(candidate["local_pdf"], str(right_pdf.resolve()))
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "doi")
        self.assertEqual(candidate["local_pdf_match"]["cache_doi"], "10.1002/asia.200800429")

    def test_search_literature_uses_codex_online_source_before_fallbacks(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "online_case",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "online:search",
                        "action_type": "search_literature",
                        "rationale": "online source scout",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "real source candidate",
                        "payload": _test_search_payload("pdf fallback local literature source"),
                    }
                ],
            }

        codex_scout = {
            "schema_version": "literature_scout_report.v1",
            "accepted": True,
            "case_id": "online_case",
            "source_candidates": [
                {
                    "schema_version": "literature_source_candidate.v1",
                    "candidate_id": "src1",
                    "source_ref": "doi:10.1000/example",
                    "title": "Example target-proximal steroid synthesis",
                    "doi": "10.1000/example",
                    "url": "https://doi.org/10.1000/example",
                    "local_pdf": "",
                    "source_type": "journal_article",
                    "relevance_rationale": "target-proximal source",
                    "expected_scheme_or_compound_labels": ["1", "2"],
                    "extraction_task_recommendations": ["resolve_source_material_or_provide_pdf"],
                    "access_status": "metadata_only",
                    "no_solved_claim": True,
                }
            ],
            "source_refs": ["doi:10.1000/example"],
            "search_queries": ["example query"],
            "reasons": [],
            "limitations": [],
            "no_solved_claim": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="online_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"codex_literature_scout": codex_scout},
            )
            evidence = result["agent_blackboard"]["literature_evidence"]
            self.assertEqual(evidence["source_discovery_mode"], "codex_online")
            self.assertEqual(evidence["confidence"], "candidate")
            self.assertEqual(evidence["source_candidates"][0]["doi"], "10.1000/example")
            self.assertEqual(len(evidence["local_pdf_proxy_requests"]), 1)
            self.assertEqual(evidence["local_pdf_proxy_requests"][0]["doi"], "10.1000/example")
            self.assertEqual(evidence["local_pdf_proxy_requests"][0]["content_scope"], "article")
            lifecycle = {
                row["source_key"]: row
                for row in evidence["source_lifecycle"]
            }
            self.assertEqual(lifecycle["doi:10.1000/example"]["stage"], "local_pdf_proxy_requested")
            self.assertEqual(
                lifecycle["doi:10.1000/example"]["next_recommended_stage"],
                "await_local_pdf_proxy_download",
            )
            self.assertEqual(lifecycle["doi:10.1000/example"]["counts"]["local_pdf_proxy_requests"], 1)
            queue_path = local_pdf_proxy_request_queue_path(Path(tmp))
            queued = load_pdf_requests(queue_path)
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["doi"], "10.1000/example")
            literature_sources = json.loads((Path(tmp) / "evidence" / "literature_sources.json").read_text(encoding="utf-8"))
            self.assertEqual(
                literature_sources["search_log"][0]["agent_access_status"],
                "agent_accessible_metadata_only",
            )
            proxy_audit = audit_local_pdf_proxy_fallback(run_dir=tmp)
            self.assertTrue(proxy_audit["accepted"], proxy_audit["reasons"])
            summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
            self.assertEqual(summary["local_pdf_proxy_request_count"], 1)
            self.assertEqual(summary["awaiting_local_pdf_proxy_count"], 1)
            self.assertEqual(summary["source_lifecycle_stage_counts"]["local_pdf_proxy_requested"], 1)
            followups = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["followup_tasks"]
            self.assertEqual(followups[0]["task_type"], "await_local_pdf_proxy_download")
            self.assertEqual(followups[0]["doi"], "10.1000/example")
            self.assertEqual(followups[0]["recommended_next_action"], "extract_pdf_literature_structures")
            self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])
            scout_artifact = result["artifact_bundle"]["artifacts"]["literature_scout_report"]
            self.assertEqual(scout_artifact["artifact_type"], "LiteratureScoutReport")
            self.assertEqual(scout_artifact["payload"]["source_discovery_mode"], "codex_online")
            self.assertEqual(scout_artifact["payload"]["local_pdf_proxy_request_summary"]["request_count"], 1)
            scout_validations = [
                row
                for row in result["artifact_bundle"]["validations"]
                if row.get("artifact_key") == "literature_scout_report"
            ]
            self.assertTrue(scout_validations)
            self.assertTrue(scout_validations[-1]["accepted"], scout_validations[-1]["reasons"])

    def test_local_pdf_proxy_download_manifest_is_reused_as_local_source(self):
        doi = "10.1000/proxy.download"

        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "proxy_download_reuse",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "proxy:search",
                        "action_type": "search_literature",
                        "rationale": "online scout fails, downloaded proxy PDF should be reused",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "local proxy PDF candidate",
                        "payload": _test_search_payload("proxy download reuse"),
                    }
                ],
            }

        failed_scout = {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": "proxy_download_reuse",
            "source_candidates": [],
            "source_refs": [],
            "search_queries": ["proxy download reuse"],
            "reasons": ["online_unavailable"],
            "limitations": [],
            "no_solved_claim": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "evidence" / "local_pdf_proxy" / "pdfs"
            pdf_dir.mkdir(parents=True)
            pdf_path = pdf_dir / "proxy_download.pdf"
            pdf_path.write_bytes(f"%PDF-1.4\nDOI {doi}\n%%EOF\n".encode("latin-1"))
            manifest = local_pdf_proxy_download_manifest_path(root)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "local_pdf_proxy_result.v1",
                        "request_id": "proxy-download",
                        "case_id": "proxy_download_reuse",
                        "source_ref": f"doi:{doi}",
                        "doi": doi,
                        "url": f"https://doi.org/{doi}",
                        "title": "Proxy downloaded source",
                        "status": "downloaded",
                        "accepted": True,
                        "pdf_path": str(pdf_path),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_agentic_blackboard_controller(
                target_name="proxy_download_reuse",
                target_smiles="CCO",
                output_dir=root,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"codex_literature_scout": failed_scout},
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_fallback_after_codex_failure")
        self.assertEqual(evidence["source_candidates"][0]["doi"], doi)
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf_path.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["access_status"], "local_pdf_available")
        self.assertEqual(evidence["source_lifecycle"][0]["stage"], "local_pdf_available")

    def test_planner_source_hint_can_trigger_auto_local_pdf_cache_match_after_scout_failure(self):
        doi = "10.4242/plannerhint2026"
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "planner_hint_cache",
            "round_index": 1,
            "mode": "codex_test",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "hint:search",
                    "action_type": "search_literature",
                    "rationale": "confirm planner-discovered source hint",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "source candidate or explicit unresolved source task",
                    "payload": _test_search_payload("planner hint cache confirmation"),
                }
            ],
            "planner_source_hints": [
                {
                    "schema_version": "planner_source_hint.v1",
                    "hint_id": "hint_doi",
                    "source_ref": f"doi:{doi}",
                    "title": "Planner hinted cache matched source",
                    "doi": doi,
                    "pii": "",
                    "url": f"https://doi.org/{doi}",
                    "local_pdf": "",
                    "local_ref": "",
                    "source_type": "planner_discovered_literature_metadata",
                    "relevance_rationale": "Codex planner found this DOI while selecting actions.",
                    "expected_scheme_or_compound_labels": ["1", "2"],
                    "extraction_task_recommendations": ["extract_pdf_literature_structures"],
                    "evidence_class": "planner_source_hint",
                    "allowed_use": "source_acquisition_hint_only",
                    "no_solved_claim": True,
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }
        failed_scout = {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": "planner_hint_cache",
            "source_candidates": [],
            "source_refs": [],
            "search_queries": ["planner hint cache confirmation"],
            "reasons": ["mock_online_unavailable"],
            "limitations": [],
            "no_solved_claim": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "pdf_cache"
            cache_dir.mkdir()
            pdf_path = cache_dir / "planner_hint_cache_source.pdf"
            pdf_path.write_bytes(f"%PDF-1.4\nRelated DOI {doi}\n%%EOF\n".encode("latin-1"))
            result = run_agentic_blackboard_controller(
                target_name="planner_hint_cache",
                target_smiles="CCO",
                output_dir=root / "run",
                max_rounds=1,
                local_pdf_search_dirs=[cache_dir],
                mock_tool_results={
                    "codex_action_planner": codex_batch,
                    "codex_literature_scout": failed_scout,
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["planner_source_hints"][0]["doi"], doi)
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_cache_match")
        self.assertEqual(evidence["source_candidates"][0]["doi"], doi)
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf_path.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["local_pdf_match"]["agent_discovered_doi"], doi)
        self.assertEqual(
            evidence["source_candidates"][0]["local_pdf_index"]["match_policy"],
            "agent_discovered_metadata_required",
        )
        self.assertEqual(evidence["confidence"], "candidate")
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertEqual(summary["planner_source_hint_count"], 1)
        self.assertTrue(summary["planner_source_hints_are_not_evidence"])
        self.assertEqual(summary["source_lifecycle_stage_counts"]["local_pdf_available"], 1)
        self.assertEqual(summary["auto_local_pdf_cache_match_count"], 1)
        self.assertFalse(result["final_verdict"]["solved"])

    def test_search_literature_falls_back_to_local_pdf_after_codex_failure(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_fallback",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:search",
                        "action_type": "search_literature",
                        "rationale": "online source scout with local fallback",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "local source candidate",
                        "payload": _test_search_payload("pdf merge online metadata local pdf"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_fallback",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_pdf_path=str(pdf),
                literature_pdf_source_ref="doi:10.1000/local",
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "pdf_fallback",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_failed"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_fallback")
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["doi"], "10.1000/local")
        self.assertEqual(evidence["source_candidates"][0]["source_role"], "user_provided_local_pdf_seed")
        self.assertTrue(evidence["source_candidates"][0]["user_provided_source_seed"])
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertTrue(summary["codex_online_attempted"])
        self.assertEqual(summary["user_provided_local_pdf_seed_count"], 1)
        self.assertEqual(summary["direct_local_pdf_after_codex_failure_count"], 1)
        self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])

    def test_search_literature_merges_local_pdf_when_codex_finds_metadata(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_merge",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:merge",
                        "action_type": "search_literature",
                        "rationale": "online source scout with known local PDF",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "online metadata and local PDF source are retained",
                        "payload": _test_search_payload("cache match DOI local PDF"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_merge",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_pdf_path=str(pdf),
                literature_pdf_source_ref="doi:10.1000/local",
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "pdf_merge",
                        "source_candidates": [
                            {
                                "source_ref": "doi:10.1000/local",
                                "doi": "10.1000/local",
                                "title": "Online metadata source",
                                "url": "https://doi.org/10.1000/local",
                            }
                        ],
                        "source_refs": ["doi:10.1000/local"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf")
        self.assertEqual(len(evidence["source_candidates"]), 1)
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["access_status"], "local_pdf_available")
        self.assertIn("extract_visual_literature_chain", evidence["source_candidates"][0]["extraction_task_recommendations"])

    def test_search_literature_matches_agent_discovered_doi_to_local_cache(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_cache_match",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:cache-match",
                        "action_type": "search_literature",
                        "rationale": "agent searches first, then local cache may satisfy access",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "online DOI is retained and local PDF cache is attached",
                        "payload": _test_search_payload("auto cache match DOI local PDF"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_cache_match",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_sources=[
                    {
                        "local_pdf": str(pdf),
                        "source_ref": "doi:10.1000/cache",
                        "title": "Cached local article",
                    }
                ],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "pdf_cache_match",
                        "source_candidates": [
                            {
                                "source_ref": "doi:10.1000/cache",
                                "doi": "10.1000/cache",
                                "title": "Agent discovered article",
                                "url": "https://doi.org/10.1000/cache",
                            }
                        ],
                        "source_refs": ["doi:10.1000/cache"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        candidate = evidence["source_candidates"][0]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf_cache")
        self.assertEqual(candidate["title"], "Agent discovered article")
        self.assertEqual(candidate["local_pdf"], str(pdf.resolve()))
        self.assertEqual(candidate["source_type"], "literature_metadata+local_pdf")
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "doi")
        self.assertIn("local_pdf_cache", [row["mode"] for row in evidence["scout_attempts"]])

    def test_search_literature_auto_discovers_local_pdf_cache_for_agent_doi(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "auto_pdf_cache_match",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:auto-cache-match",
                        "action_type": "search_literature",
                        "rationale": "agent discovers DOI, auto local PDF cache should attach matching file",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "auto-indexed local PDF is attached only after DOI match",
                        "payload": _test_search_payload("ScienceDirect PII local PDF match"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "papers"
            paper_dir.mkdir()
            pdf = paper_dir / "auto-source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n10.1234/auto.cache\n%%EOF\n")
            result = run_agentic_blackboard_controller(
                target_name="auto_pdf_cache_match",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                local_pdf_search_dirs=[paper_dir],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "auto_pdf_cache_match",
                        "source_candidates": [
                            {
                                "source_ref": "doi:10.1234/auto.cache",
                                "doi": "10.1234/auto.cache",
                                "title": "Agent discovered auto cache paper",
                                "url": "https://doi.org/10.1234/auto.cache",
                            }
                        ],
                        "source_refs": ["doi:10.1234/auto.cache"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        candidate = evidence["source_candidates"][0]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf_cache")
        self.assertEqual(candidate["local_pdf"], str(pdf.resolve()))
        self.assertEqual(candidate["source_type"], "literature_metadata+local_pdf")
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "doi")
        self.assertEqual(candidate["local_pdf_index"]["match_policy"], "agent_discovered_metadata_required")
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertEqual(summary["local_pdf_cache_match_count"], 1)
        self.assertEqual(summary["auto_local_pdf_cache_match_count"], 1)
        self.assertEqual(summary["agent_discovered_local_pdf_match_count"], 1)
        self.assertEqual(summary["local_pdf_match_bases"], ["doi"])
        self.assertFalse(summary["auto_local_pdf_blind_fallback_used"])
        capability_checks = {
            row["requirement_id"]: row
            for row in result["artifact_bundle"]["artifacts"]["agentic_capability_audit"]["payload"]["requirement_checks"]
        }
        self.assertTrue(
            capability_checks["codex_first_source_acquisition_audited"]["accepted"],
            capability_checks["codex_first_source_acquisition_audited"]["reasons"],
        )

    def test_search_literature_auto_pdf_cache_matches_sciencedirect_pii(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "auto_pdf_pii_match",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:auto-pii-match",
                        "action_type": "search_literature",
                        "rationale": "agent finds a ScienceDirect page, local filename PII should match",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "auto-indexed ScienceDirect PDF is attached by PII",
                        "payload": _test_search_payload("auto cache no blind fallback"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "papers"
            paper_dir.mkdir()
            pdf = paper_dir / "1-s2.0-S0040402025001668-main.pdf"
            pdf.write_bytes(b"%PDF-1.4\nmock science direct pdf\n%%EOF\n")
            result = run_agentic_blackboard_controller(
                target_name="auto_pdf_pii_match",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                local_pdf_search_dirs=[paper_dir],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "auto_pdf_pii_match",
                        "source_candidates": [
                            {
                                "source_ref": "sciencedirect:S0040402025001668",
                                "title": "Agent discovered ScienceDirect article",
                                "url": "https://www.sciencedirect.com/science/article/pii/S0040402025001668",
                            }
                        ],
                        "source_refs": ["sciencedirect:S0040402025001668"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        candidate = evidence["source_candidates"][0]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf_cache")
        self.assertEqual(candidate["local_pdf"], str(pdf.resolve()))
        self.assertEqual(candidate["pii"], "S0040402025001668")
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "pii")
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertEqual(summary["auto_local_pdf_cache_match_count"], 1)
        self.assertEqual(summary["local_pdf_match_bases"], ["pii"])
        self.assertEqual(summary["agent_discovered_local_pdf_match_count"], 1)
        self.assertFalse(summary["auto_local_pdf_blind_fallback_used"])

    def test_auto_local_pdf_cache_is_not_blind_fallback_after_online_miss(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "auto_pdf_no_blind_fallback",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:auto-no-blind-fallback",
                        "action_type": "search_literature",
                        "rationale": "online scout failed, auto local cache should not be used blindly",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "placeholder is emitted instead of arbitrary auto cache PDF",
                        "payload": _test_search_payload("local cache fallback after online failure"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "papers"
            paper_dir.mkdir()
            (paper_dir / "unmatched-source.pdf").write_bytes(b"%PDF-1.4\n10.1234/unmatched.cache\n%%EOF\n")
            result = run_agentic_blackboard_controller(
                target_name="auto_pdf_no_blind_fallback",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                local_pdf_search_dirs=[paper_dir],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "auto_pdf_no_blind_fallback",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_no_hit"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "placeholder")
        self.assertTrue(evidence["source_candidates"][0]["placeholder_only"])
        self.assertFalse(str(evidence["source_candidates"][0].get("local_pdf") or "").strip())

    def test_capability_audit_rejects_auto_local_pdf_without_agent_match_provenance(self):
        target = TargetInput(target_name="bad_auto_cache", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [{"mode": "local_pdf_cache", "attempted": True}]
        board["literature_evidence"]["source_discovery_mode"] = "codex_online+local_pdf_cache"
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1234/bad",
                "doi": "10.1234/bad",
                "local_pdf": "/tmp/bad.pdf",
                "source_discovery_mode": "codex_online+local_pdf_cache",
                "local_pdf_index": {
                    "schema_version": "auto_local_pdf_index.v1",
                    "match_policy": "agent_discovered_metadata_required",
                },
                "no_solved_claim": True,
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "status": "accepted",
            }
        ]

        check = _capability_check_source_acquisition(board)

        self.assertFalse(check["accepted"])
        self.assertIn("local_pdf_cache_match_missing_provenance:0", check["reasons"])
        self.assertIn("auto_local_pdf_cache_without_agent_discovered_match:0", check["reasons"])

    def test_capability_audit_rejects_direct_local_pdf_without_codex_attempt_or_user_seed(self):
        target = TargetInput(target_name="bad_direct_pdf", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [{"mode": "local_pdf", "attempted": True, "accepted": True}]
        board["literature_evidence"]["source_discovery_mode"] = "local_pdf_fallback"
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1234/direct",
                "doi": "10.1234/direct",
                "local_pdf": "/tmp/direct.pdf",
                "source_discovery_mode": "local_pdf_fallback",
                "no_solved_claim": True,
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "status": "accepted",
            }
        ]

        check = _capability_check_source_acquisition(board)

        self.assertFalse(check["accepted"])
        self.assertIn("local_pdf_fallback_without_codex_online_attempt:0", check["reasons"])
        self.assertIn("direct_local_pdf_fallback_missing_user_seed_marker:0", check["reasons"])

    def test_capability_audit_rejects_metadata_only_source_without_pdf_proxy_request(self):
        target = TargetInput(target_name="metadata_only_gap", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [
            {"mode": "codex_online", "attempted": True, "accepted": True}
        ]
        board["literature_evidence"]["source_discovery_mode"] = "codex_online"
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1234/metadata.only",
                "doi": "10.1234/metadata.only",
                "url": "https://doi.org/10.1234/metadata.only",
                "access_status": "metadata_only",
                "no_solved_claim": True,
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "status": "accepted",
            }
        ]

        check = _capability_check_source_acquisition(board)

        self.assertFalse(check["accepted"])
        self.assertIn("metadata_only_source_without_local_pdf_proxy_request:0", check["reasons"])

    def test_local_pdf_cache_falls_back_after_online_scout_has_no_source(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_cache_no_online_hit",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:cache-no-hit",
                        "action_type": "search_literature",
                        "rationale": "online source failed, local cache should be tried before placeholder",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "local PDF fallback source before placeholder",
                        "payload": _test_search_payload("placeholder after online and local fail"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_cache_no_online_hit",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_sources=[{"local_pdf": str(pdf), "source_ref": "doi:10.1000/cache"}],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "pdf_cache_no_online_hit",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_no_hit"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_fallback_after_codex_failure")
        self.assertFalse(evidence["source_candidates"][0].get("placeholder_only", False))
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["source_discovery_mode"], "local_pdf_fallback_after_codex_failure")
        cache_attempts = [row for row in evidence["scout_attempts"] if row.get("mode") == "local_pdf_cache"]
        self.assertTrue(cache_attempts[0]["accepted"])

    def test_search_literature_writes_placeholder_only_after_online_and_local_fail(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "placeholder_case",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "placeholder:search",
                        "action_type": "search_literature",
                        "rationale": "record missing source",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "placeholder if all source access fails",
                        "payload": _test_search_payload("placeholder after online and local fail"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="placeholder_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "placeholder_case",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_failed"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["confidence"], "placeholder")
        self.assertTrue(evidence["source_candidates"][0]["placeholder_only"])
        self.assertFalse(result["agent_blackboard"]["action_history"][0]["useful_artifact"])
        scout_artifact = result["artifact_bundle"]["artifacts"]["literature_scout_report"]
        self.assertEqual(scout_artifact["artifact_type"], "LiteratureScoutReport")
        self.assertFalse(scout_artifact["payload"]["accepted"])
        self.assertTrue(scout_artifact["payload"]["placeholder_only"])
        scout_validations = [
            row
            for row in result["artifact_bundle"]["validations"]
            if row.get("artifact_key") == "literature_scout_report"
        ]
        self.assertTrue(scout_validations)
        self.assertTrue(scout_validations[-1]["accepted"], scout_validations[-1]["reasons"])
        self.assertNotIn("typed_artifact_validation_failed:literature_scout_report", result["artifact_bundle"]["safety_flags"])

    def test_parent_proof_mock_is_required_for_agentic_solved(self):
        proof = {
            "schema_version": "stitched_parent_route_proof.v1",
            "accepted": True,
            "solved": True,
            "route_status": "solved",
            "reasons": [],
        }
        stitch_payload = {
            "proof_binding": {
                "schema_version": "agentic_parent_stitch_binding.v1",
                "child_route_ref": "mock:child_route",
                "parent_route_ref": "mock:parent_route",
                "exact_literature_segment_ref": "mock:exact_segment",
                "exact_literature_row_ids": ["source_detail_exact_step:mock"],
                "input_refs": ["mock:child_route", "mock:parent_route", "mock:exact_segment"],
                "missing_inputs": [],
            },
            "proof_policy": {
                "schema_version": "agentic_parent_stitch_policy.v1",
                "target_equivalence_required": True,
                "parent_route_verifier_required": True,
                "stock_audit_required": True,
                "no_unexplained_large_atom_jump_required": True,
                "child_route_connectivity_required": True,
                "exact_literature_connectivity_required": True,
                "analogy_is_not_proof": True,
                "child_route_cannot_promote_parent": True,
                "final_verdict_authority": "deterministic_parent_route_proof",
            },
        }

        def planner(**kwargs):
            del kwargs
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "proof_case",
                "round_index": 1,
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "proof:stitch",
                        "action_type": "stitch_parent_route",
                        "rationale": "mock accepted parent proof",
                        "expected_artifact": "stitched_parent_route_proof.v1",
                        "success_condition": "parent proof accepted",
                        "payload": stitch_payload,
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="proof_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"stitch_parent_route": {"accepted": True, "result": {"parent_route_proof": proof}}},
            )

        self.assertEqual(result["final_verdict"]["verdict"], "solved")
        self.assertTrue(result["final_verdict"]["solved"])
        final_validations = [
            row
            for row in result["artifact_bundle"]["validations"]
            if row.get("schema_version") == "agentic_final_verdict_validation.v1"
        ]
        self.assertTrue(final_validations)
        self.assertTrue(final_validations[-1]["accepted"], final_validations[-1]["reasons"])
        self.assertEqual(
            result["artifact_bundle"]["artifacts"]["agentic_final_verdict_validation"]["artifact_type"],
            "AgenticFinalVerdictValidation",
        )

    def test_final_verdict_validation_rejects_solved_without_parent_proof(self):
        validation = _validate_agentic_final_verdict(
            {
                "schema_version": "codex_entry_final_verdict.v1",
                "case_id": "bad_final",
                "verdict": "solved",
                "route_status": "solved",
                "solved": True,
                "stock_audit_passed": True,
            },
            blackboard={
                "case_id": "bad_final",
                "parent_route_proof": {"accepted": False, "solved": False},
                "current_belief": {"child_route_solved": True},
            },
            validations=[],
        )

        self.assertFalse(validation["accepted"])
        self.assertIn("final_solved_without_parent_proof", validation["reasons"])
        self.assertIn("child_solved_promoted_without_parent_proof", validation["reasons"])

    def test_pdf_structure_action_updates_blackboard_without_solved_claim(self):
        def planner(**kwargs):
            round_index = kwargs["round_index"]
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_case",
                "round_index": round_index,
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": f"pdf:{round_index}",
                        "action_type": "extract_pdf_literature_structures",
                        "rationale": "local PDF source should be rendered before visual chain extraction",
                        "expected_artifact": "literature_pdf_structure_evidence.v1",
                        "success_condition": "rendered pages are available",
                        "payload": {},
                    }
                ],
            }

        pdf_result = {
            "schema_version": "literature_pdf_structure_evidence.v1",
            "accepted": True,
            "source_pdf_path": "/tmp/source.pdf",
            "rendered_pages": [{"page_number": 1, "image_path": "/tmp/page-1.png"}],
            "indexed_images": [],
            "scheme_crops": [],
            "compound_text_snippets": [],
            "summary": {
                "rendered_page_count": 1,
                "indexed_image_count": 0,
                "scheme_crop_count": 0,
                "compound_text_snippet_count": 0,
            },
            "reasons": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="pdf_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"extract_pdf_literature_structures": pdf_result},
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["pdf_structure_evidence"][0]["summary"]["rendered_page_count"], 1)
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertNotEqual(result["final_verdict"]["verdict"], "solved")

    def test_action_payload_source_context_survives_tool_output_without_source_fields(self):
        target = TargetInput(target_name="multi_pdf_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:first", "local_pdf": "/tmp/first.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:second", "local_pdf": "/tmp/second.pdf"},
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "pdf:first",
                "action_type": "extract_pdf_literature_structures",
                "rationale": "render first PDF",
                "expected_artifact": "literature_pdf_structure_evidence.v1",
                "success_condition": "rendered pages",
                "payload": {"source_ref": "doi:first", "pdf_path": "/tmp/first.pdf"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "accepted": True,
                    "rendered_pages": [{"page_number": 1, "image_path": "/tmp/first-1.png"}],
                    "summary": {"rendered_page_count": 1},
                },
            },
            round_index=1,
            run_dir="/tmp",
        )
        pdf_summary = board["literature_evidence"]["pdf_structure_evidence"][0]
        self.assertEqual(pdf_summary["source_ref"], "doi:first")
        self.assertEqual(pdf_summary["source_pdf_path"], "/tmp/first.pdf")
        self.assertEqual(pdf_summary["evidence_id"], "doi:first")
        lifecycle_by_ref = {
            row["source_ref"]: row
            for row in board["literature_evidence"]["source_lifecycle"]
        }
        self.assertEqual(lifecycle_by_ref["doi:first"]["stage"], "pdf_rendered")
        self.assertEqual(lifecycle_by_ref["doi:second"]["stage"], "local_pdf_available")
        pdf_history = board["action_history"][-1]
        self.assertEqual(pdf_history["blackboard_delta"]["pdf_structure_evidence"], 1)
        self.assertEqual(pdf_history["blackboard_delta"]["source_lifecycle"], 2)
        self.assertIn("pdf_structure_evidence", pdf_history["changed_blackboard_fields"])
        self.assertIn("source_lifecycle", pdf_history["changed_blackboard_fields"])
        self.assertEqual(pdf_history["blackboard_counts_before"]["pdf_structure_evidence"], 0)
        self.assertEqual(pdf_history["blackboard_counts_after"]["pdf_structure_evidence"], 1)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:first",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract first visual chain",
                "expected_artifact": "visual_literature_chain.v1",
                "success_condition": "visual chain or explicit gaps",
                "payload": {"source_ref": "doi:first", "pdf_path": "/tmp/first.pdf"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": True,
                    "candidate_chain": {"steps": []},
                    "candidate_quality": {},
                    "reasons": [],
                },
            },
            round_index=2,
            run_dir="/tmp",
        )
        visual_summary = board["literature_evidence"]["visual_chains"][0]
        self.assertEqual(visual_summary["source_ref"], "doi:first")
        self.assertEqual(visual_summary["source_pdf_path"], "/tmp/first.pdf")
        lifecycle_by_ref = {
            row["source_ref"]: row
            for row in board["literature_evidence"]["source_lifecycle"]
        }
        self.assertEqual(lifecycle_by_ref["doi:first"]["stage"], "visual_extracted")
        visual_history = board["action_history"][-1]
        self.assertEqual(visual_history["blackboard_delta"]["visual_chains"], 1)
        self.assertIn("visual_chains", visual_history["changed_blackboard_fields"])

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_pdf_literature_structures")
        self.assertEqual(first["payload"]["source_ref"], "doi:second")
        self.assertEqual(first["payload"]["pdf_path"], "/tmp/second.pdf")

    def test_agentic_guided_payload_is_valid_chemenzy_policy(self):
        target = TargetInput(target_name="bufotalin", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "bridge:polycyclic_core",
                "task_type": "target_proximal_bridge",
                "target_handle": "polycyclic_cage_core",
                "required_bridge": "target-proximal cage intermediate",
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:10.0000/source"]
        board["analogical_hypothesis_ranking"] = {
            "selected_hypotheses": [
                {
                    "hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                    "no_solved_claim": True,
                }
            ]
        }

        payload = build_agentic_guided_payload(board)
        validation = validate_chem_enzy_search_policy(payload["search_policy"])

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(payload["search_policy"]["case_id"], board["case_id"])
        self.assertIn("doi:10.0000/source", payload["search_policy"]["evidence_refs"])
        self.assertEqual(payload["search_policy"]["rerun_reason"], "agentic_blackboard_bridge_tasks_available")

    def test_analogical_template_validation_rejects_raw_and_missing_scope_gap(self):
        template = {
            "schema_version": "analogical_reaction_template.v1",
            "template_id": "bad_tpl",
            "relation_type": "analog",
            "reaction_class": "esterification",
            "mechanistic_class": "acyl_substitution",
            "reaction_center": {"product_retron_type": "aryl_ester_acyl_oxygen"},
            "template_radius": "r1",
            "source_refs": ["doi:analog"],
            "confidence": "medium",
            "no_solved_claim": True,
            "not_raw_reaction_injection": True,
            "rxn_smiles": "CCO>>CC=O",
        }

        validation = validate_analogical_reaction_template(template)

        self.assertFalse(validation["accepted"])
        self.assertIn("analog_template_missing_scope_gap", validation["reasons"])
        self.assertIn("raw_reaction_injection", validation["reasons"])

    def test_analogical_template_extract_rank_and_apply_to_aryl_ester_target(self):
        target = TargetInput(target_name="MLA analog", target_smiles=MLA_LIKE_SMILES, family_hint="MLA alkaloid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {
            "hypotheses": [
                {
                    "hypothesis_id": "h_aryl_ester",
                    "target_handle": "aryl_ester_or_anthranilate_sidechain",
                    "proposed_disconnection_region": "aryl ester",
                }
            ]
        }
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["literature_evidence"]["source_refs"] = ["doi:analog"]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=preflight["case_id"],
            target_smiles=MLA_LIKE_SMILES,
        )
        board["analogical_templates"] = extracted["templates"]
        ranking = rank_analogical_reaction_templates_from_blackboard(board)
        board["analogical_template_ranking"] = ranking
        applied = apply_analogical_templates_to_target(blackboard=board, target_smiles=MLA_LIKE_SMILES)

        self.assertTrue(extracted["accepted"], extracted["reasons"])
        self.assertTrue(ranking["accepted"], ranking["reasons"])
        self.assertTrue(applied["accepted"], applied["reasons"])
        self.assertEqual(applied["executable_candidate_count"], 1)
        self.assertTrue(applied["no_solved_claim"])
        self.assertNotEqual(applied["applications"][0].get("route_status"), "solved")
        self.assertNotEqual(applied["applications"][0].get("verdict"), "solved")

    def test_analogical_template_extracts_steroid_core_advisory_seed(self):
        target = TargetInput(target_name="steroid target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["analogical_hypotheses"] = [
            {
                "hypothesis_id": "h_core",
                "target_handle": "polycyclic_cage_core",
                "proposed_disconnection_region": "peripheral functionalization while retaining the steroid core",
                "expected_precursor_type": "target-proximal same-core steroid intermediate",
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:steroid"]
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:steroid",
                "title": "Analog steroid route",
                "source_type": "reaction_precedent",
                "relevance_rationale": "steroid family precedent",
            }
        ]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=str(preflight["case_id"]),
            target_smiles=BUFOTALIN_SMILES,
            max_templates=4,
            radius_policy="auto",
        )

        retrons = {
            (row.get("reaction_center") or {}).get("product_retron_type")
            for row in extracted["templates"]
        }
        self.assertTrue(extracted["accepted"], extracted["reasons"])
        self.assertIn("steroid_core_retention_bridge", retrons)
        self.assertTrue(extracted["no_solved_claim"])

    def test_analogical_template_applies_broad_reaction_center_hypotheses_to_target1(self):
        target1 = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
        target = TargetInput(
            target_name="target_molecule_1",
            target_smiles=target1,
            family_hint="steroid ouabagenin analog enone alcohol",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["analogical_hypotheses"] = [
            {
                "hypothesis_id": "h_core",
                "target_handle": "polycyclic_cage_core",
                "proposed_disconnection_region": "peripheral functionalization while retaining the steroid core",
                "expected_precursor_type": "target-proximal same-core steroid intermediate",
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:steroid"]
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:steroid",
                "title": "Analog steroid route",
                "source_type": "reaction_precedent",
                "relevance_rationale": "same-core steroid redox and protection precedent",
            }
        ]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=str(preflight["case_id"]),
            target_smiles=target1,
            max_templates=8,
            radius_policy="broad",
        )
        board["analogical_templates"] = extracted["templates"]
        ranking = rank_analogical_reaction_templates_from_blackboard(board)
        board["analogical_template_ranking"] = ranking
        applied = apply_analogical_templates_to_target(
            blackboard=board,
            target_smiles=target1,
            confidence_threshold="low",
        )

        retrons = {
            (row.get("reaction_center") or {}).get("product_retron_type")
            for row in extracted["templates"]
        }
        self.assertIn("steroid_core_retention_bridge", retrons)
        self.assertIn("steroid_carbonyl_redox_adjustment", retrons)
        self.assertIn("steroid_alcohol_protection_redox_adjustment", retrons)
        self.assertTrue(applied["accepted"], applied["reasons"])
        self.assertGreaterEqual(applied["accepted_application_count"], 1)
        self.assertEqual(applied["executable_candidate_count"], 0)
        accepted = [row for row in applied["applications"] if row.get("accepted")]
        self.assertTrue(any(row.get("allowed_use") == "hypothesis_only_not_solved" for row in accepted))
        self.assertTrue(all(row.get("no_solved_claim") for row in accepted))
        self.assertTrue(all(row.get("not_parent_route_proof") for row in accepted))
        precursor_hints = [
            hint
            for row in accepted
            for hint in row.get("hypothetical_precursor_hints") or []
            if isinstance(hint, dict)
        ]
        self.assertGreaterEqual(len(precursor_hints), 1)
        self.assertTrue(all(hint.get("allowed_use") == "guided_search_subgoal_hint_only" for hint in precursor_hints))
        self.assertTrue(all(hint.get("not_parent_route_proof") for hint in precursor_hints))
        self.assertTrue(all(Chem.MolFromSmiles(str(hint.get("precursor_smiles") or "")) is not None for hint in precursor_hints))
        self.assertIn(
            "same_core_redox_or_protection_state_precursor",
            {hint.get("candidate_kind") for hint in precursor_hints},
        )

    def test_exploratory_visual_chain_drives_templates_not_exact_compile(self):
        target1 = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
        visual_precursor = "O=C1CCC2(C)C(=CCC3(O)C2CCC4(C)C3CCC4C(CO)C)C=C1"
        target = TargetInput(target_name="target_molecule_1", target_smiles=target1, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {
            "hypotheses": [{"hypothesis_id": "h_core", "target_handle": "polycyclic_cage_core"}]
        }
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "cortistatin_total_synthesis",
                "local_pdf": "/tmp/cortistatin.pdf",
                "source_type": "literature_metadata+local_pdf",
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {
                "schema_version": "agent_pdf_structure_evidence_summary.v1",
                "source_ref": "cortistatin_total_synthesis",
                "accepted": True,
            }
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "cortistatin_total_synthesis",
                "accepted": True,
                "candidate_step_count": 1,
                "acceptance_level": "exploratory_connectivity_candidate",
                "exact_ready": False,
                "exploratory_accepted": True,
                "steps": [
                    {
                        "step_id": "step_26_to_1",
                        "product_smiles": "O=C1CCC2(C)C(CCC3(O)C2CCC4(C)C3CCC4C(CO)C)=C1",
                        "reactant_smiles": [visual_precursor],
                        "allowed_use": "exploratory_template_and_guided_hint_only",
                        "not_exact_literature_segment": True,
                        "stereochemistry_status": "unspecified_or_partial",
                        "risk_flags": ["stereochemistry_unspecified"],
                    }
                ],
            }
        ]
        board["action_history"] = [
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
        ]
        board["budget_state"]["visual_calls"] = 2

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]
        guided_payload = build_agentic_guided_payload(board)

        self.assertNotIn("compile_exact_literature_rows", action_types)
        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertIn(visual_precursor, guided_payload["search_policy"]["source_budget"]["preferred_precursor_smiles"])
        self.assertIn(visual_precursor, guided_payload["search_policy"]["preferred_subgoal"]["preferred_subgoals"])
        self.assertTrue(guided_payload["search_policy"]["source_budget"]["visual_connectivity_hints_are_not_proof"])
        validation = validate_chem_enzy_search_policy(guided_payload["search_policy"])
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_visual_connectivity_candidate_becomes_low_confidence_template_hint(self):
        target1 = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
        visual_precursor = "O=C1CCC2(C)C(=CCC3(O)C2CCC4(C)C3CCC4C(CO)C)C=C1"
        visual_precursor_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(visual_precursor), isomericSmiles=True)
        target = TargetInput(target_name="target_molecule_1", target_smiles=target1, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["literature_evidence"]["source_refs"] = ["cortistatin_total_synthesis"]
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "cortistatin_total_synthesis", "source_type": "reaction_precedent"}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "source_ref": "cortistatin_total_synthesis",
                "accepted": True,
                "candidate_step_count": 1,
                "acceptance_level": "exploratory_connectivity_candidate",
                "exact_ready": False,
                "exploratory_accepted": True,
                "steps": [
                    {
                        "product_smiles": "O=C1CCC2(C)C(CCC3(O)C2CCC4(C)C3CCC4C(CO)C)=C1",
                        "reactant_smiles": [visual_precursor],
                        "source_locator": "Scheme 2, compound 26 to 1",
                        "allowed_use": "exploratory_template_and_guided_hint_only",
                        "not_exact_literature_segment": True,
                    }
                ],
            }
        ]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=str(preflight["case_id"]),
            target_smiles=target1,
            max_templates=8,
            radius_policy="broad",
        )
        board["analogical_templates"] = extracted["templates"]
        ranking = rank_analogical_reaction_templates_from_blackboard(board)
        board["analogical_template_ranking"] = ranking
        applied = apply_analogical_templates_to_target(
            blackboard=board,
            target_smiles=target1,
            confidence_threshold="low",
        )

        visual_templates = [
            row
            for row in extracted["templates"]
            if (row.get("reaction_center") or {}).get("product_retron_type") == "steroid_visual_unsaturation_adjustment"
        ]
        self.assertTrue(visual_templates)
        self.assertEqual(visual_templates[0]["visual_connectivity_hint"]["precursor_smiles"], visual_precursor_canonical)
        accepted = [row for row in applied["applications"] if row.get("accepted")]
        precursor_hints = [
            hint
            for row in accepted
            for hint in row.get("hypothetical_precursor_hints") or []
            if isinstance(hint, dict)
        ]
        self.assertTrue(any(hint.get("precursor_smiles") == visual_precursor_canonical for hint in precursor_hints))
        self.assertTrue(all(hint.get("not_exact_literature_segment") for hint in precursor_hints))

    def test_planner_selects_analogical_template_actions_before_guided(self):
        target = TargetInput(target_name="MLA analog", target_smiles=MLA_LIKE_SMILES, family_hint="MLA alkaloid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {
            "hypotheses": [
                {
                    "hypothesis_id": "h_aryl_ester",
                    "target_handle": "aryl_ester_or_anthranilate_sidechain",
                    "proposed_disconnection_region": "aryl ester",
                }
            ]
        }
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_aryl_ester"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:analog",
                "title": "Analog esterification precedent",
                "doi": "10.1000/analog",
                "source_type": "reaction_precedent",
                "relevance_rationale": "analog aryl ester precedent",
                "access_status": "metadata_only",
                "no_solved_claim": True,
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:analog"]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        template_action = next(row for row in batch["actions"] if row["action_type"] == "extract_analogical_reaction_templates")
        self.assertTrue(template_action["payload"]["analogical_template_policy"]["analogy_is_advisory_only"])
        self.assertEqual(
            template_action["payload"]["analogical_template_policy"]["final_verdict_authority"],
            "deterministic_parent_route_proof",
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_agentic_controller_compiles_validated_analogical_template_without_final_solved(self):
        def planner(**kwargs):
            round_index = kwargs["round_index"]
            actions_by_round = {
                1: ["generate_disconnection_hypotheses"],
                2: ["extract_analogical_reaction_templates"],
                3: ["rank_analogical_reaction_templates"],
                4: ["apply_analogical_template_to_target"],
                5: ["validate_template_application"],
            }
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "mla_analog",
                "round_index": round_index,
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": f"r{round_index}:{action_type}",
                        "action_type": action_type,
                        "rationale": "template test",
                        "expected_artifact": "typed template artifact",
                        "success_condition": "typed artifact or explicit rejection",
                        "payload": {
                            **_test_analogical_template_payload(action_type),
                            "max_templates": 4,
                            "max_applications": 3,
                            "template_radius_policy": "auto",
                            "analog_template_confidence_threshold": "low",
                        },
                    }
                    for action_type in actions_by_round.get(round_index, [])
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="MLA analog",
                target_smiles=MLA_LIKE_SMILES,
                family_hint="MLA alkaloid",
                output_dir=tmp,
                max_rounds=5,
                action_planner=planner,
            )
            compiled = json.loads((Path(tmp) / "compiled_analogical_template_hints.json").read_text(encoding="utf-8"))
            disabled_exact_plugin = json.loads(
                (Path(tmp) / "compiled_literature_template_plugin.json").read_text(encoding="utf-8")
            )
            audit = json.loads((Path(tmp) / "agentic_run_audit.json").read_text(encoding="utf-8"))

        board = result["agent_blackboard"]
        self.assertGreaterEqual(len(board["analogical_templates"]), 1)
        self.assertGreaterEqual(len(board["template_applications"]), 1)
        self.assertEqual(compiled["schema_version"], "analogical_template_guided_hints.v1")
        self.assertTrue(compiled["analogy_is_advisory_only"])
        self.assertTrue(compiled["not_exact_literature_segment"])
        self.assertTrue(compiled["not_parent_route_proof"])
        self.assertFalse(compiled["literature_template_plugin"]["plugin_flags"]["enabled"])
        self.assertEqual(compiled["literature_template_plugin"]["one_step_rows"], [])
        self.assertFalse(disabled_exact_plugin["plugin_flags"]["enabled"])
        self.assertEqual(disabled_exact_plugin["one_step_rows"], [])
        hints = compiled["analogical_template_hints"]
        self.assertFalse(hints["plugin_flags"]["enabled"])
        self.assertTrue(hints["plugin_flags"]["guided_hint_enabled"])
        self.assertEqual(len(hints["one_step_rows"]), 1)
        self.assertEqual(hints["one_step_rows"][0]["allowed_use"], "guided_search_hint_only")
        self.assertEqual(hints["one_step_rows"][0]["source_policy_decision"], "analogical_guided_hint_only")
        self.assertTrue(hints["one_step_rows"][0]["not_exact_literature_segment"])
        self.assertFalse(hints["one_step_rows"][0]["used_as_proof"])
        self.assertNotIn("compiled_downstream", result["artifact_bundle"]["artifacts"])
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertNotEqual(result["final_verdict"]["verdict"], "solved")
        artifacts = result["artifact_bundle"]["artifacts"]
        self.assertEqual(artifacts["analogical_reaction_template_report"]["artifact_type"], "AnalogicalReactionTemplateReport")
        self.assertEqual(artifacts["analogical_reaction_template_ranking_artifact"]["artifact_type"], "AnalogicalReactionTemplateRanking")
        self.assertEqual(artifacts["analogical_template_application_report"]["artifact_type"], "AnalogicalTemplateApplicationReport")
        self.assertEqual(
            artifacts["analogical_template_application_validation_artifact"]["artifact_type"],
            "AnalogicalTemplateApplicationValidation",
        )
        accepted_keys = audit["payload"]["typed_artifact_validation_summary"]["accepted_artifact_keys"]
        self.assertIn("analogical_reaction_template_report", accepted_keys)
        self.assertIn("analogical_reaction_template_ranking", accepted_keys)
        self.assertIn("analogical_template_application_report", accepted_keys)
        self.assertIn("analogical_template_application_validation", accepted_keys)
        self.assertTrue(audit["payload"]["analogical_template_summary"]["analogy_is_advisory_only"])
        self.assertEqual(audit["payload"]["analogical_template_summary"]["final_verdict_authority"], "none")
        self.assertEqual(audit["payload"]["analogical_template_summary"]["validated_one_step_row_count"], 1)

    def test_guided_payload_carries_template_hints_and_forbidden_ids(self):
        target = TargetInput(target_name="MLA analog", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:ester", "task_type": "target_proximal_bridge"}]
        board["analogical_template_ranking"] = {
            "selected_templates": [{"template_id": "tpl_good", "no_solved_claim": True}]
        }
        board["template_applications"] = [
            {
                "application_id": "app_good",
                "template_id": "tpl_good",
                "accepted": True,
                "allowed_use": "executable_candidate",
                "product_retron_type": "aryl_ester_acyl_oxygen",
                "executable_candidate_available": True,
            },
            {
                "application_id": "app_hypothesis",
                "template_id": "tpl_hypothesis",
                "accepted": True,
                "allowed_use": "hypothesis_only_not_solved",
                "product_retron_type": "steroid_carbonyl_redox_adjustment",
                "executable_candidate_available": False,
                "hypothetical_route_hypothesis": {
                    "schema_version": "analogical_route_hypothesis.v1",
                    "route_status": "hypothesis_only_not_solved",
                    "hypothesis_type": "carbonyl_redox_adjustment",
                    "reaction_center_idea": "transfer analog steroid redox logic",
                    "expected_precursor_type": "same-core hydroxy steroid",
                    "template_application": "prefer late-stage redox variants",
                    "required_verification": ["route_verifier", "parent_route_proof"],
                    "risk_flags": ["selectivity_not_proven"],
                    "no_solved_claim": True,
                },
                "hypothetical_precursor_hints": [
                    {
                        "schema_version": "analogical_hypothesis_precursor_hint.v1",
                        "hint_id": "hyp_precursor_1",
                        "precursor_smiles": "CCO",
                        "precursor_role": "same_core_hydroxy_steroid_carbonyl_precursor",
                        "derived_from_retron": "steroid_carbonyl_redox_adjustment",
                        "hypothesis_type": "carbonyl_redox_adjustment",
                        "candidate_kind": "same_core_redox_or_protection_state_precursor",
                        "allowed_use": "guided_search_subgoal_hint_only",
                        "risk_flags": ["selectivity_not_proven"],
                        "not_exact_literature_segment": True,
                        "not_parent_route_proof": True,
                        "requires_verifier": True,
                        "no_solved_claim": True,
                    }
                ],
            }
        ]
        board["template_failure_memory"] = [
            {"template_id": "tpl_bad", "failure_count": 2, "reasons": ["no_retron_match"]}
        ]

        payload = build_agentic_guided_payload(board)
        policy = payload["search_policy"]

        self.assertIn("tpl_good", policy["selected_analogical_template_ids"])
        self.assertIn("tpl_bad", policy["forbidden_template_ids"])
        self.assertEqual(policy["preferred_subgoal"]["template_application_hints"][0]["template_id"], "tpl_good")
        self.assertTrue(policy["source_budget"]["analogy_is_advisory_only"])
        self.assertIn("steroid_carbonyl_redox_adjustment", policy["source_budget"]["preferred_reaction_classes"])
        hypothesis_hints = policy["preferred_subgoal"]["hypothetical_reaction_center_hints"]
        self.assertEqual(hypothesis_hints[0]["template_id"], "tpl_hypothesis")
        self.assertEqual(hypothesis_hints[0]["hypothesis"]["hypothesis_type"], "carbonyl_redox_adjustment")
        self.assertTrue(hypothesis_hints[0]["not_parent_route_proof"])
        self.assertEqual(policy["anchor_whitelist"], [])
        self.assertIn("CCO", policy["preferred_subgoal"]["preferred_subgoals"])
        self.assertIn("CCO", policy["source_budget"]["preferred_precursor_smiles"])
        precursor_targets = policy["preferred_subgoal"]["hypothetical_precursor_targets"]
        self.assertEqual(precursor_targets[0]["smiles"], "CCO")
        self.assertEqual(precursor_targets[0]["allowed_use"], "guided_search_subgoal_hint_only")
        self.assertTrue(precursor_targets[0]["not_exact_literature_segment"])
        self.assertTrue(precursor_targets[0]["not_parent_route_proof"])
        self.assertTrue(policy["source_budget"]["hypothetical_route_hints_are_not_proof"])
        self.assertTrue(policy["source_budget"]["hypothesis_precursor_hints_are_not_proof"])
        guided_config = apply_chem_enzy_search_policy(RouteSearchConfig(target_smiles=MLA_LIKE_SMILES), policy)
        context = guided_config.search_flags["cascade_search_context"]
        self.assertIn("CCO", context["preferred_subgoal"]["preferred_subgoals"])
        self.assertEqual(context["preferred_subgoal"]["hypothetical_precursor_targets"][0]["smiles"], "CCO")

    def test_planner_expands_hypothetical_precursor_candidates_as_child_targets(self):
        target = TargetInput(target_name="hypothesis precursor target", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["exact_rows"] = [{"row_id": "placeholder_exact_row_blocks_scout"}]
        board["template_applications"] = [
            {
                "application_id": "app_hypothesis",
                "template_id": "tpl_hypothesis",
                "accepted": True,
                "allowed_use": "hypothesis_only_not_solved",
                "product_retron_type": "steroid_carbonyl_redox_adjustment",
                "executable_candidate_available": False,
                "hypothetical_precursor_hints": [
                    {
                        "schema_version": "analogical_hypothesis_precursor_hint.v1",
                        "hint_id": "hyp_precursor_1",
                        "precursor_smiles": "CCO",
                        "precursor_role": "same_core_hydroxy_steroid_carbonyl_precursor",
                        "derived_from_retron": "steroid_carbonyl_redox_adjustment",
                        "hypothesis_type": "carbonyl_redox_adjustment",
                        "allowed_use": "guided_search_subgoal_hint_only",
                        "not_exact_literature_segment": True,
                        "not_parent_route_proof": True,
                        "requires_verifier": True,
                        "no_solved_claim": True,
                    }
                ],
            }
        ]

        batch = plan_action_batch(board, round_index=2, max_actions=3)
        validation = validate_action_batch(batch, blackboard=board)
        child_actions = [action for action in batch["actions"] if action["action_type"] == "expand_child_target"]

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(len(child_actions), 1)
        payload = child_actions[0]["payload"]
        self.assertEqual(payload["subgoal_targets"][0]["smiles"], "CCO")
        self.assertTrue(payload["subgoal_targets"][0]["hypothesis_only_not_solved"])
        policy = payload["subgoal_targets"][0]["chem_enzy_search_policy"]
        self.assertEqual(policy["anchor_whitelist"], [])
        self.assertTrue(policy["source_budget"]["hypothesis_precursor_hint"])
        self.assertTrue(policy["source_budget"]["hypothesis_precursor_hints_are_not_proof"])

    def test_planner_advances_to_unattempted_hypothetical_precursor_child_targets(self):
        target = TargetInput(target_name="hypothesis precursor target", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "placeholder_exact_row_blocks_scout"}]
        hints = []
        for idx, smiles in enumerate(["CCO", "CCN", "CCC"], start=1):
            hints.append(
                {
                    "schema_version": "analogical_hypothesis_precursor_hint.v1",
                    "hint_id": f"hyp_precursor_{idx}",
                    "precursor_smiles": smiles,
                    "precursor_role": f"same_core_precursor_{idx}",
                    "derived_from_retron": "steroid_carbonyl_redox_adjustment",
                    "hypothesis_type": "carbonyl_redox_adjustment",
                    "allowed_use": "guided_search_subgoal_hint_only",
                    "not_exact_literature_segment": True,
                    "not_parent_route_proof": True,
                    "requires_verifier": True,
                    "no_solved_claim": True,
                }
            )
        board["template_applications"] = [
            {
                "application_id": "app_hypothesis",
                "template_id": "tpl_hypothesis",
                "accepted": True,
                "allowed_use": "hypothesis_only_not_solved",
                "product_retron_type": "steroid_carbonyl_redox_adjustment",
                "executable_candidate_available": False,
                "hypothetical_precursor_hints": hints,
            }
        ]
        board["route_failures"] = [
            {
                "schema_version": "agent_route_failure.v1",
                "reason": "large_atom_jump",
                "route_status": "fake_closed_rejected",
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "build_failure_critic_report",
                "status": "accepted",
                "useful_artifact": True,
                "stale": False,
                "reasons": [],
                "action_signature": "{}",
            },
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 2,
                "action_type": "expand_child_target",
                "status": "accepted",
                "useful_artifact": True,
                "stale": False,
                "reasons": ["no_route_expansion_subgoal_verified_solved"],
                "action_signature": json.dumps(
                    {
                        "action_type": "expand_child_target",
                        "payload": {
                            "subgoal_targets": [
                                {"smiles": "CCO"},
                                {"smiles": "CCN"},
                            ],
                        },
                    },
                    sort_keys=True,
                ),
            },
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 3,
                "action_type": "stitch_parent_route",
                "status": "rejected",
                "useful_artifact": True,
                "stale": False,
                "reasons": ["subgoal_verifier_not_accepted"],
                "action_signature": "{}",
            },
        ]

        batch = plan_action_batch(board, round_index=4, max_actions=3, exhaust_round_budget=True)
        validation = validate_action_batch(batch, blackboard=board)
        child_actions = [action for action in batch["actions"] if action["action_type"] == "expand_child_target"]

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(len(child_actions), 1)
        payload = child_actions[0]["payload"]
        self.assertEqual([row["smiles"] for row in payload["subgoal_targets"]], ["CCC"])
        self.assertEqual(payload["max_targets"], 1)

    def test_budget_exhaustive_planner_changes_direction_after_stale_rounds(self):
        target = TargetInput(target_name="stale_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["route_failures"] = [
            {
                "schema_version": "agent_route_failure.v1",
                "reason": "large_atom_jump",
                "route_status": "fake_closed_rejected",
            }
        ]
        board["action_history"] = [
            {
                "round_index": 1,
                "action_type": "compile_exact_literature_rows",
                "action_signature": "{}",
                "useful_artifact": False,
                "stale": True,
            },
            {
                "round_index": 2,
                "action_type": "extract_visual_literature_chain",
                "action_signature": "{}",
                "useful_artifact": False,
                "stale": True,
            },
        ]

        default_batch = plan_action_batch(board, round_index=3)
        exhaustive_batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)

        self.assertEqual(default_batch["actions"][0]["action_type"], "stop_unresolved")
        self.assertNotEqual(exhaustive_batch["actions"][0]["action_type"], "stop_unresolved")
        self.assertEqual(exhaustive_batch["actions"][0]["action_type"], "generate_disconnection_hypotheses")

    def test_planner_defers_guided_until_local_pdf_extraction_branch_finishes(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "bridge:core",
                "task_type": "target_proximal_bridge",
                "target_handle": "core",
            }
        ]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1016/j.tet.2025.134610",
                "local_pdf": "/tmp/bufotalin.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "11"],
            }
        ]

        batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_pdf_literature_structures", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)

    def test_planner_processes_multiple_local_pdfs_in_one_blackboard(self):
        target = TargetInput(target_name="multi_pdf_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:first", "local_pdf": "/tmp/first.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:second", "local_pdf": "/tmp/second.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {
                "schema_version": "agent_pdf_structure_evidence_summary.v1",
                "evidence_id": "doi:first",
                "source_ref": "doi:first",
                "accepted": True,
            }
        ]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_pdf_literature_structures")
        self.assertEqual(first["payload"]["source_ref"], "doi:second")
        self.assertEqual(first["payload"]["pdf_path"], "/tmp/second.pdf")

    def test_planner_visual_extracts_next_pdf_source_after_first_source_compiled(self):
        target = TargetInput(target_name="multi_pdf_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:first", "local_pdf": "/tmp/first.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:second", "local_pdf": "/tmp/second.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:first", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:second", "accepted": True},
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "first_visual",
                "source_ref": "doi:first",
                "accepted": True,
                "candidate_step_count": 1,
            }
        ]
        board["action_history"] = [
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "useful_artifact": True, "stale": False},
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_visual_literature_chain")
        self.assertEqual(first["payload"]["source_ref"], "doi:second")
        self.assertEqual(first["payload"]["pdf_path"], "/tmp/second.pdf")

    def test_planner_does_not_recompile_visual_chains_without_uncompiled_steps(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h_core"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:zhang", "local_pdf": "/tmp/zhang.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:reddy", "local_pdf": "/tmp/reddy.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:chen", "local_pdf": "/tmp/chen.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:zhang", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:reddy", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:chen", "accepted": True},
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "zhang_visual",
                "source_ref": "doi:zhang",
                "accepted": False,
                "candidate_step_count": 0,
                "gap_labels": ["ouabagenin"],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "reddy_visual",
                "source_ref": "doi:reddy",
                "accepted": False,
                "candidate_step_count": 0,
                "gap_labels": ["18", "19"],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "chen_visual",
                "source_ref": "doi:chen",
                "accepted": False,
                "candidate_step_count": 6,
                "gap_labels": ["21-33 protected tetracyclic intermediates"],
            },
        ]
        board["literature_evidence"]["exact_rows"] = [
            {"schema_version": "agent_exact_literature_row_summary.v1", "row_id": f"source_detail_exact_step:step_{idx}", "source_ref": "doi:chen"}
            for idx in range(6)
        ]
        board["literature_evidence"]["exact_chain_audits"] = [
            {"schema_version": "agent_exact_chain_audit_summary.v1", "audit_id": "chen_audit", "accepted": False, "one_step_row_count": 6}
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve:chen:21-33",
                "label": "21-33 protected tetracyclic intermediates",
                "source_ref": "doi:chen",
                "status": "open",
            }
        ]
        board["action_history"] = [
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "useful_artifact": True, "stale": False},
            {"round_index": 5, "action_type": "compile_exact_literature_rows", "useful_artifact": False, "stale": True},
            {"round_index": 6, "action_type": "compile_exact_literature_rows", "useful_artifact": False, "stale": True},
        ]
        board["budget_state"]["visual_calls"] = 6

        batch = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("compile_exact_literature_rows", action_types)
        self.assertIn("run_guided_chemenzy", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_visual_tool_selects_pdf_evidence_by_source_ref_not_latest_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first_image = run_dir / "first.png"
            second_image = run_dir / "second.png"
            first_image.write_bytes(b"first")
            second_image.write_bytes(b"second")
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "multi_pdf_case", "target_smiles": "CCO"},
                preflight={"case_id": "multi_pdf_case"},
                budget=HarnessBudget(timeout_s=60),
            )
            state.artifacts["literature_pdf_structure_evidence_history"] = [
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "source_ref": "doi:first",
                    "source_pdf_path": "/tmp/first.pdf",
                    "rendered_pages": [{"image_path": str(first_image)}],
                },
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "source_ref": "doi:second",
                    "source_pdf_path": "/tmp/second.pdf",
                    "rendered_pages": [{"image_path": str(second_image)}],
                },
            ]
            state.artifacts["literature_pdf_structure_evidence"] = state.artifacts[
                "literature_pdf_structure_evidence_history"
            ][1]

            evidence = _pdf_evidence_from_payload_or_artifacts(state, {"source_ref": "doi:first"})
            image_paths = _visual_chain_image_paths(state, {"source_ref": "doi:first"}, evidence)

        self.assertEqual(evidence["source_ref"], "doi:first")
        self.assertEqual([path.name for path in image_paths], ["first.png"])

    def test_visual_codex_prompt_disables_web_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "visual"
            output_dir.mkdir()
            image_path = root / "page.png"
            image_path.write_bytes(b"fake image")
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import json, os, pathlib, sys",
                        "args = sys.argv[1:]",
                        "last_message = ''",
                        "for index, arg in enumerate(args[:-1]):",
                        "    if arg == '--output-last-message':",
                        "        last_message = args[index + 1]",
                        "if last_message:",
                        "    pathlib.Path(last_message).write_text(json.dumps({'schema_version':'visual_structure_candidate_chain.v1','steps':[]}), encoding='utf-8')",
                        "config = pathlib.Path(os.environ['CODEX_HOME']) / 'config.toml'",
                        "pathlib.Path.cwd().joinpath('captured_codex_invocation.json').write_text(",
                        "    json.dumps({'argv': args, 'config': config.read_text(encoding='utf-8')}),",
                        "    encoding='utf-8',",
                        ")",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)

            result = _run_codex_visual_prompt(
                executable=str(fake_codex),
                api_key="sk-test",
                base_url="https://example.test/v1",
                model="fake-model",
                output_dir=output_dir,
                image_paths=[image_path],
                prompt="return json",
                timeout_s=5.0,
                prompt_filename="prompt.txt",
                event_log_filename="events.jsonl",
                stderr_log_filename="stderr.log",
                last_message_filename="last_message.txt",
            )
            captured = json.loads((output_dir / "captured_codex_invocation.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "completed")
        self.assertNotIn("--search", captured["argv"])
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", captured["argv"])
        self.assertIn("--sandbox", captured["argv"])
        self.assertIn("workspace-write", captured["argv"])
        self.assertIn("web_search = false", captured["config"])
        self.assertNotIn("web_search = true", captured["config"])

    def test_direct_visual_prompt_uses_api_payload_without_codex_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "visual"
            output_dir.mkdir()
            image_path = root / "page.png"
            image_path.write_bytes(b"fake image bytes")
            captured = {}

            def fake_post(**kwargs):
                captured.update(kwargs)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "schema_version": "visual_structure_candidate_chain.v1",
                                        "case_id": "direct_visual",
                                        "steps": [],
                                    }
                                )
                            }
                        }
                    ]
                }

            with patch(
                "cascade_planner.harness.visual_literature_chain_agent._post_visual_api_json",
                side_effect=fake_post,
            ):
                result = _run_direct_visual_prompt(
                    api_key="sk-test",
                    base_url="https://api.wellau.com/v1",
                    model="fake-model",
                    output_dir=output_dir,
                    image_paths=[image_path],
                    prompt="return json",
                    timeout_s=5.0,
                    prompt_filename="prompt.txt",
                    event_log_filename="events.jsonl",
                    stderr_log_filename="stderr.log",
                    last_message_filename="last_message.txt",
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["execution_mode"], "direct_visual_api")
        self.assertEqual(captured["endpoint"], "chat/completions")
        user_content = captured["payload"]["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(json.loads(result["raw_last_message"])["schema_version"], "visual_structure_candidate_chain.v1")

    def test_visual_parser_preserves_condition_only_repair_steps(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source": {"doi": "10.1021/ja8023466", "title": "Synthesis of (+)-Cortistatin A"},
            "steps": [
                {
                    "step_id": "step_1",
                    "visible_product_label": "cortistatinone",
                    "product_smiles": "CCO",
                    "source_scheme": "Scheme 2",
                    "source_grounding": "visible arrow and condition block",
                    "reaction_conditions": {
                        "reagent": "Dess-Martin periodinane",
                        "solvent": "CH2Cl2",
                        "temperature": "room temperature",
                        "condition_text_transcribed": "DMP, CH2Cl2, rt",
                    },
                },
                {
                    "step_id": "step_2",
                    "mapped_candidate_label": "cortistatin A",
                    "product_smiles": "CCN",
                    "source_scheme": "Scheme 3",
                    "reaction_conditions": {
                        "reagent": "reducing conditions",
                        "reported_yield": "visible yield",
                    },
                },
            ],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="steroid_target",
            target_smiles=BUFOTALIN_ACHIRAL_SMILES,
            source_ref="doi:10.1021/ja8023466",
            source_title="Synthesis of (+)-Cortistatin A",
            image_paths=[],
        )
        quality = _candidate_quality(chain, expected_labels=["cortistatinone", "cortistatin A"])

        self.assertEqual(len(chain["steps"]), 2)
        self.assertEqual(chain["steps"][0]["product_label"], "cortistatinone")
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "Dess-Martin periodinane")
        self.assertTrue(chain["steps"][0]["structure_derivation"]["structure_gap"])
        self.assertEqual(quality["smiles_precheck"]["invalid_smiles_count"], 0)
        self.assertEqual(quality["structure_gap_count"], 2)
        self.assertFalse(quality["accepted"])

    def test_visual_parser_accepts_achiral_connectivity_candidate_as_exploratory(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source_ref": "doi:10.0000/analog",
            "source_title": "Analog steroid source",
            "route_order": "retro_target_to_start",
            "confidence": "low",
            "steps": [
                {
                    "step_id": "approx_step_1",
                    "product_label": "drawn alcohol",
                    "product_smiles": "CCO",
                    "reactant_labels": ["drawn aldehyde"],
                    "reactant_smiles": ["CC=O"],
                    "main_reactant_smiles": "CC=O",
                    "source_locator": "Scheme 1",
                    "condition_candidate": {"reagent": "NaBH4", "source_grounding": "Scheme 1"},
                    "structure_derivation": {
                        "basis": "current_pdf_image_to_achiral_or_approximate_smiles",
                        "source_locator": "Scheme 1",
                        "confidence": "low",
                        "tool_checks": ["visual extraction performed in this run"],
                    },
                    "stereochemistry_status": "unspecified_or_partial",
                    "not_exact_literature_segment": True,
                    "allowed_use": "exploratory_template_and_guided_hint_only",
                    "risk_flags": ["stereochemistry_unspecified"],
                }
            ],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="steroid_target",
            target_smiles=BUFOTALIN_ACHIRAL_SMILES,
            source_ref="doi:10.0000/analog",
            source_title="Analog steroid source",
            image_paths=[],
        )
        quality = _candidate_quality(chain, expected_labels=["drawn alcohol"])

        self.assertTrue(quality["accepted"])
        self.assertTrue(quality["exploratory_accepted"])
        self.assertFalse(quality["exact_ready"])
        self.assertEqual(quality["acceptance_level"], "exploratory_connectivity_candidate")
        self.assertTrue(chain["steps"][0]["not_exact_literature_segment"])
        self.assertEqual(chain["steps"][0]["allowed_use"], "exploratory_template_and_guided_hint_only")

        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=run_preflight(target), max_rounds=3)
        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={
                    "schema_version": "agent_action.v1",
                    "action_id": "visual:approx",
                    "action_type": "extract_visual_literature_chain",
                    "rationale": "extract approximate visual candidate",
                    "expected_artifact": "visual chain",
                    "success_condition": "exploratory candidate",
                    "payload": {},
                },
                action_result={
                    "accepted": True,
                    "result": {
                        "schema_version": "visual_literature_chain_extraction_result.v1",
                        "accepted": True,
                        "acceptance_level": "exploratory_connectivity_candidate",
                        "exact_ready": False,
                        "exploratory_accepted": True,
                        "source_ref": "doi:10.0000/analog",
                        "candidate_chain": chain,
                        "candidate_quality": quality,
                        "candidate_step_count": 1,
                        "reasons": ["visual_literature_chain_structure_gaps"],
                    },
                },
                round_index=1,
                run_dir=tmp,
            )

        summary = board["literature_evidence"]["visual_chains"][0]
        self.assertTrue(summary["exploratory_accepted"])
        self.assertFalse(summary["exact_ready"])
        self.assertEqual(summary["steps"][0]["allowed_use"], "exploratory_template_and_guided_hint_only")
        self.assertTrue(summary["steps"][0]["not_exact_literature_segment"])

    def test_guided_chemenzy_large_atom_jump_overrides_backend_solved(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "steroid_target", "target_smiles": BUFOTALIN_ACHIRAL_SMILES},
                preflight={"case_id": "steroid_target"},
                budget=HarnessBudget(max_guided_chemenzy_runs=1),
            )
            payload = {
                "chem_enzy_search_policy": {
                    "schema_version": "chem_enzy_search_policy.v1",
                    "policy_id": "test_policy",
                    "operator_id": "test",
                    "case_id": "steroid_target",
                    "preferred_subgoal": {},
                    "source_budget": {},
                    "budget": {"max_depth": 3, "max_iterations": 3, "expansion_topk": 3},
                    "mode": "guided",
                    "compiler_metadata": {"requires_verifier": True},
                }
            }
            backend_result = {"ok": True, "routes": [{"route_id": "r1"}], "search_status": {"solved": True}}
            verifier = {
                "schema_version": "harness_route_verifier_report.v1",
                "accepted": True,
                "route_status": "solved",
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump", "details": {"jumps": [{"delta_heavy_atoms": 24}]}}],
            }

            with patch(
                "cascade_planner.harness.tools._execute_chemenzy_request",
                return_value=backend_result,
            ), patch(
                "cascade_planner.harness.tools.verify_chemenzy_raw_routes",
                return_value=verifier,
            ):
                output = run_guided_chemenzy_rerun(state, payload)

        result = output["result"]
        self.assertTrue(output["accepted"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["solved"])
        self.assertEqual(result["route_status"], "fake_closed_rejected")
        self.assertIn("guided_route_verifier_rejected_large_atom_jump", result["reasons"])
        self.assertTrue(result["route_failure_feedback"]["accepted"])

    def test_planner_does_not_visual_extract_placeholder_or_metadata_only_sources(self):
        target = TargetInput(target_name="metadata_case", target_smiles="CCO")
        preflight = run_preflight(target)
        for candidate in [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "query:metadata_case:bridge",
                "title": "placeholder bridge query",
                "placeholder_only": True,
                "access_status": "placeholder_only",
                "local_pdf": "",
                "url": "",
                "doi": "",
            },
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1000/metadata",
                "title": "metadata-only article",
                "doi": "10.1000/metadata",
                "url": "https://doi.org/10.1000/metadata",
                "local_pdf": "",
                "access_status": "metadata_only",
            },
        ]:
            board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
            board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
            board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
            board["bridge_tasks"] = [
                {
                    "schema_version": "agent_bridge_task.v1",
                    "task_id": "bridge:core",
                    "task_type": "target_proximal_bridge",
                    "target_handle": "core",
                }
            ]
            board["literature_evidence"]["source_candidates"] = [candidate]
            board["literature_evidence"]["source_refs"] = [candidate["source_ref"]]

            batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
            action_types = [row["action_type"] for row in batch["actions"]]

            self.assertNotIn("extract_visual_literature_chain", action_types)

    def test_planner_repairs_visual_gaps_before_compile(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "24", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [{"evidence_id": "pdf", "accepted": True}]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": True,
                "candidate_step_count": 1,
                "condition_gap_labels": ["33"],
                "steps": [
                    {
                        "product_label": "33",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC=O"],
                        "not_exact_literature_segment": True,
                    }
                ],
                "extraction_gaps": [{"labels": ["24", "25", "11"], "reason": "small structures"}],
            }
        ]
        board["budget_state"]["visual_calls"] = 1

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_visual_literature_chain")
        self.assertTrue(first["payload"]["focused_gap_repair"])
        self.assertIn("24", first["payload"]["expected_labels"])
        self.assertIn("11", first["payload"]["expected_labels"])

    def test_planner_compiles_partial_visual_steps_before_more_gap_repair(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "24", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [{"evidence_id": "pdf", "accepted": True}]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": False,
                "candidate_step_count": 3,
                "extraction_gaps": [{"labels": ["24", "25", "11"], "gap_type": "structure_gap"}],
            }
        ]
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 3,
                "action_type": "extract_visual_literature_chain",
                "useful_artifact": True,
                "stale": False,
            }
        )
        board["budget_state"]["visual_calls"] = 1

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "compile_exact_literature_rows")

    def test_planner_changes_to_templates_after_one_unresolved_visual_repair(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core", "target_handle": "steroid_core"}]}
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "title": "Analog steroid route",
                "source_type": "reaction_precedent",
                "relevance_rationale": "analog steroid family precedent",
                "expected_scheme_or_compound_labels": ["target", "27", "26", "23"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [{"evidence_id": "pdf", "accepted": True}]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": True,
                "candidate_step_count": 6,
                "extraction_gaps": [{"label": "steroid core 26", "gap_type": "structure_gap"}],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual2",
                "accepted": True,
                "candidate_step_count": 3,
                "extraction_gaps": [{"label": "steroid core 26", "gap_type": "structure_gap"}],
            },
        ]
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:sugar_branch"}]
        board["literature_evidence"]["exact_chain_audits"] = [
            {
                "schema_version": "agent_exact_chain_audit_summary.v1",
                "audit_id": "audit1",
                "accepted": False,
                "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
                "one_step_row_count": 6,
            }
        ]
        board["action_history"] = [
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "useful_artifact": True, "stale": False},
            {
                "round_index": 5,
                "action_type": "extract_visual_literature_chain",
                "useful_artifact": True,
                "stale": False,
                "action_signature": json.dumps({"payload": {"focused_gap_repair": True}}),
            },
            {"round_index": 6, "action_type": "compile_exact_literature_rows", "useful_artifact": False, "stale": True},
        ]
        board["budget_state"]["visual_calls"] = 3

        batch = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertNotIn("extract_visual_literature_chain", action_types)
        self.assertIn("run_guided_chemenzy", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_changes_to_templates_after_visual_tool_failures_without_steps(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=10)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core", "target_handle": "steroid_core"}]}
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "src:analog_source",
                "local_pdf": "/tmp/source.pdf",
                "source_type": "literature_metadata+local_pdf",
                "relevance_rationale": "analog steroid family precedent",
                "expected_scheme_or_compound_labels": ["target analogue", "core"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {
                "schema_version": "agent_pdf_structure_evidence_summary.v1",
                "source_ref": "src:analog_source",
                "accepted": True,
            }
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "src:analog_source",
                "accepted": False,
                "candidate_step_count": 0,
                "missing_expected_labels": ["target analogue", "core"],
                "reasons": ["codex_visual_chain_nonzero_exit", "visual_literature_chain_has_no_steps"],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual2",
                "source_ref": "src:analog_source",
                "accepted": False,
                "candidate_step_count": 0,
                "missing_expected_labels": ["target analogue", "core"],
                "reasons": ["codex_visual_chain_nonzero_exit", "visual_literature_chain_has_no_steps"],
            },
        ]
        board["action_history"] = [
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {
                "round_index": 4,
                "action_type": "extract_visual_literature_chain",
                "action_signature": json.dumps({"payload": {"focused_gap_repair": True}}),
                "useful_artifact": True,
                "stale": False,
            },
        ]
        board["budget_state"]["visual_calls"] = 3

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertNotIn("compile_exact_literature_rows", action_types)
        self.assertNotIn("stop_unresolved", action_types)

    def test_planner_runs_guided_after_template_extraction_failure(self):
        target = TargetInput(target_name="steroid target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core", "target_handle": "polycyclic_cage_core"}]}
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [{"source_ref": "doi:source", "doi": "10.1000/source"}]
        board["literature_evidence"]["source_refs"] = ["doi:source"]
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:anchor"}]
        board["literature_evidence"]["exact_chain_audits"] = [
            {"audit_id": "audit1", "accepted": False, "reasons": ["no_chain_unrolled"]}
        ]
        board["action_history"] = [
            {
                "round_index": 7,
                "action_type": "extract_analogical_reaction_templates",
                "status": "rejected",
                "useful_artifact": False,
                "stale": True,
            }
        ]

        batch = plan_action_batch(board, round_index=8, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("extract_analogical_reaction_templates", action_types)
        self.assertIn("run_guided_chemenzy", action_types)
        guided_action = next(row for row in batch["actions"] if row["action_type"] == "run_guided_chemenzy")
        self.assertIn("search_policy", guided_action["payload"])
        self.assertTrue(
            guided_action["payload"]["search_policy"]["compiler_metadata"]["requires_verifier"]
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_compiles_complete_visual_chain_with_stereo_warnings(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "32", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [{"evidence_id": "pdf", "accepted": True}]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": False,
                "candidate_step_count": 3,
                "missing_expected_labels": [],
                "extraction_gaps": [
                    {
                        "labels": ["30", "11"],
                        "gap_type": "stereochemical_ambiguity",
                        "detail": "valid connectivity, stereo warning only",
                    }
                ],
            }
        ]
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 3,
                "action_type": "extract_visual_literature_chain",
                "useful_artifact": True,
                "stale": False,
            }
        )

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "compile_exact_literature_rows")

    def test_planner_prefers_guided_before_exact_literature_terminal_child(self):
        target = TargetInput(target_name="bufotalin", target_smiles=BUFOTALIN_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:24_from_11"}]
        board["literature_evidence"]["terminal_candidates"] = [
            {
                "schema_version": "agent_literature_terminal_candidate.v1",
                "name": "Androstenedione",
                "smiles": "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
                "canonical_smiles": "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
            }
        ]
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "literature_terminal_child:androstenedione",
                "task_type": "upstream_terminal_synthesis",
            }
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types[0], "run_guided_chemenzy")
        self.assertIn("expand_child_target", action_types)
        child_action = next(row for row in batch["actions"] if row["action_type"] == "expand_child_target")
        self.assertEqual(child_action["payload"]["subgoal_targets"][0]["name"], "Androstenedione")
        child_target = child_action["payload"]["subgoal_targets"][0]
        self.assertTrue(child_target["target_equivalence_audit_required"])
        self.assertTrue(child_target["exact_target_override"])
        self.assertTrue(child_target["no_solved_claim"])
        self.assertTrue(child_target["child_route_cannot_promote_parent"])
        self.assertTrue(child_target["chem_enzy_search_policy"]["compiler_metadata"]["requires_verifier"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_stops_repeating_same_failed_child_terminal(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        terminal_smiles = "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:terminal"}]
        board["literature_evidence"]["terminal_candidates"] = [
            {
                "schema_version": "agent_literature_terminal_candidate.v1",
                "name": "source detail literature terminal",
                "smiles": terminal_smiles,
                "canonical_smiles": terminal_smiles,
            }
        ]
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "literature_terminal_child:terminal",
                "task_type": "upstream_terminal_synthesis",
            }
        ]
        for attempt in (1, 2):
            board["action_history"].append(
                {
                    "schema_version": "agent_action_history_record.v1",
                    "round_index": attempt,
                    "action_type": "expand_child_target",
                    "status": "accepted",
                    "useful_artifact": True,
                    "stale": False,
                    "reasons": ["no_route_expansion_subgoal_verified_solved"],
                    "action_signature": json.dumps(
                        {
                            "action_type": "expand_child_target",
                            "payload": {
                                "expansion_attempt": attempt,
                                "subgoal_targets": [{"smiles": terminal_smiles}],
                            },
                        },
                        sort_keys=True,
                    ),
                }
            )

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("expand_child_target", action_types)

    def test_planner_stitches_parent_after_exact_terminal_child_solved(self):
        target = TargetInput(target_name="bufotalin", target_smiles=BUFOTALIN_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:24_from_11"}]
        board["literature_evidence"]["terminal_candidates"] = [
            {
                "schema_version": "agent_literature_terminal_candidate.v1",
                "name": "Androstenedione",
                "smiles": "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
                "canonical_smiles": "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
            }
        ]
        board["current_belief"]["child_route_solved"] = True
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 5,
                "action_type": "expand_child_target",
                "useful_artifact": True,
                "stale": False,
            }
        )

        batch = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types[0], "stitch_parent_route")
        self.assertNotIn("expand_child_target", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        stitch_payload = batch["actions"][0]["payload"]
        self.assertEqual(stitch_payload["proof_policy"]["final_verdict_authority"], "deterministic_parent_route_proof")
        self.assertIn("exact_literature_row_ids", stitch_payload["proof_binding"])
        self.assertTrue(stitch_payload["proof_policy"]["analogy_is_not_proof"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_blackboard_records_stereo_ambiguity_as_visual_warning(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:1",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract chain",
                "expected_artifact": "visual",
                "success_condition": "chain",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": True,
                    "candidate_step_count": 15,
                    "parsed_output": {
                        "steps": [{"product_label": "bufotalin"}],
                        "extraction_gaps": [
                            {
                                "gap_type": "stereochemical_ambiguity",
                                "labels": ["25", "23"],
                                "reason": "major stereoisomer encoded",
                            }
                        ],
                    },
                    "candidate_quality": {
                        "missing_expected_labels": [],
                        "condition_gap_labels": [],
                    },
                    "reasons": [],
                },
                "reasons": [],
            },
            round_index=3,
            run_dir="/tmp",
        )

        visual = board["literature_evidence"]["visual_chains"][0]
        self.assertEqual(visual["gap_labels"], [])
        self.assertEqual(visual["warning_gap_labels"], ["25", "23"])

    def test_blackboard_structure_gaps_create_resolution_tasks(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:structure-gap",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract chain",
                "expected_artifact": "visual",
                "success_condition": "chain",
                "payload": {"source_ref": "doi:source"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": False,
                    "source_ref": "doi:source",
                    "parsed_output": {
                        "source_ref": "doi:source",
                        "extraction_gaps": [
                            {
                                "label": "compound 15",
                                "gap_type": "structure_gap",
                                "reason": "visible but not safely convertible",
                            }
                        ],
                    },
                    "reasons": ["visual_literature_chain_extraction_gaps"],
                },
                "reasons": ["visual_literature_chain_extraction_gaps"],
            },
            round_index=3,
            run_dir="/tmp",
        )

        tasks = board["literature_evidence"]["structure_resolution_tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["label"], "compound 15")
        self.assertEqual(tasks[0]["source_ref"], "doi:source")

    def test_blackboard_records_resolved_literature_structure_and_marks_task_resolved(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "resolve:15",
                "action_type": "resolve_literature_structure_task",
                "rationale": "resolve one label",
                "expected_artifact": "structure resolution",
                "success_condition": "resolved or unresolved",
                "payload": {"task_id": "resolve_structure:doi_source_compound_15", "label": "compound 15", "source_ref": "doi:source"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "literature_structure_resolution_result.v1",
                    "accepted": True,
                    "status": "resolved",
                    "task_id": "resolve_structure:doi_source_compound_15",
                    "label": "compound 15",
                    "source_ref": "doi:source",
                    "resolved_structures": [
                        {
                            "schema_version": "literature_resolved_structure_candidate.v1",
                            "structure_id": "resolve_structure_doi_source_compound_15:1",
                            "task_id": "resolve_structure:doi_source_compound_15",
                            "label": "compound 15",
                            "smiles": "CCO",
                            "source_ref": "doi:source",
                            "source_locator": "Scheme 1",
                            "accepted": True,
                            "no_solved_claim": True,
                        }
                    ],
                    "unresolved_tasks": [],
                    "reasons": [],
                    "no_solved_claim": True,
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        evidence = board["literature_evidence"]
        self.assertEqual(len(evidence["resolved_structures"]), 1)
        self.assertEqual(evidence["structure_resolution_tasks"][0]["status"], "resolved")
        self.assertEqual(evidence["structure_resolution_tasks"][0]["last_resolution_status"], "resolved")
        self.assertTrue(board["action_history"][-1]["useful_artifact"])

    def test_blackboard_records_unresolved_structure_attempt_keeps_task_open(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "resolve:15",
                "action_type": "resolve_literature_structure_task",
                "rationale": "resolve one label",
                "expected_artifact": "structure resolution",
                "success_condition": "resolved or unresolved",
                "payload": {"task_id": "resolve_structure:doi_source_compound_15", "label": "compound 15", "source_ref": "doi:source"},
            },
            action_result={
                "accepted": False,
                "result": {
                    "schema_version": "literature_structure_resolution_result.v1",
                    "accepted": False,
                    "status": "unresolved",
                    "task_id": "resolve_structure:doi_source_compound_15",
                    "label": "compound 15",
                    "source_ref": "doi:source",
                    "resolved_structures": [],
                    "unresolved_tasks": [
                        {
                            "schema_version": "literature_structure_unresolved_task.v1",
                            "task_id": "resolve_structure:doi_source_compound_15",
                            "label": "compound 15",
                            "source_ref": "doi:source",
                            "status": "unresolved",
                            "reason": "no_rdkit_valid_source_grounded_structure_candidate",
                            "no_solved_claim": True,
                        }
                    ],
                    "reasons": ["no_rdkit_valid_structure_candidate"],
                    "no_solved_claim": True,
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        task = board["literature_evidence"]["structure_resolution_tasks"][0]
        self.assertEqual(task["status"], "open")
        self.assertEqual(task["last_resolution_status"], "unresolved")
        self.assertEqual(task["resolution_attempt_count"], 1)
        self.assertEqual(len(board["literature_evidence"]["structure_resolution_attempts"]), 1)
        self.assertTrue(board["action_history"][-1]["useful_artifact"])

    def test_planner_resolves_structure_task_before_structure_scout(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_visual_calls": 8},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:source", "accepted": True}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "doi:source",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "compound 15", "gap_type": "structure_gap"}],
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]
        board["action_history"] = [
            {"round_index": 1, "action_type": "search_literature", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {
                "round_index": 4,
                "action_type": "extract_visual_literature_chain",
                "action_signature": '{"payload":{"focused_gap_repair":true}}',
                "useful_artifact": False,
                "stale": True,
            },
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "resolve_literature_structure_task")
        self.assertEqual(first["payload"]["task_id"], "resolve_structure:doi_source_compound_15")
        self.assertEqual(first["payload"]["label"], "compound 15")
        self.assertTrue(first["payload"]["no_solved_claim"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_scouts_structure_resolution_sources_after_visual_gaps(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:source", "accepted": True}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "doi:source",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "compound 15", "gap_type": "structure_gap"}],
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
                "resolution_attempt_count": 1,
            }
        ]
        board["action_history"] = [
            {"round_index": 1, "action_type": "search_literature", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "action_signature": "{}", "useful_artifact": False, "stale": True},
            {
                "round_index": 5,
                "action_type": "extract_visual_literature_chain",
                "action_signature": '{"payload":{"focused_gap_repair":true}}',
                "useful_artifact": False,
                "stale": True,
            },
        ]
        board["budget_state"]["scout_calls"] = 1

        batch = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "search_literature")
        self.assertTrue(first["payload"]["focused_structure_resolution"])
        self.assertIn("resolve_structure:doi_source_compound_15", first["payload"]["structure_resolution_task_ids"])

    def test_planner_scouts_structure_resolution_after_all_visual_sources_have_only_gaps(self):
        target = TargetInput(target_name="steroid", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h_core"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:reddy", "local_pdf": "/tmp/reddy.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:zhang", "local_pdf": "/tmp/zhang.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:chen", "local_pdf": "/tmp/chen.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:reddy", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:zhang", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:chen", "accepted": True},
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "reddy_visual",
                "source_ref": "doi:reddy",
                "accepted": False,
                "candidate_step_count": 1,
                "condition_gap_labels": ["1b ouabagenin"],
                "extraction_gaps": [{"label": "27", "gap_type": "structure_gap"}],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "zhang_visual",
                "source_ref": "doi:zhang",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "ouabagenin", "gap_type": "structure_gap"}],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "chen_visual",
                "source_ref": "doi:chen",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "ouabagenin precursor", "gap_type": "structure_gap"}],
            },
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_reddy_27",
                "task_type": "resolve_literature_structure",
                "label": "27",
                "source_ref": "doi:reddy",
                "status": "open",
            }
        ]
        board["action_history"] = [
            {"round_index": 1, "action_type": "search_literature", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 5, "action_type": "extract_visual_literature_chain", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {
                "round_index": 6,
                "action_type": "extract_visual_literature_chain",
                "action_signature": '{"payload":{"focused_gap_repair":true}}',
                "useful_artifact": False,
                "stale": True,
            },
        ]
        board["budget_state"]["scout_calls"] = 1
        board["budget_state"]["visual_calls"] = 6

        batch = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "search_literature")
        self.assertTrue(first["payload"]["focused_structure_resolution"])
        self.assertNotEqual(first["action_type"], "compile_exact_literature_rows")

    def test_compile_duplicate_disconnected_exact_rows_are_stale(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:ethanol"}]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:duplicate",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile duplicate rows",
                "expected_artifact": "compiled rows",
                "success_condition": "rows",
                "payload": {},
            },
            action_result={
                "accepted": False,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "compiled_downstream": {
                        "literature_template_plugin": {
                            "one_step_rows": [
                                {
                                    "row_id": "source_detail_exact_step:ethanol",
                                    "product_smiles": "CCO",
                                }
                            ]
                        }
                    },
                    "chain_audit": {
                        "accepted": False,
                        "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
                        "summary": {"one_step_row_count": 1, "chain_step_count": 0},
                    },
                    "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
                },
                "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
            },
            round_index=4,
            run_dir="/tmp",
        )

        self.assertFalse(board["action_history"][-1]["useful_artifact"])
        self.assertTrue(board["action_history"][-1]["stale"])
        self.assertEqual(len(board["literature_evidence"]["exact_rows"]), 1)
        self.assertFalse(board["literature_evidence"]["exact_chain_audits"][0]["accepted"])

    def test_compile_exact_rows_with_only_accepted_audit_is_stale(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:empty",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile empty chain",
                "expected_artifact": "compiled rows",
                "success_condition": "rows",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "compiled_downstream": {"literature_template_plugin": {"one_step_rows": []}},
                    "chain_audit": {
                        "accepted": True,
                        "summary": {"one_step_row_count": 0, "chain_step_count": 0},
                    },
                    "reasons": ["no_chain_unrolled"],
                },
                "reasons": ["no_chain_unrolled"],
            },
            round_index=3,
            run_dir="/tmp",
        )

        self.assertFalse(board["action_history"][-1]["useful_artifact"])
        self.assertTrue(board["action_history"][-1]["stale"])
        self.assertEqual(board["literature_evidence"]["exact_rows"], [])

    def test_compile_exact_rows_marks_sugar_fragment_not_parent_relevant(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:sugar",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile exact rows",
                "expected_artifact": "exact rows",
                "success_condition": "rows are recorded",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "exact_rows": [
                        {
                            "row_id": "source_detail_exact_step:rhamnose_donor",
                            "product_smiles": "OCC1OC(O)C(O)C(O)C1O",
                            "product_label": "rhamnose sugar donor",
                        }
                    ],
                },
            },
            round_index=3,
            run_dir="/tmp",
        )

        row = board["literature_evidence"]["exact_rows"][0]
        self.assertFalse(row["target_relevant_for_parent_bridge"])
        self.assertIn("product_ring_system_too_small_for_target_core", row["target_relevance"]["reasons"])
        self.assertEqual(board["literature_evidence"]["exact_row_target_relevance_summary"]["target_relevant_exact_rows"], 0)
        guided = build_agentic_guided_payload(board)
        self.assertEqual(guided["search_policy"]["accepted_exact_row_ids"], [])
        self.assertEqual(
            guided["search_policy"]["source_budget"]["disconnected_exact_row_ids"],
            ["source_detail_exact_step:rhamnose_donor"],
        )

    def test_planner_does_not_repeat_large_jump_guided_without_new_strong_signal(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_guided_chemenzy_runs": 2},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["route_failures"] = [{"reason": "unexplained_large_atom_jump", "route_status": "rejected"}]
        board["literature_evidence"]["exact_rows"] = [
            {
                "row_id": "source_detail_exact_step:rhamnose_donor",
                "product_smiles": "OCC1OC(O)C(O)C(O)C1O",
                "target_relevant_for_parent_bridge": False,
                "target_relevance": {"target_relevant_for_parent_bridge": False},
            }
        ]
        board["action_history"] = [
            {
                "round_index": 5,
                "action_type": "run_guided_chemenzy",
                "useful_artifact": True,
                "stale": False,
                "action_signature": json.dumps({"action_type": "run_guided_chemenzy"}),
            }
        ]
        board["budget_state"]["chemenzy_runs"] = 1

        batch = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("run_guided_chemenzy", action_types)

    def test_compile_exact_rows_promotes_visual_candidate_steps(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "ethanol_visual_chain",
            "source_ref": "doi:10.0000/source",
            "evidence_refs": ["current_image:1"],
            "steps": [
                {
                    "schema_version": "visual_structure_candidate_step.v1",
                    "step_id": "visual_step_1_ethanol",
                    "segment_id": "visual_chain",
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_labels": ["ethane"],
                    "reactant_smiles": ["CC"],
                    "condition": {
                        "schema_version": "condition_candidate.v1",
                        "source_type": "exact",
                        "condition_status": "evidence_backed",
                        "reagent": "oxidation conditions",
                        "source_grounding": "current PDF image",
                    },
                    "source_locator": "scheme 1",
                },
                {
                    "schema_version": "visual_structure_candidate_step.v1",
                    "step_id": "visual_step_2_ethane",
                    "segment_id": "visual_chain",
                    "product_label": "ethane",
                    "product_smiles": "CC",
                    "reactant_labels": ["methane"],
                    "reactant_smiles": ["C"],
                    "condition": {
                        "schema_version": "condition_candidate.v1",
                        "source_type": "exact",
                        "condition_status": "evidence_backed",
                        "reagent": "coupling conditions",
                        "source_grounding": "current PDF image",
                    },
                    "source_locator": "scheme 1",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        rows = (result["compiled_downstream"]["literature_template_plugin"] or {})["one_step_rows"]
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(result["chain_audit"]["summary"]["chain_step_count"], 2)

    def test_compile_exact_rows_repairs_bufotalin_literature_terminal_stereo(self):
        target_smiles = "CC12CCC(=O)C=C1CCC1C2CCC2(C)C1CCC21OCCO1"
        achiral_androstenedione = "CC12CCC3C(C1CCC2=O)CCC4=CC(=O)CCC34C"
        target = TargetInput(target_name="compound_24", target_smiles=target_smiles)
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "bufotalin_visual_chain",
            "source_ref": "doi:10.1016/j.tet.2025.134610",
            "evidence_refs": ["current_image:scheme3.png"],
            "steps": [
                {
                    "schema_version": "visual_structure_candidate_step.v1",
                    "step_id": "24_from_11",
                    "product_label": "24",
                    "product_smiles": target_smiles,
                    "reactant_labels": ["11"],
                    "reactant_smiles": [achiral_androstenedione],
                    "condition": {
                        "schema_version": "condition_candidate.v1",
                        "source_type": "exact",
                        "condition_status": "evidence_backed",
                        "reagent": "ethylene glycol, p-TsOH",
                        "source_grounding": "Scheme 3 ketalization of androstenedione (11) to 24",
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        audit = record.output["result"]["chain_audit"]
        self.assertTrue(record.output["result"]["accepted"], record.output["result"]["reasons"])
        self.assertTrue(audit["terminal_reached"])
        self.assertEqual(audit["terminal_stereo_repair"]["repair_basis"], "named_anchor_evidence_and_connectivity_match")
        self.assertIn("@", audit["terminal_smiles"])

    def test_blackboard_promotes_literature_terminal_to_upstream_child_task(self):
        target = TargetInput(target_name="bufotalin", target_smiles=BUFOTALIN_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        terminal_smiles = "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:1",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile exact rows",
                "expected_artifact": "rows",
                "success_condition": "rows",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "accepted": True,
                    "compiled_downstream": {
                        "literature_template_plugin": {
                            "one_step_rows": [
                                {
                                    "template": {
                                        "literature_template_trace": {
                                            "source_template_id": "source_detail_exact_step:24_from_11",
                                            "source_ref": "doi:10.1016/j.tet.2025.134610",
                                            "product_smiles": "CCO",
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "chain_audit": {
                        "accepted": True,
                        "terminal_name": "Androstenedione",
                        "terminal_smiles": terminal_smiles,
                        "terminal_canonical_smiles": terminal_smiles,
                        "terminal_reached": True,
                        "step_count": 15,
                        "chain": [{"source_ref": "doi:10.1016/j.tet.2025.134610"}],
                    },
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        terminals = board["literature_evidence"]["terminal_candidates"]
        tasks = board["bridge_tasks"]
        self.assertEqual(terminals[0]["name"], "Androstenedione")
        self.assertEqual(tasks[0]["task_type"], "upstream_terminal_synthesis")
        self.assertEqual(tasks[0]["terminal"]["smiles"], terminal_smiles)

    def test_compile_exact_rows_promotes_visual_candidate_chain_shape(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "ethanol_visual_chain",
            "doi": "10.0000/source",
            "evidence_refs": ["current_image:1"],
            "candidate_chain": [
                {
                    "label": "ethanol",
                    "smiles": "CCO",
                    "precursor_label": "ethane",
                    "precursor_smiles": "CC",
                    "source_locator": "scheme 1",
                    "conditions": {"reagents": "oxidation conditions", "reported_yield": "80%"},
                },
                {
                    "label": "ethane",
                    "smiles": "CC",
                    "precursor_label": "methane",
                    "precursor_smiles": "C",
                    "source_locator": "scheme 1",
                    "conditions": {"reagents": "coupling conditions", "reported_yield": "70%"},
                },
                {
                    "label": "methane",
                    "smiles": "C",
                    "precursor_label": None,
                    "precursor_smiles": None,
                    "source_locator": "scheme 1 starting material",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        rows = (result["compiled_downstream"]["literature_template_plugin"] or {})["one_step_rows"]
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(result["chain_audit"]["summary"]["chain_step_count"], 2)

    def test_compile_exact_rows_promotes_visual_reaction_chain_shape(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "ethanol_visual_chain",
            "source_ref": "doi:10.0000/source",
            "evidence_refs": ["current_image:1"],
            "route_order": "retro_target_to_start",
            "chain": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_label": "ethane",
                    "reactant_smiles": "CC",
                    "conditions": "oxidation conditions, 80%",
                    "source_locator": "scheme 1",
                },
                {
                    "product_label": "ethane",
                    "product_smiles": "CC",
                    "reactant_label": "methane",
                    "reactant_smiles": "C",
                    "conditions": "coupling conditions, 70%",
                    "source_locator": "scheme 1",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        rows = (result["compiled_downstream"]["literature_template_plugin"] or {})["one_step_rows"]
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(result["chain_audit"]["summary"]["chain_step_count"], 2)

    def test_compile_exact_rows_prefers_source_backed_candidate_over_history_draft(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        draft_without_source = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_labels": ["ethane"],
                    "reactant_smiles": ["CC"],
                    "condition": {"reagent": "oxidation conditions"},
                    "source_locator": "scheme 1",
                }
            ],
        }
        normalized_candidate = {
            **draft_without_source,
            "source_ref": "doi:10.0000/source",
            "evidence_refs": ["current_image:scheme1.png"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain_history"] = [draft_without_source]
            state.artifacts["visual_structure_candidate_chain"] = normalized_candidate
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        rows = (result["compiled_downstream"]["literature_template_plugin"] or {})["one_step_rows"]
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(len(rows), 1)
        trace = rows[0]["template"]["literature_template_trace"]
        self.assertEqual(trace["source_ref"], "doi:10.0000/source")

    def test_compile_exact_rows_enriches_visual_parsed_output_from_result_metadata(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_literature_chain_extraction"] = {
                "schema_version": "visual_literature_chain_extraction_result.v1",
                "accepted": True,
                "source_ref": "doi:10.0000/source",
                "source_title": "Visual source",
                "image_paths": ["/tmp/scheme1.png"],
                "parsed_output": {
                    "schema_version": "visual_structure_candidate_chain.v1",
                    "steps": [
                        {
                            "product_label": "ethanol",
                            "product_smiles": "CCO",
                            "reactant_labels": ["ethane"],
                            "reactant_smiles": "CC",
                            "condition": {"reagent": "oxidation conditions"},
                            "source_locator": "scheme 1",
                        }
                    ],
                },
            }
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        rows = (result["compiled_downstream"]["literature_template_plugin"] or {})["one_step_rows"]
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(len(rows), 1)
        trace = rows[0]["template"]["literature_template_trace"]
        self.assertEqual(trace["source_ref"], "doi:10.0000/source")
        self.assertIn("current_image:/tmp/scheme1.png", trace["evidence_refs"])

    def test_route_expansion_prioritizes_literature_starting_material_over_near_target_intermediate(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        compiled = {
            "route_expansion": {
                "child_targets": [
                    {
                        "name": "bufotalin_source_detail_exact_step_bufotalin_from_33_reactant_1",
                        "smiles": "CCO",
                        "source": "source_detail_route_expansion",
                        "source_template_id": "source_detail_exact_step:bufotalin_from_33",
                    },
                    {
                        "name": "bufotalin_source_detail_exact_step_24_from_11_reactant_1",
                        "smiles": "CC",
                        "source": "source_detail_route_expansion",
                        "source_template_id": "source_detail_exact_step:24_from_11",
                    },
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            rows = _route_expansion_child_targets(state=state, payload={}, compiled=compiled)

        self.assertEqual(rows[0]["source_template_id"], "source_detail_exact_step:24_from_11")

    def test_visual_repair_steps_accept_single_reactant_smiles_string(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_label": "ethane",
                    "reactant_smiles": "CC",
                    "condition": "oxidation conditions, 80%",
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(len(chain["steps"]), 1)
        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "oxidation conditions, 80%")

    def test_visual_repair_steps_accept_condition_candidate_string(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "chain": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_label": "ethane",
                    "reactant_smiles": "CC",
                    "condition": "oxidation conditions, 80%",
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "oxidation conditions, 80%")

    def test_visual_repair_steps_accept_precursor_chain_shape(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "chain": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "precursor_label": "ethane",
                    "precursor_smiles": "CC",
                    "forward_conditions": {"reagent": "oxidation conditions", "reported_yield": "80%"},
                    "source_location": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "oxidation conditions")

    def test_visual_repair_steps_accept_plural_precursor_fields(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "precursor_labels": ["ethane"],
                    "precursor_smiles": ["CC"],
                    "condition_candidate": {"reagent": "oxidation conditions"},
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])

    def test_visual_repair_steps_accept_reactant_object_fields(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactants": [{"label": "ethane", "smiles": "CC"}],
                    "condition_candidate": {"reagent": "oxidation conditions"},
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])

    def test_bufotalin_pdf_defaults_add_focused_pages_crops_and_labels(self):
        payload = {}
        _inject_pdf_defaults(
            payload,
            {
                "target_name": "bufotalin",
                "family_hint": "bufadienolide",
                "literature_pdf_path": "/tmp/bufotalin.pdf",
                "literature_pdf_source_ref": "doi:10.1016/j.tet.2025.134610",
            },
        )

        self.assertEqual(payload["pdf_path"], "/tmp/bufotalin.pdf")
        self.assertEqual(payload["page_numbers"], [3, 4, 5, 6])
        self.assertEqual(payload["render_zoom"], 2.5)
        self.assertEqual([row["crop_id"] for row in payload["scheme_crops"]], [
            "scheme3_full_to_20",
            "scheme4_total_synthesis",
            "table1_allylic_oxidation",
        ])
        self.assertIn("bufotalin", payload["compound_labels"])
        self.assertIn("11", payload["expected_labels"])

    def test_visual_quality_flags_condition_gaps(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "steps": [
                    {
                        "product_label": "ethanol",
                        "product_smiles": "CCO",
                        "reactant_label": "ethane",
                        "reactant_smiles": "CC",
                        "condition_candidate": {
                            "schema_version": "condition_candidate.v1",
                            "source_type": "exact",
                            "condition_status": "evidence_backed",
                            "source_grounding": "current PDF image",
                        },
                        "source_locator": "scheme 1",
                    }
                ],
            },
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        quality = _candidate_quality(chain, expected_labels=["ethanol"])

        self.assertFalse(quality["accepted"])
        self.assertEqual(quality["condition_gap_labels"], ["ethanol"])
        self.assertEqual(quality["condition_gap_count"], 1)

    def test_visual_target_label_uses_input_target_smiles_when_visual_target_smiles_is_malformed(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "steps": [
                    {
                        "product_label": "bufotalin",
                        "product_smiles": "CC(",
                        "reactant_label": "33",
                        "reactant_smiles": "CC",
                        "condition": "HF-pyridine, 93%",
                        "source_locator": "scheme 4",
                    }
                ],
            },
            target_name="bufotalin",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["product_smiles"], "CCO")
        self.assertTrue(chain["steps"][0]["structure_derivation"]["target_product_smiles_fallback"])
        quality = _candidate_quality(chain, expected_labels=["bufotalin"])
        self.assertEqual(quality["smiles_precheck"]["invalid_smiles_count"], 0)

    def test_visual_target_label_preserves_input_target_stereo_when_visual_smiles_is_achiral(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "target": {"name": "bufotalin", "smiles": BUFOTALIN_ACHIRAL_SMILES},
                "steps": [
                    {
                        "product_label": "bufotalin",
                        "product_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "reactant_label": "33",
                        "reactant_smiles": "CC",
                        "condition": "HF-pyridine, 93%",
                        "source_locator": "scheme 4",
                    }
                ],
            },
            target_name="bufotalin",
            target_smiles=BUFOTALIN_SMILES,
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["target_smiles"], BUFOTALIN_SMILES)
        self.assertEqual(chain["steps"][0]["product_smiles"], BUFOTALIN_SMILES)
        self.assertTrue(chain["steps"][0]["structure_derivation"]["target_product_stereo_repair"])
        quality = _candidate_quality(chain, expected_labels=["bufotalin"])
        self.assertEqual(quality["smiles_precheck"]["invalid_smiles_count"], 0)

    def test_visual_candidate_steps_field_is_normalized(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "candidate_steps": [
                    {
                        "product_label": "ethanol",
                        "product_smiles": "CCO",
                        "precursor_labels": ["ethene"],
                        "precursor_smiles": ["C=C"],
                        "condition_candidate": {"reagent": "hydration"},
                        "source_locator": "scheme 2",
                    }
                ],
            },
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(len(chain["steps"]), 1)
        self.assertEqual(chain["steps"][0]["product_smiles"], "CCO")
        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["C=C"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethene"])
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "hydration")

    def test_bufotalin_tet2025_label_anchor_chain_is_valid_but_not_solved_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "scheme4_total_synthesis.png"
            image.write_bytes(b"not a real image; path existence is enough for this unit test")
            chain = _bufotalin_tet2025_label_anchor_chain(
                image_paths=[image],
                target_name="bufotalin",
                target_smiles=BUFOTALIN_SMILES,
                source_ref="doi:10.1016/j.tet.2025.134610",
                source_title="Tetrahedron bufotalin total synthesis",
                expected_labels=["bufotalin", "33", "32", "31", "30", "24", "11"],
                route_sequence_hint="bufotalin <= 33 <= 32 <= 31 <= 30 <= 24 <= 11",
            )

            quality = _candidate_quality(chain, expected_labels=["bufotalin", "33", "32", "31", "30", "24", "11"])
            validation = validate_visual_structure_chain(chain, target_smiles=BUFOTALIN_SMILES)

        self.assertEqual(len(chain["steps"]), 15)
        self.assertTrue(quality["accepted"], quality)
        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertTrue(chain["candidate_generation_audit"]["no_solved_claim"])
        self.assertTrue(chain["source_policy"]["requires_source_detail_chain_audit"])
        self.assertEqual(chain["steps"][-1]["product_label"], "24")
        self.assertEqual(chain["steps"][-1]["reactant_labels"], ["11"])

    def test_blackboard_condition_gaps_trigger_focused_visual_repair(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1016/j.tet.2025.134610",
                "local_pdf": "/tmp/bufotalin.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [{"evidence_id": "pdf", "accepted": True}]
        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:1",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract chain",
                "expected_artifact": "visual",
                "success_condition": "chain",
                "payload": {},
            },
            action_result={
                "accepted": False,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": False,
                    "candidate_step_count": 3,
                    "candidate_quality": {
                        "missing_expected_labels": [],
                        "condition_gap_labels": ["bufotalin", "33", "11"],
                    },
                    "reasons": ["visual_literature_chain_condition_gaps"],
                },
                "reasons": ["visual_literature_chain_condition_gaps"],
            },
            round_index=3,
            run_dir="/tmp",
        )
        board["budget_state"]["visual_calls"] = 1

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_visual_literature_chain")
        self.assertTrue(first["payload"]["focused_gap_repair"])
        self.assertIn("condition_candidate", first["payload"]["route_sequence_hint"])


if __name__ == "__main__":
    unittest.main()
