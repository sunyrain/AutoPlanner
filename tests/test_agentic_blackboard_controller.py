import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.agent.chem_enzy_policy import validate_chem_enzy_search_policy
from cascade_planner.harness.agent_action_planner import plan_action_batch, validate_action_batch
from cascade_planner.harness.agentic_blackboard import (
    build_agentic_guided_payload,
    initialize_agent_blackboard,
    update_blackboard_from_action,
)
from cascade_planner.harness.agentic_blackboard_controller import _inject_pdf_defaults, run_agentic_blackboard_controller
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.schemas import TargetInput
from cascade_planner.harness.target_side_strategy import build_target_side_disconnection_hypotheses
from cascade_planner.harness.tools import HarnessBudget, ToolExecutionState, _route_expansion_child_targets, execute_local_tool
from cascade_planner.harness.visual_literature_chain_agent import (
    _bufotalin_tet2025_label_anchor_chain,
    _candidate_chain_from_parsed,
    _candidate_quality,
)
from cascade_planner.harness.visual_structure_extraction import validate_visual_structure_chain


MLA_LIKE_SMILES = "CN1CC2CCC1CC2OC(=O)c3ccccc3N4C(=O)CCC4=O"
BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H]"
    "(CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
BUFOTALIN_ACHIRAL_SMILES = "CC(=O)OC1CC2(O)C3CCC4CC(O)CCC4(C)C3CCC2(C)C1c1ccc(=O)oc1"


class AgenticBlackboardControllerTest(unittest.TestCase):
    def test_blackboard_initialization_writes_target_profile(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)

        self.assertEqual(board["schema_version"], "agent_blackboard.v1")
        self.assertEqual(board["target_profile"]["target_smiles"], "CCO")
        self.assertEqual(board["budget_state"]["max_rounds"], 3)

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

        action_types = [row["action_type"] for row in result["action_batches"][0]["actions"]]
        task_types = {row["task_type"] for row in board["bridge_tasks"]}
        self.assertIn("generate_disconnection_hypotheses", action_types)
        self.assertIn("build_failure_critic_report", action_types)
        self.assertIn("search_literature", action_types)
        self.assertIn("target_proximal_bridge_required", task_types)
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertNotEqual(result["final_verdict"]["verdict"], "solved")

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
                        "payload": {},
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
        self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])

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
                        "payload": {},
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
        self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])

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
                        "payload": {},
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

    def test_parent_proof_mock_is_required_for_agentic_solved(self):
        proof = {
            "schema_version": "stitched_parent_route_proof.v1",
            "accepted": True,
            "solved": True,
            "route_status": "solved",
            "reasons": [],
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
                        "payload": {},
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
                "candidate_step_count": 12,
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

    def test_planner_expands_exact_literature_terminal_before_guided_rerun(self):
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

        self.assertEqual(action_types[0], "expand_child_target")
        self.assertNotIn("run_guided_chemenzy", action_types)
        self.assertEqual(batch["actions"][0]["payload"]["subgoal_targets"][0]["name"], "Androstenedione")

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
