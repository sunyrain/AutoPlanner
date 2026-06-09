import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.agent.chem_enzy_policy import validate_chem_enzy_search_policy
from cascade_planner.harness.agent_action_planner import validate_action_batch
from cascade_planner.harness.agentic_blackboard import build_agentic_guided_payload, initialize_agent_blackboard
from cascade_planner.harness.agentic_blackboard_controller import run_agentic_blackboard_controller
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.schemas import TargetInput
from cascade_planner.harness.target_side_strategy import build_target_side_disconnection_hypotheses


MLA_LIKE_SMILES = "CN1CC2CCC1CC2OC(=O)c3ccccc3N4C(=O)CCC4=O"


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


if __name__ == "__main__":
    unittest.main()
