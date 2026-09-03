import tempfile
import unittest
from pathlib import Path

from cascade_planner.legacy.harness_runtime.agent_action_planner import (
    build_child_expansion_payload_from_blackboard,
    build_guided_chemenzy_payload_from_blackboard,
)
from cascade_planner.legacy.harness_runtime.agentic_blackboard import initialize_agent_blackboard, update_blackboard_from_action
from cascade_planner.legacy.harness_runtime.preflight import run_preflight
from cascade_planner.legacy.harness_runtime.retrosynthetic_proposals import (
    compile_retrosynthetic_proposal_bus,
)
from cascade_planner.legacy.harness_runtime.schemas import TargetInput


class RetrosyntheticProposalBusTests(unittest.TestCase):
    def test_visual_connectivity_evidence_becomes_recursive_proposal(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
        )
        visual_result = {
            "schema_version": "visual_literature_chain_extraction_result.v1",
            "accepted": True,
            "chain_id": "visual_chain:test",
            "source_ref": "local_pdf:test",
            "steps": [
                {
                    "product_smiles": "CCO",
                    "main_reactant_smiles": "CC=O",
                    "reactant_labels": ["acetaldehyde"],
                    "confidence": "low",
                    "risk_flags": ["visual_connectivity_approximation"],
                    "source_locator": "scheme 1",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r1:visual", "action_type": "extract_visual_literature_chain"},
                action_result=visual_result,
                round_index=1,
                run_dir=Path(tmp),
            )

        self.assertGreaterEqual(len(board["reaction_idea_cards"]), 1)
        self.assertGreaterEqual(len(board["retrosynthetic_proposals"]), 1)
        proposal = board["retrosynthetic_proposals"][0]
        self.assertEqual(proposal["schema_version"], "retrosynthetic_proposal.v1")
        self.assertEqual(proposal["source_type"], "visual_connectivity_candidate")
        self.assertEqual(proposal["proposal_type"], "semi_executable")
        self.assertEqual(proposal["precursor_smiles"], "CC=O")
        self.assertTrue(proposal["recursive_expandable"])
        self.assertTrue(proposal["no_solved_claim"])
        self.assertTrue(proposal["not_parent_route_proof"])

        tasks = [
            row
            for row in board["recursive_hypothesis_tasks"]
            if row.get("source") == "retrosynthetic_proposal"
        ]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["precursor_smiles"], "CC=O")
        self.assertEqual(tasks[0]["proposal_id"], proposal["proposal_id"])
        self.assertTrue(tasks[0]["child_route_cannot_promote_parent"])
        self.assertIn("retrosynthetic_proposals", board["action_history"][-1]["changed_blackboard_fields"])
        self.assertTrue(board["action_history"][-1]["useful_artifact"])

    def test_guided_payload_consumes_retrosynthetic_proposals(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
        )
        board["retrosynthetic_proposals"] = [
            {
                "schema_version": "retrosynthetic_proposal.v1",
                "proposal_id": "proposal:test",
                "proposal_type": "semi_executable",
                "source_type": "analogical_template_hint",
                "proposal_label": "aldehyde precursor",
                "target_smiles": "CCO",
                "precursor_smiles": "CC=O",
                "transformation_idea": "reduce aldehyde to alcohol",
                "confidence": "medium",
                "recursive_expandable": True,
                "executable": True,
                "risk_flags": ["analogy_not_proof"],
                "required_verification": ["route_expansion_verifier"],
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "no_solved_claim": True,
            }
        ]

        payload = build_guided_chemenzy_payload_from_blackboard(board)
        policy = payload["search_policy"]

        self.assertIn("CC=O", policy["source_budget"]["preferred_precursor_smiles"])
        self.assertEqual(policy["source_budget"]["retrosynthetic_proposals"][0]["proposal_id"], "proposal:test")
        self.assertTrue(policy["source_budget"]["retrosynthetic_proposals_are_not_proof"])
        subgoals = policy["preferred_subgoal"]["hypothetical_precursor_targets"]
        self.assertTrue(any(row.get("source") == "retrosynthetic_proposal" for row in subgoals))
        self.assertIn("CC=O", policy["preferred_subgoal"]["preferred_subgoals"])

    def test_process_evidence_compiles_strategic_proposal_without_recursive_task(self):
        board = {
            "target_profile": {"target_smiles": "CCO"},
            "literature_evidence": {
                "process_evidence_rows": [
                    {
                        "schema_version": "literature_process_evidence_row.v1",
                        "row_id": "process:test",
                        "process_type": "whole_cell_biotransformation",
                        "endpoint_labels": ["ethanol"],
                        "substrate_or_feedstock_labels": ["glucose"],
                        "source_ref": "doi:10.example/process",
                        "confidence": "medium",
                        "risk_flags": [],
                        "verification_required": ["process_endpoint_acceptability"],
                    }
                ]
            },
            "semisynthesis_anchors": [],
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        self.assertTrue(report["accepted"])
        self.assertEqual(report["retrosynthetic_proposals"][0]["proposal_type"], "strategic")
        self.assertFalse(report["retrosynthetic_proposals"][0]["recursive_expandable"])
        self.assertEqual(report["recursive_hypothesis_tasks"], [])
        self.assertTrue(report["no_solved_claim"])

    def test_strategic_reaction_idea_concretizes_common_functional_handle(self):
        board = {
            "target_profile": {"target_smiles": "CCO"},
            "target_side_disconnection_hypotheses": {
                "hypotheses": [
                    {
                        "hypothesis_id": "target_side:alcohol",
                        "target_handle": "generic_functional_handles",
                        "proposed_disconnection_region": "alcohol handle redox/protection adjustment",
                        "expected_precursor_type": "nearby redox or protected alcohol precursor",
                        "must_preserve_substructure": ["carbon_skeleton"],
                        "confidence": "medium",
                        "required_verification": ["route_expansion_verifier"],
                    }
                ]
            },
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        concrete = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "deterministic_reaction_idea_concretization"
        ]
        self.assertGreaterEqual(len(concrete), 1)
        self.assertTrue(any(row["precursor_smiles"] == "CC=O" for row in concrete))
        self.assertTrue(all(row["proposal_type"] == "semi_executable" for row in concrete))
        self.assertTrue(all(row["not_parent_route_proof"] for row in concrete))
        self.assertTrue(all(row["no_solved_claim"] for row in concrete))
        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertIn("CC=O", task_precursors)

    def test_same_core_proposals_carry_granularity_into_child_payload(self):
        steroid_like_target = (
            "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
        )
        board = {
            "case_id": "same_core_probe",
            "target_profile": {"target_smiles": steroid_like_target, "heavy_atoms": 25, "rings": 4},
            "target_side_disconnection_hypotheses": {
                "hypotheses": [
                    {
                        "hypothesis_id": "target_side:same_core_redox",
                        "target_handle": "polycyclic same-core alcohol/enone handles",
                        "proposed_disconnection_region": "same-core redox or protecting-state migration",
                        "expected_precursor_type": "same-core oxidation-state or protected precursor",
                        "must_preserve_substructure": ["polycyclic core"],
                        "confidence": "medium",
                        "required_verification": ["route_expansion_verifier"],
                    }
                ]
            },
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
            "retrosynthetic_proposals": [],
            "terminal_blacklist": [],
            "action_history": [],
            "budget_state": {"child_target_runs": 0, "max_child_target_runs": 2},
            "current_belief": {"constraints": {"max_recursive_hypothesis_depth": 3}},
        }

        report = compile_retrosynthetic_proposal_bus(board)
        board["retrosynthetic_proposals"] = report["retrosynthetic_proposals"]
        board["recursive_hypothesis_tasks"] = report["recursive_hypothesis_tasks"]

        same_core = [
            row
            for row in report["retrosynthetic_proposals"]
            if row.get("proposal_granularity") == "same_core"
        ]
        self.assertGreaterEqual(len(same_core), 1)
        self.assertTrue(all(row["failure_response_policy"]["on_no_route"] for row in same_core))
        self.assertTrue(
            any(row.get("route_objective_type") == "same_core_redox_or_protection_route" for row in same_core)
        )
        self.assertTrue(
            any(row.get("proposal_granularity") == "same_core" for row in report["recursive_hypothesis_tasks"])
        )

        payload = build_child_expansion_payload_from_blackboard(board)
        subgoal = payload["subgoal_targets"][0]
        self.assertEqual(subgoal["proposal_granularity"], "same_core")
        self.assertEqual(subgoal["chem_enzy_search_policy"]["source_budget"]["proposal_granularity"], "same_core")
        self.assertEqual(
            subgoal["chem_enzy_search_policy"]["source_budget"]["route_objective_type"],
            "same_core_redox_or_protection_route",
        )
        self.assertTrue(subgoal["chem_enzy_search_policy"]["source_budget"]["failure_response_policy"])

    def test_strategic_reaction_idea_concretizes_ester_disconnection(self):
        board = {
            "target_profile": {"target_smiles": "CC(=O)Oc1ccccc1"},
            "target_side_disconnection_hypotheses": {
                "hypotheses": [
                    {
                        "hypothesis_id": "target_side:ester",
                        "target_handle": "aryl ester",
                        "proposed_disconnection_region": "acyl-oxygen ester disconnection",
                        "expected_precursor_type": "carboxylic acid or activated acyl donor plus phenol",
                        "must_preserve_substructure": ["aryl fragment"],
                        "confidence": "medium",
                        "required_verification": ["route_expansion_verifier"],
                    }
                ]
            },
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        concrete = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "deterministic_reaction_idea_concretization"
        ]
        acid_alcohol = [
            row
            for row in concrete
            if row["proposal_label"] == "ester_to_carboxylic_acid_alcohol_precursors"
        ]
        self.assertEqual(len(acid_alcohol), 1)
        proposal = acid_alcohol[0]
        self.assertEqual(proposal["precursor_smiles"], "CC(=O)O.Oc1ccccc1")
        self.assertEqual(proposal["precursor_component_count"], 2)
        self.assertTrue(proposal["multi_component_precursor_set"])
        self.assertTrue(proposal["recursive_expandable"])
        self.assertTrue(proposal["not_parent_route_proof"])
        self.assertTrue(proposal["no_solved_claim"])
        self.assertIn("multi_component_precursor_set", proposal["risk_flags"])

        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertNotIn("CC(=O)O.Oc1ccccc1", task_precursors)
        self.assertIn("CC(=O)O", task_precursors)
        acid_task = next(row for row in report["recursive_hypothesis_tasks"] if row["precursor_smiles"] == "CC(=O)O")
        self.assertEqual(acid_task["task_scope"], "precursor_component")
        self.assertEqual(acid_task["precursor_set_smiles"], "CC(=O)O.Oc1ccccc1")
        self.assertTrue(acid_task["requires_precursor_set_stitching"])
        self.assertIn("Oc1ccccc1", acid_task["sibling_precursor_smiles"])

    def test_guided_payload_consumes_multi_component_retrosynthetic_proposal(self):
        board = {
            "case_id": "phenyl_acetate",
            "target_profile": {"target_smiles": "CC(=O)Oc1ccccc1"},
            "current_belief": {"constraints": {}},
            "terminal_blacklist": [],
            "retrosynthetic_proposals": [
                {
                    "schema_version": "retrosynthetic_proposal.v1",
                    "proposal_id": "proposal:ester",
                    "proposal_type": "semi_executable",
                    "source_type": "deterministic_reaction_idea_concretization",
                    "proposal_label": "ester_to_carboxylic_acid_alcohol_precursors",
                    "target_smiles": "CC(=O)Oc1ccccc1",
                    "precursor_smiles": "CC(=O)O.Oc1ccccc1",
                    "precursor_component_count": 2,
                    "multi_component_precursor_set": True,
                    "transformation_idea": "disconnect ester into acid and phenol",
                    "confidence": "medium",
                    "recursive_expandable": True,
                    "executable": True,
                    "risk_flags": ["multi_component_precursor_set"],
                    "required_verification": ["route_expansion_verifier"],
                    "not_exact_literature_segment": True,
                    "not_parent_route_proof": True,
                    "no_solved_claim": True,
                }
            ],
        }

        payload = build_guided_chemenzy_payload_from_blackboard(board)
        policy = payload["search_policy"]

        self.assertIn("CC(=O)O.Oc1ccccc1", policy["source_budget"]["preferred_precursor_smiles"])
        self.assertIn("CC(=O)O", policy["source_budget"]["preferred_precursor_smiles"])
        self.assertIn("Oc1ccccc1", policy["source_budget"]["preferred_precursor_smiles"])
        subgoals = policy["preferred_subgoal"]["hypothetical_precursor_targets"]
        self.assertTrue(any(row.get("smiles") == "CC(=O)O.Oc1ccccc1" for row in subgoals))
        component = next(row for row in subgoals if row.get("smiles") == "CC(=O)O")
        self.assertEqual(component["source"], "retrosynthetic_proposal_component")
        self.assertEqual(component["precursor_set_smiles"], "CC(=O)O.Oc1ccccc1")
        self.assertTrue(component["requires_precursor_set_stitching"])
        proposal_rows = policy["source_budget"]["retrosynthetic_proposals"]
        self.assertTrue(proposal_rows[0]["multi_component_precursor_set"])
        self.assertTrue(policy["source_budget"]["retrosynthetic_proposals_are_not_proof"])

    def test_child_expansion_consumes_multi_component_proposal_components_not_set(self):
        board = {
            "target_profile": {"target_smiles": "CC(=O)Oc1ccccc1"},
            "target_side_disconnection_hypotheses": {
                "hypotheses": [
                    {
                        "hypothesis_id": "target_side:ester",
                        "target_handle": "aryl ester",
                        "proposed_disconnection_region": "acyl-oxygen ester disconnection",
                        "expected_precursor_type": "carboxylic acid or activated acyl donor plus phenol",
                        "must_preserve_substructure": ["aryl fragment"],
                        "confidence": "medium",
                        "required_verification": ["route_expansion_verifier"],
                    }
                ]
            },
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
        }
        report = compile_retrosynthetic_proposal_bus(board)
        board.update(
            {
                "case_id": "phenyl_acetate",
                "recursive_hypothesis_tasks": report["recursive_hypothesis_tasks"],
                "terminal_blacklist": [],
                "action_history": [],
            }
        )

        payload = build_child_expansion_payload_from_blackboard(board)
        targets = payload["subgoal_targets"]

        self.assertTrue(all(row["smiles"] != "CC(=O)O.Oc1ccccc1" for row in targets))
        acid = next(row for row in targets if row["smiles"] == "CC(=O)O")
        self.assertEqual(acid["task_scope"], "precursor_component")
        self.assertEqual(acid["precursor_set_smiles"], "CC(=O)O.Oc1ccccc1")
        self.assertTrue(acid["requires_precursor_set_stitching"])
        self.assertEqual(
            acid["chem_enzy_search_policy"]["preferred_subgoal"]["precursor_set_smiles"],
            "CC(=O)O.Oc1ccccc1",
        )
        self.assertTrue(
            acid["chem_enzy_search_policy"]["source_budget"]["requires_precursor_set_stitching"]
        )

    def test_failure_feedback_refines_failed_acyl_component(self):
        board = {
            "target_profile": {"target_smiles": "CC(=O)Oc1ccccc1"},
            "proposal_failure_feedback": [
                {
                    "schema_version": "proposal_failure_feedback.v1",
                    "feedback_id": "feedback:acid",
                    "proposal_id": "proposal:acid",
                    "parent_smiles": "CC(=O)Oc1ccccc1",
                    "failed_precursor_smiles": "CC(=O)O",
                    "precursor_set_smiles": "CC(=O)O.Oc1ccccc1",
                    "sibling_precursor_smiles": ["Oc1ccccc1"],
                    "failure_reasons": ["no_route_expansion_subgoal_verified_solved"],
                }
            ],
            "retrosynthetic_proposals": [],
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        refined = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "failure_driven_proposal_refinement"
        ]
        precursors = {row["precursor_smiles"] for row in refined}
        self.assertIn("CC(=O)Cl.Oc1ccccc1", precursors)
        self.assertIn("COC(C)=O.Oc1ccccc1", precursors)
        self.assertNotIn("Oc1ccccc1.[OH]", precursors)
        self.assertTrue(all(row["no_solved_claim"] for row in refined))
        self.assertTrue(all("failure_driven_refinement" in row["risk_flags"] for row in refined))

        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertIn("CC(=O)Cl", task_precursors)
        self.assertIn("COC(C)=O", task_precursors)

    def test_failure_feedback_does_not_refine_back_to_parent_target(self):
        board = {
            "target_profile": {"target_smiles": "CC(=O)Oc1ccccc1"},
            "proposal_failure_feedback": [
                {
                    "schema_version": "proposal_failure_feedback.v1",
                    "feedback_id": "feedback:phenol",
                    "proposal_id": "proposal:phenol",
                    "parent_smiles": "CC(=O)Oc1ccccc1",
                    "failed_precursor_smiles": "Oc1ccccc1",
                    "precursor_set_smiles": "",
                    "sibling_precursor_smiles": [],
                    "failure_reasons": ["no_route_expansion_subgoal_verified_solved"],
                }
            ],
            "retrosynthetic_proposals": [],
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        self.assertTrue(
            all(row.get("precursor_smiles") != "CC(=O)Oc1ccccc1" for row in report["retrosynthetic_proposals"])
        )
        self.assertTrue(
            all(row.get("precursor_smiles") != "CC(=O)Oc1ccccc1" for row in report["recursive_hypothesis_tasks"])
        )

    def test_child_failure_does_not_generate_parent_target_recursive_loop(self):
        target = TargetInput(target_name="phenyl acetate", target_smiles="CC(=O)Oc1ccccc1")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
        )
        route_result = {
            "schema_version": "route_expansion_subgoal_search_result.v1",
            "accepted": False,
            "status": "failed",
            "solved": False,
            "subgoals": [
                {
                    "schema_version": "route_expansion_subgoal_result.v1",
                    "accepted": False,
                    "solved": False,
                    "reasons": ["no_route_expansion_subgoal_verified_solved"],
                    "verifier": {"accepted": False, "reasons": ["target_unresolved"]},
                    "subgoal": {
                        "schema_version": "route_expansion_child_target.v1",
                        "name": "phenol precursor",
                        "smiles": "Oc1ccccc1",
                        "source": "recursive_hypothesis_task",
                        "hypothesis_only_not_solved": True,
                        "recursive_hypothesis_task_id": "recursive_hypothesis:phenol",
                        "parent_candidate_id": "proposal:phenol",
                        "parent_smiles": "CC(=O)Oc1ccccc1",
                        "policy": {
                            "compiler_metadata": {
                                "hypothesis_only_not_solved": True,
                                "recursive_hypothesis_frontier": True,
                            }
                        },
                    },
                }
            ],
            "reasons": ["no_route_expansion_subgoal_verified_solved"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r2:expand", "action_type": "expand_child_target"},
                action_result={"accepted": True, "result": route_result},
                round_index=2,
                run_dir=Path(tmp),
            )

        self.assertTrue(
            all(row.get("precursor_smiles") != "CC(=O)Oc1ccccc1" for row in board["recursive_hypothesis_tasks"])
        )
        self.assertTrue(
            all(row.get("precursor_smiles") != "CC(=O)Oc1ccccc1" for row in board["retrosynthetic_proposals"])
        )

    def test_child_failure_updates_blackboard_with_refinement_feedback(self):
        target = TargetInput(target_name="phenyl acetate", target_smiles="CC(=O)Oc1ccccc1")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
            budget_limits={"max_route_expansion_subgoal_runs": 4},
        )
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:enone_pair",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "rejected_hypothesis_precursor",
                "parent_smiles": "O=C1C=CCCC1",
                "precursor_smiles": "O=C1CCCCC1",
                "name": "enone_to_saturated_ketone_precursor",
                "recursive_depth": 1,
                "operation_idea": "continue through a saturated ketone frontier",
                "variant_type": "enone_to_saturated_ketone_precursor",
                "allowed_use": "route_expansion_subgoal_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "child_route_cannot_promote_parent": True,
                "no_solved_claim": True,
            }
        ]
        route_result = {
            "schema_version": "route_expansion_subgoal_search_result.v1",
            "accepted": False,
            "status": "failed",
            "solved": False,
            "subgoals": [
                {
                    "schema_version": "route_expansion_subgoal_result.v1",
                    "accepted": False,
                    "solved": False,
                    "reasons": ["no_route_expansion_subgoal_verified_solved"],
                    "verifier": {"accepted": False, "reasons": ["target_unresolved"]},
                    "subgoal": {
                        "schema_version": "route_expansion_child_target.v1",
                        "name": "ester_to_carboxylic_acid_alcohol_precursors:component:1",
                        "smiles": "CC(=O)O",
                        "source": "recursive_hypothesis_task",
                        "hypothesis_only_not_solved": True,
                        "recursive_hypothesis_task_id": "recursive_hypothesis:acid",
                        "parent_candidate_id": "proposal:acid",
                        "parent_smiles": "CC(=O)Oc1ccccc1",
                        "task_scope": "precursor_component",
                        "precursor_set_smiles": "CC(=O)O.Oc1ccccc1",
                        "precursor_component_index": 1,
                        "precursor_component_count": 2,
                        "multi_component_precursor_set": True,
                        "requires_precursor_set_stitching": True,
                        "sibling_precursor_smiles": ["Oc1ccccc1"],
                        "policy": {
                            "compiler_metadata": {
                                "hypothesis_only_not_solved": True,
                                "recursive_hypothesis_frontier": True,
                            }
                        },
                    },
                }
            ],
            "reasons": ["no_route_expansion_subgoal_verified_solved"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r2:expand", "action_type": "expand_child_target"},
                action_result={"accepted": True, "result": route_result},
                round_index=2,
                run_dir=Path(tmp),
            )

        self.assertEqual(len(board["proposal_failure_feedback"]), 1)
        feedback = board["proposal_failure_feedback"][0]
        self.assertEqual(feedback["failed_precursor_smiles"], "CC(=O)O")
        self.assertEqual(feedback["precursor_set_smiles"], "CC(=O)O.Oc1ccccc1")
        self.assertIn("try_alternate_acyl_activation_state", feedback["next_refinement_bias"])
        refined = [
            row
            for row in board["retrosynthetic_proposals"]
            if row["source_type"] == "failure_driven_proposal_refinement"
        ]
        self.assertTrue(any(row["precursor_smiles"] == "COC(C)=O.Oc1ccccc1" for row in refined))
        self.assertIn("proposal_failure_feedback", board["action_history"][-1]["changed_blackboard_fields"])
        self.assertIn("retrosynthetic_proposals", board["action_history"][-1]["changed_blackboard_fields"])
        self.assertTrue(board["action_history"][-1]["useful_artifact"])

    def test_child_failure_biases_redox_refinement_for_enone_pair_transfer(self):
        target = TargetInput(target_name="enone", target_smiles="O=C1C=CCCC1")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
            budget_limits={"max_route_expansion_subgoal_runs": 4},
        )
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:enone_pair",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "rejected_hypothesis_precursor",
                "parent_smiles": "O=C1C=CCCC1",
                "precursor_smiles": "O=C1CCCCC1",
                "name": "enone_to_saturated_ketone_precursor",
                "recursive_depth": 1,
                "operation_idea": "continue through a saturated ketone frontier",
                "variant_type": "enone_to_saturated_ketone_precursor",
                "allowed_use": "route_expansion_subgoal_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "child_route_cannot_promote_parent": True,
                "no_solved_claim": True,
            }
        ]
        route_result = {
            "schema_version": "route_expansion_subgoal_search_result.v1",
            "accepted": False,
            "status": "failed",
            "solved": False,
            "subgoals": [
                {
                    "schema_version": "route_expansion_subgoal_result.v1",
                    "accepted": False,
                    "solved": False,
                    "reasons": ["no_route_expansion_subgoal_verified_solved"],
                    "verifier": {"accepted": False, "reasons": ["target_unresolved"]},
                    "subgoal": {
                        "schema_version": "route_expansion_child_target.v1",
                        "name": "enone_to_saturated_ketone_precursor",
                        "smiles": "O=C1CCCCC1",
                        "source": "recursive_hypothesis_task",
                        "hypothesis_only_not_solved": True,
                        "recursive_hypothesis_task_id": "recursive_hypothesis:enone_pair",
                        "parent_candidate_id": "proposal:enone_pair",
                        "parent_smiles": "O=C1C=CCCC1",
                        "policy": {
                            "compiler_metadata": {
                                "hypothesis_only_not_solved": True,
                                "recursive_hypothesis_frontier": True,
                            }
                        },
                    },
                }
            ],
            "reasons": ["no_route_expansion_subgoal_verified_solved"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r2:expand", "action_type": "expand_child_target"},
                action_result={"accepted": True, "result": route_result},
                round_index=2,
                run_dir=Path(tmp),
            )

        feedback = board["proposal_failure_feedback"][0]
        self.assertIn("try_redox_or_unsaturation_state_refinement", feedback["next_refinement_bias"])
        self.assertNotIn("try_alternate_acyl_activation_state", feedback["next_refinement_bias"])
        failed_task = next(
            row for row in board["recursive_hypothesis_tasks"] if row["task_id"] == "recursive_hypothesis:enone_pair"
        )
        self.assertEqual(failed_task["status"], "rejected")
        self.assertEqual(failed_task["attempt_count"], 1)
        refined = [
            row
            for row in board["retrosynthetic_proposals"]
            if row["source_type"] == "failure_driven_proposal_refinement"
        ]
        self.assertTrue(any(row["proposal_label"] == "failed_ketone_to_secondary_alcohol_component" for row in refined))
        next_payload = build_child_expansion_payload_from_blackboard(board)
        next_smiles = [row["smiles"] for row in next_payload["subgoal_targets"]]
        self.assertNotIn("O=C1CCCCC1", next_smiles)
        self.assertIn("OC1CCCCC1", next_smiles)

        with tempfile.TemporaryDirectory() as tmp:
            repeated = update_blackboard_from_action(
                board,
                action={"action_id": "r3:expand", "action_type": "expand_child_target"},
                action_result={"accepted": True, "result": route_result},
                round_index=3,
                run_dir=Path(tmp),
            )

        self.assertEqual(len(repeated["proposal_failure_feedback"]), 1)
        repeated_task = next(
            row for row in repeated["recursive_hypothesis_tasks"] if row["task_id"] == "recursive_hypothesis:enone_pair"
        )
        self.assertEqual(repeated_task["status"], "rejected")
        self.assertEqual(repeated_task["attempt_count"], 2)
        self.assertFalse(repeated["action_history"][-1]["useful_artifact"])
        self.assertTrue(repeated["action_history"][-1]["stale"])

    def test_analogical_template_without_precursor_hint_is_concretized(self):
        board = {
            "target_profile": {"target_smiles": "CC(C)=O"},
            "template_applications": [
                {
                    "application_id": "template_app:test",
                    "template_id": "template:redox",
                    "accepted": True,
                    "product_retron_type": "ketone redox handle",
                    "hypothetical_route_hypothesis": {
                        "reaction_center_idea": "related analogs use alcohol oxidation/reduction around this ketone",
                        "risk_flags": ["analogical_scope"],
                    },
                    "hypothetical_precursor_hints": [],
                    "evidence_refs": ["doi:10.example/analog"],
                }
            ],
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        concrete = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "analogical_reaction_idea_concretization"
        ]
        self.assertTrue(any(row["precursor_smiles"] == "CC(C)O" for row in concrete))
        self.assertTrue(all(row["proposal_type"] == "semi_executable" for row in concrete))
        self.assertTrue(all("analogy_not_proof" in row["risk_flags"] for row in concrete))
        self.assertTrue(all(row["no_solved_claim"] for row in concrete))
        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertIn("CC(C)O", task_precursors)

    def test_analogical_source_pair_transfers_reaction_center_to_target(self):
        board = {
            "target_profile": {"target_smiles": "O=C1C=CCCC1"},
            "analogical_hypotheses": [
                {
                    "schema_version": "analogical_hypothesis.v1",
                    "hypothesis_id": "analogy:enone_reduction",
                    "reaction_family": "enone redox transfer",
                    "source_ref": "doi:10.example/analog-enone",
                    "source_product_smiles": "O=C1C=CCCC1",
                    "source_reactant_smiles": ["O=C1CCCCC1"],
                    "analogy_strength": "medium",
                    "evidence_refs": ["scheme:1"],
                    "risk_flags": ["analog_scope_unknown"],
                }
            ],
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        transferred = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "analogical_reaction_pair_transfer"
        ]
        self.assertTrue(
            any(
                row["proposal_label"] == "enone_to_saturated_ketone_precursor"
                and row["precursor_smiles"] == "O=C1CCCCC1"
                for row in transferred
            )
        )
        self.assertTrue(all("analogical_reaction_pair_transfer" in row["risk_flags"] for row in transferred))
        self.assertTrue(all(row["not_parent_route_proof"] for row in transferred))
        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertIn("O=C1CCCCC1", task_precursors)

    def test_analogical_source_pair_transfer_is_reaction_center_bounded(self):
        board = {
            "target_profile": {"target_smiles": "CC(=O)CC(=O)OC"},
            "analogical_hypotheses": [
                {
                    "schema_version": "analogical_hypothesis.v1",
                    "hypothesis_id": "analogy:ketone_alcohol",
                    "reaction_family": "ketone alcohol redox",
                    "source_ref": "doi:10.example/analog-redox",
                    "source_product_smiles": "CC(C)=O",
                    "source_reactant_smiles": ["CC(C)O"],
                    "analogy_strength": "medium",
                    "evidence_refs": ["scheme:2"],
                }
            ],
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        transferred = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "analogical_reaction_pair_transfer"
        ]
        labels = {row["proposal_label"] for row in transferred}
        self.assertIn("ketone_to_secondary_alcohol_precursor", labels)
        self.assertNotIn("ester_to_carboxylic_acid_alcohol_precursors", labels)
        self.assertTrue(any(row["precursor_smiles"] == "COC(=O)CC(C)O" for row in transferred))
        self.assertFalse(
            any(
                row["source_type"] == "analogical_reaction_idea_concretization"
                for row in report["retrosynthetic_proposals"]
            )
        )

    def test_enone_reaction_idea_concretizes_to_saturated_ketone(self):
        board = {
            "target_profile": {"target_smiles": "O=C1C=CCCC1"},
            "target_side_disconnection_hypotheses": {
                "hypotheses": [
                    {
                        "hypothesis_id": "target_side:enone",
                        "target_handle": "enone redox handle",
                        "proposed_disconnection_region": "enone oxidation-state adjustment",
                        "expected_precursor_type": "saturated ketone or allylic redox precursor",
                        "must_preserve_substructure": ["cyclohexanone core"],
                        "confidence": "medium",
                        "required_verification": ["route_expansion_verifier"],
                    }
                ]
            },
            "literature_evidence": {},
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        concrete = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "deterministic_reaction_idea_concretization"
        ]
        self.assertTrue(
            any(
                row["proposal_label"] == "enone_to_saturated_ketone_precursor"
                and row["precursor_smiles"] == "O=C1CCCCC1"
                for row in concrete
            )
        )
        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertIn("O=C1CCCCC1", task_precursors)

    def test_process_feedstock_smiles_becomes_recursive_anchor(self):
        board = {
            "target_profile": {"target_smiles": "CC(C)=O"},
            "literature_evidence": {
                "process_evidence_rows": [
                    {
                        "schema_version": "literature_process_evidence_row.v1",
                        "row_id": "process:feedstock",
                        "process_type": "whole_cell_biotransformation",
                        "endpoint_labels": ["acetone"],
                        "substrate_or_feedstock_labels": ["isopropanol"],
                        "substrate_or_feedstock_smiles": ["CC(C)O"],
                        "source_ref": "doi:10.example/process",
                        "confidence": "medium",
                        "risk_flags": ["organism_scope_unknown"],
                        "verification_required": ["process_endpoint_acceptability"],
                    }
                ]
            },
            "semisynthesis_anchors": [],
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        anchors = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "process_feedstock_anchor"
        ]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["precursor_smiles"], "CC(C)O")
        self.assertTrue(anchors[0]["recursive_expandable"])
        self.assertTrue(anchors[0]["not_parent_route_proof"])
        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertIn("CC(C)O", task_precursors)

    def test_failure_feedback_refines_failed_secondary_alcohol_to_ketone(self):
        board = {
            "target_profile": {"target_smiles": "CC(C)OC(C)=O"},
            "proposal_failure_feedback": [
                {
                    "schema_version": "proposal_failure_feedback.v1",
                    "feedback_id": "feedback:secondary_alcohol",
                    "proposal_id": "proposal:redox",
                    "parent_smiles": "CC(C)OC(C)=O",
                    "failed_precursor_smiles": "CC(C)O",
                    "precursor_set_smiles": "CC(C)O.CC(=O)O",
                    "sibling_precursor_smiles": ["CC(=O)O"],
                    "failure_reasons": ["no_route_expansion_subgoal_verified_solved"],
                }
            ],
            "retrosynthetic_proposals": [],
            "recursive_hypothesis_tasks": [],
        }

        report = compile_retrosynthetic_proposal_bus(board)

        refined = [
            row
            for row in report["retrosynthetic_proposals"]
            if row["source_type"] == "failure_driven_proposal_refinement"
        ]
        self.assertTrue(any(row["precursor_smiles"] == "CC(=O)O.CC(C)=O" for row in refined))
        self.assertTrue(any("redox_state_changed" in row["risk_flags"] for row in refined))
        task_precursors = {row["precursor_smiles"] for row in report["recursive_hypothesis_tasks"]}
        self.assertIn("CC(C)=O", task_precursors)


if __name__ == "__main__":
    unittest.main()
