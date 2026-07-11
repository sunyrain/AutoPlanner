import tempfile
import unittest
from pathlib import Path

from cascade_planner.harness.agent_action_planner import (
    independent_literature_source_keys,
    plan_literature_evidence_followup_actions,
    planned_child_target_count,
    validate_action_batch,
)
from cascade_planner.harness.agentic_blackboard import update_budget_for_action
from cascade_planner.harness.codex_action_planner import (
    _locally_repair_invalid_codex_batch,
    _normalize_codex_batch,
)
from cascade_planner.harness.retrosynthetic_proposals import (
    deduplicate_retrosynthetic_proposals,
)


def _board() -> dict:
    return {
        "case_id": "action_evidence_loop",
        "target_profile": {
            "valid": True,
            "target_name": "complex target",
            "target_smiles": "CCO",
            "canonical_smiles": "CCO",
            "heavy_atoms": 30,
            "rings": 4,
            "stereocenters": 4,
            "family_hint": "natural product",
        },
        "target_side_disconnection_hypotheses": {"hypotheses": [{"hypothesis_id": "h1"}]},
        "literature_evidence": {
            "source_candidates": [],
            "source_refs": [],
            "pdf_structure_evidence": [],
            "visual_chains": [],
            "exact_rows": [],
            "structure_resolution_tasks": [],
        },
        "budget_state": {
            "scout_calls": 0,
            "max_scout_calls": 3,
            "visual_calls": 0,
            "max_visual_calls": 3,
            "child_target_runs": 0,
            "max_child_target_runs": 1,
        },
        "action_history": [],
        "bridge_tasks": [],
        "current_belief": {"constraints": {}},
    }


class ActionEvidenceLoopTests(unittest.TestCase):
    def test_singular_child_target_normalizes_and_costs_one_run(self):
        board = _board()
        raw = {
            "schema_version": "agent_action_batch.v1",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "one_child",
                    "action_type": "expand_child_target",
                    "rationale": "test one explicit precursor",
                    "expected_artifact": "route_expansion_subgoal_search_result.v1",
                    "success_condition": "child verifier is recorded",
                    "payload": {
                        "subgoal_target": {"name": "ethane", "smiles": "CC"},
                    },
                }
            ],
        }

        batch = _normalize_codex_batch(raw, blackboard=board, round_index=1)
        payload = batch["actions"][0]["payload"]

        self.assertEqual(payload["max_targets"], 1)
        self.assertEqual(len(payload["subgoal_targets"]), 1)
        self.assertEqual(batch["normalization_audit"]["changed_action_count"], 1)
        self.assertFalse(batch["normalization_audit"]["silent_repair"])
        self.assertEqual(planned_child_target_count(payload), 1)
        self.assertTrue(validate_action_batch(batch, blackboard=board)["accepted"])
        updated = update_budget_for_action(board, "expand_child_target", payload)
        self.assertEqual(updated["budget_state"]["child_target_runs"], 1)

    def test_local_repair_preserves_three_source_search_and_audits_drop(self):
        board = _board()
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "search",
                    "action_type": "search_literature",
                    "rationale": "find independent sources",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "three source groups",
                    "payload": {
                        "search_intent": "target_proximal_source_discovery",
                        "queries": ["complex target total synthesis"],
                        "max_sources": 3,
                    },
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "premature_critic",
                    "action_type": "build_failure_critic_report",
                    "rationale": "critic",
                    "expected_artifact": "failure_critic_report.v1",
                    "success_condition": "critic",
                    "payload": {},
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }
        repaired = _locally_repair_invalid_codex_batch(
            batch,
            validation={
                "accepted": False,
                "reasons": ["failure_critic_requires_failure_evidence:1"],
            },
            blackboard=board,
        )

        self.assertIsNotNone(repaired)
        self.assertEqual([row["action_id"] for row in repaired["actions"]], ["search"])
        self.assertEqual(repaired["actions"][0]["payload"]["max_sources"], 3)
        audit = repaired["repair_audit"]
        self.assertFalse(audit["silent_repair"])
        self.assertIn("failure_critic_requires_failure_evidence:1", audit["trigger_reasons"])
        self.assertEqual(audit["dropped_action_ids"], ["premature_critic"])

    def test_mixed_source_repair_reserves_bound_followup_before_discovery(self):
        board = _board()
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1000/bound",
                "doi": "10.1000/bound",
                "local_pdf": "bound.pdf",
            }
        ]
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "search",
                    "action_type": "search_literature",
                    "rationale": "find independent sources",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "sources",
                    "payload": {
                        "search_intent": "independent_source_expansion",
                        "queries": ["independent synthesis"],
                        "max_sources": 3,
                    },
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "render_bound",
                    "action_type": "extract_pdf_literature_structures",
                    "rationale": "render the acquired source",
                    "expected_artifact": "literature_pdf_structure_evidence.v1",
                    "success_condition": "pages rendered",
                    "payload": {"source_ref": "doi:10.1000/bound", "pdf_path": "bound.pdf"},
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }
        repaired = _locally_repair_invalid_codex_batch(
            batch,
            validation={"accepted": False, "reasons": ["literature_source_round_budget_exceeded"]},
            blackboard=board,
        )

        self.assertEqual([row["action_id"] for row in repaired["actions"]], ["search", "render_bound"])
        self.assertEqual(repaired["actions"][0]["payload"]["max_sources"], 2)
        search_change = next(
            row for row in repaired["repair_audit"]["payload_changes"] if row["action_id"] == "search"
        )
        self.assertIn("max_sources", search_change["changed_payload_fields"])

    def test_source_lifecycle_advances_pdf_then_visual(self):
        board = _board()
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            board["literature_evidence"]["source_candidates"] = [
                {
                    "source_ref": "doi:10.1000/source",
                    "doi": "10.1000/source",
                    "local_pdf": str(pdf),
                }
            ]

            first = plan_literature_evidence_followup_actions(board, round_index=2)
            self.assertEqual(first[0]["action_type"], "extract_pdf_literature_structures")

            board["literature_evidence"]["pdf_structure_evidence"] = [
                {
                    "accepted": True,
                    "source_ref": "doi:10.1000/source",
                    "source_pdf_path": str(pdf),
                    "rendered_page_count": 1,
                    "rendered_pages": [{"image_path": str(pdf)}],
                    "reasons": [],
                }
            ]
            second = plan_literature_evidence_followup_actions(board, round_index=3)
            self.assertEqual(second[0]["action_type"], "extract_visual_literature_chain")

    def test_metadata_source_requests_accessible_fulltext_and_an_independent_source(self):
        board = _board()
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1000/first",
                "doi": "10.1000/first",
                "url": "https://doi.org/10.1000/first",
                "access_status": "metadata_only",
            }
        ]

        actions = plan_literature_evidence_followup_actions(board, round_index=2)
        action = actions[0]
        self.assertEqual(action["action_type"], "search_literature")
        self.assertEqual(action["payload"]["search_intent"], "source_detail_html_or_pdf_acquisition")
        self.assertEqual(action["payload"]["max_sources"], 2)
        self.assertEqual(action["payload"]["minimum_independent_sources"], 2)

    def test_article_and_si_are_one_independent_source_group(self):
        board = _board()
        board["literature_evidence"]["source_candidates"] = [
            {"doi": "10.1000/shared", "source_ref": "doi:10.1000/shared", "local_pdf": "article.pdf"},
            {
                "doi": "10.1000/shared",
                "source_ref": "doi:10.1000/shared",
                "local_pdf": "supporting-information.pdf",
            },
            {"doi": "10.1000/independent", "source_ref": "doi:10.1000/independent"},
        ]
        self.assertEqual(
            independent_literature_source_keys(board),
            {"doi:10.1000/shared", "doi:10.1000/independent"},
        )

    def test_duplicate_projections_merge_to_one_semantic_edge(self):
        rows = [
            {
                "proposal_id": "visual",
                "proposal_type": "semi_executable",
                "source_type": "visual_chain",
                "target_smiles": "CCO",
                "precursor_smiles": "CC.O",
                "score": 60,
                "evidence_refs": ["visual:1"],
                "risk_flags": ["visual_only"],
                "required_verification": ["route_verifier"],
                "recursive_expandable": True,
                "not_exact_literature_segment": True,
            },
            {
                "proposal_id": "exact",
                "proposal_type": "exact_executable",
                "source_type": "exact_row",
                "target_smiles": "OCC",
                "precursor_smiles": "O.CC",
                "score": 90,
                "evidence_refs": ["exact:1"],
                "risk_flags": [],
                "required_verification": ["reaction_verifier"],
                "executable": True,
                "recursive_expandable": True,
                "not_exact_literature_segment": False,
            },
        ]

        merged = deduplicate_retrosynthetic_proposals(rows)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["projection_count"], 2)
        self.assertEqual(set(merged[0]["projection_source_types"]), {"visual_chain", "exact_row"})
        self.assertEqual(set(merged[0]["evidence_refs"]), {"visual:1", "exact:1"})
        self.assertFalse(merged[0]["not_exact_literature_segment"])
        self.assertTrue(merged[0]["projection_support_is_not_independent_proof"])

        refreshed = deduplicate_retrosynthetic_proposals([merged[0], rows[1]])
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(set(refreshed[0]["projection_source_types"]), {"visual_chain", "exact_row"})
        self.assertEqual(set(refreshed[0]["projection_proposal_ids"]), {"visual", "exact"})


if __name__ == "__main__":
    unittest.main()
