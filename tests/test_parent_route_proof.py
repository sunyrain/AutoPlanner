import unittest

from cascade_planner.harness.parent_route_proof import compile_stitched_parent_route_proof


class ParentRouteProofTest(unittest.TestCase):
    def test_child_only_solved_does_not_pass_parent_proof(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            child_route={"accepted": True, "solved": True},
            exact_literature_segment={"accepted": True, "parent_route_connected": False},
        )

        self.assertFalse(proof["accepted"])
        self.assertIn("parent_route_verifier_not_accepted", proof["reasons"])
        self.assertIn("child_target_route_not_connected_to_parent_bridge", proof["reasons"])

    def test_unsolved_child_does_not_report_child_solved_parent_unresolved(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            parent_verifier={
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "target_match": True,
                "target_equivalence_audit": {"target_match": True},
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump"}],
            },
            child_route={"accepted": False, "solved": False, "accepted_subgoal_count": 0},
            exact_literature_segment={"accepted": False, "row_count": 0},
            stock_audit={"stock_audit_passed": False},
        )

        self.assertFalse(proof["accepted"])
        self.assertEqual(proof["route_status"], "fake_closed_rejected")
        self.assertNotEqual(proof["route_status"], "child_solved_parent_unresolved")
        self.assertIn("child_target_route_not_connected_to_parent_bridge", proof["reasons"])

    def test_disconnected_exact_row_does_not_pass(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            parent_verifier={
                "accepted": True,
                "route_status": "solved",
                "target_match": True,
                "target_equivalence_audit": {"target_match": True},
            },
            child_route={"accepted": True, "parent_bridge_connected": True},
            exact_literature_segment={"accepted": True, "parent_route_connected": False},
            stock_audit={"stock_audit_passed": True},
        )

        self.assertFalse(proof["accepted"])
        self.assertEqual(proof["route_status"], "partial_anchor_only_not_solved")
        self.assertIn("exact_literature_segment_not_connected_to_parent_route", proof["reasons"])

    def test_verifier_and_stitch_connectivity_passes(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            parent_verifier={
                "accepted": True,
                "route_status": "solved",
                "target_match": True,
                "target_equivalence_audit": {"target_match": True},
            },
            stitched_route={
                "accepted": True,
                "stock_audit_passed": True,
                "target": {"identity_audit": {"required": True, "target_match": True}},
                "terminal_match_audit": {"accepted": True},
                "subgoal_closure": {"verifier_accepted": True},
                "literature_chain": {"chain_accepted": True},
            },
            exact_literature_segment={"accepted": True, "parent_route_connected": True},
        )

        self.assertTrue(proof["accepted"], proof["reasons"])
        self.assertTrue(proof["proof_clauses"]["analogy_used_only_as_rationale"])
        self.assertEqual(proof["route_status"], "solved")

    def test_accepted_stitch_is_not_blocked_by_prior_failed_guided_route(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            parent_verifier={
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "target_match": True,
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump"}],
            },
            stitched_route={
                "accepted": True,
                "solved": True,
                "route_status": "solved",
                "stock_audit_passed": True,
                "target": {"identity_audit": {"required": True, "target_match": True}},
                "terminal_match_audit": {"accepted": True},
                "subgoal_closure": {"verifier_accepted": True},
                "literature_chain": {"chain_accepted": True},
            },
            exact_literature_segment={"accepted": True, "parent_route_connected": True},
        )

        self.assertTrue(proof["accepted"], proof["reasons"])
        self.assertNotIn("unexplained_large_atom_jump", proof["reasons"])


if __name__ == "__main__":
    unittest.main()
