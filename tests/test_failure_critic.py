import unittest

from cascade_planner.harness.failure_critic import compile_failure_critic_report


class FailureCriticTest(unittest.TestCase):
    def test_large_atom_jump_creates_target_bridge_and_core_constraint(self):
        report = compile_failure_critic_report(
            artifacts={
                "route_verifier": {
                    "schema_version": "harness_route_verifier_report.v1",
                    "accepted": False,
                    "route_status": "fake_closed_rejected",
                    "reasons": ["large_atom_jump"],
                    "failure_events": [{"reason": "large_atom_jump"}],
                }
            },
            case_id="mla_case",
            target_name="MLA",
        )

        task_types = {row["task_type"] for row in report["bridge_tasks"]}
        self.assertTrue(report["accepted"])
        self.assertIn("target_proximal_bridge_required", task_types)
        self.assertTrue(report["constraints"]["target_core_retention_required"])
        self.assertIn("generate_disconnection_hypotheses", report["next_action_bias"])

    def test_plugin_not_invoked_creates_source_product_bridge_bias(self):
        report = compile_failure_critic_report(
            artifacts={
                "guided_chemenzy": {
                    "literature_template_plugin_runtime": {
                        "schema_version": "literature_template_plugin_runtime_diagnostics.v1",
                        "enabled_in_request": True,
                        "request_one_step_row_count": 2,
                        "calls": 0,
                        "added_candidates": 0,
                        "reasons": ["literature_template_plugin_not_invoked"],
                    }
                }
            },
            case_id="plugin_case",
            target_name="MLA",
        )

        task_types = {row["task_type"] for row in report["bridge_tasks"]}
        self.assertIn("bridge_to_literature_product_required", task_types)
        self.assertEqual(report["constraints"]["exact_replay_priority"], "lower_until_bridge_found")
        self.assertIn("rank_analogical_hypotheses", report["next_action_bias"])

    def test_advanced_terminal_enters_blacklist_and_child_task(self):
        advanced = "CN1CC2CCC1CC2OC(C)=O"
        report = compile_failure_critic_report(
            artifacts={
                "route_verifier": {
                    "accepted": False,
                    "route_status": "fake_closed_rejected",
                    "reasons": ["advanced_same_scaffold_terminal"],
                    "rejected_terminal_list": [
                        {
                            "smiles": advanced,
                            "canonical_smiles": advanced,
                            "heavy_atoms": 18,
                            "target_similarity": 0.72,
                            "reason": "advanced_same_scaffold_terminal",
                        }
                    ],
                }
            },
            case_id="advanced_case",
            target_name="MLA",
        )

        self.assertEqual(report["terminal_blacklist"][0]["canonical_smiles"], advanced)
        self.assertIn("upstream_terminal_synthesis", {row["task_type"] for row in report["bridge_tasks"]})


if __name__ == "__main__":
    unittest.main()
