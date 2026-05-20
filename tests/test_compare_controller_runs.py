import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.compare_controller_runs import compare_controller_runs, write_markdown


class CompareControllerRunsTest(unittest.TestCase):
    def test_compare_controller_runs_reports_target_flips_and_rescue_counts(self):
        baseline = {
            "summary": {
                "strict_stock_solve_any": 0.0,
                "gt_reactant_in_route_pool": 1.0,
                "avg_time_per_target_s": 2.0,
            },
            "targets": [
                {
                    "index": 1,
                    "target_smiles": "CCCC",
                    "route_domain": "all_chemical",
                    "elapsed_s": 2.0,
                    "metrics": {
                        "plan": True,
                        "strict_stock_solve_any": False,
                        "skeleton_type_GT@1": True,
                    },
                    "route_recovery": {
                        "candidate_gt_reactant_in_pool": True,
                        "gt_reactant_in_route_pool": True,
                        "exact_reaction_in_route_pool": True,
                    },
                }
            ],
        }
        candidate = {
            "summary": {
                "strict_stock_solve_any": 1.0,
                "gt_reactant_in_route_pool": 0.0,
                "avg_time_per_target_s": 3.5,
            },
            "targets": [
                {
                    "index": 1,
                    "target_smiles": "CCCC",
                    "route_domain": "all_chemical",
                    "elapsed_s": 3.5,
                    "metrics": {
                        "plan": True,
                        "strict_stock_solve_any": True,
                        "skeleton_type_GT@1": False,
                    },
                    "route_recovery": {
                        "candidate_gt_reactant_in_pool": True,
                        "gt_reactant_in_route_pool": False,
                        "exact_reaction_in_route_pool": True,
                    },
                    "planner_output": {
                        "routes": [
                            {
                                "explanation": {
                                    "uncertainty_table": {"stock_rescue_retries": 3}
                                }
                            }
                        ]
                    },
                }
            ],
        }

        report = compare_controller_runs(baseline, candidate)

        self.assertEqual(report["change_counts"]["strict_stock_solve_any_gained"], 1)
        self.assertEqual(report["change_counts"]["gt_reactant_in_route_pool_lost"], 1)
        self.assertEqual(report["change_counts"]["skeleton_type_GT@1_lost"], 1)
        self.assertEqual(report["changed_targets"][0]["candidate_stock_rescue_retries"], 3)
        self.assertEqual(report["delta_summary"]["avg_time_per_target_s"], 1.5)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "compare.md"
            write_markdown(report, path)
            self.assertIn("strict_stock_solve_any_gained", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
