import unittest

from cascade_planner.eval.stock_failure_audit import build_stock_failure_audit


class StockFailureAuditTest(unittest.TestCase):
    def test_classifies_generator_selector_and_stock_closure_failures(self):
        run = {
            "targets": [
                {
                    "index": 1,
                    "target_smiles": "A",
                    "route_domain": "all_chemical",
                    "metrics": {"plan": True, "strict_stock_solve_any": False},
                    "route_recovery": {"candidate_gt_reactant_in_pool": False},
                },
                {
                    "index": 2,
                    "target_smiles": "B",
                    "route_domain": "all_chemical",
                    "metrics": {"plan": True, "strict_stock_solve_any": False},
                    "route_recovery": {
                        "candidate_gt_reactant_in_pool": True,
                        "gt_reactant_in_route_pool": False,
                    },
                },
                {
                    "index": 3,
                    "target_smiles": "C",
                    "route_domain": "all_enzymatic",
                    "metrics": {"plan": True, "strict_stock_solve_any": False},
                    "route_recovery": {
                        "candidate_gt_reactant_in_pool": True,
                        "gt_reactant_in_route_pool": True,
                    },
                    "planner_output": {
                        "routes": [
                            {
                                "metrics": {
                                    "retrosynthesis_progress": {
                                        "leaf_stock_status": {"CC": True, "CCCCCCCC": False}
                                    }
                                }
                            }
                        ]
                    },
                },
                {
                    "index": 4,
                    "target_smiles": "D",
                    "route_domain": "all_enzymatic",
                    "metrics": {"plan": True, "strict_stock_solve_any": True},
                    "route_recovery": {},
                },
            ]
        }

        audit = build_stock_failure_audit(run)

        self.assertEqual(audit["stock_failure_count"], 3)
        self.assertEqual(audit["reason_counts"]["generator_gt_reactant_miss"], 1)
        self.assertEqual(audit["reason_counts"]["selector_missed_gt_reactant_candidate"], 1)
        self.assertEqual(audit["reason_counts"]["stock_closure_after_route_hit"], 1)
        self.assertEqual(audit["failures"][2]["nonstock_terminal_examples"], ["CCCCCCCC"])


if __name__ == "__main__":
    unittest.main()
