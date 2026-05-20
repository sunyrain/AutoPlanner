import unittest

from cascade_planner.eval.gate_phase_selector_promotion import _candidate_gate, _context_gate, _risk_guarded_rows, _risk_guarded_summary


class GatePhaseSelectorPromotionTest(unittest.TestCase):
    def test_candidate_not_promoted_when_only_ties_rule_post(self):
        rows = [
            {
                "dataset": "demo",
                "rule_top1_product_usable_rate": 0.8,
                "rule_top3_product_usable_rate": 0.8,
                "rule_top3_artifact_rate": 0.1,
                "rule_top3_trivial_stock_closure_rate": 0.1,
                "rule_top3_generic_route_rate": 0.2,
                "model_top1_product_usable_rate": 0.8,
                "model_top3_product_usable_rate": 0.8,
                "model_top3_artifact_rate": 0.1,
                "model_top3_trivial_stock_closure_rate": 0.1,
                "model_top3_generic_route_rate": 0.2,
            }
        ]

        gate = _candidate_gate(
            "model",
            rows,
            candidate_prefix="model",
            baseline_prefix="rule",
            min_product_usable_gain=0.01,
            max_quality_regression=0.0,
        )

        self.assertFalse(gate["promote_product_default"])
        self.assertIn("does not beat", gate["reason"])

    def test_candidate_not_promoted_when_generic_regresses(self):
        rows = [
            {
                "dataset": "demo",
                "rule_top1_product_usable_rate": 0.8,
                "rule_top3_product_usable_rate": 0.8,
                "rule_top3_artifact_rate": 0.1,
                "rule_top3_trivial_stock_closure_rate": 0.1,
                "rule_top3_generic_route_rate": 0.2,
                "model_top1_product_usable_rate": 0.82,
                "model_top3_product_usable_rate": 0.82,
                "model_top3_artifact_rate": 0.1,
                "model_top3_trivial_stock_closure_rate": 0.1,
                "model_top3_generic_route_rate": 0.3,
            }
        ]

        gate = _candidate_gate(
            "model",
            rows,
            candidate_prefix="model",
            baseline_prefix="rule",
            min_product_usable_gain=0.01,
            max_quality_regression=0.0,
        )

        self.assertFalse(gate["promote_product_default"])
        generic_check = [
            row for row in gate["checks"]
            if row["name"] == "no_regression_top3_generic_route_rate"
        ][0]
        self.assertFalse(generic_check["ok"])

    def test_context_gate_passes_when_all_deltas_are_large_enough(self):
        gate = _context_gate(
            {
                "diagnostics": {
                    "cascade_original_minus_feature_shuffle_mrr": 0.15,
                    "cascade_original_minus_label_shuffle_mrr": 0.14,
                    "cascade_original_minus_native_rank_mrr": 0.47,
                }
            },
            min_delta=0.05,
        )

        self.assertTrue(gate["passed"])

    def test_risk_guarded_summary_reports_no_regression_without_product_gain(self):
        comparison = {
            "rows": [
                {
                    "dataset": "demo",
                    "rule": {
                        "top1_product_usable_rate": 0.8,
                        "top3_product_usable_rate": 0.8,
                        "top3_artifact_rate": 0.1,
                        "top3_trivial_stock_closure_rate": 0.1,
                        "top3_generic_route_rate": 0.2,
                    },
                    "runtime_ccts": {
                        "top1_product_usable_rate": 0.8,
                        "top3_product_usable_rate": 0.8,
                        "top3_artifact_rate": 0.1,
                        "top3_trivial_stock_closure_rate": 0.1,
                        "top3_generic_route_rate": 0.2,
                    },
                    "cascade_only": {
                        "top1_product_usable_rate": 0.8,
                        "top3_product_usable_rate": 0.8,
                        "top3_artifact_rate": 0.1,
                        "top3_trivial_stock_closure_rate": 0.0,
                        "top3_generic_route_rate": 0.1,
                    },
                }
            ]
        }

        rows = _risk_guarded_rows(comparison)
        summary = _risk_guarded_summary(comparison, {})

        self.assertEqual(rows[0]["runtime_ccts_top3_product_usable_rate"], 0.8)
        self.assertTrue(summary["runtime_ccts_no_regression_vs_rule"])
        self.assertTrue(summary["cascade_only_no_regression_vs_rule"])
        self.assertEqual(summary["top3_usable_gain_any"], 0.0)


if __name__ == "__main__":
    unittest.main()
