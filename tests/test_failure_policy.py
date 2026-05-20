import tempfile
import unittest
from pathlib import Path

import torch

from cascade_planner.agent.failure_policy import (
    failure_row_from_payload,
    predict_failure_risk,
    retry_policy_from_failure_risk,
    suggestions_from_labels,
)
from cascade_planner.eval.train_failure_classifier_from_pack import (
    FailureClassifier,
    build_dataset,
)


class FailurePolicyTest(unittest.TestCase):
    def test_failure_row_from_payload_aggregates_route_metrics(self):
        payload = {
            "target": "CCO",
            "ui_metadata": {"domain": "chemoenzymatic"},
            "search_status": {"best_depth": 3, "solved": False},
            "routes": [
                {
                    "n_steps": 3,
                    "metrics": {
                        "filled_route": True,
                        "strict_stock_solve": False,
                        "condition": {"condition_window_success": False},
                        "cascade_compatibility": {"cascade_compatibility_success": True},
                    },
                }
            ],
        }

        row = failure_row_from_payload(payload)

        self.assertEqual(row["target_smiles"], "CCO")
        self.assertEqual(row["route_domain"], "chemoenzymatic")
        self.assertEqual(row["depth"], 3)
        self.assertEqual(row["metrics"]["filled_route_any"], True)
        self.assertEqual(row["metrics"]["strict_stock_solve_any"], False)
        self.assertEqual(row["metrics"]["condition_window_success_any"], False)

    def test_suggestions_from_labels_maps_retry_actions(self):
        suggestions = suggestions_from_labels([
            {"label": "generator_exact_miss", "probability": 0.8},
            {"label": "stock_dead_end", "probability": 0.7},
        ])
        actions = [s.action for s in suggestions]

        self.assertIn("increase_candidate_budget", actions)
        self.assertIn("try_alternative_route_mode", actions)

    def test_new_recovery_bottleneck_labels_map_to_retry_actions(self):
        suggestions = suggestions_from_labels([
            {"label": "candidate_generator_reactant_miss", "probability": 0.8},
            {"label": "route_composition_or_order_miss", "probability": 0.7},
        ])
        actions = [s.action for s in suggestions]

        self.assertIn("increase_candidate_budget", actions)

        policy = retry_policy_from_failure_risk(
            {
                "active_labels": [
                    {"label": "candidate_generator_reaction_detail_miss", "probability": 0.8},
                    {"label": "selector_missed_gt_reactant_candidate", "probability": 0.7},
                ]
            },
            {
                "search_mode": "adaptive",
                "min_steps": 3,
                "max_steps": 4,
                "skeleton_samples": 2,
                "candidate_budget": 4,
                "expansion_budget": 128,
                "n_results": 3,
                "solved": False,
            },
        )

        self.assertTrue(policy["would_retry"])
        self.assertTrue(policy["automatic_retry_safe"])
        self.assertGreaterEqual(policy["adjusted_settings"]["candidate_budget"], 12)

    def test_retry_policy_raises_bounded_search_knobs(self):
        risk = {
            "active_labels": [
                {"label": "generator_exact_miss", "probability": 0.8},
                {"label": "stock_dead_end", "probability": 0.7},
            ]
        }
        policy = retry_policy_from_failure_risk(risk, {
            "search_mode": "adaptive",
            "min_steps": 3,
            "max_steps": 4,
            "skeleton_samples": 2,
            "candidate_budget": 4,
            "expansion_budget": 128,
            "n_results": 3,
            "solved": False,
        })

        self.assertTrue(policy["would_retry"])
        self.assertFalse(policy["automatic_retry_safe"])
        self.assertGreaterEqual(policy["adjusted_settings"]["candidate_budget"], 8)
        self.assertGreaterEqual(policy["adjusted_settings"]["skeleton_samples"], 4)
        self.assertEqual(policy["adjusted_settings"]["max_steps"], 5)
        self.assertEqual(policy["adjusted_settings"]["retry_search_mode"], "stock_rescue")
        self.assertEqual(policy["adjusted_settings"]["planner_strategy"], "stock_rescue")
        self.assertIn("candidate_budget", policy["changed_settings"])

    def test_retry_policy_keeps_solved_cascade_quality_retry_manual(self):
        risk = {
            "active_labels": [
                {"label": "compatibility_failure", "probability": 0.9},
                {"label": "condition_failure", "probability": 0.8},
            ]
        }
        policy = retry_policy_from_failure_risk(risk, {
            "search_mode": "adaptive",
            "min_steps": 3,
            "max_steps": 3,
            "skeleton_samples": 1,
            "candidate_budget": 2,
            "expansion_budget": 24,
            "n_results": 1,
            "solved": True,
        })

        self.assertTrue(policy["would_retry"])
        self.assertFalse(policy["automatic_retry_safe"])
        self.assertEqual(policy["adjusted_settings"]["condition_strategy"], "seek_condition_compatible_alternative")

    def test_predict_failure_risk_with_tiny_checkpoint(self):
        rows = [
            {
                "target_smiles": "CCO",
                "route_domain": "chemoenzymatic",
                "depth": 3,
                "n_routes": 1,
                "labels": ["generator_exact_miss", "stock_dead_end"],
                "has_failure_label": True,
                "metrics": {"plan": True, "strict_stock_solve_any": False},
            },
            {
                "target_smiles": "CCC",
                "route_domain": "all_chemical",
                "depth": 2,
                "n_routes": 2,
                "labels": [],
                "has_failure_label": False,
                "metrics": {"plan": True, "strict_stock_solve_any": True},
            },
        ]
        dataset = build_dataset(rows, n_bits=16, min_label_count=1)
        model = FailureClassifier(dataset.x.shape[1], len(dataset.labels), hidden=16)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "failure.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "feature_schema": dataset.feature_schema,
                "hidden": 16,
            }, path)
            result = predict_failure_risk(rows[0], model_path=path, threshold=0.0)

        self.assertTrue(result["available"])
        self.assertEqual({x["label"] for x in result["labels"]}, set(dataset.labels))
        self.assertTrue(result["search_suggestions"])


if __name__ == "__main__":
    unittest.main()
