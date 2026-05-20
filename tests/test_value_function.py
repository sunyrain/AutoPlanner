import unittest

from cascade_planner.cascadeboard import CascadeBoard
from cascade_planner.cascadeboard.value_function import (
    RouteValueFunction,
    candidate_value_features,
    metric_value_features,
)


class ValueFunctionTest(unittest.TestCase):
    def test_candidate_value_rewards_stock_and_main_reduction(self):
        cand = {
            "main_reactant": "CC",
            "aux_reactants": ["CO"],
            "rxn_smiles": "CC.CO>>CCCCCC",
            "score": 0.5,
        }

        features = candidate_value_features(
            "CCCCCC",
            cand,
            stock_checker=lambda smi: smi in {"CC", "CO"},
        )
        value = RouteValueFunction().score_candidate("CCCCCC", cand, stock_checker=lambda smi: smi in {"CC", "CO"})

        self.assertEqual(features["stock_fraction"], 1.0)
        self.assertGreater(features["main_reduction"], 0.0)
        self.assertGreater(value.probability, 0.5)

    def test_board_value_prefers_progressive_stock_closed_route(self):
        good = CascadeBoard.from_n_steps(1, "CCCCCC")
        good.slots[0].product = "CCCCCC"
        good.slots[0].main_reactant = "CC"
        good.slots[0].aux_reactants = ["CO"]
        good.slots[0].reaction_smiles = "CC.CO>>CCCCCC"
        good.slots[0].T = 30
        good.slots[0].pH = 7

        bad = CascadeBoard.from_n_steps(1, "CCCCCC")
        bad.slots[0].product = "CCCCCC"
        bad.slots[0].main_reactant = "CCCCCC"
        bad.slots[0].reaction_smiles = "CCCCCC>>CCCCCC"
        bad.slots[0].T = 30
        bad.slots[0].pH = 7

        scorer = RouteValueFunction()
        stock = lambda smi: smi in {"CC", "CO"}
        self.assertGreater(scorer.score_board(good, stock_checker=stock).score, scorer.score_board(bad, stock_checker=stock).score)

    def test_metric_value_features_match_export_contract(self):
        features = metric_value_features({
            "filled_route": True,
            "progressive_route": True,
            "route_solved": False,
            "strict_stock_solve": True,
            "retrosynthesis_progress": {
                "main_chain_reduction": 0.6,
                "largest_leaf_reduction": 0.7,
            },
            "route_naturalness": {"naturalness_score": 1.0},
            "condition": {"condition_window_success": True},
            "cascade_compatibility": {
                "cascade_compatibility_success": False,
                "issues": ["temperature_window_mismatch"],
            },
            "enzyme_evidence": {"enzyme_evidence_score": 0.5},
        })

        self.assertEqual(features["filled_route"], 1.0)
        self.assertEqual(features["strict_stock_solve"], 1.0)
        self.assertEqual(features["issue_count"], 1.0)
        self.assertEqual(features["main_chain_reduction"], 0.6)


if __name__ == "__main__":
    unittest.main()
