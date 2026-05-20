import unittest

from cascade_planner.cascadeboard.route_recovery import target_recovery_metrics


class RouteRecoveryMetricsTest(unittest.TestCase):
    def test_candidate_rank_separates_pool_hit_from_selected_route(self):
        entry = {
            "gt_route": [
                {"rxn_smiles": "CCO>>CC=O", "transformation": "oxidation"},
            ],
        }
        routes = [
            {
                "steps": [
                    {
                        "reaction_smiles": "CCC>>CC=O",
                        "reaction_type": "oxidation",
                        "main_reactant": "CCC",
                        "candidate_pool": {
                            "top_candidates": [
                                {"reaction_smiles": "CCC>>CC=O", "main_reactant": "CCC"},
                                {"reaction_smiles": "CCO>>CC=O", "main_reactant": "CCO"},
                            ],
                        },
                    },
                ],
            },
        ]

        metrics = target_recovery_metrics(routes, entry)

        self.assertFalse(metrics["exact_reaction_in_route_pool"])
        self.assertTrue(metrics["candidate_exact_reaction_in_pool"])
        self.assertEqual(metrics["candidate_exact_reaction_best_candidate_rank"], 2)
        self.assertFalse(metrics["gt_reactant_in_route_pool"])
        self.assertTrue(metrics["candidate_gt_reactant_in_pool"])
        self.assertEqual(metrics["candidate_gt_reactant_best_candidate_rank"], 2)
        self.assertEqual(metrics["best_reaction_edit_distance"], 1)
        self.assertEqual(metrics["best_type_edit_distance"], 0)
        self.assertEqual(metrics["recovery_bottleneck"], "selector_missed_exact_candidate")
        self.assertIn("selector_missed_gt_reactant_candidate", metrics["recovery_bottleneck_labels"])

    def test_selected_exact_reaction_counts_as_route_recovery(self):
        entry = {
            "gt_route": [
                {"rxn_smiles": "CCO>>CC=O", "transformation": "oxidation"},
            ],
        }
        routes = [
            {
                "steps": [
                    {
                        "reaction_smiles": "CCO>>CC=O",
                        "reaction_type": "oxidation",
                        "main_reactant": "CCO",
                        "candidate_pool": {
                            "top_candidates": [
                                {"reaction_smiles": "CCO>>CC=O", "main_reactant": "CCO"},
                            ],
                        },
                    },
                ],
            },
        ]

        metrics = target_recovery_metrics(routes, entry)

        self.assertTrue(metrics["exact_reaction_in_route_pool"])
        self.assertTrue(metrics["exact_route_reaction_match_any"])
        self.assertEqual(metrics["exact_reaction_first_rank"], 1)
        self.assertEqual(metrics["candidate_exact_reaction_best_candidate_rank"], 1)
        self.assertEqual(metrics["recovery_bottleneck"], "recovered_exact_route")

    def test_candidate_reactant_without_exact_reaction_is_reaction_detail_miss(self):
        entry = {
            "gt_route": [
                {"rxn_smiles": "CCO>>CC=O", "transformation": "oxidation"},
            ],
        }
        routes = [
            {
                "steps": [
                    {
                        "reaction_smiles": "CCC>>CC=O",
                        "reaction_type": "oxidation",
                        "main_reactant": "CCC",
                        "candidate_pool": {
                            "top_candidates": [
                                {"reaction_smiles": "CCO>>CC", "main_reactant": "CCO"},
                            ],
                        },
                    },
                ],
            },
        ]

        metrics = target_recovery_metrics(routes, entry)

        self.assertFalse(metrics["candidate_exact_reaction_in_pool"])
        self.assertTrue(metrics["candidate_gt_reactant_in_pool"])
        self.assertEqual(metrics["recovery_bottleneck"], "candidate_generator_reaction_detail_miss")

    def test_missing_gt_reactant_candidate_is_reactant_miss(self):
        entry = {
            "gt_route": [
                {"rxn_smiles": "CCO>>CC=O", "transformation": "oxidation"},
            ],
        }
        routes = [
            {
                "steps": [
                    {
                        "reaction_smiles": "CCC>>CC=O",
                        "reaction_type": "oxidation",
                        "main_reactant": "CCC",
                        "candidate_pool": {
                            "top_candidates": [
                                {"reaction_smiles": "CCC>>CC=O", "main_reactant": "CCC"},
                            ],
                        },
                    },
                ],
            },
        ]

        metrics = target_recovery_metrics(routes, entry)

        self.assertFalse(metrics["candidate_gt_reactant_in_pool"])
        self.assertEqual(metrics["recovery_bottleneck"], "candidate_generator_reactant_miss")


if __name__ == "__main__":
    unittest.main()
