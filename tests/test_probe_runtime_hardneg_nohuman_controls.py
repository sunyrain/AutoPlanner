import unittest

import numpy as np

from cascade_planner.eval.probe_runtime_hardneg_nohuman_controls import _alpha_grid
from cascade_planner.eval.probe_runtime_hardneg_nohuman_controls import _compact_metrics
from cascade_planner.eval.probe_runtime_hardneg_nohuman_controls import _hgb_labels
from cascade_planner.eval.probe_runtime_hardneg_nohuman_controls import _material_feature_matrix


class RuntimeHardnegNoHumanProbeTest(unittest.TestCase):
    def test_material_features_are_numeric_and_capture_atom_delta(self):
        rows = [
            {
                "product_smiles": "CCO",
                "candidate_reactants": ["CC", "O"],
                "candidate_rank": 2,
                "candidate_score": 0.5,
                "runtime_nearest_any_transition_sim": 0.4,
                "runtime_nearest_pair_compatible_sim": 0.3,
                "runtime_inferred_transform_prior": 0.1,
                "runtime_pair_bucket_size": 3,
            }
        ]

        matrix = _material_feature_matrix(rows)

        self.assertEqual(matrix.shape, (1, 32))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix[0, 15], 0.0)
        self.assertEqual(matrix[0, 19], 1.0)

    def test_hgb_labels_support_block_and_exact_or_block(self):
        rows = [
            {"block_supported_positive_label": True, "exact_label": False},
            {"block_supported_positive_label": False, "exact_label": True},
            {"block_supported_positive_label": False, "exact_label": False},
        ]

        self.assertEqual(_hgb_labels(rows, "block_cls").tolist(), [1, 0, 0])
        self.assertEqual(_hgb_labels(rows, "exact_or_block_cls").tolist(), [1, 1, 0])

    def test_alpha_grid_includes_zero_and_extra_values(self):
        grid = _alpha_grid(extra=(3.0,))

        self.assertIn(0.0, grid)
        self.assertIn(3.0, grid)
        self.assertEqual(grid, sorted(grid))

    def test_compact_metrics_extracts_selection_score(self):
        report = {
            "block_supported_positive_label": {
                "mrr_covered": 0.5,
                "recall_at_k_all": {"5": 0.25},
            },
            "exact_label": {
                "mrr_covered": 0.1,
                "recall_at_k_all": {"5": 0.2},
            },
        }

        compact = _compact_metrics(report)

        self.assertAlmostEqual(compact["selection_score"], 0.8)
        self.assertEqual(compact["block_mrr"], 0.5)
        self.assertEqual(compact["exact_mrr"], 0.1)


if __name__ == "__main__":
    unittest.main()
