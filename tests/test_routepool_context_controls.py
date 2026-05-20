import unittest

from cascade_planner.eval.audit_routepool_context_controls import (
    _shuffle_features_within_target,
    _shuffle_labels_within_target,
)


class RoutePoolContextControlsTest(unittest.TestCase):
    def test_feature_shuffle_preserves_labels_and_native_features(self):
        rows = [
            _row("t1", "r1", 1, native_rank=0, ccts=0.1, block=0.2),
            _row("t1", "r2", 0, native_rank=1, ccts=0.9, block=0.8),
            _row("t2", "r3", 2, native_rank=0, ccts=0.3, block=0.4),
        ]

        shuffled = _shuffle_features_within_target(
            rows,
            ["ccts_model_mean", "block_rerank_score"],
            seed=1,
        )

        self.assertEqual([row["route_label"] for row in shuffled], [1, 0, 2])
        self.assertEqual([row["feature"]["native_rank"] for row in shuffled], [0.0, 1.0, 0.0])
        self.assertEqual(
            sorted(row["feature"]["ccts_model_mean"] for row in shuffled[:2]),
            [0.1, 0.9],
        )
        self.assertEqual(
            sorted(row["feature"]["block_rerank_score"] for row in shuffled[:2]),
            [0.2, 0.8],
        )
        self.assertEqual(shuffled[2]["feature"]["ccts_model_mean"], 0.3)
        self.assertNotEqual(id(rows[0]), id(shuffled[0]))

    def test_label_shuffle_preserves_features_and_label_multiset_per_target(self):
        rows = [
            _row("t1", "r1", 1, native_rank=0, ccts=0.1, block=0.2),
            _row("t1", "r2", 0, native_rank=1, ccts=0.9, block=0.8),
            _row("t1", "r3", 3, native_rank=2, ccts=0.5, block=0.6),
            _row("t2", "r4", 2, native_rank=0, ccts=0.3, block=0.4),
        ]

        shuffled = _shuffle_labels_within_target(rows, seed=4)

        self.assertEqual(
            [row["feature"]["ccts_model_mean"] for row in shuffled],
            [0.1, 0.9, 0.5, 0.3],
        )
        self.assertEqual(
            sorted(row["route_label"] for row in shuffled[:3]),
            [0, 1, 3],
        )
        self.assertEqual(shuffled[3]["route_label"], 2)
        self.assertNotEqual(id(rows[0]), id(shuffled[0]))


def _row(target_id, route_id, label, *, native_rank, ccts, block):
    return {
        "target_id": target_id,
        "route_id": route_id,
        "route_label": label,
        "native_rank": native_rank,
        "feature": {
            "native_rank": float(native_rank),
            "native_inv_rank": 1.0 / float(native_rank + 1),
            "ccts_model_mean": float(ccts),
            "block_rerank_score": float(block),
        },
    }


if __name__ == "__main__":
    unittest.main()
