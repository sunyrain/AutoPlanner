import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cascade_planner.cascadeboard.skeleton_reranker import (
    rerank_skeletons_with_model,
    skeleton_reranker_enabled,
    skeleton_reranker_metadata,
    skeleton_reranker_weight,
)
from cascade_planner.web.app import _skeleton_to_dict


class FakeSkeletonRanker:
    def score_skeleton(self, *, target_smiles, skeleton):
        return 0.9 if skeleton.types == ["oxidation"] else 0.1


class SkeletonRerankerRuntimeTest(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(skeleton_reranker_enabled())
            rows = [SimpleNamespace(types=["hydrolysis"], ec1s=[3], log_prob=0.0)]
            self.assertIs(rerank_skeletons_with_model(rows, target_smiles="CCO"), rows)

    def test_reranks_with_injected_ranker(self):
        low = SimpleNamespace(types=["hydrolysis"], ec1s=[3], log_prob=0.2)
        high = SimpleNamespace(types=["oxidation"], ec1s=[1], log_prob=0.0)

        out = rerank_skeletons_with_model(
            [low, high],
            target_smiles="CCO",
            weight=1.0,
            ranker=FakeSkeletonRanker(),
        )

        self.assertIs(out[0], high)
        self.assertAlmostEqual(high.skeleton_reranker_score, 0.9)

    def test_metadata_and_weight_are_safe(self):
        with patch.dict(os.environ, {"AUTOPLANNER_SKELETON_RERANKER_WEIGHT": "bad"}):
            self.assertEqual(skeleton_reranker_weight(default=0.4), 0.4)
        with patch.dict(os.environ, {"AUTOPLANNER_ENABLE_SKELETON_RERANKER": "1"}):
            self.assertTrue(skeleton_reranker_metadata()["enabled"])

    def test_web_serialization_includes_reranker_score(self):
        skel = SimpleNamespace(
            types=["oxidation"],
            ec1s=[1],
            Ts=[25.0],
            pHs=[7.0],
            compat_pred="",
            opmode_pred="",
            issues_pred=[],
            log_prob=1.0,
            skeleton_reranker_score=0.7,
        )

        row = _skeleton_to_dict(skel)

        self.assertEqual(row["skeleton_reranker_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
