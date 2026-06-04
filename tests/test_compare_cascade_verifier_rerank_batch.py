import json
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer

from cascade_planner.cascade_verifier import cascade_verifier_features
from scripts.compare_cascade_verifier_rerank_batch import compare_batch


class CascadeVerifierRerankBatchComparisonTest(unittest.TestCase):
    def test_compares_original_and_rule_verifier_top1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "target_top3_routes.json"
            output = root / "summary.json"
            out_dir = root / "reranked"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("bad", "C", "CCCCO", risk_order=30, score=0.99),
                            _route("good", "CCCC", "CCCCO", risk_order=10, score=0.20),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = compare_batch(inputs=[source], output=output, output_dir=out_dir)

            summary = result["summary"]
            row = result["targets"][0]
            self.assertEqual(summary["n_targets"], 1)
            self.assertEqual(summary["n_routes"], 2)
            self.assertEqual(summary["rule_feasible_routes"], 1)
            self.assertEqual(summary["rule_top1_changed"], 1)
            self.assertEqual(summary["rule_top1_audit_improved"], 1)
            self.assertEqual(summary["promotion_decision"]["rule_verifier"], "promote_as_conservative_gate")
            self.assertIsNone(summary["learned_top1_differs_from_rule"])
            self.assertEqual(summary["audit_bucket_counts"]["triage_fragment:risk10"], 1)
            self.assertEqual(summary["audit_bucket_feasible_counts"]["triage_fragment:risk10"], 1)
            self.assertEqual(row["original_top1_id"], "bad")
            self.assertEqual(row["rule_top1_id"], "good")
            self.assertGreater(row["rule_top1_audit_delta"], 0)
            self.assertTrue(output.exists())
            self.assertTrue((out_dir / "target_top3_routes.rule_verifier_rerank.json").exists())

    def test_learned_annotation_only_does_not_create_extra_top1_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "target_top2_routes.json"
            output = root / "summary.json"
            out_dir = root / "reranked"
            model = root / "learned.joblib"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("first", "CCCC", "CCCCO", risk_order=10, score=0.10),
                            _route("second", "CCCC", "CCCCO", risk_order=10, score=0.99),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _constant_learned_verifier(model, probability=0.99, threshold=0.98)

            result = compare_batch(
                inputs=[source],
                output=output,
                output_dir=out_dir,
                learned_verifier_model=model,
            )

            summary = result["summary"]
            row = result["targets"][0]
            self.assertEqual(summary["learned_verifier_policy"], "annotation_only")
            self.assertFalse(row["learned_top1_changed"])
            self.assertFalse(row["learned_top1_differs_from_rule"])
            self.assertEqual(summary["promotion_decision"]["learned_verifier"], "annotation_only_not_ranked")


def _route(route_id: str, reactant: str, product: str, *, risk_order: int, score: float) -> dict:
    return {
        "id": route_id,
        "score": score,
        "product_audit": {"risk_order": risk_order, "route_class": "triage_fragment"},
        "steps": [
            {
                "product": product,
                "main_reactant": reactant,
                "reaction_smiles": f"{reactant}>>{product}",
                "T": 30,
                "pH": 7,
                "solvent": "water",
            }
        ],
    }


def _constant_learned_verifier(path: Path, *, probability: float, threshold: float) -> None:
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([cascade_verifier_features({"target_smiles": "CCCCO", "cascade": {"steps": []}})])
    model = _FixedProbabilityModel(probability)
    model.fit(x, [1])
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "feasible_model": model,
            "reason_models": {},
            "reason_labels": [],
            "recommended_feasible_threshold": threshold,
        },
        path,
    )


class _FixedProbabilityModel:
    classes_ = [0, 1]

    def __init__(self, probability: float):
        self.probability = float(probability)

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        return [[1.0 - self.probability, self.probability] for _ in range(x.shape[0])]


if __name__ == "__main__":
    unittest.main()
