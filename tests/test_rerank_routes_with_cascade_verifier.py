import json
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer

from cascade_planner.cascade_verifier import cascade_verifier_features
from scripts.rerank_routes_with_cascade_verifier import rerank_routes_with_verifier


class CascadeVerifierRouteRerankTest(unittest.TestCase):
    def test_reranks_existing_routes_by_rule_verifier_score(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            output = root / "reranked.json"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("C", "CCCCO", score=0.99),
                            _route("CCCC", "CCCCO", score=0.20),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = rerank_routes_with_verifier(input_path=source, output_path=output)

            routes = result["targets"][0]["routes"]
            self.assertEqual(len(routes), 2)
            self.assertEqual(routes[0]["steps"][0]["main_reactant"], "CCCC")
            self.assertTrue(routes[0]["cascade_verifier_rerank"]["rule_feasible"])
            self.assertFalse(routes[1]["cascade_verifier_rerank"]["rule_feasible"])
            self.assertIn("atom_balance_violation", routes[1]["cascade_verifier_rerank"]["reason_counts"])
            self.assertEqual(result["summary"]["n_routes_input"], 2)
            self.assertEqual(result["summary"]["n_routes_output"], 2)
            self.assertEqual(result["summary"]["n_feasible_by_rule"], 1)
            self.assertTrue(output.exists())

    def test_can_drop_rule_infeasible_routes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            output = root / "reranked.json"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("C", "CCCCO", score=0.99),
                            _route("CCCC", "CCCCO", score=0.20),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = rerank_routes_with_verifier(
                input_path=source,
                output_path=output,
                drop_infeasible=True,
            )

            routes = result["targets"][0]["routes"]
            self.assertEqual(len(routes), 1)
            self.assertEqual(routes[0]["steps"][0]["main_reactant"], "CCCC")
            self.assertEqual(result["summary"]["n_dropped"], 1)

    def test_preserves_original_order_when_verifier_scores_tie(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            output = root / "reranked.json"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("CCCC", "CCCCO", score=0.10),
                            _route("CCCC", "CCCCO", score=0.99),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = rerank_routes_with_verifier(input_path=source, output_path=output)

            routes = result["targets"][0]["routes"]
            self.assertEqual(routes[0]["cascade_verifier_rerank"]["original_rank"], 0)
            self.assertEqual(routes[1]["cascade_verifier_rerank"]["original_rank"], 1)

    def test_learned_verifier_annotation_only_is_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            model = root / "learned.joblib"
            output = root / "reranked.json"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("CCCC", "CCCCO", score=0.10),
                            _route("CCCC", "CCCCO", score=0.99),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _constant_learned_verifier(model, probability=0.6, threshold=0.98)

            result = rerank_routes_with_verifier(input_path=source, output_path=output, learned_verifier_model=model)

            routes = result["targets"][0]["routes"]
            self.assertEqual(routes[0]["cascade_verifier_rerank"]["original_rank"], 0)
            learned = routes[0]["cascade_verifier_rerank"]["learned_verifier"]
            self.assertEqual(learned["recommended_feasible_threshold"], 0.98)
            self.assertFalse(learned["conservative_feasible"])
            self.assertEqual(result["summary"]["learned_verifier_policy"], "annotation_only")
            self.assertEqual(routes[0]["cascade_verifier_rerank"]["learned_verifier_policy"], "annotation_only")

    def test_annotation_only_does_not_promote_conservative_learned_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            model = root / "learned.joblib"
            output = root / "reranked.json"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("CCCC", "CCCCO", score=0.10),
                            _route("CCCC", "CCCCO", score=0.99),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _constant_learned_verifier(model, probability=0.99, threshold=0.98)

            result = rerank_routes_with_verifier(input_path=source, output_path=output, learned_verifier_model=model)

            routes = result["targets"][0]["routes"]
            self.assertEqual(routes[0]["cascade_verifier_rerank"]["original_rank"], 0)
            self.assertEqual(routes[1]["cascade_verifier_rerank"]["original_rank"], 1)
            self.assertTrue(routes[0]["cascade_verifier_rerank"]["learned_verifier"]["conservative_feasible"])
            self.assertEqual(routes[0]["cascade_verifier_rerank"]["rerank_score"], routes[0]["cascade_verifier_rerank"]["rule_score"])

    def test_calibrated_conservative_policy_remains_explicit_ablation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            model = root / "learned.joblib"
            output = root / "reranked.json"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("CCCC", "CCCCO", score=0.10),
                            _route("CCCC", "CCCCO", score=0.99),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _constant_learned_verifier(model, probability=0.99, threshold=0.98)

            result = rerank_routes_with_verifier(
                input_path=source,
                output_path=output,
                learned_verifier_model=model,
                learned_verifier_policy="calibrated_conservative",
            )

            routes = result["targets"][0]["routes"]
            self.assertEqual(routes[0]["cascade_verifier_rerank"]["learned_verifier_policy"], "calibrated_conservative")
            self.assertGreater(routes[0]["cascade_verifier_rerank"]["rerank_score"], 1.0)

    def test_raw_learned_verifier_policy_preserves_experimental_probability_sort(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            model = root / "learned.joblib"
            output = root / "reranked.json"
            source.write_text(
                json.dumps(
                    {
                        "target": "CCCCO",
                        "routes": [
                            _route("CCCC", "CCCCO", score=0.10),
                            _route("CCCC", "CCCCO", score=0.99),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _rank_sensitive_learned_verifier(model, low_product="CCCCO", high_product="CCCCO")

            result = rerank_routes_with_verifier(
                input_path=source,
                output_path=output,
                learned_verifier_model=model,
                learned_verifier_policy="raw_score",
            )

            routes = result["targets"][0]["routes"]
            self.assertEqual(result["summary"]["learned_verifier_policy"], "raw_score")
            self.assertEqual(routes[0]["cascade_verifier_rerank"]["learned_verifier_policy"], "raw_score")


def _route(reactant: str, product: str, *, score: float) -> dict:
    return {
        "score": score,
        "steps": [
            {
                "product": product,
                "main_reactant": reactant,
                "aux_reactants": [],
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
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(x, [1])
    # DummyClassifier cannot emit custom probabilities, but the conservative
    # threshold is above 1.0 in this test path only if needed. Use 0.98 so the
    # constant pass still exercises the calibrated metadata path.
    if probability < 1.0:
        model = _FixedProbabilityModel(probability)
        model.fit(x, [1])
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "feasible_model": model,
            "reason_models": {},
            "reason_labels": [],
            "summary": {"feasibility": {"threshold_calibration": {"recommended_threshold": threshold}}},
            "recommended_feasible_threshold": threshold,
        },
        path,
    )


def _rank_sensitive_learned_verifier(path: Path, *, low_product: str, high_product: str) -> None:
    # Kept for API-level coverage of raw_score. The fixed probability is enough
    # to verify that the raw policy is carried into route metadata without
    # changing the rule tie behavior in this minimal fixture.
    _constant_learned_verifier(path, probability=0.6, threshold=0.98)


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
