import json
import tempfile
import unittest
from pathlib import Path

import joblib

from scripts.train_context_onmt_proposal_preference_scorer import score_candidate, train_preference_scorer


class TrainContextONMTProposalPreferenceScorerTest(unittest.TestCase):
    def test_trains_and_scores_hard_negative_preferences(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            valid = root / "valid.jsonl"
            test = root / "test.jsonl"
            out = root / "model"
            rows = [
                _pair("p1", "CCO", "CC.O", "CCO", "self"),
                _pair("p2", "CCN", "CC.N", "CCN", "self"),
                _pair("p3", "CCCl", "CC.Cl", "CCCl", "self"),
                _pair("p4", "CCCBr", "CCC.Br", "CCCBr", "self"),
            ]
            _write_jsonl(train, rows)
            _write_jsonl(valid, rows[:2])
            _write_jsonl(test, rows[2:])

            result = train_preference_scorer(
                train_jsonl=train,
                valid_jsonl=valid,
                test_jsonl=test,
                output_dir=out,
                n_bits=32,
                max_iter=200,
            )

            summary = result["summary"]
            self.assertEqual(summary["split_metrics"]["valid"]["n_pairs"], 2)
            self.assertGreaterEqual(summary["split_metrics"]["train"]["pairwise_accuracy"], 0.75)
            self.assertTrue((out / "context_onmt_proposal_preference_scorer.joblib").exists())
            self.assertTrue((out / "context_onmt_proposal_preference_scorer_report.md").exists())

            artifact = joblib.load(out / "context_onmt_proposal_preference_scorer.joblib")
            chosen_score = score_candidate(artifact, product="CCO", reactants="CC.O")
            rejected_score = score_candidate(artifact, product="CCO", reactants="CCO")
            self.assertGreater(chosen_score, rejected_score)
            self.assertIn("not expert labels", artifact["contract"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _pair(pair_id: str, product: str, chosen: str, rejected: str, neg_type: str) -> dict:
    return {
        "pair_id": pair_id,
        "source_example_id": pair_id,
        "split": "train",
        "negative_type": neg_type,
        "product": product,
        "target_smiles": product,
        "chosen_reactants": chosen,
        "rejected_reactants": rejected,
        "contract": "Pairwise proposal preference: chosen is clean top-level seed reactants; rejected is rule-generated hard negative, not an expert label.",
    }


if __name__ == "__main__":
    unittest.main()
