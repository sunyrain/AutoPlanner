import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_reactant_completion_predictions import diagnose_completion_predictions


class DiagnoseReactantCompletionPredictionsTest(unittest.TestCase):
    def test_groups_completion_failures_by_corruption_type(self):
        exact = {
            "results": [
                {
                    "model_path": "m.pt",
                    "limit": 3,
                    "rows": [
                        {
                            "idx": 0,
                            "target_reactants": "CC.O",
                            "predictions": ["CC.O", "CC"],
                            "top1_exact": True,
                            "top5_exact": True,
                        },
                        {
                            "idx": 1,
                            "target_reactants": "CC.N",
                            "predictions": ["CCN", "CCC"],
                            "top1_exact": False,
                            "top5_exact": False,
                        },
                        {
                            "idx": 2,
                            "target_reactants": "CC.Cl",
                            "predictions": ["C1", "CC"],
                            "top1_exact": False,
                            "top5_exact": False,
                        },
                    ],
                }
            ]
        }
        meta = [
            _meta("drop_one", "CC", "CC.O"),
            _meta("self", "CCN", "CC.N"),
            _meta("cross_swap", "CCBr", "CC.Cl"),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exact_path = root / "exact.json"
            meta_path = root / "meta.jsonl"
            output = root / "diag.json"
            markdown = root / "diag.md"
            exact_path.write_text(json.dumps(exact), encoding="utf-8")
            meta_path.write_text("\n".join(json.dumps(row) for row in meta) + "\n", encoding="utf-8")

            payload = diagnose_completion_predictions(
                exact_recall_json=exact_path,
                metadata_jsonl=meta_path,
                output_json=output,
                output_md=markdown,
                model_selector="first",
                topk=2,
            )
            self.assertTrue(markdown.exists())

        self.assertEqual(payload["summary"]["n_rows"], 3)
        self.assertEqual(payload["summary"]["topk_exact"], 1)
        self.assertEqual(payload["summary"]["rows_copying_given_side"], 2)
        self.assertEqual(payload["by_corruption_type"]["drop_one"]["topk_exact"], 1)
        self.assertEqual(payload["by_corruption_type"]["self"]["rows_copying_given_side"], 1)
        self.assertEqual(payload["decision"]["status"], "completion_exact_signal_present")


def _meta(corruption_type: str, given: str, chosen: str) -> dict:
    return {
        "corruption_type": corruption_type,
        "product": "CCO",
        "given_reactants": given,
        "chosen_reactants": chosen,
    }


if __name__ == "__main__":
    unittest.main()
