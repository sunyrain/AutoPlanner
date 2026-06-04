import json
import tempfile
import unittest
from pathlib import Path

from scripts.filter_onmt_exact_recall_predictions import filter_exact_recall_predictions


class FilterONMTExactRecallPredictionsTest(unittest.TestCase):
    def test_filters_existing_exact_recall_predictions_and_parses_context_product(self):
        payload = {
            "results": [
                {
                    "model_path": "native.pt",
                    "topk": 5,
                    "rows": [
                        {
                            "idx": 0,
                            "product": "<target> C C O <product> C C O",
                            "target_reactants": "CC.O",
                            "predictions": ["C1", "CCO", "O.CC", "CC.O"],
                            "scores": [-0.1, -0.2, -0.3, -0.4],
                            "top1_exact": False,
                            "top5_exact": True,
                        }
                    ],
                },
                {
                    "model_path": "adapter.pt",
                    "topk": 5,
                    "rows": [
                        {
                            "idx": 0,
                            "product": "<target> C C O <product> C C O",
                            "target_reactants": "CC.O",
                            "predictions": ["C1", "CCO", "CCC"],
                            "scores": [-0.1, -0.2, -0.3],
                            "top1_exact": False,
                            "top5_exact": False,
                        }
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exact = root / "exact.json"
            output = root / "filtered.json"
            markdown = root / "filtered.md"
            exact.write_text(json.dumps(payload), encoding="utf-8")

            result = filter_exact_recall_predictions(
                exact_recall_json=exact,
                output_json=output,
                output_md=markdown,
            )
            markdown_exists = markdown.exists()

        native = result["results"][0]
        adapter = result["results"][1]
        self.assertEqual(native["rows"][0]["product_smiles"], "CCO")
        self.assertEqual(native["rows"][0]["filtered_predictions"], ["O.CC"])
        self.assertEqual(native["rows"][0]["filtered_scores"], [-0.3])
        self.assertEqual(native["filtered_topk_exact"], 1)
        self.assertEqual(adapter["filtered_topk_exact"], 0)
        self.assertEqual(result["decision"]["status"], "filtered_no_exact_lift")
        self.assertTrue(markdown_exists)


if __name__ == "__main__":
    unittest.main()
