import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_onmt_prediction_errors import diagnose_prediction_errors, render_markdown


class DiagnoseONMTPredictionErrorsTest(unittest.TestCase):
    def test_labels_no_overlap_and_partial_overlap(self):
        payload = {
            "results": [
                {
                    "model_path": "native.pt",
                    "rows": [],
                },
                {
                    "model_path": "adapter.pt",
                    "rows": [
                        {
                            "idx": 0,
                            "product": "CCO",
                            "target_reactants": "CC.O",
                            "predictions": ["CCC.N", "CC.N"],
                            "scores": [-1.0, -2.0],
                            "top1_exact": False,
                            "top5_exact": False,
                        },
                        {
                            "idx": 1,
                            "product": "CCN",
                            "target_reactants": "CC.N",
                            "predictions": ["c1ccccc1", "CCC"],
                            "scores": [-1.0, -2.0],
                            "top1_exact": False,
                            "top5_exact": False,
                        },
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exact = root / "exact.json"
            output = root / "diag.json"
            markdown = root / "diag.md"
            exact.write_text(json.dumps(payload), encoding="utf-8")

            result = diagnose_prediction_errors(
                exact_recall_json=exact,
                output_json=output,
                output_md=markdown,
            )
            rendered = render_markdown(result)
            output_exists = output.exists()
            markdown_exists = markdown.exists()

        self.assertEqual(result["summary"]["n_rows"], 2)
        self.assertEqual(result["summary"]["rows_with_any_gt_reactant_overlap"], 1)
        self.assertIn("partial_gt_reactant_overlap", result["rows"][0]["labels"])
        self.assertIn("no_gt_reactant_overlap", result["rows"][1]["labels"])
        self.assertEqual(result["decision"]["status"], "reactant_set_generation_failure")
        self.assertTrue(output_exists)
        self.assertTrue(markdown_exists)
        self.assertIn("ONMT Prediction Error Diagnostic", rendered)


if __name__ == "__main__":
    unittest.main()
