import unittest
import tempfile
from pathlib import Path

from scripts.compare_chem_enzy_onmt_tokenizer_ab import render_markdown, run_ab
from scripts.compare_chem_enzy_onmt_tokenizer_ab import _comparisons


class CompareChemEnzyONMTTokenizerABTest(unittest.TestCase):
    def test_dry_run_writes_char_and_token_requests(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ab"
            result = run_ab(
                targets=["CCO"],
                output_dir=out,
                vendor_root=Path("vendor/ChemEnzyRetroPlanner"),
                gpu=-1,
                preset="quick",
                max_steps=4,
                iterations=2,
                expansion_topk=5,
                stock_mode="commercial",
                timeout_s=10,
                execute=False,
            )
            markdown = render_markdown(result)

            self.assertEqual(result["summary"]["n_targets"], 1)
            self.assertEqual(result["summary"]["n_runs"], 2)
            self.assertEqual([row["tokenizer"] for row in result["runs"]], ["char", "token"])
            self.assertTrue((out / "target00_char_request.json").exists())
            self.assertTrue((out / "target00_token_request.json").exists())
            self.assertIn("ChemEnzy ONMT Tokenizer A/B", markdown)
            self.assertIn("planned", markdown)

    def test_comparisons_include_route_quality_counts(self):
        comparisons = _comparisons([
            {
                "target_index": 0,
                "target_smiles": "CCO",
                "tokenizer": "char",
                "status": "solved",
                "n_routes": 2,
                "best_steps": 1,
                "multistep_routes": 1,
                "ge3_step_routes": 0,
                "rule_feasible_routes": 2,
                "avg_steps": 1.5,
                "solved": True,
            },
            {
                "target_index": 0,
                "target_smiles": "CCO",
                "tokenizer": "token",
                "status": "solved",
                "n_routes": 3,
                "best_steps": 2,
                "multistep_routes": 2,
                "ge3_step_routes": 1,
                "rule_feasible_routes": 3,
                "avg_steps": 2.3,
                "solved": True,
            },
        ])

        self.assertEqual(comparisons[0]["token_multistep_routes"], 2)
        self.assertEqual(comparisons[0]["token_ge3_step_routes"], 1)
        self.assertEqual(comparisons[0]["char_rule_feasible_routes"], 2)


if __name__ == "__main__":
    unittest.main()
