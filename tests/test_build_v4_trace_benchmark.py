import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_v4_trace_benchmark import build_v4_trace_benchmark


class BuildV4TraceBenchmarkTest(unittest.TestCase):
    def test_builds_single_target_non_overlap_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            v4 = root / "v4.jsonl"
            benchmark = root / "benchmark.json"
            output = root / "train.json"
            report = root / "report.json"
            benchmark.write_text(
                json.dumps([
                    {"doi": "10.overlap", "cascade_id": "cascade_1", "target_smiles": "CCO"}
                ]),
                encoding="utf-8",
            )
            rows = [
                {
                    "doi": "10.overlap",
                    "cascade_id": "cascade_1",
                    "trainable_recommended": True,
                    "target_product_smiles": "CCN",
                    "cascade_type": "all_chemical",
                    "steps": [{"rxn_smiles": "CC>>CCN"}],
                },
                {
                    "doi": "10.train",
                    "cascade_id": "cascade_2",
                    "trainable_recommended": True,
                    "target_product_smiles": "CCN",
                    "cascade_type": "chemoenzymatic",
                    "quality_tier": "gold",
                    "compatibility": {"compatibility_label": "sequential_preferred"},
                    "steps": [
                        {
                            "step_index": 1,
                            "rxn_smiles": "CC>>CCN",
                            "step_conditions": {"temperature_c": 30.0},
                            "catalyst_components": [{"ec_number": "1.1.1.1", "catalyst_class": "enzyme"}],
                        }
                    ],
                },
                {
                    "doi": "10.multi",
                    "cascade_id": "cascade_3",
                    "trainable_recommended": True,
                    "target_product_smiles": "CCN;CCC",
                    "cascade_type": "all_chemical",
                    "steps": [{"rxn_smiles": "CC>>CCN"}],
                },
            ]
            v4.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = build_v4_trace_benchmark(
                v4_jsonl=v4,
                benchmark_path=benchmark,
                output_path=output,
                report_path=report,
            )
            built = json.loads(output.read_text())

        self.assertEqual(result["counts"]["selected_rows"], 1)
        self.assertEqual(built[0]["target_smiles"], "CCN")
        self.assertEqual(built[0]["route_domain"], "chemoenzymatic")
        self.assertEqual(built[0]["gt_route"][0]["ec_number"], "1.1.1.1")
        self.assertEqual(result["counts"]["skipped"]["benchmark_key_overlap"], 1)
        self.assertEqual(result["counts"]["skipped"]["missing_or_multi_target"], 1)


if __name__ == "__main__":
    unittest.main()
