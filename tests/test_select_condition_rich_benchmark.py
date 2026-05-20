import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.select_condition_rich_benchmark import select_condition_rich_targets


class SelectConditionRichBenchmarkTest(unittest.TestCase):
    def test_selects_targets_with_condition_rich_candidates(self):
        benchmark = [
            {"target_smiles": "CCO", "route_domain": "chemoenzymatic", "depth": 2},
            {"target_smiles": "CCC", "route_domain": "all_chemical", "depth": 1},
        ]
        rows = [
            {
                "target_smiles": "CCO",
                "candidate": {
                    "source": "v3_retrieval",
                    "T": 30.0,
                    "pH": 7.0,
                    "doi": "10.test/example",
                    "uniprot_accession": "P12345",
                },
            },
            {"target_smiles": "CCC", "candidate": {"source": "retrochimera"}},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bench = root / "bench.json"
            pack = root / "pack"
            out = root / "selected.json"
            report = root / "selected.md"
            pack.mkdir()
            bench.write_text(json.dumps(benchmark), encoding="utf-8")
            (pack / "candidate_ranking.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            selected = select_condition_rich_targets(
                benchmark_path=bench,
                pack_dir=pack,
                output_path=out,
                report_path=report,
                limit=1,
            )

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["target_smiles"], "CCO")
            self.assertTrue(out.exists())
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
