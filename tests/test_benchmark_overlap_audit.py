import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.benchmark_overlap_audit import audit_benchmark_overlap


class BenchmarkOverlapAuditTest(unittest.TestCase):
    def test_detects_target_and_reaction_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps([{
                "target_smiles": "CCO",
                "route_domain": "all_chemical",
                "depth": 1,
                "doi": "10.1/test",
                "gt_route": [{"rxn_smiles": "CC>>CCO", "transformation": "oxidation"}],
            }]), encoding="utf-8")
            pack = root / "pack"
            pack.mkdir()
            (pack / "step_pairs.jsonl").write_text(
                json.dumps({"target_smiles": "CCO", "reaction_smiles": "CC>>CCO"}) + "\n",
                encoding="utf-8",
            )

            report = audit_benchmark_overlap(benchmark, [pack])

        self.assertFalse(report["summary"]["blind_safe"])
        self.assertEqual(report["summary"]["target_overlap_count"], 1)
        self.assertEqual(report["summary"]["gt_reaction_overlap_count"], 1)

    def test_clean_pack_is_blind_safe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps([{
                "target_smiles": "CCO",
                "route_domain": "all_chemical",
                "depth": 1,
                "gt_route": [{"rxn_smiles": "CC>>CCO", "transformation": "oxidation"}],
            }]), encoding="utf-8")
            pack = root / "pack"
            pack.mkdir()
            (pack / "candidate_pools.jsonl").write_text(
                json.dumps({
                    "target_smiles": "CCC",
                    "candidates": [{"candidate": {"rxn_smiles": "C>>CCC"}}],
                }) + "\n",
                encoding="utf-8",
            )

            report = audit_benchmark_overlap(benchmark, [pack])

        self.assertTrue(report["summary"]["blind_safe"])
        self.assertEqual(report["summary"]["target_overlap_count"], 0)
        self.assertEqual(report["summary"]["gt_reaction_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
