import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.audit_condition_data_coverage import (
    audit_paths,
    audit_training_pack,
)


class AuditConditionDataCoverageTest(unittest.TestCase):
    def test_audit_route_artifact_counts_step_and_candidate_fields(self):
        artifact = {
            "routes": [{
                "steps": [{
                    "source": "enzyformer",
                    "T": 30.0,
                    "pH": 7.0,
                    "solvent": "water",
                    "candidate_pool": {
                        "top_candidates": [
                            {"source": "enzyformer", "T": 30.0, "pH": 7.0, "evidence": {"doi": "10.1/test"}},
                            {"source": "retrochimera", "T": None, "pH": None, "solvent": ""},
                        ]
                    },
                }]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            report = audit_paths([path])

        self.assertEqual(report["totals"]["step_total"], 1)
        self.assertEqual(report["totals"]["candidate_total"], 2)
        self.assertEqual(report["totals"]["candidate_has_T_and_pH"], 1)
        self.assertEqual(report["by_source"]["enzyformer"]["totals"]["candidate_has_doi"], 1)

    def test_audit_training_pack_candidate_rows(self):
        rows = [
            {"candidate": {"source": "enzyformer", "T": 25.0, "pH": 7.5, "solvent": "buffer"}},
            {"candidate": {"source": "retrochimera", "T": None, "pH": None}},
        ]
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "candidate_ranking.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            report = audit_training_pack(pack)

        self.assertEqual(report["training_pack_rows"], 2)
        self.assertEqual(report["totals"]["candidate_has_T_and_pH"], 1)


if __name__ == "__main__":
    unittest.main()
