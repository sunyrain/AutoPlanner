import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.audit_skeleton_retrieval_prior import audit_skeleton_retrieval_prior


class AuditSkeletonRetrievalPriorTest(unittest.TestCase):
    def test_audit_reports_exact_target_exclusion(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            bench = root / "bench.json"
            prior = root / "prior.jsonl"
            bench.write_text(json.dumps([
                {
                    "target_smiles": "CCO",
                    "depth": 1,
                    "route_domain": "chemoenzymatic",
                    "gt_route": [{"transformation": "oxidation"}],
                }
            ]), encoding="utf-8")
            prior.write_text(json.dumps({
                "target_smiles": "CCO",
                "depth": 1,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation"],
            }) + "\n", encoding="utf-8")

            included = audit_skeleton_retrieval_prior(
                bench_path=bench,
                prior_path=prior,
                exclude_exact_target=False,
            )
            excluded = audit_skeleton_retrieval_prior(
                bench_path=bench,
                prior_path=prior,
                exclude_exact_target=True,
            )

        self.assertEqual(included["summary"]["exact_type_hit1"], 1)
        self.assertEqual(excluded["summary"]["prior_available"], 0)
        self.assertTrue(excluded["summary"]["exclude_exact_target"])


if __name__ == "__main__":
    unittest.main()
