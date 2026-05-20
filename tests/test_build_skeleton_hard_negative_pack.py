import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_skeleton_hard_negative_pack import (
    build_skeleton_hard_negative_pack,
    edit_distance,
    sequence_match_fraction,
)


class BuildSkeletonHardNegativePackTest(unittest.TestCase):
    def test_builds_real_hard_negative_pack(self):
        rows = [
            {
                "skeleton_id": "gt",
                "source": "benchmark_gt",
                "target_smiles": "CCO",
                "depth": 2,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation", "reduction"],
                "ec1_sequence": ["1", "2"],
                "label": 1.0,
            },
            {
                "skeleton_id": "planner",
                "source": "planner_route",
                "target_smiles": "CCO",
                "depth": 2,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation", "hydrolysis"],
                "ec1_sequence": ["1", "3"],
                "label": 0.25,
                "label_type": "filled_only",
                "metrics_summary": {
                    "filled_route": True,
                    "progressive_route": False,
                    "route_solved": False,
                    "strict_stock_solve": False,
                    "compatibility_success": False,
                    "issues": ["condition_window"],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            pack = root / "pack"
            pack.mkdir()
            (pack / "skeleton_prior.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            out = root / "hard"

            manifest = build_skeleton_hard_negative_pack(
                pack_dir=pack,
                output_dir=out,
                min_type_match=0.25,
            )
            hard_rows = [
                json.loads(line)
                for line in Path(manifest["files"]["hard_negatives"]).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(manifest["counts"]["hard_negative_rows"], 1)
        self.assertEqual(hard_rows[0]["type_edit_distance"], 1)
        self.assertIn("stock_open", hard_rows[0]["failure_reasons"])
        self.assertIn("compatibility_failure", hard_rows[0]["failure_reasons"])

    def test_sequence_metrics(self):
        self.assertEqual(edit_distance(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(sequence_match_fraction(["a", "b"], ["a", "c"]), 0.5)


if __name__ == "__main__":
    unittest.main()
