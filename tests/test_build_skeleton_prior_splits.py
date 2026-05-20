import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_skeleton_prior_splits import (
    build_skeleton_prior_splits,
    load_skeleton_rows,
    scaffold_key,
)


class BuildSkeletonPriorSplitsTest(unittest.TestCase):
    def test_scaffold_split_writes_non_overlapping_outputs(self):
        rows = [
            {
                "target_smiles": "c1ccccc1O",
                "depth": 1,
                "route_domain": "all_chemical",
                "type_sequence": ["hydrolysis"],
                "label": 1.0,
                "source": "benchmark_gt",
            },
            {
                "target_smiles": "c1ccccc1N",
                "depth": 1,
                "route_domain": "all_chemical",
                "type_sequence": ["amination"],
                "label": 1.0,
                "source": "benchmark_gt",
            },
            {
                "target_smiles": "C1CCCCC1",
                "depth": 1,
                "route_domain": "all_chemical",
                "type_sequence": ["reduction"],
                "label": 1.0,
                "source": "benchmark_gt",
            },
            {
                "target_smiles": "CCO",
                "depth": 1,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation"],
                "label": 1.0,
                "source": "benchmark_gt",
            },
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            prior = root / "skeleton_prior.jsonl"
            prior.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "split"

            manifest = build_skeleton_prior_splits(
                prior_path=prior,
                output_dir=output,
                val_fraction=0.25,
                test_fraction=0.25,
                seed=7,
                min_similarity=0.0,
            )

            loaded = {
                split: load_skeleton_rows(Path(path))
                for split, path in manifest["files"].items()
                if split in {"train", "val", "test"}
            }

            self.assertTrue(Path(manifest["files"]["manifest"]).exists())
            self.assertTrue(Path(manifest["files"]["report"]).exists())
            self.assertEqual(manifest["leakage"]["target_overlap"]["train_val"], 0)
            self.assertEqual(manifest["leakage"]["target_overlap"]["train_test"], 0)
            self.assertEqual(manifest["leakage"]["scaffold_overlap"]["train_test"], 0)
            self.assertGreater(sum(len(data_rows) for data_rows in loaded.values()), 0)
            self.assertIn("scaffold:", scaffold_key("c1ccccc1O"))


if __name__ == "__main__":
    unittest.main()
