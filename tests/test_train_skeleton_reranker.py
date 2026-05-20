import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_skeleton_reranker import (
    augment_synthetic_negatives,
    build_dataset,
    build_feature_schema,
    load_split_rows,
    pairwise_group_indices,
    train_skeleton_reranker,
)


class TrainSkeletonRerankerTest(unittest.TestCase):
    def test_train_small_split(self):
        split_rows = {
            "train": [
                {
                    "target_smiles": "CCO",
                    "depth": 1,
                    "route_domain": "chemoenzymatic",
                    "operation_mode": "sequential_isolated",
                    "type_sequence": ["oxidation"],
                    "ec1_sequence": ["1"],
                    "label": 1.0,
                    "source": "benchmark_gt",
                    "doi": "10.test/train",
                },
                {
                    "target_smiles": "CCO",
                    "depth": 1,
                    "route_domain": "chemoenzymatic",
                    "operation_mode": "sequential_isolated",
                    "type_sequence": ["hydrolysis"],
                    "ec1_sequence": ["3"],
                    "label": 0.25,
                    "source": "planner_route",
                },
                {
                    "target_smiles": "CCN",
                    "depth": 1,
                    "route_domain": "all_chemical",
                    "operation_mode": "unknown",
                    "type_sequence": ["amination"],
                    "ec1_sequence": [""],
                    "label": 1.0,
                    "source": "benchmark_gt",
                },
                {
                    "target_smiles": "CCN",
                    "depth": 1,
                    "route_domain": "all_chemical",
                    "operation_mode": "unknown",
                    "type_sequence": ["oxidation"],
                    "ec1_sequence": [""],
                    "label": 0.25,
                    "source": "planner_route",
                },
            ],
            "val": [
                {
                    "target_smiles": "CCC",
                    "depth": 1,
                    "route_domain": "all_chemical",
                    "operation_mode": "unknown",
                    "type_sequence": ["amination"],
                    "ec1_sequence": [""],
                    "label": 1.0,
                    "source": "benchmark_gt",
                },
                {
                    "target_smiles": "CCC",
                    "depth": 1,
                    "route_domain": "all_chemical",
                    "operation_mode": "unknown",
                    "type_sequence": ["hydrolysis"],
                    "ec1_sequence": ["3"],
                    "label": 0.25,
                    "source": "planner_route",
                },
            ],
            "test": [
                {
                    "target_smiles": "CCCl",
                    "depth": 1,
                    "route_domain": "all_chemical",
                    "operation_mode": "unknown",
                    "type_sequence": ["amination"],
                    "ec1_sequence": [""],
                    "label": 1.0,
                    "source": "benchmark_gt",
                },
                {
                    "target_smiles": "CCCl",
                    "depth": 1,
                    "route_domain": "all_chemical",
                    "operation_mode": "unknown",
                    "type_sequence": ["hydrolysis"],
                    "ec1_sequence": ["3"],
                    "label": 0.25,
                    "source": "planner_route",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            split_dir = root / "split"
            split_dir.mkdir()
            for split, rows in split_rows.items():
                (split_dir / f"skeleton_prior_{split}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )
            loaded = load_split_rows(split_dir)
            schema = build_feature_schema(loaded["train"], n_bits=16, min_vocab_count=1, positive_threshold=1.0)
            dataset = build_dataset(loaded["train"], schema)
            report = train_skeleton_reranker(
                split_dir=split_dir,
                model_output=root / "skeleton_reranker.pt",
                report_output=root / "skeleton_reranker.json",
                md_output=root / "skeleton_reranker.md",
                epochs=1,
                batch_size=2,
                n_bits=16,
                hidden=16,
                min_vocab_count=1,
                synthetic_negatives_per_positive=1,
                loss_mode="pairwise",
            )

            self.assertTrue((root / "skeleton_reranker.pt").exists())
            self.assertTrue((root / "skeleton_reranker.json").exists())
            self.assertTrue((root / "skeleton_reranker.md").exists())

        self.assertEqual(dataset.x.shape[0], 4)
        self.assertIn("val", report["metrics"])
        self.assertIn("ranking", report["metrics"]["test"])
        self.assertEqual(report["metadata"]["loss_mode"], "pairwise")
        self.assertGreater(report["metadata"]["n_train_after_synthetic"], report["metadata"]["n_rows"]["train"])

    def test_pairwise_group_indices_find_positive_negative_groups(self):
        rows = [
            {"target_smiles": "CCO", "depth": 1, "type_sequence": ["oxidation"]},
            {"target_smiles": "CCO", "depth": 1, "type_sequence": ["hydrolysis"]},
            {"target_smiles": "CCC", "depth": 1, "type_sequence": ["amination"]},
        ]
        groups = pairwise_group_indices(rows, labels=[1.0, 0.0, 1.0])

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], ([0], [1]))

    def test_synthetic_negatives_corrupt_positive_skeletons(self):
        rows = [
            {
                "target_smiles": "CCO",
                "depth": 2,
                "type_sequence": ["oxidation", "reduction"],
                "ec1_sequence": ["1", "2"],
                "label": 1.0,
            },
            {
                "target_smiles": "CCO",
                "depth": 2,
                "type_sequence": ["hydrolysis", "reduction"],
                "ec1_sequence": ["3", "2"],
                "label": 0.25,
            },
        ]

        out = augment_synthetic_negatives(
            rows,
            n_per_positive=2,
            positive_threshold=1.0,
            seed=1,
        )

        synthetic = [row for row in out if row.get("source") == "synthetic_contrastive"]
        self.assertGreaterEqual(len(synthetic), 1)
        self.assertTrue(all(row["label"] == 0.0 for row in synthetic))
        self.assertTrue(any(row["type_sequence"] != rows[0]["type_sequence"] for row in synthetic))

    def test_load_split_rows_accepts_hard_negative_subdirectories(self):
        row = {
            "target_smiles": "CCO",
            "depth": 1,
            "type_sequence": ["oxidation"],
            "ec1_sequence": ["1"],
            "label": 1.0,
        }
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            for split in ("train", "val", "test"):
                split_dir = root / split
                split_dir.mkdir()
                (split_dir / "skeleton_pairwise_training.jsonl").write_text(
                    json.dumps(row) + "\n",
                    encoding="utf-8",
                )

            loaded = load_split_rows(root)

        self.assertEqual(len(loaded["train"]), 1)
        self.assertEqual(loaded["test"][0]["type_sequence"], ["oxidation"])


if __name__ == "__main__":
    unittest.main()
