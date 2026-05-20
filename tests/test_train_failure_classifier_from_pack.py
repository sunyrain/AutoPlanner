import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_failure_classifier_from_pack import (
    build_dataset,
    load_failure_rows,
    split_by_target,
    split_summary,
    train_failure_classifier_from_pack,
)


class TrainFailureClassifierFromPackTest(unittest.TestCase):
    def test_train_small_failure_pack(self):
        rows = [
            {
                "target_smiles": "CCO",
                "route_domain": "all_chemical",
                "depth": 3,
                "n_routes": 1,
                "labels": ["generator_exact_miss", "stock_dead_end"],
                "has_failure_label": True,
                "metrics": {"strict_stock_solve_any": False, "plan": True},
            },
            {
                "target_smiles": "CCN",
                "route_domain": "chemoenzymatic",
                "depth": 2,
                "n_routes": 2,
                "labels": ["condition_failure"],
                "has_failure_label": True,
                "metrics": {"condition_window_success_any": False, "plan": True},
            },
            {
                "target_smiles": "CCC",
                "route_domain": "all_chemical",
                "depth": 2,
                "n_routes": 2,
                "labels": [],
                "has_failure_label": False,
                "metrics": {"strict_stock_solve_any": True, "plan": True},
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "failure_diagnosis.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            loaded = load_failure_rows(pack)
            dataset = build_dataset(loaded, n_bits=16, min_label_count=1)
            report = train_failure_classifier_from_pack(
                pack_dir=pack,
                model_output=Path(td) / "failure.pt",
                report_output=Path(td) / "failure.json",
                md_output=Path(td) / "failure.md",
                epochs=1,
                batch_size=2,
                n_bits=16,
                hidden=16,
                min_label_count=1,
            )

            self.assertTrue((Path(td) / "failure.pt").exists())
            self.assertTrue((Path(td) / "failure.json").exists())
            self.assertTrue((Path(td) / "failure.md").exists())

        self.assertEqual(len(loaded), 3)
        self.assertEqual(dataset.x.shape[0], 3)
        self.assertIn("stock_dead_end", dataset.labels)
        self.assertIn("val_metrics", report)
        self.assertFalse(report["metadata"]["split"]["has_target_overlap"])

    def test_split_by_target_has_no_target_overlap(self):
        rows = [
            {"target_smiles": "CCO", "labels": ["a"]},
            {"target_smiles": "OCC", "labels": ["b"]},
            {"target_smiles": "CCN", "labels": ["a"]},
            {"target_smiles": "CCC", "labels": ["b"]},
        ]

        train_idx, val_idx = split_by_target(rows, val_fraction=0.5)
        summary = split_summary(rows, train_idx, val_idx)

        self.assertFalse(summary["has_target_overlap"])
        self.assertGreater(summary["train_targets"], 0)
        self.assertGreater(summary["val_targets"], 0)


if __name__ == "__main__":
    unittest.main()
