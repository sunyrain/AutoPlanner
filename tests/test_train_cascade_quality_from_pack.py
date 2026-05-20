import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_cascade_quality_from_pack import (
    build_dataset,
    load_route_rows,
    train_cascade_quality_from_pack,
)


class TrainCascadeQualityFromPackTest(unittest.TestCase):
    def test_train_small_cascade_quality_pack(self):
        rows = [
            {
                "target_smiles": "CCO",
                "n_steps": 2,
                "type_sequence": ["oxidation", "hydrolysis"],
                "source_sequence": ["enzyformer", "retrochimera"],
                "ec1_sequence": ["1", "3"],
                "features": {"condition_success": 1.0, "compatibility_success": 1.0},
            },
            {
                "target_smiles": "CCN",
                "n_steps": 2,
                "type_sequence": ["oxidation", "C_C_coupling"],
                "source_sequence": ["retrochimera", "retrochimera"],
                "ec1_sequence": ["", ""],
                "features": {"condition_success": 0.0, "compatibility_success": 0.0},
            },
            {
                "target_smiles": "CCC",
                "n_steps": 1,
                "type_sequence": ["hydrolysis"],
                "source_sequence": ["v3_retrieval"],
                "ec1_sequence": ["3"],
                "features": {"condition_success": 1.0, "compatibility_success": 0.0},
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "route_value.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            loaded = load_route_rows(pack)
            dataset = build_dataset(loaded, n_bits=16, min_vocab_count=1)
            report = train_cascade_quality_from_pack(
                pack_dir=pack,
                model_output=Path(td) / "cascade_quality.pt",
                report_output=Path(td) / "cascade_quality.json",
                md_output=Path(td) / "cascade_quality.md",
                epochs=1,
                batch_size=2,
                n_bits=16,
                hidden=16,
                min_vocab_count=1,
            )

            self.assertTrue((Path(td) / "cascade_quality.pt").exists())
            self.assertTrue((Path(td) / "cascade_quality.json").exists())
            self.assertTrue((Path(td) / "cascade_quality.md").exists())

        self.assertEqual(len(loaded), 3)
        self.assertEqual(dataset.x.shape[0], 3)
        self.assertEqual(dataset.y.shape[1], 2)
        self.assertIn("condition_failure", report["val_metrics"]["per_label"])


if __name__ == "__main__":
    unittest.main()
