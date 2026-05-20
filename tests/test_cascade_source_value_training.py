import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_cascade_source_value import train_cascade_source_value


class CascadeSourceValueTrainingTest(unittest.TestCase):
    def test_trains_source_value_model_from_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            rows = []
            for idx in range(8):
                split = "val" if idx >= 6 else "train"
                rows.append(
                    {
                        "target_smiles": f"CCO{idx}",
                        "split": split,
                        "route_domain": "all_enzymatic" if idx % 2 else "all_chemical",
                        "state_id": f"s{idx // 2}",
                        "parent_mol": "CCO",
                        "parent_depth": idx % 3,
                        "source_model": "onmt_models.bionav_one_step" if idx % 2 else "graphfp_models.USPTO-full_remapped",
                        "reaction_domain": "enzymatic" if idx % 2 else "chemical",
                        "context_features": {
                            "adjacent_reaction_domain": "chemical" if idx % 2 else "unknown",
                        },
                        "labels": {
                            "source_value": 1.0 if idx in {0, 3, 6} else 0.0,
                            "state_has_positive_action": 1,
                        },
                    }
                )
            (pack / "source_value.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            model_output = root / "source_value.pt"
            report_output = root / "report.json"

            report = train_cascade_source_value(
                pack_dir=pack,
                model_output=model_output,
                report_output=report_output,
                epochs=2,
                batch_size=4,
                hidden=16,
                device="cpu",
            )

            self.assertTrue(model_output.exists())
            self.assertTrue(report_output.exists())
            self.assertEqual(report["metadata"]["label_contract"], "internal_search_source_value_not_record_gold.v1")
            self.assertEqual(report["metadata"]["selection_metric"], "top1_positive_state_hit_rate")
            self.assertEqual(report["metadata"]["feature_schema"]["schema_version"], "cascade_source_value_features.v2")
            self.assertIn(
                "adjacent_reaction_domain",
                report["metadata"]["feature_schema"]["categorical_fields"],
            )
            self.assertIn("best_checkpoint", report)
            self.assertIn("top1_positive_state_hit_rate", report["final_metrics"])


if __name__ == "__main__":
    unittest.main()
