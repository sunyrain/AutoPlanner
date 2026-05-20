import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_cascade_fragment_pack import build_cascade_fragment_pack
from cascade_planner.eval.train_cascade_fragment_scorer import train_cascade_fragment_scorer


class CascadeFragmentScorerTest(unittest.TestCase):
    def test_build_and_train_fragment_scorer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            v4 = root / "v4.jsonl"
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps([]), encoding="utf-8")
            records = [
                _record("10.a", "simultaneous", "simultaneous", "water", "water", "water", 30, 31, 32),
                _record("10.b", "telescoped", "sequential_addition", "water", "water", "water", 25, 26, 27),
                _record("10.c", "isolated_transfer", "isolated_transfer", "toluene", "water", "water", 90, 30, 31),
                _record("10.d", "simultaneous", "telescoped", "water", "water", "water", 28, 29, 30),
            ]
            v4.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            pack_dir = root / "fragment_pack"

            pack_report = build_cascade_fragment_pack(
                v4_jsonl=v4,
                benchmark_path=benchmark,
                output_dir=pack_dir,
                max_window_size=3,
                hard_negative_per_positive=1,
            )
            model_path = root / "fragment.pt"
            train_report = train_cascade_fragment_scorer(
                pack_dir=pack_dir,
                model_output=model_path,
                report_output=root / "fragment_report.json",
                md_output=root / "fragment_report.md",
                epochs=1,
                batch_size=4,
                n_bits=16,
                hidden=32,
                device="cpu",
            )

            model_exists = model_path.exists()

            self.assertGreaterEqual(pack_report["counts"]["rows"], 8)
            self.assertGreaterEqual(pack_report["counts"]["positive_rows"], 4)
            self.assertTrue(model_exists)
            self.assertGreaterEqual(train_report["metadata"]["n_rows"], 8)
            self.assertIn("fragment_preference_auc", train_report["final_metrics"])


def _record(doi, mode1, mode2, solvent1, solvent2, solvent3, temp1, temp2, temp3):
    return {
        "doi": doi,
        "cascade_id": "cascade_1",
        "trainable_recommended": True,
        "target_product_smiles": "CCN",
        "cascade_type": "all_enzymatic",
        "quality_tier": "gold",
        "compatibility": {"compatibility_label": "empirically_compatible"},
        "steps": [
            _step(1, "CCO>>CC=O", mode1, solvent1, temp1),
            _step(2, "CC=O>>CC=N", mode2, solvent2, temp2),
            _step(3, "CC=N>>CCN", "not_applicable", solvent3, temp3),
        ],
    }


def _step(index, rxn, mode, solvent, temp):
    return {
        "step_id": f"s{index}",
        "step_index": index,
        "rxn_smiles": rxn,
        "pairwise_mode": mode,
        "step_mode": "biocatalytic",
        "transformation_name": "amination",
        "transformation_superclass": "biocatalysis",
        "intermediate_isolated": mode == "isolated_transfer",
        "step_conditions": {"temperature_c": temp, "ph": 7.0, "solvent": solvent},
        "catalyst_components": [{"catalyst_class": "enzyme", "ec_number": "1.1.1.1"}],
    }


if __name__ == "__main__":
    unittest.main()
