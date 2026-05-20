import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_chem_enzy_cascade_onmt_corpus import build_corpus


class BuildChemEnzyCascadeONMTCorpusTest(unittest.TestCase):
    def test_builds_plain_and_context_corpora_from_positive_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.json"
            out = root / "onmt"
            pack.write_text(json.dumps({"examples": [_positive_seed(), _negative_seed()]}), encoding="utf-8")

            manifest = build_corpus(input_path=pack, output_dir=out, modes=["both"], dedupe=True)

            plain_train_src = (out / "plain.train.src").read_text(encoding="utf-8").splitlines()
            plain_train_tgt = (out / "plain.train.tgt").read_text(encoding="utf-8").splitlines()
            context_train_src = (out / "context.train.src").read_text(encoding="utf-8").splitlines()
            context_train_tgt = (out / "context.train.tgt").read_text(encoding="utf-8").splitlines()
            meta_rows = [
                json.loads(line)
                for line in (out / "context.train.meta.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "manifest.md").exists())

            self.assertEqual(manifest["summary"]["source_seed_routes"], 1)
            self.assertEqual(manifest["summary"]["total_examples"]["plain"], 2)
            self.assertEqual(manifest["summary"]["total_examples"]["context"], 2)
            self.assertEqual(plain_train_src[0], "C C C C O")
            self.assertEqual(plain_train_tgt[0], "C C C C")
            self.assertIn("<step_1>", context_train_src[0])
            self.assertIn("<product>", context_train_src[0])
            self.assertIn("<solv_water>", context_train_src[0])
            self.assertEqual(context_train_tgt[1], "C C . C C")
            self.assertEqual(meta_rows[0]["stage"], "stage_1")


def _positive_seed() -> dict:
    return {
        "example_id": "seed",
        "label": 1,
        "source_target_index": 1,
        "target_smiles": "CCCCO",
        "cascade": {
            "metadata": {"split": "train"},
            "stage_partition": ["stage_1", "stage_1"],
            "steps": [
                {
                    "product": "CCCCO",
                    "main_reactant": "CCCC",
                    "aux_reactants": [],
                    "T": 30,
                    "pH": 7,
                    "solvent": "water",
                    "ec": "1.1.1.1",
                },
                {
                    "product": "CCCC",
                    "reactants": ["CC", "CC"],
                    "T": 32,
                    "pH": 7.2,
                    "solvent": "water",
                },
            ],
        },
    }


def _negative_seed() -> dict:
    row = _positive_seed()
    row["example_id"] = "neg"
    row["label"] = 0
    return row


if __name__ == "__main__":
    unittest.main()
