import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_external_toplevel_onmt_corpus import build_external_toplevel_corpus


class BuildExternalTopLevelONMTCorpusTest(unittest.TestCase):
    def test_builds_external_plain_and_context_corpus_with_stable_split(self):
        rows = [
            {
                "source": "fixture",
                "source_row_id": "a",
                "product": "CCO",
                "reactants": ["CC", "O"],
            },
            {
                "source": "fixture",
                "source_row_id": "b",
                "product": "CCN",
                "reactants": ["CC", "N"],
                "ec": "1.1.1.1",
            },
            {
                "source": "fixture",
                "source_row_id": "self",
                "product": "CCC",
                "reactants": ["CCC"],
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "onmt"
            with patch("scripts.build_external_toplevel_onmt_corpus._iter_one_source", return_value=iter(rows)):
                first = build_external_toplevel_corpus(
                    output_dir=out,
                    modes=["both"],
                    sources=["fixture"],
                    tokenizer="smiles_token",
                    split_policy="hash_90_5_5",
                )

            with patch("scripts.build_external_toplevel_onmt_corpus._iter_one_source", return_value=iter(rows)):
                second = build_external_toplevel_corpus(
                    output_dir=out,
                    modes=["both"],
                    sources=["fixture"],
                    tokenizer="smiles_token",
                    split_policy="hash_90_5_5",
                )

            plain_total = first["summary"]["total_examples"]["plain"]
            context_total = first["summary"]["total_examples"]["context"]
            self.assertEqual(plain_total, 2)
            self.assertEqual(context_total, 2)
            self.assertEqual(first["summary"]["source_counts"], {"fixture": 2})
            self.assertEqual(first["summary"]["skipped"]["self_reaction"], 1)
            self.assertEqual(
                first["summary"]["examples_by_mode_split"],
                second["summary"]["examples_by_mode_split"],
            )
            context_meta = []
            for split in ("train", "valid", "test"):
                meta_path = out / f"context.{split}.meta.jsonl"
                context_meta.extend(json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines())
            self.assertEqual(context_meta[0]["contract"], "External single-step top-level proposal positive; not an expert preference label.")
            self.assertEqual(context_meta[0]["reaction_smiles"], "CC.O>>CCO")
            self.assertTrue((out / "manifest.md").exists())

    def test_reads_ecreact_reactions_with_ec_pipe_annotation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "ecreact.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["rxn_smiles", "ec", "source"])
                writer.writeheader()
                writer.writerow({"rxn_smiles": "CCO.O|1.2.3.4>>CC=O", "ec": "1.2.3.4", "source": "fixture"})

            with patch("scripts.build_external_toplevel_onmt_corpus.Path") as mock_path:
                original_path = Path

                def path_side_effect(value):
                    if value == "data_external/ecreact/ecreact-1.0.csv":
                        return csv_path
                    return original_path(value)

                mock_path.side_effect = path_side_effect
                from scripts.build_external_toplevel_onmt_corpus import _iter_ecreact

                rows = list(_iter_ecreact())

            self.assertEqual(rows, [
                {
                    "source": "ecreact",
                    "source_row_id": 0,
                    "product": "CC=O",
                    "reactants": ["CCO", "O"],
                    "ec": "1.2.3.4",
                }
            ])

    def test_writes_canonical_training_smiles_and_keeps_raw_metadata(self):
        rows = [
            {
                "source": "fixture",
                "source_row_id": "canonical",
                "product": "C(C)O",
                "reactants": ["O", "C(C)"],
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "onmt"
            with patch("scripts.build_external_toplevel_onmt_corpus._iter_one_source", return_value=iter(rows)):
                manifest = build_external_toplevel_corpus(
                    output_dir=out,
                    modes=["plain"],
                    sources=["fixture"],
                    tokenizer="smiles_token",
                    split_policy="all_train",
                )

            self.assertTrue(manifest["canonicalize_training_smiles"])
            self.assertEqual((out / "plain.train.src").read_text(encoding="utf-8").strip(), "C C O")
            self.assertEqual((out / "plain.train.tgt").read_text(encoding="utf-8").strip(), "C C . O")
            meta = json.loads((out / "plain.train.meta.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(meta["product"], "CCO")
            self.assertEqual(meta["reactants"], ["CC", "O"])
            self.assertEqual(meta["raw_product"], "C(C)O")
            self.assertEqual(meta["raw_reactants"], ["O", "C(C)"])


if __name__ == "__main__":
    unittest.main()
