import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_benchmark_toplevel_onmt_corpus import build_benchmark_toplevel_corpus


class BuildBenchmarkTopLevelONMTCorpusTest(unittest.TestCase):
    def test_builds_top_level_plain_and_context_corpus(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            out = root / "onmt"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "gt_route": [
                                {"rxn_smiles": "C>>CC"},
                                {"rxn_smiles": "CC.O>>CCO", "transformation": "hydration"},
                            ],
                        },
                        {
                            "target_smiles": "CCN",
                            "gt_route": [
                                {"rxn_smiles": "CC.N>>CCN", "transformation": "amination"},
                            ],
                        },
                        {
                            "target_smiles": "CCC",
                            "gt_route": [
                                {"rxn_smiles": "C>>CC"},
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            manifest = build_benchmark_toplevel_corpus(
                benchmark_path=benchmark,
                output_dir=out,
                modes=["both"],
                tokenizer="smiles_token",
                split_policy="all_train",
            )

            plain_src = (out / "plain.train.src").read_text(encoding="utf-8").splitlines()
            plain_tgt = (out / "plain.train.tgt").read_text(encoding="utf-8").splitlines()
            context_src = (out / "context.train.src").read_text(encoding="utf-8").splitlines()
            meta_rows = [
                json.loads(line)
                for line in (out / "context.train.meta.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(manifest["summary"]["n_targets"], 3)
            self.assertEqual(manifest["summary"]["n_emitted_targets"], 2)
            self.assertEqual(manifest["summary"]["skipped"]["target_without_top_level_gt_step"], 1)
            self.assertEqual(manifest["summary"]["total_examples"]["plain"], 2)
            self.assertEqual(plain_src, ["C C O", "C C N"])
            self.assertEqual(plain_tgt, ["C C . O", "C C . N"])
            self.assertIn("<target> C C O <product> C C O", context_src[0])
            self.assertEqual(meta_rows[0]["benchmark_gt_step_index"], 1)
            self.assertEqual(meta_rows[0]["source"], "benchmark_top_level_gt")
            self.assertTrue(manifest["canonicalize_training_smiles"])
            self.assertTrue((out / "manifest.md").exists())

    def test_canonicalizes_top_level_gt_training_smiles_and_keeps_raw_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            out = root / "onmt"
            benchmark.write_text(
                json.dumps([
                    {
                        "target_smiles": "C(C)O",
                        "gt_route": [
                            {"rxn_smiles": "O.C(C)>>C(C)O", "transformation": "fixture"},
                        ],
                    }
                ]),
                encoding="utf-8",
            )

            build_benchmark_toplevel_corpus(
                benchmark_path=benchmark,
                output_dir=out,
                modes=["plain"],
                tokenizer="smiles_token",
                split_policy="all_train",
            )

            self.assertEqual((out / "plain.train.src").read_text(encoding="utf-8").strip(), "C C O")
            self.assertEqual((out / "plain.train.tgt").read_text(encoding="utf-8").strip(), "C C . O")
            meta = json.loads((out / "plain.train.meta.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(meta["product"], "CCO")
            self.assertEqual(meta["reactants"], ["CC", "O"])
            self.assertEqual(meta["raw_product"], "C(C)O")
            self.assertEqual(meta["raw_reactants"], ["O", "C(C)"])


if __name__ == "__main__":
    unittest.main()
