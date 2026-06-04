import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_bionav_v2_enzyme_corpus import build_corpus


class BuildBioNavV2EnzymeCorpusTest(unittest.TestCase):
    def test_builds_training_corpus_and_locked_native_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_json = root / "train.json"
            val_json = root / "val.json"
            ecreact_csv = root / "ecreact.csv"
            out = root / "out"
            train_json.write_text(
                json.dumps(
                    [
                        {"reactants": "CC.O", "product": "CCO", "ec": "1.1.1.1"},
                        {"reactants": "CC.N", "product": "CCN", "ec": "2.7.1.1"},
                    ]
                ),
                encoding="utf-8",
            )
            val_json.write_text(
                json.dumps(
                    [
                        {"reactants": "N.CC", "product": "CCN", "ec": "2.7.1.1"},
                        {"reactants": "CCC.O", "product": "CCCO", "ec": "1.1.1.2"},
                    ]
                ),
                encoding="utf-8",
            )
            ecreact_csv.write_text(
                "rxn_smiles,ec,source\nCC.O|1.1.1.1>>CCO,1.1.1.1,unit\n",
                encoding="utf-8",
            )

            manifest = build_corpus(
                output_dir=out,
                bridge_pool=None,
                ecreact_csv=ecreact_csv,
                enzymatic_retro_train_json=train_json,
                enzymatic_retro_val_json=val_json,
                valid_fraction=0.0,
                test_fraction=0.0,
                seed=7,
            )

            self.assertEqual(manifest["benchmark"]["examples"], 2)
            self.assertEqual(manifest["training_corpus"]["examples_by_split"]["train"], 1)
            self.assertEqual(manifest["training_corpus"]["skipped"]["benchmark_exact_overlap"], 1)
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "manifest.md").exists())

            plain_train_src = (out / "plain.train.src").read_text(encoding="utf-8").splitlines()
            plain_train_tgt = (out / "plain.train.tgt").read_text(encoding="utf-8").splitlines()
            ec_train_src = (out / "ec_context.train.src").read_text(encoding="utf-8").splitlines()
            bench_src = (out / "benchmark" / "native_bionav_benchmark.src").read_text(encoding="utf-8").splitlines()
            bench_meta = [
                json.loads(line)
                for line in (out / "benchmark" / "native_bionav_benchmark.meta.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertEqual(plain_train_src, ["C C O"])
            self.assertEqual(plain_train_tgt, ["C C . O"])
            self.assertEqual(ec_train_src[0], "<ec1_1> <ec_1_1_1_1> <product> C C O")
            self.assertEqual(len(bench_src), 2)
            self.assertEqual({row["split"] for row in bench_meta}, {"benchmark"})


if __name__ == "__main__":
    unittest.main()
