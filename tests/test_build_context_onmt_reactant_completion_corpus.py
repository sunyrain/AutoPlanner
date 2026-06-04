import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_context_onmt_reactant_completion_corpus import build_completion_corpus


class BuildContextONMTReactantCompletionCorpusTest(unittest.TestCase):
    def test_builds_completion_rows_from_rule_corruptions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "corpus"
            corpus.mkdir()
            _write_meta(
                corpus / "context.train.meta.jsonl",
                [
                    _row("ex1", product="C(C)O", reactants=["O", "C(C)"]),
                    _row("ex2", product="CCN", reactants=["CC", "N"]),
                    _row("ex3", product="CCC", reactants=["CC"]),
                ],
            )
            out = root / "completion"

            manifest = build_completion_corpus(
                corpus_dir=corpus,
                output_dir=out,
                mode="context",
                tokenizer="smiles_token",
                splits=("train",),
                corruption_types=("drop_one", "self", "cross_swap", "empty"),
            )

            src_rows = (out / "context.train.src").read_text(encoding="utf-8").splitlines()
            tgt_rows = (out / "context.train.tgt").read_text(encoding="utf-8").splitlines()
            meta_rows = [json.loads(line) for line in (out / "context.train.meta.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(manifest["summary"]["counts"]["train_drop_one"], 2)
            self.assertEqual(manifest["summary"]["counts"]["train_drop_one_unavailable"], 1)
            self.assertEqual(manifest["summary"]["counts"]["train_self"], 3)
            self.assertEqual(manifest["summary"]["counts"]["train_cross_swap"], 2)
            self.assertEqual(manifest["summary"]["counts"]["train_empty"], 3)
            self.assertEqual(manifest["summary"]["counts"]["train_duplicate_completion"], 1)
            self.assertEqual(len(src_rows), 10)
            self.assertEqual(len(tgt_rows), 10)
            self.assertTrue(all("<candidate>" in row for row in src_rows))
            self.assertEqual(meta_rows[0]["product"], "CCO")
            self.assertEqual(meta_rows[0]["chosen_reactants"], "CC.O")
            self.assertEqual(meta_rows[0]["given_reactants"], "CC")
            self.assertIn("No expert preference label", meta_rows[0]["contract"])
            self.assertTrue((out / "manifest.md").exists())

    def test_dedupes_identical_completion_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "corpus"
            corpus.mkdir()
            _write_meta(
                corpus / "context.train.meta.jsonl",
                [
                    _row("a", product="CCO", reactants=["CC", "O"]),
                    _row("b", product="OCC", reactants=["O", "CC"]),
                ],
            )
            out = root / "completion"

            manifest = build_completion_corpus(
                corpus_dir=corpus,
                output_dir=out,
                mode="context",
                tokenizer="smiles_token",
                splits=("train",),
                corruption_types=("drop_one",),
            )

            self.assertEqual(manifest["summary"]["total_examples"], 1)
            self.assertEqual(manifest["summary"]["counts"]["train_duplicate_completion"], 1)


def _write_meta(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(example_id: str, *, product: str, reactants: list[str]) -> dict:
    return {
        "source_example_id": example_id,
        "source_target_index": 1,
        "route_index": 0,
        "step_index": 0,
        "split": "train",
        "product": product,
        "target_smiles": product,
        "reactants": reactants,
    }


if __name__ == "__main__":
    unittest.main()
