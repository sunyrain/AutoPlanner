import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_context_onmt_proposal_preference_pack import build_preference_pack


class BuildContextONMTProposalPreferencePackTest(unittest.TestCase):
    def test_builds_rule_generated_hard_negative_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "corpus"
            corpus.mkdir()
            _write_meta(
                corpus / "context.train.meta.jsonl",
                [
                    _row("ex1", product="CCO", reactants=["CC", "O"], step_index=0),
                    _row("ex2", product="CCN", reactants=["CC", "N"], step_index=0),
                    _row("ex3", product="CCC", reactants=["CC"], step_index=0),
                ],
            )
            out_jsonl = root / "prefs" / "pairs.jsonl"
            out_summary = root / "prefs" / "summary.md"

            summary = build_preference_pack(
                corpus_dir=corpus,
                output_jsonl=out_jsonl,
                output_summary=out_summary,
                mode="context",
                split="train",
                negative_types=("self", "drop_aux", "cross_swap"),
            )

            pairs = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["source_examples"], 3)
            self.assertEqual(summary["counts"]["self"], 3)
            self.assertEqual(summary["counts"]["drop_aux"], 2)
            self.assertEqual(summary["counts"]["drop_aux_unavailable"], 1)
            self.assertEqual(summary["counts"]["cross_swap"], 3)
            self.assertEqual(summary["n_pairs"], 8)
            self.assertEqual(len(pairs), 8)
            self.assertTrue(out_summary.exists())

            by_id = {pair["pair_id"]: pair for pair in pairs}
            self.assertEqual(by_id["train_000000_self"]["rejected_reactants"], "CCO")
            self.assertEqual(by_id["train_000000_drop_aux"]["rejected_reactants"], "CC")
            self.assertEqual(by_id["train_000000_cross_swap"]["rejected_reactants"], "CC.N")
            self.assertNotIn("train_000002_drop_aux", by_id)

            for pair in pairs:
                self.assertNotEqual(pair["chosen_reactants"], pair["rejected_reactants"])
                self.assertEqual(pair["split"], "train")
                self.assertIn("not an expert label", pair["contract"])

    def test_skips_negative_when_it_canonicalizes_to_chosen_side(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "corpus"
            corpus.mkdir()
            _write_meta(
                corpus / "context.train.meta.jsonl",
                [
                    _row("same", product="CC.O", reactants=["O", "CC"], step_index=0),
                    _row("other", product="CCN", reactants=["CC", "N"], step_index=0),
                ],
            )
            out_jsonl = root / "pairs.jsonl"

            summary = build_preference_pack(
                corpus_dir=corpus,
                output_jsonl=out_jsonl,
                mode="context",
                split="train",
                negative_types=("self", "cross_swap"),
            )

            pairs = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["counts"]["self_same_as_chosen"], 1)
            self.assertNotIn("train_000000_self", {pair["pair_id"] for pair in pairs})
            self.assertIn("train_000000_cross_swap", {pair["pair_id"] for pair in pairs})


def _write_meta(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(example_id: str, *, product: str, reactants: list[str], step_index: int) -> dict:
    return {
        "source_example_id": example_id,
        "source_target_index": 1,
        "route_index": 0,
        "step_index": step_index,
        "split": "train",
        "product": product,
        "target_smiles": product,
        "reactants": reactants,
    }


if __name__ == "__main__":
    unittest.main()
