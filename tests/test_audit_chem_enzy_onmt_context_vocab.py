import tempfile
import unittest
from pathlib import Path

from scripts.audit_chem_enzy_onmt_context_vocab import audit_context_vocab, render_markdown


class AuditChemEnzyONMTContextVocabTest(unittest.TestCase):
    def test_blocks_context_tokens_not_in_checkpoint_vocab(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = _fake_corpus(Path(td) / "corpus")

            result = audit_context_vocab(
                corpus_dir=corpus,
                checkpoint=None,
                checkpoint_src_tokens={"C", "O"},
                checkpoint_tgt_tokens={"C", "O", "."},
                modes=["plain", "context"],
                splits=["train"],
            )
            markdown = render_markdown(result)

            self.assertEqual(result["summary"]["decision"], "blocked_context_direct_train_from_checkpoint")
            self.assertFalse(result["summary"]["direct_context_train_from_checkpoint_ok"])
            self.assertEqual(result["summary"]["plain_src_oov_rate"], 0.0)
            self.assertGreater(result["summary"]["context_src_oov_rate"], 0.0)
            frequent = result["stats"]["context"]["src"]["corpus"]["frequent_oov"]
            self.assertEqual(frequent[0]["token"], "<step_1>")
            self.assertIn("native train_from will keep checkpoint vocab", result["decision"]["reason"])
            self.assertIn("Do not train the current context corpus directly", markdown)

    def test_allows_vocab_covered_context(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = _fake_corpus(Path(td) / "corpus")
            vocab = {"C", "O", ".", "<step_1>", "<target>", "<product>"}

            result = audit_context_vocab(
                corpus_dir=corpus,
                checkpoint=None,
                checkpoint_src_tokens=vocab,
                checkpoint_tgt_tokens=vocab,
                modes=["plain", "context"],
                splits=["train"],
            )

            self.assertEqual(result["summary"]["decision"], "compatible_by_vocab_audit")
            self.assertTrue(result["summary"]["direct_context_train_from_checkpoint_ok"])

    def test_blocks_context_target_oov_even_when_source_is_covered(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = _fake_corpus(Path(td) / "corpus")
            result = audit_context_vocab(
                corpus_dir=corpus,
                checkpoint=None,
                checkpoint_src_tokens={"C", "O", "<step_1>", "<target>", "<product>"},
                checkpoint_tgt_tokens={"C", "O"},
                modes=["context"],
                splits=["train"],
            )

            self.assertEqual(result["summary"]["decision"], "blocked_context_direct_train_from_checkpoint")
            self.assertEqual(result["summary"]["context_src_oov_rate"], 0.0)
            self.assertGreater(result["summary"]["context_tgt_oov_rate"], 0.0)


def _fake_corpus(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "plain.train.src").write_text("C C O\n", encoding="utf-8")
    (path / "plain.train.tgt").write_text("C . O\n", encoding="utf-8")
    (path / "context.train.src").write_text("<step_1> <target> C O <product> C O\n", encoding="utf-8")
    (path / "context.train.tgt").write_text("C . O\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
