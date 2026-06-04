import tempfile
import unittest
from pathlib import Path

import torch

from scripts.extend_chem_enzy_onmt_context_vocab import extend_checkpoint_vocab


class ExtendChemEnzyONMTContextVocabTest(unittest.TestCase):
    def test_dry_run_reports_new_context_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _fake_context_corpus(root / "corpus")
            ckpt = _fake_checkpoint(root / "base.pt")

            result = extend_checkpoint_vocab(
                checkpoint=ckpt,
                corpus_dir=corpus,
                output_checkpoint=root / "extended.pt",
                dry_run=True,
                vendor_root=Path("missing"),
            )

            self.assertEqual(result["status"], "planned_not_written")
            self.assertEqual(result["old_vocab_size"], 4)
            self.assertEqual(result["new_vocab_size"], 8)
            self.assertEqual(result["n_new_tokens"], 4)
            self.assertFalse((root / "extended.pt").exists())

    def test_writes_shared_vocab_and_expands_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _fake_context_corpus(root / "corpus")
            ckpt = _fake_checkpoint(root / "base.pt")
            out = root / "extended.pt"

            result = extend_checkpoint_vocab(
                checkpoint=ckpt,
                corpus_dir=corpus,
                output_checkpoint=out,
                dry_run=False,
                vendor_root=Path("missing"),
            )

            saved = torch.load(out, map_location="cpu")
            vocab = saved["vocab"]["src"].fields[0][1].vocab
            self.assertEqual(result["status"], "written")
            self.assertIs(saved["vocab"]["src"].fields[0][1].vocab, saved["vocab"]["tgt"].fields[0][1].vocab)
            self.assertIn("<step_1>", vocab.stoi)
            self.assertIn(".", vocab.stoi)
            self.assertEqual(saved["model"]["encoder.embeddings.make_embedding.emb_luts.0.weight"].shape, (8, 2))
            self.assertEqual(saved["model"]["decoder.embeddings.make_embedding.emb_luts.0.weight"].shape, (8, 2))
            self.assertEqual(saved["generator"]["generator.0.weight"].shape, (8, 2))
            self.assertEqual(saved["generator"]["generator.0.bias"].shape, (8,))


def _fake_context_corpus(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "context.train.src").write_text("<step_1> <target> C <product> C\n", encoding="utf-8")
    (path / "context.train.tgt").write_text("C . O\n", encoding="utf-8")
    (path / "context.valid.src").write_text("", encoding="utf-8")
    (path / "context.valid.tgt").write_text("", encoding="utf-8")
    (path / "context.test.src").write_text("", encoding="utf-8")
    (path / "context.test.tgt").write_text("", encoding="utf-8")
    return path


def _fake_checkpoint(path: Path) -> Path:
    vocab = _FakeVocab(["<unk>", "<blank>", "C", "O"])
    field = _FakeField(vocab)
    multifield = _FakeMultiField(field)
    ckpt = {
        "vocab": {"src": multifield, "tgt": multifield},
        "model": {
            "encoder.embeddings.make_embedding.emb_luts.0.weight": torch.ones((4, 2)),
            "decoder.embeddings.make_embedding.emb_luts.0.weight": torch.ones((4, 2)) * 2,
        },
        "generator": {
            "generator.0.weight": torch.ones((4, 2)) * 3,
            "generator.0.bias": torch.ones((4,)),
        },
    }
    torch.save(ckpt, path)
    return path


class _FakeVocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(tokens)}


class _FakeField:
    def __init__(self, vocab):
        self.vocab = vocab


class _FakeMultiField:
    def __init__(self, field):
        self.fields = [("src", field)]


if __name__ == "__main__":
    unittest.main()
