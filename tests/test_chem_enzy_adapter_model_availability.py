import tempfile
import unittest
from pathlib import Path

from cascade_planner.baselines.chem_enzy_adapter import (
    _patch_torchtext_vocab_legacy_api,
    _prune_unavailable_one_step_models,
)


class ChemEnzyModelAvailabilityTest(unittest.TestCase):
    def test_missing_onmt_checkpoint_is_pruned_when_native_fallback_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_root = Path(tmp) / "ChemEnzyRetroPlanner"
            native = vendor_root / "retro_planner" / "packages" / "onmt" / "checkpoints" / "np-like" / "model_step_100000.pt"
            native.parent.mkdir(parents=True)
            native.write_bytes(b"stub")
            missing = Path(tmp) / "missing" / "bionav_v2.pt"
            vendor_config = {
                "one_step_model_configs": {
                    "graphfp_models": {
                        "USPTO-full_remapped": {"graph_model_dumb": "unused"}
                    },
                    "onmt_models": {
                        "bionav_one_step": {"model_path": [str(missing)]},
                        "bionav_native_one_step": {
                            "model_path": ["packages/onmt/checkpoints/np-like/model_step_100000.pt"]
                        },
                    },
                }
            }

            selected, report = _prune_unavailable_one_step_models(
                [
                    "graphfp_models.USPTO-full_remapped",
                    "onmt_models.bionav_one_step",
                    "onmt_models.bionav_native_one_step",
                ],
                vendor_config=vendor_config,
                vendor_root=vendor_root,
            )

        self.assertEqual(
            selected,
            [
                "graphfp_models.USPTO-full_remapped",
                "onmt_models.bionav_native_one_step",
            ],
        )
        self.assertIsNotNone(report)
        self.assertEqual(report["action"], "pruned_unavailable_models")
        self.assertEqual(report["unavailable"][0]["model"], "onmt_models.bionav_one_step")
        self.assertIn(str(missing), report["unavailable"][0]["missing_paths"])

    def test_all_missing_onmt_models_are_reported_without_emptying_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_root = Path(tmp) / "ChemEnzyRetroPlanner"
            missing = Path(tmp) / "missing.pt"
            vendor_config = {
                "one_step_model_configs": {
                    "onmt_models": {
                        "bionav_one_step": {"model_path": [str(missing)]},
                    },
                }
            }

            selected, report = _prune_unavailable_one_step_models(
                ["onmt_models.bionav_one_step"],
                vendor_config=vendor_config,
                vendor_root=vendor_root,
            )

        self.assertEqual(selected, ["onmt_models.bionav_one_step"])
        self.assertIsNotNone(report)
        self.assertEqual(report["action"], "all_selected_models_unavailable_no_prune")

    def test_torchtext_vocab_legacy_patch_reads_old_checkpoint_state(self):
        from torchtext.vocab import Vocab

        _patch_torchtext_vocab_legacy_api()
        vocab = Vocab.__new__(Vocab)
        vocab.__dict__.update(
            {
                "itos": ["<unk>", "C", "O"],
                "stoi": {"<unk>": 0, "C": 1, "O": 2},
                "freqs": {},
            }
        )

        self.assertEqual(len(vocab), 3)
        self.assertIn("C", vocab)
        self.assertEqual(vocab["O"], 2)
        self.assertEqual(vocab.lookup_token(1), "C")
        self.assertEqual(vocab.lookup_tokens([2, 1]), ["O", "C"])
        self.assertEqual(vocab.lookup_indices(["C", "O"]), [1, 2])


if __name__ == "__main__":
    unittest.main()
