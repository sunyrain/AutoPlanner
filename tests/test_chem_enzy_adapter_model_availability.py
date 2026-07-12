import os
import sys
import tempfile
import unittest
from pathlib import Path

from cascade_planner.baselines.chem_enzy_adapter import (
    _materialize_selected_one_step_io_paths,
    _normal_absolute_path,
    _patch_torchtext_vocab_legacy_api,
    _prune_unavailable_one_step_models,
    _vendor_pythonpath,
    _windows_extended_path,
)


class ChemEnzyModelAvailabilityTest(unittest.TestCase):
    def test_missing_onmt_checkpoint_is_pruned_when_native_fallback_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_root = Path(tmp) / "ChemEnzyRetroPlanner"
            native = vendor_root / "retro_planner" / "packages" / "onmt" / "checkpoints" / "np-like" / "model_step_100000.pt"
            native.parent.mkdir(parents=True)
            native.write_bytes(b"stub")
            graph_model = vendor_root / "retro_planner" / "graph.ckpt"
            graph_model.write_bytes(b"stub")
            graph_dataset = vendor_root / "retro_planner" / "graph-data"
            graph_dataset.mkdir()
            missing = Path(tmp) / "missing" / "bionav_v2.pt"
            vendor_config = {
                "one_step_model_configs": {
                    "graphfp_models": {
                        "USPTO-full_remapped": {
                            "graph_model_dumb": "graph.ckpt",
                            "graph_dataset_root": "graph-data",
                        }
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

    def test_all_missing_onmt_models_fail_closed_with_empty_selection(self):
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

        self.assertEqual(selected, [])
        self.assertIsNotNone(report)
        self.assertEqual(report["action"], "all_selected_models_unavailable")
        self.assertEqual(report["selected_after"], [])

    def test_missing_graph_model_is_pruned_before_vendor_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_root = Path(tmp) / "ChemEnzyRetroPlanner"
            dataset = vendor_root / "retro_planner" / "graph-data"
            dataset.mkdir(parents=True)
            vendor_config = {
                "one_step_model_configs": {
                    "graphfp_models": {
                        "USPTO-full_remapped": {
                            "graph_model_dumb": "missing.ckpt",
                            "graph_dataset_root": "graph-data",
                        }
                    }
                }
            }

            selected, report = _prune_unavailable_one_step_models(
                ["graphfp_models.USPTO-full_remapped"],
                vendor_config=vendor_config,
                vendor_root=vendor_root,
            )

        self.assertEqual(selected, [])
        self.assertEqual(report["action"], "all_selected_models_unavailable")
        self.assertEqual(
            report["unavailable"][0]["reason"],
            "configured_checkpoint_missing",
        )

    def test_missing_model_config_rejects_entire_mixed_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_root = Path(tmp) / "ChemEnzyRetroPlanner"
            graph_model = vendor_root / "retro_planner" / "graph.ckpt"
            graph_model.parent.mkdir(parents=True)
            graph_model.write_bytes(b"stub")
            graph_dataset = vendor_root / "retro_planner" / "graph-data"
            graph_dataset.mkdir()
            vendor_config = {
                "one_step_model_configs": {
                    "graphfp_models": {
                        "valid": {
                            "graph_model_dumb": "graph.ckpt",
                            "graph_dataset_root": "graph-data",
                        }
                    }
                }
            }

            selected, report = _prune_unavailable_one_step_models(
                ["graphfp_models.valid", "onmt_models.not_configured"],
                vendor_config=vendor_config,
                vendor_root=vendor_root,
            )

        self.assertEqual(selected, [])
        self.assertTrue(report["configuration_error"])
        self.assertEqual(report["action"], "rejected_invalid_model_selection")
        self.assertEqual(report["unavailable"][0]["reason"], "missing_model_config")

    def test_invalid_full_name_and_unknown_type_fail_closed(self):
        for full_name, expected_reason in (
            ("not-a-full-name", "invalid_model_full_name"),
            ("mystery_models.variant", "unknown_model_type"),
        ):
            with self.subTest(full_name=full_name):
                selected, report = _prune_unavailable_one_step_models(
                    [full_name],
                    vendor_config={"one_step_model_configs": {}},
                    vendor_root=Path("."),
                )

                self.assertEqual(selected, [])
                self.assertTrue(report["configuration_error"])
                self.assertEqual(
                    report["unavailable"][0]["reason"],
                    expected_reason,
                )

    @unittest.skipUnless(os.name == "nt", "Windows path contract")
    def test_vendor_pythonpath_strips_device_prefix_from_import_root_and_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_root = Path(tmp) / "ChemEnzyRetroPlanner"
            (vendor_root / "retro_planner").mkdir(parents=True)
            extended_root = Path("\\\\?\\" + str(vendor_root.resolve()))

            with _vendor_pythonpath(extended_root):
                self.assertFalse(str(Path.cwd()).startswith("\\\\?\\"))
                self.assertEqual(Path.cwd(), vendor_root.resolve())
                self.assertFalse(any(item.startswith("\\\\?\\") for item in sys.path))

    @unittest.skipUnless(os.name == "nt", "Windows path contract")
    def test_only_concrete_overlong_model_path_receives_device_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_root = Path(tmp) / "ChemEnzyRetroPlanner"
            vendor_root.mkdir()
            long_relative = "models/" + ("x" * 260) + ".ckpt"
            config = {
                "one_step_model_configs": {
                    "graphfp_models": {
                        "long": {
                            "graph_model_dumb": long_relative,
                            "graph_dataset_root": "data",
                        }
                    }
                }
            }

            materialized = _materialize_selected_one_step_io_paths(
                config,
                ["graphfp_models.long"],
                vendor_root=vendor_root,
            )
            model_path = materialized["one_step_model_configs"]["graphfp_models"]["long"][
                "graph_model_dumb"
            ]

        self.assertFalse(str(_normal_absolute_path(vendor_root)).startswith("\\\\?\\"))
        self.assertTrue(model_path.startswith("\\\\?\\"))
        self.assertFalse(str(_windows_extended_path(vendor_root)).startswith("\\\\?\\"))

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
