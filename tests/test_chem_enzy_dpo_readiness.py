import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_chem_enzy_dpo_readiness import check_readiness, render_markdown


class ChemEnzyDPOReadinessTest(unittest.TestCase):
    def test_detects_trainable_vendor_but_blocks_direct_dpo(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vendor = _fake_vendor(root)
            pref_summary = root / "prefs.summary.json"
            pref_summary.write_text(
                json.dumps(
                    {
                        "summary": {
                            "schema_version": "cascade_verifier_preference_pack_summary.v1",
                            "n_groups": 2,
                            "n_pairs": 5,
                            "reason_counts": {"ph_conflict": 3, "cofactor_ledger_gap": 2},
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = check_readiness(vendor_root=vendor, preference_summary=pref_summary)

        summary = result["summary"]
        self.assertEqual(summary["overall_status"], "ready_for_supervised_adapter_manifest_not_direct_dpo")
        self.assertTrue(summary["supervised_vendor_training_ready"])
        self.assertFalse(summary["direct_dpo_ready"])
        self.assertFalse(summary["lora_ready"])
        self.assertEqual(summary["preference_pairs"], 5)
        self.assertIn("onmt_models", summary["configured_families"])
        self.assertIn("graphfp_models", summary["configured_families"])
        self.assertIn("template_relevance", summary["configured_families"])
        self.assertIn("vendor_dpo_loss_missing", summary["blockers"])
        self.assertIn("vendor_lora_adapter_missing", summary["blockers"])

        markdown = render_markdown(result)
        self.assertIn("ChemEnzy Cascade/DPO Readiness", markdown)
        self.assertIn("direct_dpo_ready: False", markdown)

    def test_reports_missing_preference_pack(self):
        with tempfile.TemporaryDirectory() as td:
            vendor = _fake_vendor(Path(td))

            result = check_readiness(vendor_root=vendor)

        self.assertEqual(result["summary"]["overall_status"], "trainable_vendor_detected_preference_pack_missing")
        self.assertIn("preference_pack_missing", result["summary"]["blockers"])


def _fake_vendor(root: Path) -> Path:
    vendor = root / "ChemEnzyRetroPlanner"
    retro = vendor / "retro_planner"
    config = retro / "config" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
one_step_model_configs:
  onmt_models:
    bionav_one_step:
      model_path:
      - packages/onmt/checkpoints/model_step_1.pt
      weight: 1.0
  graphfp_models:
    USPTO-full_remapped:
      graph_model_dumb: packages/graph_retrosyn/model.ckpt
      graph_dataset_root: packages/graph_retrosyn/data/raw
      weight: 1.0
  template_relevance:
    pistachio:
      state_name: pistachio
      weight: 1.0
""",
        encoding="utf-8",
    )
    paths = [
        retro / "packages" / "onmt" / "checkpoints" / "model_step_1.pt",
        retro / "packages" / "graph_retrosyn" / "model.ckpt",
        retro / "packages" / "onmt" / "onmt" / "bin" / "preprocess.py",
        retro / "packages" / "onmt" / "onmt" / "bin" / "train.py",
        retro / "packages" / "graph_retrosyn" / "graph_retrosyn" / "graph_train.py",
        retro / "packages" / "mlp_retrosyn" / "mlp_retrosyn" / "mlp_train.py",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n", encoding="utf-8")
    (retro / "packages" / "graph_retrosyn" / "data" / "raw").mkdir(parents=True)
    return vendor


if __name__ == "__main__":
    unittest.main()
