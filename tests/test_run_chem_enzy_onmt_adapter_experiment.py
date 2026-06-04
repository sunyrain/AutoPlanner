import tempfile
import unittest
from pathlib import Path

from scripts.run_chem_enzy_onmt_adapter_experiment import (
    _adapter_ab_summary,
    _normalize_eval_splits,
    build_experiment_plan,
    collect_outputs,
    render_markdown,
    write_manifest,
)


class ChemEnzyONMTAdapterExperimentTest(unittest.TestCase):
    def test_dry_run_plan_records_commands_and_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _fake_corpus(root / "corpus")
            vendor = _fake_vendor(root / "vendor" / "ChemEnzyRetroPlanner")
            runtime_python = root / "env" / "bin" / "python"
            runtime_python.parent.mkdir(parents=True)
            runtime_python.write_text("# python placeholder\n", encoding="utf-8")
            output_dir = root / "out"

            plan = build_experiment_plan(
                corpus_dir=corpus,
                output_dir=output_dir,
                vendor_root=vendor,
                runtime_python=runtime_python,
                base_checkpoint=vendor
                / "retro_planner"
                / "packages"
                / "onmt"
                / "checkpoints"
                / "np-like"
                / "model_step_100000.pt",
                train_steps=7,
                learning_rate=0.0001,
                translate_tokenizer="token",
                src_seq_length=150,
                tgt_seq_length=120,
                gpuid=0,
                eval_splits=["valid", "test"],
                eval_limit=12,
            )
            plan["status"] = "planned_not_executed"
            collect_outputs(plan)
            write_manifest(plan)
            markdown = render_markdown(plan)

            self.assertEqual(plan["summary"]["status"], "planned_not_executed")
            self.assertFalse(plan["summary"]["promotion_ready"])
            self.assertEqual([row["label"] for row in plan["commands"]], ["preprocess", "train", "eval_valid", "eval_test"])
            self.assertTrue(plan["inputs"]["runtime_python"]["exists"])
            self.assertTrue(plan["inputs"]["base_checkpoint"]["exists"])
            self.assertIn("-learning_rate 0.0001", plan["commands"][1]["cmd"])
            self.assertIn("-train_steps 7", plan["commands"][1]["cmd"])
            self.assertIn("-world_size 1 -gpu_ranks 0", plan["commands"][1]["cmd"])
            self.assertIn("-src_seq_length 150", plan["commands"][0]["cmd"])
            self.assertIn("-tgt_seq_length 120", plan["commands"][0]["cmd"])
            self.assertEqual(plan["settings"]["src_seq_length"], 150)
            self.assertEqual(plan["settings"]["tgt_seq_length"], 120)
            self.assertIn("--limit 12", plan["commands"][2]["cmd"])
            self.assertIn("--tokenizer token", plan["commands"][2]["cmd"])
            self.assertIn("does not promote the adapter", plan["promotion_contract"])
            self.assertIn("Promotion Gate", markdown)
            self.assertTrue((output_dir / "experiment_manifest.json").exists())
            self.assertTrue((output_dir / "experiment_manifest.md").exists())

    def test_can_skip_preprocess_and_train_for_eval_only_plan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _fake_corpus(root / "corpus")
            vendor = _fake_vendor(root / "vendor" / "ChemEnzyRetroPlanner")

            plan = build_experiment_plan(
                corpus_dir=corpus,
                output_dir=root / "out",
                vendor_root=vendor,
                runtime_python=Path("/usr/bin/python"),
                base_checkpoint=vendor
                / "retro_planner"
                / "packages"
                / "onmt"
                / "checkpoints"
                / "np-like"
                / "model_step_100000.pt",
                skip_preprocess=True,
                skip_train=True,
                eval_splits=["valid"],
                eval_limit=None,
            )

            self.assertEqual([row["label"] for row in plan["commands"]], ["eval_valid"])
            self.assertNotIn("--limit", plan["commands"][0]["cmd"])
            self.assertEqual(plan["settings"]["eval_limit"], None)

    def test_can_plan_context_mode_after_vocab_extension(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _fake_corpus(root / "corpus")
            vendor = _fake_vendor(root / "vendor" / "ChemEnzyRetroPlanner")

            plan = build_experiment_plan(
                corpus_dir=corpus,
                output_dir=root / "out",
                vendor_root=vendor,
                runtime_python=Path("/usr/bin/python"),
                base_checkpoint=vendor
                / "retro_planner"
                / "packages"
                / "onmt"
                / "checkpoints"
                / "np-like"
                / "model_step_100000.pt",
                mode="context",
                translate_tokenizer="pretokenized",
                eval_splits=["valid"],
                eval_limit=5,
            )

            self.assertIn("context.train.src", plan["commands"][0]["cmd"])
            self.assertIn("context.valid.src", plan["commands"][2]["cmd"])
            self.assertIn("--tokenizer pretokenized", plan["commands"][2]["cmd"])

    def test_rejects_context_mode_with_smiles_tokenizer_eval(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _fake_corpus(root / "corpus")
            vendor = _fake_vendor(root / "vendor" / "ChemEnzyRetroPlanner")

            with self.assertRaisesRegex(ValueError, "context mode evaluation requires"):
                build_experiment_plan(
                    corpus_dir=corpus,
                    output_dir=root / "out",
                    vendor_root=vendor,
                    runtime_python=Path("/usr/bin/python"),
                    base_checkpoint=vendor
                    / "retro_planner"
                    / "packages"
                    / "onmt"
                    / "checkpoints"
                    / "np-like"
                    / "model_step_100000.pt",
                    mode="context",
                    translate_tokenizer="token",
                    eval_splits=["valid"],
                )

    def test_eval_splits_default_and_deduplicate(self):
        self.assertEqual(_normalize_eval_splits(None), ["valid"])
        self.assertEqual(_normalize_eval_splits(["valid", "test", "valid"]), ["valid", "test"])

    def test_adapter_ab_summary_holds_when_smoke_has_no_exact_recall_lift(self):
        summary = _adapter_ab_summary(
            {
                "eval_valid": [
                    {"model_path": "native.pt", "n_examples": 20, "nonempty": 20, "top1_exact": 0, "top5_exact": 0, "top1_rate": 0.0, "top5_rate": 0.0},
                    {"model_path": "adapter.pt", "n_examples": 20, "nonempty": 20, "top1_exact": 0, "top5_exact": 0, "top1_rate": 0.0, "top5_rate": 0.0},
                ],
                "eval_test": [
                    {"model_path": "native.pt", "n_examples": 20, "nonempty": 20, "top1_exact": 5, "top5_exact": 6, "top1_rate": 0.25, "top5_rate": 0.3},
                    {"model_path": "adapter.pt", "n_examples": 20, "nonempty": 20, "top1_exact": 5, "top5_exact": 6, "top1_rate": 0.25, "top5_rate": 0.3},
                ],
            }
        )

        self.assertEqual(summary["decision"], "hold_no_exact_recall_lift")
        self.assertFalse(summary["promotion_ready"])
        self.assertEqual(summary["rows"][0]["top1_delta"], 0)
        self.assertEqual(summary["rows"][1]["topk_delta"], 0)

    def test_adapter_ab_summary_flags_larger_ab_candidate_only_on_lift(self):
        summary = _adapter_ab_summary(
            {
                "eval_valid": [
                    {"model_path": "native.pt", "n_examples": 20, "nonempty": 20, "top1_exact": 0, "top5_exact": 0},
                    {"model_path": "adapter.pt", "n_examples": 20, "nonempty": 20, "top1_exact": 1, "top5_exact": 2},
                ]
            }
        )

        self.assertEqual(summary["decision"], "candidate_for_larger_route_level_ab")
        self.assertFalse(summary["promotion_ready"])


def _fake_corpus(path: Path) -> Path:
    path.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (path / f"plain.{split}.src").write_text("C C O\n", encoding="utf-8")
        (path / f"plain.{split}.tgt").write_text("C C . O\n", encoding="utf-8")
        (path / f"context.{split}.src").write_text("<step_1> <target> C C O <product> C C O\n", encoding="utf-8")
        (path / f"context.{split}.tgt").write_text("C C . O\n", encoding="utf-8")
    return path


def _fake_vendor(path: Path) -> Path:
    retro = path / "retro_planner"
    files = [
        retro / "packages" / "onmt" / "onmt" / "bin" / "preprocess.py",
        retro / "packages" / "onmt" / "onmt" / "bin" / "train.py",
        retro / "packages" / "onmt" / "checkpoints" / "np-like" / "model_step_100000.pt",
    ]
    for file_path in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("# placeholder\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
