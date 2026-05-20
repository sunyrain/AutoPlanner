import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_v4_training_splits import build_v4_training_splits
from cascade_planner.eval.build_cascade_action_value_pack import build_cascade_action_value_pack
from cascade_planner.eval.run_pipeline_manifest_commands import run_manifest_commands
from cascade_planner.eval.run_stage3_baseline_value_training import (
    expected_baseline_inputs,
    missing_input_paths,
    run_stage3_baseline_value_training,
)
from cascade_planner.eval.run_v4_full_training_pipeline import plan_v4_full_training_pipeline


class V4FullTrainingPipelineTest(unittest.TestCase):
    def test_builds_grouped_full_v4_splits_without_doi_or_target_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            v4 = root / "v4.jsonl"
            benchmark = root / "benchmark.json"
            out = root / "split"
            benchmark.write_text(
                json.dumps([
                    {"doi": "10.locked", "cascade_id": "cascade_1", "target_smiles": "NN"}
                ]),
                encoding="utf-8",
            )
            rows = [
                _v4_row("10.a", "cascade_1", "CCO", "CC.O>>CCO", "all_chemical"),
                _v4_row("10.a", "cascade_2", "CCN", "CC.N>>CCN", "all_chemical"),
                _v4_row("10.b", "cascade_1", "CCC", "CC.C>>CCC", "chemoenzymatic"),
                _v4_row("10.c", "cascade_1", "CCC", "CC.C>>CCC", "chemoenzymatic"),
                _v4_row("10.d", "cascade_1", "CCCl", "CC.Cl>>CCCl", "all_enzymatic"),
                _v4_row("10.e", "cascade_1", "CCBr", "CC.Br>>CCBr", "all_enzymatic"),
                _v4_row("10.locked", "cascade_1", "CCF", "CC.F>>CCF", "all_chemical"),
                {**_v4_row("10.multi", "cascade_1", "CCO;CCN", "CC.O>>CCO", "all_chemical")},
            ]
            v4.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            manifest = build_v4_training_splits(
                v4_jsonl=v4,
                benchmark_path=benchmark,
                output_dir=out,
                group_by_scaffold=False,
            )
            all_rows = json.loads((out / "v4_trace_candidates_all.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["source_candidate_report"]["counts"]["candidate_rows"], 6)
        self.assertEqual(manifest["source_candidate_report"]["counts"]["skipped"]["benchmark_key_overlap"], 1)
        self.assertEqual(manifest["source_candidate_report"]["counts"]["skipped"]["missing_or_multi_target"], 1)
        self.assertEqual(manifest["leakage_checks"]["doi_cross_split"]["count"], 0)
        self.assertEqual(manifest["leakage_checks"]["target_cross_split"]["count"], 0)
        self.assertTrue(all(row.get("split") in {"train", "val", "test"} for row in all_rows))
        split_by_doi = {}
        for row in all_rows:
            split_by_doi.setdefault(row["doi"], set()).add(row["split"])
        self.assertEqual(len(split_by_doi["10.a"]), 1)
        split_by_target = {}
        for row in all_rows:
            split_by_target.setdefault(row["target_smiles"], set()).add(row["split"])
        self.assertEqual(len(split_by_target["CCC"]), 1)

    def test_pipeline_manifest_keeps_full100_out_of_training_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split_dir = root / "split"
            split_dir.mkdir()
            for name in ["v4_trace_candidates_all.json", "v4_trace_train.json", "v4_trace_val.json"]:
                (split_dir / name).write_text("[]", encoding="utf-8")
            (split_dir / "v4_trace_test.json").write_text("[]", encoding="utf-8")
            (split_dir / "v4_trace_split_manifest.json").write_text(
                json.dumps({"schema_version": "v4_trace_split_manifest.v1"}),
                encoding="utf-8",
            )

            manifest = plan_v4_full_training_pipeline(
                split_dir=split_dir,
                output_root=root / "pipeline",
                configs=["baseline"],
                splits=["train", "val"],
                num_shards=2,
                cascade_pair_reward_weight=0.005,
                cascade_pair_reward_mode="guarded_tie_break",
                cascade_pair_reward_tie_epsilon=0.01,
            )
            commands_path_exists = (root / "pipeline" / "v4_full_training_commands.sh").exists()

        pack_cmd = next(row for row in manifest["commands"] if row["stage"] == "build_action_source_pack")
        trace_cmd = next(row for row in manifest["commands"] if row["stage"] == "trace")
        bootstrap_trace_cmd = next(row for row in manifest["commands"] if row["stage"] == "bootstrap_trace")
        bootstrap_pack_cmd = next(row for row in manifest["commands"] if row["stage"] == "build_bootstrap_action_source_pack")
        bootstrap_action_cmd = next(row for row in manifest["commands"] if row["stage"] == "train_bootstrap_action_value")
        eval_cmd = next(row for row in manifest["commands"] if row["stage"] == "locked_full100_eval")
        self.assertTrue(trace_cmd["cmd"].startswith("scripts/run_cascade_benchmark_chem_enzy_env.sh "))
        self.assertIn("--include-route-outcomes", trace_cmd["cmd"])
        self.assertIn("--preserve-benchmark-splits", pack_cmd["cmd"])
        self.assertIn("v4_trace_candidates_all.json", pack_cmd["cmd"])
        self.assertNotIn("data/benchmark_v2_100.json", pack_cmd["cmd"])
        self.assertIn("--chem-enzy-cascade-source-policy", bootstrap_trace_cmd["cmd"])
        self.assertIn("source_value_model_path", bootstrap_trace_cmd["cmd"])
        self.assertIn("action_value_model_path", bootstrap_trace_cmd["cmd"])
        self.assertIn("--cascade-pair-reward-weight 0.005", bootstrap_trace_cmd["cmd"])
        self.assertIn("--cascade-pair-reward-mode guarded_tie_break", bootstrap_trace_cmd["cmd"])
        self.assertIn("--cascade-pair-reward-tie-epsilon 0.01", bootstrap_trace_cmd["cmd"])
        self.assertIn("--preserve-benchmark-splits", bootstrap_pack_cmd["cmd"])
        self.assertIn("--label-name route_outcome_action_value", bootstrap_action_cmd["cmd"])
        self.assertEqual(eval_cmd["training_use"], "forbidden")
        self.assertIn("data/benchmark_v2_100.json", eval_cmd["cmd"])
        self.assertIn("--chem-enzy-cascade-source-policy", eval_cmd["cmd"])
        self.assertIn("--cascade-pair-reward-mode guarded_tie_break", eval_cmd["cmd"])
        self.assertIn("final_action_value", manifest["models"])
        self.assertEqual(manifest["metadata"]["bootstrap_stage3"]["action_label"], "route_outcome_action_value")
        self.assertEqual(manifest["metadata"]["cascade_pair_reward"]["mode"], "guarded_tie_break")
        self.assertEqual(manifest["metadata"]["cascade_pair_reward"]["tie_epsilon"], 0.01)
        self.assertTrue(commands_path_exists)

    def test_manifest_command_runner_filters_without_execution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "commands": [
                        {"stage": "trace", "config": "baseline", "split": "train", "cmd": "echo one"},
                        {"stage": "trace", "config": "baseline", "split": "val", "cmd": "echo two"},
                        {"stage": "merge_runtime", "config": "baseline", "split": "train", "cmd": "echo three"},
                    ]
                }),
                encoding="utf-8",
            )

            report = run_manifest_commands(
                manifest_path=manifest,
                output_log_dir=root / "logs",
                stage="trace",
                config="baseline",
                split="train",
                command_indices=[1, 2],
                dry_run=True,
            )

        self.assertEqual(report["selected"][0]["index"], 1)
        self.assertEqual(len(report["selected"]), 1)

    def test_manifest_command_runner_skips_existing_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "marker.txt"
            output = root / "existing.json"
            output.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "commands": [
                        {
                            "stage": "trace",
                            "config": "baseline",
                            "split": "val",
                            "cmd": (
                                "python -c \"from pathlib import Path; "
                                f"Path(r'{marker}').write_text('ran', encoding='utf-8')\""
                            ),
                            "outputs": {"runtime": str(output)},
                        }
                    ]
                }),
                encoding="utf-8",
            )

            report = run_manifest_commands(
                manifest_path=manifest,
                output_log_dir=root / "logs",
                stage="trace",
                config="baseline",
                split="val",
                skip_existing=True,
            )

        self.assertFalse(marker.exists())
        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(report["skipped_count"], 1)
        self.assertTrue(report["results"][0]["skipped"])

    def test_action_value_pack_reads_cascade_trace_candidate_actions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "cascade_trace.jsonl"
            out = root / "pack"
            target = "CCO"
            benchmark.write_text(
                json.dumps([
                    {
                        "target_smiles": target,
                        "split": "train",
                        "route_domain": "all_chemical",
                        "gt_route": [{"rxn_smiles": "C.O>>CCO"}],
                    }
                ]),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "target_smiles": target,
                        "route_domain": "all_chemical",
                        "event": {
                            "state_id": "state0",
                            "state": {
                                "target_smiles": target,
                                "unresolved_failure_modes": [{"kind": "StockDeadEnd"}],
                                "step_annotations": [],
                                "stage_graph": {"stages": []},
                            },
                            "depth": 0,
                            "open_leaves": [target],
                            "expanded_leaf": target,
                            "failure_categories": ["StockDeadEnd"],
                            "candidate_scores": [0.9],
                            "child_summaries": [{"stock_closed": True, "cofactor_closed": True, "stage_count": 1, "step_count": 1}],
                            "candidate_actions": [
                                {
                                    "source": "ChemEnzyRetroPlanner",
                                    "target_leaf": target,
                                    "step": {
                                        "product_smiles": target,
                                        "reactant_smiles": ["C", "O"],
                                        "rxn_smiles": "C.O>>CCO",
                                        "source_model": "GraphFP",
                                        "score": 0.9,
                                        "raw_metadata": {"cost": 0.1},
                                    },
                                    "metadata": {
                                        "provider_rank": 1,
                                        "transition_value_score": 0.9,
                                        "candidate_selection_status": "selected_by_leaf",
                                    },
                                }
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_cascade_action_value_pack(
                trace_paths=[trace],
                benchmark_path=benchmark,
                output_dir=out,
                preserve_benchmark_splits=True,
            )
            rows = [
                json.loads(line)
                for line in (out / "action_value.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(report["summary"]["action_rows"], 1)
        self.assertEqual(rows[0]["parent_mol"], target)
        self.assertEqual(rows[0]["source_model"], "GraphFP")
        self.assertEqual(rows[0]["components"]["proposal_provider"], "ChemEnzyRetroPlanner")
        self.assertEqual(rows[0]["labels"]["exact_gt_reaction"], 1)

    def test_stage3_baseline_value_training_validate_only_checks_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split_dir = root / "split"
            split_dir.mkdir()
            (split_dir / "v4_trace_candidates_all.json").write_text("[]", encoding="utf-8")
            output_root = root / "pipeline"
            inputs = expected_baseline_inputs(
                output_root=output_root,
                split_dir=split_dir,
                num_shards=2,
            )
            for split in ["train", "val"]:
                for paths in (
                    inputs.runtime_shards[split],
                    inputs.expansion_traces[split],
                    inputs.cascade_traces[split],
                ):
                    for path in paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("{}" if path.suffix == ".json" else "", encoding="utf-8")

            report = run_stage3_baseline_value_training(
                output_root=output_root,
                split_dir=split_dir,
                num_shards=2,
                validate_only=True,
            )

        self.assertTrue(report["validated"])
        self.assertEqual(report["missing_inputs"], [])

    def test_stage3_baseline_value_training_reports_missing_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split_dir = root / "split"
            split_dir.mkdir()
            output_root = root / "pipeline"

            missing = missing_input_paths(
                expected_baseline_inputs(
                    output_root=output_root,
                    split_dir=split_dir,
                    num_shards=2,
                )
            )

        self.assertGreater(len(missing), 0)
        self.assertTrue(any("v4_trace_candidates_all.json" in str(path) for path in missing))


def _v4_row(doi: str, cascade_id: str, target: str, rxn: str, domain: str) -> dict:
    return {
        "doi": doi,
        "cascade_id": cascade_id,
        "trainable_recommended": True,
        "target_product_smiles": target,
        "cascade_type": domain,
        "quality_tier": "gold",
        "compatibility": {"compatibility_label": "empirically_compatible"},
        "steps": [{"step_index": 1, "rxn_smiles": rxn, "step_conditions": {}}],
    }


if __name__ == "__main__":
    unittest.main()
