import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.analyze_cascade_action_ranking import analyze_cascade_action_ranking
from cascade_planner.eval.build_cascade_action_value_pack import build_cascade_action_value_pack


class CascadeActionValuePackTest(unittest.TestCase):
    def test_builds_action_and_source_value_rows_from_expansion_trace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            out = root / "pack"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            trace.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "target_smiles": "CCO",
                                "parent_mol": "CCO",
                                "parent_depth": 0,
                                "candidate_index": 0,
                                "source_model": "graphfp_models.USPTO-full_remapped",
                                "reaction_domain": "chemical",
                                "reactants": ["CC", "O"],
                                "total_cost": 0.3,
                                "context_features": {
                                    "route_domain": "all_chemical",
                                    "adjacent_reaction_domain": "enzymatic",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "target_smiles": "CCO",
                                "parent_mol": "CCO",
                                "parent_depth": 0,
                                "candidate_index": 1,
                                "source_model": "onmt_models.bionav_one_step",
                                "reaction_domain": "enzymatic",
                                "reactants": ["C", "CO"],
                                "total_cost": 0.6,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_cascade_action_value_pack(
                trace_path=trace,
                benchmark_path=benchmark,
                output_dir=out,
                val_fraction=0.0,
            )

            actions = [json.loads(line) for line in (out / "action_value.jsonl").read_text().splitlines()]
            sources = [json.loads(line) for line in (out / "source_value.jsonl").read_text().splitlines()]

        self.assertEqual(report["summary"]["action_rows"], 2)
        self.assertEqual(report["summary"]["exact_gt_hits"], 1)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]["supervision_contract"], "internal_search_action_value_not_record_gold.v1")
        self.assertEqual(actions[0]["labels"]["exact_gt_reaction"], 1)
        self.assertEqual(actions[0]["labels"]["action_value"], 1.0)
        self.assertEqual(actions[0]["labels"]["cascade_fragment_action_value"], 0.85)
        self.assertEqual(actions[0]["labels"]["state_action_value"], 0.9)
        self.assertEqual(actions[0]["context_features"]["adjacent_reaction_domain"], "enzymatic")
        self.assertEqual(actions[1]["labels"]["action_value"], 0.0)
        by_source = {row["source_model"]: row for row in sources}
        self.assertEqual(by_source["graphfp_models.USPTO-full_remapped"]["labels"]["source_value"], 1.0)
        self.assertEqual(
            by_source["graphfp_models.USPTO-full_remapped"]["context_features"]["adjacent_reaction_domain"],
            "enzymatic",
        )
        self.assertEqual(by_source["onmt_models.bionav_one_step"]["labels"]["source_value"], 0.0)
        self.assertTrue(all(row["labels"]["state_has_positive_action"] for row in actions))

    def test_reactant_overlap_gets_weak_action_value_without_exact_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            out = root / "pack"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "target_smiles": "CCO",
                        "parent_mol": "CCO",
                        "parent_depth": 0,
                        "candidate_index": 0,
                        "source_model": "graphfp_models.USPTO-full_remapped",
                        "reaction_domain": "chemical",
                        "reactants": ["CC", "N"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            build_cascade_action_value_pack(
                trace_path=trace,
                benchmark_path=benchmark,
                output_dir=out,
                val_fraction=0.0,
            )
            action = json.loads((out / "action_value.jsonl").read_text().splitlines()[0])
            source = json.loads((out / "source_value.jsonl").read_text().splitlines()[0])

        self.assertEqual(action["labels"]["exact_gt_reaction"], 0)
        self.assertEqual(action["labels"]["gt_reactant_hit"], 1)
        self.assertEqual(action["labels"]["action_value"], 0.35)
        self.assertEqual(action["labels"]["cascade_fragment_action_value"], 0.4)
        self.assertEqual(action["labels"]["state_action_value"], 0.45)
        self.assertEqual(source["labels"]["source_value"], 0.5)

    def test_adds_route_outcome_labels_from_runtime_programs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            runtime = root / "runtime.json"
            out = root / "pack"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "target_smiles": "CCO",
                        "parent_mol": "CCO",
                        "parent_depth": 0,
                        "candidate_index": 0,
                        "source_model": "graphfp",
                        "reaction_domain": "chemical",
                        "reactants": ["CC", "O"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "cascade_search": {
                                    "result_programs": [
                                        {
                                            "rank": 1,
                                            "solved": True,
                                            "route_rxns": ["CC.O>>CCO"],
                                            "route_reactants": ["CC", "O"],
                                            "route_outcome_value": 0.8,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            build_cascade_action_value_pack(
                trace_path=trace,
                benchmark_path=benchmark,
                runtime_path=runtime,
                output_dir=out,
                val_fraction=0.0,
            )
            action = json.loads((out / "action_value.jsonl").read_text().splitlines()[0])

        self.assertEqual(action["labels"]["program_route_action"], 1)
        self.assertEqual(action["labels"]["program_reactant_action"], 1)
        self.assertEqual(action["labels"]["route_outcome_action_value"], 0.8)
        self.assertEqual(action["labels"]["program_gt_signal_action"], 1)
        self.assertEqual(action["labels"]["route_quality_exact_action"], 1)
        self.assertEqual(action["labels"]["route_quality_reactant_action"], 1)
        self.assertEqual(action["labels"]["route_quality_action_value"], 0.8)
        self.assertEqual(action["labels"]["cascade_fragment_action_value"], 0.85)
        self.assertEqual(action["labels"]["state_action_value"], 0.9)

    def test_route_quality_label_filters_solved_only_runtime_programs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            runtime = root / "runtime.json"
            out = root / "pack"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "target_smiles": "CCO",
                        "parent_mol": "CCO",
                        "parent_depth": 0,
                        "candidate_index": 0,
                        "source_model": "graphfp",
                        "reaction_domain": "chemical",
                        "reactants": ["C", "CO"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "cascade_search": {
                                    "result_programs": [
                                        {
                                            "rank": 1,
                                            "solved": True,
                                            "route_rxns": ["C.CO>>CCO"],
                                            "route_reactants": ["C", "CO"],
                                            "route_outcome_value": 0.1,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            build_cascade_action_value_pack(
                trace_path=trace,
                benchmark_path=benchmark,
                runtime_path=runtime,
                output_dir=out,
                val_fraction=0.0,
            )
            action = json.loads((out / "action_value.jsonl").read_text().splitlines()[0])

        self.assertEqual(action["labels"]["program_route_action"], 1)
        self.assertEqual(action["labels"]["route_outcome_action_value"], 0.1)
        self.assertEqual(action["labels"]["program_gt_signal_action"], 0)
        self.assertEqual(action["labels"]["route_quality_action_value"], 0.0)
        self.assertEqual(action["labels"]["cascade_fragment_action_value"], 0.05)
        self.assertEqual(action["labels"]["state_action_value"], 0.075)

    def test_merges_route_outcomes_from_multiple_runtime_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            weak_runtime = root / "weak_runtime.json"
            strong_runtime = root / "strong_runtime.json"
            out = root / "pack"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "target_smiles": "CCO",
                        "parent_mol": "CCO",
                        "parent_depth": 0,
                        "candidate_index": 0,
                        "source_model": "graphfp",
                        "reaction_domain": "chemical",
                        "reactants": ["CC", "O"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            weak_runtime.write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "cascade_search": {
                                    "result_programs": [
                                        {
                                            "rank": 1,
                                            "solved": True,
                                            "route_rxns": ["CC.O>>CCO"],
                                            "route_reactants": ["CC", "O"],
                                            "route_outcome_value": 0.1,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            strong_runtime.write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "cascade_search": {
                                    "result_programs": [
                                        {
                                            "rank": 1,
                                            "solved": True,
                                            "route_rxns": ["CC.O>>CCO"],
                                            "route_reactants": ["CC", "O"],
                                            "route_outcome_value": 0.8,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = build_cascade_action_value_pack(
                trace_path=trace,
                benchmark_path=benchmark,
                runtime_paths=[weak_runtime, strong_runtime],
                output_dir=out,
                val_fraction=0.0,
            )
            action = json.loads((out / "action_value.jsonl").read_text().splitlines()[0])

        self.assertEqual(report["metadata"]["runtime_paths"], [str(weak_runtime), str(strong_runtime)])
        self.assertEqual(action["labels"]["route_outcome_action_value"], 0.8)
        self.assertEqual(action["labels"]["route_quality_action_value"], 0.8)

    def test_route_quality_positive_targets_are_stratified_across_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            runtime = root / "runtime.json"
            out = root / "pack"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                        },
                        {
                            "target_smiles": "CCN",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.N>>CCN"}],
                        },
                        {
                            "target_smiles": "CCC",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.C>>CCC"}],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            trace.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "target_smiles": "CCO",
                            "parent_mol": "CCO",
                            "parent_depth": 0,
                            "candidate_index": 0,
                            "source_model": "graphfp",
                            "reaction_domain": "chemical",
                            "reactants": ["CC", "O"],
                        },
                        {
                            "target_smiles": "CCN",
                            "parent_mol": "CCN",
                            "parent_depth": 0,
                            "candidate_index": 0,
                            "source_model": "graphfp",
                            "reaction_domain": "chemical",
                            "reactants": ["CC", "N"],
                        },
                        {
                            "target_smiles": "CCC",
                            "parent_mol": "CCC",
                            "parent_depth": 0,
                            "candidate_index": 0,
                            "source_model": "graphfp",
                            "reaction_domain": "chemical",
                            "reactants": ["C", "CC"],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "cascade_search": {
                                    "result_programs": [
                                        {
                                            "rank": 1,
                                            "solved": True,
                                            "route_rxns": ["CC.O>>CCO"],
                                            "route_reactants": ["CC", "O"],
                                            "route_outcome_value": 0.45,
                                        }
                                    ]
                                },
                            },
                            {
                                "target_smiles": "CCN",
                                "cascade_search": {
                                    "result_programs": [
                                        {
                                            "rank": 1,
                                            "solved": True,
                                            "route_rxns": ["CC.N>>CCN"],
                                            "route_reactants": ["CC", "N"],
                                            "route_outcome_value": 0.45,
                                        }
                                    ]
                                },
                            },
                            {
                                "target_smiles": "CCC",
                                "cascade_search": {
                                    "result_programs": [
                                        {
                                            "rank": 1,
                                            "solved": True,
                                            "route_rxns": ["C.CC>>CCC"],
                                            "route_reactants": ["C", "CC"],
                                            "route_outcome_value": 0.1,
                                        }
                                    ]
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            build_cascade_action_value_pack(
                trace_path=trace,
                benchmark_path=benchmark,
                runtime_path=runtime,
                output_dir=out,
                val_fraction=0.5,
            )
            actions = [json.loads(line) for line in (out / "action_value.jsonl").read_text().splitlines()]

        positive_splits = {
            row["split"]
            for row in actions
            if row["labels"]["route_quality_action_value"] > 0.0
        }
        self.assertEqual(positive_splits, {"train", "val"})

    def test_preserves_formal_benchmark_splits_when_requested(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            out = root / "pack"
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "target_smiles": "CCO",
                            "split": "train",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                        },
                        {
                            "target_smiles": "CCN",
                            "split": "val",
                            "route_domain": "all_chemical",
                            "gt_route": [{"rxn_smiles": "CC.N>>CCN"}],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            trace.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "target_smiles": "CCO",
                            "parent_mol": "CCO",
                            "parent_depth": 0,
                            "candidate_index": 0,
                            "source_model": "graphfp",
                            "reaction_domain": "chemical",
                            "reactants": ["CC", "O"],
                        },
                        {
                            "target_smiles": "CCN",
                            "parent_mol": "CCN",
                            "parent_depth": 0,
                            "candidate_index": 0,
                            "source_model": "graphfp",
                            "reaction_domain": "chemical",
                            "reactants": ["CC", "N"],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_cascade_action_value_pack(
                trace_path=trace,
                benchmark_path=benchmark,
                output_dir=out,
                val_fraction=0.0,
                preserve_benchmark_splits=True,
            )
            actions = [json.loads(line) for line in (out / "action_value.jsonl").read_text().splitlines()]

        self.assertTrue(report["metadata"]["preserve_benchmark_splits"])
        self.assertEqual(report["metadata"]["split_policy"], "preserve_benchmark_split_fields")
        self.assertEqual({row["target_smiles"]: row["split"] for row in actions}, {"CCO": "train", "CCN": "val"})

    def test_analyzes_action_ranking_inside_expansion_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            rows = [
                {
                    "state_id": "s1",
                    "route_domain": "all_chemical",
                    "source_model": "graphfp",
                    "reaction_domain": "chemical",
                    "base_cost": 0.1,
                    "total_cost": 0.1,
                    "labels": {"action_value": 0.0, "exact_gt_reaction": 0, "gt_reactant_hit": 0},
                },
                {
                    "state_id": "s1",
                    "route_domain": "all_chemical",
                    "source_model": "graphfp",
                    "reaction_domain": "chemical",
                    "base_cost": 0.3,
                    "total_cost": 0.3,
                    "labels": {"action_value": 1.0, "exact_gt_reaction": 1, "gt_reactant_hit": 1},
                },
            ]
            (pack / "action_value.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = analyze_cascade_action_ranking(pack_dir=pack, output_path=root / "analysis.json")

        self.assertEqual(report["metadata"]["n_action_rows"], 2)
        self.assertEqual(report["cost_rankings"]["base_cost"]["positive_states"], 1)
        self.assertEqual(report["cost_rankings"]["base_cost"]["top1_positive_state_hit_rate"], 0.0)
        self.assertEqual(report["cost_rankings"]["base_cost"]["top5_positive_state_hit_rate"], 1.0)
        self.assertEqual(report["breakdowns"]["source_model"]["graphfp"]["positive_actions"], 1)

    def test_analyzes_action_ranking_with_custom_positive_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            rows = [
                {
                    "state_id": "s1",
                    "route_domain": "all_chemical",
                    "source_model": "graphfp",
                    "reaction_domain": "chemical",
                    "base_cost": 0.1,
                    "total_cost": 0.1,
                    "labels": {
                        "action_value": 1.0,
                        "program_route_action": 0,
                        "exact_gt_reaction": 0,
                        "gt_reactant_hit": 1,
                    },
                },
                {
                    "state_id": "s1",
                    "route_domain": "all_chemical",
                    "source_model": "graphfp",
                    "reaction_domain": "chemical",
                    "base_cost": 0.3,
                    "total_cost": 0.3,
                    "labels": {
                        "action_value": 0.0,
                        "program_route_action": 1,
                        "exact_gt_reaction": 0,
                        "gt_reactant_hit": 0,
                    },
                },
            ]
            (pack / "action_value.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = analyze_cascade_action_ranking(
                pack_dir=pack,
                output_path=root / "analysis.json",
                label_name="program_route_action",
            )

        self.assertEqual(report["metadata"]["positive_label"], "program_route_action")
        self.assertEqual(report["cost_rankings"]["base_cost"]["positive_states"], 1)
        self.assertEqual(report["cost_rankings"]["base_cost"]["top1_positive_state_hit_rate"], 0.0)
        self.assertEqual(report["breakdowns"]["source_model"]["graphfp"]["positive_actions"], 1)


if __name__ == "__main__":
    unittest.main()
