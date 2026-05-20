import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_training_pack import build_training_pack


class BuildTrainingPackTest(unittest.TestCase):
    def test_builds_manifest_and_labels_exact_candidate(self):
        benchmark = [{
            "doi": "10.test/example",
            "cascade_id": "c1",
            "target_smiles": "CCO",
            "route_domain": "all_chemical",
            "operation_mode": "sequential",
            "depth": 1,
            "gt_route": [{
                "rxn_smiles": "CC.O>>CCO",
                "ec_number": None,
                "transformation": "coupling",
            }],
        }]
        route = {
            "target": "CCO",
            "route_recovery": {
                "exact_route_reaction_match_any": True,
                "exact_reaction_in_route_pool": True,
                "candidate_exact_reaction_in_pool": True,
                "candidate_gt_reactant_in_pool": True,
            },
            "routes": [{
                "score": 10.0,
                "confidence": 0.8,
                "n_steps": 1,
                "steps": [{
                    "product": "CCO",
                    "main_reactant": "CC",
                    "aux_reactants": ["O"],
                    "reaction_smiles": "CC.O>>CCO",
                    "reaction_type": "coupling",
                    "ec": "",
                    "source": "fake",
                    "candidate_pool": {
                        "top_candidates": [
                            {
                                "main_reactant": "CC",
                                "aux_reactants": ["O"],
                                "reaction_smiles": "CC.O>>CCO",
                                "reaction_type": "coupling",
                                "source": "fake",
                                "score": 0.9,
                                "T": 35.0,
                                "pH": 7.5,
                                "solvent": "buffer",
                                "evidence": {"doi": "10.test/candidate"},
                            },
                            {
                                "main_reactant": "CCC",
                                "aux_reactants": [],
                                "reaction_smiles": "CCC>>CCO",
                                "reaction_type": "other",
                                "source": "fake",
                                "score": 0.1,
                            },
                        ],
                    },
                }],
                "metrics": {
                    "filled_route": True,
                    "progressive_route": True,
                    "route_solved": True,
                    "professional_solved": True,
                    "strict_stock_solve": True,
                    "retrosynthesis_progress": {
                        "main_chain_reduction": 0.5,
                        "largest_leaf_reduction": 0.5,
                    },
                    "route_naturalness": {"naturalness_score": 1.0},
                    "condition": {"condition_window_success": True},
                    "cascade_compatibility": {"cascade_compatibility_success": True, "issues": []},
                    "enzyme_evidence": {"enzyme_evidence_score": 0.0},
                },
            }],
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bench_path = root / "bench.json"
            input_path = root / "plan.json"
            out_dir = root / "pack"
            bench_path.write_text(json.dumps(benchmark), encoding="utf-8")
            input_path.write_text(json.dumps(route), encoding="utf-8")

            manifest = build_training_pack(
                input_paths=[input_path],
                benchmark_paths=[bench_path],
                output_dir=out_dir,
            )
            candidate_rows = [
                json.loads(line)
                for line in (out_dir / "candidate_ranking.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            route_rows = [
                json.loads(line)
                for line in (out_dir / "route_value.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            skeleton_rows = [
                json.loads(line)
                for line in (out_dir / "skeleton_prior.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(manifest["counts"]["route_value"], 1)
        self.assertEqual(manifest["counts"]["candidate_ranking"], 2)
        self.assertGreaterEqual(manifest["counts"]["skeleton_prior"], 2)
        self.assertEqual(manifest["quality"]["candidate_exact_gt"], 1)
        self.assertEqual(candidate_rows[0]["label_type"], "benchmark_exact")
        self.assertTrue(candidate_rows[0]["exact_gt_reactants"])
        self.assertEqual(candidate_rows[0]["candidate"]["T"], 35.0)
        self.assertEqual(candidate_rows[0]["candidate"]["pH"], 7.5)
        self.assertEqual(candidate_rows[0]["candidate"]["solvent"], "buffer")
        self.assertEqual(candidate_rows[0]["candidate"]["doi"], "10.test/candidate")
        self.assertEqual(route_rows[0]["route_domain"], "all_chemical")
        self.assertEqual(route_rows[0]["recovery_bottleneck"], "recovered_exact_route")
        self.assertTrue(route_rows[0]["recovery_summary"]["candidate_exact_reaction_in_pool"])
        self.assertEqual(candidate_rows[0]["target_recovery_bottleneck"], "recovered_exact_route")
        planner_skeletons = [row for row in skeleton_rows if row["source"] == "planner_route"]
        self.assertEqual(planner_skeletons[0]["route_domain"], "all_chemical")
        self.assertEqual(planner_skeletons[0]["operation_mode"], "sequential")


if __name__ == "__main__":
    unittest.main()
