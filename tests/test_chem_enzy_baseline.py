import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cascade_planner.baselines.chem_enzy_adapter import (
    BACKEND_NAME,
    ChemEnzyBackendAdapter,
    _vendor_pythonpath,
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate, RouteStepCandidate
from cascade_planner.eval.build_cascade_gold_smoke import build_cascade_gold_smoke
from cascade_planner.eval.analyze_chem_enzy_expansion_trace import analyze_expansion_trace
from cascade_planner.eval.chem_enzy_broad_union import build_union_report
from cascade_planner.eval.compare_chem_enzy_baseline import compare_baselines
from cascade_planner.eval import run_cascade_search_benchmark as benchmark_runner
from cascade_planner.eval.run_cascade_search_benchmark import _chem_enzy_search_flags_for_row
from cascade_planner.eval.run_cascade_search_benchmark import _cascade_result_programs
from cascade_planner.eval.run_cascade_search_benchmark import _expansion_proposal_cache_from_chem_enzy
from cascade_planner.eval.run_cascade_search_benchmark import _merge_proposal_caches
from cascade_planner.eval.run_cascade_search_benchmark import _proposal_cache_from_chem_enzy
from cascade_planner.eval.run_cascade_search_benchmark import _proposal_pool_keys
from cascade_planner.eval.run_cascade_search_benchmark import _run_one_target
from cascade_planner.eval.run_cascade_search_benchmark import _sort_proposal_cache
from cascade_planner.eval.run_cascade_search_benchmark import _validate_stock_inputs
from cascade_planner.eval.run_cascade_search_benchmark import _validate_model_inputs
from cascade_planner.eval.run_cascade_search_benchmark import ProductAuditFinalReranker
from cascade_planner.cascade_search import CascadeSearchConfig


class ChemEnzyBaselineAdapterTest(unittest.TestCase):
    def test_converts_vendor_route_dict_to_route_candidate(self):
        raw = {
            "time": 1.25,
            "iter": 3,
            "route_lens": 1,
            "all_succ_dict_routes": [
                {
                    "smiles": "CCO",
                    "type": "mol",
                    "in_stock": False,
                    "children": [
                        {
                            "type": "reaction",
                            "template": {"model_full_name": "graphfp_models.USPTO-full_remapped"},
                            "rxn_attribute": {
                                "condition": {
                                    "columns": ["Temperature", "Solvent"],
                                    "data": [[25.0, "water"]],
                                },
                                "enzyme_assign": {
                                    "columns": ["Ranks", "EC Number", "Confidence"],
                                    "data": [["Top-1", "1.1.1.1", 0.91]],
                                },
                            },
                            "children": [
                                {"smiles": "CC", "type": "mol", "in_stock": True},
                                {"smiles": "O", "type": "mol", "in_stock": True},
                            ],
                        }
                    ],
                }
            ],
        }

        routes = route_candidates_from_chem_enzy_result(raw, target_smiles="CCO")

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].backend, BACKEND_NAME)
        self.assertTrue(routes[0].solved)
        self.assertTrue(routes[0].enzymatic_step_present)
        self.assertEqual(routes[0].steps[0].rxn_smiles, "CC.O>>CCO")
        self.assertEqual(routes[0].steps[0].condition_predictions[0]["Solvent"], "water")
        self.assertEqual(routes[0].steps[0].enzyme_ec_annotations[0]["ec_number"], "1.1.1.1")

    def test_dry_run_reports_missing_vendor_as_structured_failure(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing_vendor"
            adapter = ChemEnzyBackendAdapter(vendor_root=missing)
            result = adapter.run_target(RouteSearchConfig(target_smiles="CCO"), dry_run=True)

        categories = {failure.category for failure in result.failures}
        self.assertIn("vendor_missing", categories)
        self.assertIn("config_missing", categories)
        self.assertTrue(all(failure.target_smiles == "CCO" for failure in result.failures))
        self.assertFalse(result.solved)
        self.assertTrue(result.raw_backend_metadata["dry_run"])

    def test_batch_run_reuses_planner_for_matching_configs(self):
        class FakePlanner:
            def plan(self, target):
                return {
                    "time": 0.1,
                    "iter": 1,
                    "all_succ_dict_routes": [
                        {
                            "smiles": target,
                            "type": "mol",
                            "children": [
                                {
                                    "type": "reaction",
                                    "children": [{"smiles": "CC", "type": "mol", "in_stock": True}],
                                }
                            ],
                        }
                    ],
                }

        class FakeAdapter(ChemEnzyBackendAdapter):
            builds = 0

            def _build_planner(self, search_config):
                self.builds += 1
                return FakePlanner()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "retro_planner" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("{}", encoding="utf-8")
            adapter = FakeAdapter(vendor_root=root, config_path=config)
            results = adapter.run_targets([
                RouteSearchConfig(target_smiles="CCO"),
                RouteSearchConfig(target_smiles="CCC"),
            ])

        self.assertEqual(adapter.builds, 1)
        self.assertEqual([result.target_smiles for result in results], ["CCO", "CCC"])
        self.assertTrue(all(result.solved for result in results))

    def test_batch_run_reuses_planner_for_row_runtime_context(self):
        class FakePlanner:
            def __init__(self):
                self.config = {}
                self.cascade_search_context = {}
                self.seen_contexts = []

            def plan(self, target):
                self.seen_contexts.append(dict(self.cascade_search_context))
                return {
                    "time": 0.1,
                    "iter": 1,
                    "all_succ_dict_routes": [
                        {
                            "smiles": target,
                            "type": "mol",
                            "children": [
                                {
                                    "type": "reaction",
                                    "children": [{"smiles": "CC", "type": "mol", "in_stock": True}],
                                }
                            ],
                        }
                    ],
                }

        class FakeAdapter(ChemEnzyBackendAdapter):
            builds = 0

            def _build_planner(self, search_config):
                self.builds += 1
                self.fake_planner = FakePlanner()
                return self.fake_planner

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "retro_planner" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("{}", encoding="utf-8")
            adapter = FakeAdapter(vendor_root=root, config_path=config)
            results = adapter.run_targets([
                RouteSearchConfig(
                    target_smiles="CCO",
                    search_flags={"cascade_search_context": {"route_domain": "all_chemical"}},
                ),
                RouteSearchConfig(
                    target_smiles="CCC",
                    search_flags={"cascade_search_context": {"route_domain": "all_enzymatic"}},
                ),
            ])

        self.assertEqual(adapter.builds, 1)
        self.assertEqual([result.target_smiles for result in results], ["CCO", "CCC"])
        self.assertEqual(adapter.fake_planner.seen_contexts, [
            {"route_domain": "all_chemical"},
            {"route_domain": "all_enzymatic"},
        ])

    def test_pandas_json_attrs_do_not_mark_organic_steps_as_enzymatic(self):
        raw = {
            "all_succ_dict_routes": [
                {
                    "smiles": "CCO",
                    "type": "mol",
                    "children": [
                        {
                            "type": "reaction",
                            "rxn_attribute": {
                                "organic_enzyme_rxn_classification": json.dumps({
                                    "Reaction Type": {"0": "Organic Reaction"},
                                    "Confidence": {"0": 0.99},
                                }),
                                "enzyme_assign": json.dumps({
                                    "Ranks": {"0": "Top-1"},
                                    "EC Number": {"0": "1.1.1.1"},
                                    "Confidence": {"0": 0.91},
                                }),
                            },
                            "children": [{"smiles": "CC", "type": "mol", "in_stock": True}],
                        }
                    ],
                }
            ],
        }

        routes = route_candidates_from_chem_enzy_result(raw, target_smiles="CCO")

        self.assertFalse(routes[0].enzymatic_step_present)

    def test_pandas_json_attrs_mark_classified_enzymatic_steps(self):
        raw = {
            "all_succ_dict_routes": [
                {
                    "smiles": "CCO",
                    "type": "mol",
                    "children": [
                        {
                            "type": "reaction",
                            "rxn_attribute": {
                                "organic_enzyme_rxn_classification": json.dumps({
                                    "Reaction Type": {"0": "Enzymatic Reaction"},
                                    "Confidence": {"0": 0.99},
                                }),
                                "enzyme_assign": json.dumps({
                                    "Ranks": {"0": "Top-1"},
                                    "EC Number": {"0": "1.1.1.1"},
                                    "Confidence": {"0": 0.91},
                                }),
                            },
                            "children": [{"smiles": "CC", "type": "mol", "in_stock": True}],
                        }
                    ],
                }
            ],
        }

        routes = route_candidates_from_chem_enzy_result(raw, target_smiles="CCO")

        self.assertTrue(routes[0].enzymatic_step_present)
        self.assertEqual(routes[0].steps[0].enzyme_ec_annotations[0]["ec_number"], "1.1.1.1")

    def test_vendor_config_passes_cascade_cost_context(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "retro_planner" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"stocks": {"Test-stock": "stock.csv"}}), encoding="utf-8")
            adapter = ChemEnzyBackendAdapter(vendor_root=root, config_path=config)

            vendor_config = adapter._vendor_config(
                RouteSearchConfig(
                    target_smiles="CCO",
                    stock_names=["Test-stock"],
                    search_flags={
                        "cascade_search_context": {
                            "enabled": True,
                            "preferred_reaction_domains": ["enzymatic"],
                        },
                        "cascade_cost_model": {
                            "enabled": True,
                            "weights": {"preferred_domain_reward": 1.2},
                        },
                    },
                )
            )

        self.assertTrue(vendor_config["use_cascade_cost_model"])
        self.assertEqual(vendor_config["cascade_search_context"]["preferred_reaction_domains"], ["enzymatic"])
        self.assertEqual(vendor_config["cascade_cost_model"]["weights"]["preferred_domain_reward"], 1.2)

    def test_vendor_config_can_override_onmt_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "retro_planner" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps({
                    "stocks": {"Test-stock": "stock.csv"},
                    "one_step_model_configs": {
                        "onmt_models": {
                            "bionav_one_step": {
                                "model_path": ["packages/onmt/checkpoints/np-like/model_step_100000.pt"],
                                "beam_size": 20,
                                "weight": 1.0,
                            }
                        }
                    },
                }),
                encoding="utf-8",
            )
            checkpoint = root / "trained" / "plain_continue_step_300.pt"
            adapter = ChemEnzyBackendAdapter(vendor_root=root, config_path=config)

            vendor_config = adapter._vendor_config(
                RouteSearchConfig(
                    target_smiles="CCO",
                    stock_names=["Test-stock"],
                    search_flags={"chem_enzy_onmt_model_path": str(checkpoint)},
                )
            )

        model_config = vendor_config["one_step_model_configs"]["onmt_models"]["bionav_one_step"]
        self.assertEqual(model_config["model_path"], [str(checkpoint)])

    def test_vendor_config_normalizes_action_value_model_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "retro_planner" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"stocks": {"Test-stock": "stock.csv"}}), encoding="utf-8")
            adapter = ChemEnzyBackendAdapter(vendor_root=root, config_path=config)

            vendor_config = adapter._vendor_config(
                RouteSearchConfig(
                    target_smiles="CCO",
                    stock_names=["Test-stock"],
                    search_flags={
                        "cascade_cost_model": {
                            "enabled": True,
                            "action_value_model_path": "models/action_value.pt",
                        },
                    },
                )
            )

        self.assertTrue(Path(vendor_config["cascade_cost_model"]["action_value_model_path"]).is_absolute())

    def test_vendor_config_passes_cascade_source_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "retro_planner" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"stocks": {"Test-stock": "stock.csv"}}), encoding="utf-8")
            adapter = ChemEnzyBackendAdapter(vendor_root=root, config_path=config)

            vendor_config = adapter._vendor_config(
                RouteSearchConfig(
                    target_smiles="CCO",
                    stock_names=["Test-stock"],
                    search_flags={
                        "cascade_search_context": {
                            "enabled": True,
                            "preferred_reaction_domains": ["enzymatic"],
                        },
                        "cascade_source_policy": {
                            "enabled": True,
                            "unpreferred_topk_fraction": 0.25,
                        },
                    },
                )
            )

        self.assertFalse(vendor_config.get("use_cascade_cost_model", False))
        self.assertTrue(vendor_config["use_cascade_source_policy"])
        self.assertEqual(vendor_config["cascade_source_policy"]["unpreferred_topk_fraction"], 0.25)

    def test_vendor_config_normalizes_source_value_model_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "retro_planner" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"stocks": {"Test-stock": "stock.csv"}}), encoding="utf-8")
            adapter = ChemEnzyBackendAdapter(vendor_root=root, config_path=config)

            vendor_config = adapter._vendor_config(
                RouteSearchConfig(
                    target_smiles="CCO",
                    stock_names=["Test-stock"],
                    search_flags={
                        "cascade_source_policy": {
                            "enabled": True,
                            "source_value_model_path": "models/source_value.pt",
                        },
                    },
                )
            )

        self.assertTrue(Path(vendor_config["cascade_source_policy"]["source_value_model_path"]).is_absolute())

    def test_cascade_cost_hook_changes_molstar_route_choice(self):
        vendor_root = Path("vendor/ChemEnzyRetroPlanner")

        def expand(mol):
            if mol != "P":
                return None
            return {
                "reactants": ["A", "B"],
                "scores": [0.9, 0.2],
                "template": ["chem-template", "enzyme-template"],
                "model_full_name": [
                    "graphfp_models.USPTO-full_remapped",
                    "onmt_models.bionav_one_step",
                ],
            }

        with _vendor_pythonpath(vendor_root):
            from retro_planner.search_frame.mcts_star.cascade_cost import RuleCascadeCostModel
            from retro_planner.search_frame.mcts_star.molmcts_star import mol_planner

            original_succ, original_msg = mol_planner(
                target_mol="P",
                target_mol_id=0,
                starting_mols={"A", "B"},
                expand_fn=expand,
                iterations=1,
                max_depth=2,
                exclude_target=False,
            )
            self.assertTrue(original_succ)
            original_route = original_msg[0].route_to_dict()
            self.assertEqual(original_route["children"][0]["children"][0]["smiles"], "A")

            model = RuleCascadeCostModel(
                {
                    "weights": {
                        "preferred_domain_reward": 2.0,
                        "discouraged_domain_penalty": 2.0,
                        "active_failure_match_reward": 1.0,
                        "min_cost": 0.0,
                    }
                }
            )
            cascade_succ, cascade_msg = mol_planner(
                target_mol="P",
                target_mol_id=0,
                starting_mols={"A", "B"},
                expand_fn=expand,
                iterations=1,
                max_depth=2,
                exclude_target=False,
                cascade_cost_model=model,
                cascade_search_context={
                    "enabled": True,
                    "preferred_reaction_domains": ["enzymatic"],
                    "active_failure_modes": ["EnzymeEvidenceWeak"],
                },
            )

        self.assertTrue(cascade_succ)
        cascade_route = cascade_msg[0].route_to_dict()
        reaction = cascade_route["children"][0]
        self.assertEqual(reaction["children"][0]["smiles"], "B")
        self.assertEqual(reaction["cascade_cost"]["reaction_domain"], "enzymatic")
        self.assertLess(reaction["cost"], original_route["children"][0]["cost"])
        trace = cascade_msg[4]
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[1]["source_model"], "onmt_models.bionav_one_step")
        self.assertLess(trace[1]["total_cost"], trace[0]["total_cost"])

    def test_cascade_source_policy_rebalances_source_budget(self):
        vendor_root = Path("vendor/ChemEnzyRetroPlanner")

        with _vendor_pythonpath(vendor_root):
            from retro_planner.search_frame.mcts_star.cascade_source_policy import RuleCascadeSourcePolicy

            decision = RuleCascadeSourcePolicy(
                {
                    "enabled": True,
                    "unpreferred_topk_fraction": 0.20,
                    "min_unpreferred_topk": 5,
                    "active_failure_topk_multiplier": 1.0,
                }
            ).decide(
                parent_mol="P",
                available_models=[
                    "graphfp_models.USPTO-full_remapped",
                    "onmt_models.bionav_one_step",
                ],
                expansion_topk=50,
                context={
                    "enabled": True,
                    "preferred_reaction_domains": ["enzymatic"],
                },
                parent_depth=0,
            )

        self.assertIsNone(decision.select_models)
        self.assertEqual(decision.topk_by_model["onmt_models.bionav_one_step"], 50)
        self.assertEqual(decision.topk_by_model["graphfp_models.USPTO-full_remapped"], 10)
        self.assertEqual(decision.model_domains["onmt_models.bionav_one_step"], "enzymatic")
        self.assertIn("rebalance_topk_by_preferred_domain", decision.reasons)

    def test_cascade_cost_hook_uses_learned_source_value_decision(self):
        vendor_root = Path("vendor/ChemEnzyRetroPlanner")

        with _vendor_pythonpath(vendor_root):
            from retro_planner.search_frame.mcts_star.cascade_cost import RuleCascadeCostModel

            model = RuleCascadeCostModel(
                {
                    "weights": {
                        "learned_source_value_rank_reward": 1.0,
                        "learned_source_value_rank_penalty": 1.0,
                        "min_cost": 0.0,
                    }
                }
            )
            scored = model.score_reactions(
                parent_mol="P",
                reactant_lists=[["A"], ["B"]],
                base_scores=[0.8, 0.8],
                base_costs=[1.0, 1.0],
                templates=["chem-template", "enzyme-template"],
                raw_result={
                    "model_full_name": [
                        "graphfp_models.USPTO-full_remapped",
                        "onmt_models.bionav_one_step",
                    ],
                    "source_policy_decision": [
                        {
                            "source_value_scores": {
                                "graphfp_models.USPTO-full_remapped": 0.2,
                                "onmt_models.bionav_one_step": 0.9,
                            }
                        },
                        {
                            "source_value_scores": {
                                "graphfp_models.USPTO-full_remapped": 0.2,
                                "onmt_models.bionav_one_step": 0.9,
                            }
                        },
                    ],
                },
                context={"enabled": True},
            )

        self.assertGreater(scored.total_costs[0], scored.total_costs[1])
        self.assertGreater(scored.annotations[0]["components"]["source_value"], 0.0)
        self.assertLess(scored.annotations[1]["components"]["source_value"], 0.0)

    def test_cascade_cost_hook_uses_learned_action_value_score(self):
        vendor_root = Path("vendor/ChemEnzyRetroPlanner")

        class StaticActionScorer:
            def score(self, *, reactants, **_kwargs):
                return 0.9 if reactants == ["B"] else 0.1

        with _vendor_pythonpath(vendor_root):
            from retro_planner.search_frame.mcts_star.cascade_cost import RuleCascadeCostModel

            model = RuleCascadeCostModel(
                {
                    "weights": {
                        "learned_action_value_score_reward": 1.0,
                        "min_cost": 0.0,
                    }
                }
            )
            model.action_value_scorer = StaticActionScorer()
            scored = model.score_reactions(
                parent_mol="P",
                reactant_lists=[["A"], ["B"]],
                base_scores=[0.8, 0.8],
                base_costs=[1.0, 1.0],
                templates=["chem-template", "enzyme-template"],
                raw_result={
                    "model_full_name": [
                        "graphfp_models.USPTO-full_remapped",
                        "onmt_models.bionav_one_step",
                    ],
                },
                context={"enabled": True, "node_depth": 0},
            )

        self.assertGreater(scored.total_costs[0], scored.total_costs[1])
        self.assertEqual(scored.annotations[1]["action_value_score"], 0.9)
        self.assertLess(scored.annotations[1]["components"]["action_value"], 0.0)

    def test_cascade_cost_feature_vector_uses_process_context_schema_v2(self):
        import torch

        from cascade_planner.eval.train_cascade_action_value import CascadeActionValueNetwork

        vendor_root = Path("vendor/ChemEnzyRetroPlanner")
        schema = {
            "schema_version": "cascade_action_value_features.v2",
            "categorical_fields": [
                "route_domain",
                "source_model",
                "reaction_domain",
                "adjacent_reaction_domain",
            ],
            "categories": {
                "route_domain": ["chemoenzymatic"],
                "source_model": ["onmt_models.bionav_one_step"],
                "reaction_domain": ["enzymatic"],
                "adjacent_reaction_domain": ["chemical"],
            },
            "numeric_fields": [
                "parent_depth",
                "candidate_index",
                "base_score",
                "base_cost",
                "cascade_adjustment",
                "rule_total_cost",
                "reactant_count",
                "parent_heavy_atoms",
                "parent_hetero_atoms",
                "parent_ring_count",
                "parent_mol_wt",
                "reactant_total_heavy_atoms",
                "reactant_total_hetero_atoms",
                "reactant_total_ring_count",
                "reactant_total_mol_wt",
                "reactant_max_mol_wt",
                "heavy_atom_balance",
                "preferred_domain_match",
                "failure_enzyme_evidence_weak",
                "source_policy_score_ratio",
                "component_domain_preference",
            ],
            "n_bits": 8,
            "feature_dim": 41,
        }
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "action_value.pt"
            model = CascadeActionValueNetwork(input_dim=41, hidden=16)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_schema": schema,
                    "hidden": 16,
                },
                checkpoint,
            )
            with _vendor_pythonpath(vendor_root):
                from retro_planner.search_frame.mcts_star.cascade_cost import TorchActionValueScorer

                scorer = TorchActionValueScorer(str(checkpoint))
                score = scorer.score(
                    parent_mol="CCO",
                    reactants=["CC", "O"],
                    candidate_index=0,
                    source_model="onmt_models.bionav_one_step",
                    reaction_domain="enzymatic",
                    base_score=0.8,
                    base_cost=1.0,
                    cascade_adjustment=0.0,
                    rule_total_cost=1.0,
                    components={"domain_preference": 0.0},
                    context={
                        "route_domain": "chemoenzymatic",
                        "node_depth": 1,
                        "adjacent_reaction_domain": "chemical",
                        "preferred_reaction_domains": ["enzymatic"],
                        "active_failure_modes": ["EnzymeEvidenceWeak"],
                        "source_policy_decision": {
                            "topk_by_model": {"onmt_models.bionav_one_step": 40},
                            "source_value_scores": {"onmt_models.bionav_one_step": 0.9},
                        },
                    },
                )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_moltree_propagates_node_local_cascade_context(self):
        vendor_root = Path("vendor/ChemEnzyRetroPlanner")
        calls = []

        def expand(mol, parent_depth=None, cascade_node_context=None):
            calls.append(
                {
                    "mol": mol,
                    "parent_depth": parent_depth,
                    "context": dict(cascade_node_context or {}),
                }
            )
            if mol == "P":
                return {
                    "reactants": ["A"],
                    "scores": [0.9],
                    "template": ["chem-template"],
                    "model_full_name": ["onmt_models.bionav_one_step"],
                }
            if mol == "A":
                return {
                    "reactants": ["S"],
                    "scores": [0.9],
                    "template": ["enzyme-template"],
                    "model_full_name": ["onmt_models.bionav_one_step"],
                }
            return None

        with _vendor_pythonpath(vendor_root):
            from retro_planner.search_frame.mcts_star.cascade_cost import RuleCascadeCostModel
            from retro_planner.search_frame.mcts_star.molmcts_star import mol_planner

            mol_planner(
                target_mol="P",
                target_mol_id=0,
                starting_mols={"S"},
                expand_fn=expand,
                iterations=2,
                max_depth=3,
                exclude_target=False,
                cascade_cost_model=RuleCascadeCostModel({"weights": {"min_cost": 0.0}}),
                cascade_search_context={"enabled": True, "route_domain": "chemoenzymatic"},
            )

        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0]["mol"], "P")
        self.assertEqual(calls[0]["context"]["route_domain"], "chemoenzymatic")
        self.assertNotIn("adjacent_reaction_domain", calls[0]["context"])
        self.assertEqual(calls[1]["mol"], "A")
        self.assertEqual(calls[1]["context"]["adjacent_reaction_domain"], "enzymatic")

    def test_cascade_source_policy_can_use_learned_source_value_checkpoint(self):
        import torch

        from cascade_planner.eval.train_cascade_source_value import SourceValueNetwork

        vendor_root = Path("vendor/ChemEnzyRetroPlanner")
        schema = {
            "categorical_fields": [
                "route_domain",
                "source_model",
                "reaction_domain",
                "adjacent_reaction_domain",
            ],
            "categories": {
                "route_domain": ["all_enzymatic"],
                "source_model": ["graphfp_models.USPTO-full_remapped", "onmt_models.bionav_one_step"],
                "reaction_domain": ["chemical", "enzymatic"],
                "adjacent_reaction_domain": ["chemical", "enzymatic", "unknown"],
            },
            "numeric_fields": [
                "parent_depth",
                "parent_heavy_atoms",
                "parent_hetero_atoms",
                "parent_ring_count",
                "parent_mol_wt",
            ],
            "feature_dim": 13,
        }
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "source_value.pt"
            model = SourceValueNetwork(input_dim=13, hidden=16)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_schema": schema,
                    "hidden": 16,
                },
                checkpoint,
            )
            with _vendor_pythonpath(vendor_root):
                from retro_planner.search_frame.mcts_star.cascade_source_policy import RuleCascadeSourcePolicy

                decision = RuleCascadeSourcePolicy(
                    {
                        "enabled": True,
                        "rebalance_topk": False,
                        "source_value_model_path": str(checkpoint),
                    }
                ).decide(
                    parent_mol="CCO",
                    available_models=[
                        "graphfp_models.USPTO-full_remapped",
                        "onmt_models.bionav_one_step",
                    ],
                    expansion_topk=50,
                    context={
                        "route_domain": "all_enzymatic",
                        "adjacent_reaction_domain": "chemical",
                    },
                    parent_depth=0,
                )

        self.assertIn("learned_source_value_topk", decision.reasons)
        self.assertEqual(set(decision.source_value_scores), {
            "graphfp_models.USPTO-full_remapped",
            "onmt_models.bionav_one_step",
        })
        self.assertTrue(all(value > 0 for value in decision.topk_by_model.values()))

    def test_cascade_source_policy_can_rank_learned_source_topk(self):
        vendor_root = Path("vendor/ChemEnzyRetroPlanner")

        class StaticScorer:
            def score(self, *, source_model, **_kwargs):
                if source_model == "graphfp_models.USPTO-full_remapped":
                    return 0.9
                return 0.1

        with _vendor_pythonpath(vendor_root):
            from retro_planner.search_frame.mcts_star.cascade_source_policy import RuleCascadeSourcePolicy

            policy = RuleCascadeSourcePolicy(
                {
                    "enabled": True,
                    "rebalance_topk": True,
                    "learned_topk_mode": "rank",
                    "learned_rule_combine": "learned_first",
                    "learned_min_topk_fraction": 0.2,
                    "min_unpreferred_topk": 5,
                    "active_failure_topk_multiplier": 1.0,
                }
            )
            policy.source_value_scorer = StaticScorer()
            decision = policy.decide(
                parent_mol="CCO",
                available_models=[
                    "graphfp_models.USPTO-full_remapped",
                    "onmt_models.bionav_one_step",
                ],
                expansion_topk=50,
                context={"preferred_reaction_domains": ["enzymatic"]},
                parent_depth=0,
            )

        self.assertEqual(decision.topk_by_model["graphfp_models.USPTO-full_remapped"], 50)
        self.assertEqual(decision.topk_by_model["onmt_models.bionav_one_step"], 10)
        self.assertIn("learned_source_value_topk", decision.reasons)
        self.assertIn("rebalance_topk_by_preferred_domain", decision.reasons)

    def test_row_route_domain_can_condition_cascade_cost_context(self):
        flags = _chem_enzy_search_flags_for_row(
            {"route_domain": "all_enzymatic"},
            {
                "gpu": -1,
                "use_cascade_cost_model": True,
                "cascade_search_context": {"enabled": True},
                "cascade_cost_model": {"enabled": True},
            },
            context_from_row=True,
        )

        context = flags["cascade_search_context"]
        self.assertEqual(context["context_source"], "route_domain")
        self.assertEqual(context["context_policy"], "safe")
        self.assertEqual(context["preferred_reaction_domains"], ["enzymatic"])
        self.assertEqual(context["active_failure_modes"], ["EnzymeEvidenceWeak"])

    def test_safe_row_context_does_not_penalize_bionav_for_chemical_targets(self):
        flags = _chem_enzy_search_flags_for_row(
            {"route_domain": "all_chemical"},
            {
                "gpu": -1,
                "use_cascade_cost_model": True,
                "cascade_search_context": {"enabled": True},
                "cascade_cost_model": {"enabled": True},
            },
            context_from_row=True,
        )

        context = flags["cascade_search_context"]
        self.assertEqual(context["preferred_reaction_domains"], [])
        self.assertFalse(context["penalize_unpreferred_domain"])

    def test_strict_row_context_keeps_hard_ablation_policy(self):
        flags = _chem_enzy_search_flags_for_row(
            {"route_domain": "all_chemical"},
            {
                "gpu": -1,
                "use_cascade_cost_model": True,
                "cascade_search_context": {"enabled": True},
                "cascade_cost_model": {"enabled": True},
            },
            context_from_row=True,
            context_policy="strict",
        )

        context = flags["cascade_search_context"]
        self.assertEqual(context["preferred_reaction_domains"], ["chemical"])
        self.assertTrue(context["penalize_unpreferred_domain"])

    def test_benchmark_runner_validates_model_paths_before_search(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing = root / "source_value.pt"
            existing.write_text("placeholder", encoding="utf-8")
            missing = root / "missing_transition.pt"

            with self.assertRaisesRegex(FileNotFoundError, "cascade_transition_model"):
                _validate_model_inputs(
                    cascade_value_model_path=None,
                    cascade_transition_model_path=missing,
                    cascade_pair_scorer_path=None,
                    chem_enzy_cascade_cost_model=None,
                    chem_enzy_cascade_source_policy=None,
                )

            with self.assertRaisesRegex(FileNotFoundError, "action_value_model_path"):
                _validate_model_inputs(
                    cascade_value_model_path=None,
                    cascade_transition_model_path=None,
                    cascade_pair_scorer_path=None,
                    chem_enzy_cascade_cost_model={
                        "enabled": True,
                        "action_value_model_path": str(missing),
                    },
                    chem_enzy_cascade_source_policy={
                        "enabled": True,
                        "source_value_model_path": str(existing),
                    },
                )

    def test_benchmark_runner_validates_stock_names_before_search(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "config.yaml"
            config.write_text(
                json.dumps({"stocks": {"Zinc_Fix-stock": "stock.csv"}}),
                encoding="utf-8",
            )

            _validate_stock_inputs(config, ["Zinc_Fix-stock"])
            with self.assertRaisesRegex(ValueError, "selected stock names not found"):
                _validate_stock_inputs(config, ["stocklite"])

    def test_benchmark_runner_passes_pair_reward_mode_to_search_config(self):
        captured = {}

        class FakeAdapter:
            def __init__(self, vendor_root, gpu=-1, enable_condition_prediction=False, enable_enzyme_assignment=False):
                self.config_path = Path(vendor_root) / "config.yaml"

            def run_targets(self, configs, dry_run=False, reuse_planner=True):
                return [
                    BaselineRunResult(
                        target_smiles=config.target_smiles,
                        backend=BACKEND_NAME,
                    )
                    for config in configs
                ]

        def fake_run_one_target(row, chem_result, *, cascade_config, **kwargs):
            captured["pair_reward_mode"] = cascade_config.pair_reward_mode
            captured["pair_reward_tie_epsilon"] = cascade_config.pair_reward_tie_epsilon
            return (
                {
                    "target_smiles": row["target_smiles"],
                    "chem_enzy": {"solved": False, "failures": []},
                    "cascade_search": {
                        "solved": False,
                        "stock_closed": False,
                        "cofactor_closed": False,
                        "condition_conflict_free": True,
                        "enzyme_evidence_sufficient": True,
                        "failure_categories": [],
                        "elapsed_s": 0.0,
                        "stage_count": 0,
                        "n_results": 0,
                    },
                    "recovery": {},
                },
                [],
            )

        original_adapter = benchmark_runner.ChemEnzyBackendAdapter
        original_run_one = benchmark_runner._run_one_target
        try:
            benchmark_runner.ChemEnzyBackendAdapter = FakeAdapter
            benchmark_runner._run_one_target = fake_run_one_target
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                benchmark = root / "bench.json"
                benchmark.write_text(json.dumps([{"target_smiles": "CCO"}]), encoding="utf-8")
                vendor = root / "vendor"
                vendor.mkdir()
                (vendor / "config.yaml").write_text(json.dumps({"stocks": {"stock": "stock.csv"}}), encoding="utf-8")
                output = root / "out.json"

                payload = benchmark_runner.run_cascade_search_benchmark(
                    benchmark_path=benchmark,
                    output_path=output,
                    vendor_root=vendor,
                    stock_names=["stock"],
                    one_step_models=["graphfp"],
                    cascade_pair_reward_weight=0.1,
                    cascade_pair_reward_mode="guarded_tie_break",
                    cascade_pair_reward_tie_epsilon=0.03,
                )
        finally:
            benchmark_runner.ChemEnzyBackendAdapter = original_adapter
            benchmark_runner._run_one_target = original_run_one

        self.assertEqual(captured["pair_reward_mode"], "guarded_tie_break")
        self.assertEqual(captured["pair_reward_tie_epsilon"], 0.03)
        self.assertEqual(payload["metadata"]["cascade_search"]["pair_reward_mode"], "guarded_tie_break")
        self.assertEqual(payload["metadata"]["cascade_search"]["pair_reward_tie_epsilon"], 0.03)

    def test_expansion_trace_rows_can_be_exposed_as_cascade_proposals(self):
        result = BaselineRunResult(
            target_smiles="P",
            backend=BACKEND_NAME,
            routes=[
                RouteCandidate(
                    target_smiles="P",
                    solved=True,
                    steps=[
                        RouteStepCandidate(
                            product_smiles="P",
                            reactant_smiles=["A"],
                            rxn_smiles="A>>P",
                            source_model="route_pool",
                            score=0.9,
                        )
                    ],
                )
            ],
            raw_backend_metadata={
                "cascade_expansion_trace": {
                    "rows": [
                        {
                            "parent_mol": "P",
                            "parent_depth": 0,
                            "candidate_index": 2,
                            "reactants": ["GT"],
                            "source_model": "onmt_models.bionav_one_step",
                            "reaction_domain": "enzymatic",
                            "base_score": 0.42,
                            "base_cost": 0.87,
                            "cascade_adjustment": -0.2,
                            "total_cost": 0.67,
                            "components": {"action_value": -0.2},
                        }
                    ]
                }
            },
        )

        route_pool = _proposal_cache_from_chem_enzy(result)
        expansion_pool = _expansion_proposal_cache_from_chem_enzy(result)
        merged = _merge_proposal_caches(route_pool, expansion_pool)
        proposal_rxns, proposal_reactants = _proposal_pool_keys(merged)

        self.assertIn("A>>P", proposal_rxns)
        self.assertIn("GT>>P", proposal_rxns)
        self.assertIn("GT", proposal_reactants)
        action = expansion_pool["P"][0]
        self.assertEqual(action.step.source_model, "onmt_models.bionav_one_step")
        self.assertEqual(action.step.reaction_type, "enzymatic")
        self.assertEqual(action.step.raw_metadata["source"], "chem_enzy_expansion_trace")
        self.assertEqual(action.step.raw_metadata["cascade_cost"]["candidate_index"], 2)

    def test_expansion_trace_proposals_are_ranked_and_limited_per_leaf(self):
        result = BaselineRunResult(
            target_smiles="P",
            backend=BACKEND_NAME,
            raw_backend_metadata={
                "cascade_expansion_trace": {
                    "rows": [
                        {
                            "parent_mol": "P",
                            "candidate_index": 3,
                            "reactants": ["slow"],
                            "source_model": "graphfp",
                            "base_score": 0.9,
                            "total_cost": 4.0,
                        },
                        {
                            "parent_mol": "P",
                            "candidate_index": 8,
                            "reactants": ["fast"],
                            "source_model": "graphfp",
                            "base_score": 0.1,
                            "total_cost": 0.1,
                        },
                        {
                            "parent_mol": "P",
                            "candidate_index": 1,
                            "reactants": ["middle"],
                            "source_model": "graphfp",
                            "base_score": 0.8,
                            "total_cost": 1.0,
                        },
                    ]
                }
            },
        )

        pool = _expansion_proposal_cache_from_chem_enzy(result, topk_per_leaf=2)
        reactants = [action.step.reactant_smiles[0] for action in pool["P"]]
        merged = _sort_proposal_cache(_merge_proposal_caches({
            "P": [
                RouteStepCandidate(
                    product_smiles="P",
                    reactant_smiles=["route"],
                    rxn_smiles="route>>P",
                    score=0.99,
                )
            ]
        }, pool))

        self.assertEqual(reactants, ["fast", "middle"])
        self.assertEqual(merged["P"][0].step.reactant_smiles, ["fast"])


class CascadeGoldSmokeBuilderTest(unittest.TestCase):
    def test_builds_gold_smoke_and_excludes_training_overlap(self):
        dataset = [
            _record(
                doi="10.test/overlap",
                target="CCO",
                rxn="CC>>CCO",
                cascade_id="cascade_1",
            ),
            _record(
                doi="10.test/clean",
                target="CCCC",
                rxn="CCC>>CCCC",
                cascade_id="cascade_2",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_path = root / "dataset.json"
            out = root / "smoke.json"
            pack = root / "pack"
            pack.mkdir()
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            (pack / "step_pairs.jsonl").write_text(
                json.dumps({"target_smiles": "CCO", "reaction_smiles": "CC>>CCO"}) + "\n",
                encoding="utf-8",
            )

            rows = build_cascade_gold_smoke(
                dataset_path=dataset_path,
                output_path=out,
                limit=1,
                exclude_pack_paths=[pack],
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_smiles"], "CCCC")
        self.assertEqual(rows[0]["gt_route"][0]["condition"]["temperature_c"], 30.0)
        self.assertEqual(rows[0]["gt_route"][0]["ec_number"], "1.1.1.1")


class ChemEnzyComparisonTest(unittest.TestCase):
    def test_filters_route_tree_rows_to_benchmark_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "bench.json"
            chem = root / "chem.json"
            route_tree = root / "route_tree.json"
            benchmark.write_text(json.dumps([
                {"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC>>CCO"}]},
            ]), encoding="utf-8")
            chem.write_text(json.dumps({
                "targets": [{"target_smiles": "CCO", "solved": True, "routes": [], "failures": []}],
            }), encoding="utf-8")
            route_tree.write_text(json.dumps({
                "targets": [
                    {"target_smiles": "CCO", "metrics": {"plan": True}, "planner_output": {"routes": []}},
                    {"target_smiles": "CCCC", "metrics": {"plan": True}, "planner_output": {"routes": []}},
                ],
            }), encoding="utf-8")

            report = compare_baselines(benchmark_path=benchmark, chem_enzy_path=chem, route_tree_path=route_tree)

        self.assertEqual(report["route_tree"]["n_targets"], 1)
        self.assertEqual(report["route_tree"]["solved_rate"], 1.0)

    def test_serializes_cascade_result_program_outcomes(self):
        step = SimpleNamespace(
            rxn_smiles="CC.O>>CCO",
            reactant_smiles=["CC", "O"],
        )
        state = SimpleNamespace(step_annotations=[step])
        result = SimpleNamespace(
            state=state,
            solved=True,
            score=0.7,
            failures=[],
        )

        programs = _cascade_result_programs(
            [result],
            gt_rxns={"CC.O>>CCO"},
            gt_reactants_set={"CC", "O"},
        )

        self.assertEqual(programs[0]["rank"], 1)
        self.assertTrue(programs[0]["exact_gt_route_recovered"])
        self.assertEqual(programs[0]["route_outcome_value"], 1.0)
        self.assertIn("CC.O>>CCO", programs[0]["route_rxns"])

    def test_route_block_value_final_reranker_can_promote_generated_result(self):
        class FakeFinalReranker:
            def rerank(self, results, *, search_elapsed_s=None):
                reordered = [results[1], results[0]]
                return reordered, [
                    {"original_rank": 2, "new_rank": 1, "route_block_value_score": 3.0, "model_pickle": "fake.pkl"},
                    {"original_rank": 1, "new_rank": 2, "route_block_value_score": 1.0, "model_pickle": "fake.pkl"},
                ]

        chem_result = BaselineRunResult(
            target_smiles="P",
            backend=BACKEND_NAME,
            routes=[
                RouteCandidate(
                    target_smiles="P",
                    solved=True,
                    steps=[
                        RouteStepCandidate(
                            product_smiles="P",
                            reactant_smiles=["A"],
                            rxn_smiles="A>>P",
                            source_model="route_pool",
                            score=0.9,
                            stock_status={"A": True},
                        ),
                        RouteStepCandidate(
                            product_smiles="P",
                            reactant_smiles=["GT"],
                            rxn_smiles="GT>>P",
                            source_model="route_pool",
                            score=0.8,
                            stock_status={"GT": True},
                        ),
                    ],
                )
            ],
        )

        payload, _trace = _run_one_target(
            {
                "target_smiles": "P",
                "gt_route": [{"rxn_smiles": "GT>>P"}],
                "starting_materials": [{"smiles": "A"}, {"smiles": "GT"}],
            },
            chem_result,
            cascade_config=CascadeSearchConfig(branch_factor=2, expansion_budget=4),
            include_route_outcomes=True,
            cascade_result_limit=2,
            route_block_value_final_reranker=FakeFinalReranker(),
        )

        programs = payload["cascade_search"]["result_programs"]
        self.assertEqual(programs[0]["route_rxns"], ["GT>>P"])
        self.assertEqual(programs[0]["original_rank"], 2)
        self.assertTrue(payload["cascade_search"]["route_block_value_final_rerank"]["changed_top_route"])
        self.assertTrue(payload["recovery"]["exact_reaction_in_route_pool"])

    def test_product_audit_final_reranker_can_promote_late_stage_route(self):
        target = "CCOC(=O)c1ccccc1"
        chem_result = BaselineRunResult(
            target_smiles=target,
            backend=BACKEND_NAME,
            routes=[
                RouteCandidate(
                    target_smiles=target,
                    solved=True,
                    steps=[
                        RouteStepCandidate(
                            product_smiles=target,
                            reactant_smiles=["CCCOc1ccccc1"],
                            rxn_smiles=f"CCCOc1ccccc1>>{target}",
                            source_model="route_pool",
                            score=0.9,
                            stock_status={"CCCOc1ccccc1": True},
                        ),
                        RouteStepCandidate(
                            product_smiles=target,
                            reactant_smiles=["O=C(O)c1ccccc1", "CCO"],
                            rxn_smiles=f"O=C(O)c1ccccc1.CCO>>{target}",
                            source_model="route_pool",
                            score=0.8,
                            stock_status={"O=C(O)c1ccccc1": True, "CCO": True},
                        ),
                    ],
                )
            ],
        )

        payload, _trace = _run_one_target(
            {
                "target_smiles": target,
                "gt_route": [{"rxn_smiles": f"O=C(O)c1ccccc1.CCO>>{target}"}],
                "starting_materials": [{"smiles": "CCCOc1ccccc1"}, {"smiles": "O=C(O)c1ccccc1"}, {"smiles": "CCO"}],
            },
            chem_result,
            cascade_config=CascadeSearchConfig(branch_factor=2, expansion_budget=4),
            include_route_outcomes=True,
            cascade_result_limit=2,
            product_audit_final_reranker=ProductAuditFinalReranker(),
        )

        programs = payload["cascade_search"]["result_programs"]
        self.assertEqual(programs[0]["route_rxns"], [f"CCO.O=C(O)c1ccccc1>>{target}"])
        self.assertEqual(programs[0]["product_audit_original_rank"], 2)
        self.assertIn("late_stage_derivatization", programs[0]["product_audit_tags"])
        self.assertTrue(payload["cascade_search"]["product_audit_final_rerank"]["changed_top_route"])


class ChemEnzyBroadUnionTest(unittest.TestCase):
    def test_merges_native_and_autoplanner_route_pools(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "bench.json"
            chem = root / "chem.json"
            auto = root / "auto.json"
            benchmark.write_text(
                json.dumps([
                    {
                        "target_smiles": "CCO",
                        "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                    }
                ]),
                encoding="utf-8",
            )
            chem.write_text(
                json.dumps({
                    "targets": [
                        {
                            "target_smiles": "CCO",
                            "routes": [
                                {
                                    "steps": [
                                        {
                                            "product_smiles": "CCO",
                                            "reactant_smiles": ["CC", "O"],
                                            "rxn_smiles": "CC.O>>CCO",
                                            "stock_status": {"CC": True, "O": True},
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                }),
                encoding="utf-8",
            )
            auto.write_text(
                json.dumps({
                    "targets": [
                        {
                            "target_smiles": "CCO",
                            "metrics": {"strict_stock_solve_any": False},
                            "planner_output": {
                                "routes": [
                                    {
                                        "steps": [
                                            {
                                                "reaction_smiles": "CC>>CCO",
                                                "main_reactant": "CC",
                                                "aux_reactants": [],
                                                "reaction_type": "",
                                            }
                                        ]
                                    }
                                ]
                            },
                        }
                    ]
                }),
                encoding="utf-8",
            )

            report = build_union_report(
                benchmark_path=benchmark,
                chem_enzy_path=chem,
                autoplanner_path=auto,
            )

        self.assertEqual(report["summary"]["native"]["stock_rate"], 1.0)
        self.assertEqual(report["summary"]["native"]["n_targets"], 1)
        self.assertFalse(report["targets"][0]["autoplanner_stock"])
        self.assertTrue(report["targets"][0]["union_stock"])
        self.assertGreater(report["summary"]["union"]["exact_reaction_in_route_pool"], 0.0)
        self.assertGreater(report["summary"]["union"]["gt_reactant_in_route_pool"], 0.0)

    def test_preserves_duplicate_benchmark_rows_and_respects_topk_stock_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "bench.json"
            chem = root / "chem.json"
            auto = root / "auto.json"
            benchmark.write_text(
                json.dumps([
                    {"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]},
                    {"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]},
                ]),
                encoding="utf-8",
            )
            chem.write_text(
                json.dumps({
                    "targets": [
                        {
                            "target_smiles": "CCO",
                            "routes": [
                                {
                                    "steps": [
                                        {
                                            "product_smiles": "CCO",
                                            "reactant_smiles": ["BAD"],
                                            "rxn_smiles": "BAD>>CCO",
                                            "stock_status": {"BAD": False},
                                        }
                                    ]
                                },
                                {
                                    "steps": [
                                        {
                                            "product_smiles": "CCO",
                                            "reactant_smiles": ["CC", "O"],
                                            "rxn_smiles": "CC.O>>CCO",
                                            "stock_status": {"CC": True, "O": True},
                                        }
                                    ]
                                },
                            ],
                        }
                    ]
                }),
                encoding="utf-8",
            )
            auto.write_text(
                json.dumps({
                    "targets": [
                        {
                            "index": 0,
                            "target_smiles": "CCO",
                            "metrics": {"strict_stock_solve_any": False},
                            "planner_output": {"routes": []},
                        }
                    ]
                }),
                encoding="utf-8",
            )

            rank_report = build_union_report(
                benchmark_path=benchmark,
                chem_enzy_path=chem,
                autoplanner_path=auto,
                native_topk=1,
                native_selection="rank",
            )
            stock_report = build_union_report(
                benchmark_path=benchmark,
                chem_enzy_path=chem,
                autoplanner_path=auto,
                native_topk=1,
                native_selection="stock_first",
            )
            rank_plus_stock_report = build_union_report(
                benchmark_path=benchmark,
                chem_enzy_path=chem,
                autoplanner_path=auto,
                native_topk=2,
                native_selection="rank_plus_stock",
            )

        self.assertEqual(rank_report["summary"]["union"]["n_targets"], 2)
        self.assertEqual(rank_report["summary"]["union"]["stock_rate"], 0.0)
        self.assertEqual(stock_report["summary"]["union"]["stock_rate"], 1.0)
        self.assertEqual(rank_plus_stock_report["summary"]["union"]["stock_rate"], 1.0)
        self.assertEqual(rank_plus_stock_report["summary"]["union"]["avg_route_count"], 2.0)

    def test_can_write_synthetic_reservoir_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "bench.json"
            chem = root / "chem.json"
            auto = root / "auto.json"
            synthetic = root / "synthetic.json"
            benchmark.write_text(
                json.dumps([{"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]}]),
                encoding="utf-8",
            )
            chem.write_text(
                json.dumps({
                    "targets": [
                        {
                            "target_smiles": "CCO",
                            "routes": [
                                {
                                    "steps": [
                                        {
                                            "product_smiles": "CCO",
                                            "reactant_smiles": ["CC", "O"],
                                            "rxn_smiles": "CC.O>>CCO",
                                            "stock_status": {"CC": True, "O": True},
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                }),
                encoding="utf-8",
            )
            auto.write_text(
                json.dumps({
                    "metadata": {"search_mode": "route_tree"},
                    "targets": [
                        {
                            "index": 0,
                            "target_smiles": "CCO",
                            "metrics": {"strict_stock_solve_any": False},
                            "planner_output": {"routes": []},
                        }
                    ],
                }),
                encoding="utf-8",
            )

            report = build_union_report(
                benchmark_path=benchmark,
                chem_enzy_path=chem,
                autoplanner_path=auto,
                native_topk=1,
                synthesize_output=synthetic,
            )

            payload = json.loads(synthetic.read_text(encoding="utf-8"))

        self.assertEqual(report["synthetic_output"], str(synthetic))
        self.assertEqual(payload["summary"]["plan_rate"], 1.0)
        self.assertEqual(payload["summary"]["strict_stock_solve_any"], 1.0)
        self.assertEqual(payload["summary"]["exact_reaction_in_route_pool"], 1.0)
        self.assertEqual(payload["summary"]["gt_reactant_in_route_pool"], 1.0)
        self.assertEqual(payload["summary"]["avg_route_count"], 1)
        self.assertEqual(payload["summary"]["synthesized_union"]["stock_rate"], 1.0)
        self.assertEqual(payload["targets"][0]["metrics"]["broad_reservoir_route_count"], 1)
        self.assertEqual(payload["targets"][0]["planner_output"]["routes"][0]["broad_reservoir"]["source"], "native_chem_enzy")


class ChemEnzyExpansionTraceAnalysisTest(unittest.TestCase):
    def test_analyzes_internal_trace_hits_by_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "bench.json"
            trace = root / "trace.jsonl"
            out = root / "analysis.json"
            benchmark.write_text(
                json.dumps([
                    {
                        "target_smiles": "CCO",
                        "route_domain": "all_chemical",
                        "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                    }
                ]),
                encoding="utf-8",
            )
            trace.write_text(
                "\n".join([
                    json.dumps({
                        "target_smiles": "CCO",
                        "parent_mol": "CCO",
                        "reactants": ["CC", "O"],
                        "source_model": "graphfp",
                        "reaction_domain": "chemical",
                        "cascade_adjustment": -0.1,
                    }),
                    json.dumps({
                        "target_smiles": "CCO",
                        "parent_mol": "CCO",
                        "reactants": ["C", "CO"],
                        "source_model": "bionav",
                        "reaction_domain": "enzymatic",
                        "cascade_adjustment": 0.2,
                    }),
                ])
                + "\n",
                encoding="utf-8",
            )

            report = analyze_expansion_trace(trace_path=trace, benchmark_path=benchmark, output_path=out)

        self.assertEqual(report["summary"]["candidate_rows"], 2)
        self.assertEqual(report["summary"]["exact_gt_hits"], 1)
        self.assertEqual(report["by_source"]["graphfp"]["exact_gt_hits"], 1)
        self.assertEqual(report["adjustment"]["mean_hit_adjustment"], -0.1)


def _record(*, doi: str, target: str, rxn: str, cascade_id: str) -> dict:
    return {
        "doi": doi,
        "title": "test record",
        "record_uuid": doi,
        "cascades": [
            {
                "cascade_id": cascade_id,
                "cascade_uuid": cascade_id,
                "route_domain": "all_chemical",
                "operation_mode": "sequential_isolated",
                "representation_type": "small_molecule",
                "is_demonstrated_success": True,
                "target_products": [{"smiles": target, "name": "target"}],
                "starting_materials": [{"smiles": "CC", "name": "start"}],
                "global_conditions": {},
                "overall_outcome": {},
                "catalyst_combination_summary": "test catalyst",
                "quality_control": {
                    "record_tier": "gold",
                    "allowed_tasks": ["retrosynthesis"],
                    "review_flags": [],
                    "excluded_from_training": False,
                },
                "steps": [
                    {
                        "step_id": f"{cascade_id}_s1",
                        "step_index": 1,
                        "step_type": "productive",
                        "rxn_smiles": rxn,
                        "rxn_smiles_status": "ok",
                        "transformation_name": "test",
                        "transformation_superclass": "test",
                        "step_role": "productive_transformation",
                        "input_species": [{"smiles": rxn.split(">>", 1)[0]}],
                        "output_species": [{"smiles": target}],
                        "step_conditions": {"temperature_c": 30.0, "ph": 7.0, "solvent": "water"},
                        "step_outcome": {"step_yield_percent": 75.0},
                        "catalyst_components": [
                            {"component_name": "enzyme", "ec_number": "1.1.1.1", "uniprot_id": "P00001"}
                        ],
                        "evidence_quote": "test evidence",
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
