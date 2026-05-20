import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

import cascade_planner.route_tree.cascade_oracle as cascade_oracle_module
from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.cascadeboard.route_export import route_metrics
from cascade_planner.cascadeboard.skeleton_planner import RouteSkeleton
from cascade_planner.eval.build_cascade_oracle_pack import build_cascade_oracle_pack
from cascade_planner.route_tree.cascade_oracle import CascadeOracleRuntime, build_cascade_oracle_payload_from_native
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.runtime import RouteTreeEvaluation
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.route_tree.search import NeuralGuidedAOSearch, plan_with_route_tree
from cascade_planner.route_tree.source_gate import LearnedSourceGate, SourceGate, _SOURCE_GATE_CACHE, _SourceGateMLP, source_group
from cascade_planner.route_tree.trace import RouteTreeTraceCollector
from cascade_planner.route_tree.verifier import RouteVerifier


class _RouteTreeRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCCCCCCC":
            return [
                {
                    "main_reactant": "CCCCCCCC",
                    "rxn_smiles": "CCCCCCCC>>CCCCCCCC",
                    "type": "identity",
                    "score": 99.0,
                    "source": "bad_loop",
                },
                {
                    "main_reactant": "CCCC",
                    "aux_reactants": ["CCCC"],
                    "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                    "type": "coupling",
                    "score": 0.5,
                    "source": "fake_chem",
                },
            ]
        return []


class _TypedRetro:
    def predict(self, product_smiles: str, top_k: int = 10, ec_token: str = "", skel_type: str = ""):
        return [
            {
                "main_reactant": "CCO",
                "rxn_smiles": f"CCO>>{product_smiles}",
                "type": skel_type or "reduction",
                "ec": f"{ec_token}.x" if ec_token else "",
                "score": 1.0,
                "source": "typed",
            }
        ]


class _FakeChemEnzyOneStep:
    def run(self, target: str, topk: int = 10):
        return {
            "reactants": ["CC.O", "NCC"],
            "scores": [0.9, 0.4],
            "template": ["graph_template", ""],
            "costs": [0.1, 0.9],
            "model_full_name": ["graphfp_models.USPTO-full_remapped", "onmt_models.bionav_one_step"],
            "weight": [1.0, 1.0],
        }


class _NoMetadataRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCCCCCCC":
            return [
                {
                    "main_reactant": "CCCCCC",
                    "rxn_smiles": "CCCCCC>>CCCCCCCC",
                    "score": 0.8,
                    "source": "retrochimera",
                }
            ]
        if product_smiles == "CCCCCC":
            return [
                {
                    "main_reactant": "CC",
                    "rxn_smiles": "CC>>CCCCCC",
                    "score": 0.8,
                    "source": "retrochimera",
                }
            ]
        return []


class _MismatchRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        return [
            {
                "main_reactant": "CCCC",
                "rxn_smiles": f"CCCC>>{product_smiles}",
                "type": "oxidation",
                "score": 1.0,
                "source": "typed",
            }
        ]


class _RecordingRetro:
    def __init__(self):
        self.top_k_values = []

    def predict(self, product_smiles: str, top_k: int = 10):
        self.top_k_values.append(top_k)
        return []


class _ProductRecordingRetro:
    def __init__(self):
        self.products = []

    def predict(self, product_smiles: str, top_k: int = 10):
        self.products.append(product_smiles)
        return []


class _TopKDependentStockRescueRetro:
    def __init__(self):
        self.top_k_values = []

    def predict(self, product_smiles: str, top_k: int = 10):
        self.top_k_values.append(top_k)
        rows = [
            {
                "main_reactant": "CCCCCCC",
                "rxn_smiles": f"CCCCCCC>>{product_smiles}",
                "score": 0.2,
                "source": "retrochimera",
            }
        ]
        if top_k >= 4:
            rows.append(
                {
                    "main_reactant": "CCCC",
                    "aux_reactants": ["CCCC"],
                    "rxn_smiles": f"CCCC.CCCC>>{product_smiles}",
                    "score": 0.9,
                    "source": "retrochimera",
                }
            )
        return rows[:top_k]


class _NoStockGainRetryRetro:
    def __init__(self):
        self.top_k_values = []

    def predict(self, product_smiles: str, top_k: int = 10):
        self.top_k_values.append(top_k)
        rows = [
            {
                "main_reactant": "CCCCCCC",
                "rxn_smiles": f"CCCCCCC>>{product_smiles}",
                "score": 0.2,
                "source": "retrochimera",
            }
        ]
        if top_k >= 4:
            rows.append(
                {
                    "main_reactant": "CCCCCCCCCC",
                    "rxn_smiles": f"CCCCCCCCCC>>{product_smiles}",
                    "score": 0.3,
                    "source": "retrochimera",
                }
            )
        return rows[:top_k]


class _MultiSolvedRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles != "CCCCCCCC":
            return []
        return [
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.9,
                "source": "retrochimera",
            },
            {
                "main_reactant": "CCCO",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCO.CCCC>>CCCCCCCC",
                "score": 0.8,
                "source": "retrochimera",
            },
            {
                "main_reactant": "CCN",
                "aux_reactants": ["CCCCC"],
                "rxn_smiles": "CCN.CCCCC>>CCCCCCCC",
                "score": 0.7,
                "source": "retrochimera",
            },
        ][:top_k]


class _SolvedAndDeadEndRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCCCCCCC":
            return [
                {
                    "main_reactant": "CCCC",
                    "aux_reactants": ["CCCC"],
                    "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                    "score": 0.9,
                    "source": "retrochimera",
                },
                {
                    "main_reactant": "CCCCCCC",
                    "rxn_smiles": "CCCCCCC>>CCCCCCCC",
                    "score": 0.8,
                    "source": "retrochimera",
                },
            ][:top_k]
        return []


class _SourceRecorder:
    def __init__(self, source: str, rows: list[dict] | None = None):
        self.source = source
        self.calls = 0
        self.top_k_values = []
        self.rows = rows or []

    def predict(self, product_smiles: str, top_k: int = 10, **_kwargs):
        self.calls += 1
        self.top_k_values.append(top_k)
        return [dict(row, source=self.source) for row in self.rows[:top_k]]


class _CountingController:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, leaf, actions, *, stock_checker=None):
        self.calls += 1
        return RouteTreeEvaluation(
            action_scores=[1.0 for _ in actions],
            route_value=0.5,
            solved_prob=0.25,
            stock_closed_prob=0.25,
            progressive_prob=0.25,
            compatibility_prob=0.25,
            model_active=True,
            reason="test_controller",
        )


class _ReverseProposalRankers:
    def request_k(self, source, top_k):
        return top_k + 1

    def rerank(self, product, source, candidates, *, limit, stock_checker=None):
        out = []
        for rank, row in enumerate(reversed(candidates), start=1):
            item = dict(row)
            item["rank"] = rank
            item["proposal_ranker_rank"] = rank
            out.append(item)
        return out[:limit]


def _native_route_payload():
    return {
        "targets": [
            {
                "target_smiles": "CCCCCCCC",
                "routes": [
                    {
                        "steps": [
                            {
                                "product_smiles": "CCCCCCCC",
                                "reactant_smiles": ["CCCC", "CCCC"],
                                "reaction_smiles": "CCCC.CCCC>>CCCCCCCC",
                                "stock_status": {"CCCC": True},
                                "source_model": "graphfp_models.USPTO-full_remapped",
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _trace_row_for_native_action():
    return {
        "target_smiles": "CCCCCCCC",
        "benchmark_index": 0,
        "event": {
            "state_id": "state-0",
            "depth": 0,
            "expanded_leaf": "CCCCCCCC",
            "open_leaves": ["CCCCCCCC"],
            "candidate_actions": [
                {
                    "main_reactant": "CCCC",
                    "aux_reactants": ["CCCC"],
                    "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                    "source": "retrochimera",
                    "score": 0.8,
                    "reactant_stock_fraction": 1.0,
                }
            ],
            "outcome": {"max_depth": 1},
        },
    }


class RouteTreePlannerTest(unittest.TestCase):
    def test_candidate_action_normalizes_reactants_from_reaction_smiles(self):
        action = CandidateAction.from_candidate(
            "CCN",
            {
                "main_reactant": "CCO",
                "rxn_smiles": "CCO.N>>CCN",
                "score": 1.0,
            },
        )

        self.assertEqual(action.main_reactant, "CCO")
        self.assertEqual(set(action.reactants), {"CCO", "N"})
        self.assertIn("N", action.to_candidate_dict()["aux_reactants"])

    def test_route_tree_state_has_stable_canonical_id(self):
        action = CandidateAction.from_candidate("CCO", {"main_reactant": "CC", "rxn_smiles": "CC>>CCO"})
        a = RouteTreeState.initial("CCO").advance(
            leaf="CCO",
            action=action,
            next_open_leaves=(),
            score_delta=1.0,
        )
        b = RouteTreeState.initial("CCO").advance(
            leaf="CCO",
            action=action,
            next_open_leaves=(),
            score_delta=2.0,
        )

        self.assertEqual(a.canonical_id, b.canonical_id)

    def test_route_tree_evaluation_cache_reuses_identical_state_action_pool(self):
        controller = _CountingController()
        planner = NeuralGuidedAOSearch(retro_engine={}, controller=controller)
        state = RouteTreeState.initial("CCO")
        action = CandidateAction.from_candidate("CCO", {"main_reactant": "CC", "rxn_smiles": "CC>>CCO"})

        first = planner._evaluate_actions(state, "CCO", [action])
        second = planner._evaluate_actions(state, "CCO", [action])

        self.assertIs(first, second)
        self.assertEqual(controller.calls, 1)
        self.assertEqual(planner.stats.model_calls, 1)
        self.assertEqual(planner.stats.model_active_calls, 1)
        self.assertEqual(planner.stats.evaluation_cache_hits, 1)

    def test_cascade_oracle_payload_matches_native_route_without_exact_gt_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            native_path = tmp_path / "native.json"
            oracle_path = tmp_path / "oracle.json"
            native_path.write_text(json.dumps(_native_route_payload()), encoding="utf-8")

            payload = build_cascade_oracle_payload_from_native(
                native_payload_path=native_path,
                output_path=oracle_path,
                topk=5,
                selection="rank_plus_stock",
            )
            runtime = CascadeOracleRuntime(oracle_path)
            action = CandidateAction.from_candidate(
                "CCCCCCCC",
                {
                    "main_reactant": "CCCC",
                    "aux_reactants": ["CCCC"],
                    "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                    "source": "retrochimera",
                },
            )

            match = runtime.action_value(target="CCCCCCCC", leaf="CCCCCCCC", action=action)

        self.assertEqual(payload["schema_version"], "cascade_oracle_payload.v1")
        self.assertTrue(payload["targets"][0]["routes"][0]["components"]["stock_closed"])
        self.assertEqual(payload["targets"][0]["routes"][0]["components"]["cost_model"], "reaction_cost_and_or.v1")
        self.assertIn("oracle_cost", payload["targets"][0]["routes"][0])
        self.assertIsNotNone(match)
        self.assertEqual(match.reason, "reaction_match")
        self.assertTrue(match.stock_closed)
        self.assertGreater(match.value, 0.0)

    def test_cascade_oracle_pack_uses_cascade_rubric_not_exact_or_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            native_path = tmp_path / "native.json"
            trace_path = tmp_path / "trace.jsonl"
            pack_path = tmp_path / "pack.jsonl"
            oracle_path = tmp_path / "oracle.json"
            report_path = tmp_path / "report.json"
            native_path.write_text(json.dumps(_native_route_payload()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_row_for_native_action()) + "\n", encoding="utf-8")

            report = build_cascade_oracle_pack(
                trace_path=trace_path,
                native_payload_path=native_path,
                output_pack=pack_path,
                output_payload=oracle_path,
                report_path=report_path,
                split="train",
                topk=5,
                selection="rank_plus_stock",
            )
            row = json.loads(pack_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertFalse(report["label_policy"]["uses_exact_or_gt"])
        self.assertFalse(row["teacher_exact_hit"])
        self.assertFalse(row["teacher_gt_reactant_hit"])
        self.assertTrue(row["oracle_match"])
        self.assertGreater(row["teacher_action_value"], 0.5)
        self.assertFalse(row["eval_only"])

    def test_cascade_oracle_pack_refuses_full100_train_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            native_path = tmp_path / "native.json"
            trace_path = tmp_path / "benchmark_v2_100.jsonl"
            native_path.write_text(json.dumps(_native_route_payload()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_row_for_native_action()) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                build_cascade_oracle_pack(
                    trace_path=trace_path,
                    native_payload_path=native_path,
                    output_pack=tmp_path / "pack.jsonl",
                    output_payload=tmp_path / "oracle.json",
                    report_path=tmp_path / "report.json",
                    split="train",
                    topk=5,
                    selection="rank_plus_stock",
                )

    def test_cascade_oracle_soft_boost_is_env_gated(self):
        env_keys = [
            "AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE",
            "AUTOPLANNER_CASCADE_ORACLE_PAYLOAD",
            "AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT",
        ]
        old_env = {key: os.environ.get(key) for key in env_keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                native_path = tmp_path / "native.json"
                oracle_path = tmp_path / "oracle.json"
                native_path.write_text(json.dumps(_native_route_payload()), encoding="utf-8")
                build_cascade_oracle_payload_from_native(
                    native_payload_path=native_path,
                    output_path=oracle_path,
                    topk=5,
                    selection="rank_plus_stock",
                )
                os.environ["AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE"] = "1"
                os.environ["AUTOPLANNER_CASCADE_ORACLE_PAYLOAD"] = str(oracle_path)
                os.environ["AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT"] = "0"
                cascade_oracle_module._RUNTIME_CACHE.clear()
                base_planner = NeuralGuidedAOSearch(
                    retro_engine={},
                    stock_checker=lambda smi: smi == "CCCC",
                    controller=None,
                )
                state = RouteTreeState.initial("CCCCCCCC")
                action = CandidateAction.from_candidate(
                    "CCCCCCCC",
                    {
                        "main_reactant": "CCCC",
                        "aux_reactants": ["CCCC"],
                        "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                        "source": "retrochimera",
                        "score": 0.2,
                    },
                )
                eval_result = RouteTreeEvaluation(action_scores=[])
                base = base_planner._score_delta(state, "CCCCCCCC", action, 0.0, eval_result, next_open=())

                os.environ["AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT"] = "2"
                cascade_oracle_module._RUNTIME_CACHE.clear()
                boosted_planner = NeuralGuidedAOSearch(
                    retro_engine={},
                    stock_checker=lambda smi: smi == "CCCC",
                    controller=None,
                )
                boosted = boosted_planner._score_delta(state, "CCCCCCCC", action, 0.0, eval_result, next_open=())
                boosted_components = boosted_planner._score_delta_components(
                    state,
                    "CCCCCCCC",
                    action,
                    0.0,
                    eval_result,
                    next_open=(),
                )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            cascade_oracle_module._RUNTIME_CACHE.clear()

        self.assertGreater(boosted, base)
        self.assertEqual(boosted_components["cost_model"], "reaction_cost_and_or.v1")
        self.assertGreater(boosted_components["oracle_probability"], boosted_components["proposal_probability"])

    def test_route_tree_transposition_prunes_lower_scored_same_open_leaf_state(self):
        planner = NeuralGuidedAOSearch(retro_engine={}, controller=None)
        action = CandidateAction.from_candidate("CCO", {"main_reactant": "CC", "rxn_smiles": "CC>>CCO"})
        high = RouteTreeState.initial("CCO").advance(
            leaf="CCO",
            action=action,
            next_open_leaves=("CC",),
            score_delta=2.0,
        )
        low = RouteTreeState.initial("CCO").advance(
            leaf="CCO",
            action=action,
            next_open_leaves=("CC",),
            score_delta=1.0,
        )

        self.assertTrue(planner._should_queue_state(high))
        self.assertFalse(planner._should_queue_state(low))
        self.assertEqual(planner.stats.pruned_transposition, 1)

    def test_route_tree_adaptive_budget_does_not_expand_unconstrained_large_leaf(self):
        retro = _RecordingRetro()
        planner = NeuralGuidedAOSearch(
            retro_engine={"retrochimera": retro},
            branch_factor=4,
            controller=None,
        )

        planner._expand_state(RouteTreeState.initial("CCCCCCCCCCCCCCCCCCCCCCCC"))

        self.assertEqual(retro.top_k_values, [4])

    def test_route_tree_reuses_proposal_cache_for_same_leaf_context(self):
        retro = _RecordingRetro()
        planner = NeuralGuidedAOSearch(
            retro_engine={"retrochimera": retro},
            branch_factor=4,
            controller=None,
        )
        context = planner._proposal_context(RouteTreeState.initial("CCCCCCCC"))

        planner._propose_actions("CCCCCCCC", context, top_k=4)
        planner._propose_actions("CCCCCCCC", context, top_k=4)

        self.assertEqual(retro.top_k_values, [4])
        self.assertEqual(planner.stats.proposal_calls, 1)
        self.assertEqual(planner.stats.proposal_cache_hits, 1)

    def test_route_tree_v4_expands_only_highest_remaining_cost_open_leaf_by_default(self):
        retro = _ProductRecordingRetro()
        planner = NeuralGuidedAOSearch(
            retro_engine={"retrochimera": retro},
            branch_factor=4,
            controller=None,
        )
        state = RouteTreeState(
            target="CCCCCCCCCC",
            open_leaves=("CCCCCCCC", "CCCCCCC"),
        )

        planner._expand_state(state)

        self.assertEqual(retro.products, ["CCCCCCCC"])
        self.assertEqual(planner.stats.expanded_leaf_count, 1)
        self.assertEqual(planner.stats.skipped_leaf_count, 1)

    def test_late_stock_rescue_retry_is_env_gated(self):
        old_rescue = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE")
        old_retrieval = os.environ.get("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL")
        try:
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE", None)
            os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = "0"
            baseline_retro = _TopKDependentStockRescueRetro()
            baseline = NeuralGuidedAOSearch(
                retro_engine={"retrochimera": baseline_retro},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=1,
                branch_factor=4,
                controller=None,
            )
            baseline_children = baseline._expand_state(RouteTreeState.initial("CCCCCCCC"))

            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] = "1"
            rescue_retro = _TopKDependentStockRescueRetro()
            rescue = NeuralGuidedAOSearch(
                retro_engine={"retrochimera": rescue_retro},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=1,
                branch_factor=4,
                controller=None,
            )
            rescue_children = rescue._expand_state(RouteTreeState.initial("CCCCCCCC"))
        finally:
            if old_rescue is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] = old_rescue
            if old_retrieval is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = old_retrieval

        self.assertEqual(baseline_retro.top_k_values, [2])
        self.assertFalse(any(not child.open_leaves for child in baseline_children))
        self.assertEqual(rescue_retro.top_k_values, [2, 4])
        self.assertEqual(rescue.stats.stock_rescue_retries, 1)
        self.assertTrue(any(not child.open_leaves for child in rescue_children))

    def test_empty_actions_retry_broadens_proposal_budget(self):
        class _EmptyActionRetryRetro:
            def __init__(self):
                self.top_k_values = []

            def predict(self, product_smiles: str, top_k: int = 10):
                self.top_k_values.append(top_k)
                if len(self.top_k_values) == 1:
                    return []
                return [
                    {
                        "main_reactant": "CC",
                        "rxn_smiles": f"CC>>{product_smiles}",
                        "score": 1.0,
                        "source": "retrochimera",
                    }
                ][:top_k]

        old_retry = os.environ.get("AUTOPLANNER_ROUTE_TREE_EMPTY_ACTION_RETRY")
        old_retrieval = os.environ.get("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_EMPTY_ACTION_RETRY"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = "0"
            retro = _EmptyActionRetryRetro()
            planner = NeuralGuidedAOSearch(
                retro_engine={"retrochimera": retro},
                max_depth=2,
                branch_factor=1,
                expansion_budget=2,
                controller=None,
            )
            children = planner._expand_state(RouteTreeState.initial("CCCCCCCC"))
        finally:
            if old_retry is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_EMPTY_ACTION_RETRY", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_EMPTY_ACTION_RETRY"] = old_retry
            if old_retrieval is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = old_retrieval

        self.assertEqual(retro.top_k_values[:2], [1, 2])
        self.assertTrue(children)
        self.assertEqual(children[0].steps[-1].action.main_reactant, "CC")

    def test_late_stock_rescue_retry_respects_max_retry_cap(self):
        old_rescue = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE")
        old_cap = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MAX_RETRIES")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MAX_RETRIES"] = "1"
            planner = NeuralGuidedAOSearch(
                retro_engine={},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=1,
                branch_factor=4,
                controller=None,
            )
            state = RouteTreeState.initial("CCCCCCCC")
            context = planner._proposal_context(state)
            planner.stats.stock_rescue_retries = 1
            fallback_budget = planner._fallback_proposal_budget_for_leaf(
                state,
                "CCCCCCCC",
                context,
                base_budget=2,
            )
        finally:
            if old_rescue is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] = old_rescue
            if old_cap is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MAX_RETRIES", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MAX_RETRIES"] = old_cap

        self.assertEqual(fallback_budget, 2)

    def test_late_stock_rescue_quality_gate_rejects_no_stock_gain(self):
        old_rescue = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE")
        old_quality = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REQUIRE_STOCK_GAIN")
        old_retrieval = os.environ.get("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REQUIRE_STOCK_GAIN"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = "0"
            retro = _NoStockGainRetryRetro()
            planner = NeuralGuidedAOSearch(
                retro_engine={"retrochimera": retro},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=1,
                branch_factor=4,
                controller=None,
            )
            children = planner._expand_state(RouteTreeState.initial("CCCCCCCC"))
        finally:
            if old_rescue is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] = old_rescue
            if old_quality is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REQUIRE_STOCK_GAIN", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REQUIRE_STOCK_GAIN"] = old_quality
            if old_retrieval is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = old_retrieval

        self.assertEqual(retro.top_k_values, [2, 4])
        self.assertEqual(planner.stats.stock_rescue_retries, 1)
        self.assertEqual(planner.stats.stock_rescue_rejected, 1)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].steps[-1].action.main_reactant, "CCCCCCC")

    def test_route_tree_v4_ignores_uncalibrated_value_heads_in_action_score(self):
        planner = NeuralGuidedAOSearch(retro_engine={}, controller=None)
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {"main_reactant": "CCCCCC", "rxn_smiles": "CCCCCC>>CCCCCCCC", "score": 0.0},
        )
        state = RouteTreeState.initial("CCCCCCCC")
        base = RouteTreeEvaluation(action_scores=[1.0], value_calibrated=False)
        uncalibrated = RouteTreeEvaluation(
            action_scores=[1.0],
            route_value=0.9,
            solved_prob=0.9,
            stock_closed_prob=0.9,
            progressive_prob=0.9,
            value_calibrated=False,
        )
        calibrated = RouteTreeEvaluation(
            action_scores=[1.0],
            route_value=0.9,
            solved_prob=0.9,
            stock_closed_prob=0.9,
            progressive_prob=0.9,
            value_calibrated=True,
        )
        old = os.environ.get("AUTOPLANNER_ROUTE_TREE_USE_UNCALIBRATED_VALUE_HEADS")
        try:
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_USE_UNCALIBRATED_VALUE_HEADS", None)
            base_score = planner._score_delta(state, "CCCCCCCC", action, 1.0, base, next_open=("CCCCCC",))
            uncalibrated_score = planner._score_delta(state, "CCCCCCCC", action, 1.0, uncalibrated, next_open=("CCCCCC",))
            calibrated_score = planner._score_delta(state, "CCCCCCCC", action, 1.0, calibrated, next_open=("CCCCCC",))
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_USE_UNCALIBRATED_VALUE_HEADS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_USE_UNCALIBRATED_VALUE_HEADS"] = old

        self.assertAlmostEqual(base_score, uncalibrated_score)
        self.assertGreater(calibrated_score, uncalibrated_score)

    def test_action_cost_treats_stock_closure_as_zero_child_value(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            stock_checker=lambda smi: smi in {"CCCC"},
        )
        state = RouteTreeState.initial("CCCCCCCC")
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.0,
            },
        )
        eval_result = RouteTreeEvaluation(action_scores=[1.0])
        stock_closed_score = planner._score_delta(state, "CCCCCCCC", action, 1.0, eval_result, next_open=())
        open_leaf_score = planner._score_delta(state, "CCCCCCCC", action, 1.0, eval_result, next_open=("CCCCCC",))

        self.assertGreater(stock_closed_score, open_leaf_score)

    def test_action_cost_prefers_lower_open_leaf_cost(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            stock_checker=lambda smi: False,
        )
        state = RouteTreeState.initial("CCCCCCCC")
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.0,
            },
        )
        eval_result = RouteTreeEvaluation(action_scores=[1.0])
        base_score = planner._score_delta(state, "CCCCCCCC", action, 1.0, eval_result, next_open=("CCCCCCC",))
        frontier_score = planner._score_delta(state, "CCCCCCCC", action, 1.0, eval_result, next_open=())

        self.assertGreater(frontier_score, base_score)

    def test_action_delta_penalizes_anti_progress_actions(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            stock_checker=lambda smi: False,
        )
        state = RouteTreeState.initial("CCCC")
        progressive = CandidateAction.from_candidate(
            "CCCC",
            {"main_reactant": "CC", "rxn_smiles": "CC>>CCCC", "score": 0.0},
        )
        anti_progress = CandidateAction.from_candidate(
            "CCCC",
            {"main_reactant": "CCCCC", "rxn_smiles": "CCCCC>>CCCC", "score": 0.0},
        )
        eval_result = RouteTreeEvaluation(action_scores=[1.0])

        progressive_score = planner._score_delta(state, "CCCC", progressive, 1.0, eval_result, next_open=("CC",))
        anti_score = planner._score_delta(state, "CCCC", anti_progress, 1.0, eval_result, next_open=("CCCCC",))

        self.assertGreater(progressive_score, anti_score)

    def test_strict_stock_mode_does_not_reward_small_nonstock_terminals(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            stock_checker=lambda smi: smi == "CCCC",
        )

        self.assertTrue(planner._is_stock_or_small_terminal("CCCC"))
        self.assertFalse(planner._is_stock_or_small_terminal("O=O"))

    def test_action_delta_penalizes_nonstock_small_reactants(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            stock_checker=lambda smi: smi == "CCCC",
        )
        state = RouteTreeState.initial("CCCCCCCC")
        stock_action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.0,
            },
        )
        nonstock_small_action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "O=O",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "O=O.CCCC>>CCCCCCCC",
                "score": 0.0,
            },
        )
        eval_result = RouteTreeEvaluation(action_scores=[1.0])

        stock_score = planner._score_delta(state, "CCCCCCCC", stock_action, 1.0, eval_result, next_open=())
        nonstock_score = planner._score_delta(
            state,
            "CCCCCCCC",
            nonstock_small_action,
            1.0,
            eval_result,
            next_open=("O=O",),
        )

        self.assertGreater(stock_score, nonstock_score)
        self.assertEqual(planner._nonstock_small_reactant_count(nonstock_small_action), 1)

    def test_root_branch_retention_keeps_chem_enzy_source_diversity(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            branch_factor=2,
        )
        state = RouteTreeState.initial("CCCCCCCC")
        actions = [
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCCCC", "rxn_smiles": "CCCCCCC>>CCCCCCCC", "source": "retrochimera", "score": 3.0},
            ),
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCCC", "rxn_smiles": "CCCCCC>>CCCCCCCC", "source": "enzyformer", "score": 2.0},
            ),
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCC", "rxn_smiles": "CCCCC>>CCCCCCCC", "source": "chem_enzy_graphfp", "score": -1.0},
            ),
        ]
        scored = [
            (
                float(len(actions) - idx),
                state.advance(
                    leaf="CCCCCCCC",
                    action=action,
                    next_open_leaves=(action.main_reactant,),
                    score_delta=float(len(actions) - idx),
                ),
                "CCCCCCCC",
                action,
                actions,
                RouteTreeEvaluation(action_scores=[]),
            )
            for idx, action in enumerate(actions)
        ]

        selected = planner._select_scored_children(scored, state=state)

        self.assertEqual(len(selected), 2)
        self.assertIn("chem_enzy_graphfp", {item[3].source for item in selected})

    def test_source_reserve_can_reject_low_scoring_outlier_source(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            branch_factor=2,
        )
        state = RouteTreeState.initial("CCCCCCCC")
        actions = [
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCCCC", "rxn_smiles": "CCCCCCC>>CCCCCCCC", "source": "retrochimera", "score": 3.0},
            ),
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCCC", "rxn_smiles": "CCCCCC>>CCCCCCCC", "source": "enzyformer", "score": 2.0},
            ),
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCCCCCCC", "rxn_smiles": "CCCCCCCCCC>>CCCCCCCC", "source": "chem_enzy_graphfp", "score": -10.0},
            ),
        ]
        scored = [
            (
                score,
                state.advance(
                    leaf="CCCCCCCC",
                    action=action,
                    next_open_leaves=(action.main_reactant,),
                    score_delta=score,
                ),
                "CCCCCCCC",
                action,
                actions,
                RouteTreeEvaluation(action_scores=[]),
            )
            for score, action in zip([3.0, 2.0, -10.0], actions)
        ]
        old_gap = os.environ.get("AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MAX_SCORE_GAP")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MAX_SCORE_GAP"] = "2.0"
            selected = planner._select_scored_children(scored, state=state)
        finally:
            if old_gap is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MAX_SCORE_GAP", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MAX_SCORE_GAP"] = old_gap

        self.assertEqual(len(selected), 2)
        self.assertNotIn("chem_enzy_graphfp", {item[3].source for item in selected})

    def test_source_diverse_branch_bonus_expands_root_width_when_enabled(self):
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            branch_factor=2,
        )
        state = RouteTreeState.initial("CCCCCCCC")
        actions = [
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCCCC", "rxn_smiles": "CCCCCCC>>CCCCCCCC", "source": "retrochimera", "score": 3.0},
            ),
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCCC", "rxn_smiles": "CCCCCC>>CCCCCCCC", "source": "enzyformer", "score": 2.0},
            ),
            CandidateAction.from_candidate(
                "CCCCCCCC",
                {"main_reactant": "CCCCC", "rxn_smiles": "CCCCC>>CCCCCCCC", "source": "chem_enzy_graphfp", "score": 1.0},
            ),
        ]
        scored = [
            (
                float(len(actions) - idx),
                state.advance(
                    leaf="CCCCCCCC",
                    action=action,
                    next_open_leaves=(action.main_reactant,),
                    score_delta=float(len(actions) - idx),
                ),
                "CCCCCCCC",
                action,
                actions,
                RouteTreeEvaluation(action_scores=[]),
            )
            for idx, action in enumerate(actions)
        ]
        old_bonus = os.environ.get("AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS")
        old_cap = os.environ.get("AUTOPLANNER_ROUTE_TREE_BRANCH_FACTOR_CAP")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_BRANCH_FACTOR_CAP"] = "4"
            selected = planner._select_scored_children(scored, state=state)
        finally:
            if old_bonus is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS"] = old_bonus
            if old_cap is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_BRANCH_FACTOR_CAP", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_BRANCH_FACTOR_CAP"] = old_cap

        self.assertEqual(len(selected), 3)

    def test_route_tree_prunes_explicit_skeleton_type_mismatch(self):
        skeleton = RouteSkeleton(
            n_steps=1,
            types=["reduction"],
            ec1s=[0],
            Ts=[],
            pHs=[],
        )
        planner = NeuralGuidedAOSearch(
            retro_engine={"retrochimera": _MismatchRetro()},
            branch_factor=4,
            skeletons=[skeleton],
            controller=None,
        )

        children = planner._expand_state(RouteTreeState.initial("CCCCCCCC"))

        self.assertEqual(children, [])
        self.assertEqual(planner.stats.pruned_contract, 1)
        self.assertEqual(planner.stats.pruned_invalid, 0)

    def test_route_tree_reads_skeleton_context_in_reverse_order_by_default(self):
        skeleton = RouteSkeleton(
            n_steps=3,
            types=["oxidation", "reduction", "amination"],
            ec1s=[1, 1, 3],
            Ts=[25.0, 30.0, 37.0],
            pHs=[6.5, 7.0, 8.0],
        )
        planner = NeuralGuidedAOSearch(
            retro_engine={"retrochimera": _RecordingRetro()},
            branch_factor=4,
            skeletons=[skeleton],
            controller=None,
        )
        old = os.environ.get("AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT")
        try:
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT", None)
            context = planner._proposal_context(RouteTreeState.initial("CCCCCCCC"))
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT"] = old

        self.assertEqual(context.reaction_type, "amination")
        self.assertEqual(context.ec1, 3)
        self.assertEqual(context.T, 37.0)
        self.assertEqual(context.pH, 8.0)
        self.assertEqual(context.route_metadata["skeleton_context_index"], 2)
        self.assertTrue(context.route_metadata["skeleton_context_reversed"])

    def test_route_tree_can_opt_out_of_reverse_skeleton_context(self):
        skeleton = RouteSkeleton(
            n_steps=3,
            types=["oxidation", "reduction", "amination"],
            ec1s=[1, 1, 3],
            Ts=[25.0, 30.0, 37.0],
            pHs=[6.5, 7.0, 8.0],
        )
        planner = NeuralGuidedAOSearch(
            retro_engine={"retrochimera": _RecordingRetro()},
            branch_factor=4,
            skeletons=[skeleton],
            controller=None,
        )
        old = os.environ.get("AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT")
        os.environ["AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT"] = "0"
        try:
            context = planner._proposal_context(RouteTreeState.initial("CCCCCCCC"))
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT"] = old

        self.assertEqual(context.reaction_type, "oxidation")
        self.assertEqual(context.ec1, 1)
        self.assertEqual(context.T, 25.0)
        self.assertEqual(context.pH, 6.5)
        self.assertEqual(context.route_metadata["skeleton_context_index"], 0)
        self.assertFalse(context.route_metadata["skeleton_context_reversed"])

    def test_retro_engine_proposal_tool_passes_context_without_ranking_route(self):
        tool = RetroEngineProposalTool({"enzyformer": _TypedRetro()})
        actions = tool.propose(
            "CC=O",
            ProposalContext(ec1=1, reaction_type="reduction"),
            top_k=3,
        )

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].ec, "1.x")
        self.assertEqual(actions[0].reaction_type, "reduction")

    def test_chem_enzy_onestep_provider_normalizes_graphfp_and_onmt_rows(self):
        provider = ChemEnzyOneStepProposalProvider(one_step=_FakeChemEnzyOneStep())

        rows = provider.predict("CCO", top_k=2)

        self.assertEqual([row["source"] for row in rows], ["chem_enzy_graphfp", "chem_enzy_onmt"])
        self.assertEqual(rows[0]["main_reactant"], "CC")
        self.assertEqual(rows[0]["aux_reactants"], ["O"])
        self.assertEqual(rows[0]["rxn_smiles"], "CC.O>>CCO")
        self.assertEqual(rows[0]["template"], "graph_template")
        self.assertEqual(rows[0]["type"], "template")
        self.assertEqual(rows[0]["proposal_type"], "chem_enzy_one_step")
        self.assertTrue(rows[0]["teacher_one_step"])
        self.assertEqual(source_group(rows[0]["source"]), "chemical")

    def test_chem_enzy_onestep_provider_reads_trained_checkpoint_env(self):
        key = "AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH"
        old = os.environ.get(key)
        try:
            os.environ[key] = "/tmp/plain_continue_lr1e4_step_300.pt"
            provider = ChemEnzyOneStepProposalProvider.from_env()
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

        self.assertEqual(provider.onmt_model_path, "/tmp/plain_continue_lr1e4_step_300.pt")

    def test_retro_engine_proposal_tool_can_query_chem_enzy_onestep_source(self):
        provider = ChemEnzyOneStepProposalProvider(one_step=_FakeChemEnzyOneStep())
        tool = RetroEngineProposalTool(
            {"chem_enzy_onestep": provider},
            source_order=("chem_enzy_onestep",),
        )

        actions = tool.propose("CCO", ProposalContext(), top_k=2)

        self.assertEqual(len(actions), 2)
        self.assertEqual({action.source for action in actions}, {"chem_enzy_graphfp", "chem_enzy_onmt"})
        self.assertEqual(actions[0].rxn_smiles, "CC.O>>CCO")
        diagnostics = tool.last_diagnostics["sources"]["chem_enzy_onestep"]
        self.assertEqual(diagnostics["raw_returned"], 2)
        self.assertEqual(diagnostics["kept_returned"], 2)

    def test_retro_engine_proposal_tool_can_query_v3_retrieval_source(self):
        tool = RetroEngineProposalTool({"v3_retrieval": _TypedRetro()})
        actions = tool.propose(
            "CC=O",
            ProposalContext(ec1=1, reaction_type="reduction"),
            top_k=3,
        )

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].ec, "1.x")
        self.assertEqual(actions[0].reaction_type, "reduction")

    def test_retro_engine_proposal_tool_uses_source_specific_ranker_only_within_source(self):
        retro = _SourceRecorder("retrochimera", rows=[
            {"main_reactant": "CC", "rxn_smiles": "CC>>CCCC", "score": 0.9},
            {"main_reactant": "CCC", "rxn_smiles": "CCC>>CCCC", "score": 0.1},
        ])
        tool = RetroEngineProposalTool(
            {"retrochimera": retro},
            proposal_rankers=_ReverseProposalRankers(),
        )

        actions = tool.propose("CCCC", ProposalContext(), top_k=1)

        self.assertEqual(retro.top_k_values, [2])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].main_reactant, "CCC")
        self.assertEqual(actions[0].metadata["proposal_ranker_rank"], 1)

    def test_retro_engine_proposal_tool_skips_chemical_sources_for_ec_context(self):
        product = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        enzymatic = _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": f"CCO>>{product}",
            "ec": "1.1.1.1",
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": f"CC>>{product}",
        }])
        tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})

        actions = tool.propose(product, ProposalContext(ec1=1, reaction_type="reduction"), top_k=2)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "enzyformer")
        self.assertEqual(enzymatic.calls, 1)
        self.assertEqual(chemical.calls, 0)

    def test_retro_engine_proposal_tool_applies_v4_contextual_enzymatic_floors(self):
        rows = [
            {"main_reactant": "CCO", "rxn_smiles": "CCO>>CC=O", "ec": "1.1.1.1"}
            for _ in range(10)
        ]
        v3 = _SourceRecorder("v3_retrieval", rows=rows)
        enzyformer = _SourceRecorder("enzyformer", rows=rows)
        enzexpand = _SourceRecorder("enzexpand", rows=rows)
        tool = RetroEngineProposalTool(
            {
                "v3_retrieval": v3,
                "enzyformer": enzyformer,
                "enzexpand": enzexpand,
            }
        )

        tool.propose("CC=O", ProposalContext(ec1=1, reaction_type="reduction"), top_k=8)

        self.assertGreaterEqual(v3.top_k_values[0], 3)
        self.assertGreaterEqual(enzyformer.top_k_values[0], 2)
        self.assertGreaterEqual(enzexpand.top_k_values[0], 1)
        self.assertIn("v3_retrieval", tool.last_diagnostics["sources"])

    def test_retro_engine_proposal_tool_falls_back_when_ec_sources_are_empty(self):
        product = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        enzymatic = _SourceRecorder("enzyformer", rows=[])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": f"CC>>{product}",
        }])
        tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})

        actions = tool.propose(product, ProposalContext(ec1=1, reaction_type="reduction"), top_k=2)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "retrochimera")
        self.assertEqual(enzymatic.calls, 1)
        self.assertEqual(chemical.calls, 1)

    def test_source_gate_preserves_evidence_and_allocates_enzymatic_budget(self):
        product = "O=C1OC(O)C(O)C(O)C1O"
        gate = SourceGate()
        allocation = gate.allocate(
            product,
            context=ProposalContext(ec1=1, reaction_type="reduction"),
            available_sources=["retrochimera", "enzyformer", "retrorules"],
            total_budget=6,
        )

        self.assertEqual(allocation.source_budgets["retrochimera"], 0)
        self.assertGreater(allocation.source_budgets["enzyformer"], 0)
        self.assertGreater(allocation.source_budgets["retrorules"], 0)
        self.assertEqual(allocation.safety_guard, "carbohydrate_ec_prefers_enzymatic_sources")

        tool = RetroEngineProposalTool({"enzyformer": _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "ec": "1.1.1.1",
            "evidence": {"rhea_ids": ["1"], "doi": "10.1/test"},
        }])})
        actions = tool.propose("CC=O", ProposalContext(ec1=1, reaction_type="reduction"), top_k=2)

        self.assertEqual(actions[0].metadata["evidence"]["doi"], "10.1/test")
        self.assertIn("source_gate", actions[0].metadata)
        self.assertEqual(actions[0].metadata["source_provenance"]["source"], "enzyformer")

    def test_learned_source_gate_loads_checkpoint_and_default_tool_uses_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "source_gate.pt")
            model = _SourceGateMLP(17, n_classes=4)
            with torch.no_grad():
                for param in model.parameters():
                    param.zero_()
                model.net[5].bias[1] = 8.0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "n_bits": 4,
                        "input_dim": 17,
                        "source_budget_groups": ["chemical", "enzymatic", "rhea_retrorules", "fallback"],
                    },
                },
                path,
            )

            gate = LearnedSourceGate(path)
            allocation = gate.allocate(
                "CC=O",
                context=ProposalContext(ec1=1, reaction_type="reduction"),
                available_sources=["retrochimera", "enzyformer"],
                total_budget=4,
            )
            self.assertEqual(allocation.source_budgets["retrochimera"], 0)
            self.assertEqual(allocation.source_budgets["enzyformer"], 4)

            enzymatic = _SourceRecorder("enzyformer", rows=[{
                "main_reactant": "CCO",
                "rxn_smiles": "CCO>>CC=O",
                "ec": "1.1.1.1",
            }])
            chemical = _SourceRecorder("retrochimera", rows=[{
                "main_reactant": "CC",
                "rxn_smiles": "CC>>CC=O",
            }])
            previous_env = os.environ.get("AUTOPLANNER_SOURCE_GATE")
            os.environ["AUTOPLANNER_SOURCE_GATE"] = path
            _SOURCE_GATE_CACHE.clear()
            try:
                tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})
                actions = tool.propose("CC=O", ProposalContext(ec1=1, reaction_type="reduction"), top_k=4)
            finally:
                if previous_env is None:
                    os.environ.pop("AUTOPLANNER_SOURCE_GATE", None)
                else:
                    os.environ["AUTOPLANNER_SOURCE_GATE"] = previous_env
                _SOURCE_GATE_CACHE.clear()

            self.assertEqual([action.source for action in actions], ["enzyformer", "retrochimera"])
            self.assertEqual(enzymatic.calls, 1)
            self.assertEqual(chemical.calls, 1)

    def test_route_verifier_rejects_type_ec_and_condition_mismatch(self):
        verifier = RouteVerifier()
        state = RouteTreeState.initial("CCCCCCCC")
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "rxn_smiles": "CCCC>>CCCCCCCC",
                "type": "oxidation",
                "ec": "2.1.1.1",
                "T": 95.0,
                "pH": 12.0,
            },
        )

        result = verifier.verify_action(
            state=state,
            leaf="CCCCCCCC",
            action=action,
            context=ProposalContext(ec1=1, reaction_type="reduction", T=25.0, pH=7.0),
        )

        self.assertFalse(result.accepted)
        self.assertIn("skeleton_type_mismatch", result.reasons)
        self.assertIn("ec_mismatch", result.reasons)
        self.assertIn("condition_temperature_mismatch", result.reasons)
        self.assertIn("condition_pH_mismatch", result.reasons)

    def test_retro_engine_proposal_tool_keeps_chemical_sources_for_small_ec_context(self):
        enzymatic = _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "ec": "1.1.1.1",
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
        }])
        tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})

        actions = tool.propose("CC=O", ProposalContext(ec1=1, reaction_type="reduction"), top_k=2)

        self.assertEqual({action.source for action in actions}, {"retrochimera", "enzyformer"})
        self.assertEqual(enzymatic.calls, 1)
        self.assertEqual(chemical.calls, 1)

    def test_retro_engine_proposal_tool_skips_chemical_sources_for_carbohydrate_like_ec_context(self):
        product = "O=C1OC(O)C(O)C(O)C1O"
        enzymatic = _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "O=C1OC(O)C(O)C(O)C1O",
            "rxn_smiles": f"O=C1OC(O)C(O)C(O)C1O>>{product}",
            "ec": "1.1.1.1",
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": f"CC>>{product}",
        }])
        tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})

        actions = tool.propose(product, ProposalContext(ec1=1, reaction_type="reduction"), top_k=2)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "enzyformer")
        self.assertEqual(enzymatic.calls, 1)
        self.assertEqual(chemical.calls, 0)

    def test_retro_engine_proposal_tool_keeps_reserve_for_phosphorylated_oxygen_rich_ec_context(self):
        product = "O=C(C(O)CO)C(O)C(O)COP(=O)(O)O"
        enzymatic = _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "O=C(C(O)CO)C(O)C(O)COP(=O)(O)O",
            "rxn_smiles": f"O=C(C(O)CO)C(O)C(O)COP(=O)(O)O>>{product}",
            "ec": "1.1.1.1",
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": f"CC>>{product}",
        }])
        tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})

        actions = tool.propose(product, ProposalContext(ec1=1, reaction_type="reduction"), top_k=2)

        self.assertEqual({action.source for action in actions}, {"enzyformer", "retrochimera"})
        self.assertEqual(enzymatic.calls, 1)
        self.assertEqual(chemical.calls, 1)

    def test_retro_engine_proposal_tool_keeps_small_chemical_reserve_for_enzymatic_route_metadata(self):
        enzymatic = _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "ec": "1.1.1.1",
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
        }])
        tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})

        actions = tool.propose(
            "CC=O",
            ProposalContext(
                ec1=1,
                reaction_type="reduction",
                route_metadata={"enzymatic_only_route": True},
            ),
            top_k=2,
        )

        self.assertEqual({action.source for action in actions}, {"enzyformer", "retrochimera"})
        self.assertEqual(enzymatic.calls, 1)
        self.assertEqual(chemical.calls, 1)

    def test_retro_engine_proposal_tool_skips_reserve_for_carbohydrate_like_route_metadata(self):
        enzymatic = _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "ec": "1.1.1.1",
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
        }])
        tool = RetroEngineProposalTool({"retrochimera": chemical, "enzyformer": enzymatic})

        actions = tool.propose(
            "CC=O",
            ProposalContext(
                ec1=1,
                reaction_type="reduction",
                route_metadata={
                    "enzymatic_only_route": True,
                    "carbohydrate_like_route": True,
                },
            ),
            top_k=2,
        )

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "enzyformer")
        self.assertEqual(enzymatic.calls, 1)
        self.assertEqual(chemical.calls, 0)

    def test_route_tree_search_solves_stock_closed_route_and_prunes_loop(self):
        results = plan_with_route_tree(
            target="CCCCCCCC",
            retro_engine={"retrochimera": _RouteTreeRetro()},
            stock_checker=lambda smi: smi == "CCCC",
            max_depth=2,
            n_results=1,
            branch_factor=4,
            expansion_budget=8,
            controller=None,
        )

        self.assertEqual(len(results), 1)
        board = results[0].board
        self.assertEqual(board.slots[0].main_reactant, "CCCC")
        self.assertEqual(board.slots[0].reaction_smiles, "CCCC.CCCC>>CCCCCCCC")
        metrics = route_metrics(board, stock_checker=lambda smi: smi == "CCCC")
        self.assertTrue(metrics["strict_stock_solve"])
        self.assertEqual(results[0].constraint_report["search_mode"], "route_tree")
        diag = results[0].explanation.uncertainty_table
        self.assertEqual(diag["route_tree_version"], "v4_runtime_controlled_node_action_budget")
        self.assertEqual(diag["route_tree_selected_node_sequence"], ["CCCCCCCC"])
        self.assertTrue(diag["route_tree_selected_action_sequence"])
        self.assertTrue(diag["route_tree_value_trajectory"])
        self.assertTrue(diag["route_tree_bottleneck_trajectory"])
        self.assertTrue(diag["route_tree_source_budgets"])
        self.assertTrue(diag["route_tree_proposal_recall_diagnostics"])
        self.assertIn("proposal_source_stats", diag)
        self.assertIn("route_tree_runtime_bottlenecks", diag)

    def test_route_tree_can_collect_result_pool_larger_than_requested(self):
        old_multiplier = os.environ.get("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER")
        old_keep = os.environ.get("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES")
        old_rerank = os.environ.get("AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER"] = "2"
            os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK"] = "1"
            results = plan_with_route_tree(
                target="CCCCCCCC",
                retro_engine={"retrochimera": _MultiSolvedRetro()},
                stock_checker=lambda smi: smi in {"CCCC", "CCCO", "CCN", "CCCCC"},
                max_depth=2,
                n_results=1,
                branch_factor=4,
                expansion_budget=8,
                controller=None,
            )
        finally:
            if old_multiplier is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER"] = old_multiplier
            if old_keep is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = old_keep
            if old_rerank is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK"] = old_rerank

        self.assertEqual(len(results), 1)
        outcome = results[0].explanation.uncertainty_table["route_tree_final_outcome"]
        self.assertEqual(outcome["requested_results"], 1)
        self.assertEqual(outcome["route_tree_result_pool_target"], 2)
        self.assertGreaterEqual(outcome["solved_routes"], 2)

    def test_route_tree_can_return_contrast_fallbacks_for_route_pool_coverage(self):
        old_multiplier = os.environ.get("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER")
        old_keep = os.environ.get("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES")
        old_contrast = os.environ.get("AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS")
        old_contrast_max = os.environ.get("AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER"] = "2"
            os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX"] = "1"
            results = plan_with_route_tree(
                target="CCCCCCCC",
                retro_engine={"retrochimera": _SolvedAndDeadEndRetro()},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=2,
                n_results=1,
                branch_factor=2,
                expansion_budget=4,
                controller=None,
            )
        finally:
            if old_multiplier is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER"] = old_multiplier
            if old_keep is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = old_keep
            if old_contrast is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS"] = old_contrast
            if old_contrast_max is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX"] = old_contrast_max

        self.assertEqual(len(results), 2)
        self.assertTrue(route_metrics(results[0].board, stock_checker=lambda smi: smi == "CCCC")["strict_stock_solve"])
        self.assertFalse(route_metrics(results[1].board, stock_checker=lambda smi: smi == "CCCC")["strict_stock_solve"])
        self.assertEqual(results[1].board.slots[0].main_reactant, "CCCCCCC")

    def test_route_tree_trace_records_real_expansion_rows(self):
        trace = RouteTreeTraceCollector()
        results = plan_with_route_tree(
            target="CCCCCCCC",
            retro_engine={"retrochimera": _RouteTreeRetro()},
            stock_checker=lambda smi: smi == "CCCC",
            max_depth=2,
            n_results=1,
            branch_factor=4,
            expansion_budget=8,
            controller=None,
            trace_collector=trace,
        )

        self.assertEqual(len(results), 1)
        rows = trace.to_rows()
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["expanded_leaf"], "CCCCCCCC")
        self.assertEqual(rows[0]["state"]["target"], "CCCCCCCC")
        self.assertTrue(rows[0]["candidate_actions"])
        self.assertTrue(rows[0]["selection_scores"])
        self.assertTrue(rows[0]["selection_score_breakdown"])
        self.assertEqual(rows[0]["selection_score_breakdown"][0]["cost_model"], "reaction_cost_and_or.v1")
        self.assertIn("total_cost", rows[0]["selection_score_breakdown"][0])
        self.assertEqual(rows[0]["outcome"]["search_status"], "solved")

    def test_route_tree_v2_returns_skeleton_conditioned_partial_route(self):
        skeleton = RouteSkeleton(
            n_steps=1,
            types=["reduction"],
            ec1s=[0],
            Ts=[35.0],
            pHs=[7.2],
        )
        results = plan_with_route_tree(
            target="CCCCCCCC",
            retro_engine={"retrochimera": _NoMetadataRetro()},
            stock_checker=lambda smi: False,
            max_depth=4,
            n_results=1,
            branch_factor=4,
            expansion_budget=8,
            skeletons=[skeleton],
            controller=None,
        )

        self.assertEqual(len(results), 1)
        board = results[0].board
        self.assertEqual(len(board.slots), 1)
        self.assertEqual(board.slots[0].reaction_type, "reduction")
        self.assertEqual(board.slots[0].T, 35.0)
        self.assertEqual(board.slots[0].pH, 7.2)
        self.assertEqual(
            results[0].explanation.uncertainty_table["route_tree_search_status"],
            "depth_limit",
        )


if __name__ == "__main__":
    unittest.main()
