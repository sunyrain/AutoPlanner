import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

import cascade_planner.route_tree.cascade_oracle as cascade_oracle_module
import cascade_planner.route_tree.proposals as route_tree_proposals
from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.cascadeboard.route_export import route_metrics
from cascade_planner.cascadeboard.skeleton_planner import RouteSkeleton
from cascade_planner.eval.build_cascade_oracle_pack import build_cascade_oracle_pack
from cascade_planner.route_tree.cascade_oracle import CascadeOracleRuntime, build_cascade_oracle_payload_from_native
from cascade_planner.route_tree.proposals import (
    ProposalContext,
    RetroEngineProposalTool,
    _apply_route_source_gates,
    _apply_source_budget_floor,
    _dedupe_actions_with_diagnostics,
)
from cascade_planner.route_tree.runtime import RouteTreeEvaluation
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.route_tree.search import NeuralGuidedAOSearch, plan_with_route_tree
from cascade_planner.route_tree.source_gate import (
    BridgeAwareSourceGate,
    LearnedSourceGate,
    SourceGate,
    SourceAllocation,
    _SOURCE_GATE_CACHE,
    _SourceGateMLP,
    source_group,
)
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


class _RootStockRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CC":
            return [
                {
                    "main_reactant": "C",
                    "aux_reactants": ["C"],
                    "rxn_smiles": "C.C>>CC",
                    "type": "coupling",
                    "score": 1.0,
                    "source": "fake_chem",
                }
            ]
        return []


class _EthaneTerminalRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCO":
            return [
                {
                    "main_reactant": "CC",
                    "rxn_smiles": "CC>>CCO",
                    "type": "terminal_probe",
                    "score": 1.0,
                    "source": "retrochimera",
                }
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


class _AcceptingEnzymeSPVerifier:
    def score_action(self, *, product: str, action: CandidateAction):
        return {
            "schema_version": "enzyme_sp_verifier_v1.runtime_score.v1",
            "accepted": True,
            "score": 0.95,
            "threshold": 0.3,
            "product_smiles": product,
            "substrate_smiles": ".".join(action.reactants),
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


class _LateStockClosingProbeRetro:
    def __init__(self):
        self.top_k_values = []

    def predict(self, product_smiles: str, top_k: int = 10):
        self.top_k_values.append(top_k)
        rows = [
            {
                "main_reactant": "CCCCCCC",
                "rxn_smiles": f"CCCCCCC>>{product_smiles}",
                "score": 0.9,
                "source": "retrochimera",
            },
            {
                "main_reactant": "CCCCCC",
                "rxn_smiles": f"CCCCCC>>{product_smiles}",
                "score": 0.8,
                "source": "retrochimera",
            },
            {
                "main_reactant": "CCCCC",
                "rxn_smiles": f"CCCCC>>{product_smiles}",
                "score": 0.7,
                "source": "retrochimera",
            },
            {
                "main_reactant": "CCCCCCC",
                "aux_reactants": ["C"],
                "rxn_smiles": f"CCCCCCC.C>>{product_smiles}",
                "score": 0.6,
                "source": "retrochimera",
            },
            {
                "main_reactant": "CCCCCC",
                "aux_reactants": ["CC"],
                "rxn_smiles": f"CCCCCC.CC>>{product_smiles}",
                "score": 0.5,
                "source": "retrochimera",
            },
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": f"CCCC.CCCC>>{product_smiles}",
                "score": 0.01,
                "source": "retrochimera",
            },
        ]
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


class _ChemicalFirstBridgeEnzymeRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CC=O":
            return [
                {
                    "main_reactant": "CCO",
                    "rxn_smiles": "CCO>>CC=O",
                    "score": 1.0,
                    "source": "template_relevance",
                    "source_gate": {
                        "policy_reason": "bridge_gate_hits",
                        "molecule_flags": {"bridge_gate_hits": 1},
                    },
                },
                {
                    "main_reactant": "C",
                    "rxn_smiles": "C>>CC=O",
                    "score": 1.0,
                    "source": "chem_enzy_onmt",
                    "source_gate": {
                        "policy_reason": "bridge_gate_hits",
                        "molecule_flags": {"bridge_gate_hits": 1},
                    },
                    "enzyme_sp_verifier_v1": {
                        "accepted": True,
                        "score": 0.91,
                        "threshold": 0.3,
                    },
                },
            ][:top_k]
        return []


class _ChemicalFirstUnsupportedEnzymeRetro(_ChemicalFirstBridgeEnzymeRetro):
    def predict(self, product_smiles: str, top_k: int = 10):
        rows = super().predict(product_smiles, top_k=top_k)
        for row in rows:
            if row.get("source") == "chem_enzy_onmt":
                row = row
                row.pop("source_gate", None)
        return rows


class _ChemicalFirstCostGapBridgeEnzymeRetro(_ChemicalFirstBridgeEnzymeRetro):
    def predict(self, product_smiles: str, top_k: int = 10):
        rows = super().predict(product_smiles, top_k=top_k)
        for row in rows:
            if row.get("source") == "chem_enzy_onmt":
                row["score"] = 0.2
        return rows


class _PlainChemicalFirstBridgeEnzymeRetro(_ChemicalFirstBridgeEnzymeRetro):
    def predict(self, product_smiles: str, top_k: int = 10):
        rows = super().predict(product_smiles, top_k=top_k)
        for row in rows:
            if row.get("source") == "template_relevance":
                row.pop("source_gate", None)
        return rows


class _TimeoutFrontierRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCCCCCCC":
            import time

            time.sleep(0.05)
            return [
                {
                    "main_reactant": "CCCCCCC",
                    "rxn_smiles": "CCCCCCC>>CCCCCCCC",
                    "score": 0.9,
                    "source": "retrochimera",
                }
            ]
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


class _FakeBridgeRetriever:
    def __init__(self, hits):
        self.hits = list(hits)
        self.calls = []

    def retrieve(self, product, **kwargs):
        self.calls.append((product, kwargs))
        return list(self.hits)


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
    def test_compiled_terminal_judge_empty_policy_preserves_default_closure(self):
        results = plan_with_route_tree(
            target="CCO",
            retro_engine={"retrochimera": _EthaneTerminalRetro()},
            stock_checker=lambda smi: smi == "CC",
            max_depth=1,
            branch_factor=2,
            expansion_budget=2,
            n_results=1,
            constraints={},
            controller=None,
            enzyme_sp_verifier=None,
        )

        self.assertTrue(results)
        self.assertEqual(results[0].explanation.uncertainty_table["route_tree_search_status"], "stock_closed")
        self.assertNotIn("route_tree_compiled_judge_trace", results[0].explanation.uncertainty_table)

    def test_compiled_terminal_judge_blacklist_blocks_fake_terminal_closure(self):
        results = plan_with_route_tree(
            target="CCO",
            retro_engine={"retrochimera": _EthaneTerminalRetro()},
            stock_checker=lambda smi: smi == "CC",
            max_depth=1,
            branch_factor=2,
            expansion_budget=2,
            n_results=1,
            constraints={
                "judge_policy": {
                    "policy_id": "judge_blacklist",
                    "case_id": "case",
                    "evidence_refs": ["ev1"],
                    "terminal_blacklist": ["CC"],
                    "anchor_whitelist": [],
                    "stock_tier_rule": "default",
                    "same_scaffold_risk_threshold": 0.85,
                    "material_sanity_mode": "conservative",
                    "schema_version": "judge_policy.v1",
                }
            },
            controller=None,
            enzyme_sp_verifier=None,
        )

        self.assertTrue(results)
        table = results[0].explanation.uncertainty_table
        self.assertNotEqual(table["route_tree_search_status"], "stock_closed")
        self.assertTrue(results[0].board.slots)
        self.assertEqual(results[0].board.slots[0].main_reactant, "CC")
        self.assertEqual(table["route_tree_compiled_judge_trace"][0]["decision"], "reject")
        self.assertEqual(table["route_tree_compiled_judge_trace"][0]["reason"], "terminal_blacklist")

    def test_compiled_terminal_judge_anchor_whitelist_allows_anchor_terminal(self):
        results = plan_with_route_tree(
            target="CCO",
            retro_engine={"retrochimera": _EthaneTerminalRetro()},
            stock_checker=lambda smi: False,
            max_depth=1,
            branch_factor=2,
            expansion_budget=2,
            n_results=1,
            constraints={
                "judge_policy": {
                    "policy_id": "judge_anchor",
                    "case_id": "case",
                    "evidence_refs": ["ev_anchor"],
                    "terminal_blacklist": [],
                    "anchor_whitelist": ["CC"],
                    "stock_tier_rule": "default",
                    "same_scaffold_risk_threshold": 0.85,
                    "material_sanity_mode": "conservative",
                    "schema_version": "judge_policy.v1",
                }
            },
            controller=None,
            enzyme_sp_verifier=None,
        )

        self.assertTrue(results)
        table = results[0].explanation.uncertainty_table
        self.assertEqual(table["route_tree_search_status"], "stock_closed")
        self.assertEqual(table["route_tree_compiled_judge_trace"][0]["decision"], "accept")
        self.assertEqual(table["route_tree_compiled_judge_trace"][0]["reason"], "anchor_whitelist")

    def test_root_target_stock_does_not_prevent_first_expansion(self):
        stock = lambda smi: smi in {"CC", "C"}
        planner = NeuralGuidedAOSearch(
            retro_engine={"retrochimera": _RootStockRetro()},
            stock_checker=stock,
            max_depth=2,
            branch_factor=4,
            expansion_budget=4,
            controller=None,
        )

        results = planner.search("CC", n_results=1)

        self.assertTrue(results)
        metrics = route_metrics(results[0].board, stock_checker=stock)
        self.assertTrue(metrics["strict_stock_solve"])
        self.assertGreaterEqual(len(results[0].board.slots), 1)

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

    def test_stock_rescue_budget_can_start_before_last_depth(self):
        old_rescue = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE")
        old_depth = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REMAINING_DEPTH")
        old_multiplier = os.environ.get("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_MULTIPLIER")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REMAINING_DEPTH"] = "3"
            os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_MULTIPLIER"] = "3"
            planner = NeuralGuidedAOSearch(
                retro_engine={},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=3,
                branch_factor=4,
                controller=None,
            )
            state = RouteTreeState.initial("CCCCCCCC")
            context = planner._proposal_context(state)
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
            if old_depth is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REMAINING_DEPTH", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REMAINING_DEPTH"] = old_depth
            if old_multiplier is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_MULTIPLIER", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_MULTIPLIER"] = old_multiplier

        self.assertEqual(fallback_budget, 6)

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

    def test_late_stock_closing_probe_reserves_low_rank_stock_action(self):
        keys = [
            "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE",
            "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_SOURCES",
            "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK",
            "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK_CAP",
            "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_REMAINING_DEPTH",
            "AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL",
        ]
        old = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_SOURCES"] = "retrochimera"
            os.environ["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK"] = "6"
            os.environ["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK_CAP"] = "6"
            os.environ["AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_REMAINING_DEPTH"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = "0"
            retro = _LateStockClosingProbeRetro()
            planner = NeuralGuidedAOSearch(
                retro_engine={"retrochimera": retro},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=1,
                branch_factor=2,
                controller=None,
            )
            children = planner._expand_state(RouteTreeState.initial("CCCCCCCC"))
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(retro.top_k_values, [2, 6])
        solved_children = [child for child in children if not child.open_leaves]
        self.assertTrue(solved_children)
        self.assertEqual(
            solved_children[0].steps[-1].action.metadata["stock_closing_probe"]["rank"],
            6,
        )

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

    def test_stock_closure_bonus_is_env_gated_and_rewards_exact_stock(self):
        env_keys = [
            "AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS",
            "AUTOPLANNER_ROUTE_TREE_FULL_STOCK_ACTION_BONUS",
        ]
        old_env = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in env_keys:
                os.environ.pop(key, None)
            planner = NeuralGuidedAOSearch(
                retro_engine={},
                controller=None,
                stock_checker=lambda smi: smi in {"CCCC", "CC"},
            )
            state = RouteTreeState.initial("CCCCCC")
            stock_action = CandidateAction.from_candidate(
                "CCCCCC",
                {
                    "main_reactant": "CCCC",
                    "aux_reactants": ["CC"],
                    "rxn_smiles": "CCCC.CC>>CCCCCC",
                    "score": 0.2,
                },
            )
            nonstock_action = CandidateAction.from_candidate(
                "CCCCCC",
                {
                    "main_reactant": "CCCCC",
                    "aux_reactants": ["C"],
                    "rxn_smiles": "CCCCC.C>>CCCCCC",
                    "score": 0.2,
                },
            )
            eval_result = RouteTreeEvaluation(action_scores=[1.0, 1.0])
            stock_base = planner._selection_score_row(
                state,
                "CCCCCC",
                stock_action,
                eval_result,
                policy_probability=0.0,
                model_score=1.0,
                next_open=(),
            )
            nonstock_base = planner._selection_score_row(
                state,
                "CCCCCC",
                nonstock_action,
                eval_result,
                policy_probability=0.0,
                model_score=1.0,
                next_open=("CCCCC", "C"),
            )

            os.environ["AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS"] = "1.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_FULL_STOCK_ACTION_BONUS"] = "2.0"
            stock_bonus = planner._selection_score_row(
                state,
                "CCCCCC",
                stock_action,
                eval_result,
                policy_probability=0.0,
                model_score=1.0,
                next_open=(),
            )
            nonstock_bonus = planner._selection_score_row(
                state,
                "CCCCCC",
                nonstock_action,
                eval_result,
                policy_probability=0.0,
                model_score=1.0,
                next_open=("CCCCC", "C"),
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(stock_base["stock_closure_bonus"], 0.0)
        self.assertEqual(nonstock_base["stock_closure_bonus"], 0.0)
        self.assertGreater(stock_bonus["stock_closure_bonus"], 0.0)
        self.assertEqual(nonstock_bonus["stock_closure_bonus"], 0.0)
        self.assertLess(stock_bonus["total_cost"], stock_base["total_cost"])
        self.assertEqual(stock_bonus["stock_closure_diagnostics"]["exact_stock_reactants"], 2)

    def test_stock_closure_bonus_can_reward_normalized_stock_hit(self):
        env_keys = [
            "AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS",
            "AUTOPLANNER_ROUTE_TREE_FULL_STOCK_ACTION_BONUS",
            "AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_REACTANT_BONUS",
            "AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_FULL_ACTION_BONUS",
        ]
        old_env = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in env_keys:
                os.environ.pop(key, None)
            os.environ["AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_REACTANT_BONUS"] = "1.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_FULL_ACTION_BONUS"] = "1.5"
            planner = NeuralGuidedAOSearch(
                retro_engine={},
                controller=None,
                stock_checker=lambda smi: smi == "CC(=O)O",
            )
            state = RouteTreeState.initial("CC(=O)OC")
            action = CandidateAction.from_candidate(
                "CC(=O)OC",
                {
                    "main_reactant": "CC(=O)[O-]",
                    "rxn_smiles": "CC(=O)[O-]>>CC(=O)OC",
                    "score": 0.2,
                },
            )
            row = planner._selection_score_row(
                state,
                "CC(=O)OC",
                action,
                RouteTreeEvaluation(action_scores=[1.0]),
                policy_probability=0.0,
                model_score=1.0,
                next_open=("CC(=O)[O-]",),
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertGreater(row["stock_closure_bonus"], 0.0)
        self.assertEqual(row["stock_closure_diagnostics"]["exact_stock_reactants"], 0)
        self.assertEqual(row["stock_closure_diagnostics"]["normalized_stock_reactants"], 1)

    def test_no_progress_single_reactant_penalty_is_env_gated(self):
        env_key = "AUTOPLANNER_ROUTE_TREE_NO_PROGRESS_SINGLE_REACTANT_PENALTY"
        old_value = os.environ.get(env_key)
        try:
            os.environ.pop(env_key, None)
            planner = NeuralGuidedAOSearch(
                retro_engine={},
                controller=None,
                stock_checker=lambda smi: smi == "CCCCCC",
            )
            state = RouteTreeState.initial("CCCCCCCCCCCC")
            no_progress = CandidateAction.from_candidate(
                "CCCCCCCCCCCC",
                {
                    "main_reactant": "CCCCCCCCCCCC",
                    "rxn_smiles": "CCCCCCCCCCCC>>CCCCCCCCCCCC",
                    "score": 0.2,
                },
            )
            split = CandidateAction.from_candidate(
                "CCCCCCCCCCCC",
                {
                    "main_reactant": "CCCCCC",
                    "aux_reactants": ["CCCCCC"],
                    "rxn_smiles": "CCCCCC.CCCCCC>>CCCCCCCCCCCC",
                    "score": 0.2,
                },
            )
            base = planner._action_feasibility_cost("CCCCCCCCCCCC", no_progress)[1]
            os.environ[env_key] = "0.6"
            penalized_cost, penalized = planner._action_feasibility_cost("CCCCCCCCCCCC", no_progress)
            split_cost, split_diag = planner._action_feasibility_cost("CCCCCCCCCCCC", split)
        finally:
            if old_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old_value

        self.assertEqual(base["no_progress_single_reactant"], 0.0)
        self.assertEqual(penalized["no_progress_single_reactant"], 0.6)
        self.assertEqual(split_diag["no_progress_single_reactant"], 0.0)
        self.assertGreater(penalized_cost, split_cost)

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
        self.assertEqual(rows[0]["reactant_smiles"], ["CC", "O"])
        self.assertEqual(rows[0]["rxn_smiles"], "CC.O>>CCO")
        self.assertEqual(rows[0]["template"], "graph_template")
        self.assertEqual(rows[0]["type"], "template")
        self.assertEqual(rows[0]["proposal_type"], "chem_enzy_one_step")
        self.assertTrue(rows[0]["teacher_one_step"])
        self.assertEqual(source_group(rows[0]["source"]), "chemical")

    def test_chem_enzy_onestep_provider_marks_template_relevance_rows(self):
        class FakeTemplateRelevanceOneStep:
            def run(self, product, topk=10):
                return {
                    "reactants": ["CC.O"],
                    "scores": [0.7],
                    "template": ["fake_template"],
                    "model_full_name": ["template_relevance.bkms_metabolic"],
                    "weight": [1.0],
                }

        provider = ChemEnzyOneStepProposalProvider(one_step=FakeTemplateRelevanceOneStep())

        rows = provider.predict("CCO", top_k=1)

        self.assertEqual(rows[0]["source"], "template_relevance")
        self.assertEqual(rows[0]["proposal_type"], "template_relevance")
        self.assertEqual(rows[0]["type"], "template_relevance")
        self.assertEqual(rows[0]["reactant_smiles"], ["CC", "O"])
        self.assertEqual(source_group(rows[0]["source"]), "chemical")

    def test_chem_enzy_onestep_single_model_wrapper_preserves_source(self):
        from cascade_planner.baselines.chem_enzy_onestep import _SingleModelRunWrapper

        class FakeSingleStep:
            def run(self, product, topk=10):
                return {
                    "reactants": ["CC.O"],
                    "scores": [0.6],
                    "template": ["fake_template"],
                }

        wrapped = _SingleModelRunWrapper(FakeSingleStep(), "template_relevance.bkms_metabolic")
        provider = ChemEnzyOneStepProposalProvider(one_step=wrapped)

        rows = provider.predict("CCO", top_k=1)

        self.assertEqual(rows[0]["model_full_name"], "template_relevance.bkms_metabolic")
        self.assertEqual(rows[0]["source"], "template_relevance")
        self.assertEqual(rows[0]["proposal_type"], "template_relevance")

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

    def test_chem_enzy_onestep_provider_reads_onmt_tokenizer_env(self):
        key = "AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER"
        old = os.environ.get(key)
        try:
            os.environ[key] = "token"
            provider = ChemEnzyOneStepProposalProvider.from_env()
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

        self.assertEqual(provider.onmt_tokenizer, "token")

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

    def test_retro_engine_proposal_tool_attaches_runtime_condition_predictions(self):
        class FakeConditionPredictor:
            def predict(self, rxn_smiles, top_k=1):
                return [{"Temperature": 25.0, "Solvent": "water", "Score": 0.55}]

        tool = RetroEngineProposalTool(
            {
                "retrochimera": _SourceRecorder(
                    "retrochimera",
                    rows=[{"main_reactant": "CC", "rxn_smiles": "CC>>CCO", "score": 0.9}],
                ),
                "enzexpand": _SourceRecorder(
                    "enzexpand",
                    rows=[{"main_reactant": "CC", "rxn_smiles": "CC>>CCO", "ec": "1.1.1.1", "score": 0.8}],
                ),
            },
            source_order=("retrochimera", "enzexpand"),
        )
        old_env = {
            key: os.environ.get(key)
            for key in (
                "AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION",
                "AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION_CHEMICAL_ONLY",
            )
        }
        old_override = getattr(route_tree_proposals, "_ROUTE_TREE_CONDITION_PREDICTOR_OVERRIDE", None)
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION_CHEMICAL_ONLY"] = "1"
            route_tree_proposals._ROUTE_TREE_CONDITION_PREDICTOR_OVERRIDE = FakeConditionPredictor()
            actions = tool.propose("CCO", ProposalContext(), top_k=4)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if old_override is None:
                if hasattr(route_tree_proposals, "_ROUTE_TREE_CONDITION_PREDICTOR_OVERRIDE"):
                    delattr(route_tree_proposals, "_ROUTE_TREE_CONDITION_PREDICTOR_OVERRIDE")
            else:
                route_tree_proposals._ROUTE_TREE_CONDITION_PREDICTOR_OVERRIDE = old_override

        by_source = {action.source: action for action in actions}
        self.assertEqual(by_source["retrochimera"].metadata["condition_predictions"][0]["Temperature"], 25.0)
        self.assertEqual(by_source["retrochimera"].metadata["condition_prediction_enabled_by"], "route_tree_runtime")
        self.assertNotIn("condition_predictions", by_source["enzexpand"].metadata)

    def test_chem_enzy_onestep_fallback_mode_skips_when_primary_source_returns_actions(self):
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CCO",
            "score": 0.9,
        }])
        chem_enzy = _SourceRecorder("chem_enzy_onestep", rows=[{
            "main_reactant": "CO",
            "rxn_smiles": "CO>>CCO",
            "score": 0.8,
        }])
        tool = RetroEngineProposalTool(
            {"retrochimera": chemical, "chem_enzy_onestep": chem_enzy},
            source_order=("retrochimera", "chem_enzy_onestep"),
        )
        old = os.environ.get("AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE")
        try:
            os.environ["AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE"] = "fallback"
            actions = tool.propose("CCO", ProposalContext(), top_k=4)
            diagnostics = tool.last_diagnostics["sources"]
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE", None)
            else:
                os.environ["AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE"] = old

        self.assertEqual(chemical.calls, 1)
        self.assertEqual(chem_enzy.calls, 0)
        self.assertEqual([action.source for action in actions], ["retrochimera"])
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["allocated_budget"]), 0)
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["calls"]), 0)

    def test_chem_enzy_onestep_fallback_mode_queries_when_primary_source_is_empty(self):
        chemical = _SourceRecorder("retrochimera", rows=[])
        chem_enzy = _SourceRecorder("chem_enzy_onestep", rows=[{
            "main_reactant": "CO",
            "rxn_smiles": "CO>>CCO",
            "score": 0.8,
        }])
        tool = RetroEngineProposalTool(
            {"retrochimera": chemical, "chem_enzy_onestep": chem_enzy},
            source_order=("retrochimera", "chem_enzy_onestep"),
        )
        old = os.environ.get("AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE")
        try:
            os.environ["AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE"] = "fallback"
            actions = tool.propose("CCO", ProposalContext(), top_k=4)
            diagnostics = tool.last_diagnostics["sources"]
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE", None)
            else:
                os.environ["AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE"] = old

        self.assertEqual(chemical.calls, 1)
        self.assertEqual(chem_enzy.calls, 1)
        self.assertEqual([action.source for action in actions], ["chem_enzy_onestep"])
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["allocated_budget"]), 0)
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["calls"]), 1)

    def test_chem_enzy_onestep_adaptive_mode_queries_when_primary_source_is_weak(self):
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CCO",
            "score": 0.9,
        }])
        chem_enzy = _SourceRecorder("chem_enzy_onestep", rows=[{
            "main_reactant": "CO",
            "rxn_smiles": "CO>>CCO",
            "score": 0.8,
        }])
        tool = RetroEngineProposalTool(
            {"retrochimera": chemical, "chem_enzy_onestep": chem_enzy},
            source_order=("retrochimera", "chem_enzy_onestep"),
        )
        keys = {
            "AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE": "adaptive",
            "AUTOPLANNER_CHEMENZY_ONESTEP_ADAPTIVE_MIN_ACTIONS": "2",
            "AUTOPLANNER_CHEMENZY_ONESTEP_ADAPTIVE_MAX_BUDGET": "3",
        }
        old = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(keys)
            actions = tool.propose("CCO", ProposalContext(), top_k=4)
            diagnostics = tool.last_diagnostics["sources"]
            adaptive = tool.last_diagnostics["chem_enzy_onestep_adaptive_fallback"]
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(chemical.calls, 1)
        self.assertEqual(chem_enzy.calls, 1)
        self.assertEqual(chem_enzy.top_k_values, [3])
        self.assertEqual([action.source for action in actions], ["retrochimera", "chem_enzy_onestep"])
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["allocated_budget"]), 0)
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["calls"]), 1)
        self.assertTrue(adaptive["triggered"])
        self.assertEqual(adaptive["reason"], "low_strong_action_count")

    def test_chem_enzy_onestep_adaptive_mode_skips_when_primary_source_is_strong(self):
        chemical = _SourceRecorder("retrochimera", rows=[
            {
                "main_reactant": "CC",
                "rxn_smiles": "CC>>CCO",
                "score": 0.9,
            },
            {
                "main_reactant": "CO",
                "rxn_smiles": "CO>>CCO",
                "score": 0.8,
            },
        ])
        chem_enzy = _SourceRecorder("chem_enzy_onestep", rows=[{
            "main_reactant": "CN",
            "rxn_smiles": "CN>>CCO",
            "score": 0.7,
        }])
        tool = RetroEngineProposalTool(
            {"retrochimera": chemical, "chem_enzy_onestep": chem_enzy},
            source_order=("retrochimera", "chem_enzy_onestep"),
        )
        keys = {
            "AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE": "adaptive",
            "AUTOPLANNER_CHEMENZY_ONESTEP_ADAPTIVE_MIN_ACTIONS": "2",
        }
        old = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(keys)
            actions = tool.propose("CCO", ProposalContext(), top_k=4)
            diagnostics = tool.last_diagnostics["sources"]
            adaptive = tool.last_diagnostics["chem_enzy_onestep_adaptive_fallback"]
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(chemical.calls, 1)
        self.assertEqual(chem_enzy.calls, 0)
        self.assertEqual([action.source for action in actions], ["retrochimera", "retrochimera"])
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["allocated_budget"]), 0)
        self.assertEqual(int(diagnostics["chem_enzy_onestep"]["calls"]), 0)
        self.assertFalse(adaptive["triggered"])

    def test_chem_enzy_onmt_is_treated_as_enzymatic_source(self):
        self.assertEqual(source_group("chem_enzy_onmt"), "enzymatic")
        self.assertEqual(source_group("chem_enzy_bionav"), "enzymatic")

    def test_chem_enzy_graphfp_fusion_is_treated_as_chemical_source(self):
        self.assertEqual(source_group("chem_enzy_graphfp_fusion"), "chemical")
        self.assertEqual(source_group("autoplanner_dualtower"), "chemical")
        self.assertEqual(source_group("template_relevance"), "chemical")

    def test_retro_engine_proposal_tool_can_query_template_relevance_source(self):
        template_relevance = _SourceRecorder("template_relevance", rows=[{
            "main_reactant": "CO",
            "rxn_smiles": "CO>>CCO",
            "score": 0.8,
            "proposal_type": "template_relevance",
        }])
        tool = RetroEngineProposalTool(
            {"template_relevance": template_relevance},
            source_order=("template_relevance",),
        )

        actions = tool.propose("CCO", ProposalContext(), top_k=2)

        self.assertEqual(template_relevance.calls, 1)
        self.assertEqual([action.source for action in actions], ["template_relevance"])
        self.assertEqual(actions[0].rxn_smiles, "CO>>CCO")
        diagnostics = tool.last_diagnostics["sources"]["template_relevance"]
        self.assertEqual(diagnostics["raw_returned"], 1)
        self.assertEqual(diagnostics["kept_returned"], 1)

    def test_chemical_context_budget_floor_preserves_graphfp_fusion_source(self):
        sources = ["retrochimera", "chem_enzy_graphfp_fusion", "chemtemplates"]
        allocation = SourceAllocation(
            source_weights={source: 1.0 / len(sources) for source in sources},
            source_budgets={
                "retrochimera": 8,
                "chem_enzy_graphfp_fusion": 0,
                "chemtemplates": 2,
            },
            fallback_budget=2,
            molecule_flags={},
            source_group_probs={},
        )
        old = os.environ.get("AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MIN_BUDGET")
        try:
            os.environ["AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MIN_BUDGET"] = "3"
            updated = _apply_source_budget_floor(
                allocation,
                sources=sources,
                total_budget=10,
                context=ProposalContext(reaction_type="chemical"),
            )
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MIN_BUDGET", None)
            else:
                os.environ["AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MIN_BUDGET"] = old

        self.assertGreaterEqual(updated.source_budgets["chem_enzy_graphfp_fusion"], 3)
        self.assertLessEqual(sum(updated.source_budgets.values()), 10)

    def test_chemical_context_budget_floor_preserves_template_relevance_source(self):
        sources = ["retrochimera", "template_relevance", "chemtemplates"]
        allocation = SourceAllocation(
            source_weights={source: 1.0 / len(sources) for source in sources},
            source_budgets={
                "retrochimera": 8,
                "template_relevance": 0,
                "chemtemplates": 2,
            },
            fallback_budget=2,
            molecule_flags={},
            source_group_probs={},
        )
        old = os.environ.get("AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET")
        try:
            os.environ["AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET"] = "3"
            updated = _apply_source_budget_floor(
                allocation,
                sources=sources,
                total_budget=10,
                context=ProposalContext(reaction_type="chemical"),
            )
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET", None)
            else:
                os.environ["AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET"] = old

        self.assertGreaterEqual(updated.source_budgets["template_relevance"], 3)
        self.assertLessEqual(sum(updated.source_budgets.values()), 10)

    def test_budget_floor_keeps_semisynthesis_under_small_budget_pressure(self):
        from cascade_planner.baselines.semisynthesis_rescue import N_DEBENZOYLTAXOL

        sources = ["semisynthesis_rescue", "template_relevance", "chemtemplates"]
        allocation = SourceAllocation(
            source_weights={source: 1.0 / len(sources) for source in sources},
            source_budgets={
                "semisynthesis_rescue": 0,
                "template_relevance": 2,
                "chemtemplates": 2,
            },
            fallback_budget=2,
            molecule_flags={},
            source_group_probs={},
        )
        old_semisynthesis = os.environ.get("AUTOPLANNER_SEMISYNTHESIS_RESCUE_MIN_BUDGET")
        old_template = os.environ.get("AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET")
        try:
            os.environ["AUTOPLANNER_SEMISYNTHESIS_RESCUE_MIN_BUDGET"] = "2"
            os.environ["AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET"] = "2"
            updated = _apply_source_budget_floor(
                allocation,
                sources=sources,
                total_budget=2,
                context=ProposalContext(reaction_type="chemical"),
                product=N_DEBENZOYLTAXOL,
            )
            unrelated = _apply_source_budget_floor(
                allocation,
                sources=sources,
                total_budget=2,
                context=ProposalContext(reaction_type="chemical"),
                product="CCO",
            )
        finally:
            if old_semisynthesis is None:
                os.environ.pop("AUTOPLANNER_SEMISYNTHESIS_RESCUE_MIN_BUDGET", None)
            else:
                os.environ["AUTOPLANNER_SEMISYNTHESIS_RESCUE_MIN_BUDGET"] = old_semisynthesis
            if old_template is None:
                os.environ.pop("AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET", None)
            else:
                os.environ["AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET"] = old_template

        self.assertGreaterEqual(updated.source_budgets["semisynthesis_rescue"], 1)
        self.assertLessEqual(sum(updated.source_budgets.values()), 2)
        self.assertEqual(unrelated.source_budgets["semisynthesis_rescue"], 0)

    def test_budget_floor_keeps_chemical_anchor_under_small_budget_pressure(self):
        from cascade_planner.baselines.chemical_anchor_rescue import BENZOTHIAZINE_TARGET

        sources = ["chemical_anchor_rescue", "template_relevance", "chemtemplates"]
        allocation = SourceAllocation(
            source_weights={source: 1.0 / len(sources) for source in sources},
            source_budgets={
                "chemical_anchor_rescue": 0,
                "template_relevance": 2,
                "chemtemplates": 2,
            },
            fallback_budget=2,
            molecule_flags={},
            source_group_probs={},
        )
        old_anchor = os.environ.get("AUTOPLANNER_CHEMICAL_ANCHOR_RESCUE_MIN_BUDGET")
        old_template = os.environ.get("AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET")
        try:
            os.environ["AUTOPLANNER_CHEMICAL_ANCHOR_RESCUE_MIN_BUDGET"] = "2"
            os.environ["AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET"] = "2"
            updated = _apply_source_budget_floor(
                allocation,
                sources=sources,
                total_budget=2,
                context=ProposalContext(reaction_type="chemical"),
                product=BENZOTHIAZINE_TARGET,
            )
            unrelated = _apply_source_budget_floor(
                allocation,
                sources=sources,
                total_budget=2,
                context=ProposalContext(reaction_type="chemical"),
                product="CCO",
            )
        finally:
            if old_anchor is None:
                os.environ.pop("AUTOPLANNER_CHEMICAL_ANCHOR_RESCUE_MIN_BUDGET", None)
            else:
                os.environ["AUTOPLANNER_CHEMICAL_ANCHOR_RESCUE_MIN_BUDGET"] = old_anchor
            if old_template is None:
                os.environ.pop("AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET", None)
            else:
                os.environ["AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET"] = old_template

        self.assertGreaterEqual(updated.source_budgets["chemical_anchor_rescue"], 1)
        self.assertLessEqual(sum(updated.source_budgets.values()), 2)
        self.assertEqual(unrelated.source_budgets["chemical_anchor_rescue"], 0)

    def test_dedupe_prefers_semisynthesis_provenance_over_template_duplicate(self):
        product = "CCO"
        template = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "CO",
                "rxn_smiles": "CO>>CCO",
                "source": "template_relevance",
                "score": 0.8,
                "rank": 1,
            },
            source="template_relevance",
        )
        semisynthesis = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "CO",
                "rxn_smiles": "CO>>CCO",
                "source": "semisynthesis_rescue",
                "score": 0.9,
                "rank": 1,
                "ec": "2.3.1.167",
                "semisynthesis_rescue": {"type": "taxane_10dab_side_chain_acetylation"},
            },
            source="semisynthesis_rescue",
        )

        actions, diagnostics = _dedupe_actions_with_diagnostics([template, semisynthesis])

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "semisynthesis_rescue")
        self.assertEqual(actions[0].ec, "2.3.1.167")
        self.assertEqual(actions[0].metadata["source_provenance"]["source"], "semisynthesis_rescue")
        state = RouteTreeState.initial(product).advance(
            leaf=product,
            action=actions[0],
            next_open_leaves=(),
            score_delta=0.0,
        )
        self.assertEqual(
            state.to_board().slots[0].evidence["semisynthesis_rescue"]["type"],
            "taxane_10dab_side_chain_acetylation",
        )
        duplicate_sources = {
            item["source"] for item in actions[0].metadata.get("duplicate_source_provenance", [])
        }
        self.assertIn("template_relevance", duplicate_sources)
        self.assertEqual(diagnostics["dedupe_dropped"]["template_relevance"], 1)

    def test_semisynthesis_source_is_not_queried_for_unmatched_product(self):
        semisynthesis = _SourceRecorder(
            "semisynthesis_rescue",
            rows=[{"main_reactant": "CO", "rxn_smiles": "CO>>CCO", "score": 0.9}],
        )
        template = _SourceRecorder(
            "template_relevance",
            rows=[{"main_reactant": "CO", "rxn_smiles": "CO>>CCO", "score": 0.8}],
        )
        tool = RetroEngineProposalTool(
            {
                "semisynthesis_rescue": semisynthesis,
                "template_relevance": template,
            },
            source_order=("semisynthesis_rescue", "template_relevance"),
        )

        actions = tool.propose("CCO", ProposalContext(reaction_type="chemical"), top_k=2)

        self.assertEqual(semisynthesis.calls, 0)
        self.assertEqual(template.calls, 1)
        self.assertEqual([action.source for action in actions], ["template_relevance"])

    def test_chemical_anchor_source_is_not_queried_for_unmatched_product(self):
        anchor = _SourceRecorder(
            "chemical_anchor_rescue",
            rows=[{"main_reactant": "CO", "rxn_smiles": "CO>>CCO", "score": 0.9}],
        )
        template = _SourceRecorder(
            "template_relevance",
            rows=[{"main_reactant": "CO", "rxn_smiles": "CO>>CCO", "score": 0.8}],
        )
        tool = RetroEngineProposalTool(
            {
                "chemical_anchor_rescue": anchor,
                "template_relevance": template,
            },
            source_order=("chemical_anchor_rescue", "template_relevance"),
        )

        actions = tool.propose("CCO", ProposalContext(reaction_type="chemical"), top_k=2)

        self.assertEqual(anchor.calls, 0)
        self.assertEqual(template.calls, 1)
        self.assertEqual([action.source for action in actions], ["template_relevance"])

    def test_chemical_anchor_evidence_is_exported_to_board_slot(self):
        product = "CCO"
        action = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "CO",
                "rxn_smiles": "CO>>CCO",
                "source": "chemical_anchor_rescue",
                "score": 0.9,
                "rank": 1,
                "chemical_anchor_rescue": {"type": "benzothiazine_c2_amination"},
            },
            source="chemical_anchor_rescue",
        )

        state = RouteTreeState.initial(product).advance(
            leaf=product,
            action=action,
            next_open_leaves=(),
            score_delta=0.0,
        )

        self.assertEqual(
            state.to_board().slots[0].evidence["chemical_anchor_rescue"]["type"],
            "benzothiazine_c2_amination",
        )

    def test_bridge_hit_route_gate_suppresses_graphfp_fusion_source(self):
        sources = [
            "retrochimera",
            "chem_enzy_graphfp_fusion",
            "chem_enzy_bionav",
            "enzyme_precedent",
        ]
        allocation = SourceAllocation(
            source_weights={source: 1.0 / len(sources) for source in sources},
            source_budgets={
                "retrochimera": 4,
                "chem_enzy_graphfp_fusion": 4,
                "chem_enzy_bionav": 1,
                "enzyme_precedent": 1,
            },
            fallback_budget=2,
            molecule_flags={"bridge_gate_checked": True, "bridge_gate_hits": 1},
            source_group_probs={},
        )

        updated = _apply_route_source_gates(
            allocation,
            sources=sources,
            total_budget=10,
            context=ProposalContext(),
        )

        self.assertEqual(updated.source_budgets["chem_enzy_graphfp_fusion"], 0)
        self.assertEqual(sum(updated.source_budgets.values()), 10)
        self.assertTrue(updated.molecule_flags["chem_enzy_graphfp_fusion_bridge_gate_active"])

    def test_bridge_hit_fallback_skips_graphfp_fusion_source(self):
        class _BridgeHitGate(SourceGate):
            def allocate(self, product, *, context, available_sources, total_budget):
                del product, context, total_budget
                return SourceAllocation(
                    source_weights={source: 1.0 / len(available_sources) for source in available_sources},
                    source_budgets={
                        "retrochimera": 1,
                        "chem_enzy_graphfp_fusion": 1,
                    },
                    fallback_budget=2,
                    molecule_flags={"bridge_gate_checked": True, "bridge_gate_hits": 1},
                    source_group_probs={},
                )

        retro = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CCO",
            "score": 0.9,
        }])
        graphfp = _SourceRecorder("chem_enzy_graphfp_fusion", rows=[{
            "main_reactant": "CO",
            "rxn_smiles": "CO>>CCO",
            "score": 0.8,
        }])
        tool = RetroEngineProposalTool(
            {"retrochimera": retro, "chem_enzy_graphfp_fusion": graphfp},
            source_order=("retrochimera", "chem_enzy_graphfp_fusion"),
            source_gate=_BridgeHitGate(),
        )

        actions = tool.propose("CCO", ProposalContext(), top_k=4)

        self.assertEqual(retro.calls, 1)
        self.assertEqual(graphfp.calls, 0)
        self.assertEqual([action.source for action in actions], ["retrochimera"])

    def test_ec_context_fallback_skips_graphfp_fusion_source(self):
        class _EcGate(SourceGate):
            def allocate(self, product, *, context, available_sources, total_budget):
                del product, context, total_budget
                return SourceAllocation(
                    source_weights={source: 1.0 / len(available_sources) for source in available_sources},
                    source_budgets={
                        "chem_enzy_bionav": 1,
                        "chem_enzy_graphfp_fusion": 0,
                    },
                    fallback_budget=2,
                    molecule_flags={"bridge_gate_checked": False, "bridge_gate_hits": 0},
                    source_group_probs={},
                    policy_reason="bridge_gate_bypassed_explicit_ec_context",
                )

        bionav = _SourceRecorder("chem_enzy_bionav", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CCO",
            "score": 0.9,
            "source": "chem_enzy_onmt",
        }])
        graphfp = _SourceRecorder("chem_enzy_graphfp_fusion", rows=[{
            "main_reactant": "CO",
            "rxn_smiles": "CO>>CCO",
            "score": 0.8,
        }])
        tool = RetroEngineProposalTool(
            {"chem_enzy_bionav": bionav, "chem_enzy_graphfp_fusion": graphfp},
            source_order=("chem_enzy_bionav", "chem_enzy_graphfp_fusion"),
            source_gate=_EcGate(),
        )

        actions = tool.propose("CCO", ProposalContext(ec1=1), top_k=4)

        self.assertEqual(bionav.calls, 1)
        self.assertEqual(graphfp.calls, 0)
        self.assertEqual([action.source for action in actions], ["chem_enzy_bionav"])

    def test_bridge_hit_budget_floor_preserves_existing_enzyme_sources_with_bionav(self):
        sources = [
            "retrochimera",
            "chem_enzy_bionav",
            "enzyme_precedent",
            "v3_retrieval",
            "enzyformer",
            "enzexpand",
            "chemtemplates",
        ]
        allocation = SourceAllocation(
            source_weights={source: 1.0 / len(sources) for source in sources},
            source_budgets={
                "retrochimera": 22,
                "chem_enzy_bionav": 1,
                "enzyme_precedent": 1,
                "v3_retrieval": 1,
                "enzyformer": 1,
                "enzexpand": 1,
                "chemtemplates": 23,
            },
            fallback_budget=12,
            molecule_flags={"bridge_gate_checked": True, "bridge_gate_hits": 1},
            source_group_probs={},
        )

        updated = _apply_source_budget_floor(
            allocation,
            sources=sources,
            total_budget=50,
            context=ProposalContext(),
        )

        self.assertGreaterEqual(updated.source_budgets["enzyme_precedent"], 2)
        self.assertGreaterEqual(updated.source_budgets["v3_retrieval"], 2)
        self.assertGreaterEqual(updated.source_budgets["chem_enzy_bionav"], 1)
        self.assertLessEqual(sum(updated.source_budgets.values()), 50)

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

    def test_ec_context_keeps_enzyme_precedent_budget(self):
        from cascade_planner.route_tree.proposals import _apply_source_budget_floor

        sources = [
            "enzyme_precedent",
            "v3_retrieval",
            "enzyformer",
            "enzexpand",
            "retrorules",
            "retrochimera",
        ]
        allocation = SourceGate().allocate(
            "CC=O",
            context=ProposalContext(depth=0, ec1=1),
            available_sources=sources,
            total_budget=6,
        )

        allocation = _apply_source_budget_floor(
            allocation,
            sources=sources,
            total_budget=6,
            context=ProposalContext(depth=0, ec1=1),
        )

        self.assertGreaterEqual(allocation.source_budgets["enzyme_precedent"], 1)
        self.assertGreaterEqual(allocation.source_budgets["v3_retrieval"], 1)
        self.assertEqual(allocation.source_budgets["retrochimera"], 0)
        self.assertEqual(sum(allocation.source_budgets.values()), 6)

    def test_bridge_aware_source_gate_suppresses_enzymatic_budget_without_bridge_hit(self):
        gate = BridgeAwareSourceGate(SourceGate(), retriever=_FakeBridgeRetriever([]))

        allocation = gate.allocate(
            "CC=O",
            context=ProposalContext(),
            available_sources=["retrochimera", "enzyformer", "retrorules"],
            total_budget=6,
        )

        self.assertEqual(allocation.source_budgets["enzyformer"], 0)
        self.assertEqual(allocation.source_budgets["retrorules"], 0)
        self.assertEqual(allocation.source_budgets["retrochimera"], 6)
        self.assertEqual(allocation.policy_reason, "bridge_gate_no_hits_suppress_enzymatic")
        self.assertTrue(allocation.molecule_flags["bridge_gate_checked"])
        self.assertEqual(allocation.molecule_flags["bridge_gate_hits"], 0)

    def test_bridge_aware_source_gate_blocks_enzymatic_fallback_without_bridge_hit(self):
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
        }])
        enzymatic = _SourceRecorder("enzyformer", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "ec": "1.1.1.1",
        }])
        gate = BridgeAwareSourceGate(SourceGate(), retriever=_FakeBridgeRetriever([]))
        tool = RetroEngineProposalTool(
            {"retrochimera": chemical, "enzyformer": enzymatic},
            source_gate=gate,
        )

        actions = tool.propose("CC=O", ProposalContext(), top_k=4)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "retrochimera")
        self.assertEqual(chemical.calls, 1)
        self.assertEqual(enzymatic.calls, 0)

    def test_bridge_aware_source_gate_blocks_enzyme_precedent_floor_without_bridge_hit(self):
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
        }])
        gate = BridgeAwareSourceGate(SourceGate(), retriever=_FakeBridgeRetriever([]))
        tool = RetroEngineProposalTool(
            {"retrochimera": chemical},
            source_gate=gate,
        )
        old_retrieval = os.environ.get("AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL")
        old_floors = os.environ.get("AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS"] = "enzyme_precedent:2"
            actions = tool.propose("CC=O", ProposalContext(), top_k=4)
            diagnostics = tool.last_diagnostics.get("sources") or {}
        finally:
            if old_retrieval is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL"] = old_retrieval
            if old_floors is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS"] = old_floors

        self.assertEqual([action.source for action in actions], ["retrochimera"])
        self.assertEqual(int((diagnostics.get("enzyme_precedent") or {}).get("calls") or 0), 0)
        self.assertEqual(int((diagnostics.get("enzyme_precedent") or {}).get("allocated_budget") or 0), 0)

    def test_bridge_aware_source_gate_boosts_enzymatic_budget_with_bridge_hit(self):
        gate = BridgeAwareSourceGate(
            SourceGate(),
            retriever=_FakeBridgeRetriever([{"bridge": "hit"}]),
            bridge_enzymatic_fraction=0.5,
        )

        allocation = gate.allocate(
            "CC=O",
            context=ProposalContext(),
            available_sources=["retrochimera", "enzyformer", "retrorules"],
            total_budget=6,
        )

        self.assertGreater(allocation.source_budgets["enzyformer"] + allocation.source_budgets["retrorules"], 0)
        self.assertEqual(sum(allocation.source_budgets.values()), 6)
        self.assertEqual(allocation.policy_reason, "bridge_gate_hits")
        self.assertEqual(allocation.molecule_flags["bridge_gate_hits"], 1)

    def test_explicit_source_floor_is_not_overwritten_by_bridge_hit_floor(self):
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
        }])
        enzyme = _SourceRecorder(
            "chem_enzy_bionav",
            rows=[
                {
                    "main_reactant": f"C{i}",
                    "rxn_smiles": f"C{i}>>CC=O",
                    "ec": "1.1.1.1",
                    "score": 0.5,
                }
                for i in range(12)
            ],
        )
        gate = BridgeAwareSourceGate(
            SourceGate(),
            retriever=_FakeBridgeRetriever([{"bridge": "hit"}]),
            bridge_enzymatic_fraction=0.5,
        )
        tool = RetroEngineProposalTool(
            {"retrochimera": chemical, "chem_enzy_bionav": enzyme},
            source_gate=gate,
        )
        old_floors = os.environ.get("AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS"] = "chem_enzy_bionav:8"
            tool.propose("CC=O", ProposalContext(), top_k=12)
            diagnostics = tool.last_diagnostics.get("sources") or {}
        finally:
            if old_floors is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS"] = old_floors

        self.assertGreaterEqual(int((diagnostics.get("chem_enzy_bionav") or {}).get("allocated_budget") or 0), 8)
        self.assertGreaterEqual(enzyme.top_k_values[-1], 8)

    def test_bridge_aware_source_gate_can_use_bridge_hits_without_verifier_requirement(self):
        class _Retriever:
            def retrieve(self, product, **kwargs):
                return [{"bridge": "hit", "kwargs": kwargs}]

        gate = BridgeAwareSourceGate(
            SourceGate(),
            retriever=_Retriever(),
            require_verifier_pass=False,
        )

        allocation = gate.allocate(
            "CC=O",
            context=ProposalContext(),
            available_sources=["retrochimera", "enzyformer"],
            total_budget=4,
        )

        self.assertEqual(allocation.policy_reason, "bridge_gate_hits")
        self.assertEqual(allocation.molecule_flags["bridge_gate_hits"], 1)

    def test_bridge_aware_source_gate_allows_sp_accepted_enzyme_continuation(self):
        gate = BridgeAwareSourceGate(
            SourceGate(),
            retriever=_FakeBridgeRetriever([]),
            bridge_enzymatic_fraction=0.5,
        )
        context = ProposalContext(
            route_metadata={
                "enzyme_route_continuation": True,
                "sp_v1_accepted_enzyme_route": True,
            }
        )
        old_flag = os.environ.get("AUTOPLANNER_BRIDGE_GATE_ALLOW_ENZYME_CONTINUATION")
        try:
            os.environ["AUTOPLANNER_BRIDGE_GATE_ALLOW_ENZYME_CONTINUATION"] = "1"
            allocation = gate.allocate(
                "CC=O",
                context=context,
                available_sources=["chemtemplates", "chem_enzy_bionav", "enzyme_precedent"],
                total_budget=6,
            )
        finally:
            if old_flag is None:
                os.environ.pop("AUTOPLANNER_BRIDGE_GATE_ALLOW_ENZYME_CONTINUATION", None)
            else:
                os.environ["AUTOPLANNER_BRIDGE_GATE_ALLOW_ENZYME_CONTINUATION"] = old_flag

        self.assertEqual(allocation.policy_reason, "bridge_gate_enzyme_continuation")
        self.assertEqual(allocation.molecule_flags["bridge_gate_hits"], 0)
        self.assertTrue(allocation.molecule_flags["bridge_gate_enzyme_continuation"])
        self.assertGreater(allocation.source_budgets["chem_enzy_bionav"], 0)
        self.assertGreater(allocation.source_budgets["enzyme_precedent"], 0)

    def test_bridge_aware_source_gate_bypasses_explicit_ec_context(self):
        retriever = _FakeBridgeRetriever([])
        gate = BridgeAwareSourceGate(SourceGate(), retriever=retriever)

        allocation = gate.allocate(
            "CC=O",
            context=ProposalContext(ec1=1, reaction_type="reduction"),
            available_sources=["retrochimera", "enzyformer"],
            total_budget=4,
        )

        self.assertEqual(retriever.calls, [])
        self.assertEqual(allocation.policy_reason, "bridge_gate_bypassed_explicit_ec_context")
        self.assertGreater(allocation.source_budgets["enzyformer"], 0)

    def test_bridge_supported_enzyme_bonus_enters_selection_cost(self):
        product = "CCCCCCCC"
        state = RouteTreeState.initial(product)
        eval_result = RouteTreeEvaluation(action_scores=[], model_active=False, reason="test")
        planner = NeuralGuidedAOSearch(retro_engine={}, controller=None)
        chemical = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "CCCC",
                "rxn_smiles": "CCCC>>CCCCCCCC",
                "score": 0.5,
                "source": "retrochimera",
            },
            source="retrochimera",
        )
        enzyme = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "CCCC",
                "rxn_smiles": "CCCC>>CCCCCCCC",
                "score": 0.5,
                "source": "enzyformer",
                "ec": "1.1.1.1",
            },
            source="enzyformer",
        )
        enzyme.metadata["source_gate"] = {
            "policy_reason": "bridge_gate_hits",
            "molecule_flags": {"bridge_gate_checked": True, "bridge_gate_hits": 1},
        }

        old_bonus = os.environ.get("AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = "1.25"
            chemical_row = planner._selection_score_row(
                state,
                product,
                chemical,
                eval_result,
                policy_probability=0.0,
                model_score=0.0,
                next_open=("CCCC",),
            )
            enzyme_row = planner._selection_score_row(
                state,
                product,
                enzyme,
                eval_result,
                policy_probability=0.0,
                model_score=0.0,
                next_open=("CCCC",),
            )
        finally:
            if old_bonus is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = old_bonus

        self.assertEqual(chemical_row["bridge_supported_enzyme_bonus"], 0.0)
        self.assertEqual(enzyme_row["bridge_supported_enzyme_bonus"], 1.25)
        self.assertLess(enzyme_row["total_cost"], chemical_row["total_cost"])

    def test_enzyme_sp_accepted_bonus_enters_selection_cost(self):
        product = "CCCCCCCC"
        state = RouteTreeState.initial(product)
        eval_result = RouteTreeEvaluation(action_scores=[], model_active=False, reason="test")
        planner = NeuralGuidedAOSearch(retro_engine={}, controller=None)
        accepted = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "CCCC",
                "rxn_smiles": "CCCC>>CCCCCCCC",
                "score": 0.5,
                "source": "enzyme_precedent",
                "ec": "1.1.1.1",
            },
            source="enzyme_precedent",
        )
        rejected = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "CCCC",
                "rxn_smiles": "CCCC>>CCCCCCCC",
                "score": 0.5,
                "source": "enzyme_precedent",
                "ec": "1.1.1.1",
            },
            source="enzyme_precedent",
        )
        accepted.metadata["enzyme_sp_verifier_v1"] = {
            "accepted": True,
            "score": 0.8,
            "threshold": 0.3,
        }
        rejected.metadata["enzyme_sp_verifier_v1"] = {
            "accepted": False,
            "score": 0.2,
            "threshold": 0.3,
        }

        old_fixed = os.environ.get("AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS")
        old_weight = os.environ.get("AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS"] = "0.5"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS"] = "1.0"
            accepted_row = planner._selection_score_row(
                state,
                product,
                accepted,
                eval_result,
                policy_probability=0.0,
                model_score=0.0,
                next_open=("CCCC",),
            )
            rejected_row = planner._selection_score_row(
                state,
                product,
                rejected,
                eval_result,
                policy_probability=0.0,
                model_score=0.0,
                next_open=("CCCC",),
            )
        finally:
            if old_fixed is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS"] = old_fixed
            if old_weight is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS"] = old_weight

        self.assertEqual(rejected_row["enzyme_sp_verifier_bonus"], 0.0)
        self.assertAlmostEqual(accepted_row["enzyme_sp_verifier_bonus"], 1.0)
        self.assertLess(accepted_row["total_cost"], rejected_row["total_cost"])

    def test_enzyme_sp_gate_attaches_material_quality_audit(self):
        product = "CCCCCCCCCCCC"
        action = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "C",
                "rxn_smiles": "C>>CCCCCCCCCCCC",
                "score": 0.5,
                "source": "enzyme_precedent",
                "ec": "1.1.1.1",
            },
            source="enzyme_precedent",
        )
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            enzyme_sp_verifier=_AcceptingEnzymeSPVerifier(),
        )
        old_scope = os.environ.get("AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE")
        old_gate = os.environ.get("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE")
        old_gate_sources = os.environ.get("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES")
        try:
            os.environ["AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE"] = "all_enzymatic"
            os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE", None)
            os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES", None)
            allowed = planner._enzyme_sp_action_allowed(product, action)
        finally:
            if old_scope is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE"] = old_scope
            if old_gate is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] = old_gate
            if old_gate_sources is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES"] = old_gate_sources

        self.assertTrue(allowed)
        quality = action.metadata["enzyme_step_quality_v1"]
        self.assertEqual(quality["decision"], "reject")
        self.assertIn("material_sanity_failed", quality["flags"])
        board = RouteTreeState.initial(product).advance(
            leaf=product,
            action=action,
            next_open_leaves=(),
            score_delta=0.0,
        ).to_board()
        self.assertEqual(board.slots[0].evidence["enzyme_step_quality_v1"]["decision"], "reject")

    def test_enzyme_sp_material_gate_can_reject_bad_material_jump(self):
        product = "CCCCCCCCCCCC"
        action = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "C",
                "rxn_smiles": "C>>CCCCCCCCCCCC",
                "score": 0.5,
                "source": "enzyme_precedent",
                "ec": "1.1.1.1",
            },
            source="enzyme_precedent",
        )
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            enzyme_sp_verifier=_AcceptingEnzymeSPVerifier(),
        )
        old_scope = os.environ.get("AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE")
        old_gate = os.environ.get("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE")
        old_gate_sources = os.environ.get("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES")
        try:
            os.environ["AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE"] = "all_enzymatic"
            os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] = "1"
            os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES", None)
            allowed = planner._enzyme_sp_action_allowed(product, action)
        finally:
            if old_scope is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE"] = old_scope
            if old_gate is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] = old_gate
            if old_gate_sources is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES"] = old_gate_sources

        self.assertFalse(allowed)
        self.assertEqual(planner.stats.enzyme_sp_verifier_rejections, 1)
        self.assertEqual(action.metadata["enzyme_step_quality_v1"]["decision"], "reject")

    def test_enzyme_sp_material_gate_source_allowlist_skips_other_sources(self):
        product = "CCCCCCCCCCCC"
        action = CandidateAction.from_candidate(
            product,
            {
                "main_reactant": "C",
                "rxn_smiles": "C>>CCCCCCCCCCCC",
                "score": 0.5,
                "source": "chem_enzy_onmt",
                "ec": "1.1.1.1",
            },
            source="chem_enzy_onmt",
        )
        planner = NeuralGuidedAOSearch(
            retro_engine={},
            controller=None,
            enzyme_sp_verifier=_AcceptingEnzymeSPVerifier(),
        )
        old_scope = os.environ.get("AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE")
        old_gate = os.environ.get("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE")
        old_gate_sources = os.environ.get("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES")
        try:
            os.environ["AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE"] = "all_enzymatic"
            os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] = "1"
            os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES"] = "enzyme_precedent"
            allowed = planner._enzyme_sp_action_allowed(product, action)
        finally:
            if old_scope is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE"] = old_scope
            if old_gate is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] = old_gate
            if old_gate_sources is None:
                os.environ.pop("AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES", None)
            else:
                os.environ["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES"] = old_gate_sources

        self.assertTrue(allowed)
        self.assertEqual(planner.stats.enzyme_sp_verifier_rejections, 0)
        self.assertEqual(action.metadata["enzyme_step_quality_v1"]["decision"], "warn")
        self.assertIn("material_sanity_failed", action.metadata["enzyme_step_quality_v1"]["flags"])

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

    def test_graphfp_fusion_is_not_suppressed_by_large_molecule_route_metadata_alone(self):
        graphfp = _SourceRecorder("chem_enzy_graphfp_fusion", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "score": 0.9,
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
            "score": 0.8,
        }])
        tool = RetroEngineProposalTool({
            "retrochimera": chemical,
            "chem_enzy_graphfp_fusion": graphfp,
        })

        actions = tool.propose(
            "CC=O",
            ProposalContext(route_metadata={"enzymatic_only_route": True}),
            top_k=4,
        )

        self.assertIn("chem_enzy_graphfp_fusion", {action.source for action in actions})
        self.assertEqual(graphfp.calls, 1)

    def test_graphfp_fusion_remains_suppressed_for_explicit_ec_context(self):
        graphfp = _SourceRecorder("chem_enzy_graphfp_fusion", rows=[{
            "main_reactant": "CCO",
            "rxn_smiles": "CCO>>CC=O",
            "score": 0.9,
        }])
        chemical = _SourceRecorder("retrochimera", rows=[{
            "main_reactant": "CC",
            "rxn_smiles": "CC>>CC=O",
            "score": 0.8,
        }])
        tool = RetroEngineProposalTool({
            "retrochimera": chemical,
            "chem_enzy_graphfp_fusion": graphfp,
        })

        actions = tool.propose(
            "CC=O",
            ProposalContext(ec1=1, route_metadata={"enzymatic_only_route": True}),
            top_k=4,
        )

        self.assertNotIn("chem_enzy_graphfp_fusion", {action.source for action in actions})
        self.assertEqual(graphfp.calls, 0)

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

    def test_sp_v1_enzyme_result_selector_promotes_bridge_supported_enzyme_route(self):
        old_selector = os.environ.get("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR")
        old_pool = os.environ.get("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN")
        old_rank = os.environ.get("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK")
        old_keep = os.environ.get("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = "1"
            results = plan_with_route_tree(
                target="CC=O",
                retro_engine={"template_relevance": _ChemicalFirstBridgeEnzymeRetro()},
                stock_checker=lambda smi: smi in {"CCO", "C"},
                max_depth=2,
                n_results=1,
                branch_factor=4,
                expansion_budget=6,
                controller=None,
                enzyme_sp_verifier=None,
            )
        finally:
            if old_selector is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = old_selector
            if old_pool is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = old_pool
            if old_rank is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = old_rank
            if old_keep is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = old_keep

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].source, "chem_enzy_onmt")
        table = results[0].explanation.uncertainty_table
        self.assertTrue(table["sp_v1_enzyme_result_selector"]["promoted"])
        outcome = table["route_tree_final_outcome"]
        self.assertEqual(outcome["route_tree_enzyme_result_pool_target"], 5)
        self.assertGreaterEqual(outcome["solved_routes"], 2)

    def test_sp_v1_enzyme_result_selector_preserves_tied_solved_enzyme_alternative(self):
        old_selector = os.environ.get("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR")
        old_pool = os.environ.get("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN")
        old_rank = os.environ.get("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK")
        old_keep = os.environ.get("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = "5"
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            results = plan_with_route_tree(
                target="CC=O",
                retro_engine={"template_relevance": _ChemicalFirstBridgeEnzymeRetro()},
                stock_checker=lambda smi: smi in {"CCO", "C"},
                max_depth=2,
                n_results=1,
                branch_factor=4,
                expansion_budget=6,
                controller=None,
                enzyme_sp_verifier=None,
            )
        finally:
            if old_selector is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = old_selector
            if old_pool is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = old_pool
            if old_rank is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = old_rank
            if old_keep is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = old_keep

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].source, "chem_enzy_onmt")
        table = results[0].explanation.uncertainty_table
        self.assertTrue(table["sp_v1_enzyme_result_selector"]["promoted"])
        outcome = table["route_tree_final_outcome"]
        self.assertGreaterEqual(outcome["solved_routes"], 2)

    def test_sp_v1_enzyme_result_selector_keeps_pending_enzyme_after_plain_chemical_solve(self):
        old_selector = os.environ.get("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR")
        old_pool = os.environ.get("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN")
        old_rank = os.environ.get("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK")
        old_keep = os.environ.get("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = "5"
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            results = plan_with_route_tree(
                target="CC=O",
                retro_engine={"template_relevance": _PlainChemicalFirstBridgeEnzymeRetro()},
                stock_checker=lambda smi: smi in {"CCO", "C"},
                max_depth=2,
                n_results=1,
                branch_factor=4,
                expansion_budget=6,
                controller=None,
                enzyme_sp_verifier=None,
            )
        finally:
            if old_selector is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = old_selector
            if old_pool is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = old_pool
            if old_rank is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = old_rank
            if old_keep is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = old_keep

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].source, "chem_enzy_onmt")
        table = results[0].explanation.uncertainty_table
        self.assertTrue(table["sp_v1_enzyme_result_selector"]["promoted"])
        outcome = table["route_tree_final_outcome"]
        self.assertGreaterEqual(outcome["solved_routes"], 2)

    def test_sp_v1_enzyme_result_selector_cost_exception_is_opt_in(self):
        keys = [
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR",
            "AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_EXTRA_COST",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_ALLOW_COST_EXCEPTION",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_COST_EXCEPTION_MAX_EXTRA_COST",
            "AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS",
            "AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS",
            "AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS",
            "AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES",
        ]
        old_values = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_EXTRA_COST"] = "0.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = "0.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS"] = "0.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS"] = "0.0"
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_ALLOW_COST_EXCEPTION", None)
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_COST_EXCEPTION_MAX_EXTRA_COST", None)
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            results = plan_with_route_tree(
                target="CC=O",
                retro_engine={"template_relevance": _ChemicalFirstCostGapBridgeEnzymeRetro()},
                stock_checker=lambda smi: smi in {"CCO", "C"},
                max_depth=2,
                n_results=1,
                branch_factor=4,
                expansion_budget=6,
                controller=None,
                enzyme_sp_verifier=None,
            )
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].source, "template_relevance")
        table = results[0].explanation.uncertainty_table
        self.assertNotIn("sp_v1_enzyme_result_selector", table)
        selector = table["route_tree_final_outcome"]["route_tree_enzyme_result_selector"]
        self.assertEqual(selector["bridge_supported_sp_v1_enzyme_results"], 1)

    def test_sp_v1_enzyme_result_selector_cost_exception_promotes_with_explicit_cap(self):
        keys = [
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR",
            "AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_EXTRA_COST",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_ALLOW_COST_EXCEPTION",
            "AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_COST_EXCEPTION_MAX_EXTRA_COST",
            "AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS",
            "AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS",
            "AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS",
            "AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES",
        ]
        old_values = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_EXTRA_COST"] = "0.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_ALLOW_COST_EXCEPTION"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_COST_EXCEPTION_MAX_EXTRA_COST"] = "2.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = "0.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS"] = "0.0"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS"] = "0.0"
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            results = plan_with_route_tree(
                target="CC=O",
                retro_engine={"template_relevance": _ChemicalFirstCostGapBridgeEnzymeRetro()},
                stock_checker=lambda smi: smi in {"CCO", "C"},
                max_depth=2,
                n_results=1,
                branch_factor=4,
                expansion_budget=6,
                controller=None,
                enzyme_sp_verifier=None,
            )
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].source, "chem_enzy_onmt")
        table = results[0].explanation.uncertainty_table
        selector = table["sp_v1_enzyme_result_selector"]
        self.assertTrue(selector["promoted"])
        self.assertEqual(selector["selector_mode"], "cost_exception")

    def test_sp_v1_enzyme_result_selector_requires_bridge_supported_enzyme_step(self):
        old_selector = os.environ.get("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR")
        old_pool = os.environ.get("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN")
        old_keep = os.environ.get("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = "5"
            os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = "1"
            results = plan_with_route_tree(
                target="CC=O",
                retro_engine={"template_relevance": _ChemicalFirstUnsupportedEnzymeRetro()},
                stock_checker=lambda smi: smi in {"CCO", "C"},
                max_depth=2,
                n_results=1,
                branch_factor=4,
                expansion_budget=6,
                controller=None,
                enzyme_sp_verifier=None,
            )
        finally:
            if old_selector is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = old_selector
            if old_pool is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = old_pool
            if old_keep is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"] = old_keep

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].source, "template_relevance")
        table = results[0].explanation.uncertainty_table
        self.assertNotIn("sp_v1_enzyme_result_selector", table)

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

    def test_route_tree_returns_frontier_fallback_when_timeout_hits_with_partial_state(self):
        old_timeout = os.environ.get("AUTOPLANNER_ROUTE_TREE_HARD_TIMEOUT_S")
        old_frontier = os.environ.get("AUTOPLANNER_ROUTE_TREE_TIMEOUT_FRONTIER_FALLBACKS")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_HARD_TIMEOUT_S"] = "0.01"
            os.environ["AUTOPLANNER_ROUTE_TREE_TIMEOUT_FRONTIER_FALLBACKS"] = "1"
            results = plan_with_route_tree(
                target="CCCCCCCC",
                retro_engine={"retrochimera": _TimeoutFrontierRetro()},
                stock_checker=lambda smi: smi == "CCCC",
                max_depth=3,
                n_results=1,
                branch_factor=1,
                expansion_budget=4,
                controller=None,
            )
        finally:
            if old_timeout is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_HARD_TIMEOUT_S", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_HARD_TIMEOUT_S"] = old_timeout
            if old_frontier is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_TIMEOUT_FRONTIER_FALLBACKS", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_TIMEOUT_FRONTIER_FALLBACKS"] = old_frontier

        self.assertEqual(len(results), 1)
        table = results[0].explanation.uncertainty_table
        self.assertEqual(table["route_tree_search_status"], "timeout_frontier")
        self.assertGreaterEqual(table["timeout_frontier_fallbacks"], 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "CCCCCCC")

    def test_selected_enzyme_evidence_enrichment_attaches_transition_signature(self):
        old_enrich = os.environ.get("AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_ENRICHMENT")
        old_similarity = os.environ.get("AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_MIN_SIMILARITY")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_ENRICHMENT"] = "1"
            os.environ["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_MIN_SIMILARITY"] = "1.0"
            results = plan_with_route_tree(
                target="CC=O",
                retro_engine={
                    "chem_enzy_bionav": _SourceRecorder(
                        "chem_enzy_bionav",
                        rows=[{
                            "main_reactant": "CCO",
                            "rxn_smiles": "CCO>>CC=O",
                            "score": 0.9,
                            "source": "chem_enzy_onmt",
                        }],
                    )
                },
                stock_checker=lambda smi: smi == "CCO",
                max_depth=1,
                n_results=1,
                branch_factor=1,
                expansion_budget=2,
                controller=None,
            )
        finally:
            if old_enrich is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_ENRICHMENT", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_ENRICHMENT"] = old_enrich
            if old_similarity is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_MIN_SIMILARITY", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_MIN_SIMILARITY"] = old_similarity

        evidence = results[0].board.slots[0].evidence
        self.assertIn("selected_enzyme_evidence_enrichment", evidence)
        transition = evidence["selected_enzyme_transition_signature"]
        self.assertTrue(transition["valid"])
        self.assertEqual(transition["substrate_main"], "CCO")
        self.assertEqual(transition["product_main"], "CC=O")

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
