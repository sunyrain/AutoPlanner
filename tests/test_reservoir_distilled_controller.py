import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from cascade_planner.cascadeboard.route_export import route_metrics, route_results_payload
from cascade_planner.cascadeboard import CascadeBoard, RouteExplanation, RouteResult
from cascade_planner.cascadeboard.live_benchmark import refresh_recovery_metrics, refresh_route_metrics, summarize_target_results
from cascade_planner.cascadeboard.live_benchmark import _plan_one_target
from cascade_planner.eval.build_reservoir_distill_pack import build_reservoir_distill_pack
from cascade_planner.eval.build_external_reservoir_smokes import (
    _build_paroutes_reference_rows,
    _build_paroutes_split,
    _build_uspto190,
    build_external_smokes,
    run_external_smokes,
    summarize_external_smokes,
)
from cascade_planner.eval.aggregate_external_smoke_summaries import aggregate_external_smoke_summaries
from cascade_planner.eval.cache_uspto190_targets import cache_uspto190_targets
from cascade_planner.eval.reservoir_completion_audit import _external_benchmark_evidence, _promotion_gate_status
from cascade_planner.eval.reservoir_publication_readiness import build_publication_readiness_report
from cascade_planner.eval.reservoir_statistical_report import build_statistical_report
from cascade_planner.eval.build_stock_delta_source_pack import build_stock_delta_source_pack
from cascade_planner.eval.analyze_student_route_composition_gaps import analyze_student_route_composition_gaps
from cascade_planner.eval.build_native_route_replay_pack import build_native_route_replay_pack
from cascade_planner.eval.audit_stock_closed_alternatives import build_stock_closed_alternative_audit
from cascade_planner.eval.reservoir_acceptance_manifest import (
    audit_external_benchmarks,
    build_reservoir_acceptance_manifest,
    write_smoke_benchmark,
)
from cascade_planner.eval.reservoir_distill_matrix import build_reservoir_distill_matrix
from cascade_planner.eval.train_reservoir_distilled_controller import (
    DEFAULT_LOSS_WEIGHTS,
    _loss,
    _pairwise_ranking_loss,
    build_reservoir_distill_dataset,
    train_reservoir_distilled_controller,
)
from cascade_planner.route_tree import bounded_reservoir as bounded_reservoir_module
from cascade_planner.route_tree import proposals as proposals_module
from cascade_planner.route_tree.bounded_reservoir import (
    _filter_native_routes_for_quality,
    _route_dict_to_result,
    _route_runtime_stock_closed,
    append_bounded_native_reservoir,
    annotate_bounded_reservoir_payload,
)
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.reservoir_distilled import (
    ReservoirDistilledController,
    ReservoirDistilledControllerRuntime,
    reservoir_controller_feature_dim,
    reservoir_controller_feature_vector,
)
from cascade_planner.route_tree.runtime import _RUNTIME_CACHE, RouteTreeEvaluation, default_route_tree_runtime
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.route_tree.source_gate import (
    SOURCE_GROUPS,
    SOURCE_POLICY_BUDGET_LABELS,
    SourceAllocation,
    SourceGate,
    _SOURCE_GATE_CACHE,
    default_source_gate,
)


def _action(product="CCO", main="CC", aux=None, source="retrochimera", score=0.9):
    aux = ["O"] if aux is None else aux
    return CandidateAction.from_candidate(
        product,
        {
            "main_reactant": main,
            "aux_reactants": aux,
            "rxn_smiles": ".".join([main, *aux]) + f">>{product}",
            "source": source,
            "score": score,
            "rank": 1,
            "type": "addition",
        },
        source=source,
    )


def _trace_row(index=1, target="CCO"):
    good = _action(product=target)
    bad = _action(product=target, main=target, aux=[], score=0.1)
    return {
        "schema_version": "route_tree_trace.v1",
        "benchmark_index": index,
        "target_smiles": target,
        "cascade_id": f"case_{index}",
        "gt_route": [{"rxn_smiles": "CC.O>>CCO", "transformation": "addition"}],
        "elapsed_s": 0.25,
        "event": {
            "state_id": f"s{index}",
            "state": {"target": target, "open_leaves": [target], "depth": 0, "steps": []},
            "depth": 0,
            "open_leaves": [target],
            "expanded_leaf": target,
            "candidate_actions": [good.to_dict(), bad.to_dict()],
            "selected_action_key": good.canonical_key,
            "selected_next_stock_closed": True,
            "selected_next_open_leaves": 0,
            "expanded_leaf_low_yield": False,
            "proposal_diagnostics": [
                {
                    "leaf": target,
                    "proposal_budget": 4,
                    "allocation": {
                        "budget_multiplier_label": "2x",
                        "source_group_probs": {"chemical": 0.8, "fallback": 0.2},
                        "source_weights": {"retrochimera": 1.0},
                    },
                    "sources": {
                        "retrochimera": {
                            "calls": 1,
                            "queried": True,
                            "allocated_budget": 4,
                            "requested_k_total": 4,
                            "raw_returned": 2,
                            "final_returned": 2,
                            "latency_ms_total": 12.0,
                        }
                    },
                    "raw_actions": 2,
                    "final_actions": 2,
                }
            ],
            "outcome": {"search_status": "solved", "solved_routes": 1},
        },
    }


def _chem_payload(target="CCO"):
    return {
        "targets": [
            {
                "index": 0,
                "target_smiles": target,
                "routes": [
                    {
                        "steps": [
                            {
                                "product_smiles": target,
                                "reactant_smiles": ["BAD"],
                                "rxn_smiles": f"BAD>>{target}",
                                "stock_status": {"BAD": False},
                            }
                        ]
                    },
                    {
                        "steps": [
                            {
                                "product_smiles": target,
                                "reactant_smiles": ["CC", "O"],
                                "rxn_smiles": "CC.O>>CCO",
                                "stock_status": {"CC": True, "O": True},
                            }
                        ],
                        "score": 0.8,
                        "route_rank": 2,
                    },
                ],
            }
        ]
    }


class _FixedSourceGate(SourceGate):
    def __init__(self):
        self.observed = 0

    def allocate(self, product, *, context, available_sources, total_budget):
        del product, context
        source = available_sources[0]
        return SourceAllocation(
            source_weights={source: 1.0},
            source_budgets={source: int(total_budget or 1)},
            fallback_budget=0,
            source_group_probs={"chemical": 1.0},
            budget_multiplier=1.0,
            budget_multiplier_label="1x",
            policy_confidence=0.91,
            policy_reason="delegate_source_gate",
            selected_source_group="chemical",
        )

    def observe(self, *, product, context, allocation, diagnostics):
        del product, context, allocation, diagnostics
        self.observed += 1


class _FixedRouteRuntime:
    def evaluate(self, state, leaf, actions, *, stock_checker=None):
        del state, leaf, stock_checker
        return RouteTreeEvaluation(
            action_scores=[10.0 + idx for idx, _action in enumerate(actions)],
            node_scores=[],
            route_value=0.8,
            stock_closed_prob=0.7,
            model_active=True,
            reason="delegate_runtime",
        )

    def score_open_leaves(self, state, leaves, *, stock_checker=None):
        del state, stock_checker
        return RouteTreeEvaluation(
            action_scores=[],
            node_scores=[20.0 + idx for idx, _leaf in enumerate(leaves)],
            route_value=0.6,
            stock_closed_prob=0.5,
            model_active=True,
            reason="delegate_leaf_runtime",
        )


def _controller_checkpoint(
    path: Path,
    *,
    n_bits=16,
    hidden_dim=16,
    bias_group: str = "fallback",
    bias_value: float = 8.0,
    group_biases: dict[str, float] | None = None,
) -> Path:
    input_dim = reservoir_controller_feature_dim(n_bits=n_bits)
    model = ReservoirDistilledController(
        input_dim,
        hidden_dim=hidden_dim,
        n_source_groups=len(SOURCE_GROUPS),
        n_budget_labels=len(SOURCE_POLICY_BUDGET_LABELS),
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        if group_biases is None:
            group_idx = SOURCE_GROUPS.index(bias_group)
            model.source_group_head.bias[group_idx] = float(bias_value)
        else:
            for group, value in group_biases.items():
                model.source_group_head.bias[SOURCE_GROUPS.index(group)] = float(value)
        model.budget_head.bias[1] = 8.0
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "input_dim": input_dim,
                "n_bits": n_bits,
                "hidden_dim": hidden_dim,
                "dropout": 0.10,
                "source_groups": list(SOURCE_GROUPS),
                "budget_labels": list(SOURCE_POLICY_BUDGET_LABELS),
                "min_confidence": 0.20,
            },
            "feature_schema": {
                "input_dim": input_dim,
                "n_bits": n_bits,
                "source_groups": list(SOURCE_GROUPS),
                "budget_labels": list(SOURCE_POLICY_BUDGET_LABELS),
            },
        },
        path,
    )
    return path


class ReservoirDistilledControllerTest(unittest.TestCase):
    def test_pack_builder_preserves_duplicate_rows_eval_only_and_teacher_rewards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            eval_trace = root / "eval_trace.jsonl"
            bench = root / "bench.json"
            chem = root / "chem.json"
            out = root / "distill"
            rows = [_trace_row(index=1), _trace_row(index=2)]
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            eval_trace.write_text(json.dumps(_trace_row(index=0)) + "\n", encoding="utf-8")
            bench.write_text(
                json.dumps([
                    {"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]},
                    {"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]},
                    {"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]},
                ]),
                encoding="utf-8",
            )
            chem.write_text(json.dumps(_chem_payload()), encoding="utf-8")

            manifest = build_reservoir_distill_pack(
                trace_paths=[trace],
                eval_trace_paths=[eval_trace],
                benchmark_path=bench,
                chem_enzy_path=chem,
                output_dir=out,
                native_topk=1,
                native_selection="rank_plus_stock",
            )
            train_rows = _read_jsonl(Path(manifest["files"]["train"]))
            eval_rows = _read_jsonl(Path(manifest["files"]["eval"]))
            all_rows = train_rows + _read_jsonl(Path(manifest["files"]["val"]))

        self.assertEqual(len(all_rows), 4)
        self.assertEqual(len(eval_rows), 2)
        self.assertTrue(all(row["eval_only"] for row in eval_rows))
        exact_rows = [row for row in all_rows if row["teacher_exact_hit"]]
        self.assertTrue(exact_rows)
        self.assertTrue(exact_rows[0]["teacher_gt_reactant_hit"])
        self.assertTrue(exact_rows[0]["teacher_stock_closed"])
        self.assertEqual(exact_rows[0]["teacher_value_policy"], "reaction_cost_and_or.v1")
        self.assertGreater(exact_rows[0]["teacher_route_value"], 0.0)
        self.assertLessEqual(exact_rows[0]["teacher_route_value"], 1.0)
        self.assertIn("teacher_route_cost", exact_rows[0])
        self.assertEqual({row["budget_label"] for row in all_rows}, {"2x"})

    def test_stock_closed_autoplanner_keeps_only_one_native_contrast_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            bench = root / "bench.json"
            chem = root / "chem.json"
            auto = root / "auto.json"
            out = root / "distill"
            trace.write_text(json.dumps(_trace_row(index=0)) + "\n", encoding="utf-8")
            bench.write_text(json.dumps([{"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]}]), encoding="utf-8")
            chem.write_text(json.dumps(_chem_payload()), encoding="utf-8")
            auto.write_text(
                json.dumps({
                    "targets": [
                        {
                            "index": 0,
                            "target_smiles": "CCO",
                            "metrics": {"strict_stock_solve_any": True},
                            "planner_output": {"routes": [{"steps": []}]},
                        }
                    ]
                }),
                encoding="utf-8",
            )
            manifest = build_reservoir_distill_pack(
                trace_paths=[trace],
                benchmark_path=bench,
                autoplanner_path=auto,
                chem_enzy_path=chem,
                output_dir=out,
                native_topk=5,
            )
            rows = _read_jsonl(Path(manifest["files"]["val"]))

        self.assertTrue(rows)
        self.assertEqual(rows[0]["teacher_route_count"], 2)

    def test_train_checkpoint_load_runtime_and_feature_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            bench = root / "bench.json"
            chem = root / "chem.json"
            out = root / "distill"
            trace.write_text(
                "\n".join(json.dumps(_trace_row(index=i)) for i in [0, 1, 2]),
                encoding="utf-8",
            )
            bench.write_text(
                json.dumps([{"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "CC.O>>CCO"}]} for _ in range(3)]),
                encoding="utf-8",
            )
            chem.write_text(json.dumps(_chem_payload()), encoding="utf-8")
            manifest = build_reservoir_distill_pack(
                trace_paths=[trace],
                benchmark_path=bench,
                chem_enzy_path=chem,
                output_dir=out,
                native_topk=1,
            )
            report = train_reservoir_distilled_controller(
                pack_path=Path(manifest["files"]["train"]),
                val_pack_path=Path(manifest["files"]["val"]),
                output_path=root / "controller.pt",
                report_path=root / "controller.json",
                epochs=1,
                batch_size=2,
                n_bits=16,
                hidden_dim=16,
                device="cpu",
            )
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER")
            old_path = os.environ.get("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER")
            try:
                _RUNTIME_CACHE.clear()
                os.environ["AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER"] = str(root / "controller.pt")
                runtime = default_route_tree_runtime()
                action = _action()
                result = runtime.evaluate(RouteTreeState.initial("CCO"), "CCO", [action])
            finally:
                _RUNTIME_CACHE.clear()
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER"] = old_enable
                if old_path is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER"] = old_path

        self.assertIn("best_val_loss", report)
        self.assertTrue(result.model_active)
        self.assertEqual(len(result.action_scores), 1)
        vector = reservoir_controller_feature_vector(product="CCO", leaf="CCO", candidate={}, n_bits=16)
        self.assertEqual(len(vector), reservoir_controller_feature_dim(n_bits=16))
        padded = reservoir_controller_feature_vector(product="CCO", leaf="CCO", candidate={}, n_bits=16, input_dim=len(vector) + 3)
        self.assertEqual(len(padded), len(vector) + 3)

    def test_reservoir_source_fallback_uses_delegate_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = _controller_checkpoint(root / "controller.pt", bias_group="fallback")
            delegate = _FixedSourceGate()
            runtime = ReservoirDistilledControllerRuntime(checkpoint, fallback_source_gate=delegate)
            allocation = runtime.allocate(
                "CCO",
                context=None,
                available_sources=["retrochimera"],
                total_budget=4,
            )
            runtime.observe(product="CCO", context=None, allocation=allocation, diagnostics={})

        self.assertEqual(allocation.policy_reason, "reservoir_distilled_fallback:fallback_group:0.998")
        self.assertEqual(allocation.source_budgets, {"retrochimera": 4})
        self.assertEqual(allocation.selected_source_group, "chemical")
        self.assertEqual(delegate.observed, 1)

    def test_reservoir_min_confidence_env_can_force_delegate_gate(self):
        old_override = os.environ.get("AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE")
        old_threshold = os.environ.get("AUTOPLANNER_RESERVOIR_MIN_CONFIDENCE")
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                checkpoint = _controller_checkpoint(
                    root / "controller.pt",
                    bias_group="chemical",
                    bias_value=1.0,
                )
                delegate = _FixedSourceGate()
                os.environ["AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE"] = "1"
                os.environ.pop("AUTOPLANNER_RESERVOIR_MIN_CONFIDENCE", None)
                runtime = ReservoirDistilledControllerRuntime(checkpoint, fallback_source_gate=delegate)
                controller_allocation = runtime.allocate(
                    "CCO",
                    context=None,
                    available_sources=["enzyformer", "retrochimera"],
                    total_budget=4,
                )

                os.environ["AUTOPLANNER_RESERVOIR_MIN_CONFIDENCE"] = "0.65"
                runtime = ReservoirDistilledControllerRuntime(checkpoint, fallback_source_gate=delegate)
                fallback_allocation = runtime.allocate(
                    "CCO",
                    context=None,
                    available_sources=["enzyformer", "retrochimera"],
                    total_budget=4,
                )
        finally:
            if old_override is None:
                os.environ.pop("AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE", None)
            else:
                os.environ["AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE"] = old_override
            if old_threshold is None:
                os.environ.pop("AUTOPLANNER_RESERVOIR_MIN_CONFIDENCE", None)
            else:
                os.environ["AUTOPLANNER_RESERVOIR_MIN_CONFIDENCE"] = old_threshold

        self.assertEqual(controller_allocation.policy_reason, "reservoir_distilled")
        self.assertGreater(controller_allocation.source_budgets.get("retrochimera", 0), 0)
        self.assertIn("reservoir_distilled_fallback:low_confidence", fallback_allocation.policy_reason)
        self.assertEqual(fallback_allocation.source_budgets, {"enzyformer": 4})

    def test_reservoir_ambiguous_source_env_can_force_delegate_gate(self):
        keys = [
            "AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE",
            "AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_FALLBACK",
            "AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MAX_CONFIDENCE",
            "AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MAX_MARGIN",
            "AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MIN_ALT_PROB",
        ]
        old_values = {key: os.environ.get(key) for key in keys}
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                checkpoint = _controller_checkpoint(
                    root / "controller.pt",
                    group_biases={"chemical": 2.0, "enzymatic": 2.1},
                )
                delegate = _FixedSourceGate()
                os.environ["AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE"] = "1"
                os.environ.pop("AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_FALLBACK", None)
                runtime = ReservoirDistilledControllerRuntime(checkpoint, fallback_source_gate=delegate)
                controller_allocation = runtime.allocate(
                    "CCO",
                    context=None,
                    available_sources=["enzyformer", "retrochimera"],
                    total_budget=4,
                )

                os.environ["AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_FALLBACK"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MAX_CONFIDENCE"] = "0.65"
                os.environ["AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MAX_MARGIN"] = "0.20"
                os.environ["AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MIN_ALT_PROB"] = "0.20"
                runtime = ReservoirDistilledControllerRuntime(checkpoint, fallback_source_gate=delegate)
                fallback_allocation = runtime.allocate(
                    "CCO",
                    context=None,
                    available_sources=["enzyformer", "retrochimera"],
                    total_budget=4,
                )
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        self.assertEqual(controller_allocation.policy_reason, "reservoir_distilled")
        self.assertEqual(controller_allocation.selected_source_group, "enzymatic")
        self.assertIn("reservoir_distilled_fallback:ambiguous_source", fallback_allocation.policy_reason)
        self.assertEqual(fallback_allocation.source_budgets, {"enzyformer": 4})

    def test_reservoir_can_split_group_probability_across_sources(self):
        old_override = os.environ.get("AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE")
        old_split = os.environ.get("AUTOPLANNER_RESERVOIR_SPLIT_GROUP_SOURCE_WEIGHT")
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                checkpoint = _controller_checkpoint(
                    root / "controller.pt",
                    bias_group="enzymatic",
                    bias_value=1.0,
                )
                os.environ["AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE"] = "1"
                os.environ.pop("AUTOPLANNER_RESERVOIR_SPLIT_GROUP_SOURCE_WEIGHT", None)
                runtime = ReservoirDistilledControllerRuntime(checkpoint)
                duplicated = runtime.allocate(
                    "CCO",
                    context=None,
                    available_sources=["retrochimera", "enzyformer", "enzexpand"],
                    total_budget=4,
                )

                os.environ["AUTOPLANNER_RESERVOIR_SPLIT_GROUP_SOURCE_WEIGHT"] = "1"
                runtime = ReservoirDistilledControllerRuntime(checkpoint)
                split = runtime.allocate(
                    "CCO",
                    context=None,
                    available_sources=["retrochimera", "enzyformer", "enzexpand"],
                    total_budget=4,
                )
        finally:
            if old_override is None:
                os.environ.pop("AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE", None)
            else:
                os.environ["AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE"] = old_override
            if old_split is None:
                os.environ.pop("AUTOPLANNER_RESERVOIR_SPLIT_GROUP_SOURCE_WEIGHT", None)
            else:
                os.environ["AUTOPLANNER_RESERVOIR_SPLIT_GROUP_SOURCE_WEIGHT"] = old_split

        self.assertGreater(split.source_weights["retrochimera"], duplicated.source_weights["retrochimera"])
        self.assertLess(split.source_weights["enzyformer"], duplicated.source_weights["enzyformer"])

    def test_reservoir_runtime_blends_delegate_action_and_leaf_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = _controller_checkpoint(root / "controller.pt", bias_group="chemical")
            runtime = ReservoirDistilledControllerRuntime(
                checkpoint,
                fallback_runtime=_FixedRouteRuntime(),
            )
            state = RouteTreeState.initial("CCO")
            action_eval = runtime.evaluate(state, "CCO", [_action(), _action(main="C")])
            leaf_eval = runtime.score_open_leaves(state, ["CCO", "CCN"])

        self.assertTrue(action_eval.model_active)
        self.assertIn("reservoir_distilled+delegate_runtime", action_eval.reason)
        self.assertGreater(action_eval.action_scores[0], 9.9)
        self.assertLess(action_eval.action_scores[0], 10.1)
        self.assertEqual(len(action_eval.action_scores), 2)
        self.assertIn("reservoir_distilled+delegate_leaf_runtime", leaf_eval.reason)
        self.assertGreater(leaf_eval.node_scores[0], 19.9)
        self.assertLess(leaf_eval.node_scores[0], 20.1)

    def test_training_rejects_eval_only_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "eval_only.jsonl"
            row = {
                "target_smiles": "CCO",
                "leaf": "CCO",
                "source": "retrochimera",
                "candidate_reaction": "CC.O>>CCO",
                "reactants": ["CC", "O"],
                "budget_label": "1x",
                "teacher_route_value": 1.0,
                "teacher_action_value": 1.0,
                "eval_only": True,
            }
            pack.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                train_reservoir_distilled_controller(
                    pack_path=pack,
                    val_pack_path=pack,
                    output_path=root / "controller.pt",
                    report_path=root / "controller.json",
                    epochs=1,
                    n_bits=16,
                    hidden_dim=16,
                    device="cpu",
                )

    def test_stock_delta_source_pack_adds_source_only_rows_without_eval_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            run = root / "run.json"
            out = root / "augmented.jsonl"
            report = root / "report.json"
            base_row = {
                "state_id": "s1",
                "target_id": "t1",
                "target_smiles": "CCO",
                "benchmark_index": 7,
                "depth": 0,
                "remaining_depth": 2,
                "leaf": "CCO",
                "source": "retrochimera",
                "source_policy_group": "chemical",
                "candidate_reaction": "CC.O>>CCO",
                "reactants": ["CC", "O"],
                "budget_label": "1x",
                "teacher_source_group_distribution": {"chemical": 1.0},
                "teacher_route_value": 1.0,
                "teacher_action_value": 1.0,
                "eval_only": False,
            }
            pack.write_text(json.dumps(base_row) + "\n", encoding="utf-8")
            run.write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "index": 0,
                                "target_smiles": "CCO",
                                "metrics": {"strict_stock_solve_any": True},
                                "route_recovery": {
                                    "exact_reaction_in_route_pool": True,
                                    "gt_reactant_in_route_pool": True,
                                },
                                "planner_output": {
                                    "routes": [
                                        {
                                            "steps": [
                                                {
                                                    "source": "v3_retrieval",
                                                    "stock_status": {"CC": True, "O": True},
                                                }
                                            ]
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = build_stock_delta_source_pack(
                pack_path=pack,
                run_path=run,
                output_pack=out,
                report_path=report,
            )
            rows = _read_jsonl(out)

        self.assertEqual(result["synthetic_rows"], 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1]["source_only"])
        self.assertFalse(rows[1]["eval_only"])
        self.assertEqual(rows[1]["teacher_source_group_distribution"]["retrieval"], 1.0)

    def test_stock_closed_rows_keep_cost_derived_value_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            row = _stock_closed_pack_row()
            pack.write_text(json.dumps(row) + "\n", encoding="utf-8")
            base = build_reservoir_distill_dataset(pack, n_bits=16)

        self.assertAlmostEqual(float(base.route_value[0]), 0.50, places=6)
        self.assertAlmostEqual(float(base.action_value[0]), 0.50, places=6)
        self.assertEqual(base.feature_schema["stock_closed_head_weight"], 1.0)

    def test_stock_closed_head_weight_replays_stock_positive_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            stock_row = _stock_closed_pack_row()
            nonstock_row = dict(stock_row)
            nonstock_row["teacher_stock_closed"] = False
            nonstock_row["reactants"] = ["CCN"]
            pack.write_text(
                "\n".join(json.dumps(row) for row in [stock_row, nonstock_row]) + "\n",
                encoding="utf-8",
            )
            weighted = build_reservoir_distill_dataset(pack, n_bits=16, stock_closed_head_weight=3.0)

        self.assertAlmostEqual(float(weighted.head_weight[0]), 3.0, places=6)
        self.assertAlmostEqual(float(weighted.head_weight[1]), 1.0, places=6)
        self.assertEqual(weighted.feature_schema["stock_closed_head_weight"], 3.0)

    def test_action_pairwise_ranking_is_state_local_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for state_id, value in [("s1", 1.0), ("s1", 0.0), ("s2", 1.0), ("s2", 0.0)]:
                row = _stock_closed_pack_row()
                row["state_id"] = state_id
                row["teacher_action_value"] = value
                row["teacher_route_value"] = value
                row["teacher_stock_closed"] = False
                row["reactants"] = [f"C{state_id}{value}"]
                rows.append(row)
            pack.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            dataset = build_reservoir_distill_dataset(pack, n_bits=16)
            global_dataset = build_reservoir_distill_dataset(pack, n_bits=16, pairwise_group_key="")

        self.assertEqual(dataset.feature_schema["pairwise_group_key"], "state_id")
        self.assertEqual(list(dataset.pair_group), [0, 0, 1, 1])
        self.assertEqual(list(global_dataset.pair_group), [0, 0, 0, 0])

    def test_pairwise_ranking_loss_ignores_cross_state_pairs(self):
        scores = torch.tensor([0.0, 1.0], dtype=torch.float32)
        labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
        groups = torch.tensor([0, 1], dtype=torch.long)

        local_loss = _pairwise_ranking_loss(scores, labels, group_ids=groups)
        global_loss = _pairwise_ranking_loss(scores, labels)

        self.assertAlmostEqual(float(local_loss), 0.0, places=6)
        self.assertGreater(float(global_loss), 0.0)

    def test_route_pairwise_ranking_loss_prefers_teacher_order(self):
        weights = {key: 0.0 for key in DEFAULT_LOSS_WEIGHTS}
        weights["route_rank_regression"] = 1.0
        source_logits = torch.zeros((2, len(SOURCE_GROUPS)), dtype=torch.float32)
        budget_logits = torch.zeros((2, len(SOURCE_POLICY_BUDGET_LABELS)), dtype=torch.float32)
        batch = [
            torch.zeros((2, 4), dtype=torch.float32),  # x
            torch.zeros((2, 4), dtype=torch.float32),  # source_x
            torch.zeros(2, dtype=torch.long),  # source_y
            torch.zeros(2, dtype=torch.long),  # budget_y
            torch.tensor([[1.0] + [0.0] * (len(SOURCE_GROUPS) - 1)] * 2, dtype=torch.float32),  # source_dist
            torch.ones(2, dtype=torch.float32),  # source_weight
            torch.ones(2, dtype=torch.float32),  # head_weight
            torch.zeros(2, dtype=torch.long),  # pair_group
            torch.zeros(2, dtype=torch.float32),  # leaf_y
            torch.zeros(2, dtype=torch.float32),  # action_y
            torch.tensor([1.0, 0.0], dtype=torch.float32),  # route_y
            torch.zeros(2, dtype=torch.float32),  # stock_y
            torch.zeros(2, dtype=torch.float32),  # latency_y
        ]
        ordered_out = {
            "source_group_logits": source_logits,
            "budget_logits": budget_logits,
            "action_value": torch.zeros(2, dtype=torch.float32),
            "route_rerank_value": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "stock_dead_end_logit": torch.zeros(2, dtype=torch.float32),
            "latency_cost": torch.zeros(2, dtype=torch.float32),
            "leaf_value": torch.zeros(2, dtype=torch.float32),
            "_source_view": {
                "source_group_logits": source_logits,
                "budget_logits": budget_logits,
            },
        }
        misordered_out = dict(ordered_out)
        misordered_out["route_rerank_value"] = torch.tensor([0.0, 1.0], dtype=torch.float32)

        ordered_loss = _loss(ordered_out, batch, weights=weights)
        misordered_loss = _loss(misordered_out, batch, weights=weights)

        self.assertLess(float(ordered_loss), float(misordered_loss))

    def test_training_features_hide_teacher_labels_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            row = {
                "target_smiles": "CCO",
                "leaf": "CCO",
                "source": "retrochimera",
                "candidate_reaction": "CC.O>>CCO",
                "reactants": ["CC", "O"],
                "budget_label": "1x",
                "reservoir_rank": 10,
                "teacher_route_rank": 1,
                "teacher_route_value": 1.5,
                "teacher_action_value": 1.5,
                "teacher_stock_closed": True,
                "teacher_exact_hit": True,
                "teacher_gt_reactant_hit": True,
                "eval_only": False,
            }
            pack.write_text(json.dumps(row) + "\n", encoding="utf-8")
            default = build_reservoir_distill_dataset(pack, n_bits=16)
            diagnostic = build_reservoir_distill_dataset(
                pack,
                n_bits=16,
                include_teacher_label_features=True,
            )

        self.assertGreater(float(default.x[0, -6]), 0.0)
        self.assertTrue(all(float(value) == 0.0 for value in default.x[0, -5:]))
        self.assertTrue(any(float(value) > 0.0 for value in diagnostic.x[0, -5:]))
        self.assertFalse(default.feature_schema["include_teacher_label_features"])
        self.assertTrue(diagnostic.feature_schema["include_teacher_label_features"])

    def test_stock_dead_end_label_uses_nonclosed_teacher_route_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = [
                {
                    "target_smiles": "CCO",
                    "leaf": "CCO",
                    "source": "retrochimera",
                    "candidate_reaction": "N>>CCO",
                    "reactants": ["N"],
                    "budget_label": "1x",
                    "teacher_route_rank": 1,
                    "teacher_stock_closed": False,
                    "failure_labels": [],
                    "eval_only": False,
                },
                {
                    "target_smiles": "CCO",
                    "leaf": "CCO",
                    "source": "retrochimera",
                    "candidate_reaction": "CC.O>>CCO",
                    "reactants": ["CC", "O"],
                    "budget_label": "1x",
                    "teacher_route_rank": 2,
                    "teacher_stock_closed": True,
                    "failure_labels": ["stock_dead_end"],
                    "eval_only": False,
                },
            ]
            pack.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            dataset = build_reservoir_distill_dataset(pack, n_bits=16)

        self.assertEqual(float(dataset.stock_dead_end[0]), 1.0)
        self.assertEqual(float(dataset.stock_dead_end[1]), 0.0)

    def test_missing_checkpoint_falls_back_to_heuristics_with_reason(self):
        old_enable = os.environ.get("AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER")
        old_path = os.environ.get("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER")
        try:
            _SOURCE_GATE_CACHE.clear()
            _RUNTIME_CACHE.clear()
            missing = "/tmp/autoplanner_missing_reservoir_controller.pt"
            os.environ["AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER"] = "1"
            os.environ["AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER"] = missing
            gate = default_source_gate()
            allocation = gate.allocate("CCO", context=None, available_sources=["retrochimera"], total_budget=4)
            runtime = default_route_tree_runtime()
            result = runtime.evaluate(RouteTreeState.initial("CCO"), "CCO", [_action()])
        finally:
            _SOURCE_GATE_CACHE.clear()
            _RUNTIME_CACHE.clear()
            if old_enable is None:
                os.environ.pop("AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER", None)
            else:
                os.environ["AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER"] = old_enable
            if old_path is None:
                os.environ.pop("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER", None)
            else:
                os.environ["AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER"] = old_path

        self.assertIn("reservoir_distilled_fallback", allocation.policy_reason)
        self.assertIn("reservoir_distilled_fallback", allocation.fallback_reason)
        self.assertEqual(allocation.to_dict()["fallback_reason"], allocation.fallback_reason)
        self.assertFalse(result.model_active)
        self.assertIn("reservoir_distilled_fallback", result.reason)

    def test_bounded_reservoir_records_broad_reservoir_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(_chem_payload()), encoding="utf-8")
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "1"
                results = append_bounded_native_reservoir(
                    target="CCO",
                    results=[],
                    stock_checker=lambda smi: smi in {"CC", "O"},
                    max_depth=2,
                    n_results=3,
                )
                payload = route_results_payload("CCO", results, stock_checker=lambda smi: smi in {"CC", "O"})
                annotate_bounded_reservoir_payload(payload)
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk

        self.assertTrue(results)
        self.assertIn("broad_reservoir", payload)
        self.assertEqual(payload["broad_reservoir"]["native_route_count"], 1)

    def test_bounded_reservoir_runs_when_stock_risk_high_even_with_stock_closed_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(_chem_payload()), encoding="utf-8")
            board = CascadeBoard.from_n_steps(0, "CCO")
            stock_closed = RouteResult(
                board=board,
                risk_vector={"stock_dead_end": 0.9},
                explanation=RouteExplanation(uncertainty_table={"stock_risk": 0.9}),
            )
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "1"
                results = append_bounded_native_reservoir(
                    target="CCO",
                    results=[stock_closed],
                    stock_checker=lambda smi: True,
                    max_depth=2,
                    n_results=1,
                )
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk

        self.assertEqual(len(results), 2)

    def test_bounded_reservoir_payload_miss_falls_back_to_native_collect(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(_chem_payload(target="CCO")), encoding="utf-8")
            collected_routes = _chem_payload(target="NCC")["targets"][0]["routes"]
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "1"
                with mock.patch.object(
                    bounded_reservoir_module,
                    "_collect_native_routes",
                    return_value=collected_routes,
                ) as collect:
                    results = append_bounded_native_reservoir(
                        target="NCC",
                        results=[],
                        stock_checker=lambda smi: smi in {"CC", "O"},
                        max_depth=2,
                        n_results=3,
                    )
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk

        collect.assert_called_once()
        self.assertTrue(results)
        self.assertEqual(results[-1].board.slots[0].product, "NCC")

    def test_bounded_reservoir_skips_online_collect_after_prior_elapsed_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(_chem_payload(target="CCO")), encoding="utf-8")
            stale_result = RouteResult(
                board=CascadeBoard.from_n_steps(0, "NCC"),
                explanation=RouteExplanation(uncertainty_table={"elapsed_s": 17.5}),
            )
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            old_elapsed = os.environ.get("AUTOPLANNER_RESERVOIR_ONLINE_COLLECT_MAX_PRIOR_ELAPSED_S")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "1"
                os.environ.pop("AUTOPLANNER_RESERVOIR_ONLINE_COLLECT_MAX_PRIOR_ELAPSED_S", None)
                with mock.patch.object(
                    bounded_reservoir_module,
                    "_collect_native_routes",
                    return_value=_chem_payload(target="NCC")["targets"][0]["routes"],
                ) as collect:
                    results = append_bounded_native_reservoir(
                        target="NCC",
                        results=[stale_result],
                        stock_checker=lambda _smi: False,
                        max_depth=2,
                        n_results=1,
                    )
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk
                if old_elapsed is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_ONLINE_COLLECT_MAX_PRIOR_ELAPSED_S", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_ONLINE_COLLECT_MAX_PRIOR_ELAPSED_S"] = old_elapsed

        collect.assert_not_called()
        self.assertEqual(results, [stale_result])

    def test_bounded_reservoir_rank_plus_stock_prefers_runtime_stock(self):
        payload = {
            "targets": [
                {
                    "target_smiles": "CCO",
                    "routes": [
                        {
                            "steps": [
                                {
                                    "product_smiles": "CCO",
                                    "reactant_smiles": ["N"],
                                    "rxn_smiles": "N>>CCO",
                                    "stock_status": {"N": True},
                                }
                            ],
                            "route_rank": 1,
                        },
                        {
                            "steps": [
                                {
                                    "product_smiles": "CCO",
                                    "reactant_smiles": ["CC"],
                                    "rxn_smiles": "CC>>CCO",
                                    "stock_status": {"CC": False},
                                }
                            ],
                            "route_rank": 2,
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(payload), encoding="utf-8")
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            old_selection = os.environ.get("AUTOPLANNER_RESERVOIR_SELECTION")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_SELECTION"] = "rank_plus_stock"
                results = append_bounded_native_reservoir(
                    target="CCO",
                    results=[],
                    stock_checker=lambda smi: smi == "CC",
                    max_depth=2,
                    n_results=1,
                )
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk
                if old_selection is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_SELECTION", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_SELECTION"] = old_selection

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "CC")

    def test_bounded_reservoir_quality_filter_drops_self_loop_stock_shortcut(self):
        good = {
            "steps": [
                {
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CC"],
                    "rxn_smiles": "CC>>CCO",
                    "scores": {"confidence": 0.8},
                    "stock_status": {"CC": True},
                }
            ]
        }
        bad = {
            "steps": [
                {
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CCO"],
                    "rxn_smiles": "CCO>>CCO",
                    "scores": {"confidence": 0.001},
                    "stock_status": {"CCO": True},
                }
            ]
        }
        with mock.patch.dict(os.environ, {"AUTOPLANNER_RESERVOIR_QUALITY_FILTER": "1"}):
            filtered = _filter_native_routes_for_quality(
                [bad, good],
                target="CCO",
                stock_checker=lambda smi: smi in {"CC", "CCO"},
            )

        self.assertEqual(filtered, [good])

    def test_bounded_reservoir_drops_metadata_stock_when_runtime_stock_absent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(_chem_payload()), encoding="utf-8")
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            old_require = os.environ.get("AUTOPLANNER_RESERVOIR_REQUIRE_RUNTIME_STOCK_WHEN_NO_STOCK")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "1"
                os.environ.pop("AUTOPLANNER_RESERVOIR_REQUIRE_RUNTIME_STOCK_WHEN_NO_STOCK", None)
                results = append_bounded_native_reservoir(
                    target="CCO",
                    results=[],
                    stock_checker=lambda _smi: False,
                    max_depth=2,
                    n_results=1,
                )
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk
                if old_require is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_REQUIRE_RUNTIME_STOCK_WHEN_NO_STOCK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_REQUIRE_RUNTIME_STOCK_WHEN_NO_STOCK"] = old_require

        self.assertEqual(results, [])

    def test_bounded_reservoir_trust_native_stock_adds_stock_override(self):
        route = {
            "steps": [
                {
                    "product_smiles": "CCO",
                    "reactant_smiles": ["N"],
                    "rxn_smiles": "N>>CCO",
                    "stock_status": {"N": True},
                }
            ],
            "route_rank": 1,
        }
        old_trust = os.environ.get("AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK")
        try:
            os.environ.pop("AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK", None)
            self.assertFalse(
                _route_runtime_stock_closed(
                    route,
                    target="CCO",
                    stock_checker=lambda _smi: False,
                )
            )

            os.environ["AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK"] = "1"
            self.assertTrue(
                _route_runtime_stock_closed(
                    route,
                    target="CCO",
                    stock_checker=lambda _smi: False,
                )
            )
            result = _route_dict_to_result(target="CCO", route=route, rank=1, topk=1)
            metrics = route_metrics(result.board, stock_checker=lambda _smi: False)
        finally:
            if old_trust is None:
                os.environ.pop("AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK", None)
            else:
                os.environ["AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK"] = old_trust

        self.assertEqual(metrics["stock_override_count"], 1)
        self.assertTrue(metrics["strict_stock_solve"])

    def test_bounded_reservoir_can_keep_min_native_contrast_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(_chem_payload()), encoding="utf-8")
            board = CascadeBoard.from_n_steps(1, "CCO")
            board.slots[0].product = "CCO"
            board.slots[0].main_reactant = "CC"
            board.slots[0].aux_reactants = ["O"]
            board.slots[0].reaction_smiles = "CC.O>>CCO"
            stock_closed = RouteResult(board=board)
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            old_min = os.environ.get("AUTOPLANNER_RESERVOIR_MIN_NATIVE_ROUTES")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "5"
                os.environ["AUTOPLANNER_RESERVOIR_MIN_NATIVE_ROUTES"] = "1"
                results = append_bounded_native_reservoir(
                    target="CCO",
                    results=[stock_closed],
                    stock_checker=lambda smi: True,
                    max_depth=2,
                    n_results=1,
                )
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk
                if old_min is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_MIN_NATIVE_ROUTES", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_MIN_NATIVE_ROUTES"] = old_min

        self.assertEqual(len(results), 2)

    def test_bounded_reservoir_runs_when_controller_fallback_group_high(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            native.write_text(json.dumps(_chem_payload()), encoding="utf-8")
            board = CascadeBoard.from_n_steps(0, "CCO")
            stock_closed = RouteResult(
                board=board,
                explanation=RouteExplanation(
                    uncertainty_table={
                        "route_tree_controller_active": True,
                        "route_tree_source_budgets": [
                            {
                                "proposal_gate": {
                                    "policy_reason": "reservoir_distilled_fallback:fallback_group:0.700",
                                    "selected_source_group": "chemical",
                                    "source_group_probs": {"chemical": 1.0},
                                }
                            }
                        ],
                    }
                ),
            )
            old_enable = os.environ.get("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")
            old_payload = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD")
            old_topk = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_TOPK")
            try:
                os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = "1"
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = str(native)
                os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = "1"
                results = append_bounded_native_reservoir(
                    target="CCO",
                    results=[stock_closed],
                    stock_checker=lambda smi: True,
                    max_depth=2,
                    n_results=1,
                )
            finally:
                if old_enable is None:
                    os.environ.pop("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR", None)
                else:
                    os.environ["AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR"] = old_enable
                if old_payload is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD"] = old_payload
                if old_topk is None:
                    os.environ.pop("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", None)
                else:
                    os.environ["AUTOPLANNER_RESERVOIR_NATIVE_TOPK"] = old_topk

        self.assertEqual(len(results), 2)

    def test_recovery_metrics_include_routes_beyond_top5_reservoir_pool(self):
        miss = {
            "steps": [
                {
                    "reaction_smiles": "N>>CCO",
                    "main_reactant": "N",
                    "metrics": {"strict_stock_solve": False},
                }
            ],
            "metrics": {"strict_stock_solve": False},
        }
        hit = {
            "steps": [
                {
                    "reaction_smiles": "CC.O>>CCO",
                    "main_reactant": "CC",
                    "aux_reactants": ["O"],
                }
            ],
            "metrics": {"strict_stock_solve": True},
            "broad_reservoir": {"native_rank": 1},
        }
        target = {
            "index": 0,
            "target_smiles": "CCO",
            "route_domain": "all_chemical",
            "gt_route": [{"rxn_smiles": "CC.O>>CCO", "transformation": "addition"}],
            "metrics": {"plan": True, "strict_stock_solve_any": True},
            "planner_output": {"time_s": 1.5, "routes": [miss, miss, miss, miss, miss, hit]},
        }

        refresh_recovery_metrics([target])
        summary = summarize_target_results([target], check_stock=True)

        self.assertTrue(target["route_recovery"]["exact_reaction_in_route_pool"])
        self.assertEqual(target["route_recovery"]["exact_reaction_first_rank"], 6)
        self.assertTrue(target["route_recovery"]["gt_reactant_in_route_pool"])
        self.assertEqual(summary["exact_reaction_in_route_pool"], 1.0)
        self.assertEqual(summary["avg_route_count"], 6.0)

    def test_refresh_route_metrics_preserves_native_stock_overrides(self):
        target = {
            "planner_output": {
                "routes": [
                    {
                        "global_constraints": {
                            "stock_overrides": {"N": True},
                            "stock_override_source": "native_chem_enzy",
                        },
                        "steps": [
                            {
                                "product": "CCO",
                                "main_reactant": "N",
                                "aux_reactants": [],
                                "reaction_smiles": "N>>CCO",
                            }
                        ],
                    }
                ]
            }
        }

        refresh_route_metrics([target], stock_checker=lambda _smi: False)

        route_metrics_row = target["planner_output"]["routes"][0]["metrics"]
        self.assertTrue(route_metrics_row["strict_stock_solve"])
        self.assertEqual(route_metrics_row["stock_override_count"], 1)
        self.assertTrue(target["metrics"]["strict_stock_solve_any"])

    def test_route_tree_empty_results_use_cc_aostar_fallback(self):
        fake_skeleton = SimpleNamespace(
            types=("oxidation",),
            ec1s=(0,),
            Ts=(None,),
            pHs=(None,),
            compat_pred=0.9,
            opmode_pred="candidate",
            issues_pred=(),
            log_prob=0.0,
        )
        fallback_result = RouteResult(board=CascadeBoard.from_n_steps(1, "CCO"))

        with mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons",
            return_value=[fake_skeleton],
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior",
            side_effect=lambda skeletons, *args, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.augment_skeletons_with_retrieval_prior",
            side_effect=lambda skeletons, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.rerank_skeletons_with_model",
            side_effect=lambda skeletons, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.route_tree.search.plan_with_route_tree",
            return_value=[],
        ) as route_tree_mock, mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.plan_with_cc_aostar",
            return_value=[fallback_result],
        ) as fallback_mock, mock.patch.dict(
            os.environ,
            {"AUTOPLANNER_ROUTE_TREE_EMPTY_FALLBACK": "1"},
            clear=False,
        ):
            results = _plan_one_target(
                SimpleNamespace(),
                {},
                target="CCO",
                depth=1,
                domain="all_chemical",
                model_device="cpu",
                n_results=5,
                n_candidates_per_skeleton=1,
                search_mode="route_tree",
                stock_checker=lambda _smi: True,
                constraints={},
            )

        self.assertEqual(route_tree_mock.call_count, 1)
        self.assertEqual(fallback_mock.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].product, "CCO")

    def test_route_tree_empty_results_can_fall_back_to_skeleton_fill(self):
        fake_skeleton = SimpleNamespace(
            types=("oxidation",),
            ec1s=(0,),
            Ts=(None,),
            pHs=(None,),
            compat_pred=0.9,
            opmode_pred="candidate",
            issues_pred=(),
            log_prob=0.0,
        )
        fallback_result = RouteResult(board=CascadeBoard.from_n_steps(1, "CCO"))

        with mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons",
            return_value=[fake_skeleton],
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior",
            side_effect=lambda skeletons, *args, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.augment_skeletons_with_retrieval_prior",
            side_effect=lambda skeletons, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.rerank_skeletons_with_model",
            side_effect=lambda skeletons, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.route_tree.search.plan_with_route_tree",
            return_value=[],
        ) as route_tree_mock, mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.plan_with_cc_aostar",
            return_value=[],
        ) as cc_mock, mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.plan_with_skeleton",
            return_value=[fallback_result],
        ) as skeleton_mock, mock.patch.dict(
            os.environ,
            {"AUTOPLANNER_ROUTE_TREE_EMPTY_FALLBACK": "1"},
            clear=False,
        ):
            results = _plan_one_target(
                SimpleNamespace(),
                {},
                target="CCO",
                depth=1,
                domain="all_chemical",
                model_device="cpu",
                n_results=5,
                n_candidates_per_skeleton=1,
                search_mode="route_tree",
                stock_checker=lambda _smi: True,
                constraints={},
            )

        self.assertEqual(route_tree_mock.call_count, 1)
        self.assertEqual(cc_mock.call_count, 1)
        self.assertEqual(skeleton_mock.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].product, "CCO")

    def test_route_tree_enzymatic_domain_uses_min_skeleton_budget(self):
        fake_skeleton = SimpleNamespace(
            types=("biosynthetic_step",),
            ec1s=(1,),
            Ts=(None,),
            pHs=(None,),
            compat_pred=0.9,
            opmode_pred="candidate",
            issues_pred=(),
            log_prob=0.0,
        )
        route_result = RouteResult(board=CascadeBoard.from_n_steps(1, "CCO"))

        with mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons",
            return_value=[fake_skeleton, fake_skeleton, fake_skeleton, fake_skeleton],
        ) as generate_mock, mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior",
            side_effect=lambda skeletons, *args, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.augment_skeletons_with_retrieval_prior",
            side_effect=lambda skeletons, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.cascadeboard.live_benchmark.rerank_skeletons_with_model",
            side_effect=lambda skeletons, **kwargs: list(skeletons),
        ), mock.patch(
            "cascade_planner.route_tree.search.plan_with_route_tree",
            return_value=[route_result],
        ):
            results = _plan_one_target(
                SimpleNamespace(),
                {},
                target="CCO",
                depth=1,
                domain="enzymatic",
                model_device="cpu",
                n_results=3,
                n_candidates_per_skeleton=1,
                skeleton_samples=1,
                search_mode="route_tree",
                stock_checker=lambda _smi: True,
                constraints={},
            )

        self.assertEqual(generate_mock.call_args.kwargs["k"], 4)
        self.assertEqual(results, [route_result])

    def test_external_smoke_summary_counts_broad_reservoir_routes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir(parents=True)
            run_dir = root / "paroutes_n1"
            run_dir.mkdir()
            manifest = {
                "commands": [
                    {
                        "split": "paroutes_n1",
                        "config": "D",
                        "dataset_label": "paroutes_n1",
                    }
                ],
                "datasets": [
                    {
                        "label": "paroutes_n1",
                        "ready": True,
                        "benchmark": "bench.json",
                        "source": "source.txt",
                        "route_annotations": False,
                        "n_rows": 1,
                    }
                ],
            }
            (root / "external_smoke_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "logs" / "external_smoke_run_report.json").write_text(
                json.dumps({"results": [], "failed": []}),
                encoding="utf-8",
            )
            run = {
                "summary": {
                    "plan_rate": 1.0,
                    "strict_stock_solve_any": 1.0,
                    "candidate_exact_reaction_in_pool": 0.0,
                    "candidate_gt_reactant_in_pool": 1.0,
                    "exact_reaction_in_route_pool": 0.5,
                    "gt_reactant_in_route_pool": 1.0,
                    "avg_time_per_target_s": 2.0,
                    "avg_route_count": 2.0,
                    "route_tree_runtime_bottleneck_counts": {"proposal_slow": 1},
                    "route_tree_source_latency_ms": {"retrochimera": 123.0},
                },
                "targets": [
                    {
                        "target_smiles": "CCO",
                        "planner_output": {
                            "routes": [
                                {
                                    "broad_reservoir": {"stock_closed": True},
                                    "metrics": {"strict_stock_solve": True},
                                },
                                {
                                    "broad_reservoir": {"stock_closed": False},
                                    "metrics": {"strict_stock_solve": False},
                                },
                            ]
                        }
                    }
                ],
            }
            (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
            (run_dir / "native_reservoir.json").write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "routes": [
                                    {
                                        "runtime_stock": False,
                                        "steps": [{"product_smiles": "CCO", "reactant_smiles": ["CC"], "stock_status": {"CC": True}}],
                                    },
                                    {
                                        "runtime_stock": True,
                                        "steps": [{"product_smiles": "CCO", "reactant_smiles": ["O"], "stock_status": {"O": True}}],
                                    },
                                    {
                                        "runtime_stock": False,
                                        "steps": [{"product_smiles": "CCO", "reactant_smiles": ["N"], "stock_status": {"N": False}}],
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "cascade_planner.eval.build_external_reservoir_smokes._summary_stock_checker",
                return_value=object(),
            ), mock.patch(
                "cascade_planner.eval.build_external_reservoir_smokes._route_runtime_stock_closed",
                side_effect=lambda route, **_kwargs: bool(route.get("runtime_stock")),
            ):
                summary = summarize_external_smokes(output_dir=root)

        row = summary["rows"][0]
        self.assertEqual(row["strict_stock_solve_any"], 1.0)
        self.assertEqual(row["candidate_gt_reactant_in_pool"], 1.0)
        self.assertEqual(row["exact_reaction_in_route_pool"], 0.5)
        self.assertEqual(row["gt_reactant_in_route_pool"], 1.0)
        self.assertEqual(row["avg_route_count"], 2.0)
        self.assertEqual(row["route_tree_runtime_bottleneck_counts"], {"proposal_slow": 1})
        self.assertEqual(row["route_tree_source_latency_ms"], {"retrochimera": 123.0})
        self.assertEqual(row["broad_reservoir_targets"], 1)
        self.assertEqual(row["broad_reservoir_routes"], 2)
        self.assertEqual(row["broad_reservoir_stock_routes"], 1)
        self.assertEqual(row["broad_reservoir_metadata_stock_routes"], 1)
        self.assertEqual(row["broad_reservoir_runtime_stock_routes"], 1)
        self.assertTrue(row["native_payload_exists"])
        self.assertEqual(row["native_payload_routes"], 3)
        self.assertEqual(row["native_payload_metadata_stock_routes"], 2)
        self.assertEqual(row["native_payload_runtime_stock_routes"], 1)

    def test_external_smoke_summary_reports_paired_config_deltas(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir(parents=True)
            manifest = {
                "commands": [
                    {
                        "stage": "external_smoke",
                        "split": "C_paroutes_n1",
                        "config": "C",
                        "dataset_label": "paroutes_n1",
                    },
                    {
                        "stage": "external_smoke",
                        "split": "D_paroutes_n1",
                        "config": "D",
                        "dataset_label": "paroutes_n1",
                    },
                ],
                "datasets": [
                    {
                        "label": "paroutes_n1",
                        "ready": True,
                        "benchmark": "bench.json",
                        "source": "source.txt",
                        "route_annotations": False,
                        "n_rows": 1,
                    }
                ],
            }
            (root / "external_smoke_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "logs" / "external_smoke_run_report.json").write_text(
                json.dumps({"results": [], "failed": []}),
                encoding="utf-8",
            )
            for label, stock, cand_gt, broad in [
                ("C_paroutes_n1", 0.0, 1.0, False),
                ("D_paroutes_n1", 1.0, 0.0, True),
            ]:
                run_dir = root / label
                run_dir.mkdir()
                run = {
                    "summary": {
                        "plan_rate": 1.0,
                        "strict_stock_solve_any": stock,
                        "candidate_exact_reaction_in_pool": 0.0,
                        "candidate_gt_reactant_in_pool": cand_gt,
                        "exact_reaction_in_route_pool": 0.0,
                        "gt_reactant_in_route_pool": cand_gt,
                        "avg_time_per_target_s": 10.0,
                        "avg_route_count": 3.0,
                    },
                    "targets": [
                        {
                            "target_smiles": "CCO",
                            "planner_output": {
                                "routes": [
                                    {
                                        "broad_reservoir": {"stock_closed": True} if broad else {},
                                        "metrics": {"strict_stock_solve": bool(stock)},
                                    }
                                ]
                            },
                        }
                    ],
                }
                (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")

            summary = summarize_external_smokes(output_dir=root)

        paired = summary["paired_config_deltas"]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["dataset_label"], "paroutes_n1")
        self.assertEqual(paired[0]["metric_deltas"]["strict_stock_solve_any"], 1.0)
        self.assertEqual(paired[0]["metric_deltas"]["candidate_gt_reactant_in_pool"], -1.0)
        self.assertEqual(paired[0]["coverage_gains"], ["strict_stock_solve_any"])
        self.assertIn("candidate_gt_reactant_in_pool", paired[0]["coverage_losses"])
        self.assertEqual(paired[0]["candidate_broad_reservoir_routes"], 1)
        self.assertEqual(paired[0]["likely_change_cause"], "bounded_reservoir_or_search_path")

    def test_external_smoke_aggregate_weights_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            one = root / "one.json"
            two = root / "two.json"
            one.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "label": "C_paroutes_n1",
                                "dataset_label": "paroutes_n1",
                                "config": "C",
                                "n_run_targets": 10,
                                "strict_stock_solve_any": 0.5,
                                "exact_reaction_in_route_pool": 0.0,
                                "avg_route_count": 2.0,
                            },
                            {
                                "label": "D_paroutes_n1",
                                "dataset_label": "paroutes_n1",
                                "config": "D",
                                "n_run_targets": 10,
                                "strict_stock_solve_any": 0.8,
                                "exact_reaction_in_route_pool": 0.2,
                                "avg_route_count": 5.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            two.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "label": "C_paroutes_n1",
                                "dataset_label": "paroutes_n1",
                                "config": "C",
                                "n_run_targets": 30,
                                "strict_stock_solve_any": 0.9,
                                "exact_reaction_in_route_pool": 0.4,
                                "avg_route_count": 3.0,
                            },
                            {
                                "label": "D_paroutes_n1",
                                "dataset_label": "paroutes_n1",
                                "config": "D",
                                "n_run_targets": 30,
                                "strict_stock_solve_any": 1.0,
                                "exact_reaction_in_route_pool": 0.6,
                                "avg_route_count": 7.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = aggregate_external_smoke_summaries(
                summaries=[one, two],
                output=root / "aggregate.json",
                markdown=root / "aggregate.md",
            )

        rows = {(row["dataset_label"], row["config"]): row for row in report["rows"]}
        self.assertEqual(rows[("paroutes_n1", "C")]["n_run_targets"], 40)
        self.assertAlmostEqual(rows[("paroutes_n1", "C")]["strict_stock_solve_any"], 0.8)
        self.assertAlmostEqual(rows[("paroutes_n1", "D")]["avg_route_count"], 6.5)
        self.assertEqual(report["paired_config_deltas"][0]["metric_deltas"]["strict_stock_solve_any"], 0.15)

    def test_uspto190_cache_reports_missing_and_cached_pages(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "uspto190_index.html").write_text(
                '<a href="targets/aaa">one</a><a href="targets/bbb">two</a>',
                encoding="utf-8",
            )
            (root / "uspto190_aaa.html").write_text("cached", encoding="utf-8")

            report = cache_uspto190_targets(cache_dir=root, limit=2, fetch=False)

        self.assertEqual(report["target_paths_discovered"], 2)
        self.assertEqual(report["selected_targets"], 2)
        self.assertEqual(report["cached_target_pages"], 1)
        self.assertEqual(report["missing_selected"], 1)
        self.assertFalse(report["ready_for_selected_window"])

    def test_external_smoke_summary_accepts_manifest_runner_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir(parents=True)
            run_dir = root / "C_paroutes_n1"
            run_dir.mkdir()
            manifest = {
                "commands": [
                    {
                        "stage": "external_smoke",
                        "split": "C_paroutes_n1",
                        "config": "C",
                        "dataset_label": "paroutes_n1",
                        "outputs": {"run": str(run_dir / "run.json")},
                    }
                ],
                "datasets": [{"label": "paroutes_n1", "ready": True, "n_rows": 1}],
            }
            (root / "external_smoke_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "logs" / "manifest_command_run_report.json").write_text(
                json.dumps({"results": [{"split": "C_paroutes_n1"}], "failed": []}),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps({"summary": {"plan_rate": 1.0}, "targets": [{"target_smiles": "CCO"}]}),
                encoding="utf-8",
            )

            summary = summarize_external_smokes(output_dir=root)

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["run_report"]["failed"], [])

    def test_paroutes_builder_uses_manual_reference_routes_for_n1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            bench_dir = root / "benchmarks"
            source_dir.mkdir()
            bench_dir.mkdir()
            targets = root / "targets_n1.txt"
            refs = root / "ref_routes_n1.json"
            targets.write_text("CCO\nCCN\n", encoding="utf-8")
            refs.write_text(
                json.dumps(
                    [
                        {
                            "smiles": "CCO",
                            "type": "mol",
                            "children": [
                                {
                                    "type": "reaction",
                                    "metadata": {"smiles": "[CH3:1].[OH2:2]>>[CH3:1][OH:2]"},
                                }
                            ],
                        },
                        {
                            "smiles": "CCN",
                            "type": "mol",
                            "children": [
                                {
                                    "type": "reaction",
                                    "metadata": {"rsmi": "[CH3:1].[NH3:2]>O>[CH3:1][NH2:2]"},
                                }
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "cascade_planner.eval.build_external_reservoir_smokes.PAROUTES_MANUAL_SOURCES",
                {"n1": ((targets, refs),), "n5": ()},
            ):
                dataset = _build_paroutes_split(source_dir, bench_dir, "n1", limit=2, fetch=False)

            rows = json.loads((bench_dir / "paroutes_n1_smoke.json").read_text(encoding="utf-8"))

        self.assertTrue(dataset["ready"])
        self.assertTrue(dataset["route_annotations"])
        self.assertEqual(dataset["source"], str(targets))
        self.assertEqual(dataset["reference_source"], str(refs))
        self.assertEqual(dataset["n_rows"], 2)
        self.assertEqual(rows[0]["gt_route"][0]["rxn_smiles"], "[CH3].[OH2]>>[CH3][OH]")
        self.assertEqual(rows[0]["reference_depth"], 1)
        self.assertEqual(rows[0]["depth"], 1)
        self.assertEqual(rows[1]["gt_route"][0]["rxn_smiles"], "[CH3].[NH3]>>[CH3][NH2]")

    def test_paroutes_builder_caps_planner_depth_without_truncating_reference_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            bench_dir = root / "benchmarks"
            source_dir.mkdir()
            bench_dir.mkdir()
            targets = root / "targets_n5.txt"
            refs = root / "ref_routes_n5.json"
            targets.write_text("CCCCCCCCCCO\n", encoding="utf-8")
            refs.write_text(
                json.dumps(
                    [
                        {
                            "smiles": "CCCCCCCCCCO",
                            "type": "mol",
                            "children": [
                                {
                                    "type": "reaction",
                                    "metadata": {"smiles": f"C{idx}.O>>C{idx}O"},
                                }
                                for idx in range(10)
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "cascade_planner.eval.build_external_reservoir_smokes.PAROUTES_MANUAL_SOURCES",
                {"n1": (), "n5": ((targets, refs),)},
            ):
                dataset = _build_paroutes_split(source_dir, bench_dir, "n5", limit=1, fetch=False)

            rows = json.loads((bench_dir / "paroutes_n5_smoke.json").read_text(encoding="utf-8"))

        self.assertTrue(dataset["ready"])
        self.assertTrue(dataset["route_annotations"])
        self.assertEqual(rows[0]["reference_depth"], 10)
        self.assertEqual(rows[0]["depth"], 8)
        self.assertEqual(len(rows[0]["gt_route"]), 10)

    def test_uspto_builder_reads_paginated_cached_pages(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            bench_dir = root / "benchmarks"
            source_dir.mkdir()
            bench_dir.mkdir()
            slugs = ["target_a", "target_b", "target_c"]
            (source_dir / "uspto190_index.html").write_text(
                '<a href="targets/target_a">A</a><a href="?page=2">2</a>',
                encoding="utf-8",
            )
            (source_dir / "uspto190_page_2.html").write_text(
                '<a href="targets/target_b">B</a><a href="targets/target_c">C</a>',
                encoding="utf-8",
            )
            for idx, slug in enumerate(slugs, 1):
                payload = {
                    "route": {"id": f"route_{idx}"},
                    "target": {
                        "targetId": f"USPTO-{idx:03d}/190",
                        "molecule": {"smiles": f"C{idx}O"},
                        "routeLength": 1,
                    },
                    "rootNode": {
                        "molecule": {"smiles": f"C{idx}O"},
                        "reactionStep": {"id": f"rxn_{idx}"},
                        "children": [{"molecule": {"smiles": f"C{idx}"}}],
                    },
                }
                text = json.dumps(payload, indent=2).replace('"', "&quot;")
                (source_dir / f"uspto190_{slug}.html").write_text(f"<pre>{text}</pre>", encoding="utf-8")

            dataset = _build_uspto190(source_dir, bench_dir, limit=3, fetch=False)
            rows = json.loads((bench_dir / "uspto_190_smoke.json").read_text(encoding="utf-8"))

            self.assertTrue(dataset["ready"])
            self.assertEqual(dataset["n_rows"], 3)
            self.assertTrue(dataset["route_annotations"])
        self.assertEqual([row["cascade_id"] for row in rows], ["USPTO-001/190", "USPTO-002/190", "USPTO-003/190"])
        self.assertEqual(rows[0]["gt_route"][0]["rxn_smiles"], "C1>>C1O")

    def test_uspto_builder_can_read_resumable_cache_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_dir = root / "cache"
            source_dir = root / "sources"
            bench_dir = root / "benchmarks"
            cache_dir.mkdir()
            source_dir.mkdir()
            bench_dir.mkdir()
            (cache_dir / "uspto190_index.html").write_text(
                '<a href="targets/target_a">A</a><a href="targets/target_b">B</a>',
                encoding="utf-8",
            )
            for idx, slug in enumerate(["target_a", "target_b"], 1):
                payload = {
                    "route": {"id": f"route_{idx}"},
                    "target": {
                        "targetId": f"USPTO-{idx:03d}/190",
                        "molecule": {"smiles": f"C{idx}O"},
                        "routeLength": 1,
                    },
                    "rootNode": {
                        "molecule": {"smiles": f"C{idx}O"},
                        "reactionStep": {"id": f"rxn_{idx}"},
                        "children": [{"molecule": {"smiles": f"C{idx}"}}],
                    },
                }
                text = json.dumps(payload, indent=2).replace('"', "&quot;")
                (cache_dir / f"uspto190_{slug}.html").write_text(f"<pre>{text}</pre>", encoding="utf-8")

            dataset = _build_uspto190(
                source_dir,
                bench_dir,
                limit=2,
                fetch=False,
                uspto_cache_dir=cache_dir,
            )
            rows = json.loads((bench_dir / "uspto_190_smoke.json").read_text(encoding="utf-8"))
            copied_index = (source_dir / "uspto190_index.html").exists()
            copied_target = (source_dir / "uspto190_target_a.html").exists()

        self.assertTrue(dataset["ready"])
        self.assertEqual(dataset["n_rows"], 2)
        self.assertTrue(copied_index)
        self.assertTrue(copied_target)
        self.assertEqual(rows[1]["cascade_id"], "USPTO-002/190")

    def test_external_smoke_manifest_requires_c_for_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "D_APPEND requires config C"):
                build_external_smokes(
                    output_dir=Path(td),
                    controller_path=Path("controller.pt"),
                    limit=1,
                    fetch=False,
                    configs=("D_APPEND",),
                )

    def test_external_smoke_manifest_can_build_append_only_reservoir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "n1-targets.txt").write_text("CCO\n", encoding="utf-8")
            (source_dir / "n5-targets.txt").write_text("CCC\n", encoding="utf-8")
            (source_dir / "bionavi_testset.txt").write_text("1 CCO CC\n", encoding="utf-8")

            manifest = build_external_smokes(
                output_dir=root,
                controller_path=Path("controller.pt"),
                native_payload=None,
                limit=1,
                fetch=False,
                configs=("C", "D_APPEND"),
                native_iterations=2,
                native_max_depth=3,
                native_expansion_topk=7,
            )

        commands = manifest["commands"]
        smoke_commands = [cmd for cmd in commands if cmd["stage"] == "external_smoke"]
        native_commands = [cmd for cmd in commands if cmd["stage"] == "external_native_payload"]
        append_commands = [cmd for cmd in commands if cmd["stage"] == "external_append_only"]
        self.assertEqual(len(smoke_commands), 3)
        self.assertEqual(len(native_commands), 3)
        self.assertEqual(len(append_commands), 3)
        self.assertEqual(manifest["configs"], ["C", "D_APPEND"])
        self.assertTrue(all(cmd["config"] == "C" for cmd in smoke_commands))
        first_append = append_commands[0]
        self.assertEqual(first_append["config"], "D_APPEND")
        self.assertIn("python -m cascade_planner.eval.chem_enzy_broad_union", first_append["cmd"])
        self.assertIn("--native-topk 5", first_append["cmd"])
        self.assertIn("--native-selection rank_plus_stock", first_append["cmd"])
        self.assertIn("C_paroutes_n1/run.json", first_append["cmd"])
        self.assertIn("D_APPEND_paroutes_n1/run.json", first_append["cmd"])
        self.assertIn("--iterations 2", native_commands[0]["cmd"])
        self.assertIn("--max-depth 3", native_commands[0]["cmd"])
        self.assertIn("--expansion-topk 7", native_commands[0]["cmd"])

    def test_external_smoke_manifest_can_enable_chem_enzy_onestep_proposals(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "n1-targets.txt").write_text("CCO\n", encoding="utf-8")
            (source_dir / "n5-targets.txt").write_text("CCC\n", encoding="utf-8")

            manifest = build_external_smokes(
                output_dir=root,
                controller_path=Path("controller.pt"),
                native_payload=Path("native_payload.json"),
                limit=1,
                fetch=False,
                configs=("C", "C_CHEMSTEP", "D_CHEMSTEP"),
            )

        self.assertEqual(manifest["configs"], ["C", "C_CHEMSTEP", "D_CHEMSTEP"])
        commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "external_smoke"]
        chemstep = next(cmd for cmd in commands if cmd["config"] == "C_CHEMSTEP")
        hybrid = next(cmd for cmd in commands if cmd["config"] == "D_CHEMSTEP")
        self.assertIn("AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS=1", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_CHEMENZY_ONESTEP_MODELS=graphfp_models.USPTO-full_remapped,onmt_models.bionav_one_step", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH=retrochimera:0,chemtemplates:0,chem_enzy_onestep:0", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_ROOT_PROPOSAL_BUDGET=16", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_MIN_BRANCH_FACTOR=12", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS=1", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MAX_SCORE_GAP=2.0", chemstep["cmd"])
        self.assertNotIn("AUTOPLANNER_ROUTE_TREE_CHILD_STOCK_DELTA_WEIGHT", chemstep["cmd"])
        self.assertNotIn("AUTOPLANNER_ROUTE_TREE_FRONTIER_DELTA_WEIGHT", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER=2", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES=1", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS=1", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX=2", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK=1", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_FINAL_RERANK=1", chemstep["cmd"])
        self.assertNotIn("AUTOPLANNER_ROUTE_TREE_NONSTOCK_SMALL_LEAF_WEIGHT", chemstep["cmd"])
        self.assertNotIn("AUTOPLANNER_ROUTE_TREE_NONSTOCK_SMALL_STATE_WEIGHT", chemstep["cmd"])
        self.assertNotIn("AUTOPLANNER_ROUTE_TREE_ANTI_PROGRESS_WEIGHT", chemstep["cmd"])
        self.assertNotIn("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1", chemstep["cmd"])
        self.assertIn("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1", hybrid["cmd"])

    def test_external_smoke_manifest_can_enable_cascade_oracle_value(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "n1-targets.txt").write_text("CCO\n", encoding="utf-8")
            (source_dir / "n5-targets.txt").write_text("CCC\n", encoding="utf-8")

            manifest = build_external_smokes(
                output_dir=root,
                controller_path=Path("controller.pt"),
                native_payload=None,
                limit=1,
                fetch=False,
                configs=("C", "C_ORACLE", "C_CHEMSTEP_ORACLE"),
                native_iterations=2,
                native_max_depth=3,
                native_expansion_topk=7,
            )

        self.assertEqual(manifest["configs"], ["C", "C_ORACLE", "C_CHEMSTEP_ORACLE"])
        native_commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "external_native_payload"]
        oracle_commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "external_cascade_oracle_payload"]
        smoke_commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "external_smoke"]
        self.assertEqual(len(native_commands), 2)
        self.assertEqual(len(oracle_commands), 4)
        self.assertEqual(len(smoke_commands), 6)
        oracle_smoke = next(cmd for cmd in smoke_commands if cmd["config"] == "C_ORACLE")
        chemstep_oracle = next(cmd for cmd in smoke_commands if cmd["config"] == "C_CHEMSTEP_ORACLE")
        self.assertIn("python -m cascade_planner.eval.build_cascade_oracle_payload", oracle_commands[0]["cmd"])
        self.assertIn("AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE=1", oracle_smoke["cmd"])
        self.assertIn("AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT=1.25", oracle_smoke["cmd"])
        self.assertIn("AUTOPLANNER_CASCADE_ORACLE_PAYLOAD=", oracle_smoke["cmd"])
        self.assertIn("cascade_oracle_payload.json", oracle_smoke["cmd"])
        self.assertNotIn("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1", oracle_smoke["cmd"])
        self.assertIn("AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS=1", chemstep_oracle["cmd"])
        self.assertIn("AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE=1", chemstep_oracle["cmd"])

    def test_external_smoke_manifest_reuses_prebuilt_d_payload_for_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "n1-targets.txt").write_text("CCO\n", encoding="utf-8")
            (source_dir / "n5-targets.txt").write_text("CCC\n", encoding="utf-8")

            manifest = build_external_smokes(
                output_dir=root,
                controller_path=Path("controller.pt"),
                native_payload=None,
                limit=1,
                fetch=False,
                configs=("C", "D", "D_APPEND"),
                prebuild_native_payload=True,
            )

        native_commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "external_native_payload"]
        append_commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "external_append_only"]
        self.assertEqual(len(native_commands), 2)
        self.assertEqual(len(append_commands), 2)
        self.assertIn("D_paroutes_n1/native_reservoir.json", append_commands[0]["cmd"])
        self.assertNotIn("D_APPEND_paroutes_n1/native_reservoir.json", append_commands[0]["cmd"])

    def test_external_smoke_summary_reads_append_only_synthetic_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir(parents=True)
            manifest = {
                "commands": [
                    {
                        "stage": "external_smoke",
                        "split": "C_paroutes_n1",
                        "config": "C",
                        "dataset_label": "paroutes_n1",
                        "outputs": {"run": str(root / "C_paroutes_n1" / "run.json")},
                    },
                    {
                        "stage": "external_append_only",
                        "split": "D_APPEND_paroutes_n1",
                        "config": "D_APPEND",
                        "dataset_label": "paroutes_n1",
                        "outputs": {"run": str(root / "D_APPEND_paroutes_n1" / "run.json")},
                    },
                ],
                "datasets": [
                    {
                        "label": "paroutes_n1",
                        "ready": True,
                        "benchmark": "bench.json",
                        "source": "source.txt",
                        "route_annotations": True,
                        "n_rows": 1,
                    }
                ],
            }
            (root / "external_smoke_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "logs" / "external_smoke_run_report.json").write_text(
                json.dumps({"results": [], "failed": []}),
                encoding="utf-8",
            )
            c_dir = root / "C_paroutes_n1"
            c_dir.mkdir()
            (c_dir / "run.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "plan_rate": 1.0,
                            "strict_stock_solve_any": 0.0,
                            "candidate_exact_reaction_in_pool": 0.0,
                            "candidate_gt_reactant_in_pool": 1.0,
                            "exact_reaction_in_route_pool": 0.0,
                            "gt_reactant_in_route_pool": 1.0,
                            "avg_time_per_target_s": 12.0,
                            "avg_route_count": 1.0,
                        },
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "metrics": {"plan": True, "strict_stock_solve_any": False},
                                "planner_output": {"routes": [{"steps": []}]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            append_dir = root / "D_APPEND_paroutes_n1"
            append_dir.mkdir()
            (append_dir / "run.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "synthesized_union": {
                                "stock_rate": 1.0,
                                "exact_reaction_in_route_pool": 1.0,
                                "gt_reactant_in_route_pool": 1.0,
                                "avg_route_count": 2.0,
                            }
                        },
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "metrics": {"plan": True, "strict_stock_solve_any": True},
                                "route_recovery": {
                                    "candidate_exact_reaction_in_pool": True,
                                    "candidate_gt_reactant_in_pool": True,
                                    "exact_reaction_in_route_pool": True,
                                    "gt_reactant_in_route_pool": True,
                                },
                                "planner_output": {
                                    "routes": [
                                        {"steps": []},
                                        {
                                            "broad_reservoir": {"stock_closed": True},
                                            "metrics": {"strict_stock_solve": True},
                                            "steps": [],
                                        },
                                    ]
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_external_smokes(output_dir=root)

        rows = {row["label"]: row for row in summary["rows"]}
        append_row = rows["D_APPEND_paroutes_n1"]
        self.assertEqual(append_row["config"], "D_APPEND")
        self.assertEqual(append_row["plan_rate"], 1.0)
        self.assertEqual(append_row["strict_stock_solve_any"], 1.0)
        self.assertEqual(append_row["candidate_exact_reaction_in_pool"], 1.0)
        self.assertEqual(append_row["exact_reaction_in_route_pool"], 1.0)
        self.assertEqual(append_row["avg_route_count"], 2.0)
        self.assertEqual(append_row["broad_reservoir_routes"], 1)
        paired = summary["paired_config_deltas"]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["candidate_config"], "D_APPEND")
        self.assertEqual(paired[0]["metric_deltas"]["strict_stock_solve_any"], 1.0)
        self.assertEqual(paired[0]["metric_deltas"]["exact_reaction_in_route_pool"], 1.0)
        self.assertFalse(paired[0]["coverage_losses"])
        self.assertEqual(paired[0]["likely_change_cause"], "append_only_bounded_reservoir_gain")

    def test_external_smoke_manifest_can_prebuild_native_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "n1-targets.txt").write_text("CCO\n", encoding="utf-8")
            (source_dir / "n5-targets.txt").write_text("CCC\n", encoding="utf-8")
            (source_dir / "bionavi_testset.txt").write_text("1 CCO CC\n", encoding="utf-8")

            manifest = build_external_smokes(
                output_dir=root,
                controller_path=Path("controller.pt"),
                native_payload=Path("full100_native.json"),
                limit=1,
                fetch=False,
                configs=("D",),
                enable_source_override=True,
                prebuild_native_payload=True,
                native_iterations=2,
                native_max_depth=3,
                native_expansion_topk=7,
                trust_native_stock=True,
            )

        commands = manifest["commands"]
        native_commands = [cmd for cmd in commands if cmd["stage"] == "external_native_payload"]
        smoke_commands = [cmd for cmd in commands if cmd["stage"] == "external_smoke"]
        self.assertEqual(len(native_commands), 3)
        self.assertEqual(len(smoke_commands), 3)
        self.assertIn("--iterations 2", native_commands[0]["cmd"])
        self.assertIn("--max-depth 3", native_commands[0]["cmd"])
        self.assertIn("--expansion-topk 7", native_commands[0]["cmd"])
        self.assertIn(
            "AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD=",
            smoke_commands[0]["cmd"],
        )
        self.assertIn("AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK=1", smoke_commands[0]["cmd"])
        self.assertIn("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL=1", smoke_commands[0]["cmd"])
        self.assertIn("native_reservoir.json", smoke_commands[0]["cmd"])
        self.assertTrue(manifest["prebuild_native_payload"])
        self.assertTrue(manifest["trust_native_stock"])

    def test_external_smoke_manifest_can_build_quality_filter_configs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "n1-targets.txt").write_text("CCO\n", encoding="utf-8")
            (source_dir / "n5-targets.txt").write_text("CCC\n", encoding="utf-8")

            manifest = build_external_smokes(
                output_dir=root,
                controller_path=Path("controller.pt"),
                native_payload=Path("native_payload.json"),
                limit=1,
                fetch=False,
                configs=("C", "D_FILTER", "D_TOP10_FILTER"),
            )

        commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "external_smoke"]
        filter_cmds = [cmd["cmd"] for cmd in commands if cmd["config"] == "D_FILTER"]
        top10_cmds = [cmd["cmd"] for cmd in commands if cmd["config"] == "D_TOP10_FILTER"]
        self.assertTrue(filter_cmds)
        self.assertTrue(top10_cmds)
        self.assertTrue(all("AUTOPLANNER_RESERVOIR_QUALITY_FILTER=1" in cmd for cmd in filter_cmds))
        self.assertTrue(all("AUTOPLANNER_RESERVOIR_NATIVE_TOPK=5" in cmd for cmd in filter_cmds))
        self.assertTrue(all("AUTOPLANNER_RESERVOIR_QUALITY_FILTER=1" in cmd for cmd in top10_cmds))
        self.assertTrue(all("AUTOPLANNER_RESERVOIR_NATIVE_TOPK=10" in cmd for cmd in top10_cmds))

    def test_external_smoke_manifest_can_filter_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "bionavi_testset.txt").write_text("1 CCO CC\n", encoding="utf-8")

            manifest = build_external_smokes(
                output_dir=root,
                controller_path=Path("controller.pt"),
                native_payload=Path("native_payload.json"),
                limit=1,
                fetch=False,
                configs=("C",),
                datasets_filter=("bionavi_like",),
            )

        self.assertEqual([dataset["label"] for dataset in manifest["datasets"]], ["bionavi_like"])
        self.assertEqual([command["dataset_label"] for command in manifest["commands"]], ["bionavi_like"])

    def test_external_smoke_manifest_filters_before_building_other_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch(
                "cascade_planner.eval.build_external_reservoir_smokes._build_paroutes_split",
                side_effect=AssertionError("paroutes should not be built"),
            ), mock.patch(
                "cascade_planner.eval.build_external_reservoir_smokes._build_uspto190",
                side_effect=AssertionError("uspto should not be built"),
            ), mock.patch(
                "cascade_planner.eval.build_external_reservoir_smokes._build_bionavi_like",
                return_value={
                    "label": "bionavi_like",
                    "benchmark": str(root / "benchmarks" / "bionavi_like_smoke.json"),
                    "n_rows": 1,
                    "ready": True,
                    "route_annotations": True,
                    "error": "",
                },
            ) as build_bionavi:
                manifest = build_external_smokes(
                    output_dir=root,
                    controller_path=Path("controller.pt"),
                    native_payload=Path("native_payload.json"),
                    limit=1,
                    fetch=False,
                    configs=("C",),
                    datasets_filter=("bionavi_like",),
                )

        self.assertEqual(build_bionavi.call_count, 1)
        self.assertEqual([dataset["label"] for dataset in manifest["datasets"]], ["bionavi_like"])
        self.assertEqual([command["dataset_label"] for command in manifest["commands"]], ["bionavi_like"])

    def test_paroutes_reference_builder_supports_offsets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            targets = root / "targets_n1.txt"
            refs = root / "ref_routes_n1.json"
            targets.write_text("A\nB\nC\nD\n", encoding="utf-8")
            refs.write_text(
                json.dumps(
                    [
                        {"smiles": "A"},
                        {"smiles": "B"},
                        {"smiles": "C"},
                        {"smiles": "D"},
                    ]
                ),
                encoding="utf-8",
            )

            rows = _build_paroutes_reference_rows(
                targets_path=targets,
                refs_path=refs,
                split="n1",
                offset=2,
                limit=2,
            )

        self.assertEqual([row["cascade_id"] for row in rows], ["paroutes_n1_2", "paroutes_n1_3"])
        self.assertEqual([row["target_smiles"] for row in rows], ["C", "D"])

    def test_external_smoke_runner_can_skip_existing_native_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = root / "native_reservoir.json"
            payload.write_text(json.dumps({"targets": []}), encoding="utf-8")
            manifest_path = root / "external_smoke_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "commands": [
                            {
                                "stage": "external_native_payload",
                                "split": "paroutes_n1_native_payload",
                                "outputs": {"native_payload": str(payload)},
                                "cmd": "python -c \"raise SystemExit(7)\"",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = run_external_smokes(
                manifest_path=manifest_path,
                log_dir=root / "logs",
                skip_existing_native_payload=True,
            )

        self.assertFalse(report["failed"])
        self.assertEqual(report["results"][0]["returncode"], 0)
        self.assertTrue(report["results"][0]["skipped"])
        self.assertEqual(report["results"][0]["skip_reason"], "native_payload_exists")

    def test_matrix_report_writes_required_artifacts(self):
        run = {
            "summary": {
                "plan_rate": 1.0,
                "strict_stock_solve_any": 0.6,
                "candidate_gt_reactant_in_pool": 0.6,
                "exact_reaction_in_route_pool": 0.4,
                "gt_reactant_in_route_pool": 0.7,
                "avg_time_per_target_s": 1.0,
                "avg_route_count": 2.0,
                "route_tree_source_latency_ms": {"retrochimera": 10.0},
            },
            "targets": [
                {
                    "index": 0,
                    "target_smiles": "CCO",
                    "route_domain": "all_chemical",
                    "metrics": {"plan": True, "strict_stock_solve_any": True},
                    "route_recovery": {"candidate_gt_reactant_in_pool": True, "exact_reaction_in_route_pool": True},
                    "planner_output": {"routes": []},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "A.json"
            d = root / "D.json"
            trace = root / "D_trace.jsonl"
            a.write_text(json.dumps(run), encoding="utf-8")
            d.write_text(json.dumps(run), encoding="utf-8")
            trace.write_text(json.dumps(_trace_row(index=0)) + "\n", encoding="utf-8")
            manifest = build_reservoir_distill_matrix(
                runs={"A": a, "D": d},
                traces={"D": trace},
                output_dir=root / "matrix",
            )

            required = [
                root / "matrix" / "D" / "run.json",
                root / "matrix" / "D" / "run_trace.jsonl",
                root / "matrix" / "D" / "candidate_miss_audit.json",
                root / "matrix" / "D" / "stock_failure_audit.json",
                root / "matrix" / "D" / "closure_report.json",
                root / "matrix" / "D" / "source_policy_report.json",
                root / "matrix" / "D" / "reservoir_distill_report.json",
                root / "matrix" / "D" / "comparison.md",
                root / "matrix" / "comparison.md",
            ]
            required_exist = all(path.exists() for path in required)
            comparison_text = (root / "matrix" / "comparison.md").read_text(encoding="utf-8")

        self.assertIn("D", manifest["reports"])
        self.assertTrue(required_exist)
        self.assertIn("avg routes", comparison_text)
        self.assertIn("latency breakdown", comparison_text)

    def test_acceptance_manifest_and_external_audit_are_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bench = root / "bench.json"
            controller = root / "controller.pt"
            native = root / "native.json"
            bench.write_text(json.dumps([_trace_row(index=0)]), encoding="utf-8")
            controller.write_text("placeholder", encoding="utf-8")
            native.write_text(json.dumps({"targets": []}), encoding="utf-8")

            manifest = build_reservoir_acceptance_manifest(
                output_dir=root / "acceptance",
                benchmark_path=bench,
                controller_path=controller,
                native_payload=native,
                workers=2,
                gpus="",
                device="cpu",
                include_top10=True,
                limit=3,
            )
            audit = audit_external_benchmarks(search_roots=[root])
            smoke = write_smoke_benchmark(source=bench, output=root / "smoke.json", limit=1)
            smoke_rows = json.loads(smoke.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], "reservoir_acceptance_manifest.v1")
        self.assertEqual(set(manifest["matrix"]), {"A", "B", "C", "D", "E"})
        self.assertEqual(len(manifest["commands"]), 6)
        d_cmd = next(row["cmd"] for row in manifest["commands"] if row["config"] == "D")
        self.assertIn("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER", d_cmd)
        self.assertIn("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1", d_cmd)
        self.assertIn("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD", d_cmd)
        self.assertIn("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL=1", d_cmd)
        self.assertEqual(manifest["promotion_gates"]["avg_time_per_target_s_max"], 30.0)
        self.assertFalse(audit["ready"])
        self.assertIn("PaRoutes n1/n5", audit["missing"])
        self.assertEqual(len(smoke_rows), 1)

    def test_stock_closed_alternative_audit_separates_reference_gt(self):
        def route(source, confidence, *, progress=True):
            return {
                "score": confidence,
                "steps": [
                    {
                        "reaction_smiles": "CC>>CCO",
                        "source": source,
                        "scores": {"confidence": confidence},
                        "stock_status": {"CC": True},
                    }
                ],
                "metrics": {
                    "strict_stock_solve": True,
                    "filled_route": True,
                    "route_solved": True,
                    "route_naturalness": {
                        "unfilled_steps": 0,
                        "product_mismatch_steps": 0,
                        "atom_balance_violations": 0,
                        "self_loop_steps": 0,
                    },
                    "retrosynthesis_progress": {
                        "retrosynthesis_progress_success": progress,
                        "progressive_step_fraction": 1.0 if progress else 0.0,
                    },
                    "condition": {"condition_window_success": True},
                    "cascade_compatibility": {"issues": []},
                    "candidate_pool": {"total_candidates": 1 if source == "enzyformer" else 0},
                },
            }

        run = {
            "targets": [
                {
                    "index": 1,
                    "target_smiles": "CCO",
                    "metrics": {"strict_stock_solve_any": True},
                    "route_recovery": {"gt_reactant_in_route_pool": False},
                    "planner_output": {"routes": [route("enzyformer", 0.8)]},
                },
                {
                    "index": 2,
                    "target_smiles": "CCC",
                    "metrics": {"strict_stock_solve_any": True},
                    "route_recovery": {"gt_reactant_in_route_pool": True},
                    "planner_output": {"routes": [route("enzyformer", 0.8)]},
                },
                {
                    "index": 3,
                    "target_smiles": "CCCC",
                    "metrics": {"strict_stock_solve_any": True},
                    "route_recovery": {"gt_reactant_in_route_pool": False},
                    "planner_output": {"routes": [route("([C:1])>>[C:1]", 0.001, progress=False)]},
                },
            ]
        }

        report = build_stock_closed_alternative_audit(run, sample_size=10)

        self.assertEqual(report["n_stock_closed_targets"], 3)
        self.assertEqual(report["n_stock_closed_reference_gt_targets"], 1)
        self.assertEqual(report["n_reviewed_targets"], 2)
        self.assertEqual(report["target_best_class_counts"]["plausible_alternative"], 1)
        self.assertEqual(report["target_best_class_counts"]["suspicious_stock_shortcut"], 1)
        self.assertEqual(report["review_pass_rate"], 0.5)

    def test_acceptance_manifest_can_include_append_only_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bench = root / "bench.json"
            controller = root / "controller.pt"
            native = root / "native.json"
            bench.write_text(json.dumps([_trace_row(index=0)]), encoding="utf-8")
            controller.write_text("placeholder", encoding="utf-8")
            native.write_text(json.dumps({"targets": []}), encoding="utf-8")

            manifest = build_reservoir_acceptance_manifest(
                output_dir=root / "acceptance",
                benchmark_path=bench,
                controller_path=controller,
                native_payload=native,
                workers=1,
                gpus="",
                device="cpu",
                include_append_only=True,
                limit=1,
            )
            command_benchmark_exists = Path(manifest["command_benchmark_path"]).exists()

        self.assertIn("D_APPEND", manifest["matrix"])
        self.assertTrue(command_benchmark_exists)
        self.assertIn("benchmark_limit1.json", manifest["command_benchmark_path"])
        append_cmd = next(row for row in manifest["commands"] if row["config"] == "D_APPEND")
        self.assertEqual(append_cmd["stage"], "full100_append_only_reservoir")
        self.assertIn("python -m cascade_planner.eval.chem_enzy_broad_union", append_cmd["cmd"])
        self.assertIn("benchmark_limit1.json", append_cmd["cmd"])
        self.assertIn("--autoplanner", append_cmd["cmd"])
        self.assertIn("/C/run.json", append_cmd["cmd"])
        self.assertIn("--synthesize-output", append_cmd["cmd"])
        report_cmd = next(row for row in manifest["commands"] if row["stage"] == "full100_reports")
        self.assertIn("--run D_APPEND=", report_cmd["cmd"])
        self.assertNotIn("--trace D_APPEND=", report_cmd["cmd"])

    def test_acceptance_manifest_can_include_quality_filter_ablation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bench = root / "bench.json"
            controller = root / "controller.pt"
            native = root / "native.json"
            bench.write_text(json.dumps([_trace_row(index=0)]), encoding="utf-8")
            controller.write_text("placeholder", encoding="utf-8")
            native.write_text(json.dumps({"targets": []}), encoding="utf-8")

            manifest = build_reservoir_acceptance_manifest(
                output_dir=root / "acceptance",
                benchmark_path=bench,
                controller_path=controller,
                native_payload=native,
                workers=1,
                gpus="",
                device="cpu",
                include_quality_filter_ablation=True,
                limit=1,
            )

        self.assertIn("D_FILTER", manifest["matrix"])
        self.assertIn("D_TOP10_FILTER", manifest["matrix"])
        d_filter = next(row for row in manifest["commands"] if row["config"] == "D_FILTER")
        d_top10_filter = next(row for row in manifest["commands"] if row["config"] == "D_TOP10_FILTER")
        self.assertIn("AUTOPLANNER_RESERVOIR_QUALITY_FILTER=1", d_filter["cmd"])
        self.assertIn("AUTOPLANNER_RESERVOIR_NATIVE_TOPK=10", d_top10_filter["cmd"])

    def test_acceptance_manifest_append_only_requires_native_payload(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "include_append_only requires a native_payload"):
                build_reservoir_acceptance_manifest(
                    output_dir=Path(td) / "acceptance",
                    benchmark_path=Path(td) / "bench.json",
                    controller_path=Path(td) / "controller.pt",
                    include_append_only=True,
                )

    def test_completion_audit_treats_reference_recall_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "B").mkdir(parents=True)
            (root / "D").mkdir(parents=True)
            (root / "reservoir_acceptance_manifest.json").write_text(
                json.dumps(
                    {
                        "promotion_gates": {
                            "plan_rate": 0.95,
                            "strict_stock_solve_any": 0.60,
                            "candidate_gt_reactant_in_pool": 0.58,
                            "exact_reaction_in_route_pool": 0.40,
                            "gt_reactant_in_route_pool": 0.63,
                            "avg_time_per_target_s_max": 16.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "B" / "run.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "plan_rate": 1.0,
                            "strict_stock_solve_any": 0.91,
                            "candidate_gt_reactant_in_pool": 0.58,
                            "exact_reaction_in_route_pool": 0.40,
                            "gt_reactant_in_route_pool": 0.63,
                            "avg_time_per_target_s": 6.5,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "D" / "run.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "plan_rate": 1.0,
                            "strict_stock_solve_any": 0.91,
                            "candidate_gt_reactant_in_pool": 0.58,
                            "exact_reaction_in_route_pool": 0.39,
                            "gt_reactant_in_route_pool": 0.61,
                            "avg_time_per_target_s": 6.4,
                        }
                    }
                ),
                encoding="utf-8",
            )

            status = _promotion_gate_status(root)

        self.assertTrue(status["promotable"])
        self.assertNotIn("exact_reaction_in_route_pool", status["metric_checks"])
        self.assertNotIn("gt_reactant_in_route_pool", status["metric_checks"])
        self.assertFalse(status["reference_recall_vs_B"]["exact_vs_B"])
        self.assertFalse(status["reference_recall_vs_B"]["gt_vs_B"])

    def test_completion_audit_uses_configured_relaxed_runtime_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "B").mkdir(parents=True)
            (root / "D").mkdir(parents=True)
            gates = {
                "plan_rate": 0.95,
                "strict_stock_solve_any": 0.60,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.40,
                "gt_reactant_in_route_pool": 0.63,
                "avg_time_per_target_s_max": 30.0,
            }
            (root / "reservoir_acceptance_manifest.json").write_text(
                json.dumps({"promotion_gates": gates}),
                encoding="utf-8",
            )
            summary = {
                "plan_rate": 1.0,
                "strict_stock_solve_any": 0.91,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.40,
                "gt_reactant_in_route_pool": 0.63,
                "avg_time_per_target_s": 27.5,
            }
            (root / "B" / "run.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
            (root / "D" / "run.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")

            relaxed = _promotion_gate_status(root)
            gates["avg_time_per_target_s_max"] = 16.0
            (root / "reservoir_acceptance_manifest.json").write_text(
                json.dumps({"promotion_gates": gates}),
                encoding="utf-8",
            )
            strict = _promotion_gate_status(root)

        self.assertTrue(relaxed["promotable"])
        self.assertTrue(relaxed["metric_checks"]["avg_time_per_target_s"])
        self.assertFalse(strict["promotable"])
        self.assertFalse(strict["metric_checks"]["avg_time_per_target_s"])

    def test_completion_audit_reports_append_only_without_online_promotion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for label in ["B", "C", "D", "D_APPEND"]:
                (root / label).mkdir(parents=True)
            gates = {
                "plan_rate": 0.95,
                "strict_stock_solve_any": 0.60,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.40,
                "gt_reactant_in_route_pool": 0.63,
                "avg_time_per_target_s_max": 30.0,
            }
            (root / "reservoir_acceptance_manifest.json").write_text(
                json.dumps({"promotion_gates": gates}),
                encoding="utf-8",
            )
            teacher = {
                "plan_rate": 1.0,
                "strict_stock_solve_any": 0.60,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.40,
                "gt_reactant_in_route_pool": 0.63,
                "avg_time_per_target_s": 20.0,
            }
            student = {
                "plan_rate": 1.0,
                "strict_stock_solve_any": 0.50,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.30,
                "gt_reactant_in_route_pool": 0.63,
                "avg_time_per_target_s": 20.0,
            }
            online_d = dict(student)
            online_d["avg_time_per_target_s"] = 45.0
            append = {
                "plan_rate": 1.0,
                "strict_stock_solve_any": 0.80,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.50,
                "gt_reactant_in_route_pool": 0.63,
                "avg_route_count": 8.0,
                "avg_time_per_target_s": 20.0,
                "avg_time_source": "autoplanner_reused_offline_append_only",
            }
            (root / "B" / "run.json").write_text(json.dumps({"summary": teacher}), encoding="utf-8")
            (root / "C" / "run.json").write_text(json.dumps({"summary": student}), encoding="utf-8")
            (root / "D" / "run.json").write_text(json.dumps({"summary": online_d}), encoding="utf-8")
            (root / "D_APPEND" / "run.json").write_text(json.dumps({"summary": append}), encoding="utf-8")

            status = _promotion_gate_status(root)

        self.assertFalse(status["promotable"])
        self.assertFalse(status["metric_checks"]["avg_time_per_target_s"])
        append_status = status["append_only_diagnostic"]
        self.assertTrue(append_status["available"])
        self.assertTrue(append_status["effect_gate_pass"])
        self.assertTrue(append_status["hybrid_append_only_candidate"])
        self.assertFalse(append_status["online_promotable"])
        self.assertEqual(append_status["avg_time_source"], "autoplanner_reused_offline_append_only")

    def test_completion_audit_scans_external_smoke_summaries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            required = ["paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"]
            for name, trust in [("external_strict", False), ("external_native_stock", True)]:
                out = root / name
                out.mkdir(parents=True)
                (out / "external_smoke_manifest.json").write_text(
                    json.dumps({"trust_native_stock": trust}),
                    encoding="utf-8",
                )
                (out / "external_smoke_summary.json").write_text(
                    json.dumps(
                        {
                            "ready": True,
                            "required": required,
                            "executed": required,
                            "rows": [
                                {
                                    "label": label,
                                    "strict_stock_solve_any": 1.0 if trust else 0.0,
                                    "avg_time_per_target_s": 1.0,
                                    "broad_reservoir_runtime_stock_routes": 5 if trust else 0,
                                    "native_payload_runtime_stock_routes": 0,
                                }
                                for label in required
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            evidence = _external_benchmark_evidence(distill_dir=root, acceptance_dir=root / "acceptance")

        self.assertTrue(evidence["ready"])
        self.assertEqual(len(evidence["summaries"]), 2)
        self.assertFalse(evidence["pure_strict_summary"]["native_stock_aligned"])
        self.assertTrue(evidence["native_stock_aligned_summary"]["native_stock_aligned"])

    def test_completion_audit_scans_append_only_external_deltas(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            required = ["paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"]
            out = root / "external_append_only"
            out.mkdir(parents=True)
            paired = [
                {
                    "dataset_label": "paroutes_n5",
                    "candidate_config": "D_APPEND",
                    "metric_deltas": {
                        "strict_stock_solve_any": 1.0,
                        "candidate_gt_reactant_in_pool": 0.0,
                        "candidate_exact_reaction_in_pool": 0.0,
                        "exact_reaction_in_route_pool": 0.0,
                        "gt_reactant_in_route_pool": 0.0,
                    },
                    "coverage_gains": ["strict_stock_solve_any"],
                    "coverage_losses": [],
                    "candidate_broad_reservoir_routes": 5,
                    "likely_change_cause": "append_only_bounded_reservoir_gain",
                },
                {
                    "dataset_label": "uspto_190",
                    "candidate_config": "D_APPEND",
                    "metric_deltas": {
                        "strict_stock_solve_any": 0.0,
                        "candidate_gt_reactant_in_pool": 0.0,
                        "candidate_exact_reaction_in_pool": 0.0,
                        "exact_reaction_in_route_pool": 1.0,
                        "gt_reactant_in_route_pool": 0.0,
                    },
                    "coverage_gains": ["exact_reaction_in_route_pool"],
                    "coverage_losses": [],
                    "candidate_broad_reservoir_routes": 5,
                    "likely_change_cause": "append_only_bounded_reservoir_gain",
                },
            ]
            (out / "external_smoke_summary.json").write_text(
                json.dumps(
                    {
                        "ready": True,
                        "required": required,
                        "executed": required,
                        "rows": [{"label": label, "strict_stock_solve_any": 1.0} for label in required],
                        "paired_config_deltas": paired,
                    }
                ),
                encoding="utf-8",
            )

            evidence = _external_benchmark_evidence(distill_dir=root, acceptance_dir=root / "acceptance")

        comparison = evidence["append_only_external_comparison"]
        self.assertTrue(comparison["effect_isolation_pass"])
        self.assertEqual(comparison["effect_gain_count"], 2)
        self.assertEqual(comparison["coverage_loss_count"], 0)
        self.assertFalse(comparison["online_runtime_measured"])

    def test_completion_audit_scans_adaptive_external_cd_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "external_adaptive"
            out.mkdir(parents=True)
            rows = []
            for config in ["C controller-only adaptive", "D prebuilt adaptive"]:
                for dataset in ["PaRoutes n1", "PaRoutes n5", "USPTO-190", "BioNavi-like adaptive"]:
                    rows.append(
                        {
                            "config": config,
                            "dataset": dataset,
                            "strict_stock_solve_any": 1.0 if config.startswith("D") else 0.5,
                            "candidate_gt_reactant_in_pool": 1.0,
                            "exact_reaction_in_route_pool": 1.0 if config.startswith("D") else 0.0,
                            "gt_reactant_in_route_pool": 1.0,
                            "avg_time_per_target_s": 27.0,
                        }
                    )
            (out / "external_smoke_summary_adaptive_cd.json").write_text(
                json.dumps(
                    {
                        "schema_version": "reservoir_external_adaptive_cd_summary.v1",
                        "rows": rows,
                        "aggregates": [
                            {
                                "config": "C controller-only adaptive",
                                "strict_stock_solve_any": 0.5,
                                "candidate_gt_reactant_in_pool": 1.0,
                                "exact_reaction_in_route_pool": 0.0,
                                "gt_reactant_in_route_pool": 1.0,
                                "avg_time_per_target_s": 27.5,
                            },
                            {
                                "config": "D prebuilt adaptive",
                                "strict_stock_solve_any": 1.0,
                                "candidate_gt_reactant_in_pool": 1.0,
                                "exact_reaction_in_route_pool": 1.0,
                                "gt_reactant_in_route_pool": 1.0,
                                "avg_time_per_target_s": 27.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            evidence = _external_benchmark_evidence(distill_dir=root, acceptance_dir=root / "acceptance")

        self.assertTrue(evidence["ready"])
        self.assertEqual(evidence["summaries"][0]["executed"], ["bionavi_like", "paroutes_n1", "paroutes_n5", "uspto_190"])
        self.assertIn("aggregates", evidence["summaries"][0])
        self.assertTrue(evidence["adaptive_cd_comparison"]["effect_first_pass"])
        self.assertEqual(evidence["adaptive_cd_comparison"]["deltas"]["strict_stock_solve_any"], 0.5)

    def test_publication_readiness_separates_internal_promotion_from_strict_publication(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            distill = root / "distill"
            acceptance = root / "acceptance"
            distill.mkdir()
            (acceptance / "reports").mkdir(parents=True)
            gates = {
                "plan_rate": 0.95,
                "strict_stock_solve_any": 0.60,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.40,
                "gt_reactant_in_route_pool": 0.63,
                "avg_time_per_target_s_max": 16.0,
            }
            (acceptance / "reservoir_acceptance_manifest.json").write_text(
                json.dumps({"promotion_gates": gates}),
                encoding="utf-8",
            )
            (acceptance / "completion_audit.json").write_text(
                json.dumps({"complete": True, "blocking_incomplete": []}),
                encoding="utf-8",
            )
            (acceptance / "reports" / "comparison.md").write_text("# comparison\n", encoding="utf-8")
            summaries = {
                "A": {"plan_rate": 0.97, "strict_stock_solve_any": 0.49, "candidate_gt_reactant_in_pool": 0.58, "candidate_exact_reaction_in_pool": 0.41, "exact_reaction_in_route_pool": 0.36, "gt_reactant_in_route_pool": 0.55, "avg_time_per_target_s": 6.3, "avg_route_count": 4.2},
                "B": {"plan_rate": 1.0, "strict_stock_solve_any": 0.91, "candidate_gt_reactant_in_pool": 0.58, "candidate_exact_reaction_in_pool": 0.41, "exact_reaction_in_route_pool": 0.40, "gt_reactant_in_route_pool": 0.63, "avg_time_per_target_s": 6.5, "avg_route_count": 9.2},
                "C": {"plan_rate": 0.97, "strict_stock_solve_any": 0.43, "candidate_gt_reactant_in_pool": 0.58, "candidate_exact_reaction_in_pool": 0.41, "exact_reaction_in_route_pool": 0.36, "gt_reactant_in_route_pool": 0.55, "avg_time_per_target_s": 6.4, "avg_route_count": 4.1},
                "D": {"plan_rate": 1.0, "strict_stock_solve_any": 0.91, "candidate_gt_reactant_in_pool": 0.58, "candidate_exact_reaction_in_pool": 0.41, "exact_reaction_in_route_pool": 0.40, "gt_reactant_in_route_pool": 0.63, "avg_time_per_target_s": 6.4, "avg_route_count": 7.9},
                "D_APPEND": {"plan_rate": 1.0, "strict_stock_solve_any": 1.0, "candidate_gt_reactant_in_pool": 0.58, "candidate_exact_reaction_in_pool": 0.41, "exact_reaction_in_route_pool": 0.40, "gt_reactant_in_route_pool": 0.63, "avg_time_per_target_s": 6.4, "avg_route_count": 9.1, "avg_time_source": "autoplanner_reused_offline_append_only"},
            }
            for label, summary in summaries.items():
                run_dir = acceptance / label
                run_dir.mkdir()
                run_dir.joinpath("run.json").write_text(
                    json.dumps({"summary": summary, "targets": [{"target_smiles": f"C{i}"} for i in range(100)]}),
                    encoding="utf-8",
                )
            external = distill / "external_smoke"
            external.mkdir()
            external.joinpath("external_smoke_summary.json").write_text(
                json.dumps(
                    {
                        "ready": True,
                        "rows": [
                            {"label": "C_paroutes_n1", "dataset_label": "paroutes_n1", "n_run_targets": 1},
                            {"label": "C_paroutes_n5", "dataset_label": "paroutes_n5", "n_run_targets": 1},
                            {"label": "C_uspto_190", "dataset_label": "uspto_190", "n_run_targets": 1},
                            {"label": "C_bionavi_like", "dataset_label": "bionavi_like", "n_run_targets": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = build_publication_readiness_report(
                distill_dir=distill,
                acceptance_dir=acceptance,
                output_json=root / "publication.json",
                output_md=root / "publication.md",
                external_min_targets=10,
            )

        self.assertTrue(report["criteria"]["internal_or_technical_report_ready"])
        self.assertTrue(report["criteria"]["limited_preprint_ready"])
        self.assertFalse(report["criteria"]["publication_ready_strict"])
        self.assertFalse(report["criteria"]["student_only_claim_supported"])
        self.assertFalse(report["criteria"]["external_scale_sufficient"])
        self.assertIn("Do not claim student-only", "\n".join(report["claims"]["blocked_or_needs_more_evidence"]))

    def test_statistical_report_enables_strict_publication_when_scale_is_available(self):
        def target(i, *, stock, cand_gt, cand_exact, exact, route_gt, time_s=1.0, routes=3):
            return {
                "target_smiles": f"C{i}O",
                "metrics": {"plan": True, "strict_stock_solve_any": stock},
                "route_recovery": {
                    "candidate_gt_reactant_in_pool": cand_gt,
                    "candidate_exact_reaction_in_pool": cand_exact,
                    "exact_reaction_in_route_pool": exact,
                    "gt_reactant_in_route_pool": route_gt,
                },
                "planner_output": {
                    "time_s": time_s,
                    "routes": [{"metrics": {"strict_stock_solve": stock}} for _ in range(routes)],
                },
            }

        def run_payload(n, *, stock_n, exact_n, route_gt_n, cand_gt_n=58, cand_exact_n=41, time_s=6.0, routes=4):
            rows = [
                target(
                    i,
                    stock=i < stock_n,
                    cand_gt=i < cand_gt_n,
                    cand_exact=i < cand_exact_n,
                    exact=i < exact_n,
                    route_gt=i < route_gt_n,
                    time_s=time_s,
                    routes=routes,
                )
                for i in range(n)
            ]
            return {"summary": summarize_target_results(rows, check_stock=True), "targets": rows}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            distill = root / "distill"
            acceptance = root / "acceptance"
            external = distill / "external_scaled"
            distill.mkdir()
            external.mkdir()
            (acceptance / "reports").mkdir(parents=True)
            gates = {
                "plan_rate": 0.95,
                "strict_stock_solve_any": 0.60,
                "candidate_gt_reactant_in_pool": 0.58,
                "exact_reaction_in_route_pool": 0.40,
                "gt_reactant_in_route_pool": 0.63,
                "avg_time_per_target_s_max": 16.0,
            }
            (acceptance / "reservoir_acceptance_manifest.json").write_text(json.dumps({"promotion_gates": gates}), encoding="utf-8")
            (acceptance / "completion_audit.json").write_text(json.dumps({"complete": True, "blocking_incomplete": []}), encoding="utf-8")
            (acceptance / "reports" / "comparison.md").write_text("# comparison\n", encoding="utf-8")
            full100_runs = {
                "A": run_payload(100, stock_n=49, exact_n=36, route_gt_n=55, time_s=6.3, routes=4),
                "B": run_payload(100, stock_n=91, exact_n=40, route_gt_n=63, time_s=6.5, routes=9),
                "C": run_payload(100, stock_n=43, exact_n=36, route_gt_n=55, time_s=6.4, routes=4),
                "D": run_payload(100, stock_n=91, exact_n=40, route_gt_n=63, time_s=6.4, routes=8),
                "D_APPEND": run_payload(100, stock_n=100, exact_n=40, route_gt_n=63, time_s=6.4, routes=9),
            }
            for label, payload in full100_runs.items():
                run_dir = acceptance / label
                run_dir.mkdir()
                (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
            summary_rows = []
            for dataset in ("paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"):
                for config in ("C", "D", "D_APPEND"):
                    label = f"{config}_{dataset}"
                    run_dir = external / label
                    run_dir.mkdir()
                    payload = run_payload(10, stock_n=7, exact_n=5, route_gt_n=6, cand_gt_n=5, cand_exact_n=4, time_s=8.0, routes=5)
                    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
                    summary_rows.append({"label": label, "config": config, "dataset_label": dataset, "n_run_targets": 10})
            external_summary = external / "external_smoke_summary.json"
            external_summary.write_text(
                json.dumps({"ready": True, "output_dir": str(external), "rows": summary_rows, "paired_config_deltas": []}),
                encoding="utf-8",
            )

            stat = build_statistical_report(
                acceptance_dir=acceptance,
                external_summary=external_summary,
                output_json=distill / "reservoir_statistical_report_test.json",
                output_md=distill / "reservoir_statistical_report_test.md",
                iterations=1000,
                seed=7,
            )
            report = build_publication_readiness_report(
                distill_dir=distill,
                acceptance_dir=acceptance,
                output_json=root / "publication.json",
                output_md=root / "publication.md",
                external_min_targets=10,
            )

        self.assertTrue(stat["ready"])
        self.assertTrue(report["criteria"]["external_scale_sufficient"])
        self.assertTrue(report["criteria"]["statistical_repeats_available"])
        self.assertTrue(report["criteria"]["publication_ready_strict"])
        self.assertFalse(report["criteria"]["student_only_claim_supported"])

    def test_student_route_gap_audit_separates_native_and_ordering_gaps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b = root / "B.json"
            c = root / "C.json"
            d = root / "D.json"
            b.write_text(
                json.dumps(
                    {
                        "targets": [
                            _route_gap_target(index=7, stock=True, cand_gt=False, exact=False, route_gt=True, broad_stock=True, source="native_template"),
                            _route_gap_target(index=8, stock=True, cand_gt=True, exact=True, route_gt=True, broad_stock=True, source="native_template"),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            c.write_text(
                json.dumps(
                    {
                        "targets": [
                            _route_gap_target(index=7, stock=False, cand_gt=False, exact=False, route_gt=False, broad_stock=False, source="enzyformer"),
                            _route_gap_target(index=8, stock=True, cand_gt=True, exact=False, route_gt=False, broad_stock=False, source="enzyformer"),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            d.write_text(
                json.dumps(
                    {
                        "targets": [
                            _route_gap_target(index=7, stock=True, cand_gt=False, exact=False, route_gt=True, broad_stock=True, source="native_template"),
                            _route_gap_target(index=8, stock=True, cand_gt=True, exact=True, route_gt=True, broad_stock=True, source="native_template"),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = analyze_student_route_composition_gaps(
                runs={"B": b, "C": c, "D": d},
                teacher_label="B",
                student_label="C",
                hybrid_label="D",
                output_json=root / "audit.json",
                output_md=root / "audit.md",
                top_n=10,
            )

        self.assertEqual(report["summary"]["gap_class_counts"]["native_route_only_stock_gap"], 1)
        self.assertEqual(report["summary"]["gap_class_counts"]["route_composition_or_order_gap"], 1)
        by_index = {row["index"]: row for row in report["rows"]}
        self.assertEqual(by_index[7]["student_next_action"].split(";")[0], "distill native route-composition steps or add a non-eval native-route replay source")
        self.assertEqual(by_index[8]["configs"]["C"]["best_routes"]["selected"]["source_counts"], {"enzyformer": 1})

    def test_native_route_replay_pack_writes_eval_only_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "run.json"
            out = root / "native_replay.jsonl"
            report_path = root / "native_replay_report.json"
            run.write_text(json.dumps(_native_replay_run()), encoding="utf-8")

            report = build_native_route_replay_pack(
                run_path=run,
                output_pack=out,
                report_path=report_path,
                split="eval",
                indices={7},
            )
            rows = _read_jsonl(out)

        self.assertEqual(report["rows"], 1)
        self.assertTrue(rows[0]["eval_only"])
        self.assertEqual(rows[0]["source"], "chemtemplates")
        self.assertEqual(rows[0]["source_policy_group"], "template")
        self.assertTrue(rows[0]["teacher_stock_closed"])
        self.assertTrue(rows[0]["teacher_gt_reactant_hit"])
        self.assertGreater(rows[0]["teacher_route_value"], 0.0)
        self.assertLessEqual(rows[0]["teacher_route_value"], 1.0)
        self.assertIn("teacher_route_cost", rows[0])

    def test_native_route_replay_rejects_full100_training_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "run.json"
            run.write_text(json.dumps(_native_replay_run(benchmark="data/benchmark_v2_100.json")), encoding="utf-8")

            with self.assertRaises(ValueError):
                build_native_route_replay_pack(
                    run_path=run,
                    output_pack=root / "native_replay.jsonl",
                    report_path=root / "native_replay_report.json",
                    split="train",
                )

    def test_native_replay_proposal_source_returns_non_eval_rows(self):
        old_path = os.environ.get("AUTOPLANNER_NATIVE_REPLAY_PROPOSALS")
        old_enable = os.environ.get("AUTOPLANNER_ENABLE_NATIVE_REPLAY_PROPOSALS")
        old_allow = os.environ.get("AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY")
        old_floor = os.environ.get("AUTOPLANNER_NATIVE_REPLAY_MIN_BUDGET")
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                run = root / "run.json"
                pack = root / "native_replay.jsonl"
                run.write_text(json.dumps(_native_replay_run()), encoding="utf-8")
                build_native_route_replay_pack(
                    run_path=run,
                    output_pack=pack,
                    report_path=root / "native_replay_report.json",
                    split="train",
                )
                proposals_module._NATIVE_REPLAY_CACHE.clear()
                os.environ["AUTOPLANNER_NATIVE_REPLAY_PROPOSALS"] = str(pack)
                os.environ["AUTOPLANNER_ENABLE_NATIVE_REPLAY_PROPOSALS"] = "1"
                os.environ["AUTOPLANNER_NATIVE_REPLAY_MIN_BUDGET"] = "2"
                os.environ.pop("AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY", None)

                tool = RetroEngineProposalTool({}, source_order=("native_replay",))
                actions = tool.propose("CCO", ProposalContext(), top_k=3)
                floors = proposals_module._contextual_source_budget_floors(
                    ProposalContext(depth=0),
                    sources=["retrochimera", "native_replay"],
                    total_budget=4,
                )

            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0].source, "native_replay")
            self.assertTrue(actions[0].metadata["native_replay"])
            self.assertEqual(actions[0].main_reactant, "CC")
            self.assertEqual(floors["native_replay"], 2)
        finally:
            proposals_module._NATIVE_REPLAY_CACHE.clear()
            _restore_env("AUTOPLANNER_NATIVE_REPLAY_PROPOSALS", old_path)
            _restore_env("AUTOPLANNER_ENABLE_NATIVE_REPLAY_PROPOSALS", old_enable)
            _restore_env("AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY", old_allow)
            _restore_env("AUTOPLANNER_NATIVE_REPLAY_MIN_BUDGET", old_floor)

    def test_native_replay_proposal_source_ignores_eval_rows_by_default(self):
        old_path = os.environ.get("AUTOPLANNER_NATIVE_REPLAY_PROPOSALS")
        old_enable = os.environ.get("AUTOPLANNER_ENABLE_NATIVE_REPLAY_PROPOSALS")
        old_allow = os.environ.get("AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY")
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                run = root / "run.json"
                pack = root / "native_replay_eval.jsonl"
                run.write_text(json.dumps(_native_replay_run(benchmark="data/benchmark_v2_100.json")), encoding="utf-8")
                build_native_route_replay_pack(
                    run_path=run,
                    output_pack=pack,
                    report_path=root / "native_replay_report.json",
                    split="eval",
                )
                proposals_module._NATIVE_REPLAY_CACHE.clear()
                os.environ["AUTOPLANNER_NATIVE_REPLAY_PROPOSALS"] = str(pack)
                os.environ["AUTOPLANNER_ENABLE_NATIVE_REPLAY_PROPOSALS"] = "1"
                os.environ.pop("AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY", None)

                tool = RetroEngineProposalTool({}, source_order=("native_replay",))
                blocked = tool.propose("CCO", ProposalContext(), top_k=3)
                os.environ["AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY"] = "1"
                proposals_module._NATIVE_REPLAY_CACHE.clear()
                diagnostic = tool.propose("CCO", ProposalContext(), top_k=3)

            self.assertEqual(blocked, [])
            self.assertEqual(len(diagnostic), 1)
            self.assertTrue(diagnostic[0].metadata["native_replay_eval_only"])
        finally:
            proposals_module._NATIVE_REPLAY_CACHE.clear()
            _restore_env("AUTOPLANNER_NATIVE_REPLAY_PROPOSALS", old_path)
            _restore_env("AUTOPLANNER_ENABLE_NATIVE_REPLAY_PROPOSALS", old_enable)
            _restore_env("AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY", old_allow)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stock_closed_pack_row():
    return {
        "target_smiles": "CCO",
        "leaf": "CCO",
        "source": "retrochimera",
        "candidate_reaction": "CC.O>>CCO",
        "reactants": ["CC", "O"],
        "budget_label": "1x",
        "teacher_route_value": 0.50,
        "teacher_action_value": 0.50,
        "teacher_route_cost": 1.0,
        "teacher_action_cost": 1.0,
        "teacher_value_policy": "reaction_cost_and_or.v1",
        "teacher_stock_closed": True,
        "eval_only": False,
    }


def _route_gap_target(*, index, stock, cand_gt, exact, route_gt, broad_stock, source):
    selected_rank = 1
    return {
        "index": index,
        "target_smiles": f"C{index}O",
        "route_domain": "chemoenzymatic",
        "metrics": {
            "plan": True,
            "strict_stock_solve_any": bool(stock),
            "strict_stock_first_rank": selected_rank if stock else None,
        },
        "route_recovery": {
            "candidate_gt_reactant_in_pool": bool(cand_gt),
            "candidate_exact_reaction_in_pool": bool(exact),
            "exact_reaction_in_route_pool": bool(exact),
            "exact_reaction_first_rank": selected_rank if exact else None,
            "gt_reactant_in_route_pool": bool(route_gt),
            "recovery_bottleneck": "synthetic",
            "per_route": [{"gt_reactant_hit": bool(route_gt)}],
        },
        "planner_output": {
            "routes": [
                {
                    "score": 1.0,
                    "metrics": {
                        "strict_stock_solve": bool(stock),
                        "terminal_reactants": [f"C{index}"],
                        "candidate_source_counts": {source: 1},
                    },
                    "steps": [
                        {
                            "source": source,
                            "reaction_type": "reduction",
                            "stock_status": {f"C{index}": bool(stock)},
                        }
                    ],
                }
            ],
            "broad_reservoir": {
                "enabled": bool(broad_stock),
                "native_topk": 5 if broad_stock else 0,
                "native_route_count": 5 if broad_stock else 0,
                "routes": [{"route_rank": 1, "native_rank": 1, "stock_closed": True}] if broad_stock else [],
            },
        },
    }


def _native_replay_run(benchmark="data/train_like.json"):
    return {
        "metadata": {"benchmark": benchmark},
        "targets": [
            {
                "index": 7,
                "target_smiles": "CCO",
                "cascade_id": "case_7",
                "route_recovery": {
                    "per_route": [{"gt_reactant_hit": True, "exact_reaction_hit": False}],
                },
                "planner_output": {
                    "broad_reservoir": {
                        "enabled": True,
                        "native_topk": 5,
                        "native_route_count": 1,
                        "routes": [{"route_rank": 1, "native_rank": 1, "stock_closed": True}],
                    },
                    "routes": [
                        {
                            "metrics": {
                                "strict_stock_solve": True,
                                "terminal_reactants": ["CC"],
                                "candidate_source_counts": {"([C:1])>>[C:1]O": 1},
                            },
                            "steps": [
                                {
                                    "index": 0,
                                    "product": "CCO",
                                    "main_reactant": "CC",
                                    "aux_reactants": [],
                                    "reaction_smiles": "CC>>CCO",
                                    "source": "([C:1])>>[C:1]O",
                                    "reaction_type": "oxidation",
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    }


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
