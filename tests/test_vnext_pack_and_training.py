import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from cascade_planner.eval.build_vnext_pack import build_vnext_pack
from cascade_planner.eval.backfill_route_tree_trace_stock import backfill_trace_row
from cascade_planner.eval.train_vnext_from_pack import (
    build_route_tree_search_policy_dataset,
    build_candidate_pool_dataset,
    build_route_state_dataset,
    build_search_policy_dataset,
    build_step_pair_dataset,
    train_search_policy_from_route_tree_traces,
    train_candidate_pool_ranker_from_vnext_pack,
    train_route_state_from_vnext_pack,
    train_search_policy_from_vnext_pack,
    train_step_encoder_from_vnext_pack,
)
from cascade_planner.eval.calibrate_route_tree_value import calibrate_route_tree_value_from_traces
from cascade_planner.route_tree.runtime import RouteTreeRuntime
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.schema import RouteTreeState
from cascade_planner.vnext.features import node_feature_dim, open_leaf_feature_matrix
from cascade_planner.vnext.runtime import VNextRuntime, default_vnext_runtime, vnext_candidate_weight
from cascade_planner.vnext.schema import VNEXT_SCHEMA_VERSION


def _candidate_row(target, route_id, rank, label, label_type, main, aux, score):
    return {
        "candidate_id": f"{route_id}_{rank}",
        "route_id": route_id,
        "target_smiles": target,
        "product": target,
        "step_index": 0,
        "rank": rank,
        "label": label,
        "label_type": label_type,
        "weight": 2.0 if label >= 0.75 else 1.0,
        "gt_available": True,
        "exact_gt_reaction": label_type == "benchmark_exact",
        "exact_gt_reactants": label_type == "benchmark_exact",
        "selected_exact": label >= 0.5,
        "candidate": {
            "main_reactant": main,
            "aux_reactants": aux,
            "source": "retrochimera",
            "score": score,
            "type": "hydrolysis" if aux == ["O"] else "other",
            "rxn_smiles": f"{main}.{'.'.join(aux) if aux else 'O'}>>{target}",
        },
        "features": {
            "candidate_score": score,
            "stock_fraction": 1.0 if label else 0.0,
            "main_reduction": 0.5 if label else 0.0,
            "has_ec": 0.0,
            "has_evidence": 0.0,
            "large_aux_penalty": 0.0,
            "self_loop": 0.0 if label else 1.0,
        },
    }


def _route_row(target, route_id, solved):
    return {
        "route_id": route_id,
        "target_smiles": target,
        "label": 1.0 if solved else 0.25,
        "label_type": "professional_solved" if solved else "filled_only",
        "n_steps": 1,
        "type_sequence": ["hydrolysis"],
        "ec1_sequence": ["3" if solved else ""],
        "source_sequence": ["retrochimera"],
        "operation_mode": "sequential_isolated",
        "recovery_bottleneck_labels": [] if solved else ["stock_dead_end"],
        "features": {
            "filled_route": 1.0,
            "progressive_route": 1.0 if solved else 0.0,
            "route_solved": 1.0 if solved else 0.0,
            "strict_stock_solve": 1.0 if solved else -0.5,
            "main_chain_reduction": 0.7 if solved else 0.0,
            "leaf_reduction": 0.7 if solved else 0.0,
            "naturalness": 1.0 if solved else 0.0,
            "condition_success": 1.0,
            "compatibility_success": 1.0 if solved else 0.0,
            "enzyme_evidence": 0.0,
            "issue_count": 0.0 if solved else 1.0,
        },
    }


class VNextPackAndTrainingTest(unittest.TestCase):
    def test_open_leaf_features_include_parent_adjacency_and_extra_context(self):
        features, mask = open_leaf_feature_matrix(
            target="CCCCCCCC",
            open_leaves=["CCCCCCCC", "CCCC"],
            parent_reactants=["CCCC"],
            n_bits=16,
            max_open_leaves=4,
        )

        self.assertEqual(features.shape, (4, node_feature_dim(16)))
        self.assertEqual(mask.tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(features[0, -8], 0.0)
        self.assertEqual(features[1, -8], 1.0)

    def test_build_pack_train_models_and_load_runtime(self):
        candidate_rows = []
        route_rows = []
        for target, aux in [("CCO", ["O"]), ("CCN", ["N"])]:
            for route_idx in range(2):
                route_id = f"{target}_{route_idx}"
                candidate_rows.append(_candidate_row(target, route_id, 1, 1.0, "benchmark_exact", "CC", aux, 0.9))
                candidate_rows.append(_candidate_row(target, route_id, 2, 0.0, "negative", target, [], 0.1))
                route_rows.append(_route_row(target, route_id, solved=route_idx == 0))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            (pack / "candidate_ranking.jsonl").write_text("\n".join(json.dumps(row) for row in candidate_rows), encoding="utf-8")
            (pack / "route_value.jsonl").write_text("\n".join(json.dumps(row) for row in route_rows), encoding="utf-8")
            vnext = root / "vnext_pack"
            manifest = build_vnext_pack(pack_dir=pack, output_dir=vnext, max_candidates=4)

            step_ds = build_step_pair_dataset(vnext, n_bits=16)
            pool_ds = build_candidate_pool_dataset(vnext, n_bits=16, max_candidates=4)
            route_ds = build_route_state_dataset(vnext, max_steps=4)
            policy_ds = build_search_policy_dataset(vnext, n_bits=16, max_candidates=4, max_steps=4)

            step_report = train_step_encoder_from_vnext_pack(
                vnext_pack_dir=vnext,
                model_output=root / "step.pt",
                report_output=root / "step.json",
                epochs=1,
                batch_size=4,
                n_bits=16,
                d_model=16,
            )
            pool_report = train_candidate_pool_ranker_from_vnext_pack(
                vnext_pack_dir=vnext,
                model_output=root / "pool.pt",
                report_output=root / "pool.json",
                epochs=1,
                batch_size=2,
                n_bits=16,
                max_candidates=4,
                d_model=16,
            )
            route_report = train_route_state_from_vnext_pack(
                vnext_pack_dir=vnext,
                model_output=root / "route.pt",
                report_output=root / "route.json",
                epochs=1,
                batch_size=2,
                max_steps=4,
                d_model=16,
            )
            policy_report = train_search_policy_from_vnext_pack(
                vnext_pack_dir=vnext,
                model_output=root / "policy.pt",
                report_output=root / "policy.json",
                epochs=1,
                batch_size=2,
                n_bits=16,
                max_candidates=4,
                max_steps=4,
                d_model=16,
            )
            runtime = VNextRuntime(root / "pool.pt")
            score = runtime.score_candidate("CCO", {"main_reactant": "CC", "aux_reactants": ["O"], "score": 0.9})

        self.assertEqual(manifest["schema_version"], VNEXT_SCHEMA_VERSION)
        self.assertEqual(manifest["counts"]["step_pairs"], 8)
        self.assertEqual(pool_ds.candidate_features.shape[:2], (4, 4))
        self.assertEqual(step_ds.product_fp.shape[0], 8)
        self.assertEqual(route_ds.route_features.shape[0], 4)
        self.assertEqual(route_ds.value.shape[0], 4)
        self.assertEqual(policy_ds.action_features.shape[:2], (4, 4))
        self.assertEqual(policy_ds.node_features.shape[0], 4)
        self.assertEqual(policy_ds.node_labels[:, 0].sum(), 4.0)
        self.assertEqual(policy_ds.solved.shape[0], 4)
        self.assertEqual(policy_ds.value.shape[0], 4)
        self.assertEqual(policy_ds.bottlenecks.shape[0], 4)
        self.assertIn("best_val_loss", step_report)
        self.assertIn("best_val_loss", pool_report)
        self.assertIn("best_val_loss", route_report)
        self.assertIn("best_val_loss", policy_report)
        self.assertTrue(route_report["metadata"]["value_head_supervised"])
        self.assertTrue(policy_report["metadata"]["value_head_supervised"])
        self.assertIn("val_value_mae", route_report["history"][-1])
        self.assertIn("val_value_mae", policy_report["history"][-1])
        self.assertIn("val_value_ece", policy_report["history"][-1])
        self.assertIn("val_node_top1", policy_report["history"][-1])
        self.assertIn("val_top5_positive", policy_report["history"][-1])
        self.assertFalse(policy_report["metadata"]["value_calibrated"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_default_runtime_flag_and_weight(self):
        old_enable = os.environ.get("AUTOPLANNER_ENABLE_VNEXT")
        old_weight = os.environ.get("AUTOPLANNER_VNEXT_CANDIDATE_WEIGHT")
        try:
            os.environ.pop("AUTOPLANNER_ENABLE_VNEXT", None)
            self.assertIsNone(default_vnext_runtime())
            os.environ["AUTOPLANNER_VNEXT_CANDIDATE_WEIGHT"] = "2.5"
            self.assertEqual(vnext_candidate_weight(), 2.5)
            os.environ["AUTOPLANNER_VNEXT_CANDIDATE_WEIGHT"] = "9"
            self.assertEqual(vnext_candidate_weight(), 3.0)
        finally:
            if old_enable is None:
                os.environ.pop("AUTOPLANNER_ENABLE_VNEXT", None)
            else:
                os.environ["AUTOPLANNER_ENABLE_VNEXT"] = old_enable
            if old_weight is None:
                os.environ.pop("AUTOPLANNER_VNEXT_CANDIDATE_WEIGHT", None)
            else:
                os.environ["AUTOPLANNER_VNEXT_CANDIDATE_WEIGHT"] = old_weight

    def test_route_tree_traces_train_search_policy_value_heads(self):
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.9,
                "source": "fake_chem",
                "type": "coupling",
            },
        )
        bad_action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCCCCCC",
                "aux_reactants": [],
                "rxn_smiles": "CCCCCCCC>>CCCCCCCC",
                "score": 0.1,
                "source": "fake_chem",
                "type": "self_loop",
            },
        )
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "target_smiles": "CCCCCCCC",
            "route_metrics": [{"strict_stock_solve": True, "progressive_route": True, "route_solved": True}],
            "event": {
                "state_id": "s0",
                "state": {
                    "state_id": "s0",
                    "target": "CCCCCCCC",
                    "depth": 0,
                    "open_leaves": ["CCCCCCCC"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 0,
                "open_leaves": ["CCCCCCCC"],
                "expanded_leaf": "CCCCCCCC",
                "candidate_actions": [action.to_dict(), bad_action.to_dict()],
                "selected_action_key": action.canonical_key,
                "selected_next_state_id": "s1",
                "action_scores": [1.0],
                "model_active": False,
                "outcome": {"search_status": "solved", "solved_routes": 1},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "route_tree_traces.jsonl"
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            dataset = build_route_tree_search_policy_dataset(trace_path, n_bits=16, max_candidates=4, max_steps=4)
            report = train_search_policy_from_route_tree_traces(
                trace_paths=trace_path,
                model_output=root / "route_tree_policy.pt",
                report_output=root / "route_tree_policy.json",
                epochs=1,
                batch_size=1,
                n_bits=16,
                max_candidates=4,
                max_steps=4,
                d_model=16,
                device="cpu",
            )
            runtime = RouteTreeRuntime(root / "route_tree_policy.pt")
            runtime_eval = runtime.evaluate(RouteTreeState.initial("CCCCCCCC"), "CCCCCCCC", [action])
            calibration = calibrate_route_tree_value_from_traces(
                checkpoint_path=root / "route_tree_policy.pt",
                trace_paths=[trace_path],
                output_json=root / "calibration.json",
                output_md=root / "calibration.md",
                frozen_checkpoint_path=root / "route_tree_policy_frozen.pt",
                validation_set_id="unit_locked",
                device="cpu",
                min_rows=1,
                max_ece=1.0,
            )
            frozen_runtime = RouteTreeRuntime(root / "route_tree_policy_frozen.pt")
            frozen_eval = frozen_runtime.evaluate(RouteTreeState.initial("CCCCCCCC"), "CCCCCCCC", [action])

        self.assertEqual(dataset.action_features.shape[:2], (1, 4))
        self.assertGreater(dataset.node_labels[0, 0], 0.4)
        self.assertLessEqual(dataset.node_labels[0, 0], 1.0)
        self.assertEqual(dataset.action_labels[0, 0], 1.0)
        self.assertEqual(dataset.solved[0], 1.0)
        self.assertIn("best_val_loss", report)
        self.assertEqual(report["metadata"]["training_source"], "route_tree_traces")
        self.assertIn("val_node_top1", report["history"][-1])
        self.assertFalse(report["metadata"]["value_calibrated"])
        self.assertTrue(runtime_eval.model_active)
        self.assertEqual(len(runtime_eval.action_scores), 1)
        self.assertFalse(runtime_eval.value_calibrated)
        self.assertTrue(calibration["calibration_accepted"])
        self.assertTrue(frozen_eval.value_calibrated)
        self.assertGreater(frozen_eval.route_value, 0.0)

    def test_route_tree_trace_node_labels_are_leaf_utility_not_selected_imitation(self):
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.9,
                "source": "fake_chem",
                "type": "coupling",
            },
        )
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "benchmark_index": 7,
            "target_smiles": "CCCCCCCC",
            "gt_route": [{"rxn_smiles": "CCCC.CCCC>>CCCCCCCC"}],
            "route_metrics": [{"strict_stock_solve": True, "progressive_route": True, "route_solved": True}],
            "event": {
                "state_id": "s0",
                "state": {
                    "state_id": "s0",
                    "target": "CCCCCCCC",
                    "depth": 1,
                    "open_leaves": ["CCCCCCCC", "CCCC"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 1,
                "open_leaves": ["CCCCCCCC", "CCCC"],
                "expanded_leaf": "CCCC",
                "candidate_actions": [action.to_dict()],
                "selected_action_key": "",
                "proposal_diagnostics": [
                    {"leaf": "CCCC", "proposal_budget": 1, "raw_actions": 0, "final_actions": 0}
                ],
                "outcome": {"search_status": "failed", "solved_routes": 0},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "trace.jsonl"
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            dataset = build_route_tree_search_policy_dataset(
                trace_path,
                n_bits=16,
                max_candidates=4,
                max_steps=4,
                feature_cache_dir=None,
            )

        self.assertEqual(dataset.node_mask[0, :2].tolist(), [1.0, 1.0])
        self.assertLess(dataset.node_labels[0, 1], 0.5)
        self.assertEqual(dataset.feature_schema["node_label_target"], "trace_leaf_utility")

    def test_stock_aware_route_tree_labels_prefer_late_stock_closure(self):
        action = CandidateAction.from_candidate(
            "CCO",
            {
                "main_reactant": "C",
                "aux_reactants": ["CO"],
                "rxn_smiles": "C.CO>>CCO",
                "score": 0.9,
                "source": "retrochimera",
                "type": "disconnection",
            },
        )
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "benchmark_index": 8,
            "target_smiles": "CCCCCCCC",
            "route_metrics": [{"strict_stock_solve": True, "progressive_route": True, "route_solved": True}],
            "event": {
                "state_id": "s_late",
                "state": {
                    "state_id": "s_late",
                    "target": "CCCCCCCC",
                    "depth": 5,
                    "open_leaves": ["CCCCCCCC", "CCO"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 5,
                "open_leaves": ["CCCCCCCC", "CCO"],
                "expanded_leaf": "CCO",
                "candidate_actions": [action.to_dict()],
                "selected_action_key": action.canonical_key,
                "selected_next_stock_closed": True,
                "selected_next_open_leaves": 0,
                "expanded_leaf_stock_hit": True,
                "expanded_leaf_parent_adjacent": True,
                "expanded_leaf_low_yield": False,
                "proposal_diagnostics": [
                    {"leaf": "CCO", "proposal_budget": 2, "raw_actions": 1, "final_actions": 1}
                ],
                "outcome": {"search_status": "solved", "solved_routes": 1},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "trace.jsonl"
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            dataset = build_route_tree_search_policy_dataset(
                trace_path,
                n_bits=16,
                max_candidates=4,
                max_steps=6,
                node_label_target="stock_aware_leaf_utility",
            )

        self.assertEqual(dataset.feature_schema["node_label_target"], "stock_aware_leaf_utility")
        self.assertGreater(dataset.node_labels[0, 1], dataset.node_labels[0, 0])
        self.assertGreater(dataset.node_labels[0, 1], 0.5)

    def test_stock_aware_action_labels_reward_stock_closing_selection(self):
        action = CandidateAction.from_candidate(
            "CCO",
            {
                "main_reactant": "C",
                "aux_reactants": ["CO"],
                "rxn_smiles": "C.CO>>CCO",
                "score": 0.9,
                "source": "retrochimera",
                "type": "disconnection",
            },
        )
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "benchmark_index": 9,
            "target_smiles": "CCO",
            "route_metrics": [{"strict_stock_solve": False, "progressive_route": True, "route_solved": False}],
            "event": {
                "state_id": "s_stock_action",
                "state": {
                    "state_id": "s_stock_action",
                    "target": "CCO",
                    "depth": 5,
                    "open_leaves": ["CCO"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 5,
                "open_leaves": ["CCO"],
                "expanded_leaf": "CCO",
                "candidate_actions": [action.to_dict()],
                "selected_action_key": action.canonical_key,
                "selected_next_stock_closed": True,
                "selected_next_open_leaves": 0,
                "proposal_diagnostics": [
                    {"leaf": "CCO", "proposal_budget": 2, "raw_actions": 1, "final_actions": 1}
                ],
                "outcome": {"search_status": "partial", "solved_routes": 0},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "trace.jsonl"
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            legacy = build_route_tree_search_policy_dataset(
                trace_path,
                n_bits=16,
                max_candidates=4,
                max_steps=6,
            )
            stock_aware = build_route_tree_search_policy_dataset(
                trace_path,
                n_bits=16,
                max_candidates=4,
                max_steps=6,
                action_label_target="stock_aware_action_utility",
            )

        self.assertEqual(legacy.action_labels[0, 0], 0.0)
        self.assertGreater(stock_aware.action_labels[0, 0], 0.5)
        self.assertGreater(stock_aware.action_weights[0, 0], legacy.action_weights[0, 0])
        self.assertEqual(stock_aware.feature_schema["action_label_target"], "stock_aware_action_utility")

    def test_counterfactual_action_labels_penalize_selected_dead_end(self):
        action = CandidateAction.from_candidate(
            "CCO",
            {
                "main_reactant": "CCO",
                "aux_reactants": [],
                "rxn_smiles": "CCO>>CCO",
                "score": 0.1,
                "source": "retrochimera",
                "type": "self_loop",
            },
        )
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "benchmark_index": 10,
            "target_smiles": "CCO",
            "route_metrics": [{"strict_stock_solve": False, "progressive_route": False, "route_solved": False}],
            "event": {
                "state_id": "s_dead_action",
                "state": {
                    "state_id": "s_dead_action",
                    "target": "CCO",
                    "depth": 5,
                    "open_leaves": ["CCO"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 5,
                "open_leaves": ["CCO"],
                "expanded_leaf": "CCO",
                "candidate_actions": [action.to_dict()],
                "selected_action_key": action.canonical_key,
                "selected_next_stock_closed": False,
                "selected_next_open_leaves": 1,
                "expanded_leaf_low_yield": True,
                "outcome": {"search_status": "partial", "solved_routes": 0, "dead_ends": 1},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "trace.jsonl"
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            stock_aware = build_route_tree_search_policy_dataset(
                trace_path,
                n_bits=16,
                max_candidates=4,
                max_steps=6,
                action_label_target="stock_aware_action_utility",
            )
            counterfactual = build_route_tree_search_policy_dataset(
                trace_path,
                n_bits=16,
                max_candidates=4,
                max_steps=6,
                action_label_target="stock_counterfactual_action_utility",
            )

        self.assertGreaterEqual(stock_aware.action_labels[0, 0], 0.35)
        self.assertLess(counterfactual.action_labels[0, 0], stock_aware.action_labels[0, 0])
        self.assertLessEqual(counterfactual.action_labels[0, 0], 0.22)
        self.assertGreater(counterfactual.action_weights[0, 0], stock_aware.action_weights[0, 0])
        self.assertEqual(counterfactual.feature_schema["action_label_target"], "stock_counterfactual_action_utility")

    def test_counterfactual_action_labels_promote_stock_closing_sibling(self):
        selected_action = CandidateAction.from_candidate(
            "CCO",
            {
                "main_reactant": "CCO",
                "aux_reactants": [],
                "rxn_smiles": "CCO>>CCO",
                "score": 0.1,
                "source": "retrochimera",
                "type": "self_loop",
            },
        )
        selected = selected_action.to_dict()
        sibling = CandidateAction.from_candidate(
            "CCO",
            {
                "main_reactant": "C",
                "aux_reactants": ["CO"],
                "rxn_smiles": "C.CO>>CCO",
                "score": 0.8,
                "source": "retrochimera",
                "type": "disconnection",
            },
        ).to_dict()
        sibling["reactant_stock_status"] = {"C": True, "CO": True}
        sibling["reactant_stock_fraction"] = 1.0
        sibling["stock_closing_candidate"] = True
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "benchmark_index": 11,
            "target_smiles": "CCO",
            "route_metrics": [{"strict_stock_solve": False, "progressive_route": False, "route_solved": False}],
            "event": {
                "state_id": "s_sibling_stock",
                "state": {
                    "state_id": "s_sibling_stock",
                    "target": "CCO",
                    "depth": 5,
                    "open_leaves": ["CCO"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 5,
                "open_leaves": ["CCO"],
                "expanded_leaf": "CCO",
                "candidate_actions": [selected, sibling],
                "selected_action_key": selected_action.canonical_key,
                "selected_next_stock_closed": False,
                "selected_next_open_leaves": 1,
                "expanded_leaf_low_yield": True,
                "outcome": {"search_status": "partial", "solved_routes": 0, "dead_ends": 1},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "trace.jsonl"
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            counterfactual = build_route_tree_search_policy_dataset(
                trace_path,
                n_bits=16,
                max_candidates=4,
                max_steps=6,
                action_label_target="stock_counterfactual_action_utility",
            )

        self.assertLessEqual(counterfactual.action_labels[0, 0], 0.22)
        self.assertGreaterEqual(counterfactual.action_labels[0, 1], 0.62)
        self.assertGreater(counterfactual.action_labels[0, 1], counterfactual.action_labels[0, 0])
        self.assertGreater(counterfactual.action_weights[0, 1], 1.0)

    def test_backfill_trace_stock_fields_enables_sibling_stock_labels(self):
        row = {
            "event": {
                "candidate_actions": [
                    {
                        "reactants": ["C", "CO"],
                        "main_reactant": "C",
                        "aux_reactants": ["CO"],
                    },
                    {
                        "reactants": ["CCCCCCCC"],
                        "main_reactant": "CCCCCCCC",
                        "aux_reactants": [],
                    },
                ]
            }
        }

        enriched = backfill_trace_row(row, stock_checker=lambda smi: smi in {"C", "CO"})
        actions = enriched["event"]["candidate_actions"]

        self.assertTrue(enriched["event"]["candidate_stock_enriched"])
        self.assertEqual(actions[0]["reactant_stock_fraction"], 1.0)
        self.assertTrue(actions[0]["stock_closing_candidate"])
        self.assertEqual(actions[1]["reactant_stock_fraction"], 0.0)
        self.assertFalse(actions[1]["stock_closing_candidate"])

    def test_action_only_policy_training_freezes_non_action_heads(self):
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.9,
                "source": "fake_chem",
                "type": "coupling",
            },
        )
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "target_smiles": "CCCCCCCC",
            "route_metrics": [{"strict_stock_solve": False, "progressive_route": True, "route_solved": False}],
            "event": {
                "state_id": "s0",
                "state": {
                    "state_id": "s0",
                    "target": "CCCCCCCC",
                    "depth": 0,
                    "open_leaves": ["CCCCCCCC"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 0,
                "open_leaves": ["CCCCCCCC"],
                "expanded_leaf": "CCCCCCCC",
                "candidate_actions": [action.to_dict()],
                "selected_action_key": action.canonical_key,
                "selected_next_stock_closed": True,
                "selected_next_open_leaves": 0,
                "outcome": {"search_status": "partial", "solved_routes": 0},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "trace.jsonl"
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            train_search_policy_from_route_tree_traces(
                trace_paths=trace_path,
                model_output=root / "base.pt",
                report_output=root / "base.json",
                epochs=1,
                batch_size=1,
                n_bits=16,
                max_candidates=4,
                max_steps=4,
                d_model=16,
                device="cpu",
            )
            report = train_search_policy_from_route_tree_traces(
                trace_paths=trace_path,
                model_output=root / "action_only.pt",
                report_output=root / "action_only.json",
                epochs=1,
                batch_size=1,
                n_bits=16,
                max_candidates=4,
                max_steps=4,
                d_model=16,
                device="cpu",
                init_checkpoint=root / "base.pt",
                action_label_target="stock_aware_action_utility",
                policy_train_mode="action_only",
                action_rank_loss_weight=0.25,
            )
            base = torch.load(root / "base.pt", map_location="cpu")["state_dict"]
            action_only = torch.load(root / "action_only.pt", map_location="cpu")["state_dict"]
            runtime = RouteTreeRuntime(root / "base.pt", action_policy_path=root / "action_only.pt")
            runtime_eval = runtime.evaluate(RouteTreeState.initial("CCCCCCCC"), "CCCCCCCC", [action])
            old_stock_only = os.environ.get("AUTOPLANNER_ROUTE_TREE_ACTION_OVERRIDE_STOCK_ONLY")
            try:
                os.environ["AUTOPLANNER_ROUTE_TREE_ACTION_OVERRIDE_STOCK_ONLY"] = "1"
                gated_runtime = RouteTreeRuntime(root / "base.pt", action_policy_path=root / "action_only.pt")
                gated_eval = gated_runtime.evaluate(
                    RouteTreeState.initial("CCCCCCCC"),
                    "CCCCCCCC",
                    [action],
                    stock_checker=lambda _smi: False,
                )
                stock_eval = gated_runtime.evaluate(
                    RouteTreeState.initial("CCCCCCCC"),
                    "CCCCCCCC",
                    [action],
                    stock_checker=lambda _smi: True,
                )
            finally:
                if old_stock_only is None:
                    os.environ.pop("AUTOPLANNER_ROUTE_TREE_ACTION_OVERRIDE_STOCK_ONLY", None)
                else:
                    os.environ["AUTOPLANNER_ROUTE_TREE_ACTION_OVERRIDE_STOCK_ONLY"] = old_stock_only

        self.assertEqual(report["metadata"]["policy_train_mode"], "action_only")
        self.assertEqual(report["metadata"]["action_rank_loss_weight"], 0.25)
        self.assertEqual(report["best_selection_metric"], "val_action_loss+0.25*val_action_rank_loss")
        self.assertIn("val_action_rank_loss", report["history"][-1])
        self.assertEqual(
            sorted(report["metadata"]["policy_train_mode_report"]["trainable_prefixes"]),
            ["action_proj", "action_score"],
        )
        for key, value in base.items():
            if key.startswith("action_proj.") or key.startswith("action_score."):
                continue
        self.assertTrue(torch.equal(value, action_only[key]), key)
        self.assertIsNotNone(runtime.action_override)
        self.assertEqual(runtime_eval.reason, "search_policy+action_override")
        self.assertEqual(gated_eval.reason, "search_policy+action_override_gated")
        self.assertEqual(stock_eval.reason, "search_policy+action_override")


if __name__ == "__main__":
    unittest.main()
