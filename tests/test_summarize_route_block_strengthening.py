import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.summarize_route_block_strengthening import summarize_route_block_strengthening


class SummarizeRouteBlockStrengtheningTest(unittest.TestCase):
    def test_does_not_promote_when_retrieval_controls_are_not_cleared(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _write_inputs(root, learned_hardneg_mrr=0.423, retrieval_hardneg_mrr=0.422)
            value_report = root / "route_block_value.json"
            value_report.write_text(
                json.dumps(
                    {
                        "counts": {"rows": 12},
                        "evidence_provenance_audit": {
                            "status": "unverifiable_without_source_provenance",
                            "missing_retrieval_provenance_rows": 12,
                        },
                    }
                ),
                encoding="utf-8",
            )
            no_human_report = root / "no_human.json"
            no_human_report.write_text(json.dumps(_no_human_summary()), encoding="utf-8")

            report = summarize_route_block_strengthening(
                route_pool_report=paths["route_pool"],
                route_block_value_report=value_report,
                no_human_ablation_summary=no_human_report,
                ablation_summary=paths["ablation"],
                bootstrap_stability=paths["bootstrap"],
                transition_hardneg_summary=paths["hardneg"],
                guarded_search_comparison=paths["guarded"],
                output_json=root / "summary.json",
                output_md=root / "summary.md",
            )

        self.assertFalse(report["decision"]["promote_route_block_scorer"])
        self.assertIn("retrieval-only", report["decision"]["reason"])
        self.assertNotIn("route-pool learned scorer", report["decision"]["reason"])
        self.assertEqual(report["gates"]["fixed_pool"]["source"], "no_human_route_block_value")
        self.assertTrue(report["gates"]["fixed_pool"]["learned_beats_retrieval"]["ok"])
        self.assertTrue(report["gates"]["route_pool"]["learned_beats_native"]["ok"])
        self.assertFalse(report["gates"]["route_pool"]["learned_beats_retrieval"]["ok"])
        self.assertEqual(
            report["route_block_value"]["evidence_provenance_audit"]["status"],
            "unverifiable_without_source_provenance",
        )
        self.assertFalse(report["no_human_route_block_value"]["decision"]["expert_labels_required"])
        self.assertTrue(report["no_human_route_block_value"]["decision"]["fixed_pool_signal_present"])
        self.assertTrue(report["no_human_route_block_value"]["decision"]["fixed_pool_training_gate_passed"])
        self.assertEqual(
            report["no_human_route_block_value"]["control_model_minus_retrieval_mrr"],
            0.08,
        )
        self.assertEqual(
            report["no_human_route_block_value"]["best_no_audit_no_retrieval"]["model_mrr"],
            0.87,
        )

    def test_promotes_when_retrieval_and_live_gates_are_cleared(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _write_inputs(
                root,
                learned_hardneg_mrr=0.47,
                retrieval_hardneg_mrr=0.42,
                live_gt=0.37,
                route_retrieval_ci_low=0.01,
                route_retrieval_delta=0.05,
            )

            report = summarize_route_block_strengthening(
                route_pool_report=paths["route_pool"],
                ablation_summary=paths["ablation"],
                bootstrap_stability=paths["bootstrap"],
                transition_hardneg_summary=paths["hardneg"],
                guarded_search_comparison=paths["guarded"],
                output_json=root / "summary.json",
                output_md=root / "summary.md",
            )

        self.assertTrue(report["decision"]["promote_route_block_scorer"])

    def test_legacy_route_pool_gate_used_without_no_human_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _write_inputs(root, learned_hardneg_mrr=0.47, retrieval_hardneg_mrr=0.42)

            report = summarize_route_block_strengthening(
                route_pool_report=paths["route_pool"],
                ablation_summary=paths["ablation"],
                bootstrap_stability=paths["bootstrap"],
                transition_hardneg_summary=paths["hardneg"],
                guarded_search_comparison=paths["guarded"],
                output_json=root / "summary.json",
                output_md=root / "summary.md",
            )

        self.assertEqual(report["gates"]["fixed_pool"]["source"], "legacy_route_pool_scorer")
        self.assertFalse(report["gates"]["fixed_pool"]["learned_beats_retrieval"]["ok"])

    def test_live_gate_can_use_final_rerank_quality_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _write_inputs(
                root,
                learned_hardneg_mrr=0.47,
                retrieval_hardneg_mrr=0.42,
                live_gt=0.35,
                route_retrieval_ci_low=0.01,
                route_retrieval_delta=0.05,
            )
            guarded = json.loads(paths["guarded"].read_text(encoding="utf-8"))
            guarded["runs"].append(
                {
                    "name": "final_rerank",
                    "summary": {
                        "n_targets": 20,
                        "cascade_solved_rate": 0.9,
                        "stock_closed_rate": 1.0,
                        "top_result_exact_reaction_in_pool": 0.05,
                        "top_result_gt_reactant_in_pool": 0.37,
                        "result_exact_reaction_in_pool": 0.05,
                        "result_gt_reactant_in_pool": 0.35,
                    },
                    "route_block_value_final_rerank": {
                        "enabled_targets": 20,
                        "top_route_changed": 2,
                    },
                    "top_route_changed_vs_baseline": 2,
                }
            )
            paths["guarded"].write_text(json.dumps(guarded), encoding="utf-8")

            report = summarize_route_block_strengthening(
                route_pool_report=paths["route_pool"],
                ablation_summary=paths["ablation"],
                bootstrap_stability=paths["bootstrap"],
                transition_hardneg_summary=paths["hardneg"],
                guarded_search_comparison=paths["guarded"],
                output_json=root / "summary.json",
                output_md=root / "summary.md",
            )

        self.assertTrue(report["gates"]["guarded_live_search"]["quality_lift"]["ok"])
        self.assertEqual(report["guarded_live_search"]["best_quality"]["name"], "final_rerank")
        self.assertEqual(report["gates"]["guarded_live_search"]["search_actually_changed"]["final_rerank_changed"], 2)


def _write_inputs(
    root: Path,
    *,
    learned_hardneg_mrr: float,
    retrieval_hardneg_mrr: float,
    live_gt: float = 0.35,
    route_retrieval_delta: float = 0.02,
    route_retrieval_ci_low: float = -0.01,
):
    route_pool = root / "route_pool.json"
    route_pool.write_text(
        json.dumps(
            {
                "counts": {"test_positive_groups": 24},
                "baselines": {
                    "native_rank": {"test": _rank_report(0.17, 0.05)},
                    "ccts_model_mean": {"test": _rank_report(0.62, 0.16)},
                    "ccts_best_route_evidence": {"test": _rank_report(0.58, 0.15)},
                    "block_rerank_score": {"test": _rank_report(0.55, 0.13)},
                },
                "model": {"test": _rank_report(0.53, 0.11)},
                "selection": {
                    "selected_method": "native_rank_plus_learned",
                    "selected_test_mrr_covered": 0.56,
                },
            }
        ),
        encoding="utf-8",
    )
    ablation = root / "ablation.json"
    ablation.write_text(
        json.dumps(
            {
                "feature_sets": [
                    _feature("cascade_only", 0.65, 0.15),
                    _feature("no_cascade", 0.50, 0.11),
                    _feature("no_v4", 0.63, 0.15),
                    _feature("ccts_only", 0.56, 0.14),
                ]
            }
        ),
        encoding="utf-8",
    )
    bootstrap = root / "bootstrap.json"
    bootstrap.write_text(
        json.dumps(
            {
                "n_positive_groups": 24,
                "bootstrap_samples": 100,
                "observed_mrr": {
                    "model_cascade_only": 0.65,
                    "native_rank": 0.17,
                    "ccts_model_mean": 0.62,
                },
                "deltas": {
                    "model_cascade_only_minus_native_rank": _delta(0.48, 0.31),
                    "model_cascade_only_minus_ccts_model_mean": _delta(route_retrieval_delta, route_retrieval_ci_low),
                    "model_cascade_only_minus_model_no_cascade": _delta(0.15, 0.01),
                    "model_no_v4_minus_ccts_model_mean": _delta(0.01, -0.1),
                },
            }
        ),
        encoding="utf-8",
    )
    hardneg = root / "hardneg.json"
    selected_method = "learned"
    hardneg.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_method": selected_method,
                    "selected_test_block_mrr": learned_hardneg_mrr,
                    "selected_delta_vs_chem_block_mrr": learned_hardneg_mrr - 0.39,
                },
                "test_metric_rows": [
                    _metric("baselines", "chem_rank", "block_supported_positive_label", 0.39, coverage=0.46),
                    _metric("baselines", "chem_rank", "exact_label", 0.33, coverage=0.23),
                    _metric("nonlearned_blends", "retrieval", "block_supported_positive_label", retrieval_hardneg_mrr, coverage=0.46),
                    _metric("residual_blends", selected_method, "block_supported_positive_label", learned_hardneg_mrr, coverage=0.46),
                    _metric("residual_blends", selected_method, "exact_label", learned_hardneg_mrr - 0.05, coverage=0.23),
                ],
            }
        ),
        encoding="utf-8",
    )
    guarded = root / "guarded.json"
    guarded.write_text(
        json.dumps(
            {
                "runs": [
                    _run("baseline", gt=0.35, applied=0, changed=0),
                    _run("guarded", gt=live_gt, applied=4, changed=1),
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "route_pool": route_pool,
        "ablation": ablation,
        "bootstrap": bootstrap,
        "hardneg": hardneg,
        "guarded": guarded,
    }


def _rank_report(mrr, recall3):
    return {"mrr_covered": mrr, "recall_at_k_all": {"3": recall3}}


def _feature(name, mrr, recall3):
    return {
        "feature_set": name,
        "model_mrr_covered": mrr,
        "model_recall_at1_all": 0.1,
        "model_recall_at3_all": recall3,
        "model_recall_at5_all": recall3,
        "selected_method": "pairwise_logistic",
        "selected_test_mrr_covered": mrr,
    }


def _delta(observed, ci_low):
    return {"observed_delta": observed, "ci95_low": ci_low, "ci95_high": observed + 0.1, "p_delta_le_0": 0.01}


def _metric(family, method, label, mrr, *, coverage):
    return {
        "family": family,
        "method": method,
        "label": label,
        "mrr_covered": mrr,
        "coverage": coverage,
    }


def _run(name, *, gt, applied, changed):
    return {
        "name": name,
        "summary": {
            "n_targets": 20,
            "cascade_solved_rate": 0.9,
            "stock_closed_rate": 1.0,
            "top_result_exact_reaction_in_pool": 0.05,
            "top_result_gt_reactant_in_pool": gt,
            "result_exact_reaction_in_pool": 0.05,
            "result_gt_reactant_in_pool": gt,
        },
        "pair_diagnostics": {
            "cascade_pair_reward_applied_true": applied,
            "cascade_pair_applicable_true": 5,
        },
        "top_route_changed_vs_baseline": changed,
    }


def _no_human_summary():
    return {
        "decision": {
            "expert_labels_required": False,
            "fixed_pool_signal_present": True,
            "fixed_pool_training_gate_passed": True,
            "strict_fixed_pool_gate_passed": True,
            "promote_search_time": False,
        },
        "models": [
            {
                "model": "no_human_consensus_no_audit_no_retrieval",
                "model_mrr": 0.78,
                "native_mrr": 0.70,
                "retrieval_mrr": 0.76,
                "model_minus_retrieval_mrr": 0.02,
            },
            {
                "model": "no_human_route_no_audit_no_retrieval",
                "model_mrr": 0.87,
                "native_mrr": 0.85,
                "retrieval_mrr": 0.79,
                "model_minus_retrieval_mrr": 0.08,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
