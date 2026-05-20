import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cascade_planner.cascadeboard.live_benchmark import (
    _retro_gt_types,
    _route_tree_type_metric_order,
    _type_match,
    _plan_one_target,
    _plan_one_target_with_policy_retry,
    _rank_results,
    summarize_target_results,
)


class LiveBenchmarkPolicyRetryTest(unittest.TestCase):
    def test_route_tree_type_metrics_use_retro_order(self):
        entry = {
            "gt_route": [
                {"transformation": "oxidation"},
                {"transformation": "amination"},
            ]
        }
        route = {
            "steps": [
                {"reaction_type": "amination"},
                {"reaction_type": "oxidation"},
            ],
            "metrics": {"filled_route": True},
        }

        self.assertEqual(_retro_gt_types(entry), ["amination", "oxidation"])
        self.assertEqual(_route_tree_type_metric_order("route_tree"), "retro")
        self.assertEqual(_route_tree_type_metric_order("rerank"), "forward")
        self.assertTrue(_type_match(route, _retro_gt_types(entry)))
        self.assertFalse(_type_match(route, [step["transformation"] for step in entry["gt_route"]]))

    def test_summary_aggregates_route_set_diversity(self):
        summary = summarize_target_results([{
            "route_domain": "chemoenzymatic",
            "metrics": {"plan": True},
            "route_recovery": {
                "recovery_bottleneck": "selector_missed_exact_candidate",
            },
            "planner_output": {
                "time_s": 1.0,
                "routes": [
                    {
                        "metrics": {
                            "candidate_pool": {
                                "candidate_pool_coverage": 1.0,
                                "total_candidates": 6,
                                "avg_candidates_per_step": 3.0,
                                "avg_pool_diversity_score": 0.75,
                                "avg_duplicate_reaction_fraction": 0.25,
                                "avg_duplicate_reactant_set_fraction": 0.5,
                                "candidate_pool_source_counts": {
                                    "retrochimera": 4,
                                    "enzyformer": 2,
                                },
                            }
                        }
                    }
                ],
                "route_set_metrics": {
                    "diversity": {
                        "unique_full_signatures": 2,
                        "duplicate_route_fraction": 0.25,
                        "mean_pairwise_type_distance": 0.5,
                        "mean_pairwise_terminal_jaccard_distance": 0.75,
                    }
                },
            },
        }])

        self.assertEqual(summary["avg_route_set_unique_full_signatures"], 2.0)
        self.assertEqual(summary["avg_route_set_duplicate_route_fraction"], 0.25)
        self.assertEqual(summary["avg_route_set_pairwise_type_distance"], 0.5)
        self.assertEqual(summary["avg_route_set_terminal_jaccard_distance"], 0.75)
        self.assertEqual(summary["candidate_pool_source_counts"]["retrochimera"], 4)
        self.assertEqual(summary["avg_candidate_pool_coverage"], 1.0)
        self.assertEqual(summary["avg_candidate_pool_total_candidates"], 6.0)
        self.assertEqual(summary["avg_candidate_pool_diversity_score"], 0.75)
        self.assertEqual(summary["avg_candidate_pool_duplicate_reaction_fraction"], 0.25)
        self.assertEqual(summary["avg_candidate_pool_duplicate_reactant_set_fraction"], 0.5)
        self.assertEqual(summary["recovery_bottleneck_counts"]["selector_missed_exact_candidate"], 1)

    def test_control_ranker_prefers_metrics_over_cross_planner_raw_score(self):
        stock_bad = SimpleNamespace(board="stock_bad", score=1000.0)
        balanced = SimpleNamespace(board="balanced", score=1.0)

        def fake_metrics(board, stock_checker=None):
            if board == "stock_bad":
                return {
                    "filled_route": True,
                    "strict_stock_solve": True,
                    "progressive_route": False,
                    "condition": {"condition_window_success": False},
                    "cascade_compatibility": {"cascade_compatibility_success": False, "issues": ["missing_pH"]},
                    "retrosynthesis_progress": {"main_chain_reduction": 0.2},
                    "route_naturalness": {"naturalness_score": 1.0},
                    "enzyme_evidence": {},
                }
            return {
                "filled_route": True,
                "strict_stock_solve": True,
                "progressive_route": False,
                "condition": {"condition_window_success": True},
                "cascade_compatibility": {"cascade_compatibility_success": True, "issues": []},
                "retrosynthesis_progress": {"main_chain_reduction": 0.2},
                "route_naturalness": {"naturalness_score": 1.0},
                "enzyme_evidence": {},
            }

        with patch("cascade_planner.cascadeboard.live_benchmark.route_metrics", side_effect=fake_metrics):
            ranked = _rank_results([stock_bad, balanced], search_mode="critic_control", stock_checker=lambda _smi: True)

        self.assertIs(ranked[0], balanced)

    def test_control_ranker_prefers_lower_operation_burden_on_tie(self):
        awkward = SimpleNamespace(board="awkward", score=1.0)
        smooth = SimpleNamespace(board="smooth", score=1.0)

        def fake_metrics(board, stock_checker=None):
            return {
                "filled_route": True,
                "strict_stock_solve": None,
                "progressive_route": True,
                "condition": {"condition_window_success": True},
                "cascade_compatibility": {"cascade_compatibility_success": True, "issues": []},
                "retrosynthesis_progress": {"main_chain_reduction": 0.5},
                "route_naturalness": {"naturalness_score": 1.0},
                "enzyme_evidence": {},
                "operation_transitions": {
                    "operation_score": 0.2 if board == "awkward" else 1.0,
                    "issues": ["chemo_bio_transition"] if board == "awkward" else [],
                },
            }

        with patch("cascade_planner.cascadeboard.live_benchmark.route_metrics", side_effect=fake_metrics):
            ranked = _rank_results([awkward, smooth], search_mode="critic_control")

        self.assertIs(ranked[0], smooth)

    def test_control_ranker_uses_candidate_pool_quality_as_tie_breaker(self):
        narrow = SimpleNamespace(board="narrow", score=1.0)
        diverse = SimpleNamespace(board="diverse", score=1.0)

        def fake_metrics(board, stock_checker=None):
            return {
                "filled_route": True,
                "strict_stock_solve": None,
                "progressive_route": True,
                "condition": {"condition_window_success": True},
                "cascade_compatibility": {"cascade_compatibility_success": True, "issues": []},
                "retrosynthesis_progress": {"main_chain_reduction": 0.5},
                "route_naturalness": {"naturalness_score": 1.0},
                "enzyme_evidence": {},
                "operation_transitions": {"operation_score": 1.0, "issues": []},
                "candidate_pool": {
                    "candidate_pool_coverage": 1.0,
                    "avg_pool_diversity_score": 0.9 if board == "diverse" else 0.2,
                    "avg_duplicate_reactant_set_fraction": 0.0 if board == "diverse" else 0.8,
                },
            }

        with patch("cascade_planner.cascadeboard.live_benchmark.route_metrics", side_effect=fake_metrics):
            ranked = _rank_results([narrow, diverse], search_mode="critic_control")

        self.assertIs(ranked[0], diverse)

    def test_hybrid_runs_stock_andor_before_cc_aostar_fallback(self):
        skel = SimpleNamespace(
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
            compat_pred="",
            opmode_pred="",
            issues_pred=[],
            log_prob=0.0,
        )

        with patch("cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons", return_value=[skel]):
            with patch("cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior", side_effect=lambda rows, *_args, **_kwargs: rows):
                with patch("cascade_planner.cascadeboard.stock_andor.plan_stock_closed_andor", return_value=["stock"]) as andor:
                    with patch("cascade_planner.cascadeboard.live_benchmark.plan_with_cc_aostar", return_value=["fallback"]) as cc:
                        with patch("cascade_planner.cascadeboard.live_benchmark._result_is_acceptable", return_value=False):
                            with patch("cascade_planner.cascadeboard.live_benchmark._rank_results", side_effect=lambda rows, **_: rows):
                                results = _plan_one_target(
                                    object(),
                                    {},
                                    target="CCO",
                                    depth=2,
                                    domain="chemoenzymatic",
                                    model_device="cpu",
                                    n_results=2,
                                    n_candidates_per_skeleton=1,
                                    skeleton_samples=1,
                                    search_mode="hybrid",
                                    search_budget=4,
                                    stock_checker=lambda _smi: False,
                                )

        self.assertEqual(results, ["stock", "fallback"])
        self.assertEqual(andor.call_count, 1)
        self.assertEqual(cc.call_count, 1)

    def test_hybrid_skips_fallback_when_stock_andor_is_acceptable(self):
        skel = SimpleNamespace(
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
            compat_pred="",
            opmode_pred="",
            issues_pred=[],
            log_prob=0.0,
        )

        with patch("cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons", return_value=[skel]):
            with patch("cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior", side_effect=lambda rows, *_args, **_kwargs: rows):
                with patch("cascade_planner.cascadeboard.stock_andor.plan_stock_closed_andor", return_value=["stock"]) as andor:
                    with patch("cascade_planner.cascadeboard.live_benchmark.plan_with_cc_aostar", return_value=["fallback"]) as cc:
                        with patch("cascade_planner.cascadeboard.live_benchmark._result_is_acceptable", return_value=True):
                            with patch("cascade_planner.cascadeboard.live_benchmark._rank_results", side_effect=lambda rows, **_: rows):
                                results = _plan_one_target(
                                    object(),
                                    {},
                                    target="CCO",
                                    depth=2,
                                    domain="chemoenzymatic",
                                    model_device="cpu",
                                    n_results=2,
                                    n_candidates_per_skeleton=1,
                                    skeleton_samples=1,
                                    search_mode="hybrid",
                                    search_budget=4,
                                    stock_checker=lambda _smi: True,
                                )

        self.assertEqual(results, ["stock"])
        self.assertEqual(andor.call_count, 1)
        self.assertEqual(cc.call_count, 0)

    def test_stock_rescue_runs_andor_only_after_cc_stock_failure(self):
        skel = SimpleNamespace(
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
            compat_pred="",
            opmode_pred="",
            issues_pred=[],
            log_prob=0.0,
        )

        def fake_metrics(board, stock_checker=None):
            return {"strict_stock_solve": board == "rescue"}

        with patch("cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons", return_value=[skel]):
            with patch("cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior", side_effect=lambda rows, *_args, **_kwargs: rows):
                with patch("cascade_planner.cascadeboard.live_benchmark.plan_with_cc_aostar", return_value=[SimpleNamespace(board="base", score=1.0)]) as cc:
                    with patch("cascade_planner.cascadeboard.stock_andor.plan_stock_closed_andor", return_value=[SimpleNamespace(board="rescue", score=2.0)]) as andor:
                        with patch("cascade_planner.cascadeboard.live_benchmark.route_metrics", side_effect=fake_metrics):
                            with patch("cascade_planner.cascadeboard.live_benchmark._rank_results", side_effect=lambda rows, **_: rows):
                                results = _plan_one_target(
                                    object(),
                                    {},
                                    target="CCO",
                                    depth=2,
                                    domain="chemoenzymatic",
                                    model_device="cpu",
                                    n_results=2,
                                    n_candidates_per_skeleton=1,
                                    skeleton_samples=1,
                                    search_mode="stock_rescue",
                                    search_budget=4,
                                    stock_checker=lambda _smi: True,
                                )

        self.assertEqual([row.board for row in results], ["base", "rescue"])
        self.assertEqual(cc.call_count, 1)
        self.assertEqual(andor.call_count, 1)

    def test_stock_rescue_skips_andor_when_cc_is_stock_closed(self):
        skel = SimpleNamespace(
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
            compat_pred="",
            opmode_pred="",
            issues_pred=[],
            log_prob=0.0,
        )

        with patch("cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons", return_value=[skel]):
            with patch("cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior", side_effect=lambda rows, *_args, **_kwargs: rows):
                with patch("cascade_planner.cascadeboard.live_benchmark.plan_with_cc_aostar", return_value=[SimpleNamespace(board="base", score=1.0)]) as cc:
                    with patch("cascade_planner.cascadeboard.stock_andor.plan_stock_closed_andor", return_value=[SimpleNamespace(board="rescue", score=2.0)]) as andor:
                        with patch("cascade_planner.cascadeboard.live_benchmark.route_metrics", return_value={"strict_stock_solve": True}):
                            with patch("cascade_planner.cascadeboard.live_benchmark._rank_results", side_effect=lambda rows, **_: rows):
                                results = _plan_one_target(
                                    object(),
                                    {},
                                    target="CCO",
                                    depth=2,
                                    domain="chemoenzymatic",
                                    model_device="cpu",
                                    n_results=2,
                                    n_candidates_per_skeleton=1,
                                    skeleton_samples=1,
                                    search_mode="stock_rescue",
                                    search_budget=4,
                                    stock_checker=lambda _smi: True,
                                )

        self.assertEqual([row.board for row in results], ["base"])
        self.assertEqual(cc.call_count, 1)
        self.assertEqual(andor.call_count, 0)

    def test_cc_aostar_receives_retrieval_augmented_skeletons(self):
        generated = SimpleNamespace(
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
            compat_pred="",
            opmode_pred="",
            issues_pred=[],
            log_prob=0.0,
        )
        retrieved = SimpleNamespace(
            types=["oxidation"],
            ec1s=[1],
            Ts=[30.0],
            pHs=[7.0],
            compat_pred="",
            opmode_pred="",
            issues_pred=[],
            log_prob=3.0,
        )

        with patch("cascade_planner.cascadeboard.live_benchmark.generate_multiple_skeletons", return_value=[generated]):
            with patch("cascade_planner.cascadeboard.live_benchmark.rank_skeletons_with_prior", side_effect=lambda rows, *_args, **_kwargs: rows):
                with patch(
                    "cascade_planner.cascadeboard.live_benchmark.augment_skeletons_with_retrieval_prior",
                    return_value=[retrieved, generated],
                ) as augment:
                    with patch("cascade_planner.cascadeboard.live_benchmark.plan_with_cc_aostar", return_value=[]) as cc:
                        _plan_one_target(
                            object(),
                            {},
                            target="CCO",
                            depth=2,
                            domain="chemoenzymatic",
                            model_device="cpu",
                            n_results=2,
                            n_candidates_per_skeleton=1,
                            skeleton_samples=2,
                            search_mode="cc_aostar",
                            search_budget=4,
                        )

        self.assertEqual(augment.call_count, 1)
        passed_skeletons = cc.call_args.kwargs["skeletons"]
        self.assertEqual([skel.types for skel in passed_skeletons], [["oxidation"], ["reduction"]])

    def test_policy_retry_runs_second_search_when_auto_safe(self):
        risk = {
            "retry_policy": {
                "automatic_retry_safe": True,
                "adjusted_settings": {
                    "max_steps": 5,
                    "n_results": 4,
                    "candidate_budget": 8,
                    "skeleton_samples": 4,
                    "expansion_budget": 240,
                    "retry_search_mode": "stock_rescue",
                },
            }
        }
        with patch("cascade_planner.cascadeboard.live_benchmark._plan_one_target", side_effect=[["base"], ["retry"]]) as plan:
            with patch("cascade_planner.cascadeboard.live_benchmark.route_results_payload", return_value={"routes": []}):
                with patch("cascade_planner.cascadeboard.live_benchmark.predict_failure_risk", return_value=risk):
                    with patch("cascade_planner.cascadeboard.live_benchmark._rank_results", side_effect=lambda rows, **_: rows):
                        results, meta = _plan_one_target_with_policy_retry(
                            object(),
                            {},
                            target="CCO",
                            depth=3,
                            domain="chemoenzymatic",
                            model_device="cpu",
                            n_results=2,
                            n_candidates_per_skeleton=2,
                            skeleton_samples=2,
                            stock_checker=lambda _smi: True,
                        )

        self.assertEqual(results, ["base", "retry"])
        self.assertEqual(plan.call_count, 2)
        self.assertTrue(meta["retry_executed"])
        self.assertEqual(meta["retry_depth"], 5)
        self.assertEqual(meta["retry_candidates"], 8)
        self.assertEqual(meta["retry_plan_candidates"], 2)
        self.assertEqual(meta["requested_retry_budget"], 240)
        self.assertEqual(meta["retry_budget"], 64)
        self.assertEqual(meta["auto_retry_budget_cap"], 64)
        self.assertEqual(meta["retry_search_mode"], "stock_rescue")
        self.assertEqual(plan.call_args_list[1].kwargs["search_mode"], "stock_rescue")
        self.assertEqual(plan.call_args_list[1].kwargs["search_budget"], 64)
        self.assertEqual(plan.call_args_list[1].kwargs["n_candidates_per_skeleton"], 2)

    def test_policy_retry_skips_second_search_when_not_auto_safe(self):
        risk = {"retry_policy": {"automatic_retry_safe": False, "adjusted_settings": {}}}
        with patch("cascade_planner.cascadeboard.live_benchmark._plan_one_target", return_value=["base"]) as plan:
            with patch("cascade_planner.cascadeboard.live_benchmark.route_results_payload", return_value={"routes": []}):
                with patch("cascade_planner.cascadeboard.live_benchmark.predict_failure_risk", return_value=risk):
                    with patch("cascade_planner.cascadeboard.live_benchmark._rank_results", side_effect=lambda rows, **_: rows):
                        results, meta = _plan_one_target_with_policy_retry(
                            object(),
                            {},
                            target="CCO",
                            depth=3,
                            domain="chemoenzymatic",
                            model_device="cpu",
                            n_results=2,
                            n_candidates_per_skeleton=2,
                            skeleton_samples=2,
                        )

        self.assertEqual(results, ["base"])
        self.assertEqual(plan.call_count, 1)
        self.assertFalse(meta["retry_executed"])


if __name__ == "__main__":
    unittest.main()
