import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.compare_cascade_search_runs import compare_cascade_search_runs
from cascade_planner.eval.compare_cascade_search_runs import write_comparison_report


class CompareCascadeSearchRunsTest(unittest.TestCase):
    def test_compares_metrics_pair_counts_and_route_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = root / "baseline.json"
            guarded = root / "guarded.json"
            trace = root / "guarded_trace.jsonl"
            baseline.write_text(
                json.dumps(
                    {
                        "summary": {"cascade_solved_rate": 1.0, "stock_closed_rate": 1.0},
                        "targets": [
                            _target("A", ["r1"], stock=True, top_gt=True),
                            _target("B", ["r2"], stock=True, top_gt=False),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            guarded.write_text(
                json.dumps(
                    {
                        "summary": {"cascade_solved_rate": 1.0, "stock_closed_rate": 1.0},
                        "targets": [
                            _target("A", ["r1"], stock=True, top_gt=True),
                            _target("B", ["r3"], stock=True, top_gt=True),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            trace.write_text(
                "\n".join(
                    [
                        '{"cascade_pair_applicable": true, "cascade_pair_reward_applied": true, "cascade_pair_guard_reason": "applied"}',
                        '{"cascade_pair_applicable": true, "cascade_pair_reward_applied": false, "cascade_pair_guard_reason": "outside_base_tie_window"}',
                        '{"cascade_pair_applicable": false, "cascade_pair_reward_applied": false, "cascade_pair_guard_reason": "not_applicable"}',
                    ]
                ),
                encoding="utf-8",
            )

            report = compare_cascade_search_runs(
                baseline_path=baseline,
                run_paths={"guarded": guarded},
                trace_paths={"guarded": trace},
                metrics=["cascade_solved_rate", "stock_closed_rate", "top_result_gt_reactant_in_pool"],
            )
            write_comparison_report(report, output_json=root / "report.json", output_md=root / "report.md")

            guarded_run = next(row for row in report["runs"] if row["name"] == "guarded")

        self.assertEqual(guarded_run["top_route_changed_vs_baseline"], 1)
        self.assertEqual(guarded_run["selected_metrics"]["top_result_gt_reactant_in_pool"], 1.0)
        self.assertEqual(guarded_run["pair_diagnostics"]["cascade_pair_applicable_true"], 2)
        self.assertEqual(guarded_run["pair_diagnostics"]["cascade_pair_reward_applied_true"], 1)
        self.assertEqual(guarded_run["pair_diagnostics"]["guard_reason_outside_base_tie_window"], 1)

    def test_counts_route_block_value_final_rerank_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = root / "baseline.json"
            reranked = root / "reranked.json"
            baseline.write_text(
                json.dumps(
                    {
                        "summary": {"cascade_solved_rate": 1.0},
                        "targets": [_target("A", ["native"], stock=True, top_gt=False)],
                    }
                ),
                encoding="utf-8",
            )
            target = _target("A", ["reranked"], stock=True, top_gt=True)
            target["cascade_search"]["result_programs"][0]["original_rank"] = 2
            target["cascade_search"]["route_block_value_final_rerank"] = {
                "enabled": True,
                "changed_top_route": True,
                "scores": [{"original_rank": 2, "new_rank": 1}],
            }
            reranked.write_text(
                json.dumps({"summary": {"cascade_solved_rate": 1.0}, "targets": [target]}),
                encoding="utf-8",
            )

            report = compare_cascade_search_runs(
                baseline_path=baseline,
                run_paths={"final_rerank": reranked},
            )
            run = next(row for row in report["runs"] if row["name"] == "final_rerank")

        self.assertEqual(run["route_block_value_final_rerank"]["enabled_targets"], 1)
        self.assertEqual(run["route_block_value_final_rerank"]["top_route_changed"], 1)
        self.assertEqual(run["route_block_value_final_rerank"]["promoted_non_native_top"], 1)

    def test_counts_product_audit_final_rerank_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = root / "baseline.json"
            reranked = root / "reranked.json"
            baseline.write_text(
                json.dumps(
                    {
                        "summary": {"cascade_solved_rate": 1.0},
                        "targets": [_target("A", ["native"], stock=True, top_gt=False)],
                    }
                ),
                encoding="utf-8",
            )
            target = _target("A", ["reranked"], stock=True, top_gt=True)
            target["cascade_search"]["result_programs"][0]["product_audit_original_rank"] = 2
            target["cascade_search"]["product_audit_final_rerank"] = {
                "enabled": True,
                "changed_top_route": True,
                "scores": [{"original_rank": 2, "new_rank": 1}],
            }
            reranked.write_text(
                json.dumps({"summary": {"cascade_solved_rate": 1.0}, "targets": [target]}),
                encoding="utf-8",
            )

            report = compare_cascade_search_runs(
                baseline_path=baseline,
                run_paths={"product_audit": reranked},
            )
            run = next(row for row in report["runs"] if row["name"] == "product_audit")

        self.assertEqual(run["product_audit_final_rerank"]["enabled_targets"], 1)
        self.assertEqual(run["product_audit_final_rerank"]["top_route_changed"], 1)
        self.assertEqual(run["product_audit_final_rerank"]["promoted_non_native_top"], 1)


def _target(smiles, route_rxns, *, stock, top_gt=False):
    return {
        "target_smiles": smiles,
        "cascade_search": {
            "stock_closed": stock,
            "failure_categories": [],
            "result_programs": [
                {
                    "route_rxns": route_rxns,
                    "gt_reactant_hit_count": int(bool(top_gt)),
                }
            ],
        },
    }
