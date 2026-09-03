from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.eval.summarize_classic_multistep_benchmark import (
    summarize_classic_multistep_benchmark,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_summary_separates_retained_low_confidence_route_from_acceptance(
    tmp_path: Path,
) -> None:
    reference = _write(
        tmp_path / "reference.json",
        {
            "metadata": {"manifest_sha256": "abc"},
            "cases": [
                {
                    "case_id": "case-1",
                    "target_name": "opaque target 1",
                    "split": "n1",
                    "depth_stratum": "llr_5_6",
                    "target_smiles": "CCO",
                    "reference_metrics": {
                        "reaction_count": 5,
                        "longest_linear_depth": 5,
                    },
                }
            ],
        },
    )
    run = _write(
        tmp_path / "run.json",
        {
            "targets": [
                {
                    "target_smiles": "CCO",
                    "chem_enzy": {
                        "solved": True,
                        "route_count": 3,
                        "failures": [
                            {
                                "category": "provider_warning",
                                "target_smiles": "must-not-leak",
                            }
                        ],
                    },
                    "cascade_search": {
                        "solved": False,
                        "stock_closed": False,
                        "condition_conflict_free": True,
                        "result_programs": [{"route_steps": []}],
                        "failure_categories": ["ConditionMissing"],
                    },
                }
            ]
        },
    )
    proxy = _write(
        tmp_path / "proxy.json",
        {
            "targets": [
                {
                    "target_id": "case-1",
                    "topk": {
                        "10": {
                            "exact_reaction_sequence_hit": False,
                            "shorter_or_equal_hit": True,
                            "best_leaf_overlap": 0.5,
                        }
                    },
                }
            ]
        },
    )
    v4_panel = _write(
        tmp_path / "panel-status.json",
        {
            "target_count": 1,
            "targets": {
                "opaque target 1": {
                    "status": "completed",
                    "claim": "unresolved",
                    "accepted_under_configured_policy": False,
                    "elapsed_s": 12.5,
                    "gate_summary": {
                        "B0": True,
                        "B1": True,
                        "B2": False,
                        "B3": False,
                        "B4": True,
                        "B5": False,
                    },
                    "route_counts": {
                        "target_rooted_distinct_skeletons": 3,
                        "reaction_validated_skeletons": 0,
                        "stock_closed_skeletons": 2,
                    },
                    "chemenzy": {
                        "provider_invocation_count": 1,
                        "proposal_count": 4,
                    },
                }
            },
        },
    )

    report = summarize_classic_multistep_benchmark(
        reference_pack=reference,
        runs={"n1": run},
        proxies={"n1": proxy},
        output_json=tmp_path / "summary.json",
        output_md=tmp_path / "summary.md",
        output_html=tmp_path / "index.html",
        v4_panel_status=v4_panel,
    )

    row = report["targets"][0]
    assert row["runtime_completed"] is True
    assert row["cascade_route_retained"] is True
    assert row["cascade_accepted"] is False
    assert row["benchmark_stock_closed"] is False
    assert row["warning_counts"] == {
        "ConditionMissing": 1,
        "provider_warning": 1,
    }
    assert "target_smiles" not in row
    assert report["semantics"]["low_confidence_routes_are_retained_with_warnings"]
    v4 = report["aggregates"]["v4_panel"]
    assert v4["completion_rate"] == 1.0
    assert v4["gate_pass_rates_over_completed"]["B1"] == 1.0
    assert v4["gate_pass_rates_over_completed"]["B2"] == 0.0
    assert v4["total_stock_closed_skeletons"] == 2
    rendered = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "低可信路线保留" in rendered
    assert "完整 V4 target-only 盲测" in rendered
    assert "must-not-leak" not in rendered
