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

    report = summarize_classic_multistep_benchmark(
        reference_pack=reference,
        runs={"n1": run},
        proxies={"n1": proxy},
        output_json=tmp_path / "summary.json",
        output_md=tmp_path / "summary.md",
        output_html=tmp_path / "index.html",
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
    rendered = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "低可信路线保留" in rendered
    assert "must-not-leak" not in rendered
