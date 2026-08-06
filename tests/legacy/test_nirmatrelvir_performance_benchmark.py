from __future__ import annotations

import json
from pathlib import Path

from scripts.legacy import benchmark_nirmatrelvir_v3 as benchmark_module


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "config/benchmarks/nirmatrelvir_v3_performance_contract.json"
)


def test_performance_contract_is_model_free_and_scientifically_strict() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "retrosynthesis_performance_contract.v1"
    assert contract["engineering_limits"]["max_model_invocations"] == 0
    assert contract["scientific_minimums"]["complete_route_count"] >= 2
    assert contract["scientific_minimums"][
        "independent_support_group_count"
    ] >= 2


def test_benchmark_reports_cold_and_warm_without_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run_golden_case(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "artifact.json").write_text("{}\n", encoding="utf-8")
        metrics = {
            "schema_version": "autoplanner_run_metrics.v1",
            "run_id": "fake",
            "case_id": "nirmatrelvir-v3-real-source-dual-route",
            "producer": "test",
            "status": "completed",
            "failure_type": "",
            "started_at": "2026-07-13T00:00:00Z",
            "observed_at": "2026-07-13T00:00:01Z",
            "total_wall_ms": 1.0,
            "total_cpu_ms": 1.0,
            "stages": [],
            "stage_row_count": 0,
            "retained_stage_row_count": 0,
            "dropped_stage_row_count": 0,
            "max_retained_stage_rows": 2000,
            "stage_totals_by_category_ms": {},
            "stage_totals_by_name_ms": {},
            "stage_status_counts": {},
            "counters": {},
            "gauges": {"model_invocations": 0},
            "runtime": {},
            "semantics": {
                "metrics_grant_no_chemistry_authority": True,
            },
        }
        metrics["content_sha256"] = benchmark_module._digest(metrics)
        return {
            "accepted": True,
            "model_invocations": 0,
            "run_metrics": metrics,
            "portfolio": {
                "approved_source_step_count": 15,
                "hyperedge_count": 12,
                "complete_route_count": 2,
                "selected_route_count": 2,
                "stock_terminal_count": 7,
                "independent_support_groups": ["science", "patent"],
            },
        }

    monkeypatch.setattr(
        benchmark_module,
        "run_golden_case",
        fake_run_golden_case,
    )

    result = benchmark_module.benchmark_nirmatrelvir_v3(
        contract_path=CONTRACT,
        output_dir=tmp_path,
        iterations=2,
    )

    assert result["accepted"] is True
    assert [row["mode"] for row in result["iterations"]] == ["cold", "warm-1"]
    assert all(row["model_invocations"] == 0 for row in result["iterations"])
    assert all(row["run_metrics_reasons"] == [] for row in result["iterations"])
    saved = json.loads(
        (tmp_path / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert saved == result
