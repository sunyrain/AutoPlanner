from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cascade_planner.harness.agentic_blackboard_controller import (
    run_agentic_blackboard_controller,
)
from cascade_planner.runtime.run_metrics import (
    RunMetricsRecorder,
    record_run_metrics,
    validate_run_metrics,
)


def test_recorder_persists_digest_bound_stage_metrics(tmp_path: Path) -> None:
    recorder = RunMetricsRecorder(tmp_path, run_id="metrics-case")
    recorder.bind_case_id("case-123")
    recorder.increment("cache_hit", 2)
    recorder.gauge("reaction_hyperedge_count", 12)
    with recorder.stage(
        "graph.solve",
        category="graph",
        attributes={"dirty_nodes": 3},
    ):
        pass

    payload = recorder.finish(status="completed")
    saved = json.loads((tmp_path / "run_metrics.json").read_text(encoding="utf-8"))

    assert saved == payload
    assert validate_run_metrics(saved) == []
    assert saved["case_id"] == "case-123"
    assert saved["counters"]["cache_hit"] == 2.0
    assert saved["gauges"]["reaction_hyperedge_count"] == 12
    assert saved["stages"][0]["name"] == "graph.solve"
    assert saved["stages"][0]["attributes"]["dirty_nodes"] == 3
    assert saved["semantics"]["metrics_grant_no_chemistry_authority"] is True


def test_recorder_bounds_stage_rows_without_losing_aggregates(
    tmp_path: Path,
) -> None:
    recorder = RunMetricsRecorder(
        tmp_path,
        run_id="bounded-case",
        max_stage_rows=2,
    )
    recorder.observe("first", elapsed_s=0.001)
    recorder.observe("second", elapsed_s=0.002)
    recorder.observe("third", elapsed_s=0.003)

    payload = recorder.finish(status="completed")

    assert validate_run_metrics(payload) == []
    assert payload["stage_row_count"] == 3
    assert payload["retained_stage_row_count"] == 2
    assert payload["dropped_stage_row_count"] == 1
    assert [row["name"] for row in payload["stages"]] == ["second", "third"]
    assert payload["stage_totals_by_name_ms"] == {
        "first": 1.0,
        "second": 2.0,
        "third": 3.0,
    }
    assert payload["semantics"]["stage_aggregates_include_dropped_rows"] is True


def test_recorder_sanitizes_non_finite_observations(tmp_path: Path) -> None:
    recorder = RunMetricsRecorder(tmp_path, run_id="finite-case")
    recorder.increment("bad_counter", math.inf)
    recorder.gauge("bad_gauge", math.nan)
    recorder.observe("bad_elapsed", elapsed_s=math.nan)
    with recorder.stage("bad_attribute", attributes={"ratio": math.inf}):
        pass

    payload = recorder.finish(status="completed")

    assert validate_run_metrics(payload) == []
    assert payload["counters"]["bad_counter"] == 0.0
    assert payload["gauges"]["bad_gauge"] == "non_finite"
    assert payload["stage_totals_by_name_ms"]["bad_elapsed"] == 0.0
    attribute_row = next(
        row for row in payload["stages"] if row["name"] == "bad_attribute"
    )
    assert attribute_row["attributes"]["ratio"] is None


def test_recorder_reads_canonical_run_cost_ledger(tmp_path: Path) -> None:
    recorder = RunMetricsRecorder(tmp_path, run_id="ledger-case")
    recorder.observe_result(
        {
            "agent_blackboard": {
                "retrosynthesis_run_contract": {
                    "cost_ledger": {
                        "totals": {
                            "model_invocations": 2,
                            "input_tokens": 120,
                            "output_tokens": 30,
                            "wall_time_s": 4.5,
                            "accepted_expansions": 3,
                            "attempt_runs": 5,
                        }
                    }
                }
            }
        }
    )

    payload = recorder.finish(status="completed")

    assert payload["gauges"]["model_invocations"] == 2
    assert payload["gauges"]["model_wall_time_s"] == 4.5
    assert payload["gauges"]["accepted_expansions"] == 3
    assert payload["gauges"]["attempt_runs"] == 5


def test_record_run_metrics_persists_failure_without_swallowing_it(
    tmp_path: Path,
) -> None:
    @record_run_metrics
    def failing_run(*, output_dir: Path, target_name: str) -> dict:
        raise RuntimeError("expected test failure")

    with pytest.raises(RuntimeError, match="expected test failure"):
        failing_run(output_dir=tmp_path, target_name="failure-case")

    saved = json.loads((tmp_path / "run_metrics.json").read_text(encoding="utf-8"))
    assert validate_run_metrics(saved) == []
    assert saved["status"] == "failed"
    assert saved["failure_type"] == "RuntimeError"
    assert saved["stages"][0]["name"] == "run.total"
    assert saved["stages"][0]["status"] == "failed"


def test_controller_exposes_non_authoritative_metrics_artifact(
    tmp_path: Path,
) -> None:
    result = run_agentic_blackboard_controller(
        target_name="metrics-controller",
        target_smiles="not-a-smiles",
        output_dir=tmp_path,
        max_rounds=1,
        use_codex_action_planner=False,
        use_codex_agent_team=False,
    )

    metrics_path = Path(result["artifacts"]["run_metrics"])
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved == result["run_metrics"]
    assert validate_run_metrics(saved) == []
    assert saved["status"] == "completed"
    run_total = next(row for row in saved["stages"] if row["name"] == "run.total")
    assert run_total["status"] == "completed"
    assert run_total["wall_ms"] <= saved["total_wall_ms"]
    assert saved["gauges"]["tool_call_count"] == 0
    assert saved["semantics"]["metrics_are_observability_only"] is True
