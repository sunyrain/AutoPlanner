from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from cascade_planner.eval.strategy_closure_live import (
    StrategyClosureLiveError,
    bind_live_execution,
    live_arm_command,
    summarize_strategy_closure,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _with_digest(value: dict) -> dict:
    row = deepcopy(value)
    row["content_sha256"] = _digest(row)
    return row


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path) -> dict:
    manifest = {"schema_version": "blind_retrosynthesis_manifest.v1", "cases": []}
    for index in range(2):
        manifest["cases"].append({"case_id": f"case-{index}", "target_name": f"opaque-{index}"})
    evaluator = _with_digest({"schema_version": "strategy_closure_evaluator_pack.v1"})
    manifest_path = _write(tmp_path / "manifest.json", manifest)
    evaluator_path = _write(tmp_path / "evaluator.json", evaluator)
    stock_path = tmp_path / "stock.sqlite3"
    stock_path.write_bytes(b"stock")
    leakage = _with_digest(
        {
            "schema_version": "blind_leakage_audit_pack.v1",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "cases": {},
        }
    )
    leakage_path = _write(tmp_path / "leakage.json", leakage)
    protocol = _with_digest(
        {
            "schema_version": "strategy_closure_pilot_protocol.v1",
            "scope": {"target_count": 2},
            "bindings": {
                "target_manifest_content_sha256": _digest(manifest),
                "evaluator_pack_content_sha256": evaluator["content_sha256"],
                "stock_oracle": {
                    "index_sha256": hashlib.sha256(stock_path.read_bytes()).hexdigest()
                },
            },
            "budget": {
                "planner_facing_per_target": {
                    "max_total_wall_time_s": 1800,
                    "max_attempt_runs": 192,
                }
            },
        }
    )
    protocol_path = _write(tmp_path / "protocol.json", protocol)
    return {
        "protocol_path": protocol_path,
        "manifest_path": manifest_path,
        "evaluator_pack_path": evaluator_path,
        "leakage_pack_path": leakage_path,
        "stock_index_path": stock_path,
        "output_root": tmp_path / "live",
    }


def test_bind_live_execution_rejects_manifest_drift(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    manifest["cases"].append({"case_id": "drift", "target_name": "drift"})
    _write(paths["manifest_path"], manifest)

    with pytest.raises(StrategyClosureLiveError, match="manifest_binding_mismatch"):
        bind_live_execution(**paths)


def test_live_execution_binds_complete_python_source_bundle(tmp_path: Path) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))

    source = execution["source_bundle"]
    assert source["file_count"] == len(source["files"])
    assert "cascade_planner/agent/codex_worker.py" in source["files"]
    assert "cascade_planner/orchestration/global_campaign_director.py" in source["files"]
    assert "scripts/run_v4_blind_panel.py" in source["files"]
    assert len(source["bundle_sha256"]) == 64


def test_live_arm_command_changes_only_declared_ablation(tmp_path: Path) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))
    codex = live_arm_command(
        execution,
        arm_id="codex_only",
        python_executable="python",
        runner_script=tmp_path / "run.py",
        chemenzy_env_prefix=tmp_path / "env",
        benchmark_stock_name="stock",
    )
    unified = live_arm_command(
        execution,
        arm_id="unified_adaptive",
        python_executable="python",
        runner_script=tmp_path / "run.py",
        chemenzy_env_prefix=tmp_path / "env",
        benchmark_stock_name="stock",
    )

    assert codex[codex.index("--ablation") + 1] == "codex-only"
    assert unified[unified.index("--ablation") + 1] == "unified-adaptive"
    assert codex[codex.index("--fixed-cutoff-total-tasks") + 1] == "192"
    assert codex[codex.index("--fixed-cutoff-wall-time-s") + 1] == "1800.0"


def test_bound_host_python_rejects_runtime_drift(tmp_path: Path) -> None:
    host = tmp_path / "host-python.exe"
    host.write_bytes(b"host")
    env = tmp_path / "env"
    env.mkdir()
    (env / "python.exe").write_bytes(b"chemenzy")
    execution = bind_live_execution(
        **_fixtures(tmp_path),
        host_python_executable=host,
        chemenzy_env_prefix=env,
    )
    drift = tmp_path / "drift-python.exe"
    drift.write_bytes(b"drift")

    with pytest.raises(StrategyClosureLiveError, match="host_python_binding_mismatch"):
        live_arm_command(
            execution,
            arm_id="codex_only",
            python_executable=str(drift),
            runner_script=tmp_path / "run.py",
            chemenzy_env_prefix=env,
            benchmark_stock_name="stock",
        )


def _panel(execution: dict, arm_id: str, *, levels: dict[str, int]) -> dict:
    arm = next(row for row in execution["arms"] if row["arm_id"] == arm_id)
    runner = execution["runner"]
    return {
        "ablation": arm["ablation"],
        "complete": True,
        "model": runner["model"],
        "reasoning_effort": runner["reasoning_effort"],
        "execution_profile": runner["execution_profile"],
        "worker_count": runner["workers"],
        "target_count": len(execution["target_case_ids"]),
        "selection": {"selected_case_ids": list(execution["target_case_ids"])},
        "fixed_cutoff_policy": {
            "wall_time_s": runner["fixed_cutoff_wall_time_s"],
            "settled_task_count": runner["fixed_cutoff_total_tasks"],
        },
        "frozen_snapshot": {
            "manifest_sha256": execution["manifest"]["file_sha256"],
            "benchmark_stock_index_sha256": execution["stock_index"]["file_sha256"],
            "leakage_audit_pack_sha256": execution["leakage_pack"]["file_sha256"],
        },
        "targets": {
            f"opaque-{index}": {
                "case_id": f"case-{index}",
                "status": "completed",
                "route_counts": {
                    "target_rooted_route_count": levels.get("C0", 0),
                    "canonical_materialized_route_count": levels.get("C1", 0),
                    "strict_host_validated_route_count": levels.get("C2", 0),
                    "exact_procedure_route_count": levels.get("C3", 0),
                    "condition_complete_route_count": levels.get("C4", 0),
                    "strict_stock_closed_route_count": levels.get("C5", 0),
                },
                "fixed_cutoff_projection": {
                    "milestones": {
                        "experiment:positive_exact_boundary_claim": bool(levels.get("C6", 0)),
                        "program:action:experiment_feedback_ingest": bool(
                            levels.get("C6_assessed", levels.get("C6", 0))
                        ),
                    }
                },
            }
            for index in range(2)
        },
    }


def _external_summary(execution: dict, cases: list[dict]) -> dict:
    return _with_digest(
        {
            "schema_version": "external-summary.v1",
            "status": "completed",
            "protocol_content_sha256": execution["protocol"]["content_sha256"],
            "case_count": len(cases),
            "route_count": sum(int(case.get("route_count") or 0) for case in cases),
            "closure_counts": {
                level: sum(
                    int(case.get("closure_counts", {}).get(level) or 0)
                    for case in cases
                )
                for level in ("C0", "C1", "C2", "C3", "C4", "C5")
            },
        }
    )


def test_summary_maps_strict_levels_and_target_paired_differences(
    tmp_path: Path,
) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))
    statuses = {
        "codex_only": _panel(execution, "codex_only", levels={"C0": 1}),
        "chemenzy_only": _panel(execution, "chemenzy_only", levels={"C0": 1, "C1": 1, "C2": 1}),
        "unified_adaptive": _panel(
            execution,
            "unified_adaptive",
            levels={level: 1 for level in ("C0", "C1", "C2", "C3", "C4", "C5", "C6")},
        ),
    }
    external = [
        _with_digest({
            "case_id": f"case-{index}",
            "closure_counts": {"C0": 1, "C1": index},
        })
        for index in range(2)
    ]
    external_summary = _external_summary(execution, external)

    summary = summarize_strategy_closure(
        execution=execution,
        live_panel_statuses=statuses,
        external_summary=external_summary,
        external_cases=external,
        bootstrap_samples=100,
    )

    assert summary["arms"]["external_snapshot_only"]["level_counts"]["C1"] == 1
    assert summary["arms"]["external_snapshot_only"]["level_rates"]["C6"] is None
    assert summary["arms"]["unified_adaptive"]["level_counts"]["C6"] == 2
    paired = summary["paired_differences"]["unified_adaptive_minus_codex_only"]["C4"]
    assert paired["paired_rate_difference"] == 1.0
    assert paired["right_only"] == 2
    external_c6 = summary["paired_differences"][
        "unified_adaptive_minus_external_snapshot_only"
    ]["C6"]
    assert external_c6["status"] == "not_comparable"
    assert external_c6["paired_target_count"] == 0


def test_summary_rejects_target_set_drift(tmp_path: Path) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))
    statuses = {
        arm: _panel(execution, arm, levels={"C0": 1})
        for arm in ("codex_only", "chemenzy_only", "unified_adaptive")
    }
    statuses["codex_only"]["targets"].pop("opaque-1")

    external = [
        _with_digest({"case_id": "case-0", "closure_counts": {}, "route_count": 0}),
        _with_digest({"case_id": "case-1", "closure_counts": {}, "route_count": 0}),
    ]
    with pytest.raises(StrategyClosureLiveError, match="case_set_mismatch"):
        summarize_strategy_closure(
            execution=execution,
            live_panel_statuses=statuses,
            external_summary=_external_summary(execution, external),
            external_cases=external,
        )


def test_summary_rejects_incomplete_live_panel(tmp_path: Path) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))
    statuses = {
        arm: _panel(execution, arm, levels={"C0": 1})
        for arm in ("codex_only", "chemenzy_only", "unified_adaptive")
    }
    statuses["codex_only"]["complete"] = False

    external = [
        _with_digest(
            {"case_id": f"case-{index}", "closure_counts": {}, "route_count": 0}
        )
        for index in range(2)
    ]
    with pytest.raises(StrategyClosureLiveError, match="live_panel_incomplete:codex_only"):
        summarize_strategy_closure(
            execution=execution,
            live_panel_statuses=statuses,
            external_summary=_external_summary(execution, external),
            external_cases=external,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda panel: panel.__setitem__("model", "drift"), "panel_model_mismatch"),
        (
            lambda panel: panel["selection"]["selected_case_ids"].reverse(),
            "panel_case_order_mismatch",
        ),
        (
            lambda panel: panel["fixed_cutoff_policy"].__setitem__(
                "settled_task_count", 1
            ),
            "panel_fixed_cutoff_settled_task_count_mismatch",
        ),
    ],
)
def test_summary_rejects_live_execution_binding_drift(
    tmp_path: Path, mutation, error: str
) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))
    statuses = {
        arm: _panel(execution, arm, levels={"C0": 1})
        for arm in ("codex_only", "chemenzy_only", "unified_adaptive")
    }
    mutation(statuses["codex_only"])

    external = [
        _with_digest(
            {"case_id": f"case-{index}", "closure_counts": {}, "route_count": 0}
        )
        for index in range(2)
    ]
    with pytest.raises(StrategyClosureLiveError, match=error):
        summarize_strategy_closure(
            execution=execution,
            live_panel_statuses=statuses,
            external_summary=_external_summary(execution, external),
            external_cases=external,
        )


def test_summary_rejects_external_aggregate_drift(tmp_path: Path) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))
    statuses = {
        arm: _panel(execution, arm, levels={"C0": 1})
        for arm in ("codex_only", "chemenzy_only", "unified_adaptive")
    }
    external = [
        _with_digest(
            {
                "case_id": f"case-{index}",
                "closure_counts": {"C0": 1},
                "route_count": 1,
            }
        )
        for index in range(2)
    ]
    summary = _external_summary(execution, external)
    summary["closure_counts"]["C0"] = 1
    summary = _with_digest({key: value for key, value in summary.items() if key != "content_sha256"})

    with pytest.raises(StrategyClosureLiveError, match="external_closure_counts_mismatch"):
        summarize_strategy_closure(
            execution=execution,
            live_panel_statuses=statuses,
            external_summary=summary,
            external_cases=external,
        )


def test_summary_classifies_contract_rejection_separately_from_unsolved(
    tmp_path: Path,
) -> None:
    execution = bind_live_execution(**_fixtures(tmp_path))
    statuses = {
        arm: _panel(execution, arm, levels={})
        for arm in ("codex_only", "chemenzy_only", "unified_adaptive")
    }
    for row in statuses["codex_only"]["targets"].values():
        row["failure_events"] = [
            {
                "stage": "global_campaign",
                "status": "failed",
                "reasons": ["GlobalCampaignPlanValidationError"],
            }
        ]
    external = [
        _with_digest(
            {"case_id": f"case-{index}", "closure_counts": {}, "route_count": 0}
        )
        for index in range(2)
    ]

    summary = summarize_strategy_closure(
        execution=execution,
        live_panel_statuses=statuses,
        external_summary=_external_summary(execution, external),
        external_cases=external,
        bootstrap_samples=10,
    )

    assert summary["arms"]["codex_only"]["failure_taxonomy"] == {
        "host-contract-rejection": 2
    }
