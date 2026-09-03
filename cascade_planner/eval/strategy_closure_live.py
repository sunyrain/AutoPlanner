"""Protocol-bound execution and scoring for a strategy-closure live pilot."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


LIVE_ARM_ABLATIONS = {
    "codex_only": "codex-only",
    "chemenzy_only": "chemenzy-only",
    "unified_adaptive": "unified-adaptive",
}
STRATEGY_CLOSURE_LIVE_EXECUTION_SCHEMA = "strategy_closure_live_execution.v1"
STRATEGY_CLOSURE_PAIRED_SUMMARY_SCHEMA = "strategy_closure_paired_summary.v1"
LEVELS = tuple(f"C{index}" for index in range(7))
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ENTRYPOINTS = (
    "scripts/run_strategy_closure_live_arms.py",
    "scripts/run_v4_blind_panel.py",
    "scripts/summarize_strategy_closure_pilot.py",
)


class StrategyClosureLiveError(ValueError):
    """A live pilot input, execution, or result violated the frozen protocol."""


def bind_live_execution(
    *,
    protocol_path: str | Path,
    manifest_path: str | Path,
    evaluator_pack_path: str | Path,
    leakage_pack_path: str | Path,
    stock_index_path: str | Path,
    output_root: str | Path,
    arms: Sequence[str] = tuple(LIVE_ARM_ABLATIONS),
    model: str = "gpt-5.5",
    reasoning_effort: str = "low",
    execution_profile: str = "standard",
    workers: int = 1,
    fixed_cutoff_total_tasks: int = 192,
    host_python_executable: str | Path | None = None,
    chemenzy_env_prefix: str | Path | None = None,
) -> dict[str, Any]:
    """Validate frozen inputs and compile an immutable live execution receipt."""

    paths = {
        "protocol": _file(protocol_path, "protocol"),
        "manifest": _file(manifest_path, "manifest"),
        "evaluator_pack": _file(evaluator_pack_path, "evaluator_pack"),
        "leakage_pack": _file(leakage_pack_path, "leakage_pack"),
        "stock_index": _file(stock_index_path, "stock_index"),
    }
    protocol = _json(paths["protocol"])
    manifest = _json(paths["manifest"])
    evaluator = _json(paths["evaluator_pack"])
    leakage = _json(paths["leakage_pack"])
    _validate_frozen_inputs(
        protocol=protocol,
        manifest=manifest,
        evaluator=evaluator,
        leakage=leakage,
        paths=paths,
    )
    selected_arms = _arms(arms)
    budget = dict(dict(protocol.get("budget") or {}).get("planner_facing_per_target") or {})
    cutoff_wall_time = float(budget.get("max_total_wall_time_s") or 0)
    if cutoff_wall_time <= 0:
        raise StrategyClosureLiveError("strategy_closure_wall_time_budget_invalid")
    if isinstance(fixed_cutoff_total_tasks, bool) or fixed_cutoff_total_tasks < 1:
        raise StrategyClosureLiveError("strategy_closure_task_cutoff_invalid")
    if workers not in {1, 2}:
        raise StrategyClosureLiveError("strategy_closure_worker_count_invalid")
    arm_specs = [
        {
            "arm_id": arm,
            "ablation": LIVE_ARM_ABLATIONS[arm],
            "output_root": str(Path(output_root).expanduser().resolve() / arm),
        }
        for arm in selected_arms
    ]
    executable_bindings = _executable_bindings(
        host_python_executable=host_python_executable,
        chemenzy_env_prefix=chemenzy_env_prefix,
    )
    body = {
        "schema_version": STRATEGY_CLOSURE_LIVE_EXECUTION_SCHEMA,
        "protocol": _binding(paths["protocol"], protocol),
        "manifest": _binding(paths["manifest"], manifest),
        "evaluator_pack": _binding(paths["evaluator_pack"], evaluator),
        "leakage_pack": _binding(paths["leakage_pack"], leakage),
        "stock_index": {
            "path": str(paths["stock_index"]),
            "file_sha256": _file_sha256(paths["stock_index"]),
        },
        "source_bundle": _source_bundle(),
        "target_case_ids": [str(row.get("case_id") or "") for row in manifest["cases"]],
        "arms": arm_specs,
        "runner": {
            "model": str(model),
            "reasoning_effort": str(reasoning_effort),
            "execution_profile": str(execution_profile),
            "workers": workers,
            "fixed_cutoff_wall_time_s": cutoff_wall_time,
            "fixed_cutoff_total_tasks": int(fixed_cutoff_total_tasks),
            "per_target_budget": budget,
            **executable_bindings,
        },
        "semantics": {
            "evaluator_pack_is_supervisor_only": True,
            "planner_receives_only_target_manifest": True,
            "all_arms_share_manifest_stock_cutoff_and_host_gates": True,
            "arm_changes_only_declared_provider_subsystem": True,
            "failures_and_partial_results_remain_in_denominator": True,
        },
    }
    return _with_digest(body)


def live_arm_command(
    execution: Mapping[str, Any],
    *,
    arm_id: str,
    python_executable: str,
    runner_script: str | Path,
    chemenzy_env_prefix: str | Path,
    benchmark_stock_name: str,
    preflight_only: bool = False,
    resume: bool = False,
) -> list[str]:
    """Build one deterministic panel-runner command from a bound receipt."""

    value = dict(execution)
    if not _digest_valid(value):
        raise StrategyClosureLiveError("strategy_closure_execution_digest_invalid")
    specs = {str(row["arm_id"]): dict(row) for row in value.get("arms") or []}
    if arm_id not in specs:
        raise StrategyClosureLiveError(f"strategy_closure_arm_not_bound:{arm_id}")
    runner = dict(value.get("runner") or {})
    bound_host = dict(runner.get("host_python") or {})
    supplied_host = Path(python_executable).expanduser().resolve()
    if bound_host and (
        str(supplied_host) != bound_host.get("path")
        or _file_sha256(supplied_host) != bound_host.get("file_sha256")
    ):
        raise StrategyClosureLiveError("strategy_closure_host_python_binding_mismatch")
    bound_chemenzy = dict(runner.get("chemenzy_python") or {})
    if bound_chemenzy:
        supplied_chemenzy = _python_in_prefix(chemenzy_env_prefix)
        if str(supplied_chemenzy) != bound_chemenzy.get("path") or _file_sha256(
            supplied_chemenzy
        ) != bound_chemenzy.get("file_sha256"):
            raise StrategyClosureLiveError("strategy_closure_chemenzy_python_binding_mismatch")
    command = [
        str(python_executable),
        str(Path(runner_script).expanduser().resolve()),
        "--manifest",
        str(dict(value["manifest"])["path"]),
        "--output-root",
        str(specs[arm_id]["output_root"]),
        "--model",
        str(runner["model"]),
        "--reasoning-effort",
        str(runner["reasoning_effort"]),
        "--execution-profile",
        str(runner["execution_profile"]),
        "--fixed-cutoff-wall-time-s",
        str(runner["fixed_cutoff_wall_time_s"]),
        "--fixed-cutoff-total-tasks",
        str(runner["fixed_cutoff_total_tasks"]),
        "--workers",
        str(runner["workers"]),
        "--ablation",
        str(specs[arm_id]["ablation"]),
        "--benchmark-stock-index",
        str(dict(value["stock_index"])["path"]),
        "--benchmark-stock-name",
        str(benchmark_stock_name),
        "--leakage-audit-pack",
        str(dict(value["leakage_pack"])["path"]),
        "--chemenzy-env-prefix",
        str(Path(chemenzy_env_prefix).expanduser().resolve()),
    ]
    if preflight_only:
        command.append("--preflight-only")
    if resume:
        command.append("--resume")
    return command


def summarize_strategy_closure(
    *,
    execution: Mapping[str, Any],
    live_panel_statuses: Mapping[str, Mapping[str, Any]],
    external_summary: Mapping[str, Any],
    external_cases: Iterable[Mapping[str, Any]],
    bootstrap_samples: int = 10_000,
    seed: int = 7,
) -> dict[str, Any]:
    """Compile target-paired C0-C6 results without upgrading missing facts."""

    receipt = dict(execution)
    if not _digest_valid(receipt):
        raise StrategyClosureLiveError("strategy_closure_execution_digest_invalid")
    if dict(receipt.get("source_bundle") or {}) != _source_bundle():
        raise StrategyClosureLiveError("strategy_closure_execution_source_bundle_drift")
    if not _digest_valid(external_summary):
        raise StrategyClosureLiveError("strategy_closure_external_summary_digest_invalid")
    case_ids = [str(value) for value in receipt.get("target_case_ids") or []]
    external_values = [dict(value) for value in external_cases]
    if any(not _digest_valid(value) for value in external_values):
        raise StrategyClosureLiveError("strategy_closure_external_case_digest_invalid")
    _validate_external_binding(
        receipt,
        summary=external_summary,
        cases=external_values,
    )
    external_rows = _external_target_rows(external_values)
    if set(external_rows) != set(case_ids):
        raise StrategyClosureLiveError("strategy_closure_external_case_set_mismatch")
    arm_rows: dict[str, dict[str, dict[str, Any]]] = {"external_snapshot_only": external_rows}
    panel_bindings: dict[str, Any] = {}
    for arm in receipt.get("arms") or []:
        arm_id = str(arm.get("arm_id") or "")
        status = dict(live_panel_statuses.get(arm_id) or {})
        if status.get("complete") is not True:
            raise StrategyClosureLiveError(f"strategy_closure_live_panel_incomplete:{arm_id}")
        rows = _live_target_rows(status)
        if set(rows) != set(case_ids):
            raise StrategyClosureLiveError(f"strategy_closure_live_case_set_mismatch:{arm_id}")
        _validate_panel_binding(receipt, arm_id=arm_id, status=status)
        arm_rows[arm_id] = rows
        panel_bindings[arm_id] = {
            "panel_complete": status.get("complete") is True,
            "panel_status_sha256": _digest(status),
            "snapshot": dict(status.get("frozen_snapshot") or {}),
        }
    arm_summaries = {
        arm_id: _summarize_arm(rows, case_ids=case_ids) for arm_id, rows in arm_rows.items()
    }
    pairs: dict[str, Any] = {}
    arm_ids = list(arm_rows)
    for left_index, left in enumerate(arm_ids):
        for right in arm_ids[left_index + 1 :]:
            key = f"{right}_minus_{left}"
            pairs[key] = {
                level: _paired_difference(
                    [arm_rows[right][case_id]["levels"][level] for case_id in case_ids],
                    [arm_rows[left][case_id]["levels"][level] for case_id in case_ids],
                    right_assessed=[
                        arm_rows[right][case_id]["assessed"][level] for case_id in case_ids
                    ],
                    left_assessed=[
                        arm_rows[left][case_id]["assessed"][level] for case_id in case_ids
                    ],
                    samples=bootstrap_samples,
                    seed=seed + left_index * 31 + int(level[1:]),
                )
                for level in LEVELS
            }
    unique = {
        arm_id: {
            level: sum(
                rows[case_id]["levels"][level]
                and all(
                    other_rows[case_id]["assessed"][level]
                    for other_rows in arm_rows.values()
                )
                and not any(
                    other_rows[case_id]["levels"][level]
                    for other_id, other_rows in arm_rows.items()
                    if other_id != arm_id
                )
                for case_id in case_ids
            )
            for level in LEVELS
        }
        for arm_id, rows in arm_rows.items()
    }
    body = {
        "schema_version": STRATEGY_CLOSURE_PAIRED_SUMMARY_SCHEMA,
        "execution_sha256": str(receipt.get("content_sha256") or ""),
        "protocol_content_sha256": str(
            dict(receipt.get("protocol") or {}).get("content_sha256") or ""
        ),
        "external_summary_sha256": str(external_summary.get("content_sha256") or ""),
        "target_count": len(case_ids),
        "case_ids": case_ids,
        "arms": arm_summaries,
        "paired_differences": pairs,
        "provider_unique_target_closures": unique,
        "panel_bindings": panel_bindings,
        "semantics": {
            "target_is_the_independent_paired_unit": True,
            "route_variants_are_not_independent_samples": True,
            "missing_and_failed_targets_remain_in_denominator": True,
            "C6_requires_positive_exact_boundary_claim": True,
            "external_snapshot_generation_cost_is_not_imputed": True,
            "external_C6_is_not_assessed_not_negative": True,
        },
    }
    return _with_digest(body)


def _validate_frozen_inputs(
    *,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    leakage: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    if not _digest_valid(protocol):
        raise StrategyClosureLiveError("strategy_closure_protocol_digest_invalid")
    bindings = dict(protocol.get("bindings") or {})
    if _digest(manifest) != bindings.get("target_manifest_content_sha256"):
        raise StrategyClosureLiveError("strategy_closure_manifest_binding_mismatch")
    if not _digest_valid(evaluator) or (
        evaluator.get("content_sha256") != bindings.get("evaluator_pack_content_sha256")
    ):
        raise StrategyClosureLiveError("strategy_closure_evaluator_binding_mismatch")
    if not _digest_valid(leakage):
        raise StrategyClosureLiveError("strategy_closure_leakage_digest_invalid")
    if leakage.get("manifest_sha256") != _file_sha256(paths["manifest"]):
        raise StrategyClosureLiveError("strategy_closure_leakage_manifest_mismatch")
    stock = dict(bindings.get("stock_oracle") or {})
    if _file_sha256(paths["stock_index"]) != stock.get("index_sha256"):
        raise StrategyClosureLiveError("strategy_closure_stock_binding_mismatch")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != int(
        dict(protocol.get("scope") or {}).get("target_count") or 0
    ):
        raise StrategyClosureLiveError("strategy_closure_target_count_mismatch")
    ids = [str(row.get("case_id") or "") for row in cases if isinstance(row, Mapping)]
    if len(ids) != len(cases) or not all(ids) or len(ids) != len(set(ids)):
        raise StrategyClosureLiveError("strategy_closure_case_identity_invalid")


def _validate_panel_binding(
    execution: Mapping[str, Any], *, arm_id: str, status: Mapping[str, Any]
) -> None:
    specs = {str(row["arm_id"]): dict(row) for row in execution.get("arms") or []}
    if str(status.get("ablation") or "") != specs[arm_id]["ablation"]:
        raise StrategyClosureLiveError(f"strategy_closure_panel_ablation_mismatch:{arm_id}")
    runner = dict(execution.get("runner") or {})
    expected_runner_fields = {
        "model": str(runner.get("model") or ""),
        "reasoning_effort": str(runner.get("reasoning_effort") or ""),
        "execution_profile": str(runner.get("execution_profile") or ""),
        "worker_count": int(runner.get("workers") or 0),
    }
    for field, expected in expected_runner_fields.items():
        observed = status.get(field)
        if observed != expected:
            raise StrategyClosureLiveError(
                f"strategy_closure_panel_{field}_mismatch:{arm_id}"
            )
    expected_case_ids = [str(value) for value in execution.get("target_case_ids") or []]
    selection = dict(status.get("selection") or {})
    selected_case_ids = [str(value) for value in selection.get("selected_case_ids") or []]
    if selected_case_ids != expected_case_ids:
        raise StrategyClosureLiveError(
            f"strategy_closure_panel_case_order_mismatch:{arm_id}"
        )
    if int(status.get("target_count") or 0) != len(expected_case_ids):
        raise StrategyClosureLiveError(
            f"strategy_closure_panel_target_count_mismatch:{arm_id}"
        )
    cutoff = dict(status.get("fixed_cutoff_policy") or {})
    expected_cutoff = {
        "wall_time_s": float(runner.get("fixed_cutoff_wall_time_s") or 0),
        "settled_task_count": int(runner.get("fixed_cutoff_total_tasks") or 0),
    }
    for field, expected in expected_cutoff.items():
        if cutoff.get(field) != expected:
            raise StrategyClosureLiveError(
                f"strategy_closure_panel_fixed_cutoff_{field}_mismatch:{arm_id}"
            )
    snapshot = dict(status.get("frozen_snapshot") or {})
    if snapshot.get("manifest_sha256") != dict(execution["manifest"])["file_sha256"]:
        raise StrategyClosureLiveError(f"strategy_closure_panel_manifest_mismatch:{arm_id}")
    if (
        snapshot.get("benchmark_stock_index_sha256")
        != dict(execution["stock_index"])["file_sha256"]
    ):
        raise StrategyClosureLiveError(f"strategy_closure_panel_stock_mismatch:{arm_id}")
    if snapshot.get("leakage_audit_pack_sha256") != dict(execution["leakage_pack"])["file_sha256"]:
        raise StrategyClosureLiveError(f"strategy_closure_panel_leakage_mismatch:{arm_id}")


def _validate_external_binding(
    execution: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> None:
    expected_protocol = str(
        dict(execution.get("protocol") or {}).get("content_sha256") or ""
    )
    if str(summary.get("protocol_content_sha256") or "") != expected_protocol:
        raise StrategyClosureLiveError("strategy_closure_external_protocol_mismatch")
    if summary.get("status") != "completed":
        raise StrategyClosureLiveError("strategy_closure_external_snapshot_incomplete")
    if int(summary.get("case_count") or 0) != len(cases):
        raise StrategyClosureLiveError("strategy_closure_external_case_count_mismatch")
    route_count = sum(int(case.get("route_count") or 0) for case in cases)
    if int(summary.get("route_count") or 0) != route_count:
        raise StrategyClosureLiveError("strategy_closure_external_route_count_mismatch")
    aggregate = {
        level: sum(
            int(dict(case.get("closure_counts") or {}).get(level) or 0)
            for case in cases
        )
        for level in LEVELS[:-1]
    }
    observed = {
        level: int(dict(summary.get("closure_counts") or {}).get(level) or 0)
        for level in LEVELS[:-1]
    }
    if observed != aggregate:
        raise StrategyClosureLiveError("strategy_closure_external_closure_counts_mismatch")


def _live_target_rows(status: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in dict(status.get("targets") or {}).values():
        if not isinstance(raw, Mapping):
            continue
        value = dict(raw)
        case_id = str(value.get("case_id") or "")
        counts = dict(value.get("route_counts") or {})
        projection = dict(value.get("fixed_cutoff_projection") or {})
        milestones = dict(projection.get("milestones") or {})
        levels = {
            "C0": int(counts.get("target_rooted_route_count") or 0) > 0,
            "C1": int(counts.get("canonical_materialized_route_count") or 0) > 0,
            "C2": int(counts.get("strict_host_validated_route_count") or 0) > 0,
            "C3": int(counts.get("exact_procedure_route_count") or 0) > 0,
            "C4": int(counts.get("condition_complete_route_count") or 0) > 0,
            "C5": int(counts.get("strict_stock_closed_route_count") or 0) > 0,
            "C6": milestones.get("experiment:positive_exact_boundary_claim") is True,
        }
        rows[case_id] = {
            "case_id": case_id,
            "status": str(value.get("status") or ""),
            "levels": levels,
            "assessed": {
                **{level: True for level in LEVELS[:-1]},
                "C6": milestones.get("program:action:experiment_feedback_ingest") is True,
            },
            "C6_assessed": milestones.get("program:action:experiment_feedback_ingest") is True,
            "route_counts": counts,
            "failure_events": [
                dict(event)
                for event in value.get("failure_events") or []
                if isinstance(event, Mapping)
            ],
            "resource_usage": {
                "model_cost": dict(value.get("model_cost") or {}),
                "elapsed_s": value.get("elapsed_s"),
                "attempt_count": int(value.get("attempt_count") or 0),
                "accepted_expansion_count": int(value.get("accepted_expansion_count") or 0),
            },
            "error": str(value.get("error") or "")[:1000],
        }
    return rows


def _external_target_rows(
    cases: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in cases:
        value = dict(raw)
        case_id = str(value.get("case_id") or "")
        counts = dict(value.get("closure_counts") or {})
        rows[case_id] = {
            "case_id": case_id,
            "status": "completed",
            "levels": {
                **{level: int(counts.get(level) or 0) > 0 for level in LEVELS[:-1]},
                "C6": False,
            },
            "assessed": {**{level: True for level in LEVELS[:-1]}, "C6": False},
            "C6_assessed": False,
            "route_counts": counts,
            "resource_usage": dict(value.get("resource_usage") or {}),
            "error": "",
        }
    return rows


def _summarize_arm(
    rows: Mapping[str, Mapping[str, Any]], *, case_ids: Sequence[str]
) -> dict[str, Any]:
    level_counts = {
        level: sum(bool(rows[case_id]["levels"][level]) for case_id in case_ids) for level in LEVELS
    }
    assessed_counts = {
        level: sum(bool(rows[case_id]["assessed"][level]) for case_id in case_ids)
        for level in LEVELS
    }
    failures = Counter(_failure_class(rows[case_id]) for case_id in case_ids)
    failures.pop("none", None)
    resources: Counter[str] = Counter()
    for case_id in case_ids:
        usage = dict(rows[case_id].get("resource_usage") or {})
        model = dict(usage.get("model_cost") or {})
        for key, value in model.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                resources[f"model.{key}"] += value
        for key in ("elapsed_s", "attempt_count", "accepted_expansion_count"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                resources[key] += value
    return {
        "target_count": len(case_ids),
        "completed_count": sum(rows[case_id].get("status") == "completed" for case_id in case_ids),
        "level_counts": level_counts,
        "level_assessed_counts": assessed_counts,
        "level_rates": {
            level: (
                round(level_counts[level] / assessed_counts[level], 6)
                if assessed_counts[level]
                else None
            )
            for level in LEVELS
        },
        "C6_assessed_count": sum(rows[case_id].get("C6_assessed") is True for case_id in case_ids),
        "failure_taxonomy": dict(sorted(failures.items())),
        "resource_totals": dict(sorted(resources.items())),
        "per_target": [dict(rows[case_id]) for case_id in case_ids],
    }


def _failure_class(row: Mapping[str, Any]) -> str:
    levels = dict(row.get("levels") or {})
    if not levels.get("C0"):
        failure_text = " ".join(
            str(reason)
            for event in row.get("failure_events") or []
            if isinstance(event, Mapping)
            for reason in event.get("reasons") or []
        ).casefold()
        if "chemenzy_bounded_probe_timeout" in failure_text:
            return "provider-timeout"
        if "globalcampaignplanvalidationerror" in failure_text or (
            "plan_context_sha256_mismatch" in failure_text
        ):
            return "host-contract-rejection"
        if "provider" in failure_text and (
            "empty" in failure_text or "no_candidate" in failure_text
        ):
            return "provider-no-candidate"
        error = str(row.get("error") or "").casefold()
        if "disconnect" in error:
            return "disconnected"
        if "invalid" in error or "reject" in error:
            return "invalid"
        return "unsolved"
    if not levels.get("C1"):
        return "invalid"
    if not levels.get("C2"):
        return "unvalidated"
    if not levels.get("C3"):
        return "source-missing"
    if not levels.get("C4"):
        return "condition-incomplete"
    if not levels.get("C5"):
        return "stock-open"
    if row.get("C6_assessed") is True and not levels.get("C6"):
        return "experiment-negative"
    return "none"


def _paired_difference(
    right: Sequence[bool],
    left: Sequence[bool],
    *,
    right_assessed: Sequence[bool] | None = None,
    left_assessed: Sequence[bool] | None = None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    right_mask = list(right_assessed) if right_assessed is not None else [True] * len(right)
    left_mask = list(left_assessed) if left_assessed is not None else [True] * len(left)
    paired = [
        (right_value, left_value)
        for right_value, left_value, right_ok, left_ok in zip(
            right, left, right_mask, left_mask, strict=True
        )
        if right_ok and left_ok
    ]
    if not paired:
        return {
            "status": "not_comparable",
            "paired_target_count": 0,
            "right_successes": None,
            "left_successes": None,
            "paired_rate_difference": None,
            "bootstrap_95_ci": None,
            "right_only": None,
            "left_only": None,
            "bootstrap_samples": 0,
        }
    differences = [
        int(right_value) - int(left_value)
        for right_value, left_value in paired
    ]
    observed = sum(differences) / len(differences)
    generator = random.Random(seed)
    boot = sorted(
        sum(differences[generator.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(max(1, samples))
    )
    lower = boot[int(0.025 * (len(boot) - 1))]
    upper = boot[int(0.975 * (len(boot) - 1))]
    return {
        "status": "comparable",
        "paired_target_count": len(paired),
        "right_successes": sum(value[0] for value in paired),
        "left_successes": sum(value[1] for value in paired),
        "paired_rate_difference": round(observed, 6),
        "bootstrap_95_ci": [round(lower, 6), round(upper, 6)],
        "right_only": sum(
            right_value and not left_value
            for right_value, left_value in paired
        ),
        "left_only": sum(
            left_value and not right_value
            for right_value, left_value in paired
        ),
        "bootstrap_samples": max(1, samples),
    }


def _arms(values: Sequence[str]) -> list[str]:
    selected = list(dict.fromkeys(str(value) for value in values))
    unknown = sorted(set(selected) - set(LIVE_ARM_ABLATIONS))
    if unknown or not selected:
        raise StrategyClosureLiveError("strategy_closure_live_arms_invalid:" + ",".join(unknown))
    return selected


def _binding(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "content_sha256": str(value.get("content_sha256") or _digest(value)),
    }


def _source_bundle() -> dict[str, Any]:
    relative_paths = {
        path.relative_to(_SOURCE_ROOT).as_posix()
        for path in (_SOURCE_ROOT / "cascade_planner").rglob("*.py")
        if path.is_file()
    }
    relative_paths.update(_SOURCE_ENTRYPOINTS)
    files: dict[str, dict[str, Any]] = {}
    for relative in sorted(relative_paths):
        path = _SOURCE_ROOT / relative
        if not path.is_file():
            raise StrategyClosureLiveError(
                f"strategy_closure_execution_source_missing:{relative}"
            )
        files[relative] = {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema_version": "strategy_closure_source_bundle.v1",
        "scope": ["cascade_planner/**/*.py", *_SOURCE_ENTRYPOINTS],
        "file_count": len(files),
        "files": files,
        "bundle_sha256": _digest(files),
    }


def _executable_bindings(
    *,
    host_python_executable: str | Path | None,
    chemenzy_env_prefix: str | Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if host_python_executable is not None:
        host = _file(host_python_executable, "host_python")
        result["host_python"] = {
            "path": str(host),
            "file_sha256": _file_sha256(host),
        }
    if chemenzy_env_prefix is not None:
        chemenzy = _python_in_prefix(chemenzy_env_prefix)
        result["chemenzy_python"] = {
            "path": str(chemenzy),
            "file_sha256": _file_sha256(chemenzy),
        }
    return result


def _python_in_prefix(value: str | Path) -> Path:
    prefix = Path(value).expanduser().resolve()
    candidates = (
        prefix / "python.exe",
        prefix / "Scripts" / "python.exe",
        prefix / "bin" / "python",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise StrategyClosureLiveError(f"strategy_closure_chemenzy_python_missing:{prefix}")
    return path


def _file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise StrategyClosureLiveError(f"strategy_closure_{label}_missing:{path}")
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StrategyClosureLiveError(f"strategy_closure_json_not_object:{path}")
    return dict(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _digest_valid(value: Mapping[str, Any]) -> bool:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    return bool(observed) and observed == _digest(material)


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    row.pop("content_sha256", None)
    row["content_sha256"] = _digest(row)
    return row


__all__ = [
    "LEVELS",
    "LIVE_ARM_ABLATIONS",
    "STRATEGY_CLOSURE_LIVE_EXECUTION_SCHEMA",
    "STRATEGY_CLOSURE_PAIRED_SUMMARY_SCHEMA",
    "StrategyClosureLiveError",
    "bind_live_execution",
    "live_arm_command",
    "summarize_strategy_closure",
]
