"""Read-only launch-strategy projections from one frozen target panel."""
from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Iterable, Mapping


RESULT_FIRST_STRATEGY_PROJECTION_SCHEMA = (
    "result_first_launch_strategy_projection.v1"
)


def project_launch_strategies(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare scheduling only; preserve every frozen route outcome."""

    targets = [_project_target(value) for value in rows]
    strategies = {
        name: _summarize_strategy(targets, name)
        for name in ("frozen_batch", "concurrent_progressive", "native_first")
    }
    progressive = strategies["concurrent_progressive"]
    native_first = strategies["native_first"]
    failures = [row for row in targets if row["frozen_b4"] is not True]
    return {
        "schema_version": RESULT_FIRST_STRATEGY_PROJECTION_SCHEMA,
        "target_count": len(targets),
        "strategies": strategies,
        "comparison": {
            "native_first_minus_concurrent_progressive": {
                "b4_count_delta": (
                    native_first["b4_count"] - progressive["b4_count"]
                ),
                "mean_terminal_time_s_delta": _difference(
                    native_first, progressive, "terminal_time_s", "mean"
                ),
                "p95_terminal_time_s_delta": _difference(
                    native_first, progressive, "terminal_time_s", "p95"
                ),
                "mean_b4_time_s_delta": _difference(
                    native_first, progressive, "b4_time_s", "mean"
                ),
                "p95_b4_time_s_delta": _difference(
                    native_first, progressive, "b4_time_s", "p95"
                ),
                "initial_codex_dispatch_count_delta": (
                    native_first["initial_codex_dispatch_count"]
                    - progressive["initial_codex_dispatch_count"]
                ),
                "failed_target_added_wait_s": _distribution(
                    row["native_first"]["added_wait_s"] for row in failures
                ),
            },
            "concurrent_progressive_minus_frozen_batch": {
                "b4_count_delta": (
                    progressive["b4_count"]
                    - strategies["frozen_batch"]["b4_count"]
                ),
                "mean_terminal_time_s_delta": _difference(
                    progressive,
                    strategies["frozen_batch"],
                    "terminal_time_s",
                    "mean",
                ),
                "p95_terminal_time_s_delta": _difference(
                    progressive,
                    strategies["frozen_batch"],
                    "terminal_time_s",
                    "p95",
                ),
                "cancelled_initial_codex_plan_count": progressive[
                    "cancelled_initial_codex_plan_count"
                ],
            },
        },
        "decision": {
            "recommended_strategy": "concurrent_progressive",
            "reason": (
                "native_first_preserves_projected_recall_but_serializes_the_"
                "slow_path_and_all_frozen_failures"
            ),
            "native_first_is_not_default": True,
        },
        "targets": targets,
        "semantics": {
            "read_only_projection_from_frozen_observations": True,
            "route_outcomes_are_held_fixed": True,
            "not_a_fresh_provider_or_model_run": True,
            "not_a_causal_cross_arm_estimate": True,
            "model_dispatch_avoidance_is_not_billing_avoidance": True,
            "native_first_adds_observed_native_completion_before_codex": True,
            "progressive_delivery_reuses_observed_post_cohort_delivery_time": True,
        },
    }


def _project_target(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    latency = dict(row.get("start_cohort_latency_audit") or {})
    first_provider = dict(latency.get("chemenzy_first_proposal") or {})
    provider_done_s = _number(first_provider.get("elapsed_from_start_cohort_s"))
    cohort_s = _number(latency.get("cohort_elapsed_s"))
    codex_done_s = next(
        (
            _number(dict(action).get("completed_offset_s"))
            for action in latency.get("actions") or []
            if isinstance(action, Mapping)
            and dict(action).get("action_kind") == "codex_global_architecture"
        ),
        0.0,
    )
    native = next(
        (
            dict(attempt)
            for attempt in row.get("provider_search_attempts") or []
            if isinstance(attempt, Mapping)
            and dict(attempt).get("kind") == "native"
        ),
        {},
    )
    frozen_b4 = dict(row.get("gate_summary") or {}).get("B4") is True
    frozen_b4_s = _optional_number(
        dict(row.get("time_to_first_s") or {}).get("B4")
    )
    frozen_terminal_s = _number(row.get("elapsed_s"))
    model_calls = int(dict(row.get("model_cost") or {}).get("model_invocations") or 0)
    native_immediate_b4 = bool(
        frozen_b4
        and native.get("host_admitted_solved") is True
        and str(row.get("b4_phase") or "") == "chemenzy_seed"
    )
    delivery_s = (
        max(0.0, float(frozen_b4_s or 0.0) - cohort_s)
        if native_immediate_b4
        else 0.0
    )
    progressive_b4_s = (
        provider_done_s + delivery_s if native_immediate_b4 else frozen_b4_s
    )
    progressive_terminal_s = (
        float(progressive_b4_s or frozen_terminal_s)
        if native_immediate_b4
        else frozen_terminal_s
    )
    native_first_delay_s = 0.0 if native_immediate_b4 else provider_done_s
    native_first_b4_s = (
        progressive_b4_s
        if native_immediate_b4
        else (
            float(frozen_b4_s) + native_first_delay_s
            if frozen_b4_s is not None
            else None
        )
    )
    return {
        "case_id": str(row.get("case_id") or ""),
        "frozen_b4": frozen_b4,
        "native_host_admitted": native.get("host_admitted_solved") is True,
        "native_immediate_b4": native_immediate_b4,
        "observed": {
            "provider_completed_s": provider_done_s,
            "codex_completed_s": codex_done_s,
            "cohort_completed_s": cohort_s,
            "post_cohort_delivery_s": delivery_s,
            "model_invocations": model_calls,
        },
        "frozen_batch": {
            "b4_time_s": frozen_b4_s,
            "terminal_time_s": frozen_terminal_s,
            "initial_codex_dispatched": model_calls > 0,
            "initial_codex_plan_completed": model_calls > 0,
            "added_wait_s": 0.0,
        },
        "concurrent_progressive": {
            "b4_time_s": progressive_b4_s,
            "terminal_time_s": progressive_terminal_s,
            "initial_codex_dispatched": model_calls > 0,
            "initial_codex_plan_completed": model_calls > 0 and not native_immediate_b4,
            "added_wait_s": 0.0,
        },
        "native_first": {
            "b4_time_s": native_first_b4_s,
            "terminal_time_s": (
                progressive_terminal_s
                if native_immediate_b4
                else frozen_terminal_s + native_first_delay_s
            ),
            "initial_codex_dispatched": model_calls > 0 and not native_immediate_b4,
            "initial_codex_plan_completed": model_calls > 0 and not native_immediate_b4,
            "added_wait_s": native_first_delay_s,
        },
    }


def _summarize_strategy(
    rows: list[Mapping[str, Any]],
    strategy: str,
) -> dict[str, Any]:
    projections = [dict(row[strategy]) for row in rows]
    b4_times = [
        float(value["b4_time_s"])
        for value in projections
        if value.get("b4_time_s") is not None
    ]
    return {
        "b4_count": sum(row["frozen_b4"] is True for row in rows),
        "b4_rate": (
            sum(row["frozen_b4"] is True for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "b4_time_s": _distribution(b4_times),
        "terminal_time_s": _distribution(
            float(value["terminal_time_s"]) for value in projections
        ),
        "failed_target_terminal_time_s": _distribution(
            float(row[strategy]["terminal_time_s"])
            for row in rows
            if row["frozen_b4"] is not True
        ),
        "initial_codex_dispatch_count": sum(
            value.get("initial_codex_dispatched") is True
            for value in projections
        ),
        "completed_initial_codex_plan_count": sum(
            value.get("initial_codex_plan_completed") is True
            for value in projections
        ),
        "cancelled_initial_codex_plan_count": sum(
            value.get("initial_codex_dispatched") is True
            and value.get("initial_codex_plan_completed") is not True
            for value in projections
        ),
    }


def _difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    distribution: str,
    statistic: str,
) -> float:
    return round(
        float(dict(left[distribution])[statistic])
        - float(dict(right[distribution])[statistic]),
        6,
    )


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    rows = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not rows:
        return {"count": 0, "mean": None, "median": None, "p95": None, "sum": 0.0}
    return {
        "count": len(rows),
        "mean": round(mean(rows), 6),
        "median": round(median(rows), 6),
        "p95": round(_percentile(rows, 0.95), 6),
        "sum": round(sum(rows), 6),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _number(value: Any) -> float:
    return float(_optional_number(value) or 0.0)


__all__ = ["project_launch_strategies"]
