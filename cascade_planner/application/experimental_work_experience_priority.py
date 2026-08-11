"""Target-blind experience contribution to experimental work priority."""
from __future__ import annotations

import math
from typing import Any, Mapping


_INFORMATION_BY_DISPOSITION = {
    "unobserved": 0.12,
    "supported": 0.07,
    "contraindicated": 0.12,
    "inconclusive": 0.16,
    "conflicting": 0.18,
}
_RISK_BY_DISPOSITION = {
    "unobserved": 0.12,
    "supported": 0.04,
    "contraindicated": 0.16,
    "inconclusive": 0.12,
    "conflicting": 0.14,
}
_UNCHANGED_EXACT_REPEAT_PENALTY = {
    "unobserved": 0.0,
    "supported": 0.05,
    "contraindicated": 0.08,
    "inconclusive": 0.06,
    "conflicting": 0.03,
}


def experimental_experience_priority(
    memory: Mapping[str, Any],
    *,
    exact_boundary_dirty: bool,
) -> dict[str, Any]:
    """Compile uncertainty, risk, and unchanged exact-repeat penalty."""

    row = dict(memory)
    disposition = _experience_disposition(row)
    transfer_scope = str(row.get("strongest_transfer_scope") or "unobserved")
    transfer_factor = 0.75 if transfer_scope == "exact_boundary" else 1.0
    information = (
        round(
            (0.06 + 0.12 * _bounded_number(row["uncertainty_score"], default=1.0))
            * transfer_factor,
            6,
        )
        if "uncertainty_score" in row
        else round(_INFORMATION_BY_DISPOSITION[disposition] * transfer_factor, 6)
    )
    risk = (
        round(0.04 + 0.16 * _bounded_number(row["risk_score"], default=0.5), 6)
        if "risk_score" in row
        else _RISK_BY_DISPOSITION[disposition]
    )
    repeat_penalty = (
        _UNCHANGED_EXACT_REPEAT_PENALTY[disposition]
        if transfer_scope == "exact_boundary" and not exact_boundary_dirty
        else 0.0
    )
    return {
        "disposition": disposition,
        "transfer_scope": transfer_scope,
        "information_gain": information,
        "failure_risk_penalty": risk,
        "unchanged_exact_boundary_repeat_penalty": repeat_penalty,
    }


def _experience_disposition(memory: Mapping[str, Any]) -> str:
    positive = _count(memory.get("positive_observation_count"))
    negative = _count(memory.get("negative_observation_count"))
    inconclusive = _count(memory.get("inconclusive_observation_count"))
    if positive and negative:
        return "conflicting"
    if positive:
        return "supported"
    if negative:
        return "contraindicated"
    if inconclusive:
        return "inconclusive"
    declared = str(memory.get("disposition") or "")
    return declared if declared in _INFORMATION_BY_DISPOSITION else "unobserved"


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _bounded_number(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("experimental_work_scheduling_number_invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("experimental_work_scheduling_number_invalid")
    return min(1.0, max(0.0, number))


__all__ = ["experimental_experience_priority"]
