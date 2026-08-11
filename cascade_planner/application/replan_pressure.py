"""Target-blind pressure projection for one bounded global replan Action."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.action_convergence import (
    verified_action_convergence_ledger,
)
from cascade_planner.application.deficit_frontier import DeficitScore


REPLAN_PRESSURE_SCHEMA = "campaign_replan_pressure.v1"
DURABLE_STAGNATION_STREAK = 3


def compile_replan_pressure(
    gates: Mapping[str, Any],
    *,
    material_events: Iterable[str] = (),
    convergence_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project value from host state without labels or a second authority."""

    gate_values = {
        str(key): value is True
        for key, value in dict(gates.get("gates") or {}).items()
    }
    events = {
        str(value) for value in material_events if str(value).strip()
    }
    ledger = verified_action_convergence_ledger(convergence_ledger)
    streak = max(0, int(ledger.get("consecutive_no_gain") or 0))
    durable_stagnation = bool(
        ledger and streak >= DURABLE_STAGNATION_STREAK
    )
    route_diversity_deficit = not gate_values.get(
        "B1_global_multi_route",
        False,
    )
    components = {
        "route_diversity_deficit": float(route_diversity_deficit),
        "durable_stagnation": float(durable_stagnation),
        "stagnation_route_interaction": float(
            durable_stagnation and route_diversity_deficit
        ),
        "critical_edge_failure": float("critical_edge_rejected" in events),
        "shared_bottleneck_change": float(
            "shared_bottleneck_changed" in events
        ),
        "source_conflict": float("source_conflict_added" in events),
        "new_route_family": float("new_route_family" in events),
    }
    interaction = components["stagnation_route_interaction"]
    critical = components["critical_edge_failure"]
    bottleneck = components["shared_bottleneck_change"]
    conflict = components["source_conflict"]
    new_family = components["new_route_family"]
    diversity_deficit = components["route_diversity_deficit"]
    score = DeficitScore(
        expected_portfolio_gain=_unit(
            0.84
            + 0.10 * interaction
            + 0.08 * critical
            + 0.06 * bottleneck
            + 0.04 * conflict
            + 0.03 * new_family
        ),
        distance_to_closure=_unit(
            0.72
            + 0.08 * interaction
            + 0.10 * critical
            + 0.08 * bottleneck
            + 0.05 * conflict
        ),
        evidence_gain=_unit(0.30 + 0.22 * conflict + 0.05 * critical),
        route_diversity_gain=_unit(
            0.82
            + 0.06 * diversity_deficit
            + 0.12 * interaction
            + 0.08 * critical
            + 0.10 * bottleneck
            + 0.05 * new_family
        ),
        cost_penalty=0.55,
        failure_risk_penalty=0.35,
    ).to_dict()
    derived_events = (
        ["portfolio_stagnation"] if interaction else []
    )
    reasons = []
    if interaction:
        reasons.append("search_stagnation_with_route_diversity_deficit")
    if critical:
        reasons.append("critical_edge_failure_pressure")
    if bottleneck:
        reasons.append("shared_bottleneck_pressure")
    if conflict:
        reasons.append("source_conflict_pressure")
    if new_family:
        reasons.append("new_route_family_pressure")
    result = {
        "schema_version": REPLAN_PRESSURE_SCHEMA,
        "durable_stagnation_threshold": DURABLE_STAGNATION_STREAK,
        "consecutive_no_gain": streak,
        "convergence_ledger_verified": bool(ledger),
        "convergence_ledger_sha256": str(
            ledger.get("content_sha256") or ""
        ),
        "components": components,
        "pressure_total": round(
            min(
                1.0,
                0.28 * interaction
                + 0.22 * critical
                + 0.18 * bottleneck
                + 0.14 * conflict
                + 0.10 * new_family
                + 0.08 * diversity_deficit,
            ),
            6,
        ),
        "derived_material_events": derived_events,
        "trigger_reasons": reasons,
        "score": score,
        "semantics": {
            "target_labels_are_not_inputs": True,
            "text_stagnation_without_durable_streak_is_ignored": True,
            "durable_stagnation_requires_route_diversity_deficit": True,
            "pressure_changes_value_not_budget_or_authority": True,
            "single_deficit_frontier_and_action_loop_are_preserved": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _unit(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 6)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DURABLE_STAGNATION_STREAK",
    "REPLAN_PRESSURE_SCHEMA",
    "compile_replan_pressure",
]
