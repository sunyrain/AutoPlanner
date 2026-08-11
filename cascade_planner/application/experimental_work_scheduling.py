"""Deterministic, target-blind value/cost ranking for experiment work."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENTAL_WORK_SCHEDULING_SCHEMA = "experimental_work_scheduling.v1"
_DOMAINS = frozenset({"biocatalytic", "execution", "mechanism"})
_ACTION_SCORE_KEYS = frozenset(
    {
        "expected_portfolio_gain",
        "distance_to_closure",
        "evidence_gain",
        "route_diversity_gain",
        "dependency_unblock_count",
        "novelty_gain",
        "cost_penalty",
        "failure_risk_penalty",
    }
)
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
SCHEDULING_SEMANTICS = {
    "ranking_is_deterministic_and_target_blind": True,
    "estimated_cost_is_executor_neutral_not_a_provider_quote": True,
    "dirty_hints_affect_exact_boundary_priority_only": True,
    "experience_memory_changes_priority_only": True,
    "ranking_grants_no_validation_claim_proof_or_completion": True,
    "canonical_frontier_remains_the_single_work_authority": True,
}


def compile_experimental_work_scheduling(
    domain: str,
    plan: Mapping[str, Any],
    linked_canonical_deficit_ids: Sequence[Any],
    dirty_hint_ids: Sequence[Any],
    execution_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile an explainable rank vector without using target or dataset names."""

    domain_value = str(domain)
    if domain_value not in _DOMAINS:
        raise ValueError("experimental_work_scheduling_domain_invalid")
    plan_value = dict(plan)
    request = dict(execution_request)
    linked = _identifiers(linked_canonical_deficit_ids)
    dirty = _identifiers(dirty_hint_ids)
    priority = _bounded_number(plan_value.get("priority_score"), default=0.0)
    checks = {
        str(dict(row).get("check_id") or "")
        for row in request.get("required_checks") or []
        if isinstance(row, Mapping) and str(dict(row).get("check_id") or "")
    }
    boundary = dict(request.get("exact_boundary") or {})
    boundary_state_ids = {
        str(dict(row).get("state_id") or "")
        for side in ("input_states", "output_states")
        for row in boundary.get(side) or []
        if isinstance(row, Mapping) and str(dict(row).get("state_id") or "")
    }
    memory = dict(plan_value.get("experience_memory") or {})
    disposition = _experience_disposition(memory)
    transfer_scope = str(memory.get("strongest_transfer_scope") or "unobserved")
    transfer_factor = 0.75 if transfer_scope == "exact_boundary" else 1.0
    experience_information = round(
        _INFORMATION_BY_DISPOSITION[disposition] * transfer_factor, 6
    )
    information_components = {
        "canonical_deficit_signal": 0.28 if linked else 0.0,
        "dirty_exact_boundary_signal": 0.18 if dirty else 0.0,
        "domain_plan_priority_signal": round(0.22 * priority, 6),
        "required_check_breadth_signal": round(min(0.16, 0.025 * len(checks)), 6),
        "boundary_coverage_signal": round(
            min(0.08, 0.02 * len(boundary_state_ids)), 6
        ),
        "experience_uncertainty_signal": experience_information,
    }
    information_gain = round(min(1.0, sum(information_components.values())), 6)

    resource_hints = dict(request.get("resource_hints") or {})
    hinted_cost = _nonnegative_number(
        resource_hints.get("estimated_cost_units"), default=0.0
    )
    timeout_s = _positive_number(resource_hints.get("timeout_s"), default=3600.0)
    max_artifact_bytes = _positive_number(
        resource_hints.get("max_artifact_bytes"), default=100_000_000.0
    )
    cost_components = {
        "fixed_dispatch_units": 1.0,
        "resource_hint_units": round(hinted_cost, 6),
        "required_check_units": round(0.5 * len(checks), 6),
        "boundary_state_units": round(0.25 * len(boundary_state_ids), 6),
        "timeout_exposure_units": round(min(4.0, timeout_s / 3600.0) * 0.25, 6),
        "artifact_exposure_units": round(
            min(4.0, max_artifact_bytes / 100_000_000.0) * 0.25, 6
        ),
    }
    estimated_cost = round(sum(cost_components.values()), 6)
    value_per_cost = round(information_gain / estimated_cost, 9)
    normalized_cost_penalty = round(estimated_cost / (estimated_cost + 10.0), 6)
    failure_risk_penalty = _RISK_BY_DISPOSITION[disposition]
    action_score = {
        "expected_portfolio_gain": round(0.04 + 0.06 * priority, 6),
        "distance_to_closure": 0.08 if linked else 0.02,
        "evidence_gain": information_gain,
        "route_diversity_gain": 0.10,
        "dependency_unblock_count": len(linked),
        "novelty_gain": experience_information,
        "cost_penalty": normalized_cost_penalty,
        "failure_risk_penalty": failure_risk_penalty,
    }
    source_sha256 = str(plan_value.get("content_sha256") or "")
    if not _sha256(source_sha256):
        raise ValueError("experimental_work_scheduling_source_digest_invalid")
    reasons = [
        "canonical_deficit_linked" if linked else "route_scoped_shadow_work",
        f"experience_{disposition}",
        f"required_check_count:{len(checks)}",
        f"boundary_state_count:{len(boundary_state_ids)}",
    ]
    if dirty:
        reasons.append("exact_boundary_dirty_recompute")
    if hinted_cost > 0:
        reasons.append("executor_neutral_cost_hint_applied")
    rank_key = [
        -int(round(value_per_cost * 1_000_000_000)),
        -int(round(information_gain * 1_000_000)),
        int(round(estimated_cost * 1_000_000)),
        -len(linked),
        -len(dirty),
        domain_value,
        source_sha256,
    ]
    payload = {
        "schema_version": EXPERIMENTAL_WORK_SCHEDULING_SCHEMA,
        "information_gain_score": information_gain,
        "estimated_cost_units": estimated_cost,
        "value_per_cost": value_per_cost,
        "action_priority": round(100.0 * value_per_cost, 6),
        "rank_key": rank_key,
        "components": {
            "information_gain": information_components,
            "estimated_cost": cost_components,
            "experience_disposition": disposition,
            "experience_transfer_scope": transfer_scope,
        },
        "reasons": reasons,
        "action_score": action_score,
        "semantics": dict(SCHEDULING_SEMANTICS),
    }
    payload["content_sha256"] = strict_canonical_json_sha256(payload)
    return payload


def experimental_work_item_scheduling(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated scheduling metadata, or an empty fail-closed value."""

    if not isinstance(value, Mapping):
        return {}
    item = dict(value)
    if not isinstance(item.get("scheduling"), Mapping):
        return {}
    row = dict(item["scheduling"])
    try:
        material = dict(row)
        observed = str(material.pop("content_sha256", ""))
        digest_valid = observed == strict_canonical_json_sha256(material)
        rank_key = list(row.get("rank_key") or [])
        action_score = dict(row.get("action_score") or {})
        request = dict(item.get("execution_request") or {})
        expected = compile_experimental_work_scheduling(
            str(item.get("domain") or ""),
            dict(request.get("plan_payload") or {}),
            list(item.get("linked_canonical_deficit_ids") or []),
            list(item.get("dirty_hint_ids") or []),
            request,
        )
    except (TypeError, ValueError):
        return {}
    valid = (
        row.get("schema_version") == EXPERIMENTAL_WORK_SCHEDULING_SCHEMA
        and row.get("semantics") == SCHEDULING_SEMANTICS
        and digest_valid
        and len(rank_key) == 7
        and all(isinstance(value, int) and not isinstance(value, bool) for value in rank_key[:5])
        and all(isinstance(value, str) and value for value in rank_key[5:])
        and set(action_score) == _ACTION_SCORE_KEYS
        and row == expected
    )
    return row if valid else {}


def experimental_work_item_rank_key(entry: tuple[str, Any]) -> tuple[Any, ...]:
    """Sort valid work by compiled value/cost rank and invalid work last."""

    item_id, raw_item = entry
    scheduling = (
        experimental_work_item_scheduling(raw_item)
        if isinstance(raw_item, Mapping)
        else {}
    )
    if not scheduling:
        return (1, 0, 0, 0, 0, 0, "", "", str(item_id))
    rank_key = list(scheduling["rank_key"])
    return (0, *rank_key, str(item_id))


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


def _identifiers(values: Sequence[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _bounded_number(value: Any, *, default: float) -> float:
    number = _finite_number(value, default=default)
    return min(1.0, max(0.0, number))


def _nonnegative_number(value: Any, *, default: float) -> float:
    return max(0.0, _finite_number(value, default=default))


def _positive_number(value: Any, *, default: float) -> float:
    number = _finite_number(value, default=default)
    return number if number > 0 else default


def _finite_number(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("experimental_work_scheduling_number_invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("experimental_work_scheduling_number_invalid")
    return number


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "EXPERIMENTAL_WORK_SCHEDULING_SCHEMA",
    "compile_experimental_work_scheduling",
    "experimental_work_item_rank_key",
    "experimental_work_item_scheduling",
]
