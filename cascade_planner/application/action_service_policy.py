"""Target-blind minimum-service policy for campaign Action classes."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.campaign_actions import (
    ACTION_CLASS_ORDER,
    CampaignActionKind,
    campaign_action_kind_policy,
)


ACTION_CLASS_SERVICE_SCHEMA = "campaign_action_class_service.v1"
ACTION_CLASS_MINIMUM_SERVICE_INTERVAL = 12


def action_class_for_kind(kind: CampaignActionKind | str) -> str:
    """Return the frozen service class for one canonical Action kind."""

    return campaign_action_kind_policy(kind).action_class


def compile_action_class_service(
    *,
    prior_action_kinds: Iterable[str],
    candidates: Iterable[Mapping[str, Any]],
    scheduler_policy: str,
) -> dict[str, Any]:
    """Compile deadline-aware service debt without reserving work or budget."""

    history = tuple(str(value) for value in prior_action_kinds if str(value))
    history_classes = tuple(action_class_for_kind(value) for value in history)
    next_ordinal = len(history) + 1
    candidate_rows = [dict(value) for value in candidates]
    class_rows: list[dict[str, Any]] = []
    for action_class in ACTION_CLASS_ORDER:
        service_ordinals = [
            ordinal
            for ordinal, observed_class in enumerate(history_classes, start=1)
            if observed_class == action_class
        ]
        eligible_count = sum(
            1
            for row in candidate_rows
            if row.get("eligible") is True
            and str(row.get("action_class") or "") == action_class
        )
        blocked_count = sum(
            1
            for row in candidate_rows
            if row.get("eligible") is not True
            and str(row.get("action_class") or "") == action_class
        )
        last_service_ordinal = service_ordinals[-1] if service_ordinals else 0
        next_deadline = (
            last_service_ordinal + ACTION_CLASS_MINIMUM_SERVICE_INTERVAL
        )
        class_rows.append(
            {
                "action_class": action_class,
                "service_count": len(service_ordinals),
                "last_service_ordinal": last_service_ordinal,
                "next_service_deadline": next_deadline,
                "service_age": (
                    next_ordinal - last_service_ordinal
                    if last_service_ordinal
                    else next_ordinal
                ),
                "deadline_slack": next_deadline - next_ordinal,
                "eligible_candidate_count": eligible_count,
                "blocked_candidate_count": blocked_count,
                "eligible": eligible_count > 0,
            }
        )
    class_rank = {value: index for index, value in enumerate(ACTION_CLASS_ORDER)}
    eligible_by_deadline = sorted(
        (row for row in class_rows if row["eligible"]),
        key=lambda row: (
            int(row["next_service_deadline"]),
            class_rank[str(row["action_class"])],
        ),
    )
    pressure_rank = max(
        (
            rank
            for rank, row in enumerate(eligible_by_deadline)
            if int(row["next_service_deadline"]) <= next_ordinal + rank
        ),
        default=-1,
    )
    pressured_classes = [
        str(row["action_class"])
        for row in eligible_by_deadline[: pressure_rank + 1]
    ]
    minimum_service_enforced = str(scheduler_policy) == "adaptive"
    required_action_class = (
        str(eligible_by_deadline[0]["action_class"])
        if minimum_service_enforced and pressured_classes
        else ""
    )
    lending_classes = [
        str(row["action_class"])
        for row in class_rows
        if row["eligible"] is False
    ]
    result = {
        "schema_version": ACTION_CLASS_SERVICE_SCHEMA,
        "action_class_order": list(ACTION_CLASS_ORDER),
        "minimum_service_interval": ACTION_CLASS_MINIMUM_SERVICE_INTERVAL,
        "prior_action_count": len(history),
        "classified_prior_action_count": len(history_classes),
        "next_action_ordinal": next_ordinal,
        "prior_action_kinds_sha256": _digest(list(history)),
        "classes": class_rows,
        "pressured_action_classes": pressured_classes,
        "required_action_class": required_action_class,
        "minimum_service_enforced": minimum_service_enforced,
        "borrowable_from_action_classes": lending_classes,
        "semantics": {
            "rules_are_target_and_dataset_blind": True,
            "only_eligible_classes_accrue_dispatch_authority": True,
            "blocked_or_absent_class_capacity_is_borrowable": True,
            "borrowed_service_capacity_creates_no_run_kernel_budget": True,
            "round_robin_policy_is_not_overridden": not minimum_service_enforced,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def bind_action_class_selection(
    service: Mapping[str, Any],
    *,
    selected_action_class: str,
) -> dict[str, Any]:
    """Bind a pure service projection to the selected class and rehash it."""

    result = {
        key: value
        for key, value in dict(service).items()
        if key != "content_sha256"
    }
    selected_class = str(selected_action_class or "")
    required_class = str(result.get("required_action_class") or "")
    result.update(
        {
            "selected_action_class": selected_class,
            "minimum_service_guarantee_applied": bool(
                required_class and selected_class == required_class
            ),
            "borrowed_service_capacity": bool(
                selected_class and result.get("borrowable_from_action_classes")
            ),
        }
    )
    result["content_sha256"] = _digest(result)
    return result


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
    "ACTION_CLASS_MINIMUM_SERVICE_INTERVAL",
    "ACTION_CLASS_ORDER",
    "ACTION_CLASS_SERVICE_SCHEMA",
    "action_class_for_kind",
    "bind_action_class_selection",
    "compile_action_class_service",
]
