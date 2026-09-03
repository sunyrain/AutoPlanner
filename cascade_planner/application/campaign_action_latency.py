"""Operational latency audit for same-revision campaign action cohorts."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping


CAMPAIGN_ACTION_COHORT_LATENCY_SCHEMA = "campaign_action_cohort_latency_audit.v1"
_CHEMENZY_KIND = "chemenzy_target_expand"
_CODEX_KIND = "codex_global_architecture"


def compile_campaign_action_latency_audit(
    actions: Iterable[Mapping[str, Any]],
    executions: Iterable[Mapping[str, Any]],
    *,
    submission_offsets_s: Mapping[str, float],
    completion_offsets_s: Mapping[str, float],
    completion_order_execution_ids: Iterable[str],
    cohort_elapsed_s: float,
    cached_execution_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Bind first-provider timing to its own future, not peer cohort return."""

    action_rows = [dict(value) for value in actions]
    execution_by_id = {
        str(dict(row.get("action") or {}).get("execution_id") or ""): dict(row)
        for row in executions
        if isinstance(row, Mapping)
    }
    cached = {str(value) for value in cached_execution_ids if str(value)}
    submissions = _timing_map(submission_offsets_s)
    completions = _timing_map(completion_offsets_s)
    order = [str(value) for value in completion_order_execution_ids if str(value)]
    timing_rows = []
    for action in action_rows:
        execution_id = str(action.get("execution_id") or "")
        submitted = submissions.get(execution_id)
        completed = completions.get(execution_id)
        timing_rows.append(
            {
                "action_id": str(action.get("action_id") or ""),
                "action_execution_id": execution_id,
                "action_kind": str(action.get("kind") or ""),
                "cache_hit": execution_id in cached,
                "submitted_offset_s": submitted,
                "completed_offset_s": completed,
                "worker_elapsed_s": (
                    round(max(0.0, completed - submitted), 6)
                    if submitted is not None and completed is not None
                    else None
                ),
            }
        )
    by_kind = {row["action_kind"]: row for row in timing_rows}
    chemenzy = dict(by_kind.get(_CHEMENZY_KIND) or {})
    codex = dict(by_kind.get(_CODEX_KIND) or {})
    applicable = bool(chemenzy and codex)
    replay_cached = bool(
        applicable
        and chemenzy.get("action_execution_id") in cached
        and codex.get("action_execution_id") in cached
    )
    chemenzy_submitted = _number(chemenzy.get("submitted_offset_s"))
    chemenzy_completed = _number(chemenzy.get("completed_offset_s"))
    codex_submitted = _number(codex.get("submitted_offset_s"))
    codex_completed = _number(codex.get("completed_offset_s"))
    both_submitted_before_either_completed = bool(
        applicable
        and chemenzy_submitted is not None
        and chemenzy_completed is not None
        and codex_submitted is not None
        and codex_completed is not None
        and chemenzy_submitted <= codex_completed
        and codex_submitted <= chemenzy_completed
    )
    reasons = []
    if applicable and not replay_cached and None in {
        chemenzy_submitted,
        chemenzy_completed,
        codex_submitted,
        codex_completed,
    }:
        reasons.append("initial_peer_timing_incomplete")
    if applicable and not replay_cached and not both_submitted_before_either_completed:
        reasons.append("initial_provider_futures_did_not_overlap")
    raw_proposal = _chemenzy_first_proposal(
        chemenzy,
        execution_by_id=execution_by_id,
        codex_completed=codex_completed,
        cohort_elapsed_s=cohort_elapsed_s,
    )
    payload = {
        "schema_version": CAMPAIGN_ACTION_COHORT_LATENCY_SCHEMA,
        "applicable": applicable,
        "accepted": bool(not applicable or replay_cached or not reasons),
        "replay_cached": replay_cached,
        "cohort_elapsed_s": _number(cohort_elapsed_s),
        "completion_order_execution_ids": order,
        "completion_order_action_kinds": [
            str(
                dict(execution_by_id.get(execution_id, {}).get("action") or {}).get(
                    "kind"
                )
                or ""
            )
            for execution_id in order
        ],
        "both_initial_providers_submitted_before_either_completed": (
            both_submitted_before_either_completed
        ),
        "chemenzy_submitted_before_codex_completed": bool(
            applicable
            and chemenzy_submitted is not None
            and codex_completed is not None
            and chemenzy_submitted <= codex_completed
        ),
        "chemenzy_first_proposal": raw_proposal,
        "reasons": reasons,
        "actions": timing_rows,
        "semantics": {
            "timing_is_operational_not_scientific_authority": True,
            "offsets_use_one_monotonic_cohort_clock": True,
            "stable_observation_order_is_independent_of_completion_order": True,
            "raw_proposal_time_uses_chemenzy_completion_not_cohort_return": True,
            "codex_peer_wait_is_excluded_from_first_proposal_timing": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _chemenzy_first_proposal(
    timing: Mapping[str, Any],
    *,
    execution_by_id: Mapping[str, Mapping[str, Any]],
    codex_completed: float | None,
    cohort_elapsed_s: float,
) -> dict[str, Any]:
    execution_id = str(timing.get("action_execution_id") or "")
    execution = dict(execution_by_id.get(execution_id) or {})
    handler = dict(dict(execution.get("outcome") or {}).get("handler_result") or {})
    routes = [value for value in handler.get("routes") or [] if isinstance(value, Mapping)]
    proposal_count = max(
        0,
        int(handler.get("proposal_count") or handler.get("route_count") or len(routes)),
    )
    completed = _number(timing.get("completed_offset_s"))
    elapsed = _number(cohort_elapsed_s)
    return {
        "observed": completed is not None,
        "nonempty_raw_proposal_observed": proposal_count > 0,
        "proposal_count": proposal_count,
        "action_execution_id": execution_id,
        "elapsed_from_start_cohort_s": completed,
        "codex_peer_completed_offset_s": codex_completed,
        "codex_peer_in_flight_at_chemenzy_completion": bool(
            completed is not None
            and codex_completed is not None
            and completed < codex_completed
        ),
        "peer_wait_excluded_s": (
            round(max(0.0, elapsed - completed), 6)
            if elapsed is not None and completed is not None
            else None
        ),
        "timing_excludes_codex_peer_wait": True,
    }


def _timing_map(value: Mapping[str, float]) -> dict[str, float]:
    return {
        str(key): number
        for key, item in dict(value).items()
        if str(key) and (number := _number(item)) is not None
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, number), 6) if math.isfinite(number) else None


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CAMPAIGN_ACTION_COHORT_LATENCY_SCHEMA",
    "compile_campaign_action_latency_audit",
]
