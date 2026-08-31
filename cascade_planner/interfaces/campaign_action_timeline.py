"""Read-only live timeline projected from Action receipts and RunKernel state."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


CAMPAIGN_ACTION_TIMELINE_SCHEMA = "campaign_action_timeline.v1"

_ACTORS = {
    "chemenzy_target_expand": "ChemEnzy",
    "native_short_tail_expand": "Native short-tail",
    "codex_global_architecture": "Codex",
    "codex_global_replan": "Codex",
    "acquire_exact_evidence": "Evidence",
    "bind_exact_evidence": "Evidence",
    "resolve_conflict": "Evidence",
    "reaction_validate": "Validation",
    "condition_enrich": "Conditions",
    "stock_audit": "Stock",
    "program_discover": "Program",
    "program_review": "Program",
    "program_admit": "Program",
    "program_validate": "Program",
    "experiment_feedback_ingest": "Experiment",
    "host_materialize": "Host",
    "recompute_route_closure": "Host",
}
_FAILURE_STATUSES = {
    "cancelled",
    "failed",
    "invalid",
    "rejected",
    "timed_out",
    "timeout",
    "unavailable",
}
_SUCCESS_STATUSES = {"accepted", "completed", "reused", "succeeded"}


def compile_campaign_action_timeline(
    stages: Iterable[Mapping[str, Any]],
    *,
    active_actions: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Merge settled checkpoint Actions with active kernel reservations."""

    records = []
    seen: set[str] = set()
    for stage_index, stage in enumerate(stages):
        if not str(stage.get("stage") or "").startswith("campaign_action_"):
            continue
        detail = dict(stage.get("detail") or {})
        action = dict(detail.get("action") or {})
        outcome = dict(detail.get("outcome") or {})
        execution_id = str(
            action.get("execution_id")
            or outcome.get("action_execution_id")
            or detail.get("execution_id")
            or ""
        )
        identity = execution_id or f"stage:{stage_index}"
        if identity in seen:
            continue
        seen.add(identity)
        status = str(
            outcome.get("status")
            or detail.get("status")
            or stage.get("status")
            or "unknown"
        )
        kind = str(action.get("kind") or "unknown")
        records.append(
            {
                "sequence": len(records) + 1,
                "execution_id": execution_id,
                "action_id": str(action.get("action_id") or ""),
                "kind": kind,
                "actor": _actor(kind),
                "state": _state(status),
                "status": status,
                "producer": str(action.get("producer") or ""),
                "resource_class": str(action.get("resource_class") or ""),
                "input_revision": int(action.get("input_revision") or 0),
                "output_revision": int(outcome.get("output_revision") or 0),
                "started_at": str(stage.get("started_at") or ""),
                "completed_at": str(stage.get("completed_at") or ""),
                "elapsed_s": float(
                    outcome.get("elapsed_s") or stage.get("elapsed_s") or 0.0
                ),
                "cache_hit": detail.get("cache_hit") is True,
                "material_events": _strings(outcome.get("material_events") or []),
                "failure_type": str(outcome.get("failure_type") or ""),
                "failure_reasons": _strings(outcome.get("failure_reasons") or []),
            }
        )
    for active in active_actions:
        execution_id = str(active.get("execution_id") or "")
        if not execution_id or execution_id in seen:
            continue
        seen.add(execution_id)
        kind = str(active.get("kind") or "unknown")
        records.append(
            {
                "sequence": len(records) + 1,
                "execution_id": execution_id,
                "action_id": str(active.get("action_id") or ""),
                "kind": kind,
                "actor": _actor(kind),
                "state": "running",
                "status": "running",
                "producer": str(active.get("producer") or ""),
                "resource_class": str(active.get("resource_class") or ""),
                "input_revision": int(active.get("input_revision") or 0),
                "output_revision": 0,
                "started_at": str(active.get("started_at") or ""),
                "completed_at": "",
                "elapsed_s": 0.0,
                "cache_hit": False,
                "material_events": [],
                "failure_type": "",
                "failure_reasons": [],
            }
        )
    result = {
        "schema_version": CAMPAIGN_ACTION_TIMELINE_SCHEMA,
        "record_count": len(records),
        "records": records,
        "actor_counts": _counts(records, "actor"),
        "state_counts": _counts(records, "state"),
        "semantics": {
            "settled_actions_come_from_target_checkpoint": True,
            "running_actions_come_from_run_kernel_reservations": True,
            "single_timeline_is_a_read_only_projection": True,
            "timeline_is_not_a_queue_or_scientific_authority": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _actor(kind: str) -> str:
    return _ACTORS.get(kind, "Other")


def _state(status: str) -> str:
    normalized = str(status).casefold()
    if normalized == "running":
        return "running"
    if normalized in _FAILURE_STATUSES:
        return "failed"
    if normalized == "partial":
        return "partial"
    if normalized in _SUCCESS_STATUSES:
        return "succeeded"
    return "settled"


def _counts(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _strings(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted({str(item) for item in values if str(item).strip()})


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


__all__ = ["CAMPAIGN_ACTION_TIMELINE_SCHEMA", "compile_campaign_action_timeline"]
