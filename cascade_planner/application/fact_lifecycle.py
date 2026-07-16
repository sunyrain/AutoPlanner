"""Append-only lifecycle authority for canonical scientific facts.

Facts are never deleted when an authority revokes or expires them.  A lifecycle
event binds the exact immutable fact digest and the current state is replayed
from those events.  Restoring authority is a new event, not an in-place edit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


FACT_LIFECYCLE_EVENT_SCHEMA = "canonical_fact_lifecycle_event.v1"
FACT_LIFECYCLE_STATE_SCHEMA = "canonical_fact_lifecycle_state.v1"
FACT_LIFECYCLE_SUMMARY_SCHEMA = "canonical_fact_lifecycle_summary.v1"

_ACTIONS = {"revoke", "expire", "restore"}
_INACTIVE_ACTIONS = {"revoke": "revoked", "expire": "expired"}
_AUTHORITY_BY_KIND = {
    "source_binding": "source_fact_lifecycle_authority",
    "exact_record": "source_fact_lifecycle_authority",
    "procedure_record": "source_fact_lifecycle_authority",
    "reaction_proof": "reaction_validation_lifecycle_authority",
    "stock_observation": "stock_observation_lifecycle_authority",
}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def build_fact_lifecycle_event(
    *,
    subject_kind: str,
    subject_id: str,
    subject_content_sha256: str,
    action: str,
    effective_at: str,
    reason_codes: Iterable[str],
    supersedes_event_id: str = "",
    authority_scope: str = "",
) -> dict[str, Any]:
    """Build one canonical, digest-bound lifecycle event."""

    kind = str(subject_kind or "").strip()
    event_action = str(action or "").strip().lower()
    scope = str(authority_scope or _AUTHORITY_BY_KIND.get(kind, "")).strip()
    row = {
        "schema_version": FACT_LIFECYCLE_EVENT_SCHEMA,
        "subject_kind": kind,
        "subject_id": str(subject_id or "").strip(),
        "subject_content_sha256": str(subject_content_sha256 or "").strip().lower(),
        "action": event_action,
        "effective_at": _canonical_timestamp(effective_at),
        "reason_codes": sorted({str(value).strip() for value in reason_codes if str(value).strip()}),
        "supersedes_event_id": str(supersedes_event_id or "").strip(),
        "authority_scope": scope,
        "semantics": {
            "append_only_event": True,
            "subject_digest_bound": True,
            "grants_no_new_scientific_authority": True,
            "restore_requires_explicit_causal_predecessor": True,
        },
    }
    event_id = f"lifecycle:{_digest(row)}"
    event = {**row, "event_id": event_id}
    event["content_sha256"] = _digest(event)
    reasons = validate_fact_lifecycle_event(event)
    if reasons:
        raise ValueError("invalid fact lifecycle event:" + ",".join(reasons))
    return event


def validate_fact_lifecycle_event(value: Mapping[str, Any]) -> list[str]:
    row = dict(value)
    reasons: list[str] = []
    supplied = str(row.pop("content_sha256", ""))
    event_id = str(row.pop("event_id", ""))
    if value.get("schema_version") != FACT_LIFECYCLE_EVENT_SCHEMA:
        reasons.append("fact_lifecycle_schema_invalid")
    if not supplied or supplied != _digest({**row, "event_id": event_id}):
        reasons.append("fact_lifecycle_digest_invalid")
    expected_event_id = f"lifecycle:{_digest(row)}"
    if not event_id or event_id != expected_event_id:
        reasons.append("fact_lifecycle_event_identity_invalid")
    kind = str(value.get("subject_kind") or "")
    if kind not in _AUTHORITY_BY_KIND:
        reasons.append("fact_lifecycle_subject_kind_invalid")
    if not str(value.get("subject_id") or ""):
        reasons.append("fact_lifecycle_subject_id_missing")
    if not _HEX64.fullmatch(str(value.get("subject_content_sha256") or "")):
        reasons.append("fact_lifecycle_subject_digest_invalid")
    action = str(value.get("action") or "")
    if action not in _ACTIONS:
        reasons.append("fact_lifecycle_action_invalid")
    if str(value.get("authority_scope") or "") != _AUTHORITY_BY_KIND.get(kind, ""):
        reasons.append("fact_lifecycle_authority_scope_invalid")
    try:
        canonical_time = _canonical_timestamp(str(value.get("effective_at") or ""))
    except ValueError:
        canonical_time = ""
        reasons.append("fact_lifecycle_effective_at_invalid")
    if canonical_time and canonical_time != str(value.get("effective_at") or ""):
        reasons.append("fact_lifecycle_effective_at_not_canonical")
    reason_codes = value.get("reason_codes")
    if not isinstance(reason_codes, list) or not reason_codes:
        reasons.append("fact_lifecycle_reason_codes_missing")
    elif reason_codes != sorted({str(item).strip() for item in reason_codes if str(item).strip()}):
        reasons.append("fact_lifecycle_reason_codes_not_canonical")
    predecessor = str(value.get("supersedes_event_id") or "")
    if action == "restore" and not predecessor:
        reasons.append("fact_lifecycle_restore_predecessor_missing")
    return sorted(set(reasons))


def fact_subject(graph: Mapping[str, Any], subject_kind: str, subject_id: str) -> dict[str, Any]:
    section = {
        "source_binding": "source_bindings",
        "exact_record": "exact_records",
        "procedure_record": "procedure_records",
        "stock_observation": "stock_observations",
    }.get(str(subject_kind or ""))
    if section:
        value = dict(graph.get(section) or {}).get(str(subject_id or ""))
        return dict(value) if isinstance(value, Mapping) else {}
    if subject_kind == "reaction_proof":
        for edge in dict(graph.get("edges") or {}).values():
            if not isinstance(edge, Mapping):
                continue
            for proof in edge.get("reaction_proofs") or []:
                if isinstance(proof, Mapping) and str(proof.get("proof_digest") or "") == subject_id:
                    return dict(proof)
    return {}


def fact_subject_digest(subject_kind: str, subject: Mapping[str, Any]) -> str:
    if subject_kind == "reaction_proof":
        return str(subject.get("proof_digest") or "")
    return str(subject.get("content_sha256") or "")


def fact_lifecycle_state(
    events: Any,
    *,
    subject_kind: str,
    subject_id: str,
    subject_content_sha256: str,
) -> dict[str, Any]:
    matching = [
        dict(value)
        for value in _event_values(events)
        if isinstance(value, Mapping)
        and not validate_fact_lifecycle_event(value)
        and value.get("subject_kind") == subject_kind
        and value.get("subject_id") == subject_id
        and value.get("subject_content_sha256") == subject_content_sha256
    ]
    matching.sort(key=lambda row: (str(row.get("effective_at") or ""), str(row.get("event_id") or "")))
    latest = matching[-1] if matching else {}
    action = str(latest.get("action") or "")
    status = _INACTIVE_ACTIONS.get(action, "active")
    row = {
        "schema_version": FACT_LIFECYCLE_STATE_SCHEMA,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "subject_content_sha256": subject_content_sha256,
        "status": status,
        "active": status == "active",
        "event_count": len(matching),
        "latest_event_id": str(latest.get("event_id") or ""),
        "effective_at": str(latest.get("effective_at") or ""),
        "reason_codes": list(latest.get("reason_codes") or []),
        "authority_scope": str(latest.get("authority_scope") or ""),
    }
    row["content_sha256"] = _digest(row)
    return row


def graph_fact_lifecycle_state(
    graph: Mapping[str, Any], subject_kind: str, subject_id: str, subject: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    fact = dict(subject or fact_subject(graph, subject_kind, subject_id))
    return fact_lifecycle_state(
        graph.get("fact_lifecycle_events") or {},
        subject_kind=subject_kind,
        subject_id=subject_id,
        subject_content_sha256=fact_subject_digest(subject_kind, fact),
    )


def summarize_fact_lifecycle(graph: Mapping[str, Any]) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    subjects = {
        (str(event.get("subject_kind") or ""), str(event.get("subject_id") or ""))
        for event in _event_values(graph.get("fact_lifecycle_events") or {})
        if isinstance(event, Mapping)
    }
    for kind, subject_id in sorted(subjects):
        subject = fact_subject(graph, kind, subject_id)
        if not subject:
            continue
        states.append(graph_fact_lifecycle_state(graph, kind, subject_id, subject))
    inactive = [row for row in states if row.get("active") is not True]
    return {
        "schema_version": FACT_LIFECYCLE_SUMMARY_SCHEMA,
        "event_count": len(_event_values(graph.get("fact_lifecycle_events") or {})),
        "subject_count": len(states),
        "inactive_fact_count": len(inactive),
        "revoked_fact_count": sum(row.get("status") == "revoked" for row in inactive),
        "expired_fact_count": sum(row.get("status") == "expired" for row in inactive),
        "inactive_facts": inactive,
    }


def _event_values(events: Any) -> list[Any]:
    return list(events.values()) if isinstance(events, Mapping) else list(events or [])


def _canonical_timestamp(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("effective timestamp required")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("effective timestamp timezone required")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FACT_LIFECYCLE_EVENT_SCHEMA",
    "FACT_LIFECYCLE_STATE_SCHEMA",
    "build_fact_lifecycle_event",
    "fact_lifecycle_state",
    "fact_subject",
    "fact_subject_digest",
    "graph_fact_lifecycle_state",
    "summarize_fact_lifecycle",
    "validate_fact_lifecycle_event",
]
