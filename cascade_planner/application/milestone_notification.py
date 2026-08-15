"""Notification acknowledgement projection for milestone outbox receipts."""
from __future__ import annotations

from typing import Any, Mapping


DELIVERY_STATUSES = frozenset({"delivered", "failed"})


def notification_attempt(
    *,
    channel: str,
    status: str,
    channel_receipt: Mapping[str, Any] | None,
    digest: Any,
) -> dict[str, Any]:
    resolved_channel = str(channel or "").strip()
    resolved_status = str(status or "").strip()
    if not resolved_channel:
        raise ValueError("milestone_notification_channel_required")
    if resolved_status not in DELIVERY_STATUSES:
        raise ValueError("milestone_notification_status_invalid")
    attempt = {
        "channel": resolved_channel,
        "status": resolved_status,
        "channel_receipt": dict(channel_receipt or {}),
    }
    attempt["attempt_sha256"] = digest(attempt)
    return attempt


def append_notification_attempt(
    notification: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    attempts = [
        dict(value)
        for value in notification.get("channel_receipts") or []
        if isinstance(value, Mapping)
    ]
    if any(
        value.get("attempt_sha256") == attempt.get("attempt_sha256")
        for value in attempts
    ):
        return dict(notification), False
    attempts.append(dict(attempt))
    return (
        {
            "status": (
                "delivered"
                if any(value.get("status") == "delivered" for value in attempts)
                else "pending"
            ),
            "attempt_count": len(attempts),
            "channel_receipts": attempts,
        },
        True,
    )


__all__ = ["DELIVERY_STATUSES", "append_notification_attempt", "notification_attempt"]
