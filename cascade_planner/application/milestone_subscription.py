"""External durable outbox for first-achieved campaign milestones."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.application.campaign_trajectory import (
    CAMPAIGN_SNAPSHOT_SCHEMA,
    compile_campaign_trajectory,
)
from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.application.milestone_notification import (
    append_notification_attempt,
    notification_attempt,
)
from cascade_planner.application.milestone_outbox_store import (
    load_latest as _load_latest_from_store,
    outbox_lock as _outbox_lock,
    write_latest as _write_latest,
    write_receipt as _write_receipt,
)


MILESTONE_SUBSCRIPTION_SCHEMA = "campaign_milestone_subscription_receipt.v1"
MILESTONE_SUBSCRIPTION_RESULT_SCHEMA = "campaign_milestone_subscription_result.v1"
MILESTONE_POLICIES = frozenset({"notify-only", "notify-and-cancel"})


def observe_milestone_subscription(
    kernel: RunKernel,
    *,
    snapshots: Iterable[Mapping[str, Any]],
    policy: str,
    milestone: str = "B4_stock_boundary",
) -> dict[str, Any]:
    """Persist the first valid milestone observation and optionally cancel."""

    resolved_policy = str(policy or "").strip()
    resolved_milestone = str(milestone or "").strip()
    if resolved_policy not in MILESTONE_POLICIES:
        raise ValueError("milestone_subscription_policy_invalid")
    if resolved_milestone != "B4_stock_boundary":
        raise ValueError("milestone_subscription_milestone_unsupported")
    rows = [
        dict(value)
        for value in snapshots
        if isinstance(value, Mapping)
        and value.get("schema_version") == CAMPAIGN_SNAPSHOT_SCHEMA
    ]
    trajectory = compile_campaign_trajectory(rows)
    observation = dict(
        dict(trajectory.get("first_achieved") or {}).get(resolved_milestone) or {}
    )
    if not observation:
        return _result(
            kernel,
            policy=resolved_policy,
            milestone=resolved_milestone,
            observed=False,
        )
    snapshot_sha256 = str(observation.get("snapshot_sha256") or "")
    snapshot = next(
        (
            row
            for row in rows
            if str(row.get("content_sha256") or "") == snapshot_sha256
        ),
        {},
    )
    input_binding = dict(dict(snapshot.get("bindings") or {}).get("input") or {})
    input_value = dict(input_binding.get("value") or {})
    if (
        not snapshot
        or dict(snapshot.get("milestones") or {}).get(resolved_milestone) is not True
        or input_value.get("campaign_spec_sha256")
        != kernel.spec.campaign_spec.to_dict()["content_sha256"]
    ):
        raise ValueError("milestone_subscription_snapshot_binding_invalid")
    identity = {
        "run_id": kernel.spec.run_id,
        "policy": resolved_policy,
        "milestone": resolved_milestone,
        "first_snapshot_sha256": snapshot_sha256,
    }
    subscription_id = f"milestone-subscription:{_digest(identity)[:24]}"
    outbox = kernel.run_dir / ".autoplanner" / "milestone-subscriptions"
    with _outbox_lock(outbox):
        observed_receipt = _receipt(
            identity,
            subscription_id=subscription_id,
            observation=observation,
            notification={
                "status": "pending",
                "attempt_count": 0,
                "channel_receipts": [],
            },
            cancellation={
                "requested": False,
                "event_id": "",
                "event_sha256": "",
            },
        )
        observed_path = _write_receipt(outbox, observed_receipt)
        final_receipt = observed_receipt
        if resolved_policy == "notify-and-cancel":
            event = kernel.cancel(
                idempotency_key=f"{subscription_id}:cancel",
                reasons=(
                    "external_milestone_subscription_cancelled_after_first_B4",
                ),
            )
            final_receipt = _receipt(
                identity,
                subscription_id=subscription_id,
                observation=observation,
                notification=dict(observed_receipt["notification"]),
                cancellation={
                    "requested": True,
                    "event_id": event.event_id,
                    "event_sha256": event.event_sha256,
                },
                supersedes_sha256=str(observed_receipt["content_sha256"]),
            )
            _write_receipt(outbox, final_receipt)
        _write_latest(outbox, final_receipt)
    return _result(
        kernel,
        policy=resolved_policy,
        milestone=resolved_milestone,
        observed=True,
        receipt=final_receipt,
        receipt_path=observed_path.parent / f"{final_receipt['content_sha256']}.json",
    )


def snapshots_from_target_checkpoint(
    run_dir: Path,
    *,
    expected_run_id: str = "",
) -> tuple[dict[str, Any], ...]:
    path = run_dir / ".autoplanner" / "target-solver-checkpoint.json"
    if not path.is_file():
        return ()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("milestone_subscription_checkpoint_invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("milestone_subscription_checkpoint_invalid")
    if expected_run_id and str(value.get("run_id") or "") != expected_run_id:
        raise ValueError("milestone_subscription_checkpoint_run_mismatch")
    return tuple(
        dict(detail)
        for stage in value.get("stages") or []
        if isinstance(stage, Mapping)
        and isinstance((detail := stage.get("detail")), Mapping)
        and detail.get("schema_version") == CAMPAIGN_SNAPSHOT_SCHEMA
    )


def acknowledge_milestone_notification(
    kernel: RunKernel,
    *,
    channel: str,
    status: str,
    channel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one idempotent channel attempt without repeating cancellation."""

    attempt = notification_attempt(
        channel=channel,
        status=status,
        channel_receipt=channel_receipt,
        digest=_digest,
    )
    outbox = kernel.run_dir / ".autoplanner" / "milestone-subscriptions"
    with _outbox_lock(outbox):
        current = _load_latest_from_store(
            outbox,
            expected_run_id=kernel.spec.run_id,
            receipt_schema=MILESTONE_SUBSCRIPTION_SCHEMA,
            digest=_digest,
        )
        notification = dict(current.get("notification") or {})
        updated_notification, changed = append_notification_attempt(
            notification,
            attempt,
        )
        if not changed:
            return _result(
                kernel,
                policy=str(current["policy"]),
                milestone=str(current["milestone"]),
                observed=True,
                receipt=current,
                receipt_path=outbox / f"{current['content_sha256']}.json",
            )
        identity = {
            key: current[key]
            for key in (
                "run_id",
                "policy",
                "milestone",
                "first_snapshot_sha256",
            )
        }
        updated = _receipt(
            identity,
            subscription_id=str(current["subscription_id"]),
            observation=dict(current["first_observation"]),
            notification=updated_notification,
            cancellation=dict(current.get("cancellation") or {}),
            supersedes_sha256=str(current["content_sha256"]),
        )
        path = _write_receipt(outbox, updated)
        _write_latest(outbox, updated)
    return _result(
        kernel,
        policy=str(updated["policy"]),
        milestone=str(updated["milestone"]),
        observed=True,
        receipt=updated,
        receipt_path=path,
    )


def _receipt(
    identity: Mapping[str, Any],
    *,
    subscription_id: str,
    observation: Mapping[str, Any],
    notification: Mapping[str, Any],
    cancellation: Mapping[str, Any],
    supersedes_sha256: str = "",
) -> dict[str, Any]:
    row = {
        "schema_version": MILESTONE_SUBSCRIPTION_SCHEMA,
        "subscription_id": subscription_id,
        **dict(identity),
        "first_observation": dict(observation),
        "notification": dict(notification),
        "cancellation": dict(cancellation),
        "supersedes_sha256": str(supersedes_sha256),
        "semantics": {
            "snapshot_is_trigger_authority": True,
            "outbox_grants_no_scientific_authority": True,
            "notification_precedes_optional_cancellation": True,
            "benchmark_runners_do_not_register_this_policy": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def _result(
    kernel: RunKernel,
    *,
    policy: str,
    milestone: str,
    observed: bool,
    receipt: Mapping[str, Any] | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MILESTONE_SUBSCRIPTION_RESULT_SCHEMA,
        "run_id": kernel.spec.run_id,
        "policy": policy,
        "milestone": milestone,
        "observed": observed,
        "receipt": dict(receipt or {}),
        "receipt_path": str(receipt_path or ""),
        "kernel_status": kernel.state.status,
    }


def _digest(value: Any) -> str:
    row = dict(value) if isinstance(value, Mapping) else value
    if isinstance(row, dict):
        row.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


__all__ = [
    "MILESTONE_POLICIES",
    "MILESTONE_SUBSCRIPTION_RESULT_SCHEMA",
    "MILESTONE_SUBSCRIPTION_SCHEMA",
    "acknowledge_milestone_notification",
    "observe_milestone_subscription",
    "snapshots_from_target_checkpoint",
]
