"""Operational Action status projections derived from RunKernel state."""
from __future__ import annotations

from typing import Any

from cascade_planner.application.run_kernel import RunState


def compile_active_campaign_actions(
    state: RunState,
) -> list[dict[str, Any]]:
    """Expose live Action wrappers from the canonical Kernel lifecycle."""

    rows = []
    for task_id, reservation in sorted(state.active_task_reservations.items()):
        metadata = dict(reservation.get("metadata") or {})
        action_id = str(metadata.get("campaign_action_id") or "")
        execution_id = str(metadata.get("campaign_action_execution_id") or "")
        if not action_id or not execution_id:
            continue
        kind = str(metadata.get("campaign_action_kind") or "")
        if not kind and action_id.startswith("action:"):
            kind = action_id.split(":", 2)[1]
        rows.append(
            {
                "schema_version": "active_campaign_action.v1",
                "action_id": action_id,
                "execution_id": execution_id,
                "kind": kind,
                "status": "running",
                "task_id": str(task_id),
                "input_revision": int(reservation.get("input_revision") or 0),
                "producer": str(metadata.get("producer") or ""),
                "resource_class": str(
                    reservation.get("resource_class")
                    or metadata.get("delegated_resource_class")
                    or ""
                ),
                "expected_resources_sha256": str(
                    metadata.get("expected_resources_sha256") or ""
                ),
                "semantics": {
                    "derived_from_run_kernel_in_flight_reservation": True,
                    "run_terminality_ends_active_visibility": True,
                    "not_a_second_queue": True,
                    "grants_no_scientific_authority": True,
                },
            }
        )
    return rows


__all__ = ["compile_active_campaign_actions"]
