"""Anytime snapshots for one target-blind campaign trajectory."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


CAMPAIGN_SNAPSHOT_SCHEMA = "campaign_anytime_snapshot.v1"
CAMPAIGN_TRAJECTORY_SCHEMA = "campaign_trajectory.v1"


def compile_campaign_snapshot(
    *,
    phase: str,
    observed_at: str,
    graph_revision: int,
    gates: Mapping[str, Any],
    resource_usage: Mapping[str, Any],
    action_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate_values = {
        str(key): value is True
        for key, value in dict(gates.get("gates") or {}).items()
    }
    row = {
        "schema_version": CAMPAIGN_SNAPSHOT_SCHEMA,
        "phase": str(phase),
        "observed_at": str(observed_at),
        "graph_revision": int(graph_revision),
        "milestones": gate_values,
        "highest_contiguous_gate": str(
            gates.get("highest_contiguous_gate") or "none"
        ),
        "counts": {
            str(key): int(value or 0)
            for key, value in dict(gates.get("counts") or {}).items()
        },
        "resource_usage": _json_value(resource_usage),
        "next_action": {
            key: value
            for key, value in dict(action_decision or {}).items()
            if key
            in {
                "selected_action_id",
                "selected_action",
                "candidate_count",
                "eligible_candidate_count",
                "content_sha256",
            }
        },
        "semantics": {
            "one_trajectory_for_all_result_views": True,
            "milestones_do_not_select_solver_control_flow": True,
            "snapshot_grants_no_additional_scientific_authority": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def compile_campaign_trajectory(
    snapshots: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    by_identity: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw in snapshots:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if row.get("schema_version") != CAMPAIGN_SNAPSHOT_SCHEMA:
            continue
        identity = (
            int(row.get("graph_revision") or 0),
            str(row.get("phase") or ""),
            str(row.get("content_sha256") or ""),
        )
        by_identity[identity] = row
    rows = [
        by_identity[key]
        for key in sorted(
            by_identity,
            key=lambda value: (value[0], value[1], value[2]),
        )
    ]
    first_achieved: dict[str, dict[str, Any]] = {}
    for row in rows:
        for gate, achieved in dict(row.get("milestones") or {}).items():
            if achieved is True and gate not in first_achieved:
                first_achieved[str(gate)] = {
                    "phase": str(row.get("phase") or ""),
                    "observed_at": str(row.get("observed_at") or ""),
                    "graph_revision": int(row.get("graph_revision") or 0),
                    "snapshot_sha256": str(row.get("content_sha256") or ""),
                }
    result = {
        "schema_version": CAMPAIGN_TRAJECTORY_SCHEMA,
        "snapshot_count": len(rows),
        "snapshots": rows,
        "first_achieved": {
            key: first_achieved[key] for key in sorted(first_achieved)
        },
        "semantics": {
            "benchmark_metrics_are_fixed_cutoff_projections": True,
            "trajectory_is_independent_of_result_view": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def snapshots_from_stages(
    stages: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(dict(row).get("detail") or {})
        for row in stages
        if isinstance(row, Mapping)
        and str(row.get("stage") or "").startswith("campaign_snapshot_")
        and isinstance(row.get("detail"), Mapping)
    ]


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "compile_campaign_snapshot",
    "compile_campaign_trajectory",
    "snapshots_from_stages",
]
