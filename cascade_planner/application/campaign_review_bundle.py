"""Content-bound reviewer traces projected from one target solve report."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.candidate_provenance import candidate_review_lineage_records


CAMPAIGN_REVIEW_BUNDLE_SCHEMA = "campaign_review_bundle.v1"
CAMPAIGN_ACTION_TRACE_SCHEMA = "campaign_action_trace.v1"
CAMPAIGN_FAILURE_TRACE_SCHEMA = "campaign_failure_trace.v1"
CAMPAIGN_ROUTE_LINEAGE_EXPORT_SCHEMA = "campaign_route_lineage_export.v1"
CAMPAIGN_RESOURCE_CURVE_EXPORT_SCHEMA = "campaign_resource_curve_export.v1"

_FAILURE_STATUSES = {
    "cancelled",
    "failed",
    "invalid",
    "partial",
    "rejected",
    "timed_out",
    "timeout",
    "unavailable",
}


def compile_campaign_review_bundle(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Split a large solve report into four stable reviewer-facing traces."""

    source = _json_value(report or {})
    source_present = bool(source)
    source_valid = _content_digest_valid(source)
    trusted_source = source if source_valid else {}
    stages = [
        dict(row)
        for row in trusted_source.get("stages") or []
        if isinstance(row, Mapping)
    ]
    trajectory = dict(trusted_source.get("trajectory") or {})
    trajectory_valid = _content_digest_valid(trajectory)
    candidate_lifecycle = dict(trusted_source.get("candidate_lifecycle") or {})
    candidate_provenance = dict(trusted_source.get("candidate_provenance") or {})
    action_trace = _component(
        CAMPAIGN_ACTION_TRACE_SCHEMA,
        records=_action_records(stages),
        semantics={
            "records_are_ordered_by_report_stage": True,
            "action_execution_identity_is_preserved": True,
            "trace_grants_no_scientific_authority": True,
        },
    )
    failure_trace = _component(
        CAMPAIGN_FAILURE_TRACE_SCHEMA,
        records=_failure_records(stages, trusted_source),
        semantics={
            "explicit_failures_and_terminal_reasons_are_retained": True,
            "open_scientific_gates_are_not_relabelled_as_runtime_failures": True,
            "absence_of_records_is_not_proof_of_success": True,
        },
    )
    route_lineage = _component(
        CAMPAIGN_ROUTE_LINEAGE_EXPORT_SCHEMA,
        records=_route_lineage_records(stages, trajectory, candidate_lifecycle, candidate_provenance),
        semantics={
            "provider_and_canonical_lineage_are_separate": True,
            "raw_normalized_admitted_materialized_dispositions_are_retained": True,
            "candidate_lifecycle_dispositions_are_digest_verified": True,
            "lineage_does_not_upgrade_route_proof": True,
        },
    )
    resource_curve = {
        "schema_version": CAMPAIGN_RESOURCE_CURVE_EXPORT_SCHEMA,
        "available": trajectory_valid,
        "unavailable_reason": (
            ""
            if trajectory_valid
            else "trajectory_digest_invalid_or_missing"
            if source_valid
            else "target_solve_report_digest_invalid_or_missing"
        ),
        "trajectory_sha256": str(trajectory.get("content_sha256") or ""),
        "snapshot_count": int(trajectory.get("snapshot_count") or 0),
        "time_to_first": (
            _json_value(trajectory.get("time_to_first") or {})
            if trajectory_valid
            else {}
        ),
        "continuity": (
            _json_value(trajectory.get("continuity") or {})
            if trajectory_valid
            else {}
        ),
        "binding_epochs": (
            _json_value(trajectory.get("binding_epochs") or [])
            if trajectory_valid
            else []
        ),
        "records": (
            _json_value(trajectory.get("resource_curve") or [])
            if trajectory_valid
            else []
        ),
        "semantics": {
            "cumulative_resources_are_not_recomputed_from_stage_timings": True,
            "resume_continuity_is_exported": True,
            "trace_grants_no_scientific_authority": True,
        },
    }
    resource_curve["content_sha256"] = _digest(resource_curve)
    components = {
        "action_trace": action_trace,
        "failure_trace": failure_trace,
        "route_lineage": route_lineage,
        "resource_curve": resource_curve,
    }
    bundle = {
        "schema_version": CAMPAIGN_REVIEW_BUNDLE_SCHEMA,
        "available": source_valid,
        "unavailable_reason": (
            ""
            if source_valid
            else "target_solve_report_digest_invalid"
            if source_present
            else "target_solve_report_missing"
        ),
        "run_id": str(trusted_source.get("run_id") or ""),
        "report_sha256": str(source.get("content_sha256") or ""),
        "source_report_digest_valid": source_valid,
        "components": components,
        "component_sha256": {
            key: str(value.get("content_sha256") or "")
            for key, value in components.items()
        },
        "semantics": {
            "bundle_is_a_read_only_report_projection": True,
            "component_digests_are_independently_verifiable": True,
            "canonical_graph_and_run_events_remain_authority": True,
        },
    }
    bundle["content_sha256"] = _digest(bundle)
    return bundle


def _action_records(stages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
        records.append(
            {
                "stage_index": stage_index,
                "stage": str(stage.get("stage") or ""),
                "stage_status": str(stage.get("status") or ""),
                "execution_id": execution_id,
                "action": _json_value(action),
                "estimate": _json_value(detail.get("estimate") or {}),
                "decision": _json_value(detail.get("decision") or {}),
                "outcome": _json_value(outcome),
            }
        )
    return records


def _failure_records(
    stages: Iterable[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for stage_index, stage in enumerate(stages):
        detail = dict(stage.get("detail") or {})
        outcome = dict(detail.get("outcome") or {})
        stage_status = str(stage.get("status") or "").casefold()
        outcome_status = str(outcome.get("status") or "").casefold()
        failure_reasons = _strings(
            outcome.get("failure_reasons") or detail.get("failure_reasons") or []
        )
        if not failure_reasons and (
            stage_status in _FAILURE_STATUSES
            or outcome_status in _FAILURE_STATUSES
        ):
            failure_reasons = _strings(detail.get("reasons") or [])
        error = str(outcome.get("error") or detail.get("error") or "")
        failure_type = str(outcome.get("failure_type") or detail.get("failure_type") or "")
        if not (
            stage_status in _FAILURE_STATUSES
            or outcome_status in _FAILURE_STATUSES
            or failure_reasons
            or error
            or failure_type
        ):
            continue
        records.append(
            {
                "kind": "stage_or_action_failure",
                "stage_index": stage_index,
                "stage": str(stage.get("stage") or ""),
                "stage_status": stage_status,
                "action_execution_id": str(
                    dict(detail.get("action") or {}).get("execution_id")
                    or outcome.get("action_execution_id")
                    or ""
                ),
                "outcome_status": outcome_status,
                "failure_type": failure_type,
                "failure_reasons": failure_reasons,
                "error": error[:4000],
            }
        )
    stop = dict(report.get("stop_decision") or {})
    if stop:
        records.append(
            {
                "kind": "terminal_decision",
                "decision": str(stop.get("decision") or ""),
                "terminal": stop.get("terminal") is True,
                "reasons": _strings(stop.get("reasons") or []),
                "content_sha256": str(stop.get("content_sha256") or ""),
            }
        )
    return records


def _route_lineage_records(
    stages: Iterable[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    candidate_lifecycle: Mapping[str, Any],
    candidate_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for stage_index, stage in enumerate(stages):
        name = str(stage.get("stage") or "")
        detail = dict(stage.get("detail") or {})
        if name == "chemenzy_route_lineage":
            records.append(
                {
                    "kind": "final_provider_lineage",
                    "stage_index": stage_index,
                    "lineage": _json_value(detail),
                }
            )
        elif name == "chemenzy_baseline" and detail.get("route_lineage"):
            records.append(
                {
                    "kind": "provider_ingestion_lineage",
                    "stage_index": stage_index,
                    "lineage": _json_value(detail.get("route_lineage") or []),
                }
            )
    if _content_digest_valid(trajectory):
        snapshots = [
            dict(row)
            for row in trajectory.get("snapshots") or []
            if isinstance(row, Mapping)
        ]
        latest = snapshots[-1] if snapshots else {}
        records.append(
            {
                "kind": "canonical_pareto_lineage",
                "snapshot_sha256": str(latest.get("content_sha256") or ""),
                "routes": _json_value(latest.get("pareto_archive") or []),
            }
        )
    records.extend(
        candidate_review_lineage_records(candidate_lifecycle, candidate_provenance)
    )
    return records


def _component(
    schema_version: str,
    *,
    records: Iterable[Mapping[str, Any]],
    semantics: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [_json_value(row) for row in records]
    result = {
        "schema_version": schema_version,
        "record_count": len(rows),
        "records": rows,
        "semantics": _json_value(semantics),
    }
    result["content_sha256"] = _digest(result)
    return result


def _strings(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted({str(item) for item in values if str(item).strip()})


def _content_digest_valid(value: Mapping[str, Any]) -> bool:
    if not value:
        return False
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return len(supplied) == 64 and supplied == _digest(row)


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
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
    "CAMPAIGN_ACTION_TRACE_SCHEMA",
    "CAMPAIGN_FAILURE_TRACE_SCHEMA",
    "CAMPAIGN_RESOURCE_CURVE_EXPORT_SCHEMA",
    "CAMPAIGN_REVIEW_BUNDLE_SCHEMA",
    "CAMPAIGN_ROUTE_LINEAGE_EXPORT_SCHEMA",
    "compile_campaign_review_bundle",
]
