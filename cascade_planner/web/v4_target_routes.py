"""Register target solve and background-job HTTP routes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Callable, Mapping

from flask import Blueprint, Response, jsonify, request

from cascade_planner.application.blind_benchmark_contract import (
    BLIND_CASE_SCHEMA,
    BlindCase,
    audit_blind_preflight,
    canonical_smiles,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.web.v4_target_runtime import (
    job_projection as _job_projection,
    live_job_progress as _live_job_progress,
    new_run_id as _new_run_id,
    utc_now as _utc_now,
)
from cascade_planner.web.v4_run_catalog import (
    MAIN_REGISTRY_ID,
    catalog_identity as _catalog_identity,
    list_catalog_jobs as _list_catalog_jobs,
    make_job_id as _catalog_job_id,
    resolve_catalog_job as _resolve_catalog_job,
)
from cascade_planner.web.workspace_visibility import (
    WorkspaceVisibilityError,
    workspace_visibility_store,
)


GatewayFactory = Callable[[], CampaignGateway]
_OBJECTIVE_MODE_DEPRECATION = (
    "objective_mode is deprecated compatibility metadata; configure stock, "
    "acceptance and budgets directly. It does not change the unified solver."
)


def register_target_routes(
    blueprint: Blueprint,
    factory: GatewayFactory,
    *,
    jobs: dict[str, dict[str, Any]],
    jobs_lock: RLock,
    payload_reader: Callable[[], dict[str, Any]],
    solve_target_request: Callable[[Any, dict[str, Any]], dict[str, Any]],
    run_target_job: Callable[..., None],
) -> None:
    _payload = payload_reader
    _solve_target_request = solve_target_request
    _run_target_job = run_target_job

    @blueprint.post("/api/v4/solve-target")
    def solve_target():
        payload = _payload()
        result = _solve_target_request(factory(), payload)
        response = _with_objective_mode_deprecation(jsonify(result), payload)
        return response, 200 if payload.get("resume") is True else 201

    @blueprint.post("/api/v4/jobs")
    def start_target_job():
        payload = _payload()
        submitted_smiles = str(payload.get("target_smiles") or "").strip()
        if not submitted_smiles:
            raise ValueError("target_smiles_is_required")
        canonical = canonical_smiles(submitted_smiles)
        if not canonical:
            return jsonify(
                {
                    "error": "invalid_target_smiles",
                    "reason": "invalid_target_smiles",
                }
            ), 400
        run_scope = str(payload.get("run_scope") or "blind")
        if run_scope not in {"blind", "interactive"}:
            raise ValueError("target_run_scope_invalid")
        payload = {
            **payload,
            "run_scope": run_scope,
            "target_smiles": canonical,
        }
        if run_scope == "interactive":
            repository_matches = _interactive_repository_matches(
                factory(),
                canonical,
            )
            if repository_matches:
                repository_paths = sorted(
                    {
                        str(row.get("path") or "")
                        for row in repository_matches
                        if str(row.get("path") or "")
                    }
                )
                response = _with_objective_mode_deprecation(
                    jsonify(
                        {
                            "status": "repository_hit",
                            "phase": "repository_hit",
                            "target_name": str(payload.get("target_name") or "interactive target"),
                            "target_smiles": canonical,
                            "repository_match_count": len(repository_matches),
                            "repository_path_count": len(repository_paths),
                            "repository_matches": repository_matches,
                            "repository_paths": repository_paths,
                            "workspace_url": "/v4#routes",
                        }
                    ),
                    payload,
                )
                return response, 200
        run_id = str(payload.get("run_id") or "") or _new_run_id(
            str(payload.get("target_name") or "target")
        )
        job_id = _catalog_job_id(MAIN_REGISTRY_ID, run_id)
        payload = {**payload, "run_id": run_id}
        request_warnings = _compatibility_warnings(payload)
        now = _utc_now()
        with jobs_lock:
            existing = jobs.get(job_id)
            if existing and existing.get("status") in {"queued", "running", "cancelling"}:
                response = _with_objective_mode_deprecation(
                    jsonify(_job_projection(existing)), payload
                )
                return response, 200
            cancel_event = Event()
            jobs[job_id] = {
                "job_id": job_id,
                "run_id": run_id,
                "registry_id": MAIN_REGISTRY_ID,
                "registry_label": "Main local registry",
                "project_id": "local",
                "project_label": "Local AutoPlanner runs",
                "catalog_identity": _catalog_identity(MAIN_REGISTRY_ID, run_id),
                "registry_read_only": False,
                "target_name": str(payload.get("target_name") or "blind target"),
                "target_smiles": str(payload.get("target_smiles") or ""),
                "status": "queued",
                "phase": "queued",
                "created_at": now,
                "started_at": "",
                "finished_at": "",
                "updated_at": now,
                "elapsed_s": 0.0,
                "request_warnings": request_warnings,
                "error": "",
                "result": {},
                "cancel_requested_at": "",
                "cancelled_at": "",
                "cancellation_reason": "",
                "cancellation_available": True,
                "execution_source": "web",
                "_cancel_event": cancel_event,
            }
            row = dict(jobs[job_id])
        Thread(
            target=_run_target_job,
            args=(factory, payload, job_id, jobs, jobs_lock, cancel_event),
            daemon=True,
            name=f"autoplanner-{run_id[:32]}",
        ).start()
        response = _with_objective_mode_deprecation(jsonify(_job_projection(row)), payload)
        return response, 202

    @blueprint.get("/api/v4/jobs")
    def list_target_jobs():
        with jobs_lock:
            active_rows = [dict(value) for value in jobs.values()]
        gateway = factory()
        page = _list_catalog_jobs(
            gateway,
            active_rows=active_rows,
            limit=_query_int("limit", 30, maximum=200),
            offset=_query_int("offset", 0, minimum=0, maximum=1_000_000),
            project_id=str(request.args.get("project_id") or ""),
            registry_id=str(request.args.get("registry_id") or ""),
        )
        _apply_workspace_visibility(gateway, page["jobs"])
        return jsonify(page)

    @blueprint.get("/api/v4/jobs/<path:job_id>")
    def target_job_status(job_id: str):
        with jobs_lock:
            active_rows = [dict(value) for value in jobs.values()]
        gateway = factory()
        row, owning_gateway = _resolve_catalog_job(
            gateway,
            job_id,
            active_rows=active_rows,
        )
        if row is None or owning_gateway is None:
            return jsonify({"error": "job_not_found", "job_id": job_id}), 404
        projected = _job_with_live_progress(lambda: owning_gateway, row)
        _apply_workspace_visibility(gateway, [projected])
        return jsonify(projected)

    @blueprint.post("/api/v4/jobs/<path:job_id>/cancel")
    def cancel_target_job(job_id: str):
        payload = _payload()
        requested_reason = str(payload.get("reason") or "user_requested").strip()
        reason = requested_reason[:200] or "user_requested"
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None:
                status = str(job.get("status") or "")
                if status == "cancelled":
                    return jsonify(_job_projection(job)), 200
                if status == "cancelling":
                    event = job.get("_cancel_event")
                    if isinstance(event, Event):
                        event.set()
                    return jsonify(_job_projection(job)), 200
                if status not in {"queued", "running"}:
                    return jsonify(
                        {
                            "error": "job_cancel_conflict",
                            "reason": "job_is_not_active",
                            "job_id": job_id,
                            "status": status,
                        }
                    ), 409
                event = job.get("_cancel_event")
                if not isinstance(event, Event):
                    return jsonify(
                        {
                            "error": "job_cancel_unavailable",
                            "reason": "job_cancel_signal_missing",
                            "job_id": job_id,
                        }
                    ), 409
                requested_at = _utc_now()
                event.set()
                job.update(
                    status="cancelling",
                    phase="cancelling",
                    updated_at=requested_at,
                    cancel_requested_at=requested_at,
                    cancellation_reason=reason,
                )
                projected = _job_projection(job)
            else:
                projected = None
        if projected is not None:
            return jsonify(projected), 202

        gateway = factory()
        with jobs_lock:
            active_rows = [dict(value) for value in jobs.values()]
        registry_row, _owning_gateway = _resolve_catalog_job(
            gateway,
            job_id,
            active_rows=active_rows,
        )
        if registry_row is not None:
            run_id = str(registry_row.get("run_id") or "")
            registry_status = str(registry_row.get("status") or "historical")
            if registry_status in {"queued", "running", "paused"}:
                return jsonify(
                    {
                        "error": "job_cancel_unavailable",
                        "reason": "external_job_has_no_web_cancel_signal",
                        "job_id": job_id,
                        "run_id": run_id,
                        "status": registry_status,
                        "cancellation_available": False,
                    }
                ), 409
            return jsonify(
                {
                    "error": "job_cancel_conflict",
                    "reason": "job_is_not_active",
                    "job_id": job_id,
                    "status": registry_status,
                }
            ), 409
        return jsonify({"error": "job_not_found", "job_id": job_id}), 404

    @blueprint.delete("/api/v4/jobs/<path:job_id>")
    def delete_target_job(job_id: str):
        with jobs_lock:
            active_rows = [dict(value) for value in jobs.values()]
        gateway = factory()
        row, _owning_gateway = _resolve_catalog_job(
            gateway,
            job_id,
            active_rows=active_rows,
        )
        if row is None:
            return jsonify({"error": "job_not_found", "job_id": job_id}), 404
        run_id = str(row.get("run_id") or "")
        if str(row.get("status") or "") in {
            "queued",
            "running",
            "cancelling",
            "paused",
        }:
            return jsonify(
                {
                    "error": "job_delete_conflict",
                    "reason": "active_job_cannot_be_deleted",
                    "job_id": job_id,
                    "run_id": run_id,
                }
            ), 409
        registry_id = str(row.get("registry_id") or MAIN_REGISTRY_ID)
        identity = str(row.get("catalog_identity") or _catalog_identity(registry_id, run_id))
        try:
            result = workspace_visibility_store(gateway).hide_queue_run(identity)
        except WorkspaceVisibilityError as exc:
            return jsonify(
                {"error": "job_delete_failed", "reason": str(exc), "run_id": run_id}
            ), 400
        return jsonify({**result, "job_id": str(row.get("job_id") or job_id)})


def _compatibility_warnings(payload: Mapping[str, Any]) -> list[str]:
    return [_OBJECTIVE_MODE_DEPRECATION] if "objective_mode" in payload else []


def _interactive_repository_matches(
    gateway: CampaignGateway,
    target_smiles: str,
) -> list[dict[str, Any]]:
    """Return source-tree identity matches without applying blind rejection."""

    paths = getattr(gateway, "paths", None)
    repository_root = getattr(paths, "repository_root", None)
    if repository_root is None:
        return []
    root = Path(repository_root).expanduser().resolve()
    identity = hashlib.sha256(target_smiles.encode("utf-8")).hexdigest()
    case = BlindCase.from_dict(
        {
            "schema_version": BLIND_CASE_SCHEMA,
            "case_id": f"repository-lookup-{identity[:16]}",
            # This opaque label deliberately limits lookup to molecular
            # identity; a generic UI target name must not create false hits.
            "target_name": "target",
            "target_smiles": target_smiles,
            "acceptance": {},
            "budget": {},
        }
    )
    report = audit_blind_preflight(
        case,
        repository_root=root,
        run_dir=(root / ".autoplanner" / "interactive-repository-lookups" / identity[:16]),
    )
    return [dict(row) for row in report.get("repository_matches") or [] if isinstance(row, Mapping)]


def _with_objective_mode_deprecation(
    response: Response,
    payload: Mapping[str, Any],
) -> Response:
    if "objective_mode" not in payload:
        return response
    response.headers["Deprecation"] = "true"
    response.headers["Warning"] = (
        '299 AutoPlanner "objective_mode is deprecated compatibility metadata"'
    )
    return response


def _job_with_live_progress(
    factory: GatewayFactory,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    row = _job_projection(job)
    progress = _live_job_progress(factory, job)
    row["progress"] = progress
    for key in ("campaign_status", "campaign_terminal", "campaign_decision"):
        if key in progress:
            row[key] = progress[key]
    if progress.get("campaign_terminal") is True:
        decision = str(
            progress.get("campaign_decision")
            or progress.get("campaign_status")
            or "unresolved"
        ).casefold()
        row["status"] = {
            "completed": "complete",
            "cancelled": "cancelled",
            "failed": "failed",
            "budget_exhausted": "unresolved",
            "unresolved": "unresolved",
        }.get(decision, "unresolved")
        row["phase"] = decision
        row["cancellation_available"] = False
        return row
    if str(job.get("status") or "") in {
        "queued",
        "running",
        "cancelling",
        "paused",
    }:
        row["phase"] = str(progress.get("phase") or row["phase"])
    return row


def _apply_workspace_visibility(gateway: CampaignGateway, rows: list[dict[str, Any]]) -> None:
    try:
        visibility = workspace_visibility_store(gateway).snapshot()
        hidden_routes = set(dict(visibility.get("hidden_routes") or {}))
        hidden_queue_runs = set(dict(visibility.get("hidden_queue_runs") or {}))
        visibility_error = ""
    except WorkspaceVisibilityError as exc:
        hidden_routes = set()
        hidden_queue_runs = set()
        visibility_error = str(exc)
    for row in rows:
        run_id = str(row.get("run_id") or "")
        registry_id = str(row.get("registry_id") or MAIN_REGISTRY_ID)
        identity = str(row.get("catalog_identity") or _catalog_identity(registry_id, run_id))
        row["show_in_route_catalog"] = f"run:{identity}" not in hidden_routes
        row["show_in_task_queue"] = identity not in hidden_queue_runs
        if visibility_error:
            row["workspace_visibility_error"] = visibility_error


def _query_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int,
) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"query_{name}_invalid") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"query_{name}_invalid")
    return value


__all__ = ["register_target_routes"]
