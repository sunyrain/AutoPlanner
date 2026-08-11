"""Register target solve and background-job HTTP routes."""
from __future__ import annotations

from threading import RLock, Thread
from typing import Any, Callable, Mapping

from flask import Blueprint, Response, jsonify

from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.web.v4_target_runtime import (
    historical_job as _historical_job,
    job_projection as _job_projection,
    live_job_progress as _live_job_progress,
    new_run_id as _new_run_id,
    utc_now as _utc_now,
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
        if not str(payload.get("target_smiles") or "").strip():
            raise ValueError("target_smiles_is_required")
        run_id = str(payload.get("run_id") or "") or _new_run_id(
            str(payload.get("target_name") or "target")
        )
        job_id = f"solve:{run_id}"
        payload = {**payload, "run_id": run_id}
        request_warnings = _compatibility_warnings(payload)
        now = _utc_now()
        with jobs_lock:
            existing = jobs.get(job_id)
            if existing and existing.get("status") in {"queued", "running"}:
                response = _with_objective_mode_deprecation(
                    jsonify(_job_projection(existing)), payload
                )
                return response, 200
            jobs[job_id] = {
                "job_id": job_id,
                "run_id": run_id,
                "target_name": str(payload.get("target_name") or "blind target"),
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
            }
            row = dict(jobs[job_id])
        Thread(
            target=_run_target_job,
            args=(factory, payload, job_id, jobs, jobs_lock),
            daemon=True,
            name=f"autoplanner-{run_id[:32]}",
        ).start()
        response = _with_objective_mode_deprecation(
            jsonify(_job_projection(row)), payload
        )
        return response, 202

    @blueprint.get("/api/v4/jobs")
    def list_target_jobs():
        with jobs_lock:
            active_rows = [dict(value) for value in jobs.values()]
        rows = [_job_with_live_progress(factory, value) for value in active_rows]
        known_run_ids = {str(row.get("run_id") or "") for row in rows}
        for run in factory().list_runs(limit=30).get("runs") or []:
            if not isinstance(run, Mapping):
                continue
            run_id = str(run.get("run_id") or "")
            if run_id and run_id not in known_run_ids:
                rows.append(_historical_job(run))
        rows.sort(key=lambda value: str(value.get("created_at") or ""), reverse=True)
        _apply_workspace_visibility(factory(), rows)
        return jsonify({"jobs": rows})

    @blueprint.get("/api/v4/jobs/<path:job_id>")
    def target_job_status(job_id: str):
        with jobs_lock:
            row = dict(jobs.get(job_id) or {})
        if not row:
            run_id = job_id.removeprefix("solve:")
            historical = next(
                (
                    value
                    for value in factory().list_runs(limit=100).get("runs") or []
                    if isinstance(value, Mapping) and str(value.get("run_id") or "") == run_id
                ),
                None,
            )
            if historical is None:
                return jsonify({"error": "job_not_found", "job_id": job_id}), 404
            row = _historical_job(historical)
        projected = _job_with_live_progress(factory, row)
        _apply_workspace_visibility(factory(), [projected])
        return jsonify(projected)

    @blueprint.delete("/api/v4/jobs/<path:job_id>")
    def delete_target_job(job_id: str):
        run_id = str(job_id or "").removeprefix("solve:").strip()
        if not run_id:
            return jsonify(
                {"error": "job_delete_invalid", "reason": "run_id_missing"}
            ), 400
        with jobs_lock:
            matching = [
                dict(value)
                for value in jobs.values()
                if str(value.get("run_id") or "") == run_id
            ]
            active = any(
                str(value.get("status") or "") in {"queued", "running"}
                for value in matching
            )
        if active:
            return jsonify(
                {
                    "error": "job_delete_conflict",
                    "reason": "active_job_cannot_be_deleted",
                    "job_id": job_id,
                    "run_id": run_id,
                }
            ), 409
        known = bool(matching) or any(
            isinstance(value, Mapping)
            and str(value.get("run_id") or "") == run_id
            for value in factory().list_runs(limit=1_000).get("runs") or []
        )
        if not known:
            return jsonify(
                {"error": "job_not_found", "job_id": job_id, "run_id": run_id}
            ), 404
        try:
            result = workspace_visibility_store(factory()).hide_queue_run(run_id)
        except WorkspaceVisibilityError as exc:
            return jsonify(
                {"error": "job_delete_failed", "reason": str(exc), "run_id": run_id}
            ), 400
        return jsonify({**result, "job_id": f"solve:{run_id}"})


def _compatibility_warnings(payload: Mapping[str, Any]) -> list[str]:
    return (
        [_OBJECTIVE_MODE_DEPRECATION]
        if "objective_mode" in payload
        else []
    )


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
    if str(job.get("status") or "") in {"queued", "running"}:
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
        row["show_in_route_catalog"] = f"run:{run_id}" not in hidden_routes
        row["show_in_task_queue"] = run_id not in hidden_queue_runs
        if visibility_error:
            row["workspace_visibility_error"] = visibility_error



__all__ = ["register_target_routes"]
