"""Canonical V4 workspace helpers shared by the UI and HTTP adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, Response, abort, jsonify, redirect, request

from cascade_planner.web.workspace_catalog import compile_showcase_catalog


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
SHARED_RESULTS_DIR = ROOT / "results" / "shared"
PRESENTATION_MANIFEST = SHARED_RESULTS_DIR / "presentation_showcase_20260715" / "manifest.json"


def static_html(name: str) -> Response:
    path = STATIC_DIR / name
    return Response(path.read_text(encoding="utf-8"), mimetype="text/html")


def showcase_catalog() -> dict[str, Any]:
    return compile_showcase_catalog(
        root=ROOT,
        shared_root=SHARED_RESULTS_DIR,
        manifest_path=PRESENTATION_MANIFEST,
    )


def workspace_payload(gateway: Any) -> dict[str, Any]:
    try:
        listed = gateway.list_runs(limit=40)
        runs = [dict(row) for row in listed.get("runs") or [] if isinstance(row, dict)]
        backend = {
            "available": True,
            "state": "ready",
            "run_count": int(listed.get("run_count") or len(runs)),
        }
        error = ""
    except Exception as exc:  # UI projection must not hide the static showcase.
        runs = []
        backend = {"available": False, "state": "unavailable", "run_count": 0}
        error = f"{type(exc).__name__}:{exc}"
    for row in runs:
        run_id = str(row.get("run_id") or "")
        row["workbench_url"] = f"/api/v4/runs/{run_id}/workbench.html" if run_id else ""
        row["status_url"] = f"/api/v4/runs/{run_id}/status" if run_id else ""
    catalog = showcase_catalog()
    return {
        "schema_version": "autoplanner.workspace.v2",
        "ok": backend["available"] or catalog["ok"],
        "backend": backend,
        "backend_error": error,
        "entrypoints": {
            "primary_page": "/v4",
            "launch": "/v4#new-task",
            "routes": "/v4#routes",
            "runs": "/v4#runs",
            "audits": "/v4#audits",
            "runs_api": "/api/v4/runs",
            "jobs_api": "/api/v4/jobs",
        },
        "runs": runs,
        "showcase": catalog,
        "semantics": {
            "canonical_backend_is_the_only_run_authority": True,
            "one_user_facing_page": True,
            "showcase_artifacts_are_read_only": True,
            "workbench_is_rendered_from_the_same_gateway_read_model": True,
        },
    }


def register_workspace_routes(blueprint: Blueprint, gateway_factory: Any) -> None:
    @blueprint.get("/v4")
    def v4_index() -> Response:
        return static_html("workspace.html")

    @blueprint.get("/v4/console")
    def v4_console() -> Response:
        return redirect("/v4#new-task", code=302)

    @blueprint.get("/v4/showcase")
    def v4_showcase() -> Response:
        return redirect("/v4#routes", code=302)

    @blueprint.get("/agent")
    def legacy_agent_workbench() -> Response:
        return redirect("/v4#routes", code=302)

    @blueprint.get("/statins")
    def legacy_statin_showcase() -> Response:
        return redirect("/v4#audits", code=302)

    @blueprint.get("/showcase")
    def legacy_presentation_showcase() -> Response:
        return redirect("/v4#routes", code=302)

    @blueprint.get("/api/v4/workspace")
    def v4_workspace():
        return jsonify(workspace_payload(gateway_factory()))

    @blueprint.get("/api/v4/showcase")
    def v4_showcase_catalog():
        return jsonify(showcase_catalog())

    @blueprint.route("/api/v4/result-file", methods=["GET", "HEAD"])
    def v4_result_file():
        response = result_file_response(
            str(request.args.get("path") or ""),
            head_only=request.method == "HEAD",
        )
        if response is None:
            abort(404)
        return response


def result_file_response(relative_path: str, *, head_only: bool = False) -> Response | None:
    requested = str(relative_path or "").strip().replace("\\", "/")
    candidate = (ROOT / requested).resolve() if requested else ROOT.resolve()
    shared = SHARED_RESULTS_DIR.resolve()
    if not requested or not candidate.is_relative_to(shared) or not candidate.is_file():
        return None
    mimetype = {
        ".html": "text/html; charset=utf-8",
        ".htm": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".jsonl": "application/x-ndjson; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }.get(candidate.suffix.casefold(), "application/octet-stream")
    if head_only:
        response = Response(mimetype=mimetype)
        response.content_length = candidate.stat().st_size
    elif mimetype.startswith("text/") or "json" in mimetype or mimetype == "image/svg+xml":
        response = Response(candidate.read_text(encoding="utf-8", errors="replace"), mimetype=mimetype)
    else:
        response = Response(candidate.read_bytes(), mimetype=mimetype)
    if candidate.suffix.casefold() in {".html", ".htm"}:
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "img-src data:; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
    return response


__all__ = [
    "register_workspace_routes",
    "result_file_response",
    "showcase_catalog",
    "static_html",
    "workspace_payload",
]
