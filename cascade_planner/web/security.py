"""Shared request guards for every local AutoPlanner Web surface."""
from __future__ import annotations

import hmac
import os

from flask import Flask, Response, abort, request


def install_web_security(app: Flask) -> None:
    @app.before_request
    def protect_mutating_api() -> None:
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        if not request.is_json:
            abort(415, description="mutating API requests require application/json")
        configured_token = str(os.environ.get("AUTOPLANNER_WEB_API_TOKEN") or "")
        if configured_token:
            supplied = str(request.headers.get("X-Autoplanner-Token") or "")
            if not hmac.compare_digest(configured_token, supplied):
                abort(401, description="missing or invalid API token")
        fetch_site = str(request.headers.get("Sec-Fetch-Site") or "").lower()
        if fetch_site in {"cross-site", "same-site"}:
            abort(403, description="cross-site mutation rejected")
        origin = str(request.headers.get("Origin") or "").rstrip("/")
        if origin and origin != request.host_url.rstrip("/"):
            abort(403, description="origin does not match this AutoPlanner service")
        return None

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return response


__all__ = ["install_web_security"]
