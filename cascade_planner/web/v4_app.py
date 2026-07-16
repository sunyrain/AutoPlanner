"""Isolated Flask application for the authoritative Canonical V4 surface."""
from __future__ import annotations

from flask import Flask, redirect

from cascade_planner.web.security import install_web_security
from cascade_planner.web.v4_api import GatewayFactory, create_v4_blueprint


def create_v4_app(gateway_factory: GatewayFactory | None = None) -> Flask:
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    install_web_security(app)
    app.register_blueprint(create_v4_blueprint(gateway_factory))

    @app.get("/")
    def index():
        return redirect("/v4", code=302)

    return app


__all__ = ["create_v4_app"]
