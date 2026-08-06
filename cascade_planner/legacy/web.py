"""Explicit loader for the frozen combined Web application."""
from __future__ import annotations

import warnings

from flask import Flask


def create_combined_app() -> Flask:
    warnings.warn(
        "the combined V3/V4 Web surface is deprecated; use the V4 surface",
        FutureWarning,
        stacklevel=2,
    )
    from cascade_planner.legacy.web_runtime.app import create_app

    return create_app()


def serve_combined_web(
    *,
    host: str,
    port: int,
    server: str,
    threads: int,
    debug: bool,
) -> None:
    app = create_combined_app()
    if server in {"auto", "waitress"}:
        try:
            from waitress import serve
        except ImportError as exc:
            if server == "waitress":
                raise ValueError(
                    "waitress_not_installed; install requirements or use --server auto/flask"
                ) from exc
        else:
            serve(
                app,
                host=host,
                port=port,
                threads=max(1, min(32, int(threads))),
                channel_timeout=30,
            )
            return
    app.run(host=host, port=port, debug=debug, threaded=True)


__all__ = ["create_combined_app", "serve_combined_web"]
