"""Run either the isolated V4 surface or the frozen combined compatibility UI."""
from __future__ import annotations

def serve_web(
    *,
    surface: str,
    host: str,
    port: int,
    server: str,
    threads: int,
    debug: bool,
) -> None:
    if surface == "v4":
        from cascade_planner.web.v4_app import create_v4_app

        app = create_v4_app()
    elif surface == "combined":
        from cascade_planner.web.app import create_app

        app = create_app()
    else:
        raise ValueError(f"unsupported_web_surface:{surface}")
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


__all__ = ["serve_web"]
