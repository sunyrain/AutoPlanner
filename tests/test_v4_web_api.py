from __future__ import annotations

from pathlib import Path

from flask import Flask

from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.web.v4_api import create_v4_blueprint


def _gateway(tmp_path: Path) -> CampaignGateway:
    paths = RuntimePaths.discover(
        repository_root=tmp_path,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "run-index.sqlite3"),
        },
    )
    return CampaignGateway(paths)


def test_v4_http_and_html_use_the_same_gateway_read_model(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()

    created = client.post(
        "/api/v4/runs",
        json={
            "run_id": "web-example",
            "target_name": "ethanol",
            "target_smiles": "CCO",
        },
    )
    assert created.status_code == 201
    assert created.get_json()["status"]["model_totals"]["model_invocations"] == 0

    status = client.get("/api/v4/runs/web-example/status")
    workbench = client.get("/api/v4/runs/web-example/workbench")
    rendered = client.get("/api/v4/runs/web-example/workbench.html")
    index = client.get("/v4")

    assert status.status_code == 200
    assert workbench.status_code == 200
    assert rendered.status_code == 200
    assert index.status_code == 200
    assert workbench.get_json()["snapshot"]["run_id"] == "web-example"
    assert b"web-example" in index.data
    assert b"<!doctype html>" in rendered.data.lower()


def test_v4_http_does_not_accept_arbitrary_run_directories(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()

    response = client.get(
        "/api/v4/runs/missing/status",
        query_string={"run_dir": str(tmp_path.parent)},
    )

    assert response.status_code == 404
    assert response.get_json()["reason"] == "run_not_found:missing"
