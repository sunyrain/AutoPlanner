from __future__ import annotations

from pathlib import Path
import time

from flask import Flask

from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.web.v4_api import create_v4_blueprint
from cascade_planner.web.v4_target_runtime import historical_job
from cascade_planner.interfaces.target_delivery import delivery_projection


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
    assert b"/api/v4/runs" in index.data
    assert "逆合成控制台" in index.get_data(as_text=True)
    assert "路线候选已经可审查" in index.get_data(as_text=True)
    assert "只看每个目标最新结果" in index.get_data(as_text=True)
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


def test_v4_solve_target_maps_chemenzy_controls_to_shared_config() -> None:
    captured: dict = {}

    class RecordingGateway:
        def solve_target(self, **kwargs):
            captured.update(kwargs)
            return {"schema_version": "fixture", "run_id": "api-solve"}

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(RecordingGateway))
    response = app.test_client().post(
        "/api/v4/solve-target",
        json={
            "run_id": "api-solve",
            "target_name": "API target",
            "target_smiles": "CCOC(C)=O",
            "enable_chemenzy": True,
            "chemenzy_env_prefix": "D:/conda/envs/py312",
            "max_chemenzy_routes": 3,
            "max_chemenzy_steps": 5,
            "max_chemenzy_iterations": 4,
            "chemenzy_expansion_topk": 9,
            "chemenzy_timeout_s": 45,
            "max_model_invocations": 1,
        },
    )

    assert response.status_code == 201
    assert response.get_json()["run_id"] == "api-solve"
    config = captured["config"]
    assert config.enable_chemenzy is True
    assert config.chemenzy_env_prefix == "D:/conda/envs/py312"
    assert config.max_chemenzy_routes == 3
    assert config.max_chemenzy_steps == 5
    assert config.max_chemenzy_iterations == 4
    assert config.chemenzy_expansion_topk == 9
    assert config.chemenzy_timeout_s == 45.0
    assert config.max_guided_chemenzy_frontiers == 3
    assert config.max_guided_chemenzy_iterations == 6
    assert config.execution_profile == "standard"
    assert config.enable_initial_director_web_search is True
    assert config.max_visual_evidence_pages == 6
    assert captured["budget"].max_model_invocations == 1


def test_v4_async_job_returns_immediately_and_exposes_completion() -> None:
    class RecordingGateway:
        def solve_target(self, **kwargs):
            time.sleep(0.02)
            return {
                "run_id": kwargs["run_id"],
                "gates": {
                    "highest_contiguous_gate": "B1",
                    "gates": {"B1_global_multi_route": True},
                    "counts": {"target_rooted_distinct_skeletons": 2},
                },
                "claim": {"accepted_under_configured_policy": False},
                "model_cost": {"model_invocations": 1},
            }

        def status(self, _run_id):
            raise RuntimeError("fixture has no persistent kernel")

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(RecordingGateway))
    client = app.test_client()
    started = client.post(
        "/api/v4/jobs",
        json={"target_name": "async", "target_smiles": "CCO"},
    )

    assert started.status_code == 202
    job = started.get_json()
    assert job["run_id"].startswith("v4-async-")
    for _ in range(100):
        current = client.get(f"/api/v4/jobs/{job['job_id']}")
        assert current.status_code == 200
        value = current.get_json()
        if value["status"] == "complete":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("async V4 job did not complete")

    assert value["result"]["highest_contiguous_gate"] == "B1"
    assert value["result"]["model_cost"]["model_invocations"] == 1


def test_v4_delivery_projection_exposes_routes_before_proof_closure() -> None:
    delivery = delivery_projection(
        [
            {"stage": "global_campaign", "status": "completed"},
            {"stage": "initial_workbench", "status": "completed"},
            {"stage": "evidence_acquisition", "status": "running"},
        ],
        job_status="running",
    )

    assert delivery["state"] == "route_candidates_ready_evidence_running"
    assert delivery["route_candidates_available"] is True
    assert delivery["workbench_available"] is True
    assert delivery["proof_closure_complete"] is False


def test_historical_job_never_projects_archived_kernel_as_live_or_proven() -> None:
    job = historical_job(
        {
            "run_id": "archived-validation-fork",
            "target_name": "blind target",
            "status": "running",
            "accepted": True,
            "graph": {"complete_route_count": 3},
            "cost_totals": {"task_wall_time_s": 12.5},
        }
    )

    assert job["status"] == "historical"
    assert job["phase"] == "historical_snapshot"
    assert job["progress"]["execution_active"] is False
    assert job["progress"]["campaign_status"] == "running"
    delivery = job["progress"]["delivery"]
    assert delivery["state"] == "historical"
    assert delivery["route_candidates_available"] is True
    assert delivery["proof_closure_known"] is False
    assert delivery["proof_closure_complete"] is False
    assert delivery["semantics"]["portfolio_policy_accepted"] is True
