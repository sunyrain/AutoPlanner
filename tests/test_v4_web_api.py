from __future__ import annotations

import json
from pathlib import Path
from threading import Event, RLock
import time
from unittest.mock import Mock

from flask import Flask

from cascade_planner.application.biocatalytic_programs import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    with_biocatalysis_program_validation_digest,
)
from cascade_planner.application.experiment_execution_results import (
    build_experiment_execution_result,
)
from cascade_planner.application.experiment_external_jobs import (
    build_experiment_cancellation_request,
    build_experiment_external_job_receipt,
    build_experiment_operator_identity,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.web.v4_api import create_v4_blueprint
from cascade_planner.web.v4_app import create_v4_app
from cascade_planner.web.v4_target_runtime import historical_job
from cascade_planner.web.v4_target_runtime import run_target_job
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


def _web_reduction_capability() -> dict:
    return {
        "capability_id": "fixture:web-carbonyl-reduction",
        "enzyme": {"classes": ["alcohol dehydrogenase"]},
        "match": {
            "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
            "element_delta": {"C": 0, "O": 0},
            "min_scaffold_similarity": 0.05,
            "max_abs_heavy_atom_delta": 0,
            "min_substrate_carbons": 2,
            "min_window_steps": 1,
            "max_window_steps": 1,
            "reject_unlisted_motif_changes": True,
        },
        "selectivity_objective": "Reduce the carbonyl without changing carbon count.",
        "substrate_scope_basis": "web fixture analog",
        "precedent_refs": ["doi:10.1000/web-reduction"],
    }


def _web_biocatalysis_validation(proposal: dict) -> dict:
    return with_biocatalysis_program_validation_digest(
        {
            "schema_version": BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
            "validation_id": "validation:web-reduction",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "accepted": True,
            "evidence_tier": "exact_substrate_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "claim_refs": ["claim:web-exact-substrate-screen"],
            "condition_record_ids": [],
            "selectivity_assessed": True,
            "cofactor_ledger_closed": True,
            "outcome": {"conversion_fraction": 0.88},
        }
    )


def test_web_routes_explicit_experiment_transport_operations() -> None:
    gateway = Mock()
    methods = {
        "submit": "submit_route_experiment_job",
        "poll": "poll_route_experiment_job",
        "cancel": "transmit_route_experiment_cancellation",
    }
    for operation, method_name in methods.items():
        getattr(gateway, method_name).return_value = {
            "operation": operation, "run_id": "web-transport"
        }
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()
    payload = {
        "route_id": "route:web-transport", "capabilities": [],
        "dispatch_id": "experiment-dispatch:" + "b" * 32,
        "timeout_s": 9.5, "enable_experiment_transport": True,
    }
    for operation, method_name in methods.items():
        response = client.post(
            "/api/v4/runs/web-transport/programs/innovations/experiments/"
            f"transport/{operation}",
            json=payload,
        )
        assert response.status_code == 200
        assert response.get_json()["operation"] == operation
        getattr(gateway, method_name).assert_called_once_with(
            "web-transport", route_id="route:web-transport", capabilities=[],
            mechanism_proposals=[], validations=[],
            dispatch_id="experiment-dispatch:" + "b" * 32,
            timeout_s=9.5, enable_experiment_transport=True,
        )


def test_background_job_finishes_on_the_unified_acceptance_projection(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cascade_planner.web.v4_target_runtime.solve_target_request",
        lambda _gateway, _payload: {
            "run_id": "benchmark-objective",
            "report_path": "report.json",
            "claim": {
                "accepted_under_configured_policy": False,
                "objective_mode": "scientific_proof",
                "objective_achieved": True,
            },
            "gates": {"gates": {"B4_stock_boundary": True}, "counts": {}},
            "model_cost": {},
            "stop_decision": {"decision": "completed", "terminal": True},
        },
    )
    jobs = {"job-1": {"status": "queued"}}

    run_target_job(lambda: object(), {}, "job-1", jobs, RLock())

    assert jobs["job-1"]["status"] == "complete"
    assert jobs["job-1"]["result"]["accepted"] is False
    assert jobs["job-1"]["result"]["objective_achieved"] is True


def test_v4_api_marks_legacy_objective_mode_as_deprecated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gateway = _gateway(tmp_path)
    monkeypatch.setattr(
        "cascade_planner.web.v4_api._run_target_job",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cascade_planner.web.v4_api._solve_target_request",
        lambda _gateway, _payload: {"run_id": "legacy-sync"},
    )
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()

    synchronous = client.post(
        "/api/v4/solve-target",
        json={
            "target_name": "legacy sync view",
            "target_smiles": "CCO",
            "objective_mode": "procurement_delivery",
        },
    )
    response = client.post(
        "/api/v4/jobs",
        json={
            "target_name": "legacy view",
            "target_smiles": "CCO",
            "objective_mode": "benchmark_search",
        },
    )

    assert synchronous.status_code == 201
    assert synchronous.headers["Deprecation"] == "true"
    assert "objective_mode is deprecated" in synchronous.headers["Warning"]
    assert response.status_code == 202
    assert response.headers["Deprecation"] == "true"
    assert "objective_mode is deprecated" in response.headers["Warning"]
    assert response.get_json()["request_warnings"] == [
        "objective_mode is deprecated compatibility metadata; configure stock, "
        "acceptance and budgets directly. It does not change the unified solver."
    ]


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
            "forbidden_reagents": ["benzene"],
            "max_route_steps": 8,
            "allowed_execution_domains": ["chemical", "hybrid"],
            "safety_limits": {"max_temperature_c": 120},
            "stock_source_ids": ["host-default"],
        },
    )
    assert created.status_code == 201
    assert created.get_json()["status"]["model_totals"]["model_invocations"] == 0
    constraints = created.get_json()["campaign_spec"]["constraints"]
    assert constraints["forbidden_reagents"] == ["benzene"]
    assert constraints["max_route_steps"] == 8
    assert constraints["allowed_execution_domains"] == ["chemical", "hybrid"]
    assert constraints["safety_limits"] == {"max_temperature_c": 120}

    status = client.get("/api/v4/runs/web-example/status")
    workbench = client.get("/api/v4/runs/web-example/workbench")
    programs = client.get("/api/v4/runs/web-example/programs")
    program_routes = client.get("/api/v4/runs/web-example/programs/routes")
    program_audit = client.get("/api/v4/program-migration", query_string={"run_id": "web-example"})
    rendered = client.get("/api/v4/runs/web-example/workbench.html")
    index = client.get("/v4")
    console = client.get("/v4/console")
    showcase = client.get("/v4/showcase")
    legacy_agent = client.get("/agent")
    legacy_statins = client.get("/statins")
    legacy_showcase = client.get("/showcase")
    workspace = client.get("/api/v4/workspace")

    assert status.status_code == 200
    assert workbench.status_code == 200
    assert rendered.status_code == 200
    assert index.status_code == 200
    assert console.status_code == 302
    assert console.headers["Location"] == "/v4#new-task"
    assert showcase.status_code == 302
    assert showcase.headers["Location"] == "/v4#routes"
    assert legacy_agent.headers["Location"] == "/v4#routes"
    assert legacy_statins.headers["Location"] == "/v4#audits"
    assert legacy_showcase.headers["Location"] == "/v4#routes"
    assert workspace.status_code == 200
    assert workspace.get_json()["backend"]["available"] is True
    assert workspace.get_json()["runs"][0]["run_id"] == "web-example"
    assert workspace.get_json()["runs"][0]["workbench_pdf_url"].endswith(
        "/web-example/workbench.pdf"
    )
    assert workspace.get_json()["runs"][0]["history_delete_url"].endswith(
        "/web-example/history"
    )
    assert workspace.get_json()["self_evolution"]["schema_version"] == (
        "autoplanner.self_evolution_catalog.v1"
    )
    assert "compiled_program_benchmarks" not in workspace.get_json()["self_evolution"]
    assert workspace.get_json()["route_workbench"]["program_benchmarks"]["record_count"] >= 1
    assert workspace.get_json()["route_workbench"]["semantics"][
        "program_benchmarks_are_not_self_evolution_memory"
    ]
    assert workspace.get_json()["entrypoints"]["self_evolution"] == "/v4#evolution"
    assert workbench.get_json()["snapshot"]["run_id"] == "web-example"
    assert programs.status_code == 200
    assert programs.get_json()["oracle"]["accepted"] is True
    assert programs.get_json()["projection"]["counts"]["programs"] == 0
    assert program_routes.status_code == 200
    assert program_routes.get_json()["oracle"]["accepted"] is True
    assert program_routes.get_json()["overlay"]["counts"]["displayed_routes"] == 0
    assert program_audit.status_code == 200
    assert program_audit.get_json()["run_count"] == 1
    assert program_audit.get_json()["semantics"]["read_only"] is True
    assert b"/api/v4/runs" in index.data
    assert "AutoPlanner · 统一工作区" in index.get_data(as_text=True)
    assert "路线候选可供审查，不代表证据、库存或工艺已经闭合" in index.get_data(as_text=True)
    assert 'id="objectiveMode"' not in index.get_data(as_text=True)
    assert "objective_mode:$('objectiveMode').value" not in index.get_data(as_text=True)
    assert "所有任务进入同一条 anytime trajectory" in index.get_data(as_text=True)
    assert "/api/v4/workspace" in index.get_data(as_text=True)
    assert 'id="collapseLibrary"' in index.get_data(as_text=True)
    assert 'id="restoreLibrary"' in index.get_data(as_text=True)
    assert 'id="launchDialog"' in index.get_data(as_text=True)
    assert 'data-view-panel="overview"' in index.get_data(as_text=True)
    assert 'data-view-panel="routes"' in index.get_data(as_text=True)
    assert 'data-view-panel="runs"' in index.get_data(as_text=True)
    assert 'data-view-panel="evolution"' in index.get_data(as_text=True)
    assert 'data-view-panel="audits"' in index.get_data(as_text=True)
    assert "sidebar-collapsed" in index.get_data(as_text=True)
    assert "catalog-collapsed" in index.get_data(as_text=True)
    assert "embed=1" in index.get_data(as_text=True)
    assert 'id="solveForm"' in index.get_data(as_text=True)
    assert 'id="forbiddenReagents"' in index.get_data(as_text=True)
    assert 'id="executionDomains"' in index.get_data(as_text=True)
    assert "forbidden_reagents:csv($('forbiddenReagents').value)" in index.get_data(
        as_text=True
    )
    assert "启动逆合成" in index.get_data(as_text=True)
    assert "自进化库" in index.get_data(as_text=True)
    assert "多步 Program 作为宿主路线内的可验证替代层展示" in index.get_data(as_text=True)
    assert "Program 在所属路线的准确区间内显示" in index.get_data(as_text=True)
    assert "programHostRoutes" in index.get_data(as_text=True)
    assert "路线锚定机理假设" in index.get_data(as_text=True)
    assert 'class="targetGroup"' in index.get_data(as_text=True)
    assert 'id="runFrame"' not in index.get_data(as_text=True)
    assert 'id="runDetail"' in index.get_data(as_text=True)
    assert "统一 Action 时间线" in index.get_data(as_text=True)
    assert "renderSelectedRunActionTimeline" in index.get_data(as_text=True)
    assert "ChemEnzy · Codex · 证据 · 验证 · Program" in index.get_data(
        as_text=True
    )
    assert 'id="memoryDialog"' in index.get_data(as_text=True)
    assert "enhanceMemoryRows" in index.get_data(as_text=True)
    assert "reaction_smarts" in index.get_data(as_text=True)
    assert 'id="inputTokens"' in index.get_data(as_text=True)
    assert 'id="outputTokens"' in index.get_data(as_text=True)
    assert 'id="modelWallMinutes"' in index.get_data(as_text=True)
    assert "max_input_tokens:Number($('inputTokens').value)" in index.get_data(
        as_text=True
    )
    for field_id in (
        "totalTasks",
        "evidenceTasks",
        "stockTasks",
        "validationTasks",
        "programTasks",
        "experimentTasks",
        "runWallMinutes",
    ):
        assert f'id="{field_id}"' in index.get_data(as_text=True)
    assert "max_total_tasks:Number($('totalTasks').value)" in index.get_data(
        as_text=True
    )
    assert "max_evidence_tasks:Number($('evidenceTasks').value)" in index.get_data(
        as_text=True
    )
    assert "max_stock_tasks:Number($('stockTasks').value)" in index.get_data(
        as_text=True
    )
    assert "max_validation_tasks:Number($('validationTasks').value)" in index.get_data(
        as_text=True
    )
    assert "max_program_tasks:Number($('programTasks').value)" in index.get_data(
        as_text=True
    )
    assert "max_experiment_tasks:Number($('experimentTasks').value)" in index.get_data(
        as_text=True
    )
    assert "max_run_wall_time_s:Number($('runWallMinutes').value)*60" in index.get_data(
        as_text=True
    )
    assert "inputTokens:1200000" in index.get_data(as_text=True)
    assert "outputTokens:200000" in index.get_data(as_text=True)
    assert "modelWallMinutes:30" in index.get_data(as_text=True)
    assert "chemTimeoutMinutes:60" in index.get_data(as_text=True)
    assert "'/api/v4/jobs'" in index.get_data(as_text=True)
    assert 'id="downloadRoutePdf"' in index.get_data(as_text=True)
    assert "删除队列记录" in index.get_data(as_text=True)
    assert 'id="deleteRoute"' in index.get_data(as_text=True)
    assert 'id="restoreDeletedRoutes"' in index.get_data(as_text=True)
    assert 'id="restoreDeletedRuns"' in index.get_data(as_text=True)
    assert "计算结束，但科学结论未收敛" in index.get_data(as_text=True)
    assert "function mergeRunSnapshots" in index.get_data(as_text=True)
    assert "hydratedRuns:new Set()" in index.get_data(as_text=True)
    assert "state.jobs=mergeRunSnapshots(await jobsWithProgress())" in index.get_data(
        as_text=True
    )
    assert b"<!doctype html>" in rendered.data.lower()
    assert rendered.get_data(as_text=True).count('id="dashboardReturn"') == 1
    assert rendered.get_data(as_text=True).count('id="pdfExport"') == 1
    assert 'aria-label="返回统一总控台"' in rendered.get_data(as_text=True)

    benchmark = next(
        row
        for row in workspace.get_json()["route_workbench"]["program_benchmarks"]["records"]
        if row["target_name"] == "bufotalin" and row["chemical_step_equivalent_count"] == 6
    )
    assert benchmark["mechanism_hypothesis_count"] == 1
    materialized = client.post(benchmark["materialize_url"])
    assert materialized.status_code == 201
    assert materialized.get_json()["chemical_baseline_step_count"] == 20
    assert materialized.get_json()["hypothetical_operation_count"] == 15
    assert materialized.get_json()["semantics"]["materialization_does_not_admit_the_program"]
    host_snapshot = client.get(
        f"/api/v4/runs/{materialized.get_json()['run_id']}/workbench"
    ).get_json()["snapshot"]
    planned = next(iter(host_snapshot["planned_routes"].values()))
    assert planned["declared_step_count"] == 20
    assert planned["materialized_step_count"] == 15
    assert planned["admission_rejected_step_count"] == 5
    program_workbench = client.get(materialized.get_json()["workbench_url"])
    assert program_workbench.status_code == 200
    assert "program-overlay-card" in program_workbench.get_data(as_text=True)
    assert "mechanism-hypothesis-callout" in program_workbench.get_data(as_text=True)
    assert "proposed unprotected C16-ketone" in program_workbench.get_data(as_text=True)
    assert "anchor_evidence_not_promoted" in program_workbench.get_data(as_text=True)
    assert "PRODUCT_NOT_ROUTE_REJOINED" in program_workbench.get_data(as_text=True)
    assert "Ct3alpha-HSDH" in program_workbench.get_data(as_text=True)
    assert '"chemical_step_equivalent_count":6' in program_workbench.get_data(as_text=True)
    assert '"primary_branch_id":"planned-route:' in program_workbench.get_data(as_text=True)
    refreshed_workspace = client.get("/api/v4/workspace").get_json()
    benchmark_run = next(
        row
        for row in refreshed_workspace["runs"]
        if row["run_id"] == materialized.get_json()["run_id"]
    )
    assert benchmark_run["surface_role"] == "route_example"
    assert benchmark_run["show_in_task_queue"] is False


def test_v4_workbench_pdf_download_uses_print_renderer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gateway = _gateway(tmp_path)
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()
    created = client.post(
        "/api/v4/runs",
        json={
            "run_id": "pdf-example",
            "target_name": "ethanol",
            "target_smiles": "CCO",
        },
    )
    assert created.status_code == 201
    captured: dict[str, dict] = {}

    def fake_pdf(snapshot: dict) -> bytes:
        captured["snapshot"] = snapshot
        return b"%PDF-1.7\nfixture"

    monkeypatch.setattr(
        "cascade_planner.web.v4_api.render_workbench_pdf",
        fake_pdf,
    )
    response = client.get("/api/v4/runs/pdf-example/workbench.pdf")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-1.7")
    assert "pdf-example-retrosynthesis-dossier.pdf" in response.headers[
        "Content-Disposition"
    ]
    assert captured["snapshot"]["run_id"] == "pdf-example"


def test_v4_history_delete_hides_queue_entry_but_preserves_run_directory(
    tmp_path: Path,
) -> None:
    gateway = _gateway(tmp_path)
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()
    created = client.post(
        "/api/v4/runs",
        json={
            "run_id": "history-example",
            "target_name": "ethanol",
            "target_smiles": "CCO",
        },
    ).get_json()
    run_dir = Path(created["run_dir"])
    assert run_dir.is_dir()

    removed = client.delete("/api/v4/runs/history-example/history")

    assert removed.status_code == 200
    assert removed.get_json()["removed"] is True
    assert removed.get_json()["scientific_artifacts_preserved"] is True
    assert run_dir.is_dir()
    assert all(
        row["run_id"] != "history-example"
        for row in client.get("/api/v4/runs").get_json()["runs"]
    )


def test_v4_workspace_route_and_queue_deletions_are_independent_and_recoverable(
    tmp_path: Path,
) -> None:
    gateway = _gateway(tmp_path)
    app = create_v4_app(lambda: gateway)
    client = app.test_client()
    created = client.post(
        "/api/v4/runs",
        json={
            "run_id": "visibility-example",
            "target_name": "ethanol",
            "target_smiles": "CCO",
        },
    ).get_json()
    run_dir = Path(created["run_dir"])

    route_removed = client.delete(
        "/api/v4/workspace/routes",
        json={"route_id": "run:visibility-example"},
    )
    queue_removed = client.delete(
        "/api/v4/jobs/solve%3Avisibility-example",
        json={},
    )

    assert route_removed.status_code == 200
    assert queue_removed.status_code == 200
    assert route_removed.get_json()["scientific_artifacts_preserved"] is True
    assert queue_removed.get_json()["scientific_artifacts_preserved"] is True
    assert run_dir.is_dir()
    workspace = client.get("/api/v4/workspace").get_json()
    assert "run:visibility-example" in workspace["workspace_visibility"][
        "hidden_route_ids"
    ]
    assert "visibility-example" in workspace["workspace_visibility"][
        "hidden_queue_run_ids"
    ]
    job = next(
        row
        for row in client.get("/api/v4/jobs").get_json()["jobs"]
        if row["run_id"] == "visibility-example"
    )
    assert job["show_in_route_catalog"] is False
    assert job["show_in_task_queue"] is False

    restored = client.post(
        "/api/v4/workspace/visibility/restore",
        json={"scope": "all"},
    )

    assert restored.status_code == 200
    assert restored.get_json()["restored_count"] == 2
    workspace = client.get("/api/v4/workspace").get_json()
    assert workspace["workspace_visibility"]["hidden_route_ids"] == []
    assert workspace["workspace_visibility"]["hidden_queue_run_ids"] == []

    invalid = client.delete("/api/v4/workspace/routes", json={})
    assert invalid.status_code == 400
    assert invalid.is_json
    assert invalid.get_json()["reason"] == "route_id_missing"


def test_v4_active_job_cannot_be_deleted_from_queue(tmp_path: Path, monkeypatch) -> None:
    gateway = _gateway(tmp_path)
    release = __import__("threading").Event()

    def blocked_job(*_args, **_kwargs):
        release.wait(timeout=5)

    monkeypatch.setattr("cascade_planner.web.v4_api._run_target_job", blocked_job)
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()
    job = client.post(
        "/api/v4/jobs",
        json={"target_name": "active", "target_smiles": "CCO"},
    ).get_json()

    removed = client.delete(f"/api/v4/jobs/{job['job_id']}", json={})
    release.set()

    assert removed.status_code == 409
    assert removed.get_json()["reason"] == "active_job_cannot_be_deleted"


def test_v4_workbench_html_uses_a_human_readable_integrity_fallback() -> None:
    class BrokenGateway:
        def workbench(self, _run_id):
            return {"snapshot": {"schema_version": "retrosynthesis_route_workbench.v1"}}

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(BrokenGateway))
    response = app.test_client().get("/api/v4/runs/broken/workbench.html")

    assert response.status_code == 422
    assert response.mimetype == "text/html"
    assert "工作台暂不可用" in response.get_data(as_text=True)
    assert "原始运行快照没有被修改" in response.get_data(as_text=True)
    assert "invalid_request" not in response.get_data(as_text=True)


def test_v4_http_exposes_read_only_route_program_innovation_review(
    tmp_path: Path,
    reported_ethanol_program_pack: dict,
) -> None:
    gateway = _gateway(tmp_path)
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()
    created = client.post(
        "/api/v4/runs",
        json={
            "run_id": "web-program-innovation",
            "target_name": "ethanol",
            "target_smiles": "CCO",
            "materialize": True,
            "global_plan": {
                "schema_version": "global_campaign_plan.v1",
                "route_families": [
                    {
                        "route_family_id": "family:reduction",
                        "strategic_disconnection": "carbonyl reduction",
                    }
                ],
                "multi_step_skeletons": [
                    {
                        "skeleton_id": "skeleton:reduction",
                        "route_family_id": "family:reduction",
                        "steps": [
                            {
                                "step_id": "step:reduction",
                                "product_smiles": "CCO",
                                "precursor_smiles": ["CC=O"],
                                "transformation_hypothesis": "carbonyl reduction",
                            }
                        ],
                    }
                ],
            },
        },
    )
    route_id = next(
        iter(
            client.get("/api/v4/runs/web-program-innovation/workbench").get_json()["snapshot"][
                "routes"
            ]
        )
    )
    response = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations",
        json={
            "route_id": route_id,
            "capabilities": [],
            "reported_candidate_packs": [reported_ethanol_program_pack],
        },
    )

    assert created.status_code == 201
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["operation"] == "route-program-innovations"
    assert payload["oracle"]["accepted"] is True
    assert payload["mechanism_oracle"]["accepted"] is True
    assert payload["execution_oracle"]["accepted"] is True
    assert payload["program_optimizer_oracle"]["accepted"] is True
    assert payload["program_route_candidates"]["counts"]["candidates"] == 2
    assert payload["program_route_candidates"]["counts"]["literature"] == 1
    assert payload["program_optimizer"]["profiles"]["exploration"]["pareto_front_ids"]
    assert payload["program_bundle"]["counts"]["program_proposals"] == 0
    assert payload["mechanism_program_bundle"]["counts"]["program_proposals"] == 0
    assert payload["mechanism_validation_frontier"]["counts"]["experiment_required"] == 0
    assert payload["mechanism_experiment_feedback"]["counts"]["feedback_records"] == 0
    assert payload["mechanism_feedback_oracle"]["accepted"] is True
    assert payload["execution_program_bundle"]["counts"]["program_proposals"] == 0
    assert payload["execution_validation_frontier"]["counts"]["experiment_required"] == 0
    assert payload["execution_capability_feedback"]["counts"]["feedback_records"] == 0
    assert payload["execution_feedback_oracle"]["accepted"] is True
    assert payload["experimental_claims"]["counts"]["claims"] == 0
    assert payload["experimental_claims_oracle"]["accepted"] is True
    assert payload["capability_calibration"]["counts"]["calibrations"] == 0
    assert payload["capability_calibration_oracle"]["accepted"] is True
    assert payload["semantics"]["canonical_graph_not_mutated"] is True
    assert payload["semantics"]["execution_programs_have_no_store_admission_path"] is True
    assert payload["semantics"]["feedback_does_not_mutate_or_disable_capability_catalog"] is True

    capability = _web_reduction_capability()
    positive = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations",
        json={"route_id": route_id, "capabilities": [capability]},
    ).get_json()
    proposal = next(iter(positive["program_bundle"]["program_proposals"].values()))
    assert positive["experimental_work_frontier_oracle"]["accepted"] is True
    request = next(iter(positive["experimental_work_frontier"]["work_items"].values()))[
        "execution_request"
    ]
    executor_result = build_experiment_execution_result(
        request,
        result_id="experiment-result:web-reduction",
        executor_id="web-fixture-lab",
        executor_version="1",
        status="success",
        artifact_refs=[
            {"sha256": "e" * 64, "media_type": "application/json", "role": "raw_record"}
        ],
        domain_validation_candidate=_web_biocatalysis_validation(proposal),
    )
    audited_result = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/audit",
        json={
            "route_id": route_id,
            "capabilities": [capability],
            "result": executor_result,
        },
    )
    assert audited_result.status_code == 200
    assert audited_result.get_json()["result_audit"]["accepted_for_domain_gate"] is True

    staged = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/artifacts/json",
        json={
            "artifact": {"conversion_fraction": 0.88},
            "logical_name": "web-experiment-record.json",
            "enable_experiment_artifact_staging": True,
        },
    )
    policy = {
        "schema_version": "experiment_executor_policy.v1",
        "enabled": True,
        "allowed_provider_ids": ["autoplanner.manual_experiment_executor"],
        "preferred_provider_ids": ["autoplanner.manual_experiment_executor"],
        "allowed_domains": ["biocatalytic"],
        "allow_network_access": False,
        "max_estimated_cost_units": 0,
    }
    dispatch_payload = {
        "route_id": route_id,
        "capabilities": [capability],
        "request_id": request["request_id"],
        "provider_policy": policy,
        "enable_experiment_dispatch": True,
    }
    disabled_dispatch = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/dispatch",
        json={**dispatch_payload, "enable_experiment_dispatch": False},
    )
    dispatched = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/dispatch",
        json=dispatch_payload,
    )
    assert staged.status_code == 201
    assert disabled_dispatch.status_code == 400
    assert dispatched.status_code == 200
    dispatch = dispatched.get_json()["dispatch"]
    operator = build_experiment_operator_identity(
        principal_id="operator:web-fixture", principal_type="service",
        authentication_context_sha256="d" * 64,
    )
    running_job = build_experiment_external_job_receipt(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"],
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id="external-job:web", provider_sequence=1,
        status="running", recorded_by=operator,
    )
    disabled_job = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/job",
        json={
            "route_id": route_id, "capabilities": [capability],
            "dispatch_id": dispatch["dispatch_id"], "job_receipt": running_job,
            "enable_experiment_job_receipt": False,
        },
    )
    recorded_job = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/job",
        json={
            "route_id": route_id, "capabilities": [capability],
            "dispatch_id": dispatch["dispatch_id"], "job_receipt": running_job,
            "enable_experiment_job_receipt": True,
        },
    )
    assert disabled_job.status_code == 400
    assert recorded_job.status_code == 200
    assert recorded_job.get_json()["dispatch"] == running_job
    cancellation = build_experiment_cancellation_request(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"],
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id=running_job["external_job_id"],
        current_external_job_receipt_sha256=running_job["content_sha256"],
        requested_by=operator, reason_code="web-race-test",
    )
    cancellation_response = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/cancel",
        json={
            "route_id": route_id, "capabilities": [capability],
            "dispatch_id": dispatch["dispatch_id"],
            "cancellation_request": cancellation,
            "enable_experiment_cancellation": True,
        },
    )
    assert cancellation_response.status_code == 200
    assert cancellation_response.get_json()["dispatch"] == cancellation
    completed_job = build_experiment_external_job_receipt(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"],
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id=running_job["external_job_id"], provider_sequence=2,
        status="completed", predecessor_receipt_sha256=running_job["content_sha256"],
        cancellation_request_sha256=cancellation["content_sha256"],
        recorded_by=operator,
    )
    completed_response = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/job",
        json={
            "route_id": route_id, "capabilities": [capability],
            "dispatch_id": dispatch["dispatch_id"],
            "job_receipt": completed_job,
            "enable_experiment_job_receipt": True,
        },
    )
    assert completed_response.status_code == 200
    assert completed_response.get_json()["dispatch"] == completed_job
    manual_result = build_experiment_execution_result(
        request,
        result_id="experiment-result:web-manual-dispatch",
        executor_id="autoplanner.manual_experiment_executor",
        executor_version="1.0.0",
        status="success",
        artifact_refs=[
            {
                "sha256": staged.get_json()["artifact"]["sha256"],
                "media_type": "application/json",
                "role": "raw_record",
            }
        ],
        domain_validation_candidate=_web_biocatalysis_validation(proposal),
    )
    settled = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experiments/settle",
        json={
            "route_id": route_id,
            "capabilities": [capability],
            "dispatch_id": dispatch["dispatch_id"],
            "result": manual_result,
            "enable_experiment_settlement": True,
        },
    )
    assert settled.status_code == 200
    assert settled.get_json()["dispatch"]["status"] == "settled"
    assert settled.get_json()["dispatch"]["next_boundary"] == (
        "submit_candidate_to_existing_domain_validation_gate"
    )
    admission_payload = {
        "route_id": route_id,
        "capabilities": [capability],
        "validations": [_web_biocatalysis_validation(proposal)],
    }
    review_only_rejected = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/admit",
        json={
            **admission_payload,
            "reported_candidate_packs": [reported_ethanol_program_pack],
            "enable_biocatalytic_program_admission": True,
        },
    )
    assert review_only_rejected.status_code == 400
    assert review_only_rejected.get_json()["reason"] == "reported_candidate_packs_are_review_only"
    validated_review = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations",
        json=admission_payload,
    ).get_json()
    validated_front = validated_review["program_optimizer"]["profiles"]["shadow_optimizer"][
        "pareto_front_ids"
    ]
    assert validated_review["program_optimizer_oracle"]["accepted"] is True
    assert validated_review["experimental_claims"]["counts"]["biocatalytic"] == 1
    assert validated_review["experimental_claims_oracle"]["accepted"] is True
    assert any(
        validated_review["program_route_candidates"]["candidates"][candidate_id]["source_kind"]
        == "biocatalytic"
        for candidate_id in validated_front
    )
    graph_revision = client.get("/api/v4/runs/web-program-innovation/status").get_json()["status"][
        "graph_revision"
    ]
    disabled = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/admit",
        json=admission_payload,
    )
    empty_store = client.get("/api/v4/runs/web-program-innovation/programs/innovations/store")
    admitted = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/admit",
        json={
            **admission_payload,
            "enable_biocatalytic_program_admission": True,
        },
    )
    repeated = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/admit",
        json={
            **admission_payload,
            "enable_biocatalytic_program_admission": True,
        },
    )
    durable = client.get("/api/v4/runs/web-program-innovation/programs/innovations/store")
    claim_disabled = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/claims/admit",
        json=admission_payload,
    )
    empty_claim_store = client.get(
        "/api/v4/runs/web-program-innovation/programs/innovations/claims/store"
    )
    claim_admitted = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/claims/admit",
        json={
            **admission_payload,
            "enable_experimental_claim_admission": True,
        },
    )
    claim_repeated = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/claims/admit",
        json={
            **admission_payload,
            "enable_experimental_claim_admission": True,
        },
    )
    durable_claims = client.get(
        "/api/v4/runs/web-program-innovation/programs/innovations/claims/store"
    )
    empty_experience = client.get(
        "/api/v4/runs/web-program-innovation/programs/innovations/experience"
    )
    experience_disabled = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experience/learn",
        json={},
    )
    experience_learned = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experience/learn",
        json={"enable_program_experience_learning": True},
    )
    experience_repeated = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations/experience/learn",
        json={"enable_program_experience_learning": True},
    )
    durable_experience = client.get(
        "/api/v4/runs/web-program-innovation/programs/innovations/experience"
    )
    memory_review = client.post(
        "/api/v4/runs/web-program-innovation/programs/innovations",
        json=admission_payload,
    ).get_json()

    assert disabled.status_code == 409
    assert empty_store.get_json()["replay"]["event_count"] == 0
    assert admitted.status_code == 201
    assert admitted.get_json()["created"] is True
    assert repeated.status_code == 200
    assert repeated.get_json()["created"] is False
    assert durable.get_json()["replay"]["event_count"] == 1
    assert claim_disabled.status_code == 409
    assert empty_claim_store.get_json()["replay"]["event_count"] == 0
    assert claim_admitted.status_code == 201
    assert claim_admitted.get_json()["event"]["counts"]["claims"] == 1
    assert claim_repeated.status_code == 200
    assert durable_claims.get_json()["replay"]["event_count"] == 1
    assert empty_experience.get_json()["library"]["experiences"] == {}
    assert experience_disabled.status_code == 409
    assert experience_learned.get_json()["new_claim_count"] == 1
    assert experience_repeated.get_json()["new_claim_count"] == 0
    assert len(durable_experience.get_json()["library"]["experiences"]) == 1
    assert memory_review["program_experience"]["matched_candidate_count"] == 1
    memory_candidate = next(iter(memory_review["program_bundle"]["program_proposals"].values()))
    assert "EXACT_SUBSTRATE_UNVALIDATED" in memory_candidate["warning_codes"]
    assert (
        client.get("/api/v4/runs/web-program-innovation/status").get_json()["status"][
            "graph_revision"
        ]
        == graph_revision
    )


def test_isolated_v4_app_redirects_root_and_keeps_shared_security_guards() -> None:
    app = create_v4_app(lambda: None)
    client = app.test_client()

    root = client.get("/")
    rejected = client.post("/api/v4/runs", data="{}", content_type="text/plain")
    response = client.get("/v4")

    assert root.status_code == 302
    assert root.headers["Location"].endswith("/v4")
    assert rejected.status_code == 415
    assert response.headers["X-Content-Type-Options"] == "nosniff"


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


def test_v4_program_admission_is_explicit_idempotent_and_shadow_only(
    tmp_path: Path,
) -> None:
    gateway = _gateway(tmp_path)
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()
    client.post(
        "/api/v4/runs",
        json={
            "run_id": "web-program-store",
            "target_name": "ethanol",
            "target_smiles": "CCO",
        },
    )

    empty = client.get("/api/v4/runs/web-program-store/programs/store")
    validation_before = client.get("/api/v4/runs/web-program-store/validate").get_json()
    rejected = client.post("/api/v4/runs/web-program-store/programs/admit", json={})
    admitted = client.post(
        "/api/v4/runs/web-program-store/programs/admit",
        json={"enable_program_admission": True},
    )
    repeated = client.post(
        "/api/v4/runs/web-program-store/programs/admit",
        json={"enable_program_admission": True},
    )
    durable = client.get("/api/v4/runs/web-program-store/programs/store")
    validated = client.get("/api/v4/runs/web-program-store/validate")

    assert empty.status_code == 200
    assert empty.get_json()["status"]["event_count"] == 0
    assert rejected.status_code == 409
    assert rejected.get_json()["reason"].endswith("explicit_enable_required")
    assert admitted.status_code == 201
    assert admitted.get_json()["created"] is True
    assert repeated.status_code == 200
    assert repeated.get_json()["created"] is False
    assert durable.get_json()["status"]["oracle"]["accepted"] is True
    assert (
        durable.get_json()["status"]["semantics"]["edge_ids_remain_production_route_authority"]
        is True
    )
    validation_after = validated.get_json()
    assert validation_after["accepted"] is validation_before["accepted"]
    assert validation_after["checks"] == validation_before["checks"]
    assert validation_after["graph_revision"] == validation_before["graph_revision"]
    assert (
        validation_after["graph_scientific_sha256"] == validation_before["graph_scientific_sha256"]
    )


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
            "forbidden_reagents": ["benzene"],
            "max_route_steps": 7,
            "allowed_execution_domains": ["chemical", "hybrid"],
            "safety_limits": {"max_temperature_c": 100},
            "stock_source_ids": ["test-stock"],
            "max_total_tasks": 90,
            "max_evidence_tasks": 21,
            "max_stock_tasks": 22,
            "max_validation_tasks": 23,
            "max_program_tasks": 11,
            "max_experiment_tasks": 4,
            "max_run_wall_time_s": 1800,
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
    assert config.max_total_tasks == 90
    assert config.max_evidence_tasks == 21
    assert config.max_stock_tasks == 22
    assert config.max_validation_tasks == 23
    assert config.max_program_tasks == 11
    assert config.max_experiment_tasks == 4
    assert config.max_run_wall_time_s == 1800
    assert captured["budget"].max_model_invocations == 1
    constraints = captured["constraints"]
    assert constraints.forbidden_reagents == ("benzene",)
    assert constraints.max_route_steps == 7
    assert constraints.allowed_execution_domains == ("chemical", "hybrid")
    assert constraints.safety_limits["max_temperature_c"] == 100
    assert constraints.stock_source_ids == ("test-stock",)


def test_v4_proof_profile_keeps_depth_but_uses_a_returnable_search_width() -> None:
    captured: dict = {}

    class RecordingGateway:
        def solve_target(self, **kwargs):
            captured.update(kwargs)
            return {"schema_version": "fixture", "run_id": "proof-budget"}

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(RecordingGateway))
    response = app.test_client().post(
        "/api/v4/solve-target",
        json={
            "run_id": "proof-budget",
            "target_name": "proof target",
            "target_smiles": "CCO",
            "execution_profile": "proof",
        },
    )

    assert response.status_code == 201
    config = captured["config"]
    assert config.max_chemenzy_steps == 20
    assert config.max_chemenzy_iterations == 60
    assert config.chemenzy_expansion_topk == 120
    assert config.chemenzy_timeout_s == 3600.0
    assert config.chemenzy_pandarallel_workers == 8
    assert config.max_director_wall_time_s == 1800.0
    budget = captured["budget"]
    assert budget.max_total_input_tokens == 1_200_000
    assert budget.max_total_output_tokens == 200_000
    assert budget.max_total_wall_time_s == 1800.0


def test_v4_async_job_separates_execution_end_from_scientific_acceptance() -> None:
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
                "resource_envelope": {
                    "within_budget": True,
                    "observed": {"run_wall_time_s": 1.5},
                    "task_budget": {
                        "schema_version": "campaign_task_budget.v1",
                        "dimensions": {"program": {"limit": 7}},
                    },
                    "violations": [],
                },
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
        if value["status"] == "unresolved":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("async V4 job did not complete")

    assert value["result"]["highest_contiguous_gate"] == "B1"
    assert value["result"]["model_cost"]["model_invocations"] == 1
    assert value["result"]["resource_envelope"]["observed"] == {
        "run_wall_time_s": 1.5
    }
    assert value["result"]["resource_envelope"]["task_budget"]["dimensions"][
        "program"
    ] == {"limit": 7}
    assert value["progress"]["delivery"]["proof_closure_complete"] is False


def test_v4_async_job_automatically_resumes_a_bounded_unaccepted_pass() -> None:
    class RecordingGateway:
        calls: list[bool] = []

        def solve_target(self, **kwargs):
            self.calls.append(bool(kwargs["resume"]))
            accepted = len(self.calls) == 2
            return {
                "run_id": kwargs["run_id"],
                "gates": {},
                "claim": {"accepted_under_configured_policy": accepted},
                "model_cost": {"model_invocations": len(self.calls)},
                "stop_decision": {
                    "decision": "completed" if accepted else "paused",
                    "terminal": accepted,
                },
            }

        def status(self, _run_id):
            raise RuntimeError("fixture has no persistent kernel")

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(RecordingGateway))
    client = app.test_client()
    started = client.post(
        "/api/v4/jobs",
        json={"target_name": "auto continuation", "target_smiles": "CCO"},
    ).get_json()

    for _ in range(100):
        value = client.get(f"/api/v4/jobs/{started['job_id']}").get_json()
        if value["status"] == "complete":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("async V4 job did not automatically continue")

    assert RecordingGateway.calls == [False, True]
    assert value["continuation_pass_count"] == 1
    assert value["result"]["accepted"] is True


def test_v4_job_list_projects_the_live_checkpoint_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "active-run"
    checkpoint = run_dir / ".autoplanner" / "target-solver-checkpoint.json"
    release = Event()

    class LiveGateway:
        def solve_target(self, **kwargs):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps(
                    {
                        "stages": [
                            {
                                "stage": "campaign_action_unified_core_01",
                                "status": "completed",
                                "detail": {
                                    "action": {
                                        "execution_id": "campaign-action:chemenzy",
                                        "action_id": "action:chemenzy_target_expand:1",
                                        "kind": "chemenzy_target_expand",
                                    },
                                    "outcome": {"status": "completed"},
                                },
                            },
                            {"stage": "chemenzy_baseline", "status": "running"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            release.wait(2)
            return {"run_id": kwargs["run_id"], "gates": {}, "claim": {}}

        def status(self, _run_id):
            return {
                "run_dir": str(run_dir),
                "status": {
                    "status": "running",
                    "active_actions": [
                        {
                            "execution_id": "campaign-action:codex",
                            "action_id": "action:codex_global_architecture:1",
                            "kind": "codex_global_architecture",
                            "producer": "codex_global_director",
                            "resource_class": "model",
                        }
                    ],
                    "portfolio": {},
                    "frontier": [],
                    "stop_decision": {},
                },
            }

        def list_runs(self, **_kwargs):
            return {"runs": []}

    gateway = LiveGateway()
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))
    client = app.test_client()
    started = client.post(
        "/api/v4/jobs",
        json={"target_name": "live", "target_smiles": "CCO"},
    )
    assert started.status_code == 202
    for _ in range(100):
        if checkpoint.is_file():
            break
        time.sleep(0.01)
    else:
        release.set()
        raise AssertionError("live checkpoint was not written")

    listed = client.get("/api/v4/jobs").get_json()["jobs"]
    release.set()

    assert listed[0]["phase"] == "chemenzy_baseline"
    assert listed[0]["progress"]["stages"][-1]["status"] == "running"
    timeline = listed[0]["progress"]["action_timeline"]
    assert timeline["record_count"] == 2
    assert timeline["actor_counts"] == {"ChemEnzy": 1, "Codex": 1}
    assert timeline["state_counts"] == {"running": 1, "succeeded": 1}


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


def test_v4_delivery_projection_distinguishes_structure_binding_from_proof() -> None:
    delivery = delivery_projection(
        [
            {"stage": "initial_workbench", "status": "completed"},
            {
                "stage": "evidence_acquisition",
                "status": "structure_bound_unproven",
            },
        ],
        job_status="running",
    )

    assert delivery["state"] == "proof_review_ready"
    assert delivery["evidence_stage_complete"] is True
    assert delivery["proof_closure_complete"] is False


def test_v4_delivery_projection_marks_finished_unaccepted_job_unresolved() -> None:
    delivery = delivery_projection(
        [{"stage": "initial_workbench", "status": "completed"}],
        job_status="unresolved",
    )

    assert delivery["state"] == "unresolved"
    assert delivery["proof_closure_known"] is True
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
    assert job["progress"]["scientific_status"] == "accepted"
    delivery = job["progress"]["delivery"]
    assert delivery["state"] == "historical"
    assert delivery["route_candidates_available"] is True
    assert delivery["proof_closure_known"] is False
    assert delivery["proof_closure_complete"] is False
    assert delivery["semantics"]["portfolio_policy_accepted"] is True
