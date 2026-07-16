from __future__ import annotations

from pathlib import Path
import time

from flask import Flask

from cascade_planner.application.biocatalytic_programs import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    with_biocatalysis_program_validation_digest,
)
from cascade_planner.application.experiment_execution_results import (
    build_experiment_execution_result,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.web.v4_api import create_v4_blueprint
from cascade_planner.web.v4_app import create_v4_app
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
    programs = client.get("/api/v4/runs/web-example/programs")
    program_routes = client.get("/api/v4/runs/web-example/programs/routes")
    program_audit = client.get("/api/v4/program-migration", query_string={"run_id": "web-example"})
    rendered = client.get("/api/v4/runs/web-example/workbench.html")
    index = client.get("/v4")
    console = client.get("/v4/console")
    showcase = client.get("/v4/showcase")
    workspace = client.get("/api/v4/workspace")

    assert status.status_code == 200
    assert workbench.status_code == 200
    assert rendered.status_code == 200
    assert index.status_code == 200
    assert console.status_code == 200
    assert showcase.status_code == 200
    assert workspace.status_code == 200
    assert workspace.get_json()["backend"]["available"] is True
    assert workspace.get_json()["runs"][0]["run_id"] == "web-example"
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
    assert "统一工作区" in index.get_data(as_text=True)
    assert "路线候选已经可审查" in index.get_data(as_text=True)
    assert "/api/v4/workspace" in index.get_data(as_text=True)
    assert "V4 运行控制台" in console.get_data(as_text=True)
    assert "只看每个目标最新结果" in console.get_data(as_text=True)
    assert b"<!doctype html>" in rendered.data.lower()


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
    request = next(
        iter(positive["experimental_work_frontier"]["work_items"].values())
    )["execution_request"]
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
