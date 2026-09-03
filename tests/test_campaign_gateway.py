from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from cascade_planner.application.biocatalytic_programs import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    with_biocatalysis_program_validation_digest,
)
from cascade_planner.application.experiment_execution_results import (
    build_experiment_execution_result,
    with_experiment_execution_result_digest,
)
from cascade_planner.application.experiment_external_jobs import (
    build_experiment_cancellation_request,
    build_experiment_external_job_receipt,
    build_experiment_operator_identity,
)
from cascade_planner.interfaces.campaign_gateway import (
    CampaignGateway,
    CampaignGatewayError,
)
from cascade_planner.providers.builtins import build_default_provider_registry
from cascade_planner.providers.http_experiment import (
    HttpExperimentExecutorConfig,
    HttpExperimentExecutorProvider,
)
from cascade_planner.orchestration import experiment_job_transport_runtime
from cascade_planner.runtime.paths import RuntimePaths


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir()
    runtime = tmp_path / "runtime"
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(runtime),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index" / "runs.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(tmp_path / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(tmp_path / "vendor"),
        },
    )


def _plan() -> dict:
    return {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:ester",
                "strategic_disconnection": "late acyl substitution",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:ester",
                "route_family_id": "family:ester",
                "steps": [
                    {
                        "step_id": "step:ester",
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)Cl"],
                        "transformation_hypothesis": "acyl substitution",
                    }
                ],
            }
        ],
    }


def _reduction_plan() -> dict:
    return {
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
    }


def _reduction_capability() -> dict:
    return {
        "capability_id": "fixture:aliphatic-carbonyl-reduction",
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
        "substrate_scope_basis": "generic fixture analog",
        "precedent_refs": ["doi:10.1000/reduction-fixture"],
    }


def _biocatalysis_validation(proposal: dict) -> dict:
    return with_biocatalysis_program_validation_digest(
        {
            "schema_version": BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
            "validation_id": "validation:gateway-reduction",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "accepted": True,
            "evidence_tier": "exact_substrate_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "claim_refs": ["claim:gateway-exact-substrate-screen"],
            "condition_record_ids": [],
            "selectivity_assessed": True,
            "cofactor_ledger_closed": True,
            "outcome": {"conversion_fraction": 0.91},
        }
    )


def _manual_experiment_policy() -> dict:
    return {
        "schema_version": "experiment_executor_policy.v1",
        "enabled": True,
        "allowed_provider_ids": ["autoplanner.manual_experiment_executor"],
        "preferred_provider_ids": ["autoplanner.manual_experiment_executor"],
        "allowed_domains": ["biocatalytic"],
        "allow_network_access": False,
        "max_estimated_cost_units": 0,
    }


def _operator_identity() -> dict:
    return build_experiment_operator_identity(
        principal_id="operator:gateway-fixture", principal_type="human",
        authentication_context_sha256="f" * 64,
    )


def _prepare_dispatched_experiment(
    gateway: CampaignGateway, run_id: str
) -> tuple[str, dict, dict, dict, dict]:
    gateway.create_run(
        run_id=run_id, target_name="ethanol", target_smiles="CCO",
        global_plan=_reduction_plan(), materialize=True,
    )
    route_id = next(iter(gateway.workbench(run_id)["snapshot"]["routes"]))
    capability = _reduction_capability()
    review = gateway.route_program_innovations(
        run_id, route_id=route_id, capabilities=[capability]
    )
    proposal = next(iter(review["program_bundle"]["program_proposals"].values()))
    request = next(iter(review["experimental_work_frontier"]["work_items"].values()))[
        "execution_request"
    ]
    dispatch = gateway.dispatch_route_experiment(
        run_id, route_id=route_id, capabilities=[capability],
        request_id=request["request_id"], policy=_manual_experiment_policy(),
        enable_experiment_dispatch=True,
    )["dispatch"]
    return route_id, capability, proposal, request, dispatch


def _external_job_receipt(
    dispatch: dict, request: dict, *, sequence: int, status: str,
    predecessor: str = "", cancellation: str = "",
) -> dict:
    return build_experiment_external_job_receipt(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"],
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id="external-job:gateway-fixture", provider_sequence=sequence,
        status=status, predecessor_receipt_sha256=predecessor,
        cancellation_request_sha256=cancellation, recorded_by=_operator_identity(),
    )


class _TransportResponse:
    def __init__(self, status: int, value: dict) -> None:
        self.status_code = status
        self._body = json.dumps(value).encode("utf-8")

    def iter_content(self, *, chunk_size: int):
        del chunk_size
        yield self._body

    def close(self) -> None:
        pass


def _http_gateway(
    tmp_path: Path, responses: list[dict | Exception]
) -> tuple[CampaignGateway, list[dict]]:
    calls: list[dict] = []

    def requester(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return _TransportResponse(200, value)

    provider = HttpExperimentExecutorProvider(
        HttpExperimentExecutorConfig(
            provider_id="fixture.gateway-http-experiment", version="1.0.0",
            base_url="https://lab.example.invalid",
            auth_token_env="FIXTURE_GATEWAY_LAB_TOKEN",
            operator_principal_id="service:gateway-http-fixture",
        ),
        environ={"FIXTURE_GATEWAY_LAB_TOKEN": "gateway-secret"},
        requester=requester,
    )
    registry = build_default_provider_registry(
        include_manual_experiment_executor=True,
        include_http_experiment_executor=provider,
    )
    return CampaignGateway(_paths(tmp_path), provider_registry=registry), calls


def _http_policy() -> dict:
    return {
        "schema_version": "experiment_executor_policy.v1", "enabled": True,
        "allowed_provider_ids": ["fixture.gateway-http-experiment"],
        "preferred_provider_ids": ["fixture.gateway-http-experiment"],
        "allowed_domains": ["biocatalytic"], "allow_network_access": True,
        "max_estimated_cost_units": 0,
    }


def _prepare_http_dispatch(
    gateway: CampaignGateway, run_id: str
) -> tuple[str, dict, dict, dict, dict]:
    gateway.create_run(
        run_id=run_id, target_name="ethanol", target_smiles="CCO",
        global_plan=_reduction_plan(), materialize=True,
    )
    route_id = next(iter(gateway.workbench(run_id)["snapshot"]["routes"]))
    capability = _reduction_capability()
    review = gateway.route_program_innovations(
        run_id, route_id=route_id, capabilities=[capability]
    )
    proposal = next(iter(review["program_bundle"]["program_proposals"].values()))
    request = next(iter(review["experimental_work_frontier"]["work_items"].values()))[
        "execution_request"
    ]
    dispatch = gateway.dispatch_route_experiment(
        run_id, route_id=route_id, capabilities=[capability],
        request_id=request["request_id"], policy=_http_policy(),
        enable_experiment_dispatch=True,
    )["dispatch"]
    return route_id, capability, proposal, request, dispatch


def test_gateway_runs_every_operator_operation_without_model_calls(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    gateway = CampaignGateway(paths)

    created = gateway.create_run(
        run_id="gateway-example",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
        global_plan=_plan(),
        materialize=True,
    )

    assert created["status"]["graph_revision"] == 2
    assert created["status"]["model_totals"]["model_invocations"] == 0
    assert created["status"]["campaign_spec"]["target"] == {
        "canonical_smiles": "CCOC(C)=O"
    }
    assert set(created["status"]["quality_state"]["axes"]) == {
        "topology",
        "reaction_validation",
        "exact_evidence",
        "stock",
        "conditions",
        "procurement",
        "program_validation",
        "diversity",
    }
    assert Path(created["run_dir"]).is_relative_to(paths.runs_root)
    assert paths.artifact_store_root.is_dir()
    assert paths.run_index_path.is_file()
    assert not (paths.runtime_root / "artifacts").exists()

    assert gateway.status("gateway-example")["status"]["graph_revision"] == 2
    assert gateway.validate("gateway-example")["accepted"] is True
    assert gateway.replay("gateway-example")["accepted"] is True
    benchmark = gateway.benchmark("gateway-example", iterations=1)
    assert benchmark["model_invocations"] == 0
    assert benchmark["semantics"]["network_free"] is True

    exported = gateway.export(
        "gateway-example",
        output_dir=tmp_path / "export",
    )
    for path in exported["files"].values():
        assert Path(path).is_file()
    snapshot = json.loads(Path(exported["files"]["snapshot"]).read_text("utf-8"))
    assert snapshot["schema_version"] == "retrosynthesis_route_workbench.v1"
    review_bundle = json.loads(
        Path(exported["files"]["review_bundle"]).read_text("utf-8")
    )
    assert review_bundle["schema_version"] == "campaign_review_bundle.v1"
    assert review_bundle["available"] is False
    assert review_bundle["unavailable_reason"] == "target_solve_report_missing"
    assert exported["review_bundle_sha256"] == review_bundle["content_sha256"]
    for name in ("action_trace", "failure_trace", "route_lineage", "resource_curve"):
        component = json.loads(Path(exported["files"][name]).read_text("utf-8"))
        assert component == review_bundle["components"][name]
        assert component["content_sha256"] == review_bundle["component_sha256"][name]

    gc = gateway.gc_plan(minimum_age_s=0)
    assert gc["dry_run"] is True
    assert gc["plan"]["dry_run"] is True
    assert gc["indexed_artifact_pin_count"] > 0
    assert gateway.list_runs()["run_count"] == 1


def test_gateway_applies_a_later_global_plan_through_same_graph(tmp_path: Path) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        run_id="later-plan",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
    )

    applied = gateway.apply_plan(
        "later-plan",
        _plan(),
        materialize=True,
    )

    assert applied["operation"] == "apply-plan"
    assert applied["status"]["graph_revision"] == 2
    assert applied["status"]["accepted_expansion_count"] == 1
    assert applied["status"]["model_totals"]["model_invocations"] == 0


def test_gateway_status_projects_only_active_action_wrapper_reservations(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        run_id="active-action-status",
        target_name="ethanol",
        target_smiles="CCO",
    )
    service = gateway._open("active-action-status")
    service.kernel.reserve_task(
        task_id="campaign-action:fixture",
        kind="other",
        idempotency_key="active-action:reserve",
        input_revision=0,
        metadata={
            "campaign_action_id": "action:program_review:fixture",
            "campaign_action_execution_id": "campaign-action:fixture",
            "campaign_action_kind": "program_review",
            "producer": "program_host",
        },
    )
    service.kernel.reserve_task(
        task_id="program-child",
        kind="program",
        idempotency_key="active-action:child:reserve",
        input_revision=0,
        metadata={
            "campaign_action_execution_id": "campaign-action:fixture",
        },
    )

    active = gateway.status("active-action-status")["status"]["active_actions"]

    assert len(active) == 1
    assert active[0]["execution_id"] == "campaign-action:fixture"
    assert active[0]["kind"] == "program_review"
    assert active[0]["semantics"]["not_a_second_queue"] is True

    cancelled = gateway.cancel(
        "active-action-status",
        reasons=("operator_requested",),
        idempotency_key="gateway-test:cancel",
    )
    repeated = gateway.cancel(
        "active-action-status",
        reasons=("operator_requested",),
        idempotency_key="gateway-test:cancel",
    )

    assert cancelled["operation"] == "cancel"
    assert cancelled["status"]["status"] == "cancelled"
    assert cancelled["status"]["active_actions"] == []
    assert repeated["operations"]["cancellation"]["event_id"] == cancelled[
        "operations"
    ]["cancellation"]["event_id"]
    assert service.kernel.task_lifecycle("campaign-action:fixture")["status"] == (
        "interrupted"
    )


def test_gateway_program_store_requires_explicit_enablement_and_preserves_graph(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    created = gateway.create_run(
        run_id="program-gateway",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
        global_plan=_plan(),
        materialize=True,
    )
    run_dir = Path(created["run_dir"])
    graph_before = gateway.status("program-gateway")["status"]["graph_revision"]
    projection = gateway.program_projection("program-gateway")
    dual_read = gateway.route_program_dual_read("program-gateway")
    empty_store = gateway.program_store("program-gateway")

    assert projection["projection"]["counts"]["programs"] == 1
    assert dual_read["oracle"]["accepted"] is True
    assert dual_read["overlay"]["counts"]["displayed_routes"] == 1
    assert dual_read["overlay"]["counts"]["physical_step_count_mismatches"] == 0
    assert empty_store["status"]["event_count"] == 0
    assert not (run_dir / ".autoplanner" / "program_store").exists()
    with pytest.raises(CampaignGatewayError, match="explicit_enable_required"):
        gateway.admit_programs("program-gateway")
    assert not (run_dir / ".autoplanner" / "program_store").exists()

    first = gateway.admit_programs("program-gateway", enable_program_admission=True)
    second = gateway.admit_programs("program-gateway", enable_program_admission=True)
    durable = gateway.program_store("program-gateway")

    assert first["created"] is True
    assert second["created"] is False
    assert durable["status"]["event_count"] == 1
    assert durable["status"]["oracle"]["accepted"] is True
    assert durable["replay"]["event_count"] == 1
    indexed_scopes = {
        row["authority_scope"] for row in gateway.index.artifacts_for_run("program-gateway")
    }
    projection_sha256 = first["event"]["projection_ref"]["sha256"]
    gc_candidates = {
        row["sha256"] for row in gateway.gc_plan(minimum_age_s=0)["plan"]["candidates"]
    }
    assert "shadow_program_admission_source_graph" in indexed_scopes
    assert "shadow_program_admission_projection" in indexed_scopes
    assert projection_sha256 not in gc_candidates
    assert gateway.status("program-gateway")["status"]["graph_revision"] == graph_before
    assert gateway.validate("program-gateway")["accepted"] is True


def test_gateway_compiles_route_enzyme_candidate_as_read_only_program(
    tmp_path: Path,
    reported_ethanol_program_pack: dict,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        run_id="program-innovation-gateway",
        target_name="ethanol",
        target_smiles="CCO",
        global_plan=_reduction_plan(),
        materialize=True,
    )
    workbench = gateway.workbench("program-innovation-gateway")["snapshot"]
    route_id = next(iter(workbench["routes"]))
    graph_revision = gateway.status("program-innovation-gateway")["status"]["graph_revision"]

    result = gateway.route_program_innovations(
        "program-innovation-gateway",
        route_id=route_id,
        capabilities=[_reduction_capability()],
        reported_candidate_packs=[reported_ethanol_program_pack],
    )

    assert result["operation"] == "route-program-innovations"
    assert result["oracle"]["accepted"] is True
    assert result["mechanism_oracle"]["accepted"] is True
    assert result["execution_oracle"]["accepted"] is True
    assert result["program_bundle"]["counts"]["program_proposals"] == 1
    assert result["mechanism_program_bundle"]["counts"]["program_proposals"] == 0
    assert result["mechanism_validation_frontier"]["counts"]["experiment_required"] == 0
    assert result["mechanism_experiment_feedback"]["counts"]["feedback_records"] == 0
    assert result["mechanism_feedback_oracle"]["accepted"] is True
    assert result["execution_program_bundle"]["counts"]["program_proposals"] == 0
    assert result["execution_validation_frontier"]["counts"]["experiment_required"] == 0
    assert result["execution_capability_feedback"]["counts"]["feedback_records"] == 0
    assert result["execution_feedback_oracle"]["accepted"] is True
    assert result["experimental_claims"]["counts"]["claims"] == 0
    assert result["experimental_claims_oracle"]["accepted"] is True
    assert result["capability_calibration"]["counts"]["calibrations"] == 0
    assert result["capability_calibration_oracle"]["accepted"] is True
    assert result["experimental_work_frontier_oracle"]["accepted"] is True
    assert result["experimental_work_frontier"]["counts"]["work_items"] == 1
    work_item = next(
        iter(result["experimental_work_frontier"]["work_items"].values())
    )
    assert work_item["eligible_for_kernel_publication"] is False
    assert work_item["linked_canonical_deficit_ids"]
    assert result["program_optimizer_oracle"]["accepted"] is True
    assert result["program_route_candidates"]["counts"]["candidates"] == 3
    assert result["program_route_candidates"]["counts"]["literature"] == 1
    assert result["program_optimizer"]["counts"]["exploration_pareto_front"] == 2
    assert result["program_optimizer"]["counts"]["shadow_optimizer_pareto_front"] == 1
    proposal = next(iter(result["program_bundle"]["program_proposals"].values()))
    reported = next(
        row
        for row in result["program_route_candidates"]["candidates"].values()
        if row["source_kind"] == "literature"
    )
    assert reported["eligibility"]["exploration_visible"] is True
    assert reported["eligibility"]["shadow_optimizer"] is False
    assert reported["eligibility"]["route_completion"] is False
    assert proposal["proposal_kind"] == "biocatalytic_step"
    assert proposal["status"] == "proposal_only"
    assert proposal["semantics"]["not_a_canonical_reaction_edge"] is True
    assert result["semantics"]["unrestitched_mechanism_stays_in_discovery"] is True
    assert result["semantics"]["mechanism_success_does_not_create_canonical_reaction_proof"] is True
    assert result["semantics"]["execution_programs_have_no_store_admission_path"] is True
    assert (
        result["semantics"]["experimental_failures_are_retained_as_exact_boundary_feedback"] is True
    )
    assert result["semantics"]["experimental_claims_are_exact_boundary_observations_only"] is True
    assert (
        gateway.status("program-innovation-gateway")["status"]["graph_revision"] == graph_revision
    )


def test_gateway_audits_executor_result_without_admitting_validation_or_claim(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        run_id="experiment-result-gateway",
        target_name="ethanol",
        target_smiles="CCO",
        global_plan=_reduction_plan(),
        materialize=True,
    )
    route_id = next(
        iter(gateway.workbench("experiment-result-gateway")["snapshot"]["routes"])
    )
    capability = _reduction_capability()
    review = gateway.route_program_innovations(
        "experiment-result-gateway",
        route_id=route_id,
        capabilities=[capability],
    )
    proposal = next(iter(review["program_bundle"]["program_proposals"].values()))
    request = next(
        iter(review["experimental_work_frontier"]["work_items"].values())
    )["execution_request"]
    validation = _biocatalysis_validation(proposal)
    result = build_experiment_execution_result(
        request,
        result_id="experiment-result:gateway-reduction",
        executor_id="fixture-lab",
        executor_version="1.0",
        status="success",
        artifact_refs=[
            {"sha256": "a" * 64, "media_type": "application/json", "role": "raw_record"}
        ],
        domain_validation_candidate=validation,
    )
    graph_revision = gateway.status("experiment-result-gateway")["status"]["graph_revision"]

    audited = gateway.audit_route_experiment_result(
        "experiment-result-gateway",
        route_id=route_id,
        capabilities=[capability],
        result=result,
    )

    assert audited["operation"] == "route-experiment-result-audit"
    assert audited["result_audit"]["accepted_for_domain_gate"] is True
    assert audited["domain_validation_candidate"] == validation
    assert audited["next_boundary"] == "submit_candidate_to_existing_domain_validation_gate"
    assert gateway.status("experiment-result-gateway")["status"]["graph_revision"] == graph_revision
    assert gateway.experimental_claim_store("experiment-result-gateway")["replay"]["event_count"] == 0

    aborted = build_experiment_execution_result(
        request,
        result_id="experiment-result:gateway-aborted",
        executor_id="fixture-lab",
        executor_version="1.0",
        status="aborted",
        failure_reasons=["instrument_unavailable"],
    )
    aborted_review = gateway.audit_route_experiment_result(
        "experiment-result-gateway",
        route_id=route_id,
        capabilities=[capability],
        result=aborted,
    )
    assert aborted_review["result_audit"]["accepted_for_domain_gate"] is False
    assert aborted_review["domain_validation_candidate"] == {}

    tampered = dict(result)
    tampered["request_sha256"] = "b" * 64
    tampered = with_experiment_execution_result_digest(tampered)
    tampered_review = gateway.audit_route_experiment_result(
        "experiment-result-gateway",
        route_id=route_id,
        capabilities=[capability],
        result=tampered,
    )
    assert tampered_review["result_audit"]["accepted_for_domain_gate"] is False
    assert "request_binding_equal" in tampered_review["result_audit"]["reasons"]


def test_gateway_dispatches_recovers_and_settles_on_single_kernel_ledger(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        run_id="experiment-dispatch-gateway",
        target_name="ethanol",
        target_smiles="CCO",
        global_plan=_reduction_plan(),
        materialize=True,
    )
    route_id = next(
        iter(gateway.workbench("experiment-dispatch-gateway")["snapshot"]["routes"])
    )
    capability = _reduction_capability()
    review = gateway.route_program_innovations(
        "experiment-dispatch-gateway",
        route_id=route_id,
        capabilities=[capability],
    )
    proposal = next(iter(review["program_bundle"]["program_proposals"].values()))
    request = next(
        iter(review["experimental_work_frontier"]["work_items"].values())
    )["execution_request"]
    with pytest.raises(CampaignGatewayError, match="explicit_enable_required"):
        gateway.dispatch_route_experiment(
            "experiment-dispatch-gateway",
            route_id=route_id,
            capabilities=[capability],
            request_id=request["request_id"],
            policy=_manual_experiment_policy(),
        )

    def dispatch_once() -> dict:
        return gateway.dispatch_route_experiment(
            "experiment-dispatch-gateway",
            route_id=route_id,
            capabilities=[capability],
            request_id=request["request_id"],
            policy=_manual_experiment_policy(),
            enable_experiment_dispatch=True,
        )["dispatch"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_dispatches = list(pool.map(lambda _: dispatch_once(), range(4)))
    dispatched = concurrent_dispatches[0]
    assert all(value == dispatched for value in concurrent_dispatches)
    repeated = gateway.dispatch_route_experiment(
        "experiment-dispatch-gateway",
        route_id=route_id,
        capabilities=[capability],
        request_id=request["request_id"],
        policy=_manual_experiment_policy(),
        enable_experiment_dispatch=True,
    )["dispatch"]
    service = gateway._open("experiment-dispatch-gateway")
    assert dispatched == repeated
    assert dispatched["status"] == "awaiting_external_result"
    assert dispatched["handoff"]["semantics"][
        "handoff_grants_no_validation_claim_or_route_authority"
    ] is True
    assert service.kernel.task_lifecycle(dispatched["task_id"])["status"] == "in_flight"
    assert service.kernel.count_task_reservations(
        kind="experiment", metadata={"dispatch_id": dispatched["dispatch_id"]}
    ) == 1

    metadata = service.kernel.task_lifecycle(dispatched["task_id"])["reservation"][
        "payload"
    ]["metadata"]
    pointer_path = service.kernel.artifacts.pointers_root / (
        str(metadata["pointer_name"]) + ".json"
    )
    pointer_path.unlink()
    cooperatively_recovered = gateway.dispatch_route_experiment(
        "experiment-dispatch-gateway",
        route_id=route_id,
        capabilities=[capability],
        request_id=request["request_id"],
        policy=_manual_experiment_policy(),
        enable_experiment_dispatch=True,
    )["dispatch"]
    assert cooperatively_recovered == dispatched
    pointer_path.unlink()
    with pytest.raises(CampaignGatewayError, match="explicit_enable_required"):
        gateway.recover_route_experiment_dispatch(
            "experiment-dispatch-gateway",
            route_id=route_id,
            capabilities=[capability],
            dispatch_id=dispatched["dispatch_id"],
        )
    recovery = gateway.recover_route_experiment_dispatch(
        "experiment-dispatch-gateway",
        route_id=route_id,
        capabilities=[capability],
        dispatch_id=dispatched["dispatch_id"],
        enable_experiment_dispatch_recovery=True,
    )
    assert recovery["recovered"] is True
    assert recovery["dispatch"]["status"] == "awaiting_external_result"
    assert service.kernel.task_lifecycle(dispatched["task_id"])["status"] == "in_flight"

    raw_ref = service.kernel.artifacts.put_json(
        {"conversion_fraction": 0.91},
        logical_name="fixture-experiment-record.json",
        producer="test.fixture.lab",
    )
    validation = _biocatalysis_validation(proposal)
    result = build_experiment_execution_result(
        request,
        result_id="experiment-result:dispatched-manual",
        executor_id="autoplanner.manual_experiment_executor",
        executor_version="1.0.0",
        status="success",
        artifact_refs=[
            {
                "sha256": raw_ref.sha256,
                "media_type": "application/json",
                "role": "raw_record",
            }
        ],
        domain_validation_candidate=validation,
    )
    wrong_executor = build_experiment_execution_result(
        request,
        result_id="experiment-result:wrong-executor",
        executor_id="unbound-fixture-lab",
        executor_version="1.0.0",
        status="success",
        artifact_refs=[
            {
                "sha256": raw_ref.sha256,
                "media_type": "application/json",
                "role": "raw_record",
            }
        ],
        domain_validation_candidate=validation,
    )
    with pytest.raises(CampaignGatewayError, match="executor_not_dispatched_provider"):
        gateway.settle_route_experiment_dispatch(
            "experiment-dispatch-gateway",
            route_id=route_id,
            capabilities=[capability],
            dispatch_id=dispatched["dispatch_id"],
            result=wrong_executor,
            enable_experiment_settlement=True,
        )
    assert service.kernel.task_lifecycle(dispatched["task_id"])["status"] == "in_flight"
    graph_revision = service.kernel.state.graph_revision
    def settle_once() -> dict:
        return gateway.settle_route_experiment_dispatch(
            "experiment-dispatch-gateway",
            route_id=route_id,
            capabilities=[capability],
            dispatch_id=dispatched["dispatch_id"],
            result=result,
            enable_experiment_settlement=True,
        )["dispatch"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_settlements = list(pool.map(lambda _: settle_once(), range(4)))
    settled = concurrent_settlements[0]
    assert all(value == settled for value in concurrent_settlements)
    settled_again = gateway.settle_route_experiment_dispatch(
        "experiment-dispatch-gateway",
        route_id=route_id,
        capabilities=[capability],
        dispatch_id=dispatched["dispatch_id"],
        result=result,
        enable_experiment_settlement=True,
    )["dispatch"]

    assert settled == settled_again
    assert settled["status"] == "settled"
    assert settled["domain_validation_candidate"] == validation
    assert service.kernel.task_lifecycle(dispatched["task_id"])["status"] == "settled"
    assert gateway.status("experiment-dispatch-gateway")["status"][
        "graph_revision"
    ] == graph_revision
    assert gateway.experimental_claim_store("experiment-dispatch-gateway")["replay"][
        "event_count"
    ] == 0


def test_gateway_external_job_cancellation_requires_provider_acknowledgement(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    run_id = "experiment-external-cancellation"
    route_id, capability, _, request, dispatch = _prepare_dispatched_experiment(
        gateway, run_id
    )
    service = gateway._open(run_id)
    graph_revision = service.kernel.state.graph_revision
    claim_events = gateway.experimental_claim_store(run_id)["replay"]["event_count"]
    submitted = _external_job_receipt(
        dispatch, request, sequence=1, status="submitted"
    )
    wrong_binding = build_experiment_external_job_receipt(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"] + ":other",
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id="external-job:gateway-fixture", provider_sequence=1,
        status="submitted", recorded_by=_operator_identity(),
    )
    with pytest.raises(CampaignGatewayError, match="dispatch_binding_invalid"):
        gateway.record_route_experiment_job_receipt(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], job_receipt=wrong_binding,
            enable_experiment_job_receipt=True,
        )
    with pytest.raises(CampaignGatewayError, match="request_not_in_current_frontier"):
        gateway.record_route_experiment_job_receipt(
            run_id, route_id=route_id, capabilities=[],
            dispatch_id=dispatch["dispatch_id"], job_receipt=submitted,
            enable_experiment_job_receipt=True,
        )
    provider_id = dispatch["provider_id"]
    provider = gateway.providers.get(provider_id)
    descriptor = gateway.providers.descriptor(provider_id)
    gateway.providers.unregister(provider_id)
    with pytest.raises(CampaignGatewayError, match="unknown provider_id"):
        gateway.record_route_experiment_job_receipt(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], job_receipt=submitted,
            enable_experiment_job_receipt=True,
        )
    gateway.providers.register(
        provider, trusted_descriptor=descriptor, authority="test.host.registry"
    )

    def record_submitted() -> dict:
        return gateway.record_route_experiment_job_receipt(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], job_receipt=submitted,
            enable_experiment_job_receipt=True,
        )["dispatch"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        recorded = list(pool.map(lambda _: record_submitted(), range(4)))
    assert all(value == submitted for value in recorded)
    cancellation = build_experiment_cancellation_request(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"],
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id=submitted["external_job_id"],
        current_external_job_receipt_sha256=submitted["content_sha256"],
        requested_by=_operator_identity(), reason_code="operator_requested",
    )

    def request_cancel() -> dict:
        return gateway.request_route_experiment_cancellation(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"],
            cancellation_request=cancellation,
            enable_experiment_cancellation=True,
        )["dispatch"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        cancellations = list(pool.map(lambda _: request_cancel(), range(4)))
    assert all(value == cancellation for value in cancellations)
    lifecycle = service.kernel.task_lifecycle(dispatch["task_id"])
    assert lifecycle["status"] == "in_flight"
    assert len(lifecycle["checkpoints"]) == 2
    cancelled = _external_job_receipt(
        dispatch, request, sequence=2, status="cancelled",
        predecessor=submitted["content_sha256"],
        cancellation=cancellation["content_sha256"],
    )

    def record_cancelled() -> dict:
        return gateway.record_route_experiment_job_receipt(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], job_receipt=cancelled,
            enable_experiment_job_receipt=True,
        )["dispatch"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        acknowledged = list(pool.map(lambda _: record_cancelled(), range(4)))
    assert all(value == cancelled for value in acknowledged)
    lifecycle = service.kernel.task_lifecycle(dispatch["task_id"])
    assert lifecycle["status"] == "settled"
    assert lifecycle["settlement"]["payload"]["status"] == "cancelled"
    assert len(lifecycle["checkpoints"]) == 3
    assert service.kernel.state.graph_revision == graph_revision
    assert gateway.experimental_claim_store(run_id)["replay"][
        "event_count"
    ] == claim_events


def test_gateway_external_job_completion_gates_result_settlement(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    run_id = "experiment-external-completion"
    route_id, capability, proposal, request, dispatch = _prepare_dispatched_experiment(
        gateway, run_id
    )
    service = gateway._open(run_id)
    raw_ref = service.kernel.artifacts.put_json(
        {"conversion_fraction": 0.93}, logical_name="external-result.json",
        producer="test.fixture.lab",
    )
    result = build_experiment_execution_result(
        request, result_id="experiment-result:external-completed",
        executor_id=dispatch["provider_id"],
        executor_version=dispatch["provider_version"], status="success",
        artifact_refs=[{
            "sha256": raw_ref.sha256, "media_type": "application/json",
            "role": "raw_record",
        }],
        domain_validation_candidate=_biocatalysis_validation(proposal),
    )
    submitted = _external_job_receipt(
        dispatch, request, sequence=1, status="running"
    )
    gateway.record_route_experiment_job_receipt(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], job_receipt=submitted,
        enable_experiment_job_receipt=True,
    )
    with pytest.raises(CampaignGatewayError, match="not_ready_for_result"):
        gateway.settle_route_experiment_dispatch(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], result=result,
            enable_experiment_settlement=True,
        )
    cancellation = build_experiment_cancellation_request(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"],
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id=submitted["external_job_id"],
        current_external_job_receipt_sha256=submitted["content_sha256"],
        requested_by=_operator_identity(), reason_code="race-test",
    )
    gateway.request_route_experiment_cancellation(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], cancellation_request=cancellation,
        enable_experiment_cancellation=True,
    )
    completed = _external_job_receipt(
        dispatch, request, sequence=2, status="completed",
        predecessor=submitted["content_sha256"],
        cancellation=cancellation["content_sha256"],
    )
    gateway.record_route_experiment_job_receipt(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], job_receipt=completed,
        enable_experiment_job_receipt=True,
    )
    assert service.kernel.task_lifecycle(dispatch["task_id"])["status"] == "in_flight"
    settled = gateway.settle_route_experiment_dispatch(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], result=result,
        enable_experiment_settlement=True,
    )["dispatch"]
    assert settled["status"] == "settled"
    assert settled["domain_validation_candidate"] == _biocatalysis_validation(proposal)


def test_gateway_http_transport_submits_polls_and_acknowledges_cancellation(
    tmp_path: Path,
) -> None:
    gateway, calls = _http_gateway(tmp_path, [
        {
            "external_job_id": "gateway-job:1", "provider_sequence": 1,
            "status": "submitted", "status_detail": "queued",
        },
        {
            "external_job_id": "gateway-job:1", "provider_sequence": 2,
            "status": "running", "status_detail": "instrument running",
        },
        {
            "external_job_id": "gateway-job:1", "provider_sequence": 3,
            "status": "cancelled", "status_detail": "cancelled by device bridge",
        },
    ])
    run_id = "experiment-http-transport"
    route_id, capability, _, request, dispatch = _prepare_http_dispatch(gateway, run_id)
    service = gateway._open(run_id)
    graph_revision = service.kernel.state.graph_revision
    with pytest.raises(CampaignGatewayError, match="transport_explicit_enable_required"):
        gateway.submit_route_experiment_job(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"],
        )
    submitted = gateway.submit_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )["dispatch"]
    assert submitted["changed"] is True
    assert submitted["job_receipt"]["status"] == "submitted"
    assert calls[0]["headers"]["Idempotency-Key"] == submitted[
        "operation_request"
    ]["operation_id"]
    assert calls[0]["headers"]["Authorization"] == "Bearer gateway-secret"
    assert calls[0]["allow_redirects"] is False
    submitted_again = gateway.submit_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )["dispatch"]
    assert submitted_again["cached"] is True
    assert submitted_again["job_receipt"] == submitted["job_receipt"]
    assert len(calls) == 1
    polled = gateway.poll_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )["dispatch"]
    assert polled["job_receipt"]["status"] == "running"
    cancellation = build_experiment_cancellation_request(
        dispatch_id=dispatch["dispatch_id"], task_id=dispatch["task_id"],
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id=dispatch["provider_id"],
        provider_version=dispatch["provider_version"],
        external_job_id=polled["job_receipt"]["external_job_id"],
        current_external_job_receipt_sha256=polled["job_receipt"]["content_sha256"],
        requested_by=_operator_identity(), reason_code="device-run-no-longer-needed",
    )
    gateway.request_route_experiment_cancellation(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], cancellation_request=cancellation,
        enable_experiment_cancellation=True,
    )
    cancelled = gateway.transmit_route_experiment_cancellation(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )["dispatch"]
    assert cancelled["job_receipt"]["status"] == "cancelled"
    lifecycle = service.kernel.task_lifecycle(dispatch["task_id"])
    assert lifecycle["status"] == "settled"
    assert lifecycle["settlement"]["payload"]["status"] == "cancelled"
    assert len(lifecycle["checkpoints"]) == 7
    assert service.kernel.state.graph_revision == graph_revision
    assert "gateway-secret" not in json.dumps(lifecycle)


def test_gateway_http_transport_records_timeout_then_retries_same_task(
    tmp_path: Path,
) -> None:
    gateway, calls = _http_gateway(tmp_path, [
        requests.Timeout("fixture timeout"),
        {
            "external_job_id": "gateway-job:retry", "provider_sequence": 1,
            "status": "running", "status_detail": "accepted on retry",
        },
    ])
    run_id = "experiment-http-timeout-retry"
    route_id, capability, _, _, dispatch = _prepare_http_dispatch(gateway, run_id)
    first = gateway.submit_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], timeout_s=0.1,
        enable_experiment_transport=True,
    )["dispatch"]
    assert first["transport_result"]["outcome"] == "timeout"
    assert first["job_receipt"] == {}
    service = gateway._open(run_id)
    assert service.kernel.task_lifecycle(dispatch["task_id"])["status"] == "in_flight"
    second = gateway.submit_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], timeout_s=0.2,
        enable_experiment_transport=True,
    )["dispatch"]
    assert second["operation_request"]["attempt_number"] == 2
    assert second["transport_result"]["outcome"] == "success"
    assert second["job_receipt"]["status"] == "running"
    lifecycle = service.kernel.task_lifecycle(dispatch["task_id"])
    assert lifecycle["status"] == "in_flight"
    assert len(lifecycle["checkpoints"]) == 3
    assert len(calls) == 2


def test_gateway_http_poll_retries_after_successful_no_change_observation(
    tmp_path: Path,
) -> None:
    gateway, calls = _http_gateway(tmp_path, [
        {
            "external_job_id": "gateway-job:poll", "provider_sequence": 1,
            "status": "running", "status_detail": "started",
        },
        {
            "external_job_id": "gateway-job:poll", "provider_sequence": 1,
            "status": "running", "status_detail": "still running",
        },
        {
            "external_job_id": "gateway-job:poll", "provider_sequence": 2,
            "status": "completed", "status_detail": "finished",
        },
    ])
    run_id = "experiment-http-poll-no-change"
    route_id, capability, _, _, dispatch = _prepare_http_dispatch(gateway, run_id)
    gateway.submit_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )
    unchanged = gateway.poll_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )["dispatch"]
    assert unchanged["changed"] is False
    advanced = gateway.poll_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )["dispatch"]
    assert advanced["operation_request"]["attempt_number"] == 2
    assert advanced["changed"] is True
    assert advanced["job_receipt"]["status"] == "completed"
    assert len(calls) == 3


def test_gateway_concurrent_http_submit_uses_one_operation_and_receipt_chain(
    tmp_path: Path,
) -> None:
    response = {
        "external_job_id": "gateway-job:concurrent", "provider_sequence": 1,
        "status": "submitted", "status_detail": "queued once",
    }
    gateway, calls = _http_gateway(tmp_path, [dict(response) for _ in range(4)])
    run_id = "experiment-http-concurrent-submit"
    route_id, capability, _, _, dispatch = _prepare_http_dispatch(gateway, run_id)

    def submit() -> dict:
        return gateway.submit_route_experiment_job(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
        )["dispatch"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: submit(), range(4)))
    receipt_ids = {row["job_receipt"]["content_sha256"] for row in results}
    assert len(receipt_ids) == 1
    assert 1 <= len(calls) <= 4
    assert len({row["headers"]["Idempotency-Key"] for row in calls}) == 1
    lifecycle = gateway._open(run_id).kernel.task_lifecycle(dispatch["task_id"])
    assert lifecycle["status"] == "in_flight"
    assert len(lifecycle["checkpoints"]) == 2


def test_gateway_http_transport_recovers_success_after_checkpoint_only_crash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gateway, calls = _http_gateway(tmp_path, [{
        "external_job_id": "gateway-job:recover", "provider_sequence": 1,
        "status": "submitted", "status_detail": "submitted once",
    }])
    run_id = "experiment-http-checkpoint-recovery"
    route_id, capability, _, _, dispatch = _prepare_http_dispatch(gateway, run_id)
    original = experiment_job_transport_runtime.record_current_route_experiment_job_receipt

    def fail_after_attempt(*args, **kwargs):
        del args, kwargs
        raise experiment_job_transport_runtime.ExperimentDispatchError(
            "fixture_crash_after_transport_checkpoint"
        )

    monkeypatch.setattr(
        experiment_job_transport_runtime,
        "record_current_route_experiment_job_receipt",
        fail_after_attempt,
    )
    with pytest.raises(CampaignGatewayError, match="fixture_crash_after"):
        gateway.submit_route_experiment_job(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
        )
    service = gateway._open(run_id)
    assert len(service.kernel.task_lifecycle(dispatch["task_id"])["checkpoints"]) == 1
    monkeypatch.setattr(
        experiment_job_transport_runtime,
        "record_current_route_experiment_job_receipt",
        original,
    )
    recovered = gateway.submit_route_experiment_job(
        run_id, route_id=route_id, capabilities=[capability],
        dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
    )["dispatch"]
    assert recovered["cached"] is True
    assert recovered["job_receipt"]["status"] == "submitted"
    assert len(calls) == 1


def test_gateway_http_transport_rejects_endpoint_config_drift_before_network(
    tmp_path: Path,
) -> None:
    gateway, calls = _http_gateway(tmp_path, [])
    run_id = "experiment-http-config-drift"
    route_id, capability, _, _, dispatch = _prepare_http_dispatch(gateway, run_id)
    provider = gateway.providers.get(dispatch["provider_id"])
    provider.config = HttpExperimentExecutorConfig(
        provider_id=provider.descriptor.provider_id,
        version=provider.descriptor.version,
        base_url="https://other-lab.example.invalid",
        auth_token_env="FIXTURE_GATEWAY_LAB_TOKEN",
        operator_principal_id="service:gateway-http-fixture",
    )
    with pytest.raises(CampaignGatewayError, match="config_binding_changed"):
        gateway.submit_route_experiment_job(
            run_id, route_id=route_id, capabilities=[capability],
            dispatch_id=dispatch["dispatch_id"], enable_experiment_transport=True,
        )
    assert calls == []


def test_gateway_discovers_http_experiment_provider_only_from_host_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_EXPERIMENT_HTTP_BASE_URL", "https://lab.example.test"
    )
    monkeypatch.setenv(
        "AUTOPLANNER_EXPERIMENT_HTTP_PROVIDER_ID", "fixture.env-gateway-http"
    )
    monkeypatch.setenv(
        "AUTOPLANNER_EXPERIMENT_HTTP_BEARER_TOKEN_ENV", "ENV_GATEWAY_LAB_TOKEN"
    )
    monkeypatch.setenv("ENV_GATEWAY_LAB_TOKEN", "environment-secret")
    gateway = CampaignGateway(_paths(tmp_path))
    descriptor = gateway.providers.descriptor("fixture.env-gateway-http")
    assert descriptor.network_access is True
    assert "experiment.transport.submit" in descriptor.capabilities


def test_gateway_durably_admits_only_validated_biocatalytic_programs(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    gateway = CampaignGateway(paths)
    created = gateway.create_run(
        run_id="biocatalytic-admission-gateway",
        target_name="ethanol",
        target_smiles="CCO",
        global_plan=_reduction_plan(),
        materialize=True,
    )
    run_dir = Path(created["run_dir"])
    route_id = next(iter(gateway.workbench("biocatalytic-admission-gateway")["snapshot"]["routes"]))
    review = gateway.route_program_innovations(
        "biocatalytic-admission-gateway",
        route_id=route_id,
        capabilities=[_reduction_capability()],
    )
    proposal = next(iter(review["program_bundle"]["program_proposals"].values()))
    validation = _biocatalysis_validation(proposal)
    graph_revision = gateway.status("biocatalytic-admission-gateway")["status"]["graph_revision"]
    store_root = run_dir / ".autoplanner" / "bio_programs"

    with pytest.raises(CampaignGatewayError, match="explicit_enable_required"):
        gateway.admit_route_program_innovations(
            "biocatalytic-admission-gateway",
            route_id=route_id,
            capabilities=[_reduction_capability()],
            validations=[validation],
        )
    assert not store_root.exists()
    with pytest.raises(CampaignGatewayError, match="requires_validated_candidate"):
        gateway.admit_route_program_innovations(
            "biocatalytic-admission-gateway",
            route_id=route_id,
            capabilities=[_reduction_capability()],
            enable_biocatalytic_program_admission=True,
        )
    assert not store_root.exists()

    first = gateway.admit_route_program_innovations(
        "biocatalytic-admission-gateway",
        route_id=route_id,
        capabilities=[_reduction_capability()],
        validations=[validation],
        enable_biocatalytic_program_admission=True,
    )
    second = gateway.admit_route_program_innovations(
        "biocatalytic-admission-gateway",
        route_id=route_id,
        capabilities=[_reduction_capability()],
        validations=[validation],
        enable_biocatalytic_program_admission=True,
    )
    durable = gateway.biocatalytic_program_store("biocatalytic-admission-gateway")

    assert first["created"] is True
    assert second["created"] is False
    assert first["event"]["admitted_program_ids"] == [proposal["program_id"]]
    assert first["store"]["oracle"]["accepted"] is True
    assert durable["replay"]["event_count"] == 1
    assert durable["replay"]["admitted_program_ids"] == [proposal["program_id"]]
    assert gateway.program_store("biocatalytic-admission-gateway")["status"]["event_count"] == 0
    assert (
        gateway.status("biocatalytic-admission-gateway")["status"]["graph_revision"]
        == graph_revision
    )
    validation_report = gateway.validate("biocatalytic-admission-gateway")
    replay_report = gateway.replay("biocatalytic-admission-gateway")
    assert validation_report["accepted"] is True
    assert replay_report["accepted"] is True
    assert validation_report["biocatalytic_program_store"]["event_count"] == 1
    assert replay_report["biocatalytic_program_store"]["event_count"] == 1
    scopes = {
        row["authority_scope"]
        for row in gateway.index.artifacts_for_run("biocatalytic-admission-gateway")
        if row["authority_scope"].startswith("shadow_biocatalytic_program_")
    }
    assert len(scopes) == 6
    bundle_sha256 = first["event"]["bundle_ref"]["sha256"]
    gc_candidates = {
        row["sha256"] for row in gateway.gc_plan(minimum_age_s=0)["plan"]["candidates"]
    }
    assert bundle_sha256 not in gc_candidates
    admitted_digests = {
        first["event"][key]["sha256"]
        for key in (
            "source_graph_ref",
            "source_route_ref",
            "baseline_projection_ref",
            "discovery_ref",
            "bundle_ref",
            "validation_pack_ref",
        )
    }
    paths.run_index_path.unlink()
    rebuilt_gc = CampaignGateway(paths).gc_plan(minimum_age_s=0)
    rebuilt_candidates = {row["sha256"] for row in rebuilt_gc["plan"]["candidates"]}
    assert admitted_digests.isdisjoint(rebuilt_candidates)


def test_biocatalytic_program_store_replay_rejects_event_tampering(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    created = gateway.create_run(
        run_id="biocatalytic-event-tamper",
        target_name="ethanol",
        target_smiles="CCO",
        global_plan=_reduction_plan(),
        materialize=True,
    )
    route_id = next(iter(gateway.workbench("biocatalytic-event-tamper")["snapshot"]["routes"]))
    review = gateway.route_program_innovations(
        "biocatalytic-event-tamper",
        route_id=route_id,
        capabilities=[_reduction_capability()],
    )
    proposal = next(iter(review["program_bundle"]["program_proposals"].values()))
    gateway.admit_route_program_innovations(
        "biocatalytic-event-tamper",
        route_id=route_id,
        capabilities=[_reduction_capability()],
        validations=[_biocatalysis_validation(proposal)],
        enable_biocatalytic_program_admission=True,
    )
    event_path = next(
        (Path(created["run_dir"]) / ".autoplanner" / "bio_programs" / "e").glob("*/*.json")
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["admitted_program_ids"] = ["program:forged"]
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with pytest.raises(CampaignGatewayError, match="event_content_digest_invalid"):
        gateway.biocatalytic_program_store("biocatalytic-event-tamper")


def test_gateway_durably_admits_exact_boundary_claims_and_recovers_gc_pins(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    gateway = CampaignGateway(paths)
    created = gateway.create_run(
        run_id="experimental-claim-gateway",
        target_name="ethanol",
        target_smiles="CCO",
        global_plan=_reduction_plan(),
        materialize=True,
    )
    route_id = next(iter(gateway.workbench("experimental-claim-gateway")["snapshot"]["routes"]))
    review = gateway.route_program_innovations(
        "experimental-claim-gateway",
        route_id=route_id,
        capabilities=[_reduction_capability()],
    )
    proposal = next(iter(review["program_bundle"]["program_proposals"].values()))
    validation = _biocatalysis_validation(proposal)
    graph_revision = gateway.status("experimental-claim-gateway")["status"]["graph_revision"]
    store_root = Path(created["run_dir"]) / ".autoplanner" / "experimental_claims"

    with pytest.raises(CampaignGatewayError, match="explicit_enable_required"):
        gateway.admit_route_experimental_claims(
            "experimental-claim-gateway",
            route_id=route_id,
            capabilities=[_reduction_capability()],
            validations=[validation],
        )
    assert not store_root.exists()
    with pytest.raises(CampaignGatewayError, match="requires_observation"):
        gateway.admit_route_experimental_claims(
            "experimental-claim-gateway",
            route_id=route_id,
            capabilities=[_reduction_capability()],
            enable_experimental_claim_admission=True,
        )
    assert not store_root.exists()

    first = gateway.admit_route_experimental_claims(
        "experimental-claim-gateway",
        route_id=route_id,
        capabilities=[_reduction_capability()],
        validations=[validation],
        enable_experimental_claim_admission=True,
    )
    second = gateway.admit_route_experimental_claims(
        "experimental-claim-gateway",
        route_id=route_id,
        capabilities=[_reduction_capability()],
        validations=[validation],
        enable_experimental_claim_admission=True,
    )
    durable = gateway.experimental_claim_store("experimental-claim-gateway")

    assert first["created"] is True
    assert second["created"] is False
    assert first["event"]["counts"]["claims"] == 1
    assert first["event"]["counts"]["positive"] == 1
    assert first["event"]["semantics"]["cannot_create_canonical_reaction_proof"] is True
    assert durable["replay"]["event_count"] == 1
    assert gateway.status("experimental-claim-gateway")["status"]["graph_revision"] == graph_revision
    assert gateway.validate("experimental-claim-gateway")["experimental_claim_store"][
        "event_count"
    ] == 1
    assert gateway.replay("experimental-claim-gateway")["experimental_claim_store"][
        "event_count"
    ] == 1
    admitted_digests = {
        first["event"][key]["sha256"]
        for key in (
            "source_graph_ref",
            "source_route_ref",
            "source_projection_ref",
            "source_discovery_ref",
            "validation_pack_ref",
            "claim_set_ref",
        )
    }
    paths.run_index_path.unlink()
    rebuilt_candidates = {
        row["sha256"]
        for row in CampaignGateway(paths).gc_plan(minimum_age_s=0)["plan"]["candidates"]
    }
    assert admitted_digests.isdisjoint(rebuilt_candidates)


def test_gateway_validation_detects_and_repairs_a_stale_program_projection(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    gateway.create_run(
        run_id="program-stale",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
    )
    gateway.admit_programs("program-stale", enable_program_admission=True)
    gateway.apply_plan("program-stale", _plan(), materialize=True)

    stale = gateway.validate("program-stale")
    refreshed = gateway.admit_programs("program-stale", enable_program_admission=True)
    repaired = gateway.validate("program-stale")

    assert stale["checks"]["program_store_current_projection_equal"] is False
    assert refreshed["created"] is True
    assert refreshed["store"]["event_count"] == 2
    assert repaired["checks"]["program_store_current_projection_equal"] is True
    assert repaired["accepted"] is True


def test_program_store_cas_pins_survive_run_index_recreation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = CampaignGateway(paths)
    gateway.create_run(
        run_id="program-gc-recovery",
        target_name="ethanol",
        target_smiles="CCO",
    )
    admitted = gateway.admit_programs("program-gc-recovery", enable_program_admission=True)
    projection_sha256 = admitted["event"]["projection_ref"]["sha256"]
    paths.run_index_path.unlink()

    rebuilt = CampaignGateway(paths)
    gc = rebuilt.gc_plan(minimum_age_s=0)
    candidates = {row["sha256"] for row in gc["plan"]["candidates"]}

    assert gc["indexed_artifact_pin_count"] == 0
    assert gc["program_store_pin_count"] == 2
    assert projection_sha256 not in candidates
