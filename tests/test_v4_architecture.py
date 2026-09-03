from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import tomllib

from cascade_planner.application.compatibility_inventory import (
    compatibility_inventory,
    record_compatibility_use,
)

ROOT = Path(__file__).resolve().parents[1]
V4_MODULES = (
    "cascade_planner/application/biocatalytic_program_contracts.py",
    "cascade_planner/application/biocatalysis_validation_frontier.py",
    "cascade_planner/application/biocatalytic_programs.py",
    "cascade_planner/application/biocatalytic_program_store.py",
    "cascade_planner/application/campaign_contract_json.py",
    "cascade_planner/application/biocatalytic_program_store_contracts.py",
    "cascade_planner/application/biocatalytic_program_store_replay.py",
    "cascade_planner/application/canonical_identity.py",
    "cascade_planner/application/canonical_hypergraph.py",
    "cascade_planner/application/campaign_actions.py",
    "cascade_planner/application/campaign_action_latency.py",
    "cascade_planner/application/campaign_action_status.py",
    "cascade_planner/application/campaign_work_policy.py",
    "cascade_planner/application/campaign_trajectory.py",
    "cascade_planner/application/guided_search_progress.py",
    "cascade_planner/application/campaign_review_bundle.py",
    "cascade_planner/application/campaign_quality_state.py",
    "cascade_planner/application/candidate_lifecycle.py",
    "cascade_planner/application/candidate_provenance.py",
    "cascade_planner/application/candidate_provider_route_audit.py",
    "cascade_planner/application/action_convergence.py",
    "cascade_planner/application/action_preflight.py",
    "cascade_planner/application/action_service_policy.py",
    "cascade_planner/application/action_scheduler.py",
    "cascade_planner/application/replan_pressure.py",
    "cascade_planner/application/scientific_closure_pressure.py",
    "cascade_planner/application/strategy_experiment_closure.py",
    "cascade_planner/application/strategy_experiment_closure_route.py",
    "cascade_planner/application/unified_campaign_spec.py",
    "cascade_planner/orchestration/unified_campaign_runtime.py",
    "cascade_planner/application/candidate_innovation_screen.py",
    "cascade_planner/application/candidate_programs.py",
    "cascade_planner/application/candidate_route_observations.py",
    "cascade_planner/application/capability_applicability_calibration.py",
    "cascade_planner/application/deficit_frontier.py",
    "cascade_planner/application/execution_capability_feedback.py",
    "cascade_planner/application/execution_program_compilation.py",
    "cascade_planner/application/execution_program_route_candidates.py",
    "cascade_planner/application/execution_program_validations.py",
    "cascade_planner/application/execution_programs.py",
    "cascade_planner/application/execution_validation_frontier.py",
    "cascade_planner/application/external_strategy_routes.py",
    "cascade_planner/application/experiment_execution_contracts.py",
    "cascade_planner/application/experiment_execution_results.py",
    "cascade_planner/application/experiment_external_jobs.py",
    "cascade_planner/application/experiment_external_job_operations.py",
    "cascade_planner/application/experimental_claim_adapters.py",
    "cascade_planner/application/experimental_claim_contracts.py",
    "cascade_planner/application/experimental_claim_rows.py",
    "cascade_planner/application/experimental_claim_store.py",
    "cascade_planner/application/experimental_claim_store_contracts.py",
    "cascade_planner/application/experimental_claim_store_projection.py",
    "cascade_planner/application/experimental_claim_store_replay.py",
    "cascade_planner/application/experimental_claims.py",
    "cascade_planner/application/experimental_work_experience_priority.py",
    "cascade_planner/application/experimental_work_frontier.py",
    "cascade_planner/application/experimental_work_scheduling.py",
    "cascade_planner/application/fact_lifecycle.py",
    "cascade_planner/application/mechanism_program_route_candidates.py",
    "cascade_planner/application/mechanism_program_compilation.py",
    "cascade_planner/application/mechanism_program_validations.py",
    "cascade_planner/application/mechanism_programs.py",
    "cascade_planner/application/mechanism_program_store.py",
    "cascade_planner/application/mechanism_program_store_contracts.py",
    "cascade_planner/application/mechanism_program_store_replay.py",
    "cascade_planner/application/mechanism_experiment_feedback.py",
    "cascade_planner/application/mechanism_validation_frontier.py",
    "cascade_planner/application/milestone_subscription.py",
    "cascade_planner/application/milestone_notification.py",
    "cascade_planner/application/milestone_outbox_store.py",
    "cascade_planner/application/frontier_runtime.py",
    "cascade_planner/application/pareto.py",
    "cascade_planner/application/portfolio_selection.py",
    "cascade_planner/application/product_profiles.py",
    "cascade_planner/application/program_route_candidates.py",
    "cascade_planner/application/program_route_candidate_contracts.py",
    "cascade_planner/application/program_route_candidate_factory.py",
    "cascade_planner/application/program_route_optimizer.py",
    "cascade_planner/application/program_innovation_contracts.py",
    "cascade_planner/application/program_experience.py",
    "cascade_planner/application/program_applicability.py",
    "cascade_planner/application/program_applicability_oracle.py",
    "cascade_planner/application/program_experience_store.py",
    "cascade_planner/application/program_span_substitutions.py",
    "cascade_planner/application/program_validation_contracts.py",
    "cascade_planner/application/program_validation_feedback_contracts.py",
    "cascade_planner/application/program_validation_frontier_contracts.py",
    "cascade_planner/application/program_validation_routing.py",
    "cascade_planner/application/reported_program_route_candidates.py",
    "cascade_planner/application/proof_fact_projection.py",
    "cascade_planner/application/proof_policy.py",
    "cascade_planner/application/proof_portfolio.py",
    "cascade_planner/application/proof_portfolio_replacements.py",
    "cascade_planner/application/program_opportunity_pressure.py",
    "cascade_planner/application/reaction_template_extraction.py",
    "cascade_planner/application/reaction_template_library.py",
    "cascade_planner/application/reaction_template_store.py",
    "cascade_planner/application/reaction_condition_records.py",
    "cascade_planner/application/reactionjson_replay.py",
    "cascade_planner/application/route_variants.py",
    "cascade_planner/application/route_execution_capabilities.py",
    "cascade_planner/application/route_execution_discovery.py",
    "cascade_planner/application/route_innovations.py",
    "cascade_planner/application/route_pareto_vector.py",
    "cascade_planner/application/route_innovation_chemenzy.py",
    "cascade_planner/application/route_innovation_capabilities.py",
    "cascade_planner/application/route_innovation_discovery.py",
    "cascade_planner/application/route_innovation_windows.py",
    "cascade_planner/application/route_innovation_precedents.py",
    "cascade_planner/application/route_structure_matching.py",
    "cascade_planner/application/route_program_dual_read.py",
    "cascade_planner/application/transformation_program_store.py",
    "cascade_planner/application/transformation_program_validation.py",
    "cascade_planner/application/transformation_programs.py",
    "cascade_planner/application/route_workbench.py",
    "cascade_planner/application/route_workbench_closure.py",
    "cascade_planner/application/route_workbench_fact_rows.py",
    "cascade_planner/application/route_workbench_inspectors.py",
    "cascade_planner/application/route_workbench_planned_routes.py",
    "cascade_planner/application/route_workbench_edge_proof_vector.py",
    "cascade_planner/application/route_workbench_proof_vectors.py",
    "cascade_planner/application/route_workbench_route_rows.py",
    "cascade_planner/application/target_route_readiness.py",
    "cascade_planner/application/worker_runtime.py",
    "cascade_planner/harness/v4_route_display.py",
    "cascade_planner/harness/v4_route_branch.py",
    "cascade_planner/harness/v4_route_condition_projection.py",
    "cascade_planner/harness/v4_route_condition_resolution.py",
    "cascade_planner/harness/v4_route_evidence_projection.py",
    "cascade_planner/harness/v4_route_graph_projection.py",
    "cascade_planner/harness/v4_route_nodes.py",
    "cascade_planner/harness/v4_planned_route_branches.py",
    "cascade_planner/harness/source_condition_text.py",
    "cascade_planner/interfaces/campaign_gateway.py",
    "cascade_planner/interfaces/campaign_milestone_gateway.py",
    "cascade_planner/interfaces/campaign_benchmark.py",
    "cascade_planner/interfaces/biocatalytic_program_gc.py",
    "cascade_planner/interfaces/campaign_experimental_claim_store.py",
    "cascade_planner/interfaces/campaign_mechanism_program_store.py",
    "cascade_planner/interfaces/campaign_program_experience.py",
    "cascade_planner/interfaces/campaign_experiment_dispatch.py",
    "cascade_planner/interfaces/campaign_experiment_gateway.py",
    "cascade_planner/interfaces/campaign_experiment_job_gateway.py",
    "cascade_planner/interfaces/campaign_experiment_jobs.py",
    "cascade_planner/interfaces/campaign_experiment_transport.py",
    "cascade_planner/interfaces/campaign_experiment_transport_gateway.py",
    "cascade_planner/interfaces/campaign_gateway_contract.py",
    "cascade_planner/interfaces/campaign_gateway_projection.py",
    "cascade_planner/interfaces/campaign_gateway_stock_oracle.py",
    "cascade_planner/interfaces/campaign_program_gateway.py",
    "cascade_planner/interfaces/campaign_program_innovation_gateway.py",
    "cascade_planner/interfaces/campaign_program_innovation_store.py",
    "cascade_planner/interfaces/campaign_program_innovations.py",
    "cascade_planner/interfaces/campaign_programs.py",
    "cascade_planner/interfaces/campaign_recovery.py",
    "cascade_planner/interfaces/campaign_recovery_stores.py",
    "cascade_planner/interfaces/experimental_claim_cli.py",
    "cascade_planner/interfaces/experimental_claim_gc.py",
    "cascade_planner/interfaces/external_strategy_import.py",
    "cascade_planner/interfaces/mechanism_program_gc.py",
    "cascade_planner/interfaces/experiment_dispatch_cli.py",
    "cascade_planner/interfaces/experiment_job_cli.py",
    "cascade_planner/interfaces/experiment_transport_cli.py",
    "cascade_planner/interfaces/candidate_migration.py",
    "cascade_planner/interfaces/campaign_operations.py",
    "cascade_planner/interfaces/program_cli.py",
    "cascade_planner/interfaces/program_innovation_cli.py",
    "cascade_planner/interfaces/program_gc.py",
    "cascade_planner/interfaces/program_migration.py",
    "cascade_planner/interfaces/replay_store_gc.py",
    "cascade_planner/interfaces/chemenzy_builtin_runtime.py",
    "cascade_planner/interfaces/chemenzy_probe.py",
    "cascade_planner/interfaces/chemenzy_probe_contract.py",
    "cascade_planner/interfaces/chemenzy_parameter_binding.py",
    "cascade_planner/interfaces/chemenzy_probe_routes.py",
    "cascade_planner/interfaces/chemenzy_route_invariants.py",
    "cascade_planner/interfaces/chemenzy_route_topology.py",
    "cascade_planner/interfaces/literature_candidates.py",
    "cascade_planner/interfaces/literature_relevance.py",
    "cascade_planner/interfaces/literature_evidence.py",
    "cascade_planner/interfaces/literature_evidence_connector.py",
    "cascade_planner/interfaces/literature_evidence_contract.py",
    "cascade_planner/interfaces/literature_fulltext.py",
    "cascade_planner/interfaces/literature_browser.py",
    "cascade_planner/interfaces/literature_authorized_pdf_assets.py",
    "cascade_planner/interfaces/literature_authorized_source.py",
    "cascade_planner/interfaces/literature_html_parser.py",
    "cascade_planner/interfaces/literature_materialization.py",
    "cascade_planner/interfaces/literature_pdf_materialization.py",
    "cascade_planner/interfaces/literature_pdf_projection.py",
    "cascade_planner/interfaces/literature_procedure_line_fragments.py",
    "cascade_planner/interfaces/literature_procedure_fragments.py",
    "cascade_planner/interfaces/visual_evidence.py",
    "cascade_planner/interfaces/visual_evidence_contract.py",
    "cascade_planner/interfaces/visual_evidence_materialization.py",
    "cascade_planner/interfaces/visual_evidence_request.py",
    "cascade_planner/interfaces/case_dossier.py",
    "cascade_planner/interfaces/case_dossier_compiler.py",
    "cascade_planner/interfaces/case_dossier_contract.py",
    "cascade_planner/interfaces/case_cli.py",
    "cascade_planner/interfaces/case_runner.py",
    "cascade_planner/interfaces/patent_self_evolution.py",
    "cascade_planner/interfaces/replay_contract.py",
    "cascade_planner/interfaces/replay_lifecycle.py",
    "cascade_planner/interfaces/replay_observations.py",
    "cascade_planner/interfaces/replay_pack.py",
    "cascade_planner/interfaces/replay_reporting.py",
    "cascade_planner/interfaces/visual_observation_chemistry.py",
    "cascade_planner/interfaces/visual_observation_normalization.py",
    "cascade_planner/orchestration/global_campaign_director.py",
    "cascade_planner/orchestration/biocatalytic_program_admission_runtime.py",
    "cascade_planner/orchestration/execution_program_review_materials.py",
    "cascade_planner/orchestration/experiment_execution_runtime.py",
    "cascade_planner/orchestration/experiment_dispatch_handoff.py",
    "cascade_planner/orchestration/experiment_dispatch_runtime.py",
    "cascade_planner/orchestration/experiment_dispatch_support.py",
    "cascade_planner/orchestration/experiment_external_job_runtime.py",
    "cascade_planner/orchestration/experiment_job_transport_runtime.py",
    "cascade_planner/orchestration/experimental_claim_admission_runtime.py",
    "cascade_planner/orchestration/experimental_claim_review_materials.py",
    "cascade_planner/orchestration/mechanism_program_review_materials.py",
    "cascade_planner/orchestration/mechanism_program_admission_runtime.py",
    "cascade_planner/orchestration/program_admission_runtime.py",
    "cascade_planner/orchestration/program_candidate_review_materials.py",
    "cascade_planner/orchestration/program_innovation_materials.py",
    "cascade_planner/orchestration/program_innovation_runtime.py",
    "cascade_planner/orchestration/program_experience_runtime.py",
    "cascade_planner/orchestration/retrosynthesis_service.py",
    "cascade_planner/orchestration/retrosynthesis_service_execution.py",
    "cascade_planner/orchestration/retrosynthesis_service_planning.py",
    "cascade_planner/orchestration/route_innovation_runtime.py",
    "cascade_planner/runtime/immutable_event_store.py",
    "cascade_planner/runtime/immutable_json_events.py",
    "cascade_planner/providers/experiment.py",
    "cascade_planner/providers/http_experiment.py",
    "cascade_planner/web/v4_target_runtime.py",
    "cascade_planner/interfaces/target_solve_request.py",
    "cascade_planner/interfaces/campaign_action_timeline.py",
    "cascade_planner/interfaces/target_job_projection.py",
    "cascade_planner/interfaces/target_solver_compat.py",
)
FORBIDDEN_V4_DEPENDENCIES = (
    "cascade_planner.legacy",
    "cascade_planner.research",
    "cascade_planner.web",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_v4_modules_do_not_import_frozen_ownership_paths() -> None:
    violations = []
    for relative in V4_MODULES:
        for imported in _imports(ROOT / relative):
            if imported.startswith(FORBIDDEN_V4_DEPENDENCIES):
                violations.append(f"{relative}->{imported}")
            if relative.startswith(
                "cascade_planner/application/"
            ) and imported.startswith("cascade_planner.orchestration"):
                violations.append(
                    f"application_reverse_dependency:{relative}->{imported}"
                )
    assert violations == []


def test_unified_action_core_has_no_dataset_specific_control_tokens() -> None:
    protected = (
        ROOT / "cascade_planner/application/campaign_actions.py",
        ROOT / "cascade_planner/application/action_convergence.py",
        ROOT / "cascade_planner/application/action_preflight.py",
        ROOT / "cascade_planner/application/action_service_policy.py",
        ROOT / "cascade_planner/application/action_scheduler.py",
        ROOT / "cascade_planner/application/replan_pressure.py",
        ROOT / "cascade_planner/application/scientific_closure_pressure.py",
        ROOT / "cascade_planner/application/experimental_work_experience_priority.py",
        ROOT / "cascade_planner/application/run_kernel.py",
        ROOT / "cascade_planner/application/program_opportunity_pressure.py",
        ROOT / "cascade_planner/orchestration/unified_campaign_runtime.py",
    )
    forbidden = (
        "benchmark_search",
        "retrostar",
        "objective_mode",
        "target_index",
        "dataset_id",
    )
    violations = []
    for path in protected:
        text = path.read_text(encoding="utf-8").casefold()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.name}:{token}")
    assert violations == []


def test_benchmark_harness_does_not_pass_a_result_view_into_the_solver() -> None:
    panel_path = ROOT / "scripts/run_v4_blind_panel.py"
    tree = ast.parse(panel_path.read_text(encoding="utf-8"))
    run_case = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_case"
    )
    string_literals = {
        node.value
        for node in ast.walk(run_case)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    w8_source = (ROOT / "scripts/run_retrostar190_w8.py").read_text(encoding="utf-8")

    assert "--objective-mode" not in string_literals
    assert "--objective-mode" not in w8_source
    assert "--fixed-cutoff-wall-time-s" in w8_source


def test_action_runtime_cannot_write_the_canonical_graph_directly() -> None:
    imported = _imports(
        ROOT / "cascade_planner/orchestration/unified_campaign_runtime.py"
    )

    assert "cascade_planner.application.canonical_hypergraph" not in imported
    assert "cascade_planner.orchestration.retrosynthesis_service" not in imported


def test_all_active_packages_have_one_canonical_store_owner_and_audited_writers() -> None:
    """Keep canonical writes behind the campaign service across the full package.

    The curated ``V4_MODULES`` dependency test protects the intended mainline.
    This wider scan prevents a new provider, Program adapter, Web handler, or
    manual-import surface from quietly constructing a second graph owner or
    mutating ``service.graph_store`` outside an explicitly audited boundary.
    """

    allowed_store_constructors = {
        "cascade_planner/orchestration/retrosynthesis_service.py",
    }
    allowed_graph_store_writers = {
        "cascade_planner/orchestration/retrosynthesis_service_execution.py",
        # Replay may append digest-bound lifecycle facts through the same store;
        # it does not ingest provider chemistry or grant new proof authority.
        "cascade_planner/interfaces/replay_lifecycle.py",
    }
    constructor_calls: set[str] = set()
    graph_store_writes: set[str] = set()
    forbidden_imports: list[str] = []
    package_root = ROOT / "cascade_planner"
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(
            ("cascade_planner/legacy/", "cascade_planner/research/")
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported in _imports(path):
            if imported.startswith(
                ("cascade_planner.legacy", "cascade_planner.research")
            ):
                forbidden_imports.append(f"{relative}->{imported}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "CanonicalHypergraphStore"
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "CanonicalHypergraphStore"
            ):
                constructor_calls.add(relative)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "graph_store"
            ):
                graph_store_writes.add(relative)

    assert forbidden_imports == []
    assert constructor_calls == allowed_store_constructors
    assert graph_store_writes == allowed_graph_store_writers


def test_all_candidate_producers_enter_one_canonical_ingestion_surface() -> None:
    """Freeze every production construction and submission point for graph input.

    Provider, literature, template, manual, and Program adapters may prepare
    different proposal contracts.  They must nevertheless submit through the
    campaign service and its ``CanonicalIngestionBatch`` boundary; adding a new
    call site requires an explicit architecture review here.
    """

    allowed_batch_constructors = {
        "cascade_planner/orchestration/retrosynthesis_service_execution.py",
        "cascade_planner/orchestration/retrosynthesis_service_planning.py",
        "cascade_planner/interfaces/chemenzy_probe.py",
        "cascade_planner/interfaces/patent_self_evolution.py",
        "cascade_planner/interfaces/aizynthfinder_sidecar.py",
        # Lifecycle replay can append only validated, digest-bound fact events.
        "cascade_planner/interfaces/replay_lifecycle.py",
        "cascade_planner/interfaces/target_solver.py",
        "cascade_planner/interfaces/target_solver_stages.py",
    }
    allowed_service_batch_callers = {
        "cascade_planner/orchestration/retrosynthesis_service_execution.py",
        "cascade_planner/orchestration/retrosynthesis_service_planning.py",
        "cascade_planner/interfaces/chemenzy_probe.py",
        "cascade_planner/interfaces/patent_self_evolution.py",
        "cascade_planner/interfaces/aizynthfinder_sidecar.py",
        "cascade_planner/interfaces/target_solver.py",
        "cascade_planner/interfaces/target_solver_stages.py",
    }
    allowed_global_plan_callers = {
        "cascade_planner/orchestration/retrosynthesis_service_planning.py",
        "cascade_planner/interfaces/campaign_gateway.py",
        "cascade_planner/interfaces/external_strategy_import.py",
        "cascade_planner/interfaces/replay_pack.py",
        "cascade_planner/interfaces/validation_fork.py",
    }
    batch_constructors: set[str] = set()
    service_batch_callers: set[str] = set()
    global_plan_callers: set[str] = set()
    package_root = ROOT / "cascade_planner"
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(
            ("cascade_planner/legacy/", "cascade_planner/research/")
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "CanonicalIngestionBatch"
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "CanonicalIngestionBatch"
            ):
                batch_constructors.add(relative)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "apply_batch":
                service_batch_callers.add(relative)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply_global_plan"
            ):
                global_plan_callers.add(relative)

    assert batch_constructors == allowed_batch_constructors
    assert service_batch_callers == allowed_service_batch_callers
    assert global_plan_callers == allowed_global_plan_callers

    canonical_source = (
        ROOT / "cascade_planner/application/canonical_hypergraph.py"
    ).read_text(encoding="utf-8")
    canonical_tree = ast.parse(canonical_source)
    origin_assignment = next(
        node
        for node in canonical_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_ORIGIN_KINDS"
            for target in node.targets
        )
    )
    origin_kinds = set(ast.literal_eval(origin_assignment.value))
    assert {
        "codex_global_director",
        "chemenzy",
        "template",
        "self_evo_patent_template",
        "literature",
        "literature_source_route",
        "literature_visual_extraction",
        "external_strategy",
        "manual",
        "biocatalysis_hypothesis",
        "mechanism_hypothesis",
    }.issubset(origin_kinds)


def test_root_scripts_do_not_import_frozen_legacy_control_flow() -> None:
    """Keep V3 campaign owners behind explicit ``scripts/legacy`` entrypoints."""

    allowed_guard_import = {
        (
            "scripts/run_chem_enzy_plan_for_web.py",
            "cascade_planner.legacy.guard",
        ),
    }
    violations: list[str] = []
    scripts_root = ROOT / "scripts"
    for path in sorted(scripts_root.glob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            for module in imported:
                if not module.startswith("cascade_planner.legacy"):
                    continue
                if (relative, module) not in allowed_guard_import:
                    violations.append(f"{relative}->{module}")

    assert violations == []


def test_target_solver_constructs_one_runtime_and_projects_stages_read_only() -> None:
    path = ROOT / "cascade_planner/interfaces/target_solver.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    solve_target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve_target"
    )
    runtime_constructors = [
        node
        for node in ast.walk(solve_target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CampaignActionRuntime"
    ]
    anytime_calls = [
        node
        for node in ast.walk(solve_target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_anytime"
    ]
    projector = next(
        node
        for node in solve_target.body
        if isinstance(node, ast.FunctionDef) and node.name == "project_action_results"
    )
    projector_dispatch_calls = sorted(
        {
            node.func.attr
            for node in ast.walk(projector)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"execute", "execute_slice", "execute_concurrent_cohort", "run_anytime"}
        }
    )

    assert len(runtime_constructors) == 1
    assert anytime_calls
    assert {
        node.func.value.id
        for node in anytime_calls
        if isinstance(node.func.value, ast.Name)
    } == {"unified_core_runtime"}
    assert projector.args.args[1].arg == "action_kinds"
    assert projector_dispatch_calls == []


def test_ruff_legacy_exceptions_cannot_cover_v4_or_tests() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ignored_patterns = config["tool"]["ruff"]["lint"]["per-file-ignores"]

    protected_prefixes = (
        "cascade_planner/application/",
        "cascade_planner/interfaces/",
        "cascade_planner/orchestration/",
        "cascade_planner/runtime/",
        "cascade_planner/web/",
        "tests/",
    )
    assert all(
        not pattern.startswith(prefix)
        for pattern in ignored_patterns
        for prefix in protected_prefixes
    )


def test_v4_workbench_adapter_does_not_execute_legacy_route_forest_compiler() -> None:
    imports = _imports(ROOT / "cascade_planner/harness/v4_route_workbench.py")

    assert "cascade_planner.legacy.harness_runtime.route_forest" not in imports
    assert "cascade_planner.harness.route_forest_delivery" in imports


def test_live_synthesis_starts_the_self_correcting_profile() -> None:
    source = (
        ROOT / "cascade_planner/web/static/live_synthesis.html"
    ).read_text(encoding="utf-8")

    assert "execution_profile:'self_correcting_sequential'" in source
    assert "execution_profile:'paper_synthex'" not in source


def test_canonical_web_is_the_only_importable_web_application() -> None:
    imports = _imports(ROOT / "cascade_planner/web/v4_app.py")

    assert importlib.util.find_spec("cascade_planner.legacy.web") is None
    assert importlib.util.find_spec("cascade_planner.legacy.web_runtime.app") is None
    assert importlib.util.find_spec("scripts.legacy.serve_combined_web") is None
    assert (
        "cascade_planner.legacy.harness_runtime.agentic_blackboard_controller"
        not in imports
    )


def test_every_compatibility_shim_has_replacement_telemetry_and_milestone() -> None:
    inventory = compatibility_inventory()
    rows = inventory["shims"]

    assert inventory["content_sha256"] == _digest(
        {key: value for key, value in inventory.items() if key != "content_sha256"}
    )
    assert len({row["shim_id"] for row in rows}) == len(rows)
    assert all(row["scientific_write_authority"] is False for row in rows)
    assert all(row["removal_milestone"] and row["telemetry_source"] for row in rows)
    assert all(importlib.util.find_spec(row["module"]) is not None for row in rows)
    assert all(importlib.util.find_spec(row["replacement"]) is not None for row in rows)


def test_compatibility_usage_is_digest_bound_and_non_authoritative(
    tmp_path: Path,
) -> None:
    record_compatibility_use(
        tmp_path,
        "legacy.route_forest",
        callsite="architecture-test",
        metadata={"revision": 3},
    )
    row = json.loads(
        (tmp_path / ".autoplanner" / "compatibility_usage.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert row["scientific_authority"] is False
    assert row["content_sha256"] == _digest(
        {key: value for key, value in row.items() if key != "content_sha256"}
    )
