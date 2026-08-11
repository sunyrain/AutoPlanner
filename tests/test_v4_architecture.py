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
    "cascade_planner/application/biocatalytic_program_store_contracts.py",
    "cascade_planner/application/biocatalytic_program_store_replay.py",
    "cascade_planner/application/canonical_identity.py",
    "cascade_planner/application/canonical_hypergraph.py",
    "cascade_planner/application/campaign_actions.py",
    "cascade_planner/application/campaign_action_status.py",
    "cascade_planner/application/campaign_work_policy.py",
    "cascade_planner/application/campaign_trajectory.py",
    "cascade_planner/application/campaign_review_bundle.py",
    "cascade_planner/application/campaign_quality_state.py",
    "cascade_planner/application/action_scheduler.py",
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
    "cascade_planner/application/experiment_execution_contracts.py",
    "cascade_planner/application/experiment_execution_results.py",
    "cascade_planner/application/experimental_claim_adapters.py",
    "cascade_planner/application/experimental_claim_contracts.py",
    "cascade_planner/application/experimental_claim_rows.py",
    "cascade_planner/application/experimental_claim_store.py",
    "cascade_planner/application/experimental_claim_store_contracts.py",
    "cascade_planner/application/experimental_claim_store_projection.py",
    "cascade_planner/application/experimental_claim_store_replay.py",
    "cascade_planner/application/experimental_claims.py",
    "cascade_planner/application/experimental_work_frontier.py",
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
    "cascade_planner/application/reaction_template_extraction.py",
    "cascade_planner/application/reaction_template_library.py",
    "cascade_planner/application/reaction_template_store.py",
    "cascade_planner/application/reaction_condition_records.py",
    "cascade_planner/application/route_variants.py",
    "cascade_planner/application/route_execution_capabilities.py",
    "cascade_planner/application/route_execution_discovery.py",
    "cascade_planner/application/route_innovations.py",
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
    "cascade_planner/harness/v4_planned_route_branches.py",
    "cascade_planner/harness/source_condition_text.py",
    "cascade_planner/interfaces/campaign_gateway.py",
    "cascade_planner/interfaces/biocatalytic_program_gc.py",
    "cascade_planner/interfaces/campaign_experimental_claim_store.py",
    "cascade_planner/interfaces/campaign_mechanism_program_store.py",
    "cascade_planner/interfaces/campaign_program_experience.py",
    "cascade_planner/interfaces/campaign_experiment_dispatch.py",
    "cascade_planner/interfaces/campaign_experiment_gateway.py",
    "cascade_planner/interfaces/campaign_gateway_contract.py",
    "cascade_planner/interfaces/campaign_program_gateway.py",
    "cascade_planner/interfaces/campaign_program_innovation_gateway.py",
    "cascade_planner/interfaces/campaign_program_innovation_store.py",
    "cascade_planner/interfaces/campaign_program_innovations.py",
    "cascade_planner/interfaces/campaign_programs.py",
    "cascade_planner/interfaces/campaign_recovery.py",
    "cascade_planner/interfaces/campaign_recovery_stores.py",
    "cascade_planner/interfaces/experimental_claim_cli.py",
    "cascade_planner/interfaces/experimental_claim_gc.py",
    "cascade_planner/interfaces/mechanism_program_gc.py",
    "cascade_planner/interfaces/experiment_dispatch_cli.py",
    "cascade_planner/interfaces/candidate_migration.py",
    "cascade_planner/interfaces/campaign_operations.py",
    "cascade_planner/interfaces/program_cli.py",
    "cascade_planner/interfaces/program_innovation_cli.py",
    "cascade_planner/interfaces/program_gc.py",
    "cascade_planner/interfaces/program_migration.py",
    "cascade_planner/interfaces/replay_store_gc.py",
    "cascade_planner/interfaces/chemenzy_probe.py",
    "cascade_planner/interfaces/chemenzy_probe_contract.py",
    "cascade_planner/interfaces/chemenzy_probe_routes.py",
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
FOCUSED_LINE_BUDGETS = {
    "cascade_planner/application/campaign_action_status.py": 70,
    "cascade_planner/application/campaign_review_bundle.py": 340,
    "cascade_planner/application/campaign_quality_state.py": 280,
    "cascade_planner/application/unified_campaign_spec.py": 400,
    "cascade_planner/application/biocatalytic_program_contracts.py": 320,
    "cascade_planner/application/biocatalysis_validation_frontier.py": 220,
    "cascade_planner/application/biocatalytic_programs.py": 400,
    "cascade_planner/application/biocatalytic_program_store.py": 400,
    "cascade_planner/application/biocatalytic_program_store_contracts.py": 250,
    "cascade_planner/application/biocatalytic_program_store_replay.py": 250,
    "cascade_planner/application/canonical_identity.py": 200,
    "cascade_planner/application/candidate_innovation_screen.py": 230,
    "cascade_planner/application/candidate_programs.py": 470,
    "cascade_planner/application/candidate_route_observations.py": 280,
    "cascade_planner/application/capability_applicability_calibration.py": 320,
    "cascade_planner/application/compatibility_inventory.py": 260,
    "cascade_planner/application/execution_capability_feedback.py": 280,
    "cascade_planner/application/execution_program_compilation.py": 100,
    "cascade_planner/application/execution_program_route_candidates.py": 180,
    "cascade_planner/application/execution_program_validations.py": 260,
    "cascade_planner/application/execution_programs.py": 420,
    "cascade_planner/application/execution_validation_frontier.py": 230,
    "cascade_planner/application/experiment_execution_contracts.py": 250,
    "cascade_planner/application/experiment_execution_results.py": 280,
    "cascade_planner/application/experimental_claim_adapters.py": 300,
    "cascade_planner/application/experimental_claim_contracts.py": 300,
    "cascade_planner/application/experimental_claim_rows.py": 200,
    "cascade_planner/application/experimental_claim_store.py": 380,
    "cascade_planner/application/experimental_claim_store_contracts.py": 240,
    "cascade_planner/application/experimental_claim_store_projection.py": 140,
    "cascade_planner/application/experimental_claim_store_replay.py": 280,
    "cascade_planner/application/experimental_claims.py": 280,
    "cascade_planner/application/experimental_work_frontier.py": 370,
    "cascade_planner/application/frontier_runtime.py": 120,
    "cascade_planner/application/pareto.py": 80,
    "cascade_planner/application/fact_lifecycle.py": 260,
    "cascade_planner/application/mechanism_program_route_candidates.py": 180,
    "cascade_planner/application/mechanism_program_compilation.py": 100,
    "cascade_planner/application/mechanism_program_validations.py": 300,
    "cascade_planner/application/mechanism_programs.py": 420,
    "cascade_planner/application/mechanism_program_store.py": 380,
    "cascade_planner/application/mechanism_program_store_contracts.py": 260,
    "cascade_planner/application/mechanism_program_store_replay.py": 250,
    "cascade_planner/application/mechanism_experiment_feedback.py": 280,
    "cascade_planner/application/mechanism_validation_frontier.py": 190,
    "cascade_planner/application/portfolio_selection.py": 400,
    "cascade_planner/application/product_profiles.py": 100,
    "cascade_planner/application/program_route_candidates.py": 390,
    "cascade_planner/application/program_route_candidate_contracts.py": 330,
    "cascade_planner/application/program_route_candidate_factory.py": 240,
    "cascade_planner/application/program_route_optimizer.py": 280,
    "cascade_planner/application/program_innovation_contracts.py": 120,
    "cascade_planner/application/program_experience.py": 450,
    "cascade_planner/application/program_experience_store.py": 170,
    "cascade_planner/application/program_span_substitutions.py": 180,
    "cascade_planner/application/program_validation_contracts.py": 120,
    "cascade_planner/application/program_validation_feedback_contracts.py": 120,
    "cascade_planner/application/program_validation_frontier_contracts.py": 100,
    "cascade_planner/application/program_validation_routing.py": 60,
    "cascade_planner/application/reported_program_route_candidates.py": 380,
    "cascade_planner/application/proof_fact_projection.py": 220,
    "cascade_planner/application/proof_policy.py": 400,
    "cascade_planner/application/proof_portfolio.py": 400,
    "cascade_planner/application/proof_portfolio_replacements.py": 220,
    "cascade_planner/application/reaction_template_extraction.py": 360,
    "cascade_planner/application/reaction_template_examples.py": 180,
    "cascade_planner/application/reaction_template_library.py": 500,
    "cascade_planner/application/reaction_template_store.py": 170,
    "cascade_planner/application/reaction_condition_records.py": 220,
    "cascade_planner/application/route_variants.py": 400,
    "cascade_planner/application/route_execution_capabilities.py": 270,
    "cascade_planner/application/route_execution_discovery.py": 150,
    "cascade_planner/application/route_innovations.py": 400,
    "cascade_planner/application/route_innovation_chemenzy.py": 150,
    "cascade_planner/application/route_innovation_capabilities.py": 320,
    "cascade_planner/application/route_innovation_discovery.py": 400,
    "cascade_planner/application/route_innovation_windows.py": 110,
    "cascade_planner/application/route_innovation_precedents.py": 150,
    "cascade_planner/application/route_structure_matching.py": 280,
    "cascade_planner/application/route_program_dual_read.py": 400,
    "cascade_planner/application/transformation_program_store.py": 500,
    "cascade_planner/application/transformation_program_validation.py": 330,
    "cascade_planner/application/transformation_programs.py": 360,
    # Integration façades intentionally aggregate typed rows from several
    # focused modules.  Their limits prevent unbounded growth but are not the
    # same 300--500 line target used for chemistry/policy units.
    "cascade_planner/application/route_workbench.py": 900,
    "cascade_planner/application/route_workbench_closure.py": 180,
    "cascade_planner/application/route_workbench_fact_rows.py": 180,
    "cascade_planner/application/route_workbench_inspectors.py": 320,
    "cascade_planner/application/route_workbench_planned_routes.py": 220,
    "cascade_planner/application/route_workbench_proof_vectors.py": 260,
    "cascade_planner/application/route_workbench_route_rows.py": 220,
    "cascade_planner/harness/v4_route_display.py": 300,
    "cascade_planner/harness/v4_route_branch.py": 180,
    "cascade_planner/harness/v4_route_condition_projection.py": 100,
    "cascade_planner/harness/v4_route_condition_resolution.py": 160,
    "cascade_planner/harness/v4_route_evidence_projection.py": 200,
    "cascade_planner/harness/v4_route_graph_projection.py": 100,
    "cascade_planner/harness/v4_planned_route_branches.py": 400,
    "cascade_planner/harness/source_condition_extraction.py": 220,
    "cascade_planner/harness/source_condition_text.py": 180,
    "cascade_planner/harness/v4_route_workbench.py": 800,
    "cascade_planner/harness/v4_workbench_authority.py": 260,
    "cascade_planner/interfaces/campaign_gateway.py": 400,
    "cascade_planner/interfaces/biocatalytic_program_gc.py": 80,
    "cascade_planner/interfaces/campaign_experimental_claim_store.py": 80,
    "cascade_planner/interfaces/campaign_mechanism_program_store.py": 80,
    "cascade_planner/interfaces/campaign_program_experience.py": 70,
    "cascade_planner/interfaces/campaign_experiment_dispatch.py": 160,
    "cascade_planner/interfaces/campaign_experiment_gateway.py": 120,
    "cascade_planner/interfaces/campaign_gateway_contract.py": 30,
    "cascade_planner/interfaces/campaign_gateway_identity.py": 60,
    "cascade_planner/interfaces/campaign_program_gateway.py": 170,
    "cascade_planner/interfaces/campaign_program_innovation_gateway.py": 180,
    "cascade_planner/interfaces/campaign_program_innovation_store.py": 75,
    "cascade_planner/interfaces/campaign_operations.py": 160,
    "cascade_planner/interfaces/campaign_program_innovations.py": 90,
    "cascade_planner/interfaces/campaign_programs.py": 80,
    "cascade_planner/interfaces/campaign_recovery.py": 120,
    "cascade_planner/interfaces/campaign_recovery_stores.py": 50,
    "cascade_planner/interfaces/experimental_claim_cli.py": 80,
    "cascade_planner/interfaces/experimental_claim_gc.py": 60,
    "cascade_planner/interfaces/mechanism_program_gc.py": 60,
    "cascade_planner/interfaces/experiment_dispatch_cli.py": 140,
    "cascade_planner/interfaces/candidate_migration.py": 240,
    "cascade_planner/application/target_route_readiness.py": 390,
    "cascade_planner/interfaces/program_cli.py": 100,
    "cascade_planner/interfaces/program_innovation_cli.py": 120,
    "cascade_planner/interfaces/program_gc.py": 80,
    "cascade_planner/interfaces/program_migration.py": 160,
    "cascade_planner/interfaces/replay_store_gc.py": 100,
    "cascade_planner/interfaces/chemenzy_probe.py": 820,
    "cascade_planner/interfaces/chemenzy_probe_contract.py": 140,
    "cascade_planner/interfaces/chemenzy_probe_routes.py": 440,
    "cascade_planner/interfaces/epo_family_discovery.py": 240,
    "cascade_planner/interfaces/literature_evidence.py": 450,
    "cascade_planner/interfaces/literature_evidence_connector.py": 480,
    "cascade_planner/interfaces/literature_evidence_contract.py": 100,
    "cascade_planner/interfaces/literature_candidates.py": 180,
    "cascade_planner/interfaces/literature_relevance.py": 140,
    "cascade_planner/interfaces/literature_fulltext.py": 400,
    "cascade_planner/interfaces/literature_browser.py": 100,
    "cascade_planner/interfaces/literature_authorized_pdf_assets.py": 140,
    "cascade_planner/interfaces/literature_authorized_source.py": 260,
    "cascade_planner/interfaces/literature_html.py": 300,
    "cascade_planner/interfaces/literature_html_parser.py": 180,
    "cascade_planner/interfaces/literature_materialization.py": 280,
    "cascade_planner/interfaces/literature_pdf_materialization.py": 180,
    "cascade_planner/interfaces/literature_pdf_projection.py": 200,
    "cascade_planner/interfaces/literature_procedure_line_fragments.py": 140,
    "cascade_planner/interfaces/literature_procedure_fragments.py": 100,
    "cascade_planner/interfaces/visual_evidence.py": 700,
    "cascade_planner/interfaces/visual_evidence_contract.py": 140,
    "cascade_planner/interfaces/visual_evidence_materialization.py": 220,
    "cascade_planner/interfaces/visual_evidence_request.py": 520,
    "cascade_planner/interfaces/visual_observation_chemistry.py": 180,
    "cascade_planner/interfaces/visual_observation_normalization.py": 320,
    "cascade_planner/interfaces/case_dossier.py": 260,
    "cascade_planner/interfaces/case_dossier_compiler.py": 320,
    "cascade_planner/interfaces/case_dossier_contract.py": 80,
    "cascade_planner/interfaces/case_cli.py": 120,
    "cascade_planner/interfaces/case_runner.py": 100,
    "cascade_planner/interfaces/patent_self_evolution.py": 250,
    "cascade_planner/interfaces/replay_contract.py": 200,
    "cascade_planner/interfaces/replay_lifecycle.py": 100,
    "cascade_planner/interfaces/replay_observations.py": 140,
    "cascade_planner/interfaces/replay_pack.py": 500,
    "cascade_planner/interfaces/replay_reporting.py": 120,
    "cascade_planner/cli.py": 300,
    "cascade_planner/runtime/repository_audit.py": 360,
    "cascade_planner/runtime/ast_audit.py": 60,
    "cascade_planner/runtime/canonical_json.py": 50,
    "cascade_planner/runtime/immutable_event_store.py": 70,
    "cascade_planner/runtime/immutable_json_events.py": 70,
    "cascade_planner/web/v4_api.py": 360,
    "cascade_planner/web/v4_target_routes.py": 300,
    "cascade_planner/web/v4_experiment_api.py": 120,
    "cascade_planner/web/v4_program_innovation_api.py": 110,
    "cascade_planner/web/v4_program_payload.py": 80,
    "cascade_planner/web/v4_app.py": 80,
    "cascade_planner/web/security.py": 80,
    "cascade_planner/web/server.py": 80,
    "cascade_planner/web/v4_target_runtime.py": 520,
    "cascade_planner/interfaces/target_solve_request.py": 380,
    "cascade_planner/interfaces/campaign_action_timeline.py": 200,
    "cascade_planner/interfaces/target_job_projection.py": 100,
    "cascade_planner/orchestration/retrosynthesis_service.py": 430,
    "cascade_planner/orchestration/retrosynthesis_service_execution.py": 250,
    "cascade_planner/orchestration/retrosynthesis_service_planning.py": 280,
    "cascade_planner/orchestration/biocatalytic_program_admission_runtime.py": 70,
    "cascade_planner/orchestration/execution_program_review_materials.py": 110,
    "cascade_planner/orchestration/experiment_execution_runtime.py": 120,
    "cascade_planner/orchestration/experiment_dispatch_handoff.py": 120,
    "cascade_planner/orchestration/experiment_dispatch_runtime.py": 320,
    "cascade_planner/orchestration/experiment_dispatch_support.py": 260,
    "cascade_planner/orchestration/experimental_claim_admission_runtime.py": 80,
    "cascade_planner/orchestration/experimental_claim_review_materials.py": 100,
    "cascade_planner/orchestration/mechanism_program_review_materials.py": 90,
    "cascade_planner/orchestration/mechanism_program_admission_runtime.py": 80,
    "cascade_planner/orchestration/program_admission_runtime.py": 80,
    "cascade_planner/orchestration/program_candidate_review_materials.py": 100,
    "cascade_planner/orchestration/program_innovation_materials.py": 110,
    "cascade_planner/orchestration/program_innovation_runtime.py": 100,
    "cascade_planner/orchestration/program_experience_runtime.py": 70,
    "cascade_planner/orchestration/route_innovation_runtime.py": 80,
    "cascade_planner/orchestration/workbench_publication.py": 80,
    "cascade_planner/providers/experiment.py": 300,
}


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
            if relative.startswith("cascade_planner/application/") and imported.startswith(
                "cascade_planner.orchestration"
            ):
                violations.append(f"application_reverse_dependency:{relative}->{imported}")
    assert violations == []


def test_unified_action_core_has_no_dataset_specific_control_tokens() -> None:
    protected = (
        ROOT / "cascade_planner/application/campaign_actions.py",
        ROOT / "cascade_planner/application/action_scheduler.py",
        ROOT / "cascade_planner/application/run_kernel.py",
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
    w8_source = (ROOT / "scripts/run_retrostar190_w8.py").read_text(
        encoding="utf-8"
    )

    assert "--objective-mode" not in string_literals
    assert "--objective-mode" not in w8_source
    assert "--fixed-cutoff-wall-time-s" in w8_source


def test_action_runtime_cannot_write_the_canonical_graph_directly() -> None:
    imported = _imports(
        ROOT / "cascade_planner/orchestration/unified_campaign_runtime.py"
    )

    assert "cascade_planner.application.canonical_hypergraph" not in imported
    assert "cascade_planner.orchestration.retrosynthesis_service" not in imported


def test_new_focused_modules_stay_within_practical_line_budgets() -> None:
    observed = {
        relative: len((ROOT / relative).read_text(encoding="utf-8").splitlines())
        for relative in FOCUSED_LINE_BUDGETS
    }
    assert {
        relative: lines
        for relative, lines in observed.items()
        if lines > FOCUSED_LINE_BUDGETS[relative]
    } == {}


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


def test_isolated_v4_web_surface_does_not_import_combined_compatibility_app() -> None:
    imports = _imports(ROOT / "cascade_planner/web/v4_app.py")

    assert "cascade_planner.legacy.web_runtime.app" not in imports
    assert "cascade_planner.legacy.harness_runtime.agentic_blackboard_controller" not in imports


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
