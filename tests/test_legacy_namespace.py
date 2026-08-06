from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest


CCTS_EVAL_MODULES = (
    "audit_candidate_specific_evidence",
    "audit_ccts_block_supported_labels",
    "audit_ccts_label_semantics",
    "audit_ccts_v0_rank_delta",
    "build_ccts_v3_candidate_cache",
    "build_ccts_v3_runtime_evidence_cache",
    "replay_ccts_v0_transition_ranker",
    "replay_ccts_v3_on_controller_run",
    "replay_ccts_v3_on_route_pool",
    "replay_ccts_v3_runtime_on_controller_run",
    "replay_ccts_v3_runtime_on_native_pool",
    "rerank_runtime_ccts_with_product_audit",
    "summarize_ccts_v0_report",
    "train_ccts_v0_transition_ranker",
    "train_ccts_v1_transition_ranker",
    "train_ccts_v2_sparse_labels",
    "train_ccts_v2_transition_ranker",
    "train_ccts_v3_cached_ranker",
    "train_ccts_v3_candidate_evidence",
    "train_ccts_v3_pairwise_residual_ranker",
    "train_ccts_v3_runtime_pairwise_ranker",
)
CCTS_OLD_MODULES = tuple(f"cascade_planner.eval.{name}" for name in CCTS_EVAL_MODULES)
CCTS_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}" for name in CCTS_EVAL_MODULES
)
CCTS_ROUTE_MODULES = ("ccts_v0", "ccts_v3_runtime")
CCTS_OLD_ROUTE_MODULES = tuple(
    f"cascade_planner.route_tree.{name}" for name in CCTS_ROUTE_MODULES
)
CCTS_LEGACY_ROUTE_MODULES = tuple(
    f"cascade_planner.legacy.route_tree_runtime.{name}"
    for name in CCTS_ROUTE_MODULES
)
RESERVOIR_RUNTIME_OLD_MODULE = "cascade_planner.route_tree.reservoir_distilled"
RESERVOIR_RUNTIME_LEGACY_MODULE = (
    "cascade_planner.legacy.route_tree_runtime.reservoir_distilled"
)
RESERVOIR_BENCHMARK_WORKER_MODULE = (
    "cascade_planner.legacy.route_tree_runtime.live_benchmark"
)
CASCADE_ORACLE_EVAL_MODULES = (
    "build_cascade_oracle_pack",
    "build_cascade_oracle_payload",
)
CASCADE_ORACLE_OLD_EVAL_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in CASCADE_ORACLE_EVAL_MODULES
)
CASCADE_ORACLE_LEGACY_EVAL_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in CASCADE_ORACLE_EVAL_MODULES
)
CASCADE_ORACLE_OLD_ROUTE_MODULE = "cascade_planner.route_tree.cascade_oracle"
CASCADE_ORACLE_LEGACY_ROUTE_MODULE = (
    "cascade_planner.legacy.route_tree_runtime.cascade_oracle"
)
ROUTE_POOL_BLOCK_MODULES = (
    "build_route_pool_ranker_pack",
    "train_route_pool_ranker",
    "train_route_pool_lambdarank",
    "replay_route_pool_pairwise_ranker",
    "build_cascade_block_coherence_pack",
    "build_cascade_block_hard_pack",
    "train_cascade_block_coherence",
    "replay_block_coherence_on_route_pool",
    "audit_route_pool_cascade_evidence",
    "summarize_route_pool_cascade_evidence",
)
ROUTE_POOL_BLOCK_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in ROUTE_POOL_BLOCK_MODULES
)
ROUTE_POOL_BLOCK_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}" for name in ROUTE_POOL_BLOCK_MODULES
)
ROUTE_BLOCK_VALUE_MODULES = (
    "build_route_block_review_label_pack",
    "build_route_block_value_pack",
    "build_strict_model_review_worklist",
    "merge_route_block_review_labels",
    "probe_runtime_hardneg_nohuman_controls",
    "replay_route_block_value_model",
    "summarize_route_block_strengthening",
    "train_route_block_value_model",
)
ROUTE_BLOCK_VALUE_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in ROUTE_BLOCK_VALUE_MODULES
)
ROUTE_BLOCK_VALUE_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in ROUTE_BLOCK_VALUE_MODULES
)
RESERVOIR_CONTROLLER_MODULES = (
    "analyze_reservoir_student_calibration",
    "build_external_reservoir_smokes",
    "build_reservoir_distill_pack",
    "compare_controller_runs",
    "controller_v2_reports",
    "reservoir_acceptance_manifest",
    "reservoir_completion_audit",
    "reservoir_distill_matrix",
    "reservoir_publication_readiness",
    "reservoir_statistical_report",
    "train_reservoir_distilled_controller",
)
RESERVOIR_CONTROLLER_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in RESERVOIR_CONTROLLER_MODULES
)
RESERVOIR_CONTROLLER_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in RESERVOIR_CONTROLLER_MODULES
)
CBA_V0_MODULES = (
    "audit_cba_v0_route_sketch",
    "build_cba_v0_entry_substrate_benchmark",
    "build_cba_v0_guarded_sketch_pack",
    "train_cba_v0_block_applicability",
    "train_cba_v0_pair_classifier",
)
REVIEW_FALLBACK_MODULES = (
    "build_route_pool_evidence_review_prompts",
    "build_route_pool_review_calibration_packet",
    "calibrate_route_pool_evidence_review_signals",
    "export_route_pool_review_worklist",
    "gate_route_pool_evidence_review_promotion",
    "ingest_route_pool_evidence_review_csv",
    "ingest_route_pool_evidence_review_results",
    "run_route_pool_evidence_llm_review",
    "run_route_pool_evidence_review_csv_pipeline",
    "run_route_pool_evidence_review_pipeline",
    "sample_route_pool_evidence_review_batch",
    "select_route_pool_review_calibration_subset",
    "select_route_pool_review_prompt_subset",
    "summarize_route_pool_evidence_review_labels",
)
CBA_REVIEW_MODULES = CBA_V0_MODULES + REVIEW_FALLBACK_MODULES
CBA_REVIEW_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in CBA_REVIEW_MODULES
)
CBA_REVIEW_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}" for name in CBA_REVIEW_MODULES
)
CASCADE_PAIR_MODULES = (
    "build_cascade_pair_pack",
    "train_cascade_pair_scorer",
    "replay_cascade_pair_scorer",
)
CASCADE_PAIR_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in CASCADE_PAIR_MODULES
)
CASCADE_PAIR_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}" for name in CASCADE_PAIR_MODULES
)
PROVIDER_RESEARCH_MODULES = (
    "audit_cascade_retrieval_provider",
    "audit_nonoracle_provider_bridge",
    "audit_provider_chem_enzy_bridge",
    "audit_provider_routepool_oracle",
    "audit_v4_heldout_block_recovery",
    "audit_v4_template_upstream_bridge",
    "build_provider_injected_route_sketches",
    "build_v4_atommap_cache",
    "build_v4_cascade_route_pool",
    "build_v4_heldout_chem_enzy_benchmark",
    "build_v4_transform_pair_selector_pack",
    "run_v4_heldout_chem_enzy_pool",
    "train_v4_template_selector",
    "train_v4_transform_pair_selector",
)
PROVIDER_RESEARCH_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in PROVIDER_RESEARCH_MODULES
)
PROVIDER_RESEARCH_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in PROVIDER_RESEARCH_MODULES
)
ACTION_SOURCE_VALUE_MODULES = (
    "analyze_cascade_action_ranking",
    "audit_chem_enzy_transition_coverage",
    "build_cascade_action_value_pack",
    "build_cascade_transition_pack",
    "build_cascadebench_strict_splits",
    "build_route_tree_source_policy_pack",
    "build_stock_delta_source_pack",
    "build_v4_trace_benchmark",
    "build_v4_training_splits",
    "run_stage3_baseline_value_training",
    "run_v4_full_training_pipeline",
    "train_cascade_action_value",
    "train_cascade_source_policy",
    "train_cascade_source_value",
    "train_cascade_transition_value",
)
ACTION_SOURCE_VALUE_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in ACTION_SOURCE_VALUE_MODULES
)
ACTION_SOURCE_VALUE_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in ACTION_SOURCE_VALUE_MODULES
)
PRODUCT_VALUE_ARCHIVE_MODULES = (
    "build_routepool_preference_pack",
    "build_v4_cascade_case_studies",
    "build_v4_cascade_data_inventory",
    "build_v4_cascade_preference_pack",
    "build_v4_cascade_product_value_pack",
    "build_v4_cascade_publication_readiness",
    "compare_v4_cascade_rerank_reports",
    "rerank_native_routes_with_v4_value",
    "train_v4_cascade_product_value",
)
PRODUCT_VALUE_ARCHIVE_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in PRODUCT_VALUE_ARCHIVE_MODULES
)
PRODUCT_VALUE_ARCHIVE_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in PRODUCT_VALUE_ARCHIVE_MODULES
)
V4_PRODUCT_VALUE_OLD_RUNTIME_MODULE = (
    "cascade_planner.cascade_search.v4_product_value"
)
V4_PRODUCT_VALUE_LEGACY_RUNTIME_MODULE = (
    "cascade_planner.legacy.cascade_search_runtime.v4_product_value"
)
V4_PRODUCT_VALUE_FROZEN_EXPORTS = (
    "LoadedV4CascadeProductValue",
    "V4CascadeProductValueNetwork",
    "V4RoutePrediction",
    "V4_ROUTE_LABEL_NAMES",
    "build_route_feature_schema",
    "observable_value_target",
    "route_feature_vector",
    "route_label_vector",
    "route_record_from_native_route",
    "route_record_from_trace_candidate",
    "route_record_from_v4",
)
DATA_RELEASE_ARCHIVE_MODULES = (
    "augment_cascadebench_splits_with_v4_steps",
    "check_strict_review_pipeline_readiness",
    "compare_dataset_v4_release",
    "enrich_route_pool_steps_from_v4",
)
DATA_RELEASE_ARCHIVE_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in DATA_RELEASE_ARCHIVE_MODULES
)
DATA_RELEASE_ARCHIVE_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in DATA_RELEASE_ARCHIVE_MODULES
)
ROUTE_SELECTOR_RESEARCH_MODULES = (
    "audit_routepool_context_controls",
    "audit_selector_regression_cases",
    "build_route_pool_selector_pack",
    "compare_same_pool_route_selectors",
    "gate_phase_selector_promotion",
    "rerank_cascade_only_features_with_product_audit",
    "train_route_selector_v0",
)
ROUTE_SELECTOR_RESEARCH_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in ROUTE_SELECTOR_RESEARCH_MODULES
)
ROUTE_SELECTOR_RESEARCH_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in ROUTE_SELECTOR_RESEARCH_MODULES
)
CASCADE_SUBGOAL_RESEARCH_MODULES = (
    "audit_cascade_subgoal_discovery",
    "train_cascade_subgoal_scorer",
)
CASCADE_SUBGOAL_RESEARCH_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in CASCADE_SUBGOAL_RESEARCH_MODULES
)
CASCADE_SUBGOAL_RESEARCH_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in CASCADE_SUBGOAL_RESEARCH_MODULES
)
PHASE2_FRAGMENT_ARCHIVE_MODULES = (
    "audit_phase2_block_readiness",
    "audit_phase2_direction_guardrails",
    "build_cascade_fragment_pack",
    "evaluate_cascadebench_block",
    "summarize_cascadebench_phase2",
    "train_cascade_fragment_scorer",
    "validate_fragment_rerank",
    "verify_cascadebench_phase2_closure",
)
PHASE2_FRAGMENT_ARCHIVE_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in PHASE2_FRAGMENT_ARCHIVE_MODULES
)
PHASE2_FRAGMENT_ARCHIVE_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in PHASE2_FRAGMENT_ARCHIVE_MODULES
)
OBSOLETE_AIZ_BENCHMARK_MODULES = (
    "multistep_solvebench",
    "multistep_solvebench_hard",
    "run_benchmark_v2_100",
    "summarize_benchmark_v2_100",
)
OBSOLETE_AIZ_BENCHMARK_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in OBSOLETE_AIZ_BENCHMARK_MODULES
)
OBSOLETE_AIZ_BENCHMARK_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in OBSOLETE_AIZ_BENCHMARK_MODULES
)
RETIRED_EXPAND_TRAINING_MODULES = (
    "enzexpand_ablation",
    "per_ec1_conditions",
    "train_chemical_template_pair_ranker",
)
RETIRED_EXPAND_TRAINING_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in RETIRED_EXPAND_TRAINING_MODULES
)
RETIRED_EXPAND_TRAINING_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in RETIRED_EXPAND_TRAINING_MODULES
)
LEGACY_V2_EXTERNAL_AUDIT_MODULES = (
    "aggregate_external_smoke_summaries",
    "analyze_student_route_composition_gaps",
    "audit_cascade_full100_result",
    "audit_condition_data_coverage",
    "audit_route_pool_review_transform_sanity",
    "audit_skeleton_retrieval_prior",
    "audit_stock_closed_alternatives",
    "benchmark_overlap_audit",
    "build_cascade_gold_smoke",
    "build_locked_validation",
    "candidate_miss_audit",
    "cc_aostar_depth_benchmark",
    "compare_cascade_search_runs",
    "compare_chem_enzy_baseline",
    "eval_aizynthfinder",
    "freeze_benchmark",
    "generator_bottleneck",
    "gt_direct_candidate_recall",
    "run_pipeline_manifest_commands",
    "select_condition_rich_benchmark",
    "stock_failure_audit",
    "summarize",
    "syntheseus_eval",
    "uniprot_evidence_summary",
    "uspto50k_aggregate",
    "uspto50k_benchmark",
    "uspto50k_syntheseus",
)
LEGACY_V2_EXTERNAL_AUDIT_OLD_MODULES = tuple(
    f"cascade_planner.eval.{name}" for name in LEGACY_V2_EXTERNAL_AUDIT_MODULES
)
LEGACY_V2_EXTERNAL_AUDIT_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in LEGACY_V2_EXTERNAL_AUDIT_MODULES
)
LEGACY_REPORT_CARD_OLD_MODULES = (
    "cascade_planner.eval.condition_diagnosis",
    "cascade_planner.eval.hybrid_multi_audited",
    "cascade_planner.cascadeboard.report_card",
)
LEGACY_REPORT_CARD_LEGACY_MODULES = (
    "cascade_planner.legacy.eval_runtime.condition_diagnosis",
    "cascade_planner.legacy.eval_runtime.hybrid_multi_audited",
    "cascade_planner.legacy.eval_runtime.report_card",
)
LEGACY_DUAL_TOWER_MODULES = ("dual_tower_candidates",)
LEGACY_DUAL_TOWER_OLD_MODULES = tuple(
    f"cascade_planner.cascadeboard.{name}" for name in LEGACY_DUAL_TOWER_MODULES
)
LEGACY_DUAL_TOWER_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}" for name in LEGACY_DUAL_TOWER_MODULES
)
LEGACY_CASCADEBOARD_BENCHMARK_MODULES = (
    "ablation_benchmark",
    "benchmarks",
    "candidate_supervision",
    "constraint_benchmark",
    "counterfactual_benchmark",
    "data_audit",
    "integrated_benchmark",
    "policy_benchmark",
    "preference_dataset",
    "real_benchmark",
)
LEGACY_CASCADEBOARD_BENCHMARK_OLD_MODULES = tuple(
    f"cascade_planner.cascadeboard.{name}"
    for name in LEGACY_CASCADEBOARD_BENCHMARK_MODULES
)
LEGACY_CASCADEBOARD_BENCHMARK_LEGACY_MODULES = tuple(
    f"cascade_planner.legacy.eval_runtime.{name}"
    for name in LEGACY_CASCADEBOARD_BENCHMARK_MODULES
)
LEGACY_LEARNED_SCORER_OLD_MODULE = "cascade_planner.cascadeboard.learned_scorer"
LEGACY_LEARNED_SCORER_MODULE = "cascade_planner.legacy.eval_runtime.learned_scorer"
LEGACY_CASCADEBOARD_RUNTIME_OLD_MODULES = (
    "cascade_planner.cascadeboard.cli",
    "cascade_planner.cascadeboard.inpaint_planner",
    "cascade_planner.cascadeboard.planner",
    "cascade_planner.cascadeboard.train",
    "cascade_planner.cascadeboard.training_data",
)
LEGACY_CASCADEBOARD_RUNTIME_MODULES = (
    "cascade_planner.legacy.eval_runtime.cascadeboard_cli",
    "cascade_planner.legacy.eval_runtime.cascadeboard_inpaint_planner",
    "cascade_planner.legacy.eval_runtime.cascadeboard_planner",
    "cascade_planner.legacy.eval_runtime.train_cascadeboard",
    "cascade_planner.legacy.eval_runtime.cascadeboard_training_data",
)
LEGACY_CASCADEBOARD_GRAPH_OLD_MODULES = (
    "cascade_planner.cascadeboard.cached_candidate_graph",
    "cascade_planner.cascadeboard.candidate_graph",
    "cascade_planner.cascadeboard.lazy_expansion",
    "cascade_planner.cascadeboard.preference",
)
LEGACY_CASCADEBOARD_GRAPH_MODULES = (
    "cascade_planner.legacy.eval_runtime.cascadeboard_cached_candidate_graph",
    "cascade_planner.legacy.eval_runtime.cascadeboard_candidate_graph",
    "cascade_planner.legacy.eval_runtime.cascadeboard_lazy_expansion",
    "cascade_planner.legacy.eval_runtime.cascadeboard_preference",
)
LEGACY_CASCADEBOARD_STOCK_OLD_MODULES = (
    "cascade_planner.cascadeboard.chemical_anchor_stock",
    "cascade_planner.cascadeboard.semisynthesis_stock",
)
LEGACY_CASCADEBOARD_STOCK_MODULES = (
    "cascade_planner.legacy.cascadeboard_runtime.chemical_anchor_stock",
    "cascade_planner.legacy.cascadeboard_runtime.semisynthesis_stock",
)
LEGACY_PATH_OLD_MODULE = "cascade_planner.paths"
LEGACY_PATH_MODULE = "cascade_planner.legacy.paths"
OLD_LEGACY_GUARD_MODULE = "cascade_planner.legacy_guard"
LEGACY_GUARD_MODULE = "cascade_planner.legacy.guard"
LEGACY_CASCADE_BENCHMARK_MODULE = (
    "cascade_planner.legacy.eval_runtime.run_cascade_search_benchmark"
)
LEGACY_CASCADE_SEARCH_OLD_MODULES = (
    "cascade_planner.cascade_search.action_value_contract",
    "cascade_planner.cascade_search.pair_scorer",
    "cascade_planner.cascade_search.transition_value",
)
LEGACY_CASCADE_SEARCH_RUNTIME_MODULES = (
    "cascade_planner.legacy.cascade_search_runtime.action_value_contract",
    "cascade_planner.legacy.cascade_search_runtime.pair_scorer",
    "cascade_planner.legacy.cascade_search_runtime.transition_value",
)
FROZEN_CASCADE_SEARCH_EXPORTS = (
    "LearnedCascadePairScorer",
    "LearnedCascadeValueModel",
    "LoadedCascadeActionValueModel",
    "LoadedCascadeTransitionValueModel",
    "RuleCascadePairScorer",
    "RuleCascadeValueModel",
)
OLD_ENZYMATIC_RETRIEVAL_MODULE = "cascade_planner.cascadeboard.enz_retrieval"
CURRENT_ENZYMATIC_RETRIEVAL_MODULE = "cascade_planner.route_tree.enzymatic_retrieval"
LEGACY_EVAL_DATA_ASSETS = (
    "benchmark_condition_rich_20260507.json",
    "benchmark_gt_intermediate_seen_not_expanded_20260510.json",
    "benchmark_locked_validation_20260508.json",
    "benchmark_pairwise_stress_20260510.json",
    "statin_smoke_2026-05-05.json",
)


def test_default_public_surfaces_exclude_frozen_v3_exports() -> None:
    application = importlib.import_module("cascade_planner.application")
    harness = importlib.import_module("cascade_planner.harness")
    orchestration = importlib.import_module("cascade_planner.orchestration")
    providers = importlib.import_module("cascade_planner.providers")
    provider_builtins = importlib.import_module("cascade_planner.providers.builtins")

    assert "FrontierScheduler" not in application.__all__
    assert "run_agentic_blackboard_controller" not in harness.__all__
    assert "run_codex_retrosynthesis_campaign" not in orchestration.__all__
    assert "CodexRetrosynthesisProvider" not in providers.__all__
    assert not hasattr(provider_builtins, "CodexRetrosynthesisProvider")


def test_route_forest_layout_does_not_own_shared_digest_alias() -> None:
    layout = importlib.import_module("cascade_planner.harness.route_forest_layout")
    canonical_json = importlib.import_module("cascade_planner.runtime.canonical_json")

    assert not hasattr(layout, "canonical_sha256")
    assert callable(canonical_json.canonical_json_sha256)


def test_mainline_case_commands_exclude_solve_case_alias() -> None:
    case_cli = importlib.import_module("cascade_planner.interfaces.case_cli")

    assert "replay-dossier" in case_cli.CASE_COMMANDS
    assert "solve-case" not in case_cli.CASE_COMMANDS


@pytest.mark.parametrize(
    ("module_name", "attribute_name"),
    [
        ("cascade_planner.application", "RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA"),
        ("cascade_planner.harness", "run_agentic_blackboard_controller"),
        ("cascade_planner.orchestration", "run_codex_retrosynthesis_campaign"),
        ("cascade_planner.providers", "CodexRetrosynthesisProvider"),
        ("cascade_planner.runtime", "publish_closeout_revision"),
        ("cascade_planner.routes", "assemble_route_consensus_graph"),
    ],
)
def test_old_package_root_aliases_are_deleted(
    module_name: str,
    attribute_name: str,
) -> None:
    module = importlib.import_module(module_name)

    with pytest.raises(AttributeError):
        getattr(module, attribute_name)


def test_explicit_legacy_namespace_retains_saved_run_compatibility() -> None:
    application = importlib.import_module("cascade_planner.legacy.application")
    providers = importlib.import_module("cascade_planner.legacy.providers")

    with pytest.warns(DeprecationWarning, match="frozen V3 compatibility"):
        schema = application.RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA

    assert schema == "retrosynthesis_acceptance_report.v1"
    assert providers.CodexRetrosynthesisProvider.__module__ == (
        "cascade_planner.legacy.providers"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "cascade_planner.application.frontier_scheduler",
        "cascade_planner.application.frontier_ledger",
        "cascade_planner.application.route_deficit_queue",
        "cascade_planner.application.route_portfolio",
        "cascade_planner.application.retrosynthesis_acceptance",
        "cascade_planner.orchestration.codex_retrosynthesis",
        "cascade_planner.harness.agentic_blackboard_controller",
        "cascade_planner.harness.agent_action_planner",
        "cascade_planner.harness.codex_action_planner",
        "cascade_planner.harness.agentic_blackboard",
        "cascade_planner.harness.blackboard_events",
        "cascade_planner.harness.tools",
        "cascade_planner.harness.route_forest",
        "cascade_planner.harness.runner",
        "cascade_planner.harness.retrosynthetic_proposals",
        "cascade_planner.harness.codex_edge_verification",
        "cascade_planner.harness.parent_route_proof",
        "cascade_planner.harness.route_objectives",
        "cascade_planner.harness.target_side_strategy",
        "cascade_planner.harness.analogical_reaction_templates",
        "cascade_planner.harness.analogical_retrosynthesis",
        "cascade_planner.harness.codex_plan",
        "cascade_planner.harness.failure_critic",
        "cascade_planner.harness.hypothesis_execution_report",
        "cascade_planner.harness.hypothetical_retrosynthesis_report",
        "cascade_planner.harness.preflight",
        "cascade_planner.harness.process_evidence",
        "cascade_planner.harness.progress",
        "cascade_planner.harness.recursive_hypothesis_tasks",
        "cascade_planner.harness.v4_controller_adapter",
        "cascade_planner.harness.self_evo_memory",
        "cascade_planner.harness.self_evo_replay",
        "cascade_planner.harness.tool_registry",
        "cascade_planner.harness.tool_execution_policy",
        "cascade_planner.application.selected_route_parent_proof",
        "cascade_planner.runtime.artifact_revision",
        "cascade_planner.routes.signatures",
        "cascade_planner.harness.schemas",
        "cascade_planner.routes.graph",
        "cascade_planner.harness.visual_structure_extraction",
        "cascade_planner.web.app",
        "scripts.audit_architecture_v2",
        "scripts.run_codex_entry_agentic_blackboard",
        "scripts.run_codex_entry_controller",
        "scripts.resume_agentic_blackboard",
        "scripts.run_codex_entry_pdf_visual_followup",
        "scripts.render_route_forest",
        "scripts.refresh_agentic_closeout_artifacts",
        "scripts.validate_legacy_example_runs",
        "scripts.smoke_route_forest_history",
        "scripts.migrate_codex_campaign_v2",
        "scripts.run_fresh_agentic_smoke",
        "scripts.validate_example_runs",
        "scripts.benchmark_nirmatrelvir_v3",
        "scripts.run_nirmatrelvir_v3_golden",
        "scripts.evaluate_agentic_run",
        "scripts.render_blackboard_timeline",
        "cascade_planner.routes.adapters",
        "cascade_planner.routes.admission_receipts",
        "cascade_planner.orchestration.admitted_hyperedges",
    ],
)
def test_old_v3_module_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CCTS_OLD_MODULES)
def test_old_ccts_eval_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CCTS_LEGACY_MODULES)
def test_ccts_eval_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", CCTS_OLD_ROUTE_MODULES)
def test_old_ccts_route_tree_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CCTS_LEGACY_ROUTE_MODULES)
def test_ccts_route_tree_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


def test_reservoir_distilled_runtime_is_legacy_only() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(RESERVOIR_RUNTIME_OLD_MODULE)

    assert importlib.util.find_spec(RESERVOIR_RUNTIME_LEGACY_MODULE) is not None
    assert importlib.util.find_spec(RESERVOIR_BENCHMARK_WORKER_MODULE) is not None


@pytest.mark.parametrize("module_name", CASCADE_ORACLE_OLD_EVAL_MODULES)
def test_old_cascade_oracle_eval_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CASCADE_ORACLE_LEGACY_EVAL_MODULES)
def test_cascade_oracle_eval_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


def test_cascade_oracle_route_runtime_is_legacy_only() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(CASCADE_ORACLE_OLD_ROUTE_MODULE)

    assert importlib.util.find_spec(CASCADE_ORACLE_LEGACY_ROUTE_MODULE) is not None


@pytest.mark.parametrize("module_name", ROUTE_POOL_BLOCK_OLD_MODULES)
def test_old_route_pool_block_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ROUTE_POOL_BLOCK_LEGACY_MODULES)
def test_route_pool_block_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", ROUTE_BLOCK_VALUE_OLD_MODULES)
def test_old_route_block_value_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ROUTE_BLOCK_VALUE_LEGACY_MODULES)
def test_route_block_value_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", RESERVOIR_CONTROLLER_OLD_MODULES)
def test_old_reservoir_controller_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", RESERVOIR_CONTROLLER_LEGACY_MODULES)
def test_reservoir_controller_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", CBA_REVIEW_OLD_MODULES)
def test_old_cba_review_fallback_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CBA_REVIEW_LEGACY_MODULES)
def test_cba_review_fallback_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", CASCADE_PAIR_OLD_MODULES)
def test_old_cascade_pair_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CASCADE_PAIR_LEGACY_MODULES)
def test_cascade_pair_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", PROVIDER_RESEARCH_OLD_MODULES)
def test_old_provider_research_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", PROVIDER_RESEARCH_LEGACY_MODULES)
def test_provider_research_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", ACTION_SOURCE_VALUE_OLD_MODULES)
def test_old_action_source_value_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ACTION_SOURCE_VALUE_LEGACY_MODULES)
def test_action_source_value_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", PRODUCT_VALUE_ARCHIVE_OLD_MODULES)
def test_old_product_value_archive_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", PRODUCT_VALUE_ARCHIVE_LEGACY_MODULES)
def test_product_value_archive_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


def test_v4_product_value_runtime_is_legacy_only() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(V4_PRODUCT_VALUE_OLD_RUNTIME_MODULE)

    assert importlib.util.find_spec(V4_PRODUCT_VALUE_LEGACY_RUNTIME_MODULE) is not None


def test_active_cascade_search_excludes_v4_product_value_exports() -> None:
    module = importlib.import_module("cascade_planner.cascade_search")

    for name in V4_PRODUCT_VALUE_FROZEN_EXPORTS:
        assert name not in module.__all__
        with pytest.raises(AttributeError):
            getattr(module, name)


def test_subgoal_contract_uses_current_identity_helpers() -> None:
    module = importlib.import_module(
        "cascade_planner.cascade_search.subgoal_evidence_contract"
    )

    assert module.stable_id.__module__ == "cascade_planner.cascade_search.ids"
    assert (
        module.canonical_smiles.__module__
        == "cascade_planner.cascadeboard.route_recovery"
    )


@pytest.mark.parametrize("module_name", DATA_RELEASE_ARCHIVE_OLD_MODULES)
def test_old_data_release_archive_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", DATA_RELEASE_ARCHIVE_LEGACY_MODULES)
def test_data_release_archive_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", ROUTE_SELECTOR_RESEARCH_OLD_MODULES)
def test_old_route_selector_research_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ROUTE_SELECTOR_RESEARCH_LEGACY_MODULES)
def test_route_selector_research_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", CASCADE_SUBGOAL_RESEARCH_OLD_MODULES)
def test_old_cascade_subgoal_research_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CASCADE_SUBGOAL_RESEARCH_LEGACY_MODULES)
def test_cascade_subgoal_research_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", PHASE2_FRAGMENT_ARCHIVE_OLD_MODULES)
def test_old_phase2_fragment_archive_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", PHASE2_FRAGMENT_ARCHIVE_LEGACY_MODULES)
def test_phase2_fragment_archive_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", OBSOLETE_AIZ_BENCHMARK_OLD_MODULES)
def test_old_aiz_benchmark_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", OBSOLETE_AIZ_BENCHMARK_LEGACY_MODULES)
def test_aiz_benchmark_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", RETIRED_EXPAND_TRAINING_OLD_MODULES)
def test_old_retired_expand_training_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", RETIRED_EXPAND_TRAINING_LEGACY_MODULES)
def test_retired_expand_training_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", LEGACY_V2_EXTERNAL_AUDIT_OLD_MODULES)
def test_old_v2_external_audit_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_V2_EXTERNAL_AUDIT_LEGACY_MODULES)
def test_v2_external_audit_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", LEGACY_REPORT_CARD_OLD_MODULES)
def test_old_report_card_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_REPORT_CARD_LEGACY_MODULES)
def test_report_card_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", LEGACY_DUAL_TOWER_OLD_MODULES)
def test_old_dual_tower_adapter_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_DUAL_TOWER_LEGACY_MODULES)
def test_dual_tower_adapter_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_BENCHMARK_OLD_MODULES)
def test_old_cascadeboard_benchmark_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_BENCHMARK_LEGACY_MODULES)
def test_cascadeboard_benchmark_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


def test_learned_route_scorer_is_legacy_only() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(LEGACY_LEARNED_SCORER_OLD_MODULE)

    assert importlib.util.find_spec(LEGACY_LEARNED_SCORER_MODULE) is not None


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_RUNTIME_OLD_MODULES)
def test_old_cascadeboard_cli_and_training_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_RUNTIME_MODULES)
def test_cascadeboard_cli_and_training_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_GRAPH_OLD_MODULES)
def test_old_cascadeboard_graph_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_GRAPH_MODULES)
def test_cascadeboard_graph_legacy_paths_are_discoverable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_STOCK_OLD_MODULES)
def test_unused_cascadeboard_stock_old_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_CASCADEBOARD_STOCK_MODULES)
def test_unused_cascadeboard_stock_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


def test_old_v2_results_path_helper_is_deleted() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(LEGACY_PATH_OLD_MODULE)


def test_legacy_guard_is_owned_by_legacy_namespace() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(OLD_LEGACY_GUARD_MODULE)
    guard = importlib.import_module(LEGACY_GUARD_MODULE)
    assert guard.LEGACY_RESEARCH_ENV == "AUTOPLANNER_ALLOW_LEGACY_RESEARCH"


def test_run_manifest_compatibility_writer_is_legacy_only(tmp_path: Path) -> None:
    active = importlib.import_module("cascade_planner.runtime.run_storage")
    package = importlib.import_module("cascade_planner.runtime")
    legacy = importlib.import_module(
        "cascade_planner.legacy.runtime.run_manifest_compatibility"
    )

    assert not hasattr(active, "write_run_manifest_compatibility")
    assert not hasattr(package, "write_run_manifest_compatibility")
    target = legacy.write_run_manifest_compatibility(
        tmp_path / "run_manifest.json",
        {"run_id": "legacy-run", "revision": 3},
    )
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "revision": 3,
        "run_id": "legacy-run",
    }

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "legacy"
        / "benchmark_nirmatrelvir_v3.py"
    ).read_text(encoding="utf-8")
    assert "cascade_planner.legacy.runtime.run_manifest_compatibility" in script
    assert "write_run_manifest_compatibility" not in active.__all__


def test_v2_results_path_helper_is_legacy_only() -> None:
    assert importlib.util.find_spec(LEGACY_PATH_MODULE) is not None


def test_current_enzymatic_retrieval_is_owned_by_route_tree() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(OLD_ENZYMATIC_RETRIEVAL_MODULE)

    module = importlib.import_module(CURRENT_ENZYMATIC_RETRIEVAL_MODULE)
    assert callable(module.retrieve_enzymatic_reactions)


def test_parallel_benchmark_sets_the_current_v3_retrieval_switch(monkeypatch) -> None:
    parallel = importlib.import_module("cascade_planner.eval.run_live_benchmark_parallel")
    monkeypatch.delenv("AUTOPLANNER_ENABLE_V3_RETRIEVAL_PROPOSALS", raising=False)
    monkeypatch.delenv("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL", raising=False)
    parser = parallel.build_parser()
    args = parser.parse_args(
        ["--output", "ignored.json", "--enable-route-tree-v3-retrieval"]
    )

    env = parallel._base_env(args, None)

    assert env["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] == "1"
    assert "AUTOPLANNER_ENABLE_V3_RETRIEVAL_PROPOSALS" not in env
    assert "--enable-v3-retrieval-proposals" not in parser._option_string_actions


def test_frozen_cascade_benchmark_adapters_are_legacy_only() -> None:
    active = importlib.import_module("cascade_planner.eval.run_cascade_search_benchmark")
    frozen_parameters = {
        "cascade_value_model_path",
        "cascade_transition_model_path",
        "cascade_action_value_model_path",
        "cascade_pair_scorer_path",
        "route_block_value_final_reranker_path",
        "use_chem_enzy_cascade_cost",
        "use_chem_enzy_cascade_source_policy",
    }
    frozen_options = {
        "--cascade-value-model",
        "--cascade-transition-model",
        "--cascade-action-value-model",
        "--cascade-pair-scorer",
        "--route-block-value-final-reranker",
        "--chem-enzy-cascade-cost",
        "--chem-enzy-cascade-source-policy",
    }

    assert frozen_parameters.isdisjoint(
        inspect.signature(active.run_cascade_search_benchmark).parameters
    )
    assert frozen_options.isdisjoint(active.build_parser()._option_string_actions)
    assert importlib.util.find_spec(LEGACY_CASCADE_BENCHMARK_MODULE) is not None

    pipeline_path = (
        Path(__file__).resolve().parents[1]
        / "cascade_planner"
        / "legacy"
        / "eval_runtime"
        / "run_v4_full_training_pipeline.py"
    )
    pipeline_source = pipeline_path.read_text(encoding="utf-8")
    assert LEGACY_CASCADE_BENCHMARK_MODULE in pipeline_source
    assert "cascade_planner.eval.run_cascade_search_benchmark" not in pipeline_source


@pytest.mark.parametrize("module_name", LEGACY_CASCADE_SEARCH_OLD_MODULES)
def test_frozen_cascade_search_runtime_old_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_CASCADE_SEARCH_RUNTIME_MODULES)
def test_frozen_cascade_search_runtime_legacy_paths_are_discoverable(
    module_name: str,
) -> None:
    assert importlib.util.find_spec(module_name) is not None


def test_active_cascade_action_value_is_hint_only() -> None:
    active = importlib.import_module("cascade_planner.cascade_search.action_value")
    legacy = importlib.import_module(
        "cascade_planner.legacy.cascade_search_runtime.action_value"
    )

    assert callable(active.SubgoalHintActionScorer)
    assert not hasattr(active, "LoadedCascadeActionValueModel")
    assert callable(legacy.LoadedCascadeActionValueModel)
    assert not hasattr(legacy, "SubgoalHintActionScorer")


def test_active_cascade_search_package_excludes_frozen_runtime_exports() -> None:
    module = importlib.import_module("cascade_planner.cascade_search")

    assert "SubgoalHintActionScorer" in module.__all__
    for name in FROZEN_CASCADE_SEARCH_EXPORTS:
        assert name not in module.__all__
        with pytest.raises(AttributeError):
            getattr(module, name)


def test_learned_cascade_value_adapter_is_legacy_only() -> None:
    active = importlib.import_module("cascade_planner.cascade_search.value")
    search = importlib.import_module("cascade_planner.cascade_search.search")
    legacy = importlib.import_module(
        "cascade_planner.legacy.cascade_search_runtime.value"
    )

    assert not hasattr(active, "LearnedCascadeValueModel")
    assert callable(legacy.LearnedCascadeValueModel)
    assert legacy.LearnedCascadeValueModel.__module__ == (
        "cascade_planner.legacy.cascade_search_runtime.value"
    )
    assert active.LoadedLearnedVerifierValueModel.is_learned_value_model is True
    assert legacy.LearnedCascadeValueModel.is_learned_value_model is True
    controller = type(
        "Controller",
        (),
        {"value_model": legacy.LearnedCascadeValueModel},
    )()
    assert search._learned_value_active(controller) is True


def test_obsolete_rule_value_alias_is_deleted() -> None:
    active = importlib.import_module("cascade_planner.cascade_search.value")
    package = importlib.import_module("cascade_planner.cascade_search")

    assert not hasattr(active, "RuleCascadeValueModel")
    assert not hasattr(package, "RuleCascadeValueModel")


def test_legacy_eval_data_assets_are_outside_active_data_root() -> None:
    root = Path(__file__).resolve().parents[1]
    archive_root = root / "archive" / "data" / "legacy_eval_202605"

    for name in LEGACY_EVAL_DATA_ASSETS:
        assert not (root / "data" / name).exists()
        assert (archive_root / name).is_file()


def test_legacy_eval_cli_entrypoints_require_explicit_opt_in() -> None:
    root = Path(__file__).resolve().parents[1] / "cascade_planner" / "legacy" / "eval_runtime"
    missing_guard = []
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if 'if __name__ == "__main__":' not in source:
            continue
        if "require_legacy_research_enabled" not in source:
            missing_guard.append(path.name)

    assert missing_guard == []


def test_active_eval_package_has_no_legacy_research_guards() -> None:
    root = Path(__file__).resolve().parents[1] / "cascade_planner" / "eval"
    guarded = [
        path.name
        for path in sorted(root.glob("*.py"))
        if "require_legacy_research_enabled"
        in path.read_text(encoding="utf-8")
    ]

    assert guarded == []


def test_active_eval_defaults_do_not_reference_legacy_results_v2() -> None:
    root = Path(__file__).resolve().parents[1] / "cascade_planner" / "eval"
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        if "results/v2" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_active_cascadeboard_defaults_do_not_reference_legacy_results_v2() -> None:
    root = Path(__file__).resolve().parents[1] / "cascade_planner" / "cascadeboard"
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        if "results/v2" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_ccts_route_tree_runtime_requires_legacy_research_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPLANNER_ALLOW_LEGACY_RESEARCH", raising=False)
    monkeypatch.setenv("AUTOPLANNER_CCTS_V0_MODEL", "archived-checkpoint.pt")

    from cascade_planner.legacy.route_tree_runtime import (
        ccts_runtime_from_env,
        plan_with_legacy_ccts,
    )

    with pytest.raises(SystemExit, match="archived/frozen research code"):
        ccts_runtime_from_env()
    with pytest.raises(SystemExit, match="archived/frozen research code"):
        plan_with_legacy_ccts()


def test_active_route_tree_uses_explicit_ccts_injection_only() -> None:
    active = importlib.import_module("cascade_planner.route_tree.search")
    source = Path(active.__file__).read_text(encoding="utf-8")

    assert "ccts_scorer" in inspect.signature(active.NeuralGuidedAOSearch).parameters
    assert "ccts_scorer" in inspect.signature(active.plan_with_route_tree).parameters
    assert not hasattr(active, "_ccts_runtime_from_env")
    assert "AUTOPLANNER_CCTS_V0_MODEL" not in source
    assert "AUTOPLANNER_CCTS_V3_RUNTIME_MODEL" not in source
    assert "AUTOPLANNER_ROUTE_TREE_CCTS_WEIGHT" not in source
    assert "cascade_planner.legacy" not in source


def test_active_route_tree_uses_explicit_reservoir_injection_only() -> None:
    search = importlib.import_module("cascade_planner.route_tree.search")
    extensions = importlib.import_module("cascade_planner.route_tree.extensions")
    root = Path(__file__).resolve().parents[1] / "cascade_planner" / "route_tree"
    runtime_source = (root / "runtime.py").read_text(encoding="utf-8")
    source_gate_source = (root / "source_gate.py").read_text(encoding="utf-8")

    assert "source_gate" in inspect.signature(search.NeuralGuidedAOSearch).parameters
    assert "source_gate" in inspect.signature(search.plan_with_route_tree).parameters
    assert "source_gate_factory" in extensions.RouteTreeExtensions.__dataclass_fields__
    assert "AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER" not in runtime_source
    assert "AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER" not in source_gate_source
    assert "reservoir_distilled" not in runtime_source
    assert "reservoir_distilled" not in source_gate_source
    assert "cascade_planner.legacy" not in runtime_source
    assert "cascade_planner.legacy" not in source_gate_source


def test_active_route_tree_uses_explicit_action_value_advisor_only() -> None:
    search = importlib.import_module("cascade_planner.route_tree.search")
    extensions = importlib.import_module("cascade_planner.route_tree.extensions")
    source = Path(search.__file__).read_text(encoding="utf-8")

    assert "action_value_advisor" in inspect.signature(
        search.NeuralGuidedAOSearch
    ).parameters
    assert "action_value_advisor" in inspect.signature(
        search.plan_with_route_tree
    ).parameters
    assert "action_value_advisor_factory" in (
        extensions.RouteTreeExtensions.__dataclass_fields__
    )
    assert "AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE" not in source
    assert "AUTOPLANNER_CASCADE_ORACLE_PAYLOAD" not in source
    assert "AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT" not in source
    assert "cascade_oracle" not in source
    assert "cascade_planner.legacy" not in source


def test_legacy_reservoir_worker_builds_explicit_extensions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTOPLANNER_ALLOW_LEGACY_RESEARCH", "1")
    monkeypatch.setenv(
        "AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER",
        str(tmp_path / "missing-controller.pt"),
    )
    monkeypatch.setenv("AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER", "0")
    monkeypatch.delenv("AUTOPLANNER_CASCADE_SOURCE_POLICY", raising=False)
    monkeypatch.delenv("AUTOPLANNER_SOURCE_GATE", raising=False)
    monkeypatch.delenv("AUTOPLANNER_ENABLE_BRIDGE_SOURCE_GATE", raising=False)

    worker = importlib.import_module(RESERVOIR_BENCHMARK_WORKER_MODULE)
    route_tree_extensions = worker.build_route_tree_extensions()

    assert route_tree_extensions.controller_factory is not None
    assert route_tree_extensions.source_gate_factory is not None
    controller = route_tree_extensions.controller_factory()
    source_gate = route_tree_extensions.source_gate_factory()
    assert type(controller).__module__ == RESERVOIR_RUNTIME_LEGACY_MODULE
    assert type(source_gate).__module__ == RESERVOIR_RUNTIME_LEGACY_MODULE
    assert controller.reason == "missing_checkpoint"
    assert source_gate.reason == "missing_checkpoint"


def test_legacy_worker_builds_cascade_oracle_as_generic_advisor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "cascade-oracle.json"
    payload_path.write_text(
        json.dumps({"schema_version": "cascade_oracle_payload.v1", "targets": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTOPLANNER_ALLOW_LEGACY_RESEARCH", "1")
    monkeypatch.delenv("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER", raising=False)
    monkeypatch.setenv("AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE", "1")
    monkeypatch.setenv("AUTOPLANNER_CASCADE_ORACLE_PAYLOAD", str(payload_path))
    monkeypatch.setenv("AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT", "1.25")

    worker = importlib.import_module(RESERVOIR_BENCHMARK_WORKER_MODULE)
    route_tree_extensions = worker.build_route_tree_extensions()

    assert route_tree_extensions.controller_factory is None
    assert route_tree_extensions.source_gate_factory is None
    assert route_tree_extensions.action_value_advisor_factory is not None
    assert route_tree_extensions.action_value_advisor_weight == pytest.approx(1.25)
    advisor = route_tree_extensions.action_value_advisor_factory()
    assert type(advisor).__module__ == CASCADE_ORACLE_LEGACY_ROUTE_MODULE


def test_reservoir_command_generators_select_the_legacy_worker() -> None:
    root = Path(__file__).resolve().parents[1] / "cascade_planner" / "legacy" / "eval_runtime"
    worker_option = f"--worker-module {RESERVOIR_BENCHMARK_WORKER_MODULE}"

    for name in (
        "build_external_reservoir_smokes.py",
        "reservoir_acceptance_manifest.py",
    ):
        assert worker_option in (root / name).read_text(encoding="utf-8")

    external_source = (root / "build_external_reservoir_smokes.py").read_text(
        encoding="utf-8"
    )
    assert (
        "python -m cascade_planner.legacy.eval_runtime.build_cascade_oracle_payload"
        in external_source
    )
    assert "python -m cascade_planner.eval.build_cascade_oracle_payload" not in (
        external_source
    )


def test_v4_imports_do_not_load_legacy_state_owners() -> None:
    forbidden = {
        "cascade_planner.legacy",
        "cascade_planner.research",
    }
    script = f"""
import importlib
import json
import sys

importlib.import_module("cascade_planner.orchestration.retrosynthesis_service")
importlib.import_module("cascade_planner.interfaces.target_solver")
v4_app = importlib.import_module("cascade_planner.web.v4_app")
v4_app.create_v4_app(lambda: None)
print(json.dumps(sorted(set({sorted(forbidden)!r}) & set(sys.modules))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
