"""Target-only, bounded V4 retrosynthesis campaign orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Event
import time
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

from cascade_planner.application.blind_acceptance import (
    compile_blind_acceptance_report,
)
from cascade_planner.application.action_scheduler import (
    ACTION_SCHEDULER_POLICIES,
    schedule_next_action,
)
from cascade_planner.application.blind_benchmark_contract import (
    BLIND_CASE_SCHEMA,
    BlindBenchmarkError,
    BlindCase,
    audit_blind_preflight,
    canonical_smiles,
)
from cascade_planner.application.campaign_context import CampaignContextTooLargeError
from cascade_planner.application.campaign_quality_state import (
    compile_campaign_quality_state,
)
from cascade_planner.application.candidate_lifecycle import (
    compile_candidate_lifecycle,
)
from cascade_planner.application.candidate_provenance import (
    compile_candidate_provenance,
)
from cascade_planner.application.canonical_hypergraph import (
    CanonicalIngestionBatch,
    molecule_identity,
)
from cascade_planner.application.campaign_actions import (
    CampaignAction,
    CampaignActionKind,
    compile_action_opportunities,
)
from cascade_planner.application.campaign_trajectory import (
    compile_action_counts,
    compile_campaign_snapshot,
    compile_campaign_trajectory,
    compile_route_snapshot,
    compile_trajectory_bindings,
    snapshots_from_stages,
)
from cascade_planner.application.experimental_work_scheduling import (
    experimental_work_item_rank_key,
    experimental_work_item_scheduling,
)
from cascade_planner.application.experimental_claim_contracts import (
    EXPERIMENTAL_CLAIM_SET_ORACLE_SCHEMA,
    validate_experimental_claim_set,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.route_innovation_discovery import (
    canonical_innovation_batch,
)
from cascade_planner.application.run_kernel import RunKernelBudgetError
from cascade_planner.application.unified_campaign_spec import (
    CampaignResourceBudget,
    StockOracleReference,
    TargetConstraints,
    UnifiedCampaignSpec,
    stock_oracle_reference_from_builder,
)
from cascade_planner.application.reaction_mapping import ReactionMapper
from cascade_planner.application.proof_portfolio import (
    PortfolioConfig,
    compile_proof_portfolio,
)
from cascade_planner.application.paper_equivalent_metric import (
    compile_paper_equivalent_metric,
)
from cascade_planner.application.guided_search_progress import (
    compile_parent_route_stock_progress,
    evaluate_guided_stock_progress,
)
from cascade_planner.application.program_opportunity_pressure import (
    compile_program_opportunity_pressure,
    compile_program_review_pressure,
)
from cascade_planner.application.replan_pressure import compile_replan_pressure
from cascade_planner.cascade_search.proposals import local_condition_predictor
from cascade_planner.interfaces.evidence_import import (
    ingest_structured_evidence_document,
)
from cascade_planner.interfaces.chemenzy_probe import (
    ChemenzyProposalProvider,
    run_chemenzy_guided_frontier_stage,
    run_chemenzy_proposal_stage,
)
from cascade_planner.interfaces.live_evidence import (
    EvidenceConnector,
    LiveEvidenceConnectorError,
    acquire_structured_evidence,
    compile_evidence_acquisition_request,
)
from cascade_planner.interfaces.live_stock import build_pubchem_vendor_catalog
from cascade_planner.interfaces.target_runtime_dependencies import (
    SYNTHEX_MATCHED_PROFILE_DEFAULTS,
)
from cascade_planner.interfaces.patent_self_evolution import (
    PatentSelfEvolutionSession,
)
from cascade_planner.interfaces.target_solver_stages import (
    InventorySnapshotBuilder,
    StockCatalogBuilder,
    audit_authoritative_inventory_stock,
    audit_live_benchmark_stock,
    commit_materialized_edge_validation,
    discover_director_source_hints,
    enrich_materialized_edge_conditions,
    ingest_source_discovery_observation,
    materialize_discovered_source_routes,
    prepare_materialized_edge_validation,
    project_existing_stock_audit,
    repair_rejected_precursor_typos,
    validate_materialized_edges,
)
from cascade_planner.interfaces.target_identity import (
    TARGET_IDENTITY_PROVIDER_VERSION,
    resolve_target_identity,
)
from cascade_planner.interfaces.target_solver_compat import (
    TARGET_SOLVE_CHECKPOINT_SCHEMA,
    TargetObjectiveMode,
    build_target_resume_cursor,
    classify_target_resume_work,
    compile_program_validation_feedback_signals,
    compile_saved_run_objective_compatibility,
    compile_target_claim_projection as _claim,
    compile_target_solver_checkpoint,
    validate_target_objective_mode,
)
from cascade_planner.interfaces.visual_evidence import (
    VisualEvidenceProvider,
    acquire_visual_evidence_candidates,
    materialize_visual_evidence_candidates,
    rebind_visual_evidence_observation,
)
from cascade_planner.orchestration.global_campaign_director import (
    ACTIONABLE_REPLAN_EVENTS,
    DirectorConfig,
    DirectorOutcome,
    DirectorRunner,
    GlobalCampaignDirectorError,
    director_prompt,
    run_codex_cli_director_child,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    SequentialStrategyDirectorRunner,
)
from cascade_planner.orchestration.unified_campaign_runtime import (
    CampaignActionDeferredHandler,
    CampaignActionRuntime,
)


if TYPE_CHECKING:
    from cascade_planner.interfaces.campaign_gateway import CampaignGateway


TARGET_SOLVE_REPORT_SCHEMA = "target_only_retrosynthesis_solve_report.v1"
DEFAULT_TARGET_DIRECTOR_MODEL = str(SYNTHEX_MATCHED_PROFILE_DEFAULTS["model"])
_DIRECTOR_TOPOLOGY_REASONS = frozenset(
    {
        "skeleton_ancestor_cycle",
        "skeleton_contains_disconnected_steps",
        "skeleton_product_expanded_more_than_once",
        "skeleton_requires_exactly_one_target_root",
    }
)
_MAX_DIRECTOR_OUTCOMES = 10


@dataclass(frozen=True, slots=True)
class TargetSolveConfig:
    model: str = DEFAULT_TARGET_DIRECTOR_MODEL
    reasoning_effort: str = str(SYNTHEX_MATCHED_PROFILE_DEFAULTS["reasoning_effort"])
    execution_profile: str = "standard"
    strategy_search_profile: str = str(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["strategy_search_profile"]
    )
    strategy_branch_count: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["strategy_branches"]
    )
    max_node_expansions_per_branch: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["node_expansions_per_branch"]
    )
    max_route_local_repair_rounds: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["route_local_repair_rounds"]
    )
    max_node_prompt_bytes: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_node_prompt_bytes"]
    )
    max_node_call_timeout_s: float = float(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["node_call_timeout_s"]
    )
    critic_call_timeout_s: float = float(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["critic_call_timeout_s"]
    )
    # Legacy/direct callers remain compatible by default.  In the paper
    # profile this means the final plan must expose a host-assembled complete
    # RouteJSON; individual Route Builder calls still emit one edit program
    # and are compiled incrementally by the host.
    require_complete_route_json: bool = False
    allow_editor_route_mutations: bool = False
    objective_mode: TargetObjectiveMode = "scientific_proof"
    use_coordinator: bool = False
    enable_web_search: bool = True
    enable_initial_director_web_search: bool = False
    enable_codex: bool = True
    enable_replan: bool = True
    action_scheduler_policy: str = "adaptive"
    delivery_boundary: str = "full"
    enable_live_benchmark_stock: bool = True
    enable_builtin_patent_evidence: bool = False
    enable_patent_self_evolution: bool = True
    enable_chemenzy: bool = True
    enable_target_chemenzy_baseline: bool = bool(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["target_chemenzy_baseline"]
    )
    enable_guided_chemenzy: bool = True
    enable_chemenzy_condition_prediction: bool = True
    enable_condition_enrichment: bool = True
    enable_chemenzy_enzyme_assignment: bool = True
    enable_enzyme_coverage_sidecar: bool = True
    enable_program_review: bool = True
    enable_program_admission: bool = False
    enable_program_discovery: bool = True
    enable_program_validation: bool = True
    enable_experimental_claim_admission: bool = False
    enable_target_identity: bool = True
    resolve_named_target_identity: bool = False
    blind_audit_root: str = ""
    blind_audit_allowed_paths: tuple[str, ...] = ()
    chemenzy_env_prefix: str = ""
    chemenzy_stock_names: tuple[str, ...] = ()
    chemenzy_stock_paths: tuple[tuple[str, str], ...] = ()
    self_evo_library_path: str = ""
    program_capability_catalog_path: str = ""
    max_atom_mapping_reactions: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_atom_mapping_reactions"]
    )
    max_condition_prediction_reactions: int = 48
    condition_prediction_topk: int = 2
    max_live_stock_molecules: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_stock_molecules"]
    )
    max_patent_sources: int = 3
    max_self_evo_template_candidates: int = 12
    max_program_routes: int = 4
    max_total_tasks: int = int(SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_total_tasks"])
    max_evidence_tasks: int = 64
    max_stock_tasks: int = 128
    max_validation_tasks: int = 128
    max_program_tasks: int = 64
    max_experiment_tasks: int = 32
    max_run_wall_time_s: float = 7_200.0
    provider_route_reserve: int = 16
    host_route_portfolio: int = 16
    display_route_limit: int = 4
    max_chemenzy_routes: int | None = None
    max_chemenzy_steps: int = 6
    max_chemenzy_iterations: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_iterations"]
    )
    chemenzy_expansion_topk: int = 20
    chemenzy_timeout_s: float = float(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_timeout_s"]
    )
    chemenzy_search_preset: str = "standard"
    chemenzy_seed: int = 0
    chemenzy_pandarallel_workers: int = 2
    max_guided_chemenzy_frontiers: int | None = None
    max_guided_chemenzy_iterations: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_iterations"]
    )
    guided_chemenzy_timeout_s: float = float(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_timeout_s"]
    )
    max_visual_evidence_pages: int = 6
    minimum_planning_route_steps: int = 0
    max_director_output_tokens: int = 18_000
    max_director_wall_time_s: float = float(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_model_wall_time_s"]
    )
    publish_intermediate_workbenches: bool = True
    schema_version: str = "target_solve_config.v1"

    def __post_init__(self) -> None:
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("target solver reasoning effort is invalid")
        if self.execution_profile not in {"fast", "standard", "proof", "paper_synthex"}:
            raise ValueError("target solver execution profile is invalid")
        if self.strategy_search_profile not in {"legacy_global", "synthex_matched"}:
            raise ValueError("target solver strategy search profile is invalid")
        if not 1 <= self.strategy_branch_count <= 8:
            raise ValueError("target solver strategy branch count is invalid")
        if not 1 <= self.max_node_expansions_per_branch <= 64:
            raise ValueError("target solver node expansion limit is invalid")
        if not 1 <= self.max_route_local_repair_rounds <= 12:
            raise ValueError("target solver route-local repair limit is invalid")
        if self.execution_profile == "paper_synthex":
            if self.max_node_expansions_per_branch != int(
                SYNTHEX_MATCHED_PROFILE_DEFAULTS["route_builder_max_steps"]
            ):
                raise ValueError(
                    "paper_synthex requires 25 Route Builder expansions per branch"
                )
            if self.max_route_local_repair_rounds != int(
                SYNTHEX_MATCHED_PROFILE_DEFAULTS["route_local_repair_rounds"]
            ):
                raise ValueError(
                    "paper_synthex requires six Critic/Editor repair rounds"
                )
            if not self.require_complete_route_json:
                raise ValueError("paper_synthex requires complete linear RouteJSON")
            if not self.allow_editor_route_mutations:
                raise ValueError("paper_synthex requires RouteJSON Editor mutations")
        if not 4_000 <= self.max_node_prompt_bytes <= 96_000:
            raise ValueError("target solver compact node prompt limit is invalid")
        for value in (self.max_node_call_timeout_s, self.critic_call_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("target solver call timeout is invalid")
        if self.action_scheduler_policy not in ACTION_SCHEDULER_POLICIES:
            raise ValueError("target solver action scheduler policy is invalid")
        if self.delivery_boundary not in {"full", "stock_result"}:
            raise ValueError("target solver delivery boundary is invalid")
        validate_target_objective_mode(self.objective_mode)
        if self.max_atom_mapping_reactions < 1 or self.max_live_stock_molecules < 1:
            raise ValueError("target solver deterministic limits must be positive")
        if self.max_condition_prediction_reactions < 1:
            raise ValueError("target solver condition prediction limit must be positive")
        if not 1 <= self.condition_prediction_topk <= 2:
            raise ValueError("target solver condition prediction top-k is invalid")
        if not 1 <= self.max_patent_sources <= 8:
            raise ValueError("target solver patent source limit is invalid")
        if not 1 <= self.max_visual_evidence_pages <= 12:
            raise ValueError("target solver visual evidence page limit is invalid")
        if not 1 <= self.max_self_evo_template_candidates <= 64:
            raise ValueError("target solver self-evolution candidate limit is invalid")
        if not 1 <= self.max_program_routes <= 8:
            raise ValueError("target solver Program route limit is invalid")
        task_limits = (
            self.max_total_tasks,
            self.max_evidence_tasks,
            self.max_stock_tasks,
            self.max_validation_tasks,
            self.max_program_tasks,
            self.max_experiment_tasks,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in task_limits):
            raise ValueError("target solver task limits are invalid")
        if not math.isfinite(float(self.max_run_wall_time_s)) or (self.max_run_wall_time_s < 0):
            raise ValueError("target solver run wall-time limit is invalid")
        if not 1 <= self.provider_route_reserve <= 32:
            raise ValueError("target solver ChemEnzy provider reserve is invalid")
        if not 1 <= self.host_route_portfolio <= 16:
            raise ValueError("target solver ChemEnzy host portfolio is invalid")
        if not 1 <= self.display_route_limit <= 12:
            raise ValueError("target solver route display limit is invalid")
        if self.max_chemenzy_routes is not None and not (1 <= self.max_chemenzy_routes <= 32):
            raise ValueError("target solver legacy ChemEnzy route limit is invalid")
        if (
            min(
                self.max_chemenzy_steps,
                self.max_chemenzy_iterations,
                self.chemenzy_expansion_topk,
            )
            < 1
            or self.chemenzy_timeout_s <= 0
        ):
            raise ValueError("target solver ChemEnzy budget is invalid")
        if (
            isinstance(self.chemenzy_seed, bool)
            or not isinstance(self.chemenzy_seed, int)
            or not 0 <= self.chemenzy_seed <= 2**32 - 1
        ):
            raise ValueError("target solver ChemEnzy seed is invalid")
        if self.chemenzy_search_preset not in {
            "quick",
            "standard",
            "thorough",
            "enzyme_coverage",
        }:
            raise ValueError("target solver ChemEnzy search preset is invalid")
        if not 1 <= self.chemenzy_pandarallel_workers <= 8:
            raise ValueError("target solver ChemEnzy worker count is invalid")
        if self.max_guided_chemenzy_frontiers is not None and (
            isinstance(self.max_guided_chemenzy_frontiers, bool)
            or self.max_guided_chemenzy_frontiers < 1
        ):
            raise ValueError("target solver guided ChemEnzy frontier limit is invalid")
        if self.max_guided_chemenzy_iterations < 1 or self.guided_chemenzy_timeout_s <= 0:
            raise ValueError("target solver guided ChemEnzy budget is invalid")
        if self.max_director_output_tokens < 1 or self.max_director_wall_time_s <= 0:
            raise ValueError("target solver director limits must be positive")
        if (
            isinstance(self.minimum_planning_route_steps, bool)
            or not isinstance(self.minimum_planning_route_steps, int)
            or not 0 <= self.minimum_planning_route_steps <= 24
        ):
            raise ValueError("target solver minimum planning route depth is invalid")

    @property
    def effective_provider_route_reserve(self) -> int:
        """Resolve the deprecated one-knob route limit without losing compatibility."""

        return int(self.max_chemenzy_routes or self.provider_route_reserve)

    @property
    def effective_host_route_portfolio(self) -> int:
        return min(
            int(self.host_route_portfolio),
            self.effective_provider_route_reserve,
        )


def solve_target(
    gateway: "CampaignGateway",
    *,
    target_name: str,
    target_smiles: str,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    acceptance: RetrosynthesisAcceptanceSpec | None = None,
    budget: RetrosynthesisRunBudget | None = None,
    constraints: TargetConstraints | None = None,
    stock_oracle_reference: StockOracleReference | None = None,
    config: TargetSolveConfig | None = None,
    manifest_path: str | Path | None = None,
    resume: bool = False,
    director_runner: DirectorRunner | None = None,
    atom_mapper: ReactionMapper | None = None,
    stock_catalog_builder: StockCatalogBuilder | None = None,
    inventory_snapshot_builder: InventorySnapshotBuilder | None = None,
    evidence_connector: EvidenceConnector | None = None,
    visual_evidence_provider: VisualEvidenceProvider | None = None,
    chemenzy_provider: ChemenzyProposalProvider | None = None,
    condition_predictor: Any | None = None,
    program_capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    program_validation_feedback: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run or resume the real SMILES-only campaign path through one V4 kernel."""

    active = config or TargetSolveConfig()
    canonical = canonical_smiles(target_smiles)
    if not canonical:
        raise BlindBenchmarkError("blind_target_smiles_invalid")
    resolved_acceptance = acceptance or RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=2,
        minimum_edge_proof_level=2,
        require_all_selected_leaves_stock_closed=True,
        stock_boundary="benchmark_search",
        minimum_independent_source_groups=2,
        require_distinct_edge_sets=True,
    )
    matched = SYNTHEX_MATCHED_PROFILE_DEFAULTS
    requested_budget = budget or RetrosynthesisRunBudget(
        max_model_invocations=matched["max_model_invocations"],
        max_total_input_tokens=matched["max_input_tokens"],
        max_total_output_tokens=matched["max_output_tokens"],
        max_total_wall_time_s=matched["max_model_wall_time_s"],
        max_visual_invocations=0,
        max_accepted_expansions=matched["max_accepted_expansions"],
        max_attempt_runs=matched["max_attempt_runs"],
        max_prompt_context_bytes=matched["max_prompt_context_bytes"],
    )
    resolved_budget = _bind_native_search_budget(
        requested_budget,
        config=active,
    )
    guided_frontier_limit = int(
        resolved_budget.max_frontier_native_search_invocations or 0
    )
    oracle_builder: Any | None = None
    if resolved_acceptance.stock_boundary == "benchmark_search":
        if stock_catalog_builder is not None:
            oracle_builder = stock_catalog_builder
        elif active.enable_live_benchmark_stock:
            oracle_builder = build_pubchem_vendor_catalog
    elif inventory_snapshot_builder is not None:
        oracle_builder = inventory_snapshot_builder
    resolved_stock_oracle = stock_oracle_reference
    if resolved_stock_oracle is None:
        resolved_stock_oracle = (
            stock_oracle_reference_from_builder(
                oracle_builder,
                boundary=resolved_acceptance.stock_boundary,
            )
            if oracle_builder is not None
            else gateway._default_stock_oracle_reference(
                boundary=resolved_acceptance.stock_boundary
            )
        )
    if resolved_stock_oracle.boundary != resolved_acceptance.stock_boundary:
        raise BlindBenchmarkError("stock_oracle_acceptance_boundary_conflict")
    resource_budget = CampaignResourceBudget(
        model=resolved_budget,
        max_total_tasks=active.max_total_tasks,
        max_evidence_tasks=active.max_evidence_tasks,
        max_stock_tasks=active.max_stock_tasks,
        max_validation_tasks=active.max_validation_tasks,
        max_program_tasks=active.max_program_tasks,
        max_experiment_tasks=active.max_experiment_tasks,
        max_run_wall_time_s=active.max_run_wall_time_s,
    )
    resolved_campaign_spec = UnifiedCampaignSpec(
        target_smiles=canonical,
        stock_oracle=resolved_stock_oracle,
        constraints=constraints or TargetConstraints(),
        resource_budget=resource_budget,
    )
    supplied_name = " ".join(str(target_name or "").split())
    opaque_name = f"target-{hashlib.sha256(canonical.encode()).hexdigest()[:8]}"
    campaign_name = (
        supplied_name
        if supplied_name.casefold()
        not in {"", "blind target", "blind molecule", "opaque target", "target", "unknown target"}
        else opaque_name
    )
    identity = gateway._normalize_run_id(run_id or gateway._new_run_id(campaign_name, canonical))
    directory = gateway._run_dir(identity, explicit=run_dir, require=False)
    case = BlindCase.from_dict(
        {
            "schema_version": BLIND_CASE_SCHEMA,
            "case_id": (
                re.sub(r"[^A-Za-z0-9._-]+", "-", identity).strip(".-")[:80]
                or f"blind-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
            ),
            "target_name": campaign_name,
            "target_smiles": canonical,
            "acceptance": _acceptance_input(resolved_acceptance),
            "budget": _budget_input(resolved_budget),
        }
    )
    checkpoint_path = directory / ".autoplanner" / "target-solver-checkpoint.json"
    preflight_path = directory / ".autoplanner" / "blind-preflight.json"
    target_report_path = directory / "target-only-solve-report.json"
    existing = (directory / ".autoplanner" / "kernel" / "run_spec.json").is_file()
    prior_saved_report: dict[str, Any] = {}
    if existing and not resume:
        raise BlindBenchmarkError("blind_run_exists_use_resume")
    if existing:
        checkpoint = _read_checkpoint(checkpoint_path)
        preflight = _read_json_object(preflight_path, "blind_preflight_missing_on_resume")
        if target_report_path.is_file():
            prior_saved_report = _read_json_object(
                target_report_path,
                "target_solve_report_missing_on_resume",
            )
        if (
            dict(preflight.get("case") or {}).get("target_smiles") != canonical
            or preflight.get("accepted") is not True
        ):
            raise BlindBenchmarkError("blind_resume_preflight_binding_invalid")
    else:
        checkpoint = _empty_checkpoint(identity)
        preflight = audit_blind_preflight(
            case,
            repository_root=(
                Path(active.blind_audit_root).expanduser().resolve()
                if active.blind_audit_root
                else gateway.paths.repository_root
            ),
            run_dir=directory,
            manifest_path=manifest_path,
            additional_allowed_paths=active.blind_audit_allowed_paths,
        )
        if preflight.get("accepted") is not True:
            raise BlindBenchmarkError(";".join(preflight.get("reasons") or []))
        gateway.create_run(
            target_name=case.target_name,
            target_smiles=case.target_smiles,
            run_id=identity,
            run_dir=directory,
            acceptance=resolved_acceptance,
            budget=resolved_budget,
            campaign_spec=resolved_campaign_spec,
        )
        _write_json_atomic(preflight_path, preflight)
        _write_json_atomic(checkpoint_path, checkpoint)

    director_profile = {
        "fast": {
            "minimum_route_families": 2,
            "max_route_families": 2,
            "max_skeletons": 2,
            "max_steps_per_skeleton": 5,
            "max_output_tokens": 3_800,
            "max_tool_calls": 4,
        },
        "standard": {
            "minimum_route_families": 3,
            "max_route_families": 4,
            "max_skeletons": 4,
            "max_steps_per_skeleton": 8,
            "max_output_tokens": 7_000,
            "max_tool_calls": 12,
        },
        "proof": {
            "minimum_route_families": 3,
            "max_route_families": 4,
            "max_skeletons": 4,
            "max_steps_per_skeleton": 24,
            "max_output_tokens": 18_000,
            "max_tool_calls": 16,
        },
        "paper_synthex": {
            "minimum_route_families": 3,
            "max_route_families": 4,
            "max_skeletons": 4,
            "max_steps_per_skeleton": 25,
            "max_output_tokens": 18_000,
            "max_tool_calls": 16,
        },
    }[active.execution_profile]
    minimum_director_families = max(
        director_profile["minimum_route_families"],
        resolved_acceptance.minimum_complete_routes,
    )
    if active.minimum_planning_route_steps > director_profile["max_steps_per_skeleton"]:
        raise ValueError("minimum planning route depth exceeds execution profile capacity")
    result_delivery_cancel_event = Event()
    # A caller-supplied runner is an explicit implementation override.  Only
    # grant the matched profile's six host-validated local-repair rounds when
    # the actual runner implements the compact sequential policy; legacy or
    # test global planners retain the one-event replan contract.
    sequential_strategy = bool(
        active.strategy_search_profile == "synthex_matched"
        and (
            director_runner is None
            or isinstance(director_runner, SequentialStrategyDirectorRunner)
        )
    )
    director_config = DirectorConfig(
        minimum_route_families=minimum_director_families,
        minimum_planning_route_steps=active.minimum_planning_route_steps,
        max_route_families=max(
            director_profile["max_route_families"],
            minimum_director_families,
            active.strategy_branch_count if sequential_strategy else 0,
            active.max_route_local_repair_rounds if sequential_strategy else 0,
        ),
        max_skeletons=max(
            director_profile["max_skeletons"],
            minimum_director_families,
            active.strategy_branch_count if sequential_strategy else 0,
            active.max_route_local_repair_rounds if sequential_strategy else 0,
        ),
        max_steps_per_skeleton=max(
            director_profile["max_steps_per_skeleton"],
            active.max_node_expansions_per_branch if sequential_strategy else 0,
        ),
        max_output_tokens=min(
            active.max_director_output_tokens,
            director_profile["max_output_tokens"],
            resolved_budget.max_total_output_tokens,
        ),
        max_wall_time_s=min(
            active.max_director_wall_time_s,
            resolved_budget.max_total_wall_time_s,
        ),
        max_tool_calls=director_profile["max_tool_calls"],
        max_initial_architecture_calls=1,
        max_event_replan_calls=(
            active.max_route_local_repair_rounds
            if sequential_strategy
            else _MAX_DIRECTOR_OUTCOMES - 1
        ),
        max_final_portfolio_synthesis_calls=1,
        planning_mode=("sequential_branches" if sequential_strategy else "global_skeleton"),
        strategy_branch_count=active.strategy_branch_count,
        max_node_expansions_per_branch=active.max_node_expansions_per_branch,
        max_route_local_repair_rounds=active.max_route_local_repair_rounds,
        max_node_prompt_bytes=active.max_node_prompt_bytes,
        max_node_call_timeout_s=active.max_node_call_timeout_s,
        critic_call_timeout_s=active.critic_call_timeout_s,
        max_provider_requests=max(3, guided_frontier_limit),
        model=active.model,
        reasoning_effort=active.reasoning_effort,
        enable_web_search=active.enable_web_search,
        enable_initial_web_search=active.enable_initial_director_web_search,
        use_coordinator=active.use_coordinator,
        require_strategy_graph_edits=sequential_strategy,
        require_complete_route_json=(
            active.require_complete_route_json and sequential_strategy
        ),
        allow_editor_route_mutations=(
            active.allow_editor_route_mutations and sequential_strategy
        ),
    )
    resolved_director_runner = director_runner
    if resolved_director_runner is None:
        if sequential_strategy:
            resolved_director_runner = SequentialStrategyDirectorRunner(
                stock_membership=_frozen_stock_membership_checker(
                    stock_catalog_builder
                ),
                cancel_event=result_delivery_cancel_event,
            )
        else:
            def resolved_director_runner(
                spec: Any,
                context: Any,
                mode: str,
                config: DirectorConfig,
            ) -> Any:
                return run_codex_cli_director_child(
                    spec,
                    context,
                    mode,
                    config,
                    cancel_event=result_delivery_cancel_event,
                )

    service = gateway._open(
        identity,
        run_dir=directory,
        director_runner=resolved_director_runner,
        director_config=director_config,
    )
    budget_extension_event = None
    if resume and service.kernel.spec.limits.model != resolved_budget:
        budget_sha256 = str(resolved_budget.to_dict()["content_sha256"])
        budget_extension_event = service.kernel.extend_model_budget(
            resolved_budget,
            idempotency_key=f"solve-target:model-budget:{budget_sha256[:24]}",
        )
    if resume and service.kernel.state.status == "paused":
        service.kernel.resume(
            idempotency_key=f"solve-target:resume:{service.kernel.state.revision}"
        )
    if service.kernel.state.status == "running" and not (
        active.enable_chemenzy and active.enable_target_chemenzy_baseline
    ):
        native_projection = service.kernel.native_search_budget()
        protected_units = int(
            dict(native_projection.get("target") or {}).get("protected_remaining") or 0
        )
        if protected_units:
            service.kernel.release_native_target_reserve(
                units=protected_units,
                reason="target_native_search_not_registered",
                idempotency_key="solve-target:native-target-reserve:unregistered",
            )
    service.apply_batch(
        CanonicalIngestionBatch(recompute_derived=True),
        idempotency_key=f"solve-target:derived-projection:{service.kernel.state.graph_revision}",
    )
    resumed_completed_checkpoint = bool(resume and checkpoint.get("complete") is True)
    continuation_baseline = _automatic_continuation_baseline(service)
    self_evo = PatentSelfEvolutionSession.create(
        enabled=active.enable_patent_self_evolution,
        configured_path=active.self_evo_library_path,
        external_data_root=gateway.paths.external_data_root,
        target_smiles=canonical,
        max_candidates=active.max_self_evo_template_candidates,
    )
    resolved_evidence_connector = evidence_connector
    if resolved_evidence_connector is None and active.enable_builtin_patent_evidence:
        from cascade_planner.interfaces.patent_evidence import (
            BuiltinPatentEvidenceConfig,
            build_builtin_patent_evidence_connector,
        )

        resolved_evidence_connector = build_builtin_patent_evidence_connector(
            BuiltinPatentEvidenceConfig(
                cache_dir=gateway.paths.external_data_root / "patent-evidence",
                max_patents=active.max_patent_sources,
                max_validated_edges=active.max_atom_mapping_reactions,
            )
        )
    stages = list(checkpoint.get("stages") or [])
    outcomes = list(checkpoint.get("director_outcomes") or [])
    attempted_guided_frontier_smiles = _attempted_chemenzy_frontiers(stages)
    if resume:
        stages.append(
            _stage(
                "saved_run_objective_compatibility",
                "observed",
                compile_saved_run_objective_compatibility(
                    checkpoint,
                    prior_saved_report,
                    requested_objective_mode=active.objective_mode,
                ),
            )
        )
    resolved_condition_predictor = condition_predictor
    condition_predictor_error = ""
    if resolved_condition_predictor is None and active.enable_condition_enrichment:
        try:
            resolved_condition_predictor = local_condition_predictor(
                _chemenzy_vendor_root(gateway.paths.vendor_root), "rcr"
            )
        except Exception as exc:
            condition_predictor_error = f"{type(exc).__name__}:{exc}"
    resolved_program_capabilities, program_capability_error = (
        _resolve_program_capabilities(
            program_capabilities,
            configured_path=active.program_capability_catalog_path,
            repository_root=gateway.paths.repository_root,
        )
        if active.enable_program_discovery
        else ((), "")
    )
    resolved_mechanism_proposals = tuple(
        dict(value) for value in mechanism_proposals if isinstance(value, Mapping)
    )
    resolved_program_validation_feedback = tuple(
        dict(value) for value in program_validation_feedback if isinstance(value, Mapping)
    )
    resolved_feedback_signals = compile_program_validation_feedback_signals(
        resolved_program_validation_feedback
    )
    initial_director_context: Any | None = None
    latest_campaign_portfolio: dict[str, Any] = {}
    guided_progress_pending: dict[str, Any] = {}
    guided_continuation_allowed = True

    def append_condition_stage(stage_name: str) -> dict[str, Any]:
        result_first_deferred = bool(
            active.action_scheduler_policy == "adaptive"
            and _campaign_milestones(current_campaign_gates()).get(
                "B4_stock_boundary"
            )
            is not True
        )
        if result_first_deferred:
            detail = {
                "stage": "condition_enrichment",
                "status": "deferred",
                "reason": "result_first_stock_boundary_not_reached",
                "semantics": {"scheduler_owned_execution": True},
            }
        elif not active.enable_condition_enrichment:
            detail = {
                "stage": "condition_enrichment",
                "status": "skipped",
                "reason": "condition_enrichment_disabled",
                "semantics": {"scheduler_owned_execution": True},
            }
        else:
            condition_executions = project_action_results(
                stage_name,
                (CampaignActionKind.CONDITION_ENRICH,),
                max_actions=active.max_condition_prediction_reactions + 2,
            )
            detail = _aggregate_condition_action_results(condition_executions)
        if condition_predictor_error:
            detail["predictor_load_error"] = condition_predictor_error
        stages.append(_stage(stage_name, detail["status"], detail))
        return detail

    def scheduler_resources() -> dict[str, bool]:
        model_totals = dict(service.kernel.state.model_totals)
        native_search = service.kernel.native_search_budget()
        task_budget = dict(service.kernel.task_budget().get("dimensions") or {})
        target_native = dict(native_search.get("target") or {})
        frontier_native = dict(native_search.get("frontier") or {})
        return {
            "deterministic": True,
            "validation": dict(task_budget.get("validation") or {}).get("available") is True,
            "stock": dict(task_budget.get("stock") or {}).get("available") is True,
            "evidence": bool(
                resolved_evidence_connector is not None
                and dict(task_budget.get("evidence") or {}).get("available") is True
            ),
            "condition": bool(
                active.enable_condition_enrichment and resolved_condition_predictor is not None
            ),
            "program": dict(task_budget.get("program") or {}).get("available") is True,
            "experiment": dict(task_budget.get("experiment") or {}).get("available") is True,
            "model": int(model_totals.get("model_invocations") or 0)
            < resolved_budget.max_model_invocations,
            "native_search_target": bool(
                active.enable_chemenzy
                and active.enable_target_chemenzy_baseline
                and target_native.get("available") is True
            ),
            "native_search_frontier": bool(
                active.enable_chemenzy
                and active.enable_guided_chemenzy
                and guided_continuation_allowed
                and not guided_progress_pending
                and frontier_native.get("available") is True
            ),
            "native_search": bool(
                active.enable_chemenzy
                and (
                    target_native.get("available") is True
                    or frontier_native.get("available") is True
                )
            ),
        }

    def current_campaign_gates() -> dict[str, Any]:
        graph = service.graph_store.load()
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=resolved_acceptance,
            config=_portfolio_config(active, resolved_acceptance),
        )
        latest_campaign_portfolio.clear()
        latest_campaign_portfolio.update(portfolio)
        return compile_blind_acceptance_report(
            preflight=preflight,
            director_outcomes=outcomes,
            graph=graph,
            portfolio=portfolio,
        )

    def handle_materialize(action: CampaignAction) -> dict[str, Any]:
        return service.execute_frontier_materialization(
            idempotency_key=f"{action.idempotency_key}:handler",
            hypothesis_ids=action.subject_ids,
        )

    def handle_validation(action: CampaignAction) -> dict[str, Any]:
        force_revalidation = action.metadata.get("force_revalidation") is True
        return validate_materialized_edges(
            service,
            atom_mapper=atom_mapper,
            max_reactions=active.max_atom_mapping_reactions,
            edge_ids=action.subject_ids,
            revalidate_edge_ids=(action.subject_ids if force_revalidation else ()),
        )

    def prepare_validation(action: CampaignAction) -> dict[str, Any]:
        force_revalidation = action.metadata.get("force_revalidation") is True
        return prepare_materialized_edge_validation(
            service,
            atom_mapper=atom_mapper,
            max_reactions=active.max_atom_mapping_reactions,
            edge_ids=action.subject_ids,
            revalidate_edge_ids=(action.subject_ids if force_revalidation else ()),
        )

    def commit_validation(
        _action: CampaignAction,
        prepared: Mapping[str, Any],
    ) -> dict[str, Any]:
        return commit_materialized_edge_validation(service, prepared)

    deferred_validation_handler = CampaignActionDeferredHandler(
        prepare=prepare_validation,
        commit=commit_validation,
    )

    def handle_stock(_action: CampaignAction) -> dict[str, Any]:
        return _audit_stock_stage(
            service,
            acceptance=resolved_acceptance,
            config=active,
            catalog_builder=stock_catalog_builder,
            inventory_builder=inventory_snapshot_builder,
        )

    def handle_route_recompute(action: CampaignAction) -> dict[str, Any]:
        result = service.apply_batch(
            CanonicalIngestionBatch(recompute_derived=True),
            idempotency_key=f"{action.idempotency_key}:handler",
        )
        return {
            "status": "completed",
            "changed": result.get("changed") is True,
            "route_family_ids": sorted(action.route_family_ids),
            "dirty_entity_ids": list(result.get("dirty_entity_ids") or []),
            "material_events": (
                ["canonical_route_closure_recomputed"]
                if result.get("changed") is True
                else []
            ),
            "semantics": {
                "canonical_hypergraph_remains_route_closure_authority": True,
                "unchanged_recompute_is_a_bounded_no_gain_action": True,
            },
        }

    def handle_condition(action: CampaignAction) -> dict[str, Any]:
        return enrich_materialized_edge_conditions(
            service,
            predictor=resolved_condition_predictor,
            enabled=active.enable_condition_enrichment,
            max_reactions=active.max_condition_prediction_reactions,
            top_k=active.condition_prediction_topk,
            condition_model="rcr",
            edge_ids=action.subject_ids,
        )

    def run_initial_chemenzy_and_signal(
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            return operation()
        finally:
            initial_chemenzy_admission_complete.set()

    def handle_target_chemenzy(action: CampaignAction) -> dict[str, Any]:
        if action.metadata.get("target_level_native_search") is not True:
            return {
                "status": "failed",
                "reasons": ["chemenzy_target_action_scope_invalid"],
            }
        return run_initial_chemenzy_and_signal(
            lambda: run_chemenzy_proposal_stage(
                service,
                target_name=case.target_name,
                target_smiles=canonical,
                enabled=(active.enable_chemenzy and active.enable_target_chemenzy_baseline),
                provider=chemenzy_provider,
                env_prefix=active.chemenzy_env_prefix or None,
                vendor_root=_chemenzy_vendor_root(gateway.paths.vendor_root),
                max_routes=active.effective_provider_route_reserve,
                max_host_routes=active.effective_host_route_portfolio,
                max_steps=active.max_chemenzy_steps,
                max_iterations=active.max_chemenzy_iterations,
                expansion_topk=active.chemenzy_expansion_topk,
                timeout_s=active.chemenzy_timeout_s,
                search_preset=active.chemenzy_search_preset,
                random_seed=active.chemenzy_seed,
                stock_names=active.chemenzy_stock_names,
                stock_paths=dict(active.chemenzy_stock_paths),
                enable_condition_prediction=(
                    active.enable_chemenzy_condition_prediction
                    and active.action_scheduler_policy != "adaptive"
                ),
                enable_enzyme_assignment=(
                    active.enable_chemenzy_enzyme_assignment
                    and active.action_scheduler_policy != "adaptive"
                ),
                enable_enzyme_coverage_sidecar=(
                    active.enable_enzyme_coverage_sidecar
                    and active.action_scheduler_policy != "adaptive"
                ),
                pandarallel_workers=active.chemenzy_pandarallel_workers,
                stop_on_first_host_admitted_route=(
                    active.delivery_boundary == "stock_result"
                ),
            )
        )

    def handle_guided_chemenzy(action: CampaignAction) -> dict[str, Any]:
        frontier_smiles = str(action.metadata.get("frontier_smiles") or "")
        if action.metadata.get("target_level_native_search") is True or not frontier_smiles:
            return {
                "status": "failed",
                "reasons": ["chemenzy_frontier_action_scope_invalid"],
            }
        _frontier_id, frontier_key = molecule_identity(frontier_smiles)
        frontier_key = frontier_key or frontier_smiles
        if frontier_key in attempted_guided_frontier_smiles:
            return {
                "status": "skipped",
                "frontier_smiles": [frontier_key],
                "proposal_count": 0,
                "provider_invocation_count": 0,
                "reasons": ["guided_frontier_already_attempted"],
                "semantics": {
                    "duplicate_guided_provider_call_suppressed": True,
                    "deduplication_is_frontier_identity_based": True,
                },
            }
        current_graph = service.graph_store.load()
        if any(
            str(hypothesis.get("product_smiles") or "") == frontier_smiles
            and any(
                str(origin.get("origin_kind") or "") == "chemenzy"
                and str(origin.get("origin_ref") or "").startswith("chemenzy:guided-")
                for origin in hypothesis.get("origin_records") or []
                if isinstance(origin, Mapping)
            )
            for hypothesis in dict(current_graph.get("hypotheses") or {}).values()
            if isinstance(hypothesis, Mapping)
        ):
            return {
                "status": "skipped",
                "frontier_smiles": [frontier_smiles],
                "proposal_count": 0,
                "provider_invocation_count": 0,
                "reasons": ["guided_frontier_already_attempted"],
                "semantics": {
                    "duplicate_guided_provider_call_suppressed": True,
                },
            }
        before_progress = compile_parent_route_stock_progress(
            compile_proof_portfolio(
                current_graph,
                acceptance_spec=resolved_acceptance,
                config=_portfolio_config(active, resolved_acceptance),
            ),
            parent_route_family_ids=action.route_family_ids,
        )
        result = run_chemenzy_guided_frontier_stage(
            service,
            target_name=case.target_name,
            root_target_smiles=canonical,
            enabled=active.enable_chemenzy and active.enable_guided_chemenzy,
            provider=chemenzy_provider,
            env_prefix=active.chemenzy_env_prefix or None,
            vendor_root=_chemenzy_vendor_root(gateway.paths.vendor_root),
            max_frontiers=1,
            max_routes=1,
            max_steps=min(6, active.max_chemenzy_steps),
            max_iterations=active.max_guided_chemenzy_iterations,
            expansion_topk=min(80, active.chemenzy_expansion_topk),
            timeout_s=active.guided_chemenzy_timeout_s,
            include_frontier_smiles=(frontier_smiles,),
            search_preset="thorough",
            random_seed=active.chemenzy_seed,
            stock_names=active.chemenzy_stock_names,
            stock_paths=dict(active.chemenzy_stock_paths),
            enable_condition_prediction=(
                active.enable_chemenzy_condition_prediction
                and active.action_scheduler_policy != "adaptive"
            ),
            enable_enzyme_assignment=(
                active.enable_chemenzy_enzyme_assignment
                and active.action_scheduler_policy != "adaptive"
            ),
            enable_enzyme_coverage_sidecar=(
                active.enable_enzyme_coverage_sidecar
                and active.action_scheduler_policy != "adaptive"
            ),
            pandarallel_workers=active.chemenzy_pandarallel_workers,
            stop_on_first_host_admitted_route=(
                active.delivery_boundary == "stock_result"
            ),
        )
        attempted_guided_frontier_smiles.add(frontier_key)
        result = {
            **dict(result),
            "guided_progress_checkpoint": {
                "before": before_progress,
                "parent_route_family_ids": sorted(action.route_family_ids),
                "frontier_smiles": frontier_key,
                "provider_proposal_count": int(
                    result.get("proposal_count") or 0
                ),
            },
        }
        guided_progress_pending.clear()
        guided_progress_pending.update(
            dict(result["guided_progress_checkpoint"])
        )
        return result

    def handle_global_architecture(action: CampaignAction) -> dict[str, Any]:
        if action.metadata.get("global_architecture") is not True:
            return {
                "status": "failed",
                "reasons": ["codex_global_architecture_action_scope_invalid"],
            }
        return _run_director_safely(
            service,
            mode="initial_architecture",
            evidence_observations={
                **dict(initial_template_observation),
                "target_identity": target_identity,
                "chemenzy_provider_observation": chemenzy_observation,
            },
            context=initial_director_context,
            before_plan_admission=(
                initial_chemenzy_admission_complete.wait if ordered_initial_admission else None
            ),
            idempotency_key="solve-target:director:initial",
        )

    def handle_program_review(_action: CampaignAction) -> dict[str, Any]:
        projection = service.program_projection()
        return {
            "status": "completed",
            "projection": projection,
            "store": service.program_store(),
            "semantics": {
                "program_review_is_read_only": True,
                "program_projection_grants_no_canonical_authority": True,
            },
        }

    def resolve_program_route_binding(
        action: CampaignAction,
    ) -> dict[str, Any]:
        requested_route_id = str(action.metadata.get("route_id") or "")
        route_family_id = str(
            action.metadata.get("route_family_id") or next(iter(action.route_family_ids), "")
        )
        portfolio = compile_proof_portfolio(
            service.graph_store.load(),
            acceptance_spec=resolved_acceptance,
        )
        candidates = [
            dict(value)
            for value in portfolio.get("route_candidates") or []
            if isinstance(value, Mapping)
        ]
        exact = next(
            (
                value
                for value in candidates
                if str(value.get("route_id") or "") == requested_route_id
            ),
            None,
        )
        if exact is not None:
            return {
                "status": "exact",
                "requested_route_id": requested_route_id,
                "route_id": requested_route_id,
                "route_family_id": str(exact.get("route_family_id") or route_family_id),
            }
        family_candidates = sorted(
            (
                value
                for value in candidates
                if route_family_id and str(value.get("route_family_id") or "") == route_family_id
            ),
            key=lambda value: (
                value.get("selected") is not True,
                str(value.get("route_id") or ""),
            ),
        )
        if family_candidates:
            rebound = family_candidates[0]
            return {
                "status": "route_family_rebound",
                "requested_route_id": requested_route_id,
                "route_id": str(rebound.get("route_id") or ""),
                "route_family_id": route_family_id,
                "semantics": {
                    "signal_remains_bound_to_same_route_family": True,
                    "latest_canonical_route_variant_is_used": True,
                    "no_cross_family_retargeting": True,
                },
            }
        return {
            "status": "missing",
            "requested_route_id": requested_route_id,
            "route_id": "",
            "route_family_id": route_family_id,
        }

    def handle_program_discovery(action: CampaignAction) -> dict[str, Any]:
        route_binding = resolve_program_route_binding(action)
        route_id = str(route_binding.get("route_id") or "")
        if not route_id:
            return {
                "status": "failed",
                "reasons": ["program_discovery_route_binding_missing"],
                "route_binding": route_binding,
            }
        # Program review is defined over a canonical source route, not a
        # hypothesis-only family.  Resolve the bound row again at the handler
        # boundary so stale/empty signals cannot reach the strict innovation
        # contract and surface as an opaque ``source_route_invalid`` error.
        bound_portfolio = compile_proof_portfolio(
            service.graph_store.load(),
            acceptance_spec=resolved_acceptance,
            config=_portfolio_config(active, resolved_acceptance),
        )
        bound_route = next(
            (
                dict(value)
                for value in bound_portfolio.get("route_candidates") or []
                if isinstance(value, Mapping)
                and str(value.get("route_id") or "") == route_id
            ),
            None,
        )
        if not bound_route or not _route_has_canonical_edges(bound_route):
            return {
                "status": "blocked",
                "reasons": ["program_discovery_source_route_empty"],
                "route_id": route_id,
                "route_binding": route_binding,
                "semantics": {
                    "empty_or_hypothesis_only_routes_are_not_program_sources": True,
                    "no_source_route_invalid_exception_is_emitted": True,
                },
            }
        program_review = service.review_route_program_innovations(
            route_id,
            capabilities=resolved_program_capabilities,
            mechanism_proposals=resolved_mechanism_proposals,
        )
        discovery = dict(program_review.get("discovery") or {})
        innovation_batch = canonical_innovation_batch(discovery)
        hypotheses = tuple(
            dict(value)
            for value in innovation_batch.get("hypotheses") or []
            if isinstance(value, Mapping)
        )
        ingestion = (
            service.apply_batch(
                CanonicalIngestionBatch(hypotheses=hypotheses),
                idempotency_key=f"program-discovery:{action.execution_id}",
            )
            if hypotheses
            else {"changed": False}
        )
        return {
            "status": (
                "completed" if int(discovery.get("candidate_count") or 0) else "reused_or_empty"
            ),
            "route_id": route_id,
            "route_binding": route_binding,
            "candidate_count": int(discovery.get("candidate_count") or 0),
            "program_draft_candidate_ids": list(discovery.get("program_draft_candidate_ids") or []),
            "execution_program_draft_candidate_ids": list(
                discovery.get("execution_program_draft_candidate_ids") or []
            ),
            "mechanism_hypothesis_count": len(hypotheses),
            "discovery": discovery,
            "program_review": program_review,
            "experimental_work_frontier": dict(
                program_review.get("experimental_work_frontier") or {}
            ),
            "canonical_ingestion": ingestion,
            "semantics": {
                "program_candidates_are_proposal_only": True,
                "mechanism_hypotheses_use_canonical_ingestion": True,
                "enzyme_windows_do_not_masquerade_as_reaction_edges": True,
            },
        }

    def handle_program_validation(action: CampaignAction) -> dict[str, Any]:
        work_item = dict(action.metadata.get("work_item") or {})
        execution_request = dict(work_item.get("execution_request") or {})
        if (
            work_item.get("schema_version") != "experimental_work_item.v1"
            or not str(work_item.get("content_sha256") or "")
            or not str(execution_request.get("request_id") or "")
        ):
            return {
                "status": "failed",
                "reasons": ["program_validation_work_item_binding_invalid"],
            }
        return {
            "status": "awaiting_external_result",
            "changed": False,
            "route_id": str(action.metadata.get("route_id") or ""),
            "program_id": str(work_item.get("program_id") or ""),
            "work_item_id": str(work_item.get("work_item_id") or ""),
            "execution_request": execution_request,
            "semantics": {
                "request_is_dispatch_candidate_only": True,
                "no_validation_claim_has_been_granted": True,
                "canonical_graph_is_not_mutated": True,
                "external_result_requires_feedback_action": True,
            },
        }

    def handle_experiment_feedback(action: CampaignAction) -> dict[str, Any]:
        route_id = str(action.metadata.get("route_id") or "")
        validation = dict(action.metadata.get("validation") or {})
        if not route_id or not validation:
            return {
                "status": "failed",
                "reasons": ["experiment_feedback_binding_missing"],
            }
        program_review = service.review_route_program_innovations(
            route_id,
            capabilities=resolved_program_capabilities,
            mechanism_proposals=resolved_mechanism_proposals,
            validations=(validation,),
        )
        claim_oracle = dict(program_review.get("experimental_claims_oracle") or {})
        if claim_oracle.get("accepted") is not True:
            return {
                "status": "failed",
                "reasons": ["experiment_feedback_claim_projection_rejected"],
                "claim_oracle": claim_oracle,
            }
        experimental_claims = dict(program_review.get("experimental_claims") or {})
        validation_id = str(validation.get("validation_id") or "")
        matching_claims = [
            dict(value)
            for value in dict(experimental_claims.get("claims") or {}).values()
            if isinstance(value, Mapping)
            and str(dict(value.get("source_validation") or {}).get("validation_id") or "")
            == validation_id
        ]
        if not matching_claims:
            return {
                "status": "failed",
                "reasons": ["experiment_feedback_domain_gate_rejected"],
                "route_id": route_id,
                "program_id": str(validation.get("program_id") or ""),
                "validation_id": validation_id,
                "experimental_claims": experimental_claims,
                "experimental_claims_oracle": claim_oracle,
                "semantics": {
                    "invalid_feedback_is_fail_closed": True,
                    "canonical_reaction_edges_are_not_created": True,
                    "claim_store_is_not_written": True,
                },
            }
        shadow_admission = (
            service.admit_route_experimental_claims(
                route_id,
                capabilities=resolved_program_capabilities,
                mechanism_proposals=resolved_mechanism_proposals,
                validations=(validation,),
                enable_experimental_claim_admission=True,
            )
            if active.enable_experimental_claim_admission
            else {
                "status": "skipped",
                "reason": "experimental_claim_admission_not_explicitly_enabled",
            }
        )
        program_id = str(validation.get("program_id") or "")
        pending_signal_ids = sorted(
            str(signal_id)
            for signal_id, raw_signal in dict(
                service.graph_store.load().get("action_signals") or {}
            ).items()
            if isinstance(raw_signal, Mapping)
            and str(raw_signal.get("kind") or "") == "program_validation"
            and str(
                dict(dict(raw_signal.get("metadata") or {}).get("work_item") or {}).get(
                    "program_id"
                )
                or ""
            )
            == program_id
        )
        return {
            "status": "completed",
            "changed": False,
            "route_id": route_id,
            "program_id": program_id,
            "validation_id": validation_id,
            "experimental_claims": experimental_claims,
            "experimental_claims_oracle": claim_oracle,
            "shadow_admission": shadow_admission,
            "resolved_program_validation_signal_ids": pending_signal_ids,
            "semantics": {
                "feedback_is_host_gated": True,
                "claims_are_exact_boundary_observations": True,
                "shadow_admission_requires_explicit_enable": True,
                "canonical_reaction_edges_are_not_created": True,
            },
        }

    def handle_program_admission(_action: CampaignAction) -> dict[str, Any]:
        if not active.enable_program_admission:
            return {
                "status": "skipped",
                "reason": "program_admission_not_explicitly_enabled",
            }
        return {
            "status": "completed",
            "admission": service.admit_programs(enable_program_admission=True),
            "semantics": {
                "shadow_program_store_only": True,
                "canonical_graph_remains_authoritative": True,
            },
        }

    preexecuted_action_backlog: list[dict[str, Any]] = []

    def project_action_results(
        _phase: str,
        action_kinds: Iterable[CampaignActionKind | str],
        *,
        max_actions: int,
    ) -> list[dict[str, Any]]:
        """Project executions already produced by the one anytime loop."""

        del _phase
        projected_kinds = {
            kind.value if isinstance(kind, CampaignActionKind) else str(kind)
            for kind in action_kinds
        }
        preexecuted = [
            execution
            for execution in preexecuted_action_backlog
            if str(dict(execution.get("action") or {}).get("kind") or "") in projected_kinds
        ][: max(1, int(max_actions))]
        if preexecuted:
            consumed_ids = {id(execution) for execution in preexecuted}
            preexecuted_action_backlog[:] = [
                execution
                for execution in preexecuted_action_backlog
                if id(execution) not in consumed_ids
            ]
            return [dict(execution) for execution in preexecuted]
        return []

    program_action_handlers = {
        **(
            {CampaignActionKind.PROGRAM_DISCOVER: handle_program_discovery}
            if active.enable_program_discovery
            and (bool(resolved_program_capabilities) or bool(resolved_mechanism_proposals))
            else {}
        ),
        **(
            {CampaignActionKind.PROGRAM_REVIEW: handle_program_review}
            if active.enable_program_review
            else {}
        ),
        **(
            {CampaignActionKind.PROGRAM_ADMIT: handle_program_admission}
            if active.enable_program_admission
            else {}
        ),
        **(
            {CampaignActionKind.PROGRAM_VALIDATE: handle_program_validation}
            if active.enable_program_validation
            else {}
        ),
        **(
            {CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST: (handle_experiment_feedback)}
            if active.enable_program_validation
            else {}
        ),
    }

    resume_signal_kinds = {
        **({"replan": True} if active.enable_replan else {}),
        **(
            {"program_discovery": True}
            if active.enable_program_discovery
            and (bool(resolved_program_capabilities) or bool(resolved_mechanism_proposals))
            else {}
        ),
        **({"program_review": True} if active.enable_program_review else {}),
        **({"program_admission": True} if active.enable_program_admission else {}),
        **(
            {"program_validation": True, "experiment_feedback": True}
            if active.enable_program_validation
            else {}
        ),
    }
    resume_work = classify_target_resume_work(
        checkpoint,
        service.graph_store.load(),
        feedback_signals=resolved_feedback_signals,
        available_signal_kinds=resume_signal_kinds,
    )

    if budget_extension_event is not None:
        stages.append(
            _stage(
                "model_budget_extension",
                "accepted",
                {
                    "event": budget_extension_event.to_dict(),
                    "effective_budget": resolved_budget.to_dict(),
                    "semantics": {
                        "explicit_resume_policy": True,
                        "observed_usage_is_preserved": True,
                    },
                },
            )
        )
    if checkpoint.get("complete") is True and service.kernel.decide_stop().terminal:
        if resume_work.get("has_new_work") is not True:
            return _refresh_terminal_report(
                service,
                identity=identity,
                directory=directory,
                canonical=canonical,
                case=case,
                preflight=preflight,
                config=active,
                acceptance=resolved_acceptance,
                budget=resolved_budget,
                stages=stages,
                outcomes=outcomes,
            )
        prior_status = service.kernel.state.status
        reopen_event = service.kernel.reopen_for_new_work(
            work_fingerprint=str(resume_work["work_fingerprint"]),
            reasons=tuple(resume_work.get("reasons") or ()),
            idempotency_key=(
                "solve-target:terminal-reopen:" + str(resume_work["work_fingerprint"])[:24]
            ),
        )
        stages.append(
            _stage(
                "terminal_checkpoint_reopened",
                "accepted",
                {
                    "prior_status": prior_status,
                    "event": reopen_event.to_dict(),
                    "resume_work": resume_work,
                    "semantics": {
                        "same_run_kernel_and_trajectory_continue": True,
                        "new_work_reenters_single_anytime_loop": True,
                        "terminal_report_is_not_refreshed_over_new_input": True,
                    },
                },
            )
        )
    target_identity = _target_identity_stage(
        service,
        stages=stages,
        target_name=case.target_name,
        target_smiles=canonical,
        enabled=active.enable_target_identity,
        resolve_named=active.resolve_named_target_identity,
        lookup_now=False,
    )
    identity_stage = _stage("target_identity", target_identity["status"], target_identity)
    identity_indices = [
        index for index, row in enumerate(stages) if row.get("stage") == "target_identity"
    ]
    if identity_indices:
        stages[identity_indices[-1]] = identity_stage
    else:
        stages.append(identity_stage)
    resolved_target_name = str(
        dict(target_identity.get("identity") or {}).get("preferred_name") or case.target_name
    )
    evidence_prefetch_result: dict[str, Any] = {
        "status": "not_started",
        "elapsed_s": 0.0,
        "discovery": {},
    }
    evidence_prefetch_request: dict[str, Any] = {}
    evidence_prefetch_signal_id = ""
    if (
        not outcomes
        and active.action_scheduler_policy != "adaptive"
        and resolved_evidence_connector is not None
        and getattr(
            resolved_evidence_connector,
            "autoplanner_prefetch_safe",
            False,
        )
        is True
    ):
        evidence_prefetch_request = compile_evidence_acquisition_request(
            run_id=service.kernel.spec.run_id,
            target_name=resolved_target_name,
            target_smiles=service.kernel.spec.target_smiles,
            graph=service.graph_store.load(),
            source_frontier={},
            target_identity=dict(target_identity.get("identity") or {}),
            prefetch_mode=True,
        )
        prefetch_sha256 = str(evidence_prefetch_request.get("content_sha256") or "")
        evidence_prefetch_signal_id = f"event-deficit:evidence-prefetch:{prefetch_sha256}"
        prefetch_graph = service.graph_store.load()
        existing_prefetch_signal = dict(
            dict(prefetch_graph.get("action_signals") or {}).get(evidence_prefetch_signal_id) or {}
        )
        if str(existing_prefetch_signal.get("status") or "open") != "resolved":
            target_object = str(
                prefetch_graph.get("target_molecule_id") or service.kernel.spec.run_id
            )
            service.publish_action_signals(
                (
                    {
                        "signal_id": evidence_prefetch_signal_id,
                        "kind": "evidence",
                        "object_id": target_object,
                        "entity_ids": [target_object],
                        "route_family_ids": [],
                        "dependency_ids": [],
                        "deterministic": False,
                        "model_allowed": False,
                        "reason": "target_source_prefetch_requires_evidence_acquisition",
                        "score": {
                            "expected_portfolio_gain": 0.2,
                            "distance_to_closure": 0.1,
                            "evidence_gain": 0.4,
                            "route_diversity_gain": 0.0,
                            "cost_penalty": 0.15,
                            "failure_risk_penalty": 0.05,
                        },
                        "metadata": {
                            "target_level_evidence_prefetch": True,
                            "evidence_prefetch_request_sha256": prefetch_sha256,
                        },
                    },
                ),
                idempotency_key=(f"unified-evidence-prefetch-signal:{prefetch_sha256[:24]}"),
            )
    initial_template_retrieval = self_evo.start(service.graph_store.load())
    stages.append(
        _stage(
            "patent_template_retrieval",
            initial_template_retrieval["status"],
            initial_template_retrieval,
        )
    )
    initial_template_observation = self_evo.observation()
    chemenzy_observation: dict[str, Any] = {}
    trajectory_code_binding = _target_control_plane_code_binding(gateway.paths.repository_root)

    def current_trajectory_bindings() -> dict[str, Any]:
        return _compile_target_trajectory_bindings(
            code_binding=trajectory_code_binding,
            campaign_spec=service.kernel.spec.campaign_spec.to_dict(),
            config=active,
            director_config=director_config,
            chemenzy_provider=chemenzy_provider,
            evidence_connector=resolved_evidence_connector,
            condition_predictor=resolved_condition_predictor,
            program_capabilities=resolved_program_capabilities,
            chemenzy_observation=chemenzy_observation,
            chemenzy_runtime_binding=_latest_chemenzy_runtime_binding(stages),
        )

    def current_campaign_snapshot(
        *,
        phase: str,
        gates: Mapping[str, Any],
        portfolio: Mapping[str, Any] | None = None,
        action_decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        graph = service.graph_store.load()
        resolved_portfolio = dict(portfolio or latest_campaign_portfolio)
        if int(resolved_portfolio.get("graph_revision") or -1) != int(graph.get("revision") or 0):
            resolved_portfolio = compile_proof_portfolio(
                graph,
                acceptance_spec=resolved_acceptance,
                config=_portfolio_config(active, resolved_acceptance),
            )
        route_state = compile_route_snapshot(
            graph=graph,
            portfolio=resolved_portfolio,
            gates=gates,
        )
        return compile_campaign_snapshot(
            phase=phase,
            observed_at=_utc_now(),
            event_sequence=service.kernel.state.event_count,
            graph_revision=service.kernel.state.graph_revision,
            wall_time_s=service.kernel.state.task_wall_time_s,
            gates=gates,
            resource_usage={
                "model": dict(service.kernel.state.model_totals),
                "native_search": service.kernel.native_search_budget(),
                "tasks": service.kernel.task_budget(),
                "attempt_count": service.kernel.state.attempt_count,
                "accepted_expansion_count": (service.kernel.state.accepted_expansion_count),
                "settled_task_count": service.kernel.state.settled_task_count,
                "in_flight_task_count": len(service.kernel.state.in_flight_tasks),
                "cumulative_task_wall_time_s": (service.kernel.state.task_wall_time_s),
                "cumulative_task_compute_time_s": (
                    service.kernel.state.task_compute_time_s
                ),
            },
            action_counts=compile_action_counts(_campaign_action_executions_from_stages(stages)),
            route_counts=dict(route_state["counts"]),
            pareto_archive=route_state["pareto_archive"],
            bindings=current_trajectory_bindings(),
            program_milestones=_program_milestones_from_stages(stages),
            action_decision=action_decision,
        )

    initial_director_context = service.compile_global_context(
        evidence_observations={
            **dict(initial_template_observation),
            "target_identity": target_identity,
            "chemenzy_provider_observation": {},
        }
    )
    latest_evidence_action_result: dict[str, Any] = {}
    evidence_action_completed = False
    unified_material_events: set[str] = set()
    provider_search_failures: list[dict[str, Any]] = []
    unified_replan_audit_contexts: dict[str, dict[str, Any]] = {}
    program_pressure_cache: dict[str, dict[str, Any]] = {}

    def current_program_opportunity_pressure(
        graph: Mapping[str, Any],
        route: Mapping[str, Any],
    ) -> dict[str, Any]:
        edge_ids = {str(value) for value in route.get("edge_ids") or [] if str(value)}
        edges = {
            edge_id: dict(dict(graph.get("edges") or {}).get(edge_id) or {})
            for edge_id in sorted(edge_ids)
        }
        molecule_ids = {
            str(value)
            for edge in edges.values()
            for value in (
                edge.get("product_molecule_id"),
                *(edge.get("precursor_molecule_ids") or []),
            )
            if str(value)
        }
        procedure_ids = {
            str(value)
            for edge in edges.values()
            for value in edge.get("procedure_record_ids") or []
            if str(value)
        }
        input_sha256 = _digest(
            {
                "route": dict(route),
                "edges": edges,
                "molecules": {
                    molecule_id: dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
                    for molecule_id in sorted(molecule_ids)
                },
                "procedure_records": {
                    record_id: dict(dict(graph.get("procedure_records") or {}).get(record_id) or {})
                    for record_id in sorted(procedure_ids)
                },
                "fact_lifecycle_events": dict(graph.get("fact_lifecycle_events") or {}),
                "capabilities": resolved_program_capabilities,
                "mechanism_proposals": resolved_mechanism_proposals,
            }
        )
        cached = program_pressure_cache.get(input_sha256)
        if cached is None:
            cached = compile_program_opportunity_pressure(
                graph,
                route,
                capabilities=resolved_program_capabilities,
                mechanism_proposals=resolved_mechanism_proposals,
            )
            program_pressure_cache[input_sha256] = cached
        return cached

    initial_chemenzy_admission_complete = Event()
    initial_resources = scheduler_resources()
    initial_action_kinds = {
        str(value.get("kind") or "")
        for value in compile_action_opportunities(
            dict(service.graph_store.load().get("deficit_frontier") or {})
        ).get("actions")
        or []
        if isinstance(value, Mapping)
    }
    ordered_initial_admission = bool(
        active.action_scheduler_policy == "adaptive"
        and active.enable_chemenzy
        and active.enable_target_chemenzy_baseline
        and active.enable_codex
        and initial_resources.get("native_search_target") is True
        and initial_resources.get("model") is True
        and CampaignActionKind.CHEMENZY_TARGET_EXPAND.value in initial_action_kinds
        and CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE.value in initial_action_kinds
    )
    if not ordered_initial_admission:
        initial_chemenzy_admission_complete.set()

    def prepare_unified_evidence(_action: CampaignAction) -> dict[str, Any]:
        if _action.metadata.get("target_level_evidence_prefetch") is True:
            expected_sha256 = str(_action.metadata.get("evidence_prefetch_request_sha256") or "")
            if (
                not evidence_prefetch_request
                or expected_sha256 != str(evidence_prefetch_request.get("content_sha256") or "")
                or resolved_evidence_connector is None
            ):
                return {
                    "mode": "prefetch",
                    "result": {
                        "status": "failed",
                        "changed": False,
                        "reasons": ["evidence_prefetch_action_binding_invalid"],
                    },
                }
            return {
                "mode": "prefetch",
                "result": _run_evidence_prefetch(
                    evidence_prefetch_request,
                    resolved_evidence_connector,
                ),
            }
        if evidence_action_completed:
            reused = dict(latest_evidence_action_result)
            return {
                "mode": "reused",
                "result": {
                    **reused,
                    "status": "reused",
                    "reused_evidence_status": str(reused.get("status") or "completed"),
                    "changed": False,
                    "reason": "unified_evidence_action_already_executed",
                },
            }
        prepared_target_identity = _target_identity_stage(
            service,
            stages=stages,
            target_name=case.target_name,
            target_smiles=canonical,
            enabled=active.enable_target_identity,
            resolve_named=active.resolve_named_target_identity,
            lookup_now=True,
        )
        prepared_target_name = str(
            dict(prepared_target_identity.get("identity") or {}).get("preferred_name")
            or opaque_name
        )
        source_frontier = discover_director_source_hints(service, outcomes)
        return {
            "mode": "full",
            "target_identity_stage": prepared_target_identity,
            "resolved_target_name": prepared_target_name,
            "prepared_acquisition": _prepare_evidence_acquisition(
                service,
                source_stage=source_frontier,
                connector=resolved_evidence_connector,
                target_name=prepared_target_name,
                target_identity=dict(prepared_target_identity.get("identity") or {}),
                allow_target_identity_lookup=active.enable_target_identity,
            ),
        }

    def commit_unified_evidence(
        _action: CampaignAction,
        prepared: Mapping[str, Any],
    ) -> dict[str, Any]:
        nonlocal evidence_action_completed, latest_evidence_action_result
        nonlocal resolved_target_name, target_identity
        prepared_row = dict(prepared)
        mode = str(prepared_row.get("mode") or "")
        if mode in {"prefetch", "reused"}:
            return dict(prepared_row.get("result") or {})
        if mode != "full":
            return {
                "status": "failed",
                "changed": False,
                "reasons": ["prepared_evidence_action_mode_invalid"],
            }
        target_identity = dict(prepared_row.get("target_identity_stage") or {})
        resolved_target_name = str(
            prepared_row.get("resolved_target_name")
            or dict(target_identity.get("identity") or {}).get("preferred_name")
            or opaque_name
        )
        refreshed_identity_stage = _stage(
            "target_identity",
            str(target_identity.get("status") or "unresolved"),
            target_identity,
        )
        identity_rows = [
            index for index, row in enumerate(stages) if row.get("stage") == "target_identity"
        ]
        if identity_rows:
            stages[identity_rows[-1]] = refreshed_identity_stage
        else:
            stages.append(refreshed_identity_stage)
        result = _acquire_evidence_stage(
            service,
            source_stage={},
            connector=resolved_evidence_connector,
            atom_mapper=atom_mapper,
            visual_provider=visual_evidence_provider,
            max_visual_pages=active.max_visual_evidence_pages,
            target_name=resolved_target_name,
            target_identity=dict(target_identity.get("identity") or {}),
            allow_target_identity_lookup=active.enable_target_identity,
            prior_visual_observation=_latest_visual_observation(stages),
            defer_validation=True,
            prepared_acquisition=dict(prepared_row.get("prepared_acquisition") or {}),
        )
        evidence_action_completed = True
        latest_evidence_action_result = dict(result)
        return result

    deferred_unified_evidence_handler = CampaignActionDeferredHandler(
        prepare=prepare_unified_evidence,
        commit=commit_unified_evidence,
    )

    def handle_unified_replan(action: CampaignAction) -> dict[str, Any]:
        if action.metadata.get("global_replan") is not True:
            return {
                "status": "failed",
                "reasons": ["codex_global_replan_action_scope_invalid"],
            }
        unified_replan_audit_contexts[action.execution_id] = {
            "graph_before": service.graph_store.load(),
            "gates_before": current_campaign_gates(),
            "model_cost_before": dict(service.kernel.state.model_totals),
        }
        evidence_observations = {
            **_evidence_observations(latest_evidence_action_result),
            **self_evo.observation(),
            **(
                {"provider_search_failures": list(provider_search_failures)}
                if provider_search_failures
                else {}
            ),
        }
        result = _run_director_safely(
            service,
            mode="event_replan",
            material_events=tuple(
                str(value) for value in action.metadata.get("material_events") or [] if str(value)
            )
            or ("stock_boundary_changed",),
            evidence_observations=evidence_observations,
            idempotency_key=f"solve-target:director:{action.execution_id}",
        )
        if sequential_strategy:
            result = {
                **result,
                "operation_kind": "route_local_repair",
                "repair_scope": "one_failed_reaction_neighborhood",
                "maximum_repair_rounds": active.max_route_local_repair_rounds,
                "preserves_target_rooted_prefix": True,
                "legacy_scheduler_action_kind": "codex_global_replan",
                "semantics": {
                    **dict(result.get("semantics") or {}),
                    "matched_profile_does_not_run_a_second_global_plan": True,
                    "legacy_action_name_is_scheduler_compatibility_only": True,
                },
            }
        return result

    def publish_unified_replan_signal() -> None:
        event_replan_count = sum(
            str(outcome.get("mode") or "") == "event_replan"
            for outcome in outcomes
        )
        event_replan_limit = (
            active.max_route_local_repair_rounds if sequential_strategy else 1
        )
        if (
            not active.enable_replan
            or not outcomes
            or event_replan_count >= event_replan_limit
            or len(outcomes) >= _MAX_DIRECTOR_OUTCOMES
        ):
            return
        graph = service.graph_store.load()
        if any(
            str(signal.get("kind") or "") == "replan"
            for signal in dict(graph.get("action_signals") or {}).values()
            if isinstance(signal, Mapping)
        ):
            return
        gates = current_campaign_gates()
        observed_material_events = tuple(sorted(unified_material_events))
        convergence_ledger = unified_core_runtime.action_convergence_ledger(
            current_graph_revision=service.kernel.state.graph_revision
        )
        replan_pressure = compile_replan_pressure(
            gates,
            material_events=observed_material_events,
            convergence_ledger=convergence_ledger,
        )
        effective_material_events = tuple(
            sorted(
                {
                    *observed_material_events,
                    *replan_pressure["derived_material_events"],
                }
            )
        )
        reasons = _replan_reasons(
            gates,
            material_events=observed_material_events,
            convergence_ledger=convergence_ledger,
        )
        if not reasons or not _director_outcome_allows_replan(outcomes):
            return
        signal_gate = _replan_signal_gate(
            gates,
            material_events=observed_material_events,
            trigger_reasons=reasons,
            convergence_ledger=convergence_ledger,
        )
        if signal_gate.get("accepted") is not True:
            return
        stages.append(_stage("global_replan_signal_gate", "accepted", signal_gate))
        prompt_context_bytes = 0
        context_error = ""
        try:
            replan_context = service.compile_global_context(
                material_events=effective_material_events,
                evidence_observations={
                    **_evidence_observations(latest_evidence_action_result),
                    **self_evo.observation(),
                    **(
                        {
                            "provider_search_failures": list(
                                provider_search_failures
                            )
                        }
                        if provider_search_failures
                        else {}
                    ),
                },
            )
            prompt_context_bytes = len(
                director_prompt(
                    replan_context,
                    mode="event_replan",
                    config=director_config,
                ).encode("utf-8")
            )
        except CampaignContextTooLargeError as exc:
            context_error = str(exc)[:2_000]
        budget_guard = _replan_budget_guard(
            model_cost=service.kernel.state.model_totals,
            budget=resolved_budget,
            config=active,
            prompt_context_bytes=prompt_context_bytes,
        )
        if context_error:
            budget_guard = {
                **budget_guard,
                "accepted": False,
                "reasons": ["campaign_context_exceeds_bounded_replan_budget"],
                "context_error": context_error,
            }
        budget_guard = {
            **budget_guard,
            "trigger_reasons": list(reasons),
        }
        stages.append(
            _stage(
                "global_replan_budget_gate",
                "accepted" if budget_guard["accepted"] else "skipped",
                budget_guard,
            )
        )
        if budget_guard.get("accepted") is not True:
            return
        # A signal mutates the canonical graph and therefore rebuilds the
        # frontier.  Mark the target entity dirty as well as the signal itself
        # so the freshly published replan becomes visible immediately.
        target_object_id = str(
            graph.get("target_molecule_id") or service.kernel.spec.run_id
        )
        signal_payload = {
            "graph_revision": service.kernel.state.graph_revision,
            "material_events": list(effective_material_events),
            "observed_material_events": list(observed_material_events),
            "trigger_reasons": list(reasons),
            "prompt_context_bytes": prompt_context_bytes,
            "replan_pressure_sha256": replan_pressure["content_sha256"],
        }
        signal_sha256 = _digest(signal_payload)
        service.publish_action_signals(
            (
                {
                    "signal_id": f"event-deficit:replan:{signal_sha256}",
                    "kind": "replan",
                    "object_id": target_object_id,
                    "entity_ids": [
                        target_object_id,
                        f"event-deficit:replan:{signal_sha256}",
                    ],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": False,
                    "model_allowed": True,
                    "reason": "material_state_requires_global_replan",
                    "score": dict(replan_pressure["score"]),
                    "metadata": {
                        **signal_payload,
                        "global_replan": True,
                        "replan_pressure": replan_pressure,
                    },
                },
            ),
            idempotency_key=f"unified-replan-signal:{signal_sha256[:24]}",
        )

    def execute_pending_unified_replan() -> list[dict[str, Any]]:
        """Run the single replan that may be published after the core loop."""

        graph_before = service.graph_store.load()
        if any(
            str(signal.get("kind") or "") == "replan"
            for signal in dict(graph_before.get("action_signals") or {}).values()
            if isinstance(signal, Mapping)
        ):
            pass
        else:
            gate_values = current_campaign_gates()
            material_events = tuple(sorted(unified_material_events))
            replan_reasons = _replan_reasons(
                gate_values,
                material_events=material_events,
                convergence_ledger=unified_core_runtime.action_convergence_ledger(
                    current_graph_revision=service.kernel.state.graph_revision
                ),
            )
            if not replan_reasons or not _director_outcome_allows_replan(outcomes):
                return []
            signal_payload = {
                "graph_revision": service.kernel.state.graph_revision,
                "material_events": list(material_events),
                "trigger_reasons": list(replan_reasons),
                "global_replan": True,
            }
            signal_sha256 = _digest(signal_payload)
            target_object_id = str(
                graph_before.get("target_molecule_id")
                or service.kernel.spec.run_id
            )
            service.publish_action_signals(
                (
                    {
                        "signal_id": f"event-deficit:replan:{signal_sha256}",
                        "kind": "replan",
                        "object_id": target_object_id,
                        "entity_ids": [target_object_id],
                        "route_family_ids": [],
                        "dependency_ids": [],
                        "deterministic": False,
                        "model_allowed": True,
                        "reason": "material_state_requires_global_replan",
                        "metadata": signal_payload,
                    },
                ),
                idempotency_key=(
                    f"result-first-replan-signal:{signal_sha256[:24]}"
                ),
            )
        opportunities = compile_action_opportunities(
            dict(service.graph_store.load().get("deficit_frontier") or {})
        )
        execution_row = unified_core_runtime.schedule_and_execute(
            opportunities,
            milestones=_campaign_milestones(current_campaign_gates()),
            resource_availability=scheduler_resources(),
        )
        if (
            str(dict(execution_row.get("action") or {}).get("kind") or "")
            != CampaignActionKind.CODEX_REPLAN.value
        ):
            return []
        observe_unified_core_execution(
            len(unified_core_runtime.action_execution_history()),
            execution_row,
        )
        return [execution_row]

    def publish_unified_program_signals() -> None:
        if not (
            active.enable_program_discovery
            or active.enable_program_review
            or active.enable_program_admission
            or active.enable_program_validation
        ):
            return
        graph = service.graph_store.load()
        before_stock_boundary = bool(
            active.action_scheduler_policy == "adaptive"
            and _campaign_milestones(current_campaign_gates()).get(
                "B4_stock_boundary"
            )
            is not True
        )
        existing_signals = [
            dict(value)
            for value in dict(graph.get("action_signals") or {}).values()
            if isinstance(value, Mapping)
        ]
        discovered_pressure_bindings = {
            (
                str(
                    dict(signal.get("metadata") or {}).get("route_family_id")
                    or dict(signal.get("metadata") or {}).get("route_id")
                    or ""
                ),
                str(
                    dict(signal.get("metadata") or {}).get("program_pressure_sha256")
                    or dict(
                        dict(signal.get("metadata") or {}).get("program_opportunity_pressure") or {}
                    ).get("content_sha256")
                    or ""
                ),
            )
            for signal in existing_signals
            if str(signal.get("kind") or "") == "program_discovery"
            and str(
                dict(signal.get("metadata") or {}).get("program_pressure_sha256")
                or dict(
                    dict(signal.get("metadata") or {}).get("program_opportunity_pressure") or {}
                ).get("content_sha256")
                or ""
            )
        }
        review_pressure_bindings = {
            str(
                dict(signal.get("metadata") or {}).get("program_review_pressure_sha256")
                or dict(
                    dict(signal.get("metadata") or {}).get("program_review_pressure") or {}
                ).get("content_sha256")
                or ""
            )
            for signal in existing_signals
            if str(signal.get("kind") or "") == "program_review"
            and str(
                dict(signal.get("metadata") or {}).get("program_review_pressure_sha256")
                or dict(
                    dict(signal.get("metadata") or {}).get("program_review_pressure") or {}
                ).get("content_sha256")
                or ""
            )
        }
        existing_kinds = {str(signal.get("kind") or "") for signal in existing_signals}
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=resolved_acceptance,
            config=_portfolio_config(active, resolved_acceptance),
        )
        selected_route_rows = [
            dict(route)
            for route in portfolio.get("selected_routes") or []
            if isinstance(route, Mapping) and str(route.get("route_id") or "")
            and _route_has_canonical_edges(route)
        ]
        strategy_native_rows = [
            dict(route)
            for route in portfolio.get("route_candidates") or []
            if isinstance(route, Mapping)
            and str(route.get("route_id") or "")
            and _route_has_canonical_edges(route)
            and str(route.get("execution_domain") or "chemical")
            in {"enzymatic", "whole_cell", "hybrid", "mechanistic"}
        ]
        route_rows = list(
            {
                str(route.get("route_id")): route
                for route in [*strategy_native_rows, *selected_route_rows]
            }.values()
        )
        if before_stock_boundary:
            route_rows = [
                route
                for route in route_rows
                if str(route.get("execution_domain") or "chemical")
                in {"enzymatic", "whole_cell", "hybrid", "mechanistic"}
            ]
        route_rows = route_rows[: active.max_program_routes]
        if before_stock_boundary and not route_rows:
            return
        signals: list[dict[str, Any]] = []
        discovery_enabled = bool(
            active.enable_program_discovery
            and (resolved_program_capabilities or resolved_mechanism_proposals)
        )
        route_pressures = {
            str(route.get("route_id") or ""): current_program_opportunity_pressure(graph, route)
            for route in route_rows
            if discovery_enabled or active.enable_program_review
        }
        review_pressure = compile_program_review_pressure(route_pressures.values())
        review_needed = bool(
            route_rows
            and active.enable_program_review
            and not before_stock_boundary
            and review_pressure["content_sha256"] not in review_pressure_bindings
        )
        if discovery_enabled:
            for route in route_rows:
                route_id = str(route.get("route_id") or "")
                route_family_id = str(route.get("route_family_id") or route_id)
                program_pressure = route_pressures[route_id]
                if (
                    route_family_id,
                    program_pressure["content_sha256"],
                ) in discovered_pressure_bindings:
                    continue
                signal_payload = {
                    "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
                    "route_id": route_id,
                    "route_family_id": route_family_id,
                    "program_pressure_sha256": program_pressure["content_sha256"],
                    "strategy_native_competitor": bool(
                        str(route.get("execution_domain") or "chemical")
                        in {"enzymatic", "whole_cell", "hybrid", "mechanistic"}
                    ),
                }
                signal_sha256 = _digest(signal_payload)
                signals.append(
                    {
                        "signal_id": (f"event-deficit:program-discovery:{signal_sha256}"),
                        "kind": "program_discovery",
                        "object_id": route_id,
                        "entity_ids": [route_id],
                        "route_family_ids": [route_family_id],
                        "dependency_ids": [],
                        "deterministic": True,
                        "model_allowed": False,
                        "reason": (
                            "strategy_native_program_competes_before_evidence"
                            if before_stock_boundary
                            else "selected_route_requires_program_opportunity_review"
                        ),
                        "score": dict(program_pressure["score"]),
                        "metadata": {
                            **signal_payload,
                            "program_discovery": True,
                            "program_opportunity_pressure": program_pressure,
                        },
                    }
                )
        target_object = str(graph.get("target_molecule_id") or service.kernel.spec.run_id)
        if review_needed:
            signal_sha256 = _digest(
                {
                    "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
                    "operation": "review",
                    "program_review_pressure_sha256": review_pressure["content_sha256"],
                }
            )
            signals.append(
                {
                    "signal_id": f"event-deficit:program-review:{signal_sha256}",
                    "kind": "program_review",
                    "object_id": target_object,
                    "entity_ids": [target_object],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "reason": "canonical_graph_requires_program_projection_review",
                    "score": dict(review_pressure["score"]),
                    "metadata": {
                        "program_review": True,
                        "program_review_pressure_sha256": review_pressure["content_sha256"],
                        "program_review_pressure": review_pressure,
                    },
                }
            )
        if (
            route_rows
            and not before_stock_boundary
            and active.enable_program_admission
            and "program_admission" not in existing_kinds
        ):
            signal_sha256 = _digest(
                {
                    "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
                    "operation": "admit",
                }
            )
            signals.append(
                {
                    "signal_id": f"event-deficit:program-admit:{signal_sha256}",
                    "kind": "program_admission",
                    "object_id": target_object,
                    "entity_ids": [target_object],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "reason": "operator_enabled_shadow_program_admission",
                    "score": {
                        "expected_portfolio_gain": 0.08,
                        "distance_to_closure": 0.08,
                        "evidence_gain": 0.08,
                        "route_diversity_gain": 0.12,
                        "cost_penalty": 0.05,
                        "failure_risk_penalty": 0.02,
                    },
                    "metadata": {"program_admission": True},
                }
            )
        if signals:
            signals_sha256 = _digest(signals)
            service.publish_action_signals(
                signals,
                idempotency_key=(f"unified-program-signals:{signals_sha256[:24]}"),
            )

    def publish_program_validation_signals(
        discovery_result: Mapping[str, Any],
    ) -> None:
        if not active.enable_program_validation:
            return
        frontier = dict(discovery_result.get("experimental_work_frontier") or {})
        route_id = str(frontier.get("route_id") or discovery_result.get("route_id") or "")
        signals = []
        for work_item_id, raw_item in sorted(
            dict(frontier.get("work_items") or {}).items(),
            key=experimental_work_item_rank_key,
        ):
            if not isinstance(raw_item, Mapping):
                continue
            work_item = dict(raw_item)
            scheduling = experimental_work_item_scheduling(work_item)
            item_sha256 = str(work_item.get("content_sha256") or "")
            if not item_sha256 or not scheduling:
                continue
            program_id = str(work_item.get("program_id") or work_item_id)
            signals.append(
                {
                    "signal_id": f"event-deficit:program-validation:{item_sha256}",
                    "kind": "program_validation",
                    "object_id": str(work_item_id),
                    "entity_ids": [program_id],
                    "route_family_ids": [route_id] if route_id else [],
                    "dependency_ids": list(work_item.get("linked_canonical_deficit_ids") or []),
                    "priority": float(scheduling["action_priority"]),
                    "deterministic": True,
                    "model_allowed": False,
                    "reason": "program_candidate_requires_specialized_validation",
                    "score": dict(scheduling["action_score"]),
                    "metadata": {
                        "program_validation": True,
                        "route_id": route_id,
                        "work_item": work_item,
                    },
                }
            )
        if signals:
            signals_sha256 = _digest(signals)
            service.publish_action_signals(
                signals,
                idempotency_key=(f"unified-program-validation-signals:{signals_sha256[:24]}"),
            )

    def publish_unified_experiment_feedback_signals() -> None:
        if resolved_feedback_signals:
            signals_sha256 = _digest(resolved_feedback_signals)
            service.publish_action_signals(
                resolved_feedback_signals,
                idempotency_key=(f"unified-experiment-feedback-signals:{signals_sha256[:24]}"),
            )

    unified_core_handlers = {
        CampaignActionKind.MATERIALIZE: handle_materialize,
        CampaignActionKind.REACTION_VALIDATE: deferred_validation_handler,
        CampaignActionKind.STOCK_AUDIT: handle_stock,
        CampaignActionKind.RECOMPUTE_ROUTE: handle_route_recompute,
        **(
            {CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE: handle_global_architecture}
            if active.enable_codex
            else {}
        ),
        **(
            {CampaignActionKind.CODEX_REPLAN: handle_unified_replan}
            if active.enable_codex and active.enable_replan
            else {}
        ),
        **(
            {CampaignActionKind.CONDITION_ENRICH: handle_condition}
            if resolved_condition_predictor is not None
            else {}
        ),
        **(
            {CampaignActionKind.CHEMENZY_TARGET_EXPAND: handle_target_chemenzy}
            if active.enable_chemenzy and active.enable_target_chemenzy_baseline
            else {}
        ),
        **(
            {CampaignActionKind.CHEMENZY_FRONTIER_EXPAND: handle_guided_chemenzy}
            if active.enable_chemenzy and active.enable_guided_chemenzy
            else {}
        ),
        **(
            {
                CampaignActionKind.ACQUIRE_EVIDENCE: (deferred_unified_evidence_handler),
                CampaignActionKind.BIND_EVIDENCE: (deferred_unified_evidence_handler),
            }
            if resolved_evidence_connector is not None
            else {}
        ),
        **program_action_handlers,
    }
    unified_core_runtime = CampaignActionRuntime(
        service.kernel,
        unified_core_handlers,
        scheduler_policy=active.action_scheduler_policy,
    )
    projected_execution_ids = {
        str(dict(detail.get("action") or {}).get("execution_id") or "")
        for row in stages
        for detail in (dict(row.get("detail") or {}),)
        if str(row.get("stage") or "").startswith(
            "campaign_action_unified_core_"
        )
    }
    preloop_recovered_action_executions = (
        unified_core_runtime.recover_checkpointed_native_actions(
            projected_execution_ids=projected_execution_ids,
        )
    )
    durable_action_history = unified_core_runtime.action_execution_history()
    attempted_guided_frontier_smiles.update(
        _attempted_chemenzy_frontiers_from_action_history(
            durable_action_history
        )
    )
    resumed_guided_progress = _pending_guided_progress_from_action_history(
        durable_action_history,
        stages=stages,
    )
    if resumed_guided_progress:
        guided_progress_pending.update(resumed_guided_progress)
    unified_action_stage_offset = max(
        (
            int(str(row.get("stage") or "").rsplit("_", 1)[-1])
            for row in stages
            if str(row.get("stage") or "").startswith("campaign_action_unified_core_")
            and str(row.get("stage") or "").rsplit("_", 1)[-1].isdigit()
        ),
        default=0,
    )

    def observe_unified_core_execution(
        index: int,
        execution: Mapping[str, Any],
    ) -> None:
        nonlocal chemenzy_observation, evidence_prefetch_result
        nonlocal guided_continuation_allowed
        stage_sequence = unified_action_stage_offset + index
        execution_row = dict(execution)
        preexecuted_action_backlog.append(execution_row)
        action_kind = str(dict(execution_row.get("action") or {}).get("kind") or "")
        handler_result = dict(dict(execution_row.get("outcome") or {}).get("handler_result") or {})
        action_metadata = dict(
            dict(execution_row.get("action") or {}).get("metadata") or {}
        )
        if action_kind == CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value:
            frontier_candidates = [
                action_metadata.get("frontier_smiles"),
                *(handler_result.get("frontier_smiles") or []),
            ]
            for frontier in frontier_candidates:
                if not str(frontier):
                    continue
                _frontier_id, frontier_key = molecule_identity(str(frontier))
                attempted_guided_frontier_smiles.add(
                    frontier_key or str(frontier)
                )
            recovered_progress = dict(
                handler_result.get("guided_progress_checkpoint") or {}
            )
            if recovered_progress and not guided_progress_pending:
                guided_progress_pending.update(recovered_progress)
        unified_material_events.update(
            str(value)
            for value in handler_result.get("material_events") or []
            if isinstance(value, str) and str(value)
        )
        if (
            action_kind
            in {
                CampaignActionKind.CHEMENZY_TARGET_EXPAND.value,
                CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value,
            }
            and int(handler_result.get("provider_invocation_count") or 0) > 0
            and int(handler_result.get("proposal_count") or 0) == 0
        ):
            provider_search_failures.append(
                {
                    "action_kind": action_kind,
                    "frontier_smiles": str(
                        action_metadata.get("frontier_smiles")
                        or dict(handler_result.get("request") or {}).get(
                            "target_smiles"
                        )
                        or ""
                    ),
                    "target_level_native_search": (
                        action_metadata.get("target_level_native_search") is True
                    ),
                    "status": str(handler_result.get("status") or "unresolved"),
                    "provider_invocation_count": int(
                        handler_result.get("provider_invocation_count") or 0
                    ),
                    "failure_reasons": sorted(
                        str(value)
                        for value in handler_result.get("failure_reasons")
                        or handler_result.get("reasons")
                        or []
                        if str(value)
                    ),
                }
            )
            unified_material_events.add(
                "provider_search_exhausted_without_proposal"
            )
        should_measure_guided_progress = bool(
            guided_progress_pending
            and (
                (
                    action_kind
                    == CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value
                    and int(handler_result.get("proposal_count") or 0) == 0
                )
                or action_kind == CampaignActionKind.MATERIALIZE.value
                or action_kind == CampaignActionKind.STOCK_AUDIT.value
            )
        )
        if should_measure_guided_progress:
            pending_materialization = any(
                str(action.get("kind") or "")
                == CampaignActionKind.MATERIALIZE.value
                for action in compile_action_opportunities(
                    dict(service.graph_store.load().get("deficit_frontier") or {})
                ).get("actions")
                or []
                if isinstance(action, Mapping)
            )
            pending_stock = any(
                str(action.get("kind") or "")
                == CampaignActionKind.STOCK_AUDIT.value
                for action in compile_action_opportunities(
                    dict(service.graph_store.load().get("deficit_frontier") or {})
                ).get("actions")
                or []
                if isinstance(action, Mapping)
            )
            if not pending_materialization and not pending_stock:
                progress_gates = current_campaign_gates()
                after_progress = compile_parent_route_stock_progress(
                    latest_campaign_portfolio,
                    parent_route_family_ids=tuple(
                        str(value)
                        for value in guided_progress_pending.get(
                            "parent_route_family_ids"
                        )
                        or []
                        if str(value)
                    ),
                )
                progress_audit = evaluate_guided_stock_progress(
                    dict(guided_progress_pending.get("before") or {}),
                    after_progress,
                    root_b4_reached=(
                        _campaign_milestones(progress_gates).get(
                            "B4_stock_boundary"
                        )
                        is True
                    ),
                )
                guided_continuation_allowed = bool(
                    progress_audit["continue_guided_search"]
                )
                stages.append(
                    _stage(
                        f"guided_root_stock_progress_{stage_sequence:02d}",
                        "continue" if guided_continuation_allowed else "stopped",
                        {
                            **progress_audit,
                            "frontier_smiles": str(
                                guided_progress_pending.get("frontier_smiles") or ""
                            ),
                            "provider_proposal_count": int(
                                guided_progress_pending.get(
                                    "provider_proposal_count"
                                )
                                or 0
                            ),
                        },
                    )
                )
                guided_progress_pending.clear()
        action_signal_id = str(action_metadata.get("action_signal_id") or "")
        execution_status = str(execution_row.get("status") or "unresolved")
        keep_signal_pending = bool(
            action_kind == CampaignActionKind.PROGRAM_VALIDATE.value
            and execution_status == "awaiting_external_result"
        )
        if action_signal_id and not keep_signal_pending:
            service.resolve_action_signals(
                (action_signal_id,),
                resolution={
                    "status": execution_status,
                    "action_execution_id": str(
                        dict(execution_row.get("action") or {}).get("execution_id") or ""
                    ),
                },
                idempotency_key=(
                    "unified-action-signal-resolve:"
                    + str(
                        dict(execution_row.get("action") or {}).get("execution_id")
                        or action_signal_id
                    )
                ),
            )
        if action_kind == CampaignActionKind.CHEMENZY_TARGET_EXPAND.value:
            chemenzy_result = dict(
                dict(execution_row.get("outcome") or {}).get("handler_result") or {}
            )
            chemenzy_observation = _chemenzy_director_observation(
                (
                    {
                        "stage": "chemenzy_baseline",
                        "status": str(chemenzy_result.get("status") or "unresolved"),
                        "detail": chemenzy_result,
                    },
                )
            )
        elif action_kind == CampaignActionKind.PROGRAM_DISCOVER.value:
            publish_program_validation_signals(handler_result)
        elif action_kind == CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST.value:
            resolved_signal_ids = tuple(
                str(value)
                for value in handler_result.get("resolved_program_validation_signal_ids") or []
                if str(value)
            )
            if resolved_signal_ids:
                service.resolve_action_signals(
                    resolved_signal_ids,
                    resolution={
                        "status": "feedback_ingested",
                        "feedback_action_execution_id": str(
                            dict(execution_row.get("action") or {}).get("execution_id") or ""
                        ),
                    },
                    idempotency_key=(
                        "unified-program-validation-resolve:"
                        + str(
                            dict(execution_row.get("action") or {}).get("execution_id")
                            or _digest(resolved_signal_ids)
                        )
                    ),
                )
        elif action_kind == CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE.value:
            director_result = handler_result
            if director_result and not outcomes:
                outcomes.append(director_result)
                unified_material_events.update(_director_topology_replan_events(outcomes))
                unified_material_events.update(
                    _director_depth_replan_events(
                        _planning_depth_requirement(
                            outcomes,
                            minimum_steps=active.minimum_planning_route_steps,
                        )
                    )
                )
                stages.append(
                    _stage(
                        "global_campaign",
                        str(director_result.get("status") or "unresolved"),
                        director_result,
                    )
                )
            template_reuse = self_evo.materialize(service)
            stages.append(
                _stage(
                    "patent_template_reuse",
                    str(template_reuse.get("status") or "unresolved"),
                    template_reuse,
                )
            )
        elif action_kind == CampaignActionKind.CODEX_REPLAN.value:
            if handler_result:
                outcomes.append(handler_result)
                stages.append(
                    _stage(
                        "global_replan",
                        str(handler_result.get("status") or "unresolved"),
                        handler_result,
                    )
                )
            action_execution_id = str(
                dict(execution_row.get("action") or {}).get("execution_id") or ""
            )
            audit_context = unified_replan_audit_contexts.pop(
                action_execution_id,
                None,
            )
            if audit_context is not None:
                retention_audit = _replan_retention_audit(
                    dict(audit_context.get("graph_before") or {}),
                    service.graph_store.load(),
                )
                stages.append(
                    _stage(
                        "replan_retention_audit",
                        "accepted" if retention_audit["accepted"] else "failed",
                        retention_audit,
                    )
                )
                replan_gain = _replan_gain_audit(
                    dict(audit_context.get("gates_before") or {}),
                    current_campaign_gates(),
                    model_cost_before=dict(audit_context.get("model_cost_before") or {}),
                    model_cost_after=service.kernel.state.model_totals,
                )
                stages.append(
                    _stage(
                        "global_replan_gain_audit",
                        str(replan_gain["disposition"]),
                        replan_gain,
                    )
                )
        elif action_kind == CampaignActionKind.REACTION_VALIDATE.value:
            repair = repair_rejected_precursor_typos(service, handler_result)
            if repair.get("status") != "not_needed":
                stages.append(
                    _stage(
                        "precursor_repair",
                        str(repair.get("status") or "unresolved"),
                        repair,
                    )
                )
        elif (
            action_kind == CampaignActionKind.ACQUIRE_EVIDENCE.value
            and action_metadata.get("target_level_evidence_prefetch") is True
        ):
            evidence_prefetch_result = dict(handler_result)
        elif action_kind in {
            CampaignActionKind.ACQUIRE_EVIDENCE.value,
            CampaignActionKind.BIND_EVIDENCE.value,
        }:
            template_learning = self_evo.learn(service.graph_store.load())
            stages.append(
                _stage(
                    "patent_template_learning",
                    str(template_learning.get("status") or "unresolved"),
                    template_learning,
                )
            )
            learned_reuse = self_evo.materialize(service)
            stages.append(
                _stage(
                    "post_learning_template_reuse",
                    str(learned_reuse.get("status") or "unresolved"),
                    learned_reuse,
                )
            )
        publish_unified_replan_signal()
        publish_unified_program_signals()
        stages.append(
            _stage(
                f"campaign_action_unified_core_{stage_sequence:02d}",
                str(execution_row.get("status") or "unresolved"),
                execution_row,
            )
        )
        settlement_gates = current_campaign_gates()
        stages.append(
            _stage(
                f"campaign_snapshot_unified_core_{stage_sequence:02d}",
                "observed",
                current_campaign_snapshot(
                    phase=f"unified_core:{stage_sequence:02d}",
                    gates=settlement_gates,
                    action_decision=dict(execution_row.get("decision") or {}),
                ),
            )
        )

    for recovered_index, recovered_execution in enumerate(
        preloop_recovered_action_executions,
        start=1,
    ):
        observe_unified_core_execution(
            recovered_index,
            recovered_execution,
        )
    unified_action_stage_offset += len(
        preloop_recovered_action_executions
    )

    publish_unified_experiment_feedback_signals()
    unified_core_action_limit = max(
        32,
        active.effective_provider_route_reserve
        + active.max_atom_mapping_reactions
        + active.max_condition_prediction_reactions
        + guided_frontier_limit
        + 8,
    )
    unified_core_loop = unified_core_runtime.run_anytime(
        opportunity_provider=lambda: compile_action_opportunities(
            dict(service.graph_store.load().get("deficit_frontier") or {})
        ),
        milestones_provider=lambda: _campaign_milestones(current_campaign_gates()),
        resource_availability_provider=scheduler_resources,
        max_actions=unified_core_action_limit,
        max_consecutive_no_gain=unified_core_action_limit + 1,
        concurrent_start_kinds=(
            tuple(
                kind
                for kind, enabled in (
                    (
                        CampaignActionKind.CHEMENZY_TARGET_EXPAND,
                        active.enable_chemenzy and active.enable_target_chemenzy_baseline,
                    ),
                    (
                        CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
                        active.enable_codex,
                    ),
                    (
                        CampaignActionKind.ACQUIRE_EVIDENCE,
                        False,
                    ),
                )
                if enabled
            )
            if active.action_scheduler_policy == "adaptive"
            else ()
        ),
        concurrent_action_kinds=(),
        max_concurrent_actions=2,
        stop_milestone=(
            "B4_stock_boundary"
            if active.delivery_boundary == "stock_result"
            else ""
        ),
        progressive_start_kind=(
            CampaignActionKind.CHEMENZY_TARGET_EXPAND
            if active.delivery_boundary == "stock_result"
            else None
        ),
        progressive_delivery_action_kinds=(
            (
                CampaignActionKind.MATERIALIZE,
                CampaignActionKind.STOCK_AUDIT,
            )
            if active.delivery_boundary == "stock_result"
            else ()
        ),
        on_delivery_milestone=(
            result_delivery_cancel_event.set
            if active.delivery_boundary == "stock_result"
            else None
        ),
        on_execution=observe_unified_core_execution,
    )
    anytime_budget_exhausted = bool(
        unified_core_loop.get("termination") == "budget_exhausted"
        or service.kernel.state.status == "budget_exhausted"
    )

    def global_budget_exhausted_now() -> bool:
        state = service.kernel.state
        limits = service.kernel.spec.limits
        return bool(
            anytime_budget_exhausted
            or state.status == "budget_exhausted"
            or state.settled_task_count >= limits.max_total_tasks
            or state.task_wall_time_s >= limits.max_run_wall_time_s
        )

    stages.append(
        _stage(
            "campaign_anytime_core",
            str(unified_core_loop.get("termination") or "unresolved"),
            {key: value for key, value in unified_core_loop.items() if key != "executions"},
        )
    )
    prior_chemenzy_stages = [row for row in stages if row.get("stage") == "chemenzy_baseline"]
    retry_chemenzy = _should_retry_chemenzy_timeout(
        prior_chemenzy_stages,
        resume=resume,
        requested_timeout_s=active.chemenzy_timeout_s,
    )
    if not prior_chemenzy_stages or retry_chemenzy:
        _mark_stage_running(
            checkpoint_path,
            identity,
            stages,
            outcomes,
            "chemenzy_baseline",
            enabled=(active.enable_chemenzy and active.enable_target_chemenzy_baseline),
        )
        chemenzy_action_executions = project_action_results(
            "chemenzy_target_expand",
            (CampaignActionKind.CHEMENZY_TARGET_EXPAND,),
            max_actions=1,
        )
        chemenzy_action_results = _campaign_action_handler_results(
            chemenzy_action_executions,
            kind=CampaignActionKind.CHEMENZY_TARGET_EXPAND,
        )
        chemenzy_stage = (
            chemenzy_action_results[-1]
            if chemenzy_action_results
            else {
                "stage": "chemenzy_proposal",
                "status": "not_scheduled",
                "reason": "scheduler_found_no_target_native_search_action",
                "semantics": {"scheduler_owned_execution": True},
            }
        )
        stages.append(_stage("chemenzy_baseline", chemenzy_stage["status"], chemenzy_stage))
        _checkpoint(checkpoint_path, identity, stages, outcomes)
    chemenzy_observation = _chemenzy_director_observation(stages)
    seed_proposal_count = int(chemenzy_observation.get("proposal_count") or 0)
    if seed_proposal_count > 0:
        seed_executions = project_action_results(
            "chemenzy_seed",
            (
                CampaignActionKind.MATERIALIZE,
                CampaignActionKind.STOCK_AUDIT,
            ),
            max_actions=active.effective_provider_route_reserve + 2,
        )
        seed_materialization_results = [
            dict(dict(value.get("outcome") or {}).get("handler_result") or {})
            for value in seed_executions
            if dict(value.get("action") or {}).get("kind") == CampaignActionKind.MATERIALIZE.value
        ]
        seed_materialization = {
            "schema_version": "campaign_action_slice_summary.v1",
            "status": ("completed" if seed_materialization_results else "reused_or_empty"),
            "changed": any(value.get("changed") is True for value in seed_materialization_results),
            "executed_command_count": sum(
                int(value.get("executed_command_count") or 0)
                for value in seed_materialization_results
            ),
            "action_execution_count": len(seed_materialization_results),
            "semantics": {"scheduler_owned_execution": True},
        }
        stages.append(
            _stage(
                "chemenzy_seed_materialization",
                ("completed" if seed_materialization.get("changed") else "reused_or_empty"),
                seed_materialization,
            )
        )
        seed_stock_results = [
            dict(dict(value.get("outcome") or {}).get("handler_result") or {})
            for value in seed_executions
            if dict(value.get("action") or {}).get("kind") == CampaignActionKind.STOCK_AUDIT.value
        ]
        seed_stock = (
            seed_stock_results[-1]
            if seed_stock_results
            else {
                "stage": "stock",
                "status": "not_scheduled",
                "reason": "scheduler_found_no_executable_stock_deficit",
                "semantics": {"scheduler_owned_execution": True},
            }
        )
        stages.append(
            _stage(
                "chemenzy_seed_stock",
                seed_stock["status"],
                seed_stock,
            )
        )
        seed_portfolio = compile_proof_portfolio(
            service.graph_store.load(),
            acceptance_spec=resolved_acceptance,
            config=_portfolio_config(active, resolved_acceptance),
        )
        seed_gates = compile_blind_acceptance_report(
            preflight=preflight,
            director_outcomes=outcomes,
            graph=service.graph_store.load(),
            portfolio=seed_portfolio,
        )
        seed_milestones = _campaign_milestones(seed_gates)
        stages.append(
            _stage(
                "campaign_milestone",
                "observed",
                {
                    "milestones": seed_milestones,
                    "gates": dict(seed_gates.get("gates") or {}),
                    "counts": dict(seed_gates.get("counts") or {}),
                    "semantics": {
                        "milestones_do_not_select_solver_control_flow": True,
                        "B4_does_not_imply_B2_B3_or_B5": True,
                    },
                },
            )
        )
        seed_graph = service.graph_store.load()
        seed_opportunities = compile_action_opportunities(
            dict(seed_graph.get("deficit_frontier") or {})
        )
        seed_action_decision = schedule_next_action(
            seed_opportunities,
            milestones=seed_milestones,
            resource_availability=scheduler_resources(),
            prior_action_kinds=unified_core_runtime.action_service_history(),
            policy=active.action_scheduler_policy,
        )
        stages.append(
            _stage(
                "campaign_snapshot_chemenzy_seed",
                "observed",
                current_campaign_snapshot(
                    phase="chemenzy_seed",
                    gates=seed_gates,
                    portfolio=seed_portfolio,
                    action_decision=seed_action_decision,
                ),
            )
        )
        _checkpoint(checkpoint_path, identity, stages, outcomes)
    if not outcomes:
        _mark_stage_running(
            checkpoint_path,
            identity,
            stages,
            outcomes,
            "global_campaign",
            model=active.model,
            mode="initial_architecture",
        )
        global_architecture_executions = project_action_results(
            "codex_global_architecture",
            (CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,),
            max_actions=1,
        )
        global_architecture_results = _campaign_action_handler_results(
            global_architecture_executions,
            kind=CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
        )
        initial = (
            global_architecture_results[-1]
            if global_architecture_results
            else {
                "status": "skipped",
                "plan": {},
                "reason": "scheduler_found_no_global_architecture_action",
                "semantics": {"scheduler_owned_execution": True},
            }
        )
        outcomes.append(initial)
        stages.append(_stage("global_campaign", initial["status"], initial))
        _checkpoint(checkpoint_path, identity, stages, outcomes)

    post_director_materialization_executions = project_action_results(
        "post_director_materialize",
        (CampaignActionKind.MATERIALIZE,),
        max_actions=active.effective_provider_route_reserve + 2,
    )
    materialization = _aggregate_materialization_action_results(
        post_director_materialization_executions
    )
    stages.append(
        _stage(
            "materialization",
            "completed" if materialization.get("changed") else "reused_or_empty",
            materialization,
        )
    )
    template_reuse = self_evo.materialize(service)
    stages.append(_stage("patent_template_reuse", template_reuse["status"], template_reuse))

    def run_validation_action_stage(phase: str) -> dict[str, Any]:
        return _aggregate_validation_action_results(
            project_action_results(
                phase,
                (CampaignActionKind.REACTION_VALIDATE,),
                max_actions=active.max_atom_mapping_reactions + 2,
            )
        )

    def run_evidence_action_stage(phase: str) -> dict[str, Any]:
        if (
            active.action_scheduler_policy == "adaptive"
            and _campaign_milestones(current_campaign_gates()).get(
                "B4_stock_boundary"
            )
            is not True
        ):
            return {
                "stage": "evidence_acquisition",
                "status": "deferred",
                "reason": "result_first_stock_boundary_not_reached",
                "model_invocations": 0,
                "visual_invocations": 0,
                "action_execution_count": 0,
                "semantics": {"scheduler_owned_execution": True},
            }
        if phase == "evidence_acquisition" and latest_evidence_action_result:
            prior_result = dict(latest_evidence_action_result)
            return {
                **prior_result,
                "action_execution_count": 1,
                "semantics": {
                    **dict(prior_result.get("semantics") or {}),
                    "scheduler_owned_execution": True,
                    "unified_anytime_action_result_reused": True,
                },
            }

        executions = project_action_results(
            phase,
            (
                CampaignActionKind.ACQUIRE_EVIDENCE,
                CampaignActionKind.BIND_EVIDENCE,
            )
            if resolved_evidence_connector is not None
            else (),
            max_actions=1,
        )
        return _aggregate_evidence_action_results(executions)

    post_director_validation_executions = project_action_results(
        "post_director_validate",
        (CampaignActionKind.REACTION_VALIDATE,),
        max_actions=active.max_atom_mapping_reactions + 2,
    )
    validation = _aggregate_validation_action_results(post_director_validation_executions)
    stages.append(_stage("reaction_validation", validation["status"], validation))
    repair_stage = repair_rejected_precursor_typos(service, validation)
    stages.append(_stage("precursor_repair", repair_stage["status"], repair_stage))
    repair_validation: dict[str, Any] = {}
    if int(repair_stage.get("accepted_repair_count") or 0) > 0:
        repair_validation_executions = project_action_results(
            "precursor_repair_validate",
            (CampaignActionKind.REACTION_VALIDATE,),
            max_actions=active.max_atom_mapping_reactions + 2,
        )
        repair_validation = _aggregate_validation_action_results(repair_validation_executions)
        stages.append(
            _stage(
                "precursor_repair_validation",
                repair_validation["status"],
                repair_validation,
            )
        )
    append_condition_stage("condition_enrichment")

    # Publish an honest, provisional projection as soon as the global Codex
    # architecture has been materialized and host-validated.  Evidence,
    # ChemEnzy frontier work and stock closure continue below, but a user no
    # longer waits for every external timeout before seeing the first route.
    if active.publish_intermediate_workbenches:
        preview = service.publish_workbench(
            campaign_summary={
                "state": "initial_architecture_available",
                "provisional": True,
                "phase": "evidence_and_local_expansion_pending",
                "model_cost": dict(service.kernel.state.model_totals),
                "semantics": {
                    "visible_before_full_closeout": True,
                    "not_a_completed_route_claim": True,
                },
            }
        )
        snapshot = dict(preview.get("snapshot") or {})
        stages.append(
            _stage(
                "initial_workbench",
                "completed",
                {
                    "snapshot_ref": dict(preview.get("snapshot_ref") or {}),
                    "route_count": int(
                        dict(snapshot.get("portfolio") or {}).get("route_count") or 0
                    ),
                    "graph_revision": service.kernel.state.graph_revision,
                    "provisional": True,
                },
            )
        )
        _checkpoint(checkpoint_path, identity, stages, outcomes)

    delegation_audit = _chemenzy_delegation_audit(
        outcomes,
        service.graph_store.load(),
    )
    stages.append(
        _stage(
            "chemenzy_delegation",
            delegation_audit["status"],
            delegation_audit,
        )
    )

    # The target-level ChemEnzy route pool seeds complete multi-step options;
    # guided delegation below expands canonical subtargets selected by Codex.
    prior_attempted_frontiers = _attempted_chemenzy_frontiers(stages)
    prior_guided = _latest_stage(stages, "chemenzy_guided_frontier")
    prior_guided_status = str(prior_guided.get("status") or "")
    reuse_guided = (
        prior_guided_status
        in {
            "completed",
            "unresolved",
            "not_needed",
            "reused",
        }
        or len(prior_attempted_frontiers) >= guided_frontier_limit
    )
    if (
        prior_guided_status == "not_needed"
        and int(delegation_audit.get("queued_count") or 0) > 0
        and len(prior_attempted_frontiers) < guided_frontier_limit
    ):
        reuse_guided = False
    if reuse_guided:
        prior_detail = dict(prior_guided.get("detail") or {})
        guided_stage = {
            **prior_detail,
            "status": "reused",
            "reused_from_status": prior_guided_status or "frontier_budget_already_spent",
            "new_proposal_count": 0,
            "semantics": {
                **dict(prior_detail.get("semantics") or {}),
                "resume_does_not_repeat_provider_call": True,
            },
        }
    else:
        _mark_stage_running(
            checkpoint_path,
            identity,
            stages,
            outcomes,
            "chemenzy_guided_frontier",
            enabled=active.enable_chemenzy and active.enable_guided_chemenzy,
        )
        # Keep at least one local-expansion slot for leaves revealed by the
        # first validation/stock pass.  Spending the whole allowance on the
        # director's initial frontier made every new upstream leaf terminal.
        initial_guided_limit = (
            guided_frontier_limit - 1
            if guided_frontier_limit > 1
            else 1
        )
        guided_opportunities = compile_action_opportunities(
            dict(service.graph_store.load().get("deficit_frontier") or {})
        )
        stock_recovery_only = any(
            row.get("kind") == CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value
            and str(dict(row.get("metadata") or {}).get("stock_observation_id") or "")
            for row in guided_opportunities.get("actions") or []
            if isinstance(row, Mapping)
        ) or any(
            str(dict(execution.get("action") or {}).get("kind") or "")
            == CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value
            and str(
                dict(dict(execution.get("action") or {}).get("metadata") or {}).get(
                    "stock_observation_id"
                )
                or ""
            )
            for execution in preexecuted_action_backlog
            if isinstance(execution, Mapping)
        )
        guided_action_executions = (
            []
            if stock_recovery_only
            else project_action_results(
                "chemenzy_guided_frontier_expand",
                (CampaignActionKind.CHEMENZY_FRONTIER_EXPAND,),
                max_actions=initial_guided_limit,
            )
        )
        guided_stage = _aggregate_guided_chemenzy_action_results(guided_action_executions)
        if stock_recovery_only:
            guided_stage["semantics"] = {
                **dict(guided_stage.get("semantics") or {}),
                "stock_recovery_frontiers_deferred_until_stock_stage": True,
            }
        guided_stage["new_proposal_count"] = int(guided_stage.get("proposal_count") or 0)
    stages.append(_stage("chemenzy_guided_frontier", guided_stage["status"], guided_stage))
    guided_materialization: dict[str, Any] = {}
    guided_validation: dict[str, Any] = {}
    guided_stock: dict[str, Any] = {}
    if int(guided_stage.get("new_proposal_count", guided_stage.get("proposal_count")) or 0) > 0:
        guided_materialization_executions = project_action_results(
            "guided_materialize",
            (CampaignActionKind.MATERIALIZE,),
            max_actions=active.effective_provider_route_reserve + 2,
        )
        guided_materialization = _aggregate_materialization_action_results(
            guided_materialization_executions
        )
        stages.append(
            _stage(
                "guided_materialization",
                "completed" if guided_materialization.get("changed") else "reused_or_empty",
                guided_materialization,
            )
        )
        guided_validation_executions = project_action_results(
            "guided_validate",
            (CampaignActionKind.REACTION_VALIDATE,),
            max_actions=active.max_atom_mapping_reactions + 2,
        )
        guided_validation = _aggregate_validation_action_results(guided_validation_executions)
        stages.append(
            _stage("guided_reaction_validation", guided_validation["status"], guided_validation)
        )

    # Evidence discovery and leaf stock audit deliberately precede the only
    # optional replan.  Their host-owned observations therefore enter the next
    # CampaignContext instead of making the director repeat a blind first pass.
    source_stage = discover_director_source_hints(service, outcomes)
    evidence_prefetch = dict(evidence_prefetch_result)
    source_stage = _merge_prefetched_source_hints(source_stage, evidence_prefetch)
    stages.append(_stage("source_frontier", source_stage["status"], source_stage))
    append_condition_stage("post_guided_condition_enrichment")
    _mark_stage_running(
        checkpoint_path,
        identity,
        stages,
        outcomes,
        "evidence_acquisition",
        visual_enabled=visual_evidence_provider is not None,
    )
    evidence_stage = run_evidence_action_stage("evidence_acquisition")
    evidence_stage["prefetch"] = evidence_prefetch
    evidence_stage["latency_hidden_by_global_s"] = min(
        float(evidence_prefetch.get("elapsed_s") or 0.0),
        float(
            dict(_latest_stage(stages, "global_campaign")).get("elapsed_s")
            or service.kernel.state.model_totals.get("wall_time_s")
            or 0.0
        ),
    )
    stages.append(_stage("evidence_acquisition", evidence_stage["status"], evidence_stage))
    template_learning = self_evo.learn(service.graph_store.load())
    stages.append(
        _stage("patent_template_learning", template_learning["status"], template_learning)
    )
    learned_template_reuse = self_evo.materialize(service)
    stages.append(
        _stage(
            "post_learning_template_reuse",
            learned_template_reuse["status"],
            learned_template_reuse,
        )
    )
    learned_template_validation: dict[str, Any] = {}
    if dict(learned_template_reuse.get("execution") or {}).get("changed") is True:
        learned_template_validation_executions = project_action_results(
            "learned_template_validate",
            (CampaignActionKind.REACTION_VALIDATE,),
            max_actions=active.max_atom_mapping_reactions + 2,
        )
        learned_template_validation = _aggregate_validation_action_results(
            learned_template_validation_executions
        )
        stages.append(
            _stage(
                "post_learning_template_validation",
                learned_template_validation["status"],
                learned_template_validation,
            )
        )
    _mark_stage_running(
        checkpoint_path,
        identity,
        stages,
        outcomes,
        "stock",
        boundary=resolved_acceptance.stock_boundary,
    )
    stock_executions = project_action_results(
        "stock",
        (CampaignActionKind.STOCK_AUDIT,),
        max_actions=4,
    )
    stock_stage = _aggregate_stock_action_results(
        stock_executions,
        graph=service.graph_store.load(),
        required_boundary=resolved_acceptance.stock_boundary,
        max_molecules=active.max_live_stock_molecules,
    )
    stages.append(_stage("stock", stock_stage["status"], stock_stage))

    # A stock rejection can reveal a new upstream-search deficit that did not
    # exist during Codex's initial delegation pass.  Spend only the unused
    # guided-frontier allowance and never repeat an already attempted subtarget.
    recovery_stage: dict[str, Any] = {}
    recovery_materialization: dict[str, Any] = {}
    recovery_validation: dict[str, Any] = {}
    recovery_stock: dict[str, Any] = {}
    attempted_guided_frontiers = {
        *_attempted_chemenzy_frontiers(stages),
        *(str(value) for value in guided_stage.get("frontier_smiles") or [] if str(value)),
    }
    remaining_guided = max(
        0,
        guided_frontier_limit - len(attempted_guided_frontiers),
    )
    if remaining_guided:
        recovery_action_executions = project_action_results(
            "chemenzy_stock_recovery_expand",
            (CampaignActionKind.CHEMENZY_FRONTIER_EXPAND,),
            max_actions=remaining_guided,
        )
        recovery_stage = _aggregate_guided_chemenzy_action_results(recovery_action_executions)
        stages.append(_stage("chemenzy_stock_recovery", recovery_stage["status"], recovery_stage))
    if int(recovery_stage.get("proposal_count") or 0) > 0:
        recovery_materialization_executions = project_action_results(
            "recovery_materialize",
            (CampaignActionKind.MATERIALIZE,),
            max_actions=active.effective_provider_route_reserve + 2,
        )
        recovery_materialization = _aggregate_materialization_action_results(
            recovery_materialization_executions
        )
        stages.append(
            _stage(
                "recovery_materialization",
                "completed" if recovery_materialization.get("changed") else "reused_or_empty",
                recovery_materialization,
            )
        )
        recovery_validation_executions = project_action_results(
            "recovery_validate",
            (CampaignActionKind.REACTION_VALIDATE,),
            max_actions=active.max_atom_mapping_reactions + 2,
        )
        recovery_validation = _aggregate_validation_action_results(recovery_validation_executions)
        stages.append(
            _stage(
                "recovery_reaction_validation",
                recovery_validation["status"],
                recovery_validation,
            )
        )
        recovery_stock_executions = project_action_results(
            "recovery_stock",
            (CampaignActionKind.STOCK_AUDIT,),
            max_actions=4,
        )
        recovery_stock = _aggregate_stock_action_results(
            recovery_stock_executions,
            graph=service.graph_store.load(),
            required_boundary=resolved_acceptance.stock_boundary,
            max_molecules=active.max_live_stock_molecules,
        )
        stages.append(_stage("recovery_stock", recovery_stock["status"], recovery_stock))
        stock_stage = recovery_stock

    provisional = service.closeout(
        idempotency_key=f"solve-target:provisional:{service.kernel.state.graph_revision}",
        config=_portfolio_config(active, resolved_acceptance),
        budget_exhausted=global_budget_exhausted_now(),
    )["portfolio"]
    provisional_gates = compile_blind_acceptance_report(
        preflight=preflight,
        director_outcomes=outcomes,
        graph=service.graph_store.load(),
        portfolio=provisional,
    )
    provisional_planning_depth = _planning_depth_requirement(
        outcomes,
        minimum_steps=active.minimum_planning_route_steps,
    )
    material_events = tuple(
        sorted(
            {
                *_material_replan_events(
                    repair_stage,
                    repair_validation,
                    source_stage,
                    evidence_stage,
                    learned_template_reuse,
                    learned_template_validation,
                    stock_stage,
                    guided_stage,
                    guided_materialization,
                    guided_validation,
                    guided_stock,
                    recovery_stage,
                    recovery_materialization,
                    recovery_validation,
                    recovery_stock,
                ),
                *_director_topology_replan_events(outcomes),
                *_director_depth_replan_events(provisional_planning_depth),
            }
        )
    )
    convergence_ledger = dict(unified_core_loop.get("convergence_ledger") or {})
    replan_pressure = compile_replan_pressure(
        provisional_gates,
        material_events=material_events,
        convergence_ledger=convergence_ledger,
    )
    effective_material_events = tuple(
        sorted(
            {
                *material_events,
                *replan_pressure["derived_material_events"],
            }
        )
    )
    replan_reasons = _replan_reasons(
        provisional_gates,
        material_events=material_events,
        convergence_ledger=convergence_ledger,
    )
    event_replan_count = sum(
        str(outcome.get("mode") or "") == "event_replan"
        for outcome in outcomes
    )
    event_replan_limit = (
        active.max_route_local_repair_rounds if sequential_strategy else 1
    )
    replan_candidate = bool(
        active.enable_codex
        and active.enable_replan
        and replan_reasons
        and _director_outcome_allows_replan(outcomes)
        and event_replan_count < event_replan_limit
        and len(outcomes) < _MAX_DIRECTOR_OUTCOMES
    )
    replan_signal_gate = _replan_signal_gate(
        provisional_gates,
        material_events=material_events,
        trigger_reasons=replan_reasons,
        convergence_ledger=convergence_ledger,
    )
    needs_replan = bool(replan_candidate and replan_signal_gate["accepted"])
    if replan_candidate:
        stages.append(
            _stage(
                "global_replan_signal_gate",
                "accepted" if replan_signal_gate["accepted"] else "skipped",
                replan_signal_gate,
            )
        )
    evidence_observations = {
        **_evidence_observations(evidence_stage),
        **self_evo.observation(dict(learned_template_reuse.get("retrieval") or {})),
        **(
            {"provider_search_failures": list(provider_search_failures)}
            if provider_search_failures
            else {}
        ),
    }
    replan_prompt_context_bytes = 0
    replan_guard = _replan_budget_guard(
        model_cost=service.kernel.state.model_totals,
        budget=resolved_budget,
        config=active,
    )
    if needs_replan and replan_guard["accepted"]:
        try:
            replan_context = service.compile_global_context(
                material_events=effective_material_events,
                evidence_observations=evidence_observations,
            )
            replan_prompt_context_bytes = len(
                director_prompt(
                    replan_context,
                    mode="event_replan",
                    config=director_config,
                ).encode("utf-8")
            )
            replan_guard = _replan_budget_guard(
                model_cost=service.kernel.state.model_totals,
                budget=resolved_budget,
                config=active,
                prompt_context_bytes=replan_prompt_context_bytes,
            )
        except CampaignContextTooLargeError as exc:
            replan_guard = {
                **replan_guard,
                "accepted": False,
                "reasons": ["campaign_context_exceeds_bounded_replan_budget"],
                "context_error": str(exc)[:2_000],
                "semantics": {
                    **dict(replan_guard.get("semantics") or {}),
                    "optional_replan_failure_does_not_abort_campaign": True,
                },
            }
    replan_guard = {
        **replan_guard,
        "prompt_context_bytes": replan_prompt_context_bytes,
        "trigger_reasons": list(replan_reasons),
    }
    replan_executed = False
    model_cost_before_replan: dict[str, Any] = {}
    if needs_replan:
        stages.append(
            _stage(
                "global_replan_budget_gate",
                "accepted" if replan_guard["accepted"] else "skipped",
                replan_guard,
            )
        )
    if needs_replan and replan_guard["accepted"]:
        graph_before_replan = service.graph_store.load()
        model_cost_before_replan = dict(service.kernel.state.model_totals)
        _mark_stage_running(
            checkpoint_path,
            identity,
            stages,
            outcomes,
            "global_replan",
            model=active.model,
            mode="event_replan",
        )
        replan_executions = execute_pending_unified_replan()
        replan_results = _campaign_action_handler_results(
            replan_executions,
            kind=CampaignActionKind.CODEX_REPLAN,
        )
        replan_executed = bool(replan_results)
        replan = (
            replan_results[-1]
            if replan_results
            else {
                "status": "skipped",
                "plan": {},
                "reason": "scheduler_found_no_event_replan_action",
                "semantics": {"scheduler_owned_execution": True},
            }
        )
        if not any(
            str(outcome.get("mode") or "") == "event_replan"
            and str(outcome.get("context_sha256") or "")
            == str(replan.get("context_sha256") or "")
            for outcome in outcomes
        ):
            outcomes.append(replan)
        if not any(
            row.get("stage") == "global_replan"
            and str(dict(row.get("detail") or {}).get("context_sha256") or "")
            == str(replan.get("context_sha256") or "")
            for row in stages
        ):
            stages.append(_stage("global_replan", replan["status"], replan))
        _checkpoint(checkpoint_path, identity, stages, outcomes)
        if replan.get("status") == "accepted" and replan.get("plan"):
            rematerialization_executions = project_action_results(
                "replan_materialize",
                (CampaignActionKind.MATERIALIZE,),
                max_actions=active.effective_provider_route_reserve + 2,
            )
            rematerialization = _aggregate_materialization_action_results(
                rematerialization_executions
            )
            stages.append(
                _stage(
                    "replan_materialization",
                    ("completed" if rematerialization.get("changed") else "reused_or_empty"),
                    rematerialization,
                )
            )
            replan_template_reuse = self_evo.materialize(service)
            stages.append(
                _stage(
                    "replan_patent_template_reuse",
                    replan_template_reuse["status"],
                    replan_template_reuse,
                )
            )
            revalidation_executions = project_action_results(
                "replan_validate",
                (CampaignActionKind.REACTION_VALIDATE,),
                max_actions=active.max_atom_mapping_reactions + 2,
            )
            revalidation = _aggregate_validation_action_results(revalidation_executions)
            stages.append(_stage("replan_validation", revalidation["status"], revalidation))
            replan_repair = repair_rejected_precursor_typos(service, revalidation)
            stages.append(_stage("replan_precursor_repair", replan_repair["status"], replan_repair))
            if int(replan_repair.get("accepted_repair_count") or 0) > 0:
                repaired_revalidation_executions = project_action_results(
                    "replan_repair_validate",
                    (CampaignActionKind.REACTION_VALIDATE,),
                    max_actions=active.max_atom_mapping_reactions + 2,
                )
                repaired_revalidation = _aggregate_validation_action_results(
                    repaired_revalidation_executions
                )
                stages.append(
                    _stage(
                        "replan_precursor_repair_validation",
                        repaired_revalidation["status"],
                        repaired_revalidation,
                    )
                )
            source_stage = discover_director_source_hints(service, outcomes)
            stages.append(_stage("replan_source_frontier", source_stage["status"], source_stage))
            _mark_stage_running(
                checkpoint_path,
                identity,
                stages,
                outcomes,
                "replan_evidence_acquisition",
                visual_enabled=visual_evidence_provider is not None,
            )
            evidence_stage = run_evidence_action_stage("replan_evidence_acquisition")
            stages.append(
                _stage(
                    "replan_evidence_acquisition",
                    evidence_stage["status"],
                    evidence_stage,
                )
            )
            replan_template_learning = self_evo.learn(service.graph_store.load())
            stages.append(
                _stage(
                    "replan_patent_template_learning",
                    replan_template_learning["status"],
                    replan_template_learning,
                )
            )
            replan_stock_executions = project_action_results(
                "replan_stock",
                (CampaignActionKind.STOCK_AUDIT,),
                max_actions=4,
            )
            stock_stage = _aggregate_stock_action_results(
                replan_stock_executions,
                graph=service.graph_store.load(),
                required_boundary=resolved_acceptance.stock_boundary,
                max_molecules=active.max_live_stock_molecules,
            )
            stages.append(_stage("replan_stock", stock_stage["status"], stock_stage))
        retention_audit = _replan_retention_audit(
            graph_before_replan,
            service.graph_store.load(),
        )
        stages.append(
            _stage(
                "replan_retention_audit",
                "accepted" if retention_audit["accepted"] else "failed",
                retention_audit,
            )
        )

    result_first_credibility_enabled = bool(
        active.action_scheduler_policy != "adaptive"
        or _campaign_milestones(current_campaign_gates()).get(
            "B4_stock_boundary"
        )
        is True
    )
    program_discovery_changed = False
    if active.enable_program_discovery and result_first_credibility_enabled:
        program_graph = service.graph_store.load()
        program_portfolio = compile_proof_portfolio(
            program_graph,
            acceptance_spec=resolved_acceptance,
            config=_portfolio_config(active, resolved_acceptance),
        )
        program_route_rows = [
            dict(route)
            for route in program_portfolio.get("selected_routes") or []
            if str(route.get("route_id") or "")
            and _route_has_canonical_edges(route)
        ][: active.max_program_routes]
        program_route_ids = [str(route.get("route_id") or "") for route in program_route_rows]
        program_discovery_deficits = []
        for route in program_route_rows:
            route_id = str(route.get("route_id") or "")
            program_pressure = current_program_opportunity_pressure(program_graph, route)
            signal = {
                "graph_revision": service.kernel.state.graph_revision,
                "graph_scientific_sha256": str(program_graph.get("scientific_sha256") or ""),
                "route_id": route_id,
                "capability_catalog_available": bool(resolved_program_capabilities),
                "mechanism_proposal_count": len(resolved_mechanism_proposals),
                "program_pressure_sha256": program_pressure["content_sha256"],
            }
            signal_sha256 = _digest(signal)
            program_discovery_deficits.append(
                {
                    "deficit_id": (f"event-deficit:program-discovery:{signal_sha256}"),
                    "kind": "program_discovery",
                    "object_id": route_id,
                    "entity_ids": [route_id],
                    "route_family_ids": [
                        str(
                            next(
                                (
                                    route.get("route_family_id")
                                    for route in program_portfolio.get("selected_routes") or []
                                    if str(route.get("route_id") or "") == route_id
                                ),
                                "",
                            )
                        )
                    ],
                    "dependency_ids": [],
                    "deterministic": True,
                    "model_allowed": False,
                    "reason": "selected_route_requires_program_opportunity_review",
                    "priority": float(program_pressure["legacy_priority"]),
                    "score": dict(program_pressure["score"]),
                    "metadata": {
                        **signal,
                        "program_discovery": True,
                        "event_signal_sha256": signal_sha256,
                        "program_opportunity_pressure": program_pressure,
                    },
                }
            )
        program_discovery_executions = (
            project_action_results(
                "program_discovery",
                (CampaignActionKind.PROGRAM_DISCOVER,),
                max_actions=len(program_discovery_deficits),
            )
            if program_discovery_deficits
            else []
        )
        program_discovery_results = _campaign_action_handler_results(
            program_discovery_executions,
            kind=CampaignActionKind.PROGRAM_DISCOVER,
        )
        program_discovery_changed = any(
            dict(result.get("canonical_ingestion") or {}).get("changed") is True
            for result in program_discovery_results
        )
        program_discovery = {
            "schema_version": "campaign_program_discovery_summary.v1",
            "status": (
                "completed"
                if program_discovery_results
                else "unavailable"
                if program_capability_error
                else "not_needed"
            ),
            "route_count": len(program_route_ids),
            "action_execution_count": len(program_discovery_results),
            "candidate_count": sum(
                int(result.get("candidate_count") or 0) for result in program_discovery_results
            ),
            "program_draft_candidate_ids": sorted(
                {
                    str(value)
                    for result in program_discovery_results
                    for value in result.get("program_draft_candidate_ids") or []
                    if str(value)
                }
            ),
            "execution_program_draft_candidate_ids": sorted(
                {
                    str(value)
                    for result in program_discovery_results
                    for value in result.get("execution_program_draft_candidate_ids") or []
                    if str(value)
                }
            ),
            "mechanism_hypothesis_count": sum(
                int(result.get("mechanism_hypothesis_count") or 0)
                for result in program_discovery_results
            ),
            "canonical_ingestion_changed": program_discovery_changed,
            "capability_error": program_capability_error,
            "results": program_discovery_results,
            "semantics": {
                "target_names_are_not_matching_inputs": True,
                "program_candidates_are_proposal_only": True,
                "conventional_routes_remain_fallbacks": True,
            },
        }
        stages.append(
            _stage(
                "program_discovery",
                program_discovery["status"],
                program_discovery,
            )
        )
    if program_discovery_changed:
        program_materialization = _aggregate_materialization_action_results(
            project_action_results(
                "program_materialize",
                (CampaignActionKind.MATERIALIZE,),
                max_actions=active.effective_provider_route_reserve + 2,
            )
        )
        stages.append(
            _stage(
                "program_materialization",
                ("completed" if program_materialization.get("changed") else "reused_or_empty"),
                program_materialization,
            )
        )
        program_validation = run_validation_action_stage("program_validate")
        stages.append(
            _stage(
                "program_reaction_validation",
                program_validation["status"],
                program_validation,
            )
        )
        program_stock = _aggregate_stock_action_results(
            project_action_results(
                "program_stock",
                (CampaignActionKind.STOCK_AUDIT,),
                max_actions=4,
            ),
            graph=service.graph_store.load(),
            required_boundary=resolved_acceptance.stock_boundary,
            max_molecules=active.max_live_stock_molecules,
        )
        stages.append(_stage("program_stock", program_stock["status"], program_stock))
    if result_first_credibility_enabled:
        append_condition_stage("final_condition_enrichment")
    if active.enable_program_review and result_first_credibility_enabled:
        program_review_executions = project_action_results(
            "program_review",
            (CampaignActionKind.PROGRAM_REVIEW,),
            max_actions=1,
        )
        program_review_results = _campaign_action_handler_results(
            program_review_executions,
            kind=CampaignActionKind.PROGRAM_REVIEW,
        )
        program_review = (
            program_review_results[-1]
            if program_review_results
            else {
                "status": "not_scheduled",
                "reason": "scheduler_found_no_program_review_action",
            }
        )
        stages.append(
            _stage(
                "program_review", str(program_review.get("status") or "unresolved"), program_review
            )
        )
    if active.enable_program_admission and result_first_credibility_enabled:
        program_admission_executions = project_action_results(
            "program_admission",
            (CampaignActionKind.PROGRAM_ADMIT,),
            max_actions=1,
        )
        program_admission_results = _campaign_action_handler_results(
            program_admission_executions,
            kind=CampaignActionKind.PROGRAM_ADMIT,
        )
        program_admission = (
            program_admission_results[-1]
            if program_admission_results
            else {
                "status": "not_scheduled",
                "reason": "scheduler_found_no_program_admission_action",
            }
        )
        stages.append(
            _stage(
                "program_admission",
                str(program_admission.get("status") or "unresolved"),
                program_admission,
            )
        )
    _mark_stage_running(
        checkpoint_path,
        identity,
        stages,
        outcomes,
        "closeout",
    )
    closeout = service.closeout(
        idempotency_key=f"solve-target:closeout:{service.kernel.state.graph_revision}",
        config=_portfolio_config(active, resolved_acceptance),
        budget_exhausted=global_budget_exhausted_now(),
    )
    stages.append(
        _stage(
            "closeout",
            "completed",
            {
                "portfolio_accepted": closeout["portfolio"].get("accepted") is True,
                "selected_route_count": len(closeout["portfolio"].get("selected_routes") or []),
            },
        )
    )
    gates = compile_blind_acceptance_report(
        preflight=preflight,
        director_outcomes=outcomes,
        graph=service.graph_store.load(),
        portfolio=closeout["portfolio"],
    )
    paper_equivalent = compile_paper_equivalent_metric(
        gates,
        stock_oracle=resolved_stock_oracle.to_dict(),
    )
    stages.append(
        _stage(
            "paper_equivalent_metric",
            (
                "solved"
                if paper_equivalent["paper_equivalent_solved"]
                else "unsolved"
            ),
            paper_equivalent,
        )
    )
    chemenzy_lineage = _compile_chemenzy_route_lineage(
        stages,
        service.graph_store.load(),
        gates=gates,
    )
    if chemenzy_lineage["route_count"]:
        stages.append(
            _stage(
                "chemenzy_route_lineage",
                "completed",
                chemenzy_lineage,
            )
        )
    resource_envelope = _resource_envelope(
        model_cost=service.kernel.state.model_totals,
        native_search=service.kernel.native_search_budget(),
        task_budget=service.kernel.task_budget(),
        run_wall_time_s=service.kernel.state.task_wall_time_s,
        attempt_count=service.kernel.state.attempt_count,
        accepted_expansion_count=service.kernel.state.accepted_expansion_count,
        budget=resolved_budget,
        max_run_wall_time_s=service.kernel.spec.limits.max_run_wall_time_s,
    )
    final_graph = service.graph_store.load()
    final_opportunities = compile_action_opportunities(
        dict(final_graph.get("deficit_frontier") or {})
    )
    final_action_decision = schedule_next_action(
        final_opportunities,
        milestones=_campaign_milestones(gates),
        resource_availability=scheduler_resources(),
        prior_action_kinds=unified_core_runtime.action_service_history(),
        policy=active.action_scheduler_policy,
    )
    stages.append(
        _stage(
            "campaign_action_schedule",
            "selected" if final_action_decision["selected_action_id"] else "empty",
            final_action_decision,
        )
    )
    stages.append(
        _stage(
            "campaign_snapshot_closeout",
            "observed",
            current_campaign_snapshot(
                phase="closeout",
                gates=gates,
                portfolio=closeout["portfolio"],
                action_decision=final_action_decision,
            ),
        )
    )
    trajectory = compile_campaign_trajectory(snapshots_from_stages(stages))
    campaign_accepted = _record_campaign_acceptance(
        service,
        gates=gates,
        resource_envelope=resource_envelope,
        idempotency_key=(f"solve-target:campaign-acceptance:{service.kernel.state.graph_revision}"),
    )
    if replan_executed:
        replan_gain = _replan_gain_audit(
            provisional_gates,
            gates,
            model_cost_before=model_cost_before_replan,
            model_cost_after=service.kernel.state.model_totals,
        )
        stages.append(
            _stage(
                "global_replan_gain_audit",
                str(replan_gain["disposition"]),
                replan_gain,
            )
        )
    service.terminalize_global_budget_if_reached(
        idempotency_key=(
            f"solve-target:pre-disposition-global-budget:{service.kernel.state.revision}"
        )
    )
    stop_preview = service.kernel.decide_stop().to_dict()
    continuation_exhausted = _automatic_continuation_exhausted(
        resumed_completed_checkpoint=resumed_completed_checkpoint,
        baseline=continuation_baseline,
        current=_automatic_continuation_baseline(service),
        portfolio_accepted=campaign_accepted,
    )
    director_outcome_limit_exhausted = bool(
        not campaign_accepted and len(outcomes) >= _MAX_DIRECTOR_OUTCOMES
    )
    if director_outcome_limit_exhausted:
        stages.append(
            _stage(
                "director_outcome_limit",
                "exhausted",
                {
                    "observed_outcomes": len(outcomes),
                    "maximum_outcomes": _MAX_DIRECTOR_OUTCOMES,
                    "semantics": {
                        "outcome_limit_is_not_scientific_acceptance": True,
                        "resume_cannot_create_an_additional_director_outcome": True,
                    },
                },
            )
        )
        _transition_unresolved_if_active(
            service.kernel,
            idempotency_key=(
                f"solve-target:director-outcome-limit:{service.kernel.state.revision}"
            ),
            reasons=(
                "director_outcome_limit_exhausted",
                "configured_scientific_acceptance_not_met",
            ),
        )
    elif continuation_exhausted:
        stages.append(
            _stage(
                "automatic_continuation",
                "exhausted",
                {
                    "reason": "no_scientific_progress_after_completed_automatic_pass",
                    "baseline": continuation_baseline,
                    "current": _automatic_continuation_baseline(service),
                    "semantics": {
                        "terminal_unresolved_is_not_scientific_acceptance": True,
                        "new_evidence_or_a_new_campaign_can_be_run_separately": True,
                    },
                },
            )
        )
        _transition_unresolved_if_active(
            service.kernel,
            idempotency_key=(
                f"solve-target:automatic-continuation-exhausted:{service.kernel.state.revision}"
            ),
            reasons=(
                "automatic_continuation_exhausted_no_scientific_progress",
                "new_evidence_or_new_campaign_required",
            ),
        )
    elif stop_preview.get("decision") == "continue":
        service.kernel.transition(
            "paused",
            idempotency_key=f"solve-target:bounded-pass:{service.kernel.state.revision}",
            reasons=("bounded_pass_complete_requires_resume",),
        )
    stop = service.kernel.apply_stop_decision(
        idempotency_key=f"solve-target:stop:{service.kernel.state.revision}"
    ).to_dict()
    profile_projection = service.workbench()["snapshot"]
    quality_state = compile_campaign_quality_state(
        workbench=profile_projection,
        gates=gates,
    )
    claim = _claim(
        gates,
        resolved_acceptance,
        resource_envelope,
        objective_mode=active.objective_mode,
        workbench=profile_projection,
    )
    planning_depth = _planning_depth_requirement(
        outcomes,
        minimum_steps=active.minimum_planning_route_steps,
    )
    current_disposition = _current_disposition(
        kernel_status=service.kernel.state.status,
        stop_decision=stop,
        claim=claim,
        gates=gates,
    )
    workbench = service.publish_workbench(
        campaign_summary=_workbench_campaign_summary(
            gates=gates,
            resource_envelope=resource_envelope,
            model_cost=service.kernel.state.model_totals,
            stop_decision=stop,
            claim=claim,
            current_disposition=current_disposition,
            planning_depth=planning_depth,
            quality_state=quality_state,
            trajectory=trajectory,
        )
    )
    report_stages = _deduplicate_stages(stages)
    candidate_lifecycle = compile_candidate_lifecycle(
        final_graph,
        closeout["portfolio"],
        ingestion_observations=report_stages,
    )
    candidate_provenance = compile_candidate_provenance(
        candidate_lifecycle,
        lineage_observations=report_stages,
    )
    report = {
        "schema_version": TARGET_SOLVE_REPORT_SCHEMA,
        "run_id": identity,
        "run_dir": str(directory),
        "target": {"name": case.target_name, "canonical_smiles": canonical},
        "campaign_spec": service.kernel.spec.campaign_spec.to_dict(),
        "preflight": preflight,
        "config": asdict(active),
        "acceptance": resolved_acceptance.to_dict(),
        "budget": resolved_budget.to_dict(),
        "director_outcomes": outcomes,
        "stages": report_stages,
        "trajectory": trajectory,
        "candidate_lifecycle": candidate_lifecycle,
        "candidate_provenance": candidate_provenance,
        "next_action": {
            "selected_action_id": final_action_decision["selected_action_id"],
            "selected_action": final_action_decision["selected_action"],
            "candidate_count": final_action_decision["candidate_count"],
            "eligible_candidate_count": final_action_decision["eligible_candidate_count"],
            "decision_sha256": final_action_decision["content_sha256"],
        },
        "gates": gates,
        "paper_equivalent": paper_equivalent,
        "quality_state": quality_state,
        "planning_depth": planning_depth,
        "model_cost": dict(service.kernel.state.model_totals),
        "resource_envelope": resource_envelope,
        "attempt_count": service.kernel.state.attempt_count,
        "accepted_expansion_count": service.kernel.state.accepted_expansion_count,
        "stop_decision": stop,
        "current_disposition": current_disposition,
        "self_evolution": self_evo.report(),
        "portfolio_ref": closeout["portfolio_ref"],
        "workbench_ref": workbench["snapshot_ref"],
        "claim": claim,
    }
    return _persist_target_report(
        service,
        report=report,
        identity=identity,
        directory=directory,
        checkpoint_path=checkpoint_path,
        outcomes=outcomes,
    )


def _acceptance_input(value: RetrosynthesisAcceptanceSpec) -> dict[str, Any]:
    return {
        "minimum_complete_routes": value.minimum_complete_routes,
        "minimum_edge_proof_level": value.minimum_edge_proof_level,
        "minimum_independent_source_groups": value.minimum_independent_source_groups,
        "stock_boundary": value.stock_boundary,
    }


def _resolve_program_capabilities(
    supplied: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
    *,
    configured_path: str,
    repository_root: Path,
) -> tuple[Mapping[str, Any] | tuple[dict[str, Any], ...], str]:
    if isinstance(supplied, Mapping):
        return dict(supplied), ""
    if supplied is not None:
        return (
            tuple(dict(value) for value in supplied if isinstance(value, Mapping)),
            "",
        )
    path = (
        Path(configured_path).expanduser().resolve()
        if str(configured_path).strip()
        else repository_root / "config" / "route_innovation_capabilities.v1.json"
    )
    if not path.is_file():
        return (), f"program_capability_catalog_missing:{path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (), f"program_capability_catalog_invalid:{type(exc).__name__}:{exc}"
    if isinstance(value, Mapping):
        return dict(value), ""
    if isinstance(value, list):
        return tuple(dict(row) for row in value if isinstance(row, Mapping)), ""
    return (), "program_capability_catalog_not_mapping_or_list"


def _portfolio_config(
    config: TargetSolveConfig,
    acceptance: RetrosynthesisAcceptanceSpec,
) -> PortfolioConfig:
    minimum = min(12, max(1, int(acceptance.minimum_complete_routes)))
    maximum = min(12, max(minimum, int(config.display_route_limit)))
    return PortfolioConfig(
        minimum_routes_to_show=minimum,
        maximum_routes_to_show=maximum,
    )


def _campaign_milestones(gates: Mapping[str, Any]) -> dict[str, bool]:
    values = dict(gates.get("gates") or {})
    milestones = {
        str(name): values.get(name) is True
        for name in (
            "B0_blind_input",
            "B1_global_multi_route",
            "B2_host_validated_routes",
            "B3_exact_multi_source",
            "B4_stock_boundary",
            "B5_configured_portfolio_acceptance",
        )
    }
    milestones["target_rooted_route_exists"] = int(
        dict(gates.get("counts") or {}).get("target_rooted_distinct_skeletons")
        or 0
    ) > 0
    return milestones


def _campaign_action_handler_results(
    executions: Iterable[Mapping[str, Any]],
    *,
    kind: CampaignActionKind,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for execution in executions:
        if dict(execution.get("action") or {}).get("kind") != kind.value:
            continue
        outcome = dict(execution.get("outcome") or {})
        raw_result = outcome.get("handler_result")
        if not isinstance(raw_result, Mapping):
            continue
        result = dict(raw_result)
        result.setdefault(
            "status",
            str(outcome.get("status") or execution.get("status") or "unresolved"),
        )
        failure_reasons = [
            str(value) for value in outcome.get("failure_reasons") or [] if str(value)
        ]
        if failure_reasons and not result.get("reasons"):
            result["reasons"] = failure_reasons
        results.append(result)
    return results


def _aggregate_materialization_action_results(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    results = _campaign_action_handler_results(
        executions,
        kind=CampaignActionKind.MATERIALIZE,
    )
    material_events = [
        dict(event)
        for result in results
        for event in result.get("material_events") or []
        if isinstance(event, Mapping)
    ]
    return {
        "schema_version": "campaign_action_materialization_summary.v1",
        "status": "completed" if results else "reused_or_empty",
        "changed": any(result.get("changed") is True for result in results),
        "executed_command_count": sum(
            int(result.get("executed_command_count") or 0) for result in results
        ),
        "action_execution_count": len(results),
        "material_events": material_events,
        "semantics": {
            "scheduler_owned_execution": True,
            "canonical_ingestion_remains_authoritative": True,
        },
    }


def _aggregate_validation_action_results(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    results = _campaign_action_handler_results(
        executions,
        kind=CampaignActionKind.REACTION_VALIDATE,
    )
    accepted_ids = sorted(
        {
            str(edge_id)
            for result in results
            for edge_id in result.get("accepted_edge_ids") or []
            if str(edge_id)
        }
    )
    rejected_ids = sorted(
        {
            str(edge_id)
            for result in results
            for edge_id in result.get("rejected_edge_ids") or []
            if str(edge_id)
        }
        - set(accepted_ids)
    )
    diagnostics_by_edge = {
        str(row.get("edge_id") or ""): dict(row)
        for result in results
        for row in result.get("rejection_diagnostics") or []
        if isinstance(row, Mapping) and str(row.get("edge_id") or "")
    }
    rejection_reason_counts: dict[str, int] = {}
    for row in diagnostics_by_edge.values():
        for reason in row.get("reasons") or []:
            name = str(reason)
            if name:
                rejection_reason_counts[name] = rejection_reason_counts.get(name, 0) + 1
    mapped_reactions: dict[str, Any] = {}
    mapping_failures: list[dict[str, Any]] = []
    mapping_backends: set[str] = set()
    for result in results:
        mapping = dict(result.get("mapping") or {})
        mapped_reactions.update(dict(mapping.get("mapped_reactions") or {}))
        mapping_failures.extend(
            dict(row) for row in mapping.get("failures") or [] if isinstance(row, Mapping)
        )
        if str(mapping.get("backend") or ""):
            mapping_backends.add(str(mapping.get("backend")))
    statuses = {str(result.get("status") or "") for result in results}
    status = (
        "reused_or_empty"
        if not results
        else "partial"
        if statuses.intersection({"partial", "failed", "error"})
        else "completed"
    )
    material_events = [
        dict(event)
        for result in results
        for event in dict(result.get("execution") or {}).get("material_events") or []
        if isinstance(event, Mapping)
    ]
    return {
        "stage": "reaction_validation",
        "schema_version": "campaign_action_validation_summary.v1",
        "status": status,
        "pending_edge_count": sum(int(result.get("pending_edge_count") or 0) for result in results),
        "forced_revalidation_edge_count": sum(
            int(result.get("forced_revalidation_edge_count") or 0) for result in results
        ),
        "validation_command_count": sum(
            int(result.get("validation_command_count") or 0) for result in results
        ),
        "accepted_validation_count": len(accepted_ids),
        "rejected_validation_count": len(rejected_ids),
        "accepted_edge_ids": accepted_ids,
        "rejected_edge_ids": rejected_ids,
        "rejection_diagnostics": [diagnostics_by_edge[key] for key in sorted(diagnostics_by_edge)],
        "rejection_reason_counts": dict(
            sorted(
                rejection_reason_counts.items(),
                key=lambda row: (-row[1], row[0]),
            )
        ),
        "mapping": {
            "schema_version": "campaign_action_mapping_summary.v1",
            "backend": "+".join(sorted(mapping_backends)),
            "requested_count": sum(
                int(dict(result.get("mapping") or {}).get("requested_count") or 0)
                for result in results
            ),
            "mapped_count": len(mapped_reactions),
            "failure_count": len(mapping_failures),
            "truncated": any(
                dict(result.get("mapping") or {}).get("truncated") is True for result in results
            ),
            "mapped_reactions": mapped_reactions,
            "failures": mapping_failures,
        },
        "execution": {
            "executed_command_count": sum(
                int(dict(result.get("execution") or {}).get("executed_command_count") or 0)
                for result in results
            ),
            "material_events": material_events,
        },
        "action_execution_count": len(results),
        "semantics": {
            "scheduler_owned_execution": True,
            "diagnostics_preserved_for_precursor_repair": True,
        },
    }


def _scope_validation_summary(
    validation: Mapping[str, Any],
    edge_ids: Iterable[str],
) -> dict[str, Any]:
    selected = {str(value) for value in edge_ids if str(value)}
    if not selected or not validation:
        return {}
    accepted_ids = sorted(
        selected.intersection(str(value) for value in validation.get("accepted_edge_ids") or [])
    )
    rejected_ids = sorted(
        selected.intersection(str(value) for value in validation.get("rejected_edge_ids") or [])
    )
    diagnostics = [
        dict(row)
        for row in validation.get("rejection_diagnostics") or []
        if isinstance(row, Mapping) and str(row.get("edge_id") or "") in selected
    ]
    reason_counts: dict[str, int] = {}
    for row in diagnostics:
        for reason in row.get("reasons") or []:
            name = str(reason)
            if name:
                reason_counts[name] = reason_counts.get(name, 0) + 1
    return {
        **dict(validation),
        "accepted_validation_count": len(accepted_ids),
        "rejected_validation_count": len(rejected_ids),
        "accepted_edge_ids": accepted_ids,
        "rejected_edge_ids": rejected_ids,
        "rejection_diagnostics": diagnostics,
        "rejection_reason_counts": dict(
            sorted(reason_counts.items(), key=lambda row: (-row[1], row[0]))
        ),
        "scoped_edge_ids": sorted(selected),
        "semantics": {
            **dict(validation.get("semantics") or {}),
            "validation_attribution_is_edge_scoped": True,
        },
    }


def _aggregate_stock_action_results(
    executions: Iterable[Mapping[str, Any]],
    *,
    graph: Mapping[str, Any] | None = None,
    required_boundary: str = "",
    max_molecules: int = 24,
) -> dict[str, Any]:
    results = _campaign_action_handler_results(
        executions,
        kind=CampaignActionKind.STOCK_AUDIT,
    )
    projection = (
        project_existing_stock_audit(
            graph,
            required_boundary=required_boundary,
            max_molecules=max_molecules,
        )
        if graph is not None and required_boundary
        else {}
    )
    if projection.get("status") == "reused":
        executed_command_count = sum(
            int(dict(result.get("execution") or {}).get("executed_command_count") or 0)
            for result in results
        )
        best_result = max(
            results,
            key=lambda result: (
                int(result.get("selected_leaf_count") or 0),
                int(result.get("selected_stock_candidate_count") or 0),
                int(result.get("stock_closed_leaf_count") or 0),
            ),
            default={},
        )
        status = "reused"
        if executed_command_count:
            status = (
                "completed"
                if int(projection.get("stock_closed_candidate_count") or 0)
                == int(projection.get("selected_stock_candidate_count") or 0)
                else "partial"
            )
        merged = {
            **best_result,
            **projection,
            "status": status,
            "action_execution_count": len(results),
            "execution": {"executed_command_count": executed_command_count},
            "semantics": {
                **dict(best_result.get("semantics") or {}),
                **dict(projection.get("semantics") or {}),
                "scheduler_owned_execution": True,
            },
        }
        if "reason" not in projection:
            merged.pop("reason", None)
        return merged
    if not results:
        return {
            "stage": "stock",
            "status": "reused_or_empty",
            "reason": "scheduler_found_no_executable_stock_deficit",
            "action_execution_count": 0,
            "semantics": {"scheduler_owned_execution": True},
        }
    return {
        **results[-1],
        "action_execution_count": len(results),
        "semantics": {
            **dict(results[-1].get("semantics") or {}),
            "scheduler_owned_execution": True,
        },
    }


def _aggregate_condition_action_results(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    results = _campaign_action_handler_results(
        executions,
        kind=CampaignActionKind.CONDITION_ENRICH,
    )
    if not results:
        return {
            "stage": "condition_enrichment",
            "status": "reused_or_empty",
            "reason": "scheduler_found_no_executable_condition_deficit",
            "action_execution_count": 0,
            "semantics": {"scheduler_owned_execution": True},
        }
    enriched_ids = sorted(
        {
            str(edge_id)
            for result in results
            for edge_id in result.get("enriched_edge_ids") or []
            if str(edge_id)
        }
    )
    failed_ids = sorted(
        {
            str(edge_id)
            for result in results
            for edge_id in result.get("failed_edge_ids") or []
            if str(edge_id)
        }
        - set(enriched_ids)
    )
    prediction_errors = {
        str(edge_id): str(reason)
        for result in results
        for edge_id, reason in dict(result.get("prediction_errors") or {}).items()
        if str(edge_id)
    }
    statuses = {str(result.get("status") or "") for result in results}
    material_events = [
        dict(event)
        for result in results
        for event in dict(result.get("execution") or {}).get("material_events") or []
        if isinstance(event, Mapping)
    ]
    return {
        "stage": "condition_enrichment",
        "status": (
            "partial"
            if failed_ids or statuses.intersection({"partial", "failed", "error"})
            else "completed"
        ),
        "pending_edge_count": sum(int(result.get("pending_edge_count") or 0) for result in results),
        "selected_edge_count": sum(
            int(result.get("selected_edge_count") or 0) for result in results
        ),
        "condition_command_count": sum(
            int(result.get("condition_command_count") or 0) for result in results
        ),
        "enriched_edge_count": len(enriched_ids),
        "failed_edge_count": len(failed_ids),
        "enriched_edge_ids": enriched_ids,
        "failed_edge_ids": failed_ids,
        "prediction_errors": prediction_errors,
        "execution": {
            "executed_command_count": sum(
                int(dict(result.get("execution") or {}).get("executed_command_count") or 0)
                for result in results
            ),
            "material_events": material_events,
        },
        "action_execution_count": len(results),
        "semantics": {
            "scheduler_owned_execution": True,
            "prediction_grants_no_reaction_or_evidence_proof": True,
        },
    }


def _aggregate_evidence_action_results(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(executions)
    results = [
        *_campaign_action_handler_results(
            rows,
            kind=CampaignActionKind.ACQUIRE_EVIDENCE,
        ),
        *_campaign_action_handler_results(
            rows,
            kind=CampaignActionKind.BIND_EVIDENCE,
        ),
    ]
    if not results:
        return {
            "stage": "evidence_acquisition",
            "status": "unresolved",
            "reason": "scheduler_found_no_executable_evidence_deficit",
            "model_invocations": 0,
            "visual_invocations": 0,
            "action_execution_count": 0,
            "semantics": {"scheduler_owned_execution": True},
        }
    return {
        **results[-1],
        "action_execution_count": len(results),
        "semantics": {
            **dict(results[-1].get("semantics") or {}),
            "scheduler_owned_execution": True,
        },
    }


def _aggregate_guided_chemenzy_action_results(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    results = _campaign_action_handler_results(
        executions,
        kind=CampaignActionKind.CHEMENZY_FRONTIER_EXPAND,
    )
    proposal_count = sum(int(result.get("proposal_count") or 0) for result in results)
    frontier_smiles = sorted(
        {
            str(value)
            for result in results
            for value in result.get("frontier_smiles") or []
            if str(value)
        }
    )
    provider_results = [
        dict(row)
        for result in results
        for row in result.get("results") or []
        if isinstance(row, Mapping)
    ]
    route_lineage = [
        {
            **dict(lineage),
            "provider_mode": str(result.get("mode") or "guided_frontier"),
            "provider_scope": str(result.get("scope") or ""),
            "provider_request_sha256": str(result.get("request_sha256") or ""),
            "provider_raw_proposal_sha256": str(result.get("raw_proposal_sha256") or ""),
            "provider_raw_result_sha256": str(result.get("raw_result_sha256") or ""),
            "provider_replay_key_sha256": str(result.get("replay_key_sha256") or ""),
            "provider_random_seed": int(result.get("random_seed") or 0),
        }
        for result in provider_results
        for lineage in result.get("route_lineage") or []
        if isinstance(lineage, Mapping)
    ]
    return {
        "schema_version": "v4_chemenzy_guided_frontier_stage.v1",
        "stage": "chemenzy_guided_frontier",
        "status": ("completed" if proposal_count else "unresolved" if results else "not_needed"),
        "frontier_count": len(frontier_smiles),
        "executed_frontier_count": len(results),
        "provider_invocation_count": sum(
            int(result.get("provider_invocation_count") or 0) for result in results
        ),
        "codex_delegated_frontier_count": sum(
            int(result.get("codex_delegated_frontier_count") or 0) for result in results
        ),
        "frontier_smiles": frontier_smiles,
        "proposal_count": proposal_count,
        "results": provider_results,
        "route_lineage": route_lineage,
        "action_execution_count": len(results),
        "material_events": (
            ["guided_provider_proposals_added"]
            if proposal_count
            else ["provider_search_exhausted_without_proposal"]
            if results
            else []
        ),
        "semantics": {
            "canonical_frontier_queue": True,
            "frontier_batch_is_bounded": True,
            "provider_result_requires_host_materialization": True,
            "scheduler_owned_execution": True,
        },
    }


def _record_campaign_acceptance(
    service: Any,
    *,
    gates: Mapping[str, Any],
    resource_envelope: Mapping[str, Any],
    idempotency_key: str,
) -> bool:
    milestones = _campaign_milestones(gates)
    gate_achieved = milestones["B5_configured_portfolio_acceptance"]
    accepted = bool(gate_achieved and resource_envelope.get("within_budget") is True)
    service.kernel.record_acceptance(
        {
            "schema_version": "unified_campaign_acceptance_report.v1",
            "graph_revision": service.kernel.state.graph_revision,
            "accepted": accepted,
            "configured_acceptance_achieved": gate_achieved,
            "milestones": milestones,
            "within_resource_budget": (resource_envelope.get("within_budget") is True),
            "semantics": {
                "one_acceptance_rule_for_all_targets": True,
                "milestones_do_not_select_solver_control_flow": True,
                "B4_does_not_grant_reaction_or_evidence_proof": True,
            },
        },
        idempotency_key=idempotency_key,
    )
    return accepted


def _persist_target_report(
    service: Any,
    *,
    report: Mapping[str, Any],
    identity: str,
    directory: Path,
    checkpoint_path: Path,
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(report)
    payload["content_sha256"] = _digest(payload)
    report_artifact = service.kernel.artifacts.put_json(
        payload,
        logical_name="target_only_solve_report.json",
        producer="autoplanner.target_solver",
    )
    service.kernel.index.index_artifact(
        run_id=identity,
        artifact_id="target_only_solve_report",
        ref=report_artifact,
        revision=service.kernel.state.graph_revision,
        authority_scope="benchmark_measurement_only",
    )
    report_path = directory / "target-only-solve-report.json"
    _write_json_atomic(report_path, payload)
    _checkpoint(
        checkpoint_path,
        identity,
        payload["stages"],
        outcomes,
        complete=True,
        resume_cursor=build_target_resume_cursor(service.graph_store.load()),
    )
    return {
        **payload,
        "report_ref": report_artifact.to_dict(),
        "report_path": str(report_path),
    }


def _automatic_continuation_baseline(service: Any) -> dict[str, Any]:
    """Return the durable state that proves a resumed pass made progress."""

    graph = service.graph_store.load()
    state = service.kernel.state
    return {
        "scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "attempt_count": int(state.attempt_count),
        "accepted_expansion_count": int(state.accepted_expansion_count),
        "model_totals": dict(state.model_totals),
    }


def _automatic_continuation_exhausted(
    *,
    resumed_completed_checkpoint: bool,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    portfolio_accepted: bool,
) -> bool:
    """End a no-op resume instead of leaving an infinite paused queue entry."""

    return bool(
        resumed_completed_checkpoint and not portfolio_accepted and dict(baseline) == dict(current)
    )


def _transition_unresolved_if_active(
    kernel: Any,
    *,
    idempotency_key: str,
    reasons: tuple[str, ...],
) -> bool:
    """Preserve an existing terminal decision while retaining diagnostics."""

    if kernel.state.terminal:
        return False
    kernel.transition(
        "unresolved",
        idempotency_key=idempotency_key,
        reasons=reasons,
    )
    return True


def _refresh_terminal_report(
    service: Any,
    *,
    identity: str,
    directory: Path,
    canonical: str,
    case: BlindCase,
    preflight: Mapping[str, Any],
    config: TargetSolveConfig,
    acceptance: RetrosynthesisAcceptanceSpec,
    budget: RetrosynthesisRunBudget,
    stages: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute reporting projections for a terminal run without new work."""

    graph = service.graph_store.load()
    portfolio = compile_proof_portfolio(
        graph,
        acceptance_spec=acceptance,
        config=_portfolio_config(config, acceptance),
        budget_exhausted=service.kernel.state.status == "budget_exhausted",
    )
    gates = compile_blind_acceptance_report(
        preflight=preflight,
        director_outcomes=outcomes,
        graph=graph,
        portfolio=portfolio,
    )
    resource_envelope = _resource_envelope(
        model_cost=service.kernel.state.model_totals,
        native_search=service.kernel.native_search_budget(),
        task_budget=service.kernel.task_budget(),
        run_wall_time_s=service.kernel.state.task_wall_time_s,
        attempt_count=service.kernel.state.attempt_count,
        accepted_expansion_count=service.kernel.state.accepted_expansion_count,
        budget=budget,
        max_run_wall_time_s=service.kernel.spec.limits.max_run_wall_time_s,
    )
    stop_decision = service.kernel.decide_stop().to_dict()
    profile_projection = service.workbench()["snapshot"]
    quality_state = compile_campaign_quality_state(
        workbench=profile_projection,
        gates=gates,
    )
    claim = _claim(
        gates,
        acceptance,
        resource_envelope,
        objective_mode=config.objective_mode,
        workbench=profile_projection,
    )
    planning_depth = _planning_depth_requirement(
        outcomes,
        minimum_steps=config.minimum_planning_route_steps,
    )
    current_disposition = _current_disposition(
        kernel_status=service.kernel.state.status,
        stop_decision=stop_decision,
        claim=claim,
        gates=gates,
    )
    report_path = directory / "target-only-solve-report.json"
    previous = (
        _read_json_object(report_path, "target_solve_report_missing")
        if report_path.is_file()
        else {}
    )
    workbench = service.publish_workbench(
        campaign_summary=_workbench_campaign_summary(
            gates=gates,
            resource_envelope=resource_envelope,
            model_cost=service.kernel.state.model_totals,
            stop_decision=stop_decision,
            claim=claim,
            current_disposition=current_disposition,
            planning_depth=planning_depth,
            quality_state=quality_state,
            trajectory=dict(previous.get("trajectory") or {}),
        )
    )
    report = {
        **previous,
        "schema_version": TARGET_SOLVE_REPORT_SCHEMA,
        "run_id": identity,
        "run_dir": str(directory),
        "target": {"name": case.target_name, "canonical_smiles": canonical},
        "campaign_spec": service.kernel.spec.campaign_spec.to_dict(),
        "preflight": dict(preflight),
        "config": asdict(config),
        "acceptance": acceptance.to_dict(),
        "budget": budget.to_dict(),
        "director_outcomes": outcomes,
        "stages": _deduplicate_stages(stages),
        "gates": gates,
        "quality_state": quality_state,
        "planning_depth": planning_depth,
        "model_cost": dict(service.kernel.state.model_totals),
        "resource_envelope": resource_envelope,
        "attempt_count": service.kernel.state.attempt_count,
        "accepted_expansion_count": service.kernel.state.accepted_expansion_count,
        "stop_decision": stop_decision,
        "current_disposition": current_disposition,
        "workbench_ref": workbench["snapshot_ref"],
        "claim": claim,
        "report_refresh": {
            "model_invocations": 0,
            "terminal_state_unchanged": True,
            "terminal_state_scientifically_stale": (
                current_disposition["state"] == "terminal_snapshot_requires_revalidation"
            ),
            "canonical_graph_sha256": str(graph.get("scientific_sha256") or ""),
            "portfolio_sha256": str(portfolio.get("content_sha256") or ""),
        },
    }
    report.pop("report_ref", None)
    report.pop("report_path", None)
    report.pop("content_sha256", None)
    report["content_sha256"] = _digest(report)
    report_artifact = service.kernel.artifacts.put_json(
        report,
        logical_name="target_only_solve_report.json",
        producer="autoplanner.target_solver.refresh",
    )
    _write_json_atomic(report_path, report)
    return {
        **report,
        "report_ref": report_artifact.to_dict(),
        "report_path": str(report_path),
    }


def _budget_input(value: RetrosynthesisRunBudget) -> dict[str, Any]:
    return {
        key: item
        for key, item in asdict(value).items()
        if key
        in {
            "max_model_invocations",
            "max_total_input_tokens",
            "max_total_output_tokens",
            "max_total_wall_time_s",
            "max_accepted_expansions",
            "max_attempt_runs",
            "max_native_search_invocations",
            "min_target_native_search_invocations",
            "max_frontier_native_search_invocations",
            "allow_frontier_native_search_borrowing",
            "max_prompt_context_bytes",
        }
    }


def _bind_native_search_budget(
    value: RetrosynthesisRunBudget,
    *,
    config: TargetSolveConfig,
) -> RetrosynthesisRunBudget:
    """Bind broad attempt budgets to the configured target/guided call caps."""

    requested_total = max(0, int(value.max_native_search_invocations or 0))
    target_limit = int(config.enable_chemenzy and config.enable_target_chemenzy_baseline)
    target_limit = min(target_limit, requested_total)
    configured_frontier_limit = (
        (
            max(0, requested_total - target_limit)
            if config.max_guided_chemenzy_frontiers is None
            else config.max_guided_chemenzy_frontiers
        )
        if config.enable_chemenzy and config.enable_guided_chemenzy
        else 0
    )
    frontier_budget_cap = max(
        0, int(value.max_frontier_native_search_invocations or 0)
    )
    if value.allow_frontier_native_search_borrowing:
        frontier_budget_cap += max(
            0,
            int(value.min_target_native_search_invocations or 0) - target_limit,
        )
    frontier_limit = min(
        frontier_budget_cap,
        max(0, int(configured_frontier_limit)),
        max(0, requested_total - target_limit),
    )
    hard_total = target_limit + frontier_limit
    return replace(
        value,
        max_native_search_invocations=hard_total,
        min_target_native_search_invocations=target_limit,
        max_frontier_native_search_invocations=frontier_limit,
    )


def _frozen_stock_membership_checker(
    builder: StockCatalogBuilder | None,
) -> Callable[[Iterable[str]], Mapping[str, bool]] | None:
    """Expose only a frozen local stock index to compact node search.

    A live catalog lookup per Codex node would add network latency and could
    change during one experiment.  The sequential policy may terminate a leaf
    early only when the configured builder is immutable and content-addressed;
    all other leaves continue to the normal host stock stage.
    """

    if builder is None or not (
        str(getattr(builder, "index_sha256", ""))
        and int(getattr(builder, "member_count", 0) or 0) > 0
    ):
        return None

    def lookup(values: Iterable[str]) -> Mapping[str, bool]:
        canonical_values = tuple(
            sorted(
                {
                    canonical
                    for value in values
                    if (canonical := canonical_smiles(value))
                }
            )
        )
        if not canonical_values:
            return {}
        observation = builder(
            canonical_values,
            max_molecules=len(canonical_values),
        )
        members = {
            canonical_smiles(row.get("canonical_smiles"))
            for row in dict(observation or {}).get("members") or []
            if isinstance(row, Mapping)
        }
        return {value: value in members for value in canonical_values}

    return lookup


def _audit_stock_stage(
    service: Any,
    *,
    acceptance: RetrosynthesisAcceptanceSpec,
    config: TargetSolveConfig,
    catalog_builder: StockCatalogBuilder | None,
    inventory_builder: InventorySnapshotBuilder | None,
) -> dict[str, Any]:
    if config.enable_live_benchmark_stock and acceptance.stock_boundary == ("benchmark_search"):
        return audit_live_benchmark_stock(
            service,
            catalog_builder=catalog_builder,
            max_molecules=config.max_live_stock_molecules,
        )
    if acceptance.stock_boundary == "procurement" and inventory_builder is not None:
        return audit_authoritative_inventory_stock(
            service,
            inventory_builder=inventory_builder,
            required_boundary="procurement",
            max_molecules=config.max_live_stock_molecules,
        )
    return {
        "stage": "stock",
        "status": "unresolved",
        "reason": "authoritative_stock_adapter_not_configured",
    }


def _target_identity_stage(
    service: Any,
    *,
    stages: Iterable[Mapping[str, Any]],
    target_name: str,
    target_smiles: str,
    enabled: bool,
    resolve_named: bool = False,
    lookup_now: bool = False,
) -> dict[str, Any]:
    opaque_name = "target-" + hashlib.sha256(str(target_smiles).encode("utf-8")).hexdigest()[:8]
    prior = next(
        (
            dict(row.get("detail") or {})
            for row in reversed(list(stages))
            if row.get("stage") == "target_identity"
        ),
        {},
    )
    stale_non_structural_skip = prior.get("status") == "not_needed"
    stale_provider_version = (
        prior.get("status") in {"completed", "unresolved"}
        and (
            prior.get("provider_id") == "pubchem.pug_rest"
            or str(prior.get("reason") or "").startswith("pubchem_")
        )
        and prior.get("provider_version") != TARGET_IDENTITY_PROVIDER_VERSION
    )
    stale_display_name = bool(
        prior.get("status") == "not_needed"
        and str(dict(prior.get("identity") or {}).get("preferred_name") or "") != opaque_name
    )
    if (
        prior
        and not stale_non_structural_skip
        and not stale_provider_version
        and not stale_display_name
        and (lookup_now is False or prior.get("status") != "pending")
    ):
        return prior
    if not enabled:
        return {
            "stage": "target_identity",
            "status": "disabled",
            "reason": "target_identity_disabled",
        }
    if not lookup_now:
        return {
            "stage": "target_identity",
            "status": "pending",
            "reason": "target_identity_deferred_to_evidence_action",
            "identity": {"preferred_name": opaque_name},
            "semantics": {
                "initial_route_search_does_not_wait_for_identity_network": True,
                "evidence_action_owns_structure_identity_resolution": True,
                "user_supplied_name_not_used_for_lookup": True,
                "display_name_not_exposed_to_core_planning": True,
            },
        }
    result = resolve_target_identity(target_smiles)
    result = {
        **result,
        "semantics": {
            **dict(result.get("semantics") or {}),
            "user_supplied_name_not_used_for_lookup": True,
            "display_name_not_exposed_to_core_planning": True,
            "legacy_resolve_named_flag_ignored": bool(resolve_named),
            "opaque_fallback_name": opaque_name,
        },
    }
    result["content_sha256"] = _digest(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    artifact = service.kernel.artifacts.put_json(
        result,
        logical_name="target_identity_observation.json",
        producer="autoplanner.target_identity",
    )
    return {
        **result,
        "stage": "target_identity",
        "artifact_ref": artifact.to_dict(),
    }


def _target_name_requires_identity_resolution(value: str) -> bool:
    normalized = " ".join(str(value or "").lower().split())
    return (
        normalized
        in {
            "blind target",
            "target",
            "unknown target",
            "smiles-only target",
        }
        or re.fullmatch(r"target-[0-9a-f]{8}", normalized) is not None
    )


def _latest_stage(
    stages: Iterable[Mapping[str, Any]],
    stage_name: str,
) -> dict[str, Any]:
    return next(
        (dict(row) for row in reversed(list(stages)) if row.get("stage") == stage_name),
        {},
    )


def _latest_visual_observation(
    stages: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the newest frozen visual observation from either evidence pass."""

    for stage in reversed(list(stages)):
        if stage.get("stage") not in {
            "evidence_acquisition",
            "replan_evidence_acquisition",
        }:
            continue
        detail = dict(stage.get("detail") or {})
        observation = dict(dict(detail.get("visual_evidence") or {}).get("observation") or {})
        if observation.get("candidate_steps"):
            return observation
    return {}


def _attempted_chemenzy_frontiers(
    stages: Iterable[Mapping[str, Any]],
) -> set[str]:
    values: set[str] = set()
    for row in stages:
        detail = dict(row.get("detail") or {})
        candidates: list[Any] = []
        if row.get("stage") in {
            "chemenzy_guided_frontier",
            "chemenzy_stock_recovery",
        }:
            candidates.extend(detail.get("frontier_smiles") or [])
        action = dict(detail.get("action") or {})
        if action.get("kind") == CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value:
            candidates.append(dict(action.get("metadata") or {}).get("frontier_smiles"))
            handler_result = dict(
                dict(detail.get("outcome") or {}).get("handler_result") or {}
            )
            candidates.extend(handler_result.get("frontier_smiles") or [])
        for value in candidates:
            if not str(value):
                continue
            _molecule_id, canonical = molecule_identity(str(value))
            values.add(canonical or str(value))
    return values


def _attempted_chemenzy_frontiers_from_action_history(
    executions: Iterable[Mapping[str, Any]],
) -> set[str]:
    values: set[str] = set()
    for raw in executions:
        row = dict(raw)
        if (
            row.get("settled") is not True
            or row.get("action_kind")
            != CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value
        ):
            continue
        result = dict(row.get("handler_result") or {})
        candidates = [
            *(result.get("frontier_smiles") or []),
            dict(result.get("guided_progress_checkpoint") or {}).get(
                "frontier_smiles"
            ),
        ]
        for value in candidates:
            if not str(value):
                continue
            _molecule_id, canonical = molecule_identity(str(value))
            values.add(canonical or str(value))
    return values


def _pending_guided_progress_from_action_history(
    executions: Iterable[Mapping[str, Any]],
    *,
    stages: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    completed_frontiers = {
        str(dict(row.get("detail") or {}).get("frontier_smiles") or "")
        for row in stages
        if str(row.get("stage") or "").startswith(
            "guided_root_stock_progress_"
        )
    }
    for raw in reversed(list(executions)):
        row = dict(raw)
        if (
            row.get("settled") is not True
            or row.get("action_kind")
            != CampaignActionKind.CHEMENZY_FRONTIER_EXPAND.value
        ):
            continue
        progress = dict(
            dict(row.get("handler_result") or {}).get(
                "guided_progress_checkpoint"
            )
            or {}
        )
        frontier = str(progress.get("frontier_smiles") or "")
        if progress and frontier not in completed_frontiers:
            return progress
    return {}


def _chemenzy_delegation_audit(
    outcomes: Iterable[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain whether Codex's local-provider intent reached the one frontier."""

    molecules = dict(graph.get("molecules") or {})
    expansion_ids = {
        str(row.get("object_id") or "")
        for row in dict(graph.get("deficit_frontier") or {}).get("items") or []
        if isinstance(row, Mapping) and row.get("kind") == "expansion"
    }
    requests: list[dict[str, Any]] = []
    for outcome in outcomes:
        plan = dict(outcome.get("plan") or {})
        for raw in plan.get("frontier_priorities") or []:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            identifiers = " ".join(
                str(row.get(key) or "").lower()
                for key in ("priority_id", "proposal_id", "rationale")
            )
            providers = {
                str(value).strip().lower()
                for value in row.get("provider_preferences") or []
                if str(value).strip()
            }
            if "chemenzy" not in providers and "chemenzy" not in identifiers:
                continue
            molecule_id, canonical = molecule_identity(row.get("target_smiles"))
            molecule = dict(molecules.get(molecule_id) or {})
            if not canonical:
                disposition = "provider_target_missing"
            elif not molecule:
                disposition = "selected_step_not_host_admitted"
            elif molecule_id in expansion_ids:
                disposition = "queued_on_canonical_frontier"
            elif molecule.get("stock_closed") is True:
                disposition = "already_stock_closed"
            elif molecule.get("provider_expansion_requested") is not True:
                disposition = "provider_annotation_not_admitted"
            else:
                disposition = "request_not_actionable"
            requests.append(
                {
                    "priority_id": str(row.get("priority_id") or ""),
                    "proposal_id": str(row.get("proposal_id") or ""),
                    "target_smiles": canonical,
                    "disposition": disposition,
                }
            )
    queued = sum(row["disposition"] == "queued_on_canonical_frontier" for row in requests)
    rejected = sum(
        row["disposition"]
        in {
            "provider_target_missing",
            "selected_step_not_host_admitted",
            "provider_annotation_not_admitted",
        }
        for row in requests
    )
    status = (
        "queued"
        if queued and not rejected
        else "partial"
        if queued
        else "rejected"
        if requests
        else "not_requested"
    )
    return {
        "schema_version": "chemenzy_delegation_audit.v1",
        "stage": "chemenzy_delegation",
        "status": status,
        "request_count": len(requests),
        "queued_count": queued,
        "rejected_count": rejected,
        "requests": requests,
        "semantics": {
            "codex_selection_is_not_host_admission": True,
            "rejected_skeleton_never_reaches_provider": True,
            "provider_enablement_does_not_claim_invocation": True,
        },
    }


PREPARED_EVIDENCE_ACQUISITION_SCHEMA = "prepared_evidence_acquisition.v1"


def _prepare_evidence_acquisition(
    service: Any,
    *,
    source_stage: Mapping[str, Any],
    connector: EvidenceConnector | None,
    target_name: str = "",
    target_identity: Mapping[str, Any] | None = None,
    allow_target_identity_lookup: bool = True,
) -> dict[str, Any]:
    """Run connector acquisition against one frozen graph without ingesting it."""

    graph = service.graph_store.load()
    prepared: dict[str, Any] = {
        "schema_version": PREPARED_EVIDENCE_ACQUISITION_SCHEMA,
        "status": "unresolved",
        "input_graph_revision": int(graph.get("revision") or 0),
        "input_evidence_revision": int(service.kernel.state.evidence_revision),
        "input_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "request": {},
        "acquired": {},
        "effective_target_identity": dict(target_identity or {}),
        "target_identity_resolution": {},
        "reason": "structured_evidence_connector_not_configured",
        "semantics": {
            "connector_output_grants_no_canonical_authority": True,
            "canonical_ingestion_is_deferred_to_stable_action_order": True,
        },
    }
    if connector is not None:
        effective_target_identity, target_identity_resolution = _ensure_evidence_target_identity(
            service,
            target_name=target_name or service.kernel.spec.target_name,
            target_smiles=service.kernel.spec.target_smiles,
            target_identity=target_identity,
            allow_remote_lookup=allow_target_identity_lookup,
        )
        request = compile_evidence_acquisition_request(
            run_id=service.kernel.spec.run_id,
            target_name=target_name or service.kernel.spec.target_name,
            target_smiles=service.kernel.spec.target_smiles,
            graph=graph,
            source_frontier=source_stage,
            target_identity=effective_target_identity,
        )
        prepared.update(
            {
                "request": request,
                "effective_target_identity": effective_target_identity,
                "target_identity_resolution": target_identity_resolution,
                "reason": "",
            }
        )
        try:
            prepared["acquired"] = acquire_structured_evidence(
                request,
                connector=connector,
            )
            prepared["status"] = "prepared"
        except (LiveEvidenceConnectorError, ValueError) as exc:
            prepared["reason"] = f"evidence_connector_failed:{type(exc).__name__}:{exc}"
    prepared["content_sha256"] = _digest(prepared)
    return prepared


def _validated_prepared_evidence_acquisition(
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(prepared)
    supplied_sha256 = str(row.pop("content_sha256", ""))
    if (
        row.get("schema_version") != PREPARED_EVIDENCE_ACQUISITION_SCHEMA
        or not supplied_sha256
        or supplied_sha256 != _digest(row)
    ):
        raise ValueError("prepared_evidence_acquisition_invalid")
    return {**row, "content_sha256": supplied_sha256}


def _acquire_evidence_stage(
    service: Any,
    *,
    source_stage: Mapping[str, Any],
    connector: EvidenceConnector | None,
    atom_mapper: ReactionMapper | None,
    visual_provider: VisualEvidenceProvider | None = None,
    max_visual_pages: int = 6,
    target_name: str = "",
    target_identity: Mapping[str, Any] | None = None,
    allow_target_identity_lookup: bool = True,
    prior_visual_observation: Mapping[str, Any] | None = None,
    validation_runner: Callable[[str], Mapping[str, Any]] | None = None,
    defer_validation: bool = False,
    prepared_acquisition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = _validated_prepared_evidence_acquisition(
        prepared_acquisition
        or _prepare_evidence_acquisition(
            service,
            source_stage=source_stage,
            connector=connector,
            target_name=target_name,
            target_identity=target_identity,
            allow_target_identity_lookup=allow_target_identity_lookup,
        )
    )
    request = dict(prepared.get("request") or {})
    target_identity_resolution = dict(prepared.get("target_identity_resolution") or {})
    prepared_binding = {
        "prepared_input_graph_revision": int(prepared.get("input_graph_revision") or 0),
        "prepared_acquisition_sha256": str(prepared.get("content_sha256") or ""),
    }
    if prepared.get("status") != "prepared":
        return {
            "stage": "evidence_acquisition",
            "status": "unresolved",
            "reason": str(prepared.get("reason") or "structured_evidence_connector_not_configured"),
            "request_sha256": str(request.get("content_sha256") or ""),
            "target_identity_resolution": target_identity_resolution,
            "model_invocations": 0,
            "visual_invocations": 0,
            "false_evidence_claim": False,
            **prepared_binding,
        }
    acquired = dict(prepared.get("acquired") or {})
    visual_stage: dict[str, Any] = {}
    source_route_stage: dict[str, Any] = {}
    try:
        receipt = dict(acquired.get("receipt") or {})
        receipt_ref: dict[str, Any] = {}
        if receipt:
            receipt_ref = service.kernel.artifacts.put_json(
                receipt,
                logical_name="evidence_connector_receipt.json",
                producer="autoplanner.live_evidence",
            ).to_dict()
        discovery = dict(acquired.get("discovery") or {})
        discovery_ref: dict[str, Any] = {}
        if discovery:
            discovery_ref = service.kernel.artifacts.put_json(
                discovery,
                logical_name="source_discovery_observation.json",
                producer="autoplanner.live_evidence.discovery",
            ).to_dict()
        discovery_ingestion = ingest_source_discovery_observation(
            service,
            discovery,
        )
        source_route_stage = materialize_discovered_source_routes(
            service,
            discovery,
        )
        source_edge_ids = {
            str(value)
            for value in source_route_stage.get("materialized_edge_ids") or []
            if str(value)
        }
        source_route_validation: dict[str, Any] = {}
        if dict(source_route_stage.get("execution") or {}).get("changed") is True:
            if defer_validation:
                source_route_validation = {
                    "status": "deferred_to_campaign_action_frontier",
                    "deferred_edge_ids": sorted(source_edge_ids),
                }
            elif validation_runner is not None:
                source_route_validation = dict(validation_runner("evidence_source_route_validate"))
            else:
                source_route_validation = validate_materialized_edges(
                    service,
                    atom_mapper=atom_mapper,
                    edge_ids=source_edge_ids,
                )
        source_route_stage = {
            **source_route_stage,
            "validation": _scope_validation_summary(
                source_route_validation,
                source_edge_ids,
            ),
        }
        visual_stage = acquire_visual_evidence_candidates(
            service,
            evidence_request=request,
            discovery=discovery,
            provider=visual_provider,
            max_pages=max_visual_pages,
        )
        if visual_stage.get("status") == "budget_blocked" and prior_visual_observation:
            rebound = rebind_visual_evidence_observation(
                service,
                request=dict(visual_stage.get("request") or {}),
                prior_observation=prior_visual_observation,
            )
            if rebound.get("status") == "reused":
                visual_stage = rebound
        visual_materialization = materialize_visual_evidence_candidates(
            service,
            observation=dict(visual_stage.get("observation") or {}),
        )
        visual_validation: dict[str, Any] = {}
        if dict(visual_materialization.get("execution") or {}).get("changed") is True:
            if defer_validation:
                visual_validation = {
                    "status": "deferred_to_campaign_action_frontier",
                }
            elif validation_runner is not None:
                visual_validation = dict(validation_runner("evidence_visual_validate"))
            else:
                visual_validation = validate_materialized_edges(
                    service,
                    atom_mapper=atom_mapper,
                )
        visual_stage = {
            **visual_stage,
            "materialization": visual_materialization,
            "validation": visual_validation,
            "material_events": sorted(
                {
                    *[
                        str(value)
                        for value in visual_stage.get("material_events") or []
                        if str(value)
                    ],
                    *[
                        str(value)
                        for value in visual_materialization.get("material_events") or []
                        if str(value)
                    ],
                    *(
                        ["visual_literature_chain_host_validated"]
                        if int(visual_validation.get("accepted_validation_count") or 0)
                        else []
                    ),
                }
            ),
        }
        document = acquired.get("document")
        if document is None:
            visual_structure_binding_count = int(
                dict(visual_stage.get("materialization") or {}).get(
                    "exact_structure_binding_candidate_count"
                )
                or 0
            )
            source_route_structure_binding_count = len(
                source_route_stage.get("materialized_edge_ids") or []
            )
            structure_binding_count = (
                visual_structure_binding_count + source_route_structure_binding_count
            )
            return {
                "stage": "evidence_acquisition",
                "status": (
                    "structure_bound_unproven" if structure_binding_count else "discovered_unbound"
                ),
                "request_sha256": request["content_sha256"],
                "receipt_ref": receipt_ref,
                "discovery_ref": discovery_ref,
                "discovery": discovery,
                "discovery_ingestion": discovery_ingestion,
                "source_route": source_route_stage,
                "visual_evidence": visual_stage,
                "source_count": len(discovery.get("sources") or []),
                "target_identity_resolution": target_identity_resolution,
                "exact_record_count": 0,
                "exact_structure_binding_count": structure_binding_count,
                "visual_structure_binding_count": visual_structure_binding_count,
                "source_route_structure_binding_count": (source_route_structure_binding_count),
                "model_invocations": int(visual_stage.get("model_invocations") or 0),
                "visual_invocations": int(visual_stage.get("visual_invocations") or 0),
                "false_evidence_claim": False,
                **prepared_binding,
                "material_events": sorted(
                    {
                        "source_material_discovered",
                        *[
                            str(value)
                            for value in visual_stage.get("material_events") or []
                            if str(value)
                        ],
                        *[
                            str(value)
                            for value in source_route_stage.get("material_events") or []
                            if str(value)
                        ],
                    }
                ),
                "semantics": {
                    "discovery_is_not_exact_evidence": True,
                    "exact_structure_binding_is_not_reaction_proof": True,
                    "discovery_may_inform_bounded_global_replan": True,
                    "connector_cannot_grant_reaction_validation": True,
                },
            }
        imported = ingest_structured_evidence_document(
            service,
            document=dict(document),
            atom_mapper=atom_mapper,
            validation_runner=(
                (
                    lambda edge_ids: _scope_validation_summary(
                        dict(validation_runner("structured_evidence_revalidate")),
                        edge_ids,
                    )
                )
                if validation_runner is not None
                else None
            ),
            defer_validation=defer_validation,
        )
        # Structured import validates every pending edge in one canonical
        # batch.  Scope its receipt back to the source-route edge IDs instead
        # of attributing unrelated Codex/ChemEnzy validations to literature.
        shared_validation = dict(imported.get("validation") or {})
        source_edge_ids = {
            str(value)
            for value in source_route_stage.get("materialized_edge_ids") or []
            if str(value)
        }
        accepted_source_ids = sorted(
            source_edge_ids.intersection(
                str(value) for value in shared_validation.get("accepted_edge_ids") or []
            )
        )
        rejected_source_ids = sorted(
            source_edge_ids.intersection(
                str(value) for value in shared_validation.get("rejected_edge_ids") or []
            )
        )
        source_route_stage = {
            **source_route_stage,
            "validation": {
                "status": str(shared_validation.get("status") or ""),
                "accepted_validation_count": len(accepted_source_ids),
                "rejected_validation_count": len(rejected_source_ids),
                "accepted_edge_ids": accepted_source_ids,
                "rejected_edge_ids": rejected_source_ids,
                "rejection_diagnostics": [
                    dict(value)
                    for value in shared_validation.get("rejection_diagnostics") or []
                    if isinstance(value, Mapping)
                    and str(value.get("edge_id") or "") in source_edge_ids
                ],
            },
            "validation_shared_with_structured_import": True,
        }
    except (LiveEvidenceConnectorError, ValueError) as exc:
        return {
            "stage": "evidence_acquisition",
            "status": "unresolved",
            "reason": f"evidence_connector_failed:{type(exc).__name__}:{exc}",
            "request_sha256": request["content_sha256"],
            "target_identity_resolution": target_identity_resolution,
            "model_invocations": int(visual_stage.get("model_invocations") or 0),
            "visual_invocations": int(visual_stage.get("visual_invocations") or 0),
            "false_evidence_claim": False,
            **prepared_binding,
        }
    return {
        "stage": "evidence_acquisition",
        "status": ("completed" if int(imported.get("exact_record_count") or 0) else "partial"),
        "request_sha256": request["content_sha256"],
        "document_sha256": acquired["document_sha256"],
        "receipt_ref": receipt_ref,
        "discovery_ref": discovery_ref,
        "discovery": discovery,
        "discovery_ingestion": discovery_ingestion,
        "source_route": source_route_stage,
        "visual_evidence": visual_stage,
        "source_count": imported["source_count"],
        "target_identity_resolution": target_identity_resolution,
        "exact_record_count": imported["exact_record_count"],
        "source_binding_count": imported["source_binding_count"],
        "execution": imported["execution"],
        "validation": imported["validation"],
        "model_invocations": int(visual_stage.get("model_invocations") or 0),
        "visual_invocations": int(visual_stage.get("visual_invocations") or 0),
        **prepared_binding,
        "material_events": sorted(
            {
                *(["source_material_discovered"] if discovery else []),
                *(str(value) for value in visual_stage.get("material_events") or [] if str(value)),
                *(
                    str(value)
                    for value in source_route_stage.get("material_events") or []
                    if str(value)
                ),
                *(["exact_rows_added"] if int(imported.get("exact_record_count") or 0) else []),
            }
        ),
        "semantics": {
            "connector_output_requires_normal_host_ingestion": True,
            "connector_cannot_grant_reaction_validation": True,
            "receipt_grants_no_scientific_authority": True,
            "connector_acquisition_preceded_stable_order_canonical_commit": True,
        },
    }


def _ensure_evidence_target_identity(
    service: Any,
    *,
    target_name: str,
    target_smiles: str,
    target_identity: Mapping[str, Any] | None,
    allow_remote_lookup: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Guarantee structure-derived aliases before ranking downloaded papers.

    Normal target solves already resolve identity before evidence prefetch, but
    validation forks and historical continuations used to bypass that stage.
    An empty identity made native PDF alias hits invisible and could send the
    single visual pass to a merely related paper.  Evidence acquisition is the
    last safe common boundary, so it now repairs that omission itself.
    """

    supplied = dict(target_identity or {})
    structurally_resolved = bool(
        supplied.get("inchikey")
        or supplied.get("cid")
        or supplied.get("resolved_from_input_structure") is True
        or supplied.get("synonyms")
    )
    if structurally_resolved:
        return supplied, {
            "status": "reused",
            "reason": "structure_resolved_target_identity_supplied",
            "identity": supplied,
        }

    if not allow_remote_lookup:
        return supplied, {
            "status": "disabled",
            "reason": "target_identity_lookup_disabled",
            "identity": supplied,
            "semantics": {
                "evidence_acquisition_continues_without_remote_aliases": True,
                "configuration_disables_identity_network": True,
            },
        }

    result = resolve_target_identity(target_smiles, target_name=target_name)
    artifact = service.kernel.artifacts.put_json(
        result,
        logical_name="evidence_target_identity_observation.json",
        producer="autoplanner.target_identity.evidence_guard",
    )
    resolved = dict(result.get("identity") or {})
    if result.get("status") == "completed" and resolved:
        return resolved, {
            **result,
            "status": "completed",
            "reason": "evidence_boundary_recovered_structure_identity",
            "artifact_ref": artifact.to_dict(),
        }
    return supplied, {
        **result,
        "artifact_ref": artifact.to_dict(),
        "semantics": {
            **dict(result.get("semantics") or {}),
            "identity_failure_does_not_block_literature_discovery": True,
            "identity_failure_must_not_be_cached_as_success": True,
        },
    }


def _run_evidence_prefetch(
    request: Mapping[str, Any],
    connector: EvidenceConnector,
) -> dict[str, Any]:
    """Discover and warm source caches while the one global model call runs."""

    started = time.monotonic()
    try:
        acquired = acquire_structured_evidence(request, connector=connector)
    except (LiveEvidenceConnectorError, OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "unresolved",
            "elapsed_s": round(time.monotonic() - started, 3),
            "request_sha256": str(request.get("content_sha256") or ""),
            "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
            "discovery": {},
            "semantics": {
                "runs_concurrently_with_global_director": True,
                "failure_does_not_abort_campaign": True,
            },
        }
    discovery = dict(acquired.get("discovery") or {})
    return {
        "status": "completed" if discovery else "no_discovery",
        "elapsed_s": round(time.monotonic() - started, 3),
        "request_sha256": str(request.get("content_sha256") or ""),
        "source_count": len(discovery.get("sources") or []),
        "discovery": discovery,
        "receipt": dict(acquired.get("receipt") or {}),
        "semantics": {
            "runs_concurrently_with_global_director": True,
            "prefetch_grants_no_evidence_authority": True,
            "full_edge_bound_extraction_still_required": True,
        },
    }


def _merge_prefetched_source_hints(
    source_stage: Mapping[str, Any],
    prefetch: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(source_stage)
    discovery = dict(prefetch.get("discovery") or {})
    merged: dict[str, dict[str, Any]] = {}
    for source in [
        *(row.get("sources") or []),
        *(discovery.get("sources") or []),
    ]:
        if not isinstance(source, Mapping):
            continue
        source_row = dict(source)
        source_ref = str(source_row.get("source_ref") or "")
        if not source_ref:
            continue
        merged.setdefault(source_ref, source_row)
    row["sources"] = [merged[key] for key in sorted(merged)]
    row["traceable_source_hint_count"] = len(row["sources"])
    row["status"] = "completed" if row["sources"] else str(row.get("status") or "unresolved")
    row["prefetch"] = {
        key: value for key, value in prefetch.items() if key not in {"discovery", "receipt"}
    }
    return row


def _replan_budget_guard(
    *,
    model_cost: Mapping[str, Any],
    budget: RetrosynthesisRunBudget,
    config: TargetSolveConfig,
    prompt_context_bytes: int = 0,
) -> dict[str, Any]:
    """Reserve a realistic second-call envelope before starting Codex again."""

    calls = max(1, int(model_cost.get("model_invocations") or 0))
    observed = {
        "model_invocations": int(model_cost.get("model_invocations") or 0),
        "input_tokens": int(model_cost.get("input_tokens") or 0),
        "output_tokens": int(model_cost.get("output_tokens") or 0),
        "model_wall_time_s": float(model_cost.get("wall_time_s") or 0.0),
    }
    required = {
        "model_invocations": 1,
        "input_tokens": max(
            4_000,
            math.ceil(16_000 + (max(0, prompt_context_bytes) * 0.45)),
        ),
        "output_tokens": max(
            2_000,
            min(
                config.max_director_output_tokens,
                math.ceil(observed["output_tokens"] / calls),
            ),
        ),
        "model_wall_time_s": max(
            30.0,
            min(
                config.max_director_wall_time_s,
                observed["model_wall_time_s"] / calls,
            ),
        ),
    }
    limits = {
        "model_invocations": budget.max_model_invocations,
        "input_tokens": budget.max_total_input_tokens,
        "output_tokens": budget.max_total_output_tokens,
        "model_wall_time_s": budget.max_total_wall_time_s,
    }
    remaining = {key: max(0.0, float(limits[key]) - float(observed[key])) for key in limits}
    reasons = sorted(
        f"insufficient_{key}_for_bounded_replan"
        for key, value in required.items()
        if remaining[key] < float(value)
    )
    return {
        "schema_version": "target_solve_replan_budget_gate.v1",
        "accepted": not reasons,
        "observed": observed,
        "required_reserve": required,
        "remaining": remaining,
        "prompt_context_bytes": max(0, int(prompt_context_bytes)),
        "reasons": reasons,
        "semantics": {
            "forecast_uses_observed_first_call": True,
            "budget_extension_forbidden": True,
            "skipped_replan_is_not_completion": True,
        },
    }


def _material_replan_events(*stages: Mapping[str, Any]) -> tuple[str, ...]:
    events: set[str] = set()
    for stage in stages:
        events.update(
            str(value) for value in stage.get("material_events") or [] if str(value).strip()
        )
        execution = dict(stage.get("execution") or {})
        events.update(
            str(value) for value in execution.get("material_events") or [] if str(value).strip()
        )
        if int(stage.get("rejected_validation_count") or 0) > 0:
            events.add("critical_edge_rejected")
        if int(stage.get("accepted_validation_count") or 0) > 0:
            events.add("host_validated_edges_added_after_initial_plan")
        if int(stage.get("source_route_host_accepted_count") or 0) > 0:
            events.add("host_validated_source_route_added")
    return tuple(sorted(events))


def _replan_signal_gate(
    gates: Mapping[str, Any],
    *,
    material_events: Iterable[str],
    trigger_reasons: Iterable[str],
    convergence_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a new host observation before spending the optional model call.

    A portfolio deficit is an outcome, not new information.  The director may
    be called again only when validation, evidence, inventory, provider output,
    or a host contract audit gives it a fact that was absent from the initial
    architecture prompt.
    """

    observed = {str(value) for value in material_events if str(value).strip()}
    pressure = compile_replan_pressure(
        gates,
        material_events=observed,
        convergence_ledger=convergence_ledger,
    )
    actionable = observed & ACTIONABLE_REPLAN_EVENTS
    actionable.update(pressure["derived_material_events"])
    gate_values = dict(gates.get("gates") or {})
    if gate_values.get("B4_stock_boundary") is True:
        actionable.difference_update({"stock_boundary_changed", "stock_records_added"})
    reasons = [] if actionable else ["no_new_actionable_host_observation"]
    return {
        "schema_version": "target_solve_replan_signal_gate.v1",
        "accepted": not reasons,
        "trigger_reasons": sorted({str(value) for value in trigger_reasons if str(value).strip()}),
        "observed_material_events": sorted(observed),
        "actionable_material_events": sorted(actionable),
        "ignored_material_events": sorted(observed - actionable),
        "durable_state_events": list(pressure["derived_material_events"]),
        "replan_pressure": pressure,
        "reasons": reasons,
        "semantics": {
            "portfolio_deficit_alone_never_spends_a_model_call": True,
            "new_host_observation_or_verified_durable_state_required": True,
            "text_stagnation_alone_is_not_actionable": True,
            "already_closed_stock_boundary_is_not_a_replan_signal": True,
            "skipped_replan_is_not_completion": True,
        },
    }


def _replan_retention_audit(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a replan extends the canonical graph instead of replacing it."""

    collections = ("molecules", "edges", "route_families")
    counts: dict[str, dict[str, int]] = {}
    missing: dict[str, list[str]] = {}
    for collection in collections:
        before_ids = set(dict(before.get(collection) or {}))
        after_ids = set(dict(after.get(collection) or {}))
        counts[collection] = {
            "before": len(before_ids),
            "after": len(after_ids),
            "added": len(after_ids - before_ids),
            "missing": len(before_ids - after_ids),
        }
        if before_ids - after_ids:
            missing[collection] = sorted(before_ids - after_ids)
    return {
        "schema_version": "target_solve_replan_retention_audit.v1",
        "accepted": not missing,
        "counts": counts,
        "missing_ids": missing,
        "semantics": {
            "canonical_graph_before_replan_is_retained": not missing,
            "replan_is_union_not_replacement": True,
            "proof_state_may_be_strengthened_in_place": True,
        },
    }


def _replan_gain_audit(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    model_cost_before: Mapping[str, Any],
    model_cost_after: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure one replan's observed scientific delta without causal overclaim."""

    before_gates = dict(before.get("gates") or {})
    after_gates = dict(after.get("gates") or {})
    gate_names = sorted(set(before_gates) | set(after_gates))
    gained_gates = [
        key
        for key in gate_names
        if before_gates.get(key) is not True and after_gates.get(key) is True
    ]
    regressed_gates = [
        key
        for key in gate_names
        if before_gates.get(key) is True and after_gates.get(key) is not True
    ]

    before_counts = dict(before.get("counts") or {})
    after_counts = dict(after.get("counts") or {})
    count_deltas = {
        key: int(after_counts.get(key) or 0) - int(before_counts.get(key) or 0)
        for key in sorted(set(before_counts) | set(after_counts))
    }
    positive_counts = {key: value for key, value in count_deltas.items() if value > 0}
    negative_counts = {key: value for key, value in count_deltas.items() if value < 0}

    cost_keys = (
        "model_invocations",
        "input_tokens",
        "output_tokens",
        "wall_time_s",
    )
    model_cost_delta = {
        key: round(
            float(model_cost_after.get(key) or 0.0) - float(model_cost_before.get(key) or 0.0),
            6,
        )
        for key in cost_keys
    }
    observed_gain = bool(gained_gates or positive_counts)
    observed_regression = bool(regressed_gates or negative_counts)
    disposition = (
        "regressed" if observed_regression else "positive_gain" if observed_gain else "no_gain"
    )
    return {
        "schema_version": "target_solve_replan_gain_audit.v1",
        "disposition": disposition,
        "observed_gain": observed_gain,
        "observed_regression": observed_regression,
        "gained_gates": gained_gates,
        "regressed_gates": regressed_gates,
        "count_deltas": count_deltas,
        "positive_count_deltas": positive_counts,
        "negative_count_deltas": negative_counts,
        "model_cost_delta": model_cost_delta,
        "semantics": {
            "within_run_before_after_measurement": True,
            "remote_model_sampling_is_not_bitwise_frozen": True,
            "observed_delta_is_not_a_cross_arm_causal_estimate": True,
            "no_gain_does_not_delete_retained_routes": True,
        },
    }


def _director_topology_replan_events(
    outcomes: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Expose malformed multi-step skeletons to one bounded global replan."""

    for outcome in outcomes:
        outcome_reasons = {
            str(value) for value in outcome.get("reasons") or [] if str(value).strip()
        }
        if (
            str(outcome.get("status") or "") == "failed"
            and "GlobalCampaignPlanValidationError" in outcome_reasons
        ):
            return ("director_contract_rejected",)
        if str(outcome.get("status") or "") != "accepted":
            continue
        for audit in outcome.get("proposal_audits") or []:
            if not isinstance(audit, Mapping) or audit.get("accepted") is True:
                continue
            reasons = {str(value) for value in audit.get("reasons") or [] if str(value).strip()}
            if reasons & _DIRECTOR_TOPOLOGY_REASONS:
                return ("director_topology_rejected",)
    return ()


def _planning_depth_requirement(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    minimum_steps: int,
) -> dict[str, Any]:
    """Audit planning depth without granting reaction proof or hiding short routes."""

    observed: list[dict[str, Any]] = []
    for outcome in outcomes:
        if str(outcome.get("status") or "") != "accepted":
            continue
        plan = outcome.get("plan")
        if not isinstance(plan, Mapping):
            continue
        audits_by_skeleton: dict[str, list[Mapping[str, Any]]] = {}
        for audit in outcome.get("proposal_audits") or []:
            if not isinstance(audit, Mapping):
                continue
            audits_by_skeleton.setdefault(str(audit.get("skeleton_id") or ""), []).append(audit)
        for skeleton in plan.get("multi_step_skeletons") or []:
            if not isinstance(skeleton, Mapping):
                continue
            skeleton_id = str(skeleton.get("skeleton_id") or "")
            steps = [step for step in skeleton.get("steps") or [] if isinstance(step, Mapping)]
            expected_ids = [str(step.get("step_id") or "") for step in steps]
            audits = audits_by_skeleton.get(skeleton_id, [])
            audited_ids = [str(audit.get("proposal_id") or "") for audit in audits]
            host_accepted = bool(
                steps
                and len(audits) == len(steps)
                and sorted(audited_ids) == sorted(expected_ids)
                and all(audit.get("accepted") is True for audit in audits)
            )
            observed.append(
                {
                    "skeleton_id": skeleton_id,
                    "step_count": len(steps),
                    "host_contract_accepted": host_accepted,
                }
            )
    accepted = [row for row in observed if row["host_contract_accepted"]]
    maximum = max((int(row["step_count"]) for row in accepted), default=0)
    return {
        "schema_version": "planning_depth_requirement.v1",
        "minimum_requested_steps": int(minimum_steps),
        "maximum_host_contract_accepted_steps": maximum,
        "requirement_met": minimum_steps <= 0 or maximum >= minimum_steps,
        "qualifying_skeleton_ids": sorted(
            str(row["skeleton_id"])
            for row in accepted
            if int(row["step_count"]) >= minimum_steps > 0
        ),
        "observed_skeletons": observed,
        "semantics": {
            "planning_depth_is_not_reaction_proof": True,
            "shorter_routes_remain_visible": True,
            "depth_deficit_triggers_at_most_the_existing_bounded_replan": True,
        },
    }


def _director_depth_replan_events(
    planning_depth: Mapping[str, Any],
) -> tuple[str, ...]:
    if (
        int(planning_depth.get("minimum_requested_steps") or 0) > 0
        and planning_depth.get("requirement_met") is not True
    ):
        return ("director_depth_deficit",)
    return ()


def _director_outcome_allows_replan(
    outcomes: Iterable[Mapping[str, Any]],
) -> bool:
    """Allow one bounded retry for accepted plans or host contract rejection."""

    for outcome in outcomes:
        if outcome.get("status") == "accepted" and outcome.get("plan"):
            return True
        reasons = {str(value) for value in outcome.get("reasons") or [] if str(value).strip()}
        if outcome.get("status") == "failed" and "GlobalCampaignPlanValidationError" in reasons:
            return True
    return False


def _replan_reasons(
    gates: Mapping[str, Any],
    *,
    material_events: tuple[str, ...],
    convergence_ledger: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    values = dict(gates.get("gates") or {})
    events = set(material_events)
    pressure = compile_replan_pressure(
        gates,
        material_events=events,
        convergence_ledger=convergence_ledger,
    )
    reasons: list[str] = []
    if "director_contract_rejected" in events:
        reasons.append("director_contract_deficit")
    if "director_depth_deficit" in events:
        reasons.append("planning_depth_deficit")
    if "director_topology_rejected" in events:
        reasons.append("director_topology_deficit")
    if "critical_edge_rejected" in events:
        reasons.append("critical_edge_failure_pressure")
    if "new_route_family" in events:
        reasons.append("new_route_family_pressure")
    if "provider_search_exhausted_without_proposal" in events:
        reasons.append("provider_search_failure_requires_new_frontier")
    if "shared_bottleneck_changed" in events:
        reasons.append("shared_bottleneck_pressure")
    if "source_conflict_added" in events:
        reasons.append("source_conflict_pressure")
    if pressure["derived_material_events"]:
        reasons.append("search_stagnation_with_route_diversity_deficit")
    if values.get("B2_host_validated_routes") is not True:
        reasons.append("host_validated_route_deficit")
    if values.get("B3_exact_multi_source") is not True and events & {
        "exact_rows_added",
        "material_evidence_added",
        "source_material_discovered",
        "visual_source_candidates_added",
    }:
        reasons.append("evidence_deficit_with_new_source_material")
    if values.get("B4_stock_boundary") is not True and events & {
        "stock_boundary_changed",
        "stock_records_added",
    }:
        reasons.append("stock_deficit_with_new_inventory_observation")
    return tuple(reasons)


def _evidence_observations(stage: Mapping[str, Any]) -> dict[str, Any]:
    discovery = dict(stage.get("discovery") or {})
    if not discovery:
        return {}
    visual = dict(dict(stage.get("visual_evidence") or {}).get("observation") or {})
    raw_sources = [
        dict(value) for value in discovery.get("sources") or [] if isinstance(value, Mapping)
    ]
    ranked_sources = sorted(
        raw_sources,
        key=lambda row: (
            -int(row.get("exact_row_count") or 0),
            -int(row.get("source_route_proposal_count") or 0),
            -len(row.get("procedure_inventory") or []),
            str(row.get("source_ref") or row.get("publication_number") or ""),
        ),
    )
    selected_sources = ranked_sources[:3]
    return {
        "schema_version": "campaign_evidence_observations.v1",
        "source_discovery": {
            "schema_version": str(discovery.get("schema_version") or ""),
            "provider_id": str(discovery.get("provider_id") or ""),
            "request_sha256": str(discovery.get("request_sha256") or ""),
            "content_sha256": str(discovery.get("content_sha256") or ""),
            "source_count": len(raw_sources),
            "selected_source_count": len(selected_sources),
            "omitted_source_count": max(0, len(raw_sources) - len(selected_sources)),
            "sources": [_source_replan_observation(value) for value in selected_sources],
            "semantics": {
                "bounded_chemistry_event_projection": True,
                "raw_source_documents_omitted": True,
                "source_text_grants_no_authority": True,
            },
        },
        "visual_source_candidates": _visual_replan_observation(visual),
        "semantics": {
            "untrusted_source_text_data_only": True,
            "grants_no_scientific_authority": True,
            "projection_is_for_source_consistent_replanning": True,
        },
    }


def _source_replan_observation(source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    procedures = [
        _procedure_replan_observation(value)
        for value in row.get("procedure_inventory") or []
        if isinstance(value, Mapping)
    ][:2]
    route = _source_route_replan_observation(dict(row.get("source_route_observation") or {}))
    return {
        **{
            key: row[key]
            for key in (
                "source_kind",
                "source_ref",
                "publication_number",
                "family_id",
                "doi",
                "pmid",
                "pmcid",
                "title",
                "acquisition_method",
                "source_fulltext_sha256",
                "html_sha256",
                "pdf_sha256",
                "page_count",
                "exact_row_count",
                "unresolved_edge_count",
                "source_route_proposal_count",
            )
            if key in row
        },
        "procedure_count": len(row.get("procedure_inventory") or []),
        "procedure_inventory": procedures,
        "source_route_observation": route,
        "semantics": {
            "procedure_text_is_untrusted_source_data": True,
            "route_proposals_require_normal_host_validation": True,
        },
    }


def _procedure_replan_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    excerpt = str(row.get("procedure_excerpt") or row.get("procedure") or row.get("text") or "")
    return {
        **{
            key: row[key]
            for key in (
                "label",
                "name",
                "page_number",
                "section",
                "source_artifact_kind",
                "source_artifact_sha256",
                "visual_expected",
            )
            if key in row
        },
        "procedure_excerpt": " ".join(excerpt.split())[:500],
    }


def _visual_replan_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project visual hypotheses without paths, hashes, or duplicated prose.

    The frozen observation remains in the artifact store.  Codex only needs
    connectivity, admission diagnostics, and concise conditions to decide
    whether a source-consistent route family should be proposed.
    """

    if not value:
        return {}
    row = dict(value)
    steps = [dict(item) for item in row.get("candidate_steps") or [] if isinstance(item, Mapping)]
    projected_steps = []
    for step in steps[:8]:
        condition = dict(step.get("condition_candidate") or {})
        projected_steps.append(
            {
                **{
                    key: step[key]
                    for key in (
                        "candidate_id",
                        "admission_eligible",
                        "allowed_use",
                        "chain_rejection_reasons",
                        "grants_exact_evidence",
                        "matched_current_edge_id",
                        "precursor_smiles",
                        "product_label",
                        "product_smiles",
                        "root_anchor",
                        "source_locator",
                    )
                    if key in step
                },
                "condition_candidate": {
                    key: condition[key]
                    for key in (
                        "condition_status",
                        "duration",
                        "reagent",
                        "reported_yield",
                        "solvent",
                        "source_grounding",
                        "temperature",
                    )
                    if key in condition
                },
            }
        )
    return {
        **{
            key: row[key]
            for key in (
                "schema_version",
                "source_ref",
                "source_artifact_kind",
                "provider_status",
                "candidate_step_count",
                "admission_eligible_step_count",
                "frontier_anchored_step_count",
                "matched_current_edge_count",
                "chain_admission_accepted",
                "chain_admission_reasons",
            )
            if key in row
        },
        "candidate_steps": projected_steps,
        "omitted_candidate_step_count": max(0, len(steps) - len(projected_steps)),
        "semantics": {
            "visual_hypotheses_are_not_proof": True,
            "full_observation_is_content_addressed_outside_prompt": True,
        },
    }


def _source_route_replan_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    proposals = [dict(row) for row in value.get("proposals") or [] if isinstance(row, Mapping)]
    diagnostics = [dict(row) for row in value.get("diagnostics") or [] if isinstance(row, Mapping)]
    return {
        "schema_version": str(value.get("schema_version") or ""),
        "source_ref": str(value.get("source_ref") or ""),
        "route_family": dict(value.get("route_family") or {}),
        "proposal_count": int(value.get("proposal_count") or len(proposals)),
        "resolved_procedure_count": int(value.get("resolved_procedure_count") or 0),
        "unconnected_proposal_count": int(value.get("unconnected_proposal_count") or 0),
        "proposals": [
            {
                **{
                    key: row[key]
                    for key in (
                        "proposal_id",
                        "source_ref",
                        "source_location",
                        "product_name",
                        "product_smiles",
                        "precursor_smiles",
                        "reactant_names",
                        "transformation_hypothesis",
                        "condition_candidate",
                        "product_structure_recovery_mode",
                        "admission_audit",
                    )
                    if key in row
                }
            }
            for row in proposals[:8]
        ],
        "diagnostics": [
            {
                key: row[key]
                for key in (
                    "label",
                    "product_name",
                    "status",
                    "reasons",
                    "selected_precursor_count",
                    "element_deficit",
                )
                if key in row
            }
            for row in diagnostics[:8]
        ],
        "semantics": {
            "source_route_is_observation_not_proof": True,
            "source_consistent_alternative_should_be_replanned_when_needed": True,
        },
    }


def _workbench_campaign_summary(
    *,
    gates: Mapping[str, Any],
    resource_envelope: Mapping[str, Any],
    model_cost: Mapping[str, Any],
    stop_decision: Mapping[str, Any],
    claim: Mapping[str, Any],
    current_disposition: Mapping[str, Any],
    planning_depth: Mapping[str, Any] | None = None,
    quality_state: Mapping[str, Any] | None = None,
    trajectory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "gates": dict(gates.get("gates") or {}),
        "highest_contiguous_gate": str(gates.get("highest_contiguous_gate") or "none"),
        "counts": dict(gates.get("counts") or {}),
        "resource_envelope": dict(resource_envelope),
        "model_cost": dict(model_cost),
        "stop_decision": dict(stop_decision),
        "claim": dict(claim),
        "current_disposition": dict(current_disposition),
        "planning_depth": dict(planning_depth or {}),
        "quality_state": dict(quality_state or {}),
        "trajectory": dict(trajectory or {}),
    }


def _current_disposition(
    *,
    kernel_status: str,
    stop_decision: Mapping[str, Any],
    claim: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    objective_achieved = claim.get("objective_achieved") is True
    scientifically_accepted = claim.get("accepted_under_configured_policy") is True
    stock_closed = claim.get("configured_stock_boundary_closed") is True
    historical_completion = bool(
        str(kernel_status) == "completed" or stop_decision.get("decision") == "completed"
    )
    proof_audit = dict(gates.get("reaction_proof_version_audit") or {})
    stale_terminal = historical_completion and not objective_achieved
    if scientifically_accepted:
        state = "accepted"
        reasons: list[str] = []
    elif stale_terminal:
        state = "terminal_snapshot_requires_revalidation"
        reasons = ["current_proof_policy_does_not_accept_historical_terminal_snapshot"]
        if proof_audit.get("requires_revalidation") is True:
            reasons.append("stale_reaction_validator_proofs_present")
    elif stock_closed:
        state = "stock_closed_proof_open"
        reasons = _open_gate_reasons(gates, include_host_validation=True)
    elif stop_decision.get("terminal") is True:
        state = str(stop_decision.get("decision") or "terminal_unresolved")
        reasons = [str(value) for value in stop_decision.get("reasons") or []]
    elif dict(gates.get("gates") or {}).get("B2_host_validated_routes") is True:
        state = "routes_validated_proof_open"
        reasons = _open_gate_reasons(gates, include_host_validation=False)
    elif dict(gates.get("gates") or {}).get("B1_global_multi_route") is True:
        state = "route_hypotheses_available_validation_open"
        reasons = _open_gate_reasons(gates, include_host_validation=True)
    else:
        state = "unresolved"
        reasons = [str(value) for value in stop_decision.get("reasons") or []]
    return {
        "schema_version": "target_solve_current_disposition.v1",
        "state": state,
        "objective_achieved": objective_achieved,
        "scientifically_accepted": scientifically_accepted,
        "historical_kernel_status": str(kernel_status),
        "historical_kernel_terminal": stop_decision.get("terminal") is True,
        "requires_revalidation": bool(
            stale_terminal or proof_audit.get("requires_revalidation") is True
        ),
        "reasons": reasons,
        "semantics": {
            "historical_terminal_state_cannot_override_current_proof_policy": True,
            "kernel_event_history_is_not_rewritten": True,
        },
    }


def _open_gate_reasons(
    gates: Mapping[str, Any],
    *,
    include_host_validation: bool,
) -> list[str]:
    values = dict(gates.get("gates") or {})
    reasons: list[str] = []
    if include_host_validation and values.get("B2_host_validated_routes") is not True:
        reasons.append("host_route_validation_open")
    if values.get("B3_exact_multi_source") is not True:
        reasons.append("exact_multi_source_proof_open")
    if values.get("B4_stock_boundary") is not True:
        reasons.append("configured_stock_boundary_open")
    if values.get("B5_configured_portfolio_acceptance") is not True:
        reasons.append("configured_portfolio_acceptance_open")
    return reasons


def _resource_envelope(
    *,
    model_cost: Mapping[str, Any],
    native_search: Mapping[str, Any],
    task_budget: Mapping[str, Any],
    run_wall_time_s: float,
    attempt_count: int,
    accepted_expansion_count: int,
    budget: RetrosynthesisRunBudget,
    max_run_wall_time_s: float,
) -> dict[str, Any]:
    observed = {
        "model_invocations": int(model_cost.get("model_invocations") or 0),
        "input_tokens": int(model_cost.get("input_tokens") or 0),
        "output_tokens": int(model_cost.get("output_tokens") or 0),
        "model_wall_time_s": float(model_cost.get("wall_time_s") or 0.0),
        "visual_invocations": int(model_cost.get("visual_invocations") or 0),
        "attempt_runs": int(attempt_count),
        "accepted_expansions": int(accepted_expansion_count),
        "native_search_committed": int(native_search.get("committed_total") or 0),
        "run_wall_time_s": float(run_wall_time_s),
    }
    limits = {
        "model_invocations": budget.max_model_invocations,
        "input_tokens": budget.max_total_input_tokens,
        "output_tokens": budget.max_total_output_tokens,
        "model_wall_time_s": budget.max_total_wall_time_s,
        "visual_invocations": budget.max_visual_invocations,
        "attempt_runs": budget.max_attempt_runs,
        "accepted_expansions": budget.max_accepted_expansions,
        "native_search_committed": budget.max_native_search_invocations,
        "run_wall_time_s": float(max_run_wall_time_s),
    }
    violations = sorted(
        f"{key}_budget_violated" for key, value in observed.items() if value > limits[key]
    )
    return {
        "schema_version": "target_solve_resource_envelope.v1",
        "within_budget": not violations,
        "observed": observed,
        "limits": limits,
        "native_search": dict(native_search),
        "task_budget": dict(task_budget),
        "violations": violations,
        "semantics": {
            "reaching_a_cap_is_compliant": True,
            "observed_overrun_blocks_qualified_acceptance": True,
            "task_dimensions_include_settled_and_reserved_capacity": True,
        },
    }


def _compile_target_trajectory_bindings(
    *,
    code_binding: Mapping[str, Any],
    campaign_spec: Mapping[str, Any],
    config: TargetSolveConfig,
    director_config: DirectorConfig,
    chemenzy_provider: Any,
    evidence_connector: Any,
    condition_predictor: Any,
    program_capabilities: Any,
    chemenzy_observation: Mapping[str, Any],
    chemenzy_runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    stock_oracle = dict(campaign_spec.get("stock_oracle") or {})
    stock_binding = dict(stock_oracle.get("binding") or {})
    config_row = asdict(config)
    director_row = director_config.to_dict()
    provider_row = {
        "codex": {
            "enabled": config.enable_codex,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "director_config_sha256": str(
                director_row.get("content_sha256") or _digest(director_row)
            ),
        },
        "chemenzy": {
            "enabled": config.enable_chemenzy,
            "adapter": _callable_runtime_binding(
                chemenzy_provider,
                fallback="autoplanner.builtin_chemenzy_probe",
            ),
            "search_preset": config.chemenzy_search_preset,
            "configured_environment_sha256": hashlib.sha256(
                str(config.chemenzy_env_prefix or "auto-discovery").encode("utf-8")
            ).hexdigest(),
            "observation_sha256": (
                _digest(dict(chemenzy_observation)) if chemenzy_observation else ""
            ),
            "provider_capability": dict(chemenzy_observation.get("provider_capability") or {}),
            "runtime": dict(chemenzy_runtime_binding),
        },
        "evidence": _callable_runtime_binding(
            evidence_connector,
            fallback="not_registered",
        ),
        "conditions": _callable_runtime_binding(
            condition_predictor,
            fallback="not_registered",
        ),
        "program_capability_catalog": {
            "available": bool(program_capabilities),
            "content_sha256": (_digest(program_capabilities) if program_capabilities else ""),
        },
    }
    return compile_trajectory_bindings(
        code=code_binding,
        config={
            "schema_version": config.schema_version,
            "target_solve_config_sha256": _digest(config_row),
            "director_config_sha256": str(
                director_row.get("content_sha256") or _digest(director_row)
            ),
            "scheduler_policy": config.action_scheduler_policy,
        },
        input_summary={
            "campaign_spec_schema": str(campaign_spec.get("schema_version") or ""),
            "campaign_spec_sha256": str(campaign_spec.get("content_sha256") or ""),
            "target_structure_sha256": hashlib.sha256(
                str(dict(campaign_spec.get("target") or {}).get("canonical_smiles") or "").encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
        stock_oracle={
            "oracle_id": str(stock_oracle.get("oracle_id") or ""),
            "boundary": str(stock_oracle.get("boundary") or ""),
            "reference_sha256": str(stock_oracle.get("content_sha256") or ""),
            "binding_sha256": str(stock_binding.get("content_sha256") or ""),
        },
        providers=provider_row,
    )


def _target_control_plane_code_binding(repository_root: Path) -> dict[str, Any]:
    component_paths = (
        "cascade_planner/application/action_scheduler.py",
        "cascade_planner/application/campaign_actions.py",
        "cascade_planner/application/campaign_trajectory.py",
        "cascade_planner/application/run_kernel.py",
        "cascade_planner/orchestration/unified_campaign_runtime.py",
        "cascade_planner/interfaces/target_solver.py",
    )
    components: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    loaded_package_root = Path(__file__).resolve().parents[2]
    for relative in component_paths:
        configured_path = repository_root / relative
        loaded_path = loaded_package_root / relative
        path = configured_path if configured_path.is_file() else loaded_path
        if not path.is_file():
            missing.append(relative)
            continue
        content = path.read_bytes()
        components[relative] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "source": ("configured_repository" if path == configured_path else "loaded_package"),
        }
    return {
        "producer": "autoplanner",
        "components": components,
        "component_bundle_sha256": _digest(components),
        "missing_components": missing,
        "source_bundle_complete": not missing,
    }


def _callable_runtime_binding(value: Any, *, fallback: str) -> dict[str, Any]:
    if value is None:
        return {"registered": False, "identity": fallback}
    subject = value if hasattr(value, "__module__") else type(value)
    row = {
        "registered": True,
        "module": str(getattr(subject, "__module__", "")),
        "qualname": str(getattr(subject, "__qualname__", type(value).__qualname__)),
    }
    descriptor = getattr(value, "descriptor", None)
    if descriptor is not None:
        if hasattr(descriptor, "to_dict"):
            row["descriptor"] = descriptor.to_dict()
        elif isinstance(descriptor, Mapping):
            row["descriptor"] = dict(descriptor)
    row["identity_sha256"] = _digest(row)
    return row


def _campaign_action_executions_from_stages(
    stages: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(row.get("detail") or {})
        for row in stages
        if isinstance(row, Mapping)
        and str(row.get("stage") or "").startswith("campaign_action_")
        and isinstance(row.get("detail"), Mapping)
        and isinstance(dict(row.get("detail") or {}).get("action"), Mapping)
    ]


def _latest_chemenzy_runtime_binding(
    stages: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    executions = _campaign_action_executions_from_stages(stages)
    for execution in reversed(executions):
        action = dict(execution.get("action") or {})
        if action.get("kind") != CampaignActionKind.CHEMENZY_TARGET_EXPAND.value:
            continue
        handler = dict(dict(execution.get("outcome") or {}).get("handler_result") or {})
        preflight = dict(handler.get("runtime_preflight") or {})
        capability = dict(preflight.get("capability_probe") or {})
        return {
            "provider_envelope": dict(handler.get("provider_envelope") or {}),
            "provider_registration": dict(handler.get("provider_registration") or {}),
            "request_sha256": str(handler.get("request_sha256") or ""),
            "raw_proposal_sha256": str(handler.get("raw_proposal_sha256") or ""),
            "raw_result_sha256": str(handler.get("raw_result_sha256") or ""),
            "replay_key_sha256": str(handler.get("replay_key_sha256") or ""),
            "random_seed": int(handler.get("random_seed") or 0),
            "provider_invocation_binding": dict(handler.get("provider_invocation_binding") or {}),
            "runtime_preflight_sha256": _digest(preflight) if preflight else "",
            "requested_one_step_models": list(preflight.get("requested_one_step_models") or []),
            "model_override_digest": str(preflight.get("model_override_digest") or ""),
            "model_content_binding_sha256": str(
                capability.get("model_content_binding_sha256")
                or preflight.get("model_content_binding_sha256")
                or ""
            ),
            "model_content_identity_complete": bool(
                capability.get(
                    "model_content_identity_complete",
                    preflight.get("model_content_identity_complete"),
                )
                is True
            ),
            "model_path_checks": list(capability.get("model_path_checks") or []),
            "stock_path_checks": list(capability.get("stock_path_checks") or []),
        }
    return {}


def _program_milestones_from_stages(
    stages: Iterable[Mapping[str, Any]],
) -> dict[str, bool]:
    milestones: dict[str, bool] = {}
    for execution in _campaign_action_executions_from_stages(stages):
        action = dict(execution.get("action") or {})
        outcome = dict(execution.get("outcome") or {})
        kind = str(action.get("kind") or "")
        if not (kind.startswith("program_") or kind == "experiment_feedback_ingest"):
            continue
        status = str(outcome.get("status") or execution.get("status") or "")
        if status in {"accepted", "completed", "reused"}:
            milestones[f"program:action:{kind}"] = True
        handler = dict(outcome.get("handler_result") or {})
        if kind == CampaignActionKind.PROGRAM_VALIDATE.value and (
            handler.get("accepted") is True or int(handler.get("validated_count") or 0) > 0
        ):
            milestones["program:validation_accepted"] = True
        if kind == CampaignActionKind.PROGRAM_ADMIT.value and (
            handler.get("accepted") is True
            or dict(handler.get("admission") or {}).get("accepted") is True
        ):
            milestones["program:admission_accepted"] = True
        if (
            kind == CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST.value
            and _positive_exact_boundary_claim(handler)
        ):
            milestones["experiment:positive_exact_boundary_claim"] = True
    return milestones


def _positive_exact_boundary_claim(handler: Mapping[str, Any]) -> bool:
    claim_set = dict(handler.get("experimental_claims") or {})
    oracle = dict(handler.get("experimental_claims_oracle") or {})
    if validate_experimental_claim_set(claim_set):
        return False
    oracle_material = dict(oracle)
    observed_oracle_digest = str(oracle_material.pop("content_sha256", ""))
    claim_set_digest = str(claim_set.get("content_sha256") or "")
    if not (
        oracle.get("schema_version") == EXPERIMENTAL_CLAIM_SET_ORACLE_SCHEMA
        and oracle.get("accepted") is True
        and observed_oracle_digest
        and observed_oracle_digest == _digest(oracle_material)
        and oracle.get("expected_claim_set_sha256") == claim_set_digest
        and oracle.get("observed_claim_set_sha256") == claim_set_digest
    ):
        return False
    return any(
        isinstance(value, Mapping)
        and value.get("polarity") == "positive"
        and value.get("grants_domain_validation") is True
        and value.get("generalization_scope") == "exact_boundary_only"
        for value in dict(claim_set.get("claims") or {}).values()
    )


def _stage(name: str, status: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    now = _utc_now()
    detail_row = dict(detail or {})
    content_bound_detail = bool(
        (
            str(name).startswith("campaign_snapshot_")
            and detail_row.get("schema_version") == "campaign_anytime_snapshot.v2"
        )
        or detail_row.get("schema_version") == "chemenzy_route_lineage.v1"
    ) and bool(
        str(detail_row.get("content_sha256") or "")
        and str(detail_row.get("content_sha256") or "")
        == _digest({key: value for key, value in detail_row.items() if key != "content_sha256"})
    )
    row = {
        "stage": name,
        "status": str(status),
        # A trajectory snapshot is itself the content-addressed authority.
        # Truncating it for a display-oriented stage projection would corrupt
        # the digest and make resume/replay unverifiable.
        "detail": (detail_row if content_bound_detail else _bounded_detail(detail_row)),
    }
    if status == "running":
        row["started_at"] = now
    else:
        row["completed_at"] = now
    return row


def _chemenzy_director_observation(
    stages: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    stage = next(
        (
            dict(value)
            for value in reversed(list(stages))
            if value.get("stage") == "chemenzy_baseline"
        ),
        {},
    )
    detail = dict(stage.get("detail") or {})
    return {
        "schema_version": "chemenzy_director_observation.v1",
        "provider_id": "chemenzy",
        "status": str(stage.get("status") or "not_run"),
        "provider_capability": dict(detail.get("provider_capability") or {}),
        "route_count": int(detail.get("route_count") or 0),
        "host_admitted_route_count": int(detail.get("host_admitted_route_count") or 0),
        "selected_proposal_route_count": int(detail.get("selected_proposal_route_count") or 0),
        "proposal_count": int(detail.get("proposal_count") or 0),
        "route_admission": list(detail.get("route_admission") or []),
        "semantics": {
            "provider_result_is_proposal_only": True,
            "topology_contains_exact_seed_steps": True,
            "director_should_reason_over_seed_as_a_global_route": True,
        },
    }


def _compile_chemenzy_route_lineage(
    stages: Iterable[Mapping[str, Any]],
    graph: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    source_rows = _chemenzy_provider_lineage_rows(stages)
    families = dict(graph.get("route_families") or {})
    hypotheses = dict(graph.get("hypotheses") or {})
    edges = dict(graph.get("edges") or {})
    measured_routes = [
        dict(value) for value in gates.get("routes") or [] if isinstance(value, Mapping)
    ]
    alias_to_id = {
        str(alias): str(route_id)
        for route_id, value in families.items()
        for alias in dict(value).get("aliases") or []
        if str(alias)
    }
    proposal_hypothesis_ids: dict[str, set[str]] = {}
    for hypothesis_id, value in hypotheses.items():
        for origin in dict(value).get("origin_records") or []:
            if not isinstance(origin, Mapping):
                continue
            proposal_id = str(origin.get("proposal_id") or "")
            if proposal_id:
                proposal_hypothesis_ids.setdefault(proposal_id, set()).add(str(hypothesis_id))
    rows: list[dict[str, Any]] = []
    disposition_counts: dict[str, int] = {}
    for source in source_rows:
        alias = str(source.get("canonical_route_family_alias") or "")
        proposal_ids = {str(value) for value in source.get("step_proposal_ids") or [] if str(value)}
        direct_hypothesis_ids = {
            hypothesis_id
            for proposal_id in proposal_ids
            for hypothesis_id in proposal_hypothesis_ids.get(proposal_id, set())
        }
        route_ids = {
            str(source.get("canonical_route_family_id") or ""),
            alias_to_id.get(alias, ""),
            *(
                str(route_id)
                for hypothesis_id in direct_hypothesis_ids
                for route_id in dict(hypotheses.get(hypothesis_id) or {}).get("route_family_ids")
                or []
                if str(route_id)
            ),
        } - {""}
        route_id = next(iter(sorted(route_ids)), "")
        family_rows = [dict(families.get(value) or {}) for value in sorted(route_ids)]
        route_measurements = [
            value
            for value in measured_routes
            if str(value.get("route_family_id") or "") in route_ids
        ]
        hypothesis_ids = sorted(
            direct_hypothesis_ids
            or {
                str(value)
                for family in family_rows
                for value in family.get("hypothesis_ids") or []
                if str(value)
            }
        )
        edge_ids = sorted(
            {
                f"edge:{dict(hypotheses.get(value) or {}).get('edge_digest')}"
                for value in hypothesis_ids
                if f"edge:{dict(hypotheses.get(value) or {}).get('edge_digest')}" in edges
            }
            or {
                str(value)
                for family in family_rows
                for value in family.get("edge_ids") or []
                if str(value)
            }
        )
        if (
            source.get("topology_conservation_applicable") is True
            and source.get("topology_conservation_accepted") is not True
        ):
            disposition = "canonical_topology_not_conserved"
        elif any(value.get("stock_closed") is True for value in route_measurements):
            disposition = "stock_closed"
        elif any(value.get("materialized") is True for value in route_measurements):
            disposition = "materialized_or_partially_materialized"
        elif edge_ids:
            disposition = "canonical_edges_present_outside_complete_measured_route"
        elif hypothesis_ids:
            disposition = "canonical_hypothesis_only"
        else:
            disposition = str(source.get("disposition") or "unresolved")
        disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1
        rows.append(
            {
                **source,
                "canonical_route_family_id": route_id,
                "canonical_route_family_ids": sorted(route_ids),
                "canonical_route_ids": sorted(
                    str(value.get("route_id") or value.get("skeleton_id") or "")
                    for value in route_measurements
                    if str(value.get("route_id") or value.get("skeleton_id") or "")
                ),
                "stock_closed_route_ids": sorted(
                    str(value.get("route_id") or value.get("skeleton_id") or "")
                    for value in route_measurements
                    if value.get("stock_closed") is True
                    and str(value.get("route_id") or value.get("skeleton_id") or "")
                ),
                "canonical_hypothesis_ids": hypothesis_ids,
                "canonical_edge_ids": edge_ids,
                "canonical_route_closed": bool(route_measurements),
                "canonical_stock_closure_rate": (
                    sum(1 for value in route_measurements if value.get("stock_closed") is True)
                    / len(route_measurements)
                    if route_measurements
                    else 0.0
                ),
                "canonical_minimum_proof_level": int(
                    max(
                        (int(family.get("minimum_proof_level") or 0) for family in family_rows),
                        default=0,
                    )
                ),
                "blocking_deficit_ids": sorted(
                    {
                        str(value)
                        for family in family_rows
                        for value in family.get("blocking_deficit_ids") or []
                        if str(value)
                    }
                ),
                "final_disposition": disposition,
            }
        )
    payload = {
        "schema_version": "chemenzy_route_lineage.v1",
        "route_count": len(rows),
        "disposition_counts": {key: disposition_counts[key] for key in sorted(disposition_counts)},
        "campaign_B4_stock_boundary": bool(
            dict(gates.get("gates") or {}).get("B4_stock_boundary") is True
        ),
        "routes": rows,
        "semantics": {
            "raw_normalized_canonical_bound_by_digest": True,
            "campaign_B4_does_not_imply_each_provider_route_is_closed": True,
            "missing_proof_never_erases_provider_lineage": True,
            "partial_step_binding_never_counts_as_complete_route_conservation": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _chemenzy_provider_lineage_rows(
    stages: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    provider_results = [
        dict(stage.get("detail") or {})
        for stage in stages
        if stage.get("stage")
        in {"chemenzy_baseline", "chemenzy_guided_frontier", "chemenzy_stock_recovery"}
        and dict(stage.get("detail") or {}).get("route_lineage")
    ]
    source_rows: dict[str, dict[str, Any]] = {}
    for result in provider_results:
        invocation = {
            "provider_mode": str(result.get("mode") or "seed"),
            "provider_scope": str(result.get("scope") or "seed"),
            "provider_request_sha256": str(result.get("request_sha256") or ""),
            "provider_raw_proposal_sha256": str(result.get("raw_proposal_sha256") or ""),
            "provider_raw_result_sha256": str(result.get("raw_result_sha256") or ""),
            "provider_replay_key_sha256": str(result.get("replay_key_sha256") or ""),
            "provider_random_seed": int(result.get("random_seed") or 0),
        }
        for value in result.get("route_lineage") or []:
            if not isinstance(value, Mapping):
                continue
            source = {**invocation, **dict(value)}
            identity = {
                **invocation,
                "route_trace_id": str(source.get("route_trace_id") or ""),
                "normalized_route_sha256": str(source.get("normalized_route_sha256") or ""),
            }
            source_rows[_digest(identity)] = source
    return [source_rows[key] for key in sorted(source_rows)]


def _chemenzy_vendor_root(configured_root: Path) -> Path:
    """Bind discovery to this gateway instead of process cwd global state."""

    direct = configured_root.resolve()
    if (direct / "retro_planner").is_dir():
        return direct
    return direct / "ChemEnzyRetroPlanner"


def _bounded_detail(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "<depth-limited>"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name in {"graph", "portfolio", "snapshot"} and isinstance(child, Mapping):
                out[f"{name}_summary"] = {
                    "revision": child.get("revision") or child.get("graph_revision"),
                    "molecule_count": len(child.get("molecules") or {}),
                    "edge_count": len(child.get("edges") or {}),
                    "accepted": child.get("accepted"),
                    "content_sha256": child.get("content_sha256"),
                }
                continue
            out[name] = _bounded_detail(child, depth=depth + 1)
        return out
    if isinstance(value, list | tuple):
        rows = [_bounded_detail(child, depth=depth + 1) for child in value[:64]]
        if len(value) > 64:
            rows.append({"omitted_count": len(value) - 64})
        return rows
    if isinstance(value, str) and len(value) > 2_000:
        return value[:2_000] + f"... <{len(value) - 2_000} chars omitted>"
    return value


def _deduplicate_stages(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The RunKernel event chain and content-addressed artifacts retain full
    # history.  Checkpoints/reports are current projections and must not append
    # megabytes of near-identical stage payload on every resume.
    rows: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(values):
        name = str(row.get("stage") or "")
        key = name or f"unnamed:{_digest(row.get('detail') or {})}"
        current = dict(row)
        previous = dict(rows.get(key, (-1, {}))[1])
        if (
            previous.get("status") == "running"
            and current.get("status") != "running"
            and previous.get("started_at")
        ):
            current["started_at"] = previous["started_at"]
            elapsed_s = _elapsed_between(
                str(previous["started_at"]),
                str(current.get("completed_at") or _utc_now()),
            )
            if elapsed_s is not None:
                current["elapsed_s"] = elapsed_s
        rows[key] = (index, current)
    return [row for _, row in sorted(rows.values(), key=lambda value: value[0])]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _elapsed_between(started_at: str, completed_at: str) -> float | None:
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(max(0.0, (completed - started).total_seconds()), 3)


def _should_retry_chemenzy_timeout(
    stages: Iterable[Mapping[str, Any]],
    *,
    resume: bool,
    requested_timeout_s: float,
) -> bool:
    """Retry only an earlier timeout and only with a genuinely larger window."""

    rows = list(stages)
    if not resume or not rows or rows[-1].get("status") != "timeout":
        return False
    detail = dict(rows[-1].get("detail") or {})
    limits = dict(detail.get("limits") or {})
    previous_timeout_s = float(limits.get("timeout_s") or 0.0)
    return requested_timeout_s > previous_timeout_s


def _run_director_safely(
    service: Any,
    *,
    mode: str,
    material_events: tuple[str, ...] = (),
    evidence_observations: Mapping[str, Any] | tuple[Mapping[str, Any], ...] | None = None,
    context: Any | None = None,
    before_plan_admission: Callable[[], None] | None = None,
    idempotency_key: str,
) -> dict[str, Any]:
    """Turn a bounded provider failure into an auditable unresolved outcome."""

    try:
        if context is not None:
            return service.run_global_director_with_context(
                context,
                mode=mode,
                before_plan_admission=before_plan_admission,
                idempotency_key=idempotency_key,
            ).to_dict()
        return service.run_global_director(
            mode=mode,
            material_events=material_events,
            evidence_observations=evidence_observations,
            idempotency_key=idempotency_key,
        ).to_dict()
    except (GlobalCampaignDirectorError, RunKernelBudgetError) as exc:
        reason = str(exc).strip() or type(exc).__name__
        budget_exhausted = isinstance(exc, RunKernelBudgetError)
        return DirectorOutcome(
            status="skipped" if budget_exhausted else "failed",
            invoked=not budget_exhausted,
            cache_hit=False,
            mode=mode,
            context_sha256="",
            reasons=(type(exc).__name__, reason[:2_000]),
        ).to_dict()


def _empty_checkpoint(run_id: str) -> dict[str, Any]:
    return compile_target_solver_checkpoint(
        run_id,
        (),
        (),
        complete=False,
    )


def _checkpoint(
    path: Path,
    run_id: str,
    stages: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    complete: bool = False,
    resume_cursor: Mapping[str, Any] | None = None,
) -> None:
    _write_json_atomic(
        path,
        compile_target_solver_checkpoint(
            run_id,
            _deduplicate_stages(stages),
            outcomes,
            complete=complete,
            resume_cursor=resume_cursor,
        ),
    )


def _mark_stage_running(
    path: Path,
    run_id: str,
    stages: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    name: str,
    **detail: Any,
) -> None:
    """Persist a small live stage marker before one potentially slow call."""

    stages.append(_stage(name, "running", detail))
    _checkpoint(path, run_id, stages, outcomes)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    value = _read_json_object(path, "target_solver_checkpoint_missing")
    if value.get("schema_version") != TARGET_SOLVE_CHECKPOINT_SCHEMA:
        raise BlindBenchmarkError("target_solver_checkpoint_schema_invalid")
    return value


def _read_json_object(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindBenchmarkError(reason) from exc
    if not isinstance(value, Mapping):
        raise BlindBenchmarkError(reason)
    return dict(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _route_has_canonical_edges(route: Mapping[str, Any]) -> bool:
    """Return whether a portfolio row is a materialized source route."""

    return bool(
        str(route.get("route_id") or "")
        and {
            str(value)
            for value in route.get("edge_ids") or []
            if str(value)
        }
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DEFAULT_TARGET_DIRECTOR_MODEL",
    "TARGET_SOLVE_CHECKPOINT_SCHEMA",
    "TARGET_SOLVE_REPORT_SCHEMA",
    "TargetSolveConfig",
    "solve_target",
]
