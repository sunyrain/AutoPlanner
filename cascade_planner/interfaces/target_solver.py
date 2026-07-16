"""Target-only, bounded V4 retrosynthesis campaign orchestration."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from cascade_planner.application.blind_acceptance import (
    compile_blind_acceptance_report,
)
from cascade_planner.application.blind_benchmark_contract import (
    BLIND_CASE_SCHEMA,
    BlindBenchmarkError,
    BlindCase,
    audit_blind_preflight,
    canonical_smiles,
)
from cascade_planner.application.campaign_context import CampaignContextTooLargeError
from cascade_planner.application.canonical_hypergraph import molecule_identity
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.reaction_mapping import ReactionMapper
from cascade_planner.application.proof_portfolio import compile_proof_portfolio
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
from cascade_planner.interfaces.patent_self_evolution import (
    PatentSelfEvolutionSession,
)
from cascade_planner.interfaces.target_solver_stages import (
    InventorySnapshotBuilder,
    StockCatalogBuilder,
    audit_authoritative_inventory_stock,
    audit_live_benchmark_stock,
    discover_director_source_hints,
    ingest_source_discovery_observation,
    materialize_discovered_source_routes,
    repair_rejected_precursor_typos,
    validate_materialized_edges,
)
from cascade_planner.interfaces.target_identity import (
    TARGET_IDENTITY_PROVIDER_VERSION,
    resolve_target_identity,
)
from cascade_planner.interfaces.visual_evidence import (
    VisualEvidenceProvider,
    acquire_visual_evidence_candidates,
    materialize_visual_evidence_candidates,
)
from cascade_planner.orchestration.global_campaign_director import (
    DirectorConfig,
    DirectorOutcome,
    DirectorRunner,
    GlobalCampaignDirectorError,
    director_prompt,
)


if TYPE_CHECKING:
    from cascade_planner.interfaces.campaign_gateway import CampaignGateway


TARGET_SOLVE_REPORT_SCHEMA = "target_only_retrosynthesis_solve_report.v1"
TARGET_SOLVE_CHECKPOINT_SCHEMA = "target_only_solve_checkpoint.v1"
DEFAULT_TARGET_DIRECTOR_MODEL = "gpt-5.5"
_DIRECTOR_TOPOLOGY_REASONS = frozenset(
    {
        "skeleton_ancestor_cycle",
        "skeleton_contains_disconnected_steps",
        "skeleton_product_expanded_more_than_once",
        "skeleton_requires_exactly_one_target_root",
    }
)


@dataclass(frozen=True, slots=True)
class TargetSolveConfig:
    model: str = DEFAULT_TARGET_DIRECTOR_MODEL
    reasoning_effort: str = "low"
    execution_profile: str = "standard"
    use_coordinator: bool = False
    enable_web_search: bool = True
    enable_initial_director_web_search: bool = False
    enable_replan: bool = True
    enable_live_benchmark_stock: bool = True
    enable_builtin_patent_evidence: bool = False
    enable_patent_self_evolution: bool = True
    enable_chemenzy: bool = True
    enable_target_chemenzy_baseline: bool = False
    enable_guided_chemenzy: bool = True
    enable_target_identity: bool = True
    resolve_named_target_identity: bool = False
    blind_audit_root: str = ""
    chemenzy_env_prefix: str = ""
    self_evo_library_path: str = ""
    max_atom_mapping_reactions: int = 48
    max_live_stock_molecules: int = 24
    max_patent_sources: int = 3
    max_self_evo_template_candidates: int = 12
    max_chemenzy_routes: int = 2
    max_chemenzy_steps: int = 6
    max_chemenzy_iterations: int = 10
    chemenzy_expansion_topk: int = 20
    chemenzy_timeout_s: float = 90.0
    max_guided_chemenzy_frontiers: int = 3
    max_guided_chemenzy_iterations: int = 6
    guided_chemenzy_timeout_s: float = 60.0
    max_visual_evidence_pages: int = 6
    minimum_planning_route_steps: int = 0
    max_director_output_tokens: int = 18_000
    max_director_wall_time_s: float = 360.0
    publish_intermediate_workbenches: bool = True
    schema_version: str = "target_solve_config.v1"

    def __post_init__(self) -> None:
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("target solver reasoning effort is invalid")
        if self.execution_profile not in {"fast", "standard", "proof"}:
            raise ValueError("target solver execution profile is invalid")
        if self.max_atom_mapping_reactions < 1 or self.max_live_stock_molecules < 1:
            raise ValueError("target solver deterministic limits must be positive")
        if not 1 <= self.max_patent_sources <= 8:
            raise ValueError("target solver patent source limit is invalid")
        if not 1 <= self.max_visual_evidence_pages <= 12:
            raise ValueError("target solver visual evidence page limit is invalid")
        if not 1 <= self.max_self_evo_template_candidates <= 64:
            raise ValueError("target solver self-evolution candidate limit is invalid")
        if not 1 <= self.max_chemenzy_routes <= 4:
            raise ValueError("target solver ChemEnzy route limit is invalid")
        if min(
            self.max_chemenzy_steps,
            self.max_chemenzy_iterations,
            self.chemenzy_expansion_topk,
        ) < 1 or self.chemenzy_timeout_s <= 0:
            raise ValueError("target solver ChemEnzy budget is invalid")
        if not 1 <= self.max_guided_chemenzy_frontiers <= 6:
            raise ValueError("target solver guided ChemEnzy frontier limit is invalid")
        if (
            self.max_guided_chemenzy_iterations < 1
            or self.guided_chemenzy_timeout_s <= 0
        ):
            raise ValueError("target solver guided ChemEnzy budget is invalid")
        if self.max_director_output_tokens < 1 or self.max_director_wall_time_s <= 0:
            raise ValueError("target solver director limits must be positive")
        if (
            isinstance(self.minimum_planning_route_steps, bool)
            or not isinstance(self.minimum_planning_route_steps, int)
            or not 0 <= self.minimum_planning_route_steps <= 24
        ):
            raise ValueError("target solver minimum planning route depth is invalid")


def solve_target(
    gateway: "CampaignGateway",
    *,
    target_name: str,
    target_smiles: str,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    acceptance: RetrosynthesisAcceptanceSpec | None = None,
    budget: RetrosynthesisRunBudget | None = None,
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
    resolved_budget = budget or RetrosynthesisRunBudget(
        max_model_invocations=2,
        max_total_input_tokens=50_000,
        max_total_output_tokens=14_000,
        max_total_wall_time_s=720.0,
        max_visual_invocations=0,
        max_accepted_expansions=32,
        max_attempt_runs=72,
        max_prompt_context_bytes=96_000,
    )
    supplied_name = " ".join(str(target_name or "").split())
    opaque_name = f"target-{hashlib.sha256(canonical.encode()).hexdigest()[:8]}"
    campaign_name = (
        supplied_name
        if supplied_name.casefold()
        not in {"", "blind target", "blind molecule", "opaque target", "target", "unknown target"}
        else opaque_name
    )
    identity = gateway._normalize_run_id(
        run_id or gateway._new_run_id(campaign_name, canonical)
    )
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
    existing = (directory / ".autoplanner" / "kernel" / "run_spec.json").is_file()
    if existing and not resume:
        raise BlindBenchmarkError("blind_run_exists_use_resume")
    if existing:
        checkpoint = _read_checkpoint(checkpoint_path)
        preflight = _read_json_object(preflight_path, "blind_preflight_missing_on_resume")
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
    }[active.execution_profile]
    minimum_director_families = max(
        director_profile["minimum_route_families"],
        resolved_acceptance.minimum_complete_routes,
    )
    if active.minimum_planning_route_steps > director_profile["max_steps_per_skeleton"]:
        raise ValueError(
            "minimum planning route depth exceeds execution profile capacity"
        )
    director_config = DirectorConfig(
        minimum_route_families=minimum_director_families,
        minimum_planning_route_steps=active.minimum_planning_route_steps,
        max_route_families=max(
            director_profile["max_route_families"],
            minimum_director_families,
        ),
        max_skeletons=max(
            director_profile["max_skeletons"],
            minimum_director_families,
        ),
        max_steps_per_skeleton=director_profile["max_steps_per_skeleton"],
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
        max_event_replan_calls=1,
        max_final_portfolio_synthesis_calls=1,
        model=active.model,
        reasoning_effort=active.reasoning_effort,
        enable_web_search=active.enable_web_search,
        enable_initial_web_search=active.enable_initial_director_web_search,
        use_coordinator=active.use_coordinator,
    )
    service = gateway._open(
        identity,
        run_dir=directory,
        director_runner=director_runner,
        director_config=director_config,
    )
    self_evo = PatentSelfEvolutionSession.create(
        enabled=active.enable_patent_self_evolution,
        configured_path=active.self_evo_library_path,
        external_data_root=gateway.paths.external_data_root,
        target_smiles=canonical,
        max_candidates=active.max_self_evo_template_candidates,
    )
    resolved_evidence_connector = evidence_connector
    if (
        resolved_evidence_connector is None
        and active.enable_builtin_patent_evidence
    ):
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
    if checkpoint.get("complete") is True and service.kernel.decide_stop().terminal:
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
    target_identity = _target_identity_stage(
        service,
        stages=stages,
        target_name=case.target_name,
        target_smiles=canonical,
        enabled=active.enable_target_identity,
        resolve_named=active.resolve_named_target_identity,
    )
    identity_stage = _stage(
        "target_identity", target_identity["status"], target_identity
    )
    identity_indices = [
        index
        for index, row in enumerate(stages)
        if row.get("stage") == "target_identity"
    ]
    if identity_indices:
        stages[identity_indices[-1]] = identity_stage
    else:
        stages.append(identity_stage)
    resolved_target_name = str(
        dict(target_identity.get("identity") or {}).get("preferred_name")
        or case.target_name
    )
    evidence_prefetch_executor: ThreadPoolExecutor | None = None
    evidence_prefetch_future: Future[dict[str, Any]] | None = None
    if (
        not outcomes
        and resolved_evidence_connector is not None
        and getattr(
            resolved_evidence_connector,
            "autoplanner_prefetch_safe",
            False,
        )
        is True
    ):
        prefetch_request = compile_evidence_acquisition_request(
            run_id=service.kernel.spec.run_id,
            target_name=resolved_target_name,
            target_smiles=service.kernel.spec.target_smiles,
            graph=service.graph_store.load(),
            source_frontier={},
            target_identity=dict(target_identity.get("identity") or {}),
            prefetch_mode=True,
        )
        evidence_prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="autoplanner-evidence-prefetch",
        )
        evidence_prefetch_future = evidence_prefetch_executor.submit(
            _run_evidence_prefetch,
            prefetch_request,
            resolved_evidence_connector,
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
    prior_chemenzy_stages = [
        row for row in stages if row.get("stage") == "chemenzy_baseline"
    ]
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
        chemenzy_stage = run_chemenzy_proposal_stage(
            service,
            target_name=case.target_name,
            target_smiles=canonical,
            enabled=(
                active.enable_chemenzy and active.enable_target_chemenzy_baseline
            ),
            provider=chemenzy_provider,
            env_prefix=active.chemenzy_env_prefix or None,
            vendor_root=_chemenzy_vendor_root(gateway.paths.vendor_root),
            max_routes=active.max_chemenzy_routes,
            max_steps=active.max_chemenzy_steps,
            max_iterations=active.max_chemenzy_iterations,
            expansion_topk=active.chemenzy_expansion_topk,
            timeout_s=active.chemenzy_timeout_s,
        )
        stages.append(_stage("chemenzy_baseline", chemenzy_stage["status"], chemenzy_stage))
        _checkpoint(checkpoint_path, identity, stages, outcomes)
    chemenzy_observation = _chemenzy_director_observation(stages)
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
        initial = _run_director_safely(
            service,
            mode="initial_architecture",
            evidence_observations={
                **dict(initial_template_observation),
                "target_identity": target_identity,
                "chemenzy_provider_observation": chemenzy_observation,
            },
            idempotency_key="solve-target:director:initial",
        )
        outcomes.append(initial)
        stages.append(_stage("global_campaign", initial["status"], initial))
        _checkpoint(checkpoint_path, identity, stages, outcomes)

    materialization = service.execute_frontier_materialization(
        idempotency_key=f"solve-target:materialize:{service.kernel.state.graph_revision}"
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
    validation = validate_materialized_edges(
        service,
        atom_mapper=atom_mapper,
        max_reactions=active.max_atom_mapping_reactions,
    )
    stages.append(_stage("reaction_validation", validation["status"], validation))
    repair_stage = repair_rejected_precursor_typos(service, validation)
    stages.append(_stage("precursor_repair", repair_stage["status"], repair_stage))
    repair_validation: dict[str, Any] = {}
    if int(repair_stage.get("accepted_repair_count") or 0) > 0:
        repair_validation = validate_materialized_edges(
            service,
            atom_mapper=atom_mapper,
            max_reactions=active.max_atom_mapping_reactions,
        )
        stages.append(
            _stage(
                "precursor_repair_validation",
                repair_validation["status"],
                repair_validation,
            )
        )

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

    # Codex owns the global route architecture.  ChemEnzy is delegated only
    # canonical subtargets selected in that architecture, before source search
    # and stock closure make those local expansions more expensive to revisit.
    prior_attempted_frontiers = _attempted_chemenzy_frontiers(stages)
    prior_guided = _latest_stage(stages, "chemenzy_guided_frontier")
    prior_guided_status = str(prior_guided.get("status") or "")
    reuse_guided = prior_guided_status in {
        "completed",
        "unresolved",
        "not_needed",
        "reused",
    } or len(prior_attempted_frontiers) >= active.max_guided_chemenzy_frontiers
    if (
        prior_guided_status == "not_needed"
        and int(delegation_audit.get("queued_count") or 0) > 0
        and len(prior_attempted_frontiers)
        < active.max_guided_chemenzy_frontiers
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
            active.max_guided_chemenzy_frontiers - 1
            if active.max_guided_chemenzy_frontiers > 1
            else 1
        )
        guided_stage = run_chemenzy_guided_frontier_stage(
            service,
            target_name=case.target_name,
            root_target_smiles=canonical,
            enabled=active.enable_chemenzy and active.enable_guided_chemenzy,
            provider=chemenzy_provider,
            env_prefix=active.chemenzy_env_prefix or None,
            vendor_root=_chemenzy_vendor_root(gateway.paths.vendor_root),
            max_frontiers=initial_guided_limit,
            max_routes=1,
            max_steps=active.max_chemenzy_steps,
            max_iterations=active.max_guided_chemenzy_iterations,
            expansion_topk=min(10, active.chemenzy_expansion_topk),
            timeout_s=active.guided_chemenzy_timeout_s,
            exclude_frontier_smiles=tuple(sorted(prior_attempted_frontiers)),
        )
        guided_stage["new_proposal_count"] = int(
            guided_stage.get("proposal_count") or 0
        )
    stages.append(
        _stage("chemenzy_guided_frontier", guided_stage["status"], guided_stage)
    )
    guided_materialization: dict[str, Any] = {}
    guided_validation: dict[str, Any] = {}
    guided_stock: dict[str, Any] = {}
    if int(
        guided_stage.get("new_proposal_count", guided_stage.get("proposal_count"))
        or 0
    ) > 0:
        guided_materialization = service.execute_frontier_materialization(
            idempotency_key=(
                "solve-target:guided-materialize:"
                f"{service.kernel.state.graph_revision}"
            )
        )
        stages.append(
            _stage(
                "guided_materialization",
                "completed" if guided_materialization.get("changed") else "reused_or_empty",
                guided_materialization,
            )
        )
        guided_validation = validate_materialized_edges(
            service,
            atom_mapper=atom_mapper,
            max_reactions=active.max_atom_mapping_reactions,
        )
        stages.append(
            _stage("guided_reaction_validation", guided_validation["status"], guided_validation)
        )

    # Evidence discovery and leaf stock audit deliberately precede the only
    # optional replan.  Their host-owned observations therefore enter the next
    # CampaignContext instead of making the director repeat a blind first pass.
    source_stage = discover_director_source_hints(service, outcomes)
    evidence_prefetch = _resolve_evidence_prefetch(
        evidence_prefetch_future,
        evidence_prefetch_executor,
    )
    source_stage = _merge_prefetched_source_hints(source_stage, evidence_prefetch)
    stages.append(_stage("source_frontier", source_stage["status"], source_stage))
    _mark_stage_running(
        checkpoint_path,
        identity,
        stages,
        outcomes,
        "evidence_acquisition",
        visual_enabled=visual_evidence_provider is not None,
    )
    evidence_stage = _acquire_evidence_stage(
        service,
        source_stage=source_stage,
        connector=resolved_evidence_connector,
        atom_mapper=atom_mapper,
        visual_provider=visual_evidence_provider,
        max_visual_pages=active.max_visual_evidence_pages,
        target_name=resolved_target_name,
        target_identity=dict(target_identity.get("identity") or {}),
    )
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
        learned_template_validation = validate_materialized_edges(
            service,
            atom_mapper=atom_mapper,
            max_reactions=active.max_atom_mapping_reactions,
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
    stock_stage = _audit_stock_stage(
        service,
        acceptance=resolved_acceptance,
        config=active,
        catalog_builder=stock_catalog_builder,
        inventory_builder=inventory_snapshot_builder,
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
        *(
            str(value)
            for value in guided_stage.get("frontier_smiles") or []
            if str(value)
        ),
    }
    remaining_guided = max(
        0,
        active.max_guided_chemenzy_frontiers - len(attempted_guided_frontiers),
    )
    if remaining_guided:
        recovery_stage = run_chemenzy_guided_frontier_stage(
            service,
            target_name=case.target_name,
            root_target_smiles=canonical,
            enabled=active.enable_chemenzy and active.enable_guided_chemenzy,
            provider=chemenzy_provider,
            env_prefix=active.chemenzy_env_prefix or None,
            vendor_root=_chemenzy_vendor_root(gateway.paths.vendor_root),
            max_frontiers=remaining_guided,
            max_routes=1,
            max_steps=active.max_chemenzy_steps,
            max_iterations=active.max_guided_chemenzy_iterations,
            expansion_topk=min(10, active.chemenzy_expansion_topk),
            timeout_s=active.guided_chemenzy_timeout_s,
            exclude_frontier_smiles=tuple(
                sorted(attempted_guided_frontiers)
            ),
        )
        stages.append(
            _stage("chemenzy_stock_recovery", recovery_stage["status"], recovery_stage)
        )
    if int(recovery_stage.get("proposal_count") or 0) > 0:
        recovery_materialization = service.execute_frontier_materialization(
            idempotency_key=(
                "solve-target:recovery-materialize:"
                f"{service.kernel.state.graph_revision}"
            )
        )
        stages.append(
            _stage(
                "recovery_materialization",
                "completed"
                if recovery_materialization.get("changed")
                else "reused_or_empty",
                recovery_materialization,
            )
        )
        recovery_validation = validate_materialized_edges(
            service,
            atom_mapper=atom_mapper,
            max_reactions=active.max_atom_mapping_reactions,
        )
        stages.append(
            _stage(
                "recovery_reaction_validation",
                recovery_validation["status"],
                recovery_validation,
            )
        )
        recovery_stock = _audit_stock_stage(
            service,
            acceptance=resolved_acceptance,
            config=active,
            catalog_builder=stock_catalog_builder,
            inventory_builder=inventory_snapshot_builder,
        )
        stages.append(_stage("recovery_stock", recovery_stock["status"], recovery_stock))
        stock_stage = recovery_stock

    provisional = service.closeout(
        idempotency_key=f"solve-target:provisional:{service.kernel.state.graph_revision}"
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
                    materialization,
                    template_reuse,
                    validation,
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
    replan_reasons = _replan_reasons(
        provisional_gates,
        material_events=material_events,
    )
    needs_replan = bool(
        active.enable_replan
        and replan_reasons
        and _director_outcome_allows_replan(outcomes)
        and len(outcomes) < 2
    )
    evidence_observations = {
        **_evidence_observations(evidence_stage),
        **self_evo.observation(dict(learned_template_reuse.get("retrieval") or {})),
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
                material_events=material_events,
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
    if needs_replan:
        stages.append(
            _stage(
                "global_replan_budget_gate",
                "accepted" if replan_guard["accepted"] else "skipped",
                replan_guard,
            )
        )
    if needs_replan and replan_guard["accepted"]:
        _mark_stage_running(
            checkpoint_path,
            identity,
            stages,
            outcomes,
            "global_replan",
            model=active.model,
            mode="event_replan",
        )
        replan = _run_director_safely(
            service,
            mode="event_replan",
            material_events=material_events,
            evidence_observations=evidence_observations,
            idempotency_key="solve-target:director:replan",
        )
        outcomes.append(replan)
        stages.append(_stage("global_replan", replan["status"], replan))
        _checkpoint(checkpoint_path, identity, stages, outcomes)
        if replan.get("status") == "accepted" and replan.get("plan"):
            rematerialization = service.execute_frontier_materialization(
                idempotency_key=(
                    "solve-target:replan-materialize:"
                    f"{service.kernel.state.graph_revision}"
                )
            )
            stages.append(
                _stage(
                    "replan_materialization",
                    (
                        "completed"
                        if rematerialization.get("changed")
                        else "reused_or_empty"
                    ),
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
            revalidation = validate_materialized_edges(
                service,
                atom_mapper=atom_mapper,
                max_reactions=active.max_atom_mapping_reactions,
            )
            stages.append(
                _stage("replan_validation", revalidation["status"], revalidation)
            )
            replan_repair = repair_rejected_precursor_typos(service, revalidation)
            stages.append(
                _stage("replan_precursor_repair", replan_repair["status"], replan_repair)
            )
            if int(replan_repair.get("accepted_repair_count") or 0) > 0:
                repaired_revalidation = validate_materialized_edges(
                    service,
                    atom_mapper=atom_mapper,
                    max_reactions=active.max_atom_mapping_reactions,
                )
                stages.append(
                    _stage(
                        "replan_precursor_repair_validation",
                        repaired_revalidation["status"],
                        repaired_revalidation,
                    )
                )
            source_stage = discover_director_source_hints(service, outcomes)
            stages.append(
                _stage("replan_source_frontier", source_stage["status"], source_stage)
            )
            _mark_stage_running(
                checkpoint_path,
                identity,
                stages,
                outcomes,
                "replan_evidence_acquisition",
                visual_enabled=visual_evidence_provider is not None,
            )
            evidence_stage = _acquire_evidence_stage(
                service,
                source_stage=source_stage,
                connector=resolved_evidence_connector,
                atom_mapper=atom_mapper,
                visual_provider=visual_evidence_provider,
                max_visual_pages=active.max_visual_evidence_pages,
                target_name=resolved_target_name,
                target_identity=dict(target_identity.get("identity") or {}),
            )
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
            stock_stage = _audit_stock_stage(
                service,
                acceptance=resolved_acceptance,
                config=active,
                catalog_builder=stock_catalog_builder,
                inventory_builder=inventory_snapshot_builder,
            )
            stages.append(_stage("replan_stock", stock_stage["status"], stock_stage))

    _mark_stage_running(
        checkpoint_path,
        identity,
        stages,
        outcomes,
        "closeout",
    )
    closeout = service.closeout(
        idempotency_key=f"solve-target:closeout:{service.kernel.state.graph_revision}"
    )
    stages.append(
        _stage(
            "closeout",
            "completed",
            {
                "portfolio_accepted": closeout["portfolio"].get("accepted") is True,
                "selected_route_count": len(
                    closeout["portfolio"].get("selected_routes") or []
                ),
            },
        )
    )
    gates = compile_blind_acceptance_report(
        preflight=preflight,
        director_outcomes=outcomes,
        graph=service.graph_store.load(),
        portfolio=closeout["portfolio"],
    )
    resource_envelope = _resource_envelope(
        model_cost=service.kernel.state.model_totals,
        attempt_count=service.kernel.state.attempt_count,
        accepted_expansion_count=service.kernel.state.accepted_expansion_count,
        budget=resolved_budget,
    )
    stop_preview = service.kernel.decide_stop().to_dict()
    profile_projection = service.workbench()["snapshot"]
    claim = _claim(
        gates,
        resolved_acceptance,
        resource_envelope,
        workbench=profile_projection,
    )
    planning_depth = _planning_depth_requirement(
        outcomes,
        minimum_steps=active.minimum_planning_route_steps,
    )
    current_disposition = _current_disposition(
        kernel_status=service.kernel.state.status,
        stop_decision=stop_preview,
        claim=claim,
        gates=gates,
    )
    workbench = service.publish_workbench(
        campaign_summary=_workbench_campaign_summary(
            gates=gates,
            resource_envelope=resource_envelope,
            model_cost=service.kernel.state.model_totals,
            stop_decision=stop_preview,
            claim=claim,
            current_disposition=current_disposition,
            planning_depth=planning_depth,
        )
    )
    stop = service.kernel.apply_stop_decision(
        idempotency_key=f"solve-target:stop:{service.kernel.state.revision}"
    ).to_dict()
    report = {
        "schema_version": TARGET_SOLVE_REPORT_SCHEMA,
        "run_id": identity,
        "run_dir": str(directory),
        "target": {"name": case.target_name, "canonical_smiles": canonical},
        "preflight": preflight,
        "config": asdict(active),
        "acceptance": resolved_acceptance.to_dict(),
        "budget": resolved_budget.to_dict(),
        "director_outcomes": outcomes,
        "stages": _deduplicate_stages(stages),
        "gates": gates,
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
    report["content_sha256"] = _digest(report)
    report_artifact = service.kernel.artifacts.put_json(
        report,
        logical_name="target_only_solve_report.json",
        producer="autoplanner.target_solver",
    )
    report_ref = report_artifact.to_dict()
    service.kernel.index.index_artifact(
        run_id=identity,
        artifact_id="target_only_solve_report",
        ref=report_artifact,
        revision=service.kernel.state.graph_revision,
        authority_scope="benchmark_measurement_only",
    )
    report_path = directory / "target-only-solve-report.json"
    _write_json_atomic(report_path, report)
    _checkpoint(checkpoint_path, identity, report["stages"], outcomes, complete=True)
    return {**report, "report_ref": report_ref, "report_path": str(report_path)}


def _acceptance_input(value: RetrosynthesisAcceptanceSpec) -> dict[str, Any]:
    return {
        "minimum_complete_routes": value.minimum_complete_routes,
        "minimum_edge_proof_level": value.minimum_edge_proof_level,
        "minimum_independent_source_groups": value.minimum_independent_source_groups,
        "stock_boundary": value.stock_boundary,
    }


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
        attempt_count=service.kernel.state.attempt_count,
        accepted_expansion_count=service.kernel.state.accepted_expansion_count,
        budget=budget,
    )
    stop_decision = service.kernel.decide_stop().to_dict()
    profile_projection = service.workbench()["snapshot"]
    claim = _claim(
        gates,
        acceptance,
        resource_envelope,
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
    workbench = service.publish_workbench(
        campaign_summary=_workbench_campaign_summary(
            gates=gates,
            resource_envelope=resource_envelope,
            model_cost=service.kernel.state.model_totals,
            stop_decision=stop_decision,
            claim=claim,
            current_disposition=current_disposition,
            planning_depth=planning_depth,
        )
    )
    report_path = directory / "target-only-solve-report.json"
    previous = (
        _read_json_object(report_path, "target_solve_report_missing")
        if report_path.is_file()
        else {}
    )
    report = {
        **previous,
        "schema_version": TARGET_SOLVE_REPORT_SCHEMA,
        "run_id": identity,
        "run_dir": str(directory),
        "target": {"name": case.target_name, "canonical_smiles": canonical},
        "preflight": dict(preflight),
        "config": asdict(config),
        "acceptance": acceptance.to_dict(),
        "budget": budget.to_dict(),
        "director_outcomes": outcomes,
        "stages": _deduplicate_stages(stages),
        "gates": gates,
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
                current_disposition["state"]
                == "terminal_snapshot_requires_revalidation"
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
            "max_prompt_context_bytes",
        }
    }


def _audit_stock_stage(
    service: Any,
    *,
    acceptance: RetrosynthesisAcceptanceSpec,
    config: TargetSolveConfig,
    catalog_builder: StockCatalogBuilder | None,
    inventory_builder: InventorySnapshotBuilder | None,
) -> dict[str, Any]:
    if config.enable_live_benchmark_stock and acceptance.stock_boundary == (
        "benchmark_search"
    ):
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
) -> dict[str, Any]:
    generic = _target_name_requires_identity_resolution(target_name)
    prior = next(
        (
            dict(row.get("detail") or {})
            for row in reversed(list(stages))
            if row.get("stage") == "target_identity"
        ),
        {},
    )
    stale_opaque_skip = (
        generic
        and prior.get("status") == "not_needed"
        and re.fullmatch(
            r"target-[0-9a-f]{8}",
            str(dict(prior.get("identity") or {}).get("preferred_name") or "").lower(),
        )
        is not None
    )
    stale_provider_version = (
        generic
        and prior.get("status") == "completed"
        and prior.get("provider_id") == "pubchem.pug_rest"
        and prior.get("provider_version") != TARGET_IDENTITY_PROVIDER_VERSION
    )
    if prior and not stale_opaque_skip and not stale_provider_version:
        return prior
    if not enabled:
        return {
            "stage": "target_identity",
            "status": "disabled",
            "reason": "target_identity_disabled",
        }
    if not generic and not resolve_named:
        return {
            "stage": "target_identity",
            "status": "not_needed",
            "identity": {"preferred_name": target_name},
            "semantics": {"user_supplied_name_not_treated_as_evidence": True},
        }
    result = resolve_target_identity(target_smiles)
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
    return normalized in {
        "blind target",
        "target",
        "unknown target",
        "smiles-only target",
    } or re.fullmatch(r"target-[0-9a-f]{8}", normalized) is not None


def _latest_stage(
    stages: Iterable[Mapping[str, Any]],
    stage_name: str,
) -> dict[str, Any]:
    return next(
        (
            dict(row)
            for row in reversed(list(stages))
            if row.get("stage") == stage_name
        ),
        {},
    )


def _attempted_chemenzy_frontiers(
    stages: Iterable[Mapping[str, Any]],
) -> set[str]:
    return {
        str(value)
        for row in stages
        if row.get("stage")
        in {"chemenzy_guided_frontier", "chemenzy_stock_recovery"}
        for value in dict(row.get("detail") or {}).get("frontier_smiles") or []
        if str(value)
    }


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
    queued = sum(
        row["disposition"] == "queued_on_canonical_frontier" for row in requests
    )
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
) -> dict[str, Any]:
    if connector is None:
        return {
            "stage": "evidence_acquisition",
            "status": "unresolved",
            "reason": "structured_evidence_connector_not_configured",
            "model_invocations": 0,
        }
    request = compile_evidence_acquisition_request(
        run_id=service.kernel.spec.run_id,
        target_name=target_name or service.kernel.spec.target_name,
        target_smiles=service.kernel.spec.target_smiles,
        graph=service.graph_store.load(),
        source_frontier=source_stage,
        target_identity=target_identity,
    )
    visual_stage: dict[str, Any] = {}
    source_route_stage: dict[str, Any] = {}
    try:
        acquired = acquire_structured_evidence(request, connector=connector)
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
        visual_stage = acquire_visual_evidence_candidates(
            service,
            evidence_request=request,
            discovery=discovery,
            provider=visual_provider,
            max_pages=max_visual_pages,
        )
        visual_materialization = materialize_visual_evidence_candidates(
            service,
            observation=dict(visual_stage.get("observation") or {}),
        )
        visual_validation: dict[str, Any] = {}
        if dict(visual_materialization.get("execution") or {}).get("changed") is True:
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
            source_route_validation: dict[str, Any] = {}
            if dict(source_route_stage.get("execution") or {}).get("changed") is True:
                source_route_validation = validate_materialized_edges(
                    service,
                    atom_mapper=atom_mapper,
                )
            source_route_stage = {
                **source_route_stage,
                "validation": source_route_validation,
            }
            return {
                "stage": "evidence_acquisition",
                "status": "discovered_unbound",
                "request_sha256": request["content_sha256"],
                "receipt_ref": receipt_ref,
                "discovery_ref": discovery_ref,
                "discovery": discovery,
                "discovery_ingestion": discovery_ingestion,
                "source_route": source_route_stage,
                "visual_evidence": visual_stage,
                "source_count": len(discovery.get("sources") or []),
                "exact_record_count": 0,
                "model_invocations": int(
                    visual_stage.get("model_invocations") or 0
                ),
                "visual_invocations": int(
                    visual_stage.get("visual_invocations") or 0
                ),
                "false_evidence_claim": False,
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
                    "discovery_may_inform_bounded_global_replan": True,
                    "connector_cannot_grant_reaction_validation": True,
                },
            }
        imported = ingest_structured_evidence_document(
            service,
            document=dict(document),
            atom_mapper=atom_mapper,
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
                str(value)
                for value in shared_validation.get("accepted_edge_ids") or []
            )
        )
        rejected_source_ids = sorted(
            source_edge_ids.intersection(
                str(value)
                for value in shared_validation.get("rejected_edge_ids") or []
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
                    for value in shared_validation.get("rejection_diagnostics")
                    or []
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
            "model_invocations": int(visual_stage.get("model_invocations") or 0),
            "visual_invocations": int(visual_stage.get("visual_invocations") or 0),
            "false_evidence_claim": False,
        }
    return {
        "stage": "evidence_acquisition",
        "status": (
            "completed" if int(imported.get("exact_record_count") or 0) else "partial"
        ),
        "request_sha256": request["content_sha256"],
        "document_sha256": acquired["document_sha256"],
        "receipt_ref": receipt_ref,
        "discovery_ref": discovery_ref,
        "discovery": discovery,
        "discovery_ingestion": discovery_ingestion,
        "source_route": source_route_stage,
        "visual_evidence": visual_stage,
        "source_count": imported["source_count"],
        "exact_record_count": imported["exact_record_count"],
        "source_binding_count": imported["source_binding_count"],
        "execution": imported["execution"],
        "validation": imported["validation"],
        "model_invocations": int(visual_stage.get("model_invocations") or 0),
        "visual_invocations": int(visual_stage.get("visual_invocations") or 0),
        "material_events": sorted(
            {
                *(
                    ["source_material_discovered"]
                    if discovery
                    else []
                ),
                *(
                    str(value)
                    for value in visual_stage.get("material_events") or []
                    if str(value)
                ),
                *(
                    str(value)
                    for value in source_route_stage.get("material_events") or []
                    if str(value)
                ),
                *(
                    ["exact_rows_added"]
                    if int(imported.get("exact_record_count") or 0)
                    else []
                ),
            }
        ),
        "semantics": {
            "connector_output_requires_normal_host_ingestion": True,
            "connector_cannot_grant_reaction_validation": True,
            "receipt_grants_no_scientific_authority": True,
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


def _resolve_evidence_prefetch(
    future: Future[dict[str, Any]] | None,
    executor: ThreadPoolExecutor | None,
) -> dict[str, Any]:
    if future is None:
        return {
            "status": "not_started",
            "elapsed_s": 0.0,
            "discovery": {},
        }
    try:
        return dict(future.result(timeout=180.0))
    except Exception as exc:  # bounded optional latency-hiding optimization
        return {
            "status": "unresolved",
            "elapsed_s": 0.0,
            "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
            "discovery": {},
            "semantics": {"failure_does_not_abort_campaign": True},
        }
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


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
        key: value
        for key, value in prefetch.items()
        if key not in {"discovery", "receipt"}
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
    remaining = {
        key: max(0.0, float(limits[key]) - float(observed[key]))
        for key in limits
    }
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
    events = {"portfolio_stagnation"}
    for stage in stages:
        events.update(
            str(value)
            for value in stage.get("material_events") or []
            if str(value).strip()
        )
        execution = dict(stage.get("execution") or {})
        events.update(
            str(value)
            for value in execution.get("material_events") or []
            if str(value).strip()
        )
        if int(stage.get("rejected_validation_count") or 0) > 0:
            events.add("critical_edge_rejected")
    return tuple(sorted(events))


def _director_topology_replan_events(
    outcomes: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Expose malformed multi-step skeletons to one bounded global replan."""

    for outcome in outcomes:
        outcome_reasons = {
            str(value)
            for value in outcome.get("reasons") or []
            if str(value).strip()
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
            reasons = {
                str(value)
                for value in audit.get("reasons") or []
                if str(value).strip()
            }
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
            audits_by_skeleton.setdefault(str(audit.get("skeleton_id") or ""), []).append(
                audit
            )
        for skeleton in plan.get("multi_step_skeletons") or []:
            if not isinstance(skeleton, Mapping):
                continue
            skeleton_id = str(skeleton.get("skeleton_id") or "")
            steps = [
                step
                for step in skeleton.get("steps") or []
                if isinstance(step, Mapping)
            ]
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
        reasons = {
            str(value)
            for value in outcome.get("reasons") or []
            if str(value).strip()
        }
        if (
            outcome.get("status") == "failed"
            and "GlobalCampaignPlanValidationError" in reasons
        ):
            return True
    return False


def _replan_reasons(
    gates: Mapping[str, Any],
    *,
    material_events: tuple[str, ...],
) -> tuple[str, ...]:
    values = dict(gates.get("gates") or {})
    events = set(material_events)
    reasons: list[str] = []
    if "director_contract_rejected" in events:
        reasons.append("director_contract_deficit")
    if "director_depth_deficit" in events:
        reasons.append("planning_depth_deficit")
    if "director_topology_rejected" in events:
        reasons.append("director_topology_deficit")
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
        dict(value)
        for value in discovery.get("sources") or []
        if isinstance(value, Mapping)
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
            "sources": [
                _source_replan_observation(value) for value in selected_sources
            ],
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
    route = _source_route_replan_observation(
        dict(row.get("source_route_observation") or {})
    )
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
    excerpt = str(
        row.get("procedure_excerpt")
        or row.get("procedure")
        or row.get("text")
        or ""
    )
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
    steps = [
        dict(item)
        for item in row.get("candidate_steps") or []
        if isinstance(item, Mapping)
    ]
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
    proposals = [
        dict(row)
        for row in value.get("proposals") or []
        if isinstance(row, Mapping)
    ]
    diagnostics = [
        dict(row)
        for row in value.get("diagnostics") or []
        if isinstance(row, Mapping)
    ]
    return {
        "schema_version": str(value.get("schema_version") or ""),
        "source_ref": str(value.get("source_ref") or ""),
        "route_family": dict(value.get("route_family") or {}),
        "proposal_count": int(value.get("proposal_count") or len(proposals)),
        "resolved_procedure_count": int(
            value.get("resolved_procedure_count") or 0
        ),
        "unconnected_proposal_count": int(
            value.get("unconnected_proposal_count") or 0
        ),
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


def _claim(
    gates: Mapping[str, Any],
    acceptance: RetrosynthesisAcceptanceSpec,
    resource_envelope: Mapping[str, Any],
    *,
    workbench: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = dict(gates.get("gates") or {})
    accepted = bool(
        values.get("B5_configured_portfolio_acceptance") is True
        and resource_envelope.get("within_budget") is True
    )
    acceptance_profile = {
        "benchmark_search": "exploration_closed",
        "procurement": "procurement_closed",
        "in_house": "in_house_closed",
    }.get(acceptance.stock_boundary, "configured_boundary_closed")
    workbench_portfolio = dict(dict(workbench or {}).get("portfolio") or {})
    profile_counts = {
        str(key): int(value or 0)
        for key, value in dict(
            workbench_portfolio.get("acceptance_profile_counts") or {}
        ).items()
    }
    achieved_profile = str(
        workbench_portfolio.get("achieved_profile") or "unresolved"
    )
    return {
        "generated_route_portfolio": values.get("B1_global_multi_route") is True,
        "host_validated_route_portfolio": (
            values.get("B2_host_validated_routes") is True
        ),
        "exact_multi_source_grade": values.get("B3_exact_multi_source") is True,
        "configured_stock_boundary_closed": values.get("B4_stock_boundary") is True,
        "accepted_under_configured_policy": accepted,
        "acceptance_profile": acceptance_profile,
        "achieved_profile": achieved_profile,
        "product_profile_counts": profile_counts,
        "literature_grounded": profile_counts.get("literature_grounded", 0) > 0,
        "procurement_ready": bool(
            profile_counts.get("procurement_closed", 0) > 0
        ),
        "within_resource_budget": resource_envelope.get("within_budget") is True,
        "condition_complete": profile_counts.get("condition_complete", 0) > 0,
        "process_ready": profile_counts.get("process_ready", 0) > 0,
        "no_unqualified_solved_claim": True,
        "no_unqualified_complete_claim": True,
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
) -> dict[str, Any]:
    return {
        "gates": dict(gates.get("gates") or {}),
        "highest_contiguous_gate": str(
            gates.get("highest_contiguous_gate") or "none"
        ),
        "counts": dict(gates.get("counts") or {}),
        "resource_envelope": dict(resource_envelope),
        "model_cost": dict(model_cost),
        "stop_decision": dict(stop_decision),
        "claim": dict(claim),
        "current_disposition": dict(current_disposition),
        "planning_depth": dict(planning_depth or {}),
    }


def _current_disposition(
    *,
    kernel_status: str,
    stop_decision: Mapping[str, Any],
    claim: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = claim.get("accepted_under_configured_policy") is True
    historical_completion = bool(
        str(kernel_status) == "completed"
        or stop_decision.get("decision") == "completed"
    )
    proof_audit = dict(gates.get("reaction_proof_version_audit") or {})
    stale_terminal = historical_completion and not accepted
    if accepted:
        state = "accepted"
        reasons: list[str] = []
    elif stale_terminal:
        state = "terminal_snapshot_requires_revalidation"
        reasons = ["current_proof_policy_does_not_accept_historical_terminal_snapshot"]
        if proof_audit.get("requires_revalidation") is True:
            reasons.append("stale_reaction_validator_proofs_present")
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
        "scientifically_accepted": accepted,
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
    attempt_count: int,
    accepted_expansion_count: int,
    budget: RetrosynthesisRunBudget,
) -> dict[str, Any]:
    observed = {
        "model_invocations": int(model_cost.get("model_invocations") or 0),
        "input_tokens": int(model_cost.get("input_tokens") or 0),
        "output_tokens": int(model_cost.get("output_tokens") or 0),
        "model_wall_time_s": float(model_cost.get("wall_time_s") or 0.0),
        "visual_invocations": int(model_cost.get("visual_invocations") or 0),
        "attempt_runs": int(attempt_count),
        "accepted_expansions": int(accepted_expansion_count),
    }
    limits = {
        "model_invocations": budget.max_model_invocations,
        "input_tokens": budget.max_total_input_tokens,
        "output_tokens": budget.max_total_output_tokens,
        "model_wall_time_s": budget.max_total_wall_time_s,
        "visual_invocations": budget.max_visual_invocations,
        "attempt_runs": budget.max_attempt_runs,
        "accepted_expansions": budget.max_accepted_expansions,
    }
    violations = sorted(
        f"{key}_budget_violated"
        for key, value in observed.items()
        if value > limits[key]
    )
    return {
        "schema_version": "target_solve_resource_envelope.v1",
        "within_budget": not violations,
        "observed": observed,
        "limits": limits,
        "violations": violations,
        "semantics": {
            "reaching_a_cap_is_compliant": True,
            "observed_overrun_blocks_qualified_acceptance": True,
        },
    }


def _stage(name: str, status: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    now = _utc_now()
    row = {
        "stage": name,
        "status": str(status),
        "detail": _bounded_detail(dict(detail or {})),
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
        "host_admitted_route_count": int(
            detail.get("host_admitted_route_count") or 0
        ),
        "selected_proposal_route_count": int(
            detail.get("selected_proposal_route_count") or 0
        ),
        "proposal_count": int(detail.get("proposal_count") or 0),
        "route_admission": list(detail.get("route_admission") or []),
        "semantics": {
            "provider_result_is_proposal_only": True,
            "topology_contains_exact_seed_steps": True,
            "director_should_reason_over_seed_as_a_global_route": True,
        },
    }


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
    evidence_observations: Mapping[str, Any]
    | tuple[Mapping[str, Any], ...]
    | None = None,
    idempotency_key: str,
) -> dict[str, Any]:
    """Turn a bounded provider failure into an auditable unresolved outcome."""

    try:
        return service.run_global_director(
            mode=mode,
            material_events=material_events,
            evidence_observations=evidence_observations,
            idempotency_key=idempotency_key,
        ).to_dict()
    except GlobalCampaignDirectorError as exc:
        reason = str(exc).strip() or type(exc).__name__
        return DirectorOutcome(
            status="failed",
            invoked=True,
            cache_hit=False,
            mode=mode,
            context_sha256="",
            reasons=(type(exc).__name__, reason[:2_000]),
        ).to_dict()


def _empty_checkpoint(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": TARGET_SOLVE_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "complete": False,
        "stages": [],
        "director_outcomes": [],
    }


def _checkpoint(
    path: Path,
    run_id: str,
    stages: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    complete: bool = False,
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": TARGET_SOLVE_CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "complete": complete,
            "stages": _deduplicate_stages(stages),
            "director_outcomes": outcomes,
        },
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
