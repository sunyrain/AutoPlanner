"""Validation/evidence forks for scientifically stale target campaigns.

A completed kernel is immutable even when the host reaction verifier changes.
This module therefore replays the original, host-admitted global Codex plans
into a new run and executes validation, evidence, and stock stages.  The default
fork is model-free.  Callers may explicitly admit one sparse page-vision task;
that task can only create L0 candidates and never replans the route.  The
derived run is cryptographically bound to its source report and graph and never
presents itself as a fresh blind generation campaign.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from cascade_planner.application.blind_acceptance import (
    compile_blind_acceptance_report,
)
from cascade_planner.application.campaign_quality_state import (
    compile_campaign_quality_state,
)
from cascade_planner.application.reaction_mapping import ReactionMapper
from cascade_planner.application.reaction_proof_versions import (
    CURRENT_REACTION_VALIDATOR_VERSION,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.unified_campaign_spec import UnifiedCampaignSpec
from cascade_planner.interfaces.live_evidence import EvidenceConnector
from cascade_planner.interfaces.chemenzy_probe import (
    ChemenzyProposalProvider,
    run_chemenzy_guided_frontier_stage,
)
from cascade_planner.interfaces.patent_self_evolution import (
    PatentSelfEvolutionSession,
)
from cascade_planner.interfaces.target_solver import (
    TargetSolveConfig,
    _acquire_evidence_stage,
    _audit_stock_stage,
    _chemenzy_vendor_root,
    _claim,
    _current_disposition,
    _resource_envelope,
    _workbench_campaign_summary,
)
from cascade_planner.interfaces.target_solver_stages import (
    InventorySnapshotBuilder,
    StockCatalogBuilder,
    discover_director_source_hints,
    repair_rejected_precursor_typos,
    validate_materialized_edges,
)
from cascade_planner.interfaces.visual_evidence import VisualEvidenceProvider


TARGET_VALIDATION_FORK_REPORT_SCHEMA = "target_validation_fork_report.v1"
TARGET_VALIDATION_FORK_LINEAGE_SCHEMA = "target_validation_fork_lineage.v1"


class TargetValidationForkError(RuntimeError):
    """A source campaign cannot be replayed as a bounded validation fork."""


@dataclass(frozen=True, slots=True)
class ValidationForkConfig:
    max_atom_mapping_reactions: int = 64
    max_live_stock_molecules: int = 32
    enable_live_benchmark_stock: bool = True
    enable_patent_self_evolution: bool = True
    self_evo_library_path: str = ""
    max_self_evo_template_candidates: int = 12
    max_visual_invocations: int = 0
    max_visual_evidence_pages: int = 2
    enable_guided_chemenzy: bool = False
    chemenzy_env_prefix: str = ""
    chemenzy_stock_names: tuple[str, ...] = ()
    chemenzy_stock_paths: tuple[tuple[str, str], ...] = ()
    max_guided_chemenzy_frontiers: int = 4
    max_guided_chemenzy_routes: int = 1
    max_guided_chemenzy_steps: int = 6
    max_guided_chemenzy_iterations: int = 500
    guided_chemenzy_expansion_topk: int = 20
    guided_chemenzy_timeout_s: float = 1_200.0
    chemenzy_seed: int = 0
    schema_version: str = "target_validation_fork_config.v1"

    def __post_init__(self) -> None:
        if self.max_atom_mapping_reactions < 1:
            raise ValueError("validation_fork_mapping_limit_invalid")
        if self.max_live_stock_molecules < 1:
            raise ValueError("validation_fork_stock_limit_invalid")
        if not 1 <= self.max_self_evo_template_candidates <= 64:
            raise ValueError("validation_fork_self_evo_candidate_limit_invalid")
        if self.max_visual_invocations not in {0, 1}:
            raise ValueError("validation_fork_visual_invocation_limit_invalid")
        if not 1 <= self.max_visual_evidence_pages <= 12:
            raise ValueError("validation_fork_visual_page_limit_invalid")
        integer_limits = {
            "frontiers": self.max_guided_chemenzy_frontiers,
            "routes": self.max_guided_chemenzy_routes,
            "steps": self.max_guided_chemenzy_steps,
            "iterations": self.max_guided_chemenzy_iterations,
            "expansion_topk": self.guided_chemenzy_expansion_topk,
        }
        if any(int(value) < 1 for value in integer_limits.values()):
            raise ValueError("validation_fork_chemenzy_limit_invalid")
        if self.guided_chemenzy_timeout_s < 1.0:
            raise ValueError("validation_fork_chemenzy_timeout_invalid")


def fork_target_validation(
    gateway: Any,
    *,
    source_run_id: str,
    source_run_dir: str | Path | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    config: ValidationForkConfig | None = None,
    atom_mapper: ReactionMapper | None = None,
    stock_catalog_builder: StockCatalogBuilder | None = None,
    inventory_snapshot_builder: InventorySnapshotBuilder | None = None,
    evidence_connector: EvidenceConnector | None = None,
    visual_evidence_provider: VisualEvidenceProvider | None = None,
    chemenzy_provider: ChemenzyProposalProvider | None = None,
) -> dict[str, Any]:
    """Replay one source campaign without another route-planning model call."""

    active = config or ValidationForkConfig()
    source_identity = gateway._normalize_run_id(source_run_id)
    source_directory = gateway._run_dir(
        source_identity,
        explicit=source_run_dir,
        require=True,
    )
    source_service = gateway._open(source_identity, run_dir=source_directory)
    source_report_path = source_directory / "target-only-solve-report.json"
    source_report = _read_bound_source_report(
        source_report_path,
        expected_run_id=source_identity,
    )
    target = dict(source_report.get("target") or {})
    target_smiles = str(target.get("canonical_smiles") or "")
    target_name = str(target.get("name") or source_service.kernel.spec.target_name)
    if (
        target_smiles != source_service.kernel.spec.target_smiles
        or not target_name
    ):
        raise TargetValidationForkError("validation_fork_source_target_mismatch")
    preflight = dict(source_report.get("preflight") or {})
    if preflight.get("accepted") is not True:
        raise TargetValidationForkError("validation_fork_source_not_blind_accepted")
    outcomes = [
        dict(row)
        for row in source_report.get("director_outcomes") or []
        if isinstance(row, Mapping)
    ]
    accepted_outcomes = [
        row
        for row in outcomes
        if row.get("status") == "accepted" and isinstance(row.get("plan"), Mapping)
    ]
    if not accepted_outcomes:
        raise TargetValidationForkError("validation_fork_source_has_no_accepted_plan")

    acceptance = _acceptance_from_report(source_report)
    derived_budget = _derived_budget(
        source_report,
        max_visual_invocations=active.max_visual_invocations,
        max_guided_chemenzy_frontiers=(
            active.max_guided_chemenzy_frontiers
            if active.enable_guided_chemenzy
            else 0
        ),
    )
    source_campaign_spec = source_service.kernel.spec.campaign_spec
    if source_campaign_spec is None:
        raise TargetValidationForkError("validation_fork_source_campaign_spec_missing")
    derived_campaign_spec = UnifiedCampaignSpec(
        target_smiles=target_smiles,
        stock_oracle=source_campaign_spec.stock_oracle,
        constraints=source_campaign_spec.constraints,
        resource_budget=replace(
            source_campaign_spec.resource_budget,
            model=derived_budget,
            max_run_wall_time_s=max(
                source_campaign_spec.resource_budget.max_run_wall_time_s,
                (
                    active.max_guided_chemenzy_frontiers
                    * active.guided_chemenzy_timeout_s
                    + 300.0
                    if active.enable_guided_chemenzy
                    else 0.0
                ),
            ),
        ),
    )
    identity = gateway._normalize_run_id(
        run_id or gateway._new_run_id(f"{target_name}-validation", target_smiles)
    )
    directory = gateway._run_dir(identity, explicit=run_dir, require=False)
    if (directory / ".autoplanner" / "kernel" / "run_spec.json").is_file():
        raise TargetValidationForkError("validation_fork_run_exists")

    source_graph = source_service.graph_store.load()
    lineage = {
        "schema_version": TARGET_VALIDATION_FORK_LINEAGE_SCHEMA,
        "source_run_id": source_identity,
        "source_report_sha256": str(source_report["content_sha256"]),
        "source_graph_scientific_sha256": str(
            source_graph.get("scientific_sha256") or ""
        ),
        "source_graph_revision": int(source_graph.get("revision") or 0),
        "source_campaign_spec_sha256": source_campaign_spec.to_dict()[
            "content_sha256"
        ],
        "source_model_cost": dict(source_report.get("model_cost") or {}),
        "replayed_plan_sha256": sorted(
            _digest(dict(row["plan"])) for row in accepted_outcomes
        ),
        "reaction_validator_version": CURRENT_REACTION_VALIDATOR_VERSION,
        "semantics": {
            "source_blind_preflight_is_inherited_by_digest": True,
            "derived_run_is_not_a_fresh_blind_generation": True,
            "source_kernel_history_is_immutable": True,
            "route_planning_model_calls_allowed_in_derived_run": 0,
            "visual_candidate_calls_allowed_in_derived_run": (
                active.max_visual_invocations
            ),
        },
    }
    lineage["content_sha256"] = _digest(lineage)

    gateway.create_run(
        target_name=target_name,
        target_smiles=target_smiles,
        run_id=identity,
        run_dir=directory,
        acceptance=acceptance,
        budget=derived_budget,
        campaign_spec=derived_campaign_spec,
    )
    service = gateway._open(identity, run_dir=directory)
    self_evo = PatentSelfEvolutionSession.create(
        enabled=active.enable_patent_self_evolution,
        configured_path=active.self_evo_library_path,
        external_data_root=gateway.paths.external_data_root,
        target_smiles=target_smiles,
        max_candidates=active.max_self_evo_template_candidates,
    )
    lineage_ref = service.kernel.artifacts.put_json(
        lineage,
        logical_name="target_validation_fork_lineage.json",
        producer="autoplanner.validation_fork",
    )
    _write_json_atomic(
        directory / ".autoplanner" / "validation-fork-lineage.json",
        lineage,
    )

    stages: list[dict[str, Any]] = []
    for index, outcome in enumerate(accepted_outcomes, start=1):
        plan = dict(outcome["plan"])
        admitted_ids = sorted(
            str(row.get("proposal_id") or "")
            for row in outcome.get("proposal_audits") or []
            if isinstance(row, Mapping)
            and row.get("accepted") is True
            and str(row.get("proposal_id") or "")
        )
        if not admitted_ids:
            raise TargetValidationForkError(
                f"validation_fork_admitted_proposals_missing:{index}"
            )
        replay_plan = {**plan, "_host_admitted_proposal_ids": admitted_ids}
        replay = service.apply_global_plan(
            replay_plan,
            idempotency_key=(
                f"validation-fork:plan:{index}:{_digest(replay_plan)[:24]}"
            ),
            proposal_origin_kind="codex_global_director",
            proposal_origin_ref=f"validation_fork:director_outcome:{index}",
        )
        stages.append(_stage("plan_replay", "completed", replay))

    materialization = service.execute_frontier_materialization(
        idempotency_key="validation-fork:materialization"
    )
    stages.append(
        _stage(
            "materialization",
            "completed" if materialization.get("changed") else "reused_or_empty",
            materialization,
        )
    )
    validation = validate_materialized_edges(
        service,
        atom_mapper=atom_mapper,
        max_reactions=active.max_atom_mapping_reactions,
    )
    stages.append(_stage("reaction_validation", validation["status"], validation))
    repair = repair_rejected_precursor_typos(service, validation)
    stages.append(_stage("precursor_repair", repair["status"], repair))
    if int(repair.get("accepted_repair_count") or 0) > 0:
        revalidation = validate_materialized_edges(
            service,
            atom_mapper=atom_mapper,
            max_reactions=active.max_atom_mapping_reactions,
        )
        stages.append(
            _stage("precursor_repair_validation", revalidation["status"], revalidation)
        )

    source_stage = discover_director_source_hints(service, outcomes)
    stages.append(_stage("source_frontier", source_stage["status"], source_stage))
    source_target_identity = _latest_source_target_identity(source_report)
    evidence_stage = _acquire_evidence_stage(
        service,
        source_stage=source_stage,
        connector=evidence_connector,
        atom_mapper=atom_mapper,
        visual_provider=(
            visual_evidence_provider
            if active.max_visual_invocations > 0
            else None
        ),
        max_visual_pages=active.max_visual_evidence_pages,
        target_name=target_name,
        target_identity=source_target_identity,
    )
    stages.append(
        _stage("evidence_acquisition", evidence_stage["status"], evidence_stage)
    )
    template_learning = self_evo.learn(service.graph_store.load())
    stages.append(
        _stage(
            "patent_template_learning",
            template_learning["status"],
            template_learning,
        )
    )
    stock_stage = _audit_stock_stage(
        service,
        acceptance=acceptance,
        config=TargetSolveConfig(
            enable_live_benchmark_stock=active.enable_live_benchmark_stock,
            max_atom_mapping_reactions=active.max_atom_mapping_reactions,
            max_live_stock_molecules=active.max_live_stock_molecules,
        ),
        catalog_builder=stock_catalog_builder,
        inventory_builder=inventory_snapshot_builder,
    )
    stages.append(_stage("stock", stock_stage["status"], stock_stage))

    if active.enable_guided_chemenzy:
        guided = _run_guided_chemenzy_tail(
            service,
            target_name=target_name,
            target_smiles=target_smiles,
            config=active,
            vendor_root=_chemenzy_vendor_root(gateway.paths.vendor_root),
            provider=chemenzy_provider,
        )
        stages.append(_stage("chemenzy_guided_frontier", guided["status"], guided))
        if int(guided.get("proposal_count") or 0) > 0:
            tail_materialization = service.execute_frontier_materialization(
                idempotency_key="validation-fork:guided-materialization"
            )
            stages.append(
                _stage(
                    "guided_materialization",
                    (
                        "completed"
                        if tail_materialization.get("changed")
                        else "reused_or_empty"
                    ),
                    tail_materialization,
                )
            )
            tail_validation = validate_materialized_edges(
                service,
                atom_mapper=atom_mapper,
                max_reactions=active.max_atom_mapping_reactions,
            )
            stages.append(
                _stage(
                    "guided_reaction_validation",
                    tail_validation["status"],
                    tail_validation,
                )
            )
            tail_stock = _audit_stock_stage(
                service,
                acceptance=acceptance,
                config=TargetSolveConfig(
                    enable_live_benchmark_stock=active.enable_live_benchmark_stock,
                    max_atom_mapping_reactions=active.max_atom_mapping_reactions,
                    max_live_stock_molecules=active.max_live_stock_molecules,
                ),
                catalog_builder=stock_catalog_builder,
                inventory_builder=inventory_snapshot_builder,
            )
            stages.append(_stage("guided_stock", tail_stock["status"], tail_stock))

    closeout = service.closeout(
        idempotency_key=f"validation-fork:closeout:{service.kernel.state.graph_revision}"
    )
    graph = service.graph_store.load()
    gates = compile_blind_acceptance_report(
        preflight=preflight,
        director_outcomes=outcomes,
        graph=graph,
        portfolio=closeout["portfolio"],
    )
    resource_envelope = _resource_envelope(
        model_cost=service.kernel.state.model_totals,
        native_search=service.kernel.native_search_budget(),
        task_budget=service.kernel.task_budget(),
        run_wall_time_s=service.kernel.state.task_wall_time_s,
        attempt_count=service.kernel.state.attempt_count,
        accepted_expansion_count=service.kernel.state.accepted_expansion_count,
        budget=derived_budget,
        max_run_wall_time_s=service.kernel.spec.limits.max_run_wall_time_s,
    )
    stop_preview = service.kernel.decide_stop().to_dict()
    profile_projection = service.workbench()["snapshot"]
    quality_state = compile_campaign_quality_state(
        workbench=profile_projection,
        gates=gates,
    )
    claim = _claim(gates, acceptance, resource_envelope)
    disposition = _current_disposition(
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
            current_disposition=disposition,
            quality_state=quality_state,
        )
    )
    stop = service.kernel.apply_stop_decision(
        idempotency_key=f"validation-fork:stop:{service.kernel.state.revision}"
    ).to_dict()
    report = {
        "schema_version": TARGET_VALIDATION_FORK_REPORT_SCHEMA,
        "run_id": identity,
        "run_dir": str(directory),
        "target": {"name": target_name, "canonical_smiles": target_smiles},
        "campaign_spec": service.kernel.spec.campaign_spec.to_dict(),
        "lineage": lineage,
        "lineage_ref": lineage_ref.to_dict(),
        "acceptance": acceptance.to_dict(),
        "budget": derived_budget.to_dict(),
        "director_outcomes_replayed": outcomes,
        "stages": stages,
        "gates": gates,
        "quality_state": quality_state,
        "model_cost": dict(service.kernel.state.model_totals),
        "resource_envelope": resource_envelope,
        "attempt_count": service.kernel.state.attempt_count,
        "accepted_expansion_count": service.kernel.state.accepted_expansion_count,
        "stop_decision": stop,
        "current_disposition": disposition,
        "self_evolution": self_evo.report(),
        "portfolio_ref": closeout["portfolio_ref"],
        "workbench_ref": workbench["snapshot_ref"],
        "claim": claim,
        "semantics": {
            "B0_refers_to_bound_source_campaign": True,
            "B1_refers_to_replayed_source_global_plan": True,
            "B2_through_B5_are_recomputed_in_derived_run": True,
            "derived_route_planning_model_invocation_count_must_equal_zero": True,
            "optional_visual_candidate_is_L0_only": True,
            "derived_visual_invocation_limit": active.max_visual_invocations,
            "guided_chemenzy_is_model_free_native_search": True,
            "guided_chemenzy_enabled": active.enable_guided_chemenzy,
        },
    }
    model_invocations = int(report["model_cost"].get("model_invocations") or 0)
    visual_invocations = int(report["model_cost"].get("visual_invocations") or 0)
    if (
        model_invocations != visual_invocations
        or visual_invocations > active.max_visual_invocations
    ):
        raise TargetValidationForkError("validation_fork_unadmitted_model_usage_detected")
    report["content_sha256"] = _digest(report)
    report_artifact = service.kernel.artifacts.put_json(
        report,
        logical_name="target_validation_fork_report.json",
        producer="autoplanner.validation_fork",
    )
    report_path = directory / "target-validation-fork-report.json"
    _write_json_atomic(report_path, report)
    return {
        **report,
        "report_ref": report_artifact.to_dict(),
        "report_path": str(report_path),
    }


def _run_guided_chemenzy_tail(
    service: Any,
    *,
    target_name: str,
    target_smiles: str,
    config: ValidationForkConfig,
    vendor_root: Path,
    provider: ChemenzyProposalProvider | None,
) -> dict[str, Any]:
    """Run one separately budgeted native-search pass over distinct open leaves."""

    graph = service.graph_store.load()
    available = [
        dict(row)
        for row in dict(graph.get("deficit_frontier") or {}).get("items") or []
        if isinstance(row, Mapping)
        and row.get("kind") == "expansion"
        and dict(row.get("metadata") or {}).get("target_level_native_search")
        is not True
        and str(dict(row.get("metadata") or {}).get("frontier_smiles") or "")
    ]
    frontier_smiles = tuple(
        dict.fromkeys(
            str(dict(row.get("metadata") or {}).get("frontier_smiles") or "")
            for row in available
        )
    )[: config.max_guided_chemenzy_frontiers]
    if not frontier_smiles:
        return {
            "schema_version": "validation_fork_guided_chemenzy_tail.v1",
            "status": "not_needed",
            "frontier_count": 0,
            "proposal_count": 0,
            "provider_invocation_count": 0,
        }

    task_id = f"validation-fork-guided-chemenzy:{_digest({'leaves': frontier_smiles})[:20]}"
    service.kernel.reserve_task(
        task_id=task_id,
        kind="proposal",
        idempotency_key=f"{task_id}:reserve",
        input_revision=service.kernel.state.graph_revision,
        resource_class="native_search_frontier",
        resource_units=len(frontier_smiles),
        metadata={
            "validation_fork_guided_tail": True,
            "frontier_smiles": list(frontier_smiles),
        },
    )
    started = time.monotonic()
    result: dict[str, Any]
    try:
        result = run_chemenzy_guided_frontier_stage(
            service,
            target_name=target_name,
            root_target_smiles=target_smiles,
            enabled=True,
            provider=provider,
            env_prefix=config.chemenzy_env_prefix or None,
            vendor_root=vendor_root,
            max_frontiers=len(frontier_smiles),
            max_routes=config.max_guided_chemenzy_routes,
            max_steps=config.max_guided_chemenzy_steps,
            max_iterations=config.max_guided_chemenzy_iterations,
            expansion_topk=config.guided_chemenzy_expansion_topk,
            timeout_s=config.guided_chemenzy_timeout_s,
            include_frontier_smiles=frontier_smiles,
            search_preset="thorough",
            random_seed=config.chemenzy_seed,
            stock_names=config.chemenzy_stock_names,
            stock_paths=dict(config.chemenzy_stock_paths),
            enable_condition_prediction=False,
            enable_enzyme_assignment=False,
            enable_enzyme_coverage_sidecar=False,
            pandarallel_workers=1,
            stop_on_first_host_admitted_route=True,
        )
    except Exception as exc:
        elapsed = round(max(0.0, time.monotonic() - started), 6)
        service.kernel.settle_task(
            task_id=task_id,
            idempotency_key=f"{task_id}:settle",
            status="failed",
            failure_reasons=(f"{type(exc).__name__}:{exc}",),
            elapsed_s=elapsed,
        )
        raise
    elapsed = round(max(0.0, time.monotonic() - started), 6)
    service.kernel.settle_task(
        task_id=task_id,
        idempotency_key=f"{task_id}:settle",
        status=str(result.get("status") or "completed"),
        output_sha256=_digest(result),
        elapsed_s=elapsed,
    )
    return {
        **result,
        "schema_version": "validation_fork_guided_chemenzy_tail.v1",
        "elapsed_s": elapsed,
        "configured_frontier_smiles": list(frontier_smiles),
        "native_search_units": len(frontier_smiles),
        "semantics": {
            **dict(result.get("semantics") or {}),
            "runs_after_codex_model_budget": True,
            "one_native_search_unit_per_distinct_open_leaf": True,
            "condition_and_enzyme_sidecars_deferred_until_route_exists": True,
        },
    }


def _acceptance_from_report(
    report: Mapping[str, Any],
) -> RetrosynthesisAcceptanceSpec:
    row = dict(report.get("acceptance") or {})
    keys = {
        "minimum_complete_routes",
        "minimum_edge_proof_level",
        "require_all_selected_leaves_stock_closed",
        "stock_boundary",
        "minimum_independent_source_groups",
        "require_distinct_edge_sets",
    }
    return RetrosynthesisAcceptanceSpec(
        **{key: value for key, value in row.items() if key in keys}
    )


def _derived_budget(
    report: Mapping[str, Any],
    *,
    max_visual_invocations: int,
    max_guided_chemenzy_frontiers: int = 0,
) -> RetrosynthesisRunBudget:
    source = dict(report.get("budget") or {})
    visual_enabled = max_visual_invocations > 0
    return RetrosynthesisRunBudget(
        max_model_invocations=max_visual_invocations,
        max_total_input_tokens=30_000 if visual_enabled else 0,
        max_total_output_tokens=6_000 if visual_enabled else 0,
        max_total_wall_time_s=360.0 if visual_enabled else 0.0,
        max_visual_invocations=max_visual_invocations,
        max_accepted_expansions=max(
            64,
            int(source.get("max_accepted_expansions") or 0),
        ),
        max_attempt_runs=max(96, int(source.get("max_attempt_runs") or 0)),
        max_native_search_invocations=max(
            int(max_guided_chemenzy_frontiers),
            int(source.get("max_native_search_invocations") or 0),
        ),
        min_target_native_search_invocations=0,
        max_frontier_native_search_invocations=max(
            int(max_guided_chemenzy_frontiers),
            int(source.get("max_native_search_invocations") or 0),
        ),
        max_prompt_context_bytes=96_000 if visual_enabled else 0,
    )


def _read_bound_source_report(path: Path, *, expected_run_id: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetValidationForkError("validation_fork_source_report_unreadable") from exc
    if not isinstance(value, Mapping):
        raise TargetValidationForkError("validation_fork_source_report_not_object")
    report = dict(value)
    if report.get("schema_version") != "target_only_retrosynthesis_solve_report.v1":
        raise TargetValidationForkError("validation_fork_source_report_schema_invalid")
    if str(report.get("run_id") or "") != expected_run_id:
        raise TargetValidationForkError("validation_fork_source_report_run_mismatch")
    supplied = str(report.get("content_sha256") or "")
    body = {key: child for key, child in report.items() if key != "content_sha256"}
    if not supplied or supplied != _digest(body):
        raise TargetValidationForkError("validation_fork_source_report_digest_invalid")
    return report


def _latest_source_target_identity(
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    for stage in reversed(list(source_report.get("stages") or [])):
        if not isinstance(stage, Mapping) or stage.get("stage") != "target_identity":
            continue
        identity = dict(dict(stage.get("detail") or {}).get("identity") or {})
        if identity:
            return identity
    return {}


def _stage(name: str, status: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": str(name),
        "status": str(status),
        "detail": _bounded_detail(detail),
    }


def _bounded_detail(value: Any, *, depth: int = 0) -> Any:
    if depth >= 7:
        return "<depth-limited>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name in {"graph", "portfolio", "snapshot"} and isinstance(
                child, Mapping
            ):
                result[f"{name}_summary"] = {
                    "revision": child.get("revision") or child.get("graph_revision"),
                    "edge_count": len(child.get("edges") or {}),
                    "content_sha256": child.get("content_sha256"),
                }
            else:
                result[name] = _bounded_detail(child, depth=depth + 1)
        return result
    if isinstance(value, list | tuple):
        rows = [_bounded_detail(child, depth=depth + 1) for child in value[:64]]
        if len(value) > 64:
            rows.append({"omitted_count": len(value) - 64})
        return rows
    if isinstance(value, str) and len(value) > 2_000:
        return value[:2_000] + f"... <{len(value) - 2_000} chars omitted>"
    return value


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "TARGET_VALIDATION_FORK_REPORT_SCHEMA",
    "TargetValidationForkError",
    "ValidationForkConfig",
    "fork_target_validation",
]
