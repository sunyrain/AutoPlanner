"""Target-only, bounded V4 retrosynthesis campaign orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, TYPE_CHECKING

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
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.reaction_mapping import ReactionMapper
from cascade_planner.application.proof_portfolio import compile_proof_portfolio
from cascade_planner.interfaces.target_solver_stages import (
    StockCatalogBuilder,
    audit_live_benchmark_stock,
    discover_director_source_hints,
    repair_rejected_precursor_typos,
    validate_materialized_edges,
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


@dataclass(frozen=True, slots=True)
class TargetSolveConfig:
    model: str = "gpt-5.5"
    reasoning_effort: str = "low"
    use_coordinator: bool = True
    enable_web_search: bool = True
    enable_replan: bool = True
    enable_live_benchmark_stock: bool = True
    max_atom_mapping_reactions: int = 48
    max_live_stock_molecules: int = 24
    max_director_output_tokens: int = 7_000
    max_director_wall_time_s: float = 360.0
    schema_version: str = "target_solve_config.v1"

    def __post_init__(self) -> None:
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("target solver reasoning effort is invalid")
        if self.max_atom_mapping_reactions < 1 or self.max_live_stock_molecules < 1:
            raise ValueError("target solver deterministic limits must be positive")
        if self.max_director_output_tokens < 1 or self.max_director_wall_time_s <= 0:
            raise ValueError("target solver director limits must be positive")


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
    identity = gateway._normalize_run_id(
        run_id or gateway._new_run_id(target_name, canonical)
    )
    directory = gateway._run_dir(identity, explicit=run_dir, require=False)
    case = BlindCase.from_dict(
        {
            "schema_version": BLIND_CASE_SCHEMA,
            "case_id": (
                re.sub(r"[^A-Za-z0-9._-]+", "-", identity).strip(".-")[:80]
                or f"blind-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
            ),
            "target_name": str(target_name or f"target-{hashlib.sha256(canonical.encode()).hexdigest()[:8]}"),
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
            repository_root=gateway.paths.repository_root,
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

    director_config = DirectorConfig(
        minimum_route_families=max(3, resolved_acceptance.minimum_complete_routes),
        max_route_families=4,
        max_skeletons=4,
        max_steps_per_skeleton=8,
        max_output_tokens=active.max_director_output_tokens,
        max_wall_time_s=active.max_director_wall_time_s,
        max_tool_calls=16,
        max_initial_architecture_calls=1,
        max_event_replan_calls=1,
        max_final_portfolio_synthesis_calls=1,
        model=active.model,
        reasoning_effort=active.reasoning_effort,
        enable_web_search=active.enable_web_search,
        use_coordinator=active.use_coordinator,
    )
    service = gateway._open(
        identity,
        run_dir=directory,
        director_runner=director_runner,
        director_config=director_config,
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
    if not outcomes:
        initial = _run_director_safely(
            service,
            mode="initial_architecture",
            idempotency_key="solve-target:director:initial",
        )
        outcomes.append(initial)
        stages.append(_stage("global_campaign", initial["status"]))
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

    # Evidence discovery and leaf stock audit deliberately precede the only
    # optional replan.  Their host-owned observations therefore enter the next
    # CampaignContext instead of making the director repeat a blind first pass.
    source_stage = discover_director_source_hints(service, outcomes)
    stages.append(_stage("source_frontier", source_stage["status"], source_stage))
    stock_stage = _audit_stock_stage(
        service,
        acceptance=resolved_acceptance,
        config=active,
        catalog_builder=stock_catalog_builder,
    )
    stages.append(_stage("stock", stock_stage["status"], stock_stage))

    provisional = service.closeout(
        idempotency_key=f"solve-target:provisional:{service.kernel.state.graph_revision}"
    )["portfolio"]
    provisional_gates = compile_blind_acceptance_report(
        preflight=preflight,
        director_outcomes=outcomes,
        graph=service.graph_store.load(),
        portfolio=provisional,
    )
    needs_replan = bool(
        active.enable_replan
        and provisional_gates["gates"]["B2_host_validated_routes"] is not True
        and any(
            outcome.get("status") == "accepted" and outcome.get("plan")
            for outcome in outcomes
        )
        and len(outcomes) < 2
    )
    material_events = _material_replan_events(
        materialization,
        validation,
        repair_stage,
        repair_validation,
        source_stage,
        stock_stage,
    )
    replan_prompt_context_bytes = 0
    if needs_replan:
        replan_context = service.compile_global_context(
            material_events=material_events,
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
    if needs_replan:
        stages.append(
            _stage(
                "global_replan_budget_gate",
                "accepted" if replan_guard["accepted"] else "skipped",
                replan_guard,
            )
        )
    if needs_replan and replan_guard["accepted"]:
        replan = _run_director_safely(
            service,
            mode="event_replan",
            material_events=material_events,
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
            stock_stage = _audit_stock_stage(
                service,
                acceptance=resolved_acceptance,
                config=active,
                catalog_builder=stock_catalog_builder,
            )
            stages.append(_stage("replan_stock", stock_stage["status"], stock_stage))

    closeout = service.closeout(
        idempotency_key=f"solve-target:closeout:{service.kernel.state.graph_revision}"
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
    claim = _claim(gates, resolved_acceptance, resource_envelope)
    workbench = service.publish_workbench(
        campaign_summary=_workbench_campaign_summary(
            gates=gates,
            resource_envelope=resource_envelope,
            model_cost=service.kernel.state.model_totals,
            stop_decision=stop_preview,
            claim=claim,
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
        "model_cost": dict(service.kernel.state.model_totals),
        "resource_envelope": resource_envelope,
        "attempt_count": service.kernel.state.attempt_count,
        "accepted_expansion_count": service.kernel.state.accepted_expansion_count,
        "stop_decision": stop,
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
    claim = _claim(gates, acceptance, resource_envelope)
    workbench = service.publish_workbench(
        campaign_summary=_workbench_campaign_summary(
            gates=gates,
            resource_envelope=resource_envelope,
            model_cost=service.kernel.state.model_totals,
            stop_decision=stop_decision,
            claim=claim,
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
        "model_cost": dict(service.kernel.state.model_totals),
        "resource_envelope": resource_envelope,
        "attempt_count": service.kernel.state.attempt_count,
        "accepted_expansion_count": service.kernel.state.accepted_expansion_count,
        "stop_decision": stop_decision,
        "workbench_ref": workbench["snapshot_ref"],
        "claim": claim,
        "report_refresh": {
            "model_invocations": 0,
            "terminal_state_unchanged": True,
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
) -> dict[str, Any]:
    if config.enable_live_benchmark_stock and acceptance.stock_boundary == (
        "benchmark_search"
    ):
        return audit_live_benchmark_stock(
            service,
            catalog_builder=catalog_builder,
            max_molecules=config.max_live_stock_molecules,
        )
    return {
        "stage": "stock",
        "status": "unresolved",
        "reason": "authoritative_stock_adapter_not_configured",
    }


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
        execution = dict(stage.get("execution") or {})
        events.update(
            str(value)
            for value in execution.get("material_events") or []
            if str(value).strip()
        )
        if int(stage.get("rejected_validation_count") or 0) > 0:
            events.add("critical_edge_rejected")
    return tuple(sorted(events))


def _claim(
    gates: Mapping[str, Any],
    acceptance: RetrosynthesisAcceptanceSpec,
    resource_envelope: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(gates.get("gates") or {})
    return {
        "generated_route_portfolio": values.get("B2_host_validated_routes") is True,
        "exact_multi_source_grade": values.get("B3_exact_multi_source") is True,
        "configured_stock_boundary_closed": values.get("B4_stock_boundary") is True,
        "accepted_under_configured_policy": (
            values.get("B5_configured_portfolio_acceptance") is True
            and resource_envelope.get("within_budget") is True
        ),
        "procurement_ready": bool(
            acceptance.stock_boundary == "procurement"
            and values.get("B5_configured_portfolio_acceptance") is True
            and resource_envelope.get("within_budget") is True
        ),
        "within_resource_budget": resource_envelope.get("within_budget") is True,
        "no_unqualified_solved_claim": True,
    }


def _workbench_campaign_summary(
    *,
    gates: Mapping[str, Any],
    resource_envelope: Mapping[str, Any],
    model_cost: Mapping[str, Any],
    stop_decision: Mapping[str, Any],
    claim: Mapping[str, Any],
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
    }


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
    return {
        "stage": name,
        "status": str(status),
        "detail": _bounded_detail(dict(detail or {})),
    }


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
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in values:
        key = (str(row.get("stage") or ""), _digest(row.get("detail") or {}))
        rows[key] = row
    return list(rows.values())


def _run_director_safely(
    service: Any,
    *,
    mode: str,
    material_events: tuple[str, ...] = (),
    idempotency_key: str,
) -> dict[str, Any]:
    """Turn a bounded provider failure into an auditable unresolved outcome."""

    try:
        return service.run_global_director(
            mode=mode,
            material_events=material_events,
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
            "stages": stages,
            "director_outcomes": outcomes,
        },
    )


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
    "TARGET_SOLVE_CHECKPOINT_SCHEMA",
    "TARGET_SOLVE_REPORT_SCHEMA",
    "TargetSolveConfig",
    "solve_target",
]
