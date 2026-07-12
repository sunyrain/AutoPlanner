"""Agentic blackboard controller entry point."""
from __future__ import annotations

import json
import hashlib
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerTask,
    run_codex_worker,
)
from cascade_planner.harness.agent_action_planner import validate_action_batch
from cascade_planner.harness.agentic_blackboard import (
    _drop_large_fields,
    _merge_source_candidate_rows,
    _refresh_source_lifecycle,
    build_agentic_guided_payload,
    complete_round,
    initialize_agent_blackboard,
    rank_analogical_hypotheses_from_blackboard,
    update_blackboard_from_action_batch,
    update_blackboard_from_action,
    update_budget_for_action,
)
from cascade_planner.harness.blackboard_events import (
    append_blackboard_checkpoint,
    begin_blackboard_action,
    blackboard_controller_single_writer,
    commit_prepared_blackboard_action,
    prepare_blackboard_action_result,
    replay_prepared_blackboard_action,
    rehydrate_blackboard_from_events,
)
from cascade_planner.harness.codex_action_planner import plan_action_batch_with_codex
from cascade_planner.harness.analogical_reaction_templates import (
    apply_analogical_templates_to_target,
    extract_analogical_reaction_templates_from_blackboard,
    rank_analogical_reaction_templates_from_blackboard,
    validate_template_applications_for_guided_search,
)
from cascade_planner.harness.failure_critic import compile_failure_critic_report
from cascade_planner.harness.hypothetical_retrosynthesis_report import (
    compile_hypothesis_only_retrosynthesis_report,
)
from cascade_planner.harness.hypothesis_execution_report import (
    compile_hypothesis_execution_report,
)
from cascade_planner.harness.codex_edge_verification import (
    project_edge_evidence_binding_sets,
    verify_codex_consensus_graph,
)
from cascade_planner.harness.local_pdf_proxy import (
    build_pdf_request,
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_request_queue_path,
    write_pdf_request_queue,
)
from cascade_planner.harness.parent_route_proof import (
    compile_stitched_parent_route_proof,
    is_solved_parent_route_proof,
)
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.runner import emit_final_verdict
from cascade_planner.harness.route_objectives import (
    build_broad_transform_templates_from_blackboard,
    classify_route_objectives,
    compile_route_objective_proof_bundle,
)
from cascade_planner.harness.route_verifier import is_accepted_route_verifier_report
from cascade_planner.harness.route_forest import write_route_forest_artifacts
from cascade_planner.harness.retrosynthetic_proposals import (
    compile_retrosynthetic_proposal_bus,
    recursive_tasks_from_retrosynthetic_proposals,
)
from cascade_planner.harness.schemas import (
    ArtifactBundle,
    FinalVerdict,
    TargetInput,
    append_jsonl,
    write_json,
)
from cascade_planner.harness.target_side_strategy import build_target_side_disconnection_hypotheses
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    artifact_bundle_from_state,
    execute_local_tool,
)
from cascade_planner.orchestration.codex_retrosynthesis import (
    RetrosynthesisTeamConfig,
    campaign_closure_status,
    reconcile_codex_campaign_proof_state,
)
from cascade_planner.providers.builtins import (
    CodexRetrosynthesisProvider,
    build_default_provider_registry,
)
from cascade_planner.providers.contracts import ProviderContext
from cascade_planner.providers.stock import (
    build_trusted_stock_provider_instances,
    replay_stock_provider_result,
)
from cascade_planner.source_locators import (
    source_content_scope,
    source_document_identity,
)
from cascade_planner.application.route_portfolio import (
    build_route_verifier_bundle,
    derive_portfolio_bindings,
    solve_diverse_routes,
    validate_portfolio_replacements,
)
from cascade_planner.application.frontier_ledger import (
    project_frontier_ledger,
    validate_frontier_ledger,
)
from cascade_planner.routes import (
    assemble_route_consensus_graph,
    consensus_to_blackboard_proposals,
    rebuild_consensus_graph_from_blackboard,
)
from cascade_planner.runtime.artifact_revision import (
    ArtifactRevisionError,
    publish_closeout_revision,
    sha256_file,
)


ActionPlannerRunner = Callable[..., dict[str, Any]]
AgentTeamRunner = Callable[[WorkerTask], Any]


def _nonnegative_budget_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, parsed)


@blackboard_controller_single_writer
def run_agentic_blackboard_controller(
    *,
    target_name: str,
    target_smiles: str,
    family_hint: str = "",
    output_dir: str | Path,
    literature_pdf_path: str | Path = "",
    literature_pdf_source_ref: str = "",
    literature_sources: list[dict[str, Any]] | None = None,
    auto_discover_local_pdfs: bool = True,
    local_pdf_search_dirs: list[str | Path] | None = None,
    timeout_s: float = 1800.0,
    key_path: str | Path = "",
    base_url: str = "https://api.wellau.com/v1",
    model: str = "gpt-5.5",
    max_rounds: int = 3,
    exhaust_round_budget: bool = False,
    enable_analogical_templates: bool = True,
    max_template_applications_per_round: int = 5,
    template_radius_policy: str = "auto",
    analog_template_confidence_threshold: str = "medium",
    use_codex_action_planner: bool | None = True,
    use_codex_agent_team: bool = False,
    codex_agent_team_max_depth: int = 2,
    codex_agent_team_max_expansions: int = 4,
    codex_agent_team_max_attempt_runs: int = 0,
    codex_agent_team_bootstrap_expansions: int = 1,
    codex_agent_team_max_expansions_per_invocation: int = 2,
    codex_agent_team_max_attempt_runs_per_invocation: int = 4,
    codex_agent_team_frontier_batch_size: int = 2,
    codex_agent_team_closure_objective: str = "benchmark_search",
    codex_agent_team_exploration_mode: str = "exhaustive",
    codex_agent_team_child_acceptance_mode: str = "strict_all",
    codex_agent_team_authority_lock_timeout_s: float = 3600.0,
    codex_agent_team_model: str = "",
    codex_agent_team_auth_mode: str = "auto",
    codex_agent_team_runner: AgentTeamRunner | None = None,
    codex_agent_team_stock_snapshots: dict[str, dict[str, Any]] | None = None,
    codex_agent_team_benchmark_stock_catalog_artifact: str | Path = "",
    codex_agent_team_benchmark_stock_catalog_sha256: str = "",
    codex_agent_team_benchmark_stock_catalog_name: str = "",
    stop_on_problem: bool = False,
    action_planner: ActionPlannerRunner | None = None,
    mock_tool_results: dict[str, Any] | None = None,
    prior_artifacts: dict[str, Any] | None = None,
    budget: HarnessBudget | None = None,
    emit_blackboard_steps: bool = False,
) -> dict[str, Any]:
    """Run the policy-driven DAG + blackboard controller."""
    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tool_calls.jsonl").touch()
    (run_dir / "decision_trace.jsonl").touch()
    budget = budget or HarnessBudget(timeout_s=float(timeout_s))
    budget.max_guided_chemenzy_runs = _nonnegative_budget_int(budget.max_guided_chemenzy_runs, default=1)
    budget.max_route_expansion_subgoal_runs = _nonnegative_budget_int(
        budget.max_route_expansion_subgoal_runs,
        default=2,
    )

    target = TargetInput(
        target_name=target_name,
        target_smiles=target_smiles,
        family_hint=family_hint,
        case_id="",
    )
    source_rows = _normalize_literature_sources(
        literature_pdf_path=literature_pdf_path,
        literature_pdf_source_ref=literature_pdf_source_ref,
        literature_sources=literature_sources,
        auto_discover_local_pdfs=auto_discover_local_pdfs,
        local_pdf_search_dirs=local_pdf_search_dirs,
        run_dir=run_dir,
    )
    target_data = target.to_dict()
    _attach_literature_sources(target_data, source_rows)
    target_data["analogical_template_policy"] = {
        "enabled": bool(enable_analogical_templates),
        "max_template_applications_per_round": max(1, int(max_template_applications_per_round or 5)),
        "template_radius_policy": str(template_radius_policy or "auto"),
        "analog_template_confidence_threshold": str(analog_template_confidence_threshold or "medium"),
    }
    write_json(run_dir / "target_input.json", target_data)
    write_json(run_dir / "budget.json", budget.to_dict())
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "start", "created_at_utc": _now(), "controller": "agentic_blackboard"})

    preflight = run_preflight(target)
    target.case_id = str(preflight.get("case_id") or target.case_id)
    target_data = target.to_dict()
    _attach_literature_sources(target_data, source_rows)
    target_data["analogical_template_policy"] = {
        "enabled": bool(enable_analogical_templates),
        "max_template_applications_per_round": max(1, int(max_template_applications_per_round or 5)),
        "template_radius_policy": str(template_radius_policy or "auto"),
        "analog_template_confidence_threshold": str(analog_template_confidence_threshold or "medium"),
    }
    write_json(run_dir / "target_input.json", target_data)
    write_json(run_dir / "preflight.json", preflight)
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "preflight", "preflight": preflight})

    state = ToolExecutionState(
        run_dir=run_dir,
        target_input=target_data,
        preflight=preflight,
        budget=budget,
        key_path=key_path or Path(__file__).resolve().parents[2] / "key.txt",
        base_url=base_url,
        model=model,
        mock_tool_results=dict(mock_tool_results or {}),
    )
    codex_campaign_refresh_config = (
        _controller_codex_team_config(
            timeout_s=timeout_s,
            max_depth=codex_agent_team_max_depth,
            max_expansions=codex_agent_team_max_expansions,
            max_attempt_runs=codex_agent_team_max_attempt_runs,
            max_expansions_per_invocation=codex_agent_team_max_expansions_per_invocation,
            max_attempt_runs_per_invocation=codex_agent_team_max_attempt_runs_per_invocation,
            frontier_batch_size=codex_agent_team_frontier_batch_size,
            closure_objective=codex_agent_team_closure_objective,
            exploration_mode=codex_agent_team_exploration_mode,
            child_acceptance_mode=codex_agent_team_child_acceptance_mode,
            campaign_authority_lock_timeout_s=(
                codex_agent_team_authority_lock_timeout_s
            ),
            model=codex_agent_team_model,
            auth_mode=codex_agent_team_auth_mode,
            stock_snapshots=codex_agent_team_stock_snapshots,
            benchmark_stock_catalog_artifact=codex_agent_team_benchmark_stock_catalog_artifact,
            benchmark_stock_catalog_sha256=codex_agent_team_benchmark_stock_catalog_sha256,
            benchmark_stock_catalog_name=codex_agent_team_benchmark_stock_catalog_name,
        )
        if use_codex_agent_team
        else None
    )
    if prior_artifacts:
        state.artifacts.update(dict(prior_artifacts))
    blackboard = initialize_agent_blackboard(
        target_input=target_data,
        preflight=preflight,
        max_rounds=max_rounds,
        budget_limits={
            **budget.to_dict(),
            "enable_analogical_templates": bool(enable_analogical_templates),
            "max_template_applications_per_round": max(1, int(max_template_applications_per_round or 5)),
            "template_radius_policy": str(template_radius_policy or "auto"),
            "analog_template_confidence_threshold": str(analog_template_confidence_threshold or "medium"),
        },
        prior_artifacts=prior_artifacts,
    )
    blackboard, recovery_report = rehydrate_blackboard_from_events(
        blackboard,
        run_dir=run_dir,
    )
    recovery_report_path = run_dir / "blackboard_events" / "recovery_report.json"
    state.artifacts["blackboard_rehydration"] = recovery_report
    blackboard["blackboard_rehydration"] = recovery_report
    blackboard.setdefault("artifact_refs", {})["blackboard_rehydration"] = str(
        recovery_report_path
    )
    if recovery_report.get("recovered") is True:
        append_jsonl(
            run_dir / "decision_trace.jsonl",
            {
                "stage": "blackboard_rehydrated",
                "event_count": int(recovery_report.get("event_count") or 0),
                "last_event_sha256": str(
                    recovery_report.get("last_event_sha256") or ""
                ),
                "restored_fields": list(
                    recovery_report.get("restored_fields") or []
                ),
                "agent_blackboard_json_used": False,
                "final_or_closeout_authority_restored": False,
            },
        )
    if prior_artifacts:
        blackboard = _seed_failure_evidence_from_prior(blackboard, state=state)
        blackboard = _seed_prior_analogical_evidence(blackboard, state=state)
    blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=run_dir)
    _restore_tool_execution_counters(state, blackboard)
    blackboard = _checkpoint_blackboard_event(
        blackboard,
        state=state,
        stage=(
            "controller_rehydrated"
            if recovery_report.get("recovered") is True
            else "controller_initialized"
        ),
        metadata={
            "prior_event_count": int(recovery_report.get("event_count") or 0),
            "projection_source_used": False,
        },
    )
    step_index = 0
    if emit_blackboard_steps:
        step_index = _emit_blackboard_step(
            blackboard,
            run_dir=run_dir,
            step_index=step_index,
            stage="initialized",
        )

    tool_calls: list[dict[str, Any]] = []
    action_batches: list[dict[str, Any]] = [
        dict(row)
        for row in blackboard.get("controller_action_batches") or []
        if isinstance(row, dict)
    ]
    validations: list[dict[str, Any]] = [
        dict(row)
        for row in blackboard.get("controller_action_batch_validations") or []
        if isinstance(row, dict)
    ]
    if not preflight.get("accepted"):
        blackboard, bundle, final = _finalize_agentic_run(
            state=state,
            blackboard=blackboard,
            action_batches=action_batches,
            validations=validations,
            tool_calls=tool_calls,
            explicit_final=_invalid_final(preflight),
            codex_campaign_config=codex_campaign_refresh_config,
        )
        return _result(run_dir, target_data, preflight, blackboard, action_batches, validations, bundle, final, tool_calls)

    if use_codex_agent_team:
        blackboard = _run_and_merge_codex_agent_team(
            blackboard=blackboard,
            state=state,
            target_name=target_name,
            target_smiles=target_smiles,
            literature_sources=source_rows,
            config=_controller_codex_team_config(
                timeout_s=timeout_s,
                max_depth=codex_agent_team_max_depth,
                max_expansions=codex_agent_team_max_expansions,
                max_attempt_runs=codex_agent_team_max_attempt_runs,
                max_expansions_per_invocation=min(
                    max(1, int(codex_agent_team_bootstrap_expansions or 1)),
                    max(1, int(codex_agent_team_max_expansions or 1)),
                ),
                max_attempt_runs_per_invocation=codex_agent_team_max_attempt_runs_per_invocation,
                frontier_batch_size=codex_agent_team_frontier_batch_size,
                closure_objective=codex_agent_team_closure_objective,
                exploration_mode=codex_agent_team_exploration_mode,
                child_acceptance_mode=codex_agent_team_child_acceptance_mode,
                campaign_authority_lock_timeout_s=(
                    codex_agent_team_authority_lock_timeout_s
                ),
                model=codex_agent_team_model,
                auth_mode=codex_agent_team_auth_mode,
                stock_snapshots=codex_agent_team_stock_snapshots,
                benchmark_stock_catalog_artifact=codex_agent_team_benchmark_stock_catalog_artifact,
                benchmark_stock_catalog_sha256=codex_agent_team_benchmark_stock_catalog_sha256,
                benchmark_stock_catalog_name=codex_agent_team_benchmark_stock_catalog_name,
            ),
            runner=codex_agent_team_runner,
        )
        if emit_blackboard_steps:
            step_index = _emit_blackboard_step(
                blackboard,
                run_dir=run_dir,
                step_index=step_index,
                stage="codex_agent_team",
                detail={
                    "accepted": bool((blackboard.get("codex_agent_team") or {}).get("accepted")),
                    "child_agent_count": len(
                        ((blackboard.get("codex_agent_team") or {}).get("coordinator") or {}).get("observed_child_agents") or []
                    ),
                },
            )

    stop_requested = False
    # ``stop_unresolved`` ends evidence/action planning.  In exhaustive mode a
    # planner fast-path caused by the first solved route must not also cancel
    # the independently resumable Codex campaign.
    evidence_stop_requested = False
    first_round_index = max(
        1,
        int((blackboard.get("budget_state") or {}).get("rounds_completed") or 0)
        + 1,
    )
    for round_index in range(first_round_index, int(max_rounds or 3) + 1):
        blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=run_dir)
        action_batch = _obtain_action_batch(
            blackboard=blackboard,
            round_index=round_index,
            run_dir=run_dir,
            state=state,
            action_planner=action_planner,
            exhaust_round_budget=exhaust_round_budget,
            use_codex_action_planner=use_codex_action_planner,
            allow_deterministic_fallback=not bool(use_codex_agent_team),
        )
        validation = validate_action_batch(action_batch, blackboard=blackboard)
        blackboard = update_blackboard_from_action_batch(
            blackboard,
            action_batch=action_batch,
            validation=validation,
            round_index=round_index,
        )
        blackboard["controller_action_batches"] = [
            *[dict(row) for row in action_batches],
            dict(action_batch),
        ]
        blackboard["controller_action_batch_validations"] = [
            *[dict(row) for row in validations],
            dict(validation),
        ]
        if emit_blackboard_steps:
            step_index = _emit_blackboard_step(
                blackboard,
                run_dir=run_dir,
                step_index=step_index,
                stage="action_batch",
                round_index=round_index,
                detail={
                    "action_count": len(action_batch.get("actions") or []),
                    "validation_accepted": bool(validation.get("accepted")),
                    "validation_reasons": [str(item) for item in validation.get("reasons") or []],
                },
            )
        validations.append(validation)
        action_batches.append(action_batch)
        batch_path = run_dir / f"action_batch_round_{round_index}.json"
        validation_path = run_dir / f"action_batch_validation_round_{round_index}.json"
        write_json(batch_path, action_batch)
        write_json(validation_path, validation)
        _record_action_batch_artifacts(
            state=state,
            blackboard=blackboard,
            action_batch=action_batch,
            validation=validation,
            round_index=round_index,
            batch_path=batch_path,
            validation_path=validation_path,
        )
        blackboard = _checkpoint_blackboard_event(
            blackboard,
            state=state,
            stage="action_batch_merged",
            metadata={
                "round_index": int(round_index),
                "validation_accepted": validation.get("accepted") is True,
                "action_count": len(action_batch.get("actions") or []),
            },
        )
        append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "action_batch", "round_index": round_index, "validation": validation})
        problem_reason = _stop_on_problem_action_batch_reason(action_batch, validation)
        if stop_on_problem and problem_reason:
            blackboard = _record_stop_on_problem(
                blackboard,
                run_dir=run_dir,
                round_index=round_index,
                reason=problem_reason,
                action_type="action_batch",
            )
            stop_requested = True
            break
        if not validation.get("accepted"):
            state.validations.append(validation)
            break

        round_useful = False
        action_validation_rows = {
            int(row.get("index") or 0): dict(row)
            for row in validation.get("action_validations") or []
            if isinstance(row, dict)
        }
        for action_index, raw_action in enumerate(action_batch.get("actions") or []):
            journal_action = deepcopy(dict(raw_action))
            action = deepcopy(journal_action)
            action_validation = dict(action_validation_rows.get(action_index) or {})
            effective_payload = action_validation.get("effective_payload")
            if isinstance(effective_payload, dict):
                action["payload"] = dict(effective_payload)
            host_resource_cost = dict(action_validation.get("cost") or {})
            action["_host_resource_cost"] = host_resource_cost
            action_type = str(action.get("action_type") or "")
            reserved_board = update_budget_for_action(
                blackboard,
                action_type,
                payload=dict(action.get("payload") or {}),
                resource_cost=host_resource_cost,
            )
            blackboard, action_lifecycle = begin_blackboard_action(
                run_dir,
                blackboard,
                action=journal_action,
                round_index=round_index,
                reserved_budget_state=dict(
                    reserved_board.get("budget_state") or {}
                ),
            )
            lifecycle_status = str(action_lifecycle.get("status") or "")
            prepared_event = action_lifecycle.get("prepared_event")
            if lifecycle_status == "indeterminate":
                reason = str(
                    action_lifecycle.get("reason")
                    or "prior_action_started_without_prepared_result"
                )
                blackboard.setdefault("safety_flags", []).append(
                    f"blackboard_action_recovery_blocked:{reason}"
                )
                blackboard.setdefault("route_failures", []).append(
                    {
                        "schema_version": "agent_action_recovery_failure.v1",
                        "round_index": int(round_index),
                        "action_id": str(action.get("action_id") or ""),
                        "action_type": action_type,
                        "reason": reason,
                        "charged_attempt_count": int(
                            action_lifecycle.get("charged_attempt_count") or 1
                        ),
                        "automatic_retry_allowed": False,
                        "requires_explicit_operator_resolution": True,
                        "no_solved_claim": True,
                    }
                )
                blackboard = _checkpoint_blackboard_event(
                    blackboard,
                    state=state,
                    stage="action_indeterminate_recovery_blocked",
                    metadata={
                        "round_index": int(round_index),
                        "action_id": str(action.get("action_id") or ""),
                        "action_type": action_type,
                        "reason": reason,
                    },
                )
                append_jsonl(
                    run_dir / "decision_trace.jsonl",
                    {
                        "stage": "agent_action_recovery_blocked",
                        "round_index": int(round_index),
                        "action_id": str(action.get("action_id") or ""),
                        "action_type": action_type,
                        "reason": reason,
                        "tool_reexecuted": False,
                    },
                )
                stop_requested = True
                break
            if lifecycle_status == "committed":
                prior_history = next(
                    (
                        dict(row)
                        for row in reversed(blackboard.get("action_history") or [])
                        if isinstance(row, dict)
                        and int(row.get("round_index") or 0) == int(round_index)
                        and str(row.get("action_id") or "")
                        == str(action.get("action_id") or "")
                    ),
                    {},
                )
                round_useful = round_useful or bool(
                    prior_history.get("useful_artifact")
                )
                append_jsonl(
                    run_dir / "decision_trace.jsonl",
                    {
                        "stage": "agent_action_commit_replayed",
                        "round_index": round_index,
                        "action_id": str(action.get("action_id") or ""),
                        "action_type": action_type,
                        "tool_reexecuted": False,
                    },
                )
                if action_type == "stop_unresolved":
                    if use_codex_agent_team and _controller_evidence_stop_preserves_campaign(
                        blackboard,
                        exploration_mode=codex_agent_team_exploration_mode,
                    ):
                        evidence_stop_requested = True
                    else:
                        stop_requested = True
                    break
                continue
            if lifecycle_status == "prepared" and isinstance(
                prepared_event, dict
            ):
                replay_payload = replay_prepared_blackboard_action(
                    prepared_event
                )
                action_result = dict(replay_payload.get("action_result") or {})
                records = [
                    dict(row)
                    for row in replay_payload.get("tool_records") or []
                    if isinstance(row, dict)
                ]
                state.artifacts.update(
                    deepcopy(
                        dict(replay_payload.get("artifact_updates") or {})
                    )
                )
                append_jsonl(
                    run_dir / "decision_trace.jsonl",
                    {
                        "stage": "agent_action_result_replayed",
                        "round_index": round_index,
                        "action_id": str(action.get("action_id") or ""),
                        "action_type": action_type,
                        "prepared_event_id": str(
                            prepared_event.get("event_id") or ""
                        ),
                        "tool_reexecuted": False,
                    },
                )
            else:
                artifacts_before = _artifact_digest_index(state.artifacts)
                action_result, records = _execute_agent_action(
                    action=action,
                    state=state,
                    blackboard=blackboard,
                )
                artifact_updates = _changed_artifact_updates(
                    state.artifacts,
                    before=artifacts_before,
                )
                blackboard, prepared_event = prepare_blackboard_action_result(
                    run_dir,
                    blackboard,
                    action=journal_action,
                    round_index=round_index,
                    started_event=dict(
                        action_lifecycle.get("started_event") or {}
                    ),
                    action_result=action_result,
                    tool_records=records,
                    artifact_updates=artifact_updates,
                )
            tool_calls.extend(records)
            blackboard = update_blackboard_from_action(
                blackboard,
                action=action,
                action_result=action_result,
                round_index=round_index,
                run_dir=run_dir,
            )
            blackboard, committed_event = commit_prepared_blackboard_action(
                run_dir,
                blackboard,
                action=journal_action,
                round_index=round_index,
                prepared_event=dict(prepared_event or {}),
            )
            state.artifacts["blackboard_event_journal"] = dict(
                blackboard.get("blackboard_event_journal") or {}
            )
            append_jsonl(
                run_dir / "decision_trace.jsonl",
                {
                    "stage": "blackboard_action_committed",
                    "round_index": int(round_index),
                    "action_id": str(action.get("action_id") or ""),
                    "action_type": action_type,
                    "event_id": str(committed_event.get("event_id") or ""),
                    "event_sha256": str(
                        committed_event.get("event_sha256") or ""
                    ),
                    "accepted": action_result.get("accepted", True) is True,
                },
            )
            if emit_blackboard_steps:
                step_index = _emit_blackboard_step(
                    blackboard,
                    run_dir=run_dir,
                    step_index=step_index,
                    stage="agent_action",
                    round_index=round_index,
                    action_id=str(action.get("action_id") or ""),
                    action_type=action_type,
                    detail={
                        "accepted": bool(action_result.get("accepted", True)),
                        "useful_artifact": bool(blackboard["action_history"][-1].get("useful_artifact")),
                    },
                )
            round_useful = round_useful or bool(blackboard["action_history"][-1].get("useful_artifact"))
            append_jsonl(
                run_dir / "decision_trace.jsonl",
                {
                    "stage": "agent_action",
                    "round_index": round_index,
                    "action_type": action_type,
                    "accepted": bool(action_result.get("accepted", True)),
                    "useful_artifact": bool(blackboard["action_history"][-1].get("useful_artifact")),
                },
            )
            problem_reason = _stop_on_problem_action_result_reason(
                action=action,
                action_result=action_result,
                history_record=dict(blackboard["action_history"][-1]),
            )
            if stop_on_problem and problem_reason:
                blackboard = _record_stop_on_problem(
                    blackboard,
                    run_dir=run_dir,
                    round_index=round_index,
                    reason=problem_reason,
                    action_type=action_type,
                )
                stop_requested = True
                break
            if action_type == "stop_unresolved":
                if use_codex_agent_team and _controller_evidence_stop_preserves_campaign(
                    blackboard,
                    exploration_mode=codex_agent_team_exploration_mode,
                ):
                    evidence_stop_requested = True
                else:
                    stop_requested = True
                break
        blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=run_dir)
        blackboard = _auto_update_critic(blackboard, state=state, run_dir=run_dir, round_index=round_index)
        if emit_blackboard_steps:
            step_index = _emit_blackboard_step(
                blackboard,
                run_dir=run_dir,
                step_index=step_index,
                stage="auto_critic",
                round_index=round_index,
            )
        if (
            use_codex_agent_team
            and not stop_requested
            and not _controller_codex_search_should_stop(
                blackboard,
                exploration_mode=codex_agent_team_exploration_mode,
            )
        ):
            # Evidence/proof refresh is independent of whether the action
            # planner labelled this round useful and independent of proposal
            # budget.  Late exact rows, stock results and verifier material can
            # therefore close durable requests without another model call.
            blackboard = _refresh_multisource_route_consensus(
                state=state,
                blackboard=blackboard,
                codex_campaign_config=codex_campaign_refresh_config,
            )
        if (
            use_codex_agent_team
            and not stop_requested
            and not _controller_codex_search_should_stop(
                blackboard,
                exploration_mode=codex_agent_team_exploration_mode,
            )
            and _codex_team_has_remaining_campaign_work(blackboard)
        ):
            prior_accepted_expansions = int(
                ((blackboard.get("codex_agent_team") or {}).get("campaign") or {}).get(
                    "accepted_expansion_count"
                )
                or 0
            )
            blackboard = _run_and_merge_codex_agent_team(
                blackboard=blackboard,
                state=state,
                target_name=target_name,
                target_smiles=target_smiles,
                literature_sources=_blackboard_literature_sources(blackboard),
                config=_controller_codex_team_config(
                    timeout_s=timeout_s,
                    max_depth=codex_agent_team_max_depth,
                    max_expansions=codex_agent_team_max_expansions,
                    max_attempt_runs=codex_agent_team_max_attempt_runs,
                    max_expansions_per_invocation=codex_agent_team_max_expansions_per_invocation,
                    max_attempt_runs_per_invocation=codex_agent_team_max_attempt_runs_per_invocation,
                    frontier_batch_size=codex_agent_team_frontier_batch_size,
                    closure_objective=codex_agent_team_closure_objective,
                    exploration_mode=codex_agent_team_exploration_mode,
                    child_acceptance_mode=codex_agent_team_child_acceptance_mode,
                    campaign_authority_lock_timeout_s=(
                        codex_agent_team_authority_lock_timeout_s
                    ),
                    model=codex_agent_team_model,
                    auth_mode=codex_agent_team_auth_mode,
                    stock_snapshots=codex_agent_team_stock_snapshots,
                    benchmark_stock_catalog_artifact=codex_agent_team_benchmark_stock_catalog_artifact,
                    benchmark_stock_catalog_sha256=codex_agent_team_benchmark_stock_catalog_sha256,
                    benchmark_stock_catalog_name=codex_agent_team_benchmark_stock_catalog_name,
                    reaction_proofs=_codex_edge_reaction_proofs(state.artifacts),
                ),
                runner=codex_agent_team_runner,
            )
            # A campaign report is reconstructed from immutable Codex commits
            # and is intentionally narrower than the fused graph.  Rebuild
            # immediately so external exact/ChemEnzy edges and late proof
            # updates cannot be overwritten until the next controller round.
            blackboard = _refresh_multisource_route_consensus(
                state=state,
                blackboard=blackboard,
                codex_campaign_config=codex_campaign_refresh_config,
            )
            current_campaign = dict(
                (blackboard.get("codex_agent_team") or {}).get("campaign") or {}
            )
            append_jsonl(
                run_dir / "decision_trace.jsonl",
                {
                    "stage": "codex_agent_team_resume",
                    "round_index": round_index,
                    "prior_accepted_expansion_count": prior_accepted_expansions,
                    "accepted_expansion_count": int(
                        current_campaign.get("accepted_expansion_count") or 0
                    ),
                    "invocation_accepted_expansion_count": int(
                        current_campaign.get("invocation_accepted_expansion_count")
                        or 0
                    ),
                    "stop_reason": str(current_campaign.get("stop_reason") or ""),
                },
            )
            if emit_blackboard_steps:
                step_index = _emit_blackboard_step(
                    blackboard,
                    run_dir=run_dir,
                    step_index=step_index,
                    stage="codex_agent_team_resume",
                    round_index=round_index,
                    detail={
                        "accepted_expansion_count": int(
                            current_campaign.get("accepted_expansion_count") or 0
                        ),
                        "graph_complete": bool(current_campaign.get("graph_complete")),
                        "stop_reason": str(current_campaign.get("stop_reason") or ""),
                    },
                )
        blackboard = complete_round(blackboard, round_index)
        blackboard = _checkpoint_blackboard_event(
            blackboard,
            state=state,
            stage="round_completed",
            metadata={
                "round_index": int(round_index),
                "round_useful": bool(round_useful),
            },
        )
        if emit_blackboard_steps:
            step_index = _emit_blackboard_step(
                blackboard,
                run_dir=run_dir,
                step_index=step_index,
                stage="round_complete",
                round_index=round_index,
                detail={"round_useful": bool(round_useful)},
            )
        write_json(run_dir / "agent_blackboard.json", blackboard)
        controller_search_complete = (
            _controller_codex_search_should_stop(
                blackboard,
                exploration_mode=codex_agent_team_exploration_mode,
            )
            if use_codex_agent_team
            else _parent_proof_accepted(blackboard)
        )
        if stop_requested or evidence_stop_requested or controller_search_complete:
            break
        if not round_useful and round_index >= int(max_rounds or 3):
            break

    # ``max_rounds`` bounds evidence/action planning; it is not a claim that
    # the durable synthesis graph is closed.  Drain resumable proposal work in
    # separately bounded campaign invocations, refreshing host proof/stock
    # state before and after every invocation.  This makes the terminal state
    # closure-driven while retaining explicit accepted/attempt/depth caps.
    campaign_drain_records: list[dict[str, Any]] = []
    campaign_drain_stop_reason = "not_enabled"
    if (
        use_codex_agent_team
        and not stop_requested
        and not _controller_codex_search_should_stop(
            blackboard,
            exploration_mode=codex_agent_team_exploration_mode,
        )
    ):
        blackboard = _refresh_multisource_route_consensus(
            state=state,
            blackboard=blackboard,
            codex_campaign_config=codex_campaign_refresh_config,
        )
        per_invocation = max(
            1, int(codex_agent_team_max_expansions_per_invocation or 1)
        )
        # One extra pass permits queue maintenance/reconciliation when an
        # invocation accepts no proposal expansion.
        max_drain_invocations = max(
            1,
            (
                max(1, int(codex_agent_team_max_expansions or 1))
                + per_invocation
                - 1
            )
            // per_invocation
            + 1,
        )
        seen_progress: set[str] = set()
        for drain_index in range(1, max_drain_invocations + 1):
            team_before = dict(blackboard.get("codex_agent_team") or {})
            campaign_before = dict(team_before.get("campaign") or {})
            if _campaign_search_complete(campaign_before):
                campaign_drain_stop_reason = "campaign_search_complete"
                break
            if not _codex_team_has_remaining_campaign_work(blackboard):
                campaign_drain_stop_reason = _campaign_nonresumable_reason(
                    campaign_before
                )
                break
            before_signature = _codex_campaign_progress_signature(
                blackboard
            )
            prior_accepted = int(
                campaign_before.get("accepted_expansion_count") or 0
            )
            blackboard = _run_and_merge_codex_agent_team(
                blackboard=blackboard,
                state=state,
                target_name=target_name,
                target_smiles=target_smiles,
                literature_sources=_blackboard_literature_sources(blackboard),
                config=_controller_codex_team_config(
                    timeout_s=timeout_s,
                    max_depth=codex_agent_team_max_depth,
                    max_expansions=codex_agent_team_max_expansions,
                    max_attempt_runs=codex_agent_team_max_attempt_runs,
                    max_expansions_per_invocation=codex_agent_team_max_expansions_per_invocation,
                    max_attempt_runs_per_invocation=codex_agent_team_max_attempt_runs_per_invocation,
                    frontier_batch_size=codex_agent_team_frontier_batch_size,
                    closure_objective=codex_agent_team_closure_objective,
                    exploration_mode=codex_agent_team_exploration_mode,
                    child_acceptance_mode=codex_agent_team_child_acceptance_mode,
                    campaign_authority_lock_timeout_s=(
                        codex_agent_team_authority_lock_timeout_s
                    ),
                    model=codex_agent_team_model,
                    auth_mode=codex_agent_team_auth_mode,
                    stock_snapshots=codex_agent_team_stock_snapshots,
                    benchmark_stock_catalog_artifact=codex_agent_team_benchmark_stock_catalog_artifact,
                    benchmark_stock_catalog_sha256=codex_agent_team_benchmark_stock_catalog_sha256,
                    benchmark_stock_catalog_name=codex_agent_team_benchmark_stock_catalog_name,
                    reaction_proofs=_codex_edge_reaction_proofs(state.artifacts),
                ),
                runner=codex_agent_team_runner,
            )
            blackboard = _refresh_multisource_route_consensus(
                state=state,
                blackboard=blackboard,
                codex_campaign_config=codex_campaign_refresh_config,
            )
            campaign_after = dict(
                (blackboard.get("codex_agent_team") or {}).get("campaign") or {}
            )
            after_signature = _codex_campaign_progress_signature(blackboard)
            record = {
                "schema_version": "codex_campaign_drain_invocation.v1",
                "drain_index": drain_index,
                "prior_accepted_expansion_count": prior_accepted,
                "accepted_expansion_count": int(
                    campaign_after.get("accepted_expansion_count") or 0
                ),
                "invocation_accepted_expansion_count": int(
                    campaign_after.get("invocation_accepted_expansion_count")
                    or 0
                ),
                "invocation_attempt_run_count": int(
                    campaign_after.get("invocation_attempt_run_count") or 0
                ),
                "graph_complete": campaign_after.get("graph_complete") is True,
                "route_solved": campaign_after.get("route_solved") is True,
                "campaign_search_complete": _campaign_search_complete(
                    campaign_after
                ),
                "stop_reason": str(campaign_after.get("stop_reason") or ""),
                "progressed": before_signature != after_signature,
            }
            campaign_drain_records.append(record)
            append_jsonl(
                run_dir / "decision_trace.jsonl",
                {"stage": "codex_campaign_drain", **record},
            )
            if _controller_codex_search_should_stop(
                blackboard,
                exploration_mode=codex_agent_team_exploration_mode,
            ):
                campaign_drain_stop_reason = _controller_codex_search_stop_reason(
                    blackboard,
                    exploration_mode=codex_agent_team_exploration_mode,
                )
                break
            if before_signature == after_signature or after_signature in seen_progress:
                campaign_drain_stop_reason = "campaign_drain_no_progress"
                break
            seen_progress.add(after_signature)
        else:
            campaign_drain_stop_reason = "campaign_drain_invocation_cap_reached"
    elif stop_requested:
        campaign_drain_stop_reason = "controller_stop_requested"
    elif use_codex_agent_team and _controller_codex_search_should_stop(
        blackboard,
        exploration_mode=codex_agent_team_exploration_mode,
    ):
        campaign_drain_stop_reason = _controller_codex_search_stop_reason(
            blackboard,
            exploration_mode=codex_agent_team_exploration_mode,
        )

    campaign_drain = {
        "schema_version": "codex_campaign_drain.v1",
        "enabled": bool(use_codex_agent_team),
        "evidence_round_budget": max(0, int(max_rounds or 0)),
        "invocation_count": len(campaign_drain_records),
        "stop_reason": campaign_drain_stop_reason,
        "graph_complete": bool(
            ((blackboard.get("codex_agent_team") or {}).get("campaign") or {}).get(
                "graph_complete"
            )
        ),
        "route_solved": bool(
            ((blackboard.get("codex_agent_team") or {}).get("campaign") or {}).get(
                "route_solved"
            )
        ),
        "campaign_search_complete": _campaign_search_complete(
            dict(
                ((blackboard.get("codex_agent_team") or {}).get("campaign") or {})
            )
        ),
        "invocations": campaign_drain_records,
        "semantics": {
            "max_rounds_is_evidence_budget_not_closure_claim": True,
            "proposal_budget_remains_hard_cap": True,
            "no_progress_stops_drain": True,
        },
    }
    campaign_drain_path = run_dir / "codex_campaign_drain.json"
    write_json(campaign_drain_path, campaign_drain)
    state.artifacts["codex_campaign_drain"] = campaign_drain
    blackboard.setdefault("artifact_refs", {})["codex_campaign_drain"] = str(
        campaign_drain_path
    )
    team_after_drain = dict(blackboard.get("codex_agent_team") or {})
    if team_after_drain:
        campaign_after_drain = dict(team_after_drain.get("campaign") or {})
        campaign_after_drain["controller_drain"] = {
            key: campaign_drain[key]
            for key in ("schema_version", "invocation_count", "stop_reason", "graph_complete")
        }
        team_after_drain["campaign"] = campaign_after_drain
        blackboard["codex_agent_team"] = team_after_drain
        state.artifacts["codex_retrosynthesis_team"] = team_after_drain
        blackboard = _write_codex_controller_projection(
            state=state,
            blackboard=blackboard,
            stage="codex_campaign_drain",
        )
        blackboard = _checkpoint_blackboard_event(
            blackboard,
            state=state,
            stage="codex_campaign_drain_checkpoint",
            metadata={
                "invocation_count": len(campaign_drain_records),
                "stop_reason": campaign_drain_stop_reason,
            },
        )

    blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=run_dir)
    blackboard, bundle, final = _finalize_agentic_run(
        state=state,
        blackboard=blackboard,
        action_batches=action_batches,
        validations=validations,
        tool_calls=tool_calls,
        codex_campaign_config=codex_campaign_refresh_config,
    )
    return _result(run_dir, target_data, preflight, blackboard, action_batches, validations, bundle, final, tool_calls)


def _checkpoint_blackboard_event(
    blackboard: dict[str, Any],
    *,
    state: ToolExecutionState,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    board, event = append_blackboard_checkpoint(
        state.run_dir,
        blackboard,
        stage=stage,
        metadata=metadata,
    )
    state.artifacts["blackboard_event_journal"] = dict(
        board.get("blackboard_event_journal") or {}
    )
    append_jsonl(
        state.run_dir / "decision_trace.jsonl",
        {
            "stage": "blackboard_checkpoint",
            "checkpoint_stage": str(stage),
            "event_id": str(event.get("event_id") or ""),
            "event_sha256": str(event.get("event_sha256") or ""),
            "sequence": int(event.get("sequence") or 0),
        },
    )
    return board


def _artifact_digest_index(artifacts: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        for key, value in artifacts.items()
    }


def _changed_artifact_updates(
    artifacts: dict[str, Any],
    *,
    before: dict[str, str],
) -> dict[str, Any]:
    current = _artifact_digest_index(artifacts)
    return {
        str(key): deepcopy(value)
        for key, value in artifacts.items()
        if before.get(str(key)) != current.get(str(key))
    }


def _restore_tool_execution_counters(
    state: ToolExecutionState,
    blackboard: dict[str, Any],
) -> None:
    budget = dict(blackboard.get("budget_state") or {})
    state.guided_chemenzy_runs = max(
        int(state.guided_chemenzy_runs), int(budget.get("chemenzy_runs") or 0)
    )
    state.route_expansion_subgoal_runs = max(
        int(state.route_expansion_subgoal_runs),
        int(budget.get("child_target_runs") or 0),
    )
    state.codex_research_runs = max(
        int(state.codex_research_runs),
        int(budget.get("codex_research_runs") or 0),
    )


def _controller_codex_team_config(
    *,
    timeout_s: float,
    max_depth: int,
    max_expansions: int,
    max_attempt_runs: int,
    max_expansions_per_invocation: int,
    max_attempt_runs_per_invocation: int,
    frontier_batch_size: int,
    closure_objective: str,
    exploration_mode: str,
    child_acceptance_mode: str,
    campaign_authority_lock_timeout_s: float,
    model: str,
    auth_mode: str,
    stock_snapshots: dict[str, dict[str, Any]] | None,
    benchmark_stock_catalog_artifact: str | Path,
    benchmark_stock_catalog_sha256: str,
    benchmark_stock_catalog_name: str,
    reaction_proofs: dict[str, dict[str, Any]] | None = None,
) -> RetrosynthesisTeamConfig:
    return RetrosynthesisTeamConfig(
        timeout_s=min(float(timeout_s), 900.0),
        max_depth=max(1, int(max_depth or 1)),
        max_expansions=max(1, int(max_expansions or 1)),
        max_attempt_runs=max(0, int(max_attempt_runs or 0)),
        max_expansions_per_invocation=max(
            1, int(max_expansions_per_invocation or 1)
        ),
        max_attempt_runs_per_invocation=max(
            1, int(max_attempt_runs_per_invocation or 1)
        ),
        frontier_batch_size=max(1, int(frontier_batch_size or 1)),
        closure_objective=str(closure_objective or "benchmark_search"),
        exploration_mode=str(exploration_mode or "exhaustive"),
        child_acceptance_mode=str(child_acceptance_mode or "strict_all"),
        campaign_authority_lock_timeout_s=max(
            0.1, float(campaign_authority_lock_timeout_s or 3600.0)
        ),
        model=str(model or ""),
        auth_mode=str(auth_mode or "auto"),
        stock_snapshots=dict(stock_snapshots or {}),
        benchmark_stock_catalog_artifact=str(
            benchmark_stock_catalog_artifact or ""
        ),
        benchmark_stock_catalog_sha256=str(
            benchmark_stock_catalog_sha256 or ""
        ),
        benchmark_stock_catalog_name=str(benchmark_stock_catalog_name or ""),
        reaction_proofs=dict(reaction_proofs or {}),
    )


def _write_codex_controller_projection(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    stage: str,
    failure: dict[str, Any] | None = None,
    prior_accepted_team_preserved: bool = False,
) -> dict[str, Any]:
    """Persist controller-owned annotations without mutating the team report."""

    board = dict(blackboard)
    projection_path = (
        state.run_dir / "codex_retrosynthesis_team" / "controller_projection.json"
    )
    durable_report_path = (
        state.run_dir / "codex_retrosynthesis_team" / "team_report.json"
    )
    previous: dict[str, Any] = {}
    if projection_path.is_file():
        try:
            raw = json.loads(projection_path.read_text(encoding="utf-8"))
            previous = dict(raw) if isinstance(raw, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            previous = {}
    failure_row = dict(failure or {})
    event = {
        "schema_version": "codex_controller_projection_event.v1",
        "stage": str(stage or "controller_refresh"),
        "accepted": not failure_row,
        "prior_accepted_team_preserved": bool(prior_accepted_team_preserved),
        "failure": failure_row,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    events = [
        dict(row)
        for row in previous.get("events") or []
        if isinstance(row, dict)
    ]
    events.append(event)
    payload = {
        "schema_version": "codex_retrosynthesis_controller_projection.v1",
        "case_id": str(
            board.get("case_id")
            or state.preflight.get("case_id")
            or "target"
        ),
        "durable_team_report_ref": str(durable_report_path),
        "durable_team_report_exists": durable_report_path.is_file(),
        "durable_team_report_sha256": (
            sha256_file(durable_report_path) if durable_report_path.is_file() else ""
        ),
        "team_projection": dict(board.get("codex_agent_team") or {}),
        "latest_stage": event["stage"],
        "latest_failure": failure_row,
        "prior_accepted_team_preserved": bool(prior_accepted_team_preserved),
        "events": events,
        "semantics": {
            "durable_team_report_owned_by_provider_orchestration": True,
            "controller_never_rewrites_durable_team_report": True,
            "controller_projection_is_not_campaign_recovery_authority": True,
        },
    }
    digest_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["content_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    write_json(projection_path, payload)
    state.artifacts["codex_retrosynthesis_controller_projection"] = payload
    refs = dict(board.get("artifact_refs") or {})
    refs["codex_retrosynthesis_controller_projection"] = str(projection_path)
    durable_report_current_host_verified = _prior_codex_team_current_host_verified(
        state,
        dict(board.get("codex_agent_team") or {}),
        case_id=str(board.get("case_id") or state.preflight.get("case_id") or ""),
    )
    if durable_report_path.is_file() and durable_report_current_host_verified:
        refs["codex_retrosynthesis_team"] = str(durable_report_path)
    else:
        refs.pop("codex_retrosynthesis_team", None)
    board["artifact_refs"] = refs
    return board


def _run_and_merge_codex_agent_team(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    target_name: str,
    target_smiles: str,
    literature_sources: list[dict[str, Any]],
    config: RetrosynthesisTeamConfig,
    runner: AgentTeamRunner | None,
) -> dict[str, Any]:
    board = dict(blackboard)
    prior_team = dict(board.get("codex_agent_team") or {})
    controller_failure: dict[str, Any] = {}
    prior_accepted_team_preserved = False
    prior_team_current_host_verified = _prior_codex_team_current_host_verified(
        state,
        prior_team,
        case_id=str(board.get("case_id") or state.preflight.get("case_id") or ""),
    )
    try:
        case_id = str(board.get("case_id") or state.preflight.get("case_id") or "target")
        backend = CodexRetrosynthesisProvider(
            run_dir=state.run_dir,
            repository_root=Path(__file__).resolve().parents[2],
            config=config,
            runner=runner,
        )
        registry = build_default_provider_registry(include_codex=backend)
        envelope = registry.invoke(
            backend.descriptor.provider_id,
            {
                "schema_version": "codex_retrosynthesis_campaign_request.v1",
                "case_id": case_id,
                "target_name": target_name,
                "target_smiles": target_smiles,
                "blackboard_context": board,
                "literature_sources": literature_sources,
            },
            context=ProviderContext(
                run_id=case_id,
                case_id=case_id,
                target_smiles=target_smiles,
            ),
        )
        report = dict(envelope.payload)
        report["provider_envelope"] = envelope.to_dict()
    except Exception as exc:
        controller_failure = {
            "schema_version": "codex_retrosynthesis_team_run.v1",
            "accepted": False,
            "case_id": str(board.get("case_id") or "target"),
            "reasons": [f"team_runtime_error:{type(exc).__name__}:{exc}"],
            "semantics": {
                "codex_child_agents_required": True,
                "deterministic_scientific_fallback_used": False,
                "no_solved_claim": True,
            },
        }
        if prior_team.get("accepted") is True and prior_team_current_host_verified:
            report = prior_team
            prior_accepted_team_preserved = True
        else:
            report = controller_failure
    if (
        not controller_failure
        and prior_team.get("accepted") is True
        and prior_team_current_host_verified
        and report.get("accepted") is not True
    ):
        # Providers may encode a resumed runtime failure as a rejected report
        # rather than raising.  It has the same preservation semantics: retain
        # the last accepted orchestration projection and record this attempt
        # only in controller-owned history/projection.
        controller_failure = dict(report)
        report = prior_team
        prior_accepted_team_preserved = True
    if report.get("accepted") is True and not controller_failure:
        _mark_codex_team_current_host_verified(
            state,
            report,
            case_id=str(
                board.get("case_id") or state.preflight.get("case_id") or ""
            ),
        )
    if report.get("accepted") is not True and not prior_accepted_team_preserved:
        board = _clear_unverified_codex_scientific_projection(board)
    board["codex_agent_team"] = report
    state.artifacts["codex_retrosynthesis_team"] = report
    refs = dict(board.get("artifact_refs") or {})
    durable_team_report_path = (
        state.run_dir / "codex_retrosynthesis_team" / "team_report.json"
    )
    if durable_team_report_path.is_file() and (
        (report.get("accepted") is True and not controller_failure)
        or prior_accepted_team_preserved
    ):
        refs["codex_retrosynthesis_team"] = str(durable_team_report_path)
    else:
        refs.pop("codex_retrosynthesis_team", None)
    if report.get("route_consensus_ref"):
        refs["route_consensus"] = str(report["route_consensus_ref"])
    if report.get("route_consensus_graph_ref"):
        refs["route_consensus_graph"] = str(report["route_consensus_graph_ref"])
    board["artifact_refs"] = refs
    board.setdefault("agent_team_history", []).append({
        "schema_version": "codex_agent_team_history.v1",
        "accepted": bool(report.get("accepted")) and not controller_failure,
        "prior_accepted_team_preserved": prior_accepted_team_preserved,
        "coordinator": dict(report.get("coordinator") or {}),
        "reasons": [
            str(item)
            for item in (
                controller_failure.get("reasons")
                if controller_failure
                else report.get("reasons")
            )
            or []
        ],
    })
    if report.get("accepted") and not controller_failure:
        board = _merge_codex_team_source_hints(board, report=report)
        board["route_consensus"] = dict(report.get("route_consensus") or {})
        board["route_consensus_graph"] = dict(report.get("route_consensus_graph") or {})
        existing = {
            str(row.get("proposal_id") or ""): dict(row)
            for row in board.get("retrosynthetic_proposals") or []
            if isinstance(row, dict) and str(row.get("proposal_id") or "")
        }
        for row in report.get("blackboard_proposals") or []:
            if isinstance(row, dict) and str(row.get("proposal_id") or ""):
                existing[str(row["proposal_id"])] = dict(row)
        board["retrosynthetic_proposals"] = list(existing.values())
        board = _inject_codex_precursor_frontiers(
            board,
            proposals=[
                dict(row)
                for row in report.get("blackboard_proposals") or []
                if isinstance(row, dict)
            ],
        )
    else:
        board.setdefault("safety_flags", []).append("codex_agent_team_not_accepted")
    board = _write_codex_controller_projection(
        state=state,
        blackboard=board,
        stage="codex_agent_team_invocation",
        failure=controller_failure,
        prior_accepted_team_preserved=prior_accepted_team_preserved,
    )
    if report.get("accepted") is True and not controller_failure:
        board = _checkpoint_blackboard_event(
            board,
            state=state,
            stage="codex_agent_team_accepted",
            metadata={
                "accepted_expansion_count": int(
                    (report.get("campaign") or {}).get(
                        "accepted_expansion_count"
                    )
                    or 0
                ),
                "recursive_hypothesis_task_count": len(
                    board.get("recursive_hypothesis_tasks") or []
                ),
            },
        )
    write_json(state.run_dir / "agent_blackboard.json", board)
    append_jsonl(
        state.run_dir / "decision_trace.jsonl",
        {
            "stage": "codex_agent_team",
            "accepted": bool(report.get("accepted")) and not controller_failure,
            "prior_accepted_team_preserved": prior_accepted_team_preserved,
            "coordinator": dict(report.get("coordinator") or {}),
            "reasons": [
                str(item)
                for item in (
                    controller_failure.get("reasons")
                    if controller_failure
                    else report.get("reasons")
                )
                or []
            ],
        },
    )
    return board


def _codex_team_replay_binding(
    report: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, str]:
    campaign = dict(report.get("campaign") or {})
    identity_sha256 = str(
        report.get("campaign_identity_sha256")
        or campaign.get("campaign_identity_sha256")
        or ""
    )
    return {
        "case_id": str(report.get("case_id") or case_id or ""),
        "campaign_identity_sha256": identity_sha256,
    }


def _clear_unverified_codex_scientific_projection(
    board: dict[str, Any],
) -> dict[str, Any]:
    cleared = dict(board)
    for field in (
        "route_consensus",
        "route_consensus_graph",
        "codex_precursor_frontier_injection",
    ):
        cleared.pop(field, None)
    cleared["retrosynthetic_proposals"] = [
        dict(row)
        for row in cleared.get("retrosynthetic_proposals") or []
        if isinstance(row, dict)
        and "codex_strategy"
        not in {
            str(item)
            for item in (
                row.get("source_channels")
                or [row.get("source_channel")]
            )
            if str(item or "")
        }
    ]
    refs = dict(cleared.get("artifact_refs") or {})
    for key in (
        "codex_retrosynthesis_team",
        "route_consensus",
        "route_consensus_graph",
    ):
        refs.pop(key, None)
    cleared["artifact_refs"] = refs
    return cleared


def _mark_codex_team_current_host_verified(
    state: ToolExecutionState,
    report: dict[str, Any],
    *,
    case_id: str,
) -> None:
    setattr(
        state,
        "_codex_current_host_replay_authority",
        _codex_team_replay_binding(report, case_id=case_id),
    )


def _prior_codex_team_current_host_verified(
    state: ToolExecutionState,
    report: dict[str, Any],
    *,
    case_id: str,
) -> bool:
    authority = getattr(state, "_codex_current_host_replay_authority", None)
    if not isinstance(authority, dict) or report.get("accepted") is not True:
        return False
    binding = _codex_team_replay_binding(report, case_id=case_id)
    if not binding["case_id"] or binding["case_id"] != str(
        authority.get("case_id") or ""
    ):
        return False
    expected_identity = str(authority.get("campaign_identity_sha256") or "")
    return bool(
        expected_identity
        and binding["campaign_identity_sha256"] == expected_identity
    )


def _inject_codex_precursor_frontiers(
    blackboard: dict[str, Any],
    *,
    proposals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project admitted Codex precursors into bounded child-target work.

    This is molecule-frontier injection, never raw reaction injection. The
    candidate parent edge remains advisory and must independently reach L2.
    """

    board = dict(blackboard)
    existing_rows = [
        dict(row)
        for row in board.get("recursive_hypothesis_tasks") or []
        if isinstance(row, dict)
    ]
    existing_ids = {
        str(row.get("task_id") or "") for row in existing_rows if row.get("task_id")
    }
    generated = recursive_tasks_from_retrosynthetic_proposals(
        board,
        proposals,
        max_tasks=24,
    )
    injected: list[dict[str, Any]] = []
    for raw in generated:
        task = {
            **dict(raw),
            "frontier_origin": "codex_consensus_precursor",
            "raw_reaction_injection": False,
            "parent_edge_requires_independent_l2_validation": True,
        }
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in existing_ids:
            continue
        existing_ids.add(task_id)
        existing_rows.append(task)
        injected.append(task)
    board["recursive_hypothesis_tasks"] = existing_rows
    audit = {
        "schema_version": "codex_precursor_frontier_injection.v1",
        "input_proposal_count": len(proposals),
        "generated_frontier_count": len(generated),
        "new_frontier_count": len(injected),
        "new_frontier_task_ids": [str(row.get("task_id") or "") for row in injected],
        "semantics": {
            "molecule_frontier_injection": True,
            "raw_reaction_injection": False,
            "advisory_parent_edge_requires_l2": True,
            "child_route_cannot_promote_parent": True,
        },
    }
    board["codex_precursor_frontier_injection"] = audit
    if injected:
        belief = dict(board.get("current_belief") or {})
        biases = [str(item) for item in belief.get("next_action_bias") or []]
        if "expand_child_target" not in biases:
            biases.append("expand_child_target")
        belief["next_action_bias"] = biases
        board["current_belief"] = belief
    return board


def _merge_codex_team_source_hints(
    blackboard: dict[str, Any],
    *,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Promote traceable team citations into the normal source lifecycle.

    The promotion is deliberately metadata-only.  A Codex child citation can
    schedule deterministic acquisition/extraction, but it never becomes an
    exact literature row or a reaction proof merely because several child
    roles repeated it.
    """

    board = dict(blackboard)
    consensus = dict(report.get("route_consensus") or {})
    by_ref: dict[str, dict[str, Any]] = {}
    for proposal in consensus.get("proposals") or []:
        if not isinstance(proposal, dict):
            continue
        proposal_id = str(proposal.get("consensus_id") or "")
        family = str(proposal.get("reaction_family") or "").strip()
        evidence_refs = [
            str(value).strip()
            for value in proposal.get("evidence_refs") or []
            if str(value or "").strip()
        ]
        for raw_ref in proposal.get("source_refs") or []:
            parsed = _parse_codex_source_ref(raw_ref)
            source_ref = str(parsed.get("source_ref") or "")
            if not source_ref:
                continue
            row = by_ref.setdefault(
                source_ref,
                {
                    "schema_version": "literature_source_candidate.v1",
                    "candidate_id": f"codex_team_source_{hashlib.sha256(source_ref.encode('utf-8')).hexdigest()[:16]}",
                    "source_ref": source_ref,
                    "source_type": str(parsed.get("source_type") or "web_source"),
                    "title": str(parsed.get("title") or source_ref),
                    "doi": str(parsed.get("doi") or ""),
                    "url": str(parsed.get("url") or ""),
                    "local_pdf": "",
                    "access_status": "metadata_pointer_only",
                    "source_discovery_mode": "codex_agent_team",
                    "relevance_rationale": "A Codex literature role cited this source for a route hypothesis; acquire and parse the source before granting evidence authority.",
                    "expected_scheme_or_compound_labels": [],
                    "extraction_task_recommendations": [
                        "acquire_source_detail",
                        "extract_structured_reaction_steps",
                        "bind_exact_structures_and_source_locations",
                    ],
                    "proposal_ids": [],
                    "evidence_refs": [],
                    "no_solved_claim": True,
                    "not_exact_literature_evidence": True,
                },
            )
            if proposal_id and proposal_id not in row["proposal_ids"]:
                row["proposal_ids"].append(proposal_id)
            if family and family not in row["expected_scheme_or_compound_labels"]:
                row["expected_scheme_or_compound_labels"].append(family)
            row["evidence_refs"] = _dedupe(
                [*row.get("evidence_refs", []), *evidence_refs]
            )

    if not by_ref:
        return board
    evidence = dict(board.get("literature_evidence") or {})
    evidence["source_candidates"] = _merge_source_candidate_rows(
        list(evidence.get("source_candidates") or []),
        list(by_ref.values()),
    )
    evidence["source_refs"] = _dedupe(
        [
            *[str(value) for value in evidence.get("source_refs") or []],
            *sorted(by_ref),
        ]
    )
    evidence["team_source_hint_count"] = len(by_ref)
    evidence["team_source_hints_are_metadata_only"] = True
    board["literature_evidence"] = evidence
    _refresh_source_lifecycle(board)
    return board


def _parse_codex_source_ref(value: Any) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    fields: dict[str, str] = {}
    parts = [part.strip() for part in text.split(";") if part.strip()]
    head = parts[0] if parts else text
    for part in parts[1:]:
        if ":" in part:
            key, raw = part.split(":", 1)
            fields[key.strip().lower()] = raw.strip()
    lower = head.lower()
    if lower.startswith("doi:"):
        doi = head.split(":", 1)[1].strip()
        source_ref = f"doi:{doi.lower()}"
        return {
            "source_ref": source_ref,
            "source_type": "peer_reviewed_article",
            "doi": doi,
            "url": fields.get("url") or f"https://doi.org/{doi}",
            "title": source_ref,
        }
    if lower.startswith(("patent_publication:", "patent:")):
        patent_id = head.split(":", 1)[1].strip()
        source_ref = f"patent:{patent_id.upper()}"
        return {
            "source_ref": source_ref,
            "source_type": "patent",
            "doi": "",
            "url": fields.get("url", ""),
            "title": patent_id.upper(),
        }
    url = fields.get("url") or (text if lower.startswith(("http://", "https://")) else "")
    if url:
        return {
            "source_ref": f"url:{url}",
            "source_type": "web_source",
            "doi": "",
            "url": url,
            "title": url,
        }
    return {}


def _emit_blackboard_step(
    blackboard: dict[str, Any],
    *,
    run_dir: Path,
    step_index: int,
    stage: str,
    round_index: int | None = None,
    action_id: str = "",
    action_type: str = "",
    detail: dict[str, Any] | None = None,
) -> int:
    step_dir = run_dir / "blackboard_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    next_index = int(step_index) + 1
    token_parts = [f"{next_index:04d}", stage]
    if round_index is not None:
        token_parts.append(f"r{int(round_index)}")
    if action_type:
        token_parts.append(action_type)
    if action_id:
        token_parts.append(action_id)
    filename = _safe_artifact_filename("_".join(token_parts)) + ".json"
    summary = _blackboard_step_summary(
        blackboard,
        step_index=next_index,
        stage=stage,
        round_index=round_index,
        action_id=action_id,
        action_type=action_type,
        detail=detail,
    )
    write_json(
        step_dir / filename,
        {
            "schema_version": "agent_blackboard_step_snapshot.v1",
            "created_at_utc": _now(),
            "summary": summary,
            "blackboard": blackboard,
        },
    )
    append_jsonl(step_dir / "summary.jsonl", summary)
    return next_index


def _blackboard_step_summary(
    blackboard: dict[str, Any],
    *,
    step_index: int,
    stage: str,
    round_index: int | None,
    action_id: str,
    action_type: str,
    detail: dict[str, Any] | None,
) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    budget = dict(blackboard.get("budget_state") or {})
    history = [row for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    planner_history = [row for row in blackboard.get("planner_history") or [] if isinstance(row, dict)]
    last_action = dict(history[-1]) if history else {}
    last_planner = dict(planner_history[-1]) if planner_history else {}
    source_lifecycle = [row for row in evidence.get("source_lifecycle") or [] if isinstance(row, dict)]
    current_belief = dict(blackboard.get("current_belief") or {})
    route_objective_summary = dict(blackboard.get("route_objective_summary") or {})
    source_stage_counts: dict[str, int] = {}
    for row in source_lifecycle:
        stage_name = str(row.get("stage") or "unknown")
        source_stage_counts[stage_name] = source_stage_counts.get(stage_name, 0) + 1
    return {
        "schema_version": "agent_blackboard_step_summary.v1",
        "step_index": int(step_index),
        "stage": stage,
        "round_index": round_index,
        "action_id": action_id,
        "action_type": action_type,
        "case_id": str(blackboard.get("case_id") or ""),
        "detail": dict(detail or {}),
        "budget_state": budget,
        "counts": {
            "action_history": len(history),
            "planner_history": len(planner_history),
            "source_candidates": len(evidence.get("source_candidates") or []),
            "source_refs": len(evidence.get("source_refs") or []),
            "source_lifecycle": len(source_lifecycle),
            "exact_rows": len(evidence.get("exact_rows") or []),
            "pdf_structure_evidence": len(evidence.get("pdf_structure_evidence") or []),
            "visual_chains": len(evidence.get("visual_chains") or []),
            "structure_resolution_tasks": len(evidence.get("structure_resolution_tasks") or []),
            "route_failures": len(blackboard.get("route_failures") or []),
            "blocked_directions": len(current_belief.get("blocked_directions") or blackboard.get("blocked_directions") or []),
            "next_action_bias": len(current_belief.get("next_action_bias") or blackboard.get("next_action_bias") or []),
            "bridge_tasks": len(blackboard.get("bridge_tasks") or []),
            "route_objectives": len(route_objective_summary.get("objectives") or blackboard.get("route_objectives") or []),
            "endpoint_candidates": len(blackboard.get("endpoint_candidates") or []),
            "semisynthesis_anchors": len(blackboard.get("semisynthesis_anchors") or []),
            "reaction_idea_cards": len(blackboard.get("reaction_idea_cards") or []),
            "retrosynthetic_proposals": len(blackboard.get("retrosynthetic_proposals") or []),
            "recursive_hypothesis_tasks": len(blackboard.get("recursive_hypothesis_tasks") or []),
            "artifact_refs": len(blackboard.get("artifact_refs") or {}),
        },
        "source_lifecycle_stage_counts": source_stage_counts,
        "last_planner": {
            "mode": str(last_planner.get("mode") or ""),
            "action_types": [str(item) for item in last_planner.get("action_types") or []],
            "fallback_used": bool((last_planner.get("codex_action_planner") or {}).get("fallback_used")),
            "status": str((last_planner.get("codex_action_planner") or {}).get("status") or ""),
        },
        "last_action": {
            "action_id": str(last_action.get("action_id") or ""),
            "action_type": str(last_action.get("action_type") or ""),
            "useful_artifact": bool(last_action.get("useful_artifact")),
            "delta": dict(last_action.get("delta") or {}),
        },
        "final_verdict": dict(blackboard.get("final_verdict") or {}),
    }


def _stop_on_problem_action_batch_reason(action_batch: dict[str, Any], validation: dict[str, Any]) -> str:
    if not validation.get("accepted"):
        return "action_batch_validation_rejected:" + ",".join(str(item) for item in validation.get("reasons") or [])
    planner_meta = dict(action_batch.get("codex_action_planner") or {})
    if planner_meta.get("fallback_used"):
        return f"codex_action_planner_fallback_used:{planner_meta.get('fallback_reason') or ''}"
    if not action_batch.get("actions"):
        return "action_batch_empty"
    return ""


def _stop_on_problem_action_result_reason(
    *,
    action: dict[str, Any],
    action_result: dict[str, Any],
    history_record: dict[str, Any],
) -> str:
    action_type = str(action.get("action_type") or "")
    if action_type == "stop_unresolved":
        return ""
    if action_result.get("accepted") is False:
        reasons = ",".join(str(item) for item in action_result.get("reasons") or history_record.get("reasons") or [])
        return f"agent_action_rejected:{action_type}:{reasons}"
    if history_record.get("stale") or not history_record.get("useful_artifact"):
        reasons = ",".join(str(item) for item in history_record.get("reasons") or action_result.get("reasons") or [])
        return f"agent_action_no_useful_artifact:{action_type}:{reasons}"
    return ""


def _record_stop_on_problem(
    blackboard: dict[str, Any],
    *,
    run_dir: Path,
    round_index: int,
    reason: str,
    action_type: str,
) -> dict[str, Any]:
    board = dict(blackboard)
    belief = dict(board.get("current_belief") or {})
    notes = list(belief.get("planner_notes") or [])
    notes.append(
        {
            "schema_version": "agentic_stop_on_problem_note.v1",
            "round_index": int(round_index),
            "action_type": str(action_type or ""),
            "reason": str(reason or ""),
            "requires_user_discussion": True,
        }
    )
    belief["planner_notes"] = notes
    belief["stop_on_problem"] = {
        "schema_version": "agentic_stop_on_problem.v1",
        "round_index": int(round_index),
        "action_type": str(action_type or ""),
        "reason": str(reason or ""),
    }
    board["current_belief"] = belief
    append_jsonl(
        run_dir / "decision_trace.jsonl",
        {
            "stage": "stop_on_problem",
            "round_index": int(round_index),
            "action_type": str(action_type or ""),
            "reason": str(reason or ""),
        },
    )
    return board


def _refresh_multisource_route_consensus(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    codex_campaign_config: RetrosynthesisTeamConfig | None = None,
) -> dict[str, Any]:
    """Rebuild the advisory graph after every planner/source channel has run."""
    board = dict(blackboard)
    existing_graph = dict(board.get("route_consensus_graph") or {})
    max_depth = int((existing_graph.get("limits") or {}).get("max_depth") or 2)
    rebuild = rebuild_consensus_graph_from_blackboard(board, max_depth=max_depth)
    rebuild_path = state.run_dir / "route_consensus_rebuild.json"
    consensus_path = state.run_dir / "route_consensus_fused.json"
    graph_path = state.run_dir / "route_consensus_graph_fused.json"
    write_json(rebuild_path, rebuild)
    state.artifacts["route_consensus_rebuild"] = rebuild
    refs = dict(board.get("artifact_refs") or {})
    refs["route_consensus_rebuild"] = str(rebuild_path)
    external_admission_materials = dict(
        rebuild.get("admission_receipts") or {}
    )
    has_external_admission_material = any(
        isinstance(material, dict) and bool(material)
        for values in external_admission_materials.values()
        if isinstance(values, list)
        for material in values
    )
    external_admission_material_count = sum(
        1
        for values in external_admission_materials.values()
        if isinstance(values, list)
        for material in values
        if isinstance(material, dict) and bool(material)
    )
    if not rebuild.get("accepted") and not has_external_admission_material:
        board, existing_graph, refs = _record_frontier_ledger_projection(
            state=state,
            blackboard=board,
            graph=existing_graph,
            refs=refs,
            codex_campaign_config=codex_campaign_config,
        )
        if existing_graph:
            board["route_consensus_graph"] = existing_graph
            state.artifacts["route_consensus_graph"] = existing_graph
        board["artifact_refs"] = refs
        return _checkpoint_blackboard_event(
            board,
            state=state,
            stage="consensus_refresh_checkpoint",
            metadata={
                "rebuild_accepted": False,
                "existing_graph_preserved": bool(existing_graph),
            },
        )

    consensus = dict(rebuild.get("consensus") or {})
    graph = dict(rebuild.get("graph") or {})
    overlay = dict(graph.get("v2_overlay") or {})
    stock_provider_results = _codex_campaign_stock_provider_results(board)
    trusted_stock_providers: dict[str, Any] = {}
    stock_provider_construction_reasons: tuple[str, ...] = ()
    if codex_campaign_config is not None:
        (
            trusted_stock_providers,
            stock_provider_construction_reasons,
        ) = build_trusted_stock_provider_instances(
            stock_snapshots=codex_campaign_config.stock_snapshots,
            benchmark_catalog_artifact=(
                codex_campaign_config.benchmark_stock_catalog_artifact
            ),
            benchmark_catalog_sha256=(
                codex_campaign_config.benchmark_stock_catalog_sha256
            ),
            benchmark_catalog_name=(
                codex_campaign_config.benchmark_stock_catalog_name
            ),
        )
    stock_host_replay = _replay_codex_campaign_stock_provider_results(
        stock_provider_results,
        trusted_stock_provider_instances=trusted_stock_providers,
        provider_construction_reasons=stock_provider_construction_reasons,
    )
    stock_host_replay_path = state.run_dir / "stock_provider_host_replay.json"
    write_json(stock_host_replay_path, stock_host_replay)
    state.artifacts["stock_provider_host_replay"] = stock_host_replay
    refs["stock_provider_host_replay"] = str(stock_host_replay_path)
    stock_closed_smiles = list(stock_host_replay["closed_smiles"])
    edge_verification = verify_codex_consensus_graph(
        graph,
        exact_rows=[
            dict(row)
            for row in (board.get("literature_evidence") or {}).get("exact_rows") or []
            if isinstance(row, dict)
        ],
        stock_closed_smiles=stock_closed_smiles,
        enable_optional_rxnmapper=str(
            os.environ.get("AUTOPLANNER_ENABLE_RXNMAPPER", "1")
        ).strip().lower()
        not in {"0", "false", "no", "off"},
        work_dir=state.run_dir / ".autoplanner",
    )
    edge_verification_path = state.run_dir / "codex_edge_verification_report.json"
    write_json(edge_verification_path, edge_verification)
    state.artifacts["codex_edge_verification"] = edge_verification
    refs["codex_edge_verification"] = str(edge_verification_path)
    edge_evidence_binding_sets = project_edge_evidence_binding_sets(
        edge_verification
    )
    edge_evidence_binding_sets_path = (
        state.run_dir / "edge_evidence_binding_sets.json"
    )
    write_json(edge_evidence_binding_sets_path, edge_evidence_binding_sets)
    state.artifacts["edge_evidence_binding_sets"] = edge_evidence_binding_sets
    refs["edge_evidence_binding_sets"] = str(edge_evidence_binding_sets_path)
    graph["edge_evidence_binding_sets"] = edge_evidence_binding_sets
    graph["codex_edge_verification_summary"] = {
        key: edge_verification.get(key)
        for key in (
            "edge_count",
            "materialized_edge_count",
            "mapped_edge_count",
            "reaction_validated_edge_count",
            "proof_closed_edge_count",
            "trusted_precedent_binding_count",
            "corroborated_edge_count",
            "work_cache",
            "content_sha256",
        )
    }
    proof = dict(board.get("parent_route_proof") or {})
    proof_evidence = dict(proof.get("proof_evidence") or {})
    target_smiles = str((board.get("target_profile") or {}).get("target_smiles") or "")
    solved_verifier = (
        dict(proof_evidence.get("parent_verifier") or {})
        if is_solved_parent_route_proof(proof, expected_target_smiles=target_smiles)
        else {}
    )
    # A verifier-owned proof bank may contain several replayable, stock-closed
    # routes even when none has yet become the authoritative parent proof.  The
    # binding layer replays every bank entry (and strictly replays a legacy best
    # route) before granting an edge or stock binding, so proposal payloads can
    # never promote themselves here.
    verifier = _portfolio_verifier_bundle(
        artifacts=state.artifacts,
        parent_proof=proof,
        solved_parent_verifier=solved_verifier,
    )
    bindings = derive_portfolio_bindings(
        overlay,
        verifier,
        supplemental_edge_verification_reports=[edge_verification],
        supplemental_stock_provider_results=stock_provider_results,
        trusted_stock_provider_instances=trusted_stock_providers,
    )
    portfolio = solve_diverse_routes(
        overlay,
        stock_molecule_ids=bindings["stock_molecule_ids"],
        edge_proof_levels=bindings["edge_proof_levels"],
        stock_bindings=bindings["stock_bindings"],
        top_k=5,
    )
    # Keep the portfolio payload immutable after its content hash is computed.
    # Runtime proof/stock bindings are derived material and therefore live next
    # to, rather than inside, the content-addressed portfolio report.
    graph["route_portfolio"] = portfolio.to_dict()
    graph["route_portfolio_bindings"] = bindings
    graph["route_replacement_catalog"] = validate_portfolio_replacements(
        overlay,
        portfolio=graph["route_portfolio"],
        stock_molecule_ids=bindings["stock_molecule_ids"],
        edge_proof_levels=bindings["edge_proof_levels"],
        stock_bindings=bindings["stock_bindings"],
    )

    # Proposal generation and proof closure are deliberately separate phases.
    # Reconcile the fused graph after every refresh so newly materialized edge
    # candidates can close durable proof requests even when no Codex proposal
    # worker is invoked in this round.  The public reconciliation API replays
    # the current host verifier and the existing campaign queue; it consumes no
    # proposal/attempt budget and does not trust caller-supplied proof flags.
    proof_reconciliation: dict[str, Any] | None = None
    team_snapshot = dict(board.get("codex_agent_team") or {})
    codex_team_accepted = bool(
        team_snapshot and team_snapshot.get("accepted") is True
    )
    reconciliation_trigger = (
        "accepted_codex_team"
        if codex_team_accepted
        else "host_external_admission_material"
        if has_external_admission_material
        else "none"
    )
    if codex_team_accepted or has_external_admission_material:
        try:
            proof_reconciliation = reconcile_codex_campaign_proof_state(
                graph=graph,
                run_dir=state.run_dir,
                case_id=str(board.get("case_id") or state.preflight.get("case_id") or ""),
                reaction_proof_reports=[edge_verification],
                stock_evidence=stock_provider_results,
                required_proof_level=2,
                campaign_config=codex_campaign_config,
                external_hyperedge_admission_receipts=(
                    external_admission_materials
                ),
            )
        except Exception as exc:
            proof_reconciliation = {
                "schema_version": "codex_campaign_proof_reconciliation.v1",
                "accepted": False,
                "graph_complete": False,
                "proposal_runner_invoked": False,
                "expansion_budget_consumed": 0,
                "reasons": [
                    f"proof_reconciliation_error:{type(exc).__name__}:{exc}"
                ],
            }
            board.setdefault("safety_flags", []).append(
                "codex_campaign_proof_reconciliation_failed"
            )
        proof_reconciliation_path = (
            state.run_dir / "codex_campaign_proof_reconciliation.json"
        )
        write_json(proof_reconciliation_path, proof_reconciliation)
        state.artifacts["codex_campaign_proof_reconciliation"] = proof_reconciliation
        refs["codex_campaign_proof_reconciliation"] = str(
            proof_reconciliation_path
        )
        graph["codex_campaign_proof_reconciliation_summary"] = {
            "accepted": proof_reconciliation.get("accepted") is True,
            "graph_complete": proof_reconciliation.get("graph_complete") is True,
            **{
                field: proof_reconciliation.get(field)
                for field in (
                    "closure_objective",
                    "exploration_mode",
                    "route_solved",
                    "campaign_search_complete",
                    "all_reaction_edges_closed",
                    "all_benchmark_leaves_closed",
                    "all_procurement_leaves_closed",
                )
            },
            "proposal_runner_invoked": proof_reconciliation.get(
                "proposal_runner_invoked"
            )
            is True,
            "expansion_budget_consumed": int(
                proof_reconciliation.get("expansion_budget_consumed") or 0
            ),
            "durable_accepted_expansion_count": int(
                proof_reconciliation.get(
                    "durable_accepted_expansion_count"
                )
                or 0
            ),
            "admitted_external_expansion_count": int(
                proof_reconciliation.get(
                    "admitted_external_expansion_count"
                )
                or 0
            ),
            "canonical_input_expansion_event_count": int(
                proof_reconciliation.get(
                    "canonical_input_expansion_event_count"
                )
                or 0
            ),
            "canonical_reaction_edge_count": int(
                proof_reconciliation.get("canonical_reaction_edge_count")
                or 0
            ),
            "canonical_expansion_count": int(
                proof_reconciliation.get("canonical_expansion_count") or 0
            ),
            "canonical_expansion_count_semantics": str(
                proof_reconciliation.get(
                    "canonical_expansion_count_semantics"
                )
                or "deprecated_alias_of_canonical_input_expansion_event_count"
            ),
            "trigger": reconciliation_trigger,
            "codex_team_present": bool(team_snapshot),
            "codex_team_accepted": codex_team_accepted,
            "external_admission_material_edge_count": (
                external_admission_material_count
            ),
            "external_admission_material_count": (
                external_admission_material_count
            ),
            "open_reaction_proof_count": len(
                proof_reconciliation.get("open_reaction_proofs") or []
            ),
            "frontier_completeness": dict(
                proof_reconciliation.get("frontier_completeness") or {}
            ),
        }
    caller_advisory_graph = graph
    canonical_graph = (
        dict(proof_reconciliation.get("canonical_route_consensus_graph") or {})
        if proof_reconciliation is not None
        and proof_reconciliation.get("accepted") is True
        else {}
    )
    if proof_reconciliation is not None:
        # Once a canonical reconciliation is attempted, an absent/failed
        # canonical return must never promote the mutable caller graph into
        # ledger authority. Rebuild an empty identity-bound graph instead;
        # durable journal/commit replay can restore its edges on a later pass.
        ledger_input_graph = canonical_graph or assemble_route_consensus_graph(
            [],
            case_id=str(caller_advisory_graph.get("case_id") or ""),
            target_smiles=str(
                caller_advisory_graph.get("target_smiles") or ""
            ),
            max_depth=max(
                1,
                int(
                    (caller_advisory_graph.get("limits") or {}).get(
                        "max_depth"
                    )
                    or max_depth
                ),
            ),
        )
    else:
        ledger_input_graph = caller_advisory_graph
    board, ledger_graph, refs = _record_frontier_ledger_projection(
        state=state,
        blackboard=board,
        graph=ledger_input_graph,
        refs=refs,
        proof_reconciliation=proof_reconciliation,
        codex_campaign_config=codex_campaign_config,
    )
    if canonical_graph:
        canonical_graph_path = (
            state.run_dir / "route_consensus_graph_canonical.json"
        )
        write_json(canonical_graph_path, ledger_graph)
        refs["canonical_route_consensus_graph"] = str(canonical_graph_path)
        board["canonical_route_consensus_graph"] = ledger_graph
        state.artifacts["canonical_route_consensus_graph"] = ledger_graph
        board["codex_campaign_authority_projection"] = {
            "schema_version": "codex_campaign_authority_projection.v1",
            "accepted": True,
            "reconciliation_trigger": reconciliation_trigger,
            "codex_team_present": bool(team_snapshot),
            "codex_team_accepted": codex_team_accepted,
            "campaign_identity_sha256": str(
                proof_reconciliation.get("campaign_identity_sha256") or ""
            ),
            "campaign_policy_sha256": str(
                proof_reconciliation.get("campaign_policy_sha256") or ""
            ),
            "campaign_policy_ref": str(
                proof_reconciliation.get("campaign_policy_ref") or ""
            ),
            "canonical_route_consensus_graph_ref": str(canonical_graph_path),
            "frontier_ledger_ref": str(
                refs.get("frontier_ledger") or ""
            ),
            "proposal_runner_invoked": (
                proof_reconciliation.get("proposal_runner_invoked") is True
            ),
            "expansion_budget_consumed": int(
                proof_reconciliation.get("expansion_budget_consumed") or 0
            ),
            "durable_accepted_expansion_count": int(
                proof_reconciliation.get(
                    "durable_accepted_expansion_count"
                )
                or 0
            ),
            "admitted_external_expansion_count": int(
                proof_reconciliation.get(
                    "admitted_external_expansion_count"
                )
                or 0
            ),
            "canonical_input_expansion_event_count": int(
                proof_reconciliation.get(
                    "canonical_input_expansion_event_count"
                )
                or 0
            ),
            "canonical_reaction_edge_count": int(
                proof_reconciliation.get("canonical_reaction_edge_count")
                or 0
            ),
            "canonical_expansion_count": int(
                proof_reconciliation.get("canonical_expansion_count") or 0
            ),
            "canonical_expansion_count_semantics": str(
                proof_reconciliation.get(
                    "canonical_expansion_count_semantics"
                )
                or "deprecated_alias_of_canonical_input_expansion_event_count"
            ),
            "semantics": {
                "projection_only": True,
                "durable_campaign_files_remain_authority": True,
                "caller_advisory_graph_is_not_authority": True,
                "external_admissions_require_current_host_replay": True,
            },
        }
        campaign_authority_projection_path = (
            state.run_dir / "codex_campaign_authority_projection.json"
        )
        write_json(
            campaign_authority_projection_path,
            board["codex_campaign_authority_projection"],
        )
        refs["codex_campaign_authority_projection"] = str(
            campaign_authority_projection_path
        )
        state.artifacts["codex_campaign_authority_projection"] = dict(
            board["codex_campaign_authority_projection"]
        )
    # Keep unsupported analogy/template rows available as caller-advisory L0
    # suggestions, but never feed them back into the authority ledger above.
    graph = caller_advisory_graph
    board["route_consensus_graph"] = graph
    board["caller_advisory_route_consensus_graph"] = graph
    state.artifacts["caller_advisory_route_consensus_graph"] = graph
    rebuild["graph"] = graph
    write_json(rebuild_path, rebuild)
    write_json(consensus_path, consensus)
    write_json(graph_path, graph)
    refs["route_consensus"] = str(consensus_path)
    refs["route_consensus_graph"] = str(graph_path)
    board["artifact_refs"] = refs
    board["route_consensus"] = consensus
    board["route_consensus_graph"] = graph
    state.artifacts["route_consensus"] = consensus
    state.artifacts["route_consensus_graph"] = graph
    existing = {
        str(row.get("proposal_id") or ""): dict(row)
        for row in board.get("retrosynthetic_proposals") or []
        if isinstance(row, dict) and str(row.get("proposal_id") or "")
    }
    for row in consensus_to_blackboard_proposals(consensus):
        if str(row.get("proposal_id") or ""):
            existing[str(row["proposal_id"])] = dict(row)
    board["retrosynthetic_proposals"] = list(existing.values())

    team = dict(board.get("codex_agent_team") or {})
    if team:
        team["route_consensus"] = consensus
        team["route_consensus_ref"] = str(consensus_path)
        team["route_consensus_graph"] = graph
        team["route_consensus_graph_ref"] = str(graph_path)
        campaign_expansions = [
            dict(row)
            for row in team_snapshot.get("route_consensus_expansions") or []
            if isinstance(row, dict)
        ]
        fused_projection_expansions = [
            dict(row)
            for row in rebuild.get("expansions") or []
            if isinstance(row, dict)
        ]
        team["route_consensus_expansions"] = _durable_first_expansion_union(
            campaign_expansions,
            fused_projection_expansions,
        )
        team["route_consensus_expansion_projection"] = {
            "schema_version": "route_consensus_expansion_projection.v1",
            "campaign_expansion_ids": [
                str(row.get("expansion_id") or "") for row in campaign_expansions
            ],
            "fused_projection_expansion_ids": [
                str(row.get("expansion_id") or "")
                for row in fused_projection_expansions
            ],
            "union_expansion_ids": [
                str(row.get("expansion_id") or "")
                for row in team["route_consensus_expansions"]
            ],
            "semantics": {
                "campaign_commits_are_accepted_budget_authority": True,
                "fused_expansions_are_graph_projection_only": True,
                "durable_campaign_rows_win_identity_collisions": True,
            },
        }
        team["post_run_multisource_rebuild_ref"] = str(rebuild_path)
        if proof_reconciliation is not None:
            campaign = dict(team.get("campaign") or {})
            if proof_reconciliation.get("accepted") is not True:
                # The orchestration-owned accepted campaign remains the resume
                # authority.  A failed controller refresh is a separate
                # projection event and must not erase accepted counts/state.
                board["codex_agent_team"] = team_snapshot
                state.artifacts["codex_retrosynthesis_team"] = team_snapshot
                board = _write_codex_controller_projection(
                    state=state,
                    blackboard=board,
                    stage="proof_reconciliation_failed",
                    failure=proof_reconciliation,
                    prior_accepted_team_preserved=True,
                )
                return _checkpoint_blackboard_event(
                    board,
                    state=state,
                    stage="consensus_refresh_checkpoint",
                    metadata={
                        "rebuild_accepted": True,
                        "proof_reconciliation_accepted": False,
                        "prior_accepted_team_preserved": True,
                    },
                )
            campaign["reaction_proof_state"] = dict(
                proof_reconciliation.get("reaction_proof_state") or {}
            )
            campaign["reaction_proof_state_ref"] = str(
                proof_reconciliation.get("reaction_proof_state_ref") or ""
            )
            campaign["open_reaction_proofs"] = list(
                proof_reconciliation.get("open_reaction_proofs") or []
            )
            campaign["frontier_completeness"] = dict(
                proof_reconciliation.get("frontier_completeness") or {}
            )
            if isinstance(proof_reconciliation.get("frontier_queue"), dict):
                campaign["frontier_queue"] = dict(
                    proof_reconciliation["frontier_queue"]
                )
            campaign["queue_state_counts"] = dict(
                proof_reconciliation.get("queue_state_counts") or {}
            )
            campaign["remaining_frontier"] = list(
                proof_reconciliation.get("remaining_frontier") or []
            )
            campaign["proposal_graph_exhausted"] = (
                proof_reconciliation.get("proposal_graph_exhausted") is True
            )
            campaign["reconciliation_graph_complete"] = (
                proof_reconciliation.get("graph_complete") is True
            )
            ledger_summary = dict(board.get("frontier_ledger_summary") or {})
            ledger_core = dict(ledger_summary.get("summary") or {})
            for field in (
                "closure_objective",
                "exploration_mode",
                "route_solved",
                "campaign_search_complete",
                "all_reaction_edges_closed",
                "all_benchmark_leaves_closed",
                "all_procurement_leaves_closed",
                "all_in_house_leaves_closed",
                "any_in_house_route_closed",
                "all_explored_in_house_closed",
                "selected_objective_all_explored_closed",
            ):
                campaign[field] = ledger_summary.get(field)
            campaign["graph_complete"] = (
                campaign.get("campaign_search_complete") is True
            )
            campaign["resumable"] = bool(
                not campaign["campaign_search_complete"]
                and (
                    campaign["remaining_frontier"]
                    or campaign["queue_state_counts"].get("pending", 0)
                    or campaign["queue_state_counts"].get("retry_wait", 0)
                    or campaign["open_reaction_proofs"]
                    or ledger_core.get("proposal_pending_molecule_count", 0)
                    or ledger_core.get("work_pending_molecule_count", 0)
                    or ledger_core.get("stock_pending_leaf_count", 0)
                    or ledger_core.get("reaction_proof_pending_edge_count", 0)
                    or ledger_core.get("dependency_pending_edge_count", 0)
                )
            )
            campaign["proof_reconciliation_ref"] = str(
                refs.get("codex_campaign_proof_reconciliation") or ""
            )
            team["campaign"] = campaign
            team["proof_closed"] = campaign["graph_complete"]
        board["codex_agent_team"] = team
        state.artifacts["codex_retrosynthesis_team"] = team
        board = _write_codex_controller_projection(
            state=state,
            blackboard=board,
            stage="multisource_refresh",
        )
    return _checkpoint_blackboard_event(
        board,
        state=state,
        stage="consensus_refresh_checkpoint",
        metadata={
            "rebuild_accepted": True,
            "proof_reconciliation_accepted": (
                proof_reconciliation.get("accepted") is True
                if proof_reconciliation is not None
                else None
            ),
            "consensus_proposal_count": len(consensus.get("proposals") or []),
            "graph_step_count": len(graph.get("steps") or []),
        },
    )


def _durable_first_expansion_union(
    campaign_expansions: list[dict[str, Any]],
    fused_projection_expansions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source, values in (
        ("durable_campaign_commit", campaign_expansions),
        ("fused_graph_projection", fused_projection_expansions),
    ):
        for raw in values:
            row = dict(raw)
            expansion_id = str(row.get("expansion_id") or "")
            if not expansion_id:
                expansion_id = "projection:" + _stable_json_digest(row)
                row["expansion_id"] = expansion_id
            row.setdefault("controller_projection_source", source)
            rows.setdefault(expansion_id, row)
    return list(rows.values())


def _record_frontier_ledger_projection(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    graph: dict[str, Any],
    refs: dict[str, Any],
    proof_reconciliation: dict[str, Any] | None = None,
    codex_campaign_config: RetrosynthesisTeamConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Write the complete-graph closure projection and bind every consumer.

    The ledger deliberately receives the graph's authoritative ``steps`` and
    never the bounded ``route_hypotheses`` presentation projection.  Queue
    work, stock boundaries, and replayed reaction proof remain independent
    inputs.  The compact summary is digest-bound to the full ledger, but only
    the validated ledger is closure authority.
    """

    board = dict(blackboard)
    projected_graph = dict(graph)
    reconciliation = dict(proof_reconciliation or {})
    team = dict(board.get("codex_agent_team") or {})
    campaign = dict(team.get("campaign") or {})
    reconciliation_attempted = proof_reconciliation is not None
    reconciliation_accepted = bool(
        reconciliation_attempted and reconciliation.get("accepted") is True
    )
    if reconciliation_attempted:
        # A failed current refresh invalidates closure for this projection.  Do
        # not silently substitute an older queue/proof snapshot.
        frontier_queue = (
            dict(reconciliation["frontier_queue"])
            if reconciliation_accepted
            and isinstance(reconciliation.get("frontier_queue"), dict)
            else {}
        )
        reaction_proof_state = (
            dict(reconciliation["reaction_proof_state"])
            if reconciliation_accepted
            and isinstance(reconciliation.get("reaction_proof_state"), dict)
            else {}
        )
    else:
        frontier_queue = dict(campaign.get("frontier_queue") or {})
        reaction_proof_state = dict(campaign.get("reaction_proof_state") or {})
    trusted_stock_providers: dict[str, Any] = {}
    stock_provider_construction_reasons: tuple[str, ...] = ()
    if codex_campaign_config is not None:
        (
            trusted_stock_providers,
            stock_provider_construction_reasons,
        ) = build_trusted_stock_provider_instances(
            stock_snapshots=codex_campaign_config.stock_snapshots,
            benchmark_catalog_artifact=(
                codex_campaign_config.benchmark_stock_catalog_artifact
            ),
            benchmark_catalog_sha256=(
                codex_campaign_config.benchmark_stock_catalog_sha256
            ),
            benchmark_catalog_name=(
                codex_campaign_config.benchmark_stock_catalog_name
            ),
        )
    ledger = project_frontier_ledger(
        projected_graph,
        frontier_queue,
        reaction_proof_state,
        required_reaction_proof_level=2,
        trusted_stock_provider_instances=trusted_stock_providers,
        campaign_policy_sha256=str(
            reconciliation.get("campaign_policy_sha256")
            or campaign.get("campaign_policy_sha256")
            or ""
        ),
    )
    ledger_path = state.run_dir / "frontier_ledger.json"
    write_json(ledger_path, ledger)
    ledger_ref = str(ledger_path)
    ledger_validation_reasons = validate_frontier_ledger(
        ledger,
        trusted_stock_provider_instances=trusted_stock_providers,
        expected_input_bindings=dict(ledger.get("input_bindings") or {}),
    )
    ledger_core = dict(ledger.get("summary") or {})
    input_validation = dict(ledger.get("input_validation") or {})
    input_valid = all(
        isinstance(input_validation.get(key), dict)
        and input_validation[key].get("valid") is True
        for key in ("graph", "frontier_queue", "reaction_proof_state")
    )
    campaign_policy = dict(campaign.get("campaign_policy") or {})
    closure_objective = str(
        campaign_policy.get("closure_objective")
        or campaign.get("closure_objective")
        or (
            codex_campaign_config.closure_objective
            if codex_campaign_config is not None
            else "benchmark_search"
        )
    )
    exploration_mode = str(
        campaign_policy.get("exploration_mode")
        or campaign.get("exploration_mode")
        or (
            codex_campaign_config.exploration_mode
            if codex_campaign_config is not None
            else "exhaustive"
        )
    )
    closure_status = campaign_closure_status(
        ledger,
        authoritative=bool(not ledger_validation_reasons and input_valid),
        closure_objective=closure_objective,
        exploration_mode=exploration_mode,
    )
    ledger_digest = str(ledger.get("content_sha256") or "")
    semantic_summary = {
        "schema_version": "frontier_ledger_summary.v1",
        # ``content_sha256`` is intentionally the full-ledger digest, not a
        # detached digest of this convenience envelope.
        "content_sha256": ledger_digest,
        "frontier_ledger_content_sha256": ledger_digest,
        "source_ref": ledger_ref,
        "input_valid": input_valid,
        "ledger_validation_accepted": not ledger_validation_reasons,
        "ledger_validation_reasons": list(ledger_validation_reasons),
        "reconciliation_attempted": reconciliation_attempted,
        "reconciliation_accepted": reconciliation_accepted,
        "stock_provider_construction_reasons": list(
            stock_provider_construction_reasons
        ),
        **closure_status,
        "any_route_closed": ledger_core.get("any_route_closed") is True,
        "all_explored_graph_closed": (
            ledger_core.get("all_explored_graph_closed") is True
        ),
        "any_benchmark_route_closed": (
            ledger_core.get("any_benchmark_route_closed") is True
        ),
        "all_explored_benchmark_closed": (
            ledger_core.get("all_explored_benchmark_closed") is True
        ),
        "any_procurement_route_closed": (
            ledger_core.get("any_procurement_route_closed") is True
        ),
        "all_explored_procurement_closed": (
            ledger_core.get("all_explored_procurement_closed") is True
        ),
        "summary": ledger_core,
        "semantics": {
            "full_reachable_hypergraph_projection": True,
            "route_hypotheses_are_not_closure_authority": True,
            "summary_is_digest_bound_locator_not_standalone_authority": True,
            "proposal_work_stock_reaction_proof_dependencies_are_orthogonal": True,
            "generic_any_all_mean_benchmark_search_closure": True,
            "procurement_closure_has_an_independent_fixed_point": True,
        },
    }
    board["frontier_ledger"] = ledger
    board["frontier_ledger_summary"] = semantic_summary
    board["route_consensus_graph"] = projected_graph
    refs = dict(refs)
    refs["frontier_ledger"] = ledger_ref
    board["artifact_refs"] = refs
    state.artifacts["frontier_ledger"] = ledger

    if team:
        team["frontier_ledger_ref"] = ledger_ref
        team["frontier_ledger_summary"] = semantic_summary
        campaign["frontier_ledger_ref"] = ledger_ref
        campaign["frontier_ledger_summary"] = semantic_summary
        # Preserve the campaign's pre-projection decision for diagnostics;
        # only the complete hypergraph fixed point may drive controller drain.
        campaign.setdefault(
            "reconciliation_graph_complete",
            campaign.get("graph_complete") is True,
        )
        campaign.update(closure_status)
        campaign["graph_complete"] = campaign["campaign_search_complete"] is True
        team["campaign"] = campaign
        team["proof_closed"] = campaign["graph_complete"]
        board["codex_agent_team"] = team
        state.artifacts["codex_retrosynthesis_team"] = team
        board = _write_codex_controller_projection(
            state=state,
            blackboard=board,
            stage="frontier_ledger_projection",
            failure=(
                reconciliation
                if reconciliation_attempted and not reconciliation_accepted
                else None
            ),
            prior_accepted_team_preserved=False,
        )

    append_jsonl(
        state.run_dir / "decision_trace.jsonl",
        {
            "stage": "frontier_ledger_projection",
            "frontier_ledger_ref": ledger_ref,
            "frontier_ledger_content_sha256": ledger_digest,
            "input_valid": input_valid,
            "ledger_validation_accepted": not ledger_validation_reasons,
            **closure_status,
            **{
                key: ledger_core.get(key)
                for key in (
                    "any_benchmark_route_closed",
                    "all_explored_benchmark_closed",
                    "any_procurement_route_closed",
                    "all_explored_procurement_closed",
                    "any_route_closed",
                    "all_explored_graph_closed",
                    "reachable_molecule_count",
                    "reachable_edge_count",
                    "proposal_pending_molecule_count",
                    "work_pending_molecule_count",
                    "stock_pending_leaf_count",
                    "reaction_proof_pending_edge_count",
                    "dependency_pending_edge_count",
                )
            },
        },
    )
    return board, projected_graph, refs


def _codex_campaign_stock_provider_results(
    blackboard: dict[str, Any],
) -> list[dict[str, Any]]:
    team = dict(blackboard.get("codex_agent_team") or {})
    campaign = dict(team.get("campaign") or {})
    queue = dict(campaign.get("frontier_queue") or {})
    rows: dict[str, dict[str, Any]] = {}
    for raw_job in queue.get("jobs") or []:
        if not isinstance(raw_job, dict):
            continue
        metadata = dict(raw_job.get("metadata") or {})
        candidates = [metadata.get("stock_audit")]
        observations = dict(metadata.get("stock_observations") or {})
        candidates.extend(
            dict(observation).get("provider_result")
            for observation in observations.get("current") or []
            if isinstance(observation, dict)
        )
        for raw_result in candidates:
            if not isinstance(raw_result, dict):
                continue
            result = dict(raw_result)
            key = str(result.get("content_hash") or _stable_json_digest(result))
            rows[key] = result
    return [rows[key] for key in sorted(rows)]


def _replay_codex_campaign_stock_provider_results(
    values: list[dict[str, Any]],
    *,
    trusted_stock_provider_instances: dict[str, Any],
    provider_construction_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Grant stock closure only after exact current-host provider replay."""

    replay_rows: list[dict[str, Any]] = []
    closed_smiles: set[str] = set()
    for index, raw in enumerate(values):
        result = dict(raw) if isinstance(raw, dict) else {}
        payload = dict(result.get("payload") or {})
        expected_smiles = str(payload.get("canonical_smiles") or "")
        binding, reasons = replay_stock_provider_result(
            result,
            expected_smiles=expected_smiles,
            trusted_provider_instances=trusted_stock_provider_instances,
        )
        accepted = bool(binding and not reasons)
        if accepted:
            closed_smiles.add(expected_smiles)
        replay_rows.append(
            {
                "index": index,
                "provider_id": str(result.get("provider_id") or ""),
                "provider_result_content_hash": str(
                    result.get("content_hash") or ""
                ),
                "canonical_smiles": expected_smiles,
                "accepted": accepted,
                "reasons": list(reasons),
                "authority_binding": binding if accepted else {},
            }
        )
    report = {
        "schema_version": "codex_campaign_stock_host_replay.v1",
        "input_result_count": len(values),
        "accepted_result_count": sum(row["accepted"] for row in replay_rows),
        "rejected_result_count": sum(not row["accepted"] for row in replay_rows),
        "closed_smiles": sorted(closed_smiles),
        "trusted_provider_ids": sorted(trusted_stock_provider_instances),
        "provider_construction_reasons": list(provider_construction_reasons),
        "replays": replay_rows,
        "semantics": {
            "serialized_accepted_flag_is_not_stock_authority": True,
            "current_host_provider_replay_required": True,
            "missing_provider_fails_open_for_search": True,
        },
    }
    report["content_sha256"] = _stable_json_digest(report)
    return report


def _stable_json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _campaign_search_complete(campaign: dict[str, Any]) -> bool:
    """Read the objective-aware terminal bit with legacy replay compatibility."""

    if "campaign_search_complete" in campaign:
        return campaign.get("campaign_search_complete") is True
    # Reports written before the closure-objective policy only exposed this
    # compatibility alias.  New projections always publish both fields.
    return campaign.get("graph_complete") is True


def _controller_codex_search_stop_reason(
    blackboard: dict[str, Any],
    *,
    exploration_mode: str = "exhaustive",
) -> str:
    """Return a closure-policy stop reason, never a generic solved-route claim."""

    campaign = dict(
        ((blackboard.get("codex_agent_team") or {}).get("campaign") or {})
    )
    if _campaign_search_complete(campaign):
        return "campaign_search_complete"
    mode = str(
        campaign.get("exploration_mode") or exploration_mode or "exhaustive"
    ).strip().lower()
    if mode != "first_solved":
        return ""
    if campaign.get("route_solved") is True:
        return "first_route_solved"
    if _parent_proof_accepted(blackboard):
        return "parent_proof_accepted_first_solved"
    return ""


def _controller_codex_search_should_stop(
    blackboard: dict[str, Any],
    *,
    exploration_mode: str = "exhaustive",
) -> bool:
    """Keep exhaustive campaigns running after the first accepted parent route."""

    return bool(
        _controller_codex_search_stop_reason(
            blackboard,
            exploration_mode=exploration_mode,
        )
    )


def _controller_evidence_stop_preserves_campaign(
    blackboard: dict[str, Any],
    *,
    exploration_mode: str = "exhaustive",
) -> bool:
    """Keep draining after a solved-route planner fast-path in exhaustive mode."""

    campaign = dict(
        ((blackboard.get("codex_agent_team") or {}).get("campaign") or {})
    )
    mode = str(
        campaign.get("exploration_mode") or exploration_mode or "exhaustive"
    ).strip().lower()
    route_milestone = bool(
        campaign.get("route_solved") is True
        or _parent_proof_accepted(blackboard)
    )
    return bool(
        mode == "exhaustive"
        and route_milestone
        and not _campaign_search_complete(campaign)
    )


def _codex_team_has_remaining_campaign_work(blackboard: dict[str, Any]) -> bool:
    """Return whether a proposal-expansion worker can make queue progress.

    Open reaction-proof requests are host materialization/reconciliation work;
    they must not trigger another Codex proposal invocation.  Likewise,
    ``remaining_frontier`` and ``proposal_graph_exhausted`` are derived search
    summaries, not leases.  The durable queue is the sole worker authority.
    """

    team = dict(blackboard.get("codex_agent_team") or {})
    if not team or team.get("accepted") is not True:
        return False
    campaign = dict(team.get("campaign") or {})
    if _campaign_search_complete(campaign):
        return False
    try:
        accepted = int(campaign.get("accepted_expansion_count") or 0)
        maximum = int(campaign.get("max_expansions") or 0)
    except (TypeError, ValueError):
        return False
    if maximum > 0 and accepted >= maximum:
        return False
    queue = dict(campaign.get("frontier_queue") or {})
    return any(
        str(row.get("state") or "") in {"pending", "retry_wait", "leased"}
        and dict(row.get("metadata") or {}).get("proposal_expansion_allowed")
        is not False
        for row in queue.get("jobs") or []
        if isinstance(row, dict)
    )


def _codex_campaign_progress_signature(blackboard: dict[str, Any]) -> str:
    """Hash only authoritative work/proof progress, not timestamps/UI fields."""

    team = dict(blackboard.get("codex_agent_team") or {})
    campaign = dict(team.get("campaign") or {})
    queue = dict(campaign.get("frontier_queue") or {})
    proof_state = dict(campaign.get("reaction_proof_state") or {})
    graph = dict(blackboard.get("route_consensus_graph") or {})
    payload = {
        "accepted_expansion_count": int(
            campaign.get("accepted_expansion_count") or 0
        ),
        "graph_complete": campaign.get("graph_complete") is True,
        "route_solved": campaign.get("route_solved") is True,
        "campaign_search_complete": _campaign_search_complete(campaign),
        "proposal_graph_exhausted": campaign.get("proposal_graph_exhausted")
        is True,
        "queue": sorted(
            (
                str(row.get("job_id") or ""),
                str(row.get("state") or ""),
                str(row.get("closure_kind") or ""),
                int(row.get("achieved_proof_level") or 0),
                int(row.get("attempt") or 0),
            )
            for row in queue.get("jobs") or []
            if isinstance(row, dict)
        ),
        "edge_proofs": sorted(
            (
                str(row.get("step_id") or ""),
                str(row.get("status") or ""),
                int(row.get("achieved_proof_level") or 0),
            )
            for row in proof_state.get("records") or []
            if isinstance(row, dict)
        ),
        "graph_steps": sorted(
            str(row.get("signature") or row.get("step_id") or "")
            for row in graph.get("steps") or []
            if isinstance(row, dict)
        ),
    }
    return _stable_json_digest(payload)


def _campaign_nonresumable_reason(campaign: dict[str, Any]) -> str:
    if _campaign_search_complete(campaign):
        return "campaign_search_complete"
    try:
        accepted = int(campaign.get("accepted_expansion_count") or 0)
        maximum = int(campaign.get("max_expansions") or 0)
    except (TypeError, ValueError):
        accepted = maximum = 0
    completeness = dict(campaign.get("frontier_completeness") or {})
    unresolved = list(completeness.get("unresolved_frontiers") or [])
    if maximum > 0 and accepted >= maximum:
        return "accepted_expansion_budget_exhausted_with_open_frontiers"
    if str(campaign.get("stop_reason") or "") == "frontier_retry_wait":
        return "awaiting_frontier_retry"
    if campaign.get("open_reaction_proofs") or any(
        isinstance(row, dict)
        and str(row.get("reason") or "").startswith("open_proof")
        for row in unresolved
    ):
        return "awaiting_reaction_proof_materialization"
    if campaign.get("proposal_graph_exhausted") is True:
        return "proposal_graph_exhausted_with_open_frontiers"
    return "no_resumable_campaign_work"


def _blackboard_literature_sources(
    blackboard: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in (blackboard.get("literature_evidence") or {}).get(
        "source_candidates"
    ) or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if row.get("placeholder_only") is True or str(
            row.get("access_status") or ""
        ).lower() == "placeholder_only":
            continue
        if not any(
            str(row.get(key) or "").strip()
            for key in ("source_ref", "doi", "url", "local_pdf")
        ):
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("source_ref") or ""),
            str(row.get("doi") or ""),
            str(row.get("url") or ""),
        )
    )
    return rows


def _codex_edge_reaction_proofs(
    artifacts: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return replay inputs, never detached self-reported proof booleans."""

    report = artifacts.get("codex_edge_verification")
    if not isinstance(report, dict):
        return {}
    proofs: dict[str, dict[str, Any]] = {}
    for raw in report.get("edge_verifications") or []:
        if not isinstance(raw, dict):
            continue
        step_id = str(raw.get("step_id") or "")
        materialized = raw.get("materialized_candidate")
        if step_id and isinstance(materialized, dict):
            # The campaign consumer replays this candidate through the current
            # host verifier and, when supplied, compares the nested step proof.
            # A detached reaction_step_proof has no materialization and must
            # never be able to promote itself.
            proofs[step_id] = dict(raw)
    return proofs


def _finalize_agentic_run(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    explicit_final: FinalVerdict | None = None,
    codex_campaign_config: RetrosynthesisTeamConfig | None = None,
) -> tuple[dict[str, Any], ArtifactBundle, FinalVerdict]:
    """Run the single audited closeout path for every controller exit."""
    board = dict(blackboard)
    board = _refresh_multisource_route_consensus(
        state=state,
        blackboard=board,
        codex_campaign_config=codex_campaign_config,
    )
    board.setdefault("artifact_refs", {})["agentic_run_audit"] = str(state.run_dir / "agentic_run_audit.json")
    board.setdefault("artifact_refs", {})["agentic_final_verdict_validation"] = str(state.run_dir / "agentic_final_verdict_validation.json")
    board.setdefault("artifact_refs", {})["agent_blackboard_snapshot"] = str(state.run_dir / "agent_blackboard_snapshot.json")
    board.setdefault("artifact_refs", {})["agentic_capability_audit"] = str(state.run_dir / "agentic_capability_audit.json")
    board.setdefault("artifact_refs", {})["hypothesis_only_retrosynthesis_report"] = str(
        state.run_dir / "hypothesis_only_retrosynthesis_report.json"
    )
    board.setdefault("artifact_refs", {})["hypothesis_execution_report"] = str(
        state.run_dir / "hypothesis_execution_report.json"
    )
    state.artifacts["agent_blackboard"] = board

    hypothesis_report_artifact = _record_hypothesis_only_retrosynthesis_report_artifact(
        state=state,
        blackboard=board,
    )
    state.artifacts["hypothesis_only_retrosynthesis_report"] = hypothesis_report_artifact
    _validate_and_record_typed_artifact(state, "hypothesis_only_retrosynthesis_report", hypothesis_report_artifact)

    hypothesis_execution_artifact = _record_hypothesis_execution_report_artifact(
        state=state,
        blackboard=board,
        hypothesis_report_artifact=hypothesis_report_artifact,
    )
    state.artifacts["hypothesis_execution_report"] = hypothesis_execution_artifact
    _validate_and_record_typed_artifact(state, "hypothesis_execution_report", hypothesis_execution_artifact)

    workflow_plan = _workflow_plan_from_actions(action_batches)
    preliminary_bundle = artifact_bundle_from_state(state=state, workflow_plan=workflow_plan, tool_calls=tool_calls)
    preliminary_bundle.validations = [*preliminary_bundle.validations, *validations]
    final = explicit_final or emit_agentic_final_verdict(
        blackboard=board,
        artifacts=state.artifacts,
        bundle=preliminary_bundle.to_dict(),
    )
    final.artifact_refs = dict(board.get("artifact_refs") or {})

    final_validation = _validate_agentic_final_verdict(
        final.to_dict(),
        blackboard=board,
        validations=[*state.validations, *validations],
    )
    if not final_validation.get("accepted"):
        final = _downgrade_invalid_agentic_final_verdict(final, final_validation)
        final.artifact_refs = dict(board.get("artifact_refs") or {})
        final_validation["corrected_final_verdict"] = final.to_dict()
        final_validation["corrected_validation"] = _validate_agentic_final_verdict(
            final.to_dict(),
            blackboard=board,
            validations=[*state.validations, *validations],
        )
        state.safety_flags.append("agentic_final_verdict_validation_failed")
    state.validations.append(final_validation)

    # Presentation is a projection of the corrected final verdict, not an
    # input to it. Rendering earlier allowed advisory branches to label
    # themselves as a final integrated solution before closeout validation.
    board["final_verdict"] = final.to_dict()
    route_forest_artifact = _record_route_forest_display_artifact(
        state=state,
        blackboard=board,
        codex_campaign_config=codex_campaign_config,
    )
    state.artifacts["route_forest_display"] = route_forest_artifact
    _commit_route_closeout_revision(
        state=state,
        blackboard=board,
        final_verdict=final,
        final_validation=final_validation,
    )
    # Rendering registers the forest/HTML (or error) refs on the board. Keep
    # every closeout projection, including invalid-input runs, on the same
    # authoritative reference set.
    final.artifact_refs = dict(board.get("artifact_refs") or {})
    board["final_verdict"] = final.to_dict()
    state.artifacts["agent_blackboard"] = board

    final_validation_artifact = _record_agentic_final_verdict_validation_artifact(
        state=state,
        blackboard=board,
        final_validation=final_validation,
    )
    state.artifacts["agentic_final_verdict_validation"] = final_validation_artifact
    _validate_and_record_typed_artifact(state, "agentic_final_verdict_validation", final_validation_artifact)

    state.artifacts["agent_blackboard"] = board
    blackboard_snapshot_artifact = _record_agent_blackboard_snapshot_artifact(
        state=state,
        blackboard=board,
    )
    state.artifacts["agent_blackboard_snapshot"] = blackboard_snapshot_artifact
    _validate_and_record_typed_artifact(state, "agent_blackboard_snapshot", blackboard_snapshot_artifact)

    capability_artifact = _record_agentic_capability_audit_artifact(
        state=state,
        blackboard=board,
        action_batches=action_batches,
        action_batch_validations=validations,
        typed_validations=list(state.validations),
        tool_calls=tool_calls,
        final_verdict=final.to_dict(),
        final_validation=final_validation,
    )
    state.artifacts["agentic_capability_audit"] = capability_artifact
    _validate_and_record_typed_artifact(state, "agentic_capability_audit", capability_artifact)

    state.artifacts["agent_blackboard"] = board
    audit_artifact = _record_agentic_run_audit_artifact(
        state=state,
        blackboard=board,
        action_batches=action_batches,
        validations=validations,
        typed_validations=list(state.validations),
        tool_calls=tool_calls,
        final_verdict=final.to_dict(),
    )
    state.artifacts["agentic_run_audit"] = audit_artifact
    _validate_and_record_typed_artifact(state, "agentic_run_audit", audit_artifact)

    bundle = artifact_bundle_from_state(state=state, workflow_plan=workflow_plan, tool_calls=tool_calls)
    bundle.validations = [*bundle.validations, *validations]
    write_json(state.run_dir / "artifact_bundle.json", bundle.to_dict())
    write_json(state.run_dir / "agent_blackboard.json", board)
    write_json(state.run_dir / "final_verdict.json", final.to_dict())
    append_jsonl(state.run_dir / "decision_trace.jsonl", {"stage": "final_verdict", "final_verdict": final.to_dict()})
    return board, bundle, final


def emit_agentic_final_verdict(
    *,
    blackboard: dict[str, Any],
    artifacts: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> FinalVerdict:
    proof = dict(blackboard.get("parent_route_proof") or artifacts.get("parent_route_proof") or {})
    case_id = str(blackboard.get("case_id") or (bundle or {}).get("case_id") or "target")
    if _blackboard_parent_proof_solved(blackboard, proof):
        return FinalVerdict(
            case_id=case_id,
            verdict="solved",
            route_status="solved",
            solved=True,
            stock_audit_passed=True,
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    latest_verdict = emit_final_verdict(bundle or {
        "case_id": case_id,
        "target_input": {},
        "preflight": {"accepted": True},
        "workflow_plan": {},
        "artifacts": artifacts,
        "validations": [],
    })
    if latest_verdict.solved:
        hypothesis_status = _hypothesis_route_status_from_artifacts(artifacts)
        return FinalVerdict(
            case_id=case_id,
            verdict="hypothesis_route_proposed" if hypothesis_status else "unresolved",
            reasons=["candidate_route_found_parent_proof_missing"],
            route_status=hypothesis_status or "candidate_route_found_parent_proof_missing",
            stock_audit_passed=bool(latest_verdict.stock_audit_passed),
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    if proof:
        hypothesis_status = _hypothesis_route_status_from_artifacts(artifacts)
        if hypothesis_status:
            return FinalVerdict(
                case_id=case_id,
                verdict="hypothesis_route_proposed",
                reasons=sorted(
                    set(
                        [
                            "hypothesis_only_retrosynthesis_available",
                            *[str(item) for item in proof.get("reasons") or latest_verdict.reasons],
                        ]
                    )
                ),
                route_status=hypothesis_status,
                solved=False,
                stock_audit_passed=False,
                artifact_refs=dict(blackboard.get("artifact_refs") or {}),
            )
        status = str(proof.get("route_status") or latest_verdict.route_status or "unresolved")
        invalid_solved_claim = status == "solved" and not _blackboard_parent_proof_solved(blackboard, proof)
        if invalid_solved_claim:
            status = "unresolved"
        verdict = "fake_closed_rejected" if status == "fake_closed_rejected" else (
            "partial_anchor_only_not_solved" if status == "partial_anchor_only_not_solved" else "unresolved"
        )
        return FinalVerdict(
            case_id=case_id,
            verdict=verdict,
            reasons=sorted(
                set(
                    [
                        *[str(item) for item in proof.get("reasons") or latest_verdict.reasons],
                        *(["invalid_parent_route_proof_contract"] if invalid_solved_claim else []),
                    ]
                )
            ),
            route_status=status,
            stock_audit_passed=False,
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    child_solved = bool((blackboard.get("current_belief") or {}).get("child_route_solved"))
    if child_solved:
        return FinalVerdict(
            case_id=case_id,
            verdict="unresolved",
            reasons=["child_target_solved_parent_proof_missing"],
            route_status="child_solved_parent_unresolved",
            stock_audit_passed=False,
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    reasons = [str(item) for item in latest_verdict.reasons or []]
    if not reasons:
        reasons = ["no_deterministic_parent_route_proof"]
    hypothesis_status = _hypothesis_route_status_from_artifacts(artifacts)
    if hypothesis_status:
        return FinalVerdict(
            case_id=case_id,
            verdict="hypothesis_route_proposed",
            reasons=sorted(set(reasons + ["hypothesis_only_retrosynthesis_available", "no_deterministic_parent_route_proof"])),
            route_status=hypothesis_status,
            solved=False,
            stock_audit_passed=False,
            artifact_refs=dict(blackboard.get("artifact_refs") or {}),
        )
    return FinalVerdict(
        case_id=case_id,
        verdict=latest_verdict.verdict if latest_verdict.verdict != "solved" else "unresolved",
        reasons=sorted(set(reasons + ["no_deterministic_parent_route_proof"])),
        route_status=latest_verdict.route_status or "unresolved",
        stock_audit_passed=False,
        artifact_refs=dict(blackboard.get("artifact_refs") or {}),
    )


def _hypothesis_report_available(artifacts: dict[str, Any]) -> bool:
    report = artifacts.get("hypothesis_only_retrosynthesis_report")
    if not isinstance(report, dict):
        return False
    payload = dict(report.get("payload") or report)
    return bool(
        payload.get("accepted")
        and int(payload.get("candidate_precursor_count") or 0) > 0
        and payload.get("no_solved_claim") is True
    )


def _hypothesis_route_status_from_artifacts(artifacts: dict[str, Any]) -> str:
    execution_status = _hypothesis_execution_status_from_artifacts(artifacts)
    if execution_status:
        return execution_status
    proof_bundle = artifacts.get("route_proof_bundle")
    if isinstance(proof_bundle, dict):
        payload = dict(proof_bundle.get("payload") or proof_bundle)
        proof_result = payload.get("result")
        proof_views = [dict(proof_result)] if isinstance(proof_result, dict) else []
        proof_views.append(payload)
        for proof in proof_views:
            status = str(proof.get("route_status") or "")
            if status and status not in {"solved", "unresolved"}:
                return status
            for row in proof.get("objective_proofs") or []:
                if not isinstance(row, dict):
                    continue
                objective_status = str(row.get("route_status") or "")
                if objective_status and objective_status not in {"solved", "unresolved"}:
                    return objective_status
            if proof.get("objective_proofs"):
                return "hypothesis_route_proposed"
    if _hypothesis_report_available(artifacts):
        return "hypothesis_route_proposed"
    return ""


def _hypothesis_execution_status_from_artifacts(artifacts: dict[str, Any]) -> str:
    execution = artifacts.get("hypothesis_execution_report")
    if not isinstance(execution, dict):
        return ""
    payload = dict(execution.get("payload") or execution)
    status = str(payload.get("route_status") or "")
    if not status or status == "no_hypothesis_candidates":
        return ""
    return status


def _validate_agentic_final_verdict(
    final_verdict: dict[str, Any],
    *,
    blackboard: dict[str, Any],
    validations: list[dict[str, Any]],
) -> dict[str, Any]:
    proof = dict(blackboard.get("parent_route_proof") or {})
    belief = dict(blackboard.get("current_belief") or {})
    proof_solved = _blackboard_parent_proof_solved(blackboard, proof)
    reasons: list[str] = []
    verdict_solved = str(final_verdict.get("verdict") or "") == "solved"
    solved_claim = bool(final_verdict.get("solved")) or verdict_solved
    route_solved_claim = str(final_verdict.get("route_status") or "") == "solved"
    if bool(final_verdict.get("solved")) != verdict_solved:
        reasons.append("final_solved_flag_mismatch")
    if route_solved_claim and not bool(final_verdict.get("solved")):
        reasons.append("final_route_status_solved_without_solved_flag")
    if solved_claim or route_solved_claim:
        if not proof_solved:
            reasons.append("final_solved_without_parent_proof")
        if not bool(final_verdict.get("stock_audit_passed")):
            reasons.append("final_solved_without_stock_audit")
        if bool(belief.get("child_route_solved")) and not proof_solved:
            reasons.append("child_solved_promoted_without_parent_proof")
        rejected_action_batches = [
            str(row.get("case_id") or row.get("artifact_id") or row.get("artifact_key") or "action_batch")
            for row in validations
            if isinstance(row, dict)
            and row.get("schema_version") == "agent_action_batch_validation.v1"
            and not row.get("accepted")
        ]
        if rejected_action_batches:
            reasons.append("final_solved_with_rejected_action_batch")
    return {
        "schema_version": "agentic_final_verdict_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "case_id": str(final_verdict.get("case_id") or blackboard.get("case_id") or ""),
        "final_verdict": dict(final_verdict),
        "parent_route_proof_summary": {
            "accepted": proof_solved,
            "solved": proof_solved,
            "route_status": str(proof.get("route_status") or ""),
        },
        "checked_invariants": [
            "solved_requires_parent_proof",
            "solved_requires_stock_audit",
            "child_route_cannot_promote_parent",
            "solved_requires_accepted_action_batches",
        ],
    }


def _downgrade_invalid_agentic_final_verdict(
    final_verdict: FinalVerdict,
    validation: dict[str, Any],
) -> FinalVerdict:
    reasons = sorted(
        set(
            [
                *[str(item) for item in final_verdict.reasons or []],
                *[str(item) for item in validation.get("reasons") or []],
                "agentic_final_verdict_validation_failed",
            ]
        )
    )
    return FinalVerdict(
        case_id=final_verdict.case_id,
        verdict="unresolved",
        reasons=reasons,
        route_status="agentic_final_verdict_validation_failed",
        solved=False,
        stock_audit_passed=False,
        artifact_refs=dict(final_verdict.artifact_refs or {}),
    )


def _obtain_action_batch(
    *,
    blackboard: dict[str, Any],
    round_index: int,
    run_dir: Path,
    state: ToolExecutionState | None,
    action_planner: ActionPlannerRunner | None,
    exhaust_round_budget: bool = False,
    use_codex_action_planner: bool | None = None,
    allow_deterministic_fallback: bool = True,
) -> dict[str, Any]:
    if action_planner is not None:
        return action_planner(blackboard=blackboard, round_index=round_index, run_dir=run_dir)
    return plan_action_batch_with_codex(
        blackboard=blackboard,
        round_index=round_index,
        run_dir=run_dir,
        enabled=use_codex_action_planner,
        exhaust_round_budget=exhaust_round_budget,
        mock_output=_mock_codex_action_planner(state, blackboard, round_index) if state is not None else None,
        allow_deterministic_fallback=allow_deterministic_fallback,
    )


def _execute_agent_action(
    *,
    action: dict[str, Any],
    state: ToolExecutionState,
    blackboard: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    action_type = str(action.get("action_type") or "")
    mock = _mock_action_result(state, action, blackboard)
    if mock is not None:
        return _wrap_action_result(mock), []
    if action_type == "classify_route_objectives":
        result = classify_route_objectives(
            target_smiles=str(state.target_input.get("target_smiles") or ""),
            target_name=str(state.target_input.get("target_name") or ""),
            family_hint=str(state.target_input.get("family_hint") or ""),
            failure_reasons=[
                str(row.get("reason") or "")
                for row in blackboard.get("route_failures") or []
                if isinstance(row, dict) and str(row.get("reason") or "").strip()
            ],
            source_evidence_refs=[
                str(item)
                for item in (blackboard.get("literature_evidence") or {}).get("source_refs") or []
            ],
            case_id=str(state.preflight.get("case_id") or ""),
        )
        state.artifacts["route_objective_summary"] = result
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "generate_disconnection_hypotheses":
        result = build_target_side_disconnection_hypotheses(
            target_smiles=str(state.target_input.get("target_smiles") or ""),
            target_name=str(state.target_input.get("target_name") or ""),
            family_hint=str(state.target_input.get("family_hint") or ""),
            source_evidence_refs=[str(item) for item in (blackboard.get("literature_evidence") or {}).get("source_refs") or []],
            case_id=str(state.preflight.get("case_id") or ""),
        )
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "build_failure_critic_report":
        result = compile_failure_critic_report(
            blackboard=blackboard,
            artifacts=state.artifacts,
            case_id=str(state.preflight.get("case_id") or ""),
            target_name=str(state.target_input.get("target_name") or ""),
        )
        state.artifacts["failure_critic_report"] = result
        return {"accepted": True, "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "search_literature":
        result = _codex_first_literature_scout(blackboard=blackboard, state=state, payload=dict(action.get("payload") or {}))
        state.artifacts["literature_scout"] = result
        scout_artifact = _record_literature_scout_report_artifact(
            state=state,
            action=action,
            blackboard=blackboard,
            scout_report=result,
        )
        state.artifacts["literature_scout_report"] = scout_artifact
        _validate_and_record_typed_artifact(state, "literature_scout_report", scout_artifact)
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "rank_analogical_hypotheses":
        result = rank_analogical_hypotheses_from_blackboard(blackboard)
        state.artifacts["analogical_hypothesis_ranking"] = result
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "extract_analogical_reaction_templates":
        payload = dict(action.get("payload") or {})
        result = extract_analogical_reaction_templates_from_blackboard(
            blackboard=blackboard,
            case_id=str(state.preflight.get("case_id") or ""),
            target_smiles=str(state.target_input.get("target_smiles") or ""),
            max_templates=int(payload.get("max_templates") or 12),
            radius_policy=str(payload.get("template_radius_policy") or "auto"),
        )
        state.artifacts["analogical_reaction_templates"] = result
        artifact = _record_analogical_template_artifact(
            state=state,
            action=action,
            blackboard=blackboard,
            artifact_key="analogical_reaction_template_report",
            artifact_type="AnalogicalReactionTemplateReport",
            schema_version="analogical_reaction_template_report.v1",
            payload=result,
        )
        state.artifacts["analogical_reaction_template_report"] = artifact
        _validate_and_record_typed_artifact(state, "analogical_reaction_template_report", artifact)
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "rank_analogical_reaction_templates":
        result = rank_analogical_reaction_templates_from_blackboard(blackboard)
        state.artifacts["analogical_template_ranking"] = result
        artifact = _record_analogical_template_artifact(
            state=state,
            action=action,
            blackboard=blackboard,
            artifact_key="analogical_reaction_template_ranking",
            artifact_type="AnalogicalReactionTemplateRanking",
            schema_version="analogical_reaction_template_ranking_artifact.v1",
            payload=result,
        )
        state.artifacts["analogical_reaction_template_ranking_artifact"] = artifact
        _validate_and_record_typed_artifact(state, "analogical_reaction_template_ranking", artifact)
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "apply_analogical_template_to_target":
        payload = dict(action.get("payload") or {})
        result = apply_analogical_templates_to_target(
            blackboard=blackboard,
            target_smiles=str(state.target_input.get("target_smiles") or ""),
            max_applications=int(payload.get("max_applications") or 5),
            radius_policy=str(payload.get("template_radius_policy") or "auto"),
            confidence_threshold=str(payload.get("analog_template_confidence_threshold") or "medium"),
        )
        state.artifacts["analogical_template_application_result"] = result
        artifact = _record_analogical_template_artifact(
            state=state,
            action=action,
            blackboard=blackboard,
            artifact_key="analogical_template_application_report",
            artifact_type="AnalogicalTemplateApplicationReport",
            schema_version="analogical_template_application_report_artifact.v1",
            payload=result,
        )
        state.artifacts["analogical_template_application_report"] = artifact
        _validate_and_record_typed_artifact(state, "analogical_template_application_report", artifact)
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "validate_template_application":
        policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
        application_report = apply_analogical_templates_to_target(
            blackboard=blackboard,
            target_smiles=str(state.target_input.get("target_smiles") or ""),
            max_applications=int(policy.get("max_template_applications_per_round") or 5),
            radius_policy=str(policy.get("template_radius_policy") or "auto"),
            confidence_threshold=str(policy.get("analog_template_confidence_threshold") or "medium"),
            include_executable_candidates=True,
        )
        result = validate_template_applications_for_guided_search(
            application_report=application_report,
            case_id=str(state.preflight.get("case_id") or ""),
            target_smiles=str(state.target_input.get("target_smiles") or ""),
            output_dir=state.run_dir,
        )
        state.artifacts["analogical_template_application_validation"] = result
        compiled = dict(result.get("compiled_downstream") or {})
        if compiled:
            state.artifacts["analogical_template_guided_hints"] = compiled
            state.artifacts["analogical_template_guided_hints_payload"] = compiled
        artifact = _record_analogical_template_artifact(
            state=state,
            action=action,
            blackboard=blackboard,
            artifact_key="analogical_template_application_validation",
            artifact_type="AnalogicalTemplateApplicationValidation",
            schema_version="analogical_template_application_validation_artifact.v1",
            payload=result,
        )
        state.artifacts["analogical_template_application_validation_artifact"] = artifact
        _validate_and_record_typed_artifact(state, "analogical_template_application_validation", artifact)
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "derive_broad_reaction_template":
        result = build_broad_transform_templates_from_blackboard(blackboard)
        state.artifacts["broad_transform_template_report"] = result
        return {"accepted": bool(result.get("accepted")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "extract_pdf_literature_structures":
        payload = dict(action.get("payload") or {})
        _inject_pdf_defaults(payload, state.target_input, blackboard=blackboard)
        payload.setdefault("output_dir", _pdf_action_output_dir(action))
        record = execute_local_tool("extract_pdf_literature_structures", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "extract_visual_literature_chain":
        payload = dict(action.get("payload") or {})
        _inject_pdf_defaults(payload, state.target_input, blackboard=blackboard)
        payload.setdefault("output_dir", _visual_action_output_dir(action))
        record = execute_local_tool("extract_visual_literature_chain", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "resolve_literature_structure_task":
        payload = dict(action.get("payload") or {})
        _inject_pdf_defaults(payload, state.target_input, blackboard=blackboard)
        payload.setdefault("output_dir", _structure_resolution_action_output_dir(action))
        record = execute_local_tool("resolve_literature_structure_task", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "compile_exact_literature_rows":
        record = execute_local_tool("compile_source_detail_chain_route", dict(action.get("payload") or {}), state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "run_guided_chemenzy":
        payload = {**build_agentic_guided_payload(blackboard), **dict(action.get("payload") or {})}
        record = execute_local_tool("run_guided_chemenzy_rerun", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "expand_child_target":
        payload = {"max_targets": 2, **dict(action.get("payload") or {})}
        record = execute_local_tool("run_route_expansion_subgoal_search", payload, state)
        return _tool_record_to_action_result(record), [record.to_dict()]
    if action_type == "stitch_parent_route":
        record = execute_local_tool("stitch_literature_chain_with_subgoal_route", dict(action.get("payload") or {}), state)
        stitched = dict((record.output.get("result") if isinstance(record.output, dict) else {}) or {})
        proof = _compile_parent_proof_from_state(state=state, blackboard=blackboard, stitched=stitched)
        state.artifacts["parent_route_proof"] = proof
        return {
            "accepted": bool(proof.get("accepted") or proof.get("reasons")),
            "result": {"parent_route_proof": proof, "stitched_route": stitched},
            "reasons": [str(item) for item in proof.get("reasons") or []],
        }, [record.to_dict()]
    if action_type == "compile_objective_route_proof":
        result = compile_route_objective_proof_bundle(
            blackboard=blackboard,
            parent_route_proof=blackboard.get("parent_route_proof") or state.artifacts.get("parent_route_proof") or {},
        )
        state.artifacts["route_proof_bundle"] = result
        return {"accepted": bool(result.get("accepted") or result.get("objective_proofs")), "result": result, "reasons": [str(item) for item in result.get("reasons") or []]}, []
    if action_type == "stop_unresolved":
        return {"accepted": True, "schema_version": "agent_stop_unresolved.v1", "reasons": ["stop_unresolved_selected"]}, []
    return {"accepted": False, "reasons": [f"unknown_action:{action_type}"]}, []


def _record_action_batch_artifacts(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    action_batch: dict[str, Any],
    validation: dict[str, Any],
    round_index: int,
    batch_path: Path,
    validation_path: Path,
) -> None:
    case_id = str(action_batch.get("case_id") or state.preflight.get("case_id") or "case")
    batch_key = f"agent_action_batch_round_{int(round_index)}"
    validation_key = f"agent_action_batch_validation_round_{int(round_index)}"
    batch_artifact = {
        "schema_version": "agent_action_batch_artifact.v1",
        "artifact_type": "AgentActionBatch",
        "artifact_id": f"{case_id}:action_batch:{int(round_index)}",
        "case_id": case_id,
        "source": str(action_batch.get("mode") or "agentic_blackboard_controller"),
        "input_refs": [str(state.run_dir / "agent_blackboard.json")],
        "evidence_refs": [],
        "validation_status": "accepted" if validation.get("accepted") else "rejected",
        "summary": f"{len(action_batch.get('actions') or [])} planned actions",
        "payload": dict(action_batch),
        "artifact_ref": str(batch_path),
        "validation_ref": str(validation_path),
        "no_solved_claim": True,
    }
    validation_artifact = {
        "schema_version": "agent_action_batch_validation_artifact.v1",
        "artifact_type": "AgentActionBatchValidation",
        "artifact_id": f"{case_id}:action_batch_validation:{int(round_index)}",
        "case_id": case_id,
        "source": "agent_action_batch_validator",
        "input_refs": [str(batch_path)],
        "evidence_refs": [],
        "validation_status": "accepted" if validation.get("accepted") else "rejected",
        "payload": dict(validation),
        "artifact_ref": str(validation_path),
        "no_solved_claim": True,
    }
    state.artifacts[batch_key] = batch_artifact
    state.artifacts[validation_key] = validation_artifact
    _validate_and_record_typed_artifact(state, batch_key, batch_artifact)
    _validate_and_record_typed_artifact(state, validation_key, validation_artifact)
    blackboard.setdefault("artifact_refs", {})[batch_key] = str(batch_path)
    blackboard.setdefault("artifact_refs", {})[validation_key] = str(validation_path)


def _validate_and_record_typed_artifact(
    state: ToolExecutionState,
    artifact_key: str,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_typed_artifact(artifact)
    record = {
        **dict(validation),
        "schema_version": "agentic_typed_artifact_validation_record.v1",
        "artifact_key": str(artifact_key),
        "validator": "validate_typed_artifact",
    }
    state.validations.append(record)
    if not validation.get("accepted"):
        state.safety_flags.append(f"typed_artifact_validation_failed:{artifact_key}")
    return record


def _record_literature_scout_report_artifact(
    *,
    state: ToolExecutionState,
    action: dict[str, Any],
    blackboard: dict[str, Any],
    scout_report: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(scout_report.get("case_id") or blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    scout_index = int((blackboard.get("budget_state") or {}).get("scout_calls") or 0) + 1
    path = state.run_dir / f"literature_scout_report_{scout_index}.json"
    source_refs = [str(item) for item in scout_report.get("source_refs") or [] if str(item or "").strip()]
    artifact = {
        "schema_version": "literature_scout_report_artifact.v1",
        "artifact_type": "LiteratureScoutReport",
        "artifact_id": f"{case_id}:literature_scout:{scout_index}",
        "case_id": case_id,
        "source": "codex_first_literature_scout",
        "input_refs": [
            str(state.run_dir / "agent_blackboard.json"),
            str(action.get("action_id") or "search_literature"),
        ],
        "evidence_refs": source_refs,
        "validation_status": "accepted" if scout_report.get("accepted") else "draft_only",
        "payload": dict(scout_report),
        "artifact_ref": str(path),
        "no_solved_claim": True,
    }
    write_json(path, artifact)
    return artifact


def _record_analogical_template_artifact(
    *,
    state: ToolExecutionState,
    action: dict[str, Any],
    blackboard: dict[str, Any],
    artifact_key: str,
    artifact_type: str,
    schema_version: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(payload.get("case_id") or blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    safe_key = _safe_artifact_filename(artifact_key)
    path = state.run_dir / f"{safe_key}.json"
    source_refs = [str(item) for item in payload.get("source_refs") or [] if str(item or "").strip()]
    if not source_refs and artifact_type != "AnalogicalReactionTemplateReport":
        source_refs = [
            str(row.get("template_id") or "")
            for row in payload.get("selected_templates") or payload.get("applications") or []
            if isinstance(row, dict) and str(row.get("template_id") or "").strip()
        ]
    artifact = {
        "schema_version": schema_version,
        "artifact_type": artifact_type,
        "artifact_id": f"{case_id}:{safe_key}",
        "case_id": case_id,
        "source": "agentic_blackboard_analogical_template_action",
        "input_refs": [
            str(state.run_dir / "agent_blackboard.json"),
            str(action.get("action_id") or artifact_key),
        ],
        "evidence_refs": _dedupe(source_refs),
        "validation_status": "accepted" if payload.get("accepted") else "draft_only",
        "payload": dict(payload),
        "artifact_ref": str(path),
        "no_solved_claim": True,
    }
    write_json(path, artifact)
    return artifact


def _record_agent_blackboard_snapshot_artifact(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    path = state.run_dir / "agent_blackboard_snapshot.json"
    artifact = {
        "schema_version": "agent_blackboard_snapshot_artifact.v1",
        "artifact_type": "AgentBlackboardSnapshot",
        "artifact_id": f"{case_id}:agent_blackboard_snapshot",
        "case_id": case_id,
        "source": "agentic_blackboard_controller",
        "input_refs": [str(state.run_dir / "agent_blackboard.json")],
        "evidence_refs": [],
        "validation_status": "accepted",
        "payload": dict(blackboard),
        "artifact_ref": str(path),
        "no_solved_claim": True,
    }
    write_json(path, artifact)
    return artifact


def _record_agentic_final_verdict_validation_artifact(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    final_validation: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(final_validation.get("case_id") or blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    path = state.run_dir / "agentic_final_verdict_validation.json"
    artifact = {
        "schema_version": "agentic_final_verdict_validation_artifact.v1",
        "artifact_type": "AgenticFinalVerdictValidation",
        "artifact_id": f"{case_id}:agentic_final_verdict_validation",
        "case_id": case_id,
        "source": "agentic_final_verdict_gate",
        "input_refs": [str(state.run_dir / "agent_blackboard.json"), str(state.run_dir / "final_verdict.json")],
        "evidence_refs": [],
        "validation_status": "accepted" if final_validation.get("accepted") else "rejected",
        "payload": dict(final_validation),
        "artifact_ref": str(path),
        "no_solved_claim": True,
    }
    write_json(path, artifact)
    return artifact


def _record_agentic_capability_audit_artifact(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    action_batch_validations: list[dict[str, Any]],
    typed_validations: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    final_verdict: dict[str, Any],
    final_validation: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    payload = _compile_agentic_capability_audit_payload(
        blackboard=blackboard,
        action_batches=action_batches,
        action_batch_validations=action_batch_validations,
        typed_validations=typed_validations,
        tool_calls=tool_calls,
        final_verdict=final_verdict,
        final_validation=final_validation,
        run_dir=state.run_dir,
    )
    artifact = {
        "schema_version": "agentic_capability_audit_artifact.v1",
        "artifact_type": "AgenticCapabilityAudit",
        "artifact_id": f"{case_id}:agentic_capability_audit",
        "case_id": case_id,
        "source": "agentic_blackboard_capability_audit",
        "input_refs": [
            str(state.run_dir / "agent_blackboard.json"),
            str(state.run_dir / "final_verdict.json"),
            str(state.run_dir / "agentic_final_verdict_validation.json"),
        ],
        "evidence_refs": [],
        "validation_status": "accepted" if payload.get("accepted") else "rejected",
        "payload": payload,
        "artifact_ref": str(state.run_dir / "agentic_capability_audit.json"),
        "no_solved_claim": True,
    }
    write_json(state.run_dir / "agentic_capability_audit.json", artifact)
    return artifact


def _compile_agentic_capability_audit_payload(
    *,
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    action_batch_validations: list[dict[str, Any]],
    typed_validations: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    final_verdict: dict[str, Any],
    final_validation: dict[str, Any],
    run_dir: Path | None = None,
) -> dict[str, Any]:
    checks = [
        _capability_check_blackboard_state(blackboard),
        _capability_check_policy_action_batches(blackboard, action_batches),
        _capability_check_action_validation(blackboard, action_batch_validations),
        _capability_check_planner_history(blackboard, action_batches, run_dir=run_dir),
        _capability_check_blackboard_transition_history(blackboard),
        _capability_check_budget_limits(blackboard, action_batches),
        _capability_check_source_acquisition(blackboard),
        _capability_check_failure_critic(blackboard),
        _capability_check_analogical_boundary(blackboard),
        _capability_check_guided_chemenzy_boundary(blackboard),
        _capability_check_final_verdict_gate(blackboard, final_verdict, final_validation),
        _capability_check_typed_artifact_validations(typed_validations),
        _capability_check_artifact_ref_integrity(
            blackboard=blackboard,
            final_verdict=final_verdict,
            typed_validations=typed_validations,
            run_dir=run_dir,
        ),
    ]
    failed = [str(row.get("requirement_id") or "") for row in checks if not row.get("accepted")]
    return {
        "schema_version": "agentic_capability_audit.v1",
        "case_id": str(blackboard.get("case_id") or ""),
        "accepted": not failed,
        "audit_authority": "diagnostic_only",
        "deterministic_final_verdict_required": True,
        "final_verdict_authority": "deterministic_parent_route_proof",
        "requirement_checks": checks,
        "failed_requirements": [item for item in failed if item],
        "warning_requirements": [
            str(row.get("requirement_id") or "")
            for row in checks
            if row.get("accepted") and row.get("status") == "not_exercised"
        ],
        "observed_action_types": _dedupe(
            [
                str(action.get("action_type") or "")
                for batch in action_batches
                for action in batch.get("actions") or []
                if isinstance(action, dict)
            ]
        ),
        "tool_call_count": len(tool_calls),
        "final_verdict_summary": {
            "verdict": str(final_verdict.get("verdict") or ""),
            "route_status": str(final_verdict.get("route_status") or ""),
            "solved": bool(final_verdict.get("solved")),
        },
        "no_solved_claim": True,
    }


def _capability_check(
    requirement_id: str,
    accepted: bool,
    *,
    evidence: list[str] | None = None,
    reasons: list[str] | None = None,
    status: str = "checked",
) -> dict[str, Any]:
    return {
        "schema_version": "agentic_capability_requirement_check.v1",
        "requirement_id": requirement_id,
        "accepted": bool(accepted),
        "status": status,
        "evidence": [str(item) for item in evidence or [] if str(item or "").strip()],
        "reasons": [str(item) for item in reasons or [] if str(item or "").strip()],
        "no_solved_claim": True,
    }


def _capability_check_blackboard_state(blackboard: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if blackboard.get("schema_version") != "agent_blackboard.v1":
        reasons.append("invalid_or_missing_blackboard_schema")
    for key in ("target_profile", "literature_evidence", "budget_state", "current_belief", "planner_history", "action_history"):
        if key not in blackboard:
            reasons.append(f"missing_blackboard_field:{key}")
    journal = dict(blackboard.get("blackboard_event_journal") or {})
    if journal.get("schema_version") != "agent_blackboard_event_journal_summary.v1":
        reasons.append("blackboard_event_journal_summary_missing")
    if journal.get("authority") != "digest_chained_blackboard_events":
        reasons.append("blackboard_event_journal_authority_invalid")
    if not str(journal.get("last_event_sha256") or ""):
        reasons.append("blackboard_event_journal_digest_missing")
    return _capability_check(
        "blackboard_single_state_source",
        not reasons,
        evidence=[
            str(journal.get("journal_path") or "blackboard_events/events.jsonl"),
            "agent_blackboard.json:projection_only",
            f"case_id:{blackboard.get('case_id') or ''}",
        ],
        reasons=reasons,
    )


def _capability_check_policy_action_batches(blackboard: dict[str, Any], action_batches: list[dict[str, Any]]) -> dict[str, Any]:
    if not ((blackboard.get("target_profile") or {}).get("valid", True)) and not action_batches:
        return _capability_check(
            "policy_driven_typed_action_batches",
            True,
            evidence=["preflight_rejected:no_action_batch_required"],
            status="preflight_rejected",
        )
    reasons: list[str] = []
    if not action_batches:
        reasons.append("no_action_batches_recorded")
    for idx, batch in enumerate(action_batches, start=1):
        if batch.get("schema_version") != "agent_action_batch.v1":
            reasons.append(f"invalid_action_batch_schema:{idx}")
        actions = batch.get("actions")
        if not isinstance(actions, list):
            reasons.append(f"action_batch_actions_not_list:{idx}")
            continue
        if len(actions) > 3:
            reasons.append(f"action_batch_exceeds_three_actions:{idx}")
        for action_idx, action in enumerate(actions):
            if not isinstance(action, dict):
                reasons.append(f"action_not_object:{idx}:{action_idx}")
                continue
            if not str(action.get("action_type") or "").strip():
                reasons.append(f"action_missing_type:{idx}:{action_idx}")
    return _capability_check(
        "policy_driven_typed_action_batches",
        not reasons,
        evidence=[f"action_batch_count:{len(action_batches)}"],
        reasons=reasons,
    )


def _capability_check_action_validation(blackboard: dict[str, Any], validations: list[dict[str, Any]]) -> dict[str, Any]:
    if not ((blackboard.get("target_profile") or {}).get("valid", True)) and not validations:
        return _capability_check(
            "deterministic_action_batch_validation_gate",
            True,
            evidence=["preflight_rejected:no_action_batch_validation_required"],
            status="preflight_rejected",
        )
    reasons: list[str] = []
    if not validations:
        reasons.append("no_action_batch_validations_recorded")
    rejected = [row for row in validations if isinstance(row, dict) and not row.get("accepted")]
    if rejected:
        reasons.extend(
            f"rejected_action_batch:{idx}:{','.join(str(item) for item in row.get('reasons') or [])}"
            for idx, row in enumerate(rejected, start=1)
        )
    return _capability_check(
        "deterministic_action_batch_validation_gate",
        not reasons,
        evidence=[f"validation_count:{len(validations)}"],
        reasons=reasons,
    )


def _capability_check_planner_history(
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    history = [dict(row) for row in blackboard.get("planner_history") or [] if isinstance(row, dict)]
    if not ((blackboard.get("target_profile") or {}).get("valid", True)) and not action_batches and not history:
        return _capability_check(
            "planner_decision_history_audited",
            True,
            evidence=["preflight_rejected:no_planner_history_required"],
            status="preflight_rejected",
        )
    reasons: list[str] = []
    if len(history) < len(action_batches):
        reasons.append("planner_history_shorter_than_action_batches")
    codex_attempts = sum(1 for row in history if (row.get("codex_action_planner") or {}).get("attempted"))
    fallback_rounds = [row for row in history if (row.get("codex_action_planner") or {}).get("fallback_used")]
    snapshot_context_count = 0
    snapshot_payload_requirement_count = 0
    snapshot_tool_policy_count = 0
    history_tool_policy_count = 0
    for idx, row in enumerate(history):
        planner = dict(row.get("codex_action_planner") or {})
        if not planner.get("attempted"):
            continue
        history_tool_policy = dict(planner.get("tool_policy") or {})
        if not history_tool_policy:
            reasons.append(f"codex_planner_history_missing_tool_policy:{idx}")
        elif history_tool_policy.get("schema_version") != "codex_action_planner_tool_policy.v1":
            reasons.append(f"codex_planner_history_invalid_tool_policy_schema:{idx}")
        elif history_tool_policy.get("raw_reaction_output_allowed") is not False:
            reasons.append(f"codex_planner_history_tool_policy_allows_raw_reaction:{idx}")
        elif history_tool_policy.get("final_verdict_authority") != "deterministic_parent_route_proof":
            reasons.append(f"codex_planner_history_tool_policy_invalid_final_authority:{idx}")
        else:
            history_tool_policy_count += 1
        snapshot_ref = str(planner.get("blackboard_snapshot_ref") or "").strip()
        record_ref = str(planner.get("record_ref") or "").strip()
        if not snapshot_ref:
            reasons.append(f"codex_planner_missing_blackboard_snapshot_ref:{idx}")
        else:
            snapshot = _load_json_ref(snapshot_ref, run_dir=run_dir)
            if not snapshot:
                reasons.append(f"codex_planner_blackboard_snapshot_unreadable:{idx}")
            elif snapshot.get("schema_version") != "codex_action_planner_blackboard_snapshot.v1":
                reasons.append(f"codex_planner_blackboard_snapshot_invalid_schema:{idx}")
            else:
                context = dict(snapshot.get("planner_context") or {})
                if context.get("schema_version") != "codex_action_planner_context.v1":
                    reasons.append(f"codex_planner_snapshot_missing_context:{idx}")
                elif context.get("no_solved_claim") is not True:
                    reasons.append(f"codex_planner_snapshot_context_missing_no_solved_claim:{idx}")
                else:
                    snapshot_context_count += 1
                    tool_policy = dict(context.get("planner_tool_policy") or {})
                    if not tool_policy:
                        reasons.append(f"codex_planner_snapshot_missing_tool_policy:{idx}")
                    elif tool_policy.get("schema_version") != "codex_action_planner_tool_policy.v1":
                        reasons.append(f"codex_planner_snapshot_invalid_tool_policy_schema:{idx}")
                    elif tool_policy.get("raw_reaction_output_allowed") is not False:
                        reasons.append(f"codex_planner_snapshot_tool_policy_allows_raw_reaction:{idx}")
                    elif tool_policy.get("final_verdict_authority") != "deterministic_parent_route_proof":
                        reasons.append(f"codex_planner_snapshot_tool_policy_invalid_final_authority:{idx}")
                    else:
                        snapshot_tool_policy_count += 1
                    requirements = dict(context.get("action_payload_requirements") or {})
                    search_actions = dict(requirements.get("search_actions") or {})
                    source_actions = dict(requirements.get("source_sensitive_actions") or {})
                    guided_actions = dict(requirements.get("guided_actions") or {})
                    child_actions = dict(requirements.get("child_expansion_actions") or {})
                    stitch_actions = dict(requirements.get("stitch_actions") or {})
                    template_actions = dict(requirements.get("analogical_template_actions") or {})
                    expected_actions = {
                        "extract_pdf_literature_structures",
                        "extract_visual_literature_chain",
                        "resolve_literature_structure_task",
                        "compile_exact_literature_rows",
                    }
                    expected_template_actions = {
                        "extract_analogical_reaction_templates",
                        "rank_analogical_reaction_templates",
                        "apply_analogical_template_to_target",
                        "validate_template_application",
                    }
                    missing_actions = sorted(expected_actions - set(source_actions))
                    missing_template_actions = sorted(expected_template_actions - set(template_actions))
                    search = dict(search_actions.get("search_literature") or {})
                    guided = dict(guided_actions.get("run_guided_chemenzy") or {})
                    child = dict(child_actions.get("expand_child_target") or {})
                    stitch = dict(stitch_actions.get("stitch_parent_route") or {})
                    guided_fields = [str(field) for field in guided.get("accepted_payload_fields") or []]
                    guided_policy_contract_present = bool(
                        "guided_policy_runtime_rebuild" in guided_fields
                        or guided.get("runtime_policy_rebuild") is True
                        or "search_policy" in guided_fields
                        or "chem_enzy_search_policy" in guided_fields
                    )
                    template_policy_missing = [
                        action_type
                        for action_type, requirement in template_actions.items()
                        if action_type in expected_template_actions
                        and "analogical_template_policy" not in (dict(requirement).get("accepted_payload_fields") or [])
                    ]
                    if requirements.get("schema_version") != "codex_action_payload_requirements.v1":
                        reasons.append(f"codex_planner_snapshot_missing_payload_requirements:{idx}")
                    elif not search:
                        reasons.append(f"codex_planner_snapshot_missing_search_requirements:{idx}")
                    elif "source_acquisition_policy" not in (search.get("accepted_payload_fields") or []):
                        reasons.append(f"codex_planner_snapshot_search_requirements_missing_policy_field:{idx}")
                    elif missing_actions:
                        reasons.append(
                            f"codex_planner_snapshot_missing_source_sensitive_requirements:{idx}:{','.join(missing_actions)}"
                        )
                    elif not guided:
                        reasons.append(f"codex_planner_snapshot_missing_guided_action_requirements:{idx}")
                    elif not guided_policy_contract_present:
                        reasons.append(f"codex_planner_snapshot_guided_requirements_missing_policy_field:{idx}")
                    elif not child:
                        reasons.append(f"codex_planner_snapshot_missing_child_expansion_requirements:{idx}")
                    elif "subgoal_targets" not in (child.get("accepted_payload_fields") or []):
                        reasons.append(f"codex_planner_snapshot_child_requirements_missing_subgoal_field:{idx}")
                    elif not stitch:
                        reasons.append(f"codex_planner_snapshot_missing_stitch_requirements:{idx}")
                    elif "proof_binding" not in (stitch.get("accepted_payload_fields") or []):
                        reasons.append(f"codex_planner_snapshot_stitch_requirements_missing_binding_field:{idx}")
                    elif not template_actions:
                        reasons.append(f"codex_planner_snapshot_missing_analogical_template_requirements:{idx}")
                    elif missing_template_actions:
                        reasons.append(
                            f"codex_planner_snapshot_missing_analogical_template_action_requirements:{idx}:"
                            f"{','.join(missing_template_actions)}"
                        )
                    elif template_policy_missing:
                        reasons.append(
                            f"codex_planner_snapshot_template_requirements_missing_policy_field:{idx}:"
                            f"{','.join(sorted(template_policy_missing))}"
                        )
                    else:
                        snapshot_payload_requirement_count += 1
        if not record_ref:
            reasons.append(f"codex_planner_missing_run_record_ref:{idx}")
        elif not _ref_path_exists(record_ref, run_dir=run_dir):
            reasons.append(f"codex_planner_run_record_missing:{idx}")
    return _capability_check(
        "planner_decision_history_audited",
        not reasons,
        evidence=[
            f"planner_history_count:{len(history)}",
            f"codex_attempted_rounds:{codex_attempts}",
            f"fallback_rounds:{len(fallback_rounds)}",
            f"codex_snapshot_context_count:{snapshot_context_count}",
            f"codex_snapshot_payload_requirement_count:{snapshot_payload_requirement_count}",
            f"codex_snapshot_tool_policy_count:{snapshot_tool_policy_count}",
            f"codex_history_tool_policy_count:{history_tool_policy_count}",
        ],
        reasons=reasons,
    )


def _load_json_ref(ref: str, *, run_dir: Path | None = None) -> dict[str, Any]:
    path = _ref_path(ref, run_dir=run_dir)
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _ref_path_exists(ref: str, *, run_dir: Path | None = None) -> bool:
    path = _ref_path(ref, run_dir=run_dir)
    return bool(path is not None and path.exists())


def _ref_path(ref: str, *, run_dir: Path | None = None) -> Path | None:
    text = str(ref or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute() and run_dir is not None:
        path = Path(run_dir) / path
    return path


def _capability_check_blackboard_transition_history(blackboard: dict[str, Any]) -> dict[str, Any]:
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    if not history:
        return _capability_check(
            "blackboard_transition_history_audited",
            True,
            evidence=["action_history:not_exercised"],
            status="not_exercised",
        )
    reasons: list[str] = []
    changed = 0
    for idx, row in enumerate(history):
        if not isinstance(row.get("blackboard_counts_before"), dict):
            reasons.append(f"missing_blackboard_counts_before:{idx}")
        if not isinstance(row.get("blackboard_counts_after"), dict):
            reasons.append(f"missing_blackboard_counts_after:{idx}")
        delta = row.get("blackboard_delta")
        if not isinstance(delta, dict):
            reasons.append(f"missing_blackboard_delta:{idx}")
            continue
        if delta:
            changed += 1
            if not isinstance(row.get("changed_blackboard_fields"), list):
                reasons.append(f"missing_changed_blackboard_fields:{idx}")
    return _capability_check(
        "blackboard_transition_history_audited",
        not reasons,
        evidence=[f"action_transition_count:{len(history)}", f"changed_transition_count:{changed}"],
        reasons=reasons,
    )


def _capability_check_budget_limits(blackboard: dict[str, Any], action_batches: list[dict[str, Any]]) -> dict[str, Any]:
    budget = dict(blackboard.get("budget_state") or {})
    reasons: list[str] = []
    checks = [
        ("rounds_completed", "max_rounds"),
        ("scout_calls", "max_scout_calls"),
        ("visual_calls", "max_visual_calls"),
        ("chemenzy_runs", "max_chemenzy_runs"),
        ("child_target_runs", "max_child_target_runs"),
        ("template_application_actions", "max_template_application_actions"),
    ]
    for used_key, max_key in checks:
        if max_key not in budget:
            continue
        try:
            used = int(budget.get(used_key) or 0)
            maximum = int(budget.get(max_key) or 0)
        except (TypeError, ValueError):
            reasons.append(f"invalid_budget_value:{used_key}")
            continue
        if maximum >= 0 and used > maximum:
            reasons.append(f"budget_exceeded:{used_key}")
    if len(action_batches) > int(budget.get("max_rounds") or len(action_batches) or 0):
        reasons.append("action_batch_count_exceeds_max_rounds")
    return _capability_check(
        "budgeted_round_and_tool_limits",
        not reasons,
        evidence=[
            f"rounds_completed:{budget.get('rounds_completed') or 0}",
            f"max_rounds:{budget.get('max_rounds') or 0}",
            f"action_batch_count:{len(action_batches)}",
        ],
        reasons=reasons,
    )


def _capability_check_source_acquisition(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    candidates = [dict(row) for row in evidence.get("source_candidates") or [] if isinstance(row, dict)]
    lifecycle = [dict(row) for row in evidence.get("source_lifecycle") or [] if isinstance(row, dict)]
    search_seen = any(
        isinstance(row, dict) and row.get("action_type") == "search_literature"
        for row in blackboard.get("action_history") or []
    )
    if not search_seen:
        return _capability_check(
            "codex_first_source_acquisition_audited",
            True,
            evidence=["search_literature:not_exercised"],
            status="not_exercised",
        )
    reasons: list[str] = []
    if evidence.get("fallback_order") not in (
        ["codex_online", "local_pdf", "placeholder"],
        ["codex_online", "placeholder"],
    ):
        reasons.append("source_acquisition_fallback_order_not_recorded")
    if not isinstance(evidence.get("scout_attempts"), list):
        reasons.append("source_acquisition_attempts_not_recorded")
    if not evidence.get("source_candidates"):
        reasons.append("source_acquisition_no_candidates_or_placeholders")
    source_material_count = sum(
        len(evidence.get(key) or [])
        for key in (
            "planner_source_hints",
            "source_candidates",
            "pdf_structure_evidence",
            "visual_chains",
            "exact_rows",
        )
    )
    if source_material_count and not lifecycle:
        reasons.append("source_lifecycle_missing_for_source_material")
    for idx, row in enumerate(lifecycle):
        if not str(row.get("source_key") or "").strip():
            reasons.append(f"source_lifecycle_missing_source_key:{idx}")
        if row.get("no_solved_claim") is not True:
            reasons.append(f"source_lifecycle_missing_no_solved_claim:{idx}")
    proxy_requests = [
        dict(row)
        for row in evidence.get("local_pdf_proxy_requests") or []
        if isinstance(row, dict)
    ]
    for idx, candidate in enumerate(candidates):
        match = candidate.get("local_pdf_match")
        local_index = candidate.get("local_pdf_index")
        discovery_mode = str(candidate.get("source_discovery_mode") or evidence.get("source_discovery_mode") or "")
        local_pdf_fallback_mode = discovery_mode in {"local_pdf_fallback", "local_pdf_fallback_after_codex_failure"}
        if local_pdf_fallback_mode and not _source_acquisition_codex_online_attempted(evidence):
            reasons.append(f"local_pdf_fallback_without_codex_online_attempt:{idx}")
        if discovery_mode == "local_pdf_fallback" and not _candidate_is_user_provided_local_pdf_seed(candidate):
            reasons.append(f"direct_local_pdf_fallback_missing_user_seed_marker:{idx}")
        if discovery_mode in {"codex_online+local_pdf_cache", "local_pdf_cache_match"}:
            if not isinstance(match, dict) or not str(match.get("match_basis") or "").strip():
                reasons.append(f"local_pdf_cache_match_missing_provenance:{idx}")
        if isinstance(local_index, dict) and local_index.get("match_policy") == "agent_discovered_metadata_required":
            if not isinstance(match, dict) or not _local_pdf_match_has_agent_discovered_metadata(match):
                reasons.append(f"auto_local_pdf_cache_without_agent_discovered_match:{idx}")
        if _candidate_needs_local_pdf_proxy(candidate) and not _proxy_request_matches_candidate(candidate, proxy_requests):
            reasons.append(f"metadata_only_source_without_local_pdf_proxy_request:{idx}")
    return _capability_check(
        "codex_first_source_acquisition_audited",
        not reasons,
        evidence=[
            f"source_discovery_mode:{evidence.get('source_discovery_mode') or ''}",
            f"planner_source_hint_count:{len(evidence.get('planner_source_hints') or [])}",
            f"source_lifecycle_count:{len(lifecycle)}",
            f"source_candidate_count:{len(evidence.get('source_candidates') or [])}",
            f"local_pdf_cache_match_count:{sum(1 for row in candidates if isinstance(row.get('local_pdf_match'), dict))}",
        ],
        reasons=reasons,
    )


def _source_acquisition_codex_online_attempted(evidence: dict[str, Any]) -> bool:
    return any(
        str(row.get("mode") or "") == "codex_online" and bool(row.get("attempted"))
        for row in evidence.get("scout_attempts") or []
        if isinstance(row, dict)
    )


def _candidate_is_user_provided_local_pdf_seed(candidate: dict[str, Any]) -> bool:
    role = str(candidate.get("source_role") or candidate.get("source_usage") or "").strip().lower()
    return bool(candidate.get("user_provided_source_seed")) or role == "user_provided_local_pdf_seed"


def _local_pdf_match_has_agent_discovered_metadata(match: dict[str, Any]) -> bool:
    return bool(
        str(match.get("agent_discovered_source_ref") or "").strip()
        or str(match.get("agent_discovered_doi") or "").strip()
        or str(match.get("agent_discovered_pii") or "").strip()
        or str(match.get("agent_discovered_title") or "").strip()
        or str(match.get("agent_discovered_url") or "").strip()
    )


def _proxy_request_matches_candidate(candidate: dict[str, Any], requests: list[dict[str, Any]]) -> bool:
    candidate_doi = _normalize_doi(str(candidate.get("doi") or candidate.get("source_ref") or ""))
    candidate_url = str(candidate.get("url") or "").strip().lower()
    candidate_ref = str(candidate.get("source_ref") or "").strip().lower()
    for request in requests:
        request_doi = _normalize_doi(str(request.get("doi") or request.get("source_ref") or request.get("url") or ""))
        if candidate_doi and request_doi and candidate_doi == request_doi:
            return True
        request_url = str(request.get("url") or "").strip().lower()
        if candidate_url and request_url and candidate_url == request_url:
            return True
        request_ref = str(request.get("source_ref") or "").strip().lower()
        if candidate_ref and request_ref and candidate_ref == request_ref:
            return True
    return False


def _capability_check_failure_critic(blackboard: dict[str, Any]) -> dict[str, Any]:
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    critic_seen = any(row.get("action_type") == "build_failure_critic_report" for row in history)
    route_failures = [row for row in blackboard.get("route_failures") or [] if isinstance(row, dict)]
    if not route_failures:
        return _capability_check(
            "failure_critic_updates_blackboard",
            True,
            evidence=["route_failures:not_observed"],
            status="not_exercised",
        )
    reasons = [] if critic_seen else ["route_failures_present_without_failure_critic_history"]
    return _capability_check(
        "failure_critic_updates_blackboard",
        not reasons,
        evidence=[f"route_failure_count:{len(route_failures)}", f"critic_seen:{critic_seen}"],
        reasons=reasons,
    )


def _capability_check_analogical_boundary(blackboard: dict[str, Any]) -> dict[str, Any]:
    belief = dict(blackboard.get("current_belief") or {})
    policy = dict(belief.get("template_policy") or {})
    summary = _analogical_template_summary(blackboard)
    reasons: list[str] = []
    if policy.get("analogy_is_advisory_only", True) is not True:
        reasons.append("analogy_policy_not_advisory_only")
    if policy.get("analogical_template_hints_are_not_exact_rows", True) is not True:
        reasons.append("analogical_template_hints_can_be_misread_as_exact_rows")
    if summary.get("final_verdict_authority") != "none":
        reasons.append("analogical_template_marked_as_final_authority")
    return _capability_check(
        "analogy_advisory_only_not_solved_proof",
        not reasons,
        evidence=[
            f"template_count:{summary.get('template_count') or 0}",
            f"validated_one_step_row_count:{summary.get('validated_one_step_row_count') or 0}",
            f"validated_guided_hint_count:{summary.get('validated_guided_hint_count') or 0}",
        ],
        reasons=reasons,
    )


def _capability_check_guided_chemenzy_boundary(blackboard: dict[str, Any]) -> dict[str, Any]:
    budget = dict(blackboard.get("budget_state") or {})
    guided_seen = any(
        isinstance(row, dict) and row.get("action_type") == "run_guided_chemenzy"
        for row in blackboard.get("action_history") or []
    )
    try:
        used = int(budget.get("chemenzy_runs") or 0)
        maximum = int(budget.get("max_chemenzy_runs") or 0)
    except (TypeError, ValueError):
        used = 0
        maximum = 0
    reasons = []
    if maximum >= 0 and used > maximum:
        reasons.append("guided_chemenzy_budget_exceeded")
    return _capability_check(
        "guided_chemenzy_is_budgeted_action",
        not reasons,
        evidence=[f"guided_seen:{guided_seen}", f"chemenzy_runs:{used}", f"max_chemenzy_runs:{maximum}"],
        reasons=reasons,
        status="checked" if guided_seen else "not_exercised",
    )


def _capability_check_final_verdict_gate(
    blackboard: dict[str, Any],
    final_verdict: dict[str, Any],
    final_validation: dict[str, Any],
) -> dict[str, Any]:
    proof = dict(blackboard.get("parent_route_proof") or {})
    solved = bool(final_verdict.get("solved")) or str(final_verdict.get("verdict") or "") == "solved"
    reasons: list[str] = []
    if not final_validation.get("accepted"):
        reasons.append("final_verdict_validation_rejected")
    if solved and not _blackboard_parent_proof_solved(blackboard, proof):
        reasons.append("solved_without_parent_proof")
    return _capability_check(
        "final_verdict_requires_parent_route_proof",
        not reasons,
        evidence=[
            f"final_verdict:{final_verdict.get('verdict') or ''}",
            f"final_solved:{bool(final_verdict.get('solved'))}",
            f"parent_proof_accepted:{_blackboard_parent_proof_solved(blackboard, proof)}",
            f"final_validation_accepted:{bool(final_validation.get('accepted'))}",
        ],
        reasons=reasons,
    )


def _capability_check_typed_artifact_validations(typed_validations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in typed_validations
        if isinstance(row, dict) and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
    ]
    failed = [row for row in rows if not row.get("accepted")]
    return _capability_check(
        "typed_artifacts_self_validated",
        not failed,
        evidence=[f"typed_validation_count:{len(rows)}", f"failed_typed_validation_count:{len(failed)}"],
        reasons=[
            f"{row.get('artifact_key') or 'artifact'}:{','.join(str(item) for item in row.get('reasons') or [])}"
            for row in failed
        ],
    )


def _capability_check_artifact_ref_integrity(
    *,
    blackboard: dict[str, Any],
    final_verdict: dict[str, Any],
    typed_validations: list[dict[str, Any]],
    run_dir: Path | None,
) -> dict[str, Any]:
    """Check that closeout artifact refs name real files or deterministic closeout outputs."""
    run_path = Path(run_dir).resolve() if run_dir is not None else None
    blackboard_refs = {
        str(key): str(value)
        for key, value in (blackboard.get("artifact_refs") or {}).items()
        if str(value or "").strip()
    }
    final_refs = {
        str(key): str(value)
        for key, value in (final_verdict.get("artifact_refs") or {}).items()
        if str(value or "").strip()
    }
    merged_refs = {**blackboard_refs, **final_refs}
    required_closeout_refs = {
        "agentic_final_verdict_validation": "agentic_final_verdict_validation.json",
        "agent_blackboard_snapshot": "agent_blackboard_snapshot.json",
        "agentic_capability_audit": "agentic_capability_audit.json",
        "agentic_run_audit": "agentic_run_audit.json",
    }
    future_closeout_refs = {"agentic_capability_audit", "agentic_run_audit"}
    reasons: list[str] = []
    evidence: list[str] = []
    for key, filename in required_closeout_refs.items():
        ref = merged_refs.get(key)
        if not ref:
            reasons.append(f"missing_artifact_ref:{key}")
            continue
        if blackboard_refs.get(key) and final_refs.get(key) and blackboard_refs[key] != final_refs[key]:
            reasons.append(f"blackboard_final_artifact_ref_mismatch:{key}")
        path = Path(ref)
        if not path.is_absolute() and run_path is not None:
            path = run_path / path
        expected_path = (run_path / filename) if run_path is not None else None
        if key in future_closeout_refs and expected_path is not None and _same_path(path, expected_path):
            evidence.append(f"expected_closeout_ref:{key}")
            continue
        if path.exists():
            evidence.append(f"artifact_ref_exists:{key}")
            continue
        reasons.append(f"missing_artifact_ref_file:{key}")

    accepted_validation_keys = {
        str(row.get("artifact_key") or "")
        for row in typed_validations
        if isinstance(row, dict)
        and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        and row.get("accepted")
    }
    for key in ("agentic_final_verdict_validation", "agent_blackboard_snapshot"):
        if key not in accepted_validation_keys:
            reasons.append(f"missing_typed_validation:{key}")
        else:
            evidence.append(f"typed_validation_exists:{key}")
    for key in ("agentic_capability_audit", "agentic_run_audit"):
        if key in accepted_validation_keys:
            evidence.append(f"typed_validation_exists:{key}")
        else:
            evidence.append(f"typed_validation_expected_after_closeout:{key}")

    return _capability_check(
        "artifact_refs_and_typed_validation_integrity",
        not reasons,
        evidence=evidence,
        reasons=reasons,
    )


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _record_agentic_run_audit_artifact(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    final_verdict: dict[str, Any],
    typed_validations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    payload = _compile_agentic_run_audit_payload(
        blackboard=blackboard,
        action_batches=action_batches,
        validations=validations,
        typed_validations=typed_validations or [],
        tool_calls=tool_calls,
        final_verdict=final_verdict,
    )
    artifact = {
        "schema_version": "agentic_run_audit_artifact.v1",
        "artifact_type": "AgenticRunAudit",
        "artifact_id": f"{case_id}:agentic_run_audit",
        "case_id": case_id,
        "source": "agentic_blackboard_controller",
        "input_refs": [
            str(state.run_dir / "agent_blackboard.json"),
            *[str(state.run_dir / f"action_batch_round_{idx}.json") for idx in range(1, len(action_batches) + 1)],
        ],
        "evidence_refs": [],
        "validation_status": "accepted",
        "payload": payload,
        "artifact_ref": str(state.run_dir / "agentic_run_audit.json"),
    }
    write_json(state.run_dir / "agentic_run_audit.json", artifact)
    return artifact


def _record_hypothesis_only_retrosynthesis_report_artifact(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    payload = compile_hypothesis_only_retrosynthesis_report(
        blackboard=blackboard,
        artifacts=state.artifacts,
    )
    artifact = {
        "schema_version": "hypothesis_only_retrosynthesis_report_artifact.v1",
        "artifact_type": "HypothesisOnlyRetrosynthesisReport",
        "artifact_id": f"{case_id}:hypothesis_only_retrosynthesis_report",
        "case_id": case_id,
        "source": "agentic_blackboard_controller",
        "input_refs": [str(state.run_dir / "agent_blackboard.json")],
        "evidence_refs": [
            str(ref)
            for ref in (
                blackboard.get("artifact_refs") or {}
            ).values()
            if str(ref or "").strip()
        ][:20],
        "validation_status": "accepted",
        "payload": payload,
        "artifact_ref": str(state.run_dir / "hypothesis_only_retrosynthesis_report.json"),
    }
    write_json(state.run_dir / "hypothesis_only_retrosynthesis_report.json", artifact)
    return artifact


def _record_hypothesis_execution_report_artifact(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    hypothesis_report_artifact: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    payload = compile_hypothesis_execution_report(
        blackboard=blackboard,
        hypothesis_report=hypothesis_report_artifact,
        artifacts=state.artifacts,
    )
    artifact = {
        "schema_version": "hypothesis_execution_report_artifact.v1",
        "artifact_type": "HypothesisExecutionReport",
        "artifact_id": f"{case_id}:hypothesis_execution_report",
        "case_id": case_id,
        "source": "agentic_blackboard_controller",
        "input_refs": [
            str(state.run_dir / "agent_blackboard.json"),
            str(state.run_dir / "hypothesis_only_retrosynthesis_report.json"),
        ],
        "evidence_refs": [
            str(state.run_dir / "route_expansion_subgoal_search_result.json"),
            str(state.run_dir / "guided_chemenzy_result.json"),
        ],
        "validation_status": "accepted",
        "payload": payload,
        "artifact_ref": str(state.run_dir / "hypothesis_execution_report.json"),
    }
    write_json(state.run_dir / "hypothesis_execution_report.json", artifact)
    return artifact


def _record_route_forest_display_artifact(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    codex_campaign_config: RetrosynthesisTeamConfig | None = None,
) -> dict[str, Any]:
    case_id = str(blackboard.get("case_id") or state.preflight.get("case_id") or "case")
    forest_path = state.run_dir / "explored_route_forest.json"
    html_path = state.run_dir / "route_forest.html"
    refs = blackboard.setdefault("artifact_refs", {})
    refs["explored_route_forest"] = str(forest_path)
    refs["route_forest_html"] = str(html_path)
    try:
        trusted_stock_providers: dict[str, Any] = {}
        if codex_campaign_config is not None:
            trusted_stock_providers, _ = build_trusted_stock_provider_instances(
                stock_snapshots=codex_campaign_config.stock_snapshots,
                benchmark_catalog_artifact=(
                    codex_campaign_config.benchmark_stock_catalog_artifact
                ),
                benchmark_catalog_sha256=(
                    codex_campaign_config.benchmark_stock_catalog_sha256
                ),
                benchmark_catalog_name=(
                    codex_campaign_config.benchmark_stock_catalog_name
                ),
            )
        rendered = write_route_forest_artifacts(
            blackboard,
            run_dir=state.run_dir,
            forest_output=forest_path,
            html_output=html_path,
            trusted_stock_provider_instances=trusted_stock_providers,
        )
        forest = dict(rendered.get("forest") or {})
        primary_branch_id = str(forest.get("primary_branch_id") or "")
        primary_branch = next(
            (
                branch
                for branch in forest.get("branches") or []
                if isinstance(branch, dict) and str(branch.get("branch_id") or "") == primary_branch_id
            ),
            {},
        )
        payload = {
            "schema_version": "route_forest_display_payload.v1",
            "accepted": True,
            "read_only": True,
            "html_path": str(html_path),
            "forest_path": str(forest_path),
            "target": dict(forest.get("target") or {}),
            "counts": dict(forest.get("counts") or {}),
            "primary_branch": {
                "branch_id": str(primary_branch.get("branch_id") or ""),
                "title": str(primary_branch.get("title") or ""),
                "kind": str(primary_branch.get("kind") or ""),
                "step_count": len(primary_branch.get("step_ids") or []),
                "synthesis_class": str(primary_branch.get("synthesis_class") or "unspecified"),
                "solved": bool(primary_branch.get("solved")),
                "executable": bool(primary_branch.get("executable")),
                "advisory_only": bool(primary_branch.get("advisory_only", True)),
            },
            "primary_selection": dict(forest.get("primary_selection") or {}),
            "interaction_model": "main_route_with_step_scoped_read_only_replacement_preview",
        }
        append_jsonl(
            state.run_dir / "decision_trace.jsonl",
            {
                "stage": "route_forest_rendered",
                "html_path": str(html_path),
                "forest_path": str(forest_path),
                "counts": payload["counts"],
            },
        )
    except Exception as exc:  # pragma: no cover - defensive closeout path
        error_path = state.run_dir / "route_forest_error.json"
        refs["route_forest_error"] = str(error_path)
        payload = {
            "schema_version": "route_forest_display_payload.v1",
            "accepted": False,
            "read_only": True,
            "html_path": str(html_path),
            "forest_path": str(forest_path),
            "error_path": str(error_path),
            "error": repr(exc),
            "target": {},
            "counts": {},
        }
        write_json(
            error_path,
            {
                "schema_version": "route_forest_render_error.v1",
                "accepted": False,
                "case_id": case_id,
                "error": repr(exc),
                "html_path": str(html_path),
                "forest_path": str(forest_path),
            },
        )
        state.safety_flags.append("route_forest_render_failed")
        append_jsonl(
            state.run_dir / "decision_trace.jsonl",
            {
                "stage": "route_forest_render_failed",
                "error": repr(exc),
                "error_path": str(error_path),
            },
        )
    artifact = {
        "schema_version": "route_forest_display_artifact.v1",
        "artifact_type": "ExploredRouteForestDisplay",
        "artifact_id": f"{case_id}:route_forest_display",
        "case_id": case_id,
        "source": "agentic_blackboard_controller",
        "input_refs": [str(state.run_dir / "agent_blackboard.json")],
        "evidence_refs": [
            str(ref)
            for ref in (blackboard.get("artifact_refs") or {}).values()
            if str(ref or "").strip()
        ][:20],
        "validation_status": "accepted" if payload.get("accepted") else "draft",
        "payload": payload,
        "artifact_ref": str(html_path),
    }
    return artifact


def _commit_route_closeout_revision(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    final_verdict: FinalVerdict,
    final_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the closeout route projection to one immutable revision.

    Historical fixed-name paths remain available, but the revision manifest is
    authoritative whenever it exists.  A failed staging validation never
    replaces the previous latest pointer.
    """
    refs = blackboard.setdefault("artifact_refs", {})
    proof_snapshot_path = state.run_dir / "parent_route_proof_snapshot.json"
    verdict_core_path = state.run_dir / "final_verdict_core.json"
    proof = dict(blackboard.get("parent_route_proof") or {})
    target_profile = dict(blackboard.get("target_profile") or {})
    proof_snapshot = {
        "schema_version": "parent_route_proof_snapshot.v1",
        "case_id": str(
            blackboard.get("case_id")
            or state.preflight.get("case_id")
            or state.target_input.get("case_id")
            or "case"
        ),
        "target_smiles": str(
            target_profile.get("target_smiles")
            or state.target_input.get("target_smiles")
            or ""
        ),
        "proof_schema_version": str(proof.get("schema_version") or "missing"),
        "solved": _blackboard_parent_proof_solved(blackboard, proof),
        "authority": "deterministic_parent_route_proof",
        "proof": proof,
    }
    verdict_payload = final_verdict.to_dict()
    # The compatibility verdict gains CAS digest references after publication.
    # Omitting those two presentation/reference fields gives the decision a
    # finite, content-addressable core instead of creating a self-reference.
    verdict_payload.pop("artifact_refs", None)
    verdict_payload.pop("artifact_digest_refs", None)
    validation = dict(final_validation or {})
    effective_validation = dict(validation.get("corrected_validation") or validation)
    verdict_core = {
        "schema_version": "final_verdict_core.v1",
        "case_id": str(verdict_payload.get("case_id") or proof_snapshot["case_id"]),
        "authority": "deterministic_parent_route_proof",
        "parent_route_proof_solved": proof_snapshot["solved"],
        "validation": {
            "schema_version": str(effective_validation.get("schema_version") or ""),
            "accepted": effective_validation.get("accepted") is True,
            "original_attempt_accepted": validation.get("accepted") is True,
            "reasons": [str(item) for item in effective_validation.get("reasons") or []],
        },
        "verdict": verdict_payload,
    }
    write_json(proof_snapshot_path, proof_snapshot)
    write_json(verdict_core_path, verdict_core)
    refs["parent_route_proof_snapshot"] = str(proof_snapshot_path)
    refs["final_verdict_core"] = str(verdict_core_path)
    state.artifacts["parent_route_proof_snapshot"] = proof_snapshot
    state.artifacts["final_verdict_core"] = verdict_core
    candidate_keys = (
        "route_consensus_rebuild",
        "route_consensus",
        "route_consensus_graph",
        "canonical_route_consensus_graph",
        "codex_edge_verification",
        "codex_campaign_proof_reconciliation",
        "frontier_ledger",
        "parent_route_proof_snapshot",
        "final_verdict_core",
        "explored_route_forest",
        "route_forest_html",
    )
    run_root = state.run_dir.resolve()
    artifacts: dict[str, Path] = {}
    for artifact_id in candidate_keys:
        ref = str(refs.get(artifact_id) or "").strip()
        if not ref:
            continue
        path = Path(ref).expanduser()
        path = path.resolve() if path.is_absolute() else (run_root / path).resolve()
        try:
            path.relative_to(run_root)
        except ValueError:
            continue
        if path.is_file():
            artifacts[artifact_id] = path

    if not artifacts:
        failure = {
            "schema_version": "closeout_revision_state.v1",
            "accepted": False,
            "status": "not_committed",
            "reasons": ["no_closeout_route_artifacts"],
            "authority": "compatibility_paths_only",
        }
        blackboard["closeout_revision"] = failure
        state.safety_flags.append("closeout_revision_not_committed")
        return failure

    dependencies: dict[str, tuple[str, ...]] = {}
    if "route_consensus_rebuild" in artifacts and "route_consensus" in artifacts:
        dependencies["route_consensus"] = ("route_consensus_rebuild",)
    if "route_consensus" in artifacts and "route_consensus_graph" in artifacts:
        dependencies["route_consensus_graph"] = ("route_consensus",)
    if (
        "route_consensus_graph" in artifacts
        and "codex_campaign_proof_reconciliation" in artifacts
    ):
        reconciliation_dependencies = ["route_consensus_graph"]
        if "codex_edge_verification" in artifacts:
            reconciliation_dependencies.append("codex_edge_verification")
        dependencies["codex_campaign_proof_reconciliation"] = tuple(
            reconciliation_dependencies
        )
    if (
        "codex_campaign_proof_reconciliation" in artifacts
        and "canonical_route_consensus_graph" in artifacts
    ):
        dependencies["canonical_route_consensus_graph"] = (
            "codex_campaign_proof_reconciliation",
        )
    if "route_consensus_graph" in artifacts and "codex_edge_verification" in artifacts:
        dependencies["codex_edge_verification"] = ("route_consensus_graph",)
    authority_graph_artifact = (
        "canonical_route_consensus_graph"
        if "canonical_route_consensus_graph" in artifacts
        else "route_consensus_graph"
        if "route_consensus_graph" in artifacts
        else ""
    )
    if authority_graph_artifact and "frontier_ledger" in artifacts:
        ledger_dependencies = [authority_graph_artifact]
        if "codex_campaign_proof_reconciliation" in artifacts:
            ledger_dependencies.append("codex_campaign_proof_reconciliation")
        dependencies["frontier_ledger"] = tuple(ledger_dependencies)
    if authority_graph_artifact and "parent_route_proof_snapshot" in artifacts:
        dependencies["parent_route_proof_snapshot"] = (
            authority_graph_artifact,
        )
    if "parent_route_proof_snapshot" in artifacts and "final_verdict_core" in artifacts:
        dependencies["final_verdict_core"] = ("parent_route_proof_snapshot",)
    forest_dependencies = tuple(
        artifact_id
        for artifact_id in (
            "route_consensus",
            "route_consensus_graph",
            "canonical_route_consensus_graph",
            "frontier_ledger",
            "parent_route_proof_snapshot",
            "final_verdict_core",
        )
        if artifact_id in artifacts
    )
    if "explored_route_forest" in artifacts and forest_dependencies:
        dependencies["explored_route_forest"] = forest_dependencies
    if "explored_route_forest" in artifacts and "route_forest_html" in artifacts:
        dependencies["route_forest_html"] = ("explored_route_forest",)

    expected_digests = {
        artifact_id: sha256_file(path)
        for artifact_id, path in artifacts.items()
    }
    try:
        published = publish_closeout_revision(
            run_root,
            artifacts=artifacts,
            dependencies=dependencies,
            producer="agentic_blackboard_controller.closeout",
            case_id=str(blackboard.get("case_id") or state.preflight.get("case_id") or "case"),
            expected_digests=expected_digests,
        )
    except (ArtifactRevisionError, OSError) as exc:
        failure = {
            "schema_version": "closeout_revision_state.v1",
            "accepted": False,
            "status": "not_committed",
            "reasons": [f"closeout_revision_commit_failed:{type(exc).__name__}:{exc}"],
            "authority": "compatibility_paths_only",
        }
        blackboard["closeout_revision"] = failure
        state.safety_flags.append("closeout_revision_not_committed")
        append_jsonl(
            state.run_dir / "decision_trace.jsonl",
            {"stage": "closeout_revision_rejected", **failure},
        )
        return failure

    revision_id = str(published.get("revision_id") or "")
    manifest_path = str(published.get("manifest_path") or "")
    pointer_path = str(published.get("latest_pointer_path") or "")
    digest_refs: dict[str, dict[str, Any]] = {}
    for row in (published.get("manifest") or {}).get("artifacts") or []:
        if not isinstance(row, dict):
            continue
        artifact_id = str(row.get("artifact_id") or "")
        if not artifact_id:
            continue
        digest_refs[artifact_id] = {
            "schema_version": "closeout_artifact_digest_ref.v1",
            "artifact_id": artifact_id,
            "path": str(row.get("path") or ""),
            "content_path": str(row.get("content_path") or ""),
            "sha256": str(row.get("sha256") or ""),
            "artifact_schema_version": str(row.get("artifact_schema_version") or ""),
            "producer": str(row.get("producer") or ""),
            "dependencies": [
                dict(item)
                for item in row.get("dependencies") or []
                if isinstance(item, dict)
            ],
            "revision_id": revision_id,
            "manifest_path": manifest_path,
            "manifest_sha256": str(published.get("manifest_sha256") or ""),
        }
    closeout_state = {
        "schema_version": "closeout_revision_state.v1",
        "accepted": True,
        "status": "committed",
        "revision_id": revision_id,
        "manifest_path": manifest_path,
        "manifest_sha256": str(published.get("manifest_sha256") or ""),
        "latest_pointer_path": pointer_path,
        "authority": "content_addressed_closeout_manifest",
        "artifact_count": len(digest_refs),
    }
    refs["closeout_revision_manifest"] = manifest_path
    refs["closeout_latest_pointer"] = pointer_path
    blackboard["artifact_digest_refs"] = digest_refs
    blackboard["closeout_revision"] = closeout_state
    final_verdict.artifact_digest_refs = digest_refs
    final_verdict.artifact_refs = dict(refs)
    state.artifacts["closeout_revision_manifest"] = dict(published.get("manifest") or {})
    state.artifacts["closeout_revision"] = closeout_state
    state.validations.append(dict(published.get("validation") or {}))
    append_jsonl(
        state.run_dir / "decision_trace.jsonl",
        {
            "stage": "closeout_revision_committed",
            "revision_id": revision_id,
            "manifest_path": manifest_path,
            "latest_pointer_path": pointer_path,
            "artifact_count": len(digest_refs),
        },
    )
    return closeout_state


def _compile_agentic_run_audit_payload(
    *,
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    final_verdict: dict[str, Any],
    typed_validations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    belief = dict(blackboard.get("current_belief") or {})
    parent_proof = dict(blackboard.get("parent_route_proof") or {})
    route_proof_bundle = dict(blackboard.get("route_proof_bundle") or {})
    objective_summary = dict(blackboard.get("route_objective_summary") or {})
    round_summaries = _round_summaries_from_blackboard(
        blackboard=blackboard,
        action_batches=action_batches,
        validations=validations,
    )
    validation_reasons = [
        str(reason)
        for validation in validations
        for reason in validation.get("reasons") or []
        if str(reason or "").strip()
    ]
    unresolved_reasons = _dedupe(
        [
            *[str(item) for item in final_verdict.get("reasons") or []],
            *[str(item) for item in parent_proof.get("reasons") or []],
            *validation_reasons,
            *[str(row.get("reason") or "") for row in blackboard.get("route_failures") or [] if isinstance(row, dict)],
        ]
    )
    return {
        "schema_version": "agentic_blackboard_run_audit.v1",
        "case_id": str(blackboard.get("case_id") or ""),
        "audit_authority": "diagnostic_only",
        "deterministic_final_verdict_required": True,
        "round_summaries": round_summaries,
        "budget_state": dict(blackboard.get("budget_state") or {}),
        "evidence_counts": {
            "source_candidates": len(evidence.get("source_candidates") or []),
            "source_lifecycle": len(evidence.get("source_lifecycle") or []),
            "pdf_structure_evidence": len(evidence.get("pdf_structure_evidence") or []),
            "visual_chains": len(evidence.get("visual_chains") or []),
            "exact_rows": len(evidence.get("exact_rows") or []),
            "terminal_candidates": len(evidence.get("terminal_candidates") or []),
            "structure_resolution_tasks": len(evidence.get("structure_resolution_tasks") or []),
            "bridge_tasks": len(blackboard.get("bridge_tasks") or []),
            "route_objectives": len(objective_summary.get("objectives") or []),
            "endpoint_candidates": len(blackboard.get("endpoint_candidates") or []),
            "broad_transform_templates": len(blackboard.get("broad_transform_templates") or []),
            "analogical_templates": len(blackboard.get("analogical_templates") or []),
            "template_applications": len(blackboard.get("template_applications") or []),
            "route_failures": len(blackboard.get("route_failures") or []),
        },
        "planner_summary": {
            "round_count": len(blackboard.get("planner_history") or []),
            "codex_attempted_rounds": sum(
                1
                for row in blackboard.get("planner_history") or []
                if isinstance(row, dict) and (row.get("codex_action_planner") or {}).get("attempted")
            ),
            "fallback_rounds": [
                int(row.get("round_index") or 0)
                for row in blackboard.get("planner_history") or []
                if isinstance(row, dict) and (row.get("codex_action_planner") or {}).get("fallback_used")
            ],
            "planner_notes": list(belief.get("planner_notes") or []),
        },
        "blackboard_transition_summary": _blackboard_transition_summary(blackboard),
        "source_acquisition_summary": _source_acquisition_summary(evidence),
        "analogical_template_summary": _analogical_template_summary(blackboard),
        "typed_artifact_validation_summary": _typed_artifact_validation_summary(typed_validations or []),
        "tool_call_count": len(tool_calls),
        "parent_route_proof": {
            "accepted": bool(parent_proof.get("accepted")),
            "solved": bool(parent_proof.get("solved")),
            "route_status": str(parent_proof.get("route_status") or ""),
            "reasons": [str(item) for item in parent_proof.get("reasons") or []],
        },
        "route_objective_summary": {
            "selected_objective_types": [
                str(row.get("objective_type") or "")
                for row in objective_summary.get("selected_objectives") or []
                if isinstance(row, dict)
            ],
            "route_scope": str((objective_summary.get("route_scope") or {}).get("route_scope") or ""),
            "small_molecule_stock_closure_deprioritized": bool(
                (objective_summary.get("route_scope") or {}).get("small_molecule_stock_closure_deprioritized")
            ),
        },
        "route_proof_bundle": {
            "accepted": bool(route_proof_bundle.get("accepted")),
            "solved": bool(route_proof_bundle.get("solved")),
            "route_status": str(route_proof_bundle.get("route_status") or ""),
            "objective_proof_count": len(route_proof_bundle.get("objective_proofs") or []),
        },
        "final_verdict": dict(final_verdict),
        "unresolved_reasons": unresolved_reasons,
        "followup_tasks": _followup_tasks_from_blackboard(
            blackboard=blackboard,
            final_verdict=final_verdict,
        ),
        "safety_invariants": {
            "analogy_is_advisory_only": bool((belief.get("template_policy") or {}).get("analogy_is_advisory_only", True)),
            "child_route_never_promotes_parent_solved": True,
            "parent_proof_required_for_solved": True,
            "raw_reaction_output_allowed": False,
        },
    }


def _followup_tasks_from_blackboard(
    *,
    blackboard: dict[str, Any],
    final_verdict: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    parent_proof = dict(blackboard.get("parent_route_proof") or {})
    tasks: list[dict[str, Any]] = []

    queue_path = str((evidence.get("local_pdf_proxy_request_summary") or {}).get("queue_path") or "")
    for idx, request in enumerate(evidence.get("local_pdf_proxy_requests") or [], start=1):
        if not isinstance(request, dict):
            continue
        tasks.append(
            {
                "schema_version": "agentic_followup_task.v1",
                "task_id": f"await_local_pdf:{_safe_task_token(request.get('request_id') or request.get('source_ref') or idx)}",
                "task_type": "await_local_pdf_proxy_download",
                "priority": 100,
                "status": "external_input_required",
                "source_ref": str(request.get("source_ref") or ""),
                "doi": str(request.get("doi") or ""),
                "url": str(request.get("url") or ""),
                "queue_path": queue_path,
                "reason": "agent_found_metadata_but_no_agent_readable_pdf",
                "resume_hint": "sync downloaded PDF manifest, then rerun source extraction for this source",
                "recommended_next_action": "extract_pdf_literature_structures",
                "no_solved_claim": True,
            }
        )

    for idx, task in enumerate(evidence.get("structure_resolution_tasks") or [], start=1):
        if not isinstance(task, dict):
            continue
        status = str(task.get("status") or "open")
        if status not in {"open", "pending", "unresolved"}:
            continue
        tasks.append(
            {
                "schema_version": "agentic_followup_task.v1",
                "task_id": str(task.get("task_id") or f"resolve_structure:{idx}"),
                "task_type": "resolve_literature_structure",
                "priority": 80,
                "status": "open",
                "source_ref": str(task.get("source_ref") or ""),
                "label": str(task.get("label") or ""),
                "artifact_ref": str(task.get("artifact_ref") or ""),
                "reason": str(task.get("reason") or "visual_structure_not_confidently_convertible_to_smiles"),
                "resume_hint": "provide source detail, supplementary information, or name-to-structure evidence",
                "recommended_next_action": "search_literature",
                "no_solved_claim": True,
            }
        )

    exact_rows = [dict(row) for row in evidence.get("exact_rows") or [] if isinstance(row, dict)]
    if exact_rows and not _blackboard_parent_proof_solved(blackboard, parent_proof):
        tasks.append(
            {
                "schema_version": "agentic_followup_task.v1",
                "task_id": "prove_parent_route_connectivity",
                "task_type": "prove_parent_route_connectivity",
                "priority": 70,
                "status": "open",
                "exact_row_count": len(exact_rows),
                "reason": "exact_literature_rows_exist_but_parent_route_proof_not_solved",
                "resume_hint": "run parent stitching only after child/parent route and exact literature segment are all bound",
                "recommended_next_action": "stitch_parent_route",
                "no_solved_claim": True,
            }
        )

    if (
        blackboard.get("analogical_templates")
        and not blackboard.get("template_applications")
        and not exact_rows
    ):
        tasks.append(
            {
                "schema_version": "agentic_followup_task.v1",
                "task_id": "apply_ranked_analogical_templates",
                "task_type": "apply_analogical_templates_under_guard",
                "priority": 60,
                "status": "open",
                "template_count": len(blackboard.get("analogical_templates") or []),
                "reason": "analogical_templates_available_but_not_applied",
                "resume_hint": "apply templates only as guided-search hints, never as proof",
                "recommended_next_action": "rank_analogical_reaction_templates",
                "no_solved_claim": True,
            }
        )

    for idx, bridge in enumerate(blackboard.get("bridge_tasks") or [], start=1):
        if not isinstance(bridge, dict):
            continue
        tasks.append(
            {
                "schema_version": "agentic_followup_task.v1",
                "task_id": str(bridge.get("task_id") or f"bridge_task:{idx}"),
                "task_type": "continue_bridge_task",
                "priority": 50,
                "status": "open",
                "bridge_task_type": str(bridge.get("task_type") or ""),
                "target_handle": str(bridge.get("target_handle") or ""),
                "reason": str(bridge.get("required_bridge") or bridge.get("reason") or "bridge_task_unresolved"),
                "resume_hint": "search or guided search should target this bridge before claiming parent closure",
                "recommended_next_action": "search_literature",
                "no_solved_claim": True,
            }
        )

    solved = bool(final_verdict.get("solved")) or str(final_verdict.get("verdict") or "") == "solved"
    if not solved and not tasks:
        tasks.append(
            {
                "schema_version": "agentic_followup_task.v1",
                "task_id": "continue_agentic_blackboard",
                "task_type": "continue_agentic_blackboard",
                "priority": 10,
                "status": "open",
                "reason": "unresolved_without_specific_pending_artifact",
                "resume_hint": "rerun with broader source search or inspect failure critic output",
                "recommended_next_action": "build_failure_critic_report",
                "no_solved_claim": True,
            }
        )

    return _dedupe_followup_tasks(tasks)[:20]


def _dedupe_followup_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in sorted(tasks, key=lambda row: (-int(row.get("priority") or 0), str(row.get("task_id") or ""))):
        key = str(task.get("task_id") or task.get("task_type") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(task)
    return out


def _safe_task_token(value: Any) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "task"))
    return "_".join(part for part in safe.split("_") if part)[:120] or "task"


def _analogical_template_summary(blackboard: dict[str, Any]) -> dict[str, Any]:
    ranking = dict(blackboard.get("analogical_template_ranking") or {})
    applications = [dict(row) for row in blackboard.get("template_applications") or [] if isinstance(row, dict)]
    policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    return {
        "schema_version": "agent_analogical_template_summary.v1",
        "template_count": len(blackboard.get("analogical_templates") or []),
        "selected_template_ids": [
            str(row.get("template_id") or "")
            for row in ranking.get("selected_templates") or []
            if isinstance(row, dict) and str(row.get("template_id") or "").strip()
        ],
        "application_count": len(applications),
        "accepted_application_count": sum(1 for row in applications if row.get("accepted")),
        "executable_candidate_hint_count": sum(1 for row in applications if row.get("executable_candidate_available")),
        "allowed_uses": _dedupe([str(row.get("allowed_use") or "") for row in applications]),
        "validated_one_step_row_count": int(policy.get("validated_one_step_row_count") or 0),
        "validated_guided_hint_count": int(policy.get("validated_guided_hint_count") or 0),
        "analogy_is_advisory_only": bool(policy.get("analogy_is_advisory_only", True)),
        "analogical_template_hints_are_not_exact_rows": bool(policy.get("analogical_template_hints_are_not_exact_rows", True)),
        "final_verdict_authority": "none",
        "requires_parent_route_proof": True,
        "no_solved_claim": True,
    }


def _source_acquisition_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    candidates = [dict(row) for row in evidence.get("source_candidates") or [] if isinstance(row, dict)]
    planner_hints = [dict(row) for row in evidence.get("planner_source_hints") or [] if isinstance(row, dict)]
    lifecycle = [dict(row) for row in evidence.get("source_lifecycle") or [] if isinstance(row, dict)]
    lifecycle_stage_counts = _source_lifecycle_stage_counts(lifecycle)
    attempts = [dict(row) for row in evidence.get("scout_attempts") or [] if isinstance(row, dict)]
    real_candidates = [row for row in candidates if _candidate_has_real_source(row)]
    identity_summary = dict(evidence.get("source_identity_summary") or {})
    independent_source_groups = {
        str(row.get("independent_source_group") or "")
        for row in lifecycle
        if str(row.get("independent_source_group") or "")
    }
    independent_source_count = int(
        identity_summary["independent_source_group_count"]
        if "independent_source_group_count" in identity_summary
        else len(independent_source_groups)
    )
    source_document_count = int(
        identity_summary["document_count"]
        if "document_count" in identity_summary
        else len(lifecycle)
    )
    source_representation_count = int(
        identity_summary["representation_count"]
        if "representation_count" in identity_summary
        else len(
            {
                str(item)
                for row in lifecycle
                for item in row.get("representations") or []
                if str(item or "")
            }
        )
    )
    local_pdf_match_candidates = [row for row in candidates if isinstance(row.get("local_pdf_match"), dict)]
    auto_cache_candidates = [row for row in candidates if isinstance(row.get("local_pdf_index"), dict)]
    auto_cache_blind_candidates = [
        row
        for row in auto_cache_candidates
        if not isinstance(row.get("local_pdf_match"), dict)
        and (row.get("local_pdf_index") or {}).get("match_policy") == "agent_discovered_metadata_required"
    ]
    local_pdf_match_bases = _dedupe(
        [
            str((row.get("local_pdf_match") or {}).get("match_basis") or "")
            for row in local_pdf_match_candidates
            if isinstance(row.get("local_pdf_match"), dict)
        ]
    )
    placeholder_candidates = [
        row
        for row in candidates
        if bool(row.get("placeholder_only")) or str(row.get("access_status") or "").strip().lower() == "placeholder_only"
    ]
    user_seed_candidates = [row for row in candidates if _candidate_is_user_provided_local_pdf_seed(row)]
    return {
        "schema_version": "agent_source_acquisition_summary.v1",
        "source_discovery_mode": str(evidence.get("source_discovery_mode") or ""),
        "confidence": str(evidence.get("confidence") or ""),
        "planner_source_hint_count": len(planner_hints),
        "planner_source_hints_are_not_evidence": True,
        "source_lifecycle_count": len(lifecycle),
        "source_lifecycle_stage_counts": lifecycle_stage_counts,
        "fallback_order": [str(item) for item in evidence.get("fallback_order") or []],
        "scout_attempts": attempts,
        "codex_online_attempted": _source_acquisition_codex_online_attempted(evidence),
        "local_pdf_attempted": any(str(row.get("mode") or "") in {"local_pdf", "local_pdf_cache"} and row.get("attempted") for row in attempts),
        "placeholder_used": bool(placeholder_candidates) or str(evidence.get("source_discovery_mode") or "") == "placeholder",
        # Historical candidate rows could count an article URL, its local PDF,
        # and its SI as three "sources".  Independence is publication/family
        # based; retain the candidate count under an explicit diagnostic name.
        "real_source_count": independent_source_count,
        "real_source_candidate_record_count": len(real_candidates),
        "source_document_count": source_document_count,
        "independent_source_group_count": independent_source_count,
        "source_representation_count": source_representation_count,
        "source_identity_semantics": {
            "real_source_count_means_independent_source_groups": True,
            "document_and_representation_counts_are_not_independence": True,
        },
        "placeholder_candidate_count": len(placeholder_candidates),
        "local_pdf_available_count": sum(1 for row in candidates if str(row.get("local_pdf") or "").strip()),
        "user_provided_local_pdf_seed_count": len(user_seed_candidates),
        "direct_local_pdf_after_codex_failure_count": sum(
            1
            for row in user_seed_candidates
            if str(row.get("source_discovery_mode") or evidence.get("source_discovery_mode") or "") == "local_pdf_fallback"
        ),
        "local_pdf_cache_match_count": len(local_pdf_match_candidates),
        "auto_local_pdf_cache_match_count": sum(
            1 for row in local_pdf_match_candidates if isinstance(row.get("local_pdf_index"), dict)
        ),
        "agent_discovered_local_pdf_match_count": sum(
            1
            for row in local_pdf_match_candidates
            if _local_pdf_match_has_agent_discovered_metadata(dict(row.get("local_pdf_match") or {}))
        ),
        "local_pdf_match_bases": local_pdf_match_bases,
        "auto_local_pdf_blind_fallback_used": bool(auto_cache_blind_candidates),
        "metadata_only_count": sum(1 for row in candidates if str(row.get("access_status") or "") == "metadata_only"),
        "local_pdf_proxy_request_count": len(evidence.get("local_pdf_proxy_requests") or []),
        "awaiting_local_pdf_proxy_count": int(lifecycle_stage_counts.get("local_pdf_proxy_requested") or 0),
        "local_pdf_proxy_queue_path": str(
            (evidence.get("local_pdf_proxy_request_summary") or {}).get("queue_path") or ""
        ),
        "source_refs": [str(item) for item in evidence.get("source_refs") or []],
        "no_solved_claim": True,
    }


def _source_lifecycle_stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("stage") or "unresolved")
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def _typed_artifact_validation_summary(validations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in validations
        if isinstance(row, dict) and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
    ]
    failed = [row for row in rows if not row.get("accepted")]
    return {
        "schema_version": "agentic_typed_artifact_validation_summary.v1",
        "validated_artifact_count": len(rows),
        "failed_artifact_count": len(failed),
        "accepted_artifact_keys": [
            str(row.get("artifact_key") or "")
            for row in rows
            if row.get("accepted") and str(row.get("artifact_key") or "").strip()
        ],
        "failed_artifact_keys": [
            str(row.get("artifact_key") or "")
            for row in failed
            if str(row.get("artifact_key") or "").strip()
        ],
        "failure_reasons": _dedupe(
            [
                str(reason)
                for row in failed
                for reason in row.get("reasons") or []
                if str(reason or "").strip()
            ]
        ),
        "no_solved_claim": True,
    }


def _safe_artifact_filename(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "artifact"))
    return "_".join(part for part in safe.split("_") if part) or "artifact"


def _round_summaries_from_blackboard(
    *,
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    validations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    history_by_round: dict[int, list[dict[str, Any]]] = {}
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        history_by_round.setdefault(int(row.get("round_index") or 0), []).append(dict(row))
    planner_by_round = {
        int(row.get("round_index") or 0): dict(row)
        for row in blackboard.get("planner_history") or []
        if isinstance(row, dict)
    }
    summaries: list[dict[str, Any]] = []
    for idx, batch in enumerate(action_batches, start=1):
        validation = dict(validations[idx - 1]) if idx - 1 < len(validations) else {}
        rows = history_by_round.get(idx, [])
        summaries.append(
            {
                "schema_version": "agentic_run_round_summary.v1",
                "round_index": idx,
                "planner_mode": str(batch.get("mode") or ""),
                "planner": planner_by_round.get(idx, {}),
                "validation_accepted": bool(validation.get("accepted")),
                "validation_reasons": [str(item) for item in validation.get("reasons") or []],
                "planned_action_types": [
                    str(action.get("action_type") or "")
                    for action in batch.get("actions") or []
                    if isinstance(action, dict)
                ],
                "executed_action_types": [str(row.get("action_type") or "") for row in rows],
                "useful_artifact_count": sum(1 for row in rows if row.get("useful_artifact")),
                "stale_action_count": sum(1 for row in rows if row.get("stale")),
                "changed_blackboard_fields": _dedupe(
                    [
                        str(field)
                        for row in rows
                        for field in row.get("changed_blackboard_fields") or []
                        if str(field or "").strip()
                    ]
                ),
                "blackboard_deltas": [
                    {
                        "action_type": str(row.get("action_type") or ""),
                        "action_id": str(row.get("action_id") or ""),
                        "delta": dict(row.get("blackboard_delta") or {}),
                    }
                    for row in rows
                ],
                "reasons": _dedupe(
                    [
                        str(reason)
                        for row in rows
                        for reason in row.get("reasons") or []
                        if str(reason or "").strip()
                    ]
                ),
            }
        )
    return summaries


def _blackboard_transition_summary(blackboard: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    total_delta: dict[str, int] = {}
    changed_fields: list[str] = []
    useful_changed = 0
    stale_changed = 0
    for row in rows:
        delta = dict(row.get("blackboard_delta") or {})
        if delta and row.get("useful_artifact"):
            useful_changed += 1
        if delta and row.get("stale"):
            stale_changed += 1
        for key, value in delta.items():
            try:
                amount = int(value)
            except (TypeError, ValueError):
                continue
            total_delta[str(key)] = int(total_delta.get(str(key), 0)) + amount
            changed_fields.append(str(key))
    return {
        "schema_version": "agent_blackboard_transition_summary.v1",
        "action_transition_count": len(rows),
        "changed_transition_count": sum(1 for row in rows if dict(row.get("blackboard_delta") or {})),
        "useful_changed_transition_count": useful_changed,
        "stale_changed_transition_count": stale_changed,
        "changed_blackboard_fields": _dedupe(changed_fields),
        "total_delta": {key: value for key, value in sorted(total_delta.items()) if value},
        "no_solved_claim": True,
    }


def _auto_update_critic(blackboard: dict[str, Any], *, state: ToolExecutionState, run_dir: Path, round_index: int) -> dict[str, Any]:
    if not _new_failure_evidence_since_last_critic(blackboard):
        return blackboard
    report = compile_failure_critic_report(
        blackboard=blackboard,
        artifacts=state.artifacts,
        case_id=str(state.preflight.get("case_id") or ""),
        target_name=str(state.target_input.get("target_name") or ""),
    )
    if not report.get("accepted"):
        return blackboard
    state.artifacts["failure_critic_report"] = report
    action = {
        "action_id": f"r{round_index}:auto_failure_critic",
        "action_type": "build_failure_critic_report",
        "payload": {},
    }
    return update_blackboard_from_action(
        blackboard,
        action=action,
        action_result={"accepted": True, "result": report},
        round_index=round_index,
        run_dir=run_dir,
    )


def _new_failure_evidence_since_last_critic(blackboard: dict[str, Any]) -> bool:
    if not (
        blackboard.get("route_failures")
        or any(
            isinstance(row, dict) and row.get("reasons")
            for row in blackboard.get("action_history") or []
        )
    ):
        return False
    last_critic = _last_action_round_for_controller(blackboard, "build_failure_critic_report")
    if last_critic <= 0:
        return True
    failure_producing_actions = {
        "run_guided_chemenzy",
        "expand_child_target",
        "stitch_parent_route",
        "compile_exact_literature_rows",
        "validate_template_application",
    }
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        try:
            action_round = int(row.get("round_index") or 0)
        except (TypeError, ValueError):
            action_round = 0
        if action_round <= last_critic:
            continue
        if str(row.get("action_type") or "") not in failure_producing_actions:
            continue
        if row.get("reasons") or row.get("useful_artifact"):
            return True
    return False


def _last_action_round_for_controller(blackboard: dict[str, Any], action_type: str) -> int:
    rounds: list[int] = []
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != action_type:
            continue
        try:
            rounds.append(int(row.get("round_index") or 0))
        except (TypeError, ValueError):
            continue
    return max(rounds) if rounds else 0


def _seed_failure_evidence_from_prior(blackboard: dict[str, Any], *, state: ToolExecutionState) -> dict[str, Any]:
    report = compile_failure_critic_report(
        blackboard=blackboard,
        artifacts=state.artifacts,
        case_id=str(state.preflight.get("case_id") or ""),
        target_name=str(state.target_input.get("target_name") or ""),
    )
    if not report.get("accepted"):
        return blackboard
    board = dict(blackboard)
    board["route_failures"] = list(report.get("route_failures") or [])
    board["bridge_tasks"] = list(report.get("bridge_tasks") or [])
    board["terminal_blacklist"] = list(report.get("terminal_blacklist") or [])
    belief = dict(board.get("current_belief") or {})
    belief["blocked_directions"] = list(report.get("blocked_directions") or [])
    belief["next_action_bias"] = [str(item) for item in report.get("next_action_bias") or [] if str(item).strip()]
    constraints = dict(belief.get("constraints") or {})
    constraints.update(dict(report.get("constraints") or {}))
    belief["constraints"] = constraints
    board["current_belief"] = belief
    if report.get("source_reasons"):
        board["plugin_runtime_diagnostics"] = [
            {
                "schema_version": "agent_prior_failure_source_reasons.v1",
                "reasons": [str(item) for item in report.get("source_reasons") or []],
            }
        ]
    state.artifacts["failure_critic_report"] = report
    return board


def _seed_prior_analogical_evidence(blackboard: dict[str, Any], *, state: ToolExecutionState) -> dict[str, Any]:
    board = dict(blackboard)
    rows: list[dict[str, Any]] = []
    explicit_rows = state.artifacts.get("analogical_hypotheses")
    if isinstance(explicit_rows, list):
        rows.extend(dict(row) for row in explicit_rows if isinstance(row, dict))
    analogical_artifact = state.artifacts.get("analogical_retrosynthesis_hypotheses")
    if isinstance(analogical_artifact, dict):
        rows.extend(dict(row) for row in analogical_artifact.get("hypotheses") or [] if isinstance(row, dict))
        board["analogical_retrosynthesis_hypotheses"] = _drop_large_fields(analogical_artifact)
    ranking = state.artifacts.get("analogical_hypothesis_ranking")
    if isinstance(ranking, dict):
        board["analogical_hypothesis_ranking"] = _drop_large_fields(ranking)
        rows.extend(dict(row) for row in ranking.get("ranked_hypotheses") or [] if isinstance(row, dict))
        rows.extend(dict(row) for row in ranking.get("selected_hypotheses") or [] if isinstance(row, dict))
    if not rows and not isinstance(ranking, dict):
        return board
    _extend_unique_by_key(board, "analogical_hypotheses", rows, "hypothesis_id")
    report = compile_retrosynthetic_proposal_bus(board)
    _extend_unique_by_key(board, "reaction_idea_cards", report.get("reaction_idea_cards") or [], "card_id")
    _extend_unique_by_key(board, "retrosynthetic_proposals", report.get("retrosynthetic_proposals") or [], "proposal_id")
    _extend_unique_by_key(board, "recursive_hypothesis_tasks", report.get("recursive_hypothesis_tasks") or [], "task_id")
    board["retrosynthetic_proposal_compile_report"] = {
        "schema_version": str(report.get("schema_version") or "retrosynthetic_proposal_compile_report.v1"),
        "accepted": bool(report.get("accepted")),
        "counts": dict(report.get("counts") or {}),
        "source": "prior_analogical_evidence_seed",
        "allowed_use": "proposal_bus_and_recursive_search_seed_only",
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }
    if report.get("recursive_hypothesis_tasks"):
        belief = dict(board.get("current_belief") or {})
        bias = [str(item) for item in belief.get("next_action_bias") or [] if str(item).strip()]
        if "expand_child_target" not in bias:
            bias.append("expand_child_target")
        belief["next_action_bias"] = bias
        board["current_belief"] = belief
    return board


def _extend_unique_by_key(board: dict[str, Any], key: str, rows: list[Any], unique_key: str) -> None:
    existing = list(board.get(key) or [])
    seen = {
        str(row.get(unique_key) or "")
        for row in existing
        if isinstance(row, dict) and str(row.get(unique_key) or "").strip()
    }
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        marker = str(raw.get(unique_key) or "").strip()
        if marker and marker in seen:
            continue
        existing.append(dict(raw))
        if marker:
            seen.add(marker)
    board[key] = existing


def _compile_parent_proof_from_state(
    *,
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    stitched: dict[str, Any],
) -> dict[str, Any]:
    parent_verifier = {} if stitched.get("accepted") else _latest_parent_verifier(state.artifacts)
    direct_parent_route = _direct_parent_verifier_ready_for_proof(blackboard, parent_verifier) and not stitched.get("accepted")
    route_expansion = {} if direct_parent_route else dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    exact_rows = [] if direct_parent_route else (blackboard.get("literature_evidence") or {}).get("exact_rows") or []
    exact_segment = {
        "accepted": bool(exact_rows),
        "parent_route_connected": bool(stitched.get("accepted")),
        "row_count": len(exact_rows),
    }
    return compile_stitched_parent_route_proof(
        target_smiles=str(state.target_input.get("target_smiles") or ""),
        target_name=str(state.target_input.get("target_name") or ""),
        case_id=str(state.preflight.get("case_id") or ""),
        parent_verifier=parent_verifier,
        stitched_route=stitched,
        child_route=route_expansion,
        exact_literature_segment=exact_segment,
        analogy_refs=(blackboard.get("analogical_hypothesis_ranking") or {}).get("selected_hypotheses") or [],
    )


def _direct_parent_verifier_ready_for_proof(blackboard: dict[str, Any], parent_verifier: dict[str, Any]) -> bool:
    summary = dict((blackboard.get("current_belief") or {}).get("parent_route_verifier") or {})
    verifier_ready = is_accepted_route_verifier_report(parent_verifier)
    try:
        summary_route_count = int(summary.get("accepted_route_count") or 0)
        summary_step_count = int(summary.get("best_route_step_count") or 0)
    except (TypeError, ValueError):
        summary_route_count = 0
        summary_step_count = 0
    summary_ready = bool(
        summary.get("schema_version") == "agent_parent_route_verifier_summary.v1"
        and summary.get("verifier_schema_version") == "harness_route_verifier_report.v1"
        and summary.get("accepted") is True
        and summary.get("solved") is True
        and str(summary.get("route_status") or "") == "solved"
        and summary.get("target_match") is True
        and summary_route_count > 0
        and summary.get("best_route_rank") is not None
        and summary_step_count > 0
        and isinstance(summary.get("reasons"), list)
        and not summary["reasons"]
        and isinstance(summary.get("warnings"), list)
    )
    return bool(verifier_ready and (summary_ready or not summary))


def _latest_parent_verifier(artifacts: dict[str, Any]) -> dict[str, Any]:
    guided = artifacts.get("guided_chemenzy")
    if isinstance(guided, dict):
        verifier = guided.get("raw_route_verifier")
        if isinstance(verifier, dict) and verifier:
            return dict(verifier)
    route_verifier = artifacts.get("route_verifier")
    if isinstance(route_verifier, dict):
        return dict(route_verifier)
    return {}


def _portfolio_verifier_bundle(
    *,
    artifacts: dict[str, Any],
    parent_proof: dict[str, Any],
    solved_parent_verifier: dict[str, Any],
) -> dict[str, Any]:
    """Collect host-produced parent and child verifier reports for replay.

    The bundle itself grants no authority. ``derive_portfolio_bindings``
    validates its digest, then independently replays every report/proof-bank
    entry against that report's own target and stock context. Keeping the
    collection paths explicit avoids treating an arbitrary nested model object
    as a verifier report while still preserving verified child-route segments.
    """

    reports: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if not isinstance(value, dict):
            return
        row = dict(value)
        if row.get("schema_version") == "harness_route_verifier_report.v1":
            reports.append(row)

    add(solved_parent_verifier)
    proof_evidence = dict(parent_proof.get("proof_evidence") or {})
    proof_attempt = dict(parent_proof.get("proof_attempt") or {})
    add(proof_evidence.get("parent_verifier"))
    add(proof_evidence.get("parent_verifier_attempt"))
    add(proof_attempt.get("parent_verifier"))
    add(proof_attempt.get("parent_verifier_attempt"))
    add(_latest_parent_verifier(artifacts))

    expansion = artifacts.get("route_expansion_subgoal_search")
    if isinstance(expansion, dict):
        for raw_subgoal in expansion.get("subgoals") or []:
            if isinstance(raw_subgoal, dict):
                add(raw_subgoal.get("verifier"))

    route_verifier = artifacts.get("route_verifier")
    add(route_verifier)
    if isinstance(route_verifier, dict):
        add(route_verifier.get("payload"))
        add(route_verifier.get("result"))

    return build_route_verifier_bundle(reports)


def _codex_first_literature_scout(*, blackboard: dict[str, Any], state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    """Scout sources through Codex first, then local PDF, then placeholders.

    Python keeps this action bounded and auditable. The open-ended literature
    search belongs to Codex; local PDFs only rescue cases where live source
    access fails or is disabled.
    """
    local_source_count = len(
        [row for row in _target_literature_sources(state.target_input, payload) if not _source_is_auto_local_cache(row)]
    )
    max_sources = max(local_source_count, max(1, min(3, int(payload.get("max_sources") or 3))))
    attempts: list[dict[str, Any]] = []
    reasons: list[str] = []

    if _codex_scout_enabled(payload):
        codex_report = _run_codex_literature_scout(
            blackboard=blackboard,
            state=state,
            payload=payload,
            max_sources=max_sources,
        )
        attempts.append(dict(codex_report.get("attempt_summary") or {}))
        if _real_source_candidates(codex_report):
            local_report = _local_pdf_cache_match_report(
                codex_report=codex_report,
                state=state,
                payload=payload,
                max_sources=max_sources,
            )
            local_attempt = dict(local_report.get("attempt_summary") or {})
            if local_attempt.get("attempted"):
                attempts.append(local_attempt)
            codex_report = _merge_local_pdf_scout_report(
                codex_report,
                local_report,
                max_sources=max_sources,
            )
            proxy_summary = _queue_local_pdf_proxy_requests_for_metadata_only_sources(
                scout_report=codex_report,
                state=state,
                max_sources=max_sources,
            )
            proxy_attempt = dict(proxy_summary.get("attempt_summary") or {})
            if proxy_attempt.get("attempted"):
                attempts.append(proxy_attempt)
            if proxy_summary.get("request_count"):
                codex_report["local_pdf_proxy_request_summary"] = proxy_summary
                codex_report["local_pdf_proxy_requests"] = list(proxy_summary.get("requests") or [])
            codex_report["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
            codex_report["scout_attempts"] = attempts
            codex_report["codex_worker_run_attempted"] = True
            codex_report["codex_research_runs"] = int(state.codex_research_runs)
            return codex_report
        reasons.extend(str(item) for item in codex_report.get("reasons") or ["codex_online_scout_no_real_sources"])
    else:
        attempts.append({"mode": "codex_online", "attempted": False, "reason": "codex_online_scout_disabled"})
        reasons.append("codex_online_scout_disabled")

    hint_report = _planner_source_hint_report(
        blackboard=blackboard,
        state=state,
        payload=payload,
        max_sources=max_sources,
    )
    hint_attempt = dict(hint_report.get("attempt_summary") or {})
    if hint_attempt.get("attempted"):
        attempts.append(hint_attempt)
    if _real_source_candidates(hint_report):
        local_report = _local_pdf_cache_match_report(
            codex_report=hint_report,
            state=state,
            payload=payload,
            max_sources=max_sources,
        )
        local_attempt = dict(local_report.get("attempt_summary") or {})
        if local_attempt.get("attempted"):
            local_attempt["trigger"] = "planner_source_hints"
            attempts.append(local_attempt)
        if _real_source_candidates(local_report):
            local_report["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
            local_report["scout_attempts"] = attempts
            local_report["codex_worker_run_attempted"] = any(bool(row.get("attempted")) for row in attempts if row.get("mode") == "codex_online")
            local_report["codex_research_runs"] = int(state.codex_research_runs)
            local_report["source_hint_triggered"] = True
            return local_report
        reasons.extend(str(item) for item in local_report.get("reasons") or ["planner_source_hint_cache_no_match"])

    local_report = _local_pdf_literature_scout(
        blackboard=blackboard,
        state=state,
        payload=payload,
        max_sources=max_sources,
        prior_reasons=reasons,
    )
    attempts.append(dict(local_report.get("attempt_summary") or {}))
    if _real_source_candidates(local_report):
        local_report["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        local_report["scout_attempts"] = attempts
        local_report["codex_worker_run_attempted"] = any(bool(row.get("attempted")) for row in attempts if row.get("mode") == "codex_online")
        local_report["codex_research_runs"] = int(state.codex_research_runs)
        return local_report

    reasons.extend(str(item) for item in local_report.get("reasons") or ["local_pdf_scout_no_real_sources"])
    placeholder = _placeholder_literature_scout(
        blackboard=blackboard,
        state=state,
        payload=payload,
        max_sources=max_sources,
        prior_reasons=reasons,
    )
    placeholder["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
    placeholder["scout_attempts"] = attempts + [dict(placeholder.get("attempt_summary") or {})]
    placeholder["codex_worker_run_attempted"] = any(bool(row.get("attempted")) for row in attempts if row.get("mode") == "codex_online")
    placeholder["codex_research_runs"] = int(state.codex_research_runs)
    return placeholder


def _merge_local_pdf_scout_report(codex_report: dict[str, Any], local_report: dict[str, Any], *, max_sources: int) -> dict[str, Any]:
    """Keep agent-discovered metadata as primary, then annotate local cache hits."""
    if not _real_source_candidates(local_report):
        return dict(codex_report)
    merged = dict(codex_report)
    candidates: list[dict[str, Any]] = []
    index: dict[str, int] = {}

    def add_or_merge(row: dict[str, Any]) -> None:
        candidate = dict(row)
        key = _candidate_merge_key(candidate)
        if key and key in index:
            pos = index[key]
            candidates[pos] = _merge_source_candidate(candidates[pos], candidate)
            return
        index[key or f"row:{len(candidates)}"] = len(candidates)
        candidates.append(candidate)

    for row in codex_report.get("source_candidates") or []:
        if isinstance(row, dict):
            add_or_merge(row)
    for row in local_report.get("source_candidates") or []:
        if isinstance(row, dict):
            add_or_merge(row)

    candidates = _merge_metadata_into_concrete_source_documents(candidates)
    merged["source_candidates"] = candidates[:max_sources]
    merged["source_refs"] = _dedupe(
        [
            *[str(item) for item in codex_report.get("source_refs") or []],
            *[str(item) for item in local_report.get("source_refs") or []],
            *[str(row.get("source_ref") or "") for row in candidates],
        ]
    )[:max_sources]
    if any(str(row.get("local_pdf") or "").strip() for row in candidates):
        cache_hit = any(
            str(row.get("source_discovery_mode") or "")
            in {"codex_online+local_pdf_cache", "local_pdf_cache_match"}
            for row in candidates
        )
        merged["source_discovery_mode"] = "codex_online+local_pdf_cache" if cache_hit else "codex_online+local_pdf"
        merged["accepted"] = True
    return merged


def _merge_metadata_into_concrete_source_documents(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach source-level metadata to every concrete document for that source.

    Article PDFs and supplementary-information PDFs often share a DOI.  The
    same rule applies to publisher PII records.  The metadata lead must not
    consume a result slot and thereby hide an independently extractable
    document.
    """
    metadata_by_source: dict[str, list[dict[str, Any]]] = {}
    concrete_sources: set[str] = set()
    for candidate in candidates:
        source_key = _candidate_logical_source_key(candidate)
        if not source_key:
            continue
        if _candidate_has_concrete_document_identity(candidate):
            concrete_sources.add(source_key)
        else:
            metadata_by_source.setdefault(source_key, []).append(candidate)

    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        source_key = _candidate_logical_source_key(candidate)
        is_concrete = _candidate_has_concrete_document_identity(candidate)
        if source_key in concrete_sources and not is_concrete:
            continue
        row = dict(candidate)
        if source_key and is_concrete:
            for metadata in metadata_by_source.get(source_key, []):
                row = _merge_source_candidate(metadata, row)
        merged.append(row)
    return merged


def _candidate_has_concrete_document_identity(row: dict[str, Any]) -> bool:
    return any(
        str(row.get(key) or "").strip()
        for key in ("local_pdf", "source_pdf_path", "pdf_path", "document_id")
    )


def _candidate_logical_source_key(row: dict[str, Any]) -> str:
    doi = _source_doi(row)
    if doi:
        return f"doi:{doi}"
    pii = _source_pii(row)
    return f"pii:{pii.lower()}" if pii else ""


def _queue_local_pdf_proxy_requests_for_metadata_only_sources(
    *,
    scout_report: dict[str, Any],
    state: ToolExecutionState,
    max_sources: int,
) -> dict[str, Any]:
    candidates = [
        dict(row)
        for row in scout_report.get("source_candidates") or []
        if isinstance(row, dict) and _candidate_needs_local_pdf_proxy(row)
    ][: max(0, int(max_sources or 0))]
    if not candidates:
        return {
            "schema_version": "agentic_local_pdf_proxy_request_summary.v1",
            "accepted": True,
            "request_count": 0,
            "requests": [],
            "reasons": ["no_metadata_only_sources_requiring_local_pdf_proxy"],
            "attempt_summary": {
                "mode": "local_pdf_proxy_queue",
                "attempted": False,
                "reason": "no_metadata_only_sources_requiring_local_pdf_proxy",
            },
            "no_solved_claim": True,
        }

    evidence_dir = state.run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    access_record_path = evidence_dir / "literature_sources.json"
    access_records = [_agent_access_record_from_metadata_candidate(row, state=state) for row in candidates]
    access_write = _write_agent_access_records(access_record_path, access_records, case_id=str(state.preflight.get("case_id") or ""))

    requests: list[dict[str, Any]] = []
    rejected: list[str] = []
    for candidate in candidates:
        request_record = dict(candidate)
        doi = _normalize_doi(str(request_record.get("doi") or request_record.get("source_ref") or ""))
        if doi and not str(request_record.get("doi") or "").strip():
            request_record["doi"] = doi
            request_record.setdefault("url", f"https://doi.org/{doi}")
        try:
            requests.append(
                build_pdf_request(
                    {
                        **request_record,
                        "content_scope": "article",
                        "requested_content_scope": "article",
                        "material_type": request_record.get("source_type") or "journal_article",
                    },
                    case_id=str(state.preflight.get("case_id") or ""),
                    source_ref=str(request_record.get("source_ref") or ""),
                    reason="agent_access_failed_pdf_needed",
                    requested_by="agentic_blackboard_controller",
                )
            )
        except ValueError:
            rejected.append(str(candidate.get("source_ref") or candidate.get("doi") or candidate.get("url") or "metadata_source"))

    queue_path = local_pdf_proxy_request_queue_path(state.run_dir)
    queue_write = (
        write_pdf_request_queue(requests, queue_path, append=True, dedupe=True)
        if requests
        else {"accepted": False, "path": str(queue_path), "request_count": 0}
    )
    return {
        "schema_version": "agentic_local_pdf_proxy_request_summary.v1",
        "accepted": bool(queue_write.get("accepted")) and not rejected,
        "request_count": len(requests),
        "requests": requests,
        "queue_path": str(queue_path.resolve()),
        "access_record_path": str(access_record_path.resolve()),
        "access_record_count": int(access_write.get("record_count") or 0),
        "source_refs": _dedupe([str(row.get("source_ref") or "") for row in candidates]),
        "dois": _dedupe([str(row.get("doi") or "") for row in candidates if str(row.get("doi") or "").strip()]),
        "rejected_source_refs": rejected,
        "queue_write": dict(queue_write),
        "access_record_write": dict(access_write),
        "reasons": [] if requests and not rejected else ["no_pdf_proxy_requests_written", *[f"pdf_proxy_request_rejected:{item}" for item in rejected]],
        "attempt_summary": {
            "mode": "local_pdf_proxy_queue",
            "attempted": True,
            "accepted": bool(requests) and bool(queue_write.get("accepted")) and not rejected,
            "request_count": len(requests),
            "queue_path": str(queue_path.resolve()),
            "access_record_path": str(access_record_path.resolve()),
        },
        "no_solved_claim": True,
    }


def _candidate_needs_local_pdf_proxy(row: dict[str, Any]) -> bool:
    if str(row.get("local_pdf") or "").strip():
        return False
    if bool(row.get("placeholder_only")):
        return False
    if not str(row.get("doi") or row.get("url") or row.get("source_ref") or "").strip():
        return False
    status = str(row.get("access_status") or "").strip().lower()
    positive_markers = {
        "local_pdf_available",
        "agent_accessible_full_text",
        "full_text_available",
        "open_full_text",
        "pdf_available",
    }
    if status in positive_markers:
        return False
    if any(marker in status for marker in ("full text available", "open full text", "pdf available", "local pdf available")):
        return False
    if _candidate_has_real_source(row):
        return True
    return status in {
        "",
        "metadata_only",
        "agent_accessible_metadata_only",
        "agent_access_blocked_login_or_paywall",
        "agent_access_unavailable",
    }


def _agent_access_record_from_metadata_candidate(row: dict[str, Any], *, state: ToolExecutionState) -> dict[str, Any]:
    status = str(row.get("access_status") or "metadata_only").strip()
    if status == "metadata_only" or _candidate_needs_local_pdf_proxy(row):
        status = "agent_accessible_metadata_only"
    elif not status.startswith("agent_access"):
        status = "agent_accessible_metadata_only"
    return {
        "schema_version": "agentic_literature_source_access_record.v1",
        "source": "codex_online_scout",
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_ref": str(row.get("source_ref") or ""),
        "doi": str(row.get("doi") or ""),
        "pii": str(row.get("pii") or ""),
        "url": str(row.get("url") or ""),
        "title": str(row.get("title") or ""),
        "content_scope": "article",
        "agent_access_status": status,
        "access_status": status,
        "reason": "agent_discovered_metadata_without_agent_readable_pdf",
        "no_solved_claim": True,
    }


def _write_agent_access_records(path: Path, records: list[dict[str, Any]], *, case_id: str) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    payload = {
        "schema_version": "open_literature_sources.v1",
        "case_id": str(existing.get("case_id") or case_id),
        "source_relation_policy": dict(existing.get("source_relation_policy") or {}),
        "sources": list(existing.get("sources") or []),
        "excluded_sources": list(existing.get("excluded_sources") or []),
        "search_log": list(existing.get("search_log") or []),
    }
    seen = {_agent_access_record_key(row) for row in payload["search_log"] if isinstance(row, dict)}
    added = 0
    for record in records:
        key = _agent_access_record_key(record)
        if key in seen:
            continue
        seen.add(key)
        payload["search_log"].append(record)
        added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return {
        "schema_version": "agentic_literature_source_access_record_write.v1",
        "accepted": True,
        "path": str(path.resolve()),
        "record_count": len(payload["search_log"]),
        "added_count": added,
    }


def _agent_access_record_key(row: dict[str, Any]) -> str:
    doi = _normalize_doi(str(row.get("doi") or ""))
    url = str(row.get("url") or "").strip().lower()
    source_ref = str(row.get("source_ref") or "").strip().lower()
    scope = str(row.get("content_scope") or row.get("requested_content_scope") or "article").strip().lower()
    status = str(row.get("agent_access_status") or row.get("access_status") or "").strip().lower()
    return "|".join([doi, url, source_ref, scope, status])


def _candidate_merge_key(row: dict[str, Any]) -> str:
    logical_document = source_document_identity(row)
    if logical_document:
        return logical_document
    doi = _normalize_doi(str(row.get("doi") or ""))
    if doi:
        return f"doi:{doi}"
    pii = _source_pii(row)
    if pii:
        return f"pii:{pii.lower()}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref.startswith("doi:"):
        return f"doi:{_normalize_doi(source_ref)}"
    if source_ref.startswith("pii:"):
        ref_pii = _source_pii({"source_ref": source_ref})
        if ref_pii:
            return f"pii:{ref_pii.lower()}"
    url = str(row.get("url") or "").strip().lower()
    return url or source_ref


def _merge_source_candidate(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key in (
        "doi",
        "pii",
        "url",
        "local_pdf",
        "source_pdf_path",
        "pdf_path",
        "document_id",
        "content_scope",
        "source_ref",
        "title",
        "source_type",
        "access_status",
        "route_sequence_hint",
    ):
        value = incoming.get(key)
        if str(value or "").strip() and (key == "local_pdf" or not str(merged.get(key) or "").strip()):
            merged[key] = value
    for key in ("local_pdf_index", "local_pdf_match"):
        value = incoming.get(key)
        if isinstance(value, dict) and value and not isinstance(merged.get(key), dict):
            merged[key] = dict(value)
    incoming_profile = incoming.get("visual_extraction_profile")
    if isinstance(incoming_profile, dict) and incoming_profile:
        merged["visual_extraction_profile"] = {
            **dict(merged.get("visual_extraction_profile") or {}),
            **incoming_profile,
        }
    for key in ("expected_scheme_or_compound_labels", "extraction_task_recommendations"):
        merged[key] = _dedupe(
            [
                *[str(item) for item in merged.get(key) or [] if str(item or "").strip()],
                *[str(item) for item in incoming.get(key) or [] if str(item or "").strip()],
            ]
        )
    rationale_parts = _dedupe(
        [
            str(merged.get("relevance_rationale") or ""),
            str(incoming.get("relevance_rationale") or ""),
        ]
    )
    if rationale_parts:
        merged["relevance_rationale"] = " | ".join(rationale_parts)
    if str(incoming.get("local_pdf") or "").strip():
        incoming_mode = str(incoming.get("source_discovery_mode") or "")
        if incoming_mode == "local_pdf_cache_match":
            merged["source_type"] = "literature_metadata+local_pdf"
            merged["source_discovery_mode"] = "codex_online+local_pdf_cache"
            merged["local_pdf_match"] = dict(incoming.get("local_pdf_match") or {})
        else:
            merged["source_type"] = "local_pdf"
            merged["source_discovery_mode"] = "codex_online+local_pdf"
        merged["access_status"] = "local_pdf_available"
    return merged


def _codex_scout_enabled(payload: dict[str, Any]) -> bool:
    raw = payload.get("codex_scout_enabled")
    if raw is None:
        raw = payload.get("use_codex_worker")
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    return bool(raw)


def _run_codex_literature_scout(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
) -> dict[str, Any]:
    mock = state.mock_tool_results.get("codex_literature_scout")
    if mock is not None:
        value = mock(state, payload, blackboard) if callable(mock) else mock
        report = _normalize_literature_scout_report(
            dict(value or {}),
            blackboard=blackboard,
            state=state,
            max_sources=max_sources,
            discovery_mode="codex_online",
        )
        report["attempt_summary"] = {"mode": "codex_online", "attempted": True, "backend": "mock"}
        return report

    max_codex_runs = int(state.budget.max_codex_research_runs)
    if state.codex_research_runs >= max_codex_runs:
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": str(state.preflight.get("case_id") or ""),
            "source_candidates": [],
            "source_refs": [],
            "reasons": ["codex_research_budget_exhausted"],
            "no_solved_claim": True,
            "source_discovery_mode": "codex_online",
            "attempt_summary": {"mode": "codex_online", "attempted": False, "reason": "codex_research_budget_exhausted"},
        }

    state.codex_research_runs += 1
    task = _codex_literature_scout_task(
        blackboard=blackboard,
        state=state,
        payload=payload,
        max_sources=max_sources,
    )
    record = run_codex_worker(
        task,
        use_codex_cli=_payload_bool(payload, "use_codex_cli", default=True),
        use_api_json=_payload_bool(payload, "use_api_json", default=False),
    )
    record_payload = record.to_dict()
    write_json(state.run_dir / "codex_literature_scout_run_record.json", record_payload)
    state.artifacts["codex_literature_scout_run_record"] = record_payload

    artifact = dict(record.output_artifact or {})
    scout_payload = dict(artifact.get("payload") or {})
    report = _normalize_literature_scout_report(
        scout_payload,
        blackboard=blackboard,
        state=state,
        max_sources=max_sources,
        discovery_mode="codex_online",
    )
    reasons = [str(item) for item in report.get("reasons") or []]
    if record.status != "accepted_draft":
        reasons.extend(str(item) for item in (record.output_validation or {}).get("reasons") or [record.status])
    if not _real_source_candidates(report) and not reasons:
        reasons.append("codex_online_scout_no_real_sources")
    report["accepted"] = bool(_real_source_candidates(report))
    report["reasons"] = sorted(set(reasons))
    report["codex_worker_run_attempted"] = True
    report["codex_worker_status"] = str(record.status or "")
    report["codex_worker_backend"] = str(record.backend or "")
    report["codex_research_runs"] = int(state.codex_research_runs)
    report["attempt_summary"] = {
        "mode": "codex_online",
        "attempted": True,
        "backend": str(record.backend or ""),
        "status": str(record.status or ""),
        "accepted": bool(record.status == "accepted_draft" and _real_source_candidates(report)),
    }
    return report


def _codex_literature_scout_task(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
) -> WorkerTask:
    target = dict(blackboard.get("target_profile") or {})
    bridge_tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    query_terms = _literature_scout_queries(blackboard=blackboard, state=state, payload=payload)
    planner_hints = _planner_source_hints(blackboard=blackboard, payload=payload)
    objective = (
        "Use Codex native web search to find real literature or source-material leads for this retrosynthesis target. "
        "Return only traceable source metadata, not routes. Prefer exact target, target-proximal intermediates, "
        "close steroid/polycyclic analogues, DOI pages, publisher pages, PDFs, or supporting information pages. "
        f"Return up to {max_sources} candidates. If no real source is found, return accepted=false and an empty source_candidates list. "
        f"Target profile: {json.dumps(target, ensure_ascii=False, sort_keys=True)}. "
        f"Bridge tasks: {json.dumps(bridge_tasks[:6], ensure_ascii=False, sort_keys=True)}. "
        f"Planner source hints to confirm, not evidence by themselves: {json.dumps(planner_hints[:8], ensure_ascii=False, sort_keys=True)}. "
        f"Suggested queries: {json.dumps(query_terms, ensure_ascii=False)}."
    )
    return WorkerTask(
        task_id=f"{str(state.preflight.get('case_id') or 'case')}:literature_scout:{int((blackboard.get('budget_state') or {}).get('scout_calls') or 0) + 1}",
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        task_type="target_research",
        required_artifact_type="LiteratureScoutReport",
        model=str(state.model or ""),
        input_refs=[str(state.run_dir / "agent_blackboard.json")],
        allowed_tools=["web_search", "browser", "local_search"],
        budget=WorkerBudget(
            timeout_s=_codex_scout_timeout_s(state, payload),
            max_output_bytes=int(payload.get("max_output_bytes") or 120_000),
            max_tool_calls=int(payload.get("max_tool_calls") or 12),
            max_worker_runs=1,
            reasoning_effort=_codex_scout_reasoning_effort(payload),
        ),
        objective=objective,
        allowed_workdir=str(state.run_dir),
    )


def _planner_source_hint_report(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
) -> dict[str, Any]:
    hints = _planner_source_hints(blackboard=blackboard, payload=payload)
    candidates = [
        _normalize_source_candidate(row, idx=idx, discovery_mode="codex_action_planner_hint")
        for idx, row in enumerate(hints, start=1)
    ]
    candidates = [
        {**row, "source_type": str(row.get("source_type") or "planner_source_hint"), "access_status": "metadata_hint"}
        for row in _dedupe_candidates(candidates)
        if _candidate_has_real_source(row)
    ][:max_sources]
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": bool(candidates),
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": candidates,
        "source_refs": [str(row.get("source_ref") or "") for row in candidates],
        "search_queries": _literature_scout_queries(blackboard=blackboard, state=state, payload=payload),
        "reasons": [] if candidates else (["planner_source_hints_not_available"] if not hints else ["planner_source_hints_without_real_source_metadata"]),
        "limitations": ["planner_source_hints_require_scout_or_local_pdf_confirmation"] if hints else [],
        "no_solved_claim": True,
        "source_discovery_mode": "codex_action_planner_hint",
        "attempt_summary": {
            "mode": "planner_source_hints",
            "attempted": bool(hints),
            "accepted": bool(candidates),
            "hint_count": len(hints),
            "candidate_count": len(candidates),
        },
    }


def _planner_source_hints(*, blackboard: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for row in payload.get("planner_source_hints") or []:
        if isinstance(row, dict):
            raw_rows.append(dict(row))
    for row in (blackboard.get("literature_evidence") or {}).get("planner_source_hints") or []:
        if isinstance(row, dict):
            raw_rows.append(dict(row))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_rows:
        doi = _normalize_doi(str(row.get("doi") or ""))
        pii = str(row.get("pii") or _source_pii(row)).strip()
        url = str(row.get("url") or "").strip()
        local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or "").strip()
        title = str(row.get("title") or row.get("source_title") or "").strip()
        source_ref = str(row.get("source_ref") or "").strip()
        if not source_ref:
            source_ref = f"doi:{doi}" if doi else (f"pii:{pii}" if pii else (url or (f"local_pdf:{Path(local_pdf).name}" if local_pdf else "")))
        key = str(doi or pii or url or local_pdf or source_ref or title).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "schema_version": "planner_source_hint.v1",
                "hint_id": str(row.get("hint_id") or f"planner_source_hint_{len(out) + 1}"),
                "source_ref": source_ref,
                "title": title,
                "doi": doi,
                "pii": pii,
                "url": url,
                "local_pdf": local_pdf,
                "local_ref": str(row.get("local_ref") or ""),
                "source_type": str(row.get("source_type") or "planner_source_hint"),
                "relevance_rationale": str(row.get("relevance_rationale") or ""),
                "expected_scheme_or_compound_labels": [
                    str(item)
                    for item in row.get("expected_scheme_or_compound_labels") or row.get("expected_labels") or []
                    if str(item or "").strip()
                ],
                "extraction_task_recommendations": [
                    str(item)
                    for item in row.get("extraction_task_recommendations") or []
                    if str(item or "").strip()
                ],
                "evidence_class": "planner_source_hint",
                "allowed_use": "source_acquisition_hint_only",
                "no_solved_claim": True,
            }
        )
        if len(out) >= 8:
            break
    return out


def _codex_scout_timeout_s(state: ToolExecutionState, payload: dict[str, Any]) -> float:
    floor = _positive_float(os.environ.get("AUTOPLANNER_CODEX_SCOUT_TIMEOUT_MIN_S")) or 180.0
    for value in (
        payload.get("codex_timeout_s"),
        os.environ.get("AUTOPLANNER_CODEX_SCOUT_TIMEOUT_S"),
        payload.get("timeout_s"),
    ):
        explicit = _positive_float(value)
        if explicit is not None:
            return max(floor, explicit)
    budget_timeout = (
        _positive_float(getattr(state.budget, "open_research_timeout_s", None))
        or _positive_float(getattr(state.budget, "timeout_s", None))
        or floor
    )
    return max(floor, budget_timeout)


def _codex_scout_reasoning_effort(payload: dict[str, Any]) -> str:
    explicit = str(payload.get("reasoning_effort") or payload.get("codex_reasoning_effort") or "").strip()
    if explicit:
        return explicit
    return str(os.environ.get("AUTOPLANNER_CODEX_SCOUT_REASONING_EFFORT") or "high").strip() or "high"


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _local_pdf_cache_match_report(
    *,
    codex_report: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
) -> dict[str, Any]:
    cache_sources = _target_literature_cache_sources(state.target_input, payload)
    direct_sources = _direct_local_pdf_sources(state.target_input, payload)
    local_sources = [*cache_sources, *direct_sources]
    if not local_sources:
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": str(state.preflight.get("case_id") or ""),
            "source_candidates": [],
            "source_refs": [],
            "reasons": ["local_pdf_cache_not_provided"],
            "no_solved_claim": True,
            "source_discovery_mode": "local_pdf_cache_match",
            "attempt_summary": {"mode": "local_pdf_cache", "attempted": False, "reason": "local_pdf_cache_not_provided"},
        }

    target = str(state.target_input.get("target_name") or "target")
    candidates: list[dict[str, Any]] = []
    missing: list[str] = []
    unmatched: list[str] = []
    for idx, discovered in enumerate(_real_source_candidates(codex_report), start=1):
        match_source: dict[str, Any] = {}
        match_basis = ""
        for source in local_sources:
            basis = _local_pdf_cache_match_basis(discovered, source)
            if basis:
                match_source = dict(source)
                match_basis = basis
                break
        if not match_source:
            unmatched.append(str(discovered.get("source_ref") or discovered.get("doi") or discovered.get("title") or f"source:{idx}"))
            continue
        pdf_path = str(match_source.get("local_pdf") or match_source.get("pdf_path") or "").strip()
        if not pdf_path:
            continue
        resolved = Path(pdf_path).expanduser()
        if not resolved.is_file():
            missing.append(str(resolved))
            continue
        matched = dict(match_source)
        if str(discovered.get("source_ref") or "").strip():
            matched["source_ref"] = str(discovered.get("source_ref") or "").strip()
        if str(discovered.get("doi") or "").strip():
            matched["doi"] = str(discovered.get("doi") or "").strip()
        if str(discovered.get("url") or "").strip():
            matched["url"] = str(discovered.get("url") or "").strip()
        if str(discovered.get("title") or "").strip():
            matched["title"] = str(discovered.get("title") or "").strip()
        matched["local_pdf"] = str(resolved.resolve())
        matched["local_pdf_match"] = {
            "schema_version": "local_pdf_cache_match.v1",
            "match_basis": match_basis,
            "agent_discovered_source_ref": str(discovered.get("source_ref") or ""),
            "agent_discovered_doi": str(discovered.get("doi") or ""),
            "agent_discovered_pii": str(discovered.get("pii") or _source_pii(discovered)),
            "agent_discovered_title": str(discovered.get("title") or ""),
            "agent_discovered_url": str(discovered.get("url") or ""),
            "cache_source_ref": str(match_source.get("source_ref") or ""),
            "cache_doi": str(match_source.get("doi") or _source_doi(match_source)),
            "cache_pii": str(match_source.get("pii") or _source_pii(match_source)),
            "cache_title": str(match_source.get("title") or ""),
        }
        is_cache_match = _source_is_local_cache(match_source)
        candidates.append(
            _local_pdf_source_candidate(
                matched,
                idx=idx,
                state=state,
                target=target,
                discovery_mode="local_pdf_cache_match" if is_cache_match else "local_pdf_match",
                relevance_rationale=(
                    "agent-discovered DOI/title matched an available local PDF cache entry"
                    if is_cache_match
                    else "agent-discovered DOI/title matched a direct local PDF fallback entry"
                ),
            )
        )
        if len(candidates) >= max_sources:
            break

    reasons: list[str] = []
    if unmatched:
        reasons.append("local_pdf_cache_no_match_for_some_agent_sources")
    if missing:
        reasons.append("local_pdf_cache_match_missing_file")
    if not candidates:
        reasons.append("local_pdf_cache_no_agent_discovered_match")
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": bool(candidates),
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": candidates[:max_sources],
        "source_refs": [str(candidate.get("source_ref") or "") for candidate in candidates[:max_sources]],
        "reasons": sorted(set(reasons)),
        "no_solved_claim": True,
        "source_discovery_mode": "local_pdf_cache_match",
        "attempt_summary": {
            "mode": "local_pdf_cache",
            "attempted": True,
            "accepted": bool(candidates),
            "matched_source_count": len(candidates),
            "cache_source_count": len(cache_sources),
            "auto_cache_source_count": sum(1 for source in cache_sources if _source_is_auto_local_cache(source)),
            "direct_source_count": len(direct_sources),
            "missing_paths": missing,
            "unmatched_agent_sources": unmatched[:8],
        },
    }


def _local_pdf_literature_scout(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
    prior_reasons: list[str],
) -> dict[str, Any]:
    reasons = [str(item) for item in prior_reasons]
    direct_sources = _direct_local_pdf_sources(state.target_input, payload)
    cache_sources = [
        row
        for row in _target_literature_cache_sources(state.target_input, payload)
        if not _source_is_auto_local_cache(row)
    ]
    auto_cache_sources = [
        row
        for row in _target_literature_cache_sources(state.target_input, payload)
        if _source_is_auto_local_cache(row)
    ]
    cache_fallback = False
    auto_cache_fallback = False
    sources = direct_sources
    if not sources and cache_sources:
        sources = cache_sources
        cache_fallback = True
    if not sources and auto_cache_sources and _auto_local_pdf_blind_fallback_allowed(payload=payload, prior_reasons=reasons):
        sources = _rank_auto_local_pdf_fallback_sources(
            auto_cache_sources,
            blackboard=blackboard,
            state=state,
            payload=payload,
            max_sources=max_sources,
        )
        auto_cache_fallback = bool(sources)
    if not sources:
        reasons.append("local_pdf_not_provided")
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": str(state.preflight.get("case_id") or ""),
            "source_candidates": [],
            "source_refs": [],
            "reasons": sorted(set(reasons)),
            "no_solved_claim": True,
            "source_discovery_mode": "local_pdf_fallback",
            "attempt_summary": {
                "mode": "local_pdf",
                "attempted": False,
                "reason": "local_pdf_not_provided",
            },
        }
    target = str(state.target_input.get("target_name") or "target")
    candidates: list[dict[str, Any]] = []
    missing: list[str] = []
    discovery_mode = (
        "auto_local_pdf_blind_fallback_after_codex_failure"
        if auto_cache_fallback
        else ("local_pdf_fallback_after_codex_failure" if cache_fallback else "local_pdf_fallback")
    )
    for idx, source in enumerate(sources, start=1):
        pdf_path = str(source.get("local_pdf") or source.get("pdf_path") or "").strip()
        if not pdf_path:
            continue
        resolved = Path(pdf_path).expanduser()
        if not resolved.is_file():
            missing.append(str(resolved))
            continue
        candidates.append(
            _local_pdf_source_candidate(
                {**source, "local_pdf": str(resolved.resolve())},
                idx=idx,
                state=state,
                target=target,
                discovery_mode=discovery_mode,
                relevance_rationale=(
                    "local PDF filename/metadata fallback after Codex online source access failed; requires downstream visual/exact validation"
                    if auto_cache_fallback
                    else (
                    "local PDF cache fallback after online source access failed or timed out"
                    if cache_fallback
                    else "direct local PDF fallback after online source access failed"
                    )
                ),
            )
        )
    if missing:
        reasons.append("local_pdf_missing")
    if not candidates:
        return {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": str(state.preflight.get("case_id") or ""),
            "source_candidates": [],
            "source_refs": [],
            "reasons": sorted(set(reasons)),
            "no_solved_claim": True,
            "source_discovery_mode": discovery_mode,
            "attempt_summary": {
                "mode": "local_pdf_cache" if cache_fallback else "local_pdf",
                "attempted": True,
                "reason": "local_pdf_missing" if missing else "local_pdf_not_provided",
                "missing_paths": missing,
            },
        }
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": True,
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": candidates[:max_sources],
        "source_refs": [str(candidate.get("source_ref") or "") for candidate in candidates[:max_sources]],
        "reasons": [],
        "no_solved_claim": True,
        "source_discovery_mode": discovery_mode,
        "attempt_summary": {
            "mode": "auto_local_pdf_cache" if auto_cache_fallback else ("local_pdf_cache" if cache_fallback else "local_pdf"),
            "attempted": True,
            "accepted": True,
            "source_count": len(candidates),
            "auto_cache_blind_fallback": bool(auto_cache_fallback),
            "missing_paths": missing,
        },
    }


def _auto_local_pdf_blind_fallback_allowed(*, payload: dict[str, Any], prior_reasons: list[str]) -> bool:
    policy = dict(payload.get("source_acquisition_policy") or {})
    if policy.get("local_pdf_fallback_allowed") is False:
        return False
    if str(os.environ.get("AUTOPLANNER_AUTO_LOCAL_PDF_BLIND_FALLBACK", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return False
    text = " ".join(str(item) for item in prior_reasons).lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "timeout",
            "502",
            "bad gateway",
            "worker_exit_code_nonzero",
            "rejected_output",
            "output_not_json_object",
            "codex_online_scout_no_real_sources",
            "no_real_literature_source_found",
        )
    )


def _rank_auto_local_pdf_fallback_sources(
    sources: list[dict[str, Any]],
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        score = _auto_local_pdf_fallback_score(source, blackboard=blackboard, state=state, payload=payload)
        if score <= 0:
            continue
        row = dict(source)
        index = dict(row.get("local_pdf_index") or {})
        index["match_policy"] = "local_filename_keyword_fallback_after_codex_failure"
        index["agent_discovered_metadata_required_for_proof"] = True
        row["local_pdf_index"] = index
        row["local_pdf_blind_fallback"] = True
        row["source_usage_policy"] = "local_pdf_fallback_after_codex_online_failure_requires_visual_exact_validation"
        scored.append((score, str(row.get("title") or row.get("local_pdf") or ""), row))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    return [row for _, _, row in scored[: max(1, int(max_sources or 1))]]


def _auto_local_pdf_fallback_score(
    source: dict[str, Any],
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
) -> int:
    target = dict(blackboard.get("target_profile") or {})
    haystack = " ".join(
        str(source.get(key) or "")
        for key in ("title", "source_ref", "doi", "pii", "url", "local_pdf")
    ).lower()
    query_text = " ".join(
        [
            str(payload.get("search_intent") or ""),
            *[str(item) for item in payload.get("queries") or []],
            *[str(item) for item in payload.get("search_queries") or []],
            str(target.get("family_hint") or state.target_input.get("family_hint") or ""),
            *[str(item) for item in target.get("functional_handles") or []],
        ]
    ).lower()
    score = 0
    query_terms = _priority_terms([query_text])
    score += 6 * sum(1 for token in query_terms if token in haystack)
    for token in (
        "total synthesis",
        "semisynthesis",
        "synthesis",
        "preparation",
        "process",
        "route",
        "scheme",
        "intermediate",
        "supporting information",
        "supplementary information",
    ):
        if token in haystack:
            score += 1
    if str(source.get("doi") or "").strip() or str(source.get("pii") or "").strip():
        score += 1
    return score


def _local_pdf_source_candidate(
    source: dict[str, Any],
    *,
    idx: int,
    state: ToolExecutionState,
    target: str,
    discovery_mode: str,
    relevance_rationale: str,
) -> dict[str, Any]:
    pdf_path = str(source.get("local_pdf") or source.get("pdf_path") or "").strip()
    source_ref = str(source.get("source_ref") or source.get("ref") or "").strip()
    pii = str(source.get("pii") or _source_pii(source)).strip()
    source_role = str(source.get("source_role") or source.get("source_usage") or "").strip()
    is_user_seed = _candidate_is_user_provided_local_pdf_seed(source)
    source_hints = _local_pdf_source_hints(
        target=target,
        source_ref=source_ref,
        family_hint=str(state.target_input.get("family_hint") or ""),
    )
    candidate = {
        "schema_version": "literature_source_candidate.v1",
        "candidate_id": str(source.get("candidate_id") or f"{discovery_mode}_{idx}"),
        "source_ref": source_ref or f"local_pdf:{Path(pdf_path).name}",
        "title": str(source.get("title") or source_hints.get("title") or f"{target} local PDF source"),
        "doi": str(source.get("doi") or _source_doi(source) or _doi_from_source_ref(source_ref)),
        "pii": pii,
        "url": str(source.get("url") or (_sciencedirect_url_from_pii(pii) if pii else "")),
        "local_pdf": pdf_path,
        "document_id": str(source.get("document_id") or ""),
        "content_scope": source_content_scope(source),
        "source_type": "user_provided_local_pdf_seed" if is_user_seed else "local_pdf",
        "source_role": source_role,
        "user_provided_source_seed": bool(source.get("user_provided_source_seed")),
        "source_usage_policy": str(source.get("source_usage_policy") or ""),
        "source_discovery_mode": discovery_mode,
        "access_status": "local_pdf_available",
        "relevance_rationale": str(source.get("relevance_rationale") or relevance_rationale),
        "expected_scheme_or_compound_labels": list(
            source.get("expected_scheme_or_compound_labels")
            or source.get("expected_labels")
            or source_hints.get("expected_labels")
            or []
        ),
        "extraction_task_recommendations": [
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "compile_exact_literature_rows",
        ],
        "route_sequence_hint": str(source.get("route_sequence_hint") or source_hints.get("route_sequence_hint") or ""),
        "visual_extraction_profile": (
            dict(source.get("visual_extraction_profile") or {})
            if isinstance(source.get("visual_extraction_profile"), dict)
            else {}
        ),
        "no_solved_claim": True,
    }
    if isinstance(source.get("local_pdf_match"), dict):
        candidate["local_pdf_match"] = dict(source.get("local_pdf_match") or {})
    if isinstance(source.get("local_pdf_index"), dict):
        candidate["local_pdf_index"] = dict(source.get("local_pdf_index") or {})
    return candidate


def _placeholder_literature_scout(
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    payload: dict[str, Any],
    max_sources: int,
    prior_reasons: list[str],
) -> dict[str, Any]:
    target = str(state.target_input.get("target_name") or "target")
    tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    candidates: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks[:max_sources], start=1):
        handle = str(task.get("target_handle") or task.get("task_type") or "bridge")
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": f"placeholder_query_{idx}",
                "source_ref": f"query:{target}:{handle}",
                "title": f"{target} synthesis {handle} bridge",
                "doi": "",
                "url": "",
                "local_pdf": "",
                "source_type": "placeholder_query",
                "source_discovery_mode": "placeholder",
                "access_status": "placeholder_only",
                "placeholder_only": True,
                "relevance_rationale": str(task.get("required_bridge") or "target-proximal bridge required"),
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": ["retry_codex_online_scout", "request_local_pdf_or_source_url"],
                "no_solved_claim": True,
            }
        )
    if not candidates:
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "placeholder_target_synthesis_query",
                "source_ref": f"query:{target}:target_proximal_synthesis",
                "title": f"{target} synthesis target-proximal intermediate",
                "doi": "",
                "url": "",
                "local_pdf": "",
                "source_type": "placeholder_query",
                "source_discovery_mode": "placeholder",
                "access_status": "placeholder_only",
                "placeholder_only": True,
                "relevance_rationale": "online and local PDF scout failed; placeholder records the missing target-proximal source need",
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": ["retry_codex_online_scout", "request_local_pdf_or_source_url"],
                "no_solved_claim": True,
            }
        )
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": False,
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": candidates[:max_sources],
        "source_refs": [str(row.get("source_ref") or "") for row in candidates[:max_sources]],
        "reasons": sorted(set([*prior_reasons, "no_real_literature_source_found", "placeholder_source_candidates_only"])),
        "no_solved_claim": True,
        "source_discovery_mode": "placeholder",
        "placeholder_only": True,
        "attempt_summary": {"mode": "placeholder", "attempted": True, "accepted": False},
    }


def _normalize_literature_scout_report(
    report: dict[str, Any],
    *,
    blackboard: dict[str, Any],
    state: ToolExecutionState,
    max_sources: int,
    discovery_mode: str,
) -> dict[str, Any]:
    payload = dict(report.get("payload") or report)
    raw_candidates = payload.get("source_candidates") or []
    if not raw_candidates and payload.get("source_title"):
        raw_candidates = [payload]
    candidates = [
        _normalize_source_candidate(row, idx=idx, discovery_mode=discovery_mode)
        for idx, row in enumerate(raw_candidates, start=1)
        if isinstance(row, dict)
    ]
    candidates = [row for row in candidates if row.get("source_ref") or row.get("title")]
    candidates = _dedupe_candidates(candidates)[:max_sources]
    refs = [str(row.get("source_ref") or "") for row in candidates if str(row.get("source_ref") or "").strip()]
    if not refs:
        refs = [str(item) for item in payload.get("source_refs") or [] if str(item or "").strip()]
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": bool(candidates and any(_candidate_has_real_source(row) for row in candidates)),
        "case_id": str(payload.get("case_id") or state.preflight.get("case_id") or ""),
        "source_candidates": candidates,
        "source_refs": _dedupe(refs),
        "search_queries": [str(item) for item in payload.get("search_queries") or _literature_scout_queries(blackboard=blackboard, state=state, payload={})],
        "reasons": [str(item) for item in payload.get("reasons") or []],
        "limitations": [str(item) for item in payload.get("limitations") or []],
        "no_solved_claim": True,
        "source_discovery_mode": discovery_mode,
    }


def _normalize_source_candidate(row: dict[str, Any], *, idx: int, discovery_mode: str) -> dict[str, Any]:
    doi = _normalize_doi(str(row.get("doi") or ""))
    url = str(row.get("url") or "").strip()
    pii = str(row.get("pii") or _source_pii(row)).strip()
    local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or "").strip()
    title = str(row.get("title") or row.get("source_title") or "").strip()
    source_ref = str(row.get("source_ref") or "").strip()
    if not source_ref:
        source_ref = (
            f"doi:{doi}"
            if doi
            else (f"pii:{pii}" if pii else (url or (f"local_pdf:{Path(local_pdf).name}" if local_pdf else f"source:{idx}")))
        )
    tasks = [str(item) for item in row.get("extraction_task_recommendations") or [] if str(item or "").strip()]
    if not tasks:
        tasks = ["extract_pdf_literature_structures", "extract_visual_literature_chain", "compile_exact_literature_rows"] if local_pdf else [
            "resolve_source_material_or_provide_pdf",
            "codex_source_detail_followup",
        ]
    return {
        "schema_version": "literature_source_candidate.v1",
        "candidate_id": str(row.get("candidate_id") or f"{discovery_mode}_{idx}"),
        "source_ref": source_ref,
        "title": title,
        "doi": doi,
        "pii": pii,
        "url": url or (f"https://doi.org/{doi}" if doi else (_sciencedirect_url_from_pii(pii) if pii else "")),
        "local_pdf": local_pdf,
        "document_id": str(row.get("document_id") or ""),
        "content_scope": source_content_scope(row),
        "source_type": str(row.get("source_type") or ("local_pdf" if local_pdf else "literature_metadata")),
        "source_discovery_mode": str(row.get("source_discovery_mode") or discovery_mode),
        "access_status": str(row.get("access_status") or ("local_pdf_available" if local_pdf else "metadata_only")),
        "relevance_rationale": str(row.get("relevance_rationale") or row.get("route_role_detail") or ""),
        "expected_scheme_or_compound_labels": [
            str(item)
            for item in row.get("expected_scheme_or_compound_labels") or row.get("expected_labels") or []
            if str(item or "").strip()
        ],
        "extraction_task_recommendations": tasks,
        "route_sequence_hint": str(row.get("route_sequence_hint") or ""),
        "visual_extraction_profile": (
            dict(row.get("visual_extraction_profile") or {})
            if isinstance(row.get("visual_extraction_profile"), dict)
            else {}
        ),
        "no_solved_claim": True,
    }


def _real_source_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in report.get("source_candidates") or []
        if isinstance(row, dict) and _candidate_has_real_source(row)
    ]


def _candidate_has_real_source(row: dict[str, Any]) -> bool:
    if bool(row.get("placeholder_only")):
        return False
    if str(row.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    return bool(str(row.get("doi") or row.get("pii") or row.get("url") or row.get("local_pdf") or "").strip())


def _literature_scout_queries(*, blackboard: dict[str, Any], state: ToolExecutionState, payload: dict[str, Any]) -> list[str]:
    explicit = [str(item) for item in payload.get("queries") or payload.get("search_queries") or [] if str(item or "").strip()]
    if explicit:
        return _dedupe(explicit)[:6]
    target = str(state.target_input.get("target_name") or (blackboard.get("target_profile") or {}).get("target_name") or "target")
    family = str(state.target_input.get("family_hint") or (blackboard.get("target_profile") or {}).get("family_hint") or "")
    handles = [
        str(row.get("target_handle") or row.get("task_type") or "")
        for row in blackboard.get("bridge_tasks") or []
        if isinstance(row, dict)
    ]
    base = [part for part in [target, family] if part]
    queries = [
        " ".join([*base, "synthesis", "total synthesis", "semisynthesis"]),
        " ".join([*base, "target proximal intermediate", "synthetic route"]),
    ]
    for handle in handles[:4]:
        if handle:
            queries.append(" ".join([*base, handle, "synthesis bridge"]))
    for hint in _planner_source_hints(blackboard=blackboard, payload=payload)[:4]:
        doi = str(hint.get("doi") or "").strip()
        pii = str(hint.get("pii") or "").strip()
        title = str(hint.get("title") or "").strip()
        source_ref = str(hint.get("source_ref") or "").strip()
        if doi:
            queries.append(doi)
        if pii:
            queries.append(pii)
        if title:
            queries.append(" ".join([title, "synthesis scheme"]))
        elif source_ref:
            queries.append(" ".join([source_ref, target, "synthesis"]))
    return _dedupe([q for q in queries if q.strip()])[:8]


def _payload_bool(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    raw = payload.get(key)
    if raw is None:
        return bool(default)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    return bool(raw)


def _normalize_doi(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.replace("\\", "").strip()
    for separator in ("&", "?", "#"):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text.strip().strip(".,;:)]}'\"").lower()


def _is_specific_doi(value: str) -> bool:
    text = _normalize_doi(value)
    return bool(re.match(r"^10\.\d{4,9}/\S{2,}$", text, flags=re.IGNORECASE))


def _doi_from_source_ref(value: str) -> str:
    text = str(value or "").strip()
    doi = _normalize_doi(text) if text.lower().startswith(("doi:", "10.")) else ""
    return doi if _is_specific_doi(doi) else ""


def _source_doi(row: dict[str, Any]) -> str:
    for key in ("doi", "source_ref", "url"):
        doi = _normalize_doi(str(row.get(key) or ""))
        if _is_specific_doi(doi):
            return doi
    return ""


def _source_pii(row: dict[str, Any]) -> str:
    for key in ("pii", "source_ref", "url", "local_pdf", "pdf_path", "title"):
        value = str(row.get(key) or "")
        pii = _extract_pii(value)
        if pii:
            return pii
    return ""


def _extract_pii(value: str) -> str:
    text = str(value or "")
    match = re.search(r"S\d{12,24}", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else ""


def _sciencedirect_url_from_pii(pii: str) -> str:
    value = str(pii or "").strip().upper()
    return f"https://www.sciencedirect.com/science/article/pii/{value}" if value else ""


def _local_pdf_cache_match_basis(discovered: dict[str, Any], cache_source: dict[str, Any]) -> str:
    discovered_doi = _source_doi(discovered)
    cache_doi = _source_doi(cache_source)
    if discovered_doi and cache_doi and discovered_doi == cache_doi:
        return "doi"
    if discovered_doi:
        return ""

    discovered_pii = _source_pii(discovered)
    cache_pii = _source_pii(cache_source)
    if discovered_pii and cache_pii and discovered_pii == cache_pii:
        return "pii"

    discovered_ref = str(discovered.get("source_ref") or "").strip().lower()
    cache_ref = str(cache_source.get("source_ref") or "").strip().lower()
    if discovered_ref and cache_ref and discovered_ref == cache_ref:
        return "source_ref"

    discovered_url = str(discovered.get("url") or "").strip().lower()
    cache_url = str(cache_source.get("url") or "").strip().lower()
    if discovered_url and cache_url and discovered_url == cache_url:
        return "url"

    discovered_title = str(discovered.get("title") or "").strip()
    cache_title = str(cache_source.get("title") or Path(str(cache_source.get("local_pdf") or "")).stem).strip()
    if _title_matches(discovered_title, cache_title):
        return "title"
    return ""


def _title_matches(left: str, right: str) -> bool:
    left_norm = _title_key(left)
    right_norm = _title_key(right)
    if not left_norm or not right_norm:
        return False
    if len(left_norm) >= 12 and len(right_norm) >= 12 and (left_norm in right_norm or right_norm in left_norm):
        return True
    left_tokens = set(_title_tokens(left_norm))
    right_tokens = set(_title_tokens(right_norm))
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap >= 3 and overlap / max(1, min(len(left_tokens), len(right_tokens))) >= 0.6


def _title_key(value: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(value or "")).split())


def _title_tokens(value: str) -> list[str]:
    stop = {"a", "an", "and", "of", "the", "to", "for", "in", "on", "with", "by", "total", "synthesis"}
    return [token for token in _title_key(value).split() if len(token) > 2 and token not in stop]


def _dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _candidate_merge_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _deterministic_literature_scout(*, blackboard: dict[str, Any], state: ToolExecutionState, payload: dict[str, Any]) -> dict[str, Any]:
    target = str(state.target_input.get("target_name") or "target")
    tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    max_sources = max(1, min(3, int(payload.get("max_sources") or 3)))
    candidates: list[dict[str, Any]] = []
    sources = _target_literature_sources(state.target_input, payload)
    for idx, source in enumerate(sources, start=1):
        if len(candidates) >= max_sources:
            break
        pdf_path = str(source.get("local_pdf") or source.get("pdf_path") or "")
        source_ref = str(source.get("source_ref") or "")
        source_hints = _local_pdf_source_hints(target=target, source_ref=source_ref, family_hint=str(state.target_input.get("family_hint") or ""))
        if not pdf_path:
            continue
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": str(source.get("candidate_id") or f"local_pdf_{idx}"),
                "source_ref": source_ref or f"local_pdf:{Path(pdf_path).name}",
                "title": str(source.get("title") or source_hints.get("title") or f"{target} local PDF source"),
                "url": "",
                "local_pdf": pdf_path,
                "relevance_rationale": "user-provided local PDF for source-detail extraction",
                "expected_scheme_or_compound_labels": source_hints.get("expected_labels") or [],
                "extraction_task_recommendations": [
                    "extract_pdf_literature_structures",
                    "extract_visual_literature_chain",
                    "compile_exact_literature_rows",
                ],
                "route_sequence_hint": source_hints.get("route_sequence_hint") or "",
            }
        )
    for idx, task in enumerate(tasks[:max_sources], start=1):
        if len(candidates) >= max_sources:
            break
        handle = str(task.get("target_handle") or task.get("task_type") or "bridge")
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": f"query_{idx}",
                "source_ref": f"query:{target}:{handle}",
                "title": f"{target} synthesis {handle} bridge",
                "url": "",
                "local_pdf": "",
                "relevance_rationale": str(task.get("required_bridge") or "target-proximal bridge required"),
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": [
                    "extract_pdf_literature_structures",
                    "extract_visual_literature_chain",
                    "compile_exact_literature_rows",
                ],
            }
        )
    if not candidates:
        candidates.append(
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "target_synthesis_query",
                "source_ref": f"query:{target}:target_proximal_synthesis",
                "title": f"{target} synthesis target-proximal intermediate",
                "url": "",
                "local_pdf": "",
                "relevance_rationale": "initial scout for target-proximal route evidence",
                "expected_scheme_or_compound_labels": [],
                "extraction_task_recommendations": ["extract_visual_literature_chain"],
            }
        )
    return {
        "schema_version": "literature_scout_report.v1",
        "accepted": bool(candidates),
        "case_id": str(state.preflight.get("case_id") or ""),
        "source_candidates": candidates[:max_sources],
        "source_refs": [str(row.get("source_ref") or "") for row in candidates[:max_sources]],
        "reasons": [] if candidates else ["no_source_candidates"],
        "no_solved_claim": True,
    }


def _local_pdf_source_hints(*, target: str, source_ref: str, family_hint: str) -> dict[str, Any]:
    del target, source_ref, family_hint
    return {"title": "", "expected_labels": [], "route_sequence_hint": ""}


def _tool_record_to_action_result(record: Any) -> dict[str, Any]:
    output = dict(record.output or {})
    result = output.get("result") if isinstance(output.get("result"), dict) else output
    return {
        "accepted": record.status not in {"rejected", "error"} or bool(output.get("result")),
        "result": result,
        "reasons": [str(item) for item in getattr(record, "reasons", []) or output.get("reasons") or []],
    }


def _mock_action_result(state: ToolExecutionState, action: dict[str, Any], blackboard: dict[str, Any]) -> Any | None:
    action_type = str(action.get("action_type") or "")
    action_id = str(action.get("action_id") or "")
    value = state.mock_tool_results.get(action_id)
    if value is None:
        value = state.mock_tool_results.get(action_type)
    if value is None:
        return None
    if callable(value):
        return value(state, action, blackboard)
    return value


def _mock_codex_action_planner(
    state: ToolExecutionState,
    blackboard: dict[str, Any],
    round_index: int,
) -> dict[str, Any] | None:
    value = state.mock_tool_results.get("codex_action_planner")
    if value is None:
        value = state.mock_tool_results.get("agent_action_planner")
    if value is None:
        return None
    if callable(value):
        return value(state, blackboard, round_index)
    return dict(value or {})


def _wrap_action_result(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "accepted" in value and ("result" in value or "schema_version" in value):
            return dict(value)
        return {"accepted": bool(value.get("accepted", True)), "result": dict(value), "reasons": [str(item) for item in value.get("reasons") or []]}
    return {"accepted": True, "result": {"value": str(value)}, "reasons": []}


def _inject_pdf_defaults(
    payload: dict[str, Any],
    target_input: dict[str, Any],
    *,
    blackboard: dict[str, Any] | None = None,
) -> None:
    source = _matching_literature_source(target_input, payload)
    if blackboard is not None:
        source = _matching_blackboard_pdf_source(blackboard, payload) or source
    if source:
        if not payload.get("source_ref") and str(source.get("source_ref") or "").strip():
            payload["source_ref"] = str(source.get("source_ref") or "").strip()
        if not payload.get("source_title") and str(source.get("title") or "").strip():
            payload["source_title"] = str(source.get("title") or "").strip()
        if not payload.get("pdf_path") and str(source.get("local_pdf") or source.get("pdf_path") or "").strip():
            payload["pdf_path"] = str(source.get("local_pdf") or source.get("pdf_path") or "").strip()
        if not payload.get("expected_labels"):
            labels = source.get("expected_scheme_or_compound_labels") or source.get("expected_labels") or []
            if labels:
                payload["expected_labels"] = [str(item) for item in labels if str(item or "").strip()]
        if not payload.get("compound_labels") and payload.get("expected_labels"):
            payload["compound_labels"] = list(payload.get("expected_labels") or [])
        if not payload.get("route_sequence_hint") and str(source.get("route_sequence_hint") or "").strip():
            payload["route_sequence_hint"] = str(source.get("route_sequence_hint") or "").strip()
    if not payload.get("source_ref") and str(target_input.get("literature_pdf_source_ref") or "").strip():
        payload["source_ref"] = str(target_input.get("literature_pdf_source_ref") or "").strip()
    if not payload.get("pdf_path") and str(target_input.get("literature_pdf_path") or "").strip():
        payload["pdf_path"] = str(target_input.get("literature_pdf_path") or "").strip()
    hints = _local_pdf_source_hints(
        target=str(target_input.get("target_name") or ""),
        source_ref=str(payload.get("source_ref") or target_input.get("literature_pdf_source_ref") or ""),
        family_hint=str(target_input.get("family_hint") or ""),
    )
    if hints.get("expected_labels") and not payload.get("compound_labels"):
        payload["compound_labels"] = list(hints.get("expected_labels") or [])
    if hints.get("pdf_page_numbers") and not payload.get("page_numbers"):
        payload["page_numbers"] = list(hints.get("pdf_page_numbers") or [])
    if hints.get("pdf_render_zoom") and not payload.get("render_zoom"):
        payload["render_zoom"] = float(hints.get("pdf_render_zoom") or 2.0)
    if hints.get("scheme_crops") and not payload.get("scheme_crops"):
        payload["scheme_crops"] = [dict(item) for item in hints.get("scheme_crops") or [] if isinstance(item, dict)]
    if hints.get("route_sequence_hint") and not payload.get("route_sequence_hint"):
        payload["route_sequence_hint"] = str(hints.get("route_sequence_hint") or "")
    if hints.get("expected_labels") and not payload.get("expected_labels"):
        payload["expected_labels"] = list(hints.get("expected_labels") or [])


def _matching_blackboard_pdf_source(blackboard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    candidates = [
        dict(row)
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict) and str(row.get("local_pdf") or row.get("pdf_path") or row.get("source_pdf_path") or "").strip()
    ]
    if not candidates:
        return {}
    for row in candidates:
        if _blackboard_source_matches_payload(row, payload):
            return row
    if payload.get("source_ref") or payload.get("doi") or payload.get("pdf_path"):
        return {}
    rendered = {_blackboard_source_key(row) for row in evidence.get("pdf_structure_evidence") or [] if isinstance(row, dict)}
    ranked = _rank_blackboard_pdf_sources(blackboard, candidates, rendered={key for key in rendered if key})
    return dict(ranked[0]) if ranked else {}


def _blackboard_source_matches_payload(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    payload_key = _blackboard_source_key(payload)
    row_key = _blackboard_source_key(row)
    if payload_key and row_key and payload_key == row_key:
        return True
    for field in ("source_ref", "doi", "pii", "url"):
        requested = str(payload.get(field) or "").strip().lower()
        if requested and requested == str(row.get(field) or "").strip().lower():
            return True
    requested_pdf = str(payload.get("pdf_path") or payload.get("local_pdf") or payload.get("source_pdf_path") or "").strip()
    row_pdf = str(row.get("local_pdf") or row.get("pdf_path") or row.get("source_pdf_path") or "").strip()
    return bool(requested_pdf and row_pdf and Path(requested_pdf).expanduser().resolve() == Path(row_pdf).expanduser().resolve())


def _rank_blackboard_pdf_sources(
    blackboard: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    rendered: set[str],
) -> list[dict[str, Any]]:
    target = dict(blackboard.get("target_profile") or {})
    terms = _priority_terms(
        [
            str(target.get("target_name") or ""),
            str(target.get("family_hint") or ""),
            *[str(item) for item in target.get("functional_handles") or []],
        ]
    )

    def score(row: dict[str, Any]) -> tuple[int, str]:
        key = _blackboard_source_key(row)
        text = _priority_source_text(row)
        pdf_path = str(row.get("local_pdf") or row.get("pdf_path") or row.get("source_pdf_path") or "").strip()
        value = 0
        if pdf_path and Path(pdf_path).exists():
            value += 60
        if key and key not in rendered:
            value += 80
        if str(row.get("access_status") or "").lower() == "local_pdf_available":
            value += 15
        if str(row.get("source_role") or "").lower() == "local_pdf_proxy_download":
            value += 10
        if str(row.get("doi") or "").strip() or str(row.get("source_ref") or "").lower().startswith("doi:"):
            value += 18
            title = str(row.get("title") or row.get("source_title") or "").strip().lower()
            if not title or title.startswith("pdfreq"):
                value += 35
        value += 10 * sum(1 for term in terms if term and term in text)
        process_terms = (
            "synthesis",
            "preparation",
            "process",
            "route",
            "scheme",
            "intermediate",
            "kilogram",
            "kg",
            "scale",
        )
        value += 8 * sum(1 for term in process_terms if term in text)
        if "improved kilogram-scale preparation" in text:
            value += 40
        if "discovery" in text and not any(term in text for term in ("synthesis", "preparation", "process", "scheme")):
            value -= 10
        if key and key in rendered:
            value -= 120
        return value, key or pdf_path

    return sorted(candidates, key=score, reverse=True)


def _blackboard_source_key(row: dict[str, Any]) -> str:
    pdf = str(row.get("local_pdf") or row.get("source_pdf_path") or row.get("pdf_path") or "").strip().lower()
    if pdf:
        return f"pdf:{pdf}"
    document_id = str(row.get("document_id") or "").strip().lower()
    if document_id:
        return f"document:{document_id}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        return f"pii:{pii}"
    title = str(row.get("title") or row.get("source_title") or "").strip().lower()
    return f"title:{title}" if title else ""


def _priority_source_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "source_ref",
            "title",
            "source_title",
            "doi",
            "url",
            "route_sequence_hint",
            "relevance_rationale",
        )
    ).lower()


def _priority_terms(values: list[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        for token in str(value or "").lower().replace(";", " ").replace(",", " ").split():
            token = token.strip("()[]{}:._-/")
            if len(token) >= 5 and token not in {"online", "local", "cache", "source", "target"}:
                terms.append(token)
    return _dedupe(terms)[:10]


def _visual_action_output_dir(action: dict[str, Any]) -> str:
    action_id = str(action.get("action_id") or action.get("action_type") or "visual").strip()
    source = str((action.get("payload") or {}).get("source_ref") or (action.get("payload") or {}).get("pdf_path") or "").strip()
    raw = "__".join(part for part in [action_id, source] if part)
    safe_action = _compact_safe_path_token(action_id, max_len=34) or "visual"
    safe_source = _compact_safe_path_token(source, max_len=24)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    suffix = "__".join(part for part in [safe_action, safe_source, digest] if part)
    return f"visual_lit_chain_{suffix}"


def _structure_resolution_action_output_dir(action: dict[str, Any]) -> str:
    action_id = str(action.get("action_id") or action.get("action_type") or "structure_resolution").strip()
    payload = dict(action.get("payload") or {})
    label = str(payload.get("label") or payload.get("compound_label") or "").strip()
    source = str(payload.get("source_ref") or payload.get("pdf_path") or "").strip()
    raw = "__".join(part for part in [action_id, source, label] if part)
    safe_action = _compact_safe_path_token(action_id, max_len=36) or "structure"
    safe_label = _compact_safe_path_token(label, max_len=24)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    suffix = "__".join(part for part in [safe_action, safe_label, digest] if part)
    return f"lit_struct_res_{suffix}"


def _compact_safe_path_token(value: str, *, max_len: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:max_len]


def _pdf_action_output_dir(action: dict[str, Any]) -> str:
    action_id = str(action.get("action_id") or action.get("action_type") or "pdf").strip()
    source = str((action.get("payload") or {}).get("source_ref") or (action.get("payload") or {}).get("pdf_path") or "").strip()
    raw = "__".join(part for part in [action_id, source] if part)
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return f"literature_pdf_structure_extraction_{safe or 'pdf'}"


def _normalize_literature_sources(
    *,
    literature_pdf_path: str | Path,
    literature_pdf_source_ref: str,
    literature_sources: list[dict[str, Any]] | None,
    auto_discover_local_pdfs: bool = True,
    local_pdf_search_dirs: list[str | Path] | None = None,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(literature_sources or [], start=1):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or row.get("path") or "").strip()
        if local_pdf:
            resolved_pdf = str(Path(local_pdf).expanduser().resolve())
            row["local_pdf"] = resolved_pdf
            row.setdefault(
                "document_id",
                f"pdf:{hashlib.sha256(resolved_pdf.lower().encode('utf-8')).hexdigest()[:16]}",
            )
            inferred_scope = _infer_literature_content_scope(resolved_pdf)
            explicit_scope = str(row.get("content_scope") or "").strip().lower()
            if inferred_scope == "supplementary_information" and explicit_scope in {
                "",
                "article",
                "main_article",
            }:
                row["content_scope"] = inferred_scope
                if explicit_scope:
                    row["content_scope_normalization"] = {
                        "input": explicit_scope,
                        "normalized": inferred_scope,
                        "reason": "supplementary_filename_overrides_generic_article_default",
                    }
            else:
                row.setdefault("content_scope", inferred_scope)
        row.setdefault("source_ref", str(row.get("ref") or row.get("source_ref") or "").strip())
        row.setdefault("source_role", "local_cache")
        row.setdefault("candidate_id", f"provided_pdf_{idx}")
        rows.append(row)
    if str(literature_pdf_path or "").strip():
        rows.append(
            {
                "candidate_id": "provided_pdf_legacy",
                "source_ref": str(literature_pdf_source_ref or "").strip(),
                "local_pdf": str(Path(literature_pdf_path).expanduser().resolve()),
                "source_role": "user_provided_local_pdf_seed",
                "user_provided_source_seed": True,
                "source_usage_policy": "codex_online_first_then_user_seed_fallback_or_agent_metadata_match",
            }
        )
    rows.extend(_local_pdf_proxy_download_sources(run_dir))
    rows.extend(
        _discover_auto_local_pdf_cache(
            enabled=auto_discover_local_pdfs,
            search_dirs=local_pdf_search_dirs,
            run_dir=run_dir,
        )
    )
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for row in rows:
        key = _literature_document_key(row)
        if not key:
            continue
        if key in seen:
            pos = seen[key]
            if str(row.get("local_pdf") or "").strip() and not str(out[pos].get("local_pdf") or "").strip():
                out[pos] = {**out[pos], **row}
            continue
        seen[key] = len(out)
        out.append(row)
    return out


def _refresh_blackboard_from_local_pdf_proxy_downloads(
    blackboard: dict[str, Any],
    *,
    run_dir: Path | None,
) -> dict[str, Any]:
    """Merge late browser/local-proxy PDF downloads into an active blackboard."""
    downloaded_sources = _local_pdf_proxy_download_sources(run_dir)
    if not downloaded_sources:
        return blackboard

    board = dict(blackboard)
    evidence = dict(board.get("literature_evidence") or {})
    candidates: list[dict[str, Any]] = [
        dict(row)
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict)
    ]
    index: dict[str, int] = {}
    for pos, candidate in enumerate(candidates):
        key = _candidate_merge_key(candidate) or f"existing:{pos}"
        index[key] = pos

    added = 0
    for source in downloaded_sources:
        key = _candidate_merge_key(source) or f"download:{len(candidates)}"
        candidate = {
            **source,
            "schema_version": "literature_source_candidate.v1",
            "source_type": str(source.get("source_type") or "local_pdf"),
            "source_discovery_mode": "local_pdf_proxy_manifest_refresh",
            "access_status": "local_pdf_available",
            "extraction_task_recommendations": _dedupe(
                [
                    *[str(item) for item in source.get("extraction_task_recommendations") or []],
                    "extract_pdf_literature_structures",
                    "extract_visual_literature_chain",
                    "compile_exact_literature_rows",
                ]
            ),
            "relevance_rationale": str(
                source.get("relevance_rationale")
                or "browser/local-proxy PDF download is now available for source-detail extraction"
            ),
            "no_solved_claim": True,
        }
        if key in index:
            pos = index[key]
            before_pdf = str(candidates[pos].get("local_pdf") or "").strip()
            candidates[pos] = _merge_source_candidate(candidates[pos], candidate)
            if not before_pdf and str(candidates[pos].get("local_pdf") or "").strip():
                added += 1
            continue
        index[key] = len(candidates)
        candidates.append(candidate)
        added += 1

    evidence["source_candidates"] = candidates
    evidence["source_refs"] = _dedupe(
        [
            *[str(item) for item in evidence.get("source_refs") or []],
            *[str(row.get("source_ref") or "") for row in downloaded_sources],
            *[
                f"doi:{str(row.get('doi') or '').strip()}"
                for row in downloaded_sources
                if str(row.get("doi") or "").strip()
            ],
        ]
    )
    if added:
        evidence["confidence"] = "local_pdf_available"
    prior_mode = str(evidence.get("source_discovery_mode") or "").strip()
    if added:
        if prior_mode and "local_pdf_proxy_manifest" not in prior_mode:
            evidence["source_discovery_mode"] = f"{prior_mode}+local_pdf_proxy_manifest"
        else:
            evidence["source_discovery_mode"] = prior_mode or "local_pdf_proxy_manifest_refresh"
    evidence["fallback_order"] = _dedupe(
        [*[str(item) for item in evidence.get("fallback_order") or []], "codex_online", "local_pdf", "placeholder"]
    )
    evidence["local_pdf_proxy_download_count"] = len(downloaded_sources)
    evidence["late_local_pdf_proxy_downloads_merged"] = int(added)
    board["literature_evidence"] = evidence

    belief = dict(board.get("current_belief") or {})
    belief["next_action_bias"] = _dedupe(
        [
            *[str(item) for item in belief.get("next_action_bias") or []],
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "compile_exact_literature_rows",
        ]
    )
    board["current_belief"] = belief

    if run_dir is not None:
        artifact_refs = dict(board.get("artifact_refs") or {})
        artifact_refs["local_pdf_proxy_download_manifest"] = str(local_pdf_proxy_download_manifest_path(run_dir).resolve())
        board["artifact_refs"] = artifact_refs

    _refresh_source_lifecycle(board)
    return board


def _local_pdf_proxy_download_sources(run_dir: Path | None) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    manifest = local_pdf_proxy_download_manifest_path(run_dir)
    if not manifest.exists():
        return []
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if str(record.get("status") or "").strip().lower() != "downloaded":
            continue
        pdf_path = str(record.get("pdf_path") or "").strip()
        if not pdf_path:
            continue
        resolved = Path(pdf_path).expanduser()
        if not resolved.is_file():
            continue
        if not _local_pdf_file_has_pdf_magic(resolved):
            continue
        doi = _normalize_doi(str(record.get("doi") or ""))
        pii = _source_pii(record)
        source_ref = str(record.get("source_ref") or "").strip()
        if not source_ref:
            source_ref = f"doi:{doi}" if doi else (f"pii:{pii}" if pii else f"local_pdf:{resolved.name}")
        rows.append(
            {
                "candidate_id": str(record.get("request_id") or f"local_pdf_proxy_download_{idx}"),
                "source_ref": source_ref,
                "title": str(record.get("title") or resolved.stem),
                "doi": doi,
                "pii": pii,
                "url": str(record.get("url") or (f"https://doi.org/{doi}" if doi else "")),
                "local_pdf": str(resolved.resolve()),
                "source_role": "local_pdf_proxy_download",
                "source_usage_policy": "downloaded_from_local_pdf_proxy_after_agent_metadata_request",
                "local_pdf_index": {
                    "schema_version": "local_pdf_proxy_download_index.v1",
                    "indexed_from": "local_pdf_proxy_download_manifest",
                    "manifest_path": str(manifest.resolve()),
                    "request_id": str(record.get("request_id") or ""),
                    "match_policy": "agent_requested_metadata_source",
                },
            }
        )
    return rows


def _local_pdf_file_has_pdf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(1024).lstrip().startswith(b"%PDF-")
    except OSError:
        return False


def _discover_auto_local_pdf_cache(
    *,
    enabled: bool,
    search_dirs: list[str | Path] | None,
    run_dir: Path | None,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    roots = _local_pdf_search_roots(search_dirs)
    if not roots:
        return []
    max_files = max(0, int(os.environ.get("AUTOPLANNER_AUTO_LOCAL_PDF_MAX_FILES", "64")))
    max_depth = max(0, int(os.environ.get("AUTOPLANNER_AUTO_LOCAL_PDF_MAX_DEPTH", "2")))
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path in _iter_local_pdf_paths(roots=roots, max_depth=max_depth, run_dir=run_dir):
        path_key = str(path.resolve()).lower()
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        row = _auto_local_pdf_cache_row(path, idx=len(rows) + 1)
        if row:
            rows.append(row)
        if max_files and len(rows) >= max_files:
            break
    return rows


def _local_pdf_search_roots(search_dirs: list[str | Path] | None) -> list[Path]:
    raw_dirs: list[str | Path] = []
    if search_dirs:
        raw_dirs.extend(search_dirs)
    else:
        env_dirs = [item for item in os.environ.get("AUTOPLANNER_LOCAL_PDF_SEARCH_DIRS", "").split(os.pathsep) if item.strip()]
        raw_dirs.extend(env_dirs)
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in raw_dirs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def _iter_local_pdf_paths(*, roots: list[Path], max_depth: int, run_dir: Path | None) -> list[Path]:
    out: list[Path] = []
    run_path = Path(run_dir).expanduser().resolve() if run_dir is not None else None
    for root in roots:
        if root.is_file():
            if root.suffix.lower() == ".pdf" and not _should_skip_auto_pdf(root, run_path=run_path):
                out.append(root)
            continue
        for path in sorted(root.rglob("*.pdf")):
            if _pdf_depth(root, path) > max_depth:
                continue
            if _should_skip_auto_pdf(path, run_path=run_path):
                continue
            out.append(path)
    return out


def _pdf_depth(root: Path, path: Path) -> int:
    try:
        return max(0, len(path.relative_to(root).parts) - 1)
    except ValueError:
        return 0


def _should_skip_auto_pdf(path: Path, *, run_path: Path | None) -> bool:
    resolved = path.expanduser().resolve()
    if run_path is not None and _path_is_relative_to(resolved, run_path):
        return True
    parts = {part.lower() for part in resolved.parts}
    if parts & {".git", "__pycache__", ".pytest_cache", "node_modules", "results"}:
        return True
    name = resolved.name.lower()
    generated_markers = (
        "agentic_blackboard",
        "fully_connected_route_graph",
        "current_fullflow_process",
        "retrosynthesis_report",
        "retrosynthesis_structure_route",
    )
    if name.startswith("scheme_route_") or any(marker in name for marker in generated_markers):
        return True
    return False


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _auto_local_pdf_cache_row(path: Path, *, idx: int) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    doi = _first_doi_from_pdf_bytes(resolved)
    pii = _extract_pii(resolved.name)
    title = _title_from_pdf_filename(resolved)
    source_ref = f"doi:{doi}" if doi else (f"pii:{pii}" if pii else f"local_pdf:{resolved.name}")
    url = f"https://doi.org/{doi}" if doi else (_sciencedirect_url_from_pii(pii) if pii else "")
    return {
        "candidate_id": f"auto_local_pdf_{idx}",
        "source_ref": source_ref,
        "title": title,
        "doi": doi,
        "pii": pii,
        "url": url,
        "local_pdf": str(resolved),
        "source_role": "auto_local_pdf_cache",
        "local_pdf_index": {
            "schema_version": "auto_local_pdf_index.v1",
            "indexed_from": "local_pdf_search_dirs",
            "match_policy": "agent_discovered_metadata_required",
            "doi_from_pdf_bytes": bool(doi),
            "pii_from_filename": bool(pii),
            "filename_title_key": _title_key(title),
        },
    }


def _first_doi_from_pdf_bytes(path: Path) -> str:
    try:
        limit = max(8192, int(os.environ.get("AUTOPLANNER_AUTO_LOCAL_PDF_DOI_SCAN_BYTES", "2000000")))
        text = path.read_bytes()[:limit].decode("latin-1", errors="ignore")
    except OSError:
        return ""
    for raw in re.findall(r"10\.\d{4,9}/[^\s<>()\"'\]\[]+", text, flags=re.IGNORECASE):
        doi = _normalize_doi(raw)
        if _is_specific_doi(doi):
            return doi
    try:
        import fitz  # type: ignore

        with fitz.open(path) as doc:
            page_limit = min(len(doc), max(1, int(os.environ.get("AUTOPLANNER_AUTO_LOCAL_PDF_DOI_SCAN_PAGES", "3"))))
            text = "\n".join(doc[index].get_text() for index in range(page_limit))
    except Exception:
        return ""
    for raw in re.findall(r"10\.\d{4,9}/[^\s<>()\"'\]\[]+", text, flags=re.IGNORECASE):
        doi = _normalize_doi(raw)
        if _is_specific_doi(doi):
            return doi
    return ""


def _title_from_pdf_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^1-s2\.0-", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"-?main$", "", stem, flags=re.IGNORECASE)
    text = " ".join(part for part in re.split(r"[_\-]+", stem) if part)
    return text.strip() or path.stem


def _attach_literature_sources(target_data: dict[str, Any], sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    target_data["literature_sources"] = [dict(row) for row in sources]
    cache_rows = [dict(row) for row in sources if _source_is_local_cache(row)]
    if cache_rows:
        target_data["local_literature_cache"] = cache_rows
    direct_rows = [dict(row) for row in sources if _source_is_direct_local_pdf(row)]
    first = direct_rows[0] if direct_rows else {}
    if first and str(first.get("local_pdf") or "").strip():
        target_data["literature_pdf_path"] = str(first.get("local_pdf") or "").strip()
    if first and str(first.get("source_ref") or "").strip():
        target_data["literature_pdf_source_ref"] = str(first.get("source_ref") or "").strip()


def _target_literature_sources(target_input: dict[str, Any], payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = dict(payload or {})
    rows = [dict(row) for row in target_input.get("literature_sources") or [] if isinstance(row, dict)]
    rows.extend(dict(row) for row in target_input.get("local_literature_cache") or [] if isinstance(row, dict))
    if str(payload.get("pdf_path") or "").strip():
        rows.insert(
            0,
            {
                "source_ref": str(payload.get("source_ref") or "").strip(),
                "title": str(payload.get("source_title") or "").strip(),
                "local_pdf": str(payload.get("pdf_path") or "").strip(),
                "source_role": "direct_action_payload",
            },
        )
    elif str(target_input.get("literature_pdf_path") or "").strip() and not rows:
        rows.append(
            {
                "source_ref": str(target_input.get("literature_pdf_source_ref") or "").strip(),
                "local_pdf": str(target_input.get("literature_pdf_path") or "").strip(),
                "source_role": "direct_local_pdf_fallback",
            }
        )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _literature_document_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _literature_document_key(row: dict[str, Any]) -> str:
    """Identify an extractable document without collapsing article and SI files.

    A DOI identifies a scholarly source, but a source can contain multiple
    independently extractable documents.  The host-derived identity joins a
    metadata pointer to its downloaded representation while retaining article
    and SI as separate documents in one independent source group.
    """
    logical_document = source_document_identity(row)
    if logical_document:
        return logical_document
    doi = _source_doi(row)
    if doi:
        return f"doi:{doi}"
    pii = _source_pii(row).strip().lower()
    if pii:
        return f"pii:{pii}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    url = str(row.get("url") or "").strip().lower()
    return f"url:{url}" if url else ""


def _infer_literature_content_scope(path: str) -> str:
    name = Path(path).name.lower()
    if any(token in name for token in ("thesis", "dissertation")):
        return "thesis"
    return source_content_scope({"local_pdf": path})


def _target_literature_cache_sources(target_input: dict[str, Any], payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in _target_literature_sources(target_input, payload) if _source_is_local_cache(row)]


def _direct_local_pdf_sources(target_input: dict[str, Any], payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in _target_literature_sources(target_input, payload) if _source_is_direct_local_pdf(row)]


def _source_is_local_cache(row: dict[str, Any]) -> bool:
    role = str(row.get("source_role") or row.get("source_usage") or "").strip().lower()
    return role in {
        "local_cache",
        "literature_cache",
        "pdf_cache",
        "cache",
        "auto_local_pdf_cache",
        "local_pdf_proxy_download",
    }


def _source_is_auto_local_cache(row: dict[str, Any]) -> bool:
    role = str(row.get("source_role") or row.get("source_usage") or "").strip().lower()
    return role == "auto_local_pdf_cache"


def _source_is_direct_local_pdf(row: dict[str, Any]) -> bool:
    role = str(row.get("source_role") or row.get("source_usage") or "").strip().lower()
    return role in {
        "user_provided_local_pdf_seed",
        "direct_local_pdf_fallback",
        "direct_action_payload",
        "forced_local_pdf",
        "legacy_direct",
        "direct",
    }


def _matching_literature_source(target_input: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    sources = _target_literature_sources(target_input, {})
    source_ref = str(payload.get("source_ref") or "").strip().lower()
    pdf_path = str(payload.get("pdf_path") or "").strip()
    pdf_key = str(Path(pdf_path).expanduser().resolve()).lower() if pdf_path else ""
    for row in sources:
        row_ref = str(row.get("source_ref") or "").strip().lower()
        row_pdf = str(row.get("local_pdf") or row.get("pdf_path") or "").strip()
        row_pdf_key = str(Path(row_pdf).expanduser().resolve()).lower() if row_pdf else ""
        if source_ref and row_ref == source_ref:
            return dict(row)
        if pdf_key and row_pdf_key == pdf_key:
            return dict(row)
    return dict(sources[0]) if sources else {}


def _workflow_plan_from_actions(action_batches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "agentic_blackboard_workflow_plan.v1",
        "recommended_strategy": "policy_driven_dag_blackboard",
        "planned_action_batches": action_batches,
        "planner_authority": "action_selection_only",
        "final_verdict_authority": "deterministic_parent_route_proof",
        "raw_reaction_output_allowed": False,
    }


def _invalid_final(preflight: dict[str, Any]) -> FinalVerdict:
    return FinalVerdict(
        case_id=str(preflight.get("case_id") or "target"),
        verdict="invalid_input",
        route_status="invalid_input",
        reasons=[str(item) for item in preflight.get("reasons") or ["invalid_smiles"]],
    )


def _parent_proof_accepted(blackboard: dict[str, Any]) -> bool:
    proof = dict(blackboard.get("parent_route_proof") or {})
    return _blackboard_parent_proof_solved(blackboard, proof)


def _blackboard_parent_proof_solved(
    blackboard: dict[str, Any],
    proof: dict[str, Any] | None = None,
) -> bool:
    target = dict(blackboard.get("target_profile") or {})
    return is_solved_parent_route_proof(
        dict(proof or blackboard.get("parent_route_proof") or {}),
        expected_target_smiles=str(target.get("target_smiles") or ""),
    )


def _result(
    run_dir: Path,
    target_input: dict[str, Any],
    preflight: dict[str, Any],
    blackboard: dict[str, Any],
    action_batches: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    artifact_bundle: ArtifactBundle,
    final_verdict: FinalVerdict,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_refs = dict(blackboard.get("artifact_refs") or {})
    artifacts = {
        "target_input": str(run_dir / "target_input.json"),
        "preflight": str(run_dir / "preflight.json"),
        "agent_blackboard": str(run_dir / "agent_blackboard.json"),
        "decision_trace": str(run_dir / "decision_trace.jsonl"),
        "tool_calls": str(run_dir / "tool_calls.jsonl"),
        "artifact_bundle": str(run_dir / "artifact_bundle.json"),
        "final_verdict": str(run_dir / "final_verdict.json"),
    }
    for key in (
        "explored_route_forest",
        "route_forest_html",
        "route_forest_error",
        "closeout_revision_manifest",
        "closeout_latest_pointer",
    ):
        if artifact_refs.get(key):
            artifacts[key] = artifact_refs[key]
    return {
        "schema_version": "agentic_blackboard_controller_result.v1",
        "run_dir": str(run_dir),
        "target_input": target_input,
        "preflight": preflight,
        "agent_blackboard": blackboard,
        "action_batches": action_batches,
        "validations": validations,
        "tool_calls": tool_calls,
        "artifact_bundle": artifact_bundle.to_dict(),
        "final_verdict": final_verdict.to_dict(),
        "artifacts": artifacts,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
