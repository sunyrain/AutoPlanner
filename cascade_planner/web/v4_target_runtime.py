"""Target-solve dependency assembly and background-job progress projections."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from threading import Event, RLock
import time
from typing import Any, Callable, Mapping

from cascade_planner.application.run_kernel import project_observed_model_totals
from cascade_planner.interfaces.campaign_action_timeline import (
    compile_campaign_action_timeline,
)
from cascade_planner.interfaces.campaign_gateway import (
    CampaignGateway,
    CampaignGatewayError,
)
from cascade_planner.interfaces.target_delivery import (
    delivery_projection as _delivery_projection,
)
from cascade_planner.interfaces.target_job_projection import (
    compact_solve_result as _compact_solve_result,
    job_projection,
    new_run_id,
    utc_now,
)
from cascade_planner.interfaces.target_solve_request import (
    _bool,
    solve_target_request,
)
from cascade_planner.interfaces.target_solver import TargetSolveCancelled


GatewayFactory = Callable[[], CampaignGateway]
_NONTERMINAL_CAMPAIGN_STATES = frozenset({"created", "running", "paused"})
_TERMINAL_CAMPAIGN_STATES = frozenset(
    {"completed", "unresolved", "budget_exhausted", "cancelled", "failed"}
)
_RUN_ACTIVITY_STALE_AFTER_S = 2 * 60 * 60


def run_target_job(
    factory: GatewayFactory,
    payload: dict[str, Any],
    job_id: str,
    jobs: dict[str, dict[str, Any]],
    lock: RLock,
    cancel_event: Event | None = None,
) -> None:
    started = time.monotonic()
    continuation_pass_count = 0
    compact: dict[str, Any] = {}
    runtime_pause = False
    gateway: CampaignGateway | None = None
    with lock:
        if (cancel_event is not None and cancel_event.is_set()) or str(
            jobs[job_id].get("status") or ""
        ) == "cancelling":
            jobs[job_id].update(updated_at=utc_now())
        else:
            jobs[job_id].update(
                status="running",
                phase="initializing_campaign",
                started_at=utc_now(),
                updated_at=utc_now(),
            )
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise TargetSolveCancelled("target_solve_cancelled_by_user")
        request_payload = dict(payload)
        while True:
            gateway = factory()
            result = (
                solve_target_request(
                    gateway,
                    request_payload,
                    cancel_event=cancel_event,
                )
                if cancel_event is not None
                else solve_target_request(gateway, request_payload)
            )
            compact = _compact_solve_result(result)
            runtime_pause = _target_result_runtime_pause(result)
            if cancel_event is not None and cancel_event.is_set():
                raise TargetSolveCancelled("target_solve_cancelled_by_user")
            accepted = compact.get("accepted") is True
            objective_achieved = bool(compact.get("objective_achieved") is True or accepted)
            stop = dict(result.get("stop_decision") or {})
            should_continue = bool(
                not objective_achieved
                and _bool(payload, "auto_continue", True)
                and stop.get("decision") == "paused"
                and stop.get("terminal") is not True
                and not runtime_pause
            )
            if not should_continue:
                break
            if cancel_event is not None and cancel_event.is_set():
                raise TargetSolveCancelled("target_solve_cancelled_by_user")
            continuation_pass_count += 1
            with lock:
                jobs[job_id].update(
                    phase="automatic_deficit_continuation",
                    updated_at=utc_now(),
                    result=compact,
                    continuation_pass_count=continuation_pass_count,
                )
            request_payload = {**payload, "resume": True}
        status, error = (
            ("paused" if runtime_pause else "complete" if objective_achieved else "unresolved"),
            "",
        )
        phase = (
            "runtime_unavailable"
            if runtime_pause
            else "complete"
            if objective_achieved
            else "unresolved"
        )
    except TargetSolveCancelled:
        with lock:
            cancellation_reason = str(jobs[job_id].get("cancellation_reason") or "user_requested")
        try:
            (gateway or factory()).cancel(
                str(payload.get("run_id") or ""),
                reasons=("user_requested_termination", cancellation_reason),
                idempotency_key=f"web:{job_id}:cancel",
            )
        except CampaignGatewayError as exc:
            if not str(exc).startswith("run_not_found:"):
                status, phase = "failed", "failed"
                error = f"campaign_cancel_failed: {exc}"[:4_000]
            else:
                status, phase, error = "cancelled", "cancelled", ""
        except Exception as exc:  # pragma: no cover - durable cancellation boundary
            status, phase = "failed", "failed"
            error = f"campaign_cancel_failed: {type(exc).__name__}: {exc}"[:4_000]
        else:
            status, phase, error = "cancelled", "cancelled", ""
    except Exception as exc:  # pragma: no cover - integration failure boundary
        if cancel_event is not None and cancel_event.is_set():
            status, phase, error = "cancelled", "cancelled", ""
        else:
            compact, status, phase = {}, "failed", "failed"
            error = f"{type(exc).__name__}: {exc}"[:4_000]
    elapsed = round(time.monotonic() - started, 3)
    with lock:
        terminal_fields = (
            {
                "cancelled_at": utc_now(),
                "cancellation_reason": str(
                    jobs[job_id].get("cancellation_reason") or "user_requested"
                ),
            }
            if status == "cancelled"
            else {}
        )
        jobs[job_id].update(
            status=status,
            phase=phase,
            runtime_pause=runtime_pause,
            finished_at=utc_now(),
            updated_at=utc_now(),
            elapsed_s=elapsed,
            error=error,
            result=compact,
            continuation_pass_count=continuation_pass_count,
            **terminal_fields,
        )


def _target_result_runtime_pause(result: Mapping[str, Any]) -> bool:
    if result.get("runtime_pause") is True:
        return True
    return any(
        isinstance(row, Mapping)
        and (
            row.get("runtime_pause") is True
            or row.get("runtime_unavailable") is True
            or str(row.get("status") or "") == "runtime_unavailable"
        )
        for row in result.get("director_outcomes") or ()
    )


def live_job_progress(factory: GatewayFactory, job: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(job.get("run_id") or "")
    job_status = str(job.get("status") or "")
    execution_active = job_status in {"queued", "running", "cancelling"}
    result: dict[str, Any] = {
        "phase": str(job.get("phase") or job.get("status") or "unknown"),
        "elapsed_s": float(job.get("elapsed_s") or 0.0),
        "execution_active": execution_active,
        "cancellation_available": job.get("cancellation_available") is not False,
        "model_cost": {},
        "frontier_counts": {},
        "stages": [],
        "action_timeline": compile_campaign_action_timeline(()),
        "delivery": _delivery_projection([], job_status=str(job.get("status") or "")),
    }
    # A catalog row is labelled ``historical`` because no Web worker owns it,
    # not because its canonical Kernel state is unavailable.  A resolved
    # registry lookup carries that state in ``_status_result``; consume it so
    # the detail page can show the saved routes, stock closure, and stages.
    # List-only manifest rows intentionally omit this private value and keep
    # the cheap historical summary below.
    resolved_kernel_status = isinstance(job.get("_status_result"), Mapping)
    if (job_status == "historical" and not resolved_kernel_status) or (
        job_status == "paused" and job.get("runtime_pause") is not True
    ):
        persisted_progress = job.get("progress")
        if isinstance(persisted_progress, Mapping):
            result.update(dict(persisted_progress))
        result.update(
            phase=str(job.get("phase") or job_status),
            execution_active=False,
            cancellation_available=False,
        )
        delivery = dict(result.get("delivery") or {})
        delivery.update(execution_active=False, cancellation_available=False)
        result["delivery"] = delivery
        return result
    if job.get("status") in {"running", "cancelling"} and job.get("started_at"):
        try:
            started = datetime.fromisoformat(str(job["started_at"]).replace("Z", "+00:00"))
            result["elapsed_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
        except ValueError:
            pass
    status_result = job.get("_status_result")
    if not isinstance(status_result, Mapping):
        try:
            status_result = factory().status(run_id)
        except Exception:
            return result
    status_result = dict(status_result)
    status = dict(status_result.get("status") or {})
    stop_decision = dict(status.get("stop_decision") or {})
    campaign_status = str(status.get("status") or "").casefold()
    campaign_terminal = bool(
        stop_decision.get("terminal") is True or campaign_status in _TERMINAL_CAMPAIGN_STATES
    )
    campaign_decision = str(stop_decision.get("decision") or campaign_status)
    execution_active = bool(execution_active and not campaign_terminal)
    cancellation_available = bool(
        job.get("cancellation_available") is not False
        and not campaign_terminal
        and job_status != "paused"
    )
    portfolio = dict(status.get("portfolio") or {})
    selected_routes = [
        dict(value)
        for value in portfolio.get("selected_routes") or []
        if isinstance(value, Mapping)
    ]
    strict_complete_route_count = int(
        dict(portfolio.get("closeout") or {}).get("complete_route_count") or 0
    )
    stock_closed_route_count = sum(
        route.get("all_leaves_stock_closed") is True for route in selected_routes
    )
    frontier = [dict(value) for value in status.get("frontier") or [] if isinstance(value, dict)]
    frontier_counts: dict[str, int] = {}
    for value in frontier:
        kind = str(value.get("kind") or "other")
        frontier_counts[kind] = frontier_counts.get(kind, 0) + 1
    result.update(
        campaign_status=campaign_status,
        campaign_terminal=campaign_terminal,
        campaign_decision=campaign_decision,
        execution_active=execution_active,
        cancellation_available=cancellation_available,
        graph_revision=int(status.get("graph_revision") or 0),
        evidence_revision=int(status.get("evidence_revision") or 0),
        attempt_count=int(status.get("attempt_count") or 0),
        accepted_expansion_count=int(status.get("accepted_expansion_count") or 0),
        model_cost=_model_cost_with_in_flight_checkpoints(status),
        frontier_counts=frontier_counts,
        next_deficit_id=str(dict(status.get("stop_decision") or {}).get("next_deficit_id") or ""),
        portfolio={
            "accepted": portfolio.get("accepted") is True,
            "route_count": len(selected_routes),
            "selected_route_count": len(selected_routes),
            "stock_closed_route_count": stock_closed_route_count,
            "strict_proof_complete_route_count": strict_complete_route_count,
            "complete_route_count": strict_complete_route_count,
            "deficit_count": len(portfolio.get("deficits") or []),
            "semantics": {
                "route_count_means_selected_candidates": True,
                "stock_closed_is_separate_from_strict_proof_completion": True,
                "complete_route_count_is_strict_proof_completion": True,
            },
        },
    )
    run_dir = Path(str(status_result.get("run_dir") or ""))
    checkpoint = run_dir / ".autoplanner" / "target-solver-checkpoint.json"
    timeline_stages: list[dict[str, Any]] = []
    if checkpoint.is_file():
        try:
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            timeline_stages = [
                dict(row) for row in value.get("stages") or [] if isinstance(row, dict)
            ]
            stages = [
                {
                    "stage": str(row.get("stage") or ""),
                    "status": str(row.get("status") or ""),
                    "elapsed_s": float(row.get("elapsed_s") or 0.0),
                    "started_at": str(row.get("started_at") or ""),
                    "completed_at": str(row.get("completed_at") or ""),
                    "metrics": _stage_progress_metrics(row),
                }
                for row in timeline_stages
            ]
            result["stages"] = stages
            recoverable_outcomes = [
                dict(row)
                for row in value.get("director_outcomes") or []
                if isinstance(row, Mapping)
                and str(row.get("status") or "") == "runtime_unavailable"
                and isinstance(row.get("plan"), Mapping)
            ]
            if recoverable_outcomes:
                recoverable = recoverable_outcomes[-1]
                partial_plan = dict(recoverable.get("plan") or {})
                partial_skeleton_count = len(partial_plan.get("multi_step_skeletons") or [])
                partial_family_count = len(partial_plan.get("route_families") or [])
                projected_portfolio = dict(result.get("portfolio") or {})
                projected_portfolio["route_count"] = max(
                    int(projected_portfolio.get("route_count") or 0),
                    partial_skeleton_count,
                )
                result["portfolio"] = projected_portfolio
                result["recoverable_director_prefix"] = {
                    "available": True,
                    "route_family_count": partial_family_count,
                    "route_skeleton_count": partial_skeleton_count,
                    "resume_required_task_ids": list(
                        recoverable.get("resume_required_task_ids") or []
                    ),
                    "artifact_sha256": str(recoverable.get("artifact_sha256") or ""),
                    "semantics": {
                        "route_count_is_not_canonical_admission": True,
                        "resume_replays_completed_worker_records": True,
                    },
                }
            if stages:
                result["phase"] = stages[-1]["stage"]
            initial = next(
                (
                    row
                    for row in stages
                    if row["stage"] == "initial_workbench"
                    and row["status"] == "completed"
                    and row["completed_at"]
                ),
                None,
            )
            if initial and job.get("started_at"):
                try:
                    job_started = datetime.fromisoformat(
                        str(job["started_at"]).replace("Z", "+00:00")
                    )
                    first_route = datetime.fromisoformat(
                        str(initial["completed_at"]).replace("Z", "+00:00")
                    )
                    result["time_to_first_route_s"] = round(
                        max(0.0, (first_route - job_started).total_seconds()), 3
                    )
                except ValueError:
                    pass
        except (OSError, json.JSONDecodeError):
            pass
    result["action_timeline"] = compile_campaign_action_timeline(
        timeline_stages,
        active_actions=(
            dict(row)
            for row in (() if campaign_terminal else status.get("active_actions") or [])
            if isinstance(row, Mapping)
        ),
    )
    if not timeline_stages and not campaign_terminal:
        active_records = [
            dict(row)
            for row in result["action_timeline"].get("records") or []
            if isinstance(row, Mapping) and row.get("state") == "running"
        ]
        if active_records:
            result["phase"] = str(active_records[0].get("kind") or result["phase"])
    if campaign_terminal:
        result["phase"] = campaign_decision or campaign_status
    final_projection = campaign_terminal or job_status not in {
        "queued",
        "running",
        "cancelling",
        "paused",
    }
    if final_projection:
        accepted = dict(result.get("portfolio") or {}).get("accepted") is True
        result["scientific_status"] = "accepted" if accepted else "unresolved"
        try:
            snapshot = dict(factory().workbench(run_id).get("snapshot") or {})
            workbench_portfolio = dict(snapshot.get("portfolio") or {})
            profile_counts = {
                str(key): int(value or 0)
                for key, value in dict(
                    workbench_portfolio.get("acceptance_profile_counts") or {}
                ).items()
            }
            result.update(
                proof_profile_known=True,
                achieved_profile=str(workbench_portfolio.get("achieved_profile") or "unresolved"),
                acceptance_profile_counts=profile_counts,
                condition_complete_route_count=int(profile_counts.get("condition_complete") or 0),
                process_ready_route_count=int(profile_counts.get("process_ready") or 0),
                closeout_reasons=list(
                    dict(workbench_portfolio.get("closeout") or {}).get("reasons") or []
                ),
            )
        except Exception:
            result["proof_profile_known"] = False
    delivery = _delivery_projection(
        list(result.get("stages") or []),
        job_status=(campaign_decision or campaign_status) if campaign_terminal else job_status,
    )
    if job_status == "historical":
        accepted = result.get("scientific_status") == "accepted"
        route_candidates = int(dict(result.get("portfolio") or {}).get("route_count") or 0) > 0
        delivery.update(
            state="historical_accepted" if accepted else "historical_unresolved",
            route_candidates_available=route_candidates,
            workbench_available=route_candidates,
            proof_closure_known=True,
            proof_closure_complete=accepted,
        )
    delivery.update(
        execution_active=execution_active,
        cancellation_available=cancellation_available,
        scientific_status=str(result.get("scientific_status") or ""),
        achieved_profile=str(result.get("achieved_profile") or ""),
        condition_complete_route_count=int(result.get("condition_complete_route_count") or 0),
        proof_profile_known=result.get("proof_profile_known") is True,
    )
    result["delivery"] = delivery
    return result


def _model_cost_with_in_flight_checkpoints(
    status: Mapping[str, Any],
) -> dict[str, int | float | bool]:
    """Project measured paused work without charging it twice.

    Settled usage remains authoritative in ``model_totals``. A recoverable
    Director keeps its task in flight, so its latest Kernel checkpoint is the
    only honest live observation until final settlement replaces it.
    """

    return project_observed_model_totals(status)


def _stage_progress_metrics(row: Mapping[str, Any]) -> dict[str, int | float]:
    stage = str(row.get("stage") or "")
    detail = dict(row.get("detail") or {})
    if stage in {
        "chemenzy_guided_frontier",
        "chemenzy_stock_recovery",
        "aizynthfinder_guided_frontier",
        "aizynthfinder_stock_recovery",
    }:
        return {
            "frontiers": int(detail.get("frontier_count") or 0),
            "provider_calls": int(
                detail.get("provider_invocation_count")
                or detail.get("executed_frontier_count")
                or 0
            ),
            "proposals": int(detail.get("proposal_count") or 0),
        }
    if stage in {"chemenzy_delegation", "aizynthfinder_delegation"}:
        return {
            "requests": int(detail.get("request_count") or 0),
            "queued": int(detail.get("queued_count") or 0),
            "rejected": int(detail.get("rejected_count") or 0),
        }
    if stage in {"evidence_acquisition", "replan_evidence_acquisition"}:
        prefetch = dict(detail.get("prefetch") or {})
        source_route = dict(detail.get("source_route") or {})
        source_validation = dict(source_route.get("validation") or {})
        return {
            "sources": int(detail.get("source_count") or 0),
            "exact_rows": int(detail.get("exact_record_count") or 0),
            "visual_calls": int(detail.get("visual_invocations") or 0),
            "source_route_proposals": int(source_route.get("proposal_count") or 0),
            "source_route_validated": int(source_validation.get("accepted_validation_count") or 0),
            "prefetch_s": round(float(prefetch.get("elapsed_s") or 0.0), 3),
            "hidden_s": round(float(detail.get("latency_hidden_by_global_s") or 0.0), 3),
        }
    return {}


def historical_job(run: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(run.get("run_id") or "")
    costs = dict(run.get("cost_totals") or {})
    graph = dict(run.get("graph") or {})
    deficits = dict(run.get("deficits") or {})
    # A persisted manifest is an immutable snapshot, not a live background job.
    # The canonical kernel status can legitimately remain ``running`` when a
    # validation fork stops with unresolved deficits.  Exposing that value as
    # the console phase makes an archived run look active, so preserve it as
    # provenance while projecting an explicit historical phase.
    campaign_status = str(run.get("status") or "unknown")
    portfolio_accepted = run.get("accepted") is True
    scientific_status = "accepted" if portfolio_accepted else "unresolved"
    route_candidates_available = int(graph.get("complete_route_count") or 0) > 0
    progress = {
        "phase": "historical_snapshot",
        "elapsed_s": float(costs.get("task_wall_time_s") or 0.0),
        "campaign_status": campaign_status,
        "scientific_status": scientific_status,
        "execution_active": False,
        "graph_revision": int(run.get("revision") or 0),
        "model_cost": {
            "model_invocations": int(costs.get("model_invocations") or 0),
            "visual_invocations": int(costs.get("visual_invocations") or 0),
            "input_tokens": int(costs.get("input_tokens") or 0),
            "output_tokens": int(costs.get("output_tokens") or 0),
            "wall_time_s": float(costs.get("wall_time_s") or 0.0),
        },
        "portfolio": {
            "accepted": run.get("accepted") is True,
            "route_count": int(graph.get("complete_route_count") or 0),
            "selected_route_count": int(graph.get("complete_route_count") or 0),
            "stock_closed_route_count": 0,
            "strict_proof_complete_route_count": int(graph.get("complete_route_count") or 0),
            "complete_route_count": int(graph.get("complete_route_count") or 0),
            "deficit_count": sum(int(value or 0) for value in deficits.values()),
            "semantics": {
                "historical_manifest_does_not_encode_stock_closure": True,
                "complete_route_count_is_strict_proof_completion": True,
            },
        },
        "frontier_counts": deficits,
        "stages": [],
        "action_timeline": compile_campaign_action_timeline(()),
        "delivery": {
            "state": "historical",
            "execution_active": False,
            "route_candidates_available": route_candidates_available,
            # Run manifests deliberately do not grant scientific authority.
            # Exact evidence and B3 closure must be read from the validation
            # report/workbench, therefore absence here means unknown, not true.
            "proof_closure_complete": False,
            "proof_closure_known": False,
            "evidence_stage_complete": False,
            "workbench_available": True,
            "semantics": {
                "historical_projection_only": True,
                "historical_campaign_status": campaign_status,
                "portfolio_policy_accepted": portfolio_accepted,
                "scientific_status": scientific_status,
                "route_candidates_do_not_imply_exact_evidence": True,
                "execution_finished_does_not_imply_scientific_acceptance": True,
            },
        },
    }
    return {
        "job_id": f"solve:@main:{run_id}",
        "run_id": run_id,
        "target_name": str(run.get("target_name") or run_id),
        "status": "historical",
        "phase": "historical_snapshot",
        "created_at": str(run.get("updated_at") or ""),
        "started_at": "",
        "finished_at": str(run.get("updated_at") or ""),
        "updated_at": str(run.get("updated_at") or ""),
        "elapsed_s": float(costs.get("task_wall_time_s") or 0.0),
        "error": "",
        "result": {},
        "progress": progress,
        "execution_source": "registry_snapshot",
        "cancellation_available": False,
    }


def run_may_be_live(run: Mapping[str, Any]) -> bool:
    """Return whether an index row requires a canonical Kernel status read."""

    return str(run.get("status") or "").casefold() in _NONTERMINAL_CAMPAIGN_STATES


def registry_job(
    gateway: CampaignGateway,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a persisted run from the canonical Kernel lifecycle state.

    The run index is discovery-only.  In particular, a run discovered outside
    the Web process is not necessarily historical: CLI and batch launchers own
    valid live Kernel executions that never enter the Web ``jobs`` dictionary.
    """

    fallback = historical_job(run)
    run_id = str(run.get("run_id") or "")
    if not run_id:
        return fallback
    run_status = str(run.get("status") or "").casefold()
    if run_status == "paused":
        activity_observed_at, activity_stale = _run_activity_observation(run, run)
        experiment_outcome = _panel_experiment_outcome(run)
        fallback.update(
            target_smiles=str(run.get("target_smiles") or ""),
            status="paused",
            phase="paused",
            finished_at="",
            run_dir=str(run.get("run_dir") or ""),
            execution_source="registry_snapshot",
            activity_observed_at=activity_observed_at,
            activity_stale=activity_stale,
            campaign_status="paused",
            campaign_terminal=False,
            campaign_decision="paused",
            cancellation_available=False,
            **experiment_outcome,
        )
        progress = dict(fallback.get("progress") or {})
        progress.update(
            phase="paused",
            campaign_status="paused",
            execution_active=False,
            cancellation_available=False,
            **experiment_outcome,
        )
        fallback["progress"] = progress
        return fallback
    try:
        status_result = dict(gateway.status(run_id) or {})
    except Exception:
        return fallback
    campaign = dict(status_result.get("status") or {})
    campaign_spec = dict(status_result.get("campaign_spec") or campaign.get("campaign_spec") or {})
    target = dict(campaign_spec.get("target") or {})
    stop_decision = dict(campaign.get("stop_decision") or {})
    campaign_status = str(campaign.get("status") or run.get("status") or "unknown").casefold()
    campaign_terminal = stop_decision.get("terminal") is True
    campaign_decision = str(stop_decision.get("decision") or "")
    activity_observed_at, activity_stale = _run_activity_observation(
        run,
        status_result,
    )
    reported_activity_stale = (
        activity_stale
        if not campaign_terminal and campaign_status in _NONTERMINAL_CAMPAIGN_STATES
        else False
    )
    common = {
        "target_smiles": str(target.get("canonical_smiles") or ""),
        "campaign_status": campaign_status,
        "campaign_terminal": campaign_terminal,
        "campaign_decision": campaign_decision,
        "run_dir": str(status_result.get("run_dir") or run.get("run_dir") or ""),
        "execution_source": "kernel_registry",
        "activity_observed_at": activity_observed_at,
        "activity_stale": reported_activity_stale,
        "cancellation_available": False,
        "_status_result": status_result,
    }
    active_state_is_stale = campaign_status in {"created", "running"} and activity_stale
    if (
        campaign_terminal
        or campaign_status not in _NONTERMINAL_CAMPAIGN_STATES
        or active_state_is_stale
    ):
        fallback.update(common)
        progress = dict(fallback.get("progress") or {})
        progress.update(
            campaign_status=campaign_status,
            execution_active=False,
            cancellation_available=False,
        )
        fallback["progress"] = progress
        return fallback

    status = {
        "created": "queued",
        "paused": "paused",
    }.get(campaign_status, "running")
    costs = dict(run.get("cost_totals") or {})
    updated_at = str(run.get("updated_at") or "")
    return {
        "job_id": f"solve:@main:{run_id}",
        "run_id": run_id,
        "target_name": str(run.get("target_name") or run_id),
        "target_smiles": common["target_smiles"],
        "status": status,
        "phase": status,
        "created_at": updated_at,
        "started_at": "",
        "finished_at": "",
        "updated_at": updated_at,
        "elapsed_s": float(costs.get("task_wall_time_s") or 0.0),
        "error": "",
        "result": {},
        **common,
    }


def _panel_experiment_outcome(run: Mapping[str, Any]) -> dict[str, Any]:
    """Read bounded-panel outcome semantics without rewriting campaign state."""

    run_id = str(run.get("run_id") or "")
    raw_run_dir = str(run.get("run_dir") or "")
    if not run_id or not raw_run_dir:
        return {}
    panel_path = Path(raw_run_dir).parent.parent / "panel-status.json"
    try:
        stat = panel_path.stat()
    except OSError:
        return {}
    return dict(
        _cached_panel_experiment_outcome(
            str(panel_path),
            stat.st_mtime_ns,
            stat.st_size,
            run_id,
        )
    )


@lru_cache(maxsize=256)
def _cached_panel_experiment_outcome(
    panel_path: str,
    _mtime_ns: int,
    _size: int,
    run_id: str,
) -> dict[str, Any]:
    try:
        panel = json.loads(Path(panel_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(panel, Mapping) or panel.get("complete") is not True:
        return {}
    targets = panel.get("targets")
    if not isinstance(targets, Mapping):
        return {}
    target = next(
        (
            dict(value)
            for value in targets.values()
            if isinstance(value, Mapping) and str(value.get("run_id") or "") == run_id
        ),
        None,
    )
    if target is None:
        return {}
    final_state = dict(target.get("final_state") or {})
    claim = dict(final_state.get("claim") or {})
    if claim.get("benchmark_search_completed") is not True:
        return {}
    paper_equivalent = dict(target.get("paper_equivalent") or {})
    stop_decision = dict(final_state.get("stop_decision") or {})
    campaign_paused = str(target.get("status") or "").casefold() == "paused"
    return {
        "experiment_status": "complete",
        "paper_equivalent_status": (
            "solved"
            if paper_equivalent.get("paper_equivalent_solved") is True
            else "reached"
            if paper_equivalent.get("paper_reach") is True
            else "unresolved"
        ),
        "campaign_resumable": (campaign_paused and stop_decision.get("terminal") is not True),
        "scientific_status": str(target.get("scientific_status") or "unresolved"),
    }


def _run_activity_observation(
    run: Mapping[str, Any],
    status_result: Mapping[str, Any],
) -> tuple[str, bool]:
    """Read fixed durable progress markers without inventing another writer."""

    timestamps: list[float] = []
    updated_at = str(run.get("updated_at") or "")
    if updated_at:
        try:
            timestamps.append(datetime.fromisoformat(updated_at.replace("Z", "+00:00")).timestamp())
        except ValueError:
            pass
    raw_run_dir = str(status_result.get("run_dir") or run.get("run_dir") or "")
    if raw_run_dir:
        run_dir = Path(raw_run_dir)
        for marker in (
            run_dir / ".autoplanner" / "target-solver-checkpoint.json",
            run_dir / ".autoplanner" / "director-workspace" / "model-io.jsonl",
            run_dir / ".autoplanner" / "kernel" / "events.jsonl",
        ):
            try:
                timestamps.append(marker.stat().st_mtime)
            except OSError:
                continue
    if not timestamps:
        return "", True
    observed = max(timestamps)
    observed_at = datetime.fromtimestamp(observed, timezone.utc).isoformat().replace("+00:00", "Z")
    return observed_at, (time.time() - observed) > _RUN_ACTIVITY_STALE_AFTER_S


__all__ = [
    "historical_job",
    "job_projection",
    "live_job_progress",
    "new_run_id",
    "registry_job",
    "run_may_be_live",
    "run_target_job",
    "solve_target_request",
    "utc_now",
]
