"""Target-solve dependency assembly and background-job progress projections."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event, RLock
import time
from typing import Any, Callable, Mapping

from cascade_planner.interfaces.campaign_action_timeline import (
    compile_campaign_action_timeline,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
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
    with lock:
        if (
            cancel_event is not None
            and cancel_event.is_set()
        ) or str(jobs[job_id].get("status") or "") == "cancelling":
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
            result = (
                solve_target_request(
                    factory(),
                    request_payload,
                    cancel_event=cancel_event,
                )
                if cancel_event is not None
                else solve_target_request(factory(), request_payload)
            )
            compact = _compact_solve_result(result)
            if cancel_event is not None and cancel_event.is_set():
                raise TargetSolveCancelled("target_solve_cancelled_by_user")
            accepted = compact.get("accepted") is True
            objective_achieved = bool(
                compact.get("objective_achieved") is True or accepted
            )
            stop = dict(result.get("stop_decision") or {})
            should_continue = bool(
                not objective_achieved
                and _bool(payload, "auto_continue", True)
                and stop.get("decision") == "paused"
                and stop.get("terminal") is not True
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
        status, error = ("complete" if objective_achieved else "unresolved"), ""
        phase = "complete" if objective_achieved else "unresolved"
    except TargetSolveCancelled:
        status, phase, error = "cancelled", "cancelled", ""
    except Exception as exc:  # pragma: no cover - integration failure boundary
        if cancel_event is not None and cancel_event.is_set():
            status, phase, error = "cancelled", "cancelled", ""
        else:
            compact, status, phase = {}, "failed", "failed"
            error = f"{type(exc).__name__}: {exc}"[:4_000]
    elapsed = round(time.monotonic() - started, 3)
    with lock:
        terminal_fields = {
            "cancelled_at": utc_now(),
            "cancellation_reason": str(
                jobs[job_id].get("cancellation_reason") or "user_requested"
            ),
        } if status == "cancelled" else {}
        jobs[job_id].update(
            status=status,
            phase=phase,
            finished_at=utc_now(),
            updated_at=utc_now(),
            elapsed_s=elapsed,
            error=error,
            result=compact,
            continuation_pass_count=continuation_pass_count,
            **terminal_fields,
        )


def live_job_progress(factory: GatewayFactory, job: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(job.get("run_id") or "")
    result: dict[str, Any] = {
        "phase": str(job.get("phase") or job.get("status") or "unknown"),
        "elapsed_s": float(job.get("elapsed_s") or 0.0),
        "model_cost": {},
        "frontier_counts": {},
        "stages": [],
        "action_timeline": compile_campaign_action_timeline(()),
        "delivery": _delivery_projection([], job_status=str(job.get("status") or "")),
    }
    if job.get("status") in {"running", "cancelling"} and job.get("started_at"):
        try:
            started = datetime.fromisoformat(str(job["started_at"]).replace("Z", "+00:00"))
            result["elapsed_s"] = round(
                (datetime.now(timezone.utc) - started).total_seconds(), 3
            )
        except ValueError:
            pass
    try:
        status_result = factory().status(run_id)
    except Exception:
        return result
    status = dict(status_result.get("status") or {})
    portfolio = dict(status.get("portfolio") or {})
    frontier = [
        dict(value) for value in status.get("frontier") or [] if isinstance(value, dict)
    ]
    frontier_counts: dict[str, int] = {}
    for value in frontier:
        kind = str(value.get("kind") or "other")
        frontier_counts[kind] = frontier_counts.get(kind, 0) + 1
    result.update(
        campaign_status=str(status.get("status") or ""),
        graph_revision=int(status.get("graph_revision") or 0),
        evidence_revision=int(status.get("evidence_revision") or 0),
        attempt_count=int(status.get("attempt_count") or 0),
        accepted_expansion_count=int(status.get("accepted_expansion_count") or 0),
        model_cost=dict(status.get("model_totals") or {}),
        frontier_counts=frontier_counts,
        next_deficit_id=str(
            dict(status.get("stop_decision") or {}).get("next_deficit_id") or ""
        ),
        portfolio={
            "accepted": portfolio.get("accepted") is True,
            "route_count": len(portfolio.get("selected_routes") or []),
            "complete_route_count": int(
                dict(portfolio.get("closeout") or {}).get("complete_route_count") or 0
            ),
            "deficit_count": len(portfolio.get("deficits") or []),
        },
    )
    run_dir = Path(str(status_result.get("run_dir") or ""))
    checkpoint = run_dir / ".autoplanner" / "target-solver-checkpoint.json"
    timeline_stages: list[dict[str, Any]] = []
    if checkpoint.is_file():
        try:
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            timeline_stages = [
                dict(row)
                for row in value.get("stages") or []
                if isinstance(row, dict)
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
            for row in status.get("active_actions") or []
            if isinstance(row, Mapping)
        ),
    )
    job_status = str(job.get("status") or "")
    final_projection = job_status not in {"queued", "running", "cancelling"}
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
                achieved_profile=str(
                    workbench_portfolio.get("achieved_profile") or "unresolved"
                ),
                acceptance_profile_counts=profile_counts,
                condition_complete_route_count=int(
                    profile_counts.get("condition_complete") or 0
                ),
                process_ready_route_count=int(profile_counts.get("process_ready") or 0),
                closeout_reasons=list(
                    dict(workbench_portfolio.get("closeout") or {}).get("reasons")
                    or []
                ),
            )
        except Exception:
            result["proof_profile_known"] = False
    delivery = _delivery_projection(
        list(result.get("stages") or []),
        job_status=job_status,
    )
    if job_status == "historical":
        accepted = result.get("scientific_status") == "accepted"
        route_candidates = int(
            dict(result.get("portfolio") or {}).get("route_count") or 0
        ) > 0
        delivery.update(
            state="historical_accepted" if accepted else "historical_unresolved",
            route_candidates_available=route_candidates,
            workbench_available=route_candidates,
            proof_closure_known=True,
            proof_closure_complete=accepted,
        )
    delivery.update(
        scientific_status=str(result.get("scientific_status") or ""),
        achieved_profile=str(result.get("achieved_profile") or ""),
        condition_complete_route_count=int(
            result.get("condition_complete_route_count") or 0
        ),
        proof_profile_known=result.get("proof_profile_known") is True,
    )
    result["delivery"] = delivery
    return result


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
            "source_route_validated": int(
                source_validation.get("accepted_validation_count") or 0
            ),
            "prefetch_s": round(float(prefetch.get("elapsed_s") or 0.0), 3),
            "hidden_s": round(
                float(detail.get("latency_hidden_by_global_s") or 0.0), 3
            ),
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
            "complete_route_count": int(graph.get("complete_route_count") or 0),
            "deficit_count": sum(int(value or 0) for value in deficits.values()),
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
        "job_id": f"solve:{run_id}",
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
    }


__all__ = [
    "historical_job",
    "job_projection",
    "live_job_progress",
    "new_run_id",
    "run_target_job",
    "solve_target_request",
    "utc_now",
]
