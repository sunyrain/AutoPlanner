"""Target-solve dependency assembly and background-job progress projections."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.unified_campaign_spec import TargetConstraints
from cascade_planner.interfaces.campaign_action_timeline import (
    compile_campaign_action_timeline,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.interfaces.target_solver import (
    DEFAULT_TARGET_DIRECTOR_MODEL,
    TargetSolveConfig,
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
from cascade_planner.interfaces.target_runtime_dependencies import (
    TARGET_PROFILE_DEFAULTS,
    inventory_snapshot_builder,
)


GatewayFactory = Callable[[], CampaignGateway]


def solve_target_request(gateway: Any, payload: dict[str, Any]) -> dict[str, Any]:
    max_visual_invocations = _int(payload, "max_visual_invocations", 0)
    execution_profile = str(payload.get("execution_profile") or "standard")
    profile_defaults = TARGET_PROFILE_DEFAULTS.get(
        execution_profile, TARGET_PROFILE_DEFAULTS["standard"]
    )
    evidence_connector = _web_evidence_connector(gateway, payload)
    visual_provider = _web_visual_provider(
        gateway, payload, enabled=max_visual_invocations > 0
    )
    inventory_builder = inventory_snapshot_builder(payload)
    return gateway.solve_target(
        target_name=str(payload.get("target_name") or "blind target"),
        target_smiles=str(payload.get("target_smiles") or ""),
        run_id=str(payload.get("run_id") or "") or None,
        resume=_bool(payload, "resume", False),
        evidence_connector=evidence_connector,
        visual_evidence_provider=visual_provider,
        inventory_snapshot_builder=inventory_builder,
        constraints=_target_constraints(payload),
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=_int(payload, "minimum_complete_routes", 2),
            minimum_edge_proof_level=_int(payload, "minimum_edge_proof_level", 2),
            minimum_independent_source_groups=_int(
                payload, "minimum_source_groups", 2
            ),
            stock_boundary=str(payload.get("stock_boundary") or "benchmark_search"),
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=_int(
                payload,
                "max_model_invocations",
                3 if max_visual_invocations else 2,
            ),
            max_total_input_tokens=_int(
                payload, "max_input_tokens", profile_defaults["max_input_tokens"]
            ),
            max_total_output_tokens=_int(
                payload, "max_output_tokens", profile_defaults["max_output_tokens"]
            ),
            max_total_wall_time_s=float(
                payload.get(
                    "max_model_wall_time_s",
                    profile_defaults["max_model_wall_time_s"],
                )
            ),
            max_visual_invocations=max_visual_invocations,
            max_accepted_expansions=_int(payload, "max_accepted_expansions", 64),
            max_attempt_runs=_int(payload, "max_attempt_runs", 128),
            max_prompt_context_bytes=_int(
                payload, "max_prompt_context_bytes", 160_000
            ),
        ),
        config=TargetSolveConfig(
            model=str(payload.get("model") or DEFAULT_TARGET_DIRECTOR_MODEL),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            execution_profile=execution_profile,
            objective_mode=str(
                payload.get("objective_mode") or "scientific_proof"
            ),
            use_coordinator=_bool(payload, "use_coordinator", False),
            enable_web_search=_bool(payload, "enable_web_search", True),
            enable_initial_director_web_search=_bool(
                payload,
                "enable_initial_director_web_search",
                execution_profile in {"standard", "proof"},
            ),
            enable_target_identity=_bool(payload, "enable_target_identity", True),
            resolve_named_target_identity=_bool(
                payload, "resolve_named_target_identity", True
            ),
            blind_audit_root=str(payload.get("blind_audit_root") or ""),
            enable_replan=_bool(payload, "enable_replan", True),
            enable_live_benchmark_stock=_bool(
                payload, "enable_live_benchmark_stock", True
            ),
            enable_builtin_patent_evidence=(
                evidence_connector is None
                and _bool(payload, "enable_auto_patent_evidence", True)
            ),
            enable_patent_self_evolution=_bool(
                payload, "enable_patent_self_evolution", True
            ),
            enable_chemenzy=_bool(payload, "enable_chemenzy", True),
            enable_target_chemenzy_baseline=_bool(
                payload, "enable_target_chemenzy_baseline", True
            ),
            enable_guided_chemenzy=_bool(payload, "enable_guided_chemenzy", True),
            enable_chemenzy_condition_prediction=_bool(
                payload, "enable_chemenzy_condition_prediction", True
            ),
            enable_chemenzy_enzyme_assignment=_bool(
                payload, "enable_chemenzy_enzyme_assignment", True
            ),
            enable_enzyme_coverage_sidecar=_bool(
                payload, "enable_enzyme_coverage_sidecar", True
            ),
            chemenzy_env_prefix=str(payload.get("chemenzy_env_prefix") or ""),
            self_evo_library_path=str(payload.get("self_evo_library_path") or ""),
            max_atom_mapping_reactions=_int(payload, "max_atom_mapping_reactions", 48),
            max_live_stock_molecules=_int(payload, "max_live_stock_molecules", 24),
            max_patent_sources=_int(payload, "max_patent_sources", 3),
            max_self_evo_template_candidates=_int(
                payload, "max_self_evo_template_candidates", 12
            ),
            max_total_tasks=_int(payload, "max_total_tasks", 256),
            max_evidence_tasks=_int(payload, "max_evidence_tasks", 64),
            max_stock_tasks=_int(payload, "max_stock_tasks", 128),
            max_validation_tasks=_int(payload, "max_validation_tasks", 128),
            max_program_tasks=_int(payload, "max_program_tasks", 64),
            max_experiment_tasks=_int(payload, "max_experiment_tasks", 32),
            max_run_wall_time_s=float(
                payload.get("max_run_wall_time_s", 7_200.0)
            ),
            provider_route_reserve=_int(
                payload, "provider_route_reserve", 16
            ),
            host_route_portfolio=_int(
                payload, "host_route_portfolio", 8
            ),
            display_route_limit=_int(payload, "display_route_limit", 4),
            max_chemenzy_routes=(
                int(payload["max_chemenzy_routes"])
                if payload.get("max_chemenzy_routes") is not None
                else None
            ),
            max_chemenzy_steps=_int(
                payload, "max_chemenzy_steps", profile_defaults["steps"]
            ),
            max_chemenzy_iterations=_int(
                payload, "max_chemenzy_iterations", profile_defaults["iterations"]
            ),
            chemenzy_expansion_topk=_int(
                payload, "chemenzy_expansion_topk", profile_defaults["topk"]
            ),
            chemenzy_timeout_s=float(
                payload.get("chemenzy_timeout_s", profile_defaults["timeout"])
            ),
            chemenzy_search_preset=str(
                payload.get("chemenzy_search_preset")
                or ("thorough" if execution_profile == "proof" else "standard")
            ),
            chemenzy_pandarallel_workers=_int(
                payload,
                "chemenzy_pandarallel_workers",
                profile_defaults["workers"],
            ),
            max_guided_chemenzy_frontiers=_int(
                payload, "max_guided_chemenzy_frontiers", 3
            ),
            max_guided_chemenzy_iterations=_int(
                payload, "max_guided_chemenzy_iterations", 6
            ),
            guided_chemenzy_timeout_s=float(
                payload.get("guided_chemenzy_timeout_s", 60.0)
            ),
            max_visual_evidence_pages=_int(payload, "max_visual_evidence_pages", 6),
            minimum_planning_route_steps=_int(
                payload, "minimum_planning_route_steps", 0
            ),
            max_director_output_tokens=_int(
                payload,
                "max_director_output_tokens",
                18_000 if execution_profile == "proof" else 7_000,
            ),
            max_director_wall_time_s=float(
                payload.get(
                    "max_director_wall_time_s",
                    profile_defaults["max_director_wall_time_s"],
                )
            ),
        ),
    )


def _web_evidence_connector(gateway: Any, payload: Mapping[str, Any]) -> Any:
    paths = getattr(gateway, "paths", None)
    if paths is None:
        return None
    connectors = []
    if _bool(dict(payload), "enable_auto_patent_evidence", True):
        from cascade_planner.interfaces.patent_evidence import (
            BuiltinPatentEvidenceConfig,
            build_builtin_patent_evidence_connector,
        )

        connectors.append(
            build_builtin_patent_evidence_connector(
                BuiltinPatentEvidenceConfig(
                    cache_dir=paths.external_data_root / "patent-evidence",
                    seed_publications=tuple(
                        _string_list(payload.get("patent_publications"))
                    ),
                    max_patents=_int(dict(payload), "max_patent_sources", 3),
                    max_validated_edges=_int(
                        dict(payload), "max_atom_mapping_reactions", 48
                    ),
                )
            )
        )
    if _bool(dict(payload), "enable_auto_literature_evidence", True):
        from cascade_planner.interfaces.literature_evidence import (
            BuiltinLiteratureEvidenceConfig,
            build_builtin_literature_evidence_connector,
        )

        connectors.append(
            build_builtin_literature_evidence_connector(
                BuiltinLiteratureEvidenceConfig(
                    cache_dir=paths.external_data_root / "literature-evidence",
                    authorized_proxy_output_dir=str(
                        payload.get("authorized_proxy_output_dir") or ""
                    ),
                    seed_dois=tuple(_string_list(payload.get("literature_dois"))),
                    max_sources=_int(dict(payload), "max_literature_sources", 4),
                    max_visual_pages=_int(
                        dict(payload), "max_visual_evidence_pages", 6
                    ),
                    auto_fetch_restricted_sources=_bool(
                        dict(payload),
                        "auto_fetch_restricted_literature",
                        True,
                    ),
                    auto_fetch_timeout_s=float(
                        _int(dict(payload), "literature_browser_timeout_s", 180)
                    ),
                    auto_fetch_max_items=_int(
                        dict(payload),
                        "max_literature_sources",
                        4,
                    ),
                )
            )
        )
    if len(connectors) == 1:
        return connectors[0]
    if connectors:
        from cascade_planner.interfaces.live_evidence import compose_evidence_connectors

        return compose_evidence_connectors(*connectors)
    return None


def _web_visual_provider(
    gateway: Any, payload: Mapping[str, Any], *, enabled: bool
) -> Any:
    paths = getattr(gateway, "paths", None)
    if not enabled or paths is None:
        return None
    from cascade_planner.interfaces.visual_evidence import (
        CodexVisualEvidenceConfig,
        build_codex_visual_evidence_provider,
    )

    return build_codex_visual_evidence_provider(
        CodexVisualEvidenceConfig(
            cache_dir=paths.external_data_root / "visual-evidence",
            model=str(payload.get("model") or DEFAULT_TARGET_DIRECTOR_MODEL),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            max_pages=_int(dict(payload), "max_visual_evidence_pages", 6),
        )
    )


def run_target_job(
    factory: GatewayFactory,
    payload: dict[str, Any],
    job_id: str,
    jobs: dict[str, dict[str, Any]],
    lock: RLock,
) -> None:
    started = time.monotonic()
    continuation_pass_count = 0
    with lock:
        jobs[job_id].update(
            status="running",
            phase="initializing_campaign",
            started_at=utc_now(),
            updated_at=utc_now(),
        )
    try:
        request_payload = dict(payload)
        while True:
            result = solve_target_request(factory(), request_payload)
            compact = _compact_solve_result(result)
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
    except Exception as exc:  # pragma: no cover - integration failure boundary
        compact, status, phase = {}, "failed", "failed"
        error = f"{type(exc).__name__}: {exc}"[:4_000]
    elapsed = round(time.monotonic() - started, 3)
    with lock:
        jobs[job_id].update(
            status=status,
            phase=phase,
            finished_at=utc_now(),
            updated_at=utc_now(),
            elapsed_s=elapsed,
            error=error,
            result=compact,
            continuation_pass_count=continuation_pass_count,
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
    if job.get("status") == "running" and job.get("started_at"):
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
    final_projection = job_status not in {"queued", "running"}
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
    if stage in {"chemenzy_guided_frontier", "chemenzy_stock_recovery"}:
        return {
            "frontiers": int(detail.get("frontier_count") or 0),
            "provider_calls": int(
                detail.get("provider_invocation_count")
                or detail.get("executed_frontier_count")
                or 0
            ),
            "proposals": int(detail.get("proposal_count") or 0),
        }
    if stage == "chemenzy_delegation":
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


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("expected_string_or_list")


def _target_constraints(payload: Mapping[str, Any]) -> TargetConstraints:
    safety = payload.get("safety_limits") or {}
    if not isinstance(safety, Mapping):
        raise ValueError("safety_limits_must_be_an_object")
    max_route_steps = payload.get("max_route_steps")
    return TargetConstraints(
        forbidden_reagents=tuple(_string_list(payload.get("forbidden_reagents"))),
        max_route_steps=(
            None
            if max_route_steps in {None, ""}
            else _int(payload, "max_route_steps", 0)
        ),
        allowed_execution_domains=tuple(
            _string_list(payload.get("allowed_execution_domains"))
            or TargetConstraints().allowed_execution_domains
        ),
        safety_limits=dict(safety),
        stock_source_ids=tuple(_string_list(payload.get("stock_source_ids"))),
    )


def _int(value: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(value.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}_must_be_an_integer") from exc


def _bool(value: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = value.get(key, default)
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"{key}_must_be_a_boolean")


__all__ = [
    "historical_job",
    "job_projection",
    "live_job_progress",
    "new_run_id",
    "run_target_job",
    "solve_target_request",
    "utc_now",
]
