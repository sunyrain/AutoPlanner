"""Target-solve dependency assembly and background-job progress projections."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any, Callable, Mapping
from urllib.parse import quote
from uuid import uuid4

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.interfaces.target_solver import (
    DEFAULT_TARGET_DIRECTOR_MODEL,
    TargetSolveConfig,
)


GatewayFactory = Callable[[], CampaignGateway]


def solve_target_request(gateway: Any, payload: dict[str, Any]) -> dict[str, Any]:
    max_visual_invocations = _int(payload, "max_visual_invocations", 0)
    evidence_connector = _web_evidence_connector(gateway, payload)
    visual_provider = _web_visual_provider(
        gateway, payload, enabled=max_visual_invocations > 0
    )
    return gateway.solve_target(
        target_name=str(payload.get("target_name") or "blind target"),
        target_smiles=str(payload.get("target_smiles") or ""),
        run_id=str(payload.get("run_id") or "") or None,
        resume=_bool(payload, "resume", False),
        evidence_connector=evidence_connector,
        visual_evidence_provider=visual_provider,
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=_int(payload, "minimum_complete_routes", 2),
            minimum_edge_proof_level=_int(payload, "minimum_edge_proof_level", 2),
            minimum_independent_source_groups=_int(
                payload, "minimum_source_groups", 2
            ),
            stock_boundary=str(payload.get("stock_boundary") or "benchmark_search"),
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=_int(payload, "max_model_invocations", 2),
            max_total_input_tokens=_int(payload, "max_input_tokens", 50_000),
            max_total_output_tokens=_int(payload, "max_output_tokens", 14_000),
            max_total_wall_time_s=float(payload.get("max_model_wall_time_s", 720.0)),
            max_visual_invocations=max_visual_invocations,
            max_accepted_expansions=_int(payload, "max_accepted_expansions", 32),
            max_attempt_runs=_int(payload, "max_attempt_runs", 72),
            max_prompt_context_bytes=96_000,
        ),
        config=TargetSolveConfig(
            model=str(payload.get("model") or DEFAULT_TARGET_DIRECTOR_MODEL),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            execution_profile=str(payload.get("execution_profile") or "fast"),
            use_coordinator=_bool(payload, "use_coordinator", False),
            enable_web_search=_bool(payload, "enable_web_search", True),
            enable_initial_director_web_search=_bool(
                payload, "enable_initial_director_web_search", False
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
                payload, "enable_target_chemenzy_baseline", False
            ),
            enable_guided_chemenzy=_bool(payload, "enable_guided_chemenzy", True),
            chemenzy_env_prefix=str(payload.get("chemenzy_env_prefix") or ""),
            self_evo_library_path=str(payload.get("self_evo_library_path") or ""),
            max_atom_mapping_reactions=_int(payload, "max_atom_mapping_reactions", 48),
            max_live_stock_molecules=_int(payload, "max_live_stock_molecules", 24),
            max_patent_sources=_int(payload, "max_patent_sources", 3),
            max_self_evo_template_candidates=_int(
                payload, "max_self_evo_template_candidates", 12
            ),
            max_chemenzy_routes=_int(payload, "max_chemenzy_routes", 2),
            max_chemenzy_steps=_int(payload, "max_chemenzy_steps", 6),
            max_chemenzy_iterations=_int(payload, "max_chemenzy_iterations", 10),
            chemenzy_expansion_topk=_int(payload, "chemenzy_expansion_topk", 20),
            chemenzy_timeout_s=float(payload.get("chemenzy_timeout_s", 90.0)),
            max_guided_chemenzy_frontiers=_int(
                payload, "max_guided_chemenzy_frontiers", 1
            ),
            max_guided_chemenzy_iterations=_int(
                payload, "max_guided_chemenzy_iterations", 4
            ),
            guided_chemenzy_timeout_s=float(
                payload.get("guided_chemenzy_timeout_s", 60.0)
            ),
            max_visual_evidence_pages=_int(payload, "max_visual_evidence_pages", 2),
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
                    seed_dois=tuple(_string_list(payload.get("literature_dois"))),
                    max_sources=_int(dict(payload), "max_literature_sources", 3),
                    max_visual_pages=_int(
                        dict(payload), "max_visual_evidence_pages", 2
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
            max_pages=_int(dict(payload), "max_visual_evidence_pages", 2),
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
    with lock:
        jobs[job_id].update(
            status="running",
            phase="initializing_campaign",
            started_at=utc_now(),
            updated_at=utc_now(),
        )
    try:
        result = solve_target_request(factory(), payload)
        compact = _compact_solve_result(result)
        status, error = "complete", ""
    except Exception as exc:  # pragma: no cover - integration failure boundary
        compact, status = {}, "failed"
        error = f"{type(exc).__name__}: {exc}"[:4_000]
    elapsed = round(time.monotonic() - started, 3)
    with lock:
        jobs[job_id].update(
            status=status,
            phase=status,
            finished_at=utc_now(),
            updated_at=utc_now(),
            elapsed_s=elapsed,
            error=error,
            result=compact,
        )


def live_job_progress(factory: GatewayFactory, job: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(job.get("run_id") or "")
    result: dict[str, Any] = {
        "phase": str(job.get("phase") or job.get("status") or "unknown"),
        "elapsed_s": float(job.get("elapsed_s") or 0.0),
        "model_cost": {},
        "frontier_counts": {},
        "stages": [],
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
    if checkpoint.is_file():
        try:
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            stages = [
                {
                    "stage": str(row.get("stage") or ""),
                    "status": str(row.get("status") or ""),
                    "elapsed_s": float(row.get("elapsed_s") or 0.0),
                    "started_at": str(row.get("started_at") or ""),
                    "completed_at": str(row.get("completed_at") or ""),
                    "metrics": _stage_progress_metrics(row),
                }
                for row in value.get("stages") or []
                if isinstance(row, dict)
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
    if stage == "evidence_acquisition":
        prefetch = dict(detail.get("prefetch") or {})
        return {
            "sources": int(detail.get("source_count") or 0),
            "exact_rows": int(detail.get("exact_record_count") or 0),
            "visual_calls": int(detail.get("visual_invocations") or 0),
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
    phase = str(run.get("status") or "historical")
    progress = {
        "phase": phase,
        "elapsed_s": float(costs.get("task_wall_time_s") or 0.0),
        "campaign_status": phase,
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
    }
    return {
        "job_id": f"solve:{run_id}",
        "run_id": run_id,
        "target_name": str(run.get("target_name") or run_id),
        "status": "historical",
        "phase": phase,
        "created_at": str(run.get("updated_at") or ""),
        "started_at": "",
        "finished_at": str(run.get("updated_at") or ""),
        "updated_at": str(run.get("updated_at") or ""),
        "elapsed_s": float(costs.get("task_wall_time_s") or 0.0),
        "error": "",
        "result": {},
        "progress": progress,
    }


def _compact_solve_result(value: Mapping[str, Any]) -> dict[str, Any]:
    gates = dict(value.get("gates") or {})
    return {
        "run_id": str(value.get("run_id") or ""),
        "report_path": str(value.get("report_path") or ""),
        "accepted": dict(value.get("claim") or {}).get(
            "accepted_under_configured_policy"
        )
        is True,
        "highest_contiguous_gate": str(gates.get("highest_contiguous_gate") or "none"),
        "gates": dict(gates.get("gates") or {}),
        "counts": dict(gates.get("counts") or {}),
        "model_cost": dict(value.get("model_cost") or {}),
        "workbench_url": (
            "/api/v4/runs/"
            + quote(str(value.get("run_id") or ""), safe="")
            + "/workbench.html"
        ),
    }


def job_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "job_id",
            "run_id",
            "target_name",
            "status",
            "phase",
            "created_at",
            "started_at",
            "finished_at",
            "updated_at",
            "elapsed_s",
            "error",
            "result",
        )
    }


def new_run_id(target_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", target_name.lower()).strip("-") or "target"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"v4-{slug[:28]}-{stamp}-{uuid4().hex[:6]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("expected_string_or_list")


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
