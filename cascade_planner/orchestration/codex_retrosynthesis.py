"""Run a Codex coordinator that directly delegates to specialist child agents."""
from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Iterator

from rdkit import Chem

from cascade_planner.application.frontier_scheduler import (
    FrontierJob,
    FrontierJobState,
    FrontierLeaseError,
    FrontierQueueError,
    FrontierScheduler,
    PersistentFrontierQueue,
    assess_frontier_completeness,
)
from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.agent.codex_worker import WorkerBudget, WorkerRunRecord, WorkerTask, run_codex_worker
from cascade_planner.routes.consensus import (
    consensus_to_blackboard_proposals,
    fuse_route_candidates,
    validate_retrosynthesis_report_payload,
)
from cascade_planner.routes.graph import (
    assemble_route_consensus_graph,
    make_route_consensus_expansion,
    select_route_consensus_frontier,
)
from cascade_planner.providers.contracts import StockProvider
from cascade_planner.providers.stock import SnapshotStockProvider
from cascade_planner.runtime import Budget as RuntimeBudget
from cascade_planner.runtime import CodexTeamRuntimeTracker


CODEX_RETROSYNTHESIS_TEAM_SCHEMA = "codex_retrosynthesis_team_run.v1"
CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA = "codex_retrosynthesis_campaign.v1"
CODEX_RETROSYNTHESIS_EXPANSION_COMMIT_SCHEMA = (
    "codex_retrosynthesis_expansion_commit.v1"
)
DEFAULT_CHILD_ROLES = (
    "target_structure_strategist",
    "literature_route_scout",
    "chemoenzymatic_route_specialist",
    "route_evidence_critic",
)
ROLE_SOURCE_CHANNELS = {
    "target_structure_strategist": "codex_strategy",
    "literature_route_scout": "codex_literature",
    "chemoenzymatic_route_specialist": "codex_chemoenzymatic",
    "route_evidence_critic": "codex_critic",
}
MAX_CHILD_REPORT_BYTES = 250_000
CHILD_REPORT_KEYS = {
    "schema_version",
    "case_id",
    "agent_role",
    "target_smiles",
    "candidates",
    "evidence_refs",
    "limitations",
    "no_solved_claim",
}
CHILD_CANDIDATE_KEYS = {
    "schema_version",
    "candidate_id",
    "product_smiles",
    "precursor_smiles",
    "reaction_family",
    "transformation_rationale",
    "source_channel",
    "source_refs",
    "evidence_refs",
    "evidence_level",
    "confidence",
    "conditions",
    "catalyst",
    "enzyme",
    "limitations",
    "required_validation",
    "no_solved_claim",
    "not_parent_route_proof",
}

TeamRunner = Callable[[WorkerTask], WorkerRunRecord]


@dataclass
class RetrosynthesisTeamConfig:
    child_roles: list[str] = field(default_factory=lambda: list(DEFAULT_CHILD_ROLES))
    timeout_s: float = 900.0
    max_output_bytes: int = 600_000
    max_tool_calls: int = 32
    reasoning_effort: str = "high"
    web_search: bool = True
    auth_mode: str = "auto"
    model: str = ""
    max_depth: int = 2
    max_expansions: int = 4
    frontier_batch_size: int = 2
    frontier_lease_seconds: float = 1800.0
    frontier_heartbeat_interval_seconds: float = 0.0
    frontier_retry_base_seconds: float = 1.0
    frontier_retry_max_seconds: float = 60.0
    frontier_retry_wait_seconds: float = 5.0
    stock_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_retrosynthesis_coordinator_task(
    *,
    case_id: str,
    target_name: str,
    target_smiles: str,
    context_ref: str,
    allowed_workdir: str | Path,
    config: RetrosynthesisTeamConfig | None = None,
) -> WorkerTask:
    config = config or RetrosynthesisTeamConfig()
    roles = _normalized_roles(config.child_roles)
    # Codex CLI 0.142.x reports the multi-agent wait tool as ``wait``.
    # The worker validator also accepts the older ``wait_agent`` alias.
    # Codex reads the bounded context snapshot through a read-only shell call;
    # the worker itself runs under the read-only Codex sandbox by default.
    allowed_tools = ["spawn_agent", "wait", "send_message", "shell"]
    if config.web_search:
        allowed_tools.extend(["web_search", "browser", "literature_search"])
    role_lines = "\n".join(f"- {role}" for role in roles)
    child_report_contract = {
        "schema_version": "retrosynthesis_proposal_report.v1",
        "case_id": case_id,
        "agent_role": "<exact assigned role>",
        "target_smiles": target_smiles,
        "candidates": [
            {
                "schema_version": "retrosynthesis_candidate.v1",
                "candidate_id": "<stable child-local id>",
                "product_smiles": target_smiles,
                "precursor_smiles": ["<one canonical SMILES per component>"],
                "reaction_family": "<concise family>",
                "transformation_rationale": "<bounded rationale>",
                "source_channel": "other",
                "source_refs": [],
                "evidence_refs": [],
                "evidence_level": "model_only",
                "confidence": "low",
                "conditions": [],
                "catalyst": "",
                "enzyme": "",
                "limitations": [],
                "required_validation": [],
                "no_solved_claim": True,
                "not_parent_route_proof": True,
            }
        ],
        "evidence_refs": [],
        "limitations": [],
        "no_solved_claim": True,
    }
    objective = f"""Coordinate an independent retrosynthesis review for {target_name}.

Target SMILES: {target_smiles}
Context snapshot: {context_ref}

You must directly spawn one child agent for every role below:
{role_lines}

Every child spawn prompt must contain the exact machine-readable marker
AUTOPLANNER_CHILD_ROLE=<assigned_role> for that role.

Every child spawn prompt must also state these strict JSON type rules: no field
may be null; confidence, catalyst, enzyme, evidence_level, source_channel, and
all other scalar candidate fields are strings; precursor_smiles, source_refs,
evidence_refs, conditions, limitations, and required_validation are arrays of
strings; no_solved_claim and not_parent_route_proof are literal true. Use
confidence="low", catalyst="", enzyme="", and conditions=[] when those values
are unknown. Returning candidates=[] is always preferable to violating this
contract.

Give each child only the target, the context reference, its role, the shared
RetrosynthesisProposalReport candidate contract below, and a bounded task.
Each child must return exactly one JSON object matching this payload contract,
with agent_role equal to its assigned role; no markdown or prose may surround
the JSON. A child with no defensible proposal must return candidates=[] rather
than invent one:
{json.dumps(child_report_contract, ensure_ascii=False, indent=2, sort_keys=True)}

Require
the literature child to attach traceable DOI/URL/local source references; do
not let a model-only agreement masquerade as independent evidence. Require the
critic to report identity, stereo, feasibility, evidence, and route-connection
problems. After all children finish, preserve the child JSON messages and synthesize one draft
RetrosynthesisProposalReport. Preserve disagreements and rejected hypotheses
in limitations. Do not emit reaction SMILES, mutate a route tree, or claim the
target solved. Candidate product_smiles must equal the requested target; use
precursor_smiles as a list of individual components.
"""
    return WorkerTask(
        task_id=f"{case_id}:retrosynthesis_team",
        case_id=case_id,
        task_type="target_research",
        required_artifact_type="RetrosynthesisProposalReport",
        input_refs=[str(context_ref)],
        allowed_tools=allowed_tools,
        budget=WorkerBudget(
            timeout_s=max(30.0, float(config.timeout_s)),
            max_output_bytes=max(20_000, int(config.max_output_bytes)),
            max_tool_calls=max(len(roles) * 3, int(config.max_tool_calls)),
            max_worker_runs=1,
            reasoning_effort=str(config.reasoning_effort or "high"),
        ),
        objective=objective,
        allowed_workdir=str(Path(allowed_workdir).resolve()),
        agent_mode="coordinator",
        child_roles=roles,
        codex_auth_mode=str(config.auth_mode or "auto"),
        model=str(config.model or ""),
    )


def run_codex_retrosynthesis_team(
    *,
    case_id: str,
    target_name: str,
    target_smiles: str,
    run_dir: str | Path,
    repository_root: str | Path,
    blackboard_context: dict[str, Any] | None = None,
    literature_sources: list[dict[str, Any]] | None = None,
    config: RetrosynthesisTeamConfig | None = None,
    runner: TeamRunner | None = None,
) -> dict[str, Any]:
    """Run the coordinator, validate its artifact, and fuse child proposals.

    There is deliberately no deterministic proposal fallback. If Codex or a
    required child fails, this report is rejected and the caller can retry or
    stop unresolved while deterministic safety/proof logic remains available.
    """
    output_dir = Path(run_dir).resolve() / "codex_retrosynthesis_team"
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / "context_snapshot.json"
    context = {
        "schema_version": "retrosynthesis_team_context.v1",
        "case_id": case_id,
        "target": {"name": target_name, "smiles": target_smiles},
        "repository_root": str(Path(repository_root).resolve()),
        "blackboard": _compact_blackboard_context(blackboard_context or {}),
        "literature_sources": [dict(row) for row in literature_sources or [] if isinstance(row, dict)][:24],
        "semantics": {
            "child_outputs_are_draft": True,
            "deterministic_parent_proof_required": True,
            "no_solved_claim": True,
        },
    }
    _write_json(context_path, context)
    task = build_retrosynthesis_coordinator_task(
        case_id=case_id,
        target_name=target_name,
        target_smiles=target_smiles,
        context_ref=str(context_path),
        # Keep Codex runtime traces and any bounded scratch work inside this
        # run. The repository remains discoverable as the Git ancestor.
        allowed_workdir=output_dir,
        config=config,
    )
    _write_json(output_dir / "coordinator_task.json", task.to_dict())

    runtime_tracker = CodexTeamRuntimeTracker.start(
        root=output_dir / "runtime",
        run_id=f"{case_id}:retrosynthesis_team",
        coordinator_agent_id=task.task_id,
        child_roles=task.child_roles,
        objective=task.objective,
        context=context,
        context_ref=str(context_path),
        capabilities=task.allowed_tools,
        budget=RuntimeBudget(
            max_wall_time_s=task.budget.timeout_s,
            max_tool_calls=task.budget.max_tool_calls,
            max_output_bytes=task.budget.max_output_bytes,
            max_children=len(task.child_roles),
        ),
    )
    runtime_summary_path = output_dir / "runtime_summary.json"
    try:
        record = runner(task) if runner is not None else run_codex_worker(task, use_codex_cli=True)
    except Exception as exc:
        _write_json(runtime_summary_path, runtime_tracker.fail(exc))
        raise
    child_reports, child_candidates = _validated_child_reports(
        record=record,
        required_roles=task.child_roles,
        case_id=case_id,
        target_smiles=target_smiles,
        output_dir=output_dir,
    )
    _annotate_child_report_validation(record, child_reports)
    record_payload = record.to_dict()
    record_path = output_dir / "coordinator_run_record.json"
    _write_json(record_path, record_payload)
    runtime_summary = runtime_tracker.complete(record, artifacts=(str(record_path),))
    _write_json(runtime_summary_path, runtime_summary)

    artifact = dict(record.output_artifact or {})
    artifact_validation = validate_typed_artifact(artifact) if artifact else {
        "schema_version": "artifact_validator.v1",
        "accepted": False,
        "reasons": ["missing_coordinator_output_artifact"],
    }
    payload = dict(artifact.get("payload") or {})
    coordinator_candidates = [dict(row) for row in payload.get("candidates") or [] if isinstance(row, dict)]
    # Build consensus from independently observed child final messages, not
    # from the coordinator's restatement. The latter remains an audit-only
    # synthesis artifact and cannot invent a missing specialist.
    consensus = fuse_route_candidates(child_candidates, case_id=case_id, target_smiles=target_smiles)
    consensus_path = output_dir / "route_consensus.json"
    _write_json(consensus_path, consensus)
    proposals = consensus_to_blackboard_proposals(consensus)

    required_children = len(task.child_roles)
    observed_children = len((record.metadata or {}).get("child_agents") or [])
    reasons: list[str] = []
    if record.status != "accepted_draft":
        reasons.append(f"coordinator_status:{record.status}")
    if not artifact_validation.get("accepted"):
        reasons.extend(str(item) for item in artifact_validation.get("reasons") or [])
    if observed_children != required_children:
        reasons.append("required_child_agents_not_observed")
    valid_child_roles = {
        str(row.get("role") or "")
        for row in child_reports
        if row.get("accepted") is True
    }
    if valid_child_roles != set(task.child_roles):
        reasons.append("required_child_reports_not_valid")
    if any(row.get("accepted") is not True for row in child_reports):
        reasons.append("one_or_more_child_reports_rejected")
    runtime_children = [dict(row) for row in runtime_summary.get("children") or [] if isinstance(row, dict)]
    if len(runtime_children) < required_children or any(
        str(row.get("state") or "") != "succeeded" for row in runtime_children
    ):
        reasons.append("required_child_agents_not_succeeded")
    if not consensus.get("accepted"):
        reasons.append("no_valid_retrosynthesis_candidates")
    if not runtime_summary.get("consistent"):
        reasons.append("child_agent_runtime_reconciliation_failed")
    accepted = not reasons
    report = {
        "schema_version": CODEX_RETROSYNTHESIS_TEAM_SCHEMA,
        "accepted": accepted,
        "case_id": case_id,
        "target_name": target_name,
        "target_smiles": target_smiles,
        "coordinator": {
            "task_ref": str(output_dir / "coordinator_task.json"),
            "run_record_ref": str(record_path),
            "status": record.status,
            "backend": record.backend,
            "session_id": str((record.metadata or {}).get("session_id") or ""),
            "required_child_roles": list(task.child_roles),
            "observed_child_agents": list((record.metadata or {}).get("child_agents") or []),
            "event_summary": dict((record.metadata or {}).get("event_summary") or {}),
            "usage": dict(record.usage or {}),
            "coordinator_candidate_count": len(coordinator_candidates),
            "child_report_normalization_repair_count": sum(
                len(row.get("normalization_repairs") or []) for row in child_reports
            ),
        },
        "child_reports": child_reports,
        "artifact_validation": artifact_validation,
        "route_consensus_ref": str(consensus_path),
        "route_consensus": consensus,
        "runtime_summary_ref": str(runtime_summary_path),
        "runtime_summary": runtime_summary,
        "blackboard_proposals": proposals,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "codex_child_agents_required": True,
            "deterministic_scientific_fallback_used": False,
            "deterministic_parent_proof_required": True,
            "child_shape_repairs_are_conservative_defaults_only": True,
            "no_solved_claim": True,
        },
    }
    _write_json(output_dir / "team_report.json", report)
    return report


def run_codex_retrosynthesis_campaign(
    *,
    case_id: str,
    target_name: str,
    target_smiles: str,
    run_dir: str | Path,
    repository_root: str | Path,
    blackboard_context: dict[str, Any] | None = None,
    literature_sources: list[dict[str, Any]] | None = None,
    config: RetrosynthesisTeamConfig | None = None,
    runner: TeamRunner | None = None,
    stock_provider: StockProvider | None = None,
) -> dict[str, Any]:
    """Recursively run direct Codex teams through a durable frontier queue.

    Team output remains advisory. A successfully expanded proposal frontier is
    recorded separately from stock/reaction closure, so exhausting the model
    work budget cannot be mislabeled as a complete synthesis route.
    """
    config = config or RetrosynthesisTeamConfig()
    max_depth = max(1, int(config.max_depth or 1))
    max_expansions = max(1, int(config.max_expansions or 1))
    frontier_batch_size = max(1, int(config.frontier_batch_size or 1))
    root_run_dir = Path(run_dir).resolve()
    root_output_dir = root_run_dir / "codex_retrosynthesis_team"
    root_output_dir.mkdir(parents=True, exist_ok=True)
    queue_store = PersistentFrontierQueue(root_output_dir / "frontier_queue")
    scheduler = FrontierScheduler(queue_store, stock_provider or SnapshotStockProvider())
    campaign_state_path = root_output_dir / "campaign_state.json"
    restored = _load_campaign_state(
        campaign_state_path,
        case_id=case_id,
        target_smiles=target_smiles,
    )
    expansions = [dict(row) for row in restored.get("expansions") or []]
    run_summaries = [dict(row) for row in restored.get("runs") or []]
    root_report = _read_json_object(root_output_dir / "team_report.json")
    graph = assemble_route_consensus_graph(
        expansions,
        case_id=case_id,
        target_smiles=target_smiles,
        max_depth=max_depth,
    )
    root_smiles = _canonical_target_smiles(target_smiles)
    scheduler.submit(
        run_id=case_id,
        case_id=case_id,
        frontier_smiles=root_smiles,
        frontier_node_id=str(graph.get("root_node_id") or f"root:{root_smiles}"),
        idempotency_key=f"{case_id}:frontier:{hashlib.sha256(root_smiles.encode()).hexdigest()}",
        stock_request=_stock_request(config, root_smiles),
        required_proof_level=2,
        proof_deficit=2,
        closure_probability=1.0,
        diversity_gain=1.0,
        max_attempts=3,
        metadata={
            "target_name": target_name,
            "depth": 0,
            "parent_step_ids": [],
        },
    )
    prior_expansion_count = len(expansions)
    prior_run_count = len(run_summaries)
    recovery_errors = _reconcile_expansion_commits(
        queue=queue_store,
        run_id=case_id,
        root_output_dir=root_output_dir,
        expansions=expansions,
        runs=run_summaries,
    )
    expanded_smiles = {
        _canonical_target_smiles(row.get("target_smiles"))
        for row in run_summaries
        if row.get("proposal_expansion_recorded") is True
    }
    expanded_smiles.discard("")
    graph = assemble_route_consensus_graph(
        expansions,
        case_id=case_id,
        target_smiles=target_smiles,
        max_depth=max_depth,
    )
    _submit_graph_frontiers(
        graph=graph,
        scheduler=scheduler,
        config=config,
        case_id=case_id,
        expanded_smiles=expanded_smiles,
        max_depth=max_depth,
        max_expansions=max_expansions,
        frontier_batch_size=frontier_batch_size,
    )
    if len(expansions) != prior_expansion_count or len(run_summaries) != prior_run_count:
        _write_campaign_state(
            campaign_state_path,
            case_id=case_id,
            target_smiles=target_smiles,
            expansions=expansions,
            runs=run_summaries,
        )

    while len(run_summaries) < max_expansions:
        lease_seconds = max(30.0, float(config.frontier_lease_seconds or 1800.0))
        claimed = queue_store.claim(
            case_id,
            worker_id=f"codex-campaign:{case_id}",
            limit=1,
            lease_seconds=lease_seconds,
        )
        if not claimed:
            retry_delay = _next_retry_delay(queue_store.list_jobs(case_id))
            retry_wait_limit = max(0.0, float(config.frontier_retry_wait_seconds or 0.0))
            if (
                retry_delay is not None
                and retry_delay <= retry_wait_limit
                and len(run_summaries) < max_expansions
            ):
                time.sleep(max(0.01, retry_delay))
                continue
            break
        job = claimed[0]
        frontier = dict(job.metadata)
        frontier_smiles = job.frontier_smiles
        depth = int(frontier.get("depth") or 0)
        if frontier_smiles in expanded_smiles or depth >= max_depth:
            queue_store.complete(
                case_id,
                job.job_id,
                lease_token=job.lease_token,
                result_ref=f"frontier-skip:{frontier_smiles}",
                closure_kind="proposal_expansion",
                achieved_proof_level=0,
            )
            continue
        digest = hashlib.sha256(frontier_smiles.encode("utf-8")).hexdigest()[:12]
        expansion_case_id = case_id if depth == 0 else f"{case_id}:frontier:d{depth}:{digest}"
        expansion_run_dir = (
            root_run_dir
            if depth == 0
            else root_run_dir / "codex_retrosynthesis_frontiers" / f"d{depth}-{digest}"
        )
        expansion_context = dict(blackboard_context or {})
        if expansions:
            expansion_context["route_consensus_graph"] = graph
            expansion_context["frontier_request"] = dict(frontier)
        team_report: dict[str, Any] = {}
        team_error: Exception | None = None
        with _frontier_lease_heartbeat(
            queue_store,
            run_id=case_id,
            job=job,
            lease_seconds=lease_seconds,
            interval_seconds=float(config.frontier_heartbeat_interval_seconds or 0.0),
        ) as heartbeat_errors:
            try:
                team_report = run_codex_retrosynthesis_team(
                    case_id=expansion_case_id,
                    target_name=str(frontier.get("target_name") or frontier_smiles),
                    target_smiles=frontier_smiles,
                    run_dir=expansion_run_dir,
                    repository_root=repository_root,
                    blackboard_context=expansion_context,
                    literature_sources=literature_sources,
                    config=config,
                    runner=runner,
                )
            except Exception as exc:  # noqa: BLE001 - persisted below
                team_error = exc
        heartbeat_failures = list(heartbeat_errors)
        if team_error is not None:
            failure_reason = (
                f"team_runtime_error:{type(team_error).__name__}:{team_error}"
            )
            current = queue_store.get(case_id, job.job_id)
            if not heartbeat_failures:
                try:
                    current = queue_store.fail(
                        case_id,
                        job.job_id,
                        lease_token=job.lease_token,
                        reason=failure_reason,
                        retryable=True,
                        retry_base_seconds=max(
                            0.0, float(config.frontier_retry_base_seconds or 0.0)
                        ),
                        retry_max_seconds=max(
                            max(0.0, float(config.frontier_retry_base_seconds or 0.0)),
                            float(config.frontier_retry_max_seconds or 0.0),
                        ),
                    )
                except FrontierLeaseError:
                    current = queue_store.get(case_id, job.job_id)
                    heartbeat_failures.append("lease_fencing_lost_before_failure_commit")
            run_summaries.append(
                {
                    "case_id": expansion_case_id,
                    "target_smiles": frontier_smiles,
                    "depth": depth,
                    "accepted": False,
                    "frontier_job_id": job.job_id,
                    "frontier_job_state": current.state.value if current else "missing",
                    "reasons": [failure_reason],
                    "lease_heartbeat_errors": heartbeat_failures,
                    "result_quarantined": bool(heartbeat_failures),
                    "proposal_expansion_recorded": False,
                }
            )
            _write_campaign_state(
                campaign_state_path,
                case_id=case_id,
                target_smiles=target_smiles,
                expansions=expansions,
                runs=run_summaries,
            )
            continue
        team_report_ref = expansion_run_dir / "codex_retrosynthesis_team" / "team_report.json"
        summary = {
            "case_id": expansion_case_id,
            "target_smiles": frontier_smiles,
            "depth": depth,
            "accepted": False,
            "team_report_accepted": bool(team_report.get("accepted")),
            "frontier_job_id": job.job_id,
            "team_report_ref": str(team_report_ref),
            "route_consensus_ref": str(team_report.get("route_consensus_ref") or ""),
            "reasons": [str(item) for item in team_report.get("reasons") or []],
            "proposal_expansion_recorded": False,
            "lease_heartbeat_errors": heartbeat_failures,
        }
        if heartbeat_failures:
            current = queue_store.get(case_id, job.job_id)
            summary["frontier_job_state"] = current.state.value if current else "missing"
            summary["result_quarantined"] = True
            summary["reasons"] = sorted(
                {*summary["reasons"], "lease_fencing_lost_result_quarantined"}
            )
            run_summaries.append(summary)
            _write_campaign_state(
                campaign_state_path,
                case_id=case_id,
                target_smiles=target_smiles,
                expansions=expansions,
                runs=run_summaries,
            )
            continue
        if team_report.get("accepted"):
            expansion = make_route_consensus_expansion(
                dict(team_report.get("route_consensus") or {}),
                requested_product_smiles=frontier_smiles,
                consensus_ref=str(team_report.get("route_consensus_ref") or ""),
                agent_run_ref=str((team_report.get("coordinator") or {}).get("run_record_ref") or ""),
                depth=depth,
            )
            committed_summary = {**summary, "accepted": True}
            try:
                expansion_commit_path = _write_expansion_commit(
                    root_output_dir=root_output_dir,
                    case_id=case_id,
                    job=job,
                    team_report_ref=team_report_ref,
                    expansion=expansion,
                    summary=committed_summary,
                )
                finalized_job = queue_store.complete(
                    case_id,
                    job.job_id,
                    lease_token=job.lease_token,
                    result_ref=str(expansion_commit_path),
                    closure_kind="proposal_expansion",
                    achieved_proof_level=0,
                )
            except FrontierLeaseError:
                current = queue_store.get(case_id, job.job_id)
                summary["frontier_job_state"] = current.state.value if current else "missing"
                summary["result_quarantined"] = True
                summary["reasons"] = sorted(
                    {*summary["reasons"], "lease_fencing_lost_result_quarantined"}
                )
                run_summaries.append(summary)
                _write_campaign_state(
                    campaign_state_path,
                    case_id=case_id,
                    target_smiles=target_smiles,
                    expansions=expansions,
                    runs=run_summaries,
                )
                continue
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                reason = f"expansion_commit_error:{type(exc).__name__}:{exc}"
                try:
                    finalized_job = queue_store.fail(
                        case_id,
                        job.job_id,
                        lease_token=job.lease_token,
                        reason=reason,
                        retryable=True,
                        retry_base_seconds=max(
                            0.0, float(config.frontier_retry_base_seconds or 0.0)
                        ),
                        retry_max_seconds=max(
                            max(0.0, float(config.frontier_retry_base_seconds or 0.0)),
                            float(config.frontier_retry_max_seconds or 0.0),
                        ),
                    )
                except FrontierLeaseError:
                    finalized_job = queue_store.get(case_id, job.job_id)
                summary["frontier_job_state"] = (
                    finalized_job.state.value if finalized_job else "missing"
                )
                summary["reasons"] = sorted({*summary["reasons"], reason})
                run_summaries.append(summary)
                _write_campaign_state(
                    campaign_state_path,
                    case_id=case_id,
                    target_smiles=target_smiles,
                    expansions=expansions,
                    runs=run_summaries,
                )
                continue

            summary = committed_summary
            summary["frontier_job_state"] = finalized_job.state.value
            summary["proposal_expansion_recorded"] = True
            summary["expansion_commit_ref"] = str(expansion_commit_path)
            expansion_id = str(expansion.get("expansion_id") or "")
            if expansion_id not in {
                str(row.get("expansion_id") or "") for row in expansions
            }:
                expansions.append(expansion)
            expanded_smiles.add(frontier_smiles)
            summary["proposal_expansion_recorded"] = True
            graph = assemble_route_consensus_graph(
                expansions,
                case_id=case_id,
                target_smiles=target_smiles,
                max_depth=max_depth,
            )
            _submit_graph_frontiers(
                graph=graph,
                scheduler=scheduler,
                config=config,
                case_id=case_id,
                expanded_smiles=expanded_smiles,
                max_depth=max_depth,
                max_expansions=max_expansions,
                frontier_batch_size=frontier_batch_size,
            )
            if depth == 0:
                root_report = dict(team_report)
        else:
            try:
                finalized_job = queue_store.fail(
                    case_id,
                    job.job_id,
                    lease_token=job.lease_token,
                    reason="codex_team_report_rejected",
                    retryable=True,
                    retry_base_seconds=max(
                        0.0, float(config.frontier_retry_base_seconds or 0.0)
                    ),
                    retry_max_seconds=max(
                        max(0.0, float(config.frontier_retry_base_seconds or 0.0)),
                        float(config.frontier_retry_max_seconds or 0.0),
                    ),
                )
            except FrontierLeaseError:
                finalized_job = queue_store.get(case_id, job.job_id)
                summary["result_quarantined"] = True
                summary["reasons"] = sorted(
                    {*summary["reasons"], "lease_fencing_lost_result_quarantined"}
                )
            summary["frontier_job_state"] = (
                finalized_job.state.value if finalized_job else "missing"
            )
            if depth == 0:
                root_report = dict(team_report)
        run_summaries.append(summary)
        _write_campaign_state(
            campaign_state_path,
            case_id=case_id,
            target_smiles=target_smiles,
            expansions=expansions,
            runs=run_summaries,
        )

    if not root_report:
        root_report = {
            "schema_version": CODEX_RETROSYNTHESIS_TEAM_SCHEMA,
            "accepted": False,
            "case_id": case_id,
            "target_name": target_name,
            "target_smiles": target_smiles,
            "reasons": ["root_retrosynthesis_team_missing"],
            "semantics": {"no_solved_claim": True},
        }
    remaining_frontier = select_route_consensus_frontier(
        graph,
        limit=max_expansions * frontier_batch_size,
    )
    queue_jobs = queue_store.list_jobs(case_id)
    terminal_smiles = _campaign_terminal_smiles(graph) or [root_smiles]
    open_reaction_proofs = [
        {
            "frontier": str(step.get("step_id") or ""),
            "product_smiles": str(step.get("product_smiles") or ""),
            "reason": "proposal_hyperedge_not_reaction_validated",
        }
        for step in graph.get("steps") or []
        if isinstance(step, dict)
    ]
    completeness = assess_frontier_completeness(
        terminal_smiles,
        queue_jobs,
        open_proof_frontiers=open_reaction_proofs,
        required_proof_level=2,
    )
    graph_path = root_output_dir / "route_consensus_graph.json"
    _write_json(graph_path, graph)
    campaign = {
        "schema_version": CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA,
        "root_case_id": case_id,
        "max_depth": max_depth,
        "max_expansions": max_expansions,
        "expansion_run_count": len(run_summaries),
        "attempt_run_count": len(run_summaries),
        "unique_frontier_run_count": len(
            {str(row.get("frontier_job_id") or "") for row in run_summaries}
        ),
        "accepted_expansion_count": sum(1 for row in run_summaries if row.get("accepted")),
        "graph_complete": completeness.complete,
        "proposal_graph_exhausted": not remaining_frontier,
        "remaining_frontier": remaining_frontier,
        "frontier_completeness": completeness.to_dict(),
        "frontier_queue": queue_store.snapshot(case_id),
        "frontier_queue_ref": str(root_output_dir / "frontier_queue"),
        "runs": run_summaries,
        "recovery_errors": recovery_errors,
        "resumable_at": _next_retry_available_at(queue_jobs),
        "semantics": {
            "frontier_reexpanded_by_direct_codex_teams": True,
            "persistent_stock_first_frontier_scheduler": True,
            "fenced_expansion_commit_before_queue_completion": True,
            "child_frontiers_published_only_after_parent_queue_commit": True,
            "campaign_state_is_rebuildable_atomic_cache": True,
            "lease_heartbeat_enabled": True,
            "queue_exhaustion_is_not_route_completion": True,
            "reaction_validation_required_for_graph_complete": True,
            "advisory_only": True,
            "no_solved_claim": True,
        },
    }
    root_report["campaign"] = campaign
    root_report["route_consensus_graph_ref"] = str(graph_path)
    root_report["route_consensus_graph"] = graph
    root_report["route_consensus_expansions"] = expansions
    root_report["route_expansion_count"] = len(expansions)
    _write_json(root_output_dir / "team_report.json", root_report)
    return root_report


def migrate_legacy_campaign_commits(
    *,
    case_id: str,
    target_smiles: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Upgrade a pre-outbox campaign without rerunning any Codex team."""

    root_run_dir = Path(run_dir).resolve()
    root_output_dir = root_run_dir / "codex_retrosynthesis_team"
    state_path = root_output_dir / "campaign_state.json"
    raw_state = _read_json_object(state_path)
    if (
        raw_state.get("schema_version") != "codex_retrosynthesis_campaign_state.v1"
        or raw_state.get("case_id") != case_id
        or _canonical_target_smiles(raw_state.get("target_smiles"))
        != _canonical_target_smiles(target_smiles)
    ):
        raise ValueError("legacy campaign state identity is invalid")
    expansions = [
        dict(row) for row in raw_state.get("expansions") or [] if isinstance(row, dict)
    ]
    runs = [dict(row) for row in raw_state.get("runs") or [] if isinstance(row, dict)]
    queue = PersistentFrontierQueue(root_output_dir / "frontier_queue")
    migrated: list[str] = []
    skipped: list[str] = []
    for job in queue.list_jobs(case_id):
        if (
            job.state != FrontierJobState.SUCCEEDED
            or job.closure_kind != "proposal_expansion"
            or str(job.result_ref).startswith("frontier-skip:")
        ):
            continue
        try:
            Path(job.result_ref).resolve().relative_to(
                (root_output_dir / "campaign_commits").resolve()
            )
            skipped.append(job.job_id)
            continue
        except ValueError:
            pass
        summary = next(
            (
                dict(row)
                for row in runs
                if row.get("frontier_job_id") == job.job_id
                and row.get("accepted") is True
                and row.get("proposal_expansion_recorded") is True
            ),
            {},
        )
        expansion = next(
            (
                dict(row)
                for row in expansions
                if _canonical_target_smiles(row.get("requested_product_smiles"))
                == job.frontier_smiles
            ),
            {},
        )
        if not summary or not expansion:
            skipped.append(job.job_id)
            continue
        synthetic_token = "legacy-succeeded:" + hashlib.sha256(
            str(job.result_ref).encode("utf-8")
        ).hexdigest()
        migrated_job = replace(job, lease_token=synthetic_token)
        commit_path = _write_expansion_commit(
            root_output_dir=root_output_dir,
            case_id=case_id,
            job=migrated_job,
            team_report_ref=Path(job.result_ref),
            expansion=expansion,
            summary={**summary, "legacy_commit_migration": True},
        )
        queue.rebind_succeeded_result(
            case_id,
            job.job_id,
            expected_result_ref=job.result_ref,
            result_ref=str(commit_path),
            metadata_updates={
                "legacy_result_migrated": True,
                "legacy_result_ref_sha256": hashlib.sha256(
                    str(job.result_ref).encode("utf-8")
                ).hexdigest(),
            },
        )
        migrated.append(job.job_id)
    _write_campaign_state(
        state_path,
        case_id=case_id,
        target_smiles=target_smiles,
        expansions=expansions,
        runs=runs,
    )
    recovery_errors = _reconcile_expansion_commits(
        queue=queue,
        run_id=case_id,
        root_output_dir=root_output_dir,
        expansions=expansions,
        runs=runs,
    )
    return {
        "schema_version": "codex_retrosynthesis_campaign_migration.v1",
        "accepted": not recovery_errors,
        "run_dir": str(root_run_dir),
        "case_id": case_id,
        "migrated_job_ids": migrated,
        "skipped_job_ids": skipped,
        "recovery_errors": recovery_errors,
        "campaign_state_ref": str(state_path),
        "frontier_queue": queue.snapshot(case_id),
    }


def _stock_request(config: RetrosynthesisTeamConfig, canonical_smiles: str) -> dict[str, Any]:
    raw = (config.stock_snapshots or {}).get(canonical_smiles) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _submit_graph_frontiers(
    *,
    graph: dict[str, Any],
    scheduler: FrontierScheduler,
    config: RetrosynthesisTeamConfig,
    case_id: str,
    expanded_smiles: set[str],
    max_depth: int,
    max_expansions: int,
    frontier_batch_size: int,
) -> int:
    added = 0
    for next_frontier in select_route_consensus_frontier(
        graph,
        limit=max_expansions * frontier_batch_size,
    ):
        next_smiles = _canonical_target_smiles(next_frontier.get("target_smiles"))
        next_depth = int(next_frontier.get("depth") or 0)
        if not next_smiles or next_smiles in expanded_smiles or next_depth >= max_depth:
            continue
        scheduler.submit(
            run_id=case_id,
            case_id=case_id,
            frontier_smiles=next_smiles,
            frontier_node_id=str(next_frontier.get("node_id") or f"frontier:{next_smiles}"),
            idempotency_key=(
                f"{case_id}:frontier:{hashlib.sha256(next_smiles.encode()).hexdigest()}"
            ),
            stock_request=_stock_request(config, next_smiles),
            required_proof_level=2,
            proof_deficit=2,
            closure_probability=max(
                0.0,
                min(1.0, float(next_frontier.get("priority_score") or 0.5)),
            ),
            diversity_gain=0.5,
            dependency_ids=(),
            max_attempts=3,
            metadata={
                **dict(next_frontier),
                "target_name": next_smiles,
            },
        )
        added += 1
        if added >= frontier_batch_size:
            break
    return added


def _next_retry_available_at(jobs: list[FrontierJob]) -> str:
    values = sorted(
        row.available_at
        for row in jobs
        if row.state == FrontierJobState.RETRY_WAIT and row.available_at
    )
    return values[0] if values else ""


def _next_retry_delay(jobs: list[FrontierJob]) -> float | None:
    value = _next_retry_available_at(jobs)
    if not value:
        return None
    try:
        available = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if available.tzinfo is None:
        available = available.replace(tzinfo=timezone.utc)
    return max(0.0, (available - datetime.now(timezone.utc)).total_seconds())


def _campaign_terminal_smiles(graph: dict[str, Any]) -> list[str]:
    nodes = {
        str(row.get("node_id") or ""): str(row.get("smiles") or "")
        for row in graph.get("nodes") or []
        if isinstance(row, dict)
    }
    result: list[str] = []
    for route in graph.get("route_hypotheses") or []:
        if not isinstance(route, dict):
            continue
        for frontier in route.get("frontier") or []:
            if not isinstance(frontier, dict):
                continue
            smiles = _canonical_target_smiles(nodes.get(str(frontier.get("node_id") or "")))
            if smiles and smiles not in result:
                result.append(smiles)
    return result


def _load_campaign_state(
    path: Path,
    *,
    case_id: str,
    target_smiles: str,
) -> dict[str, Any]:
    row = _read_json_object(path)
    recorded_digest = str(row.get("content_sha256") or "")
    digest_payload = dict(row)
    digest_payload.pop("content_sha256", None)
    if (
        not recorded_digest
        or recorded_digest != _payload_digest(digest_payload)
        or row.get("schema_version") != "codex_retrosynthesis_campaign_state.v1"
        or row.get("case_id") != case_id
        or _canonical_target_smiles(row.get("target_smiles"))
        != _canonical_target_smiles(target_smiles)
    ):
        return {}
    return row


def _write_campaign_state(
    path: Path,
    *,
    case_id: str,
    target_smiles: str,
    expansions: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": "codex_retrosynthesis_campaign_state.v1",
        "case_id": case_id,
        "target_smiles": target_smiles,
        "expansions": expansions,
        "runs": runs,
    }
    payload["content_sha256"] = _payload_digest(payload)
    _write_json(path, payload)


@contextmanager
def _frontier_lease_heartbeat(
    queue: PersistentFrontierQueue,
    *,
    run_id: str,
    job: FrontierJob,
    lease_seconds: float,
    interval_seconds: float = 0.0,
) -> Iterator[list[str]]:
    """Renew one synchronous team lease and expose any fencing failure."""

    stop = threading.Event()
    errors: list[str] = []
    interval = (
        float(interval_seconds)
        if interval_seconds > 0
        else max(1.0, min(float(lease_seconds) / 3.0, 60.0))
    )

    def beat() -> None:
        while not stop.wait(interval):
            try:
                queue.heartbeat(
                    run_id,
                    job.job_id,
                    lease_token=job.lease_token,
                    extend_seconds=lease_seconds,
                )
            except (FrontierLeaseError, FrontierQueueError, KeyError, OSError) as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
                stop.set()

    thread = threading.Thread(
        target=beat,
        name=f"frontier-heartbeat-{job.job_id[-12:]}",
        daemon=True,
    )
    thread.start()
    try:
        yield errors
    finally:
        stop.set()
        thread.join(timeout=max(1.0, min(interval + 1.0, 5.0)))


def _write_expansion_commit(
    *,
    root_output_dir: Path,
    case_id: str,
    job: FrontierJob,
    team_report_ref: Path,
    expansion: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    """Persist the accepted expansion before mutating its queue job."""

    report_path = team_report_ref.resolve()
    try:
        report_path.relative_to(root_output_dir.parent)
    except ValueError as exc:
        raise ValueError("team report is outside the campaign run") from exc
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    if not isinstance(report, dict) or report.get("accepted") is not True:
        raise ValueError("expansion commit requires an accepted team report")
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    commit_root = root_output_dir / "campaign_commits"
    object_path = (
        commit_root
        / "objects"
        / "sha256"
        / report_digest[:2]
        / report_digest
        / "team_report.json"
    )
    _write_immutable_bytes(object_path, report_bytes)
    token_digest = hashlib.sha256(job.lease_token.encode("utf-8")).hexdigest()
    expansion_digest = _payload_digest(expansion)
    payload = {
        "schema_version": CODEX_RETROSYNTHESIS_EXPANSION_COMMIT_SCHEMA,
        "case_id": case_id,
        "job_id": job.job_id,
        "attempt": job.attempt,
        "lease_token_sha256": token_digest,
        "frontier_smiles": job.frontier_smiles,
        "team_report_ref": str(report_path),
        "team_report_content_path": str(object_path.resolve()),
        "team_report_sha256": report_digest,
        "expansion": expansion,
        "expansion_sha256": expansion_digest,
        "summary": summary,
    }
    payload["content_sha256"] = _payload_digest(payload)
    commit_name = (
        f"{hashlib.sha256(job.job_id.encode()).hexdigest()[:20]}"
        f"-a{job.attempt}-{token_digest[:12]}.json"
    )
    commit_path = commit_root / commit_name
    if commit_path.is_file():
        existing = _read_json_object(commit_path)
        if existing != payload:
            raise ValueError("immutable expansion commit conflict")
        return commit_path
    _write_json(commit_path, payload)
    return commit_path


def _load_expansion_commit(
    path: Path,
    *,
    root_output_dir: Path,
    expected_job: FrontierJob,
) -> tuple[dict[str, Any], str]:
    try:
        resolved = path.resolve()
        resolved.relative_to((root_output_dir / "campaign_commits").resolve())
    except (OSError, ValueError):
        return {}, "expansion_commit_path_outside_campaign"
    row = _read_json_object(resolved)
    digest_payload = dict(row)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    if (
        row.get("schema_version") != CODEX_RETROSYNTHESIS_EXPANSION_COMMIT_SCHEMA
        or row.get("case_id") != expected_job.run_id
        or row.get("job_id") != expected_job.job_id
        or int(row.get("attempt") or 0) != expected_job.attempt
        or not recorded_digest
        or recorded_digest != _payload_digest(digest_payload)
    ):
        return {}, "expansion_commit_identity_or_digest_invalid"
    expansion = row.get("expansion")
    if not isinstance(expansion, dict) or row.get("expansion_sha256") != _payload_digest(expansion):
        return {}, "expansion_commit_payload_digest_invalid"
    object_path = Path(str(row.get("team_report_content_path") or ""))
    try:
        object_path.resolve().relative_to(
            (root_output_dir / "campaign_commits" / "objects").resolve()
        )
    except (OSError, ValueError):
        return {}, "expansion_commit_report_object_outside_campaign"
    if not object_path.is_file() or _sha256_file(object_path) != row.get("team_report_sha256"):
        return {}, "expansion_commit_report_object_invalid"
    report = _read_json_object(object_path)
    if report.get("accepted") is not True:
        return {}, "expansion_commit_report_not_accepted"
    return row, ""


def _reconcile_expansion_commits(
    *,
    queue: PersistentFrontierQueue,
    run_id: str,
    root_output_dir: Path,
    expansions: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> list[str]:
    """Rebuild the mutable campaign cache from fenced succeeded-job commits."""

    expansion_ids = {str(row.get("expansion_id") or "") for row in expansions}
    run_job_ids = {
        str(row.get("frontier_job_id") or "")
        for row in runs
        if row.get("proposal_expansion_recorded") is True
    }
    errors: list[str] = []
    for job in queue.list_jobs(run_id):
        if (
            job.state != FrontierJobState.SUCCEEDED
            or job.closure_kind != "proposal_expansion"
            or str(job.result_ref).startswith("frontier-skip:")
        ):
            continue
        commit, reason = _load_expansion_commit(
            Path(job.result_ref),
            root_output_dir=root_output_dir,
            expected_job=job,
        )
        if reason:
            errors.append(f"{job.job_id}:{reason}")
            try:
                queue.invalidate_succeeded_result(
                    run_id,
                    job.job_id,
                    expected_result_ref=job.result_ref,
                    reason=reason,
                )
            except (FrontierQueueError, KeyError, ValueError) as exc:
                errors.append(
                    f"{job.job_id}:result_invalidation_failed:{type(exc).__name__}:{exc}"
                )
            continue
        expansion = dict(commit["expansion"])
        expansion_id = str(expansion.get("expansion_id") or "")
        if expansion_id and expansion_id not in expansion_ids:
            expansions.append(expansion)
            expansion_ids.add(expansion_id)
        if job.job_id not in run_job_ids:
            summary = dict(commit.get("summary") or {})
            summary["frontier_job_state"] = FrontierJobState.SUCCEEDED.value
            summary["proposal_expansion_recorded"] = True
            summary["recovered_from_expansion_commit"] = True
            runs.append(summary)
            run_job_ids.add(job.job_id)
    return errors


def _payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError("immutable campaign object conflict")
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _compact_blackboard_context(board: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "target_profile",
        "current_belief",
        "route_objective_summary",
        "target_side_disconnection_hypotheses",
        "route_failures",
        "terminal_blacklist",
        "retrosynthetic_proposals",
        "proposal_failure_feedback",
        "route_consensus_graph",
        "frontier_request",
    )
    compact = {key: board.get(key) for key in keys if board.get(key) not in (None, [], {})}
    evidence = dict(board.get("literature_evidence") or {})
    if evidence:
        compact["literature_evidence"] = {
            key: evidence.get(key)
            for key in (
                "source_candidates",
                "source_refs",
                "exact_literature_rows",
                "process_evidence_rows",
                "visual_chains",
            )
            if evidence.get(key)
        }
    return compact


def _normalized_roles(values: list[str]) -> list[str]:
    seen: set[str] = set()
    roles: list[str] = []
    for value in values:
        role = "_".join(part for part in str(value or "").strip().lower().replace("-", "_").split("_") if part)
        if not role or role in seen:
            continue
        seen.add(role)
        roles.append(role)
    if len(roles) < 2:
        raise ValueError("retrosynthesis coordinator requires at least two distinct child roles")
    if len(roles) > 8:
        raise ValueError("retrosynthesis coordinator supports at most eight child roles")
    return roles


def _validated_child_reports(
    *,
    record: WorkerRunRecord,
    required_roles: list[str],
    case_id: str,
    target_smiles: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_agent_ids: set[str] = set()
    report_dir = output_dir / "child_reports"
    for index, raw_child in enumerate((record.metadata or {}).get("child_agents") or [], start=1):
        if not isinstance(raw_child, dict):
            continue
        child = dict(raw_child)
        role = _normalize_role(child.get("role"))
        agent_id = str(child.get("agent_id") or child.get("call_id") or f"child-{index}")
        message = child.get("message")
        message_text = message if isinstance(message, str) else ""
        message_bytes = len(message_text.encode("utf-8"))
        message_sha256 = hashlib.sha256(message_text.encode("utf-8")).hexdigest() if message_text else ""
        parsed = _child_report_payload(message_text) if message_bytes <= MAX_CHILD_REPORT_BYTES else {}
        parsed, normalization_repairs = _conservative_child_report_shape_repair(parsed)
        validation_reasons: list[str] = []
        status = str(child.get("status") or "").strip().lower()
        if status not in {"completed", "succeeded", "success", "accepted"}:
            validation_reasons.append("child_not_completed")
        if not str(child.get("wait_call_id") or "").strip():
            validation_reasons.append("child_completion_not_observed_by_wait")
        if agent_id in seen_agent_ids:
            validation_reasons.append("duplicate_child_agent_id")
        seen_agent_ids.add(agent_id)
        if role not in required_roles:
            validation_reasons.append("child_role_not_required_or_missing")
        if str(child.get("role_binding_method") or "") != "explicit_spawn_contract":
            validation_reasons.append("child_role_not_bound_to_spawn_prompt")
        if role in seen_roles:
            validation_reasons.append("duplicate_child_role_report")
        if message_bytes > MAX_CHILD_REPORT_BYTES:
            validation_reasons.append("child_report_too_large")
        if not parsed:
            validation_reasons.append("child_report_json_missing_or_invalid")
        else:
            validation_reasons.extend(_strict_child_report_shape_reasons(parsed))
            validation_reasons.extend(validate_retrosynthesis_report_payload(parsed))
            if str(parsed.get("case_id") or "") != str(case_id):
                validation_reasons.append("child_report_case_id_mismatch")
            if _normalize_role(parsed.get("agent_role")) != role:
                validation_reasons.append("child_report_role_mismatch")
            if not _same_smiles(parsed.get("target_smiles"), target_smiles=target_smiles):
                validation_reasons.append("child_report_target_mismatch")
        accepted = not validation_reasons
        safe_role = role if role else f"unassigned-{index}"
        report_path = report_dir / f"{index:02d}-{safe_role}-{message_sha256[:12] or 'empty'}.json"
        report_ref = f"{report_path}#agent={agent_id}"
        candidate_count = 0
        if accepted:
            seen_roles.add(role)
            for raw_candidate in parsed.get("candidates") or []:
                if not isinstance(raw_candidate, dict):
                    continue
                candidate = dict(raw_candidate)
                # Source channels are assigned by the trusted orchestrator;
                # child prose cannot impersonate another independent source.
                candidate["source_channel"] = ROLE_SOURCE_CHANNELS.get(role, "other")
                candidate["report_ref"] = report_ref
                candidates.append(candidate)
                candidate_count += 1
        persisted = {
            "schema_version": "codex_child_report_observation.v1",
            "accepted": accepted,
            "case_id": case_id,
            "role": role,
            "agent_id": agent_id,
            "status": str(child.get("status") or ""),
            "payload": parsed,
            "candidate_count": candidate_count,
            "message_bytes": message_bytes,
            "message_sha256": message_sha256,
            "normalization_repairs": normalization_repairs,
            "validation_reasons": sorted(set(validation_reasons)),
            "source_event_log_ref": str((record.metadata or {}).get("event_log_path") or ""),
        }
        _write_json(report_path, persisted)
        reports.append(
            {
                "accepted": accepted,
                "role": role,
                "agent_id": agent_id,
                "status": str(child.get("status") or ""),
                "report_ref": report_ref,
                "candidate_count": candidate_count,
                "message_bytes": message_bytes,
                "message_sha256": message_sha256,
                "normalization_repairs": normalization_repairs,
                "validation_reasons": sorted(set(validation_reasons)),
            }
        )
    return reports, candidates


def _child_report_payload(message: Any) -> dict[str, Any]:
    if not isinstance(message, str):
        return {}
    value = _json_object_from_text(message)
    if value.get("schema_version") != "retrosynthesis_proposal_report.v1":
        return {}
    return dict(value)


def _conservative_child_report_shape_repair(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Repair only syntax-level unknowns to conservative advisory defaults."""
    if not payload:
        return {}, []
    repaired = dict(payload)
    repairs: list[str] = []
    for key in ("evidence_refs", "limitations"):
        if repaired.get(key) is None:
            repaired[key] = []
            repairs.append(f"report:{key}:null_to_empty_list")
    raw_candidates = repaired.get("candidates")
    if not isinstance(raw_candidates, list):
        return repaired, repairs
    candidates: list[Any] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            candidates.append(raw)
            continue
        candidate = dict(raw)
        for key, default in (
            ("candidate_id", ""),
            ("reaction_family", "unspecified"),
            ("transformation_rationale", ""),
            ("source_channel", "other"),
            ("evidence_level", "model_only"),
            ("catalyst", ""),
            ("enzyme", ""),
        ):
            if candidate.get(key) is None:
                candidate[key] = default
                repairs.append(f"candidate:{index}:{key}:null_to_conservative_default")
        if not isinstance(candidate.get("confidence"), str):
            candidate["confidence"] = "low"
            repairs.append(f"candidate:{index}:confidence:non_string_to_low")
        for key in (
            "source_refs",
            "evidence_refs",
            "conditions",
            "limitations",
            "required_validation",
        ):
            value = candidate.get(key)
            if value is None:
                candidate[key] = []
                repairs.append(f"candidate:{index}:{key}:null_to_empty_list")
            elif isinstance(value, str):
                candidate[key] = [value] if value.strip() else []
                repairs.append(f"candidate:{index}:{key}:string_to_string_list")
        candidates.append(candidate)
    repaired["candidates"] = candidates
    return repaired, repairs


def _strict_child_report_shape_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    keys = set(payload)
    if keys != CHILD_REPORT_KEYS:
        reasons.append("child_report_fields_not_exact")
    if not isinstance(payload.get("candidates"), list):
        reasons.append("child_report_candidates_not_list")
        candidates: list[Any] = []
    else:
        candidates = list(payload.get("candidates") or [])
    if not isinstance(payload.get("evidence_refs"), list) or not all(
        isinstance(item, str) for item in payload.get("evidence_refs") or []
    ):
        reasons.append("child_report_evidence_refs_not_string_list")
    if not isinstance(payload.get("limitations"), list) or not all(
        isinstance(item, str) for item in payload.get("limitations") or []
    ):
        reasons.append("child_report_limitations_not_string_list")
    for index, candidate in enumerate(candidates):
        prefix = f"child_candidate:{index}:"
        if not isinstance(candidate, dict):
            reasons.append(prefix + "not_object")
            continue
        if set(candidate) != CHILD_CANDIDATE_KEYS:
            reasons.append(prefix + "fields_not_exact")
        for key in (
            "schema_version",
            "candidate_id",
            "product_smiles",
            "reaction_family",
            "transformation_rationale",
            "source_channel",
            "evidence_level",
            "confidence",
            "catalyst",
            "enzyme",
        ):
            if not isinstance(candidate.get(key), str):
                reasons.append(prefix + f"{key}_not_string")
        for key in (
            "precursor_smiles",
            "source_refs",
            "evidence_refs",
            "conditions",
            "limitations",
            "required_validation",
        ):
            if not isinstance(candidate.get(key), list) or not all(
                isinstance(item, str) for item in candidate.get(key) or []
            ):
                reasons.append(prefix + f"{key}_not_string_list")
        if candidate.get("no_solved_claim") is not True:
            reasons.append(prefix + "missing_no_solved_claim")
        if candidate.get("not_parent_route_proof") is not True:
            reasons.append(prefix + "missing_not_parent_route_proof")
    return sorted(set(reasons))


def _annotate_child_report_validation(
    record: WorkerRunRecord,
    reports: list[dict[str, Any]],
) -> None:
    by_agent_id = {
        str(row.get("agent_id") or ""): row
        for row in reports
        if str(row.get("agent_id") or "")
    }
    metadata = dict(record.metadata or {})
    children: list[dict[str, Any]] = []
    for raw_child in metadata.get("child_agents") or []:
        if not isinstance(raw_child, dict):
            continue
        child = dict(raw_child)
        agent_id = str(child.get("agent_id") or child.get("call_id") or "")
        report = dict(by_agent_id.get(agent_id) or {})
        child.pop("message", None)
        child["report_accepted"] = report.get("accepted") is True
        child["report_ref"] = str(report.get("report_ref") or "")
        child["report_normalization_repairs"] = list(
            report.get("normalization_repairs") or []
        )
        child["report_validation_reasons"] = list(report.get("validation_reasons") or [])
        child["message_bytes"] = int(report.get("message_bytes") or 0)
        child["message_sha256"] = str(report.get("message_sha256") or "")
        children.append(child)
    metadata["child_agents"] = children
    record.metadata = metadata


def _json_object_from_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
        return dict(parsed) if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_nonfinite_json_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _same_smiles(left: Any, *, target_smiles: str) -> bool:
    left_mol = Chem.MolFromSmiles(str(left or ""))
    target_mol = Chem.MolFromSmiles(str(target_smiles or ""))
    if left_mol is None or target_mol is None:
        return False
    return Chem.MolToSmiles(left_mol, isomericSmiles=True) == Chem.MolToSmiles(
        target_mol,
        isomericSmiles=True,
    )


def _normalize_role(value: Any) -> str:
    return "_".join(
        part
        for part in str(value or "").strip().lower().replace("-", "_").split("_")
        if part
    )


def _canonical_target_smiles(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
