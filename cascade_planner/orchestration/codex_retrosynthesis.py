"""Run a Codex coordinator that directly delegates to specialist child agents."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem

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
from cascade_planner.runtime import Budget as RuntimeBudget
from cascade_planner.runtime import CodexTeamRuntimeTracker


CODEX_RETROSYNTHESIS_TEAM_SCHEMA = "codex_retrosynthesis_team_run.v1"
CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA = "codex_retrosynthesis_campaign.v1"
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
) -> dict[str, Any]:
    """Recursively run direct Codex teams for target and frontier molecules."""
    config = config or RetrosynthesisTeamConfig()
    max_depth = max(1, int(config.max_depth or 1))
    max_expansions = max(1, int(config.max_expansions or 1))
    frontier_batch_size = max(1, int(config.frontier_batch_size or 1))
    root_run_dir = Path(run_dir).resolve()
    root_output_dir = root_run_dir / "codex_retrosynthesis_team"
    queue: list[dict[str, Any]] = [
        {
            "target_name": target_name,
            "target_smiles": target_smiles,
            "depth": 0,
            "node_id": "",
            "parent_step_ids": [],
        }
    ]
    queued_smiles = {_canonical_target_smiles(target_smiles)}
    expanded_smiles: set[str] = set()
    expansions: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    root_report: dict[str, Any] = {}
    graph = assemble_route_consensus_graph(
        [],
        case_id=case_id,
        target_smiles=target_smiles,
        max_depth=max_depth,
    )

    while queue and len(run_summaries) < max_expansions:
        frontier = queue.pop(0)
        frontier_smiles = _canonical_target_smiles(frontier.get("target_smiles"))
        depth = int(frontier.get("depth") or 0)
        if not frontier_smiles or frontier_smiles in expanded_smiles or depth >= max_depth:
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
        except Exception as exc:
            run_summaries.append(
                {
                    "case_id": expansion_case_id,
                    "target_smiles": frontier_smiles,
                    "depth": depth,
                    "accepted": False,
                    "reasons": [f"team_runtime_error:{type(exc).__name__}:{exc}"],
                }
            )
            if depth == 0:
                raise
            expanded_smiles.add(frontier_smiles)
            continue
        if depth == 0:
            root_report = dict(team_report)
        expanded_smiles.add(frontier_smiles)
        team_report_ref = expansion_run_dir / "codex_retrosynthesis_team" / "team_report.json"
        run_summaries.append(
            {
                "case_id": expansion_case_id,
                "target_smiles": frontier_smiles,
                "depth": depth,
                "accepted": bool(team_report.get("accepted")),
                "team_report_ref": str(team_report_ref),
                "route_consensus_ref": str(team_report.get("route_consensus_ref") or ""),
                "reasons": [str(item) for item in team_report.get("reasons") or []],
            }
        )
        if team_report.get("accepted"):
            expansions.append(
                make_route_consensus_expansion(
                    dict(team_report.get("route_consensus") or {}),
                    requested_product_smiles=frontier_smiles,
                    consensus_ref=str(team_report.get("route_consensus_ref") or ""),
                    agent_run_ref=str((team_report.get("coordinator") or {}).get("run_record_ref") or ""),
                    depth=depth,
                )
            )
            graph = assemble_route_consensus_graph(
                expansions,
                case_id=case_id,
                target_smiles=target_smiles,
                max_depth=max_depth,
            )
            added_frontiers = 0
            for next_frontier in select_route_consensus_frontier(graph, limit=max_expansions):
                next_smiles = _canonical_target_smiles(next_frontier.get("target_smiles"))
                if not next_smiles or next_smiles in expanded_smiles or next_smiles in queued_smiles:
                    continue
                queued_smiles.add(next_smiles)
                queue.append(
                    {
                        **dict(next_frontier),
                        "target_name": next_smiles,
                        "target_smiles": next_smiles,
                    }
                )
                added_frontiers += 1
                if added_frontiers >= frontier_batch_size:
                    break

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
    remaining_frontier = select_route_consensus_frontier(graph, limit=max_expansions)
    graph_path = root_output_dir / "route_consensus_graph.json"
    _write_json(graph_path, graph)
    campaign = {
        "schema_version": CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA,
        "root_case_id": case_id,
        "max_depth": max_depth,
        "max_expansions": max_expansions,
        "expansion_run_count": len(run_summaries),
        "accepted_expansion_count": sum(1 for row in run_summaries if row.get("accepted")),
        "graph_complete": not remaining_frontier,
        "remaining_frontier": remaining_frontier,
        "runs": run_summaries,
        "semantics": {
            "frontier_reexpanded_by_direct_codex_teams": True,
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
