"""Run a Codex coordinator that directly delegates to specialist child agents."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable, Mapping
import errno
from functools import wraps
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
from cascade_planner.application.frontier_ledger import (
    project_frontier_ledger,
    validate_frontier_ledger,
)
from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerRunRecord,
    WorkerTask,
    run_codex_worker,
    validate_worker_output,
)
from cascade_planner.harness.reaction_step_verifier import (
    REACTION_STEP_VERIFIER_VERSION,
    verify_reaction_route,
    verify_reaction_step,
)
from cascade_planner.routes.consensus import (
    consensus_to_blackboard_proposals,
    fuse_route_candidates,
    normalize_route_candidate,
    validate_retrosynthesis_report_envelope_payload,
)
from cascade_planner.routes.graph import (
    assemble_route_consensus_graph,
    make_route_consensus_expansion,
    route_consensus_frontier_records,
    select_route_consensus_frontier,
    validate_route_consensus_expansion,
)
from cascade_planner.providers.contracts import StockProvider, validate_provider_result
from cascade_planner.providers.stock import (
    BenchmarkCatalogStockProvider,
    SnapshotStockProvider,
    stock_provider_set_authority_binding,
    stock_snapshot_sha256,
)
from cascade_planner.runtime import Budget as RuntimeBudget
from cascade_planner.runtime import CodexTeamRuntimeTracker
from cascade_planner.orchestration.admitted_hyperedges import (
    expansions_from_external_hyperedge_events,
    graph_exact_edge_signatures,
    load_external_hyperedge_events,
    record_external_hyperedges,
)


CODEX_RETROSYNTHESIS_TEAM_SCHEMA = "codex_retrosynthesis_team_run.v1"
CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA = "codex_retrosynthesis_campaign.v1"
CODEX_RETROSYNTHESIS_REACTION_PROOF_STATE_SCHEMA = (
    "codex_retrosynthesis_reaction_proof_state.v1"
)
CODEX_RETROSYNTHESIS_EXPANSION_COMMIT_SCHEMA = (
    "codex_retrosynthesis_expansion_commit.v2"
)
CODEX_RETROSYNTHESIS_CAMPAIGN_IDENTITY_SCHEMA = (
    "codex_retrosynthesis_campaign_identity.v1"
)
CODEX_RETROSYNTHESIS_ATTEMPT_EVENT_SCHEMA = (
    "codex_retrosynthesis_attempt_event.v1"
)
CODEX_RETROSYNTHESIS_CAMPAIGN_POLICY_SCHEMA = (
    "codex_retrosynthesis_campaign_policy.v1"
)
RETROSYNTHESIS_COORDINATOR_CONTRACT_VERSION = (
    "autoplanner.retrosynthesis_coordinator.v4"
)
CHILD_ACCEPTANCE_CONTRACT_VERSION = "autoplanner.child_acceptance.v3"
CODEX_RETROSYNTHESIS_BUDGET_EVENT_SCHEMA = (
    "codex_retrosynthesis_budget_event.v1"
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
CLOSURE_OBJECTIVES = frozenset(
    {"benchmark_search", "procurement", "in_house"}
)
EXPLORATION_MODES = frozenset({"first_solved", "exhaustive"})
CHILD_ACCEPTANCE_MODES = frozenset({"strict_all", "valid_subset_l0"})
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
# ``not_parent_route_proof`` is a harmless, fail-closed report-level guard.
# Older child contracts only required it on each candidate, while the shared
# prose contract also told agents to emit it for every scalar claim.  Accept it
# only when it is the literal safety value; arbitrary extra fields remain a
# hard schema failure.
CHILD_REPORT_OPTIONAL_SAFETY_KEYS = {"not_parent_route_proof"}
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
ISOLATABLE_CHILD_CANDIDATE_REASONS = frozenset(
    {
        "invalid_product_smiles",
        "invalid_precursor_smiles",
        "invalid_or_missing_material",
        "identity_proposal",
        "target_or_current_node_self_loop",
        "candidate_product_does_not_match_requested_target",
        "ancestor_or_target_cycle",
        "element_inventory_not_conserved",
        "large_atom_jump",
    }
)

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
    # Scientific completion is policy-fenced separately from operational
    # budgets. ``benchmark_search`` preserves the historical search-boundary
    # behavior without implying procurement or in-house availability.
    closure_objective: str = "benchmark_search"
    # Full route delivery is the default: one solved route is an intermediate
    # milestone, not permission to stop exploring the accepted hypergraph.
    exploration_mode: str = "exhaustive"
    # Strict all-child acceptance remains the default. ``valid_subset_l0`` is
    # an explicit campaign policy that tolerates only malformed/incomplete
    # reports from children whose explicit spawn was still host-observed.
    child_acceptance_mode: str = "strict_all"
    # One advisory OS lock is held for the complete campaign/reconciliation
    # transaction, including long model calls. It is never stolen by age.
    campaign_authority_lock_timeout_s: float = 3600.0
    # ``max_expansions`` is a cumulative campaign budget for unique, durably
    # committed proposal expansions.  Invocation caps keep a resumable
    # campaign from consuming that entire budget before newly discovered
    # blackboard/literature evidence can influence later frontiers.
    max_expansions: int = 4
    max_expansions_per_invocation: int = 2
    # Zero derives an attempt cap from the accepted-expansion invocation cap.
    # Agent failures consume this operational cap, never ``max_expansions``.
    max_attempt_runs_per_invocation: int = 0
    # Campaign-wide durable Agent-call budget.  Unlike the per-invocation cap,
    # this is restored from append-only started events after every restart.
    # Zero derives a conservative three-attempt allowance per accepted slot.
    max_attempt_runs: int = 0
    frontier_batch_size: int = 2
    frontier_lease_seconds: float = 1800.0
    frontier_heartbeat_interval_seconds: float = 0.0
    frontier_retry_base_seconds: float = 1.0
    frontier_retry_max_seconds: float = 60.0
    frontier_retry_wait_seconds: float = 5.0
    stock_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark_stock_catalog_artifact: str = ""
    benchmark_stock_catalog_sha256: str = ""
    benchmark_stock_catalog_name: str = ""
    # Materialized verifier inputs keyed by graph step id/signature.  A caller
    # supplied proof object is never authority: the current host verifier must
    # be able to replay its materialized candidate before an edge can close.
    reaction_proofs: dict[str, dict[str, Any]] = field(default_factory=dict)
    reaction_proof_reports: list[dict[str, Any]] = field(default_factory=list)


def _normalize_closure_objective(value: Any) -> str:
    objective = str(value or "benchmark_search").strip().lower()
    if objective not in CLOSURE_OBJECTIVES:
        raise ValueError(
            "closure_objective must be one of: "
            + ", ".join(sorted(CLOSURE_OBJECTIVES))
        )
    return objective


def _normalize_exploration_mode(value: Any) -> str:
    mode = str(value or "exhaustive").strip().lower()
    if mode not in EXPLORATION_MODES:
        raise ValueError(
            "exploration_mode must be one of: "
            + ", ".join(sorted(EXPLORATION_MODES))
        )
    return mode


def _normalize_child_acceptance_mode(value: Any) -> str:
    mode = str(value or "strict_all").strip().lower()
    if mode not in CHILD_ACCEPTANCE_MODES:
        raise ValueError(
            "child_acceptance_mode must be one of: "
            + ", ".join(sorted(CHILD_ACCEPTANCE_MODES))
        )
    return mode


def _derived_valid_child_quorum(roles: Iterable[str]) -> int:
    role_count = len(tuple(roles))
    return max(2, (role_count + 1) // 2)


@contextmanager
def _campaign_authority_lock(
    root_output_dir: Path,
    *,
    timeout_s: float,
) -> Iterator[None]:
    """Hold one non-stealable OS advisory lock for a campaign transaction."""

    timeout = float(timeout_s)
    if timeout <= 0:
        raise ValueError("campaign_authority_lock_timeout_s must be > 0")
    root_output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = root_output_dir / ".campaign-authority.lock"
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while not acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"campaign authority lock timeout: {lock_path}"
                    )
                time.sleep(0.02)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _campaign_authority_single_writer(function: Callable[..., dict[str, Any]]):
    """Serialize full campaign/reconciliation entry points on one run dir."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        run_dir = kwargs.get("run_dir")
        if run_dir is None:
            return function(*args, **kwargs)
        config = kwargs.get("config") or kwargs.get("campaign_config")
        timeout_s = float(
            getattr(config, "campaign_authority_lock_timeout_s", 3600.0)
        )
        root_output_dir = (
            Path(run_dir).resolve() / "codex_retrosynthesis_team"
        )
        with _campaign_authority_lock(root_output_dir, timeout_s=timeout_s):
            return function(*args, **kwargs)

    return wrapped


def build_retrosynthesis_coordinator_task(
    *,
    case_id: str,
    target_name: str,
    target_smiles: str,
    context_ref: str,
    allowed_workdir: str | Path,
    context_snapshot: dict[str, Any] | None = None,
    config: RetrosynthesisTeamConfig | None = None,
) -> WorkerTask:
    config = config or RetrosynthesisTeamConfig()
    roles = _normalized_roles(config.child_roles)
    child_acceptance_mode = _normalize_child_acceptance_mode(
        config.child_acceptance_mode
    )
    valid_child_quorum = _derived_valid_child_quorum(roles)
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
        "not_parent_route_proof": True,
    }
    inline_context = _bounded_coordinator_context(
        context_snapshot or {
            "target": {"name": target_name, "smiles": target_smiles},
            "blackboard": {},
            "literature_sources": [],
        }
    )
    completion_policy = (
        "Every required child must complete with a strictly valid final JSON report; "
        "otherwise the team report is rejected."
        if child_acceptance_mode == "strict_all"
        else (
            "You must spawn and observe every required role. If an already spawned "
            "sibling fails to complete or returns invalid JSON after it has been waited "
            f"on, the host may retain only a strictly valid subset of at least "
            f"{valid_child_quorum} unique roles as L0/model-only/low-confidence advice. "
            "Never replace or restate a missing child's final message."
        )
    )
    wait_policy = (
        "Never emit the coordinator's final report while any required child is "
        "pending, running, unobserved, or missing its final JSON message."
        if child_acceptance_mode == "strict_all"
        else (
            "Wait on every spawned child. A final report may preserve an incomplete "
            "or invalid sibling only as an explicitly degraded observation; it may "
            "never synthesize that sibling's missing final message."
        )
    )
    objective = f"""Coordinate an independent retrosynthesis review for {target_name}.

Target SMILES: {target_smiles}
Context snapshot: {context_ref}

Bounded inline context (use this even when the context file cannot be opened):
{json.dumps(inline_context, ensure_ascii=False, indent=2, sort_keys=True)}

The forbidden_return_smiles values are hard constraints. No child may propose
the current product itself or any ancestor/target in that list as a precursor;
doing so creates a retrosynthetic cycle and the parent will reject the entire
team report before publishing the expansion.

You must directly spawn one child agent for every role below:
{role_lines}

Concurrency is a correctness constraint: the coordinator itself occupies one
slot. Spawn at most three children at once, wait until every child in that
batch is completed, then spawn any remaining role. Repeatedly wait while a
child is `pending_init` or `running`; neither state is completion.
{wait_policy}

Child acceptance policy: {child_acceptance_mode}
{completion_policy}

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

Every candidate is exactly one retrosynthetic hyperedge: its listed precursor
multiset must jointly form candidate product_smiles in the single transformation
named by reaction_family and transformation_rationale. An enzyme or operation
that only prepares one precursor further upstream is not a target-forming edge;
record that idea only as a limitation or required validation, never by pairing
the upstream substrate with unrelated advanced fragments. Do not telescope
unlisted acylation, coupling, protection, resolution, or dehydration operations
into one candidate.

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
    config = config or RetrosynthesisTeamConfig()
    child_acceptance_mode = _normalize_child_acceptance_mode(
        config.child_acceptance_mode
    )
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
        context_snapshot=context,
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
    worker_artifact_validation = validate_worker_output(task, artifact)
    payload = dict(artifact.get("payload") or {})
    coordinator_candidates = [dict(row) for row in payload.get("candidates") or [] if isinstance(row, dict)]
    required_role_set = set(task.child_roles)
    required_children = len(task.child_roles)
    observed_children = len((record.metadata or {}).get("child_agents") or [])
    valid_child_quorum = _derived_valid_child_quorum(task.child_roles)
    valid_child_roles = {
        str(row.get("role") or "")
        for row in child_reports
        if row.get("accepted") is True
    }
    runtime_children = [
        dict(row)
        for row in runtime_summary.get("children") or []
        if isinstance(row, dict)
    ]
    spawn_audit = _audit_explicit_child_spawn_coverage(
        record,
        required_roles=task.child_roles,
    )
    strict_full_child_completion = bool(
        spawn_audit.get("accepted") is True
        and valid_child_roles == required_role_set
        and all(row.get("accepted") is True for row in child_reports)
        and len(runtime_children) >= required_children
        and all(
            str(row.get("state") or "") == "succeeded"
            for row in runtime_children
        )
    )
    partial_coordinator_safety_reasons = (
        _partial_coordinator_safety_reasons(
            record,
            task=task,
            artifact_validation=artifact_validation,
            worker_artifact_validation=worker_artifact_validation,
            runtime_summary=runtime_summary,
        )
        if child_acceptance_mode == "valid_subset_l0"
        else []
    )
    # ``valid_subset_l0`` is permission to recover an actually degraded
    # attempt. It must not downgrade a 4/4 (or otherwise complete) strict
    # child set merely because the campaign enabled the recovery policy.
    partial_fallback_used = bool(
        child_acceptance_mode == "valid_subset_l0"
        and not strict_full_child_completion
        and spawn_audit.get("accepted") is True
        and not partial_coordinator_safety_reasons
    )
    if partial_fallback_used:
        child_candidates = [
            _partial_l0_candidate(candidate) for candidate in child_candidates
        ]
    # Build consensus from independently observed child final messages, not
    # from the coordinator's restatement. The latter remains an audit-only
    # synthesis artifact and cannot invent a missing specialist.
    consensus = fuse_route_candidates(child_candidates, case_id=case_id, target_smiles=target_smiles)
    if partial_fallback_used:
        consensus = _cap_partial_consensus_to_l0(consensus)
    accepted_report_admissions = [
        dict(row.get("candidate_admission") or {})
        for row in child_reports
        if row.get("accepted") is True
    ]
    raw_candidate_count = sum(
        int((row.get("candidate_admission") or {}).get("raw_candidate_count") or 0)
        for row in child_reports
    )
    admitted_candidate_count = sum(
        int(row.get("admitted_candidate_count") or 0)
        for row in accepted_report_admissions
    )
    quarantined_candidate_count = sum(
        int(row.get("rejected_candidate_count") or 0)
        for row in accepted_report_admissions
    )
    discarded_with_rejected_reports_count = sum(
        int(
            (row.get("candidate_admission") or {}).get(
                "discarded_with_rejected_report_count"
            )
            or 0
        )
        for row in child_reports
    )
    filtered_child_roles = sorted(
        str(row.get("role") or "")
        for row in child_reports
        if row.get("accepted") is True
        and int(
            (row.get("candidate_admission") or {}).get(
                "rejected_candidate_count"
            )
            or 0
        )
        > 0
    )
    consensus_summary = dict(consensus.get("source_summary") or {})
    candidate_admission_reconciliation_reasons: list[str] = []
    if int(consensus_summary.get("candidate_count") or 0) != admitted_candidate_count:
        candidate_admission_reconciliation_reasons.append(
            "child_candidate_admitted_count_consensus_mismatch"
        )
    if (
        int(consensus_summary.get("rejected_count") or 0)
        != quarantined_candidate_count
    ):
        candidate_admission_reconciliation_reasons.append(
            "child_candidate_rejected_count_consensus_mismatch"
        )
    consensus_path = output_dir / "route_consensus.json"
    _write_json(consensus_path, consensus)
    proposals = consensus_to_blackboard_proposals(consensus)

    reasons: list[str] = []
    reasons.extend(partial_coordinator_safety_reasons)
    reasons.extend(candidate_admission_reconciliation_reasons)
    if not partial_fallback_used:
        if record.status != "accepted_draft":
            reasons.append(f"coordinator_status:{record.status}")
    if not artifact_validation.get("accepted"):
        reasons.extend(str(item) for item in artifact_validation.get("reasons") or [])
    if not worker_artifact_validation.get("accepted"):
        reasons.extend(
            str(item)
            for item in worker_artifact_validation.get("reasons") or []
        )
    if observed_children != required_children:
        reasons.append("required_child_agents_not_observed")
    reasons.extend(str(item) for item in spawn_audit.get("hard_reasons") or [])
    if not partial_fallback_used:
        if valid_child_roles != required_role_set:
            reasons.append("required_child_reports_not_valid")
        if any(row.get("accepted") is not True for row in child_reports):
            reasons.append("one_or_more_child_reports_rejected")
    elif len(valid_child_roles) < valid_child_quorum:
        reasons.append("valid_child_role_quorum_not_met")
    if not partial_fallback_used and (
        len(runtime_children) < required_children
        or any(
            str(row.get("state") or "") != "succeeded"
            for row in runtime_children
        )
    ):
        reasons.append("required_child_agents_not_succeeded")
    if not consensus.get("accepted"):
        reasons.append("no_valid_retrosynthesis_candidates")
    if not runtime_summary.get("consistent"):
        reasons.append("child_agent_runtime_reconciliation_failed")
    accepted = not reasons
    accepted_child_roles = sorted(valid_child_roles)
    degraded_child_roles = sorted(required_role_set - valid_child_roles)
    acceptance_tier = (
        "valid_subset_l0"
        if partial_fallback_used
        else "strict_all"
    )
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
        "worker_artifact_validation": worker_artifact_validation,
        "child_acceptance": {
            "contract_version": CHILD_ACCEPTANCE_CONTRACT_VERSION,
            "candidate_admission_contract_version": (
                "autoplanner.child_candidate_admission.v1"
            ),
            "mode": child_acceptance_mode,
            "acceptance_tier": acceptance_tier,
            "required_child_roles": list(task.child_roles),
            "observed_explicit_spawn_roles": list(
                spawn_audit.get("observed_roles") or []
            ),
            "valid_child_roles": accepted_child_roles,
            "degraded_child_roles": degraded_child_roles,
            "derived_valid_child_quorum": valid_child_quorum,
            "valid_child_role_count": len(valid_child_roles),
            "all_required_roles_explicitly_spawned": bool(
                spawn_audit.get("accepted")
            ),
            "partial_proposals_forced_to_l0": (
                partial_fallback_used
            ),
            "partial_fallback_used": partial_fallback_used,
            "strict_full_child_completion": strict_full_child_completion,
            "raw_candidate_count": raw_candidate_count,
            "admitted_candidate_count": admitted_candidate_count,
            "quarantined_candidate_count": quarantined_candidate_count,
            "discarded_with_rejected_reports_count": (
                discarded_with_rejected_reports_count
            ),
            "filtered_child_roles": filtered_child_roles,
            "candidate_admission_reconciliation_reasons": (
                candidate_admission_reconciliation_reasons
            ),
            "candidate_quarantine_does_not_trigger_partial_fallback": True,
        },
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
            "child_acceptance_mode": child_acceptance_mode,
            "child_acceptance_tier": acceptance_tier,
            "partial_fallback_used": partial_fallback_used,
            "candidate_quarantine_is_search_admission_only": True,
            "no_solved_claim": True,
        },
    }
    _write_json(output_dir / "team_report.json", report)
    return report


def campaign_closure_status(
    frontier_ledger: dict[str, Any],
    *,
    authoritative: bool,
    closure_objective: str = "benchmark_search",
    exploration_mode: str = "exhaustive",
) -> dict[str, Any]:
    """Return objective-aware route and exhaustive campaign completion.

    Generic ledger closure is deliberately never consumed here: its legacy
    aliases mean benchmark/search closure. Procurement and in-house objectives
    use their own authority planes so a benchmark hit cannot silently satisfy
    a stronger operational objective.
    """

    objective = _normalize_closure_objective(closure_objective)
    mode = _normalize_exploration_mode(exploration_mode)
    ledger = dict(frontier_ledger or {})
    summary = dict(ledger.get("summary") or {})
    molecules = {
        str(smiles): dict(row)
        for smiles, row in dict(ledger.get("molecules") or {}).items()
        if isinstance(row, dict)
    }
    edges = {
        str(signature): dict(row)
        for signature, row in dict(ledger.get("edges") or {}).items()
        if isinstance(row, dict)
    }
    leaf_rows = [
        row
        for row in molecules.values()
        if not list(
            dict(row.get("proposal") or {}).get("outgoing_edge_signatures") or []
        )
    ]
    all_reaction_edges_closed = bool(
        authoritative
        and all(
            dict(edge.get("reaction_proof") or {}).get("closed") is True
            for edge in edges.values()
        )
    )
    all_benchmark_leaves_closed = bool(
        authoritative
        and leaf_rows
        and all(dict(row.get("stock") or {}).get("closed") is True for row in leaf_rows)
    )
    all_procurement_leaves_closed = bool(
        authoritative
        and leaf_rows
        and all(
            dict(row.get("stock") or {}).get("procurement_boundary_closed") is True
            for row in leaf_rows
        )
    )
    all_in_house_leaves_closed = bool(
        authoritative
        and leaf_rows
        and all(
            "in_house_available"
            in {
                str(value)
                for value in dict(row.get("stock") or {}).get("boundary_types")
                or []
            }
            for row in leaf_rows
        )
    )
    in_house_any, in_house_all = _in_house_closure_fixed_point(ledger)
    route_by_objective = {
        "benchmark_search": summary.get("any_benchmark_route_closed") is True,
        "procurement": summary.get("any_procurement_route_closed") is True,
        "in_house": in_house_any,
    }
    exhaustive_by_objective = {
        "benchmark_search": summary.get("all_explored_benchmark_closed") is True,
        "procurement": summary.get("all_explored_procurement_closed") is True,
        "in_house": in_house_all,
    }
    route_solved = bool(authoritative and route_by_objective[objective])
    exhaustive_complete = bool(
        authoritative and exhaustive_by_objective[objective]
    )
    campaign_search_complete = bool(
        route_solved if mode == "first_solved" else exhaustive_complete
    )
    return {
        "closure_objective": objective,
        "exploration_mode": mode,
        "route_solved": route_solved,
        "campaign_search_complete": campaign_search_complete,
        "all_reaction_edges_closed": all_reaction_edges_closed,
        "all_benchmark_leaves_closed": all_benchmark_leaves_closed,
        "all_procurement_leaves_closed": all_procurement_leaves_closed,
        "all_in_house_leaves_closed": all_in_house_leaves_closed,
        "any_in_house_route_closed": bool(authoritative and in_house_any),
        "all_explored_in_house_closed": bool(authoritative and in_house_all),
        "selected_objective_all_explored_closed": exhaustive_complete,
    }


def _in_house_closure_fixed_point(
    frontier_ledger: dict[str, Any],
) -> tuple[bool, bool]:
    """Evaluate an in-house-only AND/OR fixed point from validated ledger rows."""

    ledger = dict(frontier_ledger or {})
    molecules = {
        str(smiles): dict(row)
        for smiles, row in dict(ledger.get("molecules") or {}).items()
        if isinstance(row, dict)
    }
    edges = {
        str(signature): dict(row)
        for signature, row in dict(ledger.get("edges") or {}).items()
        if isinstance(row, dict)
    }
    root = str(dict(ledger.get("root") or {}).get("canonical_smiles") or "")
    if not root or root not in molecules:
        return False, False
    outgoing = {
        smiles: [
            str(value)
            for value in dict(row.get("proposal") or {}).get(
                "outgoing_edge_signatures"
            )
            or []
            if str(value)
        ]
        for smiles, row in molecules.items()
    }
    boundary = {
        smiles: "in_house_available"
        in {
            str(value)
            for value in dict(row.get("stock") or {}).get("boundary_types") or []
        }
        for smiles, row in molecules.items()
    }
    any_closed = dict(boundary)
    all_closed = {
        smiles: bool(not outgoing.get(smiles) and boundary[smiles])
        for smiles in molecules
    }
    while True:
        next_any = dict(any_closed)
        next_all = dict(all_closed)
        for smiles in sorted(molecules):
            alternatives = outgoing.get(smiles) or []
            if not alternatives:
                continue
            next_any[smiles] = bool(
                boundary[smiles]
                or any(
                    dict(edges.get(signature, {}).get("reaction_proof") or {}).get(
                        "closed"
                    )
                    is True
                    and all(
                        any_closed.get(str(precursor), False)
                        for precursor in edges.get(signature, {}).get(
                            "precursor_smiles"
                        )
                        or []
                    )
                    for signature in alternatives
                    if signature in edges
                )
            )
            next_all[smiles] = bool(
                alternatives
                and all(
                    signature in edges
                    and dict(edges[signature].get("reaction_proof") or {}).get(
                        "closed"
                    )
                    is True
                    and all(
                        all_closed.get(str(precursor), False)
                        for precursor in edges[signature].get("precursor_smiles")
                        or []
                    )
                    for signature in alternatives
                )
            )
        if next_any == any_closed and next_all == all_closed:
            return bool(next_any[root]), bool(next_all[root])
        any_closed, all_closed = next_any, next_all


@_campaign_authority_single_writer
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
    closure_objective = _normalize_closure_objective(config.closure_objective)
    exploration_mode = _normalize_exploration_mode(config.exploration_mode)
    child_acceptance_mode = _normalize_child_acceptance_mode(
        config.child_acceptance_mode
    )
    max_depth = max(1, int(config.max_depth or 1))
    max_expansions = max(1, int(config.max_expansions or 1))
    max_expansions_per_invocation = min(
        max_expansions,
        max(1, int(config.max_expansions_per_invocation or 1)),
    )
    max_attempt_runs_per_invocation = max(
        1,
        int(config.max_attempt_runs_per_invocation or max_expansions_per_invocation),
    )
    max_attempt_runs = max(
        1,
        int(
            config.max_attempt_runs
            or max(max_expansions * 3, max_attempt_runs_per_invocation)
        ),
    )
    frontier_batch_size = max(1, int(config.frontier_batch_size or 1))
    root_run_dir = Path(run_dir).resolve()
    root_output_dir = root_run_dir / "codex_retrosynthesis_team"
    root_output_dir.mkdir(parents=True, exist_ok=True)
    root_smiles = _canonical_target_smiles(target_smiles)
    if not str(case_id or "").strip():
        raise ValueError("campaign case_id is required")
    if not root_smiles:
        raise ValueError("campaign target_smiles is invalid")
    identity_path = root_output_dir / "campaign_identity.json"
    campaign_identity = _ensure_campaign_identity(
        identity_path,
        case_id=case_id,
        target_smiles=root_smiles,
        root_output_dir=root_output_dir,
    )
    campaign_identity_sha256 = str(campaign_identity["content_sha256"])
    queue_store = PersistentFrontierQueue(root_output_dir / "frontier_queue")
    queue_store.migrate_legacy_benchmark_stock_authority(case_id)
    _validate_campaign_queue_identity(
        queue_store.list_jobs(case_id),
        case_id=case_id,
        target_smiles=root_smiles,
        campaign_identity_sha256=campaign_identity_sha256,
    )
    resolved_stock_provider, stock_authority = _campaign_stock_provider(
        config,
        stock_provider=stock_provider,
    )
    campaign_policy_path = root_output_dir / "campaign_policy.json"
    campaign_policy = _ensure_campaign_policy(
        campaign_policy_path,
        case_id=case_id,
        target_smiles=root_smiles,
        campaign_identity_sha256=campaign_identity_sha256,
        max_depth=max_depth,
        required_proof_level=2,
        stock_provider=resolved_stock_provider,
        stock_authority=stock_authority,
        coordinator_contract_version=RETROSYNTHESIS_COORDINATOR_CONTRACT_VERSION,
        child_roles=tuple(_normalized_roles(config.child_roles)),
        model=str(config.model or ""),
        closure_objective=closure_objective,
        exploration_mode=exploration_mode,
        child_acceptance_mode=child_acceptance_mode,
    )
    campaign_policy_sha256 = str(campaign_policy["content_sha256"])
    campaign_budget = _ensure_campaign_budget_envelope(
        root_output_dir=root_output_dir,
        case_id=case_id,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
        max_expansions=max_expansions,
        max_attempt_runs=max_attempt_runs,
    )
    max_expansions = int(campaign_budget["max_expansions"])
    max_attempt_runs = int(campaign_budget["max_attempt_runs"])
    scheduler = FrontierScheduler(queue_store, resolved_stock_provider)
    scheduler_stock_provider_instances = {
        provider.descriptor.provider_id: provider
        for provider in resolved_stock_provider
    }
    admitted_hyperedge_journal_root = root_output_dir / "admitted_hyperedges"
    admitted_hyperedge_events = load_external_hyperedge_events(
        admitted_hyperedge_journal_root,
        case_id=case_id,
        target_smiles=root_smiles,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    admitted_hyperedge_expansions = expansions_from_external_hyperedge_events(
        admitted_hyperedge_events,
        case_id=case_id,
    )
    campaign_state_path = root_output_dir / "campaign_state.json"
    # ``campaign_state.json`` is a disposable projection.  Recovery always
    # starts empty and replays only queue-fenced, succeeded immutable commits;
    # a stale or attacker-rehashed cache therefore cannot consume budget.
    expansions: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    root_report = _load_root_report_for_campaign(
        root_output_dir / "team_report.json",
        case_id=case_id,
        target_smiles=root_smiles,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    graph = _assemble_canonical_route_consensus_graph(
        [*expansions, *admitted_hyperedge_expansions],
        case_id=case_id,
        target_smiles=root_smiles,
        max_depth=max_depth,
    )
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
            "campaign_identity_sha256": campaign_identity_sha256,
            "campaign_policy_sha256": campaign_policy_sha256,
            "campaign_root_smiles": root_smiles,
            "proposal_expansion_allowed": True,
            "proposal_expansion_gate": {
                "schema_version": "proposal_expansion_gate.v1",
                "status": "campaign_root",
                "validated_parent_step_ids": [],
            },
        },
    )
    _validate_campaign_queue_identity(
        queue_store.list_jobs(case_id),
        case_id=case_id,
        target_smiles=root_smiles,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    recovery_errors = _reconcile_expansion_commits(
        queue=queue_store,
        run_id=case_id,
        root_output_dir=root_output_dir,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_target_smiles=root_smiles,
        expansions=expansions,
        runs=run_summaries,
    )
    attempt_records = _load_campaign_attempt_ledger(
        root_output_dir=root_output_dir,
        case_id=case_id,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_target_smiles=root_smiles,
        jobs=queue_store.list_jobs(case_id),
    )
    if _close_recovered_campaign_attempts(
        root_output_dir=root_output_dir,
        runs=run_summaries,
        attempt_records=attempt_records,
    ):
        attempt_records = _load_campaign_attempt_ledger(
            root_output_dir=root_output_dir,
            case_id=case_id,
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_target_smiles=root_smiles,
            jobs=queue_store.list_jobs(case_id),
        )
    _merge_attempt_run_projection(run_summaries, attempt_records)
    committed_root_report = _root_report_from_reconciled_runs(run_summaries)
    if committed_root_report:
        root_report = committed_root_report
    elif root_report.get("accepted") is True:
        # An accepted mutable root report without a replayable succeeded
        # commit is not a campaign result and must not survive recovery.
        root_report = {}
    expanded_smiles = {
        _canonical_target_smiles(row.get("target_smiles"))
        for row in run_summaries
        if row.get("proposal_expansion_recorded") is True
    }
    expanded_smiles.discard("")
    graph = _assemble_canonical_route_consensus_graph(
        [*expansions, *admitted_hyperedge_expansions],
        case_id=case_id,
        target_smiles=target_smiles,
        max_depth=max_depth,
    )
    reaction_proof_state_path = root_output_dir / "reaction_proof_state.json"
    reaction_proof_state = _reconcile_reaction_proof_state(
        graph,
        path=reaction_proof_state_path,
        configured_proofs=config.reaction_proofs,
        configured_reports=config.reaction_proof_reports,
    )
    validated_parent_step_ids = _validated_parent_step_ids(reaction_proof_state)
    _validate_current_proposal_expansion_gates(
        queue_store.list_jobs(case_id),
        validated_parent_step_ids=validated_parent_step_ids,
    )
    _submit_graph_frontiers(
        graph=graph,
        scheduler=scheduler,
        config=config,
        case_id=case_id,
        expanded_smiles=expanded_smiles,
        max_depth=max_depth,
        frontier_batch_size=frontier_batch_size,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
        campaign_target_smiles=root_smiles,
        validated_parent_step_ids=validated_parent_step_ids,
    )
    _write_campaign_state(
        campaign_state_path,
        case_id=case_id,
        target_smiles=root_smiles,
        expansions=expansions,
        runs=run_summaries,
        attempt_records=attempt_records,
        max_attempt_runs=max_attempt_runs,
        attempt_ledger_ref=str(root_output_dir / "campaign_attempts"),
    )

    invocation_attempt_run_count = 0
    invocation_accepted_expansion_count = 0
    queue_maintenance_count = 0
    stop_reason = ""
    while True:
        objective_route_solved_this_iteration = False
        if len(expansions) >= max_expansions:
            stop_reason = "campaign_accepted_expansion_budget_exhausted"
            break
        if invocation_accepted_expansion_count >= max_expansions_per_invocation:
            stop_reason = "invocation_accepted_expansion_cap_reached"
            break
        if invocation_attempt_run_count >= max_attempt_runs_per_invocation:
            stop_reason = "invocation_attempt_run_cap_reached"
            break
        lease_seconds = max(30.0, float(config.frontier_lease_seconds or 1800.0))
        claimed: list[FrontierJob] = []
        attempt_budget_exhausted = False
        attempt_started: dict[str, Any] | None = None
        frontier: dict[str, Any] = {}
        frontier_smiles = ""
        depth = 0
        expansion_case_id = ""
        expansion_allowed = False
        with _campaign_attempt_ledger_lock(root_output_dir):
            attempt_records = _load_campaign_attempt_ledger(
                root_output_dir=root_output_dir,
                case_id=case_id,
                campaign_identity_sha256=campaign_identity_sha256,
                campaign_target_smiles=root_smiles,
                jobs=queue_store.list_jobs(case_id),
            )
            if len(attempt_records) >= max_attempt_runs:
                attempt_budget_exhausted = True
            else:
                claimed = queue_store.claim(
                    case_id,
                    worker_id=f"codex-campaign:{case_id}",
                    limit=1,
                    lease_seconds=lease_seconds,
                    trusted_stock_provider_instances=(
                        scheduler_stock_provider_instances
                    ),
                )
                if claimed:
                    candidate_job = claimed[0]
                    frontier = dict(candidate_job.metadata)
                    frontier_smiles = candidate_job.frontier_smiles
                    depth = int(frontier.get("depth") or 0)
                    expansion_allowed = bool(
                        frontier.get("proposal_expansion_allowed", depth < max_depth)
                    )
                    if (
                        frontier_smiles not in expanded_smiles
                        and expansion_allowed
                        and depth < max_depth
                    ):
                        digest = hashlib.sha256(
                            frontier_smiles.encode("utf-8")
                        ).hexdigest()[:12]
                        expansion_case_id = (
                            case_id
                            if depth == 0
                            else f"{case_id}:frontier:d{depth}:{digest}"
                        )
                        attempt_started = _write_campaign_attempt_started(
                            root_output_dir=root_output_dir,
                            case_id=case_id,
                            campaign_identity_sha256=campaign_identity_sha256,
                            campaign_target_smiles=root_smiles,
                            job=candidate_job,
                            expansion_case_id=expansion_case_id,
                            depth=depth,
                        )
                        attempt_records = _load_campaign_attempt_ledger(
                            root_output_dir=root_output_dir,
                            case_id=case_id,
                            campaign_identity_sha256=campaign_identity_sha256,
                            campaign_target_smiles=root_smiles,
                            jobs=queue_store.list_jobs(case_id),
                        )
        if attempt_budget_exhausted:
            stop_reason = "campaign_attempt_run_budget_exhausted"
            break
        if not claimed:
            retry_delay = _next_retry_delay(queue_store.list_jobs(case_id))
            retry_wait_limit = max(0.0, float(config.frontier_retry_wait_seconds or 0.0))
            if retry_delay is not None and retry_delay <= retry_wait_limit:
                time.sleep(max(0.01, retry_delay))
                continue
            stop_reason = "frontier_retry_wait" if retry_delay is not None else "frontier_queue_idle"
            break
        job = claimed[0]
        if frontier_smiles in expanded_smiles or not expansion_allowed or depth >= max_depth:
            reason = (
                "frontier_already_proposal_expanded"
                if frontier_smiles in expanded_smiles
                else "depth_or_route_boundary_reached_after_stock_audit"
            )
            queue_store.fail(
                case_id,
                job.job_id,
                lease_token=job.lease_token,
                reason=reason,
                retryable=False,
                retry_base_seconds=0.0,
                retry_max_seconds=0.0,
            )
            queue_maintenance_count += 1
            continue
        if attempt_started is None:
            raise RuntimeError("campaign attempt was not durably started before Agent work")
        invocation_attempt_run_count += 1
        digest = hashlib.sha256(frontier_smiles.encode("utf-8")).hexdigest()[:12]
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
            failure_summary = {
                "case_id": expansion_case_id,
                "target_smiles": frontier_smiles,
                "depth": depth,
                "accepted": False,
                "agent_task_succeeded": False,
                "team_report_accepted": False,
                "proof_closed": False,
                "frontier_job_id": job.job_id,
                "frontier_job_state": current.state.value if current else "missing",
                "reasons": [failure_reason],
                "lease_heartbeat_errors": heartbeat_failures,
                "result_quarantined": bool(heartbeat_failures),
                "proposal_expansion_recorded": False,
                "campaign_attempt_id": attempt_started["attempt_id"],
                "job_attempt": job.attempt,
            }
            terminal_event = _write_campaign_attempt_terminal(
                root_output_dir=root_output_dir,
                started_event=attempt_started,
                terminal_status="team_runtime_error",
                summary=failure_summary,
            )
            failure_summary["attempt_terminal_ref"] = str(
                _campaign_attempt_event_dir(
                    root_output_dir,
                    str(attempt_started["attempt_id"]),
                )
                / "terminal.json"
            )
            failure_summary["attempt_terminal_status"] = terminal_event[
                "terminal_status"
            ]
            run_summaries.append(failure_summary)
            attempt_records = _load_campaign_attempt_ledger(
                root_output_dir=root_output_dir,
                case_id=case_id,
                campaign_identity_sha256=campaign_identity_sha256,
                campaign_target_smiles=root_smiles,
                jobs=queue_store.list_jobs(case_id),
            )
            _write_campaign_state(
                campaign_state_path,
                case_id=case_id,
                target_smiles=target_smiles,
                expansions=expansions,
                runs=run_summaries,
                attempt_records=attempt_records,
                max_attempt_runs=max_attempt_runs,
                attempt_ledger_ref=str(root_output_dir / "campaign_attempts"),
            )
            continue
        team_report_ref = expansion_run_dir / "codex_retrosynthesis_team" / "team_report.json"
        team_report, cycle_audit = _enforce_campaign_cycle_boundary(
            team_report,
            frontier=frontier,
            frontier_smiles=frontier_smiles,
        )
        if cycle_audit.get("rejected_proposal_count"):
            consensus_ref = Path(str(team_report.get("route_consensus_ref") or ""))
            try:
                consensus_ref.resolve().relative_to(
                    (expansion_run_dir / "codex_retrosynthesis_team").resolve()
                )
            except (OSError, ValueError):
                pass
            else:
                _write_json(consensus_ref, dict(team_report.get("route_consensus") or {}))
            _write_json(team_report_ref, team_report)
        summary = {
            "case_id": expansion_case_id,
            "target_smiles": frontier_smiles,
            "depth": depth,
            "accepted": False,
            "agent_task_succeeded": True,
            "team_report_accepted": bool(team_report.get("accepted")),
            "proof_closed": False,
            "frontier_job_id": job.job_id,
            "team_report_ref": str(team_report_ref),
            "route_consensus_ref": str(team_report.get("route_consensus_ref") or ""),
            "reasons": [str(item) for item in team_report.get("reasons") or []],
            "proposal_expansion_recorded": False,
            "lease_heartbeat_errors": heartbeat_failures,
            "campaign_attempt_id": attempt_started["attempt_id"],
            "job_attempt": job.attempt,
        }
        if heartbeat_failures:
            current = queue_store.get(case_id, job.job_id)
            summary["frontier_job_state"] = current.state.value if current else "missing"
            summary["result_quarantined"] = True
            summary["reasons"] = sorted(
                {*summary["reasons"], "lease_fencing_lost_result_quarantined"}
            )
            _write_campaign_attempt_terminal(
                root_output_dir=root_output_dir,
                started_event=attempt_started,
                terminal_status="lease_fencing_lost",
                summary=summary,
            )
            run_summaries.append(summary)
            attempt_records = _load_campaign_attempt_ledger(
                root_output_dir=root_output_dir,
                case_id=case_id,
                campaign_identity_sha256=campaign_identity_sha256,
                campaign_target_smiles=root_smiles,
                jobs=queue_store.list_jobs(case_id),
            )
            _write_campaign_state(
                campaign_state_path,
                case_id=case_id,
                target_smiles=target_smiles,
                expansions=expansions,
                runs=run_summaries,
                attempt_records=attempt_records,
                max_attempt_runs=max_attempt_runs,
                attempt_ledger_ref=str(root_output_dir / "campaign_attempts"),
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
            committed_summary = {
                **summary,
                "accepted": True,
                "proposal_expansion_recorded": True,
            }
            try:
                expansion_commit_path = _write_expansion_commit(
                    root_output_dir=root_output_dir,
                    case_id=case_id,
                    campaign_identity_sha256=campaign_identity_sha256,
                    campaign_target_smiles=root_smiles,
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
                _write_campaign_attempt_terminal(
                    root_output_dir=root_output_dir,
                    started_event=attempt_started,
                    terminal_status="prepared_commit_queue_fencing_lost",
                    summary=summary,
                )
                run_summaries.append(summary)
                attempt_records = _load_campaign_attempt_ledger(
                    root_output_dir=root_output_dir,
                    case_id=case_id,
                    campaign_identity_sha256=campaign_identity_sha256,
                    campaign_target_smiles=root_smiles,
                    jobs=queue_store.list_jobs(case_id),
                )
                _write_campaign_state(
                    campaign_state_path,
                    case_id=case_id,
                    target_smiles=target_smiles,
                    expansions=expansions,
                    runs=run_summaries,
                    attempt_records=attempt_records,
                    max_attempt_runs=max_attempt_runs,
                    attempt_ledger_ref=str(root_output_dir / "campaign_attempts"),
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
                _write_campaign_attempt_terminal(
                    root_output_dir=root_output_dir,
                    started_event=attempt_started,
                    terminal_status="expansion_commit_error",
                    summary=summary,
                )
                run_summaries.append(summary)
                attempt_records = _load_campaign_attempt_ledger(
                    root_output_dir=root_output_dir,
                    case_id=case_id,
                    campaign_identity_sha256=campaign_identity_sha256,
                    campaign_target_smiles=root_smiles,
                    jobs=queue_store.list_jobs(case_id),
                )
                _write_campaign_state(
                    campaign_state_path,
                    case_id=case_id,
                    target_smiles=target_smiles,
                    expansions=expansions,
                    runs=run_summaries,
                    attempt_records=attempt_records,
                    max_attempt_runs=max_attempt_runs,
                    attempt_ledger_ref=str(root_output_dir / "campaign_attempts"),
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
            invocation_accepted_expansion_count += 1
            summary["proposal_expansion_recorded"] = True
            graph = _assemble_canonical_route_consensus_graph(
                [*expansions, *admitted_hyperedge_expansions],
                case_id=case_id,
                target_smiles=target_smiles,
                max_depth=max_depth,
            )
            reaction_proof_state = _reconcile_reaction_proof_state(
                graph,
                path=reaction_proof_state_path,
                configured_proofs=config.reaction_proofs,
                configured_reports=config.reaction_proof_reports,
            )
            validated_parent_step_ids = _validated_parent_step_ids(
                reaction_proof_state
            )
            _validate_current_proposal_expansion_gates(
                queue_store.list_jobs(case_id),
                validated_parent_step_ids=validated_parent_step_ids,
            )
            _submit_graph_frontiers(
                graph=graph,
                scheduler=scheduler,
                config=config,
                case_id=case_id,
                expanded_smiles=expanded_smiles,
                max_depth=max_depth,
                frontier_batch_size=frontier_batch_size,
                campaign_identity_sha256=campaign_identity_sha256,
                campaign_policy_sha256=campaign_policy_sha256,
                campaign_target_smiles=root_smiles,
                validated_parent_step_ids=validated_parent_step_ids,
            )
            if exploration_mode == "first_solved":
                current_providers = {
                    provider.descriptor.provider_id: provider
                    for provider in resolved_stock_provider
                }
                current_ledger = project_frontier_ledger(
                    graph,
                    queue_store.snapshot(case_id),
                    reaction_proof_state,
                    required_reaction_proof_level=2,
                    trusted_stock_provider_instances=current_providers,
                    campaign_policy_sha256=campaign_policy_sha256,
                )
                current_validation = validate_frontier_ledger(
                    current_ledger,
                    trusted_stock_provider_instances=current_providers,
                )
                current_inputs = dict(
                    current_ledger.get("input_validation") or {}
                )
                current_authoritative = bool(
                    not current_validation
                    and all(
                        isinstance(current_inputs.get(key), dict)
                        and current_inputs[key].get("valid") is True
                        for key in (
                            "graph",
                            "frontier_queue",
                            "reaction_proof_state",
                        )
                    )
                )
                objective_route_solved_this_iteration = bool(
                    campaign_closure_status(
                        current_ledger,
                        authoritative=current_authoritative,
                        closure_objective=closure_objective,
                        exploration_mode=exploration_mode,
                    )["route_solved"]
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
        _write_campaign_attempt_terminal(
            root_output_dir=root_output_dir,
            started_event=attempt_started,
            terminal_status=(
                "accepted_expansion_committed"
                if summary.get("proposal_expansion_recorded") is True
                else "team_report_rejected"
            ),
            summary=summary,
        )
        run_summaries.append(summary)
        attempt_records = _load_campaign_attempt_ledger(
            root_output_dir=root_output_dir,
            case_id=case_id,
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_target_smiles=root_smiles,
            jobs=queue_store.list_jobs(case_id),
        )
        _write_campaign_state(
            campaign_state_path,
            case_id=case_id,
            target_smiles=target_smiles,
            expansions=expansions,
            runs=run_summaries,
            attempt_records=attempt_records,
            max_attempt_runs=max_attempt_runs,
            attempt_ledger_ref=str(root_output_dir / "campaign_attempts"),
        )
        if objective_route_solved_this_iteration:
            stop_reason = "objective_route_solved"
            break

    attempt_records = _load_campaign_attempt_ledger(
        root_output_dir=root_output_dir,
        case_id=case_id,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_target_smiles=root_smiles,
        jobs=queue_store.list_jobs(case_id),
    )
    _merge_attempt_run_projection(run_summaries, attempt_records)
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
    queue_state_counts = {
        state.value: sum(1 for row in queue_jobs if row.state == state)
        for state in FrontierJobState
    }
    terminal_smiles = _campaign_terminal_smiles(graph) or [root_smiles]
    reaction_proof_state_path = root_output_dir / "reaction_proof_state.json"
    reaction_proof_state = _reconcile_reaction_proof_state(
        graph,
        path=reaction_proof_state_path,
        configured_proofs=config.reaction_proofs,
        configured_reports=config.reaction_proof_reports,
    )
    open_reaction_proofs = _open_reaction_proof_frontiers(reaction_proof_state)
    trusted_stock_providers = {
        provider.descriptor.provider_id: provider
        for provider in resolved_stock_provider
    }
    completeness = assess_frontier_completeness(
        terminal_smiles,
        queue_jobs,
        open_proof_frontiers=open_reaction_proofs,
        required_proof_level=2,
        trusted_stock_provider_instances=trusted_stock_providers,
    )
    queue_snapshot = queue_store.snapshot(case_id)
    frontier_ledger = project_frontier_ledger(
        graph,
        queue_snapshot,
        reaction_proof_state,
        required_reaction_proof_level=2,
        trusted_stock_provider_instances=trusted_stock_providers,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    frontier_ledger_validation = validate_frontier_ledger(
        frontier_ledger,
        trusted_stock_provider_instances=trusted_stock_providers,
    )
    ledger_inputs = dict(frontier_ledger.get("input_validation") or {})
    frontier_ledger_authoritative = bool(
        not frontier_ledger_validation
        and all(
            isinstance(ledger_inputs.get(key), dict)
            and ledger_inputs[key].get("valid") is True
            for key in ("graph", "frontier_queue", "reaction_proof_state")
        )
    )
    closure_status = campaign_closure_status(
        frontier_ledger,
        authoritative=frontier_ledger_authoritative,
        closure_objective=closure_objective,
        exploration_mode=exploration_mode,
    )
    graph_complete = closure_status["campaign_search_complete"] is True
    frontier_ledger_path = root_output_dir / "frontier_ledger.json"
    _write_json(frontier_ledger_path, frontier_ledger)
    if graph_complete:
        stop_reason = "graph_proof_complete"
    graph_path = root_output_dir / "route_consensus_graph.json"
    _write_json(graph_path, graph)
    campaign = {
        "schema_version": CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA,
        "root_case_id": case_id,
        "campaign_identity": campaign_identity,
        "campaign_identity_ref": str(identity_path),
        "campaign_identity_sha256": campaign_identity_sha256,
        "campaign_policy": campaign_policy,
        "campaign_policy_ref": str(campaign_policy_path),
        "campaign_policy_sha256": campaign_policy_sha256,
        "admitted_hyperedge_journal_ref": str(
            admitted_hyperedge_journal_root
        ),
        "admitted_hyperedge_event_count": len(admitted_hyperedge_events),
        "admitted_hyperedge_expansion_count": len(
            admitted_hyperedge_expansions
        ),
        "admitted_hyperedge_event_refs": sorted(
            str(row.get("event_ref") or "")
            for row in admitted_hyperedge_events
            if str(row.get("event_ref") or "")
        ),
        **closure_status,
        "campaign_budget": campaign_budget,
        "campaign_budget_ref": str(root_output_dir / "campaign_budget.json"),
        "max_depth": max_depth,
        "max_expansions": max_expansions,
        "max_expansions_per_invocation": max_expansions_per_invocation,
        "max_attempt_runs_per_invocation": max_attempt_runs_per_invocation,
        "max_attempt_runs": max_attempt_runs,
        "remaining_attempt_runs": max(0, max_attempt_runs - len(attempt_records)),
        "attempt_ledger_ref": str(root_output_dir / "campaign_attempts"),
        "attempt_ledger": [
            {
                "attempt_id": row["attempt_id"],
                "started_ref": row["started_ref"],
                "terminal_ref": row["terminal_ref"],
                "terminal_status": (
                    str((row.get("terminal") or {}).get("terminal_status") or "")
                ),
            }
            for row in attempt_records
        ],
        # Backward-compatible field: this is now the number of unique durable
        # proposal expansions, not the number of Agent attempts.
        "expansion_run_count": len(expansions),
        "attempt_run_count": len(attempt_records),
        "terminal_attempt_run_count": sum(
            1 for row in attempt_records if isinstance(row.get("terminal"), dict)
        ),
        "invocation_attempt_run_count": invocation_attempt_run_count,
        "invocation_accepted_expansion_count": invocation_accepted_expansion_count,
        "queue_maintenance_count": queue_maintenance_count,
        "unique_frontier_run_count": len(
            {str(row.get("frontier_job_id") or "") for row in run_summaries}
        ),
        "accepted_expansion_count": len(expansions),
        "team_accepted_attempt_count": sum(
            1 for row in run_summaries if row.get("team_report_accepted") is True
        ),
        "proof_closed_attempt_count": sum(
            1 for row in run_summaries if row.get("proof_closed") is True
        ),
        "stop_reason": stop_reason or "frontier_queue_idle",
        "resumable": bool(
            not graph_complete
            and (
                queue_state_counts.get("pending", 0)
                or queue_state_counts.get("retry_wait", 0)
                or open_reaction_proofs
            )
        ),
        "resume_guidance": _campaign_resume_guidance(
            stop_reason=stop_reason or "frontier_queue_idle",
            accepted_expansion_count=len(expansions),
            max_expansions=max_expansions,
            queue_state_counts=queue_state_counts,
            open_reaction_proof_count=len(open_reaction_proofs),
        ),
        "queue_state_counts": queue_state_counts,
        "graph_complete": graph_complete,
        "proposal_graph_exhausted": not remaining_frontier,
        "remaining_frontier": remaining_frontier,
        "frontier_completeness": completeness.to_dict(),
        "frontier_queue": queue_snapshot,
        "frontier_queue_ref": str(root_output_dir / "frontier_queue"),
        "frontier_ledger": frontier_ledger,
        "frontier_ledger_ref": str(frontier_ledger_path),
        "frontier_ledger_validation_reasons": frontier_ledger_validation,
        "frontier_ledger_authoritative": frontier_ledger_authoritative,
        "stock_authority": stock_authority,
        "reaction_proof_state": reaction_proof_state,
        "reaction_proof_state_ref": str(reaction_proof_state_path),
        "runs": run_summaries,
        "recovery_errors": recovery_errors,
        "resumable_at": _next_retry_available_at(queue_jobs),
        "semantics": {
            "frontier_reexpanded_by_direct_codex_teams": True,
            "persistent_stock_first_frontier_scheduler": True,
            "fenced_expansion_commit_before_queue_completion": True,
            "child_frontiers_published_only_after_parent_queue_commit": True,
            "campaign_state_is_rebuildable_atomic_cache": True,
            "campaign_identity_is_immutable_and_content_addressed": True,
            "campaign_policy_is_immutable_and_authority_bound": True,
            "campaign_budget_growth_is_append_only_and_monotonic": True,
            "recovery_replays_only_fenced_succeeded_commits": True,
            "recovery_replays_host_admitted_external_hyperedges": True,
            "external_hyperedge_events_carry_no_proof_stock_or_completion_authority": True,
            "prepared_commit_outbox_is_crash_recoverable": True,
            "lease_heartbeat_enabled": True,
            "queue_exhaustion_is_not_route_completion": True,
            "agent_task_success_is_not_proof_closure": True,
            "attempt_budget_is_independent_of_accepted_expansion_budget": True,
            "attempt_budget_is_restored_from_immutable_started_events": True,
            "non_root_agent_work_requires_current_host_l2_parent_proof": True,
            "campaign_is_resumable_across_invocations": True,
            "depth_boundary_leaves_are_stock_audited": True,
            "reaction_proof_state_is_externally_consumable": True,
            "reaction_validation_required_for_graph_complete": True,
            "graph_complete_is_campaign_search_complete_compatibility_alias": True,
            "campaign_completion_uses_selected_closure_objective": True,
            "exhaustive_mode_does_not_stop_at_first_solved_route": True,
            "advisory_only": True,
            "no_solved_claim": True,
        },
    }
    root_report["campaign"] = campaign
    root_report["campaign_identity_sha256"] = campaign_identity_sha256
    root_report["campaign_identity_ref"] = str(identity_path)
    root_report["campaign_policy_sha256"] = campaign_policy_sha256
    root_report["campaign_policy_ref"] = str(campaign_policy_path)
    root_report["proposal_accepted"] = root_report.get("accepted") is True
    root_report["proof_closed"] = graph_complete
    root_report.update(
        {
            key: closure_status[key]
            for key in (
                "closure_objective",
                "exploration_mode",
                "route_solved",
                "campaign_search_complete",
                "all_reaction_edges_closed",
                "all_benchmark_leaves_closed",
                "all_procurement_leaves_closed",
                "all_in_house_leaves_closed",
            )
        }
    )
    root_report["accepted_semantics"] = "advisory_proposal_artifact_only"
    root_report["route_consensus_graph_ref"] = str(graph_path)
    root_report["route_consensus_graph"] = graph
    root_report["route_consensus_expansions"] = expansions
    root_report["route_expansion_count"] = len(expansions)
    _write_json(root_output_dir / "team_report.json", root_report)
    return root_report


@_campaign_authority_single_writer
def reconcile_codex_campaign_proof_state(
    *,
    graph: dict[str, Any],
    run_dir: str | Path,
    case_id: str = "",
    reaction_proofs: dict[str, dict[str, Any]] | None = None,
    reaction_proof_reports: list[dict[str, Any]] | None = None,
    stock_evidence: list[dict[str, Any]] | None = None,
    required_proof_level: int = 2,
    campaign_config: RetrosynthesisTeamConfig | None = None,
    stock_provider: StockProvider | None = None,
    external_hyperedge_admission_receipts: Mapping[
        str,
        Iterable[Mapping[str, Any]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Refresh campaign proof closure without invoking proposal agents.

    This is the closeout bridge used after a controller materializes new edge
    candidates.  It replays the current host reaction verifier, reads the
    existing stock-first queue, and never consumes an expansion/attempt budget.
    """

    if graph.get("schema_version") != "route_consensus_graph.v1":
        raise ValueError("proof reconciliation requires route_consensus_graph.v1")
    resolved_case_id = str(case_id or graph.get("case_id") or "").strip()
    if not resolved_case_id or (
        str(graph.get("case_id") or "")
        and str(graph.get("case_id") or "") != resolved_case_id
    ):
        raise ValueError("proof reconciliation graph case identity mismatch")
    target_smiles = _canonical_target_smiles(graph.get("target_smiles"))
    if not target_smiles:
        raise ValueError("proof reconciliation graph target is invalid")
    root_output_dir = Path(run_dir).resolve() / "codex_retrosynthesis_team"
    root_output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = root_output_dir / "campaign_identity.json"
    identity = _ensure_campaign_identity(
        identity_path,
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        root_output_dir=root_output_dir,
    )
    identity_sha256 = str(identity["content_sha256"])
    sync_config = campaign_config or RetrosynthesisTeamConfig()
    sync_closure_objective = _normalize_closure_objective(
        sync_config.closure_objective
    )
    sync_exploration_mode = _normalize_exploration_mode(
        sync_config.exploration_mode
    )
    sync_child_acceptance_mode = _normalize_child_acceptance_mode(
        sync_config.child_acceptance_mode
    )
    sync_max_depth = max(
        1,
        int(
            (graph.get("limits") or {}).get("max_depth")
            or sync_config.max_depth
            or 1
        ),
    )
    policy_path = root_output_dir / "campaign_policy.json"
    resolved_provider: tuple[StockProvider, ...] | None = None
    stock_authority: dict[str, Any] = {}
    if campaign_config is not None or stock_provider is not None or not policy_path.exists():
        resolved_provider, stock_authority = _campaign_stock_provider(
            sync_config,
            stock_provider=stock_provider,
        )
        campaign_policy = _ensure_campaign_policy(
            policy_path,
            case_id=resolved_case_id,
            target_smiles=target_smiles,
            campaign_identity_sha256=identity_sha256,
            max_depth=sync_max_depth,
            required_proof_level=2,
            stock_provider=resolved_provider,
            stock_authority=stock_authority,
            coordinator_contract_version=RETROSYNTHESIS_COORDINATOR_CONTRACT_VERSION,
            child_roles=tuple(_normalized_roles(sync_config.child_roles)),
            model=str(sync_config.model or ""),
            closure_objective=sync_closure_objective,
            exploration_mode=sync_exploration_mode,
            child_acceptance_mode=sync_child_acceptance_mode,
        )
    else:
        campaign_policy = _load_campaign_policy(
            policy_path,
            case_id=resolved_case_id,
            target_smiles=target_smiles,
            campaign_identity_sha256=identity_sha256,
        )
        if int(campaign_policy.get("max_depth") or 0) != sync_max_depth:
            raise ValueError("proof reconciliation graph depth violates campaign policy")
        resolved_provider, stock_authority = (
            _rehydrate_stock_providers_from_campaign_policy(campaign_policy)
        )
        if (
            stock_authority.get("rehydration_required") is True
            and stock_authority.get("available") is not True
        ):
            raise ValueError(
                "campaign policy stock provider rehydration failed:"
                + ",".join(str(item) for item in stock_authority.get("reasons") or [])
            )
    closure_objective = _normalize_closure_objective(
        campaign_policy.get("closure_objective") or sync_closure_objective
    )
    exploration_mode = _normalize_exploration_mode(
        campaign_policy.get("exploration_mode") or sync_exploration_mode
    )
    campaign_policy_sha256 = str(campaign_policy["content_sha256"])
    queue = PersistentFrontierQueue(root_output_dir / "frontier_queue")
    queue.migrate_legacy_benchmark_stock_authority(resolved_case_id)
    jobs = queue.list_jobs(resolved_case_id)
    _validate_campaign_queue_identity(
        jobs,
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        campaign_identity_sha256=identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    # The controller's fused graph is mutable orchestration state.  Before it
    # can change proof state or the durable frontier queue, recover the Codex
    # commit authority and append every additional host-admitted exact edge to
    # an immutable, campaign-bound outbox.  A crash after this point can replay
    # the edge without invoking an external producer again; a failed/tampered
    # event aborts before the graph can affect either downstream authority.
    committed_expansions: list[dict[str, Any]] = []
    committed_runs: list[dict[str, Any]] = []
    journal_commit_recovery_errors = _reconcile_expansion_commits(
        queue=queue,
        run_id=resolved_case_id,
        root_output_dir=root_output_dir,
        campaign_identity_sha256=identity_sha256,
        campaign_target_smiles=target_smiles,
        expansions=committed_expansions,
        runs=committed_runs,
    )
    admitted_hyperedge_journal_root = root_output_dir / "admitted_hyperedges"
    admitted_hyperedge_events = load_external_hyperedge_events(
        admitted_hyperedge_journal_root,
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        campaign_identity_sha256=identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    admitted_hyperedge_expansions = expansions_from_external_hyperedge_events(
        admitted_hyperedge_events,
        case_id=resolved_case_id,
    )
    durable_graph = _assemble_canonical_route_consensus_graph(
        [*committed_expansions, *admitted_hyperedge_expansions],
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        max_depth=sync_max_depth,
    )
    admitted_hyperedge_journal = record_external_hyperedges(
        admitted_hyperedge_journal_root,
        graph,
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        campaign_identity_sha256=identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
        known_exact_edge_signatures=graph_exact_edge_signatures(durable_graph),
        admission_receipts=external_hyperedge_admission_receipts,
    )
    # Recording is only the prepare boundary.  Reload the complete immutable
    # event set and assemble it with every queue-fenced Codex commit before any
    # proof, gate, queue, ledger, or frontier calculation.  The caller graph is
    # advisory input and may omit a journaled edge on a later process/round.
    admitted_hyperedge_events = load_external_hyperedge_events(
        admitted_hyperedge_journal_root,
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        campaign_identity_sha256=identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    admitted_hyperedge_expansions = expansions_from_external_hyperedge_events(
        admitted_hyperedge_events,
        case_id=resolved_case_id,
    )
    canonical_expansions = [
        *committed_expansions,
        *admitted_hyperedge_expansions,
    ]
    caller_graph_summary = _caller_advisory_graph_summary(graph)
    graph = _assemble_canonical_route_consensus_graph(
        canonical_expansions,
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        max_depth=sync_max_depth,
        max_route_hypotheses=max(
            24,
            int((graph.get("limits") or {}).get("max_route_hypotheses") or 24),
        ),
    )
    jobs = queue.list_jobs(resolved_case_id)
    _validate_campaign_queue_identity(
        jobs,
        case_id=resolved_case_id,
        target_smiles=target_smiles,
        campaign_identity_sha256=identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    proof_state_path = root_output_dir / "reaction_proof_state.json"
    proof_state = _reconcile_reaction_proof_state(
        graph,
        path=proof_state_path,
        configured_proofs=dict(reaction_proofs or {}),
        configured_reports=list(reaction_proof_reports or []),
    )
    validated_parent_step_ids = _validated_parent_step_ids(proof_state)
    # The fused graph may discover another inbound reaction for a molecule
    # after its queue job was created.  Synchronize that monotonic parent set
    # before replaying the gate invariant; validating the stale queue snapshot
    # first would reject a perfectly current L2 proof solely because its edge
    # id arrived later.
    enabled_job_count = _enable_existing_proven_frontiers(
        graph=graph,
        queue=queue,
        case_id=resolved_case_id,
        max_depth=sync_max_depth,
        campaign_identity_sha256=identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
        campaign_target_smiles=target_smiles,
        validated_parent_step_ids=validated_parent_step_ids,
    )
    jobs = queue.list_jobs(resolved_case_id)
    _validate_current_proposal_expansion_gates(
        jobs,
        validated_parent_step_ids=validated_parent_step_ids,
    )
    frontier_sync = {
        "enabled": bool(resolved_provider),
        "before_job_count": len(jobs),
        "after_job_count": len(jobs),
        "added_job_count": 0,
        "enabled_job_count": enabled_job_count,
        "stock_authority": stock_authority,
        "source_graph_is_fused_authority_projection": True,
        "external_hyperedges_journaled_before_queue_mutation": True,
        "source_graph_is_canonical_durable_union": True,
    }
    if resolved_provider:
        scheduler = FrontierScheduler(queue, resolved_provider)
        expanded_smiles = {
            row.frontier_smiles
            for row in jobs
            if row.state == FrontierJobState.SUCCEEDED
            and row.closure_kind == "proposal_expansion"
        }
        added = _submit_graph_frontiers(
            graph=graph,
            scheduler=scheduler,
            config=sync_config,
            case_id=resolved_case_id,
            expanded_smiles=expanded_smiles,
            max_depth=sync_max_depth,
            frontier_batch_size=max(1, int(sync_config.frontier_batch_size or 1)),
            campaign_identity_sha256=identity_sha256,
            campaign_policy_sha256=campaign_policy_sha256,
            campaign_target_smiles=target_smiles,
            validated_parent_step_ids=validated_parent_step_ids,
        )
        jobs = queue.list_jobs(resolved_case_id)
        _validate_campaign_queue_identity(
            jobs,
            case_id=resolved_case_id,
            target_smiles=target_smiles,
            campaign_identity_sha256=identity_sha256,
            campaign_policy_sha256=campaign_policy_sha256,
        )
        frontier_sync = {
            **frontier_sync,
            "after_job_count": len(jobs),
            "added_job_count": added,
        }
    open_proofs = _open_reaction_proof_frontiers(proof_state)
    terminals = _campaign_terminal_smiles(graph) or [target_smiles]
    stock_replay = _audit_supplemental_stock_evidence(
        list(stock_evidence or []),
        jobs=jobs,
    )
    trusted_stock_providers = {
        provider.descriptor.provider_id: provider
        for provider in (resolved_provider or ())
    }
    completeness = assess_frontier_completeness(
        terminals,
        jobs,
        open_proof_frontiers=open_proofs,
        required_proof_level=max(0, min(4, int(required_proof_level))),
        trusted_stock_provider_instances=trusted_stock_providers,
    )
    queue_snapshot = queue.snapshot(resolved_case_id)
    frontier_ledger = project_frontier_ledger(
        graph,
        queue_snapshot,
        proof_state,
        required_reaction_proof_level=max(
            2,
            min(4, int(required_proof_level)),
        ),
        trusted_stock_provider_instances=trusted_stock_providers,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    frontier_ledger_validation = validate_frontier_ledger(
        frontier_ledger,
        trusted_stock_provider_instances=trusted_stock_providers,
    )
    ledger_inputs = dict(frontier_ledger.get("input_validation") or {})
    frontier_ledger_authoritative = bool(
        not frontier_ledger_validation
        and all(
            isinstance(ledger_inputs.get(key), dict)
            and ledger_inputs[key].get("valid") is True
            for key in ("graph", "frontier_queue", "reaction_proof_state")
        )
    )
    closure_status = campaign_closure_status(
        frontier_ledger,
        authoritative=frontier_ledger_authoritative,
        closure_objective=closure_objective,
        exploration_mode=exploration_mode,
    )
    graph_complete = closure_status["campaign_search_complete"] is True
    frontier_ledger_path = root_output_dir / "frontier_ledger.json"
    _write_json(frontier_ledger_path, frontier_ledger)
    queue_state_counts = {
        state.value: sum(1 for row in jobs if row.state == state)
        for state in FrontierJobState
    }
    remaining_frontier = select_route_consensus_frontier(
        graph,
        limit=max(1, len(graph.get("nodes") or [])),
    )
    canonical_count_summary = _canonical_reconciliation_count_summary(
        committed_expansions=committed_expansions,
        admitted_external_expansions=admitted_hyperedge_expansions,
        canonical_graph=graph,
    )
    return {
        "schema_version": "codex_campaign_proof_reconciliation.v1",
        "accepted": True,
        "case_id": resolved_case_id,
        "target_smiles": target_smiles,
        "campaign_identity_sha256": identity_sha256,
        "campaign_policy_sha256": campaign_policy_sha256,
        "campaign_policy_ref": str(policy_path),
        "admitted_hyperedge_journal": admitted_hyperedge_journal,
        "admitted_hyperedge_journal_ref": str(
            admitted_hyperedge_journal_root
        ),
        "canonical_route_consensus_graph": graph,
        "caller_advisory_graph": caller_graph_summary,
        "journal_commit_recovery_errors": journal_commit_recovery_errors,
        **closure_status,
        "reaction_proof_state": proof_state,
        "reaction_proof_state_ref": str(proof_state_path),
        "open_reaction_proofs": open_proofs,
        "stock_evidence_replay": stock_replay,
        "frontier_sync": frontier_sync,
        "frontier_queue": queue_snapshot,
        "queue_state_counts": queue_state_counts,
        "remaining_frontier": remaining_frontier,
        "proposal_graph_exhausted": not remaining_frontier,
        "frontier_completeness": completeness.to_dict(),
        "frontier_ledger": frontier_ledger,
        "frontier_ledger_ref": str(frontier_ledger_path),
        "frontier_ledger_validation_reasons": frontier_ledger_validation,
        "frontier_ledger_authoritative": frontier_ledger_authoritative,
        "graph_complete": graph_complete,
        "proposal_runner_invoked": False,
        **canonical_count_summary,
        # Delta for this proof-only reconciliation call. The cumulative
        # durable count above prevents consumers from mistaking this zero for
        # an empty campaign.
        "expansion_budget_consumed": 0,
    }


def _canonical_reconciliation_count_summary(
    *,
    committed_expansions: Iterable[Mapping[str, Any]],
    admitted_external_expansions: Iterable[Mapping[str, Any]],
    canonical_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate durable input events from unique canonical reaction edges."""

    durable_rows = [
        row for row in committed_expansions if isinstance(row, Mapping)
    ]
    external_rows = [
        row
        for row in admitted_external_expansions
        if isinstance(row, Mapping)
    ]
    input_event_count = len(durable_rows) + len(external_rows)
    return {
        "durable_accepted_expansion_count": len(durable_rows),
        "admitted_external_expansion_count": len(external_rows),
        "canonical_input_expansion_event_count": input_event_count,
        "canonical_reaction_edge_count": len(
            graph_exact_edge_signatures(canonical_graph)
        ),
        # Transitional compatibility for artifacts/readers produced before
        # the event-vs-edge distinction was named explicitly.
        "canonical_expansion_count": input_event_count,
        "canonical_expansion_count_semantics": (
            "deprecated_alias_of_canonical_input_expansion_event_count"
        ),
    }


def _caller_advisory_graph_summary(graph: Mapping[str, Any]) -> dict[str, Any]:
    exact_signatures = sorted(graph_exact_edge_signatures(graph))
    payload = {
        "schema_version": "caller_advisory_route_graph_summary.v1",
        "graph_schema_version": str(graph.get("schema_version") or ""),
        "case_id": str(graph.get("case_id") or ""),
        "target_smiles": _canonical_target_smiles(graph.get("target_smiles")),
        "step_count": len(
            [row for row in graph.get("steps") or [] if isinstance(row, Mapping)]
        ),
        "exact_edge_signatures": exact_signatures,
        "semantics": {
            "advisory_input_only": True,
            "cannot_remove_durable_union_edges": True,
            "not_frontier_or_proof_authority": True,
        },
    }
    payload["content_sha256"] = _payload_digest(payload)
    return payload


def _assemble_canonical_route_consensus_graph(
    expansions: Iterable[Mapping[str, Any]],
    *,
    case_id: str,
    target_smiles: str,
    max_depth: int,
    max_route_hypotheses: int = 24,
) -> dict[str, Any]:
    """Assemble every durable proposal edge without the display-size cap."""

    rows = [dict(row) for row in expansions if isinstance(row, Mapping)]
    proposal_capacity = sum(
        len(((row.get("route_consensus") or {}).get("proposals")) or [])
        for row in rows
    )
    return assemble_route_consensus_graph(
        rows,
        case_id=case_id,
        target_smiles=target_smiles,
        max_depth=max_depth,
        max_route_hypotheses=max(1, int(max_route_hypotheses or 1)),
        max_graph_steps=max(256, proposal_capacity),
    )


def migrate_legacy_campaign_commits(
    *,
    case_id: str,
    target_smiles: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Upgrade a pre-outbox campaign without rerunning any Codex team."""

    root_run_dir = Path(run_dir).resolve()
    root_output_dir = root_run_dir / "codex_retrosynthesis_team"
    root_smiles = _canonical_target_smiles(target_smiles)
    if not str(case_id or "").strip() or not root_smiles:
        raise ValueError("legacy campaign identity is invalid")
    identity_path = root_output_dir / "campaign_identity.json"
    campaign_identity = _ensure_campaign_identity(
        identity_path,
        case_id=case_id,
        target_smiles=root_smiles,
        root_output_dir=root_output_dir,
    )
    campaign_identity_sha256 = str(campaign_identity["content_sha256"])
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
    queue.migrate_legacy_benchmark_stock_authority(case_id)
    _validate_campaign_queue_identity(
        queue.list_jobs(case_id),
        case_id=case_id,
        target_smiles=root_smiles,
        campaign_identity_sha256=campaign_identity_sha256,
        allow_legacy_unbound=True,
    )
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
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_target_smiles=root_smiles,
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
                "campaign_identity_sha256": campaign_identity_sha256,
                "campaign_root_smiles": root_smiles,
                "completed_lease_token_sha256": hashlib.sha256(
                    synthetic_token.encode("utf-8")
                ).hexdigest(),
                "legacy_result_ref_sha256": hashlib.sha256(
                    str(job.result_ref).encode("utf-8")
                ).hexdigest(),
            },
        )
        migrated.append(job.job_id)
    recovery_errors = _reconcile_expansion_commits(
        queue=queue,
        run_id=case_id,
        root_output_dir=root_output_dir,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_target_smiles=root_smiles,
        expansions=expansions,
        runs=runs,
    )
    _write_campaign_state(
        state_path,
        case_id=case_id,
        target_smiles=root_smiles,
        expansions=expansions,
        runs=runs,
    )
    return {
        "schema_version": "codex_retrosynthesis_campaign_migration.v1",
        "accepted": not recovery_errors,
        "run_dir": str(root_run_dir),
        "case_id": case_id,
        "campaign_identity_ref": str(identity_path),
        "campaign_identity_sha256": campaign_identity_sha256,
        "migrated_job_ids": migrated,
        "skipped_job_ids": skipped,
        "recovery_errors": recovery_errors,
        "campaign_state_ref": str(state_path),
        "frontier_queue": queue.snapshot(case_id),
    }


def _stock_request(config: RetrosynthesisTeamConfig, canonical_smiles: str) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for key, value in (config.stock_snapshots or {}).items():
        if _canonical_target_smiles(key) == canonical_smiles and isinstance(value, dict):
            raw = dict(value)
            break
    if not raw:
        return {}
    if raw.get("schema_version") == "stock_offer_snapshot.v1" or (
        "supplier" in raw and "catalog_number" in raw and "available" in raw
    ):
        offers = [raw]
    else:
        offers = [dict(row) for row in raw.get("offers") or [] if isinstance(row, dict)]
    normalized_offers: list[dict[str, Any]] = []
    for offer in offers:
        row = dict(offer)
        try:
            row["snapshot_sha256"] = stock_snapshot_sha256(row)
        except (TypeError, ValueError):
            # Preserve malformed input for the provider's fail-closed audit.
            pass
        normalized_offers.append(row)
    return {
        "schema_version": "stock_lookup_request.v1",
        "offers": normalized_offers,
    }


def _campaign_stock_provider(
    config: RetrosynthesisTeamConfig,
    *,
    stock_provider: StockProvider | None,
) -> tuple[tuple[StockProvider, ...], dict[str, Any]]:
    if stock_provider is not None:
        descriptor = getattr(stock_provider, "descriptor", None)
        trusted_snapshot_digests = sorted(
            str(item)
            for item in getattr(stock_provider, "_trusted_snapshots", {}).keys()
            if _valid_sha256(item)
        )
        return (stock_provider,), {
            "source": "injected_provider",
            "provider_id": str(getattr(descriptor, "provider_id", "") or ""),
            "provider_ids": [
                str(getattr(descriptor, "provider_id", "") or "")
            ],
            "trusted_snapshot_sha256": trusted_snapshot_digests,
            "configured_snapshot_count": None,
            "available": True,
            "reasons": [],
        }

    providers: list[StockProvider] = []
    catalog_authority: dict[str, Any] = {}
    catalog_artifact = str(
        config.benchmark_stock_catalog_artifact or ""
    ).strip()
    if catalog_artifact:
        benchmark_provider = BenchmarkCatalogStockProvider(
            catalog_artifact=catalog_artifact,
            catalog_sha256=str(config.benchmark_stock_catalog_sha256 or ""),
            catalog_name=str(config.benchmark_stock_catalog_name or "benchmark-stock"),
        )
        providers.append(benchmark_provider)
        catalog_authority = {
            "catalog_artifact": str(benchmark_provider.catalog_artifact),
            "catalog_sha256": benchmark_provider.catalog_sha256,
            "catalog_name": benchmark_provider.catalog_name,
        }

    trusted: list[dict[str, Any]] = []
    invalid_count = 0
    for key in (config.stock_snapshots or {}):
        canonical = _canonical_target_smiles(key)
        if not canonical:
            invalid_count += 1
            continue
        request = _stock_request(config, canonical)
        for row in request.get("offers") or []:
            if not isinstance(row, dict):
                invalid_count += 1
                continue
            try:
                # Construction-time loading is the authority boundary; the
                # provider recomputes and verifies the digest again.
                stock_snapshot_sha256(row)
            except (TypeError, ValueError):
                invalid_count += 1
                continue
            trusted.append(dict(row))
    snapshot_provider = SnapshotStockProvider(trusted_snapshots=trusted)
    if trusted or not providers:
        providers.append(snapshot_provider)
    reasons: list[str] = []
    if not trusted and not providers:
        reasons.append("no_trusted_stock_snapshots_configured")
    elif not trusted and not catalog_artifact:
        reasons.append("no_trusted_stock_snapshots_configured")
    if invalid_count:
        reasons.append("one_or_more_stock_snapshots_invalid")
    if catalog_artifact and trusted:
        source = "benchmark_and_commercial_provider_set"
    elif catalog_artifact:
        source = "hashed_benchmark_catalog"
    else:
        source = "config_snapshot"
    provider_ids = sorted(provider.descriptor.provider_id for provider in providers)
    return tuple(providers), {
        "source": source,
        "provider_id": provider_ids[0],
        "provider_ids": provider_ids,
        "trusted_snapshot_sha256": sorted(
            stock_snapshot_sha256(row) for row in trusted
        ),
        "configured_snapshot_count": len(trusted),
        "invalid_snapshot_count": invalid_count,
        "available": bool(trusted or catalog_artifact),
        "commercial_orderability_claimed": False,
        **catalog_authority,
        "reasons": reasons,
    }


def _rehydrate_stock_providers_from_campaign_policy(
    campaign_policy: dict[str, Any],
) -> tuple[tuple[StockProvider, ...], dict[str, Any]]:
    """Recreate only self-contained providers bound by immutable policy.

    A hashed benchmark catalog carries every input needed for current-host
    replay: an artifact path, its digest, and its meaning. Commercial snapshot
    policies intentionally retain only digests, not supplier payloads, so they
    remain unavailable unless the caller supplies the original campaign config.
    """

    policy_sha256 = str(campaign_policy.get("content_sha256") or "")
    binding = dict(campaign_policy.get("stock_authority_binding") or {})
    descriptor = dict(binding.get("provider_descriptor") or {})
    descriptor_rows = [
        dict(row)
        for row in binding.get("provider_descriptors") or []
        if isinstance(row, dict)
    ]
    reasons: list[str] = []
    provider_id = str(descriptor.get("provider_id") or "")
    rehydration_required = bool(
        str(binding.get("authority_source") or "") == "hashed_benchmark_catalog"
        and not descriptor_rows
        and not binding.get("trusted_snapshot_sha256")
    )
    if descriptor_rows:
        reasons.append("campaign_policy_provider_set_requires_original_inputs")
    elif provider_id != BenchmarkCatalogStockProvider.descriptor.provider_id:
        reasons.append("campaign_policy_stock_provider_not_self_contained")
    elif str(binding.get("authority_source") or "") != "hashed_benchmark_catalog":
        reasons.append("campaign_policy_benchmark_authority_source_invalid")
    elif binding.get("trusted_snapshot_sha256"):
        reasons.append("campaign_policy_mixes_non_rehydratable_snapshot_authority")

    provider: BenchmarkCatalogStockProvider | None = None
    if not reasons:
        try:
            provider = BenchmarkCatalogStockProvider(
                catalog_artifact=str(binding.get("catalog_artifact") or ""),
                catalog_sha256=str(binding.get("catalog_sha256") or ""),
                catalog_name=str(binding.get("catalog_name") or "benchmark-stock"),
            )
        except (OSError, TypeError, ValueError) as exc:
            reasons.append(
                f"campaign_policy_benchmark_rehydration_error:{type(exc).__name__}:{exc}"
            )
    if provider is not None and _payload_digest(
        provider.descriptor.to_dict()
    ) != _payload_digest(descriptor):
        reasons.append("campaign_policy_benchmark_descriptor_mismatch")
        provider = None

    providers: tuple[StockProvider, ...] = (provider,) if provider is not None else ()
    return providers, {
        "schema_version": "campaign_policy_stock_rehydration.v1",
        "source": "immutable_campaign_policy_rehydration",
        "campaign_policy_sha256": policy_sha256,
        "rehydration_required": rehydration_required,
        "available": bool(providers),
        "provider_ids": [row.descriptor.provider_id for row in providers],
        "catalog_artifact": str(binding.get("catalog_artifact") or ""),
        "catalog_sha256": str(binding.get("catalog_sha256") or ""),
        "catalog_name": str(binding.get("catalog_name") or ""),
        "commercial_orderability_claimed": False,
        "reasons": reasons,
    }


def _submit_graph_frontiers(
    *,
    graph: dict[str, Any],
    scheduler: FrontierScheduler,
    config: RetrosynthesisTeamConfig,
    case_id: str,
    expanded_smiles: set[str],
    max_depth: int,
    frontier_batch_size: int,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
    campaign_target_smiles: str,
    validated_parent_step_ids: set[str],
) -> int:
    del frontier_batch_size  # publishing audits is intentionally not throttled
    added = 0
    existing_by_smiles = {
        row.frontier_smiles: row for row in scheduler.queue.list_jobs(case_id)
    }
    for next_frontier in _graph_frontiers_requiring_stock_audit(graph, max_depth=max_depth):
        next_smiles = _canonical_target_smiles(next_frontier.get("target_smiles"))
        next_depth = int(next_frontier.get("depth") or 0)
        if not next_smiles or next_smiles in expanded_smiles:
            continue
        # The campaign root is submitted explicitly before any graph exists,
        # with its own stock-first priority semantics.  The complete-graph
        # frontier projection also reports that root while it is unexpanded;
        # do not enqueue the same idempotency key with different scheduling
        # weights.
        if (
            next_depth == 0
            and next_smiles
            == _canonical_target_smiles(campaign_target_smiles)
        ):
            continue
        parent_step_ids = {
            str(item)
            for item in next_frontier.get("parent_step_ids") or []
            if str(item)
        }
        matched_parent_step_ids = sorted(
            parent_step_ids.intersection(validated_parent_step_ids)
        )
        structurally_expandable = bool(
            next_frontier.get("proposal_expansion_allowed")
        )
        proposal_expansion_allowed = bool(
            structurally_expandable and matched_parent_step_ids
        )
        existing = existing_by_smiles.get(next_smiles)
        if existing is not None:
            if parent_step_ids:
                existing = scheduler.queue.merge_parent_step_ids(
                    case_id,
                    existing.job_id,
                    parent_step_ids=sorted(parent_step_ids),
                    campaign_identity_sha256=campaign_identity_sha256,
                    campaign_policy_sha256=campaign_policy_sha256,
                    campaign_root_smiles=campaign_target_smiles,
                )
            existing = scheduler.refresh(
                existing,
                case_id=case_id,
                stock_request=_stock_request(config, next_smiles),
            )
            existing_by_smiles[next_smiles] = existing
            if (
                proposal_expansion_allowed
                and existing.metadata.get("proposal_expansion_allowed") is not True
                and existing.state
                in {FrontierJobState.PENDING, FrontierJobState.RETRY_WAIT}
            ):
                scheduler.queue.enable_proposal_expansion(
                    case_id,
                    existing.job_id,
                    validated_parent_step_ids=matched_parent_step_ids,
                    campaign_identity_sha256=campaign_identity_sha256,
                    campaign_root_smiles=campaign_target_smiles,
                )
            continue
        submitted = scheduler.submit(
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
                "campaign_identity_sha256": campaign_identity_sha256,
                "campaign_policy_sha256": campaign_policy_sha256,
                "campaign_root_smiles": campaign_target_smiles,
                "proposal_expansion_allowed": proposal_expansion_allowed,
                "proposal_expansion_gate": {
                    "schema_version": "proposal_expansion_gate.v1",
                    "status": (
                        "enabled_by_current_host_l2_parent_proof"
                        if proposal_expansion_allowed
                        else "blocked_pending_current_host_l2_parent_proof"
                    ),
                    "validated_parent_step_ids": matched_parent_step_ids,
                },
            },
        )
        added += 1
        existing_by_smiles[next_smiles] = submitted
    return added


def _validated_parent_step_ids(proof_state: dict[str, Any]) -> set[str]:
    """Return only step ids replayed to L2 by the current host verifier."""

    return {
        str(row.get("step_id") or "")
        for row in proof_state.get("records") or []
        if isinstance(row, dict)
        and row.get("status") == "validated"
        and int(row.get("achieved_proof_level") or 0) >= 2
        and row.get("proof_authority") == "current_host_verifier_replay"
        and str(row.get("step_id") or "")
    }


def _validate_current_proposal_expansion_gates(
    jobs: list[FrontierJob],
    *,
    validated_parent_step_ids: set[str],
) -> None:
    """Reject any non-root queue enablement not justified by current replay."""

    for job in jobs:
        metadata = dict(job.metadata or {})
        if (
            int(metadata.get("depth") or 0) <= 0
            or metadata.get("proposal_expansion_allowed") is not True
        ):
            continue
        parent_ids = {
            str(item) for item in metadata.get("parent_step_ids") or [] if str(item)
        }
        if not parent_ids.intersection(validated_parent_step_ids):
            raise ValueError(
                f"proposal expansion gate lacks current host L2 parent proof:{job.job_id}"
            )


def _enable_existing_proven_frontiers(
    *,
    graph: dict[str, Any],
    queue: PersistentFrontierQueue,
    case_id: str,
    max_depth: int,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
    campaign_target_smiles: str,
    validated_parent_step_ids: set[str],
) -> int:
    """Monotonically enable already-audited leaves after parent L2 proof."""

    jobs_by_smiles = {row.frontier_smiles: row for row in queue.list_jobs(case_id)}
    enabled = 0
    for frontier in _graph_frontiers_requiring_stock_audit(graph, max_depth=max_depth):
        if frontier.get("proposal_expansion_allowed") is not True:
            continue
        smiles = _canonical_target_smiles(frontier.get("target_smiles"))
        job = jobs_by_smiles.get(smiles)
        if job is None:
            continue
        observed_parent_step_ids = sorted(
            {
                str(item)
                for item in frontier.get("parent_step_ids") or []
                if str(item)
            }
        )
        if observed_parent_step_ids:
            job = queue.merge_parent_step_ids(
                case_id,
                job.job_id,
                parent_step_ids=observed_parent_step_ids,
                campaign_identity_sha256=campaign_identity_sha256,
                campaign_policy_sha256=campaign_policy_sha256,
                campaign_root_smiles=campaign_target_smiles,
            )
        if (
            job.state not in {FrontierJobState.PENDING, FrontierJobState.RETRY_WAIT}
            or job.metadata.get("proposal_expansion_allowed") is True
        ):
            continue
        matched = sorted(
            {
                str(item)
                for item in frontier.get("parent_step_ids") or []
                if str(item) in validated_parent_step_ids
            }
        )
        if not matched:
            continue
        queue.enable_proposal_expansion(
            case_id,
            job.job_id,
            validated_parent_step_ids=matched,
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_root_smiles=campaign_target_smiles,
        )
        enabled += 1
    return enabled


def _graph_frontiers_requiring_stock_audit(
    graph: dict[str, Any],
    *,
    max_depth: int,
) -> list[dict[str, Any]]:
    """Return every route leaf, including depth/cycle boundaries.

    Only unexpanded leaves below ``max_depth`` may consume Codex agent work;
    every other leaf is still submitted so the stock provider audits it before
    it is reported unresolved.
    """

    nodes = {
        str(row.get("node_id") or ""): dict(row)
        for row in graph.get("nodes") or []
        if isinstance(row, dict)
    }
    expandable = {
        str(row.get("node_id") or ""): dict(row)
        for row in select_route_consensus_frontier(graph, limit=max(1, len(nodes)))
    }
    by_smiles: dict[str, dict[str, Any]] = {}
    for raw in route_consensus_frontier_records(graph):
        node_id = str(raw.get("node_id") or "")
        node = nodes.get(node_id) or {}
        smiles = _canonical_target_smiles(
            raw.get("target_smiles") or node.get("smiles")
        )
        if not smiles:
            continue
        depth = int(raw.get("depth") or node.get("min_depth") or 0)
        selected = expandable.get(node_id) or {}
        reason = str(raw.get("reason") or "unexpanded")
        expansion_allowed = bool(
            reason == "unexpanded" and depth < max_depth and selected
        )
        ancestor_smiles = _ancestor_smiles_for_node(graph, node_id)
        existing = by_smiles.get(smiles)
        row = {
            "node_id": node_id,
            "target_smiles": smiles,
            "depth": depth,
            "parent_step_ids": list(
                selected.get("parent_step_ids")
                or raw.get("parent_step_ids")
                or []
            ),
            "reason": reason,
            "terminal_reasons": [reason],
            "ancestor_smiles": ancestor_smiles,
            "forbidden_return_smiles": sorted({smiles, *ancestor_smiles}),
            "priority_score": float(
                selected.get("priority_score")
                or raw.get("priority_score")
                or 0.0
            ),
            "proposal_expansion_allowed": expansion_allowed,
        }
        if existing is None:
            by_smiles[smiles] = row
            continue
        existing["depth"] = min(int(existing.get("depth") or 0), depth)
        existing["terminal_reasons"] = sorted(
            {*existing.get("terminal_reasons", []), reason}
        )
        existing["parent_step_ids"] = sorted(
            {
                *existing.get("parent_step_ids", []),
                *row.get("parent_step_ids", []),
            }
        )
        existing["ancestor_smiles"] = sorted(
            {*existing.get("ancestor_smiles", []), *ancestor_smiles}
        )
        existing["forbidden_return_smiles"] = sorted(
            {
                smiles,
                *existing.get("forbidden_return_smiles", []),
                *ancestor_smiles,
            }
        )
        existing["proposal_expansion_allowed"] = bool(
            existing.get("proposal_expansion_allowed") or expansion_allowed
        )
        existing["priority_score"] = max(
            float(existing.get("priority_score") or 0.0),
            float(row.get("priority_score") or 0.0),
        )
    return sorted(
        by_smiles.values(),
        key=lambda row: (
            not bool(row.get("proposal_expansion_allowed")),
            -float(row.get("priority_score") or 0.0),
            int(row.get("depth") or 0),
            str(row.get("node_id") or ""),
        ),
    )


def _ancestor_smiles_for_node(graph: dict[str, Any], node_id: str) -> list[str]:
    nodes = {
        str(row.get("node_id") or ""): _canonical_target_smiles(row.get("smiles"))
        for row in graph.get("nodes") or []
        if isinstance(row, dict)
    }
    parent_ids: dict[str, set[str]] = {}
    for step in graph.get("steps") or []:
        if not isinstance(step, dict):
            continue
        product_id = str(step.get("product_node_id") or "")
        for precursor_id in step.get("precursor_node_ids") or []:
            parent_ids.setdefault(str(precursor_id), set()).add(product_id)
    ancestors: set[str] = set()
    pending = list(parent_ids.get(node_id) or [])
    seen: set[str] = {node_id}
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        smiles = nodes.get(current) or ""
        if smiles:
            ancestors.add(smiles)
        pending.extend(parent_ids.get(current) or [])
    return sorted(ancestors)


def _proposal_cycle_reasons(
    team_report: dict[str, Any],
    *,
    frontier: dict[str, Any],
    frontier_smiles: str,
) -> list[str]:
    forbidden = {
        frontier_smiles,
        *[
            _canonical_target_smiles(item)
            for item in frontier.get("ancestor_smiles") or []
        ],
        *[
            _canonical_target_smiles(item)
            for item in frontier.get("forbidden_return_smiles") or []
        ],
    } - {""}
    reasons: list[str] = []
    for proposal in (team_report.get("route_consensus") or {}).get("proposals") or []:
        if not isinstance(proposal, dict):
            continue
        proposal_id = str(proposal.get("consensus_id") or "unknown")
        product = _canonical_target_smiles(proposal.get("product_smiles"))
        precursors = {
            _canonical_target_smiles(item)
            for item in proposal.get("precursor_smiles") or []
        } - {""}
        if product and product in precursors:
            reasons.append(f"proposal_cycle:self_return:{proposal_id}")
        if precursors & forbidden:
            reasons.append(f"proposal_cycle:ancestor_or_target_return:{proposal_id}")
    return sorted(set(reasons))


def _enforce_campaign_cycle_boundary(
    team_report: dict[str, Any],
    *,
    frontier: dict[str, Any],
    frontier_smiles: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = dict(team_report)
    consensus = dict(report.get("route_consensus") or {})
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in consensus.get("proposals") or []:
        if not isinstance(raw, dict):
            continue
        proposal = dict(raw)
        reasons = _proposal_cycle_reasons(
            {"route_consensus": {"proposals": [proposal]}},
            frontier=frontier,
            frontier_smiles=frontier_smiles,
        )
        if reasons:
            rejected.append(
                {
                    "proposal_id": str(proposal.get("consensus_id") or ""),
                    "reasons": reasons,
                }
            )
        else:
            accepted.append(proposal)
    forbidden = sorted(
        {
            frontier_smiles,
            *[
                _canonical_target_smiles(item)
                for item in frontier.get("forbidden_return_smiles") or []
            ],
            *[
                _canonical_target_smiles(item)
                for item in frontier.get("ancestor_smiles") or []
            ],
        }
        - {""}
    )
    audit = {
        "accepted": bool(accepted) or not rejected,
        "input_proposal_count": len(accepted) + len(rejected),
        "accepted_proposal_count": len(accepted),
        "rejected_proposal_count": len(rejected),
        "rejected_proposals": rejected,
        "forbidden_return_smiles": forbidden,
    }
    if not rejected:
        return report, audit
    consensus["proposals"] = accepted
    consensus["accepted"] = bool(accepted)
    source_summary = dict(consensus.get("source_summary") or {})
    source_summary["proposal_count"] = len(accepted)
    source_summary["cycle_rejected_proposal_count"] = len(rejected)
    source_summary["multi_source_proposals"] = sum(
        1 for row in accepted if int(row.get("source_diversity") or 0) > 1
    )
    consensus["source_summary"] = source_summary
    consensus["cycle_rejected_proposals"] = rejected
    report["route_consensus"] = consensus
    report["blackboard_proposals"] = consensus_to_blackboard_proposals(consensus)
    report["campaign_cycle_audit"] = audit
    if not accepted:
        report["accepted"] = False
        report["reasons"] = sorted(
            {
                *[str(item) for item in report.get("reasons") or []],
                *[
                    reason
                    for row in rejected
                    for reason in row.get("reasons") or []
                ],
            }
        )
    return report, audit


def _reconcile_reaction_proof_state(
    graph: dict[str, Any],
    *,
    path: Path,
    configured_proofs: dict[str, dict[str, Any]],
    configured_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay materialized candidates into durable edge proof requests.

    Neither a caller-supplied proof nor the historical state file carries
    authority.  Every accepted record is regenerated by the current host
    ``verify_reaction_step`` implementation from a materialized candidate.
    """

    graph_identity = _reaction_graph_identity(graph)
    options: dict[str, list[dict[str, Any]]] = {}
    input_rejections: list[dict[str, Any]] = []
    previous = _read_json_object(path)
    previous_digest_payload = dict(previous)
    previous_digest = str(previous_digest_payload.pop("content_sha256", ""))
    previous_valid = bool(
        previous.get("schema_version")
        == CODEX_RETROSYNTHESIS_REACTION_PROOF_STATE_SCHEMA
        and previous_digest
        and previous_digest == _payload_digest(previous_digest_payload)
    )
    if previous_valid:
        for raw in previous.get("records") or []:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            materialized = row.get("materialized_candidate")
            if not isinstance(materialized, dict) or not materialized:
                continue
            option = {
                "materialized_candidate": dict(materialized),
                "source_kind": "historical_candidate_cache_replayed",
                "require_supplied_proof_match": False,
            }
            for key in (str(row.get("step_id") or ""), str(row.get("signature") or "")):
                if key:
                    options.setdefault(key, []).append(option)

    for key, raw in (configured_proofs or {}).items():
        if not isinstance(raw, dict):
            continue
        option = _configured_reaction_replay_option(
            dict(raw),
            source_kind="configured_materialized_candidate",
        )
        options.setdefault(str(key), []).append(option)

    for report_index, raw_report in enumerate(configured_reports or []):
        if not isinstance(raw_report, dict):
            input_rejections.append(
                {"source": f"edge_report:{report_index}", "reasons": ["report_not_object"]}
            )
            continue
        report = dict(raw_report)
        report_reasons = _codex_edge_report_replay_reasons(report, graph=graph)
        if report_reasons:
            input_rejections.append(
                {"source": f"edge_report:{report_index}", "reasons": report_reasons}
            )
            continue
        report_digest = str(report.get("content_sha256") or "")
        canonical_step_ids = {
            str(row.get("step_id") or "")
            for row in graph.get("steps") or []
            if isinstance(row, Mapping) and str(row.get("step_id") or "")
        }
        for raw_edge in report.get("edge_verifications") or []:
            edge = dict(raw_edge)
            step_id = str(edge.get("step_id") or "")
            if step_id not in canonical_step_ids:
                input_rejections.append(
                    {
                        "source": f"edge_report:{report_index}:edge:{step_id}",
                        "reasons": ["edge_not_in_canonical_durable_graph_ignored"],
                    }
                )
                continue
            option = _configured_reaction_replay_option(
                edge,
                source_kind="codex_edge_verification_report_replayed",
            )
            option["require_supplied_proof_match"] = True
            option["replay_step_index"] = 0
            option["supplied_reaction_validation"] = dict(
                edge.get("reaction_validation") or {}
            )
            option["source_report_sha256"] = report_digest
            if step_id:
                options.setdefault(step_id, []).append(option)

    records: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(graph.get("steps") or []):
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        step_id = str(step.get("step_id") or "")
        signature = str(step.get("signature") or "")
        proof_request_id = "reaction-proof:" + hashlib.sha256(
            f"{step_id}|{signature}".encode("utf-8")
        ).hexdigest()[:24]
        replay_options = [
            *options.get(step_id, []),
            *options.get(signature, []),
        ]
        selected_option: dict[str, Any] = replay_options[0] if replay_options else {}
        replayed_proof: dict[str, Any] = {}
        validation_reasons = ["materialized_reaction_candidate_missing"]
        option_audits: list[dict[str, Any]] = []
        for option in replay_options:
            proof, option_reasons = _replay_reaction_candidate(
                option,
                step=step,
                step_index=step_index,
            )
            option_audits.append(
                {
                    "source_kind": str(option.get("source_kind") or "unknown"),
                    "candidate_sha256": _safe_payload_digest(
                        option.get("materialized_candidate")
                    ),
                    "accepted": bool(proof and not option_reasons),
                    "reasons": option_reasons,
                }
            )
            if proof and not option_reasons:
                selected_option = option
                replayed_proof = proof
                validation_reasons = []
                break
            if option is selected_option:
                replayed_proof = proof
                validation_reasons = option_reasons
        materialized_candidate = selected_option.get("materialized_candidate")
        has_supplied_input = bool(replay_options)
        validated = bool(replayed_proof and not validation_reasons)
        status = "validated" if validated else ("rejected" if has_supplied_input else "pending")
        records.append(
            {
                "schema_version": "codex_retrosynthesis_reaction_proof_record.v2",
                "proof_request_id": proof_request_id,
                "step_id": step_id,
                "signature": signature,
                "product_smiles": str(step.get("product_smiles") or ""),
                "precursor_smiles": list(step.get("precursor_smiles") or []),
                "required_proof_level": 2,
                "status": status,
                "achieved_proof_level": 2 if validated else 0,
                "open_reason": (
                    ""
                    if validated
                    else (
                        "reaction_candidate_replay_rejected"
                        if has_supplied_input
                        else "proposal_hyperedge_not_reaction_validated"
                    )
                ),
                "validation_reasons": validation_reasons,
                "materialized_candidate": (
                    dict(materialized_candidate)
                    if isinstance(materialized_candidate, dict)
                    else {}
                ),
                "materialized_candidate_sha256": _safe_payload_digest(materialized_candidate),
                "proof": replayed_proof,
                "proof_authority": (
                    "current_host_verifier_replay" if validated else "none"
                ),
                "replay_source_kind": str(selected_option.get("source_kind") or ""),
                "replay_options": option_audits,
            }
        )
    payload = {
        "schema_version": CODEX_RETROSYNTHESIS_REACTION_PROOF_STATE_SCHEMA,
        "graph_identity_sha256": graph_identity,
        "records": records,
        "input_rejections": input_rejections,
        "summary": {
            "total": len(records),
            "validated": sum(1 for row in records if row["status"] == "validated"),
            "pending": sum(1 for row in records if row["status"] == "pending"),
            "rejected": sum(1 for row in records if row["status"] == "rejected"),
            "input_rejection_count": len(input_rejections),
        },
        "consumer_contract": {
            "required_input": "materialized_reaction_candidate.v1",
            "accepted_report": "complete_codex_edge_verification_report.v1",
            "output_field": "proof",
            "accepted_schema": "reaction_step_proof.v1",
            "historical_state_is_candidate_cache_only": True,
            "supplied_proof_is_not_authority": True,
            "current_host_verifier_replay_required": True,
            "status_is_not_authority": True,
        },
    }
    payload["content_sha256"] = _payload_digest(payload)
    _write_json(path, payload)
    return payload


def _configured_reaction_replay_option(
    value: dict[str, Any],
    *,
    source_kind: str,
) -> dict[str, Any]:
    candidate = value.get("materialized_candidate")
    if not isinstance(candidate, dict) and value.get("schema_version") == (
        "materialized_reaction_candidate.v1"
    ):
        candidate = value
    supplied_proof = (
        value.get("step_proof")
        or value.get("verifier_proof")
        or value.get("proof")
    )
    if value.get("schema_version") == "reaction_step_proof.v1":
        supplied_proof = value
    return {
        "materialized_candidate": dict(candidate) if isinstance(candidate, dict) else {},
        "supplied_proof": (
            dict(supplied_proof) if isinstance(supplied_proof, dict) else {}
        ),
        "source_kind": source_kind,
        "require_supplied_proof_match": bool(supplied_proof),
    }


def _replay_reaction_candidate(
    option: dict[str, Any],
    *,
    step: dict[str, Any],
    step_index: int,
) -> tuple[dict[str, Any], list[str]]:
    candidate = option.get("materialized_candidate")
    if not isinstance(candidate, dict) or not candidate:
        return {}, ["materialized_reaction_candidate_missing"]
    materialized = dict(candidate)
    reasons: list[str] = []
    if materialized.get("schema_version") != "materialized_reaction_candidate.v1":
        reasons.append("invalid_materialized_reaction_candidate_schema")
    expected_step_id = str(step.get("step_id") or "")
    if str(materialized.get("step_id") or "") != expected_step_id:
        reasons.append("materialized_reaction_candidate_step_id_mismatch")
    product = _canonical_target_smiles(materialized.get("product_smiles"))
    expected_product = _canonical_target_smiles(step.get("product_smiles"))
    reactants = sorted(
        value
        for value in (
            _canonical_target_smiles(item)
            for item in materialized.get("reactant_smiles") or []
        )
        if value
    )
    expected_reactants = sorted(
        value
        for value in (
            _canonical_target_smiles(item)
            for item in step.get("precursor_smiles") or []
        )
        if value
    )
    if not product or product != expected_product:
        reasons.append("materialized_reaction_candidate_product_mismatch")
    if not reactants or reactants != expected_reactants:
        reasons.append("materialized_reaction_candidate_precursor_mismatch")
    if reasons:
        return {}, sorted(set(reasons))
    supplied = option.get("supplied_proof")
    supplied_checks = (
        dict(supplied.get("checks") or {})
        if isinstance(supplied, dict) and isinstance(supplied.get("checks"), dict)
        else {}
    )
    graph_and_stock_closed = supplied_checks.get("graph_and_stock_closed") is True
    replay_index = int(option.get("replay_step_index", step_index))
    replayed = verify_reaction_step(
        materialized,
        step_index=replay_index,
        graph_and_stock_closed=graph_and_stock_closed,
    )
    reasons.extend(_current_host_proof_reasons(replayed, step=step))
    if option.get("require_supplied_proof_match"):
        if not isinstance(supplied, dict) or not supplied:
            reasons.append("supplied_reaction_step_proof_missing")
        elif _safe_payload_digest(supplied) != _safe_payload_digest(replayed):
            reasons.append("supplied_reaction_step_proof_not_equal_to_current_host_replay")
        supplied_route = option.get("supplied_reaction_validation")
        if isinstance(supplied_route, dict) and supplied_route:
            replayed_route = verify_reaction_route(
                [materialized],
                graph_and_stock_closed=graph_and_stock_closed,
            )
            if _safe_payload_digest(supplied_route) != _safe_payload_digest(replayed_route):
                reasons.append(
                    "supplied_reaction_route_validation_not_equal_to_current_host_replay"
                )
    return replayed, sorted(set(reasons))


def _current_host_proof_reasons(
    proof: dict[str, Any],
    *,
    step: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if proof.get("schema_version") != "reaction_step_proof.v1":
        reasons.append("invalid_current_host_reaction_step_proof_schema")
    if proof.get("validator_version") != REACTION_STEP_VERIFIER_VERSION:
        reasons.append("current_host_reaction_step_verifier_version_mismatch")
    if proof.get("accepted") is not True or proof.get("proof_level") not in {
        "L2_reaction_validated",
        "L3_precedent_supported",
        "L4_procurement_ready",
    }:
        reasons.append("current_host_reaction_step_not_validated")
    digest_payload = dict(proof)
    recorded_digest = str(digest_payload.pop("proof_digest", ""))
    if not recorded_digest or recorded_digest != _safe_payload_digest(digest_payload):
        reasons.append("current_host_reaction_step_proof_digest_invalid")
    if _canonical_target_smiles(proof.get("product_smiles")) != _canonical_target_smiles(
        step.get("product_smiles")
    ):
        reasons.append("current_host_reaction_step_product_mismatch")
    if sorted(
        _canonical_target_smiles(item) for item in proof.get("reactant_smiles") or []
    ) != sorted(
        _canonical_target_smiles(item) for item in step.get("precursor_smiles") or []
    ):
        reasons.append("current_host_reaction_step_precursor_mismatch")
    return sorted(set(reasons))


def _codex_edge_report_replay_reasons(
    report: dict[str, Any],
    *,
    graph: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if report.get("schema_version") != "codex_edge_verification_report.v1":
        reasons.append("invalid_codex_edge_report_schema")
    digest_payload = dict(report)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    if not recorded_digest or recorded_digest != _safe_payload_digest(digest_payload):
        reasons.append("codex_edge_report_content_digest_invalid")
    if report.get("graph_schema_version") != graph.get("schema_version"):
        reasons.append("codex_edge_report_graph_schema_mismatch")
    if _canonical_target_smiles(report.get("target_smiles")) != _canonical_target_smiles(
        graph.get("target_smiles")
    ):
        reasons.append("codex_edge_report_target_mismatch")
    edges = [dict(row) for row in report.get("edge_verifications") or [] if isinstance(row, dict)]
    edge_step_ids = {str(row.get("step_id") or "") for row in edges} - {""}
    try:
        reported_edge_count = int(report.get("edge_count") or 0)
    except (TypeError, ValueError):
        reported_edge_count = -1
    if (
        reported_edge_count != len(edges)
        or len(edge_step_ids) != len(edges)
    ):
        reasons.append("codex_edge_report_edge_index_invalid")
    for edge in edges:
        if (
            edge.get("schema_version") != "codex_edge_verification.v1"
            or not isinstance(edge.get("materialized_candidate"), dict)
            or not isinstance(edge.get("step_proof"), dict)
            or not isinstance(edge.get("reaction_validation"), dict)
            or (edge.get("reaction_validation") or {}).get("schema_version")
            != "reaction_route_validation.v1"
        ):
            reasons.append(
                f"codex_edge_report_edge_incomplete:{str(edge.get('step_id') or 'unknown')}"
            )
    return sorted(set(reasons))


def _reaction_graph_identity(graph: dict[str, Any]) -> str:
    payload = {
        "schema_version": str(graph.get("schema_version") or ""),
        "case_id": str(graph.get("case_id") or ""),
        "target_smiles": _canonical_target_smiles(graph.get("target_smiles")),
        "steps": sorted(
            (
                {
                    "step_id": str(row.get("step_id") or ""),
                    "signature": str(row.get("signature") or ""),
                    "product_smiles": _canonical_target_smiles(row.get("product_smiles")),
                    "precursor_smiles": sorted(
                        _canonical_target_smiles(item)
                        for item in row.get("precursor_smiles") or []
                    ),
                }
                for row in graph.get("steps") or []
                if isinstance(row, dict)
            ),
            key=lambda row: (row["step_id"], row["signature"]),
        ),
    }
    return _payload_digest(payload)


def _safe_payload_digest(value: Any) -> str:
    try:
        return _payload_digest(dict(value)) if isinstance(value, dict) else ""
    except (TypeError, ValueError):
        return ""


def _open_reaction_proof_frontiers(
    proof_state: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "frontier": str(row.get("step_id") or ""),
            "proof_request_id": str(row.get("proof_request_id") or ""),
            "product_smiles": str(row.get("product_smiles") or ""),
            "precursor_smiles": list(row.get("precursor_smiles") or []),
            "proof_status": str(row.get("status") or "pending"),
            "reason": str(
                row.get("open_reason")
                or "proposal_hyperedge_not_reaction_validated"
            ),
            "validation_reasons": list(row.get("validation_reasons") or []),
        }
        for row in proof_state.get("records") or []
        if isinstance(row, dict) and row.get("status") != "validated"
    ]


def _audit_supplemental_stock_evidence(
    values: list[dict[str, Any]],
    *,
    jobs: list[FrontierJob],
) -> dict[str, Any]:
    """Rebind supplemental envelopes to queue-observed stock closures only."""

    descriptors = {
        SnapshotStockProvider.descriptor.provider_id: SnapshotStockProvider.descriptor,
        BenchmarkCatalogStockProvider.descriptor.provider_id: (
            BenchmarkCatalogStockProvider.descriptor
        ),
    }
    queue_authority: dict[str, FrontierJob] = {}
    for job in jobs:
        observation_state = (job.metadata or {}).get("stock_observations")
        current_results = [
            dict(row.get("provider_result") or {})
            for row in (
                observation_state.get("current")
                if isinstance(observation_state, dict)
                else []
            )
            if isinstance(row, dict)
            and isinstance(row.get("provider_result"), dict)
        ]
        legacy = (job.metadata or {}).get("stock_audit")
        if not current_results and isinstance(legacy, dict):
            current_results = [legacy]
        for audit in current_results:
            content_hash = str(audit.get("content_hash") or "")
            if content_hash:
                queue_authority[content_hash] = job
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            rejected.append({"index": index, "reasons": ["stock_evidence_not_object"]})
            continue
        row = dict(raw)
        descriptor = descriptors.get(str(row.get("provider_id") or ""))
        reasons = (
            ["stock_provider_descriptor_not_trusted"]
            if descriptor is None
            else validate_provider_result(row, descriptor=descriptor)
        )
        payload = row.get("payload")
        if (
            row.get("accepted") is not True
            or row.get("provider_kind") != "stock"
            or row.get("output_schema") != "stock_boundary.v1"
            or not isinstance(payload, dict)
            or payload.get("accepted") is not True
        ):
            reasons.append("stock_provider_envelope_not_accepted_boundary")
        canonical = _canonical_target_smiles(
            payload.get("canonical_smiles") if isinstance(payload, dict) else ""
        )
        if not canonical:
            reasons.append("stock_provider_boundary_smiles_invalid")
        boundary_type = str(payload.get("boundary_type") or "") if isinstance(payload, dict) else ""
        if descriptor is SnapshotStockProvider.descriptor:
            offers = [
                dict(offer)
                for offer in (payload.get("offers") or [])
                if isinstance(offer, dict)
                and offer.get("available") is True
                and offer.get("snapshot_verified") is True
                and _valid_sha256(offer.get("snapshot_sha256"))
            ] if isinstance(payload, dict) else []
            if boundary_type != "commercially_orderable" or not offers:
                reasons.append("snapshot_stock_boundary_not_replayable")
        elif descriptor is BenchmarkCatalogStockProvider.descriptor:
            bindings = [
                dict(binding)
                for binding in (payload.get("catalog_bindings") or [])
                if isinstance(binding, dict)
                and binding.get("artifact_hash_verified") is True
                and binding.get("commercial_orderability_claimed") is False
                and _valid_sha256(binding.get("catalog_sha256"))
            ] if isinstance(payload, dict) else []
            if boundary_type != "benchmark_stock" or not bindings:
                reasons.append("benchmark_stock_boundary_not_replayable")
        content_hash = str(row.get("content_hash") or "")
        authority_job = queue_authority.get(content_hash)
        if authority_job is None or authority_job.frontier_smiles != canonical:
            reasons.append("stock_evidence_not_bound_to_queue_stock_closure")
        if reasons:
            rejected.append({"index": index, "reasons": sorted(set(reasons))})
            continue
        accepted.append(
            {
                "index": index,
                "content_hash": content_hash,
                "canonical_smiles": canonical,
                "provider_id": str(row.get("provider_id") or ""),
                "frontier_job_id": authority_job.job_id,
            }
        )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "queue_stock_closure_is_authority": True,
        "supplemental_envelope_cannot_create_closure": True,
    }


def _next_retry_available_at(jobs: list[FrontierJob]) -> str:
    values = sorted(
        row.available_at
        for row in jobs
        if row.state == FrontierJobState.RETRY_WAIT and row.available_at
    )
    return values[0] if values else ""


def _campaign_resume_guidance(
    *,
    stop_reason: str,
    accepted_expansion_count: int,
    max_expansions: int,
    queue_state_counts: dict[str, int],
    open_reaction_proof_count: int,
) -> list[str]:
    guidance: list[str] = []
    if stop_reason in {
        "invocation_accepted_expansion_cap_reached",
        "invocation_attempt_run_cap_reached",
    } and accepted_expansion_count < max_expansions:
        guidance.append("invoke_campaign_again_with_same_run_dir")
    if stop_reason == "frontier_retry_wait" or queue_state_counts.get("retry_wait", 0):
        guidance.append("resume_at_reported_resumable_at")
    if stop_reason == "campaign_accepted_expansion_budget_exhausted":
        guidance.append("increase_max_expansions_to_continue_proposal_search")
    if stop_reason == "campaign_attempt_run_budget_exhausted":
        guidance.append("increase_max_attempt_runs_to_continue_agent_work")
    if queue_state_counts.get("pending", 0):
        guidance.append("pending_frontier_jobs_are_durable")
    if open_reaction_proof_count:
        guidance.append("materialize_and_verify_open_reaction_proof_requests")
    return list(dict.fromkeys(guidance))


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
    result: list[str] = []
    for frontier in route_consensus_frontier_records(graph):
        smiles = _canonical_target_smiles(frontier.get("target_smiles"))
        if smiles and smiles not in result:
            result.append(smiles)
    return result


def _ensure_campaign_identity(
    path: Path,
    *,
    case_id: str,
    target_smiles: str,
    root_output_dir: Path,
) -> dict[str, Any]:
    """Create or verify the immutable identity fence for one run directory."""

    canonical = _canonical_target_smiles(target_smiles)
    if not str(case_id or "").strip() or not canonical:
        raise ValueError("campaign identity requires case_id and canonical target")
    if path.exists():
        row = _read_json_strict(path, label="campaign identity manifest")
        digest_payload = dict(row)
        recorded_digest = str(digest_payload.pop("content_sha256", ""))
        if (
            row.get("schema_version") != CODEX_RETROSYNTHESIS_CAMPAIGN_IDENTITY_SCHEMA
            or set(row) != {
                "schema_version",
                "case_id",
                "canonical_target_smiles",
                "identity_policy",
                "content_sha256",
            }
            or row.get("identity_policy") != "immutable_case_and_canonical_target.v1"
            or not recorded_digest
            or recorded_digest != _payload_digest(digest_payload)
        ):
            raise ValueError("campaign identity manifest digest or schema is invalid")
        if row.get("case_id") != case_id or row.get("canonical_target_smiles") != canonical:
            raise ValueError(
                "campaign run_dir is already bound to a different case_id or canonical target"
            )
        return row

    # Refuse to bless a pre-existing mutable report for another molecule when
    # introducing the manifest to an older run directory.
    report_path = root_output_dir / "team_report.json"
    if report_path.exists():
        _load_root_report_for_campaign(
            report_path,
            case_id=case_id,
            target_smiles=canonical,
            campaign_identity_sha256="",
        )
    payload = {
        "schema_version": CODEX_RETROSYNTHESIS_CAMPAIGN_IDENTITY_SCHEMA,
        "case_id": case_id,
        "canonical_target_smiles": canonical,
        "identity_policy": "immutable_case_and_canonical_target.v1",
    }
    payload["content_sha256"] = _payload_digest(payload)
    _write_immutable_json(path, payload)
    return payload


def _ensure_campaign_policy(
    path: Path,
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    max_depth: int,
    required_proof_level: int,
    stock_provider: tuple[StockProvider, ...],
    stock_authority: dict[str, Any],
    coordinator_contract_version: str,
    child_roles: tuple[str, ...],
    model: str,
    closure_objective: str,
    exploration_mode: str,
    child_acceptance_mode: str,
) -> dict[str, Any]:
    """Create or replay the immutable scientific/authority policy fence."""

    providers = tuple(stock_provider)
    if not providers:
        raise ValueError("campaign stock provider set is required")
    descriptor = getattr(providers[0], "descriptor", None)
    if descriptor is None or not callable(getattr(descriptor, "to_dict", None)):
        raise ValueError("campaign stock provider descriptor is required")
    stock_binding: dict[str, Any] = {
        "provider_descriptor": descriptor.to_dict(),
        "authority_source": str(stock_authority.get("source") or ""),
        "trusted_snapshot_sha256": sorted(
            str(item)
            for item in stock_authority.get("trusted_snapshot_sha256") or []
            if _valid_sha256(item)
        ),
        "catalog_artifact": str(stock_authority.get("catalog_artifact") or ""),
        "catalog_sha256": str(stock_authority.get("catalog_sha256") or ""),
        "catalog_name": str(stock_authority.get("catalog_name") or ""),
    }
    benchmark_provider = next(
        (
            provider
            for provider in providers
            if isinstance(provider, BenchmarkCatalogStockProvider)
        ),
        None,
    )
    snapshot_provider = next(
        (
            provider
            for provider in providers
            if isinstance(provider, SnapshotStockProvider)
        ),
        None,
    )
    if benchmark_provider is not None:
        stock_binding.update(
            {
                "catalog_artifact": str(benchmark_provider.catalog_artifact),
                "catalog_sha256": str(benchmark_provider.catalog_sha256),
                "catalog_name": str(benchmark_provider.catalog_name),
            }
        )
    if snapshot_provider is not None:
        stock_binding["trusted_snapshot_sha256"] = sorted(
            str(item)
            for item in getattr(snapshot_provider, "_trusted_snapshots", {}).keys()
            if _valid_sha256(item)
        )
    if len(providers) > 1:
        stock_binding.update(
            {
                "provider_descriptors": [
                    provider.descriptor.to_dict() for provider in providers
                ],
                "provider_set_binding": stock_provider_set_authority_binding(
                    providers
                ),
            }
        )
    normalized_child_roles = tuple(_normalized_roles(list(child_roles)))
    normalized_child_acceptance_mode = _normalize_child_acceptance_mode(
        child_acceptance_mode
    )
    payload = {
        "schema_version": CODEX_RETROSYNTHESIS_CAMPAIGN_POLICY_SCHEMA,
        "case_id": case_id,
        "canonical_target_smiles": _canonical_target_smiles(target_smiles),
        "campaign_identity_sha256": campaign_identity_sha256,
        "max_depth": int(max_depth),
        "closure_objective": _normalize_closure_objective(closure_objective),
        "exploration_mode": _normalize_exploration_mode(exploration_mode),
        "required_reaction_proof_level": int(required_proof_level),
        "reaction_verifier_version": REACTION_STEP_VERIFIER_VERSION,
        "proposal_agent_policy": {
            "coordinator_contract_version": str(coordinator_contract_version),
            "child_acceptance_contract_version": CHILD_ACCEPTANCE_CONTRACT_VERSION,
            "child_acceptance_mode": normalized_child_acceptance_mode,
            "child_roles": list(normalized_child_roles),
            "derived_valid_child_quorum": _derived_valid_child_quorum(
                normalized_child_roles
            ),
            "model": str(model),
        },
        "stock_authority_binding": stock_binding,
        "budget_fields_excluded_from_immutable_policy": [
            "max_expansions",
            "max_attempt_runs",
        ],
        "policy_semantics": {
            "authority_change_requires_new_campaign": True,
            "depth_change_requires_new_campaign": True,
            "proposal_agent_change_requires_new_campaign": True,
            "child_acceptance_change_requires_new_campaign": True,
            "closure_objective_change_requires_new_campaign": True,
            "exploration_mode_change_requires_new_campaign": True,
            "budget_growth_is_separately_audited": True,
        },
    }
    payload = json.loads(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    payload["content_sha256"] = _payload_digest(payload)
    if path.exists():
        existing = _read_json_strict(path, label="campaign policy manifest")
        if existing != payload:
            raise ValueError(
                "campaign policy mismatch: depth, closure objective, exploration mode, "
                "proof floor, or stock authority changed"
            )
        return existing
    _write_immutable_json(path, payload)
    return payload


def _load_campaign_policy(
    path: Path,
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
) -> dict[str, Any]:
    row = _read_json_strict(path, label="campaign policy manifest")
    digest_payload = dict(row)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    if (
        row.get("schema_version") != CODEX_RETROSYNTHESIS_CAMPAIGN_POLICY_SCHEMA
        or row.get("case_id") != case_id
        or _canonical_target_smiles(row.get("canonical_target_smiles"))
        != _canonical_target_smiles(target_smiles)
        or row.get("campaign_identity_sha256") != campaign_identity_sha256
        or not recorded_digest
        or recorded_digest != _payload_digest(digest_payload)
    ):
        raise ValueError("campaign policy manifest identity or digest is invalid")
    return row


def _ensure_campaign_budget_envelope(
    *,
    root_output_dir: Path,
    case_id: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
    max_expansions: int,
    max_attempt_runs: int,
) -> dict[str, Any]:
    """Replay an append-only budget chain and permit monotonic growth only."""

    event_root = root_output_dir / "campaign_budget_events"
    projection_path = root_output_dir / "campaign_budget.json"
    with _exclusive_campaign_file_lock(event_root / ".budget.lock"):
        event_root.mkdir(parents=True, exist_ok=True)
        events: list[dict[str, Any]] = []
        previous_digest = ""
        for sequence, event_path in enumerate(sorted(event_root.glob("*.json"))):
            row = _read_json_strict(event_path, label="campaign budget event")
            digest_payload = dict(row)
            recorded_digest = str(digest_payload.pop("content_sha256", ""))
            try:
                recorded_sequence = int(row.get("sequence"))
                event_max_expansions = int(row.get("max_expansions"))
                event_max_attempts = int(row.get("max_attempt_runs"))
            except (TypeError, ValueError) as exc:
                raise ValueError("campaign budget event values are invalid") from exc
            if (
                row.get("schema_version") != CODEX_RETROSYNTHESIS_BUDGET_EVENT_SCHEMA
                or recorded_sequence != sequence
                or event_path.name != f"{sequence:06d}.json"
                or row.get("case_id") != case_id
                or row.get("campaign_identity_sha256") != campaign_identity_sha256
                or row.get("campaign_policy_sha256") != campaign_policy_sha256
                or row.get("previous_event_sha256") != previous_digest
                or event_max_expansions < 1
                or event_max_attempts < 1
                or (
                    events
                    and (
                        event_max_expansions < int(events[-1]["max_expansions"])
                        or event_max_attempts < int(events[-1]["max_attempt_runs"])
                    )
                )
                or not recorded_digest
                or recorded_digest != _payload_digest(digest_payload)
            ):
                raise ValueError("campaign budget event chain is invalid")
            events.append(row)
            previous_digest = recorded_digest

        if projection_path.exists():
            projection = _read_json_strict(
                projection_path,
                label="campaign budget projection",
            )
            projection_payload = dict(projection)
            projection_digest = str(projection_payload.pop("content_sha256", ""))
            try:
                projected_event_count = int(projection.get("event_count") or 0)
            except (TypeError, ValueError):
                projected_event_count = -1
            projected_event = (
                events[projected_event_count - 1]
                if 0 < projected_event_count <= len(events)
                else {}
            )
            if (
                not events
                or projection.get("schema_version")
                != "codex_retrosynthesis_budget_envelope.v1"
                or projection.get("campaign_identity_sha256")
                != campaign_identity_sha256
                or projection.get("campaign_policy_sha256")
                != campaign_policy_sha256
                or not projected_event
                or projection.get("head_event_sha256")
                != projected_event.get("content_sha256")
                or int(projection.get("max_expansions") or 0)
                != int(projected_event.get("max_expansions") or 0)
                or int(projection.get("max_attempt_runs") or 0)
                != int(projected_event.get("max_attempt_runs") or 0)
                or not projection_digest
                or projection_digest != _payload_digest(projection_payload)
            ):
                raise ValueError("campaign budget projection does not match event chain")

        requested_expansions = int(max_expansions)
        requested_attempts = int(max_attempt_runs)
        if events and (
            requested_expansions < int(events[-1]["max_expansions"])
            or requested_attempts < int(events[-1]["max_attempt_runs"])
        ):
            raise ValueError("campaign budgets cannot shrink across invocations")
        if not events or (
            requested_expansions > int(events[-1]["max_expansions"])
            or requested_attempts > int(events[-1]["max_attempt_runs"])
        ):
            event = {
                "schema_version": CODEX_RETROSYNTHESIS_BUDGET_EVENT_SCHEMA,
                "event": "initialized" if not events else "monotonic_extension",
                "sequence": len(events),
                "case_id": case_id,
                "campaign_identity_sha256": campaign_identity_sha256,
                "campaign_policy_sha256": campaign_policy_sha256,
                "previous_event_sha256": previous_digest,
                "previous_max_expansions": (
                    int(events[-1]["max_expansions"]) if events else 0
                ),
                "previous_max_attempt_runs": (
                    int(events[-1]["max_attempt_runs"]) if events else 0
                ),
                "max_expansions": requested_expansions,
                "max_attempt_runs": requested_attempts,
                "recorded_at": _utc_now(),
            }
            event["content_sha256"] = _payload_digest(event)
            _write_immutable_json(event_root / f"{len(events):06d}.json", event)
            events.append(event)
            previous_digest = str(event["content_sha256"])

        projection = {
            "schema_version": "codex_retrosynthesis_budget_envelope.v1",
            "case_id": case_id,
            "campaign_identity_sha256": campaign_identity_sha256,
            "campaign_policy_sha256": campaign_policy_sha256,
            "max_expansions": int(events[-1]["max_expansions"]),
            "max_attempt_runs": int(events[-1]["max_attempt_runs"]),
            "event_count": len(events),
            "head_event_sha256": previous_digest,
            "events": [
                {
                    "sequence": int(row["sequence"]),
                    "event": str(row["event"]),
                    "content_sha256": str(row["content_sha256"]),
                    "ref": str(event_root / f"{int(row['sequence']):06d}.json"),
                }
                for row in events
            ],
            "monotonic_extension_only": True,
            "projection_only": True,
        }
        projection["content_sha256"] = _payload_digest(projection)
        _write_json(projection_path, projection)
        return projection


def _load_root_report_for_campaign(
    path: Path,
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str = "",
) -> dict[str, Any]:
    if not path.exists():
        return {}
    row = _read_json_strict(path, label="campaign root team report")
    if (
        row.get("schema_version") != CODEX_RETROSYNTHESIS_TEAM_SCHEMA
        or row.get("case_id") != case_id
        or _canonical_target_smiles(row.get("target_smiles"))
        != _canonical_target_smiles(target_smiles)
    ):
        raise ValueError("campaign root team report identity mismatch")
    bindings = {
        str(row.get("campaign_identity_sha256") or ""),
        str((row.get("campaign") or {}).get("campaign_identity_sha256") or ""),
    } - {""}
    if campaign_identity_sha256 and bindings and bindings != {campaign_identity_sha256}:
        raise ValueError("campaign root team report identity fence mismatch")
    policy_bindings = {
        str(row.get("campaign_policy_sha256") or ""),
        str((row.get("campaign") or {}).get("campaign_policy_sha256") or ""),
    } - {""}
    if (
        campaign_policy_sha256
        and policy_bindings
        and policy_bindings != {campaign_policy_sha256}
    ):
        raise ValueError("campaign root team report policy fence mismatch")
    return row


def _validate_campaign_queue_identity(
    jobs: list[FrontierJob],
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str = "",
    allow_legacy_unbound: bool = False,
) -> None:
    canonical = _canonical_target_smiles(target_smiles)
    for job in jobs:
        metadata = dict(job.metadata or {})
        bound_digest = str(metadata.get("campaign_identity_sha256") or "")
        bound_policy = str(metadata.get("campaign_policy_sha256") or "")
        bound_root = _canonical_target_smiles(metadata.get("campaign_root_smiles"))
        if job.run_id != case_id:
            raise ValueError("campaign frontier queue run identity mismatch")
        if not bound_digest and not bound_root and allow_legacy_unbound:
            pass
        elif bound_digest != campaign_identity_sha256 or bound_root != canonical:
            raise ValueError("campaign frontier queue identity fence mismatch")
        if (
            campaign_policy_sha256
            and bound_policy
            and bound_policy != campaign_policy_sha256
        ):
            raise ValueError("campaign frontier queue policy fence mismatch")
        if int(metadata.get("depth") or 0) == 0 and job.frontier_smiles != canonical:
            raise ValueError("campaign frontier queue root target mismatch")


def _root_report_from_reconciled_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    roots = sorted(
        (
            row
            for row in runs
            if int(row.get("depth") or 0) == 0
            and row.get("proposal_expansion_recorded") is True
        ),
        key=lambda row: str(row.get("frontier_job_id") or ""),
    )
    for row in roots:
        path = Path(str(row.get("team_report_content_path") or ""))
        if path.is_file():
            report = _read_json_object(path)
            if report.get("accepted") is True:
                return report
    return {}


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
    attempt_records: list[dict[str, Any]] | None = None,
    max_attempt_runs: int = 0,
    attempt_ledger_ref: str = "",
) -> None:
    attempts = list(attempt_records or [])
    campaign_budget = _read_json_object(path.parent / "campaign_budget.json")
    campaign_policy = _read_json_object(path.parent / "campaign_policy.json")
    terminal_count = sum(
        1 for row in attempts if isinstance(row.get("terminal"), dict)
    )
    payload = {
        "schema_version": "codex_retrosynthesis_campaign_state.v1",
        "case_id": case_id,
        "target_smiles": target_smiles,
        "expansions": expansions,
        "runs": runs,
        "campaign_policy_projection": {
            "campaign_policy_sha256": str(
                campaign_policy.get("content_sha256") or ""
            ),
            "campaign_policy_ref": str(path.parent / "campaign_policy.json"),
            "projection_only": True,
        },
        "campaign_budget_projection": {
            "max_expansions": int(campaign_budget.get("max_expansions") or 0),
            "max_attempt_runs": int(campaign_budget.get("max_attempt_runs") or 0),
            "event_count": int(campaign_budget.get("event_count") or 0),
            "head_event_sha256": str(
                campaign_budget.get("head_event_sha256") or ""
            ),
            "campaign_budget_ref": str(path.parent / "campaign_budget.json"),
            "projection_only": True,
        },
        "attempt_budget": {
            "schema_version": "codex_retrosynthesis_attempt_budget.v1",
            "max_attempt_runs": max(0, int(max_attempt_runs)),
            "started_attempt_count": len(attempts),
            "terminal_attempt_count": terminal_count,
            "dangling_started_attempt_count": len(attempts) - terminal_count,
            "remaining_attempt_runs": max(
                0,
                int(max_attempt_runs) - len(attempts),
            ),
            "attempt_ledger_ref": attempt_ledger_ref,
            "projection_only": True,
        },
    }
    payload["content_sha256"] = _payload_digest(payload)
    _write_json(path, payload)


def _campaign_attempt_id(
    *,
    campaign_identity_sha256: str,
    job_id: str,
    job_attempt: int,
    lease_token_sha256: str,
) -> str:
    binding = {
        "campaign_identity_sha256": campaign_identity_sha256,
        "job_id": job_id,
        "job_attempt": int(job_attempt),
        "lease_token_sha256": lease_token_sha256,
    }
    return f"attempt:{_payload_digest(binding)}"


def _campaign_attempt_event_dir(root_output_dir: Path, attempt_id: str) -> Path:
    digest = str(attempt_id or "").removeprefix("attempt:")
    if not _valid_sha256(digest):
        raise ValueError("campaign attempt id is invalid")
    return root_output_dir / "campaign_attempts" / digest


def _write_campaign_attempt_started(
    *,
    root_output_dir: Path,
    case_id: str,
    campaign_identity_sha256: str,
    campaign_target_smiles: str,
    job: FrontierJob,
    expansion_case_id: str,
    depth: int,
) -> dict[str, Any]:
    """Append the immutable started event before invoking an Agent team."""

    lease_digest = hashlib.sha256(job.lease_token.encode("utf-8")).hexdigest()
    attempt_id = _campaign_attempt_id(
        campaign_identity_sha256=campaign_identity_sha256,
        job_id=job.job_id,
        job_attempt=job.attempt,
        lease_token_sha256=lease_digest,
    )
    payload = {
        "schema_version": CODEX_RETROSYNTHESIS_ATTEMPT_EVENT_SCHEMA,
        "event": "started",
        "attempt_id": attempt_id,
        "case_id": case_id,
        "campaign_identity_sha256": campaign_identity_sha256,
        "campaign_target_smiles": campaign_target_smiles,
        "job_id": job.job_id,
        "job_attempt": job.attempt,
        "lease_token_sha256": lease_digest,
        "frontier_smiles": job.frontier_smiles,
        "depth": int(depth),
        "expansion_case_id": expansion_case_id,
        "started_at": _utc_now(),
    }
    payload["content_sha256"] = _payload_digest(payload)
    event_path = _campaign_attempt_event_dir(root_output_dir, attempt_id) / "started.json"
    if event_path.exists():
        raise ValueError("campaign attempt already has a started event")
    _write_immutable_json(event_path, payload)
    return payload


def _write_campaign_attempt_terminal(
    *,
    root_output_dir: Path,
    started_event: dict[str, Any],
    terminal_status: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Append one immutable terminal event without rewriting its start."""

    if not terminal_status:
        raise ValueError("campaign attempt terminal status is required")
    attempt_id = str(started_event.get("attempt_id") or "")
    payload = {
        "schema_version": CODEX_RETROSYNTHESIS_ATTEMPT_EVENT_SCHEMA,
        "event": "terminal",
        "attempt_id": attempt_id,
        "started_event_sha256": str(started_event.get("content_sha256") or ""),
        "case_id": str(started_event.get("case_id") or ""),
        "campaign_identity_sha256": str(
            started_event.get("campaign_identity_sha256") or ""
        ),
        "campaign_target_smiles": str(
            started_event.get("campaign_target_smiles") or ""
        ),
        "job_id": str(started_event.get("job_id") or ""),
        "job_attempt": int(started_event.get("job_attempt") or 0),
        "lease_token_sha256": str(
            started_event.get("lease_token_sha256") or ""
        ),
        "frontier_smiles": str(started_event.get("frontier_smiles") or ""),
        "depth": int(started_event.get("depth") or 0),
        "expansion_case_id": str(started_event.get("expansion_case_id") or ""),
        "terminal_status": terminal_status,
        "finished_at": _utc_now(),
        "summary": summary,
    }
    payload["content_sha256"] = _payload_digest(payload)
    event_path = _campaign_attempt_event_dir(root_output_dir, attempt_id) / "terminal.json"
    if event_path.exists():
        existing = _read_json_strict(event_path, label="campaign attempt terminal event")
        return existing
    _write_immutable_json(event_path, payload)
    return payload


def _load_campaign_attempt_ledger(
    *,
    root_output_dir: Path,
    case_id: str,
    campaign_identity_sha256: str,
    campaign_target_smiles: str,
    jobs: list[FrontierJob],
) -> list[dict[str, Any]]:
    """Replay immutable attempt events as the campaign-wide call authority."""

    attempt_root = root_output_dir / "campaign_attempts"
    if not attempt_root.exists():
        return []
    jobs_by_id = {row.job_id: row for row in jobs}
    records: list[dict[str, Any]] = []
    seen_attempt_ids: set[str] = set()
    seen_job_attempts: set[tuple[str, int]] = set()
    for directory in sorted(path for path in attempt_root.iterdir() if path.is_dir()):
        started_path = directory / "started.json"
        terminal_path = directory / "terminal.json"
        if not started_path.exists():
            if terminal_path.exists():
                raise ValueError("campaign attempt terminal event has no started event")
            continue
        started = _read_json_strict(
            started_path,
            label="campaign attempt started event",
        )
        _validate_campaign_attempt_event(
            started,
            event="started",
            directory=directory,
            case_id=case_id,
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_target_smiles=campaign_target_smiles,
            jobs_by_id=jobs_by_id,
        )
        attempt_id = str(started["attempt_id"])
        job_attempt = (str(started["job_id"]), int(started["job_attempt"]))
        if attempt_id in seen_attempt_ids or job_attempt in seen_job_attempts:
            raise ValueError("duplicate campaign attempt identity")
        seen_attempt_ids.add(attempt_id)
        seen_job_attempts.add(job_attempt)
        terminal: dict[str, Any] | None = None
        if terminal_path.exists():
            terminal = _read_json_strict(
                terminal_path,
                label="campaign attempt terminal event",
            )
            _validate_campaign_attempt_event(
                terminal,
                event="terminal",
                directory=directory,
                case_id=case_id,
                campaign_identity_sha256=campaign_identity_sha256,
                campaign_target_smiles=campaign_target_smiles,
                jobs_by_id=jobs_by_id,
                started_event=started,
            )
        records.append(
            {
                "attempt_id": attempt_id,
                "started_ref": str(started_path.resolve()),
                "terminal_ref": str(terminal_path.resolve()) if terminal else "",
                "started": started,
                "terminal": terminal,
            }
        )
    records.sort(
        key=lambda row: (
            str((row.get("started") or {}).get("started_at") or ""),
            str(row.get("attempt_id") or ""),
        )
    )
    return records


def _validate_campaign_attempt_event(
    row: dict[str, Any],
    *,
    event: str,
    directory: Path,
    case_id: str,
    campaign_identity_sha256: str,
    campaign_target_smiles: str,
    jobs_by_id: dict[str, FrontierJob],
    started_event: dict[str, Any] | None = None,
) -> None:
    digest_payload = dict(row)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    try:
        digest_valid = bool(
            recorded_digest and recorded_digest == _payload_digest(digest_payload)
        )
        job_attempt = int(row.get("job_attempt") or 0)
        depth = int(row.get("depth") or 0)
    except (TypeError, ValueError):
        digest_valid = False
        job_attempt = -1
        depth = -1
    job = jobs_by_id.get(str(row.get("job_id") or ""))
    expected_attempt_id = _campaign_attempt_id(
        campaign_identity_sha256=campaign_identity_sha256,
        job_id=str(row.get("job_id") or ""),
        job_attempt=job_attempt,
        lease_token_sha256=str(row.get("lease_token_sha256") or ""),
    )
    if (
        row.get("schema_version") != CODEX_RETROSYNTHESIS_ATTEMPT_EVENT_SCHEMA
        or row.get("event") != event
        or not digest_valid
        or row.get("case_id") != case_id
        or row.get("campaign_identity_sha256") != campaign_identity_sha256
        or _canonical_target_smiles(row.get("campaign_target_smiles"))
        != campaign_target_smiles
        or row.get("attempt_id") != expected_attempt_id
        or directory.name != expected_attempt_id.removeprefix("attempt:")
        or not _valid_sha256(row.get("lease_token_sha256"))
        or job is None
        or job.run_id != case_id
        or job.attempt < job_attempt
        or job.frontier_smiles
        != _canonical_target_smiles(row.get("frontier_smiles"))
        or depth < 0
    ):
        raise ValueError("campaign attempt event identity or digest is invalid")
    metadata = dict(job.metadata or {})
    if (
        metadata.get("campaign_identity_sha256") != campaign_identity_sha256
        or _canonical_target_smiles(metadata.get("campaign_root_smiles"))
        != campaign_target_smiles
    ):
        raise ValueError("campaign attempt event queue binding is invalid")
    timestamp_key = "started_at" if event == "started" else "finished_at"
    try:
        datetime.fromisoformat(str(row.get(timestamp_key) or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("campaign attempt event timestamp is invalid") from exc
    if event == "terminal":
        if not isinstance(started_event, dict):
            raise ValueError("campaign terminal event requires its started event")
        bound_keys = {
            "attempt_id",
            "case_id",
            "campaign_identity_sha256",
            "campaign_target_smiles",
            "job_id",
            "job_attempt",
            "lease_token_sha256",
            "frontier_smiles",
            "depth",
            "expansion_case_id",
        }
        if (
            row.get("started_event_sha256")
            != started_event.get("content_sha256")
            or any(row.get(key) != started_event.get(key) for key in bound_keys)
            or not str(row.get("terminal_status") or "")
            or not isinstance(row.get("summary"), dict)
        ):
            raise ValueError("campaign terminal event is not bound to its started event")


def _merge_attempt_run_projection(
    runs: list[dict[str, Any]],
    attempt_records: list[dict[str, Any]],
) -> None:
    """Project every terminal attempt while preserving commit authority."""

    by_attempt = {
        (str(row.get("frontier_job_id") or ""), int(row.get("job_attempt") or 0)): row
        for row in runs
    }
    for record in attempt_records:
        terminal = record.get("terminal")
        if not isinstance(terminal, dict):
            continue
        summary = dict(terminal.get("summary") or {})
        summary.update(
            {
                "campaign_attempt_id": record["attempt_id"],
                "job_attempt": int(terminal.get("job_attempt") or 0),
                "attempt_started_ref": record["started_ref"],
                "attempt_terminal_ref": record["terminal_ref"],
                "attempt_terminal_status": terminal.get("terminal_status"),
            }
        )
        key = (str(terminal.get("job_id") or ""), int(terminal.get("job_attempt") or 0))
        existing = by_attempt.get(key)
        if existing is None:
            # Attempt events account for Agent calls and their operational
            # outcome.  They are not proposal authority: only a queue-fenced,
            # replayed immutable expansion commit may set these two fields.
            summary["accepted"] = False
            summary["proposal_expansion_recorded"] = False
            summary["attempt_outcome_not_expansion_authority"] = True
            runs.append(summary)
            by_attempt[key] = summary
        else:
            existing.update(
                {
                    "campaign_attempt_id": record["attempt_id"],
                    "job_attempt": int(terminal.get("job_attempt") or 0),
                    "attempt_started_ref": record["started_ref"],
                    "attempt_terminal_ref": record["terminal_ref"],
                    "attempt_terminal_status": terminal.get("terminal_status"),
                }
            )


def _close_recovered_campaign_attempts(
    *,
    root_output_dir: Path,
    runs: list[dict[str, Any]],
    attempt_records: list[dict[str, Any]],
) -> bool:
    """Append a terminal event when a prepared commit proves crash recovery."""

    recovered_by_attempt = {
        (str(row.get("frontier_job_id") or ""), int(row.get("job_attempt") or 0)): row
        for row in runs
        if row.get("recovered_from_expansion_commit") is True
    }
    changed = False
    for record in attempt_records:
        if isinstance(record.get("terminal"), dict):
            continue
        started = dict(record.get("started") or {})
        summary = recovered_by_attempt.get(
            (str(started.get("job_id") or ""), int(started.get("job_attempt") or 0))
        )
        if summary is None:
            continue
        _write_campaign_attempt_terminal(
            root_output_dir=root_output_dir,
            started_event=started,
            terminal_status="prepared_commit_recovered_after_interruption",
            summary=summary,
        )
        changed = True
    return changed


@contextmanager
def _campaign_attempt_ledger_lock(root_output_dir: Path) -> Iterator[None]:
    """Serialize claim-plus-start so concurrent invocations cannot overspend."""

    lock_path = root_output_dir / "campaign_attempts" / ".attempt-ledger.lock"
    with _exclusive_campaign_file_lock(lock_path):
        yield


@contextmanager
def _exclusive_campaign_file_lock(lock_path: Path) -> Iterator[None]:
    """Small cross-process lock for append-only campaign authorities."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}"
    deadline = time.monotonic() + 5.0
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 120.0
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("campaign authority lock timeout")
            time.sleep(0.02)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        break
    try:
        yield
    finally:
        try:
            if lock_path.read_text(encoding="utf-8") == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


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
    campaign_identity_sha256: str,
    campaign_target_smiles: str,
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
    expected_expansion_case_id = _campaign_expansion_case_id(case_id, job)
    if (
        not isinstance(report, dict)
        or report.get("accepted") is not True
        or report.get("schema_version") != CODEX_RETROSYNTHESIS_TEAM_SCHEMA
        or report.get("case_id") != expected_expansion_case_id
        or _canonical_target_smiles(report.get("target_smiles")) != job.frontier_smiles
    ):
        raise ValueError("expansion commit requires an accepted team report")
    if not _valid_sha256(campaign_identity_sha256):
        raise ValueError("expansion commit requires a valid campaign identity digest")
    if _canonical_target_smiles(campaign_target_smiles) != campaign_target_smiles:
        raise ValueError("expansion commit requires the canonical campaign target")
    expansion_reasons = validate_route_consensus_expansion(expansion)
    if expansion_reasons or (
        _canonical_target_smiles(expansion.get("requested_product_smiles"))
        != job.frontier_smiles
    ):
        raise ValueError("expansion commit payload is not valid for its frontier")
    if (
        summary.get("accepted") is not True
        or summary.get("team_report_accepted") is not True
        or summary.get("proposal_expansion_recorded") is not True
        or summary.get("frontier_job_id") != job.job_id
        or summary.get("case_id") != expected_expansion_case_id
        or _canonical_target_smiles(summary.get("target_smiles")) != job.frontier_smiles
    ):
        raise ValueError("expansion commit summary identity is invalid")
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
        "campaign_identity_sha256": campaign_identity_sha256,
        "campaign_target_smiles": campaign_target_smiles,
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
    campaign_identity_sha256: str,
    campaign_target_smiles: str,
) -> tuple[dict[str, Any], str]:
    try:
        resolved = path.resolve()
        resolved.relative_to((root_output_dir / "campaign_commits").resolve())
    except (OSError, ValueError):
        return {}, "expansion_commit_path_outside_campaign"
    row = _read_json_object(resolved)
    digest_payload = dict(row)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    try:
        recorded_attempt = int(row.get("attempt") or 0)
    except (TypeError, ValueError):
        recorded_attempt = -1
    try:
        commit_digest_valid = bool(
            recorded_digest and recorded_digest == _payload_digest(digest_payload)
        )
    except (TypeError, ValueError):
        commit_digest_valid = False
    if (
        row.get("schema_version") != CODEX_RETROSYNTHESIS_EXPANSION_COMMIT_SCHEMA
        or row.get("case_id") != expected_job.run_id
        or row.get("campaign_identity_sha256") != campaign_identity_sha256
        or _canonical_target_smiles(row.get("campaign_target_smiles"))
        != campaign_target_smiles
        or row.get("job_id") != expected_job.job_id
        or recorded_attempt != expected_job.attempt
        or not commit_digest_valid
    ):
        return {}, "expansion_commit_identity_or_digest_invalid"
    expected_commit_name = (
        f"{hashlib.sha256(expected_job.job_id.encode()).hexdigest()[:20]}"
        f"-a{expected_job.attempt}-{str(row.get('lease_token_sha256') or '')[:12]}.json"
    )
    if (
        not _valid_sha256(row.get("lease_token_sha256"))
        or resolved.name != expected_commit_name
    ):
        return {}, "expansion_commit_fencing_identity_invalid"
    expected_metadata = dict(expected_job.metadata or {})
    recovery_metadata = expected_metadata.get("prepared_result_recovery")
    completed_lease_digest = str(
        expected_metadata.get("completed_lease_token_sha256")
        or (
            recovery_metadata.get("lease_token_sha256")
            if isinstance(recovery_metadata, dict)
            else ""
        )
        or ""
    )
    if (
        expected_job.state == FrontierJobState.SUCCEEDED
        and completed_lease_digest
        and completed_lease_digest != row.get("lease_token_sha256")
    ):
        return {}, "expansion_commit_queue_lease_fence_mismatch"
    expansion = row.get("expansion")
    try:
        expansion_digest_valid = bool(
            isinstance(expansion, dict)
            and row.get("expansion_sha256") == _payload_digest(expansion)
        )
    except (TypeError, ValueError):
        expansion_digest_valid = False
    if not expansion_digest_valid:
        return {}, "expansion_commit_payload_digest_invalid"
    if (
        validate_route_consensus_expansion(expansion)
        or _canonical_target_smiles(expansion.get("requested_product_smiles"))
        != expected_job.frontier_smiles
    ):
        return {}, "expansion_commit_payload_semantics_invalid"
    summary = row.get("summary")
    expected_expansion_case_id = _campaign_expansion_case_id(
        expected_job.run_id,
        expected_job,
    )
    if (
        not isinstance(summary, dict)
        or summary.get("accepted") is not True
        or summary.get("team_report_accepted") is not True
        or summary.get("frontier_job_id") != expected_job.job_id
        or summary.get("case_id") != expected_expansion_case_id
        or _canonical_target_smiles(summary.get("target_smiles"))
        != expected_job.frontier_smiles
    ):
        return {}, "expansion_commit_summary_identity_invalid"
    object_path = Path(str(row.get("team_report_content_path") or ""))
    try:
        resolved_object_path = object_path.resolve()
        resolved_object_path.relative_to(
            (root_output_dir / "campaign_commits" / "objects").resolve()
        )
    except (OSError, ValueError):
        return {}, "expansion_commit_report_object_outside_campaign"
    report_sha256 = str(row.get("team_report_sha256") or "")
    if (
        not _valid_sha256(report_sha256)
        or resolved_object_path.name != "team_report.json"
        or resolved_object_path.parent.name != report_sha256
        or resolved_object_path.parent.parent.name != report_sha256[:2]
        or not resolved_object_path.is_file()
        or _sha256_file(resolved_object_path) != report_sha256
    ):
        return {}, "expansion_commit_report_object_invalid"
    try:
        report = _read_json_strict(
            resolved_object_path,
            label="expansion commit report object",
        )
    except ValueError:
        return {}, "expansion_commit_report_object_invalid"
    if (
        report.get("accepted") is not True
        or report.get("schema_version") != CODEX_RETROSYNTHESIS_TEAM_SCHEMA
        or report.get("case_id") != expected_expansion_case_id
        or _canonical_target_smiles(report.get("target_smiles"))
        != expected_job.frontier_smiles
        or _canonical_target_smiles(
            (report.get("route_consensus") or {}).get("target_smiles")
        )
        != expected_job.frontier_smiles
    ):
        return {}, "expansion_commit_report_identity_or_acceptance_invalid"
    return row, ""


def _reconcile_expansion_commits(
    *,
    queue: PersistentFrontierQueue,
    run_id: str,
    root_output_dir: Path,
    campaign_identity_sha256: str,
    campaign_target_smiles: str,
    expansions: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> list[str]:
    """Adopt prepared commits, then rebuild the cache from queue authority."""

    # Mutable state is never merged into recovery.  Rebuild both projections
    # in full from valid, queue-fenced succeeded commits.
    expansions.clear()
    runs.clear()
    expansion_ids: set[str] = set()
    run_job_ids: set[str] = set()
    errors: list[str] = []
    jobs_by_id = {row.job_id: row for row in queue.list_jobs(run_id)}
    commit_root = root_output_dir / "campaign_commits"
    if commit_root.exists():
        for commit_path in sorted(commit_root.glob("*.json")):
            header = _read_json_object(commit_path)
            job_id = str(header.get("job_id") or "")
            job = jobs_by_id.get(job_id)
            if job is None or job.state == FrontierJobState.SUCCEEDED:
                continue
            commit, reason = _load_expansion_commit(
                commit_path,
                root_output_dir=root_output_dir,
                expected_job=job,
                campaign_identity_sha256=campaign_identity_sha256,
                campaign_target_smiles=campaign_target_smiles,
            )
            if reason:
                errors.append(f"{job_id}:prepared_{reason}")
                continue
            try:
                adopted = queue.adopt_prepared_result(
                    run_id,
                    job_id,
                    result_ref=str(commit_path.resolve()),
                    prepared_attempt=int(commit.get("attempt") or 0),
                    prepared_lease_token_sha256=str(
                        commit.get("lease_token_sha256") or ""
                    ),
                    campaign_identity_sha256=campaign_identity_sha256,
                    campaign_root_smiles=campaign_target_smiles,
                )
            except (FrontierQueueError, KeyError, ValueError) as exc:
                errors.append(
                    f"{job_id}:prepared_result_adoption_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
            else:
                jobs_by_id[job_id] = adopted
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
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_target_smiles=campaign_target_smiles,
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
            summary["job_attempt"] = int(commit.get("attempt") or job.attempt)
            summary["frontier_job_state"] = FrontierJobState.SUCCEEDED.value
            summary["proposal_expansion_recorded"] = True
            summary["recovered_from_expansion_commit"] = True
            summary["expansion_commit_ref"] = str(job.result_ref)
            summary["team_report_content_path"] = str(
                commit.get("team_report_content_path") or ""
            )
            runs.append(summary)
            run_job_ids.add(job.job_id)
    return errors


def _campaign_expansion_case_id(case_id: str, job: FrontierJob) -> str:
    depth = int((job.metadata or {}).get("depth") or 0)
    if depth == 0:
        return case_id
    digest = hashlib.sha256(job.frontier_smiles.encode("utf-8")).hexdigest()[:12]
    return f"{case_id}:frontier:d{depth}:{digest}"


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


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


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


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError("immutable campaign identity conflict")
        return
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # A partial exclusive file is intentionally left fail-closed; a later
        # invocation will reject it instead of silently rebinding the run.
        raise


def _read_json_strict(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


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


def _bounded_coordinator_context(
    context: dict[str, Any],
    *,
    max_bytes: int = 12_000,
) -> dict[str, Any]:
    """Inline safety-critical context without depending on child shell access."""

    board = dict(context.get("blackboard") or {})
    target = dict(context.get("target") or {})
    frontier = dict(board.get("frontier_request") or {})
    current_smiles = _canonical_target_smiles(
        frontier.get("target_smiles") or target.get("smiles")
    )
    ancestors = [
        value
        for value in (
            _canonical_target_smiles(item)
            for item in frontier.get("ancestor_smiles") or []
        )
        if value and value != current_smiles
    ]
    forbidden = sorted(
        {
            current_smiles,
            *ancestors,
            *(
                _canonical_target_smiles(item)
                for item in frontier.get("forbidden_return_smiles") or []
            ),
        }
        - {""}
    )
    graph = dict(board.get("route_consensus_graph") or {})
    paths = []
    for route in graph.get("route_hypotheses") or []:
        if not isinstance(route, dict):
            continue
        paths.append(
            {
                "route_id": str(route.get("route_id") or ""),
                "retrosynthetic_step_ids": [
                    str(item) for item in route.get("retrosynthetic_step_ids") or []
                ][:24],
                "frontier": [
                    {
                        "node_id": str(row.get("node_id") or ""),
                        "depth": int(row.get("depth") or 0),
                        "reason": str(row.get("reason") or ""),
                    }
                    for row in route.get("frontier") or []
                    if isinstance(row, dict)
                ][:12],
            }
        )
        if len(paths) >= 8:
            break
    source_rows = [
        dict(row)
        for row in context.get("literature_sources") or []
        if isinstance(row, dict)
    ]
    literature = dict(board.get("literature_evidence") or {})
    source_rows.extend(
        dict(row)
        for row in literature.get("source_candidates") or []
        if isinstance(row, dict)
    )
    evidence = []
    for row in source_rows:
        evidence.append(
            {
                key: str(row.get(key) or "")[:1_000]
                for key in ("title", "doi", "url", "source_ref", "evidence_level")
                if row.get(key)
            }
        )
        if len(evidence) >= 16:
            break
    payload = {
        "schema_version": "retrosynthesis_coordinator_inline_context.v1",
        "target_name": str(target.get("name") or "")[:500],
        "current_target_smiles": current_smiles,
        "ancestor_target_smiles": ancestors,
        "forbidden_return_smiles": forbidden,
        "frontier": {
            "node_id": str(frontier.get("node_id") or ""),
            "depth": int(frontier.get("depth") or 0),
            "parent_step_ids": [
                str(item) for item in frontier.get("parent_step_ids") or []
            ][:24],
        },
        "existing_paths": paths,
        "evidence_summary": evidence,
        "failure_summary": [
            str(item)[:1_000] for item in board.get("route_failures") or []
        ][:16],
    }
    # Drop least safety-critical rows until the serialized prompt stays
    # bounded. Target/ancestor/forbidden constraints are never removed.
    while len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > max_bytes:
        if payload["evidence_summary"]:
            payload["evidence_summary"].pop()
        elif payload["existing_paths"]:
            payload["existing_paths"].pop()
        elif payload["failure_summary"]:
            payload["failure_summary"].pop()
        else:
            break
    return payload


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


def _audit_explicit_child_spawn_coverage(
    record: WorkerRunRecord,
    *,
    required_roles: list[str],
) -> dict[str, Any]:
    """Audit host-observed spawn records; coordinator summaries have no authority."""

    raw_children = list((record.metadata or {}).get("child_agents") or [])
    required = list(required_roles)
    observed_roles: list[str] = []
    agent_ids: list[str] = []
    reasons: list[str] = []
    if len(raw_children) != len(required):
        reasons.append("explicit_child_spawn_count_mismatch")
    for index, raw in enumerate(raw_children):
        if not isinstance(raw, dict):
            reasons.append("child_spawn_observation_not_object")
            continue
        child = dict(raw)
        role = _normalize_role(child.get("role"))
        agent_id = str(child.get("agent_id") or child.get("call_id") or "").strip()
        observed_roles.append(role)
        agent_ids.append(agent_id)
        if not agent_id:
            reasons.append(f"child_spawn_agent_id_missing:{index}")
        if str(child.get("role_binding_method") or "") != "explicit_spawn_contract":
            reasons.append(f"child_spawn_role_not_explicitly_bound:{index}")
    if len(set(agent_ids)) != len(agent_ids):
        reasons.append("duplicate_child_spawn_agent_id")
    if len(set(observed_roles)) != len(observed_roles):
        reasons.append("duplicate_child_spawn_role")
    if sorted(observed_roles) != sorted(required):
        reasons.append("explicit_child_spawn_role_coverage_mismatch")
    return {
        "schema_version": "codex_child_spawn_coverage_audit.v1",
        "accepted": not reasons,
        "required_roles": required,
        "observed_roles": observed_roles,
        "hard_reasons": sorted(set(reasons)),
        "authority_source": "host_observed_child_agent_records",
        "coordinator_event_summary_used": False,
    }


def _partial_coordinator_safety_reasons(
    record: WorkerRunRecord,
    *,
    task: WorkerTask,
    artifact_validation: dict[str, Any],
    worker_artifact_validation: dict[str, Any],
    runtime_summary: dict[str, Any],
) -> list[str]:
    """Permit only the known child-completion deficit in partial mode."""

    reasons: list[str] = []
    allowed_worker_rejections = {"required_child_agents_not_completed"}
    output_validation = dict(record.output_validation or {})
    output_reasons = {
        str(item) for item in output_validation.get("reasons") or [] if str(item)
    }
    if record.status == "accepted_draft":
        if output_validation and output_validation.get("accepted") is not True:
            reasons.append("coordinator_output_validation_not_accepted")
    elif record.status == "rejected_output":
        unexpected = sorted(output_reasons - allowed_worker_rejections)
        if unexpected or not output_reasons:
            reasons.append("coordinator_rejection_not_child_completion_only")
            reasons.extend(f"coordinator_output:{item}" for item in unexpected)
    else:
        reasons.append(f"coordinator_status:{record.status}")
    if record.task_id != task.task_id or record.case_id != task.case_id:
        reasons.append("coordinator_run_identity_mismatch")
    if record.timed_out:
        reasons.append("coordinator_timed_out")
    if record.exit_code is not None and int(record.exit_code) != 0:
        reasons.append("coordinator_exit_code_nonzero")
    if float(record.elapsed_s or 0.0) > float(task.budget.timeout_s):
        reasons.append("coordinator_runtime_budget_exceeded")
    if not artifact_validation.get("accepted"):
        reasons.append("coordinator_typed_artifact_invalid")
    if not worker_artifact_validation.get("accepted"):
        reasons.append("coordinator_worker_artifact_invalid")
    if runtime_summary.get("consistent") is not True:
        reasons.append("coordinator_runtime_journal_inconsistent")

    tool_calls = [
        dict(row) for row in record.tool_calls or [] if isinstance(row, dict)
    ]
    if len(tool_calls) > int(task.budget.max_tool_calls):
        reasons.append("coordinator_tool_call_budget_exceeded")
    allowed_tools = {_team_runtime_tool_name(item) for item in task.allowed_tools}
    for call in tool_calls:
        observed = _team_runtime_tool_name(call.get("tool") or call.get("name"))
        if not observed or observed not in allowed_tools:
            reasons.append("coordinator_tool_not_allowed")
            break
    return sorted(set(reasons))


def _team_runtime_tool_name(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    return {
        "wait_agent": "wait",
        "agent_wait": "wait",
        "send_input": "send_message",
        "agent_message": "send_message",
        "agent_spawn": "spawn_agent",
    }.get(name, name)


def _partial_l0_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    row["evidence_level"] = "model_only"
    row["confidence"] = "low"
    row["no_solved_claim"] = True
    row["not_parent_route_proof"] = True
    row["limitations"] = sorted(
        {
            *[str(item) for item in row.get("limitations") or [] if str(item)],
            "partial_child_quorum_l0_only",
        }
    )
    row["required_validation"] = sorted(
        {
            *[
                str(item)
                for item in row.get("required_validation") or []
                if str(item)
            ],
            "independent_host_reaction_validation",
        }
    )
    return row


def _cap_partial_consensus_to_l0(consensus: dict[str, Any]) -> dict[str, Any]:
    capped = dict(consensus)
    proposals: list[dict[str, Any]] = []
    for raw in capped.get("proposals") or []:
        if not isinstance(raw, dict):
            continue
        proposal = dict(raw)
        proposal.update(
            {
                "evidence_level": "model_only",
                "authority_evidence_level": "model_only",
                "confidence": "low",
                "confidence_score": min(
                    0.25, float(proposal.get("confidence_score") or 0.0)
                ),
                "rank_score": min(0.25, float(proposal.get("rank_score") or 0.0)),
                "status": "model_hypothesis",
                "authority_policy": "partial_child_quorum_host_capped",
                "authority_bound": False,
                "validation_tier": "L0",
                "achieved_proof_level": 0,
                "no_solved_claim": True,
                "not_parent_route_proof": True,
            }
        )
        source_records: list[dict[str, Any]] = []
        for source_raw in proposal.get("source_records") or []:
            if not isinstance(source_raw, dict):
                continue
            source = dict(source_raw)
            source.update(
                {
                    "evidence_level": "model_only",
                    "confidence": "low",
                    "producer_evidence_level": "model_only",
                    "producer_confidence": "low",
                    "authority_evidence_level": "model_only",
                    "authority_confidence": "low",
                    "authority_basis": "partial_child_quorum_host_cap",
                    "authority_bound": False,
                    "validation_tier": "L0",
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                }
            )
            source_records.append(source)
        proposal["source_records"] = source_records
        proposals.append(proposal)
    capped["proposals"] = proposals
    source_summary = dict(capped.get("source_summary") or {})
    source_summary["partial_child_quorum_proposal_count"] = len(proposals)
    capped["source_summary"] = source_summary
    semantics = dict(capped.get("semantics") or {})
    semantics.update(
        {
            "child_acceptance_tier": "valid_subset_l0",
            "partial_proposals_forced_to_l0": True,
            "partial_proposals_can_close_route": False,
            "consensus_source": "host_validated_child_final_messages_only",
        }
    )
    capped["semantics"] = semantics
    return capped


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
        safe_role = role if role else f"unassigned-{index}"
        report_path = report_dir / f"{index:02d}-{safe_role}-{message_sha256[:12] or 'empty'}.json"
        report_ref = f"{report_path}#agent={agent_id}"
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
            validation_reasons.extend(
                validate_retrosynthesis_report_envelope_payload(parsed)
            )
            if str(parsed.get("case_id") or "") != str(case_id):
                validation_reasons.append("child_report_case_id_mismatch")
            if _normalize_role(parsed.get("agent_role")) != role:
                validation_reasons.append("child_report_role_mismatch")
            if not _same_smiles(parsed.get("target_smiles"), target_smiles=target_smiles):
                validation_reasons.append("child_report_target_mismatch")

        raw_candidates = (
            list(parsed.get("candidates") or [])
            if isinstance(parsed.get("candidates"), list)
            else []
        )
        candidate_audits: list[dict[str, Any]] = []
        host_candidate_rows: list[dict[str, Any]] = []
        candidate_pass_count = 0
        candidate_rejected_count = 0
        if not validation_reasons:
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                host_candidate, audit, hard_reasons = _audit_child_candidate(
                    raw_candidate,
                    index=candidate_index,
                    role=role,
                    target_smiles=target_smiles,
                    report_ref=report_ref,
                )
                candidate_audits.append(audit)
                if audit.get("accepted") is True:
                    candidate_pass_count += 1
                else:
                    candidate_rejected_count += 1
                validation_reasons.extend(hard_reasons)
                host_candidate_rows.append(host_candidate)
            if raw_candidates and candidate_pass_count == 0:
                validation_reasons.append("child_report_no_admissible_candidates")

        accepted = not validation_reasons
        candidate_count = candidate_pass_count if accepted else 0
        if accepted:
            seen_roles.add(role)
            # Keep quarantined local rows in the fusion input so the canonical
            # route_consensus rejection ledger remains complete. The same pure
            # normalizer will reject them again; only admitted rows can form a
            # proposal.
            candidates.extend(host_candidate_rows)
        report_disposition = (
            "accepted_with_candidate_quarantine"
            if accepted and candidate_rejected_count
            else "accepted_clean"
            if accepted
            else "rejected"
        )
        candidate_admission = {
            "schema_version": "codex_child_candidate_admission_summary.v1",
            "policy": "local_structural_reasons_only.v1",
            "raw_candidate_count": len(raw_candidates),
            "candidate_pass_count": candidate_pass_count,
            "admitted_candidate_count": candidate_count,
            "rejected_candidate_count": candidate_rejected_count,
            "discarded_with_rejected_report_count": (
                len(raw_candidates) if not accepted else 0
            ),
            "audits": candidate_audits,
            "report_accepted": accepted,
            "candidate_quarantine_does_not_grant_evidence_authority": True,
        }
        persisted = {
            "schema_version": "codex_child_report_observation.v2",
            "accepted": accepted,
            "report_disposition": report_disposition,
            "case_id": case_id,
            "role": role,
            "agent_id": agent_id,
            "status": str(child.get("status") or ""),
            "payload": parsed,
            "candidate_count": candidate_count,
            "candidate_admission": candidate_admission,
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
                "report_disposition": report_disposition,
                "role": role,
                "agent_id": agent_id,
                "status": str(child.get("status") or ""),
                "report_ref": report_ref,
                "candidate_count": candidate_count,
                "candidate_admission": candidate_admission,
                "message_bytes": message_bytes,
                "message_sha256": message_sha256,
                "normalization_repairs": normalization_repairs,
                "validation_reasons": sorted(set(validation_reasons)),
            }
        )
    return reports, candidates


def _audit_child_candidate(
    raw_candidate: Any,
    *,
    index: int,
    role: str,
    target_smiles: str,
    report_ref: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    candidate = dict(raw_candidate) if isinstance(raw_candidate, dict) else {}
    producer_digest = _safe_payload_digest(candidate)
    # Source identity is a host capability. Child text cannot impersonate a
    # deterministic provider or mint an independent evidence group.
    candidate["source_channel"] = ROLE_SOURCE_CHANNELS.get(role, "other")
    candidate["report_ref"] = report_ref
    normalized, reasons = normalize_route_candidate(
        candidate,
        default_source_channel=ROLE_SOURCE_CHANNELS.get(role, "other"),
        report_ref=report_ref,
    )
    candidate_reasons = list(reasons)
    if normalized is not None and not _same_smiles(
        normalized.get("product_smiles"),
        target_smiles=target_smiles,
    ):
        candidate_reasons.append(
            "candidate_product_does_not_match_requested_target"
        )
    authority_ceiling_valid = bool(
        normalized is None
        or (
            normalized.get("authority_evidence_level") == "model_only"
            and normalized.get("authority_confidence") == "low"
            and normalized.get("authority_bound") is False
        )
    )
    if not authority_ceiling_valid:
        candidate_reasons.append("codex_candidate_authority_ceiling_violation")
    candidate_reasons = sorted(set(candidate_reasons))
    hard_reasons = sorted(
        reason
        for reason in candidate_reasons
        if reason not in ISOLATABLE_CHILD_CANDIDATE_REASONS
    )
    accepted = bool(normalized is not None and not candidate_reasons)
    audit = {
        "schema_version": "codex_child_candidate_admission.v1",
        "candidate_index": int(index),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_sha256": producer_digest,
        "effective_candidate_sha256": _safe_payload_digest(candidate),
        "accepted": accepted,
        "reasons": candidate_reasons,
        "authority_ceiling": "L0_model_only_low_unbound",
        "canonical_product_smiles": str(
            (normalized or {}).get("product_smiles") or ""
        ),
        "canonical_precursor_smiles": list(
            (normalized or {}).get("precursor_smiles") or []
        ),
        "report_ref": report_ref,
    }
    return (
        candidate,
        audit,
        [f"child_candidate:{index}:non_isolatable:{reason}" for reason in hard_reasons],
    )


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
    reasons = _strict_child_report_envelope_shape_reasons(payload)
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for index, candidate in enumerate(candidates):
            reasons.extend(_strict_child_candidate_shape_reasons(candidate, index=index))
    return sorted(set(reasons))


def _strict_child_report_envelope_shape_reasons(
    payload: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    keys = set(payload)
    if not CHILD_REPORT_KEYS.issubset(keys) or not keys.issubset(
        CHILD_REPORT_KEYS | CHILD_REPORT_OPTIONAL_SAFETY_KEYS
    ):
        reasons.append("child_report_fields_not_exact")
    if (
        "not_parent_route_proof" in payload
        and payload.get("not_parent_route_proof") is not True
    ):
        reasons.append("child_report_parent_route_claim")
    if not isinstance(payload.get("candidates"), list):
        reasons.append("child_report_candidates_not_list")
    if not isinstance(payload.get("evidence_refs"), list) or not all(
        isinstance(item, str) for item in payload.get("evidence_refs") or []
    ):
        reasons.append("child_report_evidence_refs_not_string_list")
    if not isinstance(payload.get("limitations"), list) or not all(
        isinstance(item, str) for item in payload.get("limitations") or []
    ):
        reasons.append("child_report_limitations_not_string_list")
    return sorted(set(reasons))


def _strict_child_candidate_shape_reasons(
    candidate: Any,
    *,
    index: int,
) -> list[str]:
    prefix = f"child_candidate:{index}:"
    if not isinstance(candidate, dict):
        return [prefix + "not_object"]
    reasons: list[str] = []
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
        child["report_disposition"] = str(
            report.get("report_disposition") or "rejected"
        )
        child["report_ref"] = str(report.get("report_ref") or "")
        child["report_candidate_admission"] = dict(
            report.get("candidate_admission") or {}
        )
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
