"""Controlled Codex research worker wrapper for P3.

The worker contract is deliberately conservative. A worker may produce typed
draft artifacts and a trace record, but it cannot mutate route trees, write
production KB entries, or mark a case solved.
"""
from __future__ import annotations

import json
import os
import hashlib
import base64
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.agent.action_contracts import (
    ALLOWED_AGENT_ACTIONS as WORKER_AGENT_ACTION_TYPES,
    PLANNER_SOURCE_HINT_SCHEMA,
    contains_raw_reaction_payload,
)
from cascade_planner.application.strategy_contract import (
    KEY_EVENT_REPAIR_SCOPES,
    normalize_key_event_repair_scope,
)
WORKER_TASK_SCHEMA = "worker_task.v1"
WORKER_RUN_RECORD_SCHEMA = "worker_run_record.v1"
WORKER_OUTPUT_VALIDATION_SCHEMA = "worker_output_validation.v1"
DEFAULT_RETROSYNTHESIS_KEY_FILE = Path(__file__).resolve().parents[2] / "key.txt"

# Ambient ChatGPT auth is snapshotted into an isolated worker home so the
# chemistry worker cannot mutate the user's Codex configuration. Near token
# expiry, however, independent snapshots would all try to spend the same
# single-use refresh token. One process-local refresh lease lets the first
# worker update the ambient authority while unrelated, fresh-token calls stay
# parallel.
_AMBIENT_AUTH_REFRESH_LOCK = threading.RLock()
_AMBIENT_AUTH_REFRESH_WINDOW_S = 600.0

ALLOWED_WORKER_TASK_TYPES = {
    "target_research",
    "stuck_node_research",
    "strategic_disconnection_mining",
    "route_step_materialization",
    "route_chemistry_critique",
    "route_chemistry_edit",
    "paper_matched_strategy_generator",
    "paper_matched_strategy_critic",
    "paper_matched_route_step",
    "paper_matched_key_event_critic",
    "paper_matched_route_critic",
    "paper_matched_route_editor",
    "path_repair_editor",
    "route_audit_research",
    "condition_research",
    "evolution_candidate_research",
    "global_campaign_direction",
}
PAPER_MATCHED_WORKER_TASK_TYPES = frozenset(
    {
        "paper_matched_strategy_generator",
        "paper_matched_strategy_critic",
        "paper_matched_route_step",
        "paper_matched_key_event_critic",
        "paper_matched_route_critic",
        "paper_matched_route_editor",
    }
)
PATH_REPAIR_WORKER_TASK_TYPES = frozenset({"path_repair_editor"})
STRICT_CHEMISTRY_WORKER_TASK_TYPES = (
    PAPER_MATCHED_WORKER_TASK_TYPES | PATH_REPAIR_WORKER_TASK_TYPES
)
STRICT_CHEMISTRY_PERMISSION_PROFILE = "autoplanner-chemistry-worker"
LOCAL_CHEMISTRY_TOOL_TASK_TYPES = frozenset(
    {
        "paper_matched_route_step",
        "paper_matched_key_event_critic",
        "paper_matched_route_critic",
        "paper_matched_route_editor",
        "path_repair_editor",
    }
)
LOCAL_CHEMISTRY_TOOL_NAME = "inspect_mapped_smiles"
ALLOWED_WORKER_ARTIFACT_TYPES = {
    "AgentActionBatch",
    "ResearchReport",
    "RetrosynthesisProposalReport",
    "StrategyCardReport",
    "StrategyPortfolioReport",
    "ChemicalStrategyCritique",
    "GlobalCampaignPlan",
    "EvidenceCard",
    "LiteratureScoutReport",
    "AnalogicalReactionTemplateReport",
    "StrategicDisconnectionCard",
    "LiteratureRouteSegmentCard",
    "SegmentStepCandidate",
    "FailureDiagnosis",
    "StrategicOperator",
    "ConditionCandidate",
    "AuditReport",
    "ProcedureRepairDraft",
    "EvolutionCandidate",
}
FORBIDDEN_PRODUCTION_KEYS = {
    "production_kb_write",
    "write_production_kb",
    "production_write",
    "direct_production_write",
}
FORBIDDEN_ROUTE_TREE_KEYS = {
    "route_tree_write",
    "mutate_route_tree",
    "route_tree_patch",
    "route_tree_actions",
    "candidate_actions",
}


@dataclass
class WorkerBudget:
    timeout_s: float = 30.0
    max_output_bytes: int = 200_000
    max_tool_calls: int | None = 16
    max_worker_runs: int = 1
    reasoning_effort: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerTask:
    task_id: str
    case_id: str
    task_type: str
    required_artifact_type: str
    input_refs: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    budget: WorkerBudget = field(default_factory=WorkerBudget)
    objective: str = ""
    allowed_workdir: str = "."
    dry_run: bool = False
    agent_mode: str = "single"
    child_roles: list[str] = field(default_factory=list)
    codex_auth_mode: str = "auto"
    model: str = ""
    # Host-owned values needed to wrap a compact paper-task wire response.
    # They are deliberately excluded from the serialized WorkerTask so the
    # model does not have to echo orchestration metadata it cannot author.
    host_context: dict[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_TASK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("host_context", None)
        data["budget"] = self.budget.to_dict()
        return data


@dataclass
class WorkerProcessResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    command: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerRunRecord:
    run_id: str
    task_id: str
    case_id: str
    status: str
    backend: str = ""
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    output_artifact: dict[str, Any] | None = None
    output_validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    timed_out: bool = False
    schema_version: str = WORKER_RUN_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WorkerRunner = Callable[[WorkerTask], WorkerProcessResult]


class WorkerTimeoutError(TimeoutError):
    def __init__(self, message: str, *, backend: str, command: list[str] | None = None):
        super().__init__(message)
        self.backend = backend
        self.command = list(command or [])


class WorkerCancelledError(RuntimeError):
    def __init__(self, message: str, *, backend: str, command: list[str] | None = None):
        super().__init__(message)
        self.backend = backend
        self.command = list(command or [])


def run_codex_worker(
    task: WorkerTask,
    *,
    runner: WorkerRunner | None = None,
    mock_output: dict[str, Any] | None = None,
    command: list[str] | None = None,
    use_codex_cli: bool | None = None,
    use_api_json: bool | None = None,
    cancel_event: threading.Event | None = None,
) -> WorkerRunRecord:
    """Run a controlled worker task and validate the resulting draft artifact."""
    task_validation = validate_worker_task(task)
    if not task_validation["accepted"]:
        return WorkerRunRecord(
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="rejected_task",
            output_validation=task_validation,
        )
    if cancel_event is not None and cancel_event.is_set():
        return WorkerRunRecord(
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="cancelled",
            output_validation={
                "accepted": False,
                "reasons": ["delivery_milestone_reached"],
                "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA,
            },
        )

    started = time.monotonic()
    backend = "unselected"
    try:
        if mock_output is not None or task.dry_run:
            backend = "mock_output" if mock_output is not None else "dry_run_mock"
            artifact = mock_output if mock_output is not None else mock_worker_artifact(task)
            process = WorkerProcessResult(stdout=json.dumps(artifact, ensure_ascii=False), exit_code=0, backend=backend)
        elif runner is not None:
            backend = "runner"
            process = runner(task)
        elif command is not None:
            backend = "subprocess_command"
            process = _run_subprocess_worker(
                task,
                command,
                cancel_event=cancel_event,
            )
        elif _use_api_json(use_api_json):
            backend = "api_json"
            process = _run_api_json_worker(task)
        elif _use_codex_cli(use_codex_cli):
            backend = "codex_cli"
            process = _run_codex_cli_worker(task, cancel_event=cancel_event)
        else:
            raise RuntimeError(
                "no worker backend selected; use dry_run/mock_output explicitly or configure codex/api_json"
            )
    except TimeoutError as exc:
        timeout_backend = str(getattr(exc, "backend", "") or backend)
        timeout_command = list(getattr(exc, "command", []) or [])
        return WorkerRunRecord(
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="timeout",
            backend=timeout_backend,
            command=timeout_command,
            stderr=str(exc),
            output_validation={"accepted": False, "reasons": ["timeout"], "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA},
            elapsed_s=round(time.monotonic() - started, 3),
            timed_out=True,
        )
    except WorkerCancelledError as exc:
        return WorkerRunRecord(
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="cancelled",
            backend=str(exc.backend or backend),
            command=list(exc.command),
            stderr=str(exc),
            output_validation={
                "accepted": False,
                "reasons": ["delivery_milestone_reached"],
                "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA,
            },
            elapsed_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        return WorkerRunRecord(
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="worker_error",
            backend=backend,
            stderr=f"{type(exc).__name__}: {exc}",
            output_validation={
                "accepted": False,
                "reasons": ["worker_error"],
                "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA,
            },
            elapsed_s=round(time.monotonic() - started, 3),
        )

    process.backend = process.backend or backend
    canonical_stdout, unicode_repairs = _normalize_reversible_utf8_mojibake(
        process.stdout
    )
    stdout = _truncate(canonical_stdout, task.budget.max_output_bytes)
    artifact = _materialize_paper_matched_artifact(
        task,
        _parse_worker_stdout(stdout),
        backend=process.backend,
    )
    validation = validate_worker_output(task, artifact)
    runtime_reasons = _worker_runtime_reasons(task, process)
    if runtime_reasons:
        validation = dict(validation)
        validation["reasons"] = sorted(set([*validation.get("reasons", []), *runtime_reasons]))
        validation["accepted"] = False
    provider_failure_reason = _worker_provider_failure_reason_from_process(process)
    if provider_failure_reason:
        validation = dict(validation)
        validation["reasons"] = sorted(
            set([*validation.get("reasons", []), provider_failure_reason])
        )
        validation["accepted"] = False
        status = "provider_error"
    else:
        status = "accepted_draft" if validation["accepted"] else "rejected_output"
    return WorkerRunRecord(
        run_id=f"{task.task_id}:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status=status,
        backend=process.backend,
        command=list(process.command or []),
        stdout=stdout,
        stderr=_truncate(process.stderr, task.budget.max_output_bytes),
        exit_code=process.exit_code,
        tool_calls=list(process.tool_calls or []),
        output_artifact=artifact if isinstance(artifact, dict) else None,
        output_validation=validation,
        metadata={
            **dict(process.metadata or {}),
            **(
                {"provider_failure_reason": provider_failure_reason}
                if provider_failure_reason
                else {}
            ),
            **(
                {"unicode_mojibake_repairs": unicode_repairs}
                if unicode_repairs
                else {}
            ),
        },
        usage=dict(process.usage or {}),
        elapsed_s=round(time.monotonic() - started, 3),
    )


def mock_worker_artifact(task: WorkerTask) -> dict[str, Any]:
    """Return a minimal typed draft artifact for dry-run/mock execution."""
    return {
        "schema_version": _typed_artifact_schema_version(task.required_artifact_type),
        "artifact_id": f"{task.task_id}:{task.required_artifact_type}",
        "artifact_type": task.required_artifact_type,
        "case_id": task.case_id,
        "source": "codex_worker_mock",
        "input_refs": list(task.input_refs),
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": task.objective or task.task_type,
    }


def validate_worker_task(task_or_data: WorkerTask | dict[str, Any]) -> dict[str, Any]:
    task = task_or_data if isinstance(task_or_data, WorkerTask) else worker_task_from_dict(task_or_data)
    reasons: list[str] = []
    if task.schema_version != WORKER_TASK_SCHEMA:
        reasons.append("invalid_worker_task_schema")
    if not task.task_id:
        reasons.append("missing_task_id")
    if not task.case_id:
        reasons.append("missing_case_id")
    if task.task_type not in ALLOWED_WORKER_TASK_TYPES:
        reasons.append("invalid_task_type")
    if task.required_artifact_type not in ALLOWED_WORKER_ARTIFACT_TYPES:
        reasons.append("invalid_required_artifact_type")
    if task.budget.timeout_s <= 0:
        reasons.append("invalid_timeout")
    if task.budget.max_output_bytes <= 0:
        reasons.append("invalid_max_output_bytes")
    if (
        task.budget.max_tool_calls is not None
        and task.budget.max_tool_calls < 0
    ):
        reasons.append("invalid_max_tool_calls")
    if task.budget.max_worker_runs <= 0:
        reasons.append("invalid_max_worker_runs")
    if task.agent_mode not in {"single", "coordinator"}:
        reasons.append("invalid_agent_mode")
    if str(task.codex_auth_mode or "auto") not in {"auto", "ambient_codex_cli", "api_key"}:
        reasons.append("invalid_codex_auth_mode")
    if task.agent_mode == "coordinator" and len(task.child_roles) < 2:
        reasons.append("coordinator_requires_multiple_child_roles")
    if len(task.child_roles) > 8:
        reasons.append("child_role_limit_exceeded")
    return {
        "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "task_id": task.task_id,
    }


def validate_worker_output(task: WorkerTask, artifact: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(artifact, dict):
        reasons.append("output_not_json_object")
        artifact = {}
    if not artifact.get("schema_version"):
        reasons.append("missing_schema_version")
    if not artifact.get("artifact_id"):
        reasons.append("missing_artifact_id")
    if artifact.get("case_id") != task.case_id:
        reasons.append("case_id_mismatch")
    if artifact.get("artifact_type") != task.required_artifact_type:
        reasons.append("artifact_type_mismatch")
    if not artifact.get("source"):
        reasons.append("missing_source")
    if artifact.get("validation_status") not in {"draft", "draft_only"}:
        reasons.append("worker_output_must_be_draft")
    if not isinstance(artifact.get("input_refs", []), list):
        reasons.append("input_refs_not_list")
    if not isinstance(artifact.get("evidence_refs", []), list):
        reasons.append("evidence_refs_not_list")
    if _contains_forbidden_production_write(artifact):
        reasons.append("worker_direct_production_kb_write")
    if _contains_solved_claim(artifact):
        reasons.append("worker_direct_solved_claim")
    if _contains_route_tree_mutation(artifact):
        reasons.append("worker_route_tree_mutation")
    if _contains_raw_reaction_injection(artifact):
        reasons.append("worker_raw_reaction_injection")
    if task.task_type == "paper_matched_route_step":
        payload = artifact.get("payload") if isinstance(artifact, dict) else None
        reasons.extend(_paper_matched_route_step_contract_reasons(payload))
    if task.task_type == "paper_matched_key_event_critic":
        payload = artifact.get("payload") if isinstance(artifact, dict) else None
        reasons.extend(_paper_matched_key_event_critic_contract_reasons(payload))
    return {
        "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "artifact_type": str(artifact.get("artifact_type") or ""),
    }


def _paper_matched_route_step_contract_reasons(payload: Any) -> list[str]:
    """Validate one Builder expansion without granting terminal authority."""

    if not isinstance(payload, Mapping):
        return ["paper_route_step_payload_not_object"]
    reasons: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        reasons.append("paper_route_step_candidates_not_list")
        candidates = []
    if any(
        key in payload
        for key in {"builder_action", "builder_reason", "stop_signal", "stop_reason"}
    ):
        reasons.append("paper_route_step_builder_control_forbidden")
    if len(candidates) != 1:
        reasons.append("paper_route_step_disconnection_requires_one_candidate")
    elif not isinstance(candidates[0], Mapping):
        reasons.append("paper_route_step_candidate_not_object")
    else:
        candidate = candidates[0]
        if not str(candidate.get("reaction_family") or "").strip():
            reasons.append("paper_route_step_reaction_intent_missing")
        if str(candidate.get("checkpoint_relation") or "") not in {
            "preparatory",
            "executes_checkpoint",
        }:
            reasons.append("paper_route_step_checkpoint_relation_invalid")
        operations = candidate.get("reaction_operations")
        if not isinstance(operations, list) or not operations:
            reasons.append("paper_route_step_reaction_operations_missing")
        elif not all(isinstance(value, Mapping) for value in operations):
            reasons.append("paper_route_step_reaction_operation_not_object")
        if not isinstance(candidate.get("conditions"), list):
            reasons.append("paper_route_step_conditions_not_list")
    return sorted(set(reasons))


def _paper_matched_key_event_critic_contract_reasons(payload: Any) -> list[str]:
    """Keep chemical verdict and mutation ownership internally consistent."""

    if not isinstance(payload, Mapping):
        return ["paper_key_critic_payload_not_object"]
    assessments = payload.get("step_assessments")
    if not isinstance(assessments, list) or len(assessments) != 1:
        return ["paper_key_critic_focus_assessment_invalid"]
    assessment = assessments[0]
    if not isinstance(assessment, Mapping):
        return ["paper_key_critic_focus_assessment_invalid"]
    if not normalize_key_event_repair_scope(
        assessment.get("repair_scope"),
        verdict=assessment.get("verdict"),
    ):
        return ["paper_key_critic_repair_scope_inconsistent"]
    return []


def worker_task_from_dict(data: dict[str, Any]) -> WorkerTask:
    budget = data.get("budget") or {}
    raw_max_tool_calls = budget.get("max_tool_calls", 16)
    return WorkerTask(
        task_id=str(data.get("task_id") or ""),
        case_id=str(data.get("case_id") or ""),
        task_type=str(data.get("task_type") or ""),
        required_artifact_type=str(data.get("required_artifact_type") or ""),
        input_refs=[str(item) for item in data.get("input_refs") or []],
        allowed_tools=[str(item) for item in data.get("allowed_tools") or []],
        budget=WorkerBudget(
            timeout_s=float(budget.get("timeout_s") or 30.0),
            max_output_bytes=int(budget.get("max_output_bytes") or 200_000),
            max_tool_calls=(
                None
                if raw_max_tool_calls is None
                else int(raw_max_tool_calls)
            ),
            max_worker_runs=int(budget.get("max_worker_runs") or 1),
            reasoning_effort=str(budget.get("reasoning_effort") or ""),
        ),
        objective=str(data.get("objective") or ""),
        allowed_workdir=str(data.get("allowed_workdir") or "."),
        dry_run=bool(data.get("dry_run", False)),
        agent_mode=str(data.get("agent_mode") or "single"),
        child_roles=[str(item) for item in data.get("child_roles") or [] if str(item).strip()],
        codex_auth_mode=str(data.get("codex_auth_mode") or "auto"),
        model=str(data.get("model") or ""),
        host_context={},
        schema_version=str(data.get("schema_version") or WORKER_TASK_SCHEMA),
    )


def _run_subprocess_worker(
    task: WorkerTask,
    command: list[str],
    *,
    cancel_event: threading.Event | None = None,
) -> WorkerProcessResult:
    try:
        returncode, stdout, stderr = _run_worker_command(
            command,
            cwd=Path(task.allowed_workdir).resolve(),
            timeout_s=float(task.budget.timeout_s),
            cancel_event=cancel_event,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerTimeoutError(
            f"worker timeout after {task.budget.timeout_s}s",
            backend="subprocess_command",
            command=command,
        ) from exc
    return WorkerProcessResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=int(returncode),
        backend="subprocess_command",
        command=list(command),
    )


def _run_codex_cli_worker(
    task: WorkerTask,
    *,
    cancel_event: threading.Event | None = None,
) -> WorkerProcessResult:
    schema = _worker_model_output_json_schema(task)
    validate_provider_response_schema(
        schema,
        schema_name=_worker_response_schema_name(task),
    )
    executable = _codex_executable()
    if not _codex_executable_available(executable):
        raise FileNotFoundError(f"Codex CLI executable not found: {executable}")

    audit_root = Path(task.allowed_workdir or ".").resolve()
    audit_root.mkdir(parents=True, exist_ok=True)
    strict_chemistry = task.task_type in STRICT_CHEMISTRY_WORKER_TASK_TYPES
    prompt = _codex_worker_prompt(task)
    worker_temp_root: Path | None = None
    candidate_temp_roots = [] if strict_chemistry else [
        Path(__file__).resolve().parents[2] / ".codex-worker-tmp",
        audit_root / ".autoplanner" / "codex-worker-tmp",
    ]
    for candidate in candidate_temp_roots:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            worker_temp_root = candidate
            break
        except OSError:
            continue
    temp_kwargs = {
        "prefix": "autoplanner_codex_worker_",
        "ignore_cleanup_errors": True,
    }
    if worker_temp_root is not None:
        temp_kwargs["dir"] = str(worker_temp_root)
    with tempfile.TemporaryDirectory(**temp_kwargs) as tmp:
        tmp_path = Path(tmp)
        model_workspace = audit_root
        local_chemistry_tool = task.task_type in LOCAL_CHEMISTRY_TOOL_TASK_TYPES
        if strict_chemistry:
            model_workspace = tmp_path / "model_workspace"
            model_workspace.mkdir(parents=True, exist_ok=True)
            if local_chemistry_tool:
                for name in (
                    "chemistry_inspection.py",
                    "chemistry_inspection_mcp.py",
                ):
                    shutil.copyfile(
                        Path(__file__).resolve().parents[1] / "application" / name,
                        model_workspace / name,
                    )
        output_path = tmp_path / "last_message.json"
        schema_path = tmp_path / "worker_output_schema.json"
        schema_path.write_text(
            json.dumps(schema, indent=2),
            encoding="utf-8",
        )
        env, metadata = _codex_cli_runtime_environment(
            tmp_path,
            model_workspace,
            task,
        )
        if strict_chemistry:
            env = _configure_strict_chemistry_worker_environment(
                env,
                model_workspace=model_workspace,
                audit_root=audit_root,
                enable_local_chemistry_tool=local_chemistry_tool,
            )
            metadata = {
                **metadata,
                "permission_profile": STRICT_CHEMISTRY_PERMISSION_PROFILE,
                "model_workdir": str(model_workspace),
                "audit_workdir": str(audit_root),
                "command_network_enabled": False,
                "shared_workspace_readable": False,
                "local_chemistry_tool": (
                    LOCAL_CHEMISTRY_TOOL_NAME if local_chemistry_tool else ""
                ),
            }
        command = _codex_cli_command(
            executable=_codex_executable_command(executable),
            workdir=model_workspace,
            output_path=output_path,
            schema_path=schema_path,
            runtime_metadata=metadata,
            search_enabled=(
                False if strict_chemistry else _task_allows_cli_search(task)
            ),
            multi_agent_enabled=(
                task.agent_mode == "coordinator" and not strict_chemistry
            ),
            use_permission_profile=strict_chemistry,
        )
        auth_refresh_lease = False
        ambient_home = _ambient_worker_auth_home(metadata)
        worker_home = _worker_codex_home(env)
        if ambient_home is not None and _ambient_auth_refresh_due(ambient_home):
            _AMBIENT_AUTH_REFRESH_LOCK.acquire()
            auth_refresh_lease = True
            _refresh_worker_auth_snapshot(
                source_home=ambient_home,
                worker_home=worker_home,
            )
            # A preceding worker may have refreshed the authority while this
            # call waited. Fresh snapshots do not need to remain serialized.
            if not _ambient_auth_refresh_due(ambient_home):
                _AMBIENT_AUTH_REFRESH_LOCK.release()
                auth_refresh_lease = False
        try:
            try:
                returncode, stdout, stderr = _run_worker_command(
                    command,
                    cwd=model_workspace,
                    input_text=prompt,
                    env=env,
                    timeout_s=float(task.budget.timeout_s),
                    cancel_event=cancel_event,
                    cancel_backend="codex_cli",
                )
            except subprocess.TimeoutExpired as exc:
                raise WorkerTimeoutError(
                    f"worker timeout after {task.budget.timeout_s}s",
                    backend="codex_cli",
                    command=command,
                ) from exc
        finally:
            if ambient_home is not None:
                _publish_refreshed_worker_auth(
                    worker_home=worker_home,
                    ambient_home=ambient_home,
                )
            if auth_refresh_lease:
                _AMBIENT_AUTH_REFRESH_LOCK.release()
        final = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else stdout
        stderr = stderr or ""
        event_audit = _parse_codex_jsonl_events(stdout)
        event_log_path = _write_codex_event_log(
            audit_root,
            task=task,
            stdout=stdout,
        )
        child_agents = _assign_child_roles(
            event_audit["child_agents"],
            roles=task.child_roles,
        )
        metadata = {
            **metadata,
            "agent_mode": task.agent_mode,
            "child_roles": list(task.child_roles),
            "event_summary": event_audit["summary"],
            "event_log_path": str(event_log_path) if event_log_path is not None else "",
            "session_id": event_audit["session_id"],
            "child_agents": child_agents,
        }
        return WorkerProcessResult(
            stdout=final,
            stderr=stderr,
            exit_code=int(returncode),
            backend="codex_cli",
            command=list(command),
            metadata=metadata,
            tool_calls=event_audit["tool_calls"],
            usage=event_audit["usage"],
        )


def _codex_executable_available(executable: str) -> bool:
    path = Path(str(executable)).expanduser()
    separators = [sep for sep in (os.sep, os.altsep) if sep]
    if path.is_absolute() or any(sep in str(executable) for sep in separators):
        return path.exists()
    return shutil.which(executable) is not None


def _codex_executable_command(executable: str) -> list[str]:
    path = Path(str(executable)).expanduser()
    if path.exists() and path.suffix.lower() == ".py":
        return [sys.executable, str(path)]
    return [str(executable)]


def _run_worker_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    cancel_event: threading.Event | None = None,
    cancel_backend: str = "subprocess_command",
) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        start_new_session=True,
    )
    windows_job = _create_windows_kill_job(proc)
    stdin_writer = _start_worker_stdin_writer(proc, input_text)
    stdout_reader, stdout_chunks = _start_worker_pipe_reader(proc.stdout)
    stderr_reader, stderr_chunks = _start_worker_pipe_reader(proc.stderr)
    try:
        deadline = time.monotonic() + float(timeout_s)
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                if windows_job is not None:
                    _close_windows_job(windows_job)
                    windows_job = None
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        _terminate_worker_process_group(proc)
                else:
                    _terminate_worker_process_group(proc)
                _close_worker_pipe(proc.stdin)
                _join_worker_thread(stdout_reader, timeout_s=1.0)
                _join_worker_thread(stderr_reader, timeout_s=1.0)
                raise WorkerCancelledError(
                    "worker cancelled after delivery milestone",
                    backend=cancel_backend,
                    command=command,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_s)
            try:
                proc.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
        _close_windows_job(windows_job)
        windows_job = None
        _join_worker_thread(stdin_writer, timeout_s=1.0)
        _join_worker_thread(stdout_reader, timeout_s=1.0)
        _join_worker_thread(stderr_reader, timeout_s=1.0)
        return int(proc.returncode), "".join(stdout_chunks), "".join(stderr_chunks)
    except subprocess.TimeoutExpired:
        if windows_job is not None:
            _close_windows_job(windows_job)
            windows_job = None
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                _terminate_worker_process_group(proc)
        else:
            _terminate_worker_process_group(proc)
        _close_worker_pipe(proc.stdin)
        # Do not close a Windows pipe from a second thread while its reader is
        # blocked: FileIO.close waits for that read and can turn a 200 ms
        # timeout into the descendant's full lifetime. Readers are daemons and
        # terminate on EOF after the process tree is killed.
        _join_worker_thread(stdout_reader, timeout_s=1.0)
        _join_worker_thread(stderr_reader, timeout_s=1.0)
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        exc = subprocess.TimeoutExpired(command, timeout_s, output=stdout, stderr=stderr)
        raise exc
    except BaseException:
        _close_worker_pipe(proc.stdin)
        if windows_job is not None:
            _close_windows_job(windows_job)
            windows_job = None
        else:
            _terminate_worker_process_group(proc)
        _close_worker_pipe(proc.stdout)
        _close_worker_pipe(proc.stderr)
        raise


def _create_windows_kill_job(proc: subprocess.Popen[str]) -> int | None:
    """Attach a Windows process tree to a kill-on-close Job Object."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return None
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            kernel32.CloseHandle(job)
            return None
        return int(job)
    except Exception:
        return None


def _close_windows_job(handle: int | None) -> None:
    if os.name != "nt" or handle is None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(int(handle)))
    except Exception:
        return


def _start_worker_stdin_writer(proc: subprocess.Popen[str], input_text: str | None) -> threading.Thread:
    def _write_stdin() -> None:
        if proc.stdin is None:
            return
        try:
            if input_text is not None:
                proc.stdin.write(input_text)
                proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            _close_worker_pipe(proc.stdin)

    thread = threading.Thread(target=_write_stdin, name="autoplanner-worker-stdin", daemon=True)
    thread.start()
    return thread


def _start_worker_pipe_reader(pipe: Any) -> tuple[threading.Thread, list[str]]:
    chunks: list[str] = []

    def _read_pipe() -> None:
        if pipe is None:
            return
        try:
            chunks.append(pipe.read() or "")
        except (OSError, ValueError):
            return

    thread = threading.Thread(target=_read_pipe, name="autoplanner-worker-pipe-reader", daemon=True)
    thread.start()
    return thread, chunks


def _join_worker_thread(thread: threading.Thread, *, timeout_s: float) -> None:
    try:
        thread.join(timeout=float(timeout_s))
    except RuntimeError:
        return


def _timeout_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _drain_worker_pipes_after_timeout(proc: subprocess.Popen[str], *, timeout_s: float = 1.0) -> tuple[str, str]:
    """Best-effort legacy drain helper kept for tests/imports.

    Do not call this on Windows timeout paths after killing the parent process:
    descendants may still hold inherited pipe handles and block EOF delivery.
    """
    _close_worker_pipe(proc.stdout)
    _close_worker_pipe(proc.stderr)
    return "", ""


def _terminate_worker_process_group(proc: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP does not contain descendants that create a
        # second session. taskkill /T follows the Windows parent/child tree and
        # prevents timed-out Codex/subagent processes from retaining pipes or
        # the run directory.
        subprocess.run(
            ["taskkill", "/PID", str(int(proc.pid)), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10.0,
        )
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()
        return
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    _kill_worker_process_group(proc)


def _kill_worker_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()


def _close_worker_pipe(pipe: Any) -> None:
    if pipe is None:
        return
    try:
        pipe.close()
    except Exception:
        pass


def _codex_cli_runtime_environment(
    tmp_path: Path,
    workdir: Path,
    task: WorkerTask,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build the subprocess environment used by Codex CLI workers."""
    auth_mode = str(task.codex_auth_mode or "auto").strip().lower()
    if auth_mode == "ambient_codex_cli" or (auth_mode == "auto" and _use_ambient_codex_cli_auth()):
        return _ambient_codex_cli_worker_environment(tmp_path, task)
    try:
        config = _api_json_config(task)
    except RuntimeError:
        if auth_mode == "api_key":
            raise
        env = os.environ.copy()
        return env, {
            "provider": "ambient_codex_cli",
            "base_url_fingerprint": "",
            "model": str(task.model or "") or os.environ.get("AUTOPLANNER_CODEX_WORKER_MODEL") or os.environ.get("OPENAI_MODEL") or "",
            "model_reasoning_effort": _task_reasoning_effort(task),
            "auth_source": "ambient_codex_cli",
            "codex_home": "ambient",
        }
    if task.model:
        config = {**config, "model": str(task.model)}
    config = {**config, "reasoning_effort": _task_reasoning_effort(task)}
    return _codex_cli_worker_environment(tmp_path, workdir, config)


def _ambient_codex_cli_worker_environment(
    tmp_path: Path,
    task: WorkerTask,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Snapshot ambient Codex auth into a writable per-worker home.

    Managed runners commonly mount the user's real ``CODEX_HOME`` read-only.
    Codex still needs to create its state database even for ``--ephemeral``
    executions, so pointing a worker back at that directory makes the CLI fail
    before it can contact the configured provider. Copy only the small,
    read-only inputs needed by the CLI and let all runtime state live under the
    worker's temporary directory.
    """

    source_home = _ambient_codex_home()
    codex_home = tmp_path / "codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)
    copied_inputs: list[str] = []
    # The ambient model cache is tied to the installed Codex schema.  Copying
    # it into every disposable worker caused each real call to reject the
    # stale cache (for example, a missing ``base_instructions`` field) before
    # downloading the same current catalog again.  Auth and installation
    # identity are the only ambient files the worker actually needs.
    for name in ("auth.json", "installation_id"):
        source = source_home / name
        if not source.is_file():
            continue
        shutil.copyfile(source, codex_home / name)
        copied_inputs.append(name)
    source_config = source_home / "config.toml"
    configured_model_provider = ""
    if source_config.is_file():
        configured_model_provider = _ambient_codex_model_provider(source_config)
        (codex_home / "config.toml").write_text(
            _ambient_codex_provider_config(source_config),
            encoding="utf-8",
        )
        copied_inputs.append("config.toml")

    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    return env, {
        "provider": "ambient_codex_cli",
        "base_url_fingerprint": "",
        "model": str(task.model or "")
        or os.environ.get("AUTOPLANNER_CODEX_WORKER_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "",
        "model_reasoning_effort": _task_reasoning_effort(task),
        "auth_source": "ambient_codex_cli_snapshot",
        "codex_home": "ephemeral_ambient",
        "ambient_inputs": copied_inputs,
        "config_mode": "provider_only_snapshot",
        "configured_model_provider": configured_model_provider,
        "transport": _codex_worker_transport(),
        "ambient_codex_home": str(source_home),
    }


def _ambient_codex_home() -> Path:
    configured = str(os.environ.get("CODEX_HOME") or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            return candidate
    return Path.home() / ".codex"


def _ambient_worker_auth_home(metadata: Mapping[str, Any]) -> Path | None:
    if str(metadata.get("auth_source") or "") != "ambient_codex_cli_snapshot":
        return None
    value = str(metadata.get("ambient_codex_home") or "").strip()
    return Path(value).expanduser().resolve() if value else None


def _worker_codex_home(environment: Mapping[str, str]) -> Path:
    value = str(environment.get("CODEX_HOME") or "").strip()
    if not value:
        raise RuntimeError("codex worker home missing")
    return Path(value).expanduser().resolve()


def _ambient_auth_refresh_due(source_home: Path) -> bool:
    expiry = _auth_access_token_expiry(source_home / "auth.json")
    if expiry is None:
        # Unknown ambient auth formats are serialized at the refresh boundary
        # instead of being copied concurrently with no coordination.
        return True
    return expiry - time.time() <= _AMBIENT_AUTH_REFRESH_WINDOW_S


def _auth_access_token_expiry(path: Path) -> float | None:
    try:
        auth = json.loads(path.read_text(encoding="utf-8"))
        token = str(dict(auth.get("tokens") or {}).get("access_token") or "")
        payload = token.split(".", 2)[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        return float(claims["exp"])
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


def _refresh_worker_auth_snapshot(*, source_home: Path, worker_home: Path) -> None:
    source = source_home / "auth.json"
    if source.is_file():
        shutil.copyfile(source, worker_home / "auth.json")


def _publish_refreshed_worker_auth(*, worker_home: Path, ambient_home: Path) -> bool:
    """Publish only a demonstrably newer worker token back to its authority."""

    candidate_path = worker_home / "auth.json"
    ambient_path = ambient_home / "auth.json"
    if not candidate_path.is_file() or not ambient_path.is_file():
        return False
    with _AMBIENT_AUTH_REFRESH_LOCK:
        try:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            ambient = json.loads(ambient_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        candidate_refresh = str(candidate.get("last_refresh") or "")
        ambient_refresh = str(ambient.get("last_refresh") or "")
        candidate_expiry = _auth_access_token_expiry(candidate_path) or 0.0
        ambient_expiry = _auth_access_token_expiry(ambient_path) or 0.0
        if (
            candidate_refresh <= ambient_refresh
            and candidate_expiry <= ambient_expiry
        ):
            return False
        temporary = ambient_home / f".auth.autoplanner.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            temporary.write_text(
                json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(ambient_path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return True


def _ambient_codex_provider_config(source: Path) -> str:
    """Keep model/provider settings while excluding remote MCP/plugin sections."""

    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    kept: list[str] = []
    section = ""
    keep_section = True
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            keep_section = section.startswith("model_providers.")
            if keep_section:
                kept.append(line)
            continue
        if keep_section:
            kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def _ambient_codex_model_provider(source: Path) -> str:
    """Read the active top-level provider without retaining unrelated config."""

    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            break
        if not stripped.startswith("model_provider") or "=" not in stripped:
            continue
        return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _codex_cli_worker_environment(
    tmp_path: Path,
    workdir: Path,
    config: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    codex_home = tmp_path / "codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(
        json.dumps({
            "auth_mode": "apikey",
            "OPENAI_API_KEY": config["api_key"],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (codex_home / "config.toml").write_text(
        "\n".join([
            f"model = {_toml_string(config['model'])}",
            f"model_reasoning_effort = {_toml_string(_task_reasoning_effort_from_config(config))}",
            f"openai_base_url = {_toml_string(config['base_url'])}",
            "",
            f"[projects.{_toml_string(str(workdir))}]",
            'trust_level = "trusted"',
            "",
            "[features]",
            "goals = true",
            "",
        ]),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    metadata = {
        "provider": config["provider"],
        "base_url": config["base_url"],
        "base_url_fingerprint": _base_url_fingerprint(config["base_url"]),
        "model": config["model"],
        "model_reasoning_effort": _task_reasoning_effort_from_config(config),
        "auth_source": str(DEFAULT_RETROSYNTHESIS_KEY_FILE),
        "codex_home": "ephemeral",
    }
    return env, metadata


def _configure_strict_chemistry_worker_environment(
    environment: Mapping[str, str],
    *,
    model_workspace: Path,
    audit_root: Path,
    enable_local_chemistry_tool: bool = False,
) -> dict[str, str]:
    """Install one per-worker least-privilege command profile.

    The Codex parent process retains its model/auth transport.  Only commands
    authored by the chemistry worker are confined to the disposable model
    workspace and no command network.  A narrow stdio MCP tool owns RDKit
    inspection without granting the model a general Python runtime.  The
    durable audit directory and repository remain outside the model's view.
    """

    env = dict(environment)
    codex_home_value = str(env.get("CODEX_HOME") or "").strip()
    if not codex_home_value:
        raise RuntimeError("strict chemistry worker requires an ephemeral CODEX_HOME")
    codex_home = Path(codex_home_value).resolve()
    config_path = codex_home / "config.toml"
    existing = (
        config_path.read_text(encoding="utf-8", errors="replace")
        if config_path.is_file()
        else ""
    )
    retained: list[str] = []
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("sandbox_mode") and "=" in stripped:
            continue
        if stripped.startswith("default_permissions") and "=" in stripped:
            continue
        retained.append(line)
    filesystem_rules: dict[str, str] = {
        ":root": "deny",
        str(audit_root.resolve()): "deny",
        str(model_workspace.resolve()): "write",
    }
    profile_lines = [
        f"[permissions.{STRICT_CHEMISTRY_PERMISSION_PROFILE}]",
        'description = "Isolated local RDKit workspace without command network access."',
        "",
        f"[permissions.{STRICT_CHEMISTRY_PERMISSION_PROFILE}.filesystem]",
    ]
    profile_lines.extend(
        f"{_toml_string(path)} = {_toml_string(access)}"
        for path, access in filesystem_rules.items()
    )
    profile_lines.extend(
        [
            "",
            f"[permissions.{STRICT_CHEMISTRY_PERMISSION_PROFILE}.network]",
            "enabled = false",
            "",
        ]
    )
    config_lines = [
        f"default_permissions = {_toml_string(STRICT_CHEMISTRY_PERMISSION_PROFILE)}",
        "",
        *retained,
        "",
    ]
    if os.name == "nt":
        config_lines.extend(
            [
                "[windows]",
                # Official Codex guidance recommends elevated first, but the
                # current host cannot create its dedicated sandbox user
                # (CreateProcessWithLogonW error 2).  Unelevated is the
                # documented fallback; RDKit remains available through the
                # out-of-sandbox, single-purpose stdio MCP server above.
                'sandbox = "unelevated"',
                "",
            ]
        )
    if enable_local_chemistry_tool:
        server_path = model_workspace / "chemistry_inspection_mcp.py"
        config_lines.extend(
            [
                "[mcp_servers.chemistry_inspection]",
                f"command = {_toml_string(sys.executable)}",
                f"args = [{_toml_string(str(server_path))}]",
                f"cwd = {_toml_string(str(model_workspace))}",
                "required = true",
                f"enabled_tools = [{_toml_string(LOCAL_CHEMISTRY_TOOL_NAME)}]",
                "startup_timeout_sec = 15",
                "tool_timeout_sec = 30",
                "",
                "[mcp_servers.chemistry_inspection.tools.inspect_mapped_smiles]",
                'approval_mode = "approve"',
                "",
            ]
        )
    config_path.write_text(
        "\n".join([*config_lines, *profile_lines]).lstrip(),
        encoding="utf-8",
    )
    # Strict tasks cannot use environment variables to select a broader legacy
    # sandbox/config profile or to import repository code through PYTHONPATH.
    for name in (
        "AUTOPLANNER_CODEX_WORKER_SANDBOX",
        "AUTOPLANNER_CODEX_SANDBOX",
        "AUTOPLANNER_CODEX_WORKER_PROFILE",
        "PYTHONPATH",
    ):
        env.pop(name, None)
    return env


def _task_reasoning_effort(task: WorkerTask) -> str:
    explicit = str(getattr(task.budget, "reasoning_effort", "") or "").strip()
    if explicit:
        return explicit
    return str(
        os.environ.get("AUTOPLANNER_CODEX_WORKER_REASONING_EFFORT")
        or "medium"
    ).strip()


def _task_reasoning_effort_from_config(config: dict[str, str]) -> str:
    return str(
        config.get("reasoning_effort")
        or os.environ.get("AUTOPLANNER_CODEX_WORKER_REASONING_EFFORT")
        or "medium"
    ).strip()


def _use_ambient_codex_cli_auth() -> bool:
    raw = (
        os.environ.get("AUTOPLANNER_CODEX_WORKER_AUTH")
        or os.environ.get("AUTOPLANNER_CODEX_CLI_AUTH")
        or ""
    )
    return str(raw).strip().lower().replace("-", "_") in {
        "ambient",
        "ambient_codex",
        "ambient_codex_cli",
        "current",
        "current_codex",
        "codex_login",
    }


def _toml_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_api_json_worker(task: WorkerTask) -> WorkerProcessResult:
    schema = _worker_model_output_json_schema(task)
    validate_provider_response_schema(
        schema,
        schema_name=_worker_response_schema_name(task),
    )
    config = _api_json_config(task)
    prompt = _codex_worker_prompt(task).replace('"source": "codex_cli"', '"source": "api_json"')
    endpoint_order = _api_endpoint_order(config["endpoint"])
    last_error: Exception | None = None
    command: list[str] = []
    started = time.monotonic()
    for endpoint in endpoint_order:
        command = ["api_json", "POST", f"/{endpoint}"]
        try:
            response = _post_api_json(config, endpoint, prompt, schema, task)
            text = _extract_api_json_text(response, endpoint)
            return WorkerProcessResult(
                stdout=text,
                stderr="",
                exit_code=0,
                backend="api_json",
                command=command,
                metadata={
                    "provider": config["provider"],
                    "base_url_fingerprint": _base_url_fingerprint(config["base_url"]),
                    "model": config["model"],
                    "endpoint": endpoint,
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                usage=dict(response.get("usage") or {}),
            )
        except TimeoutError as exc:
            raise WorkerTimeoutError(
                f"worker timeout after {task.budget.timeout_s}s",
                backend="api_json",
                command=command,
            ) from exc
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            last_error = exc
            if not _should_try_next_api_endpoint(exc):
                break
    raise RuntimeError(f"api_json worker failed: {last_error}")


def _api_json_config(task: WorkerTask | None = None) -> dict[str, str]:
    base_url = (
        os.environ.get("AUTOPLANNER_WORKER_API_BASE_URL")
        or os.environ.get("AUTOPLANNER_CODEX_WORKER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.wellau.com/v1"
    ).rstrip("/")
    api_key = _api_json_key(task)
    model = (
        os.environ.get("AUTOPLANNER_CODEX_WORKER_MODEL")
        or os.environ.get("AUTOPLANNER_WORKER_API_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.5"
    )
    provider = os.environ.get("AUTOPLANNER_WORKER_API_PROVIDER") or _provider_from_base_url(base_url)
    endpoint = os.environ.get("AUTOPLANNER_WORKER_API_ENDPOINT") or _default_api_endpoint(provider)
    if not api_key:
        raise RuntimeError("missing API key for api_json worker")
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "provider": provider,
        "endpoint": endpoint,
    }


def retrosynthesis_worker_key_configured(task: WorkerTask | None = None) -> bool:
    """Return whether a dedicated retrosynthesis worker key is configured."""
    return bool(_api_json_key(task, include_global=False))


def _api_json_key(task: WorkerTask | None = None, *, include_global: bool = True) -> str:
    """Resolve the API key from the repository key.txt file only."""
    return _read_api_key_file(DEFAULT_RETROSYNTHESIS_KEY_FILE)


def _read_api_key_file(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return _normalize_secret_value(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return ""


def _normalize_secret_value(value: str | None) -> str:
    normalized = str(value or "").strip()
    for quote in ('"', "'"):
        if normalized.startswith(quote):
            normalized = normalized[1:]
        if normalized.endswith(quote):
            normalized = normalized[:-1]
    return normalized.strip()


def _api_endpoint_order(endpoint: str) -> list[str]:
    value = str(endpoint or "").strip().lower().replace("-", "_")
    if value in {"chat", "chat_completions", "chat/completions"}:
        return ["chat/completions"]
    if value in {"auto", "responses_then_chat"}:
        return ["responses", "chat/completions"]
    return ["responses"]


def _post_api_json(
    config: dict[str, str],
    endpoint: str,
    prompt: str,
    schema: dict[str, Any],
    task: WorkerTask,
) -> dict[str, Any]:
    payload = _api_json_payload(config["model"], endpoint, prompt, schema)
    request = urllib.request.Request(
        f"{config['base_url']}/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AutoPlanner-api-json-worker/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(task.budget.timeout_s)) as response:
            raw = response.read()
    except socket.timeout as exc:
        raise TimeoutError(str(exc)) from exc
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("api_json_response_not_object")
    return data


def _api_json_payload(model: str, endpoint: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    if endpoint == "chat/completions":
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return exactly one JSON object that satisfies the requested artifact schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "autoplanner_worker_output",
                    "schema": schema,
                    "strict": False,
                },
            },
        }
    return {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "autoplanner_worker_output",
                "schema": schema,
                "strict": False,
            }
        },
    }


def _extract_api_json_text(response: dict[str, Any], endpoint: str) -> str:
    if endpoint == "chat/completions":
        choices = response.get("choices") or []
        if choices:
            message = dict((choices[0] or {}).get("message") or {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(item.get("text") or item.get("content") or "") for item in content if isinstance(item, dict))
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = response.get("output") or []
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "".join(chunks)
    raise ValueError("api_json_missing_output_text")


def _should_try_next_api_endpoint(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code) in {404, 405, 422}
    return False


def _provider_from_base_url(base_url: str) -> str:
    host = str(base_url).split("//")[-1].split("/")[0].lower()
    if "wellau" in host:
        return "wellau"
    if "deepseek" in host:
        return "deepseek"
    if "openai" in host:
        return "openai"
    return host or "openai_compatible"


def _default_api_endpoint(provider: str) -> str:
    if str(provider or "").lower() == "wellau":
        return "chat/completions"
    return "responses"


def _base_url_fingerprint(base_url: str) -> str:
    return hashlib.sha256(str(base_url).encode("utf-8")).hexdigest()[:16]


def _codex_cli_command(
    *,
    executable: str | list[str],
    workdir: Path,
    output_path: Path,
    schema_path: Path,
    runtime_metadata: dict[str, Any] | None = None,
    search_enabled: bool | None = None,
    multi_agent_enabled: bool = False,
    use_permission_profile: bool = False,
) -> list[str]:
    command = list(executable) if isinstance(executable, list) else [executable]
    runtime_metadata = dict(runtime_metadata or {})
    search_allowed = _env_flag("AUTOPLANNER_CODEX_WORKER_SEARCH", default=True)
    if search_enabled is not None:
        search_allowed = bool(search_enabled) and search_allowed
    if search_allowed:
        command.append("--search")
    if multi_agent_enabled:
        command.extend(["--enable", "multi_agent"])
    command.extend([
        "--ask-for-approval",
        "never",
    ])
    if _codex_worker_force_https(runtime_metadata):
        provider_id = "autoplanner_http"
        base_url = str(
            os.environ.get("AUTOPLANNER_CODEX_WORKER_HTTPS_BASE_URL")
            or "https://chatgpt.com/backend-api/codex"
        ).strip()
        command.extend(
            [
                "-c",
                f"model_provider={_toml_string(provider_id)}",
                "-c",
                f"model_providers.{provider_id}.name={_toml_string('OpenAI HTTPS')}",
                "-c",
                f"model_providers.{provider_id}.base_url={_toml_string(base_url)}",
                "-c",
                f"model_providers.{provider_id}.wire_api={_toml_string('responses')}",
                "-c",
                f"model_providers.{provider_id}.requires_openai_auth=true",
                "-c",
                f"model_providers.{provider_id}.supports_websockets=false",
            ]
        )
    command.extend([
        "exec",
    ])
    if (
        str(runtime_metadata.get("codex_home") or "") == "ephemeral"
        and not use_permission_profile
    ):
        command.append("--ignore-user-config")
        base_url = str(runtime_metadata.get("base_url") or "").strip()
        if base_url:
            command.extend(["-c", f"openai_base_url={_toml_string(base_url)}"])
    reasoning = str(runtime_metadata.get("model_reasoning_effort") or "").strip()
    if reasoning:
        command.extend(["-c", f"model_reasoning_effort={_toml_string(reasoning)}"])
    command.extend([
        "--cd",
        str(workdir),
    ])
    # Worker workspaces are deliberately created under the run directory (or
    # another caller-provided temporary root), which is commonly outside a
    # Git repository and therefore not persisted in Codex's trusted-project
    # list.  The worker already has an explicit sandbox/approval policy; the
    # Git trust check is only a CLI launch precondition and otherwise causes
    # zero-token failures before the provider is contacted.
    command.append("--skip-git-repo-check")
    if not use_permission_profile:
        sandbox = _codex_worker_sandbox_mode()
        if sandbox == "bypassed":
            command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            command.extend(["--sandbox", sandbox])
    command.extend([
        "--ephemeral",
        "--color",
        "never",
        "--json",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ])
    model = os.environ.get("AUTOPLANNER_CODEX_WORKER_MODEL") or str(runtime_metadata.get("model") or "").strip()
    if model:
        command.extend(["--model", model])
    profile = os.environ.get("AUTOPLANNER_CODEX_WORKER_PROFILE")
    if profile and not use_permission_profile:
        command.extend(["--profile", profile])
    if _env_flag("AUTOPLANNER_CODEX_WORKER_OSS", default=False):
        command.append("--oss")
        provider = os.environ.get("AUTOPLANNER_CODEX_WORKER_LOCAL_PROVIDER")
        if provider:
            command.extend(["--local-provider", provider])
    command.append("-")
    return command


def _codex_worker_transport() -> str:
    value = str(
        os.environ.get("AUTOPLANNER_CODEX_WORKER_TRANSPORT") or "https"
    ).strip().lower()
    if value in {"http", "https", "sse"}:
        return "https"
    return "auto"


def _codex_worker_force_https(runtime_metadata: Mapping[str, Any]) -> bool:
    if _codex_worker_transport() != "https":
        return False
    if str(runtime_metadata.get("auth_source") or "") != "ambient_codex_cli_snapshot":
        return False
    provider = str(runtime_metadata.get("configured_model_provider") or "openai").strip()
    return provider in {"", "openai", "autoplanner_http"}


def _codex_worker_sandbox_mode() -> str:
    raw = (
        os.environ.get("AUTOPLANNER_CODEX_WORKER_SANDBOX")
        or os.environ.get("AUTOPLANNER_CODEX_SANDBOX")
        or "read-only"
    )
    value = str(raw or "").strip().lower().replace("_", "-")
    if value in {"bypass", "bypassed", "none", "off", "no-sandbox", "dangerously-bypass-approvals-and-sandbox"}:
        return "bypassed"
    if value in {"read-only", "workspace-write", "danger-full-access"}:
        return value
    return "read-only"

def _task_allows_cli_search(task: WorkerTask) -> bool:
    if task.task_type in STRICT_CHEMISTRY_WORKER_TASK_TYPES:
        return False
    if (
        task.budget.max_tool_calls is not None
        and int(task.budget.max_tool_calls) <= 0
    ):
        return False
    allowed_tools = {str(item).strip().lower() for item in task.allowed_tools or []}
    return bool(allowed_tools & {"web_search", "browser", "literature_search"})


def _codex_worker_prompt(task: WorkerTask) -> str:
    if task.task_type in STRICT_CHEMISTRY_WORKER_TASK_TYPES:
        execution_context = (
            "This is a blind self-correcting route-repair task derived from the frozen paper baseline."
            if task.task_type in PATH_REPAIR_WORKER_TASK_TYPES
            else "This is a blind paper-matched chemistry task."
        )
        return "\n".join(
            [
                "Return exactly one JSON object satisfying the supplied output schema; emit no markdown or prose outside JSON.",
                execution_context + " Judge the supplied structures and route context without inferring target identity or claiming evidence, validation, stock, or solved status.",
                "Reason deeply before choosing, but keep authored fields concise and report only the selected result, not hidden deliberation or a long explanation.",
                "Task objective:",
                task.objective,
            ]
        )
    task_json = json.dumps(task.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    coordinator_rules = []
    if task.agent_mode == "coordinator":
        coordinator_rules = [
            "",
            "Multi-agent coordination is mandatory for this task:",
            "- Directly call spawn_agent for every required child role listed in WorkerTask.child_roles.",
            "- Give each child a bounded, independent context and require structured findings.",
            "- Wait for every child, preserve disagreements and source provenance, then synthesize the final artifact.",
            "- Do not replace a failed child with an invented answer; record the failure in limitations.",
        ]
    reaction_rule = (
        "- This task may emit product_smiles and precursor_smiles only inside typed, hypothesis-only campaign steps. Never emit reaction SMILES/SMARTS or any string containing '>>'."
        if task.required_artifact_type == "GlobalCampaignPlan"
        else "- Do not inject raw reaction candidates or reaction SMILES. Avoid strings containing '>>' unless the task explicitly asks for audit of an existing input reference."
    )
    source_rule = (
        "- This is blind strategy design. Do not search for literature, infer target identity/name, or optimize for source availability. Source/evidence fields are optional and carry no strategic weight."
        if task.task_type in {
            "strategic_disconnection_mining",
            "route_chemistry_critique",
        }
        else "- Prefer traceable sources. For literature evidence, include DOI, URL, or local_ref in payload/source metadata."
    )
    return "\n".join([
        "You are a bounded Codex Research Worker inside AutoPlanner.",
        "Your job is to produce one structured draft artifact for the supplied WorkerTask.",
        "",
        "Hard rules:",
        "- Return exactly one JSON object. No markdown fences, no prose outside JSON.",
        "- The JSON must satisfy the supplied output schema.",
        "- The artifact is draft-only. Use validation_status \"draft\" or \"draft_only\".",
        "- Do not mark any route or case solved.",
        "- Do not mutate route trees, write production knowledge-base entries, or claim production promotion.",
        reaction_rule,
        source_rule,
        "- Use only the task context, repository files, and allowed tools implied by the task.",
        *coordinator_rules,
        "",
        "Required artifact wrapper:",
        json.dumps({
            "schema_version": _typed_artifact_schema_version(task.required_artifact_type),
            "artifact_id": f"{task.task_id}:{task.required_artifact_type}",
            "artifact_type": task.required_artifact_type,
            "case_id": task.case_id,
            "source": "codex_cli",
            "input_refs": task.input_refs,
            "evidence_refs": [],
            "validation_status": "draft",
            "summary": "brief summary",
            "payload": {},
        }, ensure_ascii=False, indent=2),
        "",
        _artifact_payload_instruction(task.required_artifact_type, task=task),
        "",
        "WorkerTask:",
        task_json,
    ])


def _artifact_payload_instruction(
    artifact_type: str,
    *,
    task: WorkerTask | None = None,
) -> str:
    paper_task_type = str(task.task_type if task is not None else "")
    paper_instruction = {
        ("paper_matched_strategy_generator", "StrategyCardReport"): (
            "Return one schema-defined one-sentence steering query plus its short identity signature; do not expose the internal comparison or add routes, precursor structures, conditions, evidence, or alternatives."
        ),
        ("paper_matched_strategy_generator", "StrategyPortfolioReport"): (
            "Return exactly three materially distinct one-sentence steering queries plus short identity signatures; do not expose the internal comparison or add routes, precursor structures, conditions, evidence, or extra alternatives."
        ),
        ("paper_matched_strategy_critic", "StrategyPortfolioReport"): (
            "Return exactly three reviewed and, where needed, revised one-sentence steering queries plus their critical assumptions; do not expose the critique, build routes, or add structures, conditions, evidence, or admission claims."
        ),
        ("paper_matched_route_step", "RetrosynthesisProposalReport"): (
            "Return one schema-defined ReactionJSON expansion for the selected node; the host derives structures and exclusively owns MCTS termination, budget exhaustion, stock, solved status, and short-tail stitching."
        ),
        ("paper_matched_route_editor", "RetrosynthesisProposalReport"): (
            "Return one schema-defined dependency-closed replace_span; the host preserves all unlisted rows, derives every precursor, merges the span into the full RouteJSON, and replays the complete route."
        ),
        ("path_repair_editor", "RetrosynthesisProposalReport"): (
            "Return one compact rollback directive naming a current RouteJSON step and the chemical repair goal. The host computes only the rollback-to-blocker dependency path, preserves unrelated rows and the reconnectable suffix, restores the exact mapped frontier, and ordinary one-step Builder calls perform every structural edit."
        ),
        ("paper_matched_route_critic", "ChemicalStrategyCritique"): (
            "Return the schema-defined concise forward audit, marking only concrete chemical contradictions as blocking; Strategy adherence is non-blocking observation metadata."
        ),
        ("paper_matched_key_event_critic", "ChemicalStrategyCritique"): (
            "Return the schema-defined concise audit of the first purported key construction, marking only concrete chemical or Strategy contradictions as blocking."
        ),
    }.get((paper_task_type, artifact_type))
    if paper_instruction:
        return paper_instruction
    if artifact_type == "AgentActionBatch":
        return (
            "For payload, return schema_version=agent_action_batch.v1, case_id, round_index, mode, semantics, and actions. "
            "Select at most 3 actions from the allowed action list in this task. Each action must include "
            "schema_version=agent_action.v1, action_id, action_type, rationale, expected_artifact, success_condition, and payload. "
            "Action payloads use a closed skeleton schema; fill irrelevant string fields with \"\", arrays with [], "
            "boolean fields with false, and numeric fields with 0. Local deterministic normalizers will expand valid skeletons. "
            "If you used allowed planner tools and discovered DOI/title/URL/local-PDF metadata that should guide later source acquisition, "
            "optionally include top-level planner_source_hints. Each hint must use schema_version=planner_source_hint.v1, "
            "evidence_class=planner_source_hint, allowed_use=source_acquisition_hint_only, no_solved_claim=true, and source metadata only. "
            "Allowed action types are: "
            f"{', '.join(sorted(WORKER_AGENT_ACTION_TYPES))}. "
            "This artifact is only an action-selection plan. Do not claim solved, do not include route_status/status=solved, "
            "do not include reaction SMILES or strings containing '>>', and do not include raw route mutations. "
            "If recent rounds produced no useful artifact, either change exploration direction or choose stop_unresolved."
        )
    if artifact_type == "EvidenceCard":
        return (
            "For payload, use the EvidenceCard fields: schema_version, evidence_id, case_id, source_type, "
            "source_title, target_relation, claim_type, route_role, confidence, url/doi/local_ref, "
            "source_record_id, family_id, route_role_detail, limitations, source_metadata, validation_status."
        )
    if artifact_type == "LiteratureScoutReport":
        return (
            "For payload, include schema_version=literature_scout_report.v1, accepted, case_id, source_candidates, "
            "source_refs, search_queries, reasons, limitations, and no_solved_claim=true. Each source candidate must include "
            "schema_version=literature_source_candidate.v1, candidate_id, source_ref, title, doi, url, local_pdf, "
            "source_type, relevance_rationale, expected_scheme_or_compound_labels, extraction_task_recommendations, "
            "access_status, and no_solved_claim. Use native web search for real DOI/title/URL evidence when available. "
            "Do not invent sources, do not mark solved, and do not include reaction SMILES or raw route injections."
        )
    if artifact_type == "AnalogicalReactionTemplateReport":
        return (
            "For payload, include schema_version=analogical_reaction_template_report.v1, accepted, case_id, "
            "templates, source_refs, reasons, and no_solved_claim=true. Each template must include "
            "schema_version=analogical_reaction_template.v1, template_id, relation_type, reaction_class, "
            "mechanistic_class, reaction_center.product_retron_type, template_radius, scope_gap, risk_flags, "
            "required_verification, confidence, no_solved_claim=true, and not_raw_reaction_injection=true. "
            "Analog templates may describe reaction centers and mechanisms, but must not include reaction SMILES, "
            "raw routes, executable route actions, or solved claims."
        )
    if artifact_type == "StrategicDisconnectionCard":
        return (
            "For payload, describe the strategic disconnection without raw reaction injection: "
            "evidence_refs, candidate_kind or retrosynthetic_move, target/frontier context, "
            "strategic_subgoal, anchor_candidate, limitations, and fake-terminal guardrails."
        )
    if artifact_type == "StrategyCardReport":
        return (
            "For payload, return schema_version=strategy_card_report.v1, case_id, target_smiles, "
            "one complete strategy_card, alternatives_considered, selection_rationale, limitations, "
            "and no_route_or_solved_claim=true. This is strategy selection only: do not output "
            "precursor SMILES, ReactionJSON operations, conditions, sources, or a complete route. "
            "Use anchor_bond_changes for target atom pairs that bind the route search, and use "
            "precursor_only_bond_changes for conceptual precursor bonds that may be absent from "
            "the target. Include conceptual_precursor_roles and required_reactive_features when "
            "the strategy depends on a specific reactive pair. Compare at least three materially "
            "different high-level strategies before selecting one."
        )
    if artifact_type == "StrategyPortfolioReport":
        return (
            "For payload, return schema_version=strategy_portfolio_report.v1, case_id, target_smiles, "
            "exactly three complete strategy_cards, selection_rationale, limitations, and "
            "no_route_or_solved_claim=true. Each card is a hypothesis only: do not output "
            "precursor SMILES, ReactionJSON operations, conditions, sources, or a complete route. "
            "The three cards must differ in skeletal logic and graph-edit signature."
        )
    if artifact_type == "RetrosynthesisProposalReport":
        strategy_first = bool(
            task is not None
            and task.task_type
            in {
                "strategic_disconnection_mining",
                "route_step_materialization",
                "route_chemistry_edit",
            }
        )
        route_materialization = bool(
            task is not None
            and task.task_type
            in {
                "route_step_materialization",
                "route_chemistry_edit",
            }
        )
        return (
            "For payload, return schema_version=retrosynthesis_proposal_report.v1, case_id, agent_role, "
            "target_smiles, candidates, evidence_refs, limitations, and no_solved_claim=true. Each candidate "
            "must contain product_smiles plus precursor_smiles as a list of individual components, a concise "
            "reaction_family, product_retron_type, and transformation_rationale, optional conditions/catalyst/enzyme, limitations, required_validation, "
            "no_solved_claim=true, and not_parent_route_proof=true. Product and precursor SMILES are advisory "
            "typed hypotheses, and product_retron_type is an advisory product-side classification only; never emit "
            "a reaction SMILES string, reaction SMARTS, or a key named reaction_smiles/rxn/raw_reaction. "
            + (
                " For route_step_materialization or route_chemistry_edit, do not echo the supplied immutable StrategyCard; the host binds it. "
                "candidate.reaction_operations describes the candidate root step and candidate.route_json may contain the complete ordered linear route. "
                "Every route_json step must contain its own ordered atom-map reaction_operations. Set candidate.precursor_smiles to [] and every route_json step precursor_smiles to []: "
                "the host deterministically derives canonical precursors from ReactionJSON and never treats a second model-redrawn precursor as structure authority. "
                "For route_chemistry_edit, return a complete revised route_json by default; use route_patch only for conditions or an isolated single-step repair whose product boundary and dependencies stay unchanged. The Editor may insert, delete, reorder, change functional-group states and reaction handles, replace a disconnection, and change route length or terminal leaves while preserving the campaign target, complete target-rooted connectivity, and overall Strategy intent. "
                "Repair every supplied blocker as one coordinated route. Never truncate an unresolved suffix or terminate at an unavailable advanced intermediate merely to lower the blocking fraction. If no defensible complete repair exists, return route_patch=[] and route_json=null so the host preserves the original route and critique. "
                "Conditions are optional hypotheses, but never emit placeholders such as 'screen', 'TBD', "
                "'to be determined', 'not specified', or 'as needed'; if no concrete reagent/catalyst/solvent/temperature "
                "or enzyme hypothesis is available, emit an empty conditions list and explain the gap in limitations."
                if route_materialization
                else " For strategic_disconnection_mining, candidate.strategy_card must contain the complete strategic contract and candidate.reaction_operations must encode its mapped edit hypothesis."
            )
            + (
                " For this strategy-first task, do not search for or fabricate sources, and do not add source_channel, source_refs, evidence_refs, evidence_level, or confidence to candidates."
                if strategy_first
                else " Source/evidence metadata may be supplied only when grounded in the task inputs or allowed tools."
            )
        )
    if artifact_type == "ChemicalStrategyCritique":
        return (
            "For payload, return schema_version=chemical_strategy_critique.v1 and independently forward-audit the supplied frozen route. "
            "Assess every step for atom provenance, plausible mechanism, functional-group compatibility, site/chemoselectivity, stereochemical outcome, sequence ordering, competing pathways, and enzyme identity/capability where applicable. "
            "When a Strategy is supplied, independently verify that an actual serialized transformation executes its named key construction; self-reported key or anchor labels are not evidence. In a final route audit, record a missing or substituted construction only as strategy_adherence=false and do not reject chemistry merely to force the steering Strategy. In a key-event checkpoint audit, a concrete topology or sequence contradiction in the focus action may still block that proposed action. "
            "Use step verdict pass for coherent chemistry, uncertain for unresolved conditions/precedent/scope/selectivity without contradiction, and reject only for a concrete chemical contradiction in the serialized route (or the focus-action contradiction allowed by a key-event audit). Set overall_assessment=reject if any step rejects, uncertain if none reject and at least one is uncertain, otherwise viable. "
            "For each reject, identify the exact step_id and smallest structure-local replacement boundary while preserving unrelated non-blocking steps and the complete target-to-terminal-leaf synthesis boundary. Never recommend truncating the route or deleting an unresolved suffix to improve the blocking fraction. "
            "Include strategy_adherence, step_assessments, route_level_risks, repair_actions, experimental_variables, no_reaction_proof=true, no_source_authority=true, and no_solved_claim=true. "
            "Do not search the web, cite sources, infer the target name, or defer chemical judgment to literature availability."
        )
    if artifact_type == "GlobalCampaignPlan":
        return (
            "For payload, return schema_version=global_campaign_plan.v1 and a whole-campaign plan bound to the run_id, mode, context_sha256, and graph_revision in the objective. "
            "context_sha256 must exactly equal the first WorkerTask.input_refs value; never substitute prompt_context_sha256 or recompute a digest. "
            "Include route_families, multi_step_skeletons, strategic_disconnections, shared_intermediates, critical_unknowns, source_plan, fallback_strategies, frontier_priorities, pivot_conditions, stop_conditions, portfolio_rationale, and limitations. "
            "Every skeleton step must contain parseable product_smiles and precursor_smiles, transformation_hypothesis, required_validation, hypothesis_only=true, and one or two advisory condition_predictions. "
            "Each condition prediction must use authority_scope=model_predicted_condition and not_reaction_proof=true; use empty strings or arrays only for inapplicable condition fields, and do not attach source authority to a model prediction. "
            "Every source_plan entry must expose verified DOI, patent-publication, or primary-URL identifiers in source_refs, or an empty source_refs list when none was verified. "
            "Coordinate alternatives and shared intermediates globally. Never claim validation, proof, stock closure, completion, or solved status, and never emit reaction SMILES/SMARTS or '>>'."
        )
    if artifact_type == "LiteratureRouteSegmentCard":
        return (
            "For payload, include schema_version, segment_id, case_id, target_smiles, evidence_refs, source_title, "
            "source_type, trigger_reasons, validation_status, and 2-5 structured steps. Each step must "
            "include schema_version, step_id, product_smiles, reactant_smiles, evidence_refs, source_ref, "
            "relation_type, applicability, condition_candidate, and scope_gap for analogs. Do not include "
            "reaction SMILES."
        )
    if artifact_type == "SegmentStepCandidate":
        return (
            "For payload, include one structured segment step: step_id, product_smiles, reactant_smiles, "
            "evidence_refs, source_ref, relation_type, applicability, condition_candidate, and scope_gap "
            "for analogs. Do not include reaction SMILES."
        )
    if artifact_type == "ConditionCandidate":
        return (
            "For payload, include step_id, source_type or condition_source_type, condition_status, "
            "reagent/catalyst/enzyme/solvent/temperature/ph/buffer/atmosphere where evidence-backed, "
            "evidence_refs, hazard_flags or risk_flags, and confidence."
        )
    if artifact_type == "ProcedureRepairDraft":
        return (
            "For payload, return schema_version=procedure_repair_draft.v1, step_id, reaction_class, "
            "diagnosis, conditions, missing_information, risk_flags, repair_actions, "
            "authority_scope=model_predicted_condition, no_exact_source_authority=true, and "
            "no_experimental_validation_claim=true. Conditions must include reagents, catalyst, "
            "base, solvent, temperature, time, atmosphere, addition_order, workup, purification, "
            "and yield_percent; use empty values when the blind input does not support a field."
        )
    if artifact_type == "FailureDiagnosis":
        return "For payload, include failure_mode or reason, evidence_refs/input_refs, affected frontier, and bounded next actions."
    if artifact_type == "StrategicOperator":
        return (
            "For payload, include a bounded search-policy draft derived from validated evidence/disconnection refs. "
            "Do not include raw reactions; include budgets and guardrails."
        )
    if artifact_type == "EvolutionCandidate":
        return "For payload, include candidate_id, candidate_type, evidence_refs, validation_status, and candidate-layer only promotion intent."
    if artifact_type == "AuditReport":
        return "For payload, include audit findings, risk flags, evidence/input refs, and no solved claim unless externally audited."
    return "For payload, include concise findings, evidence/input references, limitations, and recommended bounded next actions."


def _worker_output_json_schema(task: WorkerTask) -> dict[str, Any]:
    paper_matched = task.task_type in STRICT_CHEMISTRY_WORKER_TASK_TYPES
    properties = {
        "schema_version": (
            {
                "type": "string",
                "enum": [_typed_artifact_schema_version(task.required_artifact_type)],
            }
            if paper_matched
            else {"type": "string"}
        ),
        "artifact_id": (
            {
                "type": "string",
                "enum": [f"{task.task_id}:{task.required_artifact_type}"],
            }
            if paper_matched
            else {"type": "string"}
        ),
        "artifact_type": {"type": "string", "enum": [task.required_artifact_type]},
        "case_id": {"type": "string", "enum": [task.case_id]},
        "source": (
            {"type": "string", "enum": ["codex_cli", "api_json"]}
            if paper_matched
            else {"type": "string"}
        ),
        "input_refs": _string_array_schema(max_items=0 if paper_matched else None),
        "evidence_refs": _string_array_schema(max_items=0 if paper_matched else None),
        "validation_status": {"type": "string", "enum": ["draft", "draft_only"]},
        "summary": _short_text_schema(160) if paper_matched else {"type": "string"},
        "payload": _worker_payload_json_schema(task),
    }
    return _strict_object_schema(
        properties,
        required=[
            "schema_version",
            "artifact_id",
            "artifact_type",
            "case_id",
            "source",
            "input_refs",
            "evidence_refs",
            "validation_status",
            "summary",
            "payload",
        ],
    )


def _worker_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    artifact_type = str(task.required_artifact_type or "")
    if artifact_type == "AgentActionBatch":
        return _agent_action_batch_payload_json_schema(task)
    if artifact_type == "EvidenceCard":
        return _evidence_card_payload_json_schema(task)
    if artifact_type == "LiteratureScoutReport":
        return _literature_scout_payload_json_schema(task)
    if artifact_type == "AnalogicalReactionTemplateReport":
        return _analogical_template_report_payload_json_schema(task)
    if artifact_type == "RetrosynthesisProposalReport":
        return _retrosynthesis_proposal_report_payload_json_schema(task)
    if artifact_type == "StrategyCardReport":
        return _strategy_card_report_payload_json_schema(task)
    if artifact_type == "StrategyPortfolioReport":
        return _strategy_portfolio_report_payload_json_schema(task)
    if artifact_type == "ChemicalStrategyCritique":
        return _chemical_strategy_critique_payload_json_schema(task)
    if artifact_type == "GlobalCampaignPlan":
        return _global_campaign_plan_payload_json_schema(task)
    if artifact_type == "LiteratureRouteSegmentCard":
        return _literature_route_segment_payload_json_schema(task)
    if artifact_type == "SegmentStepCandidate":
        return _segment_step_json_schema()
    if artifact_type == "ConditionCandidate":
        return _condition_candidate_json_schema()
    if artifact_type == "ProcedureRepairDraft":
        return _procedure_repair_draft_json_schema()
    return _generic_payload_json_schema()


def _strict_object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required or properties.keys()),
    }


def _short_text_schema(max_length: int) -> dict[str, Any]:
    return {"type": "string", "maxLength": max(1, int(max_length))}


def _string_array_schema(
    *,
    min_items: int | None = None,
    max_items: int | None = None,
    item_max_length: int | None = None,
) -> dict[str, Any]:
    item_schema: dict[str, Any] = {"type": "string"}
    if item_max_length is not None:
        item_schema["maxLength"] = max(1, int(item_max_length))
    schema: dict[str, Any] = {"type": "array", "items": item_schema}
    if min_items is not None:
        schema["minItems"] = max(0, int(min_items))
    if max_items is not None:
        schema["maxItems"] = max(0, int(max_items))
    return schema


def _nullable_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Make a strict-schema property nullable while keeping it required.

    OpenAI Structured Outputs requires every property of a strict object to be
    listed in ``required``.  Null is the explicit representation for fields
    that are inapplicable to one reaction operation or strategy variant.
    """

    result = dict(schema)
    schema_type = result.get("type")
    if isinstance(schema_type, str):
        result["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list) and "null" not in schema_type:
        result["type"] = [*schema_type, "null"]
    if isinstance(result.get("enum"), list) and None not in result["enum"]:
        result["enum"] = [*result["enum"], None]
    return result


def _reaction_operation_json_schema() -> dict[str, Any]:
    """Return the strict, semantic ReactionJSON operation union.

    Structured Outputs requires every field of an object variant to be
    required.  It does not require mutually exclusive operations to share one
    large nullable object, so each primitive owns only the fields it uses.
    """

    positive_map = {"type": "integer", "minimum": 1}
    def operation(op: str, properties: Mapping[str, Any]) -> dict[str, Any]:
        return _strict_object_schema(
            {"op": {"type": "string", "enum": [op]}, **dict(properties)}
        )

    return {
        "anyOf": [
            operation("break_bond", {"map_a": positive_map, "map_b": positive_map}),
            operation(
                "add_bond",
                {
                    "map_a": positive_map,
                    "map_b": positive_map,
                },
            ),
            operation(
                "change_bond_order",
                {
                    "map_a": positive_map,
                    "map_b": positive_map,
                    "delta": {"type": "number"},
                },
            ),
            operation(
                "change_atom",
                {"map_idx": positive_map, "formal_charge": {"type": "integer"}},
            ),
            operation(
                "change_atom",
                {
                    "map_idx": positive_map,
                    "isotope": {"type": "integer", "minimum": 0},
                },
            ),
            operation(
                "set_explicit_h",
                {
                    "map_idx": positive_map,
                    "count": {"type": "integer", "minimum": 0},
                    "no_implicit": {"type": "boolean"},
                },
            ),
            operation(
                "add_group",
                {"map_idx": positive_map, "fragment_smiles": {"type": "string"}},
            ),
            operation(
                "remove_group",
                {
                    "map_indices": {
                        "type": "array",
                        "items": positive_map,
                        "minItems": 1,
                    }
                },
            ),
            operation("invert_stereocenter", {"map_idx": positive_map}),
            operation("clear_stereocenter", {"map_idx": positive_map}),
            operation(
                "set_bond_stereo",
                {
                    "map_a": positive_map,
                    "map_b": positive_map,
                    "stereo": {
                        "type": "string",
                        "enum": ["NONE", "ANY"],
                    },
                },
            ),
            operation(
                "set_bond_stereo",
                {
                    "map_a": positive_map,
                    "map_b": positive_map,
                    "stereo": {
                        "type": "string",
                        "enum": ["Z", "E", "CIS", "TRANS"],
                    },
                },
            ),
            operation(
                "set_tetrahedral_stereo",
                {
                    "map_idx": positive_map,
                    "configuration": {
                        "type": "string",
                        "enum": ["R", "S"],
                    },
                },
            ),
        ]
    }


def _paper_editor_route_step_json_schema() -> dict[str, Any]:
    """Only the fields the Editor must author for one revised route row."""

    return _strict_object_schema(
        {
            "step_id": _short_text_schema(160),
            "product_smiles": {"type": "string"},
            "reaction_family": _short_text_schema(160),
            "conditions": _string_array_schema(max_items=4, item_max_length=160),
            "catalyst": _short_text_schema(160),
                "reaction_operations": {
                    "type": "array",
                    "items": _reaction_operation_json_schema(),
                    "minItems": 1,
                },
        }
    )


def _worker_model_output_json_schema(task: WorkerTask) -> dict[str, Any]:
    """Schema shown to the model; the host adds the durable artifact wrapper."""

    if task.task_type not in STRICT_CHEMISTRY_WORKER_TASK_TYPES:
        return _worker_output_json_schema(task)
    if task.task_type in {
        "paper_matched_strategy_generator",
        "paper_matched_strategy_critic",
    }:
        if task.required_artifact_type == "StrategyPortfolioReport":
            return _strict_object_schema(
                {
                    "strategy_cards": {
                        "type": "array",
                        "items": _paper_strategy_card_json_schema(),
                        "minItems": 3,
                        "maxItems": 3,
                    }
                }
            )
        return _paper_strategy_card_json_schema()
    if task.task_type == "paper_matched_route_step":
        return _strict_object_schema(
            {
                "checkpoint_relation": {
                    "type": "string",
                    "enum": ["preparatory", "executes_checkpoint"],
                },
                "reaction_intent": _short_text_schema(300),
                "reaction_operations": {
                    "type": "array",
                    "items": _reaction_operation_json_schema(),
                    "minItems": 1,
                },
                "conditions": _string_array_schema(
                    max_items=4, item_max_length=160
                ),
            }
        )
    if task.task_type == "paper_matched_route_editor":
        return _strict_object_schema(
            {
                "repair_summary": _short_text_schema(500),
                "replace_span": _strict_object_schema(
                    {
                        "remove_step_ids": {
                            "type": "array",
                            "items": _short_text_schema(160),
                            "minItems": 1,
                            "maxItems": 25,
                        },
                        "revised_steps": {
                            "type": "array",
                            "items": _paper_editor_route_step_json_schema(),
                            "minItems": 1,
                            "maxItems": 25,
                        },
                    }
                ),
            }
        )

    if task.task_type == "path_repair_editor":
        return _strict_object_schema(
            {
                "rollback_start_step_id": _short_text_schema(160),
                "rebuild_through_step_id": _short_text_schema(160),
                "additional_coupled_blocker_step_ids": {
                    "type": "array",
                    "items": _short_text_schema(160),
                },
                "preserved_suffix_compatible": {"type": "boolean"},
                "repair_goal": _short_text_schema(500),
                "active_constraints": _string_array_schema(
                    max_items=5,
                    item_max_length=240,
                ),
            }
        )
    if task.task_type == "paper_matched_key_event_critic":
        return _strict_object_schema(
            {
                "checkpoint_match": {"type": "boolean"},
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "uncertain", "reject"],
                },
                "blocking_type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "structure",
                        "missing_reactive_handle",
                        "mechanism",
                        "atom_provenance",
                        "conditions",
                        "functional_group_compatibility",
                        "chemoselectivity",
                        "stereochemistry",
                        "sequence_dependency",
                        "competing_pathway",
                    ],
                },
                "repair_scope": {
                    "type": "string",
                    "enum": list(KEY_EVENT_REPAIR_SCOPES),
                },
                "reasons": _string_array_schema(max_items=2, item_max_length=260),
                "suggested_revision": _short_text_schema(420),
            }
        )
    if task.task_type == "paper_matched_route_critic":
        step_assessment = _strict_object_schema(
            {
                "step_id": _short_text_schema(160),
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "uncertain", "reject"],
                },
                "blocking": {"type": "boolean"},
                "blocking_type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "structure",
                        "missing_reactive_handle",
                        "mechanism",
                        "atom_provenance",
                        "conditions",
                        "functional_group_compatibility",
                        "chemoselectivity",
                        "stereochemistry",
                        "sequence_dependency",
                        "competing_pathway",
                    ],
                },
                "reasons": _string_array_schema(max_items=2, item_max_length=260),
                "condition_assessment": _short_text_schema(320),
                "suggested_revision": _short_text_schema(420),
            }
        )
        properties: dict[str, Any] = {
                "overall_assessment": {
                    "type": "string",
                    "enum": ["viable", "uncertain", "reject"],
                },
                "strategy_adherence": {"type": "boolean"},
                "step_assessments": {
                    "type": "array",
                    "items": step_assessment,
                    "minItems": 1,
                    "maxItems": 32,
                },
                "route_level_risks": _string_array_schema(
                    max_items=4, item_max_length=280
                ),
                "repair_actions": _string_array_schema(
                    max_items=4, item_max_length=360
                ),
                "coupled_blocker_groups": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": _short_text_schema(160),
                        "minItems": 2,
                    },
                },
                "limitations": _string_array_schema(
                    max_items=2, item_max_length=240
                ),
        }
        return _strict_object_schema(properties)
    return _worker_output_json_schema(task)


_PROVIDER_RESPONSE_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maximum",
        "maxItems",
        "maxLength",
        "minimum",
        "minItems",
        "minLength",
        "multipleOf",
        "pattern",
        "patternProperties",
        "properties",
        "required",
        "title",
        "type",
    }
)
_PROVIDER_RESPONSE_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_PROVIDER_RESPONSE_STRING_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "time",
        "uuid",
    }
)


def _worker_response_schema_name(task: WorkerTask) -> str:
    return f"{task.task_type}/{task.required_artifact_type}"


def _provider_schema_path(path: str, key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
        return f"{path}.{key}"
    return f"{path}[{json.dumps(str(key), ensure_ascii=False)}]"


def _provider_schema_error(
    reason: str,
    *,
    schema_name: str,
    path: str,
    detail: str,
) -> None:
    raise ValueError(
        "provider_response_schema_"
        f"{reason}:{schema_name}:{path}:{detail}"
    )


def validate_provider_response_schema(
    schema: Mapping[str, Any],
    *,
    schema_name: str,
) -> None:
    """Validate the actual model schema against the Structured Outputs subset.

    This is deliberately the only local provider-schema validator.  Both
    worker backends and orchestration startup call it on the output of
    ``_worker_model_output_json_schema``; no copied schema or web-layer gate is
    involved.
    """

    name = str(schema_name or "unnamed")

    def fail(reason: str, path: str, detail: str) -> None:
        _provider_schema_error(
            reason,
            schema_name=name,
            path=path,
            detail=detail,
        )

    if not isinstance(schema, Mapping):
        fail("invalid_root", "$", "schema_not_object")
    if schema.get("type") != "object" or "anyOf" in schema:
        fail("invalid_root", "$", "root_must_be_object_without_anyOf")

    stats = {
        "enum_values": 0,
        "object_properties": 0,
        "schema_string_length": 0,
    }

    def account_schema_string(value: Any, path: str) -> None:
        if not isinstance(value, str):
            return
        stats["schema_string_length"] += len(value)
        if stats["schema_string_length"] > 120_000:
            fail("limit_exceeded", path, "schema_string_length_gt_120000")

    def schema_types(node: Mapping[str, Any], path: str) -> set[str]:
        value = node.get("type")
        if value is None:
            return set()
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not values:
            fail("invalid_keyword", f"{path}.type", "type_must_be_string_or_array")
        normalized = [str(item) for item in values]
        if len(normalized) != len(set(normalized)):
            fail("invalid_keyword", f"{path}.type", "duplicate_type")
        unsupported = sorted(set(normalized) - _PROVIDER_RESPONSE_SCHEMA_TYPES)
        if unsupported:
            fail(
                "unsupported_type",
                f"{path}.type",
                ",".join(unsupported),
            )
        return set(normalized)

    def require_keyword_type(
        node: Mapping[str, Any],
        types: set[str],
        path: str,
        keywords: set[str],
        allowed_types: set[str],
    ) -> None:
        if not types:
            return
        for keyword in sorted(keywords.intersection(node)):
            if not types.intersection(allowed_types):
                fail(
                    "invalid_keyword_context",
                    f"{path}.{keyword}",
                    "type=" + ",".join(sorted(types)),
                )

    def walk(node: Any, path: str, object_depth: int) -> None:
        if not isinstance(node, Mapping):
            fail("invalid_schema_node", path, "schema_node_not_object")

        for keyword in node:
            if keyword not in _PROVIDER_RESPONSE_SCHEMA_KEYWORDS:
                fail(
                    "unsupported_keyword",
                    _provider_schema_path(path, str(keyword)),
                    str(keyword),
                )

        types = schema_types(node, path)
        require_keyword_type(
            node,
            types,
            path,
            {"maxLength", "minLength", "pattern", "format"},
            {"string"},
        )
        require_keyword_type(
            node,
            types,
            path,
            {
                "exclusiveMaximum",
                "exclusiveMinimum",
                "maximum",
                "minimum",
                "multipleOf",
            },
            {"integer", "number"},
        )
        require_keyword_type(
            node,
            types,
            path,
            {"items", "maxItems", "minItems"},
            {"array"},
        )
        require_keyword_type(
            node,
            types,
            path,
            {
                "additionalProperties",
                "patternProperties",
                "properties",
                "required",
            },
            {"object"},
        )

        if "$ref" in node:
            reference = node["$ref"]
            if not isinstance(reference, str) or not reference.startswith("#"):
                fail("invalid_keyword", f"{path}.$ref", "only_local_refs_supported")

        if "format" in node and node["format"] not in _PROVIDER_RESPONSE_STRING_FORMATS:
            fail(
                "unsupported_format",
                f"{path}.format",
                str(node["format"]),
            )
        if "pattern" in node and not isinstance(node["pattern"], str):
            fail("invalid_keyword", f"{path}.pattern", "pattern_not_string")

        for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
            if keyword in node and (
                not isinstance(node[keyword], int)
                or isinstance(node[keyword], bool)
                or node[keyword] < 0
            ):
                fail(
                    "invalid_keyword",
                    f"{path}.{keyword}",
                    "nonnegative_integer_required",
                )
        for minimum_key, maximum_key in (
            ("minLength", "maxLength"),
            ("minItems", "maxItems"),
        ):
            if (
                minimum_key in node
                and maximum_key in node
                and node[minimum_key] > node[maximum_key]
            ):
                fail(
                    "invalid_keyword",
                    f"{path}.{maximum_key}",
                    f"less_than_{minimum_key}",
                )

        for keyword in (
            "exclusiveMaximum",
            "exclusiveMinimum",
            "maximum",
            "minimum",
            "multipleOf",
        ):
            if keyword in node and (
                not isinstance(node[keyword], (int, float))
                or isinstance(node[keyword], bool)
            ):
                fail(
                    "invalid_keyword",
                    f"{path}.{keyword}",
                    "number_required",
                )
        if "multipleOf" in node and node["multipleOf"] <= 0:
            fail("invalid_keyword", f"{path}.multipleOf", "must_be_positive")

        enum = node.get("enum")
        if enum is not None:
            if not isinstance(enum, list) or not enum:
                fail("invalid_keyword", f"{path}.enum", "nonempty_array_required")
            stats["enum_values"] += len(enum)
            if stats["enum_values"] > 1_000:
                fail("limit_exceeded", f"{path}.enum", "enum_values_gt_1000")
            enum_string_length = 0
            for index, value in enumerate(enum):
                account_schema_string(value, f"{path}.enum[{index}]")
                if isinstance(value, str):
                    enum_string_length += len(value)
            if len(enum) > 250 and enum_string_length > 15_000:
                fail(
                    "limit_exceeded",
                    f"{path}.enum",
                    "large_enum_string_length_gt_15000",
                )
        if "const" in node:
            account_schema_string(node["const"], f"{path}.const")

        child_depth = object_depth
        is_object = "object" in types or any(
            key in node
            for key in ("additionalProperties", "patternProperties", "properties")
        )
        if is_object:
            child_depth += 1
            if child_depth > 10:
                fail("limit_exceeded", path, "object_nesting_depth_gt_10")
            properties = node.get("properties")
            if not isinstance(properties, Mapping):
                fail("invalid_object", f"{path}.properties", "object_required")
            if node.get("additionalProperties") is not False:
                fail(
                    "invalid_object",
                    f"{path}.additionalProperties",
                    "must_be_false",
                )
            required = node.get("required")
            if not isinstance(required, list) or not all(
                isinstance(value, str) for value in required
            ):
                fail("invalid_object", f"{path}.required", "string_array_required")
            if len(required) != len(set(required)) or set(required) != set(properties):
                fail(
                    "invalid_object",
                    f"{path}.required",
                    "all_and_only_properties_must_be_required",
                )
            stats["object_properties"] += len(properties)
            if stats["object_properties"] > 5_000:
                fail(
                    "limit_exceeded",
                    f"{path}.properties",
                    "object_properties_gt_5000",
                )
            for property_name, property_schema in properties.items():
                property_path = _provider_schema_path(
                    f"{path}.properties",
                    str(property_name),
                )
                account_schema_string(property_name, property_path)
                walk(property_schema, property_path, child_depth)

            pattern_properties = node.get("patternProperties", {})
            if not isinstance(pattern_properties, Mapping):
                fail(
                    "invalid_object",
                    f"{path}.patternProperties",
                    "object_required",
                )
            for pattern, pattern_schema in pattern_properties.items():
                pattern_path = _provider_schema_path(
                    f"{path}.patternProperties",
                    str(pattern),
                )
                account_schema_string(pattern, pattern_path)
                walk(pattern_schema, pattern_path, child_depth)

        definitions = node.get("$defs", {})
        if not isinstance(definitions, Mapping):
            fail("invalid_keyword", f"{path}.$defs", "object_required")
        for definition_name, definition_schema in definitions.items():
            definition_path = _provider_schema_path(
                f"{path}.$defs",
                str(definition_name),
            )
            account_schema_string(definition_name, definition_path)
            walk(definition_schema, definition_path, object_depth)

        if "items" in node:
            walk(node["items"], f"{path}.items", child_depth)

        if "anyOf" in node:
            variants = node["anyOf"]
            if not isinstance(variants, list) or not variants:
                fail("invalid_keyword", f"{path}.anyOf", "nonempty_array_required")
            for index, variant in enumerate(variants):
                walk(variant, f"{path}.anyOf[{index}]", child_depth)

    walk(schema, "$", 0)


def preflight_worker_response_schemas(tasks: Iterable[WorkerTask]) -> None:
    """Compile and validate every reachable Worker schema without a model call."""

    for task in tasks:
        schema = _worker_model_output_json_schema(task)
        validate_provider_response_schema(
            schema,
            schema_name=_worker_response_schema_name(task),
        )


def _materialize_paper_matched_artifact(
    task: WorkerTask,
    raw: Any,
    *,
    backend: str,
) -> Any:
    """Wrap a compact paper-task wire result in the durable host artifact."""

    if task.task_type not in STRICT_CHEMISTRY_WORKER_TASK_TYPES:
        return raw
    if not isinstance(raw, Mapping):
        return raw
    result = dict(raw)
    # Tests, journal replay, and older workers may already return the durable
    # envelope.  Keep that input stable during the contract migration.
    if result.get("artifact_type") == task.required_artifact_type:
        return result

    context = dict(task.host_context or {})
    target = str(context.get("target_smiles") or "")
    payload: dict[str, Any]
    if task.task_type in {
        "paper_matched_strategy_generator",
        "paper_matched_strategy_critic",
    }:
        if task.required_artifact_type == "StrategyPortfolioReport":
            payload = {
                "schema_version": "strategy_portfolio_report.v1",
                "case_id": task.case_id,
                "target_smiles": target,
                "strategy_cards": [
                    dict(value)
                    for value in result.get("strategy_cards") or []
                    if isinstance(value, Mapping)
                ],
                "no_route_or_solved_claim": True,
            }
        else:
            payload = {
                "schema_version": "strategy_card_report.v1",
                "case_id": task.case_id,
                "target_smiles": target,
                "strategy_card": result,
                "no_route_or_solved_claim": True,
            }
    elif task.task_type == "paper_matched_route_step":
        selected_product = str(context.get("selected_product") or "")
        candidate = {
            "schema_version": "retrosynthesis_candidate.v1",
            "candidate_id": f"{task.task_id}:candidate:1",
            "product_smiles": selected_product,
            "precursor_smiles": [],
            "checkpoint_relation": str(
                result.get("checkpoint_relation") or ""
            ),
            "reaction_family": str(result.get("reaction_intent") or ""),
            "conditions": list(result.get("conditions") or []),
            "limitations": [],
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "reaction_operations": [
                dict(value)
                for value in result.get("reaction_operations") or []
                if isinstance(value, Mapping)
            ],
        }
        payload = {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": task.case_id,
            "agent_role": "route_builder",
            "target_smiles": target,
            "candidates": [candidate],
            "evidence_refs": [],
            "limitations": [],
            "no_solved_claim": True,
        }
    elif task.task_type == "paper_matched_route_editor":
        raw_span = result.get("replace_span")
        span = dict(raw_span) if isinstance(raw_span, Mapping) else {}
        revised_steps = []
        for value in span.get("revised_steps") or []:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            row["precursor_smiles"] = []
            row["no_solved_claim"] = True
            row["not_parent_route_proof"] = True
            revised_steps.append(row)
        payload = {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": task.case_id,
            "agent_role": "route_editor",
            "target_smiles": target,
            "candidates": [
                {
                    "schema_version": "retrosynthesis_candidate.v1",
                    "candidate_id": f"{task.task_id}:candidate:1",
                    "repair_summary": str(result.get("repair_summary") or ""),
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "replace_span": {
                        "remove_step_ids": [
                            str(value)
                            for value in span.get("remove_step_ids") or []
                            if str(value)
                        ],
                        "revised_steps": revised_steps,
                    },
                }
            ],
            "evidence_refs": [],
            "limitations": [],
            "no_solved_claim": True,
        }
    elif task.task_type == "path_repair_editor":
        payload = {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": task.case_id,
            "agent_role": "route_editor",
            "target_smiles": target,
            "candidates": [
                {
                    "schema_version": "retrosynthesis_candidate.v1",
                    "candidate_id": f"{task.task_id}:candidate:1",
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                    "repair_directive": {
                        "rollback_start_step_id": str(
                            result.get("rollback_start_step_id") or ""
                        ),
                        "rebuild_through_step_id": str(
                            result.get("rebuild_through_step_id") or ""
                        ),
                        "additional_coupled_blocker_step_ids": [
                            str(value)
                            for value in result.get(
                                "additional_coupled_blocker_step_ids"
                            )
                            or []
                            if str(value).strip()
                        ],
                        "preserved_suffix_compatible": (
                            result.get("preserved_suffix_compatible") is True
                        ),
                        "repair_goal": str(result.get("repair_goal") or ""),
                        "active_constraints": [
                            str(value)
                            for value in result.get("active_constraints") or []
                            if str(value).strip()
                        ],
                    },
                }
            ],
            "evidence_refs": [],
            "limitations": [],
            "no_solved_claim": True,
        }
    elif task.task_type == "paper_matched_key_event_critic":
        verdict = str(result.get("verdict") or "uncertain")
        checkpoint_match = result.get("checkpoint_match") is True
        blocking_type = str(result.get("blocking_type") or "none")
        repair_scope = str(result.get("repair_scope") or "")
        focus_step_id = str(task.host_context.get("focus_step_id") or "")
        payload = {
            "schema_version": "chemical_strategy_critique.v1",
            "case_id": task.case_id,
            "checkpoint_match": checkpoint_match,
            "overall_assessment": {
                "pass": "viable",
                "reject": "reject",
            }.get(verdict, "uncertain"),
            # Adherence records whether the serialized graph event matches the
            # Strategy; chemical uncertainty is represented by verdict rather
            # than contradicting that structural fact.
            "strategy_adherence": checkpoint_match,
            "step_assessments": [
                {
                    "step_id": focus_step_id,
                    "verdict": verdict,
                    "blocking": verdict == "reject",
                    "blocking_type": blocking_type,
                    "repair_scope": repair_scope,
                    "reasons": list(result.get("reasons") or [])[:2],
                    "condition_assessment": "",
                    "suggested_revision": str(
                        result.get("suggested_revision") or ""
                    ),
                }
            ],
            "route_level_risks": [],
            "repair_actions": [],
            "limitations": [],
            "no_reaction_proof": True,
            "no_source_authority": True,
            "no_solved_claim": True,
        }
    elif task.task_type == "paper_matched_route_critic":
        payload = {
            "schema_version": "chemical_strategy_critique.v1",
            "case_id": task.case_id,
            **result,
            "no_reaction_proof": True,
            "no_source_authority": True,
            "no_solved_claim": True,
        }
    else:
        return raw

    source = "api_json" if str(backend) == "api_json" else "codex_cli"
    return {
        "schema_version": _typed_artifact_schema_version(
            task.required_artifact_type
        ),
        "artifact_id": f"{task.task_id}:{task.required_artifact_type}",
        "artifact_type": task.required_artifact_type,
        "case_id": task.case_id,
        "source": source,
        "input_refs": [],
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": task.task_type,
        "payload": payload,
    }


def _generic_payload_json_schema() -> dict[str, Any]:
    return _strict_object_schema({
        "schema_version": {"type": "string"},
        "summary": {"type": "string"},
        "findings": _string_array_schema(),
        "evidence_refs": _string_array_schema(),
        "limitations": _string_array_schema(),
        "recommended_next_actions": _string_array_schema(),
        "validation_status": {"type": "string", "enum": ["draft", "draft_only"]},
    })


def _strategy_card_json_schema() -> dict[str, Any]:
    biocatalytic_intent = _strict_object_schema(
        {
            "mode": {
                "type": "string",
                "enum": [
                    "enzyme_reaction",
                    "whole_cell_transformation",
                    "chemoenzymatic_cascade",
                ],
            },
            "enzyme_classes": _string_array_schema(),
            "ec_numbers": _string_array_schema(),
            "candidate_ids": _string_array_schema(),
            "whole_cell_hosts": _string_array_schema(),
            "selectivity_objective": {"type": "string"},
            "substrate_scope_basis": {"type": "string"},
            "cofactor_assessment": {
                "type": "string",
                "enum": ["required", "not_required", "unresolved"],
            },
            "intended_chemical_step_equivalent_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
            },
            "fallback_policy": {"type": "string"},
            "validation_plan": _string_array_schema(),
        }
    )
    strategy_card_properties = {
        "scaffold_motif": {"type": "string"},
        "key_forward_transformation": {"type": "string"},
        "forward_transformation_class": {"type": "string"},
        "retrosynthetic_simplification": {"type": "string"},
        "key_bond_changes": _string_array_schema(),
        "anchor_bond_changes": _string_array_schema(),
        "precursor_only_bond_changes": _string_array_schema(),
        "bond_order_changes": _string_array_schema(),
        "conceptual_precursor_roles": _string_array_schema(),
        "required_reactive_features": _string_array_schema(),
        "atom_fragment_provenance": _string_array_schema(),
        "functional_group_conflicts": _string_array_schema(),
        "protection_policy": {"type": "string"},
        "stereochemical_plan": {"type": "string"},
        "stereochemical_control_basis": {"type": "string"},
        "convergence_plan": {"type": "string"},
        "strategic_step_count": {"type": "integer", "minimum": 1, "maximum": 2},
        "skeleton_change_class": {"type": "string"},
        "expected_complexity_drop": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "orthogonality_basis": {"type": "string"},
        "strategy_signature": {"type": "string"},
        "substrate_specific_failure_modes": _string_array_schema(),
        "fallback_strategy": {"type": "string"},
        "strategy_basis": {"type": "string"},
        "execution_domain": _nullable_schema({
            "type": "string",
            "enum": ["chemical", "enzymatic", "whole_cell", "hybrid", "mechanistic"],
        }),
        "biocatalytic_intent": _nullable_schema(biocatalytic_intent),
    }
    return _strict_object_schema(strategy_card_properties)


def _biocatalytic_step_json_schema() -> dict[str, Any]:
    """Model-authored biological metadata; exact structures remain host-owned."""

    return _strict_object_schema(
        {
            "mode": {
                "type": "string",
                "enum": [
                    "enzyme_reaction",
                    "whole_cell_transformation",
                    "chemoenzymatic_cascade",
                ],
            },
            "enzyme_label": {"type": "string"},
            "enzyme_classes": _string_array_schema(),
            "ec_numbers": _string_array_schema(),
            "candidate_ids": _string_array_schema(),
            "sequence_refs": _string_array_schema(),
            "whole_cell_hosts": _string_array_schema(),
            "selectivity_objective": {"type": "string"},
            "substrate_scope_basis": {"type": "string"},
            "cofactor_assessment": {
                "type": "string",
                "enum": ["required", "not_required", "unresolved"],
            },
            "cofactor_requirements": _string_array_schema(),
            "cofactor_regenerations": _string_array_schema(),
            "cosubstrates": _string_array_schema(),
            "precedent_refs": _string_array_schema(),
            "validation_plan": _string_array_schema(),
        }
    )


def _paper_strategy_card_json_schema() -> dict[str, Any]:
    return _strict_object_schema(
        {
            "strategy_query": _short_text_schema(600),
            "critical_assumption": _short_text_schema(300),
            "critic_checkpoint": _short_text_schema(360),
        }
    )


def _strategy_card_report_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    if task.task_type == "paper_matched_strategy_generator":
        return _strict_object_schema(
            {
                "schema_version": {
                    "type": "string",
                    "enum": ["strategy_card_report.v1"],
                },
                "case_id": {"type": "string", "enum": [task.case_id]},
                "target_smiles": {"type": "string"},
                "strategy_card": _paper_strategy_card_json_schema(),
                "no_route_or_solved_claim": {"type": "boolean", "enum": [True]},
            }
        )
    alternative = _strict_object_schema(
        {
            "candidate_label": {"type": "string"},
            "key_forward_transformation": {"type": "string"},
            "key_bond_changes": _string_array_schema(),
            "advantages": _string_array_schema(),
            "risks": _string_array_schema(),
            "decision": {"type": "string", "enum": ["selected", "rejected"]},
        }
    )
    return _strict_object_schema(
        {
            "schema_version": {
                "type": "string",
                "enum": ["strategy_card_report.v1"],
            },
            "case_id": {"type": "string", "enum": [task.case_id]},
            "target_smiles": {"type": "string"},
            "strategy_card": _strategy_card_json_schema(),
            "alternatives_considered": {
                "type": "array",
                "items": alternative,
                "minItems": 3,
                "maxItems": 5,
            },
            "selection_rationale": {"type": "string"},
            "limitations": _string_array_schema(),
            "no_route_or_solved_claim": {"type": "boolean", "enum": [True]},
        }
    )


def _retrosynthesis_proposal_report_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    strategy_card = _strategy_card_json_schema()
    reaction_operation = _reaction_operation_json_schema()
    host_derived_precursor_schema = (
        {"type": "array", "items": {"type": "string"}, "maxItems": 0}
        if task.task_type
        in {
            "route_step_materialization",
            "route_chemistry_edit",
            "paper_matched_route_step",
            "paper_matched_route_editor",
        }
        else _string_array_schema()
    )

    if task.task_type == "paper_matched_route_step":
        candidate = _strict_object_schema(
            {
                "schema_version": {
                    "type": "string",
                    "enum": ["retrosynthesis_candidate.v1"],
                },
                "candidate_id": {"type": "string"},
                "product_smiles": {"type": "string"},
                "precursor_smiles": host_derived_precursor_schema,
                "checkpoint_relation": {
                    "type": "string",
                    "enum": ["preparatory", "executes_checkpoint"],
                },
                "reaction_family": _short_text_schema(160),
                "conditions": _string_array_schema(
                    max_items=4,
                    item_max_length=160,
                ),
                "limitations": _string_array_schema(
                    max_items=2,
                    item_max_length=220,
                ),
                "no_solved_claim": {"type": "boolean", "enum": [True]},
                "not_parent_route_proof": {"type": "boolean", "enum": [True]},
                "reaction_operations": {
                    "type": "array",
                    "items": reaction_operation,
                    "minItems": 1,
                },
            }
        )
        return _strict_object_schema(
            {
                "schema_version": {
                    "type": "string",
                    "enum": ["retrosynthesis_proposal_report.v1"],
                },
                "case_id": {"type": "string", "enum": [task.case_id]},
                "agent_role": {"type": "string"},
                "target_smiles": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": candidate,
                    "minItems": 1,
                    "maxItems": 1,
                },
                "evidence_refs": _string_array_schema(max_items=0),
                "limitations": _string_array_schema(max_items=0),
                "no_solved_claim": {"type": "boolean", "enum": [True]},
            }
        )

    if task.task_type == "paper_matched_route_editor":
        wire_route_step = _paper_editor_route_step_json_schema()
        route_step = _strict_object_schema(
            {
                **dict(wire_route_step.get("properties") or {}),
                "precursor_smiles": host_derived_precursor_schema,
                "no_solved_claim": {"type": "boolean", "enum": [True]},
                "not_parent_route_proof": {
                    "type": "boolean",
                    "enum": [True],
                },
            }
        )
        candidate = _strict_object_schema(
            {
                "schema_version": {
                    "type": "string",
                    "enum": ["retrosynthesis_candidate.v1"],
                },
                "candidate_id": _short_text_schema(160),
                "repair_summary": _short_text_schema(500),
                "no_solved_claim": {"type": "boolean", "enum": [True]},
                "not_parent_route_proof": {"type": "boolean", "enum": [True]},
                "replace_span": _strict_object_schema(
                    {
                        "remove_step_ids": {
                            "type": "array",
                            "items": _short_text_schema(160),
                            "minItems": 1,
                            "maxItems": 25,
                        },
                        "revised_steps": {
                            "type": "array",
                            "items": route_step,
                            "minItems": 1,
                            "maxItems": 25,
                        },
                    }
                ),
            }
        )
        return _strict_object_schema(
            {
                "schema_version": {
                    "type": "string",
                    "enum": ["retrosynthesis_proposal_report.v1"],
                },
                "case_id": {"type": "string", "enum": [task.case_id]},
                "agent_role": {"type": "string"},
                "target_smiles": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": candidate,
                    "minItems": 1,
                    "maxItems": 1,
                },
                "evidence_refs": _string_array_schema(max_items=0),
                "limitations": _string_array_schema(max_items=0),
                "no_solved_claim": {"type": "boolean", "enum": [True]},
            }
        )

    if task.task_type == "path_repair_editor":
        candidate = _strict_object_schema(
            {
                "schema_version": {
                    "type": "string",
                    "enum": ["retrosynthesis_candidate.v1"],
                },
                "candidate_id": _short_text_schema(160),
                "no_solved_claim": {"type": "boolean", "enum": [True]},
                "not_parent_route_proof": {
                    "type": "boolean",
                    "enum": [True],
                },
                "repair_directive": _strict_object_schema(
                    {
                        "rollback_start_step_id": _short_text_schema(160),
                        "rebuild_through_step_id": _short_text_schema(160),
                        "additional_coupled_blocker_step_ids": {
                            "type": "array",
                            "items": _short_text_schema(160),
                        },
                        "preserved_suffix_compatible": {"type": "boolean"},
                        "repair_goal": _short_text_schema(500),
                        "active_constraints": _string_array_schema(
                            max_items=5,
                            item_max_length=240,
                        ),
                    }
                ),
            }
        )
        return _strict_object_schema(
            {
                "schema_version": {
                    "type": "string",
                    "enum": ["retrosynthesis_proposal_report.v1"],
                },
                "case_id": {"type": "string", "enum": [task.case_id]},
                "agent_role": {"type": "string"},
                "target_smiles": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": candidate,
                    "minItems": 1,
                    "maxItems": 1,
                },
                "evidence_refs": _string_array_schema(max_items=0),
                "limitations": _string_array_schema(max_items=0),
                "no_solved_claim": {"type": "boolean", "enum": [True]},
            }
        )


    candidate_properties = {
        "schema_version": {"type": "string", "enum": ["retrosynthesis_candidate.v1"]},
        "candidate_id": {"type": "string"},
        "product_smiles": {"type": "string"},
        "precursor_smiles": host_derived_precursor_schema,
        "reaction_family": {"type": "string"},
        "product_retron_type": {"type": "string"},
        "transformation_rationale": {"type": "string"},
        "conditions": _string_array_schema(),
        "catalyst": {"type": "string"},
        "enzyme": {"type": "string"},
        "execution_domain": {
            "type": "string",
            "enum": ["chemical", "enzymatic", "whole_cell", "hybrid", "mechanistic"],
        },
        "biocatalytic_step": _nullable_schema(_biocatalytic_step_json_schema()),
        "limitations": _string_array_schema(),
        "required_validation": _string_array_schema(),
        "no_solved_claim": {"type": "boolean", "enum": [True]},
        "not_parent_route_proof": {"type": "boolean", "enum": [True]},
        "reaction_operations": {"type": "array", "items": reaction_operation},
    }
    # A paper-protocol Route Builder returns an editable complete linear route
    # object.  Keep this nullable for legacy proposal callers, but the
    # paper_synthex director profile makes the array mandatory at admission.
    route_step = _strict_object_schema(
        {
            "step_id": {"type": "string"},
            "product_smiles": {"type": "string"},
            "precursor_smiles": host_derived_precursor_schema,
            "reaction_family": {"type": "string"},
            "product_retron_type": {"type": "string"},
            "transformation_rationale": {"type": "string"},
            "conditions": _string_array_schema(),
            "catalyst": {"type": "string"},
            "enzyme": {"type": "string"},
            "execution_domain": {
                "type": "string",
                "enum": ["chemical", "enzymatic", "whole_cell", "hybrid", "mechanistic"],
            },
            "biocatalytic_step": _nullable_schema(_biocatalytic_step_json_schema()),
            "limitations": _string_array_schema(),
            "required_validation": _string_array_schema(),
            "no_solved_claim": {"type": "boolean", "enum": [True]},
            "not_parent_route_proof": {"type": "boolean", "enum": [True]},
            "reaction_operations": {"type": "array", "items": reaction_operation},
        }
    )
    candidate_properties["route_json"] = _nullable_schema(
        {"type": "array", "items": route_step, "maxItems": 25}
    )
    if task.task_type == "route_chemistry_edit":
        route_patch_item = _strict_object_schema(
            {
                "op": {
                    "type": "string",
                    "enum": [
                        "replace_step",
                        "insert_after",
                        "delete_step",
                        "reorder",
                    ],
                },
                "step_id": {"type": "string"},
                "after_step_id": {"type": "string"},
                "step_ids": _string_array_schema(),
                "step": _nullable_schema(route_step),
            }
        )
        candidate_properties["route_patch"] = _nullable_schema(
            {"type": "array", "items": route_patch_item, "maxItems": 50}
        )
    if task.task_type not in {
        "route_step_materialization",
        "route_chemistry_edit",
    }:
        candidate_properties["strategy_card"] = strategy_card
    if task.task_type not in {
        "strategic_disconnection_mining",
        "route_step_materialization",
        "route_chemistry_edit",
    }:
        candidate_properties["source_channel"] = {
            "type": "string",
            "enum": [
                "codex_strategy",
                "codex_literature",
                "codex_chemoenzymatic",
                "codex_critic",
                "chem_enzy",
                "literature_exact",
                "literature_analogy",
                "template",
                "stock",
                "human",
                "other",
            ],
        }
        candidate_properties["source_refs"] = _string_array_schema()
        candidate_properties["evidence_refs"] = _string_array_schema()
        candidate_properties["evidence_level"] = {
            "type": "string",
            "enum": ["model_only", "analogy", "computational", "literature_exact"],
        }
        candidate_properties["confidence"] = {
            "type": "string",
            "enum": ["low", "medium", "medium_high", "high"],
        }
    candidate = _strict_object_schema(
        candidate_properties,
    )
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["retrosynthesis_proposal_report.v1"]},
        "case_id": {"type": "string", "enum": [task.case_id]},
        "agent_role": {"type": "string"},
        "target_smiles": {"type": "string"},
        "candidates": {"type": "array", "items": candidate, "maxItems": 24},
        "evidence_refs": _string_array_schema(),
        "limitations": _string_array_schema(),
        "no_solved_claim": {"type": "boolean", "enum": [True]},
    })


def _strategy_portfolio_report_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    paper_matched = task.task_type in {
        "paper_matched_strategy_generator",
        "paper_matched_strategy_critic",
    }
    strategy_card_schema = (
        _paper_strategy_card_json_schema()
        if paper_matched
        else _strategy_card_json_schema()
    )
    properties: dict[str, Any] = {
        "schema_version": {
            "type": "string",
            "enum": ["strategy_portfolio_report.v1"],
        },
        "case_id": {"type": "string", "enum": [task.case_id]},
        "target_smiles": {"type": "string"},
        "strategy_cards": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": strategy_card_schema,
        },
        "no_route_or_solved_claim": {"type": "boolean", "enum": [True]},
    }
    if not paper_matched:
        properties.update(
            {
                "selection_rationale": _short_text_schema(360),
                "limitations": _string_array_schema(
                    max_items=2,
                    item_max_length=220,
                ),
            }
        )
    return _strict_object_schema(
        properties
    )


def _chemical_strategy_critique_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    if task.task_type in {
        "paper_matched_route_critic",
        "paper_matched_key_event_critic",
    }:
        step_assessment_properties: dict[str, Any] = {
                "step_id": _short_text_schema(160),
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "uncertain", "reject"],
                },
                "blocking": {"type": "boolean"},
                "blocking_type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "structure",
                        "missing_reactive_handle",
                        "mechanism",
                        "atom_provenance",
                        "conditions",
                        "functional_group_compatibility",
                        "chemoselectivity",
                        "stereochemistry",
                        "sequence_dependency",
                        "competing_pathway",
                    ],
                },
                "reasons": _string_array_schema(
                    max_items=2,
                    item_max_length=260,
                ),
                "condition_assessment": _short_text_schema(320),
                "suggested_revision": _short_text_schema(420),
        }
        if task.task_type == "paper_matched_key_event_critic":
            step_assessment_properties["repair_scope"] = {
                "type": "string",
                "enum": list(KEY_EVENT_REPAIR_SCOPES),
            }
        step_assessment = _strict_object_schema(step_assessment_properties)
        properties: dict[str, Any] = {
                "schema_version": {
                    "type": "string",
                    "enum": ["chemical_strategy_critique.v1"],
                },
                "case_id": {"type": "string", "enum": [task.case_id]},
                "overall_assessment": {
                    "type": "string",
                    "enum": ["viable", "uncertain", "reject"],
                },
                "strategy_adherence": {"type": "boolean"},
                "step_assessments": {
                    "type": "array",
                    "items": step_assessment,
                    "minItems": 1,
                    "maxItems": 32,
                },
                "route_level_risks": _string_array_schema(
                    max_items=4,
                    item_max_length=280,
                ),
                "repair_actions": _string_array_schema(
                    max_items=4,
                    item_max_length=360,
                ),
                "limitations": _string_array_schema(
                    max_items=2,
                    item_max_length=240,
                ),
                "no_reaction_proof": {"type": "boolean", "enum": [True]},
                "no_source_authority": {"type": "boolean", "enum": [True]},
                "no_solved_claim": {"type": "boolean", "enum": [True]},
        }
        if task.task_type == "paper_matched_key_event_critic":
            properties["checkpoint_match"] = {"type": "boolean"}
        else:
            properties["coupled_blocker_groups"] = {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": _short_text_schema(160),
                    "minItems": 2,
                },
            }
        return _strict_object_schema(properties)

    assessment_properties = {
            "step_id": {"type": "string"},
            "mechanistic_analysis": {"type": "string"},
            "atom_provenance": {"type": "string"},
            "functional_group_compatibility": {"type": "string"},
            "chemoselectivity": {"type": "string"},
            "stereochemistry": {"type": "string"},
            "sequence_ordering": {"type": "string"},
            "competing_pathways": _string_array_schema(),
            "verdict": {
                "type": "string",
                "enum": ["pass", "uncertain", "reject"],
            },
            "reasons": _string_array_schema(),
        }
    assessment_properties["enzyme_assessment"] = {"type": "string"}
    step_assessment = _strict_object_schema(assessment_properties)
    return _strict_object_schema(
        {
            "schema_version": {
                "type": "string",
                "enum": ["chemical_strategy_critique.v1"],
            },
            "case_id": {"type": "string", "enum": [task.case_id]},
            "strategy_id": {"type": "string"},
            "strategy_digest": {"type": "string"},
            "route_family_id": {"type": "string"},
            "overall_assessment": {
                "type": "string",
                "enum": ["viable", "uncertain", "reject"],
            },
            "strategy_adherence": {"type": "boolean"},
            "step_assessments": {
                "type": "array",
                "items": step_assessment,
                "minItems": 1,
                "maxItems": 32,
            },
            "route_level_risks": _string_array_schema(),
            "repair_actions": _string_array_schema(),
            "experimental_variables": _string_array_schema(),
            "limitations": _string_array_schema(),
            "no_reaction_proof": {"type": "boolean", "enum": [True]},
            "no_source_authority": {"type": "boolean", "enum": [True]},
            "no_solved_claim": {"type": "boolean", "enum": [True]},
        }
    )


def _global_campaign_plan_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    context_refs = [str(value) for value in task.input_refs if str(value).strip()]
    route_family = _strict_object_schema({
        "route_family_id": {"type": "string"},
        "title": {"type": "string"},
        "strategy": {"type": "string"},
        "target_smiles": {"type": "string"},
        "advantages": _string_array_schema(),
        "risks": _string_array_schema(),
        "diversity_basis": {"type": "string"},
    })
    condition_prediction = _strict_object_schema({
        "reagents": _string_array_schema(),
        "catalyst": {"type": "string"},
        "base": {"type": "string"},
        "solvent": {"type": "string"},
        "temperature_c": {"type": "number"},
        "time": {"type": "string"},
        "atmosphere": {"type": "string"},
        "authority_scope": {
            "type": "string",
            "enum": ["model_predicted_condition"],
        },
        "not_reaction_proof": {"type": "boolean", "enum": [True]},
    })
    step = _strict_object_schema({
        "step_id": {"type": "string"},
        "product_smiles": {"type": "string"},
        "precursor_smiles": _string_array_schema(),
        "transformation_hypothesis": {"type": "string"},
        "strategic_role": {"type": "string"},
        "source_hints": _string_array_schema(),
        "required_validation": _string_array_schema(),
        "hypothesis_only": {"type": "boolean", "enum": [True]},
        "condition_predictions": {
            "type": "array",
            "items": condition_prediction,
            "minItems": 1,
            "maxItems": 2,
        },
    })
    skeleton = _strict_object_schema({
        "skeleton_id": {"type": "string"},
        "route_family_id": {"type": "string"},
        "summary": {"type": "string"},
        "steps": {"type": "array", "items": step, "maxItems": 12},
    })
    disconnection = _strict_object_schema({
        "disconnection_id": {"type": "string"},
        "target_smiles": {"type": "string"},
        "bond_or_retron": {"type": "string"},
        "rationale": {"type": "string"},
        "route_family_ids": _string_array_schema(),
        "required_validation": _string_array_schema(),
    })
    shared = _strict_object_schema({
        "intermediate_id": {"type": "string"},
        "smiles": {"type": "string"},
        "route_family_ids": _string_array_schema(),
        "strategic_role": {"type": "string"},
        "risk": {"type": "string"},
    })
    unknown = _strict_object_schema({
        "unknown_id": {"type": "string"},
        "description": {"type": "string"},
        "affected_proposal_ids": _string_array_schema(),
        "resolution_task": {"type": "string"},
        "priority": {"type": "number"},
    })
    source_task = _strict_object_schema({
        "source_task_id": {"type": "string"},
        "query": {"type": "string"},
        "source_types": _string_array_schema(),
        "source_refs": _string_array_schema(),
        "target_claims": _string_array_schema(),
        "affected_proposal_ids": _string_array_schema(),
        "priority": {"type": "number"},
    })
    fallback = _strict_object_schema({
        "fallback_id": {"type": "string"},
        "trigger": {"type": "string"},
        "action": {"type": "string"},
        "route_family_ids": _string_array_schema(),
    })
    frontier = _strict_object_schema({
        "priority_id": {"type": "string"},
        "proposal_id": {"type": "string"},
        "target_smiles": {"type": "string"},
        "provider_preferences": _string_array_schema(),
        "retron_hints": _string_array_schema(),
        "route_family_ids": _string_array_schema(),
        "priority": {"type": "number"},
        "rationale": {"type": "string"},
        "expected_portfolio_gain": {"type": "string"},
    })
    pivot = _strict_object_schema({
        "pivot_id": {"type": "string"},
        "condition": {"type": "string"},
        "action": {"type": "string"},
    })
    stop = _strict_object_schema({
        "stop_id": {"type": "string"},
        "condition": {"type": "string"},
        "disposition": {"type": "string", "enum": ["continue", "unresolved", "budget_exhausted", "request_acceptance_evaluation"]},
    })
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["global_campaign_plan.v1"]},
        "plan_id": {"type": "string"},
        "run_id": {"type": "string", "enum": [task.case_id]},
        "mode": {"type": "string", "enum": ["initial_architecture", "event_replan", "final_portfolio_synthesis"]},
        "context_sha256": {
            "type": "string",
            **({"enum": [context_refs[0]]} if context_refs else {}),
        },
        "graph_revision": {"type": "integer"},
        "route_families": {"type": "array", "items": route_family, "maxItems": 6},
        "multi_step_skeletons": {"type": "array", "items": skeleton, "maxItems": 8},
        "strategic_disconnections": {
            "type": "array", "items": disconnection, "maxItems": 8
        },
        "shared_intermediates": {"type": "array", "items": shared, "maxItems": 8},
        "critical_unknowns": {"type": "array", "items": unknown, "maxItems": 8},
        "source_plan": {"type": "array", "items": source_task, "maxItems": 8},
        "fallback_strategies": {"type": "array", "items": fallback, "maxItems": 8},
        "frontier_priorities": {"type": "array", "items": frontier, "maxItems": 8},
        "pivot_conditions": {"type": "array", "items": pivot, "maxItems": 8},
        "stop_conditions": {"type": "array", "items": stop, "maxItems": 8},
        "portfolio_rationale": {"type": "string"},
        "limitations": _string_array_schema(),
    })
def _agent_action_batch_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    hint_schema = _strict_object_schema(
        {
            "schema_version": {"type": "string", "enum": [PLANNER_SOURCE_HINT_SCHEMA]},
            "hint_id": {"type": "string"},
            "source_ref": {"type": "string"},
            "title": {"type": "string"},
            "doi": {"type": "string"},
            "pii": {"type": "string"},
            "url": {"type": "string"},
            "local_pdf": {"type": "string"},
            "local_ref": {"type": "string"},
            "source_type": {"type": "string"},
            "relevance_rationale": {"type": "string"},
            "expected_scheme_or_compound_labels": _string_array_schema(),
            "extraction_task_recommendations": _string_array_schema(),
            "evidence_class": {"type": "string", "enum": ["planner_source_hint"]},
            "allowed_use": {"type": "string", "enum": ["source_acquisition_hint_only"]},
            "no_solved_claim": {"type": "boolean"},
        },
        required=[
            "schema_version",
            "hint_id",
            "source_ref",
            "title",
            "doi",
            "pii",
            "url",
            "local_pdf",
            "local_ref",
            "source_type",
            "relevance_rationale",
            "expected_scheme_or_compound_labels",
            "extraction_task_recommendations",
            "evidence_class",
            "allowed_use",
            "no_solved_claim",
        ],
    )
    action_schema = _strict_object_schema(
        {
            "schema_version": {"type": "string", "enum": ["agent_action.v1"]},
            "action_id": {"type": "string"},
            "action_type": {"type": "string", "enum": sorted(WORKER_AGENT_ACTION_TYPES)},
            "rationale": {"type": "string"},
            "expected_artifact": {"type": "string"},
            "success_condition": {"type": "string"},
            "payload": _agent_action_skeleton_payload_json_schema(),
        },
        required=[
            "schema_version",
            "action_id",
            "action_type",
            "rationale",
            "expected_artifact",
            "success_condition",
            "payload",
        ],
    )
    return _strict_object_schema(
        {
            "schema_version": {"type": "string", "enum": ["agent_action_batch.v1"]},
            "case_id": {"type": "string", "enum": [task.case_id]},
            "round_index": {"type": "integer"},
            "mode": {"type": "string"},
            "actions": {
                "type": "array",
                "items": action_schema,
            },
            "planner_source_hints": {
                "type": "array",
                "items": hint_schema,
            },
            "semantics": _strict_object_schema(
                {
                    "planner_can_emit_solved": {"type": "boolean"},
                    "raw_reaction_output_allowed": {"type": "boolean"},
                    "deterministic_validator_required": {"type": "boolean"},
                },
                required=[
                    "planner_can_emit_solved",
                    "raw_reaction_output_allowed",
                    "deterministic_validator_required",
                ],
            ),
        },
        required=["schema_version", "case_id", "round_index", "mode", "actions", "planner_source_hints", "semantics"],
    )


def _agent_action_skeleton_payload_json_schema() -> dict[str, Any]:
    """Closed generic payload schema for Codex structured-output compatibility.

    The action planner should emit compact skeletons. Local normalizers then
    expand those skeletons into tool-specific payloads under deterministic
    validation, so this schema intentionally exposes only common hint fields.
    """
    properties = {
        "schema_version": {"type": "string"},
        "no_solved_claim": {"type": "boolean"},
        "reason": {"type": "string"},
        "search_intent": {"type": "string"},
        "queries": _string_array_schema(),
        "search_queries": _string_array_schema(),
        "source_ref": {"type": "string"},
        "source_title": {"type": "string"},
        "pdf_path": {"type": "string"},
        "page_numbers": {"type": "array", "items": {"type": "integer"}},
        "expected_labels": _string_array_schema(),
        "route_sequence_hint": {"type": "string"},
        "template_ids": _string_array_schema(),
        "hypothesis_ids": _string_array_schema(),
        "selected_analogy_hypothesis_ids": _string_array_schema(),
        "route_objectives": _string_array_schema(),
        "endpoint_candidates": _string_array_schema(),
        "planner_source_hints": _string_array_schema(),
        "initial_probe": {"type": "boolean"},
        "search_mode": {"type": "string"},
        "max_steps": {"type": "integer"},
        "chem_enzy_iterations": {"type": "integer"},
        "chem_enzy_expansion_topk": {"type": "integer"},
        "timeout_s": {"type": "number"},
        "max_candidates": {"type": "integer"},
        "target_name": {"type": "string"},
        "target_smiles": {"type": "string"},
        "precursor_smiles": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }


def _evidence_card_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["evidence_card.v1"]},
        "evidence_id": {"type": "string"},
        "case_id": {"type": "string", "enum": [task.case_id]},
        "source_type": {"type": "string"},
        "source_title": {"type": "string"},
        "target_relation": {
            "type": "string",
            "enum": ["exact_target_or_intermediate", "family_precedent", "reaction_precedent", "analogy_only"],
        },
        "claim_type": {"type": "string"},
        "route_role": {
            "type": "string",
            "enum": ["scaffold_family", "strategic_disconnection", "route_anchor", "condition_hint", "negative_guidance", "unknown"],
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "medium_high", "high"]},
        "url": {"type": "string"},
        "doi": {"type": "string"},
        "local_ref": {"type": "string"},
        "source_record_id": {"type": "string"},
        "family_id": {"type": "string"},
        "route_role_detail": {"type": "string"},
        "limitations": _string_array_schema(),
        "source_metadata": _source_metadata_json_schema(),
        "validation_status": {"type": "string", "enum": ["draft", "validated"]},
    })


def _literature_scout_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["literature_scout_report.v1"]},
        "accepted": {"type": "boolean"},
        "case_id": {"type": "string", "enum": [task.case_id]},
        "source_candidates": {
            "type": "array",
            "items": _strict_object_schema({
                "schema_version": {"type": "string", "enum": ["literature_source_candidate.v1"]},
                "candidate_id": {"type": "string"},
                "source_ref": {"type": "string"},
                "title": {"type": "string"},
                "doi": {"type": "string"},
                "url": {"type": "string"},
                "local_pdf": {"type": "string"},
                "source_type": {"type": "string"},
                "relevance_rationale": {"type": "string"},
                "expected_scheme_or_compound_labels": _string_array_schema(),
                "extraction_task_recommendations": _string_array_schema(),
                "access_status": {"type": "string"},
                "no_solved_claim": {"type": "boolean"},
            }),
        },
        "source_refs": _string_array_schema(),
        "search_queries": _string_array_schema(),
        "reasons": _string_array_schema(),
        "limitations": _string_array_schema(),
        "no_solved_claim": {"type": "boolean"},
    })


def _analogical_template_report_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["analogical_reaction_template_report.v1"]},
        "accepted": {"type": "boolean"},
        "case_id": {"type": "string", "enum": [task.case_id]},
        "templates": {
            "type": "array",
            "items": _strict_object_schema({
                "schema_version": {"type": "string", "enum": ["analogical_reaction_template.v1"]},
                "template_id": {"type": "string"},
                "relation_type": {"type": "string", "enum": ["analog", "family_precedent", "mechanistic_hint"]},
                "reaction_class": {"type": "string"},
                "mechanistic_class": {"type": "string"},
                "reaction_center": _strict_object_schema({
                    "product_retron_type": {"type": "string"},
                    "template_radius": {"type": "string"},
                    "local_environment": {"type": "string"},
                    "not_raw_reaction_injection": {"type": "boolean"},
                }),
                "template_radius": {"type": "string"},
                "scope_gap": {"type": "string"},
                "risk_flags": _string_array_schema(),
                "required_verification": _string_array_schema(),
                "confidence": {"type": "string", "enum": ["low", "medium", "medium_high", "high"]},
                "source_refs": _string_array_schema(),
                "no_solved_claim": {"type": "boolean"},
                "not_raw_reaction_injection": {"type": "boolean"},
            }),
        },
        "source_refs": _string_array_schema(),
        "reasons": _string_array_schema(),
        "no_solved_claim": {"type": "boolean"},
    })


def _source_metadata_json_schema() -> dict[str, Any]:
    return _strict_object_schema({
        "title": {"type": "string"},
        "journal": {"type": "string"},
        "publication_year": {"type": "string"},
        "authors": _string_array_schema(),
        "doi": {"type": "string"},
        "url": {"type": "string"},
        "pmid": {"type": "string"},
        "source_name": {"type": "string"},
        "search_query": {"type": "string"},
        "accessed_date": {"type": "string"},
        "notes": {"type": "string"},
    })


def _literature_route_segment_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["literature_route_segment_card.v1"]},
        "segment_id": {"type": "string"},
        "case_id": {"type": "string", "enum": [task.case_id]},
        "target_smiles": {"type": "string"},
        "steps": {"type": "array", "items": _segment_step_json_schema()},
        "evidence_refs": _string_array_schema(),
        "source_title": {"type": "string"},
        "source_type": {"type": "string"},
        "trigger_reasons": _string_array_schema(),
        "validation_status": {"type": "string", "enum": ["draft", "draft_only", "rejected"]},
    })


def _segment_step_json_schema() -> dict[str, Any]:
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["segment_step_candidate.v1"]},
        "step_id": {"type": "string"},
        "product_smiles": {"type": "string"},
        "reactant_smiles": _string_array_schema(),
        "evidence_refs": _string_array_schema(),
        "source_ref": {"type": "string"},
        "relation_type": {"type": "string", "enum": ["exact", "analog", "mismatch"]},
        "applicability": _applicability_json_schema(),
        "condition_candidate": _condition_candidate_json_schema(),
        "scope_gap": {"type": "string"},
    })


def _applicability_json_schema() -> dict[str, Any]:
    return _strict_object_schema({
        "status": {"type": "string", "enum": ["passed", "exact", "rejected", "analog", "unknown"]},
        "product_reconstruction_passed": {"type": "boolean"},
        "reconstructed_product_smiles": {"type": "string"},
        "notes": {"type": "string"},
        "limitations": _string_array_schema(),
    })


def _condition_candidate_json_schema() -> dict[str, Any]:
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["condition_candidate.v1"]},
        "step_id": {"type": "string"},
        "source_type": {"type": "string", "enum": ["exact", "analog", "template", "model-only", "unknown"]},
        "condition_status": {
            "type": "string",
            "enum": ["evidence_backed", "analog_scope_gap", "feasibility_hint", "gap"],
        },
        "reagent": {"type": "string"},
        "catalyst": {"type": "string"},
        "enzyme": {"type": "string"},
        "solvent": {"type": "string"},
        "temperature": {"type": "string"},
        "ph": {"type": "string"},
        "buffer": {"type": "string"},
        "atmosphere": {"type": "string"},
        "evidence_refs": _string_array_schema(),
        "scope_gap": {"type": "string"},
        "risk_flags": _string_array_schema(),
        "confidence": {"type": "string"},
    })


def _procedure_repair_draft_json_schema() -> dict[str, Any]:
    conditions = _strict_object_schema({
        "reagents": _string_array_schema(),
        "catalyst": {"type": "string"},
        "base": _string_array_schema(),
        "solvent": _string_array_schema(),
        "temperature": {"type": "string"},
        "time": {"type": "string"},
        "atmosphere": {"type": "string"},
        "addition_order": {"type": "string"},
        "workup": {"type": "string"},
        "purification": {"type": "string"},
        "yield_percent": {"type": "number"},
    })
    return _strict_object_schema({
        "schema_version": {"type": "string", "enum": ["procedure_repair_draft.v1"]},
        "step_id": {"type": "string"},
        "reaction_class": {"type": "string"},
        "diagnosis": _string_array_schema(),
        "conditions": conditions,
        "missing_information": _string_array_schema(),
        "risk_flags": _string_array_schema(),
        "repair_actions": _string_array_schema(),
        "authority_scope": {
            "type": "string",
            "enum": ["model_predicted_condition"],
        },
        "no_exact_source_authority": {"type": "boolean", "enum": [True]},
        "no_experimental_validation_claim": {"type": "boolean", "enum": [True]},
    })


def _typed_artifact_schema_version(artifact_type: str) -> str:
    return {
        "AgentActionBatch": "agent_action_batch_artifact.v1",
        "ResearchReport": "research_report.v1",
        "RetrosynthesisProposalReport": "retrosynthesis_proposal_report_artifact.v1",
        "StrategyCardReport": "strategy_card_report_artifact.v1",
        "ChemicalStrategyCritique": "chemical_strategy_critique_artifact.v1",
        "GlobalCampaignPlan": "global_campaign_plan_artifact.v1",
        "EvidenceCard": "evidence_card_artifact.v1",
        "LiteratureScoutReport": "literature_scout_report_artifact.v1",
        "StrategicDisconnectionCard": "strategic_disconnection_card.v1",
        "LiteratureRouteSegmentCard": "literature_route_segment_card.v1",
        "SegmentStepCandidate": "segment_step_candidate.v1",
        "FailureDiagnosis": "failure_diagnosis.v1",
        "StrategicOperator": "strategic_operator_artifact.v1",
        "ConditionCandidate": "condition_candidate.v1",
        "AuditReport": "route_audit_report.v1",
        "ProcedureRepairDraft": "procedure_repair_draft_artifact.v1",
        "EvolutionCandidate": "evolution_candidate_artifact.v1",
    }.get(artifact_type, f"{artifact_type.lower()}.draft.v1")


def _use_codex_cli(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    raw = os.environ.get("AUTOPLANNER_CODEX_WORKER_BACKEND") or os.environ.get("AUTOPLANNER_CODEX_CLI")
    if raw is None:
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "mock", "disabled", "none"}


def _use_api_json(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    raw = os.environ.get("AUTOPLANNER_CODEX_WORKER_BACKEND")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"api", "api_json", "json", "openai_compatible"}


def _codex_executable() -> str:
    return os.environ.get("AUTOPLANNER_CODEX_CLI_BIN") or "codex"


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_worker_stdout(stdout: str) -> Any:
    try:
        return json.loads(stdout or "")
    except json.JSONDecodeError:
        return None


_NON_ASCII_RUN = re.compile(r"[^\x00-\x7f]+")


def _normalize_reversible_utf8_mojibake(value: str) -> tuple[str, int]:
    """Repair only reversible UTF-8 text decoded as Windows GBK.

    Provider/CLI boundaries have emitted sequences such as ``鈥揔`` for an
    en dash and ``掳`` for a degree sign.  Re-encoding a suspicious non-ASCII
    run as GBK and decoding those exact bytes as UTF-8 recovers the original
    punctuation.  Normal Chinese is left untouched: a repair is accepted only
    when the round trip removes every CJK ideograph and yields punctuation or
    symbols without a replacement character.
    """

    text = str(value or "")
    repair_count = 0

    def repair(match: re.Match[str]) -> str:
        nonlocal repair_count
        raw = match.group(0)
        try:
            candidate = raw.encode("gbk").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return raw
        if candidate == raw or "\ufffd" in candidate:
            return raw
        raw_cjk = sum(_is_cjk_ideograph(char) for char in raw)
        candidate_cjk = sum(_is_cjk_ideograph(char) for char in candidate)
        has_punctuation_or_symbol = any(
            unicodedata.category(char).startswith(("P", "S"))
            for char in candidate
        )
        if raw_cjk and candidate_cjk == 0 and has_punctuation_or_symbol:
            repair_count += 1
            return candidate
        return raw

    return _NON_ASCII_RUN.sub(repair, text), repair_count


def _is_cjk_ideograph(value: str) -> bool:
    codepoint = ord(value)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _parse_codex_jsonl_events(stdout: str) -> dict[str, Any]:
    """Normalize Codex ``exec --json`` events for audit and budget checks."""
    events: list[dict[str, Any]] = []
    invalid_lines = 0
    for line in str(stdout or "").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if isinstance(event, dict):
            events.append(event)

    session_id = ""
    usage: dict[str, Any] = {}
    calls_by_id: dict[str, dict[str, Any]] = {}
    child_agents_by_id: dict[str, dict[str, Any]] = {}
    orphan_wait_state_count = 0
    turn_completed = False
    turn_failed = False
    last_terminal_event_type = ""
    last_terminal_event_index = -1
    errors: list[str] = []
    error_events: list[tuple[int, str]] = []
    for index, event in enumerate(events):
        event_type = str(event.get("type") or event.get("event") or "")
        if event_type == "turn.completed":
            turn_completed = True
            last_terminal_event_type = event_type
            last_terminal_event_index = index
        elif event_type == "turn.failed":
            turn_failed = True
            last_terminal_event_type = event_type
            last_terminal_event_index = index
        if event_type in {"error", "turn.failed"}:
            message = _find_first_key(event, {"message", "detail"})
            if message and message not in errors:
                errors.append(message)
                error_events.append((index, message))
        if not session_id:
            session_id = _find_first_key(event, {"thread_id", "session_id", "conversation_id"})
        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
            usage = _merge_usage(usage, event_usage)
        item = event.get("item") if isinstance(event.get("item"), dict) else event
        item_type = str(item.get("type") or event_type).lower()
        tool_name = _codex_event_tool_name(item, item_type)
        if not tool_name:
            continue
        call_id = str(item.get("id") or item.get("call_id") or item.get("tool_call_id") or f"event:{index}")
        receiver_thread_ids = [
            str(value)
            for value in item.get("receiver_thread_ids") or []
            if str(value or "").strip()
        ]
        agent_states = {
            str(agent_id): dict(state)
            for agent_id, state in (item.get("agents_states") or {}).items()
            if str(agent_id or "").strip() and isinstance(state, dict)
        }
        arguments = item.get("arguments") if isinstance(item.get("arguments"), (dict, list, str)) else {}
        record = {
            "call_id": call_id,
            "tool": tool_name,
            "event_type": event_type,
            "status": str(item.get("status") or ""),
            "exit_code": item.get("exit_code"),
            "aggregated_output": str(item.get("aggregated_output") or ""),
            "arguments": arguments,
            "prompt": str(item.get("prompt") or ""),
            "sender_thread_id": str(item.get("sender_thread_id") or ""),
            "receiver_thread_ids": receiver_thread_ids,
            "agents_states": agent_states,
        }
        calls_by_id[call_id] = record
        if tool_name == "spawn_agent":
            sender_thread_id = str(record.get("sender_thread_id") or "")
            # A coordinator run may contain nested child activity in the same
            # JSONL stream. Only the root thread's spawns satisfy the direct
            # specialist contract. Older streams without thread identifiers
            # remain parseable, but they still need an observed spawn event.
            direct_spawn = bool(
                not session_id
                or not sender_thread_id
                or sender_thread_id == session_id
            )
            if not direct_spawn:
                continue
            for receiver_thread_id in receiver_thread_ids:
                state = dict(agent_states.get(receiver_thread_id) or {})
                child_agents_by_id[receiver_thread_id] = {
                    "agent_id": receiver_thread_id,
                    "spawn_call_id": call_id,
                    "sender_thread_id": record["sender_thread_id"],
                    "status": str(state.get("status") or "spawned"),
                    "message": state.get("message"),
                    "prompt": record["prompt"],
                    "arguments": arguments,
                }
        if tool_name in {"wait", "wait_agent"}:
            for receiver_thread_id, state in agent_states.items():
                child = child_agents_by_id.get(receiver_thread_id)
                if child is None:
                    # A wait/status snapshot is not proof that this coordinator
                    # directly spawned the agent.
                    orphan_wait_state_count += 1
                    continue
                child["status"] = str(state.get("status") or child.get("status") or "unknown")
                child["message"] = state.get("message")
                child["wait_call_id"] = call_id

    child_agents = list(child_agents_by_id.values())
    completed_children = [
        row
        for row in child_agents
        if str(row.get("status") or "").strip().lower() in {"completed", "succeeded", "success", "accepted"}
    ]
    terminal_completed = last_terminal_event_type == "turn.completed"
    fatal_error = ""
    if terminal_completed:
        post_completion_errors = [
            message
            for index, message in error_events
            if index > last_terminal_event_index
        ]
        if post_completion_errors:
            fatal_error = post_completion_errors[-1]
    elif errors:
        fatal_error = errors[-1]
    recovered_error_count = sum(
        1
        for index, _message in error_events
        if terminal_completed and index < last_terminal_event_index
    )

    return {
        "session_id": session_id,
        "usage": usage,
        "tool_calls": list(calls_by_id.values()),
        "child_agents": child_agents,
        "summary": {
            "schema_version": "codex_event_summary.v1",
            "event_count": len(events),
            "invalid_line_count": invalid_lines,
            "turn_completed": turn_completed,
            "turn_failed": turn_failed,
            "last_terminal_event_type": last_terminal_event_type,
            "last_event_type": str(events[-1].get("type") or "") if events else "",
            "tool_call_count": len(calls_by_id),
            "child_agent_spawn_count": len(child_agents),
            "child_agent_completed_count": len(completed_children),
            "orphan_wait_state_count": orphan_wait_state_count,
            "errors": errors[-16:],
            "fatal_error": fatal_error,
            "recovered_error_count": recovered_error_count,
        },
    }


def _assign_child_roles(children: Any, *, roles: list[str]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in children or [] if isinstance(row, dict)]
    remaining = [str(role) for role in roles]
    # Prefer evidence in the actual spawn prompt/arguments. This prevents four
    # same-role children from being relabelled as four distinct specialists.
    for row in rows:
        haystack = " ".join(
            [
                str(row.get("prompt") or ""),
                json.dumps(row.get("arguments") or {}, ensure_ascii=False, sort_keys=True),
            ]
        )
        matches = [role for role in remaining if _child_prompt_mentions_role(haystack, role)]
        if len(matches) == 1:
            row["role"] = matches[0]
            row["role_binding_method"] = "explicit_spawn_contract"
            remaining.remove(matches[0])
    # Some CLI versions omit the prompt from collab events. Preserve ordered
    # compatibility only for those genuinely unlabelled observations.
    for row in rows:
        if row.get("role") or not remaining:
            continue
        if str(row.get("prompt") or "").strip() or row.get("arguments"):
            continue
        row["role"] = remaining.pop(0)
        row["role_binding_method"] = "legacy_event_order"
    return rows


def _child_prompt_mentions_role(text: str, role: str) -> bool:
    raw = str(text or "").lower().replace(" ", "")
    normalized_role = "_".join(
        part
        for part in str(role or "").lower().replace("-", "_").split("_")
        if part
    )
    markers = (
        f"autoplanner_child_role={normalized_role}",
        f'"autoplanner_child_role":"{normalized_role}"',
    )
    return bool(normalized_role and any(marker in raw for marker in markers))


def _write_codex_event_log(workdir: Path, *, task: WorkerTask, stdout: str) -> Path | None:
    if not str(stdout or "").strip():
        return None
    event_dir = Path(workdir).resolve() / "codex_worker_events"
    event_dir.mkdir(parents=True, exist_ok=True)
    task_digest = hashlib.sha256(str(task.task_id).encode("utf-8")).hexdigest()[:12]
    content_digest = hashlib.sha256(str(stdout or "").encode("utf-8")).hexdigest()[:16]
    path = event_dir / f"{task_digest}-{content_digest}.jsonl"
    path.write_text(str(stdout or ""), encoding="utf-8")
    return path


def _codex_event_tool_name(item: dict[str, Any], item_type: str) -> str:
    raw = str(item.get("tool") or item.get("name") or item.get("tool_name") or "").strip()
    lowered = raw.lower()
    if lowered:
        if "spawn_agent" in lowered:
            return "spawn_agent"
        if "wait_agent" in lowered:
            return "wait_agent"
        if "send_message" in lowered or "followup_task" in lowered:
            return "send_message"
        if "web" in lowered and "search" in lowered:
            return "web_search"
        if "browser" in lowered:
            return "browser"
        return raw
    if item_type in {"web_search", "web_search_call"} or "web_search" in item_type:
        return "web_search"
    if item_type in {"command_execution", "shell_command", "shell"} or "command_execution" in item_type:
        return "shell"
    if "mcp_tool" in item_type or item_type in {"function_call", "tool_call"}:
        return "unknown_tool"
    return ""


def _find_first_key(value: Any, keys: set[str]) -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys and str(item or "").strip():
                return str(item)
        for item in value.values():
            found = _find_first_key(item, keys)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_first_key(item, keys)
            if found:
                return found
    return ""


def _merge_usage(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(existing)
    for key, value in update.items():
        if isinstance(value, dict):
            out[key] = _merge_usage(dict(out.get(key) or {}), value)
        elif isinstance(value, (int, float)):
            out[key] = max(float(out.get(key) or 0), value)
            if isinstance(value, int):
                out[key] = int(out[key])
        else:
            out[key] = value
    return out


def _truncate(text: str, max_bytes: int) -> str:
    raw = str(text or "")
    encoded = raw.encode("utf-8")
    if len(encoded) <= int(max_bytes):
        return raw
    return encoded[: int(max_bytes)].decode("utf-8", errors="ignore")


def _contains_forbidden_production_write(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in FORBIDDEN_PRODUCTION_KEYS:
                return True
            if key_l in {"kb_layer", "target_layer", "write_layer"} and str(item).lower() == "production":
                return True
            if _contains_forbidden_production_write(item):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_production_write(item) for item in value)
    return False


def _worker_runtime_reasons(task: WorkerTask, process: WorkerProcessResult) -> list[str]:
    reasons: list[str] = []
    event_summary = dict((process.metadata or {}).get("event_summary") or {})
    recovered_codex_completion = bool(
        process.backend == "codex_cli"
        and event_summary.get("turn_completed") is True
        and event_summary.get("last_terminal_event_type") == "turn.completed"
        and not str(event_summary.get("fatal_error") or "")
    )
    if int(process.exit_code or 0) != 0 and not recovered_codex_completion:
        reasons.append("worker_exit_code_nonzero")
    tool_calls = list(process.tool_calls or [])
    # A sandbox launch rejection is an observed model attempt, but it is not
    # an executed tool call.  Keep the raw record for audit while preventing
    # a blocked, side-effect-free attempt from invalidating an otherwise
    # complete structured artifact or consuming the runtime tool budget.
    # Calls that actually started still remain subject to both the budget and
    # the allow-list below.
    executed_tool_calls = [
        call
        for call in tool_calls
        if not (recovered_codex_completion and _tool_failed_before_execution(call))
    ]
    if (
        task.budget.max_tool_calls is not None
        and len(executed_tool_calls) > int(task.budget.max_tool_calls)
    ):
        reasons.append("tool_call_budget_exceeded")
    allowed = {_canonical_runtime_tool_name(tool) for tool in task.allowed_tools}
    if allowed:
        for call in executed_tool_calls:
            observed = _canonical_runtime_tool_name(call.get("tool") or call.get("name") or "")
            if observed not in allowed:
                reasons.append("tool_not_allowed")
                break
    if task.agent_mode == "coordinator":
        spawned = list((process.metadata or {}).get("child_agents") or [])
        if len(spawned) < len(task.child_roles):
            reasons.append("required_child_agents_not_spawned")
        completed = [
            row
            for row in spawned
            if str((row or {}).get("status") or "").strip().lower()
            in {"completed", "succeeded", "success", "accepted"}
        ]
        if len(completed) < len(task.child_roles):
            reasons.append("required_child_agents_not_completed")
        observed_roles = [str((row or {}).get("role") or "") for row in spawned]
        if sorted(observed_roles) != sorted(str(role) for role in task.child_roles):
            reasons.append("required_child_role_coverage_mismatch")
        if any(
            str((row or {}).get("role_binding_method") or "") != "explicit_spawn_contract"
            for row in spawned
        ):
            reasons.append("required_child_roles_not_prompt_bound")
    return reasons


def _worker_provider_failure_reason_from_process(
    process: WorkerProcessResult,
) -> str:
    """Classify a missing provider turn separately from schema rejection."""

    event_summary = dict((process.metadata or {}).get("event_summary") or {})
    if (
        event_summary.get("turn_completed") is True
        and event_summary.get("last_terminal_event_type") == "turn.completed"
        and not str(event_summary.get("fatal_error") or "")
    ):
        return ""
    text = "\n".join(
        (
            str(process.stderr or ""),
            str(process.stdout or ""),
            str(event_summary.get("fatal_error") or ""),
        )
    )
    return _provider_failure_reason_from_text(text)


def _provider_failure_reason_from_text(value: str) -> str:
    """Classify known provider outages from one unfinished worker turn."""

    text = str(value or "").casefold()
    if any(
        marker in text
        for marker in (
            "refresh_token_reused",
            "token_expired",
            "failed to refresh token",
            "authentication token is expired",
            "401 unauthorized",
            "403 forbidden",
        )
    ):
        return "provider_auth_unavailable"
    if any(
        marker in text
        for marker in (
            "429 too many requests",
            "rate limit",
            "rate_limit",
        )
    ):
        return "provider_rate_limited"
    if any(
        marker in text
        for marker in (
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
            "service unavailable",
            "selected model is at capacity",
            "model is at capacity",
            "upstream connect error",
        )
    ):
        return "provider_service_unavailable"
    if any(
        marker in text
        for marker in (
            "stream disconnected",
            "error sending request",
            "connection reset",
            "connection refused",
            "dns error",
            "timed out while waiting for provider",
        )
    ):
        return "provider_transport_unavailable"
    return ""


def worker_provider_failure_reason(record: WorkerRunRecord) -> str:
    """Return the durable runtime classification for one worker record."""

    if str(record.status or "") != "provider_error":
        event_summary = dict((record.metadata or {}).get("event_summary") or {})
        if not (
            event_summary.get("turn_failed") is True
            or event_summary.get("last_terminal_event_type") == "turn.failed"
        ):
            return ""
        return _provider_failure_reason_from_text(
            "\n".join(
                (
                    str(record.stderr or ""),
                    str(record.stdout or ""),
                    str(event_summary.get("fatal_error") or ""),
                )
            )
        )
    return str(
        dict(record.metadata or {}).get("provider_failure_reason")
        or next(
            (
                value
                for value in dict(record.output_validation or {}).get("reasons") or []
                if str(value).startswith("provider_")
            ),
            "provider_unavailable",
        )
    )


def _tool_failed_before_execution(call: Mapping[str, Any]) -> bool:
    """Recognize a sandbox launch rejection, not a command that ran and failed."""

    try:
        exit_code = int(call.get("exit_code"))
    except (TypeError, ValueError):
        return False
    output = str(call.get("aggregated_output") or "").strip().casefold()
    return (
        str(call.get("status") or "").strip().casefold() == "failed"
        and exit_code == -1
        and output.startswith("execution error:")
    )


def _canonical_runtime_tool_name(value: Any) -> str:
    """Normalize Codex CLI tool names that changed across multi-agent releases."""
    name = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "wait_agent": "wait",
        "agent_wait": "wait",
        "send_input": "send_message",
        "agent_message": "send_message",
        "agent_spawn": "spawn_agent",
    }
    return aliases.get(name, name)


def _contains_solved_claim(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in {"solved", "mark_solved", "final_solved"} and bool(item):
                return True
            if key_l in {"route_status", "final_route_status", "status"} and str(item).lower() == "solved":
                return True
            if _contains_solved_claim(item):
                return True
    if isinstance(value, list):
        return any(_contains_solved_claim(item) for item in value)
    return False


def _contains_route_tree_mutation(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_ROUTE_TREE_KEYS:
                return True
            if _contains_route_tree_mutation(item):
                return True
    if isinstance(value, list):
        return any(_contains_route_tree_mutation(item) for item in value)
    return False


def _contains_raw_reaction_injection(value: Any) -> bool:
    return contains_raw_reaction_payload(value)
