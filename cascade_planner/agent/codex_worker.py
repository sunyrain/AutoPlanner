"""Controlled Codex research worker wrapper for P3.

The worker contract is deliberately conservative. A worker may produce typed
draft artifacts and a trace record, but it cannot mutate route trees, write
production KB entries, or mark a case solved.
"""
from __future__ import annotations

import json
import os
import hashlib
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from cascade_planner.agent.action_contracts import (
    ALLOWED_AGENT_ACTIONS as WORKER_AGENT_ACTION_TYPES,
    PLANNER_SOURCE_HINT_SCHEMA,
    contains_raw_reaction_payload,
)


WORKER_TASK_SCHEMA = "worker_task.v1"
WORKER_RUN_RECORD_SCHEMA = "worker_run_record.v1"
WORKER_OUTPUT_VALIDATION_SCHEMA = "worker_output_validation.v1"
DEFAULT_RETROSYNTHESIS_KEY_FILE = Path(__file__).resolve().parents[2] / "key.txt"

ALLOWED_WORKER_TASK_TYPES = {
    "target_research",
    "stuck_node_research",
    "strategic_disconnection_mining",
    "route_step_materialization",
    "route_chemistry_critique",
    "route_chemistry_edit",
    "route_audit_research",
    "condition_research",
    "evolution_candidate_research",
    "global_campaign_direction",
}
ALLOWED_WORKER_ARTIFACT_TYPES = {
    "AgentActionBatch",
    "ResearchReport",
    "RetrosynthesisProposalReport",
    "StrategyCardReport",
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
    max_tool_calls: int = 16
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
    schema_version: str = WORKER_TASK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
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
    stdout = _truncate(process.stdout, task.budget.max_output_bytes)
    artifact = _parse_worker_stdout(stdout)
    validation = validate_worker_output(task, artifact)
    runtime_reasons = _worker_runtime_reasons(task, process)
    if runtime_reasons:
        validation = dict(validation)
        validation["reasons"] = sorted(set([*validation.get("reasons", []), *runtime_reasons]))
        validation["accepted"] = False
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
        metadata=dict(process.metadata or {}),
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
    if task.budget.max_tool_calls < 0:
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
    return {
        "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "artifact_type": str(artifact.get("artifact_type") or ""),
    }


def worker_task_from_dict(data: dict[str, Any]) -> WorkerTask:
    budget = data.get("budget") or {}
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
            max_tool_calls=int(budget.get("max_tool_calls") or 16),
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
    executable = _codex_executable()
    if not _codex_executable_available(executable):
        raise FileNotFoundError(f"Codex CLI executable not found: {executable}")

    workdir = Path(task.allowed_workdir or ".").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    prompt = _codex_worker_prompt(task)
    worker_temp_root: Path | None = None
    try:
        worker_temp_root = workdir / ".autoplanner" / "codex-worker-tmp"
        worker_temp_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        worker_temp_root = None
    temp_kwargs = {
        "prefix": "autoplanner_codex_worker_",
        "ignore_cleanup_errors": True,
    }
    if worker_temp_root is not None:
        temp_kwargs["dir"] = str(worker_temp_root)
    with tempfile.TemporaryDirectory(**temp_kwargs) as tmp:
        tmp_path = Path(tmp)
        output_path = tmp_path / "last_message.json"
        schema_path = tmp_path / "worker_output_schema.json"
        schema_path.write_text(json.dumps(_worker_output_json_schema(task), indent=2), encoding="utf-8")
        env, metadata = _codex_cli_runtime_environment(tmp_path, workdir, task)
        command = _codex_cli_command(
            executable=_codex_executable_command(executable),
            workdir=workdir,
            output_path=output_path,
            schema_path=schema_path,
            runtime_metadata=metadata,
            search_enabled=_task_allows_cli_search(task),
            multi_agent_enabled=task.agent_mode == "coordinator",
        )
        try:
            returncode, stdout, stderr = _run_worker_command(
                command,
                cwd=workdir,
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
        final = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else stdout
        stderr = stderr or ""
        event_audit = _parse_codex_jsonl_events(stdout)
        event_log_path = _write_codex_event_log(workdir, task=task, stdout=stdout)
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
    for name in ("auth.json", "installation_id", "models_cache.json"):
        source = source_home / name
        if not source.is_file():
            continue
        shutil.copyfile(source, codex_home / name)
        copied_inputs.append(name)
    source_config = source_home / "config.toml"
    if source_config.is_file():
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
    }


def _ambient_codex_home() -> Path:
    configured = str(os.environ.get("CODEX_HOME") or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            return candidate
    return Path.home() / ".codex"


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
    config = _api_json_config(task)
    prompt = _codex_worker_prompt(task).replace('"source": "codex_cli"', '"source": "api_json"')
    schema = _worker_output_json_schema(task)
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
) -> list[str]:
    command = list(executable) if isinstance(executable, list) else [executable]
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
    command.extend([
        "exec",
    ])
    runtime_metadata = dict(runtime_metadata or {})
    if str(runtime_metadata.get("codex_home") or "") == "ephemeral":
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
    if profile:
        command.extend(["--profile", profile])
    if _env_flag("AUTOPLANNER_CODEX_WORKER_OSS", default=False):
        command.append("--oss")
        provider = os.environ.get("AUTOPLANNER_CODEX_WORKER_LOCAL_PROVIDER")
        if provider:
            command.extend(["--local-provider", provider])
    command.append("-")
    return command


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
    if int(task.budget.max_tool_calls or 0) <= 0:
        return False
    allowed_tools = {str(item).strip().lower() for item in task.allowed_tools or []}
    return bool(allowed_tools & {"web_search", "browser", "literature_search"})


def _codex_worker_prompt(task: WorkerTask) -> str:
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
        if task.task_type in {"strategic_disconnection_mining", "route_chemistry_critique"}
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
            "Anchor the selected strategy on mapped target atom pairs in key_bond_changes whenever "
            "the key forward construction changes target bonds. Compare at least three materially "
            "different high-level strategies before selecting one."
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
            in {"route_step_materialization", "route_chemistry_edit"}
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
                "For route_chemistry_edit, prefer route_patch with replace_step, insert_after, delete_step, or reorder operations; the host applies the patch to the frozen route and recompiles it from the target. "
                "route_json remains a complete replacement-route fallback and may reorder, insert, or delete steps and change conditions or functional-group operations while preserving the immutable strategy. "
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
            "Use overall_assessment=viable|uncertain|reject; reject only a concrete chemical contradiction, not merely missing literature. "
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
    properties = {
        "schema_version": {"type": "string"},
        "artifact_id": {"type": "string"},
        "artifact_type": {"type": "string", "enum": [task.required_artifact_type]},
        "case_id": {"type": "string", "enum": [task.case_id]},
        "source": {"type": "string"},
        "input_refs": _string_array_schema(),
        "evidence_refs": _string_array_schema(),
        "validation_status": {"type": "string", "enum": ["draft", "draft_only"]},
        "summary": {"type": "string"},
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


def _string_array_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


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
    strategy_card_properties = {
        "scaffold_motif": {"type": "string"},
        "key_forward_transformation": {"type": "string"},
        "key_bond_changes": _string_array_schema(),
        "functional_group_conflicts": _string_array_schema(),
        "protection_policy": {"type": "string"},
        "stereochemical_plan": {"type": "string"},
        "convergence_plan": {"type": "string"},
        "strategic_step_count": {"type": "integer", "minimum": 1, "maximum": 2},
        "skeleton_change_class": {"type": "string"},
        "expected_complexity_drop": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "orthogonality_basis": {"type": "string"},
        "strategy_signature": {"type": "string"},
        "execution_domain": _nullable_schema({
            "type": "string",
            "enum": ["chemical", "enzymatic", "whole_cell", "hybrid", "mechanistic"],
        }),
    }
    return _strict_object_schema(strategy_card_properties)


def _strategy_card_report_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
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
    reaction_operation = _strict_object_schema(
        {
            "op": {
                "type": "string",
                "enum": [
                    "break_bond",
                    "add_bond",
                    "change_bond_order",
                    "change_atom",
                    "set_explicit_h",
                    "add_group",
                    "remove_group",
                    "invert_stereocenter",
                    "clear_stereocenter",
                    "set_bond_stereo",
                ],
            },
            "map_a": _nullable_schema({"type": "integer"}),
            "map_b": _nullable_schema({"type": "integer"}),
            "map_idx": _nullable_schema({"type": "integer"}),
            "order": _nullable_schema({"type": "number"}),
            "delta": _nullable_schema({"type": "number"}),
            "atomic_num": _nullable_schema({"type": "integer"}),
            "element": _nullable_schema({"type": "string"}),
            "formal_charge": _nullable_schema({"type": "integer"}),
            "isotope": _nullable_schema({"type": "integer"}),
            "count": _nullable_schema({"type": "integer"}),
            "no_implicit": _nullable_schema({"type": "boolean"}),
            "fragment_smiles": _nullable_schema({"type": "string"}),
            "map_indices": _nullable_schema(
                {"type": "array", "items": {"type": "integer"}}
            ),
            "stereo": _nullable_schema({"type": "string"}),
            "stereo_atom_maps": _nullable_schema({
                "type": "array",
                "items": {"type": "integer"},
            }),
        },
    )
    candidate_properties = {
        "schema_version": {"type": "string", "enum": ["retrosynthesis_candidate.v1"]},
        "candidate_id": {"type": "string"},
        "product_smiles": {"type": "string"},
        "precursor_smiles": _string_array_schema(),
        "reaction_family": {"type": "string"},
        "product_retron_type": {"type": "string"},
        "transformation_rationale": {"type": "string"},
        "conditions": _string_array_schema(),
        "catalyst": {"type": "string"},
        "enzyme": {"type": "string"},
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
            "precursor_smiles": _string_array_schema(),
            "reaction_family": {"type": "string"},
            "product_retron_type": {"type": "string"},
            "transformation_rationale": {"type": "string"},
            "conditions": _string_array_schema(),
            "catalyst": {"type": "string"},
            "enzyme": {"type": "string"},
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
    if task.task_type not in {"route_step_materialization", "route_chemistry_edit"}:
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


def _chemical_strategy_critique_payload_json_schema(task: WorkerTask) -> dict[str, Any]:
    step_assessment = _strict_object_schema(
        {
            "step_id": {"type": "string"},
            "mechanistic_analysis": {"type": "string"},
            "atom_provenance": {"type": "string"},
            "functional_group_compatibility": {"type": "string"},
            "chemoselectivity": {"type": "string"},
            "stereochemistry": {"type": "string"},
            "sequence_ordering": {"type": "string"},
            "competing_pathways": _string_array_schema(),
            "enzyme_assessment": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["pass", "uncertain", "reject"],
            },
            "reasons": _string_array_schema(),
        }
    )
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
    if len(tool_calls) > int(task.budget.max_tool_calls):
        reasons.append("tool_call_budget_exceeded")
    allowed = {_canonical_runtime_tool_name(tool) for tool in task.allowed_tools}
    if allowed:
        for call in tool_calls:
            observed = _canonical_runtime_tool_name(call.get("tool") or call.get("name") or "")
            if observed not in allowed:
                if recovered_codex_completion and _tool_failed_before_execution(call):
                    continue
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
