"""Controlled Codex research worker wrapper for P3.

The worker contract is deliberately conservative. A worker may produce typed
draft artifacts and a trace record, but it cannot mutate route trees, write
production KB entries, or mark a case solved.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


WORKER_TASK_SCHEMA = "worker_task.v1"
WORKER_RUN_RECORD_SCHEMA = "worker_run_record.v1"
WORKER_OUTPUT_VALIDATION_SCHEMA = "worker_output_validation.v1"

ALLOWED_WORKER_TASK_TYPES = {
    "target_research",
    "stuck_node_research",
    "strategic_disconnection_mining",
    "route_audit_research",
    "condition_research",
    "evolution_candidate_research",
}
ALLOWED_WORKER_ARTIFACT_TYPES = {
    "ResearchReport",
    "EvidenceCard",
    "StrategicDisconnectionCard",
    "FailureDiagnosis",
    "StrategicOperator",
    "ConditionCandidate",
    "AuditReport",
    "EvolutionCandidate",
}
FORBIDDEN_RAW_REACTION_KEYS = {
    "rxn",
    "rxn_smiles",
    "rxn_smiles_list",
    "reaction_smiles",
    "raw_reaction",
    "raw_reactions",
    "raw_reaction_candidates",
    "reaction_candidates",
    "route_tree_actions",
    "candidate_actions",
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


@dataclass
class WorkerRunRecord:
    run_id: str
    task_id: str
    case_id: str
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    output_artifact: dict[str, Any] | None = None
    output_validation: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    timed_out: bool = False
    schema_version: str = WORKER_RUN_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WorkerRunner = Callable[[WorkerTask], WorkerProcessResult]


def run_codex_worker(
    task: WorkerTask,
    *,
    runner: WorkerRunner | None = None,
    mock_output: dict[str, Any] | None = None,
    command: list[str] | None = None,
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

    started = time.monotonic()
    try:
        if mock_output is not None or task.dry_run:
            artifact = mock_output if mock_output is not None else mock_worker_artifact(task)
            process = WorkerProcessResult(stdout=json.dumps(artifact, ensure_ascii=False), exit_code=0)
        elif runner is not None:
            process = runner(task)
        elif command is not None:
            process = _run_subprocess_worker(task, command)
        else:
            artifact = mock_worker_artifact(task)
            process = WorkerProcessResult(stdout=json.dumps(artifact, ensure_ascii=False), exit_code=0)
    except TimeoutError as exc:
        return WorkerRunRecord(
            run_id=f"{task.task_id}:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="timeout",
            stderr=str(exc),
            output_validation={"accepted": False, "reasons": ["timeout"], "schema_version": WORKER_OUTPUT_VALIDATION_SCHEMA},
            elapsed_s=round(time.monotonic() - started, 3),
            timed_out=True,
        )

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
        stdout=stdout,
        stderr=_truncate(process.stderr, task.budget.max_output_bytes),
        exit_code=process.exit_code,
        tool_calls=list(process.tool_calls or []),
        output_artifact=artifact if isinstance(artifact, dict) else None,
        output_validation=validation,
        elapsed_s=round(time.monotonic() - started, 3),
    )


def mock_worker_artifact(task: WorkerTask) -> dict[str, Any]:
    """Return a minimal typed draft artifact for dry-run/mock execution."""
    return {
        "schema_version": f"{task.required_artifact_type.lower()}.draft.v1",
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
        ),
        objective=str(data.get("objective") or ""),
        allowed_workdir=str(data.get("allowed_workdir") or "."),
        dry_run=bool(data.get("dry_run", False)),
        schema_version=str(data.get("schema_version") or WORKER_TASK_SCHEMA),
    )


def _run_subprocess_worker(task: WorkerTask, command: list[str]) -> WorkerProcessResult:
    try:
        proc = subprocess.run(
            command,
            cwd=str(Path(task.allowed_workdir).resolve()),
            capture_output=True,
            text=True,
            timeout=float(task.budget.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"worker timeout after {task.budget.timeout_s}s") from exc
    return WorkerProcessResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=int(proc.returncode))


def _parse_worker_stdout(stdout: str) -> Any:
    try:
        return json.loads(stdout or "")
    except json.JSONDecodeError:
        return None


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
    if int(process.exit_code or 0) != 0:
        reasons.append("worker_exit_code_nonzero")
    tool_calls = list(process.tool_calls or [])
    if len(tool_calls) > int(task.budget.max_tool_calls):
        reasons.append("tool_call_budget_exceeded")
    allowed = {str(tool) for tool in task.allowed_tools}
    if allowed:
        for call in tool_calls:
            if str(call.get("tool") or call.get("name") or "") not in allowed:
                reasons.append("tool_not_allowed")
                break
    return reasons


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
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RAW_REACTION_KEYS:
                return True
            if _contains_raw_reaction_injection(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction_injection(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False
