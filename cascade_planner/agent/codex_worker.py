"""Controlled Codex research worker wrapper for P3.

The worker contract is deliberately conservative. A worker may produce typed
draft artifacts and a trace record, but it cannot mutate route trees, write
production KB entries, or mark a case solved.
"""
from __future__ import annotations

import json
import os
import hashlib
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


WORKER_TASK_SCHEMA = "worker_task.v1"
WORKER_RUN_RECORD_SCHEMA = "worker_run_record.v1"
WORKER_OUTPUT_VALIDATION_SCHEMA = "worker_output_validation.v1"
DEFAULT_RETROSYNTHESIS_KEY_FILE = Path(__file__).resolve().parents[2] / "key.txt"

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
    "LiteratureScoutReport",
    "StrategicDisconnectionCard",
    "LiteratureRouteSegmentCard",
    "SegmentStepCandidate",
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


def run_codex_worker(
    task: WorkerTask,
    *,
    runner: WorkerRunner | None = None,
    mock_output: dict[str, Any] | None = None,
    command: list[str] | None = None,
    use_codex_cli: bool | None = None,
    use_api_json: bool | None = None,
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
            process = _run_subprocess_worker(task, command)
        elif _use_api_json(use_api_json):
            backend = "api_json"
            process = _run_api_json_worker(task)
        elif _use_codex_cli(use_codex_cli):
            backend = "codex_cli"
            process = _run_codex_cli_worker(task)
        else:
            backend = "default_mock"
            artifact = mock_worker_artifact(task)
            process = WorkerProcessResult(stdout=json.dumps(artifact, ensure_ascii=False), exit_code=0, backend=backend)
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
        raise WorkerTimeoutError(
            f"worker timeout after {task.budget.timeout_s}s",
            backend="subprocess_command",
            command=command,
        ) from exc
    return WorkerProcessResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=int(proc.returncode),
        backend="subprocess_command",
        command=list(command),
    )


def _run_codex_cli_worker(task: WorkerTask) -> WorkerProcessResult:
    executable = _codex_executable()
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"Codex CLI executable not found: {executable}")

    workdir = Path(task.allowed_workdir or ".").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    prompt = _codex_worker_prompt(task)
    with tempfile.TemporaryDirectory(prefix="autoplanner_codex_worker_") as tmp:
        tmp_path = Path(tmp)
        output_path = tmp_path / "last_message.json"
        schema_path = tmp_path / "worker_output_schema.json"
        schema_path.write_text(json.dumps(_worker_output_json_schema(task), indent=2), encoding="utf-8")
        command = _codex_cli_command(
            executable=executable,
            workdir=workdir,
            output_path=output_path,
            schema_path=schema_path,
        )
        try:
            proc = subprocess.run(
                command,
                cwd=str(workdir),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=float(task.budget.timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerTimeoutError(
                f"worker timeout after {task.budget.timeout_s}s",
                backend="codex_cli",
                command=command,
            ) from exc
        final = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else proc.stdout
        stderr = proc.stderr or ""
        if proc.stdout and final != proc.stdout:
            stderr = (stderr + "\n" if stderr else "") + "codex_cli_stdout:\n" + proc.stdout
        return WorkerProcessResult(
            stdout=final,
            stderr=stderr,
            exit_code=int(proc.returncode),
            backend="codex_cli",
            command=list(command),
            metadata={},
        )


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
            f"model_reasoning_effort = {_toml_string(os.environ.get('AUTOPLANNER_CODEX_WORKER_REASONING_EFFORT') or 'xhigh')}",
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
        "base_url_fingerprint": _base_url_fingerprint(config["base_url"]),
        "model": config["model"],
        "auth_source": str(DEFAULT_RETROSYNTHESIS_KEY_FILE),
        "codex_home": "ephemeral",
    }
    return env, metadata


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
    executable: str,
    workdir: Path,
    output_path: Path,
    schema_path: Path,
) -> list[str]:
    command = [executable]
    if _env_flag("AUTOPLANNER_CODEX_WORKER_SEARCH", default=True):
        command.append("--search")
    command.extend([
        "--ask-for-approval",
        "never",
    ])
    command.extend([
        "exec",
        "--cd",
        str(workdir),
        "--sandbox",
        os.environ.get("AUTOPLANNER_CODEX_WORKER_SANDBOX") or "read-only",
        "--ephemeral",
        "--color",
        "never",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ])
    model = os.environ.get("AUTOPLANNER_CODEX_WORKER_MODEL")
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


def _codex_worker_prompt(task: WorkerTask) -> str:
    task_json = json.dumps(task.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
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
        "- Do not inject raw reaction candidates or reaction SMILES. Avoid strings containing '>>' unless the task explicitly asks for audit of an existing input reference.",
        "- Prefer traceable sources. For literature evidence, include DOI, URL, or local_ref in payload/source metadata.",
        "- Use only the task context, repository files, and allowed tools implied by the task.",
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
        _artifact_payload_instruction(task.required_artifact_type),
        "",
        "WorkerTask:",
        task_json,
    ])


def _artifact_payload_instruction(artifact_type: str) -> str:
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
    if artifact_type == "StrategicDisconnectionCard":
        return (
            "For payload, describe the strategic disconnection without raw reaction injection: "
            "evidence_refs, candidate_kind or retrosynthetic_move, target/frontier context, "
            "strategic_subgoal, anchor_candidate, limitations, and fake-terminal guardrails."
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
    if artifact_type == "EvidenceCard":
        return _evidence_card_payload_json_schema(task)
    if artifact_type == "LiteratureScoutReport":
        return _literature_scout_payload_json_schema(task)
    if artifact_type == "LiteratureRouteSegmentCard":
        return _literature_route_segment_payload_json_schema(task)
    if artifact_type == "SegmentStepCandidate":
        return _segment_step_json_schema()
    if artifact_type == "ConditionCandidate":
        return _condition_candidate_json_schema()
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


def _typed_artifact_schema_version(artifact_type: str) -> str:
    return {
        "ResearchReport": "research_report.v1",
        "EvidenceCard": "evidence_card_artifact.v1",
        "LiteratureScoutReport": "literature_scout_report_artifact.v1",
        "StrategicDisconnectionCard": "strategic_disconnection_card.v1",
        "LiteratureRouteSegmentCard": "literature_route_segment_card.v1",
        "SegmentStepCandidate": "segment_step_candidate.v1",
        "FailureDiagnosis": "failure_diagnosis.v1",
        "StrategicOperator": "strategic_operator_artifact.v1",
        "ConditionCandidate": "condition_candidate.v1",
        "AuditReport": "route_audit_report.v1",
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
