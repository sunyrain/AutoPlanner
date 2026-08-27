"""Main-runtime controller for interactive AiZ strategy-search sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Mapping

from cascade_planner.agent.codex_worker import (
    _close_windows_job,
    _create_windows_kill_job,
    _terminate_worker_process_group,
)


SCHEMA = "aizynthfinder_reactionjson_branch_sidecar.v2"
ExpansionRequestHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_aizynthfinder_strategy_branch_sidecar(
    *,
    target_smiles: str,
    strategy_id: str,
    strategy_text: str,
    request_handler: ExpansionRequestHandler,
    stock_index_path: str = "",
    inline_stock_smiles: tuple[str, ...] = (),
    python_executable: str = "",
    max_policy_calls: int = 25,
    max_candidates_per_call: int = 1,
    max_transforms: int = 25,
    exploration_constant: float = 1.4,
    max_mcts_iterations: int = 0,
    timeout_s: float = 18_000.0,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run one AiZ MCTS branch while model/replay calls stay in the host."""

    root = Path(__file__).resolve().parents[2]
    python_path = Path(
        python_executable
        or os.environ.get("AUTOPLANNER_AIZYNTH_PYTHON", "")
        or root / ".venv_aizynth" / "Scripts" / "python.exe"
    ).expanduser().resolve()
    script = root / "scripts" / "run_aizynthfinder_reactionjson_branch.py"
    if not python_path.is_file():
        raise RuntimeError(f"aizynthfinder strategy python missing: {python_path}")
    if not script.is_file():
        raise RuntimeError(f"aizynthfinder strategy sidecar missing: {script}")
    if not inline_stock_smiles and not Path(stock_index_path).is_file():
        raise RuntimeError("aizynthfinder strategy stock index missing")

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("aizynthfinder_strategy_sidecar_cancelled")

    process = subprocess.Popen(
        [str(python_path), str(script)],
        cwd=str(root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=(
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        ),
        start_new_session=(os.name != "nt"),
    )
    windows_job = _create_windows_kill_job(process)
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("aizynthfinder strategy sidecar pipes unavailable")

    stdout_queue: queue.Queue[str | None] = queue.Queue()
    stderr_rows: list[str] = []

    def read_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            stdout_queue.put(line)
        stdout_queue.put(None)

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_rows.append(line.rstrip())
            if len(stderr_rows) > 200:
                del stderr_rows[:50]

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()
    launch = {
        "schema_version": SCHEMA,
        "target_smiles": target_smiles,
        "strategy_id": strategy_id,
        "strategy_text": strategy_text,
        "stock_index_path": str(stock_index_path),
        "inline_stock_smiles": list(inline_stock_smiles),
        "max_policy_calls": int(max_policy_calls),
        "max_candidates_per_call": int(max_candidates_per_call),
        "max_transforms": int(max_transforms),
        "exploration_constant": float(exploration_constant),
        "max_mcts_iterations": int(max_mcts_iterations),
    }
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    diagnostics: list[str] = []
    try:
        _send(process, launch)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("aizynthfinder_strategy_sidecar_cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("aizynthfinder strategy sidecar timed out")
            try:
                line = stdout_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if process.poll() is not None:
                    raise RuntimeError(
                        "aizynthfinder strategy sidecar exited without result"
                    )
                continue
            if line is None:
                raise RuntimeError(
                    "aizynthfinder strategy sidecar closed without result"
                )
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                diagnostics.append(line.rstrip())
                continue
            if not isinstance(message, dict) or message.get("schema_version") != SCHEMA:
                diagnostics.append(line.rstrip())
                continue
            message_type = str(message.get("type") or "")
            if message_type == "expansion_request":
                try:
                    response = dict(
                        request_handler(dict(message.get("request") or {}))
                    )
                    _send(
                        process,
                        {
                            "type": "expansion_response",
                            "candidates": list(response.get("candidates") or []),
                            "model_call_consumed": bool(
                                response.get("model_call_consumed", True)
                            ),
                            "stop_search": bool(response.get("stop_search", False)),
                            "stop_reason": str(response.get("stop_reason") or ""),
                            "error": str(response.get("error") or ""),
                        },
                    )
                except Exception as exc:
                    _send(
                        process,
                        {
                            "type": "expansion_response",
                            "candidates": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    raise
                continue
            if message_type == "result":
                result = dict(message.get("result") or {})
                result["sidecar_stdout_diagnostics"] = diagnostics[-20:]
                result["sidecar_stderr_tail"] = stderr_rows[-40:]
                process.wait(timeout=10.0)
                if process.returncode != 0:
                    raise RuntimeError(
                        "aizynthfinder strategy sidecar failed after result"
                    )
                return result
            if message_type == "error":
                raise RuntimeError(
                    str(message.get("error") or "aizynthfinder strategy error")
                )
    finally:
        if windows_job is not None:
            _close_windows_job(windows_job)
            windows_job = None
        elif process.poll() is None:
            _terminate_worker_process_group(process)
        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                _terminate_worker_process_group(process)


def _send(process: subprocess.Popen[str], payload: Mapping[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("aizynthfinder strategy sidecar stdin unavailable")
    message = {"schema_version": SCHEMA, **dict(payload)}
    process.stdin.write(
        json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n"
    )
    process.stdin.flush()


__all__ = ["run_aizynthfinder_strategy_branch_sidecar"]
