"""Execute the built-in ChemEnzy launcher under one bounded subprocess call."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Mapping

from cascade_planner.interfaces.chemenzy_guidance import guided_native_search_policy
from cascade_planner.interfaces.chemenzy_probe_contract import ChemEnzyProposalRequest
from cascade_planner.interfaces.chemenzy_runtime_selection import select_chemenzy_runtime


_SUBPROCESS_LOCK = threading.Lock()


def run_builtin_chemenzy_probe(
    run_dir: Path,
    *,
    target_name: str,
    target_smiles: str,
    proposal_request: ChemEnzyProposalRequest,
    scope: str,
    env_prefix: str | Path | None,
    vendor_root: str | Path | None,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    preflight, discovery = select_chemenzy_runtime(
        env_prefix=env_prefix,
        vendor_root=vendor_root,
        timeout_s=min(30.0, float(limits["timeout_s"])),
        one_step_models=tuple(limits.get("one_step_models") or ()),
    )
    if preflight.get("production_ready") is not True:
        return _failure(
            "runtime_unavailable",
            "chemenzy_runtime_not_production_ready",
            preflight,
            discovery,
        )
    request = _launcher_request(
        target_name=target_name,
        target_smiles=target_smiles,
        proposal_request=proposal_request,
        limits=limits,
    )
    artifact_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(scope or "probe"))[:80]
    request_path = run_dir / f"chemenzy-v4-{artifact_stem}-request.json"
    output_path = run_dir / f"chemenzy-v4-{artifact_stem}-result.json"
    stdout_path = run_dir / f"chemenzy-v4-{artifact_stem}-stdout.log"
    stderr_path = run_dir / f"chemenzy-v4-{artifact_stem}-stderr.log"
    replay = _load_completed_probe(request_path, output_path, request=request)
    if replay is not None:
        environment = _launcher_environment(limits)
        return {
            **replay,
            "runtime_preflight": preflight,
            "runtime_discovery": discovery,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "search_executed": True,
            "provider_result_replayed": True,
            "request_path": str(request_path),
            "output_path": str(output_path),
            "queue_wait_s": 0.0,
            "pandarallel_workers": int(environment["CHEMENZY_PANDARALLEL_WORKERS"]),
            "launcher_request": request,
        }
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        str(preflight["python_executable"]),
        str(preflight["launcher_path"]),
        "--input",
        str(request_path),
        "--output",
        str(output_path),
        "--vendor-root",
        str(preflight["vendor_root"]),
        "--gpu",
        "-1",
    ]
    environment = _launcher_environment(limits)
    queued_at = time.monotonic()
    try:
        with _SUBPROCESS_LOCK:
            queue_wait_s = max(0.0, time.monotonic() - queued_at)
            completed = subprocess.run(
                command,
                cwd=str(Path(preflight["launcher_path"]).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=float(limits["timeout_s"]),
                check=False,
                env=environment,
            )
    except subprocess.TimeoutExpired:
        return _failure(
            "timeout",
            "chemenzy_bounded_probe_timeout",
            preflight,
            discovery,
        )
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0 or not output_path.is_file():
        return {
            **_failure(
                "failed",
                f"chemenzy_exit_{completed.returncode}",
                preflight,
                discovery,
            ),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            **_failure(
                "failed",
                "chemenzy_result_invalid",
                preflight,
                discovery,
            ),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    return {
        **dict(result),
        "runtime_preflight": preflight,
        "runtime_discovery": discovery,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "search_executed": True,
        "request_path": str(request_path),
        "output_path": str(output_path),
        "queue_wait_s": round(queue_wait_s, 3),
        "pandarallel_workers": int(environment["CHEMENZY_PANDARALLEL_WORKERS"]),
        "launcher_request": request,
    }


def _load_completed_probe(
    request_path: Path,
    output_path: Path,
    *, request: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Replay a complete same-request launcher result after host interruption."""

    if not request_path.is_file() or not output_path.is_file():
        return None
    try:
        persisted_request = json.loads(request_path.read_text(encoding="utf-8"))
        persisted_result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if persisted_request != dict(request) or not isinstance(persisted_result, Mapping):
        return None
    return dict(persisted_result)


def _launcher_request(
    *,
    target_name: str,
    target_smiles: str,
    proposal_request: ChemEnzyProposalRequest,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    request = {
        "target_name": target_name,
        "target_smiles": target_smiles,
        "planner_backend": "chem_enzy_native",
        "search_preset": str(limits.get("search_preset") or "standard"),
        "max_routes": limits["max_routes"],
        "max_steps": limits["max_steps"],
        "chem_enzy_iterations": limits["max_iterations"],
        "chem_enzy_expansion_topk": limits["expansion_topk"],
        "chemenzy_seed": int(limits.get("random_seed") or 0),
        "timeout_s": float(limits["timeout_s"]),
        "pandarallel_workers": int(limits.get("pandarallel_workers") or 2),
        "one_step_models": list(limits.get("one_step_models") or []),
        "stock_mode": "building-block",
        "device": "cpu",
        "enable_rule_verifier_gate": True,
        "stop_on_first_host_admitted_route": bool(
            limits.get("stop_on_first_host_admitted_route", False)
        ),
        "enable_condition_prediction": bool(
            limits.get("enable_condition_prediction", True)
        ),
        "enable_enzyme_assignment": bool(limits.get("enable_enzyme_assignment", True)),
        "enable_enzyme_coverage_sidecar": bool(
            limits.get("enable_enzyme_coverage_sidecar", True)
        ),
    }
    if limits.get("stock_names"):
        request["stock_names"] = list(limits["stock_names"])
    if limits.get("stock_paths"):
        request["stock_paths"] = dict(limits["stock_paths"])
    if proposal_request.mode == "guided_frontier":
        request["chem_enzy_search_policy"] = guided_native_search_policy(
            proposal_request,
            limits=limits,
        )
        request["search_preset"] = "thorough"
    return request


def _launcher_environment(limits: Mapping[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CHEMENZY_PANDARALLEL_WORKERS"] = str(
        max(1, min(8, int(limits.get("pandarallel_workers") or 2)))
    )
    environment["PYTHONHASHSEED"] = str(int(limits.get("random_seed") or 0))
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    return environment


def _failure(
    status: str,
    reason: str,
    preflight: Mapping[str, Any],
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "runtime_preflight": dict(preflight),
        "runtime_discovery": dict(discovery),
        "routes": [],
    }
