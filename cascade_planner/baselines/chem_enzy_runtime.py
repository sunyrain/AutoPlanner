"""ChemEnzy runtime discovery and isolated capability diagnostics.

The parent process never imports ChemEnzy, Torch, or model code.  Production
readiness is established by a bounded subprocess running under the selected
isolated interpreter.  A filesystem-only report is explicitly non-production
and cannot authorize a ChemEnzy launch.
"""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import threading
import time
from copy import deepcopy
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA = "chemenzy_runtime_preflight.v2"
CHEMENZY_ENV_VARIABLE = "CHEMENZY_ENV_PREFIX"
CHEMENZY_ENV_SOURCE_VARIABLE = "AUTOPLANNER_CHEMENZY_ENV_PREFIX_SOURCE"
CHEMENZY_CAPABILITY_PROBE_SCHEMA = "chemenzy_runtime_capability_probe.v1"
CHEMENZY_RUNTIME_PROBE_CACHE_SCHEMA = "chemenzy_runtime_probe_cache.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHEMENZY_ENV_PREFIX = (
    PROJECT_ROOT.parent / "chem_enzy_runtime" / "envs" / "retro_planner_env"
)
DEFAULT_CHEMENZY_VENDOR_ROOT = PROJECT_ROOT / "vendor" / "ChemEnzyRetroPlanner"
DEFAULT_CHEMENZY_LAUNCHER = PROJECT_ROOT / "scripts" / "run_chem_enzy_plan_for_web.py"
DEFAULT_CAPABILITY_ONE_STEP_MODELS = (
    "graphfp_models.USPTO-full_remapped",
    "onmt_models.bionav_one_step",
    "onmt_models.bionav_native_one_step",
)
DEFAULT_CAPABILITY_STOCK_NAMES = (
    "Zinc_Fix-stock",
    "PaRotes_n1-stock",
)
DEFAULT_CAPABILITY_PROBE_CACHE_TTL_S = 45.0
_CAPABILITY_OVERRIDE_ENVIRONMENT_KEYS = (
    "AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH",
    "AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER",
    "AUTOPLANNER_CHEMENZY_ONMT_SOURCE_PREFIX",
    "AUTOPLANNER_CHEMENZY_ONMT_PRETOKENIZE_MODE",
)
CHEMENZY_MODEL_OVERRIDE_REQUEST_FIELDS = (
    "chem_enzy_onmt_model_path",
    "onmt_model_path",
    "chem_enzy_onmt_tokenizer",
    "onmt_tokenizer",
)
_CAPABILITY_PROBE_CACHE: dict[
    str,
    tuple[float, float, dict[str, Any]],
] = {}
_CAPABILITY_PROBE_INFLIGHT: dict[str, threading.Event] = {}
_CAPABILITY_PROBE_CACHE_LOCK = threading.RLock()


def chem_enzy_stock_content_binding(
    *,
    stock_paths: Mapping[str, Any],
    capability_stock_checks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile full-file stock identity for provider replay bindings."""

    if stock_paths:
        rows = [
            _stock_file_content_check(name, path)
            for name, path in sorted(stock_paths.items(), key=lambda item: str(item[0]))
        ]
    else:
        rows = [
            {
                "stock": str(row.get("stock") or ""),
                "path": str(row.get("path") or ""),
                "size_bytes": int(row.get("size_bytes") or 0),
                "content_sha256": str(row.get("content_sha256") or ""),
                "content_digest_scope": str(row.get("content_digest_scope") or ""),
                "content_digest_status": str(row.get("content_digest_status") or ""),
            }
            for row in capability_stock_checks
            if isinstance(row, Mapping)
        ]
    complete = bool(rows) and all(
        row["content_digest_status"] == "complete" and bool(row["content_sha256"])
        for row in rows
    )
    return {
        "checks": rows,
        "binding_sha256": _stable_digest(rows) if complete else "",
        "identity_complete": complete,
    }


def _stock_file_content_check(name: Any, raw_path: Any) -> dict[str, Any]:
    path = Path(str(raw_path or "")).expanduser().resolve()
    if not path.is_file():
        return _stock_content_error(name, path, "missing")
    try:
        stat = path.stat()
        content_sha256 = _cached_stock_file_sha256(
            str(path), int(stat.st_size), int(stat.st_mtime_ns)
        )
        final_stat = path.stat()
    except OSError:
        return _stock_content_error(name, path, "error")
    complete = (
        int(final_stat.st_size) == int(stat.st_size)
        and int(final_stat.st_mtime_ns) == int(stat.st_mtime_ns)
    )
    return {
        "stock": str(name),
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "content_sha256": content_sha256 if complete else "",
        "content_digest_scope": "full_file_bytes",
        "content_digest_status": "complete" if complete else "changed_during_read",
    }


def _stock_content_error(name: Any, path: Path, status: str) -> dict[str, Any]:
    return {
        "stock": str(name),
        "path": str(path),
        "size_bytes": 0,
        "content_sha256": "",
        "content_digest_scope": "full_file_bytes",
        "content_digest_status": status,
    }


@lru_cache(maxsize=128)
def _cached_stock_file_sha256(path: str, size_bytes: int, mtime_ns: int) -> str:
    del size_bytes, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def chem_enzy_python_candidates(env_prefix: Path | str) -> tuple[tuple[str, Path], ...]:
    """Return supported interpreter layouts in deterministic preference order."""

    prefix = Path(env_prefix).expanduser()
    return (
        ("windows_venv_scripts", prefix / "Scripts" / "python.exe"),
        ("windows_conda_root", prefix / "python.exe"),
        ("posix_bin", prefix / "bin" / "python"),
    )


def resolve_chem_enzy_python(
    env_prefix: Path | str,
    *,
    platform_name: str | None = None,
) -> Path | None:
    """Resolve an isolated ChemEnzy interpreter without falling back to this process."""

    host_family = _host_family(platform_name)
    for layout, candidate in chem_enzy_python_candidates(env_prefix):
        if candidate.is_file() and _layout_host_compatible(layout, host_family):
            return candidate
    return None


def chem_enzy_runtime_selection_from_request(
    request_payload: Mapping[str, Any] | None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Mirror the launcher model/stock selection for production preflight."""

    payload = dict(request_payload or {})
    raw_models = payload.get("one_step_models")
    one_step_models: Any = (
        raw_models if raw_models else list(DEFAULT_CAPABILITY_ONE_STEP_MODELS)
    )
    raw_stocks = payload.get("stock_names")
    if raw_stocks:
        stock_names: Any = raw_stocks
    else:
        stock_mode = str(payload.get("stock_mode") or "building-block").strip().lower()
        if stock_mode in {"commercial", "zinc", "zinc_fix", "zinc-fix"}:
            stock_names = ["Zinc_Fix-stock"]
        elif stock_mode in {"benchmark-n5", "paroutes-n5", "n5"}:
            stock_names = ["PaRotes_n5-stock"]
        elif stock_mode in {
            "building-block",
            "building_block",
            "strict",
            "paroutes-n1",
            "n1",
        }:
            stock_names = ["PaRotes_n1-stock"]
        else:
            # Mirrors scripts/run_chem_enzy_plan_for_web.py::_stock_names_from_payload.
            stock_names = ["Zinc_Fix-stock"]
    model_overrides = {
        key: payload[key]
        for key in CHEMENZY_MODEL_OVERRIDE_REQUEST_FIELDS
        if key in payload
    }
    return one_step_models, stock_names, model_overrides


def diagnose_chem_enzy_runtime(
    *,
    env_prefix: Path | str | None = None,
    vendor_root: Path | str | None = None,
    launcher_path: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    capability_probe: bool = False,
    capability_probe_timeout_s: float = 60.0,
    one_step_models: list[str] | tuple[str, ...] | None = None,
    stock_names: list[str] | tuple[str, ...] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    capability_probe_cache_ttl_s: float = DEFAULT_CAPABILITY_PROBE_CACHE_TTL_S,
) -> dict[str, Any]:
    """Return a launch preflight report for the configured ChemEnzy runtime.

    With ``capability_probe=False`` this performs only path/file checks and
    reports ``filesystem_ready_capability_unverified``.  Production callers
    must request the isolated subprocess probe; that probe imports the vendor
    API and checks configured files without constructing a planner, loading a
    checkpoint, or executing search.
    """

    environment = os.environ if environ is None else environ
    requested_models, model_selection_issues = _normalize_requested_names(
        one_step_models,
        defaults=DEFAULT_CAPABILITY_ONE_STEP_MODELS,
        field="one_step_models",
    )
    requested_stocks, stock_selection_issues = _normalize_requested_names(
        stock_names,
        defaults=DEFAULT_CAPABILITY_STOCK_NAMES,
        field="stock_names",
    )
    normalized_overrides, override_issues = _normalize_model_overrides(
        model_overrides
    )
    request_selection_issues = [
        *model_selection_issues,
        *stock_selection_issues,
        *override_issues,
    ]
    if env_prefix is not None:
        configured_prefix_value = env_prefix
        selection_source = str(
            environment.get(CHEMENZY_ENV_SOURCE_VARIABLE) or "explicit_argument"
        )
    elif environment.get(CHEMENZY_ENV_VARIABLE):
        configured_prefix_value = environment[CHEMENZY_ENV_VARIABLE]
        selection_source = str(
            environment.get(CHEMENZY_ENV_SOURCE_VARIABLE) or "environment"
        )
    else:
        configured_prefix_value = DEFAULT_CHEMENZY_ENV_PREFIX
        selection_source = "default"
    configured_prefix = Path(configured_prefix_value).expanduser()
    configured_vendor_root = Path(vendor_root or DEFAULT_CHEMENZY_VENDOR_ROOT).expanduser()
    configured_launcher = Path(launcher_path or DEFAULT_CHEMENZY_LAUNCHER).expanduser()
    candidates = chem_enzy_python_candidates(configured_prefix)
    host_family = _host_family(platform_name)

    selected_layout = ""
    selected_python: Path | None = None
    candidate_rows: list[dict[str, Any]] = []
    for layout, candidate in candidates:
        exists = candidate.is_file()
        host_compatible = _layout_host_compatible(layout, host_family)
        candidate_rows.append(
            {
                "layout": layout,
                "path": str(candidate),
                "exists": exists,
                "host_compatible": host_compatible,
            }
        )
        if selected_python is None and exists and host_compatible:
            selected_layout = layout
            selected_python = candidate

    vendor_config = configured_vendor_root / "retro_planner" / "config" / "config.yaml"
    checks = {
        "env_prefix_exists": configured_prefix.is_dir(),
        "python_interpreter_found": selected_python is not None,
        "vendor_root_exists": configured_vendor_root.is_dir(),
        "vendor_config_exists": vendor_config.is_file(),
        "launcher_exists": configured_launcher.is_file(),
        "request_selection_accepted": not request_selection_issues,
        "capability_probe_accepted": False,
    }
    issues: list[str] = list(request_selection_issues)
    if not checks["env_prefix_exists"]:
        issues.append("chem_enzy_env_prefix_not_found")
    if not checks["python_interpreter_found"]:
        issue = (
            "chem_enzy_runtime_python_incompatible_with_host"
            if any(row["exists"] for row in candidate_rows)
            else "chem_enzy_runtime_python_not_found"
        )
        issues.append(issue)
    if not checks["vendor_root_exists"]:
        issues.append("chem_enzy_vendor_root_not_found")
    elif not checks["vendor_config_exists"]:
        issues.append("chem_enzy_vendor_config_not_found")
    if not checks["launcher_exists"]:
        issues.append("chem_enzy_launcher_not_found")

    filesystem_issues = [
        issue
        for issue in issues
        if not issue.startswith("chem_enzy_request_")
    ]
    filesystem_accepted = not filesystem_issues
    request_selection_accepted = not request_selection_issues
    capability_report: dict[str, Any] = {}
    capability_cache: dict[str, Any] = {}
    if (
        filesystem_accepted
        and request_selection_accepted
        and capability_probe
        and selected_python is not None
    ):
        cache_key, cache_components = _capability_probe_cache_key(
            env_prefix=configured_prefix,
            python_executable=selected_python,
            vendor_root=configured_vendor_root,
            vendor_config=vendor_config,
            launcher_path=configured_launcher,
            one_step_models=requested_models,
            stock_names=requested_stocks,
            model_overrides=normalized_overrides,
            environ=environment,
            timeout_s=capability_probe_timeout_s,
        )
        capability_report, capability_cache = _cached_capability_probe(
            cache_key=cache_key,
            cache_components=cache_components,
            ttl_s=capability_probe_cache_ttl_s,
            python_executable=selected_python,
            vendor_root=configured_vendor_root,
            timeout_s=capability_probe_timeout_s,
            environ=environment,
            one_step_models=requested_models,
            stock_names=requested_stocks,
            model_overrides=normalized_overrides,
        )
        checks["capability_probe_accepted"] = (
            capability_report.get("accepted") is True
        )
        if not checks["capability_probe_accepted"]:
            issues.extend(
                str(item)
                for item in capability_report.get("issues")
                or ["chem_enzy_capability_probe_failed"]
            )
    production_ready = bool(
        filesystem_accepted
        and request_selection_accepted
        and capability_probe
        and checks["capability_probe_accepted"]
    )
    # ``accepted`` is deliberately production semantics.  A caller that only
    # needs discovery can inspect ``filesystem_accepted`` without accidentally
    # authorizing execution.
    accepted = production_ready
    status = (
        "ready"
        if production_ready
        else (
            "filesystem_ready_capability_unverified"
            if (
                filesystem_accepted
                and request_selection_accepted
                and not capability_probe
            )
            else "runtime_unavailable"
        )
    )
    return {
        "schema_version": CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA,
        "accepted": accepted,
        "filesystem_accepted": filesystem_accepted,
        "request_selection_accepted": request_selection_accepted,
        "production_ready": production_ready,
        "status": status,
        "probe_scope": (
            "isolated_import_and_config_capability_no_planner_model_or_search"
            if capability_probe
            else "filesystem_only_no_process_or_model_execution"
        ),
        "env_variable": CHEMENZY_ENV_VARIABLE,
        "env_prefix_selection_source": selection_source,
        "host_family": host_family,
        "env_prefix": str(configured_prefix),
        "python_executable": str(selected_python) if selected_python is not None else "",
        "python_layout": selected_layout,
        "python_candidates": candidate_rows,
        "vendor_root": str(configured_vendor_root),
        "vendor_config": str(vendor_config),
        "launcher_path": str(configured_launcher),
        "checks": checks,
        "requested_one_step_models": requested_models,
        "requested_stock_names": requested_stocks,
        "model_override_digest": _stable_digest(normalized_overrides),
        "capability_probe": capability_report,
        "capability_probe_cache": capability_cache,
        "issues": list(dict.fromkeys(issues)),
    }


def clear_chem_enzy_runtime_probe_cache() -> None:
    """Clear completed process-local probe entries (primarily for tests)."""

    with _CAPABILITY_PROBE_CACHE_LOCK:
        _CAPABILITY_PROBE_CACHE.clear()


def _normalize_requested_names(
    value: list[str] | tuple[str, ...] | None,
    *,
    defaults: tuple[str, ...],
    field: str,
) -> tuple[list[str], list[str]]:
    if value is None or value == [] or value == ():
        return list(defaults), []
    if not isinstance(value, (list, tuple)):
        return [], [f"chem_enzy_request_{field}_must_be_list"]
    names: list[str] = []
    issues: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            issues.append(f"chem_enzy_request_{field}_invalid_item:{index}")
            continue
        names.append(item)
    if not names:
        issues.append(f"chem_enzy_request_{field}_empty")
    return names, issues


def _normalize_model_overrides(
    value: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    if value is None:
        return {}, []
    if not isinstance(value, Mapping):
        return {}, ["chem_enzy_request_model_overrides_must_be_mapping"]
    try:
        normalized = _json_safe(dict(value))
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return {}, ["chem_enzy_request_model_overrides_not_serializable"]
    if not isinstance(normalized, dict):
        return {}, ["chem_enzy_request_model_overrides_must_be_mapping"]
    return normalized, []


def _capability_probe_cache_key(
    *,
    env_prefix: Path,
    python_executable: Path,
    vendor_root: Path,
    vendor_config: Path,
    launcher_path: Path,
    one_step_models: list[str],
    stock_names: list[str],
    model_overrides: Mapping[str, Any],
    environ: Mapping[str, str],
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    adapter_path = PROJECT_ROOT / "cascade_planner" / "baselines" / "chem_enzy_adapter.py"
    probe_path = PROJECT_ROOT / "cascade_planner" / "baselines" / "chem_enzy_runtime_probe.py"
    vendor_runtime_paths = (
        vendor_root / "retro_planner" / "api.py",
        vendor_root / "retro_planner" / "common" / "prepare_utils.py",
        vendor_root
        / "retro_planner"
        / "search_frame"
        / "mcts_star"
        / "mol_tree.py",
    )
    environment_marker_paths = (
        env_prefix / "conda-meta" / "history",
        env_prefix / "pyvenv.cfg",
    )
    environment_overrides = {
        key: str(environ.get(key) or "")
        for key in _CAPABILITY_OVERRIDE_ENVIRONMENT_KEYS
    }
    request_selection = {
        "one_step_models": list(one_step_models),
        "stock_names": list(stock_names),
        "model_overrides": _json_safe(dict(model_overrides)),
        "environment_overrides": environment_overrides,
    }
    components = {
        "schema_version": CHEMENZY_RUNTIME_PROBE_CACHE_SCHEMA,
        "env_prefix": _path_fingerprint(env_prefix, content_digest=False),
        "python_executable": _path_fingerprint(
            python_executable,
            content_digest=False,
        ),
        "environment_markers": [
            _path_fingerprint(path, content_digest=True)
            for path in environment_marker_paths
        ],
        "vendor_root": _path_fingerprint(vendor_root, content_digest=False),
        "vendor_runtime_files": [
            _path_fingerprint(path, content_digest=True)
            for path in vendor_runtime_paths
        ],
        "vendor_config": _path_fingerprint(vendor_config, content_digest=True),
        "launcher": _path_fingerprint(launcher_path, content_digest=True),
        "adapter": _path_fingerprint(adapter_path, content_digest=True),
        "probe_module": _path_fingerprint(probe_path, content_digest=True),
        "request_selection_digest": _stable_digest(request_selection),
        "timeout_s": min(300.0, max(1.0, float(timeout_s or 60.0))),
    }
    return _stable_digest(components), components


def _cached_capability_probe(
    *,
    cache_key: str,
    cache_components: Mapping[str, Any],
    ttl_s: float,
    python_executable: Path,
    vendor_root: Path,
    timeout_s: float,
    environ: Mapping[str, str],
    one_step_models: list[str],
    stock_names: list[str],
    model_overrides: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ttl = min(300.0, max(1.0, float(ttl_s or DEFAULT_CAPABILITY_PROBE_CACHE_TTL_S)))
    wait_deadline = time.monotonic() + min(
        310.0,
        max(2.0, float(timeout_s or 60.0) + 5.0),
    )
    owner = False
    while not owner:
        now = time.monotonic()
        with _CAPABILITY_PROBE_CACHE_LOCK:
            _prune_expired_capability_probe_cache(now)
            cached = _CAPABILITY_PROBE_CACHE.get(cache_key)
            if cached is not None and cached[0] > now:
                report = deepcopy(cached[2])
                return report, _capability_cache_audit(
                    cache_key,
                    cache_components,
                    hit=True,
                    ttl_s=cached[1],
                    remaining_ttl_s=cached[0] - now,
                )
            event = _CAPABILITY_PROBE_INFLIGHT.get(cache_key)
            if event is None:
                event = threading.Event()
                _CAPABILITY_PROBE_INFLIGHT[cache_key] = event
                owner = True
        if not owner:
            remaining = wait_deadline - time.monotonic()
            if remaining <= 0 or not event.wait(timeout=remaining):
                report = {
                    "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
                    "accepted": False,
                    "status": "runtime_unavailable",
                    "issues": ["chem_enzy_capability_probe_cache_wait_timeout"],
                }
                return report, _capability_cache_audit(
                    cache_key,
                    cache_components,
                    hit=False,
                    ttl_s=ttl,
                    remaining_ttl_s=0.0,
                    wait_timeout=True,
                )

    try:
        try:
            report = _run_capability_probe(
                python_executable=python_executable,
                vendor_root=vendor_root,
                timeout_s=timeout_s,
                environ=environ,
                one_step_models=one_step_models,
                stock_names=stock_names,
                model_overrides=model_overrides,
            )
        except Exception as exc:  # cache boundary must release waiters
            report = {
                "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
                "accepted": False,
                "status": "runtime_unavailable",
                "issues": ["chem_enzy_capability_probe_parent_failure"],
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                },
            }
        expiry = time.monotonic() + ttl
        with _CAPABILITY_PROBE_CACHE_LOCK:
            _CAPABILITY_PROBE_CACHE[cache_key] = (
                expiry,
                ttl,
                deepcopy(report),
            )
        return deepcopy(report), _capability_cache_audit(
            cache_key,
            cache_components,
            hit=False,
            ttl_s=ttl,
            remaining_ttl_s=ttl,
        )
    finally:
        with _CAPABILITY_PROBE_CACHE_LOCK:
            event = _CAPABILITY_PROBE_INFLIGHT.pop(cache_key, None)
            if event is not None:
                event.set()


def _prune_expired_capability_probe_cache(now: float) -> None:
    expired = [
        key
        for key, (expires_at, _ttl, _report) in _CAPABILITY_PROBE_CACHE.items()
        if expires_at <= now
    ]
    for key in expired:
        _CAPABILITY_PROBE_CACHE.pop(key, None)
    if len(_CAPABILITY_PROBE_CACHE) > 64:
        oldest = sorted(
            _CAPABILITY_PROBE_CACHE.items(),
            key=lambda item: item[1][0],
        )
        for key, _value in oldest[: len(oldest) - 64]:
            _CAPABILITY_PROBE_CACHE.pop(key, None)


def _capability_cache_audit(
    cache_key: str,
    components: Mapping[str, Any],
    *,
    hit: bool,
    ttl_s: float,
    remaining_ttl_s: float,
    wait_timeout: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": CHEMENZY_RUNTIME_PROBE_CACHE_SCHEMA,
        "cache_key": cache_key,
        "hit": hit,
        "ttl_s": round(ttl_s, 3),
        "remaining_ttl_s": round(max(0.0, remaining_ttl_s), 3),
        "request_selection_digest": str(
            components.get("request_selection_digest") or ""
        ),
        "runtime_components_digest": _stable_digest(dict(components)),
        "wait_timeout": wait_timeout,
    }


def _path_fingerprint(path: Path, *, content_digest: bool) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    out: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    try:
        stat = resolved.stat()
    except OSError as exc:
        out["stat_error"] = type(exc).__name__
        return out
    out.update(
        {
            "kind": "file" if resolved.is_file() else "directory" if resolved.is_dir() else "other",
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    if content_digest and resolved.is_file():
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            out["sha256"] = digest.hexdigest()
        except OSError as exc:
            out["digest_error"] = type(exc).__name__
    return out


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported cache-key value: {type(value).__name__}")


def _run_capability_probe(
    *,
    python_executable: Path,
    vendor_root: Path,
    timeout_s: float,
    environ: Mapping[str, str],
    one_step_models: list[str] | None = None,
    stock_names: list[str] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the model-free probe under the selected interpreter."""

    timeout = min(300.0, max(1.0, float(timeout_s or 60.0)))
    child_env = os.environ.copy()
    child_env.update({str(key): str(value) for key, value in environ.items()})
    # Probe the same self-contained import surface used for execution; an
    # ambient PYTHONPATH could otherwise make an incomplete runtime appear
    # capable through packages borrowed from the controller environment.
    child_env["PYTHONPATH"] = str(PROJECT_ROOT)
    child_env["CHEMENZY_PANDARALLEL_WORKERS"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    requested_models = list(
        one_step_models
        if one_step_models is not None
        else DEFAULT_CAPABILITY_ONE_STEP_MODELS
    )
    requested_stocks = list(
        stock_names
        if stock_names is not None
        else DEFAULT_CAPABILITY_STOCK_NAMES
    )
    probe_request = {
        "one_step_models": requested_models,
        "stock_names": requested_stocks,
        "model_overrides": _json_safe(dict(model_overrides or {})),
    }
    probe_request_digest = _stable_digest(probe_request)
    with tempfile.TemporaryDirectory(prefix="autoplanner-chemenzy-probe-") as tmp:
        output_path = Path(tmp) / "capability.json"
        request_path = Path(tmp) / "request.json"
        request_path.write_text(
            json.dumps(
                probe_request,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        command = [
            str(python_executable),
            "-m",
            "cascade_planner.baselines.chem_enzy_runtime_probe",
            "--vendor-root",
            str(vendor_root),
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
                "accepted": False,
                "status": "runtime_unavailable",
                "issues": ["chem_enzy_capability_probe_timeout"],
                "timeout_s": timeout,
                "stdout_tail": _bounded_tail(exc.stdout),
                "stderr_tail": _bounded_tail(exc.stderr),
            }
        if not output_path.is_file():
            return {
                "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
                "accepted": False,
                "status": "runtime_unavailable",
                "issues": ["chem_enzy_capability_probe_output_missing"],
                "exit_code": int(completed.returncode),
                "stdout_tail": _bounded_tail(completed.stdout),
                "stderr_tail": _bounded_tail(completed.stderr),
            }
        try:
            if output_path.stat().st_size > 2_000_000:
                raise ValueError("capability probe output exceeds 2 MB")
            report = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {
                "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
                "accepted": False,
                "status": "runtime_unavailable",
                "issues": ["chem_enzy_capability_probe_output_invalid"],
                "failure": {"type": type(exc).__name__, "message": str(exc)[:500]},
                "exit_code": int(completed.returncode),
            }
    if not isinstance(report, dict) or report.get("schema_version") != (
        CHEMENZY_CAPABILITY_PROBE_SCHEMA
    ):
        return {
            "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
            "accepted": False,
            "status": "runtime_unavailable",
            "issues": ["chem_enzy_capability_probe_schema_invalid"],
            "exit_code": int(completed.returncode),
        }
    expected_python = os.path.normcase(str(Path(python_executable).resolve()))
    observed_python = os.path.normcase(
        str(Path(str(report.get("python_executable") or "")).resolve())
    )
    issues = [str(item) for item in report.get("issues") or []]
    if report.get("accepted") is not True or report.get("status") != "ready":
        issues.append("chem_enzy_capability_probe_not_ready")
    semantics = report.get("semantics")
    if semantics != {
        "planner_constructed": False,
        "checkpoint_deserialized": False,
        "model_loaded": False,
        "search_executed": False,
    }:
        issues.append("chem_enzy_capability_probe_semantics_invalid")
    selected_models = report.get("selected_one_step_models")
    if not isinstance(selected_models, list) or not selected_models:
        issues.append("chem_enzy_capability_probe_no_selected_model")
    elif any(model not in requested_models for model in selected_models):
        issues.append("chem_enzy_capability_probe_selected_model_mismatch")
    if report.get("requested_one_step_models") != requested_models:
        issues.append("chem_enzy_capability_probe_requested_models_mismatch")
    if report.get("requested_stock_names") != requested_stocks:
        issues.append("chem_enzy_capability_probe_requested_stocks_mismatch")
    if report.get("probe_request_digest") != probe_request_digest:
        issues.append("chem_enzy_capability_probe_request_digest_mismatch")
    for field in ("model_path_checks", "stock_path_checks"):
        rows = report.get(field)
        if (
            not isinstance(rows, list)
            or not rows
            or not all(
                isinstance(row, dict) and row.get("readable") is True
                for row in rows
            )
        ):
            issues.append(f"chem_enzy_capability_probe_{field}_invalid")
    model_rows = report.get("model_path_checks")
    if isinstance(selected_models, list) and isinstance(model_rows, list):
        checked_models = {
            str(row.get("model") or "")
            for row in model_rows
            if isinstance(row, dict)
        }
        if any(model not in checked_models for model in selected_models):
            issues.append("chem_enzy_capability_probe_model_coverage_incomplete")
    stock_rows = report.get("stock_path_checks")
    if isinstance(stock_rows, list):
        checked_stocks = {
            str(row.get("stock") or "")
            for row in stock_rows
            if isinstance(row, dict)
        }
        if any(stock not in checked_stocks for stock in requested_stocks):
            issues.append("chem_enzy_capability_probe_stock_coverage_incomplete")
    imports = report.get("vendor_imports")
    if not isinstance(imports, dict) or any(
        imports.get(name) is not True
        for name in (
            "retro_planner.api",
            "retro_planner.search_frame.mcts_star.mol_tree",
        )
    ):
        issues.append("chem_enzy_capability_probe_vendor_import_invalid")
    if observed_python != expected_python:
        issues.append("chem_enzy_capability_probe_interpreter_mismatch")
    if completed.returncode != 0:
        issues.append("chem_enzy_capability_probe_nonzero_exit")
    if issues:
        report["accepted"] = False
        report["status"] = "runtime_unavailable"
        report["issues"] = sorted(set(issues))
        report["exit_code"] = int(completed.returncode)
        report["stdout_tail"] = _bounded_tail(completed.stdout)
        report["stderr_tail"] = _bounded_tail(completed.stderr)
    return report


def _bounded_tail(value: str | bytes | None, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else value.decode("utf-8", errors="replace")
    return text[-limit:]


def _host_family(platform_name: str | None) -> str:
    value = str(platform_name or os.name).strip().lower()
    return "windows" if value in {"nt", "windows", "win32"} or value.startswith("win") else "posix"


def _layout_host_compatible(layout: str, host_family: str) -> bool:
    return layout.startswith("windows_") if host_family == "windows" else layout == "posix_bin"


def format_chem_enzy_runtime_diagnostic(report: Mapping[str, Any]) -> str:
    """Format a concise operator-facing failure message from a preflight report."""

    issues = ", ".join(str(item) for item in report.get("issues") or []) or "unknown_runtime_error"
    candidates = ", ".join(
        str(row.get("path") or "")
        for row in report.get("python_candidates") or []
        if isinstance(row, Mapping)
    )
    return f"ChemEnzy runtime preflight failed: {issues}; interpreter candidates: {candidates}"


__all__ = [
    "CHEMENZY_CAPABILITY_PROBE_SCHEMA",
    "CHEMENZY_ENV_SOURCE_VARIABLE",
    "CHEMENZY_MODEL_OVERRIDE_REQUEST_FIELDS",
    "CHEMENZY_RUNTIME_PROBE_CACHE_SCHEMA",
    "CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA",
    "DEFAULT_CAPABILITY_ONE_STEP_MODELS",
    "DEFAULT_CAPABILITY_STOCK_NAMES",
    "DEFAULT_CHEMENZY_ENV_PREFIX",
    "chem_enzy_python_candidates",
    "chem_enzy_runtime_selection_from_request",
    "chem_enzy_stock_content_binding",
    "clear_chem_enzy_runtime_probe_cache",
    "diagnose_chem_enzy_runtime",
    "format_chem_enzy_runtime_diagnostic",
    "resolve_chem_enzy_python",
]
