"""Filesystem-only ChemEnzy runtime discovery and diagnostics.

This module deliberately does not import ChemEnzy, Torch, or model code.  It
is safe to call before a route-search request to explain whether the isolated
runtime can be launched.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA = "chemenzy_runtime_preflight.v1"
CHEMENZY_ENV_VARIABLE = "CHEMENZY_ENV_PREFIX"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHEMENZY_ENV_PREFIX = (
    PROJECT_ROOT.parent / "chem_enzy_runtime" / "envs" / "retro_planner_env"
)
DEFAULT_CHEMENZY_VENDOR_ROOT = PROJECT_ROOT / "vendor" / "ChemEnzyRetroPlanner"
DEFAULT_CHEMENZY_LAUNCHER = PROJECT_ROOT / "scripts" / "run_chem_enzy_plan_for_web.py"


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


def diagnose_chem_enzy_runtime(
    *,
    env_prefix: Path | str | None = None,
    vendor_root: Path | str | None = None,
    launcher_path: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Return a model-free launch preflight report for the ChemEnzy runtime.

    The diagnostic performs only path and file checks.  In particular, it does
    not start Python, import vendor packages, load checkpoints, or run a model.
    """

    environment = os.environ if environ is None else environ
    configured_prefix = Path(
        env_prefix
        if env_prefix is not None
        else environment.get(CHEMENZY_ENV_VARIABLE, str(DEFAULT_CHEMENZY_ENV_PREFIX))
    ).expanduser()
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
    }
    issues: list[str] = []
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

    accepted = not issues
    return {
        "schema_version": CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA,
        "accepted": accepted,
        "status": "ready" if accepted else "runtime_unavailable",
        "probe_scope": "filesystem_only_no_process_or_model_execution",
        "env_variable": CHEMENZY_ENV_VARIABLE,
        "host_family": host_family,
        "env_prefix": str(configured_prefix),
        "python_executable": str(selected_python) if selected_python is not None else "",
        "python_layout": selected_layout,
        "python_candidates": candidate_rows,
        "vendor_root": str(configured_vendor_root),
        "vendor_config": str(vendor_config),
        "launcher_path": str(configured_launcher),
        "checks": checks,
        "issues": issues,
    }


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
    "CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA",
    "DEFAULT_CHEMENZY_ENV_PREFIX",
    "chem_enzy_python_candidates",
    "diagnose_chem_enzy_runtime",
    "format_chem_enzy_runtime_diagnostic",
    "resolve_chem_enzy_python",
]
