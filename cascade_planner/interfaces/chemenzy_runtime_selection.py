"""Bounded host-local discovery for an executable ChemEnzy environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

from cascade_planner.baselines.chem_enzy_runtime import diagnose_chem_enzy_runtime


_ENV_PREFIX = "CHEMENZY_ENV_PREFIX"
_ENV_SOURCE = "AUTOPLANNER_CHEMENZY_ENV_PREFIX_SOURCE"


def select_chemenzy_runtime(
    *,
    env_prefix: str | Path | None,
    timeout_s: float,
    vendor_root: str | Path | None = None,
    one_step_models: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one runtime without overriding explicit operator intent."""

    explicit = str(env_prefix or "").strip()
    environment = dict(os.environ)
    environment_prefix = str(environment.get(_ENV_PREFIX) or "").strip()
    if explicit or environment_prefix:
        selected = explicit or environment_prefix
        source = "target_solve_config" if explicit else "environment"
        environment[_ENV_SOURCE] = source
        report = _diagnose(
            selected,
            vendor_root,
            environment,
            timeout_s,
            one_step_models=one_step_models,
        )
        return report, _runtime_discovery(source, selected, [report])

    default_report = _diagnose(
        None,
        vendor_root,
        environment,
        timeout_s,
        one_step_models=one_step_models,
    )
    if default_report.get("production_ready") is True:
        selected = str(default_report.get("env_prefix") or "")
        return default_report, _runtime_discovery(
            "repository_default", selected, [default_report]
        )

    attempts = [default_report]
    attempted_prefixes = {Path(str(default_report.get("env_prefix") or ""))}
    discovery_groups = (
        ("conda_auto_discovery", _registered_conda_prefixes(environment)),
        ("host_python_auto_discovery", _host_python_prefixes()),
    )
    for source, candidates in discovery_groups:
        for candidate in candidates:
            if candidate in attempted_prefixes:
                continue
            attempted_prefixes.add(candidate)
            discovered = dict(environment)
            discovered[_ENV_SOURCE] = source
            report = _diagnose(
                candidate,
                vendor_root,
                discovered,
                timeout_s,
                one_step_models=one_step_models,
            )
            attempts.append(report)
            if report.get("production_ready") is True:
                return report, _runtime_discovery(source, str(candidate), attempts)
    return default_report, _runtime_discovery("unresolved", "", attempts)


def _diagnose(
    env_prefix: str | Path | None,
    vendor_root: str | Path | None,
    environ: Mapping[str, str],
    timeout_s: float,
    *,
    one_step_models: tuple[str, ...] | list[str] | None,
) -> dict[str, Any]:
    return diagnose_chem_enzy_runtime(
        env_prefix=env_prefix,
        vendor_root=vendor_root,
        environ=environ,
        capability_probe=True,
        capability_probe_timeout_s=timeout_s,
        one_step_models=one_step_models,
    )


def _registered_conda_prefixes(environ: Mapping[str, str]) -> list[Path]:
    candidates: set[Path] = set()
    current = str(environ.get("CONDA_PREFIX") or "").strip()
    if current:
        candidates.add(Path(current).expanduser())
    registry = Path.home() / ".conda" / "environments.txt"
    if registry.is_file():
        try:
            candidates.update(
                Path(line.strip()).expanduser()
                for line in registry.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:
            pass
    conda = shutil.which("conda")
    if conda:
        try:
            completed = subprocess.run(
                [conda, "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if completed.returncode == 0:
                payload = json.loads(completed.stdout or "{}")
                candidates.update(
                    Path(str(value)).expanduser()
                    for value in payload.get("envs") or []
                    if str(value).strip()
                )
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
    valid = [value for value in candidates if _candidate_python_exists(value)]
    return sorted(valid, key=lambda value: _candidate_rank(value, current))[:8]


def _host_python_prefixes() -> list[Path]:
    """Return the current host interpreter prefix as a final probed fallback."""

    candidates = {
        Path(sys.prefix).expanduser(),
        Path(sys.executable).expanduser().resolve().parent,
    }
    return sorted(
        (value for value in candidates if _candidate_python_exists(value)),
        key=lambda value: str(value).lower(),
    )


def _candidate_python_exists(prefix: Path) -> bool:
    paths = (
        prefix / "python.exe",
        prefix / "Scripts" / "python.exe",
        prefix / "bin" / "python",
    )
    return any(path.is_file() for path in paths)


def _candidate_rank(prefix: Path, current: str) -> tuple[Any, ...]:
    name = prefix.name.lower()
    hint = 0 if any(value in name for value in ("chem", "retro", "planner")) else 1
    site_packages = prefix / ("Lib" if os.name == "nt" else "lib")
    package_score = sum(
        any(site_packages.glob(pattern))
        for pattern in (
            "site-packages/torch*",
            "site-packages/rdkit*",
            "site-packages/torchtext*",
        )
    )
    return (
        0 if current and prefix == Path(current) else 1,
        hint,
        -package_score,
        str(prefix).lower(),
    )


def _runtime_discovery(
    source: str, selected_prefix: str, attempts: list[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": "chemenzy_runtime_discovery.v1",
        "source": source,
        "selected_env_prefix": selected_prefix,
        "attempts": [
            {
                "env_prefix": str(value.get("env_prefix") or ""),
                "selection_source": str(
                    value.get("env_prefix_selection_source") or ""
                ),
                "filesystem_accepted": value.get("filesystem_accepted") is True,
                "importable": value.get("production_ready") is True,
                "issues": list(value.get("issues") or []),
            }
            for value in attempts
        ],
        "semantics": {
            "explicit_configuration_never_silently_overridden": True,
            "auto_discovery_is_bounded": True,
            "all_auto_discovered_runtimes_require_capability_probe": True,
        },
    }


__all__ = ["select_chemenzy_runtime"]
