from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cascade_planner.baselines.chem_enzy_runtime import (
    CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA,
    chem_enzy_python_candidates,
    diagnose_chem_enzy_runtime,
    resolve_chem_enzy_python,
)
from cascade_planner.harness.tools import ToolExecutionState, _execute_chemenzy_request


def _runtime_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    env_prefix = tmp_path / "runtime" / "env"
    vendor_root = tmp_path / "ChemEnzyRetroPlanner"
    vendor_config = vendor_root / "retro_planner" / "config" / "config.yaml"
    vendor_config.parent.mkdir(parents=True)
    vendor_config.write_text("stock: {}\n", encoding="utf-8")
    launcher = tmp_path / "run_chem_enzy_plan_for_web.py"
    launcher.write_text("# diagnostic fixture only\n", encoding="utf-8")
    return env_prefix, vendor_root, launcher


@pytest.mark.parametrize(
    ("relative_python", "expected_layout", "platform_name"),
    [
        (Path("Scripts/python.exe"), "windows_venv_scripts", "nt"),
        (Path("python.exe"), "windows_conda_root", "win32"),
        (Path("bin/python"), "posix_bin", "posix"),
    ],
)
def test_runtime_preflight_supports_windows_and_posix_interpreter_layouts(
    tmp_path: Path,
    relative_python: Path,
    expected_layout: str,
    platform_name: str,
) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    python_path = env_prefix / relative_python
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")

    report = diagnose_chem_enzy_runtime(
        env_prefix=env_prefix,
        vendor_root=vendor_root,
        launcher_path=launcher,
        platform_name=platform_name,
    )

    assert report["schema_version"] == CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA
    assert report["accepted"] is True
    assert report["status"] == "ready"
    assert report["probe_scope"] == "filesystem_only_no_process_or_model_execution"
    assert report["python_executable"] == str(python_path)
    assert report["python_layout"] == expected_layout
    assert resolve_chem_enzy_python(env_prefix, platform_name=platform_name) == python_path


def test_runtime_preflight_uses_deterministic_interpreter_precedence(tmp_path: Path) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    for _layout, candidate in chem_enzy_python_candidates(env_prefix):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("", encoding="utf-8")

    report = diagnose_chem_enzy_runtime(
        env_prefix=env_prefix,
        vendor_root=vendor_root,
        launcher_path=launcher,
        platform_name="nt",
    )

    assert report["python_layout"] == "windows_venv_scripts"
    assert report["python_executable"] == str(env_prefix / "Scripts" / "python.exe")


def test_runtime_preflight_reports_all_missing_launch_dependencies_without_running_models(tmp_path: Path) -> None:
    env_prefix = tmp_path / "missing-env"
    report = diagnose_chem_enzy_runtime(
        env_prefix=env_prefix,
        vendor_root=tmp_path / "missing-vendor",
        launcher_path=tmp_path / "missing-launcher.py",
    )

    assert report["accepted"] is False
    assert report["status"] == "runtime_unavailable"
    assert report["python_executable"] == ""
    assert report["issues"] == [
        "chem_enzy_env_prefix_not_found",
        "chem_enzy_runtime_python_not_found",
        "chem_enzy_vendor_root_not_found",
        "chem_enzy_launcher_not_found",
    ]
    assert all(row["exists"] is False for row in report["python_candidates"])


def test_runtime_preflight_rejects_posix_interpreter_on_windows_host(tmp_path: Path) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    posix_python = env_prefix / "bin" / "python"
    posix_python.parent.mkdir(parents=True)
    posix_python.write_text("", encoding="utf-8")

    report = diagnose_chem_enzy_runtime(
        env_prefix=env_prefix,
        vendor_root=vendor_root,
        launcher_path=launcher,
        platform_name="nt",
    )

    assert report["accepted"] is False
    assert report["issues"] == ["chem_enzy_runtime_python_incompatible_with_host"]
    assert report["python_executable"] == ""
    assert report["python_candidates"][2] == {
        "layout": "posix_bin",
        "path": str(posix_python),
        "exists": True,
        "host_compatible": False,
    }


def test_runtime_preflight_reads_configured_prefix_from_environment(tmp_path: Path) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    python_path = env_prefix / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")

    report = diagnose_chem_enzy_runtime(
        vendor_root=vendor_root,
        launcher_path=launcher,
        environ={"CHEMENZY_ENV_PREFIX": str(env_prefix)},
        platform_name="nt",
    )

    assert report["accepted"] is True
    assert report["env_prefix"] == str(env_prefix)
    assert report["python_executable"] == str(python_path)


def test_runtime_unavailable_chemenzy_result_is_persisted_at_referenced_output(
    tmp_path: Path,
) -> None:
    state = ToolExecutionState(run_dir=tmp_path, target_input={}, preflight={})
    request_path = tmp_path / "subgoal_request.json"
    output_path = tmp_path / "subgoal_raw_result.json"
    runtime_preflight = {
        "status": "runtime_unavailable",
        "issues": ["chem_enzy_runtime_python_incompatible_with_host"],
    }

    with (
        patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=None),
        patch(
            "cascade_planner.harness.tools._chem_enzy_runtime_preflight",
            return_value=runtime_preflight,
        ),
    ):
        result = _execute_chemenzy_request(
            state=state,
            request={"target_smiles": "CCO"},
            request_path=request_path,
            output_path=output_path,
            timeout_s=1.0,
        )

    assert request_path.is_file()
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert result["status"] == "runtime_unavailable"
    assert result["reasons"] == [
        "chem_enzy_runtime_python_incompatible_with_host"
    ]


def test_timed_out_chemenzy_result_is_persisted_at_referenced_output(
    tmp_path: Path,
) -> None:
    state = ToolExecutionState(run_dir=tmp_path, target_input={}, preflight={})
    request_path = tmp_path / "timeout_request.json"
    output_path = tmp_path / "timeout_raw_result.json"
    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired(cmd="chemenzy", timeout=1.0)
    process.returncode = None

    with (
        patch(
            "cascade_planner.harness.tools._chem_enzy_python_bin",
            return_value=Path("python"),
        ),
        patch("cascade_planner.harness.tools.subprocess.Popen", return_value=process),
        patch("cascade_planner.harness.tools._terminate_process_group"),
    ):
        result = _execute_chemenzy_request(
            state=state,
            request={"target_smiles": "CCO"},
            request_path=request_path,
            output_path=output_path,
            timeout_s=1.0,
        )

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert result["status"] == "timeout"
    assert result["reasons"] == ["chem_enzy_timeout"]
