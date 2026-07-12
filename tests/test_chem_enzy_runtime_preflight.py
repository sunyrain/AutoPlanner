from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cascade_planner.baselines.chem_enzy_runtime import (
    CHEMENZY_CAPABILITY_PROBE_SCHEMA,
    CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA,
    _run_capability_probe,
    _stable_digest,
    chem_enzy_python_candidates,
    chem_enzy_runtime_selection_from_request,
    clear_chem_enzy_runtime_probe_cache,
    diagnose_chem_enzy_runtime,
    resolve_chem_enzy_python,
)
from cascade_planner.baselines.chem_enzy_runtime_probe import probe_chem_enzy_runtime
from cascade_planner.harness.tools import (
    ToolExecutionState,
    _apply_chemenzy_request_runtime_overrides,
    _chem_enzy_runtime_preflight,
    _execute_chemenzy_request,
)


@pytest.fixture(autouse=True)
def _clear_runtime_probe_cache() -> None:
    clear_chem_enzy_runtime_probe_cache()
    yield
    clear_chem_enzy_runtime_probe_cache()


def _runtime_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    env_prefix = tmp_path / "runtime" / "env"
    vendor_root = tmp_path / "ChemEnzyRetroPlanner"
    vendor_config = vendor_root / "retro_planner" / "config" / "config.yaml"
    vendor_config.parent.mkdir(parents=True)
    vendor_config.write_text("stock: {}\n", encoding="utf-8")
    launcher = tmp_path / "run_chem_enzy_plan_for_web.py"
    launcher.write_text("# diagnostic fixture only\n", encoding="utf-8")
    return env_prefix, vendor_root, launcher


def test_runtime_selection_mirrors_actual_request_models_stocks_and_overrides() -> None:
    models, stocks, overrides = chem_enzy_runtime_selection_from_request(
        {
            "one_step_models": ["template_relevance.reaxys"],
            "stock_mode": "paroutes-n5",
            "chem_enzy_onmt_tokenizer": "token",
        }
    )

    assert models == ["template_relevance.reaxys"]
    assert stocks == ["PaRotes_n5-stock"]
    assert overrides == {"chem_enzy_onmt_tokenizer": "token"}


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
    assert report["accepted"] is False
    assert report["filesystem_accepted"] is True
    assert report["production_ready"] is False
    assert report["status"] == "filesystem_ready_capability_unverified"
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

    assert report["accepted"] is False
    assert report["filesystem_accepted"] is True
    assert report["env_prefix"] == str(env_prefix)
    assert report["env_prefix_selection_source"] == "environment"
    assert report["python_executable"] == str(python_path)


def test_runtime_preflight_requires_successful_isolated_capability_probe(
    tmp_path: Path,
) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    python_path = env_prefix / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    capability = {
        "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
        "accepted": True,
        "status": "ready",
        "python_executable": str(python_path),
        "issues": [],
        "semantics": {
            "planner_constructed": False,
            "checkpoint_deserialized": False,
            "model_loaded": False,
            "search_executed": False,
        },
    }

    with patch(
        "cascade_planner.baselines.chem_enzy_runtime._run_capability_probe",
        return_value=capability,
    ) as probe:
        report = diagnose_chem_enzy_runtime(
            env_prefix=env_prefix,
            vendor_root=vendor_root,
            launcher_path=launcher,
            platform_name="nt",
            capability_probe=True,
        )

    assert report["accepted"] is True
    assert report["production_ready"] is True
    assert report["status"] == "ready"
    assert report["capability_probe"] == capability
    assert probe.call_args.kwargs["python_executable"] == python_path


def test_runtime_preflight_rejects_filesystem_ready_but_import_incapable_runtime(
    tmp_path: Path,
) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    python_path = env_prefix / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")

    with patch(
        "cascade_planner.baselines.chem_enzy_runtime._run_capability_probe",
        return_value={
            "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
            "accepted": False,
            "status": "runtime_unavailable",
            "issues": ["chem_enzy_capability_probe_failed"],
        },
    ):
        report = diagnose_chem_enzy_runtime(
            env_prefix=env_prefix,
            vendor_root=vendor_root,
            launcher_path=launcher,
            platform_name="nt",
            capability_probe=True,
        )

    assert report["filesystem_accepted"] is True
    assert report["accepted"] is False
    assert report["production_ready"] is False
    assert report["status"] == "runtime_unavailable"
    assert report["issues"] == ["chem_enzy_capability_probe_failed"]


def test_capability_probe_uses_selected_interpreter_and_bounded_child_environment(
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "python.exe"
    python_path.write_text("", encoding="utf-8")
    vendor_root = tmp_path / "vendor"
    vendor_root.mkdir()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        request_path = Path(command[command.index("--request") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        selected = [request["one_step_models"][0]]
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
                    "accepted": True,
                    "status": "ready",
                    "python_executable": str(python_path.resolve()),
                    "issues": [],
                    "requested_one_step_models": request["one_step_models"],
                    "requested_stock_names": request["stock_names"],
                    "probe_request_digest": _stable_digest(request),
                    "selected_one_step_models": selected,
                    "model_path_checks": [
                        {"model": selected[0], "readable": True}
                    ],
                    "stock_path_checks": [
                        {"stock": name, "readable": True}
                        for name in request["stock_names"]
                    ],
                    "vendor_imports": {
                        "retro_planner.api": True,
                        "retro_planner.search_frame.mcts_star.mol_tree": True,
                    },
                    "semantics": {
                        "planner_constructed": False,
                        "checkpoint_deserialized": False,
                        "model_loaded": False,
                        "search_executed": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        assert command[0] == str(python_path)
        assert kwargs["env"]["CHEMENZY_PANDARALLEL_WORKERS"] == "1"  # type: ignore[index]
        assert kwargs["env"]["PYTHONPATH"] == str(Path(__file__).resolve().parents[1])  # type: ignore[index]
        assert kwargs["timeout"] == 12.0
        return subprocess.CompletedProcess(command, 0, stdout="noise", stderr="")

    with patch(
        "cascade_planner.baselines.chem_enzy_runtime.subprocess.run",
        side_effect=fake_run,
    ):
        report = _run_capability_probe(
            python_executable=python_path,
            vendor_root=vendor_root,
            timeout_s=12.0,
            environ={},
        )

    assert report["accepted"] is True
    assert report["status"] == "ready"


@pytest.mark.parametrize("accepted", [True, False])
def test_runtime_capability_probe_cache_reuses_success_and_failure_by_identity(
    tmp_path: Path,
    accepted: bool,
) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    python_path = env_prefix / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("python", encoding="utf-8")
    capability = {
        "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
        "accepted": accepted,
        "status": "ready" if accepted else "runtime_unavailable",
        "issues": [] if accepted else ["probe_fixture_failure"],
    }

    with patch(
        "cascade_planner.baselines.chem_enzy_runtime._run_capability_probe",
        return_value=capability,
    ) as probe:
        first = diagnose_chem_enzy_runtime(
            env_prefix=env_prefix,
            vendor_root=vendor_root,
            launcher_path=launcher,
            platform_name="nt",
            capability_probe=True,
            one_step_models=["graphfp_models.request_a"],
            stock_names=["stock_a"],
            model_overrides={"chem_enzy_onmt_tokenizer": "char"},
        )
        second = diagnose_chem_enzy_runtime(
            env_prefix=env_prefix,
            vendor_root=vendor_root,
            launcher_path=launcher,
            platform_name="nt",
            capability_probe=True,
            one_step_models=["graphfp_models.request_a"],
            stock_names=["stock_a"],
            model_overrides={"chem_enzy_onmt_tokenizer": "char"},
        )

    assert probe.call_count == 1
    assert first["capability_probe_cache"]["hit"] is False
    assert second["capability_probe_cache"]["hit"] is True
    assert first["capability_probe_cache"]["cache_key"] == second[
        "capability_probe_cache"
    ]["cache_key"]
    assert second["capability_probe"] == capability


def test_runtime_capability_probe_cache_separates_request_and_config_identity(
    tmp_path: Path,
) -> None:
    env_prefix, vendor_root, launcher = _runtime_fixture(tmp_path)
    python_path = env_prefix / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("python", encoding="utf-8")
    capability = {
        "schema_version": CHEMENZY_CAPABILITY_PROBE_SCHEMA,
        "accepted": True,
        "status": "ready",
        "issues": [],
    }
    common = {
        "env_prefix": env_prefix,
        "vendor_root": vendor_root,
        "launcher_path": launcher,
        "platform_name": "nt",
        "capability_probe": True,
    }

    with patch(
        "cascade_planner.baselines.chem_enzy_runtime._run_capability_probe",
        return_value=capability,
    ) as probe:
        model_a = diagnose_chem_enzy_runtime(
            **common,
            one_step_models=["graphfp_models.a"],
            stock_names=["stock_a"],
        )
        model_b = diagnose_chem_enzy_runtime(
            **common,
            one_step_models=["graphfp_models.b"],
            stock_names=["stock_a"],
        )
        stock_b = diagnose_chem_enzy_runtime(
            **common,
            one_step_models=["graphfp_models.b"],
            stock_names=["stock_b"],
        )
        override_b = diagnose_chem_enzy_runtime(
            **common,
            one_step_models=["graphfp_models.b"],
            stock_names=["stock_b"],
            model_overrides={"chem_enzy_onmt_tokenizer": "token"},
        )
        config_path = vendor_root / "retro_planner" / "config" / "config.yaml"
        config_path.write_text("stocks: {changed: changed.csv}\n", encoding="utf-8")
        changed_config = diagnose_chem_enzy_runtime(
            **common,
            one_step_models=["graphfp_models.b"],
            stock_names=["stock_b"],
            model_overrides={"chem_enzy_onmt_tokenizer": "token"},
        )

    assert probe.call_count == 5
    keys = {
        report["capability_probe_cache"]["cache_key"]
        for report in (model_a, model_b, stock_b, override_b, changed_config)
    }
    assert len(keys) == 5


def test_tools_runtime_preflight_passes_actual_request_selection() -> None:
    with patch(
        "cascade_planner.harness.tools.diagnose_chem_enzy_runtime",
        return_value={"accepted": False},
    ) as diagnose:
        _chem_enzy_runtime_preflight(
            request={
                "one_step_models": ["onmt_models.custom"],
                "stock_names": ["Custom-stock"],
                "chem_enzy_onmt_model_path": "models/custom.pt",
            }
        )

    assert diagnose.call_args.kwargs["one_step_models"] == [
        "onmt_models.custom"
    ]
    assert diagnose.call_args.kwargs["stock_names"] == ["Custom-stock"]
    assert diagnose.call_args.kwargs["model_overrides"] == {
        "chem_enzy_onmt_model_path": "models/custom.pt"
    }


def test_request_onmt_override_is_bound_to_actual_runtime_environment() -> None:
    env: dict[str, str] = {}

    _apply_chemenzy_request_runtime_overrides(
        env,
        {
            "chem_enzy_onmt_model_path": ["first.pt", "second.pt"],
            "chem_enzy_onmt_tokenizer": "token",
        },
    )

    assert env["AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH"] == "first.pt,second.pt"
    assert env["AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER"] == "token"


def test_probe_fails_closed_for_missing_stock_and_unknown_model_config(
    tmp_path: Path,
) -> None:
    vendor_root = tmp_path / "ChemEnzyRetroPlanner"
    config_path = vendor_root / "retro_planner" / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "stocks:\n  Known-stock: known.csv\none_step_model_configs: {}\n",
        encoding="utf-8",
    )

    missing_stock = probe_chem_enzy_runtime(
        vendor_root=vendor_root,
        one_step_models=["graphfp_models.missing"],
        stock_names=["Missing-stock"],
    )
    missing_model = probe_chem_enzy_runtime(
        vendor_root=vendor_root,
        one_step_models=["mystery_models.unknown"],
        stock_names=["Known-stock"],
    )
    invalid_override = probe_chem_enzy_runtime(
        vendor_root=vendor_root,
        one_step_models=["graphfp_models.missing"],
        stock_names=["Known-stock"],
        model_overrides={"chem_enzy_onmt_tokenizer": "pretokenized"},
    )

    assert missing_stock["accepted"] is False
    assert "chem_enzy_selected_stock_config_missing" in missing_stock["issues"]
    assert missing_model["accepted"] is False
    assert "chem_enzy_selected_model_configuration_invalid" in missing_model[
        "issues"
    ]
    assert missing_model["semantics"] == {
        "planner_constructed": False,
        "checkpoint_deserialized": False,
        "model_loaded": False,
        "search_executed": False,
    }
    assert invalid_override["accepted"] is False
    assert "chem_enzy_probe_request_onmt_tokenizer_invalid" in invalid_override[
        "issues"
    ]


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

    with patch(
        "cascade_planner.harness.tools._chem_enzy_runtime_preflight",
        return_value=runtime_preflight,
    ) as preflight:
        result = _execute_chemenzy_request(
            state=state,
            request={"target_smiles": "CCO"},
            request_path=request_path,
            output_path=output_path,
            timeout_s=1.0,
        )

    assert preflight.call_count == 1
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
    runtime_preflight = {
        "schema_version": CHEMENZY_RUNTIME_PREFLIGHT_SCHEMA,
        "accepted": True,
        "production_ready": True,
        "status": "ready",
        "python_executable": "python",
        "env_prefix_selection_source": "cli",
        "issues": [],
    }

    with (
        patch(
            "cascade_planner.harness.tools._chem_enzy_python_bin",
            return_value=Path("python"),
        ),
        patch("cascade_planner.harness.tools.subprocess.Popen", return_value=process),
        patch("cascade_planner.harness.tools._terminate_process_group"),
        patch(
            "cascade_planner.harness.tools._runtime_preflight_for_python",
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

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert result["status"] == "timeout"
    assert result["reasons"] == ["chem_enzy_timeout"]
    assert result["runtime_preflight"]["env_prefix_selection_source"] == "cli"
    assert json.loads(
        (tmp_path / "chem_enzy_runtime_preflight.json").read_text(encoding="utf-8")
    ) == runtime_preflight
