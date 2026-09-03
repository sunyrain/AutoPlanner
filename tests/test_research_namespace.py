from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPEN_RESEARCH_MODULES = (
    "cascade_planner.research.open_research_contract",
    "cascade_planner.research.open_research_experience",
    "cascade_planner.research.open_research_retrieval",
    "cascade_planner.research.open_research_seed_consumables",
    "cascade_planner.research.source_detail_resolution",
    "cascade_planner.research.source_material_locator",
    "cascade_planner.research.downstream_compiler",
    "cascade_planner.research.source_detail_chain_builder",
    "cascade_planner.research.route_failure_feedback",
    "cascade_planner.research.real_patent_procedure_gate",
)
AUTOPLANNRELLM_MODULES = (
    "cascade_planner.research.autoplannrellm.controller",
    "cascade_planner.research.autoplannrellm.deepseek_client",
    "cascade_planner.research.autoplannrellm.prior_benchmark",
    "cascade_planner.research.autoplannrellm.proposals",
    "cascade_planner.research.autoplannrellm.runner",
    "cascade_planner.research.autoplannrellm.live_benchmark",
)
RESEARCH_MODULES = OPEN_RESEARCH_MODULES + AUTOPLANNRELLM_MODULES
OPEN_STRUCTURE_MODULES = OPEN_RESEARCH_MODULES[:6]
OLD_HARNESS_MODULES = tuple(
    value.replace("cascade_planner.research", "cascade_planner.harness")
    for value in OPEN_RESEARCH_MODULES
)
OLD_AUTOPLANNRELLM_MODULES = tuple(
    value.replace(
        "cascade_planner.research.autoplannrellm",
        "AUTOPLANNRELLM",
    )
    for value in AUTOPLANNRELLM_MODULES
)
OLD_RESEARCH_BENCHMARK_MODULES = (
    "cascade_planner.cascadeboard.prior_benchmark",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    return imports


@pytest.mark.parametrize("module_name", OLD_HARNESS_MODULES)
def test_old_research_harness_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", OLD_AUTOPLANNRELLM_MODULES)
def test_old_autoplannrellm_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", OLD_RESEARCH_BENCHMARK_MODULES)
def test_old_research_benchmark_paths_are_deleted(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_research_runtime_uses_explicit_namespace() -> None:
    for module_name in RESEARCH_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_research_runtime_does_not_depend_on_legacy() -> None:
    for module_name in RESEARCH_MODULES:
        path = ROOT / f"{module_name.replace('.', '/')}.py"
        assert not any(
            imported.startswith("cascade_planner.legacy")
            for imported in _imports(path)
        )


def test_open_structure_launcher_imports_research_runtime_explicitly() -> None:
    path = ROOT / "scripts" / "run_open_structure_template_agent.py"
    imports = _imports(path)

    assert set(OPEN_STRUCTURE_MODULES) <= imports
    assert not set(OLD_HARNESS_MODULES) & imports


@pytest.mark.parametrize(
    ("script_name", "module_name"),
    [
        (
            "replay_deterministic_literature_registry.py",
            "cascade_planner.research.source_detail_chain_builder",
        ),
        (
            "replay_patent_xml_gate_suite.py",
            "cascade_planner.research.real_patent_procedure_gate",
        ),
    ],
)
def test_research_replay_scripts_use_explicit_modules(
    script_name: str,
    module_name: str,
) -> None:
    path = ROOT / "scripts" / script_name
    imports = _imports(path)

    assert module_name in imports
    assert module_name.replace("cascade_planner.research", "cascade_planner.harness") not in imports


def test_v4_fresh_import_does_not_load_research_runtime() -> None:
    script = """
import importlib
import json
import sys

importlib.import_module("cascade_planner.orchestration.retrosynthesis_service")
importlib.import_module("cascade_planner.interfaces.target_solver")
v4_app = importlib.import_module("cascade_planner.web.v4_app")
v4_app.create_v4_app(lambda: None)
print(json.dumps(sorted(name for name in sys.modules if name.startswith("cascade_planner.research"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_active_prior_surfaces_do_not_load_research_runtime() -> None:
    script = """
import importlib
import json
import sys

importlib.import_module("cascade_planner.agent.prior_generator")
importlib.import_module("cascade_planner.agent.cli")
print(json.dumps(sorted(name for name in sys.modules if name.startswith("cascade_planner.research"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_route_tree_mainline_does_not_discover_autoplannrellm() -> None:
    active_paths = (
        ROOT / "cascade_planner" / "route_tree" / "runtime.py",
        ROOT / "cascade_planner" / "route_tree" / "proposals.py",
        ROOT / "cascade_planner" / "route_tree" / "search.py",
        ROOT / "cascade_planner" / "cascadeboard" / "live_benchmark.py",
    )
    for path in active_paths:
        source = path.read_text(encoding="utf-8")
        assert "AUTOPLANNRELLM" not in source
        assert "cascade_planner.research" not in source


def test_autoplannrellm_environment_does_not_activate_mainline_imports() -> None:
    script = """
import importlib
import json
import os
import sys

os.environ["AUTOPLANNRELLM_ENABLE"] = "1"
os.environ["AUTOPLANNRELLM_LLM_SELECTION"] = "1"
os.environ["AUTOPLANNRELLM_ADD_LLM_CANDIDATE"] = "1"
importlib.import_module("cascade_planner.route_tree.runtime")
importlib.import_module("cascade_planner.route_tree.proposals")
print(json.dumps(sorted(name for name in sys.modules if name.startswith("cascade_planner.research"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_proposal_candidate_appender_is_explicitly_injected() -> None:
    proposals = importlib.import_module("cascade_planner.route_tree.proposals")
    calls = []

    def appender(**kwargs):
        calls.append(kwargs)
        return list(kwargs["actions"])

    tool = proposals.RetroEngineProposalTool(
        {},
        source_order=(),
        candidate_appender=appender,
    )
    assert tool.propose("CCO", top_k=1) == []
    assert len(calls) == 1
    assert calls[0]["product"] == "CCO"


def test_research_worker_builds_explicit_route_tree_extensions(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPLANNRELLM_ENABLE", "1")
    monkeypatch.setenv("AUTOPLANNRELLM_LLM_SELECTION", "0")
    monkeypatch.setenv("AUTOPLANNRELLM_ADD_LLM_CANDIDATE", "1")

    worker = importlib.import_module("cascade_planner.research.autoplannrellm.live_benchmark")
    extensions = worker.build_route_tree_extensions()

    assert extensions.controller_factory is None
    assert extensions.candidate_appender is not None


def test_research_runner_selects_explicit_worker_module() -> None:
    runner = importlib.import_module("cascade_planner.research.autoplannrellm.runner")
    forwarded = runner.with_research_worker(["--output", "run.json"])
    assert forwarded[-2:] == ["--worker-module", runner.RESEARCH_WORKER_MODULE]
    assert runner.with_research_worker(["--worker-module", "custom.worker"]) == [
        "--worker-module",
        "custom.worker",
    ]


def test_live_benchmark_and_route_tree_expose_extension_injection() -> None:
    live = importlib.import_module("cascade_planner.cascadeboard.live_benchmark")
    search = importlib.import_module("cascade_planner.route_tree.search")

    assert "route_tree_extensions" in inspect.signature(live.run_live_benchmark).parameters
    assert "source_gate" in inspect.signature(search.plan_with_route_tree).parameters
    assert "action_value_advisor" in inspect.signature(search.plan_with_route_tree).parameters
    assert "proposal_candidate_appender" in inspect.signature(search.plan_with_route_tree).parameters


def test_parallel_worker_module_is_forwarded_to_child_and_merge() -> None:
    parallel = importlib.import_module("cascade_planner.eval.run_live_benchmark_parallel")
    args = parallel.build_parser().parse_args(
        ["--output", "run.json", "--worker-module", "research.worker"]
    )
    worker_cmd = parallel._benchmark_cmd(args, Path("run_shard.json"), 0)
    merge_cmd = parallel._merge_cmd(args, [Path("run_shard.json")])

    assert worker_cmd[2:4] == ["research.worker", "--bench"]
    assert merge_cmd[2:4] == ["research.worker", "--merge"]
