from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib

from cascade_planner.application.compatibility_inventory import (
    compatibility_inventory,
    record_compatibility_use,
)
from cascade_planner.harness.tool_execution_policy import execute_registered_tool
from cascade_planner.harness.tool_registry import (
    LEGACY_LOCAL_TOOL_NAMES,
    bind_legacy_tool_registry,
)


ROOT = Path(__file__).resolve().parents[1]
V4_MODULES = (
    "cascade_planner/application/canonical_identity.py",
    "cascade_planner/application/canonical_hypergraph.py",
    "cascade_planner/application/deficit_frontier.py",
    "cascade_planner/application/frontier_runtime.py",
    "cascade_planner/application/portfolio_selection.py",
    "cascade_planner/application/proof_policy.py",
    "cascade_planner/application/proof_portfolio.py",
    "cascade_planner/application/route_variants.py",
    "cascade_planner/application/route_workbench.py",
    "cascade_planner/application/worker_runtime.py",
    "cascade_planner/interfaces/campaign_gateway.py",
    "cascade_planner/interfaces/campaign_operations.py",
    "cascade_planner/interfaces/replay_contract.py",
    "cascade_planner/interfaces/replay_pack.py",
    "cascade_planner/interfaces/replay_reporting.py",
    "cascade_planner/orchestration/global_campaign_director.py",
    "cascade_planner/orchestration/retrosynthesis_service.py",
)
FORBIDDEN_V4_DEPENDENCIES = (
    "cascade_planner.application.frontier_scheduler",
    "cascade_planner.application.retrosynthesis_acceptance",
    "cascade_planner.application.route_deficit_queue",
    "cascade_planner.application.route_portfolio",
    "cascade_planner.harness.agentic_blackboard_controller",
    "cascade_planner.harness.route_forest",
    "cascade_planner.orchestration.codex_retrosynthesis",
    "cascade_planner.web",
)
FOCUSED_LINE_BUDGETS = {
    "cascade_planner/application/canonical_identity.py": 200,
    "cascade_planner/application/compatibility_inventory.py": 260,
    "cascade_planner/application/frontier_runtime.py": 120,
    "cascade_planner/application/portfolio_selection.py": 400,
    "cascade_planner/application/proof_policy.py": 400,
    "cascade_planner/application/proof_portfolio.py": 400,
    "cascade_planner/application/route_variants.py": 400,
    "cascade_planner/application/route_workbench.py": 700,
    "cascade_planner/harness/tool_execution_policy.py": 120,
    "cascade_planner/harness/tool_registry.py": 120,
    "cascade_planner/harness/v4_controller_adapter.py": 180,
    "cascade_planner/harness/v4_route_workbench.py": 750,
    "cascade_planner/interfaces/campaign_gateway.py": 400,
    "cascade_planner/interfaces/campaign_operations.py": 160,
    "cascade_planner/interfaces/replay_contract.py": 200,
    "cascade_planner/interfaces/replay_pack.py": 500,
    "cascade_planner/interfaces/replay_reporting.py": 120,
    "cascade_planner/cli.py": 300,
    "cascade_planner/runtime/repository_audit.py": 360,
    "cascade_planner/web/v4_api.py": 220,
    "cascade_planner/orchestration/retrosynthesis_service.py": 400,
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_v4_modules_do_not_import_frozen_ownership_paths() -> None:
    violations = []
    for relative in V4_MODULES:
        for imported in _imports(ROOT / relative):
            if imported.startswith(FORBIDDEN_V4_DEPENDENCIES):
                violations.append(f"{relative}->{imported}")
            if relative.startswith("cascade_planner/application/") and imported.startswith(
                "cascade_planner.orchestration"
            ):
                violations.append(f"application_reverse_dependency:{relative}->{imported}")
    assert violations == []


def test_new_focused_modules_stay_within_practical_line_budgets() -> None:
    observed = {
        relative: len((ROOT / relative).read_text(encoding="utf-8").splitlines())
        for relative in FOCUSED_LINE_BUDGETS
    }
    assert {
        relative: lines
        for relative, lines in observed.items()
        if lines > FOCUSED_LINE_BUDGETS[relative]
    } == {}


def test_ruff_legacy_exceptions_cannot_cover_v4_or_tests() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ignored_patterns = config["tool"]["ruff"]["lint"]["per-file-ignores"]

    protected_prefixes = (
        "cascade_planner/application/",
        "cascade_planner/interfaces/",
        "cascade_planner/orchestration/",
        "cascade_planner/runtime/",
        "cascade_planner/web/",
        "tests/",
    )
    assert all(
        not pattern.startswith(prefix)
        for pattern in ignored_patterns
        for prefix in protected_prefixes
    )


def test_v4_workbench_adapter_does_not_execute_legacy_route_forest_compiler() -> None:
    imports = _imports(ROOT / "cascade_planner/harness/v4_route_workbench.py")

    assert "cascade_planner.harness.route_forest" not in imports
    assert "cascade_planner.harness.route_forest_delivery" in imports


def test_every_compatibility_shim_has_replacement_telemetry_and_milestone() -> None:
    inventory = compatibility_inventory()
    rows = inventory["shims"]

    assert inventory["content_sha256"] == _digest(
        {key: value for key, value in inventory.items() if key != "content_sha256"}
    )
    assert len({row["shim_id"] for row in rows}) == len(rows)
    assert all(row["scientific_write_authority"] is False for row in rows)
    assert all(row["removal_milestone"] and row["telemetry_source"] for row in rows)
    assert all(importlib.util.find_spec(row["module"]) is not None for row in rows)
    assert all(importlib.util.find_spec(row["replacement"]) is not None for row in rows)


def test_compatibility_usage_is_digest_bound_and_non_authoritative(
    tmp_path: Path,
) -> None:
    record_compatibility_use(
        tmp_path,
        "legacy.route_forest",
        callsite="architecture-test",
        metadata={"revision": 3},
    )
    row = json.loads(
        (tmp_path / ".autoplanner" / "compatibility_usage.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert row["scientific_authority"] is False
    assert row["content_sha256"] == _digest(
        {key: value for key, value in row.items() if key != "content_sha256"}
    )


def test_legacy_tool_registry_and_execution_policy_are_separate() -> None:
    def handler(_state: object, payload: dict) -> dict:
        return {"accepted": True, "echo": payload["value"]}

    handlers = {name: handler for name in LEGACY_LOCAL_TOOL_NAMES}
    registry = bind_legacy_tool_registry(handlers)
    outcome = execute_registered_tool(
        "run_chemenzy",
        {"value": 7},
        object(),
        registry=registry,
        exception_policy=lambda *_args: ("error", {"accepted": False}),
    )

    assert tuple(registry) == LEGACY_LOCAL_TOOL_NAMES
    assert outcome.status == "accepted"
    assert outcome.output["echo"] == 7
