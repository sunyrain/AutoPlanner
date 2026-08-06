from __future__ import annotations

from cascade_planner.legacy.harness_runtime.tool_execution_policy import (
    execute_registered_tool,
)
from cascade_planner.legacy.harness_runtime.tool_registry import (
    LEGACY_LOCAL_TOOL_NAMES,
    bind_legacy_tool_registry,
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
