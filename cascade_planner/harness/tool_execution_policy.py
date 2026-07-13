"""Execution policy separated from legacy harness tool implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from cascade_planner.harness.tool_registry import LegacyToolHandler


ExceptionPolicy = Callable[
    [str, Any, dict[str, Any], Exception],
    tuple[str, dict[str, Any]],
]


@dataclass(frozen=True, slots=True)
class ToolExecutionOutcome:
    status: str
    output: Mapping[str, Any]


def execute_registered_tool(
    tool_name: str,
    payload: dict[str, Any],
    state: Any,
    *,
    registry: Mapping[str, LegacyToolHandler],
    exception_policy: ExceptionPolicy,
) -> ToolExecutionOutcome:
    handler = registry.get(tool_name)
    if handler is None:
        return ToolExecutionOutcome(
            status="rejected",
            output={"accepted": False, "reasons": ["forbidden_tool"]},
        )
    try:
        output = dict(handler(state, payload) or {})
        status = "accepted" if output.get("accepted", True) else "rejected"
    except Exception as exc:
        status, output = exception_policy(tool_name, state, payload, exc)
    return ToolExecutionOutcome(status=status, output=output)


__all__ = ["ExceptionPolicy", "ToolExecutionOutcome", "execute_registered_tool"]
