"""Frozen controller harness exports retained for explicit legacy callers."""
from __future__ import annotations

from typing import Any

from cascade_planner.legacy._exports import load_legacy_export


_EXPORTS = {
    "run_codex_entry_controller": (
        "cascade_planner.legacy.harness_runtime.runner",
        "run_codex_entry_controller",
    ),
    "run_agentic_blackboard_controller": (
        "cascade_planner.legacy.harness_runtime.agentic_blackboard_controller",
        "run_agentic_blackboard_controller",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    value = load_legacy_export(
        name,
        _EXPORTS,
        replacement="cascade_planner.orchestration.retrosynthesis_service",
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
