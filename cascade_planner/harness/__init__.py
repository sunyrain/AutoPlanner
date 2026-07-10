"""Codex-entry controller harness for AutoPlanner.

The package initializer intentionally avoids importing the full controller.
Most harness modules are reusable compilers or schemas; importing one of them
must not initialize RDKit, Torch, ChemEnzy, and the web runtime as a side
effect.  These two compatibility exports are resolved only when requested.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "run_codex_entry_controller": ("cascade_planner.harness.runner", "run_codex_entry_controller"),
    "run_agentic_blackboard_controller": (
        "cascade_planner.harness.agentic_blackboard_controller",
        "run_agentic_blackboard_controller",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
