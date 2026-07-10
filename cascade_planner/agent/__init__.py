"""LLM-prior and Codex-agent surfaces for AutoPlanner.

Keep package import dependency-light.  The former eager imports pulled Torch
and the learned failure policy into every ``cascade_planner.agent.*`` import,
including schema validation and the web application.  Public helpers remain
available through lazy attribute loading so existing callers keep working
without making optional ML dependencies a process-wide requirement.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "generate_strategic_prior": ("cascade_planner.agent.prior_generator", "generate_strategic_prior"),
    "critique_route_payload": ("cascade_planner.agent.route_critic", "critique_route_payload"),
    "predict_failure_risk": ("cascade_planner.agent.failure_policy", "predict_failure_risk"),
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
