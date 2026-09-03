"""Explicit, provider-neutral extensions for route-tree execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RouteTreeExtensions:
    """Optional adapters injected by an explicit benchmark or research worker."""

    controller_factory: Callable[[], Any] | None = None
    source_gate_factory: Callable[[], Any] | None = None
    action_value_advisor_factory: Callable[[], Any] | None = None
    action_value_advisor_weight: float = 0.0
    candidate_appender: Callable[..., list[Any]] | None = None
