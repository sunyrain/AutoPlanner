"""Frozen V3 Codex campaign exports retained for saved-run compatibility."""
from __future__ import annotations

from typing import Any

from cascade_planner.legacy._exports import load_legacy_export


_EXPORTS = {
    "CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA": (
        "cascade_planner.legacy.orchestration_runtime.codex_retrosynthesis",
        "CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA",
    ),
    "CODEX_RETROSYNTHESIS_TEAM_SCHEMA": (
        "cascade_planner.legacy.orchestration_runtime.codex_retrosynthesis",
        "CODEX_RETROSYNTHESIS_TEAM_SCHEMA",
    ),
    "DEFAULT_CHILD_ROLES": (
        "cascade_planner.legacy.orchestration_runtime.codex_retrosynthesis",
        "DEFAULT_CHILD_ROLES",
    ),
    "build_retrosynthesis_coordinator_task": (
        "cascade_planner.legacy.orchestration_runtime.codex_retrosynthesis",
        "build_retrosynthesis_coordinator_task",
    ),
    "run_codex_retrosynthesis_campaign": (
        "cascade_planner.legacy.orchestration_runtime.codex_retrosynthesis",
        "run_codex_retrosynthesis_campaign",
    ),
    "run_codex_retrosynthesis_team": (
        "cascade_planner.legacy.orchestration_runtime.codex_retrosynthesis",
        "run_codex_retrosynthesis_team",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    value = load_legacy_export(
        name,
        _EXPORTS,
        replacement="cascade_planner.orchestration.GlobalCampaignDirector",
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
