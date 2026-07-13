"""Lazy orchestration API with the single-kernel V4 service first."""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DIRECTOR_DISPOSITIONS": (
        "cascade_planner.orchestration.global_campaign_director",
        "DIRECTOR_DISPOSITIONS",
    ),
    "DIRECTOR_MODES": (
        "cascade_planner.orchestration.global_campaign_director",
        "DIRECTOR_MODES",
    ),
    "GLOBAL_CAMPAIGN_PLAN_SCHEMA": (
        "cascade_planner.orchestration.global_campaign_director",
        "GLOBAL_CAMPAIGN_PLAN_SCHEMA",
    ),
    "RetrosynthesisCampaignService": (
        "cascade_planner.orchestration.retrosynthesis_service",
        "RetrosynthesisCampaignService",
    ),
    "DirectorConfig": (
        "cascade_planner.orchestration.global_campaign_director",
        "DirectorConfig",
    ),
    "DirectorOutcome": (
        "cascade_planner.orchestration.global_campaign_director",
        "DirectorOutcome",
    ),
    "GlobalCampaignDirector": (
        "cascade_planner.orchestration.global_campaign_director",
        "GlobalCampaignDirector",
    ),
    "GlobalCampaignDirectorError": (
        "cascade_planner.orchestration.global_campaign_director",
        "GlobalCampaignDirectorError",
    ),
    "GlobalCampaignPlan": (
        "cascade_planner.orchestration.global_campaign_director",
        "GlobalCampaignPlan",
    ),
    "GlobalCampaignPlanValidationError": (
        "cascade_planner.orchestration.global_campaign_director",
        "GlobalCampaignPlanValidationError",
    ),
    "ReplayDirectorRunner": (
        "cascade_planner.orchestration.global_campaign_director",
        "ReplayDirectorRunner",
    ),
    "director_trigger_reasons": (
        "cascade_planner.orchestration.global_campaign_director",
        "director_trigger_reasons",
    ),
    "validate_global_campaign_plan": (
        "cascade_planner.orchestration.global_campaign_director",
        "validate_global_campaign_plan",
    ),
    # Frozen V3 compatibility exports.
    "CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA": (
        "cascade_planner.orchestration.codex_retrosynthesis",
        "CODEX_RETROSYNTHESIS_CAMPAIGN_SCHEMA",
    ),
    "CODEX_RETROSYNTHESIS_TEAM_SCHEMA": (
        "cascade_planner.orchestration.codex_retrosynthesis",
        "CODEX_RETROSYNTHESIS_TEAM_SCHEMA",
    ),
    "DEFAULT_CHILD_ROLES": (
        "cascade_planner.orchestration.codex_retrosynthesis",
        "DEFAULT_CHILD_ROLES",
    ),
    "build_retrosynthesis_coordinator_task": (
        "cascade_planner.orchestration.codex_retrosynthesis",
        "build_retrosynthesis_coordinator_task",
    ),
    "run_codex_retrosynthesis_campaign": (
        "cascade_planner.orchestration.codex_retrosynthesis",
        "run_codex_retrosynthesis_campaign",
    ),
    "run_codex_retrosynthesis_team": (
        "cascade_planner.orchestration.codex_retrosynthesis",
        "run_codex_retrosynthesis_team",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
