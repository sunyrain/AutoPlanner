"""Shared contracts for blackboard agent action planning.

This module is intentionally dependency-light so both the agent package and
the harness package can import the same action and safety contract without
creating circular imports.
"""
from __future__ import annotations

from typing import Any


ACTION_BATCH_SCHEMA = "agent_action_batch.v1"
ACTION_SCHEMA = "agent_action.v1"
PLANNER_SOURCE_HINT_SCHEMA = "planner_source_hint.v1"

ALLOWED_AGENT_ACTIONS = {
    "classify_route_objectives",
    "generate_disconnection_hypotheses",
    "rank_analogical_hypotheses",
    "build_failure_critic_report",
    "search_literature",
    "extract_pdf_literature_structures",
    "extract_visual_literature_chain",
    "resolve_literature_structure_task",
    "compile_exact_literature_rows",
    "extract_analogical_reaction_templates",
    "rank_analogical_reaction_templates",
    "apply_analogical_template_to_target",
    "validate_template_application",
    "derive_broad_reaction_template",
    "run_guided_chemenzy",
    "expand_child_target",
    "stitch_parent_route",
    "compile_objective_route_proof",
    "stop_unresolved",
}

FORBIDDEN_RAW_REACTION_KEYS = {
    "rxn",
    "rxn_smiles",
    "rxn_smiles_list",
    "reaction_smiles",
    "raw_reaction",
    "raw_reactions",
    "raw_reaction_candidates",
    "reaction_candidates",
    "route_tree_actions",
    "candidate_actions",
}

PLANNER_SOURCE_HINT_ID_FIELDS = {
    "source_ref",
    "doi",
    "pii",
    "url",
    "local_pdf",
    "local_ref",
    "title",
}


def contains_raw_reaction_payload(value: Any) -> bool:
    """Return True when a payload carries executable/raw reaction content."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RAW_REACTION_KEYS:
                return True
            if contains_raw_reaction_payload(item):
                return True
    if isinstance(value, list):
        return any(contains_raw_reaction_payload(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False


def planner_source_hint_reasons(value: Any) -> list[str]:
    """Validate planner-discovered source metadata hints.

    Hints are allowed to steer later source acquisition. They are not source
    evidence, exact literature rows, route proof, or final-verdict authority.
    """
    reasons: list[str] = []
    if value in (None, ""):
        return reasons
    if not isinstance(value, list):
        return ["planner_source_hints_not_list"]
    if len(value) > 8:
        reasons.append("planner_source_hints_exceeds_limit")
    for idx, raw in enumerate(value):
        if not isinstance(raw, dict):
            reasons.append(f"planner_source_hint_not_object:{idx}")
            continue
        row = dict(raw)
        if row.get("schema_version") != PLANNER_SOURCE_HINT_SCHEMA:
            reasons.append(f"planner_source_hint_invalid_schema:{idx}")
        if row.get("evidence_class") != "planner_source_hint":
            reasons.append(f"planner_source_hint_invalid_evidence_class:{idx}")
        if row.get("allowed_use") != "source_acquisition_hint_only":
            reasons.append(f"planner_source_hint_invalid_allowed_use:{idx}")
        if row.get("no_solved_claim") is not True:
            reasons.append(f"planner_source_hint_missing_no_solved_claim:{idx}")
        if row.get("solved") is True or str(row.get("verdict") or "").lower() == "solved":
            reasons.append(f"planner_source_hint_direct_solved_claim:{idx}")
        if str(row.get("route_status") or "").lower() == "solved":
            reasons.append(f"planner_source_hint_direct_solved_claim:{idx}")
        if not any(str(row.get(field) or "").strip() for field in PLANNER_SOURCE_HINT_ID_FIELDS):
            reasons.append(f"planner_source_hint_missing_source_identifier:{idx}")
        if contains_raw_reaction_payload(row):
            reasons.append(f"planner_source_hint_raw_reaction_injection:{idx}")
    return sorted(set(reasons))
