"""Policy-driven action planning for agentic blackboard runs."""
from __future__ import annotations

import json
from typing import Any

from cascade_planner.harness.schemas import FORBIDDEN_RAW_REACTION_KEYS


ACTION_BATCH_SCHEMA = "agent_action_batch.v1"

ALLOWED_AGENT_ACTIONS = {
    "generate_disconnection_hypotheses",
    "rank_analogical_hypotheses",
    "build_failure_critic_report",
    "search_literature",
    "extract_visual_literature_chain",
    "compile_exact_literature_rows",
    "run_guided_chemenzy",
    "expand_child_target",
    "stitch_parent_route",
    "stop_unresolved",
}


def plan_action_batch(
    blackboard: dict[str, Any],
    *,
    round_index: int,
    max_actions: int = 3,
) -> dict[str, Any]:
    """Select a bounded action batch from current blackboard state."""
    actions: list[dict[str, Any]] = []
    case_id = str(blackboard.get("case_id") or "")
    if not ((blackboard.get("target_profile") or {}).get("valid", True)):
        actions.append(_action(round_index, "stop_unresolved", "invalid target profile", "final unresolved state", "preflight rejected"))
        return _batch(case_id, round_index, actions[:max_actions], mode="deterministic_policy")

    if _parent_proof_accepted(blackboard):
        actions.append(
            _action(
                round_index,
                "stop_unresolved",
                "parent proof already accepted; controller will emit deterministic verdict",
                "no new artifact",
                "controller stops",
            )
        )
        return _batch(case_id, round_index, actions[:max_actions], mode="deterministic_policy")

    if _two_recent_rounds_without_useful_artifact(blackboard):
        actions.append(
            _action(
                round_index,
                "stop_unresolved",
                "two consecutive rounds produced no useful artifact",
                "unresolved stop marker",
                "avoid stale repeated exploration",
            )
        )
        return _batch(case_id, round_index, actions[:max_actions], mode="deterministic_policy")

    if not blackboard.get("target_side_disconnection_hypotheses"):
        actions.append(
            _action(
                round_index,
                "generate_disconnection_hypotheses",
                "target-side bridge tasks are missing",
                "target_side_disconnection_hypotheses.v1",
                "at least one advisory hypothesis and bridge task",
            )
        )

    if _failure_evidence_available(blackboard) and not _action_seen(blackboard, "build_failure_critic_report"):
        actions.append(
            _action(
                round_index,
                "build_failure_critic_report",
                "verifier/plugin failure evidence needs blackboard normalization",
                "failure_critic_report.v1",
                "critic emits bridge task or blocked direction",
            )
        )

    if _needs_literature_bridge(blackboard) and _budget_remaining(blackboard, "scout_calls"):
        actions.append(
            _action(
                round_index,
                "search_literature",
                "blackboard lacks target-proximal literature/source evidence",
                "literature_scout_report.v1",
                "source candidate or extraction recommendation generated",
            )
        )

    if not actions and _source_candidates_available(blackboard) and not _exact_rows_available(blackboard):
        actions.append(
            _action(
                round_index,
                "extract_visual_literature_chain",
                "source candidates exist but exact rows are missing",
                "visual_literature_chain/exact rows artifact",
                "validated source-detail chain or extraction failure reason",
            )
        )

    if not actions and _visual_chain_available(blackboard) and not _exact_rows_available(blackboard):
        actions.append(
            _action(
                round_index,
                "compile_exact_literature_rows",
                "visual chain needs exact source-detail rows for plugin use",
                "compiled exact literature rows",
                "one or more exact row summaries",
            )
        )

    if not actions and _exact_rows_available(blackboard) and not blackboard.get("analogical_hypothesis_ranking"):
        actions.append(
            _action(
                round_index,
                "rank_analogical_hypotheses",
                "exact rows can provide advisory analogical search priorities",
                "analogical_hypothesis_ranking.v1",
                "ranked advisory hypotheses with no solved claim",
            )
        )

    if _can_run_guided_chemenzy(blackboard) and not _round_has_action(actions, "run_guided_chemenzy"):
        actions.append(
            _action(
                round_index,
                "run_guided_chemenzy",
                "bridge tasks and search hints are available for one guided rerun",
                "guided_chemenzy_result plus verifier report",
                "route verifier accepts or returns actionable failure evidence",
            )
        )

    if _can_expand_child_target(blackboard) and not _round_has_action(actions, "expand_child_target"):
        actions.append(
            _action(
                round_index,
                "expand_child_target",
                "advanced terminal/upstream bridge task exists",
                "route_expansion_subgoal_search_result.v1",
                "child target verifier result is recorded without parent solved claim",
            )
        )

    if not actions and _can_stitch_parent_route(blackboard):
        actions.append(
            _action(
                round_index,
                "stitch_parent_route",
                "child route and literature chain need deterministic parent connectivity proof",
                "stitched_parent_route_proof.v1",
                "parent proof accepted or explicit connectivity rejection",
            )
        )

    if not actions:
        actions.append(
            _action(
                round_index,
                "stop_unresolved",
                "no non-stale action satisfies current policy and budget",
                "unresolved stop marker",
                "controller emits unresolved verdict",
            )
        )
    return _batch(case_id, round_index, actions[:max_actions], mode="deterministic_policy")


def validate_action_batch(
    action_batch: dict[str, Any],
    *,
    blackboard: dict[str, Any] | None = None,
    max_actions_per_round: int = 3,
    max_chemenzy_per_round: int = 1,
    max_child_expansions_per_round: int = 2,
    max_literature_sources_per_round: int = 3,
) -> dict[str, Any]:
    reasons: list[str] = []
    batch = dict(action_batch or {})
    board = dict(blackboard or {})
    if batch.get("schema_version") != ACTION_BATCH_SCHEMA:
        reasons.append("invalid_action_batch_schema")
    actions = batch.get("actions")
    if not isinstance(actions, list):
        reasons.append("actions_not_list")
        actions = []
    if len(actions) > max_actions_per_round:
        reasons.append("action_batch_exceeds_max_actions")
    chemenzy_count = 0
    child_count = 0
    source_count = 0
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            reasons.append(f"action_not_object:{idx}")
            continue
        action_type = str(action.get("action_type") or "")
        if action_type not in ALLOWED_AGENT_ACTIONS:
            reasons.append(f"unknown_action:{idx}:{action_type or 'missing'}")
        for field in ("rationale", "expected_artifact", "success_condition"):
            if not str(action.get(field) or "").strip():
                reasons.append(f"action_missing_{field}:{idx}")
        if action.get("verdict") == "solved" or action.get("route_status") == "solved":
            reasons.append("planner_direct_solved_claim")
        if _contains_raw_reaction_payload(action):
            reasons.append("raw_reaction_injection")
        if action_type == "run_guided_chemenzy":
            chemenzy_count += 1
        if action_type == "expand_child_target":
            child_count += 1
        if action_type in {"search_literature", "extract_visual_literature_chain", "compile_exact_literature_rows"}:
            payload = dict(action.get("payload") or {})
            source_count += max(1, int(payload.get("max_sources") or 1))
        if _stale_action_repeated(board, action):
            reasons.append(f"stale_action_repeated:{idx}:{action_type}")
    if chemenzy_count > max_chemenzy_per_round:
        reasons.append("guided_chemenzy_round_budget_exceeded")
    if child_count > max_child_expansions_per_round:
        reasons.append("child_expansion_round_budget_exceeded")
    if source_count > max_literature_sources_per_round:
        reasons.append("literature_source_round_budget_exceeded")
    return {
        "schema_version": "agent_action_batch_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "case_id": str(batch.get("case_id") or ""),
        "action_count": len(actions),
    }


def build_guided_chemenzy_payload_from_blackboard(blackboard: dict[str, Any]) -> dict[str, Any]:
    """Build a guided rerun payload from blackboard constraints and evidence."""
    terminal_blacklist = [
        str(row.get("canonical_smiles") or row.get("smiles") or "")
        for row in blackboard.get("terminal_blacklist") or []
        if isinstance(row, dict)
    ]
    exact_rows = [
        str(row.get("row_id") or row.get("source_template_id") or row.get("template_id") or "")
        for row in ((blackboard.get("literature_evidence") or {}).get("exact_rows") or [])
        if isinstance(row, dict)
    ]
    selected_analogy_ids = [
        str(row.get("hypothesis_id") or "")
        for row in ((blackboard.get("analogical_hypothesis_ranking") or {}).get("selected_hypotheses") or [])
        if isinstance(row, dict)
    ]
    bridge_tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    constraints = dict((blackboard.get("current_belief") or {}).get("constraints") or {})
    return {
        "search_policy": {
            "schema_version": "chem_enzy_search_policy.v1",
            "policy_id": f"{blackboard.get('case_id') or 'case'}_agentic_blackboard_guided",
            "operator_id": "agentic_blackboard_controller",
            "terminal_blacklist": _dedupe(terminal_blacklist),
            "active_bridge_tasks": bridge_tasks,
            "accepted_exact_row_ids": _dedupe(exact_rows),
            "selected_analogical_hypothesis_ids": _dedupe(selected_analogy_ids),
            "source_budget": {
                "require_target_core_retention": bool(constraints.get("target_core_retention_required", True)),
                "max_unexplained_heavy_atom_jump": int(constraints.get("max_unexplained_heavy_atom_jump") or 15),
                "analogy_is_advisory_only": True,
            },
            "preferred_subgoal": {
                "target": dict(blackboard.get("target_profile") or {}),
                "bridge_tasks": bridge_tasks,
            },
            "compiler_metadata": {
                "source": "agentic_blackboard",
                "no_solved_claim": True,
                "requires_verifier": True,
            },
            "mode": "guided",
        }
    }


def _batch(case_id: str, round_index: int, actions: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    return {
        "schema_version": ACTION_BATCH_SCHEMA,
        "case_id": case_id,
        "round_index": int(round_index),
        "mode": mode,
        "actions": actions,
        "semantics": {
            "planner_can_emit_solved": False,
            "raw_reaction_output_allowed": False,
            "deterministic_validator_required": True,
        },
    }


def _action(round_index: int, action_type: str, rationale: str, expected_artifact: str, success_condition: str) -> dict[str, Any]:
    return {
        "schema_version": "agent_action.v1",
        "action_id": f"r{int(round_index)}:{action_type}",
        "action_type": action_type,
        "rationale": rationale,
        "expected_artifact": expected_artifact,
        "success_condition": success_condition,
        "payload": {},
    }


def _parent_proof_accepted(blackboard: dict[str, Any]) -> bool:
    proof = dict(blackboard.get("parent_route_proof") or {})
    return bool(proof.get("accepted") and proof.get("solved"))


def _failure_evidence_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("route_failures") or []) or (blackboard.get("plugin_runtime_diagnostics") or []))


def _needs_literature_bridge(blackboard: dict[str, Any]) -> bool:
    evidence = dict(blackboard.get("literature_evidence") or {})
    if evidence.get("source_candidates"):
        return False
    tasks = [str(row.get("task_type") or "") for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    if any("bridge" in item for item in tasks):
        return True
    return not evidence.get("exact_rows")


def _source_candidates_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("source_candidates"))


def _visual_chain_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("visual_chains"))


def _exact_rows_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("exact_rows"))


def _can_run_guided_chemenzy(blackboard: dict[str, Any]) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("chemenzy_runs") or 0) >= int(budget.get("max_chemenzy_runs") or 1):
        return False
    if _action_seen(blackboard, "run_guided_chemenzy"):
        return False
    return bool(blackboard.get("bridge_tasks") or _exact_rows_available(blackboard) or blackboard.get("analogical_hypothesis_ranking"))


def _can_expand_child_target(blackboard: dict[str, Any]) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("child_target_runs") or 0) >= int(budget.get("max_child_target_runs") or 2):
        return False
    if _action_seen(blackboard, "expand_child_target"):
        return False
    tasks = [str(row.get("task_type") or "") for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    return "upstream_terminal_synthesis" in tasks


def _can_stitch_parent_route(blackboard: dict[str, Any]) -> bool:
    belief = dict(blackboard.get("current_belief") or {})
    return bool(belief.get("child_route_solved") and (blackboard.get("literature_evidence") or {}).get("exact_rows"))


def _round_has_action(actions: list[dict[str, Any]], action_type: str) -> bool:
    return any(str(row.get("action_type") or "") == action_type for row in actions)


def _action_seen(blackboard: dict[str, Any], action_type: str) -> bool:
    return any(str(row.get("action_type") or "") == action_type for row in blackboard.get("action_history") or [] if isinstance(row, dict))


def _two_recent_rounds_without_useful_artifact(blackboard: dict[str, Any]) -> bool:
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    rounds = sorted({int(row.get("round_index") or 0) for row in history if row.get("round_index")}, reverse=True)
    if len(rounds) < 2:
        return False
    for round_index in rounds[:2]:
        rows = [row for row in history if int(row.get("round_index") or 0) == round_index]
        if any(row.get("useful_artifact") for row in rows):
            return False
    return True


def _budget_remaining(blackboard: dict[str, Any], field: str) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    max_field = f"max_{field}"
    fallback = 3 if field == "scout_calls" else 1
    return int(budget.get(field) or 0) < int(budget.get(max_field) or fallback)


def _stale_action_repeated(blackboard: dict[str, Any], action: dict[str, Any]) -> bool:
    signature = _action_signature(action)
    stale_count = 0
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        if row.get("stale") and row.get("action_signature") == signature:
            stale_count += 1
    return stale_count > 1


def _action_signature(action: dict[str, Any]) -> str:
    payload = {key: value for key, value in dict(action.get("payload") or {}).items() if key != "timestamp"}
    return json.dumps({"action_type": action.get("action_type"), "payload": payload}, sort_keys=True, default=str)


def _contains_raw_reaction_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RAW_REACTION_KEYS:
                return True
            if _contains_raw_reaction_payload(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction_payload(item) for item in value)
    return False


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
