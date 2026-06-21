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
    "extract_pdf_literature_structures",
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
    exhaust_round_budget: bool = False,
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

    if _two_recent_rounds_without_useful_artifact(blackboard) and not exhaust_round_budget:
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

    mode = "deterministic_policy_budget_exhaustive" if exhaust_round_budget else "deterministic_policy"

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

    if (
        not actions
        and _source_candidates_available(blackboard)
        and _source_candidates_include_local_pdf(blackboard)
        and not _pdf_structure_evidence_available(blackboard)
    ):
        actions.append(
            _action(
                round_index,
                "extract_pdf_literature_structures",
                "local PDF source is available and must be converted into current-run visual evidence",
                "literature_pdf_structure_evidence.v1",
                "rendered pages or indexed images are available for visual extraction",
            )
        )

    if (
        not actions
        and _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _visual_gap_repair_needed(blackboard)
        and _budget_remaining(blackboard, "visual_calls")
    ):
        actions.append(
            _action(
                round_index,
                "extract_visual_literature_chain",
                "prior visual extraction left target-relevant source-detail gaps; repair those gaps before compiling exact rows",
                "visual_literature_chain/exact rows artifact",
                "missing source-detail labels are either filled or explicitly rejected",
                _focused_visual_repair_payload(blackboard),
            )
        )

    if (
        not actions
        and _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _action_count(blackboard, "compile_exact_literature_rows") < _action_count(blackboard, "extract_visual_literature_chain")
    ):
        actions.append(
            _action(
                round_index,
                "compile_exact_literature_rows",
                "visual chain needs exact source-detail rows for plugin use",
                "compiled exact literature rows",
                "one or more exact row summaries",
                {"compile_attempt": _action_count(blackboard, "compile_exact_literature_rows") + 1},
            )
        )

    if (
        not actions
        and _source_candidates_available(blackboard)
        and _source_candidates_visual_ready(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and not _action_seen(blackboard, "extract_visual_literature_chain")
    ):
        actions.append(
            _action(
                round_index,
                "extract_visual_literature_chain",
                "source candidates exist but exact rows are missing",
                "visual_literature_chain/exact rows artifact",
                "validated source-detail chain or extraction failure reason",
                _visual_extraction_payload_from_blackboard(blackboard),
            )
        )

    if (
        len(actions) < max_actions
        and _hypotheses_available(blackboard)
        and not blackboard.get("analogical_hypothesis_ranking")
        and not _round_has_action(actions, "rank_analogical_hypotheses")
    ):
        actions.append(
            _action(
                round_index,
                "rank_analogical_hypotheses",
                "advisory target-side hypotheses can be ranked before rerun selection",
                "analogical_hypothesis_ranking.v1",
                "ranked advisory hypotheses with no solved claim",
            )
        )

    if (
        _can_stitch_parent_route(blackboard)
        and not _round_has_action(actions, "stitch_parent_route")
        and not _round_has_action(actions, "expand_child_target")
    ):
        actions.append(
            _action(
                round_index,
                "stitch_parent_route",
                "guided/child/literature artifacts need deterministic parent connectivity proof",
                "stitched_parent_route_proof.v1",
                "parent proof accepted or explicit connectivity rejection",
                _stitch_retry_payload(blackboard),
            )
        )

    if (
        _can_expand_child_target(blackboard)
        and not _round_has_action(actions, "expand_child_target")
        and not _round_has_action(actions, "stitch_parent_route")
    ):
        actions.append(
            _action(
                round_index,
                "expand_child_target",
                "an exact literature terminal or upstream bridge task exists",
                "route_expansion_subgoal_search_result.v1",
                "child target verifier result is recorded without parent solved claim",
                _child_expansion_payload(blackboard),
            )
        )

    if (
        _can_run_guided_chemenzy(blackboard)
        and not _round_has_action(actions, "run_guided_chemenzy")
        and not _literature_extraction_pending(blackboard, actions)
        and not _literature_terminal_expansion_pending(blackboard)
        and not _round_has_action(actions, "stitch_parent_route")
    ):
        actions.append(
            _action(
                round_index,
                "run_guided_chemenzy",
                "bridge tasks and search hints are available for one guided rerun",
                "guided_chemenzy_result plus verifier report",
                "route verifier accepts or returns actionable failure evidence",
                _guided_retry_payload(blackboard),
            )
        )

    if not actions and exhaust_round_budget and _failure_evidence_available(blackboard):
        actions.append(
            _action(
                round_index,
                "build_failure_critic_report",
                "budget-exhaustive policy found no new executable branch, so it records a final failure-state audit instead of an early stop",
                "failure_critic_report.v1",
                "critic emits an auditable unresolved-state update",
                {"audit_attempt": _action_count(blackboard, "build_failure_critic_report") + 1},
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
    return _batch(case_id, round_index, actions[:max_actions], mode=mode)


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
        if action_type in {
            "search_literature",
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "compile_exact_literature_rows",
        }:
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
    case_id = str(blackboard.get("case_id") or "case")
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
    evidence_refs = _guided_evidence_refs(
        blackboard=blackboard,
        bridge_tasks=bridge_tasks,
        exact_rows=exact_rows,
        selected_analogy_ids=selected_analogy_ids,
    )
    return {
        "search_policy": {
            "schema_version": "chem_enzy_search_policy.v1",
            "policy_id": f"{case_id}_agentic_blackboard_guided",
            "operator_id": "agentic_blackboard_controller",
            "case_id": case_id,
            "evidence_refs": evidence_refs,
            "terminal_blacklist": _dedupe(terminal_blacklist),
            "anchor_whitelist": [],
            "active_bridge_tasks": bridge_tasks,
            "accepted_exact_row_ids": _dedupe(exact_rows),
            "selected_analogical_hypothesis_ids": _dedupe(selected_analogy_ids),
            "source_budget": {
                "require_target_core_retention": bool(constraints.get("target_core_retention_required", True)),
                "max_unexplained_heavy_atom_jump": int(constraints.get("max_unexplained_heavy_atom_jump") or 15),
                "analogy_is_advisory_only": True,
                "preferred_reaction_classes": ["target_proximal_bridge_search"],
            },
            "preferred_subgoal": {
                "target": dict(blackboard.get("target_profile") or {}),
                "bridge_tasks": bridge_tasks,
            },
            "rerun_reason": "agentic_blackboard_bridge_tasks_available",
            "budget": {
                "max_reruns": 1,
                "max_iterations": 50,
                "max_depth": 15,
                "expansion_topk": 100,
            },
            "compiler_metadata": {
                "source": "agentic_blackboard",
                "no_solved_claim": True,
                "requires_verifier": True,
            },
            "mode": "guided",
        }
    }


def _guided_evidence_refs(
    *,
    blackboard: dict[str, Any],
    bridge_tasks: list[dict[str, Any]],
    exact_rows: list[str],
    selected_analogy_ids: list[str],
) -> list[str]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    refs = [str(item) for item in evidence.get("source_refs") or [] if str(item or "").strip()]
    refs.extend(str(item) for item in exact_rows if str(item or "").strip())
    refs.extend(str(item) for item in selected_analogy_ids if str(item or "").strip())
    refs.extend(str(row.get("task_id") or "") for row in bridge_tasks if str(row.get("task_id") or "").strip())
    refs.extend(
        str(row.get("artifact_ref") or row.get("source_pdf_path") or "")
        for row in evidence.get("pdf_structure_evidence") or []
        if isinstance(row, dict)
    )
    artifact_refs = dict(blackboard.get("artifact_refs") or {})
    for key, value in artifact_refs.items():
        if any(token in str(key) for token in ("literature", "disconnection", "analogical")):
            refs.append(str(value))
    if not refs:
        refs.append(f"{blackboard.get('case_id') or 'case'}:agentic_blackboard_state")
    return _dedupe(refs)


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


def _action(
    round_index: int,
    action_type: str,
    rationale: str,
    expected_artifact: str,
    success_condition: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "agent_action.v1",
        "action_id": f"r{int(round_index)}:{action_type}",
        "action_type": action_type,
        "rationale": rationale,
        "expected_artifact": expected_artifact,
        "success_condition": success_condition,
        "payload": dict(payload or {}),
    }


def _parent_proof_accepted(blackboard: dict[str, Any]) -> bool:
    proof = dict(blackboard.get("parent_route_proof") or {})
    return bool(proof.get("accepted") and proof.get("solved"))


def _failure_evidence_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("route_failures") or []) or (blackboard.get("plugin_runtime_diagnostics") or []))


def _needs_literature_bridge(blackboard: dict[str, Any]) -> bool:
    evidence = dict(blackboard.get("literature_evidence") or {})
    if evidence.get("source_candidates"):
        return not _source_candidates_have_real_source(blackboard)
    tasks = [str(row.get("task_type") or "") for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    if any("bridge" in item for item in tasks):
        return True
    return not evidence.get("exact_rows")


def _source_candidates_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("source_candidates"))


def _source_candidates_include_local_pdf(blackboard: dict[str, Any]) -> bool:
    return any(
        bool(str(row.get("local_pdf") or "").strip())
        for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(row, dict)
    )


def _source_candidates_have_real_source(blackboard: dict[str, Any]) -> bool:
    return any(
        _candidate_has_real_source(row)
        for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(row, dict)
    )


def _candidate_has_real_source(row: dict[str, Any]) -> bool:
    if bool(row.get("placeholder_only")):
        return False
    if str(row.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    return bool(str(row.get("doi") or row.get("url") or row.get("local_pdf") or "").strip())


def _source_candidates_visual_ready(blackboard: dict[str, Any]) -> bool:
    if _source_candidates_include_local_pdf(blackboard):
        return True
    return _pdf_structure_evidence_available(blackboard)


def _pdf_structure_evidence_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("pdf_structure_evidence"))


def _visual_chain_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("visual_chains"))


def _literature_extraction_pending(blackboard: dict[str, Any], actions: list[dict[str, Any]]) -> bool:
    if _round_has_any_action(
        actions,
        {
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "compile_exact_literature_rows",
        },
    ):
        return True
    if _source_candidates_include_local_pdf(blackboard) and not _pdf_structure_evidence_available(blackboard):
        return True
    if (
        _source_candidates_available(blackboard)
        and _pdf_structure_evidence_available(blackboard)
        and not _visual_chain_available(blackboard)
        and not _action_seen(blackboard, "extract_visual_literature_chain")
    ):
        return True
    if (
        _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _visual_gap_repair_needed(blackboard)
        and _budget_remaining(blackboard, "visual_calls")
    ):
        return True
    if (
        _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _action_count(blackboard, "compile_exact_literature_rows") < _action_count(blackboard, "extract_visual_literature_chain")
    ):
        return True
    return False


def _visual_extraction_payload_from_blackboard(blackboard: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(row, dict)
    ]
    payload: dict[str, Any] = {}
    if candidates:
        first = candidates[0]
        if first.get("source_ref"):
            payload["source_ref"] = str(first.get("source_ref") or "")
        if first.get("title"):
            payload["source_title"] = str(first.get("title") or "")
        if first.get("route_sequence_hint"):
            payload["route_sequence_hint"] = str(first.get("route_sequence_hint") or "")
    labels = _expected_labels_from_source_candidates(blackboard)
    if labels:
        payload["expected_labels"] = labels
        payload["route_sequence_hint"] = " ".join(
            part
            for part in [
                str(payload.get("route_sequence_hint") or ""),
                "Extract a contiguous source-detail chain covering the expected labels when visible. "
                "If any label cannot be converted into RDKit-valid SMILES from current images, record it in extraction_gaps.",
            ]
            if part
        )
    return payload


def _focused_visual_repair_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    gap_labels = _visual_gap_labels(blackboard)
    expected = _expected_labels_from_source_candidates(blackboard)
    labels = _dedupe([*gap_labels, *expected])
    payload = _visual_extraction_payload_from_blackboard(blackboard)
    if labels:
        payload["expected_labels"] = labels
    payload["focused_gap_repair"] = True
    payload["repair_attempt"] = _focused_visual_repair_attempts(blackboard) + 1
    payload["route_sequence_hint"] = (
        "Focused repair: re-inspect the current PDF images for the missing source-detail labels "
        f"{', '.join(gap_labels) or 'recorded extraction gaps'}. "
        "For labels already covered by valid structures, repair missing condition_candidate fields from visible scheme/table text. "
        "Prefer fewer high-confidence RDKit-valid exact steps over guessed structures or conditions; keep unresolved labels/conditions in extraction_gaps."
    )
    return payload


def _expected_labels_from_source_candidates(blackboard: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []:
        if not isinstance(row, dict):
            continue
        labels.extend(str(item) for item in row.get("expected_scheme_or_compound_labels") or [] if str(item or "").strip())
    return _dedupe(labels)


def _visual_gap_repair_needed(blackboard: dict[str, Any]) -> bool:
    return bool(_visual_gap_labels(blackboard))


def _visual_gap_labels(blackboard: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        labels.extend(str(item) for item in row.get("missing_expected_labels") or [] if str(item or "").strip())
        labels.extend(str(item) for item in row.get("gap_labels") or [] if str(item or "").strip())
        for gap in row.get("extraction_gaps") or []:
            if isinstance(gap, dict):
                if _nonblocking_visual_gap(gap):
                    continue
                raw_labels = gap.get("labels") if isinstance(gap.get("labels"), list) else [gap.get("label")]
                labels.extend(str(item) for item in raw_labels if str(item or "").strip())
    return _dedupe(labels)


def _nonblocking_visual_gap(gap: dict[str, Any]) -> bool:
    gap_type = str(gap.get("gap_type") or gap.get("type") or "").strip().lower()
    return gap_type in {
        "stereochemical_precision",
        "stereochemistry_precision",
        "stereo_precision",
        "stereochemical_ambiguity",
        "stereochemistry_ambiguity",
        "stereo_ambiguity",
        "diastereomeric_ambiguity",
    }


def _focused_visual_repair_seen(blackboard: dict[str, Any]) -> bool:
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("action_type") or "") != "extract_visual_literature_chain":
            continue
        signature = str(row.get("action_signature") or "")
        if "focused_gap_repair" in signature:
            return True
    return False


def _exact_rows_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("exact_rows"))


def _hypotheses_available(blackboard: dict[str, Any]) -> bool:
    return bool(
        (blackboard.get("target_side_disconnection_hypotheses") or {}).get("hypotheses")
        or blackboard.get("analogical_hypotheses")
    )


def _can_run_guided_chemenzy(blackboard: dict[str, Any]) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("chemenzy_runs") or 0) >= int(budget.get("max_chemenzy_runs") or 1):
        return False
    if _can_stitch_parent_route(blackboard):
        return False
    return bool(blackboard.get("bridge_tasks") or _exact_rows_available(blackboard) or blackboard.get("analogical_hypothesis_ranking"))


def _can_expand_child_target(blackboard: dict[str, Any]) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("child_target_runs") or 0) >= int(budget.get("max_child_target_runs") or 2):
        return False
    if _can_stitch_parent_route(blackboard):
        return False
    if _literature_terminal_candidates(blackboard):
        return True
    tasks = [str(row.get("task_type") or "") for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    return "upstream_terminal_synthesis" in tasks


def _can_stitch_parent_route(blackboard: dict[str, Any]) -> bool:
    belief = dict(blackboard.get("current_belief") or {})
    evidence = dict(blackboard.get("literature_evidence") or {})
    stitch_count = _action_count(blackboard, "stitch_parent_route")
    child_count = _action_count(blackboard, "expand_child_target")
    if belief.get("child_route_solved") and evidence.get("exact_rows") and stitch_count < max(1, child_count):
        return True
    if evidence.get("exact_rows") and child_count and stitch_count < child_count:
        return True
    parent_artifacts_attempted = _action_seen(blackboard, "run_guided_chemenzy") or _action_seen(blackboard, "expand_child_target")
    return bool(
        stitch_count == 0
        and parent_artifacts_attempted
        and (evidence.get("visual_chains") or evidence.get("exact_rows") or blackboard.get("route_failures"))
    )


def _literature_terminal_expansion_pending(blackboard: dict[str, Any]) -> bool:
    return bool(_literature_terminal_candidates(blackboard) and not _action_seen(blackboard, "expand_child_target"))


def _literature_terminal_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("terminal_candidates") or []
        if isinstance(row, dict) and str(row.get("smiles") or "").strip()
    ]


def _round_has_action(actions: list[dict[str, Any]], action_type: str) -> bool:
    return any(str(row.get("action_type") or "") == action_type for row in actions)


def _round_has_any_action(actions: list[dict[str, Any]], action_types: set[str]) -> bool:
    return any(str(row.get("action_type") or "") in action_types for row in actions)


def _action_seen(blackboard: dict[str, Any], action_type: str) -> bool:
    return any(str(row.get("action_type") or "") == action_type for row in blackboard.get("action_history") or [] if isinstance(row, dict))


def _action_count(blackboard: dict[str, Any], action_type: str) -> int:
    return sum(
        1
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict) and str(row.get("action_type") or "") == action_type
    )


def _focused_visual_repair_attempts(blackboard: dict[str, Any]) -> int:
    count = 0
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("action_type") or "") != "extract_visual_literature_chain":
            continue
        if "focused_gap_repair" in str(row.get("action_signature") or ""):
            count += 1
    return count


def _exact_row_count(blackboard: dict[str, Any]) -> int:
    return len((blackboard.get("literature_evidence") or {}).get("exact_rows") or [])


def _expected_step_count(blackboard: dict[str, Any]) -> int:
    labels = _expected_labels_from_source_candidates(blackboard)
    return max(0, len(labels) - 1)


def _exact_rows_incomplete(blackboard: dict[str, Any]) -> bool:
    expected = _expected_step_count(blackboard)
    if expected <= 0:
        return False
    return _exact_row_count(blackboard) < expected


def _guided_retry_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    attempt = _action_count(blackboard, "run_guided_chemenzy") + 1
    failures = _blackboard_failure_reasons(blackboard)
    payload: dict[str, Any] = {
        "rerun_attempt": attempt,
        "failure_mode_focus": failures[:6],
    }
    if attempt > 1:
        payload.update(
            {
                "search_preset": "thorough",
                "max_steps": 20,
                "chem_enzy_iterations": min(200, 75 + 25 * attempt),
                "chem_enzy_expansion_topk": min(300, 120 + 30 * attempt),
            }
        )
    return payload


def _blackboard_failure_reasons(blackboard: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for row in blackboard.get("route_failures") or []:
        if isinstance(row, dict):
            reason = str(row.get("reason") or "").strip()
            if reason:
                reasons.append(reason)
        elif str(row or "").strip():
            reasons.append(str(row).strip())
    for row in blackboard.get("plugin_runtime_diagnostics") or []:
        if not isinstance(row, dict):
            continue
        reasons.extend(str(item) for item in row.get("reasons") or [] if str(item or "").strip())
    return _dedupe(reasons)


def _child_expansion_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    attempt = _action_count(blackboard, "expand_child_target") + 1
    payload: dict[str, Any] = {
        "expansion_attempt": attempt,
        "target_offset": max(0, (attempt - 1) * 2),
        "max_targets": 2,
        "search_preset": "thorough",
        "chem_enzy_iterations": min(200, 75 + 25 * attempt),
        "chem_enzy_expansion_topk": min(300, 120 + 30 * attempt),
    }
    terminals = _literature_terminal_candidates(blackboard)
    if terminals:
        payload["target_offset"] = 0
        payload["max_targets"] = min(2, len(terminals))
        payload["subgoal_targets"] = [
            {
                "name": str(row.get("name") or "source detail literature terminal"),
                "smiles": str(row.get("smiles") or ""),
                "exact_target_override": True,
                "target_equivalence_audit_required": True,
                "chem_enzy_search_policy": {
                    "schema_version": "chem_enzy_search_policy.v1",
                    "policy_id": f"{blackboard.get('case_id') or 'case'}_literature_terminal_{idx}_policy",
                    "operator_id": "agentic_blackboard_controller",
                    "case_id": str(blackboard.get("case_id") or ""),
                    "evidence_refs": [str(row.get("source_ref") or row.get("terminal_id") or "")],
                    "anchor_whitelist": [str(row.get("smiles") or "")],
                    "preferred_subgoal": {
                        "schema_version": "source_detail_literature_terminal_subgoal.v1",
                        "preferred_subgoals": [str(row.get("name") or ""), str(row.get("smiles") or "")],
                        "terminal_candidate": dict(row),
                    },
                    "source_budget": {
                        "preferred_reaction_classes": ["steroid_semisynthesis", "source_detail_terminal_upstream_expansion"],
                        "exact_literature_terminal": True,
                    },
                    "rerun_reason": "explore upstream route to exact source-detail literature terminal",
                    "mode": "guided",
                    "compiler_metadata": {
                        "compiler_schema": "agentic_blackboard_literature_terminal_child_target.v1",
                        "not_raw_reaction_injection": True,
                    },
                },
            }
            for idx, row in enumerate(terminals[:2], start=1)
        ]
    return payload


def _stitch_retry_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    return {
        "stitch_attempt": _action_count(blackboard, "stitch_parent_route") + 1,
        "exact_row_count_at_attempt": _exact_row_count(blackboard),
        "child_attempt_count_at_attempt": _action_count(blackboard, "expand_child_target"),
        "guided_attempt_count_at_attempt": _action_count(blackboard, "run_guided_chemenzy"),
    }


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
