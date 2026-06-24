"""Policy-driven action planning for agentic blackboard runs."""
from __future__ import annotations

import json
from typing import Any

from cascade_planner.agent.action_contracts import (
    ACTION_BATCH_SCHEMA,
    ALLOWED_AGENT_ACTIONS,
    contains_raw_reaction_payload,
    planner_source_hint_reasons,
)
from cascade_planner.agent.chem_enzy_policy import validate_chem_enzy_search_policy


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

    _append_next_action_bias_actions(actions, blackboard, round_index=round_index, max_actions=max_actions)

    if (
        not actions
        and _awaiting_local_pdf_proxy_download(blackboard)
        and not _source_candidates_include_local_pdf(blackboard)
    ):
        return _batch(
            case_id,
            round_index,
            [
                _action(
                    round_index,
                    "stop_unresolved",
                    "real source metadata was found but the agent-readable PDF is still queued for local authorized retrieval",
                    "external PDF wait marker",
                    "follow-up task records the DOI/URL to download before extraction resumes",
                    {"wait_state": "local_pdf_proxy_requested", "no_solved_claim": True},
                )
            ],
            mode=mode,
        )

    if (
        not actions
        and _objective_evidence_validation_needed(blackboard)
        and _budget_remaining(blackboard, "scout_calls")
        and not _round_has_action(actions, "search_literature")
    ):
        actions.append(
            _action(
                round_index,
                "search_literature",
                "route-objective ranking identified a non-stock endpoint; validate endpoint evidence before recursive small-molecule expansion",
                "route_objective_endpoint_scout_report.v1",
                "source candidate confirms or rejects the selected route objective endpoint",
                _literature_search_payload(blackboard, intent="route_objective_endpoint_validation"),
            )
        )

    if (
        _needs_literature_bridge(blackboard)
        and _budget_remaining(blackboard, "scout_calls")
        and not _stale_literature_search_repeated(blackboard)
        and not _round_has_action(actions, "search_literature")
    ):
        actions.append(
            _action(
                round_index,
                "search_literature",
                "blackboard lacks target-proximal literature/source evidence",
                "literature_scout_report.v1",
                "source candidate or extraction recommendation generated",
                _literature_search_payload(blackboard, intent="target_proximal_source_discovery"),
            )
        )

    next_pdf_source = _next_local_pdf_source_for_pdf_extraction(blackboard)
    if not actions and next_pdf_source and _budget_remaining(blackboard, "visual_calls"):
        actions.append(
            _action(
                round_index,
                "extract_pdf_literature_structures",
                "local PDF source is available and must be converted into current-run visual evidence",
                "literature_pdf_structure_evidence.v1",
                "rendered pages or indexed images are available for visual extraction",
                _source_candidate_payload(next_pdf_source),
            )
        )

    if (
        not actions
        and _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _visual_gap_repair_needed(blackboard)
        and _visual_gap_repair_budget_remaining(blackboard)
        and (_condition_gap_repair_needed(blackboard) or not _uncompiled_visual_steps_available(blackboard))
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
        and _uncompiled_visual_steps_available(blackboard)
    ):
        actions.append(
            _action(
                round_index,
                "compile_exact_literature_rows",
                "visual chain needs exact source-detail rows for plugin use",
                "compiled exact literature rows",
                "one or more exact row summaries",
                _compile_exact_rows_payload(blackboard),
            )
        )

    next_visual_source = _next_local_pdf_source_for_visual_extraction(blackboard)
    if (
        not actions
        and next_visual_source
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _budget_remaining(blackboard, "visual_calls")
    ):
        actions.append(
            _action(
                round_index,
                "extract_visual_literature_chain",
                "source candidates exist but exact rows are missing",
                "visual_literature_chain/exact rows artifact",
                "validated source-detail chain or extraction failure reason",
                _visual_extraction_payload_from_blackboard(blackboard, source_candidate=next_visual_source),
            )
        )

    next_structure_task = _next_structure_resolution_task_for_local_resolve(blackboard)
    if (
        not actions
        and next_structure_task
        and _budget_remaining(blackboard, "visual_calls")
    ):
        actions.append(
            _action(
                round_index,
                "resolve_literature_structure_task",
                "visual extraction identified a specific compound label but not a machine-readable structure; resolve that label from current source evidence before broader scouting",
                "literature_structure_resolution_result.v1",
                "the label is converted into a source-grounded RDKit-valid structure candidate or recorded as unresolved",
                _structure_resolution_task_payload(blackboard, next_structure_task),
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
        not actions
        and _structure_resolution_scout_needed(blackboard)
        and _budget_remaining(blackboard, "scout_calls")
    ):
        actions.append(
            _action(
                round_index,
                "search_literature",
                "visual extraction produced structure gaps; search supplementary/full-text/name-to-structure sources before relying on analogy",
                "structure_resolution_source_scout_report.v1",
                "new structure-resolution source candidate or explicit unresolved task evidence",
                _structure_resolution_scout_payload(blackboard),
            )
        )

    if (
        len(actions) < max_actions
        and _source_detail_one_step_rows_ready(blackboard)
        and _can_run_guided_chemenzy(blackboard)
        and not _round_has_action(actions, "run_guided_chemenzy")
        and not _literature_extraction_pending(blackboard, actions)
        and not _round_has_action(actions, "stitch_parent_route")
    ):
        actions.append(
            _action(
                round_index,
                "run_guided_chemenzy",
                "compiled source-detail one-step rows are available; try a guarded parent-side guided rerun before spawning child targets",
                "guided_chemenzy_result plus verifier report",
                "route verifier accepts or returns actionable failure evidence",
                _guided_retry_payload(blackboard),
            )
        )

    if (
        _analogical_templates_enabled(blackboard)
        and len(actions) < max_actions
        and _broad_template_derivation_ready(blackboard)
        and not _round_has_action(actions, "derive_broad_reaction_template")
        and not _deterministic_route_action_ready(blackboard)
    ):
        actions.append(
            _action(
                round_index,
                "derive_broad_reaction_template",
                "objective and target-side hypotheses can be converted into broad, non-exact transform templates for guided search",
                "broad_transform_template_report.v1",
                "one or more advisory broad templates are available without solved claim",
                _broad_template_payload(blackboard),
            )
        )

    if (
        _analogical_templates_enabled(blackboard)
        and len(actions) < max_actions
        and _analogical_template_sources_available(blackboard)
        and not _exact_literature_segment_usable(blackboard)
        and not blackboard.get("analogical_templates")
        and not _failed_action_seen(blackboard, "extract_analogical_reaction_templates")
        and not _round_has_action(actions, "extract_analogical_reaction_templates")
        and not _literature_extraction_pending(blackboard, actions)
        and not _deterministic_route_action_ready(blackboard)
    ):
        actions.append(
            _action(
                round_index,
                "extract_analogical_reaction_templates",
                "exact target-side evidence is incomplete, but analog or family precedent can be converted into guarded template hypotheses",
                "analogical_reaction_template_report.v1",
                "one or more no-solved-claim analogical templates with scope gaps",
                _analogical_template_payload(blackboard, action_type="extract_analogical_reaction_templates"),
            )
        )

    if (
        _analogical_templates_enabled(blackboard)
        and len(actions) < max_actions
        and blackboard.get("analogical_templates")
        and not _exact_literature_segment_usable(blackboard)
        and not blackboard.get("analogical_template_ranking")
        and not _round_has_action(actions, "rank_analogical_reaction_templates")
        and not _deterministic_route_action_ready(blackboard)
    ):
        actions.append(
            _action(
                round_index,
                "rank_analogical_reaction_templates",
                "analogical templates need target-specific ranking before any target application",
                "analogical_reaction_template_ranking.v1",
                "selected templates are ranked with required verification and no solved claim",
                _analogical_template_payload(blackboard, action_type="rank_analogical_reaction_templates"),
            )
        )

    if (
        _analogical_templates_enabled(blackboard)
        and len(actions) < max_actions
        and _ranked_analogical_templates_available(blackboard)
        and not _exact_literature_segment_usable(blackboard)
        and not _template_applications_available(blackboard)
        and _budget_remaining(blackboard, "template_application_actions")
        and not _round_has_action(actions, "apply_analogical_template_to_target")
        and not _deterministic_route_action_ready(blackboard)
    ):
        actions.append(
            _action(
                round_index,
                "apply_analogical_template_to_target",
                "ranked analogical templates can now be tried against the target/frontier through deterministic retron matching",
                "analogical_template_application_report.v1",
                "template applications are accepted or rejected with explicit scope and reconstruction reasons",
                _analogical_template_payload(blackboard, action_type="apply_analogical_template_to_target"),
            )
        )

    if (
        _analogical_templates_enabled(blackboard)
        and len(actions) < max_actions
        and _template_applications_need_validation(blackboard)
        and not _exact_literature_segment_usable(blackboard)
        and _budget_remaining(blackboard, "template_application_actions")
        and not _round_has_action(actions, "validate_template_application")
        and not _deterministic_route_action_ready(blackboard)
    ):
        actions.append(
            _action(
                round_index,
                "validate_template_application",
                "accepted template applications must be compiled through deterministic downstream gates before guided search can consume them",
                "analogical_template_application_validation.v1",
                "validated one-step rows are available as guided-search material or rejected with reasons",
                _analogical_template_payload(blackboard, action_type="validate_template_application"),
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
        len(actions) < max_actions
        and not blackboard.get("route_proof_bundle")
        and _route_objective_proof_ready(blackboard)
        and not _round_has_action(actions, "compile_objective_route_proof")
    ):
        actions.append(
            _action(
                round_index,
                "compile_objective_route_proof",
                "objective-specific evidence should be audited before closeout",
                "route_proof_bundle.v1",
                "objective proof bundle records solved/plausible/unresolved status",
                {"proof_scope": "objective_specific", "no_solved_claim": True},
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
                "an exact literature terminal or hypothesis-only same-core precursor subgoal exists",
                "route_expansion_subgoal_search_result.v1",
                "child target verifier result is recorded without parent solved claim",
                _child_expansion_payload(blackboard),
            )
        )

    if (
        _can_run_guided_chemenzy(blackboard)
        and not _round_has_action(actions, "run_guided_chemenzy")
        and not _literature_extraction_pending(blackboard, actions)
        and not _analogical_template_work_pending(blackboard, actions)
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

    if (
        not actions
        and exhaust_round_budget
        and _failure_evidence_available(blackboard)
        and _new_failure_evidence_since_last_critic(blackboard)
    ):
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
    semantics = batch.get("semantics") or {}
    if isinstance(semantics, dict):
        if bool(semantics.get("planner_can_emit_solved")):
            reasons.append("planner_semantics_allow_solved_claim")
        if bool(semantics.get("raw_reaction_output_allowed")):
            reasons.append("planner_semantics_allow_raw_reaction_output")
    if batch.get("verdict") == "solved" or batch.get("route_status") == "solved":
        reasons.append("planner_direct_solved_claim")
    if contains_raw_reaction_payload({key: value for key, value in batch.items() if key != "actions"}):
        reasons.append("raw_reaction_injection")
    reasons.extend(planner_source_hint_reasons(batch.get("planner_source_hints")))
    actions = batch.get("actions")
    if not isinstance(actions, list):
        reasons.append("actions_not_list")
        actions = []
    if len(actions) > max_actions_per_round:
        reasons.append("action_batch_exceeds_max_actions")
    chemenzy_count = 0
    child_count = 0
    source_count = 0
    scout_action_count = 0
    visual_action_count = 0
    template_action_count = 0
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
        if contains_raw_reaction_payload(action):
            reasons.append("raw_reaction_injection")
        payload = dict(action.get("payload") or {})
        if action_type == "run_guided_chemenzy":
            chemenzy_count += 1
            reasons.extend(
                f"guided_chemenzy_payload:{idx}:{reason}"
                for reason in _guided_chemenzy_payload_reasons(payload, blackboard=board)
            )
        if action_type == "expand_child_target":
            child_count += 1
            reasons.extend(f"child_expansion_payload:{idx}:{reason}" for reason in _child_expansion_payload_reasons(payload))
        if action_type == "stitch_parent_route":
            reasons.extend(f"stitch_parent_route_payload:{idx}:{reason}" for reason in _stitch_parent_route_payload_reasons(payload))
        if action_type == "search_literature":
            scout_action_count += 1
            reasons.extend(f"search_literature_payload:{idx}:{reason}" for reason in _search_literature_payload_reasons(payload))
        if action_type == "extract_visual_literature_chain":
            visual_action_count += 1
        if action_type == "resolve_literature_structure_task" and bool(payload.get("run_visual", True)):
            visual_action_count += 1
        if action_type == "resolve_literature_structure_task":
            reasons.extend(
                f"resolve_literature_structure_task_payload:{idx}:{reason}"
                for reason in _structure_resolution_payload_reasons(payload)
            )
        if action_type in {"apply_analogical_template_to_target", "validate_template_application"}:
            template_action_count += 1
        if action_type in {
            "extract_analogical_reaction_templates",
            "rank_analogical_reaction_templates",
            "apply_analogical_template_to_target",
            "validate_template_application",
        }:
            reasons.extend(
                f"analogical_template_payload:{idx}:{reason}"
                for reason in _analogical_template_action_payload_reasons(payload, action_type=action_type)
            )
        if action_type in {
            "search_literature",
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "resolve_literature_structure_task",
            "compile_exact_literature_rows",
        }:
            source_count += max(1, int(payload.get("max_sources") or 1))
            if _source_binding_required(board, action_type) and not _payload_has_source_binding(payload):
                reasons.append(f"source_sensitive_action_missing_source_binding:{idx}:{action_type}")
        if _stale_action_repeated(board, action):
            reasons.append(f"stale_action_repeated:{idx}:{action_type}")
    if _must_stop_or_change_direction(board, actions):
        reasons.append("planner_must_stop_or_change_direction_after_two_unproductive_rounds")
    if chemenzy_count > max_chemenzy_per_round:
        reasons.append("guided_chemenzy_round_budget_exceeded")
    if child_count > max_child_expansions_per_round:
        reasons.append("child_expansion_round_budget_exceeded")
    if source_count > max_literature_sources_per_round:
        reasons.append("literature_source_round_budget_exceeded")
    budget = dict(board.get("budget_state") or {})
    if _budget_exceeded(budget, used_key="scout_calls", max_key="max_scout_calls", planned=scout_action_count):
        reasons.append("scout_total_budget_exceeded")
    if _budget_exceeded(budget, used_key="visual_calls", max_key="max_visual_calls", planned=visual_action_count):
        reasons.append("visual_total_budget_exceeded")
    if _budget_exceeded(budget, used_key="chemenzy_runs", max_key="max_chemenzy_runs", planned=chemenzy_count):
        reasons.append("guided_chemenzy_total_budget_exceeded")
    if _budget_exceeded(budget, used_key="child_target_runs", max_key="max_child_target_runs", planned=child_count):
        reasons.append("child_expansion_total_budget_exceeded")
    if _budget_exceeded(
        budget,
        used_key="template_application_actions",
        max_key="max_template_application_actions",
        planned=template_action_count,
    ):
        reasons.append("template_application_total_budget_exceeded")
    return {
        "schema_version": "agent_action_batch_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "case_id": str(batch.get("case_id") or ""),
        "action_count": len(actions),
    }


def _budget_exceeded(budget: dict[str, Any], *, used_key: str, max_key: str, planned: int) -> bool:
    if int(planned or 0) <= 0:
        return False
    if max_key not in budget:
        return False
    try:
        used = int(budget.get(used_key) or 0)
        maximum = int(budget.get(max_key) or 0)
    except (TypeError, ValueError):
        return False
    return maximum >= 0 and used + int(planned or 0) > maximum


def _source_binding_required(blackboard: dict[str, Any], action_type: str) -> bool:
    if action_type not in {
        "extract_pdf_literature_structures",
        "extract_visual_literature_chain",
        "resolve_literature_structure_task",
        "compile_exact_literature_rows",
    }:
        return False
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    source_candidates = [
        dict(row)
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict) and _source_key(row)
    ]
    local_pdf_candidates = [
        row
        for row in source_candidates
        if str(row.get("local_pdf") or row.get("pdf_path") or row.get("source_pdf_path") or "").strip()
    ]
    visual_chains = [
        dict(row)
        for row in evidence.get("visual_chains") or []
        if isinstance(row, dict) and (_source_key(row) or str(row.get("chain_id") or row.get("artifact_ref") or "").strip())
    ]
    if action_type in {"extract_pdf_literature_structures", "extract_visual_literature_chain"}:
        return len(_distinct_source_keys(local_pdf_candidates or source_candidates)) > 1
    if action_type == "resolve_literature_structure_task":
        task_sources = [
            row
            for row in (evidence.get("structure_resolution_tasks") or [])
            if isinstance(row, dict) and str(row.get("status") or "open") == "open"
        ]
        return len(_distinct_source_keys(local_pdf_candidates or source_candidates or task_sources)) > 1
    if action_type == "compile_exact_literature_rows":
        return len(_distinct_source_keys(visual_chains)) > 1 or len(_distinct_source_keys(source_candidates)) > 1
    return False


def _payload_has_source_binding(payload: dict[str, Any]) -> bool:
    binding_fields = {
        "source_ref",
        "doi",
        "pii",
        "url",
        "source_title",
        "title",
        "pdf_path",
        "local_pdf",
        "source_pdf_path",
        "chain_id",
        "visual_chain_id",
        "artifact_ref",
    }
    return any(str(payload.get(field) or "").strip() for field in binding_fields)


def _distinct_source_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    fallback_prefix = "row"
    for index, row in enumerate(rows):
        key = _source_key(row)
        if not key:
            key = str(row.get("chain_id") or row.get("artifact_ref") or "").strip().lower()
            if key:
                key = f"chain:{key}"
        if key:
            keys.add(key)
        else:
            keys.add(f"{fallback_prefix}:{index}")
    return keys


def _guided_chemenzy_payload_reasons(
    payload: dict[str, Any],
    *,
    blackboard: dict[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    board = dict(blackboard or {})
    policy = payload.get("chem_enzy_search_policy") or payload.get("search_policy")
    if not isinstance(policy, dict) or not policy:
        return ["missing_search_policy"]
    validation = validate_chem_enzy_search_policy(policy)
    if not validation.get("accepted"):
        reasons.extend(f"invalid_search_policy:{reason}" for reason in validation.get("reasons") or [])
    required_list_fields = [
        "terminal_blacklist",
        "active_bridge_tasks",
        "accepted_exact_row_ids",
        "selected_analogical_hypothesis_ids",
        "selected_analogical_template_ids",
        "forbidden_template_ids",
    ]
    for field in required_list_fields:
        if field not in policy:
            reasons.append(f"missing_policy_field:{field}")
        elif not isinstance(policy.get(field), list):
            reasons.append(f"policy_field_not_list:{field}")
    source_budget = dict(policy.get("source_budget") or {})
    if source_budget.get("require_target_core_retention") is not True:
        reasons.append("source_budget_missing_target_core_retention")
    try:
        max_jump = int(source_budget.get("max_unexplained_heavy_atom_jump"))
    except (TypeError, ValueError):
        max_jump = 0
    if max_jump <= 0:
        reasons.append("source_budget_missing_max_unexplained_heavy_atom_jump")
    if source_budget.get("analogy_is_advisory_only") is not True:
        reasons.append("source_budget_missing_analogy_advisory_boundary")
    compiler = dict(policy.get("compiler_metadata") or {})
    if compiler.get("requires_verifier") is not True:
        reasons.append("compiler_metadata_missing_requires_verifier")
    if compiler.get("no_solved_claim") is not True:
        reasons.append("compiler_metadata_missing_no_solved_claim")
    if not _guided_policy_has_consumable_signal(policy, board):
        if not _simple_direct_chemenzy_target(board) and not _bounded_complex_initial_probe(payload, policy):
            reasons.append("guided_chemenzy_missing_prior_signal_for_complex_target")
    return reasons


def _guided_policy_has_consumable_signal(policy: dict[str, Any], blackboard: dict[str, Any]) -> bool:
    for field in (
        "active_bridge_tasks",
        "accepted_exact_row_ids",
        "selected_analogical_hypothesis_ids",
        "selected_analogical_template_ids",
    ):
        if policy.get(field):
            return True
    source_budget = dict(policy.get("source_budget") or {})
    for field in (
        "preferred_precursor_smiles",
        "semisynthesis_anchor_smiles",
        "semisynthesis_anchors",
        "route_objectives",
        "endpoint_candidates",
        "broad_transform_templates",
        "visual_exploratory_hints",
    ):
        if source_budget.get(field):
            return True
    template_rows = int(((blackboard.get("current_belief") or {}).get("template_policy") or {}).get("validated_one_step_row_count") or 0)
    hypotheses = dict(blackboard.get("target_side_disconnection_hypotheses") or {}).get("hypotheses") or []
    route_objectives = dict(blackboard.get("route_objective_summary") or {}).get("selected_objectives") or []
    return bool(
        blackboard.get("bridge_tasks")
        or _target_relevant_exact_rows_available(blackboard)
        or blackboard.get("analogical_hypothesis_ranking")
        or blackboard.get("analogical_templates")
        or blackboard.get("analogical_template_ranking")
        or blackboard.get("template_applications")
        or blackboard.get("broad_transform_templates")
        or blackboard.get("endpoint_candidates")
        or hypotheses
        or route_objectives
        or template_rows
    )


def _simple_direct_chemenzy_target(blackboard: dict[str, Any]) -> bool:
    target = dict(blackboard.get("target_profile") or {})
    try:
        heavy_atoms = int(target.get("heavy_atoms") or 0)
        rings = int(target.get("rings") or 0)
        stereocenters = int(target.get("stereocenters") or 0)
    except (TypeError, ValueError):
        return False
    hints = " ".join(
        [
            str(target.get("family_hint") or ""),
            *[str(item) for item in target.get("family_hints") or []],
        ]
    ).lower()
    complex_tokens = ("steroid", "polycyclic", "macrocycle", "peptide", "glycoside", "natural product")
    return bool(
        0 < heavy_atoms <= 18
        and rings <= 2
        and stereocenters <= 2
        and not any(token in hints for token in complex_tokens)
    )


def _bounded_complex_initial_probe(payload: dict[str, Any], policy: dict[str, Any]) -> bool:
    source_budget = dict(policy.get("source_budget") or {})
    compiler = dict(policy.get("compiler_metadata") or {})
    mode_text = " ".join(
        str(value or "")
        for value in (
            payload.get("search_preset"),
            payload.get("mode"),
            payload.get("search_mode"),
            policy.get("mode"),
            policy.get("search_mode"),
        )
    ).lower()
    explicit_probe = bool(
        payload.get("initial_probe")
        or source_budget.get("initial_scan_allowed")
        or compiler.get("initial_scan_probe")
        or any(token in mode_text for token in ("initial", "probe", "baseline", "cheap_scan"))
    )
    if not explicit_probe:
        return False
    budget = dict(policy.get("budget") or {})
    max_steps = _positive_int(payload.get("max_steps") or budget.get("max_depth"))
    iterations = _positive_int(payload.get("chem_enzy_iterations") or budget.get("max_iterations"))
    topk = _positive_int(payload.get("chem_enzy_expansion_topk") or budget.get("expansion_topk"))
    timeout_s = _positive_float(payload.get("timeout_s") or budget.get("timeout_s"))
    max_candidates = _positive_int(source_budget.get("max_candidates"))
    return bool(
        0 < max_steps <= 6
        and 0 < iterations <= 10
        and 0 < topk <= 20
        and 0 < timeout_s <= 180
        and (max_candidates == 0 or max_candidates <= 5)
    )


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _positive_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _child_expansion_payload_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    targets = payload.get("subgoal_targets") or payload.get("child_targets")
    if not isinstance(targets, list) or not targets:
        return ["missing_subgoal_targets"]
    for idx, raw in enumerate(targets):
        if not isinstance(raw, dict):
            reasons.append(f"target_not_object:{idx}")
            continue
        target = dict(raw)
        if not str(target.get("smiles") or target.get("target_smiles") or "").strip():
            reasons.append(f"target_missing_smiles:{idx}")
        if target.get("target_equivalence_audit_required") is not True:
            reasons.append(f"target_missing_equivalence_audit_required:{idx}")
        if not bool(target.get("exact_target_override") or target.get("strict_exact_target")):
            reasons.append(f"target_missing_exact_target_override:{idx}")
        if target.get("no_solved_claim") is not True:
            reasons.append(f"target_missing_no_solved_claim:{idx}")
        if target.get("child_route_cannot_promote_parent") is not True:
            reasons.append(f"target_missing_child_parent_boundary:{idx}")
        policy = target.get("chem_enzy_search_policy") or target.get("policy")
        if not isinstance(policy, dict) or not policy:
            reasons.append(f"target_missing_search_policy:{idx}")
            continue
        validation = validate_chem_enzy_search_policy(policy)
        if not validation.get("accepted"):
            reasons.extend(f"target_invalid_search_policy:{idx}:{reason}" for reason in validation.get("reasons") or [])
        compiler = dict(policy.get("compiler_metadata") or {})
        if compiler.get("requires_verifier") is not True:
            reasons.append(f"target_policy_missing_requires_verifier:{idx}")
        if compiler.get("no_solved_claim") is not True:
            reasons.append(f"target_policy_missing_no_solved_claim:{idx}")
        if compiler.get("child_route_cannot_promote_parent") is not True:
            reasons.append(f"target_policy_missing_child_parent_boundary:{idx}")
    return reasons


def _stitch_parent_route_payload_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    binding = dict(payload.get("proof_binding") or {})
    if not binding:
        return ["missing_proof_binding"]
    if binding.get("schema_version") != "agentic_parent_stitch_binding.v1":
        reasons.append("invalid_proof_binding_schema")
    input_refs = binding.get("input_refs")
    exact_rows = binding.get("exact_literature_row_ids")
    missing_inputs = {str(item) for item in binding.get("missing_inputs") or [] if str(item or "").strip()}
    if not isinstance(input_refs, list) or not [item for item in input_refs if str(item or "").strip()]:
        reasons.append("proof_binding_missing_input_refs")
    if not isinstance(exact_rows, list):
        reasons.append("proof_binding_exact_rows_not_list")
    elif not exact_rows and "exact_literature_rows_missing" not in missing_inputs:
        reasons.append("proof_binding_missing_exact_rows_without_reason")
    if not str(binding.get("child_route_ref") or "").strip() and "child_route_ref_missing" not in missing_inputs:
        reasons.append("proof_binding_missing_child_route_ref_without_reason")
    if not str(binding.get("parent_route_ref") or "").strip() and "parent_route_ref_missing" not in missing_inputs:
        reasons.append("proof_binding_missing_parent_route_ref_without_reason")

    policy = dict(payload.get("proof_policy") or {})
    if not policy:
        reasons.append("missing_proof_policy")
    required_true = [
        "target_equivalence_required",
        "parent_route_verifier_required",
        "stock_audit_required",
        "no_unexplained_large_atom_jump_required",
        "child_route_connectivity_required",
        "exact_literature_connectivity_required",
        "analogy_is_not_proof",
        "child_route_cannot_promote_parent",
    ]
    for field in required_true:
        if policy.get(field) is not True:
            reasons.append(f"proof_policy_missing_{field}")
    if policy.get("final_verdict_authority") != "deterministic_parent_route_proof":
        reasons.append("proof_policy_invalid_final_verdict_authority")
    for idx, row in enumerate(payload.get("analogy_refs") or []):
        if isinstance(row, dict) and row.get("used_as_proof"):
            reasons.append(f"analogy_ref_used_as_proof:{idx}")
    return reasons


def _analogical_template_action_payload_reasons(payload: dict[str, Any], *, action_type: str) -> list[str]:
    reasons: list[str] = []
    policy = dict(payload.get("analogical_template_policy") or {})
    if not policy:
        return ["missing_analogical_template_policy"]
    if policy.get("schema_version") != "agentic_analogical_template_action_policy.v1":
        reasons.append("invalid_analogical_template_policy_schema")
    if policy.get("action_type") not in {"", None, action_type}:
        reasons.append("analogical_template_policy_action_type_mismatch")
    if policy.get("analogy_is_advisory_only") is not True:
        reasons.append("analogical_template_policy_not_advisory_only")
    if policy.get("no_solved_claim") is not True:
        reasons.append("analogical_template_policy_missing_no_solved_claim")
    if policy.get("requires_verifier") is not True:
        reasons.append("analogical_template_policy_missing_requires_verifier")
    if policy.get("requires_parent_route_proof") is not True:
        reasons.append("analogical_template_policy_missing_parent_proof_gate")
    if policy.get("production_write_blocked") is not True:
        reasons.append("analogical_template_policy_missing_production_block")
    if policy.get("raw_reaction_output_allowed") is not False:
        reasons.append("analogical_template_policy_allows_raw_reaction_output")
    if policy.get("final_verdict_authority") != "deterministic_parent_route_proof":
        reasons.append("analogical_template_policy_invalid_final_verdict_authority")
    allowed_uses = {str(item) for item in policy.get("allowed_use") or [] if str(item or "").strip()}
    invalid_uses = sorted(
        allowed_uses
        - {"planner_priority", "guided_policy_hint", "template_candidate_validation", "bridge_task_triage"}
    )
    if not allowed_uses:
        reasons.append("analogical_template_policy_missing_allowed_use")
    if invalid_uses:
        reasons.append(f"analogical_template_policy_invalid_allowed_use:{','.join(invalid_uses)}")
    if any(item in allowed_uses for item in {"solved_proof", "final_verdict", "parent_route_proof"}):
        reasons.append("analogical_template_policy_uses_analogy_as_proof")
    if action_type == "validate_template_application" and policy.get("deterministic_template_validation_required") is not True:
        reasons.append("analogical_template_policy_missing_deterministic_validation_gate")
    return reasons


def _search_literature_payload_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    queries = _search_payload_queries(payload)
    if not str(payload.get("search_intent") or payload.get("query") or "").strip() and not queries:
        reasons.append("missing_search_intent_or_queries")
    policy = dict(payload.get("source_acquisition_policy") or {})
    if not policy:
        reasons.append("missing_source_acquisition_policy")
        return reasons
    if policy.get("schema_version") not in {"agentic_source_acquisition_policy.v1", "source_acquisition_policy.v1"}:
        reasons.append("invalid_source_acquisition_policy_schema")
    if policy.get("codex_online_first") is not True:
        reasons.append("source_policy_missing_codex_online_first")
    if policy.get("local_pdf_fallback_allowed") is not True:
        reasons.append("source_policy_missing_local_pdf_fallback")
    if policy.get("placeholder_allowed_after_failures") is not True:
        reasons.append("source_policy_missing_placeholder_fallback")
    if policy.get("auto_local_pdf_requires_agent_discovered_metadata") is not True:
        reasons.append("source_policy_missing_auto_pdf_metadata_guard")
    if policy.get("no_solved_claim") is not True:
        reasons.append("source_policy_missing_no_solved_claim")
    fallback_order = [str(item) for item in policy.get("fallback_order") or []]
    if fallback_order != ["codex_online", "local_pdf", "placeholder"]:
        reasons.append("source_policy_invalid_fallback_order")
    return reasons


def _search_payload_queries(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if str(payload.get("query") or "").strip():
        values.append(str(payload.get("query") or ""))
    values.extend(str(item) for item in payload.get("queries") or [] if str(item or "").strip())
    values.extend(str(item) for item in payload.get("search_queries") or [] if str(item or "").strip())
    return _dedupe(values)


def _structure_resolution_payload_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not str(payload.get("task_id") or "").strip():
        reasons.append("missing_task_id")
    if not str(payload.get("label") or payload.get("compound_label") or "").strip():
        reasons.append("missing_label")
    if payload.get("no_solved_claim") is not True:
        reasons.append("missing_no_solved_claim")
    return reasons


def _must_stop_or_change_direction(blackboard: dict[str, Any], actions: list[Any]) -> bool:
    if not _two_recent_rounds_without_useful_artifact(blackboard):
        return False
    rows = [dict(action) for action in actions if isinstance(action, dict)]
    if not rows:
        return False
    if any(str(action.get("action_type") or "") == "stop_unresolved" for action in rows):
        return False
    planned_signatures = {_action_signature(action) for action in rows}
    recent_signatures = _recent_unproductive_action_signatures(blackboard, round_count=2)
    return bool(planned_signatures) and planned_signatures <= recent_signatures


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
        for row in _target_relevant_exact_literature_rows(blackboard)
        if isinstance(row, dict)
    ]
    selected_analogy_ids = [
        str(row.get("hypothesis_id") or "")
        for row in ((blackboard.get("analogical_hypothesis_ranking") or {}).get("selected_hypotheses") or [])
        if isinstance(row, dict)
    ]
    selected_template_ids = [
        str(row.get("template_id") or "")
        for row in ((blackboard.get("analogical_template_ranking") or {}).get("selected_templates") or [])
        if isinstance(row, dict)
    ]
    accepted_template_applications = [
        dict(row)
        for row in blackboard.get("template_applications") or []
        if isinstance(row, dict) and bool(row.get("accepted"))
    ]
    template_hints = [
        {
            "application_id": str(row.get("application_id") or ""),
            "template_id": str(row.get("template_id") or ""),
            "allowed_use": str(row.get("allowed_use") or ""),
            "product_retron_type": str(row.get("product_retron_type") or ""),
            "executable_candidate_available": bool(row.get("executable_candidate_available")),
            "hypothetical_route_hypothesis": dict(row.get("hypothetical_route_hypothesis") or {}),
            "hypothetical_precursor_hints": _compact_hypothetical_precursor_hints_for_policy(
                row.get("hypothetical_precursor_hints") or []
            ),
            "reaction_center_summary": dict(row.get("reaction_center_summary") or {}),
            "not_exact_literature_segment": bool(row.get("not_exact_literature_segment", True)),
            "not_parent_route_proof": bool(row.get("not_parent_route_proof", True)),
            "no_solved_claim": True,
        }
        for row in accepted_template_applications
    ][:6]
    hypothetical_route_hints = [
        {
            "application_id": str(row.get("application_id") or ""),
            "template_id": str(row.get("template_id") or ""),
            "product_retron_type": str(row.get("product_retron_type") or ""),
            "allowed_use": str(row.get("allowed_use") or ""),
            "hypothesis": dict(row.get("hypothetical_route_hypothesis") or {}),
            "hypothetical_precursor_hints": _compact_hypothetical_precursor_hints_for_policy(
                row.get("hypothetical_precursor_hints") or []
            ),
            "reaction_center_summary": dict(row.get("reaction_center_summary") or {}),
            "analogy_is_advisory_only": True,
            "not_exact_literature_segment": True,
            "not_parent_route_proof": True,
            "no_solved_claim": True,
        }
        for row in accepted_template_applications
        if str(row.get("allowed_use") or "") == "hypothesis_only_not_solved"
    ][:6]
    hypothetical_precursor_targets = _hypothetical_precursor_targets_from_template_applications(
        accepted_template_applications,
        limit=12,
    )
    visual_exploratory_hints = _visual_exploratory_hints_for_policy(blackboard)
    visual_precursor_targets = _visual_precursor_targets_from_hints(visual_exploratory_hints, limit=8)
    hypothetical_precursor_smiles = [str(row.get("smiles") or "") for row in hypothetical_precursor_targets]
    visual_precursor_smiles = [str(row.get("smiles") or "") for row in visual_precursor_targets]
    forbidden_template_ids = [
        str(row.get("template_id") or "")
        for row in blackboard.get("template_failure_memory") or []
        if isinstance(row, dict) and int(row.get("failure_count") or 0) >= 2
    ]
    bridge_tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    route_objectives = [
        dict(row)
        for row in (blackboard.get("route_objective_summary") or {}).get("selected_objectives") or []
        if isinstance(row, dict)
    ]
    endpoint_candidates = [
        dict(row)
        for row in blackboard.get("endpoint_candidates") or []
        if isinstance(row, dict)
    ]
    broad_templates = [
        dict(row)
        for row in blackboard.get("broad_transform_templates") or []
        if isinstance(row, dict)
    ]
    semisynthesis_anchors = [
        dict(row)
        for row in blackboard.get("semisynthesis_anchors") or []
        if isinstance(row, dict)
    ]
    semisynthesis_anchor_smiles = [
        str(row.get("smiles") or "")
        for row in semisynthesis_anchors
        if str(row.get("smiles") or "").strip()
    ]
    constraints = dict((blackboard.get("current_belief") or {}).get("constraints") or {})
    evidence_refs = _guided_evidence_refs(
        blackboard=blackboard,
        bridge_tasks=bridge_tasks,
        exact_rows=exact_rows,
        selected_analogy_ids=selected_analogy_ids,
        selected_template_ids=selected_template_ids,
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
            "selected_analogical_template_ids": _dedupe(selected_template_ids),
            "forbidden_template_ids": _dedupe(forbidden_template_ids),
            "source_budget": {
                "require_target_core_retention": bool(constraints.get("target_core_retention_required", True)),
                "max_unexplained_heavy_atom_jump": int(constraints.get("max_unexplained_heavy_atom_jump") or 15),
                "analogy_is_advisory_only": True,
                "target_relevant_exact_rows_only": True,
                "disconnected_exact_row_ids": _disconnected_exact_literature_row_ids(blackboard),
                "max_template_generated_precursors_per_round": int(
                    ((blackboard.get("current_belief") or {}).get("template_policy") or {}).get("max_template_applications_per_round")
                    or 5
                ),
                "preferred_reaction_classes": _dedupe(
                    [
                        *[
                            str(row.get("objective_type") or "").replace("_", " ")
                            for row in route_objectives
                            if str(row.get("objective_type") or "").strip()
                        ],
                        *[
                            str(row.get("endpoint_type") or "").replace("_", " ")
                            for row in endpoint_candidates[:6]
                            if str(row.get("endpoint_type") or "").strip()
                        ],
                        *[
                            str(row.get("objective_type") or "").replace("_", " ")
                            for row in broad_templates[:6]
                            if str(row.get("objective_type") or "").strip()
                        ],
                        "target_proximal_bridge_search",
                        *(
                            ["visual_connectivity_approximation"]
                            if visual_exploratory_hints
                            else []
                        ),
                        *[
                            str(row.get("product_retron_type") or "")
                            for row in hypothetical_route_hints
                            if str(row.get("product_retron_type") or "").strip()
                        ],
                    ]
                ),
                "hypothetical_route_hints_are_not_proof": True,
                "hypothesis_precursor_hints_are_not_proof": True,
                "visual_connectivity_hints_are_not_proof": bool(visual_exploratory_hints),
                "semisynthesis_anchor_hints_are_not_proof": bool(semisynthesis_anchors),
                "route_objective_hints_are_not_proof": bool(route_objectives),
                "broad_transform_templates_are_not_proof": bool(broad_templates),
                "de_novo_core_construction_deprioritized": bool(
                    constraints.get("de_novo_core_construction_deprioritized")
                ),
                "small_molecule_stock_closure_deprioritized": bool(
                    constraints.get("small_molecule_stock_closure_deprioritized")
                ),
                "preferred_precursor_smiles": _dedupe([*hypothetical_precursor_smiles, *visual_precursor_smiles]),
                "semisynthesis_anchor_smiles": _dedupe(semisynthesis_anchor_smiles),
                "semisynthesis_anchors": semisynthesis_anchors,
                "route_objectives": route_objectives,
                "endpoint_candidates": endpoint_candidates,
                "broad_transform_templates": broad_templates,
                "visual_exploratory_hints": visual_exploratory_hints,
            },
            "preferred_subgoal": {
                "target": dict(blackboard.get("target_profile") or {}),
                "bridge_tasks": bridge_tasks,
                "preferred_subgoals": _dedupe(
                    [
                        *semisynthesis_anchor_smiles,
                        *hypothetical_precursor_smiles,
                        *visual_precursor_smiles,
                    ]
                ),
                "semisynthesis_anchors": semisynthesis_anchors,
                "route_objectives": route_objectives,
                "endpoint_candidates": endpoint_candidates,
                "broad_transform_templates": broad_templates,
                "hypothetical_precursor_targets": [*hypothetical_precursor_targets, *visual_precursor_targets],
                "template_application_hints": template_hints,
                "hypothetical_reaction_center_hints": hypothetical_route_hints,
                "visual_connectivity_hints": visual_exploratory_hints,
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


def build_child_expansion_payload_from_blackboard(blackboard: dict[str, Any]) -> dict[str, Any]:
    """Build a child-target expansion payload from explicit blackboard terminals."""
    return _child_expansion_payload(blackboard)


def _compact_hypothetical_precursor_hints_for_policy(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        precursor = str(row.get("precursor_smiles") or "").strip()
        if not precursor or precursor in seen:
            continue
        seen.add(precursor)
        out.append(
            {
                "schema_version": str(row.get("schema_version") or "analogical_hypothesis_precursor_hint.v1"),
                "hint_id": str(row.get("hint_id") or ""),
                "precursor_smiles": precursor,
                "precursor_role": str(row.get("precursor_role") or ""),
                "derived_from_retron": str(row.get("derived_from_retron") or ""),
                "hypothesis_type": str(row.get("hypothesis_type") or ""),
                "candidate_kind": "same_core_redox_or_protection_state_precursor",
                "allowed_use": "guided_search_subgoal_hint_only",
                "risk_flags": [str(item) for item in row.get("risk_flags") or [] if str(item or "").strip()],
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "no_solved_claim": True,
            }
        )
        if len(out) >= 8:
            break
    return out


def _hypothetical_precursor_targets_from_template_applications(
    applications: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for application in applications:
        app_id = str(application.get("application_id") or "")
        template_id = str(application.get("template_id") or "")
        retron = str(application.get("product_retron_type") or "")
        for hint in _compact_hypothetical_precursor_hints_for_policy(
            application.get("hypothetical_precursor_hints") or []
        ):
            smiles = str(hint.get("precursor_smiles") or "").strip()
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            out.append(
                {
                    "schema_version": "guided_search_hypothetical_precursor_target.v1",
                    "smiles": smiles,
                    "role": str(hint.get("precursor_role") or "same_core_precursor"),
                    "source_application_id": app_id,
                    "source_template_id": template_id,
                    "derived_from_retron": str(hint.get("derived_from_retron") or retron),
                    "hypothesis_type": str(hint.get("hypothesis_type") or ""),
                    "allowed_use": "guided_search_subgoal_hint_only",
                    "analogy_is_advisory_only": True,
                    "not_exact_literature_segment": True,
                    "not_parent_route_proof": True,
                    "requires_verifier": True,
                    "no_solved_claim": True,
                }
            )
            if len(out) >= max(1, int(limit or 1)):
                return out
    return out


def _visual_exploratory_hints_for_policy(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chain in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(chain, dict) or not _visual_chain_is_exploratory_only(chain):
            continue
        source_ref = str(chain.get("source_ref") or chain.get("source_pdf_path") or chain.get("artifact_ref") or "")
        for step in chain.get("steps") or []:
            if not isinstance(step, dict):
                continue
            reactants = [
                str(item).strip()
                for item in step.get("reactant_smiles") or []
                if str(item or "").strip()
            ]
            product = str(step.get("product_smiles") or "").strip()
            if not product or not reactants:
                continue
            key = f"{source_ref}:{product}:{'|'.join(reactants)}"
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                {
                    "schema_version": "visual_connectivity_guided_hint.v1",
                    "hint_id": f"visual_hint:{_safe_id(source_ref or 'source')}:{len(hints) + 1}",
                    "source_ref": source_ref,
                    "source_locator": str(step.get("source_locator") or ""),
                    "product_label": str(step.get("product_label") or ""),
                    "product_smiles": product,
                    "precursor_smiles": reactants[0],
                    "all_precursor_smiles": reactants[:4],
                    "stereochemistry_status": str(step.get("stereochemistry_status") or "unspecified_or_partial"),
                    "allowed_use": "guided_search_subgoal_hint_only",
                    "evidence_class": "visual_connectivity_approximation",
                    "not_exact_literature_segment": True,
                    "not_parent_route_proof": True,
                    "requires_verifier": True,
                    "risk_flags": _dedupe(
                        [
                            "visual_connectivity_approximation",
                            *[str(item) for item in step.get("risk_flags") or [] if str(item or "").strip()],
                        ]
                    ),
                    "no_solved_claim": True,
                }
            )
            if len(hints) >= 8:
                return hints
    return hints


def _visual_precursor_targets_from_hints(hints: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hint in hints:
        smiles = str(hint.get("precursor_smiles") or "").strip()
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        out.append(
            {
                "schema_version": "guided_search_hypothetical_precursor_target.v1",
                "smiles": smiles,
                "role": "visual_connectivity_same_core_precursor",
                "source_visual_hint_id": str(hint.get("hint_id") or ""),
                "source_ref": str(hint.get("source_ref") or ""),
                "allowed_use": "guided_search_subgoal_hint_only",
                "analogy_is_advisory_only": True,
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "no_solved_claim": True,
            }
        )
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def _guided_evidence_refs(
    *,
    blackboard: dict[str, Any],
    bridge_tasks: list[dict[str, Any]],
    exact_rows: list[str],
    selected_analogy_ids: list[str],
    selected_template_ids: list[str],
) -> list[str]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    refs = [str(item) for item in evidence.get("source_refs") or [] if str(item or "").strip()]
    refs.extend(str(item) for item in exact_rows if str(item or "").strip())
    refs.extend(str(item) for item in selected_analogy_ids if str(item or "").strip())
    refs.extend(str(item) for item in selected_template_ids if str(item or "").strip())
    refs.extend(str(row.get("task_id") or "") for row in bridge_tasks if str(row.get("task_id") or "").strip())
    refs.extend(
        str(row.get("application_id") or "")
        for row in blackboard.get("template_applications") or []
        if isinstance(row, dict) and str(row.get("application_id") or "").strip()
    )
    refs.extend(
        str(row.get("artifact_ref") or row.get("source_pdf_path") or "")
        for row in evidence.get("pdf_structure_evidence") or []
        if isinstance(row, dict)
    )
    refs.extend(
        str(row.get("artifact_ref") or row.get("source_ref") or row.get("source_pdf_path") or "")
        for row in evidence.get("visual_chains") or []
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


def _append_next_action_bias_actions(
    actions: list[dict[str, Any]],
    blackboard: dict[str, Any],
    *,
    round_index: int,
    max_actions: int,
) -> None:
    """Let critic-written blackboard bias influence the next deterministic plan."""
    for action_type in _next_action_biases(blackboard):
        if len(actions) >= max_actions:
            return
        if _round_has_action(actions, action_type):
            continue
        candidate = _biased_action_candidate(blackboard, round_index=round_index, action_type=action_type)
        if not candidate:
            continue
        if _stale_action_repeated(blackboard, candidate):
            continue
        actions.append(candidate)


def _next_action_biases(blackboard: dict[str, Any]) -> list[str]:
    belief = dict(blackboard.get("current_belief") or {})
    return _dedupe([str(item) for item in belief.get("next_action_bias") or [] if str(item or "").strip()])


def _biased_action_candidate(
    blackboard: dict[str, Any],
    *,
    round_index: int,
    action_type: str,
) -> dict[str, Any]:
    action_type = str(action_type or "").strip()
    if action_type == "classify_route_objectives":
        if blackboard.get("route_objective_summary"):
            return {}
        return _action(
            round_index,
            "classify_route_objectives",
            "blackboard bias requests route-objective classification",
            "route_objective_summary.v1",
            "ranked route objectives and endpoint candidates are recorded",
        )
    if action_type == "generate_disconnection_hypotheses":
        if blackboard.get("target_side_disconnection_hypotheses"):
            return {}
        return _action(
            round_index,
            "generate_disconnection_hypotheses",
            "failure critic biased the planner toward target-side disconnection",
            "target_side_disconnection_hypotheses.v1",
            "at least one advisory hypothesis and bridge task",
        )
    if action_type == "build_failure_critic_report":
        if not _failure_evidence_available(blackboard) or _action_seen(blackboard, "build_failure_critic_report"):
            return {}
        return _action(
            round_index,
            "build_failure_critic_report",
            "blackboard bias requests deterministic failure normalization",
            "failure_critic_report.v1",
            "critic emits bridge task or blocked direction",
        )
    if action_type == "search_literature":
        if not _budget_remaining(blackboard, "scout_calls"):
            return {}
        if _stale_literature_search_repeated(blackboard):
            return {}
        if not (_needs_literature_bridge(blackboard) or blackboard.get("bridge_tasks") or not _source_candidates_have_real_source(blackboard)):
            return {}
        return _action(
            round_index,
            "search_literature",
            "failure critic biased the planner toward target-proximal/source bridge evidence",
            "literature_scout_report.v1",
            "source candidate or extraction recommendation generated",
            _literature_search_payload(blackboard, intent="critic_biased_target_proximal_source_discovery"),
        )
    if action_type == "extract_pdf_literature_structures":
        source = _next_local_pdf_source_for_pdf_extraction(blackboard)
        if not source:
            return {}
        return _action(
            round_index,
            "extract_pdf_literature_structures",
            "blackboard bias requests converting an available PDF into extraction-ready evidence",
            "literature_pdf_structure_evidence.v1",
            "rendered pages or indexed images are available for visual extraction",
            _source_candidate_payload(source),
        )
    if action_type == "extract_visual_literature_chain":
        source = _next_local_pdf_source_for_visual_extraction(blackboard)
        if source and _budget_remaining(blackboard, "visual_calls"):
            return _action(
                round_index,
                "extract_visual_literature_chain",
                "blackboard bias requests visual extraction from an available source",
                "visual_literature_chain/exact rows artifact",
                "validated source-detail chain or extraction failure reason",
                _visual_extraction_payload_from_blackboard(blackboard, source_candidate=source),
            )
        return {}
    if action_type == "compile_exact_literature_rows":
        if not (
            _visual_chain_available(blackboard)
            and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
            and _uncompiled_visual_steps_available(blackboard)
        ):
            return {}
        return _action(
            round_index,
            "compile_exact_literature_rows",
            "blackboard bias requests compiling visual evidence into exact source-detail rows",
            "compiled exact literature rows",
            "one or more exact row summaries",
            _compile_exact_rows_payload(blackboard),
        )
    if action_type == "rank_analogical_hypotheses":
        if not _hypotheses_available(blackboard) or blackboard.get("analogical_hypothesis_ranking"):
            return {}
        return _action(
            round_index,
            "rank_analogical_hypotheses",
            "failure critic biased the planner toward analogy ranking after exact replay weakness",
            "analogical_hypothesis_ranking.v1",
            "ranked advisory hypotheses with no solved claim",
        )
    if action_type == "derive_broad_reaction_template":
        if not _broad_template_derivation_ready(blackboard):
            return {}
        return _action(
            round_index,
            "derive_broad_reaction_template",
            "blackboard bias requests broad non-exact transform templates for guided search",
            "broad_transform_template_report.v1",
            "one or more advisory broad templates are available without solved claim",
            _broad_template_payload(blackboard),
        )
    if action_type == "run_guided_chemenzy":
        if not _can_run_guided_chemenzy(blackboard):
            return {}
        return _action(
            round_index,
            "run_guided_chemenzy",
            "blackboard bias requests a guarded guided rerun after bridge evidence accumulated",
            "guided_chemenzy_result plus verifier report",
            "route verifier accepts or returns actionable failure evidence",
            _guided_retry_payload(blackboard),
        )
    if action_type == "expand_child_target":
        if not _can_expand_child_target(blackboard):
            return {}
        return _action(
            round_index,
            "expand_child_target",
            "blackboard bias requests upstream synthesis of an exact literature terminal",
            "route_expansion_subgoal_search_result.v1",
            "child target verifier result is recorded without parent solved claim",
            _child_expansion_payload(blackboard),
        )
    if action_type == "stitch_parent_route":
        if not _can_stitch_parent_route(blackboard):
            return {}
        return _action(
            round_index,
            "stitch_parent_route",
            "blackboard bias requests deterministic parent-route connectivity proof",
            "stitched_parent_route_proof.v1",
            "parent proof accepted or explicit connectivity rejection",
            _stitch_retry_payload(blackboard),
        )
    if action_type == "compile_objective_route_proof":
        if not _route_objective_proof_ready(blackboard):
            return {}
        return _action(
            round_index,
            "compile_objective_route_proof",
            "blackboard bias requests objective-specific proof bundle",
            "route_proof_bundle.v1",
            "objective proof bundle records solved/plausible/unresolved status",
            {"proof_scope": "objective_specific", "no_solved_claim": True},
        )
    return {}


def _failure_evidence_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("route_failures") or []) or (blackboard.get("plugin_runtime_diagnostics") or []))


def _needs_literature_bridge(blackboard: dict[str, Any]) -> bool:
    evidence = dict(blackboard.get("literature_evidence") or {})
    if _awaiting_local_pdf_proxy_download(blackboard) and _source_candidates_have_real_source(blackboard):
        return False
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


def _semisynthesis_anchor_validation_needed(blackboard: dict[str, Any]) -> bool:
    return _objective_evidence_validation_needed(blackboard)


def _objective_evidence_validation_needed(blackboard: dict[str, Any]) -> bool:
    summary = dict(blackboard.get("route_objective_summary") or {})
    route_scope = dict(summary.get("route_scope") or (blackboard.get("current_belief") or {}).get("route_scope") or {})
    selected = [
        dict(row)
        for row in summary.get("selected_objectives") or []
        if isinstance(row, dict)
    ]
    non_stock_selected = [
        row
        for row in selected[:3]
        if str(row.get("objective_type") or "") != "small_molecule_stock_closure"
    ]
    if not non_stock_selected and not blackboard.get("endpoint_candidates") and not blackboard.get("semisynthesis_anchors"):
        return False
    if _target_relevant_exact_rows_available(blackboard):
        return False
    if _objective_endpoint_search_attempted(blackboard):
        return False
    if route_scope.get("objective_evidence_validation_required"):
        return True
    if (blackboard.get("current_belief") or {}).get("constraints", {}).get("objective_evidence_validation_required"):
        return True
    if non_stock_selected and not _source_candidates_have_real_source(blackboard):
        return True
    return True


def _semisynthesis_anchor_search_attempted(blackboard: dict[str, Any]) -> bool:
    return _objective_endpoint_search_attempted(blackboard)


def _objective_endpoint_search_attempted(blackboard: dict[str, Any]) -> bool:
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != "search_literature":
            continue
        signature = str(row.get("action_signature") or "")
        if "route_objective_endpoint_validation" in signature or "semisynthesis_anchor_validation" in signature:
            return True
    return False


def _broad_template_derivation_ready(blackboard: dict[str, Any]) -> bool:
    if blackboard.get("broad_transform_templates"):
        return False
    if _exact_literature_segment_usable(blackboard):
        return False
    if blackboard.get("route_objective_summary") and (
        blackboard.get("target_side_disconnection_hypotheses")
        or blackboard.get("analogical_hypotheses")
        or blackboard.get("endpoint_candidates")
    ):
        return True
    return False


def _broad_template_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    selected = [
        str(row.get("objective_id") or row.get("objective_type") or "")
        for row in (blackboard.get("route_objective_summary") or {}).get("selected_objectives") or []
        if isinstance(row, dict) and str(row.get("objective_id") or row.get("objective_type") or "").strip()
    ]
    return {
        "schema_version": "broad_transform_template_action_payload.v1",
        "selected_objective_ids": _dedupe(selected),
        "source_material": {
            "target_side_hypotheses": bool(blackboard.get("target_side_disconnection_hypotheses")),
            "endpoint_candidates": len(blackboard.get("endpoint_candidates") or []),
            "analogical_hypotheses": len(blackboard.get("analogical_hypotheses") or []),
        },
        "allowed_use": "planner_priority_and_guided_search_hint_only",
        "no_solved_claim": True,
    }


def _route_objective_proof_ready(blackboard: dict[str, Any]) -> bool:
    if not blackboard.get("route_objective_summary"):
        return False
    if blackboard.get("parent_route_proof") or blackboard.get("broad_transform_templates"):
        return True
    evidence = dict(blackboard.get("literature_evidence") or {})
    return bool(evidence.get("source_candidates") or evidence.get("visual_chains") or evidence.get("exact_rows"))


def _awaiting_local_pdf_proxy_download(blackboard: dict[str, Any]) -> bool:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    if evidence.get("local_pdf_proxy_requests"):
        return True
    return any(
        isinstance(row, dict) and str(row.get("stage") or "") == "local_pdf_proxy_requested"
        for row in evidence.get("source_lifecycle") or []
    )


def _candidate_has_real_source(row: dict[str, Any]) -> bool:
    if bool(row.get("placeholder_only")):
        return False
    if str(row.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    return bool(str(row.get("doi") or row.get("pii") or row.get("url") or row.get("local_pdf") or "").strip())


def _source_candidates_visual_ready(blackboard: dict[str, Any]) -> bool:
    if _source_candidates_include_local_pdf(blackboard):
        return True
    return _pdf_structure_evidence_available(blackboard)


def _local_pdf_source_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(row, dict) and str(row.get("local_pdf") or "").strip()
    ]


def _next_local_pdf_source_for_pdf_extraction(blackboard: dict[str, Any]) -> dict[str, Any]:
    candidates = _local_pdf_source_candidates(blackboard)
    if len(candidates) == 1 and _pdf_structure_evidence_available(blackboard):
        return {}
    rendered = _pdf_structure_source_keys(blackboard)
    for row in candidates:
        key = _source_key(row)
        if key and key not in rendered:
            return row
    return {}


def _next_local_pdf_source_for_visual_extraction(blackboard: dict[str, Any]) -> dict[str, Any]:
    candidates = _local_pdf_source_candidates(blackboard)
    if len(candidates) == 1 and _visual_chain_available(blackboard):
        return {}
    rendered = _pdf_structure_source_keys(blackboard)
    visualized = _visual_chain_source_keys(blackboard)
    for row in candidates:
        key = _source_key(row)
        if not key or key in visualized:
            continue
        if key in rendered or (len(candidates) == 1 and _pdf_structure_evidence_available(blackboard)):
            return row
    return {}


def _pdf_structure_source_keys(blackboard: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for row in (blackboard.get("literature_evidence") or {}).get("pdf_structure_evidence") or []:
        if not isinstance(row, dict):
            continue
        key = _source_key(row)
        if key:
            keys.add(key)
    return keys


def _visual_chain_source_keys(blackboard: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        key = _source_key(row)
        if key:
            keys.add(key)
    return keys


def _source_key(row: dict[str, Any]) -> str:
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        return f"pii:{pii}"
    local_pdf = str(row.get("local_pdf") or row.get("source_pdf_path") or row.get("pdf_path") or "").strip().lower()
    if local_pdf:
        return f"pdf:{local_pdf}"
    title = str(row.get("title") or row.get("source_title") or "").strip().lower()
    return f"title:{title}" if title else ""


def _source_candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if row.get("source_ref"):
        payload["source_ref"] = str(row.get("source_ref") or "")
    if row.get("title"):
        payload["source_title"] = str(row.get("title") or "")
    if row.get("local_pdf"):
        payload["pdf_path"] = str(row.get("local_pdf") or "")
        payload.setdefault("render_zoom", 1.35)
        payload.setdefault("compress_images", True)
        payload.setdefault("max_images", 6)
        payload.setdefault("visual_max_side_px", 1400)
        payload.setdefault("visual_jpeg_quality", 70)
    if row.get("route_sequence_hint"):
        payload["route_sequence_hint"] = str(row.get("route_sequence_hint") or "")
    labels = [str(item) for item in row.get("expected_scheme_or_compound_labels") or [] if str(item or "").strip()]
    if labels:
        payload["expected_labels"] = labels
        payload["compound_labels"] = labels
    _add_source_visual_focus_defaults(payload, row)
    return payload


def _add_source_visual_focus_defaults(payload: dict[str, Any], row: dict[str, Any]) -> None:
    text = " ".join(
        str(row.get(key) or payload.get(key) or "")
        for key in ("title", "source_title", "source_ref", "doi", "local_pdf", "pdf_path")
    ).lower()
    if "ouabagenin" in text or "ouabain" in text or any(doi in text for doi in ("10.1002/asia.200800429", "10.1002/anie.200704959", "10.1139/v05-042")):
        payload.setdefault("compress_images", True)
        payload.setdefault("max_images", 6)
        payload.setdefault("visual_max_side_px", 1400)
        payload.setdefault("visual_jpeg_quality", 70)
        payload.setdefault("route_sequence_hint", "")
        payload["route_sequence_hint"] = " ".join(
            part
            for part in [
                str(payload.get("route_sequence_hint") or ""),
                "For ouabagenin/ouabain sources, inspect source-detail schemes for target-proximal steroid-core intermediates. "
                "Use only visible labels and structures; unresolved labels must remain extraction gaps.",
            ]
            if part
        )


def _literature_search_payload(blackboard: dict[str, Any], *, intent: str) -> dict[str, Any]:
    target = dict(blackboard.get("target_profile") or {})
    evidence = dict(blackboard.get("literature_evidence") or {})
    target_name = str(target.get("target_name") or "target")
    family = str(target.get("family_hint") or "")
    bridge_tasks = [dict(row) for row in blackboard.get("bridge_tasks") or [] if isinstance(row, dict)]
    planner_hints = _planner_source_hints_for_search(blackboard)
    objective_hints = _route_objective_source_hints_for_search(blackboard)
    handles = _dedupe(
        [
            str(row.get("target_handle") or row.get("task_type") or "")
            for row in bridge_tasks
            if str(row.get("target_handle") or row.get("task_type") or "").strip()
        ]
    )
    base = [part for part in [target_name, family] if part]
    queries = [
        " ".join([*base, "synthesis", "total synthesis", "semisynthesis"]),
        " ".join([*base, "target proximal intermediate", "retrosynthesis"]),
    ]
    for handle in handles[:4]:
            queries.append(" ".join([*base, handle, "synthesis bridge"]))
    for hint in [*objective_hints, *planner_hints][:8]:
        doi = str(hint.get("doi") or "").strip()
        pii = str(hint.get("pii") or "").strip()
        title = str(hint.get("title") or "").strip()
        source_ref = str(hint.get("source_ref") or "").strip()
        if doi:
            queries.append(doi)
        if pii:
            queries.append(pii)
        if title:
            queries.append(" ".join([title, "synthesis scheme"]))
        elif source_ref:
            queries.append(" ".join([source_ref, target_name, "synthesis"]))
    for endpoint in blackboard.get("endpoint_candidates") or []:
        if not isinstance(endpoint, dict):
            continue
        endpoint_text = str(endpoint.get("endpoint_type") or endpoint.get("description") or "").strip()
        objective_type = str(endpoint.get("objective_type") or "").strip()
        if endpoint_text:
            queries.append(" ".join([target_name, endpoint_text, objective_type, "synthesis evidence"]))
    query_list = _dedupe([query for query in queries if query.strip()])[:12]
    return {
        "schema_version": "agentic_literature_search_payload.v1",
        "search_intent": str(intent or "target_proximal_source_discovery"),
        "queries": query_list,
        "search_queries": query_list,
        "max_sources": 3,
        "planner_source_hints": [*objective_hints, *planner_hints],
        "route_objectives": [
            dict(row)
            for row in (blackboard.get("route_objective_summary") or {}).get("selected_objectives") or []
            if isinstance(row, dict)
        ],
        "endpoint_candidates": [
            dict(row)
            for row in blackboard.get("endpoint_candidates") or []
            if isinstance(row, dict)
        ],
        "prior_source_candidate_count": len(evidence.get("source_candidates") or []),
        "source_acquisition_policy": _source_acquisition_policy(),
        "no_solved_claim": True,
    }


def _semisynthesis_source_hints_for_search(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    return _route_objective_source_hints_for_search(blackboard)


def _route_objective_source_hints_for_search(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    guidance = [
        dict(row)
        for row in (blackboard.get("route_objective_summary") or {}).get("source_search_guidance") or []
        if isinstance(row, dict)
    ]
    for guide in guidance[:4]:
        objective_type = str(guide.get("objective_type") or "").strip()
        terms = [str(item) for item in guide.get("search_terms") or [] if str(item or "").strip()]
        rows.append(
            {
                "schema_version": "planner_source_hint.v1",
                "hint_id": f"route_objective_source_hint:{len(rows) + 1}",
                "source_ref": f"route_objective:{objective_type}" if objective_type else "",
                "title": " ".join(terms[:5]),
                "doi": "",
                "pii": "",
                "url": "",
                "local_pdf": "",
                "local_ref": "",
                "source_type": str(objective_type or "route_objective_search_guidance"),
                "relevance_rationale": "source search guidance generated from route-objective ranking",
                "expected_scheme_or_compound_labels": terms[:6],
                "extraction_task_recommendations": [
                    "extract endpoint identity",
                    "extract same-scaffold or same-family transformation evidence",
                    "extract whether the source supports exact, broad-template, or hypothesis-only use",
                ],
                "evidence_class": "planner_source_hint",
                "allowed_use": "source_acquisition_hint_only",
                "no_solved_claim": True,
            }
        )
        if len(rows) >= 4:
            break
    return rows


def _planner_source_hints_for_search(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in (blackboard.get("literature_evidence") or {}).get("planner_source_hints") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "schema_version": str(row.get("schema_version") or "planner_source_hint.v1"),
                "hint_id": str(row.get("hint_id") or ""),
                "source_ref": str(row.get("source_ref") or ""),
                "title": str(row.get("title") or ""),
                "doi": str(row.get("doi") or ""),
                "pii": str(row.get("pii") or ""),
                "url": str(row.get("url") or ""),
                "local_pdf": str(row.get("local_pdf") or ""),
                "local_ref": str(row.get("local_ref") or ""),
                "source_type": str(row.get("source_type") or ""),
                "relevance_rationale": str(row.get("relevance_rationale") or ""),
                "expected_scheme_or_compound_labels": [
                    str(item)
                    for item in row.get("expected_scheme_or_compound_labels") or []
                    if str(item or "").strip()
                ],
                "extraction_task_recommendations": [
                    str(item)
                    for item in row.get("extraction_task_recommendations") or []
                    if str(item or "").strip()
                ],
                "evidence_class": "planner_source_hint",
                "allowed_use": "source_acquisition_hint_only",
                "no_solved_claim": True,
            }
        )
        if len(rows) >= 8:
            break
    return rows


def _source_acquisition_policy() -> dict[str, Any]:
    return {
        "schema_version": "agentic_source_acquisition_policy.v1",
        "codex_online_first": True,
        "local_pdf_fallback_allowed": True,
        "placeholder_allowed_after_failures": True,
        "auto_local_pdf_requires_agent_discovered_metadata": True,
        "fallback_order": ["codex_online", "local_pdf", "placeholder"],
        "no_solved_claim": True,
    }


def _pdf_structure_evidence_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("pdf_structure_evidence"))


def _visual_chain_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("visual_chains"))


def _uncompiled_visual_steps_available(blackboard: dict[str, Any]) -> bool:
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        if _visual_chain_uncompiled_step_count(blackboard, dict(row)) > 0:
            return True
    return False


def _compile_exact_rows_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"compile_attempt": _action_count(blackboard, "compile_exact_literature_rows") + 1}
    source = _next_uncompiled_visual_source(blackboard) or _latest_visual_source(blackboard)
    if source:
        payload.update(_source_candidate_payload(source))
    return payload


def _next_uncompiled_visual_source(blackboard: dict[str, Any]) -> dict[str, Any]:
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        visual = dict(row)
        if _visual_chain_uncompiled_step_count(blackboard, visual) <= 0:
            continue
        key = _source_key(visual)
        for candidate in _local_pdf_source_candidates(blackboard):
            if key and _source_key(candidate) == key:
                return candidate
        return {
            "source_ref": str(visual.get("source_ref") or ""),
            "title": str(visual.get("source_title") or ""),
            "local_pdf": str(visual.get("source_pdf_path") or ""),
            "chain_id": str(visual.get("chain_id") or visual.get("artifact_ref") or ""),
        }
    return {}


def _visual_chain_uncompiled_step_count(blackboard: dict[str, Any], visual_chain: dict[str, Any]) -> int:
    candidate_count = _visual_chain_compilable_step_count(visual_chain)
    if candidate_count <= 0:
        return 0
    source_key = _source_key(visual_chain)
    if not source_key:
        if _failed_action_seen(blackboard, "compile_exact_literature_rows"):
            return 0
        compiled_total = len(_compiled_source_detail_step_ids(blackboard))
        if compiled_total > 0:
            return max(0, candidate_count - compiled_total)
        ordinal = _visual_candidate_chain_ordinal(blackboard, visual_chain)
        if ordinal and _useful_action_count(blackboard, "compile_exact_literature_rows") >= ordinal:
            return 0
        return candidate_count
    compiled_for_source = _compiled_exact_row_count_by_source(blackboard).get(source_key, 0)
    if compiled_for_source <= 0:
        ordinal = _visual_candidate_chain_ordinal(blackboard, visual_chain)
        if ordinal and _useful_action_count(blackboard, "compile_exact_literature_rows") >= ordinal:
            return 0
    return max(0, candidate_count - compiled_for_source)


def _visual_chain_candidate_step_count(visual_chain: dict[str, Any]) -> int:
    for field in ("candidate_step_count", "step_count"):
        try:
            count = int(visual_chain.get(field) or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count
    steps = visual_chain.get("steps")
    if isinstance(steps, list):
        return len([step for step in steps if isinstance(step, dict)])
    return 0


def _visual_chain_compilable_step_count(visual_chain: dict[str, Any]) -> int:
    if _visual_chain_is_exploratory_only(visual_chain):
        return 0
    candidate_count = _visual_chain_candidate_step_count(visual_chain)
    if candidate_count <= 0:
        return 0
    condition_gap_labels = {
        str(label).strip()
        for label in visual_chain.get("condition_gap_labels") or []
        if str(label).strip()
    }
    if not condition_gap_labels:
        return candidate_count
    return max(0, candidate_count - len(condition_gap_labels))


def _visual_chain_is_exploratory_only(visual_chain: dict[str, Any]) -> bool:
    if bool(visual_chain.get("exact_ready")):
        return False
    acceptance = str(visual_chain.get("acceptance_level") or "").lower()
    if bool(visual_chain.get("exploratory_accepted")) or "exploratory" in acceptance:
        return True
    for step in visual_chain.get("steps") or []:
        if not isinstance(step, dict):
            continue
        allowed_use = str(step.get("allowed_use") or "").lower()
        if (
            bool(step.get("not_exact_literature_segment"))
            or "exploratory" in allowed_use
            or allowed_use == "exploratory_template_and_guided_hint_only"
        ):
            return True
    return False


def _compiled_exact_row_count_by_source(blackboard: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in (blackboard.get("literature_evidence") or {}).get("exact_rows") or []:
        if not isinstance(row, dict):
            continue
        key = _source_key(row)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _compiled_source_detail_step_ids(blackboard: dict[str, Any]) -> set[str]:
    step_ids: set[str] = set()
    for row in (blackboard.get("literature_evidence") or {}).get("exact_rows") or []:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("row_id") or row.get("source_template_id") or row.get("template_id") or "")
        prefix = "source_detail_exact_step:"
        if row_id.startswith(prefix) and row_id[len(prefix):]:
            step_ids.add(row_id[len(prefix):])
    return step_ids


def _visual_candidate_chain_ordinal(blackboard: dict[str, Any], visual_chain: dict[str, Any]) -> int:
    target_key = _source_key(visual_chain)
    target_chain = str(visual_chain.get("chain_id") or visual_chain.get("artifact_ref") or "")
    ordinal = 0
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        visual = dict(row)
        if _visual_chain_candidate_step_count(visual) <= 0:
            continue
        ordinal += 1
        row_key = _source_key(visual)
        row_chain = str(visual.get("chain_id") or visual.get("artifact_ref") or "")
        if (target_key and row_key == target_key) or (target_chain and row_chain == target_chain):
            return ordinal
    return 0


def _latest_visual_source(blackboard: dict[str, Any]) -> dict[str, Any]:
    visual_rows = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []
        if isinstance(row, dict)
    ]
    if not visual_rows:
        return {}
    latest = visual_rows[-1]
    key = _source_key(latest)
    for candidate in _local_pdf_source_candidates(blackboard):
        if key and _source_key(candidate) == key:
            return candidate
    return {
        "source_ref": str(latest.get("source_ref") or ""),
        "title": str(latest.get("source_title") or ""),
        "local_pdf": str(latest.get("source_pdf_path") or ""),
    }


def _literature_extraction_pending(blackboard: dict[str, Any], actions: list[dict[str, Any]]) -> bool:
    if _round_has_any_action(
        actions,
        {
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "resolve_literature_structure_task",
            "compile_exact_literature_rows",
        },
    ):
        return True
    if _next_local_pdf_source_for_pdf_extraction(blackboard) and _budget_remaining(blackboard, "visual_calls"):
        return True
    if _next_local_pdf_source_for_visual_extraction(blackboard) and _budget_remaining(blackboard, "visual_calls"):
        return True
    if (
        _source_candidates_available(blackboard)
        and _pdf_structure_evidence_available(blackboard)
        and not _visual_chain_available(blackboard)
        and not _action_seen(blackboard, "extract_visual_literature_chain")
        and _budget_remaining(blackboard, "visual_calls")
    ):
        return True
    if (
        _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _visual_gap_repair_needed(blackboard)
        and _visual_gap_repair_budget_remaining(blackboard)
        and (_condition_gap_repair_needed(blackboard) or not _uncompiled_visual_steps_available(blackboard))
            and _budget_remaining(blackboard, "visual_calls")
    ):
        return True
    if _next_structure_resolution_task_for_local_resolve(blackboard) and _budget_remaining(blackboard, "visual_calls"):
        return True
    if (
        _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _uncompiled_visual_steps_available(blackboard)
        and _budget_remaining(blackboard, "visual_calls")
    ):
        return True
    return False


def _visual_extraction_payload_from_blackboard(
    blackboard: dict[str, Any],
    *,
    source_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(row, dict)
    ]
    selected = dict(source_candidate or (candidates[0] if candidates else {}))
    payload: dict[str, Any] = _source_candidate_payload(selected)
    labels = _expected_labels_from_source_candidates(blackboard, source_ref=str(selected.get("source_ref") or ""))
    if labels:
        payload["expected_labels"] = labels
        payload["route_sequence_hint"] = " ".join(
            part
            for part in [
                str(payload.get("route_sequence_hint") or ""),
                "Extract a contiguous source-detail chain covering the expected labels when visible. "
                "If exact stereochemistry is unclear but connectivity/protecting groups are visible, output RDKit-valid achiral/connectivity-only SMILES "
                "with not_exact_literature_segment=true and allowed_use=exploratory_template_and_guided_hint_only. "
                "Use extraction_gaps only when connectivity or protecting groups are not visible even at exploratory level.",
            ]
            if part
        )
    return payload


def _focused_visual_repair_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    gap_labels = _visual_gap_labels(blackboard)
    source = _visual_gap_source_candidate(blackboard)
    source_ref = str(source.get("source_ref") or "")
    expected = _expected_labels_from_source_candidates(blackboard, source_ref=source_ref)
    labels = _dedupe([*gap_labels, *expected])
    payload = _visual_extraction_payload_from_blackboard(blackboard, source_candidate=source)
    if labels:
        payload["expected_labels"] = labels
    payload["focused_gap_repair"] = True
    payload["repair_attempt"] = _focused_visual_repair_attempts(blackboard) + 1
    payload["route_sequence_hint"] = (
        "Focused repair: re-inspect the current PDF images for the missing source-detail labels "
        f"{', '.join(gap_labels) or 'recorded extraction gaps'}. "
        "For labels already covered by valid structures, repair missing condition_candidate fields from visible scheme/table text. "
        "For unresolved labels, provide RDKit-valid achiral/connectivity-only SMILES when connectivity is visible; mark those as exploratory and not exact. "
        "Keep only labels with no visible connectivity/protecting-group assignment in extraction_gaps."
    )
    return payload


def _visual_gap_source_candidate(blackboard: dict[str, Any]) -> dict[str, Any]:
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        if not (row.get("gap_labels") or row.get("extraction_gaps") or row.get("missing_expected_labels")):
            continue
        key = _source_key(row)
        for candidate in _local_pdf_source_candidates(blackboard):
            if key and _source_key(candidate) == key:
                return candidate
        return {
            "source_ref": str(row.get("source_ref") or ""),
            "title": str(row.get("source_title") or ""),
            "local_pdf": str(row.get("source_pdf_path") or ""),
        }
    return {}


def _expected_labels_from_source_candidates(blackboard: dict[str, Any], *, source_ref: str = "") -> list[str]:
    labels: list[str] = []
    for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []:
        if not isinstance(row, dict):
            continue
        if source_ref and _source_key(row) != _source_key({"source_ref": source_ref}):
            continue
        labels.extend(str(item) for item in row.get("expected_scheme_or_compound_labels") or [] if str(item or "").strip())
    return _dedupe(labels)


def _structure_resolution_scout_needed(blackboard: dict[str, Any]) -> bool:
    if _next_structure_resolution_task_for_local_resolve(blackboard) and _budget_remaining(blackboard, "visual_calls"):
        return False
    if (
        _budget_remaining(blackboard, "visual_calls")
        and (_next_local_pdf_source_for_pdf_extraction(blackboard) or _next_local_pdf_source_for_visual_extraction(blackboard))
    ):
        return False
    if _uncompiled_visual_steps_available(blackboard):
        return False
    if _structure_resolution_scout_seen(blackboard):
        return False
    return bool(_open_structure_resolution_tasks(blackboard))


def _open_structure_resolution_tasks(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("structure_resolution_tasks") or []
        if isinstance(row, dict) and str(row.get("status") or "open") == "open"
    ]


def _next_structure_resolution_task_for_local_resolve(blackboard: dict[str, Any]) -> dict[str, Any]:
    if _next_local_pdf_source_for_pdf_extraction(blackboard) or _next_local_pdf_source_for_visual_extraction(blackboard):
        return {}
    if _uncompiled_visual_steps_available(blackboard):
        return {}
    for task in _open_structure_resolution_tasks(blackboard):
        if _structure_resolution_task_locally_attempted(blackboard, task):
            continue
        return task
    return {}


def _structure_resolution_task_locally_attempted(blackboard: dict[str, Any], task: dict[str, Any]) -> bool:
    try:
        if int(task.get("resolution_attempt_count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        return False
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("action_type") or "") != "resolve_literature_structure_task":
            continue
        if task_id and task_id in str(row.get("action_signature") or ""):
            return True
    return False


def _structure_resolution_task_payload(blackboard: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    source = _source_candidate_for_structure_task(blackboard, task)
    payload = _source_candidate_payload(source) if source else {}
    label = str(task.get("label") or "").strip()
    payload.update(
        {
            "schema_version": "literature_structure_resolution_payload.v1",
            "task_id": str(task.get("task_id") or ""),
            "label": label,
            "compound_label": label,
            "source_ref": str(task.get("source_ref") or payload.get("source_ref") or ""),
            "source_title": str(task.get("source_title") or payload.get("source_title") or payload.get("title") or ""),
            "source_locator": str(task.get("source_locator") or ""),
            "artifact_ref": str(task.get("artifact_ref") or payload.get("artifact_ref") or ""),
            "run_visual": True,
            "compress_images": True,
            "max_images": int(payload.get("max_images") or 6),
            "visual_max_side_px": int(payload.get("visual_max_side_px") or 1400),
            "visual_jpeg_quality": int(payload.get("visual_jpeg_quality") or 70),
            "no_solved_claim": True,
        }
    )
    hint_parts = [
        str(payload.get("route_sequence_hint") or ""),
        f"Focused structure resolution for {label}. Use only the exact drawn/source-grounded structure for this label.",
    ]
    if task.get("source_locator"):
        hint_parts.append(f"Prior locator: {task.get('source_locator')}.")
    payload["route_sequence_hint"] = " ".join(part for part in hint_parts if str(part or "").strip())
    return payload


def _source_candidate_for_structure_task(blackboard: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    task_key = _source_key(task)
    for candidate in _local_pdf_source_candidates(blackboard):
        if task_key and _source_key(candidate) == task_key:
            return candidate
    for candidate in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []:
        if isinstance(candidate, dict) and task_key and _source_key(candidate) == task_key:
            return dict(candidate)
    source_ref = str(task.get("source_ref") or "").strip()
    if source_ref:
        return {"source_ref": source_ref, "title": str(task.get("source_title") or "")}
    return {}


def _structure_resolution_scout_seen(blackboard: dict[str, Any]) -> bool:
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("action_type") or "") != "search_literature":
            continue
        if "focused_structure_resolution" in str(row.get("action_signature") or ""):
            return True
    return False


def _structure_resolution_scout_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    tasks = _open_structure_resolution_tasks(blackboard)[:8]
    labels = _dedupe([str(row.get("label") or "") for row in tasks if str(row.get("label") or "").strip()])
    source_refs = _dedupe([str(row.get("source_ref") or "") for row in tasks if str(row.get("source_ref") or "").strip()])
    target = dict(blackboard.get("target_profile") or {})
    target_name = str(target.get("target_name") or "target")
    queries: list[str] = []
    for ref in source_refs[:3]:
        for label in labels[:4]:
            queries.append(f"{ref} {label} supplementary information structure")
            queries.append(f"{ref} {label} SMILES compound")
    if not queries and labels:
        queries.extend(f"{target_name} {label} structure SMILES supplementary" for label in labels[:6])
    payload = _literature_search_payload(blackboard, intent="structure_resolution_for_visual_gaps")
    payload.update({
        "focused_structure_resolution": True,
        "max_sources": 3,
        "structure_resolution_task_ids": [str(row.get("task_id") or "") for row in tasks],
        "queries": _dedupe(queries)[:8],
        "search_queries": _dedupe(queries)[:8],
        "reasons": ["visual_structure_gaps_require_nonvisual_resolution"],
    })
    return payload


def _visual_gap_repair_needed(blackboard: dict[str, Any]) -> bool:
    if _condition_gap_repair_needed(blackboard):
        return True
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        try:
            step_count = int(row.get("candidate_step_count") or row.get("step_count") or 0)
        except (TypeError, ValueError):
            step_count = 0
        if step_count > 0 and (row.get("gap_labels") or row.get("missing_expected_labels")):
            return True
    return False


def _condition_gap_repair_needed(blackboard: dict[str, Any]) -> bool:
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if isinstance(row, dict) and row.get("condition_gap_labels"):
            return True
    return False


def _visual_gap_repair_budget_remaining(blackboard: dict[str, Any]) -> bool:
    # One focused repair is useful; repeated repairs on the same image tend to
    # re-emit the same structure gaps and block template/guided branches.
    return _focused_visual_repair_attempts(blackboard) < 1


def _visual_gap_labels(blackboard: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        labels.extend(str(item) for item in row.get("missing_expected_labels") or [] if str(item or "").strip())
        labels.extend(str(item) for item in row.get("condition_gap_labels") or [] if str(item or "").strip())
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


def _target_relevant_exact_rows_available(blackboard: dict[str, Any]) -> bool:
    return bool(_target_relevant_exact_literature_rows(blackboard))


def _target_relevant_exact_literature_rows(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("exact_rows") or []
        if isinstance(row, dict) and _exact_row_relevant_for_parent_bridge(row)
    ]


def _exact_row_relevant_for_parent_bridge(row: dict[str, Any]) -> bool:
    if "target_relevant_for_parent_bridge" in row:
        return bool(row.get("target_relevant_for_parent_bridge"))
    relevance = row.get("target_relevance")
    if isinstance(relevance, dict) and "target_relevant_for_parent_bridge" in relevance:
        return bool(relevance.get("target_relevant_for_parent_bridge"))
    # Legacy rows predate the relevance audit. Treat them as usable so old
    # artifacts and tests keep their existing behavior until recompiled.
    return True


def _disconnected_exact_literature_row_ids(blackboard: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for row in (blackboard.get("literature_evidence") or {}).get("exact_rows") or []:
        if not isinstance(row, dict) or _exact_row_relevant_for_parent_bridge(row):
            continue
        row_id = str(row.get("row_id") or row.get("source_template_id") or row.get("template_id") or "").strip()
        if row_id:
            ids.append(row_id)
    return _dedupe(ids)


def _source_detail_one_step_rows_ready(blackboard: dict[str, Any]) -> bool:
    if _exact_rows_available(blackboard):
        return True
    policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    return int(policy.get("validated_one_step_row_count") or 0) > 0


def _exact_literature_segment_usable(blackboard: dict[str, Any]) -> bool:
    """True only when exact rows form an accepted source-detail segment."""
    if _parent_proof_accepted(blackboard):
        return True
    evidence = dict(blackboard.get("literature_evidence") or {})
    for audit in evidence.get("exact_chain_audits") or []:
        if isinstance(audit, dict) and bool(audit.get("accepted")):
            return True
    return False


def _hypotheses_available(blackboard: dict[str, Any]) -> bool:
    return bool(
        (blackboard.get("target_side_disconnection_hypotheses") or {}).get("hypotheses")
        or blackboard.get("analogical_hypotheses")
    )


def _analogical_templates_enabled(blackboard: dict[str, Any]) -> bool:
    policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    return bool(policy.get("enabled", True))


def _analogical_template_sources_available(blackboard: dict[str, Any]) -> bool:
    if _hypotheses_available(blackboard):
        return True
    candidates = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(row, dict)
    ]
    return any(
        any(token in " ".join(str(row.get(key) or "") for key in ("source_type", "relevance_rationale", "title")).lower() for token in ("analog", "precedent", "family", "reaction"))
        for row in candidates
    )


def _ranked_analogical_templates_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("analogical_template_ranking") or {}).get("selected_templates"))


def _template_applications_available(blackboard: dict[str, Any]) -> bool:
    return bool(blackboard.get("template_applications"))


def _template_applications_need_validation(blackboard: dict[str, Any]) -> bool:
    policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    if int(policy.get("validated_one_step_row_count") or 0) > 0:
        return False
    return any(
        isinstance(row, dict)
        and bool(row.get("accepted"))
        and bool(row.get("executable_candidate_available"))
        for row in blackboard.get("template_applications") or []
    )


def _analogical_template_work_pending(blackboard: dict[str, Any], actions: list[dict[str, Any]]) -> bool:
    if _round_has_any_action(
        actions,
        {
            "extract_analogical_reaction_templates",
            "rank_analogical_reaction_templates",
            "apply_analogical_template_to_target",
            "validate_template_application",
        },
    ):
        return True
    if not _analogical_templates_enabled(blackboard):
        return False
    if _exact_literature_segment_usable(blackboard):
        return False
    if (
        _analogical_template_sources_available(blackboard)
        and not blackboard.get("analogical_templates")
        and not _failed_action_seen(blackboard, "extract_analogical_reaction_templates")
    ):
        return True
    if blackboard.get("analogical_templates") and not blackboard.get("analogical_template_ranking"):
        return True
    if _ranked_analogical_templates_available(blackboard) and not _template_applications_available(blackboard):
        return True
    if _template_applications_need_validation(blackboard):
        return True
    return False


def _analogical_template_payload(blackboard: dict[str, Any], *, action_type: str = "analogical_template_action") -> dict[str, Any]:
    policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    return {
        "max_templates": min(12, max(1, int(policy.get("max_template_applications_per_round") or 5) * 2)),
        "max_applications": max(1, int(policy.get("max_template_applications_per_round") or 5)),
        "template_radius_policy": str(policy.get("template_radius_policy") or "auto"),
        "analog_template_confidence_threshold": str(policy.get("analog_template_confidence_threshold") or "medium"),
        "analogical_template_policy": {
            "schema_version": "agentic_analogical_template_action_policy.v1",
            "action_type": str(action_type or "analogical_template_action"),
            "analogy_is_advisory_only": True,
            "no_solved_claim": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "production_write_blocked": True,
            "raw_reaction_output_allowed": False,
            "final_verdict_authority": "deterministic_parent_route_proof",
            "allowed_use": ["planner_priority", "guided_policy_hint", "template_candidate_validation"],
            "deterministic_template_validation_required": True,
        },
    }


def _deterministic_route_action_ready(blackboard: dict[str, Any]) -> bool:
    return _can_stitch_parent_route(blackboard) or _can_expand_child_target(blackboard)


def _guided_failure_requires_new_signal(blackboard: dict[str, Any]) -> bool:
    failure_text = " ".join(_blackboard_failure_reasons(blackboard)).lower()
    if not failure_text:
        return False
    blocking_tokens = (
        "large_atom_jump",
        "large atom jump",
        "unexplained_large_atom_jump",
        "literature_template_plugin_not_invoked",
        "template_plugin_not_invoked",
        "plugin_product_hits=0",
        "plugin_product_hits_zero",
    )
    return any(token in failure_text for token in blocking_tokens)


def _new_failure_evidence_since_last_critic(blackboard: dict[str, Any]) -> bool:
    if not _failure_evidence_available(blackboard):
        return False
    last_critic = _last_action_round(blackboard, "build_failure_critic_report")
    if last_critic <= 0:
        return True
    failure_producing_actions = {
        "run_guided_chemenzy",
        "expand_child_target",
        "stitch_parent_route",
        "compile_exact_literature_rows",
        "validate_template_application",
    }
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        try:
            round_index = int(row.get("round_index") or 0)
        except (TypeError, ValueError):
            round_index = 0
        if round_index <= last_critic:
            continue
        if str(row.get("action_type") or "") not in failure_producing_actions:
            continue
        if row.get("reasons") or row.get("useful_artifact"):
            return True
    return False


def _new_strong_guided_signal_since_last_run(blackboard: dict[str, Any]) -> bool:
    last_round = _last_action_round(blackboard, "run_guided_chemenzy")
    if last_round <= 0:
        return True
    if not (
        _target_relevant_exact_rows_available(blackboard)
        or _validated_template_rows_available(blackboard)
        or _accepted_template_applications_available(blackboard)
        or _visual_exploratory_hints_available(blackboard)
        or _literature_terminal_candidates(blackboard)
        or blackboard.get("broad_transform_templates")
    ):
        return False
    signal_actions = {
        "compile_exact_literature_rows",
        "validate_template_application",
        "apply_analogical_template_to_target",
        "extract_analogical_reaction_templates",
        "rank_analogical_reaction_templates",
        "derive_broad_reaction_template",
        "extract_visual_literature_chain",
        "expand_child_target",
    }
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or not row.get("useful_artifact"):
            continue
        try:
            round_index = int(row.get("round_index") or 0)
        except (TypeError, ValueError):
            round_index = 0
        if round_index > last_round and str(row.get("action_type") or "") in signal_actions:
            return True
    return False


def _last_action_round(blackboard: dict[str, Any], action_type: str) -> int:
    rounds: list[int] = []
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != action_type:
            continue
        try:
            rounds.append(int(row.get("round_index") or 0))
        except (TypeError, ValueError):
            continue
    return max(rounds) if rounds else 0


def _validated_template_rows_available(blackboard: dict[str, Any]) -> bool:
    policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    return int(policy.get("validated_one_step_row_count") or 0) > 0


def _accepted_template_applications_available(blackboard: dict[str, Any]) -> bool:
    return any(
        isinstance(row, dict)
        and bool(row.get("accepted"))
        and bool(row.get("executable_candidate_available"))
        for row in blackboard.get("template_applications") or []
    )


def _visual_exploratory_hints_available(blackboard: dict[str, Any]) -> bool:
    return bool(_visual_exploratory_hints_for_policy(blackboard))


def _can_run_guided_chemenzy(blackboard: dict[str, Any]) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("chemenzy_runs") or 0) >= int(budget.get("max_chemenzy_runs") or 1):
        return False
    if _semisynthesis_anchor_validation_needed(blackboard):
        return False
    if _can_stitch_parent_route(blackboard):
        return False
    if _action_count(blackboard, "run_guided_chemenzy") > 0 and _guided_failure_requires_new_signal(blackboard):
        return _new_strong_guided_signal_since_last_run(blackboard)
    template_rows = int(((blackboard.get("current_belief") or {}).get("template_policy") or {}).get("validated_one_step_row_count") or 0)
    return bool(
        blackboard.get("bridge_tasks")
        or _target_relevant_exact_rows_available(blackboard)
        or blackboard.get("analogical_hypothesis_ranking")
        or template_rows
        or _visual_exploratory_hints_available(blackboard)
        or blackboard.get("broad_transform_templates")
    )



def _can_expand_child_target(blackboard: dict[str, Any]) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("child_target_runs") or 0) >= int(budget.get("max_child_target_runs") or 2):
        return False
    if _semisynthesis_anchor_validation_needed(blackboard):
        return False
    if _can_stitch_parent_route(blackboard):
        return False
    unattempted_terminals = _unattempted_literature_terminal_candidates(blackboard)
    unattempted_hypothetical = _unattempted_hypothetical_precursor_candidates(blackboard)
    if _child_expansion_repeated_terminal_blocked(blackboard) and not (unattempted_terminals or unattempted_hypothetical):
        return False
    if unattempted_terminals:
        return True
    if unattempted_hypothetical:
        return True
    return False


def _can_stitch_parent_route(blackboard: dict[str, Any]) -> bool:
    belief = dict(blackboard.get("current_belief") or {})
    evidence = dict(blackboard.get("literature_evidence") or {})
    stitch_count = _action_count(blackboard, "stitch_parent_route")
    child_count = _action_count(blackboard, "expand_child_target")
    target_relevant_rows = _target_relevant_exact_literature_rows(blackboard)
    if belief.get("child_route_solved") and target_relevant_rows and stitch_count < max(1, child_count):
        return True
    if target_relevant_rows and child_count and stitch_count < child_count:
        return True
    parent_artifacts_attempted = _action_seen(blackboard, "run_guided_chemenzy") or _action_seen(blackboard, "expand_child_target")
    return bool(
        stitch_count == 0
        and parent_artifacts_attempted
        and (evidence.get("visual_chains") or target_relevant_rows or blackboard.get("route_failures"))
    )


def _literature_terminal_expansion_pending(blackboard: dict[str, Any]) -> bool:
    return bool(_literature_terminal_candidates(blackboard) and not _action_seen(blackboard, "expand_child_target"))


def _literature_terminal_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("terminal_candidates") or []
        if isinstance(row, dict) and str(row.get("smiles") or "").strip()
    ]
    for task in blackboard.get("bridge_tasks") or []:
        if not isinstance(task, dict):
            continue
        terminal = task.get("terminal")
        if isinstance(terminal, dict) and str(terminal.get("smiles") or "").strip():
            row = {**dict(terminal), "source_bridge_task_id": str(task.get("task_id") or "")}
            candidates.append(row)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in candidates:
        key = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _hypothetical_precursor_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for task in blackboard.get("recursive_hypothesis_tasks") or []:
        if not isinstance(task, dict):
            continue
        smiles = str(task.get("precursor_smiles") or "").strip()
        if not smiles:
            continue
        candidates.append(
            {
                "schema_version": "agentic_recursive_hypothesis_child_target.v1",
                "name": str(task.get("name") or task.get("variant_type") or f"recursive_hypothesis_{len(candidates) + 1}"),
                "smiles": smiles,
                "source": "recursive_hypothesis_task",
                "recursive_hypothesis_task_id": str(task.get("task_id") or ""),
                "parent_candidate_id": str(task.get("parent_candidate_id") or ""),
                "parent_subgoal_name": str(task.get("parent_subgoal_name") or ""),
                "parent_smiles": str(task.get("parent_smiles") or ""),
                "recursive_depth": int(task.get("recursive_depth") or 1),
                "operation_idea": str(task.get("operation_idea") or ""),
                "variant_type": str(task.get("variant_type") or ""),
                "failure_reasons": [str(item) for item in task.get("failure_reasons") or [] if str(item or "").strip()],
                "risk_flags": [str(item) for item in task.get("risk_flags") or [] if str(item or "").strip()],
                "allowed_use": "route_expansion_subgoal_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "child_route_cannot_promote_parent": True,
                "no_solved_claim": True,
            }
        )
    for application in blackboard.get("template_applications") or []:
        if not isinstance(application, dict) or not application.get("accepted"):
            continue
        app_id = str(application.get("application_id") or "")
        template_id = str(application.get("template_id") or "")
        retron = str(application.get("product_retron_type") or "")
        for hint in application.get("hypothetical_precursor_hints") or []:
            if not isinstance(hint, dict):
                continue
            smiles = str(hint.get("precursor_smiles") or "").strip()
            if not smiles:
                continue
            candidates.append(
                {
                    "schema_version": "agentic_hypothetical_precursor_child_target.v1",
                    "name": str(hint.get("precursor_role") or f"hypothetical_precursor_{len(candidates) + 1}"),
                    "smiles": smiles,
                    "source": "analogical_hypothesis_precursor_hint",
                    "application_id": app_id,
                    "template_id": template_id,
                    "derived_from_retron": str(hint.get("derived_from_retron") or retron),
                    "hypothesis_type": str(hint.get("hypothesis_type") or ""),
                    "precursor_role": str(hint.get("precursor_role") or ""),
                    "risk_flags": [str(item) for item in hint.get("risk_flags") or [] if str(item or "").strip()],
                    "allowed_use": "route_expansion_subgoal_hint_only",
                    "not_exact_literature_segment": True,
                    "not_parent_route_proof": True,
                    "requires_verifier": True,
                    "no_solved_claim": True,
                }
            )
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in candidates:
        key = str(row.get("smiles") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _unattempted_literature_terminal_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    attempted = set(_attempted_child_target_smiles(blackboard))
    return [
        row
        for row in _literature_terminal_candidates(blackboard)
        if _child_target_smiles(row) not in attempted
    ]


def _unattempted_hypothetical_precursor_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    attempted = set(_attempted_child_target_smiles(blackboard))
    return [
        row
        for row in _hypothetical_precursor_candidates(blackboard)
        if _child_target_smiles(row) not in attempted
    ]


def _attempted_child_target_smiles(blackboard: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != "expand_child_target":
            continue
        values.extend(_terminal_smiles_from_action_signature(str(row.get("action_signature") or "")))
    return _dedupe(values)


def _child_target_smiles(row: dict[str, Any]) -> str:
    return str(row.get("canonical_smiles") or row.get("smiles") or "").strip()


def _round_has_action(actions: list[dict[str, Any]], action_type: str) -> bool:
    return any(str(row.get("action_type") or "") == action_type for row in actions)


def _round_has_any_action(actions: list[dict[str, Any]], action_types: set[str]) -> bool:
    return any(str(row.get("action_type") or "") in action_types for row in actions)


def _action_seen(blackboard: dict[str, Any], action_type: str) -> bool:
    return any(str(row.get("action_type") or "") == action_type for row in blackboard.get("action_history") or [] if isinstance(row, dict))


def _failed_action_seen(blackboard: dict[str, Any], action_type: str) -> bool:
    return any(
        str(row.get("action_type") or "") == action_type
        and (str(row.get("status") or "") == "rejected" or row.get("stale"))
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict)
    )


def _action_count(blackboard: dict[str, Any], action_type: str) -> int:
    return sum(
        1
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict) and str(row.get("action_type") or "") == action_type
    )


def _useful_action_count(blackboard: dict[str, Any], action_type: str) -> int:
    return sum(
        1
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict)
        and str(row.get("action_type") or "") == action_type
        and bool(row.get("useful_artifact"))
    )


def _child_expansion_repeated_terminal_blocked(blackboard: dict[str, Any]) -> bool:
    failures_by_terminal: dict[str, int] = {}
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != "expand_child_target":
            continue
        reasons = {str(item) for item in row.get("reasons") or [] if str(item or "").strip()}
        if "no_route_expansion_subgoal_verified_solved" not in reasons and str(row.get("status") or "") != "rejected":
            continue
        for terminal in _terminal_smiles_from_action_signature(str(row.get("action_signature") or "")):
            failures_by_terminal[terminal] = failures_by_terminal.get(terminal, 0) + 1
    return any(count >= 2 for count in failures_by_terminal.values())


def _terminal_smiles_from_action_signature(signature: str) -> list[str]:
    try:
        payload = dict(json.loads(signature or "{}").get("payload") or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    out: list[str] = []
    for target in payload.get("subgoal_targets") or []:
        if not isinstance(target, dict):
            continue
        text = str(target.get("smiles") or "").strip()
        if text:
            out.append(text)
    return _dedupe(out)


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
    payload: dict[str, Any] = build_guided_chemenzy_payload_from_blackboard(blackboard)
    payload.update({
        "rerun_attempt": attempt,
        "failure_mode_focus": failures[:6],
    })
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


def _terminal_blacklist_from_blackboard(blackboard: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for row in blackboard.get("terminal_blacklist") or []:
        if isinstance(row, dict):
            values.append(str(row.get("canonical_smiles") or row.get("smiles") or ""))
        else:
            values.append(str(row or ""))
    return _dedupe([item for item in values if str(item or "").strip()])


def _child_expansion_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    attempt = _action_count(blackboard, "expand_child_target") + 1
    payload: dict[str, Any] = {
        "expansion_attempt": attempt,
        "target_offset": max(0, (attempt - 1) * 2),
        "max_targets": 2,
        "search_preset": "thorough",
        "chem_enzy_iterations": min(200, 75 + 25 * attempt),
        "chem_enzy_expansion_topk": min(300, 120 + 30 * attempt),
        "max_unexplained_heavy_atom_jump": 12,
    }
    terminal_blacklist = _terminal_blacklist_from_blackboard(blackboard)
    if terminal_blacklist:
        payload["terminal_blacklist"] = terminal_blacklist
    terminals = _unattempted_literature_terminal_candidates(blackboard)
    if terminals:
        payload["target_offset"] = 0
        payload["max_targets"] = min(2, len(terminals))
        payload["subgoal_targets"] = [
            {
                "name": str(row.get("name") or "source detail literature terminal"),
                "smiles": str(row.get("smiles") or ""),
                "exact_target_override": True,
                "target_equivalence_audit_required": True,
                "no_solved_claim": True,
                "child_route_cannot_promote_parent": True,
                "chem_enzy_search_policy": {
                    "schema_version": "chem_enzy_search_policy.v1",
                    "policy_id": f"{blackboard.get('case_id') or 'case'}_literature_terminal_{idx}_policy",
                    "operator_id": "agentic_blackboard_controller",
                    "case_id": str(blackboard.get("case_id") or ""),
                    "evidence_refs": _child_terminal_evidence_refs(blackboard, row),
                    "terminal_blacklist": terminal_blacklist,
                    "anchor_whitelist": [str(row.get("smiles") or "")],
                    "preferred_subgoal": {
                        "schema_version": "source_detail_literature_terminal_subgoal.v1",
                        "preferred_subgoals": [str(row.get("name") or ""), str(row.get("smiles") or "")],
                        "terminal_candidate": dict(row),
                    },
                    "source_budget": {
                        "preferred_reaction_classes": ["steroid_semisynthesis", "source_detail_terminal_upstream_expansion"],
                        "exact_literature_terminal": True,
                        "max_unexplained_heavy_atom_jump": 12,
                    },
                    "rerun_reason": "explore upstream route to exact source-detail literature terminal",
                    "mode": "guided",
                    "compiler_metadata": {
                        "compiler_schema": "agentic_blackboard_literature_terminal_child_target.v1",
                        "not_raw_reaction_injection": True,
                        "requires_verifier": True,
                        "no_solved_claim": True,
                        "child_route_cannot_promote_parent": True,
                    },
                },
            }
            for idx, row in enumerate(terminals[:2], start=1)
        ]
    else:
        hypothetical_precursors = _unattempted_hypothetical_precursor_candidates(blackboard)
        if hypothetical_precursors:
            payload["target_offset"] = 0
            payload["max_targets"] = min(2, len(hypothetical_precursors))
            payload["subgoal_targets"] = [
                {
                    "name": str(row.get("name") or f"hypothetical precursor {idx}"),
                    "smiles": str(row.get("smiles") or ""),
                    "exact_target_override": True,
                    "target_equivalence_audit_required": True,
                    "no_solved_claim": True,
                    "child_route_cannot_promote_parent": True,
                    "hypothesis_only_not_solved": True,
                    "source": str(row.get("source") or "analogical_hypothesis_precursor_hint"),
                    "template_id": str(row.get("template_id") or ""),
                    "application_id": str(row.get("application_id") or ""),
                    "recursive_hypothesis_task_id": str(row.get("recursive_hypothesis_task_id") or ""),
                    "recursive_depth": int(row.get("recursive_depth") or 0),
                    "parent_smiles": str(row.get("parent_smiles") or ""),
                    "parent_candidate_id": str(row.get("parent_candidate_id") or ""),
                    "chem_enzy_search_policy": {
                        "schema_version": "chem_enzy_search_policy.v1",
                        "policy_id": (
                            f"{blackboard.get('case_id') or 'case'}_recursive_hypothesis_{idx}_policy"
                            if str(row.get("source") or "") == "recursive_hypothesis_task"
                            else f"{blackboard.get('case_id') or 'case'}_hypothesis_precursor_{idx}_policy"
                        ),
                        "operator_id": "agentic_blackboard_controller",
                        "case_id": str(blackboard.get("case_id") or ""),
                        "evidence_refs": _hypothetical_precursor_evidence_refs(blackboard, row),
                        "terminal_blacklist": terminal_blacklist,
                        "anchor_whitelist": [],
                        "preferred_subgoal": {
                            "schema_version": "hypothetical_precursor_subgoal.v1",
                            "preferred_subgoals": [str(row.get("name") or ""), str(row.get("smiles") or "")],
                            "hypothetical_precursor_target": dict(row),
                            "recursive_hypothesis_task": dict(row) if str(row.get("source") or "") == "recursive_hypothesis_task" else {},
                            "analogy_is_advisory_only": True,
                            "not_exact_literature_segment": True,
                            "not_parent_route_proof": True,
                            "no_solved_claim": True,
                        },
                        "source_budget": {
                            "preferred_reaction_classes": _dedupe(
                                [
                                    (
                                        "recursive_failed_hypothesis_frontier_expansion"
                                        if str(row.get("source") or "") == "recursive_hypothesis_task"
                                        else "same_core_redox_or_protection_state_precursor_search"
                                    ),
                                    str(row.get("derived_from_retron") or ""),
                                    str(row.get("variant_type") or ""),
                                ]
                            ),
                            "hypothesis_precursor_hint": True,
                            "hypothesis_precursor_hints_are_not_proof": True,
                            "recursive_hypothesis_frontier": str(row.get("source") or "") == "recursive_hypothesis_task",
                            "recursive_depth": int(row.get("recursive_depth") or 0),
                            "parent_smiles": str(row.get("parent_smiles") or ""),
                            "require_target_core_retention": True,
                            "max_unexplained_heavy_atom_jump": 12,
                        },
                        "rerun_reason": (
                            "continue recursively from failed hypothesis precursor"
                            if str(row.get("source") or "") == "recursive_hypothesis_task"
                            else "explore hypothesis-only same-core precursor as child subgoal"
                        ),
                        "mode": "guided",
                        "compiler_metadata": {
                            "compiler_schema": "agentic_blackboard_hypothesis_precursor_child_target.v1",
                            "not_raw_reaction_injection": True,
                            "requires_verifier": True,
                            "no_solved_claim": True,
                            "child_route_cannot_promote_parent": True,
                            "hypothesis_only_not_solved": True,
                            "recursive_hypothesis_frontier": str(row.get("source") or "") == "recursive_hypothesis_task",
                        },
                    },
                }
                for idx, row in enumerate(hypothetical_precursors[:2], start=1)
            ]
    return payload


def _hypothetical_precursor_evidence_refs(blackboard: dict[str, Any], row: dict[str, Any]) -> list[str]:
    refs = [
        str(row.get("recursive_hypothesis_task_id") or ""),
        str(row.get("parent_candidate_id") or ""),
        str(row.get("parent_subgoal_name") or ""),
        str(row.get("application_id") or ""),
        str(row.get("template_id") or ""),
        str(row.get("derived_from_retron") or ""),
        str(row.get("variant_type") or ""),
    ]
    refs.extend(
        str(item.get("task_id") or "")
        for item in blackboard.get("bridge_tasks") or []
        if isinstance(item, dict)
    )
    refs.extend(
        str(item.get("row_id") or item.get("source_template_id") or "")
        for item in _target_relevant_exact_literature_rows(blackboard)
        if isinstance(item, dict)
    )
    return _dedupe([item for item in refs if str(item or "").strip()]) or [str(row.get("smiles") or "hypothesis_precursor")]


def _child_terminal_evidence_refs(blackboard: dict[str, Any], row: dict[str, Any]) -> list[str]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    refs: list[str] = [
        str(row.get("source_ref") or ""),
        str(row.get("terminal_id") or ""),
        *[str(item) for item in evidence.get("source_refs") or []],
    ]
    refs.extend(
        str(item.get("row_id") or item.get("source_template_id") or "")
        for item in _target_relevant_exact_literature_rows(blackboard)
        if isinstance(item, dict)
    )
    refs.extend(
        str(item.get("task_id") or "")
        for item in blackboard.get("bridge_tasks") or []
        if isinstance(item, dict)
    )
    refs = _dedupe([item for item in refs if str(item or "").strip()])
    return refs or [str(row.get("name") or row.get("smiles") or "child_target")]


def _stitch_retry_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    binding = _stitch_parent_route_binding(blackboard)
    return {
        "stitch_attempt": _action_count(blackboard, "stitch_parent_route") + 1,
        "exact_row_count_at_attempt": _exact_row_count(blackboard),
        "child_attempt_count_at_attempt": _action_count(blackboard, "expand_child_target"),
        "guided_attempt_count_at_attempt": _action_count(blackboard, "run_guided_chemenzy"),
        "proof_binding": binding,
        "proof_policy": {
            "schema_version": "agentic_parent_stitch_policy.v1",
            "target_equivalence_required": True,
            "parent_route_verifier_required": True,
            "stock_audit_required": True,
            "no_unexplained_large_atom_jump_required": True,
            "child_route_connectivity_required": True,
            "exact_literature_connectivity_required": True,
            "analogy_is_not_proof": True,
            "child_route_cannot_promote_parent": True,
            "final_verdict_authority": "deterministic_parent_route_proof",
        },
        "analogy_refs": [
            {**dict(row), "used_as_proof": False}
            for row in ((blackboard.get("analogical_hypothesis_ranking") or {}).get("selected_hypotheses") or [])
            if isinstance(row, dict)
        ],
    }


def _stitch_parent_route_binding(blackboard: dict[str, Any]) -> dict[str, Any]:
    refs = dict(blackboard.get("artifact_refs") or {})
    evidence = dict(blackboard.get("literature_evidence") or {})
    exact_row_ids = [
        str(row.get("row_id") or row.get("source_template_id") or row.get("template_id") or "")
        for row in _target_relevant_exact_literature_rows(blackboard)
        if isinstance(row, dict) and str(row.get("row_id") or row.get("source_template_id") or row.get("template_id") or "").strip()
    ]
    child_ref = str(refs.get("route_expansion_subgoal_search") or _first_ref_containing(refs, "route_expansion") or "")
    parent_ref = str(refs.get("guided_chemenzy") or _first_ref_containing(refs, "guided_chemenzy") or _first_ref_containing(refs, "route_verifier") or "")
    exact_ref = str(
        refs.get("compile_source_detail_chain_route")
        or _first_ref_containing(refs, "compile_exact_literature_rows")
        or _first_ref_containing(refs, "compiled_source_detail")
        or ""
    )
    missing_inputs: list[str] = []
    if not child_ref:
        missing_inputs.append("child_route_ref_missing")
    if not parent_ref:
        missing_inputs.append("parent_route_ref_missing")
    if not exact_row_ids:
        missing_inputs.append("exact_literature_rows_missing")
    input_refs = _dedupe(
        [
            child_ref,
            parent_ref,
            exact_ref,
            *[f"exact_row:{row_id}" for row_id in exact_row_ids],
        ]
    )
    return {
        "schema_version": "agentic_parent_stitch_binding.v1",
        "child_route_ref": child_ref,
        "parent_route_ref": parent_ref,
        "exact_literature_segment_ref": exact_ref,
        "exact_literature_row_ids": _dedupe(exact_row_ids),
        "input_refs": [str(item) for item in input_refs if str(item or "").strip()],
        "missing_inputs": missing_inputs,
    }


def _first_ref_containing(refs: dict[str, Any], token: str) -> str:
    token_l = str(token or "").lower()
    for key, value in refs.items():
        if token_l in str(key).lower() and str(value or "").strip():
            return str(value)
    return ""


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


def _recent_unproductive_action_signatures(blackboard: dict[str, Any], *, round_count: int) -> set[str]:
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    rounds = sorted({int(row.get("round_index") or 0) for row in history if row.get("round_index")}, reverse=True)
    selected_rounds = set(rounds[: max(0, int(round_count or 0))])
    signatures: set[str] = set()
    for row in history:
        if int(row.get("round_index") or 0) not in selected_rounds:
            continue
        if row.get("useful_artifact"):
            continue
        signature = str(row.get("action_signature") or "")
        if signature:
            signatures.add(signature)
    return signatures


def _budget_remaining(blackboard: dict[str, Any], field: str) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    max_field = f"max_{field}"
    fallback = 3 if field == "scout_calls" else 1
    return int(budget.get(field) or 0) < int(budget.get(max_field) or fallback)


def _stale_action_repeated(blackboard: dict[str, Any], action: dict[str, Any]) -> bool:
    if str(action.get("action_type") or "") == "search_literature" and _stale_literature_search_repeated(blackboard):
        return True
    signature = _action_signature(action)
    stale_count = 0
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        if row.get("stale") and row.get("action_signature") == signature:
            stale_count += 1
    return stale_count > 1


def _stale_literature_search_repeated(blackboard: dict[str, Any], *, threshold: int = 2) -> bool:
    count = 0
    for row in reversed([item for item in blackboard.get("action_history") or [] if isinstance(item, dict)]):
        if str(row.get("action_type") or "") != "search_literature":
            continue
        if row.get("useful_artifact"):
            break
        if not _empty_literature_search_history_row(row):
            break
        count += 1
        if count >= max(1, int(threshold or 1)):
            return True
    return False


def _empty_literature_search_history_row(row: dict[str, Any]) -> bool:
    delta = dict(row.get("blackboard_delta") or {})
    if delta:
        productive_fields = {
            "source_candidates",
            "source_refs",
            "local_pdf_proxy_requests",
            "planner_source_hints",
        }
        return not any(int(delta.get(field) or 0) > 0 for field in productive_fields)
    reasons = {str(item) for item in row.get("reasons") or [] if str(item or "").strip()}
    return bool(reasons & {"no_source_candidates", "no_real_literature_source_found", "no_new_source_candidates"})


def _action_signature(action: dict[str, Any]) -> str:
    payload = {key: value for key, value in dict(action.get("payload") or {}).items() if key != "timestamp"}
    return json.dumps({"action_type": action.get("action_type"), "payload": payload}, sort_keys=True, default=str)


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


def _safe_id(value: str) -> str:
    text = str(value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    out = "_".join(part for part in out.split("_") if part)
    return out[:80] or "item"
