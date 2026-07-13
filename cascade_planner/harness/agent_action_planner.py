"""Policy-driven action planning for agentic blackboard runs."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, DataStructs

from cascade_planner.agent.action_contracts import (
    ACTION_BATCH_SCHEMA,
    ALLOWED_AGENT_ACTIONS,
    contains_raw_reaction_payload,
    planner_source_hint_reasons,
)
from cascade_planner.agent.chem_enzy_policy import validate_chem_enzy_search_policy
from cascade_planner.harness.parent_route_proof import is_solved_parent_route_proof
from cascade_planner.harness.deterministic_literature_registry import (
    PARSER_AUTHORITY_ID,
)
from cascade_planner.harness.source_capabilities import (
    SOURCE_SENSITIVE_ACTIONS,
    action_resource_cost,
    build_source_capability_queue,
    eligible_source_capabilities,
    matching_source_capabilities,
    meaningful_compound_labels,
    pdf_evidence_has_materialized_render,
    pdf_evidence_render_paths,
    source_capability_effective_payload,
)
from cascade_planner.source_locators import independent_source_group


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

    mode = "deterministic_policy_budget_exhaustive" if exhaust_round_budget else "deterministic_policy"

    if _direct_parent_route_proof_ready(blackboard) and _can_stitch_parent_route(blackboard):
        actions.append(
            _action(
                round_index,
                "stitch_parent_route",
                "guided parent route verifier is already solved; compile deterministic direct parent-route proof",
                "stitched_parent_route_proof.v1",
                "direct parent proof accepted or explicit verifier rejection",
                _stitch_retry_payload(blackboard),
            )
        )
        return _batch(case_id, round_index, actions[:max_actions], mode=mode)

    if _can_stitch_parent_route(blackboard):
        actions.append(
            _action(
                round_index,
                "stitch_parent_route",
                "child route and literature bridge artifacts are ready for deterministic parent connectivity proof",
                "stitched_parent_route_proof.v1",
                "parent proof accepted or explicit connectivity rejection",
                _stitch_retry_payload(blackboard),
            )
        )
        return _batch(case_id, round_index, actions[:max_actions], mode=mode)

    if _simple_direct_guided_probe_ready(blackboard):
        actions.append(
            _action(
                round_index,
                "run_guided_chemenzy",
                "simple target has no deterministic parent proof; run one bounded direct retrosynthesis probe",
                "guided_chemenzy_result plus verifier report",
                "route verifier accepts or returns actionable failure evidence",
                _direct_initial_guided_payload(blackboard),
            )
        )
        return _batch(case_id, round_index, actions[:max_actions], mode=mode)

    # Evidence acquisition is a lifecycle, not a one-shot scout action.  Give
    # the next materialized stage one reserved slot before generic hypothesis
    # and critic work can fill the round.  This makes a source discovered in
    # round N deterministically progress through PDF/render, visual/structure,
    # and exact-row compilation in later rounds.
    guided_retry_ready = (
        _action_count(blackboard, "run_guided_chemenzy") > 0
        and _can_run_guided_chemenzy(blackboard)
    )
    pending_exact_compile = bool(
        _visual_chain_available(blackboard)
        and _uncompiled_visual_steps_available(blackboard)
        and not _compile_exact_rows_exhausted_to_advisory(blackboard)
        and not _stale_compile_requires_structure_resolution(blackboard)
    )
    if (
        pending_exact_compile
        or (
            not _deterministic_route_action_ready(blackboard)
            and not guided_retry_ready
            # Structured process evidence carries an explicit route-first bias
            # only when no already-materialized exact candidate is waiting.
            and not _process_evidence_available(blackboard)
        )
    ):
        actions.extend(
            plan_literature_evidence_followup_actions(
                blackboard,
                round_index=round_index,
                max_actions=min(1, max_actions),
            )
        )

    if (
        not actions
        and _two_recent_rounds_without_useful_artifact(blackboard)
        and not exhaust_round_budget
        and not _next_local_pdf_source_for_pdf_extraction(blackboard)
        and not _next_local_pdf_source_for_visual_extraction(blackboard)
    ):
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
        and not _literature_extraction_pending(blackboard, actions)
        and not guided_retry_ready
        and _evidence_followup_search_allowed(blackboard)
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
        and not _process_evidence_available(blackboard)
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
        and _uncompiled_visual_steps_available(blackboard)
        and not _compile_exact_rows_exhausted_to_advisory(blackboard)
        and not _stale_compile_requires_structure_resolution(blackboard)
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

    if (
        not actions
        and _analogical_templates_enabled(blackboard)
        and _compile_exact_rows_exhausted_to_advisory(blackboard)
        and _broad_template_derivation_ready(blackboard)
        and not _open_structure_resolution_tasks(blackboard)
        and not _round_has_action(actions, "derive_broad_reaction_template")
    ):
        actions.append(
            _action(
                round_index,
                "derive_broad_reaction_template",
                "exact row compilation found advisory visual chemistry rather than exact rows; convert it into broad templates for guided search",
                "broad_transform_template_report.v1",
                "one or more advisory broad templates are available without solved claim",
                _broad_template_payload(blackboard),
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
        _analogical_templates_enabled(blackboard)
        and len(actions) < max_actions
        and _compile_exact_rows_exhausted_to_advisory(blackboard)
        and _broad_template_derivation_ready(blackboard)
        and (actions or not _structure_resolution_scout_needed(blackboard))
        and not _round_has_action(actions, "derive_broad_reaction_template")
    ):
        actions.append(
            _action(
                round_index,
                "derive_broad_reaction_template",
                "exact row compilation reached an advisory visual-template state; convert the visual chemistry into broad templates before more expansion",
                "broad_transform_template_report.v1",
                "one or more advisory broad templates are available without solved claim",
                _broad_template_payload(blackboard),
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
        _analogical_templates_enabled(blackboard)
        and len(actions) < max_actions
        and _compile_exact_rows_exhausted_to_advisory(blackboard)
        and _broad_template_derivation_ready(blackboard)
        and not _round_has_action(actions, "derive_broad_reaction_template")
    ):
        actions.append(
            _action(
                round_index,
                "derive_broad_reaction_template",
                "structure-resolution scouting can proceed, but advisory visual chemistry should also become broad templates in this round",
                "broad_transform_template_report.v1",
                "one or more advisory broad templates are available without solved claim",
                _broad_template_payload(blackboard),
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
        and (not blackboard.get("analogical_templates") or _analogical_templates_need_refresh(blackboard))
        and (
            _analogical_templates_need_refresh(blackboard)
            or not _failed_action_seen(blackboard, "extract_analogical_reaction_templates")
        )
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
        and (not blackboard.get("analogical_template_ranking") or _analogical_template_ranking_needs_refresh(blackboard))
        and not _round_has_action(actions, "extract_analogical_reaction_templates")
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
        and not _analogical_template_ranking_needs_refresh(blackboard)
        and not _exact_literature_segment_usable(blackboard)
        and not _template_applications_available(blackboard)
        and _budget_remaining(blackboard, "template_application_actions")
        and not _round_has_action(actions, "extract_analogical_reaction_templates")
        and not _round_has_action(actions, "rank_analogical_reaction_templates")
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
                _child_expansion_rationale(blackboard),
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
    capability_queue = build_source_capability_queue(
        board,
        round_index=int(batch.get("round_index") or 0),
        max_literature_sources_per_round=max_literature_sources_per_round,
    )
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
    host_action_costs = [
        action_resource_cost(action if isinstance(action, dict) else "")
        for action in actions
    ]
    effective_payloads = [
        dict(action.get("payload") or {}) if isinstance(action, dict) else {}
        for action in actions
    ]
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
        source_capability_eligible = False
        source_capability: dict[str, Any] = {}
        if action_type in SOURCE_SENSITIVE_ACTIONS:
            source_matches = matching_source_capabilities(
                capability_queue,
                action_type=action_type,
                payload=payload,
            )
            source_capability_eligible = len(source_matches) == 1
            if source_capability_eligible:
                source_capability = dict(source_matches[0])
                host_action_costs[idx] = dict(
                    source_capability.get("cost") or action_resource_cost(action)
                )
                effective_payloads[idx] = source_capability_effective_payload(
                    payload,
                    source_capability,
                )
                payload = effective_payloads[idx]
            # Focused visual gap repair is the one deliberate capability-queue
            # exception: its current rendered input is revalidated directly.
            if (
                action_type == "extract_visual_literature_chain"
                and bool(payload.get("focused_gap_repair"))
            ):
                source_capability_eligible = _focused_visual_gap_repair_is_eligible(
                    payload,
                    board,
                )
        if action_type == "run_guided_chemenzy":
            chemenzy_count += 1
            reasons.extend(
                f"guided_chemenzy_payload:{idx}:{reason}"
                for reason in _guided_chemenzy_payload_reasons(payload, blackboard=board)
            )
        if action_type == "expand_child_target":
            child_count += planned_child_target_count(payload)
            reasons.extend(f"child_expansion_payload:{idx}:{reason}" for reason in _child_expansion_payload_reasons(payload))
        if action_type == "stitch_parent_route":
            reasons.extend(f"stitch_parent_route_payload:{idx}:{reason}" for reason in _stitch_parent_route_payload_reasons(payload))
        if action_type == "search_literature":
            scout_action_count += 1
            reasons.extend(f"search_literature_payload:{idx}:{reason}" for reason in _search_literature_payload_reasons(payload))
        if action_type == "build_failure_critic_report" and not _failure_evidence_available(board):
            reasons.append(f"failure_critic_requires_failure_evidence:{idx}")
        if action_type == "extract_pdf_literature_structures":
            if _pdf_extraction_binding_missing(payload, board):
                reasons.append(
                    f"extract_pdf_literature_structures_requires_pdf_binding:{idx}"
                )
            elif not source_capability_eligible:
                reasons.append(
                    f"source_capability_not_eligible:{idx}:{action_type}"
                )
        if action_type == "extract_visual_literature_chain":
            visual_action_count += int(
                host_action_costs[idx].get("visual_calls") or 0
            )
            if not source_capability_eligible:
                reasons.append(
                    f"extract_visual_literature_chain_requires_rendered_pdf_evidence:{idx}"
                )
        if (
            action_type in SOURCE_SENSITIVE_ACTIONS
            and action_type
            not in {
                "extract_pdf_literature_structures",
                "extract_visual_literature_chain",
            }
            and not source_capability_eligible
        ):
            reasons.append(f"source_capability_not_eligible:{idx}:{action_type}")
        if (
            action_type in SOURCE_SENSITIVE_ACTIONS
            and str(
                payload.get("source_capability_id")
                or payload.get("capability_id")
                or ""
            ).strip()
            and not source_capability_eligible
        ):
            reasons.append(f"source_capability_not_current:{idx}:{action_type}")
        if action_type == "resolve_literature_structure_task":
            visual_action_count += int(
                host_action_costs[idx].get("visual_calls") or 0
            )
        if action_type == "resolve_literature_structure_task":
            reasons.extend(
                f"resolve_literature_structure_task_payload:{idx}:{reason}"
                for reason in _structure_resolution_payload_reasons(payload)
            )
        if action_type == "compile_exact_literature_rows" and (
            not _visual_chain_available(board)
            or (_process_evidence_available(board) and not _uncompiled_visual_steps_available(board))
            or _compile_exact_rows_exhausted_to_advisory(board)
            or (bool(board.get("broad_transform_templates")) and not _uncompiled_visual_steps_available(board))
        ):
            reasons.append(f"compile_exact_literature_rows_requires_uncompiled_visual_steps:{idx}")
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
            source_count += host_action_costs[idx].get(
                "literature_source_units",
                0,
            )
            if _source_binding_required(board, action_type) and not _payload_has_source_binding(
                payload,
                blackboard=board,
            ):
                reasons.append(f"source_sensitive_action_missing_source_binding:{idx}:{action_type}")
        effective_action = dict(action)
        effective_action["payload"] = dict(effective_payloads[idx])
        if _stale_action_repeated(board, effective_action):
            reasons.append(f"stale_action_repeated:{idx}:{action_type}")
    effective_actions = [
        {
            **(dict(action) if isinstance(action, dict) else {}),
            "payload": dict(effective_payloads[index]),
        }
        for index, action in enumerate(actions)
    ]
    if _must_stop_or_change_direction(board, effective_actions):
        reasons.append("planner_must_stop_or_change_direction_after_two_unproductive_rounds")
    if _advisory_broad_template_required(board, actions):
        reasons.append("advisory_visual_template_requires_broad_template")
    if _complex_target_frontier_bootstrap_required(board, actions):
        reasons.append("complex_target_requires_frontier_bootstrap_after_initial_probe")
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
    action_validations, batch_reasons = _partition_action_validation_reasons(
        actions,
        sorted(set(reasons)),
        host_action_costs=host_action_costs,
        effective_payloads=effective_payloads,
    )
    return {
        "schema_version": "agent_action_batch_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "action_validations": action_validations,
        "batch_reasons": batch_reasons,
        "salvage_allowed": not _unsafe_batch_validation_reasons(batch_reasons),
        "source_capability_queue_sha256": str(
            capability_queue.get("content_sha256") or ""
        ),
        "literature_source_units_max_this_round": max_literature_sources_per_round,
        "case_id": str(batch.get("case_id") or ""),
        "action_count": len(actions),
    }


_ACTION_LOCAL_REASON_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^action_not_object:(\d+)$",
        r"^unknown_action:(\d+):",
        r"^action_missing_[^:]+:(\d+)$",
        r"^(?:guided_chemenzy_payload|child_expansion_payload|stitch_parent_route_payload|search_literature_payload):(\d+)(?::|$)",
        r"^(?:failure_critic_requires_failure_evidence|extract_pdf_literature_structures_requires_pdf_binding|extract_visual_literature_chain_requires_rendered_pdf_evidence):(\d+)$",
        r"^(?:resolve_literature_structure_task_payload|compile_exact_literature_rows_requires_uncompiled_visual_steps|analogical_template_payload):(\d+)(?::|$)",
        r"^(?:source_sensitive_action_missing_source_binding|source_capability_not_eligible|source_capability_not_current|stale_action_repeated):(\d+)(?::|$)",
    )
)


def _partition_action_validation_reasons(
    actions: list[Any],
    reasons: list[str],
    *,
    host_action_costs: list[dict[str, int]],
    effective_payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    per_action: dict[int, list[str]] = {index: [] for index in range(len(actions))}
    batch_reasons: list[str] = []
    for reason in reasons:
        index: int | None = None
        for pattern in _ACTION_LOCAL_REASON_PATTERNS:
            match = pattern.match(reason)
            if match:
                index = int(match.group(1))
                break
        if index is None or index not in per_action:
            batch_reasons.append(reason)
            continue
        per_action[index].append(reason)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(actions):
        action = dict(raw) if isinstance(raw, dict) else {}
        local_reasons = sorted(set(per_action.get(index) or []))
        rows.append(
            {
                "index": index,
                "action_id": str(action.get("action_id") or f"action:{index}"),
                "action_type": str(action.get("action_type") or ""),
                "accepted": not local_reasons,
                "reasons": local_reasons,
                "cost": dict(host_action_costs[index]),
                "effective_payload": dict(effective_payloads[index]),
            }
        )
    return rows, sorted(set(batch_reasons))


def _unsafe_batch_validation_reasons(reasons: Iterable[Any]) -> bool:
    values = {str(reason) for reason in reasons}
    unsafe = {
        "planner_direct_solved_claim",
        "planner_semantics_allow_solved_claim",
        "planner_semantics_allow_raw_reaction_output",
        "raw_reaction_injection",
    }
    return bool(values & unsafe) or any(
        "raw_reaction_injection" in reason for reason in values
    )


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


def _payload_has_source_binding(
    payload: dict[str, Any],
    *,
    blackboard: dict[str, Any] | None = None,
) -> bool:
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
        "document_id",
        "chain_id",
        "visual_chain_id",
        "artifact_ref",
    }
    if not any(str(payload.get(field) or "").strip() for field in binding_fields):
        return False
    if any(
        str(payload.get(field) or "").strip()
        for field in (
            "pdf_path",
            "local_pdf",
            "source_pdf_path",
            "document_id",
            "chain_id",
            "visual_chain_id",
            "artifact_ref",
        )
    ):
        return True
    if blackboard:
        matches = _matching_local_pdf_document_keys(payload, blackboard)
        if len(matches) > 1:
            return False
    return True


def _payload_has_local_pdf_binding(payload: dict[str, Any]) -> bool:
    return any(
        str(payload.get(field) or "").strip()
        for field in ("pdf_path", "local_pdf", "source_pdf_path")
    )


def _pdf_extraction_binding_missing(payload: dict[str, Any], blackboard: dict[str, Any]) -> bool:
    if _payload_has_local_pdf_binding(payload):
        return False
    if not _payload_has_source_metadata(payload):
        return False
    local_pdf_candidates = _local_pdf_source_candidates(blackboard)
    if not local_pdf_candidates:
        return True
    return len(_matching_local_pdf_document_keys(payload, blackboard)) != 1


def _payload_has_source_metadata(payload: dict[str, Any]) -> bool:
    return any(
        str(payload.get(field) or "").strip()
        for field in ("source_ref", "doi", "pii", "url", "source_title", "title")
    )


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
        if payload.get("guided_policy_runtime_rebuild") and board:
            rebuilt = build_guided_chemenzy_payload_from_blackboard(board)
            policy = rebuilt.get("chem_enzy_search_policy") or rebuilt.get("search_policy")
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
        "source_candidates",
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
        or _source_candidates_have_real_source(blackboard)
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


def _simple_direct_guided_probe_ready(blackboard: dict[str, Any]) -> bool:
    if _action_count(blackboard, "run_guided_chemenzy") > 0:
        return False
    if blackboard.get("action_history"):
        return False
    evidence = dict(blackboard.get("literature_evidence") or {})
    if any(
        evidence.get(key)
        for key in (
            "source_candidates",
            "planner_source_hints",
            "pdf_structure_evidence",
            "visual_chains",
            "exact_rows",
            "structure_resolution_tasks",
            "process_evidence_rows",
        )
    ):
        return False
    if blackboard.get("route_failures") or blackboard.get("bridge_tasks"):
        return False
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("chemenzy_runs") or 0) >= _budget_limit(budget, "max_chemenzy_runs", default=1):
        return False
    if blackboard.get("parent_route_proof"):
        return False
    return _simple_direct_chemenzy_target(blackboard)


def _complex_target_frontier_bootstrap_required(blackboard: dict[str, Any], actions: list[dict[str, Any]]) -> bool:
    """Require proposal-frontier bootstrapping after a complex-target exploratory probe."""
    action_types = [str(row.get("action_type") or "") for row in actions if isinstance(row, dict)]
    if not action_types:
        return False
    if all(action_type == "stop_unresolved" for action_type in action_types):
        return False
    if "generate_disconnection_hypotheses" in action_types:
        return False
    if _frontier_bootstrap_available(blackboard):
        return False
    if _simple_direct_chemenzy_target(blackboard):
        return False
    if not _target_is_complex_for_frontier_bootstrap(blackboard):
        return False
    prior_exploration = any(
        _action_seen(blackboard, action_type)
        for action_type in (
            "run_guided_chemenzy",
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "compile_exact_literature_rows",
            "resolve_literature_structure_task",
            "expand_child_target",
        )
    )
    if not prior_exploration:
        return False
    exploratory_actions = {
        "search_literature",
        "extract_pdf_literature_structures",
        "extract_visual_literature_chain",
        "compile_exact_literature_rows",
        "resolve_literature_structure_task",
        "run_guided_chemenzy",
        "expand_child_target",
        "derive_broad_reaction_template",
        "extract_analogical_reaction_templates",
        "rank_analogical_reaction_templates",
        "apply_analogical_template_to_target",
        "validate_template_application",
        "stitch_parent_route",
    }
    return any(action_type in exploratory_actions for action_type in action_types)


def _frontier_bootstrap_available(blackboard: dict[str, Any]) -> bool:
    return bool(
        blackboard.get("target_side_disconnection_hypotheses")
        or blackboard.get("reaction_idea_cards")
        or _retrosynthetic_proposals_available(blackboard)
        or blackboard.get("recursive_hypothesis_tasks")
        or blackboard.get("broad_transform_templates")
    )


def _target_is_complex_for_frontier_bootstrap(blackboard: dict[str, Any]) -> bool:
    target = dict(blackboard.get("target_profile") or {})
    try:
        heavy_atoms = int(target.get("heavy_atoms") or 0)
        rings = int(target.get("rings") or 0)
        stereocenters = int(target.get("stereocenters") or 0)
    except (TypeError, ValueError):
        heavy_atoms = rings = stereocenters = 0
    hints = " ".join(
        [
            str(target.get("target_name") or ""),
            str(target.get("family_hint") or ""),
            *[str(item) for item in target.get("family_hints") or []],
            *[str(item) for item in target.get("initial_risk_flags") or []],
        ]
    ).lower()
    complex_tokens = ("steroid", "polycyclic", "macrocycle", "peptide", "glycoside", "natural product")
    return bool(
        rings >= 3
        or heavy_atoms >= 24
        or stereocenters >= 4
        or any(token in hints for token in complex_tokens)
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


def _bounded_int_env(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw not in (None, "") else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(int(lo), min(int(hi), value))


def _guided_retry_runtime_budget() -> dict[str, int]:
    return {
        "max_steps": _bounded_int_env("AUTOPLANNER_GUIDED_RETRY_MAX_STEPS", 12, lo=1, hi=20),
        "iterations": _bounded_int_env("AUTOPLANNER_GUIDED_RETRY_ITERATIONS", 60, lo=1, hi=200),
        "expansion_topk": _bounded_int_env("AUTOPLANNER_GUIDED_RETRY_TOPK", 120, lo=1, hi=300),
        "timeout_s": _bounded_int_env("AUTOPLANNER_GUIDED_RETRY_TIMEOUT_S", 600, lo=30, hi=3600),
    }


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
            if target.get("policy_runtime_rebuild") or payload.get("child_policy_runtime_rebuild"):
                continue
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
    direct_parent_mode = (
        str(binding.get("proof_mode") or "") == "direct_parent_route"
        or bool(binding.get("direct_parent_route_verifier_ready"))
    )
    if binding.get("schema_version") != "agentic_parent_stitch_binding.v1":
        reasons.append("invalid_proof_binding_schema")
    input_refs = binding.get("input_refs")
    exact_rows = binding.get("exact_literature_row_ids")
    visual_rows = binding.get("visual_literature_chain_ids")
    missing_inputs = {str(item) for item in binding.get("missing_inputs") or [] if str(item or "").strip()}
    if not isinstance(input_refs, list) or not [item for item in input_refs if str(item or "").strip()]:
        reasons.append("proof_binding_missing_input_refs")
    if not isinstance(exact_rows, list):
        reasons.append("proof_binding_exact_rows_not_list")
    elif (
        not direct_parent_mode
        and not exact_rows
        and not [item for item in visual_rows or [] if str(item or "").strip()]
        and "exact_literature_rows_missing" not in missing_inputs
    ):
        reasons.append("proof_binding_missing_exact_rows_without_reason")
    if visual_rows is not None and not isinstance(visual_rows, list):
        reasons.append("proof_binding_visual_chains_not_list")
    if not str(binding.get("child_route_ref") or "").strip() and "child_route_ref_missing" not in missing_inputs:
        reasons.append("proof_binding_missing_child_route_ref_without_reason")
    if not str(binding.get("parent_route_ref") or "").strip() and "parent_route_ref_missing" not in missing_inputs:
        reasons.append("proof_binding_missing_parent_route_ref_without_reason")

    policy = dict(payload.get("proof_policy") or {})
    if not policy:
        reasons.append("missing_proof_policy")
    direct_parent_mode = direct_parent_mode or (
        str(policy.get("proof_mode") or "") == "direct_parent_route"
        or bool(policy.get("direct_parent_route_verifier_allowed"))
    )
    required_true = [
        "target_equivalence_required",
        "parent_route_verifier_required",
        "stock_audit_required",
        "no_unexplained_large_atom_jump_required",
        "analogy_is_not_proof",
        "child_route_cannot_promote_parent",
    ]
    if direct_parent_mode:
        required_true.append("direct_parent_route_verifier_allowed")
    else:
        required_true.extend(
            [
                "child_route_connectivity_required",
                "exact_literature_connectivity_required",
            ]
        )
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
    local_pdf_allowed = policy.get("local_pdf_fallback_allowed")
    if local_pdf_allowed not in {True, False}:
        reasons.append("source_policy_invalid_local_pdf_fallback_flag")
    if policy.get("placeholder_allowed_after_failures") is not True:
        reasons.append("source_policy_missing_placeholder_fallback")
    if policy.get("auto_local_pdf_requires_agent_discovered_metadata") is not True:
        reasons.append("source_policy_missing_auto_pdf_metadata_guard")
    if policy.get("no_solved_claim") is not True:
        reasons.append("source_policy_missing_no_solved_claim")
    fallback_order = [str(item) for item in policy.get("fallback_order") or []]
    expected_order = ["codex_online", *(["local_pdf"] if local_pdf_allowed is not False else []), "placeholder"]
    if fallback_order != expected_order:
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
    recent_signatures = _recent_unproductive_action_signatures(blackboard, round_count=2)
    return bool(recent_signatures) and all(
        bool(_action_signature_variants(action) & recent_signatures)
        for action in rows
    )


def _advisory_broad_template_required(blackboard: dict[str, Any], actions: list[Any]) -> bool:
    if not _analogical_templates_enabled(blackboard):
        return False
    if not _compile_exact_rows_exhausted_to_advisory(blackboard):
        return False
    if not _broad_template_derivation_ready(blackboard):
        return False
    rows = [dict(row) for row in actions if isinstance(row, dict)]
    if any(str(row.get("action_type") or "") == "derive_broad_reaction_template" for row in rows):
        return False
    if any(str(row.get("action_type") or "") in {"stitch_parent_route", "stop_unresolved"} for row in rows):
        return False
    return True


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
    proposal_rows, proposal_selection_audit = select_guided_retrosynthetic_proposals(
        blackboard,
        proposals=[
            dict(row)
            for row in blackboard.get("retrosynthetic_proposals") or []
            if isinstance(row, dict)
        ],
        limit=12,
    )
    proposal_reaction_classes = _dedupe(
        [
            str(item)
            for row in proposal_rows
            for item in [
                str(row.get("reaction_family") or ""),
                *[str(value) for value in row.get("reaction_families") or []],
            ]
            if str(item or "").strip()
        ]
    )
    proposal_retrons = _dedupe(
        [
            str(item)
            for row in proposal_rows
            for item in [
                str(row.get("product_retron_type") or ""),
                str(row.get("derived_from_retron") or ""),
                *[str(value) for value in row.get("product_retron_types") or []],
            ]
            if str(item or "").strip()
        ]
    )
    proposal_precursor_targets = _precursor_targets_from_retrosynthetic_proposals(proposal_rows, limit=12)
    hypothetical_precursor_smiles = [str(row.get("smiles") or "") for row in hypothetical_precursor_targets]
    visual_precursor_smiles = [str(row.get("smiles") or "") for row in visual_precursor_targets]
    proposal_precursor_smiles = [str(row.get("smiles") or "") for row in proposal_precursor_targets]
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
    process_evidence_rows = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("process_evidence_rows") or []
        if isinstance(row, dict)
    ]
    source_candidates = [
        _compact_source_candidate_for_policy(row)
        for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(row, dict) and _candidate_has_real_source(row)
    ][:8]
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
                        *[
                            str(row.get("process_type") or "").replace("_", " ")
                            for row in process_evidence_rows[:6]
                            if str(row.get("process_type") or "").strip()
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
                        *proposal_reaction_classes,
                    ]
                ),
                "preferred_retrons": proposal_retrons,
                "reaction_and_retron_priors_are_advisory_only": True,
                "hypothetical_route_hints_are_not_proof": True,
                "hypothesis_precursor_hints_are_not_proof": True,
                "visual_connectivity_hints_are_not_proof": bool(visual_exploratory_hints),
                "retrosynthetic_proposals_are_not_proof": bool(proposal_rows),
                "retrosynthetic_proposal_selection_audit": proposal_selection_audit,
                "process_evidence_hints_are_not_proof": bool(process_evidence_rows),
                "semisynthesis_anchor_hints_are_not_proof": bool(semisynthesis_anchors),
                "route_objective_hints_are_not_proof": bool(route_objectives),
                "broad_transform_templates_are_not_proof": bool(broad_templates),
                "de_novo_core_construction_deprioritized": bool(
                    constraints.get("de_novo_core_construction_deprioritized")
                ),
                "small_molecule_stock_closure_deprioritized": bool(
                    constraints.get("small_molecule_stock_closure_deprioritized")
                ),
                "preferred_precursor_smiles": _dedupe(
                    [*hypothetical_precursor_smiles, *visual_precursor_smiles, *proposal_precursor_smiles]
                ),
                "retrosynthetic_proposals": proposal_rows,
                "semisynthesis_anchor_smiles": _dedupe(semisynthesis_anchor_smiles),
                "semisynthesis_anchors": semisynthesis_anchors,
                "route_objectives": route_objectives,
                "endpoint_candidates": endpoint_candidates,
                "broad_transform_templates": broad_templates,
                "process_evidence_rows": process_evidence_rows[:8],
                "source_candidates": source_candidates,
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
                        *proposal_precursor_smiles,
                    ]
                ),
                "semisynthesis_anchors": semisynthesis_anchors,
                "route_objectives": route_objectives,
                "endpoint_candidates": endpoint_candidates,
                "broad_transform_templates": broad_templates,
                "process_evidence_rows": process_evidence_rows[:8],
                "source_candidates": source_candidates,
                "hypothetical_precursor_targets": [
                    *hypothetical_precursor_targets,
                    *visual_precursor_targets,
                    *proposal_precursor_targets,
                ],
                "retrosynthetic_proposals": proposal_rows,
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


_GUIDED_PROPOSAL_IGNORED_SELF_REPORTED_FIELDS = (
    "accepted",
    "achieved_proof_level",
    "authority_bound",
    "confidence",
    "evidence_level",
    "evidence_refs",
    "executable",
    "recursive_expandable",
    "score",
    "validated",
    "validation_tier",
)


def select_guided_retrosynthetic_proposals(
    blackboard: dict[str, Any],
    *,
    proposals: list[dict[str, Any]] | None = None,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose ChemEnzy proposal hints from current host-derived frontier state.

    The proposal bus is advisory and commonly contains more rows than the
    guided-search contract can carry.  Selection therefore derives priority
    from the current canonical frontier ledger, never from model-reported
    confidence, evidence, validation, or authority fields.  If the ledger is
    not digest-bound to the current canonical graph, the selector remains
    deterministic and structurally diverse but marks the result fail-soft and
    consumes none of the ledger's apparent authority.
    """

    capacity = max(0, int(limit or 0))
    raw_rows = proposals
    if raw_rows is None:
        raw_rows = [
            dict(row)
            for row in blackboard.get("retrosynthetic_proposals") or []
            if isinstance(row, dict)
        ]
    rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
    authority = _guided_frontier_authority(blackboard)
    ledger_molecules = (
        dict(authority["ledger"].get("molecules") or {})
        if authority["authoritative"]
        else {}
    )
    min_depth_by_smiles = (
        _canonical_graph_min_depths(authority["canonical_graph"])
        if authority["authoritative"]
        else {}
    )
    fallback_target = _guided_board_target_smiles(blackboard)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        target_smiles = _guided_canonical_smiles(
            row.get("target_smiles") or row.get("product_smiles") or fallback_target
        )
        precursor_smiles = _guided_canonical_smiles(
            row.get("precursor_smiles")
            or row.get("reactant_smiles")
            or row.get("precursors_smiles")
        )
        proposal_id = _guided_proposal_identifier(
            row,
            target_smiles=target_smiles,
            precursor_smiles=precursor_smiles,
        )
        ledger_row = (
            dict(ledger_molecules.get(target_smiles) or {})
            if target_smiles
            else {}
        )
        frontier_priority = _guided_frontier_priority(
            ledger_row,
            min_depth=min_depth_by_smiles.get(target_smiles),
            authoritative=bool(authority["authoritative"]),
        )
        structure_smiles = precursor_smiles or target_smiles
        candidates.append(
            {
                "row": row,
                "proposal_id": proposal_id,
                "target_smiles": target_smiles,
                "precursor_smiles": precursor_smiles,
                "structure_smiles": structure_smiles,
                "fingerprint": _guided_structure_fingerprint(structure_smiles),
                "frontier_priority": frontier_priority,
                "authority_reasons": _guided_candidate_authority_reasons(
                    ledger_row,
                    min_depth=min_depth_by_smiles.get(target_smiles),
                    authoritative=bool(authority["authoritative"]),
                ),
            }
        )

    ordered = _rank_guided_proposal_candidates(candidates)
    selected_candidates = ordered[:capacity]
    selected_ids = [str(row["proposal_id"]) for row in selected_candidates]
    dropped_ids = [str(row["proposal_id"]) for row in ordered[capacity:]]
    ranking = []
    for rank, candidate in enumerate(ordered, start=1):
        ranking.append(
            {
                "proposal_id": str(candidate["proposal_id"]),
                "rank": rank,
                "selected": rank <= capacity,
                "canonical_target_smiles": str(candidate["target_smiles"]),
                "canonical_precursor_smiles": str(candidate["precursor_smiles"]),
                "frontier_priority_tier": str(
                    candidate["frontier_priority"]["tier"]
                ),
                "canonical_min_depth": candidate["frontier_priority"]["min_depth"],
                "max_similarity_to_prior_selection": round(
                    float(candidate.get("max_similarity_to_prior_selection") or 0.0),
                    6,
                ),
                "ranking_reasons": [
                    *candidate["authority_reasons"],
                    str(candidate.get("diversity_reason") or ""),
                    f"stable_proposal_id_tiebreaker:{candidate['proposal_id']}",
                ],
            }
        )
    audit = {
        "schema_version": "guided_retrosynthetic_proposal_selection_audit.v1",
        "capacity": capacity,
        "candidate_count": len(ordered),
        "selected_count": len(selected_candidates),
        "dropped_count": max(0, len(ordered) - len(selected_candidates)),
        "selected_proposal_ids": selected_ids,
        "dropped_proposal_ids": dropped_ids,
        "selection_authority": (
            "current_canonical_frontier_ledger"
            if authority["authoritative"]
            else "stable_fail_soft_without_frontier_authority"
        ),
        "authoritative_frontier_ledger": bool(authority["authoritative"]),
        "frontier_authority_reasons": list(authority["reasons"]),
        "frontier_ledger_content_sha256": str(
            authority["ledger"].get("content_sha256") or ""
        ),
        "canonical_graph_identity_sha256": str(
            authority.get("canonical_graph_identity_sha256") or ""
        ),
        "ranking_contract": [
            "host_ledger_target_stock_open",
            "host_ledger_target_proposal_frontier",
            "host_ledger_target_work_open_and_expansion_allowed",
            "recomputed_canonical_graph_min_depth",
            "greedy_precursor_structural_diversity",
            "stable_proposal_id_tiebreaker",
        ],
        "ignored_self_reported_fields": list(
            _GUIDED_PROPOSAL_IGNORED_SELF_REPORTED_FIELDS
        ),
        "ranking": ranking,
        "proposals_are_advisory_not_proof": True,
        "no_solved_claim": True,
    }
    return [dict(candidate["row"]) for candidate in selected_candidates], audit


def _guided_frontier_authority(blackboard: dict[str, Any]) -> dict[str, Any]:
    ledger = (
        dict(blackboard.get("frontier_ledger") or {})
        if isinstance(blackboard.get("frontier_ledger"), dict)
        else {}
    )
    summary = (
        dict(blackboard.get("frontier_ledger_summary") or {})
        if isinstance(blackboard.get("frontier_ledger_summary"), dict)
        else {}
    )
    canonical_graph = (
        dict(blackboard.get("canonical_route_consensus_graph") or {})
        if isinstance(blackboard.get("canonical_route_consensus_graph"), dict)
        else {}
    )
    reasons: list[str] = []
    if ledger.get("schema_version") != "frontier_ledger.v1":
        reasons.append("frontier_ledger_missing_or_schema_invalid")
    if summary.get("schema_version") != "frontier_ledger_summary.v1":
        reasons.append("frontier_ledger_summary_missing_or_schema_invalid")
    if canonical_graph.get("schema_version") != "route_consensus_graph.v1":
        reasons.append("canonical_route_consensus_graph_missing_or_schema_invalid")
    if not isinstance(ledger.get("root"), dict):
        reasons.append("frontier_ledger_root_invalid")
    if not isinstance(ledger.get("molecules"), dict):
        reasons.append("frontier_ledger_molecules_invalid")
    if not isinstance(ledger.get("edges"), dict):
        reasons.append("frontier_ledger_edges_invalid")

    supplied_digest = str(ledger.get("content_sha256") or "")
    digest_payload = dict(ledger)
    digest_payload.pop("content_sha256", None)
    calculated_digest = _guided_json_sha256(digest_payload)
    if not supplied_digest or supplied_digest != calculated_digest:
        reasons.append("frontier_ledger_content_digest_invalid")
    if str(summary.get("frontier_ledger_content_sha256") or "") != supplied_digest:
        reasons.append("frontier_ledger_summary_digest_mismatch")
    if summary.get("input_valid") is not True:
        reasons.append("frontier_ledger_summary_inputs_invalid")
    if summary.get("ledger_validation_accepted") is not True:
        reasons.append("frontier_ledger_summary_validation_not_accepted")

    input_validation = (
        dict(ledger.get("input_validation") or {})
        if isinstance(ledger.get("input_validation"), dict)
        else {}
    )
    for field in ("graph", "frontier_queue", "reaction_proof_state"):
        row = input_validation.get(field)
        if not isinstance(row, dict) or row.get("valid") is not True:
            reasons.append(f"frontier_ledger_{field}_input_invalid")
    stock_authority = input_validation.get("stock_authority")
    if (
        not isinstance(stock_authority, dict)
        or stock_authority.get("valid") is not True
        or stock_authority.get("authority_boundary")
        != "current_host_stock_provider_replay"
    ):
        reasons.append("frontier_ledger_stock_authority_invalid")

    graph_identity, graph_identity_reasons = _guided_canonical_graph_identity(
        canonical_graph
    )
    reasons.extend(graph_identity_reasons)
    input_bindings = (
        dict(ledger.get("input_bindings") or {})
        if isinstance(ledger.get("input_bindings"), dict)
        else {}
    )
    if input_bindings.get("schema_version") != "frontier_ledger_input_bindings.v1":
        reasons.append("frontier_ledger_input_bindings_invalid")
    if str(input_bindings.get("graph_identity_sha256") or "") != graph_identity:
        reasons.append("frontier_ledger_canonical_graph_binding_mismatch")
    ledger_root = (
        dict(ledger.get("root") or {})
        if isinstance(ledger.get("root"), dict)
        else {}
    )
    if str(ledger_root.get("canonical_smiles") or "") != _guided_canonical_smiles(
        canonical_graph.get("target_smiles")
    ):
        reasons.append("frontier_ledger_canonical_target_mismatch")
    if str(canonical_graph.get("case_id") or "") != str(
        blackboard.get("case_id") or canonical_graph.get("case_id") or ""
    ):
        reasons.append("canonical_route_consensus_graph_case_mismatch")
    return {
        "authoritative": not reasons,
        "reasons": sorted(set(reasons)),
        "ledger": ledger,
        "canonical_graph": canonical_graph,
        "canonical_graph_identity_sha256": graph_identity,
    }


def _guided_canonical_graph_identity(
    graph: dict[str, Any],
) -> tuple[str, list[str]]:
    if graph.get("schema_version") != "route_consensus_graph.v1":
        return "", ["canonical_route_consensus_graph_schema_invalid"]
    case_id = str(graph.get("case_id") or "").strip()
    target = _guided_canonical_smiles(graph.get("target_smiles"))
    reasons: list[str] = []
    if not case_id:
        reasons.append("canonical_route_consensus_graph_case_id_missing")
    if not target:
        reasons.append("canonical_route_consensus_graph_target_invalid")
    raw_steps = graph.get("steps")
    if not isinstance(raw_steps, list):
        reasons.append("canonical_route_consensus_graph_steps_invalid")
        raw_steps = []
    identity_steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            reasons.append(f"canonical_route_consensus_graph_step_invalid:{index}")
            continue
        product = _guided_canonical_smiles(raw.get("product_smiles"))
        precursors = sorted(
            _guided_canonical_smiles(item)
            for item in raw.get("precursor_smiles") or []
        )
        if (
            not str(raw.get("step_id") or "")
            or not str(raw.get("signature") or "")
            or not product
            or not precursors
            or any(not item for item in precursors)
        ):
            reasons.append(
                f"canonical_route_consensus_graph_step_identity_invalid:{index}"
            )
            continue
        identity_steps.append(
            {
                "step_id": str(raw.get("step_id") or ""),
                "signature": str(raw.get("signature") or ""),
                "product_smiles": product,
                "precursor_smiles": precursors,
            }
        )
    if reasons:
        return "", sorted(set(reasons))
    return (
        _guided_json_sha256(
            {
                "schema_version": "route_consensus_graph.v1",
                "case_id": case_id,
                "target_smiles": target,
                "steps": sorted(
                    identity_steps,
                    key=lambda row: (row["step_id"], row["signature"]),
                ),
            }
        ),
        [],
    )


def _canonical_graph_min_depths(graph: dict[str, Any]) -> dict[str, int]:
    """Derive depth from digest-bound steps instead of trusting node labels."""

    target = _guided_canonical_smiles(graph.get("target_smiles"))
    if not target:
        return {}
    steps: list[tuple[str, tuple[str, ...]]] = []
    for raw in graph.get("steps") or []:
        if not isinstance(raw, dict):
            continue
        product = _guided_canonical_smiles(raw.get("product_smiles"))
        precursors = tuple(
            smiles
            for smiles in (
                _guided_canonical_smiles(item)
                for item in raw.get("precursor_smiles") or []
            )
            if smiles
        )
        if product and precursors:
            steps.append((product, precursors))
    depths = {target: 0}
    changed = True
    while changed:
        changed = False
        for product, precursors in steps:
            if product not in depths:
                continue
            next_depth = depths[product] + 1
            for precursor in precursors:
                if next_depth < depths.get(precursor, 1_000_000):
                    depths[precursor] = next_depth
                    changed = True
    return depths


def _guided_frontier_priority(
    ledger_row: dict[str, Any],
    *,
    min_depth: int | None,
    authoritative: bool,
) -> dict[str, Any]:
    if not authoritative:
        return {
            "sort_key": (0, 0, 0),
            "tier": "fail_soft_unranked_by_frontier",
            "min_depth": None,
        }
    if not ledger_row:
        return {
            "sort_key": (6, 1_000_000, 0),
            "tier": "not_in_canonical_frontier",
            "min_depth": None,
        }
    proposal = dict(ledger_row.get("proposal") or {})
    work = dict(ledger_row.get("work") or {})
    stock = dict(ledger_row.get("stock") or {})
    stock_open = stock.get("closed") is not True
    proposal_frontier = str(proposal.get("state") or "") == "frontier"
    work_open = work.get("open") is True
    expansion_allowed = work.get("proposal_expansion_allowed") is True
    if stock_open and proposal_frontier and work_open and expansion_allowed:
        tier, tier_rank = "open_expandable_frontier", 0
    elif stock_open and proposal_frontier and work_open:
        tier, tier_rank = "open_gated_frontier_work", 1
    elif stock_open and proposal_frontier:
        tier, tier_rank = "open_frontier_without_active_work", 2
    elif stock_open and work_open:
        tier, tier_rank = "expanded_target_with_open_work", 3
    elif stock_open:
        tier, tier_rank = "canonical_target_already_expanded", 4
    else:
        tier, tier_rank = "stock_closed_target", 5
    resolved_depth = max(0, int(min_depth)) if min_depth is not None else None
    return {
        "sort_key": (
            tier_rank,
            resolved_depth if resolved_depth is not None else 1_000_000,
            0,
        ),
        "tier": tier,
        "min_depth": resolved_depth,
    }


def _guided_candidate_authority_reasons(
    ledger_row: dict[str, Any],
    *,
    min_depth: int | None,
    authoritative: bool,
) -> list[str]:
    if not authoritative:
        return ["fail_soft:no_current_canonical_frontier_authority"]
    if not ledger_row:
        return ["canonical_frontier:target_not_present"]
    proposal = dict(ledger_row.get("proposal") or {})
    work = dict(ledger_row.get("work") or {})
    stock = dict(ledger_row.get("stock") or {})
    return [
        f"canonical_frontier:stock_open={str(stock.get('closed') is not True).lower()}",
        f"canonical_frontier:proposal_state={str(proposal.get('state') or 'unknown')}",
        f"canonical_frontier:work_open={str(work.get('open') is True).lower()}",
        "canonical_frontier:proposal_expansion_allowed="
        + str(work.get("proposal_expansion_allowed") is True).lower(),
        "canonical_frontier:min_depth="
        + (str(max(0, int(min_depth))) if min_depth is not None else "unknown"),
    ]


def _rank_guided_proposal_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    selected_targets: set[str] = set()
    selected_fingerprints: list[Any] = []
    while remaining:
        best_frontier_key = min(
            tuple(candidate["frontier_priority"]["sort_key"])
            for candidate in remaining
        )
        tier = [
            candidate
            for candidate in remaining
            if tuple(candidate["frontier_priority"]["sort_key"])
            == best_frontier_key
        ]
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], float]] = []
        for candidate in tier:
            fingerprint = candidate.get("fingerprint")
            similarities = (
                DataStructs.BulkTanimotoSimilarity(
                    fingerprint,
                    selected_fingerprints,
                )
                if fingerprint is not None and selected_fingerprints
                else []
            )
            max_similarity = max(similarities, default=0.0)
            target_seen = bool(
                candidate["target_smiles"]
                and candidate["target_smiles"] in selected_targets
            )
            no_structure = fingerprint is None
            key = (
                target_seen,
                no_structure,
                round(float(max_similarity), 12),
                str(candidate["structure_smiles"]),
                str(candidate["proposal_id"]),
            )
            ranked.append((key, candidate, float(max_similarity)))
        _, chosen, max_similarity = min(ranked, key=lambda item: item[0])
        chosen["max_similarity_to_prior_selection"] = max_similarity
        chosen["diversity_reason"] = (
            "structural_diversity:first_selected_structure"
            if not selected
            else "structural_diversity:max_similarity_to_prior="
            f"{max_similarity:.6f}"
        )
        selected.append(chosen)
        remaining.remove(chosen)
        if chosen["target_smiles"]:
            selected_targets.add(str(chosen["target_smiles"]))
        if chosen.get("fingerprint") is not None:
            selected_fingerprints.append(chosen["fingerprint"])
    return selected


def _guided_proposal_identifier(
    row: dict[str, Any],
    *,
    target_smiles: str,
    precursor_smiles: str,
) -> str:
    explicit = str(row.get("proposal_id") or row.get("semantic_edge_key") or "").strip()
    if explicit:
        return explicit
    retron = " ".join(
        str(row.get("proposal_label") or row.get("transformation_idea") or "")
        .lower()
        .split()
    )
    return "anonymous_proposal:" + _guided_json_sha256(
        {
            "target_smiles": target_smiles,
            "precursor_smiles": precursor_smiles,
            "retron": retron,
        }
    )[:16]


def _guided_board_target_smiles(blackboard: dict[str, Any]) -> str:
    target = dict(blackboard.get("target_profile") or {})
    return _guided_canonical_smiles(
        target.get("canonical_smiles")
        or target.get("target_smiles")
        or blackboard.get("target_smiles")
    )


def _guided_canonical_smiles(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _guided_structure_fingerprint(smiles: str) -> Any:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return None
    return Chem.RDKFingerprint(molecule, fpSize=512)


def _guided_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return ""
    return hashlib.sha256(payload).hexdigest()


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


def _precursor_targets_from_retrosynthetic_proposals(
    proposals: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        smiles = str(proposal.get("precursor_smiles") or "").strip()
        if not smiles:
            continue
        if not bool(proposal.get("recursive_expandable") or proposal.get("executable")):
            continue
        target_rows = [
            _retrosynthetic_proposal_precursor_target(
                proposal,
                smiles=smiles,
                source="retrosynthetic_proposal",
                component_index=0,
                component_count=int(proposal.get("precursor_component_count") or 1),
                precursor_set_smiles=smiles if bool(proposal.get("multi_component_precursor_set")) else "",
            )
        ]
        components = _precursor_components_from_smiles(smiles)
        if len(components) > 1:
            target_rows.extend(
                _retrosynthetic_proposal_precursor_target(
                    proposal,
                    smiles=component,
                    source="retrosynthetic_proposal_component",
                    component_index=idx,
                    component_count=len(components),
                    precursor_set_smiles=smiles,
                )
                for idx, component in enumerate(components, start=1)
            )
        for target_row in target_rows:
            target_smiles = str(target_row.get("smiles") or "").strip()
            if not target_smiles or target_smiles in seen:
                continue
            seen.add(target_smiles)
            out.append(target_row)
            if len(out) >= max(1, int(limit or 1)):
                return out
    return out


def _retrosynthetic_proposal_precursor_target(
    proposal: dict[str, Any],
    *,
    smiles: str,
    source: str,
    component_index: int,
    component_count: int,
    precursor_set_smiles: str,
) -> dict[str, Any]:
    return {
        "schema_version": "guided_search_hypothetical_precursor_target.v1",
        "smiles": smiles,
        "role": str(proposal.get("proposal_label") or proposal.get("source_type") or "retrosynthetic proposal"),
        "source": source,
        "source_proposal_id": str(proposal.get("proposal_id") or ""),
        "proposal_type": str(proposal.get("proposal_type") or ""),
        "proposal_granularity": str(proposal.get("proposal_granularity") or ""),
        "route_objective_type": str(proposal.get("route_objective_type") or ""),
        "failure_response_policy": dict(proposal.get("failure_response_policy") or {}),
        "transformation_idea": str(proposal.get("transformation_idea") or ""),
        "reaction_family": str(proposal.get("reaction_family") or ""),
        "reaction_families": [
            str(item)
            for item in proposal.get("reaction_families") or []
            if str(item or "").strip()
        ],
        "product_retron_type": str(proposal.get("product_retron_type") or ""),
        "product_retron_types": [
            str(item)
            for item in proposal.get("product_retron_types") or []
            if str(item or "").strip()
        ],
        "derived_from_retron": str(
            proposal.get("derived_from_retron")
            or proposal.get("product_retron_type")
            or ""
        ),
        "retron_authority": "advisory_search_prior_only",
        "precursor_set_smiles": precursor_set_smiles,
        "precursor_component_index": int(component_index or 0),
        "precursor_component_count": int(component_count or 1),
        "multi_component_precursor_set": int(component_count or 1) > 1,
        "requires_precursor_set_stitching": source == "retrosynthetic_proposal_component",
        "allowed_use": "guided_search_subgoal_hint_only",
        "analogy_is_advisory_only": str(proposal.get("source_type") or "").startswith("analogical"),
        "not_exact_literature_segment": bool(proposal.get("not_exact_literature_segment", True)),
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "no_solved_claim": True,
    }


def _precursor_components_from_smiles(smiles: str) -> list[str]:
    components = [part for part in str(smiles or "").split(".") if part]
    if not components:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in components:
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
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
    refs.extend(
        str(
            row.get("source_ref")
            or row.get("doi")
            or row.get("pii")
            or row.get("url")
            or row.get("local_pdf")
            or ""
        )
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict)
    )
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
    return is_solved_parent_route_proof(
        proof,
        expected_target_smiles=str((blackboard.get("target_profile") or {}).get("target_smiles") or ""),
    )


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
    biases = _dedupe([str(item) for item in belief.get("next_action_bias") or [] if str(item or "").strip()])
    if _process_evidence_available(blackboard):
        route_first = [
            "derive_broad_reaction_template",
            "compile_objective_route_proof",
            "run_guided_chemenzy",
            "expand_child_target",
        ]
        promoted = [item for item in route_first if item in biases]
        if promoted:
            return _dedupe([*promoted, *[item for item in biases if item not in promoted]])
    return biases


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
            and _uncompiled_visual_steps_available(blackboard)
            and not _compile_exact_rows_exhausted_to_advisory(blackboard)
            and not _stale_compile_requires_structure_resolution(blackboard)
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
            _child_expansion_rationale(blackboard),
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
        if not _new_objective_proof_signal_since_last_compile(blackboard):
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


def _new_objective_proof_signal_since_last_compile(blackboard: dict[str, Any]) -> bool:
    history = [row for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    last_compile_index = -1
    for idx, row in enumerate(history):
        if str(row.get("action_type") or "") == "compile_objective_route_proof":
            last_compile_index = idx
    if last_compile_index < 0:
        return True
    if not blackboard.get("route_proof_bundle"):
        return True
    signal_fields = {
        "broad_transform_templates",
        "exact_rows",
        "parent_route_proof_present",
        "pdf_structure_evidence",
        "process_evidence_rows",
        "reaction_idea_cards",
        "resolved_structures",
        "retrosynthetic_proposals",
        "semisynthesis_anchors",
        "template_applications",
        "visual_chains",
    }
    for row in history[last_compile_index + 1 :]:
        changed = {str(item) for item in row.get("changed_blackboard_fields") or [] if str(item or "").strip()}
        if changed & signal_fields:
            return True
    return False


def _visual_extraction_has_rendered_input(
    payload: dict[str, Any],
    blackboard: dict[str, Any],
) -> bool:
    """Require materialized images before a visual-call budget is consumed."""
    image_paths = [
        str(item or "").strip()
        for item in payload.get("image_paths") or []
        if str(item or "").strip()
    ]
    if image_paths:
        if not all(Path(item).expanduser().is_file() for item in image_paths):
            return False
        rendered = _pdf_structure_source_keys(blackboard)
        key = _source_key(payload)
        return bool(key and key in rendered)
    rendered = _pdf_structure_source_keys(blackboard)
    key = _source_key(payload)
    if any(
        str(payload.get(field) or "").strip()
        for field in ("pdf_path", "local_pdf", "source_pdf_path", "document_id")
    ):
        return key in rendered
    matches = _matching_local_pdf_document_keys(payload, blackboard)
    if matches:
        return len(matches) == 1 and matches <= rendered
    if key:
        return key in rendered
    candidates = _local_pdf_source_candidates(blackboard)
    return len(candidates) == 1 and bool(rendered)


def _focused_visual_gap_repair_is_eligible(
    payload: dict[str, Any],
    blackboard: dict[str, Any],
) -> bool:
    """Validate the one source-capability exception against current host state."""

    if not _visual_extraction_has_rendered_input(payload, blackboard):
        return False
    if not _visual_gap_repair_needed(blackboard):
        return False
    if not _visual_gap_repair_budget_remaining(blackboard):
        return False
    if not _budget_remaining(blackboard, "visual_calls"):
        return False

    gap_document_keys = _visual_gap_document_keys(blackboard)
    payload_document_keys = _matching_local_pdf_document_keys(payload, blackboard)
    payload_key = _source_key(payload)
    if payload_key:
        payload_document_keys.add(payload_key)
    return bool(
        len(gap_document_keys) == 1
        and len(payload_document_keys & gap_document_keys) == 1
    )


def _visual_gap_document_keys(blackboard: dict[str, Any]) -> set[str]:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    rendered = _pdf_structure_source_keys(blackboard)
    gap_rows = [
        dict(row)
        for row in evidence.get("visual_chains") or []
        if isinstance(row, dict)
        and (
            row.get("gap_labels")
            or row.get("condition_gap_labels")
            or row.get("missing_expected_labels")
            or row.get("extraction_gaps")
        )
    ]
    keys: set[str] = set()
    for row in gap_rows:
        direct_key = _source_key(row)
        mapped_keys = (
            _matching_local_pdf_document_keys(row, blackboard) & rendered
        )
        if direct_key and direct_key in mapped_keys:
            keys.add(direct_key)
            continue
        if mapped_keys:
            keys.update(mapped_keys)
            continue
        if direct_key and direct_key in rendered:
            keys.add(direct_key)
    if not keys and len(gap_rows) == 1 and len(rendered) == 1:
        # A locator-free legacy summary can bind only when both sides are
        # unambiguous in current host state.
        keys = set(rendered)
    return keys


def _needs_literature_bridge(blackboard: dict[str, Any]) -> bool:
    evidence = dict(blackboard.get("literature_evidence") or {})
    independent_count = len(independent_literature_source_keys(blackboard))
    if (
        independent_count < _minimum_independent_literature_sources(blackboard)
        and not evidence.get("exact_rows")
    ):
        return True
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
    if _compile_exact_rows_exhausted_to_advisory(blackboard):
        return True
    if _visual_exploratory_hints_available(blackboard):
        return True
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
    return bool(
        evidence.get("source_candidates")
        or evidence.get("visual_chains")
        or evidence.get("process_evidence_rows")
        or evidence.get("exact_rows")
    )


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


def _compact_source_candidate_for_policy(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source_ref": str(row.get("source_ref") or ""),
        "title": str(row.get("title") or row.get("source_title") or ""),
        "doi": str(row.get("doi") or ""),
        "pii": str(row.get("pii") or ""),
        "url": str(row.get("url") or ""),
        "source_role": str(row.get("source_role") or ""),
        "source_usage_policy": str(row.get("source_usage_policy") or ""),
        "no_solved_claim": True,
    }
    if str(row.get("local_pdf") or "").strip():
        out["local_pdf"] = str(row.get("local_pdf") or "")
    labels = meaningful_compound_labels(
        row.get("expected_scheme_or_compound_labels") or []
    )
    if labels:
        out["expected_scheme_or_compound_labels"] = labels[:12]
    route_hint = str(row.get("route_sequence_hint") or "").strip()
    if route_hint:
        out["route_sequence_hint"] = route_hint[:500]
    return {key: value for key, value in out.items() if value not in ("", [], {})}


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
    queue = build_source_capability_queue(
        blackboard,
        round_index=_source_capability_round_index(blackboard),
    )
    capabilities = eligible_source_capabilities(
        queue,
        "extract_pdf_literature_structures",
    )
    if not capabilities:
        return {}
    source = dict(capabilities[0].get("source") or {})
    source["source_capability_id"] = str(
        capabilities[0].get("capability_id") or ""
    )
    return source


def _next_local_pdf_source_for_visual_extraction(blackboard: dict[str, Any]) -> dict[str, Any]:
    queue = build_source_capability_queue(
        blackboard,
        round_index=_source_capability_round_index(blackboard),
    )
    capabilities = eligible_source_capabilities(
        queue,
        "extract_visual_literature_chain",
    )
    if not capabilities:
        return {}
    source = dict(capabilities[0].get("source") or {})
    source["source_capability_id"] = str(
        capabilities[0].get("capability_id") or ""
    )
    return source


def _pdf_structure_source_keys(blackboard: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for row in (blackboard.get("literature_evidence") or {}).get("pdf_structure_evidence") or []:
        if not isinstance(row, dict):
            continue
        if not _pdf_evidence_has_materialized_render(row):
            continue
        keys.update(_evidence_document_source_keys(row, blackboard))
    return keys


def _pdf_evidence_has_materialized_render(row: dict[str, Any]) -> bool:
    return pdf_evidence_has_materialized_render(row)


def _pdf_evidence_render_paths(row: dict[str, Any]) -> list[str]:
    return pdf_evidence_render_paths(row)


def _source_capability_round_index(blackboard: dict[str, Any]) -> int:
    budget = dict((blackboard or {}).get("budget_state") or {})
    try:
        return max(1, int(budget.get("rounds_completed") or 0) + 1)
    except (TypeError, ValueError):
        return 1


def _visual_chain_source_keys(blackboard: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        if (
            not bool(row.get("accepted"))
            and _visual_chain_candidate_step_count(dict(row)) <= 0
            and "visual_input_images_missing" in {str(item) for item in row.get("reasons") or []}
        ):
            continue
        keys.update(_evidence_document_source_keys(row, blackboard))
    return keys


def _evidence_document_source_keys(
    row: dict[str, Any],
    blackboard: dict[str, Any],
) -> set[str]:
    """Resolve legacy source-only evidence only when it names one document.

    Current extractor artifacts carry ``source_pdf_path`` and therefore have
    an exact document key.  Older artifacts can carry only DOI/source_ref; in
    that case a logical source may be upgraded to a document key only when the
    blackboard has exactly one matching local document.  This keeps legacy
    single-document runs usable without letting one DOI-level record consume
    every article/SI document attached to that DOI.
    """
    key = _source_key(row)
    keys = {key} if key else set()
    if any(
        str(row.get(field) or "").strip()
        for field in ("local_pdf", "source_pdf_path", "pdf_path", "document_id")
    ):
        return keys
    matches = _matching_local_pdf_document_keys(row, blackboard)
    if len(matches) == 1:
        keys.update(matches)
    elif not key and not matches:
        candidate_keys = {
            _source_key(candidate)
            for candidate in _local_pdf_source_candidates(blackboard)
            if _source_key(candidate)
        }
        if len(candidate_keys) == 1:
            keys.update(candidate_keys)
    return keys


def _matching_local_pdf_document_keys(
    row: dict[str, Any],
    blackboard: dict[str, Any],
) -> set[str]:
    logical_keys = _logical_source_identifiers(row)
    if not logical_keys:
        return set()
    matches: set[str] = set()
    for candidate in _local_pdf_source_candidates(blackboard):
        if not (logical_keys & _logical_source_identifiers(candidate)):
            continue
        key = _source_key(candidate)
        if key:
            matches.add(key)
    return matches


def _logical_source_identifiers(row: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for field in ("doi", "source_ref", "url"):
        doi = _logical_source_doi(str(row.get(field) or ""))
        if doi:
            identifiers.add(f"doi:{doi}")
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        identifiers.add(f"pii:{pii}")
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref and not _logical_source_doi(source_ref):
        identifiers.add(f"ref:{source_ref}")
    url = str(row.get("url") or "").strip().lower()
    if url and not _logical_source_doi(url):
        identifiers.add(f"url:{url}")
    title = str(row.get("title") or row.get("source_title") or "").strip().lower()
    if title:
        identifiers.add(f"title:{' '.join(title.split())}")
    return identifiers


def _logical_source_doi(value: str) -> str:
    text = str(value or "").strip().lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if not text.startswith("10.") or "/" not in text:
        return ""
    for separator in ("&", "?", "#"):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text.strip().strip(".,;:)]}'\"")


def _source_key(row: dict[str, Any]) -> str:
    # One DOI can own several extractable documents (article, SI, corrections).
    # Prefer a concrete document identity so rendering one file does not mark
    # every document attached to that DOI as already processed.
    local_pdf = str(row.get("local_pdf") or row.get("source_pdf_path") or row.get("pdf_path") or "").strip().lower()
    if local_pdf:
        return f"pdf:{local_pdf}"
    document_id = str(row.get("document_id") or "").strip().lower()
    if document_id:
        return f"document:{document_id}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        return f"pii:{pii}"
    title = str(row.get("title") or row.get("source_title") or "").strip().lower()
    return f"title:{title}" if title else ""


def _source_candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if row.get("source_capability_id"):
        payload["source_capability_id"] = str(
            row.get("source_capability_id") or ""
        )
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
        payload.setdefault("timeout_s", _default_visual_timeout_s())
    if row.get("document_id"):
        payload["document_id"] = str(row.get("document_id") or "")
    if row.get("content_scope"):
        payload["content_scope"] = str(row.get("content_scope") or "")
    if row.get("chain_id"):
        payload["chain_id"] = str(row.get("chain_id") or "")
    if row.get("artifact_ref"):
        payload["artifact_ref"] = str(row.get("artifact_ref") or "")
    if row.get("route_sequence_hint"):
        payload["route_sequence_hint"] = str(row.get("route_sequence_hint") or "")
    labels = meaningful_compound_labels(
        row.get("expected_scheme_or_compound_labels") or []
    )
    if labels:
        payload["expected_labels"] = labels
        payload["compound_labels"] = labels
    _apply_source_visual_extraction_profile(payload, row)
    return payload


def _default_visual_timeout_s() -> int:
    raw = os.environ.get("AUTOPLANNER_VISUAL_TIMEOUT_S")
    try:
        value = float(raw) if raw not in (None, "") else 900.0
    except (TypeError, ValueError):
        value = 900.0
    return int(max(120.0, value))


def _apply_source_visual_extraction_profile(payload: dict[str, Any], row: dict[str, Any]) -> None:
    """Apply only caller-supplied rendering/focus policy.

    Source titles and identifiers are evidence locators, not configuration.
    A source may opt into specialized page selection or route-focus prompts by
    carrying an explicit ``visual_extraction_profile``; otherwise the generic
    extraction defaults above remain unchanged.
    """
    profile = row.get("visual_extraction_profile")
    if not isinstance(profile, dict):
        return
    for key in (
        "compress_images",
        "max_images",
        "visual_max_side_px",
        "visual_jpeg_quality",
        "render_zoom",
        "timeout_s",
        "route_sequence_hint",
    ):
        value = profile.get(key)
        if value not in (None, ""):
            if key == "route_sequence_hint" and str(row.get(key) or "").strip():
                continue
            payload[key] = value
    for key in ("page_numbers", "scheme_crops", "expected_labels", "compound_labels"):
        values = profile.get(key)
        if isinstance(values, list) and values:
            payload.setdefault(key, list(values))


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
    independent_keys = sorted(independent_literature_source_keys(blackboard))
    known_source_refs = _dedupe(
        [
            str(row.get("source_ref") or row.get("doi") or row.get("url") or "")
            for row in evidence.get("source_candidates") or []
            if isinstance(row, dict)
            and _candidate_has_real_source(row)
            and str(row.get("source_ref") or row.get("doi") or row.get("url") or "").strip()
        ]
    )
    requested_sources = max(1, min(3, 3 - len(independent_keys)))
    return {
        "schema_version": "agentic_literature_search_payload.v1",
        "search_intent": str(intent or "target_proximal_source_discovery"),
        "queries": query_list,
        "search_queries": query_list,
        "max_sources": requested_sources,
        "minimum_independent_sources": _minimum_independent_literature_sources(blackboard),
        "preferred_independent_sources": 3,
        "known_independent_source_keys": independent_keys,
        "exclude_source_refs": known_source_refs,
        "source_independence_policy": {
            "schema_version": "agentic_source_independence_policy.v1",
            "group_by": ["doi", "pii", "patent_family", "canonical_source_ref", "title"],
            "article_and_supporting_information_share_source_group": True,
            "require_distinct_source_groups": True,
            "no_solved_claim": True,
        },
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


def independent_literature_source_keys(blackboard: dict[str, Any]) -> set[str]:
    """Return logical source groups, not per-document PDF identities.

    An article and its SI may be two extractable documents but are one
    independent source.  This distinction is essential both for scheduling a
    genuinely multi-source scout and for preventing document count from being
    reported as source independence.
    """
    keys: set[str] = set()
    for row in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []:
        if not isinstance(row, dict) or not _candidate_has_real_source(row):
            continue
        key = _independent_literature_source_key(row)
        if key:
            keys.add(key)
    return keys


def _independent_literature_source_key(row: dict[str, Any]) -> str:
    host_group = independent_source_group(row)
    if host_group:
        return host_group
    doi = _logical_source_doi(str(row.get("doi") or row.get("source_ref") or row.get("url") or ""))
    if doi:
        return f"doi:{doi}"
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        return f"pii:{pii}"
    patent_family = str(row.get("patent_family") or row.get("family_id") or "").strip().lower()
    if patent_family:
        return f"patent_family:{patent_family}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    title = " ".join(str(row.get("title") or row.get("source_title") or "").lower().split())
    if title:
        return f"title:{title}"
    url = str(row.get("url") or "").strip().lower()
    return f"url:{url}" if url else ""


def _minimum_independent_literature_sources(blackboard: dict[str, Any]) -> int:
    belief = dict(blackboard.get("current_belief") or {})
    constraints = dict(belief.get("constraints") or {})
    raw = constraints.get("minimum_independent_literature_sources", 2)
    try:
        return max(2, min(3, int(raw)))
    except (TypeError, ValueError):
        return 2


def plan_literature_evidence_followup_actions(
    blackboard: dict[str, Any],
    *,
    round_index: int,
    max_actions: int = 1,
) -> list[dict[str, Any]]:
    """Plan the next executable stage of the literature evidence lifecycle.

    This intentionally returns only actions executable from the *current*
    blackboard.  Results written by those actions become inputs to the next
    controller round, so critic/bridge tasks can consume them without a hidden
    same-snapshot dependency.
    """
    limit = max(0, int(max_actions or 0))
    if limit <= 0:
        return []

    # Finish the proof-bearing transition already in hand before opening a new
    # document lifecycle.  Compilation remains fail-closed and cannot itself
    # promote a row; it only gives the deterministic verifier/registry a
    # bounded candidate to accept or reject.
    if (
        _visual_chain_available(blackboard)
        and _uncompiled_visual_steps_available(blackboard)
        and not _compile_exact_rows_exhausted_to_advisory(blackboard)
        and not _stale_compile_requires_structure_resolution(blackboard)
    ):
        return [
            _action(
                round_index,
                "compile_exact_literature_rows",
                "finish the materialized source candidate before opening another document lifecycle",
                "compiled exact literature rows",
                "exact rows are accepted or rejected with an auditable source-detail reason",
                _compile_exact_rows_payload(blackboard),
            )
        ]

    source = _next_local_pdf_source_for_pdf_extraction(blackboard)
    if source and not _source_has_pdf_evidence_record(blackboard, source):
        return [
            _action(
                round_index,
                "extract_pdf_literature_structures",
                "continue the discovered source lifecycle by materializing PDF pages before visual interpretation",
                "literature_pdf_structure_evidence.v1",
                "rendered pages or indexed images are bound to the selected source",
                _source_candidate_payload(source),
            )
        ]

    source = _next_local_pdf_source_for_visual_extraction(blackboard)
    if source and _budget_remaining(blackboard, "visual_calls"):
        return [
            _action(
                round_index,
                "extract_visual_literature_chain",
                "continue the rendered source lifecycle with source-bound structure and scheme extraction",
                "visual_literature_chain/exact rows artifact",
                "source-detail steps or explicit structure gaps are recorded",
                _visual_extraction_payload_from_blackboard(blackboard, source_candidate=source),
            )
        ]

    if (
        _visual_chain_available(blackboard)
        and (not _exact_rows_available(blackboard) or _exact_rows_incomplete(blackboard))
        and _visual_gap_repair_needed(blackboard)
        and not _process_evidence_available(blackboard)
        and _visual_gap_repair_budget_remaining(blackboard)
        and (_condition_gap_repair_needed(blackboard) or not _uncompiled_visual_steps_available(blackboard))
        and _budget_remaining(blackboard, "visual_calls")
    ):
        return [
            _action(
                round_index,
                "extract_visual_literature_chain",
                "continue the evidence lifecycle with a focused repair of source-detail structure or condition gaps",
                "visual_literature_chain/exact rows artifact",
                "missing source-detail labels are filled or explicitly rejected",
                _focused_visual_repair_payload(blackboard),
            )
        ]

    task = _next_structure_resolution_task_for_local_resolve(blackboard)
    if task and _budget_remaining(blackboard, "visual_calls"):
        return [
            _action(
                round_index,
                "resolve_literature_structure_task",
                "continue the source lifecycle by resolving an open compound label before exact-row compilation",
                "literature_structure_resolution_result.v1",
                "the selected label is source-bound and resolved or explicitly rejected",
                _structure_resolution_task_payload(blackboard, task),
            )
        ]

    if (
        _structure_resolution_scout_needed(blackboard)
        and _budget_remaining(blackboard, "scout_calls")
        and not _stale_literature_search_repeated(blackboard)
    ):
        return [
            _action(
                round_index,
                "search_literature",
                "continue the evidence lifecycle by finding source-detail material for unresolved compound labels",
                "structure_resolution_source_scout_report.v1",
                "a new source-bound structure-resolution lead or an explicit unresolved record is produced",
                _structure_resolution_scout_payload(blackboard),
            )
        ]

    evidence = dict(blackboard.get("literature_evidence") or {})
    independent_count = len(independent_literature_source_keys(blackboard))
    metadata_without_document = any(
        isinstance(row, dict)
        and _candidate_has_real_source(row)
        and not str(row.get("local_pdf") or row.get("source_pdf_path") or row.get("pdf_path") or "").strip()
        for row in evidence.get("source_candidates") or []
    )
    needs_independent_source = independent_count < _minimum_independent_literature_sources(blackboard)
    if (
        evidence.get("source_candidates")
        and (metadata_without_document or needs_independent_source)
        and _budget_remaining(blackboard, "scout_calls")
        and not _stale_literature_search_repeated(blackboard)
        and _evidence_followup_search_allowed(blackboard)
    ):
        intent = (
            "source_detail_html_or_pdf_acquisition"
            if metadata_without_document
            else "independent_source_expansion"
        )
        return [
            _action(
                round_index,
                "search_literature",
                "continue source acquisition toward accessible full text and genuinely independent corroborating sources",
                "literature_scout_report.v1",
                "an accessible HTML/PDF lead or a new independent source group is recorded",
                _literature_search_payload(blackboard, intent=intent),
            )
        ]
    return []


def _evidence_followup_search_allowed(blackboard: dict[str, Any]) -> bool:
    # A queued PDF proxy is an explicit external wait state.  Repeatedly
    # scouting the same DOI while it is pending wastes the global scout budget.
    if _awaiting_local_pdf_proxy_download(blackboard):
        return False
    if (
        _failure_evidence_available(blackboard)
        and _action_seen(blackboard, "build_failure_critic_report")
        and not _new_failure_evidence_since_last_critic(blackboard)
    ):
        return False
    return True


def _source_has_pdf_evidence_record(blackboard: dict[str, Any], source: dict[str, Any]) -> bool:
    source_key = _source_key(source)
    if not source_key:
        return False
    for raw in (blackboard.get("literature_evidence") or {}).get("pdf_structure_evidence") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if row.get("accepted") is False:
            continue
        # Article and SI belong to one logical source group for corroboration,
        # but they remain different extraction documents.  Match the concrete
        # PDF/document key; the legacy resolver below upgrades a DOI-only
        # record only when exactly one local document matches it.
        if source_key in _evidence_document_source_keys(row, blackboard):
            return True
    return False


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


def _process_evidence_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("literature_evidence") or {}).get("process_evidence_rows"))


def _uncompiled_visual_steps_available(blackboard: dict[str, Any]) -> bool:
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        if _visual_chain_uncompiled_step_count(blackboard, dict(row)) > 0:
            return True
    return False


def _stale_compile_requires_structure_resolution(blackboard: dict[str, Any]) -> bool:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    if evidence.get("exact_rows"):
        return False
    if not _open_structure_resolution_tasks(blackboard):
        return False
    if _next_uncompiled_compile_capability(blackboard):
        return False
    if _fresh_uncompiled_visual_steps_after_latest_compile(blackboard):
        return False
    compile_attempt_count = _action_count(blackboard, "compile_exact_literature_rows")
    stale_compile_history = sum(
        1
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict)
        and str(row.get("action_type") or "") == "compile_exact_literature_rows"
        and (row.get("useful_artifact") is False or row.get("stale") is True)
    )
    stale_audits = 0
    for audit in evidence.get("exact_chain_audits") or []:
        if not isinstance(audit, dict) or audit.get("accepted"):
            continue
        reasons = {str(item) for item in audit.get("reasons") or []}
        if reasons & {"no_chain_unrolled", "missing_one_step_row_for_product"}:
            stale_audits += 1
    return bool(compile_attempt_count >= 2 or stale_compile_history >= 1 or stale_audits >= 1)


def _fresh_uncompiled_visual_steps_after_latest_compile(
    blackboard: dict[str, Any],
) -> bool:
    """Allow a compile retry only when a later visual artifact changed input.

    Structure-resolution tasks and failed compile audits are source-local.  A
    stale compile for one document must not globally block a newly extracted,
    exact-capable chain from another document (or a repaired chain from the
    same document).  Artifact refs bind the history row to the materialized
    chain, so merely repeating a planner action cannot bypass the stale gate.
    """

    history = [
        dict(row)
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict)
    ]
    compile_rounds = [
        int(row.get("round_index") or 0)
        for row in history
        if str(row.get("action_type") or "") == "compile_exact_literature_rows"
    ]
    if not compile_rounds:
        return False
    latest_compile_round = max(compile_rounds)
    fresh_artifact_refs = {
        str(row.get("artifact_ref") or "").strip().casefold()
        for row in history
        if str(row.get("action_type") or "")
        == "extract_visual_literature_chain"
        and int(row.get("round_index") or 0) > latest_compile_round
        and row.get("useful_artifact") is not False
        and str(row.get("artifact_ref") or "").strip()
    }
    if not fresh_artifact_refs:
        return False
    for row in (blackboard.get("literature_evidence") or {}).get(
        "visual_chains"
    ) or []:
        if not isinstance(row, dict):
            continue
        refs = {
            str(row.get(field) or "").strip().casefold()
            for field in ("artifact_ref", "chain_id")
            if str(row.get(field) or "").strip()
        }
        if refs & fresh_artifact_refs and _visual_chain_uncompiled_step_count(
            blackboard, row
        ) > 0:
            return True
    return False


def _compile_exact_rows_exhausted_to_advisory(blackboard: dict[str, Any]) -> bool:
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    if evidence.get("exact_rows"):
        return False
    if blackboard.get("broad_transform_templates"):
        return False
    advisory_reason_tokens = {
        "advisory_visual_template_card_available",
        "applicability_failed",
        "product_reconstruction_failed",
        "source_detail_step_not_exact",
        "missing_one_step_row_for_product",
        "no_chain_unrolled",
    }
    for audit in evidence.get("exact_chain_audits") or []:
        if not isinstance(audit, dict) or audit.get("accepted"):
            continue
        reasons = {str(item) for item in audit.get("reasons") or [] if str(item or "").strip()}
        if reasons & advisory_reason_tokens:
            return True
    return False


def _compile_exact_rows_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "compile_attempt": _action_count(
            blackboard, "compile_exact_literature_rows"
        )
        + 1,
        "deterministic_parser_authority_id": PARSER_AUTHORITY_ID,
    }
    capability = _next_uncompiled_compile_capability(blackboard)
    if capability:
        return source_capability_effective_payload(payload, capability)
    source = _next_uncompiled_visual_source(blackboard) or _latest_visual_source(blackboard)
    if source:
        payload.update(_source_candidate_payload(source))
    return payload


def _next_uncompiled_compile_capability(
    blackboard: dict[str, Any],
) -> dict[str, Any]:
    """Select the queue-authoritative compile binding for an exact-capable chain."""

    queue = build_source_capability_queue(
        blackboard,
        round_index=_source_capability_round_index(blackboard),
    )
    capabilities = eligible_source_capabilities(
        queue,
        "compile_exact_literature_rows",
    )
    visual_rows = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get(
            "visual_chains"
        )
        or []
        if isinstance(row, dict)
    ]
    for capability in capabilities:
        binding = dict(capability.get("payload_binding") or {})
        refs = {
            str(binding.get(field) or "").strip().casefold()
            for field in ("chain_id", "visual_chain_id", "artifact_ref")
            if str(binding.get(field) or "").strip()
        }
        if not refs:
            continue
        for visual in visual_rows:
            if not any(
                isinstance(step, dict) for step in visual.get("steps") or []
            ):
                continue
            visual_refs = {
                str(visual.get(field) or "").strip().casefold()
                for field in ("chain_id", "artifact_ref")
                if str(visual.get(field) or "").strip()
            }
            if refs & visual_refs and _visual_chain_uncompiled_step_count(
                blackboard, visual
            ) > 0:
                return dict(capability)
    return {}


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
                return {
                    **dict(candidate),
                    "chain_id": str(
                        visual.get("chain_id")
                        or visual.get("artifact_ref")
                        or ""
                    ),
                    "artifact_ref": str(visual.get("artifact_ref") or ""),
                }
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
    if _compile_attempted_for_visual_chain(blackboard, visual_chain):
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
    legacy_compile_count = _useful_unbound_legacy_compile_count(blackboard)
    if legacy_compile_count:
        ordinal = _visual_candidate_chain_ordinal(blackboard, visual_chain)
        if ordinal and legacy_compile_count >= ordinal:
            return 0
    # Compilation authority is bound to a concrete visual artifact and parser
    # version.  Rows previously compiled for the same DOI cannot prove that
    # this chain was processed by the current parser: doing a source-wide row
    # count here caused v1/v2 rows to suppress mandatory v3 replay.  The
    # artifact-bound action-history check above is the sole completion gate.
    return candidate_count


def _useful_unbound_legacy_compile_count(
    blackboard: dict[str, Any],
) -> int:
    """Count pre-binding compile rows without treating versioned rows as current."""

    count = 0
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != (
            "compile_exact_literature_rows"
        ):
            continue
        if row.get("useful_artifact") is False or row.get("stale") is True:
            continue
        try:
            signature = json.loads(str(row.get("action_signature") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            signature = {}
        payload = dict(signature.get("payload") or {})
        if any(
            str(payload.get(field) or "").strip()
            for field in ("chain_id", "visual_chain_id", "artifact_ref")
        ):
            continue
        count += 1
    return count


def _compile_attempted_for_visual_chain(
    blackboard: dict[str, Any],
    visual_chain: dict[str, Any],
) -> bool:
    refs = {
        str(visual_chain.get(field) or "").strip().casefold()
        for field in ("chain_id", "artifact_ref")
        if str(visual_chain.get(field) or "").strip()
    }
    if not refs:
        return False
    materialized_rounds = [
        int(row.get("round_index") or 0)
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict)
        and str(row.get("action_type") or "")
        == "extract_visual_literature_chain"
        and str(row.get("artifact_ref") or "").strip().casefold() in refs
        and row.get("useful_artifact") is not False
    ]
    latest_materialized_round = max(materialized_rounds, default=0)
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != (
            "compile_exact_literature_rows"
        ):
            continue
        try:
            signature = json.loads(str(row.get("action_signature") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        payload = dict(signature.get("payload") or {})
        attempted_authority = str(
            payload.get("deterministic_parser_authority_id")
            or "autoplanner.opsin_pubchem_source_text.v1"
        )
        if attempted_authority != PARSER_AUTHORITY_ID:
            continue
        attempted_refs = {
            str(payload.get(field) or "").strip().casefold()
            for field in ("chain_id", "visual_chain_id", "artifact_ref")
            if str(payload.get(field) or "").strip()
        }
        if not refs & attempted_refs:
            continue
        compile_round = int(row.get("round_index") or 0)
        # A failed, stale, or completed replay consumes only the concrete
        # visual artifact it was bound to.  It must not be selected forever
        # merely because the compiler produced no exact row.  A later
        # materialization of the same artifact ref deliberately reopens it.
        if latest_materialized_round > compile_round:
            continue
        if row.get("compile_replay_completed") is True and str(
            row.get("compile_parser_authority_id") or ""
        ) == PARSER_AUTHORITY_ID:
            return True
        if row.get("useful_artifact") is False or row.get("stale") is True:
            return True
    return False


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
    steps = [
        dict(step)
        for step in visual_chain.get("steps") or []
        if isinstance(step, dict)
    ]
    if steps:
        # ``exact_ready`` is a whole-chain completeness flag.  A source can
        # contain independently exact, host-checkable steps while other
        # expected labels remain unresolved.  Do not discard those steps just
        # because the aggregate chain is retained as exploratory; the exact
        # compiler and deterministic registry are the promotion boundary.
        compilable_steps = [
            step for step in steps if not _visual_step_is_exploratory_only(step)
        ]
        condition_gap_labels = {
            str(label).strip().casefold()
            for label in visual_chain.get("condition_gap_labels") or []
            if str(label).strip()
        }
        if condition_gap_labels:
            compilable_steps = [
                step
                for step in compilable_steps
                if not {
                    str(step.get("step_id") or "").strip().casefold(),
                    str(step.get("product_label") or "").strip().casefold(),
                }
                & condition_gap_labels
            ]
        return len(compilable_steps)
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


def _visual_step_is_exploratory_only(step: dict[str, Any]) -> bool:
    derivation = dict(step.get("structure_derivation") or {})
    allowed_use = str(
        step.get("allowed_use") or derivation.get("allowed_use") or ""
    ).strip().lower()
    return bool(
        step.get("not_exact_literature_segment")
        or derivation.get("not_exact_literature_segment")
        or derivation.get("approximate_structure")
        or "exploratory" in allowed_use
        or allowed_use == "exploratory_template_and_guided_hint_only"
    )


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
        and not _process_evidence_available(blackboard)
        and _visual_gap_repair_budget_remaining(blackboard)
        and (_condition_gap_repair_needed(blackboard) or not _uncompiled_visual_steps_available(blackboard))
            and _budget_remaining(blackboard, "visual_calls")
    ):
        return True
    if _next_structure_resolution_task_for_local_resolve(blackboard) and _budget_remaining(blackboard, "visual_calls"):
        return True
    if (
        _visual_chain_available(blackboard)
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
        if not (
            row.get("gap_labels")
            or row.get("condition_gap_labels")
            or row.get("extraction_gaps")
            or row.get("missing_expected_labels")
        ):
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
        labels.extend(
            meaningful_compound_labels(
                row.get("expected_scheme_or_compound_labels") or []
            )
        )
    return meaningful_compound_labels(labels)


def _structure_resolution_scout_needed(blackboard: dict[str, Any]) -> bool:
    if _process_evidence_available(blackboard):
        return False
    if _next_structure_resolution_task_for_local_resolve(blackboard) and _budget_remaining(blackboard, "visual_calls"):
        return False
    if (
        _budget_remaining(blackboard, "visual_calls")
        and (_next_local_pdf_source_for_pdf_extraction(blackboard) or _next_local_pdf_source_for_visual_extraction(blackboard))
    ):
        return False
    if _uncompiled_visual_steps_available(blackboard) and not _stale_compile_requires_structure_resolution(blackboard):
        return False
    if _structure_resolution_scout_seen(blackboard):
        return False
    return bool(_open_structure_resolution_tasks(blackboard))


def _open_structure_resolution_tasks(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in (blackboard.get("literature_evidence") or {}).get(
        "structure_resolution_tasks"
    ) or []:
        if not isinstance(row, dict) or str(row.get("status") or "open") != "open":
            continue
        label = str(row.get("label") or "").strip()
        # Missing labels remain compatible with old task records, but an
        # explicit planner sentinel is provenance metadata rather than a
        # resolvable compound and must never consume another Agent call.
        if label and not meaningful_compound_labels([label]):
            continue
        tasks.append(dict(row))
    return tasks


def _next_structure_resolution_task_for_local_resolve(blackboard: dict[str, Any]) -> dict[str, Any]:
    if _next_local_pdf_source_for_pdf_extraction(blackboard) or _next_local_pdf_source_for_visual_extraction(blackboard):
        return {}
    if _uncompiled_visual_steps_available(blackboard) and not _stale_compile_requires_structure_resolution(blackboard):
        return {}
    for task in _prioritized_structure_resolution_tasks(blackboard):
        if _structure_resolution_task_locally_attempted(blackboard, task):
            continue
        return task
    return {}


def _prioritized_structure_resolution_tasks(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = _open_structure_resolution_tasks(blackboard)
    visual_labels = _structure_labels_available_from_visual_chains(blackboard)
    if not visual_labels:
        return tasks
    return sorted(
        tasks,
        key=lambda task: (
            0 if _structure_task_label_in_set(task, visual_labels) else 1,
            str(task.get("label") or ""),
        ),
    )


def _structure_labels_available_from_visual_chains(blackboard: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for chain in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        for step in chain.get("steps") or []:
            if not isinstance(step, dict):
                continue
            product_label = str(step.get("product_label") or step.get("label") or "").strip()
            if product_label:
                labels.add(_structure_label_key(product_label))
            for reactant_label in step.get("reactant_labels") or []:
                text = str(reactant_label or "").strip()
                if text:
                    labels.add(_structure_label_key(text))
    return labels


def _structure_task_label_in_set(task: dict[str, Any], labels: set[str]) -> bool:
    label = _structure_label_key(str(task.get("label") or ""))
    return bool(label and label in labels)


def _structure_label_key(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


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
    task_id = str(task.get("task_id") or "")
    if task_id:
        queue = build_source_capability_queue(
            blackboard,
            round_index=_source_capability_round_index(blackboard),
        )
        matches = matching_source_capabilities(
            queue,
            action_type="resolve_literature_structure_task",
            payload={"task_id": task_id},
        )
        if len(matches) == 1:
            payload = source_capability_effective_payload(payload, matches[0])
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
        if isinstance(audit, dict) and audit.get("strict_source_proof_eligible") is True:
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


def _analogical_templates_need_refresh(blackboard: dict[str, Any]) -> bool:
    templates = [dict(row) for row in blackboard.get("analogical_templates") or [] if isinstance(row, dict)]
    if not templates:
        return False
    if _nonsteroid_target_has_steroid_templates(blackboard, templates):
        return True
    latest_visual = _latest_accepted_visual_chain_with_steps(blackboard)
    if not latest_visual:
        return False
    visual_source = _source_key(latest_visual)
    if not visual_source:
        return False
    return not any(_template_covers_visual_source(template, visual_source) for template in templates)


def _nonsteroid_target_has_steroid_templates(blackboard: dict[str, Any], templates: list[dict[str, Any]]) -> bool:
    target_text = " ".join(
        [
            str((blackboard.get("target_input") or {}).get("target_name") or ""),
            str((blackboard.get("target_input") or {}).get("family_hint") or ""),
            str((blackboard.get("target_profile") or {}).get("target_name") or ""),
            str((blackboard.get("target_profile") or {}).get("family_hint") or ""),
        ]
    ).lower()
    target_is_steroid_family = any(
        token in target_text
        for token in ("steroid", "bufadienolide", "cardenolide", "taxane")
    )
    if target_is_steroid_family:
        return False
    return any(
        "steroid" in str((template.get("reaction_center") or {}).get("product_retron_type") or "").lower()
        or "steroid" in str(template.get("reaction_class") or "").lower()
        for template in templates
    )


def _latest_accepted_visual_chain_with_steps(blackboard: dict[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []
        if isinstance(row, dict)
        and bool(row.get("accepted"))
        and _visual_chain_candidate_step_count(dict(row)) > 0
    ]
    return rows[-1] if rows else {}


def _template_covers_visual_source(template: dict[str, Any], visual_source: str) -> bool:
    retron = str((template.get("reaction_center") or {}).get("product_retron_type") or "")
    if not retron.startswith("visual_") and retron != "steroid_visual_unsaturation_adjustment":
        return False
    hint = dict(template.get("visual_connectivity_hint") or {})
    hint_source = str(hint.get("source_ref") or "")
    return bool(hint_source and _source_keys_match(hint_source, visual_source))


def _source_keys_match(left: str, right: str) -> bool:
    left_key = _normalized_source_key_text(left)
    right_key = _normalized_source_key_text(right)
    return bool(left_key and right_key and left_key == right_key)


def _normalized_source_key_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("ref:"):
        return text
    if text.startswith(("doi:", "pii:")):
        return f"ref:{text}"
    if text.startswith(("pdf:", "title:")):
        return text
    return _source_key({"source_ref": text})


def _ranked_analogical_templates_available(blackboard: dict[str, Any]) -> bool:
    return bool((blackboard.get("analogical_template_ranking") or {}).get("selected_templates"))


def _analogical_template_ranking_needs_refresh(blackboard: dict[str, Any]) -> bool:
    ranking = dict(blackboard.get("analogical_template_ranking") or {})
    selected = [dict(row) for row in ranking.get("selected_templates") or [] if isinstance(row, dict)]
    if not selected:
        return False
    current_template_ids = {
        str(row.get("template_id") or "")
        for row in blackboard.get("analogical_templates") or []
        if isinstance(row, dict)
    }
    selected_ids = {str(row.get("template_id") or "") for row in selected if str(row.get("template_id") or "").strip()}
    if selected_ids and current_template_ids and not selected_ids <= current_template_ids:
        return True
    latest_visual = _latest_accepted_visual_chain_with_steps(blackboard)
    if not latest_visual:
        return False
    visual_source = _source_key(latest_visual)
    if not visual_source:
        return False
    selected_templates = [
        dict(row)
        for row in blackboard.get("analogical_templates") or []
        if isinstance(row, dict) and str(row.get("template_id") or "") in selected_ids
    ]
    if not selected_templates:
        return False
    return not any(_template_covers_visual_source(template, visual_source) for template in selected_templates)


def _template_applications_available(blackboard: dict[str, Any]) -> bool:
    applications = [dict(row) for row in blackboard.get("template_applications") or [] if isinstance(row, dict)]
    if not applications:
        return False
    if _template_applications_only_threshold_rejected(applications) and _selected_templates_are_visual_hints(blackboard):
        return False
    return True


def _template_applications_only_threshold_rejected(applications: list[dict[str, Any]]) -> bool:
    return bool(applications) and all(
        not bool(row.get("accepted"))
        and "template_confidence_below_threshold" in {str(item) for item in row.get("reasons") or []}
        for row in applications
    )


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
        and (not blackboard.get("analogical_templates") or _analogical_templates_need_refresh(blackboard))
        and (
            _analogical_templates_need_refresh(blackboard)
            or not _failed_action_seen(blackboard, "extract_analogical_reaction_templates")
        )
    ):
        return True
    if blackboard.get("analogical_templates") and (
        not blackboard.get("analogical_template_ranking") or _analogical_template_ranking_needs_refresh(blackboard)
    ):
        return True
    if (
        _ranked_analogical_templates_available(blackboard)
        and not _analogical_template_ranking_needs_refresh(blackboard)
        and not _template_applications_available(blackboard)
    ):
        return True
    if _template_applications_need_validation(blackboard):
        return True
    return False


def _analogical_template_payload(blackboard: dict[str, Any], *, action_type: str = "analogical_template_action") -> dict[str, Any]:
    policy = dict((blackboard.get("current_belief") or {}).get("template_policy") or {})
    threshold = str(policy.get("analog_template_confidence_threshold") or "medium")
    if action_type == "apply_analogical_template_to_target" and _selected_templates_are_visual_hints(blackboard):
        threshold = "low"
    return {
        "max_templates": min(12, max(1, int(policy.get("max_template_applications_per_round") or 5) * 2)),
        "max_applications": max(1, int(policy.get("max_template_applications_per_round") or 5)),
        "template_radius_policy": str(policy.get("template_radius_policy") or "auto"),
        "analog_template_confidence_threshold": threshold,
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


def _selected_templates_are_visual_hints(blackboard: dict[str, Any]) -> bool:
    selected_ids = {
        str(row.get("template_id") or "")
        for row in (blackboard.get("analogical_template_ranking") or {}).get("selected_templates") or []
        if isinstance(row, dict) and str(row.get("template_id") or "").strip()
    }
    if not selected_ids:
        return False
    templates = [
        dict(row)
        for row in blackboard.get("analogical_templates") or []
        if isinstance(row, dict) and str(row.get("template_id") or "") in selected_ids
    ]
    return bool(templates) and all(
        str((template.get("reaction_center") or {}).get("product_retron_type") or "").startswith("visual_")
        or str((template.get("reaction_center") or {}).get("product_retron_type") or "") == "steroid_visual_unsaturation_adjustment"
        for template in templates
    )


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
        or _semisynthesis_anchor_smiles_available(blackboard)
        or _validated_template_rows_available(blackboard)
        or _accepted_template_applications_available(blackboard)
        or _visual_exploratory_hints_available(blackboard)
        or _literature_terminal_candidates(blackboard)
        or _process_evidence_available(blackboard)
        or _retrosynthetic_proposals_available(blackboard)
        or blackboard.get("broad_transform_templates")
        or _source_candidates_have_real_source(blackboard)
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
        "resolve_literature_structure_task",
        "search_literature",
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


def _semisynthesis_anchor_smiles_available(blackboard: dict[str, Any]) -> bool:
    return any(
        isinstance(row, dict) and bool(str(row.get("smiles") or "").strip())
        for row in blackboard.get("semisynthesis_anchors") or []
    )


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


def _retrosynthetic_proposals_available(blackboard: dict[str, Any]) -> bool:
    return any(
        isinstance(row, dict)
        and bool(row.get("recursive_expandable") or row.get("executable"))
        and str(row.get("precursor_smiles") or "").strip()
        for row in blackboard.get("retrosynthetic_proposals") or []
    )


def _can_run_guided_chemenzy(blackboard: dict[str, Any]) -> bool:
    budget = dict(blackboard.get("budget_state") or {})
    if int(budget.get("chemenzy_runs") or 0) >= _budget_limit(budget, "max_chemenzy_runs", default=1):
        return False
    if _semisynthesis_anchor_validation_needed(blackboard) and not _hypothesis_only_search_signal_available(blackboard):
        return False
    if _can_stitch_parent_route(blackboard):
        return False
    pending = dict((blackboard.get("current_belief") or {}).get("pending_chemenzy_attempt") or {})
    if str(pending.get("attempt_kind") or "") == "standard":
        return _guided_policy_has_consumable_signal({}, blackboard)
    if _action_count(blackboard, "run_guided_chemenzy") > 0:
        if _guided_failure_requires_new_signal(blackboard) or not _simple_direct_chemenzy_target(blackboard):
            return _new_strong_guided_signal_since_last_run(blackboard)
    template_rows = int(((blackboard.get("current_belief") or {}).get("template_policy") or {}).get("validated_one_step_row_count") or 0)
    return bool(
        blackboard.get("bridge_tasks")
        or _target_relevant_exact_rows_available(blackboard)
        or blackboard.get("analogical_hypothesis_ranking")
        or template_rows
        or _accepted_template_applications_available(blackboard)
        or _visual_exploratory_hints_available(blackboard)
        or _retrosynthetic_proposals_available(blackboard)
        or blackboard.get("broad_transform_templates")
        or _source_candidates_have_real_source(blackboard)
    )



def _can_expand_child_target(blackboard: dict[str, Any]) -> bool:
    if _remaining_child_target_budget(blackboard) <= 0:
        return False
    unattempted_terminals = _unattempted_literature_terminal_candidates(blackboard)
    unattempted_hypothetical = _unattempted_hypothetical_precursor_candidates(blackboard)
    if (
        _semisynthesis_anchor_validation_needed(blackboard)
        and not unattempted_terminals
        and not _hypothesis_frontier_rows_available(unattempted_hypothetical)
    ):
        return False
    if _can_stitch_parent_route(blackboard):
        return False
    if _literature_extraction_pending(blackboard, []) and not _child_expansion_can_run_with_pending_literature(
        blackboard,
        unattempted_terminals=unattempted_terminals,
        unattempted_hypothetical=unattempted_hypothetical,
    ):
        return False
    if _child_expansion_repeated_terminal_blocked(blackboard) and not (unattempted_terminals or unattempted_hypothetical):
        return False
    if unattempted_terminals:
        return True
    if unattempted_hypothetical:
        return True
    return False


def _child_expansion_rationale(blackboard: dict[str, Any]) -> str:
    if _unattempted_literature_terminal_candidates(blackboard):
        return "an exact literature terminal needs upstream child-target expansion"
    hypothetical = _unattempted_hypothetical_precursor_candidates(blackboard)
    if any(str(row.get("source") or "") == "recursive_hypothesis_task" for row in hypothetical):
        return "a pending recursive hypothesis frontier should be tested as a child target"
    if hypothetical:
        return "a hypothesis-only same-core precursor should be tested as a child target"
    return "a child target expansion candidate is available"


def _hypothesis_frontier_available(blackboard: dict[str, Any]) -> bool:
    return _hypothesis_frontier_rows_available(_unattempted_hypothetical_precursor_candidates(blackboard))


def _hypothesis_frontier_rows_available(rows: list[dict[str, Any]]) -> bool:
    return any(
        _hypothesis_frontier_can_bypass_endpoint_validation(row)
        for row in rows
    )


def _hypothesis_only_search_signal_available(blackboard: dict[str, Any]) -> bool:
    return bool(_hypothesis_frontier_available(blackboard) or _retrosynthetic_proposals_available(blackboard))


def _child_expansion_can_run_with_pending_literature(
    blackboard: dict[str, Any],
    *,
    unattempted_terminals: list[dict[str, Any]],
    unattempted_hypothetical: list[dict[str, Any]],
) -> bool:
    if unattempted_terminals:
        return True
    if not unattempted_hypothetical:
        return False
    if _process_evidence_available(blackboard) or blackboard.get("semisynthesis_anchors"):
        return True
    return any(
        _hypothetical_precursor_has_source_backed_anchor(row)
        for row in unattempted_hypothetical
    )


def _hypothetical_precursor_has_source_backed_anchor(row: dict[str, Any]) -> bool:
    if str(row.get("source") or "") == "semisynthesis_anchor":
        return True
    if str(row.get("source_ref") or row.get("source_locator") or "").strip():
        return True
    refs = [str(item).strip().lower() for item in row.get("evidence_refs") or [] if str(item or "").strip()]
    source_prefixes = ("doi:", "http://", "https://", "local_pdf:", "process:", "source:", "visual:", "pdf:")
    return any(ref.startswith(source_prefixes) for ref in refs)


def _hypothesis_frontier_can_bypass_endpoint_validation(row: dict[str, Any]) -> bool:
    if str(row.get("source") or "") == "semisynthesis_anchor":
        return bool(str(row.get("smiles") or "").strip() and (row.get("evidence_refs") or row.get("source_ref")))
    parent_id = str(row.get("parent_candidate_id") or "")
    task_id = str(row.get("recursive_hypothesis_task_id") or "")
    return bool(
        parent_id.startswith("proposal:")
        or ":proposal:" in task_id
        or str(row.get("proposal_id") or "").startswith("proposal:")
        or bool(row.get("requires_precursor_set_stitching"))
        or bool(str(row.get("precursor_set_smiles") or "").strip())
    )


def _can_stitch_parent_route(blackboard: dict[str, Any]) -> bool:
    stitch_count = _action_count(blackboard, "stitch_parent_route")
    binding = _stitch_parent_route_binding(blackboard)
    has_parent_route = bool(str(binding.get("parent_route_ref") or "").strip())
    if _direct_parent_route_proof_ready(blackboard):
        return bool(has_parent_route and stitch_count == 0)
    has_child_and_literature_segment = bool(
        str(binding.get("child_route_ref") or "").strip()
        and str(binding.get("strict_literature_chain_ref") or "").strip()
        and binding.get("all_terminal_frontiers_closed") is True
    )
    if not has_child_and_literature_segment:
        return False
    if stitch_count > 0 and not _new_stitch_signal_since_last_stitch(blackboard):
        return False
    return True


def _new_stitch_signal_since_last_stitch(blackboard: dict[str, Any]) -> bool:
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    last_stitch_index = -1
    for idx, row in enumerate(history):
        if str(row.get("action_type") or "") == "stitch_parent_route":
            last_stitch_index = idx
    if last_stitch_index < 0:
        return True

    signal_action_types = {
        "compile_exact_literature_rows",
        "compile_objective_route_proof",
        "derive_broad_reaction_template",
        "expand_child_target",
        "extract_pdf_literature_structures",
        "extract_visual_literature_chain",
        "resolve_literature_structure_task",
        "run_guided_chemenzy",
        "run_chemenzy",
    }
    for row in history[last_stitch_index + 1 :]:
        action_type = str(row.get("action_type") or "")
        if action_type not in signal_action_types:
            continue
        if row.get("stale"):
            continue
        if bool(row.get("useful_artifact")):
            return True
        delta = dict(row.get("blackboard_delta") or {})
        if any(_positive_delta_value(value) for value in delta.values()):
            return True
    return False


def _positive_delta_value(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _stitchable_visual_literature_chains(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(row, dict):
            continue
        visual = dict(row)
        if not bool(visual.get("accepted")):
            continue
        if _visual_chain_candidate_step_count(visual) <= 0:
            continue
        rows.append(visual)
    return rows


def _best_stitchable_visual_chain(blackboard: dict[str, Any]) -> dict[str, Any]:
    rows = _stitchable_visual_literature_chains(blackboard)
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("step_count") or _visual_chain_candidate_step_count(row)),
            str(row.get("source_ref") or ""),
        ),
    )[0]


def _visual_chain_identifier(row: dict[str, Any]) -> str:
    return str(
        row.get("chain_id")
        or row.get("artifact_ref")
        or row.get("source_ref")
        or row.get("source_title")
        or ""
    ).strip()


def _visual_chain_as_literature_chain_audit(row: dict[str, Any], *, blackboard: dict[str, Any]) -> dict[str, Any]:
    steps = [dict(step) for step in row.get("steps") or row.get("chain") or [] if isinstance(step, dict)]
    chain_target_smiles = str(row.get("target_smiles") or row.get("product_smiles") or "").strip()
    chain_target_name = str(row.get("target_name") or row.get("product_label") or "").strip()
    if steps and not chain_target_smiles:
        first = steps[0]
        chain_target_smiles = str(first.get("product_smiles") or "").strip()
        chain_target_name = str(first.get("product_label") or first.get("product_name") or chain_target_name)
    terminal_smiles = ""
    terminal_name = ""
    if steps:
        last = steps[-1]
        terminal_smiles = str(
            last.get("main_reactant_smiles")
            or (last.get("reactant_smiles") or [""])[0]
            or last.get("final_reactant_smiles")
            or ""
        )
        terminal_name = str(
            last.get("main_reactant_name")
            or (last.get("reactant_labels") or [""])[0]
            or last.get("final_reactant_name")
            or ""
        )
    return {
        "schema_version": "advisory_visual_literature_chain_audit.v1",
        "accepted": bool(row.get("accepted")),
        "case_id": str(blackboard.get("case_id") or ""),
        "target_smiles": chain_target_smiles,
        "target_name": chain_target_name,
        "source_ref": str(row.get("source_ref") or ""),
        "source_title": str(row.get("source_title") or ""),
        "terminal_smiles": terminal_smiles,
        "terminal_name": terminal_name,
        "terminal_reached": bool(terminal_smiles),
        "step_count": int(row.get("step_count") or len(steps)),
        "chain": steps,
        "acceptance_level": str(row.get("acceptance_level") or "advisory_visual_template"),
        "allowed_use": "mechanistic_template_parent_stitch_candidate",
        "not_exact_literature_segment": True,
        "no_solved_claim": True,
        "reasons": [str(item) for item in row.get("reasons") or []],
    }


def _literature_terminal_expansion_pending(blackboard: dict[str, Any]) -> bool:
    return bool(_unattempted_literature_terminal_candidates(blackboard))


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
        if not _recursive_hypothesis_task_is_available(task):
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
                "reaction_family": str(task.get("reaction_family") or ""),
                "reaction_families": [
                    str(item)
                    for item in task.get("reaction_families") or []
                    if str(item or "").strip()
                ],
                "product_retron_type": str(task.get("product_retron_type") or ""),
                "product_retron_types": [
                    str(item)
                    for item in task.get("product_retron_types") or []
                    if str(item or "").strip()
                ],
                "derived_from_retron": str(task.get("derived_from_retron") or ""),
                "retron_authority": "advisory_search_prior_only",
                "proposal_granularity": str(task.get("proposal_granularity") or ""),
                "proposal_score": int(task.get("proposal_score") or 0),
                "route_objective_type": str(task.get("route_objective_type") or ""),
                "source_ref": str(task.get("source_ref") or ""),
                "source_locator": str(task.get("source_locator") or ""),
                "evidence_refs": [str(item) for item in task.get("evidence_refs") or [] if str(item or "").strip()],
                "failure_response_policy": dict(task.get("failure_response_policy") or {}),
                "task_scope": str(task.get("task_scope") or "precursor"),
                "precursor_set_smiles": str(task.get("precursor_set_smiles") or ""),
                "precursor_component_index": int(task.get("precursor_component_index") or 0),
                "precursor_component_count": int(task.get("precursor_component_count") or 1),
                "multi_component_precursor_set": bool(task.get("multi_component_precursor_set")),
                "requires_precursor_set_stitching": bool(task.get("requires_precursor_set_stitching")),
                "sibling_precursor_smiles": [
                    str(item)
                    for item in task.get("sibling_precursor_smiles") or []
                    if str(item or "").strip()
                ],
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
    for anchor in blackboard.get("semisynthesis_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        smiles = str(anchor.get("smiles") or "").strip()
        if not smiles:
            continue
        candidates.append(
            {
                "schema_version": "agentic_semisynthesis_anchor_child_target.v1",
                "name": str(anchor.get("name") or "semisynthesis anchor"),
                "smiles": smiles,
                "source": "semisynthesis_anchor",
                "anchor_id": str(anchor.get("anchor_id") or ""),
                "source_ref": str(anchor.get("source_ref") or ""),
                "source_locator": str(anchor.get("source_locator") or ""),
                "evidence_refs": [str(item) for item in anchor.get("evidence_refs") or [] if str(item or "").strip()],
                "route_objective_type": str(anchor.get("objective_type") or "semisynthesis_anchor"),
                "proposal_granularity": "same_core",
                "proposal_score": 100,
                "task_scope": "source_resolved_semisynthesis_anchor",
                "failure_response_policy": {
                    "on_reject": "search_target_side_conversion_or_source_conditions",
                    "anchor_requires_source_validation": True,
                },
                "risk_flags": ["semisynthesis_anchor_not_parent_proof"],
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


def _recursive_hypothesis_task_is_available(task: dict[str, Any]) -> bool:
    status = str(task.get("status") or "pending").strip().lower()
    if status in {"", "pending", "ready", "queued"}:
        return True
    return False


def _unattempted_literature_terminal_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    attempted = set(_attempted_child_target_smiles(blackboard))
    return [
        row
        for row in _literature_terminal_candidates(blackboard)
        if _child_target_smiles(row) not in attempted
    ]


def _unattempted_hypothetical_precursor_candidates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    attempted = set(_attempted_child_target_smiles(blackboard))
    rows = [
        row
        for row in _hypothetical_precursor_candidates(blackboard)
        if _child_target_smiles(row) not in attempted
    ]
    return sorted(rows, key=_hypothetical_precursor_sort_key)


def _hypothetical_precursor_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
    source = str(row.get("source") or "")
    granularity = str(row.get("proposal_granularity") or "")
    if source == "semisynthesis_anchor":
        source_rank = 0
    elif _visual_or_source_grounded_precursor_candidate(row):
        source_rank = 1
    elif source == "recursive_hypothesis_task":
        source_rank = 2
    else:
        source_rank = 3
    granularity_rank = {"exact": 0, "process": 1, "same_core": 2, "mechanism": 3, "fallback": 4}.get(granularity, 5)
    gap_penalty = 1 if _visual_candidate_has_source_gap_penalty(row) else 0
    return (
        source_rank,
        gap_penalty,
        granularity_rank,
        -int(row.get("proposal_score") or 0),
        int(row.get("recursive_depth") or 0),
        str(row.get("smiles") or ""),
    )


def _visual_or_source_grounded_precursor_candidate(row: dict[str, Any]) -> bool:
    variant = str(row.get("variant_type") or "").lower()
    if "visual_connectivity" in variant or "source_grounded" in variant:
        return True
    flags = " ".join(str(item).lower() for item in row.get("risk_flags") or [])
    if "visual_connectivity" in flags or "exploratory_visual_candidate" in flags:
        return True
    refs = [str(item).strip().lower() for item in row.get("evidence_refs") or [] if str(item or "").strip()]
    return any(ref.startswith(("doi:", "http://", "https://", "visual:", "pdf:", "source:")) for ref in refs)


def _visual_candidate_has_source_gap_penalty(row: dict[str, Any]) -> bool:
    flags = {str(item).strip().lower() for item in row.get("risk_flags") or [] if str(item or "").strip()}
    gap_tokens = {
        "visual_literature_chain_missing_expected_labels",
        "visual_literature_chain_extraction_gaps",
        "intermediate_31_omitted_structure_gap",
    }
    return bool(flags & gap_tokens)


def _attempted_child_target_smiles(blackboard: dict[str, Any]) -> list[str]:
    values: list[str] = [
        str(row.get("canonical_smiles") or row.get("smiles") or "")
        for row in blackboard.get("route_expansion_subgoals") or []
        if isinstance(row, dict)
    ]
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
    pending = dict((blackboard.get("current_belief") or {}).get("pending_chemenzy_attempt") or {})
    if str(pending.get("attempt_kind") or "") == "standard":
        complex_target = _target_is_complex_for_frontier_bootstrap(blackboard)
        max_steps = 20 if complex_target else 6
        iterations = 50 if complex_target else 10
        expansion_topk = 100 if complex_target else 50
        timeout_s = _guided_retry_runtime_budget()["timeout_s"]
        payload.update(
            {
                "attempt_kind": "standard",
                "search_preset": "thorough" if complex_target else "quick",
                "search_mode": "guided_standard_after_probe",
                "max_steps": max_steps,
                "chem_enzy_iterations": iterations,
                "chem_enzy_expansion_topk": expansion_topk,
                "timeout_s": timeout_s,
            }
        )
        _sync_guided_policy_budget(
            payload,
            max_steps=max_steps,
            iterations=iterations,
            expansion_topk=expansion_topk,
            timeout_s=timeout_s,
            search_mode="guided_standard_after_probe",
        )
        return payload
    if attempt > 1:
        retry_budget = _guided_retry_runtime_budget()
        max_steps = retry_budget["max_steps"]
        iterations = retry_budget["iterations"]
        expansion_topk = retry_budget["expansion_topk"]
        timeout_s = retry_budget["timeout_s"]
        payload.update(
            {
                "search_preset": "bounded_retry",
                "search_mode": "guided_retry_after_initial_probe",
                "max_steps": max_steps,
                "chem_enzy_iterations": iterations,
                "chem_enzy_expansion_topk": expansion_topk,
                "timeout_s": timeout_s,
            }
        )
        _sync_guided_policy_budget(
            payload,
            max_steps=max_steps,
            iterations=iterations,
            expansion_topk=expansion_topk,
            timeout_s=timeout_s,
            search_mode="guided_retry_after_initial_probe",
        )
    return payload


def _sync_guided_policy_budget(
    payload: dict[str, Any],
    *,
    max_steps: int,
    iterations: int,
    expansion_topk: int,
    timeout_s: int | None = None,
    search_mode: str,
) -> None:
    policy = dict(payload.get("chem_enzy_search_policy") or payload.get("search_policy") or {})
    if not policy:
        return
    budget = dict(policy.get("budget") or {})
    budget.update(
        {
            "max_depth": int(max_steps),
            "max_iterations": int(iterations),
            "expansion_topk": int(expansion_topk),
        }
    )
    if timeout_s:
        budget["timeout_s"] = int(timeout_s)
    policy["budget"] = budget
    source_budget = dict(policy.get("source_budget") or {})
    source_budget["initial_scan_allowed"] = False
    policy["source_budget"] = source_budget
    compiler = dict(policy.get("compiler_metadata") or {})
    compiler["initial_scan_probe"] = False
    compiler["high_budget_retry_after_initial_probe"] = True
    compiler["requires_verifier"] = True
    compiler["no_solved_claim"] = True
    policy["compiler_metadata"] = compiler
    policy["search_mode"] = str(search_mode or "guided_retry")
    payload["search_policy"] = policy
    payload["chem_enzy_search_policy"] = policy


def _direct_initial_guided_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    payload = _guided_retry_payload(blackboard)
    payload.update(
        {
            "initial_probe": True,
            "search_mode": "direct_parent_initial_probe",
            "search_intent": "bounded direct parent-side retrosynthesis",
            "max_steps": 6,
            "chem_enzy_iterations": 10,
            "chem_enzy_expansion_topk": 20,
            "timeout_s": 180,
            "max_candidates": 5,
            "no_solved_claim": True,
        }
    )
    policy = dict(payload.get("search_policy") or payload.get("chem_enzy_search_policy") or {})
    budget = dict(policy.get("budget") or {})
    budget.update(
        {
            "max_depth": 6,
            "max_iterations": 10,
            "expansion_topk": 20,
            "timeout_s": 180,
        }
    )
    policy["budget"] = budget
    source_budget = dict(policy.get("source_budget") or {})
    source_budget["initial_scan_allowed"] = True
    source_budget["max_candidates"] = 5
    policy["source_budget"] = source_budget
    compiler = dict(policy.get("compiler_metadata") or {})
    compiler["initial_scan_probe"] = True
    compiler["requires_verifier"] = True
    compiler["no_solved_claim"] = True
    policy["compiler_metadata"] = compiler
    policy["search_mode"] = "direct_parent_initial_probe"
    payload["search_policy"] = policy
    payload["chem_enzy_search_policy"] = policy
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
    remaining_budget = _remaining_child_target_budget(blackboard)
    target_limit = max(1, min(2, remaining_budget))
    retry_budget = _guided_retry_runtime_budget()
    payload: dict[str, Any] = {
        "expansion_attempt": attempt,
        "target_offset": max(0, (attempt - 1) * 2),
        "max_targets": target_limit,
        "search_preset": "bounded_retry",
        "chem_enzy_iterations": retry_budget["iterations"],
        "chem_enzy_expansion_topk": retry_budget["expansion_topk"],
        "timeout_s": retry_budget["timeout_s"],
        "max_unexplained_heavy_atom_jump": 12,
    }
    terminal_blacklist = _terminal_blacklist_from_blackboard(blackboard)
    if terminal_blacklist:
        payload["terminal_blacklist"] = terminal_blacklist
    terminals = _unattempted_literature_terminal_candidates(blackboard)
    if terminals:
        payload["target_offset"] = 0
        payload["max_targets"] = min(target_limit, len(terminals))
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
                        "terminal_candidate": _compact_child_policy_row(row),
                    },
                    "source_budget": {
                        "preferred_reaction_classes": ["source_detail_terminal_upstream_expansion"],
                        "strict_source_detail_terminal": bool(row.get("strict_source_proof_eligible")),
                        "advisory_source_detail_terminal": not bool(row.get("strict_source_proof_eligible")),
                        "requires_all_source_frontiers_closed": True,
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
            for idx, row in enumerate(terminals[: payload["max_targets"]], start=1)
        ]
    else:
        hypothetical_precursors = _unattempted_hypothetical_precursor_candidates(blackboard)
        if hypothetical_precursors:
            payload["target_offset"] = 0
            payload["max_targets"] = min(target_limit, len(hypothetical_precursors))
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
                    "anchor_id": str(row.get("anchor_id") or ""),
                    "source_ref": str(row.get("source_ref") or ""),
                    "source_locator": str(row.get("source_locator") or ""),
                    "evidence_refs": [str(item) for item in row.get("evidence_refs") or [] if str(item or "").strip()],
                    "recursive_depth": int(row.get("recursive_depth") or 0),
                    "parent_smiles": str(row.get("parent_smiles") or ""),
                    "parent_candidate_id": str(row.get("parent_candidate_id") or ""),
                    "reaction_family": str(row.get("reaction_family") or ""),
                    "reaction_families": list(row.get("reaction_families") or []),
                    "product_retron_type": str(row.get("product_retron_type") or ""),
                    "product_retron_types": list(row.get("product_retron_types") or []),
                    "derived_from_retron": str(row.get("derived_from_retron") or ""),
                    "retron_authority": "advisory_search_prior_only",
                    "proposal_granularity": str(row.get("proposal_granularity") or ""),
                    "proposal_score": int(row.get("proposal_score") or 0),
                    "route_objective_type": str(row.get("route_objective_type") or ""),
                    "failure_response_policy": dict(row.get("failure_response_policy") or {}),
                    "task_scope": str(row.get("task_scope") or ""),
                    "precursor_set_smiles": str(row.get("precursor_set_smiles") or ""),
                    "precursor_component_index": int(row.get("precursor_component_index") or 0),
                    "precursor_component_count": int(row.get("precursor_component_count") or 1),
                    "multi_component_precursor_set": bool(row.get("multi_component_precursor_set")),
                    "requires_precursor_set_stitching": bool(row.get("requires_precursor_set_stitching")),
                    "sibling_precursor_smiles": [
                        str(item)
                        for item in row.get("sibling_precursor_smiles") or []
                        if str(item or "").strip()
                    ],
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
                        "anchor_whitelist": [str(row.get("smiles") or "")]
                        if str(row.get("source") or "") == "semisynthesis_anchor"
                        else [],
                        "preferred_subgoal": {
                            "schema_version": "hypothetical_precursor_subgoal.v1",
                            "preferred_subgoals": [str(row.get("name") or ""), str(row.get("smiles") or "")],
                            "hypothetical_precursor_target": _compact_child_policy_row(row),
                            "recursive_hypothesis_task": (
                                _compact_child_policy_row(row)
                                if str(row.get("source") or "") == "recursive_hypothesis_task"
                                else {}
                            ),
                            "semisynthesis_anchor": (
                                _compact_child_policy_row(row)
                                if str(row.get("source") or "") == "semisynthesis_anchor"
                                else {}
                            ),
                            "analogy_is_advisory_only": True,
                            "precursor_set_smiles": str(row.get("precursor_set_smiles") or ""),
                            "precursor_component_index": int(row.get("precursor_component_index") or 0),
                            "precursor_component_count": int(row.get("precursor_component_count") or 1),
                            "requires_precursor_set_stitching": bool(row.get("requires_precursor_set_stitching")),
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
                                    str(row.get("reaction_family") or ""),
                                    *[str(item) for item in row.get("reaction_families") or []],
                                    str(row.get("derived_from_retron") or ""),
                                    str(row.get("variant_type") or ""),
                                    str(row.get("proposal_granularity") or ""),
                                    str(row.get("route_objective_type") or ""),
                                ]
                            ),
                            "preferred_retrons": _dedupe(
                                [
                                    str(row.get("product_retron_type") or ""),
                                    str(row.get("derived_from_retron") or ""),
                                    *[str(item) for item in row.get("product_retron_types") or []],
                                ]
                            ),
                            "reaction_and_retron_priors_are_advisory_only": True,
                            "hypothesis_precursor_hint": True,
                            "hypothesis_precursor_hints_are_not_proof": True,
                            "semisynthesis_anchor_hint": str(row.get("source") or "") == "semisynthesis_anchor",
                            "source_ref": str(row.get("source_ref") or ""),
                            "recursive_hypothesis_frontier": str(row.get("source") or "") == "recursive_hypothesis_task",
                            "recursive_depth": int(row.get("recursive_depth") or 0),
                            "parent_smiles": str(row.get("parent_smiles") or ""),
                            "proposal_granularity": str(row.get("proposal_granularity") or ""),
                            "route_objective_type": str(row.get("route_objective_type") or ""),
                            "failure_response_policy": dict(row.get("failure_response_policy") or {}),
                            "precursor_set_smiles": str(row.get("precursor_set_smiles") or ""),
                            "precursor_component_index": int(row.get("precursor_component_index") or 0),
                            "precursor_component_count": int(row.get("precursor_component_count") or 1),
                            "requires_precursor_set_stitching": bool(row.get("requires_precursor_set_stitching")),
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
                            "semisynthesis_anchor_hint": str(row.get("source") or "") == "semisynthesis_anchor",
                            "recursive_hypothesis_frontier": str(row.get("source") or "") == "recursive_hypothesis_task",
                        },
                    },
                }
                for idx, row in enumerate(hypothetical_precursors[: payload["max_targets"]], start=1)
            ]
    return payload


def _compact_child_policy_row(row: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "name": str(row.get("name") or row.get("target_name") or ""),
        "smiles": str(row.get("smiles") or row.get("precursor_smiles") or row.get("target_smiles") or ""),
        "source": str(row.get("source") or ""),
        "source_ref": str(row.get("source_ref") or ""),
        "source_locator": str(row.get("source_locator") or ""),
        "recursive_hypothesis_task_id": str(row.get("recursive_hypothesis_task_id") or row.get("task_id") or ""),
        "parent_candidate_id": str(row.get("parent_candidate_id") or ""),
        "parent_smiles": str(row.get("parent_smiles") or ""),
        "reaction_family": str(row.get("reaction_family") or ""),
        "reaction_families": list(row.get("reaction_families") or []),
        "product_retron_type": str(row.get("product_retron_type") or ""),
        "product_retron_types": list(row.get("product_retron_types") or []),
        "derived_from_retron": str(row.get("derived_from_retron") or ""),
        "retron_authority": str(row.get("retron_authority") or ""),
        "template_id": str(row.get("template_id") or ""),
        "application_id": str(row.get("application_id") or ""),
        "anchor_id": str(row.get("anchor_id") or ""),
        "proposal_granularity": str(row.get("proposal_granularity") or ""),
        "task_scope": str(row.get("task_scope") or ""),
        "precursor_set_smiles": str(row.get("precursor_set_smiles") or ""),
        "precursor_component_index": int(row.get("precursor_component_index") or 0),
        "precursor_component_count": int(row.get("precursor_component_count") or 1),
        "requires_precursor_set_stitching": bool(row.get("requires_precursor_set_stitching")),
        "no_solved_claim": True,
    }
    evidence_refs = [str(item) for item in row.get("evidence_refs") or [] if str(item or "").strip()]
    if evidence_refs:
        compact["evidence_refs"] = evidence_refs[:6]
    return {key: value for key, value in compact.items() if value not in ("", [], {})}


def _remaining_child_target_budget(blackboard: dict[str, Any]) -> int:
    budget = dict(blackboard.get("budget_state") or {})
    try:
        used = int(budget.get("child_target_runs") or 0)
    except (TypeError, ValueError):
        return 0
    maximum = _budget_limit(budget, "max_child_target_runs", default=2)
    return max(0, maximum - used)


def planned_child_target_count(payload: dict[str, Any]) -> int:
    """Return the number of child runs explicitly requested by a payload.

    Codex commonly emits one target in a singular field (``subgoal_target``,
    ``child_target`` or a direct ``target_smiles``) while leaving
    ``max_targets`` unset.  Treating that shape as the historical default of
    two made an otherwise valid one-target action exceed a one-run global
    budget and disappear during repair.  The default remains useful only for
    policy-rebuild payloads that contain no explicit target at all.
    """
    rows = payload.get("subgoal_targets") or payload.get("child_targets") or []
    try:
        max_targets = max(1, int(payload.get("max_targets") or 2))
    except (TypeError, ValueError):
        max_targets = 2
    if isinstance(rows, list) and rows:
        return max(1, min(len(rows), max_targets))
    for field in ("subgoal_target", "child_target", "target"):
        value = payload.get(field)
        if isinstance(value, dict) and value:
            return 1
        if isinstance(value, str) and value.strip():
            return 1
    if any(str(payload.get(field) or "").strip() for field in ("target_smiles", "smiles")):
        return 1
    return max_targets


def _planned_child_target_count(payload: dict[str, Any]) -> int:
    """Backward-compatible private alias for older callers/tests."""
    return planned_child_target_count(payload)


def _hypothetical_precursor_evidence_refs(blackboard: dict[str, Any], row: dict[str, Any]) -> list[str]:
    refs = [
        str(row.get("recursive_hypothesis_task_id") or ""),
        str(row.get("parent_candidate_id") or ""),
        str(row.get("parent_subgoal_name") or ""),
        str(row.get("precursor_set_smiles") or ""),
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
    direct_parent_mode = _direct_parent_route_proof_ready(blackboard)
    payload = {
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
            "child_route_connectivity_required": not direct_parent_mode,
            "exact_literature_connectivity_required": not direct_parent_mode,
            "direct_parent_route_verifier_allowed": direct_parent_mode,
            "proof_mode": "direct_parent_route" if direct_parent_mode else "stitched_parent_route",
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
    return payload


def _stitch_parent_route_binding(blackboard: dict[str, Any]) -> dict[str, Any]:
    refs = dict(blackboard.get("artifact_refs") or {})
    direct_parent_mode = _direct_parent_route_proof_ready(blackboard)
    exact_row_ids = [
        str(row.get("row_id") or row.get("source_template_id") or row.get("template_id") or "")
        for row in _target_relevant_exact_literature_rows(blackboard)
        if isinstance(row, dict) and str(row.get("row_id") or row.get("source_template_id") or row.get("template_id") or "").strip()
    ]
    visual_chain_ids = [
        _visual_chain_identifier(row)
        for row in _stitchable_visual_literature_chains(blackboard)
        if _visual_chain_identifier(row)
    ]
    child_ref = str(refs.get("route_expansion_subgoal_search") or _first_ref_containing(refs, "route_expansion") or "")
    parent_ref = str(refs.get("guided_chemenzy") or _first_ref_containing(refs, "guided_chemenzy") or _first_ref_containing(refs, "route_verifier") or "")
    exact_ref = str(
        refs.get("compile_source_detail_chain_route")
        or _first_ref_containing(refs, "compile_exact_literature_rows")
        or _first_ref_containing(refs, "compiled_source_detail")
        or ""
    )
    strict_audits = _strict_source_detail_chain_audits(blackboard)
    ready_audits = [audit for audit in strict_audits if _strict_chain_frontier_coverage(blackboard, audit)["accepted"]]
    selected_strict_audit = dict(ready_audits[0]) if ready_audits else {}
    coverage = _strict_chain_frontier_coverage(blackboard, selected_strict_audit) if selected_strict_audit else {
        "accepted": False,
        "terminal_frontier": [],
        "accepted_frontier": [],
        "missing_frontier": [],
    }
    strict_ref = str(selected_strict_audit.get("artifact_ref") or exact_ref or "")
    missing_inputs: list[str] = []
    if not child_ref:
        missing_inputs.append("child_route_ref_missing")
    if not parent_ref:
        missing_inputs.append("parent_route_ref_missing")
    if not strict_audits:
        missing_inputs.append("strict_source_detail_chain_missing")
    elif not selected_strict_audit:
        missing_inputs.append("source_detail_terminal_frontier_coverage_incomplete")
    input_refs = _dedupe(
        [
            child_ref,
            parent_ref,
            exact_ref,
            strict_ref,
            *[f"exact_row:{row_id}" for row_id in exact_row_ids],
            *[f"visual_chain:{chain_id}" for chain_id in visual_chain_ids],
        ]
    )
    return {
        "schema_version": "agentic_parent_stitch_binding.v1",
        "proof_mode": "direct_parent_route" if direct_parent_mode else "stitched_parent_route",
        "direct_parent_route_verifier_ready": direct_parent_mode,
        "child_route_ref": child_ref,
        "parent_route_ref": parent_ref,
        "exact_literature_segment_ref": exact_ref,
        "strict_literature_chain_ref": strict_ref,
        "strict_literature_chain_audit_id": str(selected_strict_audit.get("audit_id") or ""),
        "exact_literature_row_ids": _dedupe(exact_row_ids),
        "visual_literature_chain_ids": _dedupe(visual_chain_ids),
        "visual_literature_chains_are_advisory_only": True,
        "literature_segment_level": (
            "strict_source_detail"
            if selected_strict_audit
            else "advisory_visual_template"
            if visual_chain_ids
            else "missing"
        ),
        "all_terminal_frontiers_closed": bool(coverage.get("accepted")),
        "terminal_frontier": list(coverage.get("terminal_frontier") or []),
        "accepted_terminal_frontier": list(coverage.get("accepted_frontier") or []),
        "missing_terminal_frontier": list(coverage.get("missing_frontier") or []),
        "input_refs": [str(item) for item in input_refs if str(item or "").strip()],
        "missing_inputs": missing_inputs,
    }


def _strict_source_detail_chain_audits(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    return [
        dict(audit)
        for audit in evidence.get("exact_chain_audits") or []
        if isinstance(audit, dict)
        and audit.get("strict_source_proof_eligible") is True
        and [str(item) for item in audit.get("terminal_frontier") or [] if str(item or "").strip()]
        and str(audit.get("artifact_ref") or "").strip()
    ]


def _strict_chain_frontier_coverage(
    blackboard: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    frontier = _dedupe(
        [str(item).strip() for item in audit.get("terminal_frontier") or [] if str(item or "").strip()]
    )
    accepted = {
        str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
        for row in blackboard.get("route_expansion_subgoals") or []
        if isinstance(row, dict) and row.get("accepted") is True
    }
    accepted_frontier = [smiles for smiles in frontier if smiles in accepted]
    missing_frontier = [smiles for smiles in frontier if smiles not in accepted]
    return {
        "schema_version": "agent_source_detail_frontier_coverage.v1",
        "accepted": bool(frontier) and not missing_frontier,
        "terminal_frontier": frontier,
        "accepted_frontier": accepted_frontier,
        "missing_frontier": missing_frontier,
        "requires_all_frontiers_closed": True,
    }


def _direct_parent_route_proof_ready(blackboard: dict[str, Any]) -> bool:
    belief = dict(blackboard.get("current_belief") or {})
    verifier = dict(belief.get("parent_route_verifier") or {})
    if not verifier:
        return False
    reasons = {str(item) for item in verifier.get("reasons") or [] if str(item or "").strip()}
    try:
        accepted_route_count = int(verifier.get("accepted_route_count") or 0)
        best_route_step_count = int(verifier.get("best_route_step_count") or 0)
    except (TypeError, ValueError):
        accepted_route_count = 0
        best_route_step_count = 0
    reason_blocked = (
        accepted_route_count <= 0
        and ("large_atom_jump" in reasons or "unexplained_large_atom_jump" in reasons)
    )
    return bool(
        verifier.get("schema_version") == "agent_parent_route_verifier_summary.v1"
        and verifier.get("verifier_schema_version") == "harness_route_verifier_report.v1"
        and verifier.get("accepted") is True
        and verifier.get("solved") is True
        and str(verifier.get("verification_level") or "")
        in {"L2_reaction_validated", "L3_precedent_supported", "L4_procurement_ready"}
        and verifier.get("reaction_validated") is True
        and verifier.get("target_match") is True
        and accepted_route_count > 0
        and verifier.get("best_route_rank") is not None
        and best_route_step_count > 0
        and isinstance(verifier.get("reasons"), list)
        and not verifier["reasons"]
        and isinstance(verifier.get("warnings"), list)
        and not reason_blocked
        and str(verifier.get("artifact_ref") or "").strip()
    )


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
    return int(budget.get(field) or 0) < _budget_limit(budget, max_field, default=fallback)


def _budget_limit(budget: dict[str, Any], key: str, *, default: int) -> int:
    try:
        parsed = int(budget[key]) if key in budget and budget.get(key) is not None else int(default)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, parsed)


def _stale_action_repeated(blackboard: dict[str, Any], action: dict[str, Any]) -> bool:
    signatures = _action_signature_variants(action)
    stale_count = 0
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict):
            continue
        if row.get("stale") and row.get("action_signature") in signatures:
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
    return json.dumps(
        {
            "action_type": action.get("action_type"),
            "payload": _compact_action_signature_payload(payload),
            "payload_hash": hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16],
        },
        sort_keys=True,
        default=str,
    )


def _action_signature_variants(action: dict[str, Any]) -> set[str]:
    payload = {key: value for key, value in dict(action.get("payload") or {}).items() if key != "timestamp"}
    return {
        _action_signature(action),
        json.dumps({"action_type": action.get("action_type"), "payload": payload}, sort_keys=True, default=str),
    }


def _compact_action_signature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    scalar_keys = [
        "source_ref",
        "doi",
        "pii",
        "url",
        "pdf_path",
        "source_pdf_path",
        "chain_id",
        "visual_chain_id",
        "artifact_ref",
        "query",
        "search_intent",
        "search_mode",
        "focused_gap_repair",
        "focused_structure_resolution",
        "task_id",
        "compound_label",
        "source_capability_id",
        "deterministic_parser_authority_id",
        "compile_attempt",
        "expansion_attempt",
        "timeout_s",
        "max_steps",
        "max_candidates",
    ]
    for key in scalar_keys:
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            compact[key] = value
    for key, limit in {
        "queries": 4,
        "search_queries": 4,
        "expected_labels": 8,
        "page_numbers": 8,
        "template_ids": 6,
        "hypothesis_ids": 6,
        "selected_analogy_hypothesis_ids": 6,
    }.items():
        values = [item for item in payload.get(key) or [] if str(item or "").strip()]
        if values:
            compact[key] = values[:limit]
            if len(values) > limit:
                compact[f"{key}_count"] = len(values)
    subgoals = payload.get("subgoal_targets") or payload.get("child_targets") or []
    if isinstance(subgoals, list) and subgoals:
        compact["subgoal_targets"] = [_compact_signature_target(row) for row in subgoals[:6] if isinstance(row, dict)]
        compact["subgoal_target_count"] = len(subgoals)
    policy = payload.get("search_policy") or payload.get("chem_enzy_search_policy")
    if isinstance(policy, dict):
        compact["search_policy_summary"] = {
            "policy_id": str(policy.get("policy_id") or ""),
            "mode": str(policy.get("mode") or ""),
            "active_bridge_tasks": len(policy.get("active_bridge_tasks") or []),
            "terminal_blacklist": len(policy.get("terminal_blacklist") or []),
            "accepted_exact_row_ids": len(policy.get("accepted_exact_row_ids") or []),
        }
    repair = payload.get("codex_payload_repair")
    if isinstance(repair, dict):
        compact["codex_payload_repair"] = {
            "action_type": str(repair.get("action_type") or ""),
            "completed_from_blackboard": bool(repair.get("completed_from_blackboard")),
        }
    return compact


def _compact_signature_target(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(row.get(key) or "")
        for key in ("task_id", "label", "name", "smiles", "canonical_smiles")
        if str(row.get(key) or "").strip()
    }


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
