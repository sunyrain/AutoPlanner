"""Typed blackboard state for policy-driven agentic controller runs."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:  # RDKit is expected in normal AutoPlanner runs, but tests can mock around it.
    from rdkit import Chem
except Exception:  # pragma: no cover - exercised only in stripped environments.
    Chem = None  # type: ignore[assignment]

from cascade_planner.harness.agent_action_planner import (
    build_guided_chemenzy_payload_from_blackboard,
    planned_child_target_count,
)
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.harness.analogical_reaction_templates import compact_template_application_summary
from cascade_planner.harness.process_evidence import semisynthesis_anchors_from_process_rows
from cascade_planner.harness.recursive_hypothesis_tasks import (
    recursive_hypothesis_tasks_from_route_expansion,
)
from cascade_planner.harness.retrosynthetic_proposals import (
    compile_retrosynthetic_proposal_bus,
    deduplicate_retrosynthetic_proposals,
)
from cascade_planner.harness.route_objectives import (
    build_broad_transform_templates_from_blackboard,
    classify_route_objectives,
)
from cascade_planner.harness.route_verifier import (
    is_reaction_validated_route_verifier_report,
)
from cascade_planner.harness.schemas import write_json
from cascade_planner.harness.source_capabilities import (
    pdf_evidence_has_materialized_render,
)
from cascade_planner.harness.stitched_route import is_validated_source_detail_literature_step
from cascade_planner.harness.target_side_strategy import build_target_side_disconnection_hypotheses
from cascade_planner.agent.action_contracts import PLANNER_SOURCE_HINT_SCHEMA
from cascade_planner.source_locators import (
    independent_source_group,
    source_content_scope,
    source_document_identity,
    source_record_representations,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisCostLedger,
    RetrosynthesisRunBudget,
)


AGENT_BLACKBOARD_SCHEMA = "agent_blackboard.v1"


def initialize_agent_blackboard(
    *,
    target_input: dict[str, Any],
    preflight: dict[str, Any],
    max_rounds: int = 3,
    budget_limits: dict[str, Any] | None = None,
    prior_artifacts: dict[str, Any] | None = None,
    acceptance_spec: RetrosynthesisAcceptanceSpec | None = None,
    run_budget: RetrosynthesisRunBudget | None = None,
) -> dict[str, Any]:
    profile = dict(preflight.get("target_profile") or {})
    limits = dict(budget_limits or {})
    max_scout_calls = _nonnegative_int(limits.get("max_scout_calls"), 3)
    max_visual_calls = _nonnegative_int(limits.get("max_visual_calls"), 3)
    max_chemenzy_runs = _nonnegative_int(
        _first_present(limits, "max_guided_chemenzy_runs", "max_chem_enzy_runs", "max_chemenzy_runs"),
        1,
    )
    max_child_target_runs = _nonnegative_int(
        _first_present(limits, "max_route_expansion_subgoal_runs", "max_child_target_runs"),
        2,
    )
    max_codex_research_runs = _nonnegative_int(limits.get("max_codex_research_runs"), 1)
    max_template_applications_per_round = _positive_int(limits.get("max_template_applications_per_round"), 5)
    acceptance = acceptance_spec or RetrosynthesisAcceptanceSpec()
    cost_budget = run_budget or RetrosynthesisRunBudget()
    board = {
        "schema_version": AGENT_BLACKBOARD_SCHEMA,
        "case_id": str(preflight.get("case_id") or target_input.get("case_id") or "target"),
        "target_profile": {
            "schema_version": "agent_target_profile_summary.v1",
            "target_name": str(target_input.get("target_name") or profile.get("target_name") or ""),
            "target_smiles": str(target_input.get("target_smiles") or profile.get("input_smiles") or ""),
            "canonical_smiles": str(preflight.get("canonical_smiles") or profile.get("canonical_smiles") or ""),
            "isomeric_smiles": str(preflight.get("isomeric_smiles") or profile.get("isomeric_smiles") or ""),
            "inchi_key": str(preflight.get("inchi_key") or profile.get("inchi_key") or ""),
            "valid": bool(preflight.get("accepted")),
            "heavy_atoms": int(profile.get("heavy_atoms") or 0),
            "rings": int(profile.get("rings") or 0),
            "functional_handles": list(profile.get("family_hints") or []),
            "family_hint": str(target_input.get("family_hint") or ""),
        },
        "route_failures": [],
        "plugin_runtime_diagnostics": [],
        "literature_evidence": {
            "schema_version": "agent_literature_evidence_summary.v1",
            "source_candidates": [],
            "planner_source_hints": [],
            "source_lifecycle": [],
            "pdf_structure_evidence": [],
            "visual_chains": [],
            "process_evidence_rows": [],
            "exact_rows": [],
            "terminal_candidates": [],
            "structure_resolution_tasks": [],
            "structure_resolution_attempts": [],
            "resolved_structures": [],
            "source_refs": [],
            "confidence": "none",
        },
        "analogical_hypotheses": [],
        "analogical_hypothesis_ranking": {},
        "analogical_templates": [],
        "analogical_template_ranking": {},
        "template_applications": [],
        "template_cache_refs": {},
        "template_failure_memory": [],
        "route_objective_summary": {},
        "endpoint_candidates": [],
        "objective_evidence_cards": [],
        "broad_transform_templates": [],
        "reaction_idea_cards": [],
        "retrosynthetic_proposals": [],
        "retrosynthetic_proposal_compile_report": {},
        "proposal_failure_feedback": [],
        "route_proof_bundle": {},
        "chemenzy_route_proof_banks": [],
        "chemenzy_attempts": [],
        "semisynthesis_anchors": [],
        "recursive_hypothesis_tasks": [],
        "route_expansion_subgoals": [],
        "bridge_tasks": [],
        "terminal_blacklist": [],
        "planner_history": [],
        "action_history": [],
        "budget_state": {
            "schema_version": "agent_blackboard_budget_state.v1",
            "rounds_completed": 0,
            "max_rounds": int(max_rounds or 3),
            "scout_calls": 0,
            "max_scout_calls": max_scout_calls,
            "visual_calls": 0,
            "max_visual_calls": max_visual_calls,
            "chemenzy_runs": 0,
            "max_chemenzy_runs": max_chemenzy_runs,
            "child_target_runs": 0,
            "max_child_target_runs": max_child_target_runs,
            "codex_research_runs": 0,
            "max_codex_research_runs": max_codex_research_runs,
            "codex_action_planner_runs": 0,
            "template_application_actions": 0,
            "max_template_application_actions": _nonnegative_int(limits.get("max_template_application_actions"), 3),
        },
        "current_belief": {
            "schema_version": "agent_current_belief.v1",
            "promising_directions": [],
            "blocked_directions": [],
            "next_action_bias": [],
            "constraints": {
                "target_core_retention_required": True,
                "max_unexplained_heavy_atom_jump": 15,
                "max_recursive_hypothesis_depth": _positive_int(limits.get("max_recursive_hypothesis_depth"), 3),
            },
            "template_policy": {
                "enabled": bool(limits.get("enable_analogical_templates", True)),
                "max_template_applications_per_round": max_template_applications_per_round,
                "template_radius_policy": str(limits.get("template_radius_policy") or "auto"),
                "analog_template_confidence_threshold": str(limits.get("analog_template_confidence_threshold") or "medium"),
                "analogy_is_advisory_only": True,
            },
            "stop_candidates": [],
            "child_route_solved": False,
            "parent_route_verifier": {},
        },
        "artifact_refs": dict((prior_artifacts or {}).get("artifact_refs") or {}),
        "parent_route_proof": {},
        "retrosynthesis_run_contract": {
            "schema_version": "retrosynthesis_run_contract.v1",
            "acceptance_spec": acceptance.to_dict(),
            "cost_ledger": RetrosynthesisCostLedger(
                budget=cost_budget
            ).to_dict(),
            "semantics": {
                "one_acceptance_definition_for_search_and_closeout": True,
                "run_wide_model_cost_gate": True,
                "blackboard_is_coordination_not_chemistry_authority": True,
            },
        },
        "route_deficit_queue": {},
    }
    _seed_target_literature_sources(board, target_input=target_input)
    return board


def refresh_target_derived_blackboard_priors(
    blackboard: dict[str, Any],
    *,
    target_input: dict[str, Any],
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh target-derived priors after code/prompt upgrades.

    The function only rewrites deterministic target-side priors. It preserves
    acquired source evidence, visual chains, resolved structures, route
    failures, and historical action/tool records.
    """
    board = deepcopy(blackboard)
    target_profile = dict(board.get("target_profile") or {})
    target_name = str(target_input.get("target_name") or target_profile.get("target_name") or "")
    target_smiles = str(target_input.get("target_smiles") or target_profile.get("target_smiles") or "")
    family_hint = str(target_input.get("family_hint") or target_profile.get("family_hint") or "")
    case_id = str((preflight or {}).get("case_id") or board.get("case_id") or "")
    evidence = dict(board.get("literature_evidence") or {})
    source_refs = [str(item) for item in evidence.get("source_refs") or [] if str(item or "").strip()]
    failure_reasons = [
        str(row.get("reason") or "")
        for row in board.get("route_failures") or []
        if isinstance(row, dict) and str(row.get("reason") or "").strip()
    ]
    objective_summary = classify_route_objectives(
        target_smiles=target_smiles,
        target_name=target_name,
        family_hint=family_hint,
        failure_reasons=failure_reasons,
        source_evidence_refs=source_refs,
        case_id=case_id,
    )
    target_side = build_target_side_disconnection_hypotheses(
        target_smiles=target_smiles,
        target_name=target_name,
        family_hint=family_hint,
        source_evidence_refs=source_refs,
        case_id=case_id,
    )
    current_handles = _target_side_handles(board.get("target_side_disconnection_hypotheses") or {})
    refreshed_handles = _target_side_handles(target_side)
    current_objectives = _selected_objective_types(board.get("route_objective_summary") or {})
    refreshed_objectives = _selected_objective_types(objective_summary)
    should_refresh_target_side = bool(target_side.get("accepted")) and (
        not current_handles or current_handles != refreshed_handles
    )
    should_refresh_objectives = bool(objective_summary.get("accepted")) and (
        not current_objectives or current_objectives != refreshed_objectives or should_refresh_target_side
    )
    if not (should_refresh_target_side or should_refresh_objectives):
        return board

    report = {
        "schema_version": "target_derived_prior_refresh.v1",
        "target_name": target_name,
        "target_smiles": target_smiles,
        "old_handles": sorted(current_handles),
        "new_handles": sorted(refreshed_handles),
        "old_objective_types": sorted(current_objectives),
        "new_objective_types": sorted(refreshed_objectives),
        "refreshed_target_side": bool(should_refresh_target_side),
        "refreshed_route_objectives": bool(should_refresh_objectives),
    }
    if should_refresh_objectives:
        board["route_objective_summary"] = _drop_large_fields(objective_summary)
        board["endpoint_candidates"] = [
            dict(row)
            for row in objective_summary.get("endpoint_candidates") or []
            if isinstance(row, dict)
        ]
        _apply_route_scope_to_belief(board, dict(objective_summary.get("route_scope") or {}))
    if should_refresh_target_side:
        removed_hypothesis_ids = {
            str(row.get("hypothesis_id") or "")
            for row in board.get("analogical_hypotheses") or []
            if _target_side_hypothesis_row(row)
        }
        new_hypothesis_ids = {
            str(row.get("hypothesis_id") or "")
            for row in target_side.get("hypotheses") or []
            if isinstance(row, dict) and str(row.get("hypothesis_id") or "").strip()
        }
        board["target_side_disconnection_hypotheses"] = _drop_large_fields(target_side)
        board["analogical_hypotheses"] = [
            row
            for row in board.get("analogical_hypotheses") or []
            if not _target_side_hypothesis_row(row)
        ]
        _extend_unique(board, "analogical_hypotheses", target_side.get("hypotheses") or [], unique_key="hypothesis_id")
        stale_ranked_ids = _analogical_ranking_hypothesis_ids(board.get("analogical_hypothesis_ranking") or {})
        if stale_ranked_ids & (removed_hypothesis_ids - new_hypothesis_ids):
            board["analogical_hypothesis_ranking"] = {}
        stale_ids = removed_hypothesis_ids - new_hypothesis_ids
        if stale_ids:
            _remove_stale_evidence_refs(board, stale_ids)
        board["bridge_tasks"] = [
            row
            for row in board.get("bridge_tasks") or []
            if not _target_derived_bridge_task(row)
        ]
        _extend_unique(board, "bridge_tasks", target_side.get("bridge_tasks") or [], unique_key="task_id")
        board["semisynthesis_anchors"] = [
            row
            for row in board.get("semisynthesis_anchors") or []
            if not _route_objective_anchor_row(row)
        ]
        _extend_unique(board, "semisynthesis_anchors", target_side.get("semisynthesis_anchors") or [], unique_key="anchor_id")
        _apply_route_scope_to_belief(board, dict(target_side.get("route_scope") or {}))
        template_report = build_broad_transform_templates_from_blackboard(board)
        if template_report.get("accepted"):
            board["broad_transform_templates"] = [
                dict(row)
                for row in template_report.get("templates") or []
                if isinstance(row, dict)
            ]
            belief = dict(board.get("current_belief") or {})
            template_policy = dict(belief.get("template_policy") or {})
            template_policy["broad_transform_template_count"] = len(board.get("broad_transform_templates") or [])
            template_policy["broad_templates_are_advisory_only"] = True
            template_policy["not_parent_route_proof"] = True
            belief["template_policy"] = template_policy
            board["current_belief"] = belief

    migrations = [
        dict(row)
        for row in board.get("blackboard_migrations") or []
        if isinstance(row, dict)
    ]
    migrations.append(report)
    board["blackboard_migrations"] = migrations[-20:]
    return board


def _seed_target_literature_sources(board: dict[str, Any], *, target_input: dict[str, Any]) -> None:
    rows = _target_input_literature_seed_rows(target_input)
    if not rows:
        return
    evidence = dict(board.get("literature_evidence") or {})
    candidates = list(evidence.get("source_candidates") or [])
    source_refs = [str(item) for item in evidence.get("source_refs") or [] if str(item or "").strip()]
    seen = {
        _literature_seed_key(row)
        for row in candidates
        if isinstance(row, dict) and _literature_seed_key(row)
    }
    for idx, raw in enumerate(rows, start=1):
        candidate = _literature_seed_candidate(raw, idx=idx)
        key = _literature_seed_key(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if str(candidate.get("source_ref") or "").strip() not in source_refs:
            source_refs.append(str(candidate.get("source_ref") or "").strip())
    evidence["source_candidates"] = candidates
    evidence["source_refs"] = source_refs
    evidence["source_discovery_mode"] = "target_input_local_pdf_seed"
    evidence["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
    evidence["confidence"] = "source_seeded"
    board["literature_evidence"] = evidence
    _refresh_source_lifecycle(board)


def _target_input_literature_seed_rows(target_input: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("literature_sources", "local_literature_cache"):
        for raw in target_input.get(key) or []:
            if isinstance(raw, dict):
                row = dict(raw)
                if _literature_seed_is_auto_local_pdf_cache(row):
                    continue
                rows.append(row)
    pdf_path = str(target_input.get("literature_pdf_path") or "").strip()
    if pdf_path:
        rows.append(
            {
                "candidate_id": "target_input_literature_pdf",
                "source_ref": str(target_input.get("literature_pdf_source_ref") or "").strip(),
                "local_pdf": pdf_path,
                "source_role": "user_provided_local_pdf_seed",
            }
        )
    return rows


def _literature_seed_is_auto_local_pdf_cache(row: dict[str, Any]) -> bool:
    role = str(row.get("source_role") or "").strip().lower()
    index = dict(row.get("local_pdf_index") or {})
    return bool(role == "auto_local_pdf_cache" or str(index.get("schema_version") or "") == "auto_local_pdf_index.v1")


def _literature_seed_candidate(row: dict[str, Any], *, idx: int) -> dict[str, Any]:
    local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or row.get("path") or "").strip()
    doi = _normalize_hint_doi(str(row.get("doi") or ""))
    source_ref = str(row.get("source_ref") or "").strip()
    if not source_ref:
        source_ref = f"doi:{doi}" if doi else (f"local_pdf:{Path(local_pdf).name}" if local_pdf else f"target_input_source:{idx}")
    role = str(row.get("source_role") or ("user_provided_local_pdf_seed" if local_pdf else "target_input_literature_seed"))
    tasks = ["extract_pdf_literature_structures", "extract_visual_literature_chain", "compile_exact_literature_rows"] if local_pdf else [
        "search_literature"
    ]
    document_id = str(row.get("document_id") or "").strip()
    if local_pdf and not document_id:
        document_id = f"pdf:{hashlib.sha256(str(Path(local_pdf)).lower().encode('utf-8')).hexdigest()[:16]}"
    content_scope = str(row.get("content_scope") or row.get("document_type") or "").strip()
    if local_pdf and not content_scope:
        content_scope = _infer_literature_content_scope(local_pdf)
    return {
        "schema_version": "literature_source_candidate.v1",
        "candidate_id": str(row.get("candidate_id") or f"target_input_source_{idx}"),
        "source_ref": source_ref,
        "doi": doi,
        "pii": str(row.get("pii") or ""),
        "url": str(row.get("url") or ""),
        "title": str(row.get("title") or row.get("source_title") or ""),
        "local_pdf": local_pdf,
        "document_id": document_id,
        "content_scope": content_scope,
        "source_type": str(row.get("source_type") or ("user_provided_local_pdf_seed" if local_pdf else "target_input_literature_seed")),
        "source_role": role,
        "source_discovery_mode": "target_input_local_pdf_seed",
        "access_status": "local_pdf_available" if local_pdf else "metadata_only",
        "relevance_rationale": str(row.get("relevance_rationale") or "target input supplied literature source"),
        "expected_scheme_or_compound_labels": [
            str(item)
            for item in row.get("expected_scheme_or_compound_labels") or row.get("expected_labels") or []
            if str(item or "").strip()
        ],
        "route_sequence_hint": str(row.get("route_sequence_hint") or ""),
        "visual_extraction_profile": (
            dict(row.get("visual_extraction_profile") or {})
            if isinstance(row.get("visual_extraction_profile"), dict)
            else {}
        ),
        "extraction_task_recommendations": tasks,
        "user_provided_source_seed": bool(local_pdf) or bool(row.get("user_provided_source_seed")),
        "no_solved_claim": True,
    }


def _literature_seed_key(row: dict[str, Any]) -> str:
    local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or "").strip().lower()
    if local_pdf:
        return f"pdf:{local_pdf}"
    document_id = str(row.get("document_id") or "").strip().lower()
    if document_id:
        return f"document:{document_id}"
    doi = _normalize_hint_doi(str(row.get("doi") or ""))
    if doi:
        return f"doi:{doi}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref:
        return f"ref:{source_ref}"
    url = str(row.get("url") or "").strip().lower()
    if url:
        return f"url:{url}"
    title = str(row.get("title") or row.get("source_title") or "").strip().lower()
    return f"title:{title}" if title else ""


def _infer_literature_content_scope(path: str) -> str:
    name = Path(path).name.lower()
    if any(token in name for token in ("supporting", "supplement", "supp_info", "_si.", "-si.")):
        return "supplementary_information"
    if any(token in name for token in ("thesis", "dissertation")):
        return "thesis"
    return "article"


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, parsed)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def update_blackboard_from_action_batch(
    blackboard: dict[str, Any],
    *,
    action_batch: dict[str, Any],
    validation: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    """Record planner-level decisions and fallbacks in the blackboard."""
    board = deepcopy(blackboard)
    planner = dict(action_batch.get("codex_action_planner") or {})
    action_types = [
        str(row.get("action_type") or "")
        for row in action_batch.get("actions") or []
        if isinstance(row, dict)
    ]
    record = {
        "schema_version": "agent_planner_history_record.v1",
        "round_index": int(round_index),
        "mode": str(action_batch.get("mode") or ""),
        "validation_accepted": bool(validation.get("accepted")),
        "validation_reasons": [str(item) for item in validation.get("reasons") or []],
        "action_count": len(action_types),
        "action_types": action_types,
        "planner_source_hint_count": len(action_batch.get("planner_source_hints") or []),
        "codex_action_planner": {
            "attempted": _codex_planner_attempted(planner),
            "fallback_used": bool(planner.get("fallback_used")),
            "fallback_reason": str(planner.get("fallback_reason") or ""),
            "backend": str(planner.get("backend") or planner.get("record_backend") or ""),
            "status": str(planner.get("status") or planner.get("record_status") or ""),
            "record_ref": str(planner.get("record_ref") or ""),
            "blackboard_snapshot_ref": str(planner.get("blackboard_snapshot_ref") or ""),
            "tool_policy": dict(planner.get("tool_policy") or {}),
        },
        "raw_reaction_output_allowed": bool((action_batch.get("semantics") or {}).get("raw_reaction_output_allowed")),
        "planner_can_emit_solved": bool((action_batch.get("semantics") or {}).get("planner_can_emit_solved")),
    }
    board.setdefault("planner_history", []).append(record)
    if validation.get("accepted"):
        evidence = dict(board.get("literature_evidence") or {})
        _extend_unique(
            evidence,
            "planner_source_hints",
            _normalize_planner_source_hints(action_batch.get("planner_source_hints") or [], round_index=round_index),
            unique_key="hint_key",
        )
        board["literature_evidence"] = evidence
        _refresh_source_lifecycle(board)

    record_ref = str(record["codex_action_planner"].get("record_ref") or "")
    if record_ref:
        board.setdefault("artifact_refs", {})[f"codex_action_planner_round_{int(round_index)}"] = record_ref
    snapshot_ref = str(record["codex_action_planner"].get("blackboard_snapshot_ref") or "")
    if snapshot_ref:
        board.setdefault("artifact_refs", {})[f"codex_action_planner_blackboard_snapshot_round_{int(round_index)}"] = snapshot_ref

    if record["codex_action_planner"]["attempted"]:
        budget = dict(board.get("budget_state") or {})
        budget["codex_action_planner_runs"] = int(budget.get("codex_action_planner_runs") or 0) + 1
        board["budget_state"] = budget

    fallback_reason = str(record["codex_action_planner"].get("fallback_reason") or "")
    if fallback_reason and fallback_reason != "codex_action_planner_disabled":
        belief = dict(board.get("current_belief") or {})
        planner_notes = list(belief.get("planner_notes") or [])
        planner_notes.append(
            {
                "schema_version": "agent_planner_note.v1",
                "round_index": int(round_index),
                "reason": fallback_reason,
                "next_round_bias": "inspect blackboard state and change action selection if fallback repeats",
            }
        )
        belief["planner_notes"] = planner_notes[-10:]
        board["current_belief"] = belief
    return board


def _codex_planner_attempted(planner: dict[str, Any]) -> bool:
    if not planner:
        return False
    if str(planner.get("fallback_reason") or "") == "codex_action_planner_disabled":
        return False
    return bool(
        planner.get("backend")
        or planner.get("record_backend")
        or planner.get("status")
        or planner.get("record_status")
        or planner.get("record_ref")
        or planner.get("blackboard_snapshot_ref")
    )


def _normalize_planner_source_hints(rows: list[Any], *, round_index: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        doi = _normalize_hint_doi(str(row.get("doi") or ""))
        pii = str(row.get("pii") or "").strip()
        url = str(row.get("url") or "").strip()
        local_pdf = str(row.get("local_pdf") or row.get("pdf_path") or "").strip()
        local_ref = str(row.get("local_ref") or "").strip()
        title = str(row.get("title") or row.get("source_title") or "").strip()
        source_ref = str(row.get("source_ref") or "").strip()
        if not source_ref:
            source_ref = f"doi:{doi}" if doi else (f"pii:{pii}" if pii else (url or local_ref or (f"local_pdf:{Path(local_pdf).name}" if local_pdf else "")))
        hint_key = str(doi or pii or url or local_pdf or local_ref or source_ref or title).strip().lower()
        if not hint_key or hint_key in seen:
            continue
        seen.add(hint_key)
        out.append(
            {
                "schema_version": PLANNER_SOURCE_HINT_SCHEMA,
                "hint_id": str(row.get("hint_id") or f"r{int(round_index)}:planner_source_hint_{idx}"),
                "hint_key": hint_key,
                "source_ref": source_ref,
                "title": title,
                "doi": doi,
                "pii": pii,
                "url": url,
                "local_pdf": local_pdf,
                "local_ref": local_ref,
                "source_type": str(row.get("source_type") or "planner_discovered_literature_metadata"),
                "relevance_rationale": str(row.get("relevance_rationale") or row.get("rationale") or ""),
                "expected_scheme_or_compound_labels": [
                    str(item)
                    for item in row.get("expected_scheme_or_compound_labels") or row.get("expected_labels") or []
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
        if len(out) >= 8:
            break
    return out


def _normalize_hint_doi(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    for separator in ("?", "#", "&"):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text.strip().strip(".,;:)]}'\"").lower()


def _refresh_source_lifecycle(board: dict[str, Any]) -> None:
    evidence = dict(board.get("literature_evidence") or {})
    lifecycle = _build_source_lifecycle(evidence)
    evidence["source_lifecycle"] = lifecycle
    real_lifecycle = [
        row
        for row in lifecycle
        if (
            (row.get("stage_flags") or {}).get("source_candidate") is True
            and (row.get("stage_flags") or {}).get("placeholder_only") is not True
        )
        or (row.get("stage_flags") or {}).get("exact_rows_compiled") is True
    ]
    groups = {
        str(row.get("independent_source_group") or "")
        for row in real_lifecycle
        if str(row.get("independent_source_group") or "")
    }
    representations = {
        str(item)
        for row in real_lifecycle
        for item in row.get("representations") or []
        if str(item or "")
    }
    evidence["source_identity_summary"] = {
        "schema_version": "literature_source_identity_summary.v1",
        "document_count": len(real_lifecycle),
        "independent_source_group_count": len(groups),
        "representation_count": len(representations),
        "independent_source_groups": sorted(groups),
        "semantics": {
            "document_is_not_independent_source": True,
            "article_and_si_share_independence_group": True,
            "url_and_local_pdf_can_represent_one_document": True,
        },
    }
    board["literature_evidence"] = evidence


def _build_source_lifecycle(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    def get_row(source: dict[str, Any]) -> dict[str, Any]:
        key = _lifecycle_source_key(source)
        if not key:
            key = f"unknown:{len(rows) + 1}"
        if key not in rows:
            rows[key] = {
                "schema_version": "agent_source_lifecycle.v1",
                "source_key": key,
                "document_identity": key,
                "independent_source_group": independent_source_group(source),
                "representations": [],
                "source_ref": "",
                "title": "",
                "doi": "",
                "pii": "",
                "url": "",
                "local_pdf": "",
                "stage": "unresolved",
                "next_recommended_stage": "search_literature",
                "stage_flags": {
                    "planner_hint": False,
                    "source_candidate": False,
                    "local_pdf_proxy_requested": False,
                    "local_pdf_available": False,
                    "pdf_rendered": False,
                    "visual_extracted": False,
                    "exact_rows_compiled": False,
                    "placeholder_only": False,
                },
                "refs": {
                    "planner_hint_ids": [],
                    "candidate_ids": [],
                    "local_pdf_proxy_request_ids": [],
                    "pdf_evidence_ids": [],
                    "visual_chain_ids": [],
                    "exact_row_ids": [],
                },
                "counts": {
                    "planner_hints": 0,
                    "source_candidates": 0,
                    "local_pdf_proxy_requests": 0,
                    "pdf_structure_evidence": 0,
                    "visual_chains": 0,
                    "exact_rows": 0,
                },
                "provenance": [],
                "no_solved_claim": True,
            }
        _merge_lifecycle_identity(rows[key], source)
        return rows[key]

    for hint in evidence.get("planner_source_hints") or []:
        if not isinstance(hint, dict):
            continue
        row = get_row(hint)
        row["stage_flags"]["planner_hint"] = True
        _append_unique(row["refs"], "planner_hint_ids", str(hint.get("hint_id") or ""))
        row["counts"]["planner_hints"] += 1
        _append_lifecycle_provenance(row, "planner_source_hint", hint)

    for candidate in evidence.get("source_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        row = get_row(candidate)
        row["stage_flags"]["source_candidate"] = True
        if _candidate_has_real_source(candidate):
            row["stage_flags"]["placeholder_only"] = False
        elif bool(candidate.get("placeholder_only")) or str(candidate.get("access_status") or "").lower() == "placeholder_only":
            row["stage_flags"]["placeholder_only"] = True
        if str(candidate.get("local_pdf") or "").strip():
            row["stage_flags"]["local_pdf_available"] = True
        _append_unique(row["refs"], "candidate_ids", str(candidate.get("candidate_id") or candidate.get("source_ref") or ""))
        row["counts"]["source_candidates"] += 1
        _append_lifecycle_provenance(row, "source_candidate", candidate)

    for request in evidence.get("local_pdf_proxy_requests") or []:
        if not isinstance(request, dict):
            continue
        row = get_row(request)
        row["stage_flags"]["local_pdf_proxy_requested"] = True
        _append_unique(row["refs"], "local_pdf_proxy_request_ids", str(request.get("request_id") or request.get("source_ref") or ""))
        row["counts"]["local_pdf_proxy_requests"] += 1
        _append_lifecycle_provenance(row, "local_pdf_proxy_request", request)

    for pdf in evidence.get("pdf_structure_evidence") or []:
        if not isinstance(pdf, dict):
            continue
        row = get_row(pdf)
        row["stage_flags"]["pdf_rendered"] = (
            pdf_evidence_has_materialized_render(pdf)
            or row["stage_flags"]["pdf_rendered"]
        )
        if str(pdf.get("source_pdf_path") or pdf.get("pdf_path") or "").strip():
            row["stage_flags"]["local_pdf_available"] = True
        _append_unique(row["refs"], "pdf_evidence_ids", str(pdf.get("evidence_id") or pdf.get("source_ref") or ""))
        row["counts"]["pdf_structure_evidence"] += 1
        _append_lifecycle_provenance(row, "pdf_structure_evidence", pdf)

    for chain in evidence.get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        row = get_row(chain)
        row["stage_flags"]["visual_extracted"] = (
            bool(chain.get("accepted"))
            or int(chain.get("candidate_step_count") or chain.get("step_count") or 0) > 0
            or row["stage_flags"]["visual_extracted"]
        )
        if str(chain.get("source_pdf_path") or chain.get("pdf_path") or "").strip():
            row["stage_flags"]["local_pdf_available"] = True
        _append_unique(row["refs"], "visual_chain_ids", str(chain.get("chain_id") or chain.get("artifact_ref") or ""))
        row["counts"]["visual_chains"] += 1
        _append_lifecycle_provenance(row, "visual_chain", chain)

    for exact in evidence.get("exact_rows") or []:
        if not isinstance(exact, dict):
            continue
        row = get_row(exact)
        row["stage_flags"]["exact_rows_compiled"] = True
        _append_unique(row["refs"], "exact_row_ids", str(exact.get("row_id") or ""))
        row["counts"]["exact_rows"] += 1
        _append_lifecycle_provenance(row, "exact_row", exact)

    out = []
    for row in rows.values():
        _finalize_lifecycle_stage(row)
        out.append(row)
    return sorted(out, key=lambda item: str(item.get("source_key") or ""))


def _merge_lifecycle_identity(row: dict[str, Any], source: dict[str, Any]) -> None:
    group = independent_source_group(source)
    if group and not str(row.get("independent_source_group") or ""):
        row["independent_source_group"] = group
    row["document_identity"] = str(
        row.get("document_identity") or source_document_identity(source)
    )
    row["representations"] = sorted(
        {
            *[str(item) for item in row.get("representations") or []],
            *source_record_representations(source),
        }
    )
    identity_fields = {
        "document_id": str(source.get("document_id") or ""),
        "content_scope": source_content_scope(source),
        "source_ref": str(source.get("source_ref") or ""),
        "title": str(source.get("title") or source.get("source_title") or ""),
        "doi": _normalize_hint_doi(str(source.get("doi") or "")),
        "pii": str(source.get("pii") or ""),
        "url": str(source.get("url") or ""),
        "local_pdf": str(source.get("local_pdf") or source.get("source_pdf_path") or source.get("pdf_path") or ""),
    }
    for key, value in identity_fields.items():
        if str(value or "").strip() and not str(row.get(key) or "").strip():
            row[key] = value


def _append_lifecycle_provenance(row: dict[str, Any], event_type: str, source: dict[str, Any]) -> None:
    entry = {
        "schema_version": "agent_source_lifecycle_event.v1",
        "event_type": event_type,
        "source_ref": str(source.get("source_ref") or ""),
        "artifact_ref": str(source.get("artifact_ref") or ""),
        "no_solved_claim": True,
    }
    marker = "|".join([entry["event_type"], entry["source_ref"], entry["artifact_ref"]])
    existing = {
        "|".join([str(item.get("event_type") or ""), str(item.get("source_ref") or ""), str(item.get("artifact_ref") or "")])
        for item in row.get("provenance") or []
        if isinstance(item, dict)
    }
    if marker not in existing:
        row.setdefault("provenance", []).append(entry)


def _append_unique(target: dict[str, Any], key: str, value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    rows = list(target.get(key) or [])
    if text not in rows:
        rows.append(text)
    target[key] = rows


def _finalize_lifecycle_stage(row: dict[str, Any]) -> None:
    flags = dict(row.get("stage_flags") or {})
    if flags.get("exact_rows_compiled"):
        row["stage"] = "exact_rows_compiled"
        row["next_recommended_stage"] = "stitch_parent_route_or_guided_chemenzy"
    elif flags.get("visual_extracted"):
        row["stage"] = "visual_extracted"
        row["next_recommended_stage"] = "compile_exact_literature_rows"
    elif flags.get("pdf_rendered"):
        row["stage"] = "pdf_rendered"
        row["next_recommended_stage"] = "extract_visual_literature_chain"
    elif flags.get("local_pdf_available"):
        row["stage"] = "local_pdf_available"
        row["next_recommended_stage"] = "extract_pdf_literature_structures"
    elif flags.get("local_pdf_proxy_requested"):
        row["stage"] = "local_pdf_proxy_requested"
        row["next_recommended_stage"] = "await_local_pdf_proxy_download"
    elif flags.get("placeholder_only"):
        row["stage"] = "placeholder_only"
        row["next_recommended_stage"] = "retry_source_acquisition"
    elif flags.get("source_candidate"):
        row["stage"] = "source_candidate"
        row["next_recommended_stage"] = "resolve_source_material_or_local_pdf"
    elif flags.get("planner_hint"):
        row["stage"] = "planner_hint"
        row["next_recommended_stage"] = "search_literature"
    else:
        row["stage"] = "unresolved"
        row["next_recommended_stage"] = "search_literature"
    row["provenance"] = list(row.get("provenance") or [])[-8:]


def _lifecycle_source_key(source: dict[str, Any]) -> str:
    document_identity = source_document_identity(source)
    if document_identity:
        return document_identity
    local_pdf = str(
        source.get("local_pdf")
        or source.get("source_pdf_path")
        or source.get("pdf_path")
        or ""
    ).strip().lower()
    if local_pdf:
        return f"pdf:{local_pdf}"
    document_id = str(source.get("document_id") or "").strip().lower()
    if document_id:
        return f"document:{document_id}"
    doi = _normalize_hint_doi(str(source.get("doi") or ""))
    if doi:
        return f"doi:{doi}"
    source_ref = str(source.get("source_ref") or "").strip()
    if source_ref.lower().startswith("doi:"):
        doi = _normalize_hint_doi(source_ref)
        if doi:
            return f"doi:{doi}"
    pii = str(source.get("pii") or "").strip().upper()
    if pii:
        return f"pii:{pii}"
    if source_ref.lower().startswith("pii:"):
        return f"pii:{source_ref.split(':', 1)[1].strip().upper()}"
    if source_ref:
        return f"ref:{source_ref.lower()}"
    url = str(source.get("url") or "").strip().lower()
    if url:
        return f"url:{url}"
    title = str(source.get("title") or source.get("source_title") or "").strip().lower()
    return f"title:{title}" if title else ""


def update_blackboard_from_action(
    blackboard: dict[str, Any],
    *,
    action: dict[str, Any],
    action_result: dict[str, Any],
    round_index: int,
    run_dir: str | Path,
) -> dict[str, Any]:
    board = deepcopy(blackboard)
    action_type = str(action.get("action_type") or "")
    enriched_result = _enrich_action_result_with_source_context(action, action_result)
    before_counts = _blackboard_count_summary(board)
    artifact_name = _artifact_name(action, enriched_result)
    artifact_ref = ""
    if artifact_name:
        artifact_path = Path(run_dir) / f"{artifact_name}.json"
        write_json(artifact_path, _summary_safe(enriched_result))
        artifact_ref = str(artifact_path)
        board.setdefault("artifact_refs", {})[artifact_name] = artifact_ref

    useful = _normalize_action_output(board, action_type=action_type, result=enriched_result, artifact_ref=artifact_ref)
    proposal_refresh = _refresh_retrosynthetic_proposal_bus(board)
    useful = bool(useful or proposal_refresh.get("useful_artifact"))
    _refresh_source_lifecycle(board)
    after_counts = _blackboard_count_summary(board)
    delta = _blackboard_count_delta(before_counts, after_counts)
    history = {
        "schema_version": "agent_action_history_record.v1",
        "round_index": int(round_index),
        "action_id": str(action.get("action_id") or ""),
        "action_type": action_type,
        "status": "accepted" if enriched_result.get("accepted", True) else "rejected",
        "artifact_ref": artifact_ref,
        "useful_artifact": bool(useful),
        "stale": not bool(useful),
        "action_signature": _action_signature(action),
        "reasons": [str(item) for item in enriched_result.get("reasons") or []],
        "blackboard_counts_before": before_counts,
        "blackboard_counts_after": after_counts,
        "blackboard_delta": delta,
        "changed_blackboard_fields": [str(key) for key, value in delta.items() if value],
        "resource_cost": dict(action.get("_host_resource_cost") or {}),
    }
    if action_type == "compile_exact_literature_rows":
        history.update(_compile_replay_history_fields(enriched_result))
    board.setdefault("action_history", []).append(history)
    return board


def _compile_replay_history_fields(
    action_result: dict[str, Any],
) -> dict[str, Any]:
    """Persist whether the current host parser actually completed its replay.

    Generic usefulness is intentionally insufficient here: a rejected compile
    can still add diagnostic terminals or an audit artifact.  Only a complete,
    authority-labelled per-step parser record may suppress a later retry.
    """

    payload = (
        dict(action_result.get("result") or {})
        if isinstance(action_result.get("result"), dict)
        else dict(action_result)
    )
    audit = dict(payload.get("deterministic_literature_registry") or {})
    authority = dict(audit.get("authority") or {})
    records = audit.get("records")
    try:
        input_step_count = int(audit.get("input_step_count") or 0)
    except (TypeError, ValueError):
        input_step_count = 0
    replay_completed = bool(
        audit.get("schema_version")
        == "deterministic_literature_registry_audit.v1"
        and authority.get("type") == "deterministic_structure_parser"
        and str(authority.get("id") or "").strip()
        and input_step_count > 0
        and isinstance(records, list)
        and len(records) == input_step_count
        and str(audit.get("registry_sha256") or "").strip()
    )
    return {
        "compile_replay_completed": replay_completed,
        "compile_parser_authority_id": str(authority.get("id") or ""),
        "compile_input_step_count": input_step_count,
        "compile_approved_binding_count": _nonnegative_int(
            audit.get("approved_binding_count"), 0
        ),
        "compile_rejected_step_count": _nonnegative_int(
            audit.get("rejected_step_count"), 0
        ),
        "compile_replay_reasons": [
            str(item) for item in audit.get("reasons") or []
        ],
    }


def update_budget_for_action(
    blackboard: dict[str, Any],
    action_type: str,
    payload: dict[str, Any] | None = None,
    *,
    resource_cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    board = deepcopy(blackboard)
    budget = dict(board.get("budget_state") or {})
    action_payload = dict(payload or {})
    if resource_cost is not None:
        cost = dict(resource_cost)
        for cost_key, budget_key in (
            ("scout_calls", "scout_calls"),
            ("visual_calls", "visual_calls"),
            ("chemenzy_runs", "chemenzy_runs"),
            ("child_target_runs", "child_target_runs"),
            ("template_application_actions", "template_application_actions"),
        ):
            try:
                increment = max(0, int(cost.get(cost_key) or 0))
            except (TypeError, ValueError):
                increment = 0
            budget[budget_key] = int(budget.get(budget_key) or 0) + increment
        board["budget_state"] = budget
        return board
    if action_type == "search_literature":
        budget["scout_calls"] = int(budget.get("scout_calls") or 0) + 1
    if action_type == "extract_visual_literature_chain":
        budget["visual_calls"] = int(budget.get("visual_calls") or 0) + 1
    if action_type == "resolve_literature_structure_task" and bool(action_payload.get("run_visual", True)):
        budget["visual_calls"] = int(budget.get("visual_calls") or 0) + 1
    if action_type == "run_guided_chemenzy":
        budget["chemenzy_runs"] = int(budget.get("chemenzy_runs") or 0) + 1
    if action_type == "expand_child_target":
        budget["child_target_runs"] = int(budget.get("child_target_runs") or 0) + planned_child_target_count(action_payload)
    if action_type in {"apply_analogical_template_to_target", "validate_template_application"}:
        budget["template_application_actions"] = int(budget.get("template_application_actions") or 0) + 1
    board["budget_state"] = budget
    return board


def _blackboard_count_summary(blackboard: dict[str, Any]) -> dict[str, int]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    belief = dict(blackboard.get("current_belief") or {})
    proof = dict(blackboard.get("parent_route_proof") or {})
    recursive_status_counts = _recursive_hypothesis_task_status_counts(blackboard)
    return {
        "source_candidates": len(evidence.get("source_candidates") or []),
        "planner_source_hints": len(evidence.get("planner_source_hints") or []),
        "source_lifecycle": len(evidence.get("source_lifecycle") or []),
        "source_refs": len(evidence.get("source_refs") or []),
        "local_pdf_proxy_requests": len(evidence.get("local_pdf_proxy_requests") or []),
        "pdf_structure_evidence": len(evidence.get("pdf_structure_evidence") or []),
        "visual_chains": len(evidence.get("visual_chains") or []),
        "process_evidence_rows": len(evidence.get("process_evidence_rows") or []),
        "exact_rows": len(evidence.get("exact_rows") or []),
        "target_relevant_exact_rows": _target_relevant_exact_row_count(evidence.get("exact_rows") or []),
        "exact_chain_audits": len(evidence.get("exact_chain_audits") or []),
        "terminal_candidates": len(evidence.get("terminal_candidates") or []),
        "chemenzy_attempts": len(blackboard.get("chemenzy_attempts") or []),
        "structure_resolution_tasks": len(evidence.get("structure_resolution_tasks") or []),
        "structure_resolution_attempts": len(evidence.get("structure_resolution_attempts") or []),
        "resolved_structures": len(evidence.get("resolved_structures") or []),
        "route_failures": len(blackboard.get("route_failures") or []),
        "plugin_runtime_diagnostics": len(blackboard.get("plugin_runtime_diagnostics") or []),
        "analogical_hypotheses": len(blackboard.get("analogical_hypotheses") or []),
        "analogical_templates": len(blackboard.get("analogical_templates") or []),
        "template_applications": len(blackboard.get("template_applications") or []),
        "template_failure_memory": len(blackboard.get("template_failure_memory") or []),
        "route_objectives": len((blackboard.get("route_objective_summary") or {}).get("objectives") or []),
        "endpoint_candidates": len(blackboard.get("endpoint_candidates") or []),
        "broad_transform_templates": len(blackboard.get("broad_transform_templates") or []),
        "reaction_idea_cards": len(blackboard.get("reaction_idea_cards") or []),
        "retrosynthetic_proposals": len(blackboard.get("retrosynthetic_proposals") or []),
        "proposal_failure_feedback": len(blackboard.get("proposal_failure_feedback") or []),
        "semisynthesis_anchors": len(blackboard.get("semisynthesis_anchors") or []),
        "recursive_hypothesis_tasks": len(blackboard.get("recursive_hypothesis_tasks") or []),
        "recursive_hypothesis_tasks_pending": recursive_status_counts.get("pending", 0),
        "recursive_hypothesis_tasks_rejected": recursive_status_counts.get("rejected", 0),
        "recursive_hypothesis_tasks_accepted": recursive_status_counts.get("accepted_child_route", 0),
        "bridge_tasks": len(blackboard.get("bridge_tasks") or []),
        "terminal_blacklist": len(blackboard.get("terminal_blacklist") or []),
        "blocked_directions": len(belief.get("blocked_directions") or []),
        "next_action_bias": len(belief.get("next_action_bias") or []),
        "stop_candidates": len(belief.get("stop_candidates") or []),
        "artifact_refs": len(blackboard.get("artifact_refs") or {}),
        "parent_route_proof_present": 1 if proof else 0,
        "parent_route_proof_accepted": 1 if proof.get("accepted") else 0,
    }


def _recursive_hypothesis_task_status_counts(blackboard: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in blackboard.get("recursive_hypothesis_tasks") or []:
        if not isinstance(task, dict):
            continue
        status = str(task.get("status") or "pending").strip() or "pending"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _visual_reason_is_runtime_failure(reason: str) -> bool:
    return str(reason or "") in {
        "visual_direct_api_failed",
        "visual_model_unavailable",
        "visual_api_auth_failed",
    }


def _blackboard_count_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {
        key: int(after.get(key) or 0) - int(before.get(key) or 0)
        for key in keys
        if int(after.get(key) or 0) - int(before.get(key) or 0)
    }


def _refresh_retrosynthetic_proposal_bus(board: dict[str, Any]) -> dict[str, Any]:
    """Refresh derived reaction-idea/proposal rows after evidence changes."""
    before = {
        "reaction_idea_cards": len(board.get("reaction_idea_cards") or []),
        "retrosynthetic_proposals": len(board.get("retrosynthetic_proposals") or []),
        "recursive_hypothesis_tasks": len(board.get("recursive_hypothesis_tasks") or []),
    }
    report = compile_retrosynthetic_proposal_bus(board)
    _extend_unique(board, "reaction_idea_cards", report.get("reaction_idea_cards") or [], unique_key="card_id")
    board["retrosynthetic_proposals"] = deduplicate_retrosynthetic_proposals(
        [
            *[row for row in board.get("retrosynthetic_proposals") or [] if isinstance(row, dict)],
            *[row for row in report.get("retrosynthetic_proposals") or [] if isinstance(row, dict)],
        ]
    )
    _extend_unique(board, "recursive_hypothesis_tasks", report.get("recursive_hypothesis_tasks") or [], unique_key="task_id")
    refreshed_counts = {
        "reaction_idea_cards": len(board.get("reaction_idea_cards") or []),
        "retrosynthetic_proposals": len(board.get("retrosynthetic_proposals") or []),
        "recursive_hypothesis_tasks": len(board.get("recursive_hypothesis_tasks") or []),
    }
    current_proposals = [row for row in board.get("retrosynthetic_proposals") or [] if isinstance(row, dict)]
    report_counts = {
        **refreshed_counts,
        "executable_or_semi_executable_proposals": sum(
            1
            for row in current_proposals
            if str(row.get("proposal_type") or "") in {"exact_executable", "semi_executable"}
        ),
        "strategic_proposals": sum(
            1 for row in current_proposals if str(row.get("proposal_type") or "") == "strategic"
        ),
        "exact_proposals": sum(1 for row in current_proposals if str(row.get("proposal_granularity") or "") == "exact"),
        "same_core_proposals": sum(1 for row in current_proposals if str(row.get("proposal_granularity") or "") == "same_core"),
        "mechanism_proposals": sum(1 for row in current_proposals if str(row.get("proposal_granularity") or "") == "mechanism"),
        "process_proposals": sum(1 for row in current_proposals if str(row.get("proposal_granularity") or "") == "process"),
        "fallback_proposals": sum(1 for row in current_proposals if str(row.get("proposal_granularity") or "") == "fallback"),
    }
    changed = {
        key: int(refreshed_counts.get(key) or 0) - int(before.get(key) or 0)
        for key in refreshed_counts
        if int(refreshed_counts.get(key) or 0) - int(before.get(key) or 0)
    }
    board["retrosynthetic_proposal_compile_report"] = {
        "schema_version": str(report.get("schema_version") or "retrosynthetic_proposal_compile_report.v1"),
        "accepted": bool(report.get("accepted")),
        "counts": report_counts,
        "blackboard_counts_after_refresh": refreshed_counts,
        "new_counts_this_refresh": changed,
        "allowed_use": "proposal_bus_and_recursive_search_seed_only",
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }
    if changed:
        belief = dict(board.get("current_belief") or {})
        _extend_unique(
            belief,
            "next_action_bias",
            ["expand_child_target" if changed.get("recursive_hypothesis_tasks") else "run_guided_chemenzy"],
            unique_key=None,
        )
        board["current_belief"] = belief
    return {"useful_artifact": bool(changed), "changed": changed}


def complete_round(blackboard: dict[str, Any], round_index: int) -> dict[str, Any]:
    board = deepcopy(blackboard)
    budget = dict(board.get("budget_state") or {})
    budget["rounds_completed"] = max(int(budget.get("rounds_completed") or 0), int(round_index))
    board["budget_state"] = budget
    return board


def build_agentic_guided_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    return build_guided_chemenzy_payload_from_blackboard(blackboard)


def rank_analogical_hypotheses_from_blackboard(blackboard: dict[str, Any]) -> dict[str, Any]:
    target_handles = set((blackboard.get("target_profile") or {}).get("functional_handles") or [])
    target_side = [
        dict(row)
        for row in (blackboard.get("target_side_disconnection_hypotheses") or {}).get("hypotheses") or []
        if isinstance(row, dict)
    ]
    analogical = [dict(row) for row in blackboard.get("analogical_hypotheses") or [] if isinstance(row, dict)]
    exact_rows = [dict(row) for row in (blackboard.get("literature_evidence") or {}).get("exact_rows") or [] if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate([*target_side, *analogical], start=1):
        handle = str(row.get("target_handle") or row.get("reaction_family") or "")
        preserves = {str(item) for item in row.get("must_preserve_substructure") or []}
        source_confidence = _source_confidence_score(row, exact_rows)
        score = 0
        if handle in target_handles:
            score += 25
        if "polycyclic_cage_core" in preserves:
            score += 20
        if source_confidence:
            score += source_confidence
        if row.get("target_proximity") == "target_proximal":
            score += 20
        if "large_atom_jump" in _blackboard_failure_reasons(blackboard):
            score -= 5
        rows.append(
            {
                "schema_version": "ranked_analogical_hypothesis.v1",
                "rank_input_index": idx,
                "hypothesis_id": str(row.get("hypothesis_id") or f"hypothesis_{idx}"),
                "target_handle": handle,
                "score": score,
                "score_components": {
                    "target_handle_overlap": handle in target_handles,
                    "target_core_retention": "polycyclic_cage_core" in preserves,
                    "source_confidence": source_confidence,
                    "previous_failure_penalty": "large_atom_jump" in _blackboard_failure_reasons(blackboard),
                },
                "required_verification": [str(item) for item in row.get("required_verification") or []],
                "no_solved_claim": True,
            }
        )
    ranked = sorted(rows, key=lambda item: (-int(item.get("score") or 0), str(item.get("hypothesis_id") or "")))
    return {
        "schema_version": "analogical_hypothesis_ranking.v1",
        "accepted": bool(ranked),
        "ranked_hypotheses": ranked,
        "selected_hypotheses": ranked[:3],
        "ranking_factors": [
            "target_handle_overlap",
            "target_core_retention",
            "heavy_atom_retention",
            "source_confidence",
            "likelihood_to_bridge_exact_rows",
            "previous_round_failure_penalty",
        ],
        "no_solved_claim": True,
        "reasons": [] if ranked else ["no_hypotheses_to_rank"],
    }


def _normalize_action_output(board: dict[str, Any], *, action_type: str, result: dict[str, Any], artifact_ref: str) -> bool:
    if action_type == "run_guided_chemenzy" and str(result.get("schema_version") or "") == "guided_chemenzy_rerun_result.v1":
        payload = dict(result)
    else:
        payload = dict(result.get("result") or result.get("artifact") or result)
    useful = bool(result.get("accepted", True))
    if action_type == "classify_route_objectives":
        artifact = payload
        board["route_objective_summary"] = _drop_large_fields(artifact)
        _extend_unique(board, "endpoint_candidates", artifact.get("endpoint_candidates") or [], unique_key="endpoint_id")
        route_scope = dict(artifact.get("route_scope") or {})
        if route_scope:
            belief = dict(board.get("current_belief") or {})
            belief["route_scope"] = route_scope
            constraints = dict(belief.get("constraints") or {})
            if route_scope.get("de_novo_core_construction_deprioritized"):
                constraints["de_novo_core_construction_deprioritized"] = True
                constraints["small_molecule_stock_closure_deprioritized"] = True
            if route_scope.get("objective_evidence_validation_required"):
                constraints["objective_evidence_validation_required"] = True
            belief["constraints"] = constraints
            board["current_belief"] = belief
        return bool(artifact.get("selected_objectives"))

    if action_type == "generate_disconnection_hypotheses":
        artifact = payload
        board["target_side_disconnection_hypotheses"] = _drop_large_fields(artifact)
        _extend_unique(board, "bridge_tasks", artifact.get("bridge_tasks") or [], unique_key="task_id")
        _extend_unique(board, "analogical_hypotheses", artifact.get("hypotheses") or [], unique_key="hypothesis_id")
        if artifact.get("route_objective_summary") and not board.get("route_objective_summary"):
            board["route_objective_summary"] = _drop_large_fields(dict(artifact.get("route_objective_summary") or {}))
        _extend_unique(board, "endpoint_candidates", artifact.get("endpoint_candidates") or [], unique_key="endpoint_id")
        _extend_unique(board, "semisynthesis_anchors", artifact.get("semisynthesis_anchors") or [], unique_key="anchor_id")
        evidence = dict(board.get("literature_evidence") or {})
        _extend_unique(evidence, "source_candidates", artifact.get("source_candidates") or [], unique_key="source_ref")
        if artifact.get("source_candidates"):
            evidence["confidence"] = "strategy_source_hints"
        board["literature_evidence"] = evidence
        route_scope = dict(artifact.get("route_scope") or {})
        if route_scope:
            belief = dict(board.get("current_belief") or {})
            belief["route_scope"] = route_scope
            constraints = dict(belief.get("constraints") or {})
            if route_scope.get("de_novo_core_construction_deprioritized"):
                constraints["de_novo_core_construction_deprioritized"] = True
                constraints["small_molecule_stock_closure_deprioritized"] = True
            if route_scope.get("objective_evidence_validation_required"):
                constraints["objective_evidence_validation_required"] = True
            belief["constraints"] = constraints
            board["current_belief"] = belief
        return bool(artifact.get("hypotheses") or artifact.get("semisynthesis_anchors"))

    if action_type == "build_failure_critic_report":
        artifact = payload
        existing_failures = {
            str(row.get("reason") or "")
            for row in board.get("route_failures") or []
            if isinstance(row, dict)
        }
        existing_tasks = {
            str(row.get("task_id") or "")
            for row in board.get("bridge_tasks") or []
            if isinstance(row, dict)
        }
        existing_blacklist = {
            str(row.get("canonical_smiles") or "")
            for row in board.get("terminal_blacklist") or []
            if isinstance(row, dict)
        }
        existing_blocked = {
            str(row.get("direction") or "")
            for row in (board.get("current_belief") or {}).get("blocked_directions") or []
            if isinstance(row, dict)
        }
        existing_bias = {
            str(item)
            for item in (board.get("current_belief") or {}).get("next_action_bias") or []
            if str(item).strip()
        }
        existing_constraints = dict((board.get("current_belief") or {}).get("constraints") or {})
        _extend_unique(board, "route_failures", artifact.get("route_failures") or [], unique_key="reason")
        _extend_unique(board, "bridge_tasks", artifact.get("bridge_tasks") or [], unique_key="task_id")
        _extend_unique(board, "terminal_blacklist", artifact.get("terminal_blacklist") or [], unique_key="canonical_smiles")
        belief = dict(board.get("current_belief") or {})
        _extend_unique(belief, "blocked_directions", artifact.get("blocked_directions") or [], unique_key="direction")
        _extend_unique(belief, "next_action_bias", [str(item) for item in artifact.get("next_action_bias") or []], unique_key=None)
        constraints = dict(belief.get("constraints") or {})
        constraints.update(dict(artifact.get("constraints") or {}))
        belief["constraints"] = constraints
        board["current_belief"] = belief
        new_failures = {
            str(row.get("reason") or "")
            for row in board.get("route_failures") or []
            if isinstance(row, dict)
        } - existing_failures
        new_tasks = {
            str(row.get("task_id") or "")
            for row in board.get("bridge_tasks") or []
            if isinstance(row, dict)
        } - existing_tasks
        new_blacklist = {
            str(row.get("canonical_smiles") or "")
            for row in board.get("terminal_blacklist") or []
            if isinstance(row, dict)
        } - existing_blacklist
        new_blocked = {
            str(row.get("direction") or "")
            for row in belief.get("blocked_directions") or []
            if isinstance(row, dict)
        } - existing_blocked
        new_bias = {
            str(item)
            for item in belief.get("next_action_bias") or []
            if str(item).strip()
        } - existing_bias
        return bool(
            artifact.get("accepted")
            and (
                new_failures
                or new_tasks
                or new_blacklist
                or new_blocked
                or new_bias
                or constraints != existing_constraints
            )
        )

    if action_type == "search_literature":
        evidence = dict(board.get("literature_evidence") or {})
        prior_source_mode = str(evidence.get("source_discovery_mode") or "")
        existing_source_keys = {
            _source_candidate_merge_key(row)
            for row in evidence.get("source_candidates") or []
            if isinstance(row, dict) and _source_candidate_merge_key(row)
        }
        existing_source_refs = {
            str(item).strip()
            for item in evidence.get("source_refs") or []
            if str(item).strip()
        }
        existing_proxy_request_ids = {
            str(row.get("request_id") or "").strip()
            for row in evidence.get("local_pdf_proxy_requests") or []
            if isinstance(row, dict)
        }
        evidence["source_candidates"] = _merge_source_candidate_rows(
            evidence.get("source_candidates") or [],
            payload.get("source_candidates") or [],
        )
        _extend_unique(evidence, "source_refs", payload.get("source_refs") or [], unique_key=None)
        real_sources = [
            dict(row)
            for row in evidence.get("source_candidates") or []
            if isinstance(row, dict) and _candidate_has_real_source(row)
        ]
        evidence["confidence"] = "candidate" if real_sources else (
            "placeholder" if evidence.get("source_candidates") else evidence.get("confidence", "none")
        )
        evidence["source_discovery_mode"] = str(payload.get("source_discovery_mode") or evidence.get("source_discovery_mode") or "")
        evidence["fallback_order"] = [str(item) for item in payload.get("fallback_order") or evidence.get("fallback_order") or []]
        evidence["scout_attempts"] = [
            dict(item)
            for item in payload.get("scout_attempts") or evidence.get("scout_attempts") or []
            if isinstance(item, dict)
        ]
        proxy_summary = payload.get("local_pdf_proxy_request_summary")
        if isinstance(proxy_summary, dict) and proxy_summary:
            evidence["local_pdf_proxy_request_summary"] = dict(proxy_summary)
        _extend_unique(
            evidence,
            "local_pdf_proxy_requests",
            payload.get("local_pdf_proxy_requests") or [],
            unique_key="request_id",
        )
        new_real_sources = [
            row
            for row in real_sources
            if _source_candidate_merge_key(row)
            and _source_candidate_merge_key(row) not in existing_source_keys
        ]
        placeholder_only = bool(payload.get("placeholder_only")) or bool(
            evidence.get("source_candidates")
            and not real_sources
            and all(
                isinstance(row, dict)
                and (
                    bool(row.get("placeholder_only"))
                    or str(row.get("access_status") or "").strip().lower() == "placeholder_only"
                )
                for row in evidence.get("source_candidates") or []
            )
        )
        new_source_refs = [
            str(item).strip()
            for item in evidence.get("source_refs") or []
            if not placeholder_only
            and str(item).strip()
            and str(item).strip() not in existing_source_refs
        ]
        new_proxy_requests = [
            row
            for row in evidence.get("local_pdf_proxy_requests") or []
            if isinstance(row, dict)
            and str(row.get("request_id") or "").strip()
            and str(row.get("request_id") or "").strip() not in existing_proxy_request_ids
        ]
        board["literature_evidence"] = evidence
        if payload.get("codex_worker_run_attempted"):
            budget = dict(board.get("budget_state") or {})
            budget["codex_research_runs"] = max(
                int(budget.get("codex_research_runs") or 0),
                int(payload.get("codex_research_runs") or 0),
            )
            board["budget_state"] = budget
        source_mode_changed = str(evidence.get("source_discovery_mode") or "") != prior_source_mode
        source_mode_upgrade = bool(source_mode_changed and str(evidence.get("source_discovery_mode") or "") != "placeholder")
        return bool(new_real_sources or new_source_refs or new_proxy_requests or source_mode_upgrade)

    if action_type == "extract_pdf_literature_structures":
        evidence = dict(board.get("literature_evidence") or {})
        summary = _pdf_structure_summary(payload, artifact_ref=artifact_ref)
        _extend_unique(evidence, "pdf_structure_evidence", [summary], unique_key="evidence_id")
        process_rows = [
            dict(row)
            for row in payload.get("literature_process_evidence_rows") or payload.get("process_evidence_rows") or []
            if isinstance(row, dict)
        ]
        _extend_unique(evidence, "process_evidence_rows", process_rows, unique_key="row_id")
        _extend_unique(board, "semisynthesis_anchors", semisynthesis_anchors_from_process_rows(process_rows), unique_key="anchor_id")
        if process_rows:
            belief = dict(board.get("current_belief") or {})
            _extend_unique(
                belief,
                "next_action_bias",
                ["derive_broad_reaction_template", "compile_objective_route_proof"],
                unique_key=None,
            )
            process_policy = dict(belief.get("process_policy") or {})
            process_policy["process_evidence_row_count"] = len(evidence.get("process_evidence_rows") or [])
            process_policy["process_evidence_is_not_exact_row"] = True
            process_policy["process_evidence_is_not_parent_route_proof"] = True
            belief["process_policy"] = process_policy
            board["current_belief"] = belief
        evidence["confidence"] = "pdf_rendered" if summary.get("accepted") else evidence.get("confidence", "none")
        board["literature_evidence"] = evidence
        counts = dict(summary.get("summary") or {})
        return bool(
            counts.get("rendered_page_count")
            or counts.get("indexed_image_count")
            or counts.get("scheme_crop_count")
            or process_rows
        )

    if action_type == "extract_visual_literature_chain":
        evidence = dict(board.get("literature_evidence") or {})
        chain_rows = payload.get("chains") or payload.get("visual_chains") or []
        if not chain_rows and payload:
            chain_rows = [_compact_artifact(payload, artifact_ref=artifact_ref)]
        _extend_unique(evidence, "visual_chains", chain_rows, unique_key="chain_id")
        visual_reasons = [
            str(item)
            for item in payload.get("reasons") or []
            if str(item or "").strip() and _visual_reason_is_runtime_failure(str(item))
        ]
        if visual_reasons:
            _extend_unique(
                board,
                "plugin_runtime_diagnostics",
                [
                    {
                        "schema_version": "agent_visual_runtime_diagnostic.v1",
                        "diagnostic_id": "visual_runtime:" + ":".join(visual_reasons[:3]),
                        "reasons": visual_reasons,
                        "source_ref": str(payload.get("source_ref") or ""),
                        "artifact_ref": artifact_ref,
                        "allowed_use": "blackboard_failure_feedback_only",
                        "not_parent_route_proof": True,
                        "no_solved_claim": True,
                    }
                ],
                unique_key="diagnostic_id",
            )
        process_rows = [
            dict(row)
            for row in payload.get("literature_process_evidence_rows") or payload.get("process_evidence_rows") or []
            if isinstance(row, dict)
        ]
        _extend_unique(evidence, "process_evidence_rows", process_rows, unique_key="row_id")
        _extend_unique(board, "semisynthesis_anchors", semisynthesis_anchors_from_process_rows(process_rows), unique_key="anchor_id")
        if process_rows:
            belief = dict(board.get("current_belief") or {})
            _extend_unique(
                belief,
                "next_action_bias",
                ["derive_broad_reaction_template", "compile_objective_route_proof"],
                unique_key=None,
            )
            process_policy = dict(belief.get("process_policy") or {})
            process_policy["process_evidence_row_count"] = len(evidence.get("process_evidence_rows") or [])
            process_policy["process_evidence_is_not_exact_row"] = True
            process_policy["process_evidence_is_not_parent_route_proof"] = True
            belief["process_policy"] = process_policy
            board["current_belief"] = belief
        structure_tasks: list[dict[str, Any]] = []
        for row in chain_rows:
            if isinstance(row, dict):
                structure_tasks.extend(_structure_resolution_tasks_from_visual_chain(row))
        _extend_unique(evidence, "structure_resolution_tasks", structure_tasks, unique_key="task_id")
        board["literature_evidence"] = evidence
        return bool(chain_rows or process_rows)

    if action_type == "resolve_literature_structure_task":
        evidence = dict(board.get("literature_evidence") or {})
        resolved = [
            dict(row)
            for row in payload.get("resolved_structures") or []
            if isinstance(row, dict) and row.get("accepted")
        ]
        unresolved = [
            dict(row)
            for row in payload.get("unresolved_tasks") or []
            if isinstance(row, dict)
        ]
        attempt = _structure_resolution_attempt_summary(payload, artifact_ref=artifact_ref)
        existing_structure_ids = {
            str(row.get("structure_id") or "")
            for row in evidence.get("resolved_structures") or []
            if isinstance(row, dict)
        }
        existing_attempt_ids = {
            str(row.get("attempt_id") or "")
            for row in evidence.get("structure_resolution_attempts") or []
            if isinstance(row, dict)
        }
        _extend_unique(evidence, "resolved_structures", resolved, unique_key="structure_id")
        if attempt:
            _extend_unique(evidence, "structure_resolution_attempts", [attempt], unique_key="attempt_id")
        if resolved or unresolved:
            _update_structure_resolution_task_statuses(evidence, resolved=resolved, unresolved=unresolved)
        board["literature_evidence"] = evidence
        promoted_anchors = _semisynthesis_anchors_from_resolved_structures(
            board,
            resolved,
            artifact_ref=artifact_ref,
        )
        if promoted_anchors:
            _extend_unique(board, "semisynthesis_anchors", promoted_anchors, unique_key="anchor_id")
            _extend_unique(
                board,
                "bridge_tasks",
                [
                    _semisynthesis_bridge_task_from_anchor(board, anchor)
                    for anchor in promoted_anchors
                ],
                unique_key="task_id",
            )
            belief = dict(board.get("current_belief") or {})
            _extend_unique(
                belief,
                "next_action_bias",
                ["run_guided_chemenzy", "compile_objective_route_proof"],
                unique_key=None,
            )
            board["current_belief"] = belief
        new_resolved = [
            row
            for row in resolved
            if str(row.get("structure_id") or "") not in existing_structure_ids
        ]
        new_attempt = bool(attempt and str(attempt.get("attempt_id") or "") not in existing_attempt_ids)
        return bool(new_resolved or new_attempt or promoted_anchors)

    if action_type == "compile_exact_literature_rows":
        evidence = dict(board.get("literature_evidence") or {})
        existing_rows = [
            dict(row)
            for row in evidence.get("exact_rows") or []
            if isinstance(row, dict)
        ]
        rows = _annotate_exact_rows_with_target_relevance(_exact_rows_from_payload(payload), board)
        merged_rows, exact_rows_changed = _merge_versioned_exact_rows(
            existing_rows,
            rows,
        )
        evidence["exact_rows"] = merged_rows
        evidence["exact_row_target_relevance_summary"] = _exact_row_target_relevance_summary(evidence.get("exact_rows") or [])
        audit = _exact_chain_audit_summary(payload, artifact_ref=artifact_ref)
        if audit:
            _extend_unique(evidence, "exact_chain_audits", [audit], unique_key="audit_id")
        terminals = _literature_terminal_candidates_from_payload(payload, artifact_ref=artifact_ref)
        if terminals:
            _extend_unique(evidence, "terminal_candidates", terminals, unique_key="terminal_id")
            _extend_unique(
                board,
                "bridge_tasks",
                [_literature_terminal_bridge_task(board, terminal) for terminal in terminals],
                unique_key="task_id",
            )
        evidence["confidence"] = "exact_rows" if evidence.get("exact_rows") else evidence.get("confidence", "none")
        board["literature_evidence"] = evidence
        return bool(exact_rows_changed or terminals)

    if action_type == "rank_analogical_hypotheses":
        board["analogical_hypothesis_ranking"] = payload
        return bool(payload.get("selected_hypotheses"))

    if action_type == "extract_analogical_reaction_templates":
        templates = [dict(row) for row in payload.get("templates") or [] if isinstance(row, dict)]
        board["analogical_templates"] = templates
        board["analogical_template_ranking"] = {}
        board["template_applications"] = []
        board["template_failure_memory"] = []
        return bool(payload.get("templates"))

    if action_type == "rank_analogical_reaction_templates":
        board["analogical_template_ranking"] = payload
        return bool(payload.get("selected_templates"))

    if action_type == "apply_analogical_template_to_target":
        summaries = [
            compact_template_application_summary(row)
            for row in payload.get("applications") or []
            if isinstance(row, dict)
        ]
        if summaries:
            board["template_applications"] = summaries
        _merge_template_failure_memory(board, payload.get("template_failure_memory") or [])
        cache_key = str(payload.get("target_smiles") or "")
        if cache_key:
            board.setdefault("template_cache_refs", {})[cache_key] = artifact_ref
        return bool(payload.get("accepted_application_count") or payload.get("executable_candidate_count"))

    if action_type == "validate_template_application":
        refs = dict(payload.get("compiled_downstream_refs") or {})
        for key, value in refs.items():
            if str(value or ""):
                board.setdefault("artifact_refs", {})[f"analogical_template_{key}"] = str(value)
        belief = dict(board.get("current_belief") or {})
        template_policy = dict(belief.get("template_policy") or {})
        template_policy["validated_one_step_row_count"] = int(payload.get("one_step_row_count") or 0)
        template_policy["validated_guided_hint_count"] = int(payload.get("one_step_row_count") or 0)
        template_policy["analogy_is_advisory_only"] = True
        template_policy["analogical_template_hints_are_not_exact_rows"] = True
        template_policy["not_parent_route_proof"] = True
        belief["template_policy"] = template_policy
        board["current_belief"] = belief
        return bool(payload.get("one_step_row_count"))

    if action_type == "derive_broad_reaction_template":
        _extend_unique(board, "broad_transform_templates", payload.get("templates") or [], unique_key="template_id")
        belief = dict(board.get("current_belief") or {})
        template_policy = dict(belief.get("template_policy") or {})
        template_policy["broad_transform_template_count"] = len(board.get("broad_transform_templates") or [])
        template_policy["broad_templates_are_advisory_only"] = True
        template_policy["not_parent_route_proof"] = True
        belief["template_policy"] = template_policy
        board["current_belief"] = belief
        return bool(payload.get("templates"))

    if action_type == "run_guided_chemenzy":
        _update_from_guided_chemenzy(board, payload, artifact_ref)
        return bool(
            payload.get("raw_route_verifier")
            or payload.get("route_failure_feedback")
            or payload.get("chemenzy_runtime_diagnostic")
            or payload.get("accepted")
        )

    if action_type == "expand_child_target":
        result_payload = dict(payload.get("result") or payload)
        _extend_unique(
            board,
            "chemenzy_attempts",
            [
                dict(row.get("chem_enzy_attempt_outcome") or {})
                for row in result_payload.get("subgoals") or []
                if isinstance(row, dict) and isinstance(row.get("chem_enzy_attempt_outcome"), dict)
            ],
            unique_key="attempt_id",
        )
        board["route_expansion_subgoals"] = _route_expansion_subgoal_summaries(payload)
        task_status_changed = _mark_recursive_hypothesis_task_attempts(
            board,
            route_expansion_result=payload,
            artifact_ref=artifact_ref,
        )
        failure_feedback = _proposal_failure_feedback_from_route_expansion(
            board,
            route_expansion_result=payload,
            artifact_ref=artifact_ref,
        )
        existing_feedback_ids = {
            str(row.get("feedback_id") or "")
            for row in board.get("proposal_failure_feedback") or []
            if isinstance(row, dict)
        }
        _extend_unique(board, "proposal_failure_feedback", failure_feedback, unique_key="feedback_id")
        new_feedback_ids = {
            str(row.get("feedback_id") or "")
            for row in board.get("proposal_failure_feedback") or []
            if isinstance(row, dict)
        } - existing_feedback_ids
        recursive_tasks = recursive_hypothesis_tasks_from_route_expansion(
            blackboard=board,
            route_expansion_result=payload,
        )
        existing_task_ids = {
            str(row.get("task_id") or "")
            for row in board.get("recursive_hypothesis_tasks") or []
            if isinstance(row, dict)
        }
        _extend_unique(board, "recursive_hypothesis_tasks", recursive_tasks, unique_key="task_id")
        new_task_ids = {
            str(row.get("task_id") or "")
            for row in board.get("recursive_hypothesis_tasks") or []
            if isinstance(row, dict)
        } - existing_task_ids
        if new_task_ids:
            belief = dict(board.get("current_belief") or {})
            _extend_unique(belief, "next_action_bias", ["expand_child_target"], unique_key=None)
            board["current_belief"] = belief
        belief = dict(board.get("current_belief") or {})
        belief["child_route_any_solved"] = bool(payload.get("accepted_subgoal_count") or payload.get("solved"))
        strict_frontiers = _strict_literature_frontiers(board)
        accepted_frontiers = {
            str(row.get("canonical_smiles") or "")
            for row in board.get("route_expansion_subgoals") or []
            if isinstance(row, dict) and row.get("accepted") is True
        }
        belief["child_route_solved"] = bool(
            strict_frontiers
            and any(
                frontier
                and all(smiles in accepted_frontiers for smiles in frontier)
                for frontier in strict_frontiers
            )
        ) if strict_frontiers else bool(payload.get("accepted_subgoal_count") or payload.get("solved"))
        board["current_belief"] = belief
        board.setdefault("artifact_refs", {})["route_expansion_subgoal_search"] = artifact_ref
        return bool(payload.get("accepted_subgoal_count") or payload.get("solved") or new_task_ids or new_feedback_ids or task_status_changed)

    if action_type == "stitch_parent_route":
        proof = dict(payload.get("parent_route_proof") or payload)
        board["parent_route_proof"] = proof
        return bool(proof.get("accepted") or proof.get("reasons"))

    if action_type == "compile_objective_route_proof":
        proof = dict(payload.get("route_proof_bundle") or payload)
        board["route_proof_bundle"] = proof
        return bool(proof.get("accepted") or proof.get("objective_proofs") or proof.get("reasons"))

    if action_type == "stop_unresolved":
        belief = dict(board.get("current_belief") or {})
        stops = list(belief.get("stop_candidates") or [])
        stops.append({"schema_version": "agent_stop_candidate.v1", "reason": "stop_unresolved_action"})
        belief["stop_candidates"] = stops
        board["current_belief"] = belief
        return False
    return useful


def _proposal_failure_feedback_from_route_expansion(
    board: dict[str, Any],
    *,
    route_expansion_result: dict[str, Any],
    artifact_ref: str,
) -> list[dict[str, Any]]:
    payload = dict(route_expansion_result.get("result") or route_expansion_result or {})
    target = dict(board.get("target_profile") or {})
    default_parent = _canonical_or_raw_smiles(str(target.get("target_smiles") or target.get("canonical_smiles") or ""))
    rows: list[dict[str, Any]] = []
    for raw in payload.get("subgoals") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if bool(row.get("accepted") or row.get("solved")):
            continue
        verifier = dict(row.get("verifier") or {})
        subgoal = dict(row.get("subgoal") or {})
        if not _is_hypothesis_like_subgoal(subgoal):
            continue
        failed_smiles = _canonical_or_raw_smiles(str(subgoal.get("smiles") or ""))
        if not failed_smiles:
            continue
        policy = dict(subgoal.get("policy") or subgoal.get("chem_enzy_search_policy") or {})
        preferred = dict(policy.get("preferred_subgoal") or {})
        nested_target = dict(preferred.get("hypothetical_precursor_target") or {})
        precursor_set = _canonical_or_raw_smiles(
            str(
                subgoal.get("precursor_set_smiles")
                or preferred.get("precursor_set_smiles")
                or nested_target.get("precursor_set_smiles")
                or ""
            )
        )
        siblings = _dedupe_strings(
            [
                *[str(item) for item in subgoal.get("sibling_precursor_smiles") or []],
                *[str(item) for item in nested_target.get("sibling_precursor_smiles") or []],
            ]
        )
        reasons = _dedupe_strings(
            [
                *[str(item) for item in row.get("reasons") or []],
                *[str(item) for item in verifier.get("reasons") or []],
                *[str(item) for item in payload.get("reasons") or []],
            ]
        )
        proposal_id = str(
            subgoal.get("parent_candidate_id")
            or nested_target.get("parent_candidate_id")
            or nested_target.get("source_proposal_id")
            or ""
        )
        feedback_id = "proposal_failure:" + _stable_hash(
            "|".join(
                [
                    proposal_id,
                    str(subgoal.get("recursive_hypothesis_task_id") or ""),
                    str(subgoal.get("name") or ""),
                    failed_smiles,
                    precursor_set,
                    "|".join(reasons),
                ]
            )
        )
        rows.append(
            {
                "schema_version": "proposal_failure_feedback.v1",
                "feedback_id": feedback_id,
                "source": "route_expansion_subgoal_search",
                "artifact_ref": artifact_ref,
                "proposal_id": proposal_id,
                "recursive_hypothesis_task_id": str(subgoal.get("recursive_hypothesis_task_id") or ""),
                "subgoal_name": str(subgoal.get("name") or ""),
                "parent_smiles": _canonical_or_raw_smiles(str(subgoal.get("parent_smiles") or "")) or default_parent,
                "failed_precursor_smiles": failed_smiles,
                "precursor_set_smiles": precursor_set,
                "precursor_component_index": int(subgoal.get("precursor_component_index") or nested_target.get("precursor_component_index") or 0),
                "precursor_component_count": int(subgoal.get("precursor_component_count") or nested_target.get("precursor_component_count") or 1),
                "sibling_precursor_smiles": siblings,
                "requires_precursor_set_stitching": bool(
                    subgoal.get("requires_precursor_set_stitching")
                    or nested_target.get("requires_precursor_set_stitching")
                    or precursor_set
                ),
                "failure_reasons": reasons,
                "next_refinement_bias": _proposal_failure_refinement_bias(failed_smiles, precursor_set),
                "allowed_use": "proposal_refinement_and_recursive_search_seed_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "child_route_cannot_promote_parent": True,
                "no_solved_claim": True,
            }
        )
    return rows


def _mark_recursive_hypothesis_task_attempts(
    board: dict[str, Any],
    *,
    route_expansion_result: dict[str, Any],
    artifact_ref: str,
) -> bool:
    tasks = [row for row in board.get("recursive_hypothesis_tasks") or [] if isinstance(row, dict)]
    if not tasks:
        return False
    payload = dict(route_expansion_result.get("result") or route_expansion_result or {})
    attempts = _recursive_hypothesis_attempts_from_route_expansion(payload)
    if not attempts:
        return False
    by_id: dict[str, dict[str, Any]] = {
        str(task.get("task_id") or ""): task
        for task in tasks
        if str(task.get("task_id") or "")
    }
    by_smiles: dict[str, dict[str, Any]] = {}
    for task in tasks:
        smiles = _canonical_or_raw_smiles(str(task.get("precursor_smiles") or ""))
        if smiles and smiles not in by_smiles:
            by_smiles[smiles] = task
    status_changed = False
    for attempt in attempts:
        task = by_id.get(str(attempt.get("task_id") or ""))
        if task is None:
            task = by_smiles.get(str(attempt.get("smiles") or ""))
        if task is None:
            continue
        before_status = str(task.get("status") or "pending")
        previous_attempts = _nonnegative_int(task.get("attempt_count"), 0)
        task["attempt_count"] = previous_attempts + 1
        task["last_attempt_artifact_ref"] = artifact_ref
        task["last_attempt_smiles"] = str(attempt.get("smiles") or "")
        task["last_attempt_reasons"] = [str(item) for item in attempt.get("reasons") or [] if str(item or "").strip()]
        task["last_attempt_accepted"] = bool(attempt.get("accepted"))
        task["status"] = "accepted_child_route" if attempt.get("accepted") else "rejected"
        status_changed = status_changed or str(task.get("status") or "") != before_status
    return status_changed


def _recursive_hypothesis_attempts_from_route_expansion(payload: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for raw in payload.get("subgoals") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        subgoal = dict(row.get("subgoal") or {})
        if not _is_hypothesis_like_subgoal(subgoal):
            continue
        task_id = str(subgoal.get("recursive_hypothesis_task_id") or "")
        smiles = _canonical_or_raw_smiles(str(subgoal.get("smiles") or ""))
        if not task_id and not smiles:
            continue
        verifier = dict(row.get("verifier") or {})
        accepted = bool(row.get("accepted") or row.get("solved"))
        reasons = _dedupe_strings(
            [
                *[str(item) for item in row.get("reasons") or []],
                *[str(item) for item in verifier.get("reasons") or []],
                *[str(item) for item in payload.get("reasons") or []],
            ]
        )
        attempts.append(
            {
                "task_id": task_id,
                "smiles": smiles,
                "accepted": accepted,
                "reasons": reasons,
            }
        )
    return attempts


def _is_hypothesis_like_subgoal(subgoal: dict[str, Any]) -> bool:
    policy = dict(subgoal.get("policy") or subgoal.get("chem_enzy_search_policy") or {})
    compiler = dict(policy.get("compiler_metadata") or {})
    text = " ".join(
        str(value or "")
        for value in (
            subgoal.get("source"),
            subgoal.get("name"),
            subgoal.get("recursive_hypothesis_task_id"),
            subgoal.get("task_scope"),
            compiler.get("compiler_schema"),
            subgoal.get("precursor_set_smiles"),
        )
    ).lower()
    return bool(
        subgoal.get("hypothesis_only_not_solved")
        or subgoal.get("recursive_hypothesis_task_id")
        or subgoal.get("requires_precursor_set_stitching")
        or compiler.get("hypothesis_only_not_solved")
        or compiler.get("recursive_hypothesis_frontier")
        or "hypothesis" in text
        or "proposal" in text
    )


def _proposal_failure_refinement_bias(failed_smiles: str, precursor_set: str) -> list[str]:
    bias = ["change_hypothesis_granularity"]
    smiles_values = [
        str(failed_smiles or ""),
        *[part for part in str(precursor_set or "").split(".") if part],
    ]
    if any(_smiles_has_substructure(value, "[C](=O)[OX2H]") or _smiles_has_substructure(value, "[C](=O)Cl") for value in smiles_values):
        bias.append("try_alternate_acyl_activation_state")
    if any(
        _smiles_has_substructure(value, "[C](=O)([#6])[#6]")
        or _smiles_has_substructure(value, "[C](=O)[C]=[C]")
        or _smiles_has_substructure(value, "[CH]=O")
        for value in smiles_values
    ):
        bias.append("try_redox_or_unsaturation_state_refinement")
    if any(
        _smiles_has_substructure(value, "[CX4][OX2H]")
        or _smiles_has_substructure(value, "[CX4]Cl")
        or _smiles_has_substructure(value, "[OX2]C(C)=O")
        for value in smiles_values
    ):
        bias.append("try_protection_or_leaving_group_state_refinement")
    if "." in precursor_set:
        bias.append("refine_failed_precursor_component")
    return _dedupe_strings(bias)


def _smiles_has_substructure(smiles: str, smarts: str) -> bool:
    if Chem is None:
        return False
    mol = Chem.MolFromSmiles(str(smiles or ""))
    query = Chem.MolFromSmarts(str(smarts or ""))
    return bool(mol is not None and query is not None and mol.HasSubstructMatch(query))


def _canonical_or_raw_smiles(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    try:
        return str(canonical_smiles(text) or text)
    except Exception:
        return text


def _stable_hash(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:12]


def _merge_template_failure_memory(board: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    existing = {
        str(row.get("template_id") or ""): dict(row)
        for row in board.get("template_failure_memory") or []
        if isinstance(row, dict)
    }
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        template_id = str(raw.get("template_id") or "")
        if not template_id:
            continue
        current = dict(existing.get(template_id) or {})
        previous_count = int(current.get("failure_count") or 0)
        reasons = [str(item) for item in current.get("reasons") or []]
        for item in raw.get("reasons") or []:
            text = str(item)
            if text and text not in reasons:
                reasons.append(text)
        current.update(dict(raw))
        current["failure_count"] = previous_count + int(raw.get("failure_count") or 1)
        current["reasons"] = reasons
        existing[template_id] = current
    board["template_failure_memory"] = list(existing.values())


def _semisynthesis_anchors_from_resolved_structures(
    board: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    artifact_ref: str,
) -> list[dict[str, Any]]:
    target = dict(board.get("target_profile") or {})
    target_smiles = str(target.get("canonical_smiles") or target.get("target_smiles") or "")
    target_canonical = _canonical_or_raw_smiles(target_smiles)
    target_heavy = _smiles_heavy_atom_count(target_canonical or target_smiles)
    existing = {
        _canonical_or_raw_smiles(str(row.get("smiles") or ""))
        for row in board.get("semisynthesis_anchors") or []
        if isinstance(row, dict)
    }
    anchors: list[dict[str, Any]] = []
    for row in rows:
        smiles = str(row.get("smiles") or "").strip()
        canonical = _canonical_or_raw_smiles(smiles)
        if not canonical or canonical in existing:
            continue
        text = _resolved_structure_text(row)
        if _resolved_structure_is_target_like(canonical, target_canonical=target_canonical):
            continue
        candidate_heavy = _smiles_heavy_atom_count(canonical or smiles)
        if not _resolved_structure_is_semisynthesis_anchor_candidate(
            row,
            text=text,
            candidate_heavy=candidate_heavy,
            target_heavy=target_heavy,
            candidate_smiles=canonical,
            target_smiles=target_canonical,
        ):
            continue
        name = _resolved_structure_anchor_name(row)
        source_ref = str(row.get("source_ref") or "").strip()
        structure_id = str(row.get("structure_id") or "")
        anchor_id = "resolved_structure_anchor:" + _safe_anchor_token(
            f"{source_ref}:{name}:{structure_id or canonical}"
        )
        anchor = {
            "schema_version": "semisynthesis_anchor.v1",
            "anchor_id": anchor_id,
            "case_id": str(board.get("case_id") or ""),
            "anchor_type": "source_resolved_same_scaffold_intermediate",
            "objective_type": "semisynthesis_from_source_resolved_intermediate",
            "name": name,
            "smiles": smiles,
            "canonical_smiles": canonical,
            "role": "Source-resolved same-scaffold intermediate for target-proximal semisynthesis or bridge search.",
            "route_role": "objective_endpoint_candidate",
            "source_ref": source_ref,
            "source_locator": str(row.get("source_locator") or ""),
            "evidence_refs": _dedupe_strings(
                [
                    artifact_ref,
                    source_ref,
                    str(row.get("task_id") or ""),
                    structure_id,
                    *[str(item) for item in row.get("evidence_refs") or []],
                ]
            ),
            "required_verification": [
                "same_core_anchor_identity",
                "target_side_conversion_logic",
                "objective_endpoint_source_validation",
                "route_verifier_acceptance",
            ],
            "allowed_use": "route_objective_anchor_hint_only",
            "not_exact_literature_segment": True,
            "not_parent_route_proof": True,
            "requires_source_validation": True,
            "no_solved_claim": True,
            "confidence": str(row.get("confidence") or "low"),
            "source_structure_id": structure_id,
        }
        anchors.append(anchor)
        existing.add(canonical)
    return anchors


def _semisynthesis_bridge_task_from_anchor(board: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    anchor_id = str(anchor.get("anchor_id") or "")
    return {
        "schema_version": "agent_bridge_task.v1",
        "task_id": f"semisynthesis_bridge:{anchor_id}",
        "case_id": str(board.get("case_id") or ""),
        "task_type": "objective_endpoint_anchor_validation",
        "target_name": str((board.get("target_profile") or {}).get("target_name") or ""),
        "target_handle": "semisynthesis_from_source_resolved_intermediate",
        "required_bridge": "Validate and connect the source-resolved same-scaffold intermediate to the parent target.",
        "source_hypothesis_id": "resolved_structure_semisynthesis_anchor",
        "anchor_id": anchor_id,
        "anchor": dict(anchor),
        "status": "open",
        "required_verification": [
            "same_core_anchor_identity",
            "target_side_conversion_logic",
            "objective_endpoint_source_validation",
            "route_verifier_acceptance",
        ],
        "no_solved_claim": True,
    }


def _resolved_structure_is_semisynthesis_anchor_candidate(
    row: dict[str, Any],
    *,
    text: str,
    candidate_heavy: int = 0,
    target_heavy: int = 0,
    candidate_smiles: str = "",
    target_smiles: str = "",
) -> bool:
    if not bool(row.get("accepted")):
        return False
    if not str(row.get("smiles") or "").strip():
        return False
    if row.get("rdkit_valid") is False:
        return False
    if candidate_heavy:
        if target_heavy >= 30:
            minimum_heavy = 18
        elif target_heavy:
            minimum_heavy = max(8, int(target_heavy * 0.35))
        else:
            minimum_heavy = 12
        if candidate_heavy < minimum_heavy:
            return False
    explicit_role = str(
        row.get("route_role")
        or row.get("structure_role")
        or row.get("semantic_role")
        or ""
    ).strip().lower()
    anchor_tokens = (
        "same scaffold",
        "same-core",
        "semisynthesis",
        "advanced intermediate",
        "advanced precursor",
        "route intermediate",
    )
    role_tokens = {
        "advanced_intermediate",
        "advanced_precursor",
        "route_intermediate",
        "semisynthesis_anchor",
        "same_scaffold_intermediate",
    }
    semantic_signal = explicit_role in role_tokens or any(token in text for token in anchor_tokens)
    return bool(
        semantic_signal
        and _resolved_structure_shares_target_scaffold(candidate_smiles, target_smiles)
    )


def _resolved_structure_is_target_like(
    smiles: str,
    *,
    target_canonical: str,
) -> bool:
    if not target_canonical:
        return False
    if smiles == target_canonical:
        return True
    if Chem is None:
        return False
    candidate = Chem.MolFromSmiles(smiles)
    target = Chem.MolFromSmiles(target_canonical)
    if candidate is None or target is None:
        return False
    return Chem.MolToSmiles(candidate, isomericSmiles=False) == Chem.MolToSmiles(
        target,
        isomericSmiles=False,
    )


def _resolved_structure_anchor_name(row: dict[str, Any]) -> str:
    label = str(row.get("label") or "").strip()
    generic_labels = {
        "all structures",
        "all route structures",
        "all intermediates",
        "advanced intermediate",
    }
    if label and label.lower() not in generic_labels:
        return label
    return "source-resolved same-scaffold intermediate"


def _resolved_structure_shares_target_scaffold(candidate_smiles: str, target_smiles: str) -> bool:
    if Chem is None or not candidate_smiles or not target_smiles:
        return False
    candidate = Chem.MolFromSmiles(candidate_smiles)
    target = Chem.MolFromSmiles(target_smiles)
    if candidate is None or target is None:
        return False
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        candidate_scaffold = MurckoScaffold.GetScaffoldForMol(candidate)
        target_scaffold = MurckoScaffold.GetScaffoldForMol(target)
    except Exception:
        return False
    candidate_atoms = int(candidate_scaffold.GetNumHeavyAtoms())
    target_atoms = int(target_scaffold.GetNumHeavyAtoms())
    if candidate_atoms < 6 or target_atoms < 6:
        return False
    smaller, larger = (
        (candidate_scaffold, target_scaffold)
        if candidate_atoms <= target_atoms
        else (target_scaffold, candidate_scaffold)
    )
    smaller_atoms = min(candidate_atoms, target_atoms)
    larger_atoms = max(candidate_atoms, target_atoms)
    return bool(
        smaller_atoms / max(1, larger_atoms) >= 0.55
        and larger.HasSubstructMatch(smaller, useChirality=False)
    )


def _resolved_structure_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("label") or ""),
            str(row.get("source_locator") or ""),
            str(row.get("source_ref") or ""),
            str(row.get("task_id") or ""),
            str(row.get("derivation_mode") or ""),
        ]
    ).lower()


def _smiles_heavy_atom_count(smiles: str) -> int:
    if Chem is None:
        return 0
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _safe_anchor_token(value: str) -> str:
    text = str(value or "").strip().lower()
    token = "".join(ch if ch.isalnum() else "_" for ch in text)
    token = "_".join(part for part in token.split("_") if part)
    return f"{token[:80]}:{_stable_hash(text)}"


def _update_from_guided_chemenzy(board: dict[str, Any], payload: dict[str, Any], artifact_ref: str) -> None:
    guided_payload = _guided_chemenzy_result_payload(payload)
    attempt_outcome = dict(
        payload.get("chem_enzy_attempt_outcome")
        or guided_payload.get("chem_enzy_attempt_outcome")
        or {}
    )
    if attempt_outcome:
        attempt_outcome["artifact_ref"] = artifact_ref
        _extend_unique(
            board,
            "chemenzy_attempts",
            [attempt_outcome],
            unique_key="attempt_id",
        )
        if str(attempt_outcome.get("attempt_kind") or "") in {"standard", "retry"}:
            belief = dict(board.get("current_belief") or {})
            belief.pop("pending_chemenzy_attempt", None)
            board["current_belief"] = belief
    verifier = dict(payload.get("raw_route_verifier") or {})
    if not verifier and guided_payload:
        verifier = dict(guided_payload.get("raw_route_verifier") or {})
    if verifier:
        proof_bank = verifier.get("route_proof_bank")
        if isinstance(proof_bank, dict):
            bank_digest = str(proof_bank.get("content_hash") or "")
            if not bank_digest:
                bank_digest = hashlib.sha256(
                    json.dumps(
                        proof_bank,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
            _extend_unique(
                board,
                "chemenzy_route_proof_banks",
                [
                    {
                        "schema_version": "blackboard_chemenzy_route_proof_bank.v1",
                        "bank_id": f"chemenzy-proof-bank:{bank_digest}",
                        "artifact_ref": artifact_ref,
                        "route_proof_bank": deepcopy(proof_bank),
                        "no_solved_claim": True,
                        "requires_current_host_replay": True,
                    }
                ],
                unique_key="bank_id",
            )
        expected_target_smiles = str((board.get("target_profile") or {}).get("target_smiles") or "")
        belief = dict(board.get("current_belief") or {})
        belief["parent_route_verifier"] = _compact_parent_route_verifier(
            verifier,
            artifact_ref=artifact_ref,
            expected_target_smiles=expected_target_smiles,
        )
        if _parent_route_verifier_solved(
            verifier,
            expected_target_smiles=expected_target_smiles,
        ):
            _extend_unique(belief, "next_action_bias", ["stitch_parent_route"], unique_key=None)
        board["current_belief"] = belief
        if not verifier.get("accepted"):
            _extend_unique(
                board,
                "route_failures",
                [
                    {
                        "schema_version": "agent_route_failure.v1",
                        "reason": str(reason),
                        "route_status": str(verifier.get("route_status") or ""),
                        "artifact_ref": artifact_ref,
                    }
                    for reason in verifier.get("reasons") or []
                ],
                unique_key="reason",
            )
    runtime = dict(payload.get("literature_template_plugin_runtime") or guided_payload.get("literature_template_plugin_runtime") or {})
    if runtime:
        _extend_unique(board, "plugin_runtime_diagnostics", [runtime], unique_key="schema_version")
    chemenzy_runtime = dict(payload.get("chemenzy_runtime_diagnostic") or guided_payload.get("chemenzy_runtime_diagnostic") or {})
    if chemenzy_runtime:
        _extend_unique(board, "plugin_runtime_diagnostics", [chemenzy_runtime], unique_key="diagnostic_id")
        _extend_unique(
            board,
            "route_failures",
            [
                {
                    "schema_version": "agent_route_failure.v1",
                    "reason": str(reason),
                    "route_status": "unresolved",
                    "artifact_ref": artifact_ref,
                    "failure_class": "chemenzy_runtime_diagnostic",
                }
                for reason in chemenzy_runtime.get("reasons") or []
            ],
            unique_key="reason",
        )
    unresolved_failures = _guided_chemenzy_unresolved_failures(guided_payload, artifact_ref=artifact_ref)
    if unresolved_failures:
        _extend_unique(board, "route_failures", unresolved_failures, unique_key="reason")
        belief = dict(board.get("current_belief") or {})
        probe_exhausted = any(
            str(row.get("reason") or "") == "guided_chemenzy_probe_exhausted"
            for row in unresolved_failures
        )
        if probe_exhausted:
            _extend_unique(belief, "next_action_bias", ["run_guided_chemenzy"], unique_key=None)
            belief["pending_chemenzy_attempt"] = {
                "schema_version": "pending_chemenzy_attempt.v1",
                "action_kind": "guided",
                "attempt_kind": "standard",
                "from_attempt_id": str(attempt_outcome.get("attempt_id") or ""),
                "reason": "probe_exhausted",
            }
        else:
            _extend_unique(
                belief,
                "next_action_bias",
                ["build_failure_critic_report", "search_literature"],
                unique_key=None,
            )
        board["current_belief"] = belief
    feedback = dict(payload.get("route_failure_feedback") or {})
    if feedback and isinstance(feedback.get("path"), str):
        board.setdefault("artifact_refs", {})["route_failure_feedback"] = str(feedback.get("path") or "")


def _guided_chemenzy_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if str(payload.get("schema_version") or "") == "guided_chemenzy_rerun_result.v1":
        return dict(payload)
    result = payload.get("result")
    if isinstance(result, dict) and str(result.get("schema_version") or "") == "guided_chemenzy_rerun_result.v1":
        return dict(result)
    return {}


def _guided_chemenzy_unresolved_failures(payload: dict[str, Any], *, artifact_ref: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    attempt_outcome = dict(payload.get("chem_enzy_attempt_outcome") or {})
    outcome_name = str(attempt_outcome.get("outcome") or "")
    attempt_kind = str(attempt_outcome.get("attempt_kind") or "")
    next_attempt_kind = str(attempt_outcome.get("next_attempt_kind") or "")
    if (
        outcome_name == "probe_exhausted"
        or (
            outcome_name in {"verification_rejected", "verification_missing"}
            and attempt_kind == "probe"
            and next_attempt_kind == "standard"
        )
    ):
        return [
            {
                "schema_version": "agent_route_failure.v1",
                "reason": "guided_chemenzy_probe_exhausted",
                "route_status": "unresolved",
                "artifact_ref": artifact_ref,
                "failure_class": "guided_chemenzy_probe_status",
                "attempt_id": str(attempt_outcome.get("attempt_id") or ""),
                "attempt_kind": "probe",
                "next_attempt_kind": "standard",
                "search_exhaustive": False,
                "attempt_outcome": outcome_name,
                "raw_solved": bool(attempt_outcome.get("raw_solved")),
                "verified_solved": False,
            }
        ]
    if outcome_name in {"verification_rejected", "verification_missing"}:
        return [
            {
                "schema_version": "agent_route_failure.v1",
                "reason": (
                    "guided_chemenzy_verification_rejected"
                    if outcome_name == "verification_rejected"
                    else "guided_chemenzy_verification_missing"
                ),
                "route_status": "unresolved",
                "artifact_ref": artifact_ref,
                "failure_class": "guided_chemenzy_host_verification",
                "attempt_id": str(attempt_outcome.get("attempt_id") or ""),
                "attempt_kind": attempt_kind,
                "next_attempt_kind": next_attempt_kind,
                "search_exhaustive": bool(
                    attempt_outcome.get("search_exhaustive")
                ),
                "blocks_same_attempt": bool(
                    attempt_outcome.get("blocks_same_attempt")
                ),
                "raw_solved": bool(attempt_outcome.get("raw_solved")),
                "verified_solved": False,
            }
        ]
    raw = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    search_status = raw.get("search_status") if isinstance(raw.get("search_status"), dict) else {}
    route_status = str(payload.get("route_status") or search_status.get("status") or "").strip().lower()
    solved = bool(payload.get("solved") or search_status.get("solved"))
    ok = raw.get("ok")
    n_results = raw.get("n_results")
    try:
        n_results_int = int(n_results)
    except (TypeError, ValueError):
        n_results_int = -1
    diagnosis = [
        str(item)
        for item in raw.get("failure_diagnosis") or []
        if str(item or "").strip()
    ]
    for item in raw.get("backend_failures") or []:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or item.get("category") or "").strip()
        if reason:
            diagnosis.append(reason)
    search_failed = str(search_status.get("status") or "").strip().lower() in {"failed", "unresolved", "timeout"}
    explicit_no_route = any("no_route" in str(item).lower() for item in diagnosis)
    if solved or not (
        route_status in {"unresolved", "failed", "timeout"}
        or search_failed
        or explicit_no_route
        or ok is False
        or n_results_int == 0
    ):
        return []
    reasons = ["no_route_found"] if explicit_no_route or n_results_int == 0 else ["guided_chemenzy_unresolved"]
    return [
        {
            "schema_version": "agent_route_failure.v1",
            "reason": reason,
            "route_status": route_status or "unresolved",
            "artifact_ref": artifact_ref,
            "failure_class": "guided_chemenzy_search_status",
            "search_status": str(search_status.get("status") or ""),
            "n_results": n_results_int if n_results_int >= 0 else None,
        }
        for reason in reasons
    ]


def _compact_parent_route_verifier(
    verifier: dict[str, Any],
    *,
    artifact_ref: str,
    expected_target_smiles: str = "",
) -> dict[str, Any]:
    audit = dict(verifier.get("target_equivalence_audit") or {})
    solved = is_reaction_validated_route_verifier_report(
        verifier,
        expected_target_smiles=expected_target_smiles,
    )
    return {
        "schema_version": "agent_parent_route_verifier_summary.v1",
        "verifier_schema_version": str(verifier.get("schema_version") or ""),
        "accepted": solved,
        "solved": solved,
        "route_status": str(verifier.get("route_status") or ""),
        "verification_level": str(verifier.get("verification_level") or ""),
        "reaction_validated": bool(verifier.get("reaction_validated")),
        "target_match": bool(verifier.get("target_match") or audit.get("target_match")),
        "accepted_route_count": _nonnegative_int(verifier.get("accepted_route_count"), 0),
        "best_route_rank": verifier.get("best_route_rank"),
        "best_route_step_count": _nonnegative_int(verifier.get("best_route_step_count"), 0),
        "rejected_route_count": _nonnegative_int(verifier.get("rejected_route_count"), 0),
        "reasons": [str(item) for item in verifier.get("reasons") or []],
        "warnings": [str(item) for item in verifier.get("warnings") or []],
        "artifact_ref": artifact_ref,
        "proof_ready_action": "stitch_parent_route",
        "final_verdict_authority": "deterministic_parent_route_proof",
        "raw_route_output_not_embedded": True,
    }


def _parent_route_verifier_solved(
    verifier: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> bool:
    return is_reaction_validated_route_verifier_report(
        verifier,
        expected_target_smiles=expected_target_smiles,
    )


def _exact_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = payload.get("exact_rows") or payload.get("one_step_rows")
    if isinstance(explicit, list):
        return [
            _row_summary(
                row,
                idx,
                compilation_accepted=payload.get("accepted") is True,
            )
            for idx, row in enumerate(explicit, start=1)
            if isinstance(row, dict)
        ]
    compiled = dict(payload.get("compiled_downstream") or payload)
    plugin = dict(compiled.get("literature_template_plugin") or {})
    rows = plugin.get("one_step_rows") or (plugin.get("plugin_flags") or {}).get("one_step_rows") or []
    return [
        _row_summary(
            row,
            idx,
            compilation_accepted=(
                payload.get("accepted") is True or compiled.get("accepted") is True
            ),
        )
        for idx, row in enumerate(rows, start=1)
        if isinstance(row, dict)
    ]


def _merge_versioned_exact_rows(
    existing_rows: list[dict[str, Any]],
    incoming_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Upsert current parser rows while preserving independent human curation."""

    incoming_by_id = {
        str(row.get("row_id") or ""): dict(row)
        for row in incoming_rows
        if str(row.get("row_id") or "")
    }
    current_parser_rows = [
        row
        for row in incoming_rows
        if row.get("accepted") is True
        and str(row.get("deterministic_parser_authority_id") or "").strip()
    ]
    current_parser_ids = {
        str(row.get("deterministic_parser_authority_id") or "").strip()
        for row in current_parser_rows
    }
    current_sources = {
        str(row.get("source_ref") or "").strip().casefold()
        for row in current_parser_rows
        if str(row.get("source_ref") or "").strip()
    }

    merged: list[dict[str, Any]] = []
    for existing in existing_rows:
        row_id = str(existing.get("row_id") or "")
        replacement = incoming_by_id.get(row_id)
        if replacement and replacement.get("accepted") is True:
            continue
        source_ref = str(existing.get("source_ref") or "").strip().casefold()
        parser_id = str(
            existing.get("deterministic_parser_authority_id") or ""
        ).strip()
        is_source_detail = row_id.startswith("source_detail_exact_step:")
        has_replayable_binding_shape = bool(
            existing.get("source_evidence")
            and existing.get("exact_step_validation")
        )
        obsolete_machine_row = bool(
            is_source_detail
            and source_ref in current_sources
            and (
                (parser_id and parser_id not in current_parser_ids)
                or (not parser_id and not has_replayable_binding_shape)
            )
        )
        if obsolete_machine_row:
            continue
        merged.append(existing)

    existing_ids = {str(row.get("row_id") or "") for row in merged}
    for incoming in incoming_rows:
        row_id = str(incoming.get("row_id") or "")
        if row_id in existing_ids and incoming.get("accepted") is not True:
            continue
        if row_id in existing_ids:
            merged = [
                row
                for row in merged
                if str(row.get("row_id") or "") != row_id
            ]
        merged.append(dict(incoming))
        existing_ids.add(row_id)

    before = json.dumps(
        existing_rows,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    after = json.dumps(
        merged,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return merged, before != after


def _literature_terminal_candidates_from_payload(
    payload: dict[str, Any],
    *,
    artifact_ref: str,
) -> list[dict[str, Any]]:
    chain = dict(payload.get("chain_audit") or {})
    if not chain and isinstance(payload.get("result"), dict):
        chain = dict((payload.get("result") or {}).get("chain_audit") or {})
    if not chain:
        return []
    steps = [dict(row) for row in chain.get("chain") or chain.get("steps") or [] if isinstance(row, dict)]
    products = {
        canonical_smiles(str(step.get("product_smiles") or ""))
        for step in steps
        if canonical_smiles(str(step.get("product_smiles") or ""))
    }
    reactant_by_key: dict[str, str] = {}
    for step in steps:
        for raw in step.get("reactant_smiles") or []:
            smiles = str(raw or "").strip()
            canonical = canonical_smiles(smiles)
            if smiles and canonical:
                reactant_by_key.setdefault(canonical, smiles)
    frontier = sorted(key for key in reactant_by_key if key not in products)
    if not frontier:
        fallback = str(
            chain.get("terminal_smiles")
            or chain.get("observed_terminal_smiles")
            or _last_chain_main_reactant(steps)
            or ""
        ).strip()
        fallback_key = canonical_smiles(fallback)
        if fallback and fallback_key:
            frontier = [fallback_key]
            reactant_by_key[fallback_key] = fallback
    if not frontier:
        return []

    strict_source_proof = _source_detail_chain_strict_proof_eligible(chain)
    audit_id = str(artifact_ref or chain.get("case_id") or "source_detail_chain_audit")
    source_ref = _chain_source_ref(chain)
    repair = dict(chain.get("terminal_stereo_repair") or {})
    candidates: list[dict[str, Any]] = []
    for index, canonical in enumerate(frontier, start=1):
        smiles = reactant_by_key[canonical]
        is_named_terminal = canonical in {
            canonical_smiles(str(chain.get("terminal_smiles") or "")),
            canonical_smiles(str(chain.get("observed_terminal_smiles") or "")),
        }
        candidates.append(
            {
                "schema_version": "agent_literature_terminal_candidate.v1",
                "terminal_id": f"source_detail_terminal:{_stable_hash(audit_id + '|' + canonical)}",
                "name": str(
                    chain.get("terminal_name") or repair.get("name") or "source detail literature terminal"
                ) if is_named_terminal else f"source detail frontier {index}",
                "smiles": smiles,
                "canonical_smiles": canonical,
                "source_ref": source_ref,
                "source": "source_detail_chain_route",
                "source_chain_audit_id": audit_id,
                "frontier_index": index,
                "frontier_count": len(frontier),
                "step_count": int(chain.get("step_count") or len(steps)),
                "terminal_reached": bool(chain.get("terminal_reached")),
                "terminal_requested": bool(chain.get("terminal_requested")),
                "stereo_repair": repair if is_named_terminal else {},
                "strict_source_proof_eligible": strict_source_proof,
                "terminal_candidate_level": (
                    "strict_source_detail" if strict_source_proof else "advisory_source_detail"
                ),
                "requires_all_frontiers_closed": True,
                "exact_target_override": True,
                "target_equivalence_audit_required": True,
                "no_solved_claim": True,
            }
        )
    return candidates


def _exact_chain_audit_summary(payload: dict[str, Any], *, artifact_ref: str) -> dict[str, Any]:
    chain = dict(payload.get("chain_audit") or {})
    if not chain and isinstance(payload.get("result"), dict):
        chain = dict((payload.get("result") or {}).get("chain_audit") or {})
    if not chain:
        return {}
    summary = dict(chain.get("summary") or {})
    frontier = _source_detail_chain_frontier(chain)
    strict_source_proof = _source_detail_chain_strict_proof_eligible(chain)
    return {
        "schema_version": "agent_exact_chain_audit_summary.v1",
        "audit_id": str(artifact_ref or chain.get("case_id") or "source_detail_chain_audit"),
        "accepted": bool(chain.get("accepted")),
        "reasons": [str(item) for item in chain.get("reasons") or []],
        "step_count": int(chain.get("step_count") or summary.get("chain_step_count") or 0),
        "one_step_row_count": int(summary.get("one_step_row_count") or 0),
        "terminal_reached": bool(chain.get("terminal_reached")),
        "terminal_requested": bool(chain.get("terminal_requested")),
        "observed_terminal_smiles": str(chain.get("observed_terminal_smiles") or ""),
        "target_smiles": str(chain.get("target_smiles") or ""),
        "chain_schema_version": str(chain.get("schema_version") or ""),
        "strict_source_proof_eligible": strict_source_proof,
        "terminal_frontier": frontier,
        "terminal_frontier_count": len(frontier),
        "requires_all_frontiers_closed": True,
        "artifact_ref": str(artifact_ref or ""),
    }


def _source_detail_chain_strict_proof_eligible(chain: dict[str, Any]) -> bool:
    steps = [dict(row) for row in chain.get("chain") or chain.get("steps") or [] if isinstance(row, dict)]
    return bool(
        chain.get("schema_version") == "source_detail_route_chain_audit.v1"
        and chain.get("accepted") is True
        and steps
        and _source_detail_chain_frontier(chain)
        and all(is_validated_source_detail_literature_step(step) for step in steps)
    )


def _source_detail_chain_frontier(chain: dict[str, Any]) -> list[str]:
    steps = [dict(row) for row in chain.get("chain") or chain.get("steps") or [] if isinstance(row, dict)]
    products = {
        canonical_smiles(str(step.get("product_smiles") or ""))
        for step in steps
        if canonical_smiles(str(step.get("product_smiles") or ""))
    }
    reactants = {
        canonical_smiles(str(smiles or ""))
        for step in steps
        for smiles in step.get("reactant_smiles") or []
        if canonical_smiles(str(smiles or ""))
    }
    return sorted(reactants - products)


def _strict_literature_frontiers(board: dict[str, Any]) -> list[list[str]]:
    evidence = dict(board.get("literature_evidence") or {})
    frontiers: list[list[str]] = []
    for raw in evidence.get("exact_chain_audits") or []:
        if not isinstance(raw, dict) or raw.get("strict_source_proof_eligible") is not True:
            continue
        frontier = sorted(
            {
                canonical_smiles(str(smiles or ""))
                for smiles in raw.get("terminal_frontier") or []
                if canonical_smiles(str(smiles or ""))
            }
        )
        if frontier:
            frontiers.append(frontier)
    return frontiers


def _route_expansion_subgoal_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = dict(payload.get("result") or payload)
    summaries: list[dict[str, Any]] = []
    for raw in result.get("subgoals") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        subgoal = dict(row.get("subgoal") or {})
        smiles = str(subgoal.get("smiles") or subgoal.get("target_smiles") or "").strip()
        canonical = canonical_smiles(smiles)
        verifier = dict(row.get("verifier") or {})
        verifier_accepted = is_reaction_validated_route_verifier_report(
            verifier,
            expected_target_smiles=smiles,
        ) if smiles else False
        summaries.append(
            {
                "schema_version": "agent_route_expansion_subgoal_summary.v1",
                "subgoal_id": str(subgoal.get("child_target_id") or subgoal.get("terminal_id") or canonical or smiles),
                "name": str(subgoal.get("name") or subgoal.get("target_name") or ""),
                "smiles": smiles,
                "canonical_smiles": canonical,
                "accepted": bool(
                    row.get("accepted") is True
                    and row.get("solved") is True
                    and verifier_accepted
                ),
                "verifier_accepted": verifier_accepted,
                "route_status": str(row.get("route_status") or verifier.get("route_status") or ""),
                "raw_result_path": str(row.get("raw_result_path") or ""),
                "request_path": str(row.get("request_path") or ""),
                "reasons": [str(item) for item in row.get("reasons") or []],
            }
        )
    return summaries


def _literature_terminal_bridge_task(board: dict[str, Any], terminal: dict[str, Any]) -> dict[str, Any]:
    target = dict(board.get("target_profile") or {})
    canonical = str(terminal.get("canonical_smiles") or "")
    return {
        "schema_version": "agent_bridge_task.v1",
        "task_id": f"literature_terminal_child:{canonical}",
        "task_type": "upstream_terminal_synthesis",
        "target_name": str(target.get("target_name") or ""),
        "target_handle": "source_detail_literature_terminal",
        "required_bridge": "Find an independently verified upstream route for this source-detail frontier component.",
        "required_verification": [
            "child_target_route_verifier",
            "parent_bridge_connectivity",
            "exact_frontier_identity",
            "all_source_chain_frontiers_closed",
        ],
        "strict_source_proof_eligible": bool(terminal.get("strict_source_proof_eligible")),
        "requires_all_frontiers_closed": True,
        "priority": "high",
        "status": "open",
        "terminal": dict(terminal),
    }


def _last_chain_main_reactant(steps: Any) -> str:
    rows = [dict(row) for row in steps or [] if isinstance(row, dict)]
    if not rows:
        return ""
    last = rows[-1]
    return str(last.get("main_reactant_smiles") or "")


def _chain_source_ref(chain: dict[str, Any]) -> str:
    if chain.get("source_ref"):
        return str(chain.get("source_ref") or "")
    for row in chain.get("chain") or []:
        if isinstance(row, dict) and row.get("source_ref"):
            return str(row.get("source_ref") or "")
    return ""


def _row_summary(
    row: dict[str, Any],
    idx: int,
    *,
    compilation_accepted: bool = False,
) -> dict[str, Any]:
    def string_list(value: Any) -> list[str]:
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, (list, tuple)):
            return []
        return _dedupe_strings(
            [str(item).strip() for item in values if str(item).strip()]
        )

    trace = dict(row.get("literature_template_trace") or {})
    template = row.get("template") if isinstance(row.get("template"), dict) else row.get("templates")
    if isinstance(template, dict):
        trace = {**dict(template.get("literature_template_trace") or {}), **trace}
    raw_reactants = (
        trace.get("reactant_smiles")
        or trace.get("precursor_smiles")
        or row.get("reactant_smiles")
        or row.get("precursor_smiles")
        or []
    )
    if isinstance(raw_reactants, str):
        raw_reactants = raw_reactants.split(".")
    reactant_smiles = string_list(raw_reactants)
    mapped_reaction = str(
        trace.get("atom_mapped_reaction_smiles")
        or trace.get("mapped_reaction_smiles")
        or row.get("atom_mapped_reaction_smiles")
        or row.get("mapped_reaction_smiles")
        or ""
    ).strip()
    accepted = bool(
        compilation_accepted
        and row.get("accepted", True) is not False
        and row.get("validated", True) is not False
    )
    condition_candidate = dict(
        trace.get("condition_candidate")
        or row.get("condition_candidate")
        or {}
    )
    condition_text = string_list(trace.get("conditions") or row.get("conditions"))
    if not condition_text:
        condition_text = _condition_candidate_text_values(condition_candidate)
    return {
        "schema_version": "agent_exact_literature_row_summary.v1",
        "row_id": str(row.get("row_id") or trace.get("source_template_id") or f"exact_row_{idx}"),
        "source_template_id": str(
            trace.get("source_template_id") or row.get("source_template_id") or ""
        ),
        "source_ref": str(trace.get("source_ref") or row.get("source_ref") or ""),
        "source_refs": _dedupe_strings(
            [
                *string_list(trace.get("source_refs")),
                *string_list(row.get("source_refs")),
            ]
        ),
        "product_smiles": str(trace.get("product_smiles") or row.get("product_smiles") or ""),
        "product_label": str(trace.get("product_label") or row.get("product_label") or row.get("product_name") or ""),
        "reactant_smiles": reactant_smiles,
        "reaction_smiles": str(
            trace.get("reaction_smiles") or row.get("reaction_smiles") or ""
        ),
        "atom_mapped_reaction_smiles": mapped_reaction,
        "reaction_family": str(
            trace.get("reaction_family") or row.get("reaction_family") or ""
        ),
        "conditions": condition_text,
        "condition_candidate": _drop_large_fields(condition_candidate),
        "source_locator": str(trace.get("source_locator") or row.get("source_locator") or ""),
        "evidence_refs": string_list(
            trace.get("evidence_refs") or row.get("evidence_refs")
        ),
        "source_detail_exact_step": bool(
            trace.get("source_detail_exact_step")
            or row.get("source_detail_exact_step")
        ),
        "relation_type": str(
            trace.get("relation_type") or row.get("relation_type") or ""
        ),
        "exact_step_validation": _drop_large_fields(
            trace.get("exact_step_validation")
            or row.get("exact_step_validation")
            or {}
        ),
        "source_evidence": _drop_large_fields(
            trace.get("source_evidence") or row.get("source_evidence") or {}
        ),
        "curator_record_id": str(
            trace.get("curator_record_id")
            or row.get("curator_record_id")
            or ""
        ),
        "deterministic_parser_authority_id": str(
            trace.get("deterministic_parser_authority_id")
            or row.get("deterministic_parser_authority_id")
            or ""
        ),
        "source_binding_reaction_digest": str(
            trace.get("source_binding_reaction_digest")
            or row.get("source_binding_reaction_digest")
            or ""
        ),
        "source_formulation": _drop_large_fields(
            trace.get("source_formulation")
            or row.get("source_formulation")
            or {}
        ),
        "accepted": accepted,
        "validated": accepted,
        "validation_status": "accepted_by_compiler" if accepted else "unvalidated",
        "confidence": str(row.get("confidence") or "source_detail_exact"),
        "no_solved_claim": True,
    }


def _condition_candidate_text_values(value: dict[str, Any]) -> list[str]:
    """Project structured source conditions without dropping the source row."""
    rows: list[str] = []
    raw_conditions = value.get("conditions")
    if isinstance(raw_conditions, str):
        rows.append(raw_conditions)
    elif isinstance(raw_conditions, (list, tuple)):
        rows.extend(str(item) for item in raw_conditions if str(item or "").strip())
    for key in (
        "reagent",
        "reagents",
        "catalyst",
        "enzyme",
        "solvent",
        "temperature",
        "time",
        "ph",
        "buffer",
        "atmosphere",
        "workup",
    ):
        raw = value.get(key)
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        text = ", ".join(str(item).strip() for item in values if str(item or "").strip())
        if text:
            rows.append(f"{key}={text}")
    return _dedupe_strings(rows)


def _annotate_exact_rows_with_target_relevance(rows: list[dict[str, Any]], board: dict[str, Any]) -> list[dict[str, Any]]:
    return [_annotate_exact_row_with_target_relevance(dict(row), board) for row in rows if isinstance(row, dict)]


def _annotate_exact_row_with_target_relevance(row: dict[str, Any], board: dict[str, Any]) -> dict[str, Any]:
    relevance = _exact_row_target_relevance(row, board)
    annotated = dict(row)
    annotated["target_relevance"] = relevance
    annotated["target_core_retained"] = bool(relevance.get("target_core_retained"))
    annotated["target_relevant_for_parent_bridge"] = bool(relevance.get("target_relevant_for_parent_bridge"))
    return annotated


def _exact_row_target_relevance(row: dict[str, Any], board: dict[str, Any]) -> dict[str, Any]:
    target = dict(board.get("target_profile") or {})
    target_smiles = str(target.get("isomeric_smiles") or target.get("canonical_smiles") or target.get("target_smiles") or "").strip()
    product_smiles = str(row.get("product_smiles") or row.get("product") or row.get("product_canonical_smiles") or "").strip()
    target_heavy, target_rings = _smiles_heavy_atoms_and_rings(target_smiles)
    product_heavy, product_rings = _smiles_heavy_atoms_and_rings(product_smiles)
    target_heavy = int(target.get("heavy_atoms") or target_heavy or 0)
    target_rings = int(target.get("rings") or target_rings or 0)
    reasons: list[str] = []
    if not product_smiles:
        reasons.append("missing_product_smiles")
    if target_rings >= 3 and product_rings < max(2, target_rings - 1):
        reasons.append("product_ring_system_too_small_for_target_core")
    if target_heavy and product_heavy and product_heavy < max(8, int(target_heavy * 0.45)):
        reasons.append("product_heavy_atoms_too_small_for_target_core")
    row_text = " ".join(
        str(row.get(key) or "")
        for key in ("row_id", "product_label", "source_locator", "source_ref")
    ).lower()
    sugar_like = any(
        token in row_text
        for token in ("sugar", "rhamnose", "glycoside", "trichloroacetimidate", "benzoyl", "acetyl")
    )
    if sugar_like and target_rings >= 3 and product_rings <= 2:
        reasons.append("row_describes_low_ring_glycosyl_or_sugar_fragment")
    target_core_retained = bool(
        product_smiles
        and (target_rings <= 0 or product_rings >= max(2, target_rings - 1))
        and (not target_heavy or not product_heavy or product_heavy >= max(8, int(target_heavy * 0.45)))
        and "row_describes_low_ring_glycosyl_or_sugar_fragment" not in reasons
    )
    target_relevant = bool(target_core_retained)
    if target_relevant:
        reasons.append("product_size_and_ring_system_are_compatible_with_target_core")
    return {
        "schema_version": "exact_row_target_relevance.v1",
        "target_core_retained": target_core_retained,
        "target_relevant_for_parent_bridge": target_relevant,
        "product_heavy_atoms": int(product_heavy or 0),
        "product_ring_count": int(product_rings or 0),
        "target_heavy_atoms": int(target_heavy or 0),
        "target_ring_count": int(target_rings or 0),
        "reasons": reasons,
    }


def _smiles_heavy_atoms_and_rings(smiles: str) -> tuple[int, int]:
    if not smiles or Chem is None:
        return 0, 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, 0
    return int(mol.GetNumHeavyAtoms()), int(mol.GetRingInfo().NumRings())


def _target_relevant_exact_row_count(rows: list[Any]) -> int:
    count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("target_relevant_for_parent_bridge")):
            count += 1
            continue
        relevance = row.get("target_relevance")
        if isinstance(relevance, dict) and bool(relevance.get("target_relevant_for_parent_bridge")):
            count += 1
    return count


def _exact_row_target_relevance_summary(rows: list[Any]) -> dict[str, Any]:
    total = sum(1 for row in rows if isinstance(row, dict))
    relevant = _target_relevant_exact_row_count(rows)
    disconnected = total - relevant
    return {
        "schema_version": "exact_row_target_relevance_summary.v1",
        "total_exact_rows": total,
        "target_relevant_exact_rows": relevant,
        "disconnected_exact_rows": max(0, disconnected),
    }


def _artifact_name(action: dict[str, Any], result: dict[str, Any]) -> str:
    action_type = str(action.get("action_type") or "action")
    schema = str((result.get("result") or result).get("schema_version") or "")
    suffix = schema.replace(".", "_") if schema else action_type
    return f"{str(action.get('action_id') or action_type).replace(':', '_')}_{suffix}"


def _summary_safe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _drop_large_fields(value)
    return {"value": str(value)}


def _enrich_action_result_with_source_context(action: dict[str, Any], action_result: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type") or "")
    if action_type not in {
        "extract_pdf_literature_structures",
        "extract_visual_literature_chain",
        "compile_exact_literature_rows",
        "resolve_literature_structure_task",
    }:
        return dict(action_result)
    payload_context = _source_context_from_action_payload(dict(action.get("payload") or {}))
    if not payload_context:
        return dict(action_result)
    enriched = dict(action_result)
    if isinstance(enriched.get("result"), dict):
        result_payload = dict(enriched.get("result") or {})
        enriched["result"] = _merge_missing_source_context(result_payload, payload_context)
        return enriched
    if isinstance(enriched.get("artifact"), dict):
        artifact_payload = dict(enriched.get("artifact") or {})
        enriched["artifact"] = _merge_missing_source_context(artifact_payload, payload_context)
        return enriched
    return _merge_missing_source_context(enriched, payload_context)


def _source_context_from_action_payload(payload: dict[str, Any]) -> dict[str, str]:
    context: dict[str, str] = {}
    source_ref = str(payload.get("source_ref") or "").strip()
    source_title = str(payload.get("source_title") or payload.get("title") or "").strip()
    pdf_path = str(payload.get("pdf_path") or payload.get("local_pdf") or payload.get("source_pdf_path") or "").strip()
    if source_ref:
        context["source_ref"] = source_ref
    if source_title:
        context["source_title"] = source_title
    if pdf_path:
        context["pdf_path"] = pdf_path
        context["source_pdf_path"] = pdf_path
    return context


def _merge_missing_source_context(payload: dict[str, Any], context: dict[str, str]) -> dict[str, Any]:
    merged = dict(payload)
    for key, value in context.items():
        if value and not str(merged.get(key) or "").strip():
            merged[key] = value
    if str(context.get("source_pdf_path") or "").strip() and not str(merged.get("source_pdf_path") or "").strip():
        merged["source_pdf_path"] = str(context.get("source_pdf_path") or "")
    if str(context.get("pdf_path") or "").strip() and not str(merged.get("pdf_path") or "").strip():
        merged["pdf_path"] = str(context.get("pdf_path") or "")
    candidate = merged.get("candidate_chain")
    if isinstance(candidate, dict):
        merged["candidate_chain"] = _merge_missing_source_context(dict(candidate), context)
    parsed = merged.get("parsed_output")
    if isinstance(parsed, dict):
        merged["parsed_output"] = _merge_missing_source_context(dict(parsed), context)
    for list_key in ("chains", "visual_chains"):
        rows = merged.get(list_key)
        if isinstance(rows, list):
            merged[list_key] = [
                _merge_missing_source_context(dict(item), context) if isinstance(item, dict) else item
                for item in rows
            ]
    return merged


def _drop_large_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in {"routes", "raw_routes", "route_tree", "reaction_candidates"}:
                out[key] = f"<omitted:{str(key)}>"
            else:
                out[key] = _drop_large_fields(item)
        return out
    if isinstance(value, list):
        return [_drop_large_fields(item) for item in value[:50]]
    return value


def _compact_artifact(payload: dict[str, Any], *, artifact_ref: str) -> dict[str, Any]:
    quality = dict(payload.get("candidate_quality") or {})
    candidate = dict(payload.get("candidate_chain") or payload.get("parsed_output") or {})
    extraction_gaps = candidate.get("extraction_gaps") or payload.get("extraction_gaps") or []
    missing_expected = quality.get("missing_expected_labels") or payload.get("missing_expected_labels") or []
    condition_gap_labels = quality.get("condition_gap_labels") or payload.get("condition_gap_labels") or []
    gap_labels: list[str] = []
    warning_gap_labels: list[str] = []
    for gap in extraction_gaps:
        if not isinstance(gap, dict):
            continue
        raw_labels = gap.get("labels") if isinstance(gap.get("labels"), list) else [gap.get("label")]
        target = warning_gap_labels if _nonblocking_visual_gap(gap) else gap_labels
        target.extend(str(item) for item in raw_labels if str(item or "").strip())
    gap_labels.extend(str(item) for item in condition_gap_labels if str(item or "").strip())
    payload_source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    candidate_source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    source_ref = str(
        payload.get("source_ref")
        or candidate.get("source_ref")
        or payload_source.get("source_ref")
        or payload_source.get("doi")
        or candidate_source.get("source_ref")
        or candidate_source.get("doi")
        or ""
    )
    source_title = str(
        payload.get("source_title")
        or candidate.get("source_title")
        or payload_source.get("title")
        or candidate_source.get("title")
        or ""
    )
    steps = candidate.get("steps") or candidate.get("candidate_steps") or payload.get("steps") or []
    compact_steps = _compact_visual_candidate_steps(steps)
    structure_tasks_preview = _structure_resolution_tasks_from_gaps(
        extraction_gaps,
        source_ref=source_ref,
        source_title=source_title,
        artifact_ref=artifact_ref,
    )
    return {
        "schema_version": "agent_visual_chain_summary.v1",
        "chain_id": str(payload.get("chain_id") or payload.get("case_id") or artifact_ref or "visual_chain"),
        "accepted": bool(payload.get("accepted", True)),
        "source_ref": source_ref,
        "source_title": source_title,
        "source_pdf_path": str(payload.get("source_pdf_path") or payload.get("pdf_path") or ""),
        "artifact_ref": artifact_ref,
        "candidate_step_count": int(payload.get("candidate_step_count") or len(steps)),
        "step_count": len(compact_steps),
        "steps": compact_steps,
        "acceptance_level": str(payload.get("acceptance_level") or (quality.get("acceptance_level") if isinstance(quality, dict) else "") or ""),
        "exact_ready": bool(payload.get("exact_ready") or (quality.get("exact_ready") if isinstance(quality, dict) else False)),
        "exploratory_accepted": bool(payload.get("exploratory_accepted") or (quality.get("exploratory_accepted") if isinstance(quality, dict) else False)),
        "missing_expected_labels": [str(item) for item in missing_expected if str(item or "").strip()],
        "condition_gap_labels": [str(item) for item in condition_gap_labels if str(item or "").strip()],
        "gap_labels": _dedupe_strings(gap_labels),
        "warning_gap_labels": _dedupe_strings(warning_gap_labels),
        "extraction_gaps": _drop_large_fields(extraction_gaps),
        "structure_resolution_task_count": len(structure_tasks_preview),
        "structure_resolution_tasks": structure_tasks_preview,
        "reasons": [str(item) for item in payload.get("reasons") or []],
        "page_focus_refresh_audit": _drop_large_fields(
            dict(payload.get("page_focus_refresh_audit") or {})
        ),
    }


def _compact_visual_candidate_steps(steps: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(steps, list):
        return rows
    for idx, raw in enumerate(steps[:30], start=1):
        if not isinstance(raw, dict):
            continue
        derivation = dict(raw.get("structure_derivation") or {})
        not_exact = bool(
            raw.get("not_exact_literature_segment")
            or derivation.get("not_exact_literature_segment")
            or derivation.get("approximate_structure")
            or str(raw.get("allowed_use") or derivation.get("allowed_use") or "").startswith("exploratory")
        )
        rows.append(
            {
                "schema_version": "agent_visual_candidate_step_summary.v1",
                "step_id": str(raw.get("step_id") or f"visual_step_{idx}"),
                "product_label": str(raw.get("product_label") or ""),
                "product_smiles": str(raw.get("product_smiles") or ""),
                "reactant_labels": [str(item) for item in raw.get("reactant_labels") or [] if str(item or "").strip()][:6],
                "reactant_smiles": [str(item) for item in raw.get("reactant_smiles") or [] if str(item or "").strip()][:6],
                "main_reactant_smiles": str(raw.get("main_reactant_smiles") or ""),
                "source_locator": str(raw.get("source_locator") or ""),
                "confidence": str(raw.get("confidence") or derivation.get("confidence") or "low"),
                "stereochemistry_status": str(raw.get("stereochemistry_status") or derivation.get("stereochemistry_status") or ""),
                "allowed_use": str(raw.get("allowed_use") or derivation.get("allowed_use") or ("exploratory_template_and_guided_hint_only" if not_exact else "exact_candidate")),
                "not_exact_literature_segment": bool(not_exact),
                "risk_flags": _dedupe_strings(
                    [
                        *[str(item) for item in raw.get("risk_flags") or [] if str(item or "").strip()],
                        *[str(item) for item in derivation.get("risk_flags") or [] if str(item or "").strip()],
                    ]
                ),
                "no_solved_claim": True,
            }
        )
    return rows


def _structure_resolution_tasks_from_visual_chain(row: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = row.get("structure_resolution_tasks")
    if isinstance(explicit, list) and explicit:
        return [dict(item) for item in explicit if isinstance(item, dict)]
    return _structure_resolution_tasks_from_gaps(
        row.get("extraction_gaps") or [],
        source_ref=str(row.get("source_ref") or ""),
        source_title=str(row.get("source_title") or ""),
        artifact_ref=str(row.get("artifact_ref") or ""),
    )


def _structure_resolution_tasks_from_gaps(
    gaps: Any,
    *,
    source_ref: str,
    source_title: str,
    artifact_ref: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps or [], start=1):
        if not isinstance(gap, dict):
            continue
        gap_type = str(gap.get("gap_type") or gap.get("type") or "").strip().lower()
        if "structure" not in gap_type:
            continue
        raw_labels = gap.get("labels") if isinstance(gap.get("labels"), list) else [gap.get("label")]
        for label in raw_labels:
            text = str(label or "").strip()
            if not text:
                continue
            source_key = source_ref or source_title or "unknown_source"
            task_key = _safe_task_key(f"{source_key}:{text}")
            tasks.append(
                {
                    "schema_version": "agent_structure_resolution_task.v1",
                    "task_id": f"resolve_structure:{task_key}",
                    "task_type": "resolve_literature_structure",
                    "label": text,
                    "source_ref": source_ref,
                    "source_title": source_title,
                    "artifact_ref": artifact_ref,
                    "reason": str(gap.get("reason") or "visual_structure_not_confidently_convertible_to_smiles"),
                    "source_locator": str(gap.get("source_locator") or ""),
                    "visible_conditions": gap.get("visible_conditions"),
                    "status": "open",
                    "recommended_resolution_channels": [
                        "supplementary_information",
                        "publisher_full_text",
                        "compound_name_to_structure",
                        "source_detail_followup",
                    ],
                    "no_solved_claim": True,
                }
            )
    return tasks


def _safe_task_key(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    return "_".join(part for part in safe.split("_") if part)[:120] or "structure"


def _structure_resolution_attempt_summary(payload: dict[str, Any], *, artifact_ref: str) -> dict[str, Any]:
    task_id = str(payload.get("task_id") or "").strip()
    label = str(payload.get("label") or "").strip()
    source_ref = str(payload.get("source_ref") or "").strip()
    source_title = str(payload.get("source_title") or "").strip()
    status = str(payload.get("status") or ("resolved" if payload.get("accepted") else "unresolved")).strip()
    seed = str(payload.get("attempt_id") or task_id or label or artifact_ref or "structure_resolution_attempt")
    attempt_id = f"structure_resolution_attempt:{_safe_task_key(seed)}"
    resolved_count = len([row for row in payload.get("resolved_structures") or [] if isinstance(row, dict)])
    unresolved_count = len([row for row in payload.get("unresolved_tasks") or [] if isinstance(row, dict)])
    return {
        "schema_version": "agent_structure_resolution_attempt_summary.v1",
        "attempt_id": attempt_id,
        "task_id": task_id,
        "label": label,
        "source_ref": source_ref,
        "source_title": source_title,
        "accepted": bool(payload.get("accepted")),
        "status": status,
        "resolved_structure_count": resolved_count,
        "unresolved_task_count": unresolved_count,
        "artifact_ref": str(artifact_ref or ""),
        "reasons": [str(item) for item in payload.get("reasons") or [] if str(item or "").strip()],
        "no_solved_claim": True,
    }


def _update_structure_resolution_task_statuses(
    evidence: dict[str, Any],
    *,
    resolved: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> None:
    resolved_by_task: dict[str, list[str]] = {}
    for row in resolved:
        task_id = str(row.get("task_id") or "").strip()
        structure_id = str(row.get("structure_id") or "").strip()
        if task_id and structure_id:
            resolved_by_task.setdefault(task_id, []).append(structure_id)
    unresolved_by_task = {
        str(row.get("task_id") or "").strip(): dict(row)
        for row in unresolved
        if isinstance(row, dict) and str(row.get("task_id") or "").strip()
    }
    updated: list[dict[str, Any]] = []
    for raw in evidence.get("structure_resolution_tasks") or []:
        if not isinstance(raw, dict):
            updated.append(raw)
            continue
        task = dict(raw)
        task_id = str(task.get("task_id") or "").strip()
        if task_id in resolved_by_task:
            prior = [str(item) for item in task.get("resolved_structure_ids") or [] if str(item or "").strip()]
            task["resolved_structure_ids"] = _dedupe_strings([*prior, *resolved_by_task[task_id]])
            task["status"] = "resolved"
            task["last_resolution_status"] = "resolved"
            task["resolution_attempt_count"] = int(task.get("resolution_attempt_count") or 0) + 1
        elif task_id in unresolved_by_task:
            unresolved_row = dict(unresolved_by_task.get(task_id) or {})
            task["status"] = "open"
            task["last_resolution_status"] = "unresolved"
            task["resolution_attempt_count"] = int(task.get("resolution_attempt_count") or 0) + 1
            reasons = [str(item) for item in task.get("last_resolution_reasons") or [] if str(item or "").strip()]
            reason = str(unresolved_row.get("reason") or "").strip()
            task["last_resolution_reasons"] = _dedupe_strings([*reasons, reason])
        updated.append(task)
    evidence["structure_resolution_tasks"] = updated


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


def _pdf_structure_summary(payload: dict[str, Any], *, artifact_ref: str) -> dict[str, Any]:
    summary = dict(payload.get("summary") or {})
    if not summary:
        summary = {
            "rendered_page_count": len(payload.get("rendered_pages") or []),
            "indexed_image_count": len(payload.get("indexed_images") or []),
            "scheme_crop_count": len(payload.get("scheme_crops") or []),
            "compound_text_snippet_count": len(payload.get("compound_text_snippets") or []),
        }
    evidence_id = str(payload.get("evidence_id") or payload.get("source_ref") or payload.get("source_pdf_path") or "pdf_structure_evidence")
    return {
        "schema_version": "agent_pdf_structure_evidence_summary.v1",
        "evidence_id": evidence_id,
        "accepted": bool(payload.get("accepted", True)),
        "source_ref": str(payload.get("source_ref") or ""),
        "source_title": str(payload.get("source_title") or ""),
        "source_pdf_path": str(payload.get("source_pdf_path") or payload.get("pdf_path") or ""),
        "artifact_ref": artifact_ref,
        "summary": summary,
        "focus": _compact_pdf_focus(payload),
        "reasons": [str(item) for item in payload.get("reasons") or []],
        "no_solved_claim": True,
    }


def _compact_pdf_focus(payload: dict[str, Any]) -> dict[str, Any]:
    audit = dict(payload.get("focus_audit") or {})
    relevance = [
        dict(row)
        for row in payload.get("page_relevance") or []
        if isinstance(row, dict) and _nonnegative_int(row.get("score"), 0) > 0
    ]
    return {
        "schema_version": "agent_pdf_page_focus_summary.v1",
        "focus_terms": [
            str(item)[:96]
            for item in payload.get("focus_terms") or []
            if str(item or "").strip()
        ][:24],
        "focus_page_numbers": [
            _nonnegative_int(item, 0)
            for item in payload.get("focus_page_numbers") or []
            if _nonnegative_int(item, 0) > 0
        ][:16],
        "page_relevance": [
            {
                "page_number": _nonnegative_int(row.get("page_number"), 0),
                "score": _nonnegative_int(row.get("score"), 0),
                "matched_terms": [
                    str(match.get("term") or "")[:96]
                    for match in row.get("matched_terms") or []
                    if isinstance(match, dict) and str(match.get("term") or "").strip()
                ][:8],
            }
            for row in relevance[:16]
        ],
        "selection_strategy": str(audit.get("selection_strategy") or "")[:80],
        "algorithm_version": str(audit.get("algorithm_version") or "")[:80],
        "relevance_available": audit.get("relevance_available") is True,
        "text_page_count": _nonnegative_int(audit.get("text_page_count"), 0),
        "scan_truncated": audit.get("scan_truncated") is True,
        "no_ocr_or_relevance_fabrication": audit.get("no_ocr_or_relevance_fabrication") is True,
        "no_solved_claim": True,
    }


def _extend_unique(target: dict[str, Any], list_name: str, rows: list[Any], *, unique_key: str | None) -> None:
    existing = list(target.get(list_name) or [])
    seen = {_unique_key(row, unique_key) for row in existing}
    for row in rows:
        value = row if isinstance(row, dict) else str(row)
        marker = _unique_key(value, unique_key)
        if marker in seen:
            continue
        seen.add(marker)
        existing.append(value)
    target[list_name] = existing


def _target_side_handles(target_side: dict[str, Any]) -> set[str]:
    target = dict(target_side.get("target") or {})
    handles = {str(item) for item in target.get("handles") or [] if str(item or "").strip()}
    handles.update(
        str(row.get("target_handle") or "")
        for row in target_side.get("hypotheses") or []
        if isinstance(row, dict) and str(row.get("target_handle") or "").strip()
    )
    return handles


def _selected_objective_types(summary: dict[str, Any]) -> set[str]:
    return {
        str(row.get("objective_type") or "")
        for row in summary.get("selected_objectives") or []
        if isinstance(row, dict) and str(row.get("objective_type") or "").strip()
    }


def _analogical_ranking_hypothesis_ids(ranking: dict[str, Any]) -> set[str]:
    rows = [
        *[row for row in ranking.get("ranked_hypotheses") or [] if isinstance(row, dict)],
        *[row for row in ranking.get("selected_hypotheses") or [] if isinstance(row, dict)],
    ]
    return {str(row.get("hypothesis_id") or "") for row in rows if str(row.get("hypothesis_id") or "").strip()}


def _target_side_hypothesis_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return str(row.get("hypothesis_id") or "").startswith("target_side_")


def _target_derived_bridge_task(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    task_id = str(row.get("task_id") or "")
    source_id = str(row.get("source_hypothesis_id") or "")
    anchor_id = str(row.get("anchor_id") or "")
    return bool(
        task_id.startswith("bridge:")
        or source_id.startswith("target_side_")
        or anchor_id.startswith("route_objective_anchor:")
    )


def _route_objective_anchor_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return str(row.get("anchor_id") or "").startswith("route_objective_anchor:")


def _remove_stale_evidence_refs(board: dict[str, Any], stale_ids: set[str]) -> None:
    if not stale_ids:
        return
    for list_name in (
        "retrosynthetic_proposals",
        "recursive_hypothesis_tasks",
        "proposal_failure_feedback",
        "reaction_idea_cards",
    ):
        cleaned: list[Any] = []
        for row in board.get(list_name) or []:
            if not isinstance(row, dict):
                cleaned.append(row)
                continue
            updated = dict(row)
            refs = [str(item) for item in updated.get("evidence_refs") or [] if str(item or "").strip()]
            if refs:
                updated["evidence_refs"] = [ref for ref in refs if ref not in stale_ids]
            cleaned.append(updated)
        board[list_name] = cleaned


def _apply_route_scope_to_belief(board: dict[str, Any], route_scope: dict[str, Any]) -> None:
    if not route_scope:
        return
    belief = dict(board.get("current_belief") or {})
    belief["route_scope"] = route_scope
    constraints = dict(belief.get("constraints") or {})
    for key, scope_key in (
        ("de_novo_core_construction_deprioritized", "de_novo_core_construction_deprioritized"),
        ("small_molecule_stock_closure_deprioritized", "small_molecule_stock_closure_deprioritized"),
        ("objective_evidence_validation_required", "objective_evidence_validation_required"),
    ):
        constraints[key] = bool(route_scope.get(scope_key))
    belief["constraints"] = constraints
    board["current_belief"] = belief


def _merge_source_candidate_rows(existing_rows: list[Any], incoming_rows: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index: dict[str, int] = {}

    def add(row: dict[str, Any], *, incoming: bool) -> None:
        key = _source_candidate_merge_key(row)
        if key and key in index:
            pos = index[key]
            rows[pos] = _merge_source_candidate_row(rows[pos], row, prefer_incoming=incoming)
            return
        if key:
            index[key] = len(rows)
        rows.append(dict(row))

    for raw in existing_rows:
        if isinstance(raw, dict):
            add(dict(raw), incoming=False)
    for raw in incoming_rows:
        if isinstance(raw, dict):
            add(dict(raw), incoming=True)
    return rows


def _source_candidate_merge_key(row: dict[str, Any]) -> str:
    logical_document = source_document_identity(row)
    if logical_document:
        return logical_document
    doi = _normalize_hint_doi(str(row.get("doi") or ""))
    if doi:
        return f"doi:{doi.lower()}"
    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref.startswith("doi:"):
        doi = _normalize_hint_doi(source_ref)
        if doi:
            return f"doi:{doi.lower()}"
    pii = str(row.get("pii") or "").strip().lower()
    if pii:
        return f"pii:{pii}"
    if source_ref.startswith("pii:") and source_ref[4:].strip():
        return f"pii:{source_ref[4:].strip().lower()}"
    url = str(row.get("url") or "").strip().lower()
    if url:
        return f"url:{url}"
    if source_ref:
        return f"ref:{source_ref}"
    return ""


def _merge_source_candidate_row(existing: dict[str, Any], incoming: dict[str, Any], *, prefer_incoming: bool) -> dict[str, Any]:
    merged = dict(existing)
    for key in (
        "doi",
        "pii",
        "url",
        "source_ref",
        "title",
        "route_sequence_hint",
        "document_id",
        "content_scope",
        "source_role",
    ):
        value = incoming.get(key)
        if str(value or "").strip() and (prefer_incoming or not str(merged.get(key) or "").strip()):
            merged[key] = value
    for key in ("local_pdf", "source_type", "source_discovery_mode", "access_status"):
        value = incoming.get(key)
        if str(value or "").strip():
            merged[key] = value
    for key in ("local_pdf_index", "local_pdf_match"):
        value = incoming.get(key)
        if isinstance(value, dict) and value:
            merged[key] = dict(value)
    for key in ("expected_scheme_or_compound_labels", "extraction_task_recommendations"):
        merged[key] = _dedupe_strings(
            [
                *[str(item) for item in merged.get(key) or [] if str(item or "").strip()],
                *[str(item) for item in incoming.get(key) or [] if str(item or "").strip()],
            ]
        )
    if incoming.get("placeholder_only") is not None:
        merged["placeholder_only"] = bool(incoming.get("placeholder_only"))
    if incoming.get("no_solved_claim") is not None:
        merged["no_solved_claim"] = bool(incoming.get("no_solved_claim"))
    rationale = _dedupe_strings(
        [
            str(existing.get("relevance_rationale") or ""),
            str(incoming.get("relevance_rationale") or ""),
        ]
    )
    if rationale:
        merged["relevance_rationale"] = " | ".join(rationale)
    return merged


def _unique_key(row: Any, key: str | None) -> str:
    if isinstance(row, dict) and key:
        value = row.get(key)
        if value:
            return str(value)
    if isinstance(row, dict):
        for fallback in ("task_id", "hypothesis_id", "reason", "source_ref", "canonical_smiles", "schema_version"):
            if row.get(fallback):
                return f"{fallback}:{row[fallback]}"
    return json.dumps(row, sort_keys=True, default=str)


def _source_confidence_score(row: dict[str, Any], exact_rows: list[dict[str, Any]]) -> int:
    refs = {str(item) for item in row.get("related_source_evidence") or row.get("evidence_refs") or []}
    if not refs:
        return 0
    exact_refs = {str(ref) for exact in exact_rows for ref in exact.get("evidence_refs") or []}
    return 20 if refs & exact_refs else 10


def _blackboard_failure_reasons(blackboard: dict[str, Any]) -> set[str]:
    return {str(row.get("reason") or "") for row in blackboard.get("route_failures") or [] if isinstance(row, dict)}


def _candidate_has_real_source(row: dict[str, Any]) -> bool:
    if bool(row.get("placeholder_only")):
        return False
    if str(row.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    return bool(str(row.get("doi") or row.get("url") or row.get("local_pdf") or "").strip())


def _dedupe_strings(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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
