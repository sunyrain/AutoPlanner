"""Typed blackboard state for policy-driven agentic controller runs."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:  # RDKit is expected in normal AutoPlanner runs, but tests can mock around it.
    from rdkit import Chem
except Exception:  # pragma: no cover - exercised only in stripped environments.
    Chem = None  # type: ignore[assignment]

from cascade_planner.harness.agent_action_planner import build_guided_chemenzy_payload_from_blackboard
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.harness.analogical_reaction_templates import compact_template_application_summary
from cascade_planner.harness.recursive_hypothesis_tasks import (
    recursive_hypothesis_tasks_from_route_expansion,
)
from cascade_planner.harness.route_objectives import (
    build_broad_transform_templates_from_blackboard,
    classify_route_objectives,
    compile_route_objective_proof_bundle,
)
from cascade_planner.harness.schemas import write_json
from cascade_planner.agent.action_contracts import PLANNER_SOURCE_HINT_SCHEMA


AGENT_BLACKBOARD_SCHEMA = "agent_blackboard.v1"


def initialize_agent_blackboard(
    *,
    target_input: dict[str, Any],
    preflight: dict[str, Any],
    max_rounds: int = 3,
    budget_limits: dict[str, Any] | None = None,
    prior_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = dict(preflight.get("target_profile") or {})
    limits = dict(budget_limits or {})
    max_scout_calls = _positive_int(limits.get("max_scout_calls"), 3)
    max_visual_calls = _positive_int(limits.get("max_visual_calls"), 3)
    max_chemenzy_runs = _positive_int(
        limits.get("max_guided_chemenzy_runs") or limits.get("max_chemenzy_runs"),
        1,
    )
    max_child_target_runs = _positive_int(
        limits.get("max_route_expansion_subgoal_runs") or limits.get("max_child_target_runs"),
        2,
    )
    max_codex_research_runs = _positive_int(limits.get("max_codex_research_runs"), 1)
    max_template_applications_per_round = _positive_int(limits.get("max_template_applications_per_round"), 5)
    return {
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
        "route_proof_bundle": {},
        "semisynthesis_anchors": [],
        "recursive_hypothesis_tasks": [],
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
            "max_template_application_actions": _positive_int(limits.get("max_template_application_actions"), 3),
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
        },
        "artifact_refs": dict((prior_artifacts or {}).get("artifact_refs") or {}),
        "parent_route_proof": {},
    }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


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
    evidence["source_lifecycle"] = _build_source_lifecycle(evidence)
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
        row["stage_flags"]["pdf_rendered"] = bool(pdf.get("accepted", True)) or row["stage_flags"]["pdf_rendered"]
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
    identity_fields = {
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
    local_pdf = str(source.get("local_pdf") or source.get("source_pdf_path") or source.get("pdf_path") or "").strip().lower()
    if local_pdf:
        return f"pdf:{local_pdf}"
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
    }
    board.setdefault("action_history", []).append(history)
    return board


def update_budget_for_action(
    blackboard: dict[str, Any],
    action_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    board = deepcopy(blackboard)
    budget = dict(board.get("budget_state") or {})
    action_payload = dict(payload or {})
    if action_type == "search_literature":
        budget["scout_calls"] = int(budget.get("scout_calls") or 0) + 1
    if action_type == "extract_visual_literature_chain":
        budget["visual_calls"] = int(budget.get("visual_calls") or 0) + 1
    if action_type == "resolve_literature_structure_task" and bool(action_payload.get("run_visual", True)):
        budget["visual_calls"] = int(budget.get("visual_calls") or 0) + 1
    if action_type == "run_guided_chemenzy":
        budget["chemenzy_runs"] = int(budget.get("chemenzy_runs") or 0) + 1
    if action_type == "expand_child_target":
        budget["child_target_runs"] = int(budget.get("child_target_runs") or 0) + 1
    if action_type in {"apply_analogical_template_to_target", "validate_template_application"}:
        budget["template_application_actions"] = int(budget.get("template_application_actions") or 0) + 1
    board["budget_state"] = budget
    return board


def _blackboard_count_summary(blackboard: dict[str, Any]) -> dict[str, int]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    belief = dict(blackboard.get("current_belief") or {})
    proof = dict(blackboard.get("parent_route_proof") or {})
    return {
        "source_candidates": len(evidence.get("source_candidates") or []),
        "planner_source_hints": len(evidence.get("planner_source_hints") or []),
        "source_lifecycle": len(evidence.get("source_lifecycle") or []),
        "source_refs": len(evidence.get("source_refs") or []),
        "local_pdf_proxy_requests": len(evidence.get("local_pdf_proxy_requests") or []),
        "pdf_structure_evidence": len(evidence.get("pdf_structure_evidence") or []),
        "visual_chains": len(evidence.get("visual_chains") or []),
        "exact_rows": len(evidence.get("exact_rows") or []),
        "target_relevant_exact_rows": _target_relevant_exact_row_count(evidence.get("exact_rows") or []),
        "exact_chain_audits": len(evidence.get("exact_chain_audits") or []),
        "terminal_candidates": len(evidence.get("terminal_candidates") or []),
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
        "semisynthesis_anchors": len(blackboard.get("semisynthesis_anchors") or []),
        "recursive_hypothesis_tasks": len(blackboard.get("recursive_hypothesis_tasks") or []),
        "bridge_tasks": len(blackboard.get("bridge_tasks") or []),
        "terminal_blacklist": len(blackboard.get("terminal_blacklist") or []),
        "blocked_directions": len(belief.get("blocked_directions") or []),
        "next_action_bias": len(belief.get("next_action_bias") or []),
        "stop_candidates": len(belief.get("stop_candidates") or []),
        "artifact_refs": len(blackboard.get("artifact_refs") or {}),
        "parent_route_proof_present": 1 if proof else 0,
        "parent_route_proof_accepted": 1 if proof.get("accepted") else 0,
    }


def _blackboard_count_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {
        key: int(after.get(key) or 0) - int(before.get(key) or 0)
        for key in keys
        if int(after.get(key) or 0) - int(before.get(key) or 0)
    }


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
        existing_source_keys = {
            str(row.get("source_ref") or row.get("doi") or row.get("pii") or row.get("url") or "").strip()
            for row in evidence.get("source_candidates") or []
            if isinstance(row, dict)
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
        _extend_unique(evidence, "source_candidates", payload.get("source_candidates") or [], unique_key="source_ref")
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
            if str(row.get("source_ref") or row.get("doi") or row.get("pii") or row.get("url") or "").strip()
            not in existing_source_keys
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
        return bool(new_real_sources or new_source_refs or new_proxy_requests)

    if action_type == "extract_pdf_literature_structures":
        evidence = dict(board.get("literature_evidence") or {})
        summary = _pdf_structure_summary(payload, artifact_ref=artifact_ref)
        _extend_unique(evidence, "pdf_structure_evidence", [summary], unique_key="evidence_id")
        evidence["confidence"] = "pdf_rendered" if summary.get("accepted") else evidence.get("confidence", "none")
        board["literature_evidence"] = evidence
        counts = dict(summary.get("summary") or {})
        return bool(counts.get("rendered_page_count") or counts.get("indexed_image_count") or counts.get("scheme_crop_count"))

    if action_type == "extract_visual_literature_chain":
        evidence = dict(board.get("literature_evidence") or {})
        chain_rows = payload.get("chains") or payload.get("visual_chains") or []
        if not chain_rows and payload:
            chain_rows = [_compact_artifact(payload, artifact_ref=artifact_ref)]
        _extend_unique(evidence, "visual_chains", chain_rows, unique_key="chain_id")
        structure_tasks: list[dict[str, Any]] = []
        for row in chain_rows:
            if isinstance(row, dict):
                structure_tasks.extend(_structure_resolution_tasks_from_visual_chain(row))
        _extend_unique(evidence, "structure_resolution_tasks", structure_tasks, unique_key="task_id")
        board["literature_evidence"] = evidence
        return bool(chain_rows)

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
        new_resolved = [
            row
            for row in resolved
            if str(row.get("structure_id") or "") not in existing_structure_ids
        ]
        new_attempt = bool(attempt and str(attempt.get("attempt_id") or "") not in existing_attempt_ids)
        return bool(new_resolved or new_attempt)

    if action_type == "compile_exact_literature_rows":
        evidence = dict(board.get("literature_evidence") or {})
        existing_row_ids = {
            str(row.get("row_id") or "")
            for row in evidence.get("exact_rows") or []
            if isinstance(row, dict)
        }
        rows = _annotate_exact_rows_with_target_relevance(_exact_rows_from_payload(payload), board)
        _extend_unique(evidence, "exact_rows", rows, unique_key="row_id")
        evidence["exact_row_target_relevance_summary"] = _exact_row_target_relevance_summary(evidence.get("exact_rows") or [])
        audit = _exact_chain_audit_summary(payload, artifact_ref=artifact_ref)
        if audit:
            _extend_unique(evidence, "exact_chain_audits", [audit], unique_key="audit_id")
        terminal = _literature_terminal_candidate_from_payload(payload)
        if terminal:
            _extend_unique(evidence, "terminal_candidates", [terminal], unique_key="canonical_smiles")
            _extend_unique(board, "bridge_tasks", [_literature_terminal_bridge_task(board, terminal)], unique_key="task_id")
        evidence["confidence"] = "exact_rows" if evidence.get("exact_rows") else evidence.get("confidence", "none")
        board["literature_evidence"] = evidence
        new_row_count = sum(1 for row in rows if str(row.get("row_id") or "") not in existing_row_ids)
        return bool(new_row_count or terminal)

    if action_type == "rank_analogical_hypotheses":
        board["analogical_hypothesis_ranking"] = payload
        return bool(payload.get("selected_hypotheses"))

    if action_type == "extract_analogical_reaction_templates":
        _extend_unique(board, "analogical_templates", payload.get("templates") or [], unique_key="template_id")
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
        _extend_unique(board, "template_applications", summaries, unique_key="application_id")
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
        return bool(payload.get("raw_route_verifier") or payload.get("route_failure_feedback") or payload.get("accepted"))

    if action_type == "expand_child_target":
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
        belief["child_route_solved"] = bool(payload.get("accepted_subgoal_count") or payload.get("solved"))
        board["current_belief"] = belief
        board.setdefault("artifact_refs", {})["route_expansion_subgoal_search"] = artifact_ref
        return bool(payload.get("subgoals") or payload.get("accepted_subgoal_count") or new_task_ids)

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


def _update_from_guided_chemenzy(board: dict[str, Any], payload: dict[str, Any], artifact_ref: str) -> None:
    verifier = dict(payload.get("raw_route_verifier") or {})
    if verifier:
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
    runtime = dict(payload.get("literature_template_plugin_runtime") or {})
    if runtime:
        _extend_unique(board, "plugin_runtime_diagnostics", [runtime], unique_key="schema_version")
    feedback = dict(payload.get("route_failure_feedback") or {})
    if feedback and isinstance(feedback.get("path"), str):
        board.setdefault("artifact_refs", {})["route_failure_feedback"] = str(feedback.get("path") or "")


def _exact_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = payload.get("exact_rows") or payload.get("one_step_rows")
    if isinstance(explicit, list):
        return [_row_summary(row, idx) for idx, row in enumerate(explicit, start=1) if isinstance(row, dict)]
    compiled = dict(payload.get("compiled_downstream") or payload)
    plugin = dict(compiled.get("literature_template_plugin") or {})
    rows = plugin.get("one_step_rows") or (plugin.get("plugin_flags") or {}).get("one_step_rows") or []
    return [_row_summary(row, idx) for idx, row in enumerate(rows, start=1) if isinstance(row, dict)]


def _literature_terminal_candidate_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    chain = dict(payload.get("chain_audit") or {})
    if not chain and isinstance(payload.get("result"), dict):
        chain = dict((payload.get("result") or {}).get("chain_audit") or {})
    if not chain:
        return {}
    smiles = str(
        chain.get("terminal_smiles")
        or chain.get("observed_terminal_smiles")
        or _last_chain_main_reactant(chain.get("chain") or chain.get("steps") or [])
        or ""
    ).strip()
    canonical = str(chain.get("terminal_canonical_smiles") or canonical_smiles(smiles) or "").strip()
    if not smiles or not canonical:
        return {}
    repair = dict(chain.get("terminal_stereo_repair") or {})
    return {
        "schema_version": "agent_literature_terminal_candidate.v1",
        "terminal_id": f"source_detail_terminal:{canonical}",
        "name": str(chain.get("terminal_name") or repair.get("name") or "source detail literature terminal"),
        "smiles": smiles,
        "canonical_smiles": canonical,
        "source_ref": _chain_source_ref(chain),
        "source": "source_detail_chain_route",
        "step_count": int(chain.get("step_count") or len(chain.get("chain") or [])),
        "terminal_reached": bool(chain.get("terminal_reached")),
        "terminal_requested": bool(chain.get("terminal_requested")),
        "stereo_repair": repair,
        "exact_target_override": True,
        "target_equivalence_audit_required": True,
        "no_solved_claim": True,
    }


def _exact_chain_audit_summary(payload: dict[str, Any], *, artifact_ref: str) -> dict[str, Any]:
    chain = dict(payload.get("chain_audit") or {})
    if not chain and isinstance(payload.get("result"), dict):
        chain = dict((payload.get("result") or {}).get("chain_audit") or {})
    if not chain:
        return {}
    summary = dict(chain.get("summary") or {})
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
        "artifact_ref": str(artifact_ref or ""),
    }


def _literature_terminal_bridge_task(board: dict[str, Any], terminal: dict[str, Any]) -> dict[str, Any]:
    target = dict(board.get("target_profile") or {})
    canonical = str(terminal.get("canonical_smiles") or "")
    return {
        "schema_version": "agent_bridge_task.v1",
        "task_id": f"literature_terminal_child:{canonical}",
        "task_type": "upstream_terminal_synthesis",
        "target_name": str(target.get("target_name") or ""),
        "target_handle": "source_detail_literature_terminal",
        "required_bridge": "Find upstream synthesis for the exact terminal of the accepted source-detail literature chain.",
        "required_verification": ["child_target_route_verifier", "parent_bridge_connectivity", "exact_terminal_identity"],
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


def _row_summary(row: dict[str, Any], idx: int) -> dict[str, Any]:
    trace = dict(row.get("literature_template_trace") or {})
    template = row.get("template") if isinstance(row.get("template"), dict) else row.get("templates")
    if isinstance(template, dict):
        trace = {**dict(template.get("literature_template_trace") or {}), **trace}
    return {
        "schema_version": "agent_exact_literature_row_summary.v1",
        "row_id": str(row.get("row_id") or trace.get("source_template_id") or f"exact_row_{idx}"),
        "source_ref": str(trace.get("source_ref") or row.get("source_ref") or ""),
        "product_smiles": str(trace.get("product_smiles") or row.get("product_smiles") or ""),
        "product_label": str(trace.get("product_label") or row.get("product_label") or row.get("product_name") or ""),
        "source_locator": str(trace.get("source_locator") or row.get("source_locator") or ""),
        "evidence_refs": [str(item) for item in trace.get("evidence_refs") or row.get("evidence_refs") or []],
        "confidence": str(row.get("confidence") or "source_detail_exact"),
        "no_solved_claim": True,
    }


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
    steps = candidate.get("steps") or candidate.get("candidate_steps") or []
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
        "reasons": [str(item) for item in payload.get("reasons") or []],
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
    return json.dumps(
        {"action_type": action.get("action_type"), "payload": action.get("payload") or {}},
        sort_keys=True,
        default=str,
    )
