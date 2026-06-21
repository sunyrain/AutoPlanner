"""Typed blackboard state for policy-driven agentic controller runs."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from cascade_planner.harness.agent_action_planner import build_guided_chemenzy_payload_from_blackboard
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.harness.schemas import write_json


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
            "pdf_structure_evidence": [],
            "visual_chains": [],
            "exact_rows": [],
            "terminal_candidates": [],
            "source_refs": [],
            "confidence": "none",
        },
        "analogical_hypotheses": [],
        "analogical_hypothesis_ranking": {},
        "bridge_tasks": [],
        "terminal_blacklist": [],
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
        },
        "current_belief": {
            "schema_version": "agent_current_belief.v1",
            "promising_directions": [],
            "blocked_directions": [],
            "constraints": {
                "target_core_retention_required": True,
                "max_unexplained_heavy_atom_jump": 15,
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
    artifact_name = _artifact_name(action, action_result)
    artifact_ref = ""
    if artifact_name:
        artifact_path = Path(run_dir) / f"{artifact_name}.json"
        write_json(artifact_path, _summary_safe(action_result))
        artifact_ref = str(artifact_path)
        board.setdefault("artifact_refs", {})[artifact_name] = artifact_ref

    useful = _normalize_action_output(board, action_type=action_type, result=action_result, artifact_ref=artifact_ref)
    history = {
        "schema_version": "agent_action_history_record.v1",
        "round_index": int(round_index),
        "action_id": str(action.get("action_id") or ""),
        "action_type": action_type,
        "status": "accepted" if action_result.get("accepted", True) else "rejected",
        "artifact_ref": artifact_ref,
        "useful_artifact": bool(useful),
        "stale": not bool(useful),
        "action_signature": _action_signature(action),
        "reasons": [str(item) for item in action_result.get("reasons") or []],
    }
    board.setdefault("action_history", []).append(history)
    return board


def update_budget_for_action(blackboard: dict[str, Any], action_type: str) -> dict[str, Any]:
    board = deepcopy(blackboard)
    budget = dict(board.get("budget_state") or {})
    if action_type == "search_literature":
        budget["scout_calls"] = int(budget.get("scout_calls") or 0) + 1
    if action_type in {"extract_pdf_literature_structures", "extract_visual_literature_chain"}:
        budget["visual_calls"] = int(budget.get("visual_calls") or 0) + 1
    if action_type == "run_guided_chemenzy":
        budget["chemenzy_runs"] = int(budget.get("chemenzy_runs") or 0) + 1
    if action_type == "expand_child_target":
        budget["child_target_runs"] = int(budget.get("child_target_runs") or 0) + 1
    board["budget_state"] = budget
    return board


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
    if action_type == "generate_disconnection_hypotheses":
        artifact = payload
        board["target_side_disconnection_hypotheses"] = _drop_large_fields(artifact)
        _extend_unique(board, "bridge_tasks", artifact.get("bridge_tasks") or [], unique_key="task_id")
        _extend_unique(board, "analogical_hypotheses", artifact.get("hypotheses") or [], unique_key="hypothesis_id")
        return bool(artifact.get("hypotheses"))

    if action_type == "build_failure_critic_report":
        artifact = payload
        _extend_unique(board, "route_failures", artifact.get("route_failures") or [], unique_key="reason")
        _extend_unique(board, "bridge_tasks", artifact.get("bridge_tasks") or [], unique_key="task_id")
        _extend_unique(board, "terminal_blacklist", artifact.get("terminal_blacklist") or [], unique_key="canonical_smiles")
        belief = dict(board.get("current_belief") or {})
        _extend_unique(belief, "blocked_directions", artifact.get("blocked_directions") or [], unique_key="direction")
        constraints = dict(belief.get("constraints") or {})
        constraints.update(dict(artifact.get("constraints") or {}))
        belief["constraints"] = constraints
        board["current_belief"] = belief
        return bool(artifact.get("accepted"))

    if action_type == "search_literature":
        evidence = dict(board.get("literature_evidence") or {})
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
        board["literature_evidence"] = evidence
        if payload.get("codex_worker_run_attempted"):
            budget = dict(board.get("budget_state") or {})
            budget["codex_research_runs"] = max(
                int(budget.get("codex_research_runs") or 0),
                int(payload.get("codex_research_runs") or 0),
            )
            board["budget_state"] = budget
        return bool(real_sources)

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
        board["literature_evidence"] = evidence
        return bool(chain_rows)

    if action_type == "compile_exact_literature_rows":
        evidence = dict(board.get("literature_evidence") or {})
        rows = _exact_rows_from_payload(payload)
        _extend_unique(evidence, "exact_rows", rows, unique_key="row_id")
        terminal = _literature_terminal_candidate_from_payload(payload)
        if terminal:
            _extend_unique(evidence, "terminal_candidates", [terminal], unique_key="canonical_smiles")
            _extend_unique(board, "bridge_tasks", [_literature_terminal_bridge_task(board, terminal)], unique_key="task_id")
        evidence["confidence"] = "exact_rows" if evidence.get("exact_rows") else evidence.get("confidence", "none")
        board["literature_evidence"] = evidence
        return bool(rows)

    if action_type == "rank_analogical_hypotheses":
        board["analogical_hypothesis_ranking"] = payload
        return bool(payload.get("selected_hypotheses"))

    if action_type == "run_guided_chemenzy":
        _update_from_guided_chemenzy(board, payload, artifact_ref)
        return bool(payload.get("raw_route_verifier") or payload.get("route_failure_feedback") or payload.get("accepted"))

    if action_type == "expand_child_target":
        belief = dict(board.get("current_belief") or {})
        belief["child_route_solved"] = bool(payload.get("accepted_subgoal_count") or payload.get("solved"))
        board["current_belief"] = belief
        board.setdefault("artifact_refs", {})["route_expansion_subgoal_search"] = artifact_ref
        return bool(payload.get("subgoals") or payload.get("accepted_subgoal_count"))

    if action_type == "stitch_parent_route":
        proof = dict(payload.get("parent_route_proof") or payload)
        board["parent_route_proof"] = proof
        return bool(proof.get("accepted") or proof.get("reasons"))

    if action_type == "stop_unresolved":
        belief = dict(board.get("current_belief") or {})
        stops = list(belief.get("stop_candidates") or [])
        stops.append({"schema_version": "agent_stop_candidate.v1", "reason": "stop_unresolved_action"})
        belief["stop_candidates"] = stops
        board["current_belief"] = belief
        return False
    return useful


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
        "evidence_refs": [str(item) for item in trace.get("evidence_refs") or row.get("evidence_refs") or []],
        "confidence": str(row.get("confidence") or "source_detail_exact"),
        "no_solved_claim": True,
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
    return {
        "schema_version": "agent_visual_chain_summary.v1",
        "chain_id": str(payload.get("chain_id") or payload.get("case_id") or artifact_ref or "visual_chain"),
        "accepted": bool(payload.get("accepted", True)),
        "artifact_ref": artifact_ref,
        "candidate_step_count": int(payload.get("candidate_step_count") or len(candidate.get("steps") or [])),
        "missing_expected_labels": [str(item) for item in missing_expected if str(item or "").strip()],
        "condition_gap_labels": [str(item) for item in condition_gap_labels if str(item or "").strip()],
        "gap_labels": _dedupe_strings(gap_labels),
        "warning_gap_labels": _dedupe_strings(warning_gap_labels),
        "extraction_gaps": _drop_large_fields(extraction_gaps),
        "reasons": [str(item) for item in payload.get("reasons") or []],
    }


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
        "source_pdf_path": str(payload.get("source_pdf_path") or ""),
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
