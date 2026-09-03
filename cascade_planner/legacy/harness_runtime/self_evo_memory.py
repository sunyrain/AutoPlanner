"""Compile selfEVO replay output into reusable, non-production memory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cascade_planner.agent.evolution_manager import (
    evolution_candidate_from_dict,
    validate_evolution_candidate,
)


SELF_EVO_MEMORY_SCHEMA = "self_evo_reusable_memory.v1"


def compile_self_evo_memory(
    replay_report: dict[str, Any],
    *,
    compiled_downstream: dict[str, Any] | None = None,
    case_id: str = "",
) -> dict[str, Any]:
    """Build a future-run memory artifact from replay-accepted selfEVO state."""
    report = dict(replay_report or {})
    compiled = dict(compiled_downstream or {})
    reasons: list[str] = []
    rejected: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    template_cards: list[dict[str, Any]] = []
    one_step_rows: list[dict[str, Any]] = []
    route_segments: list[dict[str, Any]] = []
    route_expansion_tasks: list[dict[str, Any]] = []
    extraction_tasks: list[dict[str, Any]] = []
    query_hints: list[dict[str, Any]] = []

    if report.get("skipped"):
        reasons.append("self_evo_replay_skipped")
    if not report.get("accepted"):
        reasons.extend(str(item) for item in report.get("reasons") or ["self_evo_replay_not_accepted"])

    layers = dict(((report.get("kb") or {}).get("layers") or {}))
    source_layer = "production" if dict(layers.get("production") or {}) and not report.get("production_write_blocked") else "staging"
    candidates = dict(layers.get(source_layer) or {})
    if not candidates and source_layer == "staging":
        candidates = dict(layers.get("shadow") or {})
        source_layer = "shadow" if candidates else source_layer
    if not candidates:
        reasons.append("no_reusable_self_evo_candidates")

    for candidate_id, raw in candidates.items():
        candidate = evolution_candidate_from_dict(dict(raw or {}))
        validation = validate_evolution_candidate(candidate)
        if not validation.get("accepted"):
            rejected.append({"candidate_id": candidate_id, "reasons": validation.get("reasons") or []})
            reasons.extend(str(item) for item in validation.get("reasons") or [])
            continue
        candidate_summaries.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "validation_status": candidate.validation_status,
                "source_layer": source_layer,
                "evidence_refs": list(candidate.evidence_refs),
            }
        )
        assets = _assets_from_candidate(candidate.to_dict())
        template_cards.extend(assets["template_cards"])
        one_step_rows.extend(assets["one_step_rows"])
        route_segments.extend(assets["route_segments"])
        route_expansion_tasks.extend(assets["route_expansion_tasks"])
        extraction_tasks.extend(assets["extraction_tasks"])
        query_hints.extend(assets["query_hints"])

    if candidate_summaries:
        compiled_assets = _assets_from_compiled_downstream(compiled)
        template_cards.extend(compiled_assets["template_cards"])
        one_step_rows.extend(compiled_assets["one_step_rows"])
        route_expansion_tasks.extend(compiled_assets["route_expansion_tasks"])
        extraction_tasks.extend(compiled_assets["extraction_tasks"])
        query_hints.extend(compiled_assets["query_hints"])

    template_cards = _dedupe_objects(_reject_raw_assets(template_cards, "template_card", rejected, reasons))
    one_step_rows = _dedupe_objects(_reject_raw_assets(one_step_rows, "one_step_row", rejected, reasons))
    route_segments = _dedupe_objects(_reject_raw_assets(route_segments, "route_segment", rejected, reasons))
    route_expansion_tasks = _dedupe_objects(_reject_raw_assets(route_expansion_tasks, "route_expansion_task", rejected, reasons))
    extraction_tasks = _dedupe_objects(_reject_raw_assets(extraction_tasks, "extraction_task", rejected, reasons))
    query_hints = _dedupe_objects(query_hints)

    accepted = bool(candidate_summaries) and bool(
        template_cards or one_step_rows or route_segments or route_expansion_tasks or extraction_tasks or query_hints
    ) and not any(reason == "raw_reaction_injection" for reason in reasons)
    if not accepted and not reasons:
        reasons.append("no_reusable_self_evo_assets")
    return {
        "schema_version": SELF_EVO_MEMORY_SCHEMA,
        "accepted": accepted,
        "case_id": str(case_id or report.get("case_id") or ""),
        "source_replay_schema": str(report.get("schema_version") or ""),
        "source_layer": source_layer,
        "target_run": bool(report.get("target_run", True)),
        "production_write_blocked": bool(report.get("production_write_blocked", True)),
        "production_promoted_count": int(report.get("production_promoted_count") or 0),
        "future_use_policy": {
            "allowed_use": "query_seed|template_plugin_candidate|guided_rerun_hint|executable_template_extraction_task",
            "not_route_evidence_until_current_target_relation_checked": True,
            "requires_replay_gate_before_production": True,
            "no_solved_claim": True,
        },
        "candidate_summaries": candidate_summaries,
        "reusable_template_cards": template_cards,
        "reusable_one_step_rows": one_step_rows,
        "reusable_route_segments": route_segments,
        "reusable_route_expansion_tasks": route_expansion_tasks,
        "reusable_executable_template_extraction_tasks": extraction_tasks,
        "query_hints": query_hints,
        "rejected_items": rejected,
        "reasons": sorted(set(str(item) for item in reasons)),
    }


def write_self_evo_memory(memory: dict[str, Any], *, output_dir: str | Path) -> str:
    path = Path(output_dir) / "self_evo_memory.json"
    path.write_text(json.dumps(memory, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return str(path)


def _assets_from_candidate(candidate: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    payload = dict(candidate.get("payload") or {})
    out = _empty_assets()
    candidate_type = str(candidate.get("candidate_type") or "")
    if candidate_type == "LiteratureRouteSegmentCard" or payload.get("schema_version") == "literature_route_segment_card.v1":
        out["route_segments"].append(payload)
    if candidate_type == "SegmentStepCandidate" or payload.get("schema_version") == "segment_step_candidate.v1":
        out["query_hints"].append(_query_hint(candidate, hint_type="segment_step"))
    for key in ("template_card", "literature_template_card"):
        if isinstance(payload.get(key), dict):
            out["template_cards"].append(dict(payload[key]))
    for key in ("template_cards", "literature_template_cards"):
        out["template_cards"].extend(dict(item) for item in payload.get(key) or [] if isinstance(item, dict))
    for key in ("one_step_row", "executable_one_step_row"):
        if isinstance(payload.get(key), dict):
            out["one_step_rows"].append(dict(payload[key]))
    for key in ("one_step_rows", "executable_one_step_rows"):
        out["one_step_rows"].extend(dict(item) for item in payload.get(key) or [] if isinstance(item, dict))
    for key in ("route_expansion_task", "expansion_task"):
        if isinstance(payload.get(key), dict):
            out["route_expansion_tasks"].append(dict(payload[key]))
    for key in ("route_expansion_tasks", "expansion_tasks"):
        out["route_expansion_tasks"].extend(dict(item) for item in payload.get(key) or [] if isinstance(item, dict))
    for key in ("executable_template_extraction_task", "extraction_task"):
        if isinstance(payload.get(key), dict):
            out["extraction_tasks"].append(dict(payload[key]))
    for key in ("executable_template_extraction_tasks", "extraction_tasks"):
        out["extraction_tasks"].extend(dict(item) for item in payload.get(key) or [] if isinstance(item, dict))
    if payload.get("template_id") or payload.get("template_ref") or payload.get("reaction_class"):
        out["query_hints"].append(_query_hint(candidate, hint_type="template"))
    return out


def _assets_from_compiled_downstream(compiled: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = _empty_assets()
    plugin = dict(compiled.get("literature_template_plugin") or {})
    out["template_cards"].extend(dict(item) for item in plugin.get("template_cards") or [] if isinstance(item, dict))
    out["one_step_rows"].extend(dict(item) for item in plugin.get("one_step_rows") or [] if isinstance(item, dict))
    expansion = dict(compiled.get("route_expansion") or {})
    out["route_expansion_tasks"].extend(dict(item) for item in expansion.get("tasks") or [] if isinstance(item, dict))
    maturity = dict(compiled.get("executable_template_maturity") or {})
    out["extraction_tasks"].extend(dict(item) for item in maturity.get("extraction_tasks") or [] if isinstance(item, dict))
    for row in out["template_cards"]:
        out["query_hints"].append(
            {
                "schema_version": "self_evo_query_hint.v1",
                "hint_type": "template_card",
                "query_terms": [str(row.get("template_id") or ""), str(row.get("reaction_class") or "")],
                "evidence_refs": [str(item) for item in row.get("evidence_refs") or []],
            }
        )
    for task in out["extraction_tasks"]:
        out["query_hints"].append(
            {
                "schema_version": "self_evo_query_hint.v1",
                "hint_type": "executable_template_extraction_task",
                "query_terms": [
                    str(task.get("source_title") or ""),
                    str(task.get("reaction_class") or ""),
                    *[str(item) for item in task.get("precursor_roles") or []],
                ],
                "evidence_refs": [str(item) for item in task.get("evidence_refs") or []],
                "required_structured_fields": [str(item) for item in task.get("required_structured_fields") or []],
            }
        )
    return out


def _query_hint(candidate: dict[str, Any], *, hint_type: str) -> dict[str, Any]:
    payload = dict(candidate.get("payload") or {})
    return {
        "schema_version": "self_evo_query_hint.v1",
        "hint_type": hint_type,
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_type": str(candidate.get("candidate_type") or ""),
        "query_terms": [
            str(payload.get("template_id") or payload.get("template_ref") or ""),
            str(payload.get("reaction_class") or payload.get("source_title") or ""),
        ],
        "evidence_refs": [str(item) for item in candidate.get("evidence_refs") or []],
    }


def _empty_assets() -> dict[str, list[dict[str, Any]]]:
    return {
        "template_cards": [],
        "one_step_rows": [],
        "route_segments": [],
        "route_expansion_tasks": [],
        "extraction_tasks": [],
        "query_hints": [],
    }


def _reject_raw_assets(
    rows: list[dict[str, Any]],
    asset_type: str,
    rejected: list[dict[str, Any]],
    reasons: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if _contains_raw_reaction(row):
            rejected.append({"asset_type": asset_type, "item_index": idx, "reason": "raw_reaction_injection"})
            reasons.append("raw_reaction_injection")
            continue
        out.append(row)
    return out


def _contains_raw_reaction(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"rxn", "rxn_smiles", "reaction_smiles", "raw_reaction", "reaction_candidates"}:
                return True
            if _contains_raw_reaction(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False


def _dedupe_objects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
