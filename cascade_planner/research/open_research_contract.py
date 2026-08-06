"""Shared contract for open Codex structure/template research outputs."""
from __future__ import annotations

from typing import Any


REQUIRED_OPEN_RESEARCH_ARTIFACTS = [
    "structure_template_report.md",
    "structure_template_candidates.json",
    "downstream_consumables.json",
    "evidence/literature_sources.json",
    "evidence/pubchem_validated_compounds.json",
    "validated_compounds.smi",
    "open_agent_audit.json",
]

REQUIRED_OPEN_RESEARCH_JSON_ARTIFACTS = [
    "structure_template_candidates.json",
    "downstream_consumables.json",
    "evidence/literature_sources.json",
    "evidence/pubchem_validated_compounds.json",
    "open_agent_audit.json",
]

OPEN_RESEARCH_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "structure_template_candidates.json": {
        "schema_version": "open_structure_template_candidates.v1",
        "required_keys": [
            "schema_version",
            "case_id",
            "target",
            "candidate_generation_policy",
            "candidates",
            "rejected_items",
            "source_refs",
            "audit_summary",
        ],
        "list_keys": ["candidates", "rejected_items", "source_refs"],
    },
    "downstream_consumables.json": {
        "schema_version": "open_downstream_consumables.v1",
        "required_keys": [
            "schema_version",
            "case_id",
            "planner_handoff",
            "guided_rerun_requests",
            "literature_template_cards",
            "literature_route_segments",
            "executable_template_candidates",
            "route_expansion_tasks",
            "evolution_candidates",
            "rejected_consumables",
        ],
        "list_keys": [
            "guided_rerun_requests",
            "literature_template_cards",
            "literature_route_segments",
            "executable_template_candidates",
            "executable_template_extraction_tasks",
            "source_detail_route_steps",
            "route_expansion_tasks",
            "evolution_candidates",
            "rejected_consumables",
        ],
    },
    "evidence/literature_sources.json": {
        "schema_version": "open_literature_sources.v1",
        "required_keys": [
            "schema_version",
            "case_id",
            "source_relation_policy",
            "sources",
            "excluded_sources",
            "search_log",
        ],
        "list_keys": ["sources", "excluded_sources", "search_log"],
    },
    "evidence/pubchem_validated_compounds.json": {
        "schema_version": "open_pubchem_validated_compounds.v1",
        "required_keys": [
            "schema_version",
            "case_id",
            "compound_source_policy",
            "compounds",
            "rejected_items",
        ],
        "list_keys": ["compounds", "rejected_items"],
    },
    "open_agent_audit.json": {
        "schema_version": "open_structure_agent_audit.v1",
        "required_keys": [
            "schema_version",
            "case_id",
            "final_status",
            "solved",
            "production_kb_promotion",
            "checks",
            "limitations",
            "next_actions",
        ],
        "list_keys": ["checks", "limitations", "next_actions"],
    },
}


def validate_open_research_json_payload(*, name: str, payload: Any) -> list[str]:
    """Return schema-level reasons for a parsed open research JSON payload."""
    spec = OPEN_RESEARCH_JSON_SCHEMAS.get(name)
    if spec is None:
        return []
    reasons: list[str] = []
    if not isinstance(payload, dict):
        return [f"open_agent_json_not_object:{name}"]
    payload = normalize_open_research_json_payload(name=name, payload=payload)
    expected_schema = str(spec["schema_version"])
    if payload.get("schema_version") != expected_schema:
        reasons.append(f"open_agent_json_schema_mismatch:{name}")
    missing = [key for key in spec["required_keys"] if key not in payload]
    reasons.extend(f"open_agent_json_missing_key:{name}:{key}" for key in missing)
    for key in spec.get("list_keys", []):
        if key in payload and not isinstance(payload[key], list):
            reasons.append(f"open_agent_json_key_not_list:{name}:{key}")
    if name == "downstream_consumables.json":
        reasons.extend(_validate_downstream_consumables(payload))
    return reasons


def normalize_open_research_json_payload(*, name: str, payload: Any) -> Any:
    if name != "downstream_consumables.json" or not isinstance(payload, dict):
        return payload
    data = dict(payload)
    for key in (
        "guided_rerun_requests",
        "literature_template_cards",
        "literature_route_segments",
        "executable_template_candidates",
        "executable_template_extraction_tasks",
        "source_detail_route_steps",
        "route_expansion_tasks",
        "evolution_candidates",
        "rejected_consumables",
    ):
        data[key] = [dict(item) for item in data.get(key) or [] if isinstance(item, dict)]
    data["planner_handoff"] = _normalize_planner_handoff(dict(data.get("planner_handoff") or {}), data)
    data["guided_rerun_requests"] = [
        _normalize_evidence_refs(item)
        for item in data.get("guided_rerun_requests") or []
    ]
    data["route_expansion_tasks"] = [
        _normalize_evidence_refs(item)
        for item in data.get("route_expansion_tasks") or []
    ]
    data["executable_template_extraction_tasks"] = [
        _normalize_evidence_refs(item)
        for item in data.get("executable_template_extraction_tasks") or []
    ]
    data["literature_template_cards"] = [
        _normalize_literature_template_card(item)
        for item in data.get("literature_template_cards") or []
    ]
    data["literature_route_segments"] = [
        _normalize_literature_route_segment(item)
        for item in data.get("literature_route_segments") or []
    ]
    data["evolution_candidates"] = [
        _normalize_evolution_candidate(item)
        for item in data.get("evolution_candidates") or []
    ]
    return data


def _normalize_planner_handoff(handoff: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "guided_chemenzy_rerun",
        "subgoal_chemenzy_rerun",
        "template_plugin_rerun",
        "route_segment_unroll",
        "self_evo_candidate_only",
        "chemist_review",
        "no_consumable_found",
    }
    if str(handoff.get("next_action") or "") in allowed:
        return handoff
    if payload.get("guided_rerun_requests"):
        handoff["next_action"] = "guided_chemenzy_rerun"
    elif payload.get("route_expansion_tasks"):
        handoff["next_action"] = "subgoal_chemenzy_rerun"
    elif payload.get("source_detail_route_steps") or payload.get("literature_route_segments"):
        handoff["next_action"] = "route_segment_unroll"
    elif payload.get("evolution_candidates"):
        handoff["next_action"] = "self_evo_candidate_only"
    else:
        handoff["next_action"] = "chemist_review"
    return handoff


def _normalize_evidence_refs(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    refs = [str(ref) for ref in out.get("evidence_refs") or [] if str(ref)]
    refs.extend(str(ref) for ref in out.get("source_refs") or [] if str(ref))
    source_ref = str(out.get("source_ref") or "").strip()
    if source_ref:
        refs.append(source_ref)
    if refs and not out.get("evidence_refs"):
        out["evidence_refs"] = _dedupe(refs)
    return out


def _normalize_literature_template_card(item: dict[str, Any]) -> dict[str, Any]:
    out = _normalize_evidence_refs(dict(item))
    out.setdefault("template_id", out.get("card_id") or out.get("id") or "")
    out.setdefault("validation_status", "draft")
    out.setdefault("reaction_class", out.get("title") or out.get("relation_type") or "literature_advisory")
    out.setdefault("template_level", "advisory_strategy")
    if not isinstance(out.get("applicability"), dict):
        out["applicability"] = {"status": "draft", "note": str(out.get("applicability") or "")}
    if not isinstance(out.get("product_retron"), dict) or not out.get("product_retron"):
        out["product_retron"] = {
            "retron_type": str(out.get("relation_type") or "literature_advisory"),
            "description": str(out.get("title") or out.get("template_id") or ""),
        }
    out.setdefault("scope_limits", [])
    out.setdefault("safety_flags", ["not_raw_reaction_injection", "requires_current_target_audit"])
    out.setdefault("promotion_status", "draft")
    out.setdefault("not_raw_reaction_injection", True)
    return out


def _normalize_literature_route_segment(item: dict[str, Any]) -> dict[str, Any]:
    out = _normalize_evidence_refs(dict(item))
    out.setdefault("validation_status", "draft")
    return out


def _normalize_evolution_candidate(item: dict[str, Any]) -> dict[str, Any]:
    out = _normalize_evidence_refs(dict(item))
    out.setdefault("validation_status", "draft")
    if str(out.get("candidate_type") or "") in {"template_seed", "template", "TemplateSeed"}:
        out["candidate_type"] = "TemplateCandidate"
    out.setdefault("candidate_type", "TemplateCandidate")
    if not isinstance(out.get("payload"), dict):
        out["payload"] = {
            key: value
            for key, value in out.items()
            if key not in {"target_layer", "validation_status", "candidate_type", "evidence_refs"}
        }
    return out


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _validate_downstream_consumables(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    handoff = payload.get("planner_handoff")
    if not isinstance(handoff, dict):
        reasons.append("downstream_consumables_missing_planner_handoff_object")
    else:
        next_action = str(handoff.get("next_action") or "")
        allowed = {
            "guided_chemenzy_rerun",
            "subgoal_chemenzy_rerun",
            "template_plugin_rerun",
            "route_segment_unroll",
            "self_evo_candidate_only",
            "chemist_review",
            "no_consumable_found",
        }
        if next_action not in allowed:
            reasons.append("downstream_consumables_invalid_next_action")
        if bool(handoff.get("solved")):
            reasons.append("downstream_consumables_must_not_claim_solved")
        if bool(handoff.get("production_kb_promotion")):
            reasons.append("downstream_consumables_must_not_promote_production")
    for key in ("guided_rerun_requests", "route_expansion_tasks"):
        for idx, item in enumerate(payload.get(key) or []):
            if not isinstance(item, dict):
                reasons.append(f"downstream_consumables_item_not_object:{key}:{idx}")
                continue
            if not item.get("evidence_refs"):
                reasons.append(f"downstream_consumables_missing_evidence_refs:{key}:{idx}")
            if item.get("write_layer") == "production" or item.get("kb_layer") == "production":
                reasons.append(f"downstream_consumables_direct_production_write:{key}:{idx}")
    for key in ("literature_template_cards", "literature_route_segments", "executable_template_candidates", "evolution_candidates"):
        for idx, item in enumerate(payload.get(key) or []):
            if not isinstance(item, dict):
                reasons.append(f"downstream_consumables_item_not_object:{key}:{idx}")
                continue
            if not item.get("validation_status"):
                reasons.append(f"downstream_consumables_missing_validation_status:{key}:{idx}")
            if key == "evolution_candidates" and item.get("target_layer") == "production":
                reasons.append(f"downstream_consumables_evolution_candidate_targets_production:{idx}")
            if _contains_raw_reaction(item) and key != "executable_template_candidates":
                reasons.append(f"downstream_consumables_raw_reaction_in_non_executable:{key}:{idx}")
    for idx, item in enumerate(payload.get("executable_template_extraction_tasks") or []):
        if not isinstance(item, dict):
            reasons.append(f"downstream_consumables_item_not_object:executable_template_extraction_tasks:{idx}")
            continue
        if str(item.get("schema_version") or "") != "executable_template_extraction_task.v1":
            reasons.append(f"downstream_consumables_invalid_extraction_task_schema:{idx}")
        if not item.get("task_id"):
            reasons.append(f"downstream_consumables_missing_task_id:executable_template_extraction_tasks:{idx}")
        if not item.get("evidence_refs"):
            reasons.append(f"downstream_consumables_missing_evidence_refs:executable_template_extraction_tasks:{idx}")
        if item.get("write_layer") == "production" or item.get("kb_layer") == "production":
            reasons.append(f"downstream_consumables_direct_production_write:executable_template_extraction_tasks:{idx}")
        if _contains_raw_reaction(item):
            reasons.append(f"downstream_consumables_raw_reaction_in_non_executable:executable_template_extraction_tasks:{idx}")
    for idx, item in enumerate(payload.get("source_detail_route_steps") or []):
        if not isinstance(item, dict):
            reasons.append(f"downstream_consumables_item_not_object:source_detail_route_steps:{idx}")
            continue
        if str(item.get("schema_version") or "") != "source_detail_route_step.v1":
            reasons.append(f"downstream_consumables_invalid_source_detail_step_schema:{idx}")
        if not item.get("step_id"):
            reasons.append(f"downstream_consumables_missing_step_id:source_detail_route_steps:{idx}")
        if not item.get("source_ref"):
            reasons.append(f"downstream_consumables_missing_source_ref:source_detail_route_steps:{idx}")
        if not item.get("evidence_refs"):
            reasons.append(f"downstream_consumables_missing_evidence_refs:source_detail_route_steps:{idx}")
        if item.get("write_layer") == "production" or item.get("kb_layer") == "production":
            reasons.append(f"downstream_consumables_direct_production_write:source_detail_route_steps:{idx}")
        if _contains_raw_reaction(item):
            reasons.append(f"downstream_consumables_raw_reaction_in_non_executable:source_detail_route_steps:{idx}")
        reasons.extend(
            f"downstream_consumables_{reason}:source_detail_route_steps:{idx}"
            for reason in _codex_translation_step_reasons(item)
        )
    return reasons


def _codex_translation_step_reasons(item: dict[str, Any]) -> list[str]:
    if str(item.get("provenance") or "") != "codex_source_text_translation":
        return []
    reasons: list[str] = []
    derivation = item.get("structure_derivation")
    if not isinstance(derivation, dict):
        reasons.append("codex_translation_missing_structure_derivation")
    else:
        basis = str(derivation.get("basis") or "").strip()
        if basis not in {
            "explicit_smiles",
            "source_name_to_smiles",
            "source_iupac_to_smiles",
            "source_structure_diagram_to_smiles",
            "source_compound_number_to_smiles",
            "source_table_to_smiles",
            "tool_assisted_source_text_translation",
            "codex_source_text_translation",
            "current_pdf_image_to_smiles",
            "current_image_to_smiles",
            "visual_pdf_image_to_smiles",
            "visual_structure_chain_to_smiles",
        }:
            reasons.append("codex_translation_invalid_structure_basis")
        source_locator = derivation.get("source_locator")
        if isinstance(source_locator, dict):
            locator_present = bool(
                str(source_locator.get("source_ref") or "").strip()
                or str(source_locator.get("url") or "").strip()
                or str(source_locator.get("source_title") or "").strip()
            )
        else:
            locator_present = bool(str(source_locator or "").strip())
        if not locator_present:
            reasons.append("codex_translation_missing_source_locator")
        confidence = str(derivation.get("confidence") or "").strip()
        confidence_prefix = confidence.split("_for_", 1)[0]
        if confidence not in {"high", "medium_high", "medium", "low"} and confidence_prefix not in {
            "high",
            "medium_high",
            "medium",
            "low",
        }:
            reasons.append("codex_translation_missing_confidence")
        tool_checks = derivation.get("tool_checks")
        if isinstance(tool_checks, dict):
            has_tool_checks = bool(tool_checks)
        else:
            has_tool_checks = isinstance(tool_checks, list) and any(str(item).strip() for item in tool_checks)
        if not has_tool_checks:
            reasons.append("codex_translation_missing_tool_checks")
    excerpt = str(item.get("source_excerpt") or "").strip()
    if not excerpt:
        reasons.append("codex_translation_missing_source_excerpt")
    if len(excerpt.split()) > 40:
        reasons.append("codex_translation_source_excerpt_too_long")
    if bool(item.get("full_source_text_stored")):
        reasons.append("codex_translation_stores_full_source_text")
    return reasons


def _contains_raw_reaction(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"rxn", "rxn_smiles", "reaction_smiles", "raw_reaction", "raw_reactions"}:
                return True
            if _contains_raw_reaction(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False
