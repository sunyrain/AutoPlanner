"""Analogical reaction template extraction and guarded target application."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.agent.action_contracts import (
    contains_raw_reaction_payload,
)
from cascade_planner.agent.executable_template_validation import instantiate_literature_template
from cascade_planner.agent.literature_templates import (
    LiteratureTemplateCard,
    LiteratureTemplateLevel,
)
from cascade_planner.research.downstream_compiler import (
    compile_downstream_consumables,
)


RDLogger.DisableLog("rdApp.*")

ANALOGICAL_REACTION_TEMPLATE_SCHEMA = "analogical_reaction_template.v1"
ANALOGICAL_TEMPLATE_REPORT_SCHEMA = "analogical_reaction_template_report.v1"
ANALOGICAL_TEMPLATE_RANKING_SCHEMA = "analogical_reaction_template_ranking.v1"
ANALOGICAL_TEMPLATE_APPLICATION_SCHEMA = "analogical_template_application.v1"
ANALOGICAL_TEMPLATE_APPLICATION_REPORT_SCHEMA = "analogical_template_application_report.v1"
ANALOGICAL_TEMPLATE_VALIDATION_SCHEMA = "analogical_template_application_validation.v1"
ANALOGICAL_TEMPLATE_GUIDED_HINTS_SCHEMA = "analogical_template_guided_hints.v1"
ANALOGICAL_TEMPLATE_PLUGIN_HINTS_SCHEMA = "analogical_template_plugin_hints.v1"

ALLOWED_RELATION_TYPES = {"analog", "family_precedent", "mechanistic_hint"}
ALLOWED_CONFIDENCE = {"low", "medium", "medium_high", "high"}
SUPPORTED_EXECUTABLE_RETRONS = {
    "aryl_ester_acyl_oxygen",
    "bufadienolide_c17_pyrone",
    "macrolactone",
    "o_glycoside",
    "c_glycoside",
    "taxane_c13_side_chain",
    "corey_lactone_side_chain",
}

HYPOTHETICAL_REACTION_CENTER_RETRONS = {
    "steroid_core_retention_bridge",
    "steroid_enone_alcohol_adjustment",
    "steroid_enone_redox_adjustment",
    "steroid_visual_unsaturation_adjustment",
    "steroid_carbonyl_redox_adjustment",
    "steroid_alcohol_protection_redox_adjustment",
}

STEROID_REACTION_CENTER_RETRONS = set(HYPOTHETICAL_REACTION_CENTER_RETRONS)

SOURCE_GROUNDED_VISUAL_RETRONS = {
    "visual_hydrolysis_salt_bridge",
    "visual_deprotection_unmasking",
    "visual_paal_knorr_pyrrole_assembly",
    "visual_source_grounded_connectivity_bridge",
}

HYPOTHETICAL_REACTION_CENTER_RETRONS.update(SOURCE_GROUNDED_VISUAL_RETRONS)


def extract_analogical_reaction_templates_from_blackboard(
    *,
    blackboard: dict[str, Any],
    case_id: str,
    target_smiles: str,
    max_templates: int = 12,
    radius_policy: str = "auto",
) -> dict[str, Any]:
    """Build bounded advisory templates from analog sources and hypotheses."""
    target_profile = dict(blackboard.get("target_profile") or {})
    source_candidates = [
        dict(item)
        for item in (blackboard.get("literature_evidence") or {}).get("source_candidates") or []
        if isinstance(item, dict)
    ]
    hypotheses = [
        dict(item)
        for item in blackboard.get("analogical_hypotheses") or []
        if isinstance(item, dict)
    ]
    hypotheses.extend(
        dict(item)
        for item in (blackboard.get("target_side_disconnection_hypotheses") or {}).get("hypotheses") or []
        if isinstance(item, dict)
    )
    source_refs = _source_refs(blackboard)
    templates: list[dict[str, Any]] = []
    seed_rows: list[dict[str, str]] = []
    for hypothesis in hypotheses:
        seed_rows.extend(_template_seeds_from_hypothesis(hypothesis))
    seed_rows.extend(_template_seeds_from_visual_candidates(blackboard, target_smiles=target_smiles))
    seed_rows.extend(_template_seeds_from_target_functional_centers(target_smiles, target_profile=target_profile))
    if not _target_has_fused_steroid_like_core(target_smiles):
        seed_rows = [seed for seed in seed_rows if not _seed_is_steroid_like(seed)]
    for seed in seed_rows:
        template = _template_from_seed(
            seed,
            case_id=case_id,
            target_smiles=target_smiles,
            target_profile=target_profile,
            source_refs=source_refs,
            source_candidates=source_candidates,
            index=len(templates) + 1,
            radius_policy=radius_policy,
        )
        validation = validate_analogical_reaction_template(template)
        if validation["accepted"]:
            templates.append(template)
        if len(templates) >= max(1, int(max_templates or 12)):
            break
    templates = _dedupe_templates(templates)
    return {
        "schema_version": ANALOGICAL_TEMPLATE_REPORT_SCHEMA,
        "accepted": bool(templates),
        "case_id": case_id,
        "target_smiles": target_smiles,
        "template_count": len(templates),
        "templates": templates,
        "source_refs": source_refs,
        "source_candidate_count": len(source_candidates),
        "hypothesis_count": len(hypotheses),
        "target_functional_center_seed_count": len(_template_seeds_from_target_functional_centers(target_smiles, target_profile=target_profile)),
        "radius_policy": str(radius_policy or "auto"),
        "no_solved_claim": True,
        "reasons": [] if templates else ["no_analogical_template_seeds"],
    }


def validate_analogical_reaction_template(template: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if str(template.get("schema_version") or "") != ANALOGICAL_REACTION_TEMPLATE_SCHEMA:
        reasons.append("invalid_analogical_template_schema")
    if not str(template.get("template_id") or "").strip():
        reasons.append("missing_template_id")
    relation_type = str(template.get("relation_type") or "")
    if relation_type not in ALLOWED_RELATION_TYPES:
        reasons.append("invalid_relation_type")
    if relation_type in {"analog", "family_precedent"} and not str(template.get("scope_gap") or "").strip():
        reasons.append("analog_template_missing_scope_gap")
    if not (template.get("source_refs") or template.get("evidence_refs")):
        reasons.append("missing_source_refs")
    if not str(template.get("reaction_class") or "").strip():
        reasons.append("missing_reaction_class")
    center = dict(template.get("reaction_center") or {})
    if not str(center.get("product_retron_type") or "").strip():
        reasons.append("missing_product_retron_type")
    if str(template.get("confidence") or "") not in ALLOWED_CONFIDENCE:
        reasons.append("invalid_confidence")
    if template.get("no_solved_claim") is not True:
        reasons.append("missing_no_solved_claim_guard")
    if template.get("not_raw_reaction_injection") is not True:
        reasons.append("missing_not_raw_reaction_guard")
    if _contains_raw_reaction_payload(template):
        reasons.append("raw_reaction_injection")
    return {
        "schema_version": "analogical_reaction_template_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "template_id": str(template.get("template_id") or ""),
    }


def rank_analogical_reaction_templates_from_blackboard(blackboard: dict[str, Any]) -> dict[str, Any]:
    templates = [dict(item) for item in blackboard.get("analogical_templates") or [] if isinstance(item, dict)]
    failures = _failure_reasons(blackboard)
    blocked = {
        str(item.get("template_id") or "")
        for item in blackboard.get("template_failure_memory") or []
        if isinstance(item, dict)
        and int(item.get("failure_count") or 0) >= 2
    }
    target_handles = set((blackboard.get("target_profile") or {}).get("functional_handles") or [])
    ranked: list[dict[str, Any]] = []
    for idx, template in enumerate(templates, start=1):
        validation = validate_analogical_reaction_template(template)
        score = 0
        if validation["accepted"]:
            score += 20
        if str(template.get("target_handle") or "") in target_handles:
            score += 20
        if str(template.get("confidence") or "") == "high":
            score += 25
        elif str(template.get("confidence") or "") == "medium_high":
            score += 18
        elif str(template.get("confidence") or "") == "medium":
            score += 10
        if bool(template.get("target_proximal")):
            score += 20
        if "preserve_target_core" in template.get("required_verification", []):
            score += 10
        retron_type = str((template.get("reaction_center") or {}).get("product_retron_type") or "")
        if retron_type in SUPPORTED_EXECUTABLE_RETRONS:
            score += 12
        elif retron_type in HYPOTHETICAL_REACTION_CENTER_RETRONS:
            score += 8
        if "large_atom_jump" in failures:
            score -= 5
        if str(template.get("template_id") or "") in blocked:
            score -= 50
        ranked.append(
            {
                "schema_version": "ranked_analogical_reaction_template.v1",
                "rank_input_index": idx,
                "template_id": str(template.get("template_id") or ""),
                "reaction_class": str(template.get("reaction_class") or ""),
                "target_handle": str(template.get("target_handle") or ""),
                "product_retron_type": str((template.get("reaction_center") or {}).get("product_retron_type") or ""),
                "score": score,
                "score_components": {
                    "validation_passed": bool(validation["accepted"]),
                    "target_handle_overlap": str(template.get("target_handle") or "") in target_handles,
                    "source_confidence": str(template.get("confidence") or ""),
                    "target_proximal": bool(template.get("target_proximal")),
                    "supported_executable_retron": retron_type in SUPPORTED_EXECUTABLE_RETRONS,
                    "supported_hypothetical_reaction_center": retron_type in HYPOTHETICAL_REACTION_CENTER_RETRONS,
                    "previous_failure_penalty": str(template.get("template_id") or "") in blocked,
                },
                "required_verification": [str(item) for item in template.get("required_verification") or []],
                "no_solved_claim": True,
            }
        )
    ranked = sorted(ranked, key=lambda row: (-int(row.get("score") or 0), str(row.get("template_id") or "")))
    selected = [row for row in ranked if int(row.get("score") or 0) > 0][:5]
    return {
        "schema_version": ANALOGICAL_TEMPLATE_RANKING_SCHEMA,
        "accepted": bool(selected),
        "ranked_templates": ranked,
        "selected_templates": selected,
        "ranking_factors": [
            "validation",
            "target_handle_overlap",
            "source_confidence",
            "target_proximity",
            "supported_executable_retron",
            "previous_failure_penalty",
        ],
        "no_solved_claim": True,
        "reasons": [] if selected else ["no_ranked_templates_selected"],
    }


def apply_analogical_templates_to_target(
    *,
    blackboard: dict[str, Any],
    target_smiles: str,
    max_applications: int = 5,
    radius_policy: str = "auto",
    confidence_threshold: str = "medium",
    include_executable_candidates: bool = False,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(target_smiles or ""))
    if mol is None:
        return {
            "schema_version": ANALOGICAL_TEMPLATE_APPLICATION_REPORT_SCHEMA,
            "accepted": False,
            "target_smiles": str(target_smiles or ""),
            "applications": [],
            "executable_template_candidates": [],
            "template_failure_memory": [],
            "reasons": ["invalid_target_smiles"],
            "no_solved_claim": True,
        }
    target_canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    templates_by_id = {
        str(item.get("template_id") or ""): dict(item)
        for item in blackboard.get("analogical_templates") or []
        if isinstance(item, dict)
    }
    selected_ids = [
        str(row.get("template_id") or "")
        for row in (blackboard.get("analogical_template_ranking") or {}).get("selected_templates") or []
        if isinstance(row, dict)
    ]
    if not selected_ids:
        selected_ids = list(templates_by_id)[: max(1, int(max_applications or 5))]
    raw_applications: list[dict[str, Any]] = []
    executable_candidates: list[dict[str, Any]] = []
    failure_memory: list[dict[str, Any]] = []
    threshold_rank = _confidence_rank(confidence_threshold)
    for template_id in selected_ids:
        if len(raw_applications) >= max(1, int(max_applications or 5)):
            break
        template = templates_by_id.get(template_id)
        if not template:
            continue
        validation = validate_analogical_reaction_template(template)
        if not validation["accepted"]:
            raw_applications.append(_rejected_application(template, target_canonical, validation["reasons"]))
            failure_memory.append(_failure_row(template, validation["reasons"]))
            continue
        if _confidence_rank(str(template.get("confidence") or "low")) < threshold_rank:
            reasons = ["template_confidence_below_threshold"]
            raw_applications.append(_rejected_application(template, target_canonical, reasons))
            failure_memory.append(_failure_row(template, reasons))
            continue
        application = _apply_one_template(template, target_canonical, radius_policy=radius_policy)
        raw_applications.append(application)
        if application.get("accepted") and application.get("executable_candidate"):
            executable_candidates.append(dict(application["executable_candidate"]))
        if not application.get("accepted"):
            failure_memory.append(_failure_row(template, application.get("reasons") or ["template_application_rejected"]))
    public_applications = raw_applications if include_executable_candidates else [
        _sanitize_application(row) for row in raw_applications
    ]
    return {
        "schema_version": ANALOGICAL_TEMPLATE_APPLICATION_REPORT_SCHEMA,
        "accepted": bool(raw_applications),
        "target_smiles": target_canonical,
        "application_count": len(raw_applications),
        "accepted_application_count": sum(1 for item in raw_applications if item.get("accepted")),
        "executable_candidate_count": len(executable_candidates),
        "applications": public_applications,
        "executable_template_candidates": executable_candidates if include_executable_candidates else [],
        "candidate_payload_redacted": not bool(include_executable_candidates),
        "template_failure_memory": failure_memory,
        "radius_policy": str(radius_policy or "auto"),
        "confidence_threshold": str(confidence_threshold or "medium"),
        "no_solved_claim": True,
        "reasons": [] if raw_applications else ["no_templates_to_apply"],
    }


def validate_template_applications_for_guided_search(
    *,
    application_report: dict[str, Any],
    case_id: str,
    target_smiles: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    candidates = [
        dict(item)
        for item in application_report.get("executable_template_candidates") or []
        if isinstance(item, dict)
    ]
    payload = {
        "schema_version": "open_research_downstream_consumables.v1",
        "case_id": case_id,
        "target_smiles": target_smiles,
        "planner_handoff": {
            "next_action": "template_plugin_rerun",
            "reason": "analogical_template_applications_validated_for_guided_search",
            "no_solved_claim": True,
        },
        "guided_rerun_requests": [],
        "literature_template_cards": [],
        "literature_route_segments": [],
        "source_detail_route_steps": [],
        "route_expansion_tasks": [],
        "executable_template_candidates": candidates,
        "executable_template_extraction_tasks": [],
        "evolution_candidates": [],
        "rejected_consumables": [],
    }
    compiled = compile_downstream_consumables(
        payload,
        case_id=case_id,
        target_smiles=target_smiles,
        enable_online_anchor_resolution=False,
    )
    guarded = _analogical_guided_hint_bundle(
        compiled,
        case_id=case_id,
        target_smiles=target_smiles,
    )
    refs = _write_analogical_guided_hint_artifacts(guarded, output_dir=output_dir)
    one_step_rows = ((guarded.get("analogical_template_hints") or {}).get("one_step_rows") or [])
    return {
        "schema_version": ANALOGICAL_TEMPLATE_VALIDATION_SCHEMA,
        "accepted": bool(one_step_rows),
        "case_id": case_id,
        "target_smiles": target_smiles,
        "executable_candidate_count": len(candidates),
        "one_step_row_count": len(one_step_rows),
        "compiled_downstream": guarded,
        "compiled_guided_hints": guarded,
        "compiled_downstream_refs": refs,
        "evidence_class": "analogical_template_hint",
        "allowed_use": "guided_search_hint_only",
        "analogy_is_advisory_only": True,
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "requires_parent_route_proof": True,
        "production_write_blocked": True,
        "final_verdict_authority": "deterministic_parent_route_proof",
        "no_solved_claim": True,
        "reasons": [] if one_step_rows else sorted(set(compiled.get("reasons") or ["no_valid_template_applications"])),
    }


def _analogical_guided_hint_bundle(
    compiled: dict[str, Any],
    *,
    case_id: str,
    target_smiles: str,
) -> dict[str, Any]:
    raw_plugin = dict(compiled.get("literature_template_plugin") or {})
    raw_rows = [
        dict(item)
        for item in raw_plugin.get("one_step_rows") or []
        if isinstance(item, dict)
    ]
    rows = [_guard_analogical_hint_row(row) for row in raw_rows]
    hint_plugin = {
        "schema_version": ANALOGICAL_TEMPLATE_PLUGIN_HINTS_SCHEMA,
        "enabled": False,
        "guided_hint_enabled": bool(rows),
        "case_id": case_id,
        "target_smiles": target_smiles,
        "one_step_rows": rows,
        "template_cards": [],
        "validation_reports": [
            dict(item)
            for item in raw_plugin.get("validation_reports") or []
            if isinstance(item, dict)
        ],
        "plugin_flags": {
            "enabled": False,
            "guided_hint_enabled": bool(rows),
            "one_step_rows": rows,
            "template_cards": [],
            "requires_audit": True,
            "not_raw_reaction_injection": True,
            "analogy_is_advisory_only": True,
            "not_exact_literature_segment": True,
            "not_parent_route_proof": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "no_solved_claim": True,
            "source_policy_decision": "analogical_guided_hint_only",
        },
        "evidence_class": "analogical_template_hint",
        "allowed_use": "guided_search_hint_only",
        "analogy_is_advisory_only": True,
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "requires_parent_route_proof": True,
        "production_write_blocked": True,
        "no_solved_claim": True,
    }
    exact_plugin_disabled = _disabled_exact_literature_plugin(row_count=len(rows))
    return {
        "schema_version": ANALOGICAL_TEMPLATE_GUIDED_HINTS_SCHEMA,
        "case_id": case_id,
        "target_smiles": target_smiles,
        "accepted": bool(rows),
        "reasons": [] if rows else sorted(set(compiled.get("reasons") or ["no_valid_template_applications"])),
        "source": "analogical_template_application_validation",
        "source_compiler_schema": str(compiled.get("schema_version") or ""),
        "source_compiler_reasons": [str(item) for item in compiled.get("reasons") or []],
        "analogical_template_hints": hint_plugin,
        "guided_search_hints": hint_plugin,
        "literature_template_plugin": exact_plugin_disabled,
        "one_step_row_count": len(rows),
        "evidence_class": "analogical_template_hint",
        "allowed_use": "guided_search_hint_only",
        "analogy_is_advisory_only": True,
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "requires_parent_route_proof": True,
        "production_write_blocked": True,
        "final_verdict_authority": "deterministic_parent_route_proof",
        "no_solved_claim": True,
    }


def _guard_analogical_hint_row(row: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    trace = dict(guarded.get("literature_template_trace") or {})
    trace.update(
        {
            "source_evidence_class": "analogical_template_hint",
            "analogical_template_hint": True,
            "source_detail_exact_step": False,
            "structured_segment_step": False,
            "not_exact_literature_segment": True,
            "not_parent_route_proof": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "no_solved_claim": True,
        }
    )
    template_payload = dict(guarded.get("template") or guarded.get("templates") or {})
    template_payload.update(
        {
            "source_policy_decision": "analogical_guided_hint_only",
            "evidence_class": "analogical_template_hint",
            "allowed_use": "guided_search_hint_only",
            "used_as_proof": False,
            "not_exact_literature_segment": True,
            "not_parent_route_proof": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "production_write_blocked": True,
            "no_solved_claim": True,
        }
    )
    template_payload["literature_template_trace"] = trace
    guarded.update(
        {
            "template": template_payload,
            "templates": template_payload,
            "literature_template_trace": trace,
            "source_policy_decision": "analogical_guided_hint_only",
            "row_source": "analogical_template_application",
            "evidence_class": "analogical_template_hint",
            "allowed_use": "guided_search_hint_only",
            "used_as_proof": False,
            "not_exact_literature_segment": True,
            "not_parent_route_proof": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "production_write_blocked": True,
            "no_solved_claim": True,
        }
    )
    return guarded


def _disabled_exact_literature_plugin(*, row_count: int) -> dict[str, Any]:
    return {
        "enabled": False,
        "template_cards": [],
        "one_step_rows": [],
        "validation_reports": [],
        "plugin_flags": {
            "enabled": False,
            "template_cards": [],
            "one_step_rows": [],
            "requires_audit": True,
            "not_raw_reaction_injection": True,
            "disabled_reason": "analogical_template_hints_are_not_exact_literature_rows",
            "analogical_hint_row_count": int(row_count),
            "analogy_is_advisory_only": True,
            "not_exact_literature_segment": True,
            "not_parent_route_proof": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "no_solved_claim": True,
        },
        "disabled_reason": "analogical_template_hints_are_not_exact_literature_rows",
        "analogical_hint_row_count": int(row_count),
        "analogy_is_advisory_only": True,
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }


def _write_analogical_guided_hint_artifacts(
    guided: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hints_path = out / "compiled_analogical_template_hints.json"
    disabled_plugin_path = out / "compiled_literature_template_plugin.json"
    _write_json(hints_path, guided)
    _write_json(disabled_plugin_path, guided.get("literature_template_plugin") or {})
    return {
        "compiled_analogical_template_hints": str(hints_path),
        "compiled_literature_template_plugin": str(disabled_plugin_path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def compact_template_application_summary(row: dict[str, Any]) -> dict[str, Any]:
    hypothesis = _compact_hypothetical_route_hypothesis(dict(row.get("hypothetical_route_hypothesis") or {}))
    return {
        "schema_version": "agent_template_application_summary.v1",
        "application_id": str(row.get("application_id") or ""),
        "template_id": str(row.get("template_id") or ""),
        "accepted": bool(row.get("accepted")),
        "allowed_use": str(row.get("allowed_use") or ""),
        "product_retron_type": str(row.get("product_retron_type") or ""),
        "cut_fragment_count": len(row.get("cut_fragments") or []),
        "executable_candidate_available": bool(row.get("executable_candidate_available") or row.get("executable_candidate")),
        "hypothetical_route_hypothesis": hypothesis,
        "hypothetical_precursor_hints": _compact_hypothetical_precursor_hints(row.get("hypothetical_precursor_hints") or []),
        "reaction_center_summary": dict(row.get("reaction_center_summary") or {}),
        "reasons": [str(item) for item in row.get("reasons") or []],
        "analogy_is_advisory_only": True,
        "not_exact_literature_segment": bool(row.get("not_exact_literature_segment", True)),
        "not_parent_route_proof": bool(row.get("not_parent_route_proof", True)),
        "no_solved_claim": True,
    }


def _compact_hypothetical_route_hypothesis(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "schema_version": str(row.get("schema_version") or "analogical_route_hypothesis.v1"),
        "route_status": str(row.get("route_status") or "hypothesis_only_not_solved"),
        "hypothesis_type": str(row.get("hypothesis_type") or ""),
        "reaction_center_idea": str(row.get("reaction_center_idea") or ""),
        "expected_precursor_type": str(row.get("expected_precursor_type") or ""),
        "template_application": str(row.get("template_application") or ""),
        "must_preserve": [str(item) for item in row.get("must_preserve") or [] if str(item or "").strip()],
        "risk_flags": [str(item) for item in row.get("risk_flags") or [] if str(item or "").strip()],
        "required_verification": [
            str(item)
            for item in row.get("required_verification") or []
            if str(item or "").strip()
        ],
        "evidence_class": "reaction_center_template_analogy",
        "allowed_use": "planner_priority_and_guided_search_hint_only",
        "no_solved_claim": True,
    }


def _compact_hypothetical_precursor_hints(rows: list[Any]) -> list[dict[str, Any]]:
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
                "hint_id": str(row.get("hint_id") or f"hyp_precursor:{_short_hash(precursor)}"),
                "target_smiles": str(row.get("target_smiles") or ""),
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


def _sanitize_application(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value
        for key, value in dict(row).items()
        if key not in {"executable_candidate"}
    }
    sanitized["executable_candidate_available"] = bool(row.get("executable_candidate"))
    sanitized["candidate_payload_redacted"] = bool(row.get("executable_candidate"))
    return sanitized


def _template_seeds_from_visual_candidates(blackboard: dict[str, Any], *, target_smiles: str) -> list[dict[str, str]]:
    seeds: list[dict[str, str]] = []
    target_mol = Chem.MolFromSmiles(str(target_smiles or ""))
    for chain in (blackboard.get("literature_evidence") or {}).get("visual_chains") or []:
        if not isinstance(chain, dict) or not _visual_chain_is_exploratory(chain):
            continue
        source_ref = str(chain.get("source_ref") or chain.get("source_pdf_path") or chain.get("artifact_ref") or "")
        for step in chain.get("steps") or []:
            if not isinstance(step, dict):
                continue
            product = _canonical_smiles(str(step.get("product_smiles") or ""))
            reactants = [
                _canonical_smiles(str(item or ""))
                for item in step.get("reactant_smiles") or []
            ]
            reactants = [item for item in reactants if item]
            if not product or not reactants:
                continue
            product_mol = Chem.MolFromSmiles(product)
            precursor_mol = Chem.MolFromSmiles(reactants[0])
            if product_mol is None or precursor_mol is None:
                continue
            if not _visual_step_matches_target(product_mol, precursor_mol, target_mol):
                continue
            if _mol_has_fused_steroid_like_core(product_mol) and _mol_has_fused_steroid_like_core(precursor_mol):
                seeds.append(
                    {
                        "target_handle": "visual_same_core_steroid_precursor",
                        "reaction_class": "visual_steroid_same_core_unsaturation_or_redox_adjustment",
                        "mechanistic_class": "visual_connectivity_reaction_center_transfer",
                        "product_retron_type": "steroid_visual_unsaturation_adjustment",
                        "scope_gap": "visual extraction supplied an achiral/connectivity-only same-core precursor; stereochemistry and exact literature identity are not proven",
                        "confidence": "low",
                        "target_proximal": "true",
                        "visual_source_ref": source_ref,
                        "visual_source_locator": str(step.get("source_locator") or ""),
                        "visual_product_smiles": product,
                        "visual_precursor_smiles": reactants[0],
                        "visual_stereochemistry_status": str(step.get("stereochemistry_status") or "unspecified_or_partial"),
                    }
                )
            else:
                seeds.append(
                    _visual_source_grounded_seed_from_step(
                        step,
                        product_smiles=product,
                        precursor_smiles=reactants[0],
                        reactant_count=len(reactants),
                        source_ref=source_ref,
                    )
                )
            if len(seeds) >= 4:
                return seeds
    return seeds


def _visual_step_matches_target(product_mol: Any, precursor_mol: Any, target_mol: Any) -> bool:
    if target_mol is None:
        return True
    return max(
        _fingerprint_similarity(product_mol, target_mol),
        _fingerprint_similarity(precursor_mol, target_mol),
    ) >= 0.35


def _fingerprint_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    try:
        left_fp = Chem.RDKFingerprint(left)
        right_fp = Chem.RDKFingerprint(right)
        return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))
    except Exception:
        return 0.0


def _visual_source_grounded_seed_from_step(
    step: dict[str, Any],
    *,
    product_smiles: str,
    precursor_smiles: str,
    reactant_count: int,
    source_ref: str,
) -> dict[str, str]:
    text = _visual_step_text(step)
    if "paal" in text or "pyrrole" in text or reactant_count > 1:
        retron_type = "visual_paal_knorr_pyrrole_assembly"
        reaction_class = "visual_paal_knorr_or_pyrrole_assembly"
        mechanistic_class = "visual_source_grounded_condensation"
        target_handle = "visual_pyrrole_core_assembly"
        scope_gap = "visual source suggests pyrrole-core assembly, but stereochemistry and exact atom mapping require verifier review"
    elif "naoh" in text or "hydrolysis" in text or "calcium" in text or "salt" in text:
        retron_type = "visual_hydrolysis_salt_bridge"
        reaction_class = "visual_ester_hydrolysis_or_salt_formation"
        mechanistic_class = "visual_source_grounded_deprotection_salt_formation"
        target_handle = "visual_carboxylate_or_salt_state"
        scope_gap = "visual source suggests hydrolysis/salt formation, but salt state and exact stoichiometry are not proof"
    elif "hcl" in text or "hydrochloric" in text or "deprotect" in text or "acetonide" in text or "dihydroxy" in text:
        retron_type = "visual_deprotection_unmasking"
        reaction_class = "visual_acetal_or_protecting_group_deprotection"
        mechanistic_class = "visual_source_grounded_deprotection"
        target_handle = "visual_protecting_group_unmasking"
        scope_gap = "visual source suggests deprotection/unmasking, but exact stereochemistry and protecting-group identity require verifier review"
    else:
        retron_type = "visual_source_grounded_connectivity_bridge"
        reaction_class = "visual_source_grounded_connectivity_transform"
        mechanistic_class = "visual_connectivity_reaction_center_transfer"
        target_handle = "visual_source_grounded_precursor"
        scope_gap = "visual extraction supplied a connectivity-only literature step; exact source-detail proof still requires verifier review"
    return {
        "target_handle": target_handle,
        "reaction_class": reaction_class,
        "mechanistic_class": mechanistic_class,
        "product_retron_type": retron_type,
        "scope_gap": scope_gap,
        "confidence": "low",
        "target_proximal": "true",
        "visual_source_ref": source_ref,
        "visual_source_locator": str(step.get("source_locator") or ""),
        "visual_product_smiles": product_smiles,
        "visual_precursor_smiles": precursor_smiles,
        "visual_stereochemistry_status": str(step.get("stereochemistry_status") or "unspecified_or_partial"),
    }


def _visual_step_text(step: dict[str, Any]) -> str:
    condition = dict(step.get("condition_candidate") or {})
    values = [
        str(step.get("step_id") or ""),
        str(step.get("product_label") or ""),
        str(step.get("source_locator") or ""),
        str(step.get("source_excerpt") or ""),
        str(condition.get("reagent") or ""),
        str(condition.get("catalyst") or ""),
        str(condition.get("solvent") or ""),
        str(condition.get("source_grounding") or ""),
        str(condition.get("condition_text_transcribed") or ""),
    ]
    return " ".join(values).lower()


def _visual_chain_is_exploratory(chain: dict[str, Any]) -> bool:
    if bool(chain.get("exact_ready")):
        return False
    acceptance = str(chain.get("acceptance_level") or "").lower()
    if bool(chain.get("exploratory_accepted")) or "exploratory" in acceptance:
        return True
    for step in chain.get("steps") or []:
        if not isinstance(step, dict):
            continue
        allowed_use = str(step.get("allowed_use") or "").lower()
        if bool(step.get("not_exact_literature_segment")) or "exploratory" in allowed_use:
            return True
    return False


def _template_seeds_from_hypothesis(hypothesis: dict[str, Any]) -> list[dict[str, str]]:
    text = " ".join(
        str(hypothesis.get(key) or "")
        for key in ("target_handle", "reaction_family", "proposed_disconnection_region", "expected_precursor_type")
    ).lower()
    seeds: list[dict[str, str]] = []
    if "aryl_ester" in text or "anthranilate" in text or "ester" in text:
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "aryl_ester"),
            "reaction_class": "esterification_or_acyl_transfer",
            "mechanistic_class": "acyl_substitution",
            "product_retron_type": "aryl_ester_acyl_oxygen",
            "scope_gap": "analog ester precedent must be rechecked on the target steric and electronic environment",
            "confidence": "medium_high",
            "target_proximal": "true",
        })
    if "bufadienolide" in text or "pyrone" in text:
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "bufadienolide_c17_pyrone"),
            "reaction_class": "bufadienolide_side_chain_installation",
            "mechanistic_class": "c17_pyrone_fragment_disconnection",
            "product_retron_type": "bufadienolide_c17_pyrone",
            "scope_gap": "analog pyrone installation requires exact target-side stereochemical and core-retention verification",
            "confidence": "medium_high",
            "target_proximal": "true",
        })
    if "macrolactone" in text or "lactone" in text:
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "lactone"),
            "reaction_class": "lactonization",
            "mechanistic_class": "intramolecular_acyl_substitution",
            "product_retron_type": "macrolactone",
            "scope_gap": "analog lactonization must match ring size and functional-group tolerance",
            "confidence": "medium",
            "target_proximal": "true",
        })
    if "glycoside" in text:
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "glycoside"),
            "reaction_class": "glycosylation",
            "mechanistic_class": "glycosidic_bond_disconnection",
            "product_retron_type": "o_glycoside",
            "scope_gap": "analog glycosylation must preserve anomeric linkage type and acceptor compatibility",
            "confidence": "medium",
            "target_proximal": "true",
        })
    if "imide" in text or "succinimide" in text:
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "imide"),
            "reaction_class": "imide_fragment_preparation",
            "mechanistic_class": "imide_acylation_or_cyclization",
            "product_retron_type": "imide_or_succinimide_n_acyl",
            "scope_gap": "imide analog precedent is advisory until a deterministic retron matcher is available",
            "confidence": "medium",
            "target_proximal": "false",
        })
    if "tertiary_amine" in text or "amine" in text:
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "tertiary_amine"),
            "reaction_class": "amine_state_adjustment",
            "mechanistic_class": "alkylation_or_redox_state_adjustment",
            "product_retron_type": "tertiary_amine_state_adjustment",
            "scope_gap": "amine-state analog precedent is advisory and requires condition compatibility verification",
            "confidence": "low",
            "target_proximal": "false",
        })
    if (
        "steroid" in text
        or "cardenolide" in text
        or "polycyclic_cage_core" in text
        or "cage core" in text
        or "same-core" in text
        or "core preservation" in text
    ):
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "steroid_core"),
            "reaction_class": "steroid_core_retention_functionalization",
            "mechanistic_class": "same_core_redox_or_protecting_group_adjustment",
            "product_retron_type": "steroid_core_retention_bridge",
            "scope_gap": "steroid-core precedent is advisory unless a target-proximal same-core precursor and parent proof are verified",
            "confidence": "medium",
            "target_proximal": "true",
        })
    if "enone" in text or "side-chain alcohol" in text or "alcohol" in text:
        seeds.append({
            "target_handle": str(hypothesis.get("target_handle") or "steroid_functional_handle"),
            "reaction_class": "steroid_enone_or_alcohol_state_adjustment",
            "mechanistic_class": "redox_or_protecting_group_level_transform",
            "product_retron_type": "steroid_enone_alcohol_adjustment",
            "scope_gap": "redox/protection analogy must retain the tetracyclic core and match the target oxidation state deterministically",
            "confidence": "medium",
            "target_proximal": "true",
        })
    return seeds


def _template_seeds_from_target_functional_centers(
    target_smiles: str,
    *,
    target_profile: dict[str, Any],
) -> list[dict[str, str]]:
    mol = Chem.MolFromSmiles(str(target_smiles or ""))
    if mol is None:
        return []
    if not _mol_has_fused_steroid_like_core(mol):
        return []
    seeds: list[dict[str, str]] = []
    seeds.append(
        {
            "target_handle": "steroid_core",
            "reaction_class": "steroid_core_retention_functionalization",
            "mechanistic_class": "same_core_redox_or_protecting_group_adjustment",
            "product_retron_type": "steroid_core_retention_bridge",
            "scope_gap": "family precedent supports preserving the polycyclic core while exploring peripheral redox/protection changes; not a proof of target synthesis",
            "confidence": "medium",
            "target_proximal": "true",
        }
    )
    if _has_substructure(mol, "[#6]=[#6]-[CX3](=O)[#6]") or _has_substructure(mol, "[CX3](=O)-[#6]=[#6]"):
        seeds.append(
            {
                "target_handle": "steroid_enone",
                "reaction_class": "steroid_enone_redox_or_allylic_adjustment",
                "mechanistic_class": "enone_oxidation_state_migration_or_enone_unmasking",
                "product_retron_type": "steroid_enone_redox_adjustment",
                "scope_gap": "enone logic is transferred from analog steroid routes and must be checked for regioselectivity and core retention",
                "confidence": "medium",
                "target_proximal": "true",
            }
        )
    if _has_substructure(mol, "[CX3](=O)[#6]"):
        seeds.append(
            {
                "target_handle": "steroid_ketone",
                "reaction_class": "steroid_carbonyl_redox_adjustment",
                "mechanistic_class": "alcohol_to_ketone_or_ketone_to_alcohol_state_change",
                "product_retron_type": "steroid_carbonyl_redox_adjustment",
                "scope_gap": "carbonyl placement is target-specific; analog precedent only supports the redox idea, not exact selectivity",
                "confidence": "medium",
                "target_proximal": "true",
            }
        )
    if _has_substructure(mol, "[CX4][OX2H]") or _has_substructure(mol, "[CH2][OX2H]"):
        seeds.append(
            {
                "target_handle": "steroid_alcohol",
                "reaction_class": "steroid_alcohol_protection_or_redox_adjustment",
                "mechanistic_class": "alcohol_protection_deprotection_or_redox_state_adjustment",
                "product_retron_type": "steroid_alcohol_protection_redox_adjustment",
                "scope_gap": "alcohol protection/redox precedent is broad and must be tested against neighboring carbonyl/enone compatibility",
                "confidence": "medium",
                "target_proximal": "true",
            }
        )
    return seeds


def _target_has_fused_steroid_like_core(target_smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(target_smiles or ""))
    return bool(mol is not None and _mol_has_fused_steroid_like_core(mol))


def _mol_has_fused_steroid_like_core(mol: Any) -> bool:
    return _largest_fused_ring_system_size(mol) >= 4


def _largest_fused_ring_system_size(mol: Any) -> int:
    if mol is None:
        return 0
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    if not rings:
        return 0
    graph: dict[int, set[int]] = {idx: set() for idx in range(len(rings))}
    for left_idx, left in enumerate(rings):
        for right_idx in range(left_idx + 1, len(rings)):
            if len(left & rings[right_idx]) >= 2:
                graph[left_idx].add(right_idx)
                graph[right_idx].add(left_idx)
    seen: set[int] = set()
    largest = 0
    for start in graph:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in graph[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return largest


def _seed_is_steroid_like(seed: dict[str, str]) -> bool:
    retron_type = str(seed.get("product_retron_type") or "")
    if retron_type in STEROID_REACTION_CENTER_RETRONS:
        return True
    text = " ".join(
        str(seed.get(key) or "")
        for key in ("target_handle", "reaction_class", "mechanistic_class", "scope_gap")
    ).lower()
    return "steroid" in text


def _template_from_seed(
    seed: dict[str, str],
    *,
    case_id: str,
    target_smiles: str,
    target_profile: dict[str, Any],
    source_refs: list[str],
    source_candidates: list[dict[str, Any]],
    index: int,
    radius_policy: str,
) -> dict[str, Any]:
    source_ref = source_refs[0] if source_refs else f"{case_id}:analogical_hypothesis"
    retron_type = str(seed.get("product_retron_type") or "")
    template_id = _template_id(case_id, retron_type, index)
    radius = _default_radius(retron_type, radius_policy=radius_policy)
    confidence = str(seed.get("confidence") or "medium")
    if not source_candidates and confidence == "medium_high":
        confidence = "medium"
    template = {
        "schema_version": ANALOGICAL_REACTION_TEMPLATE_SCHEMA,
        "template_id": template_id,
        "case_id": case_id,
        "target_smiles": target_smiles,
        "target_handle": str(seed.get("target_handle") or ""),
        "source_refs": source_refs or [source_ref],
        "evidence_refs": source_refs or [source_ref],
        "source_candidate_count": len(source_candidates),
        "relation_type": "analog" if source_candidates else "mechanistic_hint",
        "reaction_class": str(seed.get("reaction_class") or ""),
        "mechanistic_class": str(seed.get("mechanistic_class") or ""),
        "reaction_center": {
            "product_retron_type": retron_type,
            "template_radius": radius,
            "local_environment": "reaction_center_plus_nearest_functional_context",
            "not_raw_reaction_injection": True,
        },
        "template_radius": radius,
        "required_substructure": [retron_type],
        "forbidden_substructure": [],
        "preserve_substructure": _preserve_substructures(target_profile),
        "applicability_notes": [
            "template is a search hypothesis",
            "exact target proof requires verifier and parent-route proof",
        ],
        "scope_gap": str(seed.get("scope_gap") or "analogical template requires target-specific verification"),
        "risk_flags": _risk_flags(retron_type),
        "required_verification": [
            "template_applicability",
            "product_reconstruction",
            "preserve_target_core",
            "route_verifier",
            "parent_route_proof",
        ],
        "confidence": confidence,
        "target_proximal": str(seed.get("target_proximal") or "").lower() == "true",
        "no_solved_claim": True,
        "not_raw_reaction_injection": True,
    }
    visual_hint = _visual_hint_from_seed(seed)
    if visual_hint:
        template["visual_connectivity_hint"] = visual_hint
        template["evidence_refs"] = _dedupe([*template["evidence_refs"], str(visual_hint.get("source_ref") or "")])
        template["source_refs"] = _dedupe([*template["source_refs"], str(visual_hint.get("source_ref") or "")])
    return template


def _visual_hint_from_seed(seed: dict[str, str]) -> dict[str, Any]:
    precursor = str(seed.get("visual_precursor_smiles") or "").strip()
    product = str(seed.get("visual_product_smiles") or "").strip()
    if not precursor or not product:
        return {}
    return {
        "schema_version": "analogical_visual_connectivity_hint.v1",
        "source_ref": str(seed.get("visual_source_ref") or ""),
        "source_locator": str(seed.get("visual_source_locator") or ""),
        "product_smiles": product,
        "precursor_smiles": precursor,
        "stereochemistry_status": str(seed.get("visual_stereochemistry_status") or "unspecified_or_partial"),
        "allowed_use": "guided_search_subgoal_hint_only",
        "evidence_class": "visual_connectivity_approximation",
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "no_solved_claim": True,
    }


def _apply_one_template(template: dict[str, Any], target_smiles: str, *, radius_policy: str) -> dict[str, Any]:
    retron_type = str((template.get("reaction_center") or {}).get("product_retron_type") or "")
    application_id = f"apply:{template.get('template_id')}:{_short_hash(target_smiles)}"
    if retron_type not in SUPPORTED_EXECUTABLE_RETRONS:
        if retron_type in HYPOTHETICAL_REACTION_CENTER_RETRONS:
            return _hypothetical_reaction_center_application(
                template,
                target_smiles,
                retron_type=retron_type,
                application_id=application_id,
                radius_policy=radius_policy,
            )
        return {
            "schema_version": ANALOGICAL_TEMPLATE_APPLICATION_SCHEMA,
            "application_id": application_id,
            "template_id": str(template.get("template_id") or ""),
            "target_smiles": target_smiles,
            "product_retron_type": retron_type,
            "accepted": False,
            "allowed_use": "advisory_or_rerank_only",
            "cut_fragments": [],
            "reasons": ["unsupported_executable_retron"],
            "no_solved_claim": True,
        }
    card = _literature_card_from_analogical_template(template)
    candidate = instantiate_literature_template(target_smiles, card, target_smiles=target_smiles)
    validation = dict(candidate.validation_report or {})
    applicability = dict(candidate.applicability_report or {})
    accepted = bool(validation.get("allowed_for_one_step_source"))
    reasons = [str(item) for item in validation.get("reasons") or []]
    if not accepted and not reasons:
        reasons = [str(item) for item in applicability.get("mismatch_reasons") or ["template_application_rejected"]]
    return {
        "schema_version": ANALOGICAL_TEMPLATE_APPLICATION_SCHEMA,
        "application_id": application_id,
        "template_id": str(template.get("template_id") or ""),
        "target_smiles": target_smiles,
        "product_retron_type": retron_type,
        "accepted": accepted,
        "allowed_use": "executable_candidate" if accepted else "advisory_or_rerank_only",
        "match_confidence": str(applicability.get("match_confidence") or "none"),
        "cut_fragments": [str(item) for item in applicability.get("cut_fragments") or []],
        "executable_candidate_available": accepted,
        "executable_candidate": candidate.to_dict() if accepted else {},
        "validation_report": validation,
        "template_radius": str((template.get("reaction_center") or {}).get("template_radius") or radius_policy or "auto"),
        "reasons": reasons,
        "no_solved_claim": True,
    }


def _hypothetical_reaction_center_application(
    template: dict[str, Any],
    target_smiles: str,
    *,
    retron_type: str,
    application_id: str,
    radius_policy: str,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(target_smiles or ""))
    center_summary = _target_reaction_center_summary(mol)
    hypothesis = _hypothesis_for_retron(retron_type, center_summary=center_summary)
    reasons: list[str] = []
    source_grounded_visual = retron_type in SOURCE_GROUNDED_VISUAL_RETRONS
    steroid_retron = retron_type in STEROID_REACTION_CENTER_RETRONS
    if steroid_retron and not center_summary.get("polycyclic_core_like"):
        reasons.append("target_not_polycyclic_core_like")
    if source_grounded_visual and not template.get("visual_connectivity_hint"):
        reasons.append("missing_visual_connectivity_hint")
    if steroid_retron and retron_type == "steroid_enone_redox_adjustment" and int(center_summary.get("enone_count") or 0) <= 0:
        reasons.append("target_enone_retron_not_detected")
    if steroid_retron and retron_type == "steroid_enone_alcohol_adjustment" and not (
        int(center_summary.get("enone_count") or 0) > 0 or int(center_summary.get("alcohol_count") or 0) > 0
    ):
        reasons.append("target_enone_or_alcohol_retron_not_detected")
    if steroid_retron and retron_type == "steroid_carbonyl_redox_adjustment" and int(center_summary.get("carbonyl_count") or 0) <= 0:
        reasons.append("target_carbonyl_retron_not_detected")
    if steroid_retron and retron_type == "steroid_alcohol_protection_redox_adjustment" and int(center_summary.get("alcohol_count") or 0) <= 0:
        reasons.append("target_alcohol_retron_not_detected")
    accepted = not reasons
    if not accepted:
        reasons.append("hypothetical_reaction_center_rejected_by_target_scan")
    if accepted and source_grounded_visual:
        precursor_hints = _visual_precursor_hints_from_template(
            template,
            target_smiles=target_smiles,
            retron_type=retron_type,
            application_id=application_id,
        )
    else:
        precursor_hints = _hypothetical_precursor_hints(
            target_smiles=target_smiles,
            retron_type=retron_type,
            application_id=application_id,
            hypothesis=hypothesis,
        )
    if accepted and not source_grounded_visual:
        precursor_hints.extend(
            _visual_precursor_hints_from_template(
                template,
                target_smiles=target_smiles,
                retron_type=retron_type,
                application_id=application_id,
            )
        )
        precursor_hints = _dedupe_precursor_hints(
            precursor_hints,
            target_canonical=_canonical_smiles(target_smiles),
            limit=8,
        )
    return {
        "schema_version": ANALOGICAL_TEMPLATE_APPLICATION_SCHEMA,
        "application_id": application_id,
        "template_id": str(template.get("template_id") or ""),
        "target_smiles": target_smiles,
        "product_retron_type": retron_type,
        "accepted": accepted,
        "allowed_use": "hypothesis_only_not_solved" if accepted else "advisory_or_rerank_only",
        "match_confidence": "broad_reaction_center" if accepted else "none",
        "cut_fragments": [],
        "executable_candidate_available": False,
        "hypothetical_route_hypothesis": hypothesis if accepted else {},
        "hypothetical_precursor_hints": precursor_hints,
        "reaction_center_summary": center_summary,
        "template_radius": str((template.get("reaction_center") or {}).get("template_radius") or radius_policy or "auto"),
        "reasons": reasons,
        "analogy_is_advisory_only": True,
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "requires_parent_route_proof": True,
        "no_solved_claim": True,
    }


def _hypothetical_precursor_hints(
    *,
    target_smiles: str,
    retron_type: str,
    application_id: str,
    hypothesis: dict[str, Any],
) -> list[dict[str, Any]]:
    target_canonical = _canonical_smiles(target_smiles)
    if not target_canonical:
        return []
    risk_flags = [str(item) for item in hypothesis.get("risk_flags") or [] if str(item or "").strip()]
    hints: list[dict[str, Any]] = []
    if retron_type in {
        "steroid_enone_redox_adjustment",
        "steroid_enone_alcohol_adjustment",
        "steroid_visual_unsaturation_adjustment",
    }:
        for precursor in _enone_saturated_ketone_variants(target_canonical):
            hints.append(
                _precursor_hint(
                    target_smiles=target_canonical,
                    precursor_smiles=precursor,
                    retron_type=retron_type,
                    application_id=application_id,
                    precursor_role="same_core_saturated_ketone_enone_precursor",
                    hypothesis_type=str(hypothesis.get("hypothesis_type") or "enone_redox_or_unmasking"),
                    risk_flags=[*risk_flags, "enone_regioselectivity_unproven"],
                )
            )
    if retron_type in {
        "steroid_carbonyl_redox_adjustment",
        "steroid_enone_redox_adjustment",
        "steroid_enone_alcohol_adjustment",
        "steroid_visual_unsaturation_adjustment",
    }:
        for precursor in _reaction_product_variants(
            target_canonical,
            "[C:1](=[O:2])>>[C:1]([O:2])",
            max_products=4,
        ):
            hints.append(
                _precursor_hint(
                    target_smiles=target_canonical,
                    precursor_smiles=precursor,
                    retron_type=retron_type,
                    application_id=application_id,
                    precursor_role="same_core_hydroxy_steroid_carbonyl_precursor",
                    hypothesis_type=str(hypothesis.get("hypothesis_type") or "carbonyl_redox_adjustment"),
                    risk_flags=[*risk_flags, "alcohol_stereochemistry_unassigned"],
                )
            )
    if retron_type in {"steroid_alcohol_protection_redox_adjustment", "steroid_enone_alcohol_adjustment"}:
        for precursor in _reaction_product_variants(
            target_canonical,
            "[CH2:1][OX2H:2]>>[CH:1]=[O:2]",
            max_products=3,
        ):
            hints.append(
                _precursor_hint(
                    target_smiles=target_canonical,
                    precursor_smiles=precursor,
                    retron_type=retron_type,
                    application_id=application_id,
                    precursor_role="same_core_aldehyde_primary_alcohol_precursor",
                    hypothesis_type=str(hypothesis.get("hypothesis_type") or "alcohol_protection_or_redox_adjustment"),
                    risk_flags=[*risk_flags, "primary_alcohol_redox_direction_hypothetical"],
                )
            )
        for precursor in _reaction_product_variants(
            target_canonical,
            "[OX2H:1]>>[O:1]C(C)=O",
            max_products=4,
        ):
            hints.append(
                _precursor_hint(
                    target_smiles=target_canonical,
                    precursor_smiles=precursor,
                    retron_type=retron_type,
                    application_id=application_id,
                    precursor_role="same_core_acetate_protected_alcohol_precursor",
                    hypothesis_type=str(hypothesis.get("hypothesis_type") or "alcohol_protection_or_redox_adjustment"),
                    risk_flags=[*risk_flags, "protecting_group_choice_hypothetical"],
                )
            )
    if retron_type == "steroid_core_retention_bridge":
        variants = [
            *_reaction_product_variants(
                target_canonical,
                "[C:1](=[O:2])>>[C:1]([O:2])",
                max_products=2,
            ),
            *_reaction_product_variants(
                target_canonical,
                "[OX2H:1]>>[O:1]C(C)=O",
                max_products=2,
            ),
        ]
        for precursor in variants:
            hints.append(
                _precursor_hint(
                    target_smiles=target_canonical,
                    precursor_smiles=precursor,
                    retron_type=retron_type,
                    application_id=application_id,
                    precursor_role="same_core_redox_or_protection_state_precursor",
                    hypothesis_type=str(hypothesis.get("hypothesis_type") or "same_core_late_stage_functionalization"),
                    risk_flags=[*risk_flags, "same_core_similarity_not_route_proof"],
                )
            )
    return _dedupe_precursor_hints(hints, target_canonical=target_canonical, limit=8)


def _visual_precursor_hints_from_template(
    template: dict[str, Any],
    *,
    target_smiles: str,
    retron_type: str,
    application_id: str,
) -> list[dict[str, Any]]:
    hint = dict(template.get("visual_connectivity_hint") or {})
    precursor = _canonical_smiles(str(hint.get("precursor_smiles") or ""))
    target_canonical = _canonical_smiles(target_smiles)
    if not precursor or precursor == target_canonical:
        return []
    row = _precursor_hint(
        target_smiles=target_canonical,
        precursor_smiles=precursor,
        retron_type=retron_type,
        application_id=application_id,
        precursor_role="visual_same_core_connectivity_precursor",
        hypothesis_type="visual_connectivity_approximation",
        risk_flags=[
            "visual_connectivity_approximation",
            "stereochemistry_unspecified_or_partial",
            "not_literature_exact_row",
        ],
    )
    row.update(
        {
            "source_ref": str(hint.get("source_ref") or ""),
            "source_locator": str(hint.get("source_locator") or ""),
            "visual_product_smiles": str(hint.get("product_smiles") or ""),
            "stereochemistry_status": str(hint.get("stereochemistry_status") or "unspecified_or_partial"),
            "evidence_class": "visual_connectivity_approximation",
        }
    )
    return [row]


def _precursor_hint(
    *,
    target_smiles: str,
    precursor_smiles: str,
    retron_type: str,
    application_id: str,
    precursor_role: str,
    hypothesis_type: str,
    risk_flags: list[str],
) -> dict[str, Any]:
    precursor = _canonical_smiles(precursor_smiles)
    return {
        "schema_version": "analogical_hypothesis_precursor_hint.v1",
        "hint_id": f"hyp_precursor:{_safe_id(retron_type)}:{_short_hash(application_id + ':' + precursor)}",
        "target_smiles": target_smiles,
        "precursor_smiles": precursor,
        "precursor_role": precursor_role,
        "derived_from_retron": retron_type,
        "hypothesis_type": hypothesis_type,
        "candidate_kind": "same_core_redox_or_protection_state_precursor",
        "allowed_use": "guided_search_subgoal_hint_only",
        "risk_flags": _dedupe([*risk_flags, "hypothesis_only_not_literature_exact"]),
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "no_solved_claim": True,
    }


def _enone_saturated_ketone_variants(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    variants: list[str] = []
    patterns = [
        (Chem.MolFromSmarts("[#6]=[#6]-[CX3](=O)[#6]"), (0, 1)),
        (Chem.MolFromSmarts("[CX3](=O)-[#6]=[#6]"), (2, 3)),
    ]
    for query, bond_indices in patterns:
        if query is None:
            continue
        for match in mol.GetSubstructMatches(query):
            atom_a = int(match[bond_indices[0]])
            atom_b = int(match[bond_indices[1]])
            rw_mol = Chem.RWMol(mol)
            bond = rw_mol.GetBondBetweenAtoms(atom_a, atom_b)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            bond.SetBondType(Chem.BondType.SINGLE)
            candidate = rw_mol.GetMol()
            try:
                Chem.SanitizeMol(candidate)
            except Exception:
                continue
            canonical = Chem.MolToSmiles(candidate, isomericSmiles=True)
            if canonical:
                variants.append(canonical)
    return _dedupe(variants)


def _reaction_product_variants(smiles: str, reaction_smarts: str, *, max_products: int) -> list[str]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    try:
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
    except Exception:
        return []
    variants: list[str] = []
    for product_tuple in reaction.RunReactants((mol,)):
        if not product_tuple:
            continue
        product = product_tuple[0]
        try:
            Chem.SanitizeMol(product)
        except Exception:
            continue
        canonical = Chem.MolToSmiles(product, isomericSmiles=True)
        if canonical:
            variants.append(canonical)
        if len(_dedupe(variants)) >= max(1, int(max_products or 1)):
            break
    return _dedupe(variants)[: max(1, int(max_products or 1))]


def _dedupe_precursor_hints(
    rows: list[dict[str, Any]],
    *,
    target_canonical: str,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = {target_canonical}
    for row in rows:
        precursor = _canonical_smiles(str(row.get("precursor_smiles") or ""))
        if not precursor or precursor in seen:
            continue
        seen.add(precursor)
        row = dict(row)
        row["precursor_smiles"] = precursor
        out.append(row)
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def _target_reaction_center_summary(mol: Chem.Mol | None) -> dict[str, Any]:
    if mol is None:
        return {
            "polycyclic_core_like": False,
            "ring_count": 0,
            "carbonyl_count": 0,
            "enone_count": 0,
            "alcohol_count": 0,
            "primary_alcohol_count": 0,
        }
    return {
        "polycyclic_core_like": int(mol.GetRingInfo().NumRings()) >= 4,
        "ring_count": int(mol.GetRingInfo().NumRings()),
        "carbonyl_count": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[#6]"))),
        "enone_count": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]=[#6]-[CX3](=O)[#6]"))) + len(
            mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)-[#6]=[#6]"))
        ),
        "alcohol_count": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX4][OX2H]"))),
        "primary_alcohol_count": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CH2][OX2H]"))),
    }


def _hypothesis_for_retron(retron_type: str, *, center_summary: dict[str, Any]) -> dict[str, Any]:
    common = {
        "schema_version": "analogical_route_hypothesis.v1",
        "route_status": "hypothesis_only_not_solved",
        "evidence_class": "reaction_center_template_analogy",
        "allowed_use": "planner_priority_and_guided_search_hint_only",
        "must_preserve": ["polycyclic_steroid_like_core", "configured_ring_junctions"],
        "reaction_center_summary": dict(center_summary),
        "required_verification": [
            "substructure_match_audit",
            "condition_compatibility_review",
            "product_reconstruction",
            "route_verifier",
            "parent_route_proof",
        ],
        "risk_flags": ["broad_template_scope", "selectivity_not_proven", "not_literature_exact_row"],
        "no_solved_claim": True,
    }
    if retron_type == "steroid_core_retention_bridge":
        return {
            **common,
            "hypothesis_type": "same_core_late_stage_functionalization",
            "reaction_center_idea": "treat analog steroid literature as evidence that the tetracyclic core should be preserved and peripheral oxidation/protection states explored",
            "expected_precursor_type": "same-core advanced steroid intermediate with fewer or differently masked oxygenated handles",
            "template_application": "search around same-core precursors before considering de novo ring construction",
        }
    if retron_type == "steroid_enone_redox_adjustment":
        return {
            **common,
            "hypothesis_type": "enone_redox_or_unmasking",
            "reaction_center_idea": "view the target enone as accessible from an allylic alcohol, saturated ketone, or protected enone precursor in an analog steroid route",
            "expected_precursor_type": "same-core steroid with masked/shifted enone oxidation state",
            "template_application": "generate target-proximal precursors by relaxing the enone to allylic alcohol or saturated ketone motifs",
        }
    if retron_type == "steroid_enone_alcohol_adjustment":
        return {
            **common,
            "hypothesis_type": "enone_or_alcohol_state_adjustment",
            "reaction_center_idea": "use analog steroid literature as support for interconverting enone, allylic alcohol, and protected alcohol states while retaining the core",
            "expected_precursor_type": "same-core steroid with an enone/alcohol handle at a compatible oxidation or protection state",
            "template_application": "generate same-core redox/protection variants around the enone/alcohol handles",
        }
    if retron_type == "steroid_visual_unsaturation_adjustment":
        return {
            **common,
            "hypothesis_type": "visual_same_core_unsaturation_adjustment",
            "reaction_center_idea": "use the visually observed same-core precursor as a connectivity-only hypothesis for late-stage unsaturation or redox adjustment",
            "expected_precursor_type": "same-core steroid precursor from visual extraction, with stereochemistry treated as unresolved",
            "template_application": "pass the achiral/connectivity precursor to guided search as a preferred subgoal, not as exact proof",
        }
    if retron_type == "steroid_carbonyl_redox_adjustment":
        return {
            **common,
            "hypothesis_type": "carbonyl_redox_adjustment",
            "reaction_center_idea": "transfer analog steroid oxidation/reduction logic for secondary alcohol to ketone interconversion",
            "expected_precursor_type": "same-core hydroxy steroid or protected alcohol precursor at the carbonyl-bearing position",
            "template_application": "prefer late-stage redox variants over distant scaffold changes",
        }
    if retron_type == "steroid_alcohol_protection_redox_adjustment":
        return {
            **common,
            "hypothesis_type": "alcohol_protection_or_redox_adjustment",
            "reaction_center_idea": "use analog protection/deprotection and redox-state management to expose the target alcohol pattern late",
            "expected_precursor_type": "same-core protected alcohol, primary alcohol surrogate, or oxidized aldehyde/ester-level precursor",
            "template_application": "search same-core precursors that differ only in alcohol protection or oxidation state",
        }
    if retron_type in SOURCE_GROUNDED_VISUAL_RETRONS:
        return {
            **common,
            "must_preserve": ["target_core_or_named_visual_scaffold"],
            "hypothesis_type": "source_grounded_visual_connectivity_hint",
            "reaction_center_idea": "use the visually extracted literature step as a connectivity-only template for guided search",
            "expected_precursor_type": "visual source precursor or named intermediate requiring verifier review",
            "template_application": "pass the visual precursor to guided search as a preferred subgoal, not as exact proof",
            "risk_flags": ["visual_connectivity_approximation", "not_literature_exact_row", "requires_verifier"],
        }
    return {
        **common,
        "hypothesis_type": "generic_reaction_center_transfer",
        "reaction_center_idea": "apply broad reaction-center analogy under verifier control",
        "expected_precursor_type": "target-proximal analog precursor",
        "template_application": "planner hint only",
    }


def _literature_card_from_analogical_template(template: dict[str, Any]) -> LiteratureTemplateCard:
    center = dict(template.get("reaction_center") or {})
    retron_type = str(center.get("product_retron_type") or "")
    return LiteratureTemplateCard(
        template_id=str(template.get("template_id") or ""),
        evidence_refs=[str(item) for item in template.get("evidence_refs") or template.get("source_refs") or []],
        reaction_class=str(template.get("reaction_class") or retron_type),
        template_level=LiteratureTemplateLevel.EXECUTABLE_TEMPLATE_CANDIDATE.value,
        product_retron={
            "retron_type": retron_type,
            "description": f"analogical retron for {retron_type}",
            "smarts": str(center.get("smarts") or retron_type),
        },
        break_bonds=[{"role": retron_type, "source": "analogical_reaction_template"}],
        precursor_roles=["target_side_fragment", "leaving_or_coupling_partner"],
        applicability={
            "source": "analogical_reaction_template",
            "scope_gap": str(template.get("scope_gap") or ""),
            "direct_one_step_consumption": True,
        },
        scope_limits=[str(template.get("scope_gap") or "analog scope gap")],
        safety_flags=["analog_template_requires_verifier", "no_solved_claim"],
        promotion_status="candidate",
        source_family=str(template.get("mechanistic_class") or ""),
        condition_source="literature_analog",
        not_raw_reaction_injection=True,
    )


def _source_refs(blackboard: dict[str, Any]) -> list[str]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    refs = [str(item) for item in evidence.get("source_refs") or [] if str(item or "").strip()]
    for row in evidence.get("source_candidates") or []:
        if isinstance(row, dict):
            ref = str(row.get("source_ref") or row.get("doi") or row.get("url") or row.get("local_pdf") or "")
            if ref:
                refs.append(ref)
    return _dedupe(refs)


def _dedupe_templates(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for template in templates:
        key = str((template.get("reaction_center") or {}).get("product_retron_type") or template.get("template_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(template)
    return out


def _preserve_substructures(target_profile: dict[str, Any]) -> list[str]:
    handles = [str(item) for item in target_profile.get("functional_handles") or []]
    out = []
    if int(target_profile.get("rings") or 0) >= 3:
        out.append("polycyclic_core")
    if any("steroid" in item.lower() or "natural_product" in item.lower() for item in handles):
        out.append("target_core")
    return out or ["target_core_if_present"]


def _risk_flags(retron_type: str) -> list[str]:
    flags = ["analog_scope_gap", "requires_verifier"]
    if retron_type in {"tertiary_amine_state_adjustment", "imide_or_succinimide_n_acyl"}:
        flags.append("advisory_only_until_matcher_available")
    if retron_type == "aryl_ester_acyl_oxygen":
        flags.append("steric_or_acyl_migration_risk")
    if retron_type == "steroid_visual_unsaturation_adjustment":
        flags.extend(["visual_connectivity_approximation", "stereochemistry_unresolved"])
    if retron_type in SOURCE_GROUNDED_VISUAL_RETRONS:
        flags.extend(["visual_connectivity_approximation", "source_grounded_visual_template", "stereochemistry_unresolved"])
    return flags


def _default_radius(retron_type: str, *, radius_policy: str) -> str:
    policy = str(radius_policy or "auto")
    if policy == "local":
        return "r1"
    if policy == "broad":
        return "r0"
    if retron_type in {"aryl_ester_acyl_oxygen", "bufadienolide_c17_pyrone"}:
        return "r2"
    return "r1"


def _template_id(case_id: str, retron_type: str, index: int) -> str:
    digest = _short_hash(f"{case_id}:{retron_type}:{index}")
    return f"analog_tpl_{_safe_id(retron_type)}_{digest}"


def _rejected_application(template: dict[str, Any], target_smiles: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": ANALOGICAL_TEMPLATE_APPLICATION_SCHEMA,
        "application_id": f"apply:{template.get('template_id')}:{_short_hash(target_smiles)}",
        "template_id": str(template.get("template_id") or ""),
        "target_smiles": target_smiles,
        "product_retron_type": str((template.get("reaction_center") or {}).get("product_retron_type") or ""),
        "accepted": False,
        "allowed_use": "forbidden",
        "cut_fragments": [],
        "reasons": [str(item) for item in reasons],
        "no_solved_claim": True,
    }


def _failure_row(template: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "agent_template_failure_memory.v1",
        "template_id": str(template.get("template_id") or ""),
        "product_retron_type": str((template.get("reaction_center") or {}).get("product_retron_type") or ""),
        "failure_count": 1,
        "reasons": [str(item) for item in reasons],
    }


def _failure_reasons(blackboard: dict[str, Any]) -> set[str]:
    return {
        str(row.get("reason") or "")
        for row in blackboard.get("route_failures") or []
        if isinstance(row, dict)
    }


def _confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "medium_high": 3, "high": 4}.get(str(value or "low"), 1)


def _contains_raw_reaction_payload(value: Any) -> bool:
    return contains_raw_reaction_payload(value)


def _has_substructure(mol: Chem.Mol, smarts: str) -> bool:
    query = Chem.MolFromSmarts(str(smarts or ""))
    return bool(query is not None and mol.HasSubstructMatch(query))


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _short_hash(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:10]


def _safe_id(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    return "_".join(part for part in safe.split("_") if part) or "template"


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
