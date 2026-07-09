"""Step-level enzyme audit rows for ChemEnzy route outputs.

The audit rows separate three concepts that are otherwise easy to mix up:
proposal source, post-hoc enzyme/EC annotation, and lightweight material sanity.
They are intended as training/debug data for search-time enzyme routing and
selection, not as final chemistry validation.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from cascade_planner.baselines.chem_enzy_native_enzyme_plugin import PLUGIN_MODEL_FULL_NAME
from cascade_planner.baselines.enzyme_step_enhancement import (
    EnzymeStepEnhancementConfig,
    evaluate_step_enhancement,
)
from cascade_planner.baselines.chem_enzy_step_quality import evaluate_enzyme_step_quality
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate, RouteStepCandidate
from cascade_planner.baselines.route_plausibility import audit_step_plausibility


SCHEMA_VERSION = "chem_enzy_enzyme_step_audit.v1"

ENZYMATIC_SOURCE_TOKENS = (
    "enzyme",
    "enzymatic",
    "bionav",
    "bkms",
    "biocatalysis",
    "ecreact",
    "ec_",
)
CHEMICAL_SOURCE_TOKENS = (
    "uspto",
    "graphfp",
    "reaxys",
    "pistachio",
    "template",
    "mlp",
)


def audit_baseline_results(
    results: Iterable[BaselineRunResult],
    *,
    target_metadata: dict[str, dict[str, Any]] | None = None,
    enable_step_enhancement: bool = False,
    step_enhancement_config: EnzymeStepEnhancementConfig | None = None,
    step_enhancement_scorer: Any | None = None,
) -> list[dict[str, Any]]:
    """Return one JSON-safe audit row per route step."""
    rows: list[dict[str, Any]] = []
    metadata = target_metadata or {}
    for result in results:
        target_meta = metadata.get(result.target_smiles) or {}
        rows.extend(
            audit_result_steps(
                result,
                target_meta=target_meta,
                enable_step_enhancement=enable_step_enhancement,
                step_enhancement_config=step_enhancement_config,
                step_enhancement_scorer=step_enhancement_scorer,
            )
        )
    return rows


def audit_result_steps(
    result: BaselineRunResult,
    *,
    target_meta: dict[str, Any] | None = None,
    enable_step_enhancement: bool = False,
    step_enhancement_config: EnzymeStepEnhancementConfig | None = None,
    step_enhancement_scorer: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in result.routes:
        rows.extend(
            audit_route_steps(
                result,
                route,
                target_meta=target_meta or {},
                enable_step_enhancement=enable_step_enhancement,
                step_enhancement_config=step_enhancement_config,
                step_enhancement_scorer=step_enhancement_scorer,
            )
        )
    return rows


def audit_route_steps(
    result: BaselineRunResult,
    route: RouteCandidate,
    *,
    target_meta: dict[str, Any] | None = None,
    enable_step_enhancement: bool = False,
    step_enhancement_config: EnzymeStepEnhancementConfig | None = None,
    step_enhancement_scorer: Any | None = None,
) -> list[dict[str, Any]]:
    target_meta = target_meta or {}
    route_domains = [proposal_domain_for_step(step) for step in route.steps]
    route_has_enzyme_source = any(domain == "enzymatic" for domain in route_domains)
    route_has_plugin = any(is_plugin_step(step) for step in route.steps)
    route_has_posthoc_enzyme = any(posthoc_reaction_type(step).get("is_enzymatic") for step in route.steps)
    rows = []
    for idx, step in enumerate(route.steps):
        previous_step = route.steps[idx - 1] if idx > 0 else None
        next_step = route.steps[idx + 1] if idx + 1 < len(route.steps) else None
        rows.append(
            audit_step_row(
                result=result,
                route=route,
                step=step,
                step_index=idx,
                target_meta=target_meta,
                previous_step=previous_step,
                next_step=next_step,
                route_has_enzyme_source=route_has_enzyme_source,
                route_has_posthoc_enzyme=route_has_posthoc_enzyme,
                route_has_plugin=route_has_plugin,
                enable_step_enhancement=enable_step_enhancement,
                step_enhancement_config=step_enhancement_config,
                step_enhancement_scorer=step_enhancement_scorer,
            )
        )
    return rows


def audit_step_row(
    *,
    result: BaselineRunResult,
    route: RouteCandidate,
    step: RouteStepCandidate,
    step_index: int,
    target_meta: dict[str, Any],
    previous_step: RouteStepCandidate | None,
    next_step: RouteStepCandidate | None,
    route_has_enzyme_source: bool,
    route_has_posthoc_enzyme: bool,
    route_has_plugin: bool,
    enable_step_enhancement: bool = False,
    step_enhancement_config: EnzymeStepEnhancementConfig | None = None,
    step_enhancement_scorer: Any | None = None,
) -> dict[str, Any]:
    attrs = step.raw_backend_metadata.get("rxn_attribute") if isinstance(step.raw_backend_metadata, dict) else {}
    attrs = attrs if isinstance(attrs, dict) else {}
    template = step.raw_backend_metadata.get("template") if isinstance(step.raw_backend_metadata, dict) else None
    template_payload = template if isinstance(template, dict) else {}
    material = audit_step_plausibility(step)
    rxn_type = posthoc_reaction_type(step)
    ec_rows = ec_assignment_rows(step)
    top_ec = ec_rows[0] if ec_rows else {}
    source_domain = proposal_domain_for_step(step)
    template_evidence = template_payload.get("evidence") if isinstance(template_payload.get("evidence"), dict) else {}
    sp_payload = (
        template_payload.get("enzyme_sp_verifier_v1")
        if isinstance(template_payload.get("enzyme_sp_verifier_v1"), dict)
        else {}
    )
    template_quality_payload = (
        template_payload.get("autoplanner_enzyme_quality_v1")
        if isinstance(template_payload.get("autoplanner_enzyme_quality_v1"), dict)
        else {}
    )
    cascade_cost = (
        step.raw_backend_metadata.get("cascade_cost")
        if isinstance(step.raw_backend_metadata, dict) and isinstance(step.raw_backend_metadata.get("cascade_cost"), dict)
        else {}
    )
    cascade_material = cascade_cost.get("material_sanity") if isinstance(cascade_cost.get("material_sanity"), dict) else {}
    cascade_components = cascade_cost.get("components") if isinstance(cascade_cost.get("components"), dict) else {}
    quality_payload = template_quality_payload or derived_enzyme_quality_payload(
        step=step,
        source_domain=source_domain,
        posthoc_is_enzymatic=bool(rxn_type.get("is_enzymatic")),
        template_payload=template_payload,
        sp_payload=sp_payload,
        ec_rows=ec_rows,
        cascade_cost=cascade_cost,
    )
    flags = weakness_flags(
        step=step,
        material=material,
        source_domain=source_domain,
        posthoc_is_enzymatic=bool(rxn_type.get("is_enzymatic")),
        posthoc_is_organic=bool(rxn_type.get("is_organic")),
        ec_rows=ec_rows,
        quality_payload=quality_payload,
    )
    enhancement_payload = (
        evaluate_step_enhancement(
            step,
            scorer=step_enhancement_scorer,
            config=step_enhancement_config,
        )
        if enable_step_enhancement
        else {}
    )
    best_enhancement = enhancement_payload.get("best_candidate") if isinstance(enhancement_payload.get("best_candidate"), dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "target_name": target_meta.get("name") or target_meta.get("safe") or "",
        "target_safe": target_meta.get("safe") or "",
        "target_smiles": result.target_smiles,
        "backend": result.backend,
        "result_solved": bool(result.solved),
        "route_rank": int(route.route_rank),
        "route_solved": bool(route.solved),
        "route_score": route.score,
        "route_step_count": len(route.steps),
        "route_has_enzyme_source": route_has_enzyme_source,
        "route_has_posthoc_enzyme": route_has_posthoc_enzyme,
        "route_has_plugin_injected": route_has_plugin,
        "step_index": int(step_index),
        "step_position": step_position(step_index, len(route.steps)),
        "rxn_smiles": step.rxn_smiles,
        "product_smiles": step.product_smiles,
        "reactant_smiles": list(step.reactant_smiles),
        "reactant_count": len(step.reactant_smiles),
        "source_model": step.source_model,
        "proposal_domain": source_domain,
        "proposal_source_kind": proposal_source_kind(step),
        "generated_by_enzyme_like_source": source_domain == "enzymatic",
        "generated_by_chemical_like_source": source_domain == "chemical",
        "plugin_injected": is_plugin_step(step),
        "previous_source_model": previous_step.source_model if previous_step else "",
        "next_source_model": next_step.source_model if next_step else "",
        "previous_proposal_domain": proposal_domain_for_step(previous_step) if previous_step else "",
        "next_proposal_domain": proposal_domain_for_step(next_step) if next_step else "",
        "posthoc_reaction_type": rxn_type.get("reaction_type") or "",
        "posthoc_reaction_type_confidence": rxn_type.get("confidence"),
        "posthoc_classified_enzymatic": bool(rxn_type.get("is_enzymatic")),
        "posthoc_classified_organic": bool(rxn_type.get("is_organic")),
        "ec_numbers": [row.get("ec_number") for row in ec_rows if row.get("ec_number")],
        "ec_confidences": [row.get("confidence") for row in ec_rows if row.get("confidence") is not None],
        "ec_top1": top_ec.get("ec_number"),
        "ec_top1_confidence": top_ec.get("confidence"),
        "has_ec_annotation": bool(ec_rows),
        "condition_count": len(step.condition_predictions),
        "condition_predictions": list(step.condition_predictions),
        "material_audit_passed": bool(material.get("passed")),
        "material_audit_reasons": list(material.get("reasons") or []),
        "heavy_atom_gain": material.get("heavy_atom_gain"),
        "carbon_gain": material.get("carbon_gain"),
        "hetero_atom_gain": material.get("hetero_atom_gain"),
        "unexplained_element_gains": material.get("unexplained_element_gains") or {},
        "template_summary": template_summary(template),
        "template_ec": template_payload.get("ec"),
        "template_autoplanner_plugin": bool(template_payload.get("autoplanner_native_enzyme_plugin")),
        "template_evidence_keys": sorted(str(key) for key in template_evidence.keys()),
        "template_transition_signature": template_evidence.get("transition_signature"),
        "sp_v1_score": sp_payload.get("score"),
        "sp_v1_threshold": sp_payload.get("threshold"),
        "sp_v1_accepted": sp_payload.get("accepted"),
        "enzyme_quality_origin": "template" if template_quality_payload else "derived" if quality_payload else "",
        "enzyme_quality_score": quality_payload.get("quality_score"),
        "enzyme_quality_decision": quality_payload.get("decision") or "",
        "enzyme_quality_flags": list(quality_payload.get("flags") or []),
        "enzyme_quality_material_passed": (quality_payload.get("material_sanity") or {}).get("passed")
        if isinstance(quality_payload.get("material_sanity"), dict)
        else None,
        "cascade_cost_adjustment": cascade_cost.get("cascade_adjustment"),
        "cascade_cost_total": cascade_cost.get("total_cost"),
        "cascade_cost_material_component": cascade_components.get("material_sanity"),
        "cascade_cost_enzyme_evidence_component": cascade_components.get("enzyme_evidence"),
        "cascade_cost_material_passed": cascade_material.get("passed"),
        "cascade_cost_material_reasons": list(cascade_material.get("reasons") or []),
        "enzyme_step_enhancement_available": bool(enhancement_payload.get("available")),
        "enzyme_step_enhancement_kind": enhancement_payload.get("recommended_kind") or "",
        "enzyme_step_enhancement_reasons": list(enhancement_payload.get("reasons") or []),
        "enzyme_step_enhancement_candidate_count": enhancement_payload.get("candidate_count"),
        "enzyme_step_enhancement_viable_candidate_count": enhancement_payload.get("viable_candidate_count"),
        "enzyme_step_enhancement_best_score": best_enhancement.get("efficiency_score"),
        "enzyme_step_enhancement_best_quality_score": best_enhancement.get("quality_score"),
        "enzyme_step_enhancement_best_ec": best_enhancement.get("ec"),
        "enzyme_step_enhancement_best_main_reactant": best_enhancement.get("main_reactant"),
        "enzyme_step_enhancement_best_sp_v1_score": best_enhancement.get("sp_v1_score"),
        "enzyme_step_enhancement_top_candidates": list(enhancement_payload.get("top_candidates") or []),
        "weakness_flags": flags,
        "weakness_count": len(flags),
        "raw_attribute_keys": sorted(str(key) for key in attrs.keys()),
    }


def derived_enzyme_quality_payload(
    *,
    step: RouteStepCandidate,
    source_domain: str,
    posthoc_is_enzymatic: bool,
    template_payload: dict[str, Any],
    sp_payload: dict[str, Any],
    ec_rows: list[dict[str, Any]],
    cascade_cost: dict[str, Any],
) -> dict[str, Any]:
    """Build a visible quality row for native ChemEnzy enzyme-like steps.

    Plugin candidates carry quality at search time.  Native BioNav/BKMS-style
    steps generally do not, so this derived row makes the missing evidence
    explicit in audit/web outputs while preserving the selected route.
    """
    if source_domain != "enzymatic" and not posthoc_is_enzymatic:
        return {}
    evidence = {}
    if isinstance(template_payload.get("evidence"), dict):
        evidence.update(template_payload.get("evidence") or {})
    if cascade_cost:
        evidence.setdefault("cascade_cost_available", True)
    ec_numbers = [str(row.get("ec_number") or "") for row in ec_rows if row.get("ec_number")]
    quality = evaluate_enzyme_step_quality(
        product_smiles=step.product_smiles,
        reactants=step.reactant_smiles,
        source_model=step.source_model,
        template={
            "model_full_name": step.source_model,
            "source": step.source_model,
            "evidence": evidence,
            "enzyme_sp_verifier_v1": sp_payload,
        },
        sp_payload=sp_payload,
        ec_numbers=ec_numbers,
    )
    flags = list(quality.get("flags") or [])
    if source_domain == "enzymatic" and "native_or_posthoc_derived_quality" not in flags:
        flags.append("native_or_posthoc_derived_quality")
    if cascade_cost and "cascade_costed_during_search" not in flags:
        flags.append("cascade_costed_during_search")
    quality["flags"] = flags
    quality["origin"] = "derived_from_selected_step"
    quality["search_time_costed"] = bool(cascade_cost)
    return quality


def summarize_enzyme_step_audit(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    targets = {row.get("target_smiles") for row in row_list if row.get("target_smiles")}
    routes = {
        (row.get("target_smiles"), row.get("route_rank"))
        for row in row_list
        if row.get("target_smiles") is not None and row.get("route_rank") is not None
    }
    flag_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    for row in row_list:
        flag_counts.update(str(flag) for flag in row.get("weakness_flags") or [])
        source_counts.update([str(row.get("source_model") or "")])
        domain_counts.update([str(row.get("proposal_domain") or "unknown")])
    return {
        "schema_version": f"{SCHEMA_VERSION}.summary",
        "targets": len(targets),
        "routes": len(routes),
        "steps": len(row_list),
        "enzyme_like_source_steps": sum(1 for row in row_list if row.get("generated_by_enzyme_like_source")),
        "chemical_like_source_steps": sum(1 for row in row_list if row.get("generated_by_chemical_like_source")),
        "posthoc_enzymatic_steps": sum(1 for row in row_list if row.get("posthoc_classified_enzymatic")),
        "ec_annotated_steps": sum(1 for row in row_list if row.get("has_ec_annotation")),
        "plugin_injected_steps": sum(1 for row in row_list if row.get("plugin_injected")),
        "enzyme_quality_scored_steps": sum(1 for row in row_list if row.get("enzyme_quality_score") is not None),
        "enzyme_quality_passed_steps": sum(1 for row in row_list if row.get("enzyme_quality_decision") == "pass"),
        "enzyme_quality_warned_steps": sum(1 for row in row_list if row.get("enzyme_quality_decision") == "warn"),
        "enzyme_quality_rejected_steps": sum(1 for row in row_list if row.get("enzyme_quality_decision") == "reject"),
        "search_time_quality_scored_steps": sum(
            1
            for row in row_list
            if row.get("enzyme_quality_score") is not None and row.get("enzyme_quality_origin") == "template"
        ),
        "search_time_quality_passed_steps": sum(
            1
            for row in row_list
            if row.get("enzyme_quality_decision") == "pass" and row.get("enzyme_quality_origin") == "template"
        ),
        "search_time_quality_warned_steps": sum(
            1
            for row in row_list
            if row.get("enzyme_quality_decision") == "warn" and row.get("enzyme_quality_origin") == "template"
        ),
        "search_time_quality_rejected_steps": sum(
            1
            for row in row_list
            if row.get("enzyme_quality_decision") == "reject" and row.get("enzyme_quality_origin") == "template"
        ),
        "derived_quality_scored_steps": sum(1 for row in row_list if row.get("enzyme_quality_origin") == "derived"),
        "step_enhancement_available_steps": sum(1 for row in row_list if row.get("enzyme_step_enhancement_available")),
        "missing_enzyme_step_opportunities": sum(
            1 for row in row_list if row.get("enzyme_step_enhancement_kind") == "missing_enzyme_step"
        ),
        "wrong_enzyme_step_replacements": sum(
            1 for row in row_list if row.get("enzyme_step_enhancement_kind") == "wrong_enzyme_step_replacement"
        ),
        "efficient_enzyme_step_upgrades": sum(
            1 for row in row_list if row.get("enzyme_step_enhancement_kind") == "efficient_enzyme_step_upgrade"
        ),
        "step_enhancement_viable_candidates": sum(
            int(row.get("enzyme_step_enhancement_viable_candidate_count") or 0) for row in row_list
        ),
        "cascade_costed_steps": sum(1 for row in row_list if row.get("cascade_cost_total") is not None),
        "material_failed_steps": sum(1 for row in row_list if not row.get("material_audit_passed")),
        "enzyme_like_material_failed_steps": sum(
            1 for row in row_list if row.get("generated_by_enzyme_like_source") and not row.get("material_audit_passed")
        ),
        "posthoc_enzymatic_on_chemical_source_steps": sum(
            1
            for row in row_list
            if row.get("posthoc_classified_enzymatic") and row.get("proposal_domain") == "chemical"
        ),
        "enzyme_source_without_ec_steps": sum(
            1
            for row in row_list
            if row.get("generated_by_enzyme_like_source") and not row.get("has_ec_annotation")
        ),
        "weakness_flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "proposal_domain_counts": dict(sorted(domain_counts.items())),
        "source_model_counts": dict(sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def posthoc_reaction_type(step: RouteStepCandidate) -> dict[str, Any]:
    attrs = step.raw_backend_metadata.get("rxn_attribute") if isinstance(step.raw_backend_metadata, dict) else {}
    rows = records_from_backend_table((attrs or {}).get("organic_enzyme_rxn_classification"))
    if not rows:
        return {"reaction_type": "", "confidence": None, "is_enzymatic": False, "is_organic": False}
    row = rows[0]
    reaction_type = str(
        row.get("Reaction Type")
        or row.get("reaction_type")
        or row.get("type")
        or row.get("class")
        or ""
    )
    normalized = reaction_type.lower()
    return {
        "reaction_type": reaction_type,
        "confidence": float_or_none(row.get("Confidence") or row.get("confidence")),
        "is_enzymatic": "enzymatic" in normalized,
        "is_organic": "organic" in normalized,
    }


def ec_assignment_rows(step: RouteStepCandidate) -> list[dict[str, Any]]:
    rows = []
    for item in step.enzyme_ec_annotations:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "rank": item.get("rank"),
                "ec_number": item.get("ec_number"),
                "confidence": float_or_none(item.get("confidence")),
                "raw": item.get("raw") if isinstance(item.get("raw"), dict) else {},
            }
        )
    if rows:
        return rows
    attrs = step.raw_backend_metadata.get("rxn_attribute") if isinstance(step.raw_backend_metadata, dict) else {}
    for item in records_from_backend_table((attrs or {}).get("enzyme_assign")):
        rows.append(
            {
                "rank": item.get("Ranks") or item.get("rank"),
                "ec_number": item.get("EC Number") or item.get("ec_number"),
                "confidence": float_or_none(item.get("Confidence") or item.get("confidence")),
                "raw": item,
            }
        )
    return rows


def proposal_domain_for_step(step: RouteStepCandidate | None) -> str:
    if step is None:
        return ""
    cascade_cost = (step.raw_backend_metadata or {}).get("cascade_cost")
    cascade_cost = cascade_cost if isinstance(cascade_cost, dict) else {}
    text = " ".join(
        [
            str(step.source_model or ""),
            str(cascade_cost.get("source_model") or ""),
            str(cascade_cost.get("reaction_domain") or ""),
            json.dumps(template_summary((step.raw_backend_metadata or {}).get("template")), sort_keys=True),
        ]
    ).lower()
    if any(token in text for token in ENZYMATIC_SOURCE_TOKENS):
        return "enzymatic"
    if any(token in text for token in CHEMICAL_SOURCE_TOKENS):
        return "chemical"
    return "unknown"


def proposal_source_kind(step: RouteStepCandidate) -> str:
    if is_plugin_step(step):
        return "autoplanner_plugin"
    domain = proposal_domain_for_step(step)
    source = str(step.source_model or "").lower()
    if domain == "enzymatic":
        if "bionav" in source:
            return "native_bionav"
        if "bkms" in source:
            return "native_bkms_template"
        if "biocatalysis" in source:
            return "native_biocatalysis_template"
        return "native_enzyme_like"
    if domain == "chemical":
        return "native_chemical_like"
    return "unknown"


def is_plugin_step(step: RouteStepCandidate) -> bool:
    return str(step.source_model or "") == PLUGIN_MODEL_FULL_NAME


def weakness_flags(
    *,
    step: RouteStepCandidate,
    material: dict[str, Any],
    source_domain: str,
    posthoc_is_enzymatic: bool,
    posthoc_is_organic: bool,
    ec_rows: list[dict[str, Any]],
    quality_payload: dict[str, Any] | None = None,
) -> list[str]:
    flags: list[str] = []
    quality_payload = quality_payload if isinstance(quality_payload, dict) else {}
    if not material.get("passed"):
        flags.append("material_audit_failed")
    if source_domain == "enzymatic" and not ec_rows:
        flags.append("enzyme_like_source_without_ec")
    if source_domain == "chemical" and posthoc_is_enzymatic:
        flags.append("posthoc_enzymatic_on_chemical_source")
    if source_domain == "enzymatic" and posthoc_is_organic:
        flags.append("enzyme_like_source_classified_organic")
    if ec_rows and not posthoc_is_enzymatic and not is_plugin_step(step):
        flags.append("ec_annotation_without_enzymatic_classification")
    if is_plugin_step(step) and not material.get("passed"):
        flags.append("plugin_injected_material_failed")
    if quality_payload.get("decision") == "reject":
        flags.append("search_time_enzyme_quality_rejected")
    if "missing_sp_v1" in set(quality_payload.get("flags") or []) and source_domain == "enzymatic":
        flags.append("enzyme_like_source_without_sp_v1")
    top_conf = float_or_none(ec_rows[0].get("confidence")) if ec_rows else None
    if top_conf is not None and top_conf < 0.30:
        flags.append("low_top1_ec_confidence")
    return flags


def step_position(step_index: int, step_count: int) -> str:
    if step_index == 0:
        return "root_disconnection"
    if step_index == step_count - 1:
        return "terminal_disconnection"
    return "internal_disconnection"


def template_summary(template: Any) -> Any:
    if template is None:
        return None
    if isinstance(template, (str, int, float, bool)):
        return template
    if isinstance(template, dict):
        keep = {}
        for key in (
            "model_full_name",
            "model_name",
            "source",
            "ec",
            "reaction_type",
            "autoplanner_native_enzyme_plugin",
        ):
            if key in template:
                keep[key] = template.get(key)
        if "enzyme_sp_verifier_v1" in template and isinstance(template["enzyme_sp_verifier_v1"], dict):
            sp = template["enzyme_sp_verifier_v1"]
            keep["enzyme_sp_verifier_v1"] = {
                "score": sp.get("score"),
                "threshold": sp.get("threshold"),
                "accepted": sp.get("accepted"),
                "ec_numbers": sp.get("ec_numbers"),
            }
        if "autoplanner_enzyme_quality_v1" in template and isinstance(template["autoplanner_enzyme_quality_v1"], dict):
            quality = template["autoplanner_enzyme_quality_v1"]
            keep["autoplanner_enzyme_quality_v1"] = {
                "quality_score": quality.get("quality_score"),
                "decision": quality.get("decision"),
                "flags": quality.get("flags"),
                "material_sanity": quality.get("material_sanity"),
            }
        return keep or {str(key): str(value) for key, value in template.items()}
    return str(template)


def records_from_backend_table(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return records_from_backend_table(json.loads(text))
        except json.JSONDecodeError:
            return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        columns = value.get("columns")
        data = value.get("data")
        if isinstance(columns, list) and isinstance(data, list):
            rows = []
            for row in data:
                if isinstance(row, list):
                    rows.append({str(columns[idx]): row[idx] for idx in range(min(len(columns), len(row)))})
            return rows
        if "data" in value and isinstance(value["data"], list):
            return [item for item in value["data"] if isinstance(item, dict)]
        column_rows = records_from_column_oriented_table(value)
        if column_rows:
            return column_rows
        return [value]
    return []


def records_from_column_oriented_table(value: dict[str, Any]) -> list[dict[str, Any]]:
    if not value or not all(isinstance(col_values, dict) for col_values in value.values()):
        return []
    row_keys = set()
    for col_values in value.values():
        row_keys.update(str(key) for key in col_values)
    rows = []
    for row_key in sorted(row_keys, key=_table_row_sort_key):
        row = {}
        for col_name, col_values in value.items():
            row[str(col_name)] = col_values.get(row_key)
        rows.append(row)
    return rows


def _table_row_sort_key(item: str) -> tuple[int, str]:
    try:
        return (int(item), item)
    except ValueError:
        return (10**9, item)


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
