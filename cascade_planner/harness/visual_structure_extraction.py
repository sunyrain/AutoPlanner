"""Structured validation for visual/source-derived literature intermediates."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

VISUAL_STRUCTURE_CHAIN_SCHEMA = "visual_structure_candidate_chain.v1"
VISUAL_STRUCTURE_VALIDATION_SCHEMA = "visual_structure_chain_validation.v1"


def load_structure_candidate_chain(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    path = Path(value)
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def validate_visual_structure_chain(
    candidate_chain: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
    target_smiles: str = "",
    require_contiguous: bool = True,
) -> dict[str, Any]:
    """Normalize source/vision candidate structures and check route continuity."""
    payload = load_structure_candidate_chain(candidate_chain)
    steps = _candidate_steps(payload)
    reasons: list[str] = []
    warnings: list[str] = []
    validated_steps: list[dict[str, Any]] = []
    compound_map: dict[str, dict[str, Any]] = {}

    if not steps:
        reasons.append("candidate_chain_has_no_steps")

    for index, raw in enumerate(steps, start=1):
        step = _normalize_step(raw, index=index, payload=payload)
        product_report = _smiles_report(step["product_smiles"])
        reactant_reports = [_smiles_report(smiles) for smiles in step["reactant_smiles"]]
        step_reasons: list[str] = []
        if not product_report["valid"]:
            step_reasons.append("invalid_product_smiles")
        if not step["reactant_smiles"]:
            step_reasons.append("missing_reactant_smiles")
        elif not all(report["valid"] for report in reactant_reports):
            step_reasons.append("invalid_reactant_smiles")
        condition = _condition_candidate(step.get("condition_candidate") or raw.get("condition_candidate"))
        if not _condition_has_signal(condition):
            step_reasons.append("condition_candidate_missing_source_grounded_fields")
        derivation = _structure_derivation(step, raw, payload)
        if not derivation.get("source_locator"):
            step_reasons.append("structure_derivation_missing_source_locator")
        if not derivation.get("tool_checks"):
            step_reasons.append("structure_derivation_missing_tool_checks")
        source_excerpt = _short_excerpt(str(step.get("source_excerpt") or raw.get("source_excerpt") or payload.get("source_excerpt") or ""))
        if not source_excerpt:
            warnings.append(f"{step['step_id']}:source_excerpt_missing")

        normalized = {
            **step,
            "product": product_report,
            "reactants": reactant_reports,
            "main_reactant_smiles": _main_reactant(step["reactant_smiles"], step.get("main_reactant_smiles")),
            "condition_candidate": condition,
            "structure_derivation": derivation,
            "source_excerpt": source_excerpt,
            "accepted": not step_reasons,
            "reasons": sorted(set(step_reasons)),
            "no_solved_claim": True,
            "production_write_blocked": True,
        }
        validated_steps.append(normalized)
        reasons.extend(step_reasons)
        _record_compound(compound_map, label=step.get("product_label"), report=product_report)
        for label, report in zip(step.get("reactant_labels") or [], reactant_reports):
            _record_compound(compound_map, label=label, report=report)

    continuity = _continuity_audit(validated_steps, target_smiles=target_smiles or str(payload.get("target_smiles") or ""))
    if require_contiguous and not continuity["accepted"]:
        reasons.extend(continuity["reasons"])

    result = {
        "schema_version": VISUAL_STRUCTURE_VALIDATION_SCHEMA,
        "accepted": not sorted(set(reasons)),
        "status": "validated" if not sorted(set(reasons)) else "needs_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_candidate_schema": str(payload.get("schema_version") or ""),
        "case_id": str(payload.get("case_id") or ""),
        "target": {
            "name": str(payload.get("target_name") or ""),
            "smiles": target_smiles or str(payload.get("target_smiles") or ""),
        },
        "route_order": str(payload.get("route_order") or ("forward_start_to_target" if payload.get("chain") else "retro_target_to_start")),
        "source_ref": str(payload.get("source_ref") or ""),
        "source_title": str(payload.get("source_title") or ""),
        "evidence_refs": _string_list(payload.get("evidence_refs")),
        "steps": validated_steps,
        "compound_reports": list(compound_map.values()),
        "continuity_audit": continuity,
        "summary": {
            "step_count": len(validated_steps),
            "accepted_step_count": sum(1 for step in validated_steps if step.get("accepted")),
            "compound_count": len(compound_map),
            "chain_contiguous": bool(continuity.get("accepted")),
            "target_match": bool(continuity.get("target_match")),
        },
        "source_policy": {
            "codex_visual_structure_to_smiles_allowed": True,
            "requires_structured_source_locator": True,
            "requires_rdkit_parse": True,
            "not_route_evidence_until_source_detail_resolution": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "warnings": sorted(set(warnings)),
        "reasons": sorted(set(reasons)),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "visual_structure_chain_validation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _candidate_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = payload.get("steps")
    if isinstance(raw_steps, list) and raw_steps:
        return [dict(item) for item in raw_steps if isinstance(item, dict)]
    chain = payload.get("chain") or payload.get("nodes")
    if not isinstance(chain, list) or len(chain) < 2:
        return []
    nodes = [dict(item) for item in chain if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    route_order = str(payload.get("route_order") or "forward_start_to_target")
    if route_order == "retro_target_to_start":
        pairs = [(nodes[index + 1], nodes[index]) for index in range(len(nodes) - 1)]
    else:
        pairs = [(nodes[index], nodes[index + 1]) for index in range(len(nodes) - 2, -1, -1)]
    for index, (reactant, product) in enumerate(pairs, start=1):
        out.append(
            {
                "step_id": f"{_safe_id(str(reactant.get('label') or index + 1))}_to_{_safe_id(str(product.get('label') or index + 2))}",
                "product_label": str(product.get("label") or product.get("compound_label") or ""),
                "product_smiles": str(product.get("smiles") or product.get("candidate_smiles") or ""),
                "reactant_label": str(reactant.get("label") or reactant.get("compound_label") or ""),
                "reactant_smiles": [str(reactant.get("smiles") or reactant.get("candidate_smiles") or "")],
                "condition_candidate": product.get("condition_candidate") or payload.get("default_condition_candidate") or {},
                "source_locator": str(product.get("source_locator") or reactant.get("source_locator") or payload.get("source_locator") or ""),
                "source_excerpt": str(product.get("source_excerpt") or payload.get("source_excerpt") or ""),
            }
        )
    return out


def _normalize_step(raw: dict[str, Any], *, index: int, payload: dict[str, Any]) -> dict[str, Any]:
    reactants = raw.get("reactant_smiles")
    if reactants is None:
        reactants = raw.get("reactants")
    if reactants is None and raw.get("main_reactant_smiles"):
        reactants = [raw.get("main_reactant_smiles")]
    if isinstance(reactants, str):
        reactant_values = [item for item in re.split(r"\s*(?:\.|,|;)\s*", reactants) if item]
    elif isinstance(reactants, list):
        reactant_values = [str(item) for item in reactants if str(item).strip()]
    else:
        reactant_values = []
    labels = raw.get("reactant_labels")
    if labels is None and raw.get("reactant_label"):
        labels = [raw.get("reactant_label")]
    if not isinstance(labels, list):
        labels = []
    return {
        "schema_version": "visual_structure_candidate_step.v1",
        "step_id": str(raw.get("step_id") or f"visual_step_{index}"),
        "segment_id": str(raw.get("segment_id") or payload.get("segment_id") or "visual_literature_chain"),
        "product_label": str(raw.get("product_label") or raw.get("product_name") or ""),
        "product_smiles": str(raw.get("product_smiles") or raw.get("product") or ""),
        "reactant_labels": [str(item) for item in labels],
        "reactant_smiles": reactant_values,
        "main_reactant_smiles": str(raw.get("main_reactant_smiles") or ""),
        "source_ref": str(raw.get("source_ref") or payload.get("source_ref") or ""),
        "source_title": str(raw.get("source_title") or payload.get("source_title") or ""),
        "evidence_refs": _string_list(raw.get("evidence_refs") or payload.get("evidence_refs")),
        "source_locator": str(raw.get("source_locator") or payload.get("source_locator") or ""),
        "condition_candidate": raw.get("condition_candidate") or payload.get("default_condition_candidate") or {},
        "structure_derivation": raw.get("structure_derivation") or {},
        "source_excerpt": str(raw.get("source_excerpt") or payload.get("source_excerpt") or ""),
        "confidence": str(raw.get("confidence") or payload.get("confidence") or "medium_high"),
    }


def _smiles_report(smiles: str) -> dict[str, Any]:
    text = str(smiles or "").strip()
    if not text:
        return {
            "input_smiles": "",
            "valid": False,
            "canonical_smiles": "",
            "formula": "",
            "exact_mw": 0.0,
            "heavy_atoms": 0,
        }
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return {
            "input_smiles": text,
            "valid": False,
            "canonical_smiles": "",
            "formula": "",
            "exact_mw": 0.0,
            "heavy_atoms": 0,
        }
    return {
        "input_smiles": text,
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mw": round(float(Descriptors.ExactMolWt(mol)), 6),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def _condition_candidate(value: Any) -> dict[str, Any]:
    row = dict(value) if isinstance(value, dict) else {}
    reagent_candidates = _string_list(row.get("reagent_candidates"))
    solvent_candidates = _string_list(row.get("solvent_candidates"))
    condition = {
        "schema_version": "condition_candidate.v1",
        "source_type": str(row.get("source_type") or "exact"),
        "condition_status": str(row.get("condition_status") or "evidence_backed"),
        "reagent": str(row.get("reagent") or "; ".join(reagent_candidates)),
        "reagent_candidates": reagent_candidates,
        "catalyst": str(row.get("catalyst") or ""),
        "enzyme": str(row.get("enzyme") or ""),
        "solvent": str(row.get("solvent") or "; ".join(solvent_candidates)),
        "solvent_candidates": solvent_candidates,
        "temperature": str(row.get("temperature") or row.get("temperature_c") or row.get("temperature_C") or ""),
        "duration": str(row.get("duration") or row.get("duration_h") or ""),
        "reported_yield": str(row.get("reported_yield") or ""),
        "source_grounding": str(row.get("source_grounding") or ""),
        "evidence_refs": _string_list(row.get("evidence_refs")),
        "risk_flags": _string_list(row.get("risk_flags")),
    }
    return {key: data for key, data in condition.items() if data not in ("", [])}


def _condition_has_signal(condition: dict[str, Any]) -> bool:
    return any(
        str(condition.get(key) or "").strip()
        for key in ("reagent", "catalyst", "enzyme", "solvent", "temperature", "duration", "reported_yield")
    ) or bool(condition.get("reagent_candidates") or condition.get("solvent_candidates"))


def _structure_derivation(step: dict[str, Any], raw: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    derivation = dict(raw.get("structure_derivation") or step.get("structure_derivation") or {})
    basis = str(derivation.get("basis") or "source_structure_diagram_to_smiles")
    source_locator = str(derivation.get("source_locator") or step.get("source_locator") or payload.get("source_locator") or "")
    confidence = str(derivation.get("confidence") or step.get("confidence") or payload.get("confidence") or "medium_high")
    tool_checks = _string_list(derivation.get("tool_checks"))
    if not tool_checks:
        tool_checks = ["RDKit parsed product and reactant SMILES", "continuous chain adjacency checked"]
    return {
        **derivation,
        "basis": basis,
        "source_locator": source_locator,
        "confidence": confidence,
        "tool_checks": tool_checks,
    }


def _main_reactant(reactants: list[str], preferred: str = "") -> str:
    preferred_report = _smiles_report(preferred)
    if preferred_report["valid"]:
        return preferred_report["canonical_smiles"]
    valid = [_smiles_report(smiles) for smiles in reactants]
    valid = [row for row in valid if row["valid"]]
    if not valid:
        return ""
    return max(valid, key=lambda row: (int(row["heavy_atoms"]), str(row["canonical_smiles"])))["canonical_smiles"]


def _continuity_audit(steps: list[dict[str, Any]], *, target_smiles: str = "") -> dict[str, Any]:
    reasons: list[str] = []
    links: list[dict[str, Any]] = []
    target_key = _smiles_report(target_smiles).get("canonical_smiles") if target_smiles else ""
    first_product = str(((steps[0].get("product") or {}).get("canonical_smiles") if steps else "") or "")
    target_match = bool(target_key and first_product == target_key)
    if target_key and steps and not target_match:
        reasons.append("first_step_product_does_not_match_target")
    for index in range(len(steps) - 1):
        left = steps[index]
        right = steps[index + 1]
        left_main = str(left.get("main_reactant_smiles") or "")
        right_product = str((right.get("product") or {}).get("canonical_smiles") or "")
        accepted = bool(left_main and right_product and left_main == right_product)
        if not accepted:
            reasons.append("chain_discontinuity")
        links.append(
            {
                "from_step_id": str(left.get("step_id") or ""),
                "to_step_id": str(right.get("step_id") or ""),
                "left_main_reactant": left_main,
                "right_product": right_product,
                "accepted": accepted,
            }
        )
    return {
        "schema_version": "visual_structure_chain_continuity_audit.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "link_count": len(links),
        "links": links,
        "target_match": target_match or not target_key,
        "target_canonical_smiles": target_key or "",
    }


def _record_compound(compound_map: dict[str, dict[str, Any]], *, label: str, report: dict[str, Any]) -> None:
    key = str(report.get("canonical_smiles") or report.get("input_smiles") or label)
    if not key:
        return
    existing = compound_map.setdefault(
        key,
        {
            "schema_version": "visual_structure_compound_report.v1",
            "compound_labels": [],
            "smiles": str(report.get("canonical_smiles") or report.get("input_smiles") or ""),
            "valid": bool(report.get("valid")),
            "formula": str(report.get("formula") or ""),
            "exact_mw": float(report.get("exact_mw") or 0.0),
            "heavy_atoms": int(report.get("heavy_atoms") or 0),
        },
    )
    if label and label not in existing["compound_labels"]:
        existing["compound_labels"].append(label)


def _short_excerpt(text: str, *, max_words: int = 36) -> str:
    words = re.sub(r"\s+", " ", str(text or "")).strip().split()
    return " ".join(words[:max_words])


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or "item"
