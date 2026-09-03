"""Advisory analogical retrosynthesis hypotheses from source-detail rows.

This module deliberately does not emit executable reactions or solved claims.
It converts exact literature fragments into auditable inspiration signals that
can be attached to guided search policies while keeping verifier authority
separate.
"""
from __future__ import annotations

from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

ANALOGICAL_RETROSYNTHESIS_SCHEMA = "analogical_retrosynthesis_hypotheses.v1"


def build_analogical_retrosynthesis_hypotheses(
    *,
    compiled_downstream: dict[str, Any],
    target_smiles: str,
    target_name: str = "",
    case_id: str = "",
    max_hypotheses: int = 12,
) -> dict[str, Any]:
    """Build bounded advisory hypotheses from compiled literature rows."""
    rows = _one_step_rows(compiled_downstream)
    target = _mol(target_smiles)
    target_summary = _target_summary(target_smiles, target)
    reasons: list[str] = []
    if target is None:
        reasons.append("invalid_target_smiles")
    if not rows:
        reasons.append("no_literature_one_step_rows")

    handles = _target_handles(target)
    row_hypotheses: list[dict[str, Any]] = []
    families: list[str] = []
    evidence_refs: list[str] = []
    for idx, row in enumerate(rows, start=1):
        trace = _trace(row)
        product = str(trace.get("product_smiles") or trace.get("frontier_smiles") or "")
        reactants = _reactants(trace, row)
        condition = dict(trace.get("condition_candidate") or {})
        family = _infer_reaction_family(product, reactants, condition)
        if family not in families:
            families.append(family)
        evidence_refs.extend(str(item) for item in trace.get("evidence_refs") or [])
        similarity = _tanimoto(product, target_smiles)
        product_heavy_atoms = _heavy_atoms(product)
        target_heavy_atoms = target_summary.get("heavy_atoms") or 0
        target_proximity = _target_proximity(similarity, product_heavy_atoms, target_heavy_atoms)
        row_hypotheses.append(
            {
                "schema_version": "analogical_hypothesis.v1",
                "hypothesis_id": f"analogy_row_{idx}",
                "source_template_id": str(trace.get("source_template_id") or f"source_row_{idx}"),
                "source_ref": str(trace.get("source_ref") or ""),
                "evidence_refs": _dedupe([str(item) for item in trace.get("evidence_refs") or []]),
                "inspiration_type": "reaction_family_transfer",
                "reaction_family": family,
                "source_product_smiles": product,
                "source_product_heavy_atoms": product_heavy_atoms,
                "source_product_tanimoto_to_target": similarity,
                "source_max_reactant_heavy_atoms": max([_heavy_atoms(item) or 0 for item in reactants] or [0]),
                "target_proximity": target_proximity,
                "analogy_strength": _analogy_strength(target_proximity, family, handles),
                "target_side_attempt": _target_side_attempt(family, handles),
                "required_verification": [
                    "find_target_proximal_intermediate_before_exact_replay",
                    "reject_unexplained_large_atom_jump",
                    "deterministic_route_verifier_must_accept_before_solved_claim",
                ],
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )

    global_hypotheses = _global_target_hypotheses(handles, families, evidence_refs)
    hypotheses = [*global_hypotheses, *row_hypotheses][: max(0, int(max_hypotheses or 0)) or 12]
    if not hypotheses and not reasons:
        reasons.append("no_analogical_hypotheses_generated")

    policy_patch = _search_policy_patch(hypotheses, families)
    return {
        "schema_version": ANALOGICAL_RETROSYNTHESIS_SCHEMA,
        "accepted": bool(hypotheses) and target is not None,
        "reasons": sorted(set(reasons)),
        "case_id": case_id,
        "target": {
            "name": target_name,
            "smiles": target_smiles,
            **target_summary,
            "handles": handles,
        },
        "mode": "advisory_inspiration_only",
        "source_row_count": len(rows),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "search_policy_patch": policy_patch,
        "query_hints": _query_hints(target_name=target_name, handles=handles, families=families),
        "semantics": {
            "exact_replay": False,
            "analogy_allowed": True,
            "requires_source_grounded_bridge": True,
            "solved_claim_allowed": False,
            "verifier_authority": "deterministic_route_verifier",
        },
        "no_solved_claim": True,
        "requires_verifier": True,
        "production_write_blocked": True,
    }


def _one_step_rows(compiled: dict[str, Any]) -> list[dict[str, Any]]:
    plugin = dict((compiled or {}).get("literature_template_plugin") or {})
    rows = plugin.get("one_step_rows")
    if rows is None:
        rows = (plugin.get("plugin_flags") or {}).get("one_step_rows")
    return [dict(item) for item in rows or [] if isinstance(item, dict)]


def _trace(row: dict[str, Any]) -> dict[str, Any]:
    template = row.get("template") if isinstance(row.get("template"), dict) else row.get("templates")
    return dict(row.get("literature_template_trace") or (template or {}).get("literature_template_trace") or {})


def _reactants(trace: dict[str, Any], row: dict[str, Any]) -> list[str]:
    raw = trace.get("reactant_smiles")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return [str(item) for item in str(row.get("reactants") or "").split(".") if str(item).strip()]


def _target_summary(smiles: str, mol: Chem.Mol | None) -> dict[str, Any]:
    if mol is None:
        return {"valid": False, "heavy_atoms": 0, "formula": ""}
    return {
        "valid": True,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "ring_count": mol.GetRingInfo().NumRings(),
    }


def _target_handles(mol: Chem.Mol | None) -> list[str]:
    if mol is None:
        return []
    handles: list[str] = []
    if _has(mol, "[OX2][CX3](=O)c"):
        handles.append("aryl_ester_or_anthranilate_sidechain")
    if _has(mol, "N1C(=O)CC(C)C1=O") or _has(mol, "N1C(=O)CCC1=O"):
        handles.append("succinimide_or_imide_sidechain")
    if _has(mol, "[NX3;H0;!$(NC=O)]"):
        handles.append("tertiary_amine")
    oxygen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    if oxygen_count >= 6:
        handles.append("polyoxygenated_core")
    if mol.GetRingInfo().NumRings() >= 4:
        handles.append("polycyclic_cage_core")
    return handles


def _infer_reaction_family(product: str, reactants: list[str], condition: dict[str, Any]) -> str:
    text = " ".join([product, *reactants, str(condition.get("reagent") or ""), str(condition.get("source_grounding") or "")]).lower()
    if "si" in " ".join(reactants).lower() and "si" not in product.lower():
        return "silyl_ether_deprotection"
    if "momcl" in text or "ocom" in product.lower() or "ococ" in product.lower():
        return "alkoxymethyl_protection"
    if "eti" in text or "ethylation" in text or "n-ethyl" in text:
        return "n_alkylation"
    if "dppa" in text or "curtius" in text or "boc" in text:
        return "curtius_carbamate_formation"
    if "lactam" in text or "[nh]c(=o)" in product.lower():
        return "lactam_or_heteroaryl_cyclization"
    if product.count("=") >= 2 or any(item.count("=") >= 2 for item in reactants):
        return "diene_or_unsaturated_imine_scaffold_analogy"
    return "functional_group_interconversion"


def _target_side_attempt(family: str, handles: list[str]) -> dict[str, Any]:
    if "aryl_ester_or_anthranilate_sidechain" in handles:
        default_focus = (
            "Attempt a target-side disconnection at the aryl ester sidechain while preserving the "
            "polycyclic alkaloid core; search separately for the core alcohol and activated anthranilate/imide fragment."
        )
    else:
        default_focus = "Use the source reaction family only after a target-proximal intermediate is identified."
    family_guidance = {
        "silyl_ether_deprotection": "Use as protecting-group logic for alcohol/phenol handles, not as evidence for cage construction.",
        "alkoxymethyl_protection": "Use as phenol/alcohol masking logic in sidechain or fragment preparation.",
        "n_alkylation": "Consider late-stage amine-state adjustment only if it preserves the target core and avoids imide N misassignment.",
        "curtius_carbamate_formation": "Use as a way to prepare protected unsaturated amine fragments; it is not target closure evidence.",
        "lactam_or_heteroaryl_cyclization": "Use as heteroaryl or imide-fragment assembly inspiration, requiring a separate target-side bridge.",
        "diene_or_unsaturated_imine_scaffold_analogy": "Use as scaffold-construction inspiration only after a matching target-proximal cage precursor is found.",
    }
    return {
        "role": "chemist_advisory_disconnection",
        "focus": default_focus,
        "family_guidance": family_guidance.get(family, "Treat as generic functional-group analogy."),
        "must_preserve": [item for item in ("polycyclic_cage_core", "aryl_ester_or_anthranilate_sidechain") if item in handles],
    }


def _global_target_hypotheses(handles: list[str], families: list[str], evidence_refs: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "aryl_ester_or_anthranilate_sidechain" in handles:
        out.append(
            {
                "schema_version": "analogical_hypothesis.v1",
                "hypothesis_id": "target_side_late_stage_aryl_ester_disconnection",
                "inspiration_type": "target_functional_handle",
                "reaction_family": "late_stage_ester_sidechain_disconnection",
                "evidence_refs": _dedupe(evidence_refs),
                "target_side_attempt": {
                    "role": "chemist_advisory_disconnection",
                    "focus": (
                        "Try retrosynthetic separation of the aryl ester sidechain from the alkaloid core, then require "
                        "source-grounded upstream evidence for both the core alcohol and the anthranilate/imide fragment."
                    ),
                    "must_preserve": [item for item in ("polycyclic_cage_core", "tertiary_amine") if item in handles],
                },
                "required_verification": [
                    "exact_target_equivalence",
                    "target_core_retention",
                    "stock_or_source_grounded_upstream_closure",
                ],
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    if "diene_or_unsaturated_imine_scaffold_analogy" in families:
        out.append(
            {
                "schema_version": "analogical_hypothesis.v1",
                "hypothesis_id": "source_analogue_scaffold_construction_inspiration",
                "inspiration_type": "analogue_scaffold_strategy",
                "reaction_family": "diene_or_unsaturated_imine_scaffold_analogy",
                "evidence_refs": _dedupe(evidence_refs),
                "target_side_attempt": {
                    "role": "chemist_advisory_disconnection",
                    "focus": (
                        "Use the analogue diene/iminium chemistry as a brainstorming source for ring construction, "
                        "but only accept it after finding a target-proximal cage intermediate."
                    ),
                    "must_preserve": [item for item in ("polycyclic_cage_core",) if item in handles],
                },
                "required_verification": [
                    "find_target_proximal_cage_intermediate",
                    "reject_fragment_only_closure",
                ],
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    return out


def _search_policy_patch(hypotheses: list[dict[str, Any]], families: list[str]) -> dict[str, Any]:
    preferred = _dedupe(
        [
            "target_side_analogical_disconnection",
            "late_stage_ester_sidechain_disconnection",
            *families,
        ]
    )
    return {
        "schema_version": "analogical_search_policy_patch.v1",
        "enabled": bool(hypotheses),
        "preferred_reaction_classes": preferred,
        "active_failure_modes": ["large_atom_jump", "literature_template_plugin_not_invoked"],
        "require_target_core_retention": True,
        "max_unexplained_heavy_atom_delta": 20,
        "hypothesis_ids": [str(item.get("hypothesis_id") or "") for item in hypotheses if item.get("hypothesis_id")],
        "not_raw_reaction_injection": True,
        "no_solved_claim": True,
    }


def _query_hints(*, target_name: str, handles: list[str], families: list[str]) -> list[dict[str, Any]]:
    name = str(target_name or "target").strip() or "target"
    hints: list[dict[str, Any]] = []
    if "aryl_ester_or_anthranilate_sidechain" in handles:
        hints.append(
            {
                "schema_version": "analogical_query_hint.v1",
                "hint_type": "target_side_bridge",
                "query": f"{name} synthesis aryl ester side chain alkaloid core alcohol intermediate",
                "reason": "need_target_proximal_sidechain_disconnection",
            }
        )
    if "diene_or_unsaturated_imine_scaffold_analogy" in families:
        hints.append(
            {
                "schema_version": "analogical_query_hint.v1",
                "hint_type": "analogue_scaffold_bridge",
                "query": f"{name} synthesis diene iminium aza annulation cage intermediate",
                "reason": "analogue_literature_needs_target_bridge",
            }
        )
    return hints


def _target_proximity(similarity: float | None, product_heavy_atoms: int | None, target_heavy_atoms: int) -> str:
    sim = float(similarity or 0.0)
    heavy_ratio = float(product_heavy_atoms or 0) / float(target_heavy_atoms or 1)
    if sim >= 0.5 or heavy_ratio >= 0.75:
        return "target_proximal"
    if sim >= 0.2 or heavy_ratio >= 0.45:
        return "fragment_or_midstage"
    return "distant_fragment"


def _analogy_strength(target_proximity: str, family: str, handles: list[str]) -> str:
    if target_proximity == "target_proximal":
        return "medium"
    if family in {"n_alkylation", "silyl_ether_deprotection", "alkoxymethyl_protection"} and (
        "polyoxygenated_core" in handles or "tertiary_amine" in handles
    ):
        return "low_to_medium"
    return "low"


def _tanimoto(a: str, b: str) -> float | None:
    ma = _mol(a)
    mb = _mol(b)
    if ma is None or mb is None:
        return None
    fpa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fpb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return round(float(DataStructs.TanimotoSimilarity(fpa, fpb)), 4)


def _heavy_atoms(smiles: str) -> int | None:
    mol = _mol(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else None


def _has(mol: Chem.Mol, smarts: str) -> bool:
    query = Chem.MolFromSmarts(smarts)
    return bool(query is not None and mol.HasSubstructMatch(query))


def _mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(str(smiles or ""))


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
