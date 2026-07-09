"""Advisory target-side disconnection strategy for agentic blackboard runs."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.harness.route_objectives import classify_route_objectives


RDLogger.DisableLog("rdApp.*")

TARGET_SIDE_DISCONNECTION_SCHEMA = "target_side_disconnection_hypotheses.v1"


def build_target_side_disconnection_hypotheses(
    *,
    target_smiles: str,
    target_name: str = "",
    family_hint: str = "",
    source_evidence_refs: list[str] | None = None,
    case_id: str = "",
) -> dict[str, Any]:
    """Build bounded, advisory target-side hypotheses.

    The artifact is intentionally non-executable: it never emits reaction
    SMILES and can only bias later search or guided reruns.
    """
    mol = Chem.MolFromSmiles(str(target_smiles or ""))
    evidence_refs = _dedupe([str(item) for item in source_evidence_refs or []])
    reasons: list[str] = []
    if mol is None:
        reasons.append("invalid_target_smiles")
    handles = _detect_handles(mol, target_name=target_name, family_hint=family_hint)
    objective_summary = classify_route_objectives(
        target_smiles=target_smiles,
        target_name=target_name,
        family_hint=family_hint,
        source_evidence_refs=evidence_refs,
        case_id=case_id,
    )
    route_scope = dict(objective_summary.get("route_scope") or {})
    endpoint_candidates = [
        dict(row)
        for row in objective_summary.get("endpoint_candidates") or []
        if isinstance(row, dict)
    ]
    semisynthesis_anchors = _anchors_from_endpoint_candidates(endpoint_candidates, case_id=case_id)
    hypotheses: list[dict[str, Any]] = []

    if "aryl_ester_or_anthranilate_sidechain" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_aryl_ester_or_anthranilate_disconnection",
                target_handle="aryl_ester_or_anthranilate_sidechain",
                proposed_disconnection_region="late-stage aryl ester/anthranilate sidechain attachment",
                must_preserve=["polycyclic_cage_core", "tertiary_amine_state"],
                expected_precursor_type="core alcohol or phenol plus activated anthranilate/imide acid fragment",
                evidence_refs=evidence_refs,
                risk_flags=["sidechain_exact_replay_requires_target_proximal_core_bridge"],
                required_verification=[
                    "target_equivalence",
                    "target_core_retention",
                    "exact_literature_segment_connected_to_parent_route",
                ],
            )
        )
    if "imide_or_succinimide_fragment" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_imide_fragment_preparation",
                target_handle="imide_or_succinimide_fragment",
                proposed_disconnection_region="imide/succinimide substituent preparation before final coupling",
                must_preserve=["polycyclic_cage_core"],
                expected_precursor_type="preassembled imide or succinimide-containing acid/activated fragment",
                evidence_refs=evidence_refs,
                risk_flags=["fragment_synthesis_cannot_close_parent_without_stitch"],
                required_verification=["fragment_identity", "parent_bridge_connectivity"],
            )
        )
    if "polycyclic_cage_core" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_polycyclic_cage_core_preservation",
                target_handle="polycyclic_cage_core",
                proposed_disconnection_region="peripheral functionalization while retaining the cage scaffold",
                must_preserve=["polycyclic_cage_core"],
                expected_precursor_type="target-proximal cage intermediate with matched ring system",
                evidence_refs=evidence_refs,
                risk_flags=["large_atom_jump_must_be_explained_by_bridge"],
                required_verification=[
                    "core_retention_audit",
                    "max_unexplained_heavy_atom_jump",
                    "route_verifier_acceptance",
                ],
            )
        )
    if "tertiary_amine" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_tertiary_amine_state_adjustment",
                target_handle="tertiary_amine",
                proposed_disconnection_region="amine-state or N-substitution adjustment",
                must_preserve=["polycyclic_cage_core"],
                expected_precursor_type="same-core secondary/tertiary amine state precursor",
                evidence_refs=evidence_refs,
                risk_flags=["amine_state_change_is_advisory_only"],
                required_verification=["same_core_identity", "no_imide_n_misassignment"],
            )
        )
    if "protecting_group_level_transformations" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_protecting_group_level_transformations",
                target_handle="protecting_group_level_transformations",
                proposed_disconnection_region="alcohol/phenol protection-state adjustments",
                must_preserve=["polycyclic_cage_core"],
                expected_precursor_type="same-core alcohol/phenol protection-state analog",
                evidence_refs=evidence_refs,
                risk_flags=["protecting_group_logic_not_parent_route_proof"],
                required_verification=["same_core_identity", "source_grounded_conditions"],
            )
        )
    if "bufadienolide_c17_pyrone_sidechain" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_bufadienolide_c17_pyrone_disconnection",
                target_handle="bufadienolide_c17_pyrone_sidechain",
                proposed_disconnection_region="bufadienolide C17 pyrone sidechain attachment while preserving the steroid core",
                must_preserve=["steroid_core", "polycyclic_cage_core"],
                expected_precursor_type="C17-functionalized steroid core plus pyrone-sidechain bridge precursor",
                evidence_refs=evidence_refs,
                risk_flags=["pyrone_installation_is_analogical_until_parent_stitch_passes"],
                required_verification=[
                    "bufadienolide_c17_retron_match",
                    "steroid_core_retention",
                    "parent_bridge_connectivity",
                    "route_verifier_acceptance",
                ],
            )
        )
    if "analogue_diene_iminium_scaffold_inspiration" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_analogue_diene_iminium_scaffold_inspiration",
                target_handle="analogue_diene_iminium_scaffold_inspiration",
                proposed_disconnection_region="scaffold-construction inspiration from analogue diene/iminium chemistry",
                must_preserve=["polycyclic_cage_core"],
                expected_precursor_type="target-proximal cage precursor, not a distant analogue product",
                evidence_refs=evidence_refs,
                risk_flags=["analogy_never_counts_as_solved_proof"],
                required_verification=["target_proximal_bridge", "deterministic_parent_proof"],
            )
        )
    if "semisynthesis_or_biotransformation_anchor" in handles:
        hypotheses.append(
            _hypothesis(
                "target_side_semisynthesis_or_biotransformation_anchor",
                target_handle="semisynthesis_or_biotransformation_anchor",
                proposed_disconnection_region="retain the complex natural-product-like core and search for a same-scaffold feedstock, advanced intermediate, or biotransformation substrate",
                must_preserve=["largest_polycyclic_core", "native_stereochemical_framework"],
                expected_precursor_type="evidence-backed same-core natural product, advanced intermediate, or biotransformation substrate",
                evidence_refs=evidence_refs,
                risk_flags=[
                    "de_novo_steroid_core_construction_deprioritized",
                    "anchor_requires_source_validation",
                    "anchor_is_not_parent_route_proof",
                ],
                required_verification=[
                    "same_core_anchor_identity",
                    "objective_endpoint_source_validation",
                    "target_side_conversion_logic",
                    "objective_specific_route_proof",
                ],
            )
        )

    if not hypotheses and mol is not None:
        hypotheses.append(
            _hypothesis(
                "target_side_generic_functional_handle_scan",
                target_handle="generic_functional_handles",
                proposed_disconnection_region="highest-confidence peripheral functional handle",
                must_preserve=["largest_ring_or_core_system"],
                expected_precursor_type="target-proximal intermediate with modest heavy-atom delta",
                evidence_refs=evidence_refs,
                risk_flags=["low_specificity_hypothesis"],
                required_verification=["target_equivalence", "route_verifier_acceptance"],
            )
        )

    bridge_tasks = [
        _bridge_task_from_hypothesis(row, case_id=case_id, target_name=target_name)
        for row in hypotheses
    ]
    if semisynthesis_anchors:
        bridge_tasks.extend(
            _bridge_tasks_from_semisynthesis_anchors(
                semisynthesis_anchors,
                case_id=case_id,
                target_name=target_name,
            )
        )
    return {
        "schema_version": TARGET_SIDE_DISCONNECTION_SCHEMA,
        "accepted": bool(hypotheses) and mol is not None,
        "case_id": case_id,
        "target": {
            "name": target_name,
            "smiles": target_smiles,
            "handles": handles,
            "heavy_atoms": int(mol.GetNumHeavyAtoms()) if mol is not None else 0,
            "rings": int(mol.GetRingInfo().NumRings()) if mol is not None else 0,
        },
        "hypotheses": hypotheses,
        "bridge_tasks": bridge_tasks,
        "route_scope": route_scope,
        "route_objective_summary": objective_summary,
        "endpoint_candidates": endpoint_candidates,
        "semisynthesis_anchors": semisynthesis_anchors,
        "source_candidates": [],
        "mode": "advisory_target_side_strategy_only",
        "semantics": {
            "raw_reaction_output_allowed": False,
            "solved_claim_allowed": False,
            "requires_parent_route_proof": True,
        },
        "no_solved_claim": True,
        "requires_verifier": True,
        "production_write_blocked": True,
        "reasons": sorted(set(reasons)),
    }


def _detect_handles(mol: Chem.Mol | None, *, target_name: str, family_hint: str) -> list[str]:
    if mol is None:
        return []
    handles: list[str] = []
    if _has(mol, "[OX2][CX3](=O)c") or _has(mol, "c[CX3](=O)[OX2]"):
        handles.append("aryl_ester_or_anthranilate_sidechain")
    if _has(mol, "[NX3]([CX3](=O))[CX3](=O)") or _has(mol, "N1C(=O)CCC1=O"):
        handles.append("imide_or_succinimide_fragment")
    if _has(mol, "[NX3;H0;!$(NC=O);!$(N=C=O)]"):
        handles.append("tertiary_amine")
    oxygen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    if oxygen_count >= 4:
        handles.append("protecting_group_level_transformations")
    text = f"{target_name} {family_hint}".lower()
    if any(key in text for key in ["bufotalin", "bufadienolide", "pyrone"]) or _has(mol, "C1=COC(=O)C=C1"):
        handles.append("bufadienolide_c17_pyrone_sidechain")
    if mol.GetRingInfo().NumRings() >= 4 or any(key in text for key in ["mla", "methyllycaconitine", "lycoctonine", "aconitine"]):
        handles.append("polycyclic_cage_core")
    if any(key in text for key in ["mla", "methyllycaconitine", "lycoctonine", "aconitine", "alkaloid"]) or mol.GetRingInfo().NumRings() >= 4:
        handles.append("analogue_diene_iminium_scaffold_inspiration")
    if _is_steroid_or_terpenoid_polycyclic(mol, text=text):
        handles.extend(
            [
                "steroid_core",
                "semisynthesis_or_biotransformation_anchor",
            ]
        )
    return _dedupe(handles)


def _anchors_from_endpoint_candidates(candidates: list[dict[str, Any]], *, case_id: str) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for row in candidates:
        objective_type = str(row.get("objective_type") or "")
        if objective_type not in {
            "semisynthesis_from_natural_product",
            "biotransformation_endpoint",
            "advanced_intermediate_anchor",
            "literature_known_scaffold_anchor",
            "parent_to_impurity_or_metabolite",
        }:
            continue
        endpoint_type = str(row.get("endpoint_type") or "endpoint")
        anchors.append(
            {
                "schema_version": "semisynthesis_anchor.v1",
                "anchor_id": f"route_objective_anchor:{objective_type}:{endpoint_type}",
                "case_id": case_id,
                "anchor_type": endpoint_type,
                "objective_type": objective_type,
                "name": endpoint_type.replace("_", " "),
                "smiles": "",
                "role": str(row.get("description") or ""),
                "route_role": "objective_endpoint_candidate",
                "evidence_refs": [],
                "required_verification": [
                    str(item)
                    for item in row.get("required_verification") or []
                    if str(item or "").strip()
                ],
                "allowed_use": "route_objective_anchor_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_source_validation": True,
                "no_solved_claim": True,
            }
        )
    return anchors[:6]


def _bridge_tasks_from_semisynthesis_anchors(
    anchors: list[dict[str, Any]],
    *,
    case_id: str,
    target_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        rows.append(
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": f"semisynthesis_bridge:{anchor.get('anchor_id')}",
                "case_id": case_id,
                "task_type": "objective_endpoint_anchor_validation",
                "target_name": target_name,
                "target_handle": str(anchor.get("objective_type") or "route_objective_anchor"),
                "required_bridge": str(anchor.get("role") or ""),
                "source_hypothesis_id": "target_side_semisynthesis_or_biotransformation_anchor",
                "anchor_id": str(anchor.get("anchor_id") or ""),
                "anchor": dict(anchor),
                "status": "open",
                "required_verification": [
                    *[str(item) for item in anchor.get("required_verification") or []],
                    "objective_endpoint_source_validation",
                ],
                "no_solved_claim": True,
            }
        )
    return rows


def _hypothesis(
    hypothesis_id: str,
    *,
    target_handle: str,
    proposed_disconnection_region: str,
    must_preserve: list[str],
    expected_precursor_type: str,
    evidence_refs: list[str],
    risk_flags: list[str],
    required_verification: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "target_side_disconnection_hypothesis.v1",
        "hypothesis_id": hypothesis_id,
        "target_handle": target_handle,
        "proposed_disconnection_region": proposed_disconnection_region,
        "must_preserve_substructure": _dedupe(must_preserve),
        "expected_precursor_type": expected_precursor_type,
        "related_source_evidence": _dedupe(evidence_refs),
        "risk_flags": _dedupe(risk_flags),
        "required_verification": _dedupe(required_verification),
        "no_solved_claim": True,
    }


def _bridge_task_from_hypothesis(row: dict[str, Any], *, case_id: str, target_name: str) -> dict[str, Any]:
    handle = str(row.get("target_handle") or "target_side_bridge")
    return {
        "schema_version": "agent_bridge_task.v1",
        "task_id": f"bridge:{handle}",
        "case_id": case_id,
        "task_type": "target_proximal_bridge",
        "target_name": target_name,
        "target_handle": handle,
        "required_bridge": str(row.get("expected_precursor_type") or ""),
        "source_hypothesis_id": str(row.get("hypothesis_id") or ""),
        "status": "open",
        "required_verification": [str(item) for item in row.get("required_verification") or []],
    }


def _is_steroid_or_terpenoid_polycyclic(mol: Chem.Mol, *, text: str) -> bool:
    if any(token in text for token in ("steroid", "sterol", "pregn", "androst", "chol", "cardenolide", "bufadienolide")):
        return True
    heavy_atoms = max(1, int(mol.GetNumHeavyAtoms()))
    carbon_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    carbon_fraction = carbon_atoms / heavy_atoms
    return bool(mol.GetRingInfo().NumRings() >= 4 and heavy_atoms >= 18 and carbon_fraction >= 0.72)


def _has(mol: Chem.Mol, smarts: str) -> bool:
    query = Chem.MolFromSmarts(smarts)
    return bool(query is not None and mol.HasSubstructMatch(query))


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
