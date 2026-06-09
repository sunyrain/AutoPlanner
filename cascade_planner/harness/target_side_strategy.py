"""Advisory target-side disconnection strategy for agentic blackboard runs."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger


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
    if mol.GetRingInfo().NumRings() >= 4 or any(key in text for key in ["mla", "methyllycaconitine", "lycoctonine", "aconitine"]):
        handles.append("polycyclic_cage_core")
    if any(key in text for key in ["mla", "methyllycaconitine", "lycoctonine", "aconitine", "alkaloid"]) or mol.GetRingInfo().NumRings() >= 4:
        handles.append("analogue_diene_iminium_scaffold_inspiration")
    return _dedupe(handles)


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
