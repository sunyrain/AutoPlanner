"""Preflight validation before any live Codex or ChemEnzy call."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.agent.target_profile import build_target_profile
from cascade_planner.harness.schemas import TargetInput


RDLogger.DisableLog("rdApp.*")
PREFLIGHT_SCHEMA = "codex_entry_preflight.v1"


def run_preflight(target: TargetInput) -> dict[str, Any]:
    profile = build_target_profile(
        target.target_smiles,
        target_name=target.target_name,
        case_id=target.case_id,
        family_hint=target.family_hint,
    )
    flags = _initial_risk_flags(profile.to_dict())
    accepted = bool(profile.valid)
    if not accepted:
        flags.append("invalid_smiles")
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "case_id": profile.case_id,
        "accepted": accepted,
        "valid_smiles": bool(profile.valid),
        "canonical_smiles": profile.canonical_smiles,
        "isomeric_smiles": profile.isomeric_smiles,
        "inchi_key": profile.inchi_key,
        "target_profile": profile.to_dict(),
        "initial_risk_flags": sorted(set(flags)),
        "reasons": [] if accepted else [profile.parse_error or "invalid_smiles"],
    }


def _initial_risk_flags(profile: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    family_hints = {str(item).lower() for item in profile.get("family_hints") or []}
    hint_blob = " ".join(family_hints)
    if int(profile.get("rings") or 0) >= 4 or "steroid" in hint_blob:
        flags.append("polycyclic_or_steroid_like")
    if "glycoside" in hint_blob or "glycoside_like_acetal" in family_hints:
        flags.append("glycoside_or_o_glycoside_like")
    if int(profile.get("heavy_atoms") or 0) >= 35:
        flags.append("high_heavy_atom_count")
    smiles = str(profile.get("isomeric_smiles") or profile.get("input_smiles") or "")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is not None:
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if any(flag == "?" for _, flag in chiral):
            flags.append("unassigned_stereochemistry")
    return flags
