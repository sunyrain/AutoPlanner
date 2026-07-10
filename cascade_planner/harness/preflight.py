"""Preflight validation before any live Codex or ChemEnzy call."""
from __future__ import annotations

import re
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.agent.target_profile import build_target_profile
from cascade_planner.harness.schemas import TargetInput


RDLogger.DisableLog("rdApp.*")
PREFLIGHT_SCHEMA = "codex_entry_preflight.v1"
KNOWN_TARGET_IDENTITY_SCHEMA = "known_target_identity_audit.v1"
KNOWN_TARGET_IDENTITIES = {
    "atorvastatin": {
        "aliases": ("atorvastatin", "lipitor"),
        "expected_inchi_key": "XUKUURHRXDUEBC-KAYWLYCHSA-N",
        "expected_smiles": (
            "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)"
            "C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
        ),
        "description": "atorvastatin free acid",
    },
    "paclitaxel": {
        "aliases": ("paclitaxel", "taxol"),
        "expected_inchi_key": "RCINICONZNJXQF-MZXODVADSA-N",
        "expected_smiles": (
            "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)"
            "(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)"
            "OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
        ),
        "description": "paclitaxel (Taxol)",
    },
}


def run_preflight(target: TargetInput) -> dict[str, Any]:
    profile = build_target_profile(
        target.target_smiles,
        target_name=target.target_name,
        case_id=target.case_id,
        family_hint=target.family_hint,
    )
    flags = _initial_risk_flags(profile.to_dict())
    identity_audit = _known_target_identity_audit(target, profile.to_dict())
    identity_mismatch = bool(identity_audit and not identity_audit.get("accepted"))
    accepted = bool(profile.valid) and not identity_mismatch
    if not accepted:
        flags.append("known_target_identity_mismatch" if identity_mismatch else "invalid_smiles")
    reasons = [] if accepted else [profile.parse_error or "invalid_smiles"]
    if identity_mismatch:
        reasons = [f"known_target_identity_mismatch:{identity_audit.get('target_key')}"]
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "case_id": profile.case_id,
        "accepted": accepted,
        "valid_smiles": bool(profile.valid),
        "canonical_smiles": profile.canonical_smiles,
        "isomeric_smiles": profile.isomeric_smiles,
        "inchi_key": profile.inchi_key,
        "known_target_identity_audit": identity_audit or {},
        "target_profile": profile.to_dict(),
        "initial_risk_flags": sorted(set(flags)),
        "reasons": reasons,
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


def _known_target_identity_audit(target: TargetInput, profile: dict[str, Any]) -> dict[str, Any] | None:
    target_key = _known_target_key(target)
    if not target_key:
        return None
    expected = KNOWN_TARGET_IDENTITIES[target_key]
    observed = str(profile.get("inchi_key") or "")
    expected_key = str(expected["expected_inchi_key"])
    return {
        "schema_version": KNOWN_TARGET_IDENTITY_SCHEMA,
        "target_key": target_key,
        "accepted": bool(observed and observed == expected_key),
        "observed_inchi_key": observed,
        "expected_inchi_key": expected_key,
        "expected_smiles": expected["expected_smiles"],
        "description": expected["description"],
    }


def _known_target_key(target: TargetInput) -> str:
    name = str(target.target_name or target.case_id or "").strip().lower()
    family_hint = str(target.family_hint or "").strip().lower()
    for target_key, identity in KNOWN_TARGET_IDENTITIES.items():
        aliases = tuple(str(alias).lower() for alias in identity.get("aliases") or ())
        if any(
            _is_analog_like_name(text, alias)
            for text in (name, family_hint)
            for alias in aliases
        ):
            continue
        if any(_contains_standalone_alias(name, alias) for alias in aliases):
            return target_key
        if any(_contains_standalone_alias(family_hint, alias) for alias in aliases):
            return target_key
    return ""


def _is_analog_like_name(text: str, alias: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?:[-_ ]?(?:like|analog(?:ue)?))(?![A-Za-z0-9])",
            text,
        )
    )


def _contains_standalone_alias(text: str, alias: str) -> bool:
    if not text:
        return False
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text))
