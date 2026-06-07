"""Target profiling and frontier extraction for SMILES-first planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.inchi import MolToInchiKey


RDLogger.DisableLog("rdApp.*")
TARGET_PROFILE_SCHEMA = "target_profile.v1"
FRONTIER_REPORT_SCHEMA = "frontier_report.v1"


@dataclass
class TargetProfile:
    case_id: str
    target_name: str
    input_smiles: str
    valid: bool
    canonical_smiles: str = ""
    isomeric_smiles: str = ""
    inchi_key: str = ""
    formula: str = ""
    exact_mw: float | None = None
    heavy_atoms: int = 0
    rings: int = 0
    stereocenters: int = 0
    ring_systems: list[dict[str, Any]] = field(default_factory=list)
    linker_bonds: list[dict[str, Any]] = field(default_factory=list)
    side_chain_bonds: list[dict[str, Any]] = field(default_factory=list)
    family_hints: list[str] = field(default_factory=list)
    parse_error: str = ""
    schema_version: str = TARGET_PROFILE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_target_profile(
    target_smiles: str,
    *,
    target_name: str = "",
    case_id: str = "",
    family_hint: str = "",
) -> TargetProfile:
    mol = Chem.MolFromSmiles(str(target_smiles or ""))
    case_id = case_id or _safe_case_id(target_name, target_smiles)
    if mol is None:
        return TargetProfile(
            case_id=case_id,
            target_name=target_name,
            input_smiles=target_smiles,
            valid=False,
            family_hints=_split_hints(family_hint),
            parse_error="invalid_smiles",
        )

    ring_systems = _ring_systems(mol)
    ring_lookup = {
        atom_idx: sys_idx
        for sys_idx, system in enumerate(ring_systems)
        for atom_idx in system["atom_indices"]
    }
    family_hints = _family_hints(mol, family_hint)
    chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return TargetProfile(
        case_id=case_id,
        target_name=target_name,
        input_smiles=target_smiles,
        valid=True,
        canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=False),
        isomeric_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
        inchi_key=MolToInchiKey(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        exact_mw=round(float(Descriptors.ExactMolWt(mol)), 6),
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        rings=int(mol.GetRingInfo().NumRings()),
        stereocenters=len(chiral),
        ring_systems=ring_systems,
        linker_bonds=_linker_bonds(mol, ring_lookup),
        side_chain_bonds=_side_chain_bonds(mol, ring_lookup),
        family_hints=family_hints,
    )


def build_frontier_report(
    profile: TargetProfile,
    *,
    frontier_smiles: str = "",
    baseline_routes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frontiers: list[dict[str, Any]] = []
    if frontier_smiles:
        item = _frontier_item(profile, frontier_smiles, source="manual_frontier")
        frontiers.append(item)
    else:
        for smi in _frontiers_from_baseline(baseline_routes or {}):
            frontiers.append(_frontier_item(profile, smi, source="baseline_unresolved_leaf"))

    reasons = sorted({reason for item in frontiers for reason in item.get("flags", [])})
    failure_frontiers = [
        item for item in frontiers
        if item.get("frontier_role") != "target_as_initial_frontier"
    ]
    return {
        "schema_version": FRONTIER_REPORT_SCHEMA,
        "case_id": profile.case_id,
        "target_smiles": profile.isomeric_smiles or profile.input_smiles,
        "frontiers": frontiers,
        "advanced_frontier_found": bool(failure_frontiers),
        "reasons": reasons,
        "baseline_summary": _baseline_summary(baseline_routes or {}),
    }


def _frontier_item(profile: TargetProfile, smiles: str, *, source: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {
            "frontier_smiles": smiles,
            "source": source,
            "valid": False,
            "flags": ["invalid_frontier_smiles"],
        }
    target_mol = Chem.MolFromSmiles(profile.isomeric_smiles or profile.input_smiles)
    frontier_canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    target_canonical = Chem.MolToSmiles(target_mol, isomericSmiles=True) if target_mol is not None else ""
    target_as_initial_frontier = bool(source == "manual_frontier" and frontier_canonical == target_canonical)
    similarity = _similarity(target_mol, mol)
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    rings = int(mol.GetRingInfo().NumRings())
    heavy_delta = int(profile.heavy_atoms - heavy_atoms)
    flags: list[str] = []
    if target_as_initial_frontier:
        flags.append("target_as_initial_frontier")
    if not target_as_initial_frontier and similarity >= 0.62 and rings >= max(2, profile.rings - 1):
        flags.append("advanced_same_scaffold")
    if not target_as_initial_frontier and heavy_delta <= 2:
        flags.append("no_complexity_drop")
    if not target_as_initial_frontier and similarity >= 0.72:
        flags.append("high_target_similarity")
    if not target_as_initial_frontier and similarity >= 0.55 and 0 <= heavy_delta <= 8:
        flags.append("ordinary_decoration_only")
    if not target_as_initial_frontier and (
        rings >= 3 or any("steroid" in hint or "macrocycle" in hint for hint in profile.family_hints)
    ):
        flags.append("unresolved_core")
    if not flags and (rings >= 2 or heavy_atoms >= 20):
        flags.append("advanced_frontier")
    return {
        "frontier_smiles": frontier_canonical,
        "source": source,
        "frontier_role": "target_as_initial_frontier" if target_as_initial_frontier else "route_audit_frontier",
        "valid": True,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atoms": heavy_atoms,
        "rings": rings,
        "target_similarity": round(float(similarity), 4),
        "heavy_atom_delta_from_target": heavy_delta,
        "flags": flags,
    }


def _ring_systems(mol: Chem.Mol) -> list[dict[str, Any]]:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    systems: list[set[int]] = []
    for ring in rings:
        merged = False
        for system in systems:
            if system & ring:
                system.update(ring)
                merged = True
                break
        if not merged:
            systems.append(set(ring))
    changed = True
    while changed:
        changed = False
        for idx in range(len(systems)):
            for jdx in range(idx + 1, len(systems)):
                if systems[idx] & systems[jdx]:
                    systems[idx].update(systems[jdx])
                    del systems[jdx]
                    changed = True
                    break
            if changed:
                break
    return [
        {
            "system_id": f"ring_system_{idx + 1}",
            "atom_indices": sorted(system),
            "atom_count": len(system),
        }
        for idx, system in enumerate(systems)
    ]


def _linker_bonds(mol: Chem.Mol, ring_lookup: dict[int, int]) -> list[dict[str, Any]]:
    rows = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in ring_lookup and b in ring_lookup and ring_lookup[a] != ring_lookup[b]:
            rows.append({
                "bond_atoms": [a, b],
                "ring_systems": [ring_lookup[a], ring_lookup[b]],
                "bond_type": str(bond.GetBondType()),
            })
    return rows


def _side_chain_bonds(mol: Chem.Mol, ring_lookup: dict[int, int]) -> list[dict[str, Any]]:
    rows = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (a in ring_lookup) ^ (b in ring_lookup):
            rows.append({
                "bond_atoms": [a, b],
                "ring_atom": a if a in ring_lookup else b,
                "side_chain_atom": b if a in ring_lookup else a,
                "bond_type": str(bond.GetBondType()),
            })
    return rows


def _family_hints(mol: Chem.Mol, family_hint: str) -> list[str]:
    hints = _split_hints(family_hint)
    smarts_hints = [
        ("macrocycle_or_large_ring", "[r{10-}]"),
        ("lactone_or_ester", "[CX3](=O)[OX2][#6]"),
        ("glycoside_like_acetal", "[OX2][CX4]([OX2])[CX4]"),
        ("aromatic_or_heteroaromatic", "a"),
        ("polyol", "[OX2H]"),
    ]
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count >= 4:
        hints.append("polycyclic_or_steroid_like")
    for hint, smarts in smarts_hints:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            hints.append(hint)
    return sorted(set(hints))


def _split_hints(family_hint: str) -> list[str]:
    return [part.strip() for part in str(family_hint or "").replace(";", ",").split(",") if part.strip()]


def _similarity(a: Chem.Mol | None, b: Chem.Mol | None) -> float:
    if a is None or b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(b, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _frontiers_from_baseline(baseline_routes: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for route in baseline_routes.get("routes") or []:
        for key in ("unresolved_frontiers", "open_leaves", "terminal_reactants"):
            for smi in route.get(key) or []:
                if smi:
                    values.append(str(smi))
    return sorted(set(values))


def _baseline_summary(baseline_routes: dict[str, Any]) -> dict[str, Any]:
    if not baseline_routes:
        return {"status": "not_run", "route_count": 0}
    routes = baseline_routes.get("routes") or []
    return {
        "status": baseline_routes.get("status") or "provided",
        "route_count": len(routes),
        "solved": bool(baseline_routes.get("solved")),
    }


def _safe_case_id(target_name: str, target_smiles: str) -> str:
    raw = target_name or target_smiles[:24] or "target"
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw)
    return "_".join(part for part in safe.split("_") if part)[:80] or "target"
