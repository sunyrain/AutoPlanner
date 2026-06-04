"""RDKit retron matching and product-specific cuts for literature templates."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.agent.literature_templates import (
    LiteratureTemplateCard,
    LiteratureTemplateLevel,
    TemplateApplicabilityReport,
    applicability_report_from_dict,
    template_card_from_dict,
)


RDLogger.DisableLog("rdApp.*")

EXECUTABLE_ALLOWED_USE = "executable_candidate"
ADVISORY_ALLOWED_USE = "advisory_or_rerank_only"
CRITIQUE_ALLOWED_USE = "critique_or_retrieve_more"
FORBIDDEN_ALLOWED_USE = "forbidden"


def assess_template_applicability(
    *,
    target_smiles: str,
    frontier_smiles: str,
    template_card: LiteratureTemplateCard | dict[str, Any],
) -> TemplateApplicabilityReport:
    """Match a literature template retron against the current product/frontier."""
    card = template_card if isinstance(template_card, LiteratureTemplateCard) else template_card_from_dict(template_card)
    product = str(frontier_smiles or target_smiles or "")
    mol = Chem.MolFromSmiles(product)
    if mol is None:
        return TemplateApplicabilityReport(
            target_smiles=str(target_smiles or ""),
            frontier_smiles=str(frontier_smiles or ""),
            match_confidence="none",
            mismatch_reasons=["invalid_frontier_smiles"],
            allowed_use=FORBIDDEN_ALLOWED_USE,
            template_id=card.template_id,
            retron_type=str((card.product_retron or {}).get("retron_type") or ""),
        )
    retron_type = str((card.product_retron or {}).get("retron_type") or "").strip().lower()
    template_level = str(card.template_level or "")
    if template_level == LiteratureTemplateLevel.ADVISORY_STRATEGY.value:
        return _blocked_report(target_smiles, frontier_smiles, card, ["advisory_template_not_executable"])
    if template_level == LiteratureTemplateLevel.ROUTE_ANCHOR_ONLY.value:
        return _blocked_report(target_smiles, frontier_smiles, card, ["route_anchor_only_not_executable"])
    if _analogy_only(card):
        return TemplateApplicabilityReport(
            target_smiles=str(target_smiles or ""),
            frontier_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
            match_confidence="analogy_only",
            mismatch_reasons=["analogy_only_not_executable"],
            allowed_use=CRITIQUE_ALLOWED_USE,
            template_id=card.template_id,
            retron_type=retron_type,
        )

    matchers = {
        "o_glycoside": _match_o_glycoside,
        "c_glycoside": _match_c_glycoside,
        "macrolactone": _match_macrolactone,
        "taxane_c13_side_chain": _match_taxane_side_chain,
        "bufadienolide_c17_pyrone": _match_bufadienolide_c17_pyrone,
        "corey_lactone_side_chain": _match_corey_lactone_side_chain,
    }
    matcher = matchers.get(retron_type)
    if matcher is None:
        return _blocked_report(target_smiles, frontier_smiles, card, ["unsupported_retron_type"])
    matches = matcher(mol)
    if not matches:
        reasons = ["no_retron_match"]
        family_mismatch = _same_family_wrong_linkage(mol, retron_type)
        if family_mismatch:
            reasons = [family_mismatch]
            allowed_use = ADVISORY_ALLOWED_USE
        elif _analogy_only(card):
            allowed_use = CRITIQUE_ALLOWED_USE
        else:
            allowed_use = FORBIDDEN_ALLOWED_USE
        return TemplateApplicabilityReport(
            target_smiles=str(target_smiles or ""),
            frontier_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
            match_confidence="family_mismatch" if family_mismatch else "none",
            mismatch_reasons=reasons,
            allowed_use=allowed_use,
            template_id=card.template_id,
            retron_type=retron_type,
        )
    matches = sorted(matches, key=_match_sort_key)
    selected = dict(matches[0])
    cut_fragments = product_specific_cut(product, selected)
    selected["cut_fragments"] = list(cut_fragments)
    selected["match_rank"] = 1
    return TemplateApplicabilityReport(
        target_smiles=str(target_smiles or ""),
        frontier_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
        matched_retron_atoms=[list(match.get("matched_atoms") or []) for match in matches],
        matched_bonds=[_public_bond(match) for match in matches],
        match_confidence="exact_retron_match",
        mismatch_reasons=[],
        allowed_use=EXECUTABLE_ALLOWED_USE,
        ambiguity_count=max(0, len(matches) - 1),
        selected_bond=_public_bond(selected),
        cut_fragments=list(cut_fragments),
        retron_type=retron_type,
        template_id=card.template_id,
    )


def product_specific_cut(product_smiles: str, matched_bond: dict[str, Any]) -> list[str]:
    mol = Chem.MolFromSmiles(str(product_smiles or ""))
    if mol is None:
        return []
    bond_idx = int(matched_bond.get("bond_idx", -1))
    if bond_idx < 0 or bond_idx >= mol.GetNumBonds():
        return []
    try:
        frag = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
        parts = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    except Exception:
        return []
    fragments = [
        Chem.MolToSmiles(part, isomericSmiles=True)
        for part in parts
        if part is not None and part.GetNumAtoms() > 0
    ]
    return sorted(fragments, key=lambda smi: (-_heavy_atoms(smi), smi))


def cut_report_from_applicability(report_or_data: TemplateApplicabilityReport | dict[str, Any]) -> dict[str, Any]:
    report = (
        report_or_data
        if isinstance(report_or_data, TemplateApplicabilityReport)
        else applicability_report_from_dict(report_or_data)
    )
    return {
        "schema_version": "product_specific_cut_report.v1",
        "frontier_smiles": report.frontier_smiles,
        "selected_bond": dict(report.selected_bond or {}),
        "cut_fragments": list(report.cut_fragments),
        "ambiguity_count": int(report.ambiguity_count or 0),
        "allowed_use": report.allowed_use,
    }


def _match_o_glycoside(mol: Chem.Mol) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        hetero, sugar = _hetero_sugar_pair(a, b)
        if hetero is None or sugar is None:
            continue
        if hetero.GetDegree() < 2:
            continue
        if not _atom_in_sugar_ring(mol, sugar.GetIdx()):
            continue
        matches.append(_match_payload(mol, bond, "o_glycoside", [hetero.GetIdx(), sugar.GetIdx()]))
    return matches


def _match_c_glycoside(mol: Chem.Mol) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() != 6 or b.GetAtomicNum() != 6:
            continue
        sugar = a if _atom_in_sugar_ring(mol, a.GetIdx()) else b if _atom_in_sugar_ring(mol, b.GetIdx()) else None
        aryl = b if sugar is a else a if sugar is b else None
        if sugar is None or aryl is None or not aryl.GetIsAromatic():
            continue
        matches.append(_match_payload(mol, bond, "c_glycoside", [sugar.GetIdx(), aryl.GetIdx()]))
    return matches


def _match_macrolactone(mol: Chem.Mol) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    large_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if len(ring) >= 8]
    if not large_rings:
        return []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or not bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        carbonyl, oxygen = _carbonyl_oxygen_pair(a, b)
        if carbonyl is None or oxygen is None:
            continue
        if not any(carbonyl.GetIdx() in ring and oxygen.GetIdx() in ring for ring in large_rings):
            continue
        matches.append(_match_payload(mol, bond, "macrolactone", [carbonyl.GetIdx(), oxygen.GetIdx()]))
    return matches


def _match_taxane_side_chain(mol: Chem.Mol) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if mol.GetRingInfo().NumRings() < 3:
        return []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        oxygen, carbonyl = _oxygen_carbonyl_pair(a, b)
        if oxygen is None or carbonyl is None:
            continue
        if not _oxygen_attached_to_ring_carbon(oxygen, exclude=carbonyl.GetIdx()):
            continue
        if _carbonyl_substituent_is_small_acetyl(carbonyl, oxygen.GetIdx()):
            continue
        matches.append(_match_payload(mol, bond, "taxane_c13_side_chain", [oxygen.GetIdx(), carbonyl.GetIdx()]))
    return matches


def _match_bufadienolide_c17_pyrone(mol: Chem.Mol) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if mol.GetRingInfo().NumRings() < 4:
        return []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        left_pyrone = _is_pyrone_ring_atom(mol, a.GetIdx())
        right_pyrone = _is_pyrone_ring_atom(mol, b.GetIdx())
        if left_pyrone == right_pyrone:
            continue
        pyrone_atom = a if left_pyrone else b
        steroid_atom = b if left_pyrone else a
        if not steroid_atom.IsInRing() or steroid_atom.GetIsAromatic():
            continue
        matches.append(_match_payload(mol, bond, "bufadienolide_c17_pyrone", [steroid_atom.GetIdx(), pyrone_atom.GetIdx()]))
    return matches


def _match_corey_lactone_side_chain(mol: Chem.Mol) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if mol.GetRingInfo().NumRings() > 3:
        return matches
    if not _has_lactone_ring(mol):
        return matches
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() != 6 or b.GetAtomicNum() != 6:
            continue
        if a.IsInRing() ^ b.IsInRing():
            ring_atom = a if a.IsInRing() else b
            side_atom = b if a.IsInRing() else a
            matches.append(_match_payload(mol, bond, "corey_lactone_side_chain", [ring_atom.GetIdx(), side_atom.GetIdx()]))
    return matches


def _blocked_report(
    target_smiles: str,
    frontier_smiles: str,
    card: LiteratureTemplateCard,
    reasons: list[str],
) -> TemplateApplicabilityReport:
    return TemplateApplicabilityReport(
        target_smiles=str(target_smiles or ""),
        frontier_smiles=str(frontier_smiles or ""),
        match_confidence="none",
        mismatch_reasons=list(reasons),
        allowed_use=FORBIDDEN_ALLOWED_USE,
        template_id=card.template_id,
        retron_type=str((card.product_retron or {}).get("retron_type") or ""),
    )


def _same_family_wrong_linkage(mol: Chem.Mol, retron_type: str) -> str:
    if retron_type == "c_glycoside" and _match_o_glycoside(mol):
        return "same_family_wrong_linkage_o_glycoside"
    if retron_type == "o_glycoside" and _match_c_glycoside(mol):
        return "same_family_wrong_linkage_c_glycoside"
    if retron_type == "macrolactone" and _match_taxane_side_chain(mol):
        return "same_family_wrong_linkage_taxane_ester_not_macrocycle"
    if retron_type == "taxane_c13_side_chain" and _match_macrolactone(mol):
        return "same_family_wrong_linkage_macrocycle_not_taxane"
    return ""


def _analogy_only(card: LiteratureTemplateCard) -> bool:
    text = " ".join([*card.scope_limits, *card.safety_flags, card.promotion_status]).lower()
    return "analogy" in text


def _hetero_sugar_pair(a: Chem.Atom, b: Chem.Atom) -> tuple[Chem.Atom | None, Chem.Atom | None]:
    hetero_nums = {7, 8, 16}
    if a.GetAtomicNum() in hetero_nums and b.GetAtomicNum() == 6:
        return a, b
    if b.GetAtomicNum() in hetero_nums and a.GetAtomicNum() == 6:
        return b, a
    return None, None


def _atom_in_sugar_ring(mol: Chem.Mol, atom_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if not atom.IsInRing() or atom.GetIsAromatic():
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx not in ring or not (5 <= len(ring) <= 7):
            continue
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if any(a.GetAtomicNum() == 8 for a in atoms) and sum(1 for a in atoms if a.GetAtomicNum() == 6) >= 4:
            return True
    return False


def _carbonyl_oxygen_pair(a: Chem.Atom, b: Chem.Atom) -> tuple[Chem.Atom | None, Chem.Atom | None]:
    if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 8 and _has_double_bond_oxygen(a):
        return a, b
    if b.GetAtomicNum() == 6 and a.GetAtomicNum() == 8 and _has_double_bond_oxygen(b):
        return b, a
    return None, None


def _oxygen_carbonyl_pair(a: Chem.Atom, b: Chem.Atom) -> tuple[Chem.Atom | None, Chem.Atom | None]:
    carbonyl, oxygen = _carbonyl_oxygen_pair(a, b)
    return oxygen, carbonyl


def _has_double_bond_oxygen(atom: Chem.Atom) -> bool:
    if atom.GetAtomicNum() != 6:
        return False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def _oxygen_attached_to_ring_carbon(oxygen: Chem.Atom, *, exclude: int) -> bool:
    for nbr in oxygen.GetNeighbors():
        if nbr.GetIdx() == exclude:
            continue
        if nbr.GetAtomicNum() == 6 and nbr.IsInRing():
            return True
    return False


def _carbonyl_substituent_is_small_acetyl(carbonyl: Chem.Atom, oxygen_idx: int) -> bool:
    carbon_neighbors = [
        nbr
        for nbr in carbonyl.GetNeighbors()
        if nbr.GetIdx() != oxygen_idx and nbr.GetAtomicNum() == 6
    ]
    if len(carbon_neighbors) != 1:
        return False
    nbr = carbon_neighbors[0]
    return not nbr.IsInRing() and nbr.GetDegree() <= 1


def _is_pyrone_ring_atom(mol: Chem.Mol, atom_idx: int) -> bool:
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx not in ring or len(ring) not in {5, 6}:
            continue
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if not any(atom.GetAtomicNum() == 8 for atom in atoms):
            continue
        if any(_has_double_bond_oxygen(atom) for atom in atoms if atom.GetAtomicNum() == 6):
            return True
    return False


def _has_lactone_ring(mol: Chem.Mol) -> bool:
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            continue
        carbonyl, oxygen = _carbonyl_oxygen_pair(bond.GetBeginAtom(), bond.GetEndAtom())
        if carbonyl is not None and oxygen is not None:
            return True
    return False


def _match_payload(mol: Chem.Mol, bond: Chem.Bond, retron_type: str, atoms: list[int]) -> dict[str, Any]:
    a_idx, b_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    return {
        "bond_idx": int(bond.GetIdx()),
        "atom_indices": [int(a_idx), int(b_idx)],
        "matched_atoms": [int(idx) for idx in atoms],
        "bond_type": str(bond.GetBondType()),
        "bond_in_ring": bool(bond.IsInRing()),
        "retron_type": retron_type,
        "atom_symbols": [mol.GetAtomWithIdx(a_idx).GetSymbol(), mol.GetAtomWithIdx(b_idx).GetSymbol()],
        "ring_count": int(mol.GetRingInfo().NumRings()),
    }


def _public_bond(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "bond_idx": int(match.get("bond_idx", -1)),
        "atom_indices": [int(idx) for idx in match.get("atom_indices") or []],
        "matched_atoms": [int(idx) for idx in match.get("matched_atoms") or []],
        "bond_type": str(match.get("bond_type") or ""),
        "bond_in_ring": bool(match.get("bond_in_ring")),
        "retron_type": str(match.get("retron_type") or ""),
        "atom_symbols": [str(item) for item in match.get("atom_symbols") or []],
    }


def _match_sort_key(match: dict[str, Any]) -> tuple[int, int, list[int]]:
    # Prefer non-ring strategic appendages, then stable atom/bond order.
    return (1 if match.get("bond_in_ring") else 0, int(match.get("bond_idx", 10**9)), list(match.get("atom_indices") or []))


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0
