"""Fail-closed, product-grounded repair of narrow precursor typos.

The repair never grants reaction proof.  It only creates a new L0 proposal when
RXNMapper exposes exactly one unmapped carbon inserted into a nucleophile ring
while the product contains the corresponding ring-contracted fragment.  The
new proposal must still pass normal admission, mapping, and reaction replay.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable

from rdkit import Chem


PRECURSOR_REPAIR_SCHEMA = "product_grounded_precursor_repair.v1"


def propose_precursor_repair(
    *,
    mapped_reaction_smiles: str,
    product_smiles: str,
    precursor_smiles: Iterable[str],
) -> dict[str, Any]:
    """Try bounded generic repairs; every accepted result remains an L0 proposal."""

    precursors = tuple(precursor_smiles)
    connectivity = _propose_aryl_isothiocyanate_connectivity_repair(
        mapped_reaction_smiles=mapped_reaction_smiles,
        product_smiles=product_smiles,
        precursor_smiles=precursors,
    )
    if connectivity.get("accepted") is True:
        return connectivity
    return propose_ring_size_typo_repair(
        mapped_reaction_smiles=mapped_reaction_smiles,
        product_smiles=product_smiles,
        precursor_smiles=precursors,
    )


def propose_ring_size_typo_repair(
    *,
    mapped_reaction_smiles: str,
    product_smiles: str,
    precursor_smiles: Iterable[str],
) -> dict[str, Any]:
    """Return one auditable repair proposal or a fail-closed rejection."""

    base = {
        "schema_version": PRECURSOR_REPAIR_SCHEMA,
        "accepted": False,
        "repair_kind": "single_unmapped_ring_carbon_contraction",
        "product_smiles": _canonical(product_smiles),
        "original_precursor_smiles": sorted(
            value for value in (_canonical(item) for item in precursor_smiles) if value
        ),
        "repaired_precursor_smiles": [],
        "reasons": [],
        "semantics": {
            "product_grounded_hypothesis_only": True,
            "grants_no_reaction_proof": True,
            "normal_host_revalidation_required": True,
        },
    }
    parts = str(mapped_reaction_smiles or "").split(">")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        return _rejected(base, "mapped_reaction_invalid")
    reactant_mols = [
        Chem.MolFromSmiles(value) for value in parts[0].split(".") if value
    ]
    product_mol = Chem.MolFromSmiles(parts[2])
    if (
        len(reactant_mols) < 2
        or any(mol is None for mol in reactant_mols)
        or product_mol is None
    ):
        return _rejected(base, "mapped_reaction_not_materialized")

    component_by_map: dict[int, int] = {}
    element_by_map: dict[int, int] = {}
    for index, mol in enumerate(reactant_mols):
        for atom in mol.GetAtoms():
            map_num = int(atom.GetAtomMapNum())
            if map_num > 0:
                component_by_map[map_num] = index
                element_by_map[map_num] = int(atom.GetAtomicNum())
    reactant_bonds = _mapped_bonds(reactant_mols)
    product_bonds = _mapped_bonds([product_mol])
    formed = sorted(product_bonds - reactant_bonds)
    candidates: list[tuple[int, int, int, int]] = []
    for left, right, order in formed:
        if order != "SINGLE":
            continue
        carbon, hetero = _ordered_carbon_hetero(left, right, element_by_map)
        if not carbon or not hetero:
            continue
        carbon_component = component_by_map.get(carbon, -1)
        hetero_component = component_by_map.get(hetero, -1)
        if carbon_component < 0 or hetero_component < 0 or carbon_component == hetero_component:
            continue
        if not _has_unmapped_halide_neighbor(
            reactant_mols[carbon_component],
            carbon,
        ):
            continue
        candidates.append((carbon, hetero, carbon_component, hetero_component))
    if len(candidates) != 1:
        return _rejected(base, "single_cross_component_substitution_not_identified")

    _carbon, _hetero, _electrophile_index, nucleophile_index = candidates[0]
    nucleophile = reactant_mols[nucleophile_index]
    reactant_extra_atoms = [
        atom
        for atom in nucleophile.GetAtoms()
        if int(atom.GetAtomMapNum()) <= 0 and atom.GetAtomicNum() > 1
    ]
    nucleophile_maps = {
        int(atom.GetAtomMapNum())
        for atom in nucleophile.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    product_extra_atoms = [
        atom
        for atom in product_mol.GetAtoms()
        if component_by_map.get(int(atom.GetAtomMapNum())) != nucleophile_index
        and atom.GetAtomicNum() > 1
        and atom.IsInRing()
        and atom.GetDegree() == 2
        and all(
            int(neighbor.GetAtomMapNum()) in nucleophile_maps
            for neighbor in atom.GetNeighbors()
        )
    ]
    keep_product_indices: set[int] = set()
    if len(reactant_extra_atoms) == 1 and not product_extra_atoms:
        extra = reactant_extra_atoms[0]
        if (
            extra.GetAtomicNum() != 6
            or not extra.IsInRing()
            or extra.GetDegree() != 2
        ):
            return _rejected(base, "unmapped_atom_is_not_one_ring_carbon")
        neighbor_maps = sorted(int(atom.GetAtomMapNum()) for atom in extra.GetNeighbors())
        if len(neighbor_maps) != 2 or any(value <= 0 for value in neighbor_maps):
            return _rejected(base, "ring_carbon_neighbors_not_mapped")
        expected_closure = (min(neighbor_maps), max(neighbor_maps), "SINGLE")
        if expected_closure not in product_bonds:
            return _rejected(base, "product_does_not_close_contracted_ring")
        repair_kind = "single_unmapped_ring_carbon_contraction"
        atom_delta = -1
    elif not reactant_extra_atoms and len(product_extra_atoms) == 1:
        extra = product_extra_atoms[0]
        if extra.GetAtomicNum() != 6:
            return _rejected(base, "unmapped_atom_is_not_one_ring_carbon")
        neighbor_maps = sorted(int(atom.GetAtomMapNum()) for atom in extra.GetNeighbors())
        expected_closure = (min(neighbor_maps), max(neighbor_maps), "SINGLE")
        if expected_closure not in reactant_bonds:
            return _rejected(base, "reactant_does_not_close_smaller_ring")
        keep_product_indices.add(int(extra.GetIdx()))
        repair_kind = "single_unmapped_ring_carbon_expansion"
        atom_delta = 1
    else:
        return _rejected(base, "single_ring_carbon_size_error_not_found")

    repaired_mol = _mapped_product_fragment(
        product_mol,
        nucleophile_maps,
        keep_unmapped_indices=keep_product_indices,
    )
    if repaired_mol is None:
        return _rejected(base, "product_grounded_fragment_invalid")
    original_component = _canonical_mol_without_maps(nucleophile)
    repaired_component = _canonical_mol_without_maps(repaired_mol)
    if not original_component or not repaired_component or original_component == repaired_component:
        return _rejected(base, "repair_does_not_change_precursor")
    original_counts = _element_counts(nucleophile)
    repaired_counts = _element_counts(repaired_mol)
    removed = original_counts - repaired_counts
    added = repaired_counts - original_counts
    if (atom_delta == -1 and (removed != Counter({6: 1}) or added)) or (
        atom_delta == 1 and (added != Counter({6: 1}) or removed)
    ):
        return _rejected(base, "repair_is_not_exactly_one_carbon_ring_size_change")

    original_precursors = list(base["original_precursor_smiles"])
    if original_precursors.count(original_component) != 1:
        return _rejected(base, "mapped_nucleophile_not_uniquely_bound_to_input")
    repaired_precursors = list(original_precursors)
    repaired_precursors[repaired_precursors.index(original_component)] = repaired_component
    repaired_precursors.sort()
    base.update(
        {
            "accepted": True,
            "repair_kind": repair_kind,
            "original_component_smiles": original_component,
            "repaired_component_smiles": repaired_component,
            "repaired_precursor_smiles": repaired_precursors,
            "carbon_atom_delta": atom_delta,
            "closed_neighbor_atom_maps": neighbor_maps,
            "reasons": [],
        }
    )
    base["content_sha256"] = _digest(base)
    return base


def _propose_aryl_isothiocyanate_connectivity_repair(
    *,
    mapped_reaction_smiles: str,
    product_smiles: str,
    precursor_smiles: Iterable[str],
) -> dict[str, Any]:
    """Swap the unique Ar-S=C=N connectivity typo to product-grounded Ar-N=C=S."""

    original_precursors = sorted(
        value for value in (_canonical(item) for item in precursor_smiles) if value
    )
    base = {
        "schema_version": PRECURSOR_REPAIR_SCHEMA,
        "accepted": False,
        "repair_kind": "aryl_isothiocyanate_connectivity_swap",
        "product_smiles": _canonical(product_smiles),
        "original_precursor_smiles": original_precursors,
        "repaired_precursor_smiles": [],
        "reasons": [],
        "semantics": {
            "product_grounded_hypothesis_only": True,
            "grants_no_reaction_proof": True,
            "normal_host_revalidation_required": True,
        },
    }
    parts = str(mapped_reaction_smiles or "").split(">")
    product = Chem.MolFromSmiles(str(product_smiles or ""))
    if len(parts) != 3 or not parts[0] or not parts[2] or product is None:
        return _rejected(base, "mapped_reaction_invalid")

    candidates: list[tuple[str, Any, tuple[int, int, int, int]]] = []
    for precursor in original_precursors:
        mol = Chem.MolFromSmiles(precursor)
        if mol is None:
            continue
        for sulfur in mol.GetAtoms():
            if sulfur.GetAtomicNum() != 16:
                continue
            aromatic = [
                neighbor
                for neighbor in sulfur.GetNeighbors()
                if neighbor.GetAtomicNum() == 6 and neighbor.GetIsAromatic()
            ]
            central = [
                neighbor
                for neighbor in sulfur.GetNeighbors()
                if neighbor.GetAtomicNum() == 6
                and str(mol.GetBondBetweenAtoms(sulfur.GetIdx(), neighbor.GetIdx()).GetBondType())
                == "DOUBLE"
            ]
            if len(aromatic) != 1 or len(central) != 1:
                continue
            nitrogens = [
                neighbor
                for neighbor in central[0].GetNeighbors()
                if neighbor.GetIdx() != sulfur.GetIdx()
                and neighbor.GetAtomicNum() == 7
                and str(
                    mol.GetBondBetweenAtoms(
                        central[0].GetIdx(), neighbor.GetIdx()
                    ).GetBondType()
                )
                == "DOUBLE"
            ]
            if len(nitrogens) != 1:
                continue
            candidates.append(
                (
                    precursor,
                    mol,
                    (
                        aromatic[0].GetIdx(),
                        sulfur.GetIdx(),
                        central[0].GetIdx(),
                        nitrogens[0].GetIdx(),
                    ),
                )
            )
    if len(candidates) != 1:
        return _rejected(base, "single_aryl_thioisocyanate_typo_not_identified")

    original_component, mol, (aryl_index, sulfur_index, _carbon_index, nitrogen_index) = (
        candidates[0]
    )
    editable = Chem.RWMol(mol)
    editable.RemoveBond(aryl_index, sulfur_index)
    editable.AddBond(aryl_index, nitrogen_index, Chem.BondType.SINGLE)
    sulfur = editable.GetAtomWithIdx(sulfur_index)
    nitrogen = editable.GetAtomWithIdx(nitrogen_index)
    sulfur.SetNumExplicitHs(0)
    sulfur.SetNoImplicit(True)
    nitrogen.SetNumExplicitHs(0)
    nitrogen.SetNoImplicit(True)
    repaired_mol = editable.GetMol()
    try:
        Chem.SanitizeMol(repaired_mol)
    except Exception:
        return _rejected(base, "repaired_isothiocyanate_invalid")
    repaired_component = Chem.MolToSmiles(
        repaired_mol,
        canonical=True,
        isomericSmiles=True,
    )
    if not repaired_component or repaired_component == original_component:
        return _rejected(base, "repair_does_not_change_precursor")
    if _element_counts(mol) != _element_counts(repaired_mol):
        return _rejected(base, "repair_changes_element_inventory")

    query_parameters = Chem.AdjustQueryParameters()
    query_parameters.makeBondsGeneric = True
    product_query = Chem.AdjustQueryProperties(repaired_mol, query_parameters)
    if not product.HasSubstructMatch(product_query):
        return _rejected(base, "repaired_connectivity_not_grounded_in_product")
    if original_precursors.count(original_component) != 1:
        return _rejected(base, "precursor_component_not_unique")

    repaired_precursors = list(original_precursors)
    repaired_precursors[repaired_precursors.index(original_component)] = (
        repaired_component
    )
    repaired_precursors.sort()
    base.update(
        {
            "accepted": True,
            "original_component_smiles": original_component,
            "repaired_component_smiles": repaired_component,
            "repaired_precursor_smiles": repaired_precursors,
            "atom_delta": 0,
            "connectivity_change": "aryl_sulfur_to_aryl_nitrogen",
            "reasons": [],
        }
    )
    base["content_sha256"] = _digest(base)
    return base


def _mapped_product_fragment(
    mol: Any,
    keep_maps: set[int],
    *,
    keep_unmapped_indices: set[int] | None = None,
) -> Any | None:
    editable = Chem.RWMol(mol)
    keep_indices = set(keep_unmapped_indices or ())
    boundary_maps = {
        int(atom.GetAtomMapNum())
        for atom in editable.GetAtoms()
        if int(atom.GetAtomMapNum()) in keep_maps
        and any(
            int(neighbor.GetAtomMapNum()) not in keep_maps
            and neighbor.GetIdx() not in keep_indices
            for neighbor in atom.GetNeighbors()
        )
    }
    remove = [
        atom.GetIdx()
        for atom in editable.GetAtoms()
        if int(atom.GetAtomMapNum()) not in keep_maps and atom.GetIdx() not in keep_indices
    ]
    for index in sorted(remove, reverse=True):
        editable.RemoveAtom(index)
    result = editable.GetMol()
    for atom in result.GetAtoms():
        if int(atom.GetAtomMapNum()) in boundary_maps:
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
            atom.SetNumRadicalElectrons(0)
            atom.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(result)
    except Exception:
        return None
    if len(Chem.GetMolFrags(result)) != 1:
        return None
    return result


def _mapped_bonds(mols: Iterable[Any]) -> set[tuple[int, int, str]]:
    bonds: set[tuple[int, int, str]] = set()
    for mol in mols:
        for bond in mol.GetBonds():
            left = int(bond.GetBeginAtom().GetAtomMapNum())
            right = int(bond.GetEndAtom().GetAtomMapNum())
            if left <= 0 or right <= 0:
                continue
            bonds.add((min(left, right), max(left, right), str(bond.GetBondType())))
    return bonds


def _ordered_carbon_hetero(
    left: int,
    right: int,
    elements: dict[int, int],
) -> tuple[int, int]:
    if elements.get(left) == 6 and elements.get(right) in {7, 8, 16}:
        return left, right
    if elements.get(right) == 6 and elements.get(left) in {7, 8, 16}:
        return right, left
    return 0, 0


def _has_unmapped_halide_neighbor(mol: Any, carbon_map: int) -> bool:
    return any(
        int(atom.GetAtomMapNum()) == carbon_map
        and any(
            int(neighbor.GetAtomMapNum()) <= 0
            and neighbor.GetAtomicNum() in {9, 17, 35, 53}
            for neighbor in atom.GetNeighbors()
        )
        for atom in mol.GetAtoms()
    )


def _canonical_mol_without_maps(mol: Any) -> str:
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(copy)
    except Exception:
        return ""
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def _canonical(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value or ""))
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if mol is not None
        else ""
    )


def _element_counts(mol: Any) -> Counter[int]:
    return Counter(
        int(atom.GetAtomicNum())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    )


def _rejected(base: dict[str, Any], reason: str) -> dict[str, Any]:
    row = {**base, "reasons": [reason]}
    row["content_sha256"] = _digest(row)
    return row


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "PRECURSOR_REPAIR_SCHEMA",
    "propose_precursor_repair",
    "propose_ring_size_typo_repair",
]
