"""Deterministic, replay-gated extraction of local retrosynthetic templates."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions


TEMPLATE_EXTRACTOR_VERSION = "autoplanner.rdkit_reaction_center.v1"
RDLogger.DisableLog("rdApp.*")


def extract_retro_template(
    mapped_reaction_smiles: str,
    *,
    radius: int = 1,
) -> dict[str, Any]:
    """Extract one local retro SMARTS and require replay of its source example."""

    if not 0 <= radius <= 2:
        return _rejected("template_radius_invalid")
    parsed = _mapped_reaction(mapped_reaction_smiles)
    if parsed is None:
        return _rejected("mapped_reaction_invalid")
    reactants, product = parsed
    reactant_bonds = _mapped_bonds(reactants)
    product_bonds = _mapped_bonds((product,))
    changed = sorted(
        key
        for key in set(reactant_bonds) | set(product_bonds)
        if reactant_bonds.get(key) != product_bonds.get(key)
    )
    reactant_unmapped = _unmapped_neighbor_signatures(reactants)
    product_unmapped = _unmapped_neighbor_signatures((product,))
    environment_changed_maps = sorted(
        map_number
        for map_number in set(reactant_unmapped) | set(product_unmapped)
        if reactant_unmapped.get(map_number) != product_unmapped.get(map_number)
    )
    if not changed and not environment_changed_maps:
        return _rejected("mapped_reaction_has_no_bond_change")
    center_maps = {
        *environment_changed_maps,
        *(value for pair in changed for value in pair),
    }
    product_smarts = _side_smarts((product,), center_maps, radius=radius)
    reactant_smarts = _side_smarts(
        reactants,
        center_maps,
        radius=radius,
        include_unmapped=True,
    )
    if not product_smarts or not reactant_smarts:
        return _rejected("reaction_center_fragment_missing")
    reaction_smarts = f"{product_smarts}>>{reactant_smarts}"
    try:
        reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        if reaction is None:
            raise ValueError("reaction_smarts_invalid")
        reaction.Initialize()
    except (RuntimeError, ValueError):
        return _rejected("reaction_smarts_invalid")
    original_product = _canonical_molecule(product)
    expected = sorted(_canonical_molecule(molecule) for molecule in reactants)
    outcomes = apply_retro_template(reaction_smarts, original_product, max_outcomes=32)
    if expected not in outcomes:
        if radius == 1:
            return extract_retro_template(mapped_reaction_smiles, radius=2)
        return _rejected("source_example_replay_failed")
    identity = {
        "extractor_version": TEMPLATE_EXTRACTOR_VERSION,
        "radius": radius,
        "reaction_smarts": reaction_smarts,
    }
    return {
        "schema_version": "retrosynthetic_reaction_template.v1",
        "accepted": True,
        "template_id": f"template:{_digest(identity)[:24]}",
        **identity,
        "changed_bonds": [list(value) for value in changed],
        "unmapped_environment_changed_maps": environment_changed_maps,
        "center_atom_maps": sorted(center_maps),
        "source_replay": {
            "accepted": True,
            "product_smiles": original_product,
            "expected_precursor_smiles": expected,
            "matching_outcome_count": sum(value == expected for value in outcomes),
        },
        "semantics": {
            "local_reaction_center_only": True,
            "source_replay_is_not_cross_substrate_validation": True,
            "template_grants_no_scientific_authority": True,
        },
    }


def apply_retro_template(
    reaction_smarts: str,
    product_smiles: str,
    *,
    max_outcomes: int = 16,
) -> list[list[str]]:
    """Apply one retro SMARTS with RDKit and return bounded canonical outcomes."""

    if max_outcomes < 1:
        return []
    product = Chem.MolFromSmiles(str(product_smiles or ""))
    if product is None:
        return []
    try:
        reaction = rdChemReactions.ReactionFromSmarts(str(reaction_smarts or ""))
        if reaction is None or reaction.GetNumReactantTemplates() != 1:
            return []
        reaction.Initialize()
        raw_outcomes = reaction.RunReactants((product,), maxProducts=max_outcomes * 4)
    except (RuntimeError, ValueError):
        return []
    outcomes: set[tuple[str, ...]] = set()
    for raw in raw_outcomes:
        values: list[str] = []
        for molecule in raw:
            canonical = _canonical_molecule(molecule)
            if not canonical:
                values = []
                break
            values.append(canonical)
        if values:
            outcomes.add(tuple(sorted(values)))
        if len(outcomes) >= max_outcomes:
            break
    return [list(value) for value in sorted(outcomes)]


def _mapped_reaction(value: str) -> tuple[tuple[Chem.Mol, ...], Chem.Mol] | None:
    parts = str(value or "").split(">")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        return None
    reactants = tuple(Chem.MolFromSmiles(item) for item in parts[0].split(".") if item)
    products = tuple(Chem.MolFromSmiles(item) for item in parts[2].split(".") if item)
    if not reactants or len(products) != 1 or any(value is None for value in reactants):
        return None
    product = products[0]
    if product is None or not _unique_complete_maps((*reactants, product)):
        return None
    product_maps = {atom.GetAtomMapNum() for atom in product.GetAtoms()}
    reactant_maps = {
        atom.GetAtomMapNum() for molecule in reactants for atom in molecule.GetAtoms()
    }
    if not product_maps <= reactant_maps:
        return None
    return reactants, product


def _unique_complete_maps(molecules: tuple[Chem.Mol, ...]) -> bool:
    product = molecules[-1]
    product_maps = [atom.GetAtomMapNum() for atom in product.GetAtoms()]
    if not product_maps or 0 in product_maps or len(product_maps) != len(set(product_maps)):
        return False
    product_values = set(product_maps)
    reactant_values = [
        atom.GetAtomMapNum()
        for molecule in molecules[:-1]
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum()
    ]
    return len(reactant_values) == len(set(reactant_values)) and bool(product_values)


def _mapped_bonds(molecules: tuple[Chem.Mol, ...]) -> dict[tuple[int, int], float]:
    return {
        tuple(
            sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))
        ): bond.GetBondTypeAsDouble()
        for molecule in molecules
        for bond in molecule.GetBonds()
        if bond.GetBeginAtom().GetAtomMapNum()
        and bond.GetEndAtom().GetAtomMapNum()
    }


def _unmapped_neighbor_signatures(
    molecules: tuple[Chem.Mol, ...],
) -> dict[int, tuple[tuple[Any, ...], ...]]:
    values: dict[int, list[tuple[Any, ...]]] = {}
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            map_number = atom.GetAtomMapNum()
            if not map_number:
                continue
            signatures = values.setdefault(map_number, [])
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetAtomMapNum():
                    continue
                signatures.append(
                    (
                        neighbor.GetAtomicNum(),
                        neighbor.GetFormalCharge(),
                        neighbor.GetIsAromatic(),
                        bond.GetBondTypeAsDouble(),
                    )
                )
    return {key: tuple(sorted(value)) for key, value in values.items()}


def _side_smarts(
    molecules: tuple[Chem.Mol, ...],
    center_maps: set[int],
    *,
    radius: int,
    include_unmapped: bool = False,
) -> str:
    fragments: list[str] = []
    for molecule in molecules:
        selected = {
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetAtomMapNum() in center_maps
        }
        frontier = set(selected)
        for _ in range(radius):
            frontier = {
                neighbor.GetIdx()
                for index in frontier
                for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            } - selected
            selected.update(frontier)
        mapped_indices = {
            atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()
        }
        if include_unmapped and not mapped_indices:
            selected.update(atom.GetIdx() for atom in molecule.GetAtoms())
        elif include_unmapped and selected:
            selected.update(
                atom.GetIdx()
                for atom in molecule.GetAtoms()
                if not atom.GetAtomMapNum()
            )
        if not selected:
            continue
        bonds = [
            bond.GetIdx()
            for bond in molecule.GetBonds()
            if bond.GetBeginAtomIdx() in selected and bond.GetEndAtomIdx() in selected
        ]
        fragment = Chem.MolFragmentToSmarts(
            molecule,
            atomsToUse=sorted(selected),
            bondsToUse=bonds,
            isomericSmarts=True,
        )
        fragment = _bind_zero_hydrogen_counts(molecule, selected, fragment)
        if fragment:
            fragments.append(fragment)
    return ".".join(sorted(fragments))


def _bind_zero_hydrogen_counts(
    molecule: Chem.Mol,
    selected: set[int],
    fragment: str,
) -> str:
    """Prevent changed mapped atoms from inheriting stale substrate hydrogens."""

    value = fragment
    for index in selected:
        atom = molecule.GetAtomWithIdx(index)
        map_number = atom.GetAtomMapNum()
        if not map_number or atom.GetTotalNumHs() != 0:
            continue
        pattern = re.compile(rf"(\[[^\]]*?)(:{map_number}\])")

        def add_h0(match: re.Match[str]) -> str:
            prefix, suffix = match.groups()
            return match.group(0) if re.search(r"H\d*", prefix) else f"{prefix}H0{suffix}"

        value = pattern.sub(add_h0, value, count=1)
    return value


def _canonical_molecule(value: Chem.Mol | str) -> str:
    molecule = (
        Chem.Mol(value) if isinstance(value, Chem.Mol) else Chem.MolFromSmiles(str(value))
    )
    if molecule is None:
        return ""
    try:
        Chem.SanitizeMol(molecule)
    except (RuntimeError, ValueError):
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _rejected(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "retrosynthetic_reaction_template.v1",
        "accepted": False,
        "reasons": [reason],
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


__all__ = ["TEMPLATE_EXTRACTOR_VERSION", "apply_retro_template", "extract_retro_template"]
