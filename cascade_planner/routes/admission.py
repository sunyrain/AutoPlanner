"""Shared fail-closed admission checks for retrosynthetic hyperedges."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
RETROSYNTHETIC_ADMISSION_SCHEMA = "retrosynthetic_candidate_admission.v1"


@dataclass(frozen=True)
class RetrosyntheticAdmissionPolicy:
    """Cheap structural checks applied before a proposal enters search state."""

    hard_filter_element_inventory: bool = True
    max_tolerated_missing_heavy_atoms: int = 3
    strict_small_product_heavy_atom_threshold: int = 12
    hard_filter_large_atom_jump: bool = True
    large_atom_jump_threshold: int = 15
    hard_filter_self_loop: bool = True
    hard_filter_surplus_advanced_fragment: bool = True
    surplus_advanced_fragment_heavy_atom_threshold: int = 8


def audit_retrosynthetic_candidate(
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
    *,
    forbidden_return_smiles: Iterable[Any] = (),
    policy: RetrosyntheticAdmissionPolicy | None = None,
) -> dict[str, Any]:
    """Audit one exact product/precursor multiset without granting proof.

    The allowance for a few missing product atoms reflects omitted transfer
    reagents in one-step model output. It is deliberately disabled for small
    products, where such an omission would amount to most of the structure.
    """

    active = policy or RetrosyntheticAdmissionPolicy()
    product = _canonical_smiles(product_smiles)
    raw_precursors = list(precursor_smiles or [])
    precursors = [_canonical_smiles(item) for item in raw_precursors]
    reasons: list[str] = []
    if not product or not raw_precursors or any(not item for item in precursors):
        reasons.append("invalid_or_missing_material")
    product_counts = _element_counts(product)
    precursor_counts: Counter[str] = Counter()
    for precursor in precursors:
        precursor_counts.update(_element_counts(precursor))
    if not product_counts or not precursor_counts:
        if "invalid_or_missing_material" not in reasons:
            reasons.append("invalid_or_missing_material")

    if active.hard_filter_self_loop and product and product in precursors:
        reasons.append("target_or_current_node_self_loop")
    forbidden = {
        canonical
        for canonical in (_canonical_smiles(item) for item in forbidden_return_smiles)
        if canonical
    }
    ancestor_returns = sorted(set(precursors).intersection(forbidden - {product}))
    if ancestor_returns:
        reasons.append("ancestor_or_target_cycle")

    deficits = {
        element: count - int(precursor_counts.get(element, 0))
        for element, count in product_counts.items()
        if count > int(precursor_counts.get(element, 0))
    }
    missing_heavy_atoms = sum(deficits.values())
    product_heavy_atoms = sum(product_counts.values())
    precursor_heavy_atoms = sum(precursor_counts.values())
    surplus_advanced_fragments: list[str] = []
    if (
        active.hard_filter_surplus_advanced_fragment
        and product_counts
        and len(precursors) > 1
    ):
        component_counts = [_element_counts(precursor) for precursor in precursors]
        for index, counts in enumerate(component_counts):
            component_heavy_atoms = sum(counts.values())
            if (
                component_heavy_atoms
                < active.surplus_advanced_fragment_heavy_atom_threshold
            ):
                continue
            other_counts: Counter[str] = Counter()
            for other_index, other in enumerate(component_counts):
                if other_index != index:
                    other_counts.update(other)
            if all(
                int(other_counts.get(element, 0)) >= count
                for element, count in product_counts.items()
            ):
                surplus_advanced_fragments.append(precursors[index])
        if surplus_advanced_fragments:
            reasons.append("surplus_advanced_precursor_fragment")
    if (
        active.hard_filter_element_inventory
        and deficits
        and (
            missing_heavy_atoms > active.max_tolerated_missing_heavy_atoms
            or product_heavy_atoms <= active.strict_small_product_heavy_atom_threshold
        )
    ):
        reasons.append("element_inventory_not_conserved")
    if (
        active.hard_filter_large_atom_jump
        and product_heavy_atoms - precursor_heavy_atoms
        >= active.large_atom_jump_threshold
    ):
        reasons.append("large_atom_jump")

    return {
        "schema_version": RETROSYNTHETIC_ADMISSION_SCHEMA,
        "accepted": not reasons,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "forbidden_return_smiles": sorted(forbidden),
        "ancestor_return_smiles": ancestor_returns,
        "product_element_counts": dict(sorted(product_counts.items())),
        "precursor_element_counts": dict(sorted(precursor_counts.items())),
        "element_deficits": dict(sorted(deficits.items())),
        "missing_product_heavy_atom_count": missing_heavy_atoms,
        "product_heavy_atom_count": product_heavy_atoms,
        "precursor_heavy_atom_count": precursor_heavy_atoms,
        "surplus_advanced_precursor_fragments": surplus_advanced_fragments,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "search_admission_only": True,
            "not_reaction_proof": True,
            "precursor_multiplicity_preserved": True,
            "large_inventory_redundancy_is_not_joint_participation": True,
            "small_salts_and_leaving_groups_are_exempt": True,
        },
    }


def _canonical_smiles(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _element_counts(smiles: str) -> Counter[str]:
    molecule = Chem.MolFromSmiles(smiles or "")
    if molecule is None:
        return Counter()
    return Counter(
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    )
