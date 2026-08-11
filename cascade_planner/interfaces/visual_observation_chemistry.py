"""Host chemistry helpers for advisory visual source observations."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

from rdkit import Chem

from cascade_planner.routes.admission import audit_retrosynthetic_candidate


def canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else ""


def connectivity_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(molecule, isomericSmiles=False) if molecule else ""


def canonical_reactants(value: Any) -> list[str]:
    values = raw_reactants(value)
    result = sorted(canonical_smiles(row) for row in values)
    return result if result and all(result) else []


def raw_reactants(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else [str(row) for row in value or []]


def atom_contributing_reactant_partition(
    product: str,
    reactants: list[str],
) -> dict[str, Any]:
    full_audit = audit_retrosynthetic_candidate(product, reactants)
    accepted_subsets = []
    for count in range(len(reactants), 0, -1):
        for subset in combinations(reactants, count):
            audit = audit_retrosynthetic_candidate(product, list(subset))
            if audit.get("accepted") is True:
                product_counts = dict(audit.get("product_element_counts") or {})
                precursor_counts = dict(audit.get("precursor_element_counts") or {})
                surplus = sum(
                    max(0, int(precursor_counts.get(element) or 0) - int(count_value))
                    for element, count_value in product_counts.items()
                ) + sum(
                    int(count_value)
                    for element, count_value in precursor_counts.items()
                    if element not in product_counts
                )
                accepted_subsets.append(
                    (
                        (
                            surplus,
                            max(
                                0,
                                int(audit.get("precursor_heavy_atom_count") or 0)
                                - int(audit.get("product_heavy_atom_count") or 0),
                            ),
                            -len(subset),
                            tuple(sorted(subset)),
                        ),
                        tuple(sorted(subset)),
                        audit,
                    )
                )
    if accepted_subsets:
        accepted_subsets.sort(key=lambda value: value[0])
        _score, selected, _audit = accepted_subsets[0]
        selected_counts = Counter(selected)
        spectators = []
        for reactant in reactants:
            if selected_counts[reactant] > 0:
                selected_counts[reactant] -= 1
            else:
                spectators.append(reactant)
        return {
            "schema_version": "visual_reactant_partition.v1",
            "accepted": True,
            "precursor_smiles": list(selected),
            "spectator_smiles": sorted(spectators),
            "partition_mode": (
                "all_reactants_atom_contributing"
                if not spectators
                else "host_inventory_conserving_subset"
            ),
            "admission_reasons": [],
        }
    return {
        "schema_version": "visual_reactant_partition.v1",
        "accepted": False,
        "precursor_smiles": list(reactants),
        "spectator_smiles": [],
        "partition_mode": "no_host_admitted_subset",
        "admission_reasons": list(full_audit.get("reasons") or []),
    }


def partition_reactant_labels(
    raw_reactants: list[str],
    raw_labels: Any,
    *,
    precursor_smiles: list[str],
) -> tuple[list[str], list[str]]:
    labels = [str(value)[:300] for value in raw_labels or []]
    selected_counts = Counter(precursor_smiles)
    selected_labels: list[str] = []
    spectator_labels: list[str] = []
    for index, raw in enumerate(raw_reactants):
        canonical = canonical_smiles(raw)
        label = labels[index] if index < len(labels) else ""
        if selected_counts[canonical] > 0:
            selected_counts[canonical] -= 1
            if label:
                selected_labels.append(label)
        elif label:
            spectator_labels.append(label)
    return selected_labels[:12], spectator_labels[:12]


__all__ = [
    "atom_contributing_reactant_partition",
    "canonical_reactants",
    "canonical_smiles",
    "connectivity_smiles",
    "partition_reactant_labels",
    "raw_reactants",
]
