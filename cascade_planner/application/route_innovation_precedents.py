"""Adapt retrieved enzyme reactions into proposal-only capability records."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.route_innovation_capabilities import (
    normalize_biocatalysis_catalog,
    structure_transition,
)


def capabilities_from_enzyme_precedents(
    values: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile retrieved reactions without granting exact-substrate authority."""

    raw_capabilities: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw in values:
        row = dict(raw)
        evidence = dict(row.get("evidence") or {})
        substrate_selection = evidence.get("substrate_component_selection")
        substrate_selection = (
            dict(substrate_selection)
            if isinstance(substrate_selection, Mapping)
            else {}
        )
        reaction_id = str(
            row.get("precedent_reaction_id")
            or evidence.get("reaction_id")
            or ""
        )
        substrate = str(
            substrate_selection.get("main_smiles")
            or row.get("main_reactant")
            or ""
        )
        product = str(
            evidence.get("precedent_product_main_smiles")
            or evidence.get("precedent_product_smiles")
            or ""
        )
        transition = structure_transition(substrate, product)
        motif_delta = {
            key: int(amount)
            for key, amount in dict(transition.get("motif_delta") or {}).items()
            if int(amount) != 0
        }
        ec_numbers = _strings(
            row.get("enzyme_ec_numbers")
            or evidence.get("ec_numbers")
            or ([row.get("ec")] if row.get("ec") else [])
        )
        if not reaction_id or not motif_delta or not ec_numbers:
            skipped.append(
                {
                    "reaction_id": reaction_id,
                    "reasons": ["precedent_capability_transition_or_enzyme_missing"],
                }
            )
            continue
        rhea_ids = _strings(row.get("rhea_ids") or evidence.get("rhea_ids") or [])
        refs = [*(f"rhea:{value}" for value in rhea_ids), f"enzyme-reaction:{reaction_id}"]
        similarity = float(transition.get("scaffold_similarity") or 0.0)
        raw_capabilities.append(
            {
                "capability_id": f"enzyme-precedent:{reaction_id}",
                "label": f"Retrieved enzyme transition {reaction_id}",
                "enzyme": {
                    "ec_numbers": ec_numbers,
                    "classes": [],
                    "candidate_ids": [reaction_id],
                },
                "match": {
                    "net_motif_delta": motif_delta,
                    "preserved_motifs": [
                        key
                        for key, amount in dict(
                            transition.get("motif_delta") or {}
                        ).items()
                        if int(amount) == 0
                    ],
                    "element_delta": dict(transition.get("element_delta") or {}),
                    "min_scaffold_similarity": max(0.3, similarity - 0.15),
                    "max_abs_heavy_atom_delta": abs(
                        int(transition.get("heavy_atom_delta") or 0)
                    ),
                    "min_substrate_carbons": max(
                        0, int(transition.get("substrate_carbon_count") or 0) - 4
                    ),
                    "min_substrate_rings": max(
                        0, int(transition.get("substrate_ring_count") or 0) - 1
                    ),
                    "min_window_steps": 1,
                    "max_window_steps": 8,
                    "reject_unlisted_motif_changes": True,
                },
                "selectivity_objective": (
                    "Reproduce the retrieved net transformation and requested "
                    "product stereochemical state on the route boundary."
                ),
                "substrate_scope_basis": (
                    "Retrieved enzyme precedent only; exact route substrate is "
                    "outside validated scope."
                ),
                "precedent_refs": refs,
            }
        )
    accepted, rejected = normalize_biocatalysis_catalog(raw_capabilities)
    return {
        "schema_version": "enzyme_precedent_capability_adaptation.v1",
        "capabilities": accepted,
        "rejected": [*skipped, *rejected],
        "semantics": {
            "retrieval_is_search_prior_only": True,
            "exact_substrate_validation_required": True,
        },
    }


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return sorted({str(item).strip() for item in value or [] if str(item).strip()})


__all__ = ["capabilities_from_enzyme_precedents"]
