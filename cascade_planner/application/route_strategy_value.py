"""Evidence-independent strategy value and independent maturity projections."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def compile_strategic_value_vector(
    graph: Mapping[str, Any],
    *,
    edge_ids: Iterable[str],
    root_edge_ids: Iterable[str],
    strategy_card: Mapping[str, Any],
    convergence_score: float,
) -> dict[str, Any]:
    del edge_ids
    root_fragment_count = max(
        (
            len(
                dict(graph.get("edges") or {})
                .get(edge_id, {})
                .get("precursor_molecule_ids")
                or []
            )
            for edge_id in root_edge_ids
        ),
        default=0,
    )
    edit_signature = dict(strategy_card.get("reaction_edit_signature") or {})
    changed_pairs = list(edit_signature.get("changed_map_pairs") or [])
    key_bonds = len(
        changed_pairs
        or strategy_card.get("key_bond_signature")
        or strategy_card.get("key_bond_changes")
        or []
    )
    key_bond_leverage = min(
        1.0,
        (key_bonds + max(0, root_fragment_count - 1)) / 3.0,
    )
    structure = _root_structure_metrics(graph, root_edge_ids=root_edge_ids)
    topology_transform = (
        1.0
        if float(structure.get("ring_count_drop") or 0.0) > 0.0
        else 0.7
        if root_fragment_count > 1
        else 0.45
        if changed_pairs
        else 0.2
    )
    complexity_drop = float(structure.get("complexity_drop") or 0.0)
    stereo_text = str(strategy_card.get("stereochemical_plan") or "").strip().lower()
    stereochemical_leverage = (
        0.0
        if not stereo_text
        else 0.35
        if stereo_text in {"none", "not applicable", "n/a"}
        else 0.8
    )
    protection_text = str(strategy_card.get("protection_policy") or "").strip().lower()
    protection_efficiency = (
        1.0
        if any(
            token in protection_text
            for token in ("avoid", "none", "minimal", "protecting-group-free")
        )
        else 0.55
        if protection_text
        else 0.25
    )
    root_convergence = min(
        1.0,
        max(float(convergence_score), max(0, root_fragment_count - 1) / 2.0),
    )
    score = (
        0.25 * key_bond_leverage
        + 0.22 * topology_transform
        + 0.20 * complexity_drop
        + 0.13 * stereochemical_leverage
        + 0.13 * root_convergence
        + 0.07 * protection_efficiency
    )
    return {
        "key_bond_leverage": round(key_bond_leverage, 6),
        "topology_transformation_value": round(topology_transform, 6),
        "complexity_drop": round(complexity_drop, 6),
        "declared_complexity_drop": str(
            strategy_card.get("expected_complexity_drop") or ""
        ),
        "structure_metrics": structure,
        "stereochemical_leverage": round(stereochemical_leverage, 6),
        "convergence": round(root_convergence, 6),
        "protection_efficiency": round(protection_efficiency, 6),
        "score": round(score, 6),
        "basis": "canonical_root_structures_and_strategy_edit_identity",
    }


def _root_structure_metrics(
    graph: Mapping[str, Any],
    *,
    root_edge_ids: Iterable[str],
) -> dict[str, Any]:
    molecules = dict(graph.get("molecules") or {})
    edges = dict(graph.get("edges") or {})
    observations: list[dict[str, float]] = []
    for edge_id in root_edge_ids:
        edge = dict(edges.get(str(edge_id)) or {})
        product = _molecule_metrics(
            dict(molecules.get(str(edge.get("product_molecule_id") or "")) or {}).get(
                "canonical_smiles"
            )
        )
        precursors = [
            _molecule_metrics(
                dict(molecules.get(str(molecule_id)) or {}).get("canonical_smiles")
            )
            for molecule_id in edge.get("precursor_molecule_ids") or []
        ]
        precursors = [value for value in precursors if value]
        if not product or not precursors:
            continue
        product_complexity = max(1.0, float(product["bertz_complexity"]))
        hardest_precursor = max(
            float(value["bertz_complexity"]) for value in precursors
        )
        observations.append(
            {
                "complexity_drop": max(
                    0.0,
                    min(1.0, (product_complexity - hardest_precursor) / product_complexity),
                ),
                "ring_count_drop": max(
                    0.0,
                    float(product["ring_count"])
                    - max(float(value["ring_count"]) for value in precursors),
                ),
                "stereocenter_drop": max(
                    0.0,
                    float(product["stereocenter_count"])
                    - max(
                        float(value["stereocenter_count"]) for value in precursors
                    ),
                ),
                "product_bertz_complexity": product_complexity,
                "hardest_precursor_bertz_complexity": hardest_precursor,
            }
        )
    if not observations:
        return {
            "known": False,
            "complexity_drop": 0.0,
            "ring_count_drop": 0.0,
            "stereocenter_drop": 0.0,
        }
    best = max(observations, key=lambda value: value["complexity_drop"])
    return {
        "known": True,
        **{key: round(float(value), 6) for key, value in best.items()},
    }


def _molecule_metrics(value: Any) -> dict[str, float]:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return {}
    return {
        "bertz_complexity": float(Descriptors.BertzCT(molecule)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(molecule)),
        "stereocenter_count": float(
            len(
                Chem.FindMolChiralCenters(
                    molecule,
                    includeUnassigned=True,
                    includeCIP=False,
                )
            )
        ),
    }


def compile_evidence_maturity_vector(
    *,
    reaction_feasibility_rate: float,
    exact_evidence_rate: float,
    condition_completeness_rate: float,
    source_independence_met: bool,
) -> dict[str, Any]:
    score = (
        0.38 * reaction_feasibility_rate
        + 0.30 * exact_evidence_rate
        + 0.17 * condition_completeness_rate
        + 0.15 * float(source_independence_met)
    )
    return {
        "reaction_validation": round(reaction_feasibility_rate, 6),
        "exact_source_binding": round(exact_evidence_rate, 6),
        "condition_completeness": round(condition_completeness_rate, 6),
        "source_independence": float(source_independence_met),
        "score": round(score, 6),
        "basis": "host_proof_and_source_records_only",
    }


__all__ = [
    "compile_evidence_maturity_vector",
    "compile_strategic_value_vector",
]
