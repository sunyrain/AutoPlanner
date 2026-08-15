"""Evidence-independent strategy value and independent maturity projections."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


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
    key_bonds = len(
        strategy_card.get("key_bond_signature")
        or strategy_card.get("key_bond_changes")
        or []
    )
    key_bond_leverage = min(
        1.0,
        (key_bonds + max(0, root_fragment_count - 1)) / 3.0,
    )
    topology_text = " ".join(
        [
            str(strategy_card.get("skeleton_change_class") or ""),
            str(strategy_card.get("key_forward_transformation") or ""),
        ]
    ).lower()
    topology_transform = (
        1.0
        if any(
            token in topology_text
            for token in (
                "cascade",
                "annulation",
                "cycloaddition",
                "rearrangement",
                "ring",
            )
        )
        else 0.55
        if any(token in topology_text for token in ("coupling", "fragment", "union"))
        else 0.2
    )
    complexity_drop = {
        "low": 0.25,
        "medium": 0.65,
        "high": 1.0,
    }.get(str(strategy_card.get("expected_complexity_drop") or "").lower(), 0.0)
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
        "stereochemical_leverage": round(stereochemical_leverage, 6),
        "convergence": round(root_convergence, 6),
        "protection_efficiency": round(protection_efficiency, 6),
        "score": round(score, 6),
        "basis": "strategy_card_and_canonical_topology_only",
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
