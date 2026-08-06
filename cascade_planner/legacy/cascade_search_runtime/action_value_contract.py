"""Stable checkpoint contract for cascade action-value models."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


COMPONENT_FIELDS = [
    "domain_preference",
    "model_preference",
    "condition",
    "cofactor",
    "enzyme_evidence",
    "stage_transition",
    "failure_match",
    "source_value",
]


class CascadeActionValueNetwork:
    """Create the network shape serialized by historical action-value trainers."""

    def __new__(cls, input_dim: int, hidden: int = 192):
        import torch.nn as nn

        class _Network(nn.Module):
            def __init__(self, input_dim: int, hidden: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.12),
                    nn.Linear(hidden, max(32, hidden // 2)),
                    nn.GELU(),
                    nn.Linear(max(32, hidden // 2), 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _Network(input_dim, hidden)


def action_value_feature_vector(
    row: dict[str, Any],
    schema: dict[str, Any],
) -> list[float]:
    """Encode one runtime or training row using a checkpoint feature schema."""
    out: list[float] = []
    for field in schema.get("categorical_fields") or []:
        value = _feature_field(row, field)
        categories = (schema.get("categories") or {}).get(field) or []
        out.extend(1.0 if value == category else 0.0 for category in categories)
    parent = str(row.get("parent_mol") or "")
    reactants = [str(item) for item in row.get("reactants") or [] if item]
    n_bits = int(schema.get("n_bits") or 128)
    out.extend(_fp(parent, n_bits=n_bits).tolist())
    out.extend(_fp(".".join(reactants), n_bits=n_bits).tolist())
    context_features = dict(row.get("context_features") or {})
    if row.get("active_failure_modes") and not context_features.get(
        "active_failure_modes"
    ):
        context_features["active_failure_modes"] = row.get("active_failure_modes")
    numeric = _numeric_features(
        parent_mol=parent,
        reactants=reactants,
        candidate_index=row.get("candidate_index"),
        parent_depth=row.get("parent_depth"),
        base_score=row.get("base_score"),
        base_cost=row.get("base_cost"),
        cascade_adjustment=row.get("cascade_adjustment"),
        rule_total_cost=row.get("total_cost"),
        components=row.get("components") or {},
        source_model=row.get("source_model"),
        reaction_domain=row.get("reaction_domain"),
        route_domain=row.get("route_domain") or context_features.get("route_domain"),
        context_features=context_features,
        source_policy_decision=row.get("source_policy_decision") or {},
    )
    out.extend(
        float(numeric.get(name, 0.0))
        for name in schema.get("numeric_fields") or []
    )
    return out


def _numeric_features(
    *,
    parent_mol: str,
    reactants: list[str],
    candidate_index: Any,
    parent_depth: Any,
    base_score: Any,
    base_cost: Any,
    cascade_adjustment: Any,
    rule_total_cost: Any,
    components: dict[str, Any],
    source_model: Any = None,
    reaction_domain: Any = None,
    route_domain: Any = None,
    context_features: dict[str, Any] | None = None,
    source_policy_decision: dict[str, Any] | None = None,
) -> dict[str, float]:
    parent_desc = _mol_desc(parent_mol)
    reactant_descs = [_mol_desc(smiles) for smiles in reactants]
    reactant_total_heavy = sum(item["heavy_atoms_raw"] for item in reactant_descs)
    parent_heavy = parent_desc["heavy_atoms_raw"]
    values = {
        "parent_depth": _scaled(parent_depth, 8.0),
        "candidate_index": _scaled(candidate_index, 100.0),
        "base_score": _bounded(base_score, 0.0, 1.0),
        "base_cost": _scaled(base_cost, 12.0),
        "cascade_adjustment": _signed_scaled(cascade_adjustment, 6.0),
        "rule_total_cost": _scaled(rule_total_cost, 12.0),
        "reactant_count": min(1.0, len(reactants) / 6.0),
        "parent_heavy_atoms": min(1.0, parent_desc["heavy_atoms_raw"] / 80.0),
        "parent_hetero_atoms": min(1.0, parent_desc["hetero_atoms_raw"] / 30.0),
        "parent_ring_count": min(1.0, parent_desc["ring_count_raw"] / 10.0),
        "parent_mol_wt": min(1.0, parent_desc["mol_wt_raw"] / 1000.0),
        "reactant_total_heavy_atoms": min(1.0, reactant_total_heavy / 160.0),
        "reactant_total_hetero_atoms": min(
            1.0,
            sum(item["hetero_atoms_raw"] for item in reactant_descs) / 60.0,
        ),
        "reactant_total_ring_count": min(
            1.0,
            sum(item["ring_count_raw"] for item in reactant_descs) / 20.0,
        ),
        "reactant_total_mol_wt": min(
            1.0,
            sum(item["mol_wt_raw"] for item in reactant_descs) / 2000.0,
        ),
        "reactant_max_mol_wt": min(
            1.0,
            max([item["mol_wt_raw"] for item in reactant_descs] or [0.0]) / 1000.0,
        ),
        "heavy_atom_balance": min(
            2.0,
            reactant_total_heavy / max(parent_heavy, 1.0),
        )
        / 2.0,
    }
    values.update(
        _process_context_features(
            source_model=source_model,
            reaction_domain=reaction_domain,
            route_domain=route_domain,
            parent_depth=parent_depth,
            context_features=context_features or {},
            source_policy_decision=source_policy_decision or {},
        )
    )
    for name in COMPONENT_FIELDS:
        values[f"component_{name}"] = _signed_scaled(components.get(name), 6.0)
    return values


def _feature_field(row: dict[str, Any], field: str) -> str:
    if field == "adjacent_reaction_domain":
        return str((row.get("context_features") or {}).get(field) or "unknown")
    if field == "route_domain":
        return str(
            row.get("route_domain")
            or (row.get("context_features") or {}).get(field)
            or "unknown"
        )
    return str(row.get(field) or "unknown")


def _process_context_features(
    *,
    source_model: Any,
    reaction_domain: Any,
    route_domain: Any,
    parent_depth: Any,
    context_features: dict[str, Any],
    source_policy_decision: dict[str, Any],
) -> dict[str, float]:
    context = context_features if isinstance(context_features, dict) else {}
    policy = source_policy_decision if isinstance(source_policy_decision, dict) else {}
    source = str(source_model or "")
    reaction = _normalize(reaction_domain)
    route = _normalize(route_domain or context.get("route_domain"))
    adjacent = _normalize(context.get("adjacent_reaction_domain"))
    preferred_domains = _tokens(context.get("preferred_reaction_domains"))
    discouraged_domains = _tokens(context.get("discouraged_reaction_domains"))
    active_failures = _tokens(context.get("active_failure_modes"))
    selected_models = {
        str(item) for item in (policy.get("select_models") or []) if item
    }
    topk_by_model = (
        policy.get("topk_by_model")
        if isinstance(policy.get("topk_by_model"), dict)
        else {}
    )
    source_scores = (
        policy.get("source_value_scores")
        if isinstance(policy.get("source_value_scores"), dict)
        else {}
    )
    topk_value = _float_or_none(topk_by_model.get(source)) if topk_by_model else None
    topk_values = [
        float(value)
        for value in (_float_or_none(value) for value in topk_by_model.values())
        if value is not None and value > 0.0
    ]
    max_topk = max(topk_values) if topk_values else None
    score_value = _float_or_none(source_scores.get(source)) if source_scores else None
    score_values = [
        float(value)
        for value in (_float_or_none(value) for value in source_scores.values())
        if value is not None
    ]
    max_score = max(score_values) if score_values else None
    min_score = min(score_values) if score_values else None
    has_adjacent = adjacent not in {"", "unknown", "none", "null"}
    known_reaction = reaction not in {"", "unknown", "none", "null"}
    return {
        "context_node_depth": _scaled(context.get("node_depth", parent_depth), 8.0),
        "route_domain_alignment": _route_domain_alignment(route, reaction),
        "preferred_domain_match": float(
            bool(preferred_domains and reaction in preferred_domains)
        ),
        "unpreferred_domain_with_preference": float(
            bool(preferred_domains and reaction not in preferred_domains)
        ),
        "discouraged_domain_match": float(
            bool(discouraged_domains and reaction in discouraged_domains)
        ),
        "has_adjacent_step": float(has_adjacent),
        "adjacent_domain_match": float(
            bool(has_adjacent and known_reaction and adjacent == reaction)
        ),
        "adjacent_domain_switch": float(
            bool(has_adjacent and known_reaction and adjacent != reaction)
        ),
        "enzymatic_after_chemical_context": float(
            adjacent == "chemical" and reaction == "enzymatic"
        ),
        "chemical_after_enzymatic_context": float(
            adjacent == "enzymatic" and reaction == "chemical"
        ),
        "active_failure_count": min(1.0, len(active_failures) / 5.0),
        "failure_candidate_missing": float("candidatemissing" in active_failures),
        "failure_stock_dead_end": float("stockdeadend" in active_failures),
        "failure_condition_conflict": float("conditionconflict" in active_failures),
        "failure_cofactor_debt": float("cofactordebt" in active_failures),
        "failure_enzyme_evidence_weak": float(
            "enzymeevidenceweak" in active_failures
        ),
        "failure_stage_over_complex": float("stageovercomplex" in active_failures),
        "failure_route_order_mismatch": float(
            "routeordermismatch" in active_failures
        ),
        "failure_low_plausibility": float("lowplausibility" in active_failures),
        "source_policy_has_decision": float(bool(policy)),
        "source_policy_selected": float(
            bool(selected_models and source in selected_models)
        ),
        "source_policy_topk_fraction": (
            max(0.0, min(1.0, float(topk_value) / float(max_topk)))
            if topk_value is not None and max_topk
            else 0.0
        ),
        "source_policy_top_budget": float(
            bool(topk_value is not None and max_topk and topk_value >= max_topk)
        ),
        "source_policy_score": (
            _bounded(score_value, 0.0, 1.0) if score_value is not None else 0.0
        ),
        "source_policy_score_ratio": (
            max(0.0, min(1.0, float(score_value) / float(max_score)))
            if score_value is not None and max_score and max_score > 0.0
            else 0.0
        ),
        "source_policy_best_source": float(
            bool(
                score_value is not None
                and max_score is not None
                and score_value >= max_score
            )
        ),
        "source_policy_worst_source": float(
            bool(
                score_value is not None
                and min_score is not None
                and max_score is not None
                and max_score > min_score
                and score_value <= min_score
            )
        ),
    }


def _route_domain_alignment(route_domain: str, reaction_domain: str) -> float:
    if not route_domain or not reaction_domain or reaction_domain == "unknown":
        return 0.0
    if route_domain == "all_chemical":
        return float(reaction_domain == "chemical")
    if route_domain in {"all_enzymatic", "whole_cell_biocatalytic"}:
        return float(reaction_domain == "enzymatic")
    if route_domain in {"chemoenzymatic", "hybrid_mimetic"}:
        return float(reaction_domain in {"chemical", "enzymatic"})
    return 0.0


def _fp(smiles: str, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _mol_desc(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {
            "heavy_atoms_raw": 0.0,
            "hetero_atoms_raw": 0.0,
            "ring_count_raw": 0.0,
            "mol_wt_raw": 0.0,
        }
    return {
        "heavy_atoms_raw": float(mol.GetNumHeavyAtoms()),
        "hetero_atoms_raw": float(
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
        ),
        "ring_count_raw": float(mol.GetRingInfo().NumRings()),
        "mol_wt_raw": float(Descriptors.MolWt(mol)),
    }


def _bounded(value: Any, low: float, high: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(low, min(high, out))


def _scaled(value: Any, scale: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out / scale))


def _signed_scaled(value: Any, scale: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(-1.0, min(1.0, out / scale))


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [str(value)]
    return {_normalize(item) for item in values if _normalize(item)}


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


__all__ = [
    "COMPONENT_FIELDS",
    "CascadeActionValueNetwork",
    "action_value_feature_vector",
]
