"""Feature bridge from RouteTreeState to existing vNext policy tensors."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from rdkit import Chem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.vnext.features import (
    candidate_feature_dim,
    candidate_feature_vector,
    node_feature_dim,
    open_leaf_feature_matrix,
    route_feature_dim,
    route_feature_vector,
    route_step_tokens,
)


StockChecker = Callable[[str], bool]


def state_route_row(
    state: RouteTreeState,
    *,
    stock_checker: StockChecker | None = None,
) -> dict[str, Any]:
    target_atoms = _heavy_atoms(state.target)
    open_atoms = [_heavy_atoms(smi) for smi in state.open_leaves]
    largest_open = max(open_atoms, default=target_atoms)
    main_reduction = max(0.0, (target_atoms - largest_open) / max(target_atoms, 1)) if target_atoms else 0.0
    stock_closed = None
    if stock_checker is not None and state.open_leaves:
        try:
            stock_closed = all(bool(stock_checker(smi)) for smi in state.open_leaves if smi)
        except Exception:
            stock_closed = None
    type_sequence = [step.action.reaction_type for step in state.steps]
    ec1_sequence = [_ec1(step.action.ec) for step in state.steps]
    source_sequence = [step.action.source for step in state.steps]
    features = {
        "filled_route": float(bool(state.steps)),
        "progressive_route": float(main_reduction > 0.05),
        "route_solved": float(bool(stock_closed)),
        "strict_stock_solve": 1.0 if stock_closed is True else -0.5 if stock_closed is False else 0.0,
        "main_chain_reduction": main_reduction,
        "leaf_reduction": main_reduction,
        "naturalness": 1.0,
        "condition_success": 0.0,
        "compatibility_success": 0.0,
        "enzyme_evidence": float(any(step.action.ec for step in state.steps)),
        "issue_count": 0.0,
    }
    return {
        "target_smiles": state.target,
        "n_steps": len(state.steps),
        "depth": state.depth,
        "type_sequence": type_sequence,
        "ec1_sequence": ec1_sequence,
        "source_sequence": source_sequence,
        "features": features,
        "metrics_summary": {},
        "score": state.score,
        "confidence": 0.0,
        "operation_mode": "unknown",
    }


def state_tensors(
    state: RouteTreeState,
    *,
    max_steps: int,
    stock_checker: StockChecker | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = state_route_row(state, stock_checker=stock_checker)
    step_tokens, step_mask = route_step_tokens(row, max_steps=max_steps)
    route_features = route_feature_vector(row)
    return step_tokens, step_mask, route_features


def action_feature_matrix(
    product: str,
    actions: list[CandidateAction],
    *,
    n_bits: int,
    max_candidates: int,
    stock_checker: StockChecker | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feat_dim = candidate_feature_dim(n_bits)
    features = np.zeros((max_candidates, feat_dim), dtype=np.float32)
    mask = np.zeros(max_candidates, dtype=np.float32)
    for idx, action in enumerate(actions[:max_candidates]):
        features[idx] = candidate_feature_vector(
            product,
            action.to_candidate_dict(),
            rank=action.rank or idx + 1,
            n_bits=n_bits,
            stock_checker=stock_checker,
        )
        mask[idx] = 1.0
    return features, mask


def node_feature_matrix(
    state: RouteTreeState,
    leaves: list[str],
    *,
    n_bits: int,
    max_open_leaves: int,
    stock_checker: StockChecker | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return open_leaf_feature_matrix(
        target=state.target,
        open_leaves=leaves,
        depth=state.depth,
        expanded=set(state.expanded),
        parent_reactants=_state_parent_reactants(state),
        max_open_leaves=max_open_leaves,
        n_bits=n_bits,
        stock_checker=stock_checker,
    )


def expected_route_feature_dim() -> int:
    return route_feature_dim()


def expected_node_feature_dim(n_bits: int = 256) -> int:
    return node_feature_dim(n_bits)


def _ec1(value: str | None) -> str:
    return str(value or "").split(".", 1)[0] if value else ""


def _state_parent_reactants(state: RouteTreeState) -> set[str]:
    out: set[str] = set()
    for step in state.steps:
        for smi in step.action.reactants:
            can = canonical_smiles(smi) or smi
            if can:
                out.add(can)
    return out


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0
