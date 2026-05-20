"""Route and candidate value functions used by search controllers.

This module is intentionally lightweight: it provides a stable inference
interface for learned value models, plus a calibrated deterministic fallback
that can be used before a checkpoint exists. A JSON weight file can override
the defaults without changing the planners.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem

from cascade_planner.cascadeboard import CascadeBoard
from cascade_planner.cascadeboard.route_export import route_metrics


StockChecker = Callable[[str], bool]


DEFAULT_BOARD_WEIGHTS = {
    "bias": -1.5,
    "filled_route": 1.0,
    "progressive_route": 1.4,
    "route_solved": 2.0,
    "strict_stock_solve": 1.6,
    "main_chain_reduction": 1.2,
    "leaf_reduction": 0.8,
    "naturalness": 1.0,
    "condition_success": 0.6,
    "compatibility_success": 0.7,
    "enzyme_evidence": 0.4,
    "issue_count": -0.35,
}

DEFAULT_CANDIDATE_WEIGHTS = {
    "bias": -0.4,
    "candidate_score": 0.8,
    "stock_fraction": 1.0,
    "main_reduction": 1.1,
    "has_ec": 0.25,
    "has_evidence": 0.2,
    "large_aux_penalty": -0.35,
    "self_loop": -1.5,
}


@dataclass
class ValueBreakdown:
    score: float
    probability: float
    features: dict[str, float] = field(default_factory=dict)


class RouteValueFunction:
    """Scoring interface for final routes and partial reaction candidates."""

    def __init__(
        self,
        board_weights: dict[str, float] | None = None,
        candidate_weights: dict[str, float] | None = None,
    ):
        self.board_weights = dict(DEFAULT_BOARD_WEIGHTS)
        self.candidate_weights = dict(DEFAULT_CANDIDATE_WEIGHTS)
        if board_weights:
            self.board_weights.update({k: float(v) for k, v in board_weights.items()})
        if candidate_weights:
            self.candidate_weights.update({k: float(v) for k, v in candidate_weights.items()})

    @classmethod
    def from_json(cls, path: str | Path | None) -> "RouteValueFunction":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            board_weights=data.get("board_weights") or {},
            candidate_weights=data.get("candidate_weights") or {},
        )

    def score_board(self, board: CascadeBoard, stock_checker: StockChecker | None = None) -> ValueBreakdown:
        features = board_value_features(board, stock_checker=stock_checker)
        score = linear_score(features, self.board_weights)
        return ValueBreakdown(score=score, probability=sigmoid(score), features=features)

    def score_candidate(
        self,
        product: str,
        candidate: dict[str, Any],
        stock_checker: StockChecker | None = None,
    ) -> ValueBreakdown:
        features = candidate_value_features(product, candidate, stock_checker=stock_checker)
        score = linear_score(features, self.candidate_weights)
        return ValueBreakdown(score=score, probability=sigmoid(score), features=features)


def default_value_function() -> RouteValueFunction:
    return RouteValueFunction.from_json(Path("results/shared/value_function/weights.json"))


def board_value_features(board: CascadeBoard, stock_checker: StockChecker | None = None) -> dict[str, float]:
    metrics = route_metrics(board, stock_checker=stock_checker)
    return metric_value_features(metrics)


def metric_value_features(metrics: dict[str, Any]) -> dict[str, float]:
    """Convert exported route metrics into the value-model feature vector."""
    progress = metrics.get("retrosynthesis_progress") or {}
    natural = metrics.get("route_naturalness") or {}
    compat = metrics.get("cascade_compatibility") or {}
    condition = metrics.get("condition") or {}
    enz = metrics.get("enzyme_evidence") or {}
    strict_stock = metrics.get("strict_stock_solve")
    return {
        "filled_route": as_float(metrics.get("filled_route")),
        "progressive_route": as_float(metrics.get("progressive_route")),
        "route_solved": as_float(metrics.get("route_solved")),
        "strict_stock_solve": 1.0 if strict_stock is True else -0.5 if strict_stock is False else 0.0,
        "main_chain_reduction": as_float(progress.get("main_chain_reduction")),
        "leaf_reduction": as_float(progress.get("largest_leaf_reduction")),
        "naturalness": as_float(natural.get("naturalness_score")),
        "condition_success": as_float(condition.get("condition_window_success")),
        "compatibility_success": as_float(compat.get("cascade_compatibility_success")),
        "enzyme_evidence": as_float(enz.get("enzyme_evidence_score")),
        "issue_count": float(len(compat.get("issues") or [])),
    }


def candidate_value_features(
    product: str,
    candidate: dict[str, Any],
    stock_checker: StockChecker | None = None,
) -> dict[str, float]:
    reactants = candidate_reactants(candidate)
    product_atoms = heavy_atoms(product)
    reactant_atoms = [heavy_atoms(smi) for smi in reactants]
    largest = max(reactant_atoms, default=product_atoms)
    stock_hits = 0
    for smi in reactants:
        try:
            stock_hits += int(bool(stock_checker(smi))) if stock_checker else int(heavy_atoms(smi) <= 6)
        except Exception:
            pass
    stock_fraction = stock_hits / max(len(reactants), 1)
    main_reduction = max(0.0, (product_atoms - largest) / max(product_atoms, 1)) if product_atoms else 0.0
    aux_atoms = sum(heavy_atoms(smi) for smi in candidate.get("aux_reactants") or [])
    evidence = candidate.get("evidence") or {}
    return {
        "candidate_score": as_float(candidate.get("score")),
        "stock_fraction": stock_fraction,
        "main_reduction": main_reduction,
        "has_ec": as_float(bool(candidate.get("ec") or candidate.get("enzyme_uid"))),
        "has_evidence": as_float(bool(evidence.get("doi") or evidence.get("uniprot_accession"))),
        "large_aux_penalty": aux_atoms / max(product_atoms, 1) if product_atoms else 0.0,
        "self_loop": as_float(canonical_or_raw(candidate.get("main_reactant")) == canonical_or_raw(product)),
    }


def linear_score(features: dict[str, float], weights: dict[str, float]) -> float:
    return float(weights.get("bias", 0.0)) + sum(
        float(weights.get(key, 0.0)) * float(value or 0.0)
        for key, value in features.items()
    )


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def candidate_reactants(candidate: dict[str, Any]) -> list[str]:
    values = []
    if candidate.get("main_reactant"):
        values.append(str(candidate["main_reactant"]))
    values.extend(str(smi) for smi in candidate.get("aux_reactants") or [] if smi)
    return values


def heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def canonical_or_raw(smiles: str | None) -> str:
    mol = Chem.MolFromSmiles(smiles or "")
    return Chem.MolToSmiles(mol) if mol is not None else str(smiles or "")


def as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0
