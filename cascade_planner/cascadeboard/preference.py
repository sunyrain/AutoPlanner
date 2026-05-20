"""Uncertainty, preference learning, and Pareto ranking for CascadeBoard++.

Implements:
- Risk/uncertainty estimation per slot and per route
- Bradley-Terry preference scoring
- Pareto front extraction
- Objective-conditioned ranking
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cascade_planner.cascadeboard import CascadeBoard, Slot, RouteResult


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def compute_slot_uncertainty(slot: Slot) -> dict[str, float]:
    """Estimate uncertainty for a single slot."""
    u: dict[str, float] = {}

    # Reaction uncertainty: low retro score = high uncertainty
    u["reaction"] = 1.0 - min(slot.e_retro or 0.0, 1.0)

    # Enzyme uncertainty: no EC or low enzyme score
    if slot.is_enzymatic():
        u["enzyme"] = 1.0 - min(slot.e_enzyme or 0.0, 1.0)
    else:
        u["enzyme"] = 0.0

    # Condition uncertainty: missing T or pH
    missing = 0
    if slot.T is None:
        missing += 1
    if slot.pH is None:
        missing += 1
    u["condition"] = missing / 2.0

    # Source uncertainty: mock > enzexpand > retrochimera
    source_conf = {"retrochimera": 0.1, "enzexpand": 0.3, "mock": 0.7, "": 0.5}
    u["source"] = source_conf.get(slot.source, 0.5)

    # Candidate pool: no candidates = high uncertainty
    u["candidate_pool"] = 0.0 if slot.candidates else 0.5

    return u


def compute_route_uncertainty(board: CascadeBoard) -> dict[str, float]:
    """Aggregate slot uncertainties into route-level risk vector."""
    if not board.slots:
        return {"overall": 1.0}

    slot_us = [compute_slot_uncertainty(s) for s in board.slots]

    risk: dict[str, float] = {}
    for key in ["reaction", "enzyme", "condition", "source", "candidate_pool"]:
        vals = [u.get(key, 0) for u in slot_us]
        risk[key] = float(np.mean(vals))

    # Overall: weighted combination
    risk["overall"] = (
        0.3 * risk["reaction"]
        + 0.25 * risk["enzyme"]
        + 0.2 * risk["condition"]
        + 0.15 * risk["source"]
        + 0.1 * risk["candidate_pool"]
    )

    return risk


def compute_confidence(board: CascadeBoard) -> float:
    """Single confidence score (1 - overall_risk)."""
    risk = compute_route_uncertainty(board)
    return max(0.0, 1.0 - risk["overall"])


# ---------------------------------------------------------------------------
# Preference scoring (Bradley-Terry style)
# ---------------------------------------------------------------------------

# Default preference weights for different objectives
OBJECTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {
        "feasibility": 3.0,
        "condition_compatibility": 2.0,
        "stock_accessibility": 2.0,
        "step_count": -1.0,
        "enzyme_availability": 1.0,
        "one_pot_feasibility": 1.0,
    },
    "industrial": {
        "feasibility": 3.0,
        "stock_accessibility": 3.0,
        "step_count": -3.0,
        "condition_compatibility": 1.0,
        "enzyme_availability": 0.5,
        "one_pot_feasibility": 0.5,
    },
    "green": {
        "one_pot_feasibility": 4.0,
        "condition_compatibility": 3.0,
        "feasibility": 2.0,
        "step_count": -1.0,
        "stock_accessibility": 1.0,
        "enzyme_availability": 1.0,
    },
    "novelty": {
        "feasibility": 2.0,
        "enzyme_availability": 2.0,
        "condition_compatibility": 1.0,
        "step_count": -0.5,
        "stock_accessibility": 1.0,
        "one_pot_feasibility": 0.5,
    },
}


def preference_score(
    quality_vector: dict[str, float],
    risk_vector: dict[str, float],
    objective: str = "balanced",
    risk_penalty: float = 0.5,
) -> float:
    """Compute preference score: quality - risk_penalty * risk."""
    weights = OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS["balanced"])

    q_score = sum(
        weights.get(k, 0) * quality_vector.get(k, 0)
        for k in weights
    )

    r_score = risk_vector.get("overall", 0) * risk_penalty

    return q_score - r_score


# ---------------------------------------------------------------------------
# Pareto front extraction
# ---------------------------------------------------------------------------

def dominates(a: dict[str, float], b: dict[str, float], keys: list[str]) -> bool:
    """Check if route a Pareto-dominates route b on given quality keys."""
    dominated = False
    for k in keys:
        va = a.get(k, 0)
        vb = b.get(k, 0)
        if va < vb:
            return False
        if va > vb:
            dominated = True
    return dominated


def extract_pareto_front(
    results: list[RouteResult],
    quality_keys: list[str] | None = None,
) -> list[RouteResult]:
    """Extract Pareto-optimal routes from a list of results."""
    if not results:
        return []

    if quality_keys is None:
        quality_keys = [
            "feasibility", "condition_compatibility",
            "stock_accessibility", "one_pot_feasibility",
        ]

    pareto: list[RouteResult] = []
    for r in results:
        is_dominated = False
        for p in pareto:
            if dominates(p.quality_vector, r.quality_vector, quality_keys):
                is_dominated = True
                break
        if not is_dominated:
            # Remove any existing Pareto members dominated by r
            pareto = [
                p for p in pareto
                if not dominates(r.quality_vector, p.quality_vector, quality_keys)
            ]
            pareto.append(r)

    return pareto


# ---------------------------------------------------------------------------
# Objective-conditioned ranking
# ---------------------------------------------------------------------------

def rank_routes(
    results: list[RouteResult],
    objective: str = "balanced",
    risk_penalty: float = 0.5,
) -> list[RouteResult]:
    """Rank routes by preference score for a given objective."""
    for r in results:
        if not r.risk_vector:
            r.risk_vector = compute_route_uncertainty(r.board)
        r.score = preference_score(
            r.quality_vector, r.risk_vector, objective, risk_penalty
        )
        r.confidence = compute_confidence(r.board)

    return sorted(results, key=lambda r: r.score, reverse=True)


# ---------------------------------------------------------------------------
# Multi-objective output
# ---------------------------------------------------------------------------

def multi_objective_output(
    results: list[RouteResult],
    objectives: list[str] | None = None,
) -> dict[str, list[RouteResult]]:
    """Rank routes under multiple objectives, return best per objective + Pareto front."""
    if objectives is None:
        objectives = ["balanced", "industrial", "green", "novelty"]

    output: dict[str, list[RouteResult]] = {}

    for obj in objectives:
        ranked = rank_routes(list(results), objective=obj)
        output[f"best_{obj}"] = ranked[:3]

    output["pareto_front"] = extract_pareto_front(results)

    return output
