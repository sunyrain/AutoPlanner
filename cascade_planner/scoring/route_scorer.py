"""Route scoring for multi-step enzymatic cascades.

Combines step-level feasibility, pairwise compatibility (learned from
AutoPlanner's unique cascade annotations), and route-level heuristics
into a single interpretable score for ranking retrosynthetic routes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cascade_planner.data.loader_v2 import (
    CascadeRowV2,
    StepPairRowV2,
    StepRowV2,
    load_v2,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default BRENDA-like enzyme operating window
_DEFAULT_T_OPT: float = 37.0  # °C
_T_TOLERANCE: float = 10.0    # ±°C considered feasible

# Pairwise mode bonuses (one-pot > telescoped > sequential)
_MODE_BONUS: dict[str, float] = {
    "one_pot": 0.15,
    "telescoped": 0.08,
    "sequential": 0.0,
    "not_applicable": 0.0,
}

# Route length penalty: longer routes are harder to execute
_LENGTH_PENALTY_PER_STEP: float = 0.02


# ---------------------------------------------------------------------------
# StepScorer
# ---------------------------------------------------------------------------

class StepScorer:
    """Score individual synthesis steps based on feasibility signals."""

    def __init__(
        self,
        t_opt: float = _DEFAULT_T_OPT,
        t_tolerance: float = _T_TOLERANCE,
        w_reranker: float = 0.50,
        w_feasibility: float = 0.30,
        w_template: float = 0.20,
    ) -> None:
        self.t_opt = t_opt
        self.t_tolerance = t_tolerance
        self.w_reranker = w_reranker
        self.w_feasibility = w_feasibility
        self.w_template = w_template

    # -- sub-scores ----------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float, k: float = 1.0) -> float:
        return 1.0 / (1.0 + math.exp(-k * x))

    def _condition_feasibility(
        self, T_pred: float | None, pH_pred: float | None,
    ) -> float:
        """1.0 when predicted conditions sit inside the enzyme comfort zone,
        decaying smoothly outside."""
        score = 1.0
        if T_pred is not None:
            delta = abs(T_pred - self.t_opt)
            # Smooth penalty: 1.0 inside tolerance, decays outside
            if delta > self.t_tolerance:
                score *= self._sigmoid(-(delta - self.t_tolerance) / 10.0)
        if pH_pred is not None:
            # Most enzymes prefer pH 6-8; penalise extremes
            ph_dev = max(0.0, abs(pH_pred - 7.0) - 1.0)
            score *= self._sigmoid(-ph_dev / 3.0)
        return score

    @staticmethod
    def _template_frequency_score(template_freq: float | None) -> float:
        """Map raw template frequency (0-1 or count) to a 0-1 score."""
        if template_freq is None:
            return 0.5  # unknown → neutral
        # Clamp to [0, 1] if already a fraction; log-scale if count
        if template_freq > 1.0:
            return min(1.0, math.log1p(template_freq) / 10.0)
        return float(np.clip(template_freq, 0.0, 1.0))

    # -- public API ----------------------------------------------------------

    def score_step(
        self,
        rxn_smiles: str,
        ec: str | None = None,
        T_pred: float | None = None,
        pH_pred: float | None = None,
        reranker_score: float | None = None,
        template_freq: float | None = None,
    ) -> float:
        """Return a score in [0, 1] for a single step."""
        r = reranker_score if reranker_score is not None else 0.5
        f = self._condition_feasibility(T_pred, pH_pred)
        t = self._template_frequency_score(template_freq)
        return float(
            self.w_reranker * r + self.w_feasibility * f + self.w_template * t
        )


# ---------------------------------------------------------------------------
# PairScorer
# ---------------------------------------------------------------------------

@dataclass
class PairScorerWeights:
    """Learned (or default) weights for pairwise compatibility scoring."""
    intercept: float = 0.0
    w_delta_t: float = -0.05       # penalty per °C difference
    w_delta_ph: float = -0.15      # penalty per pH unit difference
    w_same_solvent: float = 0.20   # bonus for matching solvents
    w_mode_one_pot: float = 0.30
    w_mode_telescoped: float = 0.15
    w_mode_sequential: float = 0.0
    w_mode_not_applicable: float = 0.0


class PairScorer:
    """Score consecutive step pairs using condition compatibility features."""

    def __init__(self, weights: PairScorerWeights | None = None) -> None:
        self.weights = weights or PairScorerWeights()

    # -- feature extraction --------------------------------------------------

    @staticmethod
    def _extract_features(step_a: dict, step_b: dict) -> dict[str, float]:
        """Build feature dict from two step info dicts."""
        t_a = step_a.get("T") or step_a.get("t_a")
        t_b = step_b.get("T") or step_b.get("t_b")
        ph_a = step_a.get("pH") or step_a.get("ph_a")
        ph_b = step_b.get("pH") or step_b.get("ph_b")
        solv_a = step_a.get("solvent") or step_a.get("solv_a")
        solv_b = step_b.get("solvent") or step_b.get("solv_b")
        mode = step_b.get("pairwise_mode", "not_applicable")

        delta_t = abs(t_a - t_b) if (t_a is not None and t_b is not None) else 0.0
        delta_ph = abs(ph_a - ph_b) if (ph_a is not None and ph_b is not None) else 0.0
        same_solv = 1.0 if (solv_a and solv_b and solv_a == solv_b) else 0.0

        return {
            "delta_t": delta_t,
            "delta_ph": delta_ph,
            "same_solvent": same_solv,
            "mode_one_pot": 1.0 if mode == "one_pot" else 0.0,
            "mode_telescoped": 1.0 if mode == "telescoped" else 0.0,
            "mode_sequential": 1.0 if mode == "sequential" else 0.0,
            "mode_not_applicable": 1.0 if mode == "not_applicable" else 0.0,
        }

    # -- scoring -------------------------------------------------------------

    def _raw_score(self, feats: dict[str, float]) -> float:
        w = self.weights
        return (
            w.intercept
            + w.w_delta_t * feats["delta_t"]
            + w.w_delta_ph * feats["delta_ph"]
            + w.w_same_solvent * feats["same_solvent"]
            + w.w_mode_one_pot * feats["mode_one_pot"]
            + w.w_mode_telescoped * feats["mode_telescoped"]
            + w.w_mode_sequential * feats["mode_sequential"]
            + w.w_mode_not_applicable * feats["mode_not_applicable"]
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def score_pair(self, step_a_info: dict, step_b_info: dict) -> float:
        """Return a compatibility score in [0, 1] for a consecutive pair."""
        feats = self._extract_features(step_a_info, step_b_info)
        return self._sigmoid(self._raw_score(feats))


# ---------------------------------------------------------------------------
# train_pair_scorer — learn weights from AutoPlanner's pairwise annotations
# ---------------------------------------------------------------------------

def _pair_row_to_feature_vector(row: StepPairRowV2) -> np.ndarray:
    """Convert a StepPairRowV2 to a numeric feature vector."""
    delta_t = abs(row.t_a - row.t_b) if (row.t_a is not None and row.t_b is not None) else 0.0
    delta_ph = abs(row.ph_a - row.ph_b) if (row.ph_a is not None and row.ph_b is not None) else 0.0
    same_solv = 1.0 if (row.solv_a and row.solv_b and row.solv_a == row.solv_b) else 0.0
    mode = row.pairwise_mode or "not_applicable"
    return np.array([
        delta_t,
        delta_ph,
        same_solv,
        1.0 if mode == "one_pot" else 0.0,
        1.0 if mode == "telescoped" else 0.0,
        1.0 if mode == "sequential" else 0.0,
        1.0 if mode == "not_applicable" else 0.0,
    ], dtype=np.float64)


def _label_to_binary(label: str | None) -> int | None:
    """Map compatibility_label to 1 (compatible) / 0 (incompatible).
    Returns None for ambiguous or missing labels (these rows are skipped)."""
    if label is None:
        return None
    low = label.strip().lower()
    if low in ("compatible", "fully_compatible", "high", "yes"):
        return 1
    if low in ("incompatible", "not_compatible", "low", "no"):
        return 0
    # Partial / conditional — treat as positive with some noise
    if "partial" in low or "conditional" in low:
        return 1
    return None


def train_pair_scorer(pair_rows: list[StepPairRowV2]) -> PairScorer:
    """Learn pair scoring weights from AutoPlanner's compatibility annotations.

    Uses simple logistic regression (no deep learning).  Falls back to
    default weights if sklearn is unavailable or data is insufficient.
    """
    # Build labelled dataset
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for row in pair_rows:
        y = _label_to_binary(row.cascade_compatibility_label)
        if y is None:
            continue
        X_list.append(_pair_row_to_feature_vector(row))
        y_list.append(y)

    if len(X_list) < 10:
        # Not enough data — return default weights
        return PairScorer()

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float64)

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X, y)
        coef = clf.coef_[0]
        weights = PairScorerWeights(
            intercept=float(clf.intercept_[0]),
            w_delta_t=float(coef[0]),
            w_delta_ph=float(coef[1]),
            w_same_solvent=float(coef[2]),
            w_mode_one_pot=float(coef[3]),
            w_mode_telescoped=float(coef[4]),
            w_mode_sequential=float(coef[5]),
            w_mode_not_applicable=float(coef[6]),
        )
        return PairScorer(weights)

    except ImportError:
        # sklearn not available — manual gradient descent on log-loss
        # Simple but functional fallback
        lr = 0.01
        w = np.zeros(X.shape[1] + 1, dtype=np.float64)  # +1 for intercept
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        for _ in range(500):
            logits = X_bias @ w
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            grad = X_bias.T @ (preds - y) / len(y)
            w -= lr * grad

        weights = PairScorerWeights(
            intercept=float(w[0]),
            w_delta_t=float(w[1]),
            w_delta_ph=float(w[2]),
            w_same_solvent=float(w[3]),
            w_mode_one_pot=float(w[4]),
            w_mode_telescoped=float(w[5]),
            w_mode_sequential=float(w[6]),
            w_mode_not_applicable=float(w[7]),
        )
        return PairScorer(weights)


# ---------------------------------------------------------------------------
# CascadeRouteScorer
# ---------------------------------------------------------------------------

class CascadeRouteScorer:
    """Combine step-level and pair-level scores into a single route score."""

    def __init__(
        self,
        step_scorer: StepScorer,
        pair_scorer: PairScorer,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.step_scorer = step_scorer
        self.pair_scorer = pair_scorer
        w = weights or {}
        self.w_step = w.get("step", 0.40)
        self.w_pair = w.get("pair", 0.35)
        self.w_length = w.get("length", 0.15)
        self.w_in_stock = w.get("in_stock", 0.10)

    def score_route(self, route: list[dict]) -> float:
        """Score a multi-step route.

        Each element of *route* is a dict with keys:
            rxn_smiles, ec, T, pH, solvent, reranker_score
        and optionally: template_freq, pairwise_mode, in_stock.
        """
        if not route:
            return 0.0

        # Step scores
        step_scores: list[float] = []
        for step in route:
            step_scores.append(
                self.step_scorer.score_step(
                    rxn_smiles=step.get("rxn_smiles", ""),
                    ec=step.get("ec"),
                    T_pred=step.get("T"),
                    pH_pred=step.get("pH"),
                    reranker_score=step.get("reranker_score"),
                    template_freq=step.get("template_freq"),
                )
            )
        mean_step = sum(step_scores) / len(step_scores)

        # Pair scores (consecutive steps)
        pair_scores: list[float] = []
        for i in range(len(route) - 1):
            pair_scores.append(
                self.pair_scorer.score_pair(route[i], route[i + 1])
            )
        mean_pair = (sum(pair_scores) / len(pair_scores)) if pair_scores else 0.5

        # Length penalty: prefer shorter routes
        length_penalty = 1.0 - _LENGTH_PENALTY_PER_STEP * max(0, len(route) - 1)
        length_penalty = max(length_penalty, 0.0)

        # In-stock bonus: fraction of starting materials available
        in_stock_flags = [step.get("in_stock", False) for step in route]
        in_stock_frac = sum(1 for f in in_stock_flags if f) / len(route)

        return float(
            self.w_step * mean_step
            + self.w_pair * mean_pair
            + self.w_length * length_penalty
            + self.w_in_stock * in_stock_frac
        )

    def rank_routes(
        self, routes: list[list[dict]],
    ) -> list[tuple[int, float]]:
        """Rank candidate routes by descending score.

        Returns list of (original_index, score) tuples, best first.
        """
        scored = [(i, self.score_route(r)) for i, r in enumerate(routes)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _evaluate_on_cascades(
    pair_rows: list[StepPairRowV2],
    cascade_rows: list[CascadeRowV2],
) -> None:
    """Train a PairScorer and report cross-validated accuracy on known cascades."""
    # Train
    scorer = train_pair_scorer(pair_rows)
    w = scorer.weights
    print("Learned PairScorer weights:")
    print(f"  intercept        = {w.intercept:+.4f}")
    print(f"  w_delta_t        = {w.w_delta_t:+.4f}")
    print(f"  w_delta_ph       = {w.w_delta_ph:+.4f}")
    print(f"  w_same_solvent   = {w.w_same_solvent:+.4f}")
    print(f"  w_mode_one_pot   = {w.w_mode_one_pot:+.4f}")
    print(f"  w_mode_telescoped= {w.w_mode_telescoped:+.4f}")
    print(f"  w_mode_sequential= {w.w_mode_sequential:+.4f}")
    print()

    # Evaluate on pair rows with labels
    correct, total = 0, 0
    for row in pair_rows:
        y = _label_to_binary(row.cascade_compatibility_label)
        if y is None:
            continue
        step_a = {"T": row.t_a, "pH": row.ph_a, "solvent": row.solv_a}
        step_b = {
            "T": row.t_b, "pH": row.ph_b, "solvent": row.solv_b,
            "pairwise_mode": row.pairwise_mode,
        }
        pred = scorer.score_pair(step_a, step_b)
        if (pred >= 0.5) == bool(y):
            correct += 1
        total += 1

    if total:
        print(f"Pair compatibility accuracy (train): {correct}/{total} = {correct / total:.3f}")
    else:
        print("No labelled pairs found for evaluation.")

    # Cascade-level stats
    labelled = [c for c in cascade_rows if c.compatibility_label]
    print(f"\nCascades with compatibility label: {len(labelled)}/{len(cascade_rows)}")
    if labelled:
        from collections import Counter
        dist = Counter(c.compatibility_label for c in labelled)
        for k, v in dist.most_common():
            print(f"  {k}: {v}")


def _score_route_file(path: str) -> None:
    """Load a route JSON and score it."""
    route = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(route, dict):
        route = route.get("steps", route.get("route", [route]))
    if not isinstance(route, list):
        print(f"Error: expected a list of steps, got {type(route).__name__}", file=sys.stderr)
        sys.exit(1)

    step_scorer = StepScorer()
    pair_scorer = PairScorer()
    scorer = CascadeRouteScorer(step_scorer, pair_scorer)
    score = scorer.score_route(route)
    print(f"Route score: {score:.4f}  ({len(route)} steps)")

    # Per-step breakdown
    for i, step in enumerate(route):
        ss = step_scorer.score_step(
            rxn_smiles=step.get("rxn_smiles", ""),
            ec=step.get("ec"),
            T_pred=step.get("T"),
            pH_pred=step.get("pH"),
            reranker_score=step.get("reranker_score"),
            template_freq=step.get("template_freq"),
        )
        print(f"  step {i}: score={ss:.3f}  rxn={step.get('rxn_smiles', '?')[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score multi-step enzymatic cascade routes.",
    )
    parser.add_argument(
        "--data", type=str, default="cascade_dataset_v2.normalized.json",
        help="Path to normalized cascade dataset JSON.",
    )
    parser.add_argument(
        "--eval", action="store_true", dest="do_eval",
        help="Evaluate pair scorer on known cascades.",
    )
    parser.add_argument(
        "--route", type=str, default=None,
        help="Path to a route JSON file to score.",
    )
    args = parser.parse_args()

    if args.route:
        _score_route_file(args.route)
        return

    if args.do_eval:
        step_rows, pair_rows, cascade_rows = load_v2(args.data)
        print(f"Loaded {len(step_rows)} steps, {len(pair_rows)} pairs, "
              f"{len(cascade_rows)} cascades.\n")
        _evaluate_on_cascades(pair_rows, cascade_rows)
        return

    # Default: print summary
    parser.print_help()


if __name__ == "__main__":
    main()
