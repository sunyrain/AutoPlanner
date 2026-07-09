"""Joblib-backed learned cascade verifier utilities.

The learned verifier is trained from rule-derived perturbation packs. Runtime
use defaults to annotation-only: it emits feasibility probabilities and failure
reason evidence without changing route ranking unless a caller explicitly uses
those scores as a value/rerank signal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_learned_verifier(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    import joblib

    model_path = Path(path)
    artifact = joblib.load(model_path)
    return {
        "path": str(model_path),
        "vectorizer": artifact["vectorizer"],
        "feasible_model": artifact["feasible_model"],
        "reason_models": dict(artifact.get("reason_models") or {}),
        "summary": dict(artifact.get("summary") or {}),
        "recommended_feasible_threshold": artifact.get("recommended_feasible_threshold")
        or (((artifact.get("summary") or {}).get("feasibility") or {}).get("threshold_calibration") or {}).get("recommended_threshold"),
    }


def predict_learned_verifier(
    learned: dict[str, Any],
    cascade: dict[str, Any],
    *,
    target_smiles: str,
) -> dict[str, Any]:
    from cascade_planner.cascade_search.value import _positive_probability
    from cascade_planner.cascade_verifier.features import cascade_verifier_features

    example = {
        "target_smiles": target_smiles,
        "cascade": cascade,
        "expected_failure_reasons": [],
    }
    x = learned["vectorizer"].transform([cascade_verifier_features(example)])
    feasible_probability = _positive_probability(learned["feasible_model"], x)
    threshold = learned.get("recommended_feasible_threshold")
    try:
        threshold_value = float(threshold) if threshold not in (None, "") else None
    except (TypeError, ValueError):
        threshold_value = None
    reason_probabilities = {
        reason: _positive_probability(model, x)
        for reason, model in learned["reason_models"].items()
    }
    return {
        "model_path": learned["path"],
        "feasible_probability": round(float(feasible_probability), 6),
        "recommended_feasible_threshold": round(float(threshold_value), 6) if threshold_value is not None else None,
        "conservative_feasible": bool(feasible_probability >= threshold_value) if threshold_value is not None else None,
        "reason_probabilities": {
            key: round(float(value), 6)
            for key, value in sorted(reason_probabilities.items())
        },
    }
