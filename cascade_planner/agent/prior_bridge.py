"""Bridge structured agent priors into CascadeBoard search.

This module keeps LLM output as a soft ranking signal. It never creates
reaction candidates, stock facts, enzyme availability facts, or condition facts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cascade_planner.agent.prior_generator import generate_strategic_prior


NO_PRIOR_PROVIDERS = {"", "none", "no_agent", "no-agent", None}


def load_prior_cache(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_prior_cache(path: str | None, cache: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2))


def prior_cache_key(provider: str, target_smiles: str) -> str:
    return f"{provider}::{target_smiles}"


def get_prior_for_target(
    target_smiles: str,
    *,
    provider: str = "none",
    cache: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a structured prior, optionally cached by provider and target."""
    if provider in NO_PRIOR_PROVIDERS:
        return None
    cache = cache if cache is not None else {}
    key = prior_cache_key(provider, target_smiles)
    if key in cache:
        return cache[key]
    prior = generate_strategic_prior(target_smiles, provider=provider)
    cache[key] = prior
    return prior


def summarize_prior(prior: dict[str, Any] | None, requested_provider: str = "none") -> dict[str, Any] | None:
    if prior is None:
        return None
    return {
        "requested_provider": requested_provider,
        "source": prior.get("source", ""),
        "route_modes": [
            {"mode": x.get("mode"), "weight": x.get("weight")}
            for x in prior.get("route_mode_priors", [])[:5]
        ],
        "reaction_type_priors": [
            {"slot": x.get("slot"), "reaction_type": x.get("reaction_type"), "weight": x.get("weight")}
            for x in prior.get("reaction_type_priors", [])[:8]
        ],
        "enzyme_priors": [
            {"ec1": x.get("ec1"), "weight": x.get("weight")}
            for x in prior.get("enzyme_priors", [])[:8]
        ],
        "condition_risks": [
            {"kind": x.get("kind"), "severity": x.get("severity")}
            for x in prior.get("condition_risks", [])[:8]
        ],
        "unsupported_claims": list(prior.get("unsupported_claims", [])),
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_skeleton_with_prior(skeleton: Any, prior: dict[str, Any] | None) -> float:
    """Score a generated skeleton against structured strategic priors.

    The score is normalized to roughly [0, 1]. It is only used for reranking
    already generated skeletons.
    """
    if not prior:
        return 0.0

    types = list(getattr(skeleton, "types", []) or [])
    ec1s = [int(x) if str(x).isdigit() else 0 for x in (getattr(skeleton, "ec1s", []) or [])]
    if not types:
        return 0.0

    raw = 0.0
    weight_sum = 0.0

    for item in prior.get("route_mode_priors", []) or []:
        mode = item.get("mode", "")
        weight = _as_float(item.get("weight"), 0.0)
        if weight <= 0:
            continue
        weight_sum += weight
        enzymatic_fraction = sum(1 for x in ec1s if x > 0) / max(len(ec1s), 1)
        if mode == "organic_only":
            raw += weight * (1.0 - enzymatic_fraction)
        elif mode == "enzymatic_only":
            raw += weight * enzymatic_fraction
        elif mode == "chemoenzymatic_cascade":
            raw += weight * (1.0 if any(x > 0 for x in ec1s) and any(x == 0 for x in ec1s) else 0.35)
        elif mode == "enzymatic_late_stage":
            raw += weight * (1.0 if ec1s and ec1s[0] > 0 else 0.0)
        elif mode == "unknown":
            raw += 0.25 * weight

    for item in prior.get("reaction_type_priors", []) or []:
        rtype = item.get("reaction_type", "")
        weight = _as_float(item.get("weight"), 0.0)
        if not rtype or weight <= 0:
            continue
        weight_sum += weight
        slot = item.get("slot")
        if slot is None:
            raw += weight * (1.0 if rtype in types else 0.0)
        else:
            try:
                idx = int(slot)
            except (TypeError, ValueError):
                idx = -1
            raw += weight * (1.0 if 0 <= idx < len(types) and types[idx] == rtype else 0.0)

    for item in prior.get("enzyme_priors", []) or []:
        ec1 = item.get("ec1")
        weight = _as_float(item.get("weight"), 0.0)
        if ec1 is None or weight <= 0:
            continue
        try:
            ec1_int = int(ec1)
        except (TypeError, ValueError):
            continue
        weight_sum += weight
        raw += weight * (1.0 if ec1_int in ec1s else 0.0)

    if weight_sum <= 0:
        return 0.0
    return max(0.0, min(1.0, raw / weight_sum))


def rank_skeletons_with_prior(
    skeletons: list[Any],
    prior: dict[str, Any] | None,
    *,
    prior_weight: float = 1.0,
) -> list[Any]:
    """Sort skeletons by model log-probability plus optional prior score."""
    if not prior:
        return sorted(skeletons, key=lambda x: getattr(x, "log_prob", 0.0), reverse=True)

    def key_fn(skel: Any) -> float:
        return float(getattr(skel, "log_prob", 0.0)) + float(prior_weight) * score_skeleton_with_prior(skel, prior)

    return sorted(skeletons, key=key_fn, reverse=True)
