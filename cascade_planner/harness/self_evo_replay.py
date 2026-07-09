"""Harness-level selfEVO replay and promotion gate."""
from __future__ import annotations

from typing import Any

from cascade_planner.agent.evolution_manager import (
    EvolutionCandidate,
    LayeredKnowledgeBase,
    evaluate_benchmark_gate,
    evolution_candidate_from_dict,
    validate_evolution_candidate,
)


SELF_EVO_REPLAY_REPORT_SCHEMA = "harness_self_evo_replay_report.v1"


def run_self_evo_replay_gate(
    staging_kb_or_report: dict[str, Any],
    *,
    replay_metrics: dict[str, Any] | None = None,
    target_run: bool = True,
    allow_production: bool = False,
) -> dict[str, Any]:
    """Replay selfEVO candidates and gate promotion.

    Target runs are allowed to validate candidate/shadow/staging writes only.
    Production promotion requires a non-target replay with allow_production=True
    and an accepted benchmark gate.
    """
    kb = _kb_from_report(staging_kb_or_report)
    staging = dict((kb.to_dict().get("layers") or {}).get("staging") or {})
    metrics = dict(replay_metrics or {})
    gate = evaluate_benchmark_gate(metrics)
    candidates: list[dict[str, Any]] = []
    production_promoted_count = 0
    reasons: list[str] = []
    if not staging:
        reasons.append("no_staging_candidates")
    if target_run:
        reasons.append("target_run_production_blocked")
    if not allow_production:
        reasons.append("production_not_requested")
    if not gate.accepted:
        reasons.extend(gate.reasons)

    for candidate_id, payload in staging.items():
        candidate = evolution_candidate_from_dict(payload)
        validation = validate_evolution_candidate(candidate)
        row = {
            "candidate_id": candidate_id,
            "candidate_validation": validation,
            "staging_present": True,
            "production_promoted": False,
            "production_blocked": True,
            "promotion_reasons": [],
        }
        if not validation.get("accepted"):
            row["promotion_reasons"].extend(validation.get("reasons") or [])
        elif allow_production and not target_run and gate.accepted:
            try:
                _ensure_candidate_in_layer(kb, candidate, "staging")
                kb.promote(candidate_id, from_layer="staging", to_layer="production", gate_report=gate, target_run=False)
                row["production_promoted"] = True
                row["production_blocked"] = False
                production_promoted_count += 1
            except ValueError as exc:
                row["promotion_reasons"].append(str(exc))
        else:
            row["promotion_reasons"].extend(reasons)
        candidates.append(row)

    accepted = bool(staging) and bool(candidates) and all(row["candidate_validation"].get("accepted") for row in candidates)
    if allow_production and not target_run:
        accepted = accepted and gate.accepted and production_promoted_count == len(candidates)
    return {
        "schema_version": SELF_EVO_REPLAY_REPORT_SCHEMA,
        "accepted": accepted,
        "target_run": bool(target_run),
        "allow_production": bool(allow_production),
        "production_write_blocked": bool(target_run or not allow_production or not gate.accepted),
        "production_promoted_count": production_promoted_count,
        "staging_candidate_count": len(staging),
        "benchmark_gate": gate.to_dict(),
        "candidate_reports": candidates,
        "kb": kb.to_dict(),
        "reasons": sorted(set(str(reason) for reason in reasons)),
    }


def _kb_from_report(report: dict[str, Any]) -> LayeredKnowledgeBase:
    data = dict(report or {})
    if data.get("schema_version") == "self_evo_staging_compile_report.v1":
        data = dict(data.get("kb") or {})
    layers = dict(data.get("layers") or {})
    kb = LayeredKnowledgeBase()
    for layer in ("candidate", "shadow", "staging", "production"):
        for candidate_id, payload in dict(layers.get(layer) or {}).items():
            candidate = evolution_candidate_from_dict(dict(payload or {}))
            _ensure_candidate_in_layer(kb, candidate, layer)
    return kb


def _ensure_candidate_in_layer(kb: LayeredKnowledgeBase, candidate: EvolutionCandidate, layer: str) -> None:
    if layer == "candidate":
        if candidate.candidate_id not in kb.layers["candidate"]:
            kb.add_candidate(candidate, target_run=True)
        return
    if candidate.candidate_id not in kb.layers["candidate"]:
        kb.add_candidate(candidate, target_run=True)
    if layer in {"shadow", "staging", "production"} and candidate.candidate_id not in kb.layers["shadow"]:
        kb.promote(candidate.candidate_id, from_layer="candidate", to_layer="shadow", target_run=True)
    if layer in {"staging", "production"} and candidate.candidate_id not in kb.layers["staging"]:
        kb.promote(candidate.candidate_id, from_layer="shadow", to_layer="staging", target_run=True)
    if layer == "production" and candidate.candidate_id not in kb.layers["production"]:
        kb.layers["production"][candidate.candidate_id] = candidate
