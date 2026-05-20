"""Learned failure-risk policy for planner retry control.

This module turns historical route-search failures into retry suggestions. It
does not judge reaction validity and does not replace the one-step proposal
models; it only predicts likely planner bottlenecks from an exported route
payload or target-level benchmark row.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from cascade_planner.agent.schemas import SearchSuggestion
from cascade_planner.eval.train_failure_classifier_from_pack import (
    FailureClassifier,
    row_features,
)


DEFAULT_FAILURE_MODEL = Path("results/shared/failure_classifier/pack_failure_classifier_20260507.pt")

GENERATOR_RECALL_LABELS = {
    "generator_exact_miss",
    "reactant_present_exact_missing",
    "candidate_generator_reactant_miss",
    "candidate_generator_reaction_detail_miss",
}

SELECTOR_OR_COMPOSITION_LABELS = {
    "selector_missed_exact_candidate",
    "selector_missed_gt_reactant_candidate",
    "partial_exact_but_route_order_or_other_steps_miss",
    "route_composition_or_order_miss",
}


def failure_row_from_payload(
    payload: dict[str, Any],
    *,
    route_domain: str | None = None,
    depth: int | None = None,
) -> dict[str, Any]:
    """Build classifier features from a route_results_payload-style object."""
    if "planner_output" in payload and isinstance(payload.get("planner_output"), dict):
        route_domain = route_domain or payload.get("route_domain")
        depth = depth or payload.get("depth")
        top_metrics = payload.get("metrics") or {}
        payload = payload["planner_output"]
    else:
        top_metrics = {}

    routes = payload.get("routes") or []
    route_metrics = [r.get("metrics") or {} for r in routes]
    professional_solved = any(bool(m.get("professional_solved")) for m in route_metrics)
    best_depth = depth or _best_depth(payload, routes)
    metrics = {
        "plan": bool(routes),
        "filled_route_any": _any_metric(route_metrics, "filled_route", fallback=top_metrics.get("filled_route_any")),
        "strict_stock_solve_any": _any_metric(
            route_metrics,
            "strict_stock_solve",
            fallback=top_metrics.get("strict_stock_solve_any"),
        ),
        "condition_window_success_any": _any_nested_metric(
            route_metrics,
            "condition",
            "condition_window_success",
            fallback=top_metrics.get("condition_window_success_any"),
        ),
        "cascade_compatibility_success_any": _any_nested_metric(
            route_metrics,
            "cascade_compatibility",
            "cascade_compatibility_success",
            fallback=top_metrics.get("cascade_compatibility_success_any"),
        ),
        "terminal_GT_reactant_in_top5": top_metrics.get("terminal_GT_reactant_in_top5"),
        "filled_type_GT@1": top_metrics.get("filled_type_GT@1"),
        "filled_type_GT@5": top_metrics.get("filled_type_GT@5"),
        "skeleton_type_GT@1": top_metrics.get("skeleton_type_GT@1"),
        "skeleton_type_GT@5": top_metrics.get("skeleton_type_GT@5"),
    }
    ui_meta = payload.get("ui_metadata") or {}
    status = payload.get("search_status") or {}
    return {
        "target_smiles": payload.get("target") or top_metrics.get("target_smiles") or "",
        "route_domain": route_domain or ui_meta.get("domain") or "chemoenzymatic",
        "depth": best_depth,
        "n_routes": len(routes),
        "labels": [],
        "has_failure_label": bool(status and not status.get("solved")),
        "metrics": metrics,
        "deterministic_status": {
            "professional_solved": professional_solved or bool(status.get("solved")),
            "stock_closed": metrics["strict_stock_solve_any"],
            "condition_ok": metrics["condition_window_success_any"],
            "compatibility_ok": metrics["cascade_compatibility_success_any"],
        },
    }


def predict_failure_risk(
    row_or_payload: dict[str, Any],
    *,
    model_path: str | Path = DEFAULT_FAILURE_MODEL,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict likely failure labels and search-control suggestions.

    The input can be a training-pack row, a web route payload, or a benchmark
    target row containing ``planner_output``.
    """
    path = Path(model_path)
    if not path.exists():
        return {
            "available": False,
            "model_path": str(path),
            "reason": "model_not_found",
            "labels": [],
            "active_labels": [],
            "search_suggestions": [],
            "source": "learned_failure_classifier",
        }

    row = (
        failure_row_from_payload(row_or_payload)
        if _looks_like_payload(row_or_payload)
        else dict(row_or_payload)
    )
    model, schema = _load_model(str(path))
    n_bits = int(schema.get("n_bits") or 128)
    labels = list(schema.get("labels") or [])
    x = torch.tensor(row_features(row, n_bits=n_bits), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().tolist()
    scored = [
        {
            "label": label,
            "probability": round(float(prob), 6),
            "support": (schema.get("label_counts") or {}).get(label),
        }
        for label, prob in zip(labels, probs)
    ]
    scored.sort(key=lambda item: item["probability"], reverse=True)
    active, suppressed = _filter_contradicted_labels(scored, row, threshold=threshold)
    result = {
        "available": True,
        "model_path": str(path),
        "threshold": threshold,
        "labels": scored,
        "active_labels": active,
        "suppressed_labels": suppressed,
        "search_suggestions": [asdict(s) for s in suggestions_from_labels(active)],
        "source": "learned_failure_classifier",
    }
    if _looks_like_payload(row_or_payload):
        result["retry_policy"] = retry_policy_from_failure_risk(
            result,
            _settings_from_payload(row_or_payload),
        )
    return result


def retry_policy_from_failure_risk(
    failure_risk: dict[str, Any],
    current_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate failure labels into bounded retry settings.

    This policy is conservative: it proposes parameter changes and marks whether
    an automatic retry is safe, but it does not execute a second planner run.
    """
    current = _normalized_settings(current_settings or {})
    adjusted = dict(current)
    actions: list[dict[str, Any]] = []
    labels = {str(item.get("label") or "") for item in failure_risk.get("active_labels") or []}
    solved = bool(current.get("solved"))

    if labels & GENERATOR_RECALL_LABELS:
        _raise_setting(adjusted, "candidate_budget", max(current["candidate_budget"] * 2, 8), limit=20)
        _raise_setting(adjusted, "skeleton_samples", max(current["skeleton_samples"], 4), limit=20)
        actions.append({
            "action": "overfetch_single_step_candidates",
            "reason": "single-step proposal recall is likely the limiting factor",
        })

    if labels & SELECTOR_OR_COMPOSITION_LABELS:
        _raise_setting(adjusted, "candidate_budget", max(current["candidate_budget"] * 2, 12), limit=20)
        _raise_setting(adjusted, "n_results", max(current["n_results"], 5), limit=10)
        actions.append({
            "action": "keep_more_route_alternatives",
            "reason": "the correct local candidate may exist but route-level selection is weak",
        })

    if "stock_dead_end" in labels:
        adjusted["search_mode"] = "adaptive"
        adjusted["retry_search_mode"] = "stock_rescue"
        adjusted["planner_strategy"] = "stock_rescue"
        _raise_setting(adjusted, "max_steps", current["max_steps"] + 1, limit=8)
        _raise_setting(adjusted, "candidate_budget", max(current["candidate_budget"], 8), limit=20)
        actions.append({
            "action": "retry_stock_closed_andor",
            "reason": "terminal leaves are likely not closing to stock",
        })

    if labels & {"condition_failure", "compatibility_failure"}:
        _raise_setting(adjusted, "skeleton_samples", max(current["skeleton_samples"], 4), limit=20)
        _raise_setting(adjusted, "n_results", max(current["n_results"], 5), limit=10)
        adjusted["condition_strategy"] = "seek_condition_compatible_alternative"
        actions.append({
            "action": "search_condition_compatible_alternatives",
            "reason": "route may be retrosynthetically solved but weak as a cascade",
        })

    if actions:
        min_expansion = adjusted["max_steps"] * adjusted["skeleton_samples"] * adjusted["candidate_budget"] * 6
        _raise_setting(adjusted, "expansion_budget", max(current["expansion_budget"], min_expansion), limit=1000)
    changed = {
        key: value
        for key, value in adjusted.items()
        if current.get(key) != value
    }
    manual_only_actions = {str(action.get("action") or "") for action in actions}
    requires_manual = bool(manual_only_actions & {"retry_stock_closed_andor"})
    automatic = bool(actions) and not solved and not requires_manual
    return {
        "would_retry": bool(actions),
        "automatic_retry_safe": automatic,
        "current_settings": current,
        "adjusted_settings": adjusted,
        "changed_settings": changed,
        "actions": actions,
        "note": (
            "safe for automatic retry" if automatic
            else "manual review recommended before retry" if actions
            else "no learned retry action above threshold"
        ),
    }


def suggestions_from_labels(active_labels: list[dict[str, Any]]) -> list[SearchSuggestion]:
    labels = {str(item.get("label") or "") for item in active_labels}
    suggestions: list[SearchSuggestion] = []
    if labels & GENERATOR_RECALL_LABELS:
        suggestions.append(SearchSuggestion(
            "increase_candidate_budget",
            "Single-step models likely need more overfetch and route-context reranking.",
            8,
        ))
    if labels & {"stock_dead_end", "no_professional_solved_route"}:
        suggestions.append(SearchSuggestion(
            "try_alternative_route_mode",
            "Retry with stricter stock-closed AND-OR search or a deeper route horizon.",
            None,
        ))
    if labels & {"condition_failure", "compatibility_failure"}:
        suggestions.append(SearchSuggestion(
            "relax_condition_window",
            "Consider sequential operation or broader cascade-condition windows.",
            None,
        ))
    if labels & SELECTOR_OR_COMPOSITION_LABELS:
        suggestions.append(SearchSuggestion(
            "increase_candidate_budget",
            "The candidate may exist but route-level selection is weak; keep more alternatives per step.",
            12,
        ))
    if not suggestions:
        suggestions.append(SearchSuggestion(
            "request_more_evidence",
            "No high-confidence learned retry action; inspect candidate pools and route critiques.",
            None,
        ))
    return _dedupe_suggestions(suggestions)


def _settings_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "planner_output" in payload and isinstance(payload.get("planner_output"), dict):
        payload = payload["planner_output"]
    ui = payload.get("ui_metadata") or {}
    status = payload.get("search_status") or {}
    return {
        "search_mode": ui.get("search_mode"),
        "planner_mode": ui.get("planner_mode"),
        "min_steps": ui.get("min_steps"),
        "max_steps": ui.get("max_steps"),
        "skeleton_samples": ui.get("skeleton_samples"),
        "candidate_budget": ui.get("candidate_budget"),
        "expansion_budget": ui.get("expansion_budget"),
        "n_results": payload.get("n_results"),
        "solved": status.get("solved"),
    }


def _normalized_settings(settings: dict[str, Any]) -> dict[str, Any]:
    min_steps = _as_int(settings.get("min_steps"), 3)
    max_steps = _as_int(settings.get("max_steps"), max(min_steps, 3))
    if min_steps > max_steps:
        min_steps, max_steps = max_steps, min_steps
    return {
        "search_mode": str(settings.get("search_mode") or "adaptive"),
        "planner_mode": str(settings.get("planner_mode") or "advanced"),
        "min_steps": max(1, min(8, min_steps)),
        "max_steps": max(1, min(8, max_steps)),
        "skeleton_samples": max(1, min(20, _as_int(settings.get("skeleton_samples"), 2))),
        "candidate_budget": max(1, min(20, _as_int(settings.get("candidate_budget"), 4))),
        "expansion_budget": max(1, min(1000, _as_int(settings.get("expansion_budget"), 128))),
        "n_results": max(1, min(10, _as_int(settings.get("n_results"), 3))),
        "solved": bool(settings.get("solved")),
    }


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _raise_setting(settings: dict[str, Any], key: str, value: int, *, limit: int) -> None:
    settings[key] = min(limit, max(int(settings.get(key) or 0), int(value)))


@lru_cache(maxsize=4)
def _load_model(path: str) -> tuple[FailureClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    schema = dict(checkpoint.get("feature_schema") or {})
    labels = list(schema.get("labels") or [])
    feature_dim = int(schema.get("feature_dim") or 0)
    hidden = int(checkpoint.get("hidden") or 160)
    model = FailureClassifier(feature_dim, len(labels), hidden=hidden)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, schema


def _looks_like_payload(data: dict[str, Any]) -> bool:
    return "planner_output" in data or "routes" in data or "ui_metadata" in data or "search_status" in data


def _best_depth(payload: dict[str, Any], routes: list[dict[str, Any]]) -> int:
    status = payload.get("search_status") or {}
    if status.get("best_depth"):
        return int(status["best_depth"])
    attempts = payload.get("depth_attempts") or []
    solved = [a for a in attempts if (a.get("status") or "").lower() == "solved"]
    if solved:
        return int(solved[0].get("depth") or 0)
    if attempts:
        return int((attempts[-1] or {}).get("depth") or 0)
    if routes:
        return int(routes[0].get("n_steps") or len(routes[0].get("steps") or []))
    return 0


def _any_metric(metrics: list[dict[str, Any]], key: str, *, fallback: Any = None) -> bool | None:
    values = [m.get(key) for m in metrics if m.get(key) is not None]
    if values:
        return any(bool(v) for v in values)
    return fallback if fallback is not None else None


def _any_nested_metric(
    metrics: list[dict[str, Any]],
    outer_key: str,
    inner_key: str,
    *,
    fallback: Any = None,
) -> bool | None:
    values = [
        (m.get(outer_key) or {}).get(inner_key)
        for m in metrics
        if (m.get(outer_key) or {}).get(inner_key) is not None
    ]
    if values:
        return any(bool(v) for v in values)
    return fallback if fallback is not None else None


def _dedupe_suggestions(suggestions: list[SearchSuggestion]) -> list[SearchSuggestion]:
    seen = set()
    out = []
    for suggestion in suggestions:
        key = (suggestion.action, suggestion.budget_hint)
        if key in seen:
            continue
        seen.add(key)
        out.append(suggestion.normalize())
    return out


def _filter_contradicted_labels(
    scored: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status = row.get("deterministic_status") or {}
    metrics = row.get("metrics") or {}
    contradictions = {
        "no_professional_solved_route": bool(status.get("professional_solved")),
        "generator_exact_miss": bool(status.get("professional_solved")),
        "reactant_present_exact_missing": bool(status.get("professional_solved")),
        "candidate_generator_reactant_miss": bool(status.get("professional_solved")),
        "candidate_generator_reaction_detail_miss": bool(status.get("professional_solved")),
        "selector_missed_exact_candidate": bool(status.get("professional_solved")),
        "selector_missed_gt_reactant_candidate": bool(status.get("professional_solved")),
        "partial_exact_but_route_order_or_other_steps_miss": bool(status.get("professional_solved")),
        "route_composition_or_order_miss": bool(status.get("professional_solved")),
        "stock_dead_end": status.get("stock_closed") is True or metrics.get("strict_stock_solve_any") is True,
        "condition_failure": status.get("condition_ok") is True or metrics.get("condition_window_success_any") is True,
        "compatibility_failure": status.get("compatibility_ok") is True or metrics.get("cascade_compatibility_success_any") is True,
    }
    active = []
    suppressed = []
    for item in scored:
        if item["probability"] < threshold:
            continue
        if contradictions.get(str(item.get("label") or "")):
            suppressed_item = dict(item)
            suppressed_item["reason"] = "contradicted_by_route_metrics"
            suppressed.append(suppressed_item)
        else:
            active.append(item)
    return active, suppressed
