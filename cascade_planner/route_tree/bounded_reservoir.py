"""Opt-in bounded native ChemEnzy route reservoir for route-tree runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from cascade_planner.cascadeboard import CascadeBoard, RouteExplanation, RouteResult
from cascade_planner.cascadeboard.route_export import route_metrics
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.eval.chem_enzy_broad_union import _chem_route_stock_closed, _select_chem_routes


StockChecker = Callable[[str], bool]


def bounded_reservoir_enabled() -> bool:
    return _env_truthy("AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR")


def append_bounded_native_reservoir(
    *,
    target: str,
    results: list[RouteResult],
    stock_checker: StockChecker | None,
    max_depth: int,
    n_results: int,
) -> list[RouteResult]:
    """Append up to ``AUTOPLANNER_RESERVOIR_NATIVE_TOPK`` native routes.

    The reservoir is a safety net, not a replacement policy. By default it only
    runs when the current result set lacks a strict stock-closed route.
    """

    if not bounded_reservoir_enabled():
        return results
    topk = max(0, _env_int("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", 5))
    if topk <= 0:
        return results
    min_native_routes = min(topk, max(0, _env_int("AUTOPLANNER_RESERVOIR_MIN_NATIVE_ROUTES", 0)))
    base_has_stock_closed = _has_stock_closed(results, stock_checker=stock_checker)
    use_full_cap = (
        not base_has_stock_closed
        or _stock_risk_high(results)
        or _controller_confidence_low(results)
        or _controller_fallback_group_high(results)
        or _controller_candidate_bottleneck(results)
        or _env_truthy("AUTOPLANNER_RESERVOIR_ALWAYS_ON")
    )
    if not use_full_cap and min_native_routes <= 0:
        return results
    native_cap = topk if use_full_cap else min_native_routes
    if native_cap <= 0:
        return results
    native_routes = _load_or_collect_native_routes(
        target=target,
        max_depth=max_depth,
        topk=native_cap,
        results=results,
    )
    if not native_routes:
        return results
    selected = _select_native_route_dicts(
        native_routes,
        topk=native_cap,
        target=target,
        stock_checker=stock_checker,
    )
    selected = _gate_unverified_stock_reservoir(
        selected,
        target=target,
        stock_checker=stock_checker,
        base_has_stock_closed=base_has_stock_closed,
    )
    selected = _filter_native_routes_for_quality(
        selected,
        target=target,
        stock_checker=stock_checker,
    )
    if not selected:
        return results
    reservoir_results = [
        _route_dict_to_result(
            target=target,
            route=route,
            rank=rank,
            topk=native_cap,
        )
        for rank, route in enumerate(selected, 1)
    ]
    return [*results[: max(0, int(n_results or 0))], *reservoir_results[:native_cap]]


def annotate_bounded_reservoir_payload(payload: dict[str, Any]) -> None:
    routes = payload.get("routes") or []
    reservoir_routes = []
    for idx, route in enumerate(routes, 1):
        table = ((route.get("explanation") or {}).get("uncertainty_table") or {})
        broad = table.get("broad_reservoir")
        if broad:
            reservoir_routes.append({"route_rank": idx, **dict(broad)})
            route["broad_reservoir"] = dict(broad)
    if reservoir_routes:
        payload["broad_reservoir"] = {
            "enabled": True,
            "native_topk": _env_int("AUTOPLANNER_RESERVOIR_NATIVE_TOPK", 5),
            "native_selection": os.environ.get("AUTOPLANNER_RESERVOIR_SELECTION") or "rank_plus_stock",
            "native_route_count": len(reservoir_routes),
            "routes": reservoir_routes,
        }


def _load_or_collect_native_routes(
    *,
    target: str,
    max_depth: int,
    topk: int,
    results: list[RouteResult] | None = None,
) -> list[dict[str, Any]]:
    payload_path = os.environ.get("AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD") or ""
    if payload_path:
        routes = _routes_from_payload(Path(payload_path), target=target)
        if routes:
            return routes
    if not _online_native_collect_allowed(results or []):
        return []
    return _collect_native_routes(target=target, max_depth=max_depth, topk=topk)


def _collect_native_routes(*, target: str, max_depth: int, topk: int) -> list[dict[str, Any]]:
    try:
        from cascade_planner.baselines.chem_enzy_adapter import ChemEnzyBackendAdapter
        from cascade_planner.baselines.route_contract import RouteSearchConfig

        adapter = ChemEnzyBackendAdapter()
        result = adapter.run_target(
            RouteSearchConfig(
                target_smiles=target,
                max_depth=max_depth,
                max_iterations=_env_int("AUTOPLANNER_RESERVOIR_NATIVE_ITERATIONS", 10),
                expansion_topk=max(topk, _env_int("AUTOPLANNER_RESERVOIR_NATIVE_EXPANSION_TOPK", 50)),
            )
        )
        return [route.to_dict() for route in result.routes]
    except Exception:
        return []


def _routes_from_payload(path: Path, *, target: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("targets") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    target_key = canonical_smiles(target) or target
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        row_key = canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
        if row_key != target_key:
            continue
        if row.get("routes"):
            return list(row.get("routes") or [])
        routes = list(((row.get("planner_output") or {}).get("routes") or []))
        reservoir_routes = [route for route in routes if route.get("broad_reservoir")]
        return reservoir_routes or routes
    return []


def _select_native_route_dicts(
    routes: list[dict[str, Any]],
    *,
    topk: int,
    target: str,
    stock_checker: StockChecker | None,
) -> list[dict[str, Any]]:
    selection = os.environ.get("AUTOPLANNER_RESERVOIR_SELECTION") or "rank_plus_stock"
    if _native_route_format(routes):
        selected = _select_chem_routes(routes, topk=topk, selection=selection)
        return _prefer_runtime_stock_route(
            routes=routes,
            selected=selected,
            topk=topk,
            selection=selection,
            target=target,
            stock_checker=stock_checker,
        )
    selected = list(routes[:topk])
    if selection == "rank_plus_stock" and topk > 0 and not any(_exported_route_stock_closed(route) for route in selected):
        stock_route = next((route for route in routes if _exported_route_stock_closed(route)), None)
        if stock_route is not None:
            selected = [*selected[: topk - 1], stock_route]
    elif selection == "stock_first":
        selected = sorted(routes, key=lambda route: (not _exported_route_stock_closed(route), _route_rank(route)))[:topk]
    return _prefer_runtime_stock_route(
        routes=routes,
        selected=selected,
        topk=topk,
        selection=selection,
        target=target,
        stock_checker=stock_checker,
    )


def _prefer_runtime_stock_route(
    *,
    routes: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    topk: int,
    selection: str,
    target: str,
    stock_checker: StockChecker | None,
) -> list[dict[str, Any]]:
    selected = list(selected[:topk])
    if selection != "rank_plus_stock" or topk <= 0 or stock_checker is None:
        return selected
    if any(_route_runtime_stock_closed(route, target=target, stock_checker=stock_checker) for route in selected):
        return selected
    ranked = _select_chem_routes(routes, topk=None, selection="rank") if _native_route_format(routes) else list(routes)
    stock_route = next(
        (
            route
            for route in ranked
            if _route_runtime_stock_closed(route, target=target, stock_checker=stock_checker)
        ),
        None,
    )
    if stock_route is None:
        return selected
    return [*selected[: topk - 1], stock_route][:topk]


def _gate_unverified_stock_reservoir(
    routes: list[dict[str, Any]],
    *,
    target: str,
    stock_checker: StockChecker | None,
    base_has_stock_closed: bool,
) -> list[dict[str, Any]]:
    if (
        base_has_stock_closed
        or stock_checker is None
        or not _env_truthy_default("AUTOPLANNER_RESERVOIR_REQUIRE_RUNTIME_STOCK_WHEN_NO_STOCK", True)
    ):
        return routes
    if any(_route_runtime_stock_closed(route, target=target, stock_checker=stock_checker) for route in routes):
        return routes
    return []


def _filter_native_routes_for_quality(
    routes: list[dict[str, Any]],
    *,
    target: str,
    stock_checker: StockChecker | None,
) -> list[dict[str, Any]]:
    if not _env_truthy("AUTOPLANNER_RESERVOIR_QUALITY_FILTER"):
        return routes
    filtered = [
        route
        for route in routes
        if _native_route_quality_pass(route, target=target, stock_checker=stock_checker)
    ]
    if filtered or _env_truthy_default("AUTOPLANNER_RESERVOIR_QUALITY_FILTER_ALLOW_EMPTY", True):
        return filtered
    return routes


def _native_route_quality_pass(
    route: dict[str, Any],
    *,
    target: str,
    stock_checker: StockChecker | None,
) -> bool:
    try:
        result = _route_dict_to_result(target=target, route=route, rank=1, topk=1)
        metrics = route_metrics(result.board, stock_checker=stock_checker) if stock_checker is not None else route_metrics(result.board)
    except Exception:
        return False
    if metrics.get("filled_route") is False or metrics.get("route_solved") is False:
        return False
    if stock_checker is not None and metrics.get("strict_stock_solve") is not True:
        return False
    naturalness = metrics.get("route_naturalness") or {}
    for key in ("unfilled_steps", "product_mismatch_steps", "atom_balance_violations", "self_loop_steps"):
        if int(naturalness.get(key) or 0) > 0:
            return False
    if _native_route_low_confidence_no_progress(route=route, metrics=metrics):
        return False
    return True


def _native_route_low_confidence_no_progress(*, route: dict[str, Any], metrics: dict[str, Any]) -> bool:
    min_prob = None
    low_confidence_steps = 0
    for step in route.get("steps") or []:
        prob = _step_probability(step)
        if prob is None:
            continue
        min_prob = prob if min_prob is None else min(min_prob, prob)
        if prob <= _env_float("AUTOPLANNER_RESERVOIR_MIN_NATIVE_STEP_PROB", 0.005):
            low_confidence_steps += 1
    if low_confidence_steps <= 0:
        return False
    progress = metrics.get("retrosynthesis_progress") or {}
    progressive = bool(
        metrics.get("progressive_route")
        or progress.get("retrosynthesis_progress_success")
        or _safe_float(progress.get("progressive_step_fraction"), 0.0) >= 0.5
        or _safe_float(progress.get("main_chain_reduction"), 0.0) >= 0.25
    )
    return not progressive


def _step_probability(step: dict[str, Any]) -> float | None:
    scores = step.get("scores") or {}
    for key in ("confidence", "retro", "probability", "score"):
        value = _safe_float(scores.get(key), None)
        if value is not None and 0.0 <= value <= 1.0:
            return value
    value = _safe_float(step.get("score"), None)
    if value is not None and 0.0 <= value <= 1.0:
        return value
    return None


def _route_runtime_stock_closed(route: dict[str, Any], *, target: str, stock_checker: StockChecker) -> bool:
    try:
        result = _route_dict_to_result(target=target, route=route, rank=1, topk=1)
        return route_metrics(result.board, stock_checker=stock_checker).get("strict_stock_solve") is True
    except Exception:
        return False


def _route_dict_to_result(*, target: str, route: dict[str, Any], rank: int, topk: int) -> RouteResult:
    board = CascadeBoard.from_n_steps(len(route.get("steps") or []), target)
    for idx, step in enumerate(route.get("steps") or []):
        slot = board.slots[idx]
        reactants = [str(smi) for smi in step.get("reactant_smiles") or [] if smi]
        if not reactants:
            if step.get("main_reactant"):
                reactants.append(str(step.get("main_reactant")))
            reactants.extend(str(smi) for smi in step.get("aux_reactants") or [] if smi)
        slot.product = step.get("product") or step.get("product_smiles") or (target if idx == 0 else None)
        slot.main_reactant = step.get("main_reactant") or (reactants[0] if reactants else None)
        slot.aux_reactants = list(step.get("aux_reactants") or reactants[1:])
        slot.reaction_smiles = step.get("reaction_smiles") or step.get("rxn_smiles")
        slot.reaction_type = step.get("reaction_type") or ""
        slot.ec = step.get("ec") or None
        slot.source = step.get("source") or step.get("source_model") or "ChemEnzyRetroPlanner"
        scores = step.get("scores") or {}
        slot.e_retro = scores.get("retro") if scores else step.get("score")
        slot.confidence = float(scores.get("confidence") or step.get("score") or 0.5)
    stock_overrides = _native_terminal_stock_overrides(route) if _env_truthy("AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK") else {}
    if stock_overrides:
        board.set_global_constraint("stock_overrides", stock_overrides)
        board.set_global_constraint("stock_override_source", "native_chem_enzy")
    stock_closed = _chem_route_stock_closed(route) if _native_route_format([route]) else _exported_route_stock_closed(route)
    explanation = RouteExplanation(
        why_selected="Bounded native ChemEnzy reservoir safety route.",
        uncertainty_table={
            "broad_reservoir": {
                "source": "native_chem_enzy",
                "native_rank": route.get("_native_rank") or route.get("route_rank") or rank,
                "native_topk": topk,
                "native_selection": os.environ.get("AUTOPLANNER_RESERVOIR_SELECTION") or "rank_plus_stock",
                "stock_closed": bool(stock_closed),
            }
        },
    )
    return RouteResult(
        board=board,
        quality_vector={"stock_closed": float(bool(stock_closed))},
        score=float(route.get("score") or 0.0),
        confidence=0.55,
        constraint_report={
            "search_mode": "bounded_reservoir",
            "native_stock_override_count": sum(1 for value in stock_overrides.values() if value),
        },
        explanation=explanation,
    )


def _native_terminal_stock_overrides(route: dict[str, Any]) -> dict[str, bool]:
    products = {
        canonical_smiles(str(step.get("product_smiles") or step.get("product") or ""))
        for step in route.get("steps") or []
        if step.get("product_smiles") or step.get("product")
    }
    products = {smi for smi in products if smi}
    overrides: dict[str, bool] = {}
    for step in route.get("steps") or []:
        stock_status = step.get("stock_status") or {}
        if not isinstance(stock_status, dict):
            continue
        status_by_key = {
            (canonical_smiles(str(smi)) or str(smi)): bool(value)
            for smi, value in stock_status.items()
            if smi
        }
        for smi in step.get("reactant_smiles") or []:
            key = canonical_smiles(str(smi)) or str(smi)
            if not key or key in products or key not in status_by_key:
                continue
            overrides[key] = bool(status_by_key[key])
    return overrides


def _has_stock_closed(results: list[RouteResult], *, stock_checker: StockChecker | None) -> bool:
    if stock_checker is None:
        return False
    for result in results:
        try:
            if route_metrics(result.board, stock_checker=stock_checker).get("strict_stock_solve") is True:
                return True
        except Exception:
            continue
    return False


def _stock_risk_high(results: list[RouteResult]) -> bool:
    for result in results:
        risk = result.risk_vector or {}
        if float(risk.get("stock_dead_end") or risk.get("stock_risk") or 0.0) >= _env_float("AUTOPLANNER_RESERVOIR_STOCK_RISK_THRESHOLD", 0.70):
            return True
        table = result.explanation.uncertainty_table if result.explanation else {}
        if float(table.get("stock_dead_end_prob") or table.get("stock_risk") or 0.0) >= _env_float("AUTOPLANNER_RESERVOIR_STOCK_RISK_THRESHOLD", 0.70):
            return True
    return False


def _controller_confidence_low(results: list[RouteResult]) -> bool:
    threshold = _env_float("AUTOPLANNER_RESERVOIR_LOW_CONFIDENCE_THRESHOLD", 0.55)
    min_events = max(1, _env_int("AUTOPLANNER_RESERVOIR_LOW_CONFIDENCE_MIN_EVENTS", 1))
    fallback_hits = 0
    fallback_total = 0
    for result in results:
        table = result.explanation.uncertainty_table if result.explanation else {}
        if table.get("route_tree_controller_active") is not True:
            continue
        for row in table.get("route_tree_source_budgets") or []:
            gate = (row.get("proposal_gate") or {}) if isinstance(row, dict) else {}
            probs = gate.get("source_group_probs") or {}
            selected = str(gate.get("selected_source_group") or "")
            fallback_prob = _safe_float(probs.get("fallback"), 0.0)
            top_prob = max((_safe_float(value, 0.0) for value in probs.values()), default=0.0)
            if selected == "fallback" and fallback_prob >= threshold and fallback_prob >= top_prob:
                fallback_hits += 1
            if probs:
                fallback_total += 1
    if fallback_total <= 0:
        return False
    return fallback_hits >= min_events


def _controller_candidate_bottleneck(results: list[RouteResult]) -> bool:
    """Trigger the native safety net for controller fallback selector misses."""

    if not _controller_fallback_reason_seen(results):
        return False
    for result in results:
        table = result.explanation.uncertainty_table if result.explanation else {}
        recall_rows = table.get("route_tree_proposal_recall_diagnostics") or []
        for row in recall_rows:
            if not isinstance(row, dict):
                continue
            if row.get("candidate_exact_reaction_hit") is True or row.get("candidate_gt_reactant_hit") is True:
                return True
        recovery = table.get("route_recovery") or table.get("recovery") or {}
        if isinstance(recovery, dict):
            if recovery.get("candidate_exact_reaction_in_pool") and not recovery.get("exact_reaction_in_route_pool"):
                return True
            if recovery.get("candidate_gt_reactant_in_pool") and not recovery.get("gt_reactant_in_route_pool"):
                return True
    return False


def _controller_fallback_group_high(results: list[RouteResult]) -> bool:
    threshold = _env_float("AUTOPLANNER_RESERVOIR_FALLBACK_GROUP_RESERVOIR_THRESHOLD", 0.625)
    min_events = max(1, _env_int("AUTOPLANNER_RESERVOIR_FALLBACK_GROUP_MIN_EVENTS", 1))
    hits = 0
    for result in results:
        table = result.explanation.uncertainty_table if result.explanation else {}
        if table.get("route_tree_controller_active") is not True:
            continue
        for row in table.get("route_tree_source_budgets") or []:
            gate = (row.get("proposal_gate") or {}) if isinstance(row, dict) else {}
            for key in ("fallback_reason", "policy_reason"):
                if _fallback_group_score(str(gate.get(key) or "")) >= threshold:
                    hits += 1
                    break
    return hits >= min_events


def _controller_fallback_reason_seen(results: list[RouteResult]) -> bool:
    threshold = _env_float("AUTOPLANNER_RESERVOIR_LOW_CONFIDENCE_THRESHOLD", 0.55)
    min_events = max(1, _env_int("AUTOPLANNER_RESERVOIR_LOW_CONFIDENCE_MIN_EVENTS", 1))
    hits = 0
    for result in results:
        table = result.explanation.uncertainty_table if result.explanation else {}
        if table.get("route_tree_controller_active") is not True:
            continue
        for row in table.get("route_tree_source_budgets") or []:
            gate = (row.get("proposal_gate") or {}) if isinstance(row, dict) else {}
            for key in ("fallback_reason", "policy_reason"):
                score = _fallback_group_score(str(gate.get(key) or ""))
                if score >= threshold:
                    hits += 1
                    break
    return hits >= min_events


def _online_native_collect_allowed(results: list[RouteResult]) -> bool:
    max_prior_elapsed = _env_float("AUTOPLANNER_RESERVOIR_ONLINE_COLLECT_MAX_PRIOR_ELAPSED_S", 16.0)
    if max_prior_elapsed <= 0:
        return True
    elapsed = _max_result_elapsed_s(results)
    if elapsed <= 0:
        return True
    return elapsed <= max_prior_elapsed


def _max_result_elapsed_s(results: list[RouteResult]) -> float:
    elapsed = 0.0
    for result in results:
        table = result.explanation.uncertainty_table if result.explanation else {}
        elapsed = max(elapsed, _safe_float(table.get("elapsed_s"), 0.0))
        outcome = table.get("route_tree_final_outcome") if isinstance(table, dict) else {}
        if isinstance(outcome, dict):
            elapsed = max(elapsed, _safe_float(outcome.get("elapsed_s"), 0.0))
    return elapsed


def _fallback_group_score(reason: str) -> float:
    marker = "fallback_group:"
    if marker not in reason:
        return 0.0
    text = reason.split(marker, 1)[1]
    value = []
    for char in text:
        if char.isdigit() or char == ".":
            value.append(char)
        else:
            break
    return _safe_float("".join(value), 0.0)


def _native_route_format(routes: list[dict[str, Any]]) -> bool:
    for route in routes:
        steps = route.get("steps") or []
        if steps and any("product_smiles" in step or "reactant_smiles" in step for step in steps):
            return True
    return False


def _exported_route_stock_closed(route: dict[str, Any]) -> bool:
    metrics = route.get("metrics") or {}
    if metrics.get("strict_stock_solve") is not None:
        return bool(metrics.get("strict_stock_solve"))
    broad = route.get("broad_reservoir") or {}
    if broad.get("stock_closed") is not None:
        return bool(broad.get("stock_closed"))
    return False


def _route_rank(route: dict[str, Any]) -> int:
    try:
        return int(route.get("_native_rank") or route.get("route_rank") or 0)
    except (TypeError, ValueError):
        return 0


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _env_truthy_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
