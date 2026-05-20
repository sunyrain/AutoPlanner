"""Cascade-native oracle values from bounded ChemEnzy route pools.

The oracle is deliberately advisory. It never owns route-tree search and never
uses benchmark exact/GT labels. Native routes are converted into route-value
and fragment-value hints scored by AutoPlanner's stock/progress/cascade rubric.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_export import route_metrics
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles
from cascade_planner.eval.chem_enzy_broad_union import _chem_route_stock_closed, _select_chem_routes
from cascade_planner.route_tree.bounded_reservoir import _exported_route_stock_closed, _native_route_format, _route_dict_to_result
from cascade_planner.route_tree.schema import CandidateAction


@dataclass(frozen=True)
class CascadeOracleMatch:
    value: float
    confidence: float
    reason: str
    route_rank: int = 0
    step_index: int = 0
    stock_closed: bool = False
    reaction_match: bool = False
    reactant_overlap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CascadeOracleRuntime:
    def __init__(self, payload_path: str | Path):
        self.path = Path(payload_path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.payload = payload if isinstance(payload, dict) else {"targets": payload if isinstance(payload, list) else []}
        self.targets: dict[str, dict[str, Any]] = {}
        for row in self.payload.get("targets") or []:
            if not isinstance(row, dict):
                continue
            key = canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
            if key:
                self.targets[key] = row

    def action_value(self, *, target: str, leaf: str, action: CandidateAction) -> CascadeOracleMatch | None:
        target_row = self._target_row(target)
        if not target_row:
            return None
        action_rxn = canonical_reaction(action.rxn_smiles)
        action_reactants = _reactant_set_from_action(action)
        leaf_key = canonical_smiles(leaf) or leaf
        best: CascadeOracleMatch | None = None
        for route in target_row.get("routes") or []:
            route_value = _safe_float(route.get("oracle_value"), 0.0)
            route_rank = _safe_int(route.get("route_rank"), 0)
            stock_closed = bool(route.get("stock_closed"))
            for step in route.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                product_key = canonical_smiles(str(step.get("product") or "")) or str(step.get("product") or "")
                product_factor = 1.0 if not product_key or product_key == leaf_key else 0.0
                step_rxn = canonical_reaction(str(step.get("reaction_smiles") or ""))
                reaction_match = bool(action_rxn and step_rxn and action_rxn == step_rxn)
                overlap = _jaccard(action_reactants, set(step.get("reactant_keys") or []))
                if product_factor <= 0.0 or (not reaction_match and overlap <= 0.0):
                    continue
                match_strength = 1.0 if reaction_match else overlap
                confidence = max(0.0, min(1.0, product_factor * match_strength))
                step_value = _safe_float(step.get("step_value"), 0.0)
                if reaction_match:
                    step_value = max(step_value, _safe_float(step.get("step_probability"), 0.0))
                value = max(route_value, step_value) * confidence
                reason = "reaction_match" if reaction_match else "reactant_overlap"
                candidate = CascadeOracleMatch(
                    value=round(float(value), 6),
                    confidence=round(float(confidence), 6),
                    reason=reason,
                    route_rank=route_rank,
                    step_index=_safe_int(step.get("step_index"), 0),
                    stock_closed=stock_closed,
                    reaction_match=reaction_match,
                    reactant_overlap=round(float(overlap), 6),
                )
                if best is None or candidate.value > best.value:
                    best = candidate
        return best

    def route_value(self, *, target: str, route_reactions: list[str]) -> CascadeOracleMatch | None:
        target_row = self._target_row(target)
        if not target_row or not route_reactions:
            return None
        pred = {canonical_reaction(rxn) for rxn in route_reactions if canonical_reaction(rxn)}
        if not pred:
            return None
        best: CascadeOracleMatch | None = None
        for route in target_row.get("routes") or []:
            oracle = {
                canonical_reaction(str(step.get("reaction_smiles") or ""))
                for step in route.get("steps") or []
                if canonical_reaction(str(step.get("reaction_smiles") or ""))
            }
            if not oracle:
                continue
            overlap = len(pred & oracle) / max(len(pred | oracle), 1)
            if overlap <= 0:
                continue
            value = _safe_float(route.get("oracle_value"), 0.0) * overlap
            candidate = CascadeOracleMatch(
                value=round(float(value), 6),
                confidence=round(float(overlap), 6),
                reason="route_reaction_overlap",
                route_rank=_safe_int(route.get("route_rank"), 0),
                stock_closed=bool(route.get("stock_closed")),
                reaction_match=bool(pred & oracle),
                reactant_overlap=round(float(overlap), 6),
            )
            if best is None or candidate.value > best.value:
                best = candidate
        return best

    def _target_row(self, target: str) -> dict[str, Any] | None:
        key = canonical_smiles(target) or target
        return self.targets.get(key)


_RUNTIME_CACHE: dict[str, CascadeOracleRuntime | None] = {}


def cascade_oracle_runtime_from_env() -> CascadeOracleRuntime | None:
    if not _env_truthy("AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE"):
        return None
    path = os.environ.get("AUTOPLANNER_CASCADE_ORACLE_PAYLOAD") or ""
    if not path:
        return None
    cached = _RUNTIME_CACHE.get(path)
    if path in _RUNTIME_CACHE:
        return cached
    try:
        runtime = CascadeOracleRuntime(path)
    except Exception:
        runtime = None
    _RUNTIME_CACHE[path] = runtime
    return runtime


def build_cascade_oracle_payload_from_native(
    *,
    native_payload_path: Path,
    output_path: Path,
    topk: int = 5,
    selection: str = "rank_plus_stock",
) -> dict[str, Any]:
    payload = json.loads(Path(native_payload_path).read_text(encoding="utf-8"))
    rows = payload.get("targets") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    targets = []
    for target_row in rows or []:
        if not isinstance(target_row, dict):
            continue
        target = str(target_row.get("target_smiles") or "")
        routes = list(target_row.get("routes") or ((target_row.get("planner_output") or {}).get("routes") or []))
        if not target or not routes:
            continue
        targets.append(
            {
                "target_smiles": target,
                "routes": _oracle_routes_for_target(target=target, routes=routes, topk=topk, selection=selection),
            }
        )
    out = {
        "schema_version": "cascade_oracle_payload.v1",
        "native_payload": str(native_payload_path),
        "topk": int(topk),
        "selection": str(selection),
        "targets": targets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _oracle_routes_for_target(*, target: str, routes: list[dict[str, Any]], topk: int, selection: str) -> list[dict[str, Any]]:
    selected = _select_chem_routes(routes, topk=topk, selection=selection) if _native_route_format(routes) else routes[:topk]
    out = []
    for rank, route in enumerate(selected, start=1):
        value, components = cascade_oracle_route_value(target=target, route=route)
        steps = []
        for idx, step in enumerate(route.get("steps") or []):
            normalized = _normalize_step(step, target=target if idx == 0 else "")
            if not normalized.get("reaction_smiles"):
                continue
            normalized["step_index"] = idx
            step_probability = _step_probability(step)
            step_cost = _step_cost(step)
            normalized["step_probability"] = round(float(step_probability), 6)
            normalized["step_cost"] = round(float(step_cost), 6)
            normalized["step_value"] = round(float(_cost_to_value(step_cost)), 6)
            steps.append(normalized)
        out.append(
            {
                "route_rank": _safe_int(route.get("_native_rank") or route.get("route_rank"), rank),
                "oracle_value": value,
                "oracle_cost": components.get("route_cost"),
                "stock_closed": bool(components.get("stock_closed")),
                "components": components,
                "steps": steps,
            }
        )
    return out


def cascade_oracle_route_value(*, target: str, route: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    stock_closed = _route_stock_closed(route)
    metrics: dict[str, Any] = {}
    try:
        result = _route_dict_to_result(target=target, route=route, rank=1, topk=1)
        metrics = route_metrics(result.board)
    except Exception:
        metrics = {}
    steps = route.get("steps") or []
    step_cost = sum(_step_cost(step) for step in steps)
    if not steps:
        step_cost = math.log1p(1.0)
    terminal_gap_cost = 0.0 if stock_closed else math.log1p(float(_nonstock_terminal_count(route) or 1))
    duplicate_fraction = _duplicate_reaction_fraction(steps)
    duplicate_cost = _negative_log_probability(1.0 - min(duplicate_fraction, 0.999))
    route_cost = step_cost + terminal_gap_cost + duplicate_cost
    value = _cost_to_value(route_cost)
    return round(float(value), 6), {
        "stock_closed": bool(stock_closed),
        "filled_route": bool(metrics.get("filled_route") or steps),
        "route_cost": round(float(route_cost), 6),
        "step_cost": round(float(step_cost), 6),
        "terminal_gap_cost": round(float(terminal_gap_cost), 6),
        "duplicate_reaction_fraction": round(float(duplicate_fraction), 6),
        "duplicate_cost": round(float(duplicate_cost), 6),
        "cost_model": "reaction_cost_and_or.v1",
    }


def _normalize_step(step: dict[str, Any], *, target: str = "") -> dict[str, Any]:
    reactants = [str(smi) for smi in step.get("reactant_smiles") or [] if smi]
    if not reactants:
        if step.get("main_reactant"):
            reactants.append(str(step.get("main_reactant")))
        reactants.extend(str(smi) for smi in step.get("aux_reactants") or [] if smi)
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    if not reactants and rxn and ">>" in rxn:
        reactants = [smi for smi in rxn.split(">>", 1)[0].split(".") if smi]
    product = str(step.get("product") or step.get("product_smiles") or target or "")
    return {
        "product": product,
        "reaction_smiles": rxn,
        "reaction_key": canonical_reaction(rxn),
        "reactants": reactants,
        "reactant_keys": list(canonical_side(".".join(reactants))) if reactants else [],
        "source": str(step.get("source") or step.get("source_model") or "native_chem_enzy"),
    }


def _route_stock_closed(route: dict[str, Any]) -> bool:
    try:
        if _native_route_format([route]):
            return bool(_chem_route_stock_closed(route))
    except Exception:
        pass
    return bool(_exported_route_stock_closed(route) or route.get("stock_closed"))


def _reactant_set_from_action(action: CandidateAction) -> set[str]:
    return {canonical_smiles(smi) or smi for smi in action.reactants if (canonical_smiles(smi) or smi)}


def _duplicate_reaction_fraction(steps: list[dict[str, Any]]) -> float:
    keys = [canonical_reaction(str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")) for step in steps]
    keys = [key for key in keys if key]
    if not keys:
        return 0.0
    return 1.0 - len(set(keys)) / max(len(keys), 1)


def _step_probability(step: dict[str, Any]) -> float:
    scores = step.get("scores") if isinstance(step.get("scores"), dict) else {}
    for value in (
        step.get("score"),
        scores.get("retro") if scores else None,
        scores.get("confidence") if scores else None,
    ):
        probability = _probability_from_score(value)
        if probability > 0.0:
            return probability
    return 0.0


def _step_cost(step: dict[str, Any]) -> float:
    probability = _step_probability(step)
    if probability > 0.0:
        return _negative_log_probability(probability)
    reactants = step.get("reactant_smiles") or []
    if not reactants:
        reactants = [step.get("main_reactant"), *(step.get("aux_reactants") or [])]
    return math.log1p(float(len([smi for smi in reactants if smi]) or 1))


def _nonstock_terminal_count(route: dict[str, Any]) -> int:
    count = 0
    for step in route.get("steps") or []:
        statuses = step.get("stock_status") or {}
        if isinstance(statuses, dict):
            count += sum(1 for value in statuses.values() if value is False)
    return count


def _probability_from_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 0.0:
        return 0.0
    return min(1.0, score)


def _negative_log_probability(probability: Any) -> float:
    probability = max(1e-6, _probability_from_score(probability))
    return -math.log(probability)


def _cost_to_value(cost: Any) -> float:
    try:
        value = float(cost)
    except (TypeError, ValueError):
        value = math.inf
    if not math.isfinite(value):
        return 0.0
    return 1.0 / (1.0 + max(0.0, value))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
