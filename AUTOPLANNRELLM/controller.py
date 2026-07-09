"""DeepSeek-backed route-tree selection controller."""
from __future__ import annotations

import math
import os
from typing import Any

from cascade_planner.route_tree.runtime import (
    RouteTreeEvaluation,
    heuristic_action_scores,
    heuristic_node_scores,
)
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState

from AUTOPLANNRELLM.deepseek_client import DeepSeekJSONClient


class DeepSeekSelectionController:
    """Route-tree controller that asks DeepSeek to rank leaves/actions.

    The controller wraps the existing AutoPlanner runtime. If DeepSeek is
    unavailable or returns invalid JSON, the wrapped runtime or heuristic scores
    remain in control and diagnostics record the fallback reason.
    """

    def __init__(
        self,
        *,
        fallback_runtime: Any | None = None,
        client: DeepSeekJSONClient | None = None,
        selection_weight: float | None = None,
    ) -> None:
        self.fallback_runtime = fallback_runtime
        self.client = client or DeepSeekJSONClient()
        self.selection_weight = float(
            selection_weight
            if selection_weight is not None
            else os.environ.get("AUTOPLANNRELLM_SELECTION_WEIGHT") or 0.75
        )

    def evaluate(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        stock_checker=None,
    ) -> RouteTreeEvaluation:
        fallback = self._fallback_action_eval(state, leaf, actions, stock_checker=stock_checker)
        if not actions:
            fallback.reason = _join_reason(fallback.reason, "autoplannrellm:no_actions")
            return fallback
        try:
            top_k = _selection_top_k("AUTOPLANNRELLM_ACTION_TOPK", n=len(actions), default=3)
            response = self.client.request_json(
                task="action_selection",
                system=_selection_system_prompt("action", top_k=top_k),
                user_payload=_action_selection_payload(state, leaf, actions, stock_checker=stock_checker, top_k=top_k),
                max_tokens=1400,
                temperature=float(os.environ.get("AUTOPLANNRELLM_ACTION_TEMPERATURE") or 0.05),
            )
            llm_scores = _scores_from_response(
                response,
                preference_key="action_preferences",
                selected_key="selected_action_indices",
                n=len(actions),
                top_k=top_k,
            )
            fallback.action_scores = _blend_scores(
                fallback.action_scores or heuristic_action_scores(leaf, actions, stock_checker=stock_checker),
                llm_scores,
                weight=self.selection_weight,
            )
            fallback.model_active = True
            fallback.reason = _join_reason(
                fallback.reason,
                (
                    "autoplannrellm:deepseek_action_selection:"
                    f"confidence={_safe_float(response.get('confidence'), 0.0):.3f}:topk={top_k}"
                ),
            )
            fallback.value_calibrated = bool(fallback.value_calibrated)
            return fallback
        except Exception as exc:
            fallback.reason = _join_reason(fallback.reason, f"autoplannrellm_action_fallback:{type(exc).__name__}")
            return fallback

    def score_open_leaves(
        self,
        state: RouteTreeState,
        leaves: list[str],
        *,
        stock_checker=None,
    ) -> RouteTreeEvaluation:
        fallback = self._fallback_leaf_eval(state, leaves, stock_checker=stock_checker)
        if not leaves:
            fallback.reason = _join_reason(fallback.reason, "autoplannrellm:no_leaves")
            return fallback
        try:
            top_k = _selection_top_k("AUTOPLANNRELLM_LEAF_TOPK", n=len(leaves), default=3)
            response = self.client.request_json(
                task="leaf_selection",
                system=_selection_system_prompt("leaf", top_k=top_k),
                user_payload=_leaf_selection_payload(state, leaves, stock_checker=stock_checker, top_k=top_k),
                max_tokens=1000,
                temperature=float(os.environ.get("AUTOPLANNRELLM_LEAF_TEMPERATURE") or 0.05),
            )
            llm_scores = _scores_from_response(
                response,
                preference_key="leaf_preferences",
                selected_key="selected_leaf_indices",
                n=len(leaves),
                top_k=top_k,
            )
            fallback.node_scores = _blend_scores(
                fallback.node_scores or heuristic_node_scores(leaves, stock_checker=stock_checker),
                llm_scores,
                weight=self.selection_weight,
            )
            fallback.model_active = True
            fallback.reason = _join_reason(
                fallback.reason,
                (
                    "autoplannrellm:deepseek_leaf_selection:"
                    f"confidence={_safe_float(response.get('confidence'), 0.0):.3f}:topk={top_k}"
                ),
            )
            return fallback
        except Exception as exc:
            fallback.reason = _join_reason(fallback.reason, f"autoplannrellm_leaf_fallback:{type(exc).__name__}")
            return fallback

    def _fallback_action_eval(self, state: RouteTreeState, leaf: str, actions: list[CandidateAction], *, stock_checker) -> RouteTreeEvaluation:
        if self.fallback_runtime is not None:
            try:
                return self.fallback_runtime.evaluate(state, leaf, actions, stock_checker=stock_checker)
            except Exception as exc:
                return RouteTreeEvaluation(
                    action_scores=heuristic_action_scores(leaf, actions, stock_checker=stock_checker),
                    model_active=False,
                    reason=f"fallback_runtime_error:{type(exc).__name__}",
                )
        return RouteTreeEvaluation(
            action_scores=heuristic_action_scores(leaf, actions, stock_checker=stock_checker),
            model_active=False,
            reason="heuristic_action_fallback",
        )

    def _fallback_leaf_eval(self, state: RouteTreeState, leaves: list[str], *, stock_checker) -> RouteTreeEvaluation:
        if self.fallback_runtime is not None:
            try:
                return self.fallback_runtime.score_open_leaves(state, leaves, stock_checker=stock_checker)
            except Exception as exc:
                return RouteTreeEvaluation(
                    action_scores=[],
                    node_scores=heuristic_node_scores(leaves, stock_checker=stock_checker),
                    model_active=False,
                    reason=f"fallback_runtime_error:{type(exc).__name__}",
                )
        return RouteTreeEvaluation(
            action_scores=[],
            node_scores=heuristic_node_scores(leaves, stock_checker=stock_checker),
            model_active=False,
            reason="heuristic_leaf_fallback",
        )


def _selection_system_prompt(kind: str, *, top_k: int) -> str:
    return (
        "You are a retrosynthesis search-control agent. Return only JSON. "
        f"Select 1 to {max(1, int(top_k))} options that should remain live for route-tree expansion, "
        "and rank only those selected options. Use the supplied fields only. Do not claim "
        "stock availability, yield, enzyme evidence, or conditions unless they "
        f"are present in the input. The task is {kind} selection, not route invention."
    )


def _action_selection_payload(
    state: RouteTreeState,
    leaf: str,
    actions: list[CandidateAction],
    *,
    stock_checker,
    top_k: int,
) -> dict[str, Any]:
    return {
        "target_smiles": state.target,
        "state_id": state.canonical_id,
        "depth": state.depth,
        "open_leaves": list(state.open_leaves),
        "selected_leaf": leaf,
        "selection_instructions": {
            "min_selected": 1,
            "max_selected": max(1, int(top_k)),
            "selected_options_are_branch_candidates": True,
            "do_not_score_unselected_options_low_just_to_force_a_single_path": True,
        },
        "required_schema": {
            "selected_action_indices": [0],
            "action_preferences": [{"index": 0, "score": 0.0, "rationale": ""}],
            "confidence": 0.0,
            "unsupported_claims": [],
        },
        "actions": [
            {
                "index": idx,
                "source": action.source,
                "rank": action.rank,
                "score": action.raw_score,
                "reaction_smiles": action.rxn_smiles,
                "reactants": list(action.reactants),
                "reaction_type": action.reaction_type,
                "ec": action.ec,
                "validity_flags": list(action.validity_flags),
                "terminal_fraction": _terminal_fraction(action, stock_checker),
            }
            for idx, action in enumerate(actions)
        ],
    }


def _leaf_selection_payload(
    state: RouteTreeState,
    leaves: list[str],
    *,
    stock_checker,
    top_k: int,
) -> dict[str, Any]:
    return {
        "target_smiles": state.target,
        "state_id": state.canonical_id,
        "depth": state.depth,
        "open_leaves": list(state.open_leaves),
        "selection_instructions": {
            "min_selected": 1,
            "max_selected": max(1, int(top_k)),
            "selected_options_are_branch_candidates": True,
            "do_not_score_unselected_options_low_just_to_force_a_single_path": True,
        },
        "required_schema": {
            "selected_leaf_indices": [0],
            "leaf_preferences": [{"index": 0, "score": 0.0, "rationale": ""}],
            "confidence": 0.0,
            "unsupported_claims": [],
        },
        "leaves": [
            {
                "index": idx,
                "smiles": leaf,
                "stock_hit": _stock_hit(leaf, stock_checker),
            }
            for idx, leaf in enumerate(leaves)
        ],
    }


def _terminal_fraction(action: CandidateAction, stock_checker) -> float:
    if not action.reactants:
        return 0.0
    hits = sum(1 for smi in action.reactants if _stock_hit(smi, stock_checker))
    return hits / max(1, len(action.reactants))


def _stock_hit(smiles: str, stock_checker) -> bool | None:
    if stock_checker is None:
        return None
    try:
        return bool(stock_checker(smiles))
    except Exception:
        return None


def _scores_from_response(
    response: dict[str, Any],
    *,
    preference_key: str,
    selected_key: str,
    n: int,
    top_k: int,
) -> list[float]:
    """Return additive LLM logits without penalizing non-selected options.

    The first full100 LLM run showed that scoring every non-favorite option low
    collapses search diversity. Treat the LLM as a branch proposer: it may boost
    1-k selected options, while unselected options keep their AutoPlanner scores.
    """
    n = max(0, int(n))
    if n <= 0:
        return []
    top_k = max(1, min(n, int(top_k or 1)))
    preference_scores = _preference_score_map(response.get(preference_key), n=n)
    selected = _selected_indices(response.get(selected_key), n=n, top_k=top_k)
    if not selected:
        selected = _selected_indices(response.get("selected_indices"), n=n, top_k=top_k)
    if not selected:
        selected = [
            idx
            for idx, _score in sorted(
                preference_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
        ]
    if not selected:
        selected = [0]
    min_selected_score = _env_float("AUTOPLANNRELLM_SELECTED_MIN_SCORE", 0.65)
    default_schedule = [0.85, 0.75, 0.65]
    scores = [0.0 for _ in range(n)]
    for rank, idx in enumerate(selected[:top_k]):
        fallback = default_schedule[min(rank, len(default_schedule) - 1)]
        score = max(float(min_selected_score), preference_scores.get(idx, fallback))
        scores[idx] = _logit(_bounded01(score))
    return scores


def _preference_score_map(rows: Any, *, n: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    if not isinstance(rows, list):
        return scores
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n:
            scores[idx] = _bounded01(_safe_float(row.get("score"), 0.5))
    return scores


def _selected_indices(value: Any, *, n: int, top_k: int) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in value:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n and idx not in seen:
            out.append(idx)
            seen.add(idx)
        if len(out) >= top_k:
            break
    return out


def _blend_scores(base_scores: list[float], llm_scores: list[float], *, weight: float) -> list[float]:
    out = []
    for idx, base in enumerate(base_scores):
        llm = llm_scores[idx] if idx < len(llm_scores) else 0.0
        out.append(float(base) + float(weight) * float(llm))
    return out


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, float(p)))
    return math.log(p / (1.0 - p))


def _bounded01(value: Any) -> float:
    return min(1.0, max(0.0, _safe_float(value, 0.5)))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _selection_top_k(name: str, *, n: int, default: int) -> int:
    return max(1, min(max(1, int(n or 1)), _env_int(name, default)))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return float(default)


def _join_reason(base: str, addition: str) -> str:
    return f"{base}+{addition}" if base else addition
