"""Neural-guided AND/OR route-tree search."""
from __future__ import annotations

import heapq
import itertools
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from rdkit import Chem

from cascade_planner.cascadeboard import RouteExplanation, RouteResult
from cascade_planner.cascadeboard.route_export import route_metrics
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.cascadeboard.skeleton_planner import RouteSkeleton
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.cascade_oracle import cascade_oracle_runtime_from_env
from cascade_planner.route_tree.runtime import RouteTreeEvaluation, RouteTreeRuntime, default_route_tree_runtime, heuristic_action_scores
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.route_tree.trace import RouteTreeTraceCollector
from cascade_planner.route_tree.verifier import RouteVerifier
from cascade_planner.vnext.schema import BOTTLENECK_LABELS


StockChecker = Callable[[str], bool]
_AUTO_CONTROLLER = object()
DEFAULT_SOURCE_RESERVE_ORDER = (
    "retrochimera",
    "chem_enzy_onestep",
    "enzyformer",
    "enzexpand",
    "v3_retrieval",
    "retrorules",
    "chemtemplates",
)


@dataclass
class RouteTreeStats:
    expansions: int = 0
    generated_actions: int = 0
    pruned_invalid: int = 0
    pruned_contract: int = 0
    pruned_cycle: int = 0
    pruned_transposition: int = 0
    solved_routes: int = 0
    dead_ends: int = 0
    max_queue_size: int = 0
    model_calls: int = 0
    model_active_calls: int = 0
    evaluation_cache_hits: int = 0
    proposal_calls: int = 0
    proposal_cache_hits: int = 0
    proposal_budget_total: int = 0
    stock_rescue_retries: int = 0
    stock_rescue_rejected: int = 0
    node_policy_calls: int = 0
    node_policy_active_calls: int = 0
    ccts_calls: int = 0
    ccts_active_calls: int = 0
    ccts_fallbacks: int = 0
    verifier_rejections: int = 0
    elapsed_s: float = 0.0
    search_stop_reason: str = ""
    soft_timeout_s: float | None = None
    hard_timeout_s: float | None = None
    expanded_leaf_count: int = 0
    skipped_leaf_count: int = 0
    proposal_source_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "expansions": self.expansions,
            "generated_actions": self.generated_actions,
            "pruned_invalid": self.pruned_invalid,
            "pruned_contract": self.pruned_contract,
            "pruned_cycle": self.pruned_cycle,
            "pruned_transposition": self.pruned_transposition,
            "solved_routes": self.solved_routes,
            "dead_ends": self.dead_ends,
            "max_queue_size": self.max_queue_size,
            "model_calls": self.model_calls,
            "model_active_calls": self.model_active_calls,
            "evaluation_cache_hits": self.evaluation_cache_hits,
            "proposal_calls": self.proposal_calls,
            "proposal_cache_hits": self.proposal_cache_hits,
            "proposal_budget_total": self.proposal_budget_total,
            "stock_rescue_retries": self.stock_rescue_retries,
            "stock_rescue_rejected": self.stock_rescue_rejected,
            "node_policy_calls": self.node_policy_calls,
            "node_policy_active_calls": self.node_policy_active_calls,
            "ccts_calls": self.ccts_calls,
            "ccts_active_calls": self.ccts_active_calls,
            "ccts_fallbacks": self.ccts_fallbacks,
            "verifier_rejections": self.verifier_rejections,
            "elapsed_s": round(float(self.elapsed_s or 0.0), 3),
            "search_stop_reason": self.search_stop_reason,
            "soft_timeout_s": self.soft_timeout_s,
            "hard_timeout_s": self.hard_timeout_s,
            "expanded_leaf_count": self.expanded_leaf_count,
            "skipped_leaf_count": self.skipped_leaf_count,
            "proposal_source_stats": _proposal_source_stats_payload(self.proposal_source_stats),
        }
        payload["route_tree_runtime_bottlenecks"] = _runtime_bottleneck_labels(payload)
        return payload


class NeuralGuidedAOSearch:
    def __init__(
        self,
        *,
        retro_engine: dict[str, Any] | None,
        stock_checker: StockChecker | None = None,
        max_depth: int = 6,
        branch_factor: int = 12,
        expansion_budget: int = 200,
        skeletons: list[RouteSkeleton] | None = None,
        constraints: dict[str, Any] | None = None,
        controller: RouteTreeRuntime | None | object = _AUTO_CONTROLLER,
        trace_collector: RouteTreeTraceCollector | None = None,
    ):
        self.retro_engine = retro_engine
        self.stock_checker = stock_checker
        self.max_depth = max(1, int(max_depth))
        self.branch_factor = max(1, int(branch_factor))
        self.expansion_budget = max(1, int(expansion_budget))
        self.skeletons = skeletons or []
        self.skeleton_contract = self.skeletons[0] if self.skeletons else None
        self.target_depth = _target_depth(self.skeletons)
        if self.target_depth:
            self.max_depth = min(self.max_depth, self.target_depth)
        self.constraints = constraints or {}
        self.controller = default_route_tree_runtime() if controller is _AUTO_CONTROLLER else controller
        self.proposals = RetroEngineProposalTool(retro_engine)
        self.verifier = RouteVerifier()
        self.trace_collector = trace_collector
        self.cascade_oracle = cascade_oracle_runtime_from_env()
        self.ccts_scorer = _ccts_runtime_from_env()
        self.stats = RouteTreeStats()
        self.stats.soft_timeout_s = _env_float_or_none("AUTOPLANNER_ROUTE_TREE_SOFT_TIMEOUT_S", 60.0)
        self.stats.hard_timeout_s = _env_float_or_none("AUTOPLANNER_ROUTE_TREE_HARD_TIMEOUT_S", 120.0)
        self.allowed_starting_materials = _terminal_materials(self.constraints)
        self.forbidden_intermediates = _canonical_set(
            self.constraints.get("forbidden_intermediate"),
            self.constraints.get("forbidden_intermediates"),
            self.constraints.get("exclude_intermediate"),
            self.constraints.get("exclude_intermediates"),
        )
        self._evaluation_cache: dict[tuple[str, str, tuple[str, ...]], RouteTreeEvaluation] = {}
        self._proposal_cache: dict[tuple[str, int, str, int, float | None, float | None, int], list[CandidateAction]] = {}
        self._best_transposition_score: dict[tuple[int, tuple[str, ...]], float] = {}
        self._search_started_at: float | None = None

    def search(self, target: str, *, n_results: int = 5) -> list[RouteResult]:
        self._search_started_at = time.monotonic()
        initial = RouteTreeState.initial(target, constraints=self.constraints)
        queue: list[tuple[float, int, RouteTreeState]] = []
        counter = itertools.count()
        heapq.heappush(queue, (-self._priority(initial), next(counter), initial))
        solved: list[RouteResult] = []
        fallback: list[RouteResult] = []
        seen: set[tuple[str, ...]] = set()
        fallback_seen: set[tuple[str, ...]] = set()
        result_pool_target = self._result_pool_target(n_results)

        while queue and self.stats.expansions < self.expansion_budget and len(solved) < result_pool_target:
            if self._hard_time_limit_reached():
                self.stats.search_stop_reason = "hard_timeout"
                break
            if self._soft_time_limit_reached() and (solved or fallback):
                self.stats.search_stop_reason = "soft_timeout_with_results"
                break
            self.stats.max_queue_size = max(self.stats.max_queue_size, len(queue))
            _, _, state = heapq.heappop(queue)
            if self._is_solved(state):
                result = self._state_to_result(state, search_status="stock_closed")
                sig = _state_signature(state)
                if sig not in seen:
                    seen.add(sig)
                    solved.append(result)
                    self.stats.solved_routes += 1
                continue
            if state.depth >= self.max_depth:
                self._add_fallback_result(fallback, fallback_seen, state, search_status="depth_limit")
                self.stats.dead_ends += 1
                continue

            expansions = self._expand_state(state)
            if not expansions:
                self._add_fallback_result(fallback, fallback_seen, state, search_status="dead_end")
                self.stats.dead_ends += 1
                continue
            self.stats.expansions += 1
            for child in expansions:
                if not self._should_queue_state(child):
                    continue
                heapq.heappush(queue, (-self._priority(child), next(counter), child))

        self.stats.elapsed_s = self._elapsed_s()
        if not self.stats.search_stop_reason:
            if len(solved) >= result_pool_target:
                self.stats.search_stop_reason = "result_limit"
            elif self.stats.expansions >= self.expansion_budget:
                self.stats.search_stop_reason = "expansion_budget"
            elif not queue:
                self.stats.search_stop_reason = "queue_exhausted"
            else:
                self.stats.search_stop_reason = "stopped"
        results = _dedupe_results([*solved, *fallback])
        if _env_truthy("AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK"):
            results.sort(key=lambda result: _route_result_sort_key(result, self.stock_checker), reverse=True)
        else:
            results.sort(key=lambda result: _route_result_sort_key(result, self.stock_checker), reverse=True)
        outcome = {
            "solved_routes": len(solved),
            "fallback_routes": len(fallback),
            "requested_results": int(n_results),
            "route_tree_result_pool_target": int(result_pool_target),
            "search_status": "solved" if solved else "partial" if fallback else "failed",
            **self.stats.to_dict(),
        }
        if self.trace_collector is not None:
            self.trace_collector.annotate_outcome(outcome)
        self._attach_final_diagnostics(results, outcome)
        return _select_return_results(
            results,
            fallback=fallback,
            n_results=n_results,
            stock_checker=self.stock_checker,
        )

    def _expand_state(self, state: RouteTreeState) -> list[RouteTreeState]:
        scored_children: list[tuple[float, RouteTreeState, str, CandidateAction, list[CandidateAction], RouteTreeEvaluation]] = []
        candidate_leaves = self._candidate_leaves(state)
        leaves_to_expand = candidate_leaves[: self._max_leaves_per_expansion(state)]
        self.stats.expanded_leaf_count += len(leaves_to_expand)
        self.stats.skipped_leaf_count += max(0, len(candidate_leaves) - len(leaves_to_expand))
        expansion_diagnostics: list[dict[str, Any]] = []
        filter_diagnostics: list[dict[str, Any]] = []
        selection_diagnostics_by_leaf: dict[str, tuple[list[float], list[dict[str, Any]]]] = {}
        for leaf in leaves_to_expand:
            if self._hard_time_limit_reached():
                self.stats.search_stop_reason = "hard_timeout"
                break
            context = self._proposal_context(state)
            self._annotate_leaf_context(state, leaf, context)
            proposal_budget = self._proposal_budget_for_leaf(state, leaf, context)
            self.stats.proposal_budget_total += proposal_budget
            raw_actions = self._propose_actions(leaf, context, top_k=proposal_budget)
            proposal_diag = self._last_leaf_proposal_diagnostics(leaf, proposal_budget)
            actions, contract_pruned, invalid_pruned = self._filter_actions(state, leaf, raw_actions, context)
            if not actions:
                fallback_budget = self._fallback_proposal_budget_for_leaf(state, leaf, context, base_budget=proposal_budget)
                retry_reason = "empty_actions" if self._stock_rescue_enabled(state, leaf) else ""
                if _env_truthy("AUTOPLANNER_ROUTE_TREE_EMPTY_ACTION_RETRY"):
                    fallback_budget = max(
                        fallback_budget,
                        self._empty_action_retry_budget(proposal_budget),
                    )
                    retry_reason = "empty_actions"
                if fallback_budget > proposal_budget:
                    self.stats.proposal_budget_total += fallback_budget
                    retry_context = self._stock_rescue_context(context, reason=retry_reason) if retry_reason else context
                    retry_raw_actions = self._propose_actions(leaf, retry_context, top_k=fallback_budget)
                    raw_actions = _dedupe_candidate_actions([*raw_actions, *retry_raw_actions])
                    proposal_diag = self._last_leaf_proposal_diagnostics(leaf, fallback_budget)
                    actions, contract_pruned, invalid_pruned = self._filter_actions(state, leaf, raw_actions, context)
                    proposal_diag["fallback_budget"] = int(fallback_budget)
                    proposal_diag["stock_rescue_retry_reason"] = retry_reason
                    if retry_reason:
                        self.stats.stock_rescue_retries += 1
            elif self._needs_stock_rescue_retry(state, leaf, actions):
                fallback_budget = self._fallback_proposal_budget_for_leaf(state, leaf, context, base_budget=proposal_budget)
                if fallback_budget > proposal_budget:
                    self.stats.proposal_budget_total += fallback_budget
                    original_raw_actions = list(raw_actions)
                    original_actions = list(actions)
                    original_contract_pruned = contract_pruned
                    original_invalid_pruned = invalid_pruned
                    retry_context = self._stock_rescue_context(context, reason="no_stock_closing_action")
                    retry_raw_actions = self._propose_actions(leaf, retry_context, top_k=fallback_budget)
                    raw_actions = _dedupe_candidate_actions([*raw_actions, *retry_raw_actions])
                    proposal_diag = self._last_leaf_proposal_diagnostics(leaf, fallback_budget)
                    actions, contract_pruned, invalid_pruned = self._filter_actions(state, leaf, raw_actions, context)
                    proposal_diag["fallback_budget"] = int(fallback_budget)
                    proposal_diag["stock_rescue_retry_reason"] = "no_stock_closing_action"
                    self.stats.stock_rescue_retries += 1
                    if self._reject_stock_rescue_retry(state, original_actions, actions):
                        raw_actions = original_raw_actions
                        actions = original_actions
                        contract_pruned = original_contract_pruned
                        invalid_pruned = original_invalid_pruned
                        proposal_diag["stock_rescue_retry_rejected_reason"] = "no_stock_closure_gain"
                        self.stats.stock_rescue_rejected += 1
            self.stats.pruned_contract += contract_pruned
            pruned = invalid_pruned
            if pruned > 0:
                self.stats.pruned_invalid += pruned
            proposal_diag.update(
                {
                    "raw_actions": len(raw_actions),
                    "contract_filtered": int(contract_pruned),
                    "invalid_filtered": int(invalid_pruned),
                    "final_actions": len(actions),
                }
            )
            expansion_diagnostics.append(proposal_diag)
            filter_diagnostics.append(
                {
                    "leaf": leaf,
                    "raw_actions": len(raw_actions),
                    "contract_filtered": int(contract_pruned),
                    "invalid_filtered": int(invalid_pruned),
                    "final_actions": len(actions),
                }
            )
            if not actions:
                continue
            eval_result = self._evaluate_actions(state, leaf, actions)
            state_with_pool = state.with_candidate_pool(leaf, actions)
            selection_rows = self._selection_score_rows(state, leaf, actions, eval_result)
            selection_rows = self._apply_ccts_selection_scores(state, leaf, actions, selection_rows)
            leaf_selection_scores = [float(row["total"]) for row in selection_rows]
            leaf_selection_breakdown = list(selection_rows)
            for action, score_components in zip(actions, selection_rows):
                next_open = tuple(score_components.get("next_open") or self._next_open_leaves(state, leaf, action))
                delta = float(score_components["total"])
                child = state_with_pool.advance(
                    leaf=leaf,
                    action=action,
                    next_open_leaves=next_open,
                    score_delta=delta,
                )
                child.search_metadata.update(
                    _extend_search_diagnostics(
                        state,
                        leaf=leaf,
                        action=action,
                        eval_result=eval_result,
                        proposal_budget=proposal_budget,
                        candidate_pool_size=len(actions),
                        proposal_diagnostics=expansion_diagnostics,
                        filter_diagnostics=filter_diagnostics,
                        selection_score=score_components,
                    )
                )
                scored_children.append((delta, child, leaf, action, actions, eval_result))
                self.stats.generated_actions += 1
            selection_diagnostics_by_leaf[canonical_smiles(leaf) or leaf] = (
                leaf_selection_scores,
                leaf_selection_breakdown,
            )

        selected = self._select_scored_children(scored_children, state=state)
        if self.trace_collector is not None:
            selected_keys = {child.canonical_id for _, child, *_ in selected}
            for _, child, leaf, action, actions, eval_result in selected[:1]:
                selection_scores, selection_breakdown = selection_diagnostics_by_leaf.get(
                    canonical_smiles(leaf) or leaf,
                    ([], []),
                )
                self.trace_collector.record_expansion(
                    state=state,
                    leaf=leaf,
                    actions=actions,
                    action_scores=eval_result.action_scores,
                    selection_scores=selection_scores,
                    selection_score_breakdown=selection_breakdown,
                    model_active=eval_result.model_active,
                    model_reason=eval_result.reason,
                    selected_action=action,
                    next_state=child if child.canonical_id in selected_keys else None,
                    proposal_diagnostics=expansion_diagnostics,
                    filter_diagnostics=filter_diagnostics,
                    stock_checker=self.stock_checker,
                )
        return [child for _, child, *_ in selected]

    def _candidate_leaves(self, state: RouteTreeState) -> list[str]:
        leaves = []
        for leaf in state.open_leaves:
            can = canonical_smiles(leaf)
            if not can:
                continue
            if can in state.expanded:
                self.stats.pruned_cycle += 1
                continue
            if can in self.forbidden_intermediates:
                self.stats.pruned_invalid += 1
                continue
            if not self._is_terminal(leaf, state=state, depth=state.depth):
                leaves.append(leaf)
        if self.controller is not None and leaves:
            eval_result = self.controller.score_open_leaves(state, leaves, stock_checker=self.stock_checker)
            self.stats.node_policy_calls += 1
            self.stats.node_policy_active_calls += int(eval_result.model_active)
            if eval_result.node_scores:
                scored = self._rank_leaves_with_reserve(state, leaves, list(eval_result.node_scores), eval_result.model_active)
                leaves = [leaf for leaf, _score in scored]
                state.search_metadata["latest_node_policy"] = {
                    "leaves": leaves,
                    "scores": [score for _leaf, score in scored],
                    "model_active": eval_result.model_active,
                    "reason": eval_result.reason,
                    "route_value": eval_result.route_value,
                    "value_calibrated": eval_result.value_calibrated,
                    "source_budgets": dict(eval_result.source_budgets),
                }
                state.search_metadata["leaf_scores"] = [
                    *state.search_metadata.get("leaf_scores", []),
                    {
                        "state_id": state.canonical_id,
                        "depth": state.depth,
                        "leaves": [{"leaf": leaf, "score": score} for leaf, score in scored],
                        "model_active": bool(eval_result.model_active),
                        "reason": eval_result.reason,
                    },
                ]
                return leaves
        scored = self._rank_leaves_with_reserve(
            state,
            leaves,
            [_heuristic_leaf_score(state, leaf, stock_checker=self.stock_checker) for leaf in leaves],
            False,
        )
        leaves = [leaf for leaf, _score in scored]
        state.search_metadata["leaf_scores"] = [
            *state.search_metadata.get("leaf_scores", []),
            {
                "state_id": state.canonical_id,
                "depth": state.depth,
                "leaves": [{"leaf": leaf, "score": score} for leaf, score in scored],
                "model_active": False,
                "reason": "coverage_heuristic",
            },
        ]
        return leaves

    def _rank_leaves_with_reserve(
        self,
        state: RouteTreeState,
        leaves: list[str],
        base_scores: list[float],
        model_active: bool,
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for leaf, score in zip(leaves, base_scores):
            scored.append((leaf, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        if len(scored) <= 2 or not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_LEAF_RESERVE", True):
            return scored
        reserve = min(_env_int("AUTOPLANNER_ROUTE_TREE_DIVERSE_LEAF_RESERVE", 1), max(0, len(scored) - 1))
        if reserve <= 0:
            return scored
        top = scored[0]
        tail = sorted(scored[1:], key=lambda item: (_heavy_atoms(item[0]), item[1]), reverse=True)
        reserve_items = tail[:reserve]
        reserve_keys = {canonical_smiles(leaf) or leaf for leaf, _score in reserve_items}
        remainder = [
            item
            for item in scored[1:]
            if (canonical_smiles(item[0]) or item[0]) not in reserve_keys
        ]
        return [top, *reserve_items, *remainder]

    def _proposal_context(self, state: RouteTreeState) -> ProposalContext:
        ec1 = 0
        reaction_type = ""
        T = None
        pH = None
        context_index = state.depth
        for skeleton in self.skeletons:
            context_index = _skeleton_context_index(skeleton, state.depth)
            if 0 <= context_index < len(skeleton.ec1s):
                ec1 = int(skeleton.ec1s[context_index] or 0)
            if 0 <= context_index < len(skeleton.types):
                reaction_type = str(skeleton.types[context_index] or "")
            if 0 <= context_index < len(skeleton.Ts):
                T = float(skeleton.Ts[context_index])
            if 0 <= context_index < len(skeleton.pHs):
                pH = float(skeleton.pHs[context_index])
            if ec1 or reaction_type or T is not None or pH is not None:
                break
        enzymatic_only_route = _heavy_atoms(state.target) >= 30 or _oxygen_rich_molecule(state.target)
        carbohydrate_like_route = _carbohydrate_like_molecule(state.target)
        return ProposalContext(
            depth=state.depth,
            ec1=ec1,
            reaction_type=reaction_type,
            T=T,
            pH=pH,
            constraints=self.constraints,
            route_metadata={
                "state_id": state.canonical_id,
                "enzymatic_only_route": enzymatic_only_route,
                "carbohydrate_like_route": carbohydrate_like_route,
                "skeleton_context_index": context_index,
                "skeleton_context_reversed": _reverse_skeleton_context_enabled(),
            },
        )

    def _annotate_leaf_context(self, state: RouteTreeState, leaf: str, context: ProposalContext) -> None:
        leaf_key = canonical_smiles(leaf) or leaf
        route_metadata = dict(context.route_metadata or {})
        previous_leaf_rows = [
            item or {}
            for item in (state.search_metadata.get("proposal_diagnostics") or [])
            if (canonical_smiles((item or {}).get("leaf")) or (item or {}).get("leaf")) == leaf_key
        ]
        previous_low_yield = any(
            int(item.get("raw_actions") or 0) > 0 and int(item.get("final_actions") or 0) <= 1
            for item in previous_leaf_rows
        )
        route_metadata.update(
            {
                "remaining_depth": max(0, self.max_depth - state.depth),
                "open_leaf_count": len(state.open_leaves),
                "nonstock_leaf_count": sum(
                    1 for smi in state.open_leaves if not _stock_check(smi, self.stock_checker)
                ),
                "leaf_stock_hit": _stock_check(leaf, self.stock_checker),
                "leaf_parent_adjacent": _leaf_has_parent_reaction(state, leaf),
                "leaf_low_yield": previous_low_yield,
                "leaf_expanded_history": leaf_key in state.expanded,
                "leaf_heavy_atoms": _heavy_atoms(leaf),
                "target_heavy_atoms": _heavy_atoms(state.target),
                "state_depth": state.depth,
            }
        )
        context.route_metadata = route_metadata

    def _contextualize_action(self, action: CandidateAction, context: ProposalContext) -> CandidateAction:
        metadata = dict(action.metadata)
        metadata["route_tree_context"] = {
            "depth": context.depth,
            "reaction_type": context.reaction_type,
            "ec1": context.ec1,
            "T": context.T,
            "pH": context.pH,
        }
        ec = action.ec
        if not ec and context.ec1 and _is_enzymatic_action(action):
            ec = f"{context.ec1}.x"
        return replace(
            action,
            reaction_type=action.reaction_type or context.reaction_type,
            ec=ec,
            T=action.T if action.T is not None else context.T,
            pH=action.pH if action.pH is not None else context.pH,
            metadata=metadata,
        )

    def _evaluate_actions(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
    ) -> RouteTreeEvaluation:
        cache_key = (
            state.canonical_id,
            canonical_smiles(leaf) or leaf,
            tuple(action.canonical_key for action in actions),
        )
        cached = self._evaluation_cache.get(cache_key)
        if cached is not None:
            self.stats.evaluation_cache_hits += 1
            return cached
        if self.controller is None:
            result = RouteTreeEvaluation(
                action_scores=[],
                model_active=False,
                reason="no_controller",
            )
        else:
            result = self.controller.evaluate(state, leaf, actions, stock_checker=self.stock_checker)
        self.stats.model_calls += 1
        self.stats.model_active_calls += int(result.model_active)
        self._evaluation_cache[cache_key] = result
        return result

    def _propose_actions(self, leaf: str, context: ProposalContext, *, top_k: int) -> list[CandidateAction]:
        cache_key = (
            canonical_smiles(leaf) or leaf,
            int(context.depth or 0),
            str(context.reaction_type or ""),
            int(context.ec1 or 0),
            context.T,
            context.pH,
            int(top_k),
        )
        cached = self._proposal_cache.get(cache_key)
        if cached is not None:
            self.stats.proposal_cache_hits += 1
            return cached[:top_k]
        self.stats.proposal_calls += 1
        actions = self.proposals.propose(leaf, context, top_k=top_k)
        self._record_proposal_diagnostics(getattr(self.proposals, "last_diagnostics", {}) or {})
        self._proposal_cache[cache_key] = list(actions)
        return actions

    def _last_leaf_proposal_diagnostics(self, leaf: str, proposal_budget: int) -> dict[str, Any]:
        diagnostics = dict(getattr(self.proposals, "last_diagnostics", {}) or {})
        return {
            "leaf": leaf,
            "proposal_budget": int(proposal_budget),
            "top_k": int(diagnostics.get("top_k") or proposal_budget),
            "ordered_sources": list(diagnostics.get("ordered_sources") or []),
            "allocation": dict(diagnostics.get("allocation") or {}),
            "sources": {str(k): dict(v or {}) for k, v in (diagnostics.get("sources") or {}).items()},
            "dedupe": dict(diagnostics.get("dedupe") or {}),
            "empty_reason": diagnostics.get("empty_reason") or "",
        }

    def _record_proposal_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        for source, row in (diagnostics.get("sources") or {}).items():
            stats = self.stats.proposal_source_stats.setdefault(
                str(source),
                {
                    "calls": 0,
                    "allocated_budget": 0,
                    "requested_k_total": 0,
                    "kept_k_total": 0,
                    "raw_returned": 0,
                    "ranker_kept": 0,
                    "ranker_dropped": 0,
                    "kept_returned": 0,
                    "dedupe_dropped": 0,
                    "invalid_dropped": 0,
                    "final_returned": 0,
                    "latency_ms_total": 0.0,
                    "latency_ms_max": 0.0,
                },
            )
            stats["calls"] = int(stats.get("calls") or 0) + int(row.get("calls") or 0)
            stats["allocated_budget"] = int(stats.get("allocated_budget") or 0) + int(row.get("allocated_budget") or 0)
            stats["requested_k_total"] = int(stats.get("requested_k_total") or 0) + int(row.get("requested_k_total") or 0)
            stats["kept_k_total"] = int(stats.get("kept_k_total") or 0) + int(row.get("kept_k_total") or 0)
            stats["raw_returned"] = int(stats.get("raw_returned") or 0) + int(row.get("raw_returned") or 0)
            stats["ranker_kept"] = int(stats.get("ranker_kept") or 0) + int(row.get("ranker_kept") or 0)
            stats["ranker_dropped"] = int(stats.get("ranker_dropped") or 0) + int(row.get("ranker_dropped") or 0)
            stats["kept_returned"] = int(stats.get("kept_returned") or 0) + int(row.get("kept_returned") or 0)
            stats["dedupe_dropped"] = int(stats.get("dedupe_dropped") or 0) + int(row.get("dedupe_dropped") or 0)
            stats["invalid_dropped"] = int(stats.get("invalid_dropped") or 0) + int(row.get("invalid_dropped") or 0)
            stats["final_returned"] = int(stats.get("final_returned") or 0) + int(row.get("final_returned") or 0)
            stats["latency_ms_total"] = round(float(stats.get("latency_ms_total") or 0.0) + float(row.get("latency_ms_total") or 0.0), 3)
            stats["latency_ms_max"] = round(max(float(stats.get("latency_ms_max") or 0.0), float(row.get("latency_ms_max") or 0.0)), 3)

    def _filter_actions(
        self,
        state: RouteTreeState,
        leaf: str,
        raw_actions: list[CandidateAction],
        context: ProposalContext,
    ) -> tuple[list[CandidateAction], int, int]:
        actions: list[CandidateAction] = []
        contract_pruned = 0
        invalid_pruned = 0
        for raw_action in raw_actions:
            action = self._contextualize_action(raw_action, context)
            verification = self.verifier.verify_action(
                state=state,
                leaf=leaf,
                action=action,
                context=context,
                stock_checker=self.stock_checker,
            )
            if not verification.accepted:
                self.stats.verifier_rejections += 1
                if any(reason in verification.reasons for reason in {"skeleton_type_mismatch", "ec_mismatch", "condition_temperature_mismatch", "condition_pH_mismatch"}):
                    contract_pruned += 1
                else:
                    invalid_pruned += 1
                metadata = dict(action.metadata)
                metadata["route_verifier"] = verification.to_dict()
                action.metadata = metadata
                continue
            if self._action_allowed(state, leaf, action):
                actions.append(action)
            else:
                invalid_pruned += 1
        return actions, contract_pruned, invalid_pruned

    def _proposal_budget_for_leaf(
        self,
        state: RouteTreeState,
        leaf: str,
        context: ProposalContext,
    ) -> int:
        if _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_ADAPTIVE_BUDGETS", True):
            if state.depth <= 0:
                budget = _env_int("AUTOPLANNER_ROUTE_TREE_ROOT_PROPOSAL_BUDGET", self.branch_factor)
            elif len([smi for smi in state.open_leaves if not self._is_terminal(smi, state=state, depth=state.depth)]) > 1:
                budget = _env_int(
                    "AUTOPLANNER_ROUTE_TREE_MULTI_LEAF_PROPOSAL_BUDGET",
                    max(4, int(round(self.branch_factor * 0.75))),
                )
            else:
                budget = _env_int("AUTOPLANNER_ROUTE_TREE_DOWNSTREAM_PROPOSAL_BUDGET", self.branch_factor)
        else:
            budget = self.branch_factor
        remaining_depth = max(0, self.max_depth - state.depth)
        if remaining_depth <= 1:
            budget = max(2, self.branch_factor // 2)
        if self._is_terminal(leaf, state=state, depth=state.depth):
            return 0
        return max(1, budget)

    def _fallback_proposal_budget_for_leaf(
        self,
        state: RouteTreeState,
        leaf: str,
        context: ProposalContext,
        *,
        base_budget: int,
    ) -> int:
        if self._is_terminal(leaf, state=state, depth=state.depth):
            return 0
        remaining_depth = max(0, self.max_depth - state.depth)
        if remaining_depth <= 1:
            if self._stock_rescue_enabled(state, leaf):
                multiplier = max(1.0, _env_float("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_MULTIPLIER", 2.0))
                cap = max(base_budget, _env_int("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_CAP", 24))
                return min(cap, max(base_budget + 1, int(round(base_budget * multiplier)), self.branch_factor))
            return base_budget
        if context.reaction_type or context.ec1:
            return min(_env_int("AUTOPLANNER_ROUTE_TREE_FALLBACK_PROPOSAL_BUDGET_CAP", 16), max(base_budget, self.branch_factor * 2))
        return base_budget

    def _empty_action_retry_budget(self, base_budget: int) -> int:
        multiplier = max(1.0, _env_float("AUTOPLANNER_ROUTE_TREE_EMPTY_ACTION_RETRY_BUDGET_MULTIPLIER", 2.0))
        cap = max(base_budget + 1, _env_int("AUTOPLANNER_ROUTE_TREE_EMPTY_ACTION_RETRY_BUDGET_CAP", 24))
        return min(cap, max(base_budget + 1, int(round(base_budget * multiplier)), self.branch_factor * 2))

    def _stock_rescue_enabled(self, state: RouteTreeState, leaf: str) -> bool:
        if not _env_truthy("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE"):
            return False
        max_retries = _env_int("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MAX_RETRIES", 0)
        if max_retries > 0 and self.stats.stock_rescue_retries >= max_retries:
            return False
        if self.stock_checker is None:
            return False
        if self._is_terminal(leaf, state=state, depth=state.depth):
            return False
        remaining_depth = max(0, self.max_depth - state.depth)
        threshold = max(0, _env_int("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REMAINING_DEPTH", 1))
        return remaining_depth <= threshold

    def _needs_stock_rescue_retry(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
    ) -> bool:
        if not actions or not self._stock_rescue_enabled(state, leaf):
            return False
        child_depth = state.depth + 1
        best_terminal_fraction = max(
            (
                _terminal_fraction(
                    action,
                    lambda smi: self._is_terminal(smi, state=state, depth=child_depth),
                )
                for action in actions
            ),
            default=0.0,
        )
        if best_terminal_fraction >= 1.0:
            return False
        min_actions = max(0, _env_int("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MIN_ACTIONS", 2))
        return len(actions) <= min_actions or best_terminal_fraction <= 0.0

    def _stock_rescue_context(self, context: ProposalContext, *, reason: str) -> ProposalContext:
        route_metadata = dict(context.route_metadata or {})
        route_metadata["stock_rescue_retry"] = True
        route_metadata["stock_rescue_retry_reason"] = reason
        return ProposalContext(
            depth=context.depth,
            ec1=context.ec1,
            reaction_type=context.reaction_type,
            T=context.T,
            pH=context.pH,
            objective=context.objective,
            constraints=dict(context.constraints or {}),
            route_metadata=route_metadata,
        )

    def _reject_stock_rescue_retry(
        self,
        state: RouteTreeState,
        original_actions: list[CandidateAction],
        retry_actions: list[CandidateAction],
    ) -> bool:
        if not _env_truthy("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REQUIRE_STOCK_GAIN"):
            return False
        if not original_actions:
            return False
        before = self._best_terminal_fraction(state, original_actions)
        after = self._best_terminal_fraction(state, retry_actions)
        accept_fraction = min(
            1.0,
            max(0.0, _env_float("AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_ACCEPT_TERMINAL_FRACTION", 1.0)),
        )
        return not (after >= accept_fraction and after > before)

    def _best_terminal_fraction(self, state: RouteTreeState, actions: list[CandidateAction]) -> float:
        child_depth = state.depth + 1
        return max(
            (
                _terminal_fraction(
                    action,
                    lambda smi: self._is_terminal(smi, state=state, depth=child_depth),
                )
                for action in actions
            ),
            default=0.0,
        )


    def _violates_contract(self, action: CandidateAction, context: ProposalContext) -> bool:
        if context.reaction_type and action.reaction_type:
            if not _reaction_type_compatible(action.reaction_type, context.reaction_type):
                return True
        if context.ec1 and action.ec:
            if _ec1(action.ec) and _ec1(action.ec) != str(context.ec1):
                return True
        if context.T is not None and action.T is not None and abs(float(action.T) - float(context.T)) > 40.0:
            return True
        if context.pH is not None and action.pH is not None and abs(float(action.pH) - float(context.pH)) > 4.0:
            return True
        return False

    def _action_allowed(self, state: RouteTreeState, leaf: str, action: CandidateAction) -> bool:
        if "product_mismatch" in action.validity_flags or "self_loop" in action.validity_flags:
            return False
        reactants = [canonical_smiles(smi) for smi in action.reactants]
        reactants = [smi for smi in reactants if smi]
        if not reactants:
            return False
        if set(reactants) & self.forbidden_intermediates:
            return False
        if any(smi in state.expanded for smi in reactants):
            self.stats.pruned_cycle += 1
            return False
        if not _candidate_atom_balance_ok(action, leaf):
            return False
        return True

    def _should_queue_state(self, state: RouteTreeState) -> bool:
        if not state.open_leaves and _env_truthy("AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES"):
            return True
        key = _transposition_key(state)
        best = self._best_transposition_score.get(key)
        if best is not None and best >= state.score:
            self.stats.pruned_transposition += 1
            return False
        self._best_transposition_score[key] = state.score
        return True

    def _next_open_leaves(self, state: RouteTreeState, leaf: str, action: CandidateAction) -> tuple[str, ...]:
        remaining = list(state.open_leaves)
        try:
            remaining.remove(leaf)
        except ValueError:
            pass
        child_depth = state.depth + 1
        remaining.extend(
            smi
            for smi in action.reactants
            if not self._is_terminal(smi, state=state, depth=child_depth)
        )
        return tuple(_dedupe_smiles(remaining))

    def _max_leaves_per_expansion(self, state: RouteTreeState) -> int:
        if not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_SINGLE_NODE_EXPANSION", True):
            return max(1, len(state.open_leaves))
        reserve = _env_int("AUTOPLANNER_ROUTE_TREE_DIVERSE_LEAF_RESERVE", 1)
        default = 1 if state.depth <= 0 else 1 + max(0, reserve)
        return max(1, _env_int("AUTOPLANNER_ROUTE_TREE_MAX_LEAVES_PER_EXPANSION", default))

    def _score_delta(
        self,
        state: RouteTreeState,
        leaf: str,
        action: CandidateAction,
        model_score: float,
        eval_result: RouteTreeEvaluation,
        *,
        next_open: tuple[str, ...],
    ) -> float:
        return float(self._score_delta_components(state, leaf, action, model_score, eval_result, next_open=next_open)["total"])

    def _score_delta_components(
        self,
        state: RouteTreeState,
        leaf: str,
        action: CandidateAction,
        model_score: float,
        eval_result: RouteTreeEvaluation,
        *,
        next_open: tuple[str, ...],
    ) -> dict[str, Any]:
        return self._selection_score_row(
            state,
            leaf,
            action,
            eval_result,
            policy_probability=0.0,
            model_score=model_score,
            next_open=next_open,
        )

    def _selection_score_rows(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        eval_result: RouteTreeEvaluation,
    ) -> list[dict[str, Any]]:
        policy_probabilities = (
            _softmax_probabilities(eval_result.action_scores, n=len(actions))
            if len(eval_result.action_scores) == len(actions)
            else [0.0 for _ in actions]
        )
        rows: list[dict[str, Any]] = []
        for idx, action in enumerate(actions):
            model_score = float(eval_result.action_scores[idx]) if idx < len(eval_result.action_scores) else 0.0
            rows.append(
                self._selection_score_row(
                    state,
                    leaf,
                    action,
                    eval_result,
                    policy_probability=policy_probabilities[idx] if idx < len(policy_probabilities) else 0.0,
                    model_score=model_score,
                    next_open=self._next_open_leaves(state, leaf, action),
                )
            )
        return rows

    def _apply_ccts_selection_scores(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.ccts_scorer is None or not actions or not rows:
            return rows
        self.stats.ccts_calls += 1
        try:
            result = self.ccts_scorer.score_actions(state, leaf, actions, max_depth=self.max_depth)
        except Exception as exc:
            self.stats.ccts_fallbacks += 1
            return [
                {
                    **row,
                    "ccts_active": False,
                    "ccts_reason": f"runtime_error:{type(exc).__name__}",
                }
                for row in rows
            ]
        if not result.active or len(result.normalized_scores) != len(rows):
            self.stats.ccts_fallbacks += 1
            reason = getattr(result, "reason", "unavailable")
            return [{**row, "ccts_active": False, "ccts_reason": reason} for row in rows]
        self.stats.ccts_active_calls += 1
        weight = _env_float("AUTOPLANNER_ROUTE_TREE_CCTS_WEIGHT", 0.35)
        out: list[dict[str, Any]] = []
        detail_rows = list(getattr(result, "rows", []) or [])
        ccts_tag = "ccts_v3" if str(getattr(result, "reason", "")).startswith("ccts_v3") else "ccts_v0"
        for idx, row in enumerate(rows):
            base_total = float(row.get("total") or 0.0)
            raw_score = float(result.scores[idx]) if idx < len(result.scores) else 0.0
            z_score = float(result.normalized_scores[idx])
            bonus = float(weight) * z_score
            next_total = base_total + bonus
            updated = dict(row)
            updated.update(
                {
                    "cost_model": f"{row.get('cost_model') or 'reaction_cost_and_or.v1'}+{ccts_tag}",
                    "total_before_ccts": round(base_total, 6),
                    "ccts_active": True,
                    "ccts_reason": result.reason,
                    "ccts_selected_score": result.selected_score,
                    "ccts_score": round(raw_score, 6),
                    "ccts_score_z": round(z_score, 6),
                    "ccts_weight": round(float(weight), 6),
                    "ccts_bonus": round(bonus, 6),
                    "ccts_adjusted_total_cost": round(max(0.0, -next_total), 6),
                    "ccts_rank": (detail_rows[idx] or {}).get("selected_rank") if idx < len(detail_rows) else None,
                    "ccts_chem_rank": (detail_rows[idx] or {}).get("chem_rank") if idx < len(detail_rows) else None,
                    "total": round(next_total, 6),
                }
            )
            out.append(updated)
        return out

    def _selection_score_row(
        self,
        state: RouteTreeState,
        leaf: str,
        action: CandidateAction,
        eval_result: RouteTreeEvaluation,
        *,
        policy_probability: float,
        model_score: float,
        next_open: tuple[str, ...],
    ) -> dict[str, Any]:
        child_depth = state.depth + 1
        terminal_fraction = _terminal_fraction(action, lambda smi: self._is_terminal(smi, state=state, depth=child_depth))
        oracle_match = self._oracle_match_for_action(state, leaf, action)
        proposal_probability = _probability_from_score(action.raw_score)
        oracle_probability = _oracle_probability(oracle_match)
        controller_probability = _controller_value_objective(eval_result)
        base_probability = max(
            proposal_probability,
            _bounded_probability(policy_probability),
            oracle_probability,
            controller_probability or 0.0,
        )
        reaction_cost = _negative_log_probability(base_probability)
        child_value_cost = sum(
            self._leaf_synthesis_cost(smi, state=state, depth=child_depth)
            for smi in next_open
        )
        feasibility_cost, feasibility_diagnostics = self._action_feasibility_cost(leaf, action)
        total_cost = reaction_cost + child_value_cost + feasibility_cost
        selection_score = -total_cost
        return {
            "action_key": action.canonical_key,
            "source": action.source,
            "rank": action.rank,
            "model_action_score": float(model_score),
            "cost_model": "reaction_cost_and_or.v1",
            "proposal_probability": round(float(proposal_probability), 6),
            "policy_probability": round(float(_bounded_probability(policy_probability)), 6),
            "oracle_probability": round(float(oracle_probability), 6),
            "controller_probability": round(float(controller_probability), 6) if controller_probability is not None else None,
            "base_probability": round(float(base_probability), 6),
            "reaction_cost": round(float(reaction_cost), 6),
            "child_value_cost": round(float(child_value_cost), 6),
            "feasibility_cost": round(float(feasibility_cost), 6),
            "total_cost": round(float(total_cost), 6),
            "terminal_fraction": round(float(terminal_fraction), 6),
            "feasibility_diagnostics": feasibility_diagnostics,
            "oracle_match": oracle_match.to_dict() if oracle_match is not None else None,
            "next_open": list(next_open),
            "total": round(float(selection_score), 6),
        }

    def _action_feasibility_cost(self, leaf: str, action: CandidateAction) -> tuple[float, dict[str, Any]]:
        reactant_count = max(len(action.reactants), 1)
        nonstock_small_fraction = self._nonstock_small_reactant_count(action) / reactant_count
        anti_progress = _bounded01(_anti_progress_penalty(leaf, action))
        invalid_flag = 1.0 if {"product_mismatch", "self_loop"} & set(action.validity_flags or ()) else 0.0
        atom_balance_failure = 0.0 if _candidate_atom_balance_ok(action, leaf) else 1.0
        risk = max(nonstock_small_fraction, anti_progress, invalid_flag, atom_balance_failure)
        cost = _negative_log_probability(1.0 - min(risk, 0.999))
        return cost, {
            "risk": round(float(_bounded01(risk)), 6),
            "nonstock_small_fraction": round(float(_bounded01(nonstock_small_fraction)), 6),
            "anti_progress": round(float(anti_progress), 6),
            "invalid_flag": bool(invalid_flag),
            "atom_balance_failure": bool(atom_balance_failure),
        }

    def _leaf_synthesis_cost(self, smiles: str, *, state: RouteTreeState, depth: int) -> float:
        if self._is_terminal(smiles, state=state, depth=depth):
            return 0.0
        atoms = max(1, _heavy_atoms(smiles))
        return math.log1p(float(atoms))

    def _oracle_match_for_action(self, state: RouteTreeState, leaf: str, action: CandidateAction):
        if not _oracle_action_value_enabled() or self.cascade_oracle is None:
            return None
        return self.cascade_oracle.action_value(target=state.target, leaf=leaf, action=action)

    def _oracle_reserve_key(self, state: RouteTreeState, leaf: str, action: CandidateAction) -> tuple[int, float, float]:
        match = self._oracle_match_for_action(state, leaf, action)
        if match is None:
            return (0, 0.0, 0.0)
        if match.reaction_match:
            tier = 2
        elif match.reactant_overlap > 0:
            tier = 1
        else:
            tier = 0
        return (tier, float(match.value), float(match.confidence))

    def _child_frontier_delta(self, state: RouteTreeState, next_open: tuple[str, ...]) -> float:
        parent_cost = self._frontier_cost(state.open_leaves, state=state, depth=state.depth)
        child_cost = self._frontier_cost(next_open, state=state, depth=state.depth + 1)
        return parent_cost - child_cost

    def _frontier_cost(self, leaves: tuple[str, ...] | list[str], *, state: RouteTreeState, depth: int) -> float:
        cost = 0.0
        for smi in leaves:
            cost += self._leaf_synthesis_cost(smi, state=state, depth=depth)
        return cost

    def _priority(self, state: RouteTreeState) -> float:
        nonterminal = [
            smi
            for smi in state.open_leaves
            if not self._is_terminal(smi, state=state, depth=state.depth)
        ]
        return state.score - self._frontier_cost(nonterminal, state=state, depth=state.depth)

    def _select_scored_children(
        self,
        scored_children: list[tuple[float, RouteTreeState, str, CandidateAction, list[CandidateAction], RouteTreeEvaluation]],
        *,
        state: RouteTreeState,
    ) -> list[tuple[float, RouteTreeState, str, CandidateAction, list[CandidateAction], RouteTreeEvaluation]]:
        ranked = sorted(scored_children, key=lambda item: item[0], reverse=True)
        limit = self._branch_limit_for_state(state, ranked)
        if len(ranked) <= limit or not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_BRANCH_RETENTION", True):
            return ranked[:limit]
        selected: list[tuple[float, RouteTreeState, str, CandidateAction, list[CandidateAction], RouteTreeEvaluation]] = []
        seen: set[str] = set()

        def add(item: tuple[float, RouteTreeState, str, CandidateAction, list[CandidateAction], RouteTreeEvaluation] | None) -> None:
            if item is None or len(selected) >= limit:
                return
            child = item[1]
            if child.canonical_id in seen:
                return
            seen.add(child.canonical_id)
            selected.append(item)

        primary_quota = max(1, limit // 2)
        for item in ranked[:primary_quota]:
            add(item)

        if _oracle_action_value_enabled() and self.cascade_oracle is not None:
            oracle_candidate = max(
                ranked,
                key=lambda item: self._oracle_reserve_key(state, item[2], item[3]),
                default=None,
            )
            if oracle_candidate is not None and self._oracle_reserve_key(state, oracle_candidate[2], oracle_candidate[3])[0] > 0:
                add(oracle_candidate)

        if state.depth == 0:
            best_score = ranked[0][0] if ranked else 0.0
            for source in _source_reserve_order():
                add(
                    next(
                        (
                            item
                            for item in ranked
                            if _action_source_family(item[3].source) == source
                            and self._source_reserve_allowed(item, state=state, best_score=best_score)
                        ),
                        None,
                    )
                )

        add(
            max(
                ranked,
                key=lambda item: _terminal_fraction(
                    item[3],
                    lambda smi: self._is_terminal(smi, state=state, depth=state.depth + 1),
                ),
                default=None,
            )
        )
        add(max(ranked, key=lambda item: _progress_delta(item[2], item[3]), default=None))
        if _env_truthy_default("AUTOPLANNER_ROUTE_TREE_FRONTIER_RESERVE", True):
            add(max(ranked, key=lambda item: self._child_frontier_delta(state, item[1].open_leaves), default=None))

        for item in ranked:
            add(item)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def _source_reserve_allowed(
        self,
        item: tuple[float, RouteTreeState, str, CandidateAction, list[CandidateAction], RouteTreeEvaluation],
        *,
        state: RouteTreeState,
        best_score: float,
    ) -> bool:
        max_gap = _env_float_or_none("AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MAX_SCORE_GAP", None)
        if max_gap is not None and float(item[0]) < float(best_score) - max(0.0, max_gap):
            return False
        min_frontier_delta = _env_float_or_none("AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MIN_FRONTIER_DELTA", None)
        if min_frontier_delta is not None:
            if self._child_frontier_delta(state, item[1].open_leaves) < float(min_frontier_delta):
                return False
        min_progress = _env_float_or_none("AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MIN_PROGRESS", None)
        if min_progress is not None:
            action = item[3]
            terminal_frac = _terminal_fraction(
                action,
                lambda smi: self._is_terminal(smi, state=state, depth=state.depth + 1),
            )
            if terminal_frac <= 0.0 and _progress_delta(item[2], action) < float(min_progress):
                return False
        return True

    def _branch_limit_for_state(
        self,
        state: RouteTreeState,
        ranked: list[tuple[float, RouteTreeState, str, CandidateAction, list[CandidateAction], RouteTreeEvaluation]],
    ) -> int:
        if not ranked:
            return 0
        if not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_ADAPTIVE_BRANCH_FACTOR", True):
            return self.branch_factor
        if state.depth == 0:
            limit = self.branch_factor
        else:
            limit = max(3, int(round(self.branch_factor * 0.75)))
        if any(
            _terminal_fraction(
                item[3],
                lambda smi: self._is_terminal(smi, state=state, depth=state.depth + 1),
            )
            >= 1.0
            for item in ranked[: self.branch_factor]
        ):
            limit = max(limit, min(self.branch_factor, limit + 1))
        if _env_truthy("AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS"):
            sources = {_action_source_family(item[3].source) for item in ranked if item[3].source}
            max_bonus = max(0, _env_int("AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS_MAX", 4))
            bonus = min(max_bonus, max(0, len(sources) - 1))
            cap = max(self.branch_factor, _env_int("AUTOPLANNER_ROUTE_TREE_BRANCH_FACTOR_CAP", self.branch_factor + bonus))
            limit = min(cap, limit + bonus)
        return min(len(ranked), max(1, limit))

    def _result_pool_target(self, n_results: int) -> int:
        requested = max(1, int(n_results or 1))
        multiplier = max(1.0, _env_float("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER", 1.0))
        minimum = max(requested, _env_int("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MIN", requested))
        maximum = max(minimum, _env_int("AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MAX", max(minimum, 12)))
        target = max(minimum, int(round(requested * multiplier)))
        return min(maximum, target)

    def _is_terminal(
        self,
        smiles: str,
        *,
        state: RouteTreeState | None = None,
        depth: int | None = None,
    ) -> bool:
        can = canonical_smiles(smiles)
        if not can:
            return False
        if self.allowed_starting_materials and can in self.allowed_starting_materials:
            return True
        small_molecule_terminal = _heavy_atoms(smiles) <= 6
        if self.stock_checker is not None:
            mode = _stock_terminal_mode()
            if mode == "always":
                try:
                    return bool(self.stock_checker(smiles))
                except Exception:
                    return False
            if mode == "late":
                current_depth = int(depth if depth is not None else (state.depth if state is not None else 0))
                remaining_depth = max(0, self.max_depth - current_depth)
                if remaining_depth <= _stock_terminal_remaining_depth():
                    try:
                        if bool(self.stock_checker(smiles)):
                            return True
                    except Exception:
                        pass
            return small_molecule_terminal
        return small_molecule_terminal

    def _is_solved(self, state: RouteTreeState) -> bool:
        return bool(state.steps) and all(
            self._is_terminal(smi, state=state, depth=state.depth)
            for smi in state.open_leaves
        )

    def _is_stock_or_small_terminal(self, smiles: str) -> bool:
        if self.stock_checker is not None and _stock_terminal_mode() == "always":
            return _stock_check(smiles, self.stock_checker)
        if _stock_check(smiles, self.stock_checker):
            return True
        if _heavy_atoms(smiles) <= 6:
            return True
        return False

    def _nonstock_small_reactant_count(self, action: CandidateAction) -> int:
        if self.stock_checker is None or _stock_terminal_mode() != "always":
            return 0
        return sum(
            1
            for smi in action.reactants
            if _heavy_atoms(smi) <= 6 and not _stock_check(smi, self.stock_checker)
        )

    def _add_fallback_result(
        self,
        fallback: list[RouteResult],
        seen: set[tuple[str, ...]],
        state: RouteTreeState,
        *,
        search_status: str,
    ) -> None:
        if not state.steps:
            return
        sig = _state_signature(state)
        if sig in seen:
            return
        seen.add(sig)
        fallback.append(self._state_to_result(state, search_status=search_status))

    def _state_to_result(self, state: RouteTreeState, *, search_status: str) -> RouteResult:
        board = state.to_board()
        metrics = route_metrics(board, stock_checker=self.stock_checker)
        depth_score = _target_depth_alignment_score(len(state.steps), self.target_depth)
        total_cost = max(0.0, -float(state.score or 0.0))
        explanation = RouteExplanation(
            why_selected="Neural-guided route-tree AO*/PUCT search",
            constraints_satisfied={"search_mode": "route_tree"},
            uncertainty_table={
                "search_mode": "route_tree",
                "route_tree_version": "v4_runtime_controlled_node_action_budget",
                "route_tree_search_status": search_status,
                "route_tree_state_id": state.canonical_id,
                "route_tree_score_semantics": "higher_score_is_lower_cost",
                "route_tree_total_cost": round(float(total_cost), 6),
                "route_tree_controller_active": bool(self.controller is not None),
                "route_tree_ccts_active": bool(self.ccts_scorer is not None),
                "route_tree_target_depth": self.target_depth,
                "route_tree_depth_alignment_score": depth_score,
                "route_tree_selected_node_sequence": list(state.search_metadata.get("selected_node_sequence") or []),
                "route_tree_selected_action_sequence": list(state.search_metadata.get("selected_action_sequence") or []),
                "route_tree_selection_trajectory": list(state.search_metadata.get("selection_trajectory") or []),
                "route_tree_value_trajectory": list(state.search_metadata.get("value_trajectory") or []),
                "route_tree_bottleneck_trajectory": list(state.search_metadata.get("bottleneck_trajectory") or []),
                "route_tree_source_budgets": list(state.search_metadata.get("source_budgets") or []),
                "route_tree_proposal_recall_diagnostics": list(state.search_metadata.get("proposal_recall_diagnostics") or []),
                "cascade_oracle_enabled": bool(self.cascade_oracle is not None),
                **self.stats.to_dict(),
            },
        )
        return RouteResult(
            board=board,
            quality_vector={
                "route_solved": float(bool(metrics.get("route_solved"))),
                "stock_closed": float(metrics.get("strict_stock_solve") is True),
                "progressive": float(bool(metrics.get("progressive_route"))),
                "route_cost": float(total_cost),
            },
            risk_vector={},
            score=-float(total_cost),
            confidence=0.75 if self.controller is not None else 0.45,
            constraint_report={"search_mode": "route_tree"},
            explanation=explanation,
        )

    def _elapsed_s(self) -> float:
        if self._search_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._search_started_at)

    def _soft_time_limit_reached(self) -> bool:
        limit = self.stats.soft_timeout_s
        return bool(limit and limit > 0 and self._elapsed_s() >= limit)

    def _hard_time_limit_reached(self) -> bool:
        limit = self.stats.hard_timeout_s
        return bool(limit and limit > 0 and self._elapsed_s() >= limit)

    def _attach_final_diagnostics(self, results: list[RouteResult], outcome: dict[str, Any]) -> None:
        final_stats = self.stats.to_dict()
        for result in results:
            table = result.explanation.uncertainty_table if result.explanation else {}
            table.update(final_stats)
            table["route_tree_final_outcome"] = dict(outcome)


def plan_with_route_tree(
    *,
    target: str,
    retro_engine: dict[str, Any] | None,
    stock_checker: StockChecker | None = None,
    max_depth: int = 6,
    n_results: int = 5,
    branch_factor: int = 12,
    expansion_budget: int = 200,
    skeletons: list[RouteSkeleton] | None = None,
    constraints: dict[str, Any] | None = None,
    controller: RouteTreeRuntime | None | object = _AUTO_CONTROLLER,
    trace_collector: RouteTreeTraceCollector | None = None,
) -> list[RouteResult]:
    planner = NeuralGuidedAOSearch(
        retro_engine=retro_engine,
        stock_checker=stock_checker,
        max_depth=max_depth,
        branch_factor=branch_factor,
        expansion_budget=expansion_budget,
        skeletons=skeletons,
        constraints=constraints,
        controller=controller,
        trace_collector=trace_collector,
    )
    return planner.search(target, n_results=n_results)


def _ccts_runtime_from_env():
    if os.environ.get("AUTOPLANNER_CCTS_V3_RUNTIME_MODEL"):
        try:
            from cascade_planner.route_tree.ccts_v3_runtime import ccts_v3_runtime_from_env

            return ccts_v3_runtime_from_env()
        except Exception:
            return None
    if not os.environ.get("AUTOPLANNER_CCTS_V0_MODEL"):
        return None
    try:
        from cascade_planner.route_tree.ccts_v0 import ccts_v0_runtime_from_env

        return ccts_v0_runtime_from_env()
    except Exception:
        return None


def _terminal_materials(constraints: dict[str, Any]) -> set[str]:
    return _canonical_set(
        constraints.get("starting_material"),
        constraints.get("starting_materials"),
        constraints.get("allowed_starting_materials"),
        constraints.get("terminal_reactants"),
        constraints.get("terminal_main_reactants"),
    )


def _canonical_set(*values: Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        if value in (None, "", [], {}, ()):
            continue
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            can = canonical_smiles(str(item))
            if can:
                out.add(can)
    return out


def _dedupe_smiles(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for smi in values:
        key = canonical_smiles(smi) or smi
        if key and key not in seen:
            seen.add(key)
            out.append(smi)
    return out


def _dedupe_candidate_actions(actions: list[CandidateAction]) -> list[CandidateAction]:
    out: list[CandidateAction] = []
    seen: set[str] = set()
    for action in actions:
        key = action.canonical_key
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _target_depth(skeletons: list[RouteSkeleton]) -> int | None:
    depths = [int(skel.n_steps) for skel in skeletons if int(getattr(skel, "n_steps", 0) or 0) > 0]
    return min(depths) if depths else None


def _target_depth_alignment_score(route_depth: int, target_depth: int | None) -> float:
    if not target_depth:
        return 0.0
    gap = abs(int(route_depth or 0) - int(target_depth or 0))
    return _bounded01(1.0 - (float(gap) / max(float(target_depth), 1.0)))


def _route_result_quality_score(result: RouteResult, stock_checker: StockChecker | None = None) -> float:
    tier, neg_cost, _neg_depth = _route_result_sort_key(result, stock_checker)
    return float(tier) + (1.0 / (1.0 + max(0.0, -neg_cost)))


def _route_result_sort_key(result: RouteResult, stock_checker: StockChecker | None = None) -> tuple[int, float, float]:
    metrics = route_metrics(result.board, stock_checker=stock_checker)
    strict_stock = metrics.get("strict_stock_solve")
    route_solved = bool(metrics.get("route_solved"))
    filled_route = bool(metrics.get("filled_route"))
    if route_solved and strict_stock is True:
        tier = 3
    elif route_solved:
        tier = 2
    elif filled_route:
        tier = 1
    else:
        tier = 0
    cost = _route_result_cost(result)
    depth = len(result.board.slots)
    return int(tier), -float(cost), -float(depth)


def _route_result_cost(result: RouteResult) -> float:
    table = result.explanation.uncertainty_table if result.explanation else {}
    if isinstance(table, dict) and table.get("route_tree_total_cost") is not None:
        try:
            return max(0.0, float(table.get("route_tree_total_cost")))
        except (TypeError, ValueError):
            pass
    try:
        return max(0.0, -float(result.score or 0.0))
    except (TypeError, ValueError):
        return math.inf


def _state_signature(state: RouteTreeState) -> tuple[str, ...]:
    return tuple(canonical_reaction(step.action.rxn_smiles) or step.action.canonical_key for step in state.steps)


def _transposition_key(state: RouteTreeState) -> tuple[int, tuple[str, ...]]:
    open_key = tuple(sorted(canonical_smiles(smi) or smi for smi in state.open_leaves if smi))
    return state.depth, open_key


def _result_signature(result: RouteResult) -> tuple[str, ...]:
    return tuple(canonical_reaction(slot.reaction_smiles) or str(slot.reaction_smiles or "") for slot in result.board.slots)


def _dedupe_results(results: list[RouteResult]) -> list[RouteResult]:
    out: list[RouteResult] = []
    seen: set[tuple[str, ...]] = set()
    for result in results:
        sig = _result_signature(result)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(result)
    return out


def _select_return_results(
    results: list[RouteResult],
    *,
    fallback: list[RouteResult],
    n_results: int,
    stock_checker: StockChecker | None = None,
) -> list[RouteResult]:
    requested = max(1, int(n_results or 1))
    returned = list(results[:requested])
    if not _env_truthy("AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS"):
        return returned
    contrast_limit = max(0, _env_int("AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX", 2))
    if contrast_limit <= 0:
        return returned
    seen = {_result_signature(result) for result in returned}
    ranked_fallback = sorted(_dedupe_results(list(fallback)), key=lambda result: _route_result_sort_key(result, stock_checker), reverse=True)
    for result in ranked_fallback:
        sig = _result_signature(result)
        if sig in seen:
            continue
        returned.append(result)
        seen.add(sig)
        if len(returned) >= requested + contrast_limit:
            break
    return returned


def _skeleton_context_index(skeleton: RouteSkeleton, depth: int) -> int:
    if not _reverse_skeleton_context_enabled():
        return depth
    n_steps = int(getattr(skeleton, "n_steps", 0) or len(getattr(skeleton, "types", []) or []))
    if n_steps <= 0:
        return depth
    return n_steps - 1 - depth


def _reverse_skeleton_context_enabled() -> bool:
    raw = os.environ.get("AUTOPLANNER_ROUTE_TREE_REVERSE_SKELETON_CONTEXT")
    if raw is None:
        return True
    return str(raw).lower() not in {"0", "false", "no", "off"}


def _extend_search_diagnostics(
    state: RouteTreeState,
    *,
    leaf: str,
    action: CandidateAction,
    eval_result: RouteTreeEvaluation,
    proposal_budget: int,
    candidate_pool_size: int,
    proposal_diagnostics: list[dict[str, Any]] | None = None,
    filter_diagnostics: list[dict[str, Any]] | None = None,
    selection_score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(state.search_metadata)
    metadata["selected_node_sequence"] = [*metadata.get("selected_node_sequence", []), leaf]
    metadata["selected_action_sequence"] = [*metadata.get("selected_action_sequence", []), action.canonical_key]
    metadata["value_trajectory"] = [
        *metadata.get("value_trajectory", []),
        {
            "route_value": eval_result.route_value,
            "value_logit": eval_result.value_logit,
            "solved_logit": eval_result.solved_logit,
            "stock_logit": eval_result.stock_logit,
            "progressive_logit": eval_result.progressive_logit,
            "compatibility_logit": eval_result.compatibility_logit,
            "solved": eval_result.solved_prob,
            "stock_closed": eval_result.stock_closed_prob,
            "progressive": eval_result.progressive_prob,
            "compatibility": eval_result.compatibility_prob,
            "value_calibrated": eval_result.value_calibrated,
        },
    ]
    metadata["bottleneck_trajectory"] = [
        *metadata.get("bottleneck_trajectory", []),
        _top_bottleneck(eval_result.bottleneck_scores),
    ]
    source_gate = (action.metadata or {}).get("source_gate") or {}
    metadata["source_budgets"] = [
        *metadata.get("source_budgets", []),
        {
            "learned_budget": dict(eval_result.source_budgets),
            "proposal_gate": source_gate,
            "proposal_budget": int(proposal_budget),
            "candidate_pool_size": int(candidate_pool_size),
        },
    ]
    metadata["proposal_recall_diagnostics"] = [
        *metadata.get("proposal_recall_diagnostics", []),
        {
            "leaf": leaf,
            "selected_action": action.canonical_key,
            "candidate_pool_size": int(candidate_pool_size),
            "selected_source": action.source,
            "selected_rank": action.rank,
        },
    ]
    if selection_score:
        metadata["selection_trajectory"] = [
            *metadata.get("selection_trajectory", []),
            {
                "leaf": leaf,
                "selected_action": action.canonical_key,
                "cost_model": selection_score.get("cost_model"),
                "base_probability": selection_score.get("base_probability"),
                "reaction_cost": selection_score.get("reaction_cost"),
                "child_value_cost": selection_score.get("child_value_cost"),
                "feasibility_cost": selection_score.get("feasibility_cost"),
                "total_cost": selection_score.get("total_cost"),
                "selection_score": selection_score.get("total"),
                "total_before_ccts": selection_score.get("total_before_ccts"),
                "ccts_active": selection_score.get("ccts_active"),
                "ccts_reason": selection_score.get("ccts_reason"),
                "ccts_selected_score": selection_score.get("ccts_selected_score"),
                "ccts_score": selection_score.get("ccts_score"),
                "ccts_score_z": selection_score.get("ccts_score_z"),
                "ccts_bonus": selection_score.get("ccts_bonus"),
                "source": action.source,
                "rank": action.rank,
            },
        ]
    if proposal_diagnostics:
        metadata["proposal_diagnostics"] = list(proposal_diagnostics)
    if filter_diagnostics:
        metadata["filter_diagnostics"] = list(filter_diagnostics)
    return metadata


def _top_bottleneck(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"label": "", "score": 0.0}
    idx = max(range(len(scores)), key=lambda i: scores[i])
    label = BOTTLENECK_LABELS[idx] if idx < len(BOTTLENECK_LABELS) else f"bottleneck_{idx}"
    return {"label": label, "score": float(scores[idx])}


def _proposal_source_stats_payload(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for source, row in sorted(rows.items()):
        calls = int(row.get("calls") or 0)
        latency = float(row.get("latency_ms_total") or 0.0)
        payload[source] = {
            "calls": calls,
            "allocated_budget": int(row.get("allocated_budget") or 0),
            "requested_k_total": int(row.get("requested_k_total") or 0),
            "kept_k_total": int(row.get("kept_k_total") or 0),
            "raw_returned": int(row.get("raw_returned") or 0),
            "ranker_kept": int(row.get("ranker_kept") or 0),
            "ranker_dropped": int(row.get("ranker_dropped") or 0),
            "kept_returned": int(row.get("kept_returned") or 0),
            "dedupe_dropped": int(row.get("dedupe_dropped") or 0),
            "invalid_dropped": int(row.get("invalid_dropped") or 0),
            "final_returned": int(row.get("final_returned") or 0),
            "latency_ms_total": round(latency, 3),
            "latency_ms_avg": round(latency / max(calls, 1), 3) if calls else 0.0,
            "latency_ms_max": round(float(row.get("latency_ms_max") or 0.0), 3),
        }
    return payload


def _runtime_bottleneck_labels(stats: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    elapsed = float(stats.get("elapsed_s") or 0.0)
    source_stats = stats.get("proposal_source_stats") or {}
    source_latency = sum(float(row.get("latency_ms_total") or 0.0) for row in source_stats.values()) / 1000.0
    expansions = int(stats.get("expansions") or 0)
    generated = int(stats.get("generated_actions") or 0)
    dead_ends = int(stats.get("dead_ends") or 0)
    proposal_calls = int(stats.get("proposal_calls") or 0)
    stop = str(stats.get("search_stop_reason") or "")
    if stop in {"hard_timeout", "soft_timeout_with_results"}:
        labels.append("source_timeout" if source_latency >= 0.4 * max(elapsed, 1e-6) else "runtime_cap")
    if proposal_calls and source_latency >= 0.5 * max(elapsed, 1e-6):
        labels.append("proposal_slow")
    if expansions and generated >= max(24, expansions * 8):
        labels.append("branch_churn")
    if expansions and dead_ends >= max(2, int(expansions * 0.5)):
        labels.append("dead_end_explosion")
    if int(stats.get("solved_routes") or 0) <= 0 and stop in {"queue_exhausted", "expansion_budget", "hard_timeout"}:
        labels.append("no_route_returned")
    if int(stats.get("skipped_leaf_count") or 0) > int(stats.get("expanded_leaf_count") or 0):
        labels.append("node_budget_limited")
    if not labels:
        labels.append("normal")
    return list(dict.fromkeys(labels))


def _is_enzymatic_action(action: CandidateAction) -> bool:
    source = str(action.source or "").lower()
    return bool(action.ec) or source in {"enzyformer", "enzexpand", "retrorules", "rhea_template", "enzymatic"}


def _reaction_type_compatible(action_type: str, expected_type: str) -> bool:
    action = _normalize_reaction_type(action_type)
    expected = _normalize_reaction_type(expected_type)
    generic = {
        "",
        "unknown",
        "enzyme",
        "enzymatic",
        "enzyme_reaction",
        "enzyme_retro_reaction",
        "rhea_reaction",
        "template",
        "external",
    }
    if action.startswith("uspto_class_") or action.startswith("class_"):
        return True
    if action in generic or expected in generic:
        return True
    return action == expected or action in expected or expected in action


def _normalize_reaction_type(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _source_reserve_order() -> tuple[str, ...]:
    raw = os.environ.get("AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_ORDER") or ""
    if not raw.strip():
        return DEFAULT_SOURCE_RESERVE_ORDER
    order = tuple(_action_source_family(item) for item in raw.replace(";", ",").split(",") if item.strip())
    return tuple(dict.fromkeys(order)) or DEFAULT_SOURCE_RESERVE_ORDER


def _action_source_family(source: str | None) -> str:
    text = str(source or "").strip()
    if text in {"chem_enzy_graphfp", "chem_enzy_onmt"}:
        return "chem_enzy_onestep"
    return text


def _softmax_probabilities(scores: list[float], *, n: int) -> list[float]:
    count = max(0, int(n))
    if count <= 0:
        return []
    values = [float(scores[idx]) if idx < len(scores) else 0.0 for idx in range(count)]
    if not values:
        return []
    max_value = max(values)
    exp_values = [math.exp(max(-60.0, min(60.0, value - max_value))) for value in values]
    denom = sum(exp_values)
    if denom <= 0.0:
        return [1.0 / count for _ in range(count)]
    return [float(value / denom) for value in exp_values]


def _probability_from_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < score <= 1.0:
        return _bounded_probability(score)
    return 0.0


def _bounded_probability(value: Any, *, floor: float = 1e-6) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability > 1.0:
        probability = 1.0
    return max(float(floor), probability)


def _negative_log_probability(probability: Any) -> float:
    return -math.log(max(1e-6, _bounded_probability(probability)))


def _oracle_probability(match: Any) -> float:
    if match is None:
        return 0.0
    value = _bounded_probability(getattr(match, "value", 0.0))
    if getattr(match, "reaction_match", False):
        return value
    overlap = _bounded01(getattr(match, "reactant_overlap", 0.0))
    return _bounded_probability(value * overlap) if overlap > 0.0 else 0.0


def _bounded01(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _controller_value_objective(eval_result: RouteTreeEvaluation) -> float | None:
    if not (eval_result.value_calibrated or _env_truthy("AUTOPLANNER_ROUTE_TREE_USE_UNCALIBRATED_VALUE_HEADS")):
        return None
    values = [
        _bounded01(eval_result.route_value),
        _bounded01(eval_result.solved_prob),
        _bounded01(eval_result.stock_closed_prob),
        _bounded01(eval_result.progressive_prob),
    ]
    return sum(values) / max(len(values), 1)


def _oracle_action_value_enabled() -> bool:
    return _env_float("AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT", 0.0) > 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_float_or_none(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if str(raw).lower() in {"", "0", "false", "none", "off", "no"}:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_truthy_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _stock_terminal_mode() -> str:
    raw = str(os.environ.get("AUTOPLANNER_ROUTE_TREE_STOCK_TERMINAL_MODE") or "always").lower()
    if raw in {"always", "stock", "stock_only"}:
        return "always"
    if raw in {"late", "late_depth", "remaining_depth"}:
        return "late"
    return "heuristic"


def _stock_terminal_remaining_depth() -> int:
    return max(0, _env_int("AUTOPLANNER_ROUTE_TREE_STOCK_TERMINAL_REMAINING_DEPTH", 1))


def _terminal_fraction(action: CandidateAction, is_terminal: Callable[[str], bool]) -> float:
    if not action.reactants:
        return 0.0
    return sum(int(is_terminal(smi)) for smi in action.reactants) / max(len(action.reactants), 1)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _ec1(value: str | None) -> str:
    text = str(value or "").strip()
    return text.split(".", 1)[0] if text else ""


def _candidate_atom_balance_ok(action: CandidateAction, product: str) -> bool:
    product_atoms = _heavy_atoms(product)
    reactant_atoms = sum(_heavy_atoms(smi) for smi in action.reactants)
    if product_atoms <= 10:
        return True
    return reactant_atoms >= max(4, int(product_atoms * 0.35))


def _progress_delta(product: str, action: CandidateAction) -> float:
    product_atoms = _heavy_atoms(product)
    largest = max((_heavy_atoms(smi) for smi in action.reactants), default=product_atoms)
    return max(0.0, (product_atoms - largest) / max(product_atoms, 1)) if product_atoms else 0.0


def _anti_progress_penalty(product: str, action: CandidateAction) -> float:
    product_atoms = _heavy_atoms(product)
    largest = max((_heavy_atoms(smi) for smi in action.reactants), default=product_atoms)
    if product_atoms <= 0 or largest <= product_atoms:
        return 0.0
    return 0.5 + (largest - product_atoms) / max(product_atoms, 1)


def _heuristic_leaf_score(
    state: RouteTreeState,
    leaf: str,
    *,
    stock_checker: StockChecker | None,
) -> float:
    can = canonical_smiles(leaf) or leaf
    if can in state.expanded:
        return -float("inf")
    if _stock_check(leaf, stock_checker):
        return 0.0
    return math.log1p(float(max(1, _heavy_atoms(leaf))))


def _stock_check(smiles: str, stock_checker: StockChecker | None) -> bool:
    if stock_checker is not None:
        try:
            return bool(stock_checker(smiles))
        except Exception:
            return False
    return _heavy_atoms(smiles) <= 6


def _leaf_has_parent_reaction(state: RouteTreeState, leaf: str) -> bool:
    can = canonical_smiles(leaf) or leaf
    for step in state.steps:
        if any((canonical_smiles(smi) or smi) == can for smi in step.action.reactants):
            return True
    return False


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _oxygen_rich_molecule(smiles: str | None) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    heavy = mol.GetNumHeavyAtoms()
    oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    return oxygen >= 5 and oxygen / max(heavy, 1) >= 0.40


def _carbohydrate_like_molecule(smiles: str | None) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None or not _oxygen_rich_molecule(smiles):
        return False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset({"C", "O"})
