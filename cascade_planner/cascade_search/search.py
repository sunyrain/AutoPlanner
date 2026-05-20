"""Cascade-native best-first program search."""
from __future__ import annotations

import heapq
import itertools
import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from cascade_planner.cascade_search.cost import score_cascade_state
from cascade_planner.cascade_search.failure import annotate_state_failures, detect_cascade_failures
from cascade_planner.cascade_search.interfaces import CascadeSearchController
from cascade_planner.cascade_search.proposals import ProposalRequest, coerce_to_cascade_action
from cascade_planner.cascade_search.repair import CascadeRepairPolicy
from cascade_planner.cascade_search.trace import CascadeTraceCollector
from cascade_planner.cascade_search.state import (
    CascadeAction,
    CascadeActionType,
    CascadeFailure,
    CascadeFailureKind,
    CascadeProgramState,
    StageTransition,
)
from cascade_planner.cascade_search.value import (
    CascadeSourcePolicy,
    CofactorClosureModel,
    ConditionTransitionModel,
    EnzymeModuleRanker,
    HeuristicCascadeValueModel,
)


StockChecker = Callable[[str], bool]


@dataclass
class CascadeSearchConfig:
    max_depth: int = 6
    branch_factor: int = 8
    leaf_beam_size: int = 1
    diverse_leaf_reserve: int = 0
    proposal_top_k: int | None = None
    expansion_budget: int = 100
    max_stages: int = 4
    allow_repair_actions: bool = True
    max_repairs_per_state: int = 4
    min_evidence_confidence: float = 0.4
    min_step_score: float = 0.05
    pair_reward_weight: float = 0.0
    pair_reward_mode: str = "additive"
    pair_reward_tie_epsilon: float = 0.0
    soft_timeout_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CascadeSearchStats:
    expansions: int = 0
    generated_actions: int = 0
    provider_calls: int = 0
    repair_actions: int = 0
    solved_programs: int = 0
    dead_ends: int = 0
    max_queue_size: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = ""
    provider_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["elapsed_s"] = round(float(self.elapsed_s or 0.0), 3)
        return data


@dataclass
class CascadeSearchResult:
    state: CascadeProgramState
    solved: bool
    score: float
    failures: list[CascadeFailure] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "solved": self.solved,
            "score": self.score,
            "failures": [failure.to_dict() for failure in self.failures],
            "diagnostics": self.diagnostics,
        }


class CascadeProgramSearch:
    """Search over CascadeProgramState rather than molecule-only route trees."""

    def __init__(
        self,
        proposal_providers: list[Any] | None = None,
        *,
        stock_checker: StockChecker | None = None,
        config: CascadeSearchConfig | None = None,
        repair_policy: CascadeRepairPolicy | None = None,
        value_model: Any | None = None,
        controller: CascadeSearchController | None = None,
        trace_collector: CascadeTraceCollector | None = None,
    ):
        self.proposal_providers = list(proposal_providers or [])
        self.stock_checker = stock_checker
        self.config = config or CascadeSearchConfig()
        self.repair_policy = repair_policy or CascadeRepairPolicy.default()
        self.controller = controller or default_cascade_search_controller(value_model=value_model)
        self.value_model = self.controller.value_model
        self.trace_collector = trace_collector
        self.stats = CascadeSearchStats()
        self._started_at = 0.0

    def search(self, target_smiles: str, *, n_results: int = 5) -> list[CascadeSearchResult]:
        self._started_at = time.monotonic()
        self.stats = CascadeSearchStats()
        initial = CascadeProgramState.initial(target_smiles)
        self._refresh_stock(initial)
        annotate_state_failures(
            initial,
            stock_checker=self.stock_checker,
            max_stages=self.config.max_stages,
            min_evidence_confidence=self.config.min_evidence_confidence,
            min_step_score=self.config.min_step_score,
        )

        queue: list[tuple[float, int, CascadeProgramState]] = []
        counter = itertools.count()
        heapq.heappush(queue, (self._priority(initial), next(counter), initial))
        results: list[CascadeSearchResult] = []
        fallbacks: list[CascadeSearchResult] = []
        seen: set[str] = set()

        while queue and self.stats.expansions < self.config.expansion_budget and len(results) < n_results:
            if self._timed_out():
                self.stats.stop_reason = "soft_timeout"
                break
            self.stats.max_queue_size = max(self.stats.max_queue_size, len(queue))
            _, _, state = heapq.heappop(queue)
            self._refresh_stock(state)
            failures = detect_cascade_failures(
                state,
                stock_checker=self.stock_checker,
                max_stages=self.config.max_stages,
                min_evidence_confidence=self.config.min_evidence_confidence,
                min_step_score=self.config.min_step_score,
            )
            state.unresolved_failure_modes = failures
            state.cascade_cost = score_cascade_state(state).total_cost

            if self._is_goal(state, failures):
                result = self._result(state, solved=True, failures=failures)
                signature = _state_signature(state)
                if signature not in seen:
                    seen.add(signature)
                    results.append(result)
                    self.stats.solved_programs += 1
                continue

            if len(state.step_annotations) >= self.config.max_depth:
                self.stats.dead_ends += 1
                self._add_fallback(fallbacks, seen, state, failures, "depth_limit")
                continue

            children = self._expand_state(state, failures)
            if not children:
                self.stats.dead_ends += 1
                self._add_fallback(fallbacks, seen, state, failures, "dead_end")
                continue
            self.stats.expansions += 1
            for child in children:
                signature = _state_signature(child)
                if signature in seen:
                    continue
                heapq.heappush(queue, (self._priority(child), next(counter), child))

        self.stats.elapsed_s = time.monotonic() - self._started_at
        if not self.stats.stop_reason:
            if len(results) >= n_results:
                self.stats.stop_reason = "result_limit"
            elif self.stats.expansions >= self.config.expansion_budget:
                self.stats.stop_reason = "expansion_budget"
            elif not queue:
                self.stats.stop_reason = "queue_exhausted"
            else:
                self.stats.stop_reason = "stopped"
        if self.trace_collector is not None:
            self.trace_collector.annotate_outcome(
                {
                    "solved": bool(results),
                    "n_results": len(results),
                    "stop_reason": self.stats.stop_reason,
                    "search_stats": self.stats.to_dict(),
                }
            )
        output = results if results else fallbacks
        output.sort(key=lambda item: item.score, reverse=True)
        for result in output:
            result.diagnostics.setdefault("search_stats", self.stats.to_dict())
        return output[:n_results]

    def _expand_state(self, state: CascadeProgramState, failures: list[CascadeFailure]) -> list[CascadeProgramState]:
        children: list[CascadeProgramState] = []
        if self.config.allow_repair_actions:
            repair_actions = self.repair_policy.propose_repairs(state, failures)[: self.config.max_repairs_per_state]
            self.stats.repair_actions += len(repair_actions)
            for action in repair_actions:
                child = apply_cascade_action(state, action, stock_checker=self.stock_checker)
                child.raw_metadata.setdefault("applied_actions", []).append(action.to_dict())
                children.append(child)

        leaves = self._select_leaves(state)
        if not leaves:
            return children

        for leaf in leaves:
            children.extend(self._expand_leaf(state, failures, leaf))
        children.sort(key=self._priority)
        selected = children[: max(self.config.branch_factor, self.config.max_repairs_per_state)]
        enqueued_ids = {id(child) for child in selected}
        for child in children:
            _mark_last_action_enqueued(child, id(child) in enqueued_ids)
        if self.trace_collector is not None:
            self.trace_collector.mark_enqueued_children(parent_state=state, enqueued_children=selected)
        return selected

    def _expand_leaf(
        self,
        state: CascadeProgramState,
        failures: list[CascadeFailure],
        leaf: str,
    ) -> list[CascadeProgramState]:
        children: list[CascadeProgramState] = []
        request = ProposalRequest(
            leaf_smiles=leaf,
            state=state,
            depth_remaining=max(0, self.config.max_depth - len(state.step_annotations)),
            top_k=self._proposal_top_k(),
            failure_modes=[failure.category for failure in failures],
        )
        actions = self._propose(request)
        context_state = state
        sidecar_evidence_actions = [action for action in actions if _is_sidecar_evidence_action(action)]
        if sidecar_evidence_actions:
            context_state = state.copy()
            for evidence_action in sidecar_evidence_actions:
                context_state = apply_cascade_action(context_state, evidence_action, stock_checker=self.stock_checker)
                context_state.raw_metadata.setdefault("applied_sidecar_evidence_actions", []).append(evidence_action.to_dict())
            actions = [action for action in actions if not _is_sidecar_evidence_action(action)]
        if not actions:
            if sidecar_evidence_actions:
                annotate_state_failures(
                    context_state,
                    stock_checker=self.stock_checker,
                    max_stages=self.config.max_stages,
                    min_evidence_confidence=self.config.min_evidence_confidence,
                    min_step_score=self.config.min_step_score,
                )
                return [context_state]
            failure = CascadeFailure(
                CascadeFailureKind.CANDIDATE_MISSING,
                message="No proposal provider returned a candidate for the selected open leaf.",
                target_leaf=leaf,
                repair_options=[CascadeActionType.RETROSYNTHETIC_STEP],
            )
            state.unresolved_failure_modes = [*failures, failure]
            if self.trace_collector is not None:
                self.trace_collector.record_expansion(
                    state=state,
                    expanded_leaf=leaf,
                    candidate_actions=[],
                    candidate_scores=[],
                    child_states=[],
                    failures=state.unresolved_failure_modes,
                    model_active=_learned_value_active(self.controller),
                )
            return children

        candidate_limit = len(actions) if self._score_all_candidate_transitions() else min(len(actions), self.config.branch_factor)
        candidate_children: list[tuple[CascadeAction, CascadeProgramState]] = []
        for action in actions[:candidate_limit]:
            child = apply_cascade_action(context_state, action, stock_checker=self.stock_checker)
            child.raw_metadata.setdefault("applied_actions", []).append(action.to_dict())
            annotate_state_failures(
                child,
                stock_checker=self.stock_checker,
                max_stages=self.config.max_stages,
                min_evidence_confidence=self.config.min_evidence_confidence,
                min_step_score=self.config.min_step_score,
            )
            candidate_children.append((action, child))
        action_scores = self._transition_scores(context_state, leaf, candidate_children)
        ranked_candidates = list(zip(candidate_children, action_scores, range(len(candidate_children))))
        if self.controller.transition_value_model is not None or self.controller.action_value_model is not None or self._pair_reward_active():
            ranked_candidates.sort(
                key=lambda item: (
                    float(item[1]),
                    float(item[0][0].step.score) if item[0][0].step is not None and item[0][0].step.score is not None else -1e9,
                    -int(item[2]),
                ),
                reverse=True,
            )
        for rank, ((action, child), score, _) in enumerate(ranked_candidates, start=1):
            action.metadata.setdefault("transition_value_score", float(score))
            action.metadata.setdefault("transition_value_rank", rank)
            if child.raw_metadata.get("applied_actions"):
                child.raw_metadata["applied_actions"][-1] = action.to_dict()
        selected_candidates = self._select_ranked_candidates(ranked_candidates)
        selected_child_ids = {id(item[0][1]) for item in selected_candidates}
        for (action, child), _, _ in selected_candidates:
            children.append(child)
            self.stats.generated_actions += 1
        for (action, child), _, _ in ranked_candidates:
            selected_by_leaf = id(child) in selected_child_ids
            action.metadata["candidate_selection_status"] = (
                str(action.metadata.get("candidate_selection_status") or "selected_by_leaf")
                if selected_by_leaf
                else "not_selected_by_leaf"
            )
            if child.raw_metadata.get("applied_actions"):
                child.raw_metadata["applied_actions"][-1] = action.to_dict()
        if self.trace_collector is not None:
            self.trace_collector.record_expansion(
                state=state,
                expanded_leaf=leaf,
                candidate_actions=[action for (action, _), _, _ in ranked_candidates],
                candidate_scores=[float(score) for _, score, _ in ranked_candidates],
                child_states=[child for (_, child), _, _ in ranked_candidates],
                failures=failures,
                model_active=_learned_value_active(self.controller) or self.controller.transition_value_model is not None,
            )
        return children

    def _select_ranked_candidates(
        self,
        ranked_candidates: list[tuple[tuple[CascadeAction, CascadeProgramState], float, int]],
    ) -> list[tuple[tuple[CascadeAction, CascadeProgramState], float, int]]:
        limit = max(1, int(self.config.branch_factor or 1))
        if len(ranked_candidates) <= limit:
            return ranked_candidates
        selected = list(ranked_candidates[:limit])
        for (action, _child), _score, _idx in selected:
            action.metadata.setdefault("selection_reason", "top_rank")
        reserve = max(0, int(self.config.diverse_leaf_reserve or 0))
        if reserve <= 0:
            return selected
        reserve = min(reserve, limit)
        seen_open = _child_open_leaf_signature(child for (_, child), _, _ in selected)
        diverse: list[tuple[tuple[CascadeAction, CascadeProgramState], float, int]] = []
        selected_ids = {id(item[0][1]) for item in selected}
        for item in ranked_candidates[limit:]:
            child = item[0][1]
            signature = _child_open_leaf_signature([child])
            if not signature or signature <= seen_open:
                continue
            item[0][0].metadata["selection_reason"] = "diverse_leaf_reserve"
            diverse.append(item)
            seen_open.update(signature)
            if len(diverse) >= reserve:
                break
        if not diverse:
            return selected
        kept = selected[: max(0, limit - len(diverse))]
        for item in diverse:
            if id(item[0][1]) not in selected_ids:
                kept.append(item)
        return kept[:limit]

    def _score_all_candidate_transitions(self) -> bool:
        return (
            self.controller.transition_value_model is not None
            or self.controller.action_value_model is not None
            or self._pair_reward_active()
            or self.trace_collector is not None
        )

    def _pair_reward_active(self) -> bool:
        return self.controller.pair_scorer is not None and float(self.config.pair_reward_weight or 0.0) != 0.0

    def _transition_scores(
        self,
        state: CascadeProgramState,
        leaf: str,
        candidate_children: list[tuple[CascadeAction, CascadeProgramState]],
    ) -> list[float]:
        if not candidate_children:
            return []
        fallback = [
            float(action.step.score or 0.0) if action.step is not None else 0.0
            for action, _ in candidate_children
        ]
        model = self.controller.transition_value_model
        action_model = self.controller.action_value_model
        if model is None:
            scores = fallback
        else:
            try:
                scores = [
                    float(score)
                    for score in model.score_transitions(
                        state,
                        [action for action, _ in candidate_children],
                        [child for _, child in candidate_children],
                        expanded_leaf=leaf,
                    )
                ]
            except Exception:
                scores = fallback
            if len(scores) != len(candidate_children):
                scores = fallback
        if action_model is not None:
            try:
                action_scores = [
                    float(score)
                    for score in action_model.score_actions(
                        state,
                        [action for action, _ in candidate_children],
                        [child for _, child in candidate_children],
                        expanded_leaf=leaf,
                    )
                ]
                if len(action_scores) == len(candidate_children):
                    for score, (action, child) in zip(action_scores, candidate_children):
                        action.metadata.setdefault("action_value_score", float(score))
                        self._record_action_value_credit(child, action, float(score))
                    if model is None:
                        scores = action_scores
                    else:
                        scores = [
                            0.5 * float(transition_score) + 0.5 * float(action_score)
                            for transition_score, action_score in zip(scores, action_scores)
                        ]
            except Exception as exc:
                for action, _ in candidate_children:
                    action.metadata.setdefault("action_value_error", f"{type(exc).__name__}: {exc}")
        return self._apply_pair_rewards(state, leaf, candidate_children, scores)

    def _apply_pair_rewards(
        self,
        state: CascadeProgramState,
        leaf: str,
        candidate_children: list[tuple[CascadeAction, CascadeProgramState]],
        base_scores: list[float],
    ) -> list[float]:
        scorer = self.controller.pair_scorer
        weight = float(self.config.pair_reward_weight or 0.0)
        if scorer is None or weight == 0.0 or not candidate_children:
            return base_scores
        rewards: list[float] = []
        applicable_flags: list[bool] = []
        for base, (action, child) in zip(base_scores, candidate_children):
            try:
                pair_score = scorer.score_action(state, action, child, expanded_leaf=leaf)
                applicable = bool(getattr(pair_score, "applicable", True))
                reward = float(getattr(pair_score, "search_reward", 0.0)) if applicable else 0.0
                if hasattr(pair_score, "to_dict"):
                    action.metadata.setdefault("cascade_pair_score", pair_score.to_dict())
                action.metadata.setdefault("cascade_pair_applicable", applicable)
            except Exception as exc:
                reward = 0.0
                action.metadata.setdefault("cascade_pair_score_error", f"{type(exc).__name__}: {exc}")
                action.metadata.setdefault("cascade_pair_applicable", False)
            action.metadata.setdefault("cascade_pair_reward", reward)
            applicable_flags.append(bool(action.metadata.get("cascade_pair_applicable")))
            rewards.append(float(reward))

        mode = str(self.config.pair_reward_mode or "additive")
        if mode == "guarded_tie_break":
            out = self._guarded_pair_reward_scores(candidate_children, base_scores, rewards, applicable_flags, weight=weight)
        else:
            out = [float(base) + weight * float(reward) for base, reward in zip(base_scores, rewards)]
            for (action, child), reward, applicable in zip(candidate_children, rewards, applicable_flags):
                action.metadata.setdefault("cascade_pair_reward_mode", "additive")
                action.metadata.setdefault("cascade_pair_reward_applied", bool(applicable and reward != 0.0))
                if applicable:
                    self._record_pair_delta(child, action, reward)
        return out

    def _guarded_pair_reward_scores(
        self,
        candidate_children: list[tuple[CascadeAction, CascadeProgramState]],
        base_scores: list[float],
        rewards: list[float],
        applicable_flags: list[bool],
        *,
        weight: float,
    ) -> list[float]:
        if not candidate_children:
            return []
        base_order = sorted(range(len(base_scores)), key=lambda idx: (float(base_scores[idx]), -idx), reverse=True)
        base_top_idx = base_order[0]
        base_top_score = float(base_scores[base_top_idx])
        _, base_top_child = candidate_children[base_top_idx]
        base_top_stock_closed = _state_stock_closed(base_top_child)
        base_top_no_failure = _state_no_failures(base_top_child)
        tie_epsilon = max(0.0, float(self.config.pair_reward_tie_epsilon or 0.0))
        out = [float(value) for value in base_scores]
        for idx, ((action, child), reward, applicable) in enumerate(zip(candidate_children, rewards, applicable_flags)):
            action.metadata.setdefault("cascade_pair_reward_mode", "guarded_tie_break")
            if not applicable or float(reward) == 0.0:
                action.metadata.setdefault("cascade_pair_reward_applied", False)
                action.metadata.setdefault("cascade_pair_guard_reason", "not_applicable")
                continue
            if float(base_scores[idx]) < base_top_score - tie_epsilon:
                action.metadata.setdefault("cascade_pair_reward_applied", False)
                action.metadata.setdefault("cascade_pair_guard_reason", "outside_base_tie_window")
                continue
            if base_top_stock_closed and not _state_stock_closed(child):
                action.metadata.setdefault("cascade_pair_reward_applied", False)
                action.metadata.setdefault("cascade_pair_guard_reason", "would_regress_stock_closure")
                continue
            if base_top_no_failure and not _state_no_failures(child):
                action.metadata.setdefault("cascade_pair_reward_applied", False)
                action.metadata.setdefault("cascade_pair_guard_reason", "would_regress_failure_status")
                continue
            action.metadata.setdefault("cascade_pair_reward_applied", True)
            action.metadata.setdefault("cascade_pair_guard_reason", "applied")
            self._record_pair_delta(child, action, reward)
            out[idx] = float(base_scores[idx]) + float(weight) * float(reward)
        return out

    def _record_pair_delta(self, child: CascadeProgramState, action: CascadeAction, reward: float) -> None:
        metadata = child.raw_metadata.setdefault("cascade_pair_summary", {})
        deltas = metadata.setdefault("deltas", [])
        payload = {
            "reward": float(reward),
            "action_rxn_smiles": action.step.rxn_smiles if action.step is not None else "",
            "action_target_leaf": action.target_leaf,
            "score": action.metadata.get("cascade_pair_score"),
        }
        deltas.append(payload)
        rewards = [float(item.get("reward") or 0.0) for item in deltas]
        metadata["valid_pair_count"] = len(rewards)
        metadata["total_reward"] = round(sum(rewards), 6)
        metadata["mean_reward"] = round(sum(rewards) / len(rewards), 6) if rewards else 0.0
        metadata["min_reward"] = round(min(rewards), 6) if rewards else 0.0

    def _record_action_value_credit(self, child: CascadeProgramState, action: CascadeAction, score: float) -> None:
        metadata = child.raw_metadata.setdefault("cascade_action_value_summary", {})
        deltas = metadata.setdefault("deltas", [])
        payload = {
            "score": float(score),
            "action_rxn_smiles": action.step.rxn_smiles if action.step is not None else "",
            "action_target_leaf": action.target_leaf,
        }
        deltas.append(payload)
        scores = [float(item.get("score") or 0.0) for item in deltas]
        metadata["count"] = len(scores)
        metadata["total_score"] = round(sum(scores), 6)
        metadata["mean_score"] = round(sum(scores) / len(scores), 6) if scores else 0.0
        metadata["max_score"] = round(max(scores), 6) if scores else 0.0

    def _propose(self, request: ProposalRequest) -> list[CascadeAction]:
        actions: list[CascadeAction] = []
        allocation = self._provider_allocation(request)
        for provider in self.proposal_providers:
            provider_name = str(getattr(provider, "provider_name", type(provider).__name__))
            if provider_name == "cascade_subgoal_evidence":
                provider_budget = max(1, int(getattr(provider, "max_hints_per_leaf", 1) or 1))
            else:
                provider_budget = int(allocation.get(provider_name, request.top_k) or 0)
            if provider_budget <= 0:
                continue
            self.stats.provider_calls += 1
            provider_request = replace(request, top_k=provider_budget)
            rows = _call_provider(provider, provider_request)
            provider_actions = [
                coerce_to_cascade_action(row, target_leaf=request.leaf_smiles, source=provider_name)
                for row in rows
            ]
            actions.extend(provider_actions)
            diagnostics = getattr(provider, "last_diagnostics", None)
            if diagnostics is not None:
                self.stats.provider_diagnostics.append(
                    diagnostics.to_dict() if hasattr(diagnostics, "to_dict") else dict(diagnostics)
                )
        deduped = _dedupe_actions(actions)
        for rank, action in enumerate(deduped, start=1):
            action.metadata.setdefault("provider_rank", rank)
        return deduped

    def _provider_allocation(self, request: ProposalRequest) -> dict[str, int]:
        if self.controller.source_policy is None or not self.proposal_providers:
            return {
                str(getattr(provider, "provider_name", type(provider).__name__)): int(request.top_k or 0)
                for provider in self.proposal_providers
            }
        names = [str(getattr(provider, "provider_name", type(provider).__name__)) for provider in self.proposal_providers]
        try:
            budget = self.controller.source_policy.allocate(
                request.state,
                available_sources=names,
                total_budget=max(1, int(request.top_k or 1)),
            )
            return dict(getattr(budget, "source_budgets", {}) or {})
        except Exception:
            return {name: int(request.top_k or 0) for name in names}

    def _proposal_top_k(self) -> int:
        if self.config.proposal_top_k is not None:
            return max(int(self.config.branch_factor or 1), int(self.config.proposal_top_k or 1))
        if self._score_all_candidate_transitions():
            return max(int(self.config.branch_factor or 1) * 4, int(self.config.branch_factor or 1) + 8)
        return int(self.config.branch_factor or 1)

    def _select_leaf(self, state: CascadeProgramState) -> str:
        leaves = self._select_leaves(state)
        return leaves[0] if leaves else ""

    def _select_leaves(self, state: CascadeProgramState) -> list[str]:
        leaves = [leaf for leaf in state.open_molecule_leaves if not self._is_stock(leaf)]
        if not leaves:
            return []
        leaves.sort(key=len, reverse=True)
        return leaves[: max(1, int(self.config.leaf_beam_size or 1))]

    def _is_stock(self, smiles: str) -> bool:
        if self.stock_checker is not None:
            try:
                return bool(self.stock_checker(smiles))
            except Exception:
                return False
        return False

    def _refresh_stock(self, state: CascadeProgramState) -> None:
        if self.stock_checker is None:
            return
        for leaf in list(state.open_molecule_leaves):
            if leaf in state.stock_status and state.stock_status[leaf] is True:
                continue
            try:
                state.stock_status[leaf] = bool(self.stock_checker(leaf))
            except Exception:
                state.stock_status[leaf] = state.stock_status.get(leaf)

    def _is_goal(self, state: CascadeProgramState, failures: list[CascadeFailure]) -> bool:
        if not state.step_annotations:
            return False
        blocking = {
            CascadeFailureKind.CONDITION_CONFLICT.value,
            CascadeFailureKind.COFACTOR_DEBT.value,
            CascadeFailureKind.STAGE_OVER_COMPLEX.value,
            CascadeFailureKind.ROUTE_ORDER_MISMATCH.value,
            CascadeFailureKind.LOW_PLAUSIBILITY.value,
            CascadeFailureKind.ENZYME_EVIDENCE_WEAK.value,
            CascadeFailureKind.STOCK_DEAD_END.value,
        }
        return state.stock_closed and not any(failure.category in blocking for failure in failures)

    def _priority(self, state: CascadeProgramState) -> float:
        cost = score_cascade_state(state).total_cost
        try:
            value = float(self.value_model.predict(state).value)
        except Exception:
            value = 0.0
        repair_cost = sum(float((action or {}).get("cost_delta") or 0.0) for action in state.raw_metadata.get("applied_actions", []))
        pair_credit = self._state_pair_credit(state)
        action_credit = self._state_action_value_credit(state)
        return cost + repair_cost - value - pair_credit - action_credit

    def _state_pair_credit(self, state: CascadeProgramState) -> float:
        if not self._pair_reward_active():
            return 0.0
        summary = state.raw_metadata.get("cascade_pair_summary") or {}
        return float(summary.get("total_reward") or 0.0) * float(self.config.pair_reward_weight or 0.0)

    def _state_action_value_credit(self, state: CascadeProgramState) -> float:
        if self.controller.action_value_model is None:
            return 0.0
        summary = state.raw_metadata.get("cascade_action_value_summary") or {}
        return float(summary.get("mean_score") or summary.get("max_score") or 0.0)

    def _result(
        self,
        state: CascadeProgramState,
        *,
        solved: bool,
        failures: list[CascadeFailure],
        status: str = "",
    ) -> CascadeSearchResult:
        cost = score_cascade_state(state)
        value = self.value_model.predict(state)
        pair_credit = self._state_pair_credit(state)
        action_credit = self._state_action_value_credit(state)
        return CascadeSearchResult(
            state=state,
            solved=solved,
            score=round(float(value.value) + pair_credit + action_credit - float(cost.total_cost), 6),
            failures=failures,
            diagnostics={
                "status": status or ("solved" if solved else "partial"),
                "cost": cost.to_dict(),
                "value": value.to_dict(),
                "condition_state": state.to_dict().get("condition_state"),
                "action_value_credit": action_credit,
                "config": self.config.to_dict(),
                "controller": self.controller.to_dict(),
            },
        )

    def _add_fallback(
        self,
        fallbacks: list[CascadeSearchResult],
        seen: set[str],
        state: CascadeProgramState,
        failures: list[CascadeFailure],
        status: str,
    ) -> None:
        if not state.step_annotations and not state.raw_metadata.get("cascade_subgoal_hints"):
            return
        signature = _state_signature(state)
        if signature in seen:
            return
        seen.add(signature)
        fallbacks.append(self._result(state, solved=False, failures=failures, status=status))

    def _timed_out(self) -> bool:
        limit = self.config.soft_timeout_s
        return bool(limit and limit > 0 and (time.monotonic() - self._started_at) >= limit)


def apply_cascade_action(
    state: CascadeProgramState,
    action: CascadeAction,
    *,
    stock_checker: StockChecker | None = None,
) -> CascadeProgramState:
    action = coerce_to_cascade_action(action, target_leaf=action.target_leaf, source=action.source)
    child = state.copy()
    if action.action_type == CascadeActionType.RETROSYNTHETIC_STEP:
        if action.step is None:
            return child
        _apply_retrosynthetic_step(child, action, stock_checker=stock_checker)
    elif action.action_type == CascadeActionType.ENZYME_MODULE:
        if action.module is not None:
            child.add_module(action.module)
    elif action.action_type == CascadeActionType.COFACTOR_REPAIR:
        if action.module is not None:
            child.add_module(action.module)
    elif action.action_type == CascadeActionType.STAGE_TRANSITION:
        if action.stage_transition is not None:
            _apply_stage_transition(child, action.stage_transition)
    elif action.action_type == CascadeActionType.EVIDENCE_RETRIEVAL:
        _apply_evidence_retrieval(child, action)
    child._sync_aliases()
    return child


def _apply_retrosynthetic_step(
    state: CascadeProgramState,
    action: CascadeAction,
    *,
    stock_checker: StockChecker | None,
) -> None:
    step = action.step
    if step is None:
        return
    leaf = action.target_leaf or step.product_smiles
    step.stage_id = step.stage_id or state.current_stage or "stage_1"
    if step.stage_id == "stage_1" and state.current_stage:
        step.stage_id = state.current_stage
    next_open = list(state.open_molecule_leaves or state.open_leaves)
    try:
        next_open.remove(leaf)
    except ValueError:
        try:
            next_open.remove(step.product_smiles)
        except ValueError:
            pass
    for reactant in step.reactant_smiles:
        in_stock = step.stock_status.get(reactant)
        if in_stock is None and stock_checker is not None:
            try:
                in_stock = bool(stock_checker(reactant))
                step.stock_status[reactant] = in_stock
            except Exception:
                in_stock = None
        state.stock_status[reactant] = in_stock
        if not in_stock:
            next_open.append(reactant)
    next_open = _dedupe(next_open)
    state.reaction_graph.setdefault("edges", []).append(
        {
            "product": step.product_smiles,
            "reactants": list(step.reactant_smiles),
            "rxn_smiles": step.rxn_smiles,
            "source": step.source_model,
        }
    )
    state.append_step(step, opened_leaves=next_open)


def _apply_stage_transition(state: CascadeProgramState, transition: StageTransition) -> None:
    state.add_stage_transition(transition)
    failure = transition.metadata.get("failure") if transition.metadata else {}
    step_index = failure.get("step_index") if isinstance(failure, dict) else None
    if step_index is None and isinstance(failure, dict):
        step_index = (failure.get("metadata") or {}).get("right_step_index")
    if step_index is None:
        return
    try:
        idx = int(step_index)
    except (TypeError, ValueError):
        return
    if idx < 0 or idx >= len(state.step_annotations):
        return
    for stage in state.stage_graph.stages:
        if idx in stage.step_indices:
            stage.step_indices = [value for value in stage.step_indices if value != idx]
    step = state.step_annotations[idx]
    step.stage_id = transition.to_stage_id
    state.stage_graph.add_step(transition.to_stage_id, idx, step)
    state.stage_partition = state.stage_graph.to_partition(len(state.step_annotations))


def _apply_evidence_retrieval(state: CascadeProgramState, action: CascadeAction) -> None:
    failure = action.evidence_payload.get("failure") or {}
    idx = failure.get("step_index")
    if idx is None:
        payload = dict(action.evidence_payload or {})
        if payload:
            payload.setdefault("target_leaf", action.target_leaf)
            payload.setdefault("source", action.source)
            state.raw_metadata.setdefault("cascade_subgoal_hints", []).append(payload)
        return
    try:
        step = state.step_annotations[int(idx)]
    except (IndexError, TypeError, ValueError):
        return
    step.raw_metadata["evidence_retrieval"] = action.evidence_payload
    if step.evidence_confidence is None:
        step.evidence_confidence = 0.5
    if step.is_enzymatic and not step.ec_numbers:
        step.ec_numbers.append("retrieved_ec_pending")


def _call_provider(provider: Any, request: ProposalRequest) -> list[Any]:
    try:
        return list(provider.propose(request) or [])
    except TypeError:
        try:
            return list(provider.propose(request.leaf_smiles, request.state, top_k=request.top_k) or [])
        except TypeError:
            return list(provider.propose(request.leaf_smiles, top_k=request.top_k) or [])


def _dedupe_actions(actions: list[CascadeAction]) -> list[CascadeAction]:
    out: list[CascadeAction] = []
    seen: set[str] = set()
    for action in actions:
        key = action.step.rxn_smiles if action.step is not None else json.dumps(action.to_dict(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _is_sidecar_evidence_action(action: CascadeAction) -> bool:
    return (
        action.action_type == CascadeActionType.EVIDENCE_RETRIEVAL
        and str((action.evidence_payload or {}).get("contract") or "").startswith("learned subgoal evidence hint")
    )


def _state_signature(state: CascadeProgramState) -> str:
    payload = {
        "open": sorted(state.open_molecule_leaves),
        "steps": [step.rxn_smiles for step in state.step_annotations],
        "stages": state.stage_partition,
        "cofactors": state.cofactor_ledger.to_dict(),
        "subgoal_hints": sorted(
            str(hint.get("subgoal_hint_id") or hint.get("evidence_id") or "")
            for hint in state.raw_metadata.get("cascade_subgoal_hints", [])
            if isinstance(hint, dict)
        ),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _mark_last_action_enqueued(child: CascadeProgramState, enqueued: bool) -> None:
    actions = child.raw_metadata.get("applied_actions")
    if not actions:
        return
    try:
        actions[-1].setdefault("metadata", {})["enqueued_from_state"] = bool(enqueued)
    except Exception:
        return


def _child_open_leaf_signature(children: Any) -> set[str]:
    out: set[str] = set()
    for child in children:
        for leaf in child.open_molecule_leaves or child.open_leaves:
            if not child.stock_status.get(leaf):
                out.add(str(leaf))
    return out


def _state_stock_closed(state: CascadeProgramState) -> bool:
    open_leaves = list(state.open_molecule_leaves or state.open_leaves or [])
    return all(bool(state.stock_status.get(leaf)) for leaf in open_leaves)


def _state_no_failures(state: CascadeProgramState) -> bool:
    return not bool(state.unresolved_failure_modes)


def _learned_value_active(controller: CascadeSearchController | None) -> bool:
    if controller is None:
        return False
    return type(controller.value_model).__name__ == "LearnedCascadeValueModel"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def plan_cascade_program(
    *,
    target_smiles: str,
    proposal_providers: list[Any] | None = None,
    stock_checker: StockChecker | None = None,
    config: CascadeSearchConfig | None = None,
    controller: CascadeSearchController | None = None,
    trace_collector: CascadeTraceCollector | None = None,
    n_results: int = 5,
) -> list[CascadeSearchResult]:
    planner = CascadeProgramSearch(
        proposal_providers=proposal_providers,
        stock_checker=stock_checker,
        config=config,
        controller=controller,
        trace_collector=trace_collector,
    )
    return planner.search(target_smiles, n_results=n_results)


def default_cascade_search_controller(
    *,
    value_model: Any | None = None,
    transition_value_model: Any | None = None,
    pair_scorer: Any | None = None,
) -> CascadeSearchController:
    return CascadeSearchController(
        value_model=value_model or HeuristicCascadeValueModel(),
        source_policy=CascadeSourcePolicy(),
        enzyme_module_ranker=EnzymeModuleRanker(),
        condition_transition_model=ConditionTransitionModel(),
        cofactor_closure_model=CofactorClosureModel(),
        transition_value_model=transition_value_model,
        pair_scorer=pair_scorer,
        metadata={"controller_version": "cascade_native_default"},
    )
