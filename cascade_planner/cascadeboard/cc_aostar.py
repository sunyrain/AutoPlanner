"""Cascade-constrained AO*-style search for CascadeBoard.

This module keeps the current skeleton-first interface but replaces greedy or
slot-beam fill with a best-first AND-OR graph expansion over molecule and
reaction nodes. It is intentionally conservative: generators still own reaction
facts, while conditions, stock, evidence, and fixed anchors only affect search
priority and pruning.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

from cascade_planner.cascadeboard import CascadeBoard, RouteExplanation, RouteResult
from cascade_planner.cascadeboard.candidate_ranker import candidate_ranker_weight, default_candidate_ranker
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.cascadeboard.route_export import route_metrics, route_naturalness_metrics
from cascade_planner.cascadeboard.skeleton_planner import (
    RouteSkeleton,
    _candidates_for_skeleton_slot,
    _fill_slot_from_candidate,
)
from cascade_planner.cascadeboard.value_function import RouteValueFunction, default_value_function


@dataclass
class MoleculeNode:
    smiles: str
    depth: int
    in_stock: bool = False
    expanded: bool = False
    children: list[int] = field(default_factory=list)


@dataclass
class ReactionNode:
    id: int
    product: str
    reactants: tuple[str, ...]
    step_index: int
    candidate: dict[str, Any]
    score: float


@dataclass
class SearchStats:
    expansions: int = 0
    generated_reactions: int = 0
    pruned_by_anchor: int = 0
    pruned_by_node_constraint: int = 0
    pruned_by_route_quality: int = 0
    candidate_cache_hits: int = 0
    completed_routes: int = 0
    molecule_nodes: int = 0
    reaction_nodes: int = 0
    progressive_repairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expansions": self.expansions,
            "generated_reactions": self.generated_reactions,
            "pruned_by_anchor": self.pruned_by_anchor,
            "pruned_by_node_constraint": self.pruned_by_node_constraint,
            "pruned_by_route_quality": self.pruned_by_route_quality,
            "candidate_cache_hits": self.candidate_cache_hits,
            "completed_routes": self.completed_routes,
            "molecule_nodes": self.molecule_nodes,
            "reaction_nodes": self.reaction_nodes,
            "progressive_repairs": self.progressive_repairs,
        }


@dataclass
class _State:
    board: CascadeBoard
    skeleton: RouteSkeleton
    next_index: int
    current_product: str
    score: float
    skeleton_log_prob: float
    visited_nodes: frozenset[str] = field(default_factory=frozenset)


class CascadeAStarPlanner:
    """Best-first cascade search with AND-OR graph bookkeeping."""

    def __init__(
        self,
        *,
        retro_engine: dict | None,
        stock_checker=None,
        candidate_budget: int = 8,
        expansion_budget: int = 50,
        constraints: dict[str, Any] | None = None,
        energy_api: EnergyAPI | None = None,
        value_function: RouteValueFunction | None = None,
    ):
        self.retro_engine = retro_engine
        self.stock_checker = stock_checker
        self.candidate_budget = max(1, int(candidate_budget))
        self.expansion_budget = max(1, int(expansion_budget))
        self.constraints = constraints or {}
        self.energy_api = energy_api or EnergyAPI()
        self.value_function = value_function or default_value_function()
        self.stats = SearchStats()
        self.molecule_nodes: dict[tuple[str, int], MoleculeNode] = {}
        self.reaction_nodes: list[ReactionNode] = []
        self._anchors = _fixed_steps_by_index(self.constraints)
        self._terminal_main_materials = _terminal_main_materials(self.constraints)
        self._required_intermediates = _required_intermediates(self.constraints)
        self._forbidden_intermediates = _forbidden_intermediates(self.constraints)
        self._strict_skeleton = bool(self.constraints.get("strict_skeleton", False))
        self._candidate_cache: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}

    def search(
        self,
        target: str,
        skeletons: list[RouteSkeleton],
        *,
        n_results: int = 5,
    ) -> list[RouteResult]:
        queue: list[tuple[float, int, _State]] = []
        counter = itertools.count()
        for skeleton in skeletons:
            board = CascadeBoard.from_n_steps(skeleton.n_steps, target)
            log_prob = float(getattr(skeleton, "log_prob", 0.0) or 0.0)
            state = _State(
                board=board,
                skeleton=skeleton,
                next_index=0,
                current_product=target,
                score=log_prob,
                skeleton_log_prob=log_prob,
                visited_nodes=frozenset(x for x in {canonical_smiles(target)} if x),
            )
            heapq.heappush(queue, (-_priority(state), next(counter), state))

        completed: list[RouteResult] = []
        seen_routes: set[tuple[tuple[str, str], ...]] = set()

        while queue and self.stats.expansions < self.expansion_budget:
            _, _, state = heapq.heappop(queue)
            if state.next_index >= state.skeleton.n_steps:
                if not self._state_satisfies_global_node_constraints(state):
                    self.stats.pruned_by_node_constraint += 1
                    continue
                result = self._finalize_state(state)
                sig = tuple(
                    (slot.reaction_smiles or "", slot.main_reactant or "")
                    for slot in result.board.slots
                )
                if sig not in seen_routes:
                    seen_routes.add(sig)
                    completed.append(result)
                    self.stats.completed_routes += 1
                continue

            self.stats.expansions += 1
            if not self._state_passes_step_anchor(state):
                self.stats.pruned_by_anchor += 1
                continue
            if not self._state_passes_node_constraints(state):
                self.stats.pruned_by_node_constraint += 1
                continue
            mol_key = (state.current_product, state.next_index)
            mol_node = self.molecule_nodes.setdefault(
                mol_key,
                MoleculeNode(
                    smiles=state.current_product,
                    depth=state.next_index,
                    in_stock=_is_stock(state.current_product, self.stock_checker),
                ),
            )
            mol_node.expanded = True

            cands = self._expand_candidates(state)
            if not cands:
                dead = self._advance_dead_end(state)
                heapq.heappush(queue, (-_priority(dead), next(counter), dead))
                continue

            for cand in cands:
                if not _passes_anchor(cand, state.next_index, self._anchors):
                    self.stats.pruned_by_anchor += 1
                    continue
                if not self._candidate_passes_node_constraints(state, cand):
                    self.stats.pruned_by_node_constraint += 1
                    continue
                if not self._candidate_passes_route_quality(state, cand):
                    self.stats.pruned_by_route_quality += 1
                    continue
                child = self._advance_with_candidate(state, cand, cands)
                reactants = tuple(_candidate_reactants(cand))
                rxn_node = ReactionNode(
                    id=len(self.reaction_nodes),
                    product=state.current_product,
                    reactants=reactants,
                    step_index=state.next_index,
                    candidate=dict(cand),
                    score=child.score - state.score,
                )
                self.reaction_nodes.append(rxn_node)
                mol_node.children.append(rxn_node.id)
                self.stats.generated_reactions += 1
                heapq.heappush(queue, (-_priority(child), next(counter), child))

        self.stats.molecule_nodes = len(self.molecule_nodes)
        self.stats.reaction_nodes = len(self.reaction_nodes)
        completed.sort(key=lambda r: self._control_score(r), reverse=True)
        return completed[:n_results]

    def _expand_candidates(self, state: _State) -> list[dict[str, Any]]:
        idx = state.next_index
        skeleton = state.skeleton
        primary = self._fetch_candidates(
            product_smiles=state.current_product,
            ec1=skeleton.ec1s[idx] if idx < len(skeleton.ec1s) else 0,
            skel_type=skeleton.types[idx] if idx < len(skeleton.types) else "",
            top_k=self.candidate_budget,
        )
        if self._strict_skeleton:
            return _annotate_vnext_pool(state.current_product, primary, stock_checker=self.stock_checker)

        # Keep the skeleton as a global prior, not a hard gate. A skeleton can
        # misclassify a slot as enzymatic/chemical; a source-diverse fallback
        # keeps chemically plausible disconnections available for the search
        # controller to score against stock, anchors, and cascade constraints.
        fallback_budget = max(2, self.candidate_budget // 2)
        fallback = self._fetch_candidates(
            product_smiles=state.current_product,
            ec1=0,
            skel_type="",
            top_k=fallback_budget,
        )
        source_priority = _source_priority_for_slot(skeleton.ec1s[idx] if idx < len(skeleton.ec1s) else 0)
        merged = _merge_candidate_lists(primary, fallback, top_k=self.candidate_budget, source_priority=source_priority)
        annotated = _annotate_vnext_pool(state.current_product, merged, stock_checker=self.stock_checker)
        annotated = _progressive_first_candidates(state.current_product, annotated, top_k=self.candidate_budget, stock_checker=self.stock_checker)
        if _progressive_candidate_count(state.current_product, annotated) >= max(1, self.candidate_budget // 2):
            return annotated
        repaired = self._progressive_repair_candidates(state, source_priority=source_priority)
        return repaired or annotated

    def _progressive_repair_candidates(
        self,
        state: _State,
        *,
        source_priority: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        idx = state.next_index
        skeleton = state.skeleton
        repair_budget = max(self.candidate_budget * 4, self.candidate_budget + 12, 16)
        primary = self._fetch_candidates(
            product_smiles=state.current_product,
            ec1=skeleton.ec1s[idx] if idx < len(skeleton.ec1s) else 0,
            skel_type=skeleton.types[idx] if idx < len(skeleton.types) else "",
            top_k=repair_budget,
        )
        fallback = self._fetch_candidates(
            product_smiles=state.current_product,
            ec1=0,
            skel_type="",
            top_k=max(4, repair_budget // 2),
        )
        widened = _merge_candidate_lists(primary, fallback, top_k=repair_budget, source_priority=source_priority)
        widened = _annotate_vnext_pool(state.current_product, widened, stock_checker=self.stock_checker)
        progressive = [cand for cand in widened if _candidate_progress_delta(state.current_product, cand) >= 0.05]
        if not progressive:
            return []
        self.stats.progressive_repairs += 1
        return _progressive_first_candidates(
            state.current_product,
            _merge_candidate_lists(progressive, widened, top_k=repair_budget, source_priority=source_priority),
            top_k=self.candidate_budget,
            stock_checker=self.stock_checker,
        )

    def _fetch_candidates(
        self,
        *,
        product_smiles: str,
        ec1: int,
        skel_type: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        key = (canonical_smiles(product_smiles), int(ec1 or 0), str(skel_type or ""), int(top_k))
        cached = self._candidate_cache.get(key)
        if cached is not None:
            self.stats.candidate_cache_hits += 1
            return list(cached)
        rows = _candidates_for_skeleton_slot(
            self.retro_engine,
            product_smiles=product_smiles,
            ec1=ec1,
            skel_type=skel_type,
            top_k=top_k,
        )
        self._candidate_cache[key] = list(rows)
        return rows

    def _advance_with_candidate(
        self,
        state: _State,
        cand: dict[str, Any],
        candidate_pool: list[dict[str, Any]],
    ) -> _State:
        idx = state.next_index
        board = state.board.copy()
        slot = board.slots[idx]
        slot.product = state.current_product
        slot.reaction_type = state.skeleton.types[idx]
        slot.T = state.skeleton.Ts[idx]
        slot.pH = state.skeleton.pHs[idx]
        ec1 = state.skeleton.ec1s[idx]
        slot.candidates = list(candidate_pool)
        _fill_slot_from_candidate(slot, cand, cand.get("source") or "candidate")
        if not slot.ec and ec1 > 0 and _is_enzymatic_candidate(cand):
            slot.ec = f"{ec1}.x"
        if not cand.get("type") and not _is_enzymatic_candidate(cand) and ec1 > 0:
            slot.reaction_type = "other"
        score = state.score + _candidate_score(
            board=board,
            step_index=idx,
            candidate=cand,
            skeleton=state.skeleton,
            visited_nodes=state.visited_nodes,
            stock_checker=self.stock_checker,
            value_function=self.value_function,
        )
        score += self._node_constraint_score(state, cand)
        visited = set(state.visited_nodes)
        for smi in [state.current_product, cand.get("main_reactant", "")]:
            can = canonical_smiles(smi)
            if can:
                visited.add(can)
        for smi in _candidate_reactants(cand):
            can = canonical_smiles(smi)
            if can:
                visited.add(can)
        return _State(
            board=board,
            skeleton=state.skeleton,
            next_index=idx + 1,
            current_product=cand.get("main_reactant", ""),
            score=score,
            skeleton_log_prob=state.skeleton_log_prob,
            visited_nodes=frozenset(visited),
        )

    def _advance_dead_end(self, state: _State) -> _State:
        idx = state.next_index
        board = state.board.copy()
        slot = board.slots[idx]
        slot.product = state.current_product
        slot.reaction_type = state.skeleton.types[idx]
        slot.T = state.skeleton.Ts[idx]
        slot.pH = state.skeleton.pHs[idx]
        ec1 = state.skeleton.ec1s[idx]
        if ec1 > 0:
            slot.ec = f"{ec1}.x"
        return _State(
            board=board,
            skeleton=state.skeleton,
            next_index=idx + 1,
            current_product="",
            score=state.score - 20.0,
            skeleton_log_prob=state.skeleton_log_prob,
            visited_nodes=state.visited_nodes,
        )

    def _state_passes_step_anchor(self, state: _State) -> bool:
        anchor = self._anchors.get(state.next_index)
        if not anchor or not anchor.get("product"):
            return True
        return canonical_smiles(state.current_product) == canonical_smiles(anchor.get("product"))

    def _state_passes_node_constraints(self, state: _State) -> bool:
        current = canonical_smiles(state.current_product)
        return not (current and current in self._forbidden_intermediates)

    def _candidate_passes_node_constraints(self, state: _State, cand: dict[str, Any]) -> bool:
        reactants = _candidate_reactant_set(cand)
        if reactants & self._forbidden_intermediates:
            return False
        if (
            self._terminal_main_materials
            and state.next_index == state.skeleton.n_steps - 1
            and canonical_smiles(cand.get("main_reactant")) not in self._terminal_main_materials
        ):
            return False
        return True

    def _candidate_passes_route_quality(self, state: _State, cand: dict[str, Any]) -> bool:
        current = canonical_smiles(state.current_product)
        main = canonical_smiles(cand.get("main_reactant"))
        if not current or not main:
            return False
        if main == current:
            return False
        if main in state.visited_nodes:
            return False
        if not _candidate_product_matches_current(cand, state.current_product):
            return False
        if not _candidate_atom_balance_ok(cand, state.current_product):
            return False
        return True

    def _state_satisfies_global_node_constraints(self, state: _State) -> bool:
        if self._required_intermediates and not self._required_intermediates.issubset(state.visited_nodes):
            return False
        if self._forbidden_intermediates and self._forbidden_intermediates & state.visited_nodes:
            return False
        if self._terminal_main_materials:
            terminal = canonical_smiles(state.board.slots[-1].main_reactant if state.board.slots else "")
            if terminal not in self._terminal_main_materials:
                return False
        return True

    def _node_constraint_score(self, state: _State, cand: dict[str, Any]) -> float:
        score = 0.0
        main = canonical_smiles(cand.get("main_reactant"))
        product = canonical_smiles(state.current_product)
        if main and main in self._required_intermediates and main not in state.visited_nodes:
            score += 8.0
        if product and product in self._required_intermediates and product not in state.visited_nodes:
            score += 3.0
        if (
            self._terminal_main_materials
            and state.next_index == state.skeleton.n_steps - 1
            and main in self._terminal_main_materials
        ):
            score += 10.0
        return score

    def _finalize_state(self, state: _State) -> RouteResult:
        board = state.board
        energy = self.energy_api.compute_energy(board, None)
        quality = self.energy_api.compute_quality(board)
        risk = self.energy_api.compute_risk(board)
        idx, reason = self.energy_api.diagnose_bottleneck(board)
        score = state.score - float(energy)
        skel = state.skeleton
        explanation = RouteExplanation(
            why_selected=f"CC-AO* over skeleton: {' -> '.join(skel.types)}",
            constraints_satisfied={"anchors": "satisfied" if _board_satisfies_anchors(board, self._anchors) else "VIOLATED"},
            global_condition_window=f"T: {min(skel.Ts):.0f}-{max(skel.Ts):.0f} C, pH: {min(skel.pHs):.1f}-{max(skel.pHs):.1f}",
            uncertainty_table={
                "search_mode": "cc_aostar",
                "skeleton_log_prob": state.skeleton_log_prob,
                "route_value_score": self.value_function.score_board(board, stock_checker=self.stock_checker).score,
                "required_intermediates": sorted(self._required_intermediates),
                "terminal_main_materials": sorted(self._terminal_main_materials),
                **self.stats.to_dict(),
            },
        )
        return RouteResult(
            board=board,
            quality_vector=quality,
            risk_vector=risk,
            score=score,
            confidence=0.8 if skel.compatibility == "empirically_compatible" else 0.45,
            constraint_report={"search_mode": "cc_aostar"},
            bottleneck_slot=idx,
            bottleneck_reason=reason,
            explanation=explanation,
        )

    def _control_score(self, result: RouteResult) -> float:
        """Final deterministic route-control score for completed candidates."""
        metrics = route_metrics(result.board, stock_checker=self.stock_checker)
        cond = metrics.get("condition") or {}
        compat = metrics.get("cascade_compatibility") or {}
        enz = metrics.get("enzyme_evidence") or {}
        natural = metrics.get("route_naturalness") or {}
        progress = metrics.get("retrosynthesis_progress") or {}
        score = float(result.score or 0.0)
        score += 15.0 * float(bool(metrics.get("filled_route")))
        score += 160.0 * float(bool(metrics.get("route_solved")))
        score += 100.0 * float(bool(metrics.get("progressive_route")))
        score += 110.0 * float(progress.get("main_chain_reduction") or 0.0)
        score += 35.0 * float(bool(progress.get("terminal_simplified")))
        if metrics.get("filled_route") and not metrics.get("progressive_route"):
            score -= 90.0
        if not metrics.get("progressive_route") and float(progress.get("main_chain_reduction") or 0.0) < 0.15:
            score -= 60.0
        if metrics.get("strict_stock_solve") is True:
            score += 80.0
        elif metrics.get("strict_stock_solve") is False:
            score -= 40.0
        score += 40.0 * float(bool(cond.get("condition_window_success")))
        score += 40.0 * float(bool(compat.get("cascade_compatibility_success")))
        score += 35.0 * float(natural.get("naturalness_score") or 0.0)
        score -= 25.0 * int(natural.get("self_loop_steps") or 0)
        score -= 20.0 * int(natural.get("repeated_main_reactants") or 0)
        score -= 20.0 * int(natural.get("product_mismatch_steps") or 0)
        score -= 10.0 * len(compat.get("issues") or [])
        if enz.get("enzyme_evidence_score") is not None:
            score += 10.0 * float(enz["enzyme_evidence_score"])
        score += 12.0 * self.value_function.score_board(result.board, stock_checker=self.stock_checker).score
        return score


def plan_with_cc_aostar(
    *,
    target: str,
    skeletons: list[RouteSkeleton],
    retro_engine: dict | None,
    n_results: int = 5,
    candidate_budget: int = 8,
    expansion_budget: int = 50,
    stock_checker=None,
    constraints: dict[str, Any] | None = None,
    energy_api: EnergyAPI | None = None,
    value_function: RouteValueFunction | None = None,
) -> list[RouteResult]:
    planner = CascadeAStarPlanner(
        retro_engine=retro_engine,
        stock_checker=stock_checker,
        candidate_budget=candidate_budget,
        expansion_budget=expansion_budget,
        constraints=constraints,
        energy_api=energy_api,
        value_function=value_function,
    )
    return planner.search(target, skeletons, n_results=n_results)


def _priority(state: _State) -> float:
    remaining = max(0, state.skeleton.n_steps - state.next_index)
    return state.score - 0.25 * remaining


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_score(
    *,
    board: CascadeBoard,
    step_index: int,
    candidate: dict[str, Any],
    skeleton: RouteSkeleton,
    visited_nodes: frozenset[str] = frozenset(),
    stock_checker=None,
    value_function: RouteValueFunction | None = None,
) -> float:
    score = _as_float(candidate.get("score"), 0.0)
    if candidate.get("type") and candidate.get("type") == skeleton.types[step_index]:
        score += 2.0
    elif candidate.get("type") and skeleton.types[step_index]:
        score -= 1.5
    elif not candidate.get("type") and skeleton.ec1s[step_index] > 0 and not _is_enzymatic_candidate(candidate):
        score -= 2.0

    evidence = candidate.get("evidence") or {}
    source = candidate.get("source") or ""
    if _is_enzymatic_candidate(candidate):
        score += 0.8
    if source == "v3_retrieval" and candidate.get("ec"):
        score += 1.0
    elif source == "enzyformer" and candidate.get("ec"):
        score += 0.5
    elif source == "enzexpand" and candidate.get("ec"):
        score += 0.3
    if candidate.get("ec") or candidate.get("uniprot_accession") or evidence.get("uniprot_accession"):
        score += 0.5
    if evidence.get("literature_precedent") or evidence.get("doi") or candidate.get("doi"):
        score += 0.4
    if evidence.get("cofactor") or candidate.get("cofactor"):
        score += 0.2
    if candidate.get("T") is not None and candidate.get("pH") is not None:
        score += 0.4
    if candidate.get("catalyst"):
        score += 0.2
    if evidence.get("condition_match") or candidate.get("condition_match"):
        score += 0.2
    if skeleton.ec1s[step_index] > 0 and source == "retrochimera" and not candidate.get("ec"):
        score -= 1.5

    product_atoms = _heavy_atoms(board.slots[step_index].product)
    main_atoms = _heavy_atoms(candidate.get("main_reactant"))
    reactant_atoms = _candidate_reactant_heavy_atoms(candidate)
    if product_atoms and reactant_atoms:
        ratio = reactant_atoms / max(product_atoms, 1)
        if ratio < 0.45:
            score -= 8.0
        elif 0.7 <= ratio <= 1.8:
            score += 0.8
        elif ratio > 2.5:
            score -= min((ratio - 2.5), 3.0)
    if product_atoms and main_atoms and product_atoms > 12 and main_atoms / product_atoms < 0.25:
        score -= 3.0
    if product_atoms and main_atoms and main_atoms > product_atoms:
        growth_fraction = (main_atoms - product_atoms) / max(product_atoms, 1)
        score -= 8.0 * min(growth_fraction, 2.0)
        if main_atoms > product_atoms * 1.3:
            score -= 6.0
    progress_delta = _candidate_progress_delta(board.slots[step_index].product, candidate)
    candidate["progress_delta"] = round(progress_delta, 4)
    if progress_delta >= 0.35:
        score += 7.0
    elif progress_delta >= 0.15:
        score += 4.0
    elif progress_delta >= 0.05:
        score += 1.5
    elif product_atoms > 8:
        score -= 4.0
    if progress_delta < -0.25:
        score -= 30.0
    elif progress_delta < -0.05:
        score -= 12.0

    if not _candidate_product_matches_current(candidate, board.slots[step_index].product):
        score -= 6.0
    if not _candidate_atom_balance_ok(candidate, board.slots[step_index].product):
        score -= 8.0
    main = canonical_smiles(candidate.get("main_reactant"))
    if main and main in visited_nodes:
        score -= 8.0
    if _route_has_repeated_reaction(board):
        score -= 8.0
    natural = route_naturalness_metrics(board)
    score -= 5.0 * int(natural.get("self_loop_steps") or 0)
    score -= 3.0 * int(natural.get("repeated_main_reactants") or 0)
    score -= 4.0 * int(natural.get("product_mismatch_steps") or 0)

    if step_index > 0:
        prev = board.slots[step_index - 1]
        cur = board.slots[step_index]
        if prev.T is not None and cur.T is not None:
            score -= min(abs(float(prev.T) - float(cur.T)) / 30.0, 2.0)
        if prev.pH is not None and cur.pH is not None:
            score -= min(abs(float(prev.pH) - float(cur.pH)) / 3.0, 2.0)

    if stock_checker and step_index == skeleton.n_steps - 1:
        reactants = _candidate_reactants(candidate)
        if reactants and all(bool(stock_checker(x)) for x in reactants):
            score += 6.0
        elif reactants:
            score -= 2.0
    if value_function is not None:
        value = value_function.score_candidate(board.slots[step_index].product, candidate, stock_checker=stock_checker)
        score += 1.5 * value.score
        candidate["value_score"] = value.score
        candidate["value_probability"] = value.probability
    ranker = default_candidate_ranker()
    if ranker is not None:
        ranker_score = ranker.score_candidate(board.slots[step_index].product, candidate, stock_checker=stock_checker)
        score += candidate_ranker_weight() * (ranker_score - 0.5)
        candidate["candidate_ranker_score"] = ranker_score
    vnext_score = candidate.get("vnext_candidate_score")
    if vnext_score is not None:
        try:
            from cascade_planner.vnext.runtime import vnext_candidate_weight

            center = _as_float(candidate.get("vnext_pool_mean_score"), 0.5)
            score += vnext_candidate_weight() * (_as_float(vnext_score) - center)
        except Exception:
            pass
    return score


def _annotate_vnext_pool(product: str, candidates: list[dict[str, Any]], *, stock_checker=None) -> list[dict[str, Any]]:
    try:
        from cascade_planner.vnext.runtime import default_vnext_runtime

        vnext = default_vnext_runtime()
        if vnext is not None:
            return vnext.annotate_candidate_pool(product, candidates, stock_checker=stock_checker)
    except Exception:
        pass
    return candidates


def _is_enzymatic_candidate(candidate: dict[str, Any]) -> bool:
    source = candidate.get("source") or ""
    evidence = candidate.get("evidence") or {}
    return bool(
        candidate.get("ec")
        or candidate.get("enzyme_uid")
        or evidence.get("uniprot_accession")
        or source in {"enzyformer", "v3_retrieval", "enzexpand", "enzymatic"}
    )


def _reaction_products(rxn_smiles: str | None) -> set[str]:
    if not rxn_smiles or ">>" not in rxn_smiles:
        return set()
    rhs = rxn_smiles.split(">>", 1)[1]
    out: set[str] = set()
    for part in rhs.split("."):
        can = canonical_smiles(part.strip())
        if can:
            out.add(can)
    return out


def _candidate_product_matches_current(candidate: dict[str, Any], current_product: str | None) -> bool:
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if not rxn:
        return True
    products = _reaction_products(rxn)
    current = canonical_smiles(current_product)
    return not products or not current or current in products


def _heavy_atoms(smiles: str | None) -> int:
    if not smiles:
        return 0
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return 0
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _candidate_reactants(candidate: dict[str, Any]) -> list[str]:
    reactants: list[str] = []
    main = candidate.get("main_reactant")
    if main:
        reactants.append(str(main))
    reactants.extend(str(smi) for smi in candidate.get("aux_reactants") or [] if smi)
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if rxn and ">>" in rxn:
        lhs = rxn.split(">>", 1)[0]
        reactants.extend(part.strip() for part in lhs.split(".") if part.strip())
    out: list[str] = []
    seen: set[str] = set()
    for smi in reactants:
        can = canonical_smiles(smi)
        key = can or smi
        if key in seen:
            continue
        seen.add(key)
        out.append(smi)
    return out


def _candidate_reactant_heavy_atoms(candidate: dict[str, Any]) -> int:
    return sum(_heavy_atoms(smi) for smi in _candidate_reactants(candidate))


def _candidate_progress_delta(product_smiles: str | None, candidate: dict[str, Any]) -> float:
    product_atoms = _heavy_atoms(product_smiles)
    main_atoms = _heavy_atoms(candidate.get("main_reactant"))
    if product_atoms <= 0 or main_atoms <= 0:
        return 0.0
    return (product_atoms - main_atoms) / max(product_atoms, 1)


def _has_progressive_candidate(product_smiles: str | None, candidates: list[dict[str, Any]]) -> bool:
    return any(_candidate_progress_delta(product_smiles, cand) >= 0.05 for cand in candidates)


def _progressive_candidate_count(product_smiles: str | None, candidates: list[dict[str, Any]]) -> int:
    return sum(1 for cand in candidates if _candidate_progress_delta(product_smiles, cand) >= 0.05)


def _progressive_first_candidates(
    product_smiles: str | None,
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    stock_checker=None,
) -> list[dict[str, Any]]:
    rows = list(candidates)
    has_progressive = _has_progressive_candidate(product_smiles, rows)

    def key(cand: dict[str, Any]) -> tuple[float, ...]:
        delta = _candidate_progress_delta(product_smiles, cand)
        severe_growth = 1.0 if has_progressive and delta < -0.25 else 0.0
        return (
            -severe_growth,
            1.0 if delta >= 0.05 else 0.0,
            delta,
            _stock_bonus(cand, stock_checker),
            _as_float(cand.get("vnext_candidate_score"), 0.0),
            _as_float(cand.get("score"), 0.0),
        )

    rows.sort(key=key, reverse=True)
    out = rows[:top_k]
    for idx, cand in enumerate(out, start=1):
        cand["rank"] = idx
    return out


def _stock_bonus(candidate: dict[str, Any], stock_checker=None) -> float:
    if not stock_checker:
        return 0.0
    reactants = _candidate_reactants(candidate)
    if not reactants:
        return 0.0
    hits = 0
    for smi in reactants:
        try:
            hits += int(bool(stock_checker(smi)))
        except Exception:
            pass
    return hits / max(len(reactants), 1)


def _candidate_atom_balance_ok(candidate: dict[str, Any], current_product: str | None) -> bool:
    product_atoms = _heavy_atoms(current_product)
    if product_atoms <= 10:
        return True
    reactant_atoms = _candidate_reactant_heavy_atoms(candidate)
    if reactant_atoms == 0:
        return False
    return reactant_atoms >= max(4, int(product_atoms * 0.45))


def _route_has_repeated_reaction(board: CascadeBoard) -> bool:
    seen: set[str] = set()
    for slot in board.slots:
        key = canonical_reaction(slot.reaction_smiles)
        if not key:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def _is_stock(smiles: str, stock_checker=None) -> bool:
    if not stock_checker or not smiles:
        return False
    try:
        return bool(stock_checker(smiles))
    except Exception:
        return False


def _fixed_steps_by_index(constraints: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    fixed: dict[int, dict[str, Any]] = {}
    for item in (constraints or {}).get("fixed_steps", []) or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        values = item.get("values") or {}
        if isinstance(values, dict):
            fixed[idx] = dict(values)
    return fixed


def _as_sequence(value: Any) -> list[Any]:
    if value in (None, "", [], {}, ()):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _canonical_smiles_set(*values: Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        for item in _as_sequence(value):
            can = canonical_smiles(str(item))
            if can:
                out.add(can)
    return out


def _terminal_main_materials(constraints: dict[str, Any] | None) -> set[str]:
    raw = constraints or {}
    return _canonical_smiles_set(
        raw.get("starting_material"),
        raw.get("starting_materials"),
        raw.get("allowed_starting_materials"),
        raw.get("terminal_main_reactants"),
    )


def _required_intermediates(constraints: dict[str, Any] | None) -> set[str]:
    raw = constraints or {}
    required = _canonical_smiles_set(
        raw.get("known_intermediate"),
        raw.get("known_intermediates"),
        raw.get("required_intermediate"),
        raw.get("required_intermediates"),
        raw.get("waypoints"),
    )
    for anchor in _fixed_steps_by_index(raw).values():
        if anchor.get("product"):
            required.update(_canonical_smiles_set(anchor.get("product")))
        if anchor.get("main_reactant"):
            required.update(_canonical_smiles_set(anchor.get("main_reactant")))
    return required


def _forbidden_intermediates(constraints: dict[str, Any] | None) -> set[str]:
    raw = constraints or {}
    return _canonical_smiles_set(
        raw.get("forbidden_intermediate"),
        raw.get("forbidden_intermediates"),
        raw.get("exclude_intermediate"),
        raw.get("exclude_intermediates"),
    )


def _candidate_key(candidate: dict[str, Any]) -> str:
    return (
        canonical_reaction(candidate.get("rxn_smiles") or candidate.get("reaction_smiles"))
        or canonical_smiles(candidate.get("main_reactant"))
        or repr(candidate)
    )


def _candidate_source(candidate: dict[str, Any]) -> str:
    return str(candidate.get("source") or candidate.get("enzyme_source") or "unknown")


def _source_priority_for_slot(ec1: int) -> tuple[str, ...]:
    if int(ec1 or 0) > 0:
        return ("v3_retrieval", "enzyformer", "retrochimera", "enzexpand")
    return ("retrochimera", "v3_retrieval", "enzyformer", "enzexpand")


def _merge_candidate_lists(
    *lists: list[dict[str, Any]],
    top_k: int,
    source_priority: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    interleaved: list[dict[str, Any]] = []
    seen: set[str] = set()
    positions = [0 for _ in lists]
    while len(interleaved) < max(top_k * 3, top_k):
        progressed = False
        for list_idx, rows in enumerate(lists):
            while positions[list_idx] < len(rows):
                cand = rows[positions[list_idx]]
                positions[list_idx] += 1
                key = _candidate_key(cand)
                if key in seen:
                    continue
                seen.add(key)
                interleaved.append(cand)
                progressed = True
                break
            if len(interleaved) >= max(top_k * 3, top_k):
                break
        if not progressed:
            break
    return _source_diverse_merge(interleaved, top_k=top_k, source_priority=source_priority)


def _source_diverse_merge(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    source_priority: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for cand in rows:
        buckets.setdefault(_candidate_source(cand), []).append(cand)
    source_order = [src for src in source_priority if src in buckets]
    for src in buckets:
        if src not in source_order:
            source_order.append(src)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(out) < top_k:
        progressed = False
        for src in source_order:
            bucket = buckets.get(src) or []
            while bucket:
                cand = bucket.pop(0)
                key = _candidate_key(cand)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
                progressed = True
                break
            if len(out) >= top_k:
                break
        if not progressed:
            break
    for idx, cand in enumerate(out, start=1):
        cand.setdefault("rank", idx)
    return out


def _candidate_reactant_set(candidate: dict[str, Any]) -> set[str]:
    return _canonical_smiles_set(_candidate_reactants(candidate))


def _passes_anchor(candidate: dict[str, Any], step_index: int, anchors: dict[int, dict[str, Any]]) -> bool:
    anchor = anchors.get(step_index)
    if not anchor:
        return True
    if anchor.get("reaction_smiles"):
        cand_rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
        if canonical_reaction(cand_rxn) != canonical_reaction(anchor.get("reaction_smiles")):
            return False
    if anchor.get("main_reactant"):
        if canonical_smiles(candidate.get("main_reactant")) != canonical_smiles(anchor.get("main_reactant")):
            return False
    if anchor.get("reaction_type") and candidate.get("type"):
        if candidate.get("type") != anchor.get("reaction_type"):
            return False
    if anchor.get("ec") and candidate.get("ec"):
        wanted = str(anchor.get("ec"))
        got = str(candidate.get("ec"))
        if wanted.endswith(".x"):
            return got.startswith(wanted[:-1])
        return got == wanted
    return True


def _board_satisfies_anchors(board: CascadeBoard, anchors: dict[int, dict[str, Any]]) -> bool:
    for idx, anchor in anchors.items():
        if idx >= board.n_steps:
            return False
        slot = board.slots[idx]
        if anchor.get("reaction_smiles") and canonical_reaction(slot.reaction_smiles) != canonical_reaction(anchor.get("reaction_smiles")):
            return False
        if anchor.get("main_reactant") and canonical_smiles(slot.main_reactant) != canonical_smiles(anchor.get("main_reactant")):
            return False
        if anchor.get("reaction_type") and slot.reaction_type and slot.reaction_type != anchor.get("reaction_type"):
            return False
        if anchor.get("ec") and slot.ec:
            wanted = str(anchor.get("ec"))
            got = str(slot.ec)
            if wanted.endswith(".x") and not got.startswith(wanted[:-1]):
                return False
            if not wanted.endswith(".x") and got != wanted:
                return False
    return True
