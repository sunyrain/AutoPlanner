"""Stock-closed AND-OR retrosynthesis search.

This module is deliberately independent of the linear skeleton-first planner.
It expands every unresolved reactant leaf until all leaves are in stock, known
starting materials, or the depth budget is exhausted. Skeletons can still be
used as soft priors, but they are not required for correctness.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from rdkit import Chem

from cascade_planner.cascadeboard import CascadeBoard, RouteExplanation, RouteResult
from cascade_planner.cascadeboard.candidate_ranker import candidate_ranker_weight, default_candidate_ranker
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.cascadeboard.skeleton_planner import (
    RouteSkeleton,
    _candidates_for_skeleton_slot,
    _fill_slot_from_candidate,
)
from cascade_planner.cascadeboard.value_function import RouteValueFunction, default_value_function


StockChecker = Callable[[str], bool]


@dataclass(frozen=True)
class AndOrStep:
    product: str
    candidate: dict[str, Any]


@dataclass
class AndOrStats:
    expansions: int = 0
    generated_reactions: int = 0
    solved_routes: int = 0
    dead_ends: int = 0
    pruned_cycles: int = 0
    pruned_forbidden: int = 0
    candidate_cache_hits: int = 0
    max_queue_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expansions": self.expansions,
            "generated_reactions": self.generated_reactions,
            "solved_routes": self.solved_routes,
            "dead_ends": self.dead_ends,
            "pruned_cycles": self.pruned_cycles,
            "pruned_forbidden": self.pruned_forbidden,
            "candidate_cache_hits": self.candidate_cache_hits,
            "max_queue_size": self.max_queue_size,
        }


@dataclass(frozen=True)
class _State:
    steps: tuple[AndOrStep, ...]
    open_leaves: tuple[str, ...]
    expanded: frozenset[str]
    score: float
    depth: int


def plan_stock_closed_andor(
    *,
    target: str,
    retro_engine: dict | None,
    stock_checker: StockChecker | None,
    max_depth: int = 6,
    n_results: int = 5,
    branch_factor: int = 8,
    expansion_budget: int = 200,
    skeletons: list[RouteSkeleton] | None = None,
    constraints: dict[str, Any] | None = None,
    value_function: RouteValueFunction | None = None,
) -> list[RouteResult]:
    """Run a best-first stock-closed AND-OR search.

    The search state is a partial reaction hypergraph. At each expansion, it
    picks one unresolved leaf molecule and adds a reaction whose reactants
    become new leaves. A route is solved only when every leaf is stock or an
    explicitly allowed starting material.
    """
    planner = StockClosedAndOrPlanner(
        retro_engine=retro_engine,
        stock_checker=stock_checker,
        max_depth=max_depth,
        branch_factor=branch_factor,
        expansion_budget=expansion_budget,
        skeletons=skeletons or [],
        constraints=constraints or {},
        value_function=value_function,
    )
    return planner.search(target, n_results=n_results)


class StockClosedAndOrPlanner:
    def __init__(
        self,
        *,
        retro_engine: dict | None,
        stock_checker: StockChecker | None,
        max_depth: int,
        branch_factor: int,
        expansion_budget: int,
        skeletons: list[RouteSkeleton],
        constraints: dict[str, Any],
        value_function: RouteValueFunction | None = None,
    ):
        self.retro_engine = retro_engine
        self.stock_checker = stock_checker
        self.max_depth = max(1, int(max_depth))
        self.branch_factor = max(1, int(branch_factor))
        self.expansion_budget = max(1, int(expansion_budget))
        self.skeletons = skeletons
        self.constraints = constraints or {}
        self.allowed_starting_materials = _terminal_materials(self.constraints)
        self.required_intermediates = _canonical_set(
            self.constraints.get("required_intermediate"),
            self.constraints.get("required_intermediates"),
            self.constraints.get("known_intermediate"),
            self.constraints.get("known_intermediates"),
            self.constraints.get("waypoints"),
        )
        self.forbidden_intermediates = _canonical_set(
            self.constraints.get("forbidden_intermediate"),
            self.constraints.get("forbidden_intermediates"),
            self.constraints.get("exclude_intermediate"),
            self.constraints.get("exclude_intermediates"),
        )
        self.strict_skeleton = bool(self.constraints.get("strict_skeleton", False))
        self.value_function = value_function or default_value_function()
        self.stats = AndOrStats()
        self._candidate_cache: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}

    def search(self, target: str, *, n_results: int) -> list[RouteResult]:
        target_can = canonical_smiles(target)
        if not target_can:
            return []
        queue: list[tuple[float, int, _State]] = []
        counter = itertools.count()
        initial = _State(
            steps=(),
            open_leaves=(target,),
            expanded=frozenset(),
            score=0.0,
            depth=0,
        )
        heapq.heappush(queue, (-self._priority(initial), next(counter), initial))
        solved: list[RouteResult] = []
        seen_solved: set[tuple[str, ...]] = set()

        while queue and self.stats.expansions < self.expansion_budget and len(solved) < n_results:
            self.stats.max_queue_size = max(self.stats.max_queue_size, len(queue))
            _, _, state = heapq.heappop(queue)
            if self._is_solved(state):
                if not self._state_satisfies_required(state):
                    self.stats.dead_ends += 1
                    continue
                sig = tuple(canonical_reaction(step.candidate.get("rxn_smiles") or step.candidate.get("reaction_smiles")) for step in state.steps)
                if sig in seen_solved:
                    continue
                seen_solved.add(sig)
                solved.append(self._state_to_result(target, state))
                self.stats.solved_routes += 1
                continue
            if state.depth >= self.max_depth:
                self.stats.dead_ends += 1
                continue

            leaf = self._choose_leaf(state.open_leaves)
            leaf_can = canonical_smiles(leaf)
            if not leaf_can or leaf_can in self.forbidden_intermediates:
                self.stats.pruned_forbidden += 1
                continue
            if leaf_can in state.expanded:
                self.stats.pruned_cycles += 1
                continue

            self.stats.expansions += 1
            for cand in self._expand_candidates(leaf, state.depth):
                if not self._candidate_allowed(cand, state, product=leaf):
                    continue
                reactants = _candidate_reactants(cand)
                if not reactants:
                    continue
                next_open = list(state.open_leaves)
                try:
                    next_open.remove(leaf)
                except ValueError:
                    pass
                next_open.extend(smi for smi in reactants if not self._is_terminal(smi))
                cand_score = _candidate_score(cand, leaf, self.stock_checker, self.value_function)
                next_state = _State(
                    steps=state.steps + (AndOrStep(product=leaf, candidate=dict(cand)),),
                    open_leaves=tuple(_dedupe_smiles(next_open)),
                    expanded=frozenset(set(state.expanded) | {leaf_can}),
                    score=state.score + cand_score,
                    depth=state.depth + 1,
                )
                self.stats.generated_reactions += 1
                heapq.heappush(queue, (-self._priority(next_state), next(counter), next_state))

        solved.sort(key=lambda result: float(result.score or 0.0), reverse=True)
        return solved[:n_results]

    def _expand_candidates(self, product: str, depth: int) -> list[dict[str, Any]]:
        primary: list[dict[str, Any]] = []
        hints = self._skeleton_hints(depth)
        source_priority = _source_priority_for_hints(hints)
        for ec1, skel_type in hints:
            primary.extend(self._fetch_candidates(product, ec1=ec1, skel_type=skel_type, top_k=self.branch_factor))
            if self.strict_skeleton:
                break
        if not self.strict_skeleton:
            primary.extend(self._fetch_candidates(product, ec1=0, skel_type="", top_k=self.branch_factor))
        merged = _merge_candidates(primary, self.branch_factor, source_priority=source_priority)
        condition_hint = self._condition_hint(depth)
        if condition_hint is not None:
            merged = [_candidate_with_condition_hint(cand, *condition_hint) for cand in merged]
        return _annotate_vnext_pool(product, merged, stock_checker=self.stock_checker)

    def _fetch_candidates(self, product: str, *, ec1: int, skel_type: str, top_k: int) -> list[dict[str, Any]]:
        key = (canonical_smiles(product), int(ec1 or 0), str(skel_type or ""), int(top_k))
        cached = self._candidate_cache.get(key)
        if cached is not None:
            self.stats.candidate_cache_hits += 1
            return list(cached)
        rows = _candidates_for_skeleton_slot(
            self.retro_engine,
            product_smiles=product,
            ec1=ec1,
            skel_type=skel_type,
            top_k=top_k,
        )
        self._candidate_cache[key] = list(rows)
        return rows

    def _skeleton_hints(self, depth: int) -> list[tuple[int, str]]:
        hints: list[tuple[int, str]] = []
        for skel in self.skeletons:
            ec1 = skel.ec1s[depth] if depth < len(skel.ec1s) else 0
            typ = skel.types[depth] if depth < len(skel.types) else ""
            hints.append((int(ec1 or 0), str(typ or "")))
        return hints or [(0, "")]

    def _condition_hint(self, depth: int) -> tuple[float | None, float | None] | None:
        for skel in self.skeletons:
            T = skel.Ts[depth] if depth < len(skel.Ts) else None
            pH = skel.pHs[depth] if depth < len(skel.pHs) else None
            if T is not None or pH is not None:
                return T, pH
        return None

    def _candidate_allowed(self, cand: dict[str, Any], state: _State, *, product: str) -> bool:
        reactants = _candidate_reactants(cand)
        if not reactants:
            return False
        if _canonical_set(reactants) & self.forbidden_intermediates:
            self.stats.pruned_forbidden += 1
            return False
        expanded = set(state.expanded)
        if any(canonical_smiles(smi) in expanded for smi in reactants):
            self.stats.pruned_cycles += 1
            return False
        if any(canonical_smiles(smi) == canonical_smiles(step.product) for step in state.steps for smi in reactants):
            self.stats.pruned_cycles += 1
            return False
        if not _candidate_product_matches(cand, product):
            return False
        if not _atom_balance_ok(cand):
            return False
        return True

    def _choose_leaf(self, leaves: tuple[str, ...]) -> str:
        return max(leaves, key=_heavy_atoms)

    def _is_terminal(self, smiles: str) -> bool:
        can = canonical_smiles(smiles)
        if not can:
            return False
        if self.allowed_starting_materials and can in self.allowed_starting_materials:
            return True
        if self.stock_checker is None:
            return _heavy_atoms(smiles) <= 6
        try:
            return bool(self.stock_checker(smiles))
        except Exception:
            return False

    def _is_solved(self, state: _State) -> bool:
        return all(self._is_terminal(smi) for smi in state.open_leaves)

    def _state_satisfies_required(self, state: _State) -> bool:
        if not self.required_intermediates:
            return True
        seen = set(state.expanded)
        for step in state.steps:
            seen.add(canonical_smiles(step.product))
            seen.update(canonical_smiles(smi) for smi in _candidate_reactants(step.candidate))
        return self.required_intermediates.issubset({x for x in seen if x})

    def _priority(self, state: _State) -> float:
        open_penalty = sum(max(_heavy_atoms(smi), 1) for smi in state.open_leaves if not self._is_terminal(smi))
        stock_bonus = sum(1 for smi in state.open_leaves if self._is_terminal(smi))
        required_bonus = sum(1 for smi in state.expanded if smi in self.required_intermediates)
        return state.score + 2.0 * stock_bonus + 3.0 * required_bonus - 0.2 * open_penalty - 0.8 * state.depth

    def _state_to_result(self, target: str, state: _State) -> RouteResult:
        board = CascadeBoard.from_n_steps(len(state.steps), target)
        product_to_index = {canonical_smiles(step.product): idx for idx, step in enumerate(state.steps)}
        ordered = sorted(state.steps, key=lambda step: product_to_index.get(canonical_smiles(step.product), 0))
        for idx, step in enumerate(ordered):
            slot = board.slots[idx]
            slot.product = step.product
            _fill_slot_from_candidate(slot, step.candidate, step.candidate.get("source") or "candidate")
            slot.candidates = [dict(step.candidate)]
            slot.T = step.candidate.get("T")
            slot.pH = step.candidate.get("pH")
            slot.solvent = step.candidate.get("solvent", "")
        board.total_energy = -state.score
        value = self.value_function.score_board(board, stock_checker=self.stock_checker)
        return RouteResult(
            board=board,
            quality_vector={"stock_closed": 1.0, "branching_steps": float(len(state.steps))},
            risk_vector={},
            score=state.score + 100.0 + 10.0 * value.score,
            confidence=0.7,
            constraint_report={"search_mode": "stock_closed_andor"},
            explanation=RouteExplanation(
                why_selected="Stock-closed AND-OR route; all open leaves are stock or allowed starting materials.",
                uncertainty_table={
                    "search_mode": "stock_closed_andor",
                    "route_value_score": value.score,
                    "route_value_probability": value.probability,
                    "max_depth": self.max_depth,
                    "branch_factor": self.branch_factor,
                    **self.stats.to_dict(),
                },
            ),
        )


def _candidate_reactants(candidate: dict[str, Any]) -> list[str]:
    reactants = []
    main = candidate.get("main_reactant")
    if main:
        reactants.append(str(main))
    reactants.extend(str(smi) for smi in candidate.get("aux_reactants") or [] if smi)
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if rxn and ">>" in rxn:
        lhs = rxn.split(">>", 1)[0]
        reactants.extend(part.strip() for part in lhs.split(".") if part.strip())
    return _dedupe_smiles(reactants)


def _candidate_product_matches(candidate: dict[str, Any], product: str) -> bool:
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if not rxn or ">>" not in rxn:
        return True
    rhs = {canonical_smiles(part.strip()) for part in rxn.split(">>", 1)[1].split(".") if part.strip()}
    expected = canonical_smiles(product)
    return bool(rhs) and (not expected or expected in rhs)


def _candidate_score(
    candidate: dict[str, Any],
    product: str,
    stock_checker: StockChecker | None,
    value_function: RouteValueFunction | None = None,
) -> float:
    score = _as_float(candidate.get("score"), 0.0)
    reactants = _candidate_reactants(candidate)
    stock_hits = 0
    for smi in reactants:
        try:
            stock_hits += int(bool(stock_checker(smi))) if stock_checker else int(_heavy_atoms(smi) <= 6)
        except Exception:
            pass
    score += 3.0 * stock_hits / max(len(reactants), 1)
    prod_atoms = _heavy_atoms(product)
    largest = max((_heavy_atoms(smi) for smi in reactants), default=prod_atoms)
    if prod_atoms:
        score += max(0.0, (prod_atoms - largest) / prod_atoms)
        if largest > prod_atoms:
            growth_fraction = (largest - prod_atoms) / max(prod_atoms, 1)
            score -= 8.0 * min(growth_fraction, 2.0)
            if largest > prod_atoms * 1.3:
                score -= 6.0
    if candidate.get("ec") or candidate.get("enzyme_uid"):
        score += 0.2
    source = candidate.get("source") or ""
    if source == "v3_retrieval" and candidate.get("ec"):
        score += 1.0
    elif source == "enzyformer" and candidate.get("ec"):
        score += 0.5
    elif source == "enzexpand" and candidate.get("ec"):
        score += 0.3
    evidence = candidate.get("evidence") or {}
    if candidate.get("T") is not None and candidate.get("pH") is not None:
        score += 0.4
    if candidate.get("doi") or evidence.get("doi") or candidate.get("uniprot_accession") or evidence.get("uniprot_accession"):
        score += 0.3
    if candidate.get("catalyst") or candidate.get("cofactor") or evidence.get("cofactor"):
        score += 0.2
    if value_function is not None:
        value = value_function.score_candidate(product, candidate, stock_checker=stock_checker)
        score += 1.5 * value.score
        candidate["value_score"] = value.score
        candidate["value_probability"] = value.probability
    ranker = default_candidate_ranker()
    if ranker is not None:
        ranker_score = ranker.score_candidate(product, candidate, stock_checker=stock_checker)
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


def _candidate_with_condition_hint(
    candidate: dict[str, Any],
    T: float | None,
    pH: float | None,
) -> dict[str, Any]:
    out = dict(candidate)
    if out.get("T") is None and T is not None:
        out["T"] = T
        out.setdefault("condition_source", "skeleton_prior")
    if out.get("pH") is None and pH is not None:
        out["pH"] = pH
        out.setdefault("condition_source", "skeleton_prior")
    return out


def _atom_balance_ok(candidate: dict[str, Any]) -> bool:
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if not rxn or ">>" not in rxn:
        return True
    lhs, rhs = rxn.split(">>", 1)
    product_atoms = sum(_heavy_atoms(smi.strip()) for smi in rhs.split("."))
    reactant_atoms = sum(_heavy_atoms(smi.strip()) for smi in lhs.split("."))
    if product_atoms <= 10:
        return True
    return reactant_atoms >= max(4, int(product_atoms * 0.35))


def _candidate_source(candidate: dict[str, Any]) -> str:
    return str(candidate.get("source") or candidate.get("enzyme_source") or "unknown")


def _source_priority_for_hints(hints: list[tuple[int, str]]) -> tuple[str, ...]:
    if any(int(ec1 or 0) > 0 for ec1, _ in hints):
        return ("v3_retrieval", "enzyformer", "retrochimera", "enzexpand")
    return ("retrochimera", "v3_retrieval", "enzyformer", "enzexpand")


def _merge_candidates(
    rows: list[dict[str, Any]],
    top_k: int,
    *,
    source_priority: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in sorted(rows, key=lambda c: _as_float(c.get("score"), 0.0), reverse=True):
        key = canonical_reaction(cand.get("rxn_smiles") or cand.get("reaction_smiles")) or repr(cand)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)

    buckets: dict[str, list[dict[str, Any]]] = {}
    for cand in deduped:
        buckets.setdefault(_candidate_source(cand), []).append(cand)
    source_order = [src for src in source_priority if src in buckets]
    for src in buckets:
        if src not in source_order:
            source_order.append(src)

    out: list[dict[str, Any]] = []
    seen.clear()
    while len(out) < top_k:
        progressed = False
        for src in source_order:
            bucket = buckets.get(src) or []
            while bucket:
                cand = bucket.pop(0)
                key = canonical_reaction(cand.get("rxn_smiles") or cand.get("reaction_smiles")) or repr(cand)
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
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        for item in items:
            can = canonical_smiles(str(item))
            if can:
                out.add(can)
    return out


def _dedupe_smiles(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for smi in values:
        can = canonical_smiles(smi)
        key = can or smi
        if key and key not in seen:
            seen.add(key)
            out.append(smi)
    return out


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
