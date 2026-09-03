"""Bridge host-replayed ReactionJSON candidates into ChemEnzy's AND/OR tree.

ChemEnzy already represents a molecule as an OR over reaction children and a
reaction as an AND over all precursor children.  This module deliberately
does not implement a second search tree.  It supplies the missing integration
boundary for AutoPlanner's Codex Route Builder: several independently replayed
ReactionJSON candidates can be appended to one ChemEnzy molecule expansion,
while their host-owned mapped structures and route rows remain available for
later prompts and RouteJSON projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
import threading
from typing import Any, Iterable, Mapping, Sequence


_IMPORT_LOCK = threading.Lock()
_DEFAULT_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "ChemEnzyRetroPlanner"


@dataclass(frozen=True, slots=True)
class ReactionJsonOrCandidate:
    """One host-compiled local disconnection ready for ChemEnzy expansion."""

    candidate_id: str
    precursor_smiles: tuple[str, ...]
    mapped_precursor_smiles: tuple[str, ...]
    route_step: Mapping[str, Any]
    score: float
    cost: float
    candidate_key: str


@dataclass(frozen=True, slots=True)
class ReactionJsonOrProjection:
    """Current best ChemEnzy route projection without discarding its siblings."""

    steps: tuple[Mapping[str, Any], ...]
    open_leaf_states: tuple[Mapping[str, str], ...]
    deferred_builder_leaf_states: tuple[Mapping[str, str], ...]
    complete: bool
    summary: Mapping[str, Any]


class ChemEnzyReactionJsonOrSearch:
    """Small stateful facade over ChemEnzy ``MolTree`` for Codex candidates.

    The underlying vendor tree owns OR/AND value propagation and backtracking.
    AutoPlanner owns candidate validation, atom maps, stock membership, and the
    final RouteJSON representation.
    """

    def __init__(
        self,
        *,
        target_smiles: str,
        mapped_target_smiles: str,
        max_depth: int,
        vendor_root: Path | str | None = None,
        deferred_node_penalty: float = 100.0,
    ) -> None:
        mol_tree_class = _load_mol_tree_class(vendor_root)
        self.tree = mol_tree_class(
            target_mol=str(target_smiles),
            known_mols=set(),
            value_fn=lambda _mol: 0.0,
            max_depth=max(1, int(max_depth)),
        )
        self.deferred_node_penalty = max(1.0, float(deferred_node_penalty))
        self._node_context: dict[int, dict[str, str]] = {
            int(self.tree.root.id): {
                "smiles": str(target_smiles),
                "mapped_smiles": str(mapped_target_smiles),
            }
        }
        self._blocked_node_ids: set[int] = set()
        self._candidate_keys: set[str] = set()
        self.accepted_candidate_count = 0
        self.duplicate_candidate_count = 0
        self.backtrack_count = 0

    def select_open_node(self) -> Any | None:
        candidates = [
            node
            for node in self.tree.mol_nodes
            if bool(getattr(node, "open", False))
            and not bool(getattr(node, "is_known", False))
        ]
        if not candidates:
            return None
        return min(candidates, key=_node_selection_key)

    def node_state(self, node: Any) -> tuple[str, str]:
        context = dict(self._node_context.get(int(getattr(node, "id", -1))) or {})
        return str(context.get("smiles") or getattr(node, "mol", "")), str(
            context.get("mapped_smiles") or ""
        )

    def context_steps_for_node(self, node: Any) -> tuple[Mapping[str, Any], ...]:
        steps: list[Mapping[str, Any]] = []
        cursor = node
        while getattr(cursor, "parent", None) is not None:
            reaction = cursor.parent
            annotation = dict(getattr(reaction, "cascade_annotation", None) or {})
            step = annotation.get("autoplanner_route_step")
            if isinstance(step, Mapping):
                steps.append(dict(step))
            cursor = reaction.parent
        steps.reverse()
        return tuple(steps)

    def expand(
        self,
        node: Any,
        candidates: Sequence[ReactionJsonOrCandidate],
        *,
        stock_smiles: Iterable[str] = (),
    ) -> int:
        """Append every distinct candidate as an OR child of ``node``."""

        if not bool(getattr(node, "open", False)) or getattr(node, "children", None):
            raise ValueError("chemenzy_reactionjson_node_not_open")

        ancestors = {
            str(value) for value in (node.get_ancestors() if hasattr(node, "get_ancestors") else ())
        }
        accepted: list[ReactionJsonOrCandidate] = []
        for candidate in candidates:
            key = str(candidate.candidate_key or "")
            if not key or key in self._candidate_keys:
                self.duplicate_candidate_count += 1
                continue
            precursors = tuple(dict.fromkeys(str(value) for value in candidate.precursor_smiles if value))
            if not precursors or any(value in ancestors for value in precursors):
                continue
            accepted.append(candidate)
            self._candidate_keys.add(key)

        if not accepted:
            return 0

        self.tree.known_mols.update(str(value) for value in stock_smiles if value)
        reactant_lists: list[list[str]] = []
        costs: list[float] = []
        templates: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for rank, candidate in enumerate(accepted, start=1):
            precursors = list(dict.fromkeys(candidate.precursor_smiles))
            reactant_lists.append(precursors)
            costs.append(max(0.0, float(candidate.cost)))
            template = {
                "source": "codex_reactionjson",
                "candidate_id": candidate.candidate_id,
                "candidate_key": candidate.candidate_key,
                "rank": rank,
            }
            templates.append(template)
            annotations.append(
                {
                    "source_model": "Codex Route Builder",
                    "reaction_domain": "codex_reactionjson",
                    "base_score": float(candidate.score),
                    "base_cost": max(0.0, float(candidate.cost)),
                    "candidate_id": candidate.candidate_id,
                    "candidate_key": candidate.candidate_key,
                    "autoplanner_route_step": dict(candidate.route_step),
                    "mapped_precursor_smiles": list(candidate.mapped_precursor_smiles),
                }
            )

        existing_children = len(getattr(node, "children", []) or [])
        self.tree.expand(
            node,
            reactant_lists,
            costs,
            templates,
            max_depth=int(getattr(node, "max_depth", 1)),
            cascade_annotations=annotations,
        )
        reactions = list(getattr(node, "children", []) or [])[existing_children:]
        for reaction, candidate in zip(reactions, accepted):
            mapped_by_smiles: dict[str, list[str]] = {}
            for canonical, mapped in zip(
                candidate.precursor_smiles,
                candidate.mapped_precursor_smiles,
            ):
                mapped_by_smiles.setdefault(str(canonical), []).append(str(mapped))
            for child in list(getattr(reaction, "children", []) or []):
                choices = mapped_by_smiles.get(str(getattr(child, "mol", ""))) or []
                mapped = choices.pop(0) if choices else ""
                self._node_context[int(child.id)] = {
                    "smiles": str(child.mol),
                    "mapped_smiles": mapped,
                }

        self.accepted_candidate_count += len(reactions)
        return len(reactions)

    def replay_route(
        self,
        steps: Sequence[Mapping[str, Any]],
        *,
        stock_smiles: Iterable[str] = (),
    ) -> int:
        """Rebuild this OR state from one host-replayed RouteJSON projection.

        A Codex Editor may replace a step after the initial OR tree has already
        propagated ``succ`` to the root.  Reusing that tree would make the old
        solved bit authoritative over the edited route.  This deterministic
        replay creates fresh molecule/reaction nodes from the edited host rows
        and therefore recomputes stock closure instead of copying stale search
        state.
        """

        stock = {str(value) for value in stock_smiles if str(value)}
        inserted = 0
        for index, raw in enumerate(steps, start=1):
            step = dict(raw)
            product = str(step.get("product_smiles") or "")
            mapped_product = str(step.get("mapped_product_smiles") or "")
            precursors = tuple(
                str(value)
                for value in step.get("precursor_smiles") or []
                if str(value)
            )
            mapped_precursors = tuple(
                str(value)
                for value in step.get("mapped_precursor_smiles") or []
            )
            matches = [
                node
                for node in self.tree.mol_nodes
                if bool(getattr(node, "open", False))
                and str(getattr(node, "mol", "")) == product
            ]
            if mapped_product:
                mapped_matches = [
                    node
                    for node in matches
                    if self.node_state(node)[1] == mapped_product
                ]
                if mapped_matches:
                    matches = mapped_matches
            if not matches or not precursors:
                raise ValueError("edited_route_cannot_rebuild_reactionjson_or_state")
            node = min(matches, key=lambda value: int(getattr(value, "id", -1)))
            step_id = str(step.get("step_id") or f"editor-replay:{index}")
            candidate = ReactionJsonOrCandidate(
                candidate_id=step_id,
                precursor_smiles=precursors,
                mapped_precursor_smiles=mapped_precursors,
                route_step=step,
                score=1.0,
                cost=0.0,
                candidate_key=(
                    f"editor-replay:{index}:{step_id}:{product}>"
                    + ".".join(precursors)
                ),
            )
            added = self.expand(node, (candidate,), stock_smiles=stock)
            if added != 1:
                raise ValueError("edited_route_reactionjson_or_replay_rejected")
            inserted += added
        return inserted

    def defer_failed_node(self, node: Any) -> None:
        """Backtrack from one repeatedly invalid edit without marking it solved.

        A finite high value makes a cheaper sibling reaction preferable, which
        is the desired backtrack.  The blocked molecule remains visible in the
        projection as an unresolved leaf for ordinary Builder continuation.
        """

        if not bool(getattr(node, "open", False)):
            return
        old_value = float(getattr(node, "value", 0.0))
        node.value = max(self.deferred_node_penalty, old_value)
        node.open = False
        node.go_back = True
        node.succ = False
        self._blocked_node_ids.add(int(node.id))
        if getattr(node, "parent", None) is not None:
            node.parent.backup(node.value - old_value, from_mol=node.mol)
        self.backtrack_count += 1

    def project(self) -> ReactionJsonOrProjection:
        steps: list[Mapping[str, Any]] = []
        active: list[Mapping[str, str]] = []
        deferred: list[Mapping[str, str]] = []
        seen_reactions: set[int] = set()

        def visit(molecule: Any) -> None:
            if bool(getattr(molecule, "is_known", False)):
                return
            reactions = list(getattr(molecule, "children", []) or [])
            if not reactions:
                state = self._state_row(molecule)
                if int(getattr(molecule, "id", -1)) in self._blocked_node_ids:
                    deferred.append(state)
                elif bool(getattr(molecule, "open", False)):
                    active.append(state)
                return
            reaction = _preferred_reaction(reactions)
            reaction_id = int(getattr(reaction, "id", -1))
            if reaction_id not in seen_reactions:
                annotation = dict(getattr(reaction, "cascade_annotation", None) or {})
                step = annotation.get("autoplanner_route_step")
                if isinstance(step, Mapping):
                    steps.append(dict(step))
                seen_reactions.add(reaction_id)
            for child in list(getattr(reaction, "children", []) or []):
                visit(child)

        visit(self.tree.root)
        active = _dedupe_states(active)
        deferred = _dedupe_states(deferred)
        summary = {
            "engine": "ChemEnzyRetroPlanner.MolTree",
            "selection_policy": "best_first_v_target",
            "ucb_active": False,
            "progressive_widening_active": False,
            "molecule_nodes": len(self.tree.mol_nodes),
            "reaction_nodes": len(self.tree.reaction_nodes),
            "accepted_candidates": int(self.accepted_candidate_count),
            "duplicate_candidates": int(self.duplicate_candidate_count),
            "backtracks": int(self.backtrack_count),
            "open_nodes": sum(bool(getattr(node, "open", False)) for node in self.tree.mol_nodes),
            "deferred_failed_nodes": len(self._blocked_node_ids),
            "root_solved": bool(getattr(self.tree.root, "succ", False)),
        }
        return ReactionJsonOrProjection(
            steps=tuple(steps),
            open_leaf_states=tuple(active),
            deferred_builder_leaf_states=tuple(deferred),
            complete=bool(getattr(self.tree.root, "succ", False)),
            summary=summary,
        )

    def _state_row(self, node: Any) -> Mapping[str, str]:
        smiles, mapped = self.node_state(node)
        return {"smiles": smiles, "mapped_smiles": mapped}


def ranked_candidate_cost(index: int, score: Any = None) -> tuple[float, float]:
    """Return a deterministic non-negative ChemEnzy prior cost."""

    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 1.0 / max(1, int(index) + 1)
    numeric_score = min(1.0, max(1.0e-6, numeric_score))
    return numeric_score, -math.log(numeric_score)


def _load_mol_tree_class(vendor_root: Path | str | None) -> Any:
    root = Path(vendor_root or _DEFAULT_VENDOR_ROOT).resolve()
    if not (root / "retro_planner" / "search_frame" / "mcts_star" / "mol_tree.py").is_file():
        raise RuntimeError(f"chemenzy_vendor_tree_missing:{root}")
    with _IMPORT_LOCK:
        added = str(root) not in sys.path
        if added:
            sys.path.insert(0, str(root))
        try:
            return importlib.import_module(
                "retro_planner.search_frame.mcts_star.mol_tree"
            ).MolTree
        finally:
            if added:
                try:
                    sys.path.remove(str(root))
                except ValueError:
                    pass


def _node_selection_key(node: Any) -> tuple[float, int, int, str]:
    try:
        value = float(node.v_target())
    except (AttributeError, TypeError, ValueError):
        value = math.inf
    if math.isnan(value):
        value = math.inf
    return (
        value,
        int(getattr(node, "id", -1)),
        int(getattr(node, "depth", -1)),
        str(getattr(node, "mol", "")),
    )


def _preferred_reaction(reactions: Sequence[Any]) -> Any:
    successful = [reaction for reaction in reactions if bool(getattr(reaction, "succ", False))]
    if successful:
        return min(
            successful,
            key=lambda reaction: (
                float(getattr(reaction, "succ_value", math.inf)),
                int(getattr(reaction, "id", -1)),
            ),
        )
    return min(
        reactions,
        key=lambda reaction: (
            float(getattr(reaction, "value", math.inf)),
            int(getattr(reaction, "id", -1)),
        ),
    )


def _dedupe_states(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    result: list[Mapping[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("smiles") or ""), str(row.get("mapped_smiles") or ""))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result
