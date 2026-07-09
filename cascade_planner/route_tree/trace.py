"""Search trace collection for route-tree training data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState


@dataclass
class RouteTreeTraceEvent:
    state_id: str
    state: dict[str, Any]
    depth: int
    open_leaves: list[str]
    expanded_leaf: str
    candidate_actions: list[dict[str, Any]]
    selected_action_key: str | None = None
    selected_next_state_id: str | None = None
    action_scores: list[float] = field(default_factory=list)
    selection_scores: list[float] = field(default_factory=list)
    selection_score_breakdown: list[dict[str, Any]] = field(default_factory=list)
    value_trajectory: list[dict[str, Any]] = field(default_factory=list)
    bottleneck_trajectory: list[dict[str, Any]] = field(default_factory=list)
    source_budgets: list[dict[str, Any]] = field(default_factory=list)
    leaf_scores: list[dict[str, Any]] = field(default_factory=list)
    proposal_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    filter_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    expanded_leaf_stock_hit: bool = False
    expanded_leaf_heavy_atoms: int = 0
    expanded_leaf_parent_adjacent: bool = False
    expanded_leaf_low_yield: bool = False
    selected_next_stock_closed: bool | None = None
    selected_next_open_leaves: int | None = None
    model_active: bool = False
    model_reason: str = ""
    outcome: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "state": self.state,
            "depth": self.depth,
            "open_leaves": self.open_leaves,
            "expanded_leaf": self.expanded_leaf,
            "candidate_actions": self.candidate_actions,
            "selected_action_key": self.selected_action_key,
            "selected_next_state_id": self.selected_next_state_id,
            "action_scores": self.action_scores,
            "selection_scores": self.selection_scores,
            "selection_score_breakdown": self.selection_score_breakdown,
            "value_trajectory": self.value_trajectory,
            "bottleneck_trajectory": self.bottleneck_trajectory,
            "source_budgets": self.source_budgets,
            "leaf_scores": self.leaf_scores,
            "proposal_diagnostics": self.proposal_diagnostics,
            "filter_diagnostics": self.filter_diagnostics,
            "expanded_leaf_stock_hit": self.expanded_leaf_stock_hit,
            "expanded_leaf_heavy_atoms": self.expanded_leaf_heavy_atoms,
            "expanded_leaf_parent_adjacent": self.expanded_leaf_parent_adjacent,
            "expanded_leaf_low_yield": self.expanded_leaf_low_yield,
            "selected_next_stock_closed": self.selected_next_stock_closed,
            "selected_next_open_leaves": self.selected_next_open_leaves,
            "model_active": self.model_active,
            "model_reason": self.model_reason,
            "outcome": self.outcome,
        }


class RouteTreeTraceCollector:
    def __init__(self):
        self.events: list[RouteTreeTraceEvent] = []

    def record_expansion(
        self,
        *,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        action_scores: list[float],
        selection_scores: list[float] | None = None,
        selection_score_breakdown: list[dict[str, Any]] | None = None,
        model_active: bool,
        model_reason: str = "",
        selected_action: CandidateAction | None = None,
        next_state: RouteTreeState | None = None,
        proposal_diagnostics: list[dict[str, Any]] | None = None,
        filter_diagnostics: list[dict[str, Any]] | None = None,
        stock_checker: Any | None = None,
    ) -> None:
        metadata = next_state.search_metadata if next_state else state.search_metadata
        leaf_can = canonical_smiles(leaf) or leaf
        proposal_rows = list(proposal_diagnostics or metadata.get("proposal_diagnostics") or [])
        leaf_rows = [row for row in proposal_rows if _canonical_leaf(row.get("leaf")) == leaf_can]
        raw_actions = sum(int(row.get("raw_actions") or 0) for row in leaf_rows)
        final_actions = sum(int(row.get("final_actions") or 0) for row in leaf_rows)
        stock_hit = False
        if callable(stock_checker):
            try:
                stock_hit = bool(stock_checker(leaf))
            except Exception:
                stock_hit = False
        parent_adjacent = _leaf_has_parent_reaction(state, leaf)
        next_open = list(next_state.open_leaves) if next_state is not None else None
        selected_next_stock_closed = None
        if next_state is not None:
            if callable(stock_checker):
                try:
                    selected_next_stock_closed = all(bool(stock_checker(smi)) for smi in next_state.open_leaves)
                except Exception:
                    selected_next_stock_closed = None
            else:
                selected_next_stock_closed = None
        self.events.append(
            RouteTreeTraceEvent(
                state_id=state.canonical_id,
                state=state.to_trace_dict(),
                depth=state.depth,
                open_leaves=list(state.open_leaves),
                expanded_leaf=leaf,
                candidate_actions=[_action_trace_dict(action, stock_checker=stock_checker) for action in actions],
                selected_action_key=selected_action.canonical_key if selected_action else None,
                selected_next_state_id=next_state.canonical_id if next_state else None,
                action_scores=list(action_scores),
                selection_scores=list(selection_scores or []),
                selection_score_breakdown=list(selection_score_breakdown or []),
                value_trajectory=list(metadata.get("value_trajectory") or []),
                bottleneck_trajectory=list(metadata.get("bottleneck_trajectory") or []),
                source_budgets=list(metadata.get("source_budgets") or []),
                leaf_scores=list(metadata.get("leaf_scores") or []),
                proposal_diagnostics=proposal_rows,
                filter_diagnostics=list(filter_diagnostics or metadata.get("filter_diagnostics") or []),
                expanded_leaf_stock_hit=stock_hit,
                expanded_leaf_heavy_atoms=_heavy_atoms(leaf),
                expanded_leaf_parent_adjacent=parent_adjacent,
                expanded_leaf_low_yield=bool(raw_actions > 0 and final_actions <= 1),
                selected_next_stock_closed=selected_next_stock_closed,
                selected_next_open_leaves=len(next_open) if next_open is not None else None,
                model_active=bool(model_active),
                model_reason=str(model_reason or ""),
            )
        )

    def annotate_outcome(self, outcome: dict[str, Any]) -> None:
        for event in self.events:
            event.outcome = dict(outcome)

    def to_rows(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


def _canonical_leaf(value: Any) -> str:
    return canonical_smiles(str(value or "")) or str(value or "")


def _leaf_has_parent_reaction(state: RouteTreeState, leaf: str) -> bool:
    leaf_can = canonical_smiles(leaf) or leaf
    for step in state.steps:
        for reactant in step.action.reactants:
            if (canonical_smiles(reactant) or reactant) == leaf_can:
                return True
    return False


def _action_trace_dict(action: CandidateAction, *, stock_checker: Any | None = None) -> dict[str, Any]:
    payload = action.to_dict()
    reactants = [str(smi) for smi in payload.get("reactants") or [] if smi]
    if not callable(stock_checker) or not reactants:
        return payload
    stock_status: dict[str, bool] = {}
    for smi in reactants:
        try:
            stock_status[smi] = bool(stock_checker(smi))
        except Exception:
            stock_status[smi] = False
    stock_hits = sum(1 for value in stock_status.values() if value)
    payload["reactant_stock_status"] = stock_status
    payload["reactant_stock_fraction"] = float(stock_hits / max(len(reactants), 1))
    payload["stock_closing_candidate"] = bool(reactants and stock_hits == len(reactants))
    return payload


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0
