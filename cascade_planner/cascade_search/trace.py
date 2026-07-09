"""Trace collection for cascade-native search."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from cascade_planner.cascade_search.state import CascadeAction, CascadeFailure, CascadeProgramState


TRACE_SCHEMA_VERSION = "cascade_search_trace.v1"


@dataclass
class CascadeSearchTraceEvent:
    state_id: str
    state: dict[str, Any]
    depth: int
    open_leaves: list[str]
    expanded_leaf: str
    candidate_actions: list[dict[str, Any]]
    candidate_scores: list[float] = field(default_factory=list)
    child_state_ids: list[str] = field(default_factory=list)
    child_summaries: list[dict[str, Any]] = field(default_factory=list)
    failure_categories: list[str] = field(default_factory=list)
    model_active: bool = False
    outcome: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "state": self.state,
            "depth": self.depth,
            "open_leaves": self.open_leaves,
            "expanded_leaf": self.expanded_leaf,
            "candidate_actions": self.candidate_actions,
            "candidate_scores": self.candidate_scores,
            "child_state_ids": self.child_state_ids,
            "child_summaries": self.child_summaries,
            "failure_categories": self.failure_categories,
            "model_active": self.model_active,
            "outcome": self.outcome,
        }


class CascadeTraceCollector:
    """Collect expansion traces from CascadeProgramSearch."""

    def __init__(self) -> None:
        self.events: list[CascadeSearchTraceEvent] = []

    def record_expansion(
        self,
        *,
        state: CascadeProgramState,
        expanded_leaf: str,
        candidate_actions: list[CascadeAction],
        candidate_scores: list[float] | None = None,
        child_states: list[CascadeProgramState] | None = None,
        failures: list[CascadeFailure] | None = None,
        model_active: bool = False,
    ) -> None:
        candidate_scores = list(candidate_scores or [])
        child_states = list(child_states or [])
        failure_categories = [failure.category for failure in failures or []]
        self.events.append(
            CascadeSearchTraceEvent(
                state_id=_state_signature(state),
                state=state.to_dict(),
                depth=len(state.step_annotations),
                open_leaves=list(state.open_molecule_leaves or state.open_leaves),
                expanded_leaf=expanded_leaf,
                candidate_actions=[action.to_dict() for action in candidate_actions],
                candidate_scores=[float(score) for score in candidate_scores],
                child_state_ids=[_state_signature(child) for child in child_states],
                child_summaries=[_child_summary(child) for child in child_states],
                failure_categories=failure_categories,
                model_active=bool(model_active),
            )
        )

    def annotate_outcome(self, outcome: dict[str, Any]) -> None:
        for event in self.events:
            event.outcome = dict(outcome)

    def mark_enqueued_children(
        self,
        *,
        parent_state: CascadeProgramState,
        enqueued_children: list[CascadeProgramState],
    ) -> None:
        parent_id = _state_signature(parent_state)
        enqueued_ids = {_state_signature(child) for child in enqueued_children}
        for event in self.events:
            if event.state_id != parent_id:
                continue
            for idx, child_id in enumerate(event.child_state_ids):
                enqueued = child_id in enqueued_ids
                if idx < len(event.child_summaries):
                    event.child_summaries[idx]["enqueued_from_state"] = enqueued
                if idx < len(event.candidate_actions):
                    metadata = event.candidate_actions[idx].setdefault("metadata", {})
                    metadata["enqueued_from_state"] = enqueued

    def to_rows(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


def _child_summary(state: CascadeProgramState) -> dict[str, Any]:
    return {
        "state_id": _state_signature(state),
        "step_count": len(state.step_annotations),
        "stage_count": state.stage_graph.n_stages,
        "stock_closed": bool(state.stock_closed),
        "cofactor_closed": not bool(state.cofactor_ledger.unclosed_requirements()),
        "failure_categories": [failure.category for failure in state.unresolved_failure_modes],
        "open_leaves": list(state.open_molecule_leaves or state.open_leaves),
        "cascade_cost": state.cascade_cost,
    }


def _state_signature(state: CascadeProgramState) -> str:
    payload = {
        "target": state.target_smiles,
        "open": sorted(state.open_molecule_leaves or state.open_leaves),
        "steps": [step.rxn_smiles for step in state.step_annotations],
        "stages": list(state.stage_partition),
        "cofactors": state.cofactor_ledger.to_dict(),
    }
    return json.dumps(payload, sort_keys=True, default=str)
