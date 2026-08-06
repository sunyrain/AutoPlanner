"""Checkpoint-backed state-action value model for cascade search.

This module loads the frozen action-value checkpoint contract owned by the
explicit legacy runtime. It scores actions as Q(S,a) candidates and is allowed
to influence both branch selection and global search priority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.cascade_search.state import CascadeAction, CascadeProgramState
from cascade_planner.legacy.cascade_search_runtime.action_value_contract import (
    CascadeActionValueNetwork,
    action_value_feature_vector,
)


class LoadedCascadeActionValueModel:
    """Torch checkpoint-backed action value model."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        import torch

        self.checkpoint_path = str(checkpoint_path)
        self._torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.feature_schema = dict(checkpoint.get("feature_schema") or {})
        self.input_dim = int(self.feature_schema.get("feature_dim") or 0)
        if self.input_dim <= 0:
            raise ValueError(f"invalid cascade action-value checkpoint feature_dim: {checkpoint_path}")
        hidden = int(checkpoint.get("hidden") or 192)
        self.model = CascadeActionValueNetwork(self.input_dim, hidden=hidden).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def score_actions(
        self,
        state: CascadeProgramState,
        actions: list[CascadeAction],
        child_states: list[CascadeProgramState] | None = None,
        *,
        expanded_leaf: str | None = None,
    ) -> list[float]:
        if not actions:
            return []
        torch = self._torch
        rows = [
            self._row_from_action(state, action, expanded_leaf=expanded_leaf)
            for action in actions
        ]
        features = np.asarray([self._feature_vector(row) for row in rows], dtype=np.float32)
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        return [float(score) for score in scores]

    def _feature_vector(self, row: dict[str, Any]) -> np.ndarray:
        vector = np.asarray(
            action_value_feature_vector(row, self.feature_schema),
            dtype=np.float32,
        )
        if len(vector) == self.input_dim:
            return vector
        if len(vector) > self.input_dim:
            return vector[: self.input_dim]
        return np.pad(vector, (0, self.input_dim - len(vector))).astype(np.float32)

    def _row_from_action(
        self,
        state: CascadeProgramState,
        action: CascadeAction,
        *,
        expanded_leaf: str | None,
    ) -> dict[str, Any]:
        step = action.step
        parent_mol = expanded_leaf or action.target_leaf or ""
        if not parent_mol and step is not None:
            parent_mol = step.product_smiles
        reactants = list(step.reactant_smiles if step is not None else [])
        raw = dict(step.raw_metadata if step is not None else {})
        cascade_cost = raw.get("cascade_cost") if isinstance(raw.get("cascade_cost"), dict) else {}
        components = dict(cascade_cost.get("components") or {})
        context_features = _state_context_features(state, action, expanded_leaf=expanded_leaf)
        return {
            "target_smiles": state.target_smiles,
            "route_domain": state.raw_metadata.get("route_domain") or context_features.get("route_domain") or "unknown",
            "state_id": _runtime_state_id(state, parent_mol),
            "parent_mol": parent_mol,
            "parent_depth": len(state.step_annotations),
            "candidate_index": action.metadata.get("provider_rank") or cascade_cost.get("candidate_index"),
            "source_model": (step.source_model if step is not None else action.source) or action.source or "unknown",
            "reaction_domain": _reaction_domain(step, action),
            "reactants": reactants,
            "rxn_smiles": step.rxn_smiles if step is not None else "",
            "base_score": step.score if step is not None else None,
            "base_cost": raw.get("cost"),
            "cascade_adjustment": cascade_cost.get("cascade_adjustment"),
            "total_cost": cascade_cost.get("total_cost"),
            "components": components,
            "context_features": context_features,
            "source_policy_decision": action.metadata.get("source_policy_decision") or {},
            "active_failure_modes": [failure.category for failure in state.unresolved_failure_modes],
            "labels": {},
        }


def _runtime_state_id(state: CascadeProgramState, parent_mol: str) -> str:
    return "|".join([
        state.target_smiles or "",
        parent_mol or "",
        str(len(state.step_annotations)),
        ".".join(sorted(state.open_molecule_leaves or state.open_leaves or [])),
    ])


def _reaction_domain(step: Any, action: CascadeAction) -> str:
    if step is None:
        return "unknown"
    raw = step.raw_metadata or {}
    cascade_cost = raw.get("cascade_cost") if isinstance(raw.get("cascade_cost"), dict) else {}
    if cascade_cost.get("reaction_domain"):
        return str(cascade_cost.get("reaction_domain"))
    if step.is_enzymatic:
        return "enzymatic"
    text = " ".join([step.reaction_type or "", step.source_model or "", action.source or ""]).lower()
    if any(token in text for token in ("enzyme", "enzymatic", "bio", "ec ")):
        return "enzymatic"
    if step.rxn_smiles:
        return "chemical"
    return "unknown"


def _state_context_features(
    state: CascadeProgramState,
    action: CascadeAction,
    *,
    expanded_leaf: str | None,
) -> dict[str, Any]:
    adjacent_domain = "unknown"
    leaf = expanded_leaf or action.target_leaf
    for step in state.step_annotations:
        if leaf and leaf in set(step.reactant_smiles or []):
            adjacent_domain = "enzymatic" if step.is_enzymatic else "chemical"
            break
    return {
        "route_domain": state.raw_metadata.get("route_domain") or "unknown",
        "node_depth": len(state.step_annotations),
        "adjacent_reaction_domain": adjacent_domain,
        "active_failure_modes": [failure.category for failure in state.unresolved_failure_modes],
    }
