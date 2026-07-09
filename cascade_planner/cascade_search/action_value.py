"""Checkpoint-backed state-action value model for cascade search.

This module is the runtime counterpart of
``cascade_planner.eval.train_cascade_action_value``.  It scores actions as
Q(S,a) candidates and is allowed to influence both branch selection and global
search priority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from cascade_planner.cascade_search.state import CascadeAction, CascadeProgramState

RDLogger.DisableLog("rdApp.*")


class SubgoalHintActionScorer:
    """Soft action scorer using sidecar cascade subgoal hints.

    The scorer is intentionally conservative: it never rejects an action and
    returns the base action score plus a small priority when an action's
    product/reactants are close to evidence-supported subgoals stored on the
    parent state.
    """

    def __init__(
        self,
        *,
        max_bonus: float = 0.20,
        min_similarity: float = 0.45,
        evidence_score_weight: float = 0.60,
        structure_weight: float = 0.40,
    ):
        self.max_bonus = float(max_bonus)
        self.min_similarity = float(min_similarity)
        self.evidence_score_weight = float(evidence_score_weight)
        self.structure_weight = float(structure_weight)

    def score_actions(
        self,
        state: CascadeProgramState,
        actions: list[CascadeAction],
        child_states: list[CascadeProgramState] | None = None,
        *,
        expanded_leaf: str | None = None,
    ) -> list[float]:
        hints = _subgoal_hints_for_leaf(state, expanded_leaf)
        if not hints or not actions:
            return [_base_action_score(action) for action in actions]
        scores = []
        for action in actions:
            score, detail = self._score_action(action, hints)
            action.metadata.setdefault("subgoal_hint_action_score", detail)
            scores.append(score)
        return scores

    def _score_action(self, action: CascadeAction, hints: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
        step = action.step
        base = _base_action_score(action)
        if step is None:
            return base, {"applicable": False, "reason": "non_step_action", "base_score": base, "bonus": 0.0}
        reactant_mols = list(step.reactant_smiles or [])
        if not reactant_mols:
            return base, {"applicable": False, "reason": "missing_reactants", "base_score": base, "bonus": 0.0}
        best: dict[str, Any] = {"score": 0.0, "similarity": 0.0, "hint": None}
        for hint in hints:
            hint_mols = [
                str(hint.get("evidence_smiles") or ""),
                str(hint.get("subgoal_smiles") or ""),
            ]
            similarity = max(_mol_similarity(left, right) for left in reactant_mols for right in hint_mols if left and right)
            if similarity < self.min_similarity:
                continue
            evidence_score = _clip01((float(hint.get("learned_subgoal_score") or 0.0) + 2.0) / 4.0)
            raw = self.evidence_score_weight * evidence_score + self.structure_weight * similarity
            bonus = self.max_bonus * _clip01(raw)
            if bonus > float(best.get("score") or 0.0):
                best = {
                    "score": round(float(base + bonus), 6),
                    "base_score": round(float(base), 6),
                    "bonus": round(float(bonus), 6),
                    "similarity": round(float(similarity), 6),
                    "evidence_score_unit": round(float(evidence_score), 6),
                    "hint": {
                        "subgoal_hint_id": hint.get("subgoal_hint_id"),
                        "doi": hint.get("doi"),
                        "evidence_transform": hint.get("evidence_transform"),
                        "evidence_smiles": hint.get("evidence_smiles"),
                        "subgoal_smiles": hint.get("subgoal_smiles"),
                    },
                }
        if not best.get("hint"):
            return base, {"applicable": False, "reason": "no_matching_hint", "base_score": base, "bonus": 0.0}
        return float(best["score"]), {"applicable": True, **best}


class LoadedCascadeActionValueModel:
    """Torch checkpoint-backed action value model."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        import torch

        from cascade_planner.eval.train_cascade_action_value import CascadeActionValueNetwork

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
        from cascade_planner.eval.train_cascade_action_value import _feature_vector

        vector = np.asarray(_feature_vector(row, self.feature_schema), dtype=np.float32)
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


def _subgoal_hints_for_leaf(state: CascadeProgramState, expanded_leaf: str | None) -> list[dict[str, Any]]:
    hints = []
    for hint in (state.raw_metadata.get("cascade_subgoal_hints") or []):
        if not isinstance(hint, dict):
            continue
        if expanded_leaf and hint.get("target_leaf") and str(hint.get("target_leaf")) != str(expanded_leaf):
            continue
        hints.append(hint)
    return hints


def _base_action_score(action: CascadeAction) -> float:
    step = action.step
    if step is None or step.score is None:
        return 0.0
    return _clip01(float(step.score or 0.0))


def _mol_similarity(left_smiles: str, right_smiles: str) -> float:
    left = Chem.MolFromSmiles(str(left_smiles or ""))
    right = Chem.MolFromSmiles(str(right_smiles or ""))
    if left is None or right is None:
        return 0.0
    left_fp = AllChem.GetMorganFingerprintAsBitVect(left, 2, nBits=2048)
    right_fp = AllChem.GetMorganFingerprintAsBitVect(right, 2, nBits=2048)
    tanimoto = float(DataStructs.TanimotoSimilarity(left_fp, right_fp))
    if tanimoto >= 0.70:
        return tanimoto
    try:
        result = rdFMCS.FindMCS(
            [left, right],
            timeout=1,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        atoms = float(result.numAtoms or 0)
    except Exception:
        atoms = 0.0
    left_cov = atoms / max(float(left.GetNumHeavyAtoms()), 1.0)
    right_cov = atoms / max(float(right.GetNumHeavyAtoms()), 1.0)
    return max(tanimoto, 0.55 * left_cov + 0.45 * right_cov)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
