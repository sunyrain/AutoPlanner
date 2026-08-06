"""Current action-scoring helpers for cascade search."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from cascade_planner.cascade_search.state import CascadeAction, CascadeProgramState

RDLogger.DisableLog("rdApp.*")


class SubgoalHintActionScorer:
    """Soft action scorer using sidecar cascade subgoal hints.

    The scorer never rejects an action. It returns the base action score plus a
    small priority when an action's reactants are close to evidence-supported
    subgoals stored on the parent state.
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

    def _score_action(
        self,
        action: CascadeAction,
        hints: list[dict[str, Any]],
    ) -> tuple[float, dict[str, Any]]:
        step = action.step
        base = _base_action_score(action)
        if step is None:
            return base, {
                "applicable": False,
                "reason": "non_step_action",
                "base_score": base,
                "bonus": 0.0,
            }
        reactant_mols = list(step.reactant_smiles or [])
        if not reactant_mols:
            return base, {
                "applicable": False,
                "reason": "missing_reactants",
                "base_score": base,
                "bonus": 0.0,
            }
        best: dict[str, Any] = {"score": 0.0, "similarity": 0.0, "hint": None}
        for hint in hints:
            hint_mols = [
                str(hint.get("evidence_smiles") or ""),
                str(hint.get("subgoal_smiles") or ""),
            ]
            similarity = max(
                _mol_similarity(left, right)
                for left in reactant_mols
                for right in hint_mols
                if left and right
            )
            if similarity < self.min_similarity:
                continue
            evidence_score = _clip01(
                (float(hint.get("learned_subgoal_score") or 0.0) + 2.0) / 4.0
            )
            raw = (
                self.evidence_score_weight * evidence_score
                + self.structure_weight * similarity
            )
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
            return base, {
                "applicable": False,
                "reason": "no_matching_hint",
                "base_score": base,
                "bonus": 0.0,
            }
        return float(best["score"]), {"applicable": True, **best}


def _subgoal_hints_for_leaf(
    state: CascadeProgramState,
    expanded_leaf: str | None,
) -> list[dict[str, Any]]:
    hints = []
    for hint in state.raw_metadata.get("cascade_subgoal_hints") or []:
        if not isinstance(hint, dict):
            continue
        if (
            expanded_leaf
            and hint.get("target_leaf")
            and str(hint.get("target_leaf")) != str(expanded_leaf)
        ):
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
