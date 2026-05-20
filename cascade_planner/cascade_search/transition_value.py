"""Process-aware transition value model for cascade-native search.

This module intentionally scores state/action/child transitions rather than
single reaction candidates. Its supervision is expected to come from process
progress signals: typed failure deltas, stock/cofactor closure, condition
compatibility, stage complexity, and rollout outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.cascade_search.state import CascadeAction, CascadeFailureKind, CascadeProgramState
from cascade_planner.vnext.features import morgan_fp


TRANSITION_FAILURE_KINDS = [kind.value for kind in CascadeFailureKind]
TRANSITION_SCALAR_FEATURES = [
    "parent_step_count_scaled",
    "parent_open_count_scaled",
    "parent_stage_count_scaled",
    "parent_stock_closed",
    "parent_cofactor_closed",
    "parent_failure_count_scaled",
    "parent_cost_scaled",
    "child_step_count_scaled",
    "child_open_count_scaled",
    "child_stage_count_scaled",
    "child_stock_closed",
    "child_cofactor_closed",
    "child_condition_compatible",
    "child_evidence_sufficient",
    "child_failure_count_scaled",
    "child_cost_scaled",
    "open_leaf_reduction_scaled",
    "failure_reduction_scaled",
    "cost_reduction_scaled",
    "stage_delta_scaled",
    "stock_gain",
    "cofactor_gain",
    "candidate_score",
    "is_enzymatic",
    "has_condition",
    "condition_field_count_scaled",
    "cofactor_requirement_scaled",
    "cofactor_regeneration_scaled",
    "enzyme_identifier_scaled",
    "reactant_count_scaled",
]


def _torch_modules() -> tuple[Any, Any]:
    import torch
    import torch.nn as nn

    return torch, nn


@dataclass
class CascadeTransitionValuePrediction:
    scores: list[float]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"scores": list(self.scores), "metadata": dict(self.metadata or {})}


class CascadeTransitionValueNetwork:
    def __new__(cls, input_dim: int, hidden: int = 192):
        _torch, nn = _torch_modules()

        class _Network(nn.Module):
            def __init__(self, input_dim: int, hidden: int = 192):
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden = int(hidden)
                self.net = nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(self.hidden, max(32, self.hidden // 2)),
                    nn.GELU(),
                    nn.Linear(max(32, self.hidden // 2), 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _Network(input_dim, hidden=hidden)


class LoadedCascadeTransitionValueModel:
    """Checkpoint-backed transition value model."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        torch, _nn = _torch_modules()

        self.checkpoint_path = str(checkpoint_path)
        self._torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.feature_schema = dict(checkpoint.get("feature_schema") or {})
        self.n_bits = int(self.feature_schema.get("n_bits") or 128)
        self.input_dim = int(self.feature_schema.get("input_dim") or transition_feature_dim(self.n_bits))
        hidden = int(checkpoint.get("hidden") or 192)
        self.model = CascadeTransitionValueNetwork(self.input_dim, hidden=hidden).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def score_transitions(
        self,
        state: CascadeProgramState,
        actions: list[CascadeAction],
        child_states: list[CascadeProgramState],
        *,
        expanded_leaf: str | None = None,
    ) -> list[float]:
        if not actions or not child_states:
            return []
        torch = self._torch
        rows = []
        parent = state.to_dict()
        for action, child in zip(actions, child_states):
            vector = transition_feature_vector(
                parent,
                action.to_dict(),
                transition_child_summary(child),
                expanded_leaf=expanded_leaf,
                n_bits=self.n_bits,
            )
            rows.append(_resize(vector, self.input_dim))
        x = torch.tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        return [float(score) for score in scores]


def transition_child_summary(state: CascadeProgramState) -> dict[str, Any]:
    return {
        "step_count": len(state.step_annotations),
        "stage_count": state.stage_graph.n_stages,
        "stock_closed": bool(state.stock_closed),
        "cofactor_closed": not bool(state.cofactor_ledger.unclosed_requirements()),
        "failure_categories": [failure.category for failure in state.unresolved_failure_modes],
        "open_leaves": list(state.open_molecule_leaves or state.open_leaves),
        "cascade_cost": state.cascade_cost,
    }


def transition_feature_dim(n_bits: int = 128) -> int:
    return n_bits * 3 + len(TRANSITION_SCALAR_FEATURES) + len(TRANSITION_FAILURE_KINDS) * 2


def transition_feature_vector(
    parent_state: dict[str, Any] | CascadeProgramState,
    action: dict[str, Any] | CascadeAction,
    child_summary: dict[str, Any],
    *,
    expanded_leaf: str | None = None,
    n_bits: int = 128,
) -> np.ndarray:
    parent = parent_state.to_dict() if hasattr(parent_state, "to_dict") else dict(parent_state or {})
    action_data = action.to_dict() if hasattr(action, "to_dict") else dict(action or {})
    child = dict(child_summary or {})
    step = dict(action_data.get("step") or {})
    target = str(parent.get("target_smiles") or parent.get("target") or "")
    product = str(step.get("product_smiles") or action_data.get("target_leaf") or expanded_leaf or "")
    reactants = ".".join(str(smi) for smi in step.get("reactant_smiles") or [] if smi)
    if not reactants and step.get("rxn_smiles") and ">>" in str(step.get("rxn_smiles")):
        reactants = str(step.get("rxn_smiles")).split(">>", 1)[0]
    parent_failures = _failure_categories(parent)
    child_failures = [str(item) for item in child.get("failure_categories") or []]
    parent_open = list(parent.get("open_molecule_leaves") or parent.get("open_leaves") or [])
    child_open = list(child.get("open_leaves") or [])
    parent_step_count = float(len(parent.get("step_annotations") or parent.get("steps") or []))
    child_step_count = float(child.get("step_count") or 0.0)
    parent_stage_count = float(_stage_count(parent))
    child_stage_count = float(child.get("stage_count") or parent_stage_count or 1.0)
    parent_cost = float(parent.get("cascade_cost") or 0.0)
    child_cost = float(child.get("cascade_cost") or 0.0)
    parent_cofactor_closed = _parent_cofactor_closed(parent)
    child_cofactor_closed = float(bool(child.get("cofactor_closed")))
    child_condition_ok = float("ConditionConflict" not in set(child_failures))
    child_evidence_ok = float("EnzymeEvidenceWeak" not in set(child_failures))
    cofactor_req = sum(float(value or 0.0) for value in (step.get("cofactor_requirements") or {}).values())
    cofactor_regen = sum(float(value or 0.0) for value in (step.get("cofactor_regenerations") or {}).values())
    condition = step.get("condition") or {}
    if not isinstance(condition, dict):
        condition = {}
    scalars = np.asarray(
        [
            min(parent_step_count, 12.0) / 12.0,
            min(float(len(parent_open)), 12.0) / 12.0,
            min(parent_stage_count, 8.0) / 8.0,
            float(bool(parent.get("stock_closed"))),
            parent_cofactor_closed,
            min(float(len(parent_failures)), 12.0) / 12.0,
            min(parent_cost, 12.0) / 12.0,
            min(child_step_count, 12.0) / 12.0,
            min(float(len(child_open)), 12.0) / 12.0,
            min(child_stage_count, 8.0) / 8.0,
            float(bool(child.get("stock_closed"))),
            child_cofactor_closed,
            child_condition_ok,
            child_evidence_ok,
            min(float(len(child_failures)), 12.0) / 12.0,
            min(child_cost, 12.0) / 12.0,
            max(-12.0, min(12.0, float(len(parent_open) - len(child_open)))) / 12.0,
            max(-12.0, min(12.0, float(len(parent_failures) - len(child_failures)))) / 12.0,
            max(-12.0, min(12.0, parent_cost - child_cost)) / 12.0,
            max(-8.0, min(8.0, child_stage_count - parent_stage_count)) / 8.0,
            float(bool(child.get("stock_closed")) and not bool(parent.get("stock_closed"))),
            float(bool(child_cofactor_closed) and not bool(parent_cofactor_closed)),
            max(0.0, min(1.0, float(step.get("score") or 0.0))),
            float(bool(step.get("is_enzymatic") or step.get("ec_numbers") or step.get("enzyme_module"))),
            float(bool(condition)),
            min(float(_condition_field_count(condition)), 8.0) / 8.0,
            min(cofactor_req, 8.0) / 8.0,
            min(cofactor_regen, 8.0) / 8.0,
            min(float(len(step.get("ec_numbers") or []) + len(step.get("uniprot_ids") or [])), 8.0) / 8.0,
            min(float(len(step.get("reactant_smiles") or [])), 8.0) / 8.0,
        ],
        dtype=np.float32,
    )
    parent_failure_vec = _failure_vector(parent_failures)
    child_failure_vec = _failure_vector(child_failures)
    return np.concatenate(
        [
            morgan_fp(target, n_bits=n_bits),
            morgan_fp(product or expanded_leaf or "", n_bits=n_bits),
            morgan_fp(reactants, n_bits=n_bits),
            scalars,
            parent_failure_vec,
            child_failure_vec,
        ]
    ).astype(np.float32)


def transition_reward(parent_state: dict[str, Any], child_summary: dict[str, Any]) -> dict[str, float]:
    parent = dict(parent_state or {})
    child = dict(child_summary or {})
    parent_failures = _failure_categories(parent)
    child_failures = [str(item) for item in child.get("failure_categories") or []]
    parent_open = list(parent.get("open_molecule_leaves") or parent.get("open_leaves") or [])
    child_open = list(child.get("open_leaves") or [])
    parent_cost = float(parent.get("cascade_cost") or 0.0)
    child_cost = float(child.get("cascade_cost") or 0.0)
    failure_reduction = float(len(parent_failures) - len(child_failures))
    open_reduction = float(len(parent_open) - len(child_open))
    stock_closed = float(bool(child.get("stock_closed")))
    cofactor_closed = float(bool(child.get("cofactor_closed")))
    condition_ok = float("ConditionConflict" not in set(child_failures))
    evidence_ok = float("EnzymeEvidenceWeak" not in set(child_failures))
    stage_penalty = max(0.0, float(child.get("stage_count") or 1.0) - 1.0) * 0.05
    raw = (
        0.28 * stock_closed
        + 0.18 * cofactor_closed
        + 0.18 * condition_ok
        + 0.12 * evidence_ok
        + 0.10 * max(-1.0, min(1.0, open_reduction))
        + 0.10 * max(-1.0, min(1.0, failure_reduction))
        + 0.10 * max(-1.0, min(1.0, parent_cost - child_cost))
        - stage_penalty
    )
    value = max(0.0, min(1.0, raw))
    return {
        "transition_value": value,
        "stock_closed": stock_closed,
        "cofactor_closed": cofactor_closed,
        "condition_compatible": condition_ok,
        "evidence_sufficient": evidence_ok,
        "failure_reduction": failure_reduction,
        "open_leaf_reduction": open_reduction,
        "cost_reduction": parent_cost - child_cost,
    }


def _failure_categories(state: dict[str, Any]) -> list[str]:
    failures = state.get("unresolved_failure_modes") or state.get("failures") or []
    out = []
    for failure in failures:
        if isinstance(failure, dict):
            out.append(str(failure.get("category") or failure.get("kind") or ""))
        else:
            out.append(str(failure))
    return [item for item in out if item]


def _failure_vector(categories: list[str]) -> np.ndarray:
    counts = {category: categories.count(category) for category in set(categories)}
    return np.asarray([min(float(counts.get(kind, 0)), 4.0) / 4.0 for kind in TRANSITION_FAILURE_KINDS], dtype=np.float32)


def _stage_count(state: dict[str, Any]) -> int:
    graph = state.get("stage_graph") or {}
    if isinstance(graph, dict) and graph.get("stages"):
        return len(graph.get("stages") or [])
    partition = state.get("stage_partition") or []
    return max(1, len(set(partition))) if partition else 1


def _parent_cofactor_closed(parent: dict[str, Any]) -> float:
    ledger = parent.get("cofactor_ledger") or {}
    if isinstance(ledger, dict):
        unclosed = ledger.get("unclosed_requirements")
        if isinstance(unclosed, dict):
            return float(not bool(unclosed))
    return 1.0


def _condition_field_count(condition: dict[str, Any]) -> int:
    keys = [
        "temperature_c_min",
        "temperature_c_max",
        "ph_min",
        "ph_max",
        "solvents",
        "buffers",
        "catalysts",
        "cofactors",
    ]
    return sum(1 for key in keys if condition.get(key) not in (None, "", [], {}))


def _resize(vector: np.ndarray, dim: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if len(vector) == dim:
        return vector
    if len(vector) > dim:
        return vector[:dim].astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: len(vector)] = vector
    return out
