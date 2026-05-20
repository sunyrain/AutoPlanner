"""Local cascade compatibility scorers for adjacent reaction steps.

This module is the Stage-1 cascade signal: it scores whether two adjacent
steps are compatible as a one-pot, telescoped, or staged cascade fragment.
It is intentionally local and inspectable; trace-based action value models can
use these scores later, but they are not required for the first search-time
integration.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.cascade_search.state import CascadeAction, CascadeProgramState, ConditionEnvelope, StepAnnotation


PAIR_LABEL_NAMES = [
    "compatibility",
    "one_pot",
    "telescoped",
    "condition_compatible",
    "cofactor_compatible",
    "isolation_required",
    "biocascade",
]

PAIR_CATEGORICAL_FIELDS = [
    "route_domain",
    "left_pairwise_mode",
    "right_pairwise_mode",
    "left_step_mode",
    "right_step_mode",
    "left_transformation_superclass",
    "right_transformation_superclass",
    "left_catalyst_class",
    "right_catalyst_class",
    "left_ec1",
    "right_ec1",
    "left_solvent",
    "right_solvent",
]

PAIR_NUMERIC_FIELDS = [
    "left_temp",
    "right_temp",
    "temp_abs_diff",
    "temp_overlap",
    "left_ph",
    "right_ph",
    "ph_abs_diff",
    "ph_overlap",
    "solvent_match",
    "both_aqueous",
    "left_is_enzymatic",
    "right_is_enzymatic",
    "both_enzymatic",
    "mixed_chemo_enzymatic",
    "same_ec1",
    "cofactor_overlap",
    "cofactor_conflict",
    "redox_conflict",
    "intermediate_isolated",
    "missing_condition_fraction",
    "rule_compatibility",
    "rule_isolation_need",
]


@dataclass
class CascadePairScore:
    compatibility_score: float
    one_pot_probability: float
    telescoped_probability: float
    condition_compatible_probability: float
    cofactor_compatible_probability: float
    isolation_required_probability: float
    biocascade_probability: float
    reason: str = ""
    applicable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_reward(self) -> float:
        """A compact soft reward used for candidate ordering."""
        if not self.applicable:
            return 0.0
        reward = (
            0.34 * self.compatibility_score
            + 0.20 * self.one_pot_probability
            + 0.18 * self.telescoped_probability
            + 0.16 * self.condition_compatible_probability
            + 0.12 * self.cofactor_compatible_probability
            - 0.18 * self.isolation_required_probability
        )
        return _clip01(reward)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["search_reward"] = self.search_reward
        return data


class RuleCascadePairScorer:
    """Deterministic adjacent-step compatibility scorer.

    This is the rule baseline and the fallback when no learned checkpoint is
    provided. It is deliberately conservative: missing information returns a
    neutral score instead of hard rejection.
    """

    def score_action(
        self,
        state: CascadeProgramState,
        action: CascadeAction,
        child_state: CascadeProgramState | None = None,
        *,
        expanded_leaf: str | None = None,
    ) -> CascadePairScore:
        step = action.step
        if step is None:
            return _neutral_score("non_step_action")
        right = adjacent_downstream_step(state, step, expanded_leaf=expanded_leaf or action.target_leaf)
        if right is None:
            return _neutral_score("no_adjacent_downstream_step")
        return self.score_pair(step, right)

    def score_pair(self, left: StepAnnotation | dict[str, Any], right: StepAnnotation | dict[str, Any]) -> CascadePairScore:
        payload = pair_payload_from_steps(left, right)
        features = pair_rule_features(payload)
        condition = _condition_score(features)
        cofactor = _cofactor_score(features)
        catalyst = _catalyst_score(features)
        mode = _mode_score(payload)
        isolation = _isolation_need(payload, features)
        redox_penalty = 0.22 if features["redox_conflict"] > 0.0 else 0.0
        compatibility = _clip01(
            0.34 * condition
            + 0.20 * cofactor
            + 0.20 * catalyst
            + 0.18 * mode
            + 0.08 * features["both_enzymatic"]
            - 0.25 * isolation
            - redox_penalty
        )
        one_pot = _clip01(0.65 * condition + 0.20 * catalyst + 0.15 * cofactor - 0.65 * isolation)
        telescoped = _clip01(0.40 * condition + 0.20 * cofactor + 0.25 * mode + 0.15 * (1.0 - isolation))
        if _norm(payload.get("left_pairwise_mode")) in {"telescoped", "sequential_addition", "compartmentalized"}:
            telescoped = max(telescoped, 0.70)
        if _norm(payload.get("left_pairwise_mode")) == "isolated_transfer":
            one_pot = min(one_pot, 0.10)
            telescoped = min(telescoped, 0.45)
        return CascadePairScore(
            compatibility_score=compatibility,
            one_pot_probability=one_pot,
            telescoped_probability=telescoped,
            condition_compatible_probability=condition,
            cofactor_compatible_probability=cofactor,
            isolation_required_probability=isolation,
            biocascade_probability=_clip01(features["both_enzymatic"] * condition),
            reason="rule_pair_scorer",
            metadata={"rule_features": features, "payload_modes": _payload_modes(payload)},
        )


class LearnedCascadePairScorer:
    """Torch checkpoint-backed scorer trained by train_cascade_pair_scorer."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        import torch
        import torch.nn as nn

        self._torch = torch
        self._nn = nn
        self.device = torch.device(device)
        self.checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.feature_schema = dict(checkpoint["feature_schema"])
        self.label_names = list(checkpoint.get("label_names") or PAIR_LABEL_NAMES)
        hidden = int(checkpoint.get("hidden") or 128)
        input_dim = int(self.feature_schema["feature_dim"])
        self.model = _RuntimePairNetwork(nn, input_dim, hidden=hidden, output_dim=len(self.label_names)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def score_action(
        self,
        state: CascadeProgramState,
        action: CascadeAction,
        child_state: CascadeProgramState | None = None,
        *,
        expanded_leaf: str | None = None,
    ) -> CascadePairScore:
        step = action.step
        if step is None:
            return _neutral_score("non_step_action")
        right = adjacent_downstream_step(state, step, expanded_leaf=expanded_leaf or action.target_leaf)
        if right is None:
            return _neutral_score("no_adjacent_downstream_step")
        return self.score_pair(step, right)

    def score_pair(self, left: StepAnnotation | dict[str, Any], right: StepAnnotation | dict[str, Any]) -> CascadePairScore:
        payload = pair_payload_from_steps(left, right)
        x = pair_feature_vector(payload, self.feature_schema)
        with self._torch.no_grad():
            tensor = self._torch.tensor(x[None, :], dtype=self._torch.float32, device=self.device)
            probs = self._torch.sigmoid(self.model(tensor))[0].detach().cpu().numpy().tolist()
        values = {name: _clip01(float(probs[idx])) for idx, name in enumerate(self.label_names)}
        rule = RuleCascadePairScorer().score_pair(left, right)
        return CascadePairScore(
            compatibility_score=values.get("compatibility", rule.compatibility_score),
            one_pot_probability=values.get("one_pot", rule.one_pot_probability),
            telescoped_probability=values.get("telescoped", rule.telescoped_probability),
            condition_compatible_probability=values.get("condition_compatible", rule.condition_compatible_probability),
            cofactor_compatible_probability=values.get("cofactor_compatible", rule.cofactor_compatible_probability),
            isolation_required_probability=values.get("isolation_required", rule.isolation_required_probability),
            biocascade_probability=values.get("biocascade", rule.biocascade_probability),
            reason="learned_pair_scorer",
            metadata={
                "checkpoint_path": self.checkpoint_path,
                "raw_probabilities": values,
                "rule_fallback": rule.to_dict(),
            },
        )


def adjacent_downstream_step(
    state: CascadeProgramState,
    new_step: StepAnnotation,
    *,
    expanded_leaf: str | None = None,
) -> StepAnnotation | None:
    """Find the downstream step that consumes the product of ``new_step``."""
    tokens = {
        _norm(expanded_leaf),
        _norm(new_step.product_smiles),
    }
    tokens = {token for token in tokens if token}
    for step in reversed(state.step_annotations or state.steps or []):
        reactants = {_norm(value) for value in step.reactant_smiles}
        if tokens & reactants:
            return step
    return None


def build_pair_feature_schema(rows: list[dict[str, Any]], *, n_bits: int = 128) -> dict[str, Any]:
    categories = {
        field: sorted({_feature_field(row, field) for row in rows})
        for field in PAIR_CATEGORICAL_FIELDS
    }
    schema = {
        "schema_version": "cascade_pair_features.v1",
        "n_bits": int(n_bits),
        "label_names": list(PAIR_LABEL_NAMES),
        "categorical_fields": list(PAIR_CATEGORICAL_FIELDS),
        "categories": categories,
        "numeric_fields": list(PAIR_NUMERIC_FIELDS),
        "fingerprint_blocks": [
            "left_reactants",
            "left_products",
            "right_reactants",
            "right_products",
            "shared_intermediate",
        ],
        "feature_contract": "adjacent_step_pair_process_compatibility.v1",
    }
    dim = len(pair_feature_vector(rows[0], schema)) if rows else 0
    schema["feature_dim"] = dim
    return schema


def pair_feature_vector(row: dict[str, Any], schema: dict[str, Any]) -> np.ndarray:
    n_bits = int(schema.get("n_bits") or 128)
    out: list[float] = []
    for block in _fingerprint_blocks(row, n_bits=n_bits):
        out.extend(block.tolist())
    for field in schema.get("categorical_fields") or []:
        value = _feature_field(row, field)
        categories = (schema.get("categories") or {}).get(field) or []
        out.extend(1.0 if value == category else 0.0 for category in categories)
    features = pair_rule_features(row)
    out.extend(float(features.get(field, 0.0)) for field in schema.get("numeric_fields") or [])
    return np.asarray(out, dtype=np.float32)


def pair_payload_from_steps(
    left: StepAnnotation | dict[str, Any],
    right: StepAnnotation | dict[str, Any],
    *,
    route_domain: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_payload = _step_payload(left)
    right_payload = _step_payload(right)
    shared = _shared_intermediate(left_payload.get("rxn_smiles"), right_payload.get("rxn_smiles"))
    return {
        "route_domain": route_domain or "unknown",
        "left_step": left_payload,
        "right_step": right_payload,
        "shared_intermediate": shared,
        "left_pairwise_mode": left_payload.get("pairwise_mode") or "unknown",
        "right_pairwise_mode": right_payload.get("pairwise_mode") or "unknown",
        "metadata": dict(metadata or {}),
    }


def pair_rule_features(row: dict[str, Any]) -> dict[str, float]:
    left = row.get("left_step") or {}
    right = row.get("right_step") or {}
    left_cond = _condition_payload(left)
    right_cond = _condition_payload(right)
    left_temp = _safe_float(left_cond.get("temperature_c"))
    right_temp = _safe_float(right_cond.get("temperature_c"))
    left_ph = _safe_float(left_cond.get("ph"))
    right_ph = _safe_float(right_cond.get("ph"))
    left_solvent = _norm(left_cond.get("solvent") or left_cond.get("cosolvent"))
    right_solvent = _norm(right_cond.get("solvent") or right_cond.get("cosolvent"))
    left_cofactors = _cofactor_tokens(left)
    right_cofactors = _cofactor_tokens(right)
    left_redox = _redox_tokens(left_cond)
    right_redox = _redox_tokens(right_cond)
    left_enzyme = _is_enzymatic_step(left)
    right_enzyme = _is_enzymatic_step(right)
    missing_condition = float(not left_cond) + float(not right_cond)
    temp_overlap = _range_or_point_overlap(
        left_cond.get("temperature_c_min"),
        left_cond.get("temperature_c_max"),
        left_temp,
        right_cond.get("temperature_c_min"),
        right_cond.get("temperature_c_max"),
        right_temp,
    )
    ph_overlap = _range_or_point_overlap(
        left_cond.get("ph_min"),
        left_cond.get("ph_max"),
        left_ph,
        right_cond.get("ph_min"),
        right_cond.get("ph_max"),
        right_ph,
    )
    solvent_match = 0.5
    if left_solvent and right_solvent:
        solvent_match = 1.0 if left_solvent == right_solvent else 0.0
    features = {
        "left_temp": _scale(left_temp, 100.0),
        "right_temp": _scale(right_temp, 100.0),
        "temp_abs_diff": _scale(_abs_diff(left_temp, right_temp), 100.0),
        "temp_overlap": temp_overlap,
        "left_ph": _scale(left_ph, 14.0),
        "right_ph": _scale(right_ph, 14.0),
        "ph_abs_diff": _scale(_abs_diff(left_ph, right_ph), 14.0),
        "ph_overlap": ph_overlap,
        "solvent_match": solvent_match,
        "both_aqueous": float(left_solvent in {"water", "aqueous", "buffer"} and right_solvent in {"water", "aqueous", "buffer"}),
        "left_is_enzymatic": float(left_enzyme),
        "right_is_enzymatic": float(right_enzyme),
        "both_enzymatic": float(left_enzyme and right_enzyme),
        "mixed_chemo_enzymatic": float(left_enzyme != right_enzyme),
        "same_ec1": float(bool(_ec1(left)) and _ec1(left) == _ec1(right)),
        "cofactor_overlap": float(bool(left_cofactors & right_cofactors)),
        "cofactor_conflict": float(_cofactor_conflict(left_cofactors, right_cofactors)),
        "redox_conflict": float(_redox_conflict(left_redox, right_redox)),
        "intermediate_isolated": float(_truthy(left.get("intermediate_isolated")) or _norm(row.get("left_pairwise_mode")) == "isolated_transfer"),
        "missing_condition_fraction": missing_condition / 2.0,
    }
    condition = _condition_score(features)
    cofactor = _cofactor_score(features)
    isolation = _isolation_need(row, features)
    features["rule_compatibility"] = _clip01(0.45 * condition + 0.25 * cofactor + 0.20 * _catalyst_score(features) - 0.20 * isolation)
    features["rule_isolation_need"] = isolation
    return features


def _condition_score(features: dict[str, float]) -> float:
    if features["missing_condition_fraction"] >= 1.0:
        return 0.55
    score = 0.45 * features["temp_overlap"] + 0.35 * features["ph_overlap"] + 0.20 * features["solvent_match"]
    if features["both_aqueous"] > 0.0:
        score += 0.08
    return _clip01(score)


def _cofactor_score(features: dict[str, float]) -> float:
    if features["cofactor_conflict"] > 0.0:
        return 0.20
    if features["cofactor_overlap"] > 0.0:
        return 0.85
    return 0.75


def _catalyst_score(features: dict[str, float]) -> float:
    if features["both_enzymatic"] > 0.0:
        return 0.85 if features["temp_overlap"] and features["ph_overlap"] else 0.55
    if features["mixed_chemo_enzymatic"] > 0.0:
        return 0.60 if features["solvent_match"] > 0.0 else 0.35
    return 0.70


def _mode_score(row: dict[str, Any]) -> float:
    mode = _norm(row.get("left_pairwise_mode"))
    if mode == "simultaneous":
        return 0.95
    if mode in {"telescoped", "sequential_addition", "compartmentalized"}:
        return 0.75
    if mode == "isolated_transfer":
        return 0.20
    return 0.55


def _isolation_need(row: dict[str, Any], features: dict[str, float]) -> float:
    mode = _norm(row.get("left_pairwise_mode"))
    if mode == "isolated_transfer" or features["intermediate_isolated"] > 0.0:
        return 0.95
    if features["redox_conflict"] > 0.0 or features["cofactor_conflict"] > 0.0:
        return 0.65
    if features["temp_overlap"] <= 0.0 and features["ph_overlap"] <= 0.0 and features["solvent_match"] <= 0.0:
        return 0.60
    if mode in {"telescoped", "sequential_addition"}:
        return 0.20
    return 0.08


def _fingerprint_blocks(row: dict[str, Any], *, n_bits: int) -> list[np.ndarray]:
    left = row.get("left_step") or {}
    right = row.get("right_step") or {}
    left_reactants, left_products = _split_reaction(left.get("rxn_smiles"))
    right_reactants, right_products = _split_reaction(right.get("rxn_smiles"))
    shared = row.get("shared_intermediate") or _shared_intermediate(left.get("rxn_smiles"), right.get("rxn_smiles"))
    return [
        _fp_many(left_reactants, n_bits=n_bits),
        _fp_many(left_products, n_bits=n_bits),
        _fp_many(right_reactants, n_bits=n_bits),
        _fp_many(right_products, n_bits=n_bits),
        _fp_many([shared], n_bits=n_bits),
    ]


def _feature_field(row: dict[str, Any], field: str) -> str:
    left = row.get("left_step") or {}
    right = row.get("right_step") or {}
    if field == "route_domain":
        return _norm(row.get("route_domain")) or "unknown"
    if field == "left_pairwise_mode":
        return _norm(row.get("left_pairwise_mode") or left.get("pairwise_mode")) or "unknown"
    if field == "right_pairwise_mode":
        return _norm(row.get("right_pairwise_mode") or right.get("pairwise_mode")) or "unknown"
    if field == "left_step_mode":
        return _norm(left.get("step_mode")) or "unknown"
    if field == "right_step_mode":
        return _norm(right.get("step_mode")) or "unknown"
    if field == "left_transformation_superclass":
        return _norm(left.get("transformation_superclass")) or "unknown"
    if field == "right_transformation_superclass":
        return _norm(right.get("transformation_superclass")) or "unknown"
    if field == "left_catalyst_class":
        return _dominant_catalyst_class(left)
    if field == "right_catalyst_class":
        return _dominant_catalyst_class(right)
    if field == "left_ec1":
        return _ec1(left) or "none"
    if field == "right_ec1":
        return _ec1(right) or "none"
    if field == "left_solvent":
        return _norm(_condition_payload(left).get("solvent")) or "unknown"
    if field == "right_solvent":
        return _norm(_condition_payload(right).get("solvent")) or "unknown"
    return "unknown"


def _step_payload(step: StepAnnotation | dict[str, Any]) -> dict[str, Any]:
    if isinstance(step, StepAnnotation):
        catalysts = []
        if step.is_enzymatic:
            catalysts.append({"catalyst_class": "enzyme", "ec_number": (step.ec_numbers or [""])[0]})
        condition = step.condition.to_dict() if step.condition is not None else {}
        return {
            "rxn_smiles": step.rxn_smiles,
            "step_conditions": _condition_to_v4_like(condition),
            "catalyst_components": catalysts,
            "pairwise_mode": step.raw_metadata.get("pairwise_mode") or "unknown",
            "step_mode": step.raw_metadata.get("step_mode") or "unknown",
            "transformation_superclass": step.reaction_type or step.raw_metadata.get("transformation_superclass") or "unknown",
            "intermediate_isolated": step.raw_metadata.get("intermediate_isolated"),
            "cofactor_requirements": step.all_cofactor_requirements(),
            "cofactor_regenerations": step.all_cofactor_regenerations(),
        }
    payload = dict(step or {})
    if "step_conditions" not in payload and "condition" in payload:
        payload["step_conditions"] = payload.get("condition") or {}
    return payload


def _condition_to_v4_like(condition: dict[str, Any]) -> dict[str, Any]:
    if not condition:
        return {}
    temp = condition.get("temperature_c")
    if temp is None:
        vals = [condition.get("temperature_c_min"), condition.get("temperature_c_max")]
        clean = [float(v) for v in vals if v is not None]
        temp = sum(clean) / len(clean) if clean else None
    ph = condition.get("ph")
    if ph is None:
        vals = [condition.get("ph_min"), condition.get("ph_max")]
        clean = [float(v) for v in vals if v is not None]
        ph = sum(clean) / len(clean) if clean else None
    solvents = condition.get("solvents") or []
    return {
        "temperature_c": temp,
        "temperature_c_min": condition.get("temperature_c_min"),
        "temperature_c_max": condition.get("temperature_c_max"),
        "ph": ph,
        "ph_min": condition.get("ph_min"),
        "ph_max": condition.get("ph_max"),
        "solvent": solvents[0] if solvents else condition.get("solvent"),
        "cosolvent": condition.get("cosolvent"),
        "atmosphere": condition.get("oxygen_requirement") or condition.get("atmosphere"),
        "cofactors": condition.get("cofactors") or [],
    }


def _condition_payload(step: dict[str, Any]) -> dict[str, Any]:
    condition = dict(step.get("step_conditions") or step.get("condition") or {})
    if isinstance(condition.get("solvents"), list) and condition["solvents"] and not condition.get("solvent"):
        condition["solvent"] = condition["solvents"][0]
    if condition.get("temperature_c") is None:
        vals = [_safe_float(condition.get("temperature_c_min")), _safe_float(condition.get("temperature_c_max"))]
        clean = [v for v in vals if v is not None]
        if clean:
            condition["temperature_c"] = sum(clean) / len(clean)
    if condition.get("ph") is None:
        vals = [_safe_float(condition.get("ph_min")), _safe_float(condition.get("ph_max"))]
        clean = [v for v in vals if v is not None]
        if clean:
            condition["ph"] = sum(clean) / len(clean)
    return condition


def _dominant_catalyst_class(step: dict[str, Any]) -> str:
    values = [_norm(item.get("catalyst_class")) for item in step.get("catalyst_components") or [] if item]
    values = [value for value in values if value]
    if not values:
        return "none"
    return Counter(values).most_common(1)[0][0]


def _is_enzymatic_step(step: dict[str, Any]) -> bool:
    for item in step.get("catalyst_components") or []:
        text = " ".join(str(item.get(key) or "") for key in ("catalyst_class", "ec_number", "component_name")).lower()
        if any(token in text for token in ("enzyme", "whole_cell", "ec ", "oxidoreductase", "transferase")) or item.get("ec_number"):
            return True
    text = " ".join(str(step.get(key) or "") for key in ("transformation_superclass", "transformation_name")).lower()
    return "enzyme" in text or "biocatal" in text


def _ec1(step: dict[str, Any]) -> str:
    for item in step.get("catalyst_components") or []:
        ec = str(item.get("ec_number") or "")
        if ec:
            return ec.split(".", 1)[0].strip().lower()
    return ""


def _cofactor_tokens(step: dict[str, Any]) -> set[str]:
    out = set()
    for item in step.get("catalyst_components") or []:
        value = item.get("cofactor_required")
        if value:
            out.add(_norm(value))
        value = item.get("cofactor_regeneration_mode")
        if value:
            out.add(_norm(value))
    for key in ("cofactor_requirements", "cofactor_regenerations"):
        values = step.get(key) or {}
        if isinstance(values, dict):
            out.update(_norm(name) for name in values if _norm(name))
    condition = _condition_payload(step)
    for value in condition.get("cofactors") or []:
        out.add(_norm(value))
    return {value for value in out if value}


def _redox_tokens(condition: dict[str, Any]) -> set[str]:
    out = set()
    atmosphere = _norm(condition.get("atmosphere"))
    if atmosphere in {"aerobic", "oxygen", "o2"}:
        out.add("oxidizing")
    if atmosphere in {"anaerobic", "nitrogen", "argon", "n2"}:
        out.add("anaerobic")
    for key in ("oxidant", "oxidants"):
        value = condition.get(key)
        if isinstance(value, list):
            out.update("oxidizing" for _ in value if _)
        elif value:
            out.add("oxidizing")
    for key in ("reductant", "reductants"):
        value = condition.get(key)
        if isinstance(value, list):
            out.update("reducing" for _ in value if _)
        elif value:
            out.add("reducing")
    return out


def _cofactor_conflict(left: set[str], right: set[str]) -> bool:
    antagonists = [
        ({"nadh", "nadph"}, {"nad+", "nadp+"}),
        ({"h2o2"}, {"nadph", "nadh"}),
    ]
    return any((left & a and right & b) or (left & b and right & a) for a, b in antagonists)


def _redox_conflict(left: set[str], right: set[str]) -> bool:
    return bool(("oxidizing" in left and "reducing" in right) or ("reducing" in left and "oxidizing" in right))


def _split_reaction(rxn_smiles: Any) -> tuple[list[str], list[str]]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return [], []
    left, right = text.split(">>", 1)
    return _split_smiles_side(left), _split_smiles_side(right)


def _split_smiles_side(text: str) -> list[str]:
    return [item.strip() for item in text.split(".") if item.strip()]


def _shared_intermediate(left_rxn: Any, right_rxn: Any) -> str:
    _, left_products = _split_reaction(left_rxn)
    right_reactants, _ = _split_reaction(right_rxn)
    left_keys = {_canonical_smiles(value): value for value in left_products if value}
    for reactant in right_reactants:
        key = _canonical_smiles(reactant)
        if key and key in left_keys:
            return key
    return ""


def _fp_many(smiles_values: list[str], *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return arr
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles or ""))
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        tmp = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, tmp)
        arr = np.maximum(arr, tmp)
    return arr


def _canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles or ""))
        return Chem.MolToSmiles(mol) if mol is not None else ""
    except Exception:
        return str(smiles or "").strip()


class _RuntimePairNetwork:
    def __init__(self, nn: Any, input_dim: int, *, hidden: int, output_dim: int):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                h2 = max(32, hidden // 2)
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden, h2),
                    nn.GELU(),
                    nn.Linear(h2, output_dim),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        self._model = Net()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def _range_or_point_overlap(
    left_min: Any,
    left_max: Any,
    left_point: float | None,
    right_min: Any,
    right_max: Any,
    right_point: float | None,
) -> float:
    lmin = _safe_float(left_min)
    lmax = _safe_float(left_max)
    rmin = _safe_float(right_min)
    rmax = _safe_float(right_max)
    if lmin is None and lmax is None and left_point is not None:
        lmin = left_point - 5.0
        lmax = left_point + 5.0
    if rmin is None and rmax is None and right_point is not None:
        rmin = right_point - 5.0
        rmax = right_point + 5.0
    if None in {lmin, lmax, rmin, rmax}:
        return 0.55
    return float(max(float(lmin), float(rmin)) <= min(float(lmax), float(rmax)))


def _payload_modes(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "left_pairwise_mode": payload.get("left_pairwise_mode"),
        "right_pairwise_mode": payload.get("right_pairwise_mode"),
    }


def _neutral_score(reason: str) -> CascadePairScore:
    return CascadePairScore(
        compatibility_score=0.50,
        one_pot_probability=0.45,
        telescoped_probability=0.40,
        condition_compatible_probability=0.55,
        cofactor_compatible_probability=0.75,
        isolation_required_probability=0.10,
        biocascade_probability=0.0,
        reason=reason,
        applicable=False,
    )


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _abs_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _scale(value: Any, denom: float) -> float:
    numeric = _safe_float(value)
    if numeric is None:
        return 0.0
    return _clip01(numeric / float(denom))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _norm(value) in {"1", "true", "yes", "y"}
