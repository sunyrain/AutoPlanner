"""Default cascade value, source-policy, and compatibility components.

These components are production search modules with deterministic behavior.
Learned implementations can replace them through the formal interfaces without
changing the cascade state machine.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.cost import score_cascade_state
from cascade_planner.cascade_search.state import (
    CascadeModule,
    CascadeProgramState,
    ConditionEnvelope,
    StepAnnotation,
)
from cascade_planner.cascade_verifier import verify_cascade_route


@dataclass
class SourceBudget:
    source_budgets: dict[str, int]
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionTransitionPrediction:
    same_pot_probability: float
    telescoped_probability: float
    isolation_required_probability: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CofactorClosurePrediction:
    status: str
    unclosed_requirements: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CascadeValuePrediction:
    p_stock_closed: float
    p_condition_compatible: float
    p_cofactor_closed: float
    p_enzyme_evidence_valid: float
    p_gt_like_cascade: float
    expected_remaining_depth: float
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CascadeSourcePolicy:
    """Default source policy over cascade state and failure modes."""

    def allocate(
        self,
        state: CascadeProgramState,
        *,
        available_sources: list[str],
        total_budget: int,
    ) -> SourceBudget:
        total_budget = max(1, int(total_budget or 1))
        if not available_sources:
            return SourceBudget(source_budgets={}, reason="no_available_sources")
        failures = {failure.category for failure in state.unresolved_failure_modes}
        weights = {source: 1.0 for source in available_sources}
        if "CandidateMissing" in failures or "StockDeadEnd" in failures:
            for source in available_sources:
                weights[source] += 1.0
        if "EnzymeEvidenceWeak" in failures or state.cofactor_ledger.unclosed_requirements():
            for source in available_sources:
                if any(token in source.lower() for token in ("enzyme", "enzy", "rhea", "retrorules", "v3")):
                    weights[source] += 2.0
        total = sum(weights.values())
        budgets = {source: max(0, int(round(total_budget * weights[source] / total))) for source in available_sources}
        for source in available_sources:
            if budgets[source] <= 0:
                budgets[source] = 1
        while sum(budgets.values()) > total_budget:
            source = max(budgets, key=lambda key: budgets[key])
            budgets[source] -= 1
        return SourceBudget(
            source_budgets=budgets,
            reason="failure_aware_allocation",
            metadata={"failures": sorted(failures)},
        )


class EnzymeModuleRanker:
    """Score whether an enzyme module fits the current cascade state."""

    def score(self, module: CascadeModule, state: CascadeProgramState, *, stage_id: str | None = None) -> float:
        score = 0.5
        sid = stage_id or state.current_stage
        stage_condition = state.condition_envelope_by_stage.get(sid)
        if module.condition_envelope is not None and stage_condition is not None:
            score += 0.25 if stage_condition.overlaps(module.condition_envelope) else -0.25
        if module.evidence_confidence is not None:
            score += 0.25 * max(0.0, min(1.0, float(module.evidence_confidence)))
        unclosed = state.cofactor_ledger.unclosed_requirements()
        if module.cofactor_regenerations and any(name in unclosed for name in module.cofactor_regenerations):
            score += 0.2
        if module.cofactor_requirements:
            score -= 0.1
        return max(0.0, min(1.0, score))


class ConditionTransitionModel:
    """Default model for same-pot/telescoped/isolation decisions."""

    def predict(
        self,
        left: ConditionEnvelope | StepAnnotation | None,
        right: ConditionEnvelope | StepAnnotation | None,
    ) -> ConditionTransitionPrediction:
        left_env = left.condition if isinstance(left, StepAnnotation) else left
        right_env = right.condition if isinstance(right, StepAnnotation) else right
        if left_env is None or right_env is None:
            return ConditionTransitionPrediction(0.55, 0.35, 0.10, reason="missing_condition")
        if left_env.overlaps(right_env):
            return ConditionTransitionPrediction(0.80, 0.18, 0.02, reason="condition_overlap")
        solvent_conflict = bool(left_env.solvents and right_env.solvents and not (left_env.normalized_solvents() & right_env.normalized_solvents()))
        ph_conflict = not _ranges_overlap(left_env.ph_min, left_env.ph_max, right_env.ph_min, right_env.ph_max)
        temp_conflict = not _ranges_overlap(
            left_env.temperature_c_min,
            left_env.temperature_c_max,
            right_env.temperature_c_min,
            right_env.temperature_c_max,
        )
        if solvent_conflict and (ph_conflict or temp_conflict):
            return ConditionTransitionPrediction(0.05, 0.35, 0.60, reason="multi_axis_condition_conflict")
        return ConditionTransitionPrediction(0.15, 0.65, 0.20, reason="single_axis_condition_conflict")


class CofactorClosureModel:
    """Default cofactor closure classifier."""

    REGENERABLE = {"nadh", "nadph", "nad+", "nadp+", "fad", "fmn", "atp", "plp"}

    def predict(self, state: CascadeProgramState) -> CofactorClosurePrediction:
        unclosed = state.cofactor_ledger.unclosed_requirements()
        if not unclosed:
            return CofactorClosurePrediction("closed", reason="no_unclosed_requirements")
        if all(name.lower() in self.REGENERABLE for name in unclosed):
            return CofactorClosurePrediction("regenerable", unclosed, reason="known_regenerable_cofactor")
        if sum(float(value) for value in unclosed.values()) <= 0.25:
            return CofactorClosurePrediction("unclosed_but_tolerable", unclosed, reason="small_debt")
        return CofactorClosurePrediction("fatal_cross_talk", unclosed, reason="unknown_or_large_cofactor_debt")


class HeuristicCascadeValueModel:
    """Default deterministic cascade value model.

    The class is intentionally named for what it is: a stable heuristic
    implementation of the formal value-model contract. It is suitable for
    search bootstrapping and trace generation, while learned value models can
    replace it behind the same ``predict(state)`` interface.
    """

    def predict(self, state: CascadeProgramState) -> CascadeValuePrediction:
        cost = score_cascade_state(state)
        p_stock = cost.stock_reachability
        p_condition = cost.condition_compatibility
        p_cofactor = cost.cofactor_closure
        p_enzyme = cost.enzyme_evidence
        p_gt_like = max(0.0, min(1.0, 1.0 - cost.total_cost / 6.0))
        remaining = float(len([leaf for leaf in state.open_molecule_leaves if not state.stock_status.get(leaf)]))
        value = (
            0.25 * p_stock
            + 0.25 * p_condition
            + 0.20 * p_cofactor
            + 0.15 * p_enzyme
            + 0.15 * p_gt_like
        )
        return CascadeValuePrediction(
            p_stock_closed=p_stock,
            p_condition_compatible=p_condition,
            p_cofactor_closed=p_cofactor,
            p_enzyme_evidence_valid=p_enzyme,
            p_gt_like_cascade=p_gt_like,
            expected_remaining_depth=remaining,
            value=max(0.0, min(1.0, value)),
            metadata={"cascade_cost": cost.to_dict(), "model_family": "heuristic"},
        )


# Backward-compatible alias for earlier cascade-search experiments.
RuleCascadeValueModel = HeuristicCascadeValueModel


class VerifierAugmentedCascadeValueModel:
    """Blend the deterministic cascade value with the rule verifier score.

    This is the first search-facing adapter for verifier-first training. It
    does not replace the existing heuristic value model; it exposes verifier
    failures as value metadata and softly lowers the value of states that fail
    material, condition, cofactor, enzyme-toxicity, or ordering checks.
    """

    def __init__(self, base_model: Any | None = None, *, verifier_weight: float = 0.35):
        self.base_model = base_model or HeuristicCascadeValueModel()
        self.verifier_weight = _clip01(verifier_weight)

    def predict(self, state: CascadeProgramState) -> CascadeValuePrediction:
        base = self.base_model.predict(state)
        verifier_report = verify_cascade_route(_route_dict_from_state(state), target_smiles=state.target_smiles).to_dict()
        verifier_score = _clip01(float(verifier_report.get("score") or 0.0))
        value = _clip01((1.0 - self.verifier_weight) * float(base.value) + self.verifier_weight * verifier_score)
        condition = min(float(base.p_condition_compatible), verifier_score)
        cofactor = float(base.p_cofactor_closed)
        if "cofactor_ledger_gap" in (verifier_report.get("reason_counts") or {}):
            cofactor = min(cofactor, verifier_score)
        enzyme = float(base.p_enzyme_evidence_valid)
        if "enzyme_toxicity" in (verifier_report.get("reason_counts") or {}):
            enzyme = min(enzyme, verifier_score)
        metadata = dict(base.metadata or {})
        metadata.update(
            {
                "model_family": "verifier_augmented_heuristic",
                "base_value": base.to_dict(),
                "verifier_report": verifier_report,
                "verifier_weight": self.verifier_weight,
            }
        )
        return CascadeValuePrediction(
            p_stock_closed=base.p_stock_closed,
            p_condition_compatible=_clip01(condition),
            p_cofactor_closed=_clip01(cofactor),
            p_enzyme_evidence_valid=_clip01(enzyme),
            p_gt_like_cascade=min(float(base.p_gt_like_cascade), verifier_score),
            expected_remaining_depth=base.expected_remaining_depth,
            value=value,
            metadata=metadata,
        )


class LoadedLearnedVerifierValueModel:
    """Joblib-backed learned verifier value adapter.

    The artifact is produced by ``scripts/train_cascade_verifier_from_pack.py``.
    It remains optional and lazy: default search still uses the heuristic/rule
    value path unless this class is explicitly configured.
    """

    def __init__(self, model_path: str | Path, base_model: Any | None = None, *, learned_weight: float = 0.35):
        import joblib

        self.model_path = str(model_path)
        self.base_model = base_model or HeuristicCascadeValueModel()
        self.learned_weight = _clip01(learned_weight)
        artifact = joblib.load(self.model_path)
        self.vectorizer = artifact["vectorizer"]
        self.feasible_model = artifact["feasible_model"]
        self.reason_models = dict(artifact.get("reason_models") or {})
        self.reason_labels = list(artifact.get("reason_labels") or [])
        self.training_summary = dict(artifact.get("summary") or {})

    def predict(self, state: CascadeProgramState) -> CascadeValuePrediction:
        # Reuse the exact feature function used for training. This import is
        # intentionally lazy so sklearn is not required for default search.
        from scripts.train_cascade_verifier_from_pack import _features

        base = self.base_model.predict(state)
        route = _route_dict_from_state(state)
        example = {
            "target_smiles": state.target_smiles,
            "cascade": route,
            "expected_failure_reasons": [],
        }
        x = self.vectorizer.transform([_features(example)])
        feasible_probability = _positive_probability(self.feasible_model, x)
        reason_probabilities = {
            reason: _positive_probability(model, x)
            for reason, model in self.reason_models.items()
        }
        learned_score = _clip01(feasible_probability)
        value = _clip01((1.0 - self.learned_weight) * float(base.value) + self.learned_weight * learned_score)
        condition = min(float(base.p_condition_compatible), learned_score)
        cofactor = float(base.p_cofactor_closed)
        if reason_probabilities.get("cofactor_ledger_gap", 0.0) >= 0.5:
            cofactor = min(cofactor, learned_score)
        enzyme = float(base.p_enzyme_evidence_valid)
        if reason_probabilities.get("enzyme_toxicity", 0.0) >= 0.5:
            enzyme = min(enzyme, learned_score)
        metadata = dict(base.metadata or {})
        metadata.update(
            {
                "model_family": "learned_verifier_augmented_heuristic",
                "base_value": base.to_dict(),
                "learned_verifier": {
                    "model_path": self.model_path,
                    "feasible_probability": round(float(feasible_probability), 6),
                    "reason_probabilities": {
                        key: round(float(value), 6)
                        for key, value in sorted(reason_probabilities.items())
                    },
                    "training_summary": self.training_summary,
                },
                "learned_weight": self.learned_weight,
            }
        )
        return CascadeValuePrediction(
            p_stock_closed=base.p_stock_closed,
            p_condition_compatible=_clip01(condition),
            p_cofactor_closed=_clip01(cofactor),
            p_enzyme_evidence_valid=_clip01(enzyme),
            p_gt_like_cascade=min(float(base.p_gt_like_cascade), learned_score),
            expected_remaining_depth=base.expected_remaining_depth,
            value=value,
            metadata=metadata,
        )


def _positive_probability(model: Any, x: Any) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return float(proba[classes.index(1)])
        if classes:
            return 1.0 if int(classes[0]) == 1 else 0.0
    pred = model.predict(x)[0]
    return float(pred)


def _route_dict_from_state(state: CascadeProgramState) -> dict[str, Any]:
    steps = []
    for step in state.step_annotations or state.steps or []:
        condition = step.condition
        temp = condition.temperature_c_min if condition is not None else None
        ph = condition.ph_min if condition is not None else None
        solvent = (condition.solvents or [""])[0] if condition is not None else ""
        steps.append(
            {
                "product": step.product_smiles,
                "main_reactant": (step.reactant_smiles or [""])[0],
                "aux_reactants": list(step.reactant_smiles or [])[1:],
                "reactants": list(step.reactant_smiles or []),
                "reaction_smiles": step.rxn_smiles,
                "source": step.source_model,
                "reaction_type": step.reaction_type,
                "ec": (step.ec_numbers or [""])[0],
                "T": temp,
                "pH": ph,
                "solvent": solvent,
                "cofactor_requirements": dict(step.cofactor_requirements or {}),
                "cofactor_regenerations": dict(step.cofactor_regenerations or {}),
                "enzyme_ec_annotations": [{"ec_number": ec, "confidence": 1.0} for ec in step.ec_numbers or []],
            }
        )
    partition = state.stage_partition or state.stage_graph.to_partition(len(steps))
    return {
        "target": state.target_smiles,
        "steps": steps,
        "stage_partition": partition,
    }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _span(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return max(clean) - min(clean) if clean else 0.0


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


class LearnedCascadeValueModel:
    """Torch checkpoint-backed value model for CascadeProgramState objects."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        import numpy as np
        import torch
        import torch.nn as nn
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        self._np = np
        self._torch = torch
        self._nn = nn
        self._Chem = Chem
        self._DataStructs = DataStructs
        self._AllChem = AllChem
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.feature_schema = dict(checkpoint.get("feature_schema") or {})
        self.binary_labels = list(checkpoint.get("binary_labels") or self.feature_schema.get("binary_labels") or [])
        hidden = int(checkpoint.get("hidden") or 192)
        in_dim = int(self.feature_schema.get("feature_dim") or 0)
        if in_dim <= 0:
            raise ValueError(f"invalid learned cascade value checkpoint feature_dim: {checkpoint_path}")
        self.model = _RuntimeCascadeValueNetwork(self._nn, in_dim, hidden=hidden, n_binary=len(self.binary_labels)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, state: CascadeProgramState) -> CascadeValuePrediction:
        x = self._state_features(state)
        with self._torch.no_grad():
            tensor = self._torch.tensor(x[None, :], dtype=self._torch.float32, device=self.device)
            out = self.model(tensor)
            probs = self._torch.sigmoid(out["binary_logits"])[0].detach().cpu().numpy().tolist()
            depth = float(out["depth"][0].detach().cpu().item()) * 8.0
        values = {label: float(probs[idx]) for idx, label in enumerate(self.binary_labels)}
        heuristic = HeuristicCascadeValueModel().predict(state)
        p_stock = values.get("p_stock_closed", heuristic.p_stock_closed)
        p_condition = values.get("p_condition_compatible", heuristic.p_condition_compatible)
        p_cofactor = values.get("p_cofactor_closed", heuristic.p_cofactor_closed)
        p_enzyme = values.get("p_enzyme_evidence_valid", heuristic.p_enzyme_evidence_valid)
        p_gt = values.get("p_gt_like_cascade", heuristic.p_gt_like_cascade)
        p_stage_transition = values.get("p_stage_transition_needed")
        value = (
            0.22 * p_stock
            + 0.22 * p_condition
            + 0.18 * p_cofactor
            + 0.13 * p_enzyme
            + 0.25 * p_gt
        )
        if p_stage_transition is not None and state.stage_graph.n_stages <= 1:
            value -= 0.05 * p_stage_transition
        return CascadeValuePrediction(
            p_stock_closed=_clip01(p_stock),
            p_condition_compatible=_clip01(p_condition),
            p_cofactor_closed=_clip01(p_cofactor),
            p_enzyme_evidence_valid=_clip01(p_enzyme),
            p_gt_like_cascade=_clip01(p_gt),
            expected_remaining_depth=max(0.0, depth),
            value=_clip01(value),
            metadata={
                "model_family": "learned_cascade_value",
                "checkpoint_path": self.checkpoint_path,
                "raw_probabilities": values,
                "p_stage_transition_needed": p_stage_transition,
            },
        )

    def _state_features(self, state: CascadeProgramState) -> Any:
        schema = self.feature_schema
        n_bits = int(schema.get("n_bits") or 128)
        steps = list(state.step_annotations or [])
        target_fp = self._fp_many([state.target_smiles], n_bits=n_bits)
        open_leaves = list(state.open_molecule_leaves or state.open_leaves or [])
        start_fp = self._fp_many(open_leaves or [state.target_smiles], n_bits=n_bits)
        total_steps = max(1.0, float(len(steps) or 1))
        rxn_steps = float(sum(1 for step in steps if step.rxn_smiles))
        reactant_count = float(sum(len(step.reactant_smiles or []) for step in steps))
        product_count = float(len({step.product_smiles for step in steps if step.product_smiles}))
        catalyst_tokens = self._catalyst_classes(state)
        catalyst_count = max(1.0, float(len(catalyst_tokens)))
        conditions = [step.condition for step in steps if step.condition is not None]
        temp_values = [
            value
            for env in conditions
            for value in (env.temperature_c_min, env.temperature_c_max)
            if value is not None
        ]
        ph_values = [
            value
            for env in conditions
            for value in (env.ph_min, env.ph_max)
            if value is not None
        ]
        enzyme_count = float(sum(1 for step in steps if step.is_enzymatic))
        identifier_count = float(sum(1 for step in steps if step.ec_numbers or step.uniprot_ids))
        cofactor_count = float(sum(1 for step in steps if step.all_cofactor_requirements()))
        condition_count = float(sum(self._condition_field_count(env) for env in conditions))
        evidence_values = [step.evidence_confidence for step in steps if step.evidence_confidence is not None]
        failures = {failure.category for failure in state.unresolved_failure_modes or []}
        unclosed = state.cofactor_ledger.unclosed_requirements()
        scalar = self._np.asarray(
            [
                total_steps / 8.0,
                rxn_steps / total_steps,
                catalyst_count / max(total_steps, 1.0) / 4.0,
                reactant_count / 16.0,
                product_count / 16.0,
                0.0,
                condition_count / 12.0,
                0.0,
                float(state.stock_closed),
                float(bool(conditions)),
                float(bool(state.unresolved_failure_modes)),
                0.0,
                _mean(evidence_values),
                enzyme_count / max(catalyst_count, 1.0),
                identifier_count / max(catalyst_count, 1.0),
                cofactor_count / max(catalyst_count, 1.0),
                _mean(temp_values) / 100.0,
                _span(temp_values) / 100.0,
                _mean(ph_values) / 14.0,
                _span(ph_values) / 14.0,
                0.0,
                float(bool(unclosed)),
                float("ConditionConflict" in failures),
                0.0,
                0.0,
            ],
            dtype=self._np.float32,
        )
        bags = [
            self._bag([self._cascade_type(state)], schema.get("cascade_type_vocab") or []),
            self._bag(self._transformations(state), schema.get("transformation_vocab") or [], denom=total_steps),
            self._bag(self._step_modes(state), schema.get("step_mode_vocab") or [], denom=total_steps),
            self._bag(catalyst_tokens, schema.get("catalyst_class_vocab") or [], denom=catalyst_count),
            self._bag(self._ec1_values(state), schema.get("ec1_vocab") or [], denom=catalyst_count),
            self._bag(self._solvents(state), schema.get("solvent_vocab") or [], denom=total_steps),
            self._bag(self._compatibility_tokens(state), schema.get("compatibility_vocab") or []),
        ]
        features = self._np.concatenate([target_fp, start_fp, scalar, *bags]).astype(self._np.float32)
        expected = int(schema.get("feature_dim") or len(features))
        if len(features) != expected:
            raise ValueError(f"learned cascade value feature dimension mismatch: got {len(features)}, expected {expected}")
        return features

    def _fp_many(self, smiles_values: list[str], *, n_bits: int) -> Any:
        arr = self._np.zeros(n_bits, dtype=self._np.float32)
        for smiles in smiles_values:
            mol = self._mol_from_smiles(smiles)
            if mol is None:
                continue
            bv = self._AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            tmp = self._np.zeros(n_bits, dtype=self._np.float32)
            self._DataStructs.ConvertToNumpyArray(bv, tmp)
            arr = self._np.maximum(arr, tmp)
        return arr

    def _mol_from_smiles(self, smiles: str | None) -> Any:
        text = str(smiles or "")
        for chunk in sorted([part.strip() for part in text.replace(";", ".").split(".") if part.strip()], key=len, reverse=True):
            mol = self._Chem.MolFromSmiles(chunk)
            if mol is not None:
                return mol
        return None

    def _bag(self, values: list[str], vocab: list[str], *, denom: float = 1.0) -> Any:
        counts = Counter(_norm(value) for value in values if _norm(value))
        return self._np.asarray([counts.get(token, 0) / max(float(denom), 1.0) for token in vocab], dtype=self._np.float32)

    def _cascade_type(self, state: CascadeProgramState) -> str:
        steps = list(state.step_annotations or [])
        if not steps:
            return ""
        enzymatic = [step.is_enzymatic for step in steps]
        if all(enzymatic):
            return "all_enzymatic"
        if any(enzymatic):
            return "chemoenzymatic"
        return "all_chemical"

    def _transformations(self, state: CascadeProgramState) -> list[str]:
        return [_norm(step.reaction_type) for step in state.step_annotations if _norm(step.reaction_type)]

    def _step_modes(self, state: CascadeProgramState) -> list[str]:
        if state.stage_graph.n_stages > 1:
            return ["sequential_addition"] * max(1, len(state.step_annotations))
        return ["charged_at_t0"] * max(1, len(state.step_annotations))

    def _catalyst_classes(self, state: CascadeProgramState) -> list[str]:
        out = []
        for step in state.step_annotations:
            out.append("enzyme" if step.is_enzymatic else "unknown")
        return out or ["unknown"]

    def _ec1_values(self, state: CascadeProgramState) -> list[str]:
        out = []
        for step in state.step_annotations:
            for ec in step.ec_numbers:
                token = str(ec or "").split(".", 1)[0].strip().lower()
                if token:
                    out.append(token)
        return out

    def _solvents(self, state: CascadeProgramState) -> list[str]:
        out = []
        for step in state.step_annotations:
            if step.condition is not None:
                out.extend(_norm(value) for value in step.condition.solvents if _norm(value))
        return out

    def _compatibility_tokens(self, state: CascadeProgramState) -> list[str]:
        out = [failure.category for failure in state.unresolved_failure_modes or []]
        if state.stage_graph.n_stages > 1:
            out.append("sequential_addition")
        if not state.cofactor_ledger.unclosed_requirements():
            out.append("empirically_compatible")
        return [_norm(value) for value in out if _norm(value)]

    @staticmethod
    def _condition_field_count(env: ConditionEnvelope) -> int:
        return sum(
            int(bool(value))
            for value in (
                env.temperature_c_min,
                env.temperature_c_max,
                env.ph_min,
                env.ph_max,
                env.solvents,
                env.catalysts,
                env.buffer,
                env.cofactors,
            )
        )


class _RuntimeCascadeValueNetwork:
    def __init__(self, nn: Any, in_dim: int, *, hidden: int, n_binary: int):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                h2 = max(48, hidden // 2)
                self.backbone = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.12),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(0.08),
                    nn.Linear(hidden, h2),
                    nn.GELU(),
                )
                self.binary_head = nn.Linear(h2, n_binary)
                self.depth_head = nn.Linear(h2, 1)

            def forward(self, x: Any) -> dict[str, Any]:
                h = self.backbone(x)
                return {"binary_logits": self.binary_head(h), "depth": self.depth_head(h).squeeze(-1)}

        self._model = Net()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def _ranges_overlap(
    left_min: float | None,
    left_max: float | None,
    right_min: float | None,
    right_max: float | None,
) -> bool:
    if None in {left_min, left_max, right_min, right_max}:
        return True
    return max(float(left_min), float(right_min)) <= min(float(left_max), float(right_max))
