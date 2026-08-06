"""Default cascade value, source-policy, and compatibility components.

These components are production search modules with deterministic behavior.
Learned implementations can replace them through the formal interfaces without
changing the cascade state machine.
"""
from __future__ import annotations

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

    is_learned_value_model = True

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
        self.recommended_feasible_threshold = (
            artifact.get("recommended_feasible_threshold")
            or ((self.training_summary.get("feasibility") or {}).get("threshold_calibration") or {}).get("recommended_threshold")
        )

    def predict(self, state: CascadeProgramState) -> CascadeValuePrediction:
        # Reuse the exact feature function used for training. This import is
        # intentionally lazy so sklearn is not required for default search.
        from cascade_planner.cascade_verifier.features import cascade_verifier_features

        base = self.base_model.predict(state)
        route = _route_dict_from_state(state)
        example = {
            "target_smiles": state.target_smiles,
            "cascade": route,
            "expected_failure_reasons": [],
        }
        x = self.vectorizer.transform([cascade_verifier_features(example)])
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
                    "recommended_feasible_threshold": self.recommended_feasible_threshold,
                    "conservative_feasible": (
                        bool(feasible_probability >= float(self.recommended_feasible_threshold))
                        if self.recommended_feasible_threshold not in (None, "")
                        else None
                    ),
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


def _ranges_overlap(
    left_min: float | None,
    left_max: float | None,
    right_min: float | None,
    right_max: float | None,
) -> bool:
    if None in {left_min, left_max, right_min, right_max}:
        return True
    return max(float(left_min), float(right_min)) <= min(float(left_max), float(right_max))
