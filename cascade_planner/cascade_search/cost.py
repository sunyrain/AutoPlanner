"""Unified cost features for cascade-aware search states."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cascade_planner.cascade_search.state import CascadeSearchState, ConditionEnvelope


@dataclass
class CascadeCostWeights:
    retrosynthesis: float = 1.0
    stock_reachability: float = 1.0
    reaction_plausibility: float = 1.0
    enzyme_evidence: float = 0.8
    condition_compatibility: float = 0.8
    cofactor_closure: float = 0.7
    stage_complexity: float = 0.4
    uncertainty: float = 0.5


@dataclass
class CascadeCostBreakdown:
    total_cost: float
    retrosynthesis_score: float
    stock_reachability: float
    reaction_plausibility: float
    enzyme_evidence: float
    condition_compatibility: float
    cofactor_closure: float
    stage_complexity_penalty: float
    uncertainty_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_cascade_state(
    state: CascadeSearchState,
    weights: CascadeCostWeights | None = None,
) -> CascadeCostBreakdown:
    """Return a lower-is-better cost from normalized cascade features."""
    weights = weights or CascadeCostWeights()
    retro = _mean_score([step.score for step in state.steps], default=0.5)
    stock = _stock_reachability(state)
    plausibility = retro
    enzyme = _enzyme_evidence(state)
    condition = _condition_compatibility(state)
    cofactor = _cofactor_closure(state)
    stage_penalty = _stage_complexity_penalty(state)
    uncertainty = 1.0 - _mean_score(
        [step.evidence_confidence for step in state.steps] + [state.evidence_confidence],
        default=0.5,
    )
    total = (
        weights.retrosynthesis * (1.0 - retro)
        + weights.stock_reachability * (1.0 - stock)
        + weights.reaction_plausibility * (1.0 - plausibility)
        + weights.enzyme_evidence * (1.0 - enzyme)
        + weights.condition_compatibility * (1.0 - condition)
        + weights.cofactor_closure * (1.0 - cofactor)
        + weights.stage_complexity * stage_penalty
        + weights.uncertainty * uncertainty
    )
    return CascadeCostBreakdown(
        total_cost=round(float(total), 6),
        retrosynthesis_score=retro,
        stock_reachability=stock,
        reaction_plausibility=plausibility,
        enzyme_evidence=enzyme,
        condition_compatibility=condition,
        cofactor_closure=cofactor,
        stage_complexity_penalty=stage_penalty,
        uncertainty_penalty=uncertainty,
    )


def _mean_score(values: list[float | None], *, default: float) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return default
    return _clamp(sum(clean) / len(clean))


def _stock_reachability(state: CascadeSearchState) -> float:
    if not state.open_leaves:
        return 1.0
    status = {}
    for step in state.steps:
        status.update(step.stock_status)
    known = [status.get(leaf) for leaf in state.open_leaves]
    if not known:
        return 0.0
    return sum(1.0 for value in known if value is True) / len(known)


def _enzyme_evidence(state: CascadeSearchState) -> float:
    enzymatic_like = [
        step
        for step in state.steps
        if step.has_enzyme_evidence or "enzyme" in (step.reaction_type or "").lower()
    ]
    if not enzymatic_like:
        return 1.0
    return sum(1.0 for step in enzymatic_like if step.has_enzyme_evidence) / len(enzymatic_like)


def _condition_compatibility(state: CascadeSearchState) -> float:
    steps = list(state.step_annotations or state.steps)
    if not steps:
        return 1.0
    missing = sum(1 for step in steps if step.condition is None)
    observed = [step for step in steps if step.condition is not None]
    if len(observed) < 2:
        return _clamp(1.0 - 0.35 * missing / max(len(steps), 1))
    compatible = 0
    total = 0
    compared = False
    stage_graph = getattr(state, "stage_graph", None)
    stage_iter = stage_graph.stages if stage_graph is not None else []
    for stage in stage_iter:
        stage_steps = [steps[idx] for idx in stage.step_indices if 0 <= idx < len(steps)]
        for left_step, right_step in zip(stage_steps, stage_steps[1:]):
            if left_step.condition is None or right_step.condition is None:
                continue
            compared = True
            total += 1
            compatible += int(_conditions_compatible(left_step.condition, right_step.condition))
    if not compared:
        conditions = [step.condition for step in steps if step.condition is not None]
        for left, right in zip(conditions, conditions[1:]):
            total += 1
            compatible += int(_conditions_compatible(left, right))
    compatibility = compatible / total if total else 1.0
    coverage = len(observed) / max(len(steps), 1)
    return _clamp(compatibility * coverage + 0.35 * (1.0 - coverage))


def _conditions_compatible(left: ConditionEnvelope, right: ConditionEnvelope) -> bool:
    if not _ranges_overlap(left.temperature_c_min, left.temperature_c_max, right.temperature_c_min, right.temperature_c_max):
        return False
    if not _ranges_overlap(left.ph_min, left.ph_max, right.ph_min, right.ph_max):
        return False
    if left.solvents and right.solvents and not (set(left.solvents) & set(right.solvents)):
        return False
    return True


def _ranges_overlap(
    left_min: float | None,
    left_max: float | None,
    right_min: float | None,
    right_max: float | None,
) -> bool:
    if None in {left_min, left_max, right_min, right_max}:
        return True
    return max(float(left_min), float(right_min)) <= min(float(left_max), float(right_max))


def _cofactor_closure(state: CascadeSearchState) -> float:
    required_total = sum(float(value or 0.0) for value in state.cofactor_ledger.required.values())
    if required_total <= 0.0:
        return 1.0
    unclosed_total = sum(state.cofactor_ledger.unclosed_requirements().values())
    return _clamp(1.0 - (unclosed_total / required_total))


def _stage_complexity_penalty(state: CascadeSearchState) -> float:
    if getattr(state, "stage_graph", None) is not None:
        n_stages = int(state.stage_graph.n_stages or 0)
        if n_stages <= 1:
            return 0.0
        return _clamp((n_stages - 1) / max(len(state.step_annotations or state.steps), 1))
    stages = [stage for stage in state.stage_partition if stage]
    if len(stages) <= 1:
        return 0.0
    return _clamp((len(set(stages)) - 1) / max(len(stages), 1))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
