"""Route-level condition state summaries for cascade search.

This module turns per-step condition envelopes into one route-level contract.
The report is deliberately conservative: missing conditions are not treated as
compatible, and wide ranges are warnings rather than validated process recipes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "cascade_condition_state.v1"


@dataclass
class StageConditionSummary:
    stage_id: str
    step_indices: list[int] = field(default_factory=list)
    step_count: int = 0
    conditioned_step_count: int = 0
    missing_condition_step_indices: list[int] = field(default_factory=list)
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    temperature_span_c: float | None = None
    ph_min: float | None = None
    ph_max: float | None = None
    ph_span: float | None = None
    solvent_union: list[str] = field(default_factory=list)
    solvent_intersection: list[str] = field(default_factory=list)
    solvent_evidence_step_count: int = 0
    condition_conflicts: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    risk: str = "ok"
    same_pot_status: str = "compatible"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionStateReport:
    schema_version: str = SCHEMA_VERSION
    contract: str = (
        "Condition envelopes are route-state hypotheses for cascade planning; "
        "they are not validated experimental conditions."
    )
    step_count: int = 0
    stage_count: int = 0
    conditioned_step_count: int = 0
    condition_coverage: float = 0.0
    missing_condition_step_indices: list[int] = field(default_factory=list)
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    temperature_span_c: float | None = None
    ph_min: float | None = None
    ph_max: float | None = None
    ph_span: float | None = None
    solvent_union: list[str] = field(default_factory=list)
    solvent_intersection: list[str] = field(default_factory=list)
    stage_summaries: list[StageConditionSummary] = field(default_factory=list)
    condition_conflicts: list[dict[str, Any]] = field(default_factory=list)
    route_issues: list[str] = field(default_factory=list)
    route_risk: str = "ok"
    stepwise_required: bool = False
    same_pot_status: str = "compatible"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage_summaries"] = [stage.to_dict() for stage in self.stage_summaries]
        return data


def summarize_condition_state(state: Any) -> ConditionStateReport:
    """Build a conservative route-level condition summary from a cascade state."""
    steps = list(getattr(state, "step_annotations", None) or getattr(state, "steps", None) or [])
    stage_rows = _stage_rows(state, steps)
    stage_summaries = [_summarize_stage(stage_id, indices, steps) for stage_id, indices in stage_rows]
    conditions = [getattr(step, "condition", None) for step in steps if getattr(step, "condition", None) is not None]
    missing = [idx for idx, step in enumerate(steps) if getattr(step, "condition", None) is None and not _condition_not_required(step)]
    temp_min, temp_max, temp_span = _range_summary(conditions, "temperature_c_min", "temperature_c_max")
    ph_min, ph_max, ph_span = _range_summary(conditions, "ph_min", "ph_max")
    solvent_sets = [_solvents(condition) for condition in conditions if _solvents(condition)]
    conflicts = []
    for stage in stage_summaries:
        conflicts.extend(stage.condition_conflicts)
    issues = set()
    for stage in stage_summaries:
        issues.update(stage.issues)
    if missing:
        issues.add("missing_condition_envelope")
    if temp_span is not None and temp_span > 100.0:
        issues.add("route_temperature_span_gt_100c")
    elif temp_span is not None and temp_span > 50.0:
        issues.add("wide_route_temperature_span")
    if ph_span is not None and ph_span > 4.0:
        issues.add("route_ph_span_gt_4")
    elif ph_span is not None and ph_span > 2.5:
        issues.add("wide_route_ph_span")
    if len(stage_summaries) > 1:
        issues.add("multi_stage_route")
    route_issues = sorted(issues)
    route_risk = _risk_from_issues(route_issues)
    same_pot_status = _same_pot_status(route_risk, route_issues, missing)
    return ConditionStateReport(
        step_count=len(steps),
        stage_count=len(stage_summaries),
        conditioned_step_count=len(conditions),
        condition_coverage=round(len(conditions) / len(steps), 4) if steps else 1.0,
        missing_condition_step_indices=missing,
        temperature_min_c=temp_min,
        temperature_max_c=temp_max,
        temperature_span_c=temp_span,
        ph_min=ph_min,
        ph_max=ph_max,
        ph_span=ph_span,
        solvent_union=sorted(set().union(*solvent_sets)) if solvent_sets else [],
        solvent_intersection=sorted(set.intersection(*solvent_sets)) if solvent_sets else [],
        stage_summaries=stage_summaries,
        condition_conflicts=conflicts,
        route_issues=route_issues,
        route_risk=route_risk,
        stepwise_required=_stepwise_required(route_issues, route_risk),
        same_pot_status=same_pot_status,
    )


def _stage_rows(state: Any, steps: list[Any]) -> list[tuple[str, list[int]]]:
    stage_graph = getattr(state, "stage_graph", None)
    stages = list(getattr(stage_graph, "stages", None) or [])
    rows = []
    for stage in stages:
        indices = [idx for idx in list(getattr(stage, "step_indices", []) or []) if 0 <= idx < len(steps)]
        if indices:
            rows.append((str(getattr(stage, "stage_id", "") or "stage_1"), indices))
    if rows:
        return rows
    partition = list(getattr(state, "stage_partition", []) or [])
    if partition and len(partition) >= len(steps):
        grouped: dict[str, list[int]] = {}
        for idx, stage_id in enumerate(partition[: len(steps)]):
            grouped.setdefault(str(stage_id or "stage_1"), []).append(idx)
        return list(grouped.items())
    if steps:
        return [(str(getattr(state, "current_stage", "") or "stage_1"), list(range(len(steps))))]
    return []


def _summarize_stage(stage_id: str, indices: list[int], steps: list[Any]) -> StageConditionSummary:
    stage_steps = [(idx, steps[idx]) for idx in indices if 0 <= idx < len(steps)]
    conditions = [(idx, getattr(step, "condition", None)) for idx, step in stage_steps if getattr(step, "condition", None) is not None]
    missing = [idx for idx, step in stage_steps if getattr(step, "condition", None) is None and not _condition_not_required(step)]
    envelopes = [condition for _idx, condition in conditions]
    temp_min, temp_max, temp_span = _range_summary(envelopes, "temperature_c_min", "temperature_c_max")
    ph_min, ph_max, ph_span = _range_summary(envelopes, "ph_min", "ph_max")
    solvent_sets = [_solvents(condition) for _idx, condition in conditions if _solvents(condition)]
    conflicts = _stage_condition_conflicts(conditions)
    issues = set()
    if missing:
        issues.add("missing_condition_envelope")
    if conflicts:
        issues.add("stage_condition_conflict")
    if temp_span is not None and temp_span > 100.0:
        issues.add("stage_temperature_span_gt_100c")
    elif temp_span is not None and temp_span > 50.0:
        issues.add("wide_stage_temperature_span")
    if ph_span is not None and ph_span > 4.0:
        issues.add("stage_ph_span_gt_4")
    elif ph_span is not None and ph_span > 2.5:
        issues.add("wide_stage_ph_span")
    if len(solvent_sets) >= 2 and not set.intersection(*solvent_sets):
        issues.add("stage_solvent_mismatch")
    sorted_issues = sorted(issues)
    risk = _risk_from_issues(sorted_issues)
    return StageConditionSummary(
        stage_id=stage_id,
        step_indices=list(indices),
        step_count=len(stage_steps),
        conditioned_step_count=len(conditions),
        missing_condition_step_indices=missing,
        temperature_min_c=temp_min,
        temperature_max_c=temp_max,
        temperature_span_c=temp_span,
        ph_min=ph_min,
        ph_max=ph_max,
        ph_span=ph_span,
        solvent_union=sorted(set().union(*solvent_sets)) if solvent_sets else [],
        solvent_intersection=sorted(set.intersection(*solvent_sets)) if solvent_sets else [],
        solvent_evidence_step_count=len(solvent_sets),
        condition_conflicts=conflicts,
        issues=sorted_issues,
        risk=risk,
        same_pot_status=_same_pot_status(risk, sorted_issues, missing),
    )


def _stage_condition_conflicts(conditions: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for left_pos, (left_idx, left) in enumerate(conditions):
        for right_idx, right in conditions[left_pos + 1 :]:
            if not left.overlaps(right):
                conflicts.append(
                    {
                        "left_step_index": left_idx,
                        "right_step_index": right_idx,
                        "reason": "condition_envelopes_do_not_overlap",
                    }
                )
    return conflicts


def _range_summary(conditions: list[Any], min_attr: str, max_attr: str) -> tuple[float | None, float | None, float | None]:
    lows: list[float] = []
    highs: list[float] = []
    for condition in conditions:
        lo = _safe_float(getattr(condition, min_attr, None))
        hi = _safe_float(getattr(condition, max_attr, None))
        if lo is None and hi is None:
            continue
        if lo is None:
            lo = hi
        if hi is None:
            hi = lo
        if lo is None or hi is None:
            continue
        lows.append(float(lo))
        highs.append(float(hi))
    if not lows or not highs:
        return None, None, None
    low = min(lows)
    high = max(highs)
    return round(low, 3), round(high, 3), round(high - low, 3)


def _solvents(condition: Any) -> set[str]:
    if condition is None:
        return set()
    if hasattr(condition, "normalized_solvents"):
        return {text for text in condition.normalized_solvents() if text}
    values = getattr(condition, "solvents", []) or []
    return {_normalize_token(value) for value in values if _normalize_token(value)}


def _condition_not_required(step: Any) -> bool:
    raw = getattr(step, "raw_metadata", {}) or {}
    return bool(isinstance(raw, dict) and raw.get("condition_not_required"))


def _risk_from_issues(issues: list[str]) -> str:
    high_markers = {
        "stage_condition_conflict",
        "stage_temperature_span_gt_100c",
        "route_temperature_span_gt_100c",
        "stage_ph_span_gt_4",
        "route_ph_span_gt_4",
        "stage_solvent_mismatch",
    }
    if any(issue in high_markers for issue in issues):
        return "high"
    if issues:
        return "warn"
    return "ok"


def _same_pot_status(risk: str, issues: list[str], missing: list[int]) -> str:
    if risk == "high":
        return "incompatible"
    if missing or issues:
        return "unknown"
    return "compatible"


def _stepwise_required(issues: list[str], risk: str) -> bool:
    if risk == "high":
        return True
    return any(
        issue
        in {
            "multi_stage_route",
            "wide_route_temperature_span",
            "wide_route_ph_span",
            "route_temperature_span_gt_100c",
            "route_ph_span_gt_4",
        }
        for issue in issues
    )


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
