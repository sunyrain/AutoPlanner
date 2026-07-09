"""Typed cascade failure detection.

Failures in this package are intended to steer search repairs. They are not
final report labels only.
"""
from __future__ import annotations

from typing import Callable

from cascade_planner.cascade_search.state import (
    CascadeActionType,
    CascadeFailure,
    CascadeFailureKind,
    CascadeProgramState,
)


StockChecker = Callable[[str], bool]


def detect_cascade_failures(
    state: CascadeProgramState,
    *,
    stock_checker: StockChecker | None = None,
    max_stages: int = 4,
    min_evidence_confidence: float = 0.4,
    min_step_score: float = 0.05,
) -> list[CascadeFailure]:
    """Return typed failures visible in a partial cascade program."""
    failures: list[CascadeFailure] = []
    failures.extend(_stock_failures(state, stock_checker=stock_checker))
    failures.extend(_condition_failures(state))
    failures.extend(_cofactor_failures(state))
    failures.extend(_enzyme_evidence_failures(state, min_evidence_confidence=min_evidence_confidence))
    failures.extend(_stage_complexity_failures(state, max_stages=max_stages))
    failures.extend(_plausibility_failures(state, min_step_score=min_step_score))
    return failures


def annotate_state_failures(
    state: CascadeProgramState,
    *,
    stock_checker: StockChecker | None = None,
    max_stages: int = 4,
    min_evidence_confidence: float = 0.4,
    min_step_score: float = 0.05,
) -> CascadeProgramState:
    state.unresolved_failure_modes = detect_cascade_failures(
        state,
        stock_checker=stock_checker,
        max_stages=max_stages,
        min_evidence_confidence=min_evidence_confidence,
        min_step_score=min_step_score,
    )
    return state


def _stock_failures(state: CascadeProgramState, *, stock_checker: StockChecker | None) -> list[CascadeFailure]:
    failures: list[CascadeFailure] = []
    for leaf in state.open_molecule_leaves or state.open_leaves:
        known = state.stock_status.get(leaf)
        if known is True:
            continue
        if stock_checker is not None:
            try:
                if bool(stock_checker(leaf)):
                    continue
            except Exception:
                pass
        if known is False:
            failures.append(
                CascadeFailure(
                    CascadeFailureKind.STOCK_DEAD_END,
                    message="Open molecule is marked out of stock.",
                    target_leaf=leaf,
                    repair_options=[CascadeActionType.RETROSYNTHETIC_STEP],
                )
            )
    return failures


def _condition_failures(state: CascadeProgramState) -> list[CascadeFailure]:
    failures: list[CascadeFailure] = []
    for conflict in state.condition_conflicts():
        failures.append(
            CascadeFailure(
                CascadeFailureKind.CONDITION_CONFLICT,
                message="Adjacent steps in the same stage have incompatible condition envelopes.",
                step_index=conflict.get("right_step_index"),
                repair_options=[CascadeActionType.STAGE_TRANSITION],
                metadata=conflict,
            )
        )
    for idx, step in enumerate(state.step_annotations or state.steps):
        if step.condition is not None:
            continue
        if step.raw_metadata.get("condition_not_required"):
            continue
        failures.append(
            CascadeFailure(
                CascadeFailureKind.CONDITION_MISSING,
                message="Step lacks a condition envelope; compatibility is unknown and should be filled by condition prediction.",
                step_index=idx,
                severity=0.35,
                repair_options=[CascadeActionType.STAGE_TRANSITION],
                metadata={
                    "reason": "missing_condition_envelope",
                    "rxn_smiles": step.rxn_smiles,
                    "source_model": step.source_model,
                    "stage_id": step.stage_id,
                },
            )
        )
    return failures


def _cofactor_failures(state: CascadeProgramState) -> list[CascadeFailure]:
    unclosed = state.cofactor_ledger.unclosed_requirements()
    if not unclosed:
        return []
    return [
        CascadeFailure(
            CascadeFailureKind.COFACTOR_DEBT,
            message="Cofactor requirements are not closed by regeneration modules.",
            severity=min(1.0, sum(float(value) for value in unclosed.values())),
            repair_options=[CascadeActionType.COFACTOR_REPAIR],
            metadata={"unclosed_requirements": unclosed},
        )
    ]


def _enzyme_evidence_failures(
    state: CascadeProgramState,
    *,
    min_evidence_confidence: float,
) -> list[CascadeFailure]:
    failures: list[CascadeFailure] = []
    for idx, step in enumerate(state.step_annotations or state.steps):
        if not step.is_enzymatic:
            continue
        if step.raw_metadata.get("evidence_retrieval"):
            continue
        weak_identifier = not step.has_enzyme_evidence
        weak_confidence = (
            step.evidence_confidence is not None
            and float(step.evidence_confidence) < float(min_evidence_confidence)
        )
        if weak_identifier or weak_confidence:
            failures.append(
                CascadeFailure(
                    CascadeFailureKind.ENZYME_EVIDENCE_WEAK,
                    message="Enzymatic candidate lacks sufficient EC, enzyme, or confidence evidence.",
                    step_index=idx,
                    repair_options=[CascadeActionType.EVIDENCE_RETRIEVAL],
                    metadata={
                        "rxn_smiles": step.rxn_smiles,
                        "source_model": step.source_model,
                        "evidence_confidence": step.evidence_confidence,
                    },
                )
            )
    return failures


def _stage_complexity_failures(state: CascadeProgramState, *, max_stages: int) -> list[CascadeFailure]:
    if state.stage_graph.n_stages <= max_stages:
        return []
    return [
        CascadeFailure(
            CascadeFailureKind.STAGE_OVER_COMPLEX,
            message="Stage graph exceeds the configured complexity limit.",
            severity=min(1.0, (state.stage_graph.n_stages - max_stages) / max(max_stages, 1)),
            repair_options=[CascadeActionType.STAGE_TRANSITION],
            metadata={"n_stages": state.stage_graph.n_stages, "max_stages": max_stages},
        )
    ]


def _plausibility_failures(state: CascadeProgramState, *, min_step_score: float) -> list[CascadeFailure]:
    failures: list[CascadeFailure] = []
    for idx, step in enumerate(state.step_annotations or state.steps):
        if step.score is not None and float(step.score) < float(min_step_score):
            failures.append(
                CascadeFailure(
                    CascadeFailureKind.LOW_PLAUSIBILITY,
                    message="Step score is below the cascade search plausibility floor.",
                    step_index=idx,
                    severity=1.0 - max(0.0, float(step.score)),
                    repair_options=[CascadeActionType.RETROSYNTHETIC_STEP],
                    metadata={"score": step.score, "rxn_smiles": step.rxn_smiles},
                )
            )
    return failures
