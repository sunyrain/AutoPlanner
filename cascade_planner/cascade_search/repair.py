"""Repair policy for typed cascade search failures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cascade_planner.cascade_search.state import (
    CascadeAction,
    CascadeActionType,
    CascadeFailure,
    CascadeFailureKind,
    CascadeModule,
    CascadeProgramState,
    StageTransition,
)


@dataclass(frozen=True)
class CascadeRepairRule:
    failure_kind: CascadeFailureKind | str
    preferred_actions: tuple[CascadeActionType | str, ...]
    penalty: float = 0.0
    max_applications: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_kind"] = _kind_value(self.failure_kind)
        data["preferred_actions"] = [_action_value(action) for action in self.preferred_actions]
        return data


@dataclass
class CascadeRepairPolicy:
    """Map typed failures to concrete search repair actions."""

    rules: dict[str, CascadeRepairRule] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "CascadeRepairPolicy":
        return cls(
            rules={
                CascadeFailureKind.CANDIDATE_MISSING.value: CascadeRepairRule(
                    CascadeFailureKind.CANDIDATE_MISSING,
                    (CascadeActionType.RETROSYNTHETIC_STEP,),
                    penalty=0.2,
                    notes="Expand more sources, relax source gate, or query fallback providers.",
                ),
                CascadeFailureKind.STOCK_DEAD_END.value: CascadeRepairRule(
                    CascadeFailureKind.STOCK_DEAD_END,
                    (CascadeActionType.RETROSYNTHETIC_STEP,),
                    penalty=0.4,
                    notes="Try alternative stock closure or one additional retrosynthetic depth.",
                ),
                CascadeFailureKind.CONDITION_CONFLICT.value: CascadeRepairRule(
                    CascadeFailureKind.CONDITION_CONFLICT,
                    (CascadeActionType.STAGE_TRANSITION,),
                    penalty=0.35,
                    notes="Split one-pot into telescoped or isolated stages.",
                ),
                CascadeFailureKind.CONDITION_MISSING.value: CascadeRepairRule(
                    CascadeFailureKind.CONDITION_MISSING,
                    (),
                    penalty=0.10,
                    notes="Condition prediction or literature lookup should fill this; it is not a proven incompatibility.",
                ),
                CascadeFailureKind.COFACTOR_DEBT.value: CascadeRepairRule(
                    CascadeFailureKind.COFACTOR_DEBT,
                    (CascadeActionType.COFACTOR_REPAIR,),
                    penalty=0.25,
                    notes="Insert cofactor regeneration module.",
                ),
                CascadeFailureKind.ENZYME_EVIDENCE_WEAK.value: CascadeRepairRule(
                    CascadeFailureKind.ENZYME_EVIDENCE_WEAK,
                    (CascadeActionType.EVIDENCE_RETRIEVAL,),
                    penalty=0.3,
                    notes="Retrieve EC, UniProt, Rhea, BRENDA, or cascade precedent evidence.",
                ),
                CascadeFailureKind.STAGE_OVER_COMPLEX.value: CascadeRepairRule(
                    CascadeFailureKind.STAGE_OVER_COMPLEX,
                    (CascadeActionType.STAGE_TRANSITION,),
                    penalty=0.5,
                    notes="Merge compatible stages or penalize excessive isolation/workup.",
                ),
                CascadeFailureKind.ROUTE_ORDER_MISMATCH.value: CascadeRepairRule(
                    CascadeFailureKind.ROUTE_ORDER_MISMATCH,
                    (CascadeActionType.STAGE_TRANSITION, CascadeActionType.RETROSYNTHETIC_STEP),
                    penalty=0.45,
                    notes="Reorder route or require a stage boundary.",
                ),
                CascadeFailureKind.LOW_PLAUSIBILITY.value: CascadeRepairRule(
                    CascadeFailureKind.LOW_PLAUSIBILITY,
                    (CascadeActionType.RETROSYNTHETIC_STEP,),
                    penalty=0.25,
                    notes="Prefer an alternative candidate from another source.",
                ),
            }
        )

    def rule_for(self, failure: CascadeFailure | CascadeFailureKind | str) -> CascadeRepairRule | None:
        key = _failure_key(failure)
        return self.rules.get(key)

    def actions_for(self, failure: CascadeFailure | CascadeFailureKind | str) -> tuple[CascadeActionType | str, ...]:
        rule = self.rule_for(failure)
        return rule.preferred_actions if rule is not None else ()

    def propose_repairs(
        self,
        state: CascadeProgramState,
        failures: list[CascadeFailure] | None = None,
    ) -> list[CascadeAction]:
        actions: list[CascadeAction] = []
        for failure in failures if failures is not None else state.unresolved_failure_modes:
            kind = _failure_key(failure)
            if kind == CascadeFailureKind.CONDITION_CONFLICT.value:
                actions.append(_stage_transition_repair(state, failure, penalty=self._penalty(failure)))
            elif kind == CascadeFailureKind.COFACTOR_DEBT.value:
                actions.extend(_cofactor_repairs(failure, penalty=self._penalty(failure)))
            elif kind == CascadeFailureKind.ENZYME_EVIDENCE_WEAK.value:
                actions.append(_evidence_retrieval_repair(failure, penalty=self._penalty(failure)))
        return actions

    def _penalty(self, failure: CascadeFailure) -> float:
        rule = self.rule_for(failure)
        return float(rule.penalty) if rule is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {key: rule.to_dict() for key, rule in self.rules.items()}


def _stage_transition_repair(
    state: CascadeProgramState,
    failure: CascadeFailure,
    *,
    penalty: float,
) -> CascadeAction:
    next_id = f"stage_{state.stage_graph.n_stages + 1}"
    from_stage = str(failure.metadata.get("stage_id") or state.current_stage or "stage_1")
    transition = StageTransition(
        from_stage_id=from_stage,
        to_stage_id=next_id,
        operation_type="telescoped",
        required_operations=["buffer_exchange"],
        cost=penalty,
        reason=failure.message or "condition conflict",
        metadata={"failure": failure.to_dict()},
    )
    return CascadeAction(
        CascadeActionType.STAGE_TRANSITION,
        stage_transition=transition,
        source="repair_policy",
        cost_delta=penalty,
        metadata={"repair_for": failure.category},
    )


def _cofactor_repairs(failure: CascadeFailure, *, penalty: float) -> list[CascadeAction]:
    unclosed = failure.metadata.get("unclosed_requirements") or {}
    actions: list[CascadeAction] = []
    for name, amount in unclosed.items():
        cofactor = str(name)
        module = CascadeModule(
            name=f"{cofactor} regeneration",
            module_kind="cofactor_regeneration",
            cofactor_regenerations={cofactor: float(amount or 0.0)},
            evidence_confidence=0.5,
            source_model="default_repair_policy",
            raw_metadata={"repair_for": failure.to_dict()},
        )
        actions.append(
            CascadeAction(
                CascadeActionType.COFACTOR_REPAIR,
                module=module,
                source="repair_policy",
                cost_delta=penalty,
                metadata={"repair_for": failure.category, "cofactor": cofactor},
            )
        )
    return actions


def _evidence_retrieval_repair(failure: CascadeFailure, *, penalty: float) -> CascadeAction:
    return CascadeAction(
        CascadeActionType.EVIDENCE_RETRIEVAL,
        evidence_payload={
            "queries": ["EC", "UniProt", "Rhea", "BRENDA", "cascade_precedent"],
            "failure": failure.to_dict(),
        },
        source="repair_policy",
        cost_delta=penalty,
        metadata={"repair_for": failure.category},
    )


def _failure_key(failure: CascadeFailure | CascadeFailureKind | str) -> str:
    if isinstance(failure, CascadeFailure):
        return failure.category
    if isinstance(failure, CascadeFailureKind):
        return failure.value
    text = str(failure or "")
    for item in CascadeFailureKind:
        if text in {item.value, item.name}:
            return item.value
    return text


def _kind_value(value: CascadeFailureKind | str) -> str:
    return value.value if isinstance(value, CascadeFailureKind) else str(value)


def _action_value(value: CascadeActionType | str) -> str:
    return value.value if isinstance(value, CascadeActionType) else str(value)
