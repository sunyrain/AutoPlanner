"""Formal extension interfaces for cascade-native search.

The search loop depends on these small contracts instead of depending on a
specific scoring, repair, or proposal implementation. Deterministic defaults
live in sibling modules; learned implementations can satisfy the same
interfaces without changing the search state machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cascade_planner.cascade_search.state import (
    CascadeAction,
    CascadeFailure,
    CascadeModule,
    CascadeProgramState,
    ConditionEnvelope,
    StepAnnotation,
)


@runtime_checkable
class CascadeProposalProvider(Protocol):
    provider_name: str

    def propose(self, request: Any, *args: Any, **kwargs: Any) -> list[CascadeAction | StepAnnotation | dict[str, Any]]:
        """Return candidate cascade actions for the requested open leaf."""


@runtime_checkable
class CascadeFailureDetector(Protocol):
    def detect(self, state: CascadeProgramState, **kwargs: Any) -> list[CascadeFailure]:
        """Return typed failures visible in the current cascade state."""


@runtime_checkable
class CascadeRepairPlanner(Protocol):
    def propose_repairs(
        self,
        state: CascadeProgramState,
        failures: list[CascadeFailure] | None = None,
    ) -> list[CascadeAction]:
        """Return repair actions for typed cascade failures."""


@runtime_checkable
class CascadeValueModelProtocol(Protocol):
    def predict(self, state: CascadeProgramState) -> Any:
        """Return a route-state value prediction with at least a ``value`` field."""


@runtime_checkable
class CascadeSourcePolicyProtocol(Protocol):
    def allocate(
        self,
        state: CascadeProgramState,
        *,
        available_sources: list[str],
        total_budget: int,
    ) -> Any:
        """Allocate proposal budget by provider/source."""


@runtime_checkable
class CascadeEnzymeModuleRankerProtocol(Protocol):
    def score(self, module: CascadeModule, state: CascadeProgramState, *, stage_id: str | None = None) -> float:
        """Score module compatibility with the current cascade state."""


@runtime_checkable
class CascadeConditionTransitionModelProtocol(Protocol):
    def predict(
        self,
        left: ConditionEnvelope | StepAnnotation | None,
        right: ConditionEnvelope | StepAnnotation | None,
    ) -> Any:
        """Predict same-pot/telescoped/isolation probabilities."""


@runtime_checkable
class CascadeCofactorClosureModelProtocol(Protocol):
    def predict(self, state: CascadeProgramState) -> Any:
        """Classify cofactor closure status for the current program."""


@runtime_checkable
class CascadeTransitionValueModelProtocol(Protocol):
    def score_transitions(
        self,
        state: CascadeProgramState,
        actions: list[CascadeAction],
        child_states: list[CascadeProgramState],
        *,
        expanded_leaf: str | None = None,
    ) -> list[float]:
        """Return process-aware scores for state/action/child transitions."""


@runtime_checkable
class CascadeActionValueModelProtocol(Protocol):
    def score_actions(
        self,
        state: CascadeProgramState,
        actions: list[CascadeAction],
        child_states: list[CascadeProgramState] | None = None,
        *,
        expanded_leaf: str | None = None,
    ) -> list[float]:
        """Return Q(S,a)-style scores for actions in the current search state."""


@runtime_checkable
class CascadePairScorerProtocol(Protocol):
    def score_action(
        self,
        state: CascadeProgramState,
        action: CascadeAction,
        child_state: CascadeProgramState | None = None,
        *,
        expanded_leaf: str | None = None,
    ) -> Any:
        """Return adjacent-step cascade compatibility for a candidate action."""


@dataclass
class CascadeSearchController:
    """Bundle swappable cascade models used by the formal search architecture."""

    value_model: CascadeValueModelProtocol
    source_policy: CascadeSourcePolicyProtocol | None = None
    enzyme_module_ranker: CascadeEnzymeModuleRankerProtocol | None = None
    condition_transition_model: CascadeConditionTransitionModelProtocol | None = None
    cofactor_closure_model: CascadeCofactorClosureModelProtocol | None = None
    transition_value_model: CascadeTransitionValueModelProtocol | None = None
    action_value_model: CascadeActionValueModelProtocol | None = None
    pair_scorer: CascadePairScorerProtocol | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_model": type(self.value_model).__name__,
            "source_policy": type(self.source_policy).__name__ if self.source_policy is not None else "",
            "enzyme_module_ranker": (
                type(self.enzyme_module_ranker).__name__ if self.enzyme_module_ranker is not None else ""
            ),
            "condition_transition_model": (
                type(self.condition_transition_model).__name__
                if self.condition_transition_model is not None
                else ""
            ),
            "cofactor_closure_model": (
                type(self.cofactor_closure_model).__name__
                if self.cofactor_closure_model is not None
                else ""
            ),
            "transition_value_model": (
                type(self.transition_value_model).__name__
                if self.transition_value_model is not None
                else ""
            ),
            "action_value_model": (
                type(self.action_value_model).__name__
                if self.action_value_model is not None
                else ""
            ),
            "pair_scorer": (
                type(self.pair_scorer).__name__
                if self.pair_scorer is not None
                else ""
            ),
            "metadata": dict(self.metadata),
        }
