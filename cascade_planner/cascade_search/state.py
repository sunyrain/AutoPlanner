"""Cascade-native data contracts for process-aware retrosynthesis search.

The route-tree planner keeps molecule-only search as an ablation path. This
module defines the richer object searched by AutoPlanner-Cascade: a molecular
route plus stage, condition, enzyme/cofactor, failure, and evidence state.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from cascade_planner.cascade_search.condition_state import summarize_condition_state


class CascadeActionType(str, Enum):
    RETROSYNTHETIC_STEP = "retrosynthetic_step"
    ENZYME_MODULE = "enzyme_module"
    COFACTOR_REPAIR = "cofactor_repair"
    STAGE_TRANSITION = "stage_transition"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"


class CascadeFailureKind(str, Enum):
    CANDIDATE_MISSING = "CandidateMissing"
    STOCK_DEAD_END = "StockDeadEnd"
    CONDITION_CONFLICT = "ConditionConflict"
    CONDITION_MISSING = "ConditionMissing"
    COFACTOR_DEBT = "CofactorDebt"
    ENZYME_EVIDENCE_WEAK = "EnzymeEvidenceWeak"
    STAGE_OVER_COMPLEX = "StageOverComplex"
    ROUTE_ORDER_MISMATCH = "RouteOrderMismatch"
    LOW_PLAUSIBILITY = "LowPlausibility"


@dataclass
class ConditionEnvelope:
    """Comparable condition range for a reaction or process stage."""

    temperature_c_min: float | None = None
    temperature_c_max: float | None = None
    ph_min: float | None = None
    ph_max: float | None = None
    solvents: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    solvent_class: str = ""
    organic_cosolvent_fraction: float | None = None
    buffer: str = ""
    salts: list[str] = field(default_factory=list)
    metals: list[str] = field(default_factory=list)
    oxygen_requirement: str = ""
    oxidants: list[str] = field(default_factory=list)
    reductants: list[str] = field(default_factory=list)
    cofactors: list[str] = field(default_factory=list)
    water_activity: float | None = None
    confidence: float | None = None
    raw_evidence: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_point(
        cls,
        *,
        temperature_c: float | None = None,
        ph: float | None = None,
        solvent: str = "",
        catalyst: str = "",
        confidence: float | None = None,
    ) -> "ConditionEnvelope":
        solvents = [solvent] if solvent else []
        catalysts = [catalyst] if catalyst else []
        return cls(
            temperature_c_min=temperature_c,
            temperature_c_max=temperature_c,
            ph_min=ph,
            ph_max=ph,
            solvents=solvents,
            catalysts=catalysts,
            confidence=confidence,
        )

    @classmethod
    def from_backend_prediction(cls, payload: dict[str, Any] | None) -> "ConditionEnvelope | None":
        if not payload:
            return None
        temp = _normalize_predicted_temperature_c(payload)
        ph = _first_present(payload, "ph", "pH", "PH")
        solvent = str(_first_present(payload, "solvent", "Solvent") or "")
        catalyst = str(_first_present(payload, "catalyst", "Catalyst") or "")
        confidence = _safe_float(_first_present(payload, "confidence", "Confidence", "score", "Score", "scores"))
        if temp is None and ph is None and not solvent and not catalyst:
            return None
        envelope = cls.from_point(
            temperature_c=temp,
            ph=_safe_float(ph),
            solvent=solvent,
            catalyst=catalyst,
            confidence=confidence,
        )
        envelope.raw_evidence = [dict(payload)]
        return envelope

    def normalized_solvents(self) -> set[str]:
        return {_normalize_token(value) for value in self.solvents if _normalize_token(value)}

    def normalized_catalysts(self) -> set[str]:
        values = [*self.catalysts, *self.metals]
        return {_normalize_token(value) for value in values if _normalize_token(value)}

    def overlaps(self, other: "ConditionEnvelope") -> bool:
        if not ranges_overlap(
            self.temperature_c_min,
            self.temperature_c_max,
            other.temperature_c_min,
            other.temperature_c_max,
        ):
            return False
        if not ranges_overlap(self.ph_min, self.ph_max, other.ph_min, other.ph_max):
            return False
        left_solvents = self.normalized_solvents()
        right_solvents = other.normalized_solvents()
        if left_solvents and right_solvents and not (left_solvents & right_solvents):
            return False
        return True

    def merged_with(self, other: "ConditionEnvelope") -> "ConditionEnvelope":
        """Return the shared envelope where ranges overlap, otherwise a union."""
        if other is None:
            return copy.deepcopy(self)
        if self.overlaps(other):
            t_min, t_max = intersect_or_union(
                self.temperature_c_min,
                self.temperature_c_max,
                other.temperature_c_min,
                other.temperature_c_max,
            )
            ph_min, ph_max = intersect_or_union(self.ph_min, self.ph_max, other.ph_min, other.ph_max)
        else:
            t_min, t_max = union_range(
                self.temperature_c_min,
                self.temperature_c_max,
                other.temperature_c_min,
                other.temperature_c_max,
            )
            ph_min, ph_max = union_range(self.ph_min, self.ph_max, other.ph_min, other.ph_max)
        solvents = sorted((self.normalized_solvents() | other.normalized_solvents()) or set(self.solvents + other.solvents))
        catalysts = sorted((self.normalized_catalysts() | other.normalized_catalysts()) or set(self.catalysts + other.catalysts))
        confidence = _mean_optional([self.confidence, other.confidence])
        return ConditionEnvelope(
            temperature_c_min=t_min,
            temperature_c_max=t_max,
            ph_min=ph_min,
            ph_max=ph_max,
            solvents=solvents,
            catalysts=catalysts,
            solvent_class=self.solvent_class or other.solvent_class,
            organic_cosolvent_fraction=_max_optional(self.organic_cosolvent_fraction, other.organic_cosolvent_fraction),
            buffer=self.buffer or other.buffer,
            salts=_dedupe_strings([*self.salts, *other.salts]),
            metals=_dedupe_strings([*self.metals, *other.metals]),
            oxygen_requirement=self.oxygen_requirement or other.oxygen_requirement,
            oxidants=_dedupe_strings([*self.oxidants, *other.oxidants]),
            reductants=_dedupe_strings([*self.reductants, *other.reductants]),
            cofactors=_dedupe_strings([*self.cofactors, *other.cofactors]),
            water_activity=_mean_optional([self.water_activity, other.water_activity]),
            confidence=confidence,
            raw_evidence=[*self.raw_evidence, *other.raw_evidence],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CascadeModule:
    """Process module carried by search actions, especially enzymes and repairs."""

    name: str
    module_kind: str = "enzyme"
    reaction_type: str = ""
    ec_numbers: list[str] = field(default_factory=list)
    enzyme_family: str = ""
    condition_envelope: ConditionEnvelope | None = None
    cofactor_requirements: dict[str, float] = field(default_factory=dict)
    cofactor_regenerations: dict[str, float] = field(default_factory=dict)
    compatible_solvents: list[str] = field(default_factory=list)
    organism: str = ""
    enzyme_ids: list[str] = field(default_factory=list)
    evidence_confidence: float | None = None
    source_model: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_enzyme(self) -> bool:
        return self.module_kind.lower() in {"enzyme", "enzymatic", "biocatalyst"}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.condition_envelope is not None:
            data["condition_envelope"] = self.condition_envelope.to_dict()
        return data


@dataclass
class StepAnnotation:
    """One retrosynthetic step plus process/evidence annotations."""

    product_smiles: str
    reactant_smiles: list[str]
    rxn_smiles: str
    source_model: str = ""
    score: float | None = None
    reaction_type: str = ""
    ec_numbers: list[str] = field(default_factory=list)
    uniprot_ids: list[str] = field(default_factory=list)
    condition: ConditionEnvelope | None = None
    enzyme_module: CascadeModule | None = None
    stage_id: str = "stage_1"
    cofactor_requirements: dict[str, float] = field(default_factory=dict)
    cofactor_regenerations: dict[str, float] = field(default_factory=dict)
    redox_change: str = ""
    evidence_confidence: float | None = None
    stock_status: dict[str, bool | None] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_enzyme_evidence(self) -> bool:
        if self.ec_numbers or self.uniprot_ids:
            return True
        return bool(self.enzyme_module and (self.enzyme_module.ec_numbers or self.enzyme_module.enzyme_ids))

    @property
    def is_enzymatic(self) -> bool:
        if self.has_enzyme_evidence:
            return True
        text = " ".join([self.reaction_type, self.source_model]).lower()
        return any(token in text for token in ("enzyme", "enzymatic", "bio", "ec "))

    def all_cofactor_requirements(self) -> dict[str, float]:
        values = dict(self.cofactor_requirements)
        if self.enzyme_module is not None:
            _add_amounts(values, self.enzyme_module.cofactor_requirements)
        for cofactor in self.condition.cofactors if self.condition else []:
            values.setdefault(str(cofactor), 1.0)
        return values

    def all_cofactor_regenerations(self) -> dict[str, float]:
        values = dict(self.cofactor_regenerations)
        if self.enzyme_module is not None:
            _add_amounts(values, self.enzyme_module.cofactor_regenerations)
        return values

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_enzyme_evidence"] = self.has_enzyme_evidence
        data["is_enzymatic"] = self.is_enzymatic
        if self.condition is not None:
            data["condition"] = self.condition.to_dict()
        if self.enzyme_module is not None:
            data["enzyme_module"] = self.enzyme_module.to_dict()
        return data


@dataclass
class CofactorLedger:
    """Required/regenerated cofactor accounting for a route state."""

    required: dict[str, float] = field(default_factory=dict)
    regenerated: dict[str, float] = field(default_factory=dict)
    consumed: dict[str, float] = field(default_factory=dict)
    produced: dict[str, float] = field(default_factory=dict)

    def record_step(self, step: StepAnnotation) -> None:
        _add_amounts(self.required, step.all_cofactor_requirements())
        _add_amounts(self.regenerated, step.all_cofactor_regenerations())

    def record_module(self, module: CascadeModule) -> None:
        _add_amounts(self.required, module.cofactor_requirements)
        _add_amounts(self.regenerated, module.cofactor_regenerations)

    def unclosed_requirements(self) -> dict[str, float]:
        out = {}
        for name, amount in self.required.items():
            gap = float(amount or 0.0) - float(self.regenerated.get(name) or 0.0)
            if gap > 1e-9:
                out[name] = gap
        return out

    def closure_status(self) -> str:
        if not self.required:
            return "closed"
        return "closed" if not self.unclosed_requirements() else "unclosed"

    def copy(self) -> "CofactorLedger":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["unclosed_requirements"] = self.unclosed_requirements()
        data["closure_status"] = self.closure_status()
        return data


@dataclass
class RedoxLedger:
    """Minimal redox bookkeeping for cascade compatibility checks."""

    oxidants: dict[str, float] = field(default_factory=dict)
    reductants: dict[str, float] = field(default_factory=dict)
    electron_acceptors: dict[str, float] = field(default_factory=dict)
    electron_donors: dict[str, float] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    def record_condition(self, condition: ConditionEnvelope | None) -> None:
        if condition is None:
            return
        for item in condition.oxidants:
            self.oxidants[_normalize_token(item)] = self.oxidants.get(_normalize_token(item), 0.0) + 1.0
        for item in condition.reductants:
            self.reductants[_normalize_token(item)] = self.reductants.get(_normalize_token(item), 0.0) + 1.0
        if self.oxidants and self.reductants and "oxidant_reductant_same_stage" not in self.conflicts:
            self.conflicts.append("oxidant_reductant_same_stage")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage:
    stage_id: str
    step_indices: list[int] = field(default_factory=list)
    operation_type: str = "one_pot"
    condition_envelope: ConditionEnvelope | None = None
    required_operations: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    enzyme_modules: list[CascadeModule] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_step(self, step_index: int, step: StepAnnotation) -> None:
        if step_index not in self.step_indices:
            self.step_indices.append(step_index)
        if step.condition is not None:
            self.condition_envelope = (
                step.condition
                if self.condition_envelope is None
                else self.condition_envelope.merged_with(step.condition)
            )
        if step.enzyme_module is not None:
            self.enzyme_modules.append(step.enzyme_module)
        self.catalysts = _dedupe_strings([*self.catalysts, *self._step_catalysts(step)])

    def condition_conflicts_with(self, step: StepAnnotation) -> bool:
        if self.condition_envelope is None or step.condition is None:
            return False
        return not self.condition_envelope.overlaps(step.condition)

    def _step_catalysts(self, step: StepAnnotation) -> list[str]:
        catalysts = list(step.condition.catalysts if step.condition else [])
        if step.enzyme_module:
            catalysts.extend(step.enzyme_module.ec_numbers)
            catalysts.extend(step.enzyme_module.enzyme_ids)
        return catalysts

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.condition_envelope is not None:
            data["condition_envelope"] = self.condition_envelope.to_dict()
        data["enzyme_modules"] = [module.to_dict() for module in self.enzyme_modules]
        return data


@dataclass
class StageTransition:
    from_stage_id: str
    to_stage_id: str
    operation_type: str = "telescoped"
    required_operations: list[str] = field(default_factory=list)
    cost: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageGraph:
    stages: list[Stage] = field(default_factory=lambda: [Stage(stage_id="stage_1")])
    transitions: list[StageTransition] = field(default_factory=list)

    def ensure_stage(self, stage_id: str, *, operation_type: str = "one_pot") -> Stage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        stage = Stage(stage_id=stage_id, operation_type=operation_type)
        self.stages.append(stage)
        return stage

    def add_step(self, stage_id: str, step_index: int, step: StepAnnotation) -> None:
        stage = self.ensure_stage(stage_id)
        stage.add_step(step_index, step)

    def add_transition(self, transition: StageTransition) -> None:
        self.ensure_stage(transition.from_stage_id)
        self.ensure_stage(transition.to_stage_id, operation_type=transition.operation_type)
        self.transitions.append(transition)

    def stage_for_step(self, step_index: int) -> str:
        for stage in self.stages:
            if step_index in stage.step_indices:
                return stage.stage_id
        return ""

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @classmethod
    def from_partition(cls, partition: list[str]) -> "StageGraph":
        graph = cls(stages=[])
        if not partition:
            return cls()
        for idx, stage_id in enumerate(partition):
            sid = stage_id or "stage_1"
            graph.ensure_stage(sid).step_indices.append(idx)
        return graph

    def to_partition(self, n_steps: int) -> list[str]:
        out = ["stage_1" for _ in range(n_steps)]
        for stage in self.stages:
            for idx in stage.step_indices:
                if 0 <= idx < n_steps:
                    out[idx] = stage.stage_id
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


@dataclass
class CascadeFailure:
    """Typed failure used as a search-control signal, not just a report label."""

    kind: CascadeFailureKind | str
    message: str = ""
    target_leaf: str = ""
    step_index: int | None = None
    severity: float = 1.0
    repair_options: list[CascadeActionType | str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = cascade_failure_kind(self.kind)
        self.repair_options = [cascade_action_type(action) for action in self.repair_options]

    @property
    def category(self) -> str:
        return self.kind.value if isinstance(self.kind, CascadeFailureKind) else str(self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.category,
            "message": self.message,
            "target_leaf": self.target_leaf,
            "step_index": self.step_index,
            "severity": self.severity,
            "repair_options": [
                action.value if isinstance(action, CascadeActionType) else str(action)
                for action in self.repair_options
            ],
            "metadata": self.metadata,
        }


@dataclass
class CascadeAction:
    """A native search action over route, stage, cofactor, or evidence state."""

    action_type: CascadeActionType | str
    target_leaf: str = ""
    step: StepAnnotation | None = None
    module: CascadeModule | None = None
    stage_transition: StageTransition | None = None
    evidence_payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    cost_delta: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action_type = cascade_action_type(self.action_type)

    @property
    def kind(self) -> str:
        return self.action_type.value if isinstance(self.action_type, CascadeActionType) else str(self.action_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.kind,
            "target_leaf": self.target_leaf,
            "step": self.step.to_dict() if self.step is not None else None,
            "module": self.module.to_dict() if self.module is not None else None,
            "stage_transition": self.stage_transition.to_dict() if self.stage_transition is not None else None,
            "evidence_payload": self.evidence_payload,
            "source": self.source,
            "cost_delta": self.cost_delta,
            "metadata": self.metadata,
        }


@dataclass
class CascadeProgramState:
    """Process-aware route state searched by AutoPlanner-Cascade."""

    target_smiles: str
    open_molecule_leaves: list[str] = field(default_factory=list)
    reaction_graph: dict[str, Any] = field(default_factory=dict)
    stage_graph: StageGraph = field(default_factory=StageGraph)
    current_stage: str = "stage_1"
    step_annotations: list[StepAnnotation] = field(default_factory=list)
    condition_envelope_by_stage: dict[str, ConditionEnvelope] = field(default_factory=dict)
    enzyme_context_by_stage: dict[str, list[CascadeModule]] = field(default_factory=dict)
    cofactor_ledger: CofactorLedger = field(default_factory=CofactorLedger)
    redox_ledger: RedoxLedger = field(default_factory=RedoxLedger)
    stock_status: dict[str, bool | None] = field(default_factory=dict)
    unresolved_failure_modes: list[CascadeFailure] = field(default_factory=list)
    evidence_confidence: float | None = None
    cascade_cost: float | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    route_graph: dict[str, Any] = field(default_factory=dict)
    open_leaves: list[str] = field(default_factory=list)
    steps: list[StepAnnotation] = field(default_factory=list)
    stage_partition: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.open_leaves and not self.open_molecule_leaves:
            self.open_molecule_leaves = self.open_leaves
        elif self.open_molecule_leaves and not self.open_leaves:
            self.open_leaves = self.open_molecule_leaves
        elif not self.open_leaves and not self.open_molecule_leaves:
            self.open_molecule_leaves = []
            self.open_leaves = self.open_molecule_leaves

        if self.steps and not self.step_annotations:
            self.step_annotations = self.steps
        elif self.step_annotations and not self.steps:
            self.steps = self.step_annotations
        elif not self.steps and not self.step_annotations:
            self.step_annotations = []
            self.steps = self.step_annotations

        if self.route_graph and not self.reaction_graph:
            self.reaction_graph = self.route_graph
        elif self.reaction_graph and not self.route_graph:
            self.route_graph = self.reaction_graph
        elif not self.route_graph and not self.reaction_graph:
            self.reaction_graph = {}
            self.route_graph = self.reaction_graph

        if self.stage_partition and self.stage_graph.n_stages <= 1 and not self.stage_graph.stages[0].step_indices:
            self.stage_graph = StageGraph.from_partition(self.stage_partition)
        elif not self.stage_partition and self.step_annotations:
            self.stage_partition = self.stage_graph.to_partition(len(self.step_annotations))

        self._sync_stage_maps()

    @classmethod
    def initial(cls, target_smiles: str, *, stock_status: dict[str, bool | None] | None = None) -> "CascadeProgramState":
        state = cls(target_smiles=target_smiles, open_molecule_leaves=[target_smiles], stock_status=dict(stock_status or {}))
        state.open_leaves = state.open_molecule_leaves
        return state

    def copy(self) -> "CascadeProgramState":
        copied = copy.deepcopy(self)
        copied._sync_aliases()
        return copied

    def append_step(self, step: StepAnnotation, *, opened_leaves: list[str] | None = None) -> None:
        step.stage_id = step.stage_id or self.current_stage
        idx = len(self.step_annotations)
        self.step_annotations.append(step)
        self.steps = self.step_annotations
        self.stage_graph.add_step(step.stage_id, idx, step)
        self.current_stage = step.stage_id
        self.cofactor_ledger.record_step(step)
        self.redox_ledger.record_condition(step.condition)
        self.stock_status.update(step.stock_status)
        if step.condition is not None:
            existing = self.condition_envelope_by_stage.get(step.stage_id)
            self.condition_envelope_by_stage[step.stage_id] = step.condition if existing is None else existing.merged_with(step.condition)
        if step.enzyme_module is not None:
            self.enzyme_context_by_stage.setdefault(step.stage_id, []).append(step.enzyme_module)
        if opened_leaves is not None:
            self.open_molecule_leaves = list(opened_leaves)
            self.open_leaves = self.open_molecule_leaves
        self.stage_partition = self.stage_graph.to_partition(len(self.step_annotations))

    def add_stage_transition(self, transition: StageTransition) -> None:
        self.stage_graph.add_transition(transition)
        self.current_stage = transition.to_stage_id
        self.stage_partition = self.stage_graph.to_partition(len(self.step_annotations))

    def add_module(self, module: CascadeModule, *, stage_id: str | None = None) -> None:
        sid = stage_id or self.current_stage or "stage_1"
        self.stage_graph.ensure_stage(sid)
        self.enzyme_context_by_stage.setdefault(sid, []).append(module)
        self.cofactor_ledger.record_module(module)
        if module.condition_envelope is not None:
            existing = self.condition_envelope_by_stage.get(sid)
            self.condition_envelope_by_stage[sid] = (
                module.condition_envelope if existing is None else existing.merged_with(module.condition_envelope)
            )

    @property
    def stock_closed(self) -> bool:
        leaves = self.open_molecule_leaves or self.open_leaves
        if not leaves:
            return True
        known_stock = dict(self.stock_status)
        for step in self.step_annotations:
            known_stock.update(step.stock_status)
        return all(bool(known_stock.get(leaf)) for leaf in leaves)

    @property
    def open_nonstock_leaves(self) -> list[str]:
        return [leaf for leaf in (self.open_molecule_leaves or self.open_leaves) if not self.stock_status.get(leaf)]

    def condition_conflicts(self) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        compared = False
        for stage in self.stage_graph.stages:
            stage_steps = [
                self.step_annotations[idx]
                for idx in stage.step_indices
                if 0 <= idx < len(self.step_annotations)
            ]
            for left, right in zip(stage_steps, stage_steps[1:]):
                compared = True
                if left.condition is not None and right.condition is not None and not left.condition.overlaps(right.condition):
                    conflicts.append(
                        {
                            "stage_id": stage.stage_id,
                            "left_step_index": self.step_annotations.index(left),
                            "right_step_index": self.step_annotations.index(right),
                        }
                    )
        if not compared:
            for idx, (left, right) in enumerate(zip(self.step_annotations, self.step_annotations[1:])):
                if left.condition is not None and right.condition is not None and not left.condition.overlaps(right.condition):
                    conflicts.append(
                        {
                            "stage_id": self.current_stage or "stage_1",
                            "left_step_index": idx,
                            "right_step_index": idx + 1,
                        }
                    )
        return conflicts

    def _sync_aliases(self) -> None:
        self.open_leaves = self.open_molecule_leaves
        self.steps = self.step_annotations
        self.route_graph = self.reaction_graph
        self.stage_partition = self.stage_graph.to_partition(len(self.step_annotations))
        self._sync_stage_maps()

    def _sync_stage_maps(self) -> None:
        for stage in self.stage_graph.stages:
            if stage.condition_envelope is not None:
                self.condition_envelope_by_stage.setdefault(stage.stage_id, stage.condition_envelope)
            if stage.enzyme_modules:
                self.enzyme_context_by_stage.setdefault(stage.stage_id, []).extend(stage.enzyme_modules)

    def to_dict(self) -> dict[str, Any]:
        self._sync_aliases()
        condition_state = summarize_condition_state(self)
        return {
            "target_smiles": self.target_smiles,
            "open_molecule_leaves": list(self.open_molecule_leaves),
            "reaction_graph": self.reaction_graph,
            "stage_graph": self.stage_graph.to_dict(),
            "current_stage": self.current_stage,
            "step_annotations": [step.to_dict() for step in self.step_annotations],
            "condition_envelope_by_stage": {
                stage_id: condition.to_dict()
                for stage_id, condition in self.condition_envelope_by_stage.items()
            },
            "enzyme_context_by_stage": {
                stage_id: [module.to_dict() for module in modules]
                for stage_id, modules in self.enzyme_context_by_stage.items()
            },
            "cofactor_ledger": self.cofactor_ledger.to_dict(),
            "redox_ledger": self.redox_ledger.to_dict(),
            "stock_status": self.stock_status,
            "unresolved_failure_modes": [failure.to_dict() for failure in self.unresolved_failure_modes],
            "evidence_confidence": self.evidence_confidence,
            "cascade_cost": self.cascade_cost,
            "stock_closed": self.stock_closed,
            "raw_metadata": self.raw_metadata,
            "condition_state": condition_state.to_dict(),
            # Legacy contract retained for route-tree and baseline tests.
            "route_graph": self.route_graph,
            "open_leaves": list(self.open_leaves),
            "steps": [step.to_dict() for step in self.steps],
            "stage_partition": list(self.stage_partition),
        }


CascadeSearchState = CascadeProgramState


def cascade_action_type(value: CascadeActionType | str) -> CascadeActionType | str:
    if isinstance(value, CascadeActionType):
        return value
    text = str(value or "")
    for item in CascadeActionType:
        if text == item.value or text == item.name:
            return item
    return text


def cascade_failure_kind(value: CascadeFailureKind | str) -> CascadeFailureKind | str:
    if isinstance(value, CascadeFailureKind):
        return value
    text = str(value or "")
    for item in CascadeFailureKind:
        if text == item.value or text == item.name:
            return item
    return text


def ranges_overlap(
    left_min: float | None,
    left_max: float | None,
    right_min: float | None,
    right_max: float | None,
) -> bool:
    if None in {left_min, left_max, right_min, right_max}:
        return True
    return max(float(left_min), float(right_min)) <= min(float(left_max), float(right_max))


def intersect_or_union(
    left_min: float | None,
    left_max: float | None,
    right_min: float | None,
    right_max: float | None,
) -> tuple[float | None, float | None]:
    if None in {left_min, left_max, right_min, right_max}:
        return union_range(left_min, left_max, right_min, right_max)
    if ranges_overlap(left_min, left_max, right_min, right_max):
        return max(float(left_min), float(right_min)), min(float(left_max), float(right_max))
    return union_range(left_min, left_max, right_min, right_max)


def union_range(
    left_min: float | None,
    left_max: float | None,
    right_min: float | None,
    right_max: float | None,
) -> tuple[float | None, float | None]:
    values = [value for value in (left_min, left_max, right_min, right_max) if value is not None]
    if not values:
        return None, None
    return min(float(value) for value in values), max(float(value) for value in values)


def _add_amounts(target: dict[str, float], values: dict[str, float]) -> None:
    for key, value in (values or {}).items():
        if not key:
            continue
        target[str(key)] = float(target.get(str(key)) or 0.0) + float(value or 0.0)


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = _normalize_token(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_predicted_temperature_c(payload: dict[str, Any]) -> float | None:
    value = _safe_float(_first_present(payload, "temperature_c", "Temperature", "T", "temperature"))
    if value is None:
        return None
    unit = str(_first_present(payload, "temperature_unit", "TemperatureUnit", "unit") or "").strip().lower()
    if unit in {"k", "kelvin"}:
        value -= 273.15
    # RCR/Parrot outputs are weak predictions. Values outside this broad
    # process-planning range are retained only in raw evidence, not used as
    # hard compatibility envelopes inside cascade search.
    if value < -100.0 or value > 220.0:
        return None
    return value


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _max_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    if not values:
        return None
    return max(float(value) for value in values)


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower().replace("_", " ")
