"""CascadeBoard core data structures.

Layer 2: Slot, CascadeBoard, EditAction, Constraint types.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Edit actions
# ---------------------------------------------------------------------------

class EditType(Enum):
    FILL_FIELD = auto()
    REPLACE_STEP = auto()
    REPLACE_ENZYME = auto()
    ADJUST_CONDITION = auto()
    INSERT_STEP = auto()
    DELETE_STEP = auto()
    SWAP_ORDER = auto()
    RELAX_CONSTRAINT = auto()


@dataclass
class EditAction:
    edit_type: EditType
    slot_index: int
    field: str | None = None
    new_value: Any = None
    reason: str = ""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Slot
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    index: int

    # Molecules
    product: str | None = None
    main_reactant: str | None = None
    aux_reactants: list[str] = field(default_factory=list)

    # Reaction
    reaction_smiles: str | None = None
    reaction_type: str | None = None  # oxidation / reduction / acylation / ...

    # Enzyme / catalyst
    ec: str | None = None
    enzyme_uid: str | None = None
    catalyst: str | None = None

    # Conditions
    T: float | None = None
    pH: float | None = None
    solvent: str | None = None

    # Candidate pool (filled by frozen experts)
    candidates: list[dict] = field(default_factory=list)

    # Energy terms (filled by frozen experts)
    e_retro: float | None = None
    e_enzyme: float | None = None
    e_condition: float | None = None

    # User constraints
    fixed_fields: set[str] = field(default_factory=set)

    # Metadata
    confidence: float = 1.0
    source: str = ""  # "retrochimera" / "enzexpand" / "user" / "inpainted"
    evidence: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    ALL_FIELDS = frozenset({
        "product", "main_reactant", "aux_reactants", "reaction_smiles",
        "reaction_type", "ec", "enzyme_uid", "catalyst",
        "T", "pH", "solvent",
    })

    def is_enzymatic(self) -> bool:
        return self.ec is not None

    def is_filled(self) -> bool:
        return self.reaction_smiles is not None

    def is_fully_fixed(self) -> bool:
        return self.ALL_FIELDS.issubset(self.fixed_fields)

    def editable_fields(self) -> set[str]:
        return self.ALL_FIELDS - self.fixed_fields

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "product": self.product,
            "main_reactant": self.main_reactant,
            "reaction_type": self.reaction_type,
            "ec": self.ec,
            "enzyme_uid": self.enzyme_uid,
            "T": self.T,
            "pH": self.pH,
            "solvent": self.solvent,
            "e_retro": self.e_retro,
            "e_enzyme": self.e_enzyme,
            "e_condition": self.e_condition,
            "confidence": self.confidence,
            "fixed": list(self.fixed_fields),
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# CascadeBoard
# ---------------------------------------------------------------------------

@dataclass
class CascadeBoard:
    """Linear slot chain + global state + constraint interface."""

    slots: list[Slot] = field(default_factory=list)

    # Global state (aggregated from slots)
    compatibility_scores: list[float] = field(default_factory=list)
    quality_vector: dict[str, float] = field(default_factory=dict)
    risk_vector: dict[str, float] = field(default_factory=dict)
    total_energy: float = float("inf")
    user_score: float = 0.0

    # Metadata
    edit_history: list[EditAction] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_n_steps(cls, n: int, target: str | None = None) -> CascadeBoard:
        board = cls(slots=[Slot(index=i) for i in range(n)])
        if target and board.slots:
            board.slots[0].product = target
        return board

    # ------------------------------------------------------------------
    # Constraint interface
    # ------------------------------------------------------------------

    def fix(self, slot_index: int, **kwargs) -> None:
        """Fix fields of a slot. Fixed fields won't be modified by inpainting."""
        slot = self.slots[slot_index]
        for key, value in kwargs.items():
            if key in Slot.ALL_FIELDS:
                setattr(slot, key, value)
                slot.fixed_fields.add(key)

    def fix_n_steps(self, n: int) -> None:
        target = self.slots[0].product if self.slots else None
        self.slots = [Slot(index=i) for i in range(n)]
        if target and self.slots:
            self.slots[0].product = target

    def fix_starting_material(self, smiles: str) -> None:
        if self.slots:
            last = self.slots[-1]
            last.main_reactant = smiles
            last.fixed_fields.add("main_reactant")

    def set_global_constraint(self, key: str, value: Any) -> None:
        """Store a global constraint (e.g. one_pot=True, max_delta_T=20)."""
        if not hasattr(self, "_global_constraints"):
            self._global_constraints: dict[str, Any] = {}
        self._global_constraints[key] = value

    @property
    def global_constraints(self) -> dict[str, Any]:
        return getattr(self, "_global_constraints", {})

    # ------------------------------------------------------------------
    # Slot access
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self.slots)

    def get_mask(self) -> list[tuple[int, set[str]]]:
        """Return (slot_index, editable_fields) for all non-fully-fixed slots."""
        return [
            (s.index, s.editable_fields())
            for s in self.slots
            if not s.is_fully_fixed()
        ]

    def update_slot(self, index: int, **kwargs) -> None:
        """Update slot fields, respecting fixed_fields."""
        slot = self.slots[index]
        for key, value in kwargs.items():
            if key not in slot.fixed_fields:
                setattr(slot, key, value)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> CascadeBoard:
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"CascadeBoard ({self.n_steps} steps, energy={self.total_energy:.3f})"]
        for s in self.slots:
            fixed = ",".join(sorted(s.fixed_fields)) if s.fixed_fields else "-"
            lines.append(
                f"  [{s.index}] type={s.reaction_type or '?':12s} "
                f"ec={s.ec or '-':10s} T={s.T or '?':>5} pH={s.pH or '?':>4} "
                f"fixed=[{fixed}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "slots": [s.to_dict() for s in self.slots],
            "compatibility_scores": self.compatibility_scores,
            "quality_vector": self.quality_vector,
            "risk_vector": self.risk_vector,
            "total_energy": self.total_energy,
            "user_score": self.user_score,
            "global_constraints": self.global_constraints,
        }


# ---------------------------------------------------------------------------
# Constraint types (for the compiler)
# ---------------------------------------------------------------------------

@dataclass
class HardMask:
    """A hard constraint compiled into a candidate filter."""
    slot_index: int | None  # None = applies to all slots
    field: str
    allowed_values: set | None = None  # None = any value OK
    excluded_values: set | None = None


@dataclass
class SoftFactor:
    """A soft constraint compiled into an energy term."""
    name: str
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class ConflictReport:
    """A detected conflict between constraints."""
    description: str
    involved_constraints: list[str] = field(default_factory=list)
    suggested_relaxations: list[str] = field(default_factory=list)


@dataclass
class CompiledConstraints:
    hard_masks: list[HardMask] = field(default_factory=list)
    soft_factors: list[SoftFactor] = field(default_factory=list)
    conflicts: list[ConflictReport] = field(default_factory=list)
    relaxations: list[str] = field(default_factory=list)

    @staticmethod
    def _value_allowed(value: Any, allowed_values: set | None) -> bool:
        if not allowed_values:
            return True
        if value in allowed_values:
            return True
        if isinstance(value, str):
            for allowed in allowed_values:
                if isinstance(allowed, str) and allowed.endswith(".x"):
                    prefix = allowed[:-1]
                    if value.startswith(prefix):
                        return True
        return False

    def hard_satisfied(self, board: CascadeBoard) -> bool:
        for mask in self.hard_masks:
            if mask.slot_index is not None:
                if mask.slot_index >= len(board.slots):
                    continue
                slots = [board.slots[mask.slot_index]]
            else:
                slots = board.slots
            for slot in slots:
                val = getattr(slot, mask.field, None)
                if val is None:
                    continue
                if not self._value_allowed(val, mask.allowed_values):
                    return False
                if mask.excluded_values and val in mask.excluded_values:
                    return False
        return True


# ---------------------------------------------------------------------------
# Route explanation
# ---------------------------------------------------------------------------

@dataclass
class RouteExplanation:
    why_selected: str = ""
    what_was_changed: list[str] = field(default_factory=list)
    constraints_satisfied: dict[str, str] = field(default_factory=dict)
    constraints_at_risk: dict[str, str] = field(default_factory=dict)
    global_condition_window: str = ""
    evidence_table: list[dict] = field(default_factory=list)
    edited_slots: list[str] = field(default_factory=list)
    alternative_edits: list[str] = field(default_factory=list)
    minimal_relaxation: str | None = None
    uncertainty_table: dict[str, float] = field(default_factory=dict)


@dataclass
class RouteResult:
    board: CascadeBoard
    quality_vector: dict[str, float] = field(default_factory=dict)
    risk_vector: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    confidence: float = 1.0
    constraint_report: dict[str, str] = field(default_factory=dict)
    bottleneck_slot: int | None = None
    bottleneck_reason: str = ""
    explanation: RouteExplanation = field(default_factory=RouteExplanation)
