"""Constraint Compiler for CascadeBoard.

Layer 0: Compiles user constraints into hard masks, soft factors,
conflict detection, and relaxation suggestions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cascade_planner.cascadeboard import (
    CascadeBoard, HardMask, SoftFactor, ConflictReport, CompiledConstraints,
)


# ---------------------------------------------------------------------------
# Known reaction types and EC classes
# ---------------------------------------------------------------------------

REACTION_TYPES = frozenset({
    "oxidation", "reduction", "acylation", "hydrolysis", "amination",
    "C_C_coupling", "isomerization", "phosphorylation", "glycosylation",
    "functional_group_interconversion", "racemization", "esterification",
    "other",
})

EC1_CLASSES = frozenset({"1", "2", "3", "4", "5", "6"})

# Typical enzyme condition ranges (from BRENDA/AutoPlanner)
EC1_T_RANGES: dict[str, tuple[float, float]] = {
    "1": (20, 60), "2": (20, 55), "3": (20, 70),
    "4": (20, 50), "5": (20, 45), "6": (25, 45),
}
EC1_PH_RANGES: dict[str, tuple[float, float]] = {
    "1": (5, 10), "2": (6, 9), "3": (5, 9),
    "4": (5, 9), "5": (6, 9), "6": (6, 8.5),
}


class ConstraintCompiler:
    """Compile user constraints into executable form."""

    def compile(
        self,
        board: CascadeBoard,
        raw_constraints: dict[str, Any] | None = None,
        objective: str = "balanced",
    ) -> CompiledConstraints:
        raw = raw_constraints or {}
        hard_masks: list[HardMask] = []
        soft_factors: list[SoftFactor] = []
        conflicts: list[ConflictReport] = []
        relaxations: list[str] = []

        # ----- 1. Compile fixed fields from board into hard masks -----
        for slot in board.slots:
            for fld in slot.fixed_fields:
                val = getattr(slot, fld, None)
                if val is not None:
                    if fld == "ec" and isinstance(val, str) and val.endswith(".x"):
                        # Prefix match: "2.6.1.x" → allow any 2.6.1.*
                        hard_masks.append(HardMask(
                            slot_index=slot.index, field=fld,
                            allowed_values={val},
                        ))
                    else:
                        hard_masks.append(HardMask(
                            slot_index=slot.index, field=fld,
                            allowed_values={val},
                        ))

        # ----- 2. Compile raw constraints -----

        # Exclude constraints
        for key in ("exclude_catalyst", "exclude_solvent", "exclude_ec"):
            if key in raw:
                fld = key.replace("exclude_", "")
                vals = raw[key] if isinstance(raw[key], set) else {raw[key]}
                hard_masks.append(HardMask(
                    slot_index=None, field=fld, excluded_values=vals,
                ))

        # One-pot constraint
        if raw.get("one_pot"):
            soft_factors.append(SoftFactor(
                name="one_pot", weight=3.0,
                params={"max_purifications": 0},
            ))

        # Max delta T
        if "max_delta_T" in raw:
            soft_factors.append(SoftFactor(
                name="max_delta_T", weight=2.0,
                params={"max_delta": raw["max_delta_T"]},
            ))

        # Max delta pH
        if "max_delta_pH" in raw:
            soft_factors.append(SoftFactor(
                name="max_delta_pH", weight=2.0,
                params={"max_delta": raw["max_delta_pH"]},
            ))

        # Prefer enzymatic
        if raw.get("prefer_enzymatic"):
            soft_factors.append(SoftFactor(name="prefer_enzymatic", weight=1.0))

        # Max cost
        if "max_cost" in raw:
            soft_factors.append(SoftFactor(
                name="max_cost", weight=1.5,
                params={"max_cost": raw["max_cost"]},
            ))

        # ----- 3. Objective-based soft factors -----
        if objective == "industrial":
            soft_factors.extend([
                SoftFactor(name="high_yield", weight=3.0),
                SoftFactor(name="few_steps", weight=2.0),
                SoftFactor(name="cheap_stock", weight=2.0),
            ])
        elif objective == "green":
            soft_factors.extend([
                SoftFactor(name="one_pot", weight=3.0),
                SoftFactor(name="mild_conditions", weight=2.0),
                SoftFactor(name="no_toxic_solvent", weight=2.0),
            ])
        elif objective == "novelty":
            soft_factors.extend([
                SoftFactor(name="low_literature_support", weight=2.0),
                SoftFactor(name="novel_enzyme", weight=1.5),
            ])

        # ----- 4. Conflict detection -----
        conflicts = self._detect_conflicts(board, hard_masks, soft_factors)
        if conflicts:
            relaxations = self._suggest_relaxations(conflicts, board)

        return CompiledConstraints(
            hard_masks=hard_masks,
            soft_factors=soft_factors,
            conflicts=conflicts,
            relaxations=relaxations,
        )

    # ------------------------------------------------------------------
    def _detect_conflicts(
        self,
        board: CascadeBoard,
        hard_masks: list[HardMask],
        soft_factors: list[SoftFactor],
    ) -> list[ConflictReport]:
        conflicts = []

        # Check: one-pot + large T/pH gap between fixed steps
        has_one_pot = any(f.name == "one_pot" for f in soft_factors)
        if has_one_pot and board.n_steps >= 2:
            for i in range(board.n_steps - 1):
                s_a, s_b = board.slots[i], board.slots[i + 1]
                if s_a.T is not None and s_b.T is not None:
                    dt = abs(s_a.T - s_b.T)
                    if dt > 30:
                        conflicts.append(ConflictReport(
                            description=(
                                f"one_pot requested but Step {i} (T={s_a.T}°C) and "
                                f"Step {i+1} (T={s_b.T}°C) have ΔT={dt:.0f}°C"
                            ),
                            involved_constraints=["one_pot", f"step_{i}_T", f"step_{i+1}_T"],
                            suggested_relaxations=[
                                f"Allow purification between Step {i} and Step {i+1}",
                                f"Relax one_pot to telescoped",
                                f"Adjust Step {i} T closer to Step {i+1}",
                            ],
                        ))

        # Check: fixed enzyme EC vs fixed condition
        for slot in board.slots:
            if "ec" in slot.fixed_fields and "T" in slot.fixed_fields:
                ec1 = slot.ec.split(".")[0] if slot.ec else None
                if ec1 and ec1 in EC1_T_RANGES:
                    t_lo, t_hi = EC1_T_RANGES[ec1]
                    if slot.T is not None and not (t_lo <= slot.T <= t_hi):
                        conflicts.append(ConflictReport(
                            description=(
                                f"Step {slot.index}: fixed EC {slot.ec} typical T range "
                                f"is {t_lo}-{t_hi}°C but fixed T={slot.T}°C"
                            ),
                            involved_constraints=[f"step_{slot.index}_ec", f"step_{slot.index}_T"],
                            suggested_relaxations=[
                                f"Relax T to {t_lo}-{t_hi}°C range",
                                f"Use a thermostable variant of EC {slot.ec}",
                            ],
                        ))

        # Check: fixed enzyme EC vs fixed condition (pH)
        for slot in board.slots:
            if "ec" in slot.fixed_fields and "pH" in slot.fixed_fields:
                ec1 = slot.ec.split(".")[0] if slot.ec else None
                if ec1 and ec1 in EC1_PH_RANGES:
                    ph_lo, ph_hi = EC1_PH_RANGES[ec1]
                    if slot.pH is not None and not (ph_lo <= slot.pH <= ph_hi):
                        conflicts.append(ConflictReport(
                            description=(
                                f"Step {slot.index}: fixed EC {slot.ec} typical pH range "
                                f"is {ph_lo}-{ph_hi} but fixed pH={slot.pH}"
                            ),
                            involved_constraints=[f"step_{slot.index}_ec", f"step_{slot.index}_pH"],
                            suggested_relaxations=[
                                f"Relax pH to {ph_lo}-{ph_hi} range",
                                f"Use a pH-tolerant variant of EC {slot.ec}",
                            ],
                        ))

        # Check: max_delta_T vs fixed temperatures
        max_dt_factors = [f for f in soft_factors if f.name == "max_delta_T"]
        if max_dt_factors and board.n_steps >= 2:
            limit = max_dt_factors[0].params.get("max_delta", 20)
            for i in range(board.n_steps - 1):
                sa, sb = board.slots[i], board.slots[i + 1]
                if (sa.T is not None and sb.T is not None
                        and "T" in sa.fixed_fields and "T" in sb.fixed_fields):
                    dt = abs(sa.T - sb.T)
                    if dt > limit:
                        conflicts.append(ConflictReport(
                            description=(
                                f"max_delta_T={limit}°C but fixed Step {i} T={sa.T}°C "
                                f"and Step {i+1} T={sb.T}°C have ΔT={dt:.0f}°C"
                            ),
                            involved_constraints=["max_delta_T", f"step_{i}_T", f"step_{i+1}_T"],
                            suggested_relaxations=[
                                f"Relax max_delta_T to {int(dt)+5}°C",
                                f"Insert a condition-transition step between Step {i} and {i+1}",
                                f"Relax one of the fixed temperatures",
                            ],
                        ))

        # Check: n_steps constraint vs available candidate depth
        n_steps_constraint = board.global_constraints.get("n_steps")
        if n_steps_constraint and n_steps_constraint > 4:
            conflicts.append(ConflictReport(
                description=f"n_steps={n_steps_constraint} requested but candidate graph depth is limited to 4",
                involved_constraints=["n_steps"],
                suggested_relaxations=[f"Reduce to n_steps<=4 or allow flexible step count"],
            ))

        return conflicts

    def _suggest_relaxations(
        self, conflicts: list[ConflictReport], board: CascadeBoard,
    ) -> list[str]:
        suggestions = []
        for c in conflicts:
            suggestions.extend(c.suggested_relaxations)
        # Deduplicate
        return list(dict.fromkeys(suggestions))
