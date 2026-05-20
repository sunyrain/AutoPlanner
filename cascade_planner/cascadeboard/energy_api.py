"""Energy API for CascadeBoard.

Layer 5: Unified interface wrapping frozen experts as energy terms.
Computes feasibility, quality vector, risk vector, and total energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cascade_planner.cascadeboard import (
    CascadeBoard, Slot, CompiledConstraints, SoftFactor,
)


@dataclass
class SlotEnergy:
    e_retro: float = 0.0
    e_enzyme: float = 0.0
    e_condition: float = 0.0


class EnergyAPI:
    """Unified energy interface wrapping frozen experts.

    All experts are optional — missing experts contribute 0 energy.
    This allows incremental integration and testing without all models.
    """

    def __init__(
        self,
        retro_scorer=None,
        enzyme_scorer=None,
        condition_scorer=None,
        pairwise_scorer=None,
        stock_checker=None,
        route_energy_model=None,
        motif_memory: dict | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.retro_scorer = retro_scorer
        self.enzyme_scorer = enzyme_scorer
        self.condition_scorer = condition_scorer
        self.pairwise_scorer = pairwise_scorer
        self.stock_checker = stock_checker or _default_stock
        self.route_energy_model = route_energy_model
        self.motif_memory = motif_memory

        self.weights = weights or {
            "retro": 2.0,
            "enzyme": 1.5,
            "condition": 1.5,
            "compatibility": 3.0,
            "stock": 2.0,
            "length": 1.0,
            "route": 1.0,
            "motif": 0.5,
        }

    # ------------------------------------------------------------------
    # Per-slot energy
    # ------------------------------------------------------------------

    def score_slot(self, slot: Slot) -> SlotEnergy:
        """Score a single slot using frozen experts."""
        e = SlotEnergy()

        # E_retro: use pre-computed score if available, else try scorer
        if slot.e_retro is not None:
            e.e_retro = slot.e_retro
        elif self.retro_scorer and slot.reaction_smiles:
            try:
                e.e_retro = self.retro_scorer(slot.reaction_smiles)
            except Exception:
                e.e_retro = 0.0

        # E_enzyme: only for enzymatic steps
        if slot.is_enzymatic():
            if slot.e_enzyme is not None:
                e.e_enzyme = slot.e_enzyme
            elif self.enzyme_scorer and slot.ec:
                try:
                    e.e_enzyme = self.enzyme_scorer(
                        slot.ec, slot.reaction_smiles, slot.enzyme_uid
                    )
                except Exception:
                    e.e_enzyme = 0.0
        else:
            e.e_enzyme = 1.0  # chemical steps get full enzyme score

        # E_condition
        if slot.e_condition is not None:
            e.e_condition = slot.e_condition
        elif self.condition_scorer and slot.ec:
            try:
                e.e_condition = self.condition_scorer(slot.ec, slot.T, slot.pH)
            except Exception:
                e.e_condition = 0.0
        elif slot.T is not None:
            # Penalize extreme temperatures more aggressively
            # Enzymatic: optimal 20-50°C; Chemical: wider range but >100°C is bad
            if slot.is_enzymatic():
                e.e_condition = max(0, 1.0 - abs(slot.T - 35) / 25)  # steep penalty outside 10-60°C
            else:
                e.e_condition = max(0, 1.0 - abs(slot.T - 50) / 80)
        else:
            e.e_condition = 0.5  # unknown condition = moderate penalty (not 0, not 1)

        return e

    # ------------------------------------------------------------------
    # Pairwise compatibility
    # ------------------------------------------------------------------

    def score_compatibility(self, slot_a: Slot, slot_b: Slot) -> float:
        """Score compatibility between adjacent slots."""
        if self.pairwise_scorer:
            try:
                return self.pairwise_scorer(slot_a, slot_b)
            except Exception:
                pass

        # Deterministic fallback: penalize condition differences
        score = 1.0
        if slot_a.T is not None and slot_b.T is not None:
            dt = abs(slot_a.T - slot_b.T)
            score *= max(0, 1.0 - dt / 50)
        if slot_a.pH is not None and slot_b.pH is not None:
            dpH = abs(slot_a.pH - slot_b.pH)
            score *= max(0, 1.0 - dpH / 5)
        if slot_a.solvent and slot_b.solvent:
            if slot_a.solvent == slot_b.solvent:
                score *= 1.1  # bonus for same solvent
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Stock check
    # ------------------------------------------------------------------

    def score_stock(self, board: CascadeBoard) -> float:
        """Fraction of leaf nodes that are in stock."""
        if not board.slots:
            return 0.0
        leaves = []
        last = board.slots[-1]
        if last.main_reactant:
            leaves.append(last.main_reactant)
        for slot in board.slots:
            leaves.extend(slot.aux_reactants)
        if not leaves:
            return 0.0
        in_stock = sum(1 for s in leaves if self.stock_checker(s))
        return in_stock / len(leaves)

    # ------------------------------------------------------------------
    # Soft factor energy
    # ------------------------------------------------------------------

    def score_soft_factors(
        self, board: CascadeBoard, factors: list[SoftFactor],
    ) -> float:
        """Evaluate soft constraint factors."""
        total = 0.0
        for f in factors:
            if f.name == "one_pot":
                # Penalize condition gaps (proxy for purification need)
                max_dt = max(
                    (abs(board.slots[i].T - board.slots[i + 1].T)
                     for i in range(board.n_steps - 1)
                     if board.slots[i].T and board.slots[i + 1].T),
                    default=0,
                )
                total += f.weight * max(0, 1.0 - max_dt / 30)
            elif f.name == "max_delta_T":
                max_dt = max(
                    (abs(board.slots[i].T - board.slots[i + 1].T)
                     for i in range(board.n_steps - 1)
                     if board.slots[i].T and board.slots[i + 1].T),
                    default=0,
                )
                limit = f.params.get("max_delta", 20)
                total += f.weight * (1.0 if max_dt <= limit else -1.0)
            elif f.name == "few_steps":
                total += f.weight * (1.0 / board.n_steps)
            elif f.name == "prefer_enzymatic":
                n_enz = sum(1 for s in board.slots if s.is_enzymatic())
                total += f.weight * (n_enz / max(board.n_steps, 1))
            elif f.name in ("high_yield", "mild_conditions", "cheap_stock",
                            "no_toxic_solvent", "low_literature_support", "novel_enzyme"):
                total += f.weight * 0.5  # placeholder
        return total

    # ------------------------------------------------------------------
    # Route motif plausibility
    # ------------------------------------------------------------------

    def score_motif(self, board: CascadeBoard) -> float:
        """Score how well the route matches known cascade motifs.

        Uses pairwise (type_i, type_i+1) and (ec1_i, ec1_i+1) bigrams
        from the motif memory. Returns 0-1 where 1 = all bigrams are
        known motifs.
        """
        if not self.motif_memory or board.n_steps < 2:
            return 0.5

        type_motifs = self.motif_memory.get("type", {})
        ec_motifs = self.motif_memory.get("ec", {})
        if not type_motifs and not ec_motifs:
            return 0.5

        max_type_count = max(type_motifs.values()) if type_motifs else 1
        max_ec_count = max(ec_motifs.values()) if ec_motifs else 1

        scores = []
        for i in range(board.n_steps - 1):
            sa, sb = board.slots[i], board.slots[i + 1]

            # Type bigram
            ta = sa.reaction_type or ""
            tb = sb.reaction_type or ""
            type_key = repr((ta, tb))
            type_count = type_motifs.get(type_key, 0)
            scores.append(min(1.0, type_count / max_type_count) if ta and tb else 0.3)

            # EC1 bigram
            ec_a = sa.ec.split(".")[0] if sa.ec and sa.ec[0].isdigit() else "-"
            ec_b = sb.ec.split(".")[0] if sb.ec and sb.ec[0].isdigit() else "-"
            ec_key = repr((ec_a, ec_b))
            ec_count = ec_motifs.get(ec_key, 0)
            scores.append(min(1.0, ec_count / max_ec_count) if ec_a != "-" or ec_b != "-" else 0.3)

        return float(sum(scores) / len(scores)) if scores else 0.5

    # ------------------------------------------------------------------
    # Full board energy
    # ------------------------------------------------------------------

    def compute_energy(
        self,
        board: CascadeBoard,
        compiled: CompiledConstraints | None = None,
    ) -> float:
        """Compute total energy for a board. Lower = better."""
        w = self.weights

        # Per-slot energies
        slot_energies = [self.score_slot(s) for s in board.slots]
        for i, (slot, se) in enumerate(zip(board.slots, slot_energies)):
            slot.e_retro = se.e_retro
            slot.e_enzyme = se.e_enzyme
            slot.e_condition = se.e_condition

        e_retro = np.mean([se.e_retro for se in slot_energies]) if slot_energies else 0
        e_enzyme = np.mean([se.e_enzyme for se in slot_energies]) if slot_energies else 0
        e_condition = np.mean([se.e_condition for se in slot_energies]) if slot_energies else 0

        # Pairwise compatibility
        compat_scores = []
        for i in range(board.n_steps - 1):
            cs = self.score_compatibility(board.slots[i], board.slots[i + 1])
            compat_scores.append(cs)
        board.compatibility_scores = compat_scores
        e_compat = min(compat_scores) if compat_scores else 1.0  # bottleneck

        # Stock
        e_stock = self.score_stock(board)

        # Length penalty
        e_length = 1.0 / board.n_steps

        # Route-level energy (trainable, optional)
        e_route = 0.0
        if self.route_energy_model:
            try:
                e_route = self.route_energy_model(board)
            except Exception:
                pass

        # Soft factors
        e_soft = 0.0
        if compiled:
            e_soft = self.score_soft_factors(board, compiled.soft_factors)

        # Route motif plausibility
        e_motif = self.score_motif(board)

        # Total energy (negative = better, we want to maximize)
        total = (
            w["retro"] * e_retro
            + w["enzyme"] * e_enzyme
            + w["condition"] * e_condition
            + w["compatibility"] * e_compat
            + w["stock"] * e_stock
            + w["length"] * e_length
            + w["route"] * e_route
            + w.get("motif", 0.5) * e_motif
            + e_soft
        )

        board.total_energy = -total  # negate so lower = better
        return board.total_energy

    # ------------------------------------------------------------------
    # Quality vector
    # ------------------------------------------------------------------

    def compute_quality(self, board: CascadeBoard) -> dict[str, float]:
        """Compute multi-dimensional quality vector."""
        q: dict[str, float] = {}

        # Feasibility (mean of per-slot retro scores)
        retro_scores = [s.e_retro for s in board.slots if s.e_retro is not None]
        q["feasibility"] = np.mean(retro_scores) if retro_scores else 0.0

        # Condition compatibility
        if board.compatibility_scores:
            q["condition_compatibility"] = min(board.compatibility_scores)
        else:
            q["condition_compatibility"] = 1.0

        # Stock accessibility
        q["stock_accessibility"] = self.score_stock(board)

        # Step count
        q["step_count"] = board.n_steps

        # Enzyme availability
        enz_scores = [s.e_enzyme for s in board.slots if s.is_enzymatic() and s.e_enzyme is not None]
        q["enzyme_availability"] = np.mean(enz_scores) if enz_scores else 1.0

        # One-pot feasibility (proxy: max ΔT between adjacent steps)
        if board.n_steps >= 2:
            max_dt = max(
                (abs(board.slots[i].T - board.slots[i + 1].T)
                 for i in range(board.n_steps - 1)
                 if board.slots[i].T and board.slots[i + 1].T),
                default=0,
            )
            q["one_pot_feasibility"] = max(0, 1.0 - max_dt / 30)
        else:
            q["one_pot_feasibility"] = 1.0

        board.quality_vector = q
        return q

    # ------------------------------------------------------------------
    # Risk vector
    # ------------------------------------------------------------------

    def compute_risk(self, board: CascadeBoard) -> dict[str, float]:
        """Compute risk/uncertainty vector."""
        r: dict[str, float] = {}

        # Reaction uncertainty: low retro score = high uncertainty
        retro_scores = [s.e_retro for s in board.slots if s.e_retro is not None]
        r["reaction_uncertainty"] = 1.0 - (np.mean(retro_scores) if retro_scores else 0.0)

        # Enzyme uncertainty
        enz_scores = [s.e_enzyme for s in board.slots if s.is_enzymatic() and s.e_enzyme is not None]
        r["enzyme_uncertainty"] = 1.0 - (np.mean(enz_scores) if enz_scores else 0.0)

        # Condition uncertainty: slots without T/pH have high uncertainty
        n_missing = sum(1 for s in board.slots if s.T is None or s.pH is None)
        r["condition_uncertainty"] = n_missing / max(board.n_steps, 1)

        # Candidate pool coverage
        n_with_candidates = sum(1 for s in board.slots if s.candidates)
        r["candidate_pool_coverage"] = 1.0 - n_with_candidates / max(board.n_steps, 1)

        board.risk_vector = r
        return r

    # ------------------------------------------------------------------
    # Bottleneck diagnosis
    # ------------------------------------------------------------------

    def diagnose_bottleneck(self, board: CascadeBoard) -> tuple[int | None, str]:
        """Find the worst slot and explain why."""
        if not board.slots:
            return None, ""

        slot_scores = []
        for i, slot in enumerate(board.slots):
            local = (slot.e_retro or 0) + (slot.e_enzyme or 0) + (slot.e_condition or 0)
            if i > 0 and i - 1 < len(board.compatibility_scores):
                local += board.compatibility_scores[i - 1]
            slot_scores.append((i, local))

        worst_idx, worst_score = min(slot_scores, key=lambda x: x[1])
        slot = board.slots[worst_idx]

        reasons = []
        if (slot.e_retro or 0) < 0.3:
            reasons.append(f"low retro score ({slot.e_retro or 0:.2f})")
        if slot.is_enzymatic() and (slot.e_enzyme or 0) < 0.3:
            reasons.append(f"low enzyme compatibility ({slot.e_enzyme or 0:.2f})")
        if worst_idx > 0 and worst_idx - 1 < len(board.compatibility_scores):
            cs = board.compatibility_scores[worst_idx - 1]
            if cs < 0.5:
                prev = board.slots[worst_idx - 1]
                dt = abs((slot.T or 0) - (prev.T or 0))
                reasons.append(f"poor compatibility with prev step (ΔT={dt:.0f}°C)")

        reason = f"Step {worst_idx}: " + "; ".join(reasons) if reasons else f"Step {worst_idx}: lowest combined score"
        board.bottleneck_slot = worst_idx
        board.bottleneck_reason = reason
        return worst_idx, reason


# ---------------------------------------------------------------------------
def _default_stock(smiles: str) -> bool:
    try:
        from cascade_planner.cascadeboard.zinc_stock import is_in_zinc_stock
        return is_in_zinc_stock(smiles)
    except Exception:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and mol.GetNumHeavyAtoms() <= 6
