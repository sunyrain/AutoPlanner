"""Independent hard-pruning verifier for route-tree search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rdkit import Chem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState


StockChecker = Callable[[str], bool]


@dataclass(frozen=True)
class RouteVerifierResult:
    accepted: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reasons": list(self.reasons)}


class RouteVerifier:
    """Validate route actions without ranking or proposing candidates."""

    def __init__(self, *, max_delta_T: float = 40.0, max_delta_pH: float = 4.0):
        self.max_delta_T = float(max_delta_T)
        self.max_delta_pH = float(max_delta_pH)

    def verify_action(
        self,
        *,
        state: RouteTreeState,
        leaf: str,
        action: CandidateAction,
        context: Any | None = None,
        stock_checker: StockChecker | None = None,
    ) -> RouteVerifierResult:
        reasons: list[str] = []
        expected_type = str(getattr(context, "reaction_type", "") or "") if context is not None else ""
        expected_ec1 = str(getattr(context, "ec1", "") or "") if context is not None else ""
        expected_T = getattr(context, "T", None) if context is not None else None
        expected_pH = getattr(context, "pH", None) if context is not None else None

        if expected_type and action.reaction_type and not _reaction_type_compatible(action.reaction_type, expected_type):
            reasons.append("skeleton_type_mismatch")
        if expected_ec1 and action.ec and _ec1(action.ec) and _ec1(action.ec) != expected_ec1:
            reasons.append("ec_mismatch")
        if expected_T is not None and action.T is not None and abs(float(action.T) - float(expected_T)) > self.max_delta_T:
            reasons.append("condition_temperature_mismatch")
        if expected_pH is not None and action.pH is not None and abs(float(action.pH) - float(expected_pH)) > self.max_delta_pH:
            reasons.append("condition_pH_mismatch")
        if "product_mismatch" in action.validity_flags:
            reasons.append("reaction_product_mismatch")
        if "self_loop" in action.validity_flags:
            reasons.append("self_loop")
        reactants = [canonical_smiles(smi) for smi in action.reactants]
        reactants = [smi for smi in reactants if smi]
        if not reactants:
            reasons.append("no_reactants")
        if any(smi in state.expanded for smi in reactants):
            reasons.append("route_cycle")
        if not _candidate_atom_balance_ok(action, leaf):
            reasons.append("atom_balance_violation")
        if stock_checker is not None and reactants:
            try:
                if all(bool(stock_checker(smi)) for smi in reactants):
                    pass
            except Exception:
                reasons.append("stock_checker_error")
        return RouteVerifierResult(accepted=not reasons, reasons=tuple(reasons))


def _reaction_type_compatible(action_type: str, expected_type: str) -> bool:
    action = _normalize_reaction_type(action_type)
    expected = _normalize_reaction_type(expected_type)
    generic = {
        "",
        "unknown",
        "enzyme",
        "enzymatic",
        "enzyme_reaction",
        "enzyme_retro_reaction",
        "rhea_reaction",
        "template",
        "external",
    }
    if action.startswith("uspto_class_") or action.startswith("class_"):
        return True
    if action in generic or expected in generic:
        return True
    return action == expected or action in expected or expected in action


def _normalize_reaction_type(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _ec1(value: str | None) -> str:
    text = str(value or "").strip()
    return text.split(".", 1)[0] if text else ""


def _candidate_atom_balance_ok(action: CandidateAction, product: str) -> bool:
    product_atoms = _heavy_atoms(product)
    reactant_atoms = sum(_heavy_atoms(smi) for smi in action.reactants)
    if product_atoms <= 10:
        return True
    return reactant_atoms >= max(4, int(product_atoms * 0.35))


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0
