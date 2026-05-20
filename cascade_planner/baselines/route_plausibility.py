"""Lightweight route plausibility checks for backend route candidates.

These checks are deliberately conservative.  Backend reaction records are not
atom mapped and may omit reagents, salts, water, cofactors, or conditions, so
this module is not a chemistry validator.  It only catches obvious route
artifacts such as a product gaining many heavy atoms or carbons relative to all
listed material sources.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from rdkit import Chem, RDLogger

from cascade_planner.baselines.route_contract import RouteCandidate, RouteStepCandidate


RDLogger.DisableLog("rdApp.*")

DEFAULT_MAX_HEAVY_GAIN = 3
DEFAULT_MAX_CARBON_GAIN = 2
DEFAULT_MAX_HETERO_GAIN = 3
MAX_CONDITION_TRANSFER_REAGENT_HEAVY_ATOMS = 12
SOURCE_CRITICAL_ELEMENTS = {
    "B",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "S",
    "Si",
    "Se",
    "Sn",
}


def audit_route_plausibility(
    route: RouteCandidate,
    *,
    max_heavy_gain: int = DEFAULT_MAX_HEAVY_GAIN,
    max_carbon_gain: int = DEFAULT_MAX_CARBON_GAIN,
    max_hetero_gain: int = DEFAULT_MAX_HETERO_GAIN,
) -> dict[str, Any]:
    """Return a JSON-safe plausibility audit for one route candidate."""
    step_rows = []
    reasons: set[str] = set()
    for step in route.steps:
        step_audit = audit_step_plausibility(
            step,
            max_heavy_gain=max_heavy_gain,
            max_carbon_gain=max_carbon_gain,
            max_hetero_gain=max_hetero_gain,
        )
        if step_audit.get("reasons"):
            reasons.update(str(reason) for reason in step_audit.get("reasons") or [])
        step_rows.append(step_audit)
    return {
        "passed": bool(route.solved and route.steps and not reasons),
        "reasons": sorted(reasons),
        "steps": step_rows,
        "thresholds": {
            "max_heavy_gain": int(max_heavy_gain),
            "max_carbon_gain": int(max_carbon_gain),
            "max_hetero_gain": int(max_hetero_gain),
        },
        "contract": (
            "minimum material-sanity screen; not an atom-mapped route feasibility validator"
        ),
    }


def audit_step_plausibility(
    step: RouteStepCandidate,
    *,
    max_heavy_gain: int = DEFAULT_MAX_HEAVY_GAIN,
    max_carbon_gain: int = DEFAULT_MAX_CARBON_GAIN,
    max_hetero_gain: int = DEFAULT_MAX_HETERO_GAIN,
) -> dict[str, Any]:
    """Return a JSON-safe plausibility audit for one retrosynthetic step."""
    product_counts = element_counts(step.product_smiles)
    reactant_counts = sum_element_counts(element_counts(smi) for smi in step.reactant_smiles)
    condition_reagent_counts = condition_transfer_element_counts(step.condition_predictions)
    if not product_counts:
        return {
            "rxn_smiles": step.rxn_smiles,
            "passed": False,
            "reasons": ["invalid_product_smiles"],
        }
    if not reactant_counts:
        return {
            "rxn_smiles": step.rxn_smiles,
            "passed": False,
            "reasons": ["invalid_or_missing_reactants"],
        }

    raw_element_gains = positive_element_deltas(product_counts, reactant_counts)
    condition_supported_gains = {
        element: min(gain, int(condition_reagent_counts.get(element, 0)))
        for element, gain in raw_element_gains.items()
        if int(condition_reagent_counts.get(element, 0)) > 0
    }
    effective_reactant_counts = sum_element_counts([reactant_counts, condition_supported_gains])

    heavy_gain = heavy_atom_count(product_counts) - heavy_atom_count(effective_reactant_counts)
    carbon_gain = int(product_counts.get("C", 0)) - int(effective_reactant_counts.get("C", 0))
    hetero_gain = hetero_atom_count(product_counts) - hetero_atom_count(effective_reactant_counts)
    raw_heavy_gain = heavy_atom_count(product_counts) - heavy_atom_count(reactant_counts)
    raw_carbon_gain = int(product_counts.get("C", 0)) - int(reactant_counts.get("C", 0))
    raw_hetero_gain = hetero_atom_count(product_counts) - hetero_atom_count(reactant_counts)
    reasons = []
    unexplained_element_gains = positive_element_deltas(product_counts, effective_reactant_counts)
    unexplained_new_elements = sorted(
        element
        for element, gain in unexplained_element_gains.items()
        if gain > 0 and int(reactant_counts.get(element, 0)) == 0 and element in SOURCE_CRITICAL_ELEMENTS
    )
    if unexplained_new_elements:
        reasons.append("unexplained_new_element_source")
    if heavy_gain > int(max_heavy_gain):
        reasons.append("large_unexplained_heavy_atom_gain")
    if carbon_gain > int(max_carbon_gain):
        reasons.append("large_unexplained_carbon_gain")
    if hetero_gain > int(max_hetero_gain):
        reasons.append("large_unexplained_hetero_atom_gain")
    return {
        "rxn_smiles": step.rxn_smiles,
        "passed": not reasons,
        "reasons": reasons,
        "product_counts": product_counts,
        "reactant_counts": reactant_counts,
        "condition_reagent_counts": condition_reagent_counts,
        "raw_element_gains": raw_element_gains,
        "condition_supported_element_gains": condition_supported_gains,
        "unexplained_element_gains": unexplained_element_gains,
        "unexplained_new_elements": unexplained_new_elements,
        "raw_heavy_atom_gain": raw_heavy_gain,
        "raw_carbon_gain": raw_carbon_gain,
        "raw_hetero_atom_gain": raw_hetero_gain,
        "heavy_atom_gain": heavy_gain,
        "carbon_gain": carbon_gain,
        "hetero_atom_gain": hetero_gain,
    }


def split_plausible_routes(
    routes: Iterable[RouteCandidate],
    *,
    max_heavy_gain: int = DEFAULT_MAX_HEAVY_GAIN,
    max_carbon_gain: int = DEFAULT_MAX_CARBON_GAIN,
    max_hetero_gain: int = DEFAULT_MAX_HETERO_GAIN,
) -> tuple[list[tuple[RouteCandidate, dict[str, Any]]], list[dict[str, Any]]]:
    """Return plausible route/audit pairs and all route audits."""
    route_list = list(routes)
    audits = [
        audit_route_plausibility(
            route,
            max_heavy_gain=max_heavy_gain,
            max_carbon_gain=max_carbon_gain,
            max_hetero_gain=max_hetero_gain,
        )
        for route in route_list
    ]
    plausible = [(route, audit) for route, audit in zip(route_list, audits) if audit.get("passed")]
    return plausible, audits


def plausibility_failure_counts(audits: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for audit in audits:
        counts.update(str(reason) for reason in audit.get("reasons") or [])
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def element_counts(smiles: str) -> dict[str, int]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {}
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def condition_transfer_element_counts(condition_predictions: list[dict[str, Any]]) -> dict[str, int]:
    """Return max per-element counts available from small predicted reagents.

    Condition predictors often put true stoichiometric reagents in a separate
    ``Reagent`` field rather than in reaction SMILES.  We use that field as a
    conservative possible atom source, while ignoring solvents and bulky
    catalysts so they cannot hide material-balance artifacts.
    """
    max_counts: dict[str, int] = {}
    for prediction in condition_predictions or []:
        if not isinstance(prediction, dict):
            continue
        row_counts: dict[str, int] = {}
        for smi in _split_condition_smiles(str(prediction.get("Reagent") or "")):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            heavy = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != "H")
            if heavy > MAX_CONDITION_TRANSFER_REAGENT_HEAVY_ATOMS:
                continue
            row_counts = sum_element_counts([row_counts, element_counts(smi)])
        for element, count in row_counts.items():
            max_counts[element] = max(int(max_counts.get(element, 0)), int(count))
    return dict(sorted(max_counts.items()))


def positive_element_deltas(product_counts: dict[str, int], reactant_counts: dict[str, int]) -> dict[str, int]:
    gains: dict[str, int] = {}
    for element in sorted(set(product_counts) | set(reactant_counts)):
        if element == "H":
            continue
        delta = int(product_counts.get(element, 0)) - int(reactant_counts.get(element, 0))
        if delta > 0:
            gains[element] = delta
    return gains


def _split_condition_smiles(text: str) -> list[str]:
    parts: list[str] = []
    for item in str(text or "").replace(";", ".").split("."):
        smi = item.strip()
        if smi:
            parts.append(smi)
    return parts


def sum_element_counts(rows: Iterable[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for counts in rows:
        for key, value in counts.items():
            out[key] = out.get(key, 0) + int(value)
    return out


def heavy_atom_count(counts: dict[str, int]) -> int:
    return sum(int(value) for key, value in counts.items() if key != "H")


def hetero_atom_count(counts: dict[str, int]) -> int:
    return sum(int(value) for key, value in counts.items() if key not in {"H", "C"})
