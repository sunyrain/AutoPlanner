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
MAX_CONDITION_WARNING_REAGENT_HEAVY_ATOMS = 60
FIRST_STEP_MATERIAL_GATE_SCHEMA = "first_step_material_gate.v1"
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
FIRST_STEP_INVALID_REASONS = {"invalid_product_smiles", "invalid_or_missing_reactants"}
FIRST_STEP_TRANSFER_NAME_HINTS = {
    "boc": ("boc_or_carbonate_protecting_reagent", {"C": 5, "O": 2}),
    "boc2o": ("boc_or_carbonate_protecting_reagent", {"C": 5, "O": 2}),
    "fmoc": ("fmoc_protecting_reagent", {"C": 15, "O": 2}),
    "cbz": ("cbz_protecting_reagent", {"C": 8, "O": 2}),
    "tms": ("silyl_protecting_group_reagent", {"Si": 1, "C": 3}),
    "tbs": ("silyl_protecting_group_reagent", {"Si": 1, "C": 6}),
    "tbdms": ("silyl_protecting_group_reagent", {"Si": 1, "C": 6}),
    "tes": ("silyl_protecting_group_reagent", {"Si": 1, "C": 6}),
    "tips": ("silyl_protecting_group_reagent", {"Si": 1, "C": 9}),
    "tscl": ("sulfonylating_reagent", {"S": 1, "Cl": 1, "O": 2}),
    "mscl": ("sulfonylating_reagent", {"S": 1, "Cl": 1, "O": 2}),
    "tf2o": ("triflate_or_sulfonylating_reagent", {"S": 1, "F": 3, "O": 3}),
    "nfsi": ("fluorinating_reagent", {"F": 1}),
    "selectfluor": ("fluorinating_reagent", {"F": 1}),
    "dast": ("fluorinating_reagent", {"F": 1, "S": 1}),
    "ncs": ("chlorinating_reagent", {"Cl": 1}),
    "nbs": ("brominating_reagent", {"Br": 1}),
    "nis": ("iodinating_reagent", {"I": 1}),
}
NON_MILD_TEMPERATURE_LOW = -20.0
NON_MILD_TEMPERATURE_HIGH = 100.0


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


def audit_first_step_material_gate(
    step: RouteStepCandidate,
    *,
    max_heavy_gain: int = DEFAULT_MAX_HEAVY_GAIN,
    max_carbon_gain: int = DEFAULT_MAX_CARBON_GAIN,
    max_hetero_gain: int = DEFAULT_MAX_HETERO_GAIN,
) -> dict[str, Any]:
    """Classify a top-level disconnection without over-rejecting reagent cases.

    The normal material audit is intentionally strict and local.  A top-level
    search gate needs a second pass because real first steps can introduce
    atoms from predicted conditions, bulky transfer reagents, or temporary
    protecting/activating groups.  This function hard-rejects only obvious
    scaffold/material artifacts; explainable reagent-transfer cases are kept as
    warnings for later validation.
    """
    audit = audit_step_plausibility(
        step,
        max_heavy_gain=max_heavy_gain,
        max_carbon_gain=max_carbon_gain,
        max_hetero_gain=max_hetero_gain,
    )
    reasons = [str(reason) for reason in audit.get("reasons") or []]
    if not reasons:
        return {
            "schema_version": FIRST_STEP_MATERIAL_GATE_SCHEMA,
            "decision": "pass",
            "hard_reject": False,
            "reasons": [],
            "hard_reasons": [],
            "warnings": [],
            "possible_explanations": [],
            "material_audit": audit,
            "contract": (
                "conservative first-step material gate; pass means no obvious "
                "unexplained material growth was detected"
            ),
        }

    hard_reasons: list[str] = []
    warnings: list[str] = []
    explanations = _first_step_possible_transfer_explanations(step, audit)

    invalid = [reason for reason in reasons if reason in FIRST_STEP_INVALID_REASONS]
    hard_reasons.extend(invalid)
    residual_gains = _residual_unexplained_gains(audit, explanations)
    transfer_explained = residual_gains != dict(audit.get("unexplained_element_gains") or {})

    if not invalid and transfer_explained:
        warnings.append("possible_condition_or_transfer_reagent_atom_source")
    if not invalid:
        if _critical_element_and_carbon_growth(residual_gains):
            hard_reasons.append("unexplained_key_element_and_carbon_growth")
        if _large_unexplained_material_growth(residual_gains):
            hard_reasons.append("large_unexplained_material_growth")
        if _complex_product_core_growth(step.product_smiles, residual_gains):
            hard_reasons.append("unexplained_complex_product_core_growth")

    if hard_reasons:
        decision = "hard_reject"
    else:
        decision = "warn"
        if not warnings:
            warnings.append("material_audit_failed_but_not_hard_rejected")

    return {
        "schema_version": FIRST_STEP_MATERIAL_GATE_SCHEMA,
        "decision": decision,
        "hard_reject": bool(hard_reasons),
        "reasons": reasons,
        "hard_reasons": sorted(set(hard_reasons)),
        "warnings": sorted(set(warnings)),
        "possible_explanations": explanations,
        "residual_unexplained_element_gains": residual_gains,
        "material_audit": audit,
        "contract": (
            "conservative first-step material gate; hard_reject is reserved for "
            "obvious unexplained scaffold/material creation, while plausible "
            "condition/protecting/transfer-reagent cases remain warnings"
        ),
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


def route_condition_risk_warnings(route: dict[str, Any]) -> list[str]:
    """Return conservative route-level condition warnings for web/display gates."""
    warnings: set[str] = set()
    for step in route.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for condition in step.get("condition_predictions") or []:
            if not isinstance(condition, dict):
                continue
            temp = condition.get("Temperature")
            try:
                temp_value = float(temp)
            except (TypeError, ValueError):
                temp_value = None
            if (
                temp_value is not None
                and (temp_value < NON_MILD_TEMPERATURE_LOW or temp_value > NON_MILD_TEMPERATURE_HIGH)
            ):
                warnings.add("non_mild_predicted_temperature")
            reagent = str(condition.get("Reagent") or "")
            if "[AlH4-]" in reagent or "LiAl" in reagent:
                warnings.add("strong_hydride_reagent_predicted")
    return sorted(warnings)


def has_unsupported_biosynthetic_prenyl_terminal(route: dict[str, Any]) -> bool:
    """Flag long-prenyl terminal reactants when no enzyme/pathway evidence exists.

    This is a terminal-reactant screen, not a size screen.  Long isoprenoid
    chains and prenyl phosphates can be legitimate in metabolic routes, so they
    are allowed when the route carries EC/enzyme/pathway support.
    """
    if has_enzyme_or_pathway_support(route):
        return False
    return any(looks_like_long_prenyl_chain(smiles) for smiles in terminal_reactants_from_route(route))


def terminal_reactants_from_route(route: dict[str, Any]) -> list[str]:
    metrics = route.get("metrics") or {}
    terminals = [str(item or "") for item in metrics.get("terminal_reactants") or [] if str(item or "")]
    if terminals:
        return terminals
    out = []
    for step in route.get("steps") or []:
        if not isinstance(step, dict):
            continue
        main = str(step.get("main_reactant") or "")
        if main:
            out.append(main)
    return out


def has_enzyme_or_pathway_support(route: dict[str, Any]) -> bool:
    metrics = route.get("metrics") or {}
    enzyme_evidence = metrics.get("enzyme_evidence")
    if isinstance(enzyme_evidence, dict) and any(enzyme_evidence.values()):
        return True
    if isinstance(enzyme_evidence, list) and enzyme_evidence:
        return True
    for step in route.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("is_enzymatic"):
            return True
        if str(step.get("ec") or "").strip():
            return True
        if step.get("enzyme_uid"):
            return True
        if step.get("enzyme_ec_annotations") or []:
            return True
        evidence = step.get("evidence") or {}
        if isinstance(evidence, dict) and evidence.get("enzyme_annotation_available"):
            return True
    return False


def looks_like_long_prenyl_chain(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return False
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    if carbon_count < 20:
        return False
    if mol.GetRingInfo().NumRings() > 0:
        return False
    cc_double_bonds = 0
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if (
            begin.GetAtomicNum() == 6
            and end.GetAtomicNum() == 6
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            cc_double_bonds += 1
    if cc_double_bonds < 4:
        return False
    hetero_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    phosphorus_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15)
    return phosphorus_count > 0 or hetero_count <= 6


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


def _first_step_possible_transfer_explanations(
    step: RouteStepCandidate,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    gains = dict(audit.get("unexplained_element_gains") or {})
    if not gains:
        return []
    rows = []
    for source, value in _candidate_condition_or_reagent_values(step):
        for reagent in _split_reagent_text(value):
            explanation = _transfer_explanation_for_reagent(reagent, gains, source=source)
            if explanation:
                rows.append(explanation)
    return _dedupe_explanations(rows)


def _candidate_condition_or_reagent_values(step: RouteStepCandidate) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in step.reactant_smiles or []:
        if item:
            rows.append(("reactant", str(item)))
    for prediction in step.condition_predictions or []:
        if not isinstance(prediction, dict):
            continue
        for key in ("Reagent", "reagent", "Catalyst", "catalyst"):
            value = str(prediction.get(key) or "")
            if value:
                rows.append((key.lower(), value))
    return rows


def _split_reagent_text(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = _split_condition_smiles(text)
    if len(parts) > 1:
        return parts
    return [text]


def _transfer_explanation_for_reagent(
    reagent: str,
    gains: dict[str, int],
    *,
    source: str,
) -> dict[str, Any] | None:
    reagent_text = str(reagent or "").strip()
    if not reagent_text:
        return None
    lower = reagent_text.lower().replace("-", "").replace("_", "")
    for hint, (role, payload) in FIRST_STEP_TRANSFER_NAME_HINTS.items():
        if hint in lower and _counts_cover_any_gain(payload, gains):
            return {
                "source": source,
                "smiles_or_name": reagent_text,
                "role": role,
                "matched_by": "name_hint",
                "supported_element_gains": _supported_gains(payload, gains),
            }

    mol = Chem.MolFromSmiles(reagent_text)
    if mol is None:
        return None
    counts = element_counts(reagent_text)
    heavy = heavy_atom_count(counts)
    if heavy > MAX_CONDITION_WARNING_REAGENT_HEAVY_ATOMS:
        return None
    role = _structural_transfer_role(reagent_text, counts)
    if not role or not _counts_cover_any_gain(counts, gains):
        return None
    if source == "reactant" and role == "organic_group_transfer_or_protecting_reagent":
        # Ordinary carbon-containing reactants are already counted as material
        # sources.  Only named/structured transfer reagents should explain a
        # residual first-step material gap.
        return None
    return {
        "source": source,
        "smiles_or_name": reagent_text,
        "role": role,
        "matched_by": "structure",
        "heavy_atoms": heavy,
        "supported_element_gains": _supported_gains(counts, gains),
    }


def _structural_transfer_role(smiles: str, counts: dict[str, int]) -> str:
    text = str(smiles or "")
    if "Si" in counts:
        return "silyl_protecting_or_transfer_reagent"
    if "B" in counts and "F" in counts:
        return "fluoroborate_or_boron_transfer_reagent"
    if "S" in counts and int(counts.get("O", 0)) >= 2:
        return "sulfonyl_or_sulfate_transfer_reagent"
    if "P" in counts:
        return "phosphorus_transfer_or_carrier_reagent"
    if any(element in counts for element in ("F", "Cl", "Br", "I")):
        return "halogenating_or_halogen_transfer_reagent"
    if "[C]" in text or "C=" in text or int(counts.get("C", 0)) > 0:
        return "organic_group_transfer_or_protecting_reagent"
    return ""


def _counts_cover_any_gain(counts: dict[str, int], gains: dict[str, int]) -> bool:
    return any(int(counts.get(element, 0)) > 0 for element in gains)


def _supported_gains(counts: dict[str, int], gains: dict[str, int]) -> dict[str, int]:
    return {
        element: min(int(gain), int(counts.get(element, 0)))
        for element, gain in sorted(gains.items())
        if int(counts.get(element, 0)) > 0
    }


def _dedupe_explanations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = (
            str(row.get("source") or ""),
            str(row.get("smiles_or_name") or ""),
            str(row.get("role") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _residual_unexplained_gains(
    audit: dict[str, Any],
    explanations: list[dict[str, Any]],
) -> dict[str, int]:
    residual = {
        str(element): int(gain)
        for element, gain in (audit.get("unexplained_element_gains") or {}).items()
        if int(gain) > 0
    }
    for explanation in explanations:
        for element, supported in (explanation.get("supported_element_gains") or {}).items():
            if element not in residual:
                continue
            residual[element] = max(0, int(residual[element]) - int(supported))
    return {element: gain for element, gain in sorted(residual.items()) if gain > 0}


def _critical_element_and_carbon_growth(unexplained: dict[str, int]) -> bool:
    critical = [element for element in unexplained if element in SOURCE_CRITICAL_ELEMENTS]
    return bool(critical and int(unexplained.get("C") or 0) > DEFAULT_MAX_CARBON_GAIN)


def _large_unexplained_material_growth(unexplained: dict[str, int]) -> bool:
    heavy_gain = sum(int(value) for value in unexplained.values())
    carbon_gain = int(unexplained.get("C") or 0)
    hetero_gain = sum(int(value) for element, value in unexplained.items() if element not in {"H", "C"})
    return carbon_gain >= 4 or heavy_gain >= 8 or (carbon_gain >= 3 and hetero_gain >= 3)


def _complex_product_core_growth(product_smiles: str, unexplained: dict[str, int]) -> bool:
    if not _large_unexplained_material_growth(unexplained):
        return False
    mol = Chem.MolFromSmiles(str(product_smiles or ""))
    if mol is None:
        return False
    rings = int(mol.GetRingInfo().NumRings())
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    return rings >= 3 or chiral_centers >= 3


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
