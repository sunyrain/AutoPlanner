"""Fail-closed invariants for ChemEnzy route normalization."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def proposal_product(step: Mapping[str, Any]) -> str:
    """Read the selected provider product without changing its chemistry."""

    return str(step.get("product_smiles") or step.get("product") or "").strip()


def proposal_reactants(step: Mapping[str, Any]) -> list[str]:
    """Read provider reactants while retaining order and duplicate entries."""

    values = step.get("reactant_smiles") or step.get("precursor_smiles")
    if isinstance(values, str):
        values = values.split(".")
    if not isinstance(values, (list, tuple)):
        main = str(
            step.get("main_reactant")
            or step.get("main_reactant_smiles")
            or ""
        ).strip()
        auxiliary = step.get("aux_reactants") or step.get(
            "aux_reactant_smiles"
        ) or []
        if isinstance(auxiliary, str):
            auxiliary = auxiliary.split(".")
        values = ([main] if main else []) + list(auxiliary or [])
    return [str(value).strip() for value in values if str(value).strip()]


def audit_chemenzy_route_normalization(
    raw_route: Mapping[str, Any],
    normalized_steps: Iterable[Mapping[str, Any]],
    *,
    target_smiles: str = "",
) -> dict[str, Any]:
    """Prove that normalization did not alter or disconnect provider steps."""

    raw_steps = raw_route.get("steps")
    raw_rows = list(raw_steps) if isinstance(raw_steps, list) else []
    normalized_rows = [dict(step) for step in normalized_steps]
    normalized_by_index = {
        int(step.get("step_index") or 0): step for step in normalized_rows
    }
    reasons: set[str] = set()
    invalid_indices = [
        index
        for index, step in enumerate(raw_rows, start=1)
        if not isinstance(step, Mapping)
    ]
    if invalid_indices:
        reasons.add("invalid_step_payload")
    if len(raw_rows) != len(normalized_rows):
        reasons.add("raw_normalized_step_count_mismatch")

    step_audits = []
    for step_index, raw_step in enumerate(raw_rows, start=1):
        if not isinstance(raw_step, Mapping):
            continue
        normalized = normalized_by_index.get(step_index)
        step_audit = _audit_step(raw_step, normalized, step_index=step_index)
        step_audits.append(step_audit)
        reasons.update(step_audit["reasons"])

    connectivity = _route_connectivity(normalized_rows, target_smiles=target_smiles)
    if not connectivity["target_identity_preserved"]:
        reasons.add("route_target_product_mismatch")
    if not connectivity["route_step_connectivity_preserved"]:
        reasons.add("disconnected_route_steps")
    payload = {
        "schema_version": "chemenzy_route_normalization_audit.v1",
        "accepted": not reasons,
        "raw_step_count": len(raw_rows),
        "normalized_step_count": len(normalized_rows),
        "invalid_raw_step_indices": invalid_indices,
        "step_count_preserved": len(raw_rows) == len(normalized_rows),
        "product_identity_preserved": all(
            row["product_identity_preserved"] for row in step_audits
        ),
        "reactant_order_and_multiplicity_preserved": all(
            row["reactant_order_and_multiplicity_preserved"]
            for row in step_audits
        ),
        "reaction_direction_consistent": all(
            row["reaction_direction_consistent"] is not False
            for row in step_audits
        ),
        "stereochemical_identity_preserved": all(
            row["stereochemical_identity_preserved"] for row in step_audits
        ),
        **connectivity,
        "reasons": sorted(reasons),
        "step_audits": step_audits,
        "semantics": {
            "provider_strings_are_not_canonicalized_in_normalized_output": True,
            "precursor_order_and_multiplicity_are_preserved": True,
            "reaction_smiles_is_advisory_but_must_not_contradict_fields": True,
            "normalization_audit_grants_no_reaction_proof": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _audit_step(
    raw_step: Mapping[str, Any],
    normalized: Mapping[str, Any] | None,
    *,
    step_index: int,
) -> dict[str, Any]:
    expected_product = proposal_product(raw_step)
    expected_reactants = proposal_reactants(raw_step)
    actual_product = str((normalized or {}).get("product_smiles") or "")
    actual_reactants = [
        str(value) for value in (normalized or {}).get("reactant_smiles") or []
    ]
    raw_reaction = str(
        raw_step.get("rxn_smiles") or raw_step.get("reaction_smiles") or ""
    )
    actual_reaction = str((normalized or {}).get("rxn_smiles") or "")
    reasons: set[str] = set()
    product_preserved = expected_product == actual_product
    reactants_preserved = expected_reactants == actual_reactants
    if normalized is None:
        reasons.add("normalized_step_missing")
    if not product_preserved:
        reasons.add("product_payload_not_preserved")
    if not reactants_preserved:
        reasons.add("reactant_order_or_multiplicity_not_preserved")
    if raw_reaction != actual_reaction:
        reasons.add("reaction_smiles_payload_not_preserved")

    reaction_audit = _reaction_consistency(
        raw_reaction,
        product_smiles=actual_product,
        reactant_smiles=actual_reactants,
    )
    reasons.update(reaction_audit["reasons"])
    stereo_preserved = not any("stereochemistry" in reason for reason in reasons)
    return {
        "step_index": step_index,
        "accepted": not reasons,
        "product_identity_preserved": product_preserved,
        "reactant_order_and_multiplicity_preserved": reactants_preserved,
        "reaction_smiles_payload_preserved": raw_reaction == actual_reaction,
        "reaction_direction_consistent": reaction_audit["direction_consistent"],
        "stereochemical_identity_preserved": stereo_preserved,
        "reasons": sorted(reasons),
    }


def _reaction_consistency(
    reaction: str,
    *,
    product_smiles: str,
    reactant_smiles: list[str],
) -> dict[str, Any]:
    if not reaction:
        return {"direction_consistent": None, "reasons": []}
    parts = reaction.split(">")
    if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
        return {
            "direction_consistent": False,
            "reasons": ["invalid_reaction_smiles_payload"],
        }
    lhs, rhs = parts[0], parts[2]
    observed_left = _component_counter([lhs], isomeric=True)
    observed_right = _component_counter([rhs], isomeric=True)
    expected_left = _component_counter(reactant_smiles, isomeric=True)
    expected_right = _component_counter([product_smiles], isomeric=True)
    if any(
        value is None
        for value in (observed_left, observed_right, expected_left, expected_right)
    ):
        return {
            "direction_consistent": False,
            "reasons": ["invalid_reaction_smiles_payload"],
        }
    if observed_left == expected_right and observed_right == expected_left:
        return {
            "direction_consistent": False,
            "reasons": ["reaction_smiles_direction_mismatch"],
        }
    reasons = []
    if observed_left != expected_left:
        reasons.append(
            _side_mismatch_reason(lhs, reactant_smiles, side="reactant")
        )
    if observed_right != expected_right:
        reasons.append(_side_mismatch_reason(rhs, [product_smiles], side="product"))
    return {"direction_consistent": not reasons, "reasons": sorted(set(reasons))}


def _side_mismatch_reason(
    observed: str,
    expected: list[str],
    *,
    side: str,
) -> str:
    if _component_counter([observed], isomeric=False) == _component_counter(
        expected, isomeric=False
    ):
        return "reaction_smiles_stereochemistry_mismatch"
    return f"reaction_smiles_{side}_mismatch"


def _route_connectivity(
    steps: list[dict[str, Any]], *, target_smiles: str
) -> dict[str, bool]:
    products = [_canonical(str(step.get("product_smiles") or "")) for step in steps]
    reactants = [
        [_canonical(str(value)) for value in step.get("reactant_smiles") or []]
        for step in steps
    ]
    valid = bool(steps) and all(products) and all(all(row) for row in reactants)
    target = _canonical(target_smiles) if target_smiles else ""
    product_set = set(products)
    if target:
        roots = {target}
        target_preserved = target in product_set
    else:
        roots = product_set - {value for row in reactants for value in row}
        target_preserved = len(roots) == 1
    frontier = set(roots)
    remaining = set(range(len(steps)))
    progressed = True
    while valid and progressed:
        progressed = False
        for index in list(remaining):
            if products[index] not in frontier:
                continue
            frontier.discard(products[index])
            frontier.update(reactants[index])
            remaining.remove(index)
            progressed = True
    connected = valid and target_preserved and not remaining and not (frontier & product_set)
    return {
        "target_identity_preserved": target_preserved,
        "route_step_connectivity_preserved": connected,
    }


def _component_counter(
    values: Iterable[str], *, isomeric: bool
) -> Counter[str] | None:
    components = [
        component.strip()
        for value in values
        for component in str(value).split(".")
        if component.strip()
    ]
    canonical = [_canonical(value, isomeric=isomeric) for value in components]
    return Counter(canonical) if components and all(canonical) else None


def _canonical(value: str, *, isomeric: bool = True) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=isomeric)


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "audit_chemenzy_route_normalization",
    "proposal_product",
    "proposal_reactants",
]
