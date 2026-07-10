"""Deterministic, replayable verification for one materialized reaction step.

The route verifier historically proved graph connectivity and stock closure.
Those are necessary route properties, but they do not establish that each
reaction edge is chemically materialized.  This module keeps the two claims
separate and emits an immutable proof record for the reaction edge itself.

The proof levels are deliberately monotonic:

``L0`` materialized structures
``L1`` graph and stock closed (assigned by the route verifier)
``L2_mapping_consistent`` complete atom-mapped reaction is internally
consistent, but remains advisory
``L2_reaction_validated`` a trusted deterministic transform was reapplied and
its reaction centre matched (reserved until such a replayer is configured)
``L3`` mapping consistency plus an exact precedent revalidated against the
trusted registry and materialized source evidence
``L4`` L3 plus complete conditions and procurement evidence

No boolean supplied inside a candidate step is treated as authority.  Every L2
claim is recomputed from the mapped reaction and materialized structures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

REACTION_STEP_PROOF_SCHEMA = "reaction_step_proof.v1"
REACTION_ROUTE_PROOF_SCHEMA = "reaction_route_validation.v1"
REACTION_STEP_VERIFIER_VERSION = "autoplanner.reaction_step_verifier.v1"

PROOF_LEVEL_ORDER = {
    "L0_materialized": 0,
    "L1_graph_and_stock_closed": 1,
    "L2_mapping_consistent": 2,
    "L2_reaction_validated": 3,
    "L3_precedent_supported": 4,
    "L4_procurement_ready": 5,
}


@dataclass(frozen=True)
class ReactionStepProof:
    step_id: str
    step_index: int
    product_smiles: str
    reactant_smiles: tuple[str, ...]
    proof_level: str
    accepted: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...] = field(default_factory=tuple)
    mapping_source: str = ""
    atom_map_audit: dict[str, Any] = field(default_factory=dict)
    bond_change_audit: dict[str, Any] = field(default_factory=dict)
    trusted_precedent_binding: dict[str, Any] = field(default_factory=dict)
    procurement_binding: dict[str, Any] = field(default_factory=dict)
    reaction_digest: str = ""
    input_digest: str = ""
    proof_digest: str = ""
    validator_version: str = REACTION_STEP_VERIFIER_VERSION
    schema_version: str = REACTION_STEP_PROOF_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_reaction_step(
    step: Mapping[str, Any],
    *,
    step_index: int = 0,
    graph_and_stock_closed: bool = False,
    trusted_precedent_binding: Mapping[str, Any] | None = None,
    procurement_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize and independently validate one reaction edge."""
    raw = dict(step or {})
    product_raw = _step_product(raw)
    reactants_raw = _step_reactants(raw)
    product = _canonical_smiles(product_raw)
    reactants = tuple(
        value
        for value in (_canonical_smiles(item) for item in reactants_raw)
        if value
    )
    reasons: list[str] = []
    checks = {
        "structures_materialized": bool(product and reactants and len(reactants) == len(reactants_raw)),
        "graph_and_stock_closed": bool(graph_and_stock_closed),
        "mapped_reaction_present": False,
        "mapped_product_matches": False,
        "mapped_reactants_match": False,
        "atom_maps_complete": False,
        "atom_maps_unique": False,
        "product_atoms_have_reactant_provenance": False,
        "mapped_elements_preserved": False,
        "mapped_reactant_components_contribute": False,
        "scaffold_continuity_plausible": False,
        "ring_change_plausible": False,
        "bond_change_present": False,
        "reaction_edit_budget_plausible": False,
        "stereochemical_product_matches": False,
        "trusted_precedent_bound": False,
        "conditions_complete": False,
        "procurement_bound": False,
    }
    if not product:
        reasons.append("invalid_or_missing_product_smiles")
    if not reactants_raw:
        reasons.append("missing_reactant_smiles")
    elif len(reactants) != len(reactants_raw):
        reasons.append("invalid_reactant_smiles")

    mapped_reaction, mapping_source = _mapped_reaction_from_step(raw)
    checks["mapped_reaction_present"] = bool(mapped_reaction)
    atom_audit: dict[str, Any] = {}
    bond_audit: dict[str, Any] = {}
    if checks["structures_materialized"] and mapped_reaction:
        atom_audit, bond_audit = _audit_mapped_reaction(
            mapped_reaction,
            expected_product=product,
            expected_reactants=reactants,
        )
        for key in (
            "mapped_product_matches",
            "mapped_reactants_match",
            "atom_maps_complete",
            "atom_maps_unique",
            "product_atoms_have_reactant_provenance",
            "mapped_elements_preserved",
            "mapped_reactant_components_contribute",
            "scaffold_continuity_plausible",
            "ring_change_plausible",
            "stereochemical_product_matches",
        ):
            checks[key] = bool(atom_audit.get(key))
        checks["bond_change_present"] = bool(bond_audit.get("bond_change_present"))
        checks["reaction_edit_budget_plausible"] = bool(
            bond_audit.get("reaction_edit_budget_plausible")
        )
        reasons.extend(str(value) for value in atom_audit.get("reasons") or [])
        reasons.extend(str(value) for value in bond_audit.get("reasons") or [])
    elif checks["structures_materialized"]:
        reasons.append("complete_atom_mapped_reaction_missing")

    reaction_digest = canonical_reaction_digest(product, reactants)
    provided_precedent = dict(trusted_precedent_binding or {})
    derived_precedent = _trusted_precedent_from_step(raw)
    precedent = derived_precedent
    if provided_precedent and not _same_trusted_precedent(
        provided_precedent,
        derived_precedent,
    ):
        reasons.append("supplied_precedent_binding_not_revalidated_from_trusted_evidence")
        precedent = {}
    procurement = dict(procurement_binding or {})
    checks["trusted_precedent_bound"] = _trusted_binding(
        precedent,
        expected_reaction_digest=reaction_digest,
    )
    checks["conditions_complete"] = _conditions_complete(raw)
    checks["procurement_bound"] = _procurement_binding(
        procurement,
        expected_reactants=reactants,
    )

    l2_checks = (
        "structures_materialized",
        "mapped_reaction_present",
        "mapped_product_matches",
        "mapped_reactants_match",
        "atom_maps_complete",
        "atom_maps_unique",
        "product_atoms_have_reactant_provenance",
        "mapped_elements_preserved",
        "mapped_reactant_components_contribute",
        "scaffold_continuity_plausible",
        "ring_change_plausible",
        "bond_change_present",
        "reaction_edit_budget_plausible",
        "stereochemical_product_matches",
    )
    mapping_consistent = all(checks[key] for key in l2_checks)
    # Atom-map consistency alone cannot establish a chemically meaningful
    # transform: a cut/glue construction can conserve atoms and maps while
    # inventing an implausible reaction.  Until a deterministic template
    # reapply + reaction-centre matcher is wired here, exact curated precedent
    # is the only authority that upgrades the edge beyond advisory L2.
    reaction_validated = bool(mapping_consistent and checks["trusted_precedent_bound"])
    if reaction_validated:
        proof_level = "L3_precedent_supported"
    elif mapping_consistent:
        proof_level = "L2_mapping_consistent"
        reasons.append("mapping_consistent_without_trusted_transform_or_precedent")
    elif checks["structures_materialized"] and checks["graph_and_stock_closed"]:
        proof_level = "L1_graph_and_stock_closed"
    elif checks["structures_materialized"]:
        proof_level = "L0_materialized"
    else:
        proof_level = "L0_materialized"
    if (
        proof_level == "L3_precedent_supported"
        and checks["conditions_complete"]
        and checks["procurement_bound"]
    ):
        proof_level = "L4_procurement_ready"

    input_payload = {
        "product_smiles": product,
        "reactant_smiles": sorted(reactants),
        "mapped_reaction": mapped_reaction,
        "mapping_source": mapping_source,
        "graph_and_stock_closed": bool(graph_and_stock_closed),
        "trusted_precedent_binding": precedent,
        "procurement_binding": procurement,
        "reaction_digest": reaction_digest,
    }
    input_digest = _digest(input_payload)
    proof_payload = {
        "schema_version": REACTION_STEP_PROOF_SCHEMA,
        "step_id": str(raw.get("step_id") or raw.get("id") or f"step:{step_index}"),
        "step_index": int(raw.get("index") if raw.get("index") is not None else step_index),
        "product_smiles": product,
        "reactant_smiles": list(reactants),
        "proof_level": proof_level,
        "accepted": reaction_validated,
        "checks": checks,
        "reasons": sorted(set(reasons)),
        "mapping_source": mapping_source,
        "atom_map_audit": atom_audit,
        "bond_change_audit": bond_audit,
        "trusted_precedent_binding": precedent,
        "procurement_binding": procurement,
        "reaction_digest": reaction_digest,
        "input_digest": input_digest,
        "validator_version": REACTION_STEP_VERIFIER_VERSION,
    }
    proof_payload["proof_digest"] = _digest(proof_payload)
    return ReactionStepProof(
        step_id=proof_payload["step_id"],
        step_index=proof_payload["step_index"],
        product_smiles=product,
        reactant_smiles=reactants,
        proof_level=proof_level,
        accepted=reaction_validated,
        checks=checks,
        reasons=tuple(proof_payload["reasons"]),
        mapping_source=mapping_source,
        atom_map_audit=atom_audit,
        bond_change_audit=bond_audit,
        trusted_precedent_binding=precedent,
        procurement_binding=procurement,
        reaction_digest=reaction_digest,
        input_digest=input_digest,
        proof_digest=proof_payload["proof_digest"],
    ).to_dict()


def verify_reaction_route(
    steps: Iterable[Mapping[str, Any]],
    *,
    graph_and_stock_closed: bool,
    trusted_precedent_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    procurement_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the weakest-link proof summary for a materialized route."""
    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    precedents = dict(trusted_precedent_bindings or {})
    procurement = dict(procurement_bindings or {})
    proofs: list[dict[str, Any]] = []
    for index, step in enumerate(rows):
        step_id = str(step.get("step_id") or step.get("id") or f"step:{index}")
        proofs.append(
            verify_reaction_step(
                step,
                step_index=index,
                graph_and_stock_closed=graph_and_stock_closed,
                trusted_precedent_binding=precedents.get(step_id),
                procurement_binding=procurement.get(step_id),
            )
        )
    weakest_level = min(
        (str(row.get("proof_level") or "L0_materialized") for row in proofs),
        key=lambda value: PROOF_LEVEL_ORDER.get(value, -1),
        default="L0_materialized",
    )
    validated_count = sum(1 for row in proofs if row.get("accepted") is True)
    payload = {
        "schema_version": REACTION_ROUTE_PROOF_SCHEMA,
        "accepted": bool(proofs) and validated_count == len(proofs),
        "proof_level": weakest_level,
        "weakest_link_policy": True,
        "step_count": len(proofs),
        "reaction_validated_step_count": validated_count,
        "step_proofs": proofs,
        "validator_version": REACTION_STEP_VERIFIER_VERSION,
    }
    payload["proof_digest"] = _digest(payload)
    return payload


def is_reaction_validated_route(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    row = dict(value)
    proofs = row.get("step_proofs")
    if row.get("schema_version") != REACTION_ROUTE_PROOF_SCHEMA or not isinstance(proofs, list):
        return False
    try:
        step_count = int(row.get("step_count") or 0)
        validated_count = int(row.get("reaction_validated_step_count") or 0)
    except (TypeError, ValueError):
        return False
    expected_digest = str(row.pop("proof_digest", ""))
    return bool(
        expected_digest
        and expected_digest == _digest(row)
        and row.get("accepted") is True
        and step_count > 0
        and validated_count == step_count == len(proofs)
        and all(
            isinstance(proof, Mapping)
            and proof.get("accepted") is True
            and str(proof.get("proof_level") or "")
            in {"L2_reaction_validated", "L3_precedent_supported", "L4_procurement_ready"}
            for proof in proofs
        )
    )


def is_precedent_supported_route(value: Any) -> bool:
    """Require a digest-valid route whose every step has revalidated precedent."""
    if not is_reaction_validated_route(value):
        return False
    proofs = list(dict(value).get("step_proofs") or [])
    return bool(
        proofs
        and all(
            str(proof.get("proof_level") or "")
            in {"L3_precedent_supported", "L4_procurement_ready"}
            and proof.get("checks", {}).get("trusted_precedent_bound") is True
            for proof in proofs
            if isinstance(proof, Mapping)
        )
        and len(proofs)
        == sum(1 for proof in proofs if isinstance(proof, Mapping))
    )


def _audit_mapped_reaction(
    reaction_smiles: str,
    *,
    expected_product: str,
    expected_reactants: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasons: list[str] = []
    parts = str(reaction_smiles or "").split(">")
    if len(parts) != 3:
        return (
            {"reasons": ["invalid_atom_mapped_reaction_shape"]},
            {"bond_change_present": False, "reasons": ["invalid_atom_mapped_reaction_shape"]},
        )
    reactant_text, _, product_text = parts
    reactant_mols = [Chem.MolFromSmiles(value) for value in reactant_text.split(".") if value]
    product_mols = [Chem.MolFromSmiles(value) for value in product_text.split(".") if value]
    if not reactant_mols or any(mol is None for mol in reactant_mols):
        reasons.append("invalid_mapped_reactants")
    if len(product_mols) != 1 or product_mols[0] is None:
        reasons.append("mapped_product_must_be_single_valid_molecule")
    if reasons:
        return (
            {"reasons": sorted(set(reasons))},
            {"bond_change_present": False, "reasons": ["mapped_reaction_not_materialized"]},
        )

    product_mol = product_mols[0]
    mapped_reactants = tuple(sorted(_canonical_without_maps(mol) for mol in reactant_mols))
    mapped_product = _canonical_without_maps(product_mol)
    expected_reactant_multiset = tuple(sorted(expected_reactants))
    mapped_product_matches = mapped_product == expected_product
    mapped_reactants_match = mapped_reactants == expected_reactant_multiset
    if not mapped_product_matches:
        reasons.append("mapped_product_does_not_match_step_product")
    if not mapped_reactants_match:
        reasons.append("mapped_reactants_do_not_match_step_reactants")

    reactant_atoms, duplicate_reactant_maps, reactant_unmapped = _mapped_atoms(reactant_mols)
    product_atoms, duplicate_product_maps, product_unmapped = _mapped_atoms([product_mol])
    atom_maps_complete = not reactant_unmapped and not product_unmapped
    atom_maps_unique = not duplicate_reactant_maps and not duplicate_product_maps
    new_product_maps = sorted(set(product_atoms) - set(reactant_atoms))
    element_mismatches = sorted(
        map_num
        for map_num in set(product_atoms) & set(reactant_atoms)
        if product_atoms[map_num] != reactant_atoms[map_num]
    )
    provenance = not new_product_maps
    elements_preserved = not element_mismatches
    if not atom_maps_complete:
        reasons.append("atom_mapping_incomplete")
    if not atom_maps_unique:
        reasons.append("atom_mapping_not_unique")
    if not provenance:
        reasons.append("product_heavy_atom_without_reactant_provenance")
    if not elements_preserved:
        reasons.append("mapped_atom_element_changed")

    product_map_set = set(product_atoms)
    component_map_sets = [
        {
            int(atom.GetAtomMapNum())
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() > 1 and int(atom.GetAtomMapNum()) > 0
        }
        for mol in reactant_mols
    ]
    contributing_components = sum(
        1 for component_maps in component_map_sets if component_maps & product_map_set
    )
    components_contribute = bool(component_map_sets) and contributing_components == len(
        component_map_sets
    )
    if not components_contribute:
        reasons.append("mapped_reactant_component_does_not_contribute_to_product")

    product_heavy_atoms = int(product_mol.GetNumHeavyAtoms())
    largest_reactant_heavy_atoms = max(
        (int(mol.GetNumHeavyAtoms()) for mol in reactant_mols),
        default=0,
    )
    scaffold_fraction = (
        min(largest_reactant_heavy_atoms, product_heavy_atoms) / product_heavy_atoms
        if product_heavy_atoms
        else 0.0
    )
    scaffold_plausible = bool(
        product_heavy_atoms <= 11 or scaffold_fraction >= 0.25
    )
    if not scaffold_plausible:
        reasons.append("reaction_lacks_continuous_precursor_scaffold")

    reactant_ring_count = sum(int(mol.GetRingInfo().NumRings()) for mol in reactant_mols)
    product_ring_count = int(product_mol.GetRingInfo().NumRings())
    net_ring_increase = product_ring_count - reactant_ring_count
    ring_change_plausible = net_ring_increase <= 2
    if not ring_change_plausible:
        reasons.append("reaction_creates_too_many_rings_in_one_step")

    stereo_matches = mapped_product_matches
    if not stereo_matches:
        reasons.append("stereochemical_product_mismatch")
    atom_audit = {
        "mapped_product_matches": mapped_product_matches,
        "mapped_reactants_match": mapped_reactants_match,
        "atom_maps_complete": atom_maps_complete,
        "atom_maps_unique": atom_maps_unique,
        "product_atoms_have_reactant_provenance": provenance,
        "mapped_elements_preserved": elements_preserved,
        "mapped_reactant_components_contribute": components_contribute,
        "contributing_reactant_component_count": contributing_components,
        "mapped_reactant_component_count": len(component_map_sets),
        "scaffold_continuity_plausible": scaffold_plausible,
        "largest_reactant_heavy_atom_count": largest_reactant_heavy_atoms,
        "product_heavy_atom_count": product_heavy_atoms,
        "largest_reactant_product_fraction": round(scaffold_fraction, 6),
        "ring_change_plausible": ring_change_plausible,
        "reactant_ring_count": reactant_ring_count,
        "product_ring_count": product_ring_count,
        "net_ring_increase": net_ring_increase,
        "stereochemical_product_matches": stereo_matches,
        "reactant_heavy_atom_map_count": len(reactant_atoms),
        "product_heavy_atom_map_count": len(product_atoms),
        "unmapped_reactant_heavy_atom_count": reactant_unmapped,
        "unmapped_product_heavy_atom_count": product_unmapped,
        "duplicate_reactant_atom_maps": sorted(duplicate_reactant_maps),
        "duplicate_product_atom_maps": sorted(duplicate_product_maps),
        "new_product_atom_maps": new_product_maps,
        "element_mismatch_atom_maps": element_mismatches,
        "reasons": sorted(set(reasons)),
    }

    reactant_bonds = _mapped_bonds(reactant_mols)
    product_bonds = _mapped_bonds([product_mol])
    formed = sorted(product_bonds - reactant_bonds)
    broken = sorted(reactant_bonds - product_bonds)
    bond_change_present = bool(formed or broken)
    edit_count = len(formed) + len(broken)
    edit_budget_plausible = edit_count <= 8
    bond_reasons = [] if bond_change_present else ["mapped_reaction_has_no_bond_change"]
    if not edit_budget_plausible:
        bond_reasons.append("reaction_edit_budget_exceeded")
    bond_audit = {
        "bond_change_present": bond_change_present,
        "reaction_edit_budget_plausible": edit_budget_plausible,
        "bond_edit_count": edit_count,
        "max_bond_edit_count": 8,
        "formed_or_changed_bonds": [list(row) for row in formed],
        "broken_or_changed_bonds": [list(row) for row in broken],
        "reasons": bond_reasons,
    }
    return atom_audit, bond_audit


def _mapped_atoms(mols: Iterable[Any]) -> tuple[dict[int, int], set[int], int]:
    atoms: dict[int, int] = {}
    duplicates: set[int] = set()
    unmapped = 0
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            map_num = int(atom.GetAtomMapNum())
            if map_num <= 0:
                unmapped += 1
                continue
            if map_num in atoms:
                duplicates.add(map_num)
            else:
                atoms[map_num] = int(atom.GetAtomicNum())
    return atoms, duplicates, unmapped


def _mapped_bonds(mols: Iterable[Any]) -> set[tuple[int, int, str]]:
    rows: set[tuple[int, int, str]] = set()
    for mol in mols:
        for bond in mol.GetBonds():
            left = int(bond.GetBeginAtom().GetAtomMapNum())
            right = int(bond.GetEndAtom().GetAtomMapNum())
            if left <= 0 or right <= 0:
                continue
            rows.add((min(left, right), max(left, right), str(bond.GetBondType())))
    return rows


def _mapped_reaction_from_step(step: Mapping[str, Any]) -> tuple[str, str]:
    for field_name in ("atom_mapped_reaction_smiles", "mapped_reaction_smiles"):
        value = str(step.get(field_name) or "").strip()
        if value:
            return value, field_name
    reaction = str(step.get("reaction_smiles") or "").strip()
    if reaction and ":" in reaction:
        return reaction, "reaction_smiles"
    provenance = step.get("atom_provenance")
    if isinstance(provenance, Mapping):
        for field_name in ("atom_mapped_reaction_smiles", "mapped_reaction_smiles", "reaction_smiles"):
            value = str(provenance.get(field_name) or "").strip()
            if value and ":" in value:
                return value, f"atom_provenance.{field_name}"
    return "", ""


def _step_product(step: Mapping[str, Any]) -> str:
    for field_name in ("product_smiles", "product", "products", "target_smiles"):
        value = step.get(field_name)
        if isinstance(value, list):
            if len(value) == 1:
                return str(value[0] or "")
            continue
        if str(value or "").strip():
            return str(value)
    return ""


def _step_reactants(step: Mapping[str, Any]) -> list[str]:
    # Select exactly one compatible representation.  Concatenating legacy
    # aliases can duplicate the same material and corrupt element accounting.
    for field_name in ("reactant_smiles", "precursor_smiles", "reactants"):
        raw = step.get(field_name)
        if not raw:
            continue
        if isinstance(raw, str):
            return [part for part in raw.split(".") if part]
        if isinstance(raw, (list, tuple)):
            return [str(value) for value in raw if str(value or "").strip()]
    values = [str(step.get("main_reactant") or step.get("main_reactant_smiles") or "")]
    aux = step.get("aux_reactants") or []
    if isinstance(aux, str):
        values.extend(part for part in aux.split(".") if part)
    elif isinstance(aux, (list, tuple)):
        values.extend(str(value) for value in aux if str(value or "").strip())
    return [value for value in values if value]


def _canonical_smiles(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def _canonical_without_maps(mol: Any) -> str:
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def canonical_reaction_digest(product: Any, reactants: Iterable[Any]) -> str:
    """Hash the canonical reaction edge used by trusted precedent registries."""
    canonical_product = _canonical_smiles(product)
    canonical_reactants = sorted(
        value for value in (_canonical_smiles(item) for item in reactants) if value
    )
    if not canonical_product or not canonical_reactants:
        return ""
    return _digest(
        {
            "product_canonical_isomeric_smiles": canonical_product,
            "reactant_canonical_isomeric_smiles": canonical_reactants,
        }
    )


def _trusted_precedent_from_step(step: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild precedent authority from configured registry + source artifacts.

    The import is deliberately local: the literature stitcher consumes this
    verifier, while this narrow proof upgrade reuses its strict source gate.
    No binding carried by a public request can bypass this reconstruction.
    """
    if not step.get("source_detail_exact_step"):
        return {}
    try:
        from cascade_planner.harness.stitched_route import (
            _trusted_literature_step_precedent,
            is_validated_source_detail_literature_step,
        )
    except ImportError:
        return {}
    row = dict(step)
    if not is_validated_source_detail_literature_step(row):
        return {}
    return dict(_trusted_literature_step_precedent(row) or {})


def _same_trusted_precedent(
    supplied: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> bool:
    if not supplied or not derived:
        return False
    fields = (
        "schema_version",
        "accepted",
        "authority",
        "authority_id",
        "binding_id",
        "reaction_digest",
        "source_ref",
    )
    return all(supplied.get(field) == derived.get(field) for field in fields)


def _trusted_binding(
    value: Mapping[str, Any],
    *,
    expected_reaction_digest: str,
) -> bool:
    if not value or value.get("accepted") is not True:
        return False
    authority = str(value.get("authority") or "")
    digest = str(value.get("reaction_digest") or "").lower()
    return bool(
        value.get("schema_version") == "trusted_precedent_binding.v1"
        and authority in {"human_curator", "deterministic_structure_parser"}
        and str(value.get("authority_id") or "").strip()
        and str(value.get("binding_id") or "").strip()
        and str(value.get("source_ref") or "").strip()
        and _is_sha256(digest)
        and digest == expected_reaction_digest
    )


def _procurement_binding(
    value: Mapping[str, Any],
    *,
    expected_reactants: tuple[str, ...],
) -> bool:
    if (
        not value
        or value.get("schema_version") != "verified_reaction_procurement.v1"
        or value.get("accepted") is not True
    ):
        return False
    results = value.get("stock_provider_results")
    if not isinstance(results, list) or not results:
        return False
    try:
        from cascade_planner.providers.contracts import validate_provider_result
    except ImportError:
        return False
    covered: set[str] = set()
    content_hashes: list[str] = []
    for result in results:
        if not isinstance(result, Mapping) or validate_provider_result(result):
            return False
        envelope = dict(result)
        payload = envelope.get("payload")
        if (
            envelope.get("provider_id") != "autoplanner.snapshot_stock"
            or envelope.get("provider_kind") != "stock"
            or envelope.get("output_schema") != "stock_boundary.v1"
            or envelope.get("accepted") is not True
            or not isinstance(payload, Mapping)
            or payload.get("accepted") is not True
        ):
            return False
        molecule = _canonical_smiles(payload.get("canonical_smiles"))
        if not molecule or molecule not in expected_reactants:
            return False
        offers = payload.get("offers")
        if payload.get("boundary_type") == "commercially_orderable" and not (
            isinstance(offers, list)
            and any(
                isinstance(offer, Mapping)
                and offer.get("available") is True
                and offer.get("snapshot_verified") is True
                and _is_sha256(str(offer.get("snapshot_sha256") or ""))
                for offer in offers
            )
        ):
            return False
        covered.add(molecule)
        content_hashes.append(str(envelope.get("content_hash") or ""))
    expected = sorted(set(expected_reactants))
    digest_payload = {
        "reactant_smiles": expected,
        "stock_provider_content_hashes": sorted(content_hashes),
    }
    return bool(
        covered == set(expected)
        and _is_sha256(str(value.get("binding_digest") or ""))
        and str(value.get("binding_digest") or "") == _digest(digest_payload)
    )


def build_verified_procurement_binding(
    stock_provider_results: Iterable[Mapping[str, Any]],
    *,
    reactant_smiles: Iterable[Any],
) -> dict[str, Any]:
    """Construct a replayable L4 binding from verified stock envelopes."""
    results = [dict(row) for row in stock_provider_results if isinstance(row, Mapping)]
    reactants = sorted(
        {value for value in (_canonical_smiles(item) for item in reactant_smiles) if value}
    )
    payload = {
        "reactant_smiles": reactants,
        "stock_provider_content_hashes": sorted(
            str(row.get("content_hash") or "") for row in results
        ),
    }
    binding = {
        "schema_version": "verified_reaction_procurement.v1",
        "accepted": True,
        "stock_provider_results": results,
        "binding_digest": _digest(payload),
    }
    if not _procurement_binding(binding, expected_reactants=tuple(reactants)):
        binding["accepted"] = False
    return binding


def _is_sha256(value: str) -> bool:
    return bool(len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower()))


def _conditions_complete(step: Mapping[str, Any]) -> bool:
    conditions = step.get("conditions")
    if isinstance(conditions, Mapping):
        present = {str(key) for key, value in conditions.items() if str(value or "").strip()}
    elif isinstance(conditions, list):
        present = {
            str(row.get("label") or row.get("field") or "")
            for row in conditions
            if isinstance(row, Mapping) and (row.get("value") or row.get("text"))
        }
    else:
        present = set()
    normalized = {value.lower().replace(" ", "_") for value in present}
    return bool(
        normalized & {"reagent", "reagents", "catalyst", "enzyme"}
        and normalized & {"solvent", "medium"}
        and normalized & {"temperature", "temp"}
        and normalized & {"duration", "time"}
    )


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
