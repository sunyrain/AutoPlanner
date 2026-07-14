"""Deterministic formulation bridges for source-derived route proposals.

Sources often report a hydroxy-acid while the requested drug is its lactone.
This module recognizes only the explicit graph edit for hydrolysis of a cyclic
ester.  The resulting edge remains an L0 proposal and must pass the ordinary
host mapper; structural equivalence never grants literature or reaction proof.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

from rdkit import Chem

from cascade_planner.routes.admission import audit_retrosynthetic_candidate


def build_lactone_form_bridge_proposals(
    proposals: Iterable[Mapping[str, Any]],
    *,
    anchor_smiles: Iterable[str],
) -> list[dict[str, Any]]:
    """Connect source hydroxy-acids to target lactones by an audited graph edit."""

    anchors = sorted(
        {
            canonical
            for value in anchor_smiles
            if (canonical := _canonical(value))
        }
    )
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in proposals:
        source = dict(raw)
        acid = _canonical(source.get("product_smiles"))
        if not acid:
            continue
        for lactone in anchors:
            if acid not in _lactone_hydrolysis_products(lactone):
                continue
            identity = (
                str(source.get("source_ref") or ""),
                lactone,
                acid,
            )
            if identity in seen:
                continue
            seen.add(identity)
            audit = audit_retrosynthetic_candidate(lactone, [acid])
            if audit.get("accepted") is not True:
                continue
            proposal_id = "source-form-bridge:" + _digest(identity)[:24]
            output.append(
                {
                    "schema_version": "deterministic_source_route_proposal.v1",
                    "proposal_id": proposal_id,
                    "step_id": proposal_id,
                    "source_ref": str(source.get("source_ref") or ""),
                    "source_artifact_sha256": str(
                        source.get("source_artifact_sha256") or ""
                    ),
                    "source_location": dict(source.get("source_location") or {}),
                    "evidence_refs": list(source.get("evidence_refs") or []),
                    "product_name": (
                        "lactone form of "
                        + str(source.get("product_name") or "source hydroxy acid")
                    ),
                    "product_smiles": lactone,
                    "source_product_smiles": acid,
                    "product_structure_recovery_mode": (
                        "deterministic_cyclic_ester_hydrolysis_equivalence"
                    ),
                    "precursor_smiles": [acid],
                    "reactant_smiles": [acid],
                    "reactant_names": [
                        str(source.get("product_name") or "source hydroxy acid")
                    ],
                    "reagent_smiles": [],
                    "condition_candidate": {},
                    "origin_kind": "deterministic_source_form_bridge",
                    "origin_ref": str(source.get("source_ref") or ""),
                    "transformation_hypothesis": (
                        "intramolecular lactonization of a source-resolved "
                        "hydroxy acid; ordinary host mapping remains required"
                    ),
                    "admission_audit": audit,
                    "semantics": {
                        "proposal_only": True,
                        "structural_form_equivalence_only": True,
                        "source_text_grants_no_reaction_validation": True,
                        "deterministic_registry_replay_required_for_exact_proof": True,
                    },
                }
            )
    return sorted(output, key=lambda row: str(row["proposal_id"]))


def _lactone_hydrolysis_products(smiles: str) -> set[str]:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return set()
    products: set[str] = set()
    for carbon in molecule.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        has_carbonyl_oxygen = any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(carbon).GetAtomicNum() == 8
            for bond in carbon.GetBonds()
        )
        if not has_carbonyl_oxygen:
            continue
        for bond in carbon.GetBonds():
            ring_oxygen = bond.GetOtherAtom(carbon)
            if (
                bond.GetBondType() != Chem.BondType.SINGLE
                or ring_oxygen.GetAtomicNum() != 8
                or not bond.IsInRing()
            ):
                continue
            editable = Chem.RWMol(molecule)
            editable.RemoveBond(carbon.GetIdx(), ring_oxygen.GetIdx())
            hydroxyl = editable.AddAtom(Chem.Atom(8))
            editable.AddBond(carbon.GetIdx(), hydroxyl, Chem.BondType.SINGLE)
            product = editable.GetMol()
            try:
                Chem.SanitizeMol(product)
            except (RuntimeError, ValueError):
                continue
            canonical = _canonical(Chem.MolToSmiles(product, isomericSmiles=True))
            if canonical:
                products.add(canonical)
    return products


def _canonical(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or ""))
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else ""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["build_lactone_form_bridge_proposals"]
