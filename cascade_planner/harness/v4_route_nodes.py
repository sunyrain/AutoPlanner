"""Build depicted molecule nodes for the V4 route-workbench adapter."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from cascade_planner.harness.route_forest_delivery import sanitize_structure_svg


def node(molecule_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
    smiles = str(value.get("canonical_smiles") or "")
    image = depiction(smiles)
    return {
        "node_id": molecule_id,
        "label": str(value.get("label") or "")
        or image["formula"]
        or str(value.get("role") or "Molecule"),
        "canonical_isomeric_smiles": smiles,
        "smiles": smiles,
        "role": str(value.get("role") or "intermediate"),
        "roles": [str(value.get("role") or "intermediate")],
        "formula": image["formula"],
        "heavy_atom_count": image["heavy_atom_count"],
        "structure_svg": image["structure_svg"],
        "stock_closed": value.get("stock_closed") is True,
        "stock_observation_id": str(value.get("stock_observation_id") or ""),
        "stock_authority_scope": str(value.get("stock_authority_scope") or ""),
        "stock_observation_accepted": value.get("stock_observation_accepted") is True,
        "inactive_fact_count": int(value.get("inactive_fact_count") or 0),
        "inactive_facts": list(value.get("inactive_facts") or []),
    }


@lru_cache(maxsize=1024)
def depiction(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"formula": "", "heavy_atom_count": None, "structure_svg": ""}
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(240, 170)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    raw = drawer.GetDrawingText().replace("svg:", "")
    raw = raw[raw.find("<svg") :] if "<svg" in raw else raw
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "structure_svg": sanitize_structure_svg(raw),
    }


__all__ = ["depiction", "node"]
