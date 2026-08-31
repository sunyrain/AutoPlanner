"""Compact, read-only RDKit inspection for Builder stereochemistry queries."""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


def inspect_mapped_smiles(
    mapped_smiles: str,
    *,
    map_ids: Iterable[int] = (),
    enumerate_unassigned: bool = False,
    max_isomers: int = 8,
) -> dict[str, Any]:
    """Return bounded mapped-graph and stereo facts for one local graph edit."""

    molecule = Chem.MolFromSmiles(str(mapped_smiles or ""))
    if molecule is None:
        return {"ok": False, "reason": "invalid_smiles"}
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    requested = {int(value) for value in map_ids if int(value) > 0}
    atom_by_index = {atom.GetIdx(): atom for atom in molecule.GetAtoms()}
    rings = [
        [
            int(atom_by_index[int(atom_index)].GetAtomMapNum())
            for atom_index in ring
        ]
        for ring in Chem.GetSymmSSSR(molecule)
    ]
    ring_sizes_by_map: dict[int, set[int]] = {}
    for ring in rings:
        ring_size = len(ring)
        for map_idx in ring:
            if map_idx > 0:
                ring_sizes_by_map.setdefault(map_idx, set()).add(ring_size)
    atoms = []
    for atom in molecule.GetAtoms():
        map_idx = int(atom.GetAtomMapNum())
        if requested and map_idx not in requested:
            continue
        atoms.append(
            {
                "map_idx": map_idx,
                "element": atom.GetSymbol(),
                "formal_charge": int(atom.GetFormalCharge()),
                "total_h": int(atom.GetTotalNumHs()),
                "degree": int(atom.GetDegree()),
                "aromatic": bool(atom.GetIsAromatic()),
                "ring_sizes": sorted(ring_sizes_by_map.get(map_idx, set())),
            }
        )
    centers: list[dict[str, Any]] = []
    for atom_index, label in Chem.FindMolChiralCenters(
        molecule,
        includeUnassigned=True,
        includeCIP=True,
    ):
        atom = atom_by_index[int(atom_index)]
        map_idx = int(atom.GetAtomMapNum())
        if requested and map_idx not in requested:
            continue
        centers.append(
            {
                "map_idx": map_idx,
                "element": atom.GetSymbol(),
                "cip": str(label),
            }
        )
    stereo_bonds: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []
    for bond in molecule.GetBonds():
        map_a = int(bond.GetBeginAtom().GetAtomMapNum())
        map_b = int(bond.GetEndAtom().GetAtomMapNum())
        if requested and not ({map_a, map_b} & requested):
            continue
        stereo = str(bond.GetStereo()).replace("STEREO", "")
        bonds.append(
            {
                "map_a": map_a,
                "map_b": map_b,
                "order": float(bond.GetBondTypeAsDouble()),
                "aromatic": bool(bond.GetIsAromatic()),
                "in_ring": bool(bond.IsInRing()),
                "stereo": stereo or "NONE",
            }
        )
        if stereo in {"", "NONE"}:
            continue
        stereo_bonds.append(
            {
                "map_a": map_a,
                "map_b": map_b,
                "stereo": stereo,
            }
        )
    result: dict[str, Any] = {
        "ok": True,
        "canonical_mapped_smiles": Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        ),
        "atoms": atoms,
        "bonds": bonds,
        "rings": [
            ring for ring in rings if not requested or requested.intersection(ring)
        ],
        "centers": centers,
        "unassigned_center_maps": [
            int(row["map_idx"])
            for row in centers
            if row["cip"] == "?" and int(row["map_idx"]) > 0
        ],
        "stereo_bonds": stereo_bonds,
    }
    if enumerate_unassigned and result["unassigned_center_maps"]:
        options = StereoEnumerationOptions(
            onlyUnassigned=True,
            unique=True,
            maxIsomers=max(1, min(int(max_isomers), 32)),
        )
        result["limited_stereoisomers"] = [
            Chem.MolToSmiles(value, canonical=True, isomericSmiles=True)
            for value in EnumerateStereoisomers(molecule, options=options)
        ]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect mapped stereochemistry without writing an RDKit script."
    )
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--map-id", action="append", type=int, default=[])
    parser.add_argument("--enumerate-unassigned", action="store_true")
    parser.add_argument("--max-isomers", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = inspect_mapped_smiles(
        args.smiles,
        map_ids=args.map_id,
        enumerate_unassigned=args.enumerate_unassigned,
        max_isomers=args.max_isomers,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
