"""Atom-map-independent stereochemical observations for the canonical Host.

Maps identify atoms; they are not chemical substituent priorities.  Work on a
copy with maps removed, retaining atom/bond indices for the caller's namespace.
Use RDKit's sequence-rule CIP labeler rather than its legacy CIP approximation.
"""

from typing import Any

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


STEREOCHEMISTRY_VERSION = "host_stereochemistry.v2"


def stereo_molecule(molecule: Chem.Mol) -> Chem.Mol:
    """Return an independently labelled copy without changing the input graph."""
    probe = Chem.Mol(molecule)
    for atom in probe.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SetDoubleBondNeighborDirections(probe)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    # Legacy labels must not survive when the sequence-rule labeler cannot
    # assign a center (e.g. equivalent substituents distinguished only by maps).
    for entity in (*probe.GetAtoms(), *probe.GetBonds()):
        if entity.HasProp("_CIPCode"):
            entity.ClearProp("_CIPCode")
    rdCIPLabeler.AssignCIPLabels(probe)
    return probe


def canonical_stereo_smiles(value: Any) -> str:
    """Chemical identity without maps, retaining physical stereo and isotopes.

    Reassign stereo *after* removing maps: map labels can distinguish chemically
    equivalent paths and leave nonphysical tetrahedral tags on bridged systems.
    Mapped edit/replay namespaces must use their separate mapped serializers.
    """
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(stereo_molecule(molecule), canonical=True, isomericSmiles=True)


def tetrahedral_centers(molecule: Chem.Mol) -> list[tuple[int, str]]:
    """Return true CIP labels, including unspecified potential centers as ?."""
    probe = stereo_molecule(molecule)
    potential = Chem.FindMolChiralCenters(
        probe,
        includeUnassigned=True,
        includeCIP=False,
        useLegacyImplementation=False,
    )
    return [
        (
            index,
            probe.GetAtomWithIdx(index).GetProp("_CIPCode")
            if probe.GetAtomWithIdx(index).HasProp("_CIPCode")
            else "?",
        )
        for index, _label in potential
    ]


def atom_cip_label(molecule: Chem.Mol, atom_index: int) -> str:
    atom = stereo_molecule(molecule).GetAtomWithIdx(atom_index)
    return atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""


def bond_stereo_label(molecule: Chem.Mol, bond_index: int) -> str:
    bond = stereo_molecule(molecule).GetBondWithIdx(bond_index)
    if bond.HasProp("_CIPCode"):
        return bond.GetProp("_CIPCode")
    return str(bond.GetStereo()).replace("STEREO", "")
