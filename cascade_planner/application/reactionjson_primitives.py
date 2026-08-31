"""Strict ReactionJSON primitive normalization and RDKit graph edits."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from rdkit import Chem


PRIMITIVES = (
    "break_bond",
    "add_bond",
    "change_bond_order",
    "change_atom",
    "set_explicit_h",
    "add_group",
    "remove_group",
    "invert_stereocenter",
    "clear_stereocenter",
    "set_bond_stereo",
)
AUTOPLANNER_EXTENSIONS = (
    "set_tetrahedral_stereo",
)
SUPPORTED_OPERATIONS = PRIMITIVES + AUTOPLANNER_EXTENSIONS
_COMMON = {"op"}
_FIELDS = {
    "break_bond": _COMMON | {"map_a", "map_b"},
    "add_bond": _COMMON | {"map_a", "map_b", "order"},
    "change_bond_order": _COMMON | {"map_a", "map_b", "delta"},
    "change_atom": _COMMON
    | {"map_idx", "atomic_num", "element", "formal_charge", "isotope"},
    "set_explicit_h": _COMMON | {"map_idx", "count", "no_implicit"},
    "add_group": _COMMON | {"map_idx", "fragment_smiles", "order"},
    "remove_group": _COMMON | {"map_indices"},
    "invert_stereocenter": _COMMON | {"map_idx"},
    "clear_stereocenter": _COMMON | {"map_idx"},
    "set_bond_stereo": _COMMON
    | {"map_a", "map_b", "stereo", "stereo_atom_maps"},
    "set_tetrahedral_stereo": _COMMON | {"map_idx", "configuration"},
}
_BOND_TYPES = {
    1.0: Chem.BondType.SINGLE,
    1.5: Chem.BondType.AROMATIC,
    2.0: Chem.BondType.DOUBLE,
    3.0: Chem.BondType.TRIPLE,
}
_STEREO = {
    "NONE": Chem.BondStereo.STEREONONE,
    "ANY": Chem.BondStereo.STEREOANY,
    "Z": Chem.BondStereo.STEREOZ,
    "E": Chem.BondStereo.STEREOE,
    "CIS": Chem.BondStereo.STEREOCIS,
    "TRANS": Chem.BondStereo.STEREOTRANS,
}


class ReactionJsonReplayError(ValueError):
    """The edit program is outside the profile or cannot be replayed safely."""

    def __init__(
        self,
        reason: str,
        *,
        operation_index: int | None = None,
        failed_operation: Mapping[str, Any] | None = None,
        failure_context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.operation_index = operation_index
        self.failed_operation = (
            dict(failed_operation) if failed_operation is not None else None
        )
        self.failure_context = dict(failure_context or {})


def normalize_operation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReactionJsonReplayError("reactionjson_operation_mapping_required")
    try:
        row = json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReactionJsonReplayError("reactionjson_operation_not_json") from exc
    kind = str(row.get("op") or "").strip().lower()
    if kind not in SUPPORTED_OPERATIONS:
        raise ReactionJsonReplayError("reactionjson_primitive_unknown")
    if set(row) - _FIELDS[kind]:
        raise ReactionJsonReplayError("reactionjson_operation_field_unknown")
    row["op"] = kind
    return row


def apply_operation(molecule: Chem.RWMol, row: Mapping[str, Any]) -> Chem.RWMol:
    kind = str(row["op"])
    if kind == "break_bond":
        a, b = _bond_atoms(molecule, row)
        if molecule.GetBondBetweenAtoms(a, b) is None:
            raise ReactionJsonReplayError("reactionjson_bond_missing")
        molecule.RemoveBond(a, b)
    elif kind == "add_bond":
        a, b = _bond_atoms(molecule, row)
        if molecule.GetBondBetweenAtoms(a, b) is not None:
            raise ReactionJsonReplayError("reactionjson_bond_already_exists")
        bond_type = _bond_type(row.get("order", 1))
        if bond_type == Chem.BondType.AROMATIC and not (
            molecule.GetAtomWithIdx(a).GetIsAromatic()
            and molecule.GetAtomWithIdx(b).GetIsAromatic()
        ):
            raise ReactionJsonReplayError(
                "reactionjson_aromatic_bond_requires_aromatic_atoms"
            )
        molecule.AddBond(a, b, bond_type)
    elif kind == "change_bond_order":
        a, b = _bond_atoms(molecule, row)
        bond = molecule.GetBondBetweenAtoms(a, b)
        if bond is None:
            raise ReactionJsonReplayError("reactionjson_bond_missing")
        delta = _finite_number(row.get("delta"), "reactionjson_bond_delta_invalid")
        bond.SetBondType(_bond_type(bond.GetBondTypeAsDouble() + delta))
        bond.SetIsAromatic(bond.GetBondType() == Chem.BondType.AROMATIC)
    elif kind == "change_atom":
        _change_atom(molecule, row)
    elif kind == "set_explicit_h":
        atom = molecule.GetAtomWithIdx(_map_index(molecule, row.get("map_idx")))
        count = _integer(row.get("count"), "reactionjson_explicit_h_invalid", minimum=0)
        atom.SetNumExplicitHs(count)
        atom.SetNoImplicit(row.get("no_implicit", True) is not False)
    elif kind == "add_group":
        molecule = _add_group(molecule, row)
    elif kind == "remove_group":
        maps = row.get("map_indices")
        if not isinstance(maps, list) or not maps:
            raise ReactionJsonReplayError("reactionjson_remove_group_maps_invalid")
        indices = sorted({_map_index(molecule, value) for value in maps}, reverse=True)
        for index in indices:
            molecule.RemoveAtom(index)
    elif kind == "invert_stereocenter":
        atom = molecule.GetAtomWithIdx(_map_index(molecule, row.get("map_idx")))
        inverse = {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        }.get(atom.GetChiralTag())
        if inverse is None:
            raise ReactionJsonReplayError("reactionjson_stereocenter_not_invertible")
        atom.SetChiralTag(inverse)
    elif kind == "clear_stereocenter":
        molecule.GetAtomWithIdx(_map_index(molecule, row.get("map_idx"))).SetChiralTag(
            Chem.ChiralType.CHI_UNSPECIFIED
        )
    elif kind == "set_bond_stereo":
        _set_bond_stereo(molecule, row)
    elif kind == "set_tetrahedral_stereo":
        _set_tetrahedral_stereo(molecule, row)
    return molecule


def valence_affected_maps(
    molecule: Chem.RWMol,
    row: Mapping[str, Any],
) -> set[int]:
    """Return mapped atoms whose ordinary valence must be recomputed."""

    kind = str(row.get("op") or "")
    if kind in {"break_bond", "add_bond", "change_bond_order"}:
        return {
            _integer(row.get("map_a"), "reactionjson_map_invalid", minimum=1),
            _integer(row.get("map_b"), "reactionjson_map_invalid", minimum=1),
        }
    if kind in {"change_atom", "add_group"}:
        return {
            _integer(row.get("map_idx"), "reactionjson_map_invalid", minimum=1)
        }
    if kind != "remove_group":
        return set()
    removed_maps = {
        _integer(value, "reactionjson_map_invalid", minimum=1)
        for value in row.get("map_indices") or []
    }
    affected: set[int] = set()
    lookup = _map_lookup(molecule)
    for removed_map in removed_maps:
        try:
            atom = molecule.GetAtomWithIdx(lookup[removed_map])
        except KeyError as exc:
            raise ReactionJsonReplayError("reactionjson_map_not_found") from exc
        affected.update(
            int(neighbor.GetAtomMapNum())
            for neighbor in atom.GetNeighbors()
            if int(neighbor.GetAtomMapNum()) > 0
            and int(neighbor.GetAtomMapNum()) not in removed_maps
        )
    return affected


def complete_edited_atom_valences(
    molecule: Chem.RWMol,
    *,
    map_indices: Iterable[int],
) -> list[int]:
    """Replace product-state hydrogens/radicals with ordinary edited valence."""

    lookup = _map_lookup(molecule)
    completed: list[int] = []
    for map_idx in sorted(set(int(value) for value in map_indices)):
        atom_index = lookup.get(map_idx)
        if atom_index is None:
            continue
        atom = molecule.GetAtomWithIdx(atom_index)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
        atom.UpdatePropertyCache(strict=False)
        completed.append(map_idx)
    return completed


def _change_atom(molecule: Chem.RWMol, row: Mapping[str, Any]) -> None:
    atom = molecule.GetAtomWithIdx(_map_index(molecule, row.get("map_idx")))
    if "atomic_num" in row or "element" in row:
        raise ReactionJsonReplayError(
            "reactionjson_change_atom_transmutation_forbidden"
        )
    fields = {
        key
        for key in ("formal_charge", "isotope")
        if key in row
    }
    if not fields:
        raise ReactionJsonReplayError("reactionjson_change_atom_field_required")
    if len(fields) != 1:
        raise ReactionJsonReplayError("reactionjson_change_atom_field_ambiguous")
    if "formal_charge" in row:
        atom.SetFormalCharge(_integer(row["formal_charge"], "reactionjson_charge_invalid"))
    if "isotope" in row:
        atom.SetIsotope(_integer(row["isotope"], "reactionjson_isotope_invalid", minimum=0))


def _add_group(molecule: Chem.RWMol, row: Mapping[str, Any]) -> Chem.RWMol:
    anchor_map = _integer(row.get("map_idx"), "reactionjson_map_invalid", minimum=1)
    anchor_index = _map_index(molecule, anchor_map)
    fragment = Chem.MolFromSmiles(str(row.get("fragment_smiles") or "").strip())
    if fragment is None:
        raise ReactionJsonReplayError("reactionjson_fragment_invalid")
    dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1 or dummies[0].GetDegree() != 1:
        raise ReactionJsonReplayError("reactionjson_fragment_attachment_invalid")
    existing_maps = set(_map_lookup(molecule))
    fragment_atoms = [
        atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() != 0
    ]
    explicit_fragment_maps = [
        int(atom.GetAtomMapNum())
        for atom in fragment_atoms
        if int(atom.GetAtomMapNum()) > 0
    ]
    if len(explicit_fragment_maps) != len(set(explicit_fragment_maps)):
        raise ReactionJsonReplayError("reactionjson_fragment_maps_invalid")
    if existing_maps & set(explicit_fragment_maps):
        raise ReactionJsonReplayError("reactionjson_fragment_map_collision")
    # SynthEx's public Editor example adds ``*MgBr`` without requiring the
    # model to invent atom-map numbers for the new handle.  Allocate those
    # maps deterministically at the replay boundary while retaining strict
    # uniqueness/collision checks for any explicit maps the model supplied.
    used_maps = existing_maps | set(explicit_fragment_maps)
    supplied_fresh_maps = iter(row.get("_fresh_atom_maps") or ())
    for atom in fragment_atoms:
        if int(atom.GetAtomMapNum()) > 0:
            continue
        try:
            next_map = int(next(supplied_fresh_maps))
        except (StopIteration, TypeError, ValueError):
            next_map = max(used_maps, default=0) + 1
            while next_map in used_maps:
                next_map += 1
        if next_map <= 0 or next_map in used_maps:
            raise ReactionJsonReplayError("reactionjson_fragment_map_collision")
        atom.SetAtomMapNum(next_map)
        used_maps.add(next_map)
    dummy = dummies[0]
    neighbor = dummy.GetNeighbors()[0]
    attachment_bond = fragment.GetBondBetweenAtoms(dummy.GetIdx(), neighbor.GetIdx())
    attachment_bond_type = (
        _bond_type(_finite_number(row.get("order"), "reactionjson_bond_order_invalid"))
        if row.get("order") is not None
        else attachment_bond.GetBondType()
    )
    anchor_atom = molecule.GetAtomWithIdx(anchor_index)
    if attachment_bond_type == Chem.BondType.AROMATIC and not (
        anchor_atom.GetIsAromatic() and neighbor.GetIsAromatic()
    ):
        raise ReactionJsonReplayError(
            "reactionjson_aromatic_bond_requires_aromatic_atoms"
        )
    offset = molecule.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(molecule.GetMol(), fragment))
    combined.AddBond(anchor_index, offset + neighbor.GetIdx(), attachment_bond_type)
    combined.RemoveAtom(offset + dummy.GetIdx())
    return combined


def _set_bond_stereo(molecule: Chem.RWMol, row: Mapping[str, Any]) -> None:
    a, b = _bond_atoms(molecule, row)
    bond = molecule.GetBondBetweenAtoms(a, b)
    if bond is None:
        raise ReactionJsonReplayError("reactionjson_bond_missing")
    stereo_name = str(row.get("stereo") or "").strip().upper()
    if stereo_name not in _STEREO:
        raise ReactionJsonReplayError("reactionjson_bond_stereo_invalid")
    if stereo_name not in {"NONE", "ANY"}:
        left, right = _host_stereo_reference_atoms(molecule, a, b)
        # RDKit requires the first reference atom to neighbour the bond's
        # internal begin atom and the second to neighbour its end atom.  Atom
        # map order is a model-facing identity and need not match that hidden
        # storage orientation, especially after remove_group/change-order
        # edits.  Reorder only the Host-derived references; E/Z intent remains
        # the model's semantic input.
        if bond.GetBeginAtomIdx() == a and bond.GetEndAtomIdx() == b:
            bond.SetStereoAtoms(left, right)
        elif bond.GetBeginAtomIdx() == b and bond.GetEndAtomIdx() == a:
            bond.SetStereoAtoms(right, left)
        else:  # pragma: no cover - RDKit bond endpoints must be {a, b}.
            raise ReactionJsonReplayError(
                "reactionjson_stereo_bond_endpoint_mismatch"
            )
    bond.SetStereo(_STEREO[stereo_name])


def _set_tetrahedral_stereo(
    molecule: Chem.RWMol,
    row: Mapping[str, Any],
) -> None:
    """Assign absolute R/S intent without exposing RDKit atom-order parity."""

    atom_index = _map_index(molecule, row.get("map_idx"))
    requested = str(row.get("configuration") or "").strip().upper()
    if requested not in {"R", "S"}:
        raise ReactionJsonReplayError(
            "reactionjson_tetrahedral_configuration_invalid"
        )
    matching_tag: Chem.ChiralType | None = None
    for tag in (
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ):
        probe = Chem.Mol(molecule)
        probe.GetAtomWithIdx(atom_index).SetChiralTag(tag)
        try:
            probe.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(probe)
            Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
        except Exception:
            continue
        atom = probe.GetAtomWithIdx(atom_index)
        if atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode") == requested:
            matching_tag = tag
            break
    if matching_tag is None:
        raise ReactionJsonReplayError(
            "reactionjson_tetrahedral_stereo_not_assignable"
        )
    molecule.GetAtomWithIdx(atom_index).SetChiralTag(matching_tag)


def _host_stereo_reference_atoms(
    molecule: Chem.RWMol,
    a: int,
    b: int,
) -> tuple[int, int]:
    """Select the highest-CIP explicit neighbour on each alkene endpoint."""

    probe = Chem.Mol(molecule)
    try:
        probe.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(probe)
        Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    except Exception as exc:
        raise ReactionJsonReplayError(
            "reactionjson_stereo_reference_derivation_failed"
        ) from exc

    def selected(endpoint: int, other: int) -> int:
        candidates = [
            neighbor.GetIdx()
            for neighbor in probe.GetAtomWithIdx(endpoint).GetNeighbors()
            if neighbor.GetIdx() != other
        ]
        if not candidates:
            raise ReactionJsonReplayError(
                "reactionjson_stereo_reference_neighbor_missing"
            )

        def priority(atom_index: int) -> tuple[int, int, int]:
            atom = probe.GetAtomWithIdx(atom_index)
            try:
                cip_rank = int(atom.GetProp("_CIPRank"))
            except (KeyError, ValueError):
                cip_rank = -1
            return (cip_rank, int(atom.GetAtomicNum()), -atom_index)

        return max(candidates, key=priority)

    return selected(a, b), selected(b, a)


def _bond_atoms(molecule: Chem.RWMol, row: Mapping[str, Any]) -> tuple[int, int]:
    a = _map_index(molecule, row.get("map_a"))
    b = _map_index(molecule, row.get("map_b"))
    if a == b:
        raise ReactionJsonReplayError("reactionjson_self_bond_invalid")
    return a, b


def _map_lookup(molecule: Chem.RWMol | Chem.Mol) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        value = atom.GetAtomMapNum()
        if value > 0:
            if value in lookup:
                raise ReactionJsonReplayError("reactionjson_map_duplicate")
            lookup[value] = atom.GetIdx()
    return lookup


def _map_index(molecule: Chem.RWMol | Chem.Mol, value: Any) -> int:
    key = _integer(value, "reactionjson_map_invalid", minimum=1)
    try:
        return _map_lookup(molecule)[key]
    except KeyError as exc:
        raise ReactionJsonReplayError("reactionjson_map_not_found") from exc


def _bond_type(value: Any) -> Chem.BondType:
    number = _finite_number(value, "reactionjson_bond_order_invalid")
    if number not in _BOND_TYPES:
        raise ReactionJsonReplayError("reactionjson_bond_order_invalid")
    return _BOND_TYPES[number]


def _finite_number(value: Any, reason: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReactionJsonReplayError(reason) from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise ReactionJsonReplayError(reason)
    return number


def _integer(value: Any, reason: str, *, minimum: int | None = None) -> int:
    number = _finite_number(value, reason)
    if not number.is_integer() or (minimum is not None and number < minimum):
        raise ReactionJsonReplayError(reason)
    return int(number)


__all__ = [
    "AUTOPLANNER_EXTENSIONS",
    "PRIMITIVES",
    "ReactionJsonReplayError",
    "apply_operation",
    "complete_edited_atom_valences",
    "normalize_operation",
    "SUPPORTED_OPERATIONS",
    "valence_affected_maps",
]
