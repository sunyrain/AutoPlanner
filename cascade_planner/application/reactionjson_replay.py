"""Deterministic, provisional ReactionJSON graph-edit replay.

The public paper names ten primitives but has no released implementation or field
specification.  This fail-closed profile treats replay as proposal, never proof.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from rdkit import Chem, RDLogger

from .reactionjson_primitives import AUTOPLANNER_EXTENSIONS, PRIMITIVES
from .reactionjson_primitives import ReactionJsonReplayError
from .reactionjson_primitives import apply_operation
from .reactionjson_primitives import complete_edited_atom_valences
from .reactionjson_primitives import normalize_operation
from .reactionjson_primitives import valence_affected_maps

RDLogger.DisableLog("rdApp.*")
REACTIONJSON_PROFILE = "reactionjson_public_profile.2026-08-17.v1"
REACTIONJSON_EXTENSION_PROFILE = "reactionjson_autoplanner_extensions.2026-08-28.v1"
REACTIONJSON_REPLAY_AUDIT_SCHEMA = "reactionjson_replay_audit.v1"
UPSTREAM_PUBLIC_COMMIT = "5f41a6b21e3906fde93e84c88bb91f9dc4d37e6f"


def replay_reactionjson(
    *,
    mapped_product_smiles: str,
    operations: Iterable[Mapping[str, Any]],
    expected_precursor_smiles: Iterable[str] | None = None,
    reserved_atom_maps: Iterable[int] = (),
) -> dict[str, Any]:
    """Replay ten bounded primitives and return a content-bound audit."""

    product = Chem.MolFromSmiles(str(mapped_product_smiles or "").strip())
    if product is None:
        raise ReactionJsonReplayError("reactionjson_product_invalid")
    _require_complete_unique_maps(product, reason="reactionjson_product_maps_invalid")
    rows = [normalize_operation(value) for value in operations]
    if not rows or len(rows) > 128:
        raise ReactionJsonReplayError("reactionjson_operation_count_invalid")
    rows, fresh_atom_maps = _resolve_add_group_atom_maps(
        product,
        rows,
        reserved_atom_maps=reserved_atom_maps,
    )
    editable = Chem.RWMol(product)
    valence_completion_maps: set[int] = set()
    explicit_h_maps = {
        int(row["map_idx"])
        for row in rows
        if row["op"] == "set_explicit_h"
    }
    deferred_stereo: list[tuple[int, dict[str, Any]]] = []
    for operation_index, row in enumerate(rows):
        if row["op"] in {"set_bond_stereo", "set_tetrahedral_stereo"}:
            # Bond stereo is serialization on the final edited graph, not a
            # separate skeletal edit.  Defer it until ordinary valences have
            # been recomputed so the Host can select the correct CIP reference
            # neighbours instead of asking the model to serialize RDKit state.
            deferred_stereo.append((operation_index, row))
            continue
        try:
            valence_completion_maps.update(valence_affected_maps(editable, row))
            editable = apply_operation(editable, row)
        except ReactionJsonReplayError as exc:
            raise ReactionJsonReplayError(
                str(exc),
                operation_index=operation_index,
                failed_operation=_public_operation(row),
                failure_context=_operation_failure_context(editable, row, exc),
            ) from exc
        except Exception as exc:
            raise ReactionJsonReplayError(
                "reactionjson_replay_failed",
                operation_index=operation_index,
                failed_operation=_public_operation(row),
            ) from exc
    try:
        completed_maps = complete_edited_atom_valences(
            editable,
            map_indices=valence_completion_maps - explicit_h_maps,
        )
        invalidated_bond_stereo = _clear_invalid_bond_stereo_references(editable)
        explicitly_reassigned_bonds = {
            frozenset((int(row["map_a"]), int(row["map_b"])))
            for _operation_index, row in deferred_stereo
            if row["op"] == "set_bond_stereo"
        }
        unresolved_bond_stereo = [
            row
            for row in invalidated_bond_stereo
            if frozenset((int(row["map_a"]), int(row["map_b"])))
            not in explicitly_reassigned_bonds
        ]
        if unresolved_bond_stereo:
            raise ReactionJsonReplayError(
                "reactionjson_bond_stereo_invalidated_by_graph_edit",
                failure_context={
                    "failure_stage": "graph_finalization",
                    "invalidated_bond_stereo": unresolved_bond_stereo,
                    "required_repair": (
                        "add set_bond_stereo for each affected retained double bond"
                    ),
                },
            )
        for operation_index, row in deferred_stereo:
            try:
                editable = apply_operation(editable, row)
            except ReactionJsonReplayError as exc:
                raise ReactionJsonReplayError(
                    str(exc),
                    operation_index=operation_index,
                    failed_operation=_public_operation(row),
                ) from exc
        Chem.SetDoubleBondNeighborDirections(editable)
        replayed = editable.GetMol()
        Chem.SanitizeMol(replayed)
        Chem.AssignStereochemistry(replayed, cleanIt=True, force=True)
        for operation_index, row in deferred_stereo:
            if row["op"] != "set_tetrahedral_stereo":
                continue
            atom = replayed.GetAtomWithIdx(
                _map_index_for_audit(replayed, int(row["map_idx"]))
            )
            requested = str(row["configuration"]).upper()
            if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != requested:
                raise ReactionJsonReplayError(
                    "reactionjson_tetrahedral_stereo_not_assignable",
                    operation_index=operation_index,
                    failed_operation=_public_operation(row),
                )
    except ReactionJsonReplayError:
        raise
    except Exception as exc:
        raise ReactionJsonReplayError(
            "reactionjson_replay_failed",
            failure_context={
                "failure_stage": "graph_finalization",
                "failure_detail": _exception_summary(exc),
            },
        ) from exc
    if any(atom.GetAtomicNum() == 0 for atom in replayed.GetAtoms()):
        raise ReactionJsonReplayError("reactionjson_unresolved_dummy_atom")
    _require_complete_unique_maps(replayed, reason="reactionjson_output_maps_invalid")

    mapped_fragments = _fragments(replayed, keep_maps=True)
    fragments = _fragments(replayed, keep_maps=False)
    expected = (
        sorted(_canonical_smiles(value) for value in expected_precursor_smiles)
        if expected_precursor_smiles is not None
        else None
    )
    if expected is not None and (not expected or not all(expected)):
        raise ReactionJsonReplayError("reactionjson_expected_precursors_invalid")
    if expected is not None and fragments != expected:
        raise ReactionJsonReplayError("reactionjson_expected_precursors_mismatch")
    extensions_used = sorted(
        {str(row["op"]) for row in rows if row["op"] in AUTOPLANNER_EXTENSIONS}
    )
    public_profile_compatible = not extensions_used
    audit = {
        "schema_version": REACTIONJSON_REPLAY_AUDIT_SCHEMA,
        "profile": (
            REACTIONJSON_PROFILE
            if public_profile_compatible
            else REACTIONJSON_EXTENSION_PROFILE
        ),
        "public_profile": REACTIONJSON_PROFILE,
        "public_profile_compatible": public_profile_compatible,
        "extensions_used": extensions_used,
        "upstream_public_commit": UPSTREAM_PUBLIC_COMMIT,
        "mapped_product_smiles": Chem.MolToSmiles(
            product, canonical=True, isomericSmiles=True
        ),
        "operation_count": len(rows),
        "resolved_operations": [_resolved_operation(row) for row in rows],
        "fresh_atom_maps_assigned": fresh_atom_maps,
        "primitive_counts": {
            key: int(Counter(row["op"] for row in rows).get(key, 0))
            for key in PRIMITIVES
        },
        "extension_counts": {
            key: int(Counter(row["op"] for row in rows).get(key, 0))
            for key in AUTOPLANNER_EXTENSIONS
        },
        "implicit_valence_completion_maps": completed_maps,
        "mapped_precursor_smiles": mapped_fragments,
        "precursor_smiles": fragments,
        "expected_precursor_smiles": expected or [],
        "expected_precursors_match": expected is not None,
        "accepted": True,
        "authority_scope": "external_structure_proposal_replay",
        "semantics": {
            "provisional_public_profile": public_profile_compatible,
            "autoplanner_extension_profile_used": bool(extensions_used),
            "deterministic_graph_edit_replay": True,
            "replay_grants_no_reaction_proof": True,
            "replay_grants_no_source_or_condition_authority": True,
            "unknown_fields_fail_closed": True,
        },
    }
    audit["content_sha256"] = _digest(audit)
    return audit


def _clear_invalid_bond_stereo_references(
    molecule: Chem.RWMol,
) -> list[dict[str, Any]]:
    """Clear stale RDKit stereo references before native finalization.

    RDKit stores E/Z reference atoms as raw atom indices. Replacing an alkene
    substituent can leave those indices pointing at atoms that are no longer
    neighbours of the double-bond endpoints. Passing that state to
    ``SetDoubleBondNeighborDirections`` can terminate Python inside RDKit
    instead of raising an exception. Clear only invalid references here. The
    caller then requires an explicit ``set_bond_stereo`` operation for every
    retained affected double bond, so process safety cannot silently weaken
    stereochemical provenance.
    """

    defined_stereo = {
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
    }
    invalidated: list[dict[str, Any]] = []
    for bond in molecule.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo not in defined_stereo:
            continue
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        references = tuple(int(value) for value in bond.GetStereoAtoms())
        begin_neighbours = {
            int(atom.GetIdx())
            for atom in molecule.GetAtomWithIdx(begin).GetNeighbors()
            if int(atom.GetIdx()) != end
        }
        end_neighbours = {
            int(atom.GetIdx())
            for atom in molecule.GetAtomWithIdx(end).GetNeighbors()
            if int(atom.GetIdx()) != begin
        }
        references_valid = (
            len(references) == 2
            and references[0] in begin_neighbours
            and references[1] in end_neighbours
        )
        if references_valid:
            continue
        begin_map = int(molecule.GetAtomWithIdx(begin).GetAtomMapNum())
        end_map = int(molecule.GetAtomWithIdx(end).GetAtomMapNum())
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        invalidated.append(
            {
                "map_a": begin_map,
                "map_b": end_map,
                "previous_stereo": str(stereo).removeprefix("STEREO"),
            }
        )
    return invalidated


def diagnose_reactionjson(
    *,
    mapped_product_smiles: str,
    operations: Iterable[Mapping[str, Any]],
    declared_precursor_smiles: Iterable[str] = (),
    reserved_atom_maps: Iterable[int] = (),
) -> dict[str, Any]:
    """Return actionable replay feedback without creating a second authority.

    ReactionJSON remains the only structural writer.  Model-declared precursor
    strings are compared for diagnostics only; callers must consume the
    deterministically replayed ``precursor_smiles`` when replay succeeds.
    """

    declared = sorted(
        value
        for item in declared_precursor_smiles
        if (value := _canonical_smiles(item))
    )
    try:
        audit = replay_reactionjson(
            mapped_product_smiles=mapped_product_smiles,
            operations=operations,
            expected_precursor_smiles=None,
            reserved_atom_maps=reserved_atom_maps,
        )
    except ReactionJsonReplayError as exc:
        return {
            "schema_version": "reactionjson_replay_diagnostic.v1",
            "replay_succeeded": False,
            "reason": str(exc),
            **reactionjson_failure_focus(exc),
            "declared_precursor_smiles": declared,
            "replayed_precursor_smiles": [],
            "declared_precursors_match": False,
        }
    replayed = list(audit.get("precursor_smiles") or [])
    return {
        "schema_version": "reactionjson_replay_diagnostic.v1",
        "replay_succeeded": True,
        "reason": "" if not declared or declared == replayed else "declared_precursors_disagree_with_replay",
        "declared_precursor_smiles": declared,
        "replayed_precursor_smiles": replayed,
        "mapped_replayed_precursor_smiles": list(
            audit.get("mapped_precursor_smiles") or []
        ),
        "declared_precursors_match": bool(declared and declared == replayed),
    }


def reactionjson_failure_focus(exc: ReactionJsonReplayError) -> dict[str, Any]:
    """Return only the operation-local fields needed for a causal retry."""

    focus: dict[str, Any] = {}
    operation_index = getattr(exc, "operation_index", None)
    if isinstance(operation_index, int):
        focus["operation_index"] = operation_index
    failed_operation = getattr(exc, "failed_operation", None)
    if isinstance(failed_operation, Mapping):
        focus["failed_operation"] = dict(failed_operation)
    failure_context = getattr(exc, "failure_context", None)
    if isinstance(failure_context, Mapping):
        focus.update(dict(failure_context))
    return focus


def _exception_summary(exc: Exception, *, maximum: int = 600) -> str:
    """Expose one bounded deterministic root cause to the next repair call."""

    value = f"{type(exc).__name__}: {exc}".strip()
    return value[: max(1, int(maximum))]


def _operation_failure_context(
    molecule: Chem.RWMol,
    row: Mapping[str, Any],
    exc: ReactionJsonReplayError,
) -> dict[str, Any]:
    if str(exc) != "reactionjson_aromatic_bond_requires_aromatic_atoms":
        return {}
    endpoint_aromaticity: dict[str, bool] = {}
    if row.get("op") == "add_bond":
        for key in ("map_a", "map_b"):
            aromatic = _mapped_atom_aromaticity(molecule, row.get(key))
            if aromatic is not None:
                endpoint_aromaticity[key] = aromatic
    elif row.get("op") == "add_group":
        anchor_aromatic = _mapped_atom_aromaticity(molecule, row.get("map_idx"))
        if anchor_aromatic is not None:
            endpoint_aromaticity["anchor"] = anchor_aromatic
        fragment = Chem.MolFromSmiles(str(row.get("fragment_smiles") or "").strip())
        if fragment is not None:
            dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummies) == 1 and dummies[0].GetDegree() == 1:
                endpoint_aromaticity["fragment_attachment"] = bool(
                    dummies[0].GetNeighbors()[0].GetIsAromatic()
                )
    return {
        "endpoint_aromaticity": endpoint_aromaticity,
        "allowed_orders": [1, 2, 3],
    }


def _mapped_atom_aromaticity(
    molecule: Chem.RWMol,
    map_value: Any,
) -> bool | None:
    try:
        map_idx = int(map_value)
    except (TypeError, ValueError):
        return None
    for atom in molecule.GetAtoms():
        if int(atom.GetAtomMapNum()) == map_idx:
            return bool(atom.GetIsAromatic())
    return None


def _require_complete_unique_maps(molecule: Chem.Mol, *, reason: str) -> None:
    maps = [
        atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    if not maps or any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
        raise ReactionJsonReplayError(reason)


def _map_index_for_audit(molecule: Chem.Mol, map_idx: int) -> int:
    for atom in molecule.GetAtoms():
        if int(atom.GetAtomMapNum()) == map_idx:
            return int(atom.GetIdx())
    raise ReactionJsonReplayError("reactionjson_map_not_found")


def _resolve_add_group_atom_maps(
    product: Chem.Mol,
    rows: Iterable[Mapping[str, Any]],
    *,
    reserved_atom_maps: Iterable[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Allocate every newly introduced atom from one replay-global namespace.

    Looking only at the current mutable graph can reuse the map of an atom that
    an earlier ``remove_group`` just deleted (for example Br:23 -> O:23).  Atom
    maps are provenance identities, so the initial product, caller-retained
    route namespace, and every explicit fragment map remain reserved for the
    entire replay even when an atom is removed before a later operation.
    """

    normalized = [dict(row) for row in rows]
    for row in normalized:
        if row.get("op") == "set_bond_stereo":
            row.pop("stereo_atom_maps", None)
    reserved = {
        int(atom.GetAtomMapNum())
        for atom in product.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    for value in reserved_atom_maps:
        try:
            map_idx = int(value)
        except (TypeError, ValueError):
            raise ReactionJsonReplayError("reactionjson_reserved_map_invalid") from None
        if map_idx <= 0:
            raise ReactionJsonReplayError("reactionjson_reserved_map_invalid")
        reserved.add(map_idx)

    fragments: dict[int, Chem.Mol] = {}
    explicit_new_maps: set[int] = set()
    for index, row in enumerate(normalized):
        if row.get("op") != "add_group":
            continue
        fragment = Chem.MolFromSmiles(str(row.get("fragment_smiles") or "").strip())
        if fragment is None:
            # Keep the existing operation-local typed failure and index.
            continue
        fragments[index] = fragment
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue
            map_idx = int(atom.GetAtomMapNum())
            if map_idx <= 0:
                continue
            if map_idx in reserved or map_idx in explicit_new_maps:
                raise ReactionJsonReplayError(
                    "reactionjson_fragment_map_collision",
                    operation_index=index,
                    failed_operation=row,
                )
            explicit_new_maps.add(map_idx)

    used = reserved | explicit_new_maps
    next_map = max(used, default=0) + 1
    assigned: list[int] = []
    for index, fragment in fragments.items():
        fresh_for_fragment: list[int] = []
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() == 0 or int(atom.GetAtomMapNum()) > 0:
                continue
            while next_map in used:
                next_map += 1
            atom.SetAtomMapNum(next_map)
            used.add(next_map)
            assigned.append(next_map)
            fresh_for_fragment.append(next_map)
            next_map += 1
        if fresh_for_fragment:
            normalized[index]["_fresh_atom_maps"] = fresh_for_fragment
            # Execute the caller's original fragment in this replay so legacy
            # ``order`` overrides retain their implicit-H semantics.  Persist
            # a separately resolved public fragment for later route replay.
            resolved_fragment, encodes_order = _resolved_add_group_fragment(
                fragment,
                row=normalized[index],
            )
            normalized[index]["_resolved_fragment_smiles"] = resolved_fragment
            normalized[index]["_resolved_fragment_encodes_order"] = encodes_order
    return normalized, assigned


def _public_operation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if not str(key).startswith("_")
    }


def _resolved_operation(row: Mapping[str, Any]) -> dict[str, Any]:
    operation = _public_operation(row)
    resolved_fragment = str(row.get("_resolved_fragment_smiles") or "")
    if resolved_fragment:
        operation["fragment_smiles"] = resolved_fragment
    if row.get("_resolved_fragment_encodes_order") is True:
        operation.pop("order", None)
    return operation


def _resolved_add_group_fragment(
    fragment: Chem.Mol,
    *,
    row: Mapping[str, Any],
) -> tuple[str, bool]:
    """Serialize assigned maps and fold a legacy order into the dummy bond."""

    resolved = Chem.Mol(fragment)
    order = row.get("order")
    encodes_order = False
    if order is not None:
        try:
            numeric_order = float(order)
        except (TypeError, ValueError):
            numeric_order = 0.0
        bond_type = {
            1.0: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2.0: Chem.BondType.DOUBLE,
            3.0: Chem.BondType.TRIPLE,
        }.get(numeric_order)
        dummies = [atom for atom in resolved.GetAtoms() if atom.GetAtomicNum() == 0]
        if bond_type is not None and len(dummies) == 1 and dummies[0].GetDegree() == 1:
            dummy = dummies[0]
            neighbor = dummy.GetNeighbors()[0]
            bond = resolved.GetBondBetweenAtoms(dummy.GetIdx(), neighbor.GetIdx())
            bond.SetBondType(bond_type)
            bond.SetIsAromatic(bond_type == Chem.BondType.AROMATIC)
            for atom in resolved.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            try:
                Chem.SanitizeMol(resolved)
            except Exception:
                pass
            else:
                encodes_order = True
    return (
        Chem.MolToSmiles(resolved, canonical=True, isomericSmiles=True),
        encodes_order,
    )


def _fragments(molecule: Chem.Mol, *, keep_maps: bool) -> list[str]:
    values: list[str] = []
    for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True):
        mapped_smiles = Chem.MolToSmiles(
            fragment,
            canonical=True,
            isomericSmiles=True,
        )
        # Atom maps participate in RDKit's canonical atom ordering.  Clearing
        # them directly on an already detached chiral fragment can leave the
        # stored tetrahedral parity relative to the old ordering and serialize
        # the opposite stereoisomer.  Reparse the mapped serialization first,
        # then remove maps in that fresh molecule so both representations name
        # the same exact structure.
        values.append(
            mapped_smiles if keep_maps else _canonical_smiles(mapped_smiles)
        )
    return sorted(values)


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    # Atom maps can make constitutionally identical substituents appear
    # distinct while a mapped edit is being replayed.  Once maps are removed,
    # recompute stereo so a tetrahedral tag that is no longer physical does
    # not leak into the canonical precursor identity.
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
