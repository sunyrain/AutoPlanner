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

from .reactionjson_primitives import PRIMITIVES
from .reactionjson_primitives import ReactionJsonReplayError
from .reactionjson_primitives import apply_operation
from .reactionjson_primitives import complete_edited_atom_valences
from .reactionjson_primitives import normalize_operation
from .reactionjson_primitives import valence_affected_maps

RDLogger.DisableLog("rdApp.*")
REACTIONJSON_PROFILE = "reactionjson_public_profile.2026-08-17.v1"
REACTIONJSON_REPLAY_AUDIT_SCHEMA = "reactionjson_replay_audit.v1"
UPSTREAM_PUBLIC_COMMIT = "5f41a6b21e3906fde93e84c88bb91f9dc4d37e6f"


def replay_reactionjson(
    *,
    mapped_product_smiles: str,
    operations: Iterable[Mapping[str, Any]],
    expected_precursor_smiles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Replay ten bounded primitives and return a content-bound audit."""

    product = Chem.MolFromSmiles(str(mapped_product_smiles or "").strip())
    if product is None:
        raise ReactionJsonReplayError("reactionjson_product_invalid")
    _require_complete_unique_maps(product, reason="reactionjson_product_maps_invalid")
    rows = [normalize_operation(value) for value in operations]
    if not rows or len(rows) > 128:
        raise ReactionJsonReplayError("reactionjson_operation_count_invalid")
    editable = Chem.RWMol(product)
    valence_completion_maps: set[int] = set()
    explicit_h_maps = {
        int(row["map_idx"])
        for row in rows
        if row["op"] == "set_explicit_h"
    }
    for operation_index, row in enumerate(rows):
        try:
            valence_completion_maps.update(valence_affected_maps(editable, row))
            editable = apply_operation(editable, row)
        except ReactionJsonReplayError as exc:
            raise ReactionJsonReplayError(
                str(exc),
                operation_index=operation_index,
                failed_operation=row,
                failure_context=_operation_failure_context(editable, row, exc),
            ) from exc
        except Exception as exc:
            raise ReactionJsonReplayError(
                "reactionjson_replay_failed",
                operation_index=operation_index,
                failed_operation=row,
            ) from exc
    try:
        completed_maps = complete_edited_atom_valences(
            editable,
            map_indices=valence_completion_maps - explicit_h_maps,
        )
        Chem.SetDoubleBondNeighborDirections(editable)
        replayed = editable.GetMol()
        Chem.SanitizeMol(replayed)
        Chem.AssignStereochemistry(replayed, cleanIt=True, force=True)
    except ReactionJsonReplayError:
        raise
    except Exception as exc:
        raise ReactionJsonReplayError("reactionjson_replay_failed") from exc
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
    audit = {
        "schema_version": REACTIONJSON_REPLAY_AUDIT_SCHEMA,
        "profile": REACTIONJSON_PROFILE,
        "upstream_public_commit": UPSTREAM_PUBLIC_COMMIT,
        "mapped_product_smiles": Chem.MolToSmiles(
            product, canonical=True, isomericSmiles=True
        ),
        "operation_count": len(rows),
        "primitive_counts": {
            key: int(Counter(row["op"] for row in rows).get(key, 0))
            for key in PRIMITIVES
        },
        "implicit_valence_completion_maps": completed_maps,
        "mapped_precursor_smiles": mapped_fragments,
        "precursor_smiles": fragments,
        "expected_precursor_smiles": expected or [],
        "expected_precursors_match": expected is not None,
        "accepted": True,
        "authority_scope": "external_structure_proposal_replay",
        "semantics": {
            "provisional_public_profile": True,
            "deterministic_graph_edit_replay": True,
            "replay_grants_no_reaction_proof": True,
            "replay_grants_no_source_or_condition_authority": True,
            "unknown_fields_fail_closed": True,
        },
    }
    audit["content_sha256"] = _digest(audit)
    return audit


def diagnose_reactionjson(
    *,
    mapped_product_smiles: str,
    operations: Iterable[Mapping[str, Any]],
    declared_precursor_smiles: Iterable[str] = (),
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


def _fragments(molecule: Chem.Mol, *, keep_maps: bool) -> list[str]:
    values: list[str] = []
    for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True):
        if not keep_maps:
            for atom in fragment.GetAtoms():
                atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        values.append(smiles if keep_maps else _canonical_smiles(smiles))
    return sorted(values)


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
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
