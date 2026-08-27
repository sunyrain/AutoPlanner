"""Shared fail-closed admission checks for retrosynthetic hyperedges."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
RETROSYNTHETIC_ADMISSION_SCHEMA = "retrosynthetic_candidate_admission.v1"
RETROSYNTHETIC_EDGE_IDENTITY_SCHEMA = "retrosynthetic_edge_identity.v1"
RETROSYNTHETIC_ADMISSION_RECORD_SCHEMA = "retrosynthetic_admission_record.v1"


@dataclass(frozen=True)
class RetrosyntheticAdmissionPolicy:
    """Cheap structural checks applied before a proposal enters search state."""

    hard_filter_element_inventory: bool = True
    max_tolerated_missing_heavy_atoms: int = 3
    strict_small_product_heavy_atom_threshold: int = 12
    hard_filter_large_atom_jump: bool = True
    large_atom_jump_threshold: int = 15
    hard_filter_self_loop: bool = True
    hard_filter_surplus_advanced_fragment: bool = True
    surplus_advanced_fragment_heavy_atom_threshold: int = 8


def audit_retrosynthetic_candidate(
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
    *,
    forbidden_return_smiles: Iterable[Any] = (),
    policy: RetrosyntheticAdmissionPolicy | None = None,
    mapped_reaction_smiles: Any = "",
    mapped_product_smiles: Any = "",
    reaction_operations: Iterable[Mapping[str, Any]] = (),
    reactionjson_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit one exact product/precursor multiset without granting proof.

    The allowance for a few missing product atoms reflects omitted transfer
    reagents in one-step model output. It is deliberately disabled for small
    products, where such an omission would amount to most of the structure.
    """

    active = policy or RetrosyntheticAdmissionPolicy()
    product = _canonical_smiles(product_smiles)
    raw_precursors = list(precursor_smiles or [])
    precursors = [_canonical_smiles(item) for item in raw_precursors]
    precursor_multiset = sorted(precursors)
    edge_identity = {
        "schema_version": RETROSYNTHETIC_EDGE_IDENTITY_SCHEMA,
        "product_smiles": product,
        # Sorting gives one identity to every ordering of the same exact
        # multiset while deliberately preserving duplicate components.
        "precursor_smiles_multiset": precursor_multiset,
    }
    edge_digest = _stable_digest(edge_identity)
    reasons: list[str] = []
    if not product or not raw_precursors or any(not item for item in precursors):
        reasons.append("invalid_or_missing_material")
    product_counts = _element_counts(product)
    precursor_counts: Counter[str] = Counter()
    for precursor in precursors:
        precursor_counts.update(_element_counts(precursor))
    if not product_counts or not precursor_counts:
        if "invalid_or_missing_material" not in reasons:
            reasons.append("invalid_or_missing_material")

    if active.hard_filter_self_loop and product and product in precursors:
        reasons.append("target_or_current_node_self_loop")
    forbidden = {
        canonical
        for canonical in (_canonical_smiles(item) for item in forbidden_return_smiles)
        if canonical
    }
    ancestor_returns = sorted(set(precursors).intersection(forbidden - {product}))
    if ancestor_returns:
        reasons.append("ancestor_or_target_cycle")

    deficits = {
        element: count - int(precursor_counts.get(element, 0))
        for element, count in product_counts.items()
        if count > int(precursor_counts.get(element, 0))
    }
    missing_heavy_atoms = sum(deficits.values())
    product_heavy_atoms = sum(product_counts.values())
    precursor_heavy_atoms = sum(precursor_counts.values())
    effective_mapped_reaction = str(mapped_reaction_smiles or "").strip()
    if not effective_mapped_reaction:
        effective_mapped_reaction = _mapped_reaction_from_replay_audit(
            reactionjson_audit
        )
    mapped_contributing_precursor_indices = (
        _mapped_atom_contributing_precursor_indices(
            product,
            precursors,
            mapped_reaction_smiles=effective_mapped_reaction,
        )
    )
    surplus_advanced_fragments: list[str] = []
    if (
        active.hard_filter_surplus_advanced_fragment
        and product_counts
        and len(precursors) > 1
    ):
        component_counts = [_element_counts(precursor) for precursor in precursors]
        for index, counts in enumerate(component_counts):
            if (
                mapped_contributing_precursor_indices is not None
                and index in mapped_contributing_precursor_indices
            ):
                continue
            component_heavy_atoms = sum(counts.values())
            if (
                component_heavy_atoms
                < active.surplus_advanced_fragment_heavy_atom_threshold
            ):
                continue
            other_counts: Counter[str] = Counter()
            for other_index, other in enumerate(component_counts):
                if other_index != index:
                    other_counts.update(other)
            if all(
                int(other_counts.get(element, 0)) >= count
                for element, count in product_counts.items()
            ):
                surplus_advanced_fragments.append(precursors[index])
        if surplus_advanced_fragments:
            reasons.append("surplus_advanced_precursor_fragment")
    replayed_external_atom_deficit_bound = bool(
        deficits
        and dict(reactionjson_audit or {}).get("accepted") is True
        and dict(reactionjson_audit or {}).get("external_atom_source_required")
        is True
        and dict(reactionjson_audit or {}).get(
            "external_atom_source_grants_reaction_proof"
        )
        is False
        and replayed_external_atom_deficit_is_bound(
            product,
            precursors,
            mapped_product_smiles=mapped_product_smiles,
            reaction_operations=reaction_operations,
        )
    )
    if (
        active.hard_filter_element_inventory
        and deficits
        and not replayed_external_atom_deficit_bound
        and (
            missing_heavy_atoms > active.max_tolerated_missing_heavy_atoms
            or product_heavy_atoms <= active.strict_small_product_heavy_atom_threshold
        )
    ):
        reasons.append("element_inventory_not_conserved")
    if (
        active.hard_filter_large_atom_jump
        and product_heavy_atoms - precursor_heavy_atoms
        >= active.large_atom_jump_threshold
    ):
        reasons.append("large_atom_jump")

    return {
        "schema_version": RETROSYNTHETIC_ADMISSION_SCHEMA,
        "accepted": not reasons,
        "edge_digest": edge_digest,
        "edge_identity": edge_identity,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "precursor_smiles_multiset": precursor_multiset,
        "forbidden_return_smiles": sorted(forbidden),
        "ancestor_return_smiles": ancestor_returns,
        "product_element_counts": dict(sorted(product_counts.items())),
        "precursor_element_counts": dict(sorted(precursor_counts.items())),
        "element_deficits": dict(sorted(deficits.items())),
        "missing_product_heavy_atom_count": missing_heavy_atoms,
        "replayed_external_atom_deficit_bound": (
            replayed_external_atom_deficit_bound
        ),
        "external_atom_source_requires_validation": (
            replayed_external_atom_deficit_bound
        ),
        "product_heavy_atom_count": product_heavy_atoms,
        "precursor_heavy_atom_count": precursor_heavy_atoms,
        "mapped_reaction_atom_mapping_used": (
            mapped_contributing_precursor_indices is not None
        ),
        "mapped_atom_contributing_precursor_indices": sorted(
            mapped_contributing_precursor_indices or ()
        ),
        "surplus_advanced_precursor_fragments": surplus_advanced_fragments,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "search_admission_only": True,
            "not_reaction_proof": True,
            "precursor_multiplicity_preserved": True,
            "large_inventory_redundancy_is_not_joint_participation": True,
            "mapped_atom_contribution_overrides_inventory_redundancy": True,
            "small_salts_and_leaving_groups_are_exempt": True,
            "replayed_external_atom_edits_grant_no_reaction_proof": True,
        },
    }


def _mapped_atom_contributing_precursor_indices(
    product_smiles: str,
    precursor_smiles: Iterable[str],
    *,
    mapped_reaction_smiles: Any,
) -> frozenset[int] | None:
    """Bind mapped precursor atoms to the product without trusting labels.

    AiZynthFinder preserves atom maps for its selected template route.  Those
    maps are stronger than the inventory-only surplus heuristic: a large
    precursor that contributes even one mapped product atom is a real route
    input, not a disconnected distractor.  Invalid, forward-oriented, or
    incomplete mapping falls back to the existing fail-closed heuristic.
    """

    raw = str(mapped_reaction_smiles or "").strip()
    if raw.count(">>") != 1:
        return None
    mapped_product_text, mapped_precursors_text = raw.split(">>", 1)
    mapped_product = Chem.MolFromSmiles(mapped_product_text)
    mapped_precursors = Chem.MolFromSmiles(mapped_precursors_text)
    if mapped_product is None or mapped_precursors is None:
        return None
    expected_product = Chem.MolFromSmiles(product_smiles)
    if expected_product is None or _canonical_unmapped_molecule(
        mapped_product,
        isomeric_smiles=False,
    ) != _canonical_unmapped_molecule(
        expected_product,
        isomeric_smiles=False,
    ):
        return None
    product_maps = {
        int(atom.GetAtomMapNum())
        for atom in mapped_product.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    if not product_maps:
        return None

    mapped_components: dict[str, list[bool]] = {}
    for component in Chem.GetMolFrags(mapped_precursors, asMols=True):
        canonical = _canonical_unmapped_molecule(component)
        if not canonical:
            return None
        component_maps = {
            int(atom.GetAtomMapNum())
            for atom in component.GetAtoms()
            if int(atom.GetAtomMapNum()) > 0
        }
        mapped_components.setdefault(canonical, []).append(
            bool(product_maps.intersection(component_maps))
        )

    precursor_rows = list(precursor_smiles)
    expected_counts = Counter(precursor_rows)
    observed_counts = Counter(
        {
            canonical: len(values)
            for canonical, values in mapped_components.items()
        }
    )
    if expected_counts != observed_counts:
        return None

    consumed: Counter[str] = Counter()
    contributing: set[int] = set()
    for index, canonical in enumerate(precursor_rows):
        occurrence = consumed[canonical]
        consumed[canonical] += 1
        if mapped_components[canonical][occurrence]:
            contributing.add(index)
    return frozenset(contributing)


def _mapped_reaction_from_replay_audit(
    reactionjson_audit: Mapping[str, Any] | None,
) -> str:
    """Read the mapped reaction already emitted by deterministic Host replay."""

    audit = dict(reactionjson_audit or {})
    semantics = dict(audit.get("semantics") or {})
    if (
        audit.get("accepted") is not True
        or semantics.get("deterministic_graph_edit_replay") is not True
    ):
        return ""
    product = str(audit.get("mapped_product_smiles") or "").strip()
    precursors = [
        str(value).strip()
        for value in audit.get("mapped_precursor_smiles") or []
        if str(value).strip()
    ]
    if not product or not precursors:
        return ""
    return product + ">>" + ".".join(precursors)


def _canonical_unmapped_molecule(
    molecule: Chem.Mol,
    *,
    isomeric_smiles: bool = True,
) -> str:
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(
        copy,
        canonical=True,
        isomericSmiles=isomeric_smiles,
    )


def replayed_external_atom_deficit_is_bound(
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
    *,
    mapped_product_smiles: Any,
    reaction_operations: Iterable[Mapping[str, Any]],
) -> bool:
    """Return whether replay edits exactly explain every missing product element.

    ``remove_group`` and ``change_atom`` are retrosynthetic operations.  They may
    remove product atoms that a forward reagent or donor must supply, so those
    atoms are not skeletal search precursors.  The allowance is structural only:
    it requires mapped atoms named by the replayed edit and never grants reaction
    proof, source authority, or stock closure.
    """

    product = _canonical_smiles(product_smiles)
    precursors = [
        canonical
        for value in precursor_smiles or ()
        if (canonical := _canonical_smiles(value))
    ]
    required = _element_counts(product)
    available: Counter[str] = Counter()
    for precursor in precursors:
        available.update(_element_counts(precursor))
    deficits = Counter(
        {
            element: count - int(available.get(element, 0))
            for element, count in required.items()
            if count > int(available.get(element, 0))
        }
    )
    if not deficits:
        return False

    molecule = Chem.MolFromSmiles(str(mapped_product_smiles or "").strip())
    if molecule is None:
        return False
    atom_by_map = {
        int(atom.GetAtomMapNum()): atom
        for atom in molecule.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    explicitly_external: Counter[str] = Counter()
    for raw in reaction_operations or ():
        operation = dict(raw)
        kind = str(operation.get("op") or "").strip().lower()
        if kind == "remove_group":
            map_indices = operation.get("map_indices") or ()
        elif kind == "change_atom":
            map_indices = (operation.get("map_idx"),)
        else:
            continue
        for raw_map in map_indices:
            try:
                atom = atom_by_map[int(raw_map)]
            except (KeyError, TypeError, ValueError):
                return False
            if atom.GetAtomicNum() > 1:
                explicitly_external[atom.GetSymbol()] += 1
    return all(
        explicitly_external[element] >= count for element, count in deficits.items()
    )


def retrosynthetic_admission_record(
    audit: dict[str, Any],
    *,
    stage: str,
    source: Any = "",
    model: Any = "",
    template: Any = None,
    candidate_index: int | None = None,
) -> dict[str, Any]:
    """Bind a host admission decision to its execution provenance.

    The full model/template payload can be large or non-JSON-native.  The
    record therefore stores a bounded human-readable reference plus a stable
    digest while retaining the exact canonical reaction-edge identity from
    :func:`audit_retrosynthetic_candidate`.
    """

    bound_audit = dict(audit or {})
    edge_identity = dict(bound_audit.get("edge_identity") or {})
    edge_digest = str(bound_audit.get("edge_digest") or _stable_digest(edge_identity))
    template_ref = _bounded_template_reference(template)
    record_identity = {
        "stage": str(stage or "unknown"),
        "edge_digest": edge_digest,
        "source": _bounded_text(source),
        "model": _bounded_text(model),
        "template_digest": template_ref.get("digest", ""),
        "candidate_index": candidate_index,
    }
    return {
        "schema_version": RETROSYNTHETIC_ADMISSION_RECORD_SCHEMA,
        "record_id": "retro-admission:" + _stable_digest(record_identity)[:24],
        "stage": record_identity["stage"],
        "decision": "accepted" if bound_audit.get("accepted") is True else "rejected",
        "accepted": bound_audit.get("accepted") is True,
        "edge_digest": edge_digest,
        "edge_identity": edge_identity,
        "product_smiles": str(bound_audit.get("product_smiles") or ""),
        "precursor_smiles_multiset": list(
            bound_audit.get("precursor_smiles_multiset") or []
        ),
        "reasons": [str(item) for item in bound_audit.get("reasons") or []],
        "source": record_identity["source"],
        "model": record_identity["model"],
        "template": template_ref,
        "candidate_index": candidate_index,
        "host_audit_schema": str(bound_audit.get("schema_version") or ""),
        "host_audit_authority": True,
    }


def _canonical_smiles(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _element_counts(smiles: str) -> Counter[str]:
    molecule = Chem.MolFromSmiles(smiles or "")
    if molecule is None:
        return Counter()
    return Counter(
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    )


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bounded_text(value: Any, *, limit: int = 512) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _bounded_template_reference(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        fields = {
            str(key): _bounded_text(value.get(key))
            for key in (
                "template_id",
                "id",
                "model_full_name",
                "model_name",
                "source_model",
                "source",
                "reaction_family",
                "reaction_class",
                "template",
                "reaction_smarts",
                "retro_template",
                "retron",
            )
            if value.get(key) not in (None, "")
        }
        return {
            "kind": "mapping",
            "digest": _stable_digest({"kind": "mapping", "fields": fields}),
            "fields": fields,
        }
    bounded_value = _bounded_text(value)
    return {
        "kind": type(value).__name__ if value is not None else "none",
        "digest": _stable_digest(
            {
                "kind": type(value).__name__ if value is not None else "none",
                "value": bounded_value,
            }
        ),
        "value": bounded_value,
    }
