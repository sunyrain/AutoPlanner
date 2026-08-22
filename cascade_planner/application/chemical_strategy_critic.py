"""Independent, evidence-free forward critic for strategy hypotheses.

The critic runs before literature acquisition.  It may reject a structural or
replay contradiction, but uncertainty about an unfamiliar reaction remains an
explicit exploration diagnostic rather than being converted into missing-proof
failure.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable, Mapping

from rdkit import Chem

from cascade_planner.application.reactionjson_replay import (
    ReactionJsonReplayError,
    replay_reactionjson,
)
from cascade_planner.application.strategy_contract import (
    normalize_reaction_operations,
    normalize_strategy_card,
    reaction_edit_digest,
)
from cascade_planner.routes.admission import (
    replayed_external_atom_deficit_is_bound,
)


CHEMICAL_STRATEGY_CRITIC_SCHEMA = "chemical_strategy_critic.v1"


def critique_strategy_candidate(
    *,
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
    strategy_card: Mapping[str, Any] | None = None,
    reaction_operations: Iterable[Mapping[str, Any]] = (),
    reactionjson_audit: Mapping[str, Any] | None = None,
    reaction_family: str = "",
    conditions: Iterable[Any] = (),
    catalyst: str = "",
    enzyme: str = "",
    is_strategy_defining_step: bool = False,
) -> dict[str, Any]:
    """Forward-audit one proposed edge without consulting evidence sources."""

    product = _canonical_smiles(product_smiles)
    precursors = tuple(
        canonical
        for value in precursor_smiles or ()
        if (canonical := _canonical_smiles(value))
    )
    operations = normalize_reaction_operations(reaction_operations)
    card = normalize_strategy_card(
        strategy_card or {},
        reaction_operations=operations,
    )
    blocking: list[str] = []
    uncertainties: list[str] = []
    observations: list[str] = []

    if not product:
        blocking.append("critic_product_identity_invalid")
    if not precursors:
        blocking.append("critic_precursor_identity_invalid")
    atom_inventory_deficit = bool(
        product and precursors and _atom_inventory_deficit(product, precursors)
    )

    replay_audit: dict[str, Any] = {}
    if product and operations:
        supplied_replay = dict(reactionjson_audit or {})
        supplied_mapped_product = str(
            supplied_replay.get("mapped_product_smiles") or ""
        ).strip()
        replay_product = _mapped_smiles(product)
        if supplied_mapped_product:
            # Reaction operations are expressed in the atom-map namespace of
            # the RouteJSON compiler output.  Canonicalizing and assigning a
            # fresh 1..N map sequence changes that namespace and creates false
            # map_not_found/bond_missing failures.  Reuse the host-bound
            # mapped product only after independently binding it to the same
            # canonical product structure; replay below remains authoritative.
            if _canonical_unmapped_smiles(supplied_mapped_product) == product:
                replay_product = supplied_mapped_product
                observations.append("reactionjson_host_map_namespace_reused")
            elif (
                supplied_replay.get("mapped_product_stereo_normalized") is True
                and _canonical_constitution_smiles(supplied_mapped_product)
                == _canonical_constitution_smiles(product)
            ):
                # RouteJSONCompiler may deliberately normalize a stale local
                # stereo label after the graph edit has changed symmetry.  In
                # that case the mapped graph is still the host-bound map
                # namespace and must be replayed as supplied.  Falling back to
                # a freshly numbered product here creates a false map_not_found
                # and quarantines an otherwise replayable candidate.
                replay_product = supplied_mapped_product
                observations.append(
                    "reactionjson_host_map_namespace_reused_stereo_normalized"
                )
            else:
                blocking.append("critic_reactionjson_product_binding_mismatch")
        try:
            replay_audit = replay_reactionjson(
                mapped_product_smiles=replay_product,
                operations=operations,
                expected_precursor_smiles=precursors,
            )
            observations.append("reaction_operations_forward_replay_matched")
        except ReactionJsonReplayError as exc:
            blocking.append("critic_reaction_operations_replay_failed")
            replay_audit = {"accepted": False, "reason": str(exc)}
    elif is_strategy_defining_step and card.get("reaction_edit_digest"):
        blocking.append("critic_strategy_edit_missing_from_step")
    else:
        uncertainties.append("critic_reaction_operations_not_supplied")

    if atom_inventory_deficit:
        if _external_atom_source_is_bound(
            product=product,
            precursors=precursors,
            operations=operations,
            supplied_audit=dict(reactionjson_audit or {}),
            replay_audit=replay_audit,
        ):
            observations.append("critic_external_atom_source_bound_by_reactionjson")
            uncertainties.append("critic_external_atom_source_requires_validation")
        else:
            blocking.append("critic_atom_provenance_deficit")

    supplied_card = dict(strategy_card or {})
    expected_edit = str(supplied_card.get("reaction_edit_digest") or "")
    observed_edit = reaction_edit_digest(operations)
    if (
        is_strategy_defining_step
        and expected_edit
        and observed_edit
        and expected_edit != observed_edit
    ):
        blocking.append("critic_strategy_edit_digest_mismatch")

    domain = str(card.get("execution_domain") or "chemical")
    if domain in {"enzymatic", "whole_cell", "hybrid"} and not str(enzyme).strip():
        uncertainties.append("critic_enzyme_identity_or_capability_missing")
    if domain == "chemical" and enzyme:
        uncertainties.append("critic_execution_domain_enzyme_mismatch")

    product_mol = Chem.MolFromSmiles(product) if product else None
    if product_mol is not None:
        stereo_count = len(
            Chem.FindMolChiralCenters(
                product_mol,
                includeUnassigned=True,
                includeCIP=False,
            )
        )
        if stereo_count and not str(card.get("stereochemical_plan") or "").strip():
            uncertainties.append("critic_stereochemical_control_unspecified")
        if _multiple_similar_reactive_sites(product_mol) and not operations:
            uncertainties.append("critic_site_selectivity_not_structurally_bound")

    condition_text = " ".join(
        [str(reaction_family), str(catalyst), str(enzyme), *(str(v) for v in conditions)]
    ).lower()
    conflicts = [
        str(value).strip().lower()
        for value in card.get("functional_group_conflicts") or []
        if str(value).strip()
    ]
    if conflicts and not any(token in condition_text for token in ("protect", "selectiv", "screen")):
        uncertainties.append("critic_functional_group_conflict_mitigation_unspecified")
    if not str(card.get("convergence_plan") or "").strip():
        uncertainties.append("critic_convergence_plan_unspecified")
    if not str(card.get("key_forward_transformation") or "").strip():
        uncertainties.append("critic_key_forward_transformation_unspecified")

    status = "rejected" if blocking else "uncertain" if uncertainties else "passed"
    result = {
        "schema_version": CHEMICAL_STRATEGY_CRITIC_SCHEMA,
        "status": status,
        "accepted": not blocking,
        "strategy_id": str(card.get("strategy_id") or ""),
        "strategy_digest": str(card.get("strategy_digest") or ""),
        "reaction_edit_digest": observed_edit,
        "blocking_reasons": sorted(set(blocking)),
        "uncertainties": sorted(set(uncertainties)),
        "observations": sorted(set(observations)),
        "checks": {
            "atom_provenance": True,
            "forward_replay": True,
            "strategy_adherence": True,
            "functional_group_compatibility": True,
            "chemoselectivity": True,
            "stereochemistry": True,
            "sequence_ordering": True,
            "enzyme_identity_and_capability": True,
        },
        "strategy_defining_step": bool(is_strategy_defining_step),
        "reactionjson_audit": replay_audit,
        "semantics": {
            "runs_before_evidence_acquisition": True,
            "grants_no_reaction_proof": True,
            "grants_no_source_authority": True,
            "unknown_reaction_class_remains_exploration_visible": True,
            "routejson_atom_map_namespace_is_preserved": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )


def _mapped_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for index, atom in enumerate(molecule.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_unmapped_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_constitution_smiles(value: Any) -> str:
    """Canonical constitution identity used only for stereo-normalized replay."""

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in molecule.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _atom_inventory_deficit(product: str, precursors: Iterable[str]) -> bool:
    required = _inventory(product)
    available: Counter[int] = Counter()
    for precursor in precursors:
        available.update(_inventory(precursor))
    return any(available[number] < count for number, count in required.items())


def _external_atom_source_is_bound(
    *,
    product: str,
    precursors: Iterable[str],
    operations: Iterable[Mapping[str, Any]],
    supplied_audit: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
) -> bool:
    """Accept only deficits named by a replayed external-atom graph edit."""

    if (
        supplied_audit.get("external_atom_source_required") is not True
        or supplied_audit.get("external_atom_source_grants_reaction_proof") is not False
        or replay_audit.get("accepted") is not True
    ):
        return False
    mapped_product = str(supplied_audit.get("mapped_product_smiles") or "")
    return replayed_external_atom_deficit_is_bound(
        product,
        precursors,
        mapped_product_smiles=mapped_product,
        reaction_operations=operations,
    )


def _inventory(smiles: str) -> Counter[int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return Counter()
    return Counter(
        atom.GetAtomicNum()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    )


def _multiple_similar_reactive_sites(molecule: Chem.Mol) -> bool:
    counts = Counter(
        (
            atom.GetAtomicNum(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
        )
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() in {6, 7, 8, 16}
    )
    return any(count >= 3 for count in counts.values())


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["CHEMICAL_STRATEGY_CRITIC_SCHEMA", "critique_strategy_candidate"]
