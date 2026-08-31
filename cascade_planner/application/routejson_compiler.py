"""Programmatic ReactionJSON -> RouteJSON materialization.

The SynthEx paper treats ReactionJSON as an executable graph-edit program.  The
LLM proposes the edit list; the host applies it to the current mapped product
and owns every downstream structure and atom-map.  This module is the narrow
compiler boundary used by the sequential director.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable, Mapping
from typing import Any

from .reactionjson_replay import (
    ReactionJsonReplayError,
    replay_reactionjson,
)
from .strategy_contract import normalize_reaction_operations
from cascade_planner.routes.admission import (
    replayed_external_atom_deficit_is_bound,
)


@dataclass(frozen=True, slots=True)
class MaterializedReaction:
    """One host-materialized retrosynthetic reaction."""

    product_smiles: str
    mapped_product_smiles: str
    precursor_smiles: tuple[str, ...]
    mapped_precursor_smiles: tuple[str, ...]
    reaction_operations: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MappedOpenPrecursor:
    """One host-owned open RouteJSON frontier boundary."""

    product_smiles: str
    mapped_product_smiles: str


@dataclass(frozen=True, slots=True)
class RouteGraphReplayState:
    """Compiled reactions plus the exact mapped frontier left open by replay."""

    reactions: tuple[MaterializedReaction, ...]
    open_precursors: tuple[MappedOpenPrecursor, ...]
    parent_step_indices: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class _OpenRouteOccurrence:
    """One unconsumed frontier occurrence with its branch-local ancestry."""

    product_smiles: str
    mapped_product_smiles: str
    ancestor_products: frozenset[str]
    producer_step_index: int


class RouteJSONCompiler:
    """Compile model edit programs into deterministic route graph states."""

    def compile_step(
        self,
        *,
        mapped_product_smiles: str,
        operations: Iterable[Mapping[str, Any]],
        expected_product_smiles: str = "",
        reserved_atom_maps: Iterable[int] = (),
    ) -> MaterializedReaction:
        mapped_product = str(mapped_product_smiles or "").strip()
        normalized = normalize_reaction_operations(operations)
        if not mapped_product or not normalized:
            raise ReactionJsonReplayError("routejson_compiler_step_input_invalid")
        audit = replay_reactionjson(
            mapped_product_smiles=mapped_product,
            operations=normalized,
            expected_precursor_smiles=None,
            reserved_atom_maps=reserved_atom_maps,
        )
        product = str(audit.get("mapped_product_smiles") or "")
        declared_canonical_product = _canonical_smiles_preserving_declared_stereo(
            product
        )
        canonical_product = _canonical_smiles(product)
        if (
            declared_canonical_product
            and declared_canonical_product != canonical_product
            and _constitution_smiles(declared_canonical_product)
            == _constitution_smiles(canonical_product)
        ):
            audit = {
                **dict(audit),
                "mapped_product_stereo_normalized": True,
                "mapped_product_declared_smiles": declared_canonical_product,
                "canonical_product_smiles": canonical_product,
            }
        if expected_product_smiles:
            expected_product = _canonical_smiles(expected_product_smiles)
            if canonical_product != expected_product:
                if _constitution_smiles(product) != _constitution_smiles(
                    expected_product_smiles
                ):
                    raise ReactionJsonReplayError(
                        "routejson_compiler_product_mismatch"
                    )
                # A mapped replay fragment can retain a stale/non-physical
                # chiral tag after its local symmetry changes.  The host's
                # canonical unmapped precursor is the product identity; keep
                # the mapped graph for subsequent edits and record the stereo
                # normalization instead of rejecting the same constitution.
                canonical_product = expected_product
                audit = {
                    **dict(audit),
                    "mapped_product_stereo_normalized": True,
                    "mapped_product_declared_smiles": declared_canonical_product,
                    "canonical_product_smiles": expected_product,
                }
        precursors = tuple(
            str(value)
            for value in audit.get("precursor_smiles") or []
            if str(value)
        )
        raw_mapped_precursors = tuple(
            str(value)
            for value in audit.get("mapped_precursor_smiles") or []
            if str(value)
        )
        mapped_precursors = _align_mapped_precursors(
            precursors,
            raw_mapped_precursors,
        )
        if not precursors or len(precursors) != len(mapped_precursors):
            raise ReactionJsonReplayError("routejson_compiler_precursor_output_invalid")
        resolved_operations = tuple(
            dict(row)
            for row in audit.get("resolved_operations") or normalized
            if isinstance(row, Mapping)
        )
        if replayed_external_atom_deficit_is_bound(
            canonical_product,
            precursors,
            mapped_product_smiles=product,
            reaction_operations=resolved_operations,
        ):
            # This is the single host-owned binding for product atoms supplied
            # by an omitted forward reagent or donor.  Every RouteJSON path
            # (Builder, Editor, linear, or DAG) consumes this compiler audit,
            # so canonical admission never depends on a caller copying the
            # same derived status fields correctly.
            audit = {
                **dict(audit),
                "external_atom_source_required": True,
                "external_atom_source_status": (
                    "declared_graph_edit_requires_validation"
                ),
                "external_atom_source_grants_reaction_proof": False,
            }
        return MaterializedReaction(
            product_smiles=canonical_product,
            mapped_product_smiles=product,
            precursor_smiles=precursors,
            mapped_precursor_smiles=mapped_precursors,
            reaction_operations=resolved_operations,
            audit=audit,
        )

    def compile_linear_route(
        self,
        *,
        mapped_target_smiles: str,
        steps: Iterable[Mapping[str, Any]],
        minimum_depth: int = 1,
    ) -> tuple[MaterializedReaction, ...]:
        """Replay a linear route using only host-derived downstream products."""

        rows = [dict(value) for value in steps if isinstance(value, Mapping)]
        if len(rows) < max(1, int(minimum_depth)):
            raise ReactionJsonReplayError("routejson_compiler_route_too_short")
        current_mapped = str(mapped_target_smiles or "").strip()
        current_product = _canonical_smiles(current_mapped)
        if not current_product:
            raise ReactionJsonReplayError("routejson_compiler_target_invalid")
        compiled: list[MaterializedReaction] = []
        seen_products: set[str] = set()
        previous_precursors: tuple[str, ...] = ()
        previous_mapped_precursors: tuple[str, ...] = ()
        reserved_atom_maps = _mapped_atom_maps(current_mapped)
        for index, row in enumerate(rows):
            declared_product = _canonical_smiles(row.get("product_smiles"))
            declaration_mismatch = False
            if declared_product != current_product:
                if _constitution_smiles(declared_product) != _constitution_smiles(
                    current_product
                ):
                    raise ReactionJsonReplayError(
                        "routejson_compiler_chain_product_mismatch"
                    )
                # Host replay is authoritative.  A model may retain or
                # invent a stereochemical label after an explicit
                # clear_stereocenter edit; preserve the mismatch as an audit
                # fact while carrying the exact host-derived mapped product.
                declaration_mismatch = True
            if declared_product in seen_products:
                raise ReactionJsonReplayError("routejson_compiler_product_cycle")
            if index > 0:
                match = _match_precursor_identity(
                    declared_product,
                    previous_precursors,
                    previous_mapped_precursors,
                )
                if match is None:
                    raise ReactionJsonReplayError(
                        "routejson_compiler_product_not_previous_precursor"
                    )
                # The transition below normally already carries this pair;
                # resolving it again here keeps the chain contract explicit
                # and protects callers that provide a constitutionally equal
                # stereo declaration.
                current_product, current_mapped = match
            materialized = self.compile_step(
                mapped_product_smiles=current_mapped,
                operations=row.get("reaction_operations") or (),
                expected_product_smiles=current_product,
                reserved_atom_maps=reserved_atom_maps,
            )
            if declaration_mismatch:
                materialized = replace(
                    materialized,
                    audit={
                        **dict(materialized.audit),
                        "declared_product_smiles": declared_product,
                        "declared_product_matches_host": False,
                        "declared_product_mismatch_type": "stereochemistry_only",
                    },
                )
            compiled.append(materialized)
            seen_products.add(declared_product)
            previous_precursors = materialized.precursor_smiles
            previous_mapped_precursors = materialized.mapped_precursor_smiles
            reserved_atom_maps.update(
                _mapped_atom_maps(materialized.mapped_product_smiles)
            )
            for mapped_precursor in materialized.mapped_precursor_smiles:
                reserved_atom_maps.update(_mapped_atom_maps(mapped_precursor))
            if index + 1 < len(rows):
                next_product = _canonical_smiles(rows[index + 1].get("product_smiles"))
                match = _match_precursor_identity(
                    next_product,
                    previous_precursors,
                    previous_mapped_precursors,
                )
                if match is None:
                    raise ReactionJsonReplayError(
                        "routejson_compiler_next_product_not_previous_precursor"
                    )
                # Carry the exact host-derived precursor forward. The next
                # RouteJSON row may declare the same constitution with a
                # different/invented stereo label; that declaration is
                # audited at the next iteration, but never becomes structure
                # authority for replay.
                current_product, current_mapped = match
        return tuple(compiled)

    def compile_route_graph(
        self,
        *,
        mapped_target_smiles: str,
        steps: Iterable[Mapping[str, Any]],
        minimum_depth: int = 1,
        reserved_atom_maps: Iterable[int] = (),
    ) -> tuple[MaterializedReaction, ...]:
        """Replay a target-rooted RouteJSON DAG and return its reactions."""

        return self.compile_route_graph_state(
            mapped_target_smiles=mapped_target_smiles,
            steps=steps,
            minimum_depth=minimum_depth,
            reserved_atom_maps=reserved_atom_maps,
        ).reactions

    def compile_route_graph_state(
        self,
        *,
        mapped_target_smiles: str,
        steps: Iterable[Mapping[str, Any]],
        minimum_depth: int = 1,
        reserved_atom_maps: Iterable[int] = (),
    ) -> RouteGraphReplayState:
        """Replay a topologically ordered RouteJSON DAG.

        A linear validator only carries the immediately previous precursor
        set. Real retrosyntheses branch, so sibling precursors exposed by an
        earlier step must remain available while another branch is expanded.
        Every non-root row is bound to one host-derived open structure before
        its ReactionJSON edit is applied. The returned frontier is the same
        host-owned mapped state used for that binding; callers must never
        reconstruct it from model-declared ``mapped_product_smiles`` fields.
        """

        rows = [dict(value) for value in steps if isinstance(value, Mapping)]
        if len(rows) < max(1, int(minimum_depth)):
            raise ReactionJsonReplayError("routejson_compiler_route_too_short")
        target_mapped = str(mapped_target_smiles or "").strip()
        target = _canonical_smiles(target_mapped)
        if not target:
            raise ReactionJsonReplayError("routejson_compiler_target_invalid")

        compiled: list[MaterializedReaction] = []
        parent_step_indices: list[int | None] = []
        available: list[_OpenRouteOccurrence] = []
        reserved_atom_maps = {
            int(value)
            for value in reserved_atom_maps
            if int(value) > 0
        } | _mapped_atom_maps(target_mapped)
        for index, row in enumerate(rows):
            declared_product = _canonical_smiles(row.get("product_smiles"))
            if not declared_product:
                raise ReactionJsonReplayError(
                    "routejson_compiler_chain_product_mismatch"
                )
            declaration_mismatch = False
            if index == 0:
                current_product = target
                current_mapped = target_mapped
                current_ancestors: frozenset[str] = frozenset()
                parent_step_index: int | None = None
                if declared_product != target:
                    if _constitution_smiles(declared_product) != _constitution_smiles(
                        target
                    ):
                        raise ReactionJsonReplayError(
                            "routejson_compiler_chain_product_mismatch"
                        )
                    declaration_mismatch = True
            else:
                match = _match_route_graph_product(
                    declared_product=declared_product,
                    declared_mapped_product=str(
                        row.get("mapped_product_smiles") or ""
                    ),
                    available=available,
                )
                if match is None:
                    raise ReactionJsonReplayError(
                        "routejson_compiler_product_not_open_precursor"
                    )
                current_product = match.product_smiles
                current_mapped = match.mapped_product_smiles
                current_ancestors = match.ancestor_products
                parent_step_index = match.producer_step_index
                # The matched boundary is no longer open after this row is
                # expanded. Keeping consumed pairs in the frontier made it
                # impossible to return a truthful mapped open-precursor state
                # to an Editor retry.
                available.remove(match)
                declaration_mismatch = declared_product != current_product

            materialized = self.compile_step(
                mapped_product_smiles=current_mapped,
                operations=row.get("reaction_operations") or (),
                expected_product_smiles=current_product,
                reserved_atom_maps=reserved_atom_maps,
            )
            if declaration_mismatch:
                materialized = replace(
                    materialized,
                    audit={
                        **dict(materialized.audit),
                        "declared_product_smiles": declared_product,
                        "declared_product_matches_host": False,
                        "declared_product_mismatch_type": "stereochemistry_only",
                    },
                )
            branch_ancestors = current_ancestors | {current_product}
            if any(
                precursor in branch_ancestors
                for precursor in materialized.precursor_smiles
            ):
                raise ReactionJsonReplayError("routejson_compiler_product_cycle")
            compiled.append(materialized)
            parent_step_indices.append(parent_step_index)
            reserved_atom_maps.update(
                _mapped_atom_maps(materialized.mapped_product_smiles)
            )
            for mapped_precursor in materialized.mapped_precursor_smiles:
                reserved_atom_maps.update(_mapped_atom_maps(mapped_precursor))
            available.extend(
                _OpenRouteOccurrence(
                    product_smiles=precursor,
                    mapped_product_smiles=mapped_precursor,
                    ancestor_products=branch_ancestors,
                    producer_step_index=index,
                )
                for precursor, mapped_precursor in zip(
                    materialized.precursor_smiles,
                    materialized.mapped_precursor_smiles,
                    strict=True,
                )
            )
        return RouteGraphReplayState(
            reactions=tuple(compiled),
            open_precursors=tuple(
                MappedOpenPrecursor(
                    product_smiles=occurrence.product_smiles,
                    mapped_product_smiles=occurrence.mapped_product_smiles,
                )
                for occurrence in available
            ),
            parent_step_indices=tuple(parent_step_indices),
        )

    @staticmethod
    def assemble_route(
        compiled: Iterable[MaterializedReaction],
        *,
        metadata: Iterable[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Serialize host-materialized reactions into a complete RouteJSON."""

        meta_rows = [dict(value) for value in (metadata or ())]
        route: list[dict[str, Any]] = []
        for index, reaction in enumerate(compiled):
            meta = meta_rows[index] if index < len(meta_rows) else {}
            route.append(
                {
                    **meta,
                    "step_id": str(meta.get("step_id") or f"compiled:step:{index + 1}"),
                    "product_smiles": reaction.product_smiles,
                    "precursor_smiles": list(reaction.precursor_smiles),
                    "mapped_product_smiles": reaction.mapped_product_smiles,
                    "mapped_precursor_smiles": list(reaction.mapped_precursor_smiles),
                    "reaction_operations": [
                        dict(value) for value in reaction.reaction_operations
                    ],
                    "reactionjson_audit": dict(reaction.audit),
                }
            )
        return route


def _canonical_smiles(value: Any) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_smiles_preserving_declared_stereo(value: Any) -> str:
    """Remove maps without silently hiding a stale mapped stereo declaration."""

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _mapped_atom_maps(value: Any) -> set[int]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return set()
    return {
        int(atom.GetAtomMapNum())
        for atom in molecule.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }


def _constitution_smiles(value: Any) -> str:
    """Canonicalize constitution while intentionally ignoring stereochemistry."""

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in molecule.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _match_precursor_identity(
    declared_product: str,
    precursors: Iterable[str],
    mapped_precursors: Iterable[str],
) -> tuple[str, str] | None:
    """Match a declared next product to one host precursor.

    Exact isomeric identity is preferred. If the model only changed a
    stereochemical declaration, constitution identity is sufficient and the
    canonical unmapped/mapped pair emitted by the host is returned.
    """

    candidates = [
        (_canonical_smiles(value), str(mapped))
        for value, mapped in zip(precursors, mapped_precursors)
        if _canonical_smiles(value) and str(mapped)
    ]
    exact = [pair for pair in candidates if pair[0] == declared_product]
    if len(exact) == 1:
        return exact[0]
    declared_constitution = _constitution_smiles(declared_product)
    approximate = [
        pair
        for pair in candidates
        if _constitution_smiles(pair[0]) == declared_constitution
    ]
    return approximate[0] if len(approximate) == 1 else None


def _match_route_graph_product(
    *,
    declared_product: str,
    declared_mapped_product: str,
    available: Iterable[_OpenRouteOccurrence],
) -> _OpenRouteOccurrence | None:
    exact = [
        occurrence
        for occurrence in available
        if occurrence.product_smiles == declared_product
    ]
    candidates = exact or [
        occurrence
        for occurrence in available
        if _constitution_smiles(occurrence.product_smiles)
        == _constitution_smiles(declared_product)
    ]
    if not candidates:
        return None
    declared_mapped = _canonical_mapped_smiles(declared_mapped_product)
    if declared_mapped:
        mapped_matches = [
            occurrence
            for occurrence in candidates
            if _canonical_mapped_smiles(occurrence.mapped_product_smiles)
            == declared_mapped
        ]
        if len(mapped_matches) == 1:
            return mapped_matches[0]
    unique = {
        (
            occurrence.product_smiles,
            _canonical_mapped_smiles(occurrence.mapped_product_smiles),
        ): occurrence
        for occurrence in candidates
    }
    if len(unique) == 1:
        return next(iter(unique.values()))
    raise ReactionJsonReplayError(
        "routejson_compiler_ambiguous_parent_map_namespace"
    )


def _canonical_mapped_smiles(value: Any) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _align_mapped_precursors(
    precursor_smiles: Iterable[str],
    mapped_precursor_smiles: Iterable[str],
) -> tuple[str, ...]:
    """Pair mapped fragments with canonical fragments by molecular identity.

    ``reactionjson_replay`` emits both arrays in deterministic order, but the
    mapped and unmapped strings are sorted independently.  Lexicographic map
    prefixes can therefore place two arrays in different fragment order.  A
    compiler must repair that representation mismatch before carrying atom
    maps into the next model call.
    """

    mapped_rows = sorted(
        (
            {
                "mapped": str(mapped),
                "canonical": _canonical_smiles(mapped),
                "constitution": _constitution_smiles(mapped),
            }
            for mapped in mapped_precursor_smiles
            if _canonical_smiles(mapped)
        ),
        key=lambda row: str(row["mapped"]),
    )
    used: set[int] = set()
    aligned: list[str] = []
    for precursor in precursor_smiles:
        canonical = _canonical_smiles(precursor)
        exact = [
            index
            for index, row in enumerate(mapped_rows)
            if index not in used and row["canonical"] == canonical
        ]
        if exact:
            selected = exact[0]
        else:
            # Public replay may preserve a chiral tag on a mapped atom whose
            # substituents become constitutionally identical after a ring
            # opening.  The corresponding unmapped canonical precursor then
            # correctly drops that non-physical stereocenter.  Accept only a
            # unique constitution match; ambiguous stereoisomer/multiplicity
            # pairings remain fail-closed.
            constitution = _constitution_smiles(precursor)
            approximate = [
                index
                for index, row in enumerate(mapped_rows)
                if index not in used and row["constitution"] == constitution
            ]
            if len(approximate) != 1:
                return ()
            selected = approximate[0]
        used.add(selected)
        aligned.append(str(mapped_rows[selected]["mapped"]))
    if len(used) != len(mapped_rows):
        return ()
    return tuple(aligned)


__all__ = [
    "MappedOpenPrecursor",
    "MaterializedReaction",
    "RouteGraphReplayState",
    "RouteJSONCompiler",
]
