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


@dataclass(frozen=True, slots=True)
class MaterializedReaction:
    """One host-materialized retrosynthetic reaction."""

    product_smiles: str
    mapped_product_smiles: str
    precursor_smiles: tuple[str, ...]
    mapped_precursor_smiles: tuple[str, ...]
    reaction_operations: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]


class RouteJSONCompiler:
    """Compile model edit programs into deterministic route graph states."""

    def compile_step(
        self,
        *,
        mapped_product_smiles: str,
        operations: Iterable[Mapping[str, Any]],
        expected_product_smiles: str = "",
    ) -> MaterializedReaction:
        mapped_product = str(mapped_product_smiles or "").strip()
        normalized = normalize_reaction_operations(operations)
        if not mapped_product or not normalized:
            raise ReactionJsonReplayError("routejson_compiler_step_input_invalid")
        audit = replay_reactionjson(
            mapped_product_smiles=mapped_product,
            operations=normalized,
            expected_precursor_smiles=None,
        )
        product = str(audit.get("mapped_product_smiles") or "")
        canonical_product = _canonical_smiles(product)
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
                    "mapped_product_declared_smiles": _canonical_smiles(product),
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
        return MaterializedReaction(
            product_smiles=canonical_product,
            mapped_product_smiles=product,
            precursor_smiles=precursors,
            mapped_precursor_smiles=mapped_precursors,
            reaction_operations=tuple(dict(row) for row in normalized),
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
    ) -> tuple[MaterializedReaction, ...]:
        """Replay a topologically ordered RouteJSON DAG.

        A linear validator only carries the immediately previous precursor
        set. Real retrosyntheses branch, so sibling precursors exposed by an
        earlier step must remain available while another branch is expanded.
        Every non-root row is bound to one host-derived open structure before
        its ReactionJSON edit is applied.
        """

        rows = [dict(value) for value in steps if isinstance(value, Mapping)]
        if len(rows) < max(1, int(minimum_depth)):
            raise ReactionJsonReplayError("routejson_compiler_route_too_short")
        target_mapped = str(mapped_target_smiles or "").strip()
        target = _canonical_smiles(target_mapped)
        if not target:
            raise ReactionJsonReplayError("routejson_compiler_target_invalid")

        compiled: list[MaterializedReaction] = []
        seen_products: set[str] = set()
        available: list[tuple[str, str]] = []
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
                current_product, current_mapped = match
                declaration_mismatch = declared_product != current_product
            if current_product in seen_products:
                raise ReactionJsonReplayError("routejson_compiler_product_cycle")

            materialized = self.compile_step(
                mapped_product_smiles=current_mapped,
                operations=row.get("reaction_operations") or (),
                expected_product_smiles=current_product,
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
            if any(
                precursor == current_product or precursor in seen_products
                for precursor in materialized.precursor_smiles
            ):
                raise ReactionJsonReplayError("routejson_compiler_product_cycle")
            compiled.append(materialized)
            seen_products.add(current_product)
            available.extend(
                zip(
                    materialized.precursor_smiles,
                    materialized.mapped_precursor_smiles,
                    strict=True,
                )
            )
        return tuple(compiled)

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
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


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
    available: Iterable[tuple[str, str]],
) -> tuple[str, str] | None:
    exact = [
        (canonical, mapped)
        for canonical, mapped in available
        if canonical == declared_product
    ]
    candidates = exact or [
        (canonical, mapped)
        for canonical, mapped in available
        if _constitution_smiles(canonical) == _constitution_smiles(declared_product)
    ]
    if not candidates:
        return None
    declared_mapped = _canonical_mapped_smiles(declared_mapped_product)
    if declared_mapped:
        mapped_matches = [
            pair
            for pair in candidates
            if _canonical_mapped_smiles(pair[1]) == declared_mapped
        ]
        if len(mapped_matches) == 1:
            return mapped_matches[0]
    unique = {
        (canonical, _canonical_mapped_smiles(mapped)): (canonical, mapped)
        for canonical, mapped in candidates
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


__all__ = ["MaterializedReaction", "RouteJSONCompiler"]
