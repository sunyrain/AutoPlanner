"""Programmatic ReactionJSON -> RouteJSON materialization.

The SynthEx paper treats ReactionJSON as an executable graph-edit program.  The
LLM proposes the edit list; the host applies it to the current mapped product
and owns every downstream structure and atom-map.  This module is the narrow
compiler boundary used by the sequential director.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from collections import defaultdict
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
        if expected_product_smiles and _canonical_smiles(product) != _canonical_smiles(
            expected_product_smiles
        ):
            raise ReactionJsonReplayError("routejson_compiler_product_mismatch")
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
            product_smiles=_canonical_smiles(product),
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
            if declared_product != current_product:
                raise ReactionJsonReplayError(
                    "routejson_compiler_chain_product_mismatch"
                )
            if declared_product in seen_products:
                raise ReactionJsonReplayError("routejson_compiler_product_cycle")
            if index > 0 and declared_product not in previous_precursors:
                raise ReactionJsonReplayError(
                    "routejson_compiler_product_not_previous_precursor"
                )
            materialized = self.compile_step(
                mapped_product_smiles=current_mapped,
                operations=row.get("reaction_operations") or (),
                expected_product_smiles=current_product,
            )
            compiled.append(materialized)
            seen_products.add(declared_product)
            previous_precursors = materialized.precursor_smiles
            previous_mapped_precursors = materialized.mapped_precursor_smiles
            if index + 1 < len(rows):
                next_product = _canonical_smiles(rows[index + 1].get("product_smiles"))
                matches = [
                    mapped
                    for precursor, mapped in zip(
                        previous_precursors, previous_mapped_precursors
                    )
                    if precursor == next_product
                ]
                if not matches:
                    raise ReactionJsonReplayError(
                        "routejson_compiler_next_product_not_previous_precursor"
                    )
                current_product = next_product
                current_mapped = matches[0]
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

    buckets: dict[str, list[str]] = defaultdict(list)
    for mapped in mapped_precursor_smiles:
        canonical = _canonical_smiles(mapped)
        if canonical:
            buckets[canonical].append(str(mapped))
    for values in buckets.values():
        values.sort()
    aligned: list[str] = []
    for precursor in precursor_smiles:
        canonical = _canonical_smiles(precursor)
        matches = buckets.get(canonical) or []
        if not matches:
            return ()
        aligned.append(matches.pop(0))
    if any(values for values in buckets.values()):
        return ()
    return tuple(aligned)


__all__ = ["MaterializedReaction", "RouteJSONCompiler"]
