"""Shared topology helpers for substituting a contiguous route Program span."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

from cascade_planner.application.route_innovation_windows import (
    enumerate_route_windows,
    molecule_smiles,
)
from cascade_planner.application.transformation_programs import program_id


class ProgramSpanError(ValueError):
    """A proposed Program span cannot be replaced without breaking the route."""


def program_span_boundary(
    graph: Mapping[str, Any], route: Mapping[str, Any], span: Sequence[str]
) -> dict[str, list[str]]:
    """Return external input/output states for one replaceable contiguous span."""

    edges = dict(graph.get("edges") or {})
    route_edges = [str(value) for value in route.get("edge_ids") or []]
    values = [str(value) for value in span]
    reasons: list[str] = []
    if not values or len(values) != len(set(values)):
        reasons.append("replacement_span_invalid")
    if any(value not in route_edges or value not in edges for value in values):
        reasons.append("replacement_span_not_on_route")
    if reasons:
        raise ProgramSpanError(";".join(reasons))
    rows = [dict(edges[value]) for value in values]
    for current, following in zip(rows, rows[1:], strict=False):
        if str(current.get("product_molecule_id") or "") not in {
            str(value) for value in following.get("precursor_molecule_ids") or []
        }:
            reasons.append("replacement_span_not_contiguous")
    produced = [str(row.get("product_molecule_id") or "") for row in rows]
    consumed = [
        str(value)
        for row in rows
        for value in row.get("precursor_molecule_ids") or []
    ]
    produced_set = set(produced)
    consumed_set = set(consumed)
    inputs = _ordered_unique(value for value in consumed if value not in produced_set)
    outputs = _ordered_unique(value for value in produced if value not in consumed_set)
    internal = produced_set & consumed_set
    external_rows = [dict(edges[value]) for value in route_edges if value not in values]
    if any(
        state in {str(value) for value in row.get("precursor_molecule_ids") or []}
        for state in internal
        for row in external_rows
    ):
        reasons.append("replacement_span_internal_state_has_external_consumer")
    if not inputs:
        reasons.append("replacement_span_input_boundary_missing")
    if len(outputs) != 1:
        reasons.append("replacement_span_requires_single_output_boundary")
    if reasons:
        raise ProgramSpanError(";".join(sorted(set(reasons))))
    return {"input_molecule_ids": inputs, "output_molecule_ids": outputs}


def matching_program_spans(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    *,
    precursor_smiles: str,
    product_smiles: str,
) -> list[list[str]]:
    """Find every replaceable route span with the exact proposed boundary."""

    precursor = _canonical_smiles(precursor_smiles)
    product = _canonical_smiles(product_smiles)
    if not precursor or not product or precursor == product:
        return []
    edge_ids = [str(value) for value in route.get("edge_ids") or []]
    enumeration = enumerate_route_windows(
        graph,
        edge_ids,
        max_window_steps=max(1, len(edge_ids)),
    )
    matches: list[list[str]] = []
    for raw_span in enumeration["windows"]:
        span = [str(value) for value in raw_span]
        try:
            boundary = program_span_boundary(graph, route, span)
        except ProgramSpanError:
            continue
        inputs = boundary["input_molecule_ids"]
        output = boundary["output_molecule_ids"][0]
        if len(inputs) != 1:
            continue
        if _canonical_smiles(molecule_smiles(graph, inputs[0])) != precursor:
            continue
        if _canonical_smiles(molecule_smiles(graph, output)) != product:
            continue
        matches.append(span)
    return sorted(matches, key=lambda value: (len(value), value))


def substitute_program_span(
    route: Mapping[str, Any],
    span: Sequence[str],
    replacement_program_id: str,
) -> dict[str, list[str]]:
    """Return fallback and substituted Program sequences without changing edges."""

    edge_ids = [str(value) for value in route.get("edge_ids") or []]
    replaced_edges = [str(value) for value in span]
    if (
        not replacement_program_id
        or not replaced_edges
        or len(replaced_edges) != len(set(replaced_edges))
        or any(value not in edge_ids for value in replaced_edges)
    ):
        raise ProgramSpanError("program_substitution_span_invalid")
    fallback = [program_id(value) for value in edge_ids]
    replaced_programs = {program_id(value) for value in replaced_edges}
    insert_at = min(
        index for index, value in enumerate(fallback) if value in replaced_programs
    )
    selected = [value for value in fallback if value not in replaced_programs]
    selected.insert(insert_at, replacement_program_id)
    return {
        "fallback_program_ids": fallback,
        "selected_program_ids": selected,
        "replaced_program_ids": [program_id(value) for value in replaced_edges],
    }


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


__all__ = [
    "ProgramSpanError",
    "matching_program_spans",
    "program_span_boundary",
    "substitute_program_span",
]
