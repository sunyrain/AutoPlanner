"""Bounded linear-window enumeration over canonical route edges."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


MAX_ENUMERATED_WINDOWS = 2048


def enumerate_route_windows(
    graph: Mapping[str, Any],
    edge_ids: Sequence[str],
    *,
    max_window_steps: int,
) -> dict[str, Any]:
    edges = dict(graph.get("edges") or {})
    allowed = {value for value in edge_ids if value in edges}
    successors = {
        edge_id: sorted(
            candidate
            for candidate in allowed - {edge_id}
            if str(edges[edge_id].get("product_molecule_id") or "")
            in {
                str(value)
                for value in edges[candidate].get("precursor_molecule_ids") or []
            }
        )
        for edge_id in allowed
    }
    windows: set[tuple[str, ...]] = set()
    truncated = False

    def walk(path: tuple[str, ...]) -> None:
        nonlocal truncated
        if len(windows) >= MAX_ENUMERATED_WINDOWS:
            truncated = True
            return
        windows.add(path)
        if len(path) >= max_window_steps:
            return
        for successor in successors[path[-1]]:
            if successor not in path:
                walk((*path, successor))

    for edge_id in sorted(allowed):
        if len(windows) >= MAX_ENUMERATED_WINDOWS:
            truncated = True
            break
        walk((edge_id,))
    rows = [list(value) for value in sorted(windows, key=lambda row: (len(row), row))]
    return {
        "windows": rows,
        "count": len(rows),
        "maximum": MAX_ENUMERATED_WINDOWS,
        "truncated": truncated,
    }


def route_window_boundary(
    graph: Mapping[str, Any], edge_ids: Sequence[str]
) -> dict[str, Any]:
    edges = dict(graph.get("edges") or {})
    first, last = dict(edges[edge_ids[0]]), dict(edges[edge_ids[-1]])
    precursors = [str(value) for value in first.get("precursor_molecule_ids") or []]
    if len(precursors) != 1:
        return {}
    precursor_smiles = molecule_smiles(graph, precursors[0])
    product_id = str(last.get("product_molecule_id") or "")
    product_smiles = molecule_smiles(graph, product_id)
    if not precursor_smiles or not product_smiles:
        return {}
    return {
        "precursor_molecule_id": precursors[0],
        "precursor_smiles": precursor_smiles,
        "product_molecule_id": product_id,
        "product_smiles": product_smiles,
        "replaced_edge_ids": list(edge_ids),
        "minimum_boundary_proof_level": min(
            (
                int(edges[edge_id].get("innovation_boundary_proof_level") or 0)
                for edge_id in edge_ids
            ),
            default=0,
        ),
    }


def molecule_smiles(graph: Mapping[str, Any], molecule_id: Any) -> str:
    return str(
        dict(dict(graph.get("molecules") or {}).get(str(molecule_id)) or {}).get(
            "canonical_smiles"
        )
        or ""
    )


__all__ = [
    "MAX_ENUMERATED_WINDOWS",
    "enumerate_route_windows",
    "molecule_smiles",
    "route_window_boundary",
]
