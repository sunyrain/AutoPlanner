"""Small deterministic graph-row helpers shared by V4 display projections."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def molecule_graph_id(molecule_id: str) -> str:
    return f"graph:molecule:{molecule_id}"


def reaction_graph_id(step_id: str) -> str:
    return f"graph:reaction:{step_id}"


def molecule_graph_node(node: Mapping[str, Any]) -> dict[str, Any]:
    molecule_id = str(node["node_id"])
    return {
        "graph_node_id": molecule_graph_id(molecule_id),
        "node_type": "molecule",
        "molecule_node_id": molecule_id,
        "canonical_isomeric_smiles": str(node.get("canonical_isomeric_smiles") or ""),
        "label": str(node.get("label") or molecule_id),
        "role": str(node.get("role") or "intermediate"),
    }


def graph_edge(
    branch_id: str,
    step_id: str,
    *,
    molecule_id: str,
    source_id: str,
    target_id: str,
    direction: str,
    visual_role: str = "main",
) -> dict[str, Any]:
    return {
        "edge_id": stable_id(
            "dependency", branch_id, step_id, direction, molecule_id
        ),
        "source_graph_node_id": source_id,
        "target_graph_node_id": target_id,
        "edge_type": (
            "molecule_to_reaction" if direction == "input" else "reaction_to_molecule"
        ),
        "reaction_step_id": step_id,
        "molecule_node_id": molecule_id,
        "branch_id": branch_id,
        "visual_role": visual_role,
    }


__all__ = [
    "graph_edge",
    "molecule_graph_id",
    "molecule_graph_node",
    "reaction_graph_id",
    "stable_id",
]
