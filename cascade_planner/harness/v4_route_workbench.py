"""Render the canonical V4 workbench projection through the route UI shell.

This is a display-only strangler adapter.  It does not call ``RouteForest`` and
does not reconstruct scientific truth from labels or counts.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from typing import Any, Mapping

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from cascade_planner.harness.route_forest_delivery import (
    render_route_forest_html,
    sanitize_structure_svg,
)
from cascade_planner.harness.route_forest_layout import canonical_sha256


V4_WORKBENCH_ADAPTER_SCHEMA = "v4_route_workbench_adapter.v1"

_PROOF_TIER = {
    0: "L0_advisory",
    1: "L0_materialized",
    2: "L2_reaction_validated",
    3: "L3_precedent_supported",
    4: "L4_procurement_ready",
}


def compile_v4_route_forest(workbench: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one bounded V4 read model to the offline route-workbench shell."""

    _validate_workbench(workbench)
    source = dict(workbench)
    nodes_by_id = {
        str(molecule_id): _node(str(molecule_id), value)
        for molecule_id, value in dict(source.get("molecules") or {}).items()
        if isinstance(value, Mapping)
    }
    steps: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    graph_nodes: dict[str, dict[str, Any]] = {
        _molecule_graph_id(node_id): _molecule_graph_node(node)
        for node_id, node in nodes_by_id.items()
    }
    graph_edges: list[dict[str, Any]] = []
    branch_views: list[dict[str, Any]] = []
    authority_edges: list[dict[str, Any]] = []

    routes = dict(source.get("routes") or {})
    edge_rows = dict(source.get("edges") or {})
    edge_inspectors = dict(dict(source.get("inspectors") or {}).get("edges") or {})
    for route_index, (route_id, route_value) in enumerate(routes.items()):
        route = dict(route_value)
        branch_id = str(route_id)
        step_ids: list[str] = []
        branch_node_ids: set[str] = set()
        for edge_index, canonical_edge_id in enumerate(route.get("edge_ids") or []):
            edge = dict(edge_rows.get(str(canonical_edge_id)) or {})
            if not edge:
                continue
            step_id = _stable_id("step", branch_id, str(canonical_edge_id))
            reaction_graph_id = _reaction_graph_id(step_id)
            product_id = str(edge.get("product_molecule_id") or "")
            precursor_ids = [str(value) for value in edge.get("precursor_molecule_ids") or []]
            if product_id not in nodes_by_id or any(value not in nodes_by_id for value in precursor_ids):
                continue
            proof_level = int(edge.get("proof_level") or 0)
            tier = _PROOF_TIER[max(0, min(4, proof_level))]
            inspector = dict(edge_inspectors.get(str(canonical_edge_id)) or {})
            source_bindings = [
                dict(value) for value in inspector.get("sources") or [] if isinstance(value, Mapping)
            ]
            exact_records = [
                dict(value)
                for value in inspector.get("exact_records") or []
                if isinstance(value, Mapping)
            ]
            conditions = _conditions(exact_records)
            step = {
                "step_id": step_id,
                "graph_step_id": str(canonical_edge_id),
                "authority_step_ids": [step_id, str(canonical_edge_id)],
                "frontier_exact_edge_signature": str(canonical_edge_id),
                "branch_id": branch_id,
                "from_node_ids": precursor_ids,
                "to_node_ids": [product_id],
                "reaction_class": str(route.get("strategy") or "Retrosynthetic step"),
                "conditions": conditions,
                "proof_tier": tier,
                "proof_level": tier,
                "source_refs": sorted(
                    str(value.get("source_ref") or "")
                    for value in source_bindings
                    if str(value.get("source_ref") or "")
                ),
                "evidence_refs": sorted(
                    str(value.get("location_ref") or "")
                    for value in exact_records
                    if str(value.get("location_ref") or "")
                ),
                "trusted_exact_source_bindings": [
                    {
                        "binding_id": str(value.get("source_binding_id") or ""),
                        "source_ref": str(value.get("source_ref") or ""),
                        "independent_source_group": str(value.get("independence_group") or ""),
                    }
                    for value in source_bindings
                ],
                "conflicts": list(inspector.get("conflicts") or []),
                "rejection_reasons": list(inspector.get("rejection_reasons") or []),
                "trust_vector": _trust_vector(edge, source_bindings),
                "visual_encoding": {
                    "color": str(edge.get("proof_color") or "#64748b"),
                    "width": 2.2 if proof_level >= 2 else 1.5,
                    "opacity": 0.92 if proof_level >= 2 else 0.64,
                    "dash_pattern": "" if proof_level >= 2 else "6 4",
                },
            }
            steps.append(step)
            step_ids.append(step_id)
            branch_node_ids.update([product_id, *precursor_ids])
            graph_nodes[reaction_graph_id] = {
                "graph_node_id": reaction_graph_id,
                "node_type": "reaction",
                "reaction_step_id": step_id,
                "branch_id": branch_id,
                "label": f"Step {edge_index + 1}",
                "proof_tier": tier,
            }
            for precursor_id in precursor_ids:
                graph_edges.append(
                    _graph_edge(
                        branch_id,
                        step_id,
                        molecule_id=precursor_id,
                        source_id=_molecule_graph_id(precursor_id),
                        target_id=reaction_graph_id,
                        direction="input",
                    )
                )
            graph_edges.append(
                _graph_edge(
                    branch_id,
                    step_id,
                    molecule_id=product_id,
                    source_id=reaction_graph_id,
                    target_id=_molecule_graph_id(product_id),
                    direction="output",
                )
            )
            authority_edges.append(
                {
                    "exact_edge_signature": str(canonical_edge_id),
                    "step_ids": [step_id, str(canonical_edge_id)],
                    "product_smiles": str(nodes_by_id[product_id]["canonical_isomeric_smiles"]),
                    "reaction_proof": {
                        "achieved_proof_level": proof_level,
                        "authority": "current_host_verifier_replay",
                        "current_host_reaction_validated": proof_level >= 2,
                        "proof_request_ids": [f"proof:{canonical_edge_id}"],
                    },
                }
            )
        branch = _route_branch(route, branch_id=branch_id, step_ids=step_ids, primary=route_index == 0)
        branches.append(branch)
        branch_views.append(
            {
                "branch_id": branch_id,
                "step_ids": step_ids,
                "topological_step_ids": step_ids,
                "root_molecule_node_ids": list(route.get("leaf_molecule_ids") or []),
                "dependencies": [str(value) for value in route.get("edge_ids") or []],
                "acyclic": True,
                "all_leaves_stock_bound": route.get("complete") is True,
            }
        )

    _append_hypothesis_branches(
        source,
        nodes_by_id=nodes_by_id,
        steps=steps,
        branches=branches,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        branch_views=branch_views,
    )
    primary_id = str(dict(source.get("portfolio") or {}).get("default_route_id") or "")
    frontier_ledger = _frontier_ledger(source, nodes_by_id=nodes_by_id, authority_edges=authority_edges)
    route_count = len(routes)
    complete_count = sum(dict(value).get("complete") is True for value in routes.values())
    forest = {
        "schema_version": "explored_route_forest.v1",
        "case_id": str(source.get("run_id") or ""),
        "target": {
            "name": str(dict(source.get("target") or {}).get("name") or "Target"),
            "smiles": str(dict(source.get("target") or {}).get("canonical_smiles") or ""),
        },
        "counts": {
            "branches": len(branches),
            "portfolio_routes": route_count,
            "complete_portfolio_routes": complete_count,
            "molecules": len(nodes_by_id),
            "reactions": len(steps),
        },
        "primary_branch_id": primary_id,
        "primary_selection": {
            "schema_version": "route_forest_primary_selection.v1",
            "status": "display_projection",
            "primary_branch_id": primary_id,
            "display_tiebreak_only": True,
            "advisory_only": not bool(primary_id),
        },
        "nodes": list(nodes_by_id.values()),
        "steps": steps,
        "branches": branches,
        "modules": list(dict(source.get("modules") or {}).values()),
        "relationships": [],
        "dependency_graph": {
            "schema_version": "molecule_reaction_dependency_graph.v1",
            "graph_kind": "canonical_v4_portfolio_projection",
            "direction": "precursors_to_target",
            "acyclic": True,
            "nodes": list(graph_nodes.values()),
            "edges": graph_edges,
            "branch_views": branch_views,
            "no_array_adjacency_edges": True,
        },
        "frontier_ledger": frontier_ledger,
        "semantic_summary": {"frontier_ledger": frontier_ledger},
        "display_policy": {
            "default_overview_top_k": min(5, max(2, route_count)),
            "default_group_visible_count": 5,
            "default_view": "current_portfolio_route",
            "all_exploration_requires_explicit_expand": True,
        },
        "artifact_revision": {
            "schema_version": "route_forest_source_revision_context.v1",
            "revision": int(dict(source.get("revision") or {}).get("graph") or 0),
            "revision_id": str(source.get("content_sha256") or ""),
            "authority": "display_projection_only",
        },
        "projection_coverage": {
            "schema_version": "route_forest_projection_coverage.v1",
            "complete": True,
            "bounded_portfolio": True,
            "omitted_route_count": 0,
        },
        "replacement_validation": {"records": []},
        "route_portfolio_projection": {
            "schema_version": "proof_stitched_route_portfolio.v1",
            "selected_route_ids": list(routes),
            "accepted": dict(source.get("portfolio") or {}).get("accepted") is True,
        },
        "evidence_index": {
            "edge_inspectors": dict(dict(source.get("inspectors") or {}).get("edges") or {}),
            "molecule_inspectors": dict(dict(source.get("inspectors") or {}).get("molecules") or {}),
            "rejections": list(dict(source.get("inspectors") or {}).get("rejections") or []),
            "conflicts": dict(dict(source.get("inspectors") or {}).get("conflicts") or {}),
        },
        "run_trace": {
            "literature_counts": {
                "independent_source_group_count": len(
                    {
                        str(group)
                        for route in routes.values()
                        for group in dict(route).get("independent_source_groups") or []
                    }
                ),
                "document_count": len(
                    {
                        str(source_row.get("source_ref") or "")
                        for inspector in edge_inspectors.values()
                        for source_row in dict(inspector).get("sources") or []
                        if isinstance(source_row, Mapping)
                    }
                ),
            }
        },
        "design_notes": [
            "V4 read model only; canonical graph and proof portfolio remain authoritative.",
            "Default route display is bounded to two to five portfolio routes.",
            "Hypotheses are separate advisory branches and never count as closed routes.",
        ],
        "adapter": {
            "schema_version": V4_WORKBENCH_ADAPTER_SCHEMA,
            "source_sha256": str(source.get("content_sha256") or ""),
            "scientific_authority": False,
        },
    }
    return forest


def render_v4_route_workbench_html(workbench: Mapping[str, Any]) -> str:
    return render_route_forest_html(compile_v4_route_forest(workbench))


def _validate_workbench(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "retrosynthesis_route_workbench.v1":
        raise ValueError("v4_route_workbench_schema_invalid")
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    if not supplied or supplied != _digest(row):
        raise ValueError("v4_route_workbench_digest_invalid")
    route_count = len(dict(value.get("routes") or {}))
    if route_count > 5:
        raise ValueError("v4_route_workbench_route_limit_exceeded")


def _node(molecule_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
    smiles = str(value.get("canonical_smiles") or "")
    depiction = _depiction(smiles)
    return {
        "node_id": molecule_id,
        "label": depiction["formula"] or str(value.get("role") or "Molecule"),
        "canonical_isomeric_smiles": smiles,
        "smiles": smiles,
        "role": str(value.get("role") or "intermediate"),
        "roles": [str(value.get("role") or "intermediate")],
        "formula": depiction["formula"],
        "heavy_atom_count": depiction["heavy_atom_count"],
        "structure_svg": depiction["structure_svg"],
        "stock_closed": value.get("stock_closed") is True,
        "stock_observation_id": str(value.get("stock_observation_id") or ""),
    }


@lru_cache(maxsize=1024)
def _depiction(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"formula": "", "heavy_atom_count": None, "structure_svg": ""}
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(240, 170)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    raw = drawer.GetDrawingText().replace("svg:", "")
    raw = raw[raw.find("<svg") :] if "<svg" in raw else raw
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "structure_svg": sanitize_structure_svg(raw),
    }


def _molecule_graph_id(molecule_id: str) -> str:
    return f"graph:molecule:{molecule_id}"


def _reaction_graph_id(step_id: str) -> str:
    return f"graph:reaction:{step_id}"


def _molecule_graph_node(node: Mapping[str, Any]) -> dict[str, Any]:
    molecule_id = str(node["node_id"])
    return {
        "graph_node_id": _molecule_graph_id(molecule_id),
        "node_type": "molecule",
        "molecule_node_id": molecule_id,
        "canonical_isomeric_smiles": str(node.get("canonical_isomeric_smiles") or ""),
        "label": str(node.get("label") or molecule_id),
        "role": str(node.get("role") or "intermediate"),
    }


def _graph_edge(
    branch_id: str,
    step_id: str,
    *,
    molecule_id: str,
    source_id: str,
    target_id: str,
    direction: str,
) -> dict[str, Any]:
    return {
        "edge_id": _stable_id("dependency", branch_id, step_id, direction, molecule_id),
        "source_graph_node_id": source_id,
        "target_graph_node_id": target_id,
        "edge_type": "molecule_to_reaction" if direction == "input" else "reaction_to_molecule",
        "reaction_step_id": step_id,
        "molecule_node_id": molecule_id,
        "branch_id": branch_id,
    }


def _route_branch(
    route: Mapping[str, Any],
    *,
    branch_id: str,
    step_ids: list[str],
    primary: bool,
) -> dict[str, Any]:
    level = max(0, min(4, int(route.get("proof_level") or 0)))
    source_refs = [
        str(value)[7:]
        for value in route.get("badges") or []
        if str(value).startswith("source:")
    ]
    return {
        "branch_id": branch_id,
        "title": str(route.get("strategy") or f"Portfolio route {branch_id[-6:]}"),
        "kind": "proof_eligible_portfolio_route",
        "listed": True,
        "is_primary": primary,
        "step_ids": step_ids,
        "solved": route.get("complete") is True,
        "complete": route.get("complete") is True,
        "executable": route.get("complete") is True,
        "advisory_only": route.get("complete") is not True,
        "not_parent_route_proof": True,
        "proof_tier": _PROOF_TIER[level],
        "confidence": "high" if level >= 3 else "medium" if level >= 2 else "low",
        "source_refs": source_refs,
        "multi_source": len(route.get("independent_source_groups") or []) >= 2,
        "synthesis_class": "canonical_portfolio",
        "trust_vector": {
            "min_trusted_source_group_count_across_steps": len(
                route.get("independent_source_groups") or []
            ),
            "corroborated_edge_count": len(step_ids) if level >= 3 else 0,
            "all_edges_corroborated": bool(step_ids) and level >= 3,
        },
    }


def _append_hypothesis_branches(
    source: Mapping[str, Any],
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    steps: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    graph_nodes: dict[str, dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    branch_views: list[dict[str, Any]],
) -> None:
    target_id = str(dict(source.get("target") or {}).get("molecule_id") or "")
    for hypothesis_id, value in dict(source.get("hypotheses") or {}).items():
        hypothesis = dict(value)
        branch_id = str(hypothesis_id)
        product_id = target_id
        precursor_ids: list[str] = []
        for smiles in hypothesis.get("precursor_smiles") or []:
            molecule_id = _stable_id("molecule", str(smiles))
            precursor_ids.append(molecule_id)
            if molecule_id not in nodes_by_id:
                nodes_by_id[molecule_id] = _node(
                    molecule_id,
                    {"canonical_smiles": str(smiles), "role": "hypothesis_precursor"},
                )
                graph_nodes[_molecule_graph_id(molecule_id)] = _molecule_graph_node(
                    nodes_by_id[molecule_id]
                )
        if product_id not in nodes_by_id or not precursor_ids:
            continue
        step_id = _stable_id("step", branch_id)
        reaction_id = _reaction_graph_id(step_id)
        steps.append(
            {
                "step_id": step_id,
                "branch_id": branch_id,
                "from_node_ids": precursor_ids,
                "to_node_ids": [product_id],
                "proof_tier": "L0_advisory",
                "proof_level": "L0_advisory",
                "conditions": [],
                "source_refs": [],
                "evidence_refs": [],
                "trust_vector": _trust_vector({"proof_level": 0}, []),
            }
        )
        graph_nodes[reaction_id] = {
            "graph_node_id": reaction_id,
            "node_type": "reaction",
            "reaction_step_id": step_id,
            "branch_id": branch_id,
            "label": "Disconnection hypothesis",
            "proof_tier": "L0_advisory",
        }
        for precursor_id in precursor_ids:
            graph_edges.append(
                _graph_edge(
                    branch_id,
                    step_id,
                    molecule_id=precursor_id,
                    source_id=_molecule_graph_id(precursor_id),
                    target_id=reaction_id,
                    direction="input",
                )
            )
        graph_edges.append(
            _graph_edge(
                branch_id,
                step_id,
                molecule_id=product_id,
                source_id=reaction_id,
                target_id=_molecule_graph_id(product_id),
                direction="output",
            )
        )
        branches.append(
            {
                "branch_id": branch_id,
                "title": "Disconnection hypothesis",
                "kind": "retrosynthetic_proposal",
                "listed": True,
                "is_primary": False,
                "step_ids": [step_id],
                "solved": False,
                "complete": False,
                "executable": False,
                "advisory_only": True,
                "not_parent_route_proof": True,
                "proof_tier": "L0_advisory",
                "confidence": "low",
                "source_refs": list(hypothesis.get("origin_kinds") or []),
                "synthesis_class": "global_hypothesis",
            }
        )
        branch_views.append(
            {
                "branch_id": branch_id,
                "step_ids": [step_id],
                "topological_step_ids": [step_id],
                "root_molecule_node_ids": precursor_ids,
                "dependencies": [],
                "acyclic": True,
                "all_leaves_stock_bound": False,
            }
        )


def _frontier_ledger(
    source: Mapping[str, Any],
    *,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    authority_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    molecules_by_smiles: dict[str, dict[str, Any]] = {}
    for molecule_id, node in nodes_by_id.items():
        smiles = str(node.get("canonical_isomeric_smiles") or "")
        if not smiles:
            continue
        stock_closed = node.get("stock_closed") is True
        observation_id = str(node.get("stock_observation_id") or "")
        current = molecules_by_smiles.setdefault(
            smiles,
            {
                "canonical_smiles": smiles,
                "work": {"proposal_expansion_succeeded": True, "job_ids": []},
                "stock": {
                    "host_replay_verified": False,
                    "current_observation_ids": [],
                    "closure_job_ids": [],
                    "benchmark_search_boundary_closed": False,
                    "procurement_boundary_closed": False,
                },
            },
        )
        current["work"]["job_ids"] = sorted(
            {*current["work"]["job_ids"], f"graph:{molecule_id}"}
        )
        if stock_closed and observation_id:
            current["stock"].update(
                {
                    "host_replay_verified": True,
                    "current_observation_ids": sorted(
                        {
                            *current["stock"]["current_observation_ids"],
                            observation_id,
                        }
                    ),
                    "closure_job_ids": sorted(
                        {
                            *current["stock"]["closure_job_ids"],
                            f"stock:{molecule_id}",
                        }
                    ),
                    "benchmark_search_boundary_closed": True,
                    "procurement_boundary_closed": True,
                }
            )
    edges_by_signature: dict[str, dict[str, Any]] = {}
    for edge in authority_edges:
        signature = str(edge.get("exact_edge_signature") or "")
        if not signature:
            continue
        current = edges_by_signature.setdefault(signature, dict(edge))
        current["step_ids"] = sorted(
            {
                *(str(value) for value in current.get("step_ids") or []),
                *(str(value) for value in edge.get("step_ids") or []),
            }
        )
        current_level = int(dict(current.get("reaction_proof") or {}).get("achieved_proof_level") or 0)
        next_level = int(dict(edge.get("reaction_proof") or {}).get("achieved_proof_level") or 0)
        if next_level > current_level:
            current["reaction_proof"] = dict(edge.get("reaction_proof") or {})
    ledger = {
        "schema_version": "route_forest_frontier_ledger_authority.v1",
        "authoritative": True,
        "stage_authority": {
            "schema_version": "route_forest_stage_authority.v1",
            "authoritative": True,
            "reasons": [],
            "molecules": [molecules_by_smiles[key] for key in sorted(molecules_by_smiles)],
            "edges": [edges_by_signature[key] for key in sorted(edges_by_signature)],
        },
        "counts": {
            "selected_routes": len(dict(source.get("routes") or {})),
            "complete_routes": sum(
                dict(value).get("complete") is True
                for value in dict(source.get("routes") or {}).values()
            ),
        },
        "semantics": {
            "display_projection_only": True,
            "aggregate_counts_never_authorize_stage_membership": True,
        },
    }
    ledger["content_sha256"] = canonical_sha256(ledger)
    return ledger


def _trust_vector(edge: Mapping[str, Any], sources: list[Mapping[str, Any]]) -> dict[str, Any]:
    level = int(edge.get("proof_level") or 0)
    source_groups = {
        str(value.get("independence_group") or value.get("source_ref") or "")
        for value in sources
        if str(value.get("independence_group") or value.get("source_ref") or "")
    }
    return {
        "proof_tier": _PROOF_TIER[max(0, min(4, level))],
        "identity": 1.0,
        "connectivity": 1.0 if level >= 1 else 0.5,
        "source_independence": min(1.0, len(source_groups) / 2),
        "stock": 1.0 if level >= 4 else 0.0,
        "conditions": 1.0 if level >= 3 else 0.0,
        "forward_feasibility": 1.0 if level >= 2 else 0.0,
        "trusted_source_group_count": len(source_groups),
        "corroborated": len(source_groups) >= 2,
    }


def _conditions(records: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    for record in records:
        raw = record.get("conditions")
        if not isinstance(raw, Mapping):
            continue
        return [
            {"label": str(key).replace("_", " "), "value": str(value)}
            for key, value in sorted(raw.items())
            if value not in (None, "", [], {})
        ]
    return []


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


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


__all__ = [
    "V4_WORKBENCH_ADAPTER_SCHEMA",
    "compile_v4_route_forest",
    "render_v4_route_workbench_html",
]
