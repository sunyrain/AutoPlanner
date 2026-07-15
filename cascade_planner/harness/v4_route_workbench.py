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
from cascade_planner.harness.v4_workbench_authority import (
    frontier_ledger,
    retrosynthesis_control,
    selected_route_proof,
)
from cascade_planner.harness.v4_route_display import compile_route_display_rows
from cascade_planner.harness.v4_route_evidence_projection import (
    PROOF_TIER as _PROOF_TIER,
    closure_label as _closure_label,
    condition_summary as _condition_summary,
    conditions as _conditions,
    replacement_validation_projection as _replacement_validation_projection,
    trust_vector as _trust_vector,
)


V4_WORKBENCH_ADAPTER_SCHEMA = "v4_route_workbench_adapter.v1"

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
    step_id_by_branch_edge: dict[tuple[str, str], str] = {}

    routes = dict(source.get("routes") or {})
    replacement_routes = dict(source.get("replacement_routes") or {})
    edge_rows = dict(source.get("edges") or {})
    edge_inspectors = dict(dict(source.get("inspectors") or {}).get("edges") or {})
    route_entries = [
        (route_id, route_value, False) for route_id, route_value in routes.items()
    ] + [
        (route_id, route_value, True)
        for route_id, route_value in replacement_routes.items()
    ]
    for route_index, (route_id, route_value, is_replacement) in enumerate(
        route_entries
    ):
        route = dict(route_value)
        branch_id = str(route_id)
        step_ids: list[str] = []
        branch_node_ids: set[str] = set()
        display_rows = compile_route_display_rows(
            route.get("edge_ids") or [],
            edge_rows=edge_rows,
            edge_inspectors=edge_inspectors,
            nodes_by_id=nodes_by_id,
        )
        for display in display_rows:
            canonical_edge_id = str(display["edge_id"])
            edge = dict(edge_rows.get(str(canonical_edge_id)) or {})
            if not edge:
                continue
            step_id = _stable_id("step", branch_id, str(canonical_edge_id))
            step_id_by_branch_edge[(branch_id, str(canonical_edge_id))] = step_id
            reaction_graph_id = _reaction_graph_id(step_id)
            product_id = str(edge.get("product_molecule_id") or "")
            raw_precursor_ids = [
                str(value) for value in edge.get("precursor_molecule_ids") or []
            ]
            precursor_ids, precursor_multiplicity = _project_stoichiometric_inputs(
                raw_precursor_ids
            )
            main_precursor_ids = [
                value
                for value in precursor_ids
                if value in set(display["main_precursor_ids"])
            ]
            auxiliary_precursor_ids = [
                value
                for value in precursor_ids
                if value in set(display["auxiliary_precursor_ids"])
            ]
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
            proof_vector = dict(
                inspector.get("proof_vector") or edge.get("proof_vector") or {}
            )
            condition_status = str(
                inspector.get("condition_status")
                or edge.get("condition_status")
                or "missing"
            )
            condition_completeness = str(
                proof_vector.get("condition_completeness") or "missing"
            )
            repeated_inputs = [
                row for row in precursor_multiplicity if int(row["count"]) > 1
            ]
            if repeated_inputs:
                conditions.append(
                    {
                        "label": "input multiplicity",
                        "value": ", ".join(
                            f"{nodes_by_id[row['molecule_node_id']]['label']} ×{row['count']}"
                            for row in repeated_inputs
                        ),
                    }
                )
            step = {
                "step_id": step_id,
                "graph_step_id": str(canonical_edge_id),
                "authority_step_ids": [step_id, str(canonical_edge_id)],
                "frontier_exact_edge_signature": str(canonical_edge_id),
                "branch_id": branch_id,
                "from_node_ids": precursor_ids,
                "main_from_node_ids": main_precursor_ids,
                "auxiliary_from_node_ids": auxiliary_precursor_ids,
                "precursor_multiplicity": precursor_multiplicity,
                "stoichiometric_input_count": len(raw_precursor_ids),
                "to_node_ids": [product_id],
                "reaction_class": str(route.get("strategy") or "Retrosynthetic step"),
                "display_label": str(display["display_label"]),
                "stage_label": str(display["stage_label"]),
                "synthesis_stage": int(display["synthesis_stage"]),
                "retrosynthesis_stage": int(display["retrosynthesis_stage"]),
                "retrosynthesis_label": str(display["retrosynthesis_label"]),
                "retrosynthesis_display_label": str(
                    display["retrosynthesis_display_label"]
                ),
                "source_step_labels": list(display["source_step_labels"]),
                "producer_kinds": list(display["producer_kinds"]),
                "producer_label": str(display["producer_label"]),
                "evidence_kinds": list(display["evidence_kinds"]),
                "evidence_label": str(display["evidence_label"]),
                "conditions": conditions,
                "condition_status": condition_status,
                "condition_completeness": condition_completeness,
                "condition_summary": _condition_summary(condition_status),
                "proof_vector": proof_vector,
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
                "label": str(display["display_label"]),
                "producer_label": str(display["producer_label"]),
                "evidence_label": str(display["evidence_label"]),
                "producer_kinds": list(display["producer_kinds"]),
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
                        visual_role=(
                            "auxiliary"
                            if precursor_id in auxiliary_precursor_ids
                            else "main"
                        ),
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
        branch = _route_branch(
            route,
            branch_id=branch_id,
            step_ids=step_ids,
            primary=route_index == 0 and not is_replacement,
        )
        if is_replacement:
            branch.update(
                {
                    "kind": "validated_replacement_route",
                    "listed": False,
                    "is_primary": False,
                    "reaction_validated": route.get("stage")
                    in {"reaction_validated", "stock_closed"},
                    "proof_eligible": route.get("complete") is True,
                    "synthesis_class": "validated_replacement",
                    "not_parent_route_proof": True,
                }
            )
        branches.append(branch)
        branch_views.append(
            {
                "branch_id": branch_id,
                "step_ids": step_ids,
                "topological_step_ids": step_ids,
                "root_molecule_node_ids": list(route.get("leaf_molecule_ids") or []),
                "dependencies": [str(value["edge_id"]) for value in display_rows],
                "acyclic": True,
                "all_leaves_configured_boundary_closed": route.get("complete") is True,
                "all_leaves_stock_bound": bool(
                    route.get("complete") is True
                    and route.get("stock_boundary") in {"procurement", "in_house"}
                ),
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
    portfolio_accepted = dict(source.get("portfolio") or {}).get("accepted") is True
    frontier_ledger_value = frontier_ledger(
        source,
        nodes_by_id=nodes_by_id,
        authority_edges=authority_edges,
    )
    selected_route_proof_value = selected_route_proof(source)
    route_count = len(routes)
    complete_count = sum(dict(value).get("complete") is True for value in routes.values())
    search_closed_count = sum(
        dict(value).get("closure_profile") == "exploration_closed"
        for value in routes.values()
    )
    procurement_closed_count = sum(
        dict(value).get("closure_profile") == "procurement_closed"
        for value in routes.values()
    )
    process_ready_count = sum(
        dict(value).get("process_ready") is True for value in routes.values()
    )
    replacement_validation = _replacement_validation_projection(
        source,
        step_id_by_branch_edge=step_id_by_branch_edge,
    )
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
            "configured_boundary_closed_routes": complete_count,
            "exploration_closed_routes": search_closed_count,
            "procurement_closed_routes": procurement_closed_count,
            "process_ready_routes": process_ready_count,
            "molecules": len(nodes_by_id),
            "reactions": len(steps),
        },
        "campaign_summary": dict(source.get("campaign_summary") or {}),
        "primary_branch_id": primary_id,
        "primary_selection": {
            "schema_version": "route_forest_primary_selection.v1",
            "status": "display_projection",
            "primary_branch_id": primary_id,
            "proof_level": "v4_proof_portfolio" if portfolio_accepted else "",
            "advisory_only": dict(source.get("portfolio") or {}).get("process_ready") is not True,
            "display_tiebreak_only": True,
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
            "default_display_direction": "retrosynthesis",
            "acyclic": True,
            "nodes": list(graph_nodes.values()),
            "edges": graph_edges,
            "branch_views": branch_views,
            "no_array_adjacency_edges": True,
        },
        "frontier_ledger": frontier_ledger_value,
        "selected_route_parent_proof": selected_route_proof_value,
        "retrosynthesis_control": retrosynthesis_control(
            source,
            selected_route_proof=selected_route_proof_value,
        ),
        "semantic_summary": {"frontier_ledger": frontier_ledger_value},
        "display_policy": {
            "default_overview_top_k": min(5, max(2, route_count)),
            "default_group_visible_count": 5,
            "default_view": "current_portfolio_route",
            "all_exploration_requires_explicit_expand": True,
            "default_route_direction": "retrosynthesis",
            "auxiliary_inputs_collapsed": True,
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
        "replacement_validation": replacement_validation,
        "route_portfolio_projection": {
            "schema_version": "proof_stitched_route_portfolio.v1",
            "selected_route_ids": list(routes),
            "accepted": dict(source.get("portfolio") or {}).get("accepted") is True,
            "closure_profile": str(
                dict(source.get("portfolio") or {}).get("closure_profile") or "unresolved"
            ),
            "process_ready": dict(source.get("portfolio") or {}).get("process_ready") is True,
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
                "representation_count": sum(
                    len(dict(inspector).get("exact_records") or [])
                    for inspector in edge_inspectors.values()
                ),
                "real_source_candidate_records": sum(
                    len(dict(inspector).get("exact_records") or [])
                    for inspector in edge_inspectors.values()
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


def _project_stoichiometric_inputs(
    values: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Keep one display node per molecule while preserving repeated equivalents."""

    counts: dict[str, int] = {}
    unique: list[str] = []
    for molecule_id in values:
        if molecule_id not in counts:
            unique.append(molecule_id)
            counts[molecule_id] = 0
        counts[molecule_id] += 1
    return unique, [
        {"molecule_node_id": molecule_id, "count": counts[molecule_id]}
        for molecule_id in unique
    ]


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
    visual_role: str = "main",
) -> dict[str, Any]:
    return {
        "edge_id": _stable_id("dependency", branch_id, step_id, direction, molecule_id),
        "source_graph_node_id": source_id,
        "target_graph_node_id": target_id,
        "edge_type": "molecule_to_reaction" if direction == "input" else "reaction_to_molecule",
        "reaction_step_id": step_id,
        "molecule_node_id": molecule_id,
        "branch_id": branch_id,
        "visual_role": visual_role,
    }


def _route_branch(
    route: Mapping[str, Any],
    *,
    branch_id: str,
    step_ids: list[str],
    primary: bool,
) -> dict[str, Any]:
    level = max(0, min(4, int(route.get("proof_level") or 0)))
    complete = route.get("complete") is True
    closure_profile = str(route.get("closure_profile") or "unresolved")
    process_ready = route.get("process_ready") is True
    proof_vector = dict(route.get("proof_vector") or {})
    condition_status = str(proof_vector.get("conditions") or "missing")
    condition_complete = route.get("condition_complete") is True
    route_state_label = (
        "工艺候选可执行"
        if process_ready
        else "采购闭合 · 条件待补"
        if closure_profile in {"procurement_closed", "in_house_closed"}
        else "搜索边界闭合 · 非采购路线"
        if closure_profile == "exploration_closed"
        else "反应已验证 · 叶节点未闭合"
        if level >= 2
        else "路线骨架 · 待递归展开"
    )
    condition_label = {
        "source_exact": "来源条件",
        "source_recorded_unverified": "来源条件候选",
        "model_predicted": "预测条件",
        "missing": "条件缺失",
    }.get(condition_status, "条件状态未知")
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
        "solved": process_ready,
        "complete": complete,
        "configured_boundary_closed": complete,
        "closure_profile": closure_profile,
        "completion_label": _closure_label(closure_profile),
        "route_state_label": route_state_label,
        "condition_status": condition_status,
        "condition_label": condition_label,
        "condition_complete": condition_complete,
        "full_synthesis_claim": process_ready,
        "executable": process_ready,
        "process_ready": process_ready,
        "advisory_only": not process_ready,
        "not_parent_route_proof": not complete,
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
