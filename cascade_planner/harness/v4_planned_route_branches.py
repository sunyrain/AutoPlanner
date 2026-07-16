"""Render complete planner skeletons as non-authoritative V4 branches."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from cascade_planner.harness.v4_route_branch import route_branch
from cascade_planner.harness.v4_route_condition_projection import (
    source_conditions as conditions,
)
from cascade_planner.harness.v4_route_evidence_projection import (
    PROOF_TIER,
    condition_summary,
    trust_vector,
)
from cascade_planner.harness.v4_route_graph_projection import (
    graph_edge,
    molecule_graph_id,
    molecule_graph_node,
    reaction_graph_id,
    stable_id,
)


NodeFactory = Callable[[str, Mapping[str, Any]], dict[str, Any]]


def append_planned_route_branches(
    source: Mapping[str, Any],
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    steps: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    graph_nodes: dict[str, dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    branch_views: list[dict[str, Any]],
    node_factory: NodeFactory,
) -> set[str]:
    """Render full low-confidence skeletons without promoting them to routes."""

    source_edges = dict(source.get("edges") or {})
    edge_inspectors = dict(dict(source.get("inspectors") or {}).get("edges") or {})
    grouped_hypothesis_ids: set[str] = set()
    for branch_id, raw_route in dict(source.get("planned_routes") or {}).items():
        route = dict(raw_route)
        planned_steps = [
            dict(value)
            for value in route.get("steps") or []
            if isinstance(value, Mapping)
        ]
        if not planned_steps:
            continue
        step_ids: list[str] = []
        product_node_ids: set[str] = set()
        precursor_node_ids: set[str] = set()
        for index, planned in enumerate(planned_steps, start=1):
            hypothesis_id = str(planned.get("hypothesis_id") or "")
            if hypothesis_id:
                grouped_hypothesis_ids.add(hypothesis_id)
            product_id = _planned_molecule_node_id(
                str(planned.get("product_smiles") or ""),
                nodes_by_id=nodes_by_id,
                graph_nodes=graph_nodes,
                role="planner_product",
                node_factory=node_factory,
            )
            precursor_ids = [
                _planned_molecule_node_id(
                    str(smiles),
                    nodes_by_id=nodes_by_id,
                    graph_nodes=graph_nodes,
                    role="planner_precursor",
                    node_factory=node_factory,
                )
                for smiles in planned.get("precursor_smiles") or []
                if str(smiles)
            ]
            if not product_id or not precursor_ids:
                continue
            canonical_edge_id = str(planned.get("edge_id") or "")
            edge = dict(source_edges.get(canonical_edge_id) or {})
            inspector = dict(edge_inspectors.get(canonical_edge_id) or {})
            status = str(planned.get("status") or "frontier_candidate")
            numeric_level = int(edge.get("proof_level") or 0) if edge else 0
            tier = (
                "L0_rejected"
                if status == "admission_rejected"
                else PROOF_TIER[max(0, min(4, numeric_level))]
                if edge
                else "L0_advisory"
            )
            reaction_step_id = stable_id(
                "step",
                str(branch_id),
                str(planned.get("step_id") or index),
            )
            reaction_node_id = reaction_graph_id(reaction_step_id)
            source_bindings = [
                dict(value)
                for value in inspector.get("sources") or []
                if isinstance(value, Mapping)
            ]
            procedure_records = [
                dict(value)
                for value in inspector.get("procedure_records") or []
                if isinstance(value, Mapping)
            ]
            exact_records = [
                dict(value)
                for value in inspector.get("exact_records") or []
                if isinstance(value, Mapping)
            ]
            rendered_conditions = conditions(procedure_records or exact_records)
            rejection_reasons = sorted(
                {
                    *(str(value) for value in planned.get("admission_reasons") or []),
                    *(str(value) for value in inspector.get("rejection_reasons") or []),
                }
                - {""}
            )
            transformation = str(
                planned.get("transformation_hypothesis")
                or route.get("strategy")
                or "Planner hypothesis"
            )
            rendered_trust = trust_vector(
                {"proof_level": numeric_level},
                source_bindings,
            )
            rendered_trust["proof_tier"] = tier
            steps.append(
                _planned_step(
                    branch_id=str(branch_id),
                    planned=planned,
                    index=index,
                    planned_step_count=len(planned_steps),
                    reaction_step_id=reaction_step_id,
                    canonical_edge_id=canonical_edge_id,
                    hypothesis_id=hypothesis_id,
                    product_id=product_id,
                    precursor_ids=precursor_ids,
                    transformation=transformation,
                    tier=tier,
                    edge=edge,
                    inspector=inspector,
                    source_bindings=source_bindings,
                    procedure_records=procedure_records,
                    rendered_conditions=rendered_conditions,
                    rejection_reasons=rejection_reasons,
                    rendered_trust=rendered_trust,
                )
            )
            step_ids.append(reaction_step_id)
            product_node_ids.add(product_id)
            precursor_node_ids.update(precursor_ids)
            graph_nodes[reaction_node_id] = {
                "graph_node_id": reaction_node_id,
                "node_type": "reaction",
                "reaction_step_id": reaction_step_id,
                "branch_id": str(branch_id),
                "label": transformation,
                "proof_tier": tier,
            }
            for precursor_id in precursor_ids:
                graph_edges.append(
                    graph_edge(
                        str(branch_id),
                        reaction_step_id,
                        molecule_id=precursor_id,
                        source_id=molecule_graph_id(precursor_id),
                        target_id=reaction_node_id,
                        direction="input",
                    )
                )
            graph_edges.append(
                graph_edge(
                    str(branch_id),
                    reaction_step_id,
                    molecule_id=product_id,
                    source_id=reaction_node_id,
                    target_id=molecule_graph_id(product_id),
                    direction="output",
                )
            )
        if not step_ids:
            continue
        branches.append(_planned_branch(route, str(branch_id), step_ids))
        branch_views.append(
            {
                "branch_id": str(branch_id),
                "step_ids": step_ids,
                "topological_step_ids": step_ids,
                "root_molecule_node_ids": sorted(precursor_node_ids - product_node_ids),
                "dependencies": [
                    str(value.get("edge_id") or value.get("hypothesis_id") or "")
                    for value in planned_steps
                ],
                "acyclic": True,
                "all_leaves_configured_boundary_closed": False,
                "all_leaves_stock_bound": False,
            }
        )
    return grouped_hypothesis_ids


def _planned_step(
    *,
    branch_id: str,
    planned: Mapping[str, Any],
    index: int,
    planned_step_count: int,
    reaction_step_id: str,
    canonical_edge_id: str,
    hypothesis_id: str,
    product_id: str,
    precursor_ids: list[str],
    transformation: str,
    tier: str,
    edge: Mapping[str, Any],
    inspector: Mapping[str, Any],
    source_bindings: list[dict[str, Any]],
    procedure_records: list[dict[str, Any]],
    rendered_conditions: list[dict[str, Any]],
    rejection_reasons: list[str],
    rendered_trust: dict[str, Any],
) -> dict[str, Any]:
    condition_status = str(inspector.get("condition_status") or "missing")
    return {
        "step_id": reaction_step_id,
        "graph_step_id": canonical_edge_id or hypothesis_id,
        "authority_step_ids": [
            value
            for value in (reaction_step_id, canonical_edge_id, hypothesis_id)
            if value
        ],
        "branch_id": branch_id,
        "from_node_ids": precursor_ids,
        "main_from_node_ids": precursor_ids,
        "auxiliary_from_node_ids": [],
        "to_node_ids": [product_id],
        "reaction_class": transformation,
        "display_label": transformation,
        "stage_label": f"Planner step {index}",
        "synthesis_stage": planned_step_count - index + 1,
        "retrosynthesis_stage": index,
        "retrosynthesis_label": f"R{index}",
        "retrosynthesis_display_label": f"R{index} · {transformation}",
        "source_step_labels": [str(planned.get("step_id") or index)],
        "producer_kinds": [str(planned.get("origin_kind") or "planner")],
        "producer_label": "Global planner hypothesis",
        "evidence_kinds": [],
        "evidence_label": (
            "Host admission rejected; retained for review"
            if tier == "L0_rejected"
            else "Materialized; proof remains route-specific"
            if edge
            else "Proposal only; materialization pending"
        ),
        "conditions": rendered_conditions,
        "condition_status": condition_status,
        "condition_completeness": "missing",
        "condition_summary": condition_summary(condition_status),
        "procedure_records": procedure_records,
        "proof_vector": dict(inspector.get("proof_vector") or {}),
        "proof_tier": tier,
        "proof_level": tier,
        "source_refs": sorted(
            str(value.get("source_ref") or "")
            for value in source_bindings
            if str(value.get("source_ref") or "")
        ),
        "evidence_refs": [],
        "rejection_reasons": rejection_reasons,
        "validation_findings": list(inspector.get("validation_findings") or []),
        "trust_vector": rendered_trust,
        "visual_encoding": {
            "color": (
                "#be123c"
                if tier == "L0_rejected"
                else str(edge.get("proof_color") or "#e76f51")
            ),
            "width": 2.0 if tier == "L0_rejected" else 1.5,
            "opacity": 0.94 if tier == "L0_rejected" else 0.68,
            "dash_pattern": "4 3" if tier == "L0_rejected" else "6 4",
        },
    }


def _planned_branch(
    route: Mapping[str, Any], branch_id: str, step_ids: list[str]
) -> dict[str, Any]:
    branch = route_branch(
        {
            **route,
            "proof_level": 0,
            "physical_step_count": len(step_ids),
            "planner_hypothesis_step_count": len(step_ids),
            "complete": False,
            "closure_profile": "unresolved",
        },
        branch_id=branch_id,
        step_ids=step_ids,
        primary=False,
    )
    branch.update(
        {
            "title": (
                f"{len(step_ids)}-step planner route · "
                f"{int(route.get('admission_rejected_step_count') or 0)} blocked"
            ),
            "kind": "planner_route_hypothesis",
            "route_state_label": (
                "完整规划骨架 · 红色步骤未通过主机入图门 · 不计入闭合"
            ),
            "solved": False,
            "complete": False,
            "executable": False,
            "advisory_only": True,
            "not_parent_route_proof": True,
            "proof_tier": "L0_advisory",
            "confidence": "low",
            "synthesis_class": "global_planner_skeleton",
        }
    )
    return branch


def _planned_molecule_node_id(
    smiles: str,
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    graph_nodes: dict[str, dict[str, Any]],
    role: str,
    node_factory: NodeFactory,
) -> str:
    if not smiles:
        return ""
    molecule_id = next(
        (
            node_id
            for node_id, node in nodes_by_id.items()
            if str(node.get("canonical_isomeric_smiles") or node.get("smiles") or "")
            == smiles
        ),
        "",
    )
    if not molecule_id:
        molecule_id = stable_id("molecule", smiles)
        nodes_by_id[molecule_id] = node_factory(
            molecule_id,
            {"canonical_smiles": smiles, "role": role},
        )
        graph_nodes[molecule_graph_id(molecule_id)] = molecule_graph_node(
            nodes_by_id[molecule_id]
        )
    return molecule_id


__all__ = ["append_planned_route_branches"]
