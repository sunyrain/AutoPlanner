"""Compile the exact materials behind one route Program innovation review."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.biocatalytic_programs import compile_biocatalytic_program_bundle
from cascade_planner.application.biocatalysis_validation_frontier import (
    compile_biocatalysis_validation_frontier,
)
from cascade_planner.application.experimental_work_frontier import (
    compile_experimental_work_frontier,
    experimental_work_frontier_oracle,
)
from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)
from cascade_planner.application.transformation_programs import (
    project_canonical_graph_to_programs,
)
from cascade_planner.application.program_validation_routing import (
    partition_program_validations,
)
from cascade_planner.orchestration.program_candidate_review_materials import (
    compile_program_candidate_review_materials,
)
from cascade_planner.orchestration.route_innovation_runtime import (
    route_innovation_context,
)


def compile_route_program_innovation_materials(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    reported_candidate_packs: Iterable[Mapping[str, Any]] = (),
    experience_library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation_rows = [dict(value) for value in validations]
    (
        biocatalytic_validations,
        execution_validations,
        mechanism_validations,
    ) = partition_program_validations(validation_rows)
    enriched_graph, route = route_innovation_context(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
    )
    discovery = discover_route_innovations(
        enriched_graph,
        route,
        capabilities=capabilities,
        mechanism_proposals=mechanism_proposals,
        experience_library=experience_library,
    )
    projection = project_canonical_graph_to_programs(graph)
    bundle = compile_biocatalytic_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=biocatalytic_validations,
    )
    candidate_materials = compile_program_candidate_review_materials(
        graph,
        route,
        projection,
        discovery,
        bundle,
        biocatalytic_validations=biocatalytic_validations,
        mechanism_validations=mechanism_validations,
        execution_validations=execution_validations,
        reported_candidate_packs=reported_candidate_packs,
    )
    frontier = compile_biocatalysis_validation_frontier(graph, discovery, bundle)
    experimental_work_frontier = compile_experimental_work_frontier(
        dict(graph.get("deficit_frontier") or {}),
        frontier,
        candidate_materials["execution_validation_frontier"],
        candidate_materials["mechanism_validation_frontier"],
        candidate_materials["capability_calibration"],
    )
    return {
        "graph": dict(graph),
        "route": route,
        "projection": projection,
        "discovery": discovery,
        "bundle": bundle,
        "validations": validation_rows,
        "oracle": candidate_materials["biocatalytic_oracle"],
        "validation_frontier": frontier,
        "experimental_work_frontier": experimental_work_frontier,
        "experimental_work_frontier_oracle": experimental_work_frontier_oracle(
            dict(graph.get("deficit_frontier") or {}),
            frontier,
            candidate_materials["execution_validation_frontier"],
            candidate_materials["mechanism_validation_frontier"],
            candidate_materials["capability_calibration"],
            experimental_work_frontier,
        ),
        **candidate_materials,
    }


__all__ = ["compile_route_program_innovation_materials"]
