"""Compose Program bundles into the shared route-candidate review space."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.program_route_candidates import (
    compile_program_route_candidate_set,
)
from cascade_planner.application.program_route_optimizer import (
    optimize_program_route_candidates,
    program_route_portfolio_oracle,
)
from cascade_planner.orchestration.execution_program_review_materials import (
    compile_execution_program_review_materials,
)
from cascade_planner.orchestration.experimental_claim_review_materials import (
    compile_experimental_claim_review_materials,
)
from cascade_planner.orchestration.mechanism_program_review_materials import (
    compile_mechanism_program_review_materials,
)


def compile_program_candidate_review_materials(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    biocatalytic_bundle: Mapping[str, Any],
    *,
    biocatalytic_validations: Iterable[Mapping[str, Any]] = (),
    mechanism_validations: Iterable[Mapping[str, Any]] = (),
    execution_validations: Iterable[Mapping[str, Any]] = (),
    reported_candidate_packs: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build mechanism/execution bundles, common candidates, and Pareto views."""

    biocatalytic_rows = [dict(value) for value in biocatalytic_validations]
    mechanism_rows = [dict(value) for value in mechanism_validations]
    execution_rows = [dict(value) for value in execution_validations]
    mechanism_materials = compile_mechanism_program_review_materials(
        graph,
        route,
        projection,
        discovery,
        validations=mechanism_rows,
    )
    mechanism_bundle = mechanism_materials["mechanism_bundle"]
    execution_materials = compile_execution_program_review_materials(
        graph,
        route,
        projection,
        discovery,
        validations=execution_rows,
    )
    validation_rows = [*biocatalytic_rows, *execution_rows, *mechanism_rows]
    experimental_materials = compile_experimental_claim_review_materials(
        graph,
        route,
        projection,
        discovery,
        biocatalytic_bundle,
        execution_materials,
        mechanism_materials,
        validation_rows,
    )
    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        biocatalytic_bundle,
        validations=biocatalytic_rows,
        reported_candidate_packs=reported_candidate_packs,
        mechanism_bundle=mechanism_bundle,
        mechanism_validations=mechanism_rows,
        execution_bundle=execution_materials["execution_bundle"],
        execution_validations=execution_rows,
    )
    optimizer = optimize_program_route_candidates(candidate_set)
    return {
        **experimental_materials,
        "program_route_candidates": candidate_set,
        "program_optimizer": optimizer,
        "program_optimizer_oracle": program_route_portfolio_oracle(candidate_set, optimizer),
        **mechanism_materials,
        **execution_materials,
    }


__all__ = ["compile_program_candidate_review_materials"]
