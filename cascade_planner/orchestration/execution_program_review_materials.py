"""Compose read-only review materials for whole-cell and hybrid Programs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.execution_capability_feedback import (
    compile_execution_capability_feedback,
    execution_capability_feedback_oracle,
)
from cascade_planner.application.execution_programs import (
    compile_execution_program_bundle,
    execution_program_bundle_oracle,
)
from cascade_planner.application.execution_validation_frontier import (
    compile_execution_validation_frontier,
)


def compile_execution_program_review_materials(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build execution bundle, frontier, feedback, and both reprojection oracles."""

    rows = [dict(value) for value in validations]
    bundle = compile_execution_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=rows,
    )
    oracle = execution_program_bundle_oracle(
        graph,
        route,
        projection,
        discovery,
        bundle,
        validations=rows,
    )
    frontier = compile_execution_validation_frontier(graph, discovery, bundle)
    feedback = compile_execution_capability_feedback(
        discovery, bundle, validations=rows
    )
    feedback_oracle = execution_capability_feedback_oracle(
        discovery,
        bundle,
        feedback,
        validations=rows,
    )
    return {
        "execution_bundle": bundle,
        "execution_oracle": oracle,
        "execution_validation_frontier": frontier,
        "execution_capability_feedback": feedback,
        "execution_feedback_oracle": feedback_oracle,
    }


__all__ = [
    "compile_execution_program_review_materials",
]
