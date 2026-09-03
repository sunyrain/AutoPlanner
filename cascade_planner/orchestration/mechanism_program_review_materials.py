"""Compose read-only review materials for restitched mechanism Programs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.mechanism_experiment_feedback import (
    compile_mechanism_experiment_feedback,
    mechanism_experiment_feedback_oracle,
)
from cascade_planner.application.mechanism_programs import (
    compile_mechanism_program_bundle,
    mechanism_program_bundle_oracle,
)
from cascade_planner.application.mechanism_validation_frontier import (
    compile_mechanism_validation_frontier,
)


def compile_mechanism_program_review_materials(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build mechanism bundle, frontier, feedback, and reprojection oracles."""

    rows = [dict(value) for value in validations]
    bundle = compile_mechanism_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=rows,
    )
    oracle = mechanism_program_bundle_oracle(
        graph,
        route,
        projection,
        discovery,
        bundle,
        validations=rows,
    )
    frontier = compile_mechanism_validation_frontier(graph, discovery, bundle)
    feedback = compile_mechanism_experiment_feedback(
        discovery, bundle, validations=rows
    )
    feedback_oracle = mechanism_experiment_feedback_oracle(
        discovery,
        bundle,
        feedback,
        validations=rows,
    )
    return {
        "mechanism_bundle": bundle,
        "mechanism_oracle": oracle,
        "mechanism_validation_frontier": frontier,
        "mechanism_experiment_feedback": feedback,
        "mechanism_feedback_oracle": feedback_oracle,
    }


__all__ = ["compile_mechanism_program_review_materials"]
