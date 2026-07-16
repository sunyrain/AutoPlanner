"""Deterministically rebuild the application inputs required by the Claim store."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.application.biocatalytic_programs import (
    biocatalytic_program_bundle_oracle,
    compile_biocatalytic_program_bundle,
)
from cascade_planner.application.execution_capability_feedback import (
    compile_execution_capability_feedback,
    execution_capability_feedback_oracle,
)
from cascade_planner.application.execution_programs import (
    compile_execution_program_bundle,
)
from cascade_planner.application.experimental_claims import (
    compile_experimental_claim_set,
    experimental_claim_set_oracle,
)
from cascade_planner.application.mechanism_experiment_feedback import (
    compile_mechanism_experiment_feedback,
    mechanism_experiment_feedback_oracle,
)
from cascade_planner.application.mechanism_programs import (
    compile_mechanism_program_bundle,
)
from cascade_planner.application.program_validation_routing import (
    partition_program_validations,
)


def reproject_experimental_claim_materials(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    validations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild all three domain sources and the exact unified Claim set."""

    rows = [dict(value) for value in validations]
    biocatalytic, execution, mechanism = partition_program_validations(rows)
    biocatalytic_bundle = compile_biocatalytic_program_bundle(
        graph, route, projection, discovery, validations=biocatalytic
    )
    biocatalytic_oracle = biocatalytic_program_bundle_oracle(
        graph,
        route,
        projection,
        discovery,
        biocatalytic_bundle,
        validations=biocatalytic,
    )
    execution_bundle = compile_execution_program_bundle(
        graph, route, projection, discovery, validations=execution
    )
    execution_feedback = compile_execution_capability_feedback(
        discovery, execution_bundle, validations=execution
    )
    execution_oracle = execution_capability_feedback_oracle(
        discovery, execution_bundle, execution_feedback, validations=execution
    )
    mechanism_bundle = compile_mechanism_program_bundle(
        graph, route, projection, discovery, validations=mechanism
    )
    mechanism_feedback = compile_mechanism_experiment_feedback(
        discovery, mechanism_bundle, validations=mechanism
    )
    mechanism_oracle = mechanism_experiment_feedback_oracle(
        discovery, mechanism_bundle, mechanism_feedback, validations=mechanism
    )
    claim_set = compile_experimental_claim_set(
        biocatalytic_bundle,
        biocatalytic_oracle,
        execution_feedback,
        execution_oracle,
        mechanism_feedback,
        mechanism_oracle,
        validations=rows,
    )
    claim_oracle = experimental_claim_set_oracle(
        biocatalytic_bundle,
        biocatalytic_oracle,
        execution_feedback,
        execution_oracle,
        mechanism_feedback,
        mechanism_oracle,
        claim_set,
        validations=rows,
    )
    return {
        "biocatalytic_bundle": biocatalytic_bundle,
        "biocatalytic_oracle": biocatalytic_oracle,
        "execution_bundle": execution_bundle,
        "execution_capability_feedback": execution_feedback,
        "execution_feedback_oracle": execution_oracle,
        "mechanism_bundle": mechanism_bundle,
        "mechanism_experiment_feedback": mechanism_feedback,
        "mechanism_feedback_oracle": mechanism_oracle,
        "experimental_claims": claim_set,
        "experimental_claims_oracle": claim_oracle,
    }


__all__ = ["reproject_experimental_claim_materials"]
