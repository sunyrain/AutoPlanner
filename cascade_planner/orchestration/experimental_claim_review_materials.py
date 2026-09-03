"""Compose unified experiment Claims and exact-boundary calibration views."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.application.biocatalytic_programs import (
    biocatalytic_program_bundle_oracle,
)
from cascade_planner.application.capability_applicability_calibration import (
    capability_calibration_oracle,
    compile_capability_applicability_calibration,
)
from cascade_planner.application.experimental_claims import (
    compile_experimental_claim_set,
    experimental_claim_set_oracle,
)
from cascade_planner.application.program_validation_routing import (
    partition_program_validations,
)


def compile_experimental_claim_review_materials(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    biocatalytic_bundle: Mapping[str, Any],
    execution_materials: Mapping[str, Any],
    mechanism_materials: Mapping[str, Any],
    validations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one deterministic Claim/calibration bundle without persistence."""

    biocatalytic_validations, _execution, _mechanism = partition_program_validations(validations)
    biocatalytic_oracle = biocatalytic_program_bundle_oracle(
        graph,
        route,
        projection,
        discovery,
        biocatalytic_bundle,
        validations=biocatalytic_validations,
    )
    claim_set = compile_experimental_claim_set(
        biocatalytic_bundle,
        biocatalytic_oracle,
        execution_materials["execution_capability_feedback"],
        execution_materials["execution_feedback_oracle"],
        mechanism_materials["mechanism_experiment_feedback"],
        mechanism_materials["mechanism_feedback_oracle"],
        validations=validations,
    )
    claim_oracle = experimental_claim_set_oracle(
        biocatalytic_bundle,
        biocatalytic_oracle,
        execution_materials["execution_capability_feedback"],
        execution_materials["execution_feedback_oracle"],
        mechanism_materials["mechanism_experiment_feedback"],
        mechanism_materials["mechanism_feedback_oracle"],
        claim_set,
        validations=validations,
    )
    calibration = compile_capability_applicability_calibration(claim_set)
    return {
        "biocatalytic_oracle": biocatalytic_oracle,
        "experimental_claims": claim_set,
        "experimental_claims_oracle": claim_oracle,
        "capability_calibration": calibration,
        "capability_calibration_oracle": capability_calibration_oracle(claim_set, calibration),
    }


__all__ = ["compile_experimental_claim_review_materials"]
