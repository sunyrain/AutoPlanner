"""Bind validated mechanism Program admissions to one V4 campaign."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.mechanism_program_store import MechanismProgramStore
from cascade_planner.application.program_validation_routing import (
    partition_program_validations,
)
from cascade_planner.orchestration.program_innovation_materials import (
    compile_route_program_innovation_materials,
)


def admit_route_mechanism_programs(
    kernel: Any,
    graph_store: Any,
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    experience_library: Mapping[str, Any] | None = None,
    enable_mechanism_program_admission: bool = False,
) -> dict[str, Any]:
    validation_rows = [dict(value) for value in validations]
    _, _, mechanism_validations = partition_program_validations(validation_rows)
    materials = compile_route_program_innovation_materials(
        graph_store.load(),
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        mechanism_proposals=mechanism_proposals,
        validations=validation_rows,
        experience_library=experience_library,
    )
    return _store(kernel).admit(
        graph=materials["graph"],
        route=materials["route"],
        projection=materials["projection"],
        discovery=materials["discovery"],
        bundle=materials["mechanism_bundle"],
        validations=mechanism_validations,
        enable_mechanism_program_admission=enable_mechanism_program_admission,
    )


def mechanism_program_store_read(kernel: Any) -> dict[str, Any]:
    return {"replay": _store(kernel).replay()}


def _store(kernel: Any) -> MechanismProgramStore:
    return MechanismProgramStore(
        run_id=kernel.spec.run_id,
        run_dir=kernel.run_dir,
        artifacts=kernel.artifacts,
        index=kernel.index,
    )


__all__ = ["admit_route_mechanism_programs", "mechanism_program_store_read"]
