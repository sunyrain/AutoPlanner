"""Bind validated biocatalytic Program admissions to one V4 campaign."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.biocatalytic_program_store import (
    BiocatalyticProgramStore,
)
from cascade_planner.orchestration.program_innovation_materials import (
    compile_route_program_innovation_materials,
)


def admit_route_biocatalytic_programs(
    kernel: Any,
    graph_store: Any,
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_biocatalytic_program_admission: bool = False,
    experience_library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    materials = compile_route_program_innovation_materials(
        graph_store.load(),
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        mechanism_proposals=mechanism_proposals,
        validations=validations,
        experience_library=experience_library,
    )
    return _store(kernel).admit(
        graph=materials["graph"],
        route=materials["route"],
        projection=materials["projection"],
        discovery=materials["discovery"],
        bundle=materials["bundle"],
        validations=materials["validations"],
        enable_biocatalytic_program_admission=(
            enable_biocatalytic_program_admission
        ),
    )


def biocatalytic_program_store_read(kernel: Any) -> dict[str, Any]:
    return {"replay": _store(kernel).replay()}


def _store(kernel: Any) -> BiocatalyticProgramStore:
    return BiocatalyticProgramStore(
        run_id=kernel.spec.run_id,
        run_dir=kernel.run_dir,
        artifacts=kernel.artifacts,
        index=kernel.index,
    )


__all__ = [
    "admit_route_biocatalytic_programs",
    "biocatalytic_program_store_read",
]
