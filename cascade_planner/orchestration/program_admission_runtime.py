"""Bind the shadow Program store to one canonical V4 campaign service."""

from __future__ import annotations

from typing import Any

from cascade_planner.application.transformation_program_store import (
    TransformationProgramStore,
)
from cascade_planner.application.transformation_programs import (
    program_projection_oracle,
    project_canonical_graph_to_programs,
)


def program_projection_read(graph_store: Any) -> dict[str, Any]:
    graph = graph_store.load()
    projection = project_canonical_graph_to_programs(graph)
    return {
        "projection": projection,
        "oracle": program_projection_oracle(graph, projection),
    }


def program_store_read(kernel: Any, graph_store: Any) -> dict[str, Any]:
    store = _store(kernel)
    graph = graph_store.load()
    return {"status": store.status(graph), "replay": store.replay()}


def admit_program_projection(
    kernel: Any,
    graph_store: Any,
    *,
    enable_program_admission: bool = False,
) -> dict[str, Any]:
    return _store(kernel).admit(
        graph_store.load(),
        enable_program_admission=enable_program_admission,
    )


def _store(kernel: Any) -> TransformationProgramStore:
    return TransformationProgramStore(
        run_id=kernel.spec.run_id,
        run_dir=kernel.run_dir,
        artifacts=kernel.artifacts,
        index=kernel.index,
    )


__all__ = [
    "admit_program_projection",
    "program_projection_read",
    "program_store_read",
]
