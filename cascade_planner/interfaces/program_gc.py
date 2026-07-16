"""Recover Program admission CAS pins without trusting the mutable RunIndex."""

from __future__ import annotations

from cascade_planner.application.transformation_program_store import (
    TransformationProgramStore,
    TransformationProgramStoreError,
)
from cascade_planner.interfaces.replay_store_gc import replay_store_pinned_digests
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RunIndex


def program_store_pinned_digests(paths: RuntimePaths, index: RunIndex) -> set[str]:
    """Replay all discovered Program stores and return their immutable refs."""

    return replay_store_pinned_digests(
        paths,
        index,
        event_glob="*/.autoplanner/program_store/events/sha256",
        store_marker=".autoplanner/program_store/events",
        store_factory=lambda run_id, directory, artifacts: TransformationProgramStore(
            run_id=run_id,
            run_dir=directory,
            artifacts=artifacts,
        ),
        store_errors=(TransformationProgramStoreError,),
        ref_keys=("source_graph_ref", "projection_ref"),
        label="program_store",
    )


__all__ = ["program_store_pinned_digests"]
