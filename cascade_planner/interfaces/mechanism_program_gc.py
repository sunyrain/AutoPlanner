"""Recover mechanism Program CAS pins without trusting the RunIndex."""

from __future__ import annotations

from cascade_planner.application.mechanism_program_store import (
    MechanismProgramStore,
    MechanismProgramStoreError,
)
from cascade_planner.interfaces.replay_store_gc import replay_store_pinned_digests
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RunIndex


_REF_KEYS = (
    "source_graph_ref",
    "source_route_ref",
    "baseline_projection_ref",
    "discovery_ref",
    "bundle_ref",
    "validation_pack_ref",
)


def mechanism_program_pinned_digests(paths: RuntimePaths, index: RunIndex) -> set[str]:
    return replay_store_pinned_digests(
        paths,
        index,
        event_glob="*/.autoplanner/mechanism_programs/events/sha256",
        store_marker=".autoplanner/mechanism_programs/events/sha256",
        store_factory=lambda run_id, directory, artifacts: MechanismProgramStore(
            run_id=run_id,
            run_dir=directory,
            artifacts=artifacts,
        ),
        store_errors=(MechanismProgramStoreError,),
        ref_keys=_REF_KEYS,
        label="mechanism_program_store",
    )


__all__ = ["mechanism_program_pinned_digests"]
