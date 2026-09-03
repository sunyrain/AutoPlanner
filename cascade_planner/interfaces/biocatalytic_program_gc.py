"""Recover biocatalytic Program CAS pins without trusting the RunIndex."""

from __future__ import annotations

from cascade_planner.application.biocatalytic_program_store import (
    BiocatalyticProgramStore,
    BiocatalyticProgramStoreError,
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


def biocatalytic_program_pinned_digests(
    paths: RuntimePaths, index: RunIndex
) -> set[str]:
    return replay_store_pinned_digests(
        paths,
        index,
        event_glob="*/.autoplanner/bio_programs/e",
        store_marker=".autoplanner/bio_programs/e",
        store_factory=lambda run_id, directory, artifacts: BiocatalyticProgramStore(
            run_id=run_id,
            run_dir=directory,
            artifacts=artifacts,
        ),
        store_errors=(BiocatalyticProgramStoreError,),
        ref_keys=_REF_KEYS,
        label="biocatalytic_program_store",
    )


__all__ = ["biocatalytic_program_pinned_digests"]
