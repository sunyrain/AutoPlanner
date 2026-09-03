"""Recover experimental Claim CAS pins without trusting the RunIndex."""

from __future__ import annotations

from cascade_planner.application.experimental_claim_store import (
    ExperimentalClaimStore,
    ExperimentalClaimStoreError,
)
from cascade_planner.interfaces.replay_store_gc import replay_store_pinned_digests
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RunIndex


_REF_KEYS = (
    "source_graph_ref",
    "source_route_ref",
    "source_projection_ref",
    "source_discovery_ref",
    "validation_pack_ref",
    "claim_set_ref",
)


def experimental_claim_pinned_digests(
    paths: RuntimePaths, index: RunIndex
) -> set[str]:
    return replay_store_pinned_digests(
        paths,
        index,
        event_glob="*/.autoplanner/experimental_claims/events/sha256",
        store_marker=".autoplanner/experimental_claims/events",
        store_factory=lambda run_id, directory, artifacts: ExperimentalClaimStore(
            run_id=run_id,
            run_dir=directory,
            artifacts=artifacts,
        ),
        store_errors=(ExperimentalClaimStoreError,),
        ref_keys=_REF_KEYS,
        label="experimental_claim_store",
    )


__all__ = ["experimental_claim_pinned_digests"]
