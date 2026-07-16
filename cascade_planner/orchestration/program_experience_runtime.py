"""Bind cross-campaign Program experience memory to replayed Claim stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cascade_planner.application.experimental_claim_store import ExperimentalClaimStore
from cascade_planner.application.program_experience import (
    synchronize_program_experience_library,
)
from cascade_planner.application.program_experience_store import (
    read_program_experience_library,
)


def synchronize_program_experience(
    kernel: Any,
    *,
    library_path: str | Path,
    enable_program_experience_learning: bool = False,
) -> dict[str, Any]:
    if enable_program_experience_learning is not True:
        raise ValueError("program_experience_learning_disabled:explicit_enable_required")
    claim_store = ExperimentalClaimStore(
        run_id=kernel.spec.run_id,
        run_dir=kernel.run_dir,
        artifacts=kernel.artifacts,
        index=kernel.index,
    )
    return synchronize_program_experience_library(
        library_path, claim_store.experience_sources()
    )


def program_experience_library_read(library_path: str | Path) -> dict[str, Any]:
    library, error = read_program_experience_library(library_path)
    if error:
        raise ValueError(error)
    return {
        "library_path": str(Path(library_path).expanduser().resolve()),
        "library": library,
        "semantics": {
            "read_is_non_mutating": True,
            "memory_is_proposal_ranking_only": True,
        },
    }


__all__ = ["program_experience_library_read", "synchronize_program_experience"]
