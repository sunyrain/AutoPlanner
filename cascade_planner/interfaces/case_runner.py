"""One-command, model-free execution of an exact-source case dossier."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths

from .case_dossier import compile_case_dossier, local_rxnmapper
from .replay_pack import run_replay_pack


def run_case_dossier(
    dossier: str | Path | Mapping[str, Any],
    *,
    paths: RuntimePaths,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    map_missing: bool = False,
) -> dict[str, Any]:
    """Compile, replay, close, and export a dossier without hosted calls."""

    started = perf_counter()
    pack = compile_case_dossier(
        dossier,
        reaction_mapper=local_rxnmapper if map_missing else None,
    )
    compiled_at = perf_counter()
    replay = run_replay_pack(
        pack,
        paths=paths,
        run_id=run_id,
        run_dir=run_dir,
    )
    replayed_at = perf_counter()
    exported = CampaignGateway(paths).export(
        str(replay["run_id"]),
        run_dir=str(replay["run_dir"]),
        output_dir=str(output_dir) if output_dir is not None else None,
    )
    exported_at = perf_counter()
    return {
        "schema_version": "retrosynthesis_case_run_result.v1",
        "case_id": pack["case_id"],
        "accepted": replay["accepted"],
        "status": replay["status"],
        "run_id": replay["run_id"],
        "run_dir": replay["run_dir"],
        "pack_sha256": pack["content_sha256"],
        "observed": replay["observed"],
        "checks": replay["checks"],
        "export": exported,
        "model_invocations": replay["observed"]["model_invocations"],
        "visual_invocations": replay["observed"]["visual_invocations"],
        "timing_seconds": {
            "compile": round(compiled_at - started, 6),
            "replay": round(replayed_at - compiled_at, 6),
            "export": round(exported_at - replayed_at, 6),
            "total": round(exported_at - started, 6),
        },
    }


__all__ = ["run_case_dossier"]
