"""Package native run replays into one offline, same-target collection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from cascade_planner.web.route_display import enhance_route_html

from cascade_planner.web.v4_showcase_export import (
    STATIC_DIR,
    build_run_export_bundle,
    canonical_smiles,
    normalize_branch_indices,
)


@dataclass(frozen=True, slots=True)
class RunReplaySelection:
    run_dir: Path
    branch_indices: tuple[int, ...] = (1, 2, 3)
    label: str = ""


def build_run_replay_collection_html(
    selections: Iterable[RunReplaySelection],
    *,
    title: str = "逆合成路线合集",
    initial_branch_index: int | None = None,
) -> str:
    """Keep each run's event timeline and final state in its native namespace.

    The collection only switches between complete native viewers. It never
    concatenates independent runs' branch IDs, events, or chemistry graphs.
    Selection order determines the initially displayed run.
    """

    entries: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    target_smiles = ""
    for selection in selections:
        branches = normalize_branch_indices(selection.branch_indices)
        bundle = build_run_export_bundle(
            run_dir=selection.run_dir,
            export_kind="interaction",
            branch_indices=branches,
        )
        metadata = bundle["metadata"]
        run_id = str(metadata["run_id"])
        if run_id in run_ids:
            raise ValueError(f"replay_collection_duplicate_run:{run_id}")
        run_ids.add(run_id)
        canonical = canonical_smiles(str(metadata["target_smiles"]))
        if not canonical or (target_smiles and canonical != target_smiles):
            raise ValueError("replay_collection_target_mismatch")
        target_smiles = canonical
        entries.append(
            {
                "label": selection.label or run_id,
                "bundle": bundle,
            }
        )
    if not entries:
        raise ValueError("replay_collection_empty")
    initial_branches = entries[0]["bundle"]["metadata"]["branch_indices"]
    initial_branch = (
        initial_branches[0]
        if initial_branch_index is None
        else initial_branch_index
    )
    if initial_branch not in initial_branches:
        raise ValueError("replay_collection_initial_branch_not_selected")
    viewer_template = enhance_route_html(
        (STATIC_DIR / "run_showcase.html").read_text(encoding="utf-8")
    )
    viewer_template = viewer_template.replace(
        "__AUTOPLANNER_EXPORT_STYLES__",
        (STATIC_DIR / "run_export.css").read_text(encoding="utf-8"),
    )
    payload = {
        "schema_version": "autoplanner.run_replay_collection.v1",
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "title": title,
        "target_smiles": target_smiles,
        "initial_branch_index": initial_branch,
        "run_count": len(entries),
        "route_count": sum(
            len(entry["bundle"]["metadata"]["branch_indices"])
            for entry in entries
        ),
        "entries": entries,
        "viewer_template": viewer_template,
    }
    template = (STATIC_DIR / "run_replay_collection.html").read_text(encoding="utf-8")
    return template.replace(
        "__AUTOPLANNER_COLLECTION_DATA__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
            "</", "<\\/"
        ),
    )


def export_run_replay_collection(
    selections: Iterable[RunReplaySelection],
    *,
    output_path: Path,
    title: str = "逆合成路线合集",
    initial_branch_index: int | None = None,
) -> dict[str, Any]:
    body = build_run_replay_collection_html(
        selections,
        title=title,
        initial_branch_index=initial_branch_index,
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "self_contained": True,
        "export_kind": "interaction_collection",
    }
