"""Build a self-contained, offline playback page for one retrosynthesis run."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

from cascade_planner.web.v4_live_synthesis import (
    _annotate_route_topology,
    canonical_smiles,
    project_live_synthesis,
    render_molecule_svg,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
SHOWCASE_TEMPLATES = {
    "interaction": STATIC_DIR / "run_showcase.html",
    "route": STATIC_DIR / "run_route_export.html",
    "graph": STATIC_DIR / "run_route_graph_export.html",
}
_DATA_MARKER = "__AUTOPLANNER_SHOWCASE_DATA__"
_STYLE_MARKER = "__AUTOPLANNER_EXPORT_STYLES__"
_EXPORT_STYLES = STATIC_DIR / "run_export.css"
_REPORT_NAMES = (
    "target-only-solve-report.json",
    "target_solve_report.json",
)


def build_run_export_html(
    *,
    run_dir: Path,
    job: Mapping[str, Any] | None = None,
    model_io_path: Path | None = None,
    export_kind: str = "interaction",
    branch_index: int = 1,
) -> str:
    """Compile one saved run into one of the three explicit export products."""

    export_kind = str(export_kind or "interaction").strip().casefold()
    if export_kind not in SHOWCASE_TEMPLATES:
        raise ValueError(f"showcase_export_kind_invalid:{export_kind}")
    if branch_index not in {1, 2, 3}:
        raise ValueError(f"showcase_branch_index_invalid:{branch_index}")

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"showcase_run_dir_missing:{run_dir}")
    report = _read_run_report(run_dir)
    target = dict(report.get("target") or {})
    source_job = dict(job or {})
    run_id = str(source_job.get("run_id") or report.get("run_id") or run_dir.name)
    target_name = str(
        source_job.get("target_name")
        or target.get("name")
        or report.get("target_name")
        or run_id
    )
    target_smiles = str(
        source_job.get("target_smiles")
        or target.get("canonical_smiles")
        or report.get("target_smiles")
        or ""
    )
    source_job.update(
        {
            "job_id": str(source_job.get("job_id") or f"export:{run_id}"),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "target_name": target_name,
            "target_smiles": target_smiles,
            "status": str(source_job.get("status") or "historical"),
            "execution_source": str(
                source_job.get("execution_source") or "saved_run_export"
            ),
        }
    )
    stream_path = model_io_path or (
        run_dir / ".autoplanner" / "director-workspace" / "model-io.jsonl"
    )
    if not stream_path.is_file():
        raise ValueError(f"showcase_model_io_missing:{stream_path}")

    projection = project_live_synthesis(
        model_io_path=stream_path,
        job=source_job,
        # The replay compiler also reconstructs host-accepted steps that older
        # runs saved only in proposal audits or later model inputs.  Always run
        # that authoritative reconstruction, then omit its frames from static
        # deliverables so route/graph exports remain compact.
        include_replay=True,
    )
    complete_replay = dict(projection.pop("replay", {}) or {})
    origin_counts = _enrich_route_provenance(projection, report)
    if export_kind == "interaction":
        # Playback can briefly expose draft structures that do not survive in
        # the final route.  Inline those molecules too, then keep the compact
        # final projection separate from the event stream in the bundle.
        projection["replay"] = complete_replay
    molecule_svgs, invalid_smiles = _render_molecule_library(projection)
    projection.pop("replay", None)
    replay = (
        complete_replay
        if export_kind == "interaction"
        else {
            "schema_version": "autoplanner.live_synthesis_replay.omitted.v1",
            "frame_count": 0,
            "frames": [],
            "omitted_for_static_export": True,
        }
    )
    bundle = {
        "schema_version": "autoplanner.run_export.v2",
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metadata": {
            "export_kind": export_kind,
            "branch_index": branch_index,
            "run_id": run_id,
            "job_id": str(source_job.get("job_id") or ""),
            "target_name": target_name,
            "target_smiles": str(projection.get("target_smiles") or target_smiles),
            "status": str(projection.get("status") or "historical"),
            "frame_count": int(replay.get("frame_count") or 0),
            "molecule_count": len(molecule_svgs),
            "invalid_molecule_count": len(invalid_smiles),
            "step_origin_counts": origin_counts,
        },
        "summary": _report_summary(report, projection),
        "projection": {
            "status": projection.get("status"),
            "phase": projection.get("phase"),
            "progress": projection.get("progress"),
            "strategies": projection.get("strategies") or [],
            "branches": projection.get("branches") or [],
            "usage": projection.get("usage") or {},
            "activities": projection.get("activities") or [],
            "model_output_count": projection.get("model_output_count") or 0,
            "semantics": projection.get("semantics") or {},
        },
        "replay": replay,
        "molecules": molecule_svgs,
        "invalid_smiles": invalid_smiles,
        "semantics": {
            "offline_single_file": True,
            "export_kind_is_explicit": True,
            "source_is_saved_model_and_host_events": True,
            "model_output_is_not_host_acceptance": True,
            "molecule_drawings_are_embedded_at_export_time": True,
            "step_origins_come_from_saved_candidate_lifecycle_records": True,
            "aizynthfinder_short_tails_are_never_inferred_from_step_position": True,
        },
    }
    template = SHOWCASE_TEMPLATES[export_kind].read_text(encoding="utf-8")
    if template.count(_DATA_MARKER) != 1:
        raise ValueError("showcase_template_data_marker_invalid")
    if template.count(_STYLE_MARKER) != 1:
        raise ValueError("showcase_template_style_marker_invalid")
    template = template.replace(
        _STYLE_MARKER,
        _EXPORT_STYLES.read_text(encoding="utf-8"),
    )
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    # A JSON string can legally contain ``</script>``.  Escaping the slash keeps
    # arbitrary saved model text from closing the inert data script element.
    payload = payload.replace("</", "<\\/")
    return template.replace(_DATA_MARKER, payload)


def export_run_showcase(
    *,
    run_dir: Path,
    output_path: Path,
    job: Mapping[str, Any] | None = None,
    export_kind: str = "interaction",
    branch_index: int = 1,
) -> dict[str, Any]:
    """Write a single-file playback page and return a compact receipt."""

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = build_run_export_html(
        run_dir=run_dir,
        job=job,
        export_kind=export_kind,
        branch_index=branch_index,
    )
    output_path.write_text(body, encoding="utf-8")
    return {
        "schema_version": "autoplanner.run_showcase_export_receipt.v1",
        "output_path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "self_contained": True,
        "export_kind": export_kind,
        "branch_index": branch_index,
    }


def showcase_filename(
    run_id: str,
    *,
    export_kind: str = "interaction",
    branch_index: int = 1,
) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(run_id or "")).strip(".-")
    suffix = {
        "interaction": "model-interaction-playback",
        "route": f"strategy-{branch_index}-static-route",
        "graph": "complete-route-graph",
    }.get(export_kind, "retrosynthesis-export")
    return f"{safe or 'retrosynthesis-run'}-{suffix}.html"


def _read_run_report(run_dir: Path) -> dict[str, Any]:
    for name in _REPORT_NAMES:
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"showcase_run_report_invalid:{path}") from exc
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _report_summary(
    report: Mapping[str, Any], projection: Mapping[str, Any]
) -> dict[str, Any]:
    model_cost = dict(report.get("model_cost") or {})
    paper = dict(report.get("paper_equivalent") or {})
    disposition = dict(report.get("current_disposition") or {})
    branches = [dict(value) for value in projection.get("branches") or []]
    usage = dict(projection.get("usage") or {})
    return {
        "model_invocations": int(
            model_cost.get("model_invocations")
            or usage.get("model_invocations")
            or projection.get("model_output_count")
            or 0
        ),
        "input_tokens": int(
            model_cost.get("input_tokens") or usage.get("input_tokens") or 0
        ),
        "output_tokens": int(
            model_cost.get("output_tokens") or usage.get("output_tokens") or 0
        ),
        "wall_time_s": float(model_cost.get("wall_time_s") or 0.0),
        "accepted_expansions": int(report.get("accepted_expansion_count") or 0),
        "paper_equivalent_solved_route_count": int(
            paper.get("paper_equivalent_solved_route_count") or 0
        ),
        "paper_equivalent_solved": bool(
            paper.get("paper_equivalent_solved")
            or report.get("paper_equivalent_solved")
        ),
        "scientific_state": str(disposition.get("state") or ""),
        "branch_step_counts": {
            str(value.get("branch_index") or index): len(value.get("steps") or [])
            for index, value in enumerate(branches, start=1)
        },
    }


def _render_molecule_library(
    projection: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    values = {str(projection.get("target_smiles") or "").strip()}
    replay = dict(projection.get("replay") or {})
    branches: list[Mapping[str, Any]] = [
        value
        for value in projection.get("branches") or []
        if isinstance(value, Mapping)
    ]
    for frame in replay.get("frames") or []:
        if not isinstance(frame, Mapping):
            continue
        branches.extend(
            value
            for value in frame.get("branch_updates") or []
            if isinstance(value, Mapping)
        )
    for branch in branches:
        steps = list(branch.get("steps") or [])
        pending = branch.get("pending_step")
        if isinstance(pending, Mapping):
            steps.append(pending)
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            values.add(str(step.get("product_smiles") or "").strip())
            values.update(
                str(value).strip()
                for value in step.get("precursor_smiles") or []
                if str(value).strip()
            )
    rendered: dict[str, str] = {}
    invalid: list[str] = []
    for smiles in sorted(value for value in values if value):
        svg, valid = render_molecule_svg(smiles)
        rendered[smiles] = svg
        if not valid:
            invalid.append(smiles)
    return rendered, invalid


_STEP_ORIGIN_LABELS = {
    "aizynthfinder_short_tail": "AIz 收尾",
    "large_model": "大模型步",
    "unknown": "来源未记录",
}


def _enrich_route_provenance(
    projection: dict[str, Any],
    report: Mapping[str, Any],
) -> dict[str, int]:
    """Attach durable step provenance and reachable native short tails.

    The saved model/Host projection and the canonical candidate lifecycle are
    separate authorities.  We join them by canonical reaction identity.  A
    provider step is appended only when it is materialized, explicitly records
    ``mode=short_tail``, belongs to the same canonical route family, and its
    product is an actual open leaf in the displayed branch.  This deliberately
    avoids the misleading historical shortcut of calling the last N steps an
    AiZ tail.
    """

    lifecycle = dict(report.get("candidate_lifecycle") or {})
    records = [
        dict(value)
        for value in lifecycle.get("records") or []
        if isinstance(value, Mapping)
    ]
    records_by_reaction: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for record in records:
        records_by_reaction.setdefault(_reaction_identity(record), []).append(record)

    short_tails = [
        record
        for record in records
        if dict(record.get("materialization") or {}).get("materialized") is True
        and "aizynthfinder_short_tail" in _record_step_origins((record,))
    ]
    aggregate = {key: 0 for key in _STEP_ORIGIN_LABELS}
    for raw_branch in projection.get("branches") or []:
        if not isinstance(raw_branch, dict):
            continue
        steps = [
            value
            for value in raw_branch.get("steps") or []
            if isinstance(value, dict)
        ]
        route_family_ids: set[str] = set()
        existing_reactions: set[tuple[str, tuple[str, ...]]] = set()
        for step in steps:
            identity = _reaction_identity(step)
            existing_reactions.add(identity)
            matches = records_by_reaction.get(identity, [])
            route_family_ids.update(
                str(route_family_id)
                for record in matches
                for route_family_id in record.get("route_family_ids") or []
                if str(route_family_id)
            )
            _annotate_step_origin(
                step,
                matches,
                fallback_from_step_id=True,
            )

        pending = [
            record
            for record in short_tails
            if _reaction_identity(record) not in existing_reactions
            and route_family_ids.intersection(
                str(value) for value in record.get("route_family_ids") or []
            )
        ]
        # A short tail can contain multiple provider steps.  Recompute leaves
        # after every pass so a downstream step becomes reachable only after
        # its parent provider step has been admitted to this displayed route.
        while pending:
            leaves = _open_leaf_identities(steps)
            reachable = [
                record
                for record in pending
                if _normal_smiles(record.get("product_smiles")) in leaves
            ]
            if not reachable:
                break
            for record in sorted(
                reachable,
                key=lambda value: _record_origin_sort_key(value),
            ):
                step = _step_from_lifecycle_record(record)
                _annotate_step_origin(step, (record,), fallback_from_step_id=False)
                steps.append(step)
                pending.remove(record)

        for index, step in enumerate(steps, start=1):
            step["index"] = index
        _annotate_route_topology(steps)
        raw_branch["steps"] = steps
        raw_branch["step_origin_counts"] = _step_origin_counts(steps)
        for key, value in raw_branch["step_origin_counts"].items():
            aggregate[key] += value
    return aggregate


def _normal_smiles(value: Any) -> str:
    raw = str(value or "").strip()
    return canonical_smiles(raw) or raw


def _reaction_identity(value: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        _normal_smiles(value.get("product_smiles")),
        tuple(
            sorted(
                _normal_smiles(precursor)
                for precursor in value.get("precursor_smiles") or []
                if str(precursor).strip()
            )
        ),
    )


def _record_step_origins(records: Any) -> set[str]:
    origins: set[str] = set()
    for record in records or []:
        for raw_origin in record.get("origin_records") or []:
            if not isinstance(raw_origin, Mapping):
                continue
            origin_kind = str(raw_origin.get("origin_kind") or "").casefold()
            if origin_kind == "codex_global_director":
                origins.add("large_model")
            elif origin_kind == "aizynthfinder":
                metadata = dict(raw_origin.get("provider_reaction_metadata") or {})
                if str(metadata.get("mode") or "").casefold() == "short_tail":
                    origins.add("aizynthfinder_short_tail")
    return origins


def _annotate_step_origin(
    step: dict[str, Any],
    records: Any,
    *,
    fallback_from_step_id: bool,
) -> None:
    origins = _record_step_origins(records)
    basis = "candidate_lifecycle.origin_records" if origins else ""
    step_id = str(step.get("step_id") or "").casefold()
    if not origins and fallback_from_step_id and step_id.startswith(("codex:", "repair:")):
        origins.add("large_model")
        basis = "saved_model_step_namespace"
    if not origins:
        origins.add("unknown")
        basis = "no_durable_step_origin_record"
    ordered = [
        value
        for value in ("aizynthfinder_short_tail", "large_model", "unknown")
        if value in origins
    ]
    step["step_origins"] = [
        {"kind": value, "label": _STEP_ORIGIN_LABELS[value]}
        for value in ordered
    ]
    step["step_origin"] = ordered[0] if len(ordered) == 1 else "mixed"
    step["step_origin_label"] = " + ".join(
        _STEP_ORIGIN_LABELS[value] for value in ordered
    )
    step["step_origin_basis"] = basis


def _record_origin_sort_key(record: Mapping[str, Any]) -> tuple[int, str]:
    proposal_id = next(
        (
            str(value.get("proposal_id") or "")
            for value in record.get("origin_records") or []
            if isinstance(value, Mapping)
            and str(value.get("origin_kind") or "").casefold() == "aizynthfinder"
        ),
        "",
    )
    match = re.search(r":step:(\d+)$", proposal_id)
    return (int(match.group(1)) if match else 1_000_000, proposal_id)


def _step_from_lifecycle_record(record: Mapping[str, Any]) -> dict[str, Any]:
    origin = next(
        (
            dict(value)
            for value in record.get("origin_records") or []
            if isinstance(value, Mapping)
            and str(value.get("origin_kind") or "").casefold() == "aizynthfinder"
            and str(
                dict(value.get("provider_reaction_metadata") or {}).get("mode") or ""
            ).casefold()
            == "short_tail"
        ),
        {},
    )
    metadata = dict(origin.get("provider_reaction_metadata") or {})
    return {
        "step_id": str(
            origin.get("proposal_id")
            or record.get("candidate_id")
            or record.get("edge_id")
            or "aizynthfinder:short-tail"
        ),
        "product_smiles": str(record.get("product_smiles") or ""),
        "precursor_smiles": [
            str(value)
            for value in record.get("precursor_smiles") or []
            if str(value).strip()
        ],
        "reaction_family": str(
            origin.get("transformation_hypothesis")
            or metadata.get("classification")
            or "AiZynthFinder template disconnection"
        ),
        "transformation_rationale": str(
            origin.get("transformation_hypothesis") or ""
        ),
        "conditions": [],
        "catalyst": "",
        "status": "host_materialized",
    }


def _open_leaf_identities(steps: list[dict[str, Any]]) -> set[str]:
    products = {
        _normal_smiles(step.get("product_smiles"))
        for step in steps
        if str(step.get("product_smiles") or "").strip()
    }
    return {
        identity
        for step in steps
        for value in step.get("precursor_smiles") or []
        if (identity := _normal_smiles(value)) and identity not in products
    }


def _step_origin_counts(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in _STEP_ORIGIN_LABELS}
    for step in steps:
        kinds = {
            str(value.get("kind") or "")
            for value in step.get("step_origins") or []
            if isinstance(value, Mapping)
        }
        for key in counts:
            counts[key] += int(key in kinds)
    return counts


__all__ = [
    "build_run_export_html",
    "export_run_showcase",
    "showcase_filename",
]
