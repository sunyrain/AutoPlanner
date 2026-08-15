#!/usr/bin/env python3
"""Summarize a sharded AutoPlanner run against a SynthAtlas snapshot.

This is a competitive screen, not a live SynthEx reproduction.  The public
snapshot contributes fixed route inputs and the live root contributes one or
more disjoint AutoPlanner panel shards.  Incomplete and failed targets remain
in every denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


LEVELS = tuple(f"C{index}" for index in range(7))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preflight-status", required=True)
    parser.add_argument("--live-root", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = _json(Path(args.manifest))
    cases = [dict(row) for row in manifest.get("cases") or []]
    case_ids = [str(row.get("case_id") or "") for row in cases]
    if not case_ids or len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise SystemExit("manifest case identities are empty or duplicated")

    preflight = _json(Path(args.preflight_status))
    preflight_rows = _rows_by_case(preflight)
    clean_ids = [
        case_id
        for case_id in case_ids
        if dict(preflight_rows.get(case_id) or {}).get("status") == "preflight_passed"
    ]

    live_root = Path(args.live_root)
    shard_paths = sorted(live_root.glob("shard_*/panel-status.json"))
    if not shard_paths:
        raise SystemExit("no live panel shards found")
    live_rows: dict[str, dict[str, Any]] = {}
    shard_bindings: list[dict[str, Any]] = []
    for path in shard_paths:
        status = _json(path)
        selected = list(dict(status.get("selection") or {}).get("selected_case_ids") or [])
        rows = _rows_by_case(status, selected_case_ids=selected)
        if selected != list(rows):
            raise SystemExit(f"panel row order/selection mismatch: {path}")
        overlap = sorted(set(live_rows).intersection(rows))
        if overlap:
            raise SystemExit(f"live shard case overlap: {overlap}")
        live_rows.update(rows)
        shard_bindings.append(
            {
                "path": str(path.resolve()),
                "case_ids": selected,
                "snapshot_sha256": str(
                    dict(status.get("frozen_snapshot") or {}).get("content_sha256") or ""
                ),
                "ablation": str(status.get("ablation") or ""),
                "model": str(status.get("model") or ""),
                "reasoning_effort": str(status.get("reasoning_effort") or ""),
            }
        )
    if set(live_rows) != set(case_ids):
        missing = sorted(set(case_ids) - set(live_rows))
        extra = sorted(set(live_rows) - set(case_ids))
        raise SystemExit(f"live shard coverage mismatch: missing={missing}, extra={extra}")

    external_root = Path(args.external_root)
    external_summary = _json(external_root / "summary.json")
    external_rows = {
        str(row.get("case_id") or ""): row
        for row in (
            _json(path) for path in sorted((external_root / "cases").glob("*.json"))
        )
    }

    subsets = {
        "all_frozen_50": case_ids,
        "strict_repository_clean": clean_ids,
    }
    subset_summaries = {
        name: _summarize_subset(ids, live_rows=live_rows, external_rows=external_rows)
        for name, ids in subsets.items()
    }
    result: dict[str, Any] = {
        "schema_version": "synthatlas_competitive_screen_summary.v1",
        "claim_boundary": (
            "Competitive screen against frozen public SynthAtlas routes; not a live "
            "SynthEx runtime reproduction and not an experimental-success comparison."
        ),
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_case_count": len(case_ids),
        "strict_repository_clean_count": len(clean_ids),
        "known_repository_overlap_count": len(case_ids) - len(clean_ids),
        "shards": shard_bindings,
        "external_snapshot": {
            "status": str(external_summary.get("status") or "missing"),
            "case_count": int(external_summary.get("case_count") or 0),
            "route_count": int(external_summary.get("route_count") or 0),
            "route_level_closure_counts": dict(external_summary.get("closure_counts") or {}),
            "failure_taxonomy": dict(external_summary.get("failure_taxonomy") or {}),
            "resource_usage": dict(external_summary.get("resource_usage") or {}),
        },
        "subsets": subset_summaries,
        "paper_reported_reference": {
            "target_count": 1098,
            "SynthEx_strategy_layer_stock_closure_rate": 0.250,
            "SynthEx_stitched_stock_closure_rate": 0.639,
            "comparison_warning": (
                "Paper rates use SynthEx stock closure and are not directly equivalent to "
                "AutoPlanner C2-C6 host-authority gates."
            ),
        },
        "semantics": {
            "all_manifest_targets_remain_in_denominator": True,
            "incomplete_failed_timeout_and_partial_are_retained": True,
            "target_level_and_route_level_metrics_are_separate": True,
            "strict_clean_subset_is_predeclared_from_preflight": True,
            "public_snapshot_generation_cost_is_not_imputed": True,
        },
    }
    result["content_sha256"] = _digest(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _rows_by_case(
    status: Mapping[str, Any],
    *,
    selected_case_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    target_rows = list(dict(status.get("targets") or {}).values())
    if selected_case_ids is not None and len(selected_case_ids) != len(target_rows):
        raise SystemExit("panel selection/target row count mismatch")
    for index, raw in enumerate(target_rows):
        row = dict(raw or {})
        case_id = str(row.get("case_id") or "")
        if not case_id and selected_case_ids is not None:
            case_id = str(selected_case_ids[index])
            row["case_id"] = case_id
        if not case_id or case_id in rows:
            raise SystemExit("panel target row has empty or duplicated case_id")
        rows[case_id] = row
    return rows


def _live_levels(row: Mapping[str, Any]) -> dict[str, bool]:
    counts = dict(row.get("route_counts") or {})
    projection = dict(row.get("fixed_cutoff_projection") or {})
    milestones = dict(projection.get("milestones") or {})
    return {
        "C0": int(counts.get("target_rooted_route_count") or 0) > 0,
        "C1": int(counts.get("canonical_materialized_route_count") or 0) > 0,
        "C2": int(counts.get("strict_host_validated_route_count") or 0) > 0,
        "C3": int(counts.get("exact_procedure_route_count") or 0) > 0,
        "C4": int(counts.get("condition_complete_route_count") or 0) > 0,
        "C5": int(counts.get("strict_stock_closed_route_count") or 0) > 0,
        "C6": milestones.get("experiment:positive_exact_boundary_claim") is True,
    }


def _runner_failure_class(row: Mapping[str, Any]) -> str:
    if "MemoryError" in str(row.get("error") or ""):
        return "runner_failed:memory_error"
    return "runner_failed:unclassified"


def _external_levels(row: Mapping[str, Any]) -> dict[str, bool]:
    counts = dict(row.get("closure_counts") or {})
    return {**{level: int(counts.get(level) or 0) > 0 for level in LEVELS[:-1]}, "C6": False}


def _summarize_subset(
    case_ids: list[str],
    *,
    live_rows: Mapping[str, Mapping[str, Any]],
    external_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    live_statuses = Counter(str(live_rows[case_id].get("status") or "unknown") for case_id in case_ids)
    live_completed = [
        case_id
        for case_id in case_ids
        if str(live_rows[case_id].get("status") or "") == "completed"
    ]
    external_completed = [case_id for case_id in case_ids if case_id in external_rows]
    live_level_counts = {
        level: sum(_live_levels(live_rows[case_id])[level] for case_id in case_ids)
        for level in LEVELS
    }
    external_level_counts = {
        level: sum(_external_levels(external_rows[case_id])[level] for case_id in external_completed)
        for level in LEVELS
    }
    paired_ids = [case_id for case_id in live_completed if case_id in external_rows]
    paired = {}
    for level in LEVELS:
        cells = Counter()
        for case_id in paired_ids:
            external = _external_levels(external_rows[case_id])[level]
            live = _live_levels(live_rows[case_id])[level]
            cells[f"external_{int(external)}_live_{int(live)}"] += 1
        paired[level] = dict(cells)
    model = Counter()
    resources = Counter()
    target_failures = Counter()
    stage_statuses = Counter()
    stage_reasons = Counter()
    for case_id in case_ids:
        row = live_rows[case_id]
        for key, value in dict(row.get("model_cost") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                model[key] += value
        for key in ("elapsed_s", "runner_elapsed_s", "attempt_count", "accepted_expansion_count"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                resources[key] += value
        status = str(row.get("status") or "unknown")
        if status not in {"completed", "running", "queued"}:
            target_failures[
                _runner_failure_class(row) if status == "failed" else status
            ] += 1
        for event in row.get("failure_events") or []:
            if isinstance(event, Mapping):
                stage_statuses[str(event.get("status") or "stage_failure")] += 1
                for reason in event.get("reasons") or []:
                    stage_reasons[str(reason)] += 1
    denominator = len(case_ids)
    return {
        "target_count": denominator,
        "live_status_counts": dict(sorted(live_statuses.items())),
        "live_completed_count": len(live_completed),
        "external_completed_count": len(external_completed),
        "live_target_level_counts": live_level_counts,
        "live_target_level_rates_full_denominator": {
            level: (live_level_counts[level] / denominator if denominator else None)
            for level in LEVELS
        },
        "external_target_level_counts": external_level_counts,
        "external_target_level_rates_full_denominator": {
            level: (external_level_counts[level] / denominator if denominator else None)
            for level in LEVELS
        },
        "paired_completed_target_count": len(paired_ids),
        "paired_level_cells": paired,
        "live_model_cost": dict(model),
        "live_resource_totals": dict(resources),
        "live_target_failure_taxonomy": dict(sorted(target_failures.items())),
        "live_stage_event_status_counts": dict(sorted(stage_statuses.items())),
        "live_stage_failure_reason_counts": dict(stage_reasons.most_common()),
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# SynthAtlas 50-target competitive screen",
        "",
        str(result.get("claim_boundary") or ""),
        "",
    ]
    for name, raw in dict(result.get("subsets") or {}).items():
        row = dict(raw or {})
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Targets: {row.get('target_count', 0)}",
                f"- Live statuses: `{json.dumps(row.get('live_status_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
                f"- Paired completed: {row.get('paired_completed_target_count', 0)}",
                f"- Target failures: `{json.dumps(row.get('live_target_failure_taxonomy') or {}, ensure_ascii=False, sort_keys=True)}`",
                "",
                "| Level | SynthAtlas target count | AutoPlanner target count |",
                "| --- | ---: | ---: |",
            ]
        )
        external = dict(row.get("external_target_level_counts") or {})
        live = dict(row.get("live_target_level_counts") or {})
        for level in LEVELS:
            lines.append(f"| {level} | {external.get(level, 0)} | {live.get(level, 0)} |")
        lines.append("")
    lines.append(f"Result digest: `{result.get('content_sha256', '')}`")
    lines.append("")
    return "\n".join(lines)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
