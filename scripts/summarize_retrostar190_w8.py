#!/usr/bin/env python3
"""Compile paired W8 metrics and evidence-backed failure categories."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_v4_blind_panel import summarize_panel


ARMS = (
    "chemenzy-only",
    "codex-only",
    "unified-round-robin",
    "unified-adaptive",
)
BOOTSTRAP_SEED = 20260806
BOOTSTRAP_REPLICATES = 5000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w8-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    root = args.w8_root.expanduser().resolve()
    output = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else root / "summary"
    )
    output.mkdir(parents=True, exist_ok=True)
    statuses = {
        arm: _read_json(root / arm / "panel-status.json") for arm in ARMS
    }
    _validate_contract(statuses)
    summaries = {arm: summarize_panel(status) for arm, status in statuses.items()}
    rows = {
        arm: {
            str(row.get("case_id") or ""): dict(row)
            for row in summary.get("per_target") or []
            if str(row.get("case_id") or "")
        }
        for arm, summary in summaries.items()
    }
    per_target = _per_target(rows)
    paired = _paired(rows)
    taxonomy = _failure_taxonomy(rows)
    run_manifest = _run_manifest(root, statuses, summaries)
    _write_json(output / "w8-run-manifest.json", run_manifest)
    _write_json(output / "w8-per-target-metrics.json", per_target)
    _write_json(output / "w8-paired-comparison.json", paired)
    _write_json(output / "w8-failure-taxonomy.json", taxonomy)
    _write_json(output / "w8-panel-summaries.json", summaries)
    print(
        json.dumps(
            {
                "output_root": str(output),
                "complete_arms": {
                    arm: bool(status.get("complete"))
                    for arm, status in statuses.items()
                },
                "paired_target_count": paired["paired_target_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _validate_contract(statuses: Mapping[str, Mapping[str, Any]]) -> None:
    selections = {
        arm: tuple(dict(status.get("selection") or {}).get("selected_case_ids") or [])
        for arm, status in statuses.items()
    }
    reference = selections["unified-adaptive"]
    if any(value != reference for value in selections.values()):
        raise RuntimeError("w8_arm_target_manifests_do_not_match")
    frozen = {
        arm: dict(status.get("frozen_snapshot") or {})
        for arm, status in statuses.items()
    }
    for key in (
        "manifest_sha256",
        "benchmark_stock_index_sha256",
        "base_environment_sha256",
    ):
        values = {str(row.get(key) or "") for row in frozen.values()}
        if len(values) != 1 or not next(iter(values), ""):
            raise RuntimeError(f"w8_arm_frozen_contract_mismatch:{key}")


def _per_target(
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    case_ids = sorted(set.intersection(*(set(values) for values in rows.values())))
    values = [
        {
            "case_id": case_id,
            "arms": {arm: dict(rows[arm][case_id]) for arm in ARMS},
        }
        for case_id in case_ids
    ]
    return _digest_bound(
        {
            "schema_version": "retrostar190_w8_per_target_metrics.v1",
            "target_count": len(values),
            "targets": values,
        }
    )


def _paired(rows: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    adaptive = rows["unified-adaptive"]
    comparisons: dict[str, Any] = {}
    common_all = {
        case_id
        for case_id in adaptive
        if all(
            case_id in rows[arm]
            and str(rows[arm][case_id].get("status") or "") == "completed"
            for arm in ARMS
        )
    }
    case_ids = sorted(common_all)
    for arm in ARMS[:-1]:
        differences = [
            int(adaptive[case_id].get("retrostar_solved") is True)
            - int(rows[arm][case_id].get("retrostar_solved") is True)
            for case_id in case_ids
        ]
        comparisons[arm] = {
            "adaptive_solved": sum(
                adaptive[case_id].get("retrostar_solved") is True
                for case_id in case_ids
            ),
            "comparator_solved": sum(
                rows[arm][case_id].get("retrostar_solved") is True
                for case_id in case_ids
            ),
            "adaptive_wins": sum(value == 1 for value in differences),
            "adaptive_losses": sum(value == -1 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "paired_success_rate_difference": round(
                sum(differences) / max(1, len(differences)), 6
            ),
            "paired_bootstrap_95_ci": _bootstrap_ci(
                differences,
                seed=BOOTSTRAP_SEED + sum(ord(value) for value in arm),
            ),
            "resource_differences": _paired_resource_differences(
                adaptive,
                rows[arm],
                case_ids,
            ),
            "per_target_difference": [
                {"case_id": case_id, "adaptive_minus_comparator": difference}
                for case_id, difference in zip(case_ids, differences, strict=True)
            ],
        }
    return _digest_bound(
        {
            "schema_version": "retrostar190_w8_paired_comparison.v1",
            "paired_target_count": len(case_ids),
            "reference_arm": "unified-adaptive",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "comparisons": comparisons,
        }
    )


def _failure_taxonomy(
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm, arm_rows in rows.items():
        categories = Counter()
        targets = []
        for case_id, row in sorted(arm_rows.items()):
            category, evidence = _failure_category(row, arm=arm)
            categories[category] += 1
            targets.append(
                {"case_id": case_id, "category": category, "evidence": evidence}
            )
        by_arm[arm] = {
            "counts": dict(sorted(categories.items())),
            "targets": targets,
        }
    return _digest_bound(
        {
            "schema_version": "retrostar190_w8_failure_taxonomy.v1",
            "arms": by_arm,
            "semantics": {
                "categories_are_derived_from_saved_reports": True,
                "uncertain_failures_are_not_forced_into_a_more_specific_class": True,
                "B2_B3_B5_deficits_do_not_redefine_retrostar_B4_failure": True,
            },
        }
    )


def _failure_category(
    row: Mapping[str, Any],
    *,
    arm: str,
) -> tuple[str, dict[str, Any]]:
    status = str(row.get("status") or "")
    error = str(row.get("error") or "")
    counts = dict(row.get("route_counts") or {})
    evidence = {"status": status, "route_counts": counts}
    if status in {"queued", "running", ""}:
        return "not_completed", evidence
    if status != "completed":
        return (
            "budget_or_timeout" if "timeout" in error.lower() else "runtime_failure",
            {**evidence, "error": error[:500]},
        )
    if row.get("retrostar_solved") is True:
        return "solved", evidence
    structural = int(counts.get("target_rooted_distinct_skeletons") or 0)
    materialized = int(counts.get("materialized_skeletons") or 0)
    stock_closed = int(counts.get("stock_closed_skeletons") or 0)
    if stock_closed > 0:
        return "portfolio_or_reporting_omission", evidence
    if structural > 0:
        if materialized <= 0:
            return "canonical_merge_or_materialization_loss", evidence
        return "stock_miss", evidence
    report = _optional_report(str(row.get("report_path") or ""))
    baseline = _last_stage(report, "chemenzy_baseline")
    route_count = int(baseline.get("route_count") or 0)
    host_admitted = int(baseline.get("host_admitted_route_count") or 0)
    evidence.update(
        {
            "chemenzy_route_count": route_count,
            "chemenzy_host_admitted_route_count": host_admitted,
        }
    )
    if arm != "codex-only" and route_count <= 0:
        return "provider_no_candidate", evidence
    if route_count > 0 and host_admitted <= 0:
        return "normalization_or_host_admission_loss", evidence
    if route_count > 0:
        return "canonical_merge_or_search_depth_loss", evidence
    return "unclassified_no_structural_route", evidence


def _run_manifest(
    root: Path,
    statuses: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return _digest_bound(
        {
            "schema_version": "retrostar190_w8_run_manifest.v1",
            "w8_root": str(root),
            "arms": {
                arm: {
                    "panel_status_path": str(root / arm / "panel-status.json"),
                    "panel_status_sha256": _file_sha256(
                        root / arm / "panel-status.json"
                    ),
                    "complete": status.get("complete") is True,
                    "completed_count": int(status.get("completed_count") or 0),
                    "target_count": int(status.get("target_count") or 0),
                    "frozen_snapshot": dict(status.get("frozen_snapshot") or {}),
                    "summary_sha256": str(
                        summaries[arm].get("content_sha256") or ""
                    ),
                }
                for arm, status in statuses.items()
            },
        }
    )


def _bootstrap_ci(values: list[int], *, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [
        round(samples[int(0.025 * (len(samples) - 1))], 6),
        round(samples[int(0.975 * (len(samples) - 1))], 6),
    ]


def _paired_resource_differences(
    adaptive: Mapping[str, Mapping[str, Any]],
    comparator: Mapping[str, Mapping[str, Any]],
    case_ids: list[str],
) -> dict[str, Any]:
    fields = ("model_invocations", "input_tokens", "output_tokens", "wall_time_s")
    result: dict[str, Any] = {}
    for field in fields:
        values = [
            float(dict(adaptive[case_id].get("model_cost") or {}).get(field) or 0.0)
            - float(
                dict(comparator[case_id].get("model_cost") or {}).get(field)
                or 0.0
            )
            for case_id in case_ids
        ]
        result[field] = {
            "adaptive_minus_comparator_sum": round(sum(values), 3),
            "adaptive_minus_comparator_mean": round(
                sum(values) / max(1, len(values)), 6
            ),
        }
    elapsed = [
        float(adaptive[case_id].get("elapsed_s") or 0.0)
        - float(comparator[case_id].get("elapsed_s") or 0.0)
        for case_id in case_ids
    ]
    result["elapsed_s"] = {
        "adaptive_minus_comparator_sum": round(sum(elapsed), 3),
        "adaptive_minus_comparator_mean": round(
            sum(elapsed) / max(1, len(elapsed)), 6
        ),
    }
    return result


def _optional_report(path: str) -> dict[str, Any]:
    candidate = Path(path)
    return _read_json(candidate) if candidate.is_file() else {}


def _last_stage(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    rows = [
        dict(value.get("detail") or {})
        for value in report.get("stages") or []
        if isinstance(value, Mapping) and value.get("stage") == name
    ]
    return rows[-1] if rows else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_bound(value: dict[str, Any]) -> dict[str, Any]:
    value["content_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
