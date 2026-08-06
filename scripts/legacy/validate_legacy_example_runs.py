"""Validate older non-blackboard AutoPlanner example artifacts offline.

The newer agentic blackboard runs are covered by scripts/legacy/validate_example_runs.py and
route_forest smoke tests. Some early statin/Codex-entry examples predate
agent_blackboard.json, so this script checks their saved JSON contracts
directly instead of forcing them through the route-forest projection.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class JsonCheck:
    file: str
    path: str
    equals: Any = None
    min_value: float | None = None


@dataclass(frozen=True)
class LegacyExpectation:
    label: str
    run_dir: Path
    required_json_files: tuple[str, ...]
    checks: tuple[JsonCheck, ...] = ()
    required_text: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_EXAMPLES: tuple[LegacyExpectation, ...] = (
    LegacyExpectation(
        label="fluvastatin_codex_entry_fullflow_20260605",
        run_dir=Path("results/shared/fluvastatin_codex_entry_fullflow_20260605"),
        required_json_files=(
            "target_input.json",
            "preflight.json",
            "artifact_bundle.json",
            "smiles_first_workflow_result.json",
            "open_structure_research_result.json",
            "final_verdict.json",
        ),
        checks=(
            JsonCheck("artifact_bundle.json", "schema_version", equals="codex_entry_artifact_bundle.v1"),
            JsonCheck("final_verdict.json", "schema_version", equals="codex_entry_final_verdict.v1"),
            JsonCheck("final_verdict.json", "verdict", equals="partial_anchor_only_not_solved"),
            JsonCheck("final_verdict.json", "route_status", equals="unresolved"),
            JsonCheck("final_verdict.json", "solved", equals=False),
            JsonCheck("artifact_bundle.json", "artifacts.chemenzy.n_results", min_value=1),
        ),
        required_text=("fluvastatin", "literature_anchor_without_executable_stock_closure"),
        notes=("Legacy Codex-entry statin run should remain readable and truthfully not solved.",),
    ),
    LegacyExpectation(
        label="statin_panel_agent_access_fullflow_20260607",
        run_dir=Path("results/shared/statin_panel_agent_access_fullflow_20260607"),
        required_json_files=(
            "statin_panel_fullflow_overview.json",
            "statin_route_closure_matrix.json",
            "statin_panel_literature_self_evo_report.json",
            "statin_closure_curation_result_set.json",
            "fluvastatin/validation.json",
            "fluvastatin/fluvastatin_statin_panel_hybrid_retrosynthesis_route.json",
            "rosuvastatin/validation.json",
            "rosuvastatin/rosuvastatin_statin_panel_hybrid_retrosynthesis_route.json",
        ),
        checks=(
            JsonCheck("statin_panel_fullflow_overview.json", "schema_version", equals="statin_panel_fullflow_overview.v1"),
            JsonCheck("statin_panel_fullflow_overview.json", "failed", equals=0),
            JsonCheck("statin_panel_fullflow_overview.json", "passed", min_value=9),
            JsonCheck("statin_panel_fullflow_overview.json", "target_count", equals=9),
            JsonCheck("statin_panel_fullflow_overview.json", "validation.accepted", equals=True),
            JsonCheck("statin_route_closure_matrix.json", "schema_version", equals="statin_route_closure_matrix.v1"),
            JsonCheck("statin_route_closure_matrix.json", "target_count", equals=9),
            JsonCheck("statin_route_closure_matrix.json", "full_trace_coverage", equals=True),
            JsonCheck("statin_route_closure_matrix.json", "full_execution_coverage", equals=True),
            JsonCheck("statin_route_closure_matrix.json", "blocker_count", min_value=1),
            JsonCheck("fluvastatin/validation.json", "accepted", equals=True),
            JsonCheck("rosuvastatin/validation.json", "accepted", equals=True),
        ),
        required_text=("atorvastatin", "fluvastatin", "rosuvastatin", "blocker traceability"),
        notes=("Nine-statin panel is a legacy dossier/matrix workflow, not an agent_blackboard route forest.",),
    ),
    LegacyExpectation(
        label="rosuvastatin_real_retrieval_latest_probe_20260606",
        run_dir=Path("results/shared/rosuvastatin_real_retrieval_latest_probe_20260606"),
        required_json_files=(
            "summary.json",
            "compiled/compiled_downstream_consumables.json",
            "compiled/compiled_guided_chemenzy_requests.json",
            "compiled/compiled_literature_template_plugin.json",
            "compiled/compiled_route_expansion_tasks.json",
        ),
        checks=(
            JsonCheck("summary.json", "schema_version", equals="real_retrieval_latest_probe_summary.v1"),
            JsonCheck("summary.json", "case_id", equals="rosuvastatin"),
            JsonCheck("summary.json", "compiled_summary.accepted", equals=True),
            JsonCheck("summary.json", "compiled_summary.template_card_count", min_value=1),
            JsonCheck("summary.json", "compiled_summary.guided_policy_count", min_value=1),
            JsonCheck("summary.json", "compiled_summary.route_expansion_task_count", min_value=1),
            JsonCheck("summary.json", "queue_count", min_value=1),
        ),
        required_text=("rosuvastatin", "compiled_guided_chemenzy_requests", "compiled_route_expansion_tasks"),
        notes=("Legacy retrieval/latest probe should preserve downstream consumables for later agentic runs.",),
    ),
)


def validate_legacy_examples(
    examples: tuple[LegacyExpectation, ...] = DEFAULT_EXAMPLES,
    *,
    skip_missing: bool = False,
) -> dict[str, Any]:
    rows = [validate_legacy_example(example, skip_missing=skip_missing) for example in examples]
    accepted = all(row.get("accepted") or row.get("skipped") for row in rows)
    return {
        "schema_version": "legacy_example_run_validation_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "accepted": accepted,
        "checked": sum(1 for row in rows if not row.get("skipped")),
        "skipped": sum(1 for row in rows if row.get("skipped")),
        "rows": rows,
    }


def validate_legacy_example(example: LegacyExpectation, *, skip_missing: bool = False) -> dict[str, Any]:
    run_dir = (ROOT / example.run_dir).resolve() if not example.run_dir.is_absolute() else example.run_dir.resolve()
    row: dict[str, Any] = {
        "schema_version": "legacy_example_run_validation_row.v1",
        "label": example.label,
        "run_dir": str(run_dir),
        "accepted": False,
        "skipped": False,
        "reasons": [],
        "notes": list(example.notes),
    }
    if not run_dir.exists():
        reason = f"run_dir_missing:{run_dir}"
        if skip_missing:
            row["skipped"] = True
            row["reasons"] = [reason]
            return row
        row["reasons"] = [reason]
        return row

    json_cache: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for relative in example.required_json_files:
        path = run_dir / relative
        if not path.exists():
            reasons.append(f"required_json_missing:{relative}")
            continue
        try:
            json_cache[relative] = _read_json(path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            reasons.append(f"required_json_unreadable:{relative}:{exc}")

    for check in example.checks:
        data = json_cache.get(check.file)
        if data is None:
            if not any(reason.startswith(f"required_json_missing:{check.file}") for reason in reasons):
                reasons.append(f"check_file_missing:{check.file}")
            continue
        value, found = _get_path(data, check.path)
        if not found:
            reasons.append(f"json_path_missing:{check.file}:{check.path}")
            continue
        if check.min_value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                reasons.append(f"json_path_not_numeric:{check.file}:{check.path}:{value!r}")
            else:
                if numeric < float(check.min_value):
                    reasons.append(f"json_path_below_min:{check.file}:{check.path}:{numeric}<{check.min_value}")
        if check.equals is not None and value != check.equals:
            reasons.append(f"json_path_mismatch:{check.file}:{check.path}:{value!r}!={check.equals!r}")

    haystack = "\n".join(json.dumps(data, ensure_ascii=False, sort_keys=True) for data in json_cache.values())
    for text in example.required_text:
        if text not in haystack:
            reasons.append(f"missing_required_text:{text}")

    row.update(
        {
            "accepted": not reasons,
            "reasons": reasons,
            "checked_json_files": sorted(json_cache),
        }
    )
    return row


def _get_path(data: dict[str, Any], path: str) -> tuple[Any, bool]:
    current: Any = data
    for token in str(path or "").split("."):
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        return None, False
    return current, True


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing default legacy run directories.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args(argv)

    summary = validate_legacy_examples(skip_missing=bool(args.skip_missing))
    if args.summary_output:
        out = args.summary_output.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
