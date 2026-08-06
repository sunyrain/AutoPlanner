"""Validate historical AutoPlanner example runs offline.

This is a lightweight regression gate for previously debugged examples. It
does not run new agent workers or download papers; it reloads saved
agent_blackboard.json files, rerenders route_forest artifacts, and checks that
the user-facing projection still contains the expected route or diagnostic.
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

from cascade_planner.legacy.harness_runtime.route_forest import write_route_forest_artifacts  # noqa: E402
from scripts.legacy.smoke_route_forest_history import route_forest_html_contract_reasons  # noqa: E402


@dataclass(frozen=True)
class ExampleExpectation:
    label: str
    run_dir: Path
    min_branches: int = 1
    min_steps: int = 1
    required_branch_kinds: tuple[str, ...] = ()
    required_text: tuple[str, ...] = ()
    expected_verdict: str = ""
    expected_route_status: tuple[str, ...] = ()
    expected_solved: bool | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_EXAMPLES: tuple[ExampleExpectation, ...] = (
    ExampleExpectation(
        label="aspirin_failed_runtime_diagnostic_20260701",
        run_dir=Path("results/shared/aspirin_fullflow_blackboard_fulltime_20260701"),
        required_branch_kinds=("diagnostic_failure",),
        required_text=("chemenzy_missing_output", "aspirin"),
        notes=("Old run records the historical ChemEnzy runtime failure instead of a blank route.",),
    ),
    ExampleExpectation(
        label="aspirin_fresh_direct_verified_20260705",
        run_dir=Path("results/shared/fresh_smoke_aspirin_20260705"),
        required_branch_kinds=("direct_verified_route",),
        required_text=("Direct verified route", "aspirin"),
        expected_verdict="solved",
        expected_route_status=("solved",),
        expected_solved=True,
        notes=("Fresh smoke run should still render the simple direct ChemEnzy route.",),
    ),
    ExampleExpectation(
        label="ibuprofen_direct_verified_20260701",
        run_dir=Path("results/shared/ibuprofen_fullflow_blackboard_online_20260701_clean_20260701_184526"),
        min_steps=3,
        required_branch_kinds=("direct_verified_route",),
        required_text=("ibuprofen", "Direct verified route"),
        expected_verdict="solved",
        expected_route_status=("solved",),
        expected_solved=True,
        notes=("Solved direct parent-route proof should render as a ChemEnzy route skeleton.",),
    ),
    ExampleExpectation(
        label="paclitaxel_recommended_semisynthesis_20260703",
        run_dir=Path("results/shared/full_rerun_advisory_template_clean3_20260703/paclitaxel"),
        min_steps=4,
        required_branch_kinds=("recommended_strategy",),
        required_text=("10-deacetylbaccatin III", "C13 beta-lactam side-chain coupling", "paclitaxel"),
    ),
    ExampleExpectation(
        label="paclitaxel_lit_panel_20260702",
        run_dir=Path("results/shared/large_lit_pdf_panel_20260702/paclitaxel"),
        min_steps=4,
        required_branch_kinds=("recommended_strategy",),
        required_text=("10-deacetylbaccatin III", "paclitaxel"),
    ),
    ExampleExpectation(
        label="bufotalin_exact_stitch_20260622",
        run_dir=Path("results/shared/bufotalin_full_exact_stitch_rerun_20260622_073847"),
        min_steps=10,
        required_branch_kinds=("exact_literature", "visual_chain"),
        required_text=("bufotalin", "source detail exact step", "exact_literature"),
        expected_verdict="solved",
        expected_route_status=("solved",),
        expected_solved=True,
        notes=("Historical bufotalin run should expose exact literature rows instead of only raw blackboard JSON.",),
    ),
    ExampleExpectation(
        label="erythromycin_advisory_visual_20260702",
        run_dir=Path("results/shared/full_rerun_advisory_visual_20260702/erythromycin_a"),
        min_steps=10,
        required_branch_kinds=("visual_chain", "retrosynthetic_proposal"),
        required_text=("erythromycin", "doi:10.1021/ja00401a051", "visual"),
        expected_verdict="hypothesis_route_proposed",
        expected_route_status=("hypothesis_routes_executed_rejected_recursive_followup_pending",),
        expected_solved=False,
        notes=("Advisory visual chain and failed/alternative proposals should remain inspectable.",),
    ),
    ExampleExpectation(
        label="artemisinin_advisory_visual_20260702",
        run_dir=Path("results/shared/full_rerun_advisory_visual_20260702/artemisinin"),
        min_steps=10,
        required_branch_kinds=("visual_chain", "retrosynthetic_proposal"),
        required_text=("artemisinin", "Artemisinic acid", "Dihydroartemisinic acid"),
        expected_verdict="hypothesis_route_proposed",
        expected_route_status=("hypothesis_routes_pending_execution",),
        expected_solved=False,
    ),
    ExampleExpectation(
        label="erythromycin_high_budget_repaired_20260702",
        run_dir=Path("results/shared/high_budget_browser_pdf_cache_20260702/erythromycin_retry_repaired3_20260702"),
        min_steps=4,
        required_branch_kinds=("visual_chain", "retrosynthetic_proposal"),
        required_text=("erythromycin", "10c", "final hydrolysis"),
        notes=("High-budget repaired run should preserve source-derived visual steps and proposals.",),
    ),
    ExampleExpectation(
        label="target1_process_evidence_20260624",
        run_dir=Path("results/shared/target1_process_evidence_clean_final_target1_9_21_dihydroxy_20_methylpregn_4_en_3_one_20260624_162137"),
        min_steps=4,
        required_branch_kinds=("visual_chain", "process_evidence", "broad_template"),
        required_text=("9-OH-4-HP", "phytosterols", "process evidence"),
        expected_verdict="fake_closed_rejected",
        expected_route_status=("fake_closed_rejected",),
        expected_solved=False,
        notes=("Steroid/biotransformation target should expose process evidence rows as advisory anchors.",),
    ),
    ExampleExpectation(
        label="atorvastatin_online_zero_20260704",
        run_dir=Path("results/shared/atorvastatin_online_zero_20260704_202940"),
        min_steps=5,
        required_branch_kinds=("recommended_strategy", "process_evidence"),
        required_text=("Paal-Knorr", "advanced ketal ester intermediate 4", "10.1186/s13065-015-0082-7"),
        expected_verdict="hypothesis_route_proposed",
        expected_route_status=("hypothesis_route_execution_partial",),
        expected_solved=False,
        notes=("Late browser-downloaded BMC/Springer PDF must be visible to the route projection.",),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="Additional run directory to validate.")
    parser.add_argument("--no-render", action="store_true", help="Validate existing explored_route_forest.json without rerendering.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing default run directories.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args(argv)

    examples = list(DEFAULT_EXAMPLES)
    for index, raw in enumerate(args.run, start=1):
        examples.append(
            ExampleExpectation(
                label=f"custom_{index}:{raw}",
                run_dir=Path(raw),
                min_branches=1,
                min_steps=1,
            )
        )

    rows = [
        validate_example(
            example,
            render=not bool(args.no_render),
            skip_missing=bool(args.skip_missing),
        )
        for example in examples
    ]
    accepted = all(row.get("accepted") or row.get("skipped") for row in rows)
    summary = {
        "schema_version": "example_run_validation_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "accepted": accepted,
        "checked": sum(1 for row in rows if not row.get("skipped")),
        "skipped": sum(1 for row in rows if row.get("skipped")),
        "rows": rows,
    }
    if args.summary_output:
        out = args.summary_output.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if accepted else 1


def validate_example(
    example: ExampleExpectation,
    *,
    render: bool = True,
    skip_missing: bool = False,
) -> dict[str, Any]:
    run_dir = (ROOT / example.run_dir).resolve() if not example.run_dir.is_absolute() else example.run_dir.resolve()
    row: dict[str, Any] = {
        "schema_version": "example_run_validation_row.v1",
        "label": example.label,
        "run_dir": str(run_dir),
        "accepted": False,
        "skipped": False,
        "reasons": [],
        "notes": list(example.notes),
    }
    blackboard_path = run_dir / "agent_blackboard.json"
    if not blackboard_path.exists():
        reason = f"agent_blackboard_missing:{blackboard_path}"
        if skip_missing:
            row["skipped"] = True
            row["reasons"] = [reason]
            return row
        row["reasons"] = [reason]
        return row

    try:
        blackboard = _load_json(blackboard_path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        row["reasons"] = [f"agent_blackboard_unreadable:{exc}"]
        return row

    try:
        if render:
            rendered = write_route_forest_artifacts(blackboard, run_dir=run_dir)
            forest = dict(rendered.get("forest") or {})
        else:
            forest = _load_json(run_dir / "explored_route_forest.json")
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        row["reasons"] = [f"route_forest_render_failed:{exc}"]
        return row

    reasons = _forest_validation_reasons(forest, example=example)
    final_verdict_path = run_dir / "final_verdict.json"
    final_verdict: dict[str, Any] = {}
    if final_verdict_path.exists():
        try:
            final_verdict = _load_json(final_verdict_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            reasons.append(f"final_verdict_unreadable:{exc}")
    reasons.extend(_final_verdict_validation_reasons(final_verdict, final_verdict_path=final_verdict_path, example=example))
    route_forest_path = run_dir / "route_forest.html"
    explored_path = run_dir / "explored_route_forest.json"
    reasons.extend(_html_validation_reasons(route_forest_path, expected_forest=forest))
    row.update(
        {
            "accepted": not reasons,
            "reasons": reasons,
            "counts": dict(forest.get("counts") or {}),
            "target": dict(forest.get("target") or {}),
            "branch_kinds": sorted({str(branch.get("kind") or "") for branch in forest.get("branches") or [] if isinstance(branch, dict)}),
            "route_forest_html": str(route_forest_path),
            "explored_route_forest": str(explored_path),
            "final_verdict": _final_verdict_summary(final_verdict) if final_verdict else {},
        }
    )
    return row


def _forest_validation_reasons(forest: dict[str, Any], *, example: ExampleExpectation) -> list[str]:
    reasons: list[str] = []
    if forest.get("schema_version") != "explored_route_forest.v1":
        reasons.append("invalid_or_missing_forest_schema")
    counts = dict(forest.get("counts") or {})
    branches = [dict(row) for row in forest.get("branches") or [] if isinstance(row, dict)]
    steps = [dict(row) for row in forest.get("steps") or [] if isinstance(row, dict)]
    if int(counts.get("branches") or len(branches)) < int(example.min_branches):
        reasons.append(f"branch_count_below_min:{counts.get('branches') or len(branches)}<{example.min_branches}")
    if int(counts.get("steps") or len(steps)) < int(example.min_steps):
        reasons.append(f"step_count_below_min:{counts.get('steps') or len(steps)}<{example.min_steps}")
    branch_kinds = {str(branch.get("kind") or "") for branch in branches}
    for kind in example.required_branch_kinds:
        if kind not in branch_kinds:
            reasons.append(f"missing_branch_kind:{kind}")
    haystack = json.dumps(forest, ensure_ascii=False)
    for text in example.required_text:
        if text not in haystack:
            reasons.append(f"missing_required_text:{text}")
    return reasons


def _html_validation_reasons(
    route_forest_path: Path,
    *,
    expected_forest: dict[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if not route_forest_path.exists():
        return [f"route_forest_html_missing:{route_forest_path}"]
    try:
        text = route_forest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"route_forest_html_unreadable:{exc}"]
    if len(text) < 1200:
        reasons.append(f"route_forest_html_too_small:{len(text)}")
    reasons.extend(route_forest_html_contract_reasons(text, expected_forest=expected_forest))
    if ">undefined<" in text or ">NaN<" in text:
        reasons.append("route_forest_html_contains_obvious_js_placeholder")
    return reasons


def _final_verdict_validation_reasons(
    final_verdict: dict[str, Any],
    *,
    final_verdict_path: Path,
    example: ExampleExpectation,
) -> list[str]:
    reasons: list[str] = []
    expected = bool(example.expected_verdict or example.expected_route_status or example.expected_solved is not None)
    if expected and not final_verdict:
        return [f"final_verdict_missing:{final_verdict_path}"]
    if not final_verdict:
        return reasons

    verdict = str(final_verdict.get("verdict") or "")
    route_status = str(final_verdict.get("route_status") or "")
    solved = bool(final_verdict.get("solved"))
    if verdict == "solved" and not solved:
        reasons.append("final_verdict_solved_flag_mismatch")
    if route_status == "solved" and not solved:
        reasons.append("final_route_status_solved_without_solved_flag")
    if solved and verdict != "solved":
        reasons.append("final_solved_true_without_solved_verdict")
    if example.expected_verdict and verdict != example.expected_verdict:
        reasons.append(f"final_verdict_mismatch:{verdict}!={example.expected_verdict}")
    if example.expected_route_status and route_status not in set(example.expected_route_status):
        allowed = "|".join(example.expected_route_status)
        reasons.append(f"final_route_status_mismatch:{route_status}!={allowed}")
    if example.expected_solved is not None and solved is not bool(example.expected_solved):
        reasons.append(f"final_solved_mismatch:{solved}!={bool(example.expected_solved)}")
    return reasons


def _final_verdict_summary(final_verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": str(final_verdict.get("verdict") or ""),
        "route_status": str(final_verdict.get("route_status") or ""),
        "solved": bool(final_verdict.get("solved")),
        "reasons": [str(item) for item in final_verdict.get("reasons") or [] if str(item).strip()][:8],
    }


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object in {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
