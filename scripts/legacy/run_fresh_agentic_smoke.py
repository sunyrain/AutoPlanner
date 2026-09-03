"""Run or validate a fresh lightweight agentic blackboard smoke.

The default target is aspirin because it exercises the real guided ChemEnzy
path, deterministic parent-route proof, route-forest rendering, and final
verdict closeout in under a minute on the usual local setup.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.legacy.harness_runtime.agentic_blackboard_controller import run_agentic_blackboard_controller  # noqa: E402
from cascade_planner.legacy.harness_runtime.tools import HarnessBudget  # noqa: E402
from scripts.legacy.validate_example_runs import ExampleExpectation, validate_example  # noqa: E402


ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def run_fresh_aspirin_smoke(
    *,
    output_dir: str | Path | None = None,
    timeout_s: float = 900.0,
    guided_chemenzy_timeout_s: float = 240.0,
    validate_only: bool = False,
) -> dict[str, Any]:
    run_dir = Path(output_dir).resolve() if output_dir else _default_output_dir()
    if not validate_only:
        run_agentic_blackboard_controller(
            target_name="aspirin",
            target_smiles=ASPIRIN_SMILES,
            family_hint="fresh deterministic aspirin smoke: guided ChemEnzy direct parent route",
            output_dir=run_dir,
            max_rounds=2,
            exhaust_round_budget=True,
            use_codex_action_planner=False,
            budget=HarnessBudget(
                max_guided_chemenzy_runs=1,
                max_route_expansion_subgoal_runs=0,
                max_codex_research_runs=0,
                max_scout_calls=0,
                max_visual_calls=0,
                max_template_application_actions=0,
                timeout_s=float(timeout_s),
                guided_chemenzy_timeout_s=float(guided_chemenzy_timeout_s),
            ),
        )

    validation = _validate_fresh_aspirin_run(run_dir)
    summary = {
        "schema_version": "fresh_agentic_smoke_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "accepted": bool(validation.get("accepted")),
        "validate_only": bool(validate_only),
        "run_dir": str(run_dir),
        "target": "aspirin",
        "validation": validation,
        "action_types": _action_types(run_dir),
        "final_verdict": _final_verdict_summary(run_dir),
        "artifacts": {
            "summary": str(run_dir / "fresh_agentic_smoke_summary.json"),
            "route_forest_html": str(run_dir / "route_forest.html"),
            "explored_route_forest": str(run_dir / "explored_route_forest.json"),
            "final_verdict": str(run_dir / "final_verdict.json"),
        },
    }
    _write_json(run_dir / "fresh_agentic_smoke_summary.json", summary)
    return summary


def _validate_fresh_aspirin_run(run_dir: Path) -> dict[str, Any]:
    expectation = ExampleExpectation(
        label="fresh_agentic_aspirin_smoke",
        run_dir=run_dir,
        min_branches=1,
        min_steps=1,
        required_branch_kinds=("direct_verified_route",),
        required_text=("Direct verified route", "aspirin"),
        expected_verdict="solved",
        expected_route_status=("solved",),
        expected_solved=True,
        notes=("Fresh smoke must prove the direct parent route, not only render an advisory hypothesis.",),
    )
    return validate_example(expectation)


def _action_types(run_dir: Path) -> list[str]:
    rows: list[tuple[int, list[str]]] = []
    for path in run_dir.glob("action_batch_round_*.json"):
        try:
            round_index = int(path.stem.replace("action_batch_round_", ""))
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        rows.append((round_index, [str(action.get("action_type") or "") for action in data.get("actions") or []]))
    out: list[str] = []
    for _, action_types in sorted(rows, key=lambda item: item[0]):
        out.extend(action_types)
    return out


def _final_verdict_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "final_verdict.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "verdict": data.get("verdict"),
        "route_status": data.get("route_status"),
        "solved": data.get("solved"),
        "reasons": data.get("reasons") or [],
    }


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "shared" / f"fresh_agentic_smoke_aspirin_{stamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output run directory. Defaults to timestamped results/shared path.")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--guided-chemenzy-timeout-s", type=float, default=240.0)
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing output dir without rerunning tools.")
    args = parser.parse_args(argv)

    summary = run_fresh_aspirin_smoke(
        output_dir=args.output_dir,
        timeout_s=float(args.timeout_s),
        guided_chemenzy_timeout_s=float(args.guided_chemenzy_timeout_s),
        validate_only=bool(args.validate_only),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
