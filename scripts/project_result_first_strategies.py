from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.result_first_strategy_projection import (
    project_launch_strategies,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel_summary.read_text(encoding="utf-8"))
    rows = [
        _hydrate_target(row)
        for row in panel.get("per_target") or []
        if isinstance(row, Mapping)
    ]
    projection = project_launch_strategies(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(projection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _hydrate_target(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    report_path = Path(str(row.get("report_path") or ""))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row["start_cohort_latency_audit"] = _start_cohort_latency(report)
    b4 = dict(
        dict(dict(report.get("trajectory") or {}).get("time_to_first") or {}).get(
            "B4"
        )
        or {}
    )
    row["b4_phase"] = str(b4.get("phase") or "")
    return row


def _start_cohort_latency(report: Mapping[str, Any]) -> dict[str, Any]:
    for stage in report.get("stages") or []:
        if not isinstance(stage, Mapping):
            continue
        audit = dict(
            dict(dict(stage.get("detail") or {}).get("start_cohort") or {}).get(
                "latency_audit"
            )
            or {}
        )
        if audit.get("schema_version") == "campaign_action_cohort_latency_audit.v1":
            return audit
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
