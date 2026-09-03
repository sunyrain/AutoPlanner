#!/usr/bin/env python3
"""Summarize the frozen external snapshot and three clean live arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.strategy_closure_live import (  # noqa: E402
    summarize_strategy_closure,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-receipt",
        default=("results/shared/synthatlas_strategy_closure_clean20_live/execution-receipt.json"),
    )
    parser.add_argument(
        "--live-root",
        default="results/shared/synthatlas_strategy_closure_clean20_live",
    )
    parser.add_argument(
        "--external-root",
        default=("results/shared/synthatlas_strategy_closure_clean20_external_snapshot"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default=("results/shared/synthatlas_strategy_closure_clean20_live/paired-summary.json"),
    )
    args = parser.parse_args(argv)

    receipt = _json(_path(args.execution_receipt))
    live_root = _path(args.live_root)
    statuses = {
        str(row["arm_id"]): _json(live_root / str(row["arm_id"]) / "panel-status.json")
        for row in receipt.get("arms") or []
    }
    external_root = _path(args.external_root)
    external_summary = _json(external_root / "summary.json")
    external_cases = [_json(path) for path in sorted((external_root / "cases").glob("*.json"))]
    summary = summarize_strategy_closure(
        execution=receipt,
        live_panel_statuses=statuses,
        external_summary=external_summary,
        external_cases=external_cases,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _markdown(summary: dict) -> str:
    lines = [
        "# Strategy-to-Experiment Closure clean-20",
        "",
        f"- Targets: {summary.get('target_count', 0)}",
        f"- Result digest: `{summary.get('content_sha256', '')}`",
        "",
        "| Arm | C0 | C1 | C2 | C3 | C4 | C5 | C6 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id, raw in dict(summary.get("arms") or {}).items():
        counts = dict(raw.get("level_counts") or {})
        lines.append(
            f"| {arm_id} | "
            + " | ".join(str(counts.get(f"C{index}", 0)) for index in range(7))
            + " |"
        )
    return "\n".join(lines) + "\n"


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
