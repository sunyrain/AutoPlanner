"""Render a completed blackboard run as a read-only explored route forest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.route_forest import write_route_forest_artifacts  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render explored_route_forest.json and route_forest.html from an agent_blackboard.json run."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing agent_blackboard.json.")
    parser.add_argument("--blackboard", type=Path, help="Blackboard JSON path. Defaults to RUN_DIR/agent_blackboard.json.")
    parser.add_argument("--forest-output", type=Path, help="Output JSON path. Defaults to RUN_DIR/explored_route_forest.json.")
    parser.add_argument("--html-output", type=Path, help="Output HTML path. Defaults to RUN_DIR/route_forest.html.")
    parser.add_argument("--max-visual-branches", type=int, default=8)
    parser.add_argument("--max-proposal-branches", type=int, default=10)
    parser.add_argument("--max-template-branches", type=int, default=8)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    blackboard_path = (args.blackboard or run_dir / "agent_blackboard.json").resolve()
    forest_output = (args.forest_output or run_dir / "explored_route_forest.json").resolve()
    html_output = (args.html_output or run_dir / "route_forest.html").resolve()

    blackboard = _load_json(blackboard_path)
    result = write_route_forest_artifacts(
        blackboard,
        run_dir=run_dir,
        forest_output=forest_output,
        html_output=html_output,
        max_visual_branches=args.max_visual_branches,
        max_proposal_branches=args.max_proposal_branches,
        max_template_branches=args.max_template_branches,
    )

    counts = result.get("counts") or {}
    print(f"wrote {result['forest_path']}")
    print(f"wrote {result['html_path']}")
    print(
        "branches={branches} steps={steps} nodes={nodes}".format(
            branches=counts.get("branches", 0),
            steps=counts.get("steps", 0),
            nodes=counts.get("nodes", 0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
