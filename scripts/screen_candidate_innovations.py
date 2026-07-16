#!/usr/bin/env python3
"""Screen a digest-bound Candidate Route observation for enzyme opportunities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.candidate_innovation_screen import (  # noqa: E402
    screen_candidate_route_innovations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation")
    parser.add_argument(
        "--capabilities",
        default=str(ROOT / "config" / "route_innovation_capabilities.v1.json"),
    )
    parser.add_argument("--max-window-steps", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    observation = _load(args.observation)
    capabilities = _load(args.capabilities)
    result = screen_candidate_route_innovations(
        observation,
        capabilities=capabilities,
        max_window_steps=args.max_window_steps,
    )
    _write(args.output, result)
    print(
        json.dumps(
            {
                "schema_version": "candidate_route_innovation_screen_result.v1",
                "content_sha256": result["content_sha256"],
                "accepted_capability_count": result["accepted_capability_count"],
                "counts": result["counts"],
                "route_statuses": {
                    route_id: row["screen_status"]
                    for route_id, row in result["route_screens"].items()
                },
                "semantics": result["semantics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load(value: str) -> dict:
    payload = json.loads(Path(value).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate_innovation_json_object_required")
    return payload


def _write(value: str | None, payload: dict) -> None:
    if not value:
        return
    path = Path(value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
