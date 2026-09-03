"""Project any digest-bound V4 Workbench into non-authoritative Programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.candidate_programs import (
    candidate_program_projection_oracle,
    candidate_route_observation_from_workbench,
    project_candidate_route_to_programs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbench")
    parser.add_argument("--observation-output")
    parser.add_argument("--projection-output")
    args = parser.parse_args()
    source = json.loads(Path(args.workbench).expanduser().resolve().read_text(encoding="utf-8"))
    observation = candidate_route_observation_from_workbench(source)
    projection = project_candidate_route_to_programs(observation)
    oracle = candidate_program_projection_oracle(observation, projection)
    if oracle["accepted"] is not True:
        raise SystemExit("candidate_program_projection_oracle_failed")
    _write(args.observation_output, observation)
    _write(args.projection_output, projection)
    print(
        json.dumps(
            {
                "schema_version": "candidate_program_projection_result.v1",
                "observation_sha256": observation["content_sha256"],
                "projection_sha256": projection["content_sha256"],
                "counts": projection["counts"],
                "oracle_accepted": True,
                "semantics": {
                    "read_only": True,
                    "canonical_graph_not_modified": True,
                    "program_store_admission_performed": False,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
