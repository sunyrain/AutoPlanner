"""Replay a model-free case after revoking one canonical source fact."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.fact_lifecycle import (  # noqa: E402
    build_fact_lifecycle_event,
)
from cascade_planner.interfaces.replay_pack import (  # noqa: E402
    load_replay_pack,
    run_replay_pack,
    with_replay_pack_digest,
)
from cascade_planner.orchestration.retrosynthesis_service import (  # noqa: E402
    RetrosynthesisCampaignService,
)
from cascade_planner.runtime.paths import RuntimePaths  # noqa: E402


def main() -> int:
    args = _parser().parse_args()
    paths = RuntimePaths.discover()
    base_pack = load_replay_pack(Path(args.pack))
    probe = run_replay_pack(
        base_pack,
        paths=paths,
        run_id=f"{args.run_id}-binding-probe",
        stop_after="evidence",
    )
    service = RetrosynthesisCampaignService.open(
        paths.runtime_root,
        probe["run_dir"],
        artifact_store_root=paths.artifact_store_root,
        run_index_path=paths.run_index_path,
    )
    source_id, source = _select_source(
        service.graph_store.load(),
        source_kind=args.source_kind,
        source_ref=args.source_ref,
    )
    event = build_fact_lifecycle_event(
        subject_kind="source_binding",
        subject_id=source_id,
        subject_content_sha256=str(source["content_sha256"]),
        action="revoke",
        effective_at=args.effective_at,
        reason_codes=[args.reason_code],
    )
    lifecycle_pack = deepcopy(base_pack)
    lifecycle_pack["fact_lifecycle_events"] = [event]
    lifecycle_pack["expected"] = {
        "accepted": False,
        "fact_lifecycle_event_count": 1,
        "inactive_fact_count": 1,
        "revoked_fact_count": 1,
        "expired_fact_count": 0,
        "model_invocations": 0,
        "visual_invocations": 0,
    }
    lifecycle_pack = with_replay_pack_digest(lifecycle_pack)
    result = run_replay_pack(
        lifecycle_pack,
        paths=paths,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "schema_version": "retrosynthesis_lifecycle_showcase_result.v1",
                "source_binding_id": source_id,
                "source_ref": str(source.get("source_ref") or ""),
                "lifecycle_event": event,
                "replay": result,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("accepted") is True else 1


def _select_source(
    graph: dict[str, Any], *, source_kind: str, source_ref: str
) -> tuple[str, dict[str, Any]]:
    matches = [
        (str(source_id), dict(source))
        for source_id, source in dict(graph.get("source_bindings") or {}).items()
        if source.get("source_kind") == source_kind
        and (not source_ref or source.get("source_ref") == source_ref)
    ]
    if len(matches) != 1:
        raise ValueError(f"source selector must match exactly once; matched={len(matches)}")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        default="config/examples/nirmatrelvir_v4_replay_pack.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-kind", default="patent")
    parser.add_argument("--source-ref", default="")
    parser.add_argument("--effective-at", default="2026-07-15T12:00:00Z")
    parser.add_argument("--reason-code", default="showcase_source_retraction")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
