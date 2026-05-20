"""CLI for building cascade-native oracle value payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade_planner.route_tree.cascade_oracle import build_cascade_oracle_payload_from_native


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a cascade-native oracle payload from native ChemEnzy routes")
    ap.add_argument("--native-payload", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--selection", default="rank_plus_stock")
    args = ap.parse_args()
    payload = build_cascade_oracle_payload_from_native(
        native_payload_path=Path(args.native_payload),
        output_path=Path(args.output),
        topk=args.topk,
        selection=args.selection,
    )
    print(
        json.dumps(
            {
                "schema_version": payload.get("schema_version"),
                "output": str(args.output),
                "native_payload": str(args.native_payload),
                "topk": int(args.topk),
                "selection": str(args.selection),
                "targets": len(payload.get("targets") or []),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
