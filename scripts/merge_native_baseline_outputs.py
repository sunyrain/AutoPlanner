"""Merge multiple native ChemEnzy baseline JSON outputs into one batch file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge native ChemEnzy batch outputs.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    targets = []
    metadata = {"merged_inputs": args.input}
    seen = set()
    for raw_path in args.input:
        path = Path(raw_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for target in data.get("targets") or []:
            smiles = str(target.get("target_smiles") or "")
            if smiles in seen:
                continue
            seen.add(smiles)
            targets.append(target)
    payload = {"metadata": metadata, "summary": _summarize(targets), "targets": targets}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": payload["summary"]}, indent=2, ensure_ascii=False))


def _summarize(targets: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(targets)
    solved = sum(1 for target in targets if target.get("solved"))
    route_counts = [int(target.get("route_count") or len(target.get("routes") or [])) for target in targets]
    failures: dict[str, int] = {}
    elapsed = []
    for target in targets:
        for failure in target.get("failures") or []:
            category = str(failure.get("category") or "unknown")
            failures[category] = failures.get(category, 0) + 1
        raw = target.get("raw_backend_metadata") or {}
        if raw.get("elapsed_s") is not None:
            try:
                elapsed.append(float(raw.get("elapsed_s")))
            except (TypeError, ValueError):
                pass
    return {
        "n_targets": n,
        "solved": solved,
        "solved_rate": solved / n if n else None,
        "total_routes": sum(route_counts),
        "avg_route_count": sum(route_counts) / n if n else None,
        "avg_search_time_s": sum(elapsed) / len(elapsed) if elapsed else None,
        "total_search_time_s": sum(elapsed) if elapsed else None,
        "failure_categories": failures,
    }


if __name__ == "__main__":
    main()
