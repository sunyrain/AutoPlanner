#!/usr/bin/env python
"""Merge primary ChemEnzy routes with sidecar proposal/checkpoint route pools."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def merge_route_pools(*, primary: Path, sidecars: dict[str, Path], output: Path) -> dict[str, Any]:
    primary_payload = json.loads(primary.read_text(encoding="utf-8"))
    sidecar_payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in sidecars.items()
    }
    sidecar_by_target = {
        name: {str(row.get("target_smiles") or ""): row for row in payload.get("targets") or []}
        for name, payload in sidecar_payloads.items()
    }
    merged_targets = []
    for primary_row in primary_payload.get("targets") or []:
        target = str(primary_row.get("target_smiles") or "")
        merged = copy.deepcopy(primary_row)
        routes = list(merged.get("routes") or [])
        route_sources = ["primary"] * len(routes)
        merged_failures = list(merged.get("failures") or [])
        for name, rows_by_target in sidecar_by_target.items():
            side_row = rows_by_target.get(target)
            if not side_row:
                continue
            for route in side_row.get("routes") or []:
                side_route = copy.deepcopy(route)
                side_route.setdefault("raw_backend_metadata", {})
                side_route["raw_backend_metadata"]["sidecar_source"] = name
                routes.append(side_route)
                route_sources.append(name)
            for failure in side_row.get("failures") or []:
                side_failure = copy.deepcopy(failure)
                side_failure.setdefault("raw_backend_metadata", {})
                side_failure["raw_backend_metadata"]["sidecar_source"] = name
                merged_failures.append(side_failure)
        merged["routes"] = routes
        merged["route_count"] = len(routes)
        merged["solved"] = bool(routes)
        merged["failures"] = merged_failures
        merged["raw_backend_metadata"] = {
            **(merged.get("raw_backend_metadata") or {}),
            "sidecar_merge": {
                "primary": str(primary),
                "sidecars": {name: str(path) for name, path in sidecars.items()},
                "route_sources": _count(route_sources),
            },
        }
        merged_targets.append(merged)
    payload = {
        "metadata": {
            "schema_version": "chem_enzy_sidecar_route_pool.v1",
            "primary": str(primary),
            "sidecars": {name: str(path) for name, path in sidecars.items()},
        },
        "summary": _summary(merged_targets),
        "targets": merged_targets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = [int(row.get("route_count") or 0) for row in targets]
    return {
        "n_targets": len(targets),
        "solved": sum(1 for row in targets if row.get("solved")),
        "solved_rate": round(sum(1 for row in targets if row.get("solved")) / max(len(targets), 1), 6),
        "total_routes": sum(route_counts),
        "avg_route_count": round(sum(route_counts) / max(len(targets), 1), 6),
    }


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _parse_sidecar(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--sidecar must be NAME=PATH")
    name, path = raw.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--sidecar", action="append", required=True, help="NAME=route_pool.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    payload = merge_route_pools(
        primary=Path(args.primary),
        sidecars=dict(_parse_sidecar(item) for item in args.sidecar),
        output=Path(args.output),
    )
    print(json.dumps({"summary": payload["summary"], "output": args.output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
