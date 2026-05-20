#!/usr/bin/env python
"""Prune merged ChemEnzy sidecar route pools with source-aware quotas."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_reaction


def prune_route_pool(
    *,
    input_path: Path,
    output_path: Path,
    primary_keep: int = 80,
    sidecar_keep: int = 80,
    max_routes: int = 160,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    out_targets = []
    for row in payload.get("targets") or []:
        pruned = copy.deepcopy(row)
        selected = _select_routes(
            row.get("routes") or [],
            primary_keep=max(0, int(primary_keep)),
            sidecar_keep=max(0, int(sidecar_keep)),
            max_routes=max(0, int(max_routes)),
        )
        pruned["routes"] = selected
        pruned["route_count"] = len(selected)
        pruned["solved"] = bool(selected)
        pruned["raw_backend_metadata"] = {
            **(pruned.get("raw_backend_metadata") or {}),
            "sidecar_pruning": {
                "input": str(input_path),
                "primary_keep": primary_keep,
                "sidecar_keep": sidecar_keep,
                "max_routes": max_routes,
                "input_route_count": len(row.get("routes") or []),
                "output_route_count": len(selected),
                "route_sources": _count(_route_source(route) for route in selected),
            },
        }
        out_targets.append(pruned)
    pruned_payload = {
        "metadata": {
            **(payload.get("metadata") or {}),
            "schema_version": "chem_enzy_sidecar_route_pool.pruned.v1",
            "source_route_pool": str(input_path),
            "primary_keep": primary_keep,
            "sidecar_keep": sidecar_keep,
            "max_routes": max_routes,
        },
        "summary": _summary(out_targets),
        "targets": out_targets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pruned_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return pruned_payload


def _select_routes(
    routes: list[dict[str, Any]],
    *,
    primary_keep: int,
    sidecar_keep: int,
    max_routes: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for route in routes:
        by_source.setdefault(_route_source(route), []).append(route)
    source_order = ["primary"] + sorted(source for source in by_source if source != "primary")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for source in source_order:
        quota = primary_keep if source == "primary" else sidecar_keep
        added = 0
        for route in by_source.get(source, []):
            key = _route_key(route)
            if key in seen:
                continue
            seen.add(key)
            selected.append(route)
            added += 1
            if quota and added >= quota:
                break
        if max_routes and len(selected) >= max_routes:
            return selected[:max_routes]
    return selected[:max_routes] if max_routes else selected


def _route_source(route: dict[str, Any]) -> str:
    return str(((route.get("raw_backend_metadata") or {}).get("sidecar_source")) or "primary")


def _route_key(route: dict[str, Any]) -> tuple[str, ...]:
    rxns = [
        canonical_reaction(step.get("rxn_smiles"))
        for step in route.get("steps") or []
        if canonical_reaction(step.get("rxn_smiles"))
    ]
    return tuple(rxns)


def _summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = [int(row.get("route_count") or 0) for row in targets]
    return {
        "n_targets": len(targets),
        "solved": sum(1 for row in targets if row.get("solved")),
        "solved_rate": round(sum(1 for row in targets if row.get("solved")) / max(len(targets), 1), 6),
        "total_routes": sum(route_counts),
        "avg_route_count": round(sum(route_counts) / max(len(targets), 1), 6),
    }


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--primary-keep", type=int, default=80)
    ap.add_argument("--sidecar-keep", type=int, default=80)
    ap.add_argument("--max-routes", type=int, default=160)
    args = ap.parse_args()
    payload = prune_route_pool(
        input_path=Path(args.input),
        output_path=Path(args.output),
        primary_keep=args.primary_keep,
        sidecar_keep=args.sidecar_keep,
        max_routes=args.max_routes,
    )
    print(json.dumps({"summary": payload["summary"], "output": args.output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
