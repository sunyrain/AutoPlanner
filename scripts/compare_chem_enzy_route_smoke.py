#!/usr/bin/env python
"""Compare ChemEnzy route-smoke outputs against embedded GT reactions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_reaction


def compare_runs(runs: dict[str, Path]) -> dict[str, Any]:
    results = {}
    for name, path in runs.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        targets = payload.get("targets") or []
        rows = [_target_recovery(row) for row in targets]
        results[name] = {
            "path": str(path),
            "run_summary": payload.get("summary") or {},
            "recovery_summary": _summarize(rows),
            "targets": rows,
        }
    return {"results": results}


def _target_recovery(row: dict[str, Any]) -> dict[str, Any]:
    gt_rxns = [
        canonical_reaction(step.get("rxn_smiles"))
        for step in row.get("gt_route") or []
        if canonical_reaction(step.get("rxn_smiles"))
    ]
    gt_set = set(gt_rxns)
    per_route = []
    best_hits = 0
    exact_route_rank = None
    any_exact_rank = None
    for route_rank, route in enumerate(row.get("routes") or [], 1):
        pred_rxns = [
            canonical_reaction(step.get("rxn_smiles"))
            for step in route.get("steps") or []
            if canonical_reaction(step.get("rxn_smiles"))
        ]
        hit_count = len(set(pred_rxns) & gt_set)
        best_hits = max(best_hits, hit_count)
        if hit_count and any_exact_rank is None:
            any_exact_rank = route_rank
        if gt_rxns and (pred_rxns == gt_rxns or pred_rxns == list(reversed(gt_rxns))):
            exact_route_rank = route_rank if exact_route_rank is None else exact_route_rank
        per_route.append({
            "route_rank": route_rank,
            "n_steps": len(pred_rxns),
            "gt_reaction_hits": hit_count,
            "rxn_smiles": pred_rxns,
        })
    return {
        "target_smiles": row.get("target_smiles"),
        "route_domain": row.get("route_domain"),
        "depth": row.get("depth"),
        "solved": bool(row.get("solved")),
        "route_count": int(row.get("route_count") or 0),
        "gt_n_reactions": len(gt_rxns),
        "exact_reaction_in_route_pool": any_exact_rank is not None,
        "exact_reaction_first_route_rank": any_exact_rank,
        "best_exact_reaction_hits": best_hits,
        "best_exact_reaction_fraction": round(best_hits / max(len(gt_rxns), 1), 6) if gt_rxns else None,
        "exact_route_reaction_match_any": exact_route_rank is not None,
        "exact_route_reaction_first_rank": exact_route_rank,
        "per_route_preview": per_route[:10],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    fractions = [row["best_exact_reaction_fraction"] for row in rows if row["best_exact_reaction_fraction"] is not None]
    return {
        "n_targets": n,
        "solved": sum(1 for row in rows if row["solved"]),
        "solved_rate": round(sum(1 for row in rows if row["solved"]) / max(n, 1), 6),
        "total_routes": sum(int(row["route_count"]) for row in rows),
        "avg_route_count": round(sum(int(row["route_count"]) for row in rows) / max(n, 1), 6),
        "exact_reaction_in_route_pool": sum(1 for row in rows if row["exact_reaction_in_route_pool"]),
        "exact_route_reaction_match_any": sum(1 for row in rows if row["exact_route_reaction_match_any"]),
        "avg_best_exact_reaction_fraction": round(sum(fractions) / max(len(fractions), 1), 6) if fractions else None,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ChemEnzy Route Smoke Comparison",
        "",
        "| run | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in payload["results"].items():
        summary = result["recovery_summary"]
        lines.append(
            f"| {name} | {summary['solved']} / {summary['n_targets']} | "
            f"{summary['total_routes']} | {summary['avg_route_count']} | "
            f"{summary['exact_reaction_in_route_pool']} / {summary['n_targets']} | "
            f"{summary['exact_route_reaction_match_any']} / {summary['n_targets']} | "
            f"{summary['avg_best_exact_reaction_fraction']} |"
        )
    return "\n".join(lines) + "\n"


def _parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--run must be NAME=PATH")
    name, path = raw.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="NAME=route_smoke.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown-output")
    args = ap.parse_args()
    runs = dict(_parse_run(item) for item in args.run)
    payload = compare_runs(runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))


if __name__ == "__main__":
    main()
