#!/usr/bin/env python
"""Audit ChemEnzy route outputs on the BioNavi-like enzymatic benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles, reaction_reactants


ENZYME_SOURCE_MARKERS = (
    "bionav",
    "enzyme",
    "enz",
    "bkms",
    "biocatalysis",
    "reaxys_biocatalysis",
    "onmt_models.bionav",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, help="NAME=chem_enzy_output.json")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 means infer from each run length.")
    args = ap.parse_args()

    benchmark_rows = _load_benchmark(args.benchmark)
    runs = dict(_parse_run(item) for item in args.run)
    payload = {
        "schema_version": "bionavi_chem_enzy_ab_audit.v1",
        "benchmark": str(args.benchmark),
        "benchmark_rows": len(benchmark_rows),
        "runs": {},
        "pairwise": {},
    }
    for name, path in runs.items():
        run_payload = json.loads(path.read_text(encoding="utf-8"))
        rows = _audit_run(
            benchmark_rows,
            run_payload,
            limit=int(args.limit or 0),
            run_name=name,
            run_path=path,
        )
        payload["runs"][name] = {
            "path": str(path),
            "native_summary": run_payload.get("summary") or {},
            "summary": _summarize(rows),
            "targets": rows,
        }
    if len(runs) >= 2:
        names = list(runs)
        for left_idx, left in enumerate(names):
            for right in names[left_idx + 1 :]:
                payload["pairwise"][f"{left}_vs_{right}"] = _pairwise(
                    payload["runs"][left]["targets"],
                    payload["runs"][right]["targets"],
                    left,
                    right,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md = _markdown(payload)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(md, encoding="utf-8")
    print(md)


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    return [row for row in data if isinstance(row, dict) and row.get("target_smiles")]


def _parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--run must be NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--run name is empty")
    return name, Path(path.strip())


def _audit_run(
    benchmark_rows: list[dict[str, Any]],
    run_payload: dict[str, Any],
    *,
    limit: int,
    run_name: str,
    run_path: Path,
) -> list[dict[str, Any]]:
    run_targets = [row for row in (run_payload.get("targets") or []) if isinstance(row, dict)]
    n = min(len(run_targets), len(benchmark_rows), limit or 10**12)
    rows = []
    for idx in range(n):
        bench = benchmark_rows[idx]
        result = run_targets[idx]
        rows.append(_audit_target(idx, bench, result, run_name=run_name, run_path=run_path))
    return rows


def _audit_target(
    idx: int,
    bench: dict[str, Any],
    result: dict[str, Any],
    *,
    run_name: str,
    run_path: Path,
) -> dict[str, Any]:
    gt_rxns = _gt_reactions(bench)
    gt_rxn_counter = Counter(gt_rxns)
    gt_reactants = _gt_reactants(bench)
    routes = result.get("routes") or []
    per_route = []
    best_hits = 0
    best_fraction = None
    best_reactant_hits = 0
    exact_rxn_rank = None
    reactant_rank = None
    exact_route_rank = None
    enzyme_route_count = 0
    enzyme_step_count = 0
    all_pred_sources = Counter()

    for route_idx, route in enumerate(routes, 1):
        pred_rxns = _route_reactions(route)
        pred_counter = Counter(pred_rxns)
        rxn_hits = sum((gt_rxn_counter & pred_counter).values())
        pred_reactants = _route_reactants(route)
        reactant_hits = sorted(gt_reactants & pred_reactants)
        route_enzyme_steps = [_is_enzyme_like_step(step) for step in route.get("steps") or []]
        if any(route_enzyme_steps):
            enzyme_route_count += 1
        enzyme_step_count += sum(1 for item in route_enzyme_steps if item)
        for step in route.get("steps") or []:
            all_pred_sources[str(step.get("source_model") or "")] += 1
        if rxn_hits and exact_rxn_rank is None:
            exact_rxn_rank = route_idx
        if reactant_hits and reactant_rank is None:
            reactant_rank = route_idx
        if _exact_route_match(pred_rxns, gt_rxns) and exact_route_rank is None:
            exact_route_rank = route_idx
        if rxn_hits > best_hits:
            best_hits = rxn_hits
        fraction = rxn_hits / len(gt_rxns) if gt_rxns else None
        if fraction is not None:
            best_fraction = fraction if best_fraction is None else max(best_fraction, fraction)
        best_reactant_hits = max(best_reactant_hits, len(reactant_hits))
        per_route.append(
            {
                "route_rank": route_idx,
                "n_steps": len(route.get("steps") or []),
                "exact_reaction_hits": rxn_hits,
                "exact_reaction_fraction": fraction,
                "gt_reactant_hit_count": len(reactant_hits),
                "enzyme_like_step_count": sum(1 for item in route_enzyme_steps if item),
                "sources": sorted({str(step.get("source_model") or "") for step in route.get("steps") or []})[:12],
            }
        )

    target_expected = canonical_smiles(str(bench.get("target_smiles") or ""))
    target_observed = canonical_smiles(str(result.get("target_smiles") or ""))
    return {
        "index": idx,
        "run": run_name,
        "run_path": str(run_path),
        "cascade_id": bench.get("cascade_id"),
        "target_smiles": bench.get("target_smiles"),
        "target_match": target_expected == target_observed,
        "route_domain": bench.get("route_domain"),
        "depth": bench.get("depth"),
        "gt_n_reactions": len(gt_rxns),
        "gt_n_reactants": len(gt_reactants),
        "solved": bool(result.get("solved")),
        "route_count": int(result.get("route_count") or len(routes)),
        "enzyme_like_route_count": enzyme_route_count,
        "enzyme_like_step_count": enzyme_step_count,
        "exact_reaction_in_route_pool": exact_rxn_rank is not None,
        "exact_reaction_first_rank": exact_rxn_rank,
        "exact_route_reaction_match_any": exact_route_rank is not None,
        "exact_route_reaction_first_rank": exact_route_rank,
        "gt_reactant_in_route_pool": reactant_rank is not None,
        "gt_reactant_first_rank": reactant_rank,
        "best_exact_reaction_hits": best_hits,
        "best_exact_reaction_fraction": best_fraction,
        "best_gt_reactant_hits": best_reactant_hits,
        "best_gt_reactant_fraction": best_reactant_hits / len(gt_reactants) if gt_reactants else None,
        "top_sources": all_pred_sources.most_common(8),
        "failure_categories": [
            item.get("category")
            for item in result.get("failures") or []
            if isinstance(item, dict) and item.get("category")
        ],
        "per_route_preview": per_route[:10],
    }


def _gt_reactions(row: dict[str, Any]) -> list[str]:
    return [
        key
        for key in (canonical_reaction(step.get("rxn_smiles")) for step in row.get("gt_route") or [])
        if key
    ]


def _gt_reactants(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for step in row.get("gt_route") or []:
        out.update(reaction_reactants(step.get("rxn_smiles")))
    return {item for item in out if item}


def _route_reactions(route: dict[str, Any]) -> list[str]:
    keys = []
    for step in route.get("steps") or []:
        key = canonical_reaction(step.get("rxn_smiles") or step.get("reaction_smiles"))
        if key:
            keys.append(key)
    return keys


def _route_reactants(route: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for step in route.get("steps") or []:
        raw_reactants = step.get("reactant_smiles") or step.get("reactants") or []
        if isinstance(raw_reactants, str):
            raw_reactants = [raw_reactants]
        for smi in raw_reactants:
            key = canonical_smiles(str(smi or ""))
            if key:
                out.add(key)
        out.update(reaction_reactants(step.get("rxn_smiles") or step.get("reaction_smiles")))
    return out


def _exact_route_match(pred_rxns: list[str], gt_rxns: list[str]) -> bool:
    if not pred_rxns or not gt_rxns or len(pred_rxns) != len(gt_rxns):
        return False
    return pred_rxns == gt_rxns or pred_rxns == list(reversed(gt_rxns))


def _is_enzyme_like_step(step: dict[str, Any]) -> bool:
    if step.get("enzyme_ec_annotations") or step.get("catalyst_annotations"):
        return True
    source = str(step.get("source_model") or "").lower()
    if any(marker in source for marker in ENZYME_SOURCE_MARKERS):
        return True
    raw = step.get("raw_backend_metadata") if isinstance(step.get("raw_backend_metadata"), dict) else {}
    template = raw.get("template")
    if isinstance(template, dict):
        template_text = json.dumps(template, sort_keys=True, default=str).lower()
        return any(marker in template_text for marker in ENZYME_SOURCE_MARKERS) or bool(template.get("ec"))
    return False


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    exact = sum(1 for row in rows if row["exact_reaction_in_route_pool"])
    exact_route = sum(1 for row in rows if row["exact_route_reaction_match_any"])
    gt_reactant = sum(1 for row in rows if row["gt_reactant_in_route_pool"])
    solved = sum(1 for row in rows if row["solved"])
    target_mismatch = sum(1 for row in rows if not row["target_match"])
    best_fracs = [row["best_exact_reaction_fraction"] for row in rows if row["best_exact_reaction_fraction"] is not None]
    reactant_fracs = [row["best_gt_reactant_fraction"] for row in rows if row["best_gt_reactant_fraction"] is not None]
    failures = Counter(cat for row in rows for cat in row.get("failure_categories") or [])
    source_counts = Counter()
    for row in rows:
        source_counts.update(dict(row.get("top_sources") or []))
    return {
        "n_targets": n,
        "target_mismatch": target_mismatch,
        "solved": solved,
        "solved_rate": _rate(solved, n),
        "total_routes": sum(int(row["route_count"]) for row in rows),
        "avg_route_count": _mean([row["route_count"] for row in rows]),
        "targets_with_enzyme_like_route": sum(1 for row in rows if row["enzyme_like_route_count"] > 0),
        "total_enzyme_like_steps": sum(int(row["enzyme_like_step_count"]) for row in rows),
        "exact_reaction_in_route_pool": exact,
        "exact_reaction_in_route_pool_rate": _rate(exact, n),
        "gt_reactant_in_route_pool": gt_reactant,
        "gt_reactant_in_route_pool_rate": _rate(gt_reactant, n),
        "exact_route_reaction_match_any": exact_route,
        "exact_route_reaction_match_rate": _rate(exact_route, n),
        "avg_best_exact_reaction_fraction": _mean(best_fracs),
        "median_best_exact_reaction_fraction": _median(best_fracs),
        "avg_best_gt_reactant_fraction": _mean(reactant_fracs),
        "failure_categories": dict(sorted(failures.items())),
        "top_sources": source_counts.most_common(12),
    }


def _pairwise(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    n = min(len(left_rows), len(right_rows))
    left_exact_only = right_exact_only = both_exact = neither_exact = 0
    left_gt_only = right_gt_only = both_gt = neither_gt = 0
    left_frac_better = right_frac_better = frac_equal = 0
    left_routes_more = right_routes_more = routes_equal = 0
    for idx in range(n):
        lrow = left_rows[idx]
        rrow = right_rows[idx]
        lexact = bool(lrow.get("exact_reaction_in_route_pool"))
        rexact = bool(rrow.get("exact_reaction_in_route_pool"))
        left_exact_only += int(lexact and not rexact)
        right_exact_only += int(rexact and not lexact)
        both_exact += int(lexact and rexact)
        neither_exact += int(not lexact and not rexact)
        lgt = bool(lrow.get("gt_reactant_in_route_pool"))
        rgt = bool(rrow.get("gt_reactant_in_route_pool"))
        left_gt_only += int(lgt and not rgt)
        right_gt_only += int(rgt and not lgt)
        both_gt += int(lgt and rgt)
        neither_gt += int(not lgt and not rgt)
        lfrac = float(lrow.get("best_exact_reaction_fraction") or 0.0)
        rfrac = float(rrow.get("best_exact_reaction_fraction") or 0.0)
        left_frac_better += int(lfrac > rfrac)
        right_frac_better += int(rfrac > lfrac)
        frac_equal += int(lfrac == rfrac)
        lroutes = int(lrow.get("route_count") or 0)
        rroutes = int(rrow.get("route_count") or 0)
        left_routes_more += int(lroutes > rroutes)
        right_routes_more += int(rroutes > lroutes)
        routes_equal += int(lroutes == rroutes)
    return {
        "n_paired": n,
        "exact_reaction": {
            f"{left}_only": left_exact_only,
            f"{right}_only": right_exact_only,
            "both": both_exact,
            "neither": neither_exact,
        },
        "gt_reactant": {
            f"{left}_only": left_gt_only,
            f"{right}_only": right_gt_only,
            "both": both_gt,
            "neither": neither_gt,
        },
        "best_exact_reaction_fraction": {
            f"{left}_better": left_frac_better,
            f"{right}_better": right_frac_better,
            "equal": frac_equal,
        },
        "route_count": {
            f"{left}_more": left_routes_more,
            f"{right}_more": right_routes_more,
            "equal": routes_equal,
        },
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return round(sum(nums) / len(nums), 6) if nums else None


def _median(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return round(statistics.median(nums), 6) if nums else None


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# BioNavi-like ChemEnzy A/B Audit",
        "",
        f"Benchmark: `{payload['benchmark']}`",
        f"Benchmark rows: {payload['benchmark_rows']}",
        "",
        "| run | n | solved | routes | avg routes | enzyme-like targets | exact GT step | GT reactant | full GT route | avg GT-step frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, run in payload["runs"].items():
        s = run["summary"]
        lines.append(
            f"| {name} | {s['n_targets']} | {s['solved']} ({s['solved_rate']}) | "
            f"{s['total_routes']} | {s['avg_route_count']} | "
            f"{s['targets_with_enzyme_like_route']} | "
            f"{s['exact_reaction_in_route_pool']} ({s['exact_reaction_in_route_pool_rate']}) | "
            f"{s['gt_reactant_in_route_pool']} ({s['gt_reactant_in_route_pool_rate']}) | "
            f"{s['exact_route_reaction_match_any']} ({s['exact_route_reaction_match_rate']}) | "
            f"{s['avg_best_exact_reaction_fraction']} |"
        )
    if payload.get("pairwise"):
        lines.extend(["", "## Pairwise"])
        for name, pair in payload["pairwise"].items():
            lines.append("")
            lines.append(f"### {name}")
            lines.append(f"- paired targets: {pair['n_paired']}")
            lines.append(f"- exact GT step: {pair['exact_reaction']}")
            lines.append(f"- GT reactant: {pair['gt_reactant']}")
            lines.append(f"- best GT-step fraction: {pair['best_exact_reaction_fraction']}")
            lines.append(f"- route count: {pair['route_count']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
