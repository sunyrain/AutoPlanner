"""Build the static data bundle for the statin showcase page.

The page intentionally mirrors the web service presentation policy: routes are
ranked and filtered with product-audit hide_rejects, then the statin showcase
removes only 1-2 step closures for presentation. It does not hard-filter large
terminal fragments because those can be legitimate group-introduction reagents.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.product_route_feasibility_audit import (
    build_product_route_feasibility_audit,
    product_audit_guard_key,
    product_audit_risk_order,
)
from cascade_planner.web.app import (
    _compact_product_audit_row,
    _missing_product_audit_row,
    _remove_route_by_product_audit,
)
from scripts.export_native_top_routes_for_render import (
    _canonical_smiles,
    _convert_route,
    _lookup_target_meta,
    _read_benchmark,
    _slug,
    _target_metadata,
)


DEFAULT_INPUT = Path(
    "results/shared/statin_panel_20260520/full_depth20_iter200_top100_conditions/"
    "native_all9_iter200_depth20_top100_search.json"
)
DEFAULT_BENCHMARKS = [
    Path("results/shared/statin_panel_20260520/full_depth20_iter200_top100_conditions/statin_panel_group_a_atorvastatin.json"),
    Path("results/shared/statin_panel_20260520/full_depth20_iter200_top100_conditions/statin_panel_group_b_flu_lova_pita_prava.json"),
    Path("results/shared/statin_panel_20260520/full_depth20_iter200_top100_conditions/statin_panel_group_c_rosu_sim_ceri_meva.json"),
]
DEFAULT_OUTPUT = Path("results/shared/statin_panel_20260520/web_showcase/statin_showcase_routes.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build statin showcase page data from native ChemEnzy output.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Merged native all-target JSON.")
    parser.add_argument(
        "--benchmark",
        action="append",
        help="Benchmark JSON with target metadata. May be supplied more than once.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output showcase JSON.")
    parser.add_argument("--mode", default="hide_rejects", choices=["hide_rejects", "hide_risky", "triage_only", "risk_guarded"])
    parser.add_argument("--min-steps", type=int, default=3, help="Presentation-only minimum route length.")
    args = parser.parse_args()

    input_path = Path(args.input)
    benchmark_paths = [Path(path) for path in args.benchmark] if args.benchmark else DEFAULT_BENCHMARKS
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    native = json.loads(input_path.read_text(encoding="utf-8"))
    targets = [row for row in native.get("targets") or [] if isinstance(row, dict)]
    benchmark_rows = _read_benchmark_many(benchmark_paths)
    benchmark_meta = _target_metadata(benchmark_rows)
    benchmark_lookup = {_canonical_smiles(str(row.get("target_smiles") or "")): row for row in benchmark_rows}

    audit_rows = []
    for target in targets:
        smiles = str(target.get("target_smiles") or "")
        meta = benchmark_lookup.get(_canonical_smiles(smiles)) or _lookup_target_meta(smiles, benchmark_meta)
        audit_rows.append(
            {
                "target_smiles": smiles,
                "target_name": meta.get("target_name") or meta.get("cascade_id") or smiles[:24],
                "cascade_id": meta.get("cascade_id") or meta.get("target_name") or smiles[:24],
                "routes": target.get("routes") or [],
                "route_count": int(target.get("route_count") or len(target.get("routes") or [])),
                "solved": bool(target.get("solved")),
            }
        )

    audit = build_product_route_feasibility_audit(
        {"metadata": {"source": "statin_showcase"}, "targets": audit_rows},
        benchmark_rows=benchmark_rows,
    )

    showcase_targets: list[dict[str, Any]] = []
    aggregate = Counter()
    class_counts_before: Counter[str] = Counter()
    class_counts_web_kept: Counter[str] = Counter()
    class_counts_showcase: Counter[str] = Counter()
    issue_counts_before: Counter[str] = Counter()
    issue_counts_removed: Counter[str] = Counter()

    for target_index, target in enumerate(targets):
        smiles = str(target.get("target_smiles") or "")
        meta = _lookup_target_meta(smiles, benchmark_meta)
        name = str(meta.get("target_name") or meta.get("cascade_id") or smiles[:24])
        native_routes = [route for route in target.get("routes") or [] if isinstance(route, dict)]
        audit_target = (audit.get("targets") or [{}])[target_index] if target_index < len(audit.get("targets") or []) else {}
        audit_by_original = {
            int(row.get("rank") or 0) - 1: row
            for row in audit_target.get("routes") or []
            if isinstance(row, dict) and row.get("rank") is not None
        }

        ranked: list[dict[str, Any]] = []
        target_class_before: Counter[str] = Counter()
        target_class_kept: Counter[str] = Counter()
        target_class_showcase: Counter[str] = Counter()
        target_step_dist: Counter[str] = Counter()
        removed_by_web = 0
        removed_short = 0
        audit_missing = 0

        for original_index, native_route in enumerate(native_routes):
            row = audit_by_original.get(original_index)
            if row is None:
                audit_missing += 1
            risk = product_audit_risk_order(row or {})
            audit_meta = _compact_product_audit_row(row, risk) if row else _missing_product_audit_row()
            route_class = str(audit_meta.get("route_class") or "audit_missing")
            target_class_before[route_class] += 1
            class_counts_before[route_class] += 1
            for issue in audit_meta.get("issues") or []:
                issue_counts_before[str(issue)] += 1

            remove_web = _remove_route_by_product_audit(row or {}, risk=risk, mode=args.mode)
            n_steps = len(native_route.get("steps") or [])
            remove_short = n_steps < max(1, args.min_steps)
            if remove_web:
                removed_by_web += 1
                for issue in audit_meta.get("issues") or []:
                    issue_counts_removed[str(issue)] += 1
                continue
            target_class_kept[route_class] += 1
            class_counts_web_kept[route_class] += 1
            if remove_short:
                removed_short += 1
                continue

            converted = _convert_route(native_route, rank=0, target_smiles=smiles)
            converted["display_rank"] = 0
            converted["original_rank"] = original_index + 1
            converted["backend_route_rank"] = native_route.get("route_rank")
            converted["product_audit"] = audit_meta
            converted["rule_post_rank_metadata"] = {
                "route_class": audit_meta.get("route_class"),
                "risk_order": audit_meta.get("risk_order"),
                "issues": audit_meta.get("issues") or [],
                "tags": audit_meta.get("tags") or [],
                "route_plausibility": audit_meta.get("route_plausibility") or {},
            }
            converted["presentation_filter"] = {
                "min_steps": int(args.min_steps),
                "excluded_short_route": False,
                "large_terminal_fragments_not_hard_filtered": True,
            }
            guard = product_audit_guard_key(row or {}) if row else (99, 99)
            ranked.append(
                {
                    "guard": [*guard, original_index],
                    "route": converted,
                }
            )
            target_class_showcase[route_class] += 1
            class_counts_showcase[route_class] += 1
            target_step_dist[str(n_steps)] += 1

        ranked.sort(key=lambda item: tuple(item["guard"]))
        routes = []
        for display_rank, item in enumerate(ranked, start=1):
            route = item["route"]
            route["rank"] = display_rank
            route["display_rank"] = display_rank
            route["id"] = f"route-{display_rank:04d}"
            routes.append(route)

        raw_count = len(native_routes)
        web_kept = raw_count - removed_by_web
        showcase_count = len(routes)
        aggregate["targets"] += 1
        aggregate["raw_routes"] += raw_count
        aggregate["web_kept_routes"] += web_kept
        aggregate["web_removed_routes"] += removed_by_web
        aggregate["short_removed_routes"] += removed_short
        aggregate["showcase_routes"] += showcase_count

        showcase_targets.append(
            {
                "target_name": name,
                "slug": _slug(name),
                "target_smiles": smiles,
                "cascade_id": meta.get("cascade_id") or name,
                "panel": (meta.get("metadata") or {}).get("panel"),
                "source_solved": bool(target.get("solved")),
                "raw_route_count": raw_count,
                "web_kept_route_count": web_kept,
                "web_removed_route_count": removed_by_web,
                "short_removed_route_count": removed_short,
                "showcase_route_count": showcase_count,
                "audit_missing_route_count": audit_missing,
                "route_class_counts_before": dict(sorted(target_class_before.items())),
                "route_class_counts_web_kept": dict(sorted(target_class_kept.items())),
                "route_class_counts_showcase": dict(sorted(target_class_showcase.items())),
                "step_count_distribution_showcase": dict(sorted(target_step_dist.items(), key=lambda row: int(row[0]))),
                "routes": routes,
            }
        )

    payload = {
        "schema_version": "statin_showcase_routes.v1",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_native": str(input_path),
        "benchmark": [str(path) for path in benchmark_paths],
        "filters": {
            "product_audit_mode": args.mode,
            "min_steps": int(args.min_steps),
            "terminal_heavy_atom_cap": None,
            "large_terminal_fragments_not_hard_filtered": True,
        },
        "aggregate": {
            **dict(aggregate),
            "route_class_counts_before": dict(sorted(class_counts_before.items())),
            "route_class_counts_web_kept": dict(sorted(class_counts_web_kept.items())),
            "route_class_counts_showcase": dict(sorted(class_counts_showcase.items())),
            "issue_counts_before": dict(sorted(issue_counts_before.items())),
            "issue_counts_removed_by_web": dict(sorted(issue_counts_removed.items())),
        },
        "targets": showcase_targets,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "targets": aggregate["targets"],
                "raw_routes": aggregate["raw_routes"],
                "web_kept_routes": aggregate["web_kept_routes"],
                "short_removed_routes": aggregate["short_removed_routes"],
                "showcase_routes": aggregate["showcase_routes"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

def _read_benchmark_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in _read_benchmark(path):
            smiles = str(row.get("target_smiles") or "")
            key = _canonical_smiles(smiles) or smiles
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
