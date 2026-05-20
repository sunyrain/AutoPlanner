"""Export representative statin routes from the web-filtered showcase bundle."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("results/shared/statin_panel_20260520/web_showcase/statin_showcase_routes.json")
DEFAULT_OUTPUT = Path("results/shared/statin_panel_20260520/report_package_all9_presentation_ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report-ready representative routes after web filtering.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preferred-min-steps", type=int, default=5)
    parser.add_argument("--include-review", action="store_true", help="Allow needs_chemist_review routes as fallback candidates.")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    route_doc_dir = output_dir / "route_docs"
    route_doc_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in data.get("targets") or []:
        if not isinstance(target, dict):
            continue
        routes = [
            route for route in target.get("routes") or []
            if isinstance(route, dict)
            and (args.include_review or (route.get("product_audit") or {}).get("route_class") != "needs_chemist_review")
        ]
        selected = _select_routes(
            routes,
            top_k=max(0, args.top_k),
            preferred_min_steps=max(1, args.preferred_min_steps),
        )
        for rank, route in enumerate(selected, start=1):
            route["rank"] = rank
            route["display_rank"] = rank
        name = str(target.get("target_name") or target.get("cascade_id") or "target")
        doc = {
            "target": target.get("target_smiles"),
            "target_smiles": target.get("target_smiles"),
            "target_name": name,
            "cascade_id": target.get("cascade_id") or name,
            "panel": target.get("panel"),
            "source_route_count": target.get("raw_route_count"),
            "web_kept_route_count": target.get("web_kept_route_count"),
            "showcase_route_count": target.get("showcase_route_count"),
            "selection_policy": {
                "source": str(args.input),
                "product_audit_mode": (data.get("filters") or {}).get("product_audit_mode"),
                "min_steps_in_showcase": (data.get("filters") or {}).get("min_steps"),
                "preferred_min_steps_for_report": args.preferred_min_steps,
                "terminal_heavy_atom_cap": None,
                "large_terminal_fragments_not_hard_filtered": True,
                "include_needs_chemist_review": bool(args.include_review),
            },
            "routes": selected,
        }
        path = route_doc_dir / f"{_slug(name)}_top{args.top_k}_routes.json"
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(
            {
                "target_name": name,
                "route_doc": str(path),
                "source_showcase_routes": target.get("showcase_route_count"),
                "candidate_routes_after_review_filter": len(routes),
                "selected_routes": len(selected),
                "selected_steps": [route.get("n_steps") for route in selected],
                "selected_original_ranks": [route.get("original_rank") for route in selected],
                "selected_classes": [(route.get("product_audit") or {}).get("route_class") for route in selected],
            }
        )

    summary = {
        "schema_version": "statin_webfiltered_representative_routes.v1",
        "source": str(args.input),
        "output_dir": str(output_dir),
        "top_k": args.top_k,
        "preferred_min_steps": args.preferred_min_steps,
        "include_needs_chemist_review": bool(args.include_review),
        "n_targets": len(rows),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _select_routes(routes: list[dict[str, Any]], *, top_k: int, preferred_min_steps: int) -> list[dict[str, Any]]:
    if not routes or top_k <= 0:
        return []
    preferred = [route for route in routes if int(route.get("n_steps") or 0) >= preferred_min_steps]
    pool = preferred if preferred else routes
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for lo, hi in _bins(preferred_min_steps if preferred else 3):
        candidates = [
            route for route in pool
            if id(route) not in used and lo <= int(route.get("n_steps") or 0) <= hi
        ]
        if not candidates:
            continue
        choice = sorted(candidates, key=_route_sort_key)[0]
        selected.append(json.loads(json.dumps(choice, ensure_ascii=False)))
        used.add(id(choice))
        if len(selected) >= top_k:
            return selected
    for route in sorted(pool, key=_route_sort_key):
        if id(route) in used:
            continue
        selected.append(json.loads(json.dumps(route, ensure_ascii=False)))
        used.add(id(route))
        if len(selected) >= top_k:
            break
    return selected


def _bins(min_steps: int) -> list[tuple[int, int]]:
    if min_steps >= 5:
        return [(11, 12), (9, 10), (13, 20), (7, 8), (5, 6)]
    return [(7, 20), (5, 6), (3, 4)]


def _route_sort_key(route: dict[str, Any]) -> tuple[int, int, int, int, float]:
    audit = route.get("product_audit") or {}
    risk = _int_or_default(audit.get("risk_order"), 99)
    cls = str(audit.get("route_class") or "")
    class_rank = {
        "triage_semisynthesis": 0,
        "triage_late_stage": 1,
        "triage_fragment": 2,
        "needs_chemist_review": 3,
    }.get(cls, 9)
    steps = _int_or_default(route.get("n_steps"), 0)
    original = _int_or_default(route.get("original_rank"), 10**9)
    score = _float_or_default(route.get("score"), -1.0)
    return (-steps, risk, class_rank, original, -score)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Web-filtered 他汀代表路线",
        "",
        f"- source: `{summary.get('source')}`",
        f"- top_k: {summary.get('top_k')}",
        f"- preferred_min_steps: {summary.get('preferred_min_steps')}",
        "",
        "| target | showcase routes | candidates | selected | steps | original ranks | classes |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in summary.get("rows") or []:
        lines.append(
            "| {target} | {source} | {candidates} | {selected} | {steps} | {ranks} | {classes} |".format(
                target=row.get("target_name"),
                source=row.get("source_showcase_routes"),
                candidates=row.get("candidate_routes_after_review_filter"),
                selected=row.get("selected_routes"),
                steps=", ".join(str(v) for v in row.get("selected_steps") or []),
                ranks=", ".join(str(v) for v in row.get("selected_original_ranks") or []),
                classes=", ".join(str(v) for v in row.get("selected_classes") or []),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip().lower()).strip("_")
    return slug or "target"


if __name__ == "__main__":
    main()
