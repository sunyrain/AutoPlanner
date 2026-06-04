"""Export formal statin benchmark rows into renderer-ready route documents."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.product_route_feasibility_audit import (  # noqa: E402
    build_product_route_feasibility_audit,
    product_audit_guard_key,
    product_audit_risk_order,
)
from scripts.summarize_statin_depth20_routes import _audit_target_payload  # noqa: E402


DEFAULT_ROWS = Path(
    "results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/"
    "enhanced_all9/native_vs_enhanced_route_rows.rcr_backfilled.jsonl"
)
DEFAULT_TARGETS = Path("results/shared/statin_enhanced_combo_20260601/statin_target_rows_all9.jsonl")
DEFAULT_OUTPUT = Path(
    "results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/"
    "top_route_render_rcr_backfilled"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help="Allow needs_chemist_review routes when better triage routes exist.",
    )
    args = parser.parse_args()

    rows_path = Path(args.rows)
    targets_path = Path(args.targets)
    output_dir = Path(args.output_dir)
    route_doc_dir = output_dir / "route_docs"
    route_doc_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(rows_path)
    targets = _read_jsonl(targets_path)
    name_by_smiles = {str(row.get("target_smiles") or ""): str(row.get("name") or "") for row in targets}
    meta_by_smiles = {str(row.get("target_smiles") or ""): row for row in targets}
    audit = build_product_route_feasibility_audit(
        {
            "metadata": {"source": "statin_formal_top_route_export", "rows": str(rows_path)},
            "targets": [_audit_target_payload(idx, row, name_by_smiles) for idx, row in enumerate(rows)],
        }
    )

    summary_rows: list[dict[str, Any]] = []
    aggregate = Counter()
    for idx, row in enumerate(rows, start=1):
        smiles = str(row.get("target_smiles") or row.get("target_canonical") or "")
        meta = meta_by_smiles.get(smiles) or {}
        name = name_by_smiles.get(smiles) or str(row.get("label") or f"target_{idx}")
        audit_target = (audit.get("targets") or [{}])[idx - 1] if idx <= len(audit.get("targets") or []) else {}
        audit_by_rank = {
            int(route_audit.get("rank") or 0): route_audit
            for route_audit in audit_target.get("routes") or []
            if isinstance(route_audit, dict)
        }
        route_records = []
        for route_rank, route in enumerate(row.get("routes") or [], start=1):
            if not isinstance(route, dict):
                continue
            audit_route = audit_by_rank.get(route_rank) or {}
            route_class = str(audit_route.get("route_class") or "")
            if not args.include_needs_review and route_class == "needs_chemist_review":
                better = any(
                    str(candidate.get("route_class") or "") != "needs_chemist_review"
                    for candidate in audit_by_rank.values()
                )
                if better:
                    continue
            route_records.append(
                {
                    "route_rank": route_rank,
                    "route": route,
                    "audit": audit_route,
                }
            )
        selected = _select_routes(route_records, top_k=max(0, int(args.top_k or 0)))
        doc_routes = [
            _route_doc_payload(item["route"], item["audit"], rank=rank, original_rank=int(item["route_rank"]))
            for rank, item in enumerate(selected, start=1)
        ]
        doc = {
            "schema_version": "statin_formal_route_doc.v1",
            "source_rows": str(rows_path),
            "target": smiles,
            "target_smiles": smiles,
            "target_name": name,
            "cascade_id": name,
            "panel": meta.get("statin_family") or meta.get("label_source"),
            "source_route_count": len([route for route in row.get("routes") or [] if isinstance(route, dict)]),
            "source_solved_routes": int(row.get("solved_routes") or 0),
            "selection_policy": {
                "top_k": int(args.top_k),
                "sort": "product_audit_guard_key, solved, condition coverage, original rank",
                "include_needs_chemist_review": bool(args.include_needs_review),
            },
            "target_audit": {
                "best_rank": (audit_target.get("best_route") or {}).get("rank"),
                "best_class": (audit_target.get("best_route") or {}).get("route_class"),
                "target_verdict": audit_target.get("target_verdict"),
            },
            "routes": doc_routes,
        }
        path = route_doc_dir / f"{_slug(name)}_top{int(args.top_k)}_routes.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(_summary_row(doc, path))
        aggregate["targets"] += 1
        aggregate["routes"] += int(doc.get("source_route_count") or 0)
        aggregate["selected_routes"] += len(doc_routes)

    summary = {
        "schema_version": "statin_formal_top_route_export.v1",
        "rows": str(rows_path),
        "targets": str(targets_path),
        "output_dir": str(output_dir),
        "route_doc_dir": str(route_doc_dir),
        "top_k": int(args.top_k),
        "aggregate": dict(aggregate),
        "rows_summary": summary_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "route_doc_dir": str(route_doc_dir), "summary": str(output_dir / "summary.json")}, indent=2))


def _select_routes(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    return sorted(records, key=_record_sort_key)[:top_k]


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int, int, int, int, int, float]:
    route = record.get("route") or {}
    audit = record.get("audit") or {}
    condition = audit.get("condition_audit") or {}
    route_rank = int(record.get("route_rank") or 10**9)
    n_steps = len(route.get("steps") or [])
    coverage = _condition_coverage(route.get("steps") or [])
    score = _safe_float(route.get("score"), default=0.0)
    guard = product_audit_guard_key(audit)
    return (
        int(guard[0]),
        int(guard[1]),
        0 if bool(route.get("route_solved") or (route.get("metrics") or {}).get("strict_stock_solve")) else 1,
        0 if str(condition.get("route_risk") or "ok") == "ok" else 1,
        -int(round(coverage * 1000)),
        route_rank,
        -score,
    )


def _route_doc_payload(route: dict[str, Any], audit: dict[str, Any], *, rank: int, original_rank: int) -> dict[str, Any]:
    steps = [_step_payload(step, index=idx) for idx, step in enumerate(route.get("steps") or [], start=1)]
    return {
        "rank": rank,
        "display_rank": rank,
        "original_rank": original_rank,
        "score": route.get("score"),
        "solved": bool(route.get("route_solved") or (route.get("metrics") or {}).get("strict_stock_solve")),
        "route_solved": bool(route.get("route_solved")),
        "progressive_route": bool(route.get("progressive_route")),
        "has_enzyme_step": bool(route.get("has_enzyme_step")),
        "has_sp_v1_accepted_enzyme_step": bool(route.get("has_sp_v1_accepted_enzyme_step")),
        "n_steps": len(steps),
        "source_counts": route.get("source_counts") or {},
        "metrics": {
            **dict(route.get("metrics") or {}),
            "condition_coverage": _condition_coverage(steps),
        },
        "product_audit": {
            **dict(audit or {}),
            "risk_order": product_audit_risk_order(audit or {}),
        },
        "steps": steps,
    }


def _step_payload(step: dict[str, Any], *, index: int) -> dict[str, Any]:
    out = dict(step)
    out["index"] = index
    out.setdefault("reaction_smiles", str(step.get("reaction_smiles") or step.get("rxn_smiles") or ""))
    if not out.get("product"):
        out["product"] = _reaction_product(out.get("reaction_smiles"))
    if not out.get("main_reactant"):
        reactants = _rxn_lhs_parts(out.get("reaction_smiles"))
        if reactants:
            out["main_reactant"] = reactants[0]
    out.setdefault("aux_reactants", [])
    out.setdefault("condition_predictions", [row for row in step.get("condition_predictions") or [] if isinstance(row, dict)])
    return out


def _summary_row(doc: dict[str, Any], path: Path) -> dict[str, Any]:
    routes = [route for route in doc.get("routes") or [] if isinstance(route, dict)]
    first = routes[0] if routes else {}
    audit = first.get("product_audit") or {}
    condition = audit.get("condition_audit") or {}
    return {
        "target": doc.get("target_name"),
        "source_routes": doc.get("source_route_count"),
        "selected_routes": len(routes),
        "best_original_rank": first.get("original_rank"),
        "best_steps": first.get("n_steps"),
        "best_class": audit.get("route_class"),
        "best_condition_risk": condition.get("route_risk"),
        "best_condition_coverage": (first.get("metrics") or {}).get("condition_coverage"),
        "route_doc": str(path),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Statin Formal Top Route Render Export",
        "",
        f"- Rows: `{summary.get('rows')}`",
        f"- Route docs: `{summary.get('route_doc_dir')}`",
        f"- Top-k: `{summary.get('top_k')}`",
        "",
        "| target | source routes | selected | best original rank | best steps | best class | condition risk | condition coverage | route doc |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in summary.get("rows_summary") or []:
        cov = row.get("best_condition_coverage")
        cov_text = "" if cov is None else f"{float(cov):.2f}"
        lines.append(
            "| {target} | {source_routes} | {selected_routes} | {rank} | {steps} | `{klass}` | `{risk}` | {cov} | `{doc}` |".format(
                target=row.get("target"),
                source_routes=row.get("source_routes"),
                selected_routes=row.get("selected_routes"),
                rank=row.get("best_original_rank"),
                steps=row.get("best_steps"),
                klass=row.get("best_class"),
                risk=row.get("best_condition_risk"),
                cov=cov_text,
                doc=row.get("route_doc"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _condition_coverage(steps: list[dict[str, Any]]) -> float:
    clean = [step for step in steps if isinstance(step, dict)]
    if not clean:
        return 0.0
    return sum(1 for step in clean if step.get("condition_predictions")) / len(clean)


def _reaction_product(rxn_smiles: Any) -> str:
    rxn = str(rxn_smiles or "")
    if ">>" not in rxn:
        return ""
    _lhs, rhs = rxn.split(">>", 1)
    return rhs.split(".")[0] if rhs else ""


def _rxn_lhs_parts(rxn_smiles: Any) -> list[str]:
    rxn = str(rxn_smiles or "")
    if ">>" not in rxn:
        return []
    lhs, _rhs = rxn.split(">>", 1)
    return [part for part in lhs.split(".") if part]


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip().lower()).strip("_")
    return slug or "target"


if __name__ == "__main__":
    main()
