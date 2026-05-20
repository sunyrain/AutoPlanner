"""Export native ChemEnzy batch results into report-ready top-route packages.

The native baseline output is convenient for machine evaluation, while the
linear scheme renderer expects a route document with explicit product/reactant
fields. This script bridges the two formats and writes per-target top-K JSON
files plus a compact Chinese summary for reporting.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - script fallback when RDKit is unavailable.
    Chem = None  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-target top route docs from native ChemEnzy JSON.")
    parser.add_argument("--input", action="append", required=True, help="Native baseline JSON. May be supplied more than once.")
    parser.add_argument("--benchmark", help="Benchmark JSON with target_name/cascade_id metadata.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=0, help="Prefer routes with at least this many steps.")
    parser.add_argument("--target-steps", type=int, default=0, help="Prefer routes closest to this step count.")
    parser.add_argument("--diverse-steps", action="store_true", help="Select top routes across different step-count bins.")
    parser.add_argument("--max-terminal-heavy", type=int, help="Hard-filter long-route terminal stock molecules by heavy atom count when possible.")
    parser.add_argument("--short-route-steps", type=int, default=2)
    parser.add_argument("--max-short-route-terminal-heavy", type=int, default=24)
    args = parser.parse_args()

    benchmark_rows = _read_benchmark(Path(args.benchmark)) if args.benchmark else []
    target_meta = _target_metadata(benchmark_rows)
    target_rows = _load_targets([Path(path) for path in args.input])
    output_dir = Path(args.output_dir)
    route_doc_dir = output_dir / "route_docs"
    route_doc_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for target in target_rows:
        smiles = str(target.get("target_smiles") or "")
        meta = _lookup_target_meta(smiles, target_meta)
        name = str(meta.get("target_name") or meta.get("cascade_id") or _short_id(smiles))
        safe_name = _slug(name)
        routes = _select_top_routes(
            target.get("routes") or [],
            top_k=max(0, args.top_k),
            min_steps=max(0, args.min_steps),
            target_steps=max(0, args.target_steps),
            diverse_steps=bool(args.diverse_steps),
            max_terminal_heavy=args.max_terminal_heavy,
            short_route_steps=max(0, args.short_route_steps),
            max_short_route_terminal_heavy=max(0, args.max_short_route_terminal_heavy),
        )
        route_docs = [_convert_route(route, rank=idx, target_smiles=smiles) for idx, route in enumerate(routes, start=1)]
        doc = {
            "target": smiles,
            "target_smiles": smiles,
            "target_name": name,
            "cascade_id": meta.get("cascade_id") or name,
            "panel": (meta.get("metadata") or {}).get("panel"),
            "source_route_count": int(target.get("route_count") or len(target.get("routes") or [])),
            "source_solved": bool(target.get("solved")),
            "top_k": args.top_k,
            "routes": route_docs,
            "failures": target.get("failures") or [],
            "raw_backend_metadata": target.get("raw_backend_metadata") or {},
        }
        doc_path = route_doc_dir / f"{safe_name}_top{args.top_k}_routes.json"
        doc_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_rows.append(_summary_row(doc, doc_path=doc_path))

    summary = {
        "n_targets": len(target_rows),
        "n_targets_solved": sum(1 for row in summary_rows if row["solved"]),
        "total_source_routes": sum(int(row["route_count"]) for row in summary_rows),
        "rows": summary_rows,
    }
    (output_dir / "statin_top_routes_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "statin_top_routes_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False))


def _read_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    return [row for row in data if isinstance(row, dict) and row.get("target_smiles")]


def _target_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        smiles = str(row.get("target_smiles") or "")
        if not smiles:
            continue
        out[smiles] = row
        canon = _canonical_smiles(smiles)
        if canon:
            out[canon] = row
    return out


def _lookup_target_meta(smiles: str, target_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if smiles in target_meta:
        return target_meta[smiles]
    canon = _canonical_smiles(smiles)
    if canon and canon in target_meta:
        return target_meta[canon]
    return {"target_name": _short_id(smiles), "cascade_id": _short_id(smiles), "metadata": {}}


def _load_targets(paths: list[Path]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    targets: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("targets") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"unsupported native output: {path}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            smiles = str(row.get("target_smiles") or "")
            key = _canonical_smiles(smiles) or smiles
            if key in seen:
                continue
            seen.add(key)
            targets.append(row)
    return targets


def _select_top_routes(
    routes: list[dict[str, Any]],
    *,
    top_k: int,
    min_steps: int = 0,
    target_steps: int = 0,
    diverse_steps: bool = False,
    max_terminal_heavy: int | None = None,
    short_route_steps: int = 2,
    max_short_route_terminal_heavy: int = 24,
) -> list[dict[str, Any]]:
    normalized = [route for route in routes if isinstance(route, dict)]
    if not normalized:
        return []
    solved = [route for route in normalized if route.get("solved")]
    pool = solved if solved else normalized
    filtered = [
        route for route in pool
        if not _is_short_large_terminal_route(
            route,
            short_route_steps=short_route_steps,
            max_short_route_terminal_heavy=max_short_route_terminal_heavy,
        )
    ]
    if len(filtered) >= top_k:
        pool = filtered
    if min_steps:
        long_routes = [route for route in pool if len(route.get("steps") or []) >= min_steps]
        if len(long_routes) >= top_k:
            pool = long_routes
    if max_terminal_heavy is not None:
        size_filtered = [route for route in pool if _max_terminal_heavy(route) <= max_terminal_heavy]
        if len(size_filtered) >= top_k:
            pool = size_filtered
    if diverse_steps:
        selected = _select_diverse_step_routes(pool, top_k=top_k)
        if len(selected) >= top_k:
            return selected
    return sorted(pool, key=lambda route: _route_sort_key(route, target_steps=target_steps))[:top_k]


def _route_sort_key(route: dict[str, Any], *, target_steps: int = 0) -> tuple[int, int, int, int, float]:
    rank = route.get("route_rank")
    try:
        rank_i = int(rank)
    except (TypeError, ValueError):
        rank_i = 10**9
    score = route.get("score")
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = -1.0
    n_steps = len(route.get("steps") or [])
    step_distance = abs(n_steps - target_steps) if target_steps else 0
    terminal_heavy = _max_terminal_heavy(route) if target_steps else 0
    return (0 if route.get("solved") else 1, step_distance, terminal_heavy, rank_i, -score_f)


def _select_diverse_step_routes(routes: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    bins = [
        ("9-10", 9, 10),
        ("11+", 11, 10**9),
        ("7-8", 7, 8),
        ("5-6", 5, 6),
        ("3-4", 3, 4),
    ]
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for _label, lo, hi in bins:
        candidates = [
            route for route in routes
            if id(route) not in used_ids and lo <= len(route.get("steps") or []) <= hi
        ]
        if not candidates:
            continue
        choice = sorted(candidates, key=lambda route: _route_sort_key(route, target_steps=0))[0]
        selected.append(choice)
        used_ids.add(id(choice))
        if len(selected) >= top_k:
            return selected
    if len(selected) < top_k:
        for route in sorted(routes, key=lambda route: _route_sort_key(route, target_steps=0)):
            if id(route) in used_ids:
                continue
            selected.append(route)
            used_ids.add(id(route))
            if len(selected) >= top_k:
                break
    return selected


def _convert_route(route: dict[str, Any], *, rank: int, target_smiles: str) -> dict[str, Any]:
    steps = [_convert_step(step, index=idx) for idx, step in enumerate(route.get("steps") or [], start=1)]
    return {
        "rank": rank,
        "backend_route_rank": route.get("route_rank"),
        "score": route.get("score"),
        "solved": bool(route.get("solved")),
        "n_steps": len(steps),
        "target_smiles": route.get("target_smiles") or target_smiles,
        "stock_status": route.get("stock_status") or {},
        "metrics": {
            "condition_coverage": _condition_coverage(steps),
            "enzymatic_step_count": sum(1 for step in steps if step.get("enzyme_ec_annotations")),
            "terminal_stock_count": sum(1 for value in (route.get("stock_status") or {}).values() if value is True),
            "max_terminal_heavy_atoms": _max_terminal_heavy(route),
        },
        "steps": steps,
        "raw_backend_metadata": route.get("raw_backend_metadata") or {},
    }


def _convert_step(step: dict[str, Any], *, index: int) -> dict[str, Any]:
    reactants = [str(smi) for smi in step.get("reactant_smiles") or [] if smi]
    main_reactant = reactants[0] if reactants else ""
    return {
        "index": index,
        "product": str(step.get("product_smiles") or _reaction_product(step.get("rxn_smiles")) or ""),
        "main_reactant": main_reactant,
        "aux_reactants": reactants[1:],
        "reactants": reactants,
        "reaction_smiles": str(step.get("rxn_smiles") or ""),
        "reaction_type": _reaction_type(step),
        "source": step.get("source_model") or "",
        "score": step.get("score"),
        "scores": {"retro": step.get("score")},
        "stock_status": step.get("stock_status") or {},
        "condition_predictions": _condition_predictions(step),
        "enzyme_ec_annotations": step.get("enzyme_ec_annotations") or [],
        "catalyst_annotations": step.get("catalyst_annotations") or [],
        "raw_backend_metadata": step.get("raw_backend_metadata") or {},
    }


def _condition_predictions(step: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = step.get("condition_predictions")
    if isinstance(predictions, list) and predictions:
        return [row for row in predictions if isinstance(row, dict)]
    raw = step.get("raw_backend_metadata") or {}
    rxn_attribute = raw.get("rxn_attribute") if isinstance(raw, dict) else None
    if isinstance(rxn_attribute, dict) and rxn_attribute:
        return [rxn_attribute]
    return []


def _reaction_type(step: dict[str, Any]) -> str:
    if step.get("enzyme_ec_annotations") or step.get("catalyst_annotations"):
        return "enzymatic"
    source = str(step.get("source_model") or "")
    if source:
        return source
    return "template"


def _reaction_product(rxn_smiles: Any) -> str:
    rxn = str(rxn_smiles or "")
    if ">>" not in rxn:
        return ""
    _lhs, rhs = rxn.split(">>", 1)
    return rhs.split(".")[0] if rhs else ""


def _condition_coverage(steps: list[dict[str, Any]]) -> float | None:
    if not steps:
        return None
    covered = sum(1 for step in steps if step.get("condition_predictions"))
    return covered / len(steps)


def _is_short_large_terminal_route(
    route: dict[str, Any],
    *,
    short_route_steps: int,
    max_short_route_terminal_heavy: int,
) -> bool:
    n_steps = len(route.get("steps") or [])
    return n_steps <= short_route_steps and _max_terminal_heavy(route) > max_short_route_terminal_heavy


def _max_terminal_heavy(route: dict[str, Any]) -> int:
    terminals = _terminal_reactants(route)
    return max((_heavy_atoms(smiles) for smiles in terminals), default=0)


def _terminal_reactants(route: dict[str, Any]) -> list[str]:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    products = {str(step.get("product_smiles") or "") for step in steps if step.get("product_smiles")}
    out: list[str] = []
    seen: set[str] = set()
    for step in steps:
        stock_status = step.get("stock_status") or {}
        if not isinstance(stock_status, dict):
            continue
        for smiles, in_stock in stock_status.items():
            smiles = str(smiles or "")
            if in_stock is True and smiles and smiles not in products and smiles not in seen:
                out.append(smiles)
                seen.add(smiles)
    if out:
        return out
    for step in steps:
        for smiles in step.get("reactant_smiles") or []:
            smiles = str(smiles or "")
            if smiles and smiles not in products and smiles not in seen:
                out.append(smiles)
                seen.add(smiles)
    return out


def _heavy_atoms(smiles: str) -> int:
    if Chem is None:
        return 0
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != "H")


def _summary_row(doc: dict[str, Any], *, doc_path: Path) -> dict[str, Any]:
    routes = [route for route in doc.get("routes") or [] if isinstance(route, dict)]
    first = routes[0] if routes else {}
    failures = doc.get("failures") or []
    return {
        "target_name": doc.get("target_name"),
        "panel": doc.get("panel"),
        "solved": bool(doc.get("source_solved")),
        "route_count": int(doc.get("source_route_count") or 0),
        "top_routes": len(routes),
        "best_route_steps": first.get("n_steps"),
        "best_route_score": first.get("score"),
        "best_route_condition_coverage": (first.get("metrics") or {}).get("condition_coverage"),
        "route_doc": str(doc_path),
        "failure_categories": [row.get("category") for row in failures if isinstance(row, dict)],
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 他汀面板 Top 路线汇总",
        "",
        f"- 目标数：{summary.get('n_targets')}",
        f"- 有路线目标数：{summary.get('n_targets_solved')}",
        f"- 原始路线总数：{summary.get('total_source_routes')}",
        "",
        "| 药物 | panel | 是否有路线 | 原始路线数 | 推荐路线数 | 最优路线步数 | 最优分数 | 条件覆盖 | 路线JSON |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("rows") or []:
        coverage = row.get("best_route_condition_coverage")
        coverage_text = "" if coverage is None else f"{float(coverage):.0%}"
        score = row.get("best_route_score")
        score_text = "" if score is None else f"{float(score):.4f}"
        lines.append(
            "| {target} | {panel} | {solved} | {route_count} | {top_routes} | {steps} | {score} | {coverage} | `{doc}` |".format(
                target=row.get("target_name") or "",
                panel=row.get("panel") or "",
                solved="是" if row.get("solved") else "否",
                route_count=row.get("route_count") or 0,
                top_routes=row.get("top_routes") or 0,
                steps=row.get("best_route_steps") or "",
                score=score_text,
                coverage=coverage_text,
                doc=row.get("route_doc") or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _canonical_smiles(smiles: str) -> str:
    if Chem is None:
        return ""
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _short_id(smiles: str) -> str:
    return (_canonical_smiles(smiles) or smiles or "unknown")[:24]


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip().lower()).strip("_")
    return slug or "target"


if __name__ == "__main__":
    main()
