#!/usr/bin/env python3
"""Evaluate GraphFP and dual-tower candidate fusion on one-step exact recovery."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side  # noqa: E402


SCHEMA_VERSION = "graphfp_dualtower_fusion_eval.v1"
EVAL_KS = (1, 3, 5, 10, 20, 50, 75, 100)


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    graphfp = _load_graphfp_rows(args.graphfp_jsonl, source="graphfp", limit_groups=args.limit_groups)
    dual = _load_dual_rows(args.dual_rows_jsonl, source="dualtower", limit_groups=args.limit_groups)
    products = _load_lines(args.src, limit=args.limit_groups)
    targets = _load_lines(args.tgt, limit=args.limit_groups)
    groups = sorted(set(graphfp) | set(dual), key=lambda x: int(x))

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "graphfp_jsonl": str(args.graphfp_jsonl),
            "dual_rows_jsonl": str(args.dual_rows_jsonl),
            "src": str(args.src),
            "tgt": str(args.tgt),
            "limit_groups": args.limit_groups,
        },
        "summary": {
            "graphfp": _evaluate_source(graphfp, groups),
            "dualtower": _evaluate_source(dual, groups),
            "union_oracle": _evaluate_union_oracle(graphfp, dual, groups),
            "fusion": {
                "graphfp_first": _evaluate_fusion(graphfp, dual, groups, mode="graphfp_first"),
                "best_rank": _evaluate_fusion(graphfp, dual, groups, mode="best_rank"),
                "rrf": _evaluate_fusion(graphfp, dual, groups, mode="rrf"),
                "score_sum": _evaluate_fusion(graphfp, dual, groups, mode="score_sum"),
            },
            "complementarity": _complementarity(graphfp, dual, groups),
            "n_groups": len(groups),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_json.with_suffix(".md").write_text(_render_markdown(report), encoding="utf-8")
    if args.output_rows_jsonl:
        _write_rows(
            args.output_rows_jsonl,
            graphfp=graphfp,
            dual=dual,
            groups=groups,
            products=products,
            targets=targets,
        )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


def _load_graphfp_rows(path: Path, *, source: str, limit_groups: int | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = _idx_from_row(row)
            if limit_groups is not None and int(idx) >= int(limit_groups):
                continue
            candidate = row.get("candidate") or {}
            reactants_text = str(candidate.get("reactants_text") or ".".join(candidate.get("reactants") or []))
            key = canonical_side(reactants_text)
            if not key:
                continue
            grouped[idx].append(
                {
                    "idx": idx,
                    "source": source,
                    "rank": int(candidate.get("rank") or len(grouped[idx]) + 1),
                    "score": _safe_float(candidate.get("score")),
                    "reactants_key": key,
                    "reactants_text": ".".join(key),
                    "exact": bool((row.get("labels") or {}).get("exact")),
                    "any_reactant": bool((row.get("labels") or {}).get("any_reactant")),
                    "raw": row,
                }
            )
    return {idx: _dedupe(rows) for idx, rows in grouped.items()}


def _load_dual_rows(path: Path, *, source: str, limit_groups: int | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("candidate") and row.get("labels") is not None:
                row_idx = _idx_from_row(row)
                if limit_groups is not None and int(row_idx) >= int(limit_groups):
                    continue
                candidate = row.get("candidate") or {}
                reactants_text = str(candidate.get("reactants_text") or ".".join(candidate.get("reactants") or []))
                key = canonical_side(reactants_text)
                if not key:
                    continue
                grouped.setdefault(row_idx, []).append(
                    {
                        "idx": row_idx,
                        "source": source,
                        "rank": int(candidate.get("rank") or len(grouped.get(row_idx, [])) + 1),
                        "template_rank": int(candidate.get("template_rank") or 10**9),
                        "score": _safe_float(candidate.get("score")),
                        "reactants_key": key,
                        "reactants_text": ".".join(key),
                        "exact": bool((row.get("labels") or {}).get("exact")),
                        "any_reactant": bool((row.get("labels") or {}).get("any_reactant")),
                        "raw": row,
                    }
                )
                continue
            if limit_groups is not None and idx >= int(limit_groups):
                break
            target_key = canonical_side(str(row.get("target_reactants") or ""))
            rows = []
            for cand in row.get("candidates_preview") or []:
                key = canonical_side(str(cand.get("reactants") or ""))
                if not key:
                    continue
                exact = bool(target_key and key == target_key)
                any_reactant = bool(set(key) & set(target_key))
                rows.append(
                    {
                        "idx": str(idx),
                        "source": source,
                        "rank": int(cand.get("candidate_rank") or len(rows) + 1),
                        "template_rank": int(cand.get("template_rank") or 10**9),
                        "score": _safe_float(cand.get("score")),
                        "reactants_key": key,
                        "reactants_text": ".".join(key),
                        "exact": exact,
                        "any_reactant": any_reactant,
                        "raw": cand,
                    }
                )
            grouped[str(idx)] = rows
    return {idx: _dedupe(rows) for idx, rows in grouped.items()}


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row["reactants_key"])
        current = best.get(key)
        if current is None or int(row["rank"]) < int(current["rank"]):
            best[key] = row
    return sorted(best.values(), key=lambda row: int(row["rank"]))


def _evaluate_source(grouped: dict[str, list[dict[str, Any]]], groups: list[str]) -> dict[str, Any]:
    hits = {k: 0 for k in EVAL_KS}
    any_hits = {k: 0 for k in EVAL_KS}
    nonempty = 0
    total_candidates = 0
    for idx in groups:
        rows = sorted(grouped.get(idx) or [], key=lambda row: int(row["rank"]))
        if rows:
            nonempty += 1
        total_candidates += len(rows)
        for k in EVAL_KS:
            hits[k] += int(any(row["exact"] for row in rows[:k]))
            any_hits[k] += int(any(row["any_reactant"] for row in rows[:k]))
    n = len(groups)
    return {
        "nonempty": nonempty,
        "avg_candidates": round(total_candidates / max(n, 1), 6),
        "exact": {f"@{k}": _rate(hits[k], n) for k in EVAL_KS},
        "any_reactant": {f"@{k}": _rate(any_hits[k], n) for k in EVAL_KS},
    }


def _evaluate_union_oracle(
    graphfp: dict[str, list[dict[str, Any]]],
    dual: dict[str, list[dict[str, Any]]],
    groups: list[str],
) -> dict[str, Any]:
    hits = 0
    any_hits = 0
    for idx in groups:
        rows = _union_rows(graphfp.get(idx) or [], dual.get(idx) or [], mode="rrf")
        hits += int(any(row["exact"] for row in rows))
        any_hits += int(any(row["any_reactant"] for row in rows))
    n = len(groups)
    return {
        "exact": _rate(hits, n),
        "any_reactant": _rate(any_hits, n),
    }


def _evaluate_fusion(
    graphfp: dict[str, list[dict[str, Any]]],
    dual: dict[str, list[dict[str, Any]]],
    groups: list[str],
    *,
    mode: str,
) -> dict[str, Any]:
    hits = {k: 0 for k in EVAL_KS}
    any_hits = {k: 0 for k in EVAL_KS}
    reciprocal_sum = 0.0
    total_candidates = 0
    for idx in groups:
        rows = _union_rows(graphfp.get(idx) or [], dual.get(idx) or [], mode=mode)
        total_candidates += len(rows)
        for k in EVAL_KS:
            hits[k] += int(any(row["exact"] for row in rows[:k]))
            any_hits[k] += int(any(row["any_reactant"] for row in rows[:k]))
        for rank, row in enumerate(rows, 1):
            if row["exact"]:
                reciprocal_sum += 1.0 / rank
                break
    n = len(groups)
    return {
        "avg_candidates": round(total_candidates / max(n, 1), 6),
        "exact": {f"@{k}": _rate(hits[k], n) for k in EVAL_KS},
        "any_reactant": {f"@{k}": _rate(any_hits[k], n) for k in EVAL_KS},
        "mrr": round(reciprocal_sum / max(n, 1), 6),
    }


def _union_rows(graphfp_rows: list[dict[str, Any]], dual_rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in graphfp_rows:
        key = tuple(row["reactants_key"])
        item = dict(row)
        item["graphfp_rank"] = int(row["rank"])
        item["dualtower_rank"] = None
        item["sources"] = {"graphfp"}
        merged[key] = item
    for row in dual_rows:
        key = tuple(row["reactants_key"])
        if key in merged:
            merged[key]["dualtower_rank"] = int(row["rank"])
            merged[key]["sources"].add("dualtower")
            merged[key]["exact"] = bool(merged[key]["exact"] or row["exact"])
            merged[key]["any_reactant"] = bool(merged[key]["any_reactant"] or row["any_reactant"])
            merged[key]["dualtower_score"] = float(row["score"])
        else:
            item = dict(row)
            item["graphfp_rank"] = None
            item["dualtower_rank"] = int(row["rank"])
            item["dualtower_score"] = float(row["score"])
            item["sources"] = {"dualtower"}
            merged[key] = item
    rows = list(merged.values())
    for row in rows:
        row["fusion_score"] = _fusion_score(row, mode=mode)
        row["sources"] = sorted(row["sources"])
    return sorted(rows, key=lambda row: row["fusion_score"], reverse=True)


def _fusion_score(row: dict[str, Any], *, mode: str) -> float:
    g_rank = row.get("graphfp_rank")
    d_rank = row.get("dualtower_rank")
    if mode == "graphfp_first":
        return 1e6 - (g_rank if g_rank is not None else 100000 + (d_rank or 100000))
    if mode == "best_rank":
        return 1e6 - min(g_rank if g_rank is not None else 100000, d_rank if d_rank is not None else 100000)
    if mode == "rrf":
        return (1.0 / (60.0 + (g_rank or 100000))) + (1.0 / (60.0 + (d_rank or 100000)))
    if mode == "score_sum":
        g_score = math.log(max(float(row.get("score") or 0.0), 1e-12)) if g_rank is not None else -12.0
        d_score = float(row.get("dualtower_score") or row.get("score") or 0.0) if d_rank is not None else 0.0
        return g_score + d_score
    raise ValueError(f"unknown fusion mode: {mode}")


def _complementarity(
    graphfp: dict[str, list[dict[str, Any]]],
    dual: dict[str, list[dict[str, Any]]],
    groups: list[str],
) -> dict[str, Any]:
    both = graphfp_only = dual_only = neither = 0
    for idx in groups:
        g = any(row["exact"] for row in graphfp.get(idx) or [])
        d = any(row["exact"] for row in dual.get(idx) or [])
        if g and d:
            both += 1
        elif g:
            graphfp_only += 1
        elif d:
            dual_only += 1
        else:
            neither += 1
    n = len(groups)
    return {
        "both_exact": _rate(both, n),
        "graphfp_only_exact": _rate(graphfp_only, n),
        "dualtower_only_exact": _rate(dual_only, n),
        "neither_exact": _rate(neither, n),
    }


def _write_rows(
    path: Path,
    *,
    graphfp: dict[str, list[dict[str, Any]]],
    dual: dict[str, list[dict[str, Any]]],
    groups: list[str],
    products: list[str],
    targets: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in groups:
            rows = _union_rows(graphfp.get(idx) or [], dual.get(idx) or [], mode="rrf")
            for rank, row in enumerate(rows, 1):
                i = int(idx)
                handle.write(
                    json.dumps(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "idx": i,
                            "group_id": f"fusion:{i}",
                            "product": products[i] if i < len(products) else "",
                            "target_reactants": targets[i] if i < len(targets) else "",
                            "rank": rank,
                            "reactants_text": row["reactants_text"],
                            "sources": row["sources"],
                            "graphfp_rank": row.get("graphfp_rank"),
                            "dualtower_rank": row.get("dualtower_rank"),
                            "fusion_score": row["fusion_score"],
                            "exact": row["exact"],
                            "any_reactant": row["any_reactant"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _load_lines(path: Path, *, limit: int | None) -> list[str]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip().replace(" ", "")
            if text:
                rows.append(text)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _idx_from_row(row: dict[str, Any]) -> str:
    raw = row.get("idx")
    if raw is not None:
        return str(int(raw))
    group_id = str(row.get("group_id") or "")
    return group_id.rsplit(":", 1)[-1]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(value: int, total: int) -> dict[str, Any]:
    return {"count": int(value), "rate": round(float(value) / max(int(total), 1), 6)}


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# GraphFP + DualTower Fusion Eval",
        "",
        f"n_groups: `{summary['n_groups']}`",
        "",
        "| source | exact@20 | exact@50 | exact@100 | any@100 | avg_candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["graphfp", "dualtower"]:
        item = summary[name]
        lines.append(
            f"| {name} | {item['exact']['@20']['rate']} | {item['exact']['@50']['rate']} | "
            f"{item['exact']['@100']['rate']} | {item['any_reactant']['@100']['rate']} | {item['avg_candidates']} |"
        )
    for name, item in summary["fusion"].items():
        lines.append(
            f"| fusion:{name} | {item['exact']['@20']['rate']} | {item['exact']['@50']['rate']} | "
            f"{item['exact']['@100']['rate']} | {item['any_reactant']['@100']['rate']} | {item['avg_candidates']} |"
        )
    oracle = summary["union_oracle"]
    lines.extend(
        [
            "",
            f"union_oracle_exact: `{oracle['exact']['count']} ({oracle['exact']['rate']})`",
            f"union_oracle_any: `{oracle['any_reactant']['count']} ({oracle['any_reactant']['rate']})`",
            "",
            "## Complementarity",
            "",
            "| bucket | count | rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, item in summary["complementarity"].items():
        lines.append(f"| {key} | {item['count']} | {item['rate']} |")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphfp-jsonl", type=Path, required=True)
    parser.add_argument("--dual-rows-jsonl", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--tgt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows-jsonl", type=Path)
    parser.add_argument("--limit-groups", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
