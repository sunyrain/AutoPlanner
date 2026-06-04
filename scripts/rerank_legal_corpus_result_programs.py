#!/usr/bin/env python
"""Offline rerank legal-corpus result programs with no-label evidence features."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_reaction


SCHEMA_VERSION = "legal_corpus_result_rerank.v1"


def rerank_legal_corpus_results(
    *,
    route_run: Path,
    output_json: Path,
    output_md: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(route_run).read_text(encoding="utf-8"))
    rows = [_rerank_target(row) for row in payload.get("targets") or []]
    out = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "route_run": str(route_run),
        "contract": (
            "Offline no-label rerank of generated result_programs using route metadata only; "
            "GT labels are used only for evaluation."
        ),
        "summary": _summary(rows),
        "targets": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(out), encoding="utf-8")
    return out


def _rerank_target(row: dict[str, Any]) -> dict[str, Any]:
    programs = list(((row.get("cascade_search") or {}).get("result_programs") or []))
    scored = []
    for program in programs:
        score, features = _program_evidence_score(program)
        scored.append((score, int(program.get("rank") or 0), program, features))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    reranked = []
    for new_rank, (score, original_rank, program, features) in enumerate(scored, start=1):
        reranked.append(
            {
                "new_rank": new_rank,
                "original_rank": original_rank,
                "rerank_score": round(float(score), 6),
                "features": features,
                "route_rxns": program.get("route_rxns") or [],
                "exact_reaction_hit_count": int(program.get("exact_reaction_hit_count") or 0),
                "gt_reactant_hit_count": int(program.get("gt_reactant_hit_count") or 0),
                "route_outcome_value": program.get("route_outcome_value"),
                "failure_categories": program.get("failure_categories") or [],
            }
        )
    original_top = programs[0] if programs else {}
    rerank_top = reranked[0] if reranked else {}
    return {
        "target_smiles": row.get("target_smiles"),
        "n_programs": len(programs),
        "original_top_exact": bool(int(original_top.get("exact_reaction_hit_count") or 0) > 0),
        "original_top_reactant": bool(int(original_top.get("gt_reactant_hit_count") or 0) > 0),
        "reranked_top_exact": bool(int(rerank_top.get("exact_reaction_hit_count") or 0) > 0),
        "reranked_top_reactant": bool(int(rerank_top.get("gt_reactant_hit_count") or 0) > 0),
        "any_exact": any(int(program.get("exact_reaction_hit_count") or 0) > 0 for program in programs),
        "any_reactant": any(int(program.get("gt_reactant_hit_count") or 0) > 0 for program in programs),
        "original_first_exact_rank": _first_rank(programs, "exact_reaction_hit_count"),
        "reranked_first_exact_rank": _first_rank(reranked, "exact_reaction_hit_count", rank_key="new_rank"),
        "original_first_reactant_rank": _first_rank(programs, "gt_reactant_hit_count"),
        "reranked_first_reactant_rank": _first_rank(reranked, "gt_reactant_hit_count", rank_key="new_rank"),
        "top_changed": bool(reranked and int(reranked[0].get("original_rank") or 0) != 1),
        "reranked_programs": reranked[:10],
    }


def _program_evidence_score(program: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    rxns = [str(rxn) for rxn in program.get("route_rxns") or [] if rxn]
    steps = list(program.get("route_steps") or [])
    step_count = max(1, len(rxns))
    features = [_reaction_evidence_features(step) for step in steps] if steps else [_reaction_evidence_features({"rxn_smiles": rxn}) for rxn in rxns]
    exact_product_count = sum(int(item["match_type"] == "exact_product") for item in features)
    avg_similarity = sum(float(item["product_similarity"]) for item in features) / max(1, len(features))
    min_similarity = min((float(item["product_similarity"]) for item in features), default=0.0)
    legal_count = sum(int(item["is_legal_corpus_like"]) for item in features)
    failure_count = len(program.get("failure_categories") or [])
    base_score = _float(program.get("score"))
    cost_total = _float(((program.get("cost") or {}).get("total_cost")))
    rerank_score = (
        3.0 * (exact_product_count / step_count)
        + 1.5 * avg_similarity
        + 0.5 * min_similarity
        + 0.2 * (legal_count / step_count)
        + 0.05 * base_score
        - 0.05 * cost_total
        - 0.10 * failure_count
    )
    return rerank_score, {
        "exact_product_fraction": round(exact_product_count / step_count, 6),
        "avg_product_similarity": round(avg_similarity, 6),
        "min_product_similarity": round(min_similarity, 6),
        "legal_corpus_like_fraction": round(legal_count / step_count, 6),
        "failure_count": failure_count,
        "base_score": round(base_score, 6),
        "cost_total": round(cost_total, 6),
        "reaction_features": features,
    }


def _reaction_evidence_features(rxn_smiles: str) -> dict[str, Any]:
    if isinstance(rxn_smiles, dict):
        step = rxn_smiles
        key = str(step.get("corpus_canonical_reaction") or step.get("rxn_smiles") or "")
        match_type = str(step.get("match_type") or "")
        product_similarity = float(step.get("product_similarity") or 0.0)
        is_exact_product = match_type == "exact_product"
        corpus_source = step.get("corpus_source")
    else:
        # Fallback for older artifacts without route_steps evidence.
        key = canonical_reaction(rxn_smiles) or rxn_smiles
        lhs, rhs = key.split(">>", 1) if ">>" in key else (key, "")
        product_similarity = 1.0 if rhs else 0.0
        is_exact_product = bool(rhs)
        corpus_source = None
    return {
        "canonical_reaction": key,
        "match_type": "exact_product" if is_exact_product else ("nearest_product" if product_similarity > 0 else "unknown"),
        "product_similarity": product_similarity,
        "is_legal_corpus_like": bool(corpus_source or is_exact_product or product_similarity > 0),
    }


def _first_rank(rows: list[dict[str, Any]], hit_key: str, *, rank_key: str = "rank") -> int | None:
    for row in rows:
        if int(row.get(hit_key) or 0) > 0:
            return int(row.get(rank_key) or 0) or None
    return None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    status_counts = Counter()
    for row in rows:
        if row.get("reranked_top_exact"):
            status_counts["top_exact"] += 1
        elif row.get("reranked_top_reactant"):
            status_counts["top_reactant_only"] += 1
        elif row.get("any_exact"):
            status_counts["exact_retained_below_top"] += 1
        elif row.get("any_reactant"):
            status_counts["reactant_retained_below_top"] += 1
        else:
            status_counts["no_hit"] += 1
    return {
        "n_targets": n,
        "original_top_exact": sum(1 for row in rows if row.get("original_top_exact")),
        "reranked_top_exact": sum(1 for row in rows if row.get("reranked_top_exact")),
        "original_top_reactant": sum(1 for row in rows if row.get("original_top_reactant")),
        "reranked_top_reactant": sum(1 for row in rows if row.get("reranked_top_reactant")),
        "any_exact": sum(1 for row in rows if row.get("any_exact")),
        "any_reactant": sum(1 for row in rows if row.get("any_reactant")),
        "top_changed": sum(1 for row in rows if row.get("top_changed")),
        "status_counts": dict(status_counts),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Legal Corpus Result Program Rerank",
        "",
        f"created_at: `{payload['created_at']}`",
        "",
        "## Summary",
        "",
        f"- original_top_exact: {summary['original_top_exact']}",
        f"- reranked_top_exact: {summary['reranked_top_exact']}",
        f"- original_top_reactant: {summary['original_top_reactant']}",
        f"- reranked_top_reactant: {summary['reranked_top_reactant']}",
        f"- any_exact: {summary['any_exact']}",
        f"- any_reactant: {summary['any_reactant']}",
        f"- top_changed: {summary['top_changed']}",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((summary.get("status_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Targets", "", "| target | original | reranked | first ranks |", "| --- | --- | --- | --- |"])
    for row in payload["targets"]:
        lines.append(
            "| `{target}` | exact={otx}, react={otr} | exact={rtx}, react={rtr} | exact {oe}->{re}, react {orx}->{rr} |".format(
                target=row.get("target_smiles"),
                otx=row.get("original_top_exact"),
                otr=row.get("original_top_reactant"),
                rtx=row.get("reranked_top_exact"),
                rtr=row.get("reranked_top_reactant"),
                oe=row.get("original_first_exact_rank"),
                re=row.get("reranked_first_exact_rank"),
                orx=row.get("original_first_reactant_rank"),
                rr=row.get("reranked_first_reactant_rank"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    args = ap.parse_args()
    payload = rerank_legal_corpus_results(
        route_run=args.route_run,
        output_json=args.output,
        output_md=args.markdown_output,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
