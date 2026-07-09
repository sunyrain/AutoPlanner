"""Audit skeleton retrieval priors with optional exact-target exclusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from cascade_planner.cascadeboard.skeleton_retrieval_prior import (
    DEFAULT_SKELETON_PRIOR_PATH,
    retrieve_skeleton_priors,
)


def audit_skeleton_retrieval_prior(
    *,
    bench_path: str | Path,
    prior_path: str | Path = DEFAULT_SKELETON_PRIOR_PATH,
    output_path: str | Path | None = None,
    limit: int = 5,
    min_similarity: float = 0.85,
    exclude_exact_target: bool = False,
) -> dict[str, Any]:
    bench = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    target_rows = []
    hit1 = 0
    hit5 = 0
    available = 0
    similarities = []
    for idx, entry in enumerate(bench):
        target = entry["target_smiles"]
        depth = int(entry.get("depth") or len(entry.get("gt_route", [])) or 3)
        domain = entry.get("route_domain") or "chemoenzymatic"
        gt_types = [step.get("transformation", "") for step in entry.get("gt_route", [])]
        priors = retrieve_skeleton_priors(
            target,
            depth=depth,
            domain=domain,
            pack_path=prior_path,
            limit=limit,
            min_similarity=min_similarity,
            exclude_exact_target=exclude_exact_target,
        )
        type_sequences = [row.get("type_sequence") or [] for row in priors]
        if priors:
            available += 1
            similarities.append(float(priors[0].get("similarity") or 0.0))
        top1_hit = bool(type_sequences and type_sequences[0] == gt_types)
        top5_hit = any(seq == gt_types for seq in type_sequences[:5])
        hit1 += int(top1_hit)
        hit5 += int(top5_hit)
        target_rows.append({
            "index": idx,
            "target_smiles": target,
            "depth": depth,
            "route_domain": domain,
            "gt_types": gt_types,
            "n_priors": len(priors),
            "top_similarity": priors[0].get("similarity") if priors else None,
            "top1_types": type_sequences[0] if type_sequences else None,
            "exact_type_hit1": top1_hit,
            "exact_type_hit5": top5_hit,
        })

    n = len(bench) or 1
    summary = {
        "n_targets": len(bench),
        "prior_available": available,
        "prior_available_rate": available / n,
        "exact_type_hit1": hit1,
        "exact_type_hit1_rate": hit1 / n,
        "exact_type_hit5": hit5,
        "exact_type_hit5_rate": hit5 / n,
        "avg_top_similarity": round(mean(similarities), 3) if similarities else None,
        "exclude_exact_target": exclude_exact_target,
        "min_similarity": min_similarity,
        "limit": limit,
        "bench_path": str(bench_path),
        "prior_path": str(prior_path),
    }
    output = {"summary": summary, "targets": target_rows}
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        md_path = out.with_suffix(".md")
        md_path.write_text(_render_markdown(summary), encoding="utf-8")
    return output


def _render_markdown(summary: dict[str, Any]) -> str:
    return "\n".join([
        "# Skeleton Retrieval Prior Audit",
        "",
        f"- benchmark: `{summary['bench_path']}`",
        f"- prior pack: `{summary['prior_path']}`",
        f"- exclude exact target: `{summary['exclude_exact_target']}`",
        f"- min similarity: `{summary['min_similarity']}`",
        f"- targets: `{summary['n_targets']}`",
        f"- prior available: `{summary['prior_available']}` (`{summary['prior_available_rate']:.3f}`)",
        f"- exact type hit@1: `{summary['exact_type_hit1']}` (`{summary['exact_type_hit1_rate']:.3f}`)",
        f"- exact type hit@5: `{summary['exact_type_hit5']}` (`{summary['exact_type_hit5_rate']:.3f}`)",
        f"- avg top similarity: `{summary['avg_top_similarity']}`",
        "",
        "Note: exact-target exclusion is the minimum leakage guard. A stronger blind split should also exclude close scaffold clusters and duplicate literature sources.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="data/benchmark_condition_rich_20260507.json")
    parser.add_argument("--prior", default=str(DEFAULT_SKELETON_PRIOR_PATH))
    parser.add_argument("--output", default="results/v2/skeleton_retrieval_prior_audit.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.85)
    parser.add_argument("--exclude-exact-target", action="store_true")
    args = parser.parse_args()

    result = audit_skeleton_retrieval_prior(
        bench_path=args.bench,
        prior_path=args.prior,
        output_path=args.output,
        limit=args.limit,
        min_similarity=args.min_similarity,
        exclude_exact_target=args.exclude_exact_target,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
