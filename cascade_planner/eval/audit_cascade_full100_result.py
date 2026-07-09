"""Audit full100 CascadeProgramSearch outputs beyond aggregate solve rates."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOP_KS = (1, 2, 3, 5)


def audit_cascade_full100_result(
    *,
    result_path: Path,
    output_path: Path | None = None,
    md_output: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    rows = payload.get("targets") or []
    if not isinstance(rows, list):
        raise ValueError(f"unsupported result format: {result_path}")

    funnel = _funnel(rows)
    topk = _topk_route_metrics(rows)
    by_domain = _by_domain(rows)
    gap_analysis = _gap_analysis(rows)
    failure_counts = Counter()
    for row in rows:
        failure_counts.update((row.get("cascade_search") or {}).get("failure_categories") or [])

    report = {
        "schema_version": "cascade_full100_audit.v2",
        "result_path": str(result_path),
        "n_targets": len(rows),
        "target_uniqueness": _target_uniqueness(rows),
        "summary": payload.get("summary") or {},
        "funnel": funnel,
        "topk_route_metrics": topk,
        "by_route_domain": by_domain,
        "gap_analysis": gap_analysis,
        "cascade_failure_counts": dict(failure_counts),
        "examples": _examples(rows),
        "diagnosis": _diagnosis(funnel, topk, gap_analysis),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output is not None:
        Path(md_output).parent.mkdir(parents=True, exist_ok=True)
        Path(md_output).write_text(_markdown(report), encoding="utf-8")
    return report


def _funnel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins: Counter[str] = Counter()
    candidate_reactant = 0
    route_reactant = 0
    candidate_exact = 0
    route_exact = 0
    for row in rows:
        rec = row.get("recovery") or {}
        ce = bool(rec.get("candidate_exact_reaction_in_pool"))
        cr = bool(rec.get("candidate_gt_reactant_in_pool"))
        re = bool(rec.get("exact_reaction_in_route_pool"))
        rr = bool(rec.get("gt_reactant_in_route_pool"))
        candidate_reactant += int(cr)
        route_reactant += int(rr)
        candidate_exact += int(ce)
        route_exact += int(re)
        if not cr:
            bins["candidate_generation_miss"] += 1
        elif rr:
            bins["route_contains_gt_reactant"] += 1
        elif ce and not re:
            bins["generated_exact_but_not_route"] += 1
        else:
            bins["generated_reactant_but_not_route"] += 1
    return {
        "bins": dict(bins),
        "candidate_gt_reactant_targets": candidate_reactant,
        "route_gt_reactant_targets": route_reactant,
        "candidate_exact_targets": candidate_exact,
        "route_exact_targets": route_exact,
        "reactant_candidate_to_route_rate": _rate(route_reactant, candidate_reactant),
        "exact_candidate_to_route_rate": _rate(route_exact, candidate_exact),
        "candidate_generation_miss_rate": _rate(bins["candidate_generation_miss"], len(rows)),
        "generated_not_route_rate": _rate(
            bins["generated_reactant_but_not_route"] + bins["generated_exact_but_not_route"],
            len(rows),
        ),
    }


def _target_uniqueness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [str(row.get("target_smiles") or "") for row in rows if row.get("target_smiles")]
    counts = Counter(targets)
    duplicates = {target: count for target, count in sorted(counts.items()) if count > 1}
    return {
        "n_rows": len(rows),
        "n_unique_targets": len(counts),
        "duplicate_row_count": sum(count - 1 for count in counts.values() if count > 1),
        "duplicates": duplicates,
    }


def _topk_route_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    first_reactant_ranks: Counter[str] = Counter()
    for k in TOP_KS:
        counts = Counter()
        for row in rows:
            programs = ((row.get("cascade_search") or {}).get("result_programs") or [])[:k]
            counts["n"] += 1
            counts["exact"] += int(_programs_have_exact(programs))
            counts["reactant"] += int(_programs_have_reactant(programs))
            counts["partial"] += int(_programs_have_partial(programs))
        out[f"top{k}"] = {
            "exact_rate": _rate(counts["exact"], counts["n"]),
            "gt_reactant_rate": _rate(counts["reactant"], counts["n"]),
            "partial_gt_step_rate": _rate(counts["partial"], counts["n"]),
            "exact_count": counts["exact"],
            "gt_reactant_count": counts["reactant"],
            "partial_gt_step_count": counts["partial"],
        }
    for row in rows:
        rank = _first_reactant_rank((row.get("cascade_search") or {}).get("result_programs") or [])
        first_reactant_ranks[str(rank) if rank is not None else "none"] += 1
    out["first_gt_reactant_rank_distribution"] = dict(first_reactant_ranks)
    top1 = out["top1"]["gt_reactant_count"]
    top5 = out["top5"]["gt_reactant_count"]
    out["top5_oracle_rerank_gain_over_top1"] = top5 - top1
    return out


def _gap_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket_rows[_funnel_bucket(row)].append(row)

    out: dict[str, Any] = {
        "by_bucket": {bucket: _gap_bucket_summary(items) for bucket, items in sorted(bucket_rows.items())},
        "bucket_by_domain": _bucket_table(rows, "route_domain"),
        "bucket_by_depth": _bucket_table(rows, "depth"),
        "gt_transformation_by_bucket": {
            bucket: dict(_gt_transformation_counts(items).most_common())
            for bucket, items in sorted(bucket_rows.items())
        },
    }
    out["early_closure"] = _early_closure_summary(rows)
    out["selection_headroom"] = _selection_headroom(rows)
    return out


def _funnel_bucket(row: dict[str, Any]) -> str:
    rec = row.get("recovery") or {}
    ce = bool(rec.get("candidate_exact_reaction_in_pool"))
    cr = bool(rec.get("candidate_gt_reactant_in_pool"))
    re = bool(rec.get("exact_reaction_in_route_pool"))
    rr = bool(rec.get("gt_reactant_in_route_pool"))
    if not cr:
        return "candidate_generation_miss"
    if rr:
        return "route_contains_gt_reactant"
    if ce and not re:
        return "generated_exact_but_not_route"
    return "generated_reactant_but_not_route"


def _gap_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "avg_depth": _avg(_numeric(row.get("depth")) for row in rows),
        "avg_result_step_count": _avg(_numeric((row.get("cascade_search") or {}).get("step_count")) for row in rows),
        "avg_expansions": _avg(_numeric(((row.get("cascade_search") or {}).get("stats") or {}).get("expansions")) for row in rows),
        "avg_generated_actions": _avg(_numeric(((row.get("cascade_search") or {}).get("stats") or {}).get("generated_actions")) for row in rows),
        "avg_chem_enzy_routes": _avg(_numeric((row.get("chem_enzy") or {}).get("route_count")) for row in rows),
        "avg_proposal_steps": _avg(_numeric((row.get("chem_enzy") or {}).get("proposal_step_count")) for row in rows),
        "avg_proposal_reactions": _avg(_numeric((row.get("recovery") or {}).get("proposal_pool_reaction_count")) for row in rows),
        "avg_proposal_reactants": _avg(_numeric((row.get("recovery") or {}).get("proposal_pool_reactant_count")) for row in rows),
        "solved_rate": _mean_bool(rows, lambda row: (row.get("cascade_search") or {}).get("solved")),
        "result_limit_stop_rate": _mean_bool(
            rows,
            lambda row: (((row.get("cascade_search") or {}).get("stats") or {}).get("stop_reason") == "result_limit"),
        ),
    }


def _bucket_table(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[str(row.get(key) if row.get(key) is not None else "unknown")][_funnel_bucket(row)] += 1
    return {name: dict(counts) for name, counts in sorted(table.items())}


def _gt_transformation_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for step in row.get("gt_route") or []:
            counts[str(step.get("transformation") or "unknown")] += 1
    return counts


def _early_closure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    early = []
    for row in rows:
        depth = _numeric(row.get("depth"))
        step_count = _numeric((row.get("cascade_search") or {}).get("step_count"))
        if depth is None or step_count is None:
            continue
        if step_count < depth:
            early.append(row)
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for row in early:
        by_bucket[_funnel_bucket(row)]["n"] += 1
    return {
        "n_result_shorter_than_gt_depth": len(early),
        "rate_result_shorter_than_gt_depth": _rate(len(early), len(rows)),
        "by_bucket": {bucket: dict(counts) for bucket, counts in sorted(by_bucket.items())},
        "avg_depth_gap_when_shorter": _avg(
            (_numeric(row.get("depth")) or 0.0) - (_numeric((row.get("cascade_search") or {}).get("step_count")) or 0.0)
            for row in early
        ),
    }


def _selection_headroom(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated_not_route = [
        row
        for row in rows
        if _funnel_bucket(row) in {"generated_reactant_but_not_route", "generated_exact_but_not_route"}
    ]
    topk = _topk_route_metrics(generated_not_route) if generated_not_route else {}
    exact_generated = [row for row in rows if _funnel_bucket(row) == "generated_exact_but_not_route"]
    return {
        "generated_not_route_targets": len(generated_not_route),
        "generated_exact_not_route_targets": len(exact_generated),
        "top5_oracle_gt_reactant_targets_in_generated_not_route": (
            (topk.get("top5") or {}).get("gt_reactant_count") if topk else 0
        ),
        "top5_oracle_exact_targets_in_generated_not_route": (
            (topk.get("top5") or {}).get("exact_count") if topk else 0
        ),
        "first_gt_reactant_rank_distribution_in_generated_not_route": (
            topk.get("first_gt_reactant_rank_distribution") if topk else {}
        ),
    }


def _by_domain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("route_domain") or "unknown")].append(row)
    out = {}
    for domain, items in sorted(buckets.items()):
        out[domain] = {
            "n": len(items),
            "cascade_solved_rate": _mean_bool(items, lambda row: (row.get("cascade_search") or {}).get("solved")),
            "candidate_exact_rate": _mean_bool(items, lambda row: (row.get("recovery") or {}).get("candidate_exact_reaction_in_pool")),
            "candidate_gt_reactant_rate": _mean_bool(items, lambda row: (row.get("recovery") or {}).get("candidate_gt_reactant_in_pool")),
            "route_exact_rate": _mean_bool(items, lambda row: (row.get("recovery") or {}).get("exact_reaction_in_route_pool")),
            "route_gt_reactant_rate": _mean_bool(items, lambda row: (row.get("recovery") or {}).get("gt_reactant_in_route_pool")),
            "funnel": _funnel(items),
            "topk_route_metrics": _topk_route_metrics(items),
        }
    return out


def _examples(rows: list[dict[str, Any]], limit: int = 8) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rec = row.get("recovery") or {}
        bucket = _funnel_bucket(row)
        if len(examples[bucket]) >= limit:
            continue
        examples[bucket].append(
            {
                "target_smiles": row.get("target_smiles"),
                "route_domain": row.get("route_domain"),
                "recovery": {
                    key: rec.get(key)
                    for key in (
                        "candidate_exact_reaction_in_pool",
                        "candidate_gt_reactant_in_pool",
                        "exact_reaction_in_route_pool",
                        "gt_reactant_in_route_pool",
                        "partial_gt_step_overlap",
                        "gt_step_overlap_fraction",
                    )
                },
            }
        )
    return dict(examples)


def _diagnosis(funnel: dict[str, Any], topk: dict[str, Any], gap_analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    candidate_miss = int((funnel.get("bins") or {}).get("candidate_generation_miss") or 0)
    generated_not_route = (
        int((funnel.get("bins") or {}).get("generated_reactant_but_not_route") or 0)
        + int((funnel.get("bins") or {}).get("generated_exact_but_not_route") or 0)
    )
    rerank_gain = int(topk.get("top5_oracle_rerank_gain_over_top1") or 0)
    if candidate_miss > generated_not_route:
        bottleneck = "proposal_generation_coverage"
    elif rerank_gain > 0:
        bottleneck = "route_ranking_and_search_priority"
    else:
        bottleneck = "mixed_or_low_oracle_headroom"
    next_actions = []
    if bottleneck == "proposal_generation_coverage":
        next_actions.append("Run a trace-enabled miss-subset benchmark to locate leaf/source/rank-level proposal misses.")
        next_actions.append("Add a retrieval or realtime ChemEnzy internal proposal provider for candidate-generation-miss targets.")
    if generated_not_route:
        next_actions.append("Use generated-but-not-route targets for action/value reranking ablations; candidates exist but are underselected.")
    early = ((gap_analysis or {}).get("early_closure") or {}).get("n_result_shorter_than_gt_depth") or 0
    if early:
        next_actions.append("Audit early stock closure and result_limit behavior; direct closure may hide multi-step GT fragments.")
    return {
        "primary_bottleneck": bottleneck,
        "candidate_generation_miss_targets": candidate_miss,
        "generated_but_not_route_targets": generated_not_route,
        "top5_oracle_rerank_gain_over_top1": rerank_gain,
        "next_actions": next_actions,
        "recommendation": (
            "Improve proposal/source coverage first, while using value/ranking models for the generated-but-not-route subset."
            if bottleneck == "proposal_generation_coverage"
            else "Prioritize state/action value and route reranking; useful candidates are present but underselected."
        ),
    }


def _programs_have_exact(programs: list[dict[str, Any]]) -> bool:
    return any((program.get("exact_reaction_hit_count") or 0) > 0 or program.get("exact_gt_route_recovered") for program in programs)


def _programs_have_reactant(programs: list[dict[str, Any]]) -> bool:
    return any((program.get("gt_reactant_hit_count") or 0) > 0 or program.get("gt_reactant_in_route") for program in programs)


def _programs_have_partial(programs: list[dict[str, Any]]) -> bool:
    return any(program.get("partial_gt_step_overlap") for program in programs)


def _first_reactant_rank(programs: list[dict[str, Any]]) -> int | None:
    for idx, program in enumerate(programs, start=1):
        if (program.get("gt_reactant_hit_count") or 0) > 0 or program.get("gt_reactant_in_route"):
            return idx
    return None


def _mean_bool(rows: list[dict[str, Any]], fn) -> float:
    return _rate(sum(1 for row in rows if fn(row)), len(rows))


def _rate(num: int, den: int) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: Any) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return round(sum(rows) / len(rows), 6) if rows else None


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cascade Full100 Audit",
        "",
        f"- targets: `{report['n_targets']}`",
        f"- unique targets: `{report['target_uniqueness']['n_unique_targets']}`",
        f"- primary bottleneck: `{report['diagnosis']['primary_bottleneck']}`",
        f"- candidate generation misses: `{report['diagnosis']['candidate_generation_miss_targets']}`",
        f"- generated but not routed: `{report['diagnosis']['generated_but_not_route_targets']}`",
        f"- top5 oracle rerank gain over top1: `{report['diagnosis']['top5_oracle_rerank_gain_over_top1']}`",
        "",
        "## Funnel",
        "",
    ]
    for key, value in report["funnel"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Top-K Route Metrics", ""])
    for key, value in report["topk_route_metrics"].items():
        if key.startswith("top"):
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Domain Summary", ""])
    for domain, value in report["by_route_domain"].items():
        lines.append(
            f"- {domain}: n=`{value['n']}`, candidate_gt_reactant=`{value['candidate_gt_reactant_rate']}`, "
            f"route_gt_reactant=`{value['route_gt_reactant_rate']}`"
        )
    lines.extend(["", "## Gap Analysis", ""])
    early = report.get("gap_analysis", {}).get("early_closure", {})
    lines.append(
        f"- result shorter than GT depth: `{early.get('n_result_shorter_than_gt_depth')}` "
        f"(`{early.get('rate_result_shorter_than_gt_depth')}`)"
    )
    lines.append(
        f"- avg depth gap when shorter: `{early.get('avg_depth_gap_when_shorter')}`"
    )
    lines.extend(["", "### Buckets By Depth", ""])
    for depth, counts in (report.get("gap_analysis", {}).get("bucket_by_depth") or {}).items():
        lines.append(f"- depth `{depth}`: `{counts}`")
    lines.extend(["", "### Buckets By Domain", ""])
    for domain, counts in (report.get("gap_analysis", {}).get("bucket_by_domain") or {}).items():
        lines.append(f"- {domain}: `{counts}`")
    lines.extend(["", "## Recommendation", "", report["diagnosis"]["recommendation"], ""])
    for action in report["diagnosis"].get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit full100 cascade result funnels and top-k route headroom")
    ap.add_argument("--result", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--md-output", default=None)
    args = ap.parse_args()
    report = audit_cascade_full100_result(
        result_path=Path(args.result),
        output_path=Path(args.output) if args.output else None,
        md_output=Path(args.md_output) if args.md_output else None,
    )
    print(json.dumps({
        "n_targets": report["n_targets"],
        "funnel": report["funnel"],
        "diagnosis": report["diagnosis"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
