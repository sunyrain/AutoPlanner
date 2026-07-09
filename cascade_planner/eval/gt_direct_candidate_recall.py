"""Direct GT-product single-step candidate recall audit.

This diagnostic asks a different question from the live route benchmark:
if the current live candidate layer is queried on the ground-truth product or
intermediate for each GT step, does the GT reaction/reactant set enter the
candidate pool?

That separates two failure modes:
  1. route search never visits the GT intermediate/product slot
  2. the single-step generator is queried on the right molecule but still misses
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import RDLogger

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_side,
    candidate_reactants as exported_candidate_reactants,
    reaction_reactants,
)
from cascade_planner.cascadeboard.skeleton_planner import _candidates_for_skeleton_slot

RDLogger.DisableLog("rdApp.*")
logging.disable(logging.CRITICAL)


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def _ec1(ec: str | None) -> int:
    if not ec:
        return 0
    first = str(ec).split(".", 1)[0]
    return int(first) if first.isdigit() else 0


def _rxn_product(rxn_smiles: str | None) -> str:
    rxn = canonical_reaction(rxn_smiles)
    if not rxn or ">>" not in rxn:
        return ""
    return ".".join(canonical_side(rxn.split(">>", 1)[1]))


def _candidate_rxn(candidate: dict[str, Any]) -> str:
    return canonical_reaction(candidate.get("reaction_smiles") or candidate.get("rxn_smiles"))


def _candidate_reactants(candidate: dict[str, Any]) -> set[str]:
    out = set(exported_candidate_reactants(candidate))
    out.update(reaction_reactants(candidate.get("rxn_smiles")))
    return {x for x in out if x}


def _candidate_source(candidate: dict[str, Any]) -> str:
    return candidate.get("source") or candidate.get("enzyme_source") or "unknown"


def _records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        rows = data.get("records_kept", data.get("records", []))
        return rows if isinstance(rows, list) else []
    return data if isinstance(data, list) else []


def _iter_gt_steps(data: Any, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for target_index, entry in enumerate(_records(data)):
        if limit is not None and target_index >= limit:
            break
        for step_index, step in enumerate(entry.get("gt_route") or [], 1):
            gt_rxn = canonical_reaction(step.get("rxn_smiles"))
            if not gt_rxn:
                continue
            ec = step.get("ec_number") or ""
            ec1 = _ec1(ec)
            rows.append({
                "target_index": target_index,
                "target_smiles": entry.get("target_smiles"),
                "route_domain": entry.get("route_domain") or "unknown",
                "doi": entry.get("doi"),
                "cascade_id": entry.get("cascade_id"),
                "gt_step_index": step_index,
                "gt_rxn": gt_rxn,
                "gt_product": _rxn_product(gt_rxn),
                "gt_reactants": reaction_reactants(gt_rxn),
                "gt_type": step.get("transformation") or "",
                "gt_ec": ec,
                "gt_ec1": ec1,
                "step_kind": "enzymatic" if ec1 > 0 else "chemical",
            })
    return rows


def _build_retro_engine(step_kind: str) -> dict[str, Any]:
    """Build only the needed live candidate sources for this audit."""
    if step_kind == "chemical":
        from cascade_planner.cascadeboard.live_retro import _RetroChimeraWrapper

        engine: dict[str, Any] = {"retrochimera": _RetroChimeraWrapper()}
        try:
            from cascade_planner.cascadeboard.chemical_template_applicator import (
                ChemicalTemplateApplicator,
                chemical_templates_enabled,
            )

            if chemical_templates_enabled():
                app = ChemicalTemplateApplicator.from_env()
                if app.available:
                    engine["chemtemplates"] = app
        except Exception:
            pass
        return engine

    from cascade_planner.cascadeboard.live_retro import build_live_retro_engine

    return build_live_retro_engine()


def audit_step(
    retro_engine: dict[str, Any],
    gt_step: dict[str, Any],
    *,
    top_k: int,
    proposal_mode: str = "skeleton",
) -> dict[str, Any]:
    if proposal_mode == "route_tree":
        from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool

        tool = RetroEngineProposalTool(retro_engine)
        actions = tool.propose(
            gt_step["gt_product"],
            ProposalContext(ec1=gt_step["gt_ec1"], reaction_type=gt_step["gt_type"]),
            top_k=top_k,
        )
        candidates = [action.to_candidate_dict() for action in actions]
    else:
        candidates = _candidates_for_skeleton_slot(
            retro_engine,
            product_smiles=gt_step["gt_product"],
            ec1=gt_step["gt_ec1"],
            skel_type=gt_step["gt_type"],
            top_k=top_k,
        )
    exact_hits = []
    exact_reactant_hits = []
    any_reactant_hits = []
    for rank, cand in enumerate(candidates, 1):
        cand_rxn = _candidate_rxn(cand)
        cand_reactants = _candidate_reactants(cand)
        row = {
            "rank": rank,
            "source": _candidate_source(cand),
            "rxn": cand_rxn,
            "reactants": cand_reactants,
        }
        if cand_rxn == gt_step["gt_rxn"]:
            exact_hits.append(row)
        if cand_reactants == gt_step["gt_reactants"] and gt_step["gt_reactants"]:
            exact_reactant_hits.append(row)
        if cand_reactants & gt_step["gt_reactants"]:
            any_reactant_hits.append(row)

    labels = []
    if not candidates:
        labels.append("no_candidates")
    if not exact_hits:
        labels.append("direct_exact_miss")
    if exact_reactant_hits and not exact_hits:
        labels.append("reactant_set_present_exact_missing")
    elif any_reactant_hits and not exact_hits:
        labels.append("reactant_fragment_present_exact_missing")
    if not labels:
        labels.append("direct_exact_hit")

    source_counts = Counter(_candidate_source(cand) for cand in candidates)
    exact_sources = Counter(row["source"] for row in exact_hits)
    reactant_sources = Counter(row["source"] for row in exact_reactant_hits or any_reactant_hits)

    return {
        **{k: v for k, v in gt_step.items() if k != "gt_reactants"},
        "n_candidates": len(candidates),
        "direct_exact_hit": bool(exact_hits),
        "direct_exact_reactant_set_hit": bool(exact_reactant_hits),
        "direct_any_gt_reactant_hit": bool(any_reactant_hits),
        "best_exact_rank": min((row["rank"] for row in exact_hits), default=None),
        "best_reactant_rank": min((row["rank"] for row in exact_reactant_hits or any_reactant_hits), default=None),
        "candidate_source_counts": dict(source_counts),
        "exact_candidate_sources": dict(exact_sources),
        "reactant_hit_sources": dict(reactant_sources),
        "labels": labels,
    }


def _summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    labels = Counter(label for row in rows for label in row["labels"])
    source_counts = Counter()
    exact_sources = Counter()
    reactant_sources = Counter()
    exact_ranks = []
    reactant_ranks = []
    for row in rows:
        source_counts.update(row.get("candidate_source_counts") or {})
        exact_sources.update(row.get("exact_candidate_sources") or {})
        reactant_sources.update(row.get("reactant_hit_sources") or {})
        if row.get("best_exact_rank") is not None:
            exact_ranks.append(float(row["best_exact_rank"]))
        if row.get("best_reactant_rank") is not None:
            reactant_ranks.append(float(row["best_reactant_rank"]))
    return {
        "n_steps": n,
        "direct_exact_hit": sum(bool(row["direct_exact_hit"]) for row in rows),
        "direct_exact_reactant_set_hit": sum(bool(row["direct_exact_reactant_set_hit"]) for row in rows),
        "direct_any_gt_reactant_hit": sum(bool(row["direct_any_gt_reactant_hit"]) for row in rows),
        "avg_candidates_per_step": round(sum(int(row["n_candidates"]) for row in rows) / max(n, 1), 3),
        "avg_best_exact_rank": round(sum(exact_ranks) / len(exact_ranks), 3) if exact_ranks else None,
        "avg_best_reactant_rank": round(sum(reactant_ranks) / len(reactant_ranks), 3) if reactant_ranks else None,
        "label_counts": dict(labels),
        "candidate_source_counts": dict(source_counts),
        "exact_candidate_sources": dict(exact_sources),
        "reactant_hit_sources": dict(reactant_sources),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = defaultdict(list)
    by_domain = defaultdict(list)
    by_type = defaultdict(list)
    for row in rows:
        by_kind[row["step_kind"]].append(row)
        by_domain[row["route_domain"]].append(row)
        by_type[row["gt_type"] or "unknown"].append(row)
    miss_examples = [row for row in rows if "direct_exact_miss" in row["labels"]][:25]
    return {
        "overall": _summarize_subset(rows),
        "by_step_kind": {k: _summarize_subset(v) for k, v in sorted(by_kind.items())},
        "by_domain": {k: _summarize_subset(v) for k, v in sorted(by_domain.items())},
        "by_transformation": {k: _summarize_subset(v) for k, v in sorted(by_type.items())},
        "step_rows": rows,
        "miss_examples": miss_examples,
    }


def audit_gt_direct_recall(
    bench_path: str,
    *,
    top_k: int,
    limit: int | None = None,
    step_kind: str = "all",
    proposal_mode: str = "skeleton",
) -> dict[str, Any]:
    data = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    gt_steps = _iter_gt_steps(data, limit=limit)
    if step_kind != "all":
        gt_steps = [row for row in gt_steps if row["step_kind"] == step_kind]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retro_engine = _build_retro_engine(step_kind)
        rows = [audit_step(retro_engine, gt, top_k=top_k, proposal_mode=proposal_mode) for gt in gt_steps]
    summary = summarize_rows(rows)
    summary["metadata"] = {
        "benchmark": bench_path,
        "top_k": top_k,
        "limit": limit,
        "step_kind": step_kind,
        "proposal_mode": proposal_mode,
        "note": (
            "GT-direct recall queries the live candidate layer on each GT "
            "product/intermediate, bypassing route traversal."
        ),
    }
    return summary


def _rate(summary: dict[str, Any], key: str) -> str:
    return _pct(int(summary.get(key) or 0), int(summary.get("n_steps") or 0))


def _label_rate(summary: dict[str, Any], label: str) -> str:
    labels = summary.get("label_counts") or {}
    return _pct(int(labels.get(label) or 0), int(summary.get("n_steps") or 0))


def _write_group_table(lines: list[str], title: str, rows: dict[str, dict[str, Any]]) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "| Group | Steps | Exact cand | Reactant set cand | Any reactant cand | Direct exact miss | Avg candidates | Avg exact rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, row in rows.items():
        lines.append(
            f"| `{key}` | {row.get('n_steps', 0)} | "
            f"{_rate(row, 'direct_exact_hit')} | "
            f"{_rate(row, 'direct_exact_reactant_set_hit')} | "
            f"{_rate(row, 'direct_any_gt_reactant_hit')} | "
            f"{_label_rate(row, 'direct_exact_miss')} | "
            f"{row.get('avg_candidates_per_step')} | "
            f"{row.get('avg_best_exact_rank')} |"
        )


def write_markdown(summary: dict[str, Any], output_path: str) -> None:
    md = summary.get("metadata") or {}
    overall = summary["overall"]
    lines = [
        "# GT-Direct Candidate Recall",
        "",
        f"Benchmark: `{md.get('benchmark')}`",
        f"top_k: `{md.get('top_k')}`",
        f"step_kind: `{md.get('step_kind')}`",
        f"proposal_mode: `{md.get('proposal_mode', 'skeleton')}`",
        "",
        "This audit queries the live candidate layer directly on each GT product/intermediate.",
        "",
        "## Overall",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| GT steps queried | {overall['n_steps']} | 100.0% |",
        f"| Exact candidate hit | {overall['direct_exact_hit']} | {_rate(overall, 'direct_exact_hit')} |",
        f"| Exact reactant-set candidate hit | {overall['direct_exact_reactant_set_hit']} | {_rate(overall, 'direct_exact_reactant_set_hit')} |",
        f"| Any GT-reactant candidate hit | {overall['direct_any_gt_reactant_hit']} | {_rate(overall, 'direct_any_gt_reactant_hit')} |",
        f"| Direct exact miss | {(overall.get('label_counts') or {}).get('direct_exact_miss', 0)} | {_label_rate(overall, 'direct_exact_miss')} |",
        f"| Avg candidates per step | {overall['avg_candidates_per_step']} | n/a |",
        f"| Avg best exact rank | {overall['avg_best_exact_rank']} | n/a |",
    ]
    _write_group_table(lines, "By Step Kind", summary.get("by_step_kind") or {})
    _write_group_table(lines, "By Domain", summary.get("by_domain") or {})

    lines.extend([
        "",
        "## Candidate Sources",
        "",
        "| Source | All candidates | Exact hits | Reactant hits |",
        "|---|---:|---:|---:|",
    ])
    sources = set((overall.get("candidate_source_counts") or {}).keys())
    sources.update((overall.get("exact_candidate_sources") or {}).keys())
    sources.update((overall.get("reactant_hit_sources") or {}).keys())
    for source in sorted(sources):
        lines.append(
            f"| `{source}` | "
            f"{(overall.get('candidate_source_counts') or {}).get(source, 0)} | "
            f"{(overall.get('exact_candidate_sources') or {}).get(source, 0)} | "
            f"{(overall.get('reactant_hit_sources') or {}).get(source, 0)} |"
        )

    lines.extend([
        "",
        "## First Direct Miss Examples",
        "",
        "| Target | Domain | Step | Kind | Type | EC1 | Candidates | Labels |",
        "|---|---|---:|---|---|---:|---:|---|",
    ])
    for row in summary.get("miss_examples") or []:
        target = row.get("target_smiles") or ""
        if len(target) > 32:
            target = target[:29] + "..."
        lines.append(
            f"| `{target}` | `{row.get('route_domain')}` | {row.get('gt_step_index')} | "
            f"`{row.get('step_kind')}` | `{row.get('gt_type')}` | "
            f"{row.get('gt_ec1') or 0} | {row.get('n_candidates')} | "
            f"`{', '.join(row.get('labels') or [])}` |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit live candidate recall when queried on GT products")
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--output", default="results/v2/gt_direct_candidate_recall.md")
    ap.add_argument("--json-output", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--step-kind", choices=["all", "chemical", "enzymatic"], default="all")
    ap.add_argument("--proposal-mode", choices=["skeleton", "route_tree"], default="skeleton")
    args = ap.parse_args()

    summary = audit_gt_direct_recall(
        args.bench,
        top_k=args.top_k,
        limit=args.limit,
        step_kind=args.step_kind,
        proposal_mode=args.proposal_mode,
    )
    write_markdown(summary, args.output)
    json_output = args.json_output or str(Path(args.output).with_suffix(".json"))
    Path(json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(json_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "json_output": json_output,
        "overall": summary["overall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
