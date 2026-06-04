#!/usr/bin/env python
"""Audit top-level context-ONMT proposals against benchmark GT reactions."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascade_search import (  # noqa: E402
    CascadeSearchState,
    ChemEnzyContextONMTProposalProvider,
    ProposalRequest,
)
from cascade_planner.cascadeboard.route_recovery import (  # noqa: E402
    canonical_reaction,
    canonical_side,
    gt_reactants,
    gt_reaction_keys,
    reaction_reactants,
)


SCHEMA_VERSION = "context_onmt_top_level_proposal_audit.v1"


def audit_top_level_proposals(
    *,
    benchmark_path: Path,
    model_path: Path,
    output_json: Path,
    output_md: Path | None = None,
    vendor_root: Path = Path("vendor/ChemEnzyRetroPlanner"),
    limit: int | None = None,
    topk: int = 8,
    beam_size: int = 8,
    batch_size: int = 16,
    min_score: float = 0.0,
    device: int = -1,
    preference_scorer_path: Path | None = None,
    preference_min_score: float | None = None,
    preference_rerank: bool = False,
    raw_topk_multiplier: int = 3,
    filter_invalid_proposals: bool = True,
    provider: Any | None = None,
) -> dict[str, Any]:
    entries = _load_benchmark(benchmark_path, limit=limit)
    provider = provider or ChemEnzyContextONMTProposalProvider(
        model_path=model_path,
        vendor_root=vendor_root,
        topk=topk,
        beam_size=beam_size,
        batch_size=batch_size,
        min_score=min_score,
        device=device,
        max_context_step=1,
        preference_scorer_path=preference_scorer_path,
        preference_min_score=preference_min_score,
        preference_rerank=preference_rerank,
        raw_topk_multiplier=raw_topk_multiplier,
        filter_invalid_proposals=filter_invalid_proposals,
    )
    rows = [_audit_entry(entry, provider=provider, topk=topk) for entry in entries]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "benchmark": str(benchmark_path),
        "model_path": str(model_path),
        "settings": {
            "limit": limit,
            "topk": topk,
            "beam_size": beam_size,
            "batch_size": batch_size,
            "min_score": min_score,
            "device": device,
            "preference_scorer_path": str(preference_scorer_path) if preference_scorer_path else None,
            "preference_min_score": preference_min_score,
            "preference_rerank": preference_rerank,
            "raw_topk_multiplier": raw_topk_multiplier,
            "filter_invalid_proposals": filter_invalid_proposals,
        },
        "summary": _summary(rows),
        "targets": rows,
        "decision": _decision(rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _audit_entry(entry: dict[str, Any], *, provider: Any, topk: int) -> dict[str, Any]:
    target = str(entry.get("target_smiles") or entry.get("smiles") or "")
    state = CascadeSearchState.initial(target)
    actions = provider.propose(ProposalRequest(target, state, top_k=topk))
    diagnostics = getattr(provider, "last_diagnostics", None)
    gt_rxns = set(gt_reaction_keys(entry))
    target_step_gt_rxns = _target_product_gt_reactions(entry, target)
    gt_reactant_set = gt_reactants(entry)
    proposal_rows = []
    for rank, action in enumerate(actions, 1):
        step = getattr(action, "step", None)
        rxn = str(getattr(step, "rxn_smiles", "") or "")
        rxn_key = canonical_reaction(rxn)
        reactant_set = reaction_reactants(rxn)
        for smi in getattr(step, "reactant_smiles", []) or []:
            reactant_set.update(canonical_side(str(smi)))
        reactant_hits = sorted(reactant_set & gt_reactant_set)
        proposal_rows.append(
            {
                "rank": rank,
                "source": str(getattr(action, "source", "") or getattr(step, "source_model", "") or ""),
                "score": getattr(step, "score", None),
                "preference_score": (getattr(step, "raw_metadata", {}) or {}).get("preference_score"),
                "preference_rank": (getattr(step, "raw_metadata", {}) or {}).get("preference_rank"),
                "match_type": (getattr(step, "raw_metadata", {}) or {}).get("match_type"),
                "product_similarity": (getattr(step, "raw_metadata", {}) or {}).get("product_similarity"),
                "corpus_source": (getattr(step, "raw_metadata", {}) or {}).get("corpus_source"),
                "corpus_source_row_id": (getattr(step, "raw_metadata", {}) or {}).get("corpus_source_row_id"),
                "corpus_reaction_smiles": (getattr(step, "raw_metadata", {}) or {}).get("corpus_reaction_smiles"),
                "rxn_smiles": rxn,
                "canonical_reaction": rxn_key,
                "reactants": sorted(reactant_set),
                "exact_gt_reaction_hit": bool(rxn_key and rxn_key in gt_rxns),
                "target_step_gt_reaction_hit": bool(rxn_key and rxn_key in target_step_gt_rxns),
                "gt_reactant_hits": reactant_hits,
                "gt_reactant_hit": bool(reactant_hits),
            }
        )
    exact_rank = _first_rank(proposal_rows, "exact_gt_reaction_hit")
    target_step_rank = _first_rank(proposal_rows, "target_step_gt_reaction_hit")
    reactant_rank = _first_rank(proposal_rows, "gt_reactant_hit")
    return {
        "target_smiles": target,
        "route_domain": entry.get("route_domain"),
        "depth": entry.get("depth"),
        "gt_n_reactions": len(gt_rxns),
        "gt_n_target_step_reactions": len(target_step_gt_rxns),
        "gt_n_reactants": len(gt_reactant_set),
        "returned": len(proposal_rows),
        "exact_gt_reaction_hit": exact_rank is not None,
        "exact_gt_reaction_best_rank": exact_rank,
        "target_step_gt_reaction_hit": target_step_rank is not None,
        "target_step_gt_reaction_best_rank": target_step_rank,
        "gt_reactant_hit": reactant_rank is not None,
        "gt_reactant_best_rank": reactant_rank,
        "provider_diagnostics": diagnostics.to_dict() if hasattr(diagnostics, "to_dict") else None,
        "proposals": proposal_rows,
    }


def _load_benchmark(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("targets") if isinstance(payload, dict) else payload
    rows = [row for row in rows or [] if isinstance(row, dict)]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def _target_product_gt_reactions(entry: dict[str, Any], target: str) -> set[str]:
    target_side = canonical_side(target)
    out: set[str] = set()
    for step in entry.get("gt_route") or []:
        key = canonical_reaction(step.get("rxn_smiles"))
        if not key or ">>" not in key:
            continue
        rhs = key.split(">>", 1)[1]
        if canonical_side(rhs) == target_side:
            out.add(key)
    return out


def _first_rank(rows: list[dict[str, Any]], key: str) -> int | None:
    for row in rows:
        if row.get(key):
            return int(row["rank"])
    return None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    returned = [int(row.get("returned") or 0) for row in rows]
    return {
        "n_targets": n,
        "targets_with_proposals": sum(1 for value in returned if value > 0),
        "avg_returned": round(sum(returned) / max(n, 1), 6),
        "exact_gt_reaction_hit": sum(1 for row in rows if row.get("exact_gt_reaction_hit")),
        "target_step_gt_reaction_hit": sum(1 for row in rows if row.get("target_step_gt_reaction_hit")),
        "gt_reactant_hit": sum(1 for row in rows if row.get("gt_reactant_hit")),
    }


def _decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    summary = _summary(rows)
    if summary["exact_gt_reaction_hit"] or summary["target_step_gt_reaction_hit"]:
        status = "proposal_hits_exist_check_search_fusion"
        reason = "context ONMT top-level proposals include exact GT reactions on at least one target."
    elif summary["gt_reactant_hit"]:
        status = "reactant_hits_without_exact_reactions"
        reason = "context ONMT proposals recover GT reactants but miss exact GT reaction detail."
    elif summary["targets_with_proposals"]:
        status = "proposal_generation_no_gt_hits"
        reason = "context ONMT returns proposals, but none match GT reactions or GT reactants under this audit."
    else:
        status = "no_proposals_returned"
        reason = "context ONMT returned no top-level proposals."
    return {"status": status, "reason": reason}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Context ONMT Top-level Proposal Audit",
        "",
        f"生成时间：{payload['created_at']}",
        "",
        "## Decision",
        "",
        f"- status: `{payload['decision']['status']}`",
        f"- reason: {payload['decision']['reason']}",
        "",
        "## Summary",
        "",
        f"- n_targets: {summary['n_targets']}",
        f"- targets_with_proposals: {summary['targets_with_proposals']}",
        f"- avg_returned: {summary['avg_returned']}",
        f"- exact_gt_reaction_hit: {summary['exact_gt_reaction_hit']}",
        f"- target_step_gt_reaction_hit: {summary['target_step_gt_reaction_hit']}",
        f"- gt_reactant_hit: {summary['gt_reactant_hit']}",
        "",
        "## Targets",
        "",
        "| target | returned | exact GT rxn | target-step GT rxn | GT reactant | best ranks |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["targets"]:
        ranks = (
            f"rxn={row.get('exact_gt_reaction_best_rank')}, "
            f"target={row.get('target_step_gt_reaction_best_rank')}, "
            f"reactant={row.get('gt_reactant_best_rank')}"
        )
        lines.append(
            f"| `{row['target_smiles']}` | {row['returned']} | {row['exact_gt_reaction_hit']} | "
            f"{row['target_step_gt_reaction_hit']} | {row['gt_reactant_hit']} | {ranks} |"
        )
    lines.append("")
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--vendor-root", type=Path, default=Path("vendor/ChemEnzyRetroPlanner"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--preference-scorer", type=Path)
    ap.add_argument("--preference-min-score", type=float)
    ap.add_argument("--preference-rerank", action="store_true")
    ap.add_argument("--raw-topk-multiplier", type=int, default=3)
    ap.add_argument("--no-filter-invalid-proposals", action="store_true")
    args = ap.parse_args()
    payload = audit_top_level_proposals(
        benchmark_path=args.benchmark,
        model_path=args.model,
        output_json=args.output,
        output_md=args.markdown_output,
        vendor_root=args.vendor_root,
        limit=args.limit,
        topk=args.topk,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        min_score=args.min_score,
        device=args.device,
        preference_scorer_path=args.preference_scorer,
        preference_min_score=args.preference_min_score,
        preference_rerank=args.preference_rerank,
        raw_topk_multiplier=args.raw_topk_multiplier,
        filter_invalid_proposals=not args.no_filter_invalid_proposals,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
