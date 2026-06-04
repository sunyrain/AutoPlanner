#!/usr/bin/env python
"""Audit known-legal corpus proposals against benchmark GT reactions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascade_search import LegalCorpusProposalProvider  # noqa: E402
from cascade_planner.cascade_search.proposal_validity import ProposalValidityConfig  # noqa: E402
from scripts.audit_context_onmt_top_level_proposals import (  # noqa: E402
    audit_top_level_proposals,
)


SCHEMA_VERSION = "legal_corpus_top_level_proposal_audit.v1"


def audit_legal_corpus_top_level_proposals(
    *,
    benchmark_path: Path,
    corpus_paths: list[Path],
    output_json: Path,
    output_md: Path | None = None,
    limit: int | None = None,
    topk: int = 100,
    max_index_rows: int | None = None,
    similarity_floor: float = 0.0,
    candidate_pool_size: int = 512,
    index_cache_path: Path | None = None,
) -> dict[str, Any]:
    provider = LegalCorpusProposalProvider(
        corpus_paths,
        max_index_rows=max_index_rows,
        similarity_floor=similarity_floor,
        candidate_pool_size=candidate_pool_size,
        validity_config=ProposalValidityConfig(max_reactant_to_product_heavy_ratio=None),
        index_cache_path=index_cache_path,
    )
    payload = audit_top_level_proposals(
        benchmark_path=benchmark_path,
        model_path=Path("known_legal_corpus"),
        output_json=output_json,
        output_md=None,
        limit=limit,
        topk=topk,
        provider=provider,
    )
    payload["schema_version"] = SCHEMA_VERSION
    payload["model_path"] = "known_legal_corpus"
    payload["decision"] = _legal_corpus_decision(payload.get("targets") or [])
    payload["settings"].update(
        {
            "corpus_paths": [str(path) for path in corpus_paths],
            "max_index_rows": max_index_rows,
            "similarity_floor": similarity_floor,
            "candidate_pool_size": candidate_pool_size,
            "index_cache_path": str(index_cache_path) if index_cache_path else None,
            "provider": LegalCorpusProposalProvider.provider_name,
            "contract": (
                "Known-legal corpus candidate-pool baseline; no expert label, no route-quality claim."
            ),
        }
    )
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_legal_corpus_markdown(payload), encoding="utf-8")
    return payload


def _legal_corpus_decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    summary = {
        "exact_gt_reaction_hit": sum(1 for row in rows if row.get("exact_gt_reaction_hit")),
        "target_step_gt_reaction_hit": sum(1 for row in rows if row.get("target_step_gt_reaction_hit")),
        "gt_reactant_hit": sum(1 for row in rows if row.get("gt_reactant_hit")),
        "targets_with_proposals": sum(1 for row in rows if int(row.get("returned") or 0) > 0),
    }
    if summary["target_step_gt_reaction_hit"]:
        return {
            "status": "legal_candidate_pool_has_exact_target_step_hits",
            "reason": (
                "Known-legal corpus candidates recover exact target-step GT reactions on at least one benchmark target; "
                "use this as proposal-pool coverage evidence, not route-quality proof."
            ),
        }
    if summary["exact_gt_reaction_hit"]:
        return {
            "status": "legal_candidate_pool_has_exact_route_step_hits",
            "reason": (
                "Known-legal corpus candidates recover exact GT reactions on at least one benchmark target; "
                "target-step recovery remains absent."
            ),
        }
    if summary["gt_reactant_hit"]:
        return {
            "status": "legal_candidate_pool_has_reactant_hits_only",
            "reason": (
                "Known-legal corpus candidates recover GT reactants but not exact GT reactions under this audit."
            ),
        }
    if summary["targets_with_proposals"]:
        return {
            "status": "legal_candidate_pool_no_gt_hits",
            "reason": (
                "Known-legal corpus candidates are returned, but none match GT reactions or GT reactants under this audit."
            ),
        }
    return {
        "status": "legal_candidate_pool_empty",
        "reason": "Known-legal corpus provider returned no top-level proposals.",
    }


def render_legal_corpus_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Legal Corpus Top-level Proposal Audit",
        "",
        f"生成时间：{payload['created_at']}",
        "",
        "## Contract",
        "",
        "- provider: `legal_corpus`",
        "- 含义：只返回 canonical external corpus 中真实出现过的合法反应物集合。",
        "- 注意：这是 proposal 候选池覆盖审计，不是完整路线质量、条件兼容性或可合成性证明。",
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--max-index-rows", type=int)
    ap.add_argument("--similarity-floor", type=float, default=0.0)
    ap.add_argument("--candidate-pool-size", type=int, default=512)
    ap.add_argument("--index-cache", type=Path)
    args = ap.parse_args()
    payload = audit_legal_corpus_top_level_proposals(
        benchmark_path=args.benchmark,
        corpus_paths=args.corpus,
        output_json=args.output,
        output_md=args.markdown_output,
        limit=args.limit,
        topk=args.topk,
        max_index_rows=args.max_index_rows,
        similarity_floor=args.similarity_floor,
        candidate_pool_size=args.candidate_pool_size,
        index_cache_path=args.index_cache,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
