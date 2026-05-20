"""Audit block-supported CCTS candidate labels.

The existing CCTS label is exact-or-similar for a single transition.  This
script asks a stricter question: for a transition that sits inside a cascade
program, does a positive candidate also preserve adjacency to the neighboring
true cascade intermediate?

For retrosynthetic direction, a candidate for step i is block-supported when:

- the candidate is exact or similar-positive, and
- if a previous forward step exists, one candidate reactant is close to the
  previous step product, and/or
- if a next forward step exists, the candidate product is close to the next
  step main reactant.

The audit is not a final chemistry validator.  It is a label-quality check for
whether same-pool positives can supervise cascade block coherence.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


SCHEMA_VERSION = "ccts_block_supported_label_audit.v1"


def audit_ccts_block_supported_labels(
    *,
    candidates_jsonl: Path,
    programs_jsonl: Path,
    output_json: Path,
    output_md: Path,
    adjacency_similarity_threshold: float = 0.70,
) -> dict[str, Any]:
    started = time.monotonic()
    candidates = _read_jsonl(candidates_jsonl)
    transition_context = _transition_context(programs_jsonl)
    rows = []
    precomputed = any("block_supported_positive_label" in row for row in candidates)
    for row in candidates:
        ctx = transition_context.get(str(row.get("transition_id") or ""))
        if precomputed:
            rows.append(_candidate_audit_row_precomputed(row, ctx))
        else:
            rows.append(_candidate_audit_row(row, ctx, adjacency_similarity_threshold=adjacency_similarity_threshold))
    summary = _summary(rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "candidates_jsonl": str(candidates_jsonl),
            "programs_jsonl": str(programs_jsonl),
            "output_json": str(output_json),
            "output_md": str(output_md),
            "adjacency_similarity_threshold": adjacency_similarity_threshold,
            "precomputed_block_labels": precomputed,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "summary": summary,
        "examples": _examples(rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    return result


def _transition_context(programs_jsonl: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for program in _read_jsonl(programs_jsonl):
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        for idx, step in enumerate(steps):
            prev_step = steps[idx - 1] if idx > 0 else None
            next_step = steps[idx + 1] if idx + 1 < len(steps) else None
            out[str(step.get("transition_id") or "")] = {
                "program_id": program.get("program_id"),
                "doi": program.get("doi"),
                "cascade_id": program.get("cascade_id"),
                "target_smiles": program.get("target_smiles"),
                "step_pos": step.get("step_pos"),
                "total_steps": len(steps),
                "transform": step.get("transformation_superclass"),
                "previous_transform": (prev_step or {}).get("transformation_superclass"),
                "next_transform": (next_step or {}).get("transformation_superclass"),
                "previous_product": canonical_smiles(str((prev_step or {}).get("product_smiles") or "")),
                "next_main_reactant": canonical_smiles(str((next_step or {}).get("main_reactant") or "")),
                "intermediate_isolated": step.get("intermediate_isolated"),
                "pairwise_mode": step.get("pairwise_mode"),
            }
    return out


def _candidate_audit_row(row: dict[str, Any], ctx: dict[str, Any] | None, *, adjacency_similarity_threshold: float) -> dict[str, Any]:
    product = canonical_smiles(str(row.get("product_smiles") or ""))
    reactants = _reactants_from_reaction(str(row.get("candidate_reaction_smiles") or ""))
    previous_product = (ctx or {}).get("previous_product") or ""
    next_main = (ctx or {}).get("next_main_reactant") or ""
    prev_sim = _best_similarity(previous_product, reactants) if previous_product else None
    next_sim = _similarity(product, next_main) if next_main else None
    has_prev = bool(previous_product)
    has_next = bool(next_main)
    prev_supported = (prev_sim is not None and prev_sim >= adjacency_similarity_threshold) if has_prev else None
    next_supported = (next_sim is not None and next_sim >= adjacency_similarity_threshold) if has_next else None
    if has_prev and has_next:
        block_supported = bool(prev_supported and next_supported)
        support_mode = "prev_and_next"
    elif has_prev:
        block_supported = bool(prev_supported)
        support_mode = "prev_only"
    elif has_next:
        block_supported = bool(next_supported)
        support_mode = "next_only"
    else:
        block_supported = False
        support_mode = "terminal_or_no_context"
    positive = bool(row.get("positive_label"))
    exact = bool(row.get("exact_label"))
    similar_only = bool(row.get("similar_label") and not row.get("exact_label"))
    return {
        "transition_id": row.get("transition_id"),
        "candidate_rank": row.get("candidate_rank"),
        "product_smiles": product,
        "candidate_reaction_smiles": row.get("candidate_reaction_smiles"),
        "candidate_reactants": reactants,
        "positive_label": positive,
        "exact_label": exact,
        "similar_only_label": similar_only,
        "reactant_similarity": row.get("reactant_similarity"),
        "has_context": ctx is not None,
        "support_mode": support_mode,
        "has_previous_neighbor": has_prev,
        "has_next_neighbor": has_next,
        "previous_product": previous_product,
        "next_main_reactant": next_main,
        "previous_support_similarity": prev_sim,
        "next_support_similarity": next_sim,
        "previous_supported": prev_supported,
        "next_supported": next_supported,
        "block_supported_positive_label": positive and block_supported,
        "block_supported_exact_label": exact and block_supported,
        "block_supported_similar_only_label": similar_only and block_supported,
        "transform": (ctx or {}).get("transform"),
        "previous_transform": (ctx or {}).get("previous_transform"),
        "next_transform": (ctx or {}).get("next_transform"),
        "total_steps": (ctx or {}).get("total_steps"),
    }


def _candidate_audit_row_precomputed(row: dict[str, Any], ctx: dict[str, Any] | None) -> dict[str, Any]:
    positive = bool(row.get("positive_label"))
    exact = bool(row.get("exact_label"))
    similar_only = bool(row.get("similar_only_label") or (row.get("similar_label") and not row.get("exact_label")))
    block_supported_positive = bool(row.get("block_supported_positive_label"))
    block_supported_exact = bool(row.get("block_supported_exact_label"))
    if "block_supported_similar_only_label" in row:
        block_supported_similar_only = bool(row.get("block_supported_similar_only_label"))
    else:
        block_supported_similar_only = bool(block_supported_positive and similar_only and not exact)
    return {
        "transition_id": row.get("transition_id"),
        "candidate_rank": row.get("candidate_rank"),
        "product_smiles": canonical_smiles(str(row.get("product_smiles") or "")),
        "candidate_reaction_smiles": row.get("candidate_reaction_smiles"),
        "candidate_reactants": row.get("candidate_reactants") or [],
        "positive_label": positive,
        "exact_label": exact,
        "similar_only_label": similar_only,
        "reactant_similarity": row.get("reactant_similarity"),
        "has_context": ctx is not None,
        "support_mode": _support_mode(ctx),
        "has_previous_neighbor": bool((ctx or {}).get("previous_product")),
        "has_next_neighbor": bool((ctx or {}).get("next_main_reactant")),
        "previous_product": (ctx or {}).get("previous_product") or "",
        "next_main_reactant": (ctx or {}).get("next_main_reactant") or "",
        "previous_support_similarity": row.get("previous_support_similarity"),
        "next_support_similarity": row.get("next_support_similarity"),
        "previous_supported": None,
        "next_supported": None,
        "block_supported_positive_label": block_supported_positive,
        "block_supported_exact_label": block_supported_exact,
        "block_supported_similar_only_label": block_supported_similar_only,
        "transform": (ctx or {}).get("transform"),
        "previous_transform": (ctx or {}).get("previous_transform"),
        "next_transform": (ctx or {}).get("next_transform"),
        "total_steps": (ctx or {}).get("total_steps"),
    }


def _support_mode(ctx: dict[str, Any] | None) -> str:
    previous_product = (ctx or {}).get("previous_product") or ""
    next_main = (ctx or {}).get("next_main_reactant") or ""
    if previous_product and next_main:
        return "prev_and_next"
    if previous_product:
        return "prev_only"
    if next_main:
        return "next_only"
    return "terminal_or_no_context"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("transition_id") or "")].append(row)
    group_rows = []
    for transition_id, group in by_group.items():
        positives = [row for row in group if row.get("positive_label")]
        exact = [row for row in group if row.get("exact_label")]
        block_pos = [row for row in group if row.get("block_supported_positive_label")]
        block_exact = [row for row in group if row.get("block_supported_exact_label")]
        group_rows.append(
            {
                "transition_id": transition_id,
                "candidate_count": len(group),
                "has_context": bool(group and group[0].get("has_context")),
                "support_mode": (group[0].get("support_mode") if group else ""),
                "positive_count": len(positives),
                "exact_count": len(exact),
                "block_supported_positive_count": len(block_pos),
                "block_supported_exact_count": len(block_exact),
                "has_positive": bool(positives),
                "has_exact": bool(exact),
                "has_block_supported_positive": bool(block_pos),
                "has_block_supported_exact": bool(block_exact),
                "best_positive_rank": min((int(row.get("candidate_rank") or 10**9) for row in positives), default=None),
                "best_block_supported_positive_rank": min((int(row.get("candidate_rank") or 10**9) for row in block_pos), default=None),
                "transform": (group[0].get("transform") if group else None),
                "previous_transform": (group[0].get("previous_transform") if group else None),
                "next_transform": (group[0].get("next_transform") if group else None),
            }
        )
    positives = [row for row in rows if row.get("positive_label")]
    exact = [row for row in rows if row.get("exact_label")]
    similar_only = [row for row in rows if row.get("similar_only_label")]
    block_pos = [row for row in rows if row.get("block_supported_positive_label")]
    block_exact = [row for row in rows if row.get("block_supported_exact_label")]
    block_similar_only = [row for row in rows if row.get("block_supported_similar_only_label")]
    groups_with_positive = [row for row in group_rows if row.get("has_positive")]
    groups_with_exact = [row for row in group_rows if row.get("has_exact")]
    groups_with_block_pos = [row for row in group_rows if row.get("has_block_supported_positive")]
    groups_with_block_exact = [row for row in group_rows if row.get("has_block_supported_exact")]
    return {
        "candidate_rows": len(rows),
        "groups": len(group_rows),
        "rows": {
            "positive": len(positives),
            "exact": len(exact),
            "similar_only": len(similar_only),
            "block_supported_positive": len(block_pos),
            "block_supported_exact": len(block_exact),
            "block_supported_similar_only": len(block_similar_only),
            "block_supported_positive_fraction_of_positive": _rate(len(block_pos), len(positives)),
            "block_supported_exact_fraction_of_exact": _rate(len(block_exact), len(exact)),
            "block_supported_similar_only_fraction_of_similar_only": _rate(len(block_similar_only), len(similar_only)),
        },
        "groups_summary": {
            "positive_group_coverage": _rate(len(groups_with_positive), len(group_rows)),
            "exact_group_coverage": _rate(len(groups_with_exact), len(group_rows)),
            "block_supported_positive_group_coverage": _rate(len(groups_with_block_pos), len(group_rows)),
            "block_supported_exact_group_coverage": _rate(len(groups_with_block_exact), len(group_rows)),
            "block_supported_positive_fraction_of_positive_groups": _rate(len(groups_with_block_pos), len(groups_with_positive)),
            "block_supported_exact_fraction_of_exact_groups": _rate(len(groups_with_block_exact), len(groups_with_exact)),
        },
        "support_mode_counts": dict(Counter(row.get("support_mode") for row in group_rows)),
        "best_block_supported_positive_rank_distribution": dict(Counter(_rank_bucket(row.get("best_block_supported_positive_rank")) for row in group_rows)),
        "best_positive_rank_distribution": dict(Counter(_rank_bucket(row.get("best_positive_rank")) for row in group_rows)),
        "transform_pair_counts_for_block_supported": dict(
            Counter(
                f"{row.get('previous_transform') or '*'}->{row.get('transform') or '*'}->{row.get('next_transform') or '*'}"
                for row in group_rows
                if row.get("has_block_supported_positive")
            ).most_common(30)
        ),
        "interpretation": _interpretation(len(block_pos), len(positives), len(groups_with_block_pos), len(groups_with_positive)),
    }


def _interpretation(block_pos_rows: int, positive_rows: int, block_pos_groups: int, positive_groups: int) -> str:
    row_rate = _rate(block_pos_rows, positive_rows)
    group_rate = _rate(block_pos_groups, positive_groups)
    if row_rate < 0.25 or group_rate < 0.35:
        return "Only a minority of single-step positives are block-supported; CCTS labels must be rebuilt before search-time use."
    if row_rate < 0.50:
        return "Block support exists but is sparse; use it as a higher-weight subset and keep similar-only labels as auxiliary coverage."
    return "Block-supported positives are common enough to train a stricter candidate ranker."


def _examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    supported = [row for row in rows if row.get("block_supported_positive_label")]
    unsupported = [row for row in rows if row.get("positive_label") and not row.get("block_supported_positive_label")]
    return {
        "block_supported_positive": [_compact(row) for row in supported[:20]],
        "positive_but_not_block_supported": [_compact(row) for row in unsupported[:20]],
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "transition_id",
        "candidate_rank",
        "product_smiles",
        "positive_label",
        "exact_label",
        "similar_only_label",
        "support_mode",
        "previous_support_similarity",
        "next_support_similarity",
        "previous_transform",
        "transform",
        "next_transform",
        "candidate_reactants",
        "previous_product",
        "next_main_reactant",
    ]
    return {key: row.get(key) for key in keep}


def _reactants_from_reaction(reaction_smiles: str) -> list[str]:
    left = str(reaction_smiles or "").split(">>", 1)[0]
    out = []
    for part in left.split("."):
        smi = canonical_smiles(part)
        if smi:
            out.append(smi)
    return sorted(set(out))


def _best_similarity(smiles: str, candidates: list[str]) -> float:
    if not smiles or not candidates:
        return 0.0
    return max((_similarity(smiles, candidate) for candidate in candidates), default=0.0)


def _similarity(left: str, right: str) -> float:
    left_fp = _fp(left)
    right_fp = _fp(right)
    if left_fp is None or right_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _rank_bucket(rank: Any) -> str:
    if rank is None:
        return "missing"
    rank = int(rank)
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51plus"


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    lines = [
        "# CCTS Block-Supported Label Audit",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        str(summary.get("interpretation") or ""),
        "",
        "## Examples",
        "",
        "```json",
        json.dumps(result.get("examples") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit block-supported CCTS labels")
    ap.add_argument("--candidates-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v2_sparse_block/ccts_v2_sparse_test_candidates.jsonl")
    ap.add_argument("--programs-jsonl", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_programs_test.jsonl")
    ap.add_argument("--output-json", default="results/shared/cascadebench_strict_20260516/ccts_v2_sparse_block/ccts_block_supported_label_audit.json")
    ap.add_argument("--output-md", default="results/shared/cascadebench_strict_20260516/ccts_v2_sparse_block/CCTS_BLOCK_SUPPORTED_LABEL_AUDIT.zh.md")
    ap.add_argument("--adjacency-similarity-threshold", type=float, default=0.70)
    args = ap.parse_args()
    result = audit_ccts_block_supported_labels(
        candidates_jsonl=Path(args.candidates_jsonl),
        programs_jsonl=Path(args.programs_jsonl),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        adjacency_similarity_threshold=args.adjacency_similarity_threshold,
    )
    print(json.dumps({"summary": result["summary"], "outputs": {"json": args.output_json, "md": args.output_md}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
