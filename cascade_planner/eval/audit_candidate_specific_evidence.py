"""Audit candidate-specific cascade evidence on fixed ChemEnzy candidates.

This is the next diagnostic after the Phase II guardrail audit.  CCTS-v2 used
route/program context, but its context features were weakly tied to the actual
candidate reaction.  Here we score each candidate by its own structural analogy
to train-split v4 transitions, then evaluate whether that candidate-specific
evidence can recover block-supported positives in the same fixed ChemEnzy pool.

The script is deliberately an audit, not a promoted model.  It answers whether
candidate-specific retrieval has signal worth turning into CCTS-v3 features.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


SCHEMA_VERSION = "candidate_specific_evidence_audit.v1"


def audit_candidate_specific_evidence(
    *,
    candidates_jsonl: Path,
    coverage_json: Path,
    cache_json: Path,
    program_manifest: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    candidates = _read_jsonl(candidates_jsonl)
    coverage = _read_json(coverage_json)
    cache = _read_json(cache_json)
    cache_index = _cache_index(cache)
    transition_context = _coverage_context(coverage, program_manifest)
    train_bank = _train_bank(program_manifest)

    rows = []
    product_sim_cache: dict[tuple[str, str], list[float]] = {}
    for row in candidates:
        transition_id = str(row.get("transition_id") or "")
        product = str(row.get("product_smiles") or "")
        candidate = _candidate_from_cache(cache_index, product, int(row.get("candidate_rank") or 0))
        ctx = transition_context.get(transition_id) or {}
        enriched = dict(row)
        enriched.update(_candidate_fields(candidate))
        enriched.update(ctx)
        enriched.update(
            _candidate_evidence_scores(
                product=product,
                candidate_main=enriched.get("candidate_main_reactant") or "",
                context_transform=str(ctx.get("transform") or ""),
                previous_transform=str(ctx.get("previous_transform") or ""),
                next_transform=str(ctx.get("next_transform") or ""),
                train_bank=train_bank,
                product_sim_cache=product_sim_cache,
            )
        )
        rows.append(enriched)

    score_names = [
        "chem_rank",
        "candidate_nearest_any_transition_sim",
        "candidate_nearest_context_transform_sim",
        "candidate_nearest_pair_compatible_sim",
        "candidate_inferred_transform_match_score",
    ]
    summary = {
        "counts": {
            "candidate_rows": len(rows),
            "groups": len({row.get("transition_id") for row in rows}),
            "train_items": train_bank["count"],
            "train_transform_counts": train_bank["transform_counts"],
        },
        "metrics": {
            label: _evaluate_scores(rows, label_name=label, score_names=score_names)
            for label in (
                "block_supported_positive_label",
                "block_supported_exact_label",
                "exact_label",
                "positive_label",
                "similar_only_label",
            )
        },
        "inferred_transform": _inferred_transform_summary(rows),
        "examples": _examples(rows),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "candidates_jsonl": str(candidates_jsonl),
            "coverage_json": str(coverage_json),
            "cache_json": str(cache_json),
            "program_manifest": str(program_manifest),
            "output_json": str(output_json),
            "output_md": str(output_md),
            "elapsed_s": round(time.monotonic() - started, 3),
            "score_contract": "candidate-specific train-transition retrieval; no training and no hand-weighted blended score",
        },
        "summary": summary,
        "decision": _decision(summary),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    return result


def _candidate_evidence_scores(
    *,
    product: str,
    candidate_main: str,
    context_transform: str,
    previous_transform: str,
    next_transform: str,
    train_bank: dict[str, Any],
    product_sim_cache: dict[tuple[str, str], list[float]],
) -> dict[str, Any]:
    product_fp = _fp(product)
    main_fp = _fp(candidate_main)
    best_any = _best_in_bucket(product, product_fp, main_fp, train_bank["all"], product_sim_cache, "all")
    same_transform_bucket = train_bank["by_transform"].get(context_transform) or []
    best_context = _best_in_bucket(product, product_fp, main_fp, same_transform_bucket, product_sim_cache, f"t:{context_transform}")
    pair_buckets = []
    if previous_transform:
        pair_buckets.extend(train_bank["by_next_pair"].get(f"{previous_transform}->{context_transform}") or [])
    if next_transform:
        pair_buckets.extend(train_bank["by_prev_pair"].get(f"{context_transform}->{next_transform}") or [])
    best_pair = _best_in_bucket(product, product_fp, main_fp, pair_buckets, product_sim_cache, f"p:{previous_transform}->{context_transform}->{next_transform}")
    inferred_transform = str((best_any.get("item") or {}).get("transform") or "")
    return {
        "candidate_nearest_any_transition_sim": best_any.get("transition_sim", 0.0),
        "candidate_nearest_any_product_sim": best_any.get("product_sim", 0.0),
        "candidate_nearest_any_main_sim": best_any.get("main_sim", 0.0),
        "candidate_inferred_transform": inferred_transform,
        "candidate_inferred_transform_matches_context": bool(inferred_transform and inferred_transform == context_transform),
        "candidate_inferred_transform_match_score": float(inferred_transform == context_transform) * float(best_any.get("transition_sim", 0.0)),
        "candidate_nearest_context_transform_sim": best_context.get("transition_sim", 0.0),
        "candidate_nearest_pair_compatible_sim": best_pair.get("transition_sim", 0.0),
        "candidate_nearest_context_support_id": (best_context.get("item") or {}).get("transition_id"),
        "candidate_nearest_any_support_id": (best_any.get("item") or {}).get("transition_id"),
    }


def _best_in_bucket(
    product: str,
    product_fp: Any,
    main_fp: Any,
    bucket: list[dict[str, Any]],
    product_sim_cache: dict[tuple[str, str], list[float]],
    cache_key: str,
) -> dict[str, Any]:
    if not bucket or product_fp is None or main_fp is None:
        return {"transition_sim": 0.0, "product_sim": 0.0, "main_sim": 0.0, "item": None}
    key = (product, cache_key)
    if key not in product_sim_cache:
        product_sim_cache[key] = list(DataStructs.BulkTanimotoSimilarity(product_fp, [item["product_fp"] for item in bucket]))
    product_sims = product_sim_cache[key]
    main_sims = DataStructs.BulkTanimotoSimilarity(main_fp, [item["main_fp"] for item in bucket])
    best_idx = -1
    best_score = -1.0
    best_product = 0.0
    best_main = 0.0
    for idx, (prod_sim, main_sim) in enumerate(zip(product_sims, main_sims)):
        # Audit score, not a promoted model: equal product/main similarity.
        score = 0.5 * float(prod_sim) + 0.5 * float(main_sim)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_product = float(prod_sim)
            best_main = float(main_sim)
    return {
        "transition_sim": round(float(best_score), 6) if best_idx >= 0 else 0.0,
        "product_sim": round(best_product, 6),
        "main_sim": round(best_main, 6),
        "item": bucket[best_idx] if best_idx >= 0 else None,
    }


def _evaluate_scores(rows: list[dict[str, Any]], *, label_name: str, score_names: list[str]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("transition_id") or "")].append(row)
    out = {}
    for score_name in score_names:
        ranks = []
        for group in by_group.values():
            order = sorted(group, key=lambda row: (-float(_score(row, score_name)), int(row.get("candidate_rank") or 10**9)))
            label_ranks = [idx + 1 for idx, row in enumerate(order) if row.get(label_name)]
            ranks.append(min(label_ranks) if label_ranks else None)
        out[score_name] = _rank_metrics(ranks)
    return out


def _score(row: dict[str, Any], score_name: str) -> float:
    if score_name == "chem_rank":
        return -float(row.get("candidate_rank") or 10**9)
    return float(row.get(score_name) or 0.0)


def _rank_metrics(ranks: list[int | None]) -> dict[str, Any]:
    covered = [int(rank) for rank in ranks if rank is not None]
    total = max(len(ranks), 1)
    covered_den = max(len(covered), 1)
    return {
        "covered_groups": len(covered),
        "coverage": round(len(covered) / total, 6),
        "mrr_covered": round(sum(1.0 / rank for rank in covered) / covered_den, 6) if covered else 0.0,
        "recall_at_k_all": {
            str(k): round(sum(1 for rank in covered if rank <= k) / total, 6)
            for k in (1, 3, 5, 10, 20, 50)
        },
        "first_rank_buckets": dict(Counter(_rank_bucket(rank) for rank in ranks)),
    }


def _inferred_transform_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subsets = {
        "all": rows,
        "block_supported_positive": [row for row in rows if row.get("block_supported_positive_label")],
        "exact": [row for row in rows if row.get("exact_label")],
        "negative": [row for row in rows if not row.get("positive_label")],
    }
    out = {}
    for name, subset in subsets.items():
        out[name] = {
            "rows": len(subset),
            "match_rate": _rate(sum(1 for row in subset if row.get("candidate_inferred_transform_matches_context")), len(subset)),
            "top_inferred_transforms": dict(Counter(str(row.get("candidate_inferred_transform") or "") for row in subset).most_common(20)),
        }
    return out


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    block_metrics = ((summary.get("metrics") or {}).get("block_supported_positive_label") or {})
    chem = block_metrics.get("chem_rank") or {}
    best_name = None
    best_mrr = -1.0
    for name, metric in block_metrics.items():
        if name == "chem_rank":
            continue
        mrr = float(metric.get("mrr_covered") or 0.0)
        if mrr > best_mrr:
            best_mrr = mrr
            best_name = name
    chem_mrr = float(chem.get("mrr_covered") or 0.0)
    return {
        "best_candidate_specific_score": best_name,
        "best_candidate_specific_mrr": round(best_mrr, 6),
        "chem_rank_mrr": round(chem_mrr, 6),
        "delta_vs_chem_rank": round(best_mrr - chem_mrr, 6),
        "candidate_specific_signal": bool(best_mrr > chem_mrr + 0.02),
        "recommendation": (
            "promote candidate-specific retrieval into CCTS-v3 features"
            if best_mrr > chem_mrr + 0.02
            else "do not promote this simple retrieval score; improve candidate transform/reaction-center evidence"
        ),
    }


def _examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positives = [row for row in rows if row.get("block_supported_positive_label")]
    negatives = [row for row in rows if not row.get("positive_label")]
    positives = sorted(positives, key=lambda row: -float(row.get("candidate_nearest_context_transform_sim") or 0.0))
    negatives = sorted(negatives, key=lambda row: -float(row.get("candidate_nearest_context_transform_sim") or 0.0))
    return {
        "high_evidence_block_supported": [_compact(row) for row in positives[:15]],
        "high_evidence_negative": [_compact(row) for row in negatives[:15]],
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "transition_id",
        "product_smiles",
        "candidate_rank",
        "candidate_main_reactant",
        "transform",
        "previous_transform",
        "next_transform",
        "block_supported_positive_label",
        "exact_label",
        "similar_only_label",
        "candidate_nearest_any_transition_sim",
        "candidate_nearest_context_transform_sim",
        "candidate_nearest_pair_compatible_sim",
        "candidate_inferred_transform",
        "candidate_inferred_transform_matches_context",
    ]
    return {key: row.get(key) for key in keep}


def _candidate_fields(candidate: dict[str, Any] | None) -> dict[str, Any]:
    candidate = candidate or {}
    main = canonical_smiles(str(candidate.get("main_reactant") or ""))
    return {
        "candidate_main_reactant": main,
        "candidate_reaction_smiles": candidate.get("reaction_smiles") or candidate.get("rxn_smiles"),
        "candidate_source": candidate.get("source"),
        "candidate_model": candidate.get("model_full_name") or candidate.get("teacher_source"),
    }


def _candidate_from_cache(cache_index: dict[str, list[dict[str, Any]]], product: str, rank: int) -> dict[str, Any] | None:
    rows = cache_index.get(product) or []
    for row in rows:
        if int(row.get("rank") or 0) == int(rank):
            return row
    if 1 <= rank <= len(rows):
        return rows[rank - 1]
    return None


def _coverage_context(coverage: dict[str, Any], program_manifest: Path) -> dict[str, dict[str, Any]]:
    program_context = _program_context(program_manifest)
    out = {}
    for row in coverage.get("transitions") or []:
        tid = str(row.get("transition_id") or "")
        ctx = dict(program_context.get(tid) or {})
        ctx.update(
            {
                "transition_id": tid,
                "product_smiles": row.get("product_smiles"),
                "target_smiles": row.get("target_smiles"),
                "transform": row.get("transformation_superclass") or ctx.get("transform"),
                "previous_transform": row.get("previous_transformation_superclass") or ctx.get("previous_transform") or "",
                "route_domain": row.get("route_domain"),
            }
        )
        out[tid] = ctx
    return out


def _program_context(program_manifest: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(program_manifest)
    outputs = manifest.get("outputs") or {}
    out = {}
    for split in ("train", "val", "test"):
        path = outputs.get(split)
        if not path:
            continue
        for program in _read_jsonl(Path(path)):
            steps = _program_steps(program)
            for idx, step in enumerate(steps):
                prev_step = steps[idx - 1] if idx > 0 else None
                next_step = steps[idx + 1] if idx + 1 < len(steps) else None
                transition_id = str(step.get("transition_id") or f"{program.get('program_id') or program.get('cascade_id') or program.get('doi') or 'program'}::{idx}")
                out[transition_id] = {
                    "program_id": program.get("program_id") or program.get("cascade_id"),
                    "doi": program.get("doi"),
                    "cascade_id": program.get("cascade_id"),
                    "transform": step.get("transformation_superclass"),
                    "previous_transform": (prev_step or {}).get("transformation_superclass") or "",
                    "next_transform": (next_step or {}).get("transformation_superclass") or "",
                }
    return out


def _train_bank(program_manifest: Path) -> dict[str, Any]:
    manifest = _read_json(program_manifest)
    outputs = manifest.get("outputs") or {}
    graph_path = outputs.get("train_evidence_graph")
    graph = _read_json(Path(graph_path)) if graph_path else {}
    items = []
    for program in _read_jsonl(Path(outputs["train"])):
        steps = _program_steps(program)
        for idx, step in enumerate(steps):
            prev_step = steps[idx - 1] if idx > 0 else None
            next_step = steps[idx + 1] if idx + 1 < len(steps) else None
            product = _step_product(step)
            main = _step_main_reactant(step)
            product_fp = _fp(product)
            main_fp = _fp(main)
            if product_fp is None or main_fp is None:
                continue
            items.append(
                {
                    "transition_id": step.get("transition_id") or f"{program.get('program_id') or program.get('cascade_id') or program.get('doi') or 'program'}::{idx}",
                    "transform": str(step.get("transformation_superclass") or ""),
                    "previous_transform": str((prev_step or {}).get("transformation_superclass") or ""),
                    "next_transform": str((next_step or {}).get("transformation_superclass") or ""),
                    "product_smiles": product,
                    "main_reactant": main,
                    "product_fp": product_fp,
                    "main_fp": main_fp,
                }
            )
    by_transform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prev_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_next_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_transform[item["transform"]].append(item)
        if item["next_transform"]:
            by_prev_pair[f"{item['transform']}->{item['next_transform']}"].append(item)
        if item["previous_transform"]:
            by_next_pair[f"{item['previous_transform']}->{item['transform']}"].append(item)
    return {
        "all": items,
        "by_transform": dict(by_transform),
        "by_prev_pair": dict(by_prev_pair),
        "by_next_pair": dict(by_next_pair),
        "count": len(items),
        "transform_counts": dict(Counter(item["transform"] for item in items)),
        "graph": graph,
    }


def _program_steps(program: dict[str, Any]) -> list[dict[str, Any]]:
    steps = program.get("steps")
    if isinstance(steps, list) and steps:
        return [step for step in steps if isinstance(step, dict)]
    gt_route = program.get("gt_route")
    if isinstance(gt_route, list):
        return [step for step in gt_route if isinstance(step, dict)]
    return []


def _step_product(step: dict[str, Any]) -> str:
    value = step.get("product_smiles")
    if value:
        return canonical_smiles(str(value)) or str(value)
    return _largest_component(_reaction_products(step.get("rxn_smiles") or step.get("reaction_smiles")))


def _step_main_reactant(step: dict[str, Any]) -> str:
    value = step.get("main_reactant")
    if value:
        return canonical_smiles(str(value)) or str(value)
    reactants = [canonical_smiles(str(part)) or str(part) for part in (step.get("reactants") or []) if part]
    if not reactants:
        reactants = [canonical_smiles(str(part)) or str(part) for part in _reaction_reactants(step.get("rxn_smiles") or step.get("reaction_smiles")) if part]
    return _largest_component(reactants)


def _reaction_reactants(rxn_smiles: Any) -> list[str]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return []
    left, _ = text.split(">>", 1)
    return [part.strip() for part in left.split(".") if part.strip()]


def _reaction_products(rxn_smiles: Any) -> list[str]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return []
    _, right = text.split(">>", 1)
    return [part.strip() for part in right.split(".") if part.strip()]


def _largest_component(parts: list[str]) -> str:
    best = ""
    best_key = (-1, -1)
    for part in parts:
        smi = canonical_smiles(str(part)) or str(part)
        mol = Chem.MolFromSmiles(smi)
        heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
        key = (heavy, len(smi))
        if key > best_key:
            best_key = key
            best = smi
    return best


def _cache_index(cache: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for raw_key, rows in cache.items():
        try:
            key = json.loads(raw_key)
        except Exception:
            continue
        product = str(key.get("product") or "")
        if product and isinstance(rows, list):
            out[product] = [row for row in rows if isinstance(row, dict)]
    return out


@lru_cache(maxsize=200000)
def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"unsupported JSON array format: {path}")
        return [row for row in payload if isinstance(row, dict)]
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
    metrics = summary.get("metrics") or {}
    lines = [
        "# Candidate-Specific Evidence Audit",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(result.get("decision") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(summary.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Ranking Metrics",
        "",
    ]
    for label, score_rows in metrics.items():
        lines.extend(
            [
                f"### {label}",
                "",
                "| Score | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for score, row in score_rows.items():
            at = row.get("recall_at_k_all") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        score,
                        str(row.get("coverage")),
                        str(row.get("mrr_covered")),
                        str(at.get("1")),
                        str(at.get("3")),
                        str(at.get("5")),
                        str(at.get("10")),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Inferred Transform",
            "",
            "```json",
            json.dumps(summary.get("inferred_transform") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Examples",
            "",
            "```json",
            json.dumps(summary.get("examples") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit candidate-specific cascade evidence")
    ap.add_argument("--candidates-jsonl", default="results/shared/cascadebench_strict_20260516/ccts_v2_sparse_block/ccts_v2_sparse_test_candidates.jsonl")
    ap.add_argument("--coverage-json", default="results/shared/cascadebench_strict_20260516/coverage/coverage_test_top100.json")
    ap.add_argument("--cache-json", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-json", default="results/shared/cascadebench_strict_20260516/candidate_specific_evidence_audit.json")
    ap.add_argument("--output-md", default="results/shared/cascadebench_strict_20260516/CANDIDATE_SPECIFIC_EVIDENCE_AUDIT.zh.md")
    args = ap.parse_args()
    result = audit_candidate_specific_evidence(
        candidates_jsonl=Path(args.candidates_jsonl),
        coverage_json=Path(args.coverage_json),
        cache_json=Path(args.cache_json),
        program_manifest=Path(args.program_manifest),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )
    print(json.dumps({"decision": result["decision"], "outputs": {"json": args.output_json, "md": args.output_md}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
