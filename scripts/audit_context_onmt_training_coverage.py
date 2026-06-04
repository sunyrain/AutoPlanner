#!/usr/bin/env python
"""Audit whether clean context-ONMT training rows cover benchmark top-level GT steps."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles  # noqa: E402


RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "context_onmt_training_coverage_audit.v1"


def audit_training_coverage(
    *,
    benchmark_path: Path,
    corpus_dir: Path,
    output_json: Path,
    output_md: Path | None = None,
    mode: str = "context",
    corpus_splits: tuple[str, ...] = ("train",),
    limit: int | None = None,
    top_neighbors: int = 3,
) -> dict[str, Any]:
    corpus_rows = _load_corpus_rows(corpus_dir, mode=mode, splits=corpus_splits)
    benchmark_rows = _load_benchmark(benchmark_path, limit=limit)
    corpus_index = _build_corpus_index(corpus_rows)
    targets = [_audit_target(row, corpus_index=corpus_index, top_neighbors=top_neighbors) for row in benchmark_rows]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "benchmark": str(benchmark_path),
        "corpus_dir": str(corpus_dir),
        "mode": mode,
        "corpus_splits": list(corpus_splits),
        "corpus_rows": len(corpus_rows),
        "settings": {"limit": limit, "top_neighbors": top_neighbors},
        "summary": _summary(targets),
        "targets": targets,
        "decision": _decision(targets),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _load_corpus_rows(corpus_dir: Path, *, mode: str, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    split_names = ("train", "valid", "test") if "all" in set(splits) else splits
    rows: list[dict[str, Any]] = []
    for split in split_names:
        path = Path(corpus_dir) / f"{mode}.{split}.meta.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"corpus meta not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_corpus_split"] = split
                    rows.append(row)
    return rows


def _load_benchmark(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("targets") if isinstance(payload, dict) else payload
    rows = [row for row in rows or [] if isinstance(row, dict)]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def _build_corpus_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    by_product: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    reaction_keys: set[str] = set()
    reactant_sides: set[tuple[str, ...]] = set()
    for idx, row in enumerate(rows):
        product = str(row.get("product") or "")
        reactants = [str(item) for item in row.get("reactants") or [] if item]
        product_side = canonical_side(product)
        reactant_side = canonical_side(".".join(reactants))
        reaction_key = _reaction_key_from_sides(reactant_side, product_side)
        entry = {
            "corpus_index": idx,
            "corpus_split": row.get("_corpus_split"),
            "source_example_id": row.get("source_example_id"),
            "source_target_index": row.get("source_target_index"),
            "route_index": row.get("route_index"),
            "step_index": row.get("step_index"),
            "product": product,
            "reactants": reactants,
            "product_side": product_side,
            "reactant_side": reactant_side,
            "reaction_key": reaction_key,
            "product_fp": _fp(product),
            "reactants_fp": _fp(".".join(reactants)),
        }
        entries.append(entry)
        by_product[product_side].append(entry)
        if reaction_key:
            reaction_keys.add(reaction_key)
        if reactant_side:
            reactant_sides.add(reactant_side)
    return {
        "entries": entries,
        "by_product": by_product,
        "reaction_keys": reaction_keys,
        "reactant_sides": reactant_sides,
    }


def _audit_target(row: dict[str, Any], *, corpus_index: dict[str, Any], top_neighbors: int) -> dict[str, Any]:
    target = str(row.get("target_smiles") or row.get("smiles") or "")
    target_side = canonical_side(target)
    gt_steps = _target_product_gt_steps(row, target)
    reactions = [
        _audit_gt_step(step, target_side=target_side, corpus_index=corpus_index, top_neighbors=top_neighbors)
        for step in gt_steps
    ]
    labels = [reaction["coverage_label"] for reaction in reactions]
    priority = ["exact_reaction_covered", "exact_product_only", "near_pair_covered", "near_product_only", "out_of_distribution", "no_target_gt_step"]
    target_label = next((label for label in priority if label in labels), "no_target_gt_step")
    return {
        "target_smiles": target,
        "route_domain": row.get("route_domain"),
        "depth": row.get("depth"),
        "n_target_gt_steps": len(gt_steps),
        "target_coverage_label": target_label,
        "target_has_exact_reaction": any(item.get("exact_reaction_in_corpus") for item in reactions),
        "target_has_exact_product": any(item.get("exact_product_in_corpus") for item in reactions),
        "target_has_near_pair": any(float(item.get("nearest_pair", {}).get("combined_similarity") or 0.0) >= 0.70 for item in reactions),
        "target_step_reactions": reactions,
    }


def _audit_gt_step(
    step: dict[str, Any],
    *,
    target_side: tuple[str, ...],
    corpus_index: dict[str, Any],
    top_neighbors: int,
) -> dict[str, Any]:
    rxn = str(step.get("rxn_smiles") or "")
    lhs, rhs = _split_reaction(rxn)
    reactant_side = canonical_side(lhs)
    product_side = canonical_side(rhs) or target_side
    reaction_key = _reaction_key_from_sides(reactant_side, product_side)
    exact_product_entries = corpus_index["by_product"].get(product_side, [])
    exact_reaction = reaction_key in corpus_index["reaction_keys"]
    exact_reactant_side = reactant_side in corpus_index["reactant_sides"]
    nearest_product_rows = _nearest_rows(
        corpus_index["entries"],
        product_fp=_fp(".".join(product_side)),
        reactants_fp=_fp(".".join(reactant_side)),
        mode="product",
        limit=top_neighbors,
    )
    nearest_pair_rows = _nearest_rows(
        corpus_index["entries"],
        product_fp=_fp(".".join(product_side)),
        reactants_fp=_fp(".".join(reactant_side)),
        mode="pair",
        limit=top_neighbors,
    )
    best_pair = nearest_pair_rows[0] if nearest_pair_rows else {}
    coverage_label = _coverage_label(
        exact_product=bool(exact_product_entries),
        exact_reaction=exact_reaction,
        best_pair=best_pair,
        best_product=nearest_product_rows[0] if nearest_product_rows else {},
    )
    return {
        "rxn_smiles": rxn,
        "canonical_reaction": canonical_reaction(rxn),
        "transformation": step.get("transformation"),
        "reactant_side": list(reactant_side),
        "product_side": list(product_side),
        "exact_product_in_corpus": bool(exact_product_entries),
        "exact_product_match_count": len(exact_product_entries),
        "exact_reaction_in_corpus": exact_reaction,
        "exact_reactant_side_any_product": exact_reactant_side,
        "nearest_product": nearest_product_rows[0] if nearest_product_rows else None,
        "nearest_pair": best_pair or None,
        "nearest_product_top": nearest_product_rows,
        "nearest_pair_top": nearest_pair_rows,
        "coverage_label": coverage_label,
    }


def _target_product_gt_steps(row: dict[str, Any], target: str) -> list[dict[str, Any]]:
    target_side = canonical_side(target)
    out = []
    for step in row.get("gt_route") or []:
        rxn = str(step.get("rxn_smiles") or "")
        _lhs, rhs = _split_reaction(rxn)
        if rhs and canonical_side(rhs) == target_side:
            out.append(step)
    return out


def _nearest_rows(
    entries: list[dict[str, Any]],
    *,
    product_fp: Any,
    reactants_fp: Any,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    scored = []
    for entry in entries:
        product_sim = _similarity(product_fp, entry["product_fp"])
        reactant_sim = _similarity(reactants_fp, entry["reactants_fp"])
        combined = 0.5 * product_sim + 0.5 * reactant_sim
        sort_value = product_sim if mode == "product" else combined
        scored.append((sort_value, product_sim, reactant_sim, combined, entry))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    out = []
    for _sort_value, product_sim, reactant_sim, combined, entry in scored[: max(1, int(limit or 1))]:
        out.append(_neighbor_entry(entry, product_sim=product_sim, reactant_sim=reactant_sim, combined=combined))
    return out


def _neighbor_entry(entry: dict[str, Any], *, product_sim: float, reactant_sim: float, combined: float) -> dict[str, Any]:
    return {
        "corpus_index": entry.get("corpus_index"),
        "corpus_split": entry.get("corpus_split"),
        "source_example_id": entry.get("source_example_id"),
        "source_target_index": entry.get("source_target_index"),
        "route_index": entry.get("route_index"),
        "step_index": entry.get("step_index"),
        "product": entry.get("product"),
        "reactants": entry.get("reactants"),
        "product_similarity": round(float(product_sim), 6),
        "reactant_similarity": round(float(reactant_sim), 6),
        "combined_similarity": round(float(combined), 6),
    }


def _coverage_label(*, exact_product: bool, exact_reaction: bool, best_pair: dict[str, Any], best_product: dict[str, Any]) -> str:
    if exact_reaction:
        return "exact_reaction_covered"
    if exact_product:
        return "exact_product_only"
    pair_sim = float((best_pair or {}).get("combined_similarity") or 0.0)
    pair_product = float((best_pair or {}).get("product_similarity") or 0.0)
    pair_reactant = float((best_pair or {}).get("reactant_similarity") or 0.0)
    product_sim = float((best_product or {}).get("product_similarity") or 0.0)
    if pair_sim >= 0.70 and pair_product >= 0.60 and pair_reactant >= 0.50:
        return "near_pair_covered"
    if product_sim >= 0.70:
        return "near_product_only"
    return "out_of_distribution"


def _summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    reactions = [reaction for target in targets for reaction in target.get("target_step_reactions") or []]
    label_counts = Counter(target.get("target_coverage_label") for target in targets)
    reaction_label_counts = Counter(reaction.get("coverage_label") for reaction in reactions)
    nearest_product = [float((reaction.get("nearest_product") or {}).get("product_similarity") or 0.0) for reaction in reactions]
    nearest_pair = [float((reaction.get("nearest_pair") or {}).get("combined_similarity") or 0.0) for reaction in reactions]
    return {
        "n_targets": len(targets),
        "n_target_step_reactions": len(reactions),
        "targets_with_target_gt_step": sum(1 for target in targets if int(target.get("n_target_gt_steps") or 0) > 0),
        "targets_with_exact_product": sum(1 for target in targets if target.get("target_has_exact_product")),
        "targets_with_exact_reaction": sum(1 for target in targets if target.get("target_has_exact_reaction")),
        "targets_with_near_pair_ge_0_70": sum(1 for target in targets if target.get("target_has_near_pair")),
        "target_label_counts": dict(label_counts),
        "reaction_label_counts": dict(reaction_label_counts),
        "reactions_exact_product": sum(1 for reaction in reactions if reaction.get("exact_product_in_corpus")),
        "reactions_exact_reaction": sum(1 for reaction in reactions if reaction.get("exact_reaction_in_corpus")),
        "reactions_exact_reactant_side_any_product": sum(1 for reaction in reactions if reaction.get("exact_reactant_side_any_product")),
        "avg_nearest_product_similarity": _avg(nearest_product),
        "avg_nearest_pair_similarity": _avg(nearest_pair),
        "nearest_product_ge_0_70": sum(1 for value in nearest_product if value >= 0.70),
        "nearest_pair_ge_0_70": sum(1 for value in nearest_pair if value >= 0.70),
    }


def _decision(targets: list[dict[str, Any]]) -> dict[str, str]:
    summary = _summary(targets)
    if summary["targets_with_exact_reaction"]:
        status = "some_exact_top_level_training_coverage"
        reason = "At least one benchmark top-level GT reaction is exactly present in the clean corpus."
    elif summary["targets_with_near_pair_ge_0_70"]:
        status = "nearest_training_coverage_without_exact_reactions"
        reason = "No exact top-level GT reaction is present, but some targets have near product/reactant pairs in the corpus."
    elif summary["targets_with_exact_product"]:
        status = "product_coverage_without_reactant_coverage"
        reason = "Benchmark products appear in the corpus, but GT reactant sides do not."
    else:
        status = "top_level_training_coverage_gap"
        reason = "Benchmark top-level GT reactions are not covered exactly or by strong nearest pairs in the clean corpus."
    return {"status": status, "reason": reason}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Context ONMT Training Coverage Audit",
        "",
        f"created_at: `{payload['created_at']}`",
        f"decision: `{payload['decision']['status']}`",
        f"reason: {payload['decision']['reason']}",
        "",
        "## Summary",
        "",
        f"- corpus_rows: {payload['corpus_rows']}",
        f"- n_targets: {summary['n_targets']}",
        f"- n_target_step_reactions: {summary['n_target_step_reactions']}",
        f"- targets_with_exact_product: {summary['targets_with_exact_product']}",
        f"- targets_with_exact_reaction: {summary['targets_with_exact_reaction']}",
        f"- targets_with_near_pair_ge_0_70: {summary['targets_with_near_pair_ge_0_70']}",
        f"- avg_nearest_product_similarity: {summary['avg_nearest_product_similarity']}",
        f"- avg_nearest_pair_similarity: {summary['avg_nearest_pair_similarity']}",
        "",
        "## Targets",
        "",
        "| target | GT steps | label | exact product | exact reaction | nearest product | nearest pair |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for target in payload["targets"]:
        reactions = target.get("target_step_reactions") or []
        best_product = max((float((r.get("nearest_product") or {}).get("product_similarity") or 0.0) for r in reactions), default=0.0)
        best_pair = max((float((r.get("nearest_pair") or {}).get("combined_similarity") or 0.0) for r in reactions), default=0.0)
        lines.append(
            "| `{target}` | {steps} | `{label}` | {exact_product} | {exact_reaction} | {product:.3f} | {pair:.3f} |".format(
                target=target.get("target_smiles"),
                steps=target.get("n_target_gt_steps"),
                label=target.get("target_coverage_label"),
                exact_product=bool(target.get("target_has_exact_product")),
                exact_reaction=bool(target.get("target_has_exact_reaction")),
                product=best_product,
                pair=best_pair,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _split_reaction(rxn: str) -> tuple[str, str]:
    if ">>" not in str(rxn or ""):
        return "", ""
    lhs, rhs = str(rxn).split(">>", 1)
    return lhs, rhs


def _reaction_key_from_sides(reactant_side: tuple[str, ...], product_side: tuple[str, ...]) -> str:
    if not reactant_side or not product_side:
        return ""
    return ".".join(reactant_side) + ">>" + ".".join(product_side)


def _fp(smiles: str) -> Any:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _avg(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(float(np.mean(clean)), 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--mode", choices=["plain", "context"], default="context")
    ap.add_argument("--corpus-split", action="append", default=[])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--top-neighbors", type=int, default=3)
    args = ap.parse_args()
    payload = audit_training_coverage(
        benchmark_path=args.benchmark,
        corpus_dir=args.corpus_dir,
        output_json=args.output,
        output_md=args.markdown_output,
        mode=args.mode,
        corpus_splits=tuple(args.corpus_split or ["train"]),
        limit=args.limit,
        top_neighbors=args.top_neighbors,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
