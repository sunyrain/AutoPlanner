#!/usr/bin/env python3
"""Evaluate dual-tower template retrieval as an end-to-end one-step proposer."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import RDLogger
from rdchiral.main import rdchiralRunText


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side  # noqa: E402
from scripts.evaluate_dual_tower_template_retriever import (  # noqa: E402
    _idx_ordered_templates,
    _load_pairs,
    _template_vectors,
)
from scripts.train_dual_tower_template_retriever import (  # noqa: E402
    DualTowerRetriever,
    feature_dims,
    product_features,
)


SCHEMA_VERSION = "dual_tower_template_exact_eval.v1"
DEFAULT_TOPKS = (20, 50, 75, 100)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = _parse_args()
    topks = tuple(sorted({int(k) for k in args.topks}))
    max_topk = max(topks)
    started = time.monotonic()

    ckpt = torch.load(args.model, map_location="cpu")
    settings = ckpt.get("settings") or {}
    n_bits = int(settings.get("n_bits") or args.n_bits)
    hidden = int(settings.get("hidden") or args.hidden)
    dim = int(settings.get("dim") or args.dim)
    dropout = float(settings.get("dropout") or 0.0)
    feature_set = str(settings.get("feature_set") or "baseline")
    architecture = str(settings.get("architecture") or "baseline")
    product_dim, template_dim = feature_dims(n_bits, feature_set)
    product_dim = int(settings.get("product_dim") or product_dim)
    template_dim = int(settings.get("template_dim") or template_dim)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = DualTowerRetriever(
        n_bits=n_bits,
        hidden=hidden,
        dim=dim,
        dropout=dropout,
        product_dim=product_dim,
        template_dim=template_dim,
        architecture=architecture,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    _template2idx, idx2template = torch.load(args.templates_index, map_location="cpu")
    templates = _idx_ordered_templates(idx2template)
    template_vectors = _template_vectors(
        model=model,
        templates=templates,
        n_bits=n_bits,
        batch_size=args.template_batch_size,
        device=device,
        cache_path=args.template_vector_cache,
        feature_set=feature_set,
        template_dim=template_dim,
        feature_workers=args.feature_workers,
    )
    rows = _load_pairs(args.pairs_jsonl, limit=args.limit)
    eval_started = time.monotonic()
    scored_rows = _evaluate_exact(
        model=model,
        rows=rows,
        templates=templates,
        template_vectors=template_vectors,
        n_bits=n_bits,
        feature_set=feature_set,
        topks=topks,
        batch_size=args.product_batch_size,
        device=device,
        progress_every=args.progress_every,
    )
    elapsed_s = time.monotonic() - eval_started
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "model": str(args.model),
            "pairs_jsonl": str(args.pairs_jsonl),
            "templates_index": str(args.templates_index),
            "limit": args.limit,
            "topks": list(topks),
            "n_bits": n_bits,
            "hidden": hidden,
            "dim": dim,
            "dropout": dropout,
            "feature_set": feature_set,
            "product_dim": product_dim,
            "template_dim": template_dim,
            "architecture": architecture,
            "device": str(device),
        },
        "summary": {
            **_summarize(scored_rows, topks=topks, elapsed_s=elapsed_s),
            "template_count": len(templates),
            "total_elapsed_s": round(time.monotonic() - started, 3),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_json.with_suffix(".md").write_text(_render_markdown(report), encoding="utf-8")
    if args.rows_jsonl:
        args.rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_jsonl.open("w", encoding="utf-8") as handle:
            for row in scored_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.candidates_jsonl:
        _write_candidate_rows(args.candidates_jsonl, scored_rows)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)


def _evaluate_exact(
    *,
    model: DualTowerRetriever,
    rows: list[dict[str, Any]],
    templates: list[str],
    template_vectors: torch.Tensor,
    n_bits: int,
    feature_set: str,
    topks: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    progress_every: int,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    max_topk = max(topks)
    processed = 0
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            fps = np.asarray([product_features(str(row["product"]), n_bits, feature_set) for row in batch], dtype=np.float32)
            product_vec = model.product_tower(torch.from_numpy(fps).to(device))
            scores = product_vec @ template_vectors.T
            probs, top_idx = torch.topk(scores, k=max_topk, dim=1)
            top_idx_np = top_idx.detach().cpu().numpy()
            score_np = probs.detach().cpu().numpy()
            for row, pred_idx, pred_scores in zip(batch, top_idx_np, score_np):
                scored_rows.append(
                    _score_one(
                        row=row,
                        template_ids=[int(i) for i in pred_idx],
                        template_scores=[float(x) for x in pred_scores],
                        templates=templates,
                        topks=topks,
                    )
                )
                processed += 1
                if progress_every and processed % int(progress_every) == 0:
                    elapsed = time.monotonic() - started
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "elapsed_s": round(elapsed, 3),
                                "examples_per_s": round(processed / max(elapsed, 1e-9), 4),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    return scored_rows


def _score_one(
    *,
    row: dict[str, Any],
    template_ids: list[int],
    template_scores: list[float],
    templates: list[str],
    topks: tuple[int, ...],
) -> dict[str, Any]:
    product = str(row.get("product") or "")
    target = str(row.get("reactants") or "")
    target_key = canonical_side(target)
    exact_rank_by_template = None
    any_rank_by_template = None
    exact_rank_by_candidate = None
    any_rank_by_candidate = None
    valid_template_count = 0
    candidate_rank = 0
    seen_reactants: set[tuple[str, ...]] = set()
    counts_by_topk: dict[int, dict[str, int]] = {k: {"valid_templates": 0, "candidates": 0} for k in topks}
    preview: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for template_rank, (template_id, score) in enumerate(zip(template_ids, template_scores), start=1):
        try:
            outcomes = sorted(rdchiralRunText(templates[template_id], product))
        except Exception:
            outcomes = []
        if outcomes:
            valid_template_count += 1
        new_candidates = 0
        for reactants in outcomes:
            pred_key = canonical_side(reactants)
            if not pred_key or pred_key in seen_reactants:
                continue
            seen_reactants.add(pred_key)
            new_candidates += 1
            candidate_rank += 1
            exact = bool(target_key and pred_key == target_key)
            any_reactant = bool(set(pred_key) & set(target_key))
            if exact and exact_rank_by_template is None:
                exact_rank_by_template = template_rank
            if any_reactant and any_rank_by_template is None:
                any_rank_by_template = template_rank
            if exact and exact_rank_by_candidate is None:
                exact_rank_by_candidate = candidate_rank
            if any_reactant and any_rank_by_candidate is None:
                any_rank_by_candidate = candidate_rank
            if len(preview) < 10:
                preview.append(_candidate_payload(template_rank, candidate_rank, template_id, score, pred_key, exact, any_reactant))
            candidates.append(_candidate_payload(template_rank, candidate_rank, template_id, score, pred_key, exact, any_reactant))
        for topk in topks:
            if template_rank <= topk:
                counts_by_topk[topk]["valid_templates"] = valid_template_count
                counts_by_topk[topk]["candidates"] = len(seen_reactants)

    return {
        "row_idx": row.get("row_idx"),
        "product": product,
        "target_reactants": target,
        "target_template_id": row.get("template_id"),
        "exact_rank_by_template": exact_rank_by_template,
        "any_rank_by_template": any_rank_by_template,
        "exact_rank_by_candidate": exact_rank_by_candidate,
        "any_rank_by_candidate": any_rank_by_candidate,
        "candidate_count": len(seen_reactants),
        "valid_template_count": valid_template_count,
        "counts_by_topk": counts_by_topk,
        "candidates_preview": preview,
        "candidates": candidates,
    }


def _candidate_payload(
    template_rank: int,
    candidate_rank: int,
    template_id: int,
    score: float,
    pred_key: tuple[str, ...],
    exact: bool,
    any_reactant: bool,
) -> dict[str, Any]:
    return {
        "template_rank": template_rank,
        "candidate_rank": candidate_rank,
        "template_id": template_id,
        "score": round(score, 6),
        "reactants": ".".join(pred_key),
        "exact": exact,
        "any_reactant": any_reactant,
    }


def _write_candidate_rows(path: Path, scored_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(scored_rows):
            for candidate in row.get("candidates") or []:
                handle.write(
                    json.dumps(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "split": "test",
                            "idx": idx,
                            "group_id": f"test:{idx}",
                            "product": row.get("product"),
                            "target_reactants": row.get("target_reactants"),
                            "candidate": {
                                "rank": candidate.get("candidate_rank"),
                                "template_rank": candidate.get("template_rank"),
                                "score": candidate.get("score"),
                                "reactants": str(candidate.get("reactants") or "").split("."),
                                "reactants_text": candidate.get("reactants"),
                                "template_id": candidate.get("template_id"),
                                "source": "dualtower_enhanced",
                                "type": "template",
                            },
                            "labels": {
                                "exact": bool(candidate.get("exact")),
                                "any_reactant": bool(candidate.get("any_reactant")),
                            },
                            "features": {
                                "rank": float(candidate.get("candidate_rank") or 0),
                                "template_rank": float(candidate.get("template_rank") or 0),
                                "candidate_score": float(candidate.get("score") or 0.0),
                                "inverse_rank": 1.0 / max(float(candidate.get("candidate_rank") or 1), 1.0),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _summarize(rows: list[dict[str, Any]], *, topks: tuple[int, ...], elapsed_s: float) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {
        "n_examples": n,
        "elapsed_s": round(elapsed_s, 3),
        "avg_elapsed_s_per_example": round(elapsed_s / max(n, 1), 6),
        "zero_candidate": sum(1 for row in rows if int(row.get("candidate_count") or 0) == 0),
        "avg_candidate_count_at_max_topk": round(
            sum(int(row.get("candidate_count") or 0) for row in rows) / max(n, 1),
            6,
        ),
        "avg_valid_template_count_at_max_topk": round(
            sum(int(row.get("valid_template_count") or 0) for row in rows) / max(n, 1),
            6,
        ),
    }
    for topk in topks:
        exact = sum(
            1
            for row in rows
            if row.get("exact_rank_by_template") is not None and int(row["exact_rank_by_template"]) <= topk
        )
        any_hit = sum(
            1
            for row in rows
            if row.get("any_rank_by_template") is not None and int(row["any_rank_by_template"]) <= topk
        )
        candidate_counts = [int((row.get("counts_by_topk") or {}).get(topk, {}).get("candidates") or 0) for row in rows]
        valid_counts = [
            int((row.get("counts_by_topk") or {}).get(topk, {}).get("valid_templates") or 0) for row in rows
        ]
        out[f"top{topk}"] = {
            "exact_hit": exact,
            "exact_hit_rate": round(exact / max(n, 1), 6),
            "any_reactant_hit": any_hit,
            "any_reactant_hit_rate": round(any_hit / max(n, 1), 6),
            "avg_candidate_count": round(sum(candidate_counts) / max(n, 1), 6),
            "avg_valid_template_count": round(sum(valid_counts) / max(n, 1), 6),
            "zero_candidate": sum(1 for value in candidate_counts if value == 0),
            "candidate_count_histogram": dict(Counter(candidate_counts).most_common(20)),
        }
    return out


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dual Tower Template Exact Eval",
        "",
        f"n_examples: `{summary['n_examples']}`",
        f"template_count: `{summary['template_count']}`",
        f"avg_elapsed_s_per_example: `{summary['avg_elapsed_s_per_example']}`",
        "",
        "| topK templates | exact | exact_rate | any_reactant | any_rate | avg_candidates | avg_valid_templates |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in report["settings"]["topks"]:
        item = summary[f"top{key}"]
        lines.append(
            f"| {key} | {item['exact_hit']} | {item['exact_hit_rate']} | "
            f"{item['any_reactant_hit']} | {item['any_reactant_hit_rate']} | "
            f"{item['avg_candidate_count']} | {item['avg_valid_template_count']} |"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument(
        "--templates-index",
        type=Path,
        default=Path(
            "vendor/ChemEnzyRetroPlanner/retro_planner/packages/graph_retrosyn/graph_retrosyn/data/raw/templates_index.pkl"
        ),
    )
    parser.add_argument("--template-vector-cache", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-jsonl", type=Path)
    parser.add_argument("--candidates-jsonl", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--topks", type=int, nargs="+", default=list(DEFAULT_TOPKS))
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--template-batch-size", type=int, default=4096)
    parser.add_argument("--product-batch-size", type=int, default=64)
    parser.add_argument("--feature-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main()
