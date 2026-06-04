#!/usr/bin/env python3
"""Evaluate a dual-tower template retriever against a template library."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_dual_tower_template_retriever import (  # noqa: E402
    DualTowerRetriever,
    feature_dims,
    product_features,
    template_features,
)


RDLogger.DisableLog("rdApp.*")
SCHEMA_VERSION = "dual_tower_template_retriever.eval.v1"
EVAL_KS = (1, 5, 10, 20, 50, 75, 100, 200, 500)


def main() -> None:
    args = _parse_args()
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

    template2idx, idx2template = torch.load(args.templates_index, map_location="cpu")
    templates = _idx_ordered_templates(idx2template)
    if args.max_templates is not None:
        templates = templates[: int(args.max_templates)]
    template_ids = np.arange(len(templates), dtype=np.int64)
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
    if args.max_templates is not None:
        rows = [row for row in rows if int(row["template_id"]) < int(args.max_templates)]

    metrics, rank_rows = _evaluate_rows(
        model=model,
        rows=rows,
        template_vectors=template_vectors,
        template_ids=template_ids,
        n_bits=n_bits,
        feature_set=feature_set,
        batch_size=args.product_batch_size,
        device=device,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "model": str(args.model),
            "pairs_jsonl": str(args.pairs_jsonl),
            "templates_index": str(args.templates_index),
            "limit": args.limit,
            "max_templates": args.max_templates,
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
            **metrics,
            "template_count": len(templates),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_json.with_suffix(".md").write_text(_render_markdown(report), encoding="utf-8")
    if args.rows_jsonl:
        args.rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_jsonl.open("w", encoding="utf-8") as handle:
            for row in rank_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


def _idx_ordered_templates(idx2template: dict[Any, Any]) -> list[str]:
    return [str(idx2template[idx]) for idx in range(len(idx2template))]


def _template_vectors(
    *,
    model: DualTowerRetriever,
    templates: list[str],
    n_bits: int,
    batch_size: int,
    device: torch.device,
    cache_path: Path | None,
    feature_set: str,
    template_dim: int,
    feature_workers: int = 0,
) -> torch.Tensor:
    if cache_path and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        cache_feature_set = str(payload.get("feature_set") or "baseline")
        cache_template_dim = int(payload.get("template_dim") or 0)
        if (
            int(payload.get("n_bits") or 0) == n_bits
            and int(payload.get("template_count") or 0) == len(templates)
            and cache_feature_set == feature_set
            and cache_template_dim == int(template_dim)
        ):
            return payload["template_vectors"].to(device)
    vectors = []
    with torch.no_grad():
        for fps in _iter_template_feature_batches(
            templates,
            n_bits=n_bits,
            feature_set=feature_set,
            batch_size=batch_size,
            feature_workers=feature_workers,
        ):
            x = fps.float().to(device) if isinstance(fps, torch.Tensor) else torch.from_numpy(fps).to(device)
            vectors.append(model.template_tower(x).detach().cpu())
    out = torch.cat(vectors, dim=0).contiguous()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_bits": n_bits,
                "template_count": len(templates),
                "feature_set": feature_set,
                "template_dim": int(template_dim),
                "template_vectors": out,
            },
            cache_path,
        )
    return out.to(device)


def _iter_template_feature_batches(
    templates: list[str],
    *,
    n_bits: int,
    feature_set: str,
    batch_size: int,
    feature_workers: int,
):
    batch_size = max(1, int(batch_size))
    if int(feature_workers) > 0:
        dataset = TemplateFeatureDataset(templates, n_bits=n_bits, feature_set=feature_set)
        kwargs: dict[str, Any] = {"num_workers": int(feature_workers)}
        if int(feature_workers) > 0:
            kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
        for batch in loader:
            yield batch
        return
    starts = list(range(0, len(templates), batch_size))
    if int(feature_workers) <= 1:
        for start in starts:
            yield _template_feature_batch(templates[start : start + batch_size], n_bits, feature_set)
        return

    def submit(executor: ThreadPoolExecutor, idx: int, start: int):
        batch = templates[start : start + batch_size]
        return executor.submit(_template_feature_batch, batch, n_bits, feature_set), idx

    max_pending = max(1, int(feature_workers)) * 2
    pending = {}
    buffered = {}
    next_submit_idx = 0
    next_yield_idx = 0
    with ThreadPoolExecutor(max_workers=int(feature_workers)) as executor:
        while next_submit_idx < len(starts) and len(pending) < max_pending:
            future, idx = submit(executor, next_submit_idx, starts[next_submit_idx])
            pending[future] = idx
            next_submit_idx += 1
        while pending:
            done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                idx = pending.pop(future)
                buffered[idx] = future.result()
            while next_yield_idx in buffered:
                yield buffered.pop(next_yield_idx)
                next_yield_idx += 1
                while next_submit_idx < len(starts) and len(pending) < max_pending:
                    future, idx = submit(executor, next_submit_idx, starts[next_submit_idx])
                    pending[future] = idx
                    next_submit_idx += 1


def _template_feature_batch(batch: list[str], n_bits: int, feature_set: str) -> np.ndarray:
    return np.asarray([template_features(template, n_bits, feature_set) for template in batch], dtype=np.float32)


class TemplateFeatureDataset(Dataset):
    def __init__(self, templates: list[str], *, n_bits: int, feature_set: str):
        self.templates = templates
        self.n_bits = int(n_bits)
        self.feature_set = feature_set

    def __len__(self) -> int:
        return len(self.templates)

    def __getitem__(self, idx: int) -> np.ndarray:
        return template_features(self.templates[idx], self.n_bits, self.feature_set)


def _evaluate_rows(
    *,
    model: DualTowerRetriever,
    rows: list[dict[str, Any]],
    template_vectors: torch.Tensor,
    template_ids: np.ndarray,
    n_bits: int,
    feature_set: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hits = {k: 0 for k in EVAL_KS}
    ranks = []
    rank_rows = []
    template_count = int(template_vectors.shape[0])
    top_max = min(max(EVAL_KS), template_count)
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            fps = np.asarray([product_features(str(row["product"]), n_bits, feature_set) for row in batch], dtype=np.float32)
            product_vec = model.product_tower(torch.from_numpy(fps).to(device))
            scores = product_vec @ template_vectors.T
            _, top_idx = torch.topk(scores, k=top_max, dim=1)
            top_idx_np = top_idx.detach().cpu().numpy()
            for row, pred_idx in zip(batch, top_idx_np):
                target = int(row["template_id"])
                found = np.where(pred_idx == target)[0]
                rank = int(found[0]) + 1 if len(found) else None
                ranks.append(rank)
                rank_rows.append(
                    {
                        "row_idx": row.get("row_idx"),
                        "product": row.get("product"),
                        "template_id": target,
                        "rank": rank,
                        "top1_template_id": int(pred_idx[0]) if len(pred_idx) else None,
                    }
                )
                for k in EVAL_KS:
                    if rank is not None and rank <= min(k, template_count):
                        hits[k] += 1
    n = len(rows)
    mrr = sum(1.0 / rank for rank in ranks if rank is not None) / max(n, 1)
    return {
        "n_examples": n,
        "retrieval_elapsed_s": round(time.monotonic() - started, 3),
        "avg_retrieval_s_per_example": round((time.monotonic() - started) / max(n, 1), 6),
        "mrr": round(mrr, 6),
        **{f"recall@{k}": {"count": hits[k], "rate": round(hits[k] / max(n, 1), 6)} for k in EVAL_KS},
    }, rank_rows


def _load_pairs(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("product") and item.get("template_id") is not None:
                rows.append(item)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows

def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dual Tower Template Retrieval Eval",
        "",
        f"template_count: `{summary['template_count']}`",
        f"n_examples: `{summary['n_examples']}`",
        f"avg_retrieval_s_per_example: `{summary['avg_retrieval_s_per_example']}`",
        f"mrr: `{summary['mrr']}`",
        "",
        "| metric | count | rate |",
        "| --- | ---: | ---: |",
    ]
    for k in EVAL_KS:
        item = summary[f"recall@{k}"]
        lines.append(f"| recall@{k} | {item['count']} | {item['rate']} |")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument(
        "--templates-index",
        type=Path,
        default=Path("vendor/ChemEnzyRetroPlanner/retro_planner/packages/graph_retrosyn/graph_retrosyn/data/raw/templates_index.pkl"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-jsonl", type=Path)
    parser.add_argument("--template-vector-cache", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-templates", type=int)
    parser.add_argument("--n-bits", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--template-batch-size", type=int, default=4096)
    parser.add_argument("--product-batch-size", type=int, default=128)
    parser.add_argument("--feature-workers", type=int, default=0)
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    main()
