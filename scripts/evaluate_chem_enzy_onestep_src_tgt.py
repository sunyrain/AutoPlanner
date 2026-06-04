#!/usr/bin/env python3
"""Evaluate ChemEnzy one-step providers on src/tgt product-reactant files."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import RDLogger  # noqa: E402

from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider  # noqa: E402
from cascade_planner.cascadeboard.route_recovery import canonical_side  # noqa: E402


SCHEMA_VERSION = "chem_enzy_onestep_src_tgt_eval.v1"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = _parse_args()
    products = _read_lines(args.src, limit=args.limit)
    targets = _read_lines(args.tgt, limit=args.limit)
    if len(products) != len(targets):
        raise ValueError(f"src/tgt length mismatch: {len(products)} != {len(targets)}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "src": str(args.src),
            "tgt": str(args.tgt),
            "limit": args.limit,
            "topk": args.topk,
            "gpu": args.gpu,
            "vendor_root": str(args.vendor_root),
            "onmt_model_path": str(args.onmt_model_path) if args.onmt_model_path else None,
            "onmt_tokenizer": args.onmt_tokenizer,
        },
        "runs": {},
    }
    for raw in args.run:
        name, models = _parse_run(raw)
        payload["runs"][name] = evaluate_run(
            products=products,
            targets=targets,
            models=models,
            vendor_root=args.vendor_root,
            gpu=args.gpu,
            topk=args.topk,
            onmt_model_path=args.onmt_model_path,
            onmt_tokenizer=args.onmt_tokenizer,
            save_rows=args.save_rows,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.summary_output:
        summary = {
            **{k: v for k, v in payload.items() if k != "runs"},
            "runs": {name: {k: v for k, v in row.items() if k != "rows"} for name, row in payload["runs"].items()},
        }
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in row.items() if k != "rows"} for name, row in payload["runs"].items()}, indent=2, ensure_ascii=False))


def evaluate_run(
    *,
    products: list[str],
    targets: list[str],
    models: list[str],
    vendor_root: Path,
    gpu: int,
    topk: int,
    onmt_model_path: Path | None,
    onmt_tokenizer: str | None,
    save_rows: bool,
) -> dict[str, Any]:
    provider = ChemEnzyOneStepProposalProvider(
        vendor_root=vendor_root,
        models=tuple(models),
        expansion_topk=int(topk),
        gpu=int(gpu),
        onmt_model_path=onmt_model_path,
        onmt_tokenizer=onmt_tokenizer,
    )
    started = time.monotonic()
    rows = []
    for idx, (product, target) in enumerate(zip(products, targets)):
        error = ""
        candidates: list[dict[str, Any]] = []
        try:
            candidates = provider.predict(product, top_k=int(topk))
        except Exception as exc:  # pragma: no cover - vendor runtime dependent
            error = f"{type(exc).__name__}:{exc}"
        rows.append(_score_row(idx, product, target, candidates, topk=topk, error=error))
    elapsed_s = round(time.monotonic() - started, 3)
    summary = _summarize(rows, topk=topk, elapsed_s=elapsed_s)
    return {
        "models": models,
        "summary": summary,
        "load_error": provider.load_error,
        "rows": rows if save_rows else [],
    }


def _score_row(idx: int, product: str, target: str, candidates: list[dict[str, Any]], *, topk: int, error: str) -> dict[str, Any]:
    target_key = canonical_side(target)
    exact_rank = None
    any_rank = None
    preview = []
    for rank, candidate in enumerate(candidates[:topk], 1):
        reaction = str(candidate.get("reaction_smiles") or candidate.get("rxn_smiles") or "")
        lhs = reaction.split(">>", 1)[0] if ">>" in reaction else ".".join(candidate.get("reactant_smiles") or [])
        pred_key = canonical_side(lhs)
        exact = bool(pred_key and pred_key == target_key)
        any_hit = bool(set(pred_key) & set(target_key))
        if exact and exact_rank is None:
            exact_rank = rank
        if any_hit and any_rank is None:
            any_rank = rank
        preview.append(
            {
                "rank": rank,
                "reactants": lhs,
                "score": candidate.get("score"),
                "source": candidate.get("source"),
                "model_full_name": candidate.get("model_full_name"),
                "exact": exact,
                "any_reactant": any_hit,
            }
        )
    return {
        "idx": idx,
        "product": product,
        "target_reactants": target,
        "candidate_count": len(candidates),
        "exact_rank": exact_rank,
        "any_reactant_rank": any_rank,
        "exact_hit": exact_rank is not None,
        "any_reactant_hit": any_rank is not None,
        "error": error,
        "candidates_preview": preview[:10],
    }


def _summarize(rows: list[dict[str, Any]], *, topk: int, elapsed_s: float) -> dict[str, Any]:
    n = len(rows)
    exact = sum(1 for row in rows if row.get("exact_hit"))
    any_hit = sum(1 for row in rows if row.get("any_reactant_hit"))
    candidate_counts = [int(row.get("candidate_count") or 0) for row in rows]
    return {
        "n_examples": n,
        "topk": topk,
        "nonempty": sum(1 for value in candidate_counts if value > 0),
        "exact_hit": exact,
        "exact_hit_rate": round(exact / max(n, 1), 6),
        "any_reactant_hit": any_hit,
        "any_reactant_hit_rate": round(any_hit / max(n, 1), 6),
        "zero_candidate": sum(1 for value in candidate_counts if value == 0),
        "avg_candidate_count": round(sum(candidate_counts) / max(n, 1), 6),
        "errors": sum(1 for row in rows if row.get("error")),
        "elapsed_s": elapsed_s,
        "avg_elapsed_s_per_example": round(elapsed_s / max(n, 1), 6),
        "candidate_count_histogram": dict(Counter(candidate_counts).most_common()),
    }


def _read_lines(path: Path, *, limit: int | None) -> list[str]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(text.replace(" ", ""))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _parse_run(raw: str) -> tuple[str, list[str]]:
    if "=" not in raw:
        raise ValueError("--run must be NAME=model1,model2")
    name, model_text = raw.split("=", 1)
    models = [item.strip() for item in model_text.split(",") if item.strip()]
    if not name.strip() or not models:
        raise ValueError("--run must include a nonempty name and at least one model")
    return name.strip(), models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--tgt", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/ChemEnzyRetroPlanner"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--onmt-model-path", type=Path)
    parser.add_argument("--onmt-tokenizer", choices=["char", "token", "pretokenized"])
    parser.add_argument("--save-rows", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
