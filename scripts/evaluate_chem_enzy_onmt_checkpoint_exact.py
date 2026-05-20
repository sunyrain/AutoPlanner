#!/usr/bin/env python
"""Evaluate ChemEnzy vendored OpenNMT checkpoints by reactant exact recall."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rdkit import RDLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side


DEFAULT_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")


def evaluate_checkpoint(
    *,
    model_path: Path,
    src_path: Path,
    tgt_path: Path,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
    beam_size: int = 5,
    topk: int = 5,
    batch_size: int = 64,
    device: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    _load_onmt(vendor_root)
    from onmt.bin.translate import load_model, run_batch_samples  # type: ignore

    products = _read_tokenized_smiles(src_path, limit=limit)
    targets = _read_tokenized_smiles(tgt_path, limit=limit)
    if len(products) != len(targets):
        raise ValueError(f"src/tgt length mismatch: {len(products)} != {len(targets)}")

    started = time.monotonic()
    opt, translator = load_model([str(model_path)], beam_size, topk, int(device), "char")
    result = run_batch_samples(translator, opt, products, int(batch_size))
    predictions = list(result.get("reactants") or [])
    scores = list(result.get("scores") or [])
    rows = []
    top1_exact = 0
    topk_exact = 0
    nonempty = 0
    for idx, (product, target) in enumerate(zip(products, targets)):
        pred_list = list(predictions[idx] if idx < len(predictions) else [])
        score_list = list(scores[idx] if idx < len(scores) else [])
        target_key = _side_key(target)
        pred_keys = [_side_key(pred) for pred in pred_list]
        if pred_list:
            nonempty += 1
        is_top1 = bool(pred_keys and pred_keys[0] == target_key)
        is_topk = target_key in pred_keys[:topk]
        top1_exact += int(is_top1)
        topk_exact += int(is_topk)
        rows.append({
            "idx": idx,
            "product": product,
            "target_reactants": target,
            "predictions": pred_list[:topk],
            "scores": score_list[:topk],
            "top1_exact": is_top1,
            f"top{topk}_exact": is_topk,
        })

    n_examples = len(products)
    return {
        "model_path": str(model_path),
        "src_path": str(src_path),
        "tgt_path": str(tgt_path),
        "beam_size": beam_size,
        "topk": topk,
        "batch_size": batch_size,
        "device": device,
        "limit": limit,
        "n_examples": n_examples,
        "nonempty": nonempty,
        "top1_exact": top1_exact,
        f"top{topk}_exact": topk_exact,
        "top1_rate": round(top1_exact / max(n_examples, 1), 6),
        f"top{topk}_rate": round(topk_exact / max(n_examples, 1), 6),
        "elapsed_s": round(time.monotonic() - started, 3),
        "rows": rows,
    }


def _load_onmt(vendor_root: Path) -> None:
    onmt_root = vendor_root / "retro_planner" / "packages" / "onmt"
    if str(onmt_root.resolve()) not in sys.path:
        sys.path.insert(0, str(onmt_root.resolve()))


def _read_tokenized_smiles(path: Path, *, limit: int | None = None) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                out.append(text.replace(" ", ""))
            if limit is not None and len(out) >= int(limit):
                break
    return out


def _side_key(side: str) -> tuple[str, ...]:
    return canonical_side(side)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, help="Checkpoint path. Repeat to compare checkpoints.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--vendor-root", default=str(DEFAULT_VENDOR_ROOT))
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-output")
    args = ap.parse_args()

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [
            evaluate_checkpoint(
                model_path=Path(model),
                src_path=Path(args.src),
                tgt_path=Path(args.tgt),
                vendor_root=Path(args.vendor_root),
                beam_size=args.beam_size,
                topk=args.topk,
                batch_size=args.batch_size,
                device=args.device,
                limit=args.limit,
            )
            for model in args.model
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        {k: v for k, v in result.items() if k != "rows"}
        for result in payload["results"]
    ]
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
