#!/usr/bin/env python3
"""Extend a ChemEnzy OpenNMT checkpoint vocab for cascade-context corpus tokens.

This is an experimental bridge for context-conditioned supervised training.
The vendored checkpoint uses shared src/tgt vocab and shared encoder/decoder
embeddings, so new context tokens must be appended to the shared vocab and the
encoder embedding, decoder embedding, and generator matrices together. Include
source and target tokens; target-side OOVs would otherwise silently corrupt
reactant generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")
DEFAULT_CHECKPOINT = DEFAULT_VENDOR_ROOT / "retro_planner/packages/onmt/checkpoints/np-like/model_step_100000.pt"
DEFAULT_CONTEXT_CORPUS = Path("results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k_smiles_token")
SCHEMA_VERSION = "chem_enzy_onmt_context_vocab_extension.v1"


def main() -> None:
    args = _parse_args()
    result = extend_checkpoint_vocab(
        checkpoint=args.checkpoint,
        corpus_dir=args.corpus_dir,
        output_checkpoint=args.output_checkpoint,
        vendor_root=args.vendor_root,
        splits=args.split,
        modes=args.mode,
        sides=args.side,
        min_count=args.min_count,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def extend_checkpoint_vocab(
    *,
    checkpoint: Path,
    corpus_dir: Path,
    output_checkpoint: Path,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
    splits: list[str] | tuple[str, ...] = ("train", "valid", "test"),
    modes: list[str] | tuple[str, ...] = ("context",),
    sides: list[str] | tuple[str, ...] = ("src", "tgt"),
    min_count: int = 1,
    dry_run: bool = False,
    seed: int = 17,
) -> dict[str, Any]:
    _load_onmt(vendor_root)
    import torch

    checkpoint = Path(checkpoint)
    corpus_dir = Path(corpus_dir)
    output_checkpoint = Path(output_checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    vocab = _shared_vocab(ckpt)
    old_size = len(vocab.itos)
    counts = _corpus_counts(corpus_dir, splits=splits, modes=modes, sides=sides)
    new_tokens = [token for token, count in counts.items() if count >= min_count and token not in vocab.stoi]
    new_tokens = sorted(new_tokens, key=lambda token: (-counts[token], token))

    row_keys = [
        "encoder.embeddings.make_embedding.emb_luts.0.weight",
        "decoder.embeddings.make_embedding.emb_luts.0.weight",
    ]
    generator_weight = _first_existing_key(ckpt["generator"], ("0.weight", "generator.0.weight"))
    generator_bias = _first_existing_key(ckpt["generator"], ("0.bias", "generator.0.bias"))
    expected_old = {
        "encoder_embedding": tuple(ckpt["model"][row_keys[0]].shape),
        "decoder_embedding": tuple(ckpt["model"][row_keys[1]].shape),
        "generator_weight": tuple(ckpt["generator"][generator_weight].shape),
        "generator_bias": tuple(ckpt["generator"][generator_bias].shape),
    }
    if not new_tokens:
        status = "no_new_tokens"
    elif dry_run:
        status = "planned_not_written"
    else:
        _append_tokens(vocab, new_tokens)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        for key in row_keys:
            ckpt["model"][key] = _extend_matrix(ckpt["model"][key], len(new_tokens), generator)
        ckpt["generator"][generator_weight] = _extend_matrix(ckpt["generator"][generator_weight], len(new_tokens), generator)
        ckpt["generator"][generator_bias] = _extend_bias(ckpt["generator"][generator_bias], len(new_tokens))
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, output_checkpoint)
        status = "written"

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": str(checkpoint),
        "corpus_dir": str(corpus_dir),
        "output_checkpoint": str(output_checkpoint),
        "dry_run": dry_run,
        "splits": list(splits),
        "min_count": min_count,
        "old_vocab_size": old_size,
        "new_vocab_size": old_size + len(new_tokens),
        "n_new_tokens": len(new_tokens),
        "new_tokens_preview": [{"token": token, "count": counts[token]} for token in new_tokens[:50]],
        "expected_old_shapes": expected_old,
        "expected_new_shapes": {
            "encoder_embedding": (old_size + len(new_tokens), expected_old["encoder_embedding"][1]),
            "decoder_embedding": (old_size + len(new_tokens), expected_old["decoder_embedding"][1]),
            "generator_weight": (old_size + len(new_tokens), expected_old["generator_weight"][1]),
            "generator_bias": (old_size + len(new_tokens),),
        },
        "status": status,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "old_vocab_size": old_size,
            "new_vocab_size": old_size + len(new_tokens),
            "n_new_tokens": len(new_tokens),
            "output_checkpoint": str(output_checkpoint),
            "dry_run": dry_run,
        },
        "modes": list(modes),
        "sides": list(sides),
        "contract": (
            "Experimental checkpoint surgery only. A written checkpoint must pass source/target vocab audit, "
            "context preprocess/train smoke, exact-recall evaluation, and route-level evaluation before being "
            "used as a proposal model."
        ),
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ChemEnzy ONMT Context Vocab Extension",
        "",
        f"生成时间：{result['created_at']}",
        "",
        "## Summary",
        "",
        f"- status: `{result['status']}`",
        f"- old_vocab_size: {result['old_vocab_size']}",
        f"- new_vocab_size: {result['new_vocab_size']}",
        f"- n_new_tokens: {result['n_new_tokens']}",
        f"- output_checkpoint: `{result['output_checkpoint']}`",
        "",
        "## New Tokens Preview",
        "",
        "| token | count |",
        "| --- | ---: |",
    ]
    for row in result["new_tokens_preview"][:30]:
        lines.append(f"| `{row['token']}` | {row['count']} |")
    lines.extend([
        "",
        "## Contract",
        "",
        result["contract"],
        "",
    ])
    return "\n".join(lines)


def _corpus_counts(
    corpus_dir: Path,
    *,
    splits: list[str] | tuple[str, ...],
    modes: list[str] | tuple[str, ...],
    sides: list[str] | tuple[str, ...],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for mode in modes:
        if mode not in {"plain", "context", "ec_context"}:
            raise ValueError(f"unsupported mode: {mode}")
        for side in sides:
            if side not in {"src", "tgt"}:
                raise ValueError(f"unsupported side: {side}")
            for split in splits:
                path = corpus_dir / f"{mode}.{split}.{side}"
                if not path.exists():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    counts.update(token for token in line.split() if token)
    return counts


def _shared_vocab(ckpt: dict[str, Any]) -> Any:
    src_vocab = ckpt["vocab"]["src"].fields[0][1].vocab
    tgt_vocab = ckpt["vocab"]["tgt"].fields[0][1].vocab
    if src_vocab is not tgt_vocab and src_vocab.itos != tgt_vocab.itos:
        raise ValueError("checkpoint does not use a compatible shared src/tgt vocab")
    return src_vocab


def _append_tokens(vocab: Any, tokens: list[str]) -> None:
    for token in tokens:
        if token in vocab.stoi:
            continue
        vocab.stoi[token] = len(vocab.itos)
        vocab.itos.append(token)


def _first_existing_key(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in mapping:
            return key
    raise KeyError(f"none of the expected keys exists: {keys}")


def _extend_matrix(matrix: Any, n_rows: int, generator: Any) -> Any:
    import torch

    if n_rows <= 0:
        return matrix
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, keepdim=True).mean().clamp(min=1e-6)
    noise = torch.randn((n_rows, matrix.shape[1]), generator=generator, dtype=matrix.dtype) * (std * 0.01)
    return torch.cat([matrix, mean.repeat(n_rows, 1) + noise], dim=0)


def _extend_bias(bias: Any, n_rows: int) -> Any:
    import torch

    if n_rows <= 0:
        return bias
    fill = bias.mean().reshape(1).repeat(n_rows).to(dtype=bias.dtype)
    return torch.cat([bias, fill], dim=0)


def _load_onmt(vendor_root: Path) -> None:
    onmt_root = Path(vendor_root) / "retro_planner" / "packages" / "onmt"
    if onmt_root.exists() and str(onmt_root.resolve()) not in sys.path:
        sys.path.insert(0, str(onmt_root.resolve()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CONTEXT_CORPUS)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--split", choices=["train", "valid", "test"], action="append", default=["train", "valid", "test"])
    parser.add_argument("--mode", choices=["plain", "context", "ec_context"], action="append", default=["context"])
    parser.add_argument("--side", choices=["src", "tgt"], action="append", default=["src", "tgt"])
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
