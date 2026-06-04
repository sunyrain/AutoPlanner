#!/usr/bin/env python3
"""Audit ChemEnzy OpenNMT checkpoint vocab coverage for cascade-context corpora.

The native ChemEnzy ONMT checkpoint loads its own checkpoint vocabulary when
``-train_from`` is used. A freshly preprocessed context corpus can therefore
create a larger ``*.vocab.pt`` without actually expanding the checkpoint
embeddings. This script makes that failure mode explicit before context-mode
training is attempted.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = Path("results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k")
DEFAULT_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")
DEFAULT_CHECKPOINT = DEFAULT_VENDOR_ROOT / "retro_planner/packages/onmt/checkpoints/np-like/model_step_100000.pt"
SCHEMA_VERSION = "chem_enzy_onmt_context_vocab_audit.v1"


def main() -> None:
    args = _parse_args()
    result = audit_context_vocab(
        corpus_dir=args.corpus_dir,
        checkpoint=args.checkpoint,
        vendor_root=args.vendor_root,
        data_vocab=args.data_vocab,
        modes=args.mode,
        splits=args.split,
        frequent_limit=args.frequent_limit,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def audit_context_vocab(
    *,
    corpus_dir: Path,
    checkpoint: Path | None = DEFAULT_CHECKPOINT,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
    data_vocab: Path | None = None,
    modes: Iterable[str] = ("plain", "context"),
    splits: Iterable[str] = ("train", "valid", "test"),
    frequent_limit: int = 20,
    checkpoint_src_tokens: Iterable[str] | None = None,
    checkpoint_tgt_tokens: Iterable[str] | None = None,
    data_src_tokens: Iterable[str] | None = None,
    data_tgt_tokens: Iterable[str] | None = None,
) -> dict[str, Any]:
    corpus_dir = Path(corpus_dir)
    modes = list(modes)
    splits = list(splits)
    unknown_modes = sorted(set(modes) - {"plain", "context"})
    if unknown_modes:
        raise ValueError(f"unsupported modes: {unknown_modes}")

    checkpoint_tokens = {
        "src": set(checkpoint_src_tokens or []),
        "tgt": set(checkpoint_tgt_tokens or []),
    }
    if checkpoint is not None and (not checkpoint_tokens["src"] or not checkpoint_tokens["tgt"]):
        loaded = _load_vocab_tokens(checkpoint, vendor_root=vendor_root)
        checkpoint_tokens["src"] = checkpoint_tokens["src"] or loaded["src"]
        checkpoint_tokens["tgt"] = checkpoint_tokens["tgt"] or loaded["tgt"]

    data_tokens = {
        "src": set(data_src_tokens or []),
        "tgt": set(data_tgt_tokens or []),
    }
    if data_vocab is not None and data_vocab.exists() and (not data_tokens["src"] or not data_tokens["tgt"]):
        loaded = _load_vocab_tokens(data_vocab, vendor_root=vendor_root)
        data_tokens["src"] = data_tokens["src"] or loaded["src"]
        data_tokens["tgt"] = data_tokens["tgt"] or loaded["tgt"]

    corpus_stats: dict[str, Any] = {}
    for mode in modes:
        corpus_stats[mode] = {}
        for side in ("src", "tgt"):
            counts = _counts_for(corpus_dir, mode=mode, side=side, splits=splits)
            corpus_stats[mode][side] = {
                "corpus": _coverage_row(counts, checkpoint_tokens[side], frequent_limit=frequent_limit),
                "data_vocab": _coverage_row(counts, data_tokens[side], frequent_limit=frequent_limit) if data_tokens[side] else None,
            }

    context_src = corpus_stats.get("context", {}).get("src", {}).get("corpus", {})
    context_tgt = corpus_stats.get("context", {}).get("tgt", {}).get("corpus", {})
    plain_src = corpus_stats.get("plain", {}).get("src", {}).get("corpus", {})
    plain_tgt = corpus_stats.get("plain", {}).get("tgt", {}).get("corpus", {})
    decision = _decision(
        context_src=context_src,
        context_tgt=context_tgt,
        plain_src=plain_src,
        plain_tgt=plain_tgt,
        modes=modes,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "corpus_dir": str(corpus_dir),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "data_vocab": str(data_vocab) if data_vocab is not None else None,
        "modes": modes,
        "splits": splits,
        "checkpoint_vocab_sizes": {side: len(tokens) for side, tokens in checkpoint_tokens.items()},
        "data_vocab_sizes": {side: len(tokens) for side, tokens in data_tokens.items() if tokens},
        "stats": corpus_stats,
        "train_from_vocab_behavior": (
            "Vendored onmt/bin/train.py loads checkpoint['vocab'] when -train_from is set; "
            "the newly preprocessed data .vocab.pt does not expand the checkpoint embeddings."
        ),
        "decision": decision,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "decision": decision["status"],
            "plain_src_oov_rate": plain_src.get("oov_rate"),
            "plain_tgt_oov_rate": plain_tgt.get("oov_rate"),
            "context_src_oov_rate": context_src.get("oov_rate"),
            "context_tgt_oov_rate": context_tgt.get("oov_rate"),
            "context_src_oov_unique": context_src.get("oov_unique_tokens"),
            "context_tgt_oov_unique": context_tgt.get("oov_unique_tokens"),
            "checkpoint_src_vocab_size": len(checkpoint_tokens["src"]),
            "checkpoint_tgt_vocab_size": len(checkpoint_tokens["tgt"]),
            "direct_context_train_from_checkpoint_ok": decision["direct_context_train_from_checkpoint_ok"],
        },
        "contract": (
            "Vocab audit only. It does not train ChemEnzy. A direct context-mode continue-train is blocked "
            "unless context source tokens are representable by the checkpoint vocabulary or model/vocab expansion is implemented."
        ),
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ChemEnzy ONMT Context Vocab Audit",
        "",
        f"生成时间：{result['created_at']}",
        "",
        "## Decision",
        "",
        f"- status: `{result['decision']['status']}`",
        f"- direct_context_train_from_checkpoint_ok: {result['decision']['direct_context_train_from_checkpoint_ok']}",
        f"- reason: {result['decision']['reason']}",
        "",
        "## Vocab Sizes",
        "",
        f"- checkpoint src: {result['checkpoint_vocab_sizes'].get('src')}",
        f"- checkpoint tgt: {result['checkpoint_vocab_sizes'].get('tgt')}",
        f"- data vocab: {result.get('data_vocab') or 'not provided'}",
        "",
        "## Coverage",
        "",
        "| mode | side | total tokens | unique | OOV tokens | OOV unique | OOV rate | frequent OOV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for mode, mode_stats in result["stats"].items():
        for side, rows in mode_stats.items():
            row = rows["corpus"]
            frequent = ", ".join(f"{item['token']}:{item['count']}" for item in row["frequent_oov"][:8])
            lines.append(
                f"| {mode} | {side} | {row['total_tokens']} | {row['unique_tokens']} | "
                f"{row['oov_tokens']} | {row['oov_unique_tokens']} | {row['oov_rate']} | {frequent or '-'} |"
            )
    lines.extend([
        "",
        "## Important ONMT Behavior",
        "",
        result["train_from_vocab_behavior"],
        "",
        "## Next Step",
        "",
        "- Do not train the current context corpus directly from the native checkpoint.",
        "- Either implement checkpoint embedding/vocab expansion, or redesign context encoding so every source token is representable by the native checkpoint vocabulary.",
        "- Until then, keep proposal-side work on native ChemEnzy plus verifier/search integration.",
        "",
    ])
    return "\n".join(lines)


def _counts_for(corpus_dir: Path, *, mode: str, side: str, splits: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for split in splits:
        path = corpus_dir / f"{mode}.{split}.{side}"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                counts.update(token for token in line.strip().split() if token)
    return counts


def _coverage_row(counts: Counter[str], vocab_tokens: set[str], *, frequent_limit: int) -> dict[str, Any]:
    total = sum(counts.values())
    unique = len(counts)
    oov_counts = Counter({token: count for token, count in counts.items() if token not in vocab_tokens})
    oov_total = sum(oov_counts.values())
    return {
        "total_tokens": total,
        "unique_tokens": unique,
        "covered_tokens": total - oov_total,
        "oov_tokens": oov_total,
        "oov_unique_tokens": len(oov_counts),
        "oov_rate": round(oov_total / max(total, 1), 6),
        "frequent_oov": [
            {"token": token, "count": count}
            for token, count in oov_counts.most_common(frequent_limit)
        ],
    }


def _decision(
    *,
    context_src: dict[str, Any],
    context_tgt: dict[str, Any],
    plain_src: dict[str, Any],
    plain_tgt: dict[str, Any],
    modes: list[str],
) -> dict[str, Any]:
    context_oov = float(context_src.get("oov_rate") or 0.0)
    context_tgt_oov = float(context_tgt.get("oov_rate") or 0.0)
    plain_oov = float(plain_src.get("oov_rate") or 0.0)
    plain_tgt_oov = float(plain_tgt.get("oov_rate") or 0.0)
    if "context" in set(modes) and (context_oov > 0 or context_tgt_oov > 0):
        return {
            "status": "blocked_context_direct_train_from_checkpoint",
            "direct_context_train_from_checkpoint_ok": False,
            "reason": (
                f"context source/target OOV rates against checkpoint vocab are {context_oov}/{context_tgt_oov}; "
                "native train_from will keep checkpoint vocab instead of expanding source/target embeddings."
            ),
        }
    if "plain" in set(modes) and (plain_oov > 0 or plain_tgt_oov > 0):
        return {
            "status": "warning_plain_oov_detected",
            "direct_context_train_from_checkpoint_ok": False,
            "reason": (
                f"plain source/target OOV rates are {plain_oov}/{plain_tgt_oov}; "
                "checkpoint/corpus tokenization should be reconciled first."
            ),
        }
    return {
        "status": "compatible_by_vocab_audit",
        "direct_context_train_from_checkpoint_ok": True,
        "reason": "context source tokens are covered by the checkpoint source vocabulary.",
    }


def _load_vocab_tokens(path: Path, *, vendor_root: Path) -> dict[str, set[str]]:
    _load_onmt(vendor_root)
    import torch

    obj = torch.load(path, map_location="cpu")
    return {
        "src": _extract_tokens(obj, "src"),
        "tgt": _extract_tokens(obj, "tgt"),
    }


def _load_onmt(vendor_root: Path) -> None:
    onmt_root = Path(vendor_root) / "retro_planner" / "packages" / "onmt"
    if onmt_root.exists() and str(onmt_root.resolve()) not in sys.path:
        sys.path.insert(0, str(onmt_root.resolve()))


def _extract_tokens(obj: Any, side: str) -> set[str]:
    if isinstance(obj, dict):
        if "vocab" in obj and isinstance(obj["vocab"], dict) and side in obj["vocab"]:
            return _extract_tokens(obj["vocab"][side], side)
        if side in obj:
            return _extract_tokens(obj[side], side)
    vocab = getattr(obj, "vocab", None)
    if vocab is not None and hasattr(vocab, "itos"):
        return set(vocab.itos)
    fields = getattr(obj, "fields", None)
    if fields:
        for name, field in fields:
            if name == side:
                return _extract_tokens(field, side)
        return _extract_tokens(fields[0][1], side)
    if hasattr(obj, "itos"):
        return set(obj.itos)
    if hasattr(obj, "stoi"):
        return set(obj.stoi.keys())
    raise TypeError(f"cannot extract {side} vocab tokens from {type(obj)!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--data-vocab", type=Path)
    parser.add_argument("--mode", choices=["plain", "context"], action="append", default=["plain", "context"])
    parser.add_argument("--split", choices=["train", "valid", "test"], action="append", default=["train", "valid", "test"])
    parser.add_argument("--frequent-limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
