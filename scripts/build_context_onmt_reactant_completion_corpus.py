#!/usr/bin/env python
"""Build context-ONMT reactant-set completion corpora from clean top-level positives."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side, canonical_smiles  # noqa: E402
from scripts.build_chem_enzy_cascade_onmt_corpus import _source_line, _tokenize_smiles  # noqa: E402
from scripts.onmt_corpus_normalization import canonicalize_side_strict  # noqa: E402


SCHEMA_VERSION = "context_onmt_reactant_completion_corpus.v1"


def build_completion_corpus(
    *,
    corpus_dir: Path,
    output_dir: Path,
    mode: str = "context",
    tokenizer: str = "smiles_token",
    splits: tuple[str, ...] = ("train", "valid", "test"),
    corruption_types: tuple[str, ...] = ("drop_one", "self", "cross_swap", "empty"),
    max_examples_per_split: int | None = None,
    dedupe: bool = True,
) -> dict[str, Any]:
    if mode not in {"plain", "context"}:
        raise ValueError(f"unsupported mode: {mode}")
    if tokenizer not in {"char", "smiles_token"}:
        raise ValueError(f"unsupported tokenizer: {tokenizer}")
    unsupported = sorted(set(corruption_types) - {"drop_one", "self", "cross_swap", "empty"})
    if unsupported:
        raise ValueError(f"unsupported corruption types: {unsupported}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, int] = {}
    counters: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    for split in splits:
        rows = _read_meta(corpus_dir / f"{mode}.{split}.meta.jsonl", max_examples=max_examples_per_split)
        src_rows: list[str] = []
        tgt_rows: list[str] = []
        meta_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            product = str(row.get("product") or "")
            target = str(row.get("target_smiles") or product)
            reactants = [str(item) for item in row.get("reactants") or [] if item]
            chosen_side = canonicalize_side_strict(".".join(reactants))
            product_side = canonicalize_side_strict(product)
            target_side = canonicalize_side_strict(target)
            if not product_side or not chosen_side:
                counters[f"{split}_invalid_clean_side"] += 1
                continue
            product = ".".join(product_side)
            target = ".".join(target_side or product_side)
            reactants = list(chosen_side)
            completion_target = ".".join(reactants)
            for corruption_type in corruption_types:
                given_side = _corrupt_side(
                    row,
                    rows=rows,
                    idx=idx,
                    product=product,
                    reactants=reactants,
                    corruption_type=corruption_type,
                )
                if given_side is None:
                    counters[f"{split}_{corruption_type}_unavailable"] += 1
                    continue
                if given_side == tuple(reactants):
                    counters[f"{split}_{corruption_type}_same_as_target"] += 1
                    continue
                try:
                    src_base = _source_line(
                        mode,
                        {"metadata": {"split": split}, "stage_partition": ["stage_1"], "steps": [{"product": product, "reactants": reactants}]},
                        {"product": product, "reactants": reactants},
                        0,
                        product,
                        target,
                        tokenizer=tokenizer,
                    )
                    given_text = ".".join(given_side)
                    src = f"{src_base} <candidate> {_tokenize_smiles(given_text, tokenizer)}".strip()
                    tgt = _tokenize_smiles(completion_target, tokenizer)
                except ValueError:
                    counters[f"{split}_{corruption_type}_tokenization_failed"] += 1
                    continue
                key = (split, src, tgt)
                if dedupe and key in seen:
                    counters[f"{split}_duplicate_completion"] += 1
                    continue
                seen.add(key)
                src_rows.append(src)
                tgt_rows.append(tgt)
                meta_rows.append({
                    "source": "context_onmt_reactant_completion",
                    "source_example_id": row.get("source_example_id") or row.get("source_row_id"),
                    "source_row_index": idx,
                    "split": split,
                    "mode": mode,
                    "corruption_type": corruption_type,
                    "product": product,
                    "target_smiles": target,
                    "chosen_reactants": completion_target,
                    "given_reactants": given_text,
                    "chosen_reactant_list": reactants,
                    "given_reactant_list": list(given_side),
                    "canonical_reaction": f"{completion_target}>>{product}",
                    "contract": (
                        "Reactant-set completion positive: source contains product plus a rule-perturbed candidate side; "
                        "target is the complete clean reactant side. No expert preference label is used."
                    ),
                })
                counters[f"{split}_{corruption_type}"] += 1
        src_path = output_dir / f"{mode}.{split}.src"
        tgt_path = output_dir / f"{mode}.{split}.tgt"
        meta_path = output_dir / f"{mode}.{split}.meta.jsonl"
        _write_lines(src_path, src_rows)
        _write_lines(tgt_path, tgt_rows)
        _write_jsonl(meta_path, meta_rows)
        files[split] = {"src": str(src_path), "tgt": str(tgt_path), "metadata": str(meta_path), "examples": len(src_rows)}
        split_counts[split] = len(src_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "tokenizer": tokenizer,
        "splits": list(splits),
        "corruption_types": list(corruption_types),
        "max_examples_per_split": max_examples_per_split,
        "dedupe": dedupe,
        "files": files,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "total_examples": sum(split_counts.values()),
            "examples_by_split": split_counts,
            "counts": dict(counters),
            "output_dir": str(output_dir),
        },
        "contract": (
            "Completion corpus only. It is intended for proposal repair/completion experiments, "
            "not for route ranking or live-model promotion by itself."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def _corrupt_side(
    row: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    idx: int,
    product: str,
    reactants: list[str],
    corruption_type: str,
) -> tuple[str, ...] | None:
    if corruption_type == "empty":
        return ()
    if corruption_type == "self":
        return canonical_side(product)
    if corruption_type == "drop_one":
        if len(reactants) < 2:
            return None
        return tuple(reactants[:-1])
    if corruption_type == "cross_swap":
        for offset in range(1, min(len(rows), 50) + 1):
            other = rows[(idx + offset) % len(rows)]
            if canonical_smiles(other.get("product")) == canonical_smiles(product):
                continue
            other_side = canonicalize_side_strict(".".join(str(item) for item in other.get("reactants") or [] if item))
            if other_side:
                return other_side
        return None
    raise ValueError(f"unsupported corruption type: {corruption_type}")


def _read_meta(path: Path, *, max_examples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= int(max_examples):
                break
    return rows


def _write_lines(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Context ONMT Reactant Completion Corpus",
        "",
        f"created_at: `{manifest['created_at']}`",
        "",
        "## Summary",
        "",
        f"- corpus_dir: `{manifest['corpus_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- mode: `{manifest['mode']}`",
        f"- tokenizer: `{manifest['tokenizer']}`",
        f"- total_examples: {manifest['summary']['total_examples']}",
        "",
        "| split | examples |",
        "| --- | ---: |",
    ]
    for split, count in manifest["summary"]["examples_by_split"].items():
        lines.append(f"| `{split}` | {count} |")
    lines.extend(["", "## Counts", "", "| key | count |", "| --- | ---: |"])
    for key, value in sorted((manifest["summary"].get("counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Contract", "", manifest["contract"], ""])
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["plain", "context"], default="context")
    ap.add_argument("--tokenizer", choices=["char", "smiles_token"], default="smiles_token")
    ap.add_argument("--split", choices=["train", "valid", "test"], action="append")
    ap.add_argument("--corruption-type", choices=["drop_one", "self", "cross_swap", "empty"], action="append")
    ap.add_argument("--max-examples-per-split", type=int)
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()
    manifest = build_completion_corpus(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        tokenizer=args.tokenizer,
        splits=tuple(args.split or ["train", "valid", "test"]),
        corruption_types=tuple(args.corruption_type or ["drop_one", "self", "cross_swap", "empty"]),
        max_examples_per_split=args.max_examples_per_split,
        dedupe=not args.no_dedupe,
    )
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
