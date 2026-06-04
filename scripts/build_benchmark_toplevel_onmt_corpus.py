#!/usr/bin/env python
"""Build a benchmark-style top-level ONMT corpus from GT target-product steps."""
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
from scripts.build_chem_enzy_cascade_onmt_corpus import (  # noqa: E402
    _source_line,
    _step_product,
    _step_reactants,
    _tokenize_smiles,
)
from scripts.onmt_corpus_normalization import canonicalize_product_and_reactants  # noqa: E402


SCHEMA_VERSION = "benchmark_toplevel_onmt_corpus.v1"


def build_benchmark_toplevel_corpus(
    *,
    benchmark_path: Path,
    output_dir: Path,
    modes: list[str],
    tokenizer: str = "smiles_token",
    split_policy: str = "index_70_15_15",
    dedupe: bool = True,
    max_targets: int | None = None,
    canonicalize_training_smiles: bool = True,
) -> dict[str, Any]:
    if "both" in modes:
        modes = ["plain", "context"]
    modes = sorted(set(modes))
    if set(modes) - {"plain", "context"}:
        raise ValueError(f"unsupported modes: {sorted(set(modes) - {'plain', 'context'})}")
    if tokenizer not in {"char", "smiles_token"}:
        raise ValueError(f"unsupported tokenizer: {tokenizer}")
    if split_policy not in {"index_70_15_15", "all_train"}:
        raise ValueError(f"unsupported split_policy: {split_policy}")

    targets = _load_benchmark(benchmark_path, max_targets=max_targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_mode_split: dict[str, dict[str, list[tuple[str, str, dict[str, Any]]]]] = {
        mode: defaultdict(list) for mode in modes
    }
    skipped = Counter()
    seen = set()
    emitted_targets: set[int] = set()
    for target_idx, target_row in enumerate(targets):
        target = str(target_row.get("target_smiles") or target_row.get("smiles") or "")
        gt_steps = _target_product_gt_steps(target_row, target)
        if not gt_steps:
            skipped["target_without_top_level_gt_step"] += 1
            continue
        split = _split_for_index(target_idx, len(targets), policy=split_policy)
        for gt_step_idx, raw_step in enumerate(gt_steps):
            step = dict(raw_step)
            product = _step_product(step) or target
            reactants = _step_reactants(step)
            if not product or not reactants:
                skipped["step_missing_product_or_reactants"] += 1
                continue
            if _is_self_reaction(product, reactants):
                skipped["self_reaction_step"] += 1
                continue
            raw_product = product
            raw_reactants = list(reactants)
            if canonicalize_training_smiles:
                product, reactants = canonicalize_product_and_reactants(product, reactants)
                if not product or not reactants:
                    skipped["strict_canonicalization_failed"] += 1
                    continue
            cascade = {
                "metadata": {"split": split},
                "stage_partition": ["stage_1"],
                "steps": [step],
            }
            reactant_line = ".".join(reactants)
            metadata = {
                "source": "benchmark_top_level_gt",
                "source_example_id": f"benchmark_{target_idx:04d}_{gt_step_idx:02d}",
                "source_target_index": target_idx,
                "target_smiles": target,
                "route_index": target_idx,
                "step_index": 0,
                "benchmark_gt_step_index": _gt_step_position(target_row, raw_step),
                "split": split,
                "stage": "stage_1",
                "product": product,
                "reactants": reactants,
                "raw_product": raw_product,
                "raw_reactants": raw_reactants,
                "rxn_smiles": step.get("rxn_smiles"),
                "canonical_reaction": _canonical_reaction_from_sides(reactants, product),
                "route_domain": target_row.get("route_domain"),
                "doi": target_row.get("doi"),
                "cascade_id": target_row.get("cascade_id"),
                "transformation": step.get("transformation"),
                "contract": (
                    "Benchmark-style top-level positive for proposal diagnosis. "
                    "Use as a held-out diagnostic/targeted experiment, not as an expert-label route preference."
                ),
            }
            for mode in modes:
                try:
                    src = _source_line(mode, cascade, step, 0, product, target, tokenizer=tokenizer)
                    tgt = _tokenize_smiles(reactant_line, tokenizer)
                except ValueError:
                    skipped[f"{mode}_tokenization_failed"] += 1
                    continue
                key = (mode, split, src, tgt)
                if dedupe and key in seen:
                    skipped["duplicate_step"] += 1
                    continue
                seen.add(key)
                rows_by_mode_split[mode][split].append((src, tgt, metadata))
                emitted_targets.add(target_idx)

    files: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for mode in modes:
        files[mode] = {}
        counts[mode] = {}
        for split in ("train", "valid", "test"):
            rows = rows_by_mode_split[mode].get(split, [])
            src_path = output_dir / f"{mode}.{split}.src"
            tgt_path = output_dir / f"{mode}.{split}.tgt"
            meta_path = output_dir / f"{mode}.{split}.meta.jsonl"
            _write_lines(src_path, [row[0] for row in rows])
            _write_lines(tgt_path, [row[1] for row in rows])
            _write_jsonl(meta_path, [row[2] for row in rows])
            files[mode][split] = {
                "src": str(src_path),
                "tgt": str(tgt_path),
                "metadata": str(meta_path),
                "examples": len(rows),
            }
            counts[mode][split] = len(rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "benchmark": str(benchmark_path),
        "output_dir": str(output_dir),
        "modes": modes,
        "tokenizer": tokenizer,
        "split_policy": split_policy,
        "dedupe": dedupe,
        "canonicalize_training_smiles": canonicalize_training_smiles,
        "n_targets": len(targets),
        "n_emitted_targets": len(emitted_targets),
        "files": files,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "n_targets": len(targets),
            "n_emitted_targets": len(emitted_targets),
            "modes": modes,
            "tokenizer": tokenizer,
            "split_policy": split_policy,
            "examples_by_mode_split": counts,
            "total_examples": {mode: sum(counts[mode].values()) for mode in modes},
            "skipped": dict(skipped),
            "output_dir": str(output_dir),
        },
        "contract": (
            "Benchmark top-level GT corpus for proposal-generation diagnosis. "
            "Do not treat benchmark-derived positives as expert preference labels or live-model promotion evidence."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def _load_benchmark(path: Path, *, max_targets: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("targets") if isinstance(payload, dict) else payload
    rows = [row for row in rows or [] if isinstance(row, dict)]
    if max_targets is not None:
        rows = rows[: max(0, int(max_targets))]
    return rows


def _target_product_gt_steps(row: dict[str, Any], target: str) -> list[dict[str, Any]]:
    target_side = canonical_side(target)
    out = []
    for step in row.get("gt_route") or []:
        rxn = str(step.get("rxn_smiles") or "")
        if ">>" not in rxn:
            continue
        rhs = rxn.split(">>", 1)[1]
        if canonical_side(rhs) == target_side:
            out.append(step)
    return out


def _gt_step_position(row: dict[str, Any], step: dict[str, Any]) -> int | None:
    for idx, item in enumerate(row.get("gt_route") or []):
        if item is step or item == step:
            return idx
    return None


def _split_for_index(index: int, total: int, *, policy: str) -> str:
    if policy == "all_train":
        return "train"
    train_cut = int(total * 0.70)
    valid_cut = int(total * 0.85)
    if index < train_cut:
        return "train"
    if index < valid_cut:
        return "valid"
    return "test"


def _is_self_reaction(product: str, reactants: list[str]) -> bool:
    product_key = canonical_smiles(product)
    if not product_key:
        return False
    return any(canonical_smiles(reactant) == product_key for reactant in reactants if reactant)


def _canonical_reaction_from_sides(reactants: list[str], product: str) -> str:
    lhs = ".".join(canonical_side(".".join(reactants)))
    rhs = ".".join(canonical_side(product))
    return f"{lhs}>>{rhs}" if lhs and rhs else ""


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Top-level ONMT Corpus",
        "",
        f"created_at: `{manifest['created_at']}`",
        "",
        "## Summary",
        "",
        f"- benchmark: `{manifest['benchmark']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- n_targets: {manifest['summary']['n_targets']}",
        f"- n_emitted_targets: {manifest['summary']['n_emitted_targets']}",
        f"- tokenizer: `{manifest['tokenizer']}`",
        f"- split_policy: `{manifest['split_policy']}`",
        f"- canonicalize_training_smiles: {manifest.get('canonicalize_training_smiles', False)}",
        "",
        "| mode | train | valid | test | total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    counts = manifest["summary"]["examples_by_mode_split"]
    totals = manifest["summary"]["total_examples"]
    for mode in manifest["modes"]:
        row = counts[mode]
        lines.append(f"| {mode} | {row.get('train', 0)} | {row.get('valid', 0)} | {row.get('test', 0)} | {totals[mode]} |")
    lines.extend([
        "",
        "## Skipped",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ])
    for key, value in sorted((manifest["summary"].get("skipped") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Contract", "", manifest["contract"], ""])
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["plain", "context", "both"], nargs="+", default=["both"])
    ap.add_argument("--tokenizer", choices=["char", "smiles_token"], default="smiles_token")
    ap.add_argument("--split-policy", choices=["index_70_15_15", "all_train"], default="index_70_15_15")
    ap.add_argument("--max-targets", type=int)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--no-canonicalize-training-smiles", action="store_true")
    args = ap.parse_args()
    manifest = build_benchmark_toplevel_corpus(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        modes=args.mode,
        tokenizer=args.tokenizer,
        split_policy=args.split_policy,
        max_targets=args.max_targets,
        dedupe=not args.no_dedupe,
        canonicalize_training_smiles=not args.no_canonicalize_training_smiles,
    )
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
