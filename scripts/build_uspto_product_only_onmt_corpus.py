#!/usr/bin/env python3
"""Build a product-only chemical OpenNMT corpus from ChemEnzy USPTO-full data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_chem_enzy_cascade_onmt_corpus import _tokenize_smiles  # noqa: E402
from scripts.onmt_corpus_normalization import canonicalize_product_and_reactants  # noqa: E402


SCHEMA_VERSION = "uspto_product_only_onmt_corpus.v1"
DEFAULT_USPTO_CSV = Path(
    "vendor/ChemEnzyRetroPlanner/retro_planner/packages/graph_retrosyn/"
    "graph_retrosyn/data/raw/USPTO-remapped_tpl_prod_react.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/shared/uspto_product_only_chem_onmt_20260530/corpus")
DEFAULT_BENCHMARK = Path(
    "results/shared/chemical_step_template_relevance_ab_20260530/"
    "uspto190_first128/uspto190_first128.json"
)


def build_corpus(
    *,
    uspto_csv: Path = DEFAULT_USPTO_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tokenizer: str = "char",
    canonicalize_training_smiles: bool = True,
    dedupe: bool = True,
    max_rows: int | None = None,
    benchmark_jsons: list[Path] | None = None,
) -> dict[str, Any]:
    if tokenizer not in {"char", "smiles_token"}:
        raise ValueError(f"unsupported tokenizer: {tokenizer}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_jsons = [Path(path) for path in (benchmark_jsons or []) if Path(path).exists()]

    handles: dict[tuple[str, str], Any] = {}
    meta_handles: dict[str, Any] = {}
    files: dict[str, dict[str, str]] = {}
    for split in ("train", "valid", "test"):
        files[split] = {
            "src": str(output_dir / f"plain.{split}.src"),
            "tgt": str(output_dir / f"plain.{split}.tgt"),
            "metadata": str(output_dir / f"plain.{split}.meta.jsonl"),
        }
        handles[(split, "src")] = (output_dir / f"plain.{split}.src").open("w", encoding="utf-8")
        handles[(split, "tgt")] = (output_dir / f"plain.{split}.tgt").open("w", encoding="utf-8")
        meta_handles[split] = (output_dir / f"plain.{split}.meta.jsonl").open("w", encoding="utf-8")

    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    train_pair_keys: set[tuple[str, str]] = set()
    train_products: set[str] = set()
    product_split_counts: dict[str, Counter[str]] = defaultdict(Counter)

    try:
        with Path(uspto_csv).open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row_idx, row in enumerate(reader):
                if max_rows is not None and row_idx >= int(max_rows):
                    break
                raw_product = str(row.get("prod_smiles") or "").strip()
                raw_reactants = str(row.get("react_smiles") or "").strip()
                if not raw_product or not raw_reactants:
                    skipped["missing_product_or_reactants"] += 1
                    continue
                product = raw_product
                reactants = [part for part in raw_reactants.split(".") if part]
                if canonicalize_training_smiles:
                    product, reactants = canonicalize_product_and_reactants(product, reactants)
                    if not product or not reactants:
                        skipped["strict_canonicalization_failed"] += 1
                        continue
                reactant_side = ".".join(reactants)
                split = _split_for_row(row_idx)
                try:
                    src = _tokenize_smiles(product, tokenizer)
                    tgt = _tokenize_smiles(reactant_side, tokenizer)
                except ValueError:
                    skipped["tokenization_failed"] += 1
                    continue
                key = (split, src, tgt)
                if dedupe and key in seen:
                    skipped["duplicate_within_split"] += 1
                    continue
                seen.add(key)
                handles[(split, "src")].write(src + "\n")
                handles[(split, "tgt")].write(tgt + "\n")
                metadata = {
                    "source": "chemenzy_uspto_full_remapped",
                    "row_idx": row_idx,
                    "split": split,
                    "product": product,
                    "reactants": reactants,
                    "raw_product": raw_product,
                    "raw_reactants": raw_reactants,
                    "reaction_smiles": reactant_side + f">>{product}",
                    "template_sha1": _sha1(str(row.get("templates") or "")),
                }
                meta_handles[split].write(json.dumps(metadata, ensure_ascii=False) + "\n")
                counts[split] += 1
                source_counts["chemenzy_uspto_full_remapped"] += 1
                product_split_counts[product][split] += 1
                if split == "train":
                    train_products.add(product)
                    train_pair_keys.add((product, reactant_side))
    finally:
        for handle in handles.values():
            handle.close()
        for handle in meta_handles.values():
            handle.close()

    benchmark_reports = []
    benchmark_files = []
    for benchmark_path in benchmark_jsons:
        report = _write_benchmark_files(
            benchmark_path=benchmark_path,
            output_dir=output_dir / "benchmark",
            tokenizer=tokenizer,
            train_products=train_products,
            train_pair_keys=train_pair_keys,
        )
        benchmark_reports.append(report)
        benchmark_files.extend(report.get("files", {}).values())

    product_overlap = sum(1 for _product, split_counts in product_split_counts.items() if len(split_counts) > 1)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "input": str(uspto_csv),
        "output_dir": str(output_dir),
        "tokenizer": tokenizer,
        "canonicalize_training_smiles": canonicalize_training_smiles,
        "dedupe": dedupe,
        "max_rows": max_rows,
        "split_policy": "ChemEnzy GraphFP row-index split: train <1032204, valid <1161229, test <1290254",
        "examples_by_split": dict(counts),
        "total_examples": sum(counts.values()),
        "skipped": dict(skipped),
        "files": files,
        "benchmark_reports": benchmark_reports,
        "diagnostics": {
            "unique_products": len(product_split_counts),
            "products_appearing_in_multiple_splits": product_overlap,
            "source_counts": dict(source_counts),
        },
        "command_hints": {
            "preprocess": (
                "python vendor/ChemEnzyRetroPlanner/retro_planner/packages/onmt/onmt/bin/preprocess.py "
                f"-train_src {output_dir / 'plain.train.src'} -train_tgt {output_dir / 'plain.train.tgt'} "
                f"-valid_src {output_dir / 'plain.valid.src'} -valid_tgt {output_dir / 'plain.valid.tgt'} "
                f"-save_data {output_dir.parent / 'onmt/uspto_product_only'} -share_vocab -overwrite"
            ),
            "evaluate_uspto190_topstep": _benchmark_eval_hint(output_dir),
        },
        "contract": (
            "Product-only chemical proposer corpus. Source contains only product SMILES; no condition, "
            "template, reaction class, EC, or target-route context is used as model input."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.md").write_text(_render_markdown(manifest), encoding="utf-8")
    return manifest


def _split_for_row(row_idx: int) -> str:
    if row_idx < 1_032_204:
        return "train"
    if row_idx < 1_161_229:
        return "valid"
    return "test"


def _write_benchmark_files(
    *,
    benchmark_path: Path,
    output_dir: Path,
    tokenizer: str,
    train_products: set[str],
    train_pair_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    rows = _load_benchmark_rows(benchmark_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = benchmark_path.stem
    transitions = []
    skipped = Counter()
    for target_idx, row in enumerate(rows):
        target = str(row.get("target_smiles") or "")
        route = [step for step in row.get("gt_route") or [] if isinstance(step, dict)]
        if not target or not route:
            skipped["missing_target_or_route"] += 1
            continue
        first = None
        for step in route:
            rxn = str(step.get("rxn_smiles") or "")
            if ">>" not in rxn:
                continue
            lhs, rhs = rxn.split(">>", 1)
            product, reactants = canonicalize_product_and_reactants(rhs, [part for part in lhs.split(".") if part])
            if not product or not reactants:
                skipped["strict_canonicalization_failed"] += 1
                continue
            if first is None:
                first = (product, reactants, step)
            if product == canonicalize_product_and_reactants(target, [target])[0]:
                first = (product, reactants, step)
                break
        if first is None:
            skipped["no_usable_first_step"] += 1
            continue
        product, reactants, step = first
        reactant_side = ".".join(reactants)
        transitions.append(
            {
                "target_index": target_idx,
                "cascade_id": row.get("cascade_id"),
                "product": product,
                "reactants": reactants,
                "reaction_smiles": reactant_side + f">>{product}",
                "source_benchmark": str(benchmark_path),
                "step_role": step.get("step_role"),
                "train_product_overlap": product in train_products,
                "train_exact_pair_overlap": (product, reactant_side) in train_pair_keys,
            }
        )

    src_path = output_dir / f"{stem}.target_step.src"
    tgt_path = output_dir / f"{stem}.target_step.tgt"
    meta_path = output_dir / f"{stem}.target_step.meta.jsonl"
    src_path.write_text(
        "".join(_tokenize_smiles(row["product"], tokenizer) + "\n" for row in transitions),
        encoding="utf-8",
    )
    tgt_path.write_text(
        "".join(_tokenize_smiles(".".join(row["reactants"]), tokenizer) + "\n" for row in transitions),
        encoding="utf-8",
    )
    meta_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transitions),
        encoding="utf-8",
    )
    return {
        "benchmark": str(benchmark_path),
        "n_input_targets": len(rows),
        "n_target_step_examples": len(transitions),
        "skipped": dict(skipped),
        "train_product_overlap": sum(1 for row in transitions if row["train_product_overlap"]),
        "train_exact_pair_overlap": sum(1 for row in transitions if row["train_exact_pair_overlap"]),
        "files": {
            "src": str(src_path),
            "tgt": str(tgt_path),
            "metadata": str(meta_path),
        },
    }


def _load_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("rows", "targets", "examples", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _benchmark_eval_hint(output_dir: Path) -> str:
    benchmark_dir = output_dir / "benchmark"
    return (
        "python scripts/evaluate_chem_enzy_onmt_checkpoint_exact.py --model CHECKPOINT.pt "
        f"--src {benchmark_dir / 'uspto190_first128.target_step.src'} "
        f"--tgt {benchmark_dir / 'uspto190_first128.target_step.tgt'} "
        "--tokenizer char --beam-size 10 --topk 10 --batch-size 64 --device 0 --output eval.json"
    )


def _render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# USPTO Product-Only Chemical ONMT Corpus",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## Summary",
        "",
        f"- total_examples: {manifest['total_examples']}",
        f"- tokenizer: `{manifest['tokenizer']}`",
        f"- canonicalize_training_smiles: `{manifest['canonicalize_training_smiles']}`",
        f"- dedupe: `{manifest['dedupe']}`",
        "",
        "| split | examples |",
        "| --- | ---: |",
    ]
    for split in ("train", "valid", "test"):
        lines.append(f"| {split} | {manifest['examples_by_split'].get(split, 0)} |")
    lines.extend(["", "## Benchmarks", "", "| benchmark | n | train product overlap | train exact pair overlap |", "| --- | ---: | ---: | ---: |"])
    for row in manifest.get("benchmark_reports") or []:
        lines.append(
            f"| `{Path(row['benchmark']).name}` | {row['n_target_step_examples']} | "
            f"{row['train_product_overlap']} | {row['train_exact_pair_overlap']} |"
        )
    lines.extend(["", "## Contract", "", manifest["contract"], ""])
    return "\n".join(lines)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uspto-csv", type=Path, default=DEFAULT_USPTO_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", choices=["char", "smiles_token"], default="char")
    parser.add_argument("--no-canonicalize-training-smiles", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--benchmark-json", type=Path, action="append", default=[DEFAULT_BENCHMARK])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_corpus(
        uspto_csv=args.uspto_csv,
        output_dir=args.output_dir,
        tokenizer=args.tokenizer,
        canonicalize_training_smiles=not args.no_canonicalize_training_smiles,
        dedupe=not args.no_dedupe,
        max_rows=args.max_rows,
        benchmark_jsons=args.benchmark_json,
    )
    print(json.dumps({k: manifest[k] for k in ("schema_version", "total_examples", "examples_by_split", "skipped", "output_dir")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
