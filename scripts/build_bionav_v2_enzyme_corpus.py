#!/usr/bin/env python3
"""Build BioNav-v2 enzyme one-step corpora and a locked native benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.onmt_corpus_normalization import canonicalize_side_strict


SCHEMA_VERSION = "bionav_v2_enzyme_corpus.v1"
DEFAULT_BRIDGE_POOL = Path("data/bridge_pack_v0/enzyme_reaction_pool.parquet")
DEFAULT_ECREACT = Path("data_external/ecreact/ecreact-1.0.csv")
DEFAULT_ENZYMATIC_RETRO_TRAIN = Path("data_external/enzymatic_retro_data/train.json")
DEFAULT_ENZYMATIC_RETRO_VAL = Path("data_external/enzymatic_retro_data/val.json")
DEFAULT_OUTPUT_DIR = Path("results/shared/bionav_v2_enzyme_corpus_20260529")


@dataclass(frozen=True)
class EnzymeReactionExample:
    product: str
    reactants: tuple[str, ...]
    ec_numbers: tuple[str, ...]
    source: str
    source_id: str = ""
    split_hint: str = ""
    occurrences: int = 1
    source_counts: dict[str, int] | None = None

    @property
    def reactant_side(self) -> str:
        return ".".join(self.reactants)

    @property
    def primary_ec(self) -> str:
        return self.ec_numbers[0] if self.ec_numbers else "unknown"

    @property
    def ec1(self) -> str:
        ec = self.primary_ec
        return ec.split(".", 1)[0] if "." in ec else ec

    @property
    def exact_key(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        return (self.product, self.reactants, self.ec_numbers)

    @property
    def product_reactant_key(self) -> tuple[str, tuple[str, ...]]:
        return (self.product, self.reactants)


def build_corpus(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    bridge_pool: Path | None = DEFAULT_BRIDGE_POOL,
    ecreact_csv: Path | None = DEFAULT_ECREACT,
    enzymatic_retro_train_json: Path | None = DEFAULT_ENZYMATIC_RETRO_TRAIN,
    enzymatic_retro_val_json: Path | None = DEFAULT_ENZYMATIC_RETRO_VAL,
    tokenizer: str = "char",
    benchmark_size: int | None = None,
    max_train_examples: int | None = None,
    valid_fraction: float = 0.05,
    test_fraction: float = 0.05,
    exclude_benchmark_products: bool = True,
    seed: int = 20260529,
) -> dict[str, Any]:
    if tokenizer not in {"char", "smiles_token"}:
        raise ValueError(f"unsupported tokenizer: {tokenizer}")
    if valid_fraction < 0 or test_fraction < 0 or valid_fraction + test_fraction >= 1:
        raise ValueError("valid_fraction and test_fraction must be non-negative and sum to < 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_pool = _load_enzymatic_retro_json(
        enzymatic_retro_val_json,
        source="enzymatic_retro:val",
        split_hint="benchmark",
    )
    benchmark_rows = _select_benchmark(_dedupe_examples(benchmark_pool), size=benchmark_size, seed=seed)
    benchmark_exact_keys = {row.product_reactant_key for row in benchmark_rows}
    benchmark_products = {row.product for row in benchmark_rows}

    raw_sources: dict[str, list[EnzymeReactionExample]] = {
        "enzymatic_retro:train": _load_enzymatic_retro_json(
            enzymatic_retro_train_json,
            source="enzymatic_retro:train",
            split_hint="train",
        ),
        "ecreact": _load_ecreact_csv(ecreact_csv),
        "bridge_pool": _load_bridge_pool(bridge_pool),
    }
    train_pool: list[EnzymeReactionExample] = []
    skipped = Counter()
    for source_name, rows in raw_sources.items():
        for row in rows:
            if row.product_reactant_key in benchmark_exact_keys:
                skipped["benchmark_exact_overlap"] += 1
                continue
            if exclude_benchmark_products and row.product in benchmark_products:
                skipped["benchmark_product_overlap"] += 1
                continue
            train_pool.append(row)

    train_pool = _dedupe_examples(train_pool)
    train_pool = _stable_sample(train_pool, max_train_examples, seed=seed)
    splits = _split_by_product(train_pool, valid_fraction=valid_fraction, test_fraction=test_fraction, seed=seed)

    plain_files = _write_mode_files(output_dir, "plain", splits, tokenizer=tokenizer)
    ec_files = _write_mode_files(output_dir, "ec_context", splits, tokenizer=tokenizer)
    benchmark_files = _write_benchmark_files(output_dir / "benchmark", benchmark_rows, tokenizer=tokenizer)

    source_counts = {name: len(rows) for name, rows in raw_sources.items()}
    split_counts = {split: len(rows) for split, rows in splits.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_utc(),
        "output_dir": str(output_dir),
        "tokenizer": tokenizer,
        "seed": seed,
        "inputs": {
            "bridge_pool": str(bridge_pool) if bridge_pool else None,
            "ecreact_csv": str(ecreact_csv) if ecreact_csv else None,
            "enzymatic_retro_train_json": str(enzymatic_retro_train_json) if enzymatic_retro_train_json else None,
            "enzymatic_retro_val_json": str(enzymatic_retro_val_json) if enzymatic_retro_val_json else None,
        },
        "source_counts_after_strict_normalization": source_counts,
        "benchmark": {
            "source": "enzymatic_retro:val",
            "selection": "all_deduped" if benchmark_size is None else f"ec1_balanced_{benchmark_size}",
            "examples": len(benchmark_rows),
            "unique_products": len(benchmark_products),
            "ec1_counts": dict(_count_ec1(benchmark_rows)),
            "files": benchmark_files,
            "contract": (
                "Locked enzyme one-step benchmark for native BioNav comparison. "
                "Training corpus excludes exact product/reactant benchmark pairs and, by default, all benchmark products."
            ),
        },
        "training_corpus": {
            "examples_by_split": split_counts,
            "ec1_counts_by_split": {split: dict(_count_ec1(rows)) for split, rows in splits.items()},
            "files": {"plain": plain_files, "ec_context": ec_files},
            "exclude_benchmark_products": exclude_benchmark_products,
            "skipped": dict(skipped),
        },
        "command_hints": {
            "native_bionav_benchmark": (
                "python scripts/benchmark_chem_enzy_native_bionav.py "
                f"--src {benchmark_files['src']} "
                f"--tgt {benchmark_files['tgt']} "
                f"--meta {benchmark_files['metadata']} "
                f"--output-dir {output_dir / 'native_bionav_baseline'} "
                "--gpu 0 --topk 10 --engine batch --batch-size 64"
            ),
            "ec_context_bionav_v2_benchmark": (
                "python scripts/benchmark_chem_enzy_native_bionav.py "
                f"--src {benchmark_files['ec_context_src']} "
                f"--tgt {benchmark_files['tgt']} "
                f"--meta {benchmark_files['metadata']} "
                f"--output-dir {output_dir / 'bionav_v2_formal_eval'} "
                "--gpu 0 --topk 10 --engine batch --batch-size 64 "
                "--tokenizer pretokenized --inference-input source"
            ),
            "plain_onmt_training_data": (
                f"train_src={plain_files['train']['src']} train_tgt={plain_files['train']['tgt']} "
                f"valid_src={plain_files['valid']['src']} valid_tgt={plain_files['valid']['tgt']}"
            ),
            "ec_context_training_data": (
                f"train_src={ec_files['train']['src']} train_tgt={ec_files['train']['tgt']} "
                f"valid_src={ec_files['valid']['src']} valid_tgt={ec_files['valid']['tgt']}"
            ),
        },
        "contract": (
            "This builds supervised corpora and a benchmark only. It does not train BioNav-v2 "
            "or claim improved model performance."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(manifest, output_dir / "manifest.md")
    return manifest


def _load_enzymatic_retro_json(path: Path | None, *, source: str, split_hint: str) -> list[EnzymeReactionExample]:
    if not path or not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[EnzymeReactionExample] = []
    for idx, item in enumerate(payload if isinstance(payload, list) else []):
        if not isinstance(item, dict):
            continue
        product = str(item.get("product") or "")
        reactants = _split_side(item.get("reactants"))
        ec_numbers = _ec_tuple([item.get("ec")])
        normalized = _normalize_example(
            product=product,
            reactants=reactants,
            ec_numbers=ec_numbers,
            source=source,
            source_id=f"{source}:{idx}",
            split_hint=split_hint,
        )
        if normalized is not None:
            rows.append(normalized)
    return rows


def _load_ecreact_csv(path: Path | None) -> list[EnzymeReactionExample]:
    if not path or not Path(path).exists():
        return []
    import pandas as pd

    df = pd.read_csv(path)
    rows: list[EnzymeReactionExample] = []
    for idx, item in enumerate(df.itertuples(index=False)):
        rxn = str(getattr(item, "rxn_smiles", "") or "")
        if ">>" not in rxn:
            continue
        lhs, rhs = rxn.split(">>", 1)
        if "|" in lhs:
            lhs = lhs.split("|", 1)[0]
        ec_numbers = _ec_tuple([getattr(item, "ec", None)])
        normalized = _normalize_example(
            product=rhs,
            reactants=_split_side(lhs),
            ec_numbers=ec_numbers,
            source=f"ecreact:{getattr(item, 'source', None) or 'unknown'}",
            source_id=f"ecreact:{idx}",
            split_hint="train",
        )
        if normalized is not None:
            rows.append(normalized)
    return rows


def _load_bridge_pool(path: Path | None) -> list[EnzymeReactionExample]:
    if not path or not Path(path).exists():
        return []
    import pandas as pd

    df = pd.read_parquet(path)
    rows: list[EnzymeReactionExample] = []
    for idx, item in enumerate(df.itertuples(index=False)):
        source_counts = _json_dict(getattr(item, "source_counts_json", None))
        ec_numbers = _ec_tuple(_json_list(getattr(item, "ec_numbers_json", None)))
        normalized = _normalize_example(
            product=str(getattr(item, "product_smiles", "") or ""),
            reactants=_split_side(getattr(item, "substrate_smiles", None)),
            ec_numbers=ec_numbers,
            source="bridge_pool",
            source_id=str(getattr(item, "reaction_id", None) or f"bridge_pool:{idx}"),
            split_hint="train",
            occurrences=_safe_int(getattr(item, "occurrences", None), default=1),
            source_counts=source_counts,
        )
        if normalized is not None:
            rows.append(normalized)
    return rows


def _normalize_example(
    *,
    product: str,
    reactants: list[str],
    ec_numbers: Iterable[Any],
    source: str,
    source_id: str,
    split_hint: str,
    occurrences: int = 1,
    source_counts: dict[str, int] | None = None,
) -> EnzymeReactionExample | None:
    if not product or not reactants:
        return None
    product_canon, reactants_canon = _canonicalize_product_and_reactants_cached(product, reactants)
    if not product_canon or not reactants_canon:
        return None
    if product_canon == ".".join(reactants_canon):
        return None
    ecs = _ec_tuple(ec_numbers)
    if not ecs:
        ecs = ("unknown",)
    return EnzymeReactionExample(
        product=product_canon,
        reactants=tuple(reactants_canon),
        ec_numbers=ecs,
        source=source,
        source_id=source_id,
        split_hint=split_hint,
        occurrences=max(1, int(occurrences or 1)),
        source_counts=dict(source_counts or {}),
    )


def _canonicalize_product_and_reactants_cached(product: str, reactants: list[str]) -> tuple[str, list[str]]:
    product_side = _canonical_side_cached(str(product or ""))
    reactant_side = _canonical_side_cached(".".join(str(item) for item in reactants if item))
    if not product_side or not reactant_side:
        return "", []
    return ".".join(product_side), list(reactant_side)


@lru_cache(maxsize=500_000)
def _canonical_side_cached(side: str) -> tuple[str, ...]:
    return canonicalize_side_strict(side)


def _dedupe_examples(rows: Iterable[EnzymeReactionExample]) -> list[EnzymeReactionExample]:
    best: dict[tuple[str, tuple[str, ...], tuple[str, ...]], EnzymeReactionExample] = {}
    for row in rows:
        current = best.get(row.exact_key)
        if current is None or row.occurrences > current.occurrences:
            best[row.exact_key] = row
    return sorted(best.values(), key=lambda item: _stable_key(item.product, item.reactant_side, item.primary_ec))


def _select_benchmark(rows: list[EnzymeReactionExample], *, size: int | None, seed: int) -> list[EnzymeReactionExample]:
    rows = list(rows)
    if size is None or size <= 0 or len(rows) <= size:
        return rows
    grouped: dict[str, list[EnzymeReactionExample]] = defaultdict(list)
    for row in rows:
        grouped[row.ec1].append(row)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
    selected: list[EnzymeReactionExample] = []
    while len(selected) < size:
        added = False
        for ec1 in sorted(grouped):
            if grouped[ec1] and len(selected) < size:
                selected.append(grouped[ec1].pop())
                added = True
        if not added:
            break
    return sorted(selected, key=lambda item: _stable_key(item.product, item.reactant_side, item.primary_ec))


def _stable_sample(rows: list[EnzymeReactionExample], size: int | None, *, seed: int) -> list[EnzymeReactionExample]:
    if size is None or size <= 0 or len(rows) <= size:
        return rows
    return sorted(rows, key=lambda item: _stable_key(str(seed), item.product, item.reactant_side, item.primary_ec))[:size]


def _split_by_product(
    rows: list[EnzymeReactionExample],
    *,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[EnzymeReactionExample]]:
    splits: dict[str, list[EnzymeReactionExample]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        bucket = _hash_fraction(str(seed), row.product)
        if bucket < test_fraction:
            split = "test"
        elif bucket < test_fraction + valid_fraction:
            split = "valid"
        else:
            split = "train"
        splits[split].append(row)
    for split in splits:
        splits[split].sort(key=lambda item: _stable_key(item.product, item.reactant_side, item.primary_ec))
    return splits


def _write_mode_files(
    output_dir: Path,
    mode: str,
    splits: dict[str, list[EnzymeReactionExample]],
    *,
    tokenizer: str,
) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        rows = splits.get(split) or []
        stem = f"{mode}.{split}"
        src_rows = [_source_line(row, mode=mode, tokenizer=tokenizer) for row in rows]
        tgt_rows = [_tokenize_smiles(row.reactant_side, tokenizer) for row in rows]
        meta_rows = [_metadata(row, split=split) for row in rows]
        files[split] = _write_triplet(output_dir, stem, src_rows, tgt_rows, meta_rows)
    return files


def _write_benchmark_files(
    output_dir: Path,
    rows: list[EnzymeReactionExample],
    *,
    tokenizer: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    src_rows = [_tokenize_smiles(row.product, tokenizer) for row in rows]
    ec_context_src_rows = [_source_line(row, mode="ec_context", tokenizer=tokenizer) for row in rows]
    tgt_rows = [_tokenize_smiles(row.reactant_side, tokenizer) for row in rows]
    meta_rows = [_metadata(row, split="benchmark") for row in rows]
    files = _write_triplet(output_dir, "native_bionav_benchmark", src_rows, tgt_rows, meta_rows)
    ec_context_src = output_dir / "native_bionav_benchmark.ec_context.src"
    _write_lines(ec_context_src, ec_context_src_rows)
    files["ec_context_src"] = str(ec_context_src)
    return files


def _write_triplet(
    output_dir: Path,
    stem: str,
    src_rows: list[str],
    tgt_rows: list[str],
    meta_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    src = output_dir / f"{stem}.src"
    tgt = output_dir / f"{stem}.tgt"
    meta = output_dir / f"{stem}.meta.jsonl"
    _write_lines(src, src_rows)
    _write_lines(tgt, tgt_rows)
    _write_jsonl(meta, meta_rows)
    return {"src": str(src), "tgt": str(tgt), "metadata": str(meta), "examples": len(src_rows)}


def _source_line(row: EnzymeReactionExample, *, mode: str, tokenizer: str) -> str:
    if mode == "plain":
        return _tokenize_smiles(row.product, tokenizer)
    if mode != "ec_context":
        raise ValueError(f"unsupported mode: {mode}")
    return " ".join(
        [
            f"<ec1_{_safe_token(row.ec1)}>",
            f"<ec_{_safe_token(row.primary_ec)}>",
            "<product>",
            *_smiles_tokens(row.product, tokenizer),
        ]
    )


def _metadata(row: EnzymeReactionExample, *, split: str) -> dict[str, Any]:
    return {
        "product": row.product,
        "reactants": list(row.reactants),
        "reactant_side": row.reactant_side,
        "ec_numbers": list(row.ec_numbers),
        "ec": row.primary_ec,
        "ec1": row.ec1,
        "source": row.source,
        "source_id": row.source_id,
        "split": split,
        "occurrences": row.occurrences,
        "source_counts": dict(row.source_counts or {}),
    }


def _split_side(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part for part in text.split(".") if part]


def _ec_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            out.extend(_ec_tuple(value))
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        out.append(text)
    return tuple(sorted(set(out), key=lambda item: (_ec_specificity(item), item), reverse=True))


def _ec_specificity(ec: str) -> int:
    return sum(1 for part in str(ec).split(".") if part and part != "-")


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    return parsed if isinstance(parsed, list) else [parsed]


def _json_dict(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _safe_int(v, default=0) for k, v in value.items()}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): _safe_int(v, default=0) for k, v in parsed.items()}


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tokenize_smiles(text: str, tokenizer: str) -> str:
    if tokenizer == "char":
        return " ".join(str(text or "").replace(" ", ""))
    return " ".join(_smiles_tokens(text, tokenizer))


def _smiles_tokens(text: str, tokenizer: str) -> list[str]:
    compact = str(text or "").replace(" ", "")
    if tokenizer == "char":
        return list(compact)
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    tokens = [token for token in re.findall(pattern, compact)]
    if compact != "".join(tokens):
        raise ValueError(f"SMILES tokenization failed for {text!r}")
    return tokens


def _safe_token(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in text.split("_") if part) or "unknown"


def _count_ec1(rows: Iterable[EnzymeReactionExample]) -> Counter:
    return Counter(row.ec1 for row in rows)


def _stable_key(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def _hash_fraction(*parts: str) -> float:
    digest = _stable_key(*parts)
    return int(digest[:12], 16) / float(16 ** 12)


def _write_lines(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_markdown(manifest: dict[str, Any], path: Path) -> None:
    training = manifest["training_corpus"]
    benchmark = manifest["benchmark"]
    lines = [
        "# BioNav-v2 Enzyme Corpus",
        "",
        f"生成时间: {manifest['created_at']}",
        "",
        "## Benchmark",
        "",
        f"- source: {benchmark['source']}",
        f"- examples: {benchmark['examples']}",
        f"- unique_products: {benchmark['unique_products']}",
        f"- files: `{benchmark['files']['src']}` / `{benchmark['files']['tgt']}`",
        "",
        "## Training Corpus",
        "",
        "| split | examples |",
        "| --- | ---: |",
    ]
    for split in ("train", "valid", "test"):
        lines.append(f"| {split} | {training['examples_by_split'].get(split, 0)} |")
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```bash",
            manifest["command_hints"]["native_bionav_benchmark"],
            "```",
            "",
            "## Contract",
            "",
            manifest["contract"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bridge-pool", type=Path, default=DEFAULT_BRIDGE_POOL)
    parser.add_argument("--ecreact-csv", type=Path, default=DEFAULT_ECREACT)
    parser.add_argument("--enzymatic-retro-train-json", type=Path, default=DEFAULT_ENZYMATIC_RETRO_TRAIN)
    parser.add_argument("--enzymatic-retro-val-json", type=Path, default=DEFAULT_ENZYMATIC_RETRO_VAL)
    parser.add_argument("--tokenizer", choices=["char", "smiles_token"], default="char")
    parser.add_argument("--benchmark-size", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--allow-benchmark-product-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_corpus(
        output_dir=args.output_dir,
        bridge_pool=args.bridge_pool,
        ecreact_csv=args.ecreact_csv,
        enzymatic_retro_train_json=args.enzymatic_retro_train_json,
        enzymatic_retro_val_json=args.enzymatic_retro_val_json,
        tokenizer=args.tokenizer,
        benchmark_size=args.benchmark_size,
        max_train_examples=args.max_train_examples,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        exclude_benchmark_products=not args.allow_benchmark_product_overlap,
        seed=args.seed,
    )
    print(json.dumps({
        "benchmark_examples": manifest["benchmark"]["examples"],
        "training_examples_by_split": manifest["training_corpus"]["examples_by_split"],
        "output_dir": manifest["output_dir"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
