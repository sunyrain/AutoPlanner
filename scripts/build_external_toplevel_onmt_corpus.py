#!/usr/bin/env python
"""Build a larger top-level ONMT corpus from external single-step reaction libraries."""
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

try:  # pragma: no cover - optional RDKit runtime hygiene.
    from rdkit import RDLogger  # type: ignore
except Exception:  # pragma: no cover
    RDLogger = None

from cascade_planner.cascadeboard.route_recovery import canonical_side, canonical_smiles  # noqa: E402
from scripts.build_chem_enzy_cascade_onmt_corpus import _source_line, _tokenize_smiles  # noqa: E402
from scripts.onmt_corpus_normalization import canonicalize_product_and_reactants  # noqa: E402


SCHEMA_VERSION = "external_toplevel_onmt_corpus.v1"


def build_external_toplevel_corpus(
    *,
    output_dir: Path,
    modes: list[str],
    tokenizer: str = "smiles_token",
    sources: list[str] | None = None,
    max_per_source: int | None = None,
    max_total: int | None = None,
    split_policy: str = "hash_90_5_5",
    dedupe: bool = True,
    canonicalize_training_smiles: bool = True,
) -> dict[str, Any]:
    if "both" in modes:
        modes = ["plain", "context"]
    modes = sorted(set(modes))
    if set(modes) - {"plain", "context"}:
        raise ValueError(f"unsupported modes: {sorted(set(modes) - {'plain', 'context'})}")
    if tokenizer not in {"char", "smiles_token"}:
        raise ValueError(f"unsupported tokenizer: {tokenizer}")
    if split_policy not in {"hash_90_5_5", "all_train"}:
        raise ValueError(f"unsupported split_policy: {split_policy}")
    selected = tuple(sources or ["uspto50k", "enzymatic_retro", "ecreact"])

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_mode_split: dict[str, dict[str, list[tuple[str, str, dict[str, Any]]]]] = {
        mode: defaultdict(list) for mode in modes
    }
    skipped = Counter()
    seen = set()
    source_counts = Counter()
    emitted = 0
    for row in _iter_source_rows(selected, max_per_source=max_per_source):
        if max_total is not None and emitted >= int(max_total):
            break
        product = str(row.get("product") or "")
        reactants = [str(item) for item in row.get("reactants") or [] if item]
        source = str(row.get("source") or "unknown")
        if not product or not reactants:
            skipped["missing_product_or_reactants"] += 1
            continue
        if _is_self_reaction(product, reactants):
            skipped["self_reaction"] += 1
            continue
        if not canonical_side(product) or not canonical_side(".".join(reactants)):
            skipped["invalid_or_empty_canonical_side"] += 1
            continue
        raw_product = product
        raw_reactants = list(reactants)
        if canonicalize_training_smiles:
            product, reactants = canonicalize_product_and_reactants(product, reactants)
            if not product or not reactants:
                skipped["strict_canonicalization_failed"] += 1
                continue
        split = _split_for_product(product, policy=split_policy)
        step = {
            "product": product,
            "reactants": reactants,
            **({"ec": row.get("ec")} if row.get("ec") else {}),
        }
        cascade = {"metadata": {"split": split}, "stage_partition": ["stage_1"], "steps": [step]}
        reactant_line = ".".join(reactants)
        metadata = {
            "source": source,
            "source_row_id": row.get("source_row_id"),
            "target_smiles": product,
            "route_index": emitted,
            "step_index": 0,
            "split": split,
            "stage": "stage_1",
            "product": product,
            "reactants": reactants,
            "raw_product": raw_product,
            "raw_reactants": raw_reactants,
            "reaction_smiles": ".".join(reactants) + f">>{product}",
            "canonical_reaction": _canonical_reaction(reactants, product),
            "ec": row.get("ec"),
            "contract": "External single-step top-level proposal positive; not an expert preference label.",
        }
        emitted_this_row = False
        for mode in modes:
            try:
                src = _source_line(mode, cascade, step, 0, product, product, tokenizer=tokenizer)
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
            emitted_this_row = True
        if emitted_this_row:
            emitted += 1
            source_counts[source] += 1

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
            files[mode][split] = {"src": str(src_path), "tgt": str(tgt_path), "metadata": str(meta_path), "examples": len(rows)}
            counts[mode][split] = len(rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "output_dir": str(output_dir),
        "modes": modes,
        "tokenizer": tokenizer,
        "sources": list(selected),
        "max_per_source": max_per_source,
        "max_total": max_total,
        "split_policy": split_policy,
        "dedupe": dedupe,
        "canonicalize_training_smiles": canonicalize_training_smiles,
        "files": files,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "emitted_examples": emitted,
            "source_counts": dict(source_counts),
            "examples_by_mode_split": counts,
            "total_examples": {mode: sum(counts[mode].values()) for mode in modes},
            "skipped": dict(skipped),
            "output_dir": str(output_dir),
        },
        "contract": "External top-level proposal corpus only. It does not train or promote ChemEnzy.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def _iter_source_rows(sources: Iterable[str], *, max_per_source: int | None) -> Iterable[dict[str, Any]]:
    for source in sources:
        count = 0
        for row in _iter_one_source(source):
            yield row
            count += 1
            if max_per_source is not None and count >= int(max_per_source):
                break


def _iter_one_source(source: str) -> Iterable[dict[str, Any]]:
    source = str(source)
    if source == "uspto50k":
        yield from _iter_uspto50k()
        return
    if source == "enzymatic_retro":
        yield from _iter_enzymatic_retro()
        return
    if source == "ecreact":
        yield from _iter_ecreact()
        return
    raise ValueError(f"unsupported source: {source}")


def _iter_uspto50k() -> Iterable[dict[str, Any]]:
    tab_path = Path("data/uspto50k.tab")
    if tab_path.exists():
        with tab_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for idx, row in enumerate(reader):
                reactant_line = str(row.get("reactant") or "")
                yield {
                    "source": "uspto50k",
                    "source_row_id": idx,
                    "product": str(row.get("product") or ""),
                    "reactants": [part for part in reactant_line.split(".") if part],
                }
    for csv_path in [Path("data_external/uspto50k/test.csv"), Path("data_external/uspto50k/tdc_test.csv")]:
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                reactant_line = str(row.get("output") or "")
                yield {
                    "source": "uspto50k",
                    "source_row_id": f"{csv_path.name}:{idx}",
                    "product": str(row.get("input") or ""),
                    "reactants": [part for part in reactant_line.split(".") if part],
                }


def _iter_enzymatic_retro() -> Iterable[dict[str, Any]]:
    for json_path in [Path("data_external/enzymatic_retro_data/train.json"), Path("data_external/enzymatic_retro_data/val.json")]:
        if not json_path.exists():
            continue
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows or []):
            reactant_line = str(row.get("reactants") or "")
            yield {
                "source": "enzymatic_retro",
                "source_row_id": f"{json_path.name}:{idx}",
                "product": str(row.get("product") or ""),
                "reactants": [part for part in reactant_line.split(".") if part],
                "ec": row.get("ec"),
            }


def _iter_ecreact() -> Iterable[dict[str, Any]]:
    csv_path = Path("data_external/ecreact/ecreact-1.0.csv")
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            rxn = str(row.get("rxn_smiles") or "")
            if ">>" not in rxn:
                continue
            lhs, product = rxn.split(">>", 1)
            reactant_line = lhs.split("|", 1)[0]
            yield {
                "source": "ecreact",
                "source_row_id": idx,
                "product": product,
                "reactants": [part for part in reactant_line.split(".") if part],
                "ec": row.get("ec"),
            }


def _split_for_product(product: str, *, policy: str) -> str:
    if policy == "all_train":
        return "train"
    bucket = _stable_bucket(canonical_smiles(product) or product)
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def _stable_bucket(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100


def _is_self_reaction(product: str, reactants: list[str]) -> bool:
    product_key = canonical_smiles(product)
    if not product_key:
        return False
    return any(canonical_smiles(reactant) == product_key for reactant in reactants if reactant)


def _canonical_reaction(reactants: list[str], product: str) -> str:
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
        "# External Top-level ONMT Corpus",
        "",
        f"created_at: `{manifest['created_at']}`",
        "",
        "## Summary",
        "",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- sources: {', '.join(manifest['sources'])}",
        f"- tokenizer: `{manifest['tokenizer']}`",
        f"- split_policy: `{manifest['split_policy']}`",
        f"- canonicalize_training_smiles: {manifest.get('canonicalize_training_smiles', False)}",
        f"- emitted_examples: {manifest['summary']['emitted_examples']}",
        "",
        "## Source Counts",
        "",
        "| source | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((manifest["summary"].get("source_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Split Counts", "", "| mode | train | valid | test | total |", "| --- | ---: | ---: | ---: | ---: |"])
    counts = manifest["summary"]["examples_by_mode_split"]
    totals = manifest["summary"]["total_examples"]
    for mode in manifest["modes"]:
        row = counts[mode]
        lines.append(f"| {mode} | {row.get('train', 0)} | {row.get('valid', 0)} | {row.get('test', 0)} | {totals[mode]} |")
    lines.extend(["", "## Skipped", "", "| reason | count |", "| --- | ---: |"])
    for key, value in sorted((manifest["summary"].get("skipped") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Contract", "", manifest["contract"], ""])
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["plain", "context", "both"], nargs="+", default=["both"])
    ap.add_argument("--tokenizer", choices=["char", "smiles_token"], default="smiles_token")
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--max-per-source", type=int)
    ap.add_argument("--max-total", type=int)
    ap.add_argument("--split-policy", choices=["hash_90_5_5", "all_train"], default="hash_90_5_5")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--no-canonicalize-training-smiles", action="store_true")
    ap.add_argument("--show-rdkit-warnings", action="store_true")
    args = ap.parse_args()
    if RDLogger is not None and not args.show_rdkit_warnings:
        RDLogger.DisableLog("rdApp.warning")
    manifest = build_external_toplevel_corpus(
        output_dir=args.output_dir,
        modes=args.mode,
        tokenizer=args.tokenizer,
        sources=args.source or None,
        max_per_source=args.max_per_source,
        max_total=args.max_total,
        split_policy=args.split_policy,
        dedupe=not args.no_dedupe,
        canonicalize_training_smiles=not args.no_canonicalize_training_smiles,
    )
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
