#!/usr/bin/env python3
"""Build ChemEnzy/OpenNMT supervised corpora from verified cascade seeds.

Two corpora are emitted:

* plain: product SMILES -> reactant SMILES. This is closest to the vendored
  ChemEnzy OpenNMT checkpoint interface.
* context: cascade state tokens + product SMILES -> reactant SMILES. This is
  the cascade-aware objective, but it requires vocab/model adaptation before it
  can be used with an existing checkpoint.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "chem_enzy_cascade_onmt_corpus.v1"
SPLIT_MAP = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def main() -> None:
    args = _parse_args()
    result = build_corpus(
        input_path=args.input,
        output_dir=args.output_dir,
        modes=args.mode,
        max_routes=args.max_routes,
        dedupe=not args.no_dedupe,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def build_corpus(
    *,
    input_path: Path,
    output_dir: Path,
    modes: list[str],
    max_routes: int | None = None,
    dedupe: bool = True,
) -> dict[str, Any]:
    if "both" in modes:
        modes = ["plain", "context"]
    modes = sorted(set(modes))
    unknown = sorted(set(modes) - {"plain", "context"})
    if unknown:
        raise ValueError(f"unsupported modes: {unknown}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    seed_routes = _positive_seed_routes(payload, max_routes=max_routes)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_mode_split: dict[str, dict[str, list[tuple[str, str, dict[str, Any]]]]] = {
        mode: defaultdict(list) for mode in modes
    }
    skipped = Counter()
    seen = set()
    for route_idx, route_row in enumerate(seed_routes):
        cascade = route_row["cascade"]
        target = route_row.get("target_smiles") or _route_target(cascade)
        route_split = _route_split(cascade)
        steps = [step for step in cascade.get("steps") or [] if isinstance(step, dict)]
        if not steps:
            skipped["route_without_steps"] += 1
            continue
        for step_idx, step in enumerate(steps):
            product = _step_product(step)
            reactants = _step_reactants(step)
            if not product or not reactants:
                skipped["step_missing_product_or_reactants"] += 1
                continue
            reactant_line = ".".join(reactants)
            metadata = {
                "source_example_id": route_row.get("example_id"),
                "source_target_index": route_row.get("source_target_index"),
                "target_smiles": target,
                "route_index": route_idx,
                "step_index": step_idx,
                "split": route_split,
                "stage": _stage_for_step(cascade, step_idx),
                "product": product,
                "reactants": reactants,
            }
            for mode in modes:
                src = _source_line(mode, cascade, step, step_idx, product, target)
                tgt = _char_tokenize(reactant_line)
                split = SPLIT_MAP.get(route_split, "train")
                key = (mode, split, src, tgt)
                if dedupe and key in seen:
                    skipped["duplicate_step"] += 1
                    continue
                seen.add(key)
                rows_by_mode_split[mode][split].append((src, tgt, metadata))

    files: dict[str, Any] = {}
    summary_counts: dict[str, Any] = {}
    for mode in modes:
        files[mode] = {}
        summary_counts[mode] = {}
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
            summary_counts[mode][split] = len(rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "modes": modes,
        "dedupe": dedupe,
        "source_seed_routes": len(seed_routes),
        "files": files,
        "compatibility": {
            "plain": "Closest to current ChemEnzy ONMT char-tokenized product-to-reactants checkpoint interface.",
            "context": "Cascade-aware source with condition/stage tokens; requires vocab/model adaptation before checkpoint continue-training.",
        },
        "openmt_command_hints": _command_hints(output_dir),
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "source_seed_routes": len(seed_routes),
            "modes": modes,
            "examples_by_mode_split": summary_counts,
            "total_examples": {mode: sum(summary_counts[mode].values()) for mode in modes},
            "skipped": dict(skipped),
            "output_dir": str(output_dir),
        },
        "contract": "Corpus builder only. It does not run OpenNMT preprocessing/training or claim ChemEnzy fine-tuning completion.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(manifest, output_dir / "manifest.md")
    return manifest


def _positive_seed_routes(payload: dict[str, Any], *, max_routes: int | None) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("examples") or []:
        if not isinstance(row, dict) or row.get("label") != 1:
            continue
        cascade = row.get("cascade")
        if not isinstance(cascade, dict):
            continue
        rows.append(row)
        if max_routes is not None and len(rows) >= max_routes:
            break
    return rows


def _route_target(cascade: dict[str, Any]) -> str:
    steps = [step for step in cascade.get("steps") or [] if isinstance(step, dict)]
    return _step_product(steps[0]) if steps else ""


def _route_split(cascade: dict[str, Any]) -> str:
    split = str(((cascade.get("metadata") or {}).get("split") or "train")).lower()
    return split if split in SPLIT_MAP else "train"


def _step_product(step: dict[str, Any]) -> str:
    product = step.get("product") or step.get("product_smiles")
    if product:
        return str(product)
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    if ">>" in rxn:
        return rxn.split(">>", 1)[1].strip()
    return ""


def _step_reactants(step: dict[str, Any]) -> list[str]:
    out = []
    for key in ("reactants", "reactant_smiles"):
        value = step.get(key)
        if isinstance(value, list):
            out.extend(str(item) for item in value if item)
            if out:
                break
    if not out and step.get("main_reactant"):
        out.append(str(step.get("main_reactant")))
        out.extend(str(item) for item in step.get("aux_reactants") or [] if item)
    if not out:
        rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
        if ">>" in rxn:
            out.extend(item for item in rxn.split(">>", 1)[0].split(".") if item)
    return out


def _source_line(mode: str, cascade: dict[str, Any], step: dict[str, Any], step_idx: int, product: str, target: str) -> str:
    if mode == "plain":
        return _char_tokenize(product)
    tokens = [
        f"<step_{step_idx + 1}>",
        f"<stage_{_stage_for_step(cascade, step_idx)}>",
        f"<temp_{_bucket_temperature(_condition_value(step, 'temperature'))}>",
        f"<ph_{_bucket_ph(_condition_value(step, 'ph'))}>",
        f"<solv_{_safe_token(_condition_value(step, 'solvent') or 'unknown')}>",
        f"<ec_{_safe_token(_ec_prefix(step))}>",
        "<target>",
        *_char_tokens(target or product),
        "<product>",
        *_char_tokens(product),
    ]
    return " ".join(tokens)


def _stage_for_step(cascade: dict[str, Any], step_idx: int) -> str:
    stages = cascade.get("stage_partition") or []
    if isinstance(stages, list) and step_idx < len(stages) and stages[step_idx]:
        return _safe_token(stages[step_idx])
    return f"stage_{step_idx + 1}"


def _condition_value(step: dict[str, Any], field: str) -> Any:
    keys = {
        "temperature": ("T", "Temperature", "temperature", "temperature_c"),
        "ph": ("pH", "ph", "PH"),
        "solvent": ("solvent", "Solvent"),
    }[field]
    for key in keys:
        if step.get(key) not in (None, ""):
            return step.get(key)
    for row in step.get("condition_predictions") or []:
        if not isinstance(row, dict):
            continue
        for key in keys:
            if row.get(key) not in (None, ""):
                return row.get(key)
    return None


def _bucket_temperature(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "unknown"
    if number < 0:
        return "freezing"
    if number < 20:
        return "cold"
    if number <= 40:
        return "ambient"
    if number <= 70:
        return "warm"
    return "hot"


def _bucket_ph(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "unknown"
    if number < 4:
        return "acidic"
    if number <= 8:
        return "neutral"
    return "basic"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ec_prefix(step: dict[str, Any]) -> str:
    ec = str(step.get("ec") or "")
    if not ec:
        annotations = step.get("enzyme_ec_annotations") or []
        if annotations and isinstance(annotations[0], dict):
            ec = str(annotations[0].get("ec_number") or "")
    return ec.split(".", 1)[0] if ec else "unknown"


def _safe_token(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else "unknown"
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text or "unknown"


def _char_tokenize(text: str) -> str:
    return " ".join(text.replace(" ", ""))


def _char_tokens(text: str) -> list[str]:
    return list(text.replace(" ", ""))


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _command_hints(output_dir: Path) -> dict[str, str]:
    save_prefix = output_dir / "plain_onmt"
    return {
        "plain_preprocess": (
            "python vendor/ChemEnzyRetroPlanner/retro_planner/packages/onmt/onmt/bin/preprocess.py "
            f"-train_src {output_dir / 'plain.train.src'} "
            f"-train_tgt {output_dir / 'plain.train.tgt'} "
            f"-valid_src {output_dir / 'plain.valid.src'} "
            f"-valid_tgt {output_dir / 'plain.valid.tgt'} "
            f"-save_data {save_prefix}"
        ),
        "plain_train_from_checkpoint": (
            "python vendor/ChemEnzyRetroPlanner/retro_planner/packages/onmt/onmt/bin/train.py "
            f"-data {save_prefix} "
            "-train_from vendor/ChemEnzyRetroPlanner/retro_planner/packages/onmt/checkpoints/np-like/model_step_100000.pt "
            "-reset_optim all "
            f"-save_model {output_dir / 'plain_cascade_adapter'}"
        ),
    }


def _write_markdown(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# ChemEnzy Cascade OpenNMT Corpus",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## Summary",
        "",
        f"- source_seed_routes: {manifest['summary']['source_seed_routes']}",
        f"- modes: {', '.join(manifest['modes'])}",
        f"- output_dir: `{manifest['output_dir']}`",
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
        "## Compatibility",
        "",
        f"- plain: {manifest['compatibility']['plain']}",
        f"- context: {manifest['compatibility']['context']}",
        "",
        "## Command Hints",
        "",
        "```bash",
        manifest["openmt_command_hints"]["plain_preprocess"],
        manifest["openmt_command_hints"]["plain_train_from_checkpoint"],
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["plain", "context", "both"], nargs="+", default=["both"])
    parser.add_argument("--max-routes", type=int)
    parser.add_argument("--no-dedupe", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
