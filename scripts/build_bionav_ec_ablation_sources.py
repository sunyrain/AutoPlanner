#!/usr/bin/env python3
"""Build EC-context ablation source files for the locked BioNav benchmark."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_ablation_sources(ec_context_src: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line_no, line in enumerate(ec_context_src.read_text(encoding="utf-8").splitlines(), start=1):
        tokens = line.strip().split()
        if len(tokens) < 4:
            raise ValueError(f"{ec_context_src}:{line_no}: expected EC context source with product tokens")
        ec1, ec_full, product_marker = tokens[:3]
        if not ec1.startswith("<ec1_") or not ec_full.startswith("<ec_") or product_marker != "<product>":
            raise ValueError(f"{ec_context_src}:{line_no}: malformed EC context prefix")
        product_tokens = tokens[3:]
        rows.append((ec1, ec_full, product_tokens))

    if not rows:
        raise ValueError(f"{ec_context_src} is empty")

    shifted = rows[1:] + rows[:1]
    outputs = {
        "product_plain": output_dir / "product_plain.src",
        "product_marker": output_dir / "product_marker.src",
        "ec1_only": output_dir / "ec1_only.src",
        "ec1_shuffled": output_dir / "ec1_shuffled.src",
        "full_ec_oracle": output_dir / "full_ec_oracle.src",
        "full_ec_shuffled": output_dir / "full_ec_shuffled.src",
    }
    buffers = {name: [] for name in outputs}

    for (ec1, ec_full, product_tokens), (shift_ec1, shift_ec_full, _) in zip(rows, shifted):
        product = " ".join(product_tokens)
        buffers["product_plain"].append(product)
        buffers["product_marker"].append(f"<product> {product}")
        buffers["ec1_only"].append(f"{ec1} <product> {product}")
        buffers["ec1_shuffled"].append(f"{shift_ec1} <product> {product}")
        buffers["full_ec_oracle"].append(f"{ec1} {ec_full} <product> {product}")
        buffers["full_ec_shuffled"].append(f"{shift_ec1} {shift_ec_full} <product> {product}")

    written = {}
    for name, path in outputs.items():
        path.write_text("\n".join(buffers[name]) + "\n", encoding="utf-8")
        written[name] = str(path)

    manifest = {
        "schema_version": "bionav_ec_ablation_sources.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(ec_context_src),
        "n_examples": len(rows),
        "outputs": written,
        "definitions": {
            "product_plain": "Only tokenized product, no special marker and no EC.",
            "product_marker": "Only <product> marker plus tokenized product.",
            "ec1_only": "Oracle EC first-level class plus product.",
            "ec1_shuffled": "Wrong EC1 control by one-row rotation.",
            "full_ec_oracle": "Oracle EC1 and full EC plus product.",
            "full_ec_shuffled": "Wrong EC1 and full EC control by one-row rotation.",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ec-context-src",
        type=Path,
        default=Path("results/shared/bionav_v2_enzyme_corpus_20260529/benchmark/native_bionav_benchmark.ec_context.src"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/shared/bionav_v2_ec_context_ablation_20260529/sources"),
    )
    args = parser.parse_args()
    written = build_ablation_sources(args.ec_context_src, args.output_dir)
    print(json.dumps(written, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
