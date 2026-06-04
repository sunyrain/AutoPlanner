#!/usr/bin/env python
"""Build no-expert proposal preference pairs for context ONMT training."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side, canonical_smiles  # noqa: E402


SCHEMA_VERSION = "context_onmt_proposal_preference_pack.v1"


def build_preference_pack(
    *,
    corpus_dir: Path,
    output_jsonl: Path,
    output_summary: Path | None = None,
    mode: str = "context",
    split: str = "train",
    max_examples: int | None = None,
    negative_types: tuple[str, ...] = ("self", "drop_aux", "cross_swap"),
) -> dict[str, Any]:
    rows = _read_meta(corpus_dir / f"{mode}.{split}.meta.jsonl", max_examples=max_examples)
    rows_by_product = {canonical_smiles(row.get("product")): row for row in rows}
    pairs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        product = str(row.get("product") or "")
        reactants = [str(item) for item in row.get("reactants") or [] if item]
        chosen = ".".join(reactants)
        if not product or not reactants:
            counts["missing_product_or_reactants"] += 1
            continue
        for neg_type in negative_types:
            rejected = _negative_for_row(row, rows=rows, rows_by_product=rows_by_product, idx=idx, neg_type=neg_type)
            if not rejected:
                counts[f"{neg_type}_unavailable"] += 1
                continue
            if canonical_side(rejected) == canonical_side(chosen):
                counts[f"{neg_type}_same_as_chosen"] += 1
                continue
            pair_id = f"{split}_{idx:06d}_{neg_type}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "source_example_id": row.get("source_example_id"),
                    "source_target_index": row.get("source_target_index"),
                    "route_index": row.get("route_index"),
                    "step_index": row.get("step_index"),
                    "split": row.get("split") or split,
                    "negative_type": neg_type,
                    "product": product,
                    "target_smiles": row.get("target_smiles") or product,
                    "chosen_reactants": chosen,
                    "rejected_reactants": rejected,
                    "chosen_reactant_list": reactants,
                    "rejected_reactant_list": [part for part in rejected.split(".") if part],
                    "contract": "Pairwise proposal preference: chosen is clean top-level seed reactants; rejected is rule-generated hard negative, not an expert label.",
                }
            )
            counts[neg_type] += 1
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "corpus_dir": str(corpus_dir),
        "mode": mode,
        "split": split,
        "source_examples": len(rows),
        "n_pairs": len(pairs),
        "negative_types": list(negative_types),
        "counts": dict(counts),
        "output_jsonl": str(output_jsonl),
        "contract": "Generated hard-negative proposal preferences only. Do not use rejected_reactants as supervised positives.",
    }
    if output_summary is not None:
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _negative_for_row(
    row: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    rows_by_product: dict[str, dict[str, Any]],
    idx: int,
    neg_type: str,
) -> str:
    product = str(row.get("product") or "")
    reactants = [str(item) for item in row.get("reactants") or [] if item]
    if neg_type == "self":
        return product
    if neg_type == "drop_aux":
        return reactants[0] if len(reactants) > 1 else ""
    if neg_type == "cross_swap":
        if not rows:
            return ""
        for offset in range(1, min(len(rows), 50) + 1):
            other = rows[(idx + offset) % len(rows)]
            other_product = canonical_smiles(other.get("product"))
            if other_product and other_product == canonical_smiles(product):
                continue
            other_reactants = [str(item) for item in other.get("reactants") or [] if item]
            if other_reactants:
                return ".".join(other_reactants)
        return ""
    if neg_type == "upstream_product":
        for reactant in reactants:
            upstream = rows_by_product.get(canonical_smiles(reactant))
            if upstream and upstream.get("product"):
                return str(upstream["product"])
        return ""
    raise ValueError(f"unsupported negative type: {neg_type}")


def _read_meta(path: Path, *, max_examples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= int(max_examples):
                break
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Context ONMT Proposal Preference Pack",
        "",
        f"生成时间：{summary['created_at']}",
        "",
        "## Summary",
        "",
        f"- source_examples: {summary['source_examples']}",
        f"- n_pairs: {summary['n_pairs']}",
        f"- mode: `{summary['mode']}`",
        f"- split: `{summary['split']}`",
        f"- output_jsonl: `{summary['output_jsonl']}`",
        "",
        "## Negative Types",
        "",
        "| type | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((summary.get("counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", summary["contract"], ""])
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--output-summary", type=Path)
    ap.add_argument("--mode", choices=["plain", "context"], default="context")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="train")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--negative-type", action="append", choices=["self", "drop_aux", "cross_swap", "upstream_product"])
    args = ap.parse_args()
    summary = build_preference_pack(
        corpus_dir=args.corpus_dir,
        output_jsonl=args.output_jsonl,
        output_summary=args.output_summary,
        mode=args.mode,
        split=args.split,
        max_examples=args.max_examples,
        negative_types=tuple(args.negative_type or ["self", "drop_aux", "cross_swap"]),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
