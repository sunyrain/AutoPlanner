#!/usr/bin/env python3
"""Build a chosen-only supervised seed pack from verifier preferences.

Verifier preference pairs are useful for future DPO, but the current ChemEnzy
vendor stack only exposes supervised training paths. This script extracts only
the clean ``chosen_cascade`` side of each pair, deduplicates by source route,
and writes a pack compatible with ``build_chem_enzy_cascade_onmt_corpus.py``.

The rejected side is deliberately excluded from positives.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "verifier_preference_chosen_seed_pack.v1"


def main() -> None:
    args = _parse_args()
    result = build_seed_pack(
        preference_jsonl=args.preference_jsonl,
        output=args.output,
        max_routes=args.max_routes,
    )
    if args.markdown:
        _write_markdown(result, args.markdown)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def build_seed_pack(
    *,
    preference_jsonl: Path,
    output: Path,
    max_routes: int | None = None,
) -> dict[str, Any]:
    seen: set[tuple[str, str, str, str]] = set()
    examples: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    route_domain_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    rejected_reason_counts: Counter[str] = Counter()
    source_pairs = 0
    skipped = Counter()

    with preference_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_pairs += 1
            row = json.loads(line)
            cascade = row.get("chosen_cascade")
            if not isinstance(cascade, dict):
                skipped["missing_chosen_cascade"] += 1
                continue
            key = (
                str(row.get("source_path") or ""),
                str(row.get("source_target_index") or ""),
                str(row.get("source_route_index") or ""),
                str(row.get("chosen_example_id") or ""),
            )
            if key in seen:
                skipped["duplicate_chosen_route"] += 1
                rejected_reason_counts.update(str(reason) for reason in row.get("rejected_expected_failure_reasons") or [])
                continue
            seen.add(key)
            rejected_reason_counts.update(str(reason) for reason in row.get("rejected_expected_failure_reasons") or [])
            meta = dict(cascade.get("metadata") or {})
            split = str(meta.get("split") or "train").lower()
            route_domain = str(meta.get("route_domain") or "unknown")
            quality = str(meta.get("quality_tier") or "unknown")
            split_counts[split] += 1
            route_domain_counts[route_domain] += 1
            quality_counts[quality] += 1
            examples.append(
                {
                    "example_id": str(row.get("chosen_example_id") or f"chosen_{len(examples):07d}"),
                    "label": 1,
                    "source_path": row.get("source_path"),
                    "source_target_index": row.get("source_target_index"),
                    "source_route_index": row.get("source_route_index"),
                    "target_smiles": row.get("target_smiles") or _route_target(cascade),
                    "cascade": cascade,
                    "metadata": {
                        "source_pair_id": row.get("pair_id"),
                        "preference_source": row.get("preference_source"),
                        "training_role": "supervised_positive_chosen_only",
                        "rejected_side_used": False,
                    },
                }
            )
            if max_routes is not None and len(examples) >= int(max_routes):
                break

    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(preference_jsonl),
        "output": str(output),
        "examples": examples,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "source_pairs_scanned": source_pairs,
            "n_examples": len(examples),
            "max_routes": max_routes,
            "split_counts": dict(sorted(split_counts.items())),
            "route_domain_counts": dict(sorted(route_domain_counts.items())),
            "quality_tier_counts": dict(sorted(quality_counts.items())),
            "rejected_reason_counts_observed_not_used_as_positive": dict(
                sorted(rejected_reason_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "skipped": dict(sorted(skipped.items())),
            "contract": (
                "Chosen-only supervised seed pack extracted from verifier preference pairs. "
                "Rejected cascades are not emitted as positives and are not used for supervised training."
            ),
        },
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _route_target(cascade: dict[str, Any]) -> str:
    steps = [step for step in cascade.get("steps") or [] if isinstance(step, dict)]
    if not steps:
        return ""
    first = steps[0]
    return str(first.get("product") or first.get("product_smiles") or "")


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Verifier Preference Chosen-Only Seed Pack",
        "",
        f"- source_pairs_scanned: `{summary['source_pairs_scanned']}`",
        f"- n_examples: `{summary['n_examples']}`",
        f"- output: `{result['output']}`",
        "",
        "## Contract",
        "",
        summary["contract"],
        "",
        "## Splits",
        "",
        "| split | routes |",
        "| --- | ---: |",
    ]
    for key, value in summary["split_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Observed Rejected Reasons",
        "",
        "These reasons are counted from rejected preference sides for audit only; rejected cascades are not emitted as positives.",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ])
    for key, value in summary["rejected_reason_counts_observed_not_used_as_positive"].items():
        lines.append(f"| `{key}` | {value} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--max-routes", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
