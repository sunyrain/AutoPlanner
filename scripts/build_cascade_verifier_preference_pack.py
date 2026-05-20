#!/usr/bin/env python3
"""Build verifier-derived preference pairs from a perturbation pack."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    result = build_preference_pack(
        args.input,
        output_jsonl=args.output_jsonl,
        summary_output=args.summary_output,
        max_negatives_per_positive=args.max_negatives_per_positive,
    )
    print(json.dumps(result["summary"], indent=2))


def build_preference_pack(
    pack_path: Path,
    *,
    output_jsonl: Path,
    summary_output: Path,
    max_negatives_per_positive: int = 32,
) -> dict[str, Any]:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    examples = [row for row in pack.get("examples") or [] if isinstance(row, dict)]
    groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"positive": [], "negative": []})
    for row in examples:
        key = (
            str(row.get("source_path") or ""),
            str(row.get("source_target_index") if row.get("source_target_index") is not None else row.get("target_smiles") or ""),
            str(row.get("source_route_index") if row.get("source_route_index") is not None else "0"),
        )
        if int(row.get("label") or 0) == 1:
            groups[key]["positive"].append(row)
        else:
            groups[key]["negative"].append(row)

    pairs = []
    reason_counts: Counter[str] = Counter()
    perturbation_counts: Counter[str] = Counter()
    for key, rows in groups.items():
        positives = rows["positive"]
        negatives = rows["negative"]
        if not positives or not negatives:
            continue
        chosen = positives[0]
        for rank, rejected in enumerate(negatives[: max(0, int(max_negatives_per_positive))]):
            reasons = [str(reason) for reason in rejected.get("expected_failure_reasons") or []]
            reason_counts.update(reasons)
            perturbation_counts[str(rejected.get("perturbation_type") or "unknown")] += 1
            pairs.append(
                {
                    "pair_id": f"pref_{len(pairs):07d}",
                    "schema_version": "cascade_verifier_preference_pair.v1",
                    "target_smiles": chosen.get("target_smiles") or rejected.get("target_smiles"),
                    "source_path": key[0],
                    "source_target_index": key[1],
                    "source_route_index": key[2],
                    "chosen_example_id": chosen.get("example_id"),
                    "rejected_example_id": rejected.get("example_id"),
                    "chosen_cascade": chosen.get("cascade"),
                    "rejected_cascade": rejected.get("cascade"),
                    "preference_source": "verifier_perturbation",
                    "rejected_perturbation_type": rejected.get("perturbation_type"),
                    "rejected_expected_failure_reasons": reasons,
                    "preference_strength": 1.0,
                    "metadata": {
                        "negative_rank_within_seed": rank,
                        "contract": "chosen is clean seed; rejected is rule-derived perturbation negative",
                    },
                }
            )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in pairs) + ("\n" if pairs else ""), encoding="utf-8")
    summary = {
        "schema_version": "cascade_verifier_preference_pack_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(pack_path),
        "output_jsonl": str(output_jsonl),
        "n_examples": len(examples),
        "n_groups": len(groups),
        "n_pairs": len(pairs),
        "max_negatives_per_positive": int(max_negatives_per_positive),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "perturbation_counts": dict(sorted(perturbation_counts.items(), key=lambda item: (-item[1], item[0]))),
        "contract": "Verifier-derived preference pairs for downstream DPO/reranking; not expert preference labels.",
    }
    result = {"summary": summary}
    summary_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verifier-derived preference pairs from perturbation pack")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-negatives-per-positive", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main()
