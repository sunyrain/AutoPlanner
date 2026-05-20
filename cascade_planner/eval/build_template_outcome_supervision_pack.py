"""Build a candidate supervision pack for template outcome review.

The Phase II selector experiments showed that pair+analog labels are too sparse
for a stable learned ranker.  This utility converts generated template proposal
pools into a compact JSONL pack of positives, near misses, and hard negatives
that can be reviewed or reused by a future weak-supervision stage.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "template_outcome_supervision_pack.v1"
DEFAULT_CLASSES = (
    "pair_and_analog_positive",
    "analog_only_positive",
    "pair_only_near_miss",
    "high_score_hard_negative",
)


def build_template_outcome_supervision_pack(
    *,
    proposal_jsons: list[Path],
    output_jsonl: Path,
    report_json: Path,
    max_per_target_class: int = 5,
    max_total_rows: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    selected: list[dict[str, Any]] = []
    pool_summaries = []
    for path in proposal_jsons:
        payload = _read_json(path)
        pool_name = path.parent.name
        rows = _proposal_rows(payload, pool_name=pool_name, source_path=path)
        selected_rows = _select_rows(rows, max_per_target_class=max_per_target_class)
        selected.extend(selected_rows)
        pool_summaries.append(
            {
                "pool_name": pool_name,
                "path": str(path),
                "proposals": len(rows),
                "selected": len(selected_rows),
                "classes": dict(Counter(row["supervision_class"] for row in selected_rows)),
            }
        )
    selected.sort(key=lambda row: (row["split"], row["target_smiles"], row["supervision_class"], row["rank_within_class"]))
    if max_total_rows is not None:
        selected = selected[: max(0, int(max_total_rows))]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "proposal_jsons": [str(path) for path in proposal_jsons],
            "output_jsonl": str(output_jsonl),
            "report_json": str(report_json),
            "max_per_target_class": max_per_target_class,
            "max_total_rows": max_total_rows,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": (
                "Rows are sampled from generated proposal pools. Labels are "
                "diagnostic labels from held-out references, not expert route "
                "feasibility judgments."
            ),
        },
        "summary": {
            "rows": len(selected),
            "targets": len({row["target_smiles"] for row in selected}),
            "pools": len(proposal_jsons),
            "classes": dict(Counter(row["supervision_class"] for row in selected)),
        },
        "pools": pool_summaries,
        "examples": selected[:20],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _proposal_rows(payload: dict[str, Any], *, pool_name: str, source_path: Path) -> list[dict[str, Any]]:
    meta = payload.get("metadata") or {}
    split = str(meta.get("split") or "")
    rows = []
    for target in payload.get("targets") or []:
        target_smiles = str(target.get("target_smiles") or "")
        for proposal in target.get("proposals") or []:
            if not isinstance(proposal, dict):
                continue
            supervision_class = _supervision_class(proposal)
            if supervision_class is None:
                continue
            row = {
                "schema_version": SCHEMA_VERSION,
                "source_pool": pool_name,
                "source_path": str(source_path),
                "split": split,
                "target_smiles": target_smiles,
                "supervision_class": supervision_class,
                "proposal_rank": proposal.get("proposal_rank"),
                "proposal_score": proposal.get("proposal_score"),
                "downstream_rank": proposal.get("downstream_rank"),
                "connector": proposal.get("connector"),
                "connector_heavy_atoms": proposal.get("connector_heavy_atoms"),
                "connector_is_main_reactant": proposal.get("connector_is_main_reactant"),
                "template_rank": proposal.get("template_rank"),
                "outcome_rank": proposal.get("outcome_rank"),
                "template_count": proposal.get("template_count"),
                "template_transform_pair": proposal.get("template_transform_pair"),
                "template": proposal.get("template"),
                "reactants": proposal.get("reactants") or [],
                "main_reactant": proposal.get("main_reactant"),
                "labels": {
                    "pair_hit": bool(proposal.get("pair_hit")),
                    "analog_hit": bool(proposal.get("analog_hit")),
                    "pair_and_analog": bool(proposal.get("pair_and_analog")),
                },
                "similarities": {
                    "upstream_similarity": proposal.get("upstream_similarity"),
                    "downstream_similarity": proposal.get("downstream_similarity"),
                },
                "reference": {
                    "best_reference_block_id": proposal.get("best_reference_block_id"),
                    "reference_transform_pair": proposal.get("reference_transform_pair"),
                },
                "features": _feature_subset(proposal),
            }
            rows.append(row)
    return rows


def _supervision_class(proposal: dict[str, Any]) -> str | None:
    if proposal.get("pair_and_analog"):
        return "pair_and_analog_positive"
    if proposal.get("analog_hit") and not proposal.get("pair_hit"):
        return "analog_only_positive"
    if proposal.get("pair_hit") and not proposal.get("analog_hit"):
        return "pair_only_near_miss"
    if not proposal.get("analog_hit") and not proposal.get("pair_hit"):
        return "high_score_hard_negative"
    return None


def _select_rows(rows: list[dict[str, Any]], *, max_per_target_class: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_pool"], row["target_smiles"], row["supervision_class"])].append(row)
    selected = []
    for _, group_rows in grouped.items():
        group_rows.sort(key=_selection_key)
        for rank, row in enumerate(group_rows[: max(0, int(max_per_target_class))], start=1):
            row = dict(row)
            row["rank_within_class"] = rank
            selected.append(row)
    return selected


def _selection_key(row: dict[str, Any]) -> tuple[float, int, int, int]:
    # Keep the highest-scoring / earliest proposal examples for compact review.
    return (
        -float(row.get("proposal_score") or 0.0),
        int(row.get("proposal_rank") or 10**9),
        int(row.get("downstream_rank") or 10**9),
        int(row.get("template_rank") or 10**9),
    )


def _feature_subset(proposal: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("app_", "rc_")
    out = {key: value for key, value in proposal.items() if key.startswith(prefixes)}
    return dict(sorted(out.items()))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Template Outcome Supervision Pack",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Pools", "", "| Pool | Proposals | Selected | Classes |", "|---|---:|---:|---|"])
    for pool in report.get("pools") or []:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` |".format(
                pool.get("pool_name"),
                pool.get("proposals"),
                pool.get("selected"),
                json.dumps(pool.get("classes") or {}, ensure_ascii=False),
            )
        )
    lines.extend(["", "## Contract", "", str((report.get("metadata") or {}).get("contract") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build template outcome supervision candidate pack")
    parser.add_argument("--proposal-json", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--max-per-target-class", type=int, default=5)
    parser.add_argument("--max-total-rows", type=int)
    args = parser.parse_args()
    report = build_template_outcome_supervision_pack(
        proposal_jsons=[Path(value) for value in args.proposal_json],
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        max_per_target_class=args.max_per_target_class,
        max_total_rows=args.max_total_rows,
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl, "report_json": args.report_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
