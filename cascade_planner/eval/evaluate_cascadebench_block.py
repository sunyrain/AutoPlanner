"""Evaluate CascadeBench block recovery from transition replay ranks."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BLOCK_SCHEMA_VERSION = "cascadebench_block_eval.v1"


def evaluate_cascadebench_block(
    *,
    coverage: Path,
    replay_jsonl: Path,
    output: Path,
    report: Path,
    block_sizes: list[int] | None = None,
    top_ks: list[int] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    block_sizes = sorted({int(size) for size in (block_sizes or [2, 3]) if int(size) > 1})
    top_ks = sorted({int(k) for k in (top_ks or [1, 3, 5, 10, 20, 50]) if int(k) > 0})
    coverage_payload = _read_json(coverage)
    transitions = [row for row in coverage_payload.get("transitions") or [] if isinstance(row, dict)]
    replay_rows = _read_jsonl(replay_jsonl)
    replay_by_id = {str(row.get("transition_id") or ""): row for row in replay_rows if row.get("transition_id")}
    scored = []
    for row in transitions:
        replay = replay_by_id.get(str(row.get("transition_id") or ""))
        if not replay:
            continue
        merged = dict(row)
        merged["best_positive_rank"] = replay.get("best_positive_rank") or {}
        merged["best_exact_rank"] = replay.get("best_exact_rank") or {}
        merged["selected_score"] = replay.get("selected_score")
        scored.append(merged)
    score_names = sorted(_score_names(scored))
    blocks = _build_blocks(scored, block_sizes=block_sizes)
    summary = _summarize_blocks(blocks, score_names=score_names, top_ks=top_ks)
    result = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "coverage": str(coverage),
            "replay_jsonl": str(replay_jsonl),
            "output": str(output),
            "report": str(report),
            "block_sizes": block_sizes,
            "top_ks": top_ks,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "coverage_transitions": len(transitions),
            "replay_transitions": len(replay_rows),
            "matched_transitions": len(scored),
            "blocks": len(blocks),
            "score_names": score_names,
            "route_groups": len(_route_groups(scored)),
        },
        "summary": summary,
        "block_examples": blocks[:50],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown(result), encoding="utf-8")
    return result


def _build_blocks(rows: list[dict[str, Any]], *, block_sizes: list[int]) -> list[dict[str, Any]]:
    blocks = []
    for route_key, route_rows in _route_groups(rows).items():
        route_rows = sorted(route_rows, key=lambda row: (int(row.get("step_pos") or 0), str(row.get("transition_id") or "")))
        for size in block_sizes:
            if len(route_rows) < size:
                continue
            for start in range(0, len(route_rows) - size + 1):
                chunk = route_rows[start : start + size]
                blocks.append(
                    {
                        "block_id": f"{route_key}::block{size}::{start}",
                        "route_key": route_key,
                        "block_size": size,
                        "transition_ids": [row.get("transition_id") for row in chunk],
                        "step_positions": [row.get("step_pos") for row in chunk],
                        "route_domain": chunk[0].get("route_domain"),
                        "has_hidden_or_nonisolated_intermediate": any(row.get("intermediate_isolated") is False for row in chunk),
                        "catalyst_classes": sorted({str(value) for row in chunk for value in row.get("catalyst_classes") or [] if value}),
                        "best_positive_rank": _merge_rank_maps(chunk, "best_positive_rank"),
                        "best_exact_rank": _merge_rank_maps(chunk, "best_exact_rank"),
                    }
                )
    return blocks


def _summarize_blocks(blocks: list[dict[str, Any]], *, score_names: list[str], top_ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for size in sorted({int(block.get("block_size") or 0) for block in blocks}):
        size_blocks = [block for block in blocks if int(block.get("block_size") or 0) == size]
        out[str(size)] = {
            "all": _score_summary(size_blocks, score_names=score_names, top_ks=top_ks),
            "hidden_or_nonisolated": _score_summary(
                [block for block in size_blocks if block.get("has_hidden_or_nonisolated_intermediate")],
                score_names=score_names,
                top_ks=top_ks,
            ),
            "route_domain_counts": dict(Counter(block.get("route_domain") for block in size_blocks)),
        }
    return out


def _score_summary(blocks: list[dict[str, Any]], *, score_names: list[str], top_ks: list[int]) -> dict[str, Any]:
    total = len(blocks)
    out: dict[str, Any] = {"blocks": total, "scores": {}}
    for score in score_names:
        out["scores"][score] = {}
        for label_name, rank_key in (("positive_label", "best_positive_rank"), ("exact_label", "best_exact_rank")):
            ranks = [((block.get(rank_key) or {}).get(score)) for block in blocks]
            covered = [rank for rank in ranks if rank is not None]
            metric = {
                "covered_blocks": len(covered),
                "coverage": round(len(covered) / max(total, 1), 6),
                "mrr_covered": round(sum(1.0 / int(rank) for rank in covered) / max(len(covered), 1), 6) if covered else 0.0,
                "recovery_at_k_all": {
                    str(k): round(sum(1 for rank in covered if int(rank) <= k) / max(total, 1), 6)
                    for k in top_ks
                },
                "recovery_at_k_covered": {
                    str(k): round(sum(1 for rank in covered if int(rank) <= k) / max(len(covered), 1), 6)
                    for k in top_ks
                },
            }
            out["scores"][score][label_name] = metric
    return out


def _merge_rank_maps(rows: list[dict[str, Any]], key: str) -> dict[str, int | None]:
    names = set()
    for row in rows:
        names.update((row.get(key) or {}).keys())
    out: dict[str, int | None] = {}
    for name in names:
        ranks = [(row.get(key) or {}).get(name) for row in rows]
        if any(rank is None for rank in ranks):
            out[name] = None
        else:
            out[name] = max(int(rank) for rank in ranks)
    return out


def _score_names(rows: list[dict[str, Any]]) -> set[str]:
    names = set()
    for row in rows:
        names.update((row.get("best_positive_rank") or {}).keys())
        names.update((row.get("best_exact_rank") or {}).keys())
    return names


def _route_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                str(row.get("doi") or ""),
                str(row.get("cascade_id") or ""),
                str(row.get("target_smiles") or ""),
            ]
        )
        groups[key].append(row)
    return groups


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CascadeBench Block Evaluation",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Block Metrics",
        "",
    ]
    for size, block_summary in (result.get("summary") or {}).items():
        lines.extend([f"### Block Size {size}", ""])
        for subset_name in ("all", "hidden_or_nonisolated"):
            subset = block_summary.get(subset_name) or {}
            lines.extend(
                [
                    f"#### {subset_name}",
                    "",
                    f"- blocks: `{subset.get('blocks')}`",
                    "",
                    "| Score | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
                    "|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for score, score_row in (subset.get("scores") or {}).items():
                for label_name in ("positive_label", "exact_label"):
                    metric = score_row.get(label_name) or {}
                    at = metric.get("recovery_at_k_all") or {}
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                score,
                                label_name,
                                str(metric.get("coverage")),
                                str(metric.get("mrr_covered")),
                                str(at.get("1")),
                                str(at.get("3")),
                                str(at.get("5")),
                                str(at.get("10")),
                            ]
                        )
                        + " |"
                    )
            lines.append("")
    return "\n".join(lines) + "\n"


def _parse_int_list(value: str) -> list[int]:
    return [int(part) for part in str(value or "").replace(";", ",").split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CascadeBench block recovery from transition replay")
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--replay-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--block-sizes", default="2,3")
    ap.add_argument("--top-ks", default="1,3,5,10,20,50")
    args = ap.parse_args()
    result = evaluate_cascadebench_block(
        coverage=Path(args.coverage),
        replay_jsonl=Path(args.replay_jsonl),
        output=Path(args.output),
        report=Path(args.report),
        block_sizes=_parse_int_list(args.block_sizes),
        top_ks=_parse_int_list(args.top_ks),
    )
    print(json.dumps({"counts": result["counts"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
