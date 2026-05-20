"""Build real skeleton hard negatives from planner route outcomes."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


def build_skeleton_hard_negative_pack(
    *,
    pack_dir: str | Path = "",
    prior_path: str | Path | None = None,
    output_dir: str | Path,
    min_type_match: float = 0.34,
    max_negatives_per_group: int = 64,
) -> dict[str, Any]:
    source_path = Path(prior_path) if prior_path is not None else Path(pack_dir) / "skeleton_prior.jsonl"
    rows = load_skeleton_rows(source_path)
    grouped = group_by_target_depth(rows)
    pairwise_rows = []
    hard_negative_rows = []
    counters = Counter()
    reason_counts = Counter()

    for group_key, group_rows in grouped.items():
        positives = [row for row in group_rows if is_positive(row)]
        planner_rows = [
            row for row in group_rows
            if row.get("source") == "planner_route" and float(row.get("label") or 0.0) < 1.0
        ]
        if not positives or not planner_rows:
            continue
        counters["groups_with_positive_and_planner"] += 1
        positive_exports = [annotate_positive(row, group_key=group_key) for row in positives]
        hard = []
        for row in planner_rows:
            best = best_gt_match(row, positives)
            if not best:
                continue
            export = annotate_negative(row, best, group_key=group_key)
            if export["type_match_fraction"] < min_type_match and float(row.get("label") or 0.0) < 0.5:
                continue
            hard.append(export)
        hard.sort(
            key=lambda row: (
                float(row.get("type_match_fraction") or 0.0),
                float(row.get("planner_label") or 0.0),
                -int(row.get("type_edit_distance") or 0),
            ),
            reverse=True,
        )
        hard = hard[:max_negatives_per_group]
        if not hard:
            continue
        pairwise_rows.extend(positive_exports)
        pairwise_rows.extend(hard)
        hard_negative_rows.extend(hard)
        counters["positive_rows_in_groups"] += len(positive_exports)
        counters["hard_negative_rows"] += len(hard)
        for row in hard:
            reason_counts.update(row.get("failure_reasons") or [])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "hard_negatives": str(out_dir / "skeleton_hard_negatives.jsonl"),
        "pairwise_training": str(out_dir / "skeleton_pairwise_training.jsonl"),
        "manifest": str(out_dir / "manifest.json"),
        "report": str(out_dir / "report.md"),
    }
    write_jsonl(Path(files["hard_negatives"]), hard_negative_rows)
    write_jsonl(Path(files["pairwise_training"]), pairwise_rows)
    manifest = {
        "schema_version": "skeleton_hard_negative_pack.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack_dir": str(pack_dir),
        "prior_path": str(source_path),
        "output_dir": str(output_dir),
        "min_type_match": min_type_match,
        "max_negatives_per_group": max_negatives_per_group,
        "files": files,
        "counts": {
            "input_skeleton_rows": len(rows),
            "target_depth_groups": len(grouped),
            **dict(counters),
        },
        "hard_negative_failure_reasons": dict(reason_counts),
        "label_distribution": dict(Counter(row.get("label_type") or "unknown" for row in hard_negative_rows)),
    }
    Path(files["manifest"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(files["report"]).write_text(render_report(manifest), encoding="utf-8")
    return manifest


def load_skeleton_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = canonical_smiles(row.get("target_smiles") or "")
        types = row.get("type_sequence") or []
        if not target or not types:
            continue
        row["_target_canonical"] = target
        rows.append(row)
    return rows


def group_by_target_depth(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        depth = int(row.get("depth") or len(row.get("type_sequence") or []) or 0)
        grouped[(row["_target_canonical"], depth)].append(row)
    return grouped


def is_positive(row: dict[str, Any]) -> bool:
    return float(row.get("label") or 0.0) >= 1.0


def best_gt_match(row: dict[str, Any], positives: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not positives:
        return None
    pred = row.get("type_sequence") or []
    scored = []
    for pos in positives:
        gt = pos.get("type_sequence") or []
        dist = edit_distance(pred, gt)
        match_fraction = sequence_match_fraction(pred, gt)
        scored.append((dist, -match_fraction, pos))
    scored.sort(key=lambda item: (item[0], item[1]))
    dist, neg_match, pos = scored[0]
    return {
        "row": pos,
        "type_edit_distance": dist,
        "type_match_fraction": -neg_match,
    }


def annotate_positive(row: dict[str, Any], *, group_key: tuple[str, int]) -> dict[str, Any]:
    return {
        "target_smiles": row.get("target_smiles") or "",
        "target_canonical": group_key[0],
        "depth": group_key[1],
        "route_domain": row.get("route_domain") or "",
        "operation_mode": row.get("operation_mode") or "",
        "type_sequence": row.get("type_sequence") or [],
        "ec1_sequence": row.get("ec1_sequence") or [],
        "label": 1.0,
        "label_type": "benchmark_positive",
        "source": row.get("source") or "",
        "source_path": row.get("source_path") or "",
        "skeleton_id": row.get("skeleton_id") or "",
        "pair_role": "positive",
    }


def annotate_negative(row: dict[str, Any], match: dict[str, Any], *, group_key: tuple[str, int]) -> dict[str, Any]:
    gt_row = match["row"]
    metrics = row.get("metrics_summary") or {}
    reasons = failure_reasons(row, metrics)
    return {
        "target_smiles": row.get("target_smiles") or "",
        "target_canonical": group_key[0],
        "depth": group_key[1],
        "route_domain": row.get("route_domain") or "",
        "operation_mode": row.get("operation_mode") or "",
        "type_sequence": row.get("type_sequence") or [],
        "ec1_sequence": row.get("ec1_sequence") or [],
        "gt_type_sequence": gt_row.get("type_sequence") or [],
        "gt_ec1_sequence": gt_row.get("ec1_sequence") or [],
        "type_edit_distance": match["type_edit_distance"],
        "type_match_fraction": match["type_match_fraction"],
        "planner_label": float(row.get("label") or 0.0),
        "label": 0.0,
        "label_type": row.get("label_type") or "planner_negative",
        "source": row.get("source") or "",
        "source_path": row.get("source_path") or "",
        "skeleton_id": row.get("skeleton_id") or "",
        "route_id": row.get("route_id") or "",
        "metrics_summary": metrics,
        "failure_reasons": reasons,
        "pair_role": "hard_negative",
    }


def failure_reasons(row: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    reasons = []
    if row.get("label_type"):
        reasons.append(str(row["label_type"]))
    if metrics.get("filled_route") is False:
        reasons.append("not_filled")
    if metrics.get("progressive_route") is False:
        reasons.append("not_progressive")
    if metrics.get("route_solved") is False:
        reasons.append("not_solved")
    if metrics.get("strict_stock_solve") is False:
        reasons.append("stock_open")
    if metrics.get("compatibility_success") is False:
        reasons.append("compatibility_failure")
    for issue in metrics.get("issues") or []:
        reasons.append(f"issue:{issue}")
    return sorted(set(reasons)) or ["planner_negative"]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, 1):
        cur = [i]
        for j, bv in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if av == bv else 1),
            ))
        prev = cur
    return prev[-1]


def sequence_match_fraction(a: list[str], b: list[str]) -> float:
    denom = max(len(a), len(b), 1)
    matches = sum(1 for av, bv in zip(a, b) if av == bv)
    return matches / denom


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Skeleton Hard Negative Pack",
        "",
        f"- source prior: `{manifest['prior_path']}`",
        f"- input skeleton rows: `{counts['input_skeleton_rows']}`",
        f"- target-depth groups: `{counts['target_depth_groups']}`",
        f"- groups with positives and planner rows: `{counts.get('groups_with_positive_and_planner', 0)}`",
        f"- positive rows exported: `{counts.get('positive_rows_in_groups', 0)}`",
        f"- hard negative rows: `{counts.get('hard_negative_rows', 0)}`",
        f"- min type match: `{manifest['min_type_match']}`",
        "",
        "## Failure Reasons",
        "",
    ]
    for reason, count in sorted(manifest["hard_negative_failure_reasons"].items(), key=lambda item: (-item[1], item[0]))[:20]:
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend([
        "",
        "## Files",
        "",
        f"- hard negatives: `{manifest['files']['hard_negatives']}`",
        f"- pairwise training: `{manifest['files']['pairwise_training']}`",
        "",
        "These rows are real planner-route hard negatives, not synthetic type corruptions. They should be preferred for future skeleton reranker training.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", default="results/shared/training_pack/condition_20260507_v2_metadata")
    parser.add_argument("--prior-path", default=None)
    parser.add_argument("--output-dir", default="results/shared/skeleton_hard_negatives/condition_20260507_v2")
    parser.add_argument("--min-type-match", type=float, default=0.34)
    parser.add_argument("--max-negatives-per-group", type=int, default=64)
    args = parser.parse_args()
    manifest = build_skeleton_hard_negative_pack(
        pack_dir=args.pack_dir,
        prior_path=args.prior_path,
        output_dir=args.output_dir,
        min_type_match=args.min_type_match,
        max_negatives_per_group=args.max_negatives_per_group,
    )
    print(json.dumps({
        "counts": manifest["counts"],
        "failure_reasons": manifest["hard_negative_failure_reasons"],
        "files": manifest["files"],
    }, indent=2))


if __name__ == "__main__":
    main()
