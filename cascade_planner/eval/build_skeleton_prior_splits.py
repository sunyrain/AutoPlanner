"""Build non-leaking skeleton-prior splits and retrieval baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.cascadeboard.skeleton_retrieval_prior import (
    DEFAULT_SKELETON_PRIOR_PATH,
    retrieve_skeleton_priors,
)


RDLogger.DisableLog("rdApp.*")


def build_skeleton_prior_splits(
    *,
    prior_path: str | Path = DEFAULT_SKELETON_PRIOR_PATH,
    output_dir: str | Path,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
    min_eval_label: float = 1.0,
    min_similarity: float = 0.60,
    limit: int = 5,
) -> dict[str, Any]:
    rows = load_skeleton_rows(Path(prior_path))
    assignments = assign_scaffold_splits(
        rows,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    split_rows = {
        split: [row for row in rows if assignments[row["_row_id"]] == split]
        for split in ("train", "val", "test")
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, data_rows in split_rows.items():
        path = out_dir / f"skeleton_prior_{split}.jsonl"
        write_jsonl(path, [_strip_internal(row) for row in data_rows])
        files[split] = str(path)

    split_summary = {
        split: summarize_split(data_rows)
        for split, data_rows in split_rows.items()
    }
    leakage = leakage_summary(split_rows)
    retrieval_eval = {
        "val": retrieval_baseline(
            split_rows["val"],
            train_prior_path=files["train"],
            min_eval_label=min_eval_label,
            min_similarity=min_similarity,
            limit=limit,
        ),
        "test": retrieval_baseline(
            split_rows["test"],
            train_prior_path=files["train"],
            min_eval_label=min_eval_label,
            min_similarity=min_similarity,
            limit=limit,
        ),
    }
    manifest = {
        "schema_version": "skeleton_prior_split.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prior_path": str(prior_path),
        "output_dir": str(out_dir),
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "min_eval_label": min_eval_label,
        "min_similarity": min_similarity,
        "limit": limit,
        "files": files,
        "split_summary": split_summary,
        "leakage": leakage,
        "retrieval_baseline": retrieval_eval,
    }
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "report.md"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest["files"]["report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest


def load_skeleton_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        target = canonical_smiles(row.get("target_smiles") or "")
        types = row.get("type_sequence") or []
        if not target or not types:
            continue
        row = dict(row)
        row["_row_id"] = idx
        row["_target_canonical"] = target
        row["_scaffold"] = scaffold_key(target)
        rows.append(row)
    return rows


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return "invalid:"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return f"scaffold:{canonical_smiles(scaffold)}"
    return f"acyclic:{canonical_smiles(smiles)}"


def assign_scaffold_splits(
    rows: list[dict[str, Any]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[int, str]:
    by_scaffold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scaffold[row["_scaffold"]].append(row)
    scaffolds = sorted(
        by_scaffold,
        key=lambda key: hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    n_groups = len(scaffolds)
    n_test = int(round(n_groups * max(test_fraction, 0.0)))
    n_val = int(round(n_groups * max(val_fraction, 0.0)))
    if test_fraction > 0 and n_groups >= 3:
        n_test = max(1, n_test)
    if val_fraction > 0 and n_groups - n_test >= 2:
        n_val = max(1, n_val)
    while n_test + n_val >= n_groups and (n_test > 0 or n_val > 0):
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
    test_scaffolds = set(scaffolds[:n_test])
    val_scaffolds = set(scaffolds[n_test:n_test + n_val])
    assignments = {}
    for row in rows:
        scaffold = row["_scaffold"]
        if scaffold in test_scaffolds:
            split = "test"
        elif scaffold in val_scaffolds:
            split = "val"
        else:
            split = "train"
        assignments[row["_row_id"]] = split
    return assignments


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [float(row.get("label") or 0.0) for row in rows]
    domains = Counter(row.get("route_domain") or "unknown" for row in rows)
    sources = Counter(row.get("source") or "unknown" for row in rows)
    return {
        "rows": len(rows),
        "targets": len({row["_target_canonical"] for row in rows}),
        "scaffolds": len({row["_scaffold"] for row in rows}),
        "positive_rows": sum(1 for label in labels if label >= 1.0),
        "avg_label": round(mean(labels), 3) if labels else None,
        "route_domains": dict(domains),
        "sources": dict(sources),
    }


def leakage_summary(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    target_sets = {
        split: {row["_target_canonical"] for row in rows}
        for split, rows in split_rows.items()
    }
    scaffold_sets = {
        split: {row["_scaffold"] for row in rows}
        for split, rows in split_rows.items()
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        "target_overlap": {
            f"{a}_{b}": len(target_sets[a].intersection(target_sets[b]))
            for a, b in pairs
        },
        "scaffold_overlap": {
            f"{a}_{b}": len(scaffold_sets[a].intersection(scaffold_sets[b]))
            for a, b in pairs
        },
    }


def retrieval_baseline(
    query_rows: list[dict[str, Any]],
    *,
    train_prior_path: str | Path,
    min_eval_label: float,
    min_similarity: float,
    limit: int,
) -> dict[str, Any]:
    eval_rows = [
        row for row in query_rows
        if float(row.get("label") or 0.0) >= min_eval_label
    ]
    available = 0
    hit1 = 0
    hit5 = 0
    similarities = []
    for row in eval_rows:
        priors = retrieve_skeleton_priors(
            row.get("target_smiles") or "",
            depth=int(row.get("depth") or len(row.get("type_sequence") or []) or 0),
            domain=row.get("route_domain") or "",
            pack_path=train_prior_path,
            limit=limit,
            min_similarity=min_similarity,
            exclude_exact_target=True,
        )
        type_sequences = [prior.get("type_sequence") or [] for prior in priors]
        if priors:
            available += 1
            similarities.append(float(priors[0].get("similarity") or 0.0))
        gt = row.get("type_sequence") or []
        hit1 += int(bool(type_sequences and type_sequences[0] == gt))
        hit5 += int(any(seq == gt for seq in type_sequences[:5]))
    n = len(eval_rows) or 1
    return {
        "n_eval": len(eval_rows),
        "prior_available": available,
        "prior_available_rate": available / n,
        "exact_type_hit1": hit1,
        "exact_type_hit1_rate": hit1 / n,
        "exact_type_hit5": hit5,
        "exact_type_hit5_rate": hit5 / n,
        "avg_top_similarity": round(mean(similarities), 3) if similarities else None,
        "min_similarity": min_similarity,
        "limit": limit,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Skeleton Prior Scaffold Split",
        "",
        f"- prior path: `{manifest['prior_path']}`",
        f"- seed: `{manifest['seed']}`",
        f"- val fraction: `{manifest['val_fraction']}`",
        f"- test fraction: `{manifest['test_fraction']}`",
        "",
        "## Split Summary",
        "",
    ]
    for split, summary in manifest["split_summary"].items():
        lines.extend([
            f"### {split}",
            f"- rows: `{summary['rows']}`",
            f"- targets: `{summary['targets']}`",
            f"- scaffolds: `{summary['scaffolds']}`",
            f"- positive rows: `{summary['positive_rows']}`",
            f"- route domains: `{summary['route_domains']}`",
            "",
        ])
    lines.extend([
        "## Leakage Checks",
        "",
        f"- target overlap: `{manifest['leakage']['target_overlap']}`",
        f"- scaffold overlap: `{manifest['leakage']['scaffold_overlap']}`",
        "",
        "## Retrieval Baseline",
        "",
    ])
    for split, metrics in manifest["retrieval_baseline"].items():
        lines.extend([
            f"### {split}",
            f"- n eval: `{metrics['n_eval']}`",
            f"- prior available: `{metrics['prior_available']}` (`{metrics['prior_available_rate']:.3f}`)",
            f"- exact type hit@1: `{metrics['exact_type_hit1']}` (`{metrics['exact_type_hit1_rate']:.3f}`)",
            f"- exact type hit@5: `{metrics['exact_type_hit5']}` (`{metrics['exact_type_hit5_rate']:.3f}`)",
            f"- avg top similarity: `{metrics['avg_top_similarity']}`",
            "",
        ])
    lines.append("This split is intended for skeleton-prior and reranker development; blind route benchmarks should still report retrieval-disabled metrics separately.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", default=str(DEFAULT_SKELETON_PRIOR_PATH))
    parser.add_argument("--output-dir", default="results/shared/skeleton_prior_split/scaffold_20260507")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-eval-label", type=float, default=1.0)
    parser.add_argument("--min-similarity", type=float, default=0.60)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    manifest = build_skeleton_prior_splits(
        prior_path=args.prior,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        min_eval_label=args.min_eval_label,
        min_similarity=args.min_similarity,
        limit=args.limit,
    )
    print(json.dumps({
        "split_summary": manifest["split_summary"],
        "leakage": manifest["leakage"],
        "retrieval_baseline": manifest["retrieval_baseline"],
    }, indent=2))


if __name__ == "__main__":
    main()
