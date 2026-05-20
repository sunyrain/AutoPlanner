"""Build formal train/val/test splits for the full v4 trace-candidate pool.

This module separates raw v4 release size from actual ChemEnzy trace coverage.
It starts from the full `dataset_v4_release/cascade_v4_high_quality.jsonl`
candidate pool, applies the same full100 exclusion/filtering as
`build_v4_trace_benchmark`, then writes stable split files plus a leakage
manifest.  It does not run ChemEnzy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.eval.build_v4_trace_benchmark import collect_v4_trace_candidates


SPLIT_SCHEMA_VERSION = "v4_trace_split_manifest.v1"
DEFAULT_SPLIT_SALT = "dataset_v4_release_full_trace_2026-05-10"


def build_v4_training_splits(
    *,
    v4_jsonl: Path,
    benchmark_path: Path,
    output_dir: Path,
    train_fraction: float = 0.80,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    split_salt: str = DEFAULT_SPLIT_SALT,
    group_by_scaffold: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    candidates, candidate_report = collect_v4_trace_candidates(
        v4_jsonl=v4_jsonl,
        benchmark_path=benchmark_path,
    )
    rows = sorted(candidates, key=_row_sort_key)
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    fractions = _normalized_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    groups, row_to_group = _build_groups(rows, group_by_scaffold=group_by_scaffold)
    group_splits = _assign_splits(groups, fractions=fractions, split_salt=split_salt)
    annotated_rows = []
    for idx, row in enumerate(rows):
        group = groups[row_to_group[idx]]
        split = group_splits[group["group_id"]]
        payload = dict(row)
        payload["split"] = split
        payload["split_group_id"] = group["group_id"]
        if group.get("scaffolds"):
            payload["split_scaffolds"] = sorted(group["scaffolds"])
        annotated_rows.append(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "v4_trace_candidates_all.json"
    train_path = output_dir / "v4_trace_train.json"
    val_path = output_dir / "v4_trace_val.json"
    test_path = output_dir / "v4_trace_test.json"
    manifest_path = output_dir / "v4_trace_split_manifest.json"
    report_path = output_dir / "v4_trace_split_report.md"
    _write_json(all_path, annotated_rows)
    _write_json(train_path, [row for row in annotated_rows if row["split"] == "train"])
    _write_json(val_path, [row for row in annotated_rows if row["split"] == "val"])
    _write_json(test_path, [row for row in annotated_rows if row["split"] == "test"])

    leakage = _leakage_report(annotated_rows)
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "benchmark_path": str(benchmark_path),
            "output_dir": str(output_dir),
            "split_salt": split_salt,
            "group_by_scaffold": group_by_scaffold,
            "limit": limit,
            "split_policy": "stable grouped DOI/target/scaffold split; full100 excluded before splitting",
            "fractions": fractions,
        },
        "source_candidate_report": candidate_report,
        "counts": {
            "candidate_rows_after_optional_limit": len(annotated_rows),
            "split_groups": len(groups),
            "unique_targets": len({row.get("target_smiles") for row in annotated_rows if row.get("target_smiles")}),
            "unique_doi": len({_norm(row.get("doi")) for row in annotated_rows if _norm(row.get("doi"))}),
        },
        "splits": _split_summary(annotated_rows),
        "leakage_checks": leakage,
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_manifest_markdown(manifest), encoding="utf-8")
    return manifest


def _normalized_fractions(*, train_fraction: float, val_fraction: float, test_fraction: float) -> dict[str, float]:
    values = {
        "train": max(0.0, float(train_fraction)),
        "val": max(0.0, float(val_fraction)),
        "test": max(0.0, float(test_fraction)),
    }
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("at least one split fraction must be positive")
    return {key: value / total for key, value in values.items()}


def _build_groups(rows: list[dict[str, Any]], *, group_by_scaffold: bool) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    uf = _UnionFind()
    row_tokens: dict[int, list[str]] = {}
    for idx, row in enumerate(rows):
        tokens = [f"row:{idx}"]
        doi = _norm(row.get("doi"))
        target = _norm(row.get("target_smiles"))
        if doi:
            tokens.append(f"doi:{doi}")
        if target:
            tokens.append(f"target:{target}")
        if group_by_scaffold:
            scaffold = _target_scaffold(str(row.get("target_smiles") or ""))
            if scaffold:
                tokens.append(f"scaffold:{scaffold}")
        for token in tokens:
            uf.add(token)
        for token in tokens[1:]:
            uf.union(tokens[0], token)
        row_tokens[idx] = tokens

    components: dict[str, list[int]] = defaultdict(list)
    for idx, tokens in row_tokens.items():
        components[uf.find(tokens[0])].append(idx)

    groups: dict[str, dict[str, Any]] = {}
    row_to_group: dict[int, str] = {}
    for root, indices in components.items():
        tokens = sorted({uf.find(token) for idx in indices for token in row_tokens[idx]})
        stable_material = sorted({token for idx in indices for token in row_tokens[idx] if not token.startswith("row:")})
        if not stable_material:
            stable_material = [root]
        group_id = _stable_id(*stable_material)
        group_rows = [rows[idx] for idx in indices]
        group = {
            "group_id": group_id,
            "row_indices": sorted(indices),
            "n_rows": len(indices),
            "route_domain": _mode(row.get("route_domain") for row in group_rows),
            "depth_bucket": _depth_bucket(max(int(row.get("depth") or len(row.get("gt_route") or [])) for row in group_rows)),
            "dois": {_norm(row.get("doi")) for row in group_rows if _norm(row.get("doi"))},
            "targets": {str(row.get("target_smiles") or "") for row in group_rows if row.get("target_smiles")},
            "scaffolds": {
                _target_scaffold(str(row.get("target_smiles") or ""))
                for row in group_rows
                if group_by_scaffold and _target_scaffold(str(row.get("target_smiles") or ""))
            },
            "tokens": tokens,
        }
        groups[group_id] = group
        for idx in indices:
            row_to_group[idx] = group_id
    return groups, row_to_group


def _assign_splits(
    groups: dict[str, dict[str, Any]],
    *,
    fractions: dict[str, float],
    split_salt: str,
) -> dict[str, str]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in groups.values():
        buckets[(str(group.get("route_domain") or "unknown"), str(group.get("depth_bucket") or "unknown"))].append(group)

    out: dict[str, str] = {}
    for bucket_key, bucket_groups in sorted(buckets.items()):
        ordered = sorted(
            bucket_groups,
            key=lambda group: _stable_hash(split_salt, bucket_key, group["group_id"]),
        )
        n = len(ordered)
        n_test = int(round(n * fractions.get("test", 0.0)))
        n_val = int(round(n * fractions.get("val", 0.0)))
        if n > 2 and fractions.get("test", 0.0) > 0.0:
            n_test = max(1, n_test)
        if n > 2 and fractions.get("val", 0.0) > 0.0:
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            overflow = n_test + n_val - max(0, n - 1)
            if overflow > 0:
                reduce_val = min(n_val, overflow)
                n_val -= reduce_val
                overflow -= reduce_val
            if overflow > 0:
                n_test = max(0, n_test - overflow)
        for idx, group in enumerate(ordered):
            if idx < n_test:
                split = "test"
            elif idx < n_test + n_val:
                split = "val"
            else:
                split = "train"
            out[group["group_id"]] = split
    return out


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        out[split] = {
            "rows": len(split_rows),
            "unique_targets": len({row.get("target_smiles") for row in split_rows if row.get("target_smiles")}),
            "unique_doi": len({_norm(row.get("doi")) for row in split_rows if _norm(row.get("doi"))}),
            "route_domain_counts": dict(Counter(row.get("route_domain") for row in split_rows)),
            "depth_counts": dict(Counter(int(row.get("depth") or len(row.get("gt_route") or [])) for row in split_rows)),
            "quality_tier_counts": dict(Counter(row.get("quality_tier") for row in split_rows)),
            "compatibility_label_counts": dict(Counter(row.get("compatibility_label") for row in split_rows)),
        }
    return out


def _leakage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "doi_cross_split": _cross_split_values(rows, lambda row: _norm(row.get("doi"))),
        "target_cross_split": _cross_split_values(rows, lambda row: str(row.get("target_smiles") or "")),
        "scaffold_cross_split": _cross_split_values(
            rows,
            lambda row: "|".join(row.get("split_scaffolds") or []),
            ignore_empty=True,
        ),
    }


def _cross_split_values(rows: list[dict[str, Any]], getter: Any, *, ignore_empty: bool = True) -> dict[str, Any]:
    value_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = str(getter(row) or "")
        if ignore_empty and not value:
            continue
        value_splits[value].add(str(row.get("split") or ""))
    offenders = {
        value: sorted(splits)
        for value, splits in value_splits.items()
        if len(splits) > 1
    }
    return {
        "count": len(offenders),
        "examples": dict(sorted(offenders.items())[:25]),
    }


def _manifest_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest.get("counts") or {}
    leakage = manifest.get("leakage_checks") or {}
    lines = [
        "# v4 Full Trace Split",
        "",
        f"- schema: `{manifest.get('schema_version')}`",
        f"- candidate rows: `{counts.get('candidate_rows_after_optional_limit')}`",
        f"- unique targets: `{counts.get('unique_targets')}`",
        f"- unique DOI: `{counts.get('unique_doi')}`",
        f"- split groups: `{counts.get('split_groups')}`",
        "",
        "## Splits",
        "",
        "| Split | Rows | Unique Targets | Unique DOI |",
        "|---|---:|---:|---:|",
    ]
    for split, summary in (manifest.get("splits") or {}).items():
        lines.append(
            f"| {split} | {summary.get('rows')} | {summary.get('unique_targets')} | {summary.get('unique_doi')} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            f"- DOI cross-split count: `{(leakage.get('doi_cross_split') or {}).get('count')}`",
            f"- target cross-split count: `{(leakage.get('target_cross_split') or {}).get('count')}`",
            f"- scaffold cross-split count: `{(leakage.get('scaffold_cross_split') or {}).get('count')}`",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, path in (manifest.get("outputs") or {}).items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def _target_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetRingInfo().NumRings() == 0:
        return ""
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_mol is None:
        return ""
    # Avoid turning generic single-ring scaffolds, especially benzene, into one
    # enormous split group.  DOI and exact target grouping already protect the
    # strongest leakage paths; scaffold grouping is only for more specific cores.
    if scaffold_mol.GetNumHeavyAtoms() < 8 and scaffold_mol.GetRingInfo().NumRings() <= 1:
        return ""
    scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)
    return scaffold or ""


def _depth_bucket(depth: int) -> str:
    if depth <= 1:
        return "1"
    if depth == 2:
        return "2"
    if depth == 3:
        return "3"
    if depth <= 5:
        return "4-5"
    return "6+"


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_norm(row.get("doi")), _norm(row.get("cascade_id")), str(row.get("target_smiles") or ""))


def _mode(values: Any) -> str:
    counter = Counter(str(value or "unknown") for value in values)
    if not counter:
        return "unknown"
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _stable_id(*parts: Any) -> str:
    text = "\t".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _stable_hash(*parts: Any) -> str:
    text = "\t".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _write_json(path: Path, rows: Any) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Build full v4 trace candidates and leakage-safe splits")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-fraction", type=float, default=0.80)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    ap.add_argument("--no-scaffold-groups", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit; omit for full v4.")
    args = ap.parse_args()
    manifest = build_v4_training_splits(
        v4_jsonl=Path(args.v4_jsonl),
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output_dir),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_salt=args.split_salt,
        group_by_scaffold=not args.no_scaffold_groups,
        limit=args.limit,
    )
    print(json.dumps({"counts": manifest["counts"], "splits": manifest["splits"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
