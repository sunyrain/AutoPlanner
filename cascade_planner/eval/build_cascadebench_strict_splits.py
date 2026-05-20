"""Build strict-blind CascadeBench splits from dataset_v4_release.

The existing v4 split protects DOI/target/scaffold leakage.  CascadeBench needs
an additional transition-level guard so that the strict test set does not share
the exact product/reaction/product-reactant keys used to train the transition
ranker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles
from cascade_planner.eval.build_v4_trace_benchmark import collect_v4_trace_candidates


STRICT_SPLIT_SCHEMA_VERSION = "cascadebench_strict_split_manifest.v1"
DEFAULT_SPLIT_SALT = "cascadebench_strict_20260516"


def build_cascadebench_strict_splits(
    *,
    v4_jsonl: Path,
    benchmark_path: Path,
    output_dir: Path,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_salt: str = DEFAULT_SPLIT_SALT,
    limit: int | None = None,
) -> dict[str, Any]:
    candidates, candidate_report = collect_v4_trace_candidates(
        v4_jsonl=v4_jsonl,
        benchmark_path=benchmark_path,
    )
    rows = sorted(candidates, key=_row_sort_key)
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    fractions = _normalized_fractions(train=train_fraction, val=val_fraction, test=test_fraction)
    groups, row_to_group = _build_strict_groups(rows)
    group_splits = _assign_splits(groups, fractions=fractions, split_salt=split_salt)
    annotated = []
    for idx, row in enumerate(rows):
        group = groups[row_to_group[idx]]
        payload = dict(row)
        payload["split"] = group_splits[group["group_id"]]
        payload["split_group_id"] = group["group_id"]
        payload["split_transition_tokens"] = sorted(group.get("transition_tokens") or [])[:200]
        payload["split_scaffolds"] = sorted(group.get("scaffolds") or [])
        annotated.append(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": output_dir / "v4_trace_candidates_all.json",
        "train": output_dir / "v4_trace_train.json",
        "val": output_dir / "v4_trace_val.json",
        "test": output_dir / "v4_trace_test.json",
        "manifest": output_dir / "cascadebench_strict_split_manifest.json",
        "report": output_dir / "cascadebench_strict_split_report.md",
    }
    _write_json(paths["all"], annotated)
    for split in ("train", "val", "test"):
        _write_json(paths[split], [row for row in annotated if row.get("split") == split])

    leakage = _leakage_report(annotated)
    manifest = {
        "schema_version": STRICT_SPLIT_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "benchmark_path": str(benchmark_path),
            "output_dir": str(output_dir),
            "split_salt": split_salt,
            "limit": limit,
            "split_policy": (
                "strict grouped split by DOI, canonical target, Murcko scaffold, "
                "step product, reaction, product+main reactant, and product+reactant set; "
                "full100 excluded before splitting"
            ),
            "fractions": fractions,
        },
        "source_candidate_report": candidate_report,
        "counts": {
            "candidate_rows_after_optional_limit": len(annotated),
            "split_groups": len(groups),
            "unique_targets": len({row.get("target_smiles") for row in annotated if row.get("target_smiles")}),
            "unique_doi": len({_norm(row.get("doi")) for row in annotated if _norm(row.get("doi"))}),
            "unique_transition_tokens": len(_tokens_by_split(annotated, "transition_token")["all"]),
        },
        "splits": _split_summary(annotated),
        "leakage_checks": leakage,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["report"].write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def _build_strict_groups(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    uf = _UnionFind()
    row_tokens: dict[int, list[str]] = {}
    row_transition_tokens: dict[int, set[str]] = {}
    row_scaffolds: dict[int, set[str]] = {}
    for idx, row in enumerate(rows):
        tokens = [f"row:{idx}"]
        doi = _norm(row.get("doi"))
        target = canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
        if doi:
            tokens.append(f"doi:{doi}")
        if target:
            tokens.append(f"target:{target}")
        scaffold = _target_scaffold(target)
        scaffolds = {scaffold} if scaffold else set()
        if scaffold:
            tokens.append(f"scaffold:{scaffold}")
        transition_tokens = _route_transition_tokens(row)
        tokens.extend(sorted(transition_tokens))
        for token in tokens:
            uf.add(token)
        for token in tokens[1:]:
            uf.union(tokens[0], token)
        row_tokens[idx] = tokens
        row_transition_tokens[idx] = transition_tokens
        row_scaffolds[idx] = scaffolds

    components: dict[str, list[int]] = defaultdict(list)
    for idx, tokens in row_tokens.items():
        components[uf.find(tokens[0])].append(idx)

    groups: dict[str, dict[str, Any]] = {}
    row_to_group: dict[int, str] = {}
    for root, indices in components.items():
        material = sorted({token for idx in indices for token in row_tokens[idx] if not token.startswith("row:")})
        group_id = _stable_id(*(material or [root]))
        group_rows = [rows[idx] for idx in indices]
        group = {
            "group_id": group_id,
            "row_indices": sorted(indices),
            "n_rows": len(indices),
            "route_domain": _mode(row.get("route_domain") for row in group_rows),
            "depth_bucket": _depth_bucket(max(int(row.get("depth") or len(row.get("gt_route") or [])) for row in group_rows)),
            "transition_tokens": {token for idx in indices for token in row_transition_tokens[idx]},
            "scaffolds": {scaffold for idx in indices for scaffold in row_scaffolds[idx]},
        }
        groups[group_id] = group
        for idx in indices:
            row_to_group[idx] = group_id
    return groups, row_to_group


def _route_transition_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for step in row.get("gt_route") or []:
        if not isinstance(step, dict):
            continue
        rxn = canonical_reaction(str(step.get("rxn_smiles") or ""))
        if not rxn or ">>" not in rxn:
            continue
        lhs, rhs = rxn.split(">>", 1)
        reactants = sorted(canonical_side(lhs))
        products = sorted(canonical_side(rhs))
        if len(products) != 1 or not reactants:
            continue
        product = products[0]
        main = _largest_smiles(reactants)
        reactant_set = ".".join(reactants)
        tokens.add(f"product:{product}")
        tokens.add(f"reaction:{rxn}")
        tokens.add(f"product_main:{product}<<{main}")
        tokens.add(f"product_reactants:{product}<<{reactant_set}")
    return tokens


def _assign_splits(groups: dict[str, dict[str, Any]], *, fractions: dict[str, float], split_salt: str) -> dict[str, str]:
    total_rows = sum(int(group.get("n_rows") or 0) for group in groups.values())
    target_rows = {split: total_rows * fraction for split, fraction in fractions.items()}
    out: dict[str, str] = {}
    current_rows = {split: 0 for split in fractions}
    ordered = sorted(
        groups.values(),
        key=lambda group: (
            -int(group.get("n_rows") or 0),
            str(group.get("route_domain") or ""),
            str(group.get("depth_bucket") or ""),
            _stable_hash(split_salt, group["group_id"]),
        ),
    )
    for group in ordered:
        underfilled = [split for split in fractions if current_rows[split] < target_rows[split]]
        if underfilled:
            split = sorted(
                underfilled,
                key=lambda name: (
                    -(target_rows[name] - current_rows[name]),
                    current_rows[name],
                    _stable_hash(split_salt, group["group_id"], name),
                ),
            )[0]
        else:
            split = sorted(
                fractions,
                key=lambda name: (
                    current_rows[name] / max(target_rows[name], 1.0),
                    current_rows[name],
                    _stable_hash(split_salt, group["group_id"], name),
                ),
            )[0]
        out[group["group_id"]] = split
        current_rows[split] += int(group.get("n_rows") or 0)
    return out


def _leakage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "doi_cross_split": _cross_split_values(rows, lambda row: _norm(row.get("doi"))),
        "target_cross_split": _cross_split_values(rows, lambda row: canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")),
        "scaffold_cross_split": _cross_split_values(rows, lambda row: "|".join(row.get("split_scaffolds") or []), ignore_empty=True),
        "transition_token_cross_split": _cross_split_transition_tokens(rows),
        "product_cross_split": _cross_split_transition_tokens(rows, prefix="product:"),
        "reaction_cross_split": _cross_split_transition_tokens(rows, prefix="reaction:"),
        "product_main_cross_split": _cross_split_transition_tokens(rows, prefix="product_main:"),
        "product_reactants_cross_split": _cross_split_transition_tokens(rows, prefix="product_reactants:"),
    }
    checks["strict_pass"] = all(int((value or {}).get("count") or 0) == 0 for value in checks.values() if isinstance(value, dict))
    return checks


def _cross_split_transition_tokens(rows: list[dict[str, Any]], *, prefix: str | None = None) -> dict[str, Any]:
    value_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split") or "")
        for token in row.get("split_transition_tokens") or []:
            token = str(token)
            if prefix and not token.startswith(prefix):
                continue
            value_splits[token].add(split)
    offenders = {value: sorted(splits) for value, splits in value_splits.items() if len(splits) > 1}
    return {"count": len(offenders), "examples": dict(sorted(offenders.items())[:25])}


def _cross_split_values(rows: list[dict[str, Any]], getter: Any, *, ignore_empty: bool = True) -> dict[str, Any]:
    value_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = str(getter(row) or "")
        if ignore_empty and not value:
            continue
        value_splits[value].add(str(row.get("split") or ""))
    offenders = {value: sorted(splits) for value, splits in value_splits.items() if len(splits) > 1}
    return {"count": len(offenders), "examples": dict(sorted(offenders.items())[:25])}


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    token_sets = _tokens_by_split(rows, "transition_token")
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        out[split] = {
            "rows": len(split_rows),
            "unique_targets": len({row.get("target_smiles") for row in split_rows if row.get("target_smiles")}),
            "unique_doi": len({_norm(row.get("doi")) for row in split_rows if _norm(row.get("doi"))}),
            "unique_transition_tokens": len(token_sets[split]),
            "route_domain_counts": dict(Counter(row.get("route_domain") for row in split_rows)),
            "depth_counts": dict(Counter(int(row.get("depth") or len(row.get("gt_route") or [])) for row in split_rows)),
            "quality_tier_counts": dict(Counter(row.get("quality_tier") for row in split_rows)),
            "compatibility_label_counts": dict(Counter(row.get("compatibility_label") for row in split_rows)),
        }
    return out


def _tokens_by_split(rows: list[dict[str, Any]], _kind: str) -> dict[str, set[str]]:
    out = {"all": set(), "train": set(), "val": set(), "test": set()}
    for row in rows:
        split = str(row.get("split") or "")
        for token in row.get("split_transition_tokens") or []:
            out["all"].add(str(token))
            if split in out:
                out[split].add(str(token))
    return out


def _markdown(manifest: dict[str, Any]) -> str:
    counts = manifest.get("counts") or {}
    leakage = manifest.get("leakage_checks") or {}
    lines = [
        "# CascadeBench Strict Split",
        "",
        f"- schema: `{manifest.get('schema_version')}`",
        f"- candidate rows: `{counts.get('candidate_rows_after_optional_limit')}`",
        f"- split groups: `{counts.get('split_groups')}`",
        f"- strict pass: `{leakage.get('strict_pass')}`",
        "",
        "## Splits",
        "",
        "| Split | Rows | Unique Targets | Unique DOI | Transition Tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in (manifest.get("splits") or {}).items():
        lines.append(
            f"| {split} | {row.get('rows')} | {row.get('unique_targets')} | {row.get('unique_doi')} | {row.get('unique_transition_tokens')} |"
        )
    lines.extend(["", "## Leakage Checks", "", "| Check | Count |", "|---|---:|"])
    for key, value in leakage.items():
        if isinstance(value, dict):
            lines.append(f"| {key} | {value.get('count')} |")
    lines.extend(["", "## Outputs", ""])
    for key, path in (manifest.get("outputs") or {}).items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def _normalized_fractions(**values: float) -> dict[str, float]:
    clean = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("split fractions must sum to a positive value")
    return {key: value / total for key, value in clean.items()}


def _target_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None or mol.GetRingInfo().NumRings() == 0:
        return ""
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_mol is None:
        return ""
    if scaffold_mol.GetNumHeavyAtoms() < 8 and scaffold_mol.GetRingInfo().NumRings() <= 1:
        return ""
    return Chem.MolToSmiles(scaffold_mol, isomericSmiles=False) or ""


def _largest_smiles(values: list[str]) -> str:
    if not values:
        return ""
    scored = []
    for smi in values:
        mol = Chem.MolFromSmiles(smi)
        scored.append((mol.GetNumHeavyAtoms() if mol is not None else len(smi), smi))
    return max(scored)[1]


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


def _mode(values: Any) -> str:
    counter = Counter(str(value or "unknown") for value in values)
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0] if counter else "unknown"


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_norm(row.get("doi")), _norm(row.get("cascade_id")), str(row.get("target_smiles") or ""))


def _stable_id(*parts: Any) -> str:
    return hashlib.sha1("\t".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def _stable_hash(*parts: Any) -> str:
    return hashlib.sha1("\t".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    ap = argparse.ArgumentParser(description="Build strict-blind CascadeBench splits")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    manifest = build_cascadebench_strict_splits(
        v4_jsonl=Path(args.v4_jsonl),
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output_dir),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_salt=args.split_salt,
        limit=args.limit,
    )
    print(json.dumps({"counts": manifest["counts"], "splits": manifest["splits"], "leakage_checks": manifest["leakage_checks"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
