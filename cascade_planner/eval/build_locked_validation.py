"""Build an overlap-audited locked validation benchmark.

The locked split is selected from local cascade data while excluding target and
reaction identities used by route-tree training traces, dev benchmarks, and
proposal-ranker packs. The output format matches ``benchmark_v2_100.json`` so
it can be consumed by the live benchmark and trace collection runners.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles


RDLogger.DisableLog("rdApp.*")


def build_locked_validation(
    *,
    source_paths: Iterable[Path],
    output_path: Path,
    audit_json_path: Path,
    audit_md_path: Path | None = None,
    exclude_benchmarks: Iterable[Path] = (),
    exclude_jsonl: Iterable[Path] = (),
    limit: int = 24,
    seed: int = 80508,
    min_depth: int = 2,
) -> dict[str, Any]:
    excluded = _collect_excluded_sets(
        benchmarks=list(exclude_benchmarks),
        jsonl_paths=list(exclude_jsonl),
    )
    candidates: list[dict[str, Any]] = []
    scan = Counter()
    for source in source_paths:
        rows = _rows_from_source(source)
        scan["source_rows"] += len(rows)
        for row in rows:
            scan["candidate_rows"] += 1
            gt_rxns: set[str] = set()
            gt_step_count = 0
            for step in row.get("gt_route") or []:
                keys = _rxn_keys(step.get("rxn_smiles") or "")
                if keys:
                    gt_step_count += 1
                    gt_rxns.update(keys)
            target = _canonical_target(row.get("target_smiles") or "")
            if not target:
                scan["invalid_target"] += 1
                continue
            if gt_step_count < min_depth:
                scan["too_shallow"] += 1
                continue
            if target in excluded["targets"]:
                scan["target_overlap_excluded"] += 1
                continue
            overlapping = sorted(set(gt_rxns) & excluded["reactions"])
            if overlapping:
                scan["reaction_overlap_excluded"] += 1
                continue
            item = dict(row)
            item["target_smiles"] = target
            item["depth"] = int(row.get("depth") or len(row.get("gt_route") or []))
            item["_locked_validation_id"] = _stable_id(
                item.get("doi"),
                item.get("cascade_id"),
                target,
                "|".join(sorted(gt_rxns)),
            )
            item["_locked_source_path"] = str(source)
            candidates.append(item)

    selected = _select_stratified(candidates, limit=limit, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    selected_targets, selected_rxns = _benchmark_sets(selected)
    target_overlap = sorted(selected_targets & excluded["targets"])
    reaction_overlap = sorted(selected_rxns & excluded["reactions"])
    audit = {
        "schema_version": "locked_validation_audit.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": str(output_path),
        "limit": int(limit),
        "seed": int(seed),
        "min_depth": int(min_depth),
        "sources": [str(path) for path in source_paths],
        "exclude_benchmarks": [str(path) for path in exclude_benchmarks],
        "exclude_jsonl": [str(path) for path in exclude_jsonl],
        "scan": dict(scan),
        "excluded": {
            "target_count": len(excluded["targets"]),
            "reaction_count": len(excluded["reactions"]),
            "sources": excluded["sources"],
        },
        "selected": _selection_summary(selected),
        "overlap": {
            "target_overlap_count": len(target_overlap),
            "reaction_overlap_count": len(reaction_overlap),
            "target_overlap_values": target_overlap,
            "reaction_overlap_values": reaction_overlap,
            "locked_safe": not target_overlap and not reaction_overlap,
        },
        "conclusion": "locked_validation_safe" if not target_overlap and not reaction_overlap else "overlap_detected",
    }
    audit_json_path.parent.mkdir(parents=True, exist_ok=True)
    audit_json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit_md_path:
        audit_md_path.parent.mkdir(parents=True, exist_ok=True)
        audit_md_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return audit


def _collect_excluded_sets(*, benchmarks: list[Path], jsonl_paths: list[Path]) -> dict[str, Any]:
    targets: set[str] = set()
    reactions: set[str] = set()
    sources: list[dict[str, Any]] = []
    for path in benchmarks:
        rows = _read_benchmark(path)
        t, r = _benchmark_sets(rows)
        targets.update(t)
        reactions.update(r)
        sources.append({"path": str(path), "kind": "benchmark", "targets": len(t), "reactions": len(r)})
    for path in _expand_jsonl_paths(jsonl_paths):
        row_count = 0
        before_t = len(targets)
        before_r = len(reactions)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row_count += 1
                row = json.loads(line)
                for key in ("target_smiles", "target", "product"):
                    target = _canonical_target(row.get(key))
                    if target:
                        targets.add(target)
                _collect_exclusion_reactions(row, reactions)
        sources.append({
            "path": str(path),
            "kind": "jsonl",
            "rows": row_count,
            "targets_added": len(targets) - before_t,
            "reactions_added": len(reactions) - before_r,
        })
    return {"targets": targets, "reactions": reactions, "sources": sources}


def _expand_jsonl_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.glob("*.jsonl")))
        elif path.exists():
            out.append(path)
    return out


def _rows_from_source(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data and all("target_smiles" in row for row in data if isinstance(row, dict)):
        return [dict(row) for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            rows = data.get(key)
            if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
                return [dict(row) for row in rows]
    return _rows_from_cascade_records(data)


def _rows_from_cascade_records(data: Any) -> list[dict[str, Any]]:
    records = data if isinstance(data, list) else data.get("records", []) if isinstance(data, dict) else []
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        doi = record.get("doi") or record.get("title") or "unknown"
        for cascade in record.get("cascades") or []:
            steps = [step for step in cascade.get("steps") or [] if isinstance(step, dict)]
            if len(steps) < 2:
                continue
            target = _target_from_cascade(cascade, steps)
            if not target:
                continue
            gt_route = []
            for step in steps:
                rxn = step.get("rxn_smiles") or step.get("reaction_smiles") or ""
                if ">>" not in str(rxn):
                    continue
                gt_route.append({
                    "rxn_smiles": str(rxn),
                    "ec_number": _step_ec(step),
                    "transformation": (
                        step.get("transformation_superclass")
                        or step.get("transformation_name")
                        or step.get("reaction_type")
                        or ""
                    ),
                    "step_role": step.get("step_role") or "",
                })
            if len(gt_route) < 2:
                continue
            rows.append({
                "doi": doi,
                "cascade_id": cascade.get("cascade_id") or "",
                "target_smiles": target,
                "route_domain": cascade.get("route_domain") or "unknown",
                "operation_mode": cascade.get("operation_mode") or "unknown",
                "depth": len(gt_route),
                "gt_route": gt_route,
            })
    return rows


def _target_from_cascade(cascade: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    choices: list[str] = []
    for item in cascade.get("target_products") or []:
        if isinstance(item, dict):
            choices.append(str(item.get("smiles") or ""))
    for item in (steps[-1].get("output_species") or []):
        if isinstance(item, dict):
            role = str(item.get("species_role") or item.get("role") or "").lower()
            if "product" in role:
                choices.append(str(item.get("smiles") or ""))
    rxn = str(steps[-1].get("rxn_smiles") or "")
    if ">>" in rxn:
        choices.extend(part.strip() for part in rxn.split(">>", 1)[1].split(".") if part.strip())
    return _largest_valid_smiles(choices)


def _step_ec(step: dict[str, Any]) -> str | None:
    if step.get("ec_number"):
        return str(step.get("ec_number"))
    for item in step.get("catalyst_components") or []:
        if isinstance(item, dict) and item.get("ec_number"):
            return str(item.get("ec_number"))
    return None


def _select_stratified(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_doi: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("doi") or ""), str(item.get("cascade_id") or ""), str(item.get("target_smiles") or ""))):
        doi = str(row.get("doi") or "")
        if doi and doi in by_doi:
            continue
        by_doi.add(doi)
        unique.append(row)
    rng.shuffle(unique)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        depth = int(row.get("depth") or len(row.get("gt_route") or []))
        depth_bucket = "2" if depth == 2 else "3" if depth == 3 else "4+"
        buckets[(str(row.get("route_domain") or "unknown"), depth_bucket)].append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: str(row.get("_locked_validation_id") or row.get("target_smiles") or ""))
    return selected


def _read_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _benchmark_sets(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    targets: set[str] = set()
    reactions: set[str] = set()
    for row in rows:
        target = _canonical_target(row.get("target_smiles") or row.get("target") or "")
        if target:
            targets.add(target)
        for step in row.get("gt_route") or []:
            reactions.update(_rxn_keys((step or {}).get("rxn_smiles") or (step or {}).get("reaction_smiles") or ""))
    return targets, reactions


def _collect_reactions(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for key in ("rxn_smiles", "reaction_smiles", "canonical_reaction"):
            rxn = _canonical_rxn(value.get(key))
            if rxn:
                out.add(rxn)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                _collect_reactions(nested, out)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                _collect_reactions(item, out)


def _collect_exclusion_reactions(row: dict[str, Any], out: set[str]) -> None:
    """Collect row-level training reactions without walking huge negative pools."""
    for key in ("rxn_smiles", "reaction_smiles", "canonical_reaction"):
        rxn = _raw_rxn(row.get(key))
        if rxn:
            out.add(rxn)
    for step in row.get("gt_route") or []:
        if isinstance(step, dict):
            rxn = _raw_rxn(step.get("rxn_smiles") or step.get("reaction_smiles") or "")
            if rxn:
                out.add(rxn)
    candidate = row.get("candidate")
    if isinstance(candidate, dict):
        rxn = _raw_rxn(candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or "")
        if rxn:
            out.add(rxn)
    event = row.get("event")
    if isinstance(event, dict):
        state = event.get("state") or {}
        for step in state.get("steps") or []:
            action = (step or {}).get("action") or {}
            rxn = _raw_rxn(action.get("rxn_smiles") or action.get("reaction_smiles") or "")
            if rxn:
                out.add(rxn)


def _selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains = Counter(str(row.get("route_domain") or "unknown") for row in rows)
    depths = Counter(str(row.get("depth") or len(row.get("gt_route") or [])) for row in rows)
    return {
        "n_rows": len(rows),
        "unique_targets": len({_canonical_target(row.get("target_smiles")) for row in rows}),
        "gt_reaction_count": sum(len(row.get("gt_route") or []) for row in rows),
        "domain_counts": dict(domains),
        "depth_counts": dict(sorted(depths.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999)),
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    selected = audit.get("selected") or {}
    overlap = audit.get("overlap") or {}
    lines = [
        "# Locked Validation Overlap Audit",
        "",
        f"Output: `{audit.get('output_path')}`",
        f"Conclusion: `{audit.get('conclusion')}`",
        "",
        "## Selected",
        "",
        f"- rows: `{selected.get('n_rows')}`",
        f"- unique targets: `{selected.get('unique_targets')}`",
        f"- GT reactions: `{selected.get('gt_reaction_count')}`",
        f"- domain_counts: `{selected.get('domain_counts')}`",
        f"- depth_counts: `{selected.get('depth_counts')}`",
        "",
        "## Overlap",
        "",
        f"- target_overlap_count: `{overlap.get('target_overlap_count')}`",
        f"- reaction_overlap_count: `{overlap.get('reaction_overlap_count')}`",
        f"- locked_safe: `{overlap.get('locked_safe')}`",
        "",
        "## Exclusion Sources",
        "",
        "| Path | Kind | Rows | Targets Added | Reactions Added |",
        "|---|---|---:|---:|---:|",
    ]
    for source in (audit.get("excluded") or {}).get("sources") or []:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                source.get("path"),
                source.get("kind"),
                source.get("rows", ""),
                source.get("targets_added", source.get("targets", "")),
                source.get("reactions_added", source.get("reactions", "")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _canonical_target(value: Any) -> str:
    return canonical_smiles(str(value or "")) or ""


def _canonical_rxn(value: Any) -> str:
    text = str(value or "")
    if ">>" not in text:
        return ""
    return canonical_reaction(text) or text


def _raw_rxn(value: Any) -> str:
    text = str(value or "").strip()
    return text if ">>" in text else ""


def _rxn_keys(value: Any) -> set[str]:
    raw = _raw_rxn(value)
    if not raw:
        return set()
    can = _canonical_rxn(raw)
    return {raw, can} if can and can != raw else {raw}


def _largest_valid_smiles(values: Iterable[str]) -> str:
    best = ""
    best_atoms = -1
    for value in values:
        can = _canonical_target(value)
        if not can:
            continue
        mol = Chem.MolFromSmiles(can)
        atoms = mol.GetNumHeavyAtoms() if mol is not None else 0
        if atoms > best_atoms:
            best = can
            best_atoms = atoms
    return best if best_atoms >= 6 else ""


def _stable_id(*parts: Any) -> str:
    import hashlib

    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an overlap-audited locked validation benchmark")
    ap.add_argument("--source", action="append", required=True, help="Source benchmark/cascade JSON; can be repeated")
    ap.add_argument("--output", default="data/benchmark_locked_validation_20260508.json")
    ap.add_argument("--audit-json", default="results/shared/locked_validation_20260508/overlap_audit.json")
    ap.add_argument("--audit-md", default="results/shared/locked_validation_20260508/overlap_audit.md")
    ap.add_argument("--exclude-benchmark", action="append", default=[])
    ap.add_argument("--exclude-jsonl", action="append", default=[])
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", type=int, default=80508)
    ap.add_argument("--min-depth", type=int, default=2)
    args = ap.parse_args()
    audit = build_locked_validation(
        source_paths=[Path(path) for path in args.source],
        output_path=Path(args.output),
        audit_json_path=Path(args.audit_json),
        audit_md_path=Path(args.audit_md) if args.audit_md else None,
        exclude_benchmarks=[Path(path) for path in args.exclude_benchmark],
        exclude_jsonl=[Path(path) for path in args.exclude_jsonl],
        limit=args.limit,
        seed=args.seed,
        min_depth=args.min_depth,
    )
    print(json.dumps({"output": args.output, "audit": args.audit_json, **audit["overlap"]}, indent=2))


if __name__ == "__main__":
    main()
