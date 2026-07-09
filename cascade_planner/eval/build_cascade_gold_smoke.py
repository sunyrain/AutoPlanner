"""Build a small v3 gold cascade benchmark for external-baseline smoke tests."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.cascadeboard.route_recovery import canonical_reaction

RDLogger.DisableLog("rdApp.*")


DEFAULT_DOMAIN_QUOTAS = {
    "all_chemical": 2,
    "chemoenzymatic": 4,
    "all_enzymatic": 4,
}


def build_cascade_gold_smoke(
    *,
    dataset_path: Path,
    output_path: Path,
    limit: int = 10,
    domain_quotas: dict[str, int] | None = None,
    exclude_pack_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows = _read_records(dataset_path)
    excluded_targets, excluded_reactions = _excluded_training_sets(exclude_pack_paths or [])
    candidates = []
    for record in rows:
        for cascade in record.get("cascades") or []:
            candidate = _candidate_from_cascade(record, cascade)
            if candidate is not None and not _overlaps_excluded(candidate, excluded_targets, excluded_reactions):
                candidates.append(candidate)

    selected = _select_stratified(candidates, limit=limit, domain_quotas=domain_quotas or DEFAULT_DOMAIN_QUOTAS)
    _validate_benchmark(selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected


def _candidate_from_cascade(record: dict[str, Any], cascade: dict[str, Any]) -> dict[str, Any] | None:
    qc = cascade.get("quality_control") or {}
    allowed_tasks = set(qc.get("allowed_tasks") or [])
    if qc.get("record_tier") != "gold":
        return None
    if "retrosynthesis" not in allowed_tasks:
        return None
    if cascade.get("representation_type") != "small_molecule":
        return None
    if cascade.get("is_demonstrated_success") is not True:
        return None

    target_products = cascade.get("target_products") or []
    target_smiles = str((target_products[0] or {}).get("smiles") or "") if target_products else ""
    if not target_smiles:
        return None

    productive_steps = [
        step for step in cascade.get("steps") or []
        if step.get("step_type") != "nonproductive"
    ]
    if not productive_steps:
        return None
    if any(not _step_route_ok(step) for step in productive_steps):
        return None
    if any(_is_identity_reaction(str(step.get("rxn_smiles") or "")) for step in productive_steps):
        return None

    gt_route = [_route_step_payload(step) for step in productive_steps]
    return {
        "doi": record.get("doi"),
        "title": record.get("title"),
        "record_uuid": record.get("record_uuid"),
        "cascade_id": cascade.get("cascade_id"),
        "cascade_uuid": cascade.get("cascade_uuid"),
        "target_smiles": target_smiles,
        "target_products": target_products,
        "starting_materials": cascade.get("starting_materials") or [],
        "route_domain": cascade.get("route_domain"),
        "operation_mode": cascade.get("operation_mode"),
        "depth": len(gt_route),
        "gt_route": gt_route,
        "global_conditions": cascade.get("global_conditions") or {},
        "overall_outcome": cascade.get("overall_outcome") or {},
        "catalyst_combination_summary": cascade.get("catalyst_combination_summary"),
        "quality_control": qc,
        "gold_smoke_selection": {
            "source_dataset": "cascade_dataset_v3.json",
            "selection_version": "v1",
            "criteria": [
                "quality_control.record_tier == gold",
                "allowed_tasks contains retrosynthesis",
                "representation_type == small_molecule",
                "is_demonstrated_success == true",
                "all productive steps have rxn_smiles_status == ok",
                "identity reactions removed",
            ],
        },
    }


def _route_step_payload(step: dict[str, Any]) -> dict[str, Any]:
    catalysts = step.get("catalyst_components") or []
    ec_numbers = [item.get("ec_number") for item in catalysts if item.get("ec_number")]
    uniprot_ids = [item.get("uniprot_id") for item in catalysts if item.get("uniprot_id")]
    return {
        "step_id": step.get("step_id"),
        "step_index": step.get("step_index"),
        "rxn_smiles": step.get("rxn_smiles"),
        "rxn_smiles_status": step.get("rxn_smiles_status"),
        "transformation": step.get("transformation_name") or step.get("transformation_superclass"),
        "transformation_superclass": step.get("transformation_superclass"),
        "step_role": step.get("step_role"),
        "input_species": step.get("input_species") or [],
        "output_species": step.get("output_species") or [],
        "condition": step.get("step_conditions") or {},
        "outcome": step.get("step_outcome") or {},
        "catalyst_components": catalysts,
        "ec_number": ec_numbers[0] if ec_numbers else None,
        "ec_numbers": ec_numbers,
        "uniprot_ids": uniprot_ids,
        "evidence_quote": step.get("evidence_quote"),
        "step_notes": step.get("step_notes"),
    }


def _select_stratified(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    domain_quotas: dict[str, int],
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_domain[str(row.get("route_domain") or "unknown")].append(row)
    for domain_rows in by_domain.values():
        domain_rows.sort(key=_selection_key)

    selected: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for domain, quota in domain_quotas.items():
        for row in by_domain.get(domain, []):
            if len([item for item in selected if item.get("route_domain") == domain]) >= quota:
                break
            target_key = _canon_smiles(row.get("target_smiles") or "") or row.get("target_smiles") or ""
            if target_key in seen_targets:
                continue
            selected.append(row)
            seen_targets.add(target_key)
            if len(selected) >= limit:
                return selected

    for row in sorted(candidates, key=_selection_key):
        target_key = _canon_smiles(row.get("target_smiles") or "") or row.get("target_smiles") or ""
        if target_key in seen_targets:
            continue
        selected.append(row)
        seen_targets.add(target_key)
        if len(selected) >= limit:
            return selected
    return selected


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    qc = row.get("quality_control") or {}
    review_flags = qc.get("review_flags") or []
    return (
        len(review_flags),
        int(row.get("depth") or 0),
        len(str(row.get("target_smiles") or "")),
        str(row.get("doi") or ""),
        str(row.get("cascade_id") or ""),
    )


def _validate_benchmark(rows: list[dict[str, Any]]) -> None:
    targets = [_canon_smiles(row.get("target_smiles") or "") or row.get("target_smiles") or "" for row in rows]
    duplicates = [target for target, count in Counter(targets).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate target_smiles in smoke benchmark: {duplicates}")
    identity = []
    for row in rows:
        for step in row.get("gt_route") or []:
            rxn = str(step.get("rxn_smiles") or "")
            if _is_identity_reaction(rxn):
                identity.append(rxn)
    if identity:
        raise ValueError(f"identity reactions in smoke benchmark: {identity[:3]}")


def _excluded_training_sets(pack_paths: list[Path]) -> tuple[set[str], set[str]]:
    targets: set[str] = set()
    reactions: set[str] = set()
    for pack in pack_paths:
        for file_path in _pack_files(pack):
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                target = row.get("target_smiles") or row.get("product")
                if target:
                    targets.add(str(target))
                    canonical = _canon_smiles(str(target))
                    if canonical:
                        targets.add(canonical)
                _collect_reactions(row, reactions)
    return targets, reactions


def _overlaps_excluded(candidate: dict[str, Any], excluded_targets: set[str], excluded_reactions: set[str]) -> bool:
    target = str(candidate.get("target_smiles") or "")
    if target in excluded_targets:
        return True
    canonical_target = _canon_smiles(target)
    if canonical_target and canonical_target in excluded_targets:
        return True
    for step in candidate.get("gt_route") or []:
        rxn = str(step.get("rxn_smiles") or "")
        if rxn in excluded_reactions:
            return True
        canonical_rxn = canonical_reaction(rxn)
        if canonical_rxn and canonical_rxn in excluded_reactions:
            return True
    return False


def _pack_files(pack: Path) -> list[Path]:
    if pack.is_file():
        return [pack]
    files = []
    for name in [
        "route_states.jsonl",
        "candidate_pools.jsonl",
        "step_pairs.jsonl",
        "search_transitions.jsonl",
        "route_value.jsonl",
        "candidate_ranking.jsonl",
        "skeleton_prior.jsonl",
        "failure_diagnosis.jsonl",
    ]:
        path = pack / name
        if path.exists():
            files.append(path)
    if not files:
        files.extend(sorted(pack.glob("*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no JSONL training files found in {pack}")
    return files


def _collect_reactions(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for key in ("rxn_smiles", "reaction_smiles"):
            rxn = value.get(key)
            if isinstance(rxn, str) and ">>" in rxn:
                out.add(rxn)
                canonical = canonical_reaction(rxn)
                if canonical:
                    out.add(canonical)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                _collect_reactions(nested, out)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                _collect_reactions(item, out)


def _step_route_ok(step: dict[str, Any]) -> bool:
    rxn = str(step.get("rxn_smiles") or "")
    return bool(rxn and ">>" in rxn and step.get("rxn_smiles_status") == "ok")


def _is_identity_reaction(rxn: str) -> bool:
    if not rxn or ">>" not in rxn:
        return False
    lhs, rhs = rxn.split(">>", 1)
    left = _canon_side(lhs)
    right = _canon_side(rhs)
    return left is not None and left == right


def _canon_side(side: str) -> str | None:
    parts = [part for part in side.split(".") if part]
    out = []
    for part in parts:
        canonical = _canon_smiles(part)
        if canonical is None:
            return None
        out.append(canonical)
    return ".".join(sorted(out))


def _canon_smiles(smiles: str) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _read_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("records_kept", "records", "items", "rows", "targets"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"unsupported cascade dataset format: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v3 gold/route-ok smoke benchmark")
    ap.add_argument("--dataset", default="cascade_dataset_v3.json")
    ap.add_argument("--output", default="data/benchmark_cascade_gold_smoke_v1.json")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--exclude-pack", action="append", default=[])
    args = ap.parse_args()
    selected = build_cascade_gold_smoke(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        limit=args.limit,
        exclude_pack_paths=[Path(path) for path in args.exclude_pack],
    )
    print(json.dumps({"output": args.output, "selected": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
