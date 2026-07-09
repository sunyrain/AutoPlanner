"""Build an explicit vNext learning pack from the consolidated training pack."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger

from cascade_planner.vnext.features import (
    candidate_reactants,
    candidate_reactant_smiles,
    normalize_candidate,
    read_jsonl,
    route_label_vector,
    stable_id,
    write_jsonl,
)
from cascade_planner.vnext.schema import VNEXT_FILES, VNEXT_SCHEMA_VERSION, schema_manifest


RDLogger.DisableLog("rdApp.warning")


def build_vnext_pack(
    *,
    pack_dir: Path,
    output_dir: Path,
    max_candidates: int = 32,
    cascade_data_paths: Iterable[Path] | None = None,
    external_step_pair_paths: Iterable[Path] | None = None,
    external_candidate_pool_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    return build_vnext_pack_from_sources(
        pack_dirs=[pack_dir],
        output_dir=output_dir,
        max_candidates=max_candidates,
        cascade_data_paths=cascade_data_paths,
        external_step_pair_paths=external_step_pair_paths,
        external_candidate_pool_paths=external_candidate_pool_paths,
    )


def build_vnext_pack_from_sources(
    *,
    pack_dirs: Iterable[Path],
    output_dir: Path,
    max_candidates: int = 32,
    cascade_data_paths: Iterable[Path] | None = None,
    external_step_pair_paths: Iterable[Path] | None = None,
    external_candidate_pool_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    pack_dirs = [Path(p) for p in pack_dirs]
    cascade_data_paths = [Path(p) for p in (cascade_data_paths or [])]
    external_step_pair_paths = [Path(p) for p in (external_step_pair_paths or [])]
    external_candidate_pool_paths = [Path(p) for p in (external_candidate_pool_paths or [])]
    candidate_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for pack_dir in pack_dirs:
        candidate_rows.extend(_read_optional(pack_dir / "candidate_ranking.jsonl"))
        route_rows.extend(_read_optional(pack_dir / "route_value.jsonl"))

    step_pairs = [_step_pair_from_candidate(row) for row in candidate_rows if row.get("product") and row.get("candidate")]
    step_pairs = [row for row in step_pairs if row]
    curated_step_pairs = _curated_step_pairs_from_cascade_data(cascade_data_paths)
    step_pairs.extend(curated_step_pairs)
    external_step_pairs = _external_step_pairs_from_paths(external_step_pair_paths)
    step_pairs.extend(external_step_pairs)
    step_pairs = _dedupe_rows(step_pairs, _step_pair_key)
    candidate_pools = _candidate_pool_rows(candidate_rows, max_candidates=max_candidates)
    external_candidate_pools = _external_candidate_pools_from_paths(external_candidate_pool_paths)
    candidate_pools.extend(external_candidate_pools)
    candidate_pools = _dedupe_rows(candidate_pools, _candidate_pool_key)
    route_states = [_route_state_from_row(row) for row in route_rows if row.get("target_smiles")]
    route_states = [row for row in route_states if row]
    route_states = _dedupe_route_states(route_states)
    search_transitions = [_search_transition_from_pool(row) for row in candidate_pools]
    search_transitions = [row for row in search_transitions if row]

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "step_pairs": str(output_dir / VNEXT_FILES["step_pairs"]),
        "candidate_pools": str(output_dir / VNEXT_FILES["candidate_pools"]),
        "route_states": str(output_dir / VNEXT_FILES["route_states"]),
        "search_transitions": str(output_dir / VNEXT_FILES["search_transitions"]),
    }
    write_jsonl(Path(files["step_pairs"]), step_pairs)
    write_jsonl(Path(files["candidate_pools"]), candidate_pools)
    write_jsonl(Path(files["route_states"]), route_states)
    write_jsonl(Path(files["search_transitions"]), search_transitions)

    counts = {
        "step_pairs": len(step_pairs),
        "candidate_pools": len(candidate_pools),
        "route_states": len(route_states),
        "search_transitions": len(search_transitions),
    }
    source_pack = ",".join(str(path) for path in pack_dirs)
    manifest = schema_manifest(source_pack=source_pack, counts=counts, files=files)
    manifest.update({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quality": _quality_summary(step_pairs, candidate_pools, route_states),
        "max_candidates": max_candidates,
        "source_packs": [str(path) for path in pack_dirs],
        "cascade_data_paths": [str(path) for path in cascade_data_paths],
        "external_step_pair_paths": [str(path) for path in external_step_pair_paths],
        "external_candidate_pool_paths": [str(path) for path in external_candidate_pool_paths],
        "curated_step_pairs": len(curated_step_pairs),
        "external_step_pairs": len(external_step_pairs),
        "external_candidate_pools": len(external_candidate_pools),
    })
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.md"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest["files"]["report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_report_markdown(manifest), encoding="utf-8")
    return manifest


def merge_external_step_pairs_into_vnext_pack(
    *,
    base_vnext_pack: Path,
    output_dir: Path,
    external_step_pair_paths: Iterable[Path],
    external_candidate_pool_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    base_vnext_pack = Path(base_vnext_pack)
    external_step_pair_paths = [Path(p) for p in external_step_pair_paths]
    external_candidate_pool_paths = [Path(p) for p in external_candidate_pool_paths]
    output_dir.mkdir(parents=True, exist_ok=True)

    base_step_pairs = _read_optional(base_vnext_pack / VNEXT_FILES["step_pairs"])
    external_step_pairs = _external_step_pairs_from_paths(external_step_pair_paths)
    step_pairs = _dedupe_rows([*base_step_pairs, *external_step_pairs], _step_pair_key)

    base_candidate_pools = _read_optional(base_vnext_pack / VNEXT_FILES["candidate_pools"])
    external_candidate_pools = _external_candidate_pools_from_paths(external_candidate_pool_paths)
    candidate_pools = _dedupe_rows([*base_candidate_pools, *external_candidate_pools], _candidate_pool_key)
    route_states = _read_optional(base_vnext_pack / VNEXT_FILES["route_states"])
    search_transitions = [_search_transition_from_pool(row) for row in candidate_pools]
    search_transitions = [row for row in search_transitions if row]

    files = {
        "step_pairs": str(output_dir / VNEXT_FILES["step_pairs"]),
        "candidate_pools": str(output_dir / VNEXT_FILES["candidate_pools"]),
        "route_states": str(output_dir / VNEXT_FILES["route_states"]),
        "search_transitions": str(output_dir / VNEXT_FILES["search_transitions"]),
    }
    write_jsonl(Path(files["step_pairs"]), step_pairs)
    write_jsonl(Path(files["candidate_pools"]), candidate_pools)
    _copy_or_write_jsonl(base_vnext_pack / VNEXT_FILES["route_states"], Path(files["route_states"]), route_states)
    write_jsonl(Path(files["search_transitions"]), search_transitions)

    counts = {
        "step_pairs": len(step_pairs),
        "candidate_pools": len(candidate_pools),
        "route_states": len(route_states),
        "search_transitions": len(search_transitions),
    }
    base_manifest = _read_manifest(base_vnext_pack / "manifest.json")
    manifest = schema_manifest(
        source_pack=base_manifest.get("source_pack") or str(base_vnext_pack),
        counts=counts,
        files=files,
    )
    manifest.update({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quality": _quality_summary(step_pairs, candidate_pools, route_states),
        "base_vnext_pack": str(base_vnext_pack),
        "base_step_pairs": len(base_step_pairs),
        "external_step_pair_paths": [str(path) for path in external_step_pair_paths],
        "external_candidate_pool_paths": [str(path) for path in external_candidate_pool_paths],
        "external_step_pairs": len(external_step_pairs),
        "external_candidate_pools": len(external_candidate_pools),
        "source_packs": base_manifest.get("source_packs") or [],
        "cascade_data_paths": base_manifest.get("cascade_data_paths") or [],
        "curated_step_pairs": base_manifest.get("curated_step_pairs", 0),
    })
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.md"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest["files"]["report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_report_markdown(manifest), encoding="utf-8")
    return manifest


def _read_optional(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _copy_or_write_jsonl(src: Path, dst: Path, rows: list[dict[str, Any]]) -> None:
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        write_jsonl(dst, rows)


def _external_step_pairs_from_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            normalized = _normalize_external_step_pair(row, source_path=path)
            if normalized:
                rows.append(normalized)
    return rows


def _external_candidate_pools_from_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            normalized = _normalize_external_candidate_pool(row)
            if normalized:
                rows.append(normalized)
    return rows


def _normalize_external_candidate_pool(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = row.get("candidates") or []
    product = row.get("product") or row.get("target_smiles") or ""
    if not product or not candidates:
        return None
    normalized_candidates = []
    for idx, item in enumerate(candidates, start=1):
        candidate = dict(item.get("candidate") or {})
        if not candidate:
            continue
        rank = int(item.get("rank") or idx)
        candidate.setdefault("rank", rank)
        normalized_candidates.append({
            "candidate_id": item.get("candidate_id") or stable_id("external_pool_candidate", row.get("pool_id"), rank, candidate),
            "rank": rank,
            "label": float(item.get("label") or 0.0),
            "label_type": item.get("label_type") or "external_candidate",
            "weight": float(item.get("weight") or 1.0),
            "gt_available": bool(item.get("gt_available", True)),
            "candidate": candidate,
        })
    if not normalized_candidates:
        return None
    return {
        "pool_id": row.get("pool_id") or stable_id("external_pool", product, len(normalized_candidates), row),
        "route_id": row.get("route_id") or "",
        "target_smiles": row.get("target_smiles") or product,
        "product": product,
        "step_index": int(row.get("step_index") or 0),
        "source": row.get("source") or "external_candidate_pool",
        "candidates": normalized_candidates,
        "has_exact_gt": bool(row.get("has_exact_gt", True)),
        "positive_count": sum(1 for item in normalized_candidates if float(item.get("label") or 0.0) >= 0.75),
    }


def _normalize_external_step_pair(row: dict[str, Any], *, source_path: Path) -> dict[str, Any] | None:
    product = row.get("product") or ""
    reaction_smiles = row.get("reaction_smiles") or ""
    candidate = dict(row.get("candidate") or {})
    reactants = row.get("reactants") or candidate_reactants(candidate)
    if not product or not reaction_smiles or not reactants:
        return None
    label_type = row.get("label_type") or "external_curated_step"
    source = row.get("source") or candidate.get("source") or "external"
    if not candidate:
        candidate = {
            "main_reactant": _largest_smiles([str(smi) for smi in reactants]),
            "aux_reactants": [str(smi) for smi in reactants],
            "source": source,
            "score": 1.0,
            "type": row.get("reaction_type") or "",
            "reaction_type": row.get("reaction_type") or "",
            "rxn_smiles": reaction_smiles,
            "ec": row.get("ec") or "",
        }
    candidate.setdefault("source", source)
    candidate.setdefault("rxn_smiles", reaction_smiles)
    candidate.setdefault("reaction_smiles", reaction_smiles)
    candidate.setdefault("reaction_type", row.get("reaction_type") or "")
    candidate.setdefault("type", row.get("reaction_type") or "")
    candidate.setdefault("ec", row.get("ec") or "")
    candidate.setdefault("score", 1.0)
    return {
        "step_id": row.get("step_id") or stable_id("external", source_path, reaction_smiles),
        "group_id": row.get("group_id") or stable_id("external_group", source, product),
        "route_id": row.get("route_id", ""),
        "target_smiles": row.get("target_smiles") or product,
        "product": product,
        "reactants": [str(smi) for smi in reactants],
        "reaction_smiles": reaction_smiles,
        "reaction_type": row.get("reaction_type") or candidate.get("reaction_type") or "",
        "ec": row.get("ec") or candidate.get("ec") or "",
        "source": source,
        "rank": int(row.get("rank") or 1),
        "label": float(row.get("label") or 1.0),
        "label_type": label_type,
        "weight": float(row.get("weight") or 1.0),
        "gt_available": bool(row.get("gt_available", True)),
        "exact_gt_reaction": bool(row.get("exact_gt_reaction", True)),
        "exact_gt_reactants": bool(row.get("exact_gt_reactants", True)),
        "selected_exact": bool(row.get("selected_exact", True)),
        "candidate": candidate,
    }


def _step_pair_from_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidate = normalize_candidate(row.get("candidate") or {})
    product = row.get("product") or ""
    if not product:
        return None
    reactants = candidate_reactants(candidate)
    return {
        "step_id": stable_id(row.get("candidate_id"), row.get("route_id"), row.get("step_index"), row.get("rank")),
        "group_id": stable_id(row.get("route_id"), row.get("step_index")),
        "route_id": row.get("route_id", ""),
        "target_smiles": row.get("target_smiles", ""),
        "product": product,
        "reactants": reactants,
        "reaction_smiles": candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or "",
        "reaction_type": candidate.get("type") or candidate.get("reaction_type") or "",
        "ec": candidate.get("ec") or "",
        "source": candidate.get("source") or "unknown",
        "rank": int(row.get("rank") or candidate.get("rank") or 1),
        "label": float(row.get("label") or 0.0),
        "label_type": row.get("label_type", ""),
        "weight": float(row.get("weight") or 1.0),
        "gt_available": bool(row.get("gt_available")),
        "exact_gt_reaction": bool(row.get("exact_gt_reaction")),
        "exact_gt_reactants": bool(row.get("exact_gt_reactants")),
        "selected_exact": bool(row.get("selected_exact")),
        "candidate": candidate,
    }


def _candidate_pool_rows(rows: list[dict[str, Any]], *, max_candidates: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("product") and row.get("candidate"):
            grouped[(str(row.get("route_id") or ""), int(row.get("step_index") or 0))].append(row)
    pools = []
    for (route_id, step_index), items in grouped.items():
        items = sorted(items, key=lambda row: int(row.get("rank") or 999999))[:max_candidates]
        first = items[0]
        candidates = []
        for row in items:
            candidate = normalize_candidate(row.get("candidate") or {})
            candidates.append({
                "candidate_id": row.get("candidate_id") or stable_id(route_id, step_index, row.get("rank"), candidate),
                "rank": int(row.get("rank") or candidate.get("rank") or len(candidates) + 1),
                "label": float(row.get("label") or 0.0),
                "label_type": row.get("label_type", ""),
                "weight": float(row.get("weight") or 1.0),
                "gt_available": bool(row.get("gt_available")),
                "candidate": candidate,
            })
        pools.append({
            "pool_id": stable_id(route_id, step_index, first.get("product"), len(candidates)),
            "route_id": route_id,
            "target_smiles": first.get("target_smiles", ""),
            "product": first.get("product", ""),
            "step_index": step_index,
            "candidates": candidates,
            "has_exact_gt": any(c.get("label_type") == "benchmark_exact" for c in candidates),
            "positive_count": sum(1 for c in candidates if float(c.get("label") or 0.0) >= 0.75),
        })
    return pools


def _route_state_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    labels = route_label_vector(row)
    return {
        "route_state_id": row.get("route_id") or stable_id(row.get("source_path"), row.get("target_smiles"), row.get("route_index")),
        "route_id": row.get("route_id", ""),
        "source_path": row.get("source_path", ""),
        "target_smiles": row.get("target_smiles", ""),
        "route_domain": row.get("route_domain", ""),
        "operation_mode": row.get("operation_mode", ""),
        "n_steps": row.get("n_steps") or len(row.get("type_sequence") or []),
        "type_sequence": row.get("type_sequence") or [],
        "ec1_sequence": row.get("ec1_sequence") or [],
        "source_sequence": row.get("source_sequence") or [],
        "step_reactions": row.get("step_reactions") or [],
        "features": row.get("features") or {},
        "metrics_summary": row.get("metrics_summary") or {},
        "label": float(row.get("label") or 0.0),
        "label_type": row.get("label_type", ""),
        "solved": float(labels["solved"]),
        "progressive": float(labels["progressive"]),
        "stock_closed": float(labels["stock_closed"]),
        "compatibility": float(labels["compatibility"]),
        "bottleneck_labels": row.get("recovery_bottleneck_labels") or [],
        "recovery_bottleneck": row.get("recovery_bottleneck", ""),
    }


def _search_transition_from_pool(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = row.get("candidates") or []
    if not candidates:
        return None
    labels = [float(item.get("label") or 0.0) for item in candidates]
    best_index = max(range(len(labels)), key=lambda idx: labels[idx]) if labels else -1
    if best_index < 0 or labels[best_index] <= 0.0:
        best_index = -1
    return {
        "transition_id": stable_id("transition", row.get("pool_id"), row.get("route_id"), row.get("step_index")),
        "source": "candidate_pool_distillation",
        "route_id": row.get("route_id", ""),
        "pool_id": row.get("pool_id", ""),
        "target_smiles": row.get("target_smiles", ""),
        "product": row.get("product", ""),
        "step_index": row.get("step_index", 0),
        "action_count": len(candidates),
        "best_action_index": best_index,
        "best_action_label": labels[best_index] if best_index >= 0 else 0.0,
        "action_labels": labels,
        "action_candidate_ids": [item.get("candidate_id", "") for item in candidates],
        "reward": max(labels) if labels else 0.0,
        "has_positive_action": any(label >= 0.75 for label in labels),
    }


def _curated_step_pairs_from_cascade_data(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for record in _iter_cascade_records(data):
            doi = record.get("doi", "")
            title = record.get("title", "")
            for cascade in record.get("cascades") or []:
                route_domain = cascade.get("route_domain", "")
                operation_mode = cascade.get("operation_mode", "")
                for step in cascade.get("steps") or []:
                    row = _curated_step_pair(record, cascade, step, source_path=str(path), doi=doi, title=title)
                    if not row:
                        continue
                    key = _step_pair_key(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    row["route_domain"] = route_domain
                    row["operation_mode"] = operation_mode
                    rows.append(row)
    return rows


def _iter_cascade_records(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        if isinstance(data.get("cascades"), list):
            yield data
        for key in ("records", "targets", "data", "items"):
            child = data.get(key)
            if child is not None:
                yield from _iter_cascade_records(child)


def _curated_step_pair(
    record: dict[str, Any],
    cascade: dict[str, Any],
    step: dict[str, Any],
    *,
    source_path: str,
    doi: str,
    title: str,
) -> dict[str, Any] | None:
    rxn = step.get("rxn_smiles") or step.get("reaction_smiles") or ""
    parsed = _parse_reaction_smiles(rxn)
    if not parsed:
        return None
    reactants, products = parsed
    main = _largest_smiles(reactants)
    if not main or not products:
        return None
    aux = [smi for smi in reactants if smi != main]
    product = _largest_smiles(products)
    catalysts = step.get("catalyst_components") or []
    ec = ""
    uniprot = ""
    catalyst_class = ""
    for catalyst in catalysts:
        ec = ec or str(catalyst.get("ec_number") or "")
        uniprot = uniprot or str(catalyst.get("uniprot_id") or catalyst.get("uniprot_accession") or "")
        catalyst_class = catalyst_class or str(catalyst.get("catalyst_class") or "")
    conditions = step.get("step_conditions") or {}
    reaction_type = step.get("transformation_superclass") or step.get("transformation_name") or step.get("step_role") or ""
    candidate = {
        "main_reactant": main,
        "aux_reactants": aux,
        "source": "curated_cascade_step",
        "score": 1.0,
        "type": reaction_type,
        "reaction_type": reaction_type,
        "rxn_smiles": rxn,
        "ec": ec,
        "T": conditions.get("temperature_c"),
        "pH": conditions.get("ph"),
        "solvent": conditions.get("solvent") or "",
        "catalyst": "; ".join(c.get("component_name", "") for c in catalysts if c.get("component_name")),
        "evidence": {
            "doi": doi,
            "title": title,
            "uniprot_accession": uniprot,
            "catalyst_class": catalyst_class,
        },
    }
    step_id = stable_id("curated", source_path, record.get("record_uuid"), cascade.get("cascade_uuid"), step.get("step_uuid"), rxn)
    return {
        "step_id": step_id,
        "group_id": stable_id("curated_group", source_path, record.get("record_uuid"), cascade.get("cascade_uuid"), step.get("step_index")),
        "route_id": cascade.get("cascade_uuid") or cascade.get("cascade_id") or "",
        "target_smiles": product,
        "product": product,
        "reactants": reactants,
        "reaction_smiles": rxn,
        "reaction_type": reaction_type,
        "ec": ec,
        "source": "curated_cascade_step",
        "rank": 1,
        "label": 1.0,
        "label_type": "curated_step",
        "weight": 2.5,
        "gt_available": True,
        "exact_gt_reaction": True,
        "exact_gt_reactants": True,
        "selected_exact": True,
        "candidate": candidate,
    }


def _parse_reaction_smiles(rxn: str) -> tuple[list[str], list[str]] | None:
    if not rxn or ">>" not in rxn:
        return None
    lhs, rhs = rxn.split(">>", 1)
    reactants = _valid_smiles_parts(lhs)
    products = _valid_smiles_parts(rhs)
    if not reactants or not products:
        return None
    return reactants, products


def _valid_smiles_parts(side: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in side.split("."):
        smi = part.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
    return out


def _largest_smiles(values: list[str]) -> str:
    return max(values, key=_heavy_atoms, default="")


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _dedupe_rows(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = key_fn(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe_route_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[str] = []
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _route_state_key(row)
        existing = by_key.get(key)
        if existing is None:
            ordered.append(key)
            by_key[key] = row
            continue
        merged_labels = sorted(set(existing.get("bottleneck_labels") or []) | set(row.get("bottleneck_labels") or []))
        if merged_labels:
            existing["bottleneck_labels"] = merged_labels
        if not existing.get("recovery_bottleneck") and row.get("recovery_bottleneck"):
            existing["recovery_bottleneck"] = row.get("recovery_bottleneck")
        if _route_state_richness(row) > _route_state_richness(existing):
            replacement = dict(row)
            replacement["bottleneck_labels"] = merged_labels
            if not replacement.get("recovery_bottleneck"):
                replacement["recovery_bottleneck"] = existing.get("recovery_bottleneck", "")
            by_key[key] = replacement
    return [by_key[key] for key in ordered]


def _route_state_richness(row: dict[str, Any]) -> int:
    return (
        5 * len(row.get("bottleneck_labels") or [])
        + int(bool(row.get("recovery_bottleneck")))
        + int(bool(row.get("operation_mode")))
        + int(bool(row.get("route_domain")))
    )


def _step_pair_key(row: dict[str, Any]) -> str:
    if row.get("step_id"):
        return str(row["step_id"])
    cand = row.get("candidate") or {}
    return stable_id(
        "step_pair",
        row.get("target_smiles") or row.get("product"),
        row.get("product"),
        row.get("reaction_smiles") or cand.get("rxn_smiles") or cand.get("reaction_smiles"),
        candidate_reactant_smiles(cand),
        row.get("reaction_type"),
        row.get("ec"),
        row.get("label_type"),
    )


def _candidate_pool_key(row: dict[str, Any]) -> str:
    if row.get("pool_id"):
        return str(row["pool_id"])
    sig = [
        (
            item.get("rank"),
            item.get("label_type"),
            (item.get("candidate") or {}).get("canonical_reaction")
            or (item.get("candidate") or {}).get("rxn_smiles")
            or candidate_reactant_smiles(item.get("candidate") or {}),
        )
        for item in row.get("candidates") or []
    ]
    return stable_id("candidate_pool", row.get("target_smiles"), row.get("product"), row.get("step_index"), json.dumps(sig, sort_keys=True))


def _route_state_key(row: dict[str, Any]) -> str:
    if row.get("route_state_id"):
        return str(row["route_state_id"])
    return stable_id(
        "route_state",
        row.get("target_smiles"),
        row.get("label_type"),
        row.get("operation_mode"),
        json.dumps(row.get("type_sequence") or [], sort_keys=True),
        json.dumps(row.get("ec1_sequence") or [], sort_keys=True),
        json.dumps(row.get("bottleneck_labels") or [], sort_keys=True),
    )


def _quality_summary(
    step_pairs: list[dict[str, Any]],
    candidate_pools: list[dict[str, Any]],
    route_states: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "step_pair_labels": dict(Counter(row.get("label_type") for row in step_pairs)),
        "curated_step_pairs": sum(1 for row in step_pairs if row.get("label_type") == "curated_step"),
        "candidate_pools_with_exact_gt": sum(1 for row in candidate_pools if row.get("has_exact_gt")),
        "candidate_pools_with_positive": sum(1 for row in candidate_pools if int(row.get("positive_count") or 0) > 0),
        "route_labels": dict(Counter(row.get("label_type") for row in route_states)),
        "route_bottlenecks": dict(Counter(label for row in route_states for label in row.get("bottleneck_labels") or [])),
    }


def _report_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest.get("counts") or {}
    quality = manifest.get("quality") or {}
    lines = [
        "# vNext Training Pack",
        "",
        f"Schema: `{manifest.get('schema_version') or VNEXT_SCHEMA_VERSION}`",
        f"Source pack: `{manifest.get('source_pack')}`",
        "",
        "## Counts",
        "",
        f"- step pairs: `{counts.get('step_pairs', 0)}`",
        f"- candidate pools: `{counts.get('candidate_pools', 0)}`",
        f"- route states: `{counts.get('route_states', 0)}`",
        f"- search transitions: `{counts.get('search_transitions', 0)}`",
        "",
        "## Quality",
        "",
        f"- step labels: `{quality.get('step_pair_labels', {})}`",
        f"- curated step pairs: `{quality.get('curated_step_pairs', 0)}`",
        f"- external step pairs: `{manifest.get('external_step_pairs', 0)}`",
        f"- external candidate pools: `{manifest.get('external_candidate_pools', 0)}`",
        f"- pools with exact GT: `{quality.get('candidate_pools_with_exact_gt', 0)}`",
        f"- pools with positive candidate: `{quality.get('candidate_pools_with_positive', 0)}`",
        f"- route labels: `{quality.get('route_labels', {})}`",
        f"- route bottlenecks: `{quality.get('route_bottlenecks', {})}`",
        "",
        "## Runtime Contract",
        "",
        "This pack trains optional route-level models. Frozen single-step engines still generate chemical candidates.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build vNext step/pool/route pack from a consolidated training pack")
    ap.add_argument("--pack-dir", action="append", default=None, help="Existing consolidated training pack; repeat for a merged full pack")
    ap.add_argument("--base-vnext-pack", default=None, help="Existing vNext pack to copy while appending external step pairs")
    ap.add_argument("--output-dir", default="results/shared/vnext_pack/current")
    ap.add_argument("--max-candidates", type=int, default=32)
    ap.add_argument("--cascade-data", action="append", default=None, help="Optional curated cascade dataset JSON to add as single-step positives")
    ap.add_argument("--external-step-pairs", action="append", default=None, help="Optional external_step_pairs.jsonl file to add as single-step positives")
    ap.add_argument("--external-candidate-pools", action="append", default=None, help="Optional external_candidate_pools.jsonl file to add as listwise ranking/search supervision")
    args = ap.parse_args()
    if args.base_vnext_pack:
        manifest = merge_external_step_pairs_into_vnext_pack(
            base_vnext_pack=Path(args.base_vnext_pack),
            output_dir=Path(args.output_dir),
            external_step_pair_paths=[Path(p) for p in (args.external_step_pairs or [])],
            external_candidate_pool_paths=[Path(p) for p in (args.external_candidate_pools or [])],
        )
    else:
        manifest = build_vnext_pack_from_sources(
            pack_dirs=[Path(p) for p in (args.pack_dir or ["results/shared/training_pack/broad_20260507"])],
            output_dir=Path(args.output_dir),
            max_candidates=args.max_candidates,
            cascade_data_paths=[Path(p) for p in (args.cascade_data or [])],
            external_step_pair_paths=[Path(p) for p in (args.external_step_pairs or [])],
            external_candidate_pool_paths=[Path(p) for p in (args.external_candidate_pools or [])],
        )
    print(json.dumps({"output_dir": args.output_dir, "counts": manifest.get("counts"), "quality": manifest.get("quality")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
