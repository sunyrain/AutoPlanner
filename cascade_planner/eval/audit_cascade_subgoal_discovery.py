"""Generic cascade subgoal discovery over native route pools.

This is a prototype audit, not a statin-specific scorer.  It asks whether
substructures or route leaves from arbitrary targets are close to v4 cascade
products, so a downstream planner could propose a cascade-supported subgoal
without hand-written drug-family rules.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, BRICS, Recap, rdFMCS

from cascade_planner.cascade_search.v4_product_value import canonical_smiles, route_record_from_native_route, stable_id
from cascade_planner.eval.audit_v4_heldout_block_recovery import _load_routes, _native_rank

RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "cascade_subgoal_discovery_audit.v1"


def audit_cascade_subgoal_discovery(
    *,
    route_pool: Path,
    program_manifest: Path,
    output_json: Path,
    output_md: Path | None = None,
    benchmark: Path | None = None,
    evidence_split: str = "all",
    min_subgoal_heavy_atoms: int = 8,
    min_evidence_heavy_atoms: int = 7,
    max_routes_per_target: int = 50,
    similarity_threshold: float = 0.42,
    mcs_prefilter_threshold: float = 0.25,
    mcs_top_n: int = 80,
    top_k: int = 8,
) -> dict[str, Any]:
    started = time.monotonic()
    target_routes = _load_target_route_groups(route_pool, benchmark=benchmark)
    evidence = _build_evidence_index(
        program_manifest,
        split=evidence_split,
        min_heavy_atoms=min_evidence_heavy_atoms,
    )
    targets = []
    for target_smiles, payload in sorted(target_routes.items()):
        group = payload["routes"]
        if max_routes_per_target > 0:
            group = group[: int(max_routes_per_target)]
        candidates = _target_subgoals(
            target_smiles,
            group,
            min_heavy_atoms=min_subgoal_heavy_atoms,
        )
        matches = _match_subgoals(
            candidates,
            evidence["items"],
            threshold=similarity_threshold,
            mcs_prefilter_threshold=mcs_prefilter_threshold,
            mcs_top_n=mcs_top_n,
            top_k=top_k,
        )
        targets.append(
            {
                "target_id": payload.get("target_id"),
                "target_smiles": target_smiles,
                "route_count": len(group),
                "subgoal_count": len(candidates),
                "matched_subgoal_count": sum(1 for row in matches if row.get("matches")),
                "best_score": max((row.get("best_score") or 0.0 for row in matches), default=0.0),
                "best_similarity": max((row.get("best_similarity") or 0.0 for row in matches), default=0.0),
                "top_subgoals": matches[:top_k],
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool": str(route_pool),
            "benchmark": str(benchmark) if benchmark else None,
            "program_manifest": str(program_manifest),
            "evidence_split": evidence_split,
            "min_subgoal_heavy_atoms": min_subgoal_heavy_atoms,
            "min_evidence_heavy_atoms": min_evidence_heavy_atoms,
            "max_routes_per_target": max_routes_per_target,
            "similarity_threshold": similarity_threshold,
            "mcs_prefilter_threshold": mcs_prefilter_threshold,
            "mcs_top_n": mcs_top_n,
            "top_k": top_k,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": (
                "generic subgoal discovery from target fragments and route leaves; "
                "no target-name, drug-family, or statin-specific scoring"
            ),
        },
        "evidence_index": evidence["summary"],
        "summary": _summary(targets),
        "targets": targets,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _build_evidence_index(program_manifest: Path, *, split: str, min_heavy_atoms: int) -> dict[str, Any]:
    manifest = json.loads(Path(program_manifest).read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    splits = ["train", "val", "test"] if split == "all" else [split]
    items_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped = Counter()
    for split_name in splits:
        if split_name not in outputs:
            raise ValueError(f"split {split_name!r} not found in {program_manifest}")
        for program in _read_jsonl(Path(outputs[split_name])):
            for role, smiles, step in _program_evidence_smiles(program):
                smi = canonical_smiles(str(smiles or ""))
                props = _mol_props(smi)
                if not props["valid"]:
                    skipped["invalid_smiles"] += 1
                    continue
                if props["heavy_atoms"] < min_heavy_atoms:
                    skipped["too_small"] += 1
                    continue
                key = (str(program.get("program_id") or ""), role, smi)
                items_by_key[key] = {
                    "evidence_id": stable_id(split_name, program.get("program_id"), role, smi),
                    "split": split_name,
                    "program_id": program.get("program_id"),
                    "doi": program.get("doi"),
                    "cascade_id": program.get("cascade_id"),
                    "cascade_type": program.get("cascade_type"),
                    "quality_tier": program.get("quality_tier"),
                    "quality_score": program.get("quality_score"),
                    "role": role,
                    "smiles": smi,
                    "heavy_atoms": props["heavy_atoms"],
                    "ring_count": props["ring_count"],
                    "hetero_atoms": props["hetero_atoms"],
                    "transformation_superclass": (step or {}).get("transformation_superclass"),
                    "transformation_name": (step or {}).get("transformation_name"),
                    "step_role": (step or {}).get("step_role"),
                    "pairwise_mode": (step or {}).get("pairwise_mode"),
                    "intermediate_isolated": (step or {}).get("intermediate_isolated"),
                    "catalyst_classes": (step or {}).get("catalyst_classes") or [],
                    "enzyme_families": (step or {}).get("enzyme_families") or [],
                    "route_transforms": [s.get("transformation_superclass") for s in program.get("steps") or [] if isinstance(s, dict)],
                    "compatibility_label": ((program.get("compatibility") or {}).get("compatibility_label")),
                    "evidence_strength": ((program.get("compatibility") or {}).get("evidence_strength")),
                    "catalyst_combination_summary": program.get("catalyst_combination_summary"),
                    "overall_yield": program.get("overall_yield"),
                    "fingerprint": _fp(smi),
                    "mol": Chem.MolFromSmiles(smi),
                }
    items = [row for row in items_by_key.values() if row.get("fingerprint") is not None]
    role_counts = Counter(row["role"] for row in items)
    transform_counts = Counter(str(row.get("transformation_superclass") or "unknown").lower() for row in items)
    return {
        "items": items,
        "summary": {
            "split": split,
            "items": len(items),
            "role_counts": dict(role_counts.most_common()),
            "top_transforms": dict(transform_counts.most_common(20)),
            "skipped": dict(skipped),
        },
    }


def _program_evidence_smiles(program: dict[str, Any]) -> list[tuple[str, str, dict[str, Any] | None]]:
    rows: list[tuple[str, str, dict[str, Any] | None]] = []
    if program.get("target_smiles"):
        rows.append(("program_target", str(program.get("target_smiles")), None))
    for step in program.get("steps") or []:
        if not isinstance(step, dict):
            continue
        product = step.get("product_smiles")
        if product:
            rows.append(("step_product", str(product), step))
        # Reactants are useful as purchasable-entry clues, but they are lower
        # ranked later to avoid tiny common-fragment pollution.
        for reactant in step.get("reactants") or []:
            if reactant:
                rows.append(("step_reactant", str(reactant), step))
    return rows


def _load_target_route_groups(route_pool: Path, *, benchmark: Path | None) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    benchmark_rows = _read_rows(benchmark) if benchmark else []
    benchmark_by_index = {idx: row for idx, row in enumerate(benchmark_rows)}
    if route_pool.suffix == ".jsonl":
        for route in _load_routes(route_pool):
            target = canonical_smiles(str(route.get("target_smiles") or ""))
            if target:
                groups.setdefault(target, {"target_id": route.get("target_id") or target, "routes": []})["routes"].append(route)
        for idx, row in benchmark_by_index.items():
            target = canonical_smiles(str(row.get("target_smiles") or ""))
            if target:
                groups.setdefault(target, {"target_id": _target_id(row, idx), "routes": []})
        return _sort_route_groups(groups)

    run = json.loads(route_pool.read_text(encoding="utf-8"))
    if not isinstance(run, dict):
        raise ValueError(f"unsupported route pool format: {route_pool}")
    for target_index, target_row in enumerate(run.get("targets") or []):
        if not isinstance(target_row, dict):
            continue
        bench_row = benchmark_by_index.get(target_index, {})
        target_smiles = canonical_smiles(str(target_row.get("target_smiles") or bench_row.get("target_smiles") or ""))
        if not target_smiles:
            continue
        target_id = _target_id(target_row, target_index)
        if target_id == str(target_index) and bench_row:
            target_id = _target_id(bench_row, target_index)
        payload = groups.setdefault(target_smiles, {"target_id": target_id, "routes": []})
        payload["target_id"] = payload.get("target_id") or target_id
        for native_rank, route in enumerate(target_row.get("routes") or []):
            if isinstance(route, dict):
                payload["routes"].append(
                    route_record_from_native_route(
                        route,
                        target_smiles=target_smiles,
                        target_id=str(payload["target_id"] or target_id),
                        native_rank=_native_rank(route, fallback=native_rank),
                        dataset=str((run.get("metadata") or {}).get("schema_version") or "native_route_pool"),
                    )
                )
    for idx, row in benchmark_by_index.items():
        target = canonical_smiles(str(row.get("target_smiles") or ""))
        if target:
            groups.setdefault(target, {"target_id": _target_id(row, idx), "routes": []})
    return _sort_route_groups(groups)


def _sort_route_groups(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for payload in groups.values():
        payload["routes"].sort(key=lambda row: int(row.get("native_rank") or 0))
    return dict(groups)


def _target_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("target_id") or row.get("cascade_id") or row.get("target_name") or row.get("name") or index)


def _read_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows = data.get("targets") or data.get("items") or data.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"unsupported row format: {path}")


def _target_subgoals(
    target_smiles: str,
    routes: list[dict[str, Any]],
    *,
    min_heavy_atoms: int,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    fragment_cache: dict[str, set[str]] = {}

    def add(smiles: str, source: str, rank: int | None = None) -> None:
        smi = canonical_smiles(smiles)
        props = _mol_props(smi)
        if not props["valid"] or props["heavy_atoms"] < min_heavy_atoms:
            return
        if props["heavy_atoms"] > 80:
            return
        row = rows.setdefault(
            smi,
            {
                "subgoal_id": stable_id(smi),
                "smiles": smi,
                "heavy_atoms": props["heavy_atoms"],
                "ring_count": props["ring_count"],
                "hetero_atoms": props["hetero_atoms"],
                "sources": [],
                "best_route_rank": None,
            },
        )
        if source not in row["sources"]:
            row["sources"].append(source)
        if rank is not None:
            if row["best_route_rank"] is None or int(rank) < int(row["best_route_rank"]):
                row["best_route_rank"] = int(rank)

    add(target_smiles, "target", None)
    for frag in _cached_fragments(target_smiles, fragment_cache):
        add(frag, "target_fragment", None)
    for route in routes:
        rank = int(route.get("native_rank") or 0)
        for step in route.get("steps") or []:
            if not isinstance(step, dict):
                continue
            if step.get("product_smiles"):
                add(str(step.get("product_smiles")), "route_step_product", rank)
            for reactant in step.get("reactant_smiles") or step.get("reactants") or []:
                if reactant:
                    add(str(reactant), "route_leaf_or_reactant", rank)
                    for frag in _cached_fragments(str(reactant), fragment_cache):
                        add(frag, "route_leaf_fragment", rank)
    return sorted(rows.values(), key=lambda row: (-int(row["heavy_atoms"]), row["smiles"]))


def _cached_fragments(smiles: str, cache: dict[str, set[str]]) -> set[str]:
    key = canonical_smiles(str(smiles or ""))
    if not key:
        return set()
    if key not in cache:
        cache[key] = _fragments(key)
    return cache[key]


def _fragments(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return set()
    out: set[str] = set()
    try:
        out.update(_strip_dummy(smi) for smi in BRICS.BRICSDecompose(mol))
    except Exception:
        pass
    try:
        recap = Recap.RecapDecompose(mol)
        out.update(_strip_dummy(smi) for smi in recap.GetLeaves())
    except Exception:
        pass
    return {canonical_smiles(smi) for smi in out if canonical_smiles(smi)}


def _strip_dummy(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    try:
        mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#0]"))
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    if mol is None or mol.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _match_subgoals(
    candidates: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    *,
    threshold: float,
    mcs_prefilter_threshold: float,
    mcs_top_n: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if not evidence_items:
        return []
    evidence_fps = [row["fingerprint"] for row in evidence_items]
    role_counts = Counter(row["role"] for row in evidence_items)
    mcs_cache: dict[tuple[str, str], dict[str, float]] = {}
    out = []
    for cand in candidates:
        cfp = _fp(str(cand["smiles"]))
        if cfp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(cfp, evidence_fps)
        candidate_indices = [
            idx for idx in np.argsort(np.asarray(sims, dtype=float))[-max(1, int(mcs_top_n)) :][::-1]
            if float(sims[int(idx)]) >= mcs_prefilter_threshold
        ]
        matches = []
        for idx in candidate_indices:
            idx = int(idx)
            sim = float(sims[idx])
            ev = evidence_items[idx]
            mcs = {"atoms": 0.0, "query_coverage": 0.0, "evidence_coverage": 0.0}
            motif_similarity = sim
            if sim < threshold:
                cache_key = (str(cand["smiles"]), str(ev.get("smiles") or ""))
                mcs = mcs_cache.get(cache_key)
                if mcs is None:
                    mcs = _mcs_coverage(cache_key[0], cache_key[1])
                    mcs_cache[cache_key] = mcs
                motif_similarity = max(sim, 0.55 * mcs["query_coverage"] + 0.45 * mcs["evidence_coverage"])
            if motif_similarity < threshold:
                continue
            score = _match_score(float(sim), motif_similarity, mcs, cand, ev, role_counts)
            matches.append({"score": score, "similarity": float(sim), "motif_similarity": motif_similarity, "mcs": mcs, "evidence": ev})
        matches.sort(key=lambda row: (row["score"], row["similarity"]), reverse=True)
        compact_matches = [_compact_match(row) for row in matches[:top_k]]
        out.append(
            {
                **cand,
                "best_score": compact_matches[0]["score"] if compact_matches else 0.0,
                "best_similarity": compact_matches[0]["similarity"] if compact_matches else 0.0,
                "matches": compact_matches,
            }
        )
    out.sort(
        key=lambda row: (
            float(row.get("best_score") or 0.0),
            float(row.get("best_similarity") or 0.0),
            int(row.get("heavy_atoms") or 0),
        ),
        reverse=True,
    )
    return out


def _match_score(
    similarity: float,
    motif_similarity: float,
    mcs: dict[str, float],
    cand: dict[str, Any],
    ev: dict[str, Any],
    role_counts: Counter[str],
) -> float:
    role = str(ev.get("role") or "")
    role_weight = {
        "program_target": 1.0,
        "step_product": 0.92,
        "step_reactant": 0.62,
    }.get(role, 0.5)
    quality = 1.0 if ev.get("quality_tier") == "gold" else 0.85
    evidence_strength = 1.08 if ev.get("evidence_strength") == "strong_process_evidence" else 1.0
    cand_info = _information_weight(cand)
    ev_info = _information_weight(ev)
    source_bonus = 0.0
    sources = set(cand.get("sources") or [])
    if "route_leaf_or_reactant" in sources or "route_leaf_fragment" in sources:
        source_bonus += 0.05
    if "target_fragment" in sources:
        source_bonus += 0.03
    rarity = 1.0 / math.log2(3 + role_counts.get(role, 1))
    mcs_bonus = 0.08 * min(mcs["query_coverage"], mcs["evidence_coverage"])
    return round(
        float(motif_similarity) * role_weight * quality * evidence_strength * cand_info * ev_info
        + source_bonus
        + 0.08 * rarity
        + mcs_bonus,
        6,
    )


def _information_weight(row: dict[str, Any]) -> float:
    heavy = int(row.get("heavy_atoms") or 0)
    hetero = int(row.get("hetero_atoms") or 0)
    ring = int(row.get("ring_count") or 0)
    size = min(1.0, max(0.0, (heavy - 7.0) / 13.0))
    hetero_bonus = min(0.18, 0.045 * max(0, hetero - 1))
    ring_bonus = 0.04 if ring else 0.0
    return min(1.2, 0.55 + 0.45 * size + hetero_bonus + ring_bonus)


def _mcs_coverage(query_smiles: str, evidence_smiles: str) -> dict[str, float]:
    query = Chem.MolFromSmiles(str(query_smiles or ""))
    evidence = Chem.MolFromSmiles(str(evidence_smiles or ""))
    if query is None or evidence is None:
        return {"atoms": 0.0, "query_coverage": 0.0, "evidence_coverage": 0.0}
    try:
        result = rdFMCS.FindMCS(
            [query, evidence],
            timeout=1,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        atoms = float(result.numAtoms or 0)
    except Exception:
        atoms = 0.0
    return {
        "atoms": atoms,
        "query_coverage": round(atoms / max(float(query.GetNumHeavyAtoms()), 1.0), 6),
        "evidence_coverage": round(atoms / max(float(evidence.GetNumHeavyAtoms()), 1.0), 6),
    }


def _compact_match(row: dict[str, Any]) -> dict[str, Any]:
    ev = row["evidence"]
    return {
        "score": round(float(row["score"]), 6),
        "similarity": round(float(row["similarity"]), 6),
        "motif_similarity": round(float(row["motif_similarity"]), 6),
        "mcs": row.get("mcs"),
        "evidence_id": ev.get("evidence_id"),
        "split": ev.get("split"),
        "role": ev.get("role"),
        "doi": ev.get("doi"),
        "program_id": ev.get("program_id"),
        "cascade_id": ev.get("cascade_id"),
        "quality_tier": ev.get("quality_tier"),
        "cascade_type": ev.get("cascade_type"),
        "smiles": ev.get("smiles"),
        "heavy_atoms": ev.get("heavy_atoms"),
        "transformation_superclass": ev.get("transformation_superclass"),
        "transformation_name": ev.get("transformation_name"),
        "route_transforms": ev.get("route_transforms"),
        "compatibility_label": ev.get("compatibility_label"),
        "evidence_strength": ev.get("evidence_strength"),
        "catalyst_summary": ev.get("catalyst_combination_summary"),
        "overall_yield": ev.get("overall_yield"),
    }


def _summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    target_count = len(targets)
    matched = [row for row in targets if row.get("matched_subgoal_count")]
    return {
        "targets": target_count,
        "targets_with_matches": len(matched),
        "target_match_rate": round(len(matched) / max(target_count, 1), 6),
        "mean_subgoal_count": round(float(np.mean([row.get("subgoal_count") or 0 for row in targets])) if targets else 0.0, 6),
        "mean_matched_subgoal_count": round(float(np.mean([row.get("matched_subgoal_count") or 0 for row in targets])) if targets else 0.0, 6),
        "best_score_max": round(max((row.get("best_score") or 0.0 for row in targets), default=0.0), 6),
        "best_similarity_max": round(max((row.get("best_similarity") or 0.0 for row in targets), default=0.0), 6),
    }


def _mol_props(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"valid": False, "heavy_atoms": 0, "ring_count": 0, "hetero_atoms": 0}
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    return {
        "valid": True,
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "ring_count": int(mol.GetRingInfo().NumRings()),
        "hetero_atoms": int(hetero),
    }


def _fp(smiles: str) -> Any:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    meta = result.get("metadata") or {}
    summary = result.get("summary") or {}
    evidence = result.get("evidence_index") or {}
    lines = [
        "# Cascade Subgoal Discovery Audit",
        "",
        "## Contract",
        "",
        f"- Route pool: `{meta.get('route_pool')}`",
        f"- Evidence split: `{meta.get('evidence_split')}`",
        f"- Similarity threshold: `{meta.get('similarity_threshold')}`",
        f"- Contract: {meta.get('contract')}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Evidence Index", "", "| Metric | Value |", "|---|---:|"])
    for key in ("split", "items", "role_counts", "top_transforms", "skipped"):
        lines.append(f"| `{key}` | `{json.dumps(evidence.get(key), ensure_ascii=False)}` |")
    lines.extend(["", "## Targets", "", "| target | routes | subgoals | matched | best score | best match | evidence |", "|---|---:|---:|---:|---:|---|---|"])
    for target in result.get("targets") or []:
        best = (target.get("top_subgoals") or [{}])[0]
        match = (best.get("matches") or [{}])[0]
        evidence_text = ""
        if match:
            evidence_text = f"{match.get('doi')} / {match.get('role')} / {match.get('transformation_superclass')}"
        lines.append(
            "| `{target}` | {routes} | {subgoals} | {matched} | {score:.4f} | `{subgoal}` | {evidence} |".format(
                target=(target.get("target_smiles") or "")[:32],
                routes=target.get("route_count"),
                subgoals=target.get("subgoal_count"),
                matched=target.get("matched_subgoal_count"),
                score=float(target.get("best_score") or 0.0),
                subgoal=(best.get("smiles") or "")[:42],
                evidence=evidence_text,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A match means a target fragment or route leaf is structurally close to a v4 cascade product or step product.",
            "- This audit does not prove a full route is feasible.",
            "- High-scoring subgoals are candidates for later evidence-conditioned route stitching.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit generic cascade-supported subgoal discovery on a route pool")
    ap.add_argument("--route-pool", required=True)
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    ap.add_argument("--benchmark")
    ap.add_argument("--evidence-split", default="all")
    ap.add_argument("--min-subgoal-heavy-atoms", type=int, default=8)
    ap.add_argument("--min-evidence-heavy-atoms", type=int, default=7)
    ap.add_argument("--max-routes-per-target", type=int, default=50)
    ap.add_argument("--similarity-threshold", type=float, default=0.42)
    ap.add_argument("--mcs-prefilter-threshold", type=float, default=0.25)
    ap.add_argument("--mcs-top-n", type=int, default=80)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()
    result = audit_cascade_subgoal_discovery(
        route_pool=Path(args.route_pool),
        program_manifest=Path(args.program_manifest),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        benchmark=Path(args.benchmark) if args.benchmark else None,
        evidence_split=args.evidence_split,
        min_subgoal_heavy_atoms=args.min_subgoal_heavy_atoms,
        min_evidence_heavy_atoms=args.min_evidence_heavy_atoms,
        max_routes_per_target=args.max_routes_per_target,
        similarity_threshold=args.similarity_threshold,
        mcs_prefilter_threshold=args.mcs_prefilter_threshold,
        mcs_top_n=args.mcs_top_n,
        top_k=args.top_k,
    )
    print(json.dumps({"summary": result["summary"], "output_json": args.output_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
