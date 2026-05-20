"""Strict non-oracle provider-to-ChemEnzy bridge audit.

This audit asks whether a train-backed cascade retrieval provider can connect
to ChemEnzy one-step candidates without peeking at the held-out cascade block.

Construction contract:
  target -> connector comes only from cached ChemEnzy one-step candidates.
  connector <- provider_reactant comes only from v4-train retrieval queried by
  the ChemEnzy candidate itself.

Held-out v4 blocks are used only after construction, as labels for reporting.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.cascade_retrieval_provider import CascadeRetrievalProvider
from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.cascadeboard.route_recovery import canonical_side
from cascade_planner.eval.audit_provider_routepool_oracle import _load_reference_blocks, _transition_fp


SCHEMA_VERSION = "strict_nonoracle_provider_bridge_audit.v1"


def audit_nonoracle_provider_bridge(
    *,
    program_manifest: Path,
    chem_enzy_cache: Path,
    output_json: Path,
    route_recovery_json: Path | None = None,
    split: str = "test",
    only_routed: bool = True,
    max_targets: int | None = None,
    max_chem_candidates: int = 100,
    main_reactant_only: bool = False,
    min_connector_heavy_atoms: int = 0,
    provider_limit: int = 20,
    min_similarity: float = 0.20,
    bridge_similarity: float = 0.70,
    analog_similarity: float = 0.55,
    connected_ref_similarity: float = 0.0,
    modes: tuple[str, ...] = ("block_downstream_transition",),
    oracle_transform_filter: bool = False,
    top_ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
) -> dict[str, Any]:
    started = time.monotonic()
    provider = CascadeRetrievalProvider(program_manifest)
    cache = _read_json(chem_enzy_cache)
    cache_index = _cache_index(cache)
    refs = _load_reference_blocks(program_manifest, split=split)
    refs_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        if connected_ref_similarity > 0.0 and _reference_connection_similarity(ref) < float(connected_ref_similarity):
            continue
        target = canonical_smiles(str(ref.get("target_smiles") or "")) or str(ref.get("target_smiles") or "")
        if target:
            refs_by_target[target].append(ref)

    targets = _target_list(
        refs_by_target=refs_by_target,
        route_recovery_json=route_recovery_json,
        only_routed=only_routed,
        cache_index=cache_index,
    )
    if max_targets is not None:
        targets = targets[: max(0, int(max_targets))]

    target_rows = []
    for target in targets:
        refs_for_target = refs_by_target.get(target) or []
        downstream_candidates = _downstream_candidates(
            cache_index,
            target,
            max_chem_candidates=max_chem_candidates,
            main_reactant_only=main_reactant_only,
            min_connector_heavy_atoms=min_connector_heavy_atoms,
        )
        downstream_label = _label_downstream_candidates(
            downstream_candidates,
            refs_for_target,
            analog_similarity=analog_similarity,
        )
        bridge_rows = _build_bridge_rows(
            provider=provider,
            target=target,
            downstream_candidates=downstream_candidates,
            refs_for_target=refs_for_target,
            modes=modes,
            provider_limit=provider_limit,
            min_similarity=min_similarity,
            bridge_similarity=bridge_similarity,
            analog_similarity=analog_similarity,
            oracle_transform_filter=oracle_transform_filter,
        )
        bridge_rows.sort(key=_nonoracle_sort_key)
        target_rows.append(
            {
                "target_smiles": target,
                "reference_blocks": len(refs_for_target),
                "chem_enzy_candidate_rows": len(_candidate_rows(cache_index, target)[:max_chem_candidates]),
                "chem_enzy_downstream_connectors": len(downstream_candidates),
                "downstream_analog_any": bool(downstream_label.get("analog_any")),
                "downstream_analog_top_rank": downstream_label.get("best_analog_rank"),
                "downstream_best_similarity": downstream_label.get("best_similarity"),
                "bridge_rows": len(bridge_rows),
                "bridge_any": any(row.get("bridge_hit") for row in bridge_rows),
                "analog_bridge_any": any(row.get("analog_hit") and row.get("bridge_hit") for row in bridge_rows),
                "pair_and_analog_bridge_any": any(row.get("pair_and_analog") and row.get("bridge_hit") for row in bridge_rows),
                "best_pair_and_analog_bridge_rank": _best_rank(bridge_rows, "pair_and_analog"),
                "best_analog_bridge_rank": _best_rank(bridge_rows, "analog_hit"),
                "topk": _topk_hits(bridge_rows, top_ks=top_ks),
                "examples": bridge_rows[:10],
                "reference_examples": _reference_examples(refs_for_target),
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "chem_enzy_cache": str(chem_enzy_cache),
            "route_recovery_json": str(route_recovery_json) if route_recovery_json else None,
            "split": split,
            "only_routed": bool(only_routed),
            "max_targets": max_targets,
            "max_chem_candidates": max_chem_candidates,
            "main_reactant_only": bool(main_reactant_only),
            "min_connector_heavy_atoms": min_connector_heavy_atoms,
            "provider_limit": provider_limit,
            "min_similarity": min_similarity,
            "bridge_similarity": bridge_similarity,
            "analog_similarity": analog_similarity,
            "connected_ref_similarity": connected_ref_similarity,
            "modes": list(modes),
            "oracle_transform_filter": bool(oracle_transform_filter),
            "top_ks": list(top_ks),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": (
                "Strict non-oracle: provider queries are built from target plus "
                "ChemEnzy candidate connector only; held-out references are used "
                "only for post-hoc labels. If oracle_transform_filter is true, "
                "held-out transform labels are deliberately used as a diagnostic "
                "upper bound and must not be treated as deployable."
            ),
        },
        "provider_summary": provider.summary,
        "reference_summary": _reference_summary(refs_by_target),
        "summary": _summary(target_rows, top_ks=top_ks),
        "top_reference_pairs": dict(Counter(ref.get("transform_pair") for ref in refs).most_common(20)),
        "targets": target_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _target_list(
    *,
    refs_by_target: dict[str, list[dict[str, Any]]],
    route_recovery_json: Path | None,
    only_routed: bool,
    cache_index: dict[str, list[dict[str, Any]]],
) -> list[str]:
    if route_recovery_json is not None:
        payload = json.loads(route_recovery_json.read_text(encoding="utf-8"))
        targets = []
        for row in payload.get("targets") or []:
            if not isinstance(row, dict):
                continue
            if only_routed and int(row.get("route_count") or 0) <= 0:
                continue
            target = canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
            if target and target in refs_by_target:
                targets.append(target)
        seen = set()
        out = []
        for target in targets:
            if target not in seen:
                seen.add(target)
                out.append(target)
        return out
    return sorted(target for target in refs_by_target if target in cache_index)


def _cache_index(cache: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for raw_key, rows in cache.items():
        try:
            key = json.loads(raw_key)
        except Exception:
            continue
        product = canonical_smiles(str(key.get("product") or "")) or str(key.get("product") or "")
        if not product or not isinstance(rows, list):
            continue
        out[product] = [row for row in rows if isinstance(row, dict)]
    return out


def _candidate_rows(cache_index: dict[str, list[dict[str, Any]]], target: str) -> list[dict[str, Any]]:
    canonical = canonical_smiles(str(target or "")) or str(target or "")
    return cache_index.get(canonical) or []


def _downstream_candidates(
    cache_index: dict[str, list[dict[str, Any]]],
    target: str,
    *,
    max_chem_candidates: int,
    main_reactant_only: bool,
    min_connector_heavy_atoms: int,
) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, int]] = set()
    for idx, cand in enumerate(_candidate_rows(cache_index, target)[: max(0, int(max_chem_candidates))], start=1):
        rxn = str(cand.get("reaction_smiles") or cand.get("rxn_smiles") or "")
        lhs = rxn.split(">>", 1)[0] if ">>" in rxn else ""
        reactants = [value for value in canonical_side(lhs) if value]
        main = canonical_smiles(str(cand.get("main_reactant") or "")) or str(cand.get("main_reactant") or "")
        if main_reactant_only:
            reactants = [main] if main else []
        elif main and main not in reactants:
            reactants.insert(0, main)
        if not reactants and main:
            reactants = [main]
        rank = _rank(cand, fallback=idx)
        for reactant_index, connector in enumerate(reactants):
            connector = canonical_smiles(str(connector or "")) or str(connector or "")
            if not connector:
                continue
            heavy_atoms = _heavy_atoms(connector)
            if heavy_atoms < int(min_connector_heavy_atoms):
                continue
            key = (connector, rank)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "downstream_rank": rank,
                    "reactant_index": reactant_index,
                    "connector": connector,
                    "connector_heavy_atoms": heavy_atoms,
                    "is_main_reactant": bool(main and connector == main),
                    "reaction_smiles": rxn,
                    "candidate_score": _float(cand.get("score")),
                    "candidate_source": cand.get("source"),
                    "candidate_model": cand.get("model_full_name"),
                    "candidate_type": cand.get("type") or cand.get("proposal_type"),
                    "all_reactants": reactants,
                }
            )
    rows.sort(key=lambda row: (int(row.get("downstream_rank") or 10**9), int(row.get("reactant_index") or 0)))
    return rows


def _label_downstream_candidates(
    candidates: list[dict[str, Any]],
    refs_for_target: list[dict[str, Any]],
    *,
    analog_similarity: float,
) -> dict[str, Any]:
    best_similarity = 0.0
    best_rank = None
    analog_any = False
    for cand in candidates:
        downstream_fp = _transition_fp(refs_for_target[0].get("target_smiles") if refs_for_target else "", cand.get("connector"))
        best_for_candidate = max((_fp_similarity(downstream_fp, _transition_fp(ref.get("downstream_product"), ref.get("downstream_main_reactant"))) for ref in refs_for_target), default=0.0)
        if best_for_candidate > best_similarity:
            best_similarity = best_for_candidate
        if best_for_candidate >= analog_similarity:
            analog_any = True
            rank = int(cand.get("downstream_rank") or 10**9)
            best_rank = rank if best_rank is None else min(best_rank, rank)
    return {
        "analog_any": analog_any,
        "best_analog_rank": best_rank,
        "best_similarity": round(float(best_similarity), 6),
    }


def _build_bridge_rows(
    *,
    provider: CascadeRetrievalProvider,
    target: str,
    downstream_candidates: list[dict[str, Any]],
    refs_for_target: list[dict[str, Any]],
    modes: tuple[str, ...],
    provider_limit: int,
    min_similarity: float,
    bridge_similarity: float,
    analog_similarity: float,
    oracle_transform_filter: bool,
) -> list[dict[str, Any]]:
    rows = []
    query_cache: dict[tuple[str, str, str], list[Any]] = {}
    for cand in downstream_candidates:
        connector = str(cand.get("connector") or "")
        if not connector:
            continue
        for mode in modes:
            ref_filters = refs_for_target if oracle_transform_filter else [None]
            for ref_filter in ref_filters:
                required_transform = (ref_filter or {}).get("upstream_transform")
                required_downstream_transform = (ref_filter or {}).get("downstream_transform")
                key = (
                    mode,
                    target,
                    connector,
                    str(required_transform or ""),
                    str(required_downstream_transform or ""),
                )
                if key not in query_cache:
                    query_cache[key] = _retrieve_from_candidate(
                        provider,
                        target=target,
                        connector=connector,
                        mode=mode,
                        limit=provider_limit,
                        min_similarity=min_similarity,
                        required_transform=required_transform,
                        required_downstream_transform=required_downstream_transform,
                    )
                for provider_rank, hit in enumerate(query_cache[key], start=1):
                    bridge_sim = _smiles_similarity(connector, hit.product_smiles)
                    label_refs = [ref_filter] if ref_filter else refs_for_target
                    labels = _best_reference_label(
                        target=target,
                        connector=connector,
                        hit=hit,
                        refs_for_target=[ref for ref in label_refs if ref],
                        analog_similarity=analog_similarity,
                    )
                    nonoracle_score = _nonoracle_score(cand, hit, bridge_sim)
                    rows.append(
                        {
                            "nonoracle_rank": None,
                            "nonoracle_score": round(float(nonoracle_score), 6),
                            "mode": mode,
                            "oracle_transform_filter": bool(oracle_transform_filter),
                            "oracle_filter_block_id": (ref_filter or {}).get("block_id"),
                            "oracle_filter_transform_pair": (ref_filter or {}).get("transform_pair"),
                            "downstream_rank": cand.get("downstream_rank"),
                            "provider_rank": provider_rank,
                            "connector": connector,
                            "connector_heavy_atoms": cand.get("connector_heavy_atoms"),
                            "connector_is_main_reactant": cand.get("is_main_reactant"),
                            "bridge_similarity": round(float(bridge_sim), 6),
                            "bridge_hit": bool(bridge_sim >= bridge_similarity),
                            "provider_similarity": round(float(hit.similarity), 6),
                            "hit_id": hit.hit_id,
                            "hit_product": hit.product_smiles,
                            "hit_main_reactant": hit.main_reactant,
                            "hit_transform_pair": hit.transform_pair,
                            "hit_reaction_smiles": hit.rxn_smiles,
                            "downstream_reaction_smiles": cand.get("reaction_smiles"),
                            "candidate_source": cand.get("candidate_source"),
                            "candidate_model": cand.get("candidate_model"),
                            "upstream_similarity": labels.get("upstream_similarity"),
                            "downstream_similarity": labels.get("downstream_similarity"),
                            "best_reference_block_id": labels.get("reference_block_id"),
                            "reference_transform_pair": labels.get("reference_transform_pair"),
                            "pair_hit": labels.get("pair_hit"),
                            "analog_hit": labels.get("analog_hit"),
                            "pair_and_analog": labels.get("pair_and_analog"),
                        }
                    )
    rows.sort(key=_nonoracle_sort_key)
    for idx, row in enumerate(rows, start=1):
        row["nonoracle_rank"] = idx
    return rows


def _retrieve_from_candidate(
    provider: CascadeRetrievalProvider,
    *,
    target: str,
    connector: str,
    mode: str,
    limit: int,
    min_similarity: float,
    required_transform: Any = None,
    required_downstream_transform: Any = None,
):
    if mode == "block_downstream_transition":
        return provider.retrieve_for_transition(
            target,
            connector,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            required_transform=str(required_transform or ""),
            required_downstream_transform=str(required_downstream_transform or ""),
        )
    if mode == "block_downstream_product":
        return provider.retrieve_for_product(
            target,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            required_transform=str(required_transform or ""),
            required_downstream_transform=str(required_downstream_transform or ""),
        )
    if mode == "transition":
        return provider.retrieve_for_transition(
            target,
            connector,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            required_transform=str(required_transform or ""),
            required_downstream_transform=str(required_downstream_transform or ""),
        )
    if mode == "step_product":
        return provider.retrieve_for_product(
            target,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            required_transform=str(required_transform or ""),
            required_downstream_transform=str(required_downstream_transform or ""),
        )
    if mode == "upstream_step_product":
        return provider.retrieve_for_product(
            connector,
            mode="step_product",
            limit=limit,
            min_similarity=min_similarity,
            required_transform=str(required_transform or ""),
            required_downstream_transform=str(required_downstream_transform or ""),
        )
    if mode == "upstream_transition_product":
        return provider.retrieve_for_transition(
            connector,
            "",
            mode="transition",
            limit=limit,
            min_similarity=min_similarity,
            required_transform=str(required_transform or ""),
            required_downstream_transform=str(required_downstream_transform or ""),
        )
    raise ValueError(f"unsupported mode: {mode}")


def _best_reference_label(
    *,
    target: str,
    connector: str,
    hit: Any,
    refs_for_target: list[dict[str, Any]],
    analog_similarity: float,
) -> dict[str, Any]:
    upstream_fp = _transition_fp(hit.product_smiles, hit.main_reactant)
    downstream_fp = _transition_fp(target, connector)
    best = {
        "score": -1.0,
        "upstream_similarity": 0.0,
        "downstream_similarity": 0.0,
        "reference_block_id": None,
        "reference_transform_pair": None,
        "pair_hit": False,
        "analog_hit": False,
        "pair_and_analog": False,
    }
    for ref in refs_for_target:
        upstream_sim = _fp_similarity(upstream_fp, ref.get("upstream_fp"))
        ref_downstream_fp = _transition_fp(ref.get("downstream_product"), ref.get("downstream_main_reactant"))
        downstream_sim = _fp_similarity(downstream_fp, ref_downstream_fp)
        analog = bool(upstream_sim >= analog_similarity and downstream_sim >= analog_similarity)
        pair = str(hit.transform_pair or "").lower() == str(ref.get("transform_pair") or "").lower()
        score = float(upstream_sim + downstream_sim + (1.0 if pair else 0.0) + (1.0 if analog else 0.0))
        if score > float(best["score"]):
            best = {
                "score": score,
                "upstream_similarity": round(float(upstream_sim), 6),
                "downstream_similarity": round(float(downstream_sim), 6),
                "reference_block_id": ref.get("block_id"),
                "reference_transform_pair": ref.get("transform_pair"),
                "pair_hit": bool(pair),
                "analog_hit": bool(analog),
                "pair_and_analog": bool(pair and analog),
            }
    best.pop("score", None)
    return best


def _nonoracle_score(cand: dict[str, Any], hit: Any, bridge_sim: float) -> float:
    rank = max(1, int(cand.get("downstream_rank") or 10**6))
    rank_score = 1.0 / math.sqrt(rank)
    chem_score = _float(cand.get("candidate_score"))
    if chem_score is None:
        chem_score = 0.0
    main_bonus = 0.05 if cand.get("is_main_reactant") else 0.0
    heavy_atoms = max(0, int(cand.get("connector_heavy_atoms") or 0))
    size_bonus = min(0.08, 0.01 * heavy_atoms)
    return 0.37 * float(hit.similarity) + 0.37 * float(bridge_sim) + 0.10 * rank_score + 0.06 * float(chem_score) + main_bonus + size_bonus


def _nonoracle_sort_key(row: dict[str, Any]) -> tuple[float, float, float, int, int]:
    return (
        -float(row.get("nonoracle_score") or 0.0),
        -float(row.get("bridge_similarity") or 0.0),
        -float(row.get("provider_similarity") or 0.0),
        int(row.get("downstream_rank") or 10**9),
        int(row.get("provider_rank") or 10**9),
    )


def _best_rank(rows: list[dict[str, Any]], label_key: str) -> int | None:
    ranks = [
        int(row.get("nonoracle_rank") or 10**9)
        for row in rows
        if row.get(label_key) and row.get("bridge_hit")
    ]
    return min(ranks) if ranks else None


def _topk_hits(rows: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, dict[str, bool]]:
    out = {}
    for k in top_ks:
        top = rows[: max(0, int(k))]
        out[str(k)] = {
            "bridge": any(row.get("bridge_hit") for row in top),
            "analog_bridge": any(row.get("bridge_hit") and row.get("analog_hit") for row in top),
            "pair_and_analog_bridge": any(row.get("bridge_hit") and row.get("pair_and_analog") for row in top),
        }
    return out


def _summary(rows: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, Any]:
    denom = max(len(rows), 1)
    out: dict[str, Any] = {
        "targets": len(rows),
        "reference_blocks": sum(int(row.get("reference_blocks") or 0) for row in rows),
        "chem_enzy_candidate_rows": sum(int(row.get("chem_enzy_candidate_rows") or 0) for row in rows),
        "chem_enzy_downstream_connectors": sum(int(row.get("chem_enzy_downstream_connectors") or 0) for row in rows),
        "bridge_rows": sum(int(row.get("bridge_rows") or 0) for row in rows),
        "targets_downstream_analog_any": sum(1 for row in rows if row.get("downstream_analog_any")),
        "targets_bridge_any": sum(1 for row in rows if row.get("bridge_any")),
        "targets_analog_bridge_any": sum(1 for row in rows if row.get("analog_bridge_any")),
        "targets_pair_and_analog_bridge_any": sum(1 for row in rows if row.get("pair_and_analog_bridge_any")),
        "downstream_analog_any_rate": round(sum(1 for row in rows if row.get("downstream_analog_any")) / denom, 6),
        "bridge_any_rate": round(sum(1 for row in rows if row.get("bridge_any")) / denom, 6),
        "analog_bridge_any_rate": round(sum(1 for row in rows if row.get("analog_bridge_any")) / denom, 6),
        "pair_and_analog_bridge_any_rate": round(sum(1 for row in rows if row.get("pair_and_analog_bridge_any")) / denom, 6),
    }
    for k in top_ks:
        key = str(k)
        out[f"bridge_at_{k}"] = round(sum(1 for row in rows if (row.get("topk") or {}).get(key, {}).get("bridge")) / denom, 6)
        out[f"analog_bridge_at_{k}"] = round(sum(1 for row in rows if (row.get("topk") or {}).get(key, {}).get("analog_bridge")) / denom, 6)
        out[f"pair_and_analog_bridge_at_{k}"] = round(
            sum(1 for row in rows if (row.get("topk") or {}).get(key, {}).get("pair_and_analog_bridge")) / denom,
            6,
        )
    return out


def _reference_summary(refs_by_target: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "targets_with_blocks": len(refs_by_target),
        "reference_blocks": sum(len(rows) for rows in refs_by_target.values()),
        "top_transform_pairs": dict(Counter(ref.get("transform_pair") for rows in refs_by_target.values() for ref in rows).most_common(20)),
    }


def _reference_connection_similarity(ref: dict[str, Any]) -> float:
    return _smiles_similarity(ref.get("upstream_product"), ref.get("downstream_main_reactant"))


def _reference_examples(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ref in refs[:5]:
        out.append(
            {
                "block_id": ref.get("block_id"),
                "transform_pair": ref.get("transform_pair"),
                "upstream_product": ref.get("upstream_product"),
                "downstream_product": ref.get("downstream_product"),
                "downstream_main_reactant": ref.get("downstream_main_reactant"),
            }
        )
    return out


def _rank(cand: dict[str, Any], *, fallback: int) -> int:
    try:
        return int(cand.get("rank") or cand.get("candidate_rank") or fallback)
    except (TypeError, ValueError):
        return fallback


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _smiles_similarity(left: Any, right: Any) -> float:
    left_fp = _fp(left)
    right_fp = _fp(right)
    if left_fp is None or right_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _fp(smiles: Any):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _heavy_atoms(smiles: Any) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return int(mol.GetNumHeavyAtoms())


def _fp_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    return modes or ("block_downstream_transition",)


def _parse_top_ks(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item.strip()) for item in str(value or "").split(",") if item.strip()})) or (1, 3, 5, 10, 20, 50)


def _markdown(report: dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    summary = report.get("summary") or {}
    lines = [
        "# Strict Non-Oracle Provider Bridge Audit",
        "",
        "## Contract",
        "",
        str(meta.get("contract") or ""),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `downstream_analog_any` means ChemEnzy top-N contains a target-to-connector step analogous to a held-out downstream cascade step.",
            "- `bridge` additionally requires a train-backed provider hit whose product is similar to the ChemEnzy connector.",
            "- `pair_and_analog_bridge` additionally requires the provider transform pair to match the held-out block transform pair.",
            "",
            "## Examples",
            "",
            "```json",
            json.dumps((report.get("targets") or [])[:8], indent=2, ensure_ascii=False)[:12000],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description="Strict non-oracle provider-to-ChemEnzy bridge audit")
    parser.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    parser.add_argument("--chem-enzy-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    parser.add_argument("--route-recovery-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--all-reference-targets", action="store_true")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-chem-candidates", type=int, default=100)
    parser.add_argument("--main-reactant-only", action="store_true")
    parser.add_argument("--min-connector-heavy-atoms", type=int, default=0)
    parser.add_argument("--provider-limit", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.20)
    parser.add_argument("--bridge-similarity", type=float, default=0.70)
    parser.add_argument("--analog-similarity", type=float, default=0.55)
    parser.add_argument("--connected-ref-similarity", type=float, default=0.0)
    parser.add_argument("--modes", default="block_downstream_transition")
    parser.add_argument("--oracle-transform-filter", action="store_true")
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    args = parser.parse_args()
    report = audit_nonoracle_provider_bridge(
        program_manifest=Path(args.program_manifest),
        chem_enzy_cache=Path(args.chem_enzy_cache),
        route_recovery_json=Path(args.route_recovery_json) if args.route_recovery_json else None,
        output_json=Path(args.output_json),
        split=args.split,
        only_routed=not args.all_reference_targets,
        max_targets=args.max_targets,
        max_chem_candidates=args.max_chem_candidates,
        main_reactant_only=args.main_reactant_only,
        min_connector_heavy_atoms=args.min_connector_heavy_atoms,
        provider_limit=args.provider_limit,
        min_similarity=args.min_similarity,
        bridge_similarity=args.bridge_similarity,
        analog_similarity=args.analog_similarity,
        connected_ref_similarity=args.connected_ref_similarity,
        modes=_parse_modes(args.modes),
        oracle_transform_filter=args.oracle_transform_filter,
        top_ks=_parse_top_ks(args.top_ks),
    )
    print(json.dumps({"summary": report["summary"], "output_json": args.output_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
