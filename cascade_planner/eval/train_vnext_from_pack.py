"""Train vNext StepEncoder, candidate-pool ranker, and route-state models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.eval.build_vnext_pack import build_vnext_pack
from cascade_planner.vnext.features import (
    candidate_feature_dim,
    candidate_feature_vector,
    candidate_reactant_smiles,
    morgan_fp,
    node_feature_dim,
    open_leaf_feature_matrix,
    read_jsonl,
    route_feature_dim,
    route_feature_vector,
    route_label_vector,
    route_step_tokens,
    safe_float,
    source_budget_vector,
    stable_bucket,
    step_token_dim,
)
from cascade_planner.vnext.models import (
    CandidatePoolCrossAttentionRanker,
    RouteStateTransformer,
    SearchPolicyNetwork,
    StepEncoder,
)
from cascade_planner.vnext.schema import SOURCE_BUDGET_GROUPS, VNEXT_FILES, VNEXT_SCHEMA_VERSION


@dataclass
class StepPairDataset:
    rows: list[dict[str, Any]]
    product_fp: np.ndarray
    reactant_fp: np.ndarray
    metadata: np.ndarray
    labels: np.ndarray
    reaction_type: np.ndarray
    ec1: np.ndarray
    condition: np.ndarray
    weights: np.ndarray
    feature_schema: dict[str, Any]


@dataclass
class CandidatePoolDataset:
    rows: list[dict[str, Any]]
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    feature_schema: dict[str, Any]


@dataclass
class RouteStateDataset:
    rows: list[dict[str, Any]]
    step_tokens: np.ndarray
    step_mask: np.ndarray
    route_features: np.ndarray
    value: np.ndarray
    solved: np.ndarray
    stock_closed: np.ndarray
    progressive: np.ndarray
    compatibility: np.ndarray
    bottlenecks: np.ndarray
    feature_schema: dict[str, Any]


@dataclass
class SearchPolicyDataset:
    rows: list[dict[str, Any]]
    step_tokens: np.ndarray
    step_mask: np.ndarray
    route_features: np.ndarray
    node_features: np.ndarray
    node_mask: np.ndarray
    node_labels: np.ndarray
    node_weights: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    action_labels: np.ndarray
    action_weights: np.ndarray
    value: np.ndarray
    solved: np.ndarray
    stock_closed: np.ndarray
    progressive: np.ndarray
    compatibility: np.ndarray
    bottlenecks: np.ndarray
    source_budget: np.ndarray
    feature_schema: dict[str, Any]


def prepare_vnext_pack(pack_dir: Path | None, vnext_pack_dir: Path | None, *, max_candidates: int = 32) -> Path:
    if vnext_pack_dir and (vnext_pack_dir / VNEXT_FILES["candidate_pools"]).exists():
        return vnext_pack_dir
    if pack_dir is None:
        raise ValueError("either pack_dir or vnext_pack_dir is required")
    if vnext_pack_dir is None:
        vnext_pack_dir = Path(tempfile.mkdtemp(prefix="autoplanner_vnext_pack_"))
    build_vnext_pack(pack_dir=pack_dir, output_dir=vnext_pack_dir, max_candidates=max_candidates)
    return vnext_pack_dir


def build_step_pair_dataset(vnext_pack_dir: Path, *, n_bits: int = 256) -> StepPairDataset:
    rows = read_jsonl(vnext_pack_dir / VNEXT_FILES["step_pairs"])
    if not rows:
        raise ValueError(f"no step pairs in {vnext_pack_dir}")
    fp_cache: dict[str, np.ndarray] = {}
    product_fp = []
    reactant_fp = []
    metadata = []
    labels = []
    reaction_type = []
    ec1 = []
    condition = []
    weights = []
    for row in rows:
        product_fp.append(_cached_morgan_fp(fp_cache, row.get("product"), n_bits=n_bits))
        reactant_fp.append(_cached_morgan_fp(fp_cache, _step_reactant_smiles(row), n_bits=n_bits))
        metadata.append(_step_metadata(row))
        labels.append(float(row.get("label") or 0.0))
        reaction_type.append(stable_bucket(str(row.get("reaction_type") or ""), 32))
        ec1.append(_ec1_id(row.get("ec")))
        condition.append([
            float(safe_float((row.get("candidate") or {}).get("T")) or 0.0) / 100.0,
            float(safe_float((row.get("candidate") or {}).get("pH")) or 0.0) / 14.0,
        ])
        weights.append(float(row.get("weight") or 1.0))
    schema = {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "n_bits": n_bits,
        "metadata_dim": len(metadata[0]),
        "model_kind": "step_encoder",
    }
    return StepPairDataset(
        rows=rows,
        product_fp=np.asarray(product_fp, dtype=np.float32),
        reactant_fp=np.asarray(reactant_fp, dtype=np.float32),
        metadata=np.asarray(metadata, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        reaction_type=np.asarray(reaction_type, dtype=np.int64),
        ec1=np.asarray(ec1, dtype=np.int64),
        condition=np.asarray(condition, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        feature_schema=schema,
    )


def _cached_morgan_fp(cache: dict[str, np.ndarray], smiles: Any, *, n_bits: int) -> np.ndarray:
    key = f"{n_bits}:{smiles or ''}"
    fp = cache.get(key)
    if fp is None:
        fp = morgan_fp(smiles, n_bits=n_bits)
        cache[key] = fp
    return fp


def _step_reactant_smiles(row: dict[str, Any]) -> str:
    reactants = row.get("reactants") or []
    if reactants:
        return ".".join(str(smi) for smi in reactants if smi)
    return candidate_reactant_smiles(row.get("candidate") or {})


def _cached_candidate_feature_vector(
    cache: dict[str, np.ndarray],
    product: str,
    candidate: dict[str, Any],
    *,
    rank: Any,
    gt_available: bool,
    n_bits: int,
) -> np.ndarray:
    key = "|".join([
        str(n_bits),
        str(product or ""),
        str(candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or ""),
        str(candidate.get("main_reactant") or ""),
        ".".join(str(smi) for smi in candidate.get("aux_reactants") or []),
        str(rank or ""),
        "1" if gt_available else "0",
    ])
    vec = cache.get(key)
    if vec is None:
        vec = candidate_feature_vector(
            product or "",
            candidate,
            rank=rank,
            gt_available=gt_available,
            n_bits=n_bits,
        )
        cache[key] = vec
    return vec


def _build_candidate_pool_arrays(
    rows: list[dict[str, Any]],
    *,
    n_bits: int,
    max_candidates: int,
    feat_dim: int,
    feature_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), max_candidates, feat_dim), dtype=np.float32)
    mask = np.zeros((len(rows), max_candidates), dtype=np.float32)
    labels = np.zeros((len(rows), max_candidates), dtype=np.float32)
    weights = np.ones((len(rows), max_candidates), dtype=np.float32)
    if int(feature_workers or 1) > 1 and len(rows) > 1:
        tasks = ((row, max_candidates, n_bits, feat_dim) for row in rows)
        with ProcessPoolExecutor(max_workers=int(feature_workers)) as pool:
            for i, (row_x, row_mask, row_labels, row_weights) in enumerate(pool.map(_candidate_pool_arrays_for_row, tasks, chunksize=64)):
                x[i] = row_x
                mask[i] = row_mask
                labels[i] = row_labels
                weights[i] = row_weights
        return x, mask, labels, weights

    feature_cache: dict[str, np.ndarray] = {}
    for i, row in enumerate(rows):
        for j, item in enumerate((row.get("candidates") or [])[:max_candidates]):
            candidate = item.get("candidate") or {}
            x[i, j, :] = _cached_candidate_feature_vector(
                feature_cache,
                row.get("product") or "",
                candidate,
                rank=item.get("rank"),
                gt_available=bool(item.get("gt_available")),
                n_bits=n_bits,
            )
            mask[i, j] = 1.0
            labels[i, j] = float(item.get("label") or 0.0)
            weights[i, j] = float(item.get("weight") or 1.0)
    return x, mask, labels, weights


def _candidate_pool_arrays_for_row(args: tuple[dict[str, Any], int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row, max_candidates, n_bits, feat_dim = args
    row_x = np.zeros((max_candidates, feat_dim), dtype=np.float32)
    row_mask = np.zeros(max_candidates, dtype=np.float32)
    row_labels = np.zeros(max_candidates, dtype=np.float32)
    row_weights = np.ones(max_candidates, dtype=np.float32)
    for j, item in enumerate((row.get("candidates") or [])[:max_candidates]):
        candidate = item.get("candidate") or {}
        row_x[j, :] = candidate_feature_vector(
            row.get("product") or "",
            candidate,
            rank=item.get("rank"),
            gt_available=bool(item.get("gt_available")),
            n_bits=n_bits,
        )
        row_mask[j] = 1.0
        row_labels[j] = float(item.get("label") or 0.0)
        row_weights[j] = float(item.get("weight") or 1.0)
    return row_x, row_mask, row_labels, row_weights


def _dataset_cache_path(
    cache_dir: Path | None,
    kind: str,
    paths: list[Path],
    params: dict[str, Any],
) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir = Path(cache_dir)
    inputs = []
    for path in paths:
        stat = path.stat()
        inputs.append({
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    payload = {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "kind": kind,
        "inputs": inputs,
        "params": params,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{kind}_{digest}.npz"


def _load_npz_arrays(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in keys if key in data}


def _save_npz_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)


def build_candidate_pool_dataset(
    vnext_pack_dir: Path,
    *,
    n_bits: int = 256,
    max_candidates: int = 32,
    feature_cache_dir: Path | None = None,
    feature_workers: int = 1,
) -> CandidatePoolDataset:
    pool_path = vnext_pack_dir / VNEXT_FILES["candidate_pools"]
    rows = read_jsonl(pool_path)
    if not rows:
        raise ValueError(f"no candidate pools in {vnext_pack_dir}")
    feat_dim = candidate_feature_dim(n_bits)
    schema = {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "n_bits": n_bits,
        "candidate_feature_dim": feat_dim,
        "max_candidates": max_candidates,
        "model_kind": "candidate_pool_ranker",
    }
    cache_path = _dataset_cache_path(
        feature_cache_dir,
        "candidate_pool",
        [pool_path],
        {"n_bits": n_bits, "max_candidates": max_candidates, "feature_dim": feat_dim},
    )
    if cache_path and cache_path.exists():
        cached = _load_npz_arrays(cache_path, ["candidate_features", "candidate_mask", "labels", "weights"])
        return CandidatePoolDataset(
            rows=rows,
            candidate_features=cached["candidate_features"],
            candidate_mask=cached["candidate_mask"],
            labels=cached["labels"],
            weights=cached["weights"],
            feature_schema={**schema, "feature_cache": str(cache_path), "feature_cache_hit": True},
        )
    x, mask, labels, weights = _build_candidate_pool_arrays(
        rows,
        n_bits=n_bits,
        max_candidates=max_candidates,
        feat_dim=feat_dim,
        feature_workers=feature_workers,
    )
    if cache_path:
        _save_npz_arrays(
            cache_path,
            {
                "candidate_features": x,
                "candidate_mask": mask,
                "labels": labels,
                "weights": weights,
            },
        )
        schema = {**schema, "feature_cache": str(cache_path), "feature_cache_hit": False}
    return CandidatePoolDataset(rows=rows, candidate_features=x, candidate_mask=mask, labels=labels, weights=weights, feature_schema=schema)


def build_route_state_dataset(vnext_pack_dir: Path, *, max_steps: int = 8) -> RouteStateDataset:
    rows = read_jsonl(vnext_pack_dir / VNEXT_FILES["route_states"])
    if not rows:
        raise ValueError(f"no route states in {vnext_pack_dir}")
    step_x = []
    step_m = []
    route_x = []
    value = []
    solved = []
    stock = []
    progressive = []
    compatibility = []
    bottlenecks = []
    for row in rows:
        tokens, mask = route_step_tokens(row, max_steps=max_steps)
        labels = route_label_vector(row)
        step_x.append(tokens)
        step_m.append(mask)
        route_x.append(route_feature_vector(row))
        value.append(float(labels["value"]))
        solved.append(float(labels["solved"]))
        stock.append(float(labels["stock_closed"]))
        progressive.append(float(labels["progressive"]))
        compatibility.append(float(labels["compatibility"]))
        bottlenecks.append(labels["bottlenecks"])
    schema = {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "step_token_dim": step_token_dim(),
        "route_feature_dim": route_feature_dim(),
        "max_steps": max_steps,
        "model_kind": "route_state_transformer",
    }
    return RouteStateDataset(
        rows=rows,
        step_tokens=np.asarray(step_x, dtype=np.float32),
        step_mask=np.asarray(step_m, dtype=np.float32),
        route_features=np.asarray(route_x, dtype=np.float32),
        value=np.asarray(value, dtype=np.float32),
        solved=np.asarray(solved, dtype=np.float32),
        stock_closed=np.asarray(stock, dtype=np.float32),
        progressive=np.asarray(progressive, dtype=np.float32),
        compatibility=np.asarray(compatibility, dtype=np.float32),
        bottlenecks=np.asarray(bottlenecks, dtype=np.float32),
        feature_schema=schema,
    )


def build_search_policy_dataset(
    vnext_pack_dir: Path,
    *,
    n_bits: int = 256,
    max_candidates: int = 32,
    max_steps: int = 8,
    max_open_leaves: int | None = None,
    feature_cache_dir: Path | None = None,
    feature_workers: int = 1,
) -> SearchPolicyDataset:
    max_open_leaves = int(max_open_leaves or max_steps)
    pool_path = vnext_pack_dir / VNEXT_FILES["candidate_pools"]
    route_path = vnext_pack_dir / VNEXT_FILES["route_states"]
    pools = read_jsonl(pool_path)
    route_rows = read_jsonl(route_path)
    routes_by_id = {row.get("route_id"): row for row in route_rows if row.get("route_id")}
    feat_dim = candidate_feature_dim(n_bits)
    schema = {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "n_bits": n_bits,
        "action_feature_dim": feat_dim,
        "node_feature_dim": node_feature_dim(n_bits),
        "step_token_dim": step_token_dim(),
        "route_feature_dim": route_feature_dim(),
        "max_candidates": max_candidates,
        "max_steps": max_steps,
        "max_open_leaves": max_open_leaves,
        "source_budget_groups": list(SOURCE_BUDGET_GROUPS),
        "model_kind": "search_policy",
    }
    cache_keys = [
        "step_tokens",
        "step_mask",
        "route_features",
        "node_features",
        "node_mask",
        "node_labels",
        "node_weights",
        "action_features",
        "action_mask",
        "action_labels",
        "action_weights",
        "value",
        "solved",
        "stock_closed",
        "progressive",
        "compatibility",
        "bottlenecks",
        "source_budget",
    ]
    cache_path = _dataset_cache_path(
        feature_cache_dir,
        "search_policy",
        [pool_path, route_path],
        {
            "n_bits": n_bits,
            "max_candidates": max_candidates,
            "max_steps": max_steps,
            "max_open_leaves": max_open_leaves,
            "feature_dim": feat_dim,
            "supervision": "policy_value_bottleneck_v3_node_budget_supervised_value",
        },
    )
    if cache_path and cache_path.exists():
        cached = _load_npz_arrays(cache_path, cache_keys)
        return SearchPolicyDataset(
            rows=pools,
            step_tokens=cached["step_tokens"],
            step_mask=cached["step_mask"],
            route_features=cached["route_features"],
            node_features=cached["node_features"],
            node_mask=cached["node_mask"],
            node_labels=cached["node_labels"],
            node_weights=cached.get("node_weights", np.ones_like(cached["node_labels"], dtype=np.float32)),
            action_features=cached["action_features"],
            action_mask=cached["action_mask"],
            action_labels=cached["action_labels"],
            action_weights=cached.get("action_weights", np.ones_like(cached["action_labels"], dtype=np.float32)),
            value=cached["value"],
            solved=cached["solved"],
            stock_closed=cached["stock_closed"],
            progressive=cached["progressive"],
            compatibility=cached["compatibility"],
            bottlenecks=cached["bottlenecks"],
            source_budget=cached["source_budget"],
            feature_schema={**schema, "feature_cache": str(cache_path), "feature_cache_hit": True},
        )
    step_x = np.zeros((len(pools), max_steps, step_token_dim()), dtype=np.float32)
    step_m = np.zeros((len(pools), max_steps), dtype=np.float32)
    route_x = np.zeros((len(pools), route_feature_dim()), dtype=np.float32)
    node_x = np.zeros((len(pools), max_open_leaves, node_feature_dim(n_bits)), dtype=np.float32)
    node_m = np.zeros((len(pools), max_open_leaves), dtype=np.float32)
    node_y = np.zeros((len(pools), max_open_leaves), dtype=np.float32)
    node_w = np.ones((len(pools), max_open_leaves), dtype=np.float32)
    value = np.zeros(len(pools), dtype=np.float32)
    solved = np.zeros(len(pools), dtype=np.float32)
    stock = np.zeros(len(pools), dtype=np.float32)
    progressive = np.zeros(len(pools), dtype=np.float32)
    compatibility = np.zeros(len(pools), dtype=np.float32)
    bottlenecks = np.zeros((len(pools), len(route_label_vector({})["bottlenecks"])), dtype=np.float32)
    source_budget = np.zeros((len(pools), len(SOURCE_BUDGET_GROUPS)), dtype=np.float32)
    for i, pool in enumerate(pools):
        route = routes_by_id.get(pool.get("route_id")) or _route_row_from_candidate_pool(pool)
        tokens, mask = route_step_tokens(route, max_steps=max_steps)
        step_x[i] = tokens
        step_m[i] = mask
        route_x[i] = route_feature_vector(route)
        labels = route_label_vector(route)
        value[i] = float(labels["value"])
        solved[i] = float(labels["solved"])
        stock[i] = float(labels["stock_closed"])
        progressive[i] = float(labels["progressive"])
        compatibility[i] = float(labels["compatibility"])
        bottlenecks[i] = labels["bottlenecks"]
        product = str(pool.get("product") or pool.get("expanded_leaf") or pool.get("target_smiles") or "")
        node_x[i], node_m[i] = open_leaf_feature_matrix(
            target=str(pool.get("target_smiles") or product),
            open_leaves=[product] if product else [],
            depth=int(pool.get("step_index") or 0),
            max_open_leaves=max_open_leaves,
            n_bits=n_bits,
        )
        if node_m[i, 0] > 0:
            node_y[i, 0] = 1.0
        source_budget[i] = source_budget_vector(pool.get("candidates") or [])
    action_x, action_m, action_y, action_w = _build_candidate_pool_arrays(
        pools,
        n_bits=n_bits,
        max_candidates=max_candidates,
        feat_dim=feat_dim,
        feature_workers=feature_workers,
    )
    if cache_path:
        _save_npz_arrays(
            cache_path,
            {
                "step_tokens": step_x,
                "step_mask": step_m,
                "route_features": route_x,
                "node_features": node_x,
                "node_mask": node_m,
                "node_labels": node_y,
                "node_weights": node_w,
                "action_features": action_x,
                "action_mask": action_m,
                "action_labels": action_y,
                "action_weights": action_w,
                "value": value,
                "solved": solved,
                "stock_closed": stock,
                "progressive": progressive,
                "compatibility": compatibility,
                "bottlenecks": bottlenecks,
                "source_budget": source_budget,
            },
        )
        schema = {**schema, "feature_cache": str(cache_path), "feature_cache_hit": False}
    return SearchPolicyDataset(
        rows=pools,
        step_tokens=step_x,
        step_mask=step_m,
        route_features=route_x,
        node_features=node_x,
        node_mask=node_m,
        node_labels=node_y,
        node_weights=node_w,
        action_features=action_x,
        action_mask=action_m,
        action_labels=action_y,
        action_weights=action_w,
        value=value,
        solved=solved,
        stock_closed=stock,
        progressive=progressive,
        compatibility=compatibility,
        bottlenecks=bottlenecks,
        source_budget=source_budget,
        feature_schema=schema,
    )


def build_route_tree_search_policy_dataset(
    trace_paths: list[Path] | tuple[Path, ...] | Path,
    *,
    n_bits: int = 256,
    max_candidates: int = 32,
    max_steps: int = 8,
    max_open_leaves: int | None = None,
    node_label_target: str = "trace_leaf_utility",
    action_label_target: str = "selected_solved_action",
    feature_cache_dir: Path | None = None,
    feature_workers: int = 1,
) -> SearchPolicyDataset:
    max_open_leaves = int(max_open_leaves or max_steps)
    paths = _as_paths(trace_paths)
    raw_rows: list[dict[str, Any]] = []
    for path in paths:
        raw_rows.extend(read_jsonl(path))
    leaf_utility = (
        _trace_stock_aware_leaf_utility_index(raw_rows)
        if node_label_target == "stock_aware_leaf_utility"
        else _trace_leaf_utility_index(raw_rows)
    )
    action_label_target = str(action_label_target or "selected_solved_action")
    pools: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        event = row.get("event") if "event" in row else row
        if not isinstance(event, dict):
            continue
        actions = event.get("candidate_actions") or []
        if not actions:
            continue
        product = str(event.get("expanded_leaf") or row.get("target_smiles") or "")
        target = str(row.get("target_smiles") or product)
        if not product:
            continue
        route_id = f"rt_{idx}_{event.get('state_id') or ''}"
        route = _route_row_from_trace(row, event, route_id=route_id, target=target)
        solved = bool(route.get("label", 0.0) >= 1.0)
        selected_key = str(event.get("selected_action_key") or "")
        candidate_items = []
        for rank, action_dict in enumerate(actions[:max_candidates], start=1):
            candidate = _trace_action_candidate(action_dict)
            key = CandidateAction.from_candidate(product, candidate, rank=rank).canonical_key
            label, weight, label_type = _trace_action_label_and_weight(
                row,
                event,
                action_dict,
                candidate,
                key=key,
                selected_key=selected_key,
                solved=solved,
                action_label_target=action_label_target,
            )
            candidate_items.append({
                "candidate_id": f"{route_id}_{rank}",
                "rank": rank,
                "label": label,
                "label_type": label_type,
                "weight": weight,
                "gt_available": solved or label >= 0.5,
                "selected_exact": bool(selected_key and key == selected_key),
                "candidate": candidate,
            })
        if not candidate_items:
            continue
        pools.append({
            "route_id": route_id,
            "target_smiles": target,
            "benchmark_index": row.get("benchmark_index"),
            "product": product,
            "expanded_leaf": product,
            "state_id": event.get("state_id"),
            "candidates": candidate_items,
            "_trace_event": event,
            "_trace_row": row,
            "_leaf_utility": leaf_utility,
            "_node_label_target": node_label_target,
        })
        routes.append(route)
    if not pools:
        raise ValueError(f"no route-tree expansion rows with candidate actions in {', '.join(str(path) for path in paths)}")

    routes_by_id = {row.get("route_id"): row for row in routes}
    feat_dim = candidate_feature_dim(n_bits)
    schema = {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "n_bits": n_bits,
        "action_feature_dim": feat_dim,
        "node_feature_dim": node_feature_dim(n_bits),
        "step_token_dim": step_token_dim(),
        "route_feature_dim": route_feature_dim(),
        "max_candidates": max_candidates,
        "max_steps": max_steps,
        "max_open_leaves": max_open_leaves,
        "model_kind": "search_policy",
        "training_source": "route_tree_traces",
        "trace_paths": [str(path) for path in paths],
        "source_budget_groups": list(SOURCE_BUDGET_GROUPS),
        "node_label_target": node_label_target,
        "action_label_target": action_label_target,
        "node_label_notes": [
            "Expanded leaves receive utility from final route outcome, stock closure, proposal yield, GT product proximity, and selected action availability.",
            "Non-expanded open leaves receive replay utility if the same target expanded them elsewhere in the same trace set.",
            "This is not pure imitation of the previously selected open leaf.",
        ],
        "node_label_mode": (
            "stock-aware" if node_label_target == "stock_aware_leaf_utility" else "trace-utility"
        ),
        "action_label_notes": _action_label_notes(action_label_target),
    }
    cache_keys = [
        "step_tokens",
        "step_mask",
        "route_features",
        "node_features",
        "node_mask",
        "node_labels",
        "node_weights",
        "action_features",
        "action_mask",
        "action_labels",
        "action_weights",
        "value",
        "solved",
        "stock_closed",
        "progressive",
        "compatibility",
        "bottlenecks",
        "source_budget",
    ]
    cache_path = _dataset_cache_path(
        feature_cache_dir,
        "route_tree_search_policy",
        paths,
        {
            "n_bits": n_bits,
            "max_candidates": max_candidates,
            "max_steps": max_steps,
            "max_open_leaves": max_open_leaves,
            "feature_dim": feat_dim,
            "supervision": "route_tree_policy_value_bottleneck_v4_leaf_utility_budget_supervised_value",
            "node_label_target": node_label_target,
            "action_label_target": action_label_target,
        },
    )
    if cache_path and cache_path.exists():
        cached = _load_npz_arrays(cache_path, cache_keys)
        return SearchPolicyDataset(
            rows=pools,
            step_tokens=cached["step_tokens"],
            step_mask=cached["step_mask"],
            route_features=cached["route_features"],
            node_features=cached["node_features"],
            node_mask=cached["node_mask"],
            node_labels=cached["node_labels"],
            node_weights=cached.get("node_weights", np.ones_like(cached["node_labels"], dtype=np.float32)),
            action_features=cached["action_features"],
            action_mask=cached["action_mask"],
            action_labels=cached["action_labels"],
            action_weights=cached.get("action_weights", np.ones_like(cached["action_labels"], dtype=np.float32)),
            value=cached["value"],
            solved=cached["solved"],
            stock_closed=cached["stock_closed"],
            progressive=cached["progressive"],
            compatibility=cached["compatibility"],
            bottlenecks=cached["bottlenecks"],
            source_budget=cached["source_budget"],
            feature_schema={**schema, "feature_cache": str(cache_path), "feature_cache_hit": True},
        )

    step_x = np.zeros((len(pools), max_steps, step_token_dim()), dtype=np.float32)
    step_m = np.zeros((len(pools), max_steps), dtype=np.float32)
    route_x = np.zeros((len(pools), route_feature_dim()), dtype=np.float32)
    node_x = np.zeros((len(pools), max_open_leaves, node_feature_dim(n_bits)), dtype=np.float32)
    node_m = np.zeros((len(pools), max_open_leaves), dtype=np.float32)
    node_y = np.zeros((len(pools), max_open_leaves), dtype=np.float32)
    node_w = np.zeros((len(pools), max_open_leaves), dtype=np.float32)
    value_y = np.zeros(len(pools), dtype=np.float32)
    solved_y = np.zeros(len(pools), dtype=np.float32)
    stock_y = np.zeros(len(pools), dtype=np.float32)
    progressive_y = np.zeros(len(pools), dtype=np.float32)
    compatibility_y = np.zeros(len(pools), dtype=np.float32)
    bottleneck_y = np.zeros((len(pools), len(route_label_vector({})["bottlenecks"])), dtype=np.float32)
    source_budget_y = np.zeros((len(pools), len(SOURCE_BUDGET_GROUPS)), dtype=np.float32)
    for i, pool in enumerate(pools):
        route = routes_by_id[pool["route_id"]]
        tokens, mask = route_step_tokens(route, max_steps=max_steps)
        labels = route_label_vector(route)
        step_x[i] = tokens
        step_m[i] = mask
        route_x[i] = route_feature_vector(route)
        value_y[i] = float(labels["value"])
        solved_y[i] = float(labels["solved"])
        stock_y[i] = float(labels["stock_closed"])
        progressive_y[i] = float(labels["progressive"])
        compatibility_y[i] = float(labels["compatibility"])
        bottleneck_y[i] = labels["bottlenecks"]
        event = pool.get("_trace_event") or {}
        open_leaves = [str(smi) for smi in (event.get("open_leaves") or [pool.get("expanded_leaf")]) if smi]
        state_obj = event.get("state") or {}
        node_x[i], node_m[i] = open_leaf_feature_matrix(
            target=str(pool.get("target_smiles") or ""),
            open_leaves=open_leaves,
            depth=int(event.get("depth") or 0),
            expanded=state_obj.get("expanded") or [],
            parent_reactants=_trace_parent_reactants(state_obj),
            max_open_leaves=max_open_leaves,
            n_bits=n_bits,
        )
        utility_by_leaf = _node_utility_labels_for_event(pool, event, open_leaves)
        for node_idx, leaf in enumerate(open_leaves[:max_open_leaves]):
            node_y[i, node_idx] = float(utility_by_leaf.get(_canonical_leaf(leaf), 0.0))
        if len(open_leaves) > 1:
            node_w[i] = node_m[i]
        source_budget_y[i] = source_budget_vector(pool.get("candidates") or [])
    action_x, action_m, action_y, action_w = _build_candidate_pool_arrays(
        pools,
        n_bits=n_bits,
        max_candidates=max_candidates,
        feat_dim=feat_dim,
        feature_workers=feature_workers,
    )
    if cache_path:
        _save_npz_arrays(
            cache_path,
            {
                "step_tokens": step_x,
                "step_mask": step_m,
                "route_features": route_x,
                "node_features": node_x,
                "node_mask": node_m,
                "node_labels": node_y,
                "node_weights": node_w,
                "action_features": action_x,
                "action_mask": action_m,
                "action_labels": action_y,
                "action_weights": action_w,
                "value": value_y,
                "solved": solved_y,
                "stock_closed": stock_y,
                "progressive": progressive_y,
                "compatibility": compatibility_y,
                "bottlenecks": bottleneck_y,
                "source_budget": source_budget_y,
            },
        )
        schema = {**schema, "feature_cache": str(cache_path), "feature_cache_hit": False}
    return SearchPolicyDataset(
        rows=pools,
        step_tokens=step_x,
        step_mask=step_m,
        route_features=route_x,
        node_features=node_x,
        node_mask=node_m,
        node_labels=node_y,
        node_weights=node_w,
        action_features=action_x,
        action_mask=action_m,
        action_labels=action_y,
        action_weights=action_w,
        value=value_y,
        solved=solved_y,
        stock_closed=stock_y,
        progressive=progressive_y,
        compatibility=compatibility_y,
        bottlenecks=bottleneck_y,
        source_budget=source_budget_y,
        feature_schema=schema,
    )


def _as_paths(paths: list[Path] | tuple[Path, ...] | Path) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _trace_leaf_utility_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Best observed utility for expanding a leaf within one benchmark target."""
    best: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        event = row.get("event") if "event" in row else row
        if not isinstance(event, dict):
            continue
        leaf = _canonical_leaf(event.get("expanded_leaf"))
        target_key = _trace_target_key(row, event)
        if not leaf or not target_key:
            continue
        best[(target_key, leaf)] = max(best[(target_key, leaf)], _expanded_leaf_utility(row, event))
    return dict(best)


def _trace_stock_aware_leaf_utility_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Best observed utility with extra late-depth stock and dead-end supervision."""
    best: dict[tuple[str, str], float] = defaultdict(float)
    low_yield_seen: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        event = row.get("event") if "event" in row else row
        if not isinstance(event, dict):
            continue
        leaf = _canonical_leaf(event.get("expanded_leaf"))
        target_key = _trace_target_key(row, event)
        if not leaf or not target_key:
            continue
        key = (target_key, leaf)
        utility = _stock_aware_expanded_leaf_utility(row, event, prior_low_yield=low_yield_seen.get(key, 0))
        best[key] = max(best[key], utility)
        if _expanded_leaf_low_yield(event):
            low_yield_seen[key] += 1
    return dict(best)


def _node_utility_labels_for_event(pool: dict[str, Any], event: dict[str, Any], open_leaves: list[str]) -> dict[str, float]:
    target_key = _trace_target_key(pool, event)
    replay = pool.get("_leaf_utility") or {}
    selected_leaf = _canonical_leaf(event.get("expanded_leaf") or pool.get("expanded_leaf"))
    if pool.get("_node_label_target") == "stock_aware_leaf_utility":
        selected_utility = _stock_aware_expanded_leaf_utility(pool.get("_trace_row") or pool, event)
    else:
        selected_utility = _expanded_leaf_utility(pool.get("_trace_row") or pool, event)
    labels: dict[str, float] = {}
    for leaf in open_leaves:
        leaf_key = _canonical_leaf(leaf)
        utility = float(replay.get((target_key, leaf_key), 0.0)) if target_key and leaf_key else 0.0
        if leaf_key == selected_leaf:
            utility = max(utility, selected_utility)
        labels[leaf_key] = float(np.clip(utility, 0.0, 1.0))
    if len(labels) > 1:
        best = max(labels.values(), default=0.0)
        for leaf_key, utility in list(labels.items()):
            if best >= 0.5 and utility >= best - 1e-6:
                labels[leaf_key] = min(1.0, max(0.65, utility))
            else:
                labels[leaf_key] = min(0.35, 0.5 * utility)
    if selected_leaf and selected_leaf in labels and max(labels.values(), default=0.0) <= 0.0:
        labels[selected_leaf] = 0.15
    return labels


def _expanded_leaf_utility(row: dict[str, Any], event: dict[str, Any]) -> float:
    outcome = event.get("outcome") or {}
    metrics = row.get("route_metrics") or []
    solved = bool(outcome.get("solved_routes")) or str(outcome.get("search_status") or "").lower() == "solved"
    stock_closed = any(bool((item or {}).get("strict_stock_solve")) for item in metrics)
    progressive = any(bool((item or {}).get("progressive_route")) for item in metrics) or solved
    compatibility = any(bool((item or {}).get("compatibility_success")) for item in metrics)
    actions = event.get("candidate_actions") or []
    selected = bool(event.get("selected_action_key"))
    gt_products = _gt_products(row.get("gt_route") or [])
    leaf_key = _canonical_leaf(event.get("expanded_leaf"))
    gt_product_hit = bool(leaf_key and leaf_key in gt_products)
    diagnostics = _diagnostics_for_leaf(event, event.get("expanded_leaf"))
    final_actions = sum(int(item.get("final_actions") or 0) for item in diagnostics)
    raw_actions = sum(int(item.get("raw_actions") or 0) for item in diagnostics)
    low_budget = any(int(item.get("proposal_budget") or 0) <= 2 for item in diagnostics)
    generated_ranked_out = bool(raw_actions > 0 and final_actions <= 0)
    utility = 0.03
    utility += 0.26 * float(bool(final_actions))
    utility += 0.12 * float(bool(actions))
    utility += 0.13 * float(selected)
    utility += 0.06 * float(solved)
    utility += 0.06 * float(stock_closed)
    utility += 0.04 * float(progressive)
    utility += 0.04 * float(compatibility)
    utility += 0.26 * float(gt_product_hit)
    if low_budget and not selected:
        utility -= 0.12
    if generated_ranked_out:
        utility -= 0.18
    if not actions and not final_actions:
        utility -= 0.20
    return float(np.clip(utility, 0.0, 1.0))


def _stock_aware_expanded_leaf_utility(
    row: dict[str, Any],
    event: dict[str, Any],
    *,
    prior_low_yield: int = 0,
) -> float:
    outcome = event.get("outcome") or {}
    metrics = row.get("route_metrics") or []
    solved = bool(outcome.get("solved_routes")) or str(outcome.get("search_status") or "").lower() == "solved"
    stock_closed_route = any(bool((item or {}).get("strict_stock_solve")) for item in metrics)
    selected_next_stock_closed = event.get("selected_next_stock_closed")
    selected = bool(event.get("selected_action_key"))
    diagnostics = _diagnostics_for_leaf(event, event.get("expanded_leaf"))
    final_actions = sum(int(item.get("final_actions") or 0) for item in diagnostics)
    raw_actions = sum(int(item.get("raw_actions") or 0) for item in diagnostics)
    invalid_filtered = sum(int(item.get("invalid_filtered") or 0) for item in diagnostics)
    open_count = len(event.get("open_leaves") or [])
    next_open = event.get("selected_next_open_leaves")
    leaf_key = _canonical_leaf(event.get("expanded_leaf"))
    gt_product_hit = bool(leaf_key and leaf_key in _gt_products(row.get("gt_route") or []))
    stock_hit = bool(event.get("expanded_leaf_stock_hit"))
    parent_adjacent = bool(event.get("expanded_leaf_parent_adjacent")) or _trace_leaf_parent_adjacent(event, leaf_key)
    low_yield = _expanded_leaf_low_yield(event) or bool(raw_actions > 0 and final_actions <= 1)
    depth = int(event.get("depth") or 0)
    remaining_depth = max(0, int(row.get("max_depth") or 6) - depth)
    late_depth = remaining_depth <= 1 or depth >= 5
    source_confidence = _source_policy_confidence_for_event(event)
    stock_progress = bool(selected_next_stock_closed) or (next_open is not None and open_count > 0 and int(next_open) < open_count)

    utility = 0.05
    utility += 0.18 * float(bool(final_actions))
    utility += 0.08 * min(float(final_actions or 0) / 4.0, 1.0)
    utility += 0.09 * float(bool(raw_actions))
    utility += 0.16 * float(selected)
    utility += 0.10 * float(solved)
    utility += 0.12 * float(stock_closed_route)
    utility += 0.20 * float(bool(selected_next_stock_closed))
    utility += 0.10 * float(stock_progress)
    utility += 0.13 * float(stock_hit and late_depth)
    utility += 0.12 * float(gt_product_hit)
    utility += 0.08 * float(parent_adjacent)
    utility += 0.06 * min(source_confidence, 1.0)

    if low_yield:
        utility -= 0.13
    if int(prior_low_yield or 0) > 0:
        utility -= min(0.20, 0.06 * int(prior_low_yield or 0))
    if late_depth and not stock_hit and not bool(selected_next_stock_closed):
        utility -= 0.14
    if not final_actions and not selected:
        utility -= 0.18
    if invalid_filtered and not final_actions:
        utility -= 0.08
    if depth >= 6 and not bool(selected_next_stock_closed):
        utility -= 0.08
    return float(np.clip(utility, 0.0, 1.0))


def _trace_action_label_and_weight(
    row: dict[str, Any],
    event: dict[str, Any],
    action_dict: dict[str, Any],
    candidate: dict[str, Any],
    *,
    key: str,
    selected_key: str,
    solved: bool,
    action_label_target: str,
) -> tuple[float, float, str]:
    selected = bool(selected_key and key == selected_key)
    if action_label_target not in {"stock_aware_action_utility", "stock_counterfactual_action_utility"}:
        label = 1.0 if solved and selected else 0.0
        return (
            label,
            2.0 if label else 1.0,
            "route_tree_selected_solved" if label else "route_tree_negative",
        )

    stock_closed_route = any(bool((item or {}).get("strict_stock_solve")) for item in row.get("route_metrics") or [])
    selected_next_stock_closed = bool(event.get("selected_next_stock_closed")) and selected
    product = str(event.get("expanded_leaf") or candidate.get("product") or "")
    progress = _candidate_progress_fraction(product, action_dict, candidate)
    terminal_frac = _candidate_small_terminal_fraction(action_dict, candidate)
    trace_stock_frac = _candidate_trace_stock_fraction(action_dict, candidate)
    trace_stock_closing = _candidate_trace_stock_closing(action_dict, candidate)
    has_condition = float(candidate.get("T") is not None and candidate.get("pH") is not None)
    has_type = float(bool(candidate.get("reaction_type") or candidate.get("type") or action_dict.get("reaction_type")))

    utility = 0.03
    utility += 0.12 * progress
    utility += 0.14 * terminal_frac
    if trace_stock_frac is not None:
        utility += 0.18 * trace_stock_frac
    utility += 0.24 * float(trace_stock_closing)
    utility += 0.03 * has_condition
    utility += 0.03 * has_type
    label_type = "route_tree_stock_aware_negative"
    if selected:
        utility = max(utility, 0.35)
        utility += 0.12 * float(solved)
        utility += 0.16 * float(stock_closed_route)
        utility += 0.26 * float(selected_next_stock_closed)
        utility += 0.08 * progress
        utility += 0.06 * terminal_frac
        label_type = "route_tree_selected_stock_aware"
        if selected_next_stock_closed:
            label_type = "route_tree_selected_stock_closing"
        elif solved:
            label_type = "route_tree_selected_solved"
    else:
        utility = min(0.35, utility)
        if action_label_target == "stock_counterfactual_action_utility" and trace_stock_closing:
            utility = max(utility, 0.62)
            label_type = "route_tree_sibling_stock_closing_positive"

    label = float(np.clip(utility, 0.0, 1.0))
    weight = 1.0
    if selected:
        weight += 1.5
    if selected_next_stock_closed:
        weight += 1.5
    if trace_stock_closing:
        weight += 1.25
    if solved and selected:
        weight += 0.75
    if label >= 0.5:
        weight += 0.5
    if action_label_target == "stock_counterfactual_action_utility":
        label, weight, label_type = _apply_counterfactual_action_penalty(
            row,
            event,
            action_dict,
            candidate,
            selected=selected,
            selected_next_stock_closed=selected_next_stock_closed,
            trace_stock_closing=trace_stock_closing,
            solved=solved,
            label=label,
            weight=weight,
            label_type=label_type,
            progress=progress,
            terminal_frac=terminal_frac,
        )
    return label, float(weight), label_type


def _action_label_notes(action_label_target: str) -> list[str]:
    if action_label_target == "stock_counterfactual_action_utility":
        return [
            "Starts from stock-aware action utility.",
            "Selected actions that lead to low-yield or dead-end non-stock states are downweighted as hard counterfactual negatives.",
            "Unselected stagnant or self-loop-like candidates receive stronger negative weight so ranking loss learns to separate them from useful stock/progress candidates.",
        ]
    if action_label_target == "stock_aware_action_utility":
        return [
            "Selected actions receive soft utility from solved outcome, strict stock closure, selected-next-state stock closure, progress, terminal reactant fraction, and condition/type completeness.",
            "Unselected actions receive only weak heuristic utility and are capped below the positive threshold.",
            "This gives stock-closing actions supervision even when the full route is not solved.",
        ]
    return [
        "Action labels are positive only for the selected action in a solved route-tree trace.",
        "This preserves legacy selected-action supervision.",
    ]


def _apply_counterfactual_action_penalty(
    row: dict[str, Any],
    event: dict[str, Any],
    action_dict: dict[str, Any],
    candidate: dict[str, Any],
    *,
    selected: bool,
    selected_next_stock_closed: bool,
    trace_stock_closing: bool,
    solved: bool,
    label: float,
    weight: float,
    label_type: str,
    progress: float,
    terminal_frac: float,
) -> tuple[float, float, str]:
    if selected_next_stock_closed or trace_stock_closing:
        return label, weight, label_type
    outcome = event.get("outcome") or {}
    depth = int(event.get("depth") or 0)
    next_open = event.get("selected_next_open_leaves")
    open_count = len(event.get("open_leaves") or [])
    route_stock_closed = any(bool((item or {}).get("strict_stock_solve")) for item in row.get("route_metrics") or [])
    route_failed = not solved and not route_stock_closed
    low_yield = _expanded_leaf_low_yield(event)
    dead_end_signal = int(outcome.get("dead_ends") or 0) > 0 or "no_route_returned" in set(outcome.get("route_tree_runtime_bottlenecks") or [])
    no_open_progress = selected and next_open is not None and open_count > 0 and int(next_open or 0) >= open_count
    self_loop = _candidate_self_loop_like(event.get("expanded_leaf") or candidate.get("product"), action_dict, candidate)
    stagnant = progress <= 0.02 and terminal_frac <= 0.0
    late_nonclosing = depth >= 5 and not selected_next_stock_closed and terminal_frac < 1.0

    hard_selected_dead_end = bool(selected and route_failed and (low_yield or dead_end_signal or no_open_progress))
    hard_unselected_stagnant = bool((not selected) and (self_loop or stagnant or late_nonclosing))

    if hard_selected_dead_end:
        return min(float(label), 0.22), float(weight + 2.0), "route_tree_selected_counterfactual_dead_end"
    if hard_unselected_stagnant:
        return min(float(label), 0.08 if self_loop else 0.14), float(weight + 1.5), "route_tree_hard_counterfactual_negative"
    if (not selected) and progress <= 0.05 and terminal_frac < 0.5:
        return min(float(label), 0.22), float(weight + 0.75), "route_tree_counterfactual_low_progress_negative"
    return label, weight, label_type


def _candidate_self_loop_like(product: Any, action_dict: dict[str, Any], candidate: dict[str, Any]) -> bool:
    product_key = canonical_smiles(product) or str(product or "")
    reactants = _candidate_reactants_for_label(action_dict, candidate)
    if any((canonical_smiles(smi) or smi) == product_key for smi in reactants):
        return True
    flags = set(str(flag) for flag in action_dict.get("validity_flags") or candidate.get("validity_flags") or [])
    return bool({"self_loop", "product_mismatch"} & flags)


def _candidate_trace_stock_fraction(action_dict: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    for source in (action_dict, candidate):
        if source.get("reactant_stock_fraction") is not None:
            try:
                return float(np.clip(float(source.get("reactant_stock_fraction")), 0.0, 1.0))
            except Exception:
                return None
        status = source.get("reactant_stock_status")
        if isinstance(status, dict) and status:
            values = [bool(value) for value in status.values()]
            return float(sum(1 for value in values if value) / max(len(values), 1))
    return None


def _candidate_trace_stock_closing(action_dict: dict[str, Any], candidate: dict[str, Any]) -> bool:
    for source in (action_dict, candidate):
        if source.get("stock_closing_candidate") is not None:
            return bool(source.get("stock_closing_candidate"))
    frac = _candidate_trace_stock_fraction(action_dict, candidate)
    return bool(frac is not None and frac >= 1.0)


def _candidate_progress_fraction(product: str, action_dict: dict[str, Any], candidate: dict[str, Any]) -> float:
    product_atoms = _heavy_atoms(product)
    if product_atoms <= 0:
        return 0.0
    largest = max((_heavy_atoms(smi) for smi in _candidate_reactants_for_label(action_dict, candidate)), default=product_atoms)
    return float(np.clip((product_atoms - largest) / max(product_atoms, 1), 0.0, 1.0))


def _candidate_small_terminal_fraction(action_dict: dict[str, Any], candidate: dict[str, Any]) -> float:
    reactants = _candidate_reactants_for_label(action_dict, candidate)
    if not reactants:
        return 0.0
    terminal = sum(1 for smi in reactants if 0 < _heavy_atoms(smi) <= 6)
    return float(terminal / max(len(reactants), 1))


def _candidate_reactants_for_label(action_dict: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    reactants = [str(smi) for smi in action_dict.get("reactants") or [] if smi]
    if reactants:
        return reactants
    out = []
    main = str(candidate.get("main_reactant") or "")
    if main:
        out.append(main)
    out.extend(str(smi) for smi in candidate.get("aux_reactants") or [] if smi)
    return out


def _heavy_atoms(smiles: Any) -> int:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles or ""))
    except Exception:
        return 0
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _expanded_leaf_low_yield(event: dict[str, Any]) -> bool:
    if event.get("expanded_leaf_low_yield") is not None:
        return bool(event.get("expanded_leaf_low_yield"))
    diagnostics = _diagnostics_for_leaf(event, event.get("expanded_leaf"))
    raw_actions = sum(int(item.get("raw_actions") or 0) for item in diagnostics)
    final_actions = sum(int(item.get("final_actions") or 0) for item in diagnostics)
    return bool(raw_actions > 0 and final_actions <= 1)


def _trace_leaf_parent_adjacent(event: dict[str, Any], leaf_key: str) -> bool:
    state = event.get("state") or {}
    for smi in _trace_parent_reactants(state):
        if _canonical_leaf(smi) == leaf_key:
            return True
    return False


def _source_policy_confidence_for_event(event: dict[str, Any]) -> float:
    scores = []
    for item in event.get("proposal_diagnostics") or []:
        allocation = (item or {}).get("allocation") or {}
        scores.append(float(allocation.get("policy_confidence") or 0.0))
    for item in event.get("source_budgets") or []:
        gate = (item or {}).get("proposal_gate") or {}
        scores.append(float(gate.get("policy_confidence") or 0.0))
    return max(scores, default=0.0)


def _trace_target_key(row: dict[str, Any], event: dict[str, Any]) -> str:
    benchmark_index = row.get("benchmark_index")
    if benchmark_index is None:
        benchmark_index = row.get("_benchmark_index")
    if benchmark_index is not None:
        return f"idx:{benchmark_index}"
    target = row.get("target_smiles") or event.get("target_smiles") or (event.get("state") or {}).get("target")
    return f"target:{canonical_smiles(target) or target or ''}"


def _canonical_leaf(value: Any) -> str:
    return canonical_smiles(str(value or "")) or str(value or "")


def _gt_products(gt_route: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for step in gt_route:
        rxn = str((step or {}).get("rxn_smiles") or (step or {}).get("reaction_smiles") or "")
        if ">>" not in rxn:
            continue
        rhs = rxn.split(">>", 1)[1]
        can = canonical_smiles(rhs)
        if can:
            out.add(can)
    return out


def _diagnostics_for_leaf(event: dict[str, Any], leaf: Any) -> list[dict[str, Any]]:
    leaf_key = _canonical_leaf(leaf)
    out = []
    for item in event.get("proposal_diagnostics") or []:
        if _canonical_leaf((item or {}).get("leaf")) == leaf_key:
            out.append(item or {})
    return out


def _trace_parent_reactants(state: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for step in state.get("steps") or []:
        action = (step or {}).get("action") or {}
        for smi in action.get("reactants") or []:
            can = canonical_smiles(smi) or smi
            if can:
                out.add(can)
    return out


def _route_row_from_trace(row: dict[str, Any], event: dict[str, Any], *, route_id: str, target: str) -> dict[str, Any]:
    state = event.get("state") or {}
    steps = state.get("steps") or []
    route_metrics = row.get("route_metrics") or []
    outcome = event.get("outcome") or {}
    solved = bool(outcome.get("solved_routes")) or str(outcome.get("search_status") or "").lower() == "solved"
    stock_closed = any(bool((metrics or {}).get("strict_stock_solve")) for metrics in route_metrics) or solved
    progressive = any(bool((metrics or {}).get("progressive_route")) for metrics in route_metrics) or solved
    compatibility = any(bool((metrics or {}).get("compatibility_success")) for metrics in route_metrics)
    bottlenecks = _trace_bottleneck_labels(solved, outcome, event)
    type_sequence: list[str] = []
    ec1_sequence: list[str] = []
    source_sequence: list[str] = []
    for step in steps:
        action = (step or {}).get("action") or {}
        typ = action.get("reaction_type") or action.get("type") or ""
        ec = action.get("ec") or ""
        src = action.get("source") or "unknown"
        type_sequence.append(str(typ))
        ec1_sequence.append(str(ec).split(".", 1)[0] if ec else "")
        source_sequence.append(str(src))
    features = {
        "filled_route": float(bool(steps)),
        "progressive_route": float(progressive),
        "route_solved": float(solved),
        "strict_stock_solve": 1.0 if stock_closed else -0.5,
        "main_chain_reduction": 1.0 if progressive else 0.0,
        "leaf_reduction": 1.0 if progressive else 0.0,
        "naturalness": 1.0 if solved else 0.0,
        "condition_success": float(solved),
        "compatibility_success": float(compatibility),
        "enzyme_evidence": float(any(ec1_sequence)),
        "issue_count": 0.0 if solved else 1.0,
    }
    return {
        "route_id": route_id,
        "target_smiles": target,
        "label": 1.0 if solved else 0.0,
        "label_type": "route_tree_solved" if solved else "route_tree_failed",
        "n_steps": int(state.get("depth") or event.get("depth") or len(steps) or 0),
        "type_sequence": type_sequence,
        "ec1_sequence": ec1_sequence,
        "source_sequence": source_sequence,
        "operation_mode": "unknown",
        "bottleneck_labels": bottlenecks,
        "recovery_bottleneck_labels": bottlenecks,
        "features": features,
        "metrics_summary": route_metrics[0] if route_metrics else {},
        "score": float(outcome.get("solved_routes") or 0.0) * 100.0,
        "confidence": 0.75 if solved else 0.25,
    }


def _route_row_from_candidate_pool(pool: dict[str, Any]) -> dict[str, Any]:
    candidates = pool.get("candidates") or []
    best = max((float((item or {}).get("label") or 0.0) for item in candidates), default=0.0)
    has_positive = best >= 0.5
    top_candidate = None
    for item in candidates:
        if float((item or {}).get("label") or 0.0) >= best:
            top_candidate = item or {}
            break
    candidate = (top_candidate or {}).get("candidate") or {}
    candidate_type = str(candidate.get("type") or candidate.get("reaction_type") or "")
    candidate_ec = str(candidate.get("ec") or "")
    candidate_source = str(candidate.get("source") or "unknown")
    features = {
        "filled_route": 1.0 if candidates else 0.0,
        "progressive_route": 1.0 if has_positive else 0.0,
        "route_solved": 1.0 if best >= 0.75 else 0.0,
        "strict_stock_solve": 1.0 if best >= 0.75 else 0.0,
        "main_chain_reduction": 0.5 if has_positive else 0.0,
        "leaf_reduction": 0.5 if has_positive else 0.0,
        "naturalness": 1.0 if has_positive else 0.0,
        "condition_success": 1.0 if candidate.get("T") is not None or candidate.get("pH") is not None else 0.0,
        "compatibility_success": 1.0 if candidate_type or candidate_ec else 0.0,
        "enzyme_evidence": 1.0 if candidate_ec else 0.0,
        "issue_count": 0.0 if has_positive else 1.0,
    }
    return {
        "route_id": str(pool.get("route_id") or pool.get("pool_id") or pool.get("state_id") or ""),
        "target_smiles": str(pool.get("target_smiles") or ""),
        "label": float(np.clip(best, 0.0, 1.0)),
        "value_target": float(np.clip(best, 0.0, 1.0)),
        "label_type": "candidate_pool_best_label" if has_positive else "candidate_pool_negative",
        "n_steps": int(pool.get("step_index") or 0) + 1,
        "type_sequence": [candidate_type] if candidate_type else [],
        "ec1_sequence": [candidate_ec.split(".", 1)[0] if candidate_ec else ""],
        "source_sequence": [candidate_source],
        "operation_mode": "unknown",
        "bottleneck_labels": [] if has_positive else ["candidate_generator_reactant_miss"],
        "recovery_bottleneck_labels": [] if has_positive else ["candidate_generator_reactant_miss"],
        "features": features,
        "metrics_summary": {},
        "score": float(best * 100.0),
        "confidence": 0.5 + 0.5 * best,
    }


def _trace_bottleneck_labels(solved: bool, outcome: dict[str, Any], event: dict[str, Any]) -> list[str]:
    if solved:
        return []
    actions = event.get("candidate_actions") or []
    if not actions:
        return ["candidate_generator_reactant_miss", "no_route_returned"]
    if int(outcome.get("dead_ends") or 0) > 0:
        return ["stock_dead_end", "no_professional_solved_route"]
    return ["no_professional_solved_route", "no_route_returned"]


def _trace_action_candidate(action: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "main_reactant": action.get("main_reactant") or "",
        "aux_reactants": action.get("aux_reactants") or [],
        "rxn_smiles": action.get("rxn_smiles") or action.get("reaction_smiles") or "",
        "reaction_smiles": action.get("rxn_smiles") or action.get("reaction_smiles") or "",
        "source": action.get("source") or "unknown",
        "score": action.get("raw_score", action.get("score", 0.0)),
        "rank": action.get("rank") or 0,
        "type": action.get("reaction_type") or action.get("type") or "",
        "reaction_type": action.get("reaction_type") or action.get("type") or "",
        "ec": action.get("ec") or "",
        "catalyst": action.get("catalyst") or "",
        "T": action.get("T"),
        "pH": action.get("pH"),
        "solvent": action.get("solvent") or "",
    }
    metadata = action.get("metadata") or {}
    if isinstance(metadata, dict):
        candidate.update(metadata)
    return candidate


def train_step_encoder_from_vnext_pack(
    *,
    vnext_pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 6,
    batch_size: int = 512,
    lr: float = 1e-3,
    n_bits: int = 256,
    d_model: int = 128,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = _select_device(device)
    dataset = build_step_pair_dataset(vnext_pack_dir, n_bits=n_bits)
    train_idx, val_idx = split_by_target(dataset.rows)
    model = StepEncoder(n_bits=n_bits, metadata_dim=dataset.metadata.shape[1], d_model=d_model).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(
        torch.tensor(dataset.product_fp[train_idx]),
        torch.tensor(dataset.reactant_fp[train_idx]),
        torch.tensor(dataset.metadata[train_idx]),
        torch.tensor(dataset.labels[train_idx]),
        torch.tensor(dataset.reaction_type[train_idx]),
        torch.tensor(dataset.ec1[train_idx]),
        torch.tensor(dataset.condition[train_idx]),
        torch.tensor(dataset.weights[train_idx]),
    ), batch_size=batch_size, shuffle=True)
    history = []
    best_state = None
    best_metrics: dict[str, Any] | None = None
    best_val = float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for product, reactants, metadata, y, typ, ec1, cond, weight in dl:
            product = product.to(device_t)
            reactants = reactants.to(device_t)
            metadata = metadata.to(device_t)
            y = y.to(device_t)
            typ = typ.to(device_t)
            ec1 = ec1.to(device_t)
            cond = cond.to(device_t)
            weight = weight.to(device_t)
            out = model(product, reactants, metadata)
            loss_match = nn.functional.binary_cross_entropy_with_logits(out["match_logit"], y, reduction="none")
            loss_type = nn.functional.cross_entropy(out["reaction_type_logits"], typ, reduction="none")
            loss_ec = nn.functional.cross_entropy(out["ec1_logits"], ec1, reduction="none")
            loss_cond = nn.functional.smooth_l1_loss(out["condition"], cond, reduction="none").mean(dim=-1)
            loss = ((loss_match + 0.15 * loss_type + 0.15 * loss_ec + 0.1 * loss_cond) * weight).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(product)
            n_seen += len(product)
        val_loss, val_acc = _eval_step_encoder(model, dataset, val_idx)
        history.append({"epoch": epoch + 1, "train_loss": round(total / max(n_seen, 1), 6), "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6)})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    report = {
        "metadata": {
            "model_kind": "step_encoder",
            "vnext_pack": str(vnext_pack_dir),
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "d_model": d_model,
            "device": str(device_t),
            "feature_schema": dataset.feature_schema,
        },
        "best_val_loss": round(best_val, 6),
        "history": history,
    }
    _save_checkpoint(model_output, model, report["metadata"], dataset.feature_schema, {"d_model": d_model})
    _write_reports(report, report_output, md_output, title="vNext StepEncoder")
    return report


def train_candidate_pool_ranker_from_vnext_pack(
    *,
    vnext_pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_bits: int = 256,
    max_candidates: int = 32,
    d_model: int = 128,
    seed: int = 42,
    device: str = "auto",
    feature_cache_dir: Path | None = None,
    feature_workers: int = 1,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = _select_device(device)
    dataset = build_candidate_pool_dataset(
        vnext_pack_dir,
        n_bits=n_bits,
        max_candidates=max_candidates,
        feature_cache_dir=feature_cache_dir,
        feature_workers=feature_workers,
    )
    train_idx, val_idx = split_by_target(dataset.rows)
    model = CandidatePoolCrossAttentionRanker(candidate_feature_dim=dataset.candidate_features.shape[-1], d_model=d_model).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(
        torch.tensor(dataset.candidate_features[train_idx]),
        torch.tensor(dataset.candidate_mask[train_idx]),
        torch.tensor(dataset.labels[train_idx]),
        torch.tensor(dataset.weights[train_idx]),
    ), batch_size=batch_size, shuffle=True)
    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for x, mask, y, weight in dl:
            x = x.to(device_t)
            mask = mask.to(device_t)
            y = y.to(device_t)
            weight = weight.to(device_t)
            out = model(x, mask.bool())
            loss = _masked_bce(out["candidate_logits"], y, mask, weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(x)
            n_seen += len(x)
        val_loss, top1 = _eval_candidate_pool(model, dataset, val_idx)
        history.append({"epoch": epoch + 1, "train_loss": round(total / max(n_seen, 1), 6), "val_loss": round(val_loss, 6), "val_top1_positive": round(top1, 6)})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    report = {
        "metadata": {
            "model_kind": "candidate_pool_ranker",
            "vnext_pack": str(vnext_pack_dir),
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "d_model": d_model,
            "device": str(device_t),
            "feature_schema": dataset.feature_schema,
            "feature_workers": int(feature_workers or 1),
        },
        "best_val_loss": round(best_val, 6),
        "history": history,
        "reranking": {"val_top1_positive": history[-1]["val_top1_positive"] if history else None},
    }
    _save_checkpoint(model_output, model, report["metadata"], dataset.feature_schema, {"d_model": d_model})
    _write_reports(report, report_output, md_output, title="vNext Candidate Pool Ranker")
    return report


def train_route_state_from_vnext_pack(
    *,
    vnext_pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_steps: int = 8,
    d_model: int = 128,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = _select_device(device)
    dataset = build_route_state_dataset(vnext_pack_dir, max_steps=max_steps)
    train_idx, val_idx = split_by_target(dataset.rows)
    model = RouteStateTransformer(
        step_token_dim=dataset.step_tokens.shape[-1],
        route_feature_dim=dataset.route_features.shape[-1],
        max_steps=max_steps,
        d_model=d_model,
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(
        torch.tensor(dataset.step_tokens[train_idx]),
        torch.tensor(dataset.step_mask[train_idx]),
        torch.tensor(dataset.route_features[train_idx]),
        torch.tensor(dataset.value[train_idx]),
        torch.tensor(dataset.solved[train_idx]),
        torch.tensor(dataset.stock_closed[train_idx]),
        torch.tensor(dataset.progressive[train_idx]),
        torch.tensor(dataset.compatibility[train_idx]),
        torch.tensor(dataset.bottlenecks[train_idx]),
    ), batch_size=batch_size, shuffle=True)
    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for steps, step_mask, route_features, value, solved, stock, progressive, compatibility, bottlenecks in dl:
            steps = steps.to(device_t)
            step_mask = step_mask.to(device_t)
            route_features = route_features.to(device_t)
            value = value.to(device_t)
            solved = solved.to(device_t)
            stock = stock.to(device_t)
            progressive = progressive.to(device_t)
            compatibility = compatibility.to(device_t)
            bottlenecks = bottlenecks.to(device_t)
            out = model(steps, step_mask.bool(), route_features)
            loss = _route_loss(out, value, solved, stock, progressive, compatibility, bottlenecks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(steps)
            n_seen += len(steps)
        val_metrics = _eval_route_state(model, dataset, val_idx)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            **_rounded_metrics(val_metrics),
        })
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    report = {
        "metadata": {
            "model_kind": "route_state_transformer",
            "vnext_pack": str(vnext_pack_dir),
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "d_model": d_model,
            "device": str(device_t),
            "feature_schema": dataset.feature_schema,
            "value_head_supervised": True,
            "value_calibrated": False,
            "value_calibration": _value_calibration_metadata(history),
        },
        "best_val_loss": round(best_val, 6),
        "history": history,
    }
    _save_checkpoint(model_output, model, report["metadata"], dataset.feature_schema, {"d_model": d_model})
    _write_reports(report, report_output, md_output, title="vNext Route State Transformer")
    return report


def train_search_policy_from_vnext_pack(
    *,
    vnext_pack_dir: Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_bits: int = 256,
    max_candidates: int = 32,
    max_steps: int = 8,
    d_model: int = 128,
    seed: int = 42,
    device: str = "auto",
    feature_cache_dir: Path | None = None,
    feature_workers: int = 1,
    init_checkpoint: Path | None = None,
    policy_train_mode: str = "all",
    action_rank_loss_weight: float = 0.0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy_train_mode = _normalize_search_policy_train_mode(policy_train_mode)
    device_t = _select_device(device)
    dataset = build_search_policy_dataset(
        vnext_pack_dir,
        n_bits=n_bits,
        max_candidates=max_candidates,
        max_steps=max_steps,
        feature_cache_dir=feature_cache_dir,
        feature_workers=feature_workers,
    )
    train_idx, val_idx = split_by_target(dataset.rows)
    route_model = RouteStateTransformer(
        step_token_dim=dataset.step_tokens.shape[-1],
        route_feature_dim=dataset.route_features.shape[-1],
        max_steps=max_steps,
        d_model=d_model,
    )
    model = SearchPolicyNetwork(
        route_model=route_model,
        action_feature_dim=dataset.action_features.shape[-1],
        node_feature_dim=dataset.node_features.shape[-1],
        d_model=d_model,
        n_source_budgets=dataset.source_budget.shape[-1],
    ).to(device_t)
    init_report = _load_initial_state(model, init_checkpoint) if init_checkpoint else None
    train_mode_report = _configure_search_policy_train_mode(model, policy_train_mode)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(
        torch.tensor(dataset.step_tokens[train_idx]),
        torch.tensor(dataset.step_mask[train_idx]),
        torch.tensor(dataset.route_features[train_idx]),
        torch.tensor(dataset.node_features[train_idx]),
        torch.tensor(dataset.node_mask[train_idx]),
        torch.tensor(dataset.node_labels[train_idx]),
        torch.tensor(dataset.node_weights[train_idx]),
        torch.tensor(dataset.action_features[train_idx]),
        torch.tensor(dataset.action_mask[train_idx]),
        torch.tensor(dataset.action_labels[train_idx]),
        torch.tensor(dataset.action_weights[train_idx]),
        torch.tensor(dataset.value[train_idx]),
        torch.tensor(dataset.solved[train_idx]),
        torch.tensor(dataset.stock_closed[train_idx]),
        torch.tensor(dataset.progressive[train_idx]),
        torch.tensor(dataset.compatibility[train_idx]),
        torch.tensor(dataset.bottlenecks[train_idx]),
        torch.tensor(dataset.source_budget[train_idx]),
    ), batch_size=batch_size, shuffle=True)
    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for (
            steps,
            step_mask,
            route_features,
            node_features,
            node_mask,
            node_labels,
            node_weights,
            actions,
            action_mask,
            labels,
            action_weights,
            value,
            solved,
            stock,
            progressive,
            compatibility,
            bottlenecks,
            source_budget,
        ) in dl:
            steps = steps.to(device_t)
            step_mask = step_mask.to(device_t)
            route_features = route_features.to(device_t)
            node_features = node_features.to(device_t)
            node_mask = node_mask.to(device_t)
            node_labels = node_labels.to(device_t)
            node_weights = node_weights.to(device_t)
            actions = actions.to(device_t)
            action_mask = action_mask.to(device_t)
            labels = labels.to(device_t)
            action_weights = action_weights.to(device_t)
            value = value.to(device_t)
            solved = solved.to(device_t)
            stock = stock.to(device_t)
            progressive = progressive.to(device_t)
            compatibility = compatibility.to(device_t)
            bottlenecks = bottlenecks.to(device_t)
            source_budget = source_budget.to(device_t)
            out = model(
                steps,
                step_mask.bool(),
                route_features,
                actions,
                action_mask.bool(),
                node_features,
                node_mask.bool(),
            )
            node_loss = _masked_bce(out["node_policy_logits"], node_labels, node_mask, node_weights)
            action_loss = _masked_bce(out["action_logits"], labels, action_mask, action_weights)
            action_rank_loss = _action_pairwise_ranking_loss(out["action_logits"], labels, action_mask, action_weights)
            route_loss = _route_loss(out, value, solved, stock, progressive, compatibility, bottlenecks)
            budget_loss = nn.functional.smooth_l1_loss(torch.sigmoid(out["budget_logits"]), source_budget)
            loss = _search_policy_training_loss(
                policy_train_mode,
                node_loss=node_loss,
                action_loss=action_loss,
                action_rank_loss=action_rank_loss,
                action_rank_loss_weight=action_rank_loss_weight,
                route_loss=route_loss,
                budget_loss=budget_loss,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(steps)
            n_seen += len(steps)
        val_metrics = _eval_search_policy(model, dataset, val_idx)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            **_rounded_metrics(val_metrics),
        })
        selection_metric = _search_policy_selection_metric(
            policy_train_mode,
            val_metrics,
            action_rank_loss_weight=action_rank_loss_weight,
        )
        if selection_metric < best_val:
            best_val = selection_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = dict(history[-1])
    if best_state:
        model.load_state_dict(best_state)
    policy_metrics = best_metrics or (history[-1] if history else {})
    report = {
        "metadata": {
            "model_kind": "search_policy",
            "vnext_pack": str(vnext_pack_dir),
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "d_model": d_model,
            "device": str(device_t),
            "feature_schema": dataset.feature_schema,
            "feature_workers": int(feature_workers or 1),
            "value_head_supervised": True,
            "value_calibrated": False,
            "value_calibration": _value_calibration_metadata(history),
            "node_policy_supervised": True,
            "budget_head_supervised": True,
            "policy_train_mode": policy_train_mode,
            "policy_train_mode_report": train_mode_report,
            "action_rank_loss_weight": float(action_rank_loss_weight or 0.0),
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "init_checkpoint_load": init_report,
        },
        "best_val_loss": round(best_val, 6),
        "best_selection_metric": _search_policy_selection_metric_name(
            policy_train_mode,
            action_rank_loss_weight=action_rank_loss_weight,
        ),
        "history": history,
        "policy": {
            "selected_epoch": policy_metrics.get("epoch"),
            "val_node_top1": policy_metrics.get("val_node_top1"),
            "val_node_top3": policy_metrics.get("val_node_top3"),
            "val_top1_positive": policy_metrics.get("val_top1_positive"),
            "val_top5_positive": policy_metrics.get("val_top5_positive"),
            "val_top1_stock_closing_positive": policy_metrics.get("val_top1_stock_closing_positive"),
            "val_top1_useful_candidate_positive": policy_metrics.get("val_top1_useful_candidate_positive"),
            "val_stock_dead_end_auc": policy_metrics.get("val_stock_dead_end_auc"),
            "val_node_calibration_by_remaining_depth": policy_metrics.get("val_node_calibration_by_remaining_depth"),
        },
    }
    _save_checkpoint(
        model_output,
        model,
        report["metadata"],
        dataset.feature_schema,
        {
            "d_model": d_model,
            "node_feature_dim": int(dataset.node_features.shape[-1]),
            "n_source_budgets": int(dataset.source_budget.shape[-1]),
        },
    )
    _write_reports(report, report_output, md_output, title="vNext Search Policy")
    return report


def train_search_policy_from_route_tree_traces(
    *,
    trace_paths: list[Path] | tuple[Path, ...] | Path,
    model_output: Path,
    report_output: Path,
    md_output: Path | None = None,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_bits: int = 256,
    max_candidates: int = 32,
    max_steps: int = 8,
    d_model: int = 128,
    node_label_target: str = "trace_leaf_utility",
    action_label_target: str = "selected_solved_action",
    seed: int = 42,
    device: str = "auto",
    feature_cache_dir: Path | None = None,
    feature_workers: int = 1,
    init_checkpoint: Path | None = None,
    policy_train_mode: str = "all",
    action_rank_loss_weight: float = 0.0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy_train_mode = _normalize_search_policy_train_mode(policy_train_mode)
    paths = _as_paths(trace_paths)
    device_t = _select_device(device)
    dataset = build_route_tree_search_policy_dataset(
        paths,
        n_bits=n_bits,
        max_candidates=max_candidates,
        max_steps=max_steps,
        node_label_target=node_label_target,
        action_label_target=action_label_target,
        feature_cache_dir=feature_cache_dir,
        feature_workers=feature_workers,
    )
    train_idx, val_idx = split_by_target(dataset.rows)
    route_model = RouteStateTransformer(
        step_token_dim=dataset.step_tokens.shape[-1],
        route_feature_dim=dataset.route_features.shape[-1],
        max_steps=max_steps,
        d_model=d_model,
    )
    model = SearchPolicyNetwork(
        route_model=route_model,
        action_feature_dim=dataset.action_features.shape[-1],
        node_feature_dim=dataset.node_features.shape[-1],
        d_model=d_model,
        n_source_budgets=dataset.source_budget.shape[-1],
    ).to(device_t)
    init_report = _load_initial_state(model, init_checkpoint) if init_checkpoint else None
    train_mode_report = _configure_search_policy_train_mode(model, policy_train_mode)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(
        torch.tensor(dataset.step_tokens[train_idx]),
        torch.tensor(dataset.step_mask[train_idx]),
        torch.tensor(dataset.route_features[train_idx]),
        torch.tensor(dataset.node_features[train_idx]),
        torch.tensor(dataset.node_mask[train_idx]),
        torch.tensor(dataset.node_labels[train_idx]),
        torch.tensor(dataset.node_weights[train_idx]),
        torch.tensor(dataset.action_features[train_idx]),
        torch.tensor(dataset.action_mask[train_idx]),
        torch.tensor(dataset.action_labels[train_idx]),
        torch.tensor(dataset.action_weights[train_idx]),
        torch.tensor(dataset.value[train_idx]),
        torch.tensor(dataset.solved[train_idx]),
        torch.tensor(dataset.stock_closed[train_idx]),
        torch.tensor(dataset.progressive[train_idx]),
        torch.tensor(dataset.compatibility[train_idx]),
        torch.tensor(dataset.bottlenecks[train_idx]),
        torch.tensor(dataset.source_budget[train_idx]),
    ), batch_size=batch_size, shuffle=True)
    history = []
    best_state = None
    best_metrics: dict[str, Any] | None = None
    best_val = float("inf")
    for epoch in range(max(1, epochs)):
        model.train()
        total = 0.0
        n_seen = 0
        for (
            steps,
            step_mask,
            route_features,
            node_features,
            node_mask,
            node_labels,
            node_weights,
            actions,
            action_mask,
            labels,
            action_weights,
            value,
            solved,
            stock,
            progressive,
            compatibility,
            bottlenecks,
            source_budget,
        ) in dl:
            steps = steps.to(device_t)
            step_mask = step_mask.to(device_t)
            route_features = route_features.to(device_t)
            node_features = node_features.to(device_t)
            node_mask = node_mask.to(device_t)
            node_labels = node_labels.to(device_t)
            node_weights = node_weights.to(device_t)
            actions = actions.to(device_t)
            action_mask = action_mask.to(device_t)
            labels = labels.to(device_t)
            action_weights = action_weights.to(device_t)
            value = value.to(device_t)
            solved = solved.to(device_t)
            stock = stock.to(device_t)
            progressive = progressive.to(device_t)
            compatibility = compatibility.to(device_t)
            bottlenecks = bottlenecks.to(device_t)
            source_budget = source_budget.to(device_t)
            out = model(
                steps,
                step_mask.bool(),
                route_features,
                actions,
                action_mask.bool(),
                node_features,
                node_mask.bool(),
            )
            node_loss = _masked_bce(out["node_policy_logits"], node_labels, node_mask, node_weights)
            action_loss = _masked_bce(out["action_logits"], labels, action_mask, action_weights)
            action_rank_loss = _action_pairwise_ranking_loss(out["action_logits"], labels, action_mask, action_weights)
            route_loss = _route_loss(out, value, solved, stock, progressive, compatibility, bottlenecks)
            budget_loss = nn.functional.smooth_l1_loss(torch.sigmoid(out["budget_logits"]), source_budget)
            loss = _search_policy_training_loss(
                policy_train_mode,
                node_loss=node_loss,
                action_loss=action_loss,
                action_rank_loss=action_rank_loss,
                action_rank_loss_weight=action_rank_loss_weight,
                route_loss=route_loss,
                budget_loss=budget_loss,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(steps)
            n_seen += len(steps)
        val_metrics = _eval_search_policy(model, dataset, val_idx)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(total / max(n_seen, 1), 6),
            **_rounded_metrics(val_metrics),
        })
        selection_metric = _search_policy_selection_metric(
            policy_train_mode,
            val_metrics,
            action_rank_loss_weight=action_rank_loss_weight,
        )
        if selection_metric < best_val:
            best_val = selection_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = dict(history[-1])
    if best_state:
        model.load_state_dict(best_state)
    policy_metrics = best_metrics or (history[-1] if history else {})
    report = {
        "metadata": {
            "model_kind": "search_policy",
            "training_source": "route_tree_traces",
            "route_tree_traces": [str(path) for path in paths],
            "model_output": str(model_output),
            "n_rows": len(dataset.rows),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "d_model": d_model,
            "device": str(device_t),
            "feature_schema": dataset.feature_schema,
            "node_label_target": node_label_target,
            "action_label_target": action_label_target,
            "feature_workers": int(feature_workers or 1),
            "value_head_supervised": True,
            "value_calibrated": False,
            "value_calibration": _value_calibration_metadata(history),
            "node_policy_supervised": True,
            "budget_head_supervised": True,
            "policy_train_mode": policy_train_mode,
            "policy_train_mode_report": train_mode_report,
            "action_rank_loss_weight": float(action_rank_loss_weight or 0.0),
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "init_checkpoint_load": init_report,
        },
        "data_summary": {
            "trace_count": len(paths),
            "supervision_row_count": len(dataset.rows),
            "target_count": _target_count(dataset.rows),
            "train_target_count": _target_count(dataset.rows, train_idx),
            "val_target_count": _target_count(dataset.rows, val_idx),
        },
        "best_val_loss": round(best_val, 6),
        "best_selection_metric": _search_policy_selection_metric_name(
            policy_train_mode,
            action_rank_loss_weight=action_rank_loss_weight,
        ),
        "history": history,
        "policy": {
            "selected_epoch": policy_metrics.get("epoch"),
            "val_node_top1": policy_metrics.get("val_node_top1"),
            "val_node_top3": policy_metrics.get("val_node_top3"),
            "val_top1_positive": policy_metrics.get("val_top1_positive"),
            "val_top5_positive": policy_metrics.get("val_top5_positive"),
            "val_top1_stock_closing_positive": policy_metrics.get("val_top1_stock_closing_positive"),
            "val_top1_useful_candidate_positive": policy_metrics.get("val_top1_useful_candidate_positive"),
            "val_stock_dead_end_auc": policy_metrics.get("val_stock_dead_end_auc"),
            "val_node_calibration_by_remaining_depth": policy_metrics.get("val_node_calibration_by_remaining_depth"),
        },
    }
    _save_checkpoint(
        model_output,
        model,
        report["metadata"],
        dataset.feature_schema,
        {
            "d_model": d_model,
            "node_feature_dim": int(dataset.node_features.shape[-1]),
            "n_source_budgets": int(dataset.source_budget.shape[-1]),
        },
    )
    _write_reports(report, report_output, md_output, title="Route Tree Search Policy")
    return report


def _eval_step_encoder(model: StepEncoder, dataset: StepPairDataset, idx: list[int]) -> tuple[float, float]:
    if not idx:
        return 0.0, 0.0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(dataset.product_fp[idx], device=device),
            torch.tensor(dataset.reactant_fp[idx], device=device),
            torch.tensor(dataset.metadata[idx], device=device),
        )
        y = torch.tensor(dataset.labels[idx], device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(out["match_logit"], y)
        pred = (torch.sigmoid(out["match_logit"]) >= 0.5).float()
        acc = float((pred == (y >= 0.5).float()).float().mean().item())
    return float(loss.item()), acc


def _eval_candidate_pool(model: CandidatePoolCrossAttentionRanker, dataset: CandidatePoolDataset, idx: list[int]) -> tuple[float, float]:
    if not idx:
        return 0.0, 0.0
    device = next(model.parameters()).device
    model.eval()
    x = torch.tensor(dataset.candidate_features[idx], device=device)
    mask = torch.tensor(dataset.candidate_mask[idx], device=device)
    y = torch.tensor(dataset.labels[idx], device=device)
    weight = torch.tensor(dataset.weights[idx], device=device)
    with torch.no_grad():
        logits = model(x, mask.bool())["candidate_logits"]
        loss = _masked_bce(logits, y, mask, weight)
        top = torch.argmax(logits, dim=1)
        hits = []
        for i, j in enumerate(top.tolist()):
            positives = (y[i] >= 0.75) & (mask[i] > 0)
            if positives.any():
                hits.append(float(bool(positives[j])))
        top1 = float(np.mean(hits)) if hits else 0.0
    return float(loss.item()), top1


def _eval_route_state(model: RouteStateTransformer, dataset: RouteStateDataset, idx: list[int]) -> dict[str, float]:
    if not idx:
        return {
            "val_loss": 0.0,
            "val_solved_acc": 0.0,
            "val_value_mae": 0.0,
            "val_value_ece": 0.0,
        }
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(dataset.step_tokens[idx], device=device),
            torch.tensor(dataset.step_mask[idx], device=device).bool(),
            torch.tensor(dataset.route_features[idx], device=device),
        )
        value = torch.tensor(dataset.value[idx], device=device)
        solved = torch.tensor(dataset.solved[idx], device=device)
        loss = _route_loss(
            out,
            value,
            solved,
            torch.tensor(dataset.stock_closed[idx], device=device),
            torch.tensor(dataset.progressive[idx], device=device),
            torch.tensor(dataset.compatibility[idx], device=device),
            torch.tensor(dataset.bottlenecks[idx], device=device),
        )
        pred = (torch.sigmoid(out["solved_logit"]) >= 0.5).float()
        acc = float((pred == solved).float().mean().item())
        value_prob = torch.sigmoid(out["value_logit"])
        value_mae = float(torch.abs(value_prob - value).mean().item())
        solved_prob = torch.sigmoid(out["solved_logit"])
        stock_prob = torch.sigmoid(out["stock_logit"])
        progressive_prob = torch.sigmoid(out["progressive_logit"])
        compatibility_prob = torch.sigmoid(out["compatibility_logit"])
        bottleneck_prob = torch.sigmoid(out["bottleneck_logits"])
    return {
        "val_loss": float(loss.item()),
        "val_solved_acc": acc,
        "val_value_mae": value_mae,
        "val_value_ece": _binary_ece(value.cpu().numpy(), value_prob.cpu().numpy()),
        "val_solved_auroc": _binary_auc(dataset.solved[idx], solved_prob.cpu().numpy()),
        "val_stock_auroc": _binary_auc(dataset.stock_closed[idx], stock_prob.cpu().numpy()),
        "val_progressive_auroc": _binary_auc(dataset.progressive[idx], progressive_prob.cpu().numpy()),
        "val_compatibility_auroc": _binary_auc(dataset.compatibility[idx], compatibility_prob.cpu().numpy()),
        "val_bottleneck_macro_f1": _macro_f1(dataset.bottlenecks[idx], bottleneck_prob.cpu().numpy()),
    }


def _eval_search_policy(model: SearchPolicyNetwork, dataset: SearchPolicyDataset, idx: list[int]) -> dict[str, float]:
    if not idx:
        return {
            "val_loss": 0.0,
            "val_node_loss": 0.0,
            "val_action_loss": 0.0,
            "val_action_rank_loss": 0.0,
            "val_route_loss": 0.0,
            "val_budget_loss": 0.0,
            "val_node_top1": 0.0,
            "val_node_top3": 0.0,
            "val_top1_positive": 0.0,
            "val_top5_positive": 0.0,
            "val_value_mae": 0.0,
            "val_value_ece": 0.0,
            "val_top1_stock_closing_positive": 0.0,
            "val_top1_useful_candidate_positive": 0.0,
            "val_stock_dead_end_auc": None,
            "val_node_calibration_by_remaining_depth": {},
        }
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(dataset.step_tokens[idx], device=device),
            torch.tensor(dataset.step_mask[idx], device=device).bool(),
            torch.tensor(dataset.route_features[idx], device=device),
            torch.tensor(dataset.action_features[idx], device=device),
            torch.tensor(dataset.action_mask[idx], device=device).bool(),
            torch.tensor(dataset.node_features[idx], device=device),
            torch.tensor(dataset.node_mask[idx], device=device).bool(),
        )
        node_labels = torch.tensor(dataset.node_labels[idx], device=device)
        node_mask = torch.tensor(dataset.node_mask[idx], device=device)
        node_weights = torch.tensor(dataset.node_weights[idx], device=device)
        labels = torch.tensor(dataset.action_labels[idx], device=device)
        action_weights = torch.tensor(dataset.action_weights[idx], device=device)
        mask = torch.tensor(dataset.action_mask[idx], device=device)
        value = torch.tensor(dataset.value[idx], device=device)
        source_budget = torch.tensor(dataset.source_budget[idx], device=device)
        node_loss = _masked_bce(out["node_policy_logits"], node_labels, node_mask, node_weights)
        action_loss = _masked_bce(out["action_logits"], labels, mask, action_weights)
        action_rank_loss = _action_pairwise_ranking_loss(out["action_logits"], labels, mask, action_weights)
        route_loss = _route_loss(
            out,
            value,
            torch.tensor(dataset.solved[idx], device=device),
            torch.tensor(dataset.stock_closed[idx], device=device),
            torch.tensor(dataset.progressive[idx], device=device),
            torch.tensor(dataset.compatibility[idx], device=device),
            torch.tensor(dataset.bottlenecks[idx], device=device),
        )
        budget_loss = nn.functional.smooth_l1_loss(torch.sigmoid(out["budget_logits"]), source_budget)
        loss = node_loss + action_loss + route_loss + 0.2 * budget_loss
        top = torch.argmax(out["action_logits"], dim=1)
        top5 = torch.topk(out["action_logits"], k=min(5, out["action_logits"].shape[1]), dim=1).indices
        node_top = torch.argmax(out["node_policy_logits"], dim=1)
        node_top3 = torch.topk(
            out["node_policy_logits"],
            k=min(3, out["node_policy_logits"].shape[1]),
            dim=1,
        ).indices
        hits = []
        hits5 = []
        node_hits = []
        node_hits3 = []
        for i, j in enumerate(top.tolist()):
            positives = (labels[i] >= 0.75) & (mask[i] > 0)
            if positives.any():
                hits.append(float(bool(positives[j])))
                hits5.append(float(any(bool(positives[k]) for k in top5[i].tolist())))
        for i, j in enumerate(node_top.tolist()):
            positives = (node_labels[i] >= 0.5) & (node_mask[i] > 0) & (node_weights[i] > 0)
            if positives.any():
                node_hits.append(float(bool(positives[j])))
                node_hits3.append(float(any(bool(positives[k]) for k in node_top3[i].tolist())))
        value_prob = torch.sigmoid(out["value_logit"])
        value_mae = float(torch.abs(value_prob - value).mean().item())
        solved_prob = torch.sigmoid(out["solved_logit"])
        stock_prob = torch.sigmoid(out["stock_logit"])
        progressive_prob = torch.sigmoid(out["progressive_logit"])
        compatibility_prob = torch.sigmoid(out["compatibility_logit"])
        bottleneck_prob = torch.sigmoid(out["bottleneck_logits"])
        trace_node_metrics = _eval_trace_node_metrics(dataset, idx, out["node_policy_logits"].detach().cpu().numpy())
    return {
        "val_loss": float(loss.item()),
        "val_node_loss": float(node_loss.item()),
        "val_action_loss": float(action_loss.item()),
        "val_action_rank_loss": float(action_rank_loss.item()),
        "val_route_loss": float(route_loss.item()),
        "val_budget_loss": float(budget_loss.item()),
        "val_node_top1": float(np.mean(node_hits)) if node_hits else 0.0,
        "val_node_top3": float(np.mean(node_hits3)) if node_hits3 else 0.0,
        "val_top1_positive": float(np.mean(hits)) if hits else 0.0,
        "val_top5_positive": float(np.mean(hits5)) if hits5 else 0.0,
        "val_value_mae": value_mae,
        "val_value_ece": _binary_ece(value.cpu().numpy(), value_prob.cpu().numpy()),
        "val_solved_auroc": _binary_auc(dataset.solved[idx], solved_prob.cpu().numpy()),
        "val_stock_auroc": _binary_auc(dataset.stock_closed[idx], stock_prob.cpu().numpy()),
        "val_progressive_auroc": _binary_auc(dataset.progressive[idx], progressive_prob.cpu().numpy()),
        "val_compatibility_auroc": _binary_auc(dataset.compatibility[idx], compatibility_prob.cpu().numpy()),
        "val_bottleneck_macro_f1": _macro_f1(dataset.bottlenecks[idx], bottleneck_prob.cpu().numpy()),
        "val_budget_mae": float(torch.abs(torch.sigmoid(out["budget_logits"]) - source_budget).mean().item()),
        **trace_node_metrics,
    }


def _eval_trace_node_metrics(
    dataset: SearchPolicyDataset,
    idx: list[int],
    node_logits: np.ndarray,
) -> dict[str, Any]:
    if not idx:
        return {
            "val_top1_stock_closing_positive": 0.0,
            "val_top1_useful_candidate_positive": 0.0,
            "val_stock_dead_end_auc": None,
            "val_node_calibration_by_remaining_depth": {},
        }
    stock_hits: list[float] = []
    useful_hits: list[float] = []
    dead_labels: list[float] = []
    dead_scores: list[float] = []
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for local_i, row_idx in enumerate(idx):
        row = dataset.rows[row_idx]
        event = row.get("_trace_event") or {}
        if not isinstance(event, dict):
            continue
        open_leaves = [str(smi) for smi in (event.get("open_leaves") or [row.get("expanded_leaf")]) if smi]
        if not open_leaves:
            continue
        valid_count = min(len(open_leaves), dataset.node_mask.shape[1], node_logits.shape[1])
        if valid_count <= 0:
            continue
        logits = np.asarray(node_logits[local_i, :valid_count], dtype=np.float32)
        top_idx = int(np.argmax(logits))
        expanded_leaf = _canonical_leaf(event.get("expanded_leaf") or row.get("expanded_leaf"))
        expanded_idx = None
        for leaf_idx, leaf in enumerate(open_leaves[:valid_count]):
            if _canonical_leaf(leaf) == expanded_leaf:
                expanded_idx = leaf_idx
                break
        if expanded_idx is None:
            continue
        if bool(event.get("selected_next_stock_closed")):
            stock_hits.append(float(top_idx == expanded_idx))
        if event.get("selected_action_key"):
            useful_hits.append(float(top_idx == expanded_idx))
        dead = _trace_stock_dead_end_label(row.get("_trace_row") or row, event)
        dead_labels.append(float(dead))
        dead_scores.append(float(1.0 / (1.0 + math.exp(-float(logits[expanded_idx])))))
        depth = int(event.get("depth") or 0)
        remaining = max(0, int((dataset.feature_schema or {}).get("max_steps") or 8) - depth)
        bucket = "late" if remaining <= 1 else "mid" if remaining <= 3 else "early"
        label = float(dataset.node_labels[row_idx, expanded_idx])
        prob = float(1.0 / (1.0 + math.exp(-float(logits[expanded_idx]))))
        buckets[bucket].append((prob, label))
    return {
        "val_top1_stock_closing_positive": float(np.mean(stock_hits)) if stock_hits else 0.0,
        "val_top1_useful_candidate_positive": float(np.mean(useful_hits)) if useful_hits else 0.0,
        "val_stock_dead_end_auc": _binary_auc(np.asarray(dead_labels, dtype=np.float32), np.asarray(dead_scores, dtype=np.float32)),
        "val_node_calibration_by_remaining_depth": _node_calibration_buckets(buckets),
    }


def _trace_stock_dead_end_label(row: dict[str, Any], event: dict[str, Any]) -> bool:
    outcome = event.get("outcome") or {}
    metrics = row.get("route_metrics") or []
    stock_closed = bool(event.get("selected_next_stock_closed")) or any(bool((item or {}).get("strict_stock_solve")) for item in metrics)
    diagnostics = _diagnostics_for_leaf(event, event.get("expanded_leaf"))
    final_actions = sum(int(item.get("final_actions") or 0) for item in diagnostics)
    return bool(
        not stock_closed
        and (
            _expanded_leaf_low_yield(event)
            or int(outcome.get("dead_ends") or 0) > 0
            or (final_actions <= 0 and not event.get("selected_action_key"))
        )
    )


def _node_calibration_buckets(buckets: dict[str, list[tuple[float, float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in ("early", "mid", "late"):
        values = buckets.get(name) or []
        if not values:
            out[name] = {"n": 0.0, "mean_pred": 0.0, "mean_label": 0.0, "mae": 0.0}
            continue
        pred = np.asarray([item[0] for item in values], dtype=np.float32)
        labels = np.asarray([item[1] for item in values], dtype=np.float32)
        out[name] = {
            "n": float(len(values)),
            "mean_pred": float(pred.mean()),
            "mean_label": float(labels.mean()),
            "mae": float(np.abs(pred - labels).mean()),
        }
    return out


def _masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weighted = loss * mask.float() * weights.float()
    denom = (mask.float() * weights.float()).sum().clamp_min(1.0)
    return weighted.sum() / denom


def _action_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    positive_threshold: float = 0.5,
    margin: float = 0.0,
) -> torch.Tensor:
    valid = mask.bool()
    positives = valid & (labels >= positive_threshold)
    negatives = valid & (labels < positive_threshold)
    losses: list[torch.Tensor] = []
    for row_idx in range(logits.shape[0]):
        pos_logits = logits[row_idx][positives[row_idx]]
        neg_logits = logits[row_idx][negatives[row_idx]]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            continue
        pos_weights = weights[row_idx][positives[row_idx]].clamp_min(0.05)
        neg_weights = (1.0 - labels[row_idx][negatives[row_idx]]).clamp_min(0.05)
        diffs = pos_logits[:, None] - neg_logits[None, :]
        pair_loss = nn.functional.softplus(-(diffs - float(margin)))
        pair_weights = pos_weights[:, None] * neg_weights[None, :]
        losses.append((pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1.0))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _normalize_search_policy_train_mode(mode: str | None) -> str:
    value = str(mode or "all").strip().lower().replace("-", "_")
    aliases = {
        "all": "all",
        "full": "all",
        "joint": "all",
        "action": "action_only",
        "actions": "action_only",
        "action_only": "action_only",
    }
    if value not in aliases:
        raise ValueError(f"unsupported policy train mode: {mode}")
    return aliases[value]


def _configure_search_policy_train_mode(model: SearchPolicyNetwork, mode: str | None) -> dict[str, Any]:
    mode = _normalize_search_policy_train_mode(mode)
    for param in model.parameters():
        param.requires_grad = True
    if mode == "action_only":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith("action_proj.") or name.startswith("action_score."):
                param.requires_grad = True
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    total_params = sum(int(param.numel()) for param in model.parameters())
    if trainable_params <= 0:
        raise ValueError(f"policy train mode {mode} leaves no trainable parameters")
    return {
        "mode": mode,
        "trainable_parameter_count": int(trainable_params),
        "total_parameter_count": int(total_params),
        "trainable_parameter_fraction": round(float(trainable_params) / max(float(total_params), 1.0), 6),
        "trainable_prefixes": sorted({name.split(".", 1)[0] for name in trainable_names}),
    }


def _search_policy_training_loss(
    mode: str | None,
    *,
    node_loss: torch.Tensor,
    action_loss: torch.Tensor,
    route_loss: torch.Tensor,
    budget_loss: torch.Tensor,
    action_rank_loss: torch.Tensor | None = None,
    action_rank_loss_weight: float = 0.0,
) -> torch.Tensor:
    mode = _normalize_search_policy_train_mode(mode)
    rank_term = float(action_rank_loss_weight or 0.0) * (
        action_rank_loss if action_rank_loss is not None else action_loss.sum() * 0.0
    )
    if mode == "action_only":
        return action_loss + rank_term
    return node_loss + action_loss + rank_term + route_loss + 0.2 * budget_loss


def _search_policy_selection_metric(
    mode: str | None,
    metrics: dict[str, Any],
    *,
    action_rank_loss_weight: float = 0.0,
) -> float:
    mode = _normalize_search_policy_train_mode(mode)
    if mode == "action_only" and float(action_rank_loss_weight or 0.0) > 0.0:
        return float(metrics.get("val_action_loss", 0.0)) + float(action_rank_loss_weight) * float(
            metrics.get("val_action_rank_loss", 0.0)
        )
    name = _search_policy_selection_metric_name(mode, action_rank_loss_weight=action_rank_loss_weight)
    value = metrics.get(name)
    if value is None:
        value = metrics.get("val_loss", 0.0)
    return float(value)


def _search_policy_selection_metric_name(mode: str | None, *, action_rank_loss_weight: float = 0.0) -> str:
    mode = _normalize_search_policy_train_mode(mode)
    if mode == "action_only" and float(action_rank_loss_weight or 0.0) > 0.0:
        return f"val_action_loss+{float(action_rank_loss_weight):g}*val_action_rank_loss"
    if mode == "action_only":
        return "val_action_loss"
    return "val_loss"


def _rounded_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            out[key] = None
        elif isinstance(value, dict):
            out[key] = _rounded_metrics(value)
        else:
            out[key] = round(float(value), 6)
    return out


def _value_calibration_metadata(history: list[dict[str, Any]]) -> dict[str, Any]:
    latest = history[-1] if history else {}
    return {
        "calibrated": False,
        "method": "validation_ece_only",
        "value_target": "solved_stock_progressive_compatibility_utility",
        "ece": latest.get("val_value_ece"),
        "note": "Runtime may score actions with this checkpoint, but value backup is disabled until a frozen calibration pass marks calibrated=true.",
    }


def _binary_ece(labels: np.ndarray, probs: np.ndarray, *, n_bins: int = 10) -> float:
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return 0.0
    y = (y >= 0.5).astype(np.float32)
    p = np.clip(p, 0.0, 1.0)
    ece = 0.0
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        if idx == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _binary_auc(labels: np.ndarray, probs: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    y = (y >= 0.5).astype(np.int32)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos_rank_sum = float(ranks[y == 1].sum())
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / max(pos * neg, 1)
    return float(auc)


def _macro_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    y = (np.asarray(labels, dtype=np.float32) >= 0.5).astype(np.int32)
    p = (np.asarray(probs, dtype=np.float32) >= 0.5).astype(np.int32)
    if y.ndim == 1:
        y = y[:, None]
        p = p[:, None]
    scores = []
    for col in range(y.shape[1]):
        yt = y[:, col]
        pt = p[:, col]
        tp = int(((yt == 1) & (pt == 1)).sum())
        fp = int(((yt == 0) & (pt == 1)).sum())
        fn = int(((yt == 1) & (pt == 0)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2.0 * precision * recall / max(precision + recall, 1e-8))
    return float(np.mean(scores)) if scores else 0.0


def _load_initial_state(model: nn.Module, checkpoint: Path | str | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state is None:
        raise ValueError(f"checkpoint has no state_dict: {checkpoint}")
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    skipped_shape_mismatch = [
        key
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape)
    ]
    result = model.load_state_dict(compatible, strict=False)
    return {
        "path": str(checkpoint),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def _route_loss(
    out: dict[str, torch.Tensor],
    value: torch.Tensor,
    solved: torch.Tensor,
    stock: torch.Tensor,
    progressive: torch.Tensor,
    compatibility: torch.Tensor,
    bottlenecks: torch.Tensor,
) -> torch.Tensor:
    return (
        0.7 * nn.functional.binary_cross_entropy_with_logits(out["value_logit"], value)
        + 0.5 * nn.functional.binary_cross_entropy_with_logits(out["solved_logit"], solved)
        + 0.5 * nn.functional.binary_cross_entropy_with_logits(out["stock_logit"], stock)
        + 0.4 * nn.functional.binary_cross_entropy_with_logits(out["progressive_logit"], progressive)
        + 0.4 * nn.functional.binary_cross_entropy_with_logits(out["compatibility_logit"], compatibility)
        + 0.5 * nn.functional.binary_cross_entropy_with_logits(out["bottleneck_logits"], bottlenecks)
    )


def split_by_target(rows: list[dict[str, Any]], val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    targets = sorted({canonical_smiles(row.get("target_smiles") or row.get("product") or "") for row in rows})
    if len(targets) >= 2:
        n_val = max(1, int(round(len(targets) * val_fraction)))
        stride = max(1, len(targets) // n_val)
        val_targets = set(targets[::stride][:n_val])
        train_idx = [idx for idx, row in enumerate(rows) if canonical_smiles(row.get("target_smiles") or row.get("product") or "") not in val_targets]
        val_idx = [idx for idx, row in enumerate(rows) if idx not in train_idx]
    else:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    if not train_idx or not val_idx:
        pivot = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx = list(range(pivot))
        val_idx = list(range(pivot, len(rows)))
    return train_idx, val_idx


def _target_count(rows: list[dict[str, Any]], indices: list[int] | None = None) -> int:
    selected = rows if indices is None else [rows[idx] for idx in indices]
    targets = {
        canonical_smiles(row.get("target_smiles") or row.get("product") or "")
        for row in selected
    }
    return len({target for target in targets if target})


def _select_device(device: str) -> torch.device:
    requested = str(device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for vNext training but torch.cuda.is_available() is false")
    return torch.device(requested)


def _step_metadata(row: dict[str, Any]) -> np.ndarray:
    cand = row.get("candidate") or {}
    evidence = cand.get("evidence") or {}
    t_value = safe_float(cand.get("T"))
    ph_value = safe_float(cand.get("pH"))
    source_bucket = stable_bucket(str(row.get("source") or cand.get("source") or "unknown"), 8)
    typ_bucket = stable_bucket(str(row.get("reaction_type") or cand.get("type") or ""), 8)
    ec1 = _ec1_id(row.get("ec") or cand.get("ec"))
    values = [
        float(row.get("rank") or 1.0) / 32.0,
        1.0 / max(float(row.get("rank") or 1.0), 1.0),
        float(bool(row.get("gt_available"))),
        float(bool(row.get("exact_gt_reaction"))),
        float(bool(row.get("exact_gt_reactants"))),
        float(bool(row.get("selected_exact"))),
        float(ec1) / 7.0,
        float(t_value is not None),
        float(ph_value is not None),
        float(t_value or 0.0) / 100.0,
        float(ph_value or 0.0) / 14.0,
        float(bool(cand.get("ec"))),
        float(bool(cand.get("type") or cand.get("reaction_type"))),
        float(bool(cand.get("enzyme_uid"))),
        float(bool(cand.get("doi") or evidence.get("doi"))),
        float(bool(cand.get("uniprot_accession") or evidence.get("uniprot_accession"))),
    ]
    source_onehot = [1.0 if i == source_bucket else 0.0 for i in range(8)]
    type_onehot = [1.0 if i == typ_bucket else 0.0 for i in range(8)]
    return np.asarray(values + source_onehot + type_onehot, dtype=np.float32)


def _ec1_id(value: Any) -> int:
    try:
        ec1 = int(str(value).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0
    return ec1 if 1 <= ec1 <= 7 else 0


def _save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any], feature_schema: dict[str, Any], model_config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "feature_schema": feature_schema,
        "model_config": model_config,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, path)


def _write_reports(report: dict[str, Any], report_output: Path, md_output: Path | None, *, title: str) -> None:
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_output:
        meta = report.get("metadata") or {}
        source = meta.get("vnext_pack") or meta.get("route_tree_traces") or meta.get("training_source")
        lines = [
            f"# {title}",
            "",
            f"Pack: `{source}`",
            f"Model: `{meta.get('model_output')}`",
            f"Rows: `{meta.get('n_rows')}`",
            f"Train/val: `{meta.get('n_train')}` / `{meta.get('n_val')}`",
            f"Best val loss: `{report.get('best_val_loss')}`",
            "",
            "## History",
            "",
            "```json",
            json.dumps(report.get("history") or [], indent=2),
            "```",
            "",
            "These models are optional vNext search priors; validators and frozen one-step engines remain authoritative.",
            "",
        ]
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train vNext AutoPlanner models from current or vNext training packs")
    ap.add_argument("--pack-dir", default=None, help="Existing consolidated training pack; converted to vNext if needed")
    ap.add_argument("--vnext-pack-dir", default=None, help="Existing or output vNext pack directory")
    ap.add_argument("--task", default="all", choices=["all", "step", "candidate_pool", "route_state", "policy"])
    ap.add_argument("--output-dir", default="results/shared/vnext")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bits", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--max-candidates", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument("--feature-cache-dir", default=None, help="Optional directory for persistent vNext feature matrices")
    ap.add_argument("--feature-workers", type=int, default=1, help="Processes for RDKit candidate/action feature construction")
    ap.add_argument("--init-checkpoint", default=None, help="Optional compatible checkpoint to initialize policy training")
    ap.add_argument(
        "--node-label-target",
        default="trace_leaf_utility",
        choices=["trace_leaf_utility", "stock_aware_leaf_utility"],
        help="Route-tree open-leaf label target when --route-tree-traces is used",
    )
    ap.add_argument(
        "--action-label-target",
        default="selected_solved_action",
        choices=["selected_solved_action", "stock_aware_action_utility", "stock_counterfactual_action_utility"],
        help="Route-tree action label target when --route-tree-traces is used",
    )
    ap.add_argument(
        "--policy-train-mode",
        default="all",
        choices=["all", "action_only"],
        help="Policy heads to train. action_only freezes the route encoder, node head, value heads, and budget head.",
    )
    ap.add_argument(
        "--action-rank-loss-weight",
        type=float,
        default=0.0,
        help="Optional pairwise action ranking loss weight. Useful for action_only stock-aware fine-tuning.",
    )
    ap.add_argument(
        "--route-tree-traces",
        nargs="*",
        default=None,
        help="Route-tree trace JSONL files for training the unified search policy/value controller",
    )
    args = ap.parse_args()

    use_route_tree_policy_data = bool(args.route_tree_traces) and args.task in {"all", "policy"}
    needs_vnext_pack = args.task != "policy" or not use_route_tree_policy_data
    vnext_pack = None
    if needs_vnext_pack:
        vnext_pack = prepare_vnext_pack(
            Path(args.pack_dir) if args.pack_dir else None,
            Path(args.vnext_pack_dir) if args.vnext_pack_dir else None,
            max_candidates=args.max_candidates,
        )
    out = Path(args.output_dir)
    reports: dict[str, Any] = {}
    if args.task in {"all", "step"}:
        reports["step"] = train_step_encoder_from_vnext_pack(
            vnext_pack_dir=vnext_pack,
            model_output=out / "step_encoder.pt",
            report_output=out / "step_encoder.json",
            md_output=out / "step_encoder.md",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_bits=args.n_bits,
            d_model=args.d_model,
            device=args.device,
        )
    if args.task in {"all", "candidate_pool"}:
        reports["candidate_pool"] = train_candidate_pool_ranker_from_vnext_pack(
            vnext_pack_dir=vnext_pack,
            model_output=out / "candidate_pool_ranker.pt",
            report_output=out / "candidate_pool_ranker.json",
            md_output=out / "candidate_pool_ranker.md",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_bits=args.n_bits,
            max_candidates=args.max_candidates,
            d_model=args.d_model,
            device=args.device,
            feature_cache_dir=Path(args.feature_cache_dir) if args.feature_cache_dir else None,
            feature_workers=args.feature_workers,
        )
    if args.task in {"all", "route_state"}:
        reports["route_state"] = train_route_state_from_vnext_pack(
            vnext_pack_dir=vnext_pack,
            model_output=out / "route_state_transformer.pt",
            report_output=out / "route_state_transformer.json",
            md_output=out / "route_state_transformer.md",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=args.max_steps,
            d_model=args.d_model,
            device=args.device,
        )
    if args.task in {"all", "policy"}:
        if use_route_tree_policy_data:
            reports["policy"] = train_search_policy_from_route_tree_traces(
                trace_paths=[Path(path) for path in args.route_tree_traces],
                model_output=out / "search_policy.pt",
                report_output=out / "search_policy.json",
                md_output=out / "search_policy.md",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                n_bits=args.n_bits,
                max_candidates=args.max_candidates,
                max_steps=args.max_steps,
                d_model=args.d_model,
                node_label_target=args.node_label_target,
                action_label_target=args.action_label_target,
                device=args.device,
                feature_cache_dir=Path(args.feature_cache_dir) if args.feature_cache_dir else None,
                feature_workers=args.feature_workers,
                init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
                policy_train_mode=args.policy_train_mode,
                action_rank_loss_weight=args.action_rank_loss_weight,
            )
        else:
            reports["policy"] = train_search_policy_from_vnext_pack(
                vnext_pack_dir=vnext_pack,
                model_output=out / "search_policy.pt",
                report_output=out / "search_policy.json",
                md_output=out / "search_policy.md",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                n_bits=args.n_bits,
                max_candidates=args.max_candidates,
                max_steps=args.max_steps,
                d_model=args.d_model,
                device=args.device,
                feature_cache_dir=Path(args.feature_cache_dir) if args.feature_cache_dir else None,
                feature_workers=args.feature_workers,
                init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
                policy_train_mode=args.policy_train_mode,
                action_rank_loss_weight=args.action_rank_loss_weight,
            )
    print(json.dumps({
        "vnext_pack": str(vnext_pack) if vnext_pack else None,
        "tasks": {name: report.get("best_val_loss") for name, report in reports.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
