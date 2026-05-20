"""Audit whether ChemEnzy route pools recover held-out v4 cascade blocks."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.v4_product_value import (
    canonical_smiles,
    route_record_from_native_route,
)
from cascade_planner.cascadeboard.route_recovery import canonical_reaction
from cascade_planner.eval.replay_block_coherence_on_route_pool import _main_reactant


SCHEMA_VERSION = "v4_heldout_block_recovery_audit.v1"


def audit_v4_heldout_block_recovery(
    *,
    route_pool: Path,
    program_manifest: Path,
    split: str,
    output_json: Path,
    analog_similarity: float = 0.55,
    top_ks: tuple[int, ...] = (1, 3, 5, 10, 50),
) -> dict[str, Any]:
    started = time.monotonic()
    refs = _load_references(program_manifest, split=split)
    routes = _load_routes(route_pool)
    routes_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        target = canonical_smiles(str(route.get("target_smiles") or "")) or str(route.get("target_smiles") or "")
        if target:
            routes_by_target[target].append(route)
    for target_routes in routes_by_target.values():
        target_routes.sort(key=_route_sort_key)

    target_rows = []
    for target in sorted(refs["by_target"]):
        target_refs = refs["by_target"][target]
        target_routes = routes_by_target.get(target, [])
        target_rows.append(_audit_target(target, target_refs, target_routes, analog_similarity=analog_similarity, top_ks=top_ks))

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool": str(route_pool),
            "program_manifest": str(program_manifest),
            "split": split,
            "analog_similarity": analog_similarity,
            "top_ks": list(top_ks),
            "elapsed_s": round(time.monotonic() - started, 3),
            "reference_contract": "held-out v4 split only; train evidence is not used as a recovery label",
        },
        "reference_summary": refs["summary"],
        "route_pool_summary": _route_pool_summary(routes, routes_by_target),
        "summary": _summary(target_rows, top_ks=top_ks),
        "targets": target_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _load_references(program_manifest: Path, *, split: str) -> dict[str, Any]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    split_path = Path((manifest.get("outputs") or {})[split])
    programs = _read_jsonl(split_path)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_count = 0
    transform_pairs = Counter()
    for program in programs:
        target = canonical_smiles(str(program.get("target_smiles") or "")) or str(program.get("target_smiles") or "")
        if not target:
            continue
        blocks = _reference_blocks(program)
        if not blocks:
            continue
        block_count += len(blocks)
        transform_pairs.update(block.get("transform_pair") for block in blocks)
        by_target[target].append(
            {
                "program_id": program.get("program_id"),
                "doi": program.get("doi"),
                "cascade_id": program.get("cascade_id"),
                "cascade_type": program.get("cascade_type") or program.get("route_domain"),
                "quality_tier": program.get("quality_tier"),
                "target_smiles": target,
                "total_steps": len(_program_steps(program)),
                "blocks": blocks,
            }
        )
    return {
        "by_target": dict(by_target),
        "summary": {
            "split": split,
            "programs_with_blocks": sum(len(rows) for rows in by_target.values()),
            "unique_targets_with_blocks": len(by_target),
            "reference_blocks": block_count,
            "top_transform_pairs": dict(transform_pairs.most_common(20)),
            "source_split_path": str(split_path),
        },
    }


def _reference_blocks(program: dict[str, Any]) -> list[dict[str, Any]]:
    steps = _program_steps(program)
    blocks = []
    for idx, (upstream, downstream) in enumerate(zip(steps, steps[1:])):
        upstream_product = _step_product(upstream)
        downstream_product = _step_product(downstream)
        upstream_main = _step_main_reactant(upstream)
        downstream_main = _step_main_reactant(downstream, preferred=upstream_product)
        blocks.append(
            {
                "block_id": f"{program.get('program_id')}::{idx}",
                "program_id": program.get("program_id"),
                "doi": program.get("doi"),
                "cascade_id": program.get("cascade_id"),
                "upstream_transition_id": upstream.get("transition_id"),
                "downstream_transition_id": downstream.get("transition_id"),
                "upstream_rxn": canonical_reaction(upstream.get("rxn_smiles")) or str(upstream.get("rxn_smiles") or ""),
                "downstream_rxn": canonical_reaction(downstream.get("rxn_smiles")) or str(downstream.get("rxn_smiles") or ""),
                "upstream_product": upstream_product,
                "downstream_product": downstream_product,
                "upstream_main_reactant": upstream_main,
                "downstream_main_reactant": downstream_main,
                "upstream_transform": _norm_transform(upstream.get("transformation_superclass")),
                "downstream_transform": _norm_transform(downstream.get("transformation_superclass")),
                "transform_pair": f"{_norm_transform(upstream.get('transformation_superclass'))}->{_norm_transform(downstream.get('transformation_superclass'))}",
                "upstream_fp": _transition_fp(upstream_product, upstream_main),
                "downstream_fp": _transition_fp(downstream_product, downstream_main),
            }
        )
    return blocks


def _program_steps(program: dict[str, Any]) -> list[dict[str, Any]]:
    steps = program.get("steps")
    if isinstance(steps, list) and steps:
        return [step for step in steps if isinstance(step, dict)]
    gt_route = program.get("gt_route")
    if isinstance(gt_route, list):
        return [step for step in gt_route if isinstance(step, dict)]
    return []


def _step_product(step: dict[str, Any]) -> str:
    value = step.get("product_smiles")
    if value:
        return canonical_smiles(str(value)) or str(value)
    return _largest_component(_reaction_products(step.get("rxn_smiles") or step.get("reaction_smiles")))


def _step_main_reactant(step: dict[str, Any], *, preferred: str = "") -> str:
    value = step.get("main_reactant")
    if value:
        return canonical_smiles(str(value)) or str(value)
    reactants = [canonical_smiles(str(part)) or str(part) for part in (step.get("reactants") or []) if part]
    if not reactants:
        reactants = [canonical_smiles(str(part)) or str(part) for part in _reaction_reactants(step.get("rxn_smiles") or step.get("reaction_smiles")) if part]
    preferred = canonical_smiles(str(preferred or "")) or str(preferred or "")
    if preferred and preferred in set(reactants):
        return preferred
    return _largest_component(reactants)


def _reaction_products(rxn_smiles: Any) -> list[str]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return []
    _, right = text.split(">>", 1)
    return [part.strip() for part in right.split(".") if part.strip()]


def _largest_component(parts: list[str]) -> str:
    best = ""
    best_key = (-1, -1)
    for part in parts:
        smi = canonical_smiles(str(part)) or str(part)
        mol = Chem.MolFromSmiles(smi)
        heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
        key = (heavy, len(smi))
        if key > best_key:
            best_key = key
            best = smi
    return best


def _load_routes(route_pool: Path) -> list[dict[str, Any]]:
    if route_pool.suffix == ".jsonl":
        return _read_jsonl(route_pool)
    run = json.loads(route_pool.read_text(encoding="utf-8"))
    if not isinstance(run, dict):
        raise ValueError(f"unsupported route pool format: {route_pool}")
    rows = []
    for target_index, target in enumerate(run.get("targets") or []):
        if not isinstance(target, dict):
            continue
        target_smiles = str(target.get("target_smiles") or "")
        target_id = str(target.get("target_id") or target.get("cascade_id") or target.get("index") or target_index)
        for native_rank, route in enumerate(target.get("routes") or []):
            if isinstance(route, dict):
                rows.append(
                    route_record_from_native_route(
                        route,
                        target_smiles=target_smiles,
                        target_id=target_id,
                        native_rank=_native_rank(route, fallback=native_rank),
                        dataset=str((run.get("metadata") or {}).get("schema_version") or "native_route_pool"),
                    )
                )
    return rows


def _audit_target(
    target: str,
    target_refs: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    *,
    analog_similarity: float,
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    ref_blocks = [block for program in target_refs for block in program.get("blocks") or []]
    route_hits = []
    for route in routes:
        hits = _route_hits(route, ref_blocks, analog_similarity=analog_similarity)
        if hits["exact_blocks"] or hits["analog_blocks"]:
            route_hits.append(hits)
    exact_ranks = [row["native_rank"] for row in route_hits if row["exact_blocks"]]
    analog_ranks = [row["native_rank"] for row in route_hits if row["analog_blocks"]]
    strict_analog_ranks = [row["native_rank"] for row in route_hits if row["transform_consistent_analog_blocks"]]
    return {
        "target_smiles": target,
        "program_count": len(target_refs),
        "reference_block_count": len(ref_blocks),
        "route_count": len(routes),
        "multistep_route_count": sum(1 for route in routes if len(route.get("steps") or []) >= 2),
        "best_exact_native_rank": min(exact_ranks) if exact_ranks else None,
        "best_analog_native_rank": min(analog_ranks) if analog_ranks else None,
        "best_transform_consistent_analog_native_rank": min(strict_analog_ranks) if strict_analog_ranks else None,
        "topk": {
            str(k): {
                "exact": any(row["native_rank"] < k and row["exact_blocks"] for row in route_hits),
                "analog": any(row["native_rank"] < k and row["analog_blocks"] for row in route_hits),
                "transform_consistent_analog": any(row["native_rank"] < k and row["transform_consistent_analog_blocks"] for row in route_hits),
            }
            for k in top_ks
        },
        "route_hits": sorted(route_hits, key=lambda row: (row["native_rank"], row["route_id"]))[:20],
        "reference_examples": [_compact_ref_block(block) for block in ref_blocks[:5]],
    }


def _route_hits(route: dict[str, Any], ref_blocks: list[dict[str, Any]], *, analog_similarity: float) -> dict[str, Any]:
    exact_blocks = []
    analog_blocks = []
    strict_analog_blocks = []
    route_blocks = _route_blocks(route)
    for route_block in route_blocks:
        for ref in ref_blocks:
            exact = (
                bool(route_block["upstream_rxn"])
                and bool(route_block["downstream_rxn"])
                and route_block["upstream_rxn"] == ref.get("upstream_rxn")
                and route_block["downstream_rxn"] == ref.get("downstream_rxn")
            )
            upstream_sim = _fp_similarity(route_block.get("upstream_fp"), ref.get("upstream_fp"))
            downstream_sim = _fp_similarity(route_block.get("downstream_fp"), ref.get("downstream_fp"))
            analog = upstream_sim >= analog_similarity and downstream_sim >= analog_similarity
            transform_consistent = _transform_match(route_block.get("upstream_transform"), ref.get("upstream_transform")) and _transform_match(
                route_block.get("downstream_transform"), ref.get("downstream_transform")
            )
            hit = {
                "route_block_index": route_block.get("route_block_index"),
                "reference_block_id": ref.get("block_id"),
                "reference_program_id": ref.get("program_id"),
                "reference_doi": ref.get("doi"),
                "upstream_similarity": round(float(upstream_sim), 6),
                "downstream_similarity": round(float(downstream_sim), 6),
                "route_transform_pair": route_block.get("transform_pair"),
                "reference_transform_pair": ref.get("transform_pair"),
                "transform_consistent": bool(transform_consistent),
            }
            if exact:
                exact_blocks.append(hit)
            if analog:
                analog_blocks.append(hit)
                if transform_consistent:
                    strict_analog_blocks.append(hit)
    return {
        "route_id": route.get("route_id"),
        "native_rank": _native_rank(route),
        "native_score": route.get("native_score"),
        "stock_closed": route.get("stock_closed"),
        "n_steps": len(route.get("steps") or []),
        "exact_blocks": exact_blocks[:10],
        "analog_blocks": sorted(analog_blocks, key=lambda row: -(row["upstream_similarity"] + row["downstream_similarity"]))[:10],
        "transform_consistent_analog_blocks": sorted(
            strict_analog_blocks,
            key=lambda row: -(row["upstream_similarity"] + row["downstream_similarity"]),
        )[:10],
    }


def _route_blocks(route: dict[str, Any]) -> list[dict[str, Any]]:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    out = []
    connected_pairs = _connected_step_pairs(steps)
    for idx, (downstream, upstream) in enumerate(connected_pairs):
        downstream_product = downstream.get("product_smiles") or ((downstream.get("products") or [""])[0] if isinstance(downstream.get("products"), list) else "")
        upstream_product = upstream.get("product_smiles") or ((upstream.get("products") or [""])[0] if isinstance(upstream.get("products"), list) else "")
        downstream_main = _main_reactant(downstream)
        upstream_main = _main_reactant(upstream)
        downstream_rxn = canonical_reaction(downstream.get("rxn_smiles")) or str(downstream.get("rxn_smiles") or "")
        upstream_rxn = canonical_reaction(upstream.get("rxn_smiles")) or str(upstream.get("rxn_smiles") or "")
        up_transform = _norm_transform(upstream.get("transformation_superclass"))
        down_transform = _norm_transform(downstream.get("transformation_superclass"))
        out.append(
            {
                "route_block_index": idx,
                "upstream_rxn": upstream_rxn,
                "downstream_rxn": downstream_rxn,
                "upstream_product": upstream_product,
                "downstream_product": downstream_product,
                "upstream_main_reactant": upstream_main,
                "downstream_main_reactant": downstream_main,
                "upstream_fp": _transition_fp(upstream_product, upstream_main),
                "downstream_fp": _transition_fp(downstream_product, downstream_main),
                "upstream_transform": up_transform,
                "downstream_transform": down_transform,
                "transform_pair": f"{up_transform}->{down_transform}",
            }
        )
    return out


def _connected_step_pairs(steps: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[int, int]] = set()
    product_by_index = {
        idx: canonical_smiles(str(step.get("product_smiles") or ((step.get("products") or [""])[0] if isinstance(step.get("products"), list) else "")))
        for idx, step in enumerate(steps)
    }
    for downstream_idx, downstream in enumerate(steps):
        reactants = {
            canonical_smiles(str(value))
            for value in downstream.get("reactants") or []
            if value
        }
        if not reactants:
            reactants = {
                canonical_smiles(str(value))
                for value in _reaction_reactants(downstream.get("rxn_smiles"))
                if value
            }
        for upstream_idx, upstream_product in product_by_index.items():
            if upstream_idx == downstream_idx or not upstream_product:
                continue
            if upstream_product in reactants:
                key = (downstream_idx, upstream_idx)
                if key not in seen:
                    seen.add(key)
                    pairs.append((downstream, steps[upstream_idx]))
    if not pairs:
        pairs = list(zip(steps, steps[1:]))
    pairs.sort(key=lambda pair: (_step_index(pair[0]), _step_index(pair[1])))
    return pairs


def _reaction_reactants(rxn_smiles: Any) -> list[str]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return []
    left, _ = text.split(">>", 1)
    return [part.strip() for part in left.split(".") if part.strip()]


def _step_index(step: dict[str, Any]) -> int:
    try:
        return int(step.get("step_index") or 0)
    except (TypeError, ValueError):
        return 0


def _summary(target_rows: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, Any]:
    with_routes = [row for row in target_rows if row.get("route_count")]
    out: dict[str, Any] = {
        "reference_targets": len(target_rows),
        "targets_with_routes": len(with_routes),
        "route_coverage_rate": round(len(with_routes) / max(len(target_rows), 1), 6),
        "reference_blocks": sum(int(row.get("reference_block_count") or 0) for row in target_rows),
        "routes": sum(int(row.get("route_count") or 0) for row in target_rows),
        "multistep_routes": sum(int(row.get("multistep_route_count") or 0) for row in target_rows),
        "targets_with_exact_any": sum(1 for row in target_rows if row.get("best_exact_native_rank") is not None),
        "targets_with_analog_any": sum(1 for row in target_rows if row.get("best_analog_native_rank") is not None),
        "targets_with_transform_consistent_analog_any": sum(
            1 for row in target_rows if row.get("best_transform_consistent_analog_native_rank") is not None
        ),
        "routed_targets_with_exact_any": sum(1 for row in with_routes if row.get("best_exact_native_rank") is not None),
        "routed_targets_with_analog_any": sum(1 for row in with_routes if row.get("best_analog_native_rank") is not None),
        "routed_targets_with_transform_consistent_analog_any": sum(
            1 for row in with_routes if row.get("best_transform_consistent_analog_native_rank") is not None
        ),
    }
    for k in top_ks:
        all_denom = max(len(target_rows), 1)
        routed_denom = max(len(with_routes), 1)
        exact_all = sum(1 for row in target_rows if (row.get("topk") or {}).get(str(k), {}).get("exact"))
        analog_all = sum(1 for row in target_rows if (row.get("topk") or {}).get(str(k), {}).get("analog"))
        strict_all = sum(1 for row in target_rows if (row.get("topk") or {}).get(str(k), {}).get("transform_consistent_analog"))
        exact_routed = sum(1 for row in with_routes if (row.get("topk") or {}).get(str(k), {}).get("exact"))
        analog_routed = sum(1 for row in with_routes if (row.get("topk") or {}).get(str(k), {}).get("analog"))
        strict_routed = sum(1 for row in with_routes if (row.get("topk") or {}).get(str(k), {}).get("transform_consistent_analog"))
        out[f"exact_recovery_at_{k}"] = round(exact_all / all_denom, 6)
        out[f"analog_recovery_at_{k}"] = round(analog_all / all_denom, 6)
        out[f"transform_consistent_analog_recovery_at_{k}"] = round(strict_all / all_denom, 6)
        out[f"routed_exact_recovery_at_{k}"] = round(exact_routed / routed_denom, 6) if with_routes else None
        out[f"routed_analog_recovery_at_{k}"] = round(analog_routed / routed_denom, 6) if with_routes else None
        out[f"routed_transform_consistent_analog_recovery_at_{k}"] = (
            round(strict_routed / routed_denom, 6) if with_routes else None
        )
    return out


def _route_pool_summary(routes: list[dict[str, Any]], routes_by_target: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    route_block_counts = [len(_route_blocks(route)) for route in routes]
    return {
        "routes": len(routes),
        "targets": len(routes_by_target),
        "multistep_routes": sum(1 for route in routes if len(route.get("steps") or []) >= 2),
        "steps": sum(len(route.get("steps") or []) for route in routes),
        "route_blocks": sum(route_block_counts),
        "avg_route_blocks": round(sum(route_block_counts) / max(len(route_block_counts), 1), 6),
        "transform_counts": dict(
            Counter(
                str(step.get("transformation_superclass") or "unknown")
                for route in routes
                for step in route.get("steps") or []
                if isinstance(step, dict)
            ).most_common(20)
        ),
    }


def _transition_fp(product_smiles: Any, main_reactant: Any):
    product_fp = _fp(product_smiles)
    main_fp = _fp(main_reactant)
    if product_fp is None and main_fp is None:
        return None
    if product_fp is None:
        return main_fp
    if main_fp is None:
        return product_fp
    arr_product = np.zeros((2048,), dtype=np.int8)
    arr_main = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(product_fp, arr_product)
    DataStructs.ConvertToNumpyArray(main_fp, arr_main)
    arr = np.maximum(arr_product, arr_main)
    fp = DataStructs.ExplicitBitVect(2048)
    for bit in np.nonzero(arr)[0]:
        fp.SetBit(int(bit))
    return fp


def _fp(smiles: Any):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _fp_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _compact_ref_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block.get("block_id"),
        "doi": block.get("doi"),
        "cascade_id": block.get("cascade_id"),
        "transform_pair": block.get("transform_pair"),
        "upstream_product": block.get("upstream_product"),
        "downstream_product": block.get("downstream_product"),
    }


def _route_sort_key(route: dict[str, Any]) -> tuple[int, str]:
    return (_native_rank(route), str(route.get("route_id") or ""))


def _native_rank(route: dict[str, Any], fallback: int = 10**9) -> int:
    value = route.get("native_rank")
    if value is None:
        value = route.get("route_rank")
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _transform_match(left: Any, right: Any) -> bool:
    left_norm = _norm_transform(left)
    right_norm = _norm_transform(right)
    if not left_norm or not right_norm or "unknown" in {left_norm, right_norm}:
        return False
    return left_norm == right_norm


def _norm_transform(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"unsupported JSON array split format: {path}")
        return [row for row in payload if isinstance(row, dict)]
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    meta = report.get("metadata") or {}
    lines = [
        "# v4 Held-Out Block Recovery Audit",
        "",
        f"- route pool: `{meta.get('route_pool')}`",
        f"- reference split: `{meta.get('split')}`",
        f"- analog similarity threshold: `{meta.get('analog_similarity')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Notes", "", "- Exact block recovery requires both adjacent reactions to match held-out v4 block reactions in retrosynthetic route order."])
    lines.append("- Analog block recovery requires both corresponding transition fingerprints to pass the similarity threshold.")
    lines.append("- Transform-consistent analog additionally requires both non-unknown transform labels to match.")
    lines.append("")
    return "\n".join(lines)


def _parse_top_ks(value: str) -> tuple[int, ...]:
    out = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    return out or (1, 3, 5, 10, 50)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit held-out v4 cascade block recovery in a ChemEnzy route pool")
    ap.add_argument("--route-pool", required=True)
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    ap.add_argument("--top-ks", default="1,3,5,10,50")
    args = ap.parse_args()
    report = audit_v4_heldout_block_recovery(
        route_pool=Path(args.route_pool),
        program_manifest=Path(args.program_manifest),
        split=args.split,
        output_json=Path(args.output_json),
        analog_similarity=args.analog_similarity,
        top_ks=_parse_top_ks(args.top_ks),
    )
    print(json.dumps({"summary": report["summary"], "outputs": {"json": args.output_json}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
