"""Build route-pool sketches injected with CascadeRetrievalProvider hits.

The output is a synthetic route-pool-like JSONL used only for offline recovery
audits.  It appends two-step provider block sketches for routed held-out
targets, preserving ChemEnzy routes and marking all injected rows explicitly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.cascade_retrieval_provider import CascadeRetrievalProvider
from cascade_planner.cascade_search.v4_product_value import canonical_smiles, stable_id
from cascade_planner.eval.audit_provider_routepool_oracle import _load_reference_blocks, _retrieve


SCHEMA_VERSION = "provider_injected_route_sketches.v1"


def build_provider_injected_route_sketches(
    *,
    route_pool: Path,
    route_recovery_json: Path,
    program_manifest: Path,
    output_jsonl: Path,
    report_json: Path,
    split: str = "test",
    limit: int = 20,
    max_injected_per_target: int = 5,
    min_similarity: float = 0.20,
    analog_similarity: float = 0.55,
    condition_on_downstream_transform: bool = True,
    include_all_hits: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    native_routes = _read_jsonl(route_pool)
    recovery = json.loads(route_recovery_json.read_text(encoding="utf-8"))
    routed_targets = {
        canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
        for row in recovery.get("targets") or []
        if int(row.get("route_count") or 0) > 0
    }
    refs = _load_reference_blocks(program_manifest, split=split)
    refs_by_target: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        refs_by_target.setdefault(str(ref.get("target_smiles") or ""), []).append(ref)
    provider = CascadeRetrievalProvider(program_manifest)

    injected_routes = []
    for target in sorted(routed_targets):
        candidates = []
        for ref in refs_by_target.get(target) or []:
            for mode in ("block_downstream_transition", "block_downstream_product"):
                hits = _retrieve(
                    provider,
                    ref,
                    mode=mode,
                    limit=limit,
                    min_similarity=min_similarity,
                    condition_on_downstream_transform=condition_on_downstream_transform,
                )
                for rank, hit in enumerate(hits, start=1):
                    upstream_sim = _fp_similarity(ref.get("upstream_fp"), _transition_fp(hit.product_smiles, hit.main_reactant))
                    pair_hit = str(hit.transform_pair or "").lower() == str(ref.get("transform_pair") or "").lower()
                    analog_hit = upstream_sim >= analog_similarity
                    if not include_all_hits and not (pair_hit and analog_hit):
                        continue
                    candidates.append(
                        {
                            "target_smiles": target,
                            "reference": ref,
                            "hit": hit,
                            "mode": mode,
                            "rank": rank,
                            "upstream_similarity": upstream_sim,
                            "pair_hit": pair_hit,
                            "analog_hit": analog_hit,
                            "pair_and_analog": bool(pair_hit and analog_hit),
                            "score": _score(rank=rank, hit_similarity=hit.similarity, upstream_similarity=upstream_sim, pair_hit=pair_hit, analog_hit=analog_hit),
                        }
                    )
        candidates.sort(key=lambda row: (-float(row["score"]), int(row["rank"]), str(row["hit"].hit_id)))
        for idx, candidate in enumerate(candidates[: int(max_injected_per_target)]):
            injected_routes.append(_sketch_route(candidate, injected_rank=idx))

    combined = []
    for route in native_routes:
        row = dict(route)
        row["provider_injected"] = False
        row["original_native_rank"] = row.get("native_rank")
        combined.append(row)
    native_by_target: dict[str, list[dict[str, Any]]] = {}
    for route in combined:
        native_by_target.setdefault(_target_key(route), []).append(route)
    injected_by_target: dict[str, list[dict[str, Any]]] = {}
    for route in injected_routes:
        injected_by_target.setdefault(_target_key(route), []).append(route)

    output_rows = []
    for target in sorted(set(native_by_target) | set(injected_by_target)):
        native_group = sorted(native_by_target.get(target) or [], key=lambda row: int(row.get("native_rank") or 10**9))
        injected_group = injected_by_target.get(target) or []
        # Put provider sketches first in the injected pool.  This is an oracle
        # upper-bound replay, not a deployable ranking policy.
        group = injected_group + native_group
        for rank, route in enumerate(group):
            row = dict(route)
            row["native_rank"] = rank
            row["provider_injected_replay_rank"] = rank
            output_rows.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, output_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool": str(route_pool),
            "route_recovery_json": str(route_recovery_json),
            "program_manifest": str(program_manifest),
            "output_jsonl": str(output_jsonl),
            "split": split,
            "limit": limit,
            "max_injected_per_target": max_injected_per_target,
            "min_similarity": min_similarity,
            "analog_similarity": analog_similarity,
            "condition_on_downstream_transform": condition_on_downstream_transform,
            "include_all_hits": include_all_hits,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "offline oracle sketch injection; provider indexes train split; injected rows are not ChemEnzy-validated complete routes",
        },
        "counts": {
            "native_routes": len(native_routes),
            "injected_routes": len(injected_routes),
            "output_routes": len(output_rows),
            "targets_with_injected": len(injected_by_target),
            "routed_targets": len(routed_targets),
        },
        "provider_summary": provider.summary,
        "examples": [_compact(route) for route in injected_routes[:20]],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _sketch_route(candidate: dict[str, Any], *, injected_rank: int) -> dict[str, Any]:
    ref = candidate["reference"]
    hit = candidate["hit"]
    target = str(candidate["target_smiles"])
    upstream_product = canonical_smiles(str(hit.product_smiles or "")) or str(hit.product_smiles or "")
    downstream_product = canonical_smiles(str(ref.get("downstream_product") or "")) or str(ref.get("downstream_product") or "")
    upstream_reactants = list(hit.reactants)
    downstream_main = canonical_smiles(str(ref.get("downstream_main_reactant") or "")) or str(ref.get("downstream_main_reactant") or "")
    downstream_reactants = [upstream_product]
    if downstream_main and downstream_main != upstream_product:
        downstream_reactants.append(downstream_main)
    upstream_transform = str(hit.transformation_superclass or "unknown")
    downstream_transform = str(hit.downstream_transformation_superclass or ref.get("downstream_transform") or "unknown")
    route_id = stable_id("provider_injected_route", target, ref.get("block_id"), hit.hit_id, injected_rank)
    return {
        "schema_version": "v4_cascade_product_route.v1",
        "route_id": route_id,
        "target_id": stable_id("target", target),
        "target_smiles": target,
        "route_source": "CascadeRetrievalProvider",
        "provider_injected": True,
        "native_rank": injected_rank,
        "original_native_rank": None,
        "native_score": float(candidate.get("score") or 0.0),
        "stock_closed": False,
        "terminal_reactants": sorted(set(upstream_reactants)),
        "metadata": {
            "provider_injection": {
                "reference_block_id": ref.get("block_id"),
                "reference_transform_pair": ref.get("transform_pair"),
                "hit_id": hit.hit_id,
                "hit_transform_pair": hit.transform_pair,
                "mode": candidate.get("mode"),
                "rank": candidate.get("rank"),
                "upstream_similarity": round(float(candidate.get("upstream_similarity") or 0.0), 6),
                "pair_hit": candidate.get("pair_hit"),
                "analog_hit": candidate.get("analog_hit"),
                "pair_and_analog": candidate.get("pair_and_analog"),
                "doi": hit.doi,
                "program_id": hit.program_id,
            }
        },
        "steps": [
            {
                "step_id": f"{route_id}_upstream",
                "step_index": 1,
                "product_smiles": upstream_product,
                "products": [upstream_product],
                "reactants": upstream_reactants,
                "rxn_smiles": hit.rxn_smiles,
                "source_model": "cascade_retrieval_provider",
                "transformation_superclass": upstream_transform,
                "transformation_name": upstream_transform,
                "step_mode": "retrieved_cascade_precedent",
                "pairwise_mode": "sequential",
                "intermediate_isolated": False,
                "condition_tokens": list(hit.condition_tokens),
                "catalyst_classes": list(hit.catalyst_classes),
                "v4_step_evidence": hit.to_dict(),
            },
            {
                "step_id": f"{route_id}_downstream",
                "step_index": 2,
                "product_smiles": downstream_product,
                "products": [downstream_product],
                "reactants": downstream_reactants,
                "rxn_smiles": f"{'.'.join(downstream_reactants)}>>{downstream_product}",
                "source_model": "cascade_retrieval_provider_sketch",
                "transformation_superclass": downstream_transform,
                "transformation_name": downstream_transform,
                "step_mode": "retrieved_cascade_context",
                "pairwise_mode": "sequential",
                "intermediate_isolated": False,
                "condition_tokens": list(hit.condition_tokens),
                "catalyst_classes": list(hit.catalyst_classes),
                "v4_step_evidence": {
                    "reference_block_id": ref.get("block_id"),
                    "reference_transform_pair": ref.get("transform_pair"),
                    "source": "heldout_reference_shape_for_oracle_sketch",
                },
            },
        ],
    }


def _score(*, rank: int, hit_similarity: float, upstream_similarity: float, pair_hit: bool, analog_hit: bool) -> float:
    return (
        float(pair_hit) * 2.0
        + float(analog_hit) * 1.0
        + float(hit_similarity)
        + float(upstream_similarity)
        + 1.0 / float(max(rank, 1))
    )


def _target_key(route: dict[str, Any]) -> str:
    return canonical_smiles(str(route.get("target_smiles") or "")) or str(route.get("target_smiles") or route.get("target_id") or "")


def _compact(route: dict[str, Any]) -> dict[str, Any]:
    meta = (route.get("metadata") or {}).get("provider_injection") or {}
    return {
        "route_id": route.get("route_id"),
        "target_smiles": route.get("target_smiles"),
        "native_score": route.get("native_score"),
        "reference_transform_pair": meta.get("reference_transform_pair"),
        "hit_transform_pair": meta.get("hit_transform_pair"),
        "upstream_similarity": meta.get("upstream_similarity"),
        "pair_and_analog": meta.get("pair_and_analog"),
        "doi": meta.get("doi"),
    }


def _transition_fp(product_smiles: Any, main_reactant: Any):
    product_fp = _fp(product_smiles)
    reactant_fp = _fp(main_reactant)
    if product_fp is None and reactant_fp is None:
        return None
    if product_fp is None:
        return reactant_fp
    if reactant_fp is None:
        return product_fp
    import numpy as np

    arr_product = np.zeros((2048,), dtype=np.int8)
    arr_reactant = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(product_fp, arr_product)
    DataStructs.ConvertToNumpyArray(reactant_fp, arr_reactant)
    arr = np.maximum(arr_product, arr_reactant)
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Provider-Injected Route Sketches",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Examples",
        "",
        "```json",
        json.dumps(report.get("examples") or [], indent=2, ensure_ascii=False)[:8000],
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Build provider-injected route-pool sketches")
    ap.add_argument("--route-pool", required=True)
    ap.add_argument("--route-recovery-json", required=True)
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-injected-per-target", type=int, default=5)
    ap.add_argument("--min-similarity", type=float, default=0.20)
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    ap.add_argument("--no-downstream-transform-condition", action="store_true")
    ap.add_argument("--include-all-hits", action="store_true")
    args = ap.parse_args()
    report = build_provider_injected_route_sketches(
        route_pool=Path(args.route_pool),
        route_recovery_json=Path(args.route_recovery_json),
        program_manifest=Path(args.program_manifest),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        split=args.split,
        limit=args.limit,
        max_injected_per_target=args.max_injected_per_target,
        min_similarity=args.min_similarity,
        analog_similarity=args.analog_similarity,
        condition_on_downstream_transform=not args.no_downstream_transform_condition,
        include_all_hits=args.include_all_hits,
    )
    print(json.dumps({"counts": report["counts"], "outputs": {"jsonl": args.output_jsonl, "report": args.report_json}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
