"""Estimate whether CascadeRetrievalProvider can fill route-pool strict gaps.

This is not a deployable planner.  It answers a specific bottleneck question:
for held-out targets where the fixed ChemEnzy route pool lacks transform-
consistent analog blocks, can a train-backed cascade retrieval provider supply
candidate block precedents for the same target?
"""
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

from cascade_planner.cascade_search.cascade_retrieval_provider import CascadeRetrievalProvider
from cascade_planner.cascade_search.v4_product_value import canonical_smiles


SCHEMA_VERSION = "provider_routepool_oracle_audit.v1"


def audit_provider_routepool_oracle(
    *,
    program_manifest: Path,
    route_recovery_json: Path,
    output_json: Path,
    split: str = "test",
    modes: tuple[str, ...] = ("block_downstream_transition", "block_downstream_product"),
    limit: int = 20,
    min_similarity: float = 0.20,
    analog_similarity: float = 0.55,
    condition_on_downstream_transform: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    provider = CascadeRetrievalProvider(program_manifest)
    refs = _load_reference_blocks(program_manifest, split=split)
    recovery = json.loads(route_recovery_json.read_text(encoding="utf-8"))
    routed_targets = {
        canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
        for row in recovery.get("targets") or []
        if int(row.get("route_count") or 0) > 0
    }
    refs_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        refs_by_target[str(ref.get("target_smiles") or "")].append(ref)
    rows = []
    for target in sorted(routed_targets):
        target_refs = refs_by_target.get(target) or []
        target_rows = []
        for ref in target_refs:
            best_hits = []
            for mode in modes:
                hits = _retrieve(provider, ref, mode=mode, limit=limit, min_similarity=min_similarity, condition_on_downstream_transform=condition_on_downstream_transform)
                for rank, hit in enumerate(hits, start=1):
                    upstream_sim = _fp_similarity(ref.get("upstream_fp"), _transition_fp(hit.product_smiles, hit.main_reactant))
                    pair_hit = str(hit.transform_pair or "").lower() == str(ref.get("transform_pair") or "").lower()
                    analog_hit = upstream_sim >= analog_similarity
                    best_hits.append(
                        {
                            "mode": mode,
                            "rank": rank,
                            "hit_id": hit.hit_id,
                            "similarity": round(float(hit.similarity), 6),
                            "upstream_similarity": round(float(upstream_sim), 6),
                            "transform_pair": hit.transform_pair,
                            "reference_transform_pair": ref.get("transform_pair"),
                            "pair_hit": pair_hit,
                            "analog_hit": analog_hit,
                            "pair_and_analog": bool(pair_hit and analog_hit),
                            "rxn_smiles": hit.rxn_smiles,
                            "main_reactant": hit.main_reactant,
                            "doi": hit.doi,
                            "program_id": hit.program_id,
                        }
                    )
            target_rows.append(
                {
                    "reference_block_id": ref.get("block_id"),
                    "transform_pair": ref.get("transform_pair"),
                    "best_pair_and_analog_rank": min((row["rank"] for row in best_hits if row.get("pair_and_analog")), default=None),
                    "best_analog_rank": min((row["rank"] for row in best_hits if row.get("analog_hit")), default=None),
                    "hits": sorted(best_hits, key=lambda row: (not row.get("pair_and_analog"), row["rank"], -row["similarity"]))[:10],
                }
            )
        rows.append(
            {
                "target_smiles": target,
                "reference_blocks": len(target_refs),
                "provider_pair_and_analog_any": any(row.get("best_pair_and_analog_rank") is not None for row in target_rows),
                "provider_analog_any": any(row.get("best_analog_rank") is not None for row in target_rows),
                "best_pair_and_analog_rank": min((row["best_pair_and_analog_rank"] for row in target_rows if row.get("best_pair_and_analog_rank") is not None), default=None),
                "best_analog_rank": min((row["best_analog_rank"] for row in target_rows if row.get("best_analog_rank") is not None), default=None),
                "blocks": target_rows[:10],
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "route_recovery_json": str(route_recovery_json),
            "split": split,
            "modes": list(modes),
            "limit": limit,
            "min_similarity": min_similarity,
            "analog_similarity": analog_similarity,
            "condition_on_downstream_transform": condition_on_downstream_transform,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "provider indexes train split; evaluated only as an oracle supplement for routed held-out targets",
        },
        "provider_summary": provider.summary,
        "summary": _summary(rows),
        "top_reference_pairs": dict(Counter(ref.get("transform_pair") for ref in refs).most_common(20)),
        "targets": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _retrieve(
    provider: CascadeRetrievalProvider,
    ref: dict[str, Any],
    *,
    mode: str,
    limit: int,
    min_similarity: float,
    condition_on_downstream_transform: bool,
):
    required_downstream = ref.get("downstream_transform") if condition_on_downstream_transform else None
    if mode in {"block_downstream_transition", "transition"}:
        return provider.retrieve_for_transition(
            str(ref.get("downstream_product") or ""),
            str(ref.get("downstream_main_reactant") or ""),
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            required_downstream_transform=required_downstream,
            exclude_program_ids={str(ref.get("program_id") or "")},
        )
    return provider.retrieve_for_product(
        str(ref.get("downstream_product") or ""),
        mode=mode,
        limit=limit,
        min_similarity=min_similarity,
        required_downstream_transform=required_downstream,
        exclude_program_ids={str(ref.get("program_id") or "")},
    )


def _load_reference_blocks(program_manifest: Path, *, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    split_path = Path((manifest.get("outputs") or {})[split])
    refs = []
    for program in _read_jsonl(split_path):
        target = canonical_smiles(str(program.get("target_smiles") or "")) or str(program.get("target_smiles") or "")
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        for idx, (upstream, downstream) in enumerate(zip(steps, steps[1:])):
            up_transform = _norm_transform(upstream.get("transformation_superclass"))
            down_transform = _norm_transform(downstream.get("transformation_superclass"))
            refs.append(
                {
                    "block_id": f"{program.get('program_id')}::{idx}",
                    "program_id": str(program.get("program_id") or ""),
                    "target_smiles": target,
                    "upstream_product": canonical_smiles(str(upstream.get("product_smiles") or "")) or str(upstream.get("product_smiles") or ""),
                    "upstream_main_reactant": canonical_smiles(str(upstream.get("main_reactant") or "")) or str(upstream.get("main_reactant") or ""),
                    "downstream_product": canonical_smiles(str(downstream.get("product_smiles") or "")) or str(downstream.get("product_smiles") or ""),
                    "downstream_main_reactant": canonical_smiles(str(downstream.get("main_reactant") or "")) or str(downstream.get("main_reactant") or ""),
                    "upstream_transform": up_transform,
                    "downstream_transform": down_transform,
                    "transform_pair": f"{up_transform}->{down_transform}",
                    "upstream_fp": _transition_fp(upstream.get("product_smiles"), upstream.get("main_reactant")),
                }
            )
    return refs


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "routed_targets": len(rows),
        "reference_blocks": sum(int(row.get("reference_blocks") or 0) for row in rows),
        "targets_provider_analog_any": sum(1 for row in rows if row.get("provider_analog_any")),
        "targets_provider_pair_and_analog_any": sum(1 for row in rows if row.get("provider_pair_and_analog_any")),
        "provider_analog_any_rate": round(sum(1 for row in rows if row.get("provider_analog_any")) / max(len(rows), 1), 6),
        "provider_pair_and_analog_any_rate": round(sum(1 for row in rows if row.get("provider_pair_and_analog_any")) / max(len(rows), 1), 6),
        "pair_and_analog_at_1": round(sum(1 for row in rows if row.get("best_pair_and_analog_rank") is not None and row["best_pair_and_analog_rank"] <= 1) / max(len(rows), 1), 6),
        "pair_and_analog_at_3": round(sum(1 for row in rows if row.get("best_pair_and_analog_rank") is not None and row["best_pair_and_analog_rank"] <= 3) / max(len(rows), 1), 6),
        "pair_and_analog_at_5": round(sum(1 for row in rows if row.get("best_pair_and_analog_rank") is not None and row["best_pair_and_analog_rank"] <= 5) / max(len(rows), 1), 6),
        "pair_and_analog_at_10": round(sum(1 for row in rows if row.get("best_pair_and_analog_rank") is not None and row["best_pair_and_analog_rank"] <= 10) / max(len(rows), 1), 6),
        "pair_and_analog_at_20": round(sum(1 for row in rows if row.get("best_pair_and_analog_rank") is not None and row["best_pair_and_analog_rank"] <= 20) / max(len(rows), 1), 6),
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


def _norm_transform(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    lines = [
        "# Provider Route-Pool Oracle Audit",
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
            "## Examples",
            "",
            "```json",
            json.dumps((report.get("targets") or [])[:10], indent=2, ensure_ascii=False)[:8000],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit provider as an oracle supplement for routed held-out targets")
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--route-recovery-json", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--modes", default="block_downstream_transition,block_downstream_product")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min-similarity", type=float, default=0.20)
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    ap.add_argument("--no-downstream-transform-condition", action="store_true")
    args = ap.parse_args()
    report = audit_provider_routepool_oracle(
        program_manifest=Path(args.program_manifest),
        route_recovery_json=Path(args.route_recovery_json),
        output_json=Path(args.output_json),
        split=args.split,
        modes=_parse_modes(args.modes),
        limit=args.limit,
        min_similarity=args.min_similarity,
        analog_similarity=args.analog_similarity,
        condition_on_downstream_transform=not args.no_downstream_transform_condition,
    )
    print(json.dumps({"summary": report["summary"], "outputs": {"json": args.output_json, "md": str(Path(args.output_json).with_suffix('.md'))}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
