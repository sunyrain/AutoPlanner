"""Audit non-oracle bridge from provider hits to ChemEnzy one-step candidates.

For each routed held-out target, retrieve train-backed cascade provider hits
for held-out reference contexts, but do not use the held-out downstream reaction
to construct a route.  Instead, ask whether the provider hit product appears as
a reactant in ChemEnzy's cached one-step candidates for the target.  This
estimates whether provider injection can connect to the existing generator
without oracle downstream sketching.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rdkit import DataStructs, RDLogger

from cascade_planner.cascade_search.cascade_retrieval_provider import CascadeRetrievalProvider
from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.cascadeboard.route_recovery import canonical_side
from cascade_planner.eval.audit_provider_routepool_oracle import _load_reference_blocks, _retrieve, _transition_fp
from cascade_planner.eval.train_ccts_v0_transition_ranker import _candidate_rows_from_cache, _read_json


SCHEMA_VERSION = "provider_chem_enzy_bridge_audit.v1"


def audit_provider_chem_enzy_bridge(
    *,
    program_manifest: Path,
    route_recovery_json: Path,
    chem_enzy_cache: Path,
    output_json: Path,
    split: str = "test",
    limit: int = 20,
    max_chem_candidates: int = 100,
    min_similarity: float = 0.20,
    analog_similarity: float = 0.55,
    bridge_similarity: float = 0.70,
    condition_on_downstream_transform: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    provider = CascadeRetrievalProvider(program_manifest)
    cache = _read_json(chem_enzy_cache)
    recovery = json.loads(route_recovery_json.read_text(encoding="utf-8"))
    routed_targets = [
        canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
        for row in recovery.get("targets") or []
        if int(row.get("route_count") or 0) > 0
    ]
    refs = _load_reference_blocks(program_manifest, split=split)
    refs_by_target: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        refs_by_target.setdefault(str(ref.get("target_smiles") or ""), []).append(ref)

    target_rows = []
    for target in routed_targets:
        downstream_candidates = _downstream_candidates(cache, target, max_chem_candidates=max_chem_candidates)
        rows = []
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
                    bridge = _best_bridge(hit.product_smiles, downstream_candidates)
                    rows.append(
                        {
                            "reference_block_id": ref.get("block_id"),
                            "reference_transform_pair": ref.get("transform_pair"),
                            "mode": mode,
                            "provider_rank": rank,
                            "hit_id": hit.hit_id,
                            "hit_product": hit.product_smiles,
                            "hit_transform_pair": hit.transform_pair,
                            "provider_similarity": round(float(hit.similarity), 6),
                            "upstream_similarity": round(float(upstream_sim), 6),
                            "pair_hit": pair_hit,
                            "analog_hit": analog_hit,
                            "pair_and_analog": bool(pair_hit and analog_hit),
                            "bridge_similarity": round(float(bridge.get("similarity") or 0.0), 6),
                            "bridge_hit": bool(float(bridge.get("similarity") or 0.0) >= bridge_similarity),
                            "bridge_candidate_rank": bridge.get("rank"),
                            "bridge_candidate_reactants": bridge.get("reactants"),
                            "bridge_candidate_reaction": bridge.get("reaction_smiles"),
                        }
                    )
        target_rows.append(
            {
                "target_smiles": target,
                "reference_blocks": len(refs_by_target.get(target) or []),
                "chem_enzy_downstream_candidates": len(downstream_candidates),
                "provider_rows": len(rows),
                "pair_and_analog_any": any(row.get("pair_and_analog") for row in rows),
                "bridge_any": any(row.get("bridge_hit") for row in rows),
                "pair_and_analog_bridge_any": any(row.get("pair_and_analog") and row.get("bridge_hit") for row in rows),
                "best_pair_and_analog_bridge_rank": min(
                    (int(row["provider_rank"]) for row in rows if row.get("pair_and_analog") and row.get("bridge_hit")),
                    default=None,
                ),
                "best_bridge_similarity": max((float(row.get("bridge_similarity") or 0.0) for row in rows), default=0.0),
                "examples": sorted(rows, key=lambda row: (not (row.get("pair_and_analog") and row.get("bridge_hit")), row.get("provider_rank") or 10**9))[:10],
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "route_recovery_json": str(route_recovery_json),
            "chem_enzy_cache": str(chem_enzy_cache),
            "output_json": str(output_json),
            "split": split,
            "limit": limit,
            "max_chem_candidates": max_chem_candidates,
            "min_similarity": min_similarity,
            "analog_similarity": analog_similarity,
            "bridge_similarity": bridge_similarity,
            "condition_on_downstream_transform": condition_on_downstream_transform,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "non-oracle bridge audit; downstream step must be present in ChemEnzy cached target candidates",
        },
        "provider_summary": provider.summary,
        "summary": _summary(target_rows),
        "targets": target_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _downstream_candidates(cache: dict[str, Any], target: str, *, max_chem_candidates: int) -> list[dict[str, Any]]:
    rows = []
    for cand in _candidate_rows_from_cache(cache, target)[: int(max_chem_candidates)]:
        rxn = cand.get("reaction_smiles") or cand.get("rxn_smiles") or ""
        lhs = rxn.split(">>", 1)[0] if ">>" in rxn else ""
        reactants = list(canonical_side(lhs))
        if not reactants and cand.get("main_reactant"):
            reactants = [canonical_smiles(str(cand.get("main_reactant") or ""))]
        rows.append(
            {
                "rank": int(cand.get("rank") or len(rows) + 1),
                "reaction_smiles": rxn,
                "reactants": reactants,
            }
        )
    return rows


def _best_bridge(provider_product: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    product = canonical_smiles(str(provider_product or "")) or str(provider_product or "")
    best = {"similarity": 0.0, "rank": None, "reactants": [], "reaction_smiles": ""}
    for cand in candidates:
        for reactant in cand.get("reactants") or []:
            sim = _smiles_similarity(product, reactant)
            if sim > float(best["similarity"]):
                best = {
                    "similarity": sim,
                    "rank": cand.get("rank"),
                    "reactants": cand.get("reactants") or [],
                    "reaction_smiles": cand.get("reaction_smiles") or "",
                }
    return best


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denom = max(len(rows), 1)
    return {
        "routed_targets": len(rows),
        "targets_pair_and_analog_any": sum(1 for row in rows if row.get("pair_and_analog_any")),
        "targets_bridge_any": sum(1 for row in rows if row.get("bridge_any")),
        "targets_pair_and_analog_bridge_any": sum(1 for row in rows if row.get("pair_and_analog_bridge_any")),
        "pair_and_analog_bridge_rate": round(sum(1 for row in rows if row.get("pair_and_analog_bridge_any")) / denom, 6),
        "pair_and_analog_bridge_at_1": round(
            sum(1 for row in rows if row.get("best_pair_and_analog_bridge_rank") is not None and row["best_pair_and_analog_bridge_rank"] <= 1) / denom,
            6,
        ),
        "pair_and_analog_bridge_at_3": round(
            sum(1 for row in rows if row.get("best_pair_and_analog_bridge_rank") is not None and row["best_pair_and_analog_bridge_rank"] <= 3) / denom,
            6,
        ),
        "pair_and_analog_bridge_at_10": round(
            sum(1 for row in rows if row.get("best_pair_and_analog_bridge_rank") is not None and row["best_pair_and_analog_bridge_rank"] <= 10) / denom,
            6,
        ),
        "pair_and_analog_bridge_at_20": round(
            sum(1 for row in rows if row.get("best_pair_and_analog_bridge_rank") is not None and row["best_pair_and_analog_bridge_rank"] <= 20) / denom,
            6,
        ),
    }


def _smiles_similarity(left: str, right: str) -> float:
    fp_left = _fp(left)
    fp_right = _fp(right)
    if fp_left is None or fp_right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_left, fp_right))


def _fp_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _fp(smiles: Any):
    mol = None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(str(smiles or ""))
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol is not None else None
    except Exception:
        return None


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Provider-ChemEnzy Bridge Audit",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Examples",
            "",
            "```json",
            json.dumps((report.get("targets") or [])[:10], indent=2, ensure_ascii=False)[:10000],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit provider-to-ChemEnzy non-oracle bridge")
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--route-recovery-json", required=True)
    ap.add_argument("--chem-enzy-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-chem-candidates", type=int, default=100)
    ap.add_argument("--min-similarity", type=float, default=0.20)
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    ap.add_argument("--bridge-similarity", type=float, default=0.70)
    ap.add_argument("--no-downstream-transform-condition", action="store_true")
    args = ap.parse_args()
    report = audit_provider_chem_enzy_bridge(
        program_manifest=Path(args.program_manifest),
        route_recovery_json=Path(args.route_recovery_json),
        chem_enzy_cache=Path(args.chem_enzy_cache),
        output_json=Path(args.output_json),
        split=args.split,
        limit=args.limit,
        max_chem_candidates=args.max_chem_candidates,
        min_similarity=args.min_similarity,
        analog_similarity=args.analog_similarity,
        bridge_similarity=args.bridge_similarity,
        condition_on_downstream_transform=not args.no_downstream_transform_condition,
    )
    print(json.dumps({"summary": report["summary"], "outputs": {"json": args.output_json, "md": str(Path(args.output_json).with_suffix('.md'))}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
