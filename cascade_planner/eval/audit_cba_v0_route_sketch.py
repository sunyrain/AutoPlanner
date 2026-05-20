"""Audit CBA-v0 route-sketch recovery on held-out cascade blocks.

This script does not claim to produce executable routes.  It checks whether a
trained CBA-v0 pair classifier can expose held-out cascade-core transform pairs
and whether those predicted pairs have train-split structural analogue support.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.eval.train_cba_v0_pair_classifier import _dataset, _model_specs


SCHEMA_VERSION = "cba_v0_route_sketch_audit.v1"
TOP_KS = (1, 3, 5, 10, 20)


def audit_cba_v0_route_sketch(
    *,
    model_bundle: Path,
    program_manifest: Path,
    output_json: Path,
    report: Path,
    split: str = "test",
    route_pool_recovery: Path | None = None,
    model_name: str = "full_context",
    weak_analog_similarity: float = 0.40,
    strong_analog_similarity: float = 0.55,
) -> dict[str, Any]:
    started = time.monotonic()
    bundle = _load_bundle(model_bundle)
    encoder = bundle["label_encoder"]
    feature_schema = dict(bundle.get("feature_schema") or {})
    models = dict(bundle.get("models") or {})
    if model_name not in models:
        raise ValueError(f"model {model_name!r} not found in bundle; available={sorted(models)}")
    model = models[model_name]
    model_specs = _model_specs(list(feature_schema.get("feature_names") or []))
    feature_indices = model_specs.get(model_name)
    if not feature_indices:
        raise ValueError(f"feature spec for model {model_name!r} is empty")

    train_blocks = _detailed_blocks(program_manifest, split="train")
    heldout_blocks = _detailed_blocks(program_manifest, split=split)
    rows_for_features = [_feature_row_from_block(row) for row in heldout_blocks]
    data = _dataset(rows_for_features, schema=feature_schema, encoder=encoder, require_seen_label=False)
    proba = model.predict_proba(data["x"][:, feature_indices], num_iteration=getattr(model, "best_iteration_", None))
    class_names = list(encoder.classes_)
    train_index = _prototype_index(train_blocks)
    route_pool = _route_pool_index(route_pool_recovery) if route_pool_recovery else {}

    block_rows = []
    for idx, block in enumerate(heldout_blocks):
        order = sorted(range(proba.shape[1]), key=lambda col: (-float(proba[idx, col]), str(class_names[col])))
        top_pairs = [str(class_names[col]) for col in order]
        block_rows.append(
            _block_audit_row(
                block,
                top_pairs=top_pairs,
                scores=[float(proba[idx, col]) for col in order],
                train_index=train_index,
                route_pool=route_pool,
                weak_analog_similarity=weak_analog_similarity,
                strong_analog_similarity=strong_analog_similarity,
            )
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "model_bundle": str(model_bundle),
            "program_manifest": str(program_manifest),
            "split": split,
            "route_pool_recovery": str(route_pool_recovery) if route_pool_recovery else None,
            "model_name": model_name,
            "weak_analog_similarity": weak_analog_similarity,
            "strong_analog_similarity": strong_analog_similarity,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "CBA predicts transform-pair sketch; held-out upstream structures are evaluation-only for analog support",
        },
        "counts": {
            "train_blocks": len(train_blocks),
            "heldout_blocks": len(heldout_blocks),
            "heldout_targets": len({row["target_smiles"] for row in heldout_blocks}),
            "model_classes": len(class_names),
            "routed_heldout_targets": len({row["target_smiles"] for row in heldout_blocks if _has_route_pool(_route_row(route_pool, row["target_smiles"]))}),
        },
        "summary": _summary(block_rows),
        "target_summary": _target_summary(block_rows),
        "route_pool_comparison": _route_pool_comparison(block_rows, route_pool),
        "top_miss_pairs": _top_miss_pairs(block_rows),
        "blocks": block_rows,
        "examples": block_rows[:40],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown(result), encoding="utf-8")
    return result


def _block_audit_row(
    block: dict[str, Any],
    *,
    top_pairs: list[str],
    scores: list[float],
    train_index: dict[str, list[dict[str, Any]]],
    route_pool: dict[str, dict[str, Any]],
    weak_analog_similarity: float,
    strong_analog_similarity: float,
) -> dict[str, Any]:
    true_pair = block["transform_pair"]
    pair_rank = top_pairs.index(true_pair) + 1 if true_pair in top_pairs else None
    true_pair_support = _best_support(block, train_index.get(true_pair) or [])
    predicted_supports = []
    for rank, pair in enumerate(top_pairs[:20], start=1):
        support = _best_support(block, train_index.get(pair) or [])
        predicted_supports.append(
            {
                "rank": rank,
                "transform_pair": pair,
                "score": round(float(scores[rank - 1]), 6),
                "best_upstream_similarity": support["best_upstream_similarity"],
                "best_target_similarity": support["best_target_similarity"],
                "support_count": support["support_count"],
                "best_support": support["best_support"],
            }
        )
    route_row = _route_row(route_pool, block["target_smiles"])
    return {
        "block_id": block["block_id"],
        "program_id": block["program_id"],
        "doi": block["doi"],
        "target_smiles": block["target_smiles"],
        "true_transform_pair": true_pair,
        "pair_rank": pair_rank,
        "true_pair_support_count": true_pair_support["support_count"],
        "true_pair_best_upstream_similarity": true_pair_support["best_upstream_similarity"],
        "true_pair_best_target_similarity": true_pair_support["best_target_similarity"],
        "true_pair_has_weak_analog": true_pair_support["best_upstream_similarity"] >= weak_analog_similarity,
        "true_pair_has_strong_analog": true_pair_support["best_upstream_similarity"] >= strong_analog_similarity,
        "routed_by_chem_enzy": _has_route_pool(route_row),
        "chem_enzy_route_count": int((route_row or {}).get("route_count") or 0),
        "chem_enzy_transform_consistent_rank": (route_row or {}).get("best_transform_consistent_analog_native_rank"),
        "top_predicted_supports": predicted_supports,
        "hit_at": _hit_at(
            pair_rank=pair_rank,
            true_pair_support=true_pair_support,
            predicted_supports=predicted_supports,
            weak_analog_similarity=weak_analog_similarity,
            strong_analog_similarity=strong_analog_similarity,
        ),
    }


def _hit_at(
    *,
    pair_rank: int | None,
    true_pair_support: dict[str, Any],
    predicted_supports: list[dict[str, Any]],
    weak_analog_similarity: float,
    strong_analog_similarity: float,
) -> dict[str, dict[str, bool]]:
    out = {}
    for k in TOP_KS:
        top = predicted_supports[:k]
        pair_hit = pair_rank is not None and pair_rank <= k
        true_weak = true_pair_support["best_upstream_similarity"] >= weak_analog_similarity
        true_strong = true_pair_support["best_upstream_similarity"] >= strong_analog_similarity
        out[str(k)] = {
            "pair": pair_hit,
            "pair_and_weak_analog": pair_hit and true_weak,
            "pair_and_strong_analog": pair_hit and true_strong,
            "any_predicted_weak_analog": any(row["best_upstream_similarity"] >= weak_analog_similarity for row in top),
            "any_predicted_strong_analog": any(row["best_upstream_similarity"] >= strong_analog_similarity for row in top),
        }
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": _metric_summary(rows),
        "routed_chem_enzy_subset": _metric_summary([row for row in rows if row.get("routed_by_chem_enzy")]),
        "hidden_or_nonisolated_support_proxy": {
            "note": "hidden labels are not in this audit row schema; use block-pair support as route sketch readiness only"
        },
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ranks = [row.get("pair_rank") for row in rows]
    covered = [int(rank) for rank in ranks if rank is not None]
    out: dict[str, Any] = {
        "blocks": total,
        "covered_pair_blocks": len(covered),
        "pair_coverage": _rate(len(covered), total),
        "mrr_all": round(sum((1.0 / int(rank)) if rank else 0.0 for rank in ranks) / max(total, 1), 6),
        "mrr_covered": round(sum(1.0 / rank for rank in covered) / max(len(covered), 1), 6) if covered else 0.0,
    }
    for k in TOP_KS:
        key = str(k)
        for metric in (
            "pair",
            "pair_and_weak_analog",
            "pair_and_strong_analog",
            "any_predicted_weak_analog",
            "any_predicted_strong_analog",
        ):
            out[f"{metric}_at_{k}"] = _rate(
                sum(1 for row in rows if (row.get("hit_at") or {}).get(key, {}).get(metric)),
                total,
            )
    return out


def _target_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[row["target_smiles"]].append(row)
    target_rows = []
    for target, target_blocks in by_target.items():
        target_rows.append(
            {
                "target_smiles": target,
                "blocks": len(target_blocks),
                "routed_by_chem_enzy": any(row.get("routed_by_chem_enzy") for row in target_blocks),
                "hit_at": {
                    str(k): {
                        "pair": any((row.get("hit_at") or {}).get(str(k), {}).get("pair") for row in target_blocks),
                        "pair_and_weak_analog": any((row.get("hit_at") or {}).get(str(k), {}).get("pair_and_weak_analog") for row in target_blocks),
                        "pair_and_strong_analog": any((row.get("hit_at") or {}).get(str(k), {}).get("pair_and_strong_analog") for row in target_blocks),
                    }
                    for k in TOP_KS
                },
            }
        )
    return {
        "all_targets": _target_metric_summary(target_rows),
        "routed_chem_enzy_targets": _target_metric_summary([row for row in target_rows if row.get("routed_by_chem_enzy")]),
    }


def _target_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    out = {"targets": total}
    for k in TOP_KS:
        key = str(k)
        for metric in ("pair", "pair_and_weak_analog", "pair_and_strong_analog"):
            out[f"{metric}_at_{k}"] = _rate(sum(1 for row in rows if (row.get("hit_at") or {}).get(key, {}).get(metric)), total)
    return out


def _route_pool_comparison(rows: list[dict[str, Any]], route_pool: dict[str, dict[str, Any]]) -> dict[str, Any]:
    routed_rows = [row for row in rows if row.get("routed_by_chem_enzy")]
    routed_targets = {row["target_smiles"] for row in routed_rows}
    route_targets_with_tc = {
        target
        for target in routed_targets
        if (_route_row(route_pool, target) or {}).get("best_transform_consistent_analog_native_rank") is not None
    }
    return {
        "routed_blocks": len(routed_rows),
        "routed_targets": len(routed_targets),
        "chem_enzy_targets_with_transform_consistent_block": len(route_targets_with_tc),
        "chem_enzy_target_transform_consistent_rate": _rate(len(route_targets_with_tc), len(routed_targets)),
        "cba_sketch_routed_block_summary": _metric_summary(routed_rows),
    }


def _top_miss_pairs(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        if not (row.get("hit_at") or {}).get("10", {}).get("pair"):
            counter[row.get("true_transform_pair") or "unknown"] += 1
    return dict(counter.most_common(20))


def _best_support(block: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid_candidates = [
        cand
        for cand in candidates
        if cand.get("upstream_product")
        and cand.get("upstream_main_reactant")
        and cand.get("upstream_fp") is not None
    ]
    if not valid_candidates:
        return {"support_count": 0, "best_upstream_similarity": 0.0, "best_target_similarity": 0.0, "best_support": None}
    up_fp = block.get("upstream_fp")
    target_fp = block.get("target_fp")
    best = None
    best_score = -1.0
    for cand in valid_candidates:
        up_sim = _fp_similarity(up_fp, cand.get("upstream_fp"))
        target_sim = _fp_similarity(target_fp, cand.get("target_fp"))
        score = 0.75 * up_sim + 0.25 * target_sim
        if score > best_score:
            best_score = score
            best = {
                "program_id": cand.get("program_id"),
                "doi": cand.get("doi"),
                "target_smiles": cand.get("target_smiles"),
                "upstream_product": cand.get("upstream_product"),
                "upstream_main_reactant": cand.get("upstream_main_reactant"),
                "upstream_similarity": round(float(up_sim), 6),
                "target_similarity": round(float(target_sim), 6),
            }
    return {
        "support_count": len(valid_candidates),
        "best_upstream_similarity": float((best or {}).get("upstream_similarity") or 0.0),
        "best_target_similarity": float((best or {}).get("target_similarity") or 0.0),
        "best_support": best,
    }


def _prototype_index(blocks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        out[block["transform_pair"]].append(block)
    return dict(out)


def _detailed_blocks(program_manifest: Path, *, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    path = Path((manifest.get("outputs") or {})[split])
    blocks = []
    for program in _read_jsonl(path):
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        compatibility = program.get("compatibility") or {}
        for idx, (left, right) in enumerate(zip(steps, steps[1:])):
            up = _norm(left.get("transformation_superclass"))
            down = _norm(right.get("transformation_superclass"))
            target = _canon(program.get("target_smiles"))
            downstream_product = _canon(right.get("product_smiles"))
            downstream_main = _canon(right.get("main_reactant"))
            upstream_product = _canon(left.get("product_smiles"))
            upstream_main = _canon(left.get("main_reactant"))
            blocks.append(
                {
                    "block_id": f"{program.get('program_id')}::{idx}",
                    "program_id": str(program.get("program_id") or ""),
                    "doi": str(program.get("doi") or ""),
                    "target_smiles": target,
                    "downstream_product": downstream_product,
                    "downstream_main_reactant": downstream_main,
                    "downstream_transform": down,
                    "upstream_product": upstream_product,
                    "upstream_main_reactant": upstream_main,
                    "transform_pair": f"{up}->{down}",
                    "cascade_type": _norm(program.get("cascade_type")),
                    "compatibility_label": _norm(compatibility.get("compatibility_label")),
                    "right_catalyst_classes": _norm_list(right.get("catalyst_classes")),
                    "right_condition_tokens": _norm_list(right.get("condition_tokens")),
                    "target_fp": _fp(target),
                    "upstream_fp": _transition_fp(upstream_product, upstream_main),
                }
            )
    return blocks


def _feature_row_from_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block["block_id"],
        "split": "",
        "program_id": block["program_id"],
        "doi": block["doi"],
        "target_smiles": block["target_smiles"],
        "downstream_product": block["downstream_product"],
        "downstream_main_reactant": block["downstream_main_reactant"],
        "downstream_transform": block["downstream_transform"],
        "transform_pair": block["transform_pair"],
        "cascade_type": block["cascade_type"],
        "compatibility_label": block["compatibility_label"],
        "right_catalyst_classes": block["right_catalyst_classes"],
        "right_condition_tokens": block["right_condition_tokens"],
    }


def _route_pool_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {_canon(row.get("target_smiles")): row for row in payload.get("targets") or [] if row.get("target_smiles")}


def _route_row(route_pool: dict[str, dict[str, Any]], target_smiles: str) -> dict[str, Any] | None:
    return route_pool.get(_canon(target_smiles)) if route_pool else None


def _has_route_pool(row: dict[str, Any] | None) -> bool:
    return bool(row and int(row.get("route_count") or 0) > 0)


def _load_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _transition_fp(product_smiles: str, main_reactant: str):
    product_fp = _fp(product_smiles)
    reactant_fp = _fp(main_reactant)
    if product_fp is None and reactant_fp is None:
        return None
    if product_fp is None:
        return reactant_fp
    if reactant_fp is None:
        return product_fp
    arr_product = np.zeros((1024,), dtype=np.int8)
    arr_reactant = np.zeros((1024,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(product_fp, arr_product)
    DataStructs.ConvertToNumpyArray(reactant_fp, arr_reactant)
    arr = np.maximum(arr_product, arr_reactant)
    fp = DataStructs.ExplicitBitVect(1024)
    for bit in np.nonzero(arr)[0]:
        fp.SetBit(int(bit))
    return fp


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _fp_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _canon(value: Any) -> str:
    return canonical_smiles(str(value or "")) or str(value or "")


def _norm(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _norm_list(values: Any) -> list[str]:
    return sorted({_norm(value) for value in (values or []) if _norm(value) != "unknown"})


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    summary = (result.get("summary") or {}).get("all") or {}
    routed = (result.get("summary") or {}).get("routed_chem_enzy_subset") or {}
    target_summary = ((result.get("target_summary") or {}).get("all_targets") or {})
    routed_targets = ((result.get("target_summary") or {}).get("routed_chem_enzy_targets") or {})
    route_cmp = result.get("route_pool_comparison") or {}
    lines = [
        "# CBA-v0 Route Sketch Audit",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Block-Level Sketch Recovery",
        "",
        "| Subset | Blocks | Pair@1 | Pair@3 | Pair@5 | Pair@10 | Pair+Weak@10 | Pair+Strong@10 | Any Weak@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_table_row("all", summary),
        _metric_table_row("chem_enzy_routed", routed),
        "",
        "## Target-Level Sketch Recovery",
        "",
        "| Subset | Targets | Pair@1 | Pair@3 | Pair@5 | Pair@10 | Pair+Weak@10 | Pair+Strong@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _target_table_row("all", target_summary),
        _target_table_row("chem_enzy_routed", routed_targets),
        "",
        "## Route-Pool Comparison",
        "",
        "```json",
        json.dumps(route_cmp, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "CBA-v0 sketch recovery measures whether the system can propose cascade-core transform-pair ideas before ChemEnzy has generated matching complete routes. "
        "ChemEnzy route-pool transform-consistent recovery remains the comparison floor; pair+analog metrics decide whether train evidence is strong enough for guarded prototype injection.",
        "",
    ]
    return "\n".join(lines)


def _metric_table_row(name: str, metric: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {metric.get('blocks')} | {metric.get('pair_at_1')} | {metric.get('pair_at_3')} | "
        f"{metric.get('pair_at_5')} | {metric.get('pair_at_10')} | {metric.get('pair_and_weak_analog_at_10')} | "
        f"{metric.get('pair_and_strong_analog_at_10')} | {metric.get('any_predicted_weak_analog_at_10')} |"
    )


def _target_table_row(name: str, metric: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {metric.get('targets')} | {metric.get('pair_at_1')} | {metric.get('pair_at_3')} | "
        f"{metric.get('pair_at_5')} | {metric.get('pair_at_10')} | {metric.get('pair_and_weak_analog_at_10')} | "
        f"{metric.get('pair_and_strong_analog_at_10')} |"
    )


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit CBA-v0 route-sketch recovery")
    ap.add_argument("--model-bundle", required=True)
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--route-pool-recovery")
    ap.add_argument("--model-name", default="full_context")
    ap.add_argument("--weak-analog-similarity", type=float, default=0.40)
    ap.add_argument("--strong-analog-similarity", type=float, default=0.55)
    args = ap.parse_args()
    result = audit_cba_v0_route_sketch(
        model_bundle=Path(args.model_bundle),
        program_manifest=Path(args.program_manifest),
        output_json=Path(args.output_json),
        report=Path(args.report),
        split=args.split,
        route_pool_recovery=Path(args.route_pool_recovery) if args.route_pool_recovery else None,
        model_name=args.model_name,
        weak_analog_similarity=args.weak_analog_similarity,
        strong_analog_similarity=args.strong_analog_similarity,
    )
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "summary": result["summary"],
                "target_summary": result["target_summary"],
                "route_pool_comparison": result["route_pool_comparison"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
