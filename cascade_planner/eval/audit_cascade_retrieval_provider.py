"""Audit v4 train-backed CascadeRetrievalProvider on held-out v4 blocks."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.cascade_retrieval_provider import CascadeRetrievalProvider
from cascade_planner.cascade_search.v4_product_value import canonical_smiles


SCHEMA_VERSION = "cascade_retrieval_provider_audit.v1"


def audit_cascade_retrieval_provider(
    *,
    program_manifest: Path,
    split: str,
    output_json: Path,
    modes: list[str] | None = None,
    limit: int = 20,
    min_similarity: float = 0.20,
    analog_similarity: float = 0.55,
    condition_on_downstream_transform: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    provider = CascadeRetrievalProvider(program_manifest)
    refs = _reference_blocks(program_manifest, split=split)
    selected_modes = modes or ["block_downstream_product", "block_downstream_transition", "step_product", "transition"]
    mode_reports = {}
    for mode in selected_modes:
        mode_reports[mode] = _audit_mode(
            provider,
            refs,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            analog_similarity=analog_similarity,
            condition_on_downstream_transform=condition_on_downstream_transform,
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "split": split,
            "limit": limit,
            "min_similarity": min_similarity,
            "analog_similarity": analog_similarity,
            "condition_on_downstream_transform": condition_on_downstream_transform,
            "elapsed_s": round(time.monotonic() - started, 3),
            "leakage_guard": "provider indexes only train split; references are from requested held-out split",
        },
        "provider_summary": provider.summary,
        "reference_summary": _reference_summary(refs),
        "modes": mode_reports,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _audit_mode(
    provider: CascadeRetrievalProvider,
    refs: list[dict[str, Any]],
    *,
    mode: str,
    limit: int,
    min_similarity: float,
    analog_similarity: float,
    condition_on_downstream_transform: bool,
) -> dict[str, Any]:
    rows = []
    for ref in refs:
        if mode in {"block_downstream_transition", "transition"}:
            hits = provider.retrieve_for_transition(
                ref["downstream_product"],
                ref["downstream_main_reactant"],
                mode=mode,
                limit=limit,
                min_similarity=min_similarity,
                required_downstream_transform=ref["downstream_transform"] if condition_on_downstream_transform else None,
                exclude_program_ids={ref["program_id"]},
            )
        else:
            hits = provider.retrieve_for_product(
                ref["downstream_product"],
                mode=mode,
                limit=limit,
                min_similarity=min_similarity,
                required_downstream_transform=ref["downstream_transform"] if condition_on_downstream_transform else None,
                exclude_program_ids={ref["program_id"]},
            )
        hit_rows = []
        for rank, hit in enumerate(hits, start=1):
            upstream_sim = _fp_similarity(ref.get("upstream_fp"), _transition_fp(hit.product_smiles, hit.main_reactant))
            transform_pair_hit = hit.transform_pair == ref["transform_pair"]
            upstream_transform_hit = hit.transformation_superclass == ref["upstream_transform"]
            analog_hit = upstream_sim >= analog_similarity
            hit_rows.append(
                {
                    "rank": rank,
                    "hit_id": hit.hit_id,
                    "similarity": round(float(hit.similarity), 6),
                    "upstream_similarity": round(float(upstream_sim), 6),
                    "transform_pair": hit.transform_pair,
                    "transform_pair_hit": transform_pair_hit,
                    "upstream_transform_hit": upstream_transform_hit,
                    "analog_hit": analog_hit,
                    "doi": hit.doi,
                    "cascade_id": hit.cascade_id,
                }
            )
        rows.append(
            {
                "reference_block_id": ref["block_id"],
                "program_id": ref["program_id"],
                "doi": ref["doi"],
                "target_smiles": ref["target_smiles"],
                "transform_pair": ref["transform_pair"],
                "n_hits": len(hit_rows),
                "best_similarity": hit_rows[0]["similarity"] if hit_rows else None,
                "best_upstream_similarity": max((row["upstream_similarity"] for row in hit_rows), default=None),
                "hit_rows": hit_rows[:10],
                "hit_at": _hit_at(hit_rows),
            }
        )
    return {
        "summary": _mode_summary(rows, limit=limit),
        "top_transform_pair_misses": _transform_pair_misses(rows),
        "examples": rows[:30],
    }


def _reference_blocks(program_manifest: Path, *, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    split_path = Path((manifest.get("outputs") or {})[split])
    programs = _read_jsonl(split_path)
    out = []
    for program in programs:
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        for idx, (upstream, downstream) in enumerate(zip(steps, steps[1:])):
            up_transform = _norm_transform(upstream.get("transformation_superclass"))
            down_transform = _norm_transform(downstream.get("transformation_superclass"))
            out.append(
                {
                    "block_id": f"{program.get('program_id')}::{idx}",
                    "program_id": str(program.get("program_id") or ""),
                    "doi": str(program.get("doi") or ""),
                    "cascade_id": str(program.get("cascade_id") or ""),
                    "target_smiles": canonical_smiles(str(program.get("target_smiles") or "")) or str(program.get("target_smiles") or ""),
                    "upstream_product": canonical_smiles(str(upstream.get("product_smiles") or "")) or str(upstream.get("product_smiles") or ""),
                    "upstream_main_reactant": canonical_smiles(str(upstream.get("main_reactant") or "")) or str(upstream.get("main_reactant") or ""),
                    "downstream_product": canonical_smiles(str(downstream.get("product_smiles") or "")) or str(downstream.get("product_smiles") or ""),
                    "downstream_main_reactant": canonical_smiles(str(downstream.get("main_reactant") or "")) or str(downstream.get("main_reactant") or ""),
                    "upstream_transform": up_transform,
                    "downstream_transform": down_transform,
                    "transform_pair": f"{up_transform}->{down_transform}",
                    "upstream_fp": _transition_fp(upstream.get("product_smiles"), upstream.get("main_reactant")),
                    "downstream_fp": _transition_fp(downstream.get("product_smiles"), downstream.get("main_reactant")),
                }
            )
    return out


def _mode_summary(rows: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "reference_blocks": len(rows),
        "with_hits": sum(1 for row in rows if row["n_hits"] > 0),
        "with_hits_rate": round(sum(1 for row in rows if row["n_hits"] > 0) / max(len(rows), 1), 6),
    }
    for k in (1, 3, 5, 10, limit):
        if k <= 0:
            continue
        key = str(k)
        out[f"transform_pair_hit_at_{k}"] = round(sum(1 for row in rows if (row.get("hit_at") or {}).get(key, {}).get("transform_pair")) / max(len(rows), 1), 6)
        out[f"upstream_transform_hit_at_{k}"] = round(sum(1 for row in rows if (row.get("hit_at") or {}).get(key, {}).get("upstream_transform")) / max(len(rows), 1), 6)
        out[f"analog_hit_at_{k}"] = round(sum(1 for row in rows if (row.get("hit_at") or {}).get(key, {}).get("analog")) / max(len(rows), 1), 6)
        out[f"pair_and_analog_hit_at_{k}"] = round(sum(1 for row in rows if (row.get("hit_at") or {}).get(key, {}).get("pair_and_analog")) / max(len(rows), 1), 6)
    return out


def _hit_at(hit_rows: list[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    out = {}
    for k in (1, 3, 5, 10, 20, 50):
        top = hit_rows[:k]
        out[str(k)] = {
            "transform_pair": any(row.get("transform_pair_hit") for row in top),
            "upstream_transform": any(row.get("upstream_transform_hit") for row in top),
            "analog": any(row.get("analog_hit") for row in top),
            "pair_and_analog": any(row.get("transform_pair_hit") and row.get("analog_hit") for row in top),
        }
    return out


def _transform_pair_misses(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        hit_at = row.get("hit_at") or {}
        if not (hit_at.get("20") or {}).get("transform_pair"):
            counter[row.get("transform_pair") or "unknown"] += 1
    return dict(counter.most_common(20))


def _reference_summary(refs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reference_blocks": len(refs),
        "reference_targets": len({row.get("target_smiles") for row in refs}),
        "transform_pairs": dict(Counter(row.get("transform_pair") for row in refs).most_common(20)),
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
    lines = [
        "# CascadeRetrievalProvider Audit",
        "",
        f"- split: `{(report.get('metadata') or {}).get('split')}`",
        f"- limit: `{(report.get('metadata') or {}).get('limit')}`",
        f"- min similarity: `{(report.get('metadata') or {}).get('min_similarity')}`",
        f"- analog similarity: `{(report.get('metadata') or {}).get('analog_similarity')}`",
        "",
        "## Mode Summary",
        "",
        "| Mode | With Hits | Pair@1 | Pair@5 | Pair@20 | Analog@20 | Pair+Analog@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, payload in (report.get("modes") or {}).items():
        s = payload.get("summary") or {}
        lines.append(
            f"| `{mode}` | {s.get('with_hits_rate')} | {s.get('transform_pair_hit_at_1')} | "
            f"{s.get('transform_pair_hit_at_5')} | {s.get('transform_pair_hit_at_20')} | "
            f"{s.get('analog_hit_at_20')} | {s.get('pair_and_analog_hit_at_20')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_modes(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit v4 train-backed cascade retrieval provider on held-out blocks")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--modes", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min-similarity", type=float, default=0.20)
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    ap.add_argument("--condition-on-downstream-transform", action="store_true")
    args = ap.parse_args()
    report = audit_cascade_retrieval_provider(
        program_manifest=Path(args.program_manifest),
        split=args.split,
        output_json=Path(args.output_json),
        modes=_parse_modes(args.modes),
        limit=args.limit,
        min_similarity=args.min_similarity,
        analog_similarity=args.analog_similarity,
        condition_on_downstream_transform=args.condition_on_downstream_transform,
    )
    print(json.dumps({"provider_summary": report["provider_summary"], "modes": {k: v["summary"] for k, v in report["modes"].items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
