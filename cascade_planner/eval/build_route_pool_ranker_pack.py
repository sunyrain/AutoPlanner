"""Build same-target route-pool ranking rows from scored ChemEnzy pools."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.eval.audit_v4_heldout_block_recovery import _load_references, _route_hits


SCHEMA_VERSION = "route_pool_ranker_pack.v1"


def build_route_pool_ranker_pack(
    *,
    program_manifest: Path,
    train_ccts_run: Path,
    train_block_jsonl: Path,
    val_ccts_run: Path,
    val_block_jsonl: Path,
    test_ccts_run: Path,
    test_block_jsonl: Path,
    output_dir: Path,
    analog_similarity: float = 0.55,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_inputs = {
        "train": (train_ccts_run, train_block_jsonl),
        "val": (val_ccts_run, val_block_jsonl),
        "test": (test_ccts_run, test_block_jsonl),
    }
    outputs: dict[str, str] = {}
    reports: dict[str, Any] = {}
    for split, (ccts_run, block_jsonl) in split_inputs.items():
        refs = _load_references(program_manifest, split=split)
        rows, report = _build_split_rows(
            split=split,
            ccts_run=ccts_run,
            block_jsonl=block_jsonl,
            refs=refs,
            analog_similarity=analog_similarity,
        )
        out_path = output_dir / f"route_pool_ranker_{split}.jsonl"
        _write_jsonl(out_path, rows)
        outputs[split] = str(out_path)
        reports[split] = report
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "analog_similarity": float(analog_similarity),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "same-target route-pool ranking rows; test split is for held-out evaluation only",
        },
        "splits": reports,
        "outputs": {
            **outputs,
            "manifest": str(output_dir / "route_pool_ranker_manifest.json"),
            "report": str(output_dir / "route_pool_ranker_report.md"),
        },
    }
    (output_dir / "route_pool_ranker_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "route_pool_ranker_report.md").write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def _build_split_rows(
    *,
    split: str,
    ccts_run: Path,
    block_jsonl: Path,
    refs: dict[str, Any],
    analog_similarity: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    block_rows = _block_index(_read_jsonl(block_jsonl))
    run = json.loads(ccts_run.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    missing_block = 0
    for target_index, target in enumerate(run.get("targets") or []):
        target_smiles = canonical_smiles(str(target.get("target_smiles") or "")) or str(target.get("target_smiles") or "")
        target_refs = refs["by_target"].get(target_smiles, [])
        ref_blocks = [block for program in target_refs for block in program.get("blocks") or []]
        routes = [route for route in target.get("routes") or [] if isinstance(route, dict)]
        route_count = len(routes)
        for route in routes:
            key = _route_key(route)
            block = block_rows.get(key)
            if block is None:
                missing_block += 1
            hits = _route_hits(route, ref_blocks, analog_similarity=analog_similarity) if ref_blocks else _empty_hits(route)
            label = _route_label(hits)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "target_index": target_index,
                    "target_id": str(target.get("target_id") or route.get("target_id") or target_smiles),
                    "target_smiles": target_smiles,
                    "route_id": str(route.get("route_id") or ""),
                    "native_rank": int(route.get("native_rank") or 0),
                    "original_native_rank": int(route.get("original_native_rank", route.get("native_rank", 0)) or 0),
                    "route_count": route_count,
                    "n_steps": len(route.get("steps") or []),
                    "stock_closed": bool(route.get("stock_closed")),
                    "route_label": label,
                    "exact_hit": bool(hits.get("exact_blocks")),
                    "analog_hit": bool(hits.get("analog_blocks")),
                    "transform_consistent_analog_hit": bool(hits.get("transform_consistent_analog_blocks")),
                    "feature": _feature_row(route, block, route_count=route_count),
                }
            )
    return rows, _split_report(rows, missing_block=missing_block, reference_summary=refs["summary"])


def _feature_row(route: dict[str, Any], block: dict[str, Any] | None, *, route_count: int) -> dict[str, float]:
    native_rank = max(0, int(route.get("native_rank") or 0))
    n_steps = len(route.get("steps") or [])
    enrich = _step_enrichment_features(route)
    block_summary = (block or {}).get("block_coherence") or {}
    return {
        "native_rank": float(native_rank),
        "native_inv_rank": 1.0 / float(native_rank + 1),
        "native_rank_fraction": float(native_rank) / max(1.0, float(route_count - 1)),
        "native_score": _float(route.get("native_score")),
        "n_steps": float(n_steps),
        "n_blocks": float(max(0, n_steps - 1)),
        "stock_closed": float(bool(route.get("stock_closed"))),
        "ccts_step_any_mean": _float(route.get("ccts_v3_runtime_step_any_mean")),
        "ccts_step_any_max": _float(route.get("ccts_v3_runtime_step_any_max")),
        "ccts_step_pair_mean": _float(route.get("ccts_v3_runtime_step_pair_mean")),
        "ccts_step_pair_max": _float(route.get("ccts_v3_runtime_step_pair_max")),
        "ccts_model_mean": _float(route.get("ccts_v3_runtime_model_mean")),
        "ccts_model_max": _float(route.get("ccts_v3_runtime_model_max")),
        "ccts_best_route_evidence": _float(route.get("ccts_v3_runtime_best_route_evidence")),
        "block_route_coherence": _float(block_summary.get("route_coherence_score")),
        "block_conservative_coherence": _float(block_summary.get("conservative_route_coherence_score")),
        "block_rerank_score": _float(block_summary.get("rerank_score")),
        "block_mean": _float(block_summary.get("mean")),
        "block_min": _float(block_summary.get("min")),
        "block_max": _float(block_summary.get("max")),
        "block_low_count_lt_0_25": _float(block_summary.get("low_block_count_lt_0_25")),
        **enrich,
    }


def _step_enrichment_features(route: dict[str, Any]) -> dict[str, float]:
    sims = []
    accepted = 0
    matched = 0
    for step in route.get("steps") or []:
        ev = (step or {}).get("v4_step_evidence") or {}
        if ev.get("matched"):
            matched += 1
        if ev.get("accepted"):
            accepted += 1
        if ev.get("similarity") is not None:
            sims.append(_float(ev.get("similarity")))
    n_steps = max(1, len(route.get("steps") or []))
    return {
        "v4_step_matched_rate": float(matched) / n_steps,
        "v4_step_accepted_rate": float(accepted) / n_steps,
        "v4_step_similarity_mean": float(np.mean(sims)) if sims else 0.0,
        "v4_step_similarity_min": float(np.min(sims)) if sims else 0.0,
        "v4_step_similarity_max": float(np.max(sims)) if sims else 0.0,
    }


def _route_label(hits: dict[str, Any]) -> int:
    if hits.get("exact_blocks"):
        return 3
    if hits.get("transform_consistent_analog_blocks"):
        return 2
    if hits.get("analog_blocks"):
        return 1
    return 0


def _empty_hits(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_id": route.get("route_id"),
        "native_rank": int(route.get("native_rank") or 0),
        "exact_blocks": [],
        "analog_blocks": [],
        "transform_consistent_analog_blocks": [],
    }


def _split_report(rows: list[dict[str, Any]], *, missing_block: int, reference_summary: dict[str, Any]) -> dict[str, Any]:
    by_target = Counter(str(row.get("target_id") or "") for row in rows)
    positives = [row for row in rows if int(row.get("route_label") or 0) > 0]
    return {
        "rows": len(rows),
        "targets": len(by_target),
        "missing_block_rows": int(missing_block),
        "positive_rows": len(positives),
        "target_with_positive_rows": len({row.get("target_id") for row in positives}),
        "label_counts": dict(Counter(str(row.get("route_label")) for row in rows)),
        "reference_summary": reference_summary,
    }


def _block_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    for row in rows:
        out[_route_key(row)] = row
    return out


def _route_key(route: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(route.get("target_id") or route.get("target_smiles") or ""),
        str(route.get("route_id") or ""),
        int(route.get("original_native_rank", route.get("native_rank", 0)) or 0),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _markdown(manifest: dict[str, Any]) -> str:
    lines = ["# Route-Pool Ranker Pack", "", "## Splits", "", "| split | rows | targets | positives | targets with positives | missing block rows |", "|---|---:|---:|---:|---:|---:|"]
    for split, report in (manifest.get("splits") or {}).items():
        lines.append(
            f"| {split} | {report.get('rows')} | {report.get('targets')} | {report.get('positive_rows')} | {report.get('target_with_positive_rows')} | {report.get('missing_block_rows')} |"
        )
    lines.extend(["", "## Label Counts", "", "```json", json.dumps({k: v.get("label_counts") for k, v in (manifest.get("splits") or {}).items()}, indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build same-target route-pool ranker pack")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--train-ccts-run", required=True)
    ap.add_argument("--train-block-jsonl", required=True)
    ap.add_argument("--val-ccts-run", required=True)
    ap.add_argument("--val-block-jsonl", required=True)
    ap.add_argument("--test-ccts-run", required=True)
    ap.add_argument("--test-block-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    args = ap.parse_args()
    manifest = build_route_pool_ranker_pack(
        program_manifest=Path(args.program_manifest),
        train_ccts_run=Path(args.train_ccts_run),
        train_block_jsonl=Path(args.train_block_jsonl),
        val_ccts_run=Path(args.val_ccts_run),
        val_block_jsonl=Path(args.val_block_jsonl),
        test_ccts_run=Path(args.test_ccts_run),
        test_block_jsonl=Path(args.test_block_jsonl),
        output_dir=Path(args.output_dir),
        analog_similarity=args.analog_similarity,
    )
    print(json.dumps({"splits": manifest["splits"], "outputs": manifest["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
