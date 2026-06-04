#!/usr/bin/env python3
"""Benchmark ChemEnzy native onmt_models.bionav_one_step on a locked corpus."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import RDLogger

from cascade_planner.baselines.chem_enzy_adapter import (
    DEFAULT_VENDOR_ROOT,
    ChemEnzyBackendAdapter,
    _patch_dgl_graphbolt_optional_import,
    _patch_numpy_legacy_aliases,
    _patch_onmt_tokenizer,
    _patch_optional_easifa_import,
    _patch_optional_graphviz_import,
    _patch_torchdata_legacy_aliases,
    _vendor_pythonpath,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.cascadeboard.route_recovery import canonical_side


SCHEMA_VERSION = "chem_enzy_native_bionav_benchmark.v1"
DEFAULT_MODEL_NAME = "onmt_models.bionav_one_step"


def run_benchmark(
    *,
    src_path: Path,
    tgt_path: Path,
    meta_path: Path | None = None,
    output_dir: Path,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
    gpu: int = 0,
    topk: int = 10,
    expansion_topk: int | None = None,
    tokenizer: str = "char",
    limit: int | None = None,
    engine: str = "native",
    batch_size: int = 64,
    inference_input: str = "product",
    model_paths: list[Path] | None = None,
    save_rows: bool = True,
) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    products = _read_source_lines(src_path, limit=limit)
    targets = _read_source_lines(tgt_path, limit=limit)
    metadata = _read_meta(meta_path, limit=limit)
    if len(products) != len(targets):
        raise ValueError(f"src/tgt length mismatch: {len(products)} != {len(targets)}")
    while len(metadata) < len(products):
        metadata.append({})

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    product_smiles = [_product_smiles_from_source(source) for source in products]
    model_inputs = product_smiles if inference_input == "product" else products
    target_reactants_list = [target.replace(" ", "") for target in targets]
    if engine == "batch":
        batch_runner, model_metadata = _load_batch_bionav(
            vendor_root=vendor_root,
            gpu=gpu,
            topk=max(int(topk), int(expansion_topk or topk)),
            tokenizer=tokenizer,
            model_paths=model_paths,
        )
        batch_started = time.monotonic()
        result = batch_runner(model_inputs, batch_size=max(1, int(batch_size)))
        predictions_by_row = [[str(item) for item in row][:topk] for row in result.get("reactants", [])]
        scores_by_row = [[_safe_float(item) for item in row][:topk] for row in result.get("scores", [])]
        batch_elapsed = round(time.monotonic() - batch_started, 3)
        while len(predictions_by_row) < len(product_smiles):
            predictions_by_row.append([])
        while len(scores_by_row) < len(product_smiles):
            scores_by_row.append([])
        for idx, (product, target_reactants, meta) in enumerate(zip(product_smiles, target_reactants_list, metadata)):
            rows.append(
                score_prediction_row(
                    idx=idx,
                    product=product,
                    target_reactants=target_reactants,
                    predictions=predictions_by_row[idx],
                    scores=scores_by_row[idx],
                    metadata=meta,
                    topk=topk,
                    elapsed_s=round(batch_elapsed / max(len(product_smiles), 1), 6),
                    error="",
                )
            )
    elif engine == "native":
        one_step, model_metadata = _load_native_bionav(
            vendor_root=vendor_root,
            gpu=gpu,
            topk=max(int(topk), int(expansion_topk or topk)),
            tokenizer=tokenizer,
            model_paths=model_paths,
        )
        for idx, (product, target_reactants, meta) in enumerate(zip(product_smiles, target_reactants_list, metadata)):
            row_started = time.monotonic()
            error = ""
            predictions: list[str] = []
            scores: list[float] = []
            try:
                result = one_step.run(product, topk=topk) or {}
                predictions = [str(item) for item in (result.get("reactants") or [])][:topk]
                scores = [_safe_float(item) for item in (result.get("scores") or [])][:topk]
            except Exception as exc:  # pragma: no cover - depends on vendored runtime
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                score_prediction_row(
                    idx=idx,
                    product=product,
                    target_reactants=target_reactants,
                    predictions=predictions,
                    scores=scores,
                    metadata=meta,
                    topk=topk,
                    elapsed_s=round(time.monotonic() - row_started, 3),
                    error=error,
                )
            )
    else:
        raise ValueError(f"unsupported engine: {engine}")

    elapsed_s = round(time.monotonic() - started, 3)
    summary = summarize_scored_rows(rows, topk=topk, elapsed_s=elapsed_s)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "settings": {
            "src_path": str(src_path),
            "tgt_path": str(tgt_path),
            "meta_path": str(meta_path) if meta_path else None,
            "vendor_root": str(vendor_root),
            "model_name": DEFAULT_MODEL_NAME,
            "gpu": gpu,
            "topk": topk,
            "expansion_topk": expansion_topk or topk,
            "tokenizer": tokenizer,
            "limit": limit,
            "engine": engine,
            "batch_size": batch_size,
            "inference_input": inference_input,
            "model_paths_override": [str(path) for path in (model_paths or [])],
        },
        "model_metadata": model_metadata,
        "summary": summary,
        "rows": rows if save_rows else [],
        "contract": (
            "Native ChemEnzy BioNav one-step benchmark. Metrics are canonical exact reactant-side recall, "
            "not route-level synthesis success."
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native_bionav_benchmark_summary.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if save_rows:
        (output_dir / "native_bionav_benchmark_rows.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output_dir / "native_bionav_benchmark.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return payload


def score_prediction_row(
    *,
    idx: int,
    product: str,
    target_reactants: str,
    predictions: list[str],
    scores: list[float],
    metadata: dict[str, Any],
    topk: int,
    elapsed_s: float = 0.0,
    error: str = "",
) -> dict[str, Any]:
    target_key = canonical_side(target_reactants)
    pred_keys = [canonical_side(pred) for pred in predictions]
    top1_exact = bool(pred_keys and pred_keys[0] == target_key)
    topk_exact = target_key in pred_keys[:topk]
    ec = _metadata_ec(metadata)
    return {
        "idx": idx,
        "product": product,
        "target_reactants": target_reactants,
        "ec": ec,
        "ec1": _ec1(ec),
        "source": metadata.get("source") or "",
        "predictions": predictions[:topk],
        "scores": scores[:topk],
        "nonempty": bool(predictions),
        "top1_exact": top1_exact,
        f"top{topk}_exact": topk_exact,
        "elapsed_s": elapsed_s,
        "error": error,
    }


def summarize_scored_rows(rows: list[dict[str, Any]], *, topk: int, elapsed_s: float) -> dict[str, Any]:
    n = len(rows)
    nonempty = sum(1 for row in rows if row.get("nonempty"))
    top1 = sum(1 for row in rows if row.get("top1_exact"))
    topk_count = sum(1 for row in rows if row.get(f"top{topk}_exact"))
    errors = sum(1 for row in rows if row.get("error"))
    by_ec1: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("ec1") or "unknown")].append(row)
    for ec1, group in sorted(grouped.items()):
        group_n = len(group)
        by_ec1[ec1] = {
            "n_examples": group_n,
            "nonempty": sum(1 for row in group if row.get("nonempty")),
            "top1_exact": sum(1 for row in group if row.get("top1_exact")),
            f"top{topk}_exact": sum(1 for row in group if row.get(f"top{topk}_exact")),
            "top1_rate": _rate(sum(1 for row in group if row.get("top1_exact")), group_n),
            f"top{topk}_rate": _rate(sum(1 for row in group if row.get(f"top{topk}_exact")), group_n),
        }
    return {
        "n_examples": n,
        "nonempty": nonempty,
        "nonempty_rate": _rate(nonempty, n),
        "top1_exact": top1,
        f"top{topk}_exact": topk_count,
        "top1_rate": _rate(top1, n),
        f"top{topk}_rate": _rate(topk_count, n),
        "errors": errors,
        "elapsed_s": elapsed_s,
        "avg_elapsed_s_per_example": round(elapsed_s / n, 6) if n else None,
        "ec1": by_ec1,
        "miss_examples": [
            {
                "idx": row.get("idx"),
                "product": row.get("product"),
                "target_reactants": row.get("target_reactants"),
                "ec": row.get("ec"),
                "predictions": row.get("predictions"),
            }
            for row in rows
            if not row.get(f"top{topk}_exact")
        ][:20],
        "error_counts": dict(Counter(str(row.get("error")) for row in rows if row.get("error"))),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    topk = int(payload["settings"]["topk"])
    lines = [
        "# ChemEnzy Native BioNav Benchmark",
        "",
        f"生成时间: {payload['created_at']}",
        "",
        "## Summary",
        "",
        f"- examples: {summary['n_examples']}",
        f"- nonempty: {summary['nonempty']} ({summary['nonempty_rate']})",
        f"- top1 exact: {summary['top1_exact']} ({summary['top1_rate']})",
        f"- top{topk} exact: {summary[f'top{topk}_exact']} ({summary[f'top{topk}_rate']})",
        f"- elapsed_s: {summary['elapsed_s']}",
        "",
        "## EC1",
        "",
        f"| EC1 | n | top1 | top{topk} |",
        "| --- | ---: | ---: | ---: |",
    ]
    for ec1, row in summary.get("ec1", {}).items():
        lines.append(
            f"| {ec1} | {row['n_examples']} | {row['top1_rate']} | {row[f'top{topk}_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            payload["contract"],
            "",
        ]
    )
    return "\n".join(lines)


def _load_native_bionav(
    *,
    vendor_root: Path,
    gpu: int,
    topk: int,
    tokenizer: str,
    model_paths: list[Path] | None = None,
) -> tuple[Any, dict[str, Any]]:
    selected_types, selected_configs, config = _selected_bionav_config(
        vendor_root=vendor_root,
        gpu=gpu,
        topk=topk,
        tokenizer=tokenizer,
        model_paths=model_paths,
    )
    with _vendor_pythonpath(Path(vendor_root)):
        _patch_vendor_runtime(tokenizer)
        import torch
        from retro_planner.common.prepare_utils import prepare_single_step

        device = torch.device(f"cuda:{int(gpu)}" if int(gpu) >= 0 else "cpu")
        one_step = prepare_single_step(
            one_step_model_type=selected_types[0],
            model_configs=selected_configs[0],
            expansion_topk=max(1, int(topk)),
            device=device,
            use_filter=bool(config.get("use_filter")),
            filter_path=str(Path(vendor_root) / "retro_planner" / str(config.get("filter_path") or "")),
            keep_score=bool(config.get("keep_score", True)),
        )
        metadata = _model_metadata(selected_configs[0], tokenizer=tokenizer, device=str(device), engine="native")
        return one_step, metadata


def _load_batch_bionav(
    *,
    vendor_root: Path,
    gpu: int,
    topk: int,
    tokenizer: str,
    model_paths: list[Path] | None = None,
) -> tuple[Any, dict[str, Any]]:
    _selected_types, selected_configs, _config = _selected_bionav_config(
        vendor_root=vendor_root,
        gpu=gpu,
        topk=topk,
        tokenizer=tokenizer,
        model_paths=model_paths,
    )
    with _vendor_pythonpath(Path(vendor_root)):
        _patch_vendor_runtime(tokenizer)
        from onmt.bin.translate import load_model, run_batch_samples

        device = int(gpu) if int(gpu) >= 0 else -1
        model_config = selected_configs[0]
        opt, translator = load_model(
            model_path=list(model_config.get("model_path") or []),
            beam_size=int(model_config.get("beam_size") or max(1, int(topk))),
            topk=max(1, int(topk)),
            device=device,
            tokenizer=tokenizer,
        )

        def run_batch(products: list[str], *, batch_size: int) -> dict[str, Any]:
            return run_batch_samples(translator, opt, products, int(batch_size))

        metadata = _model_metadata(model_config, tokenizer=tokenizer, device=str(device), engine="batch")
        return run_batch, metadata


def _selected_bionav_config(
    *,
    vendor_root: Path,
    gpu: int,
    topk: int,
    tokenizer: str,
    model_paths: list[Path] | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    adapter = ChemEnzyBackendAdapter(vendor_root=vendor_root, gpu=gpu)
    failures = adapter.preflight()
    if failures:
        raise RuntimeError("; ".join(f"{item.category}:{item.message}" for item in failures))
    search_config = RouteSearchConfig(
        target_smiles="",
        max_iterations=1,
        max_depth=1,
        expansion_topk=max(1, int(topk)),
        one_step_models=[DEFAULT_MODEL_NAME],
        search_flags={
            "gpu": int(gpu),
            **({"chem_enzy_onmt_tokenizer": tokenizer} if tokenizer in {"char", "token"} else {}),
        },
    )
    config = adapter._vendor_config(search_config)
    with _vendor_pythonpath(Path(vendor_root)):
        _patch_vendor_runtime(tokenizer)
        from retro_planner.common.prepare_utils import (
            handle_one_step_config,
            handle_one_step_path,
        )

        selected_configs, _subnames, selected_types = handle_one_step_config(
            [DEFAULT_MODEL_NAME],
            config["one_step_model_configs"],
        )
        selected_configs = handle_one_step_path(selected_types, selected_configs)
        if model_paths:
            selected_configs[0] = dict(selected_configs[0])
            selected_configs[0]["model_path"] = [str(Path(path).resolve()) for path in model_paths]
    return list(selected_types), list(selected_configs), config


def _patch_vendor_runtime(tokenizer: str) -> None:
    _patch_numpy_legacy_aliases()
    _patch_torchdata_legacy_aliases()
    _patch_dgl_graphbolt_optional_import()
    _patch_optional_easifa_import(False)
    _patch_optional_graphviz_import(False)
    import retro_planner.api as api

    if tokenizer in {"char", "token"}:
        _patch_onmt_tokenizer(api, tokenizer)


def _model_metadata(model_config: dict[str, Any], *, tokenizer: str, device: str, engine: str) -> dict[str, Any]:
    return {
        "model_name": DEFAULT_MODEL_NAME,
        "model_paths": list(model_config.get("model_path") or []),
        "beam_size": model_config.get("beam_size"),
        "tokenizer": tokenizer,
        "device": device,
        "engine": engine,
    }


def _read_source_lines(path: Path, *, limit: int | None = None) -> list[str]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(text)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _read_meta(path: Path | None, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _product_smiles_from_source(source: str) -> str:
    text = str(source or "").strip()
    if "<product>" not in text:
        return text.replace(" ", "")
    after = text.split("<product>", 1)[1].strip()
    return after.replace(" ", "")


def _metadata_ec(metadata: dict[str, Any]) -> str:
    ec = metadata.get("ec")
    if ec:
        return str(ec)
    ecs = metadata.get("ec_numbers")
    if isinstance(ecs, list) and ecs:
        return str(ecs[0])
    return "unknown"


def _ec1(ec: str) -> str:
    text = str(ec or "unknown")
    return text.split(".", 1)[0] if "." in text else text


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(num: int, den: int) -> float:
    return round(num / max(den, 1), 6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--tgt", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--expansion-topk", type=int)
    parser.add_argument("--tokenizer", choices=["char", "token", "pretokenized"], default="char")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--engine", choices=["native", "batch"], default="native")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-input", choices=["product", "source"], default="product")
    parser.add_argument("--model", action="append", type=Path, help="Override ONMT checkpoint path. Repeat for ensemble.")
    parser.add_argument("--no-save-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_benchmark(
        src_path=args.src,
        tgt_path=args.tgt,
        meta_path=args.meta,
        output_dir=args.output_dir,
        vendor_root=args.vendor_root,
        gpu=args.gpu,
        topk=args.topk,
        expansion_topk=args.expansion_topk,
        tokenizer=args.tokenizer,
        limit=args.limit,
        engine=args.engine,
        batch_size=args.batch_size,
        inference_input=args.inference_input,
        model_paths=args.model,
        save_rows=not args.no_save_rows,
    )


if __name__ == "__main__":
    main()
