"""Build and run small external benchmark smokes for reservoir distillation."""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.live_benchmark import _build_stock_checker
from cascade_planner.eval.chem_enzy_broad_union import _chem_route_stock_closed
from cascade_planner.route_tree.bounded_reservoir import _route_runtime_stock_closed


PAROUTES_RECORD = "https://zenodo.org/api/records/6275421/files/{name}/content"
SYNTHARENA_USPTO_190 = "https://syntharena.ischemist.com/benchmarks/cmisbzsr30000xvdd613ymmbx"
SYNTHARENA_TARGET = SYNTHARENA_USPTO_190 + "/{target_path}"
PAROUTES_N1_MANUAL_TARGETS = Path("targets_n1.txt")
PAROUTES_N1_MANUAL_REFS = Path("ref_routes_n1.json")
PAROUTES_N5_MANUAL_TARGETS = Path("targets_n5.txt")
PAROUTES_N5_MANUAL_REFS = Path("ref_routes_n5.json")
PAROUTES_MANUAL_SOURCES = {
    "n1": (
        (PAROUTES_N1_MANUAL_TARGETS, PAROUTES_N1_MANUAL_REFS),
        (Path("data_external/paroutes/targets_n1.txt"), Path("data_external/paroutes/ref_routes_n1.json")),
        (Path("data_external/paroutes/n1-targets.txt"), Path("data_external/paroutes/ref_routes_n1.json")),
    ),
    "n5": (
        (PAROUTES_N5_MANUAL_TARGETS, PAROUTES_N5_MANUAL_REFS),
        (Path("data_external/paroutes/targets_n5.txt"), Path("data_external/paroutes/ref_routes_n5.json")),
        (Path("data_external/paroutes/n5-targets.txt"), Path("data_external/paroutes/ref_routes_n5.json")),
    ),
}
FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
DEFAULT_BASELINE_ENV = [
    "AUTOPLANNER_CASCADE_SOURCE_POLICY=results/shared/controller_v2_20260512/fullrun/train_v8/source_policy/cascade_source_policy.pt",
    "AUTOPLANNER_ROUTE_TREE_POLICY=results/shared/controller_v2_20260512/fullrun/train_v8/open_leaf_policy/stock_aware_search_policy.pt",
    "AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH=retrochimera:0,chemtemplates:0",
    "AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL=1",
    "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
]
MAX_PLANNER_DEPTH = 8
SUMMARY_COMMAND_STAGES = {"external_smoke", "external_append_only"}
SUPPORTED_CONFIGS = {"C", "C_CHEMSTEP", "C_ORACLE", "C_CHEMSTEP_ORACLE", "D", "D_CHEMSTEP", "D_APPEND"}
SUPPORTED_CONFIGS |= {"D_FILTER", "D_TOP10_FILTER"}
PAIRED_DELTA_METRICS = (
    "plan_rate",
    "strict_stock_solve_any",
    "candidate_gt_reactant_in_pool",
    "candidate_exact_reaction_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
    "avg_time_per_target_s",
    "avg_route_count",
)
COVERAGE_DELTA_METRICS = (
    "strict_stock_solve_any",
    "candidate_gt_reactant_in_pool",
    "candidate_exact_reaction_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
)


def build_external_smokes(
    *,
    output_dir: Path,
    controller_path: Path,
    native_payload: Path | None = None,
    limit: int = 3,
    fetch: bool = True,
    configs: tuple[str, ...] | list[str] = ("D",),
    enable_source_override: bool = False,
    prebuild_native_payload: bool = False,
    native_iterations: int = 10,
    native_max_depth: int = 6,
    native_expansion_topk: int = 50,
    native_gpu: int = -1,
    native_stocks: tuple[str, ...] | list[str] | None = None,
    native_one_step_models: tuple[str, ...] | list[str] | None = None,
    trust_native_stock: bool = False,
    offset: int = 0,
    datasets_filter: tuple[str, ...] | list[str] | None = None,
    uspto_cache_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    source_dir = output_dir / "sources"
    bench_dir = output_dir / "benchmarks"
    source_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    configs = _normalize_configs(configs or ("D",))

    datasets = []
    offset = max(0, int(offset))
    requested_datasets = {str(item) for item in datasets_filter} if datasets_filter else None

    def should_build(label: str) -> bool:
        return requested_datasets is None or label in requested_datasets

    if should_build("paroutes_n1"):
        datasets.append(_build_paroutes_split(source_dir, bench_dir, "n1", limit=limit, fetch=fetch, offset=offset))
    if should_build("paroutes_n5"):
        datasets.append(_build_paroutes_split(source_dir, bench_dir, "n5", limit=limit, fetch=fetch, offset=offset))
    if should_build("uspto_190"):
        datasets.append(
            _build_uspto190(
                source_dir,
                bench_dir,
                limit=limit,
                fetch=fetch,
                offset=offset,
                uspto_cache_dir=uspto_cache_dir,
            )
        )
    if should_build("bionavi_like"):
        datasets.append(_build_bionavi_like(source_dir, bench_dir, limit=limit, fetch=fetch, offset=offset))

    commands = []
    runs = {}
    shared_native_payloads: dict[str, Path] = {}
    for dataset in datasets:
        if not dataset.get("ready"):
            continue
        dataset_label = str(dataset["label"])
        for config in configs:
            label = dataset_label if configs == ("D",) else f"{config}_{dataset_label}"
            run_dir = output_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)
            run_path = run_dir / "run.json"
            trace_path = run_dir / "run_trace.jsonl"
            if config == "D_APPEND":
                config_native_payload = shared_native_payloads.get(dataset_label)
                if config_native_payload is None:
                    config_native_payload = run_dir / "native_reservoir.json"
                    commands.append(
                        _native_payload_command(
                            config=config,
                            label=label,
                            dataset_label=dataset_label,
                            benchmark=Path(str(dataset["benchmark"])),
                            output=config_native_payload,
                            native_iterations=native_iterations,
                            native_max_depth=native_max_depth,
                            native_expansion_topk=native_expansion_topk,
                            native_gpu=native_gpu,
                            native_stocks=native_stocks,
                            native_one_step_models=native_one_step_models,
                        )
                    )
                c_label = dataset_label if configs == ("C",) else f"C_{dataset_label}"
                c_run_path = output_dir / c_label / "run.json"
                union_report_path = run_dir / "append_union_report.json"
                union_markdown_path = run_dir / "append_union_report.md"
                runs[label] = str(run_path)
                commands.append(
                    {
                        "stage": "external_append_only",
                        "config": config,
                        "split": label,
                        "dataset_label": dataset_label,
                        "outputs": {
                            "run": str(run_path),
                            "native_payload": str(config_native_payload),
                            "autoplanner": str(c_run_path),
                            "union_report": str(union_report_path),
                            "markdown": str(union_markdown_path),
                        },
                        "cmd": " ".join(
                            [
                                "PYTHONPATH=.",
                                "python -m cascade_planner.eval.chem_enzy_broad_union",
                                f"--benchmark {dataset['benchmark']}",
                                f"--chem-enzy {config_native_payload}",
                                f"--autoplanner {c_run_path}",
                                f"--output {union_report_path}",
                                f"--markdown {union_markdown_path}",
                                "--native-topk 5",
                                "--native-selection rank_plus_stock",
                                f"--synthesize-output {run_path}",
                            ]
                        ),
                        "description": f"append-only external reservoir synthesis for {dataset_label}",
                    }
                )
                continue
            config_native_payload = native_payload
            config_oracle_payload = None
            if _config_uses_cascade_oracle(config):
                config_native_payload = shared_native_payloads.get(dataset_label)
                if config_native_payload is None:
                    config_native_payload = run_dir / "native_reservoir.json"
                    shared_native_payloads[dataset_label] = config_native_payload
                    commands.append(
                        _native_payload_command(
                            config=config,
                            label=label,
                            dataset_label=dataset_label,
                            benchmark=Path(str(dataset["benchmark"])),
                            output=config_native_payload,
                            native_iterations=native_iterations,
                            native_max_depth=native_max_depth,
                            native_expansion_topk=native_expansion_topk,
                            native_gpu=native_gpu,
                            native_stocks=native_stocks,
                            native_one_step_models=native_one_step_models,
                        )
                    )
                config_oracle_payload = run_dir / "cascade_oracle_payload.json"
                commands.append(
                    _cascade_oracle_payload_command(
                        config=config,
                        label=label,
                        dataset_label=dataset_label,
                        native_payload=config_native_payload,
                        output=config_oracle_payload,
                    )
                )
            elif _config_uses_bounded_reservoir(config) and prebuild_native_payload:
                config_native_payload = shared_native_payloads.get(dataset_label)
                if config_native_payload is None:
                    config_native_payload = run_dir / "native_reservoir.json"
                    shared_native_payloads[dataset_label] = config_native_payload
                    commands.append(
                        _native_payload_command(
                            config=config,
                            label=label,
                            dataset_label=dataset_label,
                            benchmark=Path(str(dataset["benchmark"])),
                            output=config_native_payload,
                            native_iterations=native_iterations,
                            native_max_depth=native_max_depth,
                            native_expansion_topk=native_expansion_topk,
                            native_gpu=native_gpu,
                            native_stocks=native_stocks,
                            native_one_step_models=native_one_step_models,
                        )
                    )
            extra_env = _config_env(
                config=config,
                controller_path=controller_path,
                native_payload=config_native_payload,
                oracle_payload=config_oracle_payload,
                enable_source_override=enable_source_override,
                trust_native_stock=trust_native_stock,
            )
            cmd = [
                "PYTHONPATH=.",
                "python -m cascade_planner.eval.run_live_benchmark_parallel",
                f"--bench {dataset['benchmark']}",
                f"--output {run_path}",
                "--model results/shared/skeleton_inpainter/best.pt",
                "--search-mode route_tree",
                "--check-stock",
                "--workers 3",
                "--device cpu",
                "--n-results 3",
                "--n-candidates-per-skeleton 1",
                "--skeleton-samples 1",
                f"--trace-output {trace_path}",
                f"--log-dir {run_dir / 'parallel_logs'}",
                "--target-log none",
            ]
            for item in extra_env:
                cmd.append(f"--extra-env {item}")
            runs[label] = str(run_path)
            commands.append(
                {
                    "stage": "external_smoke",
                    "config": config,
                    "split": label,
                    "dataset_label": dataset_label,
                    "outputs": {"run": str(run_path), "trace": str(trace_path)},
                    "cmd": " ".join(cmd),
                    "description": f"external smoke for {config} {dataset_label}",
                }
            )

    manifest = {
        "schema_version": "reservoir_external_smoke_manifest.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "controller_path": str(controller_path),
        "datasets": datasets,
        "configs": list(configs),
        "offset": offset,
        "datasets_filter": list(datasets_filter or []),
        "enable_source_override": bool(enable_source_override),
        "prebuild_native_payload": bool(prebuild_native_payload),
        "trust_native_stock": bool(trust_native_stock),
        "native_payload_settings": {
            "iterations": int(native_iterations),
            "max_depth": int(native_max_depth),
            "expansion_topk": int(native_expansion_topk),
            "gpu": int(native_gpu),
            "stocks": list(native_stocks or []),
            "one_step_models": list(native_one_step_models or []),
        },
        "commands": commands,
        "runs": runs,
        "notes": [
            "PaRoutes and BioNavi-like smokes are target-only when route annotations are not present.",
            "USPTO-190 routes are parsed from public SynthArena acceptable-route pages.",
            "These smokes verify external benchmark execution, not full promotion recovery gates.",
        ],
    }
    path = output_dir / "external_smoke_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_external_smokes(
    *,
    manifest_path: Path,
    log_dir: Path,
    skip_existing_native_payload: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, command in enumerate(manifest.get("commands") or [], 1):
        log_path = log_dir / f"external_smoke_{idx}_{command['split']}.log"
        native_payload = ((command.get("outputs") or {}).get("native_payload") or "")
        if (
            skip_existing_native_payload
            and command.get("stage") == "external_native_payload"
            and native_payload
            and Path(native_payload).exists()
        ):
            results.append({
                "index": idx,
                "split": command["split"],
                "returncode": 0,
                "elapsed_s": 0.0,
                "log": str(log_path),
                "outputs": command.get("outputs") or {},
                "skipped": True,
                "skip_reason": "native_payload_exists",
            })
            continue
        t0 = time.monotonic()
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(command["cmd"], shell=True, cwd=Path.cwd(), text=True, stdout=fh, stderr=subprocess.STDOUT)
        results.append({
            "index": idx,
            "split": command["split"],
            "returncode": int(proc.returncode),
            "elapsed_s": round(time.monotonic() - t0, 3),
            "log": str(log_path),
            "outputs": command.get("outputs") or {},
        })
    report = {
        "schema_version": "reservoir_external_smoke_run.v1",
        "manifest": str(manifest_path),
        "results": results,
        "failed": [row for row in results if row["returncode"] != 0],
    }
    report_path = log_dir / "external_smoke_run_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["failed"]:
        raise RuntimeError(f"external smokes failed: {report_path}")
    return report


def _normalize_configs(configs: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    requested = []
    for raw in configs or ("D",):
        config = str(raw or "D").upper()
        if config not in SUPPORTED_CONFIGS:
            raise ValueError(f"unsupported external reservoir config: {config}")
        if config not in requested:
            requested.append(config)
    if "D_APPEND" in requested and "C" not in requested:
        raise ValueError("D_APPEND requires config C in the same manifest so the student-only run can be frozen")
    ordered = [
        config
        for config in (
            "C",
            "C_CHEMSTEP",
            "C_ORACLE",
            "C_CHEMSTEP_ORACLE",
            "D",
            "D_FILTER",
            "D_TOP10_FILTER",
            "D_CHEMSTEP",
            "D_APPEND",
        )
        if config in requested
    ]
    return tuple(ordered or ["D"])


def _native_payload_command(
    *,
    config: str,
    label: str,
    dataset_label: str,
    benchmark: Path,
    output: Path,
    native_iterations: int,
    native_max_depth: int,
    native_expansion_topk: int,
    native_gpu: int = -1,
    native_stocks: tuple[str, ...] | list[str] | None = None,
    native_one_step_models: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    cmd = [
        "PYTHONPATH=.",
        "python scripts/run_chem_enzy_smoke.py",
        f"--benchmark {benchmark}",
        f"--output {output}",
        f"--iterations {int(native_iterations)}",
        f"--max-depth {int(native_max_depth)}",
        f"--expansion-topk {int(native_expansion_topk)}",
        f"--gpu {int(native_gpu)}",
    ]
    for stock in _native_stocks_for_dataset(dataset_label, native_stocks):
        cmd.append(f"--stock {stock}")
    for model in native_one_step_models or ():
        cmd.append(f"--one-step-model {model}")
    return {
        "stage": "external_native_payload",
        "config": config,
        "split": f"{label}_native_payload",
        "dataset_label": dataset_label,
        "outputs": {"native_payload": str(output)},
        "cmd": " ".join(cmd),
        "description": f"prebuild native ChemEnzy reservoir payload for {dataset_label}",
    }


def _native_stocks_for_dataset(
    dataset_label: str,
    native_stocks: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if native_stocks:
        return tuple(str(stock) for stock in native_stocks)
    if dataset_label == "paroutes_n1":
        return ("PaRotes_n1-stock",)
    if dataset_label == "paroutes_n5":
        return ("PaRotes_n5-stock",)
    return ()


def _cascade_oracle_payload_command(
    *,
    config: str,
    label: str,
    dataset_label: str,
    native_payload: Path,
    output: Path,
) -> dict[str, Any]:
    return {
        "stage": "external_cascade_oracle_payload",
        "config": config,
        "split": f"{label}_cascade_oracle_payload",
        "dataset_label": dataset_label,
        "outputs": {
            "native_payload": str(native_payload),
            "cascade_oracle_payload": str(output),
        },
        "cmd": " ".join(
            [
                "PYTHONPATH=.",
                "python -m cascade_planner.eval.build_cascade_oracle_payload",
                f"--native-payload {native_payload}",
                f"--output {output}",
                "--topk 5",
                "--selection rank_plus_stock",
            ]
        ),
        "description": f"build cascade oracle value payload for {dataset_label}",
    }


def summarize_external_smokes(*, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = _load_json(output_dir / "external_smoke_manifest.json")
    run_report = _load_json(output_dir / "logs" / "external_smoke_run_report.json")
    if not run_report:
        run_report = _load_json(output_dir / "logs" / "manifest_command_run_report.json")
    rows = []
    external_commands = [
        command
        for command in manifest.get("commands") or []
        if command.get("stage") in SUMMARY_COMMAND_STAGES
    ]
    command_by_label = {command.get("split"): command for command in external_commands}
    dataset_by_label = {dataset.get("label"): dataset for dataset in manifest.get("datasets") or []}
    labels = list(command_by_label) or list(dataset_by_label)
    stock_checker = _summary_stock_checker()
    for label in labels:
        command = command_by_label.get(label) or {}
        dataset = dataset_by_label.get(command.get("dataset_label") or label) or {}
        run_path = _summary_run_path(output_dir, str(label), command)
        run = _load_json(run_path)
        broad = _broad_reservoir_counts(run)
        native_payload = _summary_native_payload_path(output_dir, str(label), command)
        native_counts = _native_payload_counts(native_payload, run=run, stock_checker=stock_checker)
        run_summary = _run_summary_fields(run)
        rows.append({
            "label": label,
            "config": command.get("config") or "D",
            "dataset_label": command.get("dataset_label") or dataset.get("label"),
            "ready": bool(dataset.get("ready")),
            "benchmark": dataset.get("benchmark"),
            "source": dataset.get("source"),
            "route_annotations": bool(dataset.get("route_annotations")),
            "n_benchmark_rows": dataset.get("n_rows"),
            "run_exists": run_path.exists(),
            "n_run_targets": len(run.get("targets") or []),
            "plan_rate": run_summary.get("plan_rate"),
            "strict_stock_solve_any": run_summary.get("strict_stock_solve_any"),
            "candidate_exact_reaction_in_pool": run_summary.get("candidate_exact_reaction_in_pool"),
            "candidate_gt_reactant_in_pool": run_summary.get("candidate_gt_reactant_in_pool"),
            "exact_reaction_in_route_pool": run_summary.get("exact_reaction_in_route_pool"),
            "gt_reactant_in_route_pool": run_summary.get("gt_reactant_in_route_pool"),
            "avg_time_per_target_s": run_summary.get("avg_time_per_target_s"),
            "avg_route_count": run_summary.get("avg_route_count"),
            "route_tree_runtime_bottleneck_counts": dict(run_summary.get("route_tree_runtime_bottleneck_counts") or {}),
            "route_tree_source_latency_ms": dict(run_summary.get("route_tree_source_latency_ms") or {}),
            **broad,
            **native_counts,
            "error": dataset.get("error"),
        })
    required = set(command_by_label) if command_by_label else {"paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"}
    executed = {str(row["label"]) for row in rows if row["run_exists"] and row["n_run_targets"] > 0}
    summary = {
        "schema_version": "reservoir_external_smoke_summary.v1",
        "output_dir": str(output_dir),
        "rows": rows,
        "paired_config_deltas": _paired_config_deltas(rows),
        "run_report": run_report,
        "required": sorted(required),
        "executed": sorted(executed),
        "ready": required.issubset(executed) and not run_report.get("failed"),
    }
    path = output_dir / "external_smoke_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _paired_config_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row.get("dataset_label") or row.get("label") or "")
        config = str(row.get("config") or "")
        if dataset and config:
            by_dataset.setdefault(dataset, {})[config] = row
    deltas = []
    for dataset, configs in sorted(by_dataset.items()):
        baseline = configs.get("C")
        if not baseline:
            continue
        for candidate_config in (
            "C_CHEMSTEP",
            "C_ORACLE",
            "C_CHEMSTEP_ORACLE",
            "D",
            "D_FILTER",
            "D_TOP10_FILTER",
            "D_CHEMSTEP",
            "D_APPEND",
        ):
            candidate = configs.get(candidate_config)
            if not candidate:
                continue
            metric_delta = {metric: _delta(candidate.get(metric), baseline.get(metric)) for metric in PAIRED_DELTA_METRICS}
            gains = []
            losses = []
            for metric in COVERAGE_DELTA_METRICS:
                delta = metric_delta.get(metric)
                if delta is None:
                    continue
                if delta > 0:
                    gains.append(metric)
                elif delta < 0:
                    losses.append(metric)
            broad_routes = int(candidate.get("broad_reservoir_routes") or 0)
            if candidate_config == "D_APPEND":
                if losses:
                    likely_cause = "append_only_union_regression"
                elif gains and broad_routes:
                    likely_cause = "append_only_bounded_reservoir_gain"
                elif gains:
                    likely_cause = "append_only_controller_or_search_gain"
                else:
                    likely_cause = "append_only_no_coverage_change"
            elif "ORACLE" in candidate_config and gains:
                likely_cause = "cascade_oracle_value_gain"
            elif "CHEMSTEP" in candidate_config and gains:
                likely_cause = "chem_enzy_onestep_proposal_gain"
            elif losses and broad_routes:
                likely_cause = "bounded_reservoir_or_search_path"
            elif losses:
                likely_cause = "search_path_or_source_policy_variance"
            elif gains and broad_routes:
                likely_cause = "bounded_reservoir_gain"
            elif gains:
                likely_cause = "controller_or_search_gain"
            else:
                likely_cause = "no_coverage_change"
            deltas.append(
                {
                    "dataset_label": dataset,
                    "baseline_label": baseline.get("label"),
                    "candidate_label": candidate.get("label"),
                    "candidate_config": candidate_config,
                    "metric_deltas": metric_delta,
                    "coverage_gains": gains,
                    "coverage_losses": losses,
                    "candidate_broad_reservoir_routes": broad_routes,
                    "candidate_broad_reservoir_runtime_stock_routes": int(
                        candidate.get("broad_reservoir_runtime_stock_routes") or 0
                    ),
                    "likely_change_cause": likely_cause,
                }
            )
    return deltas


def _summary_run_path(output_dir: Path, label: str, command: dict[str, Any]) -> Path:
    run = ((command.get("outputs") or {}).get("run") or "").strip()
    if run:
        return Path(run)
    return output_dir / label / "run.json"


def _summary_native_payload_path(output_dir: Path, label: str, command: dict[str, Any]) -> Path:
    native_payload = ((command.get("outputs") or {}).get("native_payload") or "").strip()
    if native_payload:
        return Path(native_payload)
    return output_dir / label / "native_reservoir.json"


def _run_summary_fields(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary") or {}
    synthetic = summary.get("synthesized_union") or {}
    targets = [row for row in run.get("targets") or [] if isinstance(row, dict)]

    def pick(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    out = {
        "plan_rate": pick(summary.get("plan_rate"), _target_metric_rate(targets, "plan"), _target_plan_rate_from_routes(targets)),
        "strict_stock_solve_any": pick(
            summary.get("strict_stock_solve_any"),
            _target_metric_rate(targets, "strict_stock_solve_any"),
            synthetic.get("stock_rate"),
        ),
        "candidate_exact_reaction_in_pool": pick(
            summary.get("candidate_exact_reaction_in_pool"),
            _target_recovery_rate(targets, "candidate_exact_reaction_in_pool"),
        ),
        "candidate_gt_reactant_in_pool": pick(
            summary.get("candidate_gt_reactant_in_pool"),
            _target_recovery_rate(targets, "candidate_gt_reactant_in_pool"),
        ),
        "exact_reaction_in_route_pool": pick(
            summary.get("exact_reaction_in_route_pool"),
            _target_recovery_rate(targets, "exact_reaction_in_route_pool"),
            synthetic.get("exact_reaction_in_route_pool"),
        ),
        "gt_reactant_in_route_pool": pick(
            summary.get("gt_reactant_in_route_pool"),
            _target_recovery_rate(targets, "gt_reactant_in_route_pool"),
            synthetic.get("gt_reactant_in_route_pool"),
        ),
        "avg_time_per_target_s": summary.get("avg_time_per_target_s"),
        "avg_route_count": pick(summary.get("avg_route_count"), _target_avg_route_count(targets), synthetic.get("avg_route_count")),
        "route_tree_runtime_bottleneck_counts": dict(summary.get("route_tree_runtime_bottleneck_counts") or {}),
        "route_tree_source_latency_ms": dict(summary.get("route_tree_source_latency_ms") or {}),
    }
    return out


def _target_metric_rate(targets: list[dict[str, Any]], field: str) -> float | None:
    if not targets:
        return None
    seen = False
    hits = 0
    for row in targets:
        metrics = row.get("metrics") or {}
        if field not in metrics:
            continue
        seen = True
        hits += int(bool(metrics.get(field)))
    return hits / len(targets) if seen else None


def _target_recovery_rate(targets: list[dict[str, Any]], field: str) -> float | None:
    if not targets:
        return None
    seen = False
    hits = 0
    for row in targets:
        recovery = row.get("route_recovery") or {}
        if field not in recovery:
            continue
        seen = True
        hits += int(bool(recovery.get(field)))
    return hits / len(targets) if seen else None


def _target_plan_rate_from_routes(targets: list[dict[str, Any]]) -> float | None:
    if not targets:
        return None
    return sum(1 for row in targets if (((row.get("planner_output") or {}).get("routes") or []))) / len(targets)


def _target_avg_route_count(targets: list[dict[str, Any]]) -> float | None:
    if not targets:
        return None
    return sum(len(((row.get("planner_output") or {}).get("routes") or [])) for row in targets) / len(targets)


def _broad_reservoir_counts(run: dict[str, Any]) -> dict[str, int]:
    broad_targets = 0
    broad_routes = 0
    broad_metadata_stock_routes = 0
    broad_runtime_stock_routes = 0
    for row in run.get("targets") or []:
        target_broad = 0
        for route in ((row.get("planner_output") or {}).get("routes") or []):
            broad = route.get("broad_reservoir")
            if not broad:
                broad = (((route.get("explanation") or {}).get("uncertainty_table") or {}).get("broad_reservoir"))
            if not broad:
                continue
            target_broad += 1
            broad_routes += 1
            broad_metadata_stock_routes += int(bool(broad.get("stock_closed")))
            broad_runtime_stock_routes += int(bool((route.get("metrics") or {}).get("strict_stock_solve")))
        broad_targets += int(target_broad > 0)
    return {
        "broad_reservoir_targets": broad_targets,
        "broad_reservoir_routes": broad_routes,
        "broad_reservoir_stock_routes": broad_metadata_stock_routes,
        "broad_reservoir_metadata_stock_routes": broad_metadata_stock_routes,
        "broad_reservoir_runtime_stock_routes": broad_runtime_stock_routes,
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(candidate: Any, baseline: Any) -> float | None:
    cand = _safe_float(candidate)
    base = _safe_float(baseline)
    if cand is None or base is None:
        return None
    return round(cand - base, 6)


def _summary_stock_checker():
    try:
        return _build_stock_checker(True)
    except Exception:
        return None


def _native_payload_counts(path: Path, *, run: dict[str, Any], stock_checker) -> dict[str, int | bool]:
    target = _first_run_target(run)
    out = {
        "native_payload_exists": bool(path.exists()),
        "native_payload_routes": 0,
        "native_payload_metadata_stock_routes": 0,
        "native_payload_runtime_stock_routes": 0,
    }
    if not path.exists() or not target:
        return out
    payload = _load_json(path)
    routes = _native_payload_routes_for_target(payload, target)
    out["native_payload_routes"] = len(routes)
    out["native_payload_metadata_stock_routes"] = sum(1 for route in routes if _native_route_metadata_stock_closed(route))
    if stock_checker is not None:
        out["native_payload_runtime_stock_routes"] = sum(
            1 for route in routes if _route_runtime_stock_closed(route, target=target, stock_checker=stock_checker)
        )
    return out


def _first_run_target(run: dict[str, Any]) -> str:
    for row in run.get("targets") or []:
        target = row.get("target_smiles")
        if target:
            return str(target)
    return ""


def _native_payload_routes_for_target(payload: dict[str, Any], target: str) -> list[dict[str, Any]]:
    targets = payload.get("targets") or []
    for row in targets:
        if str(row.get("target_smiles") or "") == target:
            return list(row.get("routes") or [])
    if len(targets) == 1:
        return list((targets[0] or {}).get("routes") or [])
    return []


def _native_route_metadata_stock_closed(route: dict[str, Any]) -> bool:
    try:
        if route.get("steps"):
            return bool(_chem_route_stock_closed(route))
    except Exception:
        pass
    statuses = []
    for step in route.get("steps") or []:
        stock_status = step.get("stock_status") or {}
        if isinstance(stock_status, dict):
            statuses.extend(bool(value) for value in stock_status.values())
    if statuses:
        return all(statuses)
    metrics = route.get("metrics") or {}
    return bool(route.get("stock_closed") or metrics.get("strict_stock_solve") or metrics.get("strict_stock_solve_any"))


def _config_env(
    *,
    config: str,
    controller_path: Path,
    native_payload: Path | None,
    oracle_payload: Path | None = None,
    enable_source_override: bool,
    trust_native_stock: bool = False,
) -> list[str]:
    extra_env = [
        *DEFAULT_BASELINE_ENV,
        f"AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER={controller_path}",
    ]
    if enable_source_override:
        extra_env.append("AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE=1")
    if "CHEMSTEP" in config:
        extra_env.extend(
            [
                "AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS=1",
                "AUTOPLANNER_CHEMENZY_ONESTEP_TOPK=50",
                "AUTOPLANNER_CHEMENZY_ONESTEP_MIN_BUDGET=5",
                "AUTOPLANNER_CHEMENZY_ONESTEP_MODELS=graphfp_models.USPTO-full_remapped,onmt_models.bionav_one_step",
                "AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS=chem_enzy_onestep:50",
                "AUTOPLANNER_ROUTE_TREE_ROOT_PROPOSAL_BUDGET=16",
                "AUTOPLANNER_ROUTE_TREE_MIN_BRANCH_FACTOR=12",
                "AUTOPLANNER_ROUTE_TREE_MIN_EXPANSION_BUDGET=96",
                "AUTOPLANNER_ROUTE_TREE_SOURCE_DIVERSE_BRANCH_BONUS=1",
                "AUTOPLANNER_ROUTE_TREE_BRANCH_FACTOR_CAP=16",
                "AUTOPLANNER_ROUTE_TREE_SOURCE_RESERVE_MAX_SCORE_GAP=2.0",
                "AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER=2",
                "AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES=1",
                "AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS=1",
                "AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX=2",
                "AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK=1",
                "AUTOPLANNER_ROUTE_TREE_FINAL_RERANK=1",
                "AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH=retrochimera:0,chemtemplates:0,chem_enzy_onestep:0",
            ]
        )
    if _config_uses_cascade_oracle(config):
        extra_env.extend(
            [
                "AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE=1",
                "AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT=1.25",
                "AUTOPLANNER_ROUTE_TREE_RESULT_POOL_MULTIPLIER=2",
                "AUTOPLANNER_ROUTE_TREE_KEEP_SOLVED_ROUTE_ALTERNATIVES=1",
                "AUTOPLANNER_ROUTE_TREE_RETURN_CONTRAST_FALLBACKS=1",
                "AUTOPLANNER_ROUTE_TREE_CONTRAST_FALLBACK_MAX=2",
                "AUTOPLANNER_ROUTE_TREE_QUALITY_RESULT_RERANK=1",
                "AUTOPLANNER_ROUTE_TREE_FINAL_RERANK=1",
            ]
        )
        if oracle_payload is not None:
            extra_env.append(f"AUTOPLANNER_CASCADE_ORACLE_PAYLOAD={oracle_payload}")
    if _config_uses_bounded_reservoir(config):
        native_topk = 10 if str(config or "").upper() == "D_TOP10_FILTER" else 5
        extra_env.extend(
            [
                "AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1",
                f"AUTOPLANNER_RESERVOIR_NATIVE_TOPK={native_topk}",
                "AUTOPLANNER_RESERVOIR_SELECTION=rank_plus_stock",
            ]
        )
        if str(config or "").upper() in {"D_FILTER", "D_TOP10_FILTER"}:
            extra_env.append("AUTOPLANNER_RESERVOIR_QUALITY_FILTER=1")
        if native_payload is not None:
            extra_env.append(f"AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD={native_payload}")
        if trust_native_stock:
            extra_env.append("AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK=1")
    return extra_env


def _config_uses_bounded_reservoir(config: str) -> bool:
    return str(config or "").upper() in {"D", "D_FILTER", "D_TOP10_FILTER", "D_CHEMSTEP"}


def _config_uses_cascade_oracle(config: str) -> bool:
    return "ORACLE" in str(config or "").upper()


def _build_paroutes_split(
    source_dir: Path,
    bench_dir: Path,
    split: str,
    *,
    limit: int,
    fetch: bool,
    offset: int = 0,
) -> dict[str, Any]:
    manual_paths = _paroutes_manual_paths(split)
    if manual_paths is not None:
        manual_targets, manual_refs = manual_paths
        rows = _build_paroutes_reference_rows(
            targets_path=manual_targets,
            refs_path=manual_refs,
            split=split,
            limit=limit,
            offset=offset,
        )
        bench = bench_dir / f"paroutes_{split}_smoke.json"
        bench.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "label": f"paroutes_{split}",
            "source": str(manual_targets),
            "reference_source": str(manual_refs),
            "benchmark": str(bench),
            "n_rows": len(rows),
            "ready": bool(rows),
            "route_annotations": any(row.get("gt_route") for row in rows),
            "error": "",
        }
    name = f"{split}-targets.txt"
    source = source_dir / name
    error = ""
    if fetch and not source.exists():
        try:
            _download(PAROUTES_RECORD.format(name=name), source, timeout=120)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            _copy_cached_source(name, source)
    rows = []
    if source.exists():
        for idx, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            if idx < offset:
                continue
            smi = line.strip()
            if not smi:
                continue
            rows.append(_target_only_row(f"paroutes_{split}_{idx}", smi, depth=3, route_domain="all_chemical"))
            if len(rows) >= limit:
                break
    bench = bench_dir / f"paroutes_{split}_smoke.json"
    bench.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "label": f"paroutes_{split}",
        "source": str(source),
        "benchmark": str(bench),
        "n_rows": len(rows),
        "ready": bool(rows),
        "route_annotations": False,
        "error": error,
    }


def _paroutes_manual_paths(split: str) -> tuple[Path, Path] | None:
    for targets_path, refs_path in PAROUTES_MANUAL_SOURCES.get(str(split), ()):
        if targets_path.exists() and refs_path.exists():
            return targets_path, refs_path
    return None


def _build_paroutes_reference_rows(
    *,
    targets_path: Path,
    refs_path: Path,
    split: str,
    limit: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    targets = [
        line.strip()
        for line in Path(targets_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    refs = json.loads(Path(refs_path).read_text(encoding="utf-8"))
    if not isinstance(refs, list):
        raise ValueError(f"PaRoutes reference routes must be a list: {refs_path}")
    rows = []
    offset = max(0, int(offset))
    stop = offset + max(0, int(limit))
    for idx in range(offset, min(stop, len(targets))):
        target = targets[idx]
        ref = refs[idx] if idx < len(refs) and isinstance(refs[idx], dict) else {}
        smiles = str(ref.get("smiles") or target)
        row = _target_only_row(f"paroutes_{split}_{idx}", smiles, depth=3, route_domain="all_chemical")
        route = _paroutes_ref_gt_route(ref)
        if route:
            row["gt_route"] = route
            row["reference_depth"] = len(route)
            row["depth"] = _planner_depth(len(route))
        rows.append(row)
    return rows


def _paroutes_ref_gt_route(node: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        if item.get("type") == "reaction":
            rxn = _clean_reference_reaction((item.get("metadata") or {}).get("smiles") or (item.get("metadata") or {}).get("rsmi"))
            if rxn:
                steps.append(
                    {
                        "rxn_smiles": rxn,
                        "transformation": "uspto_clean_reference",
                        "step_role": "external_paroutes_reference",
                    }
                )
        for child in item.get("children") or []:
            walk(child)

    walk(node)
    return steps


def _clean_reference_reaction(rxn: Any) -> str:
    text = str(rxn or "").strip()
    if not text:
        return ""
    if ">>" not in text and text.count(">") == 2:
        lhs, _agents, rhs = text.split(">", 2)
        text = f"{lhs}>>{rhs}"
    if ">>" not in text:
        return ""
    lhs, rhs = text.split(">>", 1)
    return f"{_strip_atom_maps(lhs)}>>{_strip_atom_maps(rhs)}"


def _strip_atom_maps(smiles: str) -> str:
    return re.sub(r":\d+(?=\])", "", str(smiles or ""))


def _build_uspto190(
    source_dir: Path,
    bench_dir: Path,
    *,
    limit: int,
    fetch: bool,
    offset: int = 0,
    uspto_cache_dir: Path | None = None,
) -> dict[str, Any]:
    index_html = source_dir / "uspto190_index.html"
    uspto_cache_dir = Path(uspto_cache_dir) if uspto_cache_dir else None
    error = ""
    offset = max(0, int(offset))
    requested = offset + max(0, int(limit))
    if uspto_cache_dir and not index_html.exists():
        _copy_uspto_cache_source(uspto_cache_dir, index_html.name, index_html)
    if fetch and _needs_uspto_index_fetch(index_html, limit=requested):
        try:
            _download(SYNTHARENA_USPTO_190, index_html, timeout=60)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            if not (uspto_cache_dir and _copy_uspto_cache_source(uspto_cache_dir, index_html.name, index_html)):
                _copy_cached_source(index_html.name, index_html)
    rows = []
    target_paths = []
    if index_html.exists():
        target_paths = _target_paths(index_html.read_text(encoding="utf-8", errors="ignore"))
        if len(target_paths) < requested:
            for page_no in _uspto_pagination_pages(index_html.read_text(encoding="utf-8", errors="ignore")):
                page_html = source_dir / f"uspto190_page_{page_no}.html"
                if uspto_cache_dir and not page_html.exists():
                    _copy_uspto_cache_source(uspto_cache_dir, page_html.name, page_html)
                if fetch and not page_html.exists():
                    try:
                        _download(f"{SYNTHARENA_USPTO_190}?page={page_no}", page_html, timeout=60)
                    except Exception as exc:
                        error = error or f"{type(exc).__name__}:{exc}"
                        if not (uspto_cache_dir and _copy_uspto_cache_source(uspto_cache_dir, page_html.name, page_html)):
                            _copy_cached_source(page_html.name, page_html)
                if page_html.exists():
                    target_paths.extend(_target_paths(page_html.read_text(encoding="utf-8", errors="ignore")))
                if len(dict.fromkeys(target_paths)) >= requested:
                    break
        target_paths = list(dict.fromkeys(target_paths))[offset:requested]
    for target_path in target_paths:
        slug = target_path.rsplit("/", 1)[-1]
        html_path = source_dir / f"uspto190_{slug}.html"
        if uspto_cache_dir and not html_path.exists():
            _copy_uspto_cache_source(uspto_cache_dir, html_path.name, html_path)
        if fetch and not html_path.exists():
            try:
                _download(SYNTHARENA_TARGET.format(target_path=target_path), html_path, timeout=90)
            except Exception as exc:
                error = error or f"{type(exc).__name__}:{exc}"
                if not (uspto_cache_dir and _copy_uspto_cache_source(uspto_cache_dir, html_path.name, html_path)):
                    _copy_cached_source(html_path.name, html_path)
        if not html_path.exists():
            continue
        row = _uspto190_row(html_path)
        if row:
            rows.append(row)
    bench = bench_dir / "uspto_190_smoke.json"
    bench.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "label": "uspto_190",
        "source": str(index_html),
        "benchmark": str(bench),
        "n_rows": len(rows),
        "ready": bool(rows),
        "route_annotations": any(row.get("gt_route") for row in rows),
        "error": error,
    }


def _build_bionavi_like(source_dir: Path, bench_dir: Path, *, limit: int, fetch: bool, offset: int = 0) -> dict[str, Any]:
    source = source_dir / "bionavi_testset.txt"
    error = ""
    if fetch and not source.exists():
        try:
            _download(
                "https://raw.githubusercontent.com/prokia/BioNavi-NP/main/multistep/bio_building_blocks/testset.txt",
                source,
                timeout=45,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            _copy_cached_source(source.name, source)
    rows = []
    if source.exists():
        offset = max(0, int(offset))
        for idx, line in enumerate(source.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if idx < offset:
                continue
            smiles_path = _smiles_tokens(line)
            if not smiles_path:
                continue
            try:
                depth = int(line.strip().split()[0])
            except (TypeError, ValueError, IndexError):
                depth = max(1, len(smiles_path) - 1)
            row = _target_only_row(
                f"bionavi_like_{idx}",
                smiles_path[0],
                depth=depth,
                route_domain="enzymatic",
            )
            row["gt_route"] = [
                {
                    "rxn_smiles": f"{reactant}>>{product}",
                    "transformation": "biosynthetic_step",
                    "step_role": "external_bionavi_like_path",
                }
                for product, reactant in zip(smiles_path, smiles_path[1:])
            ]
            rows.append(row)
            if len(rows) >= limit:
                break
    bench = bench_dir / "bionavi_like_smoke.json"
    bench.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "label": "bionavi_like",
        "source": str(source),
        "benchmark": str(bench),
        "n_rows": len(rows),
        "ready": bool(rows),
        "route_annotations": any(row.get("gt_route") for row in rows),
        "error": error,
    }


def _download(url: str, output: Path, *, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": FETCH_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        output.write_bytes(response.read())


def _needs_uspto_index_fetch(index_html: Path, *, limit: int) -> bool:
    if not index_html.exists():
        return True
    text = index_html.read_text(encoding="utf-8", errors="ignore")
    return len(_target_paths(text)) < min(int(limit), 25)


def _copy_cached_source(name: str, output: Path) -> bool:
    cache = Path("results/shared/reservoir_distill_20260513/external_smokes/sources") / name
    if not cache.exists():
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cache.read_bytes())
    return True


def _copy_uspto_cache_source(cache_dir: Path, name: str, output: Path) -> bool:
    cache = Path(cache_dir) / name
    if not cache.exists():
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cache.read_bytes())
    return True


def _target_paths(text: str) -> list[str]:
    out = []
    seen = set()
    for path in re.findall(r"targets/[A-Za-z0-9_-]+", text):
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _uspto_pagination_pages(text: str) -> list[int]:
    pages = {int(value) for value in re.findall(r"page=([0-9]+)", html.unescape(text)) if int(value) > 1}
    if not pages:
        return []
    return list(range(2, max(pages) + 1))


def _uspto190_row(path: Path) -> dict[str, Any] | None:
    text = html.unescape(path.read_text(encoding="utf-8", errors="ignore"))
    obj = _extract_route_json(text)
    if not obj:
        return None
    target = obj.get("target") or {}
    mol = target.get("molecule") or {}
    target_smiles = mol.get("smiles")
    if not target_smiles:
        return None
    steps = []
    _walk_route(obj.get("rootNode") or {}, steps)
    reference_depth = int(target.get("routeLength") or len(steps) or 3)
    return {
        "doi": "SynthArena USPTO-190",
        "cascade_id": str(target.get("targetId") or target.get("id") or path.stem),
        "target_smiles": target_smiles,
        "route_domain": "all_chemical",
        "operation_mode": "external_smoke",
        "depth": _planner_depth(reference_depth),
        "reference_depth": reference_depth,
        "gt_route": [{"rxn_smiles": rxn, "transformation": "other", "step_role": "external_acceptable_route"} for rxn in steps],
    }


def _extract_route_json(text: str) -> dict[str, Any] | None:
    start = text.find('{\n  "route"')
    if start < 0:
        start = text.find('{"route"')
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for idx, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start: idx + 1])
    return None


def _walk_route(node: dict[str, Any], steps: list[str]) -> None:
    parent = ((node.get("molecule") or {}).get("smiles") or "").strip()
    children = node.get("children") or []
    reactants = [((child.get("molecule") or {}).get("smiles") or "").strip() for child in children]
    reactants = [smi for smi in reactants if smi]
    if node.get("reactionStep") and parent and reactants:
        steps.append(".".join(reactants) + ">>" + parent)
    for child in children:
        _walk_route(child, steps)


def _target_only_row(cascade_id: str, target_smiles: str, *, depth: int, route_domain: str) -> dict[str, Any]:
    return {
        "doi": "external_smoke",
        "cascade_id": cascade_id,
        "target_smiles": target_smiles,
        "route_domain": route_domain,
        "operation_mode": "external_smoke",
        "depth": _planner_depth(depth),
        "gt_route": [],
    }


def _planner_depth(depth: Any) -> int:
    try:
        value = int(depth)
    except (TypeError, ValueError):
        value = 3
    return max(1, min(MAX_PLANNER_DEPTH, value))


def _first_smiles_token(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    for token in re.split(r"[\s,\t]+", line):
        if token and any(ch.isalpha() for ch in token):
            return token
    return ""


def _smiles_tokens(line: str) -> list[str]:
    out = []
    for token in re.split(r"[\s,\t]+", line.strip()):
        if not token or not any(ch.isalpha() for ch in token):
            continue
        out.append(token)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/run external reservoir smoke benchmarks")
    ap.add_argument("--output-dir", default="results/shared/reservoir_distill_20260513/external_smokes")
    ap.add_argument("--controller", default="results/shared/reservoir_distill_20260513/reservoir_distilled_controller.pt")
    ap.add_argument("--native-payload", default="results/shared/chem_enzy_baseline/full100_reservoir_synthesized_rankplusstock_20260512.json")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument(
        "--config",
        action="append",
        choices=[
            "C",
            "C_CHEMSTEP",
            "C_ORACLE",
            "C_CHEMSTEP_ORACLE",
            "D",
            "D_FILTER",
            "D_TOP10_FILTER",
            "D_CHEMSTEP",
            "D_APPEND",
        ],
        default=None,
    )
    ap.add_argument("--offset", type=int, default=0, help="Start external datasets from this zero-based row offset.")
    ap.add_argument(
        "--dataset",
        action="append",
        choices=["paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"],
        default=None,
        help="Restrict manifest generation to one or more dataset labels.",
    )
    ap.add_argument("--uspto-cache-dir", default=None, help="Use a resumable USPTO-190 HTML cache as source input.")
    ap.add_argument("--enable-source-override", action="store_true")
    ap.add_argument("--prebuild-native-payload", action="store_true")
    ap.add_argument("--native-iterations", type=int, default=10)
    ap.add_argument("--native-max-depth", type=int, default=6)
    ap.add_argument("--native-expansion-topk", type=int, default=50)
    ap.add_argument("--native-gpu", type=int, default=-1)
    ap.add_argument("--native-stock", action="append", default=[])
    ap.add_argument("--native-one-step-model", action="append", default=[])
    ap.add_argument("--trust-native-stock", action="store_true")
    ap.add_argument("--skip-existing-native-payload", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    manifest = build_external_smokes(
        output_dir=output_dir,
        controller_path=Path(args.controller),
        native_payload=Path(args.native_payload) if args.native_payload else None,
        limit=args.limit,
        fetch=not args.no_fetch,
        configs=tuple(args.config or ["D"]),
        enable_source_override=bool(args.enable_source_override),
        prebuild_native_payload=bool(args.prebuild_native_payload),
        native_iterations=args.native_iterations,
        native_max_depth=args.native_max_depth,
        native_expansion_topk=args.native_expansion_topk,
        native_gpu=args.native_gpu,
        native_stocks=tuple(args.native_stock or ()),
        native_one_step_models=tuple(args.native_one_step_model or ()),
        trust_native_stock=bool(args.trust_native_stock),
        offset=int(args.offset),
        datasets_filter=tuple(args.dataset or ()),
        uspto_cache_dir=Path(args.uspto_cache_dir) if args.uspto_cache_dir else None,
    )
    print(json.dumps({"manifest": str(output_dir / "external_smoke_manifest.json"), "commands": len(manifest["commands"])}, indent=2))
    if args.run:
        run_external_smokes(
            manifest_path=output_dir / "external_smoke_manifest.json",
            log_dir=output_dir / "logs",
            skip_existing_native_payload=bool(args.skip_existing_native_payload),
        )
        print(json.dumps(summarize_external_smokes(output_dir=output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
