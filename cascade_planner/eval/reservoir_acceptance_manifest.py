"""Build reproducible acceptance manifests for reservoir-distilled runs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_CONTROLLER = "results/shared/reservoir_distill_20260513/reservoir_distilled_controller_student_v2.pt"
DEFAULT_OUTPUT_DIR = "results/shared/reservoir_distill_20260513/full100_acceptance"
DEFAULT_BENCHMARK = "data/benchmark_v2_100.json"
DEFAULT_RUNTIME_GATE_SECONDS = 30.0
DEFAULT_BASELINE_ENV = [
    "AUTOPLANNER_CASCADE_SOURCE_POLICY=results/shared/controller_v2_20260512/fullrun/train_v8/source_policy/cascade_source_policy.pt",
    "AUTOPLANNER_ROUTE_TREE_POLICY=results/shared/controller_v2_20260512/fullrun/train_v8/open_leaf_policy/stock_aware_search_policy.pt",
    "AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH=retrochimera:0,chemtemplates:0",
    "AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL=1",
]

MATRIX_CONFIGS = {
    "A": {
        "name": "AutoPlanner D baseline",
        "extra_env": [
            "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
        ],
    },
    "B": {
        "name": "offline rank_plus_stock top-5 reservoir teacher",
        "extra_env": [
            "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
            "AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1",
            "AUTOPLANNER_RESERVOIR_ALWAYS_ON=1",
            "AUTOPLANNER_RESERVOIR_NATIVE_TOPK=5",
            "AUTOPLANNER_RESERVOIR_SELECTION=rank_plus_stock",
        ],
    },
    "C": {
        "name": "distilled controller only",
        "extra_env": [
            "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
            "AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE=1",
        ],
        "controller": True,
    },
    "D": {
        "name": "distilled controller + bounded reservoir fallback",
        "extra_env": [
            "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
            "AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1",
            "AUTOPLANNER_RESERVOIR_NATIVE_TOPK=5",
            "AUTOPLANNER_RESERVOIR_SELECTION=rank_plus_stock",
        ],
        "controller": True,
    },
    "D_FILTER": {
        "name": "distilled controller + bounded reservoir fallback + quality filter",
        "extra_env": [
            "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
            "AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1",
            "AUTOPLANNER_RESERVOIR_NATIVE_TOPK=5",
            "AUTOPLANNER_RESERVOIR_SELECTION=rank_plus_stock",
            "AUTOPLANNER_RESERVOIR_QUALITY_FILTER=1",
        ],
        "controller": True,
    },
    "D_TOP10_FILTER": {
        "name": "distilled controller + bounded reservoir top-10 + quality filter",
        "extra_env": [
            "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
            "AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1",
            "AUTOPLANNER_RESERVOIR_NATIVE_TOPK=10",
            "AUTOPLANNER_RESERVOIR_SELECTION=rank_plus_stock",
            "AUTOPLANNER_RESERVOIR_QUALITY_FILTER=1",
        ],
        "controller": True,
    },
    "E": {
        "name": "optional top-10 reservoir ablation",
        "extra_env": [
            "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1",
            "AUTOPLANNER_ENABLE_BOUNDED_RESERVOIR=1",
            "AUTOPLANNER_RESERVOIR_NATIVE_TOPK=10",
            "AUTOPLANNER_RESERVOIR_SELECTION=rank_plus_stock",
        ],
        "controller": True,
    },
}
APPEND_ONLY_CONFIG_NAME = "distilled controller + append-only native reservoir diagnostic"

EXTERNAL_BENCHMARK_PATTERNS = {
    "PaRoutes n1/n5": ["*paroutes*"],
    "USPTO-190": ["*uspto*190*", "*uspto190*"],
    "BioNavi-like": ["*bionavi*", "*bio*navi*"],
}


def build_reservoir_acceptance_manifest(
    *,
    output_dir: Path,
    benchmark_path: Path = Path(DEFAULT_BENCHMARK),
    controller_path: Path = Path(DEFAULT_CONTROLLER),
    model_path: str = "results/shared/skeleton_inpainter/best.pt",
    workers: int = 8,
    gpus: str = "0,1",
    device: str = "cpu",
    n_results: int = 5,
    n_candidates_per_skeleton: int = 1,
    skeleton_samples: int | None = 2,
    search_budget: int | None = None,
    native_payload: Path | None = None,
    baseline_env: list[str] | None = None,
    include_top10: bool = False,
    include_append_only: bool = False,
    limit: int | None = None,
    runtime_gate_seconds: float = DEFAULT_RUNTIME_GATE_SECONDS,
    include_quality_filter_ablation: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["A", "B", "C", "D"]
    if include_quality_filter_ablation:
        labels.extend(["D_FILTER", "D_TOP10_FILTER"])
    if include_top10:
        labels.append("E")
    if include_append_only and native_payload is None:
        raise ValueError("include_append_only requires a native_payload for offline C+native synthesis")
    command_benchmark_path = benchmark_path
    if limit is not None:
        command_benchmark_path = write_smoke_benchmark(
            source=benchmark_path,
            output=output_dir / f"benchmark_limit{max(1, int(limit))}.json",
            limit=limit,
        )
    commands = []
    runs: dict[str, str] = {}
    traces: dict[str, str] = {}
    for label in labels:
        run_dir = output_dir / label
        run_path = run_dir / "run.json"
        trace_path = run_dir / "run_trace.jsonl"
        log_dir = run_dir / "parallel_logs"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = MATRIX_CONFIGS[label]
        extra_env = [*(baseline_env if baseline_env is not None else DEFAULT_BASELINE_ENV), *config["extra_env"]]
        if config.get("controller"):
            extra_env.append(f"AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER={controller_path}")
        if native_payload is not None:
            extra_env.append(f"AUTOPLANNER_RESERVOIR_NATIVE_PAYLOAD={native_payload}")
        cmd = [
            "PYTHONPATH=.",
            "python -m cascade_planner.eval.run_live_benchmark_parallel",
            f"--bench {command_benchmark_path}",
            f"--output {run_path}",
            f"--model {model_path}",
            "--search-mode route_tree",
            "--check-stock",
            f"--workers {workers}",
            f"--device {device}",
            f"--n-results {n_results}",
            f"--n-candidates-per-skeleton {n_candidates_per_skeleton}",
            f"--trace-output {trace_path}",
            f"--log-dir {log_dir}",
        ]
        if gpus:
            cmd.append(f"--gpus {gpus}")
        if skeleton_samples is not None:
            cmd.append(f"--skeleton-samples {skeleton_samples}")
        if search_budget is not None:
            cmd.append(f"--search-budget {search_budget}")
        if limit is not None:
            cmd.append(f"--limit {limit}")
        for item in extra_env:
            cmd.append(f"--extra-env {item}")
        runs[label] = str(run_path)
        traces[label] = str(trace_path)
        commands.append(
            {
                "stage": "full100_acceptance",
                "config": label,
                "split": "full100",
                "outputs": {"run": str(run_path), "trace": str(trace_path)},
                "cmd": " ".join(cmd),
                "description": config["name"],
            }
        )

    matrix_labels = list(labels)
    if include_append_only:
        append_label = "D_APPEND"
        run_dir = output_dir / append_label
        run_path = run_dir / "run.json"
        union_report_path = run_dir / "append_union_report.json"
        union_markdown_path = run_dir / "append_union_report.md"
        run_dir.mkdir(parents=True, exist_ok=True)
        runs[append_label] = str(run_path)
        matrix_labels.append(append_label)
        commands.append(
            {
                "stage": "full100_append_only_reservoir",
                "config": append_label,
                "split": "full100",
                "outputs": {
                    "run": str(run_path),
                    "native_payload": str(native_payload),
                    "autoplanner": runs["C"],
                    "union_report": str(union_report_path),
                    "markdown": str(union_markdown_path),
                },
                "cmd": " ".join(
                    [
                        "PYTHONPATH=.",
                        "python -m cascade_planner.eval.chem_enzy_broad_union",
                        f"--benchmark {command_benchmark_path}",
                        f"--chem-enzy {native_payload}",
                        f"--autoplanner {runs['C']}",
                        f"--output {union_report_path}",
                        f"--markdown {union_markdown_path}",
                        "--native-topk 5",
                        "--native-selection rank_plus_stock",
                        f"--synthesize-output {run_path}",
                    ]
                ),
                "description": APPEND_ONLY_CONFIG_NAME,
            }
        )

    report_dir = output_dir / "reports"
    matrix_cmd = [
        "PYTHONPATH=.",
        "python -m cascade_planner.eval.reservoir_distill_matrix",
        f"--output-dir {report_dir}",
        f"--benchmark {command_benchmark_path}",
    ]
    for label in matrix_labels:
        matrix_cmd.append(f"--run {label}={runs[label]}")
    for label in matrix_labels:
        if label in traces:
            matrix_cmd.append(f"--trace {label}={traces[label]}")
    commands.append(
        {
            "stage": "full100_reports",
            "config": "matrix",
            "split": "full100",
            "outputs": {
                "manifest": str(report_dir / "reservoir_distill_matrix_manifest.json"),
                "comparison": str(report_dir / "comparison.md"),
            },
            "cmd": " ".join(matrix_cmd),
            "description": "assemble A-D/E reservoir distillation comparison reports",
        }
    )

    manifest = {
        "schema_version": "reservoir_acceptance_manifest.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_path": str(benchmark_path),
        "command_benchmark_path": str(command_benchmark_path),
        "controller_path": str(controller_path),
        "output_dir": str(output_dir),
        "baseline_env": list(baseline_env if baseline_env is not None else DEFAULT_BASELINE_ENV),
        "matrix": {
            label: (MATRIX_CONFIGS[label]["name"] if label in MATRIX_CONFIGS else APPEND_ONLY_CONFIG_NAME)
            for label in matrix_labels
        },
        "commands": commands,
        "promotion_gates": {
            "plan_rate": 0.95,
            "strict_stock_solve_any": 0.60,
            "avg_time_per_target_s_max": float(runtime_gate_seconds),
        },
        "reference_recall_diagnostics": {
            "candidate_gt_reactant_in_pool": 0.58,
            "exact_reaction_in_route_pool": 0.40,
            "gt_reactant_in_route_pool": 0.63,
            "blocking": False,
            "note": "Reference GT is one literature route, not the only valid retrosynthesis. These thresholds diagnose route recall, not promotion by themselves.",
        },
        "quality_review_gates": {
            "stock_closed_alternative_review_pass_rate": 0.70,
            "blocking": "manual_or_audit_available",
            "note": "Use cascade_planner.eval.audit_stock_closed_alternatives on stock-closed non-GT routes before strong usability claims.",
        },
        "notes": [
            "Run with cascade_planner.eval.run_pipeline_manifest_commands.",
            "D is promotable when stock, plan, configured runtime, and route-quality review gates pass.",
            "GT exact/reactant metrics are reference-route recall diagnostics; do not treat non-GT alternatives as failures without route-quality review.",
            "Runtime gate is configurable; relaxed effect-first runs use 20-30 seconds instead of the original strict 16 seconds.",
            "If gates require always-on native reservoir, mark the result hybrid promoted.",
            "D_APPEND is an optional offline append-only diagnostic that freezes C before adding native top-5 routes; it is not a direct online runtime measurement.",
        ],
    }
    path = output_dir / "reservoir_acceptance_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def audit_external_benchmarks(*, search_roots: list[Path]) -> dict[str, Any]:
    found: dict[str, list[str]] = {}
    for name, patterns in EXTERNAL_BENCHMARK_PATTERNS.items():
        paths: list[str] = []
        for root in search_roots:
            if not root.exists():
                continue
            for pattern in patterns:
                paths.extend(str(path) for path in root.rglob(pattern) if path.is_file())
        found[name] = sorted(set(paths))
    missing = [name for name, paths in found.items() if not paths]
    return {
        "schema_version": "reservoir_external_benchmark_audit.v1",
        "search_roots": [str(path) for path in search_roots],
        "found": found,
        "missing": missing,
        "ready": not missing,
    }


def write_smoke_benchmark(*, source: Path, output: Path, limit: int = 3) -> Path:
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
    if not isinstance(rows, list):
        raise ValueError(f"unsupported benchmark format: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows[: max(1, int(limit))], indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reservoir acceptance manifests and data audits")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--controller", default=DEFAULT_CONTROLLER)
    ap.add_argument("--model", default="results/shared/skeleton_inpainter/best.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-results", type=int, default=5)
    ap.add_argument("--n-candidates-per-skeleton", type=int, default=1)
    ap.add_argument("--skeleton-samples", type=int, default=2)
    ap.add_argument("--search-budget", type=int, default=None)
    ap.add_argument("--native-payload", default=None)
    ap.add_argument("--baseline-env", action="append", default=None, help="Extra baseline env KEY=VALUE; use empty string to disable defaults")
    ap.add_argument("--include-top10", action="store_true")
    ap.add_argument("--include-quality-filter-ablation", action="store_true")
    ap.add_argument("--include-append-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--runtime-gate-seconds", type=float, default=DEFAULT_RUNTIME_GATE_SECONDS)
    ap.add_argument("--audit-external", action="store_true")
    ap.add_argument("--write-smoke-benchmark", default=None)
    ap.add_argument("--smoke-limit", type=int, default=3)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    manifest = build_reservoir_acceptance_manifest(
        output_dir=output_dir,
        benchmark_path=Path(args.benchmark),
        controller_path=Path(args.controller),
        model_path=args.model,
        workers=args.workers,
        gpus=args.gpus,
        device=args.device,
        n_results=args.n_results,
        n_candidates_per_skeleton=args.n_candidates_per_skeleton,
        skeleton_samples=args.skeleton_samples,
        search_budget=args.search_budget,
        native_payload=Path(args.native_payload) if args.native_payload else None,
        baseline_env=[] if args.baseline_env == [""] else args.baseline_env,
        include_top10=args.include_top10,
        include_quality_filter_ablation=args.include_quality_filter_ablation,
        include_append_only=args.include_append_only,
        limit=args.limit,
        runtime_gate_seconds=args.runtime_gate_seconds,
    )
    outputs = {"manifest": str(output_dir / "reservoir_acceptance_manifest.json")}
    if args.audit_external:
        audit = audit_external_benchmarks(
            search_roots=[Path("data"), Path("data_external"), Path("dataset_v4_release"), Path("results/shared")]
        )
        audit_path = output_dir / "external_benchmark_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs["external_benchmark_audit"] = str(audit_path)
    if args.write_smoke_benchmark:
        smoke_path = write_smoke_benchmark(
            source=Path(args.benchmark),
            output=Path(args.write_smoke_benchmark),
            limit=args.smoke_limit,
        )
        outputs["smoke_benchmark"] = str(smoke_path)
    print(json.dumps({"outputs": outputs, "commands": len(manifest["commands"])}, indent=2))


if __name__ == "__main__":
    main()
