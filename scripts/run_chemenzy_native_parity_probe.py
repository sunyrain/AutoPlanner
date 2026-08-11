"""Run one fixed ChemEnzy request through embedded and standalone paths."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.run_kernel import RunLimits, RunSpec
from cascade_planner.interfaces.chemenzy_probe import run_chemenzy_proposal_stage
from cascade_planner.interfaces.chemenzy_probe_routes import (
    compile_chemenzy_route_fingerprints,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def compile_native_parity_report(
    *,
    request: Mapping[str, Any],
    stage: Mapping[str, Any],
    embedded_raw: Mapping[str, Any],
    standalone_raw: Mapping[str, Any],
    embedded_elapsed_s: float,
    standalone_elapsed_s: float,
    stock_content_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = str(request.get("target_smiles") or "")
    embedded = compile_chemenzy_route_fingerprints(
        embedded_raw,
        target_smiles=target,
    )
    standalone = compile_chemenzy_route_fingerprints(
        standalone_raw,
        target_smiles=target,
    )
    binding = dict(stage.get("provider_invocation_binding") or {})
    runtime = dict(binding.get("runtime_binding") or {})
    stock_binding = dict(stock_content_binding or {})
    embedded_backend_failures = list(embedded_raw.get("backend_failures") or [])
    standalone_backend_failures = list(standalone_raw.get("backend_failures") or [])
    embedded_trace = _search_trace_summary(embedded_raw)
    standalone_trace = _search_trace_summary(standalone_raw)
    report = {
        "schema_version": "chemenzy_native_parity_probe.v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "request": {
            "target_smiles": target,
            "search_preset": request.get("search_preset"),
            "max_routes": request.get("max_routes"),
            "max_steps": request.get("max_steps"),
            "iterations": request.get("chem_enzy_iterations"),
            "expansion_topk": request.get("chem_enzy_expansion_topk"),
            "random_seed": request.get("chemenzy_seed"),
            "stock_names": list(request.get("stock_names") or []),
            "stock_paths": dict(request.get("stock_paths") or {}),
            "condition_prediction": bool(request.get("enable_condition_prediction")),
            "enzyme_assignment": bool(request.get("enable_enzyme_assignment")),
        },
        "launcher_request_sha256": _digest(request),
        "proposal_request_sha256": str(stage.get("request_sha256") or ""),
        "replay_key_sha256": str(stage.get("replay_key_sha256") or ""),
        "model_content_binding_sha256": str(
            runtime.get("model_content_binding_sha256") or ""
        ),
        "model_content_identity_complete": bool(
            runtime.get("model_content_identity_complete") is True
        ),
        "stock_content_binding_sha256": str(
            stock_binding.get("content_sha256") or ""
        ),
        "stock_content_identity_complete": bool(
            stock_binding.get("identity_complete") is True
        ),
        "stock_content_binding": stock_binding,
        "embedded": {
            **_fingerprint_summary(embedded, embedded_elapsed_s),
            **embedded_trace,
        },
        "standalone": {
            **_fingerprint_summary(standalone, standalone_elapsed_s),
            **standalone_trace,
        },
        "embedded_backend_failure_count": len(embedded_backend_failures),
        "standalone_backend_failure_count": len(standalone_backend_failures),
        "backend_failure_free": not (
            embedded_backend_failures or standalone_backend_failures
        ),
        "nonempty_route_set_observed": bool(
            int(embedded.get("route_count") or 0) > 0
            and int(standalone.get("route_count") or 0) > 0
        ),
        "search_trace_identity_complete": bool(
            embedded_trace["search_trace_count"] > 0
            and standalone_trace["search_trace_count"] > 0
        ),
        "search_trace_digest_equal": bool(
            embedded_trace["search_trace_sha256"]
            and embedded_trace["search_trace_sha256"]
            == standalone_trace["search_trace_sha256"]
        ),
        "raw_proposal_digest_equal": (
            embedded.get("raw_proposal_sha256")
            == standalone.get("raw_proposal_sha256")
        ),
        "route_fingerprint_rows_equal": (
            embedded.get("routes") == standalone.get("routes")
        ),
        "semantics": {
            "two_independent_native_searches_executed": True,
            "embedded_path_uses_v4_proposal_ingestion": True,
            "standalone_path_invokes_launcher_directly": True,
            "raw_result_digest_may_differ_due_to_operational_receipts": True,
            "parity_requires_model_content_identity_complete": True,
            "parity_requires_stock_content_identity_complete": True,
            "parity_requires_backend_failure_free": True,
            "parity_requires_nonempty_route_set": True,
            "parity_requires_search_trace_identity": True,
        },
    }
    report["parity_accepted"] = bool(
        report["model_content_identity_complete"]
        and report["stock_content_identity_complete"]
        and report["backend_failure_free"]
        and report["nonempty_route_set_observed"]
        and report["search_trace_identity_complete"]
        and report["search_trace_digest_equal"]
        and report["raw_proposal_digest_equal"]
        and report["route_fingerprint_rows_equal"]
    )
    report["content_sha256"] = _digest(report)
    return report


def run_native_parity_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / "embedded-run"
    if run_dir.exists():
        raise FileExistsError(f"parity output already exists: {run_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    service = RetrosynthesisCampaignService.create(
        output_root / "runtime",
        run_dir,
        spec=RunSpec(
            run_id=str(args.run_id),
            target_name=str(args.target_name),
            target_smiles=str(args.target_smiles),
            created_at=datetime.now(timezone.utc).isoformat(),
            limits=RunLimits(max_total_tasks=64),
        ),
        artifact_store_root=output_root / "cas",
        run_index_path=output_root / "index" / "runs.sqlite3",
    )
    stock_paths = _stock_paths(args.stock_path)
    stock_content_binding = _stock_content_binding(
        stock_names=list(args.stock_name),
        stock_paths=stock_paths,
    )
    embedded_started = time.monotonic()
    stage = run_chemenzy_proposal_stage(
        service,
        target_name=str(args.target_name),
        target_smiles=str(args.target_smiles),
        enabled=True,
        env_prefix=args.env_prefix,
        vendor_root=args.vendor_root,
        max_routes=int(args.max_routes),
        max_host_routes=int(args.max_host_routes or args.max_routes),
        max_steps=int(args.max_steps),
        max_iterations=int(args.iterations),
        expansion_topk=int(args.expansion_topk),
        timeout_s=float(args.timeout_s),
        search_preset=str(args.search_preset),
        random_seed=int(args.seed),
        stock_names=tuple(args.stock_name),
        stock_paths=stock_paths,
        enable_condition_prediction=bool(args.enable_condition_prediction),
        enable_enzyme_assignment=bool(args.enable_enzyme_assignment),
        enable_enzyme_coverage_sidecar=bool(args.enable_enzyme_coverage_sidecar),
        pandarallel_workers=int(args.pandarallel_workers),
    )
    embedded_elapsed = time.monotonic() - embedded_started
    request_path = run_dir / "chemenzy-v4-seed-request.json"
    embedded_path = run_dir / "chemenzy-v4-seed-result.json"
    request = _read_object(request_path)
    embedded_raw = _read_object(embedded_path)
    preflight = dict(stage.get("runtime_preflight") or {})
    standalone_path = output_root / "standalone-result.json"
    stdout_path = output_root / "standalone-stdout.log"
    stderr_path = output_root / "standalone-stderr.log"
    environment = os.environ.copy()
    environment["CHEMENZY_PANDARALLEL_WORKERS"] = str(args.pandarallel_workers)
    environment["PYTHONHASHSEED"] = str(args.seed)
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    command = [
        str(preflight["python_executable"]),
        str(preflight["launcher_path"]),
        "--input", str(request_path),
        "--output", str(standalone_path),
        "--vendor-root", str(preflight["vendor_root"]),
        "--gpu", "-1",
    ]
    standalone_started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(Path(preflight["launcher_path"]).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=float(args.timeout_s),
        check=False,
        env=environment,
    )
    standalone_elapsed = time.monotonic() - standalone_started
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0 or not standalone_path.is_file():
        raise RuntimeError(f"standalone_launcher_failed:{completed.returncode}")
    report = compile_native_parity_report(
        request=request,
        stage=stage,
        embedded_raw=embedded_raw,
        standalone_raw=_read_object(standalone_path),
        embedded_elapsed_s=embedded_elapsed,
        standalone_elapsed_s=standalone_elapsed,
        stock_content_binding=stock_content_binding,
    )
    (output_root / "chemenzy-native-parity-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _fingerprint_summary(value: Mapping[str, Any], elapsed_s: float) -> dict[str, Any]:
    return {
        "raw_proposal_sha256": str(value.get("raw_proposal_sha256") or ""),
        "raw_result_sha256": str(value.get("raw_result_sha256") or ""),
        "route_count": int(value.get("route_count") or 0),
        "quarantined_route_count": int(value.get("quarantined_route_count") or 0),
        "elapsed_s": round(float(elapsed_s), 3),
    }


def _search_trace_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(raw.get("raw_backend_metadata") or {})
    trace = dict(metadata.get("cascade_expansion_trace") or {})
    rows = list(trace.get("rows") or [])
    return {
        "search_trace_count": len(rows),
        "search_trace_sha256": _digest(rows) if rows else "",
    }


def _stock_paths(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        name, separator, path = str(value).partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"invalid stock path override: {value}")
        out[name.strip()] = str(Path(path.strip()).expanduser().resolve())
    return out


def _stock_content_binding(
    *,
    stock_names: list[str],
    stock_paths: Mapping[str, str],
) -> dict[str, Any]:
    names = [str(value).strip() for value in stock_names if str(value).strip()]
    if not names:
        raise ValueError("at least one --stock-name is required for parity evidence")
    missing = sorted(set(names) - set(stock_paths))
    extra = sorted(set(stock_paths) - set(names))
    if missing or extra:
        raise ValueError(
            "parity stock paths must exactly cover selected stocks: "
            f"missing={missing}, extra={extra}"
        )
    rows = []
    for name in sorted(names):
        path = Path(stock_paths[name]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"parity stock file not found: {path}")
        rows.append(
            {
                "stock_name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    binding = {
        "schema_version": "chemenzy_native_parity_stock_binding.v1",
        "identity_complete": bool(rows),
        "stocks": rows,
    }
    binding["content_sha256"] = _digest(binding)
    return binding


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-smiles", required=True)
    parser.add_argument("--target-name", default="opaque parity target")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default="chemenzy-native-parity-probe")
    parser.add_argument("--vendor-root", type=Path, default=ROOT / "vendor" / "ChemEnzyRetroPlanner")
    parser.add_argument("--env-prefix", type=Path)
    parser.add_argument("--stock-name", action="append", default=[])
    parser.add_argument("--stock-path", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--max-host-routes", type=int)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--expansion-topk", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-preset", default="standard")
    parser.add_argument("--pandarallel-workers", type=int, default=2)
    parser.add_argument("--enable-condition-prediction", action="store_true")
    parser.add_argument("--enable-enzyme-assignment", action="store_true")
    parser.add_argument("--enable-enzyme-coverage-sidecar", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_native_parity_probe(_parser().parse_args(argv))
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["parity_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
