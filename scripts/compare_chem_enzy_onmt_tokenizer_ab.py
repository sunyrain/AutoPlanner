#!/usr/bin/env python3
"""Run a small ChemEnzy Web-runner A/B between ONMT char and token modes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "chem_enzy_onmt_tokenizer_ab.v1"


def main() -> None:
    args = _parse_args()
    result = run_ab(
        targets=_targets(args.target, args.targets_json),
        output_dir=args.output_dir,
        vendor_root=args.vendor_root,
        gpu=args.gpu,
        preset=args.search_preset,
        max_steps=args.max_steps,
        iterations=args.iterations,
        expansion_topk=args.expansion_topk,
        stock_mode=args.stock_mode,
        enable_condition_prediction=args.enable_condition_prediction,
        enable_enzyme_assignment=args.enable_enzyme_assignment,
        timeout_s=args.timeout_s,
        execute=not args.dry_run,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tokenizer_ab_summary.json"
    md_path = args.output_dir / "tokenizer_ab_summary.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


def run_ab(
    *,
    targets: list[str],
    output_dir: Path,
    vendor_root: Path,
    gpu: int,
    preset: str,
    max_steps: int,
    iterations: int,
    expansion_topk: int,
    stock_mode: str,
    enable_condition_prediction: bool = False,
    enable_enzyme_assignment: bool = False,
    timeout_s: int = 1800,
    execute: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = []
    for idx, target in enumerate(targets):
        for tokenizer in ("char", "token"):
            row = _run_one(
                target=target,
                target_index=idx,
                tokenizer=tokenizer,
                output_dir=output_dir,
                vendor_root=vendor_root,
                gpu=gpu,
                preset=preset,
                max_steps=max_steps,
                iterations=iterations,
                expansion_topk=expansion_topk,
                stock_mode=stock_mode,
                enable_condition_prediction=enable_condition_prediction,
                enable_enzyme_assignment=enable_enzyme_assignment,
                timeout_s=timeout_s,
                execute=execute,
            )
            rows.append(row)
    comparisons = _comparisons(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "settings": {
            "preset": preset,
            "max_steps": max_steps,
            "iterations": iterations,
            "expansion_topk": expansion_topk,
            "stock_mode": stock_mode,
            "gpu": gpu,
            "execute": execute,
        },
        "summary": {
            "n_targets": len(targets),
            "n_runs": len(rows),
            "token_better_route_count": sum(1 for row in comparisons if row["token_routes"] > row["char_routes"]),
            "token_worse_route_count": sum(1 for row in comparisons if row["token_routes"] < row["char_routes"]),
            "token_better_solved_count": sum(1 for row in comparisons if row["token_solved"] and not row["char_solved"]),
            "token_worse_solved_count": sum(1 for row in comparisons if row["char_solved"] and not row["token_solved"]),
        },
        "runs": rows,
        "comparisons": comparisons,
        "contract": "A/B route-level smoke only. It does not promote token mode without route-quality inspection.",
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ChemEnzy ONMT Tokenizer A/B",
        "",
        f"生成时间：{result['created_at']}",
        "",
        "## Summary",
        "",
        f"- n_targets: {result['summary']['n_targets']}",
        f"- token_better_route_count: {result['summary']['token_better_route_count']}",
        f"- token_worse_route_count: {result['summary']['token_worse_route_count']}",
        f"- token_better_solved_count: {result['summary']['token_better_solved_count']}",
        f"- token_worse_solved_count: {result['summary']['token_worse_solved_count']}",
        "",
        "## Comparisons",
        "",
        "| target | char status | token status | char routes | token routes | char multi | token multi | char >=3 | token >=3 | char feasible | token feasible |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["comparisons"]:
        target = row["target_smiles"]
        short = target if len(target) <= 36 else target[:33] + "..."
        lines.append(
            f"| `{short}` | {row['char_status']} | {row['token_status']} | "
            f"{row['char_routes']} | {row['token_routes']} | "
            f"{row.get('char_multistep_routes', 0)} | {row.get('token_multistep_routes', 0)} | "
            f"{row.get('char_ge3_step_routes', 0)} | {row.get('token_ge3_step_routes', 0)} | "
            f"{row.get('char_rule_feasible_routes', 0)} | {row.get('token_rule_feasible_routes', 0)} |"
        )
    lines.extend([
        "",
        "## Contract",
        "",
        result["contract"],
        "",
    ])
    return "\n".join(lines)


def _run_one(
    *,
    target: str,
    target_index: int,
    tokenizer: str,
    output_dir: Path,
    vendor_root: Path,
    gpu: int,
    preset: str,
    max_steps: int,
    iterations: int,
    expansion_topk: int,
    stock_mode: str,
    enable_condition_prediction: bool,
    enable_enzyme_assignment: bool,
    timeout_s: int,
    execute: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_smiles": target,
        "search_preset": preset,
        "stock_mode": stock_mode,
        "max_steps": max_steps,
        "chem_enzy_iterations": iterations,
        "chem_enzy_expansion_topk": expansion_topk,
        "chem_enzy_onmt_tokenizer": tokenizer,
        "enable_condition_prediction": enable_condition_prediction,
        "enable_enzyme_assignment": enable_enzyme_assignment,
        "enable_rule_verifier_gate": False,
    }
    stem = f"target{target_index:02d}_{tokenizer}"
    request_path = output_dir / f"{stem}_request.json"
    output_path = output_dir / f"{stem}_plan.json"
    request_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_chem_enzy_plan_for_web.py"),
        "--input",
        str(request_path),
        "--output",
        str(output_path),
        "--vendor-root",
        str(vendor_root),
        "--gpu",
        str(gpu),
    ]
    row = {
        "target_smiles": target,
        "target_index": target_index,
        "tokenizer": tokenizer,
        "request_path": str(request_path),
        "output_path": str(output_path),
        "cmd": " ".join(cmd),
        "executed": execute,
    }
    if not execute:
        row.update({"returncode": None, "status": "planned", "n_routes": None, "best_steps": None, "solved": False})
        return row
    started = time.monotonic()
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True, timeout=timeout_s, check=False)
    row.update({
        "returncode": proc.returncode,
        "elapsed_s": round(time.monotonic() - started, 3),
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    })
    if output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
        routes = output.get("routes") or []
        route_stats = _route_stats(routes)
        row.update({
            "status": ((output.get("search_status") or {}).get("status") or "unknown"),
            "n_routes": len(routes),
            "best_steps": len((routes[0] or {}).get("steps") or []) if routes else None,
            **route_stats,
            "solved": bool((output.get("search_status") or {}).get("solved")),
            "ok": bool(output.get("ok")),
        })
    else:
        row.update({"status": "missing_output", "n_routes": 0, "best_steps": None, "solved": False, "ok": False})
    return row


def _comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["target_index"]), {})[str(row["tokenizer"])] = row
    out = []
    for idx, pair in sorted(grouped.items()):
        char = pair.get("char") or {}
        token = pair.get("token") or {}
        out.append({
            "target_index": idx,
            "target_smiles": char.get("target_smiles") or token.get("target_smiles"),
            "char_status": char.get("status"),
            "token_status": token.get("status"),
            "char_routes": int(char.get("n_routes") or 0),
            "token_routes": int(token.get("n_routes") or 0),
            "char_best_steps": char.get("best_steps"),
            "token_best_steps": token.get("best_steps"),
            "char_multistep_routes": int(char.get("multistep_routes") or 0),
            "token_multistep_routes": int(token.get("multistep_routes") or 0),
            "char_ge3_step_routes": int(char.get("ge3_step_routes") or 0),
            "token_ge3_step_routes": int(token.get("ge3_step_routes") or 0),
            "char_rule_feasible_routes": int(char.get("rule_feasible_routes") or 0),
            "token_rule_feasible_routes": int(token.get("rule_feasible_routes") or 0),
            "char_avg_steps": char.get("avg_steps"),
            "token_avg_steps": token.get("avg_steps"),
            "char_solved": bool(char.get("solved")),
            "token_solved": bool(token.get("solved")),
        })
    return out


def _route_stats(routes: list[dict[str, Any]]) -> dict[str, Any]:
    step_counts = [len((route or {}).get("steps") or []) for route in routes]
    feasible = 0
    for route in routes:
        verifier = ((route.get("metrics") or {}).get("cascade_verifier") or {})
        if verifier.get("feasible") is True:
            feasible += 1
    return {
        "multistep_routes": sum(1 for count in step_counts if count >= 2),
        "ge3_step_routes": sum(1 for count in step_counts if count >= 3),
        "max_steps_returned": max(step_counts) if step_counts else 0,
        "avg_steps": round(sum(step_counts) / max(len(step_counts), 1), 3) if step_counts else 0.0,
        "rule_feasible_routes": feasible,
    }


def _targets(raw_targets: list[str] | None, targets_json: Path | None) -> list[str]:
    out = list(raw_targets or [])
    if targets_json:
        payload = json.loads(targets_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            out.extend(str(item) for item in payload)
        elif isinstance(payload, dict):
            for row in payload.get("targets") or []:
                if isinstance(row, dict):
                    out.append(str(row.get("smiles") or row.get("target_smiles") or ""))
                else:
                    out.append(str(row))
    out = [item for item in out if item]
    if not out:
        raise ValueError("provide --target or --targets-json")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append")
    parser.add_argument("--targets-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/shared/chem_enzy_adapter_mainline_20260521/tokenizer_route_ab"))
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/ChemEnzyRetroPlanner"))
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--search-preset", default="quick")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--expansion-topk", type=int, default=50)
    parser.add_argument("--stock-mode", default="commercial")
    parser.add_argument("--enable-condition-prediction", action="store_true")
    parser.add_argument("--enable-enzyme-assignment", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
