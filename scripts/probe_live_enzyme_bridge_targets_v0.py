"""Probe bridge-positive targets for live enzyme-provider coverage.

P5 needs real live-provider evidence routes, not only controlled routes. This
script scans verifier-pass bridge targets and asks the live enzyme providers
directly whether they can produce retrosynthetic candidates. The output becomes
the candidate target list for the larger live-provider route benchmark.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascadeboard.live_retro import build_live_retro_engine, retro_engine_cache_stats
from cascade_planner.route_tree.schema import CandidateAction

RDLogger.DisableLog("rdApp.*")
logging.getLogger().setLevel(logging.WARNING)

DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_gate_ablation_v0_20260527")
ENZYME_SOURCES = ("enzyformer", "enzexpand", "retrorules")
HARD_INVALID_FLAGS = {"no_reactants", "no_main_reactant", "product_mismatch", "self_loop"}


def heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def load_targets(pack_dir: Path, *, limit: int, min_heavy: int, max_heavy: int) -> list[dict[str, Any]]:
    path = pack_dir / "bridge_candidates_scored.parquet"
    rows = pq.read_table(path).to_pylist()
    rows = [row for row in rows if bool(row.get("verifier_pass"))]
    rows.sort(
        key=lambda row: (
            str(row.get("source") or "") != "exact_bridge_strict",
            -float(row.get("verifier_score") or 0.0),
            -float(row.get("tanimoto") or 0.0),
        )
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        smiles = str(row.get("chemical_smiles") or "")
        key = str(row.get("chemical_inchikey") or "")
        heavy = heavy_atoms(smiles)
        if not key or key in seen or "." in smiles or heavy < min_heavy or heavy > max_heavy:
            continue
        seen.add(key)
        out.append(
            {
                "target_smiles": smiles,
                "chemical_inchikey": key,
                "bridge_source": row.get("source") or "",
                "bridge_direction": row.get("bridge_direction") or "",
                "verifier_score": float(row.get("verifier_score") or 0.0),
                "tanimoto": float(row.get("tanimoto") or 0.0),
                "enzyme_ec_sample_json": row.get("enzyme_ec_sample_json") or "[]",
                "heavy_atoms": heavy,
            }
        )
        if len(out) >= limit:
            break
    return out


def predict_source(engine: Any, target: str, top_k: int) -> tuple[list[dict[str, Any]], float, str]:
    started = time.monotonic()
    try:
        rows = list(engine.predict(target, top_k=top_k) or [])
        return rows, time.monotonic() - started, ""
    except Exception as exc:
        return [], time.monotonic() - started, f"{type(exc).__name__}:{exc}"


def summarize_candidate(
    target_smiles: str,
    candidate: dict[str, Any],
    *,
    source: str,
    rank: int,
    min_largest_reactant_heavy_ratio: float,
) -> dict[str, Any]:
    action = CandidateAction.from_candidate(target_smiles, candidate, rank=rank, source=source)
    product_heavy = heavy_atoms(target_smiles)
    reactant_heavies = [heavy_atoms(smi) for smi in action.reactants]
    largest_reactant_heavy = max(reactant_heavies, default=0)
    ratio = largest_reactant_heavy / product_heavy if product_heavy else 0.0
    hard_flags = sorted(set(action.validity_flags) & HARD_INVALID_FLAGS)
    size_flag = ""
    if product_heavy >= 8 and largest_reactant_heavy and ratio < min_largest_reactant_heavy_ratio:
        size_flag = "tiny_largest_reactant"
    usable = not hard_flags and not size_flag
    return {
        "source": action.source,
        "rank": int(rank),
        "main_reactant": action.main_reactant,
        "aux_reactants": list(action.aux_reactants),
        "reactants": list(action.reactants),
        "rxn_smiles": action.rxn_smiles,
        "score": float(action.raw_score),
        "ec": action.ec,
        "validity_flags": list(action.validity_flags),
        "hard_invalid_flags": hard_flags,
        "product_heavy_atoms": product_heavy,
        "largest_reactant_heavy_atoms": largest_reactant_heavy,
        "largest_reactant_heavy_ratio": round(ratio, 4),
        "quality_flags": [size_flag] if size_flag else [],
        "usable_candidate": bool(usable),
    }


def probe_target(
    target: dict[str, Any],
    engine: dict[str, Any],
    *,
    top_k: int,
    min_largest_reactant_heavy_ratio: float,
) -> dict[str, Any]:
    source_rows: dict[str, Any] = {}
    total = 0
    usable_total = 0
    for source in ENZYME_SOURCES:
        provider = engine.get(source)
        if provider is None:
            source_rows[source] = {
                "available": False,
                "returned": 0,
                "elapsed_s": 0.0,
                "error": "missing_provider",
                "sample": [],
            }
            continue
        rows, elapsed, error = predict_source(provider, target["target_smiles"], top_k)
        candidate_summaries = [
            summarize_candidate(
                target["target_smiles"],
                row,
                source=source,
                rank=idx + 1,
                min_largest_reactant_heavy_ratio=min_largest_reactant_heavy_ratio,
            )
            for idx, row in enumerate(rows)
        ]
        usable = [row for row in candidate_summaries if row["usable_candidate"]]
        total += len(rows)
        usable_total += len(usable)
        source_rows[source] = {
            "available": True,
            "returned": len(rows),
            "usable": len(usable),
            "elapsed_s": round(elapsed, 3),
            "error": error,
            "sample": rows[:3],
            "candidate_summaries": candidate_summaries[:10],
        }
    return {
        **target,
        "total_enzyme_candidates": total,
        "usable_enzyme_candidates": usable_total,
        "source_rows": source_rows,
        "has_live_enzyme_candidate": total > 0,
        "has_usable_live_enzyme_candidate": usable_total > 0,
    }


def render_markdown(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Live Enzyme Bridge Target Probe v0",
        "",
        "Verifier-pass bridge targets probed against live enzyme proposal providers.",
        "",
        f"- Probed targets: {report['inputs']['targets']}",
        f"- Raw covered targets: {report['summary']['raw_covered_targets']}",
        f"- Raw coverage: {report['summary']['raw_coverage_rate']:.4f}",
        f"- Usable covered targets: {report['summary']['usable_covered_targets']}",
        f"- Usable coverage: {report['summary']['usable_coverage_rate']:.4f}",
        "",
        "| target | heavy | bridge | verifier | raw candidates | usable candidates | source counts |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows[:50]:
        source_counts = ", ".join(
            f"{source}:{data['returned']}/{data.get('usable', 0)}" for source, data in row["source_rows"].items()
        )
        lines.append(
            f"| `{row['target_smiles'][:80]}` | {row['heavy_atoms']} | {row['bridge_source']} | "
            f"{row['verifier_score']:.4f} | {row['total_enzyme_candidates']} | "
            f"{row['usable_enzyme_candidates']} | {source_counts} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe live enzyme providers on bridge-positive targets")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-heavy", type=int, default=8)
    parser.add_argument("--max-heavy", type=int, default=80)
    parser.add_argument("--min-largest-reactant-heavy-ratio", type=float, default=0.35)
    args = parser.parse_args()

    started = time.monotonic()
    targets = load_targets(
        args.pack_dir,
        limit=max(1, int(args.limit)),
        min_heavy=max(1, int(args.min_heavy)),
        max_heavy=max(1, int(args.max_heavy)),
    )
    engine = build_live_retro_engine()
    rows = [
        probe_target(
            target,
            engine,
            top_k=max(1, int(args.top_k)),
            min_largest_reactant_heavy_ratio=max(0.0, float(args.min_largest_reactant_heavy_ratio)),
        )
        for target in targets
    ]
    raw_covered = [row for row in rows if row["has_live_enzyme_candidate"]]
    usable_covered = [row for row in rows if row["has_usable_live_enzyme_candidate"]]
    report = {
        "schema_version": "live_enzyme_bridge_target_probe_v0",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "targets": len(targets),
            "top_k": int(args.top_k),
            "min_heavy": int(args.min_heavy),
            "max_heavy": int(args.max_heavy),
            "min_largest_reactant_heavy_ratio": float(args.min_largest_reactant_heavy_ratio),
        },
        "summary": {
            "raw_covered_targets": len(raw_covered),
            "raw_coverage_rate": len(raw_covered) / len(targets) if targets else 0.0,
            "usable_covered_targets": len(usable_covered),
            "usable_coverage_rate": len(usable_covered) / len(targets) if targets else 0.0,
            "total_enzyme_candidates": sum(int(row["total_enzyme_candidates"]) for row in rows),
            "usable_enzyme_candidates": sum(int(row["usable_enzyme_candidates"]) for row in rows),
        },
        "retro_cache_stats": retro_engine_cache_stats(engine),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "live_enzyme_bridge_target_probe_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    json_path = args.output_dir / "live_enzyme_bridge_target_probe_report.json"
    md_path = args.output_dir / "live_enzyme_bridge_target_probe_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report, rows), encoding="utf-8")
    print(json.dumps({"report": str(json_path), "rows": str(rows_path), **report["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
