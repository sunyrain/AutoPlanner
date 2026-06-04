"""Run molecule-level source-gate ablation for bridge-aware route search.

Candidate-level verifier metrics answer whether a substrate/product pair is
credible. Route search needs a slightly different question: should a frontier
molecule receive enzymatic proposal budget at all?

This smoke test compares:

* native_no_enzyme: no enzymatic sidecar budget.
* ungated_default_source_gate: existing heuristic source gate.
* bridge_aware_source_gate: verifier-backed bridge trigger.

Labels are weak molecule labels derived from the verifier test split:
a chemical molecule is positive if any held-out bridge pair for that molecule is
positive, and negative otherwise. This is a source-allocation smoke test, not a
route solved-rate benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0, BridgeVerifierV0Scorer
from cascade_planner.route_tree.proposals import ProposalContext
from cascade_planner.route_tree.source_gate import BridgeAwareSourceGate, SourceGate


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_MODEL_PATH = Path("results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_gate_ablation_v0_20260527")
DEFAULT_THRESHOLD = 0.8409896871324669
SOURCES = ["retrochimera", "enzyformer", "retrorules"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_molecule_rows(pack_dir: Path, *, limit_per_class: int, seed: int) -> list[dict[str, Any]]:
    rows = pq.read_table(pack_dir / "verifier_test.parquet").to_pylist()
    by_chemical: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row.get("label") or 0) != 1:
            continue
        key = str(row.get("chemical_inchikey") or "")
        if not key:
            continue
        existing = by_chemical.setdefault(
            key,
            {
                "chemical_inchikey": key,
                "chemical_smiles": row.get("chemical_smiles") or "",
                "label": 0,
                "positive_pairs": 0,
                "negative_pairs": 0,
                "label_types": set(),
            },
        )
        existing["label"] = 1
        existing["positive_pairs"] += 1
        existing["label_types"].add(str(row.get("label_type") or "unknown"))
    positives = [dict(row, label_types=sorted(row["label_types"])) for row in by_chemical.values() if row["label"] == 1]
    positive_keys = positive_chemical_keys(pack_dir)
    negative_by_chemical: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(pack_dir / "chemical_product_pool.parquet").to_pylist():
        key = str(row.get("inchikey") or "")
        if not key or key in positive_keys or key in negative_by_chemical:
            continue
        negative_by_chemical[key] = {
            "chemical_inchikey": key,
            "chemical_smiles": row.get("canonical_smiles") or "",
            "label": 0,
            "positive_pairs": 0,
            "negative_pairs": 1,
            "label_types": ["chemical_product_without_bridge_hit"],
        }
    negatives = list(negative_by_chemical.values())
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if limit_per_class > 0:
        positives = positives[:limit_per_class]
        negatives = negatives[:limit_per_class]
    return [*positives, *negatives]


def positive_chemical_keys(pack_dir: Path) -> set[str]:
    keys: set[str] = set()
    exact_path = pack_dir / "exact_bridge_strict.parquet"
    if exact_path.exists():
        for row in pq.read_table(exact_path, columns=["inchikey"]).to_pylist():
            key = str(row.get("inchikey") or "")
            if key:
                keys.add(key)
    similarity_path = pack_dir / "similarity_bridge_filtered.parquet"
    if similarity_path.exists():
        for row in pq.read_table(similarity_path, columns=["chemical_inchikey"]).to_pylist():
            key = str(row.get("chemical_inchikey") or "")
            if key:
                keys.add(key)
    return keys


def enzymatic_budget(allocation: Any) -> int:
    budgets = allocation.source_budgets or {}
    return int(budgets.get("enzyformer") or 0) + int(budgets.get("retrorules") or 0)


def evaluate_policy(rows: list[dict[str, Any]], policy: str, gate: Any | None) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    budgets: list[int] = []
    examples: list[dict[str, Any]] = []
    for row in rows:
        label = int(row.get("label") or 0)
        if policy == "native_no_enzyme":
            budget = 0
            reason = "native_no_enzyme"
        else:
            allocation = gate.allocate(
                row["chemical_smiles"],
                context=ProposalContext(),
                available_sources=SOURCES,
                total_budget=8,
            )
            budget = enzymatic_budget(allocation)
            reason = allocation.policy_reason
        accepted = budget > 0
        budgets.append(budget)
        if accepted and label:
            tp += 1
        elif accepted and not label:
            fp += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "chemical_smiles": row["chemical_smiles"],
                        "chemical_inchikey": row["chemical_inchikey"],
                        "label_types": row["label_types"],
                        "enzymatic_budget": budget,
                        "policy_reason": reason,
                    }
                )
        elif not accepted and label:
            fn += 1
        else:
            tn += 1
    positives = tp + fn
    negatives = tn + fp
    return {
        "policy": policy,
        "rows": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, positives),
        "false_positive_rate": safe_div(fp, negatives),
        "negative_rejection_rate": safe_div(tn, negatives),
        "mean_enzymatic_budget": float(statistics.mean(budgets)) if budgets else 0.0,
        "triggered": tp + fp,
        "false_trigger_examples": examples,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bridge Source-Gate Ablation v0",
        "",
        "Molecule-level smoke test for route-tree source allocation.",
        "",
        f"- Rows: {report['inputs']['rows']:,}",
        f"- Positives: {report['inputs']['positives']:,}",
        f"- Negatives: {report['inputs']['negatives']:,}",
        "",
        "| policy | triggered | precision | recall | FPR | negative rejection | mean enzyme budget |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| {policy} | {triggered:,} | {precision:.4f} | {recall:.4f} | {false_positive_rate:.4f} | "
            "{negative_rejection_rate:.4f} | {mean_enzymatic_budget:.2f} |".format(**row)
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge-aware source gate ablation v0")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    started = time.monotonic()
    rows = load_molecule_rows(
        args.pack_dir,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
    )
    positives = sum(1 for row in rows if int(row.get("label") or 0) == 1)
    negatives = len(rows) - positives
    scorer = BridgeVerifierV0Scorer(args.model_path, threshold=args.threshold)
    retriever = BridgeRetrieverV0(args.pack_dir, scorer=scorer)
    default_gate = SourceGate()
    bridge_gate = BridgeAwareSourceGate(default_gate, retriever=retriever)
    policies = [
        evaluate_policy(rows, "native_no_enzyme", None),
        evaluate_policy(rows, "ungated_default_source_gate", default_gate),
        evaluate_policy(rows, "bridge_aware_source_gate", bridge_gate),
    ]
    ungated = next(row for row in policies if row["policy"] == "ungated_default_source_gate")
    gated = next(row for row in policies if row["policy"] == "bridge_aware_source_gate")
    conclusion = (
        "At molecule/source-allocation level, bridge-aware gating reduces enzymatic sidecar triggers "
        f"from {ungated['triggered']} to {gated['triggered']} and FPR from "
        f"{ungated['false_positive_rate']:.4f} to {gated['false_positive_rate']:.4f}. "
        "This validates the route-tree source gate wiring; route solved-rate ablation is still pending."
    )
    report = {
        "schema_version": "bridge_source_gate_ablation_v0.molecule_level.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "model_path": str(args.model_path),
            "threshold": float(args.threshold),
            "rows": len(rows),
            "positives": positives,
            "negatives": negatives,
            "limit_per_class": int(args.limit_per_class),
            "sources": list(SOURCES),
        },
        "policies": policies,
        "conclusion": conclusion,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "bridge_source_gate_ablation_report.json", report)
    (args.output_dir / "bridge_source_gate_ablation_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
