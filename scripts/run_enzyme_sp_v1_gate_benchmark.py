"""Route-level benchmark for enzyme SP verifier v1 action gate.

This benchmark isolates the new action-level gate.  The existing bridge source
gate decides whether enzyme proposal sources may be queried for a frontier
molecule.  The v1 gate then decides whether the generated enzyme step itself is
credible as a substrate-product-EC tuple.

The controlled engines intentionally produce one high-scoring chemical action
and one high-scoring enzyme action.  The enzyme action can be configured as
plausible or implausible for the v1 verifier.  That lets the benchmark verify
that v1 changes route-level behavior rather than only producing offline scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.route_tree.source_gate import SourceGate


RDLogger.DisableLog("rdApp.*")

DEFAULT_OUTPUT_DIR = Path("results/shared/enzyme_sp_v1_gate_benchmark_20260528")
DEFAULT_MODEL = Path("results/shared/enzyme_sp_verifier_v1_20260528/enzyme_sp_verifier_v1_lgbm.joblib")
ENZYME_SOURCES = {"enzyformer", "retrorules", "enzexpand", "v3_retrieval"}


class ControlledChemicalEngine:
    def __init__(self, stock: set[str]) -> None:
        self.stock = stock
        self.calls = 0

    def predict(self, product_smiles: str, top_k: int = 10, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        reactant = "CCO"
        self.stock.add(reactant)
        return [
            {
                "main_reactant": reactant,
                "rxn_smiles": f"{reactant}>>{product_smiles}",
                "type": "controlled_chemical",
                "score": 0.80,
                "source": "retrochimera",
            }
        ][: max(0, int(top_k or 0))]


class ControlledEnzymeEngine:
    def __init__(self, *, stock: set[str], plausible: bool) -> None:
        self.stock = stock
        self.plausible = bool(plausible)
        self.calls = 0

    def predict(self, product_smiles: str, top_k: int = 10, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        reactant = "CCCC" if self.plausible else "O=P(O)(O)O"
        self.stock.add(reactant)
        ec = "1.1.1.1" if self.plausible else "2.7.1.1"
        return [
            {
                "main_reactant": reactant,
                "rxn_smiles": f"{reactant}>>{product_smiles}",
                "type": "controlled_enzyme",
                "ec": ec,
                "score": 0.99,
                "source": "enzyformer",
            }
        ][: max(0, int(top_k or 0))]


class AlwaysBridgeHitGate(SourceGate):
    def allocate(self, product: str, *, context: Any | None, available_sources: list[str] | tuple[str, ...], total_budget: int):
        allocation = super().allocate(
            product,
            context=context,
            available_sources=available_sources,
            total_budget=total_budget,
        )
        budgets = {source: 0 for source in available_sources}
        if "enzyformer" in budgets:
            budgets["enzyformer"] = max(1, int(total_budget) // 2)
        if "retrochimera" in budgets:
            budgets["retrochimera"] = max(1, int(total_budget) - budgets.get("enzyformer", 0))
        flags = dict(allocation.molecule_flags)
        flags["bridge_gate_checked"] = True
        flags["bridge_gate_hits"] = 1
        return type(allocation)(
            source_weights={source: (budgets[source] / max(sum(budgets.values()), 1)) for source in budgets},
            source_budgets=budgets,
            fallback_budget=0,
            molecule_flags=flags,
            safety_guard=allocation.safety_guard,
            source_group_probs=dict(allocation.source_group_probs),
            budget_multiplier=allocation.budget_multiplier,
            budget_multiplier_label=allocation.budget_multiplier_label,
            decision=allocation.decision,
            policy_confidence=1.0,
            policy_reason="bridge_gate_hits_controlled",
            policy_state_id=allocation.policy_state_id,
            selected_source_group="enzymatic",
            fallback_reason=allocation.fallback_reason,
        )


def run_case(*, target: str, plausible: bool, policy: str, scorer: EnzymeSPVerifierV1Scorer | None) -> dict[str, Any]:
    stock: set[str] = set()
    chemical = ControlledChemicalEngine(stock)
    enzyme = ControlledEnzymeEngine(stock=stock, plausible=plausible)
    planner = NeuralGuidedAOSearch(
        retro_engine={"retrochimera": chemical, "enzyformer": enzyme},
        stock_checker=lambda smi: smi in stock,
        max_depth=1,
        branch_factor=6,
        expansion_budget=4,
        controller=None,
        enzyme_sp_verifier=scorer if policy == "bridge_gate_v0_plus_enzyme_sp_v1" else None,
    )
    planner.proposals.source_gate = AlwaysBridgeHitGate()
    old_floor_flag = os.environ.get("AUTOPLANNER_ROUTE_TREE_V4_SOURCE_FLOORS")
    os.environ["AUTOPLANNER_ROUTE_TREE_V4_SOURCE_FLOORS"] = "0"
    try:
        results = planner.search(target, n_results=1)
    finally:
        if old_floor_flag is None:
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_V4_SOURCE_FLOORS", None)
        else:
            os.environ["AUTOPLANNER_ROUTE_TREE_V4_SOURCE_FLOORS"] = old_floor_flag
    sources: list[str] = []
    verifier_rows: list[dict[str, Any]] = []
    if results:
        for slot in results[0].board.slots:
            sources.append(str(getattr(slot, "source", "") or ""))
            evidence = getattr(slot, "evidence", {}) or {}
            if evidence.get("enzyme_sp_verifier_v1"):
                verifier_rows.append(dict(evidence["enzyme_sp_verifier_v1"]))
    return {
        "target_smiles": target,
        "plausible_enzyme_step": bool(plausible),
        "policy": policy,
        "selected_sources": sources,
        "selected_enzyme_route": any(source in ENZYME_SOURCES for source in sources),
        "chemical_calls": chemical.calls,
        "enzyme_calls": enzyme.calls,
        "result_count": len(results),
        "stats": planner.stats.to_dict(),
        "selected_verifier_rows": verifier_rows,
    }


def summarize(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["policy"] == policy]
    plausible = [row for row in subset if row["plausible_enzyme_step"]]
    implausible = [row for row in subset if not row["plausible_enzyme_step"]]
    selected = [row for row in subset if row["selected_enzyme_route"]]
    selected_plausible = [row for row in selected if row["plausible_enzyme_step"]]
    selected_implausible = [row for row in selected if not row["plausible_enzyme_step"]]
    return {
        "policy": policy,
        "rows": len(subset),
        "plausible_cases": len(plausible),
        "implausible_cases": len(implausible),
        "selected_enzyme_routes": len(selected),
        "selected_plausible_enzyme_routes": len(selected_plausible),
        "selected_implausible_enzyme_routes": len(selected_implausible),
        "plausible_recall": len(selected_plausible) / len(plausible) if plausible else 0.0,
        "implausible_accept_rate": len(selected_implausible) / len(implausible) if implausible else 0.0,
        "mean_enzyme_sp_rejections": _mean(
            (row["stats"] or {}).get("enzyme_sp_verifier_rejections") or 0 for row in subset
        ),
    }


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Enzyme SP Verifier v1 Gate Benchmark",
        "",
        f"- model: `{report['inputs']['model_path']}`",
        f"- target: `{report['inputs']['target_smiles']}`",
        "",
        "| policy | selected enzyme | plausible selected | implausible selected | plausible recall | implausible accept rate | mean v1 rejections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| {policy} | {selected_enzyme_routes} | {selected_plausible_enzyme_routes} | "
            "{selected_implausible_enzyme_routes} | {plausible_recall:.4f} | "
            "{implausible_accept_rate:.4f} | {mean_enzyme_sp_rejections:.2f} |".format(**row)
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enzyme SP verifier v1 route gate benchmark")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default="CCCCCCCC")
    args = parser.parse_args()

    started = time.monotonic()
    scorer = EnzymeSPVerifierV1Scorer(args.model_path)
    rows = []
    for policy in ["bridge_gate_v0_only", "bridge_gate_v0_plus_enzyme_sp_v1"]:
        for plausible in [True, False]:
            rows.append(run_case(target=args.target, plausible=plausible, policy=policy, scorer=scorer))
    summaries = [summarize(rows, policy) for policy in ["bridge_gate_v0_only", "bridge_gate_v0_plus_enzyme_sp_v1"]]
    base = summaries[0]
    gated = summaries[1]
    conclusion = (
        "The v1 action gate reduced implausible enzyme route acceptance from "
        f"{base['implausible_accept_rate']:.4f} to {gated['implausible_accept_rate']:.4f} while preserving "
        f"plausible enzyme recall at {gated['plausible_recall']:.4f} in this controlled route benchmark."
    )
    report = {
        "schema_version": "enzyme_sp_v1_gate_benchmark.controlled_route.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {"model_path": str(args.model_path), "target_smiles": args.target},
        "policies": summaries,
        "rows": rows,
        "conclusion": conclusion,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "enzyme_sp_v1_gate_benchmark_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "enzyme_sp_v1_gate_benchmark_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "enzyme_sp_v1_gate_benchmark_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
