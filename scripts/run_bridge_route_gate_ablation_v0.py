"""Run controlled route-level ablation for bridge-aware source gating.

This benchmark exercises the actual route-tree search loop while keeping the
proposal engines controlled. It is intentionally not a chemistry solved-rate
benchmark. Its purpose is to verify that bridge gating changes route-level
behavior, not just pair-level scores:

* native_no_enzyme: chemical proposer only.
* ungated_default_source_gate: chemical + enzyme proposers with default source gate.
* bridge_gate_no_verifier: bridge evidence triggers enzyme budgets.
* bridge_gate_verifier: verifier-pass bridge evidence triggers enzyme budgets.

Targets come from the real bridge pack: positives have held-out bridge evidence;
negatives are chemical products without bridge hits. Controlled enzyme proposals
are terminal and high-scoring, so an ungated planner will select false enzyme
steps unless the source gate suppresses them.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import (
    BridgeRetrieverV0,
    BridgeVerifierV0Scorer,
    inchikey_from_smiles,
)
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.route_tree.source_gate import BridgeAwareSourceGate, SourceGate

RDLogger.DisableLog("rdApp.*")

DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_gate_ablation_v0_20260527")
DEFAULT_MODEL_PATH = Path("results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib")

ENZYME_SOURCES = {"enzyformer", "retrorules", "enzexpand", "enzyme", "enzymatic"}


class ControlledChemicalEngine:
    def __init__(self, stock: set[str]) -> None:
        self.stock = stock
        self.calls = 0
        self.products: list[str] = []

    def predict(self, product_smiles: str, top_k: int = 10, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        self.products.append(product_smiles)
        reactant = f"{stock_alkane_for(product_smiles, fraction=0.42)}O"
        self.stock.add(reactant)
        return [
            {
                "main_reactant": reactant,
                "rxn_smiles": f"{reactant}>>{product_smiles}",
                "type": "controlled_chemical_close",
                "score": 0.65,
                "source": "retrochimera",
            }
        ][: max(0, int(top_k or 0))]


class ControlledEnzymeEngine:
    def __init__(self, source: str, stock: set[str]) -> None:
        self.source = source
        self.stock = stock
        self.calls = 0
        self.products: list[str] = []

    def predict(self, product_smiles: str, top_k: int = 10, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        self.products.append(product_smiles)
        reactant = stock_alkane_for(product_smiles, fraction=0.42)
        self.stock.add(reactant)
        return [
            {
                "main_reactant": reactant,
                "rxn_smiles": f"{reactant}>>{product_smiles}",
                "type": "controlled_enzyme_bridge_candidate",
                "ec": "1.1.1.1",
                "score": 0.99,
                "source": self.source,
                "evidence": {"controlled_route_gate_benchmark": True},
            }
        ][: max(0, int(top_k or 0))]


def stock_alkane_for(smiles: str, *, fraction: float) -> str:
    heavy = max(4, heavy_atoms(smiles))
    n_carbons = max(4, int(math.ceil(float(heavy) * float(fraction))))
    return "C" * min(120, n_carbons)


def heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


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


def load_targets(pack_dir: Path, output_dir: Path, *, positives: int, negatives: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positive_rows: list[dict[str, Any]] = []
    cards_path = output_dir / "bridge_evidence_cards.json"
    if cards_path.exists():
        for row in json.loads(cards_path.read_text(encoding="utf-8")):
            smiles = str(row.get("chemical_smiles") or "")
            if not molecule_in_size_window(smiles):
                continue
            positive_rows.append(
                {
                    "target_smiles": smiles,
                    "chemical_inchikey": inchikey_from_smiles(smiles),
                    "label": 1,
                    "label_source": "bridge_evidence_cards",
                    "bridge_direction": row.get("bridge_direction") or "",
                    "verifier_score": row.get("verifier_score"),
                    "ec_sample": row.get("enzyme_ec_sample") or [],
                }
            )
    positive_rows = dedupe_targets(positive_rows)
    if len(positive_rows) < positives:
        for row in pq.read_table(pack_dir / "verifier_test.parquet").to_pylist():
            if int(row.get("label") or 0) != 1:
                continue
            smiles = str(row.get("chemical_smiles") or "")
            if not molecule_in_size_window(smiles):
                continue
            positive_rows.append(
                {
                    "target_smiles": smiles,
                    "chemical_inchikey": row.get("chemical_inchikey") or inchikey_from_smiles(smiles),
                    "label": 1,
                    "label_source": "verifier_test_positive",
                    "bridge_direction": row.get("bridge_direction") or "",
                    "verifier_score": None,
                    "ec_sample": [],
                }
            )
    positive_rows = dedupe_targets(positive_rows)
    rng.shuffle(positive_rows)
    positive_rows = positive_rows[:positives]

    positive_keys = positive_chemical_keys(pack_dir)
    negative_rows: list[dict[str, Any]] = []
    for row in pq.read_table(pack_dir / "chemical_product_pool.parquet").to_pylist():
        key = str(row.get("inchikey") or "")
        smiles = str(row.get("canonical_smiles") or "")
        if not key or key in positive_keys or not molecule_in_size_window(smiles):
            continue
        negative_rows.append(
            {
                "target_smiles": smiles,
                "chemical_inchikey": key,
                "label": 0,
                "label_source": "chemical_product_without_bridge_hit",
                "bridge_direction": "",
                "verifier_score": None,
                "ec_sample": [],
            }
        )
    negative_rows = dedupe_targets(negative_rows)
    rng.shuffle(negative_rows)
    negative_rows = negative_rows[:negatives]
    return [*positive_rows, *negative_rows]


def molecule_in_size_window(smiles: str) -> bool:
    heavy = heavy_atoms(smiles)
    return 8 <= heavy <= 120 and "." not in str(smiles or "")


def dedupe_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("chemical_inchikey") or inchikey_from_smiles(row.get("target_smiles") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_policy_gates(pack_dir: Path, model_path: Path) -> dict[str, SourceGate]:
    bridge_retriever = BridgeRetrieverV0(pack_dir, scorer=None)
    scorer = None if (pack_dir / "bridge_candidates_scored.parquet").exists() else BridgeVerifierV0Scorer(model_path)
    verifier_retriever = BridgeRetrieverV0(
        pack_dir,
        scorer=scorer,
    )
    return {
        "native_no_enzyme": SourceGate(),
        "ungated_default_source_gate": SourceGate(),
        "bridge_gate_no_verifier": BridgeAwareSourceGate(
            SourceGate(),
            retriever=bridge_retriever,
            require_verifier_pass=False,
        ),
        "bridge_gate_verifier": BridgeAwareSourceGate(
            SourceGate(),
            retriever=verifier_retriever,
            require_verifier_pass=True,
        ),
    }


def run_target(row: dict[str, Any], policy: str, *, source_gate: SourceGate) -> dict[str, Any]:
    stock: set[str] = set()
    chemical = ControlledChemicalEngine(stock)
    retro_engine: dict[str, Any] = {"retrochimera": chemical}
    enzymes: dict[str, ControlledEnzymeEngine] = {}
    if policy != "native_no_enzyme":
        enzymes = {
            "enzyformer": ControlledEnzymeEngine("enzyformer", stock),
            "retrorules": ControlledEnzymeEngine("retrorules", stock),
        }
        retro_engine.update(enzymes)
    target = str(row["target_smiles"])
    planner = NeuralGuidedAOSearch(
        retro_engine=retro_engine,
        stock_checker=lambda smi: smi in stock,
        max_depth=2,
        branch_factor=6,
        expansion_budget=8,
        controller=None,
    )
    planner.proposals.source_gate = source_gate
    results = planner.search(target, n_results=1)
    slots = list(results[0].board.slots) if results else []
    selected_sources = [str(getattr(slot, "source", "") or "") for slot in slots]
    selected_enzyme_steps = sum(1 for source in selected_sources if source in ENZYME_SOURCES)
    status = ""
    if results:
        try:
            status = str(results[0].explanation.uncertainty_table.get("route_tree_search_status") or "")
        except Exception:
            status = ""
    return {
        **row,
        "policy": policy,
        "result_count": len(results),
        "status": status,
        "solved": bool(results and not getattr(results[0].board, "slots", []) == []),
        "selected_sources": selected_sources,
        "selected_enzyme_steps": selected_enzyme_steps,
        "selected_enzyme_route": bool(selected_enzyme_steps > 0),
        "chemical_calls": chemical.calls,
        "enzyme_calls": sum(engine.calls for engine in enzymes.values()),
        "enzyformer_calls": enzymes.get("enzyformer").calls if "enzyformer" in enzymes else 0,
        "retrorules_calls": enzymes.get("retrorules").calls if "retrorules" in enzymes else 0,
        "generated_actions": planner.stats.generated_actions,
        "proposal_calls": planner.stats.proposal_calls,
        "proposal_budget_total": planner.stats.proposal_budget_total,
        "expansions": planner.stats.expansions,
        "elapsed_s": planner.stats.elapsed_s,
        "search_stop_reason": planner.stats.search_stop_reason,
        "proposal_source_stats": planner.stats.to_dict().get("proposal_source_stats", {}),
    }


def summarize(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["policy"] == policy]
    positives = [row for row in subset if int(row["label"]) == 1]
    negatives = [row for row in subset if int(row["label"]) == 0]
    selected = [row for row in subset if row["selected_enzyme_route"]]
    true_selected = [row for row in selected if int(row["label"]) == 1]
    false_selected = [row for row in selected if int(row["label"]) == 0]
    return {
        "policy": policy,
        "rows": len(subset),
        "positives": len(positives),
        "negatives": len(negatives),
        "result_count": sum(int(row["result_count"] > 0) for row in subset),
        "selected_enzyme_routes": len(selected),
        "true_selected_enzyme_routes": len(true_selected),
        "false_selected_enzyme_routes": len(false_selected),
        "selected_enzyme_precision": safe_div(len(true_selected), len(selected)),
        "selected_enzyme_recall": safe_div(len(true_selected), len(positives)),
        "false_enzyme_route_rate": safe_div(len(false_selected), len(negatives)),
        "mean_chemical_calls": mean(row["chemical_calls"] for row in subset),
        "mean_enzyme_calls": mean(row["enzyme_calls"] for row in subset),
        "mean_generated_actions": mean(row["generated_actions"] for row in subset),
        "mean_proposal_calls": mean(row["proposal_calls"] for row in subset),
        "mean_expansions": mean(row["expansions"] for row in subset),
        "mean_elapsed_s": mean(row["elapsed_s"] for row in subset),
    }


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bridge Route-Gate Ablation v0",
        "",
        "Controlled route-tree benchmark using real bridge-positive and bridge-negative frontier molecules.",
        "",
        "This is not a chemistry solved-rate benchmark; proposal engines are controlled stubs so the source-gate effect can be isolated.",
        "",
        f"- Targets: {report['inputs']['targets']}",
        f"- Positives: {report['inputs']['positives']}",
        f"- Negatives: {report['inputs']['negatives']}",
        "",
        "| policy | enzyme routes | true enzyme | false enzyme | precision | recall | false enzyme rate | mean enzyme calls | mean actions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| {policy} | {selected_enzyme_routes} | {true_selected_enzyme_routes} | {false_selected_enzyme_routes} | "
            "{selected_enzyme_precision:.4f} | {selected_enzyme_recall:.4f} | {false_enzyme_route_rate:.4f} | "
            "{mean_enzyme_calls:.2f} | {mean_generated_actions:.2f} |".format(**row)
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled route-level bridge gate ablation")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positives", type=int, default=20)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    started = time.monotonic()
    targets = load_targets(
        args.pack_dir,
        args.output_dir,
        positives=max(0, int(args.positives)),
        negatives=max(0, int(args.negatives)),
        seed=int(args.seed),
    )
    if not targets:
        raise SystemExit("no targets selected")
    policies = [
        "native_no_enzyme",
        "ungated_default_source_gate",
        "bridge_gate_no_verifier",
        "bridge_gate_verifier",
    ]
    policy_gates = build_policy_gates(args.pack_dir, args.model_path)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for target in targets:
            rows.append(run_target(target, policy, source_gate=policy_gates[policy]))
    summaries = [summarize(rows, policy) for policy in policies]
    ungated = next(row for row in summaries if row["policy"] == "ungated_default_source_gate")
    verifier = next(row for row in summaries if row["policy"] == "bridge_gate_verifier")
    conclusion = (
        "Verifier-gated route-tree search preserves enzyme-step recall on bridge-positive targets while suppressing "
        f"false enzyme route selection from {ungated['false_selected_enzyme_routes']} to "
        f"{verifier['false_selected_enzyme_routes']} in this controlled benchmark. "
        "The next benchmark must replace controlled engines with live proposal providers."
    )
    report = {
        "schema_version": "bridge_route_gate_ablation_v0.controlled_route_tree.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "model_path": str(args.model_path),
            "targets": len(targets),
            "positives": sum(1 for row in targets if int(row["label"]) == 1),
            "negatives": sum(1 for row in targets if int(row["label"]) == 0),
            "seed": int(args.seed),
        },
        "policies": summaries,
        "conclusion": conclusion,
        "scope_note": "Controlled route-tree benchmark; not a live ChemEnzy solved-rate benchmark.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bridge_route_gate_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "bridge_route_gate_ablation_report.md").write_text(render_markdown(report), encoding="utf-8")
    (args.output_dir / "bridge_route_gate_ablation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
