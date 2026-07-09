"""Backfill candidate-level stock fields in route-tree trace JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable


StockChecker = Callable[[str], bool]


def backfill_trace_rows(rows: Iterable[dict[str, Any]], *, stock_checker: StockChecker) -> list[dict[str, Any]]:
    return [backfill_trace_row(row, stock_checker=stock_checker) for row in rows]


def backfill_trace_row(row: dict[str, Any], *, stock_checker: StockChecker) -> dict[str, Any]:
    out = dict(row)
    event = dict(out.get("event") or {})
    actions = []
    for action in event.get("candidate_actions") or []:
        actions.append(annotate_action_stock(action, stock_checker=stock_checker))
    event["candidate_actions"] = actions
    event["candidate_stock_enriched"] = bool(actions)
    out["event"] = event
    return out


def annotate_action_stock(action: dict[str, Any], *, stock_checker: StockChecker) -> dict[str, Any]:
    out = dict(action or {})
    reactants = _action_reactants(out)
    if not reactants:
        out["reactant_stock_status"] = {}
        out["reactant_stock_fraction"] = 0.0
        out["stock_closing_candidate"] = False
        return out
    status: dict[str, bool] = {}
    for smi in reactants:
        try:
            status[smi] = bool(stock_checker(smi))
        except Exception:
            status[smi] = False
    hits = sum(1 for value in status.values() if value)
    out["reactant_stock_status"] = status
    out["reactant_stock_fraction"] = float(hits / max(len(reactants), 1))
    out["stock_closing_candidate"] = bool(hits == len(reactants))
    return out


def _action_reactants(action: dict[str, Any]) -> list[str]:
    reactants = [str(smi) for smi in action.get("reactants") or [] if smi]
    if reactants:
        return reactants
    out = []
    main = str(action.get("main_reactant") or "")
    if main:
        out.append(main)
    out.extend(str(smi) for smi in action.get("aux_reactants") or [] if smi)
    return out


def _load_stock_checker(*, fallback_small_molecules: bool) -> StockChecker:
    try:
        from cascade_planner.cascadeboard.zinc_stock import is_in_zinc_stock

        return is_in_zinc_stock
    except Exception:
        if not fallback_small_molecules:
            raise

    def small_molecule_stock(smiles: str) -> bool:
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(str(smiles or ""))
            return bool(mol is not None and mol.GetNumHeavyAtoms() <= 6)
        except Exception:
            return False

    return small_molecule_stock


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill route-tree trace candidate stock fields")
    ap.add_argument("--input", required=True, help="Input route-tree trace JSONL")
    ap.add_argument("--output", required=True, help="Output stock-enriched trace JSONL")
    ap.add_argument(
        "--fallback-small-molecules",
        action="store_true",
        help="Use heavy_atom_count<=6 as stock fallback when ZINC stock is unavailable",
    )
    args = ap.parse_args()

    stock_checker = _load_stock_checker(fallback_small_molecules=bool(args.fallback_small_molecules))
    rows = backfill_trace_rows(_read_jsonl(Path(args.input)), stock_checker=stock_checker)
    _write_jsonl(Path(args.output), rows)
    enriched_actions = sum(
        1
        for row in rows
        for action in ((row.get("event") or {}).get("candidate_actions") or [])
        if "reactant_stock_fraction" in action
    )
    stock_closing = sum(
        1
        for row in rows
        for action in ((row.get("event") or {}).get("candidate_actions") or [])
        if action.get("stock_closing_candidate")
    )
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "enriched_actions": enriched_actions,
        "stock_closing_actions": stock_closing,
    }, indent=2))


if __name__ == "__main__":
    main()
