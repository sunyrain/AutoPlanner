"""Aggregate proposal-gate frontiers from a bufotalin run package."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate bufotalin proposal-gate frontier evidence.")
    parser.add_argument("root", help="Bufotalin run root")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "proposal_frontier_analysis.json"
    report = analyze_proposal_frontiers(root)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, indent=2, ensure_ascii=False))


def analyze_proposal_frontiers(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for payload_path in sorted(root.glob("*/web_payload.json")):
        payload = _read_json(payload_path)
        gate = payload.get("proposal_gate") or {}
        cycle = payload_path.parent.name
        for dropped in gate.get("dropped") or []:
            frontier = (dropped or {}).get("frontier") or {}
            smiles = str(frontier.get("smiles") or "")
            if not smiles:
                continue
            reasons = [str(item) for item in frontier.get("proposal_reasons") or []]
            if not reasons and frontier.get("reason"):
                reasons = [str(frontier.get("reason"))]
            rows.append(
                {
                    "cycle": cycle,
                    "route_rank": dropped.get("route_rank"),
                    "n_steps": dropped.get("n_steps"),
                    "score": dropped.get("score"),
                    "smiles": smiles,
                    "reason": str(frontier.get("reason") or ""),
                    "proposal_reasons": reasons,
                }
            )
    frontier_rows = _frontier_summaries(rows)
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row.get("proposal_reasons") or [])
    return {
        "schema_version": "bufotalin_proposal_frontier_analysis.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "dropped_rows_with_frontier": len(rows),
            "unique_frontiers": len(frontier_rows),
            "complex_core_frontier_count": sum(1 for row in frontier_rows if row.get("complex_core_like")),
            "unsupported_prenyl_frontier_count": sum(
                1 for row in frontier_rows if "unsupported_biosynthetic_prenyl_terminal" in row.get("reason_counts", {})
            ),
        },
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "top_frontiers": frontier_rows[:50],
        "top_complex_core_frontiers": [row for row in frontier_rows if row.get("complex_core_like")][:25],
        "top_unsupported_prenyl_frontiers": [
            row for row in frontier_rows
            if "unsupported_biosynthetic_prenyl_terminal" in row.get("reason_counts", {})
        ][:25],
    }


def _frontier_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["smiles"]].append(row)
    out = []
    for smiles, items in grouped.items():
        reason_counts: Counter[str] = Counter()
        cycles = sorted({str(item.get("cycle") or "") for item in items if item.get("cycle")})
        ranks = [
            int(item.get("route_rank"))
            for item in items
            if isinstance(item.get("route_rank"), int)
        ]
        for item in items:
            reason_counts.update(item.get("proposal_reasons") or [])
        profile = _mol_profile(smiles)
        out.append(
            {
                "smiles": smiles,
                "count": len(items),
                "cycle_count": len(cycles),
                "cycles": cycles[:10],
                "min_route_rank": min(ranks) if ranks else None,
                "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
                "profile": profile,
                "complex_core_like": bool(
                    int(profile.get("rings") or 0) >= 3
                    or int(profile.get("chiral_centers") or 0) >= 3
                    or int(profile.get("heavy_atoms") or 0) >= 25
                ),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            -int(row["count"]),
            -int(row["cycle_count"]),
            0 if row["min_route_rank"] is None else int(row["min_route_rank"]),
            row["smiles"],
        ),
    )


def _mol_profile(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"valid": False, "heavy_atoms": 0, "rings": 0, "chiral_centers": 0}
    return {
        "valid": True,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": mol.GetRingInfo().NumRings(),
        "chiral_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "formula": _formula(mol),
    }


def _formula(mol: Chem.Mol) -> str:
    counts: Counter[str] = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    main()
