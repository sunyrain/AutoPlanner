"""Probe current one-step proposal models on bufotalin proposal frontiers."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_MAINLINE_ONE_STEP_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe ChemEnzy one-step models on bufotalin frontier targets.")
    parser.add_argument("root", help="Bufotalin run root containing proposal_frontier_analysis.json")
    parser.add_argument("--top-frontiers", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.root)
    provider = ChemEnzyOneStepProposalProvider(
        vendor_root=Path(args.vendor_root),
        models=tuple(BUFOTALIN_MAINLINE_ONE_STEP_MODELS),
        expansion_topk=max(1, int(args.top_k)),
        gpu=int(args.gpu),
    )
    report = probe_frontier_proposals(
        root,
        provider=provider,
        top_frontiers=max(1, int(args.top_frontiers)),
        top_k=max(1, int(args.top_k)),
    )
    output = Path(args.output) if args.output else root / "frontier_proposal_probe.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, indent=2, ensure_ascii=False))


def probe_frontier_proposals(
    root: Path | str,
    *,
    provider: Any,
    top_frontiers: int = 3,
    top_k: int = 10,
) -> dict[str, Any]:
    root = Path(root)
    frontier_report = _read_json(root / "proposal_frontier_analysis.json")
    frontiers = list(frontier_report.get("top_frontiers") or [])[: max(1, int(top_frontiers))]
    rows = []
    started = time.monotonic()
    for index, frontier in enumerate(frontiers, start=1):
        smiles = str(frontier.get("smiles") or "")
        item_started = time.monotonic()
        proposals: list[dict[str, Any]] = []
        error = ""
        try:
            proposals = list(provider.predict(smiles, top_k=max(1, int(top_k))) or [])
        except Exception as exc:  # pragma: no cover - defensive for real model adapters
            error = f"{type(exc).__name__}: {exc}"
        if not error:
            error = str(getattr(provider, "load_error", "") or "")
        rows.append(
            {
                "frontier_rank": index,
                "frontier": frontier,
                "elapsed_s": round(time.monotonic() - item_started, 3),
                "proposal_count": len(proposals),
                "gate_keep_count": sum(1 for proposal in proposals if _proposal_gate_decision(proposal) == "keep"),
                "gate_reject_count": sum(1 for proposal in proposals if _proposal_gate_decision(proposal) == "reject"),
                "error": error,
                "top_proposals": [_proposal_summary(proposal) for proposal in proposals[:10]],
            }
        )
    return {
        "schema_version": "bufotalin_frontier_proposal_probe.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "models": list(BUFOTALIN_MAINLINE_ONE_STEP_MODELS),
        "top_k": int(top_k),
        "summary": {
            "frontier_count": len(rows),
            "proposal_count": sum(int(row.get("proposal_count") or 0) for row in rows),
            "gate_keep_count": sum(int(row.get("gate_keep_count") or 0) for row in rows),
            "gate_reject_count": sum(int(row.get("gate_reject_count") or 0) for row in rows),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "frontiers": rows,
    }


def _proposal_summary(proposal: dict[str, Any]) -> dict[str, Any]:
    gate = proposal.get("proposal_gate") or {}
    return {
        "rank": proposal.get("rank"),
        "score": proposal.get("score"),
        "source": proposal.get("source"),
        "model_full_name": proposal.get("model_full_name"),
        "main_reactant": proposal.get("main_reactant"),
        "aux_reactants": proposal.get("aux_reactants") or [],
        "reaction_smiles": proposal.get("reaction_smiles") or proposal.get("rxn_smiles"),
        "gate_decision": gate.get("decision"),
        "gate_hard_reasons": gate.get("hard_reasons") or gate.get("reason_counts") or [],
    }


def _proposal_gate_decision(proposal: dict[str, Any]) -> str:
    return str((proposal.get("proposal_gate") or {}).get("decision") or "")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    main()
