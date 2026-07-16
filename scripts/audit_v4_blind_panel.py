#!/usr/bin/env python3
"""Print a compact, truth-preserving audit of V4 blind-panel runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stage(report: dict[str, Any], name: str) -> dict[str, Any]:
    for stage in report.get("stages", []):
        if stage.get("stage") == name:
            return stage
    return {"status": "missing", "detail": {}}


def _count(value: Any) -> int:
    if isinstance(value, (list, tuple, dict, set)):
        return len(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def audit_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    evidence_stage = _stage(report, "evidence_acquisition")
    evidence = evidence_stage.get("detail", {})
    chemenzy = _stage(report, "chemenzy_guided_frontier").get("detail", {})
    replan_stage = _stage(report, "global_replan_budget_gate")
    replan = replan_stage.get("detail", {})
    source_frontier = _stage(report, "source_frontier").get("detail", {})
    gate_report = report.get("gates", {})
    counts = gate_report.get("counts", {})
    sources = source_frontier.get("sources", [])
    return {
        "target": report.get("target", {}).get("name") or path.parent.name,
        "run_id": report.get("run_id", ""),
        "profile": report.get("claim", {}).get("achieved_profile", "unknown"),
        "gates": gate_report.get("gates", {}),
        "skeleton_counts": {
            "target_rooted": counts.get("target_rooted_distinct_skeletons", 0),
            "reaction_validated": counts.get("reaction_validated_skeletons", 0),
            "stock_closed": counts.get("stock_closed_skeletons", 0),
            "evidence_closed": counts.get("evidence_closed_skeletons", 0),
        },
        "source_count": evidence.get("source_count", 0),
        "exact_record_count": evidence.get("exact_record_count", 0),
        "visual_invocation_count": _count(evidence.get("visual_invocations", [])),
        "sources": [
            {
                "kind": source.get("source_kind", ""),
                "status": source.get("acquisition_status", ""),
                "exact_rows": source.get("exact_row_count", 0),
                "title": source.get("title", ""),
                "doi": source.get("doi", ""),
                "pmcid": source.get("pmcid", ""),
            }
            for source in sources
        ],
        "chemenzy": {
            "invocations": chemenzy.get("provider_invocation_count", 0),
            "proposals": chemenzy.get("proposal_count", 0),
            "reasons": [result.get("reason", "") for result in chemenzy.get("results", [])],
        },
        "replan": {
            "status": replan_stage.get("status", "missing"),
            "accepted": replan.get("accepted", False),
            "reasons": replan.get("reasons", []),
            "trigger_reasons": replan.get("trigger_reasons", []),
            "prompt_context_bytes": replan.get("prompt_context_bytes", 0),
        },
    }


def _context_sizes(path: Path) -> dict[str, int]:
    from cascade_planner.application.campaign_context import CampaignContextCompiler
    from cascade_planner.application.proof_portfolio import compile_proof_portfolio
    from cascade_planner.interfaces.target_solver import _evidence_observations
    from cascade_planner.orchestration.retrosynthesis_service import (
        RetrosynthesisCampaignService,
    )

    panel_root = path.parent.parent.parent
    report = json.loads(path.read_text(encoding="utf-8"))
    evidence = _stage(report, "evidence_acquisition").get("detail", {})
    service = RetrosynthesisCampaignService.open(
        panel_root / "runtime",
        path.parent,
        artifact_store_root=panel_root / "artifacts",
        run_index_path=panel_root / "runtime" / "run_index.sqlite3",
    )
    graph = service.graph_store.load()
    portfolio = compile_proof_portfolio(
        graph,
        acceptance_spec=service.kernel.spec.acceptance,
    )
    context = CampaignContextCompiler(max_context_bytes=10_000_000).compile(
        kernel=service.kernel,
        hypergraph=graph,
        route_portfolio=portfolio,
        evidence_ledger=_evidence_observations(evidence),
    ).to_dict()
    keys = (
        "topology",
        "route_portfolio",
        "evidence",
        "stock",
        "deficits",
        "proposal_history",
        "failure_history",
        "acceptance_state",
        "delta",
    )
    return {
        "total": int(context.get("byte_count") or 0),
        **{
            key: len(
                json.dumps(
                    context.get(key),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for key in keys
        },
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--context-sizes", action="store_true")
    args = parser.parse_args()
    rows = [
        audit_report(path)
        for path in sorted(args.runs_root.glob("*/target-only-solve-report.json"))
    ]
    if args.context_sizes:
        by_name = {
            path.parent.name: path
            for path in args.runs_root.glob("*/target-only-solve-report.json")
        }
        for row in rows:
            path = by_name.get(str(row["target"]).casefold())
            if path is not None:
                row["context_sizes"] = _context_sizes(path)
    if args.as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    for row in rows:
        gates = " ".join(
            f"{name.split('_', 1)[0]}{'+' if accepted else '-'}"
            for name, accepted in row["gates"].items()
        )
        skeletons = row["skeleton_counts"]
        print(
            f"{row['target']:<14} {row['profile']:<18} {gates:<24} "
            f"routes={skeletons['target_rooted']}/{skeletons['reaction_validated']}/"
            f"{skeletons['stock_closed']}/{skeletons['evidence_closed']} "
            f"sources={row['source_count']} exact={row['exact_record_count']} "
            f"vision={row['visual_invocation_count']} "
            f"ChemEnzy={row['chemenzy']['invocations']}/{row['chemenzy']['proposals']} "
            f"replan={row['replan']['status']}"
        )
        for source in row["sources"]:
            locator = source["doi"] or source["pmcid"]
            print(
                f"  - {source['status']:<30} rows={source['exact_rows']} "
                f"{locator} {source['title']}"
            )
        if row["chemenzy"]["reasons"]:
            print(f"  ChemEnzy: {', '.join(row['chemenzy']['reasons'])}")
        if row["replan"]["reasons"]:
            print(f"  replan: {', '.join(row['replan']['reasons'])}")
        if row.get("context_sizes"):
            print(
                "  context: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in row["context_sizes"].items()
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
