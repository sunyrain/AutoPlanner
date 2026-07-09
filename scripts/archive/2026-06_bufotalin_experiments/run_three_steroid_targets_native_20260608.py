#!/usr/bin/env python3
"""Run native ChemEnzy attempts for three user-provided steroid-like targets."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.runner import emit_final_verdict
from cascade_planner.harness.schemas import ArtifactBundle, TargetInput, write_json
from cascade_planner.harness.tools import HarnessBudget, ToolExecutionState, execute_local_tool


TARGETS = [
    {
        "target_name": "steroid_candidate_01",
        "target_smiles": "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1",
    },
    {
        "target_name": "steroid_candidate_02",
        "target_smiles": "O=C1C=C[C@@]2(C)C(CCC3[C@]2(Cl)CC[C@@]4(C)[C@@]3(O)C[C@H](C)C4C(C)=O)=C1",
    },
    {
        "target_name": "steroid_candidate_03",
        "target_smiles": "O=C1CC[C@@]2(C)C(CCC3C2C[C@H](Cl)[C@@]4(C)C3CCC4=O)=C1",
    },
]


def main() -> None:
    out_root = ROOT / "results/shared/three_steroid_targets_native_20260608"
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target in TARGETS:
        summaries.append(run_one(out_root=out_root, target=target))
    summary = {
        "schema_version": "three_steroid_targets_native_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(out_root),
        "target_count": len(summaries),
        "targets": summaries,
    }
    write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def run_one(*, out_root: Path, target: dict[str, str]) -> dict[str, Any]:
    run_dir = out_root / target["target_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tool_calls.jsonl").touch()

    target_input = TargetInput(
        target_name=target["target_name"],
        target_smiles=target["target_smiles"],
        family_hint="steroid, polycyclic ketone, user_provided_panel",
        case_id=target["target_name"],
    ).to_dict()
    preflight = run_preflight(TargetInput(**target_input))
    write_json(run_dir / "target_input.json", target_input)
    write_json(run_dir / "preflight.json", preflight)

    workflow_plan = {
        "schema_version": "codex_entry_workflow_plan.v1",
        "case_id": str(preflight.get("case_id") or target["target_name"]),
        "recommended_strategy": "chem_enzy_first",
        "planned_tools": [
            {
                "tool_name": "run_chemenzy",
                "payload": {
                    "search_preset": "thorough",
                    "max_steps": 20,
                    "chem_enzy_iterations": 50,
                    "chem_enzy_expansion_topk": 100,
                    "stock_mode": "building-block",
                    "timeout_s": 1200.0,
                },
            },
            {"tool_name": "audit_route_and_extract_frontier", "payload": {}},
            {"tool_name": "emit_final_verdict", "payload": {}},
        ],
        "rationale": "native ChemEnzy attempt for user-provided steroid-like target",
        "risk_flags": list(preflight.get("initial_risk_flags") or []),
        "expected_verdict_floor": "needs_audit",
        "planner_decision_reason": "user_requested_native_attempt",
        "run_semantics": "canonical_agent_controller",
    }
    write_json(run_dir / "codex_workflow_plan.json", workflow_plan)

    state = ToolExecutionState(
        run_dir=run_dir,
        target_input=target_input,
        preflight=preflight,
        budget=HarnessBudget(
            max_chem_enzy_runs=1,
            max_guided_chemenzy_runs=0,
            max_route_expansion_subgoal_runs=0,
            max_codex_research_runs=0,
            timeout_s=1200.0,
            chem_enzy_timeout_s=1200.0,
        ),
        model="gpt-5.5",
    )
    tool_records = []
    if preflight.get("accepted"):
        for row in workflow_plan["planned_tools"]:
            tool_name = str(row["tool_name"])
            if tool_name == "emit_final_verdict":
                continue
            started = time.monotonic()
            record = execute_local_tool(tool_name, dict(row.get("payload") or {}), state)
            item = record.to_dict()
            item["started_elapsed_marker_s"] = round(started, 3)
            tool_records.append(item)
    bundle = ArtifactBundle(
        case_id=str(preflight.get("case_id") or target["target_name"]),
        target_input=target_input,
        preflight=preflight,
        workflow_plan=workflow_plan,
        tool_calls=tool_records,
        artifacts=dict(state.artifacts),
        validations=list(state.validations),
        safety_flags=sorted(set(state.safety_flags)),
        run_semantics="canonical_agent_controller",
    )
    verdict = emit_final_verdict(bundle).to_dict()
    write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
    write_json(run_dir / "final_verdict.json", verdict)
    return summarize_run(run_dir=run_dir, preflight=preflight, state=state, tool_records=tool_records, verdict=verdict)


def summarize_run(
    *,
    run_dir: Path,
    preflight: dict[str, Any],
    state: ToolExecutionState,
    tool_records: list[dict[str, Any]],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    verifier = dict(state.artifacts.get("route_verifier") or {})
    chemenzy = dict(state.artifacts.get("chemenzy") or {})
    route_audit = dict(state.artifacts.get("route_audit") or {})
    return {
        "target_name": str(state.target_input.get("target_name") or ""),
        "target_smiles": str(state.target_input.get("target_smiles") or ""),
        "run_dir": str(run_dir),
        "preflight": {
            "accepted": bool(preflight.get("accepted")),
            "canonical_smiles": preflight.get("canonical_smiles"),
            "isomeric_smiles": preflight.get("isomeric_smiles"),
            "inchi_key": preflight.get("inchi_key"),
            "profile": {
                "formula": (preflight.get("target_profile") or {}).get("formula"),
                "heavy_atoms": (preflight.get("target_profile") or {}).get("heavy_atoms"),
                "rings": (preflight.get("target_profile") or {}).get("rings"),
                "stereocenters": (preflight.get("target_profile") or {}).get("stereocenters"),
            },
            "risk_flags": preflight.get("initial_risk_flags") or [],
            "reasons": preflight.get("reasons") or [],
        },
        "tool_status": [
            {
                "tool_name": row.get("tool_name"),
                "status": row.get("status"),
                "elapsed_s": row.get("elapsed_s"),
                "reasons": row.get("reasons") or [],
            }
            for row in tool_records
        ],
        "chemenzy": {
            "accepted": bool(chemenzy.get("accepted", chemenzy.get("ok"))),
            "status": chemenzy.get("status"),
            "route_count": len(chemenzy.get("routes") or []),
            "search_status": chemenzy.get("search_status") or {},
        },
        "route_verifier": {
            "accepted": bool(verifier.get("accepted")),
            "route_status": verifier.get("route_status"),
            "route_count": verifier.get("route_count"),
            "accepted_route_count": verifier.get("accepted_route_count"),
            "rejected_route_count": verifier.get("rejected_route_count"),
            "best_route_rank": verifier.get("best_route_rank"),
            "reasons": verifier.get("reasons") or [],
        },
        "route_audit": {
            "route_status": route_audit.get("route_status"),
            "stock_audit_passed": bool(route_audit.get("stock_audit_passed")),
            "fake_closure_rejected": bool(route_audit.get("fake_closure_rejected")),
            "reasons": route_audit.get("reasons") or [],
        },
        "final_verdict": verdict,
        "artifact_refs": {
            "target_input": str(run_dir / "target_input.json"),
            "preflight": str(run_dir / "preflight.json"),
            "chemenzy_raw": str(run_dir / "chemenzy_native_raw_result.json"),
            "route_verifier": str(run_dir / "route_verifier_report.json"),
            "route_audit": str(run_dir / "route_audit.json"),
            "final_verdict": str(run_dir / "final_verdict.json"),
        },
    }


if __name__ == "__main__":
    main()
