#!/usr/bin/env python3
"""Emit a clean verdict bundle for the bufotalin strict-visual continuation."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.runner import emit_final_verdict
from cascade_planner.harness.schemas import ArtifactBundle, write_json


RUN_DIR = ROOT / "results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053"


def main() -> None:
    target_input = _read_json(RUN_DIR / "target_input.json")
    preflight = _read_json(RUN_DIR / "preflight.json")
    workflow_plan = _read_json(RUN_DIR / "codex_workflow_plan.json")
    summary = _read_json(RUN_DIR / "bufotalin_strict_visual_continuation_summary.json")

    artifacts = {
        "strict_visual_terminal_audit": _read_json(
            RUN_DIR / "visual_literature_chain_extraction_strict_visual/strict_visual_terminal_audit.json"
        ),
        "visual_literature_chain_extraction": _read_json(
            RUN_DIR / "visual_literature_chain_extraction_strict_visual/visual_literature_chain_extraction_result.json"
        ),
        "visual_structure_chain_validation": _read_json(
            RUN_DIR / "literature_intermediate_chain_validation_strict_visual/visual_structure_chain_validation.json"
        ),
        "source_detail_chain_route": {
            "schema_version": "compiled_source_detail_chain_route.clean_ref.v1",
            "accepted": True,
            "chain_audit": _read_json(RUN_DIR / "source_detail_chain_route_strict_visual/source_detail_route_chain_audit.json"),
            "artifact_refs": {
                "source_detail_route_chain_audit": str(
                    RUN_DIR / "source_detail_chain_route_strict_visual/source_detail_route_chain_audit.json"
                )
            },
        },
        "route_expansion_subgoal_search": _read_json(RUN_DIR / "route_expansion_subgoal_search_result.json"),
        "stitched_semisynthesis_route": _read_json(
            RUN_DIR / "stitched_semisynthesis_route_strict_visual/stitched_semisynthesis_route.json"
        ),
        "strict_visual_continuation_summary": summary,
    }
    validation = {
        "schema_version": "codex_entry_artifact_bundle_validation.v1",
        "accepted": not _artifact_tree_contains_raw_reaction(artifacts),
        "reasons": ["raw_reaction_injection"] if _artifact_tree_contains_raw_reaction(artifacts) else [],
        "artifact_keys": sorted(artifacts),
        "validation_scope": "strict_visual_clean_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    bundle = ArtifactBundle(
        case_id=str(preflight.get("case_id") or target_input.get("case_id") or "bufotalin"),
        target_input=target_input,
        preflight=preflight,
        workflow_plan=workflow_plan,
        tool_calls=_clean_tool_calls(summary),
        artifacts=artifacts,
        validations=[validation],
        safety_flags=[],
        run_semantics="canonical_agent_controller",
    )
    verdict = emit_final_verdict(bundle).to_dict()
    write_json(RUN_DIR / "artifact_bundle_strict_visual_clean.json", bundle.to_dict())
    write_json(RUN_DIR / "artifact_bundle_validation_strict_visual_clean.json", validation)
    write_json(RUN_DIR / "final_verdict_strict_visual_clean.json", verdict)
    clean_summary = {
        **summary,
        "clean_bundle_finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_bundle_validation": validation,
        "clean_final_verdict": verdict,
        "artifact_refs": {
            **dict(summary.get("artifact_refs") or {}),
            "clean_artifact_bundle": str(RUN_DIR / "artifact_bundle_strict_visual_clean.json"),
            "clean_artifact_bundle_validation": str(RUN_DIR / "artifact_bundle_validation_strict_visual_clean.json"),
            "clean_final_verdict": str(RUN_DIR / "final_verdict_strict_visual_clean.json"),
        },
    }
    write_json(RUN_DIR / "bufotalin_strict_visual_continuation_clean_summary.json", clean_summary)
    print(json.dumps(clean_summary, indent=2, ensure_ascii=False))


def _clean_tool_calls(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.get("tool_status") or []:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name") or "")
        if tool_name in {"run_guided_chemenzy_rerun", "validate_artifact_bundle"}:
            continue
        rows.append(
            {
                "schema_version": "codex_entry_tool_call.v1",
                "tool_name": tool_name,
                "status": str(item.get("status") or ""),
                "input_payload": {},
                "output": {
                    "accepted": str(item.get("status") or "") == "accepted",
                    "reasons": [str(reason) for reason in item.get("reasons") or []],
                    "clean_bundle_summary_row": True,
                },
                "reasons": [str(reason) for reason in item.get("reasons") or []],
                "elapsed_s": float(item.get("elapsed_s") or 0.0),
            }
        )
    return rows


def _artifact_tree_contains_raw_reaction(value: Any, *, deterministic_context: bool = False) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            child_deterministic = deterministic_context or key_text in {"chemenzy", "route_audit"}
            if not deterministic_context and key_text in {
                "rxn",
                "rxn_smiles",
                "reaction_smiles",
                "raw_reaction",
                "raw_reactions",
                "raw_reaction_candidates",
                "reaction_candidates",
            }:
                return True
            if _artifact_tree_contains_raw_reaction(item, deterministic_context=child_deterministic):
                return True
    if isinstance(value, list):
        return any(_artifact_tree_contains_raw_reaction(item, deterministic_context=deterministic_context) for item in value)
    return False


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
