"""Compile route-verifier failures into next-run search feedback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROUTE_FAILURE_FEEDBACK_SCHEMA = "route_failure_feedback.v1"


def compile_route_failure_feedback(
    verifier_report: dict[str, Any],
    *,
    case_id: str = "",
    target_name: str = "",
) -> dict[str, Any]:
    report = dict(verifier_report or {})
    terminal_blacklist: list[dict[str, Any]] = []
    frontier_targets: list[dict[str, Any]] = []
    query_hints: list[dict[str, Any]] = []

    for item in report.get("rejected_terminal_list") or []:
        if not isinstance(item, dict):
            continue
        row = _compound_row(item, source="rejected_terminal_list")
        terminal_blacklist.append(row)
        if row.get("reason") == "advanced_same_scaffold_terminal":
            frontier_targets.append(
                {
                    **row,
                    "frontier_role": "advanced_same_scaffold_terminal",
                    "required_action": "find_upstream_synthesis_or_disconnection",
                }
            )
        query_hints.append(_query_hint(row, target_name=target_name, hint_type="terminal_blacklist"))

    for event in report.get("failure_events") or []:
        if not isinstance(event, dict):
            continue
        reason = str(event.get("reason") or "")
        if reason == "large_atom_jump":
            query_hints.append(
                {
                    "schema_version": "route_failure_query_hint.v1",
                    "hint_type": "large_atom_jump",
                    "query": f"{target_name} upstream intermediate synthesis large skeleton construction".strip(),
                    "reason": "large_atom_jump",
                }
            )
            continue
        details = dict(event.get("details") or {})
        sample = dict(details.get("sample") or {})
        if not sample:
            continue
        row = _compound_row(sample, source=reason or "failure_event")
        if reason == "hidden_nonstock_reactants":
            terminal_blacklist.append(row)
            frontier_targets.append(
                {
                    **row,
                    "frontier_role": "hidden_nonstock_advanced_intermediate",
                    "required_action": "find_upstream_synthesis_or_disconnection",
                }
            )
            query_hints.append(_query_hint(row, target_name=target_name, hint_type="hidden_nonstock_intermediate"))

    terminal_blacklist = _dedupe_rows(terminal_blacklist)
    frontier_targets = _dedupe_rows(frontier_targets)
    query_hints = _dedupe_rows(query_hints)
    accepted = bool(terminal_blacklist or frontier_targets or query_hints)
    return {
        "schema_version": ROUTE_FAILURE_FEEDBACK_SCHEMA,
        "accepted": accepted,
        "case_id": str(case_id or report.get("case_id") or ""),
        "target_name": str(target_name or ""),
        "source_route_status": str(report.get("route_status") or ""),
        "source_reasons": [str(item) for item in report.get("reasons") or []],
        "terminal_blacklist": terminal_blacklist,
        "frontier_research_targets": frontier_targets,
        "query_hints": query_hints,
        "next_guided_policy_patch": {
            "terminal_blacklist": [row["canonical_smiles"] for row in terminal_blacklist if row.get("canonical_smiles")],
            "preferred_subgoals": [
                row["canonical_smiles"] for row in frontier_targets if row.get("canonical_smiles")
            ],
            "source_budget": {
                "active_failure_modes": [str(item) for item in report.get("reasons") or []],
                "terminal_blacklist_roles": [
                    "hidden_nonstock_advanced_intermediate",
                    "advanced_same_scaffold_terminal",
                ],
            },
        },
        "reasons": [] if accepted else ["no_route_failure_feedback"],
    }


def write_route_failure_feedback(feedback: dict[str, Any], *, output_dir: str | Path) -> str:
    path = Path(output_dir) / "route_failure_feedback.json"
    path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return str(path)


def _compound_row(item: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "schema_version": "route_failure_compound.v1",
        "smiles": str(item.get("smiles") or ""),
        "canonical_smiles": str(item.get("canonical_smiles") or item.get("smiles") or ""),
        "heavy_atoms": int(item.get("heavy_atoms") or 0),
        "target_similarity": float(item.get("target_similarity") or 0.0),
        "reason": str(item.get("reason") or source),
        "source": source,
    }


def _query_hint(row: dict[str, Any], *, target_name: str, hint_type: str) -> dict[str, Any]:
    smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
    return {
        "schema_version": "route_failure_query_hint.v1",
        "hint_type": hint_type,
        "query": f"{target_name} synthesis intermediate {smiles}".strip(),
        "smiles": smiles,
        "reason": str(row.get("reason") or ""),
    }


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
