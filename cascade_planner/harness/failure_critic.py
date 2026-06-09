"""Deterministic failure critic for agentic blackboard runs."""
from __future__ import annotations

from typing import Any


FAILURE_CRITIC_SCHEMA = "failure_critic_report.v1"


def compile_failure_critic_report(
    *,
    blackboard: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    case_id: str = "",
    target_name: str = "",
) -> dict[str, Any]:
    """Convert verifier/runtime failures into typed blackboard updates."""
    board = dict(blackboard or {})
    artifact_map = dict(artifacts or {})
    case = str(case_id or board.get("case_id") or "")
    target = str(target_name or ((board.get("target_profile") or {}).get("target_name")) or "")
    verifier = _latest_verifier(artifact_map)
    plugin_runtime = _latest_plugin_runtime(artifact_map)
    compiled = _compiled_downstream(artifact_map)

    source_reasons = _dedupe(
        [str(item) for item in verifier.get("reasons") or []]
        + [str(item) for item in plugin_runtime.get("reasons") or []]
    )
    if _plugin_product_hits_zero(plugin_runtime, compiled):
        source_reasons.append("plugin_product_hits=0")
    source_reasons = _dedupe(source_reasons)

    route_failures: list[dict[str, Any]] = []
    bridge_tasks: list[dict[str, Any]] = []
    terminal_blacklist: list[dict[str, Any]] = []
    blocked_directions: list[dict[str, Any]] = []
    next_action_bias: list[str] = []
    constraints: dict[str, Any] = {}

    if "large_atom_jump" in source_reasons:
        route_failures.append(_failure("large_atom_jump", verifier, "unexplained heavy-atom jump in route graph"))
        bridge_tasks.append(
            _bridge_task(
                case,
                target,
                "target_proximal_bridge_required",
                "Find a target-proximal intermediate that explains the large skeleton change.",
                priority="high",
            )
        )
        blocked_directions.append(
            {
                "schema_version": "agent_blocked_direction.v1",
                "direction": "current_route_family_without_core_bridge",
                "reason": "large_atom_jump",
            }
        )
        next_action_bias.extend(["generate_disconnection_hypotheses", "search_literature"])
        constraints["target_core_retention_required"] = True
        constraints["max_unexplained_heavy_atom_jump"] = 15

    if "literature_template_plugin_not_invoked" in source_reasons:
        route_failures.append(
            _failure(
                "literature_template_plugin_not_invoked",
                plugin_runtime,
                "literature template plugin was present but not exercised by backend",
            )
        )
        bridge_tasks.append(
            _bridge_task(
                case,
                target,
                "bridge_to_literature_product_required",
                "Find a target-side bridge before replaying exact source products.",
                priority="high",
            )
        )
        next_action_bias.extend(["generate_disconnection_hypotheses", "rank_analogical_hypotheses"])
        blocked_directions.append(
            {
                "schema_version": "agent_blocked_direction.v1",
                "direction": "exact_replay_without_target_bridge",
                "reason": "literature_template_plugin_not_invoked",
            }
        )
        constraints["exact_replay_priority"] = "lower_until_bridge_found"

    if "advanced_same_scaffold_terminal" in source_reasons:
        route_failures.append(_failure("advanced_same_scaffold_terminal", verifier, "advanced target-like terminal used as stock"))
        for item in verifier.get("rejected_terminal_list") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("reason") or "") != "advanced_same_scaffold_terminal":
                continue
            row = _terminal_blacklist_row(item)
            terminal_blacklist.append(row)
            bridge_tasks.append(
                {
                    "schema_version": "agent_bridge_task.v1",
                    "task_id": f"upstream_child:{row.get('canonical_smiles') or row.get('smiles')}",
                    "case_id": case,
                    "task_type": "upstream_terminal_synthesis",
                    "target_name": target,
                    "target_handle": "advanced_same_scaffold_terminal",
                    "required_bridge": "Find upstream synthesis for rejected same-scaffold terminal.",
                    "terminal": row,
                    "status": "open",
                    "priority": "high",
                    "required_verification": ["child_target_route_verifier", "parent_bridge_connectivity"],
                }
            )
        next_action_bias.extend(["expand_child_target", "generate_disconnection_hypotheses"])

    if "plugin_product_hits=0" in source_reasons:
        route_failures.append(_failure("plugin_product_hits=0", plugin_runtime or compiled, "source rows did not connect to target products"))
        bridge_tasks.append(
            _bridge_task(
                case,
                target,
                "target_side_bridge_before_source_replay",
                "Treat literature rows as disconnected until a target-proximal bridge is found.",
                priority="medium",
            )
        )
        next_action_bias.extend(["generate_disconnection_hypotheses", "search_literature"])
        constraints["literature_rows_connected"] = False

    accepted = bool(route_failures or bridge_tasks or terminal_blacklist or source_reasons)
    return {
        "schema_version": FAILURE_CRITIC_SCHEMA,
        "accepted": accepted,
        "case_id": case,
        "source_reasons": source_reasons,
        "route_failures": _dedupe_rows(route_failures),
        "bridge_tasks": _dedupe_rows(bridge_tasks),
        "terminal_blacklist": _dedupe_rows(terminal_blacklist),
        "blocked_directions": _dedupe_rows(blocked_directions),
        "next_action_bias": _dedupe(next_action_bias),
        "constraints": constraints,
        "semantics": {
            "critic_can_stop_flow": False,
            "solved_claim_allowed": False,
            "updates_blackboard_only": True,
        },
        "no_solved_claim": True,
        "reasons": [] if accepted else ["no_failure_evidence"],
    }


def _latest_verifier(artifacts: dict[str, Any]) -> dict[str, Any]:
    guided = artifacts.get("guided_chemenzy")
    if isinstance(guided, dict):
        verifier = guided.get("raw_route_verifier")
        if isinstance(verifier, dict) and verifier:
            return dict(verifier)
        result = guided.get("result")
        if isinstance(result, dict) and isinstance(result.get("raw_route_verifier"), dict):
            return dict(result["raw_route_verifier"])
    for key in ("route_verifier", "guided_route_verifier"):
        value = artifacts.get(key)
        if isinstance(value, dict) and value:
            return dict(value)
    chemenzy = artifacts.get("chemenzy")
    if isinstance(chemenzy, dict):
        embedded = chemenzy.get("raw_route_verifier")
        if isinstance(embedded, dict) and embedded:
            return dict(embedded)
        result = chemenzy.get("result")
        if isinstance(result, dict) and isinstance(result.get("raw_route_verifier"), dict):
            return dict(result["raw_route_verifier"])
    return {}


def _latest_plugin_runtime(artifacts: dict[str, Any]) -> dict[str, Any]:
    guided = artifacts.get("guided_chemenzy")
    if isinstance(guided, dict):
        runtime = guided.get("literature_template_plugin_runtime")
        if isinstance(runtime, dict):
            return dict(runtime)
        result = guided.get("result")
        if isinstance(result, dict) and isinstance(result.get("literature_template_plugin_runtime"), dict):
            return dict(result["literature_template_plugin_runtime"])
    return {}


def _compiled_downstream(artifacts: dict[str, Any]) -> dict[str, Any]:
    for key in ("compiled_downstream", "compiled_downstream_payload"):
        value = artifacts.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _plugin_product_hits_zero(plugin_runtime: dict[str, Any], compiled: dict[str, Any]) -> bool:
    if plugin_runtime:
        if plugin_runtime.get("enabled_in_request") and int(plugin_runtime.get("request_one_step_row_count") or 0) > 0:
            return int(plugin_runtime.get("added_candidates") or 0) == 0
    plugin = dict((compiled or {}).get("literature_template_plugin") or {})
    flags = dict(plugin.get("plugin_flags") or plugin)
    rows = flags.get("one_step_rows") or []
    product_hits = flags.get("product_hits")
    if rows and product_hits is not None:
        try:
            return int(product_hits) == 0
        except (TypeError, ValueError):
            return False
    return False


def _failure(reason: str, source: dict[str, Any], summary: str) -> dict[str, Any]:
    return {
        "schema_version": "agent_route_failure.v1",
        "reason": reason,
        "summary": summary,
        "route_status": str(source.get("route_status") or ""),
        "source_schema_version": str(source.get("schema_version") or ""),
    }


def _bridge_task(case_id: str, target_name: str, task_type: str, required_bridge: str, *, priority: str) -> dict[str, Any]:
    return {
        "schema_version": "agent_bridge_task.v1",
        "task_id": f"{task_type}:{target_name or case_id or 'target'}",
        "case_id": case_id,
        "task_type": task_type,
        "target_name": target_name,
        "target_handle": "target_side",
        "required_bridge": required_bridge,
        "status": "open",
        "priority": priority,
        "required_verification": ["target_equivalence", "parent_route_proof"],
    }


def _terminal_blacklist_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "agent_terminal_blacklist_entry.v1",
        "smiles": str(item.get("smiles") or ""),
        "canonical_smiles": str(item.get("canonical_smiles") or item.get("smiles") or ""),
        "heavy_atoms": int(item.get("heavy_atoms") or 0),
        "target_similarity": float(item.get("target_similarity") or 0.0),
        "reason": str(item.get("reason") or "advanced_same_scaffold_terminal"),
    }


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = (str(row.get("schema_version") or ""), str(row.get("task_id") or row.get("reason") or row.get("canonical_smiles") or row))
        if repr(key) in seen:
            continue
        seen.add(repr(key))
        out.append(row)
    return out
