"""Deterministic stitching audit for literature chains and subgoal routes."""
from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.harness.route_verifier import (
    is_accepted_route_verifier_report,
    is_precedent_supported_route_verifier_report,
    is_reaction_validated_route_verifier_report,
    verify_chemenzy_raw_routes,
)
from cascade_planner.harness.reaction_step_verifier import (
    is_precedent_supported_route,
    verify_reaction_route,
)
from cascade_planner.harness.source_text_companion import (
    validate_source_text_companion_binding,
)


RDLogger.DisableLog("rdApp.*")

STITCHED_SEMISYNTHESIS_ROUTE_SCHEMA = "stitched_semisynthesis_route.v1"
_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRUSTED_STEP_REGISTRY = Path(__file__).resolve().parent / "data" / "trusted_literature_step_registry.json"


def is_solved_stitched_semisynthesis_route(
    value: Any,
    *,
    expected_target_smiles: str = "",
) -> bool:
    """Recompute a stitched proof from its materialized, source-bound inputs."""
    route, recomputed, candidate_valid = _replay_stitched_semisynthesis_route(
        value,
        expected_target_smiles=expected_target_smiles,
    )
    if not candidate_valid:
        return False
    coverage = dict(route.get("frontier_coverage_audit") or {})
    recomputed_coverage = dict(recomputed.get("frontier_coverage_audit") or {})
    return bool(
        route.get("solved") is True
        and str(route.get("route_status") or "") == "solved"
        and recomputed.get("solved") is True
        and coverage.get("all_frontiers_precedent_supported") is True
        and recomputed_coverage.get("all_frontiers_precedent_supported") is True
        and int(coverage.get("precedent_supported_frontier_count") or 0)
        == int(
            recomputed_coverage.get("precedent_supported_frontier_count") or 0
        )
    )


def is_reaction_validated_stitched_semisynthesis_route(
    value: Any,
    *,
    expected_target_smiles: str = "",
) -> bool:
    """Return whether the complete stitch replays as a stock-closed L2 candidate.

    This is deliberately weaker than :func:`is_solved_stitched_semisynthesis_route`.
    It keeps useful L2 candidates visible without allowing them to satisfy the
    parent route's independent L3 precedent requirement.
    """

    _, _, candidate_valid = _replay_stitched_semisynthesis_route(
        value,
        expected_target_smiles=expected_target_smiles,
    )
    return candidate_valid


def _replay_stitched_semisynthesis_route(
    value: Any,
    *,
    expected_target_smiles: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Rebuild one stitch and validate its non-authoritative stored projection."""

    if not isinstance(value, dict):
        return {}, {}, False
    route = dict(value)
    inputs = route.get("proof_inputs")
    if not isinstance(inputs, dict):
        return route, {}, False
    target_smiles = str(expected_target_smiles or inputs.get("target_smiles") or "")
    if not target_smiles:
        return route, {}, False
    stored_expansion = inputs.get("route_expansion_result")
    if not isinstance(stored_expansion, dict):
        stored_expansion = {"subgoals": [dict(inputs.get("selected_subgoal") or {})]}
    recomputed = compile_stitched_semisynthesis_route(
        literature_chain_audit=dict(inputs.get("literature_chain_audit") or {}),
        route_expansion_result=dict(stored_expansion),
        subgoal_verifier=dict(inputs.get("provided_subgoal_verifier") or {}),
        subgoal_raw_result=dict(inputs.get("subgoal_raw_result") or {}),
        target_smiles=target_smiles,
        target_name=str(inputs.get("target_name") or ""),
        case_id=str(inputs.get("case_id") or ""),
    )
    coverage = dict(route.get("frontier_coverage_audit") or {})
    recomputed_coverage = dict(recomputed.get("frontier_coverage_audit") or {})
    candidate_valid = bool(
        route.get("schema_version") == STITCHED_SEMISYNTHESIS_ROUTE_SCHEMA
        and route.get("accepted") is True
        and recomputed.get("accepted") is True
        and coverage.get("schema_version") == "stitched_frontier_coverage_audit.v1"
        and coverage.get("accepted") is True
        and int(coverage.get("frontier_count") or 0) > 0
        and int(coverage.get("frontier_count") or 0)
        == int(recomputed_coverage.get("frontier_count") or 0)
        and int(coverage.get("closed_frontier_count") or 0)
        == int(recomputed_coverage.get("closed_frontier_count") or 0)
        and _same_compound(
            _compound_identity(str((route.get("target") or {}).get("smiles") or "")),
            _compound_identity(target_smiles),
        )
        and int((route.get("combined_route") or {}).get("combined_step_count") or 0)
        == int((recomputed.get("combined_route") or {}).get("combined_step_count") or 0)
    )
    return route, recomputed, candidate_valid


def compile_stitched_semisynthesis_route(
    *,
    literature_chain_audit: dict[str, Any] | str | Path | None,
    subgoal_verifier: dict[str, Any] | str | Path | None = None,
    subgoal_raw_result: dict[str, Any] | str | Path | None = None,
    route_expansion_result: dict[str, Any] | str | Path | None = None,
    output_dir: str | Path | None = None,
    case_id: str = "",
    target_smiles: str = "",
    target_name: str = "",
    subgoal_name: str = "",
) -> dict[str, Any]:
    """Audit whether a literature terminal and solved subgoal route can be joined.

    A subgoal route may only close the full natural-product route when the
    literature-chain terminal is exactly the same compound as the verified
    subgoal target. Names and labels are advisory; the acceptance gate is
    isomeric canonical SMILES plus InChIKey.
    """
    chain = _load_jsonish(literature_chain_audit)
    expansion = _load_jsonish(route_expansion_result)
    chain_summary = _literature_chain_summary(
        chain,
        expected_target_smiles=target_smiles,
    )
    provided_verifier = _load_jsonish(subgoal_verifier)
    provided_raw = _load_jsonish(subgoal_raw_result)
    frontier_smiles = [
        str(item)
        for item in chain_summary.get("graph_terminal_frontier") or []
        if str(item or "").strip()
    ]
    frontier_inputs = _frontier_subgoal_inputs(
        expansion=expansion,
        frontier_smiles=frontier_smiles,
        primary_terminal=dict(chain_summary.get("terminal") or {}),
        provided_verifier=provided_verifier,
        provided_raw=provided_raw,
    )
    primary_input = next(
        (
            item
            for item in frontier_inputs
            if _same_compound(
                _compound_identity(str(item.get("frontier_smiles") or "")),
                dict(chain_summary.get("terminal") or {}),
            )
        ),
        frontier_inputs[0] if frontier_inputs else {},
    )
    selected = dict(primary_input.get("selected_subgoal") or {})
    verifier = dict(primary_input.get("provided_verifier") or {})
    raw = dict(primary_input.get("raw_result") or {})

    reasons: list[str] = []
    warnings: list[str] = []
    artifact_refs = _artifact_refs(
        literature_chain_audit=literature_chain_audit,
        subgoal_verifier=subgoal_verifier,
        subgoal_raw_result=subgoal_raw_result,
        route_expansion_result=route_expansion_result,
        selected_subgoal=selected,
    )

    if not chain:
        reasons.append("literature_chain_missing")
    elif not chain_summary["chain_accepted"]:
        reasons.append("literature_chain_not_accepted")
    if chain_summary["step_count"] <= 0:
        reasons.append("literature_chain_materialized_steps_missing")
    if not chain_summary["graph_connected"]:
        reasons.append("literature_chain_graph_disconnected")
    if not chain_summary["terminal_bound_to_steps"]:
        reasons.append("literature_terminal_not_bound_to_steps")
    if not chain_summary["source_bound"]:
        reasons.append("literature_chain_source_missing")
    if not chain_summary["source_detail_schema_valid"]:
        reasons.append("literature_chain_not_strict_source_detail_schema")
    if chain_summary["invalid_provenance_step_count"]:
        reasons.append("literature_chain_step_provenance_not_revalidated")
    if not chain_summary["reaction_validated"]:
        reasons.append("literature_chain_reaction_steps_not_validated")
    if not chain_summary["terminal_reached"]:
        reasons.append("literature_chain_terminal_not_reached")
    if not chain_summary["terminal"]["valid"]:
        reasons.append("literature_terminal_invalid_or_missing")

    target_audit = _target_identity_audit(
        requested_target_smiles=target_smiles,
        literature_target_smiles=str(chain.get("target_smiles") or ""),
    )
    if target_audit["required"] and not target_audit["target_match"]:
        reasons.append("target_input_literature_chain_mismatch")

    subgoal_closures: list[dict[str, Any]] = []
    normalized_expansion_subgoals: list[dict[str, Any]] = []
    for frontier_index, item in enumerate(frontier_inputs, start=1):
        frontier = str(item.get("frontier_smiles") or "")
        item_selected = dict(item.get("selected_subgoal") or {})
        item_verifier = dict(item.get("provided_verifier") or {})
        item_raw = dict(item.get("raw_result") or {})
        summary = _subgoal_summary(
            verifier=item_verifier,
            raw=item_raw,
            selected_subgoal=item_selected,
            subgoal_name=subgoal_name if frontier_index == 1 else "",
            expected_target_smiles=frontier,
        )
        terminal_identity = _compound_identity(frontier)
        match = _terminal_subgoal_match_audit(
            terminal=terminal_identity,
            subgoal=summary["target"],
            parent_bridge=summary["parent_bridge"],
        )
        closure = {
            **summary,
            "frontier_index": frontier_index,
            "frontier": terminal_identity,
            "terminal_match_audit": match,
        }
        subgoal_closures.append(closure)
        normalized_expansion_subgoals.append(
            {
                **item_selected,
                "frontier_smiles": frontier,
                "verifier": item_verifier,
                "raw_result": item_raw,
            }
        )
        if not item_verifier:
            reasons.append("subgoal_verifier_missing")
        else:
            if not summary["verifier_accepted"]:
                reasons.append("subgoal_verifier_not_accepted")
            if not summary["reaction_validated"]:
                reasons.append("subgoal_reaction_steps_not_validated")
            if not summary["target_match"]:
                reasons.append("subgoal_target_not_verified")
            if not summary["raw_solved"] or summary["best_route_step_count"] <= 0:
                reasons.append("subgoal_materialized_route_missing")
            if not summary["route_materialization_complete"]:
                reasons.append("subgoal_route_materialization_mismatch")
            if summary["accepted_route_count"] <= 0:
                reasons.append("subgoal_has_no_verified_route")
            if not summary["stock_audit_passed"]:
                reasons.append("subgoal_stock_audit_not_passed")
            if not summary["provided_verifier_matched_reverification"]:
                reasons.append("subgoal_verifier_reverification_mismatch")
        if summary["verifier_reasons"]:
            warnings.extend(
                f"subgoal_verifier:{frontier_index}:{value}"
                for value in summary["verifier_reasons"]
            )
        if not match["accepted"]:
            reasons.append("literature_terminal_subgoal_target_mismatch")

    if not frontier_inputs:
        reasons.append("literature_terminal_frontier_missing")
    frontier_coverage_passed = bool(
        frontier_inputs
        and len(subgoal_closures) == len(frontier_smiles)
        and all(
            row.get("verifier_accepted") is True
            and row.get("reaction_validated") is True
            and row.get("stock_audit_passed") is True
            and (row.get("terminal_match_audit") or {}).get("accepted") is True
            for row in subgoal_closures
        )
    )
    if not frontier_coverage_passed:
        reasons.append("literature_chain_has_unclosed_precursors")
    precedent_supported_frontier_count = sum(
        1
        for row in subgoal_closures
        if row.get("precedent_supported") is True
    )
    all_frontiers_precedent_supported = bool(
        frontier_coverage_passed
        and subgoal_closures
        and precedent_supported_frontier_count == len(subgoal_closures)
    )

    subgoal_summary = next(
        (
            row
            for row in subgoal_closures
            if _same_compound(dict(row.get("frontier") or {}), dict(chain_summary.get("terminal") or {}))
        ),
        subgoal_closures[0] if subgoal_closures else _subgoal_summary(
            verifier={}, raw={}, selected_subgoal={}, subgoal_name=subgoal_name
        ),
    )
    terminal_match = dict(subgoal_summary.get("terminal_match_audit") or {})

    accepted = not sorted(set(reasons))
    precedent_supported = bool(
        accepted
        and chain_summary.get("precedent_supported") is True
        and all_frontiers_precedent_supported
    )
    if accepted and not all_frontiers_precedent_supported:
        warnings.append("subgoal_route_precedent_support_incomplete")
    if accepted and chain_summary.get("precedent_supported") is not True:
        warnings.append("literature_route_precedent_support_incomplete")
    literature_step_count = int(chain_summary["step_count"])
    subgoal_step_count = sum(int(row.get("best_route_step_count") or 0) for row in subgoal_closures)
    parent_bridge_step_count = sum(
        1 for row in subgoal_closures if (row.get("terminal_match_audit") or {}).get("parent_bridge_accepted")
    )
    result = {
        "schema_version": STITCHED_SEMISYNTHESIS_ROUTE_SCHEMA,
        "accepted": accepted,
        "solved": precedent_supported,
        "route_status": (
            "solved"
            if precedent_supported
            else "reaction_validated_l2_candidate"
            if accepted
            else _failure_status(reasons)
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id or str(chain.get("case_id") or ""),
        "target": {
            "name": target_name,
            "smiles": target_smiles or str(chain.get("target_smiles") or ""),
            "identity_audit": target_audit,
        },
        "literature_chain": chain_summary,
        "subgoal_closure": subgoal_summary,
        "subgoal_closures": subgoal_closures,
        "terminal_match_audit": terminal_match,
        "frontier_coverage_audit": {
            "schema_version": "stitched_frontier_coverage_audit.v1",
            "accepted": frontier_coverage_passed,
            "frontier_count": len(frontier_smiles),
            "closed_frontier_count": sum(
                1
                for row in subgoal_closures
                if row.get("verifier_accepted") is True
                and row.get("reaction_validated") is True
                and row.get("stock_audit_passed") is True
                and (row.get("terminal_match_audit") or {}).get("accepted") is True
            ),
            "precedent_supported_frontier_count": (
                precedent_supported_frontier_count
            ),
            "all_frontiers_precedent_supported": (
                all_frontiers_precedent_supported
            ),
            "frontier_smiles": frontier_smiles,
        },
        "stock_audit_passed": bool(
            accepted and subgoal_closures and all(row.get("stock_audit_passed") is True for row in subgoal_closures)
        ),
        "combined_route": {
            "route_type": "stitched_semisynthesis",
            "direction": "stock_to_subgoal_terminal_then_literature_to_target",
            "literature_step_count": literature_step_count,
            "subgoal_route_step_count": subgoal_step_count,
            "parent_bridge_step_count": parent_bridge_step_count,
            "combined_step_count": literature_step_count + subgoal_step_count + parent_bridge_step_count,
            "segments": [
                *[
                    {
                        "segment_id": f"subgoal_stock_closure_{index}",
                        "role": "stock_to_literature_frontier",
                        "status": (
                            "precedent_supported"
                            if row.get("precedent_supported")
                            else "reaction_validated"
                            if row.get("reaction_validated")
                            else "graph_and_stock_closed"
                            if row.get("verifier_accepted")
                            else "not_verified"
                        ),
                        "target_smiles": str((row.get("target") or {}).get("input_smiles") or ""),
                        "best_route_rank": row.get("best_route_rank"),
                        "step_count": int(row.get("best_route_step_count") or 0),
                    }
                    for index, row in enumerate(subgoal_closures, start=1)
                ],
                *[
                    segment
                    for row in subgoal_closures
                    for segment in _mechanistic_parent_bridge_segments(dict(row.get("terminal_match_audit") or {}))
                ],
                {
                    "segment_id": "source_detail_literature_chain",
                    "role": "literature_frontiers_to_target",
                    "status": (
                        "precedent_supported"
                        if chain_summary["precedent_supported"]
                        else "reaction_validated"
                        if chain_summary["reaction_validated"]
                        else "not_reaction_validated"
                    ),
                    "terminal_smiles": chain_summary["terminal"]["input_smiles"],
                    "terminal_frontier_smiles": frontier_smiles,
                    "target_smiles": str(chain.get("target_smiles") or target_smiles or ""),
                    "step_count": literature_step_count,
                    "source_ref": chain_summary["source_ref"],
                },
            ],
        },
        "source_policy": {
            "terminal_identity_match_required": True,
            "every_literature_frontier_requires_verified_closure": True,
            "mechanistic_parent_bridge_allowed": False,
            "mechanistic_parent_bridge_requires_explicit_transform_validation": True,
            "mechanistic_parent_bridge_requires_terminal_parent_identity_match": True,
            "mechanistic_parent_bridge_requires_solved_subgoal": True,
            "mechanistic_parent_bridge_is_not_exact_literature_segment": True,
            "subgoal_solved_does_not_imply_target_solved_without_stitch": True,
            "literature_segment_requires_source_detail_chain": True,
            "literature_segment_requires_reaction_validation": True,
            "literature_segment_requires_l3_precedent_for_solved": True,
            "subgoal_segment_requires_route_verifier": True,
            "subgoal_segment_requires_l3_precedent_for_solved": True,
            "l2_stitch_remains_displayable_but_not_solved": True,
            "final_verdict_authority": "deterministic_validators",
            "production_write_blocked": True,
        },
        "proof_inputs": {
            "schema_version": "stitched_semisynthesis_proof_inputs.v1",
            "target_smiles": target_smiles or str(chain.get("target_smiles") or ""),
            "target_name": target_name,
            "case_id": case_id or str(chain.get("case_id") or ""),
            "literature_chain_audit": chain,
            "route_expansion_result": {"subgoals": normalized_expansion_subgoals},
            "selected_subgoal": selected,
            "provided_subgoal_verifier": verifier,
            "subgoal_raw_result": raw,
        },
        "artifact_refs": artifact_refs,
        "warnings": sorted(set(warnings)),
        "reasons": sorted(set(reasons)),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "stitched_semisynthesis_route.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _load_jsonish(value: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    path = Path(value)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _frontier_subgoal_inputs(
    *,
    expansion: dict[str, Any],
    frontier_smiles: list[str],
    primary_terminal: dict[str, Any],
    provided_verifier: dict[str, Any],
    provided_raw: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(item) for item in expansion.get("subgoals") or [] if isinstance(item, dict)]
    primary = _select_subgoal(expansion, terminal=primary_terminal)
    primary_smiles = str(primary_terminal.get("canonical_isomeric_smiles") or "")
    out: list[dict[str, Any]] = []
    for frontier in frontier_smiles:
        selected = next(
            (
                row
                for row in rows
                if _same_compound(
                    _compound_identity(_subgoal_target_smiles(row)),
                    _compound_identity(frontier),
                )
            ),
            {},
        )
        if not selected and primary and _same_compound(
            _compound_identity(_subgoal_target_smiles(primary)),
            _compound_identity(frontier),
        ):
            selected = dict(primary)
        use_fallback = bool(
            (len(frontier_smiles) == 1 or frontier == primary_smiles)
            and (provided_verifier or provided_raw)
        )
        verifier = dict(provided_verifier) if use_fallback and provided_verifier else dict(selected.get("verifier") or {})
        raw = dict(provided_raw) if use_fallback and provided_raw else _subgoal_raw_result(selected)
        out.append(
            {
                "frontier_smiles": frontier,
                "selected_subgoal": selected,
                "provided_verifier": verifier,
                "raw_result": raw,
            }
        )
    return out


def _subgoal_target_smiles(row: dict[str, Any]) -> str:
    verifier = dict(row.get("verifier") or {})
    audit = dict(verifier.get("target_equivalence_audit") or {})
    raw = _subgoal_raw_result(row)
    selected = dict(row.get("subgoal") or {})
    return str(
        audit.get("request_canonical_isomeric_smiles")
        or audit.get("request_target_smiles")
        or raw.get("target")
        or raw.get("target_smiles")
        or selected.get("smiles")
        or ""
    )


def _subgoal_raw_result(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("raw_result", "result"):
        value = row.get(key)
        if isinstance(value, dict) and (value.get("routes") or value.get("target") or value.get("result")):
            return dict(value)
    path = str(row.get("raw_result_path") or "").strip()
    return _load_jsonish(path) if path else {}


def _select_subgoal(expansion: dict[str, Any], *, terminal: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = [dict(item) for item in expansion.get("subgoals") or [] if isinstance(item, dict)]
    accepted = [
        row
        for row in rows
        if row.get("accepted") or (isinstance(row.get("verifier"), dict) and row["verifier"].get("accepted"))
    ]
    if terminal:
        for row in accepted or rows:
            if _subgoal_row_can_connect_terminal(row, terminal=terminal):
                return dict(row)
    return dict((accepted or rows or [{}])[0])


def _subgoal_row_can_connect_terminal(row: dict[str, Any], *, terminal: dict[str, Any]) -> bool:
    verifier = dict(row.get("verifier") or {})
    selected_target = dict(row.get("subgoal") or {})
    target_smiles = str(
        (verifier.get("target_equivalence_audit") or {}).get("request_target_smiles")
        or selected_target.get("smiles")
        or ""
    )
    target = _compound_identity(target_smiles)
    if _same_compound(terminal, target):
        return True
    parent_bridge = _parent_bridge_summary(selected_target, target_smiles=target_smiles)
    return bool(
        parent_bridge.get("present")
        and parent_bridge.get("transform_allowed")
        and _same_compound(terminal, dict(parent_bridge.get("parent") or {}))
    )


def _literature_chain_summary(
    chain: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> dict[str, Any]:
    raw_steps = chain.get("chain") or chain.get("steps") or []
    candidate_steps = [dict(row) for row in raw_steps if isinstance(row, dict)] if isinstance(raw_steps, list) else []
    source_detail_schema_valid = str(chain.get("schema_version") or "") == "source_detail_route_chain_audit.v1"
    chemically_materialized = [row for row in candidate_steps if _materialized_literature_step(row)]
    steps = [row for row in chemically_materialized if _validated_source_detail_literature_step(row)]
    invalid_step_count = len(candidate_steps) - len(steps)
    invalid_provenance_step_count = len(chemically_materialized) - len(steps)
    terminal_smiles = (
        str(chain.get("terminal_smiles") or "")
        or str(chain.get("final_reactant_smiles") or "")
        or _last_main_reactant(steps)
    )
    terminal_name = (
        str(chain.get("terminal_name") or "")
        or str(chain.get("final_reactant_name") or "")
        or _last_main_reactant_name(steps)
    )
    explicit_accepted = chain.get("accepted") if "accepted" in chain else None
    terminal_reached = bool(chain.get("terminal_reached") or chain.get("chain_complete_to_literature_start"))
    source_ref = str(chain.get("source_ref") or "").strip()
    source_bound = bool(steps and all(_validated_source_detail_literature_step(row) for row in steps))
    graph_audit = _literature_chain_graph_audit(
        steps,
        target_smiles=str(expected_target_smiles or chain.get("target_smiles") or ""),
        terminal_smiles=terminal_smiles,
    )
    precedent_bindings: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(steps):
        binding = _trusted_literature_step_precedent(step)
        if binding:
            step_id = str(step.get("step_id") or step.get("id") or f"step:{index}")
            precedent_bindings[step_id] = binding
    reaction_validation = verify_reaction_route(
        steps,
        graph_and_stock_closed=False,
        trusted_precedent_bindings=precedent_bindings,
    )
    reaction_validated = reaction_validation.get("accepted") is True
    precedent_supported = is_precedent_supported_route(reaction_validation)
    chain_accepted = bool(
        explicit_accepted is True
        and source_detail_schema_valid
        and steps
        and invalid_step_count == 0
        and terminal_reached
        and source_bound
        and graph_audit["graph_connected"]
        and graph_audit["terminal_bound_to_steps"]
        and reaction_validated
    )
    return {
        "schema_version": str(chain.get("schema_version") or ""),
        "accepted": chain_accepted,
        "chain_accepted": chain_accepted,
        "source_ref": source_ref,
        "source_bound": source_bound,
        "source_detail_schema_valid": source_detail_schema_valid,
        "graph_connected": graph_audit["graph_connected"],
        "terminal_bound_to_steps": graph_audit["terminal_bound_to_steps"],
        "frontier_closed_to_terminal": graph_audit["frontier_closed_to_terminal"],
        "graph_terminal_frontier": graph_audit["terminal_frontier"],
        "reaction_validated": reaction_validated,
        "precedent_supported": precedent_supported,
        "reaction_validation": reaction_validation,
        "verification_level": str(
            reaction_validation.get("proof_level") or "L0_materialized"
        ),
        "terminal_frontiers": [
            _compound_identity(smiles) for smiles in graph_audit["terminal_frontier"]
        ],
        # Counts are derived from materialized rows; external summaries cannot
        # manufacture route evidence by claiming a positive step_count.
        "step_count": len(steps),
        "claimed_step_count": chain.get("step_count"),
        "invalid_step_count": invalid_step_count,
        "invalid_provenance_step_count": invalid_provenance_step_count,
        "terminal_name": terminal_name,
        "terminal": _compound_identity(terminal_smiles),
        "terminal_reached": terminal_reached,
        "reasons": [str(item) for item in chain.get("reasons") or []],
    }


def _materialized_literature_step(step: dict[str, Any]) -> bool:
    reaction = str(step.get("reaction_smiles") or "").strip()
    if ">>" in reaction:
        left, right = reaction.split(">>", 1)
        reactants = [item for item in left.split(".") if item.strip()]
        return bool(
            reactants
            and _compound_identity(right).get("valid")
            and all(_compound_identity(item).get("valid") for item in reactants)
        )
    product = str(
        step.get("product_smiles")
        or step.get("product")
        or step.get("final_product_smiles")
        or ""
    ).strip()
    raw_reactants = (
        step.get("reactant_smiles")
        or step.get("precursor_smiles")
        or step.get("reactants")
        or step.get("main_reactant_smiles")
        or step.get("main_reactant")
        or []
    )
    if isinstance(raw_reactants, str):
        reactants = [raw_reactants]
    elif isinstance(raw_reactants, list):
        reactants = [str(item or "") for item in raw_reactants]
    else:
        reactants = []
    reactants = [item for item in reactants if item.strip()]
    return bool(
        _compound_identity(product).get("valid")
        and reactants
        and all(_compound_identity(item).get("valid") for item in reactants)
    )


def _validated_source_detail_literature_step(step: dict[str, Any]) -> bool:
    template_id = str(step.get("source_template_id") or "").strip()
    validation = dict(step.get("exact_step_validation") or {})
    evidence = [dict(item) for item in step.get("source_evidence") or [] if isinstance(item, dict)]
    return bool(
        template_id.startswith("source_detail_exact_step:")
        and step.get("source_detail_exact_step") is True
        and str(step.get("relation_type") or "") == "exact"
        and str(step.get("source_ref") or "").strip()
        and validation.get("schema_version") == "template_validation_report.v1"
        and validation.get("accepted") is True
        and validation.get("allowed_for_one_step_source") is True
        and str(validation.get("source_template_id") or "") == template_id
        and not validation.get("reasons")
        and any(_materialized_source_evidence_valid(item) for item in evidence)
        and _trusted_literature_step_binding(step, evidence=evidence)
    )


def is_validated_source_detail_literature_step(value: Any) -> bool:
    """Public presentation gate for a proof-eligible exact literature row."""
    return bool(
        isinstance(value, dict)
        and _materialized_literature_step(value)
        and _validated_source_detail_literature_step(value)
    )


def is_materialized_source_bound_literature_step(value: Any) -> bool:
    """Return whether a source-bound row may enter search without precedent.

    This is intentionally weaker than
    :func:`is_validated_source_detail_literature_step`: it replays the exact
    structure, document, page, manifest, PDF, and image bindings, but does not
    consult the curated precedent registry.  Consumers may use it only for L0
    search admission; it grants neither literature-exact nor L3 authority.
    """

    if not isinstance(value, dict) or not _materialized_literature_step(value):
        return False
    row = dict(value)
    template_id = str(row.get("source_template_id") or "").strip()
    validation = dict(row.get("exact_step_validation") or {})
    evidence = [
        dict(item)
        for item in row.get("source_evidence") or []
        if isinstance(item, dict)
    ]
    return bool(
        template_id.startswith("source_detail_exact_step:")
        and row.get("source_detail_exact_step") is True
        and str(row.get("relation_type") or "") == "exact"
        and str(row.get("source_ref") or "").strip()
        and validation.get("schema_version") == "template_validation_report.v1"
        and validation.get("accepted") is True
        and validation.get("allowed_for_one_step_source") is True
        and str(validation.get("source_template_id") or "") == template_id
        and not validation.get("reasons")
        and any(_materialized_source_evidence_valid(item) for item in evidence)
    )


def _trusted_literature_step_binding(
    step: dict[str, Any],
    *,
    evidence: list[dict[str, Any]],
) -> bool:
    """Match a reaction edge against an out-of-band trusted curation registry.

    The registry path is process configuration, never an input field carried by
    a model-authored proof.  Consequently a PDF/page manifest alone cannot
    validate arbitrary product/reactant SMILES.
    """
    return bool(_trusted_literature_step_precedent(step, evidence=evidence))


def _trusted_literature_step_precedent(
    step: dict[str, Any],
    *,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    configured = str(os.environ.get("AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY") or "").strip()
    path = Path(configured).expanduser() if configured else _DEFAULT_TRUSTED_STEP_REGISTRY
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema_version") != "trusted_literature_step_registry.v1":
        return {}
    reaction_digest = _literature_step_chemistry_digest(step)
    source_ref = str(step.get("source_ref") or "").strip().lower()
    evidence_rows = evidence
    if evidence_rows is None:
        evidence_rows = [
            dict(item)
            for item in step.get("source_evidence") or []
            if isinstance(item, dict)
        ]
    valid_evidence = [
        row for row in evidence_rows if _materialized_source_evidence_valid(row)
    ]
    for raw in payload.get("bindings") or []:
        if not isinstance(raw, dict):
            continue
        binding = dict(raw)
        authority = dict(binding.get("authority") or {})
        if (
            binding.get("status") != "approved"
            or authority.get("type") not in {"human_curator", "deterministic_structure_parser"}
            or not str(authority.get("id") or "").strip()
            or str(binding.get("reaction_digest") or "").lower() != reaction_digest
            or str(binding.get("source_ref") or "").strip().lower() != source_ref
        ):
            continue
        if any(_binding_matches_evidence(binding, row) for row in valid_evidence):
            authority_type = str(authority.get("type") or "")
            return {
                "schema_version": "trusted_precedent_binding.v1",
                "accepted": True,
                "authority": authority_type,
                "authority_id": str(authority.get("id") or ""),
                "binding_id": str(binding.get("binding_id") or ""),
                "reaction_digest": reaction_digest,
                "source_ref": source_ref,
            }
    return {}


def _literature_step_chemistry_digest(step: dict[str, Any]) -> str:
    product, reactants = _literature_step_edge(step)
    if not product or not reactants:
        return ""
    payload = {
        "product_canonical_isomeric_smiles": product,
        "reactant_canonical_isomeric_smiles": sorted(reactants),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding_matches_evidence(binding: dict[str, Any], row: dict[str, Any]) -> bool:
    companion = binding.get("source_text_companion")
    if companion and not validate_source_text_companion_binding(
        companion,
        expected_source_ref=str(binding.get("source_ref") or ""),
    ):
        return False
    return bool(
        str(binding.get("document_id") or "") == str(row.get("document_id") or "")
        and str(binding.get("source_pdf_sha256") or "").lower()
        == str(row.get("source_pdf_sha256") or "").lower()
        and int(binding.get("page_number") or 0) == int(row.get("page_number") or 0)
        and str(binding.get("image_sha256") or "").lower()
        == str(row.get("image_sha256") or "").lower()
    )


def _materialized_source_evidence_valid(row: dict[str, Any]) -> bool:
    if row.get("schema_version") != "materialized_source_evidence.v1":
        return False
    manifest_path = Path(str(row.get("manifest_path") or "")).expanduser()
    if not manifest_path.is_file() or manifest_path.suffix.lower() != ".json":
        return False
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "literature_pdf_structure_evidence.v1":
        return False
    if manifest.get("accepted") is not True:
        return False
    binding = dict(manifest.get("source_binding_audit") or {})
    matched_document_ids = [str(item) for item in binding.get("matched_document_ids") or [] if str(item).strip()]
    source_ref = str(row.get("source_ref") or "").strip().lower()
    manifest_source_ref = str(manifest.get("source_ref") or binding.get("source_ref") or "").strip().lower()
    if not (
        binding.get("schema_version") == "local_pdf_source_binding_audit.v1"
        and binding.get("accepted") is True
        and int(binding.get("matched_source_count") or 0) > 0
        and matched_document_ids
        and str(row.get("document_id") or "") in matched_document_ids
        and source_ref
        and source_ref == manifest_source_ref
    ):
        return False
    if hashlib.sha256(manifest_bytes).hexdigest() != str(row.get("manifest_sha256") or "").lower():
        return False

    pdf_path = Path(str(manifest.get("source_pdf_path") or "")).expanduser()
    pdf_sha = str(manifest.get("source_pdf_sha256") or "").lower()
    if (
        not pdf_path.is_file()
        or pdf_path.suffix.lower() != ".pdf"
        or str(pdf_path.resolve()) != str(Path(str(row.get("source_pdf_path") or "")).expanduser().resolve())
        or pdf_sha != str(row.get("source_pdf_sha256") or "").lower()
        or _file_sha256(pdf_path) != pdf_sha
    ):
        return False
    try:
        with pdf_path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                return False
    except OSError:
        return False

    page_number = int(row.get("page_number") or 0)
    page = next(
        (
            dict(item)
            for item in manifest.get("rendered_pages") or []
            if isinstance(item, dict) and int(item.get("page_number") or 0) == page_number
        ),
        {},
    )
    image_path = Path(str(page.get("image_path") or "")).expanduser()
    image_sha = str(page.get("sha256") or "").lower()
    if not (
        page_number > 0
        and image_path.is_file()
        and str(image_path.resolve()) == str(Path(str(row.get("image_path") or "")).expanduser().resolve())
        and image_sha == str(row.get("image_sha256") or "").lower()
        and _file_sha256(image_path) == image_sha
        and _valid_image_file(image_path)
    ):
        return False
    return True


def _file_sha256(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return ""
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_SHA256_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    value = digest.hexdigest()
    _FILE_SHA256_CACHE[key] = value
    return value


def _valid_image_file(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _literature_chain_graph_audit(
    steps: list[dict[str, Any]],
    *,
    target_smiles: str,
    terminal_smiles: str,
) -> dict[str, Any]:
    target = _identity_canonical(target_smiles)
    terminal = _identity_canonical(terminal_smiles)
    frontier = {target} if target else set()
    remaining = set(range(len(steps)))
    progressed = True
    while progressed:
        progressed = False
        for index in list(remaining):
            product, reactants = _literature_step_edge(steps[index])
            if not product or product not in frontier:
                continue
            frontier.discard(product)
            frontier.update(reactants)
            remaining.remove(index)
            progressed = True
    return {
        "graph_connected": bool(target and steps and not remaining),
        "terminal_bound_to_steps": bool(terminal and terminal in frontier),
        "frontier_closed_to_terminal": bool(terminal and frontier == {terminal}),
        "terminal_frontier": sorted(frontier),
    }


def _literature_step_edge(step: dict[str, Any]) -> tuple[str, list[str]]:
    reaction = str(step.get("reaction_smiles") or "").strip()
    if ">>" in reaction:
        left, right = reaction.split(">>", 1)
        product = _identity_canonical(right)
        reactants = [
            _identity_canonical(item)
            for item in left.split(".")
            if item.strip()
        ]
        return product, [item for item in reactants if item]
    product = _identity_canonical(
        str(step.get("product_smiles") or step.get("product") or step.get("final_product_smiles") or "")
    )
    raw_reactants = (
        step.get("reactant_smiles")
        or step.get("precursor_smiles")
        or step.get("reactants")
        or step.get("main_reactant_smiles")
        or step.get("main_reactant")
        or []
    )
    values = [raw_reactants] if isinstance(raw_reactants, str) else list(raw_reactants or [])
    reactants = [
        _identity_canonical(str(item or ""))
        for item in values
        if str(item or "").strip()
    ]
    return product, [item for item in reactants if item]


def _identity_canonical(smiles: str) -> str:
    return str(_compound_identity(smiles).get("canonical_isomeric_smiles") or "")


def _last_main_reactant(steps: Any) -> str:
    if not isinstance(steps, list) or not steps:
        return ""
    last = steps[-1]
    if not isinstance(last, dict):
        return ""
    return str(last.get("main_reactant_smiles") or last.get("final_reactant_smiles") or "")


def _last_main_reactant_name(steps: Any) -> str:
    if not isinstance(steps, list) or not steps:
        return ""
    last = steps[-1]
    if not isinstance(last, dict):
        return ""
    return str(last.get("main_reactant_name") or last.get("final_reactant_name") or "")


def _subgoal_summary(
    *,
    verifier: dict[str, Any],
    raw: dict[str, Any],
    selected_subgoal: dict[str, Any],
    subgoal_name: str,
    expected_target_smiles: str = "",
) -> dict[str, Any]:
    target_audit = dict(verifier.get("target_equivalence_audit") or {})
    selected_target = dict(selected_subgoal.get("subgoal") or {})
    target_smiles = (
        str(expected_target_smiles or "")
        or str(target_audit.get("request_target_smiles") or "")
        or str(raw.get("target") or raw.get("target_smiles") or "")
        or str(selected_target.get("smiles") or "")
    )
    parent_bridge = _parent_bridge_summary(selected_target, target_smiles=target_smiles)
    reverified = verify_chemenzy_raw_routes(
        raw,
        target_smiles=target_smiles,
    )
    verifier_accepted = _verifier_stock_audit_passed(
        reverified,
        expected_target_smiles=target_smiles,
    )
    reaction_validated = is_reaction_validated_route_verifier_report(
        reverified,
        expected_target_smiles=target_smiles,
    )
    precedent_supported = is_precedent_supported_route_verifier_report(
        reverified,
        expected_target_smiles=target_smiles,
    )
    materialization = _best_route_materialization(raw, reverified)
    provided_verifier_bound = _provided_verifier_replay_binding(
        verifier,
        reverified=reverified,
        target_smiles=target_smiles,
    )
    provided_validation = dict(verifier.get("reaction_validation") or {})
    current_validation = dict(reverified.get("reaction_validation") or {})
    return {
        "accepted": verifier_accepted,
        "verifier_accepted": verifier_accepted,
        "reaction_validated": reaction_validated,
        "precedent_supported": precedent_supported,
        "verification_level": str(reverified.get("verification_level") or "L0_materialized"),
        "reaction_validation": dict(reverified.get("reaction_validation") or {}),
        "route_status": str(reverified.get("route_status") or ""),
        "target_match": bool(reverified.get("target_match")),
        "target": _compound_identity(target_smiles),
        "parent_bridge": parent_bridge,
        "target_equivalence_audit": dict(reverified.get("target_equivalence_audit") or {}),
        "route_count": int(reverified.get("route_count") or len(raw.get("routes") or [])),
        "accepted_route_count": int(reverified.get("accepted_route_count") or 0),
        "best_route_rank": reverified.get("best_route_rank"),
        "best_route_step_count": int(reverified.get("best_route_step_count") or 0),
        "raw_best_route_step_count": materialization["raw_step_count"],
        "verifier_best_route_step_count": materialization["verifier_step_count"],
        "route_materialization_complete": materialization["complete"],
        "subgoal_name": subgoal_name or str(selected_target.get("name") or ""),
        "raw_solved": bool((raw.get("search_status") or {}).get("solved")),
        # The raw backend's ``stock_closed`` flag is not an independent audit:
        # it is exactly the claim that the harness verifier exists to check.
        # Treat a route as stock closed only when the deterministic verifier
        # accepted a target-matched, materialized route.
        "stock_audit_passed": _verifier_stock_audit_passed(
            reverified,
            expected_target_smiles=target_smiles,
        ),
        "verifier_reasons": [str(item) for item in reverified.get("reasons") or []],
        # The current host replay is the authority.  A stored verifier only
        # has to bind the same target and materialized route; its proof digest
        # may legitimately become stale when exact precedent arrives later.
        # This lets evidence-first scheduling upgrade L1/L2 to L3 without
        # rerunning the route proposal, while a swapped/tampered route remains
        # rejected by exact materialization binding.
        "provided_verifier_matched_reverification": provided_verifier_bound,
        "provided_verifier_proof_refreshed": bool(
            provided_verifier_bound
            and provided_validation.get("proof_digest")
            != current_validation.get("proof_digest")
        ),
    }


def _provided_verifier_replay_binding(
    verifier: dict[str, Any],
    *,
    reverified: dict[str, Any],
    target_smiles: str,
) -> bool:
    """Bind a historical verifier snapshot to the host-replayed raw route.

    Stored acceptance/proof flags are intentionally ignored.  Exact target,
    route rank, step count and materialized route identity must match the
    current replay, which is what grants stock/reaction authority.
    """

    if verifier.get("schema_version") != "harness_route_verifier_report.v1":
        return False
    provided_target_audit = verifier.get("target_equivalence_audit")
    if not isinstance(provided_target_audit, dict):
        return False
    provided_target = _compound_identity(
        str(
            provided_target_audit.get("request_canonical_isomeric_smiles")
            or provided_target_audit.get("request_target_smiles")
            or ""
        )
    )
    if not _same_compound(provided_target, _compound_identity(target_smiles)):
        return False
    provided_route = verifier.get("accepted_route")
    replayed_route = reverified.get("accepted_route")
    return bool(
        isinstance(provided_route, dict)
        and isinstance(replayed_route, dict)
        and provided_route == replayed_route
        and verifier.get("best_route_rank") == reverified.get("best_route_rank")
        and _safe_int(verifier.get("best_route_step_count"))
        == _safe_int(reverified.get("best_route_step_count"))
    )


def _parent_bridge_summary(selected_target: dict[str, Any], *, target_smiles: str) -> dict[str, Any]:
    policy = dict(selected_target.get("policy") or selected_target.get("chem_enzy_search_policy") or {})
    preferred = dict(policy.get("preferred_subgoal") or {})
    nested = dict(preferred.get("hypothetical_precursor_target") or {})
    parent_smiles = str(selected_target.get("parent_smiles") or nested.get("parent_smiles") or "").strip()
    operation_idea = str(
        selected_target.get("operation_idea")
        or nested.get("operation_idea")
        or preferred.get("operation_idea")
        or ""
    )
    variant_type = str(
        selected_target.get("variant_type")
        or nested.get("variant_type")
        or selected_target.get("route_objective_type")
        or nested.get("route_objective_type")
        or ""
    )
    risk_flags = [
        str(item)
        for item in [
            *(selected_target.get("risk_flags") or []),
            *(nested.get("risk_flags") or []),
        ]
        if str(item or "").strip()
    ]
    mechanistic_hint = _mechanistic_parent_bridge_transform_allowed(
        operation_idea=operation_idea,
        variant_type=variant_type,
        risk_flags=risk_flags,
    )
    validation = dict(
        selected_target.get("parent_bridge_validation")
        or nested.get("parent_bridge_validation")
        or {}
    )
    method = str(validation.get("method") or validation.get("validation_method") or "").strip().lower()
    allowed_methods = {"atom_mapped_transform", "forward_reconstruction", "validated_reaction_edge"}
    validation_parent = _compound_identity(
        str(validation.get("parent_smiles") or validation.get("product_smiles") or "")
    )
    validation_child = _compound_identity(
        str(validation.get("child_smiles") or validation.get("reactant_smiles") or "")
    )
    transform_allowed = bool(
        validation.get("accepted") is True
        and method in allowed_methods
        and _same_compound(validation_parent, _compound_identity(parent_smiles))
        and _same_compound(validation_child, _compound_identity(target_smiles))
    )
    return {
        "schema_version": "mechanistic_parent_bridge_summary.v1",
        "present": bool(parent_smiles),
        "parent": _compound_identity(parent_smiles),
        "child": _compound_identity(target_smiles),
        "operation_idea": operation_idea,
        "variant_type": variant_type,
        "risk_flags": sorted(set(risk_flags)),
        "transform_allowed": transform_allowed,
        "mechanistic_hint": mechanistic_hint,
        "transform_validation": validation,
        "bridge_basis": method if transform_allowed else "unvalidated_mechanistic_hint",
        "not_exact_literature_segment": True,
    }


def _mechanistic_parent_bridge_transform_allowed(
    *,
    operation_idea: str,
    variant_type: str,
    risk_flags: list[str],
) -> bool:
    text = " ".join([operation_idea, variant_type, *risk_flags]).lower()
    blocked = ("large_atom_jump", "fragment_reassembly", "unknown_connectivity", "scaffold_hop")
    if any(token in text for token in blocked):
        return False
    allowed = (
        "redox",
        "oxid",
        "reduct",
        "carbonyl",
        "hydroxy",
        "alcohol",
        "ketone",
        "aldehyde",
        "protect",
        "deprotect",
        "acetate",
        "ester",
        "acid",
        "anhydride",
        "same_core",
    )
    return any(token in text for token in allowed)


def _best_route_step_count(raw: dict[str, Any], verifier: dict[str, Any]) -> int:
    return int(_best_route_materialization(raw, verifier)["materialized_step_count"])


def _best_route_materialization(raw: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
    routes = [dict(item) for item in raw.get("routes") or [] if isinstance(item, dict)]
    if not routes:
        return {"materialized_step_count": 0, "raw_step_count": 0, "verifier_step_count": 0, "complete": False}
    best_rank = verifier.get("best_route_rank")
    if best_rank is None:
        return {"materialized_step_count": 0, "raw_step_count": 0, "verifier_step_count": 0, "complete": False}
    route = next((item for item in routes if item.get("route_rank") == best_rank), None)
    if route is None:
        return {"materialized_step_count": 0, "raw_step_count": 0, "verifier_step_count": 0, "complete": False}
    raw_steps = route.get("steps") or []
    raw_step_count = len(raw_steps) if isinstance(raw_steps, list) else 0
    materialized_step_count = len(
        [
            step
            for step in raw_steps
            if isinstance(step, dict) and _materialized_subgoal_step(step)
        ]
    )
    try:
        verifier_step_count = int(verifier.get("best_route_step_count") or 0)
    except (TypeError, ValueError):
        verifier_step_count = 0
    return {
        "materialized_step_count": materialized_step_count,
        "raw_step_count": raw_step_count,
        "verifier_step_count": verifier_step_count,
        "complete": bool(
            materialized_step_count > 0
            and materialized_step_count == raw_step_count
            and materialized_step_count == verifier_step_count
        ),
    }


def _materialized_subgoal_step(step: dict[str, Any]) -> bool:
    product = str(step.get("product") or step.get("product_smiles") or "").strip()
    raw_reactants = (
        step.get("reactant_smiles")
        or step.get("precursor_smiles")
        or step.get("main_reactant")
        or step.get("main_reactant_smiles")
        or []
    )
    if isinstance(raw_reactants, str):
        reactants = [raw_reactants]
    elif isinstance(raw_reactants, list):
        reactants = [str(item or "") for item in raw_reactants]
    else:
        reactants = []
    reactants = [item for item in reactants if item.strip()]
    return bool(
        _compound_identity(product).get("valid")
        and reactants
        and all(_compound_identity(item).get("valid") for item in reactants)
    )


def _verifier_stock_audit_passed(
    verifier: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> bool:
    """Return true only for a fully accepted deterministic route-verifier result."""
    explicit_stock_audit = verifier.get("stock_audit_passed")
    return bool(
        is_accepted_route_verifier_report(
            verifier,
            expected_target_smiles=expected_target_smiles,
        )
        and explicit_stock_audit is not False
    )


def _target_identity_audit(*, requested_target_smiles: str, literature_target_smiles: str) -> dict[str, Any]:
    requested = _compound_identity(requested_target_smiles)
    literature = _compound_identity(literature_target_smiles)
    required = bool(str(requested_target_smiles or "").strip())
    literature_present = bool(str(literature_target_smiles or "").strip())
    target_match = bool(required and literature_present and _same_compound(requested, literature))
    reasons: list[str] = []
    if required and not literature_present:
        reasons.append("literature_target_missing")
    elif required and not target_match:
        reasons.append("target_identity_mismatch")
    return {
        "schema_version": "stitched_route_target_identity_audit.v1",
        "required": required,
        "target_match": target_match,
        "literature_target_present": literature_present,
        "requested_target": requested,
        "literature_target": literature,
        "match_basis": "canonical_isomeric_smiles_and_inchikey",
        "reasons": reasons,
    }


def _terminal_subgoal_match_audit(
    *,
    terminal: dict[str, Any],
    subgoal: dict[str, Any],
    parent_bridge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bridge = dict(parent_bridge or {})
    bridge_parent = dict(bridge.get("parent") or {})
    direct_match = _same_compound(terminal, subgoal)
    parent_bridge_identity_match = bool(bridge.get("present") and _same_compound(terminal, bridge_parent))
    # A model-authored ``parent_bridge_validation`` summary is not a
    # reconstructable reaction proof.  Until an atom-mapped/forward validator
    # is available, only exact terminal/subgoal identity may close a stitch.
    parent_bridge_accepted = False
    accepted = bool(direct_match or parent_bridge_accepted)
    reasons: list[str] = []
    if not accepted:
        reasons.append("terminal_subgoal_identity_mismatch")
        if bridge.get("present") and not parent_bridge_identity_match:
            reasons.append("terminal_parent_bridge_identity_mismatch")
        if bridge.get("present") and not bridge.get("transform_allowed"):
            reasons.append("mechanistic_parent_bridge_transform_not_allowed")
        elif bridge.get("present") and bridge.get("transform_allowed"):
            reasons.append("mechanistic_parent_bridge_transform_not_revalidated")
    return {
        "schema_version": "terminal_subgoal_match_audit.v1",
        "accepted": accepted,
        "direct_terminal_subgoal_match": direct_match,
        "parent_bridge_identity_match": parent_bridge_identity_match,
        "parent_bridge_accepted": parent_bridge_accepted,
        "terminal": terminal,
        "subgoal_target": subgoal,
        "parent_bridge": bridge,
        "match_basis": "canonical_isomeric_smiles_and_inchikey",
        "reasons": reasons,
    }


def _mechanistic_parent_bridge_segments(terminal_match: dict[str, Any]) -> list[dict[str, Any]]:
    if not terminal_match.get("parent_bridge_accepted"):
        return []
    bridge = dict(terminal_match.get("parent_bridge") or {})
    parent = dict(bridge.get("parent") or {})
    child = dict(bridge.get("child") or {})
    return [
        {
            "segment_id": "mechanistic_parent_bridge",
            "role": "verified_subgoal_to_literature_terminal",
            "status": "mechanistic_bridge_accepted",
            "from_smiles": str(child.get("input_smiles") or ""),
            "to_smiles": str(parent.get("input_smiles") or ""),
            "operation_idea": str(bridge.get("operation_idea") or ""),
            "variant_type": str(bridge.get("variant_type") or ""),
            "step_count": 1,
            "not_exact_literature_segment": True,
        }
    ]


def _same_compound(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left.get("valid")
        and right.get("valid")
        and left.get("canonical_isomeric_smiles") == right.get("canonical_isomeric_smiles")
        and left.get("inchikey") == right.get("inchikey")
    )


def _compound_identity(smiles: str) -> dict[str, Any]:
    text = str(smiles or "").strip()
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        return {
            "valid": False,
            "input_smiles": text,
            "canonical_isomeric_smiles": "",
            "inchikey": "",
        }
    return {
        "valid": True,
        "input_smiles": text,
        "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "inchikey": _inchikey(mol),
    }


def _inchikey(mol: Chem.Mol) -> str:
    try:
        return str(Chem.MolToInchiKey(mol) or "")
    except Exception:
        return ""


def _artifact_refs(**items: Any) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key, value in items.items():
        if key == "selected_subgoal" and isinstance(value, dict):
            for nested_key in ("raw_result_path", "request_path"):
                if value.get(nested_key):
                    refs[f"subgoal_{nested_key}"] = str(value[nested_key])
            continue
        if isinstance(value, (str, Path)) and str(value).strip():
            refs[key] = str(value)
    return refs


def _failure_status(reasons: list[str]) -> str:
    reason_set = set(reasons)
    if "literature_terminal_subgoal_target_mismatch" in reason_set:
        return "terminal_mismatch"
    if (
        "subgoal_verifier_not_accepted" in reason_set
        or "subgoal_route_not_solved" in reason_set
        or "subgoal_reaction_steps_not_validated" in reason_set
    ):
        return "subgoal_not_verified"
    if "literature_chain_not_accepted" in reason_set:
        return "literature_chain_not_accepted"
    return "stitch_rejected"
