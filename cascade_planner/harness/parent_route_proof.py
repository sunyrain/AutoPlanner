"""Deterministic parent-route proof gate for agentic blackboard runs."""
from __future__ import annotations

from typing import Any


PARENT_ROUTE_PROOF_SCHEMA = "stitched_parent_route_proof.v1"


def compile_stitched_parent_route_proof(
    *,
    target_smiles: str = "",
    target_name: str = "",
    case_id: str = "",
    parent_verifier: dict[str, Any] | None = None,
    stitched_route: dict[str, Any] | None = None,
    child_route: dict[str, Any] | None = None,
    exact_literature_segment: dict[str, Any] | None = None,
    stock_audit: dict[str, Any] | None = None,
    analogy_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove that a parent route, child route, and literature segment connect.

    Analogy is allowed only as rationale. It never satisfies any proof clause.
    """
    parent = dict(parent_verifier or {})
    stitch = dict(stitched_route or {})
    child = dict(child_route or {})
    exact = dict(exact_literature_segment or {})
    stock = dict(stock_audit or {})
    reasons: list[str] = []
    stitched_accepted = _stitched_route_accepted(stitch)
    parent_for_failure_audit = {} if stitched_accepted else parent

    target_equivalence = _target_equivalence_passed({} if stitched_accepted else parent, stitch)
    if not target_equivalence:
        reasons.append("target_equivalence_not_proven")

    parent_accepted = _verifier_accepted(parent) or stitched_accepted
    if not parent_accepted:
        reasons.append("parent_route_verifier_not_accepted")

    stock_passed = bool(stock.get("stock_audit_passed")) if stock else bool(parent.get("accepted") or stitch.get("stock_audit_passed"))
    if not stock_passed:
        reasons.append("stock_audit_not_passed")

    if _has_unexplained_large_atom_jump(parent_for_failure_audit):
        reasons.append("unexplained_large_atom_jump")

    child_connected = _child_connected(child, stitch)
    if not child_connected:
        reasons.append("child_target_route_not_connected_to_parent_bridge")

    literature_connected = _literature_connected(exact, stitch)
    if not literature_connected:
        reasons.append("exact_literature_segment_not_connected_to_parent_route")

    analogy_as_proof = any(bool(row.get("used_as_proof")) for row in analogy_refs or [] if isinstance(row, dict))
    if analogy_as_proof:
        reasons.append("analogy_cannot_be_parent_route_proof")

    child_solved = _child_route_solved(child, stitch)
    exact_anchor_present = _exact_anchor_present(exact, stitch)

    accepted = not sorted(set(reasons))
    return {
        "schema_version": PARENT_ROUTE_PROOF_SCHEMA,
        "accepted": accepted,
        "solved": accepted,
        "case_id": case_id or str(parent.get("case_id") or stitch.get("case_id") or ""),
        "target": {"name": target_name, "smiles": target_smiles},
        "route_status": "solved"
        if accepted
        else _failure_status(
            reasons,
            child_solved=child_solved,
            exact_anchor_present=exact_anchor_present,
        ),
        "proof_clauses": {
            "target_equivalence_passed": target_equivalence,
            "parent_route_verifier_accepted": parent_accepted,
            "stock_audit_passed": stock_passed,
            "no_unexplained_large_atom_jump": not _has_unexplained_large_atom_jump(parent_for_failure_audit),
            "child_target_route_connected_to_parent_bridge": child_connected,
            "exact_literature_segment_connected_to_parent_route": literature_connected,
            "analogy_used_only_as_rationale": not analogy_as_proof,
        },
        "source_policy": {
            "child_target_solved_does_not_imply_parent_solved": True,
            "exact_literature_segment_requires_connectivity": True,
            "analogy_is_not_proof": True,
            "final_verdict_authority": "deterministic_parent_route_proof",
        },
        "reasons": sorted(set(reasons)),
    }


def _target_equivalence_passed(parent: dict[str, Any], stitch: dict[str, Any]) -> bool:
    audit = dict(parent.get("target_equivalence_audit") or {})
    if parent:
        return bool(parent.get("target_match") or audit.get("target_match"))
    stitch_target = dict((stitch.get("target") or {}).get("identity_audit") or {})
    return bool(stitch.get("accepted") and (not stitch_target.get("required") or stitch_target.get("target_match")))


def _verifier_accepted(parent: dict[str, Any]) -> bool:
    return bool(parent.get("accepted")) and str(parent.get("route_status") or "solved") == "solved"


def _stitched_route_accepted(stitch: dict[str, Any]) -> bool:
    if not stitch:
        return False
    if not bool(stitch.get("accepted") or stitch.get("solved")):
        return False
    status = str(stitch.get("route_status") or "solved")
    if status and status != "solved":
        return False
    return True


def _has_unexplained_large_atom_jump(parent: dict[str, Any]) -> bool:
    reasons = {str(item) for item in parent.get("reasons") or []}
    if "large_atom_jump" in reasons:
        return True
    for event in parent.get("failure_events") or []:
        if isinstance(event, dict) and str(event.get("reason") or "") == "large_atom_jump":
            return True
    return False


def _child_connected(child: dict[str, Any], stitch: dict[str, Any]) -> bool:
    if stitch:
        terminal_match = dict(stitch.get("terminal_match_audit") or {})
        subgoal = dict(stitch.get("subgoal_closure") or {})
        if stitch.get("accepted") and terminal_match.get("accepted") and subgoal.get("verifier_accepted"):
            return True
    if not child:
        return False
    return bool(child.get("parent_bridge_connected") and (child.get("accepted") or child.get("solved")))


def _child_route_solved(child: dict[str, Any], stitch: dict[str, Any]) -> bool:
    if stitch:
        subgoal = dict(stitch.get("subgoal_closure") or {})
        if bool(subgoal.get("accepted") or subgoal.get("solved") or subgoal.get("verifier_accepted")):
            return True
    if not child:
        return False
    if bool(child.get("accepted") or child.get("solved")):
        return True
    try:
        return int(child.get("accepted_subgoal_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def _literature_connected(exact: dict[str, Any], stitch: dict[str, Any]) -> bool:
    if stitch:
        literature = dict(stitch.get("literature_chain") or {})
        terminal_match = dict(stitch.get("terminal_match_audit") or {})
        if stitch.get("accepted") and literature.get("chain_accepted") and terminal_match.get("accepted"):
            return True
    if not exact:
        return False
    return bool(exact.get("accepted") and exact.get("parent_route_connected"))


def _exact_anchor_present(exact: dict[str, Any], stitch: dict[str, Any]) -> bool:
    if stitch:
        literature = dict(stitch.get("literature_chain") or {})
        if bool(literature.get("chain_accepted")):
            return True
        try:
            if int(literature.get("step_count") or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        if literature.get("source_ref") or literature.get("terminal"):
            return True
    if not exact:
        return False
    if bool(exact.get("accepted")):
        return True
    try:
        return int(exact.get("row_count") or exact.get("accepted_row_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def _failure_status(
    reasons: list[str],
    *,
    child_solved: bool = False,
    exact_anchor_present: bool = False,
) -> str:
    reason_set = set(reasons)
    if child_solved and "child_target_route_not_connected_to_parent_bridge" in reason_set:
        return "child_solved_parent_unresolved"
    if exact_anchor_present and "exact_literature_segment_not_connected_to_parent_route" in reason_set:
        return "partial_anchor_only_not_solved"
    if "unexplained_large_atom_jump" in reason_set:
        return "fake_closed_rejected"
    return "unresolved"
