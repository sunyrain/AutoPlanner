"""Deterministic parent-route proof gate for agentic blackboard runs."""
from __future__ import annotations

from typing import Any

from rdkit import Chem

from cascade_planner.harness.route_verifier import is_accepted_route_verifier_report
from cascade_planner.harness.route_verifier import (
    is_precedent_supported_route_verifier_report,
)
from cascade_planner.harness.stitched_route import is_solved_stitched_semisynthesis_route


PARENT_ROUTE_PROOF_SCHEMA = "stitched_parent_route_proof.v1"
STITCHED_ROUTE_SCHEMA = "stitched_semisynthesis_route.v1"

_REQUIRED_SOLVED_PROOF_CLAUSES = (
    "target_equivalence_passed",
    "parent_route_verifier_accepted",
    "stock_audit_passed",
    "no_unexplained_large_atom_jump",
    "child_target_route_connected_to_parent_bridge",
    "exact_literature_segment_connected_to_parent_route",
    "all_reaction_steps_validated",
    "all_reaction_steps_precedent_supported",
    "analogy_used_only_as_rationale",
)


def is_solved_parent_route_proof(
    value: Any,
    *,
    expected_target_smiles: str = "",
) -> bool:
    """Return true only for a complete deterministic parent-proof artifact.

    ``accepted``/``solved`` are conclusions, not authority tokens.  Consumers
    must also verify the proof schema and every deterministic proof clause so a
    model-authored or partially deserialized pair of booleans cannot mark a
    case solved.
    """
    if not isinstance(value, dict):
        return False
    proof = dict(value)
    if not isinstance(proof.get("proof_clauses"), dict):
        return False
    if not isinstance(proof.get("source_policy"), dict):
        return False
    if not isinstance(proof.get("reasons"), list):
        return False
    clauses = dict(proof["proof_clauses"])
    source_policy = dict(proof["source_policy"])
    target = dict(proof.get("target") or {})
    proof_target_smiles = str(target.get("smiles") or "").strip()
    if proof.get("schema_version") != PARENT_ROUTE_PROOF_SCHEMA:
        return False
    if proof.get("accepted") is not True or proof.get("solved") is not True:
        return False
    if not proof_target_smiles or not _same_target_smiles(proof_target_smiles, proof_target_smiles):
        return False
    if str(expected_target_smiles or "").strip() and not _same_target_smiles(
        proof_target_smiles,
        expected_target_smiles,
    ):
        return False
    if str(proof.get("route_status") or "").strip().lower() != "solved":
        return False
    if proof["reasons"]:
        return False
    proof_attempt = proof.get("proof_attempt")
    if not isinstance(proof_attempt, dict):
        return False
    if proof_attempt.get("schema_version") != "parent_route_proof_attempt.v1":
        return False
    if proof_attempt.get("accepted") is not True or proof_attempt.get("missing_requirements") != []:
        return False
    if str(proof.get("proof_mode") or "") not in {"direct_parent_route", "stitched_parent_route"}:
        return False
    proof_evidence = proof.get("proof_evidence")
    if not isinstance(proof_evidence, dict):
        return False
    if any(clauses.get(key) is not True for key in _REQUIRED_SOLVED_PROOF_CLAUSES):
        return False
    if proof.get("proof_mode") == "direct_parent_route" and clauses.get(
        "direct_parent_route_verifier_accepted"
    ) is not True:
        return False
    if proof.get("proof_mode") == "direct_parent_route":
        verifier = proof_evidence.get("parent_verifier")
        if not is_precedent_supported_route_verifier_report(
            verifier,
            expected_target_smiles=proof_target_smiles,
        ):
            return False
    else:
        stitched = proof_evidence.get("stitched_route")
        if not isinstance(stitched, dict) or not _stitched_route_accepted(
            stitched,
            target_required=True,
            expected_target_smiles=proof_target_smiles,
        ):
            return False
    return source_policy.get("final_verdict_authority") == "deterministic_parent_route_proof"


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
    stitched_accepted = _stitched_route_accepted(
        stitch,
        target_required=bool(str(target_smiles or "").strip()),
        expected_target_smiles=target_smiles,
    )
    parent_graph_stock_closed = _verifier_accepted(
        parent,
        expected_target_smiles=target_smiles,
    )
    parent_reaction_validated = _reaction_verifier_accepted(
        parent,
        expected_target_smiles=target_smiles,
    )
    direct_parent_candidate = (
        parent_graph_stock_closed
        and not stitched_accepted
        and not _child_route_input_present(child)
        and not _exact_literature_input_present(exact)
    )
    direct_parent_route = bool(direct_parent_candidate and parent_reaction_validated)
    parent_for_failure_audit = {} if stitched_accepted else parent

    target_equivalence = _target_equivalence_passed(
        {} if stitched_accepted else parent,
        stitch,
        expected_target_smiles=target_smiles,
    )
    if not target_equivalence:
        reasons.append("target_equivalence_not_proven")

    parent_accepted = parent_reaction_validated or stitched_accepted
    if not parent_accepted:
        if parent_graph_stock_closed:
            reasons.append("parent_route_reaction_steps_not_validated")
        else:
            reasons.append("parent_route_verifier_not_accepted")

    if stock:
        stock_passed = stock.get("stock_audit_passed") is True
    elif direct_parent_candidate:
        # A strict route-verifier acceptance already includes the stock audit.
        stock_passed = parent_graph_stock_closed
    else:
        stock_passed = bool(stitched_accepted and stitch.get("stock_audit_passed") is True)
    if not stock_passed:
        reasons.append("stock_audit_not_passed")

    if _has_unexplained_large_atom_jump(parent_for_failure_audit):
        reasons.append("unexplained_large_atom_jump")

    child_connected = True if direct_parent_candidate else _child_connected(child, stitch)
    if not child_connected:
        reasons.append("child_target_route_not_connected_to_parent_bridge")

    literature_connected = True if direct_parent_candidate else _literature_connected(exact, stitch)
    if not literature_connected:
        reasons.append("exact_literature_segment_not_connected_to_parent_route")

    analogy_as_proof = any(bool(row.get("used_as_proof")) for row in analogy_refs or [] if isinstance(row, dict))
    if analogy_as_proof:
        reasons.append("analogy_cannot_be_parent_route_proof")

    child_solved = _child_route_solved(child, stitch)
    exact_anchor_present = _exact_anchor_present(exact, stitch)
    all_reaction_steps_validated = bool(parent_reaction_validated or stitched_accepted)
    if not all_reaction_steps_validated:
        reasons.append("reaction_step_proof_incomplete")

    accepted = not sorted(set(reasons))
    missing_requirements = sorted(set(reasons))
    proof_attempt = {
        "schema_version": "parent_route_proof_attempt.v1",
        "accepted": accepted,
        "target_smiles": target_smiles,
        "graph_and_stock_closed": parent_graph_stock_closed,
        "reaction_steps_validated": all_reaction_steps_validated,
        "reaction_validation": dict(parent.get("reaction_validation") or {}),
        "missing_requirements": missing_requirements,
        "open_frontiers": _open_frontiers(child, exact, stitch),
    }
    return {
        "schema_version": PARENT_ROUTE_PROOF_SCHEMA,
        "accepted": accepted,
        "solved": accepted,
        "proof_mode": "direct_parent_route" if direct_parent_candidate else "stitched_parent_route",
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
            "direct_parent_route_verifier_accepted": direct_parent_route,
            "stock_audit_passed": stock_passed,
            "no_unexplained_large_atom_jump": not _has_unexplained_large_atom_jump(parent_for_failure_audit),
            "child_target_route_connected_to_parent_bridge": child_connected,
            "exact_literature_segment_connected_to_parent_route": literature_connected,
            "all_reaction_steps_validated": all_reaction_steps_validated,
            "all_reaction_steps_precedent_supported": all_reaction_steps_validated,
            "analogy_used_only_as_rationale": not analogy_as_proof,
        },
        "proof_evidence": {
            "schema_version": "parent_route_proof_evidence.v1",
            "parent_verifier": parent if direct_parent_route else {},
            "parent_verifier_attempt": parent,
            "stitched_route": stitch if stitched_accepted else {},
        },
        "proof_attempt": proof_attempt,
        "source_policy": {
            "direct_verified_parent_route_satisfies_parent_proof": True,
            "child_target_solved_does_not_imply_parent_solved": True,
            "exact_literature_segment_requires_connectivity": True,
            "parent_solved_temporarily_requires_every_step_l3_precedent": True,
            "analogy_is_not_proof": True,
            "final_verdict_authority": "deterministic_parent_route_proof",
        },
        "reasons": sorted(set(reasons)),
    }


def _target_equivalence_passed(
    parent: dict[str, Any],
    stitch: dict[str, Any],
    *,
    expected_target_smiles: str,
) -> bool:
    if parent:
        return _verifier_accepted(parent, expected_target_smiles=expected_target_smiles)
    stitch_target = dict((stitch.get("target") or {}).get("identity_audit") or {})
    return bool(stitch.get("accepted") and (not stitch_target.get("required") or stitch_target.get("target_match")))


def _verifier_accepted(
    parent: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> bool:
    return is_accepted_route_verifier_report(
        parent,
        expected_target_smiles=expected_target_smiles,
    )


def _reaction_verifier_accepted(
    parent: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> bool:
    return is_precedent_supported_route_verifier_report(
        parent,
        expected_target_smiles=expected_target_smiles,
    )


def _open_frontiers(
    child: dict[str, Any],
    exact: dict[str, Any],
    stitch: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compile machine-readable unresolved frontiers for every proof attempt."""
    rows: list[dict[str, Any]] = []
    for raw in stitch.get("unresolved_frontiers") or stitch.get("open_frontiers") or []:
        if isinstance(raw, dict):
            rows.append(dict(raw))
    for raw in child.get("subgoals") or child.get("frontiers") or []:
        if not isinstance(raw, dict):
            continue
        accepted = bool(raw.get("accepted") or raw.get("solved") or raw.get("verifier_accepted"))
        if not accepted:
            rows.append({"source": "child_route", **dict(raw)})
    if exact and not bool(exact.get("accepted") and exact.get("parent_route_connected")):
        rows.append(
            {
                "source": "exact_literature_segment",
                "reason": "exact_literature_segment_not_connected",
                "source_ref": str(exact.get("source_ref") or ""),
            }
        )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = repr(sorted(row.items(), key=lambda item: str(item[0])))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _stitched_route_accepted(
    stitch: dict[str, Any],
    *,
    target_required: bool,
    expected_target_smiles: str = "",
) -> bool:
    del target_required  # The recomputation always enforces a target identity.
    return is_solved_stitched_semisynthesis_route(
        stitch,
        expected_target_smiles=expected_target_smiles,
    )


def _same_target_smiles(left: str, right: str) -> bool:
    try:
        left_mol = Chem.MolFromSmiles(str(left or ""))
        right_mol = Chem.MolFromSmiles(str(right or ""))
    except Exception:
        return False
    if left_mol is None or right_mol is None:
        return False
    return Chem.MolToSmiles(left_mol, isomericSmiles=True) == Chem.MolToSmiles(
        right_mol,
        isomericSmiles=True,
    )


def _has_unexplained_large_atom_jump(parent: dict[str, Any]) -> bool:
    if _accepted_route_count(parent) > 0 and _verifier_accepted(parent):
        return False
    reasons = {str(item) for item in parent.get("reasons") or []}
    if "large_atom_jump" in reasons:
        return True
    for event in parent.get("failure_events") or []:
        if isinstance(event, dict) and str(event.get("reason") or "") == "large_atom_jump":
            return True
    return False


def _accepted_route_count(parent: dict[str, Any]) -> int:
    try:
        return int(parent.get("accepted_route_count") or 0)
    except (TypeError, ValueError):
        return 0


def _child_connected(child: dict[str, Any], stitch: dict[str, Any]) -> bool:
    if stitch:
        terminal_match = dict(stitch.get("terminal_match_audit") or {})
        subgoal = dict(stitch.get("subgoal_closure") or {})
        if stitch.get("accepted") and terminal_match.get("accepted") and subgoal.get("verifier_accepted"):
            return True
    if not child:
        return False
    return bool(child.get("parent_bridge_connected") and (child.get("accepted") or child.get("solved")))


def _child_route_input_present(child: dict[str, Any]) -> bool:
    if not child:
        return False
    if any(bool(child.get(key)) for key in ("accepted", "solved", "parent_bridge_connected")):
        return True
    try:
        if int(child.get("accepted_subgoal_count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return bool(child.get("subgoals") or child.get("routes") or child.get("reasons"))


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


def _exact_literature_input_present(exact: dict[str, Any]) -> bool:
    if not exact:
        return False
    if any(bool(exact.get(key)) for key in ("accepted", "parent_route_connected", "row_id", "source_ref")):
        return True
    try:
        return int(exact.get("row_count") or exact.get("accepted_row_count") or 0) > 0
    except (TypeError, ValueError):
        return False


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
