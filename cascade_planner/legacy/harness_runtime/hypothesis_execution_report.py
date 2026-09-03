"""Audit execution results for hypothesis-only retrosynthesis candidates."""
from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def compile_hypothesis_execution_report(
    *,
    blackboard: dict[str, Any],
    hypothesis_report: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    route_expansion_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map each hypothesis precursor to an actual route-expansion result."""
    recursive_tasks = [
        dict(row)
        for row in blackboard.get("recursive_hypothesis_tasks") or []
        if isinstance(row, dict)
    ]
    pending_recursive = [
        row
        for row in recursive_tasks
        if str(row.get("status") or "pending") in {"", "pending"}
    ]
    payload = dict(hypothesis_report.get("payload") or hypothesis_report or {})
    candidates = [dict(row) for row in payload.get("candidate_precursors") or [] if isinstance(row, dict)]
    subgoal_rows = _route_expansion_subgoal_rows(
        artifacts=dict(artifacts or {}),
        route_expansion_results=list(route_expansion_results or []),
    )
    indexed: dict[str, list[dict[str, Any]]] = {}
    for row in subgoal_rows:
        key = _canonical_smiles(str(row.get("target_smiles") or ""), isomeric=False)
        if key:
            indexed.setdefault(key, []).append(row)

    executions = [
        _candidate_execution_row(candidate, indexed.get(_candidate_key(candidate), []))
        for candidate in candidates
    ]
    executed = [row for row in executions if row.get("execution_status") != "not_executed"]
    verified = [row for row in executions if row.get("verifier_accepted")]
    rejected = [row for row in executions if row.get("execution_status") == "executed_rejected"]
    pending = [row for row in executions if row.get("execution_status") == "not_executed"]
    status = _overall_status(
        candidate_count=len(candidates),
        executed_count=len(executed),
        verified_count=len(verified),
        rejected_count=len(rejected),
        pending_count=len(pending),
        pending_recursive_count=len(pending_recursive),
    )
    return {
        "schema_version": "hypothesis_execution_report.v1",
        "accepted": bool(candidates),
        "route_status": status,
        "solved": False,
        "no_parent_solved_claim": True,
        "hypotheses_must_be_executed": True,
        "candidate_count": len(candidates),
        "executed_candidate_count": len(executed),
        "verified_child_route_count": len(verified),
        "rejected_candidate_count": len(rejected),
        "pending_candidate_count": len(pending),
        "recursive_followup_task_count": len(recursive_tasks),
        "pending_recursive_followup_count": len(pending_recursive),
        "candidate_executions": executions,
        "summary_reasons": _summary_reasons(
            status,
            rejected_count=len(rejected),
            pending_count=len(pending),
            pending_recursive_count=len(pending_recursive),
        ),
        "required_next_steps": _required_next_steps(status),
    }


def _candidate_execution_row(candidate: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    match = _best_match(matches)
    if not match:
        return {
            "schema_version": "hypothesis_candidate_execution.v1",
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "precursor_role": str(candidate.get("precursor_role") or ""),
            "precursor_smiles": str(candidate.get("precursor_smiles") or ""),
            "execution_status": "not_executed",
            "verifier_accepted": False,
            "solved": False,
            "route_status": "not_executed",
            "reasons": ["candidate_not_found_in_route_expansion_results"],
            "no_parent_solved_claim": True,
        }
    if str(match.get("execution_kind") or "") == "codex_frontier_expansion":
        proof_closed = bool(match.get("proof_closed"))
        return {
            "schema_version": "hypothesis_candidate_execution.v1",
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "precursor_role": str(candidate.get("precursor_role") or match.get("target_name") or ""),
            "precursor_smiles": str(candidate.get("precursor_smiles") or match.get("target_smiles") or ""),
            "execution_status": (
                "executed_verified_child_route"
                if proof_closed
                else "executed_advisory_frontier_expansion"
            ),
            "execution_kind": "codex_frontier_expansion",
            "agent_task_completed": True,
            "frontier_proof_closed": proof_closed,
            "verifier_accepted": proof_closed,
            "solved": False,
            "route_status": (
                "child_route_verified_parent_unresolved"
                if proof_closed
                else "frontier_expanded_pending_reaction_and_stock_proof"
            ),
            "route_count": int(match.get("route_count") or 0),
            "accepted_route_count": int(bool(proof_closed)),
            "rejected_route_count": 0,
            "reasons": (
                []
                if proof_closed
                else [
                    "codex_frontier_was_expanded_but_is_not_proof_closed",
                    *[str(value) for value in match.get("reasons") or []],
                ]
            ),
            "frontier_job_id": str(match.get("frontier_job_id") or ""),
            "team_report_ref": str(match.get("team_report_ref") or ""),
            "no_parent_solved_claim": True,
        }
    verifier = dict(match.get("verifier") or {})
    accepted = bool(match.get("accepted") or verifier.get("accepted"))
    solved = bool(match.get("solved"))
    route_status = str(match.get("route_status") or verifier.get("route_status") or "")
    reasons = _dedupe_strings(
        [
            *[str(item) for item in match.get("reasons") or []],
            *[str(item) for item in verifier.get("reasons") or []],
        ]
    )
    return {
        "schema_version": "hypothesis_candidate_execution.v1",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "precursor_role": str(candidate.get("precursor_role") or match.get("target_name") or ""),
        "precursor_smiles": str(candidate.get("precursor_smiles") or match.get("target_smiles") or ""),
        "execution_status": "executed_verified_child_route" if accepted else "executed_rejected",
        "verifier_accepted": accepted,
        "solved": solved,
        "route_status": route_status or ("child_route_verified_parent_unresolved" if accepted else "fake_closed_rejected"),
        "route_count": int(match.get("route_count") or verifier.get("route_count") or 0),
        "accepted_route_count": int(verifier.get("accepted_route_count") or 0),
        "rejected_route_count": int(verifier.get("rejected_route_count") or 0),
        "reasons": reasons or ([] if accepted else ["route_expansion_verifier_rejected"]),
        "raw_backend_solved": bool(match.get("raw_solved")),
        "raw_backend_solved_not_proof": bool(match.get("raw_solved")) and not accepted,
        "request_path": str(match.get("request_path") or ""),
        "raw_result_path": str(match.get("raw_result_path") or ""),
        "no_parent_solved_claim": True,
    }


def _route_expansion_subgoal_rows(
    *,
    artifacts: dict[str, Any],
    route_expansion_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = list(route_expansion_results)
    direct = artifacts.get("route_expansion_subgoal_search")
    if isinstance(direct, dict):
        results.append(dict(direct))
    rows: list[dict[str, Any]] = []
    for result in results:
        payload = dict(result.get("result") or result)
        for subgoal_result in payload.get("subgoals") or []:
            if not isinstance(subgoal_result, dict):
                continue
            subgoal = dict(subgoal_result.get("subgoal") or {})
            row = dict(subgoal_result)
            row["target_smiles"] = str(
                subgoal.get("smiles")
                or row.get("target_smiles")
                or subgoal.get("target_smiles")
                or ""
            )
            row["target_name"] = str(subgoal.get("name") or row.get("target_name") or "")
            rows.append(row)
    rows.extend(_codex_frontier_execution_rows(artifacts))
    return rows


def _codex_frontier_execution_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    team = artifacts.get("codex_retrosynthesis_team")
    if not isinstance(team, dict):
        return []
    campaign = dict(team.get("campaign") or {})
    queue = dict(campaign.get("frontier_queue") or {})
    jobs = {
        str(row.get("job_id") or ""): dict(row)
        for row in queue.get("jobs") or []
        if isinstance(row, dict) and str(row.get("job_id") or "")
    }
    rows: list[dict[str, Any]] = []
    for raw in campaign.get("runs") or []:
        if not isinstance(raw, dict) or raw.get("accepted") is not True:
            continue
        run = dict(raw)
        job_id = str(run.get("frontier_job_id") or "")
        job = jobs.get(job_id, {})
        try:
            achieved = int(job.get("achieved_proof_level") or 0)
            required = int(job.get("required_proof_level") or 2)
        except (TypeError, ValueError):
            achieved, required = 0, 2
        proof_closed = bool(
            job
            and achieved >= required
            and str(job.get("closure_kind") or "")
            in {"stock_boundary", "reaction_validated", "verified_route"}
        )
        rows.append(
            {
                "execution_kind": "codex_frontier_expansion",
                "target_smiles": str(run.get("target_smiles") or job.get("frontier_smiles") or ""),
                "target_name": str((job.get("metadata") or {}).get("target_name") or ""),
                "frontier_job_id": job_id,
                "team_report_ref": str(run.get("team_report_ref") or ""),
                "agent_task_completed": True,
                "proof_closed": proof_closed,
                "route_count": 1,
                "reasons": [] if proof_closed else ["frontier_agent_task_completed_without_proof_closure"],
            }
        )
    return rows


def _best_match(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {}
    return sorted(
        matches,
        key=lambda row: (
            0 if (row.get("accepted") or (row.get("verifier") or {}).get("accepted")) else 1,
            -int(row.get("route_count") or 0),
        ),
    )[0]


def _candidate_key(candidate: dict[str, Any]) -> str:
    return _canonical_smiles(str(candidate.get("precursor_smiles") or ""), isomeric=False)


def _overall_status(
    *,
    candidate_count: int,
    executed_count: int,
    verified_count: int,
    rejected_count: int,
    pending_count: int,
    pending_recursive_count: int,
) -> str:
    if candidate_count <= 0:
        return "no_hypothesis_candidates"
    if verified_count > 0:
        return "hypothesis_child_route_verified_parent_unresolved"
    if executed_count > 0 and pending_count == 0 and rejected_count == candidate_count and pending_recursive_count > 0:
        return "hypothesis_routes_executed_rejected_recursive_followup_pending"
    if executed_count > 0 and pending_count == 0 and rejected_count == candidate_count:
        return "hypothesis_routes_executed_rejected"
    if executed_count > 0:
        return "hypothesis_route_execution_partial"
    return "hypothesis_routes_pending_execution"


def _summary_reasons(
    status: str,
    *,
    rejected_count: int,
    pending_count: int,
    pending_recursive_count: int,
) -> list[str]:
    if status == "hypothesis_routes_executed_rejected_recursive_followup_pending":
        return [
            "first_generation_hypothesis_precursors_rejected_by_verifier",
            "recursive_followup_hypothesis_tasks_pending",
        ]
    if status == "hypothesis_routes_executed_rejected":
        return ["all_hypothesis_precursors_executed_rejected_by_verifier"]
    if status == "hypothesis_child_route_verified_parent_unresolved":
        return ["at_least_one_hypothesis_child_route_verified_parent_still_requires_bridge"]
    if status == "hypothesis_route_execution_partial":
        reasons = ["hypothesis_execution_incomplete"]
        if rejected_count:
            reasons.append("some_hypothesis_precursors_rejected_by_verifier")
        if pending_count:
            reasons.append("some_hypothesis_precursors_not_executed")
        if pending_recursive_count:
            reasons.append("recursive_followup_hypothesis_tasks_pending")
        return reasons
    if status == "hypothesis_routes_pending_execution":
        return ["hypothesis_precursors_not_yet_executed"]
    return ["no_hypothesis_candidates"]


def _required_next_steps(status: str) -> list[str]:
    if status == "hypothesis_routes_executed_rejected_recursive_followup_pending":
        return [
            "execute_recursive_hypothesis_followup_tasks",
            "stop_only_after_recursive_budget_or_depth_exhausted",
        ]
    if status == "hypothesis_routes_executed_rejected":
        return [
            "change_hypothesis_family_or_add_manual_advanced_precursor",
            "search_for_more_target_proximal_same_core_intermediates",
            "avoid_repeating_same_chemenzy_subgoal_without_new_signal",
        ]
    if status == "hypothesis_route_execution_partial":
        return ["execute_remaining_hypothesis_precursors", "rank_surviving_failures_by_chemical_plausibility"]
    if status == "hypothesis_child_route_verified_parent_unresolved":
        return ["stitch_verified_child_route_to_parent_transformation", "audit_parent_bridge_selectivity"]
    return ["run_route_expansion_for_hypothesis_precursors"]


def _canonical_smiles(smiles: str, *, isomeric: bool) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    if not isomeric:
        mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def _dedupe_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
