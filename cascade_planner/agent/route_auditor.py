"""Final route-status audit for agent route packages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cascade_planner.agent.condition_agent import audit_conditions
from cascade_planner.agent.case_trace import RouteStatus


ROUTE_AUDIT_REPORT_SCHEMA = "route_audit_report.v1"


@dataclass
class RouteAuditReport:
    case_id: str
    route_status: str
    target_match: bool = False
    step_structural_audit: str = "unknown"
    stock_audit_passed: bool = False
    route_mode: str = "unknown"
    enzyme_step_status: str = "unknown"
    evidence_status: str = "unknown"
    condition_status: str = "unknown"
    fake_closure_rejected: bool = False
    unresolved_core: bool = False
    top_route_summary: dict[str, Any] = field(default_factory=dict)
    rejected_route_summary: list[dict[str, Any]] = field(default_factory=list)
    rejected_terminal_list: list[dict[str, Any]] = field(default_factory=list)
    failure_events: list[dict[str, Any]] = field(default_factory=list)
    route_mode_explanation: str = ""
    next_action: str = "review"
    reasons: list[str] = field(default_factory=list)
    schema_version: str = ROUTE_AUDIT_REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_route_package(
    package: dict[str, Any],
    *,
    validation: dict[str, Any] | None = None,
    stock_audit_passed: bool = False,
    target_match: bool = True,
    condition_candidates: list[dict[str, Any]] | None = None,
    enzyme_actions: list[dict[str, Any]] | None = None,
) -> RouteAuditReport:
    """Produce a conservative final RouteStatus for a package or route result."""
    validation = dict(validation or {})
    package = dict(package or {})
    condition_audit = audit_conditions(condition_candidates or [])
    route_status, reasons = _status_and_reasons(
        package,
        validation,
        stock_audit_passed=stock_audit_passed,
        target_match=target_match,
        condition_audit=condition_audit,
    )
    report = RouteAuditReport(
        case_id=str(package.get("case_id") or validation.get("case_id") or "case"),
        route_status=route_status.value,
        target_match=target_match,
        step_structural_audit="passed" if bool(validation.get("accepted", True)) else "failed",
        stock_audit_passed=bool(stock_audit_passed),
        route_mode=_route_mode(package, route_status),
        enzyme_step_status=_enzyme_status(enzyme_actions or []),
        evidence_status=_evidence_status(package, validation),
        condition_status=_condition_status(condition_audit),
        fake_closure_rejected=route_status == RouteStatus.FAKE_CLOSED_REJECTED,
        unresolved_core=_has_unresolved_core(package),
        top_route_summary=_top_route_summary(package),
        rejected_route_summary=_rejected_route_summary(package, validation),
        rejected_terminal_list=_rejected_terminals(package, validation),
        failure_events=_failure_events(package, validation, route_status, reasons),
        route_mode_explanation=_route_mode_explanation(route_status, stock_audit_passed, condition_audit),
        next_action=_next_action(route_status, condition_audit),
        reasons=reasons,
    )
    audit_validation = validate_route_audit_report(report)
    if not audit_validation["accepted"]:
        report.reasons = sorted(set([*report.reasons, *audit_validation["reasons"]]))
    return report


def validate_route_audit_report(report_or_data: RouteAuditReport | dict[str, Any]) -> dict[str, Any]:
    report = report_or_data if isinstance(report_or_data, RouteAuditReport) else route_audit_report_from_dict(report_or_data)
    reasons: list[str] = []
    allowed = {status.value for status in RouteStatus}
    if report.schema_version != ROUTE_AUDIT_REPORT_SCHEMA:
        reasons.append("invalid_route_audit_report_schema")
    if report.route_status not in allowed:
        reasons.append("invalid_route_status")
    if report.route_status == RouteStatus.SOLVED.value and not report.stock_audit_passed:
        reasons.append("solved_without_stock_audit")
    if report.route_status == RouteStatus.SEMISYNTHESIS_CLOSED.value and report.evidence_status != "anchor_evidence_present":
        reasons.append("semisynthesis_closed_without_anchor_evidence")
    if report.route_status == RouteStatus.FAKE_CLOSED_REJECTED.value and not (report.rejected_terminal_list or report.failure_events):
        reasons.append("fake_closed_rejected_without_rejection")
    if report.route_status == RouteStatus.UNRESOLVED.value and not report.reasons:
        reasons.append("unresolved_without_reason")
    return {
        "schema_version": "route_audit_report_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "route_status": report.route_status,
    }


def route_audit_report_from_dict(data: dict[str, Any]) -> RouteAuditReport:
    return RouteAuditReport(
        case_id=str(data.get("case_id") or ""),
        route_status=str(data.get("route_status") or ""),
        target_match=bool(data.get("target_match")),
        step_structural_audit=str(data.get("step_structural_audit") or "unknown"),
        stock_audit_passed=bool(data.get("stock_audit_passed")),
        route_mode=str(data.get("route_mode") or "unknown"),
        enzyme_step_status=str(data.get("enzyme_step_status") or "unknown"),
        evidence_status=str(data.get("evidence_status") or "unknown"),
        condition_status=str(data.get("condition_status") or "unknown"),
        fake_closure_rejected=bool(data.get("fake_closure_rejected")),
        unresolved_core=bool(data.get("unresolved_core")),
        top_route_summary=dict(data.get("top_route_summary") or {}),
        rejected_route_summary=[dict(item) for item in data.get("rejected_route_summary") or []],
        rejected_terminal_list=[dict(item) for item in data.get("rejected_terminal_list") or []],
        failure_events=[dict(item) for item in data.get("failure_events") or []],
        route_mode_explanation=str(data.get("route_mode_explanation") or ""),
        next_action=str(data.get("next_action") or "review"),
        reasons=[str(reason) for reason in data.get("reasons") or []],
        schema_version=str(data.get("schema_version") or ROUTE_AUDIT_REPORT_SCHEMA),
    )


def _status_and_reasons(
    package: dict[str, Any],
    validation: dict[str, Any],
    *,
    stock_audit_passed: bool,
    target_match: bool,
    condition_audit: dict[str, Any],
) -> tuple[RouteStatus, list[str]]:
    reasons: list[str] = []
    package_status = str(validation.get("route_status") or package.get("route_status") or "")
    if not target_match:
        return RouteStatus.UNRESOLVED, ["target_mismatch"]
    if validation and not validation.get("accepted", False):
        reasons.extend(str(reason) for reason in validation.get("reasons") or ["invalid_package"])
        return RouteStatus.FAKE_CLOSED_REJECTED, sorted(set(reasons))
    if package_status == "solved" and not stock_audit_passed:
        return RouteStatus.UNRESOLVED, ["solved_claim_without_stock_audit"]
    if stock_audit_passed and condition_audit.get("route_risk") == "high":
        return RouteStatus.UNRESOLVED, ["condition_high_risk"]
    if _has_anchor_evidence(package):
        if package_status in {"semisynthesis_closed", "ready_for_guided_rerun"} and stock_audit_passed:
            return RouteStatus.SEMISYNTHESIS_CLOSED, []
        return RouteStatus.PARTIAL_ANCHOR, ["anchor_evidence_without_full_stock_closure"]
    if stock_audit_passed:
        return RouteStatus.SOLVED, []
    if _has_unresolved_core(package):
        return RouteStatus.UNRESOLVED, ["unresolved_core"]
    return RouteStatus.UNRESOLVED, [package_status or "no_route_solution"]


def _route_mode(package: dict[str, Any], status: RouteStatus) -> str:
    if status == RouteStatus.SOLVED:
        return "stock_closed"
    if _has_anchor_evidence(package):
        return "semisynthesis_or_literature_anchor"
    if _has_unresolved_core(package):
        return "unresolved_core"
    return "unknown"


def _enzyme_status(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "unknown"
    if any(str(action.get("validation_status")) == "rejected" for action in actions):
        return "rejected"
    if any(str(action.get("validation_status")) == "validated" for action in actions):
        return "validated"
    if any(action.get("ec") or action.get("ec_number") for action in actions):
        return "generic_ec_only"
    return "unknown"


def _evidence_status(package: dict[str, Any], validation: dict[str, Any]) -> str:
    if _has_anchor_evidence(package):
        return "anchor_evidence_present"
    if package.get("literature_evidence_refs"):
        return "evidence_present"
    if validation.get("route_status") == "literature_gap":
        return "literature_gap"
    return "unknown"


def _condition_status(condition_audit: dict[str, Any]) -> str:
    if condition_audit.get("condition_gap"):
        return "condition_gap"
    if condition_audit.get("route_risk") == "high":
        return "condition_high_risk"
    return "condition_ok"


def _has_anchor_evidence(package: dict[str, Any]) -> bool:
    return any(
        str(candidate.get("candidate_kind") or "") == "route_anchor"
        for candidate in package.get("literature_candidates") or []
    )


def _has_unresolved_core(package: dict[str, Any]) -> bool:
    frontier = package.get("frontier") or {}
    flags = {str(flag) for flag in frontier.get("flags") or []}
    return bool(flags.intersection({"unresolved_core", "advanced_same_scaffold", "no_complexity_drop"}))


def _top_route_summary(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": package.get("case_id"),
        "target": package.get("target"),
        "route_status": package.get("route_status"),
        "candidate_count": len(package.get("literature_candidates") or []),
    }


def _rejected_route_summary(package: dict[str, Any], validation: dict[str, Any]) -> list[dict[str, Any]]:
    if validation.get("accepted", True):
        return []
    return [{
        "case_id": package.get("case_id"),
        "route_status": validation.get("route_status") or package.get("route_status"),
        "reasons": list(validation.get("reasons") or []),
    }]


def _rejected_terminals(package: dict[str, Any], validation: dict[str, Any]) -> list[dict[str, Any]]:
    frontier = package.get("frontier") or {}
    flags = {str(flag) for flag in frontier.get("flags") or []}
    if not flags.intersection({"advanced_same_scaffold", "no_complexity_drop", "ordinary_decoration_only"}):
        return []
    return [{
        "smiles": frontier.get("frontier_smiles"),
        "reason": "fake_closure_risk",
        "flags": sorted(flags),
        "validation_reasons": list(validation.get("reasons") or []),
    }]


def _failure_events(
    package: dict[str, Any],
    validation: dict[str, Any],
    route_status: RouteStatus,
    reasons: list[str],
) -> list[dict[str, Any]]:
    if route_status in {RouteStatus.SOLVED, RouteStatus.SEMISYNTHESIS_CLOSED}:
        return []
    return [
        {
            "schema_version": "route_audit_failure_event.v1",
            "case_id": package.get("case_id") or validation.get("case_id"),
            "reason": reason,
            "severity": "high" if route_status == RouteStatus.FAKE_CLOSED_REJECTED else "medium",
        }
        for reason in reasons
    ]


def _route_mode_explanation(
    status: RouteStatus,
    stock_audit_passed: bool,
    condition_audit: dict[str, Any],
) -> str:
    if status == RouteStatus.SOLVED:
        if condition_audit.get("condition_gap"):
            return "stock audit passed; conditions are pending and can be attached later"
        return "stock audit passed and no blocking condition risk was detected"
    if status == RouteStatus.PARTIAL_ANCHOR:
        return "literature anchor is present, but full stock closure/audit proof is absent"
    if status == RouteStatus.SEMISYNTHESIS_CLOSED:
        return "anchor evidence and stock audit support semisynthesis closure"
    if status == RouteStatus.FAKE_CLOSED_REJECTED:
        return "route/package was rejected by validation or fake-closure audit"
    if not stock_audit_passed:
        return "stock audit proof is absent"
    if condition_audit.get("condition_gap"):
        return "condition gap prevents high-confidence solved claim"
    return "route remains unresolved"


def _next_action(status: RouteStatus, condition_audit: dict[str, Any]) -> str:
    if status == RouteStatus.SOLVED:
        if condition_audit.get("condition_gap"):
            return "attach_or_retrieve_conditions"
        return "export_final_report"
    if status == RouteStatus.PARTIAL_ANCHOR:
        return "compile_guided_rerun_or_chemist_review"
    if status == RouteStatus.FAKE_CLOSED_REJECTED:
        return "diagnose_failure_and_blacklist_terminal"
    if condition_audit.get("condition_gap"):
        return "design_or_retrieve_conditions"
    return "research_or_extend_frontier"
