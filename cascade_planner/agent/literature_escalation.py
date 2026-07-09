"""Deterministic policy for escalating a case into literature mode."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LITERATURE_ESCALATION_POLICY_SCHEMA = "literature_escalation_policy.v1"
LITERATURE_ESCALATION_DECISION_SCHEMA = "literature_escalation_decision.v1"

TRIGGER_REASONS = {
    "native_failed",
    "unclosed_route",
    "fake_closure_risk",
    "advanced_frontier_detected",
    "route_audit_failed",
    "user_requested_literature",
}


@dataclass
class LiteratureEscalationPolicy:
    complex_case_token_budget: int = 80_000
    high_complexity_natural_product_token_budget: int = 150_000
    full_text_si_token_budget: int = 400_000
    schema_version: str = LITERATURE_ESCALATION_POLICY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureEscalationDecision:
    should_enter_literature_mode: bool
    escalation_reason: list[str] = field(default_factory=list)
    source_evidence: dict[str, Any] = field(default_factory=dict)
    token_budget_class: str = "none"
    token_budget: int = 0
    schema_version: str = LITERATURE_ESCALATION_DECISION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_literature_escalation(
    *,
    native_result: dict[str, Any] | None = None,
    route_audit: dict[str, Any] | None = None,
    frontier_report: dict[str, Any] | None = None,
    user_objective: str = "",
    user_requested_literature: bool = False,
    policy: LiteratureEscalationPolicy | None = None,
) -> LiteratureEscalationDecision:
    """Return a deterministic decision for literature-mode escalation."""
    policy = policy or LiteratureEscalationPolicy()
    native = dict(native_result or {})
    audit = dict(route_audit or {})
    frontier = dict(frontier_report or {})
    reasons: list[str] = []

    native_solved = bool(native.get("solved"))
    native_routes = native.get("routes") or []
    stock_audit_passed = bool(audit.get("stock_audit_passed") or native.get("stock_audit_passed"))
    route_status = str(audit.get("route_status") or native.get("route_status") or "")
    audit_reasons = [str(item) for item in audit.get("reasons") or native.get("audit_reasons") or []]

    if user_requested_literature or "literature" in str(user_objective or "").lower():
        reasons.append("user_requested_literature")
    if not native_solved or not native_routes:
        reasons.append("native_failed")
    if native_routes and (not native_solved or not _all_route_leaves_stock_closed(native)):
        reasons.append("unclosed_route")
    if _fake_closure_risk(native, audit, frontier):
        reasons.append("fake_closure_risk")
    if bool(frontier.get("advanced_frontier_found")) or _frontier_flags(frontier).intersection(
        {"advanced_same_scaffold", "no_complexity_drop", "unresolved_core"}
    ):
        reasons.append("advanced_frontier_detected")
    if route_status in {"fake_closed_rejected", "unresolved"} and audit_reasons:
        reasons.append("route_audit_failed")

    native_solved_audit_passed = bool(
        native_solved
        and native_routes
        and stock_audit_passed
        and route_status in {"", "solved", "semisynthesis_closed"}
        and not _fake_closure_risk(native, audit, frontier)
        and _all_route_leaves_stock_closed(native)
    )
    if native_solved_audit_passed and "user_requested_literature" not in reasons:
        reasons = []

    reasons = _dedupe([reason for reason in reasons if reason in TRIGGER_REASONS])
    budget_class, token_budget = _budget_for_decision(policy, reasons, frontier)
    return LiteratureEscalationDecision(
        should_enter_literature_mode=bool(reasons),
        escalation_reason=reasons,
        source_evidence={
            "native_solved": native_solved,
            "native_route_count": len(native_routes),
            "route_status": route_status,
            "stock_audit_passed": stock_audit_passed,
            "audit_reasons": audit_reasons,
            "frontier_flags": sorted(_frontier_flags(frontier)),
            "user_objective": user_objective,
        },
        token_budget_class=budget_class,
        token_budget=token_budget,
    )


def _budget_for_decision(
    policy: LiteratureEscalationPolicy,
    reasons: list[str],
    frontier_report: dict[str, Any],
) -> tuple[str, int]:
    if not reasons:
        return "none", 0
    flags = _frontier_flags(frontier_report)
    if "full_text_si_required" in flags:
        return "full_text_si", policy.full_text_si_token_budget
    if flags.intersection({"advanced_same_scaffold", "polycyclic_or_steroid_like"}) or "advanced_frontier_detected" in reasons:
        return "high_complexity_natural_product", policy.high_complexity_natural_product_token_budget
    return "ordinary_complex_case", policy.complex_case_token_budget


def _frontier_flags(frontier_report: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    for item in frontier_report.get("frontiers") or []:
        flags.update(str(flag) for flag in item.get("flags") or [])
    flags.update(str(flag) for flag in frontier_report.get("flags") or [])
    return flags


def _fake_closure_risk(native: dict[str, Any], audit: dict[str, Any], frontier: dict[str, Any]) -> bool:
    if bool(audit.get("fake_closure_rejected") or native.get("fake_closure_rejected")):
        return True
    if audit.get("rejected_terminal_list") or native.get("rejected_terminal_list"):
        return True
    flags = _frontier_flags(frontier)
    return bool(flags.intersection({"ordinary_decoration_only", "no_complexity_drop"}))


def _all_route_leaves_stock_closed(native: dict[str, Any]) -> bool:
    routes = native.get("routes") or []
    if not routes:
        return False
    for route in routes:
        if not isinstance(route, dict):
            return False
        if route.get("unresolved_frontiers"):
            return False
        if route.get("stock_closed") is False:
            return False
        leaves = route.get("leaves") or []
        if leaves and any(isinstance(leaf, dict) and leaf.get("in_stock") is False for leaf in leaves):
            return False
    return True


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
