"""P1b strategic-operator compiler for guided ChemEnzy reruns.

This module is intentionally narrow: it compiles validated planning artifacts
into search flags and rerun metadata. It never injects raw reaction candidates
into ChemEnzy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.agent.case_trace import CaseBundle
from cascade_planner.agent.evidence_cards import validate_evidence_card
from cascade_planner.agent.literature_templates import (
    LITERATURE_TEMPLATE_PLUGIN_SOURCE,
    LiteratureTriggerReason,
    audit_native_run_for_literature,
)
from cascade_planner.agent.strategic_candidate_generation import validate_literature_candidate
from cascade_planner.agent.strategic_disconnection_miner import validate_strategic_disconnection_card
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteSearchConfig


RDLogger.DisableLog("rdApp.*")

STRATEGIC_OPERATOR_SCHEMA = "strategic_operator.v1"
CHEM_ENZY_SEARCH_POLICY_SCHEMA = "chem_enzy_search_policy.v1"
CHEM_ENZY_POLICY_TRACE_SCHEMA = "chem_enzy_policy_trace.v1"
GUIDED_RERUN_TRACE_SCHEMA = "guided_rerun_trace.v1"
POLICY_SEARCH_FLAG = "chem_enzy_search_policy"
LITERATURE_TEMPLATE_PLUGIN_FLAG = "literature_template_plugin"

RAW_REACTION_KEYS = {
    "raw_reaction",
    "raw_reactions",
    "raw_reaction_candidates",
    "reaction_candidates",
    "reaction_injection",
    "reaction_smiles",
    "rxn",
    "rxn_smiles",
    "rxn_smiles_list",
    "template_rxn_smiles",
}


@dataclass
class RerunBudget:
    max_reruns: int = 1
    max_iterations: int = 16
    max_depth: int = 6
    expansion_topk: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategicOperator:
    operator_id: str
    case_id: str
    evidence_refs: list[str]
    terminal_blacklist: list[str] = field(default_factory=list)
    anchor_whitelist: list[str] = field(default_factory=list)
    preferred_subgoal: dict[str, Any] = field(default_factory=dict)
    source_budget: dict[str, Any] = field(default_factory=dict)
    rerun_reason: str = ""
    budget: RerunBudget = field(default_factory=RerunBudget)
    input_artifact_refs: list[str] = field(default_factory=list)
    mode: str = "literature_guided_rerun"
    schema_version: str = STRATEGIC_OPERATOR_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["budget"] = self.budget.to_dict()
        return _sorted_lists(data)


@dataclass
class ChemEnzySearchPolicy:
    policy_id: str
    operator_id: str
    case_id: str
    evidence_refs: list[str]
    terminal_blacklist: list[str] = field(default_factory=list)
    anchor_whitelist: list[str] = field(default_factory=list)
    preferred_subgoal: dict[str, Any] = field(default_factory=dict)
    source_budget: dict[str, Any] = field(default_factory=dict)
    rerun_reason: str = ""
    budget: RerunBudget = field(default_factory=RerunBudget)
    mode: str = "guided"
    compiler_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CHEM_ENZY_SEARCH_POLICY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["budget"] = self.budget.to_dict()
        data["compiler_metadata"] = {
            "compiler_schema": "strategic_operator_compiler.v1",
            "not_raw_reaction_injection": True,
            **dict(self.compiler_metadata or {}),
        }
        return _sorted_lists(data)


def compile_strategic_operator_from_case_bundle(
    bundle: CaseBundle,
    *,
    max_reruns: int = 1,
    max_iterations: int = 16,
    max_depth: int = 6,
    expansion_topk: int = 50,
) -> StrategicOperator:
    """Compile accepted P0/P1a artifacts into a bounded StrategicOperator."""
    package_artifact = _accepted_artifact_payload(bundle, "HybridRoutePackage")
    validation_artifact = _accepted_artifact_payload(bundle, "RoutePackageValidation")
    evidence_cards = _accepted_artifact_payload(bundle, "EvidenceCardList") or []
    candidates = _accepted_artifact_payload(bundle, "LiteratureCandidateList") or []
    disconnection_cards = _accepted_artifact_payload_optional(bundle, "StrategicDisconnectionCardList") or []

    if not validation_artifact.get("accepted"):
        raise ValueError("route package validation is not accepted")
    if validation_artifact.get("route_status") not in {"partial_anchor", "ready_for_guided_rerun"}:
        raise ValueError(f"route status is not policy-compilable: {validation_artifact.get('route_status')}")

    validated_evidence = _validated_evidence_cards(evidence_cards)
    validated_candidates = _validated_candidates(candidates)
    evidence_ids = {str(card.get("evidence_id")) for card in validated_evidence}
    validated_disconnection_cards = _validated_strategic_disconnection_cards(disconnection_cards, evidence_ids)
    if not validated_evidence:
        raise ValueError("no validated evidence cards are available for policy compilation")
    if not validated_candidates:
        raise ValueError("no validated literature candidates are available for policy compilation")
    if disconnection_cards and not validated_disconnection_cards:
        raise ValueError("no validated search-ready StrategicDisconnectionCard is available for policy compilation")

    candidate_evidence_refs = sorted({
        str(ref)
        for candidate in validated_candidates
        for ref in candidate.get("evidence_refs", [])
        if ref
    })
    evidence_refs = candidate_evidence_refs or sorted(str(card.get("evidence_id")) for card in validated_evidence)
    missing_refs = [ref for ref in evidence_refs if ref not in evidence_ids]
    if missing_refs:
        raise ValueError(f"candidate evidence refs are not validated: {missing_refs}")

    frontier = package_artifact.get("frontier") or {}
    terminal_blacklist = _terminal_blacklist_from_frontier(frontier)
    anchor_whitelist = _anchor_whitelist_from_candidates(validated_candidates, validated_evidence)
    preferred_subgoal = _preferred_subgoal_from_package(
        package_artifact,
        validated_candidates,
        validated_disconnection_cards,
    )
    source_budget = _source_budget_from_candidates(validated_candidates, validated_disconnection_cards)
    input_artifact_refs = [
        "hybrid_route_package",
        "route_package_validation",
        "evidence_cards",
        "literature_candidates",
    ]
    if disconnection_cards:
        input_artifact_refs.append("strategic_disconnection_cards")
    operator = StrategicOperator(
        operator_id=f"{bundle.case_id}_strategic_operator_p1b",
        case_id=bundle.case_id,
        evidence_refs=evidence_refs,
        terminal_blacklist=terminal_blacklist,
        anchor_whitelist=anchor_whitelist,
        preferred_subgoal=preferred_subgoal,
        source_budget=source_budget,
        rerun_reason=_rerun_reason_from_bundle(bundle, validation_artifact, frontier),
        budget=RerunBudget(
            max_reruns=max_reruns,
            max_iterations=max_iterations,
            max_depth=max_depth,
            expansion_topk=expansion_topk,
        ),
        input_artifact_refs=input_artifact_refs,
    )
    validation = validate_strategic_operator(operator)
    if not validation["accepted"]:
        raise ValueError(f"invalid StrategicOperator: {validation['reasons']}")
    return operator


def compile_chem_enzy_search_policy(operator: StrategicOperator) -> ChemEnzySearchPolicy:
    """Compile a validated StrategicOperator into ChemEnzy search policy flags."""
    validation = validate_strategic_operator(operator)
    if not validation["accepted"]:
        raise ValueError(f"invalid StrategicOperator: {validation['reasons']}")

    policy_seed = json.dumps(operator.to_dict(), sort_keys=True, default=str)
    policy_digest = hashlib.sha1(policy_seed.encode("utf-8")).hexdigest()[:12]
    policy = ChemEnzySearchPolicy(
        policy_id=f"{operator.case_id}_chem_enzy_policy_{policy_digest}",
        operator_id=operator.operator_id,
        case_id=operator.case_id,
        evidence_refs=list(operator.evidence_refs),
        terminal_blacklist=list(operator.terminal_blacklist),
        anchor_whitelist=list(operator.anchor_whitelist),
        preferred_subgoal=dict(operator.preferred_subgoal or {}),
        source_budget=dict(operator.source_budget or {}),
        rerun_reason=operator.rerun_reason,
        budget=operator.budget,
        compiler_metadata={"input_operator_id": operator.operator_id},
    )
    policy_validation = validate_chem_enzy_search_policy(policy)
    if not policy_validation["accepted"]:
        raise ValueError(f"invalid ChemEnzySearchPolicy: {policy_validation['reasons']}")
    return policy


def apply_chem_enzy_search_policy(
    config: RouteSearchConfig,
    policy_or_payload: ChemEnzySearchPolicy | dict[str, Any],
) -> RouteSearchConfig:
    """Return a copy of a route config with bounded guided-policy flags."""
    policy = _policy_from_payload(policy_or_payload)
    validation = validate_chem_enzy_search_policy(policy)
    if not validation["accepted"]:
        raise ValueError(f"invalid ChemEnzySearchPolicy: {validation['reasons']}")

    flags = dict(config.search_flags or {})
    context = dict(flags.get("cascade_search_context") or {})
    context["enabled"] = True
    context["context_policy"] = "literature_guided_p1b"
    context["chem_enzy_policy_id"] = policy.policy_id
    context["policy_evidence_refs"] = list(policy.evidence_refs)
    context["terminal_blacklist"] = list(policy.terminal_blacklist)
    context["anchor_whitelist"] = list(policy.anchor_whitelist)
    context["preferred_subgoal"] = dict(policy.preferred_subgoal or {})
    if policy.source_budget.get("preferred_reaction_domains"):
        context["preferred_reaction_domains"] = list(policy.source_budget["preferred_reaction_domains"])
    active_failure_modes = list(context.get("active_failure_modes") or [])
    for mode in ("unresolved_core", "fake_closure_risk", "literature_guided_rerun"):
        if mode not in active_failure_modes:
            active_failure_modes.append(mode)
    context["active_failure_modes"] = active_failure_modes
    flags["cascade_search_context"] = context

    source_policy = dict(flags.get("cascade_source_policy") or {})
    source_policy["enabled"] = True
    source_policy["chem_enzy_policy_id"] = policy.policy_id
    for key, value in dict(policy.source_budget or {}).items():
        if key not in {"preferred_reaction_classes", "preferred_anchor_roles"}:
            source_policy[key] = value
    flags["cascade_source_policy"] = source_policy
    flags["use_cascade_source_policy"] = True
    flags[POLICY_SEARCH_FLAG] = policy.to_dict()

    budget = policy.budget
    return replace(
        config,
        max_iterations=_bounded_positive_int(config.max_iterations, budget.max_iterations),
        max_depth=_bounded_positive_int(config.max_depth, budget.max_depth),
        expansion_topk=_bounded_positive_int(config.expansion_topk, budget.expansion_topk),
        search_flags=flags,
    )


def apply_literature_template_plugin_policy(
    config: RouteSearchConfig,
    *,
    trigger_report: dict[str, Any] | None = None,
    native_result: BaselineRunResult | dict[str, Any] | None = None,
    route_audit: dict[str, Any] | None = None,
    frontier_report: dict[str, Any] | None = None,
    template_cards: list[dict[str, Any]] | None = None,
    top_k: int = 6,
    max_added: int = 6,
) -> RouteSearchConfig:
    """Enable literature executable-template plugin only after explicit triggers."""
    report = dict(trigger_report or {})
    if not report:
        report = audit_native_run_for_literature(
            native_result,
            route_audit=route_audit,
            frontier_report=frontier_report,
            user_requested=False,
        )
    if not report.get("should_trigger"):
        flags = dict(config.search_flags or {})
        flags.pop(LITERATURE_TEMPLATE_PLUGIN_FLAG, None)
        flags.pop("autoplanner_literature_template_plugin", None)
        return replace(config, search_flags=flags)

    reasons = [str(reason) for reason in report.get("trigger_reasons") or []]
    flags = dict(config.search_flags or {})
    plugin = dict(flags.get(LITERATURE_TEMPLATE_PLUGIN_FLAG) or {})
    plugin["enabled"] = True
    plugin["top_k"] = max(1, int(top_k or 1))
    plugin["max_added"] = max(1, int(max_added or 1))
    plugin["trigger_reasons"] = reasons
    plugin["requires_audit"] = True
    plugin["not_raw_reaction_injection"] = True
    if template_cards is not None:
        plugin["template_cards"] = list(template_cards)
    flags[LITERATURE_TEMPLATE_PLUGIN_FLAG] = plugin

    context = dict(flags.get("cascade_search_context") or {})
    context["enabled"] = True
    context["literature_template_plugin_enabled"] = True
    context["literature_trigger_reasons"] = reasons
    preferred_context_domains = list(context.get("preferred_reaction_domains") or [])
    for domain in ("literature_chemical", "literature_biocatalytic"):
        if domain not in preferred_context_domains:
            preferred_context_domains.append(domain)
    context["preferred_reaction_domains"] = preferred_context_domains
    active = list(context.get("active_failure_modes") or [])
    for reason in reasons:
        if reason not in active:
            active.append(reason)
    context["active_failure_modes"] = active
    flags["cascade_search_context"] = context

    source_policy = dict(flags.get("cascade_source_policy") or {})
    source_policy["enabled"] = True
    source_policy["literature_template_plugin"] = {
        "enabled": True,
        "domain": "literature_chemical",
        "source": LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        "trigger_reasons": reasons,
        "native_failed_topk_multiplier": 2.0
        if LiteratureTriggerReason.NATIVE_FAILED.value in set(reasons)
        else 1.0,
        "audit_passed_native_solved_disables": True,
    }
    preferred_domains = list(source_policy.get("preferred_reaction_domains") or [])
    for domain in ("literature_chemical", "literature_biocatalytic"):
        if domain not in preferred_domains:
            preferred_domains.append(domain)
    source_policy["preferred_reaction_domains"] = preferred_domains
    flags["cascade_source_policy"] = source_policy
    flags["use_cascade_source_policy"] = True
    return replace(config, search_flags=flags)


def run_bounded_guided_rerun(
    adapter: Any,
    config: RouteSearchConfig,
    policy_or_payload: ChemEnzySearchPolicy | dict[str, Any],
    *,
    baseline_result: BaselineRunResult | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run at most one guided ChemEnzy rerun and return a trace dictionary."""
    policy = _policy_from_payload(policy_or_payload)
    validation = validate_chem_enzy_search_policy(policy)
    if not validation["accepted"]:
        raise ValueError(f"invalid ChemEnzySearchPolicy: {validation['reasons']}")

    trace = {
        "schema_version": GUIDED_RERUN_TRACE_SCHEMA,
        "case_id": policy.case_id,
        "policy_id": policy.policy_id,
        "operator_id": policy.operator_id,
        "evidence_refs": list(policy.evidence_refs),
        "rerun_budget": policy.budget.to_dict(),
        "attempts": [],
        "final_route_status": "unresolved",
        "stop_reason": "not_started",
    }
    if policy.budget.max_reruns <= 0:
        trace["stop_reason"] = "rerun_budget_exhausted"
        return {"trace": trace, "guided_result": None}

    guided_config = apply_chem_enzy_search_policy(config, policy)
    guided_result = adapter.run_target(guided_config, dry_run=dry_run)
    improved = _guided_result_improved(baseline_result, guided_result)
    attempt = {
        "attempt": 1,
        "policy_id": policy.policy_id,
        "route_count": guided_result.route_count,
        "solved": guided_result.solved,
        "failure_categories": [failure.category for failure in guided_result.failures],
        "improvement_detected": improved,
    }
    trace["attempts"].append(attempt)
    trace["stop_reason"] = "guided_improved" if improved else "no_improvement_budget_exhausted"
    if guided_result.solved and improved:
        trace["final_route_status"] = "solved"
    else:
        trace["final_route_status"] = "unresolved"
    return {"trace": trace, "guided_result": guided_result.to_dict()}


def validate_strategic_operator(operator_or_payload: StrategicOperator | dict[str, Any]) -> dict[str, Any]:
    try:
        operator = _operator_from_payload(operator_or_payload)
    except ValueError as exc:
        return {"accepted": False, "reasons": [str(exc)], "schema_version": STRATEGIC_OPERATOR_SCHEMA}
    reasons = _common_policy_reasons(operator.to_dict())
    if operator.schema_version != STRATEGIC_OPERATOR_SCHEMA:
        reasons.append("invalid_strategic_operator_schema")
    if not operator.operator_id:
        reasons.append("missing_operator_id")
    if not operator.case_id:
        reasons.append("missing_case_id")
    if operator.mode not in {"literature_guided_rerun", "stuck_node_rerun"}:
        reasons.append("invalid_operator_mode")
    if not operator.evidence_refs:
        reasons.append("missing_evidence_refs")
    if not operator.rerun_reason:
        reasons.append("missing_rerun_reason")
    reasons.extend(_budget_reasons(operator.budget))
    reasons.extend(_smiles_list_reasons("terminal_blacklist", operator.terminal_blacklist))
    reasons.extend(_smiles_list_reasons("anchor_whitelist", operator.anchor_whitelist))
    reasons.extend(_blacklist_anchor_overlap_reasons(operator.terminal_blacklist, operator.anchor_whitelist))
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "operator_id": operator.operator_id,
        "schema_version": STRATEGIC_OPERATOR_SCHEMA,
    }


def validate_chem_enzy_search_policy(policy_or_payload: ChemEnzySearchPolicy | dict[str, Any]) -> dict[str, Any]:
    try:
        policy = _policy_from_payload(policy_or_payload)
    except ValueError as exc:
        return {"accepted": False, "reasons": [str(exc)], "schema_version": CHEM_ENZY_SEARCH_POLICY_SCHEMA}
    reasons = _common_policy_reasons(policy.to_dict())
    if policy.schema_version != CHEM_ENZY_SEARCH_POLICY_SCHEMA:
        reasons.append("invalid_search_policy_schema")
    if policy.mode not in {"baseline", "guided", "literature-assisted", "stuck-node rerun"}:
        reasons.append("invalid_search_policy_mode")
    if not policy.policy_id:
        reasons.append("missing_policy_id")
    if not policy.operator_id:
        reasons.append("missing_operator_id")
    if not policy.case_id:
        reasons.append("missing_case_id")
    if not policy.evidence_refs:
        reasons.append("missing_evidence_refs")
    if not policy.rerun_reason:
        reasons.append("missing_rerun_reason")
    reasons.extend(_budget_reasons(policy.budget))
    reasons.extend(_smiles_list_reasons("terminal_blacklist", policy.terminal_blacklist))
    reasons.extend(_smiles_list_reasons("anchor_whitelist", policy.anchor_whitelist))
    reasons.extend(_blacklist_anchor_overlap_reasons(policy.terminal_blacklist, policy.anchor_whitelist))
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "policy_id": policy.policy_id,
        "schema_version": CHEM_ENZY_SEARCH_POLICY_SCHEMA,
    }


def validate_chem_enzy_search_policy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_chem_enzy_search_policy(payload)


def chem_enzy_policy_trace_from_search_flags(search_flags: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = dict(search_flags or {}).get(POLICY_SEARCH_FLAG)
    if not payload:
        return None
    validation = validate_chem_enzy_search_policy_payload(dict(payload or {}))
    budget = dict((payload or {}).get("budget") or {})
    return {
        "schema_version": CHEM_ENZY_POLICY_TRACE_SCHEMA,
        "policy_id": str((payload or {}).get("policy_id") or ""),
        "operator_id": str((payload or {}).get("operator_id") or ""),
        "case_id": str((payload or {}).get("case_id") or ""),
        "mode": str((payload or {}).get("mode") or ""),
        "evidence_refs": list((payload or {}).get("evidence_refs") or []),
        "rerun_budget": {
            "max_reruns": int(budget.get("max_reruns") or 0),
            "max_iterations": int(budget.get("max_iterations") or 0),
            "max_depth": int(budget.get("max_depth") or 0),
            "expansion_topk": int(budget.get("expansion_topk") or 0),
        },
        "validation": validation,
        "raw_reaction_injection": False,
    }


def _operator_from_payload(payload: StrategicOperator | dict[str, Any]) -> StrategicOperator:
    if isinstance(payload, StrategicOperator):
        if _has_raw_reaction_payload(payload.to_dict()):
            raise ValueError("raw_reaction_injection")
        return payload
    data = dict(payload or {})
    if _has_raw_reaction_payload(data):
        raise ValueError("raw_reaction_injection")
    budget = _budget_from_payload(data.get("budget") or {})
    return StrategicOperator(
        operator_id=str(data.get("operator_id") or ""),
        case_id=str(data.get("case_id") or ""),
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        terminal_blacklist=[str(item) for item in data.get("terminal_blacklist") or []],
        anchor_whitelist=[str(item) for item in data.get("anchor_whitelist") or []],
        preferred_subgoal=dict(data.get("preferred_subgoal") or {}),
        source_budget=dict(data.get("source_budget") or {}),
        rerun_reason=str(data.get("rerun_reason") or ""),
        budget=budget,
        input_artifact_refs=[str(item) for item in data.get("input_artifact_refs") or []],
        mode=str(data.get("mode") or "literature_guided_rerun"),
        schema_version=str(data.get("schema_version") or STRATEGIC_OPERATOR_SCHEMA),
    )


def _policy_from_payload(payload: ChemEnzySearchPolicy | dict[str, Any]) -> ChemEnzySearchPolicy:
    if isinstance(payload, ChemEnzySearchPolicy):
        if _has_raw_reaction_payload(payload.to_dict()):
            raise ValueError("raw_reaction_injection")
        return payload
    data = dict(payload or {})
    if _has_raw_reaction_payload(data):
        raise ValueError("raw_reaction_injection")
    budget = _budget_from_payload(data.get("budget") or {})
    return ChemEnzySearchPolicy(
        policy_id=str(data.get("policy_id") or ""),
        operator_id=str(data.get("operator_id") or ""),
        case_id=str(data.get("case_id") or ""),
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        terminal_blacklist=[str(item) for item in data.get("terminal_blacklist") or []],
        anchor_whitelist=[str(item) for item in data.get("anchor_whitelist") or []],
        preferred_subgoal=dict(data.get("preferred_subgoal") or {}),
        source_budget=dict(data.get("source_budget") or {}),
        rerun_reason=str(data.get("rerun_reason") or ""),
        budget=budget,
        mode=str(data.get("mode") or "guided"),
        compiler_metadata=dict(data.get("compiler_metadata") or {}),
        schema_version=str(data.get("schema_version") or CHEM_ENZY_SEARCH_POLICY_SCHEMA),
    )


def _budget_from_payload(data: dict[str, Any]) -> RerunBudget:
    return RerunBudget(
        max_reruns=int(data.get("max_reruns") or 1),
        max_iterations=int(data.get("max_iterations") or 16),
        max_depth=int(data.get("max_depth") or 6),
        expansion_topk=int(data.get("expansion_topk") or 50),
    )


def _accepted_artifact_payload(bundle: CaseBundle, artifact_type: str) -> Any:
    artifacts = bundle.accepted_artifacts(artifact_type)
    if not artifacts:
        raise ValueError(f"missing accepted artifact: {artifact_type}")
    return artifacts[-1].payload


def _accepted_artifact_payload_optional(bundle: CaseBundle, artifact_type: str) -> Any:
    artifacts = bundle.accepted_artifacts(artifact_type)
    if not artifacts:
        return None
    return artifacts[-1].payload


def _validated_evidence_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = []
    for card in cards:
        result = validate_evidence_card(card)
        if result["accepted"] and str(card.get("validation_status") or "") == "validated":
            validated.append(dict(card))
    return validated


def _validated_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = []
    for candidate in candidates:
        result = validate_literature_candidate(candidate)
        if result["accepted"] and str(candidate.get("validation_status") or "") == "validated":
            validated.append(dict(candidate))
    return validated


def _validated_strategic_disconnection_cards(
    cards: list[dict[str, Any]],
    validated_evidence_refs: set[str],
) -> list[dict[str, Any]]:
    validated = []
    for card in cards:
        result = validate_strategic_disconnection_card(
            card,
            validated_evidence_refs=validated_evidence_refs,
        )
        if (
            result["accepted"]
            and bool(result.get("usable_for_search"))
            and str(card.get("validation_status") or "") == "validated"
        ):
            validated.append(dict(card))
    return validated


def _terminal_blacklist_from_frontier(frontier: dict[str, Any]) -> list[str]:
    fake_close_flags = {
        "advanced_same_scaffold",
        "no_complexity_drop",
        "ordinary_decoration_only",
        "unresolved_core",
    }
    flags = {str(flag) for flag in frontier.get("flags") or []}
    frontier_smiles = str(frontier.get("frontier_smiles") or "")
    if frontier_smiles and flags.intersection(fake_close_flags) and _valid_smiles(frontier_smiles):
        return [frontier_smiles]
    return []


def _anchor_whitelist_from_candidates(
    candidates: list[dict[str, Any]],
    evidence_cards: list[dict[str, Any]],
) -> list[str]:
    anchors: list[str] = []
    for candidate in candidates:
        if candidate.get("candidate_kind") != "route_anchor":
            continue
        anchors.extend(str(item) for item in candidate.get("precursor_smiles") or [] if item)
    for card in evidence_cards:
        if card.get("route_role") != "route_anchor":
            continue
        record = ((card.get("source_metadata") or {}).get("record") or {})
        if record.get("smiles"):
            anchors.append(str(record["smiles"]))
    return sorted({smi for smi in anchors if _valid_smiles(smi)})


def _preferred_subgoal_from_package(
    package: dict[str, Any],
    candidates: list[dict[str, Any]],
    disconnection_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frontier = package.get("frontier") or {}
    templates = [
        dict(candidate.get("strategy_template") or {})
        for candidate in candidates
        if candidate.get("candidate_kind") in {"exact_fragment_retro", "forward_surrogate"}
    ]
    reaction_classes = sorted({
        str(candidate.get("reaction_class"))
        for candidate in candidates
        if candidate.get("reaction_class")
    })
    break_bonds = sorted({
        str(bond)
        for template in templates
        for bond in template.get("break_bonds", [])
        if bond
    })
    disconnection_cards = list(disconnection_cards or [])
    strategic_card_subgoals = sorted({
        str(card.get("strategic_subgoal"))
        for card in disconnection_cards
        if card.get("strategic_subgoal")
    })
    strategic_card_ids = sorted({
        str(card.get("card_id"))
        for card in disconnection_cards
        if card.get("card_id")
    })
    disconnection_types = sorted({
        str(card.get("disconnection_type"))
        for card in disconnection_cards
        if card.get("disconnection_type")
    })
    return {
        "schema_version": "preferred_subgoal.v1",
        "role": "advanced_frontier",
        "frontier_smiles": str(frontier.get("frontier_smiles") or ""),
        "frontier_flags": [str(flag) for flag in frontier.get("flags") or []],
        "preferred_reaction_classes": reaction_classes,
        "preferred_break_bonds": break_bonds,
        "strategic_disconnection_card_ids": strategic_card_ids,
        "strategic_disconnection_types": disconnection_types,
        "strategic_subgoals": strategic_card_subgoals,
        "not_raw_reaction_injection": True,
    }


def _source_budget_from_candidates(
    candidates: list[dict[str, Any]],
    disconnection_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reaction_classes = sorted({
        str(candidate.get("reaction_class"))
        for candidate in candidates
        if candidate.get("reaction_class")
    })
    anchor_roles = sorted({
        str(candidate.get("route_anchor_role"))
        for candidate in candidates
        if candidate.get("route_anchor_role")
    })
    disconnection_types = sorted({
        str(card.get("disconnection_type"))
        for card in list(disconnection_cards or [])
        if card.get("disconnection_type")
    })
    preferred_domains = ["chemical"]
    if any("biosynthetic" in role.lower() or "enzyme" in role.lower() for role in anchor_roles):
        preferred_domains.append("enzymatic")
    return {
        "enabled": True,
        "unpreferred_topk_fraction": 0.25,
        "min_unpreferred_topk": 5,
        "active_failure_topk_multiplier": 1.0,
        "preferred_reaction_domains": preferred_domains,
        "preferred_reaction_classes": reaction_classes,
        "preferred_anchor_roles": anchor_roles,
        "preferred_disconnection_types": disconnection_types,
    }


def _rerun_reason_from_bundle(
    bundle: CaseBundle,
    validation: dict[str, Any],
    frontier: dict[str, Any],
) -> str:
    reasons = [str(reason) for reason in validation.get("reasons") or [] if reason]
    reasons.extend(str(flag) for flag in frontier.get("flags") or [] if flag)
    reasons.extend(event.reason for event in bundle.failure_events if event.reason)
    if not reasons:
        reasons.append(str(validation.get("route_status") or "validated_literature_policy"))
    return ";".join(sorted(set(reasons)))


def _common_policy_reasons(payload: dict[str, Any]) -> list[str]:
    return ["raw_reaction_injection"] if _has_raw_reaction_payload(payload) else []


def _budget_reasons(budget: RerunBudget) -> list[str]:
    reasons: list[str] = []
    if budget.max_reruns < 0 or budget.max_reruns > 1:
        reasons.append("unbounded_or_unsupported_max_reruns")
    if budget.max_iterations <= 0 or budget.max_iterations > 5000:
        reasons.append("invalid_max_iterations")
    if budget.max_depth <= 0 or budget.max_depth > 30:
        reasons.append("invalid_max_depth")
    if budget.expansion_topk <= 0 or budget.expansion_topk > 1000:
        reasons.append("invalid_expansion_topk")
    return reasons


def _smiles_list_reasons(field_name: str, smiles_list: list[str]) -> list[str]:
    reasons: list[str] = []
    for smiles in smiles_list:
        if not _valid_smiles(smiles):
            reasons.append(f"invalid_{field_name}_smiles")
            break
    return reasons


def _blacklist_anchor_overlap_reasons(terminal_blacklist: list[str], anchor_whitelist: list[str]) -> list[str]:
    blacklist = {_canonical_smiles(smiles) for smiles in terminal_blacklist if _canonical_smiles(smiles)}
    anchors = {_canonical_smiles(smiles) for smiles in anchor_whitelist if _canonical_smiles(smiles)}
    if blacklist.intersection(anchors):
        return ["terminal_blacklist_anchor_whitelist_overlap"]
    return []


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _has_raw_reaction_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in RAW_REACTION_KEYS:
                return True
            if _has_raw_reaction_payload(item):
                return True
    elif isinstance(value, list):
        return any(_has_raw_reaction_payload(item) for item in value)
    elif isinstance(value, str) and ">>" in value:
        return True
    return False


def _guided_result_improved(
    baseline_result: BaselineRunResult | None,
    guided_result: BaselineRunResult,
) -> bool:
    if baseline_result is None:
        return bool(guided_result.solved or guided_result.route_count > 0)
    if guided_result.solved and not baseline_result.solved:
        return True
    if guided_result.route_count > baseline_result.route_count:
        return True
    if not guided_result.failures and baseline_result.failures:
        return True
    return False


def _bounded_positive_int(current: int, upper_bound: int) -> int:
    current_int = int(current)
    upper_int = int(upper_bound)
    if current_int <= 0:
        return upper_int
    return max(1, min(current_int, upper_int))


def _valid_smiles(smiles: str) -> bool:
    return bool(smiles and Chem.MolFromSmiles(str(smiles)) is not None)


def _sorted_lists(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("evidence_refs", "terminal_blacklist", "anchor_whitelist", "input_artifact_refs"):
        if key in data:
            data[key] = sorted(str(item) for item in data.get(key) or [])
    return data
