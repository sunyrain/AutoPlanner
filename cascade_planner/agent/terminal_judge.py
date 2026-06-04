"""Deterministic compiled terminal judge for P1c.

The judge is a lightweight search-time gate. It consumes validated policy
artifacts and returns accept/reject/defer decisions for terminal closure; it
does not call an LLM or generate reactions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from rdkit import Chem, RDLogger

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


RDLogger.DisableLog("rdApp.*")

JUDGE_POLICY_SCHEMA = "judge_policy.v1"
TERMINAL_JUDGE_DECISION_SCHEMA = "terminal_judge_decision.v1"


@dataclass
class JudgePolicy:
    policy_id: str
    case_id: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    terminal_blacklist: list[str] = field(default_factory=list)
    anchor_whitelist: list[str] = field(default_factory=list)
    stock_tier_rule: str = "default"
    same_scaffold_risk_threshold: float = 0.85
    material_sanity_mode: str = "conservative"
    schema_version: str = JUDGE_POLICY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["terminal_blacklist"] = sorted(_canonical_list(self.terminal_blacklist))
        data["anchor_whitelist"] = sorted(_canonical_list(self.anchor_whitelist))
        data["evidence_refs"] = sorted(str(item) for item in self.evidence_refs if item)
        return data


@dataclass
class TerminalJudgeDecision:
    decision: str
    reason: str
    smiles: str
    canonical_smiles: str
    policy_id: str = ""
    case_id: str = ""
    terminal_role: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    schema_version: str = TERMINAL_JUDGE_DECISION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_judge_policy_from_search_policy(search_policy: dict[str, Any]) -> JudgePolicy:
    """Compile a ChemEnzySearchPolicy-like payload into a terminal JudgePolicy."""
    if not search_policy:
        raise ValueError("missing_search_policy")
    policy_id = str(search_policy.get("policy_id") or "")
    if not policy_id:
        raise ValueError("missing_policy_id")
    policy = JudgePolicy(
        policy_id=f"{policy_id}_terminal_judge",
        case_id=str(search_policy.get("case_id") or ""),
        evidence_refs=[str(item) for item in search_policy.get("evidence_refs") or []],
        terminal_blacklist=[str(item) for item in search_policy.get("terminal_blacklist") or []],
        anchor_whitelist=[str(item) for item in search_policy.get("anchor_whitelist") or []],
        stock_tier_rule=str(search_policy.get("stock_tier_rule") or "default"),
        same_scaffold_risk_threshold=float(search_policy.get("same_scaffold_risk_threshold") or 0.85),
        material_sanity_mode=str(search_policy.get("material_sanity_mode") or "conservative"),
    )
    validation = validate_judge_policy(policy)
    if not validation["accepted"]:
        raise ValueError(f"invalid JudgePolicy: {validation['reasons']}")
    return policy


def judge_policy_from_constraints(constraints: dict[str, Any] | None) -> JudgePolicy | None:
    constraints = dict(constraints or {})
    raw = (
        constraints.get("judge_policy")
        or constraints.get("compiled_judge_policy")
        or constraints.get("terminal_judge_policy")
    )
    if raw in (None, "", [], {}):
        search_policy = constraints.get("chem_enzy_search_policy") or constraints.get("search_policy")
        if search_policy in (None, "", [], {}):
            return None
        return compile_judge_policy_from_search_policy(dict(search_policy or {}))
    policy = judge_policy_from_dict(dict(raw or {}))
    validation = validate_judge_policy(policy)
    if not validation["accepted"]:
        raise ValueError(f"invalid JudgePolicy: {validation['reasons']}")
    if _judge_policy_empty(policy):
        return None
    return policy


def judge_policy_from_dict(data: dict[str, Any]) -> JudgePolicy:
    return JudgePolicy(
        policy_id=str(data.get("policy_id") or ""),
        case_id=str(data.get("case_id") or ""),
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        terminal_blacklist=[str(item) for item in data.get("terminal_blacklist") or []],
        anchor_whitelist=[str(item) for item in data.get("anchor_whitelist") or []],
        stock_tier_rule=str(data.get("stock_tier_rule") or "default"),
        same_scaffold_risk_threshold=float(data.get("same_scaffold_risk_threshold") or 0.85),
        material_sanity_mode=str(data.get("material_sanity_mode") or "conservative"),
        schema_version=str(data.get("schema_version") or JUDGE_POLICY_SCHEMA),
    )


def evaluate_terminal_judge(
    smiles: str,
    policy: JudgePolicy | None,
    *,
    stock_checker: Callable[[str], bool] | None = None,
) -> TerminalJudgeDecision:
    can = canonical_smiles(smiles) or ""
    if policy is None:
        return _decision("defer", "no_policy", smiles, can)
    if not can:
        return _decision("reject", "invalid_smiles", smiles, can, policy=policy)
    blacklist = set(_canonical_list(policy.terminal_blacklist))
    if can in blacklist:
        return _decision("reject", "terminal_blacklist", smiles, can, policy=policy, role="forbidden_terminal")
    anchors = set(_canonical_list(policy.anchor_whitelist))
    if can in anchors:
        return _decision("accept", "anchor_whitelist", smiles, can, policy=policy, role="anchor_terminal")
    if policy.material_sanity_mode == "conservative" and _heavy_atoms(smiles) <= 0:
        return _decision("reject", "material_sanity_invalid", smiles, can, policy=policy)
    if policy.stock_tier_rule in {"strict_stock_only", "stock_only"}:
        if stock_checker is None:
            return _decision("defer", "stock_checker_unavailable", smiles, can, policy=policy)
        try:
            if bool(stock_checker(smiles)):
                return _decision("accept", "strict_stock_hit", smiles, can, policy=policy, role="strict_stock")
        except Exception:
            return _decision("reject", "stock_checker_error", smiles, can, policy=policy)
        return _decision("reject", "strict_stock_miss", smiles, can, policy=policy)
    return _decision("defer", "no_policy_rule_matched", smiles, can, policy=policy)


def validate_judge_policy(policy_or_data: JudgePolicy | dict[str, Any]) -> dict[str, Any]:
    policy = policy_or_data if isinstance(policy_or_data, JudgePolicy) else judge_policy_from_dict(policy_or_data)
    reasons: list[str] = []
    if policy.schema_version != JUDGE_POLICY_SCHEMA:
        reasons.append("invalid_judge_policy_schema")
    if not policy.policy_id and not _judge_policy_empty(policy):
        reasons.append("missing_policy_id")
    if policy.stock_tier_rule not in {"default", "strict_stock_only", "stock_only"}:
        reasons.append("invalid_stock_tier_rule")
    if policy.material_sanity_mode not in {"off", "conservative"}:
        reasons.append("invalid_material_sanity_mode")
    if not (0.0 <= float(policy.same_scaffold_risk_threshold) <= 1.0):
        reasons.append("invalid_same_scaffold_risk_threshold")
    if _invalid_smiles(policy.terminal_blacklist):
        reasons.append("invalid_terminal_blacklist_smiles")
    if _invalid_smiles(policy.anchor_whitelist):
        reasons.append("invalid_anchor_whitelist_smiles")
    overlap = set(_canonical_list(policy.terminal_blacklist)).intersection(_canonical_list(policy.anchor_whitelist))
    if overlap:
        reasons.append("terminal_blacklist_anchor_whitelist_overlap")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "policy_id": policy.policy_id,
        "schema_version": JUDGE_POLICY_SCHEMA,
    }


def _decision(
    decision: str,
    reason: str,
    smiles: str,
    can: str,
    *,
    policy: JudgePolicy | None = None,
    role: str = "",
) -> TerminalJudgeDecision:
    return TerminalJudgeDecision(
        decision=decision,
        reason=reason,
        smiles=str(smiles or ""),
        canonical_smiles=str(can or ""),
        policy_id=policy.policy_id if policy is not None else "",
        case_id=policy.case_id if policy is not None else "",
        terminal_role=role,
        evidence_refs=list(policy.evidence_refs if policy is not None else []),
    )


def _judge_policy_empty(policy: JudgePolicy) -> bool:
    return (
        not policy.terminal_blacklist
        and not policy.anchor_whitelist
        and policy.stock_tier_rule == "default"
        and policy.material_sanity_mode in {"", "off", "conservative"}
    )


def _canonical_list(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        can = canonical_smiles(str(value or ""))
        if can:
            out.append(can)
    return out


def _invalid_smiles(values: list[str]) -> bool:
    for value in values:
        if not value or Chem.MolFromSmiles(str(value)) is None:
            return True
    return False


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return mol.GetNumHeavyAtoms() if mol is not None else 0
