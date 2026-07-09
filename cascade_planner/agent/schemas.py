"""Schema helpers for LLM priors and route critiques.

These dataclasses are intentionally dependency-free so the planner does not
require a validation framework at runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALLOWED_ROUTE_MODES = {
    "organic_only",
    "enzymatic_only",
    "chemoenzymatic_cascade",
    "enzymatic_late_stage",
    "unknown",
}

ALLOWED_REACTION_TYPES = {
    "oxidation",
    "reduction",
    "acylation",
    "hydrolysis",
    "amination",
    "C_C_coupling",
    "isomerization",
    "phosphorylation",
    "glycosylation",
    "functional_group_interconversion",
    "racemization",
    "esterification",
    "other",
    "resolution",
    "cofactor_regeneration",
    "dehalogenation",
    "amidation",
    "epoxide_hydrolysis",
}

ALLOWED_EC1 = {1, 2, 3, 4, 5, 6, 7}
ALLOWED_SEVERITY = {"low", "medium", "high"}

ALLOWED_ACTIONS = {
    "increase_candidate_budget",
    "relax_ec_constraint",
    "relax_condition_window",
    "try_alternative_route_mode",
    "request_more_evidence",
    "accept_route",
}


def clamp_weight(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, x))


@dataclass
class RouteModePrior:
    mode: str
    weight: float
    rationale: str = ""

    def normalize(self) -> "RouteModePrior":
        if self.mode not in ALLOWED_ROUTE_MODES:
            self.mode = "unknown"
        self.weight = clamp_weight(self.weight)
        return self


@dataclass
class ReactionTypePrior:
    slot: int | None
    reaction_type: str
    weight: float
    rationale: str = ""

    def normalize(self) -> "ReactionTypePrior":
        self.weight = clamp_weight(self.weight)
        if self.reaction_type not in ALLOWED_REACTION_TYPES:
            self.reaction_type = "other"
        if self.slot is not None:
            self.slot = max(0, int(self.slot))
        return self


@dataclass
class EnzymePrior:
    ec1: int | None = None
    ec2: str = ""
    cofactor: str = ""
    substrate_family: str = ""
    weight: float = 0.5
    rationale: str = ""

    def normalize(self) -> "EnzymePrior":
        if self.ec1 is not None:
            try:
                self.ec1 = int(self.ec1)
            except (TypeError, ValueError):
                self.ec1 = None
        if self.ec1 not in ALLOWED_EC1:
            self.ec1 = None
        self.weight = clamp_weight(self.weight)
        return self


@dataclass
class ConditionRisk:
    kind: str
    severity: str = "medium"
    evidence: str = "prior_only"

    def normalize(self) -> "ConditionRisk":
        if self.severity not in ALLOWED_SEVERITY:
            self.severity = "medium"
        return self


@dataclass
class StrategicPrior:
    target_smiles: str
    route_mode_priors: list[RouteModePrior] = field(default_factory=list)
    reaction_type_priors: list[ReactionTypePrior] = field(default_factory=list)
    enzyme_priors: list[EnzymePrior] = field(default_factory=list)
    condition_risks: list[ConditionRisk] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    source: str = "deterministic"

    def normalize(self) -> "StrategicPrior":
        for item in self.reaction_type_priors:
            if item.reaction_type not in ALLOWED_REACTION_TYPES:
                self.unsupported_claims.append(f"unsupported_reaction_type_prior:{item.reaction_type}")
        for item in self.enzyme_priors:
            try:
                ec1_value = int(item.ec1) if item.ec1 is not None else None
            except (TypeError, ValueError):
                ec1_value = None
            if ec1_value is not None and ec1_value not in ALLOWED_EC1:
                self.unsupported_claims.append(f"unsupported_ec1_prior:{item.ec1}")
        self.route_mode_priors = [x.normalize() for x in self.route_mode_priors]
        self.reaction_type_priors = [x.normalize() for x in self.reaction_type_priors]
        self.enzyme_priors = [x.normalize() for x in self.enzyme_priors]
        self.condition_risks = [x.normalize() for x in self.condition_risks]
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalize())


@dataclass
class CritiqueFinding:
    severity: str
    kind: str
    message: str
    evidence_path: str


@dataclass
class SearchSuggestion:
    action: str
    rationale: str
    budget_hint: int | None = None

    def normalize(self) -> "SearchSuggestion":
        if self.action not in ALLOWED_ACTIONS:
            self.action = "request_more_evidence"
        return self


@dataclass
class RouteCritique:
    route_id: str
    acceptability: str = "uncertain"
    findings: list[CritiqueFinding] = field(default_factory=list)
    search_suggestions: list[SearchSuggestion] = field(default_factory=list)
    hallucinated_claims: list[str] = field(default_factory=list)
    source: str = "deterministic"

    def normalize(self) -> "RouteCritique":
        self.search_suggestions = [x.normalize() for x in self.search_suggestions]
        if self.acceptability not in {"acceptable", "needs_review", "reject", "uncertain"}:
            self.acceptability = "uncertain"
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalize())
