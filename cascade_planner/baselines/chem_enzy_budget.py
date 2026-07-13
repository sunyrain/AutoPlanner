"""Central ChemEnzy per-attempt budget contract.

The action planner may request a budget, but the host approves the concrete
per-attempt budget and the backend reports the effective ``RouteSearchConfig``.
Keeping those three values separate prevents a cheap probe from silently
masquerading as a full search.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from rdkit import Chem


CHEMENZY_BUDGET_RESOLUTION_SCHEMA = "chemenzy_budget_resolution.v1"
CHEMENZY_ATTEMPT_OUTCOME_SCHEMA = "chemenzy_attempt_outcome.v1"

ChemEnzyActionKind = Literal["native", "guided", "child"]
ChemEnzyAttemptKind = Literal["probe", "standard", "retry"]
ChemEnzyBudgetAuthority = Literal[
    "planner_advisory",
    "host_profile",
    "operator_explicit",
]

_ACTION_KIND_VALUES = {"native", "guided", "child"}
_ATTEMPT_KIND_VALUES = {"probe", "standard", "retry"}
_AUTHORITY_VALUES = {"planner_advisory", "host_profile", "operator_explicit"}
_COMPLEX_HEAVY_ATOM_THRESHOLD = 25


@dataclass(frozen=True, slots=True)
class ChemEnzyBudgetValues:
    max_depth: int | None = None
    max_iterations: int | None = None
    expansion_topk: int | None = None
    timeout_s: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "expansion_topk": self.expansion_topk,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True, slots=True)
class ChemEnzyBudgetResolution:
    action_kind: ChemEnzyActionKind
    attempt_kind: ChemEnzyAttemptKind
    attempt_index: int
    authority: ChemEnzyBudgetAuthority
    target_smiles: str
    canonical_target_smiles: str
    target_heavy_atoms: int
    complexity_profile: str
    requested_budget: ChemEnzyBudgetValues
    policy_budget: ChemEnzyBudgetValues
    profile_budget: ChemEnzyBudgetValues
    floor_budget: ChemEnzyBudgetValues
    cap_budget: ChemEnzyBudgetValues
    attempt_budget: ChemEnzyBudgetValues
    effective_budget: ChemEnzyBudgetValues | None = None
    adjustments: tuple[str, ...] = ()
    schema_version: str = CHEMENZY_BUDGET_RESOLUTION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_kind": self.action_kind,
            "attempt_kind": self.attempt_kind,
            "attempt_index": self.attempt_index,
            "authority": self.authority,
            "target_smiles": self.target_smiles,
            "canonical_target_smiles": self.canonical_target_smiles,
            "target_heavy_atoms": self.target_heavy_atoms,
            "complexity_profile": self.complexity_profile,
            "requested_budget": self.requested_budget.to_dict(),
            "policy_budget": self.policy_budget.to_dict(),
            "profile_budget": self.profile_budget.to_dict(),
            "floor_budget": self.floor_budget.to_dict(),
            "cap_budget": self.cap_budget.to_dict(),
            "attempt_budget": self.attempt_budget.to_dict(),
            "effective_budget": self.effective_budget.to_dict() if self.effective_budget else None,
            "adjustments": list(self.adjustments),
        }


def resolve_chemenzy_budget(
    *,
    target_smiles: str,
    action_kind: ChemEnzyActionKind,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
    authority: ChemEnzyBudgetAuthority,
    attempt_index: int,
    prior_attempt: Mapping[str, Any] | None = None,
    timeout_cap_s: float | None = None,
) -> ChemEnzyBudgetResolution:
    """Resolve raw request values into one bounded host-approved attempt."""

    if action_kind not in _ACTION_KIND_VALUES:
        raise ValueError(f"unsupported ChemEnzy action kind: {action_kind}")
    if authority not in _AUTHORITY_VALUES:
        raise ValueError(f"unsupported ChemEnzy budget authority: {authority}")

    target_raw = str(target_smiles or "").strip()
    molecule = Chem.MolFromSmiles(target_raw)
    if molecule is None:
        raise ValueError("invalid ChemEnzy budget target SMILES")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    heavy_atoms = int(molecule.GetNumHeavyAtoms())
    complex_target = heavy_atoms >= _COMPLEX_HEAVY_ATOM_THRESHOLD
    attempt_index_int = max(1, int(attempt_index or 1))
    attempt_kind = _resolve_attempt_kind(
        payload,
        authority=authority,
        attempt_index=attempt_index_int,
        prior_attempt=prior_attempt,
    )

    requested = budget_values_from_action_payload(payload)
    policy_values = budget_values_from_policy(policy or {})
    selected = _merge_budget_values(requested, policy_values)
    profile = _profile_budget(
        attempt_kind=attempt_kind,
        complex_target=complex_target,
        timeout_cap_s=timeout_cap_s,
    )
    floor = _floor_budget(attempt_kind=attempt_kind, complex_target=complex_target)
    cap = _cap_budget(
        authority=authority,
        attempt_kind=attempt_kind,
        timeout_cap_s=timeout_cap_s,
    )
    attempt, adjustments = _approve_budget(
        selected,
        profile=profile,
        floor=floor,
        cap=cap,
        apply_floor=authority != "operator_explicit" and attempt_kind != "probe",
    )

    return ChemEnzyBudgetResolution(
        action_kind=action_kind,
        attempt_kind=attempt_kind,
        attempt_index=attempt_index_int,
        authority=authority,
        target_smiles=target_raw,
        canonical_target_smiles=canonical,
        target_heavy_atoms=heavy_atoms,
        complexity_profile="complex" if complex_target else "simple",
        requested_budget=requested,
        policy_budget=policy_values,
        profile_budget=profile,
        floor_budget=floor,
        cap_budget=cap,
        attempt_budget=attempt,
        adjustments=tuple(adjustments),
    )


def budget_values_from_action_payload(payload: Mapping[str, Any]) -> ChemEnzyBudgetValues:
    return ChemEnzyBudgetValues(
        max_depth=_optional_positive_int(payload.get("max_steps")),
        max_iterations=_optional_positive_int(payload.get("chem_enzy_iterations")),
        expansion_topk=_optional_positive_int(payload.get("chem_enzy_expansion_topk")),
        timeout_s=_optional_positive_float(payload.get("timeout_s")),
    )


def budget_values_from_policy(policy: Mapping[str, Any]) -> ChemEnzyBudgetValues:
    budget = policy.get("budget") if isinstance(policy.get("budget"), Mapping) else {}
    return ChemEnzyBudgetValues(
        max_depth=_optional_positive_int(budget.get("max_depth")),
        max_iterations=_optional_positive_int(budget.get("max_iterations")),
        expansion_topk=_optional_positive_int(budget.get("expansion_topk")),
        timeout_s=_optional_positive_float(budget.get("timeout_s")),
    )


def budgeted_chemenzy_payload(
    payload: Mapping[str, Any],
    resolution: ChemEnzyBudgetResolution,
) -> dict[str, Any]:
    out = dict(payload or {})
    budget = resolution.attempt_budget
    out["max_steps"] = int(budget.max_depth or 1)
    out["chem_enzy_iterations"] = int(budget.max_iterations or 1)
    out["chem_enzy_expansion_topk"] = int(budget.expansion_topk or 1)
    if budget.timeout_s is not None:
        out["timeout_s"] = float(budget.timeout_s)
    out["chem_enzy_budget_resolution"] = resolution.to_dict()
    out["chem_enzy_action_kind"] = resolution.action_kind
    out["chem_enzy_attempt_kind"] = resolution.attempt_kind
    return out


def budgeted_chemenzy_policy(
    policy: Mapping[str, Any],
    resolution: ChemEnzyBudgetResolution,
) -> dict[str, Any]:
    out = dict(policy or {})
    budget = dict(out.get("budget") or {})
    approved = resolution.attempt_budget
    budget["max_depth"] = int(approved.max_depth or 1)
    budget["max_iterations"] = int(approved.max_iterations or 1)
    budget["expansion_topk"] = int(approved.expansion_topk or 1)
    if approved.timeout_s is not None:
        budget["timeout_s"] = float(approved.timeout_s)
    out["budget"] = budget
    compiler = dict(out.get("compiler_metadata") or {})
    compiler["budget_resolution_schema"] = resolution.schema_version
    compiler["budget_authority"] = resolution.authority
    compiler["attempt_kind"] = resolution.attempt_kind
    out["compiler_metadata"] = compiler
    return out


def finalize_effective_chemenzy_budget(
    resolution: ChemEnzyBudgetResolution,
    *,
    max_depth: int,
    max_iterations: int,
    expansion_topk: int,
    timeout_s: float | None = None,
) -> ChemEnzyBudgetResolution:
    effective = ChemEnzyBudgetValues(
        max_depth=max(1, int(max_depth)),
        max_iterations=max(1, int(max_iterations)),
        expansion_topk=max(1, int(expansion_topk)),
        timeout_s=_optional_positive_float(timeout_s),
    )
    approved = resolution.attempt_budget
    for field in ("max_depth", "max_iterations", "expansion_topk", "timeout_s"):
        actual = getattr(effective, field)
        ceiling = getattr(approved, field)
        if actual is not None and ceiling is not None and actual > ceiling:
            raise ValueError(f"effective ChemEnzy {field} exceeds approved attempt budget")
    adjustments = list(resolution.adjustments)
    for field in ("max_depth", "max_iterations", "expansion_topk", "timeout_s"):
        actual = getattr(effective, field)
        approved_value = getattr(approved, field)
        if actual is not None and approved_value is not None and actual < approved_value:
            adjustments.append(f"effective_{field}_reduced_by_backend_or_policy")
    return replace(
        resolution,
        effective_budget=effective,
        adjustments=tuple(_dedupe(adjustments)),
    )


def classify_chemenzy_attempt_outcome(
    resolution: ChemEnzyBudgetResolution,
    result: Mapping[str, Any],
    *,
    verifier: Mapping[str, Any] | None = None,
    verified_solved: bool | None = None,
) -> dict[str, Any]:
    raw_result = result.get("result") if isinstance(result.get("result"), Mapping) else result
    search_status = raw_result.get("search_status") if isinstance(raw_result.get("search_status"), Mapping) else {}
    routes = raw_result.get("routes") if isinstance(raw_result.get("routes"), list) else []
    try:
        route_count = max(len(routes), int(raw_result.get("n_results") or 0))
    except (TypeError, ValueError):
        route_count = len(routes)
    raw_solved = bool(
        raw_result.get("raw_solved")
        or search_status.get("raw_solved")
        or raw_result.get("solved")
        or search_status.get("solved")
    )
    status = str(search_status.get("status") or raw_result.get("status") or "").strip().lower()
    diagnoses = [
        str(item).strip().lower()
        for item in raw_result.get("failure_diagnosis") or []
        if str(item or "").strip()
    ]
    timed_out = status == "timeout" or any("timeout" in item for item in diagnoses)
    runtime_failed = any(
        token in item
        for item in diagnoses
        for token in ("initialization_failed", "runtime_unavailable", "capability_probe")
    )
    verifier_report = dict(verifier or {})
    verifier_present = bool(verifier_report)
    verifier_route_status = str(
        verifier_report.get("route_status") or ""
    ).strip().lower()
    verified_solved_bool = bool(verified_solved) if verified_solved is not None else bool(
        verifier_present
        and verifier_report.get("accepted") is True
        and verifier_route_status
        in {"solved", "graph_and_stock_closed", "reaction_validated"}
        and _safe_nonnegative_int(verifier_report.get("accepted_route_count")) > 0
    )
    verified_route_count = (
        _safe_nonnegative_int(verifier_report.get("accepted_route_count"))
        if verifier_present
        else 0
    )

    if verified_solved_bool:
        outcome = "solved"
        next_attempt = ""
        blocks_same_attempt = False
    elif verifier_present:
        outcome = "verification_rejected"
        next_attempt = (
            "standard"
            if resolution.attempt_kind == "probe"
            else "retry"
            if resolution.attempt_kind == "standard"
            else ""
        )
        blocks_same_attempt = resolution.attempt_kind != "probe"
    elif raw_solved:
        outcome = "verification_missing"
        next_attempt = (
            "standard"
            if resolution.attempt_kind == "probe"
            else "retry"
            if resolution.attempt_kind == "standard"
            else ""
        )
        blocks_same_attempt = resolution.attempt_kind != "probe"
    elif route_count > 0:
        outcome = "route_candidates_returned"
        next_attempt = ""
        blocks_same_attempt = False
    elif timed_out:
        outcome = "timeout"
        next_attempt = "retry"
        blocks_same_attempt = True
    elif runtime_failed:
        outcome = "runtime_failed"
        next_attempt = ""
        blocks_same_attempt = True
    elif resolution.attempt_kind == "probe":
        outcome = "probe_exhausted"
        next_attempt = "standard"
        blocks_same_attempt = False
    else:
        outcome = "search_exhausted"
        next_attempt = "retry" if resolution.attempt_kind == "standard" else ""
        blocks_same_attempt = True

    attempt_id = "chemenzy-attempt:" + hashlib.sha256(
        (
            f"{resolution.action_kind}\x1f{resolution.canonical_target_smiles}\x1f"
            f"{resolution.attempt_index}\x1f{resolution.attempt_kind}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "schema_version": CHEMENZY_ATTEMPT_OUTCOME_SCHEMA,
        "attempt_id": attempt_id,
        "outcome": outcome,
        "action_kind": resolution.action_kind,
        "attempt_kind": resolution.attempt_kind,
        "attempt_index": resolution.attempt_index,
        "target_smiles": resolution.target_smiles,
        "canonical_target_smiles": resolution.canonical_target_smiles,
        "route_count": route_count,
        "raw_route_count": route_count,
        "verified_route_count": verified_route_count,
        "raw_solved": raw_solved,
        "verified_solved": verified_solved_bool,
        "solved": verified_solved_bool,
        "search_status": status,
        "search_exhaustive": resolution.attempt_kind != "probe",
        "next_attempt_kind": next_attempt,
        "blocks_same_attempt": blocks_same_attempt,
        "verifier_present": verifier_present,
        "verifier_accepted_claim": verifier_report.get("accepted") is True,
        "verifier_route_status": verifier_route_status,
        "verifier_reasons": [
            str(item)
            for item in verifier_report.get("reasons") or []
            if str(item or "").strip()
        ][:20],
        "verification_authority": "host_route_verifier",
        "raw_search_status_is_authority": False,
        "budget_resolution": resolution.to_dict(),
    }


def resolution_from_dict(payload: Mapping[str, Any]) -> ChemEnzyBudgetResolution:
    """Rehydrate a host resolution embedded in a subprocess request."""

    def values(key: str) -> ChemEnzyBudgetValues:
        raw = payload.get(key) if isinstance(payload.get(key), Mapping) else {}
        return ChemEnzyBudgetValues(
            max_depth=_optional_positive_int(raw.get("max_depth")),
            max_iterations=_optional_positive_int(raw.get("max_iterations")),
            expansion_topk=_optional_positive_int(raw.get("expansion_topk")),
            timeout_s=_optional_positive_float(raw.get("timeout_s")),
        )

    action_kind = str(payload.get("action_kind") or "native")
    attempt_kind = str(payload.get("attempt_kind") or "standard")
    authority = str(payload.get("authority") or "host_profile")
    if action_kind not in _ACTION_KIND_VALUES or attempt_kind not in _ATTEMPT_KIND_VALUES or authority not in _AUTHORITY_VALUES:
        raise ValueError("invalid embedded ChemEnzy budget resolution")
    effective_raw = payload.get("effective_budget")
    return ChemEnzyBudgetResolution(
        schema_version=str(payload.get("schema_version") or CHEMENZY_BUDGET_RESOLUTION_SCHEMA),
        action_kind=action_kind,  # type: ignore[arg-type]
        attempt_kind=attempt_kind,  # type: ignore[arg-type]
        attempt_index=max(1, int(payload.get("attempt_index") or 1)),
        authority=authority,  # type: ignore[arg-type]
        target_smiles=str(payload.get("target_smiles") or ""),
        canonical_target_smiles=str(payload.get("canonical_target_smiles") or ""),
        target_heavy_atoms=max(0, int(payload.get("target_heavy_atoms") or 0)),
        complexity_profile=str(payload.get("complexity_profile") or "simple"),
        requested_budget=values("requested_budget"),
        policy_budget=values("policy_budget"),
        profile_budget=values("profile_budget"),
        floor_budget=values("floor_budget"),
        cap_budget=values("cap_budget"),
        attempt_budget=values("attempt_budget"),
        effective_budget=values("effective_budget") if isinstance(effective_raw, Mapping) else None,
        adjustments=tuple(str(item) for item in payload.get("adjustments") or []),
    )


def _resolve_attempt_kind(
    payload: Mapping[str, Any],
    *,
    authority: ChemEnzyBudgetAuthority,
    attempt_index: int,
    prior_attempt: Mapping[str, Any] | None,
) -> ChemEnzyAttemptKind:
    prior = dict(prior_attempt or {})
    prior_outcome = str(prior.get("outcome") or "").strip().lower()
    prior_next_attempt = str(prior.get("next_attempt_kind") or "").strip().lower()
    if prior_next_attempt in {"standard", "retry"}:
        return prior_next_attempt  # type: ignore[return-value]
    if prior_outcome == "probe_exhausted":
        return "standard"
    if prior_outcome in {
        "search_exhausted",
        "timeout",
        "verification_rejected",
        "verification_missing",
    }:
        return "retry"

    requested = str(payload.get("attempt_kind") or payload.get("chem_enzy_attempt_kind") or "").strip().lower()
    probe_requested = bool(payload.get("initial_probe")) or requested == "probe"
    if probe_requested and (attempt_index <= 1 or authority == "operator_explicit"):
        return "probe"
    if requested in _ATTEMPT_KIND_VALUES:
        return requested  # type: ignore[return-value]
    return "retry" if attempt_index > 1 else "standard"


def _profile_budget(
    *,
    attempt_kind: ChemEnzyAttemptKind,
    complex_target: bool,
    timeout_cap_s: float | None,
) -> ChemEnzyBudgetValues:
    timeout_cap = _optional_positive_float(timeout_cap_s)
    if attempt_kind == "probe":
        return ChemEnzyBudgetValues(6, 10, 20, min(180.0, timeout_cap) if timeout_cap else 180.0)
    if attempt_kind == "retry":
        return ChemEnzyBudgetValues(
            20 if complex_target else 12,
            60,
            120,
            min(600.0, timeout_cap) if timeout_cap else 600.0,
        )
    return ChemEnzyBudgetValues(
        20 if complex_target else 6,
        50 if complex_target else 10,
        100 if complex_target else 50,
        timeout_cap,
    )


def _floor_budget(*, attempt_kind: ChemEnzyAttemptKind, complex_target: bool) -> ChemEnzyBudgetValues:
    if attempt_kind == "probe":
        return ChemEnzyBudgetValues(1, 1, 1, 1.0)
    if attempt_kind == "retry":
        return ChemEnzyBudgetValues(20 if complex_target else 12, 60, 120, 1.0)
    return ChemEnzyBudgetValues(
        20 if complex_target else 6,
        50 if complex_target else 10,
        100 if complex_target else 50,
        1.0,
    )


def _cap_budget(
    *,
    authority: ChemEnzyBudgetAuthority,
    attempt_kind: ChemEnzyAttemptKind,
    timeout_cap_s: float | None,
) -> ChemEnzyBudgetValues:
    if attempt_kind == "probe":
        timeout_cap = _optional_positive_float(timeout_cap_s)
        return ChemEnzyBudgetValues(
            6,
            10,
            20,
            min(180.0, timeout_cap) if timeout_cap else 180.0,
        )
    if authority == "planner_advisory":
        return ChemEnzyBudgetValues(20, 200, 300, _optional_positive_float(timeout_cap_s))
    return ChemEnzyBudgetValues(20, 500, 500, _optional_positive_float(timeout_cap_s))


def _approve_budget(
    selected: ChemEnzyBudgetValues,
    *,
    profile: ChemEnzyBudgetValues,
    floor: ChemEnzyBudgetValues,
    cap: ChemEnzyBudgetValues,
    apply_floor: bool,
) -> tuple[ChemEnzyBudgetValues, list[str]]:
    adjustments: list[str] = []
    approved: dict[str, int | float | None] = {}
    for field in ("max_depth", "max_iterations", "expansion_topk", "timeout_s"):
        value = getattr(selected, field)
        if value is None:
            value = getattr(profile, field)
            adjustments.append(f"{field}_filled_from_attempt_profile")
        floor_value = getattr(floor, field)
        if apply_floor and value is not None and floor_value is not None and value < floor_value:
            value = floor_value
            adjustments.append(f"{field}_raised_to_attempt_floor")
        cap_value = getattr(cap, field)
        if value is not None and cap_value is not None and value > cap_value:
            value = cap_value
            adjustments.append(f"{field}_capped_by_authority")
        approved[field] = value
    return ChemEnzyBudgetValues(**approved), _dedupe(adjustments)


def _merge_budget_values(primary: ChemEnzyBudgetValues, fallback: ChemEnzyBudgetValues) -> ChemEnzyBudgetValues:
    return ChemEnzyBudgetValues(
        max_depth=primary.max_depth if primary.max_depth is not None else fallback.max_depth,
        max_iterations=primary.max_iterations if primary.max_iterations is not None else fallback.max_iterations,
        expansion_topk=primary.expansion_topk if primary.expansion_topk is not None else fallback.expansion_topk,
        timeout_s=primary.timeout_s if primary.timeout_s is not None else fallback.timeout_s,
    )


def _optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _optional_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in values if str(item or "").strip()))
