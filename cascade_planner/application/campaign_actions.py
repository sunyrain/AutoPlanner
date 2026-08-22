"""Target-blind action opportunities derived from the one deficit frontier."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping


CAMPAIGN_ACTION_OPPORTUNITY_SCHEMA = "campaign_action_opportunity.v1"
CAMPAIGN_ACTION_SET_SCHEMA = "campaign_action_opportunity_set.v1"
LEGACY_CAMPAIGN_ACTION_SCHEMA = "campaign_action.v1"
CAMPAIGN_ACTION_SCHEMA = "campaign_action.v2"
CAMPAIGN_ACTION_RESOURCE_ESTIMATE_SCHEMA = "campaign_action_resource_estimate.v1"
ACTION_ESTIMATE_SCHEMA = "campaign_action_estimate.v1"
ACTION_RESULT_SCHEMA = "campaign_action_result.v1"
MATERIALIZATION_ACTION_CONTRACT = "terminal_rejection.v1"


class CampaignActionKind(str, Enum):
    MATERIALIZE = "host_materialize"
    REACTION_VALIDATE = "reaction_validate"
    ACQUIRE_EVIDENCE = "acquire_exact_evidence"
    BIND_EVIDENCE = "bind_exact_evidence"
    CONDITION_ENRICH = "condition_enrich"
    STOCK_AUDIT = "stock_audit"
    RESOLVE_CONFLICT = "resolve_conflict"
    CHEMENZY_TARGET_EXPAND = "chemenzy_target_expand"
    CHEMENZY_FRONTIER_EXPAND = "chemenzy_frontier_expand"
    CODEX_GLOBAL_ARCHITECTURE = "codex_global_architecture"
    CODEX_REPLAN = "codex_global_replan"
    PROGRAM_DISCOVER = "program_discover"
    PROGRAM_REVIEW = "program_review"
    PROGRAM_ADMIT = "program_admit"
    PROGRAM_VALIDATE = "program_validate"
    EXPERIMENT_FEEDBACK_INGEST = "experiment_feedback_ingest"
    RECOMPUTE_ROUTE = "recompute_route_closure"


@dataclass(frozen=True, slots=True)
class CampaignActionOpportunity:
    action_id: str
    kind: CampaignActionKind
    deficit_id: str
    subject_ids: tuple[str, ...]
    route_family_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    producer: str
    resource_class: str
    deterministic: bool
    model_allowed: bool
    base_priority: float
    expected_route_gain: float
    expected_proof_gain: float
    expected_diversity_gain: float
    expected_dependency_unblock_count: int
    expected_novelty_gain: float
    success_probability_low: float
    success_probability_high: float
    success_probability_assessed: bool
    cost_penalty: float
    failure_risk_penalty: float
    uncertainty: Mapping[str, Any]
    reason: str
    metadata: Mapping[str, Any]
    schema_version: str = CAMPAIGN_ACTION_OPPORTUNITY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        row = {
            **asdict(self),
            "kind": self.kind.value,
            "subject_ids": list(self.subject_ids),
            "route_family_ids": list(self.route_family_ids),
            "dependency_ids": list(self.dependency_ids),
            "metadata": _json_value(self.metadata),
            "uncertainty": _json_value(self.uncertainty),
        }
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True, slots=True)
class CampaignAction:
    execution_id: str
    action_id: str
    kind: CampaignActionKind
    deficit_id: str
    input_revision: int
    opportunity_sha256: str
    opportunity_set_sha256: str
    subject_ids: tuple[str, ...]
    route_family_ids: tuple[str, ...]
    producer: str
    resource_class: str
    estimate: Mapping[str, Any]
    expected_resources: Mapping[str, Any]
    task_id: str
    idempotency_key: str
    reason: str
    metadata: Mapping[str, Any]
    schema_version: str = CAMPAIGN_ACTION_SCHEMA

    def __post_init__(self) -> None:
        if self.input_revision < 0:
            raise ValueError("campaign action input revision cannot be negative")
        required = (
            self.execution_id,
            self.action_id,
            self.deficit_id,
            self.opportunity_sha256,
            self.opportunity_set_sha256,
            self.task_id,
            self.idempotency_key,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError("campaign action identity is incomplete")

    def to_dict(self) -> dict[str, Any]:
        row = {
            **asdict(self),
            "kind": self.kind.value,
            "subject_ids": list(self.subject_ids),
            "route_family_ids": list(self.route_family_ids),
            "metadata": _json_value(self.metadata),
            "estimate": _json_value(self.estimate),
            "expected_resources": _json_value(self.expected_resources),
            "semantics": {
                "revision_bound": True,
                "wrapper_task_delegates_resource_accounting_to_handler": True,
                "grants_no_scientific_authority": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True, slots=True)
class ActionResult:
    action_execution_id: str
    action_sha256: str
    status: str
    input_revision: int
    output_revision: int
    immutable_artifact_refs: tuple[Mapping[str, Any], ...]
    actual_resources: Mapping[str, Any]
    resource_accounting: Mapping[str, Any]
    resource_reservation: Mapping[str, Any]
    material_events: tuple[Any, ...]
    candidate_delta: Mapping[str, Any]
    fact_delta: Mapping[str, Any]
    failure_type: str
    failure_reasons: tuple[str, ...]
    elapsed_s: float
    handler_result: Mapping[str, Any]
    schema_version: str = ACTION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if min(self.input_revision, self.output_revision) < 0:
            raise ValueError("action result revisions cannot be negative")
        if not self.action_execution_id or not self.action_sha256:
            raise ValueError("action result identity is incomplete")

    def to_dict(self) -> dict[str, Any]:
        row = {
            **asdict(self),
            "immutable_artifact_refs": _json_value(
                self.immutable_artifact_refs
            ),
            "actual_resources": _json_value(self.actual_resources),
            "resource_accounting": _json_value(self.resource_accounting),
            "resource_reservation": _json_value(self.resource_reservation),
            "material_events": _json_value(self.material_events),
            "candidate_delta": _json_value(self.candidate_delta),
            "fact_delta": _json_value(self.fact_delta),
            "failure_reasons": list(self.failure_reasons),
            "handler_result": _json_value(self.handler_result),
            "semantics": {
                "result_artifact_ref_is_owned_by_execution_envelope": True,
                "candidate_and_fact_deltas_grant_no_authority": True,
                "canonical_ingestion_remains_the_only_fact_write_path": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row


def bind_scheduled_action(
    decision: Mapping[str, Any],
    *,
    input_revision: int,
) -> CampaignAction:
    selected = dict(decision.get("selected_action") or {})
    if not selected or selected.get("eligible") is not True:
        raise ValueError("campaign action decision has no eligible selection")
    opportunity_sha256 = str(selected.get("content_sha256") or "")
    opportunity_set_sha256 = str(decision.get("opportunity_set_sha256") or "")
    execution_id = campaign_action_execution_id(
        action_id=str(selected.get("action_id") or ""),
        input_revision=input_revision,
        opportunity_sha256=opportunity_sha256,
        opportunity_set_sha256=opportunity_set_sha256,
    )
    identity = execution_id.removeprefix("campaign-action:")
    kind = CampaignActionKind(str(selected.get("kind") or ""))
    resource_class = str(selected.get("resource_class") or "")
    expected_resources = compile_action_resource_estimate(
        kind=kind,
        resource_class=resource_class,
    )
    return CampaignAction(
        execution_id=execution_id,
        action_id=str(selected.get("action_id") or ""),
        kind=kind,
        deficit_id=str(selected.get("deficit_id") or ""),
        input_revision=int(input_revision),
        opportunity_sha256=opportunity_sha256,
        opportunity_set_sha256=opportunity_set_sha256,
        subject_ids=tuple(
            sorted(str(value) for value in selected.get("subject_ids") or [] if str(value))
        ),
        route_family_ids=tuple(
            sorted(
                str(value)
                for value in selected.get("route_family_ids") or []
                if str(value)
            )
        ),
        producer=str(selected.get("producer") or ""),
        resource_class=resource_class,
        estimate=compile_action_estimate(
            selected,
            expected_resources=expected_resources,
        ),
        expected_resources=expected_resources,
        task_id=f"campaign-action:{identity[:24]}",
        idempotency_key=f"campaign-action:{identity}",
        reason=str(selected.get("reason") or ""),
        metadata={
            **dict(selected.get("metadata") or {}),
            "schedule_score": float(selected.get("schedule_score") or 0.0),
            "schedule_components": dict(
                selected.get("schedule_components") or {}
            ),
            "scheduler_policy": str(
                decision.get("scheduler_policy") or "adaptive"
            ),
            "round_robin_cursor": int(
                decision.get("round_robin_cursor") or 0
            ),
        },
    )


def campaign_action_execution_id(
    *,
    action_id: str,
    input_revision: int,
    opportunity_sha256: str,
    opportunity_set_sha256: str,
) -> str:
    """Return the semantic execution identity for one scheduled Action.

    Scheduler diagnostics such as the round-robin cursor, policy label, or
    score decomposition intentionally do not participate. Rebinding the same
    opportunity at the same canonical graph revision must therefore address
    the same durable execution receipt.
    """

    identity = _digest(
        {
            "action_id": str(action_id or ""),
            "input_revision": int(input_revision),
            "opportunity_sha256": str(opportunity_sha256 or ""),
            "opportunity_set_sha256": str(opportunity_set_sha256 or ""),
        }
    )
    return f"campaign-action:{identity}"


def action_task_kind(resource_class: str) -> str:
    """Return the RunKernel task class owned by the Action wrapper."""

    return {
        "program": "program",
        "experiment": "experiment",
    }.get(str(resource_class or ""), "other")


def compile_action_resource_estimate(
    *,
    kind: CampaignActionKind | str,
    resource_class: str,
) -> dict[str, Any]:
    """Declare a target-blind class estimate before Action reservation."""

    normalized_kind = (
        kind if isinstance(kind, CampaignActionKind) else CampaignActionKind(str(kind))
    )
    normalized_resource = str(resource_class or "")
    wrapper_kind = action_task_kind(normalized_resource)
    delegated_task_counts = {
        value: 1
        for value in (normalized_resource,)
        if value in {"model", "evidence", "stock", "validation"}
    }
    estimated_task_counts = {wrapper_kind: 1}
    for task_kind, count in delegated_task_counts.items():
        estimated_task_counts[task_kind] = (
            int(estimated_task_counts.get(task_kind) or 0) + count
        )
    native_units = (
        1
        if normalized_resource
        in {"native_search_target", "native_search_frontier"}
        else 0
    )
    result = {
        "schema_version": CAMPAIGN_ACTION_RESOURCE_ESTIMATE_SCHEMA,
        "action_kind": normalized_kind.value,
        "resource_class": normalized_resource,
        "wrapper": {
            "task_kind": wrapper_kind,
            "task_count": 1,
        },
        "estimated": {
            "task_counts": dict(sorted(estimated_task_counts.items())),
            "total_tasks": sum(estimated_task_counts.values()),
            "native_search_units": native_units,
            "model_invocations": int(normalized_resource == "model"),
            "visual_invocations": 0,
        },
        "unknown_dimensions": (
            ["input_tokens", "output_tokens", "model_wall_time_s"]
            if normalized_resource == "model"
            else []
        ),
        "basis": "target_blind_action_resource_class",
        "semantics": {
            "estimate_is_not_scientific_authority": True,
            "handler_children_may_create_a_measured_variance": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def compile_action_estimate(
    opportunity: Mapping[str, Any],
    *,
    expected_resources: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the complete pre-reservation estimate without target labels."""

    row = dict(opportunity)
    result = {
        "schema_version": ACTION_ESTIMATE_SCHEMA,
        "action_id": str(row.get("action_id") or ""),
        "action_kind": str(row.get("kind") or ""),
        "success_probability": {
            "low": float(row.get("success_probability_low") or 0.0),
            "high": float(
                row.get("success_probability_high")
                if row.get("success_probability_high") is not None
                else 1.0
            ),
            "assessed": row.get("success_probability_assessed") is True,
        },
        "expected_gain": {
            "route": float(row.get("expected_route_gain") or 0.0),
            "proof": float(row.get("expected_proof_gain") or 0.0),
            "diversity": float(row.get("expected_diversity_gain") or 0.0),
            "dependency_unblock_count": max(
                0,
                int(row.get("expected_dependency_unblock_count") or 0),
            ),
            "novelty": float(row.get("expected_novelty_gain") or 0.0),
        },
        "cost": {
            "penalty": float(row.get("cost_penalty") or 0.0),
            "resource_class": str(row.get("resource_class") or ""),
            "expected_resources": dict(expected_resources),
        },
        "uncertainty": {
            **dict(row.get("uncertainty") or {}),
            "success_probability": (
                "assessed"
                if row.get("success_probability_assessed") is True
                else "unassessed_full_interval"
            ),
            "resource_unknown_dimensions": list(
                expected_resources.get("unknown_dimensions") or []
            ),
        },
        "basis": "canonical_deficit_frontier_score",
        "semantics": {
            "target_labels_are_not_inputs": True,
            "estimate_is_not_execution_result": True,
            "unassessed_probability_is_not_imputed": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def legacy_campaign_action_sha256(action: CampaignAction) -> str:
    """Recompute the exact pre-resource-contract Action identity."""

    row = action.to_dict()
    row.pop("content_sha256", None)
    row.pop("estimate", None)
    row.pop("expected_resources", None)
    row["schema_version"] = LEGACY_CAMPAIGN_ACTION_SCHEMA
    return _digest(row)


def compile_action_opportunities(
    frontier: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile executable choices without reading task labels or datasets."""

    if isinstance(frontier, Mapping):
        raw_items = frontier.get("items") or frontier.get("deficits") or []
        frontier_sha256 = str(frontier.get("content_sha256") or "")
    else:
        raw_items = frontier
        frontier_sha256 = ""
    opportunities: list[CampaignActionOpportunity] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        kind = str(row.get("kind") or "")
        mappings = _action_mappings(row)
        for action_kind, producer, resource_class in mappings:
            score = dict(row.get("score") or {})
            probability = _success_probability_interval(score)
            action_contract = (
                MATERIALIZATION_ACTION_CONTRACT
                if action_kind is CampaignActionKind.MATERIALIZE
                else ""
            )
            identity = _digest(
                {
                    "deficit_id": str(row.get("deficit_id") or ""),
                    "kind": action_kind.value,
                    "producer": producer,
                    # A materialization contract change must create a fresh
                    # deterministic execution identity. Otherwise a resumed
                    # run can replay an old completed-wrapper/rejected-graph
                    # receipt forever and never reach canonical ingestion.
                    **(
                        {"action_contract": action_contract}
                        if action_contract
                        else {}
                    ),
                }
            )
            opportunities.append(
                CampaignActionOpportunity(
                    action_id=f"action:{action_kind.value}:{identity[:24]}",
                    kind=action_kind,
                    deficit_id=str(row.get("deficit_id") or ""),
                    subject_ids=tuple(
                        sorted(
                            str(value)
                            for value in row.get("entity_ids") or []
                            if str(value)
                        )
                    ),
                    route_family_ids=tuple(
                        sorted(
                            str(value)
                            for value in row.get("route_family_ids") or []
                            if str(value)
                        )
                    ),
                    dependency_ids=tuple(
                        sorted(
                            str(value)
                            for value in row.get("dependency_ids") or []
                            if str(value)
                        )
                    ),
                    producer=producer,
                    resource_class=resource_class,
                    deterministic=row.get("deterministic") is True,
                    model_allowed=row.get("model_allowed") is True,
                    base_priority=float(row.get("priority") or score.get("priority") or 0.0),
                    expected_route_gain=float(
                        score.get("expected_portfolio_gain") or 0.0
                    ),
                    expected_proof_gain=max(
                        float(score.get("evidence_gain") or 0.0),
                        float(score.get("distance_to_closure") or 0.0),
                    ),
                    expected_diversity_gain=float(
                        score.get("route_diversity_gain") or 0.0
                    ),
                    expected_dependency_unblock_count=max(
                        0,
                        int(
                            score.get("dependency_unblock_count")
                            or score.get("dependency_unblock_gain")
                            or 0
                        ),
                    ),
                    expected_novelty_gain=float(
                        score.get("novelty_gain") or score.get("novelty") or 0.0
                    ),
                    success_probability_low=probability[0],
                    success_probability_high=probability[1],
                    success_probability_assessed=probability[2],
                    cost_penalty=float(score.get("cost_penalty") or 0.0),
                    failure_risk_penalty=float(
                        score.get("failure_risk_penalty") or 0.0
                    ),
                    uncertainty=(
                        dict(score["uncertainty"])
                        if isinstance(score.get("uncertainty"), Mapping)
                        else {}
                    ),
                    reason=str(row.get("reason") or ""),
                    metadata={
                        **dict(row.get("metadata") or {}),
                        "frontier_kind": kind,
                        "frontier_object_id": str(row.get("object_id") or ""),
                        **(
                            {"action_contract": action_contract}
                            if action_contract
                            else {}
                        ),
                    },
                )
            )
    rows = [
        value.to_dict()
        for value in sorted(
            opportunities,
            key=lambda value: (
                -value.base_priority,
                value.kind.value,
                value.action_id,
            ),
        )
    ]
    result = {
        "schema_version": CAMPAIGN_ACTION_SET_SCHEMA,
        "frontier_sha256": frontier_sha256,
        "action_count": len(rows),
        "actions": rows,
        "semantics": {
            "single_canonical_frontier": True,
            "task_labels_are_not_inputs": True,
            "opportunities_grant_no_scientific_authority": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _success_probability_interval(
    score: Mapping[str, Any],
) -> tuple[float, float, bool]:
    interval = score.get("success_probability_interval")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        low = _probability(interval[0])
        high = _probability(interval[1])
        if low is not None and high is not None and low <= high:
            return low, high, True
    low = _probability(score.get("success_probability_low"))
    high = _probability(score.get("success_probability_high"))
    if low is not None and high is not None and low <= high:
        return low, high, True
    point = _probability(score.get("success_probability"))
    if point is not None:
        return point, point, True
    return 0.0, 1.0, False


def _probability(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0.0 <= result <= 1.0 else None


def _action_mappings(
    deficit: Mapping[str, Any],
) -> tuple[tuple[CampaignActionKind, str, str], ...]:
    kind = str(deficit.get("kind") or "")
    reason = str(deficit.get("reason") or "")
    metadata = dict(deficit.get("metadata") or {})
    if kind == "materialization":
        return ((CampaignActionKind.MATERIALIZE, "host_worker", "deterministic"),)
    if kind == "architecture":
        return (
            (
                CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE,
                "codex_global_director",
                "model",
            ),
        )
    if kind == "replan":
        return (
            (
                CampaignActionKind.CODEX_REPLAN,
                "codex_global_director",
                "model",
            ),
        )
    if kind == "program_review":
        return ((CampaignActionKind.PROGRAM_REVIEW, "program_host", "program"),)
    if kind == "program_discovery":
        return (
            (CampaignActionKind.PROGRAM_DISCOVER, "program_discovery", "program"),
        )
    if kind == "program_admission":
        return ((CampaignActionKind.PROGRAM_ADMIT, "program_host", "program"),)
    if kind == "program_validation":
        return (
            (CampaignActionKind.PROGRAM_VALIDATE, "program_validator", "program"),
        )
    if kind == "experiment_feedback":
        return (
            (
                CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST,
                "experimental_claim_host",
                "experiment",
            ),
        )
    if kind == "validation":
        return ((CampaignActionKind.REACTION_VALIDATE, "host_validator", "validation"),)
    if kind == "condition":
        return ((CampaignActionKind.CONDITION_ENRICH, "condition_provider", "condition"),)
    if kind == "stock":
        return ((CampaignActionKind.STOCK_AUDIT, "stock_oracle", "stock"),)
    if kind == "conflict":
        return ((CampaignActionKind.RESOLVE_CONFLICT, "host_evidence_gate", "evidence"),)
    if kind == "evidence":
        if "binding" in reason or "independent_source_support" in reason:
            return ((CampaignActionKind.BIND_EVIDENCE, "host_evidence_gate", "evidence"),)
        return ((CampaignActionKind.ACQUIRE_EVIDENCE, "evidence_connector", "evidence"),)
    if kind == "expansion":
        providers = {
            str(value).strip().casefold()
            for value in metadata.get("provider_preferences") or []
            if str(value).strip()
        }
        if not providers:
            providers = {"chemenzy"}
        values: list[tuple[CampaignActionKind, str, str]] = []
        if "chemenzy" in providers:
            chemenzy_kind = (
                CampaignActionKind.CHEMENZY_TARGET_EXPAND
                if metadata.get("target_level_native_search") is True
                else CampaignActionKind.CHEMENZY_FRONTIER_EXPAND
            )
            values.append(
                (
                    chemenzy_kind,
                    "chemenzy",
                    (
                        "native_search_target"
                        if chemenzy_kind
                        == CampaignActionKind.CHEMENZY_TARGET_EXPAND
                        else "native_search_frontier"
                    ),
                )
            )
        if "codex" in providers or "codex_global_director" in providers:
            values.append(
                (CampaignActionKind.CODEX_REPLAN, "codex_global_director", "model")
            )
        return tuple(values)
    if kind == "diversity":
        values: list[tuple[CampaignActionKind, str, str]] = []
        if str(metadata.get("frontier_smiles") or ""):
            values.append(
                (
                    CampaignActionKind.CHEMENZY_FRONTIER_EXPAND,
                    "chemenzy",
                    "native_search_frontier",
                )
            )
        return tuple(values)
    if kind == "route_closure":
        return ((CampaignActionKind.RECOMPUTE_ROUTE, "host_projection", "deterministic"),)
    return ()


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ACTION_ESTIMATE_SCHEMA",
    "ACTION_RESULT_SCHEMA",
    "ActionResult",
    "CAMPAIGN_ACTION_RESOURCE_ESTIMATE_SCHEMA",
    "CampaignAction",
    "CampaignActionKind",
    "CampaignActionOpportunity",
    "action_task_kind",
    "bind_scheduled_action",
    "compile_action_resource_estimate",
    "compile_action_estimate",
    "compile_action_opportunities",
    "legacy_campaign_action_sha256",
]
