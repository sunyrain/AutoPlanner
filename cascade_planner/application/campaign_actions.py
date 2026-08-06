"""Target-blind action opportunities derived from the one deficit frontier."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping


CAMPAIGN_ACTION_OPPORTUNITY_SCHEMA = "campaign_action_opportunity.v1"
CAMPAIGN_ACTION_SET_SCHEMA = "campaign_action_opportunity_set.v1"
CAMPAIGN_ACTION_SCHEMA = "campaign_action.v1"


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
    cost_penalty: float
    failure_risk_penalty: float
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
            "semantics": {
                "revision_bound": True,
                "wrapper_task_delegates_resource_accounting_to_handler": True,
                "grants_no_scientific_authority": True,
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
    identity = _digest(
        {
            "action_id": str(selected.get("action_id") or ""),
            "input_revision": int(input_revision),
            "opportunity_sha256": opportunity_sha256,
            "opportunity_set_sha256": opportunity_set_sha256,
        }
    )
    execution_id = f"campaign-action:{identity}"
    return CampaignAction(
        execution_id=execution_id,
        action_id=str(selected.get("action_id") or ""),
        kind=CampaignActionKind(str(selected.get("kind") or "")),
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
        resource_class=str(selected.get("resource_class") or ""),
        task_id=f"campaign-action:{identity[:24]}",
        idempotency_key=f"campaign-action:{identity}",
        reason=str(selected.get("reason") or ""),
        metadata={
            **dict(selected.get("metadata") or {}),
            "schedule_score": float(selected.get("schedule_score") or 0.0),
            "schedule_components": dict(
                selected.get("schedule_components") or {}
            ),
        },
    )


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
            identity = _digest(
                {
                    "deficit_id": str(row.get("deficit_id") or ""),
                    "kind": action_kind.value,
                    "producer": producer,
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
                    cost_penalty=float(score.get("cost_penalty") or 0.0),
                    failure_risk_penalty=float(
                        score.get("failure_risk_penalty") or 0.0
                    ),
                    reason=str(row.get("reason") or ""),
                    metadata={
                        **dict(row.get("metadata") or {}),
                        "frontier_kind": kind,
                        "frontier_object_id": str(row.get("object_id") or ""),
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
            (CampaignActionKind.PROGRAM_VALIDATE, "program_validator", "validation"),
        )
    if kind == "experiment_feedback":
        return (
            (
                CampaignActionKind.EXPERIMENT_FEEDBACK_INGEST,
                "experimental_claim_host",
                "validation",
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
    "CampaignAction",
    "CampaignActionKind",
    "CampaignActionOpportunity",
    "bind_scheduled_action",
    "compile_action_opportunities",
]
