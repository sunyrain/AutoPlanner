"""Run-wide acceptance and cost contracts for retrosynthesis.

The legacy controller grew several independent counters: blackboard rounds,
visual calls, campaign attempts, and accepted frontier expansions.  None of
those counters can bound the complete cost of one target.  This module keeps
the operator-owned acceptance target and the run-wide spending envelope in a
small, replayable contract that every scheduler can read.

The ledger is deliberately an accounting authority, not a chemistry
authority.  It can stop new work, but it can never mark an edge or route
solved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping


RETROSYNTHESIS_ACCEPTANCE_SPEC_SCHEMA = "retrosynthesis_acceptance_spec.v1"
RETROSYNTHESIS_RUN_BUDGET_SCHEMA = "retrosynthesis_run_budget.v1"
RETROSYNTHESIS_COST_LEDGER_SCHEMA = "retrosynthesis_cost_ledger.v1"


@dataclass(frozen=True)
class RetrosynthesisAcceptanceSpec:
    """One immutable definition of scientific completion for a target."""

    minimum_complete_routes: int = 2
    minimum_edge_proof_level: int = 3
    require_all_selected_leaves_stock_closed: bool = True
    stock_boundary: str = "procurement"
    minimum_independent_source_groups: int = 2
    require_distinct_edge_sets: bool = True
    schema_version: str = RETROSYNTHESIS_ACCEPTANCE_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.minimum_complete_routes < 1:
            raise ValueError("minimum_complete_routes must be positive")
        if not 2 <= self.minimum_edge_proof_level <= 4:
            raise ValueError("minimum_edge_proof_level must be in [2, 4]")
        if self.stock_boundary not in {
            "benchmark_search",
            "procurement",
            "in_house",
        }:
            raise ValueError("unsupported stock_boundary")
        if self.minimum_independent_source_groups < 1:
            raise ValueError(
                "minimum_independent_source_groups must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True)
class RetrosynthesisRunBudget:
    """Hard host envelope shared by every model-backed worker for one target.

    Token limits are enforced between calls from observed worker usage.  The
    prompt/context byte caps and small call count prevent one automatically
    constructed request from silently recreating an unbounded campaign.
    """

    max_model_invocations: int = 3
    max_total_input_tokens: int = 60_000
    max_total_output_tokens: int = 12_000
    max_total_wall_time_s: float = 1_800.0
    max_visual_invocations: int = 1
    max_accepted_expansions: int = 8
    max_attempt_runs: int = 12
    max_native_search_invocations: int | None = None
    min_target_native_search_invocations: int | None = None
    max_frontier_native_search_invocations: int | None = None
    allow_frontier_native_search_borrowing: bool = True
    max_prompt_context_bytes: int = 96_000
    automatic_budget_extension: bool = False
    schema_version: str = RETROSYNTHESIS_RUN_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        max_native = (
            self.max_attempt_runs
            if self.max_native_search_invocations is None
            else int(self.max_native_search_invocations)
        )
        min_target_native = (
            (1 if max_native > 0 else 0)
            if self.min_target_native_search_invocations is None
            else int(self.min_target_native_search_invocations)
        )
        max_frontier_native = (
            max(0, max_native - min_target_native)
            if self.max_frontier_native_search_invocations is None
            else int(self.max_frontier_native_search_invocations)
        )
        object.__setattr__(self, "max_native_search_invocations", max_native)
        object.__setattr__(
            self,
            "min_target_native_search_invocations",
            min_target_native,
        )
        object.__setattr__(
            self,
            "max_frontier_native_search_invocations",
            max_frontier_native,
        )
        integer_limits = {
            "max_model_invocations": self.max_model_invocations,
            "max_total_input_tokens": self.max_total_input_tokens,
            "max_total_output_tokens": self.max_total_output_tokens,
            "max_visual_invocations": self.max_visual_invocations,
            "max_accepted_expansions": self.max_accepted_expansions,
            "max_attempt_runs": self.max_attempt_runs,
            "max_native_search_invocations": max_native,
            "min_target_native_search_invocations": min_target_native,
            "max_frontier_native_search_invocations": max_frontier_native,
            "max_prompt_context_bytes": self.max_prompt_context_bytes,
        }
        if any(value < 0 for value in integer_limits.values()):
            raise ValueError("retrosynthesis run budget limits cannot be negative")
        if min_target_native > max_native:
            raise ValueError(
                "min_target_native_search_invocations cannot exceed "
                "max_native_search_invocations"
            )
        if max_frontier_native > max_native:
            raise ValueError(
                "max_frontier_native_search_invocations cannot exceed "
                "max_native_search_invocations"
            )
        if self.max_total_wall_time_s < 0:
            raise ValueError("max_total_wall_time_s cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True)
class ModelCostEvent:
    """One settled host-observed model invocation."""

    invocation_id: str
    worker_kind: str
    elapsed_s: float
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    accepted_expansions: int = 0
    attempt_runs: int = 0
    visual: bool = False
    status: str = "completed"
    usage_observed: bool = True

    def __post_init__(self) -> None:
        if not self.invocation_id or not self.worker_kind:
            raise ValueError("model cost event identity is required")
        if self.elapsed_s < 0:
            raise ValueError("model cost event elapsed_s cannot be negative")
        if any(
            value < 0
            for value in (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_output_tokens,
                self.accepted_expansions,
                self.attempt_runs,
            )
        ):
            raise ValueError("model cost event token counts cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrosynthesisCostLedger:
    """Replayable cost state with fail-closed admission for new model work."""

    budget: RetrosynthesisRunBudget = field(
        default_factory=RetrosynthesisRunBudget
    )
    events: list[ModelCostEvent] = field(default_factory=list)
    reserved_invocation_ids: list[str] = field(default_factory=list)
    schema_version: str = RETROSYNTHESIS_COST_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        event_ids = [event.invocation_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate model cost event invocation_id")
        reserved = [str(item) for item in self.reserved_invocation_ids]
        if len(reserved) != len(set(reserved)):
            raise ValueError("duplicate reserved invocation_id")
        if set(event_ids) & set(reserved):
            raise ValueError("settled invocation cannot remain reserved")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any] | None,
        *,
        budget: RetrosynthesisRunBudget | None = None,
    ) -> "RetrosynthesisCostLedger":
        row = dict(value or {})
        budget_row = dict(row.get("budget") or {})
        budget_row.pop("content_sha256", None)
        budget_row.pop("schema_version", None)
        resolved_budget = budget or RetrosynthesisRunBudget(**budget_row)
        events = [
            ModelCostEvent(**dict(item))
            for item in row.get("events") or []
            if isinstance(item, Mapping)
        ]
        return cls(
            budget=resolved_budget,
            events=events,
            reserved_invocation_ids=[
                str(item)
                for item in row.get("reserved_invocation_ids") or []
                if str(item or "").strip()
            ],
        )

    def gate_reasons(
        self,
        *,
        visual: bool = False,
        prompt_context_bytes: int = 0,
    ) -> list[str]:
        totals = self.totals()
        reasons: list[str] = []
        pending_count = len(self.reserved_invocation_ids)
        if (
            totals["model_invocations"] + pending_count
            >= self.budget.max_model_invocations
        ):
            reasons.append("run_model_invocation_budget_exhausted")
        if totals["input_tokens"] >= self.budget.max_total_input_tokens:
            reasons.append("run_input_token_budget_exhausted")
        if totals["output_tokens"] >= self.budget.max_total_output_tokens:
            reasons.append("run_output_token_budget_exhausted")
        if totals["wall_time_s"] >= self.budget.max_total_wall_time_s:
            reasons.append("run_model_wall_time_budget_exhausted")
        if (
            totals["accepted_expansions"]
            >= self.budget.max_accepted_expansions
        ):
            reasons.append("run_accepted_expansion_budget_exhausted")
        if totals["attempt_runs"] >= self.budget.max_attempt_runs:
            reasons.append("run_attempt_budget_exhausted")
        if visual and (
            totals["visual_invocations"]
            + sum(
                1
                for invocation_id in self.reserved_invocation_ids
                if invocation_id.startswith("visual:")
            )
            >= self.budget.max_visual_invocations
        ):
            reasons.append("run_visual_invocation_budget_exhausted")
        if prompt_context_bytes < 0:
            reasons.append("prompt_context_bytes_invalid")
        elif prompt_context_bytes > self.budget.max_prompt_context_bytes:
            reasons.append("prompt_context_byte_budget_exceeded")
        if any(event.usage_observed is False for event in self.events):
            reasons.append("prior_model_usage_unobserved")
        return sorted(set(reasons))

    def reserve(
        self,
        invocation_id: str,
        *,
        visual: bool = False,
        prompt_context_bytes: int = 0,
    ) -> None:
        invocation = str(invocation_id or "").strip()
        if not invocation:
            raise ValueError("invocation_id is required")
        if visual and not invocation.startswith("visual:"):
            raise ValueError("visual invocation_id must start with visual:")
        if invocation in self.reserved_invocation_ids or any(
            event.invocation_id == invocation for event in self.events
        ):
            raise ValueError("model invocation_id already exists")
        reasons = self.gate_reasons(
            visual=visual,
            prompt_context_bytes=prompt_context_bytes,
        )
        if reasons:
            raise RuntimeError(";".join(reasons))
        self.reserved_invocation_ids.append(invocation)

    def settle(self, event: ModelCostEvent) -> None:
        if event.invocation_id not in self.reserved_invocation_ids:
            raise ValueError("model invocation was not reserved")
        self.reserved_invocation_ids = [
            item
            for item in self.reserved_invocation_ids
            if item != event.invocation_id
        ]
        self.events.append(event)

    def abandon(self, invocation_id: str) -> None:
        invocation = str(invocation_id or "")
        self.reserved_invocation_ids = [
            item for item in self.reserved_invocation_ids if item != invocation
        ]

    def totals(self) -> dict[str, int | float]:
        return {
            "model_invocations": len(self.events),
            "visual_invocations": sum(1 for event in self.events if event.visual),
            "input_tokens": sum(event.input_tokens for event in self.events),
            "cached_input_tokens": sum(
                event.cached_input_tokens for event in self.events
            ),
            "output_tokens": sum(event.output_tokens for event in self.events),
            "reasoning_output_tokens": sum(
                event.reasoning_output_tokens for event in self.events
            ),
            "accepted_expansions": sum(
                event.accepted_expansions for event in self.events
            ),
            "attempt_runs": sum(event.attempt_runs for event in self.events),
            "wall_time_s": round(
                sum(event.elapsed_s for event in self.events),
                6,
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": self.schema_version,
            "budget": self.budget.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "reserved_invocation_ids": sorted(self.reserved_invocation_ids),
            "totals": self.totals(),
            "gate_reasons": self.gate_reasons(),
            "semantics": {
                "run_wide_across_all_model_workers": True,
                "budget_cannot_grant_chemistry_authority": True,
                "automatic_budget_extension": (
                    self.budget.automatic_budget_extension
                ),
                "unobserved_usage_fails_closed": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row


def model_cost_event_from_worker_record(
    record: Any,
    *,
    invocation_id: str,
    worker_kind: str,
    visual: bool = False,
) -> ModelCostEvent:
    """Normalize a WorkerRunRecord-like value without trusting model claims."""

    usage = dict(getattr(record, "usage", {}) or {})
    observed_fields = {
        key
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
        if key in usage
    }

    def token_count(key: str) -> int:
        try:
            return max(0, int(usage.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    try:
        elapsed_s = max(0.0, float(getattr(record, "elapsed_s", 0.0) or 0.0))
    except (TypeError, ValueError):
        elapsed_s = 0.0
    return ModelCostEvent(
        invocation_id=invocation_id,
        worker_kind=worker_kind,
        elapsed_s=elapsed_s,
        input_tokens=token_count("input_tokens"),
        cached_input_tokens=token_count("cached_input_tokens"),
        output_tokens=token_count("output_tokens"),
        reasoning_output_tokens=token_count("reasoning_output_tokens"),
        visual=visual,
        status=str(getattr(record, "status", "") or "unknown"),
        usage_observed=bool(
            {"input_tokens", "output_tokens"}.issubset(observed_fields)
        ),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ModelCostEvent",
    "RETROSYNTHESIS_ACCEPTANCE_SPEC_SCHEMA",
    "RETROSYNTHESIS_COST_LEDGER_SCHEMA",
    "RETROSYNTHESIS_RUN_BUDGET_SCHEMA",
    "RetrosynthesisAcceptanceSpec",
    "RetrosynthesisCostLedger",
    "RetrosynthesisRunBudget",
    "model_cost_event_from_worker_record",
]
