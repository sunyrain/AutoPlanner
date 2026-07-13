"""Bounded Codex director for global, multi-route campaign reasoning.

The director sees the whole campaign and may redesign route families, shared
intermediates, evidence acquisition, and fallback strategy in one response.
Its output is hypothesis-only: deterministic host gates decide which proposals
may enter the canonical frontier, and later workers/verifiers own every proof,
stock, and completion transition.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

from rdkit import Chem

from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerTask,
    run_codex_worker,
)
from cascade_planner.application.campaign_context import CampaignContext
from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.runtime import (
    AgentResult,
    AgentSpec,
    AgentState,
    ArtifactReferenceError,
    Budget,
)


GLOBAL_CAMPAIGN_PLAN_SCHEMA = "global_campaign_plan.v1"
GLOBAL_CAMPAIGN_DIRECTOR_CONFIG_SCHEMA = "global_campaign_director_config.v1"
GLOBAL_CAMPAIGN_DIRECTOR_OUTCOME_SCHEMA = "global_campaign_director_outcome.v1"
DIRECTOR_PROPOSAL_AUDIT_SCHEMA = "global_campaign_proposal_audit.v1"
DIRECTOR_MODES = frozenset(
    {"initial_architecture", "event_replan", "final_portfolio_synthesis"}
)
DIRECTOR_DISPOSITIONS = frozenset(
    {"accepted", "rejected", "superseded", "ignored"}
)
MATERIAL_REPLAN_EVENTS = frozenset(
    {
        "critical_edge_rejected",
        "exact_rows_added",
        "material_evidence_added",
        "new_route_family",
        "portfolio_stagnation",
        "shared_bottleneck_changed",
        "source_conflict_added",
        "stock_records_added",
        "stock_boundary_changed",
    }
)
_REQUIRED_PLAN_SECTIONS = (
    "route_families",
    "multi_step_skeletons",
    "strategic_disconnections",
    "shared_intermediates",
    "critical_unknowns",
    "source_plan",
    "fallback_strategies",
    "frontier_priorities",
    "pivot_conditions",
    "stop_conditions",
    "portfolio_rationale",
)
_FORBIDDEN_AUTHORITY_KEYS = {
    "closed",
    "completion_granted",
    "is_solved",
    "proof_granted",
    "route_solved",
    "solved",
    "stock_closed",
    "validated",
}


class GlobalCampaignDirectorError(RuntimeError):
    """Base director error."""


class GlobalCampaignPlanValidationError(GlobalCampaignDirectorError):
    """Raised when a child output violates the global plan contract."""


@dataclass(frozen=True, slots=True)
class GlobalCampaignPlan:
    plan_id: str
    run_id: str
    mode: str
    context_sha256: str
    graph_revision: int
    route_families: tuple[Mapping[str, Any], ...]
    multi_step_skeletons: tuple[Mapping[str, Any], ...]
    strategic_disconnections: tuple[Mapping[str, Any], ...]
    shared_intermediates: tuple[Mapping[str, Any], ...]
    critical_unknowns: tuple[Mapping[str, Any], ...]
    source_plan: tuple[Mapping[str, Any], ...]
    fallback_strategies: tuple[Mapping[str, Any], ...]
    frontier_priorities: tuple[Mapping[str, Any], ...]
    pivot_conditions: tuple[Mapping[str, Any], ...]
    stop_conditions: tuple[Mapping[str, Any], ...]
    portfolio_rationale: str
    limitations: tuple[str, ...] = ()
    content_sha256: str = ""
    schema_version: str = GLOBAL_CAMPAIGN_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if not self.plan_id or not self.run_id or not self.context_sha256:
            raise GlobalCampaignPlanValidationError("director_plan_identity_missing")
        if self.mode not in DIRECTOR_MODES:
            raise GlobalCampaignPlanValidationError("director_plan_mode_invalid")
        if self.graph_revision < 0:
            raise GlobalCampaignPlanValidationError("director_graph_revision_invalid")
        row = self._body()
        digest = _digest(row)
        if self.content_sha256 and self.content_sha256 != digest:
            raise GlobalCampaignPlanValidationError("director_plan_digest_invalid")
        object.__setattr__(self, "content_sha256", digest)

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "mode": self.mode,
            "context_sha256": self.context_sha256,
            "graph_revision": self.graph_revision,
            **{
                key: [dict(row) for row in getattr(self, key)]
                for key in _REQUIRED_PLAN_SECTIONS
                if key != "portfolio_rationale"
            },
            "portfolio_rationale": self.portfolio_rationale,
            "limitations": list(self.limitations),
            "semantics": {
                "hypothesis_only": True,
                "grants_no_reaction_proof": True,
                "grants_no_stock_authority": True,
                "grants_no_route_completion": True,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GlobalCampaignPlan":
        row = dict(value)
        if "payload" in row and isinstance(row.get("payload"), Mapping):
            row = dict(row["payload"])
        if row.get("schema_version") != GLOBAL_CAMPAIGN_PLAN_SCHEMA:
            raise GlobalCampaignPlanValidationError("director_plan_schema_invalid")
        missing = [key for key in _REQUIRED_PLAN_SECTIONS if key not in row]
        if missing:
            raise GlobalCampaignPlanValidationError(
                "director_plan_sections_missing:" + ",".join(missing)
            )
        list_fields: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for key in _REQUIRED_PLAN_SECTIONS:
            if key == "portfolio_rationale":
                continue
            raw = row.get(key)
            if not isinstance(raw, list) or any(
                not isinstance(item, Mapping) for item in raw
            ):
                raise GlobalCampaignPlanValidationError(
                    f"director_plan_section_not_object_list:{key}"
                )
            list_fields[key] = tuple(dict(item) for item in raw)
        return cls(
            plan_id=str(row.get("plan_id") or ""),
            run_id=str(row.get("run_id") or ""),
            mode=str(row.get("mode") or ""),
            context_sha256=str(row.get("context_sha256") or ""),
            graph_revision=int(row.get("graph_revision") or 0),
            portfolio_rationale=str(row.get("portfolio_rationale") or ""),
            limitations=tuple(str(item) for item in row.get("limitations") or ()),
            content_sha256=str(row.get("content_sha256") or ""),
            **list_fields,
        )


@dataclass(frozen=True, slots=True)
class DirectorConfig:
    max_route_families: int = 6
    max_skeletons: int = 8
    max_steps_per_skeleton: int = 12
    max_output_bytes: int = 240_000
    max_output_tokens: int = 8_000
    max_wall_time_s: float = 600.0
    max_tool_calls: int = 12
    max_initial_architecture_calls: int = 1
    max_event_replan_calls: int = 2
    max_final_portfolio_synthesis_calls: int = 1
    model: str = ""
    reasoning_effort: str = "low"
    schema_version: str = GLOBAL_CAMPAIGN_DIRECTOR_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for value in (
            self.max_route_families,
            self.max_skeletons,
            self.max_steps_per_skeleton,
            self.max_output_bytes,
            self.max_output_tokens,
            self.max_tool_calls,
            self.max_initial_architecture_calls,
            self.max_event_replan_calls,
            self.max_final_portfolio_synthesis_calls,
        ):
            if int(value) <= 0:
                raise ValueError("director integer limits must be positive")
        if not math.isfinite(self.max_wall_time_s) or self.max_wall_time_s <= 0:
            raise ValueError("director max_wall_time_s must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["content_sha256"] = _digest(row)
        return row


@dataclass(frozen=True, slots=True)
class DirectorOutcome:
    status: str
    invoked: bool
    cache_hit: bool
    mode: str
    context_sha256: str
    plan: GlobalCampaignPlan | None = None
    proposal_audits: tuple[Mapping[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()
    artifact_sha256: str = ""
    task_id: str = ""
    schema_version: str = GLOBAL_CAMPAIGN_DIRECTOR_OUTCOME_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "invoked": self.invoked,
            "cache_hit": self.cache_hit,
            "mode": self.mode,
            "context_sha256": self.context_sha256,
            "plan": self.plan.to_dict() if self.plan else None,
            "proposal_audits": [dict(row) for row in self.proposal_audits],
            "reasons": list(self.reasons),
            "artifact_sha256": self.artifact_sha256,
            "task_id": self.task_id,
        }


DirectorRunner = Callable[[AgentSpec, CampaignContext, str, DirectorConfig], AgentResult]


class ReplayDirectorRunner:
    """Deterministic, model-free director used by tests and golden replays."""

    model_free = True

    def __init__(self, plans: Mapping[str, Mapping[str, Any]]) -> None:
        self.plans = {str(key): dict(value) for key, value in plans.items()}
        self.calls: list[dict[str, str]] = []

    def __call__(
        self,
        spec: AgentSpec,
        context: CampaignContext,
        mode: str,
        _config: DirectorConfig,
    ) -> AgentResult:
        key = f"{mode}:{context.content_sha256}"
        raw = self.plans.get(key) or self.plans.get(mode)
        self.calls.append(
            {
                "agent_id": spec.agent_id,
                "mode": mode,
                "context_sha256": context.content_sha256,
            }
        )
        if raw is None:
            state = AgentState.FAILED
            output: Any = None
            error = f"replay_plan_missing:{key}"
        else:
            state = AgentState.SUCCEEDED
            output = dict(raw)
            error = ""
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=state,
            output=output,
            error=error,
            usage={
                "model_invocations": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "wall_time_s": 0.0,
            },
            metadata={"backend": "deterministic_replay", "direct_child": True},
        )


class GlobalCampaignDirector:
    """Invoke one direct Codex child through the canonical RunKernel budget."""

    def __init__(
        self,
        kernel: RunKernel,
        *,
        runner: DirectorRunner | None = None,
        config: DirectorConfig | None = None,
    ) -> None:
        self.kernel = kernel
        self.runner = runner or run_codex_cli_director_child
        self.config = config or DirectorConfig()
        self.director_dir = kernel.run_dir / ".autoplanner" / "director"
        self.director_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        context: CampaignContext,
        *,
        mode: str,
        force: bool = False,
    ) -> DirectorOutcome:
        lock_key = _digest(
            {
                "context_sha256": context.content_sha256,
                "mode": mode,
                "config_sha256": self.config.to_dict()["content_sha256"],
            }
        )
        with self._invocation_lock(lock_key):
            return self._run_unlocked(context, mode=mode, force=force)

    def _run_unlocked(
        self,
        context: CampaignContext,
        *,
        mode: str,
        force: bool = False,
    ) -> DirectorOutcome:
        if mode not in DIRECTOR_MODES:
            raise ValueError("unsupported director mode")
        if context.run_id != self.kernel.spec.run_id:
            raise GlobalCampaignDirectorError("director_context_run_mismatch")
        trigger_reasons = director_trigger_reasons(context, mode=mode)
        if mode == "event_replan" and not force and not trigger_reasons:
            return DirectorOutcome(
                status="ignored",
                invoked=False,
                cache_hit=False,
                mode=mode,
                context_sha256=context.content_sha256,
                reasons=("no_material_replan_trigger",),
            )
        cache_key = _digest(
            {
                "context_sha256": context.content_sha256,
                "mode": mode,
                "config_sha256": self.config.to_dict()["content_sha256"],
            }
        )
        task_id = f"director:{cache_key[:24]}"
        cached = self._load_cached(cache_key)
        if cached is not None:
            plan, artifact_sha256, cache_metadata = cached
            if plan.mode != mode:
                raise GlobalCampaignPlanValidationError("director_plan_mode_mismatch")
            audits = tuple(validate_global_campaign_plan(plan, context, self.config))
            if task_id in self.kernel.state.in_flight_tasks:
                self.kernel.settle_task(
                    task_id=task_id,
                    idempotency_key=f"settle:{task_id}",
                    status="completed",
                    output_sha256=artifact_sha256,
                    model_usage=dict(cache_metadata.get("model_usage") or {}),
                    elapsed_s=float(cache_metadata.get("elapsed_s") or 0.0),
                )
            return DirectorOutcome(
                status="accepted",
                invoked=False,
                cache_hit=True,
                mode=mode,
                context_sha256=context.content_sha256,
                plan=plan,
                proposal_audits=audits,
                artifact_sha256=artifact_sha256,
                task_id=task_id,
            )
        if task_id in self.kernel.state.in_flight_tasks:
            return DirectorOutcome(
                status="in_flight",
                invoked=False,
                cache_hit=False,
                mode=mode,
                context_sha256=context.content_sha256,
                reasons=("identical_director_task_requires_recovery",),
                task_id=task_id,
            )
        mode_limit = {
            "initial_architecture": self.config.max_initial_architecture_calls,
            "event_replan": self.config.max_event_replan_calls,
            "final_portfolio_synthesis": (
                self.config.max_final_portfolio_synthesis_calls
            ),
        }[mode]
        prior_mode_calls = self.kernel.count_task_reservations(
            metadata={"director_mode": mode},
        )
        if prior_mode_calls >= mode_limit:
            return DirectorOutcome(
                status="budget_exhausted",
                invoked=False,
                cache_hit=False,
                mode=mode,
                context_sha256=context.content_sha256,
                reasons=("director_mode_call_budget_exhausted",),
                task_id=task_id,
            )
        prompt = director_prompt(context, mode=mode, config=self.config)
        prompt_bytes = len(prompt.encode("utf-8"))
        uses_model = not bool(getattr(self.runner, "model_free", False))
        self.kernel.reserve_task(
            task_id=task_id,
            kind="model" if uses_model else "validation",
            idempotency_key=f"reserve:{task_id}",
            input_revision=context.revision.revision,
            uses_model=uses_model,
            prompt_context_bytes=prompt_bytes,
            metadata={
                "director_mode": mode,
                "context_sha256": context.content_sha256,
                "config_sha256": self.config.to_dict()["content_sha256"],
                "trigger_reasons": trigger_reasons,
            },
        )
        spec = AgentSpec.from_context(
            run_id=self.kernel.spec.run_id,
            agent_id=task_id,
            parent_agent_id=f"run-kernel:{self.kernel.spec.run_id}",
            role="global_campaign_director",
            objective=prompt,
            context=context.to_dict(),
            idempotency_key=f"agent:{cache_key}",
            capabilities=("global_campaign_reasoning", "structured_hypothesis"),
            write_scope=(),
            budget=Budget(
                max_wall_time_s=self.config.max_wall_time_s,
                max_turns=1,
                max_tool_calls=self.config.max_tool_calls,
                max_tokens=self.config.max_output_tokens,
                max_output_bytes=self.config.max_output_bytes,
                max_children=0,
            ),
            context_refs=(context.content_sha256,),
            metadata={
                "mode": mode,
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "no_scientific_authority": True,
            },
        )
        started = time.monotonic()
        result: AgentResult | None = None
        try:
            result = self.runner(spec, context, mode, self.config)
            if not isinstance(result, AgentResult):
                raise GlobalCampaignDirectorError("director_runner_result_invalid")
            usage = normalize_director_usage(result.usage)
            if result.state is not AgentState.SUCCEEDED:
                raise GlobalCampaignDirectorError(
                    "director_child_failed:" + (result.error or result.state.value)
                )
            plan = GlobalCampaignPlan.from_dict(_require_mapping(result.output))
            if plan.mode != mode:
                raise GlobalCampaignPlanValidationError("director_plan_mode_mismatch")
            if len(_canonical_bytes(plan.to_dict())) > self.config.max_output_bytes:
                raise GlobalCampaignPlanValidationError(
                    "director_plan_output_byte_budget_exceeded"
                )
            audits = tuple(validate_global_campaign_plan(plan, context, self.config))
            plan_ref = self.kernel.artifacts.put_json(
                plan.to_dict(),
                logical_name=f"{plan.plan_id}.json",
                producer="autoplanner.global_campaign_director",
            )
            self.kernel.index.index_artifact(
                run_id=self.kernel.spec.run_id,
                artifact_id=f"director_plan:{plan.plan_id}",
                ref=plan_ref,
                revision=context.revision.revision,
                authority_scope="director_hypothesis_only",
            )
            self.record_dispositions(
                plan,
                {
                    str(row["proposal_id"]): (
                        "accepted" if row.get("accepted") is True else "rejected"
                    )
                    for row in audits
                },
                reasons={
                    str(row["proposal_id"]): list(row.get("reasons") or [])
                    for row in audits
                },
            )
            elapsed_s = max(0.0, time.monotonic() - started)
            self.kernel.artifacts.write_pointer(
                self._cache_pointer(cache_key),
                plan_ref,
                metadata={
                    "run_id": self.kernel.spec.run_id,
                    "context_sha256": context.content_sha256,
                    "mode": mode,
                    "authority_scope": "hypothesis_only",
                    "task_id": task_id,
                    "model_usage": usage,
                    "elapsed_s": elapsed_s,
                },
            )
            self.kernel.settle_task(
                task_id=task_id,
                idempotency_key=f"settle:{task_id}",
                status="completed",
                output_sha256=plan_ref.sha256,
                model_usage=usage,
                elapsed_s=elapsed_s,
            )
            return DirectorOutcome(
                status="accepted",
                invoked=True,
                cache_hit=False,
                mode=mode,
                context_sha256=context.content_sha256,
                plan=plan,
                proposal_audits=audits,
                artifact_sha256=plan_ref.sha256,
                task_id=task_id,
            )
        except BaseException as exc:
            usage = normalize_director_usage(result.usage if result else {})
            if uses_model and result is None and usage["model_invocations"] == 0:
                # Fail conservatively: once the backend boundary was entered,
                # an unobserved exception must consume one call slot.
                usage["model_invocations"] = 1
            if task_id in self.kernel.state.in_flight_tasks:
                self.kernel.settle_task(
                    task_id=task_id,
                    idempotency_key=f"settle:{task_id}",
                    status="failed",
                    failure_reasons=(type(exc).__name__, str(exc)),
                    model_usage=usage,
                    elapsed_s=max(0.0, time.monotonic() - started),
                )
            raise

    def record_dispositions(
        self,
        plan: GlobalCampaignPlan,
        dispositions: Mapping[str, str],
        *,
        reasons: Mapping[str, Iterable[str]] | None = None,
    ) -> list[dict[str, Any]]:
        known = set(proposal_ids(plan))
        rows: list[dict[str, Any]] = []
        for proposal_id, raw_disposition in sorted(dispositions.items()):
            disposition = str(raw_disposition)
            if proposal_id not in known:
                raise ValueError(f"unknown director proposal:{proposal_id}")
            if disposition not in DIRECTOR_DISPOSITIONS:
                raise ValueError(f"unsupported director disposition:{disposition}")
            row = {
                "schema_version": DIRECTOR_PROPOSAL_AUDIT_SCHEMA,
                "run_id": self.kernel.spec.run_id,
                "plan_id": plan.plan_id,
                "plan_sha256": plan.content_sha256,
                "proposal_id": proposal_id,
                "disposition": disposition,
                "graph_revision": self.kernel.state.graph_revision,
                "reasons": sorted(
                    {
                        str(item)
                        for item in dict(reasons or {}).get(proposal_id, ())
                        if str(item).strip()
                    }
                ),
                "semantics": {
                    "accepted_means_candidate_only": True,
                    "grants_no_scientific_authority": True,
                },
            }
            row["content_sha256"] = _digest(row)
            rows.append(row)
        if rows:
            ref = self.kernel.artifacts.put_json(
                rows,
                logical_name=f"{plan.plan_id}.proposal-audit.json",
                producer="autoplanner.global_campaign_director",
            )
            self.kernel.artifacts.write_pointer(
                self._audit_pointer(plan.plan_id),
                ref,
                metadata={"run_id": self.kernel.spec.run_id, "plan_id": plan.plan_id},
            )
            self.kernel.index.index_artifact(
                run_id=self.kernel.spec.run_id,
                artifact_id=(
                    f"director_proposal_audit:{plan.plan_id}:{ref.sha256[:12]}"
                ),
                ref=ref,
                revision=self.kernel.state.revision,
                authority_scope="proposal_disposition_only",
            )
        return rows

    def _load_cached(
        self,
        cache_key: str,
    ) -> tuple[GlobalCampaignPlan, str, dict[str, Any]] | None:
        try:
            ref, pointer = self.kernel.artifacts.load_pointer(
                self._cache_pointer(cache_key)
            )
        except ArtifactReferenceError:
            return None
        plan = GlobalCampaignPlan.from_dict(
            _require_mapping(self.kernel.artifacts.read_json(ref))
        )
        return plan, ref.sha256, dict(pointer.get("metadata") or {})

    def _cache_pointer(self, cache_key: str) -> str:
        run_key = hashlib.sha256(self.kernel.spec.run_id.encode("utf-8")).hexdigest()[:24]
        return f"r/{run_key}/d/c/{cache_key[:24]}"

    def _audit_pointer(self, plan_id: str) -> str:
        run_key = hashlib.sha256(self.kernel.spec.run_id.encode("utf-8")).hexdigest()[:24]
        plan_key = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()[:24]
        return f"r/{run_key}/d/a/{plan_key}"

    @contextmanager
    def _invocation_lock(self, cache_key: str) -> Iterator[None]:
        lock_root = self.director_dir / "locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{cache_key[:24]}.lock"
        deadline = time.monotonic() + self.config.max_wall_time_s + 30.0
        stale_after_s = self.config.max_wall_time_s + 120.0
        while True:
            try:
                lock_path.mkdir()
                break
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > stale_after_s:
                        lock_path.rmdir()
                        continue
                except (FileNotFoundError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise GlobalCampaignDirectorError(
                        "director_identical_context_lock_timeout"
                    )
                time.sleep(0.02)
        try:
            yield
        finally:
            try:
                lock_path.rmdir()
            except FileNotFoundError:
                pass


def validate_global_campaign_plan(
    plan: GlobalCampaignPlan,
    context: CampaignContext,
    config: DirectorConfig | None = None,
) -> list[dict[str, Any]]:
    """Validate plan structure and each concrete reaction hypothesis."""

    limits = config or DirectorConfig()
    reasons: list[str] = []
    if plan.run_id != context.run_id:
        reasons.append("plan_run_id_mismatch")
    if plan.context_sha256 != context.content_sha256:
        reasons.append("plan_context_sha256_mismatch")
    if plan.graph_revision != context.revision.graph_revision:
        reasons.append("plan_graph_revision_mismatch")
    if not 1 <= len(plan.route_families) <= limits.max_route_families:
        reasons.append("route_family_count_out_of_bounds")
    if len(plan.multi_step_skeletons) > limits.max_skeletons:
        reasons.append("skeleton_count_out_of_bounds")
    if not plan.portfolio_rationale.strip():
        reasons.append("portfolio_rationale_missing")
    authority_paths = _forbidden_authority_paths(plan.to_dict())
    if authority_paths:
        reasons.append("director_claimed_scientific_authority")
    family_ids = _unique_ids(plan.route_families, "route_family_id", reasons)
    _unique_ids(plan.strategic_disconnections, "disconnection_id", reasons)
    _unique_ids(plan.shared_intermediates, "intermediate_id", reasons)
    _unique_ids(plan.critical_unknowns, "unknown_id", reasons)
    _unique_ids(plan.source_plan, "source_task_id", reasons)
    _unique_ids(plan.fallback_strategies, "fallback_id", reasons)
    _unique_ids(plan.frontier_priorities, "priority_id", reasons)
    _unique_ids(plan.pivot_conditions, "pivot_id", reasons)
    _unique_ids(plan.stop_conditions, "stop_id", reasons)
    audits: list[dict[str, Any]] = []
    skeleton_ids: set[str] = set()
    for skeleton in plan.multi_step_skeletons:
        skeleton_id = str(skeleton.get("skeleton_id") or "")
        if not skeleton_id or skeleton_id in skeleton_ids:
            reasons.append("skeleton_identity_invalid_or_duplicate")
        skeleton_ids.add(skeleton_id)
        if str(skeleton.get("route_family_id") or "") not in family_ids:
            reasons.append(f"skeleton_route_family_unknown:{skeleton_id}")
        steps = skeleton.get("steps")
        if not isinstance(steps, list) or not steps:
            reasons.append(f"skeleton_steps_missing:{skeleton_id}")
            continue
        if len(steps) > limits.max_steps_per_skeleton:
            reasons.append(f"skeleton_step_count_out_of_bounds:{skeleton_id}")
        seen_steps: set[str] = set()
        for raw_step in steps:
            audit = _validate_step(raw_step, skeleton_id=skeleton_id)
            step_id = str(audit["proposal_id"])
            if step_id in seen_steps:
                audit["accepted"] = False
                audit["reasons"] = sorted(
                    set([*audit["reasons"], "step_id_duplicate_in_skeleton"])
                )
            seen_steps.add(step_id)
            audits.append(audit)
    if reasons:
        raise GlobalCampaignPlanValidationError(";".join(sorted(set(reasons))))
    return audits


def _validate_step(value: Any, *, skeleton_id: str) -> dict[str, Any]:
    row = dict(value) if isinstance(value, Mapping) else {}
    step_id = str(row.get("step_id") or "")
    reasons: list[str] = []
    if not step_id:
        reasons.append("step_id_missing")
    product = _canonical_smiles(row.get("product_smiles"))
    raw_precursors = row.get("precursor_smiles")
    precursors = (
        [_canonical_smiles(item) for item in raw_precursors]
        if isinstance(raw_precursors, list)
        else []
    )
    if not product:
        reasons.append("product_identity_invalid")
    if not precursors or any(not item for item in precursors):
        reasons.append("precursor_identity_invalid")
    if product and product in precursors:
        reasons.append("product_repeated_as_precursor")
    if not str(row.get("transformation_hypothesis") or "").strip():
        reasons.append("transformation_hypothesis_missing")
    required_validation = row.get("required_validation")
    if not isinstance(required_validation, list) or not required_validation:
        reasons.append("required_validation_missing")
    if row.get("hypothesis_only") is not True:
        reasons.append("hypothesis_only_marker_missing")
    if _forbidden_authority_paths(row):
        reasons.append("step_claimed_scientific_authority")
    return {
        "schema_version": "director_reaction_proposal_audit.v1",
        "proposal_id": step_id or f"missing:{skeleton_id}",
        "skeleton_id": skeleton_id,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "canonical_product_smiles": product,
        "canonical_precursor_smiles": sorted(set(precursors)) if all(precursors) else [],
        "authority_scope": "frontier_candidate_only",
    }


def director_trigger_reasons(context: CampaignContext, *, mode: str) -> list[str]:
    if mode == "initial_architecture":
        return ["initial_architecture_requested"]
    if mode == "final_portfolio_synthesis":
        return ["final_portfolio_synthesis_requested"]
    events = set(context.delta.material_events)
    return sorted(events & MATERIAL_REPLAN_EVENTS)


def proposal_ids(plan: GlobalCampaignPlan) -> list[str]:
    identities: list[str] = []
    for skeleton in plan.multi_step_skeletons:
        for step in skeleton.get("steps") or []:
            if isinstance(step, Mapping) and step.get("step_id"):
                identities.append(str(step["step_id"]))
    return sorted(set(identities))


def director_prompt(
    context: CampaignContext,
    *,
    mode: str,
    config: DirectorConfig,
) -> str:
    return "\n".join(
        [
            "You are AutoPlanner's GlobalCampaignDirector direct child agent.",
            "Reason over the complete multi-route campaign, not one local disconnection.",
            "Return exactly one GlobalCampaignPlan JSON object matching the host schema.",
            "All molecules and reactions are hypothesis-only and must request host validation.",
            "Never claim proof, validation, stock closure, route completion, or solved status.",
            "Coordinate route families, multi-step skeletons, shared intermediates, evidence acquisition, fallbacks, pivots, and portfolio tradeoffs together.",
            f"Mode: {mode}",
            f"Limits: at most {config.max_route_families} route families, {config.max_skeletons} skeletons, and {config.max_steps_per_skeleton} steps per skeleton.",
            "Each skeleton step requires step_id, product_smiles, precursor_smiles, transformation_hypothesis, required_validation, and hypothesis_only=true.",
            "CampaignContext:",
            json.dumps(context.to_dict(), ensure_ascii=False, sort_keys=True),
        ]
    )


def run_codex_cli_director_child(
    spec: AgentSpec,
    context: CampaignContext,
    mode: str,
    config: DirectorConfig,
) -> AgentResult:
    """Default direct-child adapter over the existing controlled Codex CLI."""

    task = WorkerTask(
        task_id=spec.agent_id,
        case_id=spec.run_id,
        task_type="global_campaign_direction",
        required_artifact_type="GlobalCampaignPlan",
        input_refs=[context.content_sha256],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=config.max_wall_time_s,
            max_output_bytes=config.max_output_bytes,
            max_tool_calls=config.max_tool_calls,
            max_worker_runs=1,
            reasoning_effort=config.reasoning_effort,
        ),
        objective=spec.objective,
        allowed_workdir=str(Path.cwd()),
        agent_mode="single",
        codex_auth_mode="ambient_codex_cli",
        model=config.model,
    )
    record = run_codex_worker(task, use_codex_cli=True)
    succeeded = record.status == "accepted_draft" and isinstance(
        record.output_artifact, Mapping
    )
    output = (
        dict(record.output_artifact.get("payload") or {})
        if succeeded and isinstance(record.output_artifact, Mapping)
        else None
    )
    usage = dict(record.usage or {})
    if normalize_director_usage(usage)["model_invocations"] == 0:
        usage["model_invocations"] = 1
    return AgentResult(
        run_id=spec.run_id,
        agent_id=spec.agent_id,
        parent_agent_id=spec.parent_agent_id,
        child_agent_ids=(),
        attempt=spec.attempt,
        idempotency_key=f"{spec.idempotency_key}:result",
        context_hash=spec.context_hash,
        capabilities=spec.capabilities,
        write_scope=spec.write_scope,
        budget=spec.budget,
        state=AgentState.SUCCEEDED if succeeded else AgentState.FAILED,
        output=output,
        error="" if succeeded else (record.stderr or record.status),
        usage=usage,
        metadata={
            "backend": record.backend,
            "worker_status": record.status,
            "mode": mode,
            "direct_child": True,
        },
    )


def normalize_director_usage(value: Mapping[str, Any] | None) -> dict[str, int | float]:
    row = dict(value or {})
    wall = float(
        row.get("wall_time_s")
        or row.get("elapsed_s")
        or row.get("duration_s")
        or 0.0
    )
    if not math.isfinite(wall) or wall < 0:
        wall = 0.0
    return {
        "model_invocations": max(
            0,
            int(
                row.get("model_invocations")
                or row.get("invocations")
                or row.get("calls")
                or 0
            ),
        ),
        "visual_invocations": 0,
        "input_tokens": max(
            0,
            int(
                row.get("input_tokens")
                or row.get("prompt_tokens")
                or row.get("total_input_tokens")
                or 0
            ),
        ),
        "output_tokens": max(
            0,
            int(
                row.get("output_tokens")
                or row.get("completion_tokens")
                or row.get("total_output_tokens")
                or 0
            ),
        ),
        "wall_time_s": wall,
    }


def _unique_ids(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    reasons: list[str],
) -> set[str]:
    values = [str(row.get(key) or "") for row in rows]
    if any(not value for value in values):
        reasons.append(f"{key}_missing")
    if len(values) != len(set(values)):
        reasons.append(f"{key}_duplicate")
    return {value for value in values if value}


def _forbidden_authority_paths(value: Any, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child = f"{path}.{key}"
            allowed_values = (
                False,
                None,
                "",
                "hypothesis",
                "proposed",
                "unresolved",
            )
            if key.casefold() in _FORBIDDEN_AUTHORITY_KEYS and not any(
                item == allowed for allowed in allowed_values
            ):
                found.append(child)
            found.extend(_forbidden_authority_paths(item, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_forbidden_authority_paths(item, path=f"{path}[{index}]"))
    return found


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GlobalCampaignDirectorError("director_output_not_object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


__all__ = [
    "DIRECTOR_DISPOSITIONS",
    "DIRECTOR_MODES",
    "GLOBAL_CAMPAIGN_PLAN_SCHEMA",
    "DirectorConfig",
    "DirectorOutcome",
    "DirectorRunner",
    "GlobalCampaignDirector",
    "GlobalCampaignDirectorError",
    "GlobalCampaignPlan",
    "GlobalCampaignPlanValidationError",
    "ReplayDirectorRunner",
    "director_prompt",
    "director_trigger_reasons",
    "normalize_director_usage",
    "proposal_ids",
    "run_codex_cli_director_child",
    "validate_global_campaign_plan",
]
