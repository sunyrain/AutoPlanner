"""Compact, sequential Codex policy search for the SynthEx-matched profile.

The legacy global director asks one model call to describe several complete
routes.  This adapter instead owns three independent branch states and asks
one bounded model call to expand exactly one open molecule at a time.  It
returns the same hypothesis-only ``GlobalCampaignPlan`` contract so all host
materialisation and chemistry gates remain unchanged.
"""

from __future__ import annotations

from collections import deque
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import re
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from cascade_planner.application.reactionjson_replay import (
    ReactionJsonReplayError,
    replay_reactionjson,
)
from cascade_planner.application.strategy_contract import (
    normalize_reaction_operations,
    normalize_strategy_card,
    strategy_cards_conflict,
)

from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerRunRecord,
    WorkerTask,
    run_codex_worker,
)
from cascade_planner.application.campaign_context import CampaignContext
from cascade_planner.orchestration.global_campaign_director import DirectorConfig
from cascade_planner.runtime import AgentResult, AgentSpec, AgentState


SEQUENTIAL_STRATEGY_SEARCH_SCHEMA = "sequential_strategy_search.v1"
_BRANCH_MANDATES = (
    (
        "prioritize a convergent skeletal construction: identify a key forward "
        "bond-forming event that joins substantial fragments"
    ),
    (
        "prioritize a topology-changing construction: ring formation, cascade, "
        "cycloaddition, or skeletal rearrangement"
    ),
    (
        "prioritize an orthogonal enzymatic, whole-cell, chemoenzymatic, or one-hop "
        "mechanistic strategy with an explicit substrate-product relationship; retain "
        "a conventional chemical fallback when no credible biological capability exists"
    ),
)

_STRATEGY_CARD_FIELDS = (
    "scaffold_motif",
    "key_forward_transformation",
    "key_bond_changes",
    "functional_group_conflicts",
    "protection_policy",
    "stereochemical_plan",
    "convergence_plan",
    "strategic_step_count",
    "skeleton_change_class",
    "expected_complexity_drop",
    "orthogonality_basis",
    "strategy_signature",
)

# Codex CLI usage includes a provider/system envelope in addition to the
# compact prompt bytes.  Reserve a conservative per-critic allowance before
# spending the expansion budget so the independent critic cannot be silently
# starved after strategy generation.
_CRITIC_INPUT_TOKEN_RESERVE = 24_000
_CRITIC_OUTPUT_TOKEN_RESERVE = 8_000
_ROOT_GRAPH_EDIT_RETRY_LIMIT = 3


@dataclass(frozen=True, slots=True)
class NodeExpansion:
    product_smiles: str
    precursor_smiles: tuple[str, ...]
    reaction_family: str
    rationale: str
    conditions: tuple[str, ...] = ()
    catalyst: str = ""
    enzyme: str = ""
    limitations: tuple[str, ...] = ()
    product_retron_type: str = ""
    strategy_card: Mapping[str, Any] | None = None
    reaction_operations: tuple[Mapping[str, Any], ...] = ()
    reactionjson_audit: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _NodeCallBudget:
    model_invocations: int
    input_tokens: int
    output_tokens: int
    wall_time_s: float


NodeExecutor = Callable[[WorkerTask], WorkerRunRecord]
StockMembership = Callable[[Iterable[str]], Mapping[str, bool]]


class SequentialStrategyDirectorRunner:
    """Director runner implementing independent, continuous node expansion."""

    compact_prompt = True

    def __init__(
        self,
        *,
        node_executor: NodeExecutor | None = None,
        critic_executor: NodeExecutor | None = None,
        stock_membership: StockMembership | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.node_executor = node_executor or self._execute_node
        self.critic_executor = critic_executor or self.node_executor
        self.stock_membership = stock_membership
        self.cancel_event = cancel_event

    def prompt_for(
        self,
        context: CampaignContext,
        mode: str,
        config: DirectorConfig,
    ) -> str:
        target = _canonical_smiles(context.target.get("canonical_smiles"))
        if mode == "event_replan":
            return (
                "Repair one failed reaction neighborhood at a time; retain the "
                f"target-rooted prefix. target={target}; rounds<="
                f"{config.max_route_local_repair_rounds}."
            )
        return (
            "Run compact sequential retrosynthesis policy search; "
            f"target={target}; independent_branches={config.strategy_branch_count}; "
            f"node_expansions_per_branch={config.max_node_expansions_per_branch}."
        )

    def __call__(
        self,
        spec: AgentSpec,
        context: CampaignContext,
        mode: str,
        config: DirectorConfig,
    ) -> AgentResult:
        started = time.monotonic()
        quota = _node_call_budget(spec, mode=mode, config=config)
        if mode == "event_replan":
            branches, records = self._repair_branches(
                spec, context, config, quota=quota, started=started
            )
        else:
            branches, records = self._initial_branches(
                spec, context, config, quota=quota, started=started
            )
        branches = self._run_codex_critics(
            spec,
            context,
            branches,
            records,
            quota=quota,
            started=started,
            config=config,
        )
        unavailable_critics = [
            branch
            for branch in branches
            if branch.get("steps")
            and str(dict(branch.get("chemical_critic") or {}).get("status") or "")
            == "unavailable"
        ]
        if unavailable_critics:
            return _agent_result(
                spec,
                state=AgentState.FAILED,
                output=None,
                usage=_aggregate_usage(records, elapsed_s=time.monotonic() - started),
                error="independent_codex_critic_unavailable",
                mode=mode,
            )
        usage = _aggregate_usage(records, elapsed_s=time.monotonic() - started)
        usage["accepted_expansions"] = (
            len(branches)
            if mode == "event_replan"
            else sum(len(branch.get("steps") or []) for branch in branches)
        )
        if self.cancel_event is not None and self.cancel_event.is_set():
            return _agent_result(
                spec,
                state=AgentState.CANCELLED,
                output=None,
                usage=usage,
                error="delivery_milestone_reached",
                mode=mode,
            )
        usable = [
            branch
            for branch in branches
            if branch["steps"]
            and str(dict(branch.get("chemical_critic") or {}).get("status") or "")
            != "reject"
        ]
        if not usable:
            rejection_reasons = sorted(
                {
                    str(item.get("reason") or "")
                    for branch in branches
                    for item in branch.get("rejections") or []
                    if str(item.get("reason") or "")
                }
            )
            usage["root_graph_edit_retry_limit"] = _ROOT_GRAPH_EDIT_RETRY_LIMIT
            usage["rejection_reasons"] = rejection_reasons
            return _agent_result(
                spec,
                state=AgentState.FAILED,
                output=None,
                usage=usage,
                error=(
                    "sequential_strategy_search_produced_no_valid_expansion"
                    + (":" + ",".join(rejection_reasons) if rejection_reasons else "")
                ),
                mode=mode,
            )
        plan = _compile_plan(
            context,
            mode=mode,
            branches=usable,
            requested_branch_count=config.strategy_branch_count,
        )
        return _agent_result(
            spec,
            state=AgentState.SUCCEEDED,
            output=plan,
            usage=usage,
            error="",
            mode=mode,
        )

    def _run_codex_critics(
        self,
        spec: AgentSpec,
        context: CampaignContext,
        branches: list[dict[str, Any]],
        records: list[WorkerRunRecord],
        *,
        quota: _NodeCallBudget,
        started: float,
        config: DirectorConfig,
    ) -> list[dict[str, Any]]:
        """Run one blind Codex forward critic per constructed route family."""

        target = _canonical_smiles(context.target.get("canonical_smiles"))
        for index, branch in enumerate(branches):
            if not branch.get("steps"):
                continue
            remaining_critics = sum(
                bool(row.get("steps"))
                and not dict(row.get("chemical_critic") or {}).get("status")
                for row in branches[index:]
            )
            # A critic timeout is an upper bound for one call, not a separate
            # wall-time budget that can be multiplied by the whole portfolio.
            # Scale the reservation to the remaining director wall budget so
            # short local/unit-test budgets still admit every critic while the
            # production path keeps a real reservation for them.
            remaining_wall = _remaining_node_wall_time(started, quota)
            per_critic_wall = min(
                config.critic_call_timeout_s,
                remaining_wall / max(1, remaining_critics + 1),
            )
            if not _node_budget_allows(
                records,
                started=started,
                quota=quota,
                reserve_model_invocations=max(0, remaining_critics - 1),
                reserve_input_tokens=max(0, remaining_critics - 1)
                * _CRITIC_INPUT_TOKEN_RESERVE,
                reserve_output_tokens=max(0, remaining_critics - 1)
                * _CRITIC_OUTPUT_TOKEN_RESERVE,
                reserve_wall_time_s=max(0, remaining_critics - 1)
                * max(0.0, per_critic_wall),
            ):
                branch["chemical_critic"] = {
                    "schema_version": "chemical_strategy_critique.v1",
                    "status": "unavailable",
                    "reason": "critic_budget_exhausted",
                    "semantics": {"critic_required_before_evidence": True},
                }
                continue
            prompt = _critic_prompt(
                target=target,
                branch_index=int(branch.get("branch_index") or 0),
                strategy_card=dict(branch.get("strategy_card") or {}),
                steps=list(branch.get("steps") or []),
            )
            if len(prompt.encode("utf-8")) > config.max_node_prompt_bytes:
                prompt = _critic_prompt(
                    target=target,
                    branch_index=int(branch.get("branch_index") or 0),
                    strategy_card=dict(branch.get("strategy_card") or {}),
                    steps=list(branch.get("steps") or [])[-8:],
                )
            _assert_node_prompt_size(prompt, config.max_node_prompt_bytes)
            task = _critic_task(
                spec,
                prompt=prompt,
                branch_index=int(branch.get("branch_index") or 0),
                timeout_s=max(
                    1.0,
                    min(
                        config.critic_call_timeout_s,
                        _remaining_node_wall_time(started, quota),
                    ),
                ),
            )
            try:
                record = self.critic_executor(task)
            except Exception as exc:
                records.append(
                    WorkerRunRecord(
                        run_id=f"{task.task_id}:run",
                        task_id=task.task_id,
                        case_id=task.case_id,
                        status="worker_error",
                        backend="critic_executor",
                        stderr=f"{type(exc).__name__}: {exc}",
                        output_validation={
                            "accepted": False,
                            "reasons": ["critic_execution_failed"],
                        },
                    )
                )
                branch["chemical_critic"] = {
                    "schema_version": "chemical_strategy_critique.v1",
                    "status": "unavailable",
                    "reason": f"critic_execution_failed:{type(exc).__name__}",
                    "semantics": {"critic_required_before_evidence": True},
                }
                continue
            records.append(record)
            branch["chemical_critic"] = _critique_from_record(record)
        return branches

    def _initial_branches(
        self,
        spec: AgentSpec,
        context: CampaignContext,
        config: DirectorConfig,
        *,
        quota: _NodeCallBudget,
        started: float,
    ) -> tuple[list[dict[str, Any]], list[WorkerRunRecord]]:
        target = _canonical_smiles(context.target.get("canonical_smiles"))
        branches: list[dict[str, Any]] = [
            {
                "branch_index": branch_index,
                "lens": _BRANCH_MANDATES[branch_index % len(_BRANCH_MANDATES)],
                "strategy_seed": "",
                "steps": [],
                "open_leaves": deque([target]),
                "expanded_products": set(),
                "call_count": 0,
                "rejections": [],
                "complete_in_bound_stock": False,
                "strategy_card": {},
                "chemical_critic": {},
            }
            for branch_index in range(config.strategy_branch_count)
        ]
        records: list[WorkerRunRecord] = []

        # Phase 1 is strategy-first.  Every branch must first commit a concrete
        # StrategyCard before any branch is allowed to spend a second call.
        # This prevents a fast stock hit from collapsing the intended
        # three-strategy portfolio into one uncriticised local transformation.
        while not self._cancelled() and any(not branch["steps"] for branch in branches):
            progressed = False
            for branch in branches:
                if branch["steps"]:
                    continue
                if self._cancelled() or not _node_budget_allows(
                    records, started=started, quota=quota
                ):
                    break
                if (
                    int(branch["call_count"]) >= config.max_node_expansions_per_branch
                    or not branch["open_leaves"]
                ):
                    continue
                self._expand_one_branch_node(
                    spec,
                    target=target,
                    branch=branch,
                    records=records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    require_strategy_graph_edits=config.require_strategy_graph_edits,
                    quota=quota,
                    started=started,
                    forbidden_strategy_cards=_accepted_strategy_cards(
                        branches, exclude_index=int(branch["branch_index"])
                    ),
                )
                progressed = True
                if branch["steps"] and not branch["open_leaves"]:
                    branch["complete_in_bound_stock"] = True
            if not progressed or not _node_budget_allows(records, started=started, quota=quota):
                break

        seeded = [branch for branch in branches if branch["steps"]]
        if len(seeded) == len(branches) and any(
            branch["complete_in_bound_stock"] for branch in branches
        ):
            return [_public_branch(row) for row in branches], records

        # Phase 2 expands the already committed strategies round-robin.  Route
        # state remains isolated; only compact StrategyCard signatures are
        # shared to enforce portfolio orthogonality.
        critic_slots = len(seeded)
        critic_wall_reserve = critic_slots * min(
            config.critic_call_timeout_s,
            quota.wall_time_s / max(1, critic_slots + 1),
        )
        critic_input_reserve = critic_slots * _CRITIC_INPUT_TOKEN_RESERVE
        critic_output_reserve = critic_slots * _CRITIC_OUTPUT_TOKEN_RESERVE
        while not self._cancelled():
            progressed = False
            for branch in branches:
                if self._cancelled() or not _node_budget_allows(
                    records,
                    started=started,
                    quota=quota,
                    reserve_model_invocations=critic_slots,
                    reserve_input_tokens=critic_input_reserve,
                    reserve_output_tokens=critic_output_reserve,
                    reserve_wall_time_s=critic_wall_reserve,
                ):
                    break
                if (
                    int(branch["call_count"]) >= config.max_node_expansions_per_branch
                    or not branch["open_leaves"]
                ):
                    continue
                self._expand_one_branch_node(
                    spec,
                    target=target,
                    branch=branch,
                    records=records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    require_strategy_graph_edits=config.require_strategy_graph_edits,
                    quota=quota,
                    started=started,
                    forbidden_strategy_cards=(),
                )
                progressed = True
                if branch["steps"] and not branch["open_leaves"]:
                    branch["complete_in_bound_stock"] = True
                    return [_public_branch(row) for row in branches], records
            if not progressed or not _node_budget_allows(
                records,
                started=started,
                quota=quota,
                reserve_model_invocations=critic_slots,
                reserve_input_tokens=critic_input_reserve,
                reserve_output_tokens=critic_output_reserve,
                reserve_wall_time_s=critic_wall_reserve,
            ):
                break
        return [_public_branch(row) for row in branches], records

    def _repair_branches(
        self,
        spec: AgentSpec,
        context: CampaignContext,
        config: DirectorConfig,
        *,
        quota: _NodeCallBudget,
        started: float,
    ) -> tuple[list[dict[str, Any]], list[WorkerRunRecord]]:
        target = _canonical_smiles(context.target.get("canonical_smiles"))
        repair_product, prefix, host_failure_feedback = _repair_neighborhood(context, target=target)
        branches: list[dict[str, Any]] = []
        records: list[WorkerRunRecord] = []
        rejected: list[dict[str, Any]] = []
        # One repair event edits one failed neighborhood.  Host
        # materialisation and validation occur before another event can spend
        # the next round, up to the DirectorConfig event-call limit.
        for repair_index in range(1):
            if self._cancelled() or not _node_budget_allows(records, started=started, quota=quota):
                break
            prompt = _node_prompt(
                target=target,
                branch_index=repair_index,
                lens="route-local replacement of one failed reaction neighborhood",
                selected_product=repair_product,
                steps=prefix,
                open_leaves=[repair_product],
                prior_rejections=rejected,
                repair=True,
                strategy_card={},
                forbidden_strategy_cards=(),
                host_failure_feedback=host_failure_feedback,
            )
            if len(prompt.encode("utf-8")) > config.max_node_prompt_bytes:
                prompt = _node_prompt(
                    target=target,
                    branch_index=repair_index,
                    lens="route-local replacement",
                    selected_product=repair_product,
                    steps=prefix[-3:],
                    open_leaves=[repair_product],
                    prior_rejections=rejected[-2:],
                    repair=True,
                    strategy_card={},
                    forbidden_strategy_cards=(),
                    host_failure_feedback=host_failure_feedback,
                )
            _assert_node_prompt_size(prompt, config.max_node_prompt_bytes)
            task = _node_task(
                spec,
                prompt=prompt,
                branch_index=repair_index,
                node_index=0,
                model=str(spec.metadata.get("model") or ""),
                reasoning_effort=str(spec.metadata.get("reasoning_effort") or "medium"),
                timeout_s=max(
                    1.0,
                    min(
                        config.max_node_call_timeout_s,
                        _remaining_node_wall_time(started, quota),
                    ),
                ),
            )
            record = self.node_executor(task)
            records.append(record)
            expansion = _expansion_from_record(
                record,
                expected_product=repair_product,
                mapped_product_smiles=_mapped_smiles(repair_product),
            )
            if expansion is None:
                rejected.append({"round": repair_index + 1, "reason": "invalid_output"})
                continue
            step = _step_row(
                expansion,
                step_id=f"codex:repair:{repair_index + 1}:1",
            )
            branches.append(
                {
                    "branch_index": repair_index,
                    "lens": "route-local repair",
                    "steps": [*prefix, step],
                    "open_leaves": list(expansion.precursor_smiles),
                    "call_count": 1,
                    "complete_in_bound_stock": False,
                    "repair_scope": {
                        "product_smiles": repair_product,
                        "preserved_prefix_step_count": len(prefix),
                    },
                }
            )
        return branches, records

    def _expand_one_branch_node(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branch: dict[str, Any],
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        require_strategy_graph_edits: bool = False,
        quota: _NodeCallBudget,
        started: float,
        forbidden_strategy_cards: Iterable[Mapping[str, Any]],
    ) -> None:
        branch_index = int(branch["branch_index"])
        lens = str(branch["lens"])
        open_leaves: deque[str] = branch["open_leaves"]
        steps: list[dict[str, Any]] = branch["steps"]
        expanded_products: set[str] = branch["expanded_products"]
        rejected: list[dict[str, Any]] = branch["rejections"]
        while open_leaves:
            selected = open_leaves.popleft()
            if selected in expanded_products:
                continue
            break
        else:
            return
        branch["call_count"] = int(branch["call_count"]) + 1
        call_index = int(branch["call_count"])
        is_root_strategy = not steps and selected == target
        forbidden_cards = [dict(row) for row in forbidden_strategy_cards]
        prompt = _node_prompt(
            target=target,
            branch_index=branch_index,
            lens=lens,
            selected_product=selected,
            steps=steps,
            open_leaves=[selected, *open_leaves],
            prior_rejections=rejected,
            repair=False,
            strategy_card=dict(branch.get("strategy_card") or {}),
            forbidden_strategy_cards=forbidden_cards,
            host_failure_feedback={},
        )
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            prompt = _node_prompt(
                target=target,
                branch_index=branch_index,
                lens=lens,
                selected_product=selected,
                steps=steps[-6:],
                open_leaves=[selected, *list(open_leaves)[:12]],
                prior_rejections=rejected[-4:],
                repair=False,
                strategy_card=dict(branch.get("strategy_card") or {}),
                forbidden_strategy_cards=forbidden_cards,
                host_failure_feedback={},
            )
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            prompt = _node_prompt(
                target=target,
                branch_index=branch_index,
                lens=lens,
                selected_product=selected,
                steps=(),
                open_leaves=[selected],
                prior_rejections=(),
                repair=False,
                strategy_card=dict(branch.get("strategy_card") or {}),
                forbidden_strategy_cards=forbidden_cards,
                host_failure_feedback={},
            )
        _assert_node_prompt_size(prompt, max_prompt_bytes)
        task = _node_task(
            spec,
            prompt=prompt,
            branch_index=branch_index,
            node_index=call_index,
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(spec.metadata.get("reasoning_effort") or "medium"),
            timeout_s=max(
                1.0,
                min(
                    max_node_call_timeout_s,
                    _remaining_node_wall_time(started, quota),
                ),
            ),
        )
        record = self.node_executor(task)
        records.append(record)
        expansion = _expansion_from_record(
            record,
            expected_product=selected,
            require_strategy_card=is_root_strategy,
            mapped_product_smiles=_mapped_smiles(selected),
            require_reaction_operations=(
                bool(require_strategy_graph_edits and is_root_strategy)
            ),
        )
        if expansion is None or any(
            precursor in expanded_products for precursor in expansion.precursor_smiles
        ):
            rejection_reason = (
                _expansion_rejection_reason(
                    record,
                    expected_product=selected,
                    mapped_product_smiles=_mapped_smiles(selected),
                    require_reaction_operations=(
                        bool(require_strategy_graph_edits and is_root_strategy)
                    ),
                )
                if expansion is None
                else "ancestor_cycle"
            )
            rejected.append(
                {
                    "node": call_index,
                    "product_smiles": selected,
                    "reason": rejection_reason,
                }
            )
            if is_root_strategy and rejection_reason in {
                "strategy_graph_edit_missing",
                "strategy_graph_edit_replay_failed",
            }:
                graph_edit_rejections = int(branch.get("graph_edit_rejections") or 0) + 1
                branch["graph_edit_rejections"] = graph_edit_rejections
                if graph_edit_rejections >= _ROOT_GRAPH_EDIT_RETRY_LIMIT:
                    rejected.append(
                        {
                            "node": call_index,
                            "product_smiles": selected,
                            "reason": "strategy_graph_edit_retry_limit_reached",
                        }
                    )
                    branch["open_leaves"] = deque()
                    return
            open_leaves.append(selected)
            return
        if not is_root_strategy and branch.get("strategy_card"):
            expansion = replace(
                expansion,
                strategy_card=dict(branch.get("strategy_card") or {}),
            )
        if is_root_strategy and _strategy_conflicts(expansion.strategy_card or {}, forbidden_cards):
            rejected.append(
                {
                    "node": call_index,
                    "product_smiles": selected,
                    "reason": "root_strategy_not_orthogonal",
                    "strategy_signature": _strategy_signature(expansion.strategy_card or {}),
                }
            )
            open_leaves.append(selected)
            return
        expanded_products.add(selected)
        steps.append(
            _step_row(
                expansion,
                step_id=f"codex:branch:{branch_index + 1}:{len(steps) + 1}",
                strategy_anchor=is_root_strategy,
            )
        )
        if len(steps) == 1:
            card = dict(expansion.strategy_card or {})
            strategy_seed = _strategy_title(card, fallback=expansion)
            branch["strategy_seed"] = strategy_seed
            branch["strategy_card"] = card
            branch["lens"] = "Codex-authored strategy - " + strategy_seed
        membership = self._stock_membership(expansion.precursor_smiles)
        for precursor in expansion.precursor_smiles:
            if membership.get(precursor) is not True and precursor not in expanded_products:
                open_leaves.append(precursor)
        branch["open_leaves"] = deque(dict.fromkeys(open_leaves))

    def _stock_membership(self, values: Iterable[str]) -> dict[str, bool]:
        canonical = tuple(dict.fromkeys(_canonical_smiles(value) for value in values))
        canonical = tuple(value for value in canonical if value)
        if self.stock_membership is None:
            return {value: False for value in canonical}
        try:
            observed = self.stock_membership(canonical)
        except Exception:
            return {value: False for value in canonical}
        return {value: observed.get(value) is True for value in canonical}

    def _cancelled(self) -> bool:
        return bool(self.cancel_event is not None and self.cancel_event.is_set())

    @staticmethod
    def _execute_node(task: WorkerTask) -> WorkerRunRecord:
        return run_codex_worker(task, use_codex_cli=True)


def _node_call_budget(
    spec: AgentSpec,
    *,
    mode: str,
    config: DirectorConfig,
) -> _NodeCallBudget:
    remaining = dict(spec.metadata.get("remaining_model_budget") or {})
    default_calls = (
        config.max_route_local_repair_rounds + 1
        if mode == "event_replan"
        else config.strategy_branch_count * (config.max_node_expansions_per_branch + 1)
    )
    spec_wall = (
        float(spec.budget.max_wall_time_s)
        if spec.budget.max_wall_time_s is not None
        else float(config.max_wall_time_s)
    )
    remaining_wall = float(remaining.get("wall_time_s", spec_wall) or 0.0)
    return _NodeCallBudget(
        model_invocations=max(0, int(remaining.get("model_invocations", default_calls) or 0)),
        input_tokens=max(0, int(remaining.get("input_tokens", 2**63 - 1) or 0)),
        output_tokens=max(0, int(remaining.get("output_tokens", 2**63 - 1) or 0)),
        wall_time_s=max(0.0, min(spec_wall, remaining_wall)),
    )


def _assert_node_prompt_size(prompt: str, maximum_bytes: int) -> None:
    if len(prompt.encode("utf-8")) > maximum_bytes:
        raise ValueError("compact_node_prompt_byte_budget_exceeded")


def _node_budget_allows(
    records: Iterable[WorkerRunRecord],
    *,
    started: float,
    quota: _NodeCallBudget,
    reserve_model_invocations: int = 0,
    reserve_input_tokens: int = 0,
    reserve_output_tokens: int = 0,
    reserve_wall_time_s: float = 0.0,
) -> bool:
    rows = list(records)
    if len(rows) + max(0, int(reserve_model_invocations)) >= quota.model_invocations:
        return False
    input_tokens = sum(
        max(
            0,
            int(
                dict(row.usage or {}).get("input_tokens")
                or dict(row.usage or {}).get("prompt_tokens")
                or 0
            ),
        )
        for row in rows
    )
    output_tokens = sum(
        max(
            0,
            int(
                dict(row.usage or {}).get("output_tokens")
                or dict(row.usage or {}).get("completion_tokens")
                or 0
            ),
        )
        for row in rows
    )
    return bool(
        input_tokens + max(0, int(reserve_input_tokens)) < quota.input_tokens
        and output_tokens + max(0, int(reserve_output_tokens)) < quota.output_tokens
        and _remaining_node_wall_time(started, quota)
        > max(0.0, float(reserve_wall_time_s))
    )


def _remaining_node_wall_time(started: float, quota: _NodeCallBudget) -> float:
    return max(0.0, quota.wall_time_s - (time.monotonic() - started))


def _public_branch(branch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_index": int(branch.get("branch_index") or 0),
        "lens": str(branch.get("lens") or ""),
        "strategy_seed": str(branch.get("strategy_seed") or ""),
        "strategy_card": dict(branch.get("strategy_card") or {}),
        "steps": [dict(row) for row in branch.get("steps") or []],
        "open_leaves": list(branch.get("open_leaves") or []),
        "call_count": int(branch.get("call_count") or 0),
        "complete_in_bound_stock": bool(branch.get("complete_in_bound_stock")),
        "rejections": [dict(row) for row in branch.get("rejections") or []],
        "chemical_critic": dict(branch.get("chemical_critic") or {}),
    }


def _accepted_strategy_cards(
    branches: Iterable[Mapping[str, Any]], *, exclude_index: int
) -> list[dict[str, Any]]:
    return [
        dict(branch.get("strategy_card") or {})
        for branch in branches
        if int(branch.get("branch_index") or 0) != exclude_index
        and _valid_strategy_card(dict(branch.get("strategy_card") or {}))
    ]


def _valid_strategy_card(card: Mapping[str, Any]) -> bool:
    if any(field not in card for field in _STRATEGY_CARD_FIELDS):
        return False
    if not all(
        str(card.get(field) or "").strip()
        for field in (
            "scaffold_motif",
            "key_forward_transformation",
            "protection_policy",
            "stereochemical_plan",
            "convergence_plan",
            "skeleton_change_class",
            "expected_complexity_drop",
            "orthogonality_basis",
            "strategy_signature",
        )
    ):
        return False
    if not isinstance(card.get("key_bond_changes"), list) or not card.get("key_bond_changes"):
        return False
    if not isinstance(card.get("functional_group_conflicts"), list):
        return False
    try:
        step_count = int(card.get("strategic_step_count"))
    except (TypeError, ValueError):
        return False
    return step_count in {1, 2} and str(card.get("expected_complexity_drop")) in {
        "low",
        "medium",
        "high",
    }


def _strategy_signature(card: Mapping[str, Any]) -> str:
    raw = str(card.get("strategy_signature") or "").strip().lower()
    if not raw:
        raw = "|".join(
            [
                str(card.get("skeleton_change_class") or ""),
                str(card.get("key_forward_transformation") or ""),
                " ".join(str(value) for value in card.get("key_bond_changes") or []),
            ]
        ).lower()
    return re.sub(r"[^a-z0-9]+", " ", raw).strip()


def _strategy_bucket(card: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            str(card.get("skeleton_change_class") or ""),
            str(card.get("key_forward_transformation") or ""),
            str(card.get("strategy_signature") or ""),
        ]
    ).lower()
    groups = (
        ("chemoenzymatic", ("enzyme", "enzymatic", "synthase", "biocatal", "cyclase")),
        ("cycloaddition", ("cycloaddition", "diels", "dipolar")),
        ("rearrangement", ("rearrangement", "grob", "cope", "pinacol")),
        ("cascade", ("cascade", "domino", "tandem")),
        ("coupling", ("coupling", "arylation", "suzuki", "heck", "fragment union")),
        ("ring_formation", ("cyclization", "annulation", "ring formation", "ring closure")),
        ("redox", ("oxidation", "reduction", "deoxygen", "hydrogenation")),
        ("protection", ("protection", "deprotection", "protecting group")),
        (
            "functionalization",
            ("nitration", "nitro", "halogenation", "methylation"),
        ),
    )
    for label, tokens in groups:
        if any(token in text for token in tokens):
            return label
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "other"


def _strategy_conflicts(
    candidate: Mapping[str, Any], forbidden: Iterable[Mapping[str, Any]]
) -> bool:
    if not _valid_strategy_card(candidate):
        return True
    for prior in forbidden:
        if strategy_cards_conflict(candidate, prior):
            return True
        # Keep the text bucket only as an advisory fallback for legacy cards
        # that predate the structural contract.  It is never authoritative
        # when mapped edits or key-bond signatures are available.
        if (
            not candidate.get("reaction_edit_digest")
            and not prior.get("reaction_edit_digest")
            and _strategy_signature(candidate) == _strategy_signature(prior)
        ):
            return True
    return False


def _strategy_title(card: Mapping[str, Any], *, fallback: NodeExpansion) -> str:
    key_step = str(card.get("key_forward_transformation") or "").strip()
    basis = str(card.get("orthogonality_basis") or "").strip()
    if key_step:
        return f"{key_step}: {basis}" if basis else key_step
    return f"{fallback.reaction_family}: {fallback.rationale}".strip()


def _structure_profile(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {}
    atom_counts = Counter(
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    )
    return {
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "atom_counts": dict(sorted(atom_counts.items())),
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "stereocenter_count": len(
            Chem.FindMolChiralCenters(molecule, includeUnassigned=True, includeCIP=False)
        ),
        "fragment_count": len(Chem.GetMolFrags(molecule)),
    }


def _mapped_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for index, atom in enumerate(molecule.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _heavy_atom_inventory(smiles: str) -> Counter[int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return Counter()
    return Counter(atom.GetAtomicNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)


def _has_atom_provenance_deficit(product: str, precursors: Iterable[str]) -> bool:
    required = _heavy_atom_inventory(product)
    available: Counter[int] = Counter()
    for precursor in precursors:
        available.update(_heavy_atom_inventory(precursor))
    return any(available[atomic_num] < count for atomic_num, count in required.items())


def _node_prompt(
    *,
    target: str,
    branch_index: int,
    lens: str,
    selected_product: str,
    steps: Iterable[Mapping[str, Any]],
    open_leaves: Iterable[str],
    prior_rejections: Iterable[Mapping[str, Any]],
    repair: bool,
    strategy_card: Mapping[str, Any],
    forbidden_strategy_cards: Iterable[Mapping[str, Any]],
    host_failure_feedback: Mapping[str, Any],
) -> str:
    step_rows = [dict(step) for step in steps]
    root_strategy = not repair and not step_rows and selected_product == target
    memory = {
        "schema_version": "compact_retrosynthesis_branch_context.v1",
        "phase": (
            "local_repair" if repair else "root_strategy" if root_strategy else "strategy_execution"
        ),
        "campaign_target": target,
        "campaign_target_profile": _structure_profile(target),
        "branch_id": branch_index + 1,
        "strategy_lens": lens,
        "strategy_card": dict(strategy_card),
        "forbidden_root_strategies": [dict(row) for row in forbidden_strategy_cards],
        "selected_open_leaf": selected_product,
        "selected_open_leaf_mapped": _mapped_smiles(selected_product),
        "selected_open_leaf_profile": _structure_profile(selected_product),
        "accepted_path": [
            {
                "product_smiles": str(step.get("product_smiles") or ""),
                "precursor_smiles": list(step.get("precursor_smiles") or []),
                "reaction_family": str(step.get("transformation_hypothesis") or ""),
            }
            for step in step_rows
        ],
        "open_leaves": list(open_leaves),
        "prior_rejections": [dict(row) for row in prior_rejections],
        "host_failure_feedback": dict(host_failure_feedback),
    }
    phase_instructions = (
        [
            "This is the root Strategy Generator phase, not an ordinary one-step guess.",
            "Before selecting the candidate, compare at least three plausible high-level strategies using four explicit axes: scaffold/ring topology; the key forward bond-forming event; functional-group conflicts and protection; and stereochemical construction or control.",
            "Choose one strategy that satisfies strategy_lens and is orthogonal to every forbidden_root_strategy. Orthogonality must come from a different key bond construction, topology change, or reaction class, not merely different reagents or a nearby functional-group interconversion.",
            "Populate candidate.strategy_card completely. The strategy must name the route-defining key forward construction and a one-to-two strategic-step sequence. A redox, protection, deprotection, methylation, halogenation, or nitration may be the first retro step only when the card identifies the later skeleton-forming event it enables; that functional-group operation is not itself the strategy.",
            "Prefer convergent fragment union, ring construction, cascade, cycloaddition, rearrangement, or a chemically credible chemoenzymatic transformation over cosmetic functional-group editing.",
        ]
        if root_strategy
        else [
            "Execute the supplied StrategyCard; do not silently replace its key construction with an easier functional-group-interconversion route.",
            "Compare at least three local disconnections internally on strategy alignment, skeletal simplification, chemoselectivity, stereochemical compatibility, and precursor accessibility, then return only the best candidate.",
            "Stock availability is an endpoint test, never a chemical justification for a disconnection.",
        ]
    )
    if repair:
        phase_instructions = [
            "This is a route-local repair. Replace only the failed reaction neighborhood and preserve the supplied target-rooted prefix and route-defining key strategy.",
            "Use host_failure_feedback as a causal rejection: the replacement must directly avoid those failure reasons rather than paraphrase the rejected transformation.",
            "Compare at least three local replacements, then return only the best non-blocking candidate.",
        ]
    return "\n".join(
        [
            "Expand exactly one retrosynthetic node and return exactly one candidate.",
            "Route state is isolated from other branches; compact prior StrategyCards are supplied only to enforce portfolio orthogonality.",
            *phase_instructions,
            "The candidate product_smiles must equal selected_open_leaf exactly after canonicalization.",
            "List at most four atom-contributing precursor molecules and include every heavy-atom contributor; omit only species that contribute no product atom, such as solvent, catalyst, counterion and workup species.",
            "The combined precursor heavy-atom inventory must cover the product inventory. Preserve assigned stereochemistry and do not repeat an ancestor as a precursor.",
            "When the transformation is expressible on the mapped product, choose the ordered candidate.reaction_operations first, mentally replay them on selected_open_leaf_mapped, and then copy the resulting unmapped fragment SMILES into candidate.precursor_smiles. The edit program is the source of truth; never invent a precursor topology that the edits do not produce.",
            "Use only map indices present in selected_open_leaf_mapped. Do not use nullable schema filler fields on an operation; each primitive must contain only its semantically relevant fields.",
            "If prior_rejections contains strategy_graph_edit_replay_failed, replace the edit program and precursor together; do not merely rename the same precursor or append unrelated hydrogen edits.",
            "Return one local transformation, not a prose-only route and not multiple output candidates.",
            "The output is hypothesis-only and grants no validation, stock, evidence, condition, or solved claim.",
            "CompactBranchContext:",
            json.dumps(memory, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
    )


def _node_task(
    spec: AgentSpec,
    *,
    prompt: str,
    branch_index: int,
    node_index: int,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
) -> WorkerTask:
    return WorkerTask(
        task_id=f"{spec.agent_id}:branch:{branch_index + 1}:node:{node_index + 1}",
        # Strategy workers are blind to the operational run identity.  The
        # target structure remains in the bounded objective, but no target
        # name, run id, or evidence handle is serialized in WorkerTask.
        case_id=_opaque_strategy_case_id(spec.run_id),
        task_type="strategic_disconnection_mining",
        required_artifact_type="RetrosynthesisProposalReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=32_000,
            max_tool_calls=0,
            max_worker_runs=1,
            reasoning_effort=reasoning_effort,
        ),
        objective=prompt,
        allowed_workdir=str(spec.metadata.get("allowed_workdir") or "."),
        agent_mode="single",
        codex_auth_mode="ambient_codex_cli",
        model=model,
    )


def _opaque_strategy_case_id(run_id: str) -> str:
    return "strategy-case:" + hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:20]


def _critic_prompt(
    *,
    target: str,
    branch_index: int,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
) -> str:
    route = {
        "schema_version": "blind_route_critic_input.v1",
        "phase": "independent_chemical_critic",
        "campaign_target": target,
        "branch_id": branch_index + 1,
        "strategy_card": dict(strategy_card),
        "steps": [
            {
                "step_id": str(step.get("step_id") or ""),
                "product_smiles": str(step.get("product_smiles") or ""),
                "precursor_smiles": list(step.get("precursor_smiles") or []),
                "transformation_hypothesis": str(
                    step.get("transformation_hypothesis") or ""
                ),
                "reaction_operations": [
                    dict(value)
                    for value in step.get("reaction_operations") or []
                    if isinstance(value, Mapping)
                ],
                "condition_predictions": [
                    dict(value)
                    for value in step.get("condition_predictions") or []
                    if isinstance(value, Mapping)
                ],
                "strategy_anchor": step.get("strategy_anchor") is True,
            }
            for step in steps
            if isinstance(step, Mapping)
        ],
    }
    return "\n".join(
        [
            "Act as an independent senior synthetic chemist and forward-simulate this frozen route.",
            "This also applies to a route-local repair: audit the replacement neighborhood while preserving the frozen route strategy.",
            "You did not design the route. Do not preserve it out of politeness and do not replace its StrategyCard silently.",
            "Audit atom provenance, plausible mechanism, functional-group compatibility, site selectivity, stereochemistry, sequence order, competing pathways, and enzyme identity/capability.",
            "A missing paper is not a chemical rejection. Do not browse or use target-name knowledge; judge only the supplied structures and route contract.",
            "Reject only concrete contradictions. Mark unresolved substrate scope or selectivity as uncertain and name the smallest repair or experiment.",
            "This critique grants no reaction proof, source authority, stock authority, or solved status.",
            "BlindRouteCriticInput:",
            json.dumps(route, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
    )


def _critic_task(
    spec: AgentSpec,
    *,
    prompt: str,
    branch_index: int,
    timeout_s: float,
) -> WorkerTask:
    return WorkerTask(
        task_id=f"critic:{hashlib.sha256((spec.agent_id + ':' + str(branch_index)).encode('utf-8')).hexdigest()[:20]}",
        case_id=_opaque_strategy_case_id(spec.run_id + ":critic"),
        task_type="route_chemistry_critique",
        required_artifact_type="ChemicalStrategyCritique",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=48_000,
            max_tool_calls=0,
            max_worker_runs=1,
            reasoning_effort=str(
                spec.metadata.get("critic_reasoning_effort") or "high"
            ),
        ),
        objective=prompt,
        allowed_workdir=str(spec.metadata.get("allowed_workdir") or "."),
        agent_mode="single",
        codex_auth_mode="ambient_codex_cli",
        model=str(spec.metadata.get("model") or ""),
    )


def _critique_from_record(record: WorkerRunRecord) -> dict[str, Any]:
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    if (
        record.status != "accepted_draft"
        or artifact.get("artifact_type") != "ChemicalStrategyCritique"
        or payload.get("schema_version") != "chemical_strategy_critique.v1"
    ):
        return {
            "schema_version": "chemical_strategy_critique.v1",
            "status": "unavailable",
            "reason": "critic_output_invalid",
            "semantics": {"critic_required_before_evidence": True},
        }
    assessment = str(payload.get("overall_assessment") or "uncertain")
    return {
        **payload,
        "status": assessment,
        "critic_model": str(record.metadata.get("model") or ""),
        "critic_task_id": record.task_id,
        "semantics": {
            "independent_codex_critic": True,
            "runs_before_evidence_acquisition": True,
            "grants_no_reaction_proof": True,
            "grants_no_source_authority": True,
        },
    }


def _expansion_from_record(
    record: WorkerRunRecord,
    *,
    expected_product: str,
    require_strategy_card: bool = False,
    mapped_product_smiles: str = "",
    require_reaction_operations: bool = False,
) -> NodeExpansion | None:
    if record.status != "accepted_draft":
        return None
    payload = dict(dict(record.output_artifact or {}).get("payload") or {})
    candidates = [dict(row) for row in payload.get("candidates") or [] if isinstance(row, Mapping)]
    if len(candidates) != 1:
        return None
    row = candidates[0]
    product = _canonical_smiles(row.get("product_smiles"))
    precursors = tuple(
        dict.fromkeys(
            canonical
            for value in row.get("precursor_smiles") or []
            if (canonical := _canonical_smiles(value))
        )
    )
    if (
        product != expected_product
        or not precursors
        or len(precursors) > 4
        or product in precursors
        or _has_atom_provenance_deficit(product, precursors)
    ):
        return None
    strategy_card = normalize_strategy_card(
        row.get("strategy_card") or {},
        reaction_operations=(
            row.get("reaction_operations") or ()
            if require_strategy_card
            else ()
        ),
    )
    if require_strategy_card and not _valid_strategy_card(strategy_card):
        return None
    operations = normalize_reaction_operations(row.get("reaction_operations") or ())
    if require_reaction_operations and not operations:
        return None
    reactionjson_audit: dict[str, Any] = {}
    if operations:
        if not mapped_product_smiles:
            return None
        try:
            reactionjson_audit = replay_reactionjson(
                mapped_product_smiles=mapped_product_smiles,
                operations=operations,
                expected_precursor_smiles=precursors,
            )
        except ReactionJsonReplayError:
            return None
    return NodeExpansion(
        product_smiles=product,
        precursor_smiles=precursors,
        reaction_family=str(row.get("reaction_family") or "retrosynthetic transformation"),
        rationale=str(row.get("transformation_rationale") or "model-proposed local disconnection"),
        conditions=tuple(str(value) for value in row.get("conditions") or [] if str(value)),
        catalyst=str(row.get("catalyst") or ""),
        enzyme=str(row.get("enzyme") or ""),
        limitations=tuple(str(value) for value in row.get("limitations") or [] if str(value)),
        product_retron_type=str(row.get("product_retron_type") or ""),
        strategy_card=strategy_card,
        reaction_operations=operations,
        reactionjson_audit=reactionjson_audit,
    )


def _expansion_rejection_reason(
    record: WorkerRunRecord,
    *,
    expected_product: str,
    mapped_product_smiles: str,
    require_reaction_operations: bool,
) -> str:
    """Classify a rejected model expansion without weakening the contract."""

    if record.status != "accepted_draft":
        return "worker_output_not_accepted"
    payload = dict(dict(record.output_artifact or {}).get("payload") or {})
    candidates = [
        dict(row)
        for row in payload.get("candidates") or []
        if isinstance(row, Mapping)
    ]
    if len(candidates) != 1:
        return "candidate_count_invalid"
    row = candidates[0]
    product = _canonical_smiles(row.get("product_smiles"))
    if product != expected_product:
        return "product_mismatch"
    operations = normalize_reaction_operations(row.get("reaction_operations") or ())
    if require_reaction_operations and not operations:
        return "strategy_graph_edit_missing"
    if operations:
        precursors = tuple(
            dict.fromkeys(
                canonical
                for value in row.get("precursor_smiles") or []
                if (canonical := _canonical_smiles(value))
            )
        )
        try:
            replay_reactionjson(
                mapped_product_smiles=mapped_product_smiles,
                operations=operations,
                expected_precursor_smiles=precursors,
            )
        except ReactionJsonReplayError:
            return "strategy_graph_edit_replay_failed"
    return "invalid_expansion_contract"


def _step_row(
    expansion: NodeExpansion,
    *,
    step_id: str,
    strategy_anchor: bool = False,
) -> dict[str, Any]:
    strategy_card = normalize_strategy_card(
        expansion.strategy_card or {},
        reaction_operations=expansion.reaction_operations,
    )
    reagents = list(expansion.conditions) or ["reaction-class screen"]
    condition = {
        "reagents": reagents,
        "catalyst": expansion.catalyst,
        "enzyme": expansion.enzyme,
        "solvent": "screen",
        "temperature": "screen",
        "time": "screen",
        "authority_scope": "model_predicted_condition",
        "not_reaction_proof": True,
    }
    return {
        "step_id": step_id,
        "product_smiles": expansion.product_smiles,
        "precursor_smiles": list(expansion.precursor_smiles),
        "transformation_hypothesis": expansion.reaction_family,
        "strategic_role": expansion.rationale,
        "source_hints": [],
        "required_validation": ["structure", "reaction_feasibility"],
        "hypothesis_only": True,
        "condition_predictions": [condition],
        "limitations": list(expansion.limitations),
        "strategy_card": strategy_card,
        "reaction_operations": [dict(row) for row in expansion.reaction_operations],
        "reactionjson_audit": dict(expansion.reactionjson_audit or {}),
        "strategy_id": str(strategy_card.get("strategy_id") or ""),
        "strategy_digest": str(strategy_card.get("strategy_digest") or ""),
        "execution_domain": str(
            strategy_card.get("execution_domain") or "chemical"
        ),
        "strategy_anchor": bool(strategy_anchor),
    }


def _compile_plan(
    context: CampaignContext,
    *,
    mode: str,
    branches: list[dict[str, Any]],
    requested_branch_count: int,
) -> dict[str, Any]:
    target = _canonical_smiles(context.target.get("canonical_smiles"))
    families: list[dict[str, Any]] = []
    skeletons: list[dict[str, Any]] = []
    disconnections: list[dict[str, Any]] = []
    leaf_families: dict[str, set[str]] = {}
    for ordinal, branch in enumerate(branches, start=1):
        family_id = f"codex:sequential:family:{ordinal}"
        lens = str(branch.get("lens") or f"strategy branch {ordinal}")
        strategy_card = dict(branch.get("strategy_card") or {})
        steps = [dict(row) for row in branch.get("steps") or []]
        families.append(
            {
                "route_family_id": family_id,
                "title": f"Sequential strategy {ordinal}",
                "strategy": lens,
                "target_smiles": target,
                "advantages": [
                    "strategy-first root analysis",
                    "continuous node-level policy expansion",
                ],
                "risks": ["host validation and stock closure pending"],
                "diversity_basis": _strategy_signature(strategy_card) or lens,
                "strategy_card": strategy_card,
                "strategy_id": str(strategy_card.get("strategy_id") or ""),
                "strategy_digest": str(strategy_card.get("strategy_digest") or ""),
                "execution_domain": str(
                    strategy_card.get("execution_domain") or "chemical"
                ),
                "chemical_critic": dict(branch.get("chemical_critic") or {}),
            }
        )
        skeletons.append(
            {
                "skeleton_id": f"codex:sequential:skeleton:{ordinal}",
                "route_family_id": family_id,
                "summary": (
                    f"{len(steps)} accepted node expansions from "
                    f"{int(branch.get('call_count') or len(steps))} compact calls"
                ),
                "steps": steps,
                "chemical_critic": dict(branch.get("chemical_critic") or {}),
            }
        )
        if steps:
            disconnections.append(
                {
                    "disconnection_id": f"codex:sequential:disconnect:{ordinal}",
                    "route_family_id": family_id,
                    "proposal_id": str(steps[0].get("step_id") or ""),
                    "rationale": str(steps[0].get("strategic_role") or lens),
                }
            )
        for leaf in branch.get("open_leaves") or []:
            canonical = _canonical_smiles(leaf)
            if canonical:
                leaf_families.setdefault(canonical, set()).add(family_id)
    shared = []
    priorities = []
    for index, (leaf, family_ids) in enumerate(sorted(leaf_families.items()), start=1):
        intermediate_id = f"codex:sequential:open-leaf:{index}"
        shared.append(
            {
                "intermediate_id": intermediate_id,
                "smiles": leaf,
                "route_family_ids": sorted(family_ids),
                "strategic_role": "distinct open leaf requiring standard short-tail search",
            }
        )
        priorities.append(
            {
                "priority_id": f"codex:sequential:priority:{index}",
                "proposal_id": intermediate_id,
                "target_smiles": leaf,
                "route_family_ids": sorted(family_ids),
                "provider_preferences": ["chemenzy"],
                "retron_hints": [],
                "priority": max(1.0, 100.0 - index),
                "rationale": "run depth-6, 500-iteration, 1200-second short-tail search",
            }
        )
    body = {
        "schema_version": "global_campaign_plan.v1",
        "plan_id": "plan:sequential:"
        + _digest(
            {
                "context": context.content_sha256,
                "mode": mode,
                "steps": [row["steps"] for row in skeletons],
            }
        )[:20],
        "run_id": context.run_id,
        "mode": mode,
        "context_sha256": context.content_sha256,
        "graph_revision": context.revision.graph_revision,
        "route_families": families,
        "multi_step_skeletons": skeletons,
        "strategic_disconnections": disconnections,
        "shared_intermediates": shared,
        "critical_unknowns": [
            {
                "unknown_id": "codex:sequential:unknown:validation",
                "description": "reaction feasibility remains host-validated",
            },
            {
                "unknown_id": "codex:sequential:unknown:stock",
                "description": "all open leaves require the bound stock oracle",
            },
        ],
        "source_plan": [],
        "fallback_strategies": [
            {
                "fallback_id": "codex:sequential:fallback:local-repair",
                "trigger": "one materialized edge fails host validation",
                "action": "repair only that reaction neighborhood for at most six rounds",
            }
        ],
        "frontier_priorities": priorities,
        "pivot_conditions": [
            {
                "pivot_id": "codex:sequential:pivot:closure",
                "condition": "a target-rooted route exists",
                "action": "materialize, validate, and audit stock before new expansion",
            }
        ],
        "stop_conditions": [
            {
                "stop_id": "codex:sequential:stop:paper-equivalent",
                "condition": "one connected route has every leaf in the same bound stock",
            }
        ],
        "portfolio_rationale": (
            f"{len(families)}/{requested_branch_count} independent sequential Codex "
            "branches; open leaves are delegated individually to the standard short-tail search."
        ),
        "strategy_cards": [dict(row.get("strategy_card") or {}) for row in families],
        "strategy_domains": sorted(
            {
                str(dict(row.get("strategy_card") or {}).get("execution_domain") or "chemical")
                for row in families
            }
        ),
        "limitations": [
            "model steps are hypotheses until host materialization and validation",
            "paper-equivalent stock closure is reported independently of evidence and conditions",
        ],
    }
    return body


def _repair_neighborhood(
    context: CampaignContext,
    *,
    target: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    topology = dict(context.topology or {})
    edges = [dict(row) for row in dict(topology.get("edges") or {}).values()]
    failed = next(
        (
            row
            for row in edges
            if dict(row.get("reaction_validation") or {}).get("accepted") is not True
            and _canonical_smiles(row.get("product_smiles"))
        ),
        {},
    )
    product = _canonical_smiles(failed.get("product_smiles")) or target
    # Preserve only a deterministically connected target-to-product prefix.
    by_product = {
        _canonical_smiles(row.get("product_smiles")): row
        for row in edges
        if _canonical_smiles(row.get("product_smiles"))
    }
    prefix: list[dict[str, Any]] = []
    current = target
    visited: set[str] = set()
    while current and current != product and current not in visited:
        visited.add(current)
        edge = by_product.get(current)
        if not edge:
            break
        precursors = [
            canonical
            for value in edge.get("precursor_smiles") or []
            if (canonical := _canonical_smiles(value))
        ]
        if product not in precursors and not any(value in by_product for value in precursors):
            break
        prefix.append(
            _step_row(
                NodeExpansion(
                    product_smiles=current,
                    precursor_smiles=tuple(precursors),
                    reaction_family="preserved host route prefix",
                    rationale="unchanged upstream edge during local repair",
                ),
                step_id=f"codex:repair:preserved:{len(prefix) + 1}",
            )
        )
        current = (
            product
            if product in precursors
            else next((value for value in precursors if value in by_product), "")
        )
    feedback = {
        "failed_product_smiles": product,
        "failed_precursor_smiles": [
            _canonical_smiles(value)
            for value in failed.get("precursor_smiles") or []
            if _canonical_smiles(value)
        ],
        "reaction_validation": dict(failed.get("reaction_validation") or {}),
        "failure_reasons": list(dict(failed.get("reaction_validation") or {}).get("reasons") or []),
    }
    return product, prefix, feedback


def _aggregate_usage(
    records: Iterable[WorkerRunRecord],
    *,
    elapsed_s: float,
) -> dict[str, int | float]:
    rows = list(records)
    result: dict[str, int | float] = {
        "model_invocations": len(rows),
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "accepted_expansions": sum(row.status == "accepted_draft" for row in rows),
        "attempt_runs": len(rows),
        "wall_time_s": max(0.0, float(elapsed_s)),
    }
    for record in rows:
        usage = dict(record.usage or {})
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        ):
            result[key] = int(result[key]) + max(0, int(usage.get(key) or 0))
    return result


def _agent_result(
    spec: AgentSpec,
    *,
    state: AgentState,
    output: Mapping[str, Any] | None,
    usage: Mapping[str, Any],
    error: str,
    mode: str,
) -> AgentResult:
    return AgentResult(
        run_id=spec.run_id,
        agent_id=spec.agent_id,
        parent_agent_id=spec.parent_agent_id,
        attempt=spec.attempt,
        idempotency_key=f"{spec.idempotency_key}:sequential-result",
        context_hash=spec.context_hash,
        capabilities=spec.capabilities,
        write_scope=spec.write_scope,
        budget=spec.budget,
        state=state,
        output=dict(output) if output is not None else None,
        error=error,
        usage=dict(usage),
        metadata={
            "backend": "sequential_codex_policy",
            "mode": mode,
            "direct_child": True,
            "compact_per_node_context": True,
        },
    )


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    # Atom maps belong to the separate ReactionJSON edit contract.  Route
    # identity and precursor comparisons must remain map-invariant; replay
    # receives a freshly mapped product through ``_mapped_smiles``.
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "NodeExpansion",
    "SEQUENTIAL_STRATEGY_SEARCH_SCHEMA",
    "SequentialStrategyDirectorRunner",
]
