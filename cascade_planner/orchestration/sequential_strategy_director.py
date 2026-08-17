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
    diagnose_reactionjson,
    replay_reactionjson,
)
from cascade_planner.application.strategy_contract import (
    normalize_reaction_operations,
    normalize_strategy_card,
    normalize_strategy_policy_card,
    reaction_edit_digest,
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
_EDITOR_INPUT_TOKEN_RESERVE = 20_000
_EDITOR_OUTPUT_TOKEN_RESERVE = 8_000
_CRITIC_EDITOR_WALL_FRACTION = 0.30
_MAX_DEADLINE_SETTLEMENT_RESERVE_S = 1.0
_STRATEGY_SEED_RETRY_LIMIT = 3
_MATERIALIZATION_RETRY_LIMIT = 3
_CONDITION_PLACEHOLDER_MARKERS = (
    "to be determined",
    "determine after",
    "not specified",
    "not applicable",
    "n/a",
    "tbd",
    "screen",
    "screening",
    "as needed",
)


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
        editor_executor: NodeExecutor | None = None,
        stock_membership: StockMembership | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.node_executor = node_executor or self._execute_node
        self.critic_executor = critic_executor or self.node_executor
        self.editor_executor = editor_executor or self.node_executor
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
        usage = _aggregate_usage(records, elapsed_s=time.monotonic() - started)
        usage["critic_unavailable_branch_count"] = sum(
            bool(branch.get("steps"))
            and str(dict(branch.get("chemical_critic") or {}).get("status") or "")
            == "unavailable"
            for branch in branches
        )
        usage["critic_rejected_branch_count"] = sum(
            str(dict(branch.get("chemical_critic") or {}).get("status") or "")
            == "reject"
            for branch in branches
        )
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
            not in {"reject", "unavailable"}
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
            usage["strategy_seed_retry_limit"] = _STRATEGY_SEED_RETRY_LIMIT
            usage["materialization_retry_limit"] = _MATERIALIZATION_RETRY_LIMIT
            usage["retained_strategy_hypotheses"] = [
                dict(branch.get("strategy_card") or {})
                for branch in branches
                if branch.get("strategy_card")
            ]
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
        """Run the paper-style Codex Critic -> Editor loop.

        Critic output is deliberately non-authoritative, but it is a required
        control step.  A missing Critic or Editor therefore fails closed rather
        than silently promoting an unreviewed route to the campaign plan.
        """

        target = _canonical_smiles(context.target.get("canonical_smiles"))
        max_rounds = max(0, int(config.max_route_local_repair_rounds))
        for index, branch in enumerate(branches):
            if not branch.get("steps"):
                continue
            branch.setdefault("critic_editor_history", [])
            for iteration in range(max_rounds + 1):
                remaining_critics = sum(
                    bool(row.get("steps"))
                    and not dict(row.get("chemical_critic") or {}).get("status")
                    for row in branches[index:]
                )
                # Keep enough invocation and token budget for one Critic and
                # one possible Editor for every later route family.
                future_families = max(0, remaining_critics - 1)
                if not _node_budget_allows(
                    records,
                    started=started,
                    quota=quota,
                    reserve_model_invocations=future_families * 2,
                    reserve_input_tokens=future_families
                    * (_CRITIC_INPUT_TOKEN_RESERVE + _EDITOR_INPUT_TOKEN_RESERVE),
                    reserve_output_tokens=future_families
                    * (_CRITIC_OUTPUT_TOKEN_RESERVE + _EDITOR_OUTPUT_TOKEN_RESERVE),
                ):
                    branch["chemical_critic"] = _unavailable_critique(
                        "critic_budget_exhausted"
                    )
                    break
                remaining_wall = _remaining_node_wall_time(started, quota)
                per_critic_wall = min(
                    config.critic_call_timeout_s,
                    remaining_wall / max(1, remaining_critics + 1),
                )
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
                    iteration=iteration,
                    timeout_s=max(0.001, per_critic_wall),
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
                    branch["chemical_critic"] = _unavailable_critique(
                        f"critic_execution_failed:{type(exc).__name__}"
                    )
                    break
                records.append(record)
                critique = _critique_from_record(record)
                branch["chemical_critic"] = critique
                branch["critic_editor_history"].append(
                    {
                        "iteration": iteration,
                        "critic": dict(critique),
                    }
                )
                if str(critique.get("status") or "") == "unavailable":
                    break
                blocking_step = _blocking_critic_step(
                    critique,
                    list(branch.get("steps") or []),
                )
                if blocking_step is None:
                    break
                if iteration >= max_rounds:
                    branch["chemical_critic"] = {
                        **critique,
                        "status": "reject",
                        "reason": "critic_editor_iteration_limit_reached",
                    }
                    break
                edited = self._edit_branch_from_critique(
                    spec,
                    target=target,
                    branch=branch,
                    blocking_step=blocking_step,
                    critique=critique,
                    iteration=iteration,
                    records=records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    max_node_expansions_per_branch=config.max_node_expansions_per_branch,
                    require_strategy_graph_edits=config.require_strategy_graph_edits,
                    quota=quota,
                    started=started,
                )
                if not edited:
                    branch["chemical_critic"] = _unavailable_critique(
                        "editor_execution_failed"
                    )
                    break
        return branches

    def _edit_branch_from_critique(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branch: dict[str, Any],
        blocking_step: Mapping[str, Any],
        critique: Mapping[str, Any],
        iteration: int,
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        max_node_expansions_per_branch: int,
        require_strategy_graph_edits: bool,
        quota: _NodeCallBudget,
        started: float,
    ) -> bool:
        steps = [dict(row) for row in branch.get("steps") or []]
        step_id = str(blocking_step.get("step_id") or "")
        try:
            step_index = next(
                index for index, row in enumerate(steps)
                if str(row.get("step_id") or "") == step_id
            )
        except StopIteration:
            return False
        selected_product = _canonical_smiles(blocking_step.get("product_smiles"))
        if not selected_product:
            return False
        prefix = steps[:step_index]
        feedback = _compact_critic_feedback(
            critique,
            blocking_step,
        )
        rejection = {
            "phase": "critic_editor",
            "step_id": step_id,
            "product_smiles": selected_product,
            "failure_reasons": list(blocking_step.get("reasons") or [])[:8],
            "repair_actions": list(feedback.get("repair_actions") or [])[:6],
        }
        prompt = ""
        # Critic payloads can be much larger than a route node payload.  Keep
        # the Editor context surgical and progressively drop old prefix rows;
        # a prompt-size failure must never abort the whole campaign action.
        for prefix_view, lens in (
            (prefix, "Codex Editor: surgical repair preserving the StrategyCard"),
            (prefix[-3:], "Codex Editor: surgical repair"),
            (prefix[-1:], "Codex Editor: compact surgical repair"),
            ((), "Codex Editor: minimal surgical repair"),
        ):
            candidate_prompt = _node_prompt(
                target=target,
                branch_index=int(branch.get("branch_index") or 0),
                lens=lens,
                selected_product=selected_product,
                steps=prefix_view,
                open_leaves=[selected_product],
                prior_rejections=[rejection],
                repair=True,
                strategy_card=dict(branch.get("strategy_card") or {}),
                forbidden_strategy_cards=(),
                host_failure_feedback=feedback,
            )
            if len(candidate_prompt.encode("utf-8")) <= max_prompt_bytes:
                prompt = candidate_prompt
                break
        if not prompt:
            return False
        if not _node_budget_allows(records, started=started, quota=quota):
            return False
        task = _node_task(
            spec,
            prompt=prompt,
            branch_index=int(branch.get("branch_index") or 0),
            node_index=iteration + 1,
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(
                spec.metadata.get("editor_reasoning_effort") or "high"
            ),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
            task_type="route_chemistry_edit",
        )
        try:
            record = self.editor_executor(task)
        except Exception:
            return False
        records.append(record)
        expansion = _expansion_from_record(
            record,
            expected_product=selected_product,
            mapped_product_smiles=_mapped_smiles(selected_product),
            require_reaction_operations=require_strategy_graph_edits,
        )
        if expansion is None:
            return False
        expansion = replace(
            expansion,
            strategy_card=dict(branch.get("strategy_card") or {}),
        )
        edited_step = _step_row(
            expansion,
            step_id=step_id or f"codex:editor:{iteration + 1}:1",
            strategy_anchor=step_index == 0,
        )
        old_depth = len(steps)
        branch["steps"] = [*prefix, edited_step]
        branch["expanded_products"] = {
            _canonical_smiles(row.get("product_smiles"))
            for row in branch["steps"]
            if _canonical_smiles(row.get("product_smiles"))
        }
        branch["open_leaves"] = deque(
            dict.fromkeys(expansion.precursor_smiles)
        )
        branch["route_call_count"] = int(branch.get("route_call_count") or 0) + 1
        branch["call_count"] = int(branch.get("call_count") or 0) + 1
        branch.setdefault("editor_repairs", []).append(
            {
                "iteration": iteration,
                "step_id": edited_step["step_id"],
                "replaced_step_index": step_index,
                "old_route_depth": old_depth,
                "new_route_depth": len(branch["steps"]),
                "editor_task_id": task.task_id,
            }
        )
        # Rebuild the discarded downstream suffix before the next Critic pass.
        target_depth = max(old_depth, len(branch["steps"]))
        while (
            branch.get("open_leaves")
            and len(branch.get("steps") or []) < target_depth
            and int(branch.get("route_call_count") or 0)
            < max(1, int(max_node_expansions_per_branch))
            and _node_budget_allows(records, started=started, quota=quota)
        ):
            before = len(branch.get("steps") or [])
            self._expand_one_branch_node(
                spec,
                target=target,
                branch=branch,
                records=records,
                max_prompt_bytes=max_prompt_bytes,
                max_node_call_timeout_s=max_node_call_timeout_s,
                require_strategy_graph_edits=require_strategy_graph_edits,
                quota=quota,
                started=started,
            )
            if len(branch.get("steps") or []) <= before:
                break
        branch["complete_in_bound_stock"] = not bool(branch.get("open_leaves"))
        return True

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
                "strategy_call_count": 0,
                "route_call_count": 0,
                "rejections": [],
                "materialization_failures": {},
                "complete_in_bound_stock": False,
                "strategy_card": {},
                "chemical_critic": {},
            }
            for branch_index in range(config.strategy_branch_count)
        ]
        records: list[WorkerRunRecord] = []

        # Phase 1 records strategy hypotheses only.  It deliberately does not
        # ask for precursor structures or ReactionJSON; those belong to the
        # Route Builder boundary below.  A graph-edit failure must therefore
        # never erase an already selected strategic hypothesis.
        while not self._cancelled() and any(
            not branch["strategy_card"] for branch in branches
        ):
            progressed = False
            for branch in branches:
                if branch["strategy_card"]:
                    continue
                if self._cancelled() or not _node_budget_allows(
                    records, started=started, quota=quota
                ):
                    break
                if (
                    int(branch["strategy_call_count"])
                    >= _STRATEGY_SEED_RETRY_LIMIT
                ):
                    continue
                self._seed_one_branch_strategy(
                    spec,
                    target=target,
                    branch=branch,
                    records=records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    quota=quota,
                    started=started,
                    forbidden_strategy_cards=_accepted_strategy_cards(
                        branches, exclude_index=int(branch["branch_index"])
                    ),
                )
                progressed = True
            if not progressed or not _node_budget_allows(records, started=started, quota=quota):
                break

        seeded = [branch for branch in branches if branch["strategy_card"]]

        # Phase 2 expands the already committed strategies round-robin.  Route
        # state remains isolated; only compact StrategyCard signatures are
        # shared to enforce portfolio orthogonality.
        critic_slots = len(seeded)
        # The paper's Critic/Editor stage is a required phase, not spare-time
        # validation.  Protect one Critic and one possible Editor call per
        # route family plus a fixed fraction of the director wall clock.
        critic_editor_call_reserve = critic_slots * 2
        critic_editor_wall_reserve = (
            quota.wall_time_s * _CRITIC_EDITOR_WALL_FRACTION
            if critic_slots
            else 0.0
        )
        critic_input_reserve = critic_slots * (
            _CRITIC_INPUT_TOKEN_RESERVE + _EDITOR_INPUT_TOKEN_RESERVE
        )
        critic_output_reserve = critic_slots * (
            _CRITIC_OUTPUT_TOKEN_RESERVE + _EDITOR_OUTPUT_TOKEN_RESERVE
        )
        while not self._cancelled():
            progressed = False
            for branch in branches:
                if self._cancelled() or not _node_budget_allows(
                    records,
                    started=started,
                    quota=quota,
                    reserve_model_invocations=critic_editor_call_reserve,
                    reserve_input_tokens=critic_input_reserve,
                    reserve_output_tokens=critic_output_reserve,
                    reserve_wall_time_s=critic_editor_wall_reserve,
                ):
                    break
                if (
                    int(branch["route_call_count"])
                    >= config.max_node_expansions_per_branch
                    or not branch["open_leaves"]
                    or not branch["strategy_card"]
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
                )
                progressed = True
                if branch["steps"] and not branch["open_leaves"]:
                    branch["complete_in_bound_stock"] = True
            if seeded and all(
                branch["complete_in_bound_stock"]
                or not branch["open_leaves"]
                for branch in seeded
            ):
                # Stock closure ends Route Builder expansion only.  Critic ->
                # Editor remains a mandatory phase before any branch can be
                # promoted to the global plan, including routes that happen
                # to close at the first replayable disconnection.
                break
            if not progressed or not _node_budget_allows(
                records,
                started=started,
                quota=quota,
                reserve_model_invocations=critic_editor_call_reserve,
                reserve_input_tokens=critic_input_reserve,
                reserve_output_tokens=critic_output_reserve,
                reserve_wall_time_s=critic_editor_wall_reserve,
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
                timeout_s=_node_call_timeout_s(
                    started,
                    quota,
                    maximum=config.max_node_call_timeout_s,
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

    def _seed_one_branch_strategy(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branch: dict[str, Any],
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        quota: _NodeCallBudget,
        started: float,
        forbidden_strategy_cards: Iterable[Mapping[str, Any]],
    ) -> None:
        branch_index = int(branch["branch_index"])
        branch["strategy_call_count"] = int(branch["strategy_call_count"]) + 1
        branch["call_count"] = int(branch["call_count"]) + 1
        prompt = _strategy_prompt(
            target=target,
            branch_index=branch_index,
            lens=str(branch["lens"]),
            forbidden_strategy_cards=forbidden_strategy_cards,
            prior_rejections=branch["rejections"],
        )
        _assert_node_prompt_size(prompt, max_prompt_bytes)
        task = _strategy_task(
            spec,
            prompt=prompt,
            branch_index=branch_index,
            attempt_index=int(branch["strategy_call_count"]),
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(
                spec.metadata.get("reasoning_effort") or "medium"
            ),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
        )
        record = self.node_executor(task)
        records.append(record)
        card = _strategy_card_from_record(record, expected_target=target)
        forbidden = [dict(row) for row in forbidden_strategy_cards]
        if card is None:
            branch["rejections"].append(
                {
                    "phase": "strategy_generator",
                    "attempt": int(branch["strategy_call_count"]),
                    "reason": _strategy_card_rejection_reason(
                        record, expected_target=target
                    ),
                }
            )
            return
        if _strategy_conflicts(card, forbidden):
            branch["rejections"].append(
                {
                    "phase": "strategy_generator",
                    "attempt": int(branch["strategy_call_count"]),
                    "reason": "root_strategy_not_orthogonal",
                    "strategy_signature": _strategy_signature(card),
                    "rejected_strategy_card": card,
                }
            )
            return
        branch["strategy_card"] = card
        branch["strategy_seed"] = _strategy_title_from_card(card)
        branch["lens"] = "Codex-authored strategy - " + str(
            branch["strategy_seed"]
        )

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
        branch["route_call_count"] = int(branch["route_call_count"]) + 1
        branch["call_count"] = int(branch["call_count"]) + 1
        call_index = int(branch["route_call_count"])
        is_strategy_anchor = not steps and selected == target
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
            forbidden_strategy_cards=(),
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
                forbidden_strategy_cards=(),
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
                forbidden_strategy_cards=(),
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
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
        )
        record = self.node_executor(task)
        records.append(record)
        expansion = _expansion_from_record(
            record,
            expected_product=selected,
            mapped_product_smiles=_mapped_smiles(selected),
            require_reaction_operations=bool(require_strategy_graph_edits),
        )
        if expansion is None or any(
            precursor in expanded_products for precursor in expansion.precursor_smiles
        ):
            diagnostic = (
                _expansion_rejection_diagnostic(
                    record,
                    expected_product=selected,
                    mapped_product_smiles=_mapped_smiles(selected),
                    require_reaction_operations=bool(require_strategy_graph_edits),
                )
                if expansion is None
                else {"reason": "ancestor_cycle"}
            )
            rejection_reason = str(diagnostic.get("reason") or "invalid_expansion_contract")
            rejected.append(
                {
                    "phase": "route_builder",
                    "node": call_index,
                    "product_smiles": selected,
                    "reason": rejection_reason,
                    "replay_diagnostic": diagnostic,
                }
            )
            if rejection_reason in {
                "strategy_graph_edit_missing",
                "strategy_graph_edit_replay_failed",
                "invalid_expansion_contract",
            }:
                failures = dict(branch.get("materialization_failures") or {})
                graph_edit_rejections = int(failures.get(selected) or 0) + 1
                failures[selected] = graph_edit_rejections
                branch["materialization_failures"] = failures
                if graph_edit_rejections >= _MATERIALIZATION_RETRY_LIMIT:
                    rejected.append(
                        {
                            "phase": "route_builder",
                            "node": call_index,
                            "product_smiles": selected,
                            "reason": "materialization_retry_limit_reached",
                            "strategy_retained": True,
                        }
                    )
                    branch.setdefault("blocked_materializations", []).append(selected)
                    branch["open_leaves"] = deque(open_leaves)
                    return
            open_leaves.append(selected)
            return
        if branch.get("strategy_card"):
            expansion = replace(
                expansion,
                strategy_card=dict(branch.get("strategy_card") or {}),
            )
        expanded_products.add(selected)
        steps.append(
            _step_row(
                expansion,
                step_id=f"codex:branch:{branch_index + 1}:{len(steps) + 1}",
                strategy_anchor=is_strategy_anchor,
            )
        )
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
        else config.strategy_branch_count
        * (
            config.max_node_expansions_per_branch
            + 2
            + 2 * config.max_route_local_repair_rounds
        )
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
        + _deadline_settlement_reserve_s(quota)
    )


def _remaining_node_wall_time(started: float, quota: _NodeCallBudget) -> float:
    return max(0.0, quota.wall_time_s - (time.monotonic() - started))


def _deadline_settlement_reserve_s(quota: _NodeCallBudget) -> float:
    return min(
        _MAX_DEADLINE_SETTLEMENT_RESERVE_S,
        max(0.001, quota.wall_time_s * 0.001),
    )


def _node_call_timeout_s(
    started: float,
    quota: _NodeCallBudget,
    *,
    maximum: float,
) -> float:
    usable = (
        _remaining_node_wall_time(started, quota)
        - _deadline_settlement_reserve_s(quota)
    )
    return max(0.001, min(float(maximum), usable))


def _public_branch(branch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_index": int(branch.get("branch_index") or 0),
        "lens": str(branch.get("lens") or ""),
        "strategy_seed": str(branch.get("strategy_seed") or ""),
        "strategy_card": dict(branch.get("strategy_card") or {}),
        "steps": [dict(row) for row in branch.get("steps") or []],
        "open_leaves": list(branch.get("open_leaves") or []),
        "call_count": int(branch.get("call_count") or 0),
        "strategy_call_count": int(branch.get("strategy_call_count") or 0),
        "route_call_count": int(branch.get("route_call_count") or 0),
        "complete_in_bound_stock": bool(branch.get("complete_in_bound_stock")),
        "rejections": [dict(row) for row in branch.get("rejections") or []],
        "blocked_materializations": list(
            branch.get("blocked_materializations") or []
        ),
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


def _target_bond_pairs(smiles: str) -> set[tuple[int, int]]:
    """Return the atom-map pairs that are actual bonds in ``smiles``.

    StrategyCards are generated against ``campaign_target_mapped``.  A model
    may still emit plausible-looking map numbers that are not present in the
    immutable target graph; accepting those cards poisons every downstream
    node.  This helper intentionally uses the host graph as the authority.
    """

    mapped = Chem.MolFromSmiles(_mapped_smiles(smiles))
    if mapped is None:
        return set()
    pairs: set[tuple[int, int]] = set()
    for bond in mapped.GetBonds():
        left = int(bond.GetBeginAtom().GetAtomMapNum() or 0)
        right = int(bond.GetEndAtom().GetAtomMapNum() or 0)
        if left and right:
            pairs.add(tuple(sorted((left, right))))
    return pairs


def _parse_strategy_bond_pair(value: Any) -> tuple[int, int] | None:
    """Parse one compact ``key_bond_changes`` entry without trusting prose."""

    numbers = [int(raw) for raw in re.findall(r"\d+", str(value or ""))]
    if len(numbers) != 2 or numbers[0] == numbers[1]:
        return None
    return tuple(sorted((numbers[0], numbers[1])))


def _strategy_card_bonds_match_target(
    card: Mapping[str, Any], *, target_smiles: str
) -> bool:
    """Require every strategic forward bond to exist in the target graph."""

    target_pairs = _target_bond_pairs(target_smiles)
    if not target_pairs:
        return False
    values = card.get("key_bond_changes") or []
    parsed = [_parse_strategy_bond_pair(value) for value in values]
    return bool(parsed) and all(pair in target_pairs for pair in parsed)


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


def _strategy_title_from_card(card: Mapping[str, Any]) -> str:
    key_step = str(card.get("key_forward_transformation") or "").strip()
    basis = str(card.get("orthogonality_basis") or "").strip()
    if key_step:
        return f"{key_step}: {basis}" if basis else key_step
    return str(card.get("skeleton_change_class") or "strategy hypothesis")


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


def _strategy_prompt(
    *,
    target: str,
    branch_index: int,
    lens: str,
    forbidden_strategy_cards: Iterable[Mapping[str, Any]],
    prior_rejections: Iterable[Mapping[str, Any]],
) -> str:
    context = {
        "schema_version": "blind_strategy_generator_input.v1",
        "phase": "strategy_generator",
        "campaign_target": target,
        "campaign_target_mapped": _mapped_smiles(target),
        "campaign_target_profile": _structure_profile(target),
        "campaign_target_bond_pairs": [
            f"map {left}-map {right}"
            for left, right in sorted(_target_bond_pairs(target))
        ],
        "branch_id": branch_index + 1,
        "strategy_lens": lens,
        "forbidden_root_strategies": [
            {
                "key_bond_signature": list(
                    dict(card).get("key_bond_signature") or []
                ),
                "topology_signature": str(
                    dict(card).get("topology_signature") or ""
                ),
                "execution_domain": str(
                    dict(card).get("execution_domain") or ""
                ),
            }
            for card in forbidden_strategy_cards
        ],
        "prior_strategy_rejections": [
            dict(row)
            for row in prior_rejections
            if row.get("phase") == "strategy_generator"
        ][-3:],
    }
    return "\n".join(
        [
            "Act only as the Strategy Generator for one blind retrosynthesis branch.",
            "Do not build a route, propose precursor SMILES, write ReactionJSON, predict conditions, search literature, or use stock availability.",
            "Compare at least three materially distinct strategies on scaffold/ring topology, the key forward construction, functional-group and protection conflicts, stereochemical construction, convergence, and expected decomplexification.",
            "Select one strategy satisfying strategy_lens. It must be anchored on a route-defining one-to-two-step construction rather than a cosmetic FGI, protection, redox, nitration, halogenation, or methylation.",
            "For a key forward C-C, C-N, C-O, or other skeletal bond already present in the target, write key_bond_changes with mapped atom pairs exactly as map i-map j using campaign_target_mapped. Every pair must be an actual bond in campaign_target_mapped; do not invent atom indices, describe a future precursor bond, or use a map pair that is absent from the target graph.",
            "A biological strategy must name a chemically credible substrate-product transformation class; enzyme discovery and identity verification happen later.",
            "The selected strategy must be structurally orthogonal to forbidden_root_strategies, not just renamed or assigned different reagents.",
            "Return one StrategyCardReport. This artifact is a durable hypothesis but grants no route, reaction, evidence, stock, or solved authority.",
            "BlindStrategyGeneratorInput:",
            json.dumps(
                context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )


def _strategy_task(
    spec: AgentSpec,
    *,
    prompt: str,
    branch_index: int,
    attempt_index: int,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
) -> WorkerTask:
    return WorkerTask(
        task_id=(
            f"{spec.agent_id}:branch:{branch_index + 1}:strategy:{attempt_index}"
        ),
        case_id=_opaque_strategy_case_id(spec.run_id),
        task_type="strategic_disconnection_mining",
        required_artifact_type="StrategyCardReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=20_000,
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


def _strategy_card_from_record(
    record: WorkerRunRecord,
    *,
    expected_target: str,
) -> dict[str, Any] | None:
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    if (
        record.status != "accepted_draft"
        or artifact.get("artifact_type") != "StrategyCardReport"
        or payload.get("schema_version") != "strategy_card_report.v1"
        or _canonical_smiles(payload.get("target_smiles")) != expected_target
    ):
        return None
    card = normalize_strategy_card(payload.get("strategy_card") or {})
    if not _valid_strategy_card(card):
        return None
    if not _strategy_card_bonds_match_target(card, target_smiles=expected_target):
        return None
    return card


def _strategy_card_rejection_reason(
    record: WorkerRunRecord, *, expected_target: str
) -> str:
    """Return a stable diagnostic for a rejected StrategyCard artifact."""

    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    if record.status != "accepted_draft":
        return "strategy_worker_not_accepted_draft"
    if artifact.get("artifact_type") != "StrategyCardReport":
        return "strategy_card_artifact_type_invalid"
    if payload.get("schema_version") != "strategy_card_report.v1":
        return "strategy_card_schema_invalid"
    if _canonical_smiles(payload.get("target_smiles")) != expected_target:
        return "strategy_card_target_mismatch"
    card = normalize_strategy_card(payload.get("strategy_card") or {})
    if not _valid_strategy_card(card):
        return "strategy_card_fields_invalid"
    if not _strategy_card_bonds_match_target(card, target_smiles=expected_target):
        return "strategy_key_bond_not_in_target"
    return "strategy_card_output_invalid"


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
    memory = {
        "schema_version": "compact_retrosynthesis_branch_context.v1",
        "phase": "local_repair" if repair else "route_builder",
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
    phase_instructions = [
        "Act as the Route Builder and execute the supplied immutable StrategyCard; do not silently replace its key construction with an easier functional-group-interconversion route. The host binds the card, so do not echo it in the candidate.",
        "Compare at least three local disconnections internally on strategy alignment, skeletal simplification, chemoselectivity, stereochemical compatibility, and precursor accessibility, then return only the best candidate.",
        "Stock availability is an endpoint test, never a chemical justification for a disconnection.",
    ]
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
            "Write the ordered candidate.reaction_operations first. candidate.precursor_smiles must be []; the host applies ReactionJSON to selected_open_leaf_mapped and deterministically derives the canonical precursor structures.",
            "The replayed output may contain at most four atom-contributing precursor molecules, must preserve the heavy-atom inventory required by the product, and must not repeat an ancestor.",
            "Use only map indices present in selected_open_leaf_mapped. Do not use nullable schema filler fields on an operation; each primitive must contain only its semantically relevant fields.",
            "If prior_rejections contains strategy_graph_edit_replay_failed, use its replay_diagnostic reason, attempted operations, and replayed fragments to repair the edit program. Do not rename the same idea or append unrelated hydrogen edits.",
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
    task_type: str = "route_step_materialization",
) -> WorkerTask:
    return WorkerTask(
        task_id=(
            f"{spec.agent_id}:branch:{branch_index + 1}:"
            f"{('editor' if task_type == 'route_chemistry_edit' else 'node')}"
            f":{node_index + 1}"
        ),
        # Strategy workers are blind to the operational run identity.  The
        # target structure remains in the bounded objective, but no target
        # name, run id, or evidence handle is serialized in WorkerTask.
        case_id=_opaque_strategy_case_id(spec.run_id),
        task_type=task_type,
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
    iteration: int,
    timeout_s: float,
) -> WorkerTask:
    return WorkerTask(
        task_id=(
            "critic:"
            + hashlib.sha256(
                (
                    spec.agent_id
                    + ":"
                    + str(branch_index)
                    + ":"
                    + str(iteration)
                ).encode("utf-8")
            ).hexdigest()[:20]
        ),
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


def _unavailable_critique(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "chemical_strategy_critique.v1",
        "status": "unavailable",
        "reason": str(reason),
        "semantics": {
            "critic_required_before_evidence": True,
            "fail_closed": True,
        },
    }


def _blocking_critic_step(
    critique: Mapping[str, Any],
    steps: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Select the first concrete blocking step for the Editor.

    The Critic is allowed to mark unresolved scope as ``uncertain``.  Only a
    concrete ``reject`` verdict triggers a surgical edit, matching the paper's
    blocking-reaction loop rather than rewriting every speculative step.
    """

    by_id = {
        str(step.get("step_id") or ""): dict(step)
        for step in steps
        if str(step.get("step_id") or "")
    }
    assessments = [
        dict(value)
        for value in critique.get("step_assessments") or []
        if isinstance(value, Mapping)
    ]
    for assessment_index, assessment in enumerate(assessments):
        if str(assessment.get("verdict") or "") != "reject":
            continue
        step = by_id.get(str(assessment.get("step_id") or ""))
        # Codex may shorten an opaque host step id after route replay/editing.
        # The critic prompt preserves route order, so use an ordinal fallback
        # only when assessment and route cardinalities agree.
        if step is None and len(assessments) == len(steps):
            step = dict(steps[assessment_index])
        if step is None:
            continue
        step["reasons"] = [
            str(value)
            for value in assessment.get("reasons") or []
            if str(value)
        ]
        step["critic_assessment"] = dict(assessment)
        return step
    return None


def _compact_critic_feedback(
    critique: Mapping[str, Any],
    blocking_step: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a large Critic artifact into an Editor-sized repair brief."""

    step_id = str(blocking_step.get("step_id") or "")
    assessment = next(
        (
            dict(value)
            for value in critique.get("step_assessments") or []
            if isinstance(value, Mapping)
            and str(value.get("step_id") or "") == step_id
        ),
        {},
    )
    if not assessment:
        assessment = dict(blocking_step.get("critic_assessment") or {})
    keep_assessment = {
        key: value
        for key, value in assessment.items()
        if key
        in {
            "step_id",
            "verdict",
            "reasons",
            "mechanistic_analysis",
            "atom_provenance",
            "functional_group_compatibility",
            "chemoselectivity",
            "stereochemistry",
            "sequence_ordering",
            "enzyme_assessment",
        }
    }
    compact_step = {
        key: blocking_step.get(key)
        for key in (
            "step_id",
            "product_smiles",
            "precursor_smiles",
            "transformation_hypothesis",
            "reaction_operations",
            "condition_predictions",
            "strategy_anchor",
        )
        if blocking_step.get(key) not in (None, "", [], {})
    }
    return {
        "overall_assessment": str(critique.get("overall_assessment") or ""),
        "strategy_adherence": critique.get("strategy_adherence"),
        "blocking_step": compact_step,
        "step_assessment": keep_assessment,
        "failure_reasons": [
            str(value)
            for value in blocking_step.get("reasons") or []
            if str(value)
        ][:8],
        "repair_actions": [
            str(value)
            for value in critique.get("repair_actions") or []
            if str(value)
        ][:6],
        "route_level_risks": [
            str(value)
            for value in critique.get("route_level_risks") or []
            if str(value)
        ][:4],
        "experimental_variables": [
            str(value)
            for value in critique.get("experimental_variables") or []
            if str(value)
        ][:4],
    }


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
    declared_precursors = tuple(
        dict.fromkeys(
            canonical
            for value in row.get("precursor_smiles") or []
            if (canonical := _canonical_smiles(value))
        )
    )
    operations = normalize_reaction_operations(row.get("reaction_operations") or ())
    reactionjson_audit: dict[str, Any] = {}
    if operations:
        if not mapped_product_smiles:
            return None
        try:
            reactionjson_audit = replay_reactionjson(
                mapped_product_smiles=mapped_product_smiles,
                operations=operations,
                expected_precursor_smiles=None,
            )
        except ReactionJsonReplayError:
            return None
        precursors = tuple(
            str(value)
            for value in reactionjson_audit.get("precursor_smiles") or []
            if str(value)
        )
    else:
        precursors = declared_precursors
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
    if require_reaction_operations and not operations:
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


def _expansion_rejection_diagnostic(
    record: WorkerRunRecord,
    *,
    expected_product: str,
    mapped_product_smiles: str,
    require_reaction_operations: bool,
) -> dict[str, Any]:
    """Return causal Route Builder feedback without weakening replay."""

    if record.status != "accepted_draft":
        return {"reason": "worker_output_not_accepted"}
    payload = dict(dict(record.output_artifact or {}).get("payload") or {})
    candidates = [
        dict(row)
        for row in payload.get("candidates") or []
        if isinstance(row, Mapping)
    ]
    if len(candidates) != 1:
        return {"reason": "candidate_count_invalid"}
    row = candidates[0]
    product = _canonical_smiles(row.get("product_smiles"))
    if product != expected_product:
        return {
            "reason": "product_mismatch",
            "declared_product_smiles": product,
            "expected_product_smiles": expected_product,
        }
    operations = normalize_reaction_operations(row.get("reaction_operations") or ())
    if require_reaction_operations and not operations:
        return {
            "reason": "strategy_graph_edit_missing",
            "declared_precursor_smiles": list(row.get("precursor_smiles") or []),
        }
    if operations:
        replay = diagnose_reactionjson(
            mapped_product_smiles=mapped_product_smiles,
            operations=operations,
            declared_precursor_smiles=row.get("precursor_smiles") or [],
        )
        if replay.get("replay_succeeded") is not True:
            return {
                "reason": "strategy_graph_edit_replay_failed",
                "replay_error": str(replay.get("reason") or ""),
                "attempted_operations": [dict(value) for value in operations],
                "declared_precursor_smiles": list(
                    replay.get("declared_precursor_smiles") or []
                ),
                "replayed_precursor_smiles": [],
            }
        return {
            "reason": "invalid_expansion_contract",
            "replay_error": str(replay.get("reason") or ""),
            "attempted_operations": [dict(value) for value in operations],
            "declared_precursor_smiles": list(
                replay.get("declared_precursor_smiles") or []
            ),
            "replayed_precursor_smiles": list(
                replay.get("replayed_precursor_smiles") or []
            ),
        }
    return {"reason": "invalid_expansion_contract"}


def _step_row(
    expansion: NodeExpansion,
    *,
    step_id: str,
    strategy_anchor: bool = False,
) -> dict[str, Any]:
    # A StrategyCard identifies the branch-level policy selected before route
    # construction. Per-step ReactionJSON edits are separate execution facts;
    # folding them into the card would give every step in one branch a
    # different strategy digest and make the host reject its own frozen card.
    strategy_card = normalize_strategy_policy_card(expansion.strategy_card or {})
    edit_digest = reaction_edit_digest(expansion.reaction_operations)
    conditions = tuple(
        value
        for raw in expansion.conditions
        if (value := _clean_condition_text(raw))
    )
    catalyst = _clean_condition_text(expansion.catalyst)
    enzyme = _clean_condition_text(expansion.enzyme)
    condition_predictions: list[dict[str, Any]] = []
    if conditions or catalyst or enzyme:
        condition_predictions.append(
            {
                "reagents": list(conditions),
                "catalyst": catalyst,
                "enzyme": enzyme,
                "authority_scope": "model_predicted_condition",
                "not_reaction_proof": True,
            }
        )
    return {
        "step_id": step_id,
        "product_smiles": expansion.product_smiles,
        "precursor_smiles": list(expansion.precursor_smiles),
        "transformation_hypothesis": expansion.reaction_family,
        "strategic_role": expansion.rationale,
        "source_hints": [],
        "required_validation": ["structure", "reaction_feasibility"],
        "hypothesis_only": True,
        "condition_predictions": condition_predictions,
        "limitations": list(expansion.limitations),
        "strategy_card": strategy_card,
        "reaction_operations": [dict(row) for row in expansion.reaction_operations],
        "reaction_edit_digest": edit_digest,
        "reactionjson_audit": dict(expansion.reactionjson_audit or {}),
        "strategy_id": str(strategy_card.get("strategy_id") or ""),
        "strategy_digest": str(strategy_card.get("strategy_digest") or ""),
        "execution_domain": str(
            strategy_card.get("execution_domain") or "chemical"
        ),
        "strategy_anchor": bool(strategy_anchor),
    }


def _clean_condition_text(value: Any) -> str:
    """Keep concrete hypotheses but drop model-generated placeholders.

    A missing condition is an explicit gap.  Phrases such as ``screen`` or
    ``to be determined`` are not operational conditions and must not leak into
    route rows where downstream projections could mistake them for evidence.
    """

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    lowered = text.casefold()
    if any(marker in lowered for marker in _CONDITION_PLACEHOLDER_MARKERS):
        return ""
    return text


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
                "critic_editor_history": [
                    dict(row)
                    for row in branch.get("critic_editor_history") or []
                    if isinstance(row, Mapping)
                ],
                "editor_repairs": [
                    dict(row)
                    for row in branch.get("editor_repairs") or []
                    if isinstance(row, Mapping)
                ],
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
                "critic_editor_history": [
                    dict(row)
                    for row in branch.get("critic_editor_history") or []
                    if isinstance(row, Mapping)
                ],
                "editor_repairs": [
                    dict(row)
                    for row in branch.get("editor_repairs") or []
                    if isinstance(row, Mapping)
                ],
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
