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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from cascade_planner.application.reactionjson_replay import (
    ReactionJsonReplayError,
    diagnose_reactionjson,
    reactionjson_failure_focus,
)
from cascade_planner.application.routejson_compiler import (
    MaterializedReaction,
    RouteJSONCompiler,
)
from cascade_planner.application.biocatalytic_step_contract import (
    BIOLOGICAL_EXECUTION_DOMAINS,
    normalize_biocatalytic_step,
    normalize_step_execution_domain,
)
from cascade_planner.application.strategy_contract import (
    normalize_reaction_operations,
    normalize_strategy_card,
    normalize_strategy_policy_card,
    reaction_edit_digest,
    reaction_edit_signature,
    strategy_cards_conflict,
)
from cascade_planner.interfaces.chemenzy_reactionjson_expansion import (
    ChemEnzyReactionJsonOrSearch,
    ReactionJsonOrCandidate,
    ranked_candidate_cost,
)
from cascade_planner.interfaces.aizynthfinder_strategy_sidecar import (
    run_aizynthfinder_strategy_branch_sidecar,
)

from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerRunRecord,
    WorkerTask,
    preflight_worker_response_schemas,
    run_codex_worker,
    validate_worker_output,
)
from cascade_planner.agent.action_contracts import (
    contains_raw_reaction_payload,
)
from cascade_planner.application.campaign_context import CampaignContext
from cascade_planner.orchestration.global_campaign_director import DirectorConfig
from cascade_planner.runtime import AgentResult, AgentSpec, AgentState


SEQUENTIAL_STRATEGY_SEARCH_SCHEMA = "sequential_strategy_search.v1"
_PAPER_INDEPENDENT_BRANCH_MANDATES = (
    "neutral independent paper strategy with no execution-domain prior",
)
_AUTOPLANNER_HYBRID_BRANCH_MANDATES = (
    (
        "use execution_domain=chemical and prioritize a convergent skeletal "
        "construction: identify a key forward bond-forming event that joins "
        "substantial fragments"
    ),
    (
        "use execution_domain=chemical and prioritize a topology-changing "
        "construction: ring formation, cascade, cycloaddition, or skeletal "
        "rearrangement"
    ),
    (
        "use execution_domain=hybrid and design a genuine chemoenzymatic route: "
        "chemical chemistry should construct the scaffold while one or more local, "
        "chemically credible enzymatic or whole-cell steps solve a selectivity or "
        "functionalization problem with an explicit substrate-product relationship. "
        "Do not ask a generic cyclase to create an entire complex polycyclic scaffold "
        "without close substrate precedent. Retain a conventional chemical fallback; "
        "if no credible biological capability exists, return that fallback as chemical"
    ),
)
_ENZYME_ADVANTAGE_BRANCH_MANDATES = (
    (
        "prioritize a chemically credible enzymatic, whole-cell, or chemoenzymatic "
        "route-defining transformation with an explicit substrate-product boundary, "
        "cofactor ledger, falsifiable capability test, and retained conventional "
        "chemical fallback; do not invent enzyme capability to satisfy the lens"
    ),
)
_CHEMOENZYMATIC_FUSION_BRANCH_MANDATES = (
    (
        "use execution_domain=hybrid and require a route-level chemoenzymatic "
        "sequence with at least one chemical scaffold-forming step and at least "
        "one enzymatic or whole-cell local tailoring step. In retrosynthetic "
        "order, disconnect the biologically installed feature and continue "
        "deconstructing the resulting advanced scaffold by chemical bond-forming "
        "logic; a biological label or cofactor list alone does not satisfy this "
        "mandate. Keep exact substrate-product, atom-source, cofactor, selectivity, "
        "and validation ledgers, and do not invent enzyme capability"
    ),
)
_AUTOPLANNER_STRATEGY_V2_BRANCH_MANDATES = (
    (
        "strategy_v2_slot=convergent: prioritize a genuinely convergent "
        "fragment union that joins substantial, independently preparable "
        "pieces and creates a high-information skeletal event"
    ),
    (
        "strategy_v2_slot=topology: prioritize a topology-changing event "
        "such as annulation, cascade, cycloaddition, ring construction, "
        "fragmentation, or skeletal rearrangement when the target motifs "
        "support it"
    ),
    (
        "strategy_v2_slot=reorganization: prioritize an orthogonal "
        "skeletal reorganization, polarity inversion, radical/pericyclic "
        "sequence, late-stage ring closure, or a chemically credible local "
        "biological overlay; retain a conventional chemical fallback"
    ),
)


def _branch_mandates_for_profile(profile: str) -> tuple[str, ...]:
    if profile == "paper_independent":
        return _PAPER_INDEPENDENT_BRANCH_MANDATES
    if profile == "enzyme_advantage":
        return _ENZYME_ADVANTAGE_BRANCH_MANDATES
    if profile == "autoplanner_hybrid":
        return _AUTOPLANNER_HYBRID_BRANCH_MANDATES
    if profile == "chemoenzymatic_fusion":
        return _CHEMOENZYMATIC_FUSION_BRANCH_MANDATES
    if profile == "autoplanner_strategy_v2":
        return _AUTOPLANNER_STRATEGY_V2_BRANCH_MANDATES
    raise ValueError("sequential strategy portfolio mode is invalid")

_STRATEGY_CARD_FIELDS = (
    "strategy_query",
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
_CRITIC_OUTPUT_TOKEN_RESERVE = 16_000
_EDITOR_INPUT_TOKEN_RESERVE = 20_000
_EDITOR_OUTPUT_TOKEN_RESERVE = 20_000
_CRITIC_EDITOR_WALL_FRACTION = 0.30
_PAPER_CRITIC_EDITOR_WALL_FRACTION = 0.40
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

_STEP_ROLES = frozenset({"key", "enabling", "supporting"})
_CHECKPOINT_RELATIONS = frozenset({"preparatory", "executes_checkpoint"})
_MAX_KEY_EVENT_CRITIC_CALLS_PER_BRANCH = 2


def _normalize_step_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return role if role in _STEP_ROLES else ""


def _normalize_checkpoint_relation(value: Any) -> str:
    relation = str(value or "").strip().lower()
    return relation if relation in _CHECKPOINT_RELATIONS else ""


@dataclass(frozen=True, slots=True)
class NodeExpansion:
    product_smiles: str
    precursor_smiles: tuple[str, ...]
    reaction_family: str
    rationale: str
    step_role: str = ""
    checkpoint_relation: str = ""
    mapped_product_smiles: str = ""
    mapped_precursor_smiles: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    catalyst: str = ""
    enzyme: str = ""
    execution_domain: str = "chemical"
    biocatalytic_step: Mapping[str, Any] | None = None
    limitations: tuple[str, ...] = ()
    product_retron_type: str = ""
    strategy_card: Mapping[str, Any] | None = None
    reaction_operations: tuple[Mapping[str, Any], ...] = ()
    reactionjson_audit: Mapping[str, Any] | None = None
    step_id: str = ""


@dataclass(frozen=True, slots=True)
class _NodeCallBudget:
    model_invocations: int
    input_tokens: int
    output_tokens: int
    wall_time_s: float


@dataclass(frozen=True, slots=True)
class _CompiledReactionJsonCandidate:
    candidate_index: int
    candidate_id: str
    expansion: NodeExpansion
    score: float
    cost: float
    candidate_key: str


NodeExecutor = Callable[[WorkerTask], WorkerRunRecord]
StockMembership = Callable[[Iterable[str]], Mapping[str, bool]]


class SequentialStrategyDirectorRunner:
    """Director runner implementing independent, continuous node expansion."""

    compact_prompt = True
    durable_worker_journal = True

    def __init__(
        self,
        *,
        node_executor: NodeExecutor | None = None,
        critic_executor: NodeExecutor | None = None,
        editor_executor: NodeExecutor | None = None,
        stock_membership: StockMembership | None = None,
        aizynthfinder_strategy_python_executable: str = "",
        aizynthfinder_strategy_stock_index: str = "",
        aizynthfinder_strategy_inline_stock_smiles: tuple[str, ...] = (),
        worker_record_seed_path: str = "",
        worker_record_seed_recovery_mode: str = "",
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.node_executor = node_executor or self._execute_node
        self.critic_executor = critic_executor or self.node_executor
        self.editor_executor = editor_executor or self.node_executor
        self.stock_membership = stock_membership
        self.aizynthfinder_strategy_python_executable = str(
            aizynthfinder_strategy_python_executable or ""
        )
        self.aizynthfinder_strategy_stock_index = str(
            aizynthfinder_strategy_stock_index or ""
        )
        # Test-only deterministic stock. Production paper runs bind the
        # content-addressed ZINC+eMolecules SQLite index above.
        self.aizynthfinder_strategy_inline_stock_smiles = tuple(
            str(value)
            for value in aizynthfinder_strategy_inline_stock_smiles
            if str(value)
        )
        self.worker_record_seed_path = str(worker_record_seed_path or "").strip()
        self.worker_record_seed_recovery_mode = str(
            worker_record_seed_recovery_mode or ""
        ).strip()
        self._worker_record_seed_parent = (
            Path(self.worker_record_seed_path).expanduser().resolve().parent
            if self.worker_record_seed_path
            else None
        )
        self.cancel_event = cancel_event
        # The compiler is the sole structure authority for compiler-first
        # Route Builder calls.  The model proposes edit programs; this host
        # object materializes and carries mapped fragments across calls.
        self.routejson_compiler = RouteJSONCompiler()
        self._journal_lock = threading.Lock()
        self._worker_record_journal_path: Path | None = None
        self._model_io_journal_path: Path | None = None
        self._worker_record_cache: dict[tuple[str, str], WorkerRunRecord] = {}
        self._replayed_worker_record_count = 0
        self._seeded_worker_record_count = 0
        self._exact_seed_replay_count = 0
        self._seed_worker_records_by_model_input: dict[str, WorkerRunRecord] = {}
        self._seed_model_inputs_by_task_id: dict[str, dict[str, Any]] = {}

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
            f"node_expansions_per_branch={config.max_node_expansions_per_branch}; "
            "strategic_milestones_per_branch="
            f"{config.max_strategic_milestones_per_branch}."
        )

    def __call__(
        self,
        spec: AgentSpec,
        context: CampaignContext,
        mode: str,
        config: DirectorConfig,
    ) -> AgentResult:
        started = time.monotonic()
        target = _canonical_smiles(context.target.get("canonical_smiles"))
        if config.paper_matched_reach_profile:
            _preflight_paper_matched_worker_schemas(spec, target=target)
        self._prepare_worker_record_journal(spec)
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
        usage["durable_worker_record_journal"] = bool(
            self._worker_record_journal_path is not None
        )
        usage["replayed_worker_record_count"] = int(
            self._replayed_worker_record_count
        )
        usage["seeded_worker_record_count"] = int(
            self._seeded_worker_record_count
        )
        usage["worker_record_seed_used"] = bool(
            self._seeded_worker_record_count
        )
        usage["worker_record_seed_recovery_mode"] = (
            self.worker_record_seed_recovery_mode
            if self._seeded_worker_record_count
            else ""
        )
        usage["exact_seed_replay_count"] = int(
            self._exact_seed_replay_count
        )
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
        usage["actual_route_builder_policy_calls"] = sum(
            int(branch.get("route_call_count") or 0) for branch in branches
        )
        usage["actual_critic_calls"] = sum(
            int(branch.get("critic_call_count") or 0) for branch in branches
        )
        usage["actual_strategy_critic_calls"] = max(
            (
                int(branch.get("strategy_critic_call_count") or 0)
                for branch in branches
            ),
            default=0,
        )
        usage["actual_key_event_critic_calls"] = sum(
            int(branch.get("key_event_critic_call_count") or 0)
            for branch in branches
        )
        usage["actual_editor_calls"] = sum(
            int(branch.get("editor_attempt_count") or 0) for branch in branches
        )
        usage["strategy_branch_workers"] = int(config.strategy_branch_workers)
        usage["strategic_milestone_limit_per_branch"] = int(
            config.max_strategic_milestones_per_branch
        )
        usage["upstream_strategy_milestone_calls"] = sum(
            max(0, int(branch.get("strategy_call_count") or 0) - 1)
            for branch in branches
        )
        usage["realized_strategic_milestones"] = sum(
            int(branch.get("strategic_milestone_count") or 0)
            for branch in branches
        )
        usage["stop_on_first_stock_closed_branch"] = bool(
            config.stop_on_first_stock_closed_branch
        )
        usage["stock_closed_branch_count"] = sum(
            _branch_stock_closed(branch) for branch in branches
        )
        usage["stock_closed_early_stop_triggered"] = any(
            bool(branch.get("portfolio_early_stop_triggered"))
            for branch in branches
        )
        # SynthEx reports 25 Route Builder steps as a search ceiling.  Only a
        # Host/AiZ search termination may end a branch earlier; stock closure
        # remains a Host observation.
        paper_policy_branches = [
            branch
            for branch in branches
            if str(branch.get("strategy_tree_engine") or "")
            == "aizynthfinder_mcts"
            and config.strategy_portfolio_mode == "paper_independent"
        ]
        for branch in paper_policy_branches:
            # A missing StrategyCard or a branch that never reached its AiZ
            # sidecar is a failed independent branch, not an omitted datum.
            if not branch.get("strategy_card"):
                branch["paper_policy_budget_failure"] = {
                    "reason": "paper_strategy_branch_not_seeded",
                    "required_calls": int(config.max_node_expansions_per_branch),
                    "actual_calls": 0,
                    "stock_closed": False,
                }
            elif not branch.get("aizynthfinder_strategy_search"):
                branch["paper_policy_budget_failure"] = {
                    "reason": "paper_strategy_branch_not_started",
                    "required_calls": int(config.max_node_expansions_per_branch),
                    "actual_calls": 0,
                    "stock_closed": False,
                }
            elif dict(branch.get("aizynthfinder_strategy_search") or {}).get(
                "failed"
            ):
                sidecar = dict(branch.get("aizynthfinder_strategy_search") or {})
                branch["paper_policy_budget_failure"] = {
                    "reason": "paper_strategy_sidecar_failed",
                    "required_calls": int(config.max_node_expansions_per_branch),
                    "actual_calls": int(
                        branch.get("route_call_count") or 0
                    ),
                    "stock_closed": False,
                    "detail": str(sidecar.get("error") or "")[:800],
                }
        usage["paper_policy_call_budget"] = {
            "maximum_per_branch": int(config.max_node_expansions_per_branch),
            "branch_count": len(paper_policy_branches),
            "actual_calls": [
                int(branch.get("route_call_count") or 0)
                for branch in paper_policy_branches
            ],
            "branch_summaries": [
                {
                    "branch_index": int(branch.get("branch_index") or 0) + 1,
                    "actual_policy_calls": int(
                        branch.get("route_call_count") or 0
                    ),
                    "provider_callback_count": int(
                        dict(branch.get("aizynthfinder_strategy_search") or {}).get(
                            "provider_callback_count"
                        )
                        or 0
                    ),
                    "selected_depth": int(
                        dict(branch.get("aizynthfinder_strategy_search") or {}).get(
                            "selected_depth"
                        )
                        or 0
                    ),
                    "selected_open_leaves": int(
                        dict(branch.get("aizynthfinder_strategy_search") or {}).get(
                            "selected_open_leaves"
                        )
                        or len(branch.get("open_leaves") or [])
                    ),
                    "selected_solved": bool(
                        dict(branch.get("aizynthfinder_strategy_search") or {}).get(
                            "selected_solved"
                        )
                    ),
                    "calls_exhausted": bool(
                        dict(branch.get("aizynthfinder_strategy_search") or {}).get(
                            "calls_exhausted"
                        )
                    ),
                    "route_step_count": len(branch.get("steps") or []),
                }
                for branch in paper_policy_branches
            ],
            "stock_closed": [
                bool(_branch_stock_closed(branch))
                for branch in paper_policy_branches
            ],
            "hard_failures": [
                dict(branch.get("paper_policy_budget_failure") or {})
                for branch in paper_policy_branches
                if branch.get("paper_policy_budget_failure")
            ],
            "semantics": {
                "less_than_ceiling_is_allowed_after_host_search_termination": True,
                "configured_calls_are_a_maximum_not_a_required_minimum": True,
                "builder_has_no_terminal_authority": True,
                "policy_calls_are_distinct_from_mcts_iterations": True,
            },
        }
        if self.cancel_event is not None and self.cancel_event.is_set():
            return _agent_result(
                spec,
                state=AgentState.CANCELLED,
                output=None,
                usage=usage,
                error="delivery_milestone_reached",
                mode=mode,
            )
        strict_policy_failure = any(
            branch.get("paper_policy_budget_failure")
            for branch in paper_policy_branches
        )
        usable: list[dict[str, Any]] = []
        plan_branches: list[dict[str, Any]] = []
        branch_route_retention: list[dict[str, Any]] = []
        for branch in branches:
            steps = [
                dict(row)
                for row in branch.get("steps") or []
                if isinstance(row, Mapping)
            ]
            replay_validation = (
                _route_steps_host_replay_validation(
                    steps,
                    mapped_target_smiles=str(
                        branch.get("target_mapped_smiles") or _mapped_smiles(target)
                    ),
                )
                if steps
                else {"complete": False, "reason": "routejson_route_empty"}
            )
            policy_usable = bool(
                not branch.get("paper_policy_budget_failure")
                or branch.get("sidecar_recovered_prefix") is True
            )
            route_usable = bool(
                steps
                and policy_usable
                and (
                    not config.require_complete_route_json
                    or replay_validation.get("complete") is True
                )
            )
            retention = {
                "branch_index": int(branch.get("branch_index") or 0) + 1,
                "strategy_id": str(
                    dict(branch.get("strategy_card") or {}).get("strategy_id") or ""
                ),
                "selected_step_count": len(steps),
                "retained_as_replayable_route": route_usable,
                "paper_policy_usable": policy_usable,
                "routejson_replay_validation": dict(replay_validation),
            }
            branch_route_retention.append(retention)
            branch_with_validation = {
                **branch,
                "routejson_replay_validation": dict(replay_validation),
            }
            if route_usable:
                usable.append(branch_with_validation)
                plan_branches.append(branch_with_validation)
                continue
            # A Strategy hypothesis is useful diagnostic evidence even when
            # its selected AiZ path cannot be admitted as RouteJSON.  Preserve
            # the family and its original branch identity, but remove the
            # invalid steps so it cannot enter canonical materialization.
            plan_branches.append(
                {
                    **branch_with_validation,
                    "steps": [],
                    "open_leaves": [],
                    "open_leaf_states": deque(),
                    "routejson_rejected_step_count": len(steps),
                }
            )
        usage["branch_route_retention"] = branch_route_retention
        if strict_policy_failure:
            usage["rejection_reasons"] = sorted(
                {
                    str(
                        dict(branch.get("paper_policy_budget_failure") or {}).get(
                            "reason"
                        )
                        or "paper_policy_execution_failed"
                    )
                    for branch in paper_policy_branches
                    if branch.get("paper_policy_budget_failure")
                }
            )
            # Independent paper branches are independent evidence atoms.  A
            # sidecar or seed failure removes only that branch from the plan;
            # it must not erase other Host-replayed routes and completed
            # Critiques.  If every branch is unusable, the ordinary no-usable
            # result below still fails the Director with the causal reasons.
            usage["paper_policy_partial_branch_failure"] = True
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
            branches=plan_branches,
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

    def _prepare_worker_record_journal(self, spec: AgentSpec) -> None:
        """Load completed worker calls so a crashed Director resumes exactly.

        The branch state is deterministic given its ordered worker records and
        host compiler.  Replaying records by stable task id therefore rebuilds
        the same StrategyCard/ReactionJSON/RouteJSON state without maintaining
        a second mutable branch snapshot.  A task-contract digest prevents a
        record created by an older prompt or model contract from being reused.
        """

        self._worker_record_cache = {}
        self._replayed_worker_record_count = 0
        self._seeded_worker_record_count = 0
        self._exact_seed_replay_count = 0
        self._seed_worker_records_by_model_input = {}
        self._seed_model_inputs_by_task_id = {}
        self._worker_record_journal_path = None
        self._model_io_journal_path = None
        if spec.metadata.get("durable_worker_journal") is not True:
            return
        workdir = Path(str(spec.metadata.get("allowed_workdir") or "")).resolve()
        path = workdir / "sequential-director-worker-records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_record_journal_path = path
        self._model_io_journal_path = workdir / "model-io.jsonl"
        self._load_worker_record_journal(path, seeded=False)
        seed_path = Path(self.worker_record_seed_path).expanduser()
        if self.worker_record_seed_path and seed_path.resolve() != path:
            self._load_worker_record_journal(seed_path, seeded=True)

    def _load_worker_record_journal(self, path: Path, *, seeded: bool) -> None:
        if not path.is_file():
            return
        if seeded:
            self._load_seed_model_input_journal(path.with_name("model-io.jsonl"))
        loaded = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                record_row = dict(row.get("record") or {})
                key = (
                    str(row.get("task_id") or ""),
                    str(row.get("task_contract_sha256") or ""),
                )
                if (
                    row.get("schema_version")
                    != "sequential_director_worker_record.v1"
                    or not all(key)
                    or not record_row
                ):
                    continue
                record = WorkerRunRecord(**record_row)
                # Cancellation is an interrupted execution boundary, not a
                # completed model result.  Keep its journal row as incident
                # history, but never replay the empty/cancelled record on a
                # resume; the smallest interrupted worker call must run again.
                if record.status == "cancelled":
                    continue
                self._worker_record_cache[key] = record
                if seeded:
                    portable_digest = str(
                        row.get("portable_model_input_sha256") or ""
                    )
                    model_input = self._seed_model_inputs_by_task_id.get(
                        str(row.get("task_id") or "")
                    )
                    if model_input is None:
                        # A journal can itself contain aliases replayed from an
                        # older run.  Legacy aliases predate the portable
                        # digest and therefore have no local model-input row,
                        # but the preserved worker record still names both the
                        # original task and its durable Codex event log.  Read
                        # only that explicit provenance workspace; never scan
                        # other runs or match on output content.
                        event_log_path = str(
                            dict(record.metadata or {}).get("event_log_path")
                            or ""
                        ).strip()
                        if event_log_path and str(record.task_id or ""):
                            provenance_model_io = (
                                Path(event_log_path).expanduser().parent.parent
                                / "model-io.jsonl"
                            )
                            self._load_seed_model_input_journal(
                                provenance_model_io
                            )
                            model_input = self._seed_model_inputs_by_task_id.get(
                                str(record.task_id or "")
                            )
                    if portable_digest:
                        self._seed_worker_records_by_model_input[
                            portable_digest
                        ] = record
                    elif model_input is not None:
                        self._seed_worker_records_by_model_input[
                            _portable_model_input_sha256(model_input)
                        ] = record
                loaded += 1
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # A process can die after writing only part of the final line.
                # Earlier fsync'd records remain authoritative and reusable.
                continue
        if seeded:
            self._seeded_worker_record_count += loaded

    def _load_seed_model_input_journal(self, path: Path) -> None:
        """Index participant-visible inputs for exact cross-run replay."""

        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("event") != "model_input":
                    continue
                task_id = str(row.get("task_id") or "")
                if task_id:
                    self._seed_model_inputs_by_task_id[task_id] = row
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def _run_journaled_worker(
        self,
        executor: NodeExecutor,
        task: WorkerTask,
    ) -> WorkerRunRecord:
        contract_sha256 = _worker_task_contract_sha256(task)
        key = (task.task_id, contract_sha256)
        with self._journal_lock:
            cached = self._worker_record_cache.get(key)
            if cached is not None:
                self._replayed_worker_record_count += 1
                return cached
            relocated_key: tuple[str, str] | None = None
            relocated_cached: WorkerRunRecord | None = None
            # A recovery run has a new operational cwd but the same prompt,
            # model, schema, budget and task identity.  Director workers have
            # no tools, so relocating only this non-scientific path cannot
            # alter participant-visible inputs.  Try the seed workspace digest
            # without weakening every journal contract globally.
            if self._worker_record_seed_parent is not None and not task.allowed_tools:
                relocated = replace(
                    task,
                    allowed_workdir=str(self._worker_record_seed_parent),
                )
                relocated_key = (
                    task.task_id,
                    _worker_task_contract_sha256(relocated),
                )
                relocated_cached = self._worker_record_cache.get(relocated_key)
            exact_input_cached: WorkerRunRecord | None = None
            if (
                relocated_cached is None
                and self.worker_record_seed_recovery_mode
                == "exact_model_io_v1"
            ):
                candidate = self._seed_worker_records_by_model_input.get(
                    _portable_model_input_sha256(task)
                )
                if candidate is not None and _seed_record_matches_task(
                    candidate,
                    task,
                ):
                    exact_input_cached = candidate
        if relocated_cached is not None:
            self._persist_worker_record_alias(task, key, relocated_cached)
            self._replayed_worker_record_count += 1
            return relocated_cached
        if exact_input_cached is not None:
            self._persist_worker_record_alias(task, key, exact_input_cached)
            self._replayed_worker_record_count += 1
            self._exact_seed_replay_count += 1
            return exact_input_cached
        common_io = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task_id": task.task_id,
            "case_id": task.case_id,
            "task_type": task.task_type,
            "artifact_type": task.required_artifact_type,
            "model": task.model,
            "reasoning_effort": task.budget.reasoning_effort,
        }
        self._append_model_io_event(
            {
                **common_io,
                "event": "model_input",
                "prompt": task.objective,
                "input_refs": list(task.input_refs),
            }
        )
        try:
            record = executor(task)
        except Exception as exc:
            self._append_model_io_event(
                {
                    **common_io,
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "event": "model_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        self._append_model_io_event(
            {
                **common_io,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "model_output",
                # This journal reports whether the worker output passed its
                # declared artifact/schema boundary.  It does not decide
                # whether a proposed route step is later committed, rejected
                # by a Critic, or rolled back by the canonical Host graph.
                "status": _model_output_validation_status(record),
                "status_scope": "worker_output_schema_validation",
                "worker_record_status": record.status,
                "output_validation": dict(record.output_validation or {}),
                "stdout": record.stdout,
                "stderr": record.stderr,
                "output_artifact": record.output_artifact,
                "usage": dict(record.usage or {}),
            }
        )
        path = self._worker_record_journal_path
        if path is None:
            return record
        row = {
            "schema_version": "sequential_director_worker_record.v1",
            "task_id": task.task_id,
            "task_contract_sha256": contract_sha256,
            "portable_model_input_sha256": _portable_model_input_sha256(task),
            "record": record.to_dict(),
        }
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._journal_lock:
            # Another branch cannot own the same task id, but checking again
            # keeps this correct if an executor is ever retried concurrently.
            cached = self._worker_record_cache.get(key)
            if cached is not None:
                self._replayed_worker_record_count += 1
                return cached
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._worker_record_cache[key] = record
        return record

    def _append_model_io_event(self, row: Mapping[str, Any]) -> None:
        """Append one monitor-visible model I/O event durably."""

        path = self._model_io_journal_path
        if path is None:
            return
        encoded = json.dumps(
            {
                "schema_version": "sequential_director_model_io.v1",
                **dict(row),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._journal_lock:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _persist_worker_record_alias(
        self,
        task: WorkerTask,
        key: tuple[str, str],
        record: WorkerRunRecord,
    ) -> None:
        """Bind a relocated seed record to the fresh run's exact contract."""

        path = self._worker_record_journal_path
        if path is None:
            return
        row = {
            "schema_version": "sequential_director_worker_record.v1",
            "task_id": key[0],
            "task_contract_sha256": key[1],
            "portable_model_input_sha256": _portable_model_input_sha256(task),
            "record": record.to_dict(),
        }
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._journal_lock:
            if key in self._worker_record_cache:
                return
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._worker_record_cache[key] = record

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

        Critic output is deliberately non-authoritative for topology.  A
        concrete chemical rejection blocks recommendation/validation, but it
        must not erase a host-materialized route needed for paper-equivalent
        reach/stock scoring.  Both reject and unavailable therefore remain
        visible independent-axis deficits on the retained route family.
        """

        target = _canonical_smiles(context.target.get("canonical_smiles"))
        max_rounds = max(0, int(config.max_route_local_repair_rounds))
        for index, branch in enumerate(branches):
            if not branch.get("steps"):
                continue
            # The matched arm admits only a host-replayable RouteJSON
            # document to Critic/Editor.  Keep an incomplete skeleton for
            # diagnostics, but do not spend repair calls on it.
            routejson_validation = _route_steps_host_replay_validation(
                branch.get("steps") or (),
                mapped_target_smiles=str(
                    branch.get("target_mapped_smiles") or _mapped_smiles(target)
                ),
            )
            branch["routejson_replay_validation"] = routejson_validation
            if (
                config.require_complete_route_json
                and routejson_validation.get("complete") is not True
            ):
                branch["critic_editor_skipped_incomplete_route_json"] = True
                branch["critic_editor_skip_reason"] = (
                    "routejson_admission_requires_target_rooted_dag_replay"
                )
                branch.setdefault("critic_editor_history", []).append(
                    {
                        "round": 0,
                        "status": "skipped",
                        "reason": "incomplete_route_json",
                        "routejson_replay_validation": dict(routejson_validation),
                        "critic_task_ids": [],
                        "editor_task_ids": [],
                        "actual_critic_call_count": 0,
                        "actual_editor_call_count": 0,
                    }
                )
                continue
            branch.setdefault("critic_editor_history", [])
            branch.setdefault("critic_call_count", 0)
            branch.setdefault("editor_attempt_count", 0)
            for iteration in range(max_rounds + 1):
                future_families = sum(
                    bool(row.get("steps"))
                    and not dict(row.get("chemical_critic") or {}).get("status")
                    for row in branches[index + 1 :]
                )
                if config.paper_matched_reach_profile:
                    # Reserve only calls the remaining state machines can
                    # still execute.  Once a branch has consumed its Editor
                    # budget, its required final Critic must not be blocked by
                    # fictitious future Editor/Critic pairs.
                    (
                        reserve_critic_calls,
                        reserve_editor_calls,
                    ) = _paper_critic_editor_reserve_after_current_critic(
                        branches,
                        current_index=index,
                        iteration=iteration,
                        max_rounds=max_rounds,
                    )
                    reserve_calls_after_this_critic = (
                        reserve_critic_calls + reserve_editor_calls
                    )
                    reserve_input_after_this_critic = (
                        reserve_critic_calls * _CRITIC_INPUT_TOKEN_RESERVE
                        + reserve_editor_calls * _EDITOR_INPUT_TOKEN_RESERVE
                    )
                    reserve_output_after_this_critic = (
                        reserve_critic_calls * _CRITIC_OUTPUT_TOKEN_RESERVE
                        + reserve_editor_calls * _EDITOR_OUTPUT_TOKEN_RESERVE
                    )
                    remaining_wall = _remaining_node_wall_time(started, quota)
                    maximum_repair_call_wall = min(
                        float(config.critic_call_timeout_s),
                        float(config.max_node_call_timeout_s),
                    )
                    reserve_wall_after_this_critic = (
                        reserve_calls_after_this_critic
                        * maximum_repair_call_wall
                    )
                    per_critic_wall = min(
                        config.critic_call_timeout_s,
                        max(
                            0.001,
                            remaining_wall - reserve_wall_after_this_critic,
                        ),
                    )
                else:
                    # Preserve the original ordinary-run policy exactly: one
                    # current Critic/Editor pair plus one pair per later route.
                    remaining_critics = 1 + future_families
                    remaining_wall = _remaining_node_wall_time(started, quota)
                    family_wall = remaining_wall / max(1, remaining_critics)
                    current_editor_wall_reserve = family_wall * 0.5
                    future_pair_wall_reserve = future_families * family_wall
                    reserve_calls_after_this_critic = 2 + future_families * 2
                    reserve_input_after_this_critic = (
                        _CRITIC_INPUT_TOKEN_RESERVE
                        + _EDITOR_INPUT_TOKEN_RESERVE
                        + future_families
                        * (
                            _CRITIC_INPUT_TOKEN_RESERVE
                            + _EDITOR_INPUT_TOKEN_RESERVE
                        )
                    )
                    reserve_output_after_this_critic = (
                        _CRITIC_OUTPUT_TOKEN_RESERVE
                        + _EDITOR_OUTPUT_TOKEN_RESERVE
                        + future_families
                        * (
                            _CRITIC_OUTPUT_TOKEN_RESERVE
                            + _EDITOR_OUTPUT_TOKEN_RESERVE
                        )
                    )
                    reserve_wall_after_this_critic = (
                        current_editor_wall_reserve + future_pair_wall_reserve
                    )
                    per_critic_wall = min(
                        config.critic_call_timeout_s,
                        max(0.001, family_wall - current_editor_wall_reserve),
                    )
                budget_block_reason = _node_budget_block_reason(
                    records,
                    started=started,
                    quota=quota,
                    reserve_model_invocations=reserve_calls_after_this_critic,
                    reserve_input_tokens=reserve_input_after_this_critic,
                    reserve_output_tokens=reserve_output_after_this_critic,
                    reserve_wall_time_s=reserve_wall_after_this_critic,
                )
                if budget_block_reason:
                    skip_diagnostic = {
                        "reason": budget_block_reason,
                        "branch_index": int(branch.get("branch_index") or 0) + 1,
                        "iteration": iteration,
                        "observed_model_invocations": len(records),
                        "reserve_model_invocations": reserve_calls_after_this_critic,
                        "reserve_input_tokens": reserve_input_after_this_critic,
                        "reserve_output_tokens": reserve_output_after_this_critic,
                        "reserve_wall_time_s": reserve_wall_after_this_critic,
                        "quota": {
                            "model_invocations": quota.model_invocations,
                            "input_tokens": quota.input_tokens,
                            "output_tokens": quota.output_tokens,
                            "wall_time_s": quota.wall_time_s,
                        },
                    }
                    branch["critic_skip_diagnostic"] = skip_diagnostic
                    self._append_model_io_event(
                        {
                            "event": "model_skipped",
                            "task_type": "paper_matched_route_critic",
                            "artifact_type": "ChemicalStrategyCritique",
                            **skip_diagnostic,
                        }
                    )
                    branch["chemical_critic"] = _unavailable_critique(
                        f"critic_budget_exhausted:{budget_block_reason}"
                    )
                    break
                prompt = _bounded_critic_prompt(
                    target=target,
                    branch_index=int(branch.get("branch_index") or 0),
                    strategy_card=dict(branch.get("strategy_card") or {}),
                    strategy_milestone_cards=list(
                        branch.get("strategy_milestone_cards") or []
                    ),
                    steps=list(branch.get("steps") or []),
                    maximum_bytes=config.max_node_prompt_bytes,
                    paper_matched=config.paper_matched_reach_profile,
                )
                if prompt is None:
                    # Prompt size is a runtime resource control, not a reason
                    # to erase every already-materialized route family.  If
                    # even the structure-only projection cannot fit, fail this
                    # Critic closed and allow the other independent families
                    # and the durable Director result to survive.
                    branch["chemical_critic"] = _unavailable_critique(
                        "critic_prompt_byte_budget_exhausted"
                    )
                    branch.setdefault("rejections", []).append(
                        {
                            "phase": "chemical_critic",
                            "reason": "critic_prompt_byte_budget_exhausted",
                            "route_step_count": len(branch.get("steps") or []),
                            "branch_retained": True,
                        }
                    )
                    self._append_model_io_event(
                        {
                            "event": "model_skipped",
                            "task_type": "paper_matched_route_critic",
                            "artifact_type": "ChemicalStrategyCritique",
                            "reason": "critic_prompt_byte_budget_exhausted",
                            "branch_index": int(branch.get("branch_index") or 0)
                            + 1,
                            "iteration": iteration,
                            "route_step_count": len(branch.get("steps") or []),
                            "maximum_prompt_bytes": config.max_node_prompt_bytes,
                        }
                    )
                    break
                task = _critic_task(
                    spec,
                    prompt=prompt,
                    branch_index=int(branch.get("branch_index") or 0),
                    iteration=iteration,
                    timeout_s=max(0.001, per_critic_wall),
                    paper_matched=config.paper_matched_reach_profile,
                    target_smiles=target,
                )
                branch["critic_call_count"] = int(
                    branch.get("critic_call_count") or 0
                ) + 1
                try:
                    record = self._run_journaled_worker(self.critic_executor, task)
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
                        "round": iteration + 1,
                        "critic_task_ids": [task.task_id],
                        "actual_critic_call_count": 1,
                        "editor_task_ids": [],
                        "actual_editor_call_count": 0,
                        "critic": dict(critique),
                    }
                )
                if str(critique.get("status") or "") == "unavailable":
                    break
                blocking_steps = _blocking_critic_steps(
                    critique,
                    list(branch.get("steps") or []),
                )
                if not blocking_steps:
                    break
                if iteration >= max_rounds:
                    branch["chemical_critic"] = {
                        **critique,
                        "status": "reject",
                        "reason": "critic_editor_iteration_limit_reached",
                    }
                    break
                records_before_editor = len(records)
                edited = self._edit_branch_from_critique(
                    spec,
                    target=target,
                    branch=branch,
                    blocking_steps=blocking_steps,
                    critique=critique,
                    iteration=iteration,
                    records=records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    max_node_expansions_per_branch=config.max_node_expansions_per_branch,
                    materialization_editor_rounds=config.max_route_local_repair_rounds,
                    require_strategy_graph_edits=config.require_strategy_graph_edits,
                    allow_editor_route_mutations=config.allow_editor_route_mutations,
                    paper_matched=config.paper_matched_reach_profile,
                    quota=quota,
                    started=started,
                )
                editor_task_ids = [
                    str(row.task_id)
                    for row in records[records_before_editor:]
                    if ":editor:" in str(row.task_id or "")
                ]
                branch["critic_editor_history"][-1]["editor_task_ids"] = editor_task_ids
                branch["critic_editor_history"][-1]["editor_call_count"] = len(
                    editor_task_ids
                )
                branch["critic_editor_history"][-1][
                    "actual_editor_call_count"
                ] = len(editor_task_ids)
                if not edited:
                    editor_diagnostics = [
                        dict(row)
                        for row in branch.get("editor_rejection_diagnostics") or []
                        if isinstance(row, Mapping)
                    ]
                    if editor_diagnostics:
                        # The Editor worker did run; only its route mutation
                        # was rejected or retained as an incomplete working
                        # prefix.  Preserve the Critic's scientific verdict
                        # instead of collapsing this into an execution error.
                        branch["chemical_critic"] = {
                            **critique,
                            "status": "reject",
                            "reason": "editor_route_rejected",
                            "editor_rejection_diagnostics": editor_diagnostics[-3:],
                        }
                    else:
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
        blocking_steps: Iterable[Mapping[str, Any]],
        critique: Mapping[str, Any],
        iteration: int,
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        max_node_expansions_per_branch: int,
        materialization_editor_rounds: int,
        require_strategy_graph_edits: bool,
        allow_editor_route_mutations: bool,
        paper_matched: bool,
        quota: _NodeCallBudget,
        started: float,
    ) -> bool:
        steps = [dict(row) for row in branch.get("steps") or []]
        concrete_blockers = [
            dict(row) for row in blocking_steps if isinstance(row, Mapping)
        ]
        if not concrete_blockers:
            return False
        blocking_step = concrete_blockers[0]
        # AiZynthFinder owns the connected route projection for the paper
        # arm.  The old ``surgical`` editor path replaced the blocking step
        # with ``prefix + edited_step`` and deliberately skipped suffix
        # rebuilding for an AiZ tree.  On a 17-step projection this silently
        # turned the public route into a one-step route (the run still
        # reported the original MCTS depth).  SynthEx's Improvement stage is
        # route-document aware, so use the complete-route-context Editor for an
        # AiZ branch even when a legacy caller leaves the flag at its old
        # default.  This is a safety conversion, not an extra search call:
        # either the full edited document replays or the original topology is
        # retained by the non-improving-route guard below.
        if (
            not allow_editor_route_mutations
            and str(branch.get("strategy_tree_engine") or "")
            == "aizynthfinder_mcts"
        ):
            branch.setdefault("editor_execution_notes", []).append(
                {
                    "reason": "mcts_surgical_editor_would_drop_route_suffix",
                    "requested_mode": "surgical",
                    "effective_mode": "dependency_closed_replace_span",
                    "semantics": {
                        "suffix_is_never_discarded": True,
                        "original_route_restored_on_failed_or_non_improving_edit": True,
                    },
                }
            )
            allow_editor_route_mutations = True
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
        selected_product_mapped = str(
            blocking_step.get("mapped_product_smiles")
            or _mapped_smiles(selected_product)
        )
        prefix = steps[:step_index]
        feedback = _compact_critic_feedback(
            critique,
            concrete_blockers,
            paper_matched=paper_matched,
        )
        feedback_blockers = [
            dict(value)
            for value in feedback.get("blocking_steps") or []
            if isinstance(value, Mapping)
        ]
        feedback_reasons = [
            str(reason)
            for value in feedback_blockers
            for reason in dict(value.get("assessment") or {}).get("reasons") or []
            if str(reason)
        ]
        feedback_revisions = [
            str(dict(value.get("assessment") or {}).get("suggested_revision") or "")
            for value in feedback_blockers
            if str(
                dict(value.get("assessment") or {}).get("suggested_revision") or ""
            )
        ]
        rejection = {
            "phase": "critic_editor",
            "step_id": step_id,
            "blocking_step_ids": [
                str(row.get("step_id") or "") for row in concrete_blockers
            ],
            "product_smiles": selected_product,
            "failure_reasons": list(
                feedback.get("failure_reasons") or feedback_reasons
            ),
            "repair_actions": list(
                feedback.get("repair_actions") or feedback_revisions
            ),
        }
        record: WorkerRunRecord | None = None
        route_expansions: list[NodeExpansion] | None = None
        surgical_expansion: NodeExpansion | None = None
        editor_feedback = dict(feedback)
        # A surgical Editor response is still a host-replayed ReactionJSON
        # program.  Previously only whole-route mutations received compiler
        # feedback and retries; one malformed local edit therefore collapsed
        # the entire Critic/Editor loop into ``editor_execution_failed`` even
        # when the configured repair budget had ample capacity.  Give both
        # mutation modes the same small, bounded materialization retry window.
        # The outer Critic -> Editor loop remains the authority for the six
        # scientific repair rounds.
        configured_editor_budget = max(0, int(materialization_editor_rounds))
        editor_attempts_used = int(branch.get("editor_attempt_count") or 0)
        remaining_editor_budget = max(
            0, configured_editor_budget - editor_attempts_used
        )
        if remaining_editor_budget < 1:
            branch.setdefault("editor_execution_notes", []).append(
                {
                    "reason": "editor_repair_budget_exhausted",
                    "configured_editor_budget": configured_editor_budget,
                    "editor_attempt_count": editor_attempts_used,
                }
            )
            return False
        editor_attempt_limit = (
            remaining_editor_budget
            if paper_matched
            else min(_MATERIALIZATION_RETRY_LIMIT, remaining_editor_budget)
        )
        # The first Editor attempt sees the accepted Host route.  A failed
        # replacement span then becomes a transient Host-checked working
        # document for the next bounded retry.  It is never committed as the
        # branch route, but retaining it keeps newly introduced atom maps and
        # stable revised step ids from being regenerated on every retry.
        editor_prompt_steps = steps
        editor_base_steps = steps
        # Failed Editor documents receive exact host replay diagnostics before
        # the next bounded attempt. The paper-matched path edits the complete
        # RouteJSON; legacy callers may still request a single-step repair.
        for editor_attempt in range(1, editor_attempt_limit + 1):
            prompt = ""
            editor_views = (
                (
                    ((editor_prompt_steps, target, False), "Codex Editor: dependency-closed RouteJSON repair preserving the StrategyCard"),
                    ((editor_prompt_steps, target, True), "Codex Editor: compact dependency-closed repair"),
                )
                if allow_editor_route_mutations
                else (
                    ((prefix, selected_product, False), "Codex Editor: surgical repair preserving the StrategyCard"),
                    ((prefix[-3:], selected_product, False), "Codex Editor: surgical repair"),
                    ((prefix[-1:], selected_product, False), "Codex Editor: compact surgical repair"),
                    (((), selected_product, False), "Codex Editor: minimal surgical repair"),
                )
            )
            attempt_rejection = dict(rejection)
            if editor_attempt > 1:
                attempt_rejection["editor_materialization_attempt"] = editor_attempt
            for (prefix_view, product_view, compact_editor_context), lens in editor_views:
                candidate_prompt = _node_prompt(
                    target=target,
                    branch_index=int(branch.get("branch_index") or 0),
                    lens=lens,
                    selected_product=product_view,
                    selected_product_mapped=(
                        str(branch.get("target_mapped_smiles") or _mapped_smiles(target))
                        if allow_editor_route_mutations
                        else selected_product_mapped
                    ),
                    steps=prefix_view,
                    open_leaves=[product_view],
                    # Whole-route Editor prompts already carry the Critic and
                    # any replay diagnostic in host_failure_feedback.  Do not
                    # serialize the same long repair brief a second time.
                    prior_rejections=(
                        () if allow_editor_route_mutations else [attempt_rejection]
                    ),
                    repair=True,
                    strategy_card=dict(branch.get("strategy_card") or {}),
                    forbidden_strategy_cards=(),
                    host_failure_feedback=editor_feedback,
                    complete_route_json=allow_editor_route_mutations,
                    editor_route_mutations=allow_editor_route_mutations,
                    compact_editor_context=compact_editor_context,
                    minimum_route_depth=1,
                    paper_matched=paper_matched,
                )
                if len(candidate_prompt.encode("utf-8")) <= max_prompt_bytes:
                    prompt = candidate_prompt
                    break
            if not prompt or not _node_budget_allows(
                records, started=started, quota=quota
            ):
                return False
            # Keep the attempted-call count separate from the applied-edit
            # count.  ``editor_call_count`` is retained for compatibility and
            # means an edit that actually changed the working route; it must
            # not be used to infer that no Editor worker was invoked.
            branch["editor_attempt_count"] = int(
                branch.get("editor_attempt_count") or 0
            ) + 1
            task = _node_task(
                spec,
                prompt=prompt,
                branch_index=int(branch.get("branch_index") or 0),
                node_index=(iteration + 1) * _MATERIALIZATION_RETRY_LIMIT
                + editor_attempt,
                model=str(spec.metadata.get("model") or ""),
                reasoning_effort=str(
                    spec.metadata.get("editor_reasoning_effort")
                    or spec.metadata.get("reasoning_effort")
                    or "medium"
                ),
                timeout_s=_node_call_timeout_s(
                    started,
                    quota,
                    maximum=max_node_call_timeout_s,
                ),
                task_type="route_chemistry_edit",
                paper_matched=paper_matched,
                target_smiles=target,
                selected_product=selected_product,
            )
            try:
                record = self._run_journaled_worker(self.editor_executor, task)
            except Exception as exc:
                branch.setdefault("editor_rejection_diagnostics", []).append(
                    {
                        "task_id": task.task_id,
                        "reason": "editor_worker_exception",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                return False
            records.append(record)
            if not allow_editor_route_mutations:
                surgical_expansion = _expansion_from_record(
                    record,
                    expected_product=selected_product,
                    mapped_product_smiles=selected_product_mapped,
                    require_reaction_operations=require_strategy_graph_edits,
                    single_step_only=bool(require_strategy_graph_edits),
                )
                if surgical_expansion is not None:
                    break
                diagnostic = _expansion_rejection_diagnostic(
                    record,
                    expected_product=selected_product,
                    mapped_product_smiles=selected_product_mapped,
                    require_reaction_operations=require_strategy_graph_edits,
                    require_complete_route_json=False,
                    minimum_route_depth=1,
                    single_step_only=bool(require_strategy_graph_edits),
                )
                attempt_rejection["reason"] = str(
                    diagnostic.get("reason") or "editor_reactionjson_invalid"
                )
                attempt_rejection["replay_diagnostic"] = diagnostic
                branch.setdefault("rejections", []).append(
                    dict(attempt_rejection)
                )
                branch.setdefault("editor_rejection_diagnostics", []).append(
                    {
                        "task_id": task.task_id,
                        "reason": attempt_rejection["reason"],
                        "replay_diagnostic": dict(diagnostic),
                    }
                )
                editor_feedback = {
                    **feedback,
                    "editor_materialization_failure": diagnostic,
                    "editor_instruction": (
                        "Repair this one local ReactionJSON edit against the "
                        "host replay diagnostic. Return a replayable mapped "
                        "precursor boundary; do not reuse the rejected edit."
                    ),
                }
                continue
            route_expansions, diagnostic, mutation_mode = (
                _editor_route_expansions_from_record(
                    record,
                    current_steps=editor_base_steps,
                    mapped_target_smiles=str(
                        branch.get("target_mapped_smiles") or _mapped_smiles(target)
                    ),
                    expected_target_smiles=target,
                )
            )
            if route_expansions:
                branch["_editor_mutation_mode"] = mutation_mode
                break
            attempt_rejection["reason"] = str(
                diagnostic.get("reason") or "editor_route_json_invalid"
            )
            attempt_rejection["replay_diagnostic"] = diagnostic
            branch.setdefault("rejections", []).append(dict(attempt_rejection))
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "task_id": task.task_id,
                    "reason": attempt_rejection["reason"],
                    "replay_diagnostic": dict(diagnostic),
                }
            )
            candidate = _route_json_candidate(record) or {}
            retry_route = None
            replace_span = candidate.get("replace_span")
            if isinstance(replace_span, Mapping):
                retry_route, _ = _apply_replace_span(
                    editor_base_steps,
                    replace_span,
                )
                if retry_route:
                    retry_route = _bind_editor_failed_row_to_host_boundary(
                        retry_route,
                        diagnostic,
                    )
                    editor_base_steps = retry_route
                    editor_prompt_steps = retry_route
            else:
                retry_route = _editor_retry_route_rows(
                    record,
                    diagnostic=diagnostic,
                    mapped_target_smiles=str(
                        branch.get("target_mapped_smiles")
                        or _mapped_smiles(target)
                    ),
                )
                if retry_route:
                    editor_prompt_steps = retry_route
            editor_feedback = {
                **feedback,
                "editor_materialization_failure": diagnostic,
                "editor_instruction": (
                    (
                        "Repair and return the dependency-closed replace_span against this Host diagnostic. "
                        if paper_matched
                        else "Repair the RouteJSON or route_patch against this host diagnostic. "
                    )
                    + (
                        "Use host_selected_open_precursor.mapped_product_smiles exactly at the "
                        "failed row; never reuse guessed atom maps. Every later product must be "
                        "one exact host_open_precursor, and atom or fragment changes still "
                        "require explicit ReactionJSON edits."
                    )
                ),
            }
        if record is None:
            return False
        if allow_editor_route_mutations:
            if not route_expansions:
                return False
            old_depth = len(steps)
            edited_steps: list[dict[str, Any]] = []
            for route_index, expansion in enumerate(route_expansions):
                expansion = replace(
                    expansion,
                    strategy_card=dict(branch.get("strategy_card") or {}),
                )
                edited_steps.append(
                    _step_row(
                        expansion,
                        step_id=(
                            expansion.step_id
                            or f"codex:editor:{iteration + 1}:{route_index + 1}"
                        ),
                        strategy_anchor=_expansion_executes_strategy_anchor(
                            expansion,
                            dict(branch.get("strategy_card") or {}),
                            fallback=route_index == 0,
                        ),
                    )
                )
            mutation_mode = str(
                branch.get("_editor_mutation_mode")
                or ("replace_span" if paper_matched else "full_route_json")
            )
            if mutation_mode == "route_patch_working_prefix":
                # The Editor supplied a local ReactionJSON repair that the
                # host can replay, but its downstream suffix was not updated.
                # Keep it as an independent, explicitly incomplete candidate;
                # do not silently replace the authoritative route with a
                # truncated projection.
                terminal_pairs = _route_terminal_precursor_pairs(route_expansions)
                terminal_precursors = tuple(value[0] for value in terminal_pairs)
                membership = self._stock_membership(terminal_precursors)
                branch["editor_working_route"] = {
                    "status": "host_replayable_prefix",
                    "mutation_mode": mutation_mode,
                    "editor_task_id": task.task_id,
                    "steps": [dict(row) for row in edited_steps],
                    "route_json": _host_route_json_from_steps(edited_steps),
                    "routejson_authority": "host_routejson_dag_compiler",
                    "routejson_replay_complete": True,
                    "routejson_validation_scope": "local_host_replay_only",
                    "route_depth": len(edited_steps),
                    "open_leaf_states": [
                        {"smiles": value, "mapped_smiles": mapped}
                        for value, mapped in terminal_pairs
                        if membership.get(value) is not True
                    ],
                    "complete_in_bound_stock": all(
                        membership.get(value) is True
                        for value in terminal_precursors
                    ),
                    "not_selected_for_topology": True,
                        "semantics": {
                        "host_replayable": True,
                        "full_route_required_before_promotion": True,
                        "grants_no_reaction_proof": True,
                        "grants_no_stock_authority": True,
                    },
                }
                branch.setdefault("editor_execution_notes", []).append(
                    {
                        "reason": "editor_prefix_retained_as_working_route",
                        "editor_task_id": task.task_id,
                        "route_depth": len(edited_steps),
                        "authoritative_route_unchanged": True,
                    }
                )
                branch.pop("_editor_mutation_mode", None)
                return False
            branch["steps"] = edited_steps
            branch["expanded_products"] = {
                _canonical_smiles(row.get("product_smiles"))
                for row in edited_steps
                if _canonical_smiles(row.get("product_smiles"))
            }
            terminal_pairs = _route_terminal_precursor_pairs(route_expansions)
            terminal_precursors = tuple(value[0] for value in terminal_pairs)
            membership = self._stock_membership(terminal_precursors)
            branch["open_leaf_states"] = deque(
                {"smiles": value, "mapped_smiles": mapped}
                for value, mapped in terminal_pairs
                if membership.get(value) is not True
            )
            branch["deferred_tail_leaf_states"] = deque()
            branch["blocked_materializations"] = []
            _sync_open_leaf_projection(branch)
            self._rebuild_branch_or_search_after_editor(
                branch,
                target=target,
                max_depth=max_node_expansions_per_branch,
            )
            # Editor mutations are a separate six-round budget.  They must
            # not consume the Route Builder's independent 25-node depth.
            branch["call_count"] = int(branch.get("call_count") or 0) + 1
            branch["editor_call_count"] = int(
                branch.get("editor_call_count") or 0
            ) + 1
            branch.setdefault("editor_repairs", []).append(
                {
                    "iteration": iteration,
                    "step_id": str(blocking_step.get("step_id") or ""),
                    "blocking_step_ids": [
                        str(row.get("step_id") or "") for row in concrete_blockers
                    ],
                    "editor_task_id": task.task_id,
                    "mutation_mode": mutation_mode,
                    "old_route_depth": old_depth,
                    "new_route_depth": len(edited_steps),
                    "inserted_step_count": max(0, len(edited_steps) - old_depth),
                    "deleted_step_count": max(0, old_depth - len(edited_steps)),
                    "reordered_or_replaced": True,
                }
            )
            branch["complete_in_bound_stock"] = _branch_stock_closed(branch)
            return True

        expansion = surgical_expansion
        if expansion is None:
            return False
        expansion = replace(
            expansion,
            strategy_card=dict(branch.get("strategy_card") or {}),
        )
        edited_step = _step_row(
            expansion,
            step_id=step_id or f"codex:editor:{iteration + 1}:1",
            strategy_anchor=_expansion_executes_strategy_anchor(
                expansion,
                dict(branch.get("strategy_card") or {}),
                fallback=step_index == 0,
            ),
        )
        old_depth = len(steps)
        branch["steps"] = [*prefix, edited_step]
        branch["expanded_products"] = {
            _canonical_smiles(row.get("product_smiles"))
            for row in branch["steps"]
            if _canonical_smiles(row.get("product_smiles"))
        }
        branch["open_leaf_states"] = deque(
            {"smiles": value, "mapped_smiles": mapped}
            for value, mapped in _route_terminal_precursor_pairs((expansion,))
        )
        branch["deferred_tail_leaf_states"] = deque()
        branch["blocked_materializations"] = []
        _sync_open_leaf_projection(branch)
        self._rebuild_branch_or_search_after_editor(
            branch,
            target=target,
            max_depth=max_node_expansions_per_branch,
        )
        branch["call_count"] = int(branch.get("call_count") or 0) + 1
        branch["editor_call_count"] = int(
            branch.get("editor_call_count") or 0
        ) + 1
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
            branch.get("strategy_tree_engine") != "aizynthfinder_mcts"
            and branch.get("open_leaves")
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
                require_complete_route_json=allow_editor_route_mutations,
                materialization_editor_rounds=materialization_editor_rounds,
                paper_matched=paper_matched,
                quota=quota,
                started=started,
            )
            if len(branch.get("steps") or []) <= before:
                break
        branch["complete_in_bound_stock"] = _branch_stock_closed(branch)
        return True

    def _rebuild_branch_or_search_after_editor(
        self,
        branch: dict[str, Any],
        *,
        target: str,
        max_depth: int,
    ) -> None:
        """Replace stale OR propagation after an Editor changes RouteJSON."""

        previous_search = branch.get("_reactionjson_or_search")
        previous_summary = dict(branch.get("reactionjson_or_search") or {})
        if not isinstance(previous_search, ChemEnzyReactionJsonOrSearch):
            return
        steps = [
            dict(row)
            for row in branch.get("steps") or []
            if isinstance(row, Mapping)
        ]
        later_products = {
            _canonical_smiles(row.get("product_smiles"))
            for row in steps[1:]
            if _canonical_smiles(row.get("product_smiles"))
        }
        terminal_precursors = tuple(
            dict.fromkeys(
                precursor
                for row in steps
                for precursor in (
                    _canonical_smiles(value)
                    for value in row.get("precursor_smiles") or []
                )
                if precursor and precursor not in later_products
            )
        )
        membership = self._stock_membership(terminal_precursors)
        rebuilt = ChemEnzyReactionJsonOrSearch(
            target_smiles=target,
            mapped_target_smiles=str(branch.get("target_mapped_smiles") or ""),
            max_depth=max(1, int(max_depth)),
        )
        rebuilt.replay_route(
            steps,
            stock_smiles=(
                value
                for value in terminal_precursors
                if membership.get(value) is True
            ),
        )
        branch["_reactionjson_or_search"] = rebuilt
        _refresh_branch_from_reactionjson_or_search(branch, rebuilt)
        branch.setdefault("reactionjson_or_search_resets", []).append(
            {
                "reason": "critic_editor_route_mutation",
                "previous_summary": previous_summary,
                "rebuilt_summary": dict(
                    branch.get("reactionjson_or_search") or {}
                ),
                "semantics": {
                    "previous_root_solved_is_not_reused": True,
                    "edited_route_replayed_from_host_rows": True,
                },
            }
        )

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
        mapped_target = _mapped_smiles(target)
        branch_mandates = _branch_mandates_for_profile(
            config.strategy_portfolio_mode
        )
        branches: list[dict[str, Any]] = [
            {
                "branch_index": branch_index,
                "lens": branch_mandates[branch_index % len(branch_mandates)],
                "strategy_mandate": branch_mandates[
                    branch_index % len(branch_mandates)
                ],
                "strategy_seed": "",
                "steps": [],
                "open_leaves": deque([target]),
                "open_leaf_states": deque(
                    [{"smiles": target, "mapped_smiles": mapped_target}]
                ),
                "deferred_tail_leaf_states": deque(),
                "target_mapped_smiles": mapped_target,
                "expanded_products": set(),
                "call_count": 0,
                "strategy_call_count": 0,
                "route_call_count": 0,
                "editor_attempt_count": 0,
                "editor_call_count": 0,
                "rejections": [],
                "materialization_failures": {},
                "materialization_diagnostics": [],
                "materialization_editor_history": [],
                "complete_in_bound_stock": False,
                "strategy_tree_engine": config.strategy_tree_engine,
                "strategy_card": {},
                "root_strategy_card": {},
                "strategy_milestone_cards": [],
                "strategy_milestone_attempts": [],
                "strategy_milestone_generation_count": 0,
                "strategy_critic_call_count": 0,
                "strategy_critic": {},
                "key_event_critic_call_count": 0,
                "key_event_critic_completed": False,
                "key_event_critic_history": [],
                "pending_key_event_feedback": {},
                "chemical_critic": {},
            }
            for branch_index in range(config.strategy_branch_count)
        ]
        records: list[WorkerRunRecord] = []

        # The Critic/Editor phase is mandatory in the paper protocol.  Its
        # reservation must exist before the three StrategyCard calls start;
        # otherwise strategy generation can consume the whole director wall
        # budget and leave Route Builder with only a tiny timeout tail.  The
        # reservation is intentionally based on the configured branch count
        # (the worst case) and is therefore safe even when a seed later fails.
        # A tight Strategy -> Builder canary intentionally ends when its model
        # ceiling is exhausted. Reserving the later Critic/Editor phase in
        # that envelope would starve every Builder branch (1 portfolio call +
        # one node call per branch already consumes the whole ceiling).
        builder_only_model_ceiling = 1 + (
            int(config.strategy_branch_count)
            * int(config.max_node_expansions_per_branch)
        )
        builder_only_canary = bool(
            config.paper_matched_reach_profile
            and int(quota.model_invocations) <= builder_only_model_ceiling
        )
        critic_reserve_slots = (
            0
            if builder_only_canary
            else max(0, int(config.strategy_branch_count))
        )
        critic_rounds = max(0, int(config.max_route_local_repair_rounds))
        paper_repair_budget = bool(config.paper_matched_reach_profile)
        key_event_critic_calls_per_branch = (
            _MAX_KEY_EVENT_CRITIC_CALLS_PER_BRANCH
            if config.enable_key_event_critic
            else 0
        )
        # A strict paper route reserves one initial Critic plus one Editor and
        # one re-Critic for every allowed round.  Compatibility profiles keep
        # their former one-pair reservation.
        calls_per_branch = (
            1 + 2 * critic_rounds if paper_repair_budget else 2
        ) + key_event_critic_calls_per_branch
        critic_editor_call_reserve = critic_reserve_slots * calls_per_branch
        critic_editor_wall_reserve = (
            quota.wall_time_s
            * (
                _PAPER_CRITIC_EDITOR_WALL_FRACTION
                if paper_repair_budget
                else _CRITIC_EDITOR_WALL_FRACTION
            )
            if critic_reserve_slots
            else 0.0
        )
        critic_input_reserve = critic_reserve_slots * (
            (
                (critic_rounds + 1) * _CRITIC_INPUT_TOKEN_RESERVE
                + critic_rounds * _EDITOR_INPUT_TOKEN_RESERVE
            )
            if paper_repair_budget
            else (_CRITIC_INPUT_TOKEN_RESERVE + _EDITOR_INPUT_TOKEN_RESERVE)
        ) + critic_reserve_slots * key_event_critic_calls_per_branch * (
            _CRITIC_INPUT_TOKEN_RESERVE
        )
        critic_output_reserve = critic_reserve_slots * (
            (
                (critic_rounds + 1) * _CRITIC_OUTPUT_TOKEN_RESERVE
                + critic_rounds * _EDITOR_OUTPUT_TOKEN_RESERVE
            )
            if paper_repair_budget
            else (_CRITIC_OUTPUT_TOKEN_RESERVE + _EDITOR_OUTPUT_TOKEN_RESERVE)
        ) + critic_reserve_slots * key_event_critic_calls_per_branch * (
            _CRITIC_OUTPUT_TOKEN_RESERVE
        )
        route_quota = replace(
            quota,
            wall_time_s=max(
                0.0,
                float(quota.wall_time_s) - float(critic_editor_wall_reserve),
            ),
        )

        # Phase 1 records strategy hypotheses only.  It deliberately does not
        # ask for precursor structures or ReactionJSON; those belong to the
        # Route Builder boundary below.  A graph-edit failure must therefore
        # never erase an already selected strategic hypothesis.
        paper_portfolio_attempted = bool(
            config.paper_matched_reach_profile and len(branches) == 3
        )
        if paper_portfolio_attempted and _node_budget_allows(
            records,
            started=started,
            quota=route_quota,
        ):
            self._seed_paper_strategy_portfolio(
                spec,
                target=target,
                branches=branches,
                records=records,
                max_prompt_bytes=config.max_node_prompt_bytes,
                max_node_call_timeout_s=config.max_node_call_timeout_s,
                quota=route_quota,
                started=started,
            )
            if (
                config.enable_strategy_portfolio_critic
                and all(branch.get("strategy_card") for branch in branches)
                and _node_budget_allows(
                    records,
                    started=started,
                    quota=route_quota,
                )
            ):
                self._review_paper_strategy_portfolio(
                    spec,
                    target=target,
                    branches=branches,
                    records=records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    quota=route_quota,
                    started=started,
                )
        while not self._cancelled() and any(
            not branch["strategy_card"] for branch in branches
        ) and not paper_portfolio_attempted:
            progressed = False
            for branch in branches:
                if branch["strategy_card"]:
                    continue
                if self._cancelled() or not _node_budget_allows(
                    records,
                    started=started,
                    quota=route_quota,
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
                    quota=route_quota,
                    started=started,
                    forbidden_strategy_cards=(
                        ()
                        if config.paper_matched_reach_profile
                        else _accepted_strategy_cards(
                            branches, exclude_index=int(branch["branch_index"])
                        )
                    ),
                    paper_matched=config.paper_matched_reach_profile,
                )
                progressed = True
            if not progressed or not _node_budget_allows(
                records,
                started=started,
                quota=route_quota,
            ):
                break

        seeded = [branch for branch in branches if branch["strategy_card"]]

        # Phase 2 expands the already committed strategies round-robin.  Route
        # state remains isolated.  The paper profile co-generates all three
        # cards in one portfolio call; compatibility profiles enforce the same
        # orthogonality while selecting their cards serially.
        critic_slots = 0 if builder_only_canary else len(seeded)
        # Route Builder materialization (including its compiler-feedback
        # Editor retries) uses the already bounded quota above.  The Critic
        # phase keeps the original quota and therefore sees the preserved
        # wall-clock budget.  Recompute invocation/token reserves from the
        # actually seeded branches for the round-robin route loop; the wall
        # reserve remains worst-case and is deliberately not restored after a
        # seed failure.
        critic_editor_call_reserve = critic_slots * calls_per_branch
        critic_input_reserve = critic_slots * (
            (
                (critic_rounds + 1) * _CRITIC_INPUT_TOKEN_RESERVE
                + critic_rounds * _EDITOR_INPUT_TOKEN_RESERVE
            )
            if paper_repair_budget
            else (_CRITIC_INPUT_TOKEN_RESERVE + _EDITOR_INPUT_TOKEN_RESERVE)
        ) + critic_slots * key_event_critic_calls_per_branch * (
            _CRITIC_INPUT_TOKEN_RESERVE
        )
        critic_output_reserve = critic_slots * (
            (
                (critic_rounds + 1) * _CRITIC_OUTPUT_TOKEN_RESERVE
                + critic_rounds * _EDITOR_OUTPUT_TOKEN_RESERVE
            )
            if paper_repair_budget
            else (_CRITIC_OUTPUT_TOKEN_RESERVE + _EDITOR_OUTPUT_TOKEN_RESERVE)
        ) + critic_slots * key_event_critic_calls_per_branch * (
            _CRITIC_OUTPUT_TOKEN_RESERVE
        )
        if config.strategy_tree_engine == "aizynthfinder_mcts":
            records.extend(
                self._expand_seeded_branches_aizynthfinder(
                    spec,
                    target=target,
                    seeded=seeded,
                    existing_records=records,
                    route_quota=route_quota,
                    critic_editor_call_reserve=critic_editor_call_reserve,
                    critic_input_reserve=critic_input_reserve,
                    critic_output_reserve=critic_output_reserve,
                    config=config,
                    started=started,
                )
            )
        elif config.strategy_branch_workers > 1 and len(seeded) > 1:
            records.extend(
                self._expand_seeded_branches_parallel(
                    spec,
                    target=target,
                    seeded=seeded,
                    existing_records=records,
                    route_quota=route_quota,
                    critic_editor_call_reserve=critic_editor_call_reserve,
                    critic_input_reserve=critic_input_reserve,
                    critic_output_reserve=critic_output_reserve,
                    config=config,
                    started=started,
                )
            )
        else:
            while not self._cancelled():
                progressed = False
                for branch in branches:
                    if self._cancelled() or not _node_budget_allows(
                        records,
                        started=started,
                        quota=route_quota,
                        reserve_model_invocations=critic_editor_call_reserve,
                        reserve_input_tokens=critic_input_reserve,
                        reserve_output_tokens=critic_output_reserve,
                        # `route_quota` already subtracts the protected Critic /
                        # Editor wall slice; reserving it a second time would
                        # starve the final route node at tight test budgets.
                        reserve_wall_time_s=0.0,
                    ):
                        break
                    if (
                        int(branch["route_call_count"])
                        >= config.max_node_expansions_per_branch
                        or not _branch_has_expandable_leaf(branch)
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
                        max_reactionjson_candidates_per_node=(
                            config.max_reactionjson_candidates_per_node
                        ),
                        max_or_search_depth=config.max_node_expansions_per_branch,
                        require_strategy_graph_edits=config.require_strategy_graph_edits,
                        require_complete_route_json=config.require_complete_route_json,
                        materialization_editor_rounds=config.max_route_local_repair_rounds,
                        paper_matched=config.paper_matched_reach_profile,
                        quota=route_quota,
                        started=started,
                    )
                    progressed = True
                    if _branch_stock_closed(branch):
                        branch["complete_in_bound_stock"] = True
                        if config.stop_on_first_stock_closed_branch:
                            branch["portfolio_early_stop_triggered"] = True
                            break
                if config.stop_on_first_stock_closed_branch and any(
                    branch["complete_in_bound_stock"] for branch in seeded
                ):
                    break
                if seeded and all(
                    _branch_stock_closed(branch)
                    or not _branch_has_expandable_leaf(branch)
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
                    quota=route_quota,
                    reserve_model_invocations=critic_editor_call_reserve,
                    reserve_input_tokens=critic_input_reserve,
                    reserve_output_tokens=critic_output_reserve,
                    reserve_wall_time_s=0.0,
                ):
                    break
        # Keep the live OR state through the Critic/Editor phase.  Projecting
        # with ``_public_branch`` here used to discard the private tree while
        # retaining only its stale summary, so an Editor mutation could neither
        # rebuild nor continue the original search state.  ``_compile_plan`` is
        # the serialization boundary and already emits public data only.
        return branches, records

    def _expand_seeded_branches_aizynthfinder(
        self,
        spec: AgentSpec,
        *,
        target: str,
        seeded: list[dict[str, Any]],
        existing_records: list[WorkerRunRecord],
        route_quota: _NodeCallBudget,
        critic_editor_call_reserve: int,
        critic_input_reserve: int,
        critic_output_reserve: int,
        config: DirectorConfig,
        started: float,
    ) -> list[WorkerRunRecord]:
        """Execute each frozen strategy inside a real AiZ MCTS/UCB tree.

        AiZynthFinder runs in its Python 3.11 sidecar because its onnxruntime
        build cannot be imported into the host Python 3.12 process.  The
        sidecar owns selection, sibling actions, cycle pruning and
        back-propagation; this host remains the only authority allowed to call
        Codex or replay ReactionJSON into mapped precursor structures.
        """

        if not seeded:
            return []
        usage = _aggregate_usage(existing_records, elapsed_s=0.0)
        branch_count = len(seeded)
        available_calls = max(
            0,
            int(route_quota.model_invocations)
            - int(usage["model_invocations"])
            - max(0, int(critic_editor_call_reserve)),
        )
        available_input = max(
            0,
            int(route_quota.input_tokens)
            - int(usage["input_tokens"])
            - max(0, int(critic_input_reserve)),
        )
        available_output = max(
            0,
            int(route_quota.output_tokens)
            - int(usage["output_tokens"])
            - max(0, int(critic_output_reserve)),
        )
        call_allocations = _balanced_branch_allocations(
            available_calls,
            branch_count,
            cap=max(1, int(config.max_node_expansions_per_branch)),
        )
        input_allocations = _balanced_branch_allocations(
            available_input, branch_count
        )
        output_allocations = _balanced_branch_allocations(
            available_output, branch_count
        )
        if config.paper_matched_reach_profile:
            required_calls = int(config.max_node_expansions_per_branch)
            for index, branch in enumerate(seeded):
                if int(call_allocations[index]) >= required_calls:
                    continue
                branch["paper_policy_budget_failure"] = {
                    "reason": "paper_policy_call_budget_preflight_insufficient",
                    "required_calls": required_calls,
                    "actual_calls": 0,
                    "allocated_calls": int(call_allocations[index]),
                    "stock_closed": False,
                }

        def advance(
            branch: dict[str, Any],
            *,
            call_allowance: int,
            input_allowance: int,
            output_allowance: int,
        ) -> list[WorkerRunRecord]:
            local_records: list[WorkerRunRecord] = []
            route_records: list[WorkerRunRecord] = []
            local_quota = _NodeCallBudget(
                model_invocations=max(0, int(call_allowance)),
                input_tokens=max(0, int(input_allowance)),
                output_tokens=max(0, int(output_allowance)),
                wall_time_s=route_quota.wall_time_s,
            )
            local_total_quota = replace(
                local_quota,
                model_invocations=(
                    max(0, int(call_allowance))
                    + (
                        _MAX_KEY_EVENT_CRITIC_CALLS_PER_BRANCH
                        if config.enable_key_event_critic
                        else 0
                    )
                    + max(
                        0,
                        int(config.max_strategic_milestones_per_branch) - 1,
                    )
                ),
                input_tokens=(
                    max(0, int(input_allowance))
                    + (
                        _MAX_KEY_EVENT_CRITIC_CALLS_PER_BRANCH
                        * _CRITIC_INPUT_TOKEN_RESERVE
                        if config.enable_key_event_critic
                        else 0
                    )
                ),
                output_tokens=(
                    max(0, int(output_allowance))
                    + (
                        _MAX_KEY_EVENT_CRITIC_CALLS_PER_BRANCH
                        * _CRITIC_OUTPUT_TOKEN_RESERVE
                        if config.enable_key_event_critic
                        else 0
                    )
                ),
            )
            branch_index = int(branch["branch_index"])
            root_strategy_card = dict(
                branch.get("root_strategy_card")
                or branch.get("strategy_card")
                or {}
            )
            strategy_id = str(
                root_strategy_card.get("strategy_id")
                or root_strategy_card.get("strategy_digest")
                or f"strategy-{branch_index + 1}"
            )
            strategy_text = json.dumps(
                root_strategy_card,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            # AiZ may revisit an empty MCTS node after a host-replayed
            # ReactionJSON action fails to instantiate an advancing child.
            # Keep only node-local negative memory: the latest no-progress
            # attempt for the next prompt and candidate identities for exact
            # duplicate suppression. This does not rank or admit chemistry;
            # it closes the feedback edge between AiZ selection and Builder.
            attempted_policy_moves: dict[str, dict[str, dict[str, Any]]] = {}
            pending_policy_feedback: dict[str, dict[str, Any]] = {}

            def handle_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
                if self._cancelled():
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": "host_cancelled",
                    }
                route_budget_reason = _node_budget_block_reason(
                    route_records,
                    started=started,
                    quota=local_quota,
                )
                total_budget_reason = _node_budget_block_reason(
                    local_records,
                    started=started,
                    quota=local_total_quota,
                )
                if route_budget_reason or total_budget_reason:
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": (
                            f"route_builder_{route_budget_reason}"
                            if route_budget_reason
                            else f"branch_total_{total_budget_reason}"
                        ),
                    }
                expandable = [
                    _canonical_smiles(value)
                    for value in request.get("expandable_smiles") or []
                ]
                mapped_values = [
                    str(value or "").strip()
                    for value in request.get("expandable_mapped_smiles") or []
                ]
                prompt_steps = [
                    dict(row)
                    for row in request.get("route_steps") or []
                    if isinstance(row, Mapping)
                ]
                # The request path is already selected by AiZ and every row
                # has already crossed the Host replay boundary.  Persist the
                # deepest such prefix before making another paid call so a
                # later sidecar/provider failure cannot erase completed work.
                durable_depth = int(
                    branch.get("sidecar_durable_prefix_step_count") or 0
                )
                if len(prompt_steps) > durable_depth:
                    branch["steps"] = [dict(row) for row in prompt_steps]
                    branch["open_leaf_states"] = deque(
                        {
                            "smiles": smiles,
                            "mapped_smiles": (
                                mapped_values[index]
                                if index < len(mapped_values)
                                and mapped_values[index]
                                else _mapped_smiles(smiles)
                            ),
                        }
                        for index, smiles in enumerate(expandable)
                        if smiles
                    )
                    branch["expanded_products"] = {
                        product
                        for row in prompt_steps
                        if (
                            product := _canonical_smiles(
                                row.get("product_smiles")
                            )
                        )
                    }
                    branch["sidecar_durable_prefix_step_count"] = len(
                        prompt_steps
                    )
                    _sync_open_leaf_projection(branch)
                selected_index = next(
                    (index for index, value in enumerate(expandable) if value),
                    None,
                )
                if selected_index is None:
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": "no_canonical_expandable_molecule",
                    }
                selected = expandable[selected_index]
                selected_mapped = (
                    mapped_values[selected_index]
                    if selected_index < len(mapped_values)
                    else ""
                ) or _mapped_smiles(selected)
                policy_context_id = _aiz_policy_state_fingerprint(
                    selected_leaf_mapped=selected_mapped,
                    route_steps=prompt_steps,
                )
                prior_policy_feedback = pending_policy_feedback.pop(
                    policy_context_id,
                    None,
                )
                if prior_policy_feedback:
                    branch.setdefault("rejections", []).append(
                        dict(prior_policy_feedback)
                    )
                rejected = list(branch.get("rejections") or [])
                # In the paper-matched arm the Strategy is a steering query,
                # not a host-computed graph-completion state.  In particular,
                # mapped key-bond progress must not choose the next AiZ leaf or
                # switch the policy supplied to the Builder.  Retain the
                # historical multi-milestone selector only for compatibility
                # profiles that explicitly enable that AutoPlanner extension.
                active_strategy_card = (
                    root_strategy_card
                    if config.paper_matched_reach_profile
                    else _active_strategy_card_for_leaf(
                        root_strategy_card=root_strategy_card,
                        steps=prompt_steps,
                        selected_product_mapped=selected_mapped,
                    )
                )
                if (
                    not config.paper_matched_reach_profile
                    and
                    int(config.max_strategic_milestones_per_branch) > 1
                    and _strategy_anchor_fulfilled_for_card(
                        prompt_steps, root_strategy_card
                    )
                    and active_strategy_card is root_strategy_card
                    and int(
                        branch.get("strategy_milestone_generation_count") or 0
                    )
                    < int(config.max_strategic_milestones_per_branch) - 1
                    and _node_budget_allows(
                        local_records,
                        started=started,
                        quota=local_total_quota,
                    )
                ):
                    generated = self._generate_upstream_strategy_milestone(
                        spec,
                        campaign_target=target,
                        selected_product=selected,
                        selected_product_mapped=selected_mapped,
                        branch=branch,
                        route_steps=prompt_steps,
                        records=local_records,
                        max_prompt_bytes=config.max_node_prompt_bytes,
                        max_node_call_timeout_s=config.max_node_call_timeout_s,
                        quota=local_total_quota,
                        started=started,
                    )
                    if generated is not None:
                        active_strategy_card = generated
                if not _node_budget_allows(
                    local_records,
                    started=started,
                    quota=local_total_quota,
                ):
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": "branch_total_"
                        + _node_budget_block_reason(
                            local_records,
                            started=started,
                            quota=local_total_quota,
                        ),
                    }
                call_index = int(branch.get("route_call_count") or 0) + 1
                prompt = _node_prompt(
                    target=target,
                    branch_index=branch_index,
                    lens=str(branch.get("lens") or ""),
                    selected_product=selected,
                    selected_product_mapped=selected_mapped,
                    steps=prompt_steps,
                    open_leaves=expandable,
                    prior_rejections=rejected,
                    repair=False,
                    strategy_card=active_strategy_card,
                    forbidden_strategy_cards=(),
                    host_failure_feedback={
                        "pending_checkpoint_feedback": dict(
                            branch.get("pending_key_event_feedback") or {}
                        )
                    },
                    complete_route_json=False,
                    minimum_route_depth=1,
                    max_reactionjson_candidates=(
                        config.max_reactionjson_candidates_per_node
                    ),
                    paper_matched=config.paper_matched_reach_profile,
                )
                if len(prompt.encode("utf-8")) > config.max_node_prompt_bytes:
                    prompt = _node_prompt(
                        target=target,
                        branch_index=branch_index,
                        lens=str(branch.get("lens") or ""),
                        selected_product=selected,
                        selected_product_mapped=selected_mapped,
                        steps=prompt_steps[-6:],
                        open_leaves=expandable[:12],
                        prior_rejections=rejected[-4:],
                        repair=False,
                        strategy_card=active_strategy_card,
                        forbidden_strategy_cards=(),
                        host_failure_feedback={
                            "pending_checkpoint_feedback": dict(
                                branch.get("pending_key_event_feedback") or {}
                            )
                        },
                        complete_route_json=False,
                        minimum_route_depth=1,
                        max_reactionjson_candidates=(
                            config.max_reactionjson_candidates_per_node
                        ),
                        paper_matched=config.paper_matched_reach_profile,
                    )
                _assert_node_prompt_size(prompt, config.max_node_prompt_bytes)
                branch["route_call_count"] = call_index
                branch["call_count"] = int(branch.get("call_count") or 0) + 1
                task = _node_task(
                    spec,
                    prompt=prompt,
                    branch_index=branch_index,
                    # _node_task accepts a zero-based ordinal and renders the
                    # human-facing task id as ordinal + 1. route_call_count is
                    # already one-based, so normalize it at this boundary.
                    node_index=call_index - 1,
                    model=str(spec.metadata.get("model") or ""),
                    reasoning_effort=str(
                        spec.metadata.get("reasoning_effort") or "medium"
                    ),
                    timeout_s=_node_call_timeout_s(
                        started,
                        local_quota,
                        maximum=config.max_node_call_timeout_s,
                    ),
                    paper_matched=config.paper_matched_reach_profile,
                    target_smiles=target,
                    selected_product=selected,
                )
                record = self._run_journaled_worker(self.node_executor, task)
                local_records.append(record)
                route_records.append(record)
                compiled, candidate_rejections = (
                    _reactionjson_candidates_from_record(
                        record,
                        expected_product=selected,
                        mapped_product_smiles=selected_mapped,
                        require_reaction_operations=True,
                        compiler=self.routejson_compiler,
                        max_candidates=(
                            config.max_reactionjson_candidates_per_node
                        ),
                        reserved_atom_maps=_route_atom_map_namespace(
                            prompt_steps,
                            selected_mapped,
                        ),
                    )
                )
                branch.setdefault("reactionjson_candidate_batches", []).append(
                    {
                        "node": call_index,
                        "product_smiles": selected,
                        "reported_candidates": (
                            len(compiled) + len(candidate_rejections)
                        ),
                        "compiled_candidates": len(compiled),
                        "rejected_candidates": len(candidate_rejections),
                        "search_engine": "aizynthfinder_mcts",
                    }
                )
                for diagnostic in candidate_rejections:
                    row = {
                        "phase": "route_builder_candidate",
                        "node": call_index,
                        "product_smiles": selected,
                        **dict(diagnostic),
                    }
                    branch.setdefault("rejections", []).append(row)
                    branch.setdefault("materialization_diagnostics", []).append(
                        row
                    )
                candidates: list[dict[str, Any]] = []
                connected_ancestors = set(
                    _connected_path_ancestor_smiles(
                        prompt_steps,
                        selected,
                        selected_mapped,
                    )
                )
                for item in compiled:
                    expansion = replace(
                        item.expansion,
                        strategy_card=active_strategy_card,
                    )
                    prior_moves = attempted_policy_moves.setdefault(
                        policy_context_id,
                        {},
                    )
                    candidate_identity = str(
                        item.candidate_key
                        or _key_event_fingerprint(
                            {
                                "mapped_product_smiles": (
                                    expansion.mapped_product_smiles
                                ),
                                "mapped_precursor_smiles": list(
                                    expansion.mapped_precursor_smiles
                                ),
                                "reaction_operations": list(
                                    expansion.reaction_operations
                                ),
                            }
                        )
                    )
                    if candidate_identity in prior_moves:
                        repeated = dict(prior_moves[candidate_identity])
                        row = {
                            "phase": "aizynthfinder_mcts_feedback",
                            "node": call_index,
                            "product_smiles": selected,
                            "candidate_id": item.candidate_id,
                            "reason": "candidate_repeats_same_mcts_state_edit",
                            "mcts_state_fingerprint": policy_context_id,
                            "attempted_net_edits": list(
                                repeated.get("attempted_net_edits") or []
                            ),
                            "authority": "aizynthfinder_selected_path_state",
                        }
                        branch.setdefault("rejections", []).append(row)
                        branch.setdefault(
                            "materialization_diagnostics", []
                        ).append(row)
                        continue
                    returned_ancestors = sorted(
                        {
                            precursor
                            for precursor in (
                                _canonical_smiles(value)
                                for value in expansion.precursor_smiles
                            )
                            if precursor and precursor in connected_ancestors
                        }
                    )
                    if returned_ancestors:
                        # AiZ remains the cycle-pruning authority.  This row is
                        # compact feedback for the next paid policy call, not a
                        # second admission gate or route-completion signal.
                        branch.setdefault("rejections", []).append(
                            {
                                "phase": "aizynthfinder_cycle_feedback",
                                "node": call_index,
                                "product_smiles": selected,
                                "candidate_id": item.candidate_id,
                                "reason": "candidate_returns_to_ancestor",
                                "ancestor_smiles": returned_ancestors,
                                "authority": "diagnostic_only",
                            }
                        )
                    step = _step_row(
                        expansion,
                        step_id=(
                            expansion.step_id
                            or (
                                f"codex:branch:{branch_index + 1}:node:"
                                f"{call_index}:candidate:"
                                f"{item.candidate_index + 1}"
                            )
                        ),
                        strategy_anchor=_expansion_executes_strategy_anchor(
                            expansion,
                            active_strategy_card,
                            fallback=(
                                not config.paper_matched_reach_profile
                                and not bool(prompt_steps)
                            ),
                        ),
                        strategy_milestone_index=_strategy_milestone_index(
                            branch, active_strategy_card
                        ),
                    )
                    if (
                        config.enable_key_event_critic
                        and not bool(branch.get("key_event_critic_completed"))
                        and int(branch.get("key_event_critic_call_count") or 0)
                        < _MAX_KEY_EVENT_CRITIC_CALLS_PER_BRANCH
                        and _step_claims_strategy_key_event(
                            step, active_strategy_card
                        )
                    ):
                        focus_step_id = str(step.get("step_id") or "")
                        fingerprint = _key_event_fingerprint(step)
                        audit_steps = [*prompt_steps, step]
                        critic_prompt = _bounded_critic_prompt(
                            target=target,
                            branch_index=branch_index,
                            strategy_card=active_strategy_card,
                            steps=audit_steps,
                            maximum_bytes=config.max_node_prompt_bytes,
                            paper_matched=True,
                            audit_kind="key_event",
                            focus_step_id=focus_step_id,
                        )
                        history_row: dict[str, Any] = {
                            "focus_step_id": focus_step_id,
                            "product_smiles": selected,
                            "candidate_id": item.candidate_id,
                            "fingerprint": fingerprint,
                        }
                        if critic_prompt is None:
                            history_row["status"] = "prompt_unavailable"
                            branch.setdefault(
                                "key_event_critic_history", []
                            ).append(history_row)
                        elif not _node_budget_allows(
                            local_records,
                            started=started,
                            quota=local_total_quota,
                        ):
                            history_row["status"] = "budget_unavailable"
                            branch.setdefault(
                                "key_event_critic_history", []
                            ).append(history_row)
                        else:
                            critic_task = _critic_task(
                                spec,
                                prompt=critic_prompt,
                                branch_index=branch_index,
                                iteration=call_index,
                                timeout_s=_node_call_timeout_s(
                                    started,
                                    local_total_quota,
                                    maximum=config.critic_call_timeout_s,
                                ),
                                paper_matched=True,
                                target_smiles=target,
                                audit_kind="key_event",
                                focus_step_id=focus_step_id,
                            )
                            branch["critic_call_count"] = int(
                                branch.get("critic_call_count") or 0
                            ) + 1
                            branch["key_event_critic_call_count"] = int(
                                branch.get("key_event_critic_call_count") or 0
                            ) + 1
                            try:
                                critic_record = self._run_journaled_worker(
                                    self.critic_executor, critic_task
                                )
                            except Exception as exc:
                                critic_record = WorkerRunRecord(
                                    run_id=f"{critic_task.task_id}:run",
                                    task_id=critic_task.task_id,
                                    case_id=critic_task.case_id,
                                    status="worker_error",
                                    backend="critic_executor",
                                    stderr=f"{type(exc).__name__}: {exc}",
                                    output_validation={
                                        "accepted": False,
                                        "reasons": [
                                            "key_event_critic_execution_failed"
                                        ],
                                    },
                                )
                            local_records.append(critic_record)
                            critique = _critique_from_record(critic_record)
                            focus_assessment = _key_event_focus_assessment(
                                critique, focus_step_id
                            )
                            checkpoint_match = (
                                critique.get("checkpoint_match") is True
                                and focus_assessment is not None
                            )
                            checkpoint_rejected = (
                                focus_assessment is not None
                                and (
                                    str(
                                        focus_assessment.get("verdict") or ""
                                    )
                                    == "reject"
                                    or focus_assessment.get("blocking") is True
                                )
                            )
                            history_row.update(
                                {
                                    "task_id": critic_task.task_id,
                                    "status": (
                                        "rejected"
                                        if checkpoint_rejected
                                        else (
                                            "completed"
                                            if checkpoint_match
                                            else "not_checkpoint"
                                        )
                                    ),
                                    "critic_status": str(
                                        critique.get("status") or "unavailable"
                                    ),
                                    "checkpoint_match": checkpoint_match,
                                    "assessment": dict(focus_assessment or {}),
                                }
                            )
                            branch.setdefault(
                                "key_event_critic_history", []
                            ).append(history_row)
                            if focus_assessment is not None and (
                                not checkpoint_match or checkpoint_rejected
                            ):
                                branch["pending_key_event_feedback"] = {
                                    "checkpoint_match": checkpoint_match,
                                    "focus_step_id": focus_step_id,
                                    "blocking_type": str(
                                        focus_assessment.get("blocking_type")
                                        or "none"
                                    ),
                                    "reasons": [
                                        str(value)[:260]
                                        for value in focus_assessment.get(
                                            "reasons"
                                        )
                                        or []
                                        if str(value).strip()
                                    ][:2],
                                    "suggested_revision": str(
                                        focus_assessment.get(
                                            "suggested_revision"
                                        )
                                        or ""
                                    )[:420],
                                }
                            if not checkpoint_match:
                                # A benign scheduling mismatch remains a
                                # preparatory action.  A false substitute may
                                # still be locally rejected below by the
                                # Critic's explicit blocking verdict.
                                step["checkpoint_relation"] = "preparatory"
                            elif not checkpoint_rejected:
                                # A rejected key-event proposal has not
                                # completed the checkpoint.  Keep the second
                                # reserved audit available for the Builder's
                                # corrected candidate on the same leaf.
                                branch["key_event_critic_completed"] = True
                                branch["pending_key_event_feedback"] = {}
                            if checkpoint_rejected:
                                chemical_rejection = {
                                    "focus_step_id": focus_step_id,
                                    "reasons": [
                                        str(value)
                                        for value in focus_assessment.get(
                                            "reasons"
                                        )
                                        or []
                                        if str(value)
                                    ][:2],
                                    "suggested_revision": str(
                                        focus_assessment.get(
                                            "suggested_revision"
                                        )
                                        or ""
                                    ),
                                }
                                branch.setdefault("rejections", []).append(
                                    {
                                        "phase": "key_event_critic",
                                        "reason": "key_event_critic_reject",
                                        "product_smiles": selected,
                                        "candidate_id": item.candidate_id,
                                        "focus_step_id": focus_step_id,
                                        "chemical_rejection": chemical_rejection,
                                    }
                                )
                                # Preserve the accepted prefix and ask the
                                # same leaf for a corrected candidate. AiZ
                                # owns the empty-action retry and MCTS state.
                                continue
                    candidates.append(
                        {
                            "candidate_id": item.candidate_id,
                            "product_smiles": expansion.product_smiles,
                            "mapped_product_smiles": (
                                expansion.mapped_product_smiles
                            ),
                            "precursor_smiles": list(
                                expansion.precursor_smiles
                            ),
                            "mapped_precursor_smiles": list(
                                expansion.mapped_precursor_smiles
                            ),
                            "route_step": step,
                            "prior": item.score,
                            "candidate_key": item.candidate_key,
                        }
                    )
                    attempted_net_edits = [
                        dict(operation)
                        for operation in normalize_reaction_operations(
                            expansion.reaction_operations
                        )
                    ]
                    attempt = {
                        "candidate_id": item.candidate_id,
                        "attempted_net_edits": attempted_net_edits,
                    }
                    prior_moves[candidate_identity] = attempt
                    pending_policy_feedback[policy_context_id] = {
                        "phase": "aizynthfinder_mcts_feedback",
                        "node": call_index,
                        "product_smiles": selected,
                        "candidate_id": item.candidate_id,
                        "reason": "candidate_did_not_advance_selected_mcts_path",
                        "mcts_state_fingerprint": policy_context_id,
                        "attempted_net_edits": attempted_net_edits,
                        "authority": "aizynthfinder_selected_path_state",
                    }
                return {
                    "candidates": candidates,
                    "model_call_consumed": True,
                }

            try:
                result = run_aizynthfinder_strategy_branch_sidecar(
                    target_smiles=target,
                    strategy_id=strategy_id,
                    strategy_text=strategy_text,
                    request_handler=handle_request,
                    stock_index_path=self.aizynthfinder_strategy_stock_index,
                    inline_stock_smiles=(
                        self.aizynthfinder_strategy_inline_stock_smiles
                    ),
                    python_executable=(
                        self.aizynthfinder_strategy_python_executable
                    ),
                    max_policy_calls=max(1, int(call_allowance)),
                    max_candidates_per_call=(
                        config.max_reactionjson_candidates_per_node
                    ),
                    max_transforms=config.max_node_expansions_per_branch,
                    timeout_s=max(
                        1.0,
                        _remaining_node_wall_time(started, local_quota),
                    ),
                    cancel_event=self.cancel_event,
                )
            except Exception as exc:
                recovered_depth = int(
                    branch.get("sidecar_durable_prefix_step_count") or 0
                )
                branch.setdefault("rejections", []).append(
                    {
                        "phase": "aizynthfinder_strategy_search",
                        "reason": "aizynthfinder_strategy_sidecar_failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "fallback_used": False,
                    }
                )
                branch["aizynthfinder_strategy_search"] = {
                    "engine": "AiZynthFinder.MctsSearchTree",
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_used": False,
                    "selected_depth": recovered_depth,
                    "selected_open_leaves": len(
                        branch.get("open_leaf_states") or []
                    ),
                }
                if recovered_depth > 0 and branch.get("steps"):
                    branch["sidecar_recovered_prefix"] = True
                    branch["complete_in_bound_stock"] = False
                    _sync_open_leaf_projection(branch)
                return local_records

            branch["steps"] = [
                dict(row)
                for row in result.get("route_steps") or []
                if isinstance(row, Mapping)
            ]
            selected_cards = _ordered_strategy_cards_from_steps(
                root_strategy_card=root_strategy_card,
                steps=branch["steps"],
            )
            branch["strategy_milestone_cards"] = selected_cards
            branch["strategic_milestone_count"] = sum(
                _strategy_anchor_fulfilled_for_card(branch["steps"], card)
                for card in selected_cards
            )
            branch["strategy_anchor_diagnostics"] = [
                {
                    "strategy_id": str(card.get("strategy_id") or ""),
                    "strategy_digest": str(card.get("strategy_digest") or ""),
                    **_strategy_anchor_progress(branch["steps"], card),
                }
                for card in selected_cards
            ]
            branch["open_leaf_states"] = deque(
                dict(row)
                for row in result.get("open_leaf_states") or []
                if isinstance(row, Mapping)
            )
            branch["deferred_tail_leaf_states"] = deque()
            branch["expanded_products"] = {
                _canonical_smiles(row.get("product_smiles"))
                for row in branch["steps"]
                if _canonical_smiles(row.get("product_smiles"))
            }
            branch["complete_in_bound_stock"] = bool(result.get("solved"))
            branch["aizynthfinder_strategy_search"] = dict(
                result.get("diagnostics") or {}
            )
            search_diagnostics = dict(result.get("diagnostics") or {})
            provider_callback_count = int(
                search_diagnostics.get("provider_callback_count") or 0
            )
            sidecar_model_call_count = int(result.get("policy_calls") or 0)
            actual_policy_calls = len(route_records)
            branch["aizynthfinder_strategy_search"][
                "provider_callback_count"
            ] = provider_callback_count
            branch["aizynthfinder_strategy_search"][
                "actual_policy_calls"
            ] = actual_policy_calls
            branch["aizynthfinder_strategy_search"][
                "sidecar_model_call_count"
            ] = sidecar_model_call_count
            branch["aizynthfinder_strategy_search"][
                "model_call_ledger_matches"
            ] = sidecar_model_call_count == actual_policy_calls
            branch["aizynthfinder_strategy_search"]["reported_mcts_iterations"] = int(
                result.get("mcts_iterations") or 0
            )
            policy_calls = actual_policy_calls
            maximum_calls = int(config.max_node_expansions_per_branch)
            branch["paper_policy_call_budget"] = {
                "maximum_calls": maximum_calls,
                "actual_calls": policy_calls,
                "stock_closed": bool(result.get("solved")),
                "calls_exhausted": bool(
                    search_diagnostics.get("calls_exhausted")
                ),
                "host_stop_requested": bool(
                    search_diagnostics.get("host_stop_requested")
                ),
                "host_stop_reason": str(
                    search_diagnostics.get("host_stop_reason") or ""
                ),
                "within_cap": policy_calls <= maximum_calls,
                "stopped_before_cap": policy_calls < maximum_calls,
                "semantics": {
                    "host_mcts_or_stock_may_stop_before_call_ceiling": True,
                    "call_ceiling_is_not_a_required_minimum": True,
                    "builder_has_no_terminal_authority": True,
                    "stock_and_solved_are_host_owned": True,
                    "actual_calls_come_from_worker_records": True,
                    "provider_callbacks_are_not_model_calls": True,
                },
            }
            _sync_open_leaf_projection(branch)
            return local_records

        results: dict[int, list[WorkerRunRecord]] = {}
        maximum_workers = min(
            max(1, int(config.strategy_branch_workers)), branch_count
        )
        with ThreadPoolExecutor(
            max_workers=maximum_workers,
            thread_name_prefix="autoplanner-aizynth-strategy",
        ) as executor:
            futures = {
                executor.submit(
                    advance,
                    branch,
                    call_allowance=call_allocations[index],
                    input_allowance=input_allocations[index],
                    output_allowance=output_allocations[index],
                ): int(branch["branch_index"])
                for index, branch in enumerate(seeded)
                if call_allocations[index] > 0
                and input_allocations[index] > 0
                and output_allocations[index] > 0
                and not branch.get("paper_policy_budget_failure")
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [
            record
            for branch_index in sorted(results)
            for record in results[branch_index]
        ]

    def _expand_seeded_branches_parallel(
        self,
        spec: AgentSpec,
        *,
        target: str,
        seeded: list[dict[str, Any]],
        existing_records: list[WorkerRunRecord],
        route_quota: _NodeCallBudget,
        critic_editor_call_reserve: int,
        critic_input_reserve: int,
        critic_output_reserve: int,
        config: DirectorConfig,
        started: float,
    ) -> list[WorkerRunRecord]:
        """Advance frozen strategy states concurrently under isolated quotas.

        Strategy selection is already frozen before this method (one portfolio
        call in the paper profile, serial validated cards in compatibility
        profiles).  The route phase then partitions the remaining call/token allowance
        across seeded branches before any workers start.  This avoids a shared
        mutable budget race and bounds overshoot to the same final in-flight
        call already possible in the serial implementation.
        """

        usage = _aggregate_usage(existing_records, elapsed_s=0.0)
        branch_count = len(seeded)
        available_calls = max(
            0,
            int(route_quota.model_invocations)
            - int(usage["model_invocations"])
            - max(0, int(critic_editor_call_reserve)),
        )
        available_input = max(
            0,
            int(route_quota.input_tokens)
            - int(usage["input_tokens"])
            - max(0, int(critic_input_reserve)),
        )
        available_output = max(
            0,
            int(route_quota.output_tokens)
            - int(usage["output_tokens"])
            - max(0, int(critic_output_reserve)),
        )
        call_allocations = _balanced_branch_allocations(
            available_calls,
            branch_count,
            cap=max(1, int(config.max_node_expansions_per_branch)),
        )
        input_allocations = _balanced_branch_allocations(
            available_input,
            branch_count,
        )
        output_allocations = _balanced_branch_allocations(
            available_output,
            branch_count,
        )
        early_stop = threading.Event()

        def advance(
            branch: dict[str, Any],
            *,
            call_allowance: int,
            input_allowance: int,
            output_allowance: int,
        ) -> list[WorkerRunRecord]:
            local_records: list[WorkerRunRecord] = []
            local_quota = _NodeCallBudget(
                model_invocations=max(0, int(call_allowance)),
                input_tokens=max(0, int(input_allowance)),
                output_tokens=max(0, int(output_allowance)),
                # All workers share the same absolute deadline because
                # `_node_budget_allows` subtracts the common `started` time.
                wall_time_s=route_quota.wall_time_s,
            )
            while (
                not self._cancelled()
                and not (
                    config.stop_on_first_stock_closed_branch
                    and early_stop.is_set()
                )
                and int(branch["route_call_count"])
                < config.max_node_expansions_per_branch
                and _branch_has_expandable_leaf(branch)
                and bool(branch["strategy_card"])
                and _node_budget_allows(
                    local_records,
                    started=started,
                    quota=local_quota,
                )
            ):
                before_calls = int(branch["route_call_count"])
                self._expand_one_branch_node(
                    spec,
                    target=target,
                    branch=branch,
                    records=local_records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    max_reactionjson_candidates_per_node=(
                        config.max_reactionjson_candidates_per_node
                    ),
                    max_or_search_depth=config.max_node_expansions_per_branch,
                    require_strategy_graph_edits=config.require_strategy_graph_edits,
                    require_complete_route_json=config.require_complete_route_json,
                    materialization_editor_rounds=config.max_route_local_repair_rounds,
                    paper_matched=config.paper_matched_reach_profile,
                    quota=local_quota,
                    started=started,
                )
                if int(branch["route_call_count"]) <= before_calls:
                    break
                if _branch_stock_closed(branch):
                    branch["complete_in_bound_stock"] = True
                    if config.stop_on_first_stock_closed_branch:
                        branch["portfolio_early_stop_triggered"] = True
                        early_stop.set()
                    break
            return local_records

        results: dict[int, list[WorkerRunRecord]] = {}
        maximum_workers = min(
            max(1, int(config.strategy_branch_workers)), branch_count
        )
        with ThreadPoolExecutor(
            max_workers=maximum_workers,
            thread_name_prefix="autoplanner-strategy",
        ) as executor:
            futures = {
                executor.submit(
                    advance,
                    branch,
                    call_allowance=call_allocations[index],
                    input_allowance=input_allocations[index],
                    output_allowance=output_allocations[index],
                ): int(branch["branch_index"])
                for index, branch in enumerate(seeded)
                if call_allocations[index] > 0
                and input_allocations[index] > 0
                and output_allocations[index] > 0
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [
            record
            for branch_index in sorted(results)
            for record in results[branch_index]
        ]

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
            record = self._run_journaled_worker(self.node_executor, task)
            records.append(record)
            expansion = _expansion_from_record(
                record,
                expected_product=repair_product,
                mapped_product_smiles=_mapped_smiles(repair_product),
                require_reaction_operations=config.require_strategy_graph_edits,
                single_step_only=config.require_strategy_graph_edits,
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
                    "open_leaf_states": deque(
                        {
                            "smiles": value,
                            "mapped_smiles": mapped,
                        }
                        for value, mapped in _route_terminal_precursor_pairs((expansion,))
                    ),
                    "deferred_tail_leaf_states": deque(),
                    "target_mapped_smiles": _mapped_smiles(target),
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
        paper_matched: bool = False,
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
            paper_matched=paper_matched,
        )
        _assert_node_prompt_size(prompt, max_prompt_bytes)
        task = _strategy_task(
            spec,
            prompt=prompt,
            branch_index=branch_index,
            attempt_index=int(branch["strategy_call_count"]),
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(
                spec.metadata.get("strategy_reasoning_effort")
                or spec.metadata.get("reasoning_effort")
                or "medium"
            ),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
            paper_matched=paper_matched,
            target_smiles=target,
        )
        record = self._run_journaled_worker(self.node_executor, task)
        records.append(record)
        card = _strategy_card_from_record(
            record,
            expected_target=target,
            paper_matched=paper_matched,
        )
        forbidden = [dict(row) for row in forbidden_strategy_cards]
        if card is None:
            branch["rejections"].append(
                {
                    "phase": "strategy_generator",
                    "attempt": int(branch["strategy_call_count"]),
                    "reason": _strategy_card_rejection_reason(
                        record,
                        expected_target=target,
                        paper_matched=paper_matched,
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
        branch["root_strategy_card"] = dict(card)
        branch["strategy_milestone_cards"] = [dict(card)]
        branch["strategy_seed"] = _strategy_title_from_card(card)
        branch["lens"] = "Codex-authored strategy - " + str(
            branch["strategy_seed"]
        )

    def _seed_paper_strategy_portfolio(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branches: list[dict[str, Any]],
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        quota: _NodeCallBudget,
        started: float,
    ) -> None:
        """Generate the paper's three competing strategies in one model call."""

        prompt = _paper_strategy_portfolio_prompt(target=target)
        _assert_node_prompt_size(prompt, max_prompt_bytes)
        task = _strategy_portfolio_task(
            spec,
            prompt=prompt,
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(
                spec.metadata.get("strategy_reasoning_effort")
                or spec.metadata.get("reasoning_effort")
                or "medium"
            ),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
            target_smiles=target,
        )
        record = self._run_journaled_worker(self.node_executor, task)
        records.append(record)
        cards = _strategy_cards_from_portfolio_record(
            record,
            expected_target=target,
        )
        for branch in branches:
            branch["strategy_call_count"] = 1
            branch["call_count"] = int(branch.get("call_count") or 0) + 1
        if cards is None or len(cards) != len(branches):
            for branch in branches:
                branch["rejections"].append(
                    {
                        "phase": "strategy_generator",
                        "attempt": 1,
                        "reason": "strategy_portfolio_output_invalid",
                    }
                )
            return
        for branch, card in zip(branches, cards, strict=True):
            branch["strategy_card"] = card
            branch["root_strategy_card"] = dict(card)
            branch["strategy_milestone_cards"] = [dict(card)]
            branch["strategy_seed"] = _strategy_title_from_card(card)
            branch["lens"] = "Codex-authored strategy - " + str(
                branch["strategy_seed"]
            )

    def _review_paper_strategy_portfolio(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branches: list[dict[str, Any]],
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        quota: _NodeCallBudget,
        started: float,
    ) -> None:
        """Review the three V9 steering hypotheses once before paid search.

        The reviewer returns the same compact portfolio contract, so this is
        one refinement boundary rather than a second strategy authority.  An
        unavailable or invalid review leaves the generator portfolio intact.
        """

        original_cards = [
            dict(branch.get("strategy_card") or {}) for branch in branches
        ]
        if len(original_cards) != 3 or not all(original_cards):
            return
        prompt = _paper_strategy_portfolio_critic_prompt(
            target=target,
            strategy_cards=original_cards,
        )
        _assert_node_prompt_size(prompt, max_prompt_bytes)
        task = _strategy_portfolio_critic_task(
            spec,
            prompt=prompt,
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(
                spec.metadata.get("critic_reasoning_effort")
                or spec.metadata.get("strategy_reasoning_effort")
                or spec.metadata.get("reasoning_effort")
                or "medium"
            ),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
            target_smiles=target,
        )
        try:
            record = self._run_journaled_worker(self.critic_executor, task)
        except Exception as exc:
            record = WorkerRunRecord(
                run_id=f"{task.task_id}:run",
                task_id=task.task_id,
                case_id=task.case_id,
                status="worker_error",
                backend="critic_executor",
                stderr=f"{type(exc).__name__}: {exc}",
                output_validation={
                    "accepted": False,
                    "reasons": ["strategy_critic_execution_failed"],
                },
            )
        records.append(record)
        reviewed_cards = _strategy_cards_from_portfolio_record(
            record,
            expected_target=target,
        )
        applied = reviewed_cards is not None and len(reviewed_cards) == 3
        for index, branch in enumerate(branches):
            branch["strategy_critic_call_count"] = 1
            branch["strategy_critic"] = {
                "task_id": task.task_id,
                "status": "applied" if applied else "unavailable",
                "applied": applied,
                "reason": "" if applied else "strategy_critic_output_invalid",
            }
            if not applied:
                continue
            card = dict(reviewed_cards[index])
            branch["strategy_card"] = card
            branch["root_strategy_card"] = dict(card)
            branch["strategy_milestone_cards"] = [dict(card)]
            branch["strategy_seed"] = _strategy_title_from_card(card)
            branch["lens"] = "Critic-reviewed strategy - " + str(
                branch["strategy_seed"]
            )

    def _generate_upstream_strategy_milestone(
        self,
        spec: AgentSpec,
        *,
        campaign_target: str,
        selected_product: str,
        selected_product_mapped: str,
        branch: dict[str, Any],
        route_steps: Iterable[Mapping[str, Any]],
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        quota: _NodeCallBudget,
        started: float,
    ) -> dict[str, Any] | None:
        """Plan the next strategy against one exact upstream mapped leaf.

        This is intentionally receding-horizon.  Future precursor maps are not
        guessed at the target: a new StrategyCard is requested only after the
        previous anchor is present in the host-replayed prefix and AiZ has
        selected the next non-stock leaf.  The call is accounted separately
        from Route Builder policy calls.
        """

        branch_index = int(branch.get("branch_index") or 0)
        branch["strategy_call_count"] = int(
            branch.get("strategy_call_count") or 0
        ) + 1
        branch["strategy_milestone_generation_count"] = int(
            branch.get("strategy_milestone_generation_count") or 0
        ) + 1
        branch["call_count"] = int(branch.get("call_count") or 0) + 1
        milestone_index = int(branch["strategy_milestone_generation_count"]) + 1
        prior_cards = _ordered_strategy_cards_from_steps(
            root_strategy_card=dict(
                branch.get("root_strategy_card")
                or branch.get("strategy_card")
                or {}
            ),
            steps=route_steps,
        )
        prompt = _milestone_strategy_prompt(
            campaign_target=campaign_target,
            selected_product=selected_product,
            selected_product_mapped=selected_product_mapped,
            branch_index=branch_index,
            milestone_index=milestone_index,
            strategy_mandate=str(
                branch.get("strategy_mandate") or branch.get("lens") or ""
            ),
            completed_strategy_cards=prior_cards,
            route_steps=route_steps,
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
            target_smiles=selected_product,
        )
        record = self._run_journaled_worker(self.node_executor, task)
        records.append(record)
        card = _strategy_card_from_record(
            record,
            expected_target=selected_product,
            expected_mapped_target=selected_product_mapped,
        )
        attempt = {
            "milestone_index": milestone_index,
            "selected_product_smiles": selected_product,
            "selected_product_mapped_smiles": selected_product_mapped,
            "task_id": task.task_id,
            "accepted": card is not None,
        }
        if card is None:
            attempt["reason"] = _strategy_card_rejection_reason(
                record,
                expected_target=selected_product,
                expected_mapped_target=selected_product_mapped,
            )
            branch.setdefault("strategy_milestone_attempts", []).append(attempt)
            return None
        attempt["strategy_id"] = str(card.get("strategy_id") or "")
        attempt["strategy_digest"] = str(card.get("strategy_digest") or "")
        branch.setdefault("strategy_milestone_attempts", []).append(attempt)
        branch.setdefault("strategy_milestone_cards", []).append(dict(card))
        return card

    def _expand_one_branch_node(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branch: dict[str, Any],
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        max_reactionjson_candidates_per_node: int = 3,
        max_or_search_depth: int = 25,
        require_strategy_graph_edits: bool = False,
        require_complete_route_json: bool = False,
        materialization_editor_rounds: int = _MATERIALIZATION_RETRY_LIMIT,
        paper_matched: bool = False,
        quota: _NodeCallBudget,
        started: float,
    ) -> None:
        branch_index = int(branch["branch_index"])
        lens = str(branch["lens"])
        steps: list[dict[str, Any]] = branch["steps"]
        expanded_products: set[str] = branch["expanded_products"]
        rejected: list[dict[str, Any]] = branch["rejections"]
        # Compiler-first node expansion is backed by ChemEnzy's native
        # molecule-OR/reaction-AND tree.  Legacy declared-precursor and
        # complete-route modes retain the historical linear queue.
        compiler_first = bool(
            require_strategy_graph_edits and not require_complete_route_json
        )
        or_search: ChemEnzyReactionJsonOrSearch | None = None
        selected_or_node: Any | None = None
        if compiler_first:
            existing_search = branch.get("_reactionjson_or_search")
            if isinstance(existing_search, ChemEnzyReactionJsonOrSearch):
                or_search = existing_search
            else:
                or_search = ChemEnzyReactionJsonOrSearch(
                    target_smiles=target,
                    mapped_target_smiles=str(branch.get("target_mapped_smiles") or ""),
                    max_depth=max(1, int(max_or_search_depth)),
                )
                branch["_reactionjson_or_search"] = or_search
            selected_or_node = or_search.select_open_node()
            selected_state = (
                or_search.node_state(selected_or_node)
                if selected_or_node is not None
                else None
            )
            prompt_steps = list(
                or_search.context_steps_for_node(selected_or_node)
                if selected_or_node is not None
                else ()
            )
        else:
            selected_state = _pop_open_leaf_state(branch)
            while selected_state is not None and selected_state[0] in expanded_products:
                selected_state = _pop_open_leaf_state(branch)
            prompt_steps = steps
        if selected_state is None:
            return
        selected, selected_mapped = selected_state
        if not selected or not selected_mapped:
            return
        open_leaves = list(branch.get("open_leaves") or [])
        branch["route_call_count"] = int(branch["route_call_count"]) + 1
        branch["call_count"] = int(branch["call_count"]) + 1
        call_index = int(branch["route_call_count"])
        is_strategy_anchor = (
            selected_or_node is getattr(or_search, "tree", None).root
            if or_search is not None
            else not steps and selected == target
        )
        # The matched path is compiler-first: one selected open leaf and one
        # ReactionJSON edit per call.  ``require_complete_route_json`` remains
        # only as an explicit legacy compatibility mode.
        prompt_complete_route = bool(require_complete_route_json)
        prompt = _node_prompt(
            target=target,
            branch_index=branch_index,
            lens=lens,
            selected_product=selected,
            selected_product_mapped=selected_mapped,
            steps=prompt_steps,
            open_leaves=[selected, *open_leaves],
            prior_rejections=rejected,
            repair=False,
            strategy_card=dict(branch.get("strategy_card") or {}),
            forbidden_strategy_cards=(),
            host_failure_feedback={},
            complete_route_json=prompt_complete_route,
            minimum_route_depth=1,
            max_reactionjson_candidates=(
                max_reactionjson_candidates_per_node if compiler_first else 1
            ),
        )
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            prompt = _node_prompt(
                target=target,
                branch_index=branch_index,
                lens=lens,
                selected_product=selected,
                selected_product_mapped=selected_mapped,
                steps=prompt_steps[-6:],
                open_leaves=[selected, *list(open_leaves)[:12]],
                prior_rejections=rejected[-4:],
                repair=False,
                strategy_card=dict(branch.get("strategy_card") or {}),
                forbidden_strategy_cards=(),
                host_failure_feedback={},
                complete_route_json=prompt_complete_route,
                minimum_route_depth=1,
                max_reactionjson_candidates=(
                    max_reactionjson_candidates_per_node if compiler_first else 1
                ),
            )
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            prompt = _node_prompt(
                target=target,
                branch_index=branch_index,
                lens=lens,
                selected_product=selected,
                selected_product_mapped=selected_mapped,
                steps=(),
                open_leaves=[selected],
                prior_rejections=(),
                repair=False,
                strategy_card=dict(branch.get("strategy_card") or {}),
                forbidden_strategy_cards=(),
                host_failure_feedback={},
                complete_route_json=prompt_complete_route,
                minimum_route_depth=1,
                max_reactionjson_candidates=(
                    max_reactionjson_candidates_per_node if compiler_first else 1
                ),
            )
        _assert_node_prompt_size(prompt, max_prompt_bytes)
        task = _node_task(
            spec,
            prompt=prompt,
            branch_index=branch_index,
            # route_call_count/call_index is one-based; _node_task owns the
            # conversion to the one-based id shown in logs and model I/O.
            node_index=call_index - 1,
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(spec.metadata.get("reasoning_effort") or "medium"),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
        )
        record = self._run_journaled_worker(self.node_executor, task)
        records.append(record)
        if compiler_first and or_search is not None and selected_or_node is not None:
            self._consume_reactionjson_or_record(
                record,
                branch=branch,
                search=or_search,
                selected_node=selected_or_node,
                selected_product=selected,
                selected_product_mapped=selected_mapped,
                branch_index=branch_index,
                call_index=call_index,
                is_strategy_anchor=is_strategy_anchor,
                max_candidates=max_reactionjson_candidates_per_node,
            )
            return
        expansions = _expansions_from_record(
            record,
            expected_product=selected,
            mapped_product_smiles=selected_mapped,
            compiler=self.routejson_compiler,
            require_reaction_operations=bool(require_strategy_graph_edits),
            require_complete_route_json=prompt_complete_route,
            minimum_route_depth=1,
            single_step_only=compiler_first,
        )
        diagnostic: dict[str, Any] | None = None
        materialization_editor_attempted = False
        if expansions is None:
            diagnostic = _expansion_rejection_diagnostic(
                record,
                expected_product=selected,
                mapped_product_smiles=selected_mapped,
                require_reaction_operations=bool(require_strategy_graph_edits),
                require_complete_route_json=prompt_complete_route,
                minimum_route_depth=1,
                single_step_only=compiler_first,
            )
            # Legacy whole-route mode may still provide a repairable draft. Do
            # not throw it away and ask the Route Builder to redraw the same
            # route.  Feed the host's exact compiler diagnostic to the Codex
            # Editor, which can repair operations, ordering, and conditions
            # while preserving the frozen StrategyCard.
            if (
                prompt_complete_route
                and _route_json_has_repairable_draft(record)
                and _route_json_failure_is_editor_repairable(diagnostic)
            ):
                materialization_editor_attempted = True
                expansions = self._repair_unmaterialized_route(
                    spec,
                    target=target,
                    branch=branch,
                    selected_product=selected,
                    selected_product_mapped=selected_mapped,
                    record=record,
                    diagnostic=diagnostic,
                    records=records,
                    max_prompt_bytes=max_prompt_bytes,
                    max_node_call_timeout_s=max_node_call_timeout_s,
                    max_rounds=max(1, int(materialization_editor_rounds)),
                    paper_matched=paper_matched,
                    quota=quota,
                    started=started,
                )
                if expansions is not None:
                    diagnostic = None
                else:
                    diagnostic = dict(
                        branch.pop("_last_materialization_editor_failure", {})
                        or diagnostic
                    )
        if expansions is None or any(
            precursor in expanded_products
            for expansion in expansions
            for precursor in expansion.precursor_smiles
        ):
            diagnostic = diagnostic or {"reason": "ancestor_cycle"}
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
            branch.setdefault("materialization_diagnostics", []).append(
                {
                    "phase": "route_builder",
                    "node": call_index,
                    "product_smiles": selected,
                    **dict(diagnostic),
                }
            )
            if rejection_reason in {
                "route_json_missing",
                "route_json_invalid",
                "route_json_incomplete",
                "route_json_step_invalid",
                "route_json_chain_invalid",
                "route_json_step_reaction_operations_missing",
                "route_json_step_replay_failed",
                "strategy_graph_edit_missing",
                "strategy_graph_edit_replay_failed",
                "invalid_expansion_contract",
            }:
                failures = dict(branch.get("materialization_failures") or {})
                graph_edit_rejections = int(failures.get(selected) or 0) + 1
                failures[selected] = graph_edit_rejections
                branch["materialization_failures"] = failures
                retry_limit = (
                    1
                    if materialization_editor_attempted
                    else _MATERIALIZATION_RETRY_LIMIT
                )
                if graph_edit_rejections >= retry_limit:
                    rejected.append(
                        {
                            "phase": "route_builder",
                            "node": call_index,
                            "product_smiles": selected,
                            "reason": "materialization_retry_limit_reached",
                            "strategy_retained": True,
                        }
                    )
                    blocked = branch.setdefault("blocked_materializations", [])
                    if selected not in blocked:
                        blocked.append(selected)
                    deferred = branch.setdefault(
                        "deferred_tail_leaf_states", deque()
                    )
                    if not any(
                        _canonical_smiles(row.get("smiles")) == selected
                        for row in deferred
                        if isinstance(row, Mapping)
                    ):
                        deferred.append(
                            {
                                "smiles": selected,
                                "mapped_smiles": selected_mapped,
                            }
                        )
                    _sync_open_leaf_projection(branch)
                    return
            branch.setdefault("open_leaf_states", deque()).appendleft(
                {"smiles": selected, "mapped_smiles": selected_mapped}
            )
            _sync_open_leaf_projection(branch)
            return
        for route_index, expansion in enumerate(expansions):
            if branch.get("strategy_card"):
                expansion = replace(
                    expansion,
                    strategy_card=dict(branch.get("strategy_card") or {}),
                )
            expanded_products.add(expansion.product_smiles)
            steps.append(
                _step_row(
                    expansion,
                    step_id=(
                        expansion.step_id
                        or f"codex:branch:{branch_index + 1}:{len(steps) + 1}"
                    ),
                        strategy_anchor=_expansion_executes_strategy_anchor(
                            expansion,
                            dict(branch.get("strategy_card") or {}),
                            fallback=is_strategy_anchor and route_index == 0,
                        ),
                )
            )
        terminal_precursor_pairs = _route_terminal_precursor_pairs(expansions)
        terminal_precursors = tuple(pair[0] for pair in terminal_precursor_pairs)
        membership = self._stock_membership(terminal_precursors)
        new_states = [
            {"smiles": precursor, "mapped_smiles": mapped}
            for precursor, mapped in terminal_precursor_pairs
            if membership.get(precursor) is not True and precursor not in expanded_products
        ]
        existing_states = list(branch.get("open_leaf_states") or [])
        branch["open_leaf_states"] = deque(existing_states + new_states)
        _sync_open_leaf_projection(branch)

    def _consume_reactionjson_or_record(
        self,
        record: WorkerRunRecord,
        *,
        branch: dict[str, Any],
        search: ChemEnzyReactionJsonOrSearch,
        selected_node: Any,
        selected_product: str,
        selected_product_mapped: str,
        branch_index: int,
        call_index: int,
        is_strategy_anchor: bool,
        max_candidates: int,
    ) -> None:
        compiled, candidate_rejections = _reactionjson_candidates_from_record(
            record,
            expected_product=selected_product,
            mapped_product_smiles=selected_product_mapped,
            require_reaction_operations=True,
            compiler=self.routejson_compiler,
            max_candidates=max_candidates,
        )
        branch.setdefault("reactionjson_candidate_batches", []).append(
            {
                "node": call_index,
                "product_smiles": selected_product,
                "reported_candidates": len(compiled) + len(candidate_rejections),
                "compiled_candidates": len(compiled),
                "rejected_candidates": len(candidate_rejections),
            }
        )
        for diagnostic in candidate_rejections:
            row = {
                "phase": "route_builder_candidate",
                "node": call_index,
                "product_smiles": selected_product,
                **dict(diagnostic),
            }
            branch.setdefault("rejections", []).append(row)
            branch.setdefault("materialization_diagnostics", []).append(row)

        or_candidates: list[ReactionJsonOrCandidate] = []
        strategy_card = dict(branch.get("strategy_card") or {})
        for item in compiled:
            expansion = replace(item.expansion, strategy_card=strategy_card)
            step = _step_row(
                expansion,
                step_id=(
                    expansion.step_id
                    or (
                        f"codex:branch:{branch_index + 1}:node:{call_index}:"
                        f"candidate:{item.candidate_index + 1}"
                    )
                ),
                strategy_anchor=_expansion_executes_strategy_anchor(
                    expansion,
                    strategy_card,
                    fallback=is_strategy_anchor,
                ),
            )
            or_candidates.append(
                ReactionJsonOrCandidate(
                    candidate_id=item.candidate_id,
                    precursor_smiles=tuple(expansion.precursor_smiles),
                    mapped_precursor_smiles=tuple(
                        expansion.mapped_precursor_smiles
                    ),
                    route_step=step,
                    score=item.score,
                    cost=item.cost,
                    candidate_key=item.candidate_key,
                )
            )

        terminal_precursors = tuple(
            dict.fromkeys(
                precursor
                for candidate in or_candidates
                for precursor in candidate.precursor_smiles
            )
        )
        membership = self._stock_membership(terminal_precursors)
        inserted = search.expand(
            selected_node,
            or_candidates,
            stock_smiles=(
                precursor
                for precursor in terminal_precursors
                if membership.get(precursor) is True
            ),
        )
        if inserted:
            failures = dict(branch.get("materialization_failures") or {})
            failures.pop(selected_product, None)
            branch["materialization_failures"] = failures
            _refresh_branch_from_reactionjson_or_search(branch, search)
            return

        diagnostic = {
            "reason": (
                "all_reactionjson_candidates_rejected"
                if not compiled
                else "all_reactionjson_candidates_duplicate_or_cyclic"
            ),
            "candidate_diagnostics": [dict(value) for value in candidate_rejections],
        }
        branch.setdefault("rejections", []).append(
            {
                "phase": "route_builder",
                "node": call_index,
                "product_smiles": selected_product,
                **diagnostic,
            }
        )
        failures = dict(branch.get("materialization_failures") or {})
        failure_count = int(failures.get(selected_product) or 0) + 1
        failures[selected_product] = failure_count
        branch["materialization_failures"] = failures
        if failure_count >= _MATERIALIZATION_RETRY_LIMIT:
            search.block_for_short_tail(selected_node)
            branch.setdefault("rejections", []).append(
                {
                    "phase": "route_builder",
                    "node": call_index,
                    "product_smiles": selected_product,
                    "reason": "materialization_retry_limit_reached",
                    "strategy_retained": True,
                    "or_backtrack_enabled": True,
                }
            )
        _refresh_branch_from_reactionjson_or_search(branch, search)

    def _repair_unmaterialized_route(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branch: dict[str, Any],
        selected_product: str,
        selected_product_mapped: str,
        record: WorkerRunRecord,
        diagnostic: Mapping[str, Any],
        records: list[WorkerRunRecord],
        max_prompt_bytes: int,
        max_node_call_timeout_s: float,
        max_rounds: int,
        paper_matched: bool,
        quota: _NodeCallBudget,
        started: float,
    ) -> list[NodeExpansion] | None:
        """Repair an invalid complete RouteJSON draft before branch blocking.

        This compatibility path is used only when a caller explicitly asks one
        model response to contain a complete route.  The Route Builder owns
        strategic intent; the host owns structure.  When replay fails, an
        independent Codex Editor receives the full route and the causal
        failure.  Only a host-recompiled patch or replacement is promoted.
        """

        payload = dict(dict(record.output_artifact or {}).get("payload") or {})
        candidates = [
            dict(value)
            for value in payload.get("candidates") or []
            if isinstance(value, Mapping)
        ]
        raw_route = candidates[0].get("route_json") if len(candidates) == 1 else None
        if not isinstance(raw_route, list) or not raw_route:
            return None
        current_route = [dict(value) for value in raw_route if isinstance(value, Mapping)]
        if not current_route:
            return None

        branch.setdefault("materialization_editor_history", [])
        editor_feedback: dict[str, Any] = {
            "route_builder_materialization_failure": dict(diagnostic),
            "editor_instruction": (
                "Repair the complete RouteJSON against the host diagnostic. "
                "Every later product must be an exact host-replayed precursor "
                "with preserved atom maps; do not redraw fragments or add a "
                "terminal pseudo-step."
            ),
        }
        for attempt in range(1, max_rounds + 1):
            if not _node_budget_allows(records, started=started, quota=quota):
                failure = {
                    "reason": "materialization_editor_budget_exhausted",
                    "attempt": attempt,
                }
                branch["_last_materialization_editor_failure"] = failure
                branch["materialization_editor_history"].append(failure)
                return None
            # Replace only model-redrawn downstream product identities with
            # the exact fragments emitted by the host replay.  This is an
            # Editor scaffold, not an accepted route: operations, order, and
            # the immutable strategy remain model-controlled and are still
            # recompiled below.
            current_route = _editor_route_scaffold(
                current_route,
                mapped_target_smiles=selected_product_mapped,
            )
            editor_feedback["host_replayed_route_scaffold"] = _compact_route_rows(
                current_route
            )
            prompt = _node_prompt(
                target=target,
                branch_index=int(branch.get("branch_index") or 0),
                lens="Codex Editor: repair compiler-rejected complete RouteJSON",
                selected_product=selected_product,
                selected_product_mapped=selected_product_mapped,
                steps=_compact_route_rows(current_route),
                open_leaves=[selected_product],
                prior_rejections=[
                    {
                        "phase": "route_builder_materialization",
                        "attempt": attempt,
                        "reason": str(
                            editor_feedback.get(
                                "route_builder_materialization_failure", {}
                            ).get("reason")
                            or "route_json_replay_failed"
                        ),
                    }
                ],
                repair=True,
                strategy_card=dict(branch.get("strategy_card") or {}),
                forbidden_strategy_cards=(),
                host_failure_feedback=editor_feedback,
                complete_route_json=True,
                editor_route_mutations=True,
                minimum_route_depth=1,
                paper_matched=paper_matched,
            )
            _assert_node_prompt_size(prompt, max_prompt_bytes)
            task = _node_task(
                spec,
                prompt=prompt,
                branch_index=int(branch.get("branch_index") or 0),
                node_index=attempt - 1,
                model=str(spec.metadata.get("model") or ""),
                reasoning_effort=str(
                    spec.metadata.get("editor_reasoning_effort")
                    or spec.metadata.get("reasoning_effort")
                    or "medium"
                ),
                timeout_s=_node_call_timeout_s(
                    started,
                    quota,
                    maximum=max_node_call_timeout_s,
                ),
                task_type="route_chemistry_edit",
                paper_matched=paper_matched,
                target_smiles=target,
                selected_product=selected_product,
            )
            try:
                editor_record = self._run_journaled_worker(
                    self.editor_executor,
                    task,
                )
            except Exception as exc:
                failure = {
                    "reason": "materialization_editor_execution_failed",
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                records.append(
                    WorkerRunRecord(
                        run_id=f"{task.task_id}:run",
                        task_id=task.task_id,
                        case_id=task.case_id,
                        status="worker_error",
                        backend="editor_executor",
                        stderr=f"{type(exc).__name__}: {exc}",
                        output_validation={
                            "accepted": False,
                            "reasons": ["materialization_editor_execution_failed"],
                        },
                    )
                )
                branch["_last_materialization_editor_failure"] = failure
                branch["materialization_editor_history"].append(failure)
                return None
            records.append(editor_record)
            route_expansions, editor_diagnostic, mutation_mode = (
                _editor_route_expansions_from_record(
                    editor_record,
                    current_steps=current_route,
                    mapped_target_smiles=selected_product_mapped,
                    expected_target_smiles=selected_product,
                )
            )
            history = {
                "phase": "route_builder_materialization",
                "attempt": attempt,
                "editor_task_id": task.task_id,
                "mutation_mode": mutation_mode,
                "input_diagnostic": dict(
                    editor_feedback.get("route_builder_materialization_failure")
                    or {}
                ),
            }
            if route_expansions:
                history["outcome"] = "host_recompiled"
                branch["materialization_editor_history"].append(history)
                branch.setdefault("editor_repairs", []).append(
                    {
                        "phase": "route_builder_materialization",
                        "attempt": attempt,
                        "editor_task_id": task.task_id,
                        "mutation_mode": mutation_mode,
                        "old_route_depth": len(current_route),
                        "new_route_depth": len(route_expansions),
                    }
                )
                return route_expansions
            history["outcome"] = "rejected"
            history["diagnostic"] = dict(editor_diagnostic)
            branch["materialization_editor_history"].append(history)
            branch["_last_materialization_editor_failure"] = dict(editor_diagnostic)
            editor_candidate = _route_json_candidate(editor_record)
            if editor_candidate is not None:
                replacement = editor_candidate.get("route_json")
                if isinstance(replacement, list) and replacement:
                    current_route = [
                        dict(value)
                        for value in replacement
                        if isinstance(value, Mapping)
                    ]
                else:
                    patch_rows = editor_candidate.get("route_patch")
                    if isinstance(patch_rows, list) and patch_rows:
                        patched, _ = _apply_route_patch(current_route, patch_rows)
                        if patched is not None:
                            current_route = patched
            editor_feedback = {
                **editor_feedback,
                "editor_materialization_failure": dict(editor_diagnostic),
            }
        return None

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

    def _execute_node(self, task: WorkerTask) -> WorkerRunRecord:
        return run_codex_worker(
            task,
            use_codex_cli=True,
            cancel_event=self.cancel_event,
        )


def _paper_critic_editor_reserve_after_current_critic(
    branches: Sequence[Mapping[str, Any]],
    *,
    current_index: int,
    iteration: int,
    max_rounds: int,
) -> tuple[int, int]:
    """Return executable future Critic and Editor calls after this Critic."""

    rounds = max(0, int(max_rounds))
    current = dict(branches[current_index])
    current_editors = max(
        0,
        rounds - int(current.get("editor_attempt_count") or 0),
    )
    current_critics = min(
        max(0, rounds - int(iteration)),
        current_editors,
    )
    future_critics = 0
    future_editors = 0
    for value in branches[current_index + 1 :]:
        branch = dict(value)
        if not branch.get("steps") or dict(
            branch.get("chemical_critic") or {}
        ).get("status"):
            continue
        # Preserve one mandatory initial Critic for every untouched future
        # route. Its optional Editor loop is budget-checked when that route
        # becomes current; reserving every hypothetical repair here can block
        # the current route's required post-Editor Critic.
        future_critics += 1
    return (
        current_critics + future_critics,
        current_editors + future_editors,
    )


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
        + (
            config.strategy_branch_count
            * _MAX_KEY_EVENT_CRITIC_CALLS_PER_BRANCH
            if config.enable_key_event_critic
            else 0
        )
        + (1 if config.enable_strategy_portfolio_critic else 0)
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


def _balanced_branch_allocations(
    total: int,
    branch_count: int,
    *,
    cap: int | None = None,
) -> tuple[int, ...]:
    """Partition one non-negative allowance without shared-worker races."""

    count = max(0, int(branch_count))
    if count == 0:
        return ()
    available = max(0, int(total))
    if cap is not None:
        available = min(available, count * max(0, int(cap)))
    quotient, remainder = divmod(available, count)
    allocations = [quotient + (1 if index < remainder else 0) for index in range(count)]
    if cap is not None:
        maximum = max(0, int(cap))
        allocations = [min(value, maximum) for value in allocations]
    return tuple(allocations)


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
    return not _node_budget_block_reason(
        records,
        started=started,
        quota=quota,
        reserve_model_invocations=reserve_model_invocations,
        reserve_input_tokens=reserve_input_tokens,
        reserve_output_tokens=reserve_output_tokens,
        reserve_wall_time_s=reserve_wall_time_s,
    )


def _node_budget_block_reason(
    records: Iterable[WorkerRunRecord],
    *,
    started: float,
    quota: _NodeCallBudget,
    reserve_model_invocations: int = 0,
    reserve_input_tokens: int = 0,
    reserve_output_tokens: int = 0,
    reserve_wall_time_s: float = 0.0,
) -> str:
    """Return the single owning resource boundary blocking another call."""

    rows = list(records)
    if len(rows) + max(0, int(reserve_model_invocations)) >= quota.model_invocations:
        return "model_invocation_allocation_exhausted"
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
    if input_tokens + max(0, int(reserve_input_tokens)) >= quota.input_tokens:
        return "input_token_allocation_exhausted"
    if output_tokens + max(0, int(reserve_output_tokens)) >= quota.output_tokens:
        return "output_token_allocation_exhausted"
    if _remaining_node_wall_time(started, quota) <= (
        max(0.0, float(reserve_wall_time_s))
        + _deadline_settlement_reserve_s(quota)
    ):
        return "wall_time_allocation_exhausted"
    return ""


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
        "strategy_tree_engine": str(
            branch.get("strategy_tree_engine") or "chemenzy_best_first"
        ),
        "strategy_card": dict(branch.get("strategy_card") or {}),
        "root_strategy_card": dict(
            branch.get("root_strategy_card") or branch.get("strategy_card") or {}
        ),
        "strategy_milestone_cards": [
            dict(row)
            for row in branch.get("strategy_milestone_cards") or []
            if isinstance(row, Mapping)
        ],
        "strategy_milestone_attempts": [
            dict(row)
            for row in branch.get("strategy_milestone_attempts") or []
            if isinstance(row, Mapping)
        ],
        "strategic_milestone_count": int(
            branch.get("strategic_milestone_count") or 0
        ),
        "steps": [dict(row) for row in branch.get("steps") or []],
        "open_leaves": list(branch.get("open_leaves") or []),
        "open_leaf_states": [
            dict(row)
            for row in branch.get("open_leaf_states") or []
            if isinstance(row, Mapping)
        ],
        "deferred_tail_leaf_states": [
            dict(row)
            for row in branch.get("deferred_tail_leaf_states") or []
            if isinstance(row, Mapping)
        ],
        "call_count": int(branch.get("call_count") or 0),
        "strategy_call_count": int(branch.get("strategy_call_count") or 0),
        "route_call_count": int(branch.get("route_call_count") or 0),
        "editor_attempt_count": int(branch.get("editor_attempt_count") or 0),
        "editor_call_count": int(branch.get("editor_call_count") or 0),
        "editor_applied_count": int(branch.get("editor_call_count") or 0),
        "complete_in_bound_stock": bool(branch.get("complete_in_bound_stock")),
        "reactionjson_or_search": dict(
            branch.get("reactionjson_or_search") or {}
        ),
        "aizynthfinder_strategy_search": dict(
            branch.get("aizynthfinder_strategy_search") or {}
        ),
        "portfolio_early_stop_triggered": bool(
            branch.get("portfolio_early_stop_triggered")
        ),
        "rejections": [dict(row) for row in branch.get("rejections") or []],
        "blocked_materializations": list(
            branch.get("blocked_materializations") or []
        ),
        "materialization_failures": dict(
            branch.get("materialization_failures") or {}
        ),
        "materialization_diagnostics": [
            dict(row)
            for row in branch.get("materialization_diagnostics") or []
            if isinstance(row, Mapping)
        ],
        "materialization_editor_history": [
            dict(row)
            for row in branch.get("materialization_editor_history") or []
            if isinstance(row, Mapping)
        ],
        "reactionjson_or_search_resets": [
            dict(row)
            for row in branch.get("reactionjson_or_search_resets") or []
            if isinstance(row, Mapping)
        ],
        "reactionjson_candidate_batches": [
            dict(row)
            for row in branch.get("reactionjson_candidate_batches") or []
            if isinstance(row, Mapping)
        ],
        "editor_repairs": [
            dict(row)
            for row in branch.get("editor_repairs") or []
            if isinstance(row, Mapping)
        ],
        "editor_execution_notes": [
            dict(row)
            for row in branch.get("editor_execution_notes") or []
            if isinstance(row, Mapping)
        ],
        "editor_working_route": dict(
            branch.get("editor_working_route") or {}
        ),
        "editor_rejection_diagnostics": [
            dict(row)
            for row in branch.get("editor_rejection_diagnostics") or []
            if isinstance(row, Mapping)
        ],
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
    if str(card.get("strategy_basis") or "") == (
        "paper-matched one-sentence steering query"
    ):
        return bool(
            str(card.get("strategy_query") or "").strip()
            and str(card.get("critic_checkpoint") or "").strip()
            and (
                str(card.get("critical_assumption") or "").strip()
                or str(card.get("strategy_signature") or "").strip()
            )
        )
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
    if not any(
        isinstance(card.get(field), list) and card.get(field)
        for field in (
            "key_bond_changes",
            "anchor_bond_changes",
            "precursor_only_bond_changes",
        )
    ):
        return False
    if not isinstance(card.get("functional_group_conflicts"), list):
        return False
    try:
        step_count = int(card.get("strategic_step_count"))
    except (TypeError, ValueError):
        return False
    basic_valid = step_count in {1, 2} and str(card.get("expected_complexity_drop")) in {
        "low",
        "medium",
        "high",
    }
    if not basic_valid:
        return False
    execution_domain = str(card.get("execution_domain") or "chemical")
    if execution_domain not in BIOLOGICAL_EXECUTION_DOMAINS:
        return True
    intent = card.get("biocatalytic_intent")
    return bool(
        isinstance(intent, Mapping)
        and intent.get("design_complete") is True
        and not card.get("biocatalytic_intent_reasons")
    )


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
    card: Mapping[str, Any],
    *,
    target_smiles: str,
    mapped_target_smiles: str = "",
) -> bool:
    """Validate route anchors without rejecting precursor-only reorganization."""

    if str(card.get("strategy_basis") or "") == (
        "paper-matched one-sentence steering query"
    ):
        return bool(
            str(card.get("strategy_query") or "").strip()
            and str(card.get("critic_checkpoint") or "").strip()
            and (
                str(card.get("critical_assumption") or "").strip()
                or str(card.get("strategy_signature") or "").strip()
            )
        )

    target_pairs = (
        set(_mapped_bond_pairs(mapped_target_smiles))
        if mapped_target_smiles
        else _target_bond_pairs(target_smiles)
    )
    if not target_pairs:
        return False
    anchor_values = card.get("anchor_bond_changes") or []
    if not anchor_values:
        # v1 cards used key_bond_changes as the route anchor.
        anchor_values = card.get("key_bond_changes") or []
    anchor_pairs = [_parse_strategy_bond_pair(value) for value in anchor_values]
    anchor_pairs = [pair for pair in anchor_pairs if pair is not None]
    if anchor_pairs and not all(pair in target_pairs for pair in anchor_pairs):
        return False

    precursor_values = card.get("precursor_only_bond_changes") or []
    precursor_pairs = [
        _parse_strategy_bond_pair(value) for value in precursor_values
    ]
    precursor_pairs = [pair for pair in precursor_pairs if pair is not None]
    mapped_atoms = {atom for pair in target_pairs for atom in pair}
    if precursor_pairs and not all(
        left in mapped_atoms and right in mapped_atoms
        for left, right in precursor_pairs
    ):
        return False
    return bool(anchor_pairs or precursor_pairs or card.get("bond_order_changes"))


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
    normalized_candidate = normalize_strategy_card(candidate)
    for prior in forbidden:
        if strategy_cards_conflict(candidate, prior):
            return True
        normalized_prior = normalize_strategy_card(prior)
        # StrategyCards are hypotheses, so they usually have no authoritative
        # ReactionJSON edit yet.  Treat the same target-bond anchor plus the
        # same broad mechanism family as a duplicate even when the model
        # paraphrases the topology prose (the observed failure mode was two
        # cationic cascades over the same three target bonds).
        if (
            normalized_candidate.get("key_bond_signature")
            and normalized_candidate.get("key_bond_signature")
            == normalized_prior.get("key_bond_signature")
            and _strategy_bucket(normalized_candidate)
            == _strategy_bucket(normalized_prior)
        ):
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


def _target_topology_profile(smiles: str) -> dict[str, Any]:
    """Return only deterministic scaffold facts useful to Strategy workers."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {}
    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
    pair_counts: Counter[tuple[tuple[int, int], int]] = Counter()
    for left_index, left in enumerate(rings):
        for right in rings[left_index + 1 :]:
            ring_sizes = tuple(sorted((len(left), len(right))))
            pair_counts[(ring_sizes, len(left & right))] += 1
    return {
        "ring_sizes": sorted(len(ring) for ring in rings),
        "ring_pair_topology": [
            {
                "ring_sizes": list(ring_sizes),
                "shared_atom_count": shared_atom_count,
                "pair_count": pair_count,
            }
            for (ring_sizes, shared_atom_count), pair_count in sorted(
                pair_counts.items()
            )
        ],
    }


def _mapped_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for index, atom in enumerate(molecule.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _route_atom_map_namespace(
    steps: Iterable[Mapping[str, Any]],
    *extra_mapped_smiles: str,
) -> set[int]:
    """Collect every atom provenance identity already used on this route."""

    values = [str(value) for value in extra_mapped_smiles if str(value)]
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        values.append(str(step.get("mapped_product_smiles") or ""))
        values.extend(
            str(value) for value in step.get("mapped_precursor_smiles") or []
        )
    namespace: set[int] = set()
    for value in values:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        namespace.update(
            int(atom.GetAtomMapNum())
            for atom in molecule.GetAtoms()
            if int(atom.GetAtomMapNum()) > 0
        )
    return namespace


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


def _paper_strategy_portfolio_prompt(*, target: str) -> str:
    context = {
        "schema_version": "paper_matched_strategy_portfolio_input.v1",
        "phase": "strategy_generator",
        "campaign_target": target,
        "target_topology_profile": _target_topology_profile(target),
        "strategy_count": 3,
    }
    return "\n".join(
        [
            "Act as the paper-matched Strategy Generator and create exactly three independent high-level strategies in this single call.",
            "Internally generate more than three possibilities, compare and attack their weakest chemical assumptions across the paper's four dimensions: scaffold/backbone, one or two key forward reactions, functional-group/protection compatibility, and stereochemical construction or control. Return only the three survivors; do not expose the internal debate.",
            "For each card, output one strategy_query sentence, one critical_assumption sentence, and one critic_checkpoint sentence. strategy_query identifies the high-level construction, the reactive-handle motif that enables it, and the main stereochemical or functional-group control. critical_assumption names the make-or-break chemical claim. critic_checkpoint is the earliest non-substitutable graph transformation that directly tests that assumption; a downstream event that could succeed while the assumption remains false, or a preparatory handle installation/unmasking, is not a valid checkpoint.",
            "The three strategy_query values must differ materially in skeletal construction or reorganization and key transformation logic, not merely in reagents or labels.",
            "Routine FGI is strategic only when it directly enables the key construction. Do not output atom-map pairs, precursor structures, conditions, rationales, limitations, tables, or mechanistic essays.",
            "Return only the compact StrategyPortfolioReport. Do not build routes, write ReactionJSON, browse, inspect stock, or add evidence or enzyme fields.",
            "PaperMatchedStrategyPortfolioInput:",
            json.dumps(
                context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )


def _paper_strategy_portfolio_critic_prompt(
    *,
    target: str,
    strategy_cards: Iterable[Mapping[str, Any]],
) -> str:
    context = {
        "schema_version": "v9_strategy_portfolio_critic_input.v1",
        "phase": "strategy_portfolio_review",
        "campaign_target": target,
        "target_topology_profile": _target_topology_profile(target),
        "strategy_cards": [
            {
                key: value
                for key, value in dict(card).items()
                if key in {
                    "strategy_query",
                    "critical_assumption",
                    "critic_checkpoint",
                }
            }
            for card in strategy_cards
            if isinstance(card, Mapping)
        ],
    }
    return "\n".join(
        [
            "Act as the independent V9 Strategy Critic for one three-card portfolio before Route Builder search begins.",
            "Use target_topology_profile as deterministic scaffold fact. ring_pair_topology states how many atom pairs rings share; never infer a fused or spiro relationship from ring_sizes alone. Challenge whether each named key construction can plausibly account for the target's backbone and stereochemical burden, whether the stated reactive-handle motif is sufficient at a high level, and whether critical_assumption identifies the real make-or-break claim. Require critic_checkpoint to be the earliest non-substitutable graph transformation that directly tests that claim; reject a downstream event that could occur even if the critical assumption or an earlier required key construction never occurred.",
            "Keep useful directions. Revise or replace only weak, redundant, internally contradictory, or topologically irrelevant cards. Preserve three materially different skeletal construction logics; do not turn a strategy into a complete route or a required-map checklist.",
            "Treat each Generator card as immutable unless you can identify a specific weakness or contradiction in that card. Copy every acceptable card verbatim. If a card must be revised, preserve every unchallenged reactive-handle identity, protection or masking requirement, tether or precursor geometry clause, stereochemical-control clause, and sequencing constraint; never paraphrase merely for brevity or style.",
            "Return exactly three reviewed cards. Each card contains only strategy_query, critical_assumption, and critic_checkpoint, one concise sentence each. Do not expose the critique, score cards, write ReactionJSON, propose precursor structures or conditions, browse, inspect stock, or claim admission, validation, or solved status.",
            "V9StrategyPortfolioCriticInput:",
            json.dumps(
                context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )


def _strategy_prompt(
    *,
    target: str,
    branch_index: int,
    lens: str,
    forbidden_strategy_cards: Iterable[Mapping[str, Any]],
    prior_rejections: Iterable[Mapping[str, Any]],
    paper_matched: bool = False,
) -> str:
    if paper_matched:
        context = {
            "schema_version": "paper_matched_strategy_generator_input.v1",
            "phase": "strategy_generator",
            "campaign_target": target,
            "branch_id": branch_index + 1,
        }
        return "\n".join(
            [
                "Act as the paper-matched Strategy Generator and select one high-level strategy after internally assessing scaffold/backbone, one or two key forward reactions, functional-group/protection compatibility, and stereochemical construction or control.",
                "Output only one strategy_query sentence, one critical_assumption sentence, and one critic_checkpoint sentence. The strategy identifies the key forward event and its control logic; critic_checkpoint identifies the single actual graph transformation that should trigger a later sparse audit, not a preparatory handle installation or route stage.",
                "Routine FGI is strategic only when it directly enables the key construction. Do not output atom-map pairs, precursor structures, conditions, alternatives, rationales, limitations, tables, or mechanistic essays.",
                "Return only the compact StrategyCardReport. Do not build a route, write ReactionJSON, browse, inspect stock, or add evidence or enzyme fields.",
                "PaperMatchedStrategyGeneratorInput:",
                json.dumps(
                    context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    context = {
        "schema_version": (
            "strategy_portfolio_generator_input.v2"
            if "strategy_v2_slot=" in lens
            else "blind_strategy_generator_input.v1"
        ),
        "strategy_generation_version": (
            "v2" if "strategy_v2_slot=" in lens else "v1"
        ),
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
    if "strategy_v2_slot=" in lens:
        instructions = [
            "Act as the Strategy Generator for one branch of the AutoPlanner strategy-v2 portfolio.",
            "Use only the mapped target and the supplied strategy slot. Do not use web search, stock availability, literature provenance, or target identity lookup.",
            "Analyze scaffold topology, ring fusions and bridges, key skeletal construction or reorganization, functional-group orchestration, and stereochemical construction before selecting the strategy.",
            "The selected strategy must be a route-defining one-to-two-step construction; routine protection, redox, halogenation, methylation, or other FGI cannot be the strategic anchor unless it directly enables a subsequent skeletal event.",
            "Evaluate whether an annulation, cascade, cycloaddition, fragmentation, or skeletal rearrangement is credible when the target topology supports it. Do not force a named reaction without identifying the required reactive motifs.",
            "Separate anchor_bond_changes (target atom pairs used to bind the route search) from precursor_only_bond_changes (bonds that may exist only in the conceptual precursor and may be absent from the target). Use bond_order_changes for explicit reorganization rather than hiding it in prose.",
            "Provide conceptual_precursor_roles and required_reactive_features without drawing precursor SMILES. Explain atom_fragment_provenance and the substrate-specific failure mode so the Route Builder can compile a chemically meaningful ReactionJSON step.",
            "Return one complete StrategyCardReport for this branch. The card is a hypothesis only and grants no route, reaction proof, evidence, stock, or solved authority.",
        ]
    else:
        instructions = [
            "Act only as the Strategy Generator for one blind retrosynthesis branch.",
            "Do not build a route, propose precursor SMILES, write ReactionJSON, predict conditions, search literature, or use stock availability.",
            "Compare at least three materially distinct strategies on scaffold/ring topology, the key forward construction, functional-group and protection conflicts, stereochemical construction, convergence, and expected decomplexification.",
            "Select one strategy satisfying strategy_lens. It must be anchored on a route-defining one-to-two-step construction rather than a cosmetic FGI, protection, redox, nitration, halogenation, or methylation.",
            "For a key forward C-C, C-N, C-O, or other skeletal bond already present in the target, write key_bond_changes with mapped atom pairs exactly as map i-map j using campaign_target_mapped. Every pair must be an actual bond in campaign_target_mapped; do not invent atom indices, describe a future precursor bond, or use a map pair that is absent from the target graph.",
            "A biological strategy must name a chemically credible substrate-product transformation class; enzyme discovery and identity verification happen later.",
            "For execution_domain enzymatic, whole_cell, or hybrid, biocatalytic_intent is mandatory: name an enzyme class/EC/candidate or whole-cell host, the selectivity objective, substrate-scope basis, explicit cofactor assessment, intended chemical-step equivalence, a conventional fallback policy, and a falsifiable validation plan. This remains a strategy hypothesis and is not enzyme proof or verified step savings. For other domains return biocatalytic_intent=null.",
            "The selected strategy must be structurally orthogonal to forbidden_root_strategies, not just renamed or assigned different reagents.",
            "Return one StrategyCardReport. This artifact is a durable hypothesis but grants no route, reaction, evidence, stock, or solved authority.",
        ]
    return "\n".join(
        [
            *instructions,
            "StrategyGeneratorInput:",
            json.dumps(
                context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )


def _milestone_strategy_prompt(
    *,
    campaign_target: str,
    selected_product: str,
    selected_product_mapped: str,
    branch_index: int,
    milestone_index: int,
    strategy_mandate: str,
    completed_strategy_cards: Iterable[Mapping[str, Any]],
    route_steps: Iterable[Mapping[str, Any]],
) -> str:
    completed = [dict(card) for card in completed_strategy_cards]
    context = {
        "schema_version": "blind_upstream_strategy_milestone_input.v1",
        "phase": "route_internal_strategy_generator",
        "campaign_target": campaign_target,
        "selected_upstream_leaf": selected_product,
        "selected_upstream_leaf_mapped": selected_product_mapped,
        "selected_upstream_leaf_profile": _structure_profile(selected_product),
        "selected_upstream_leaf_bond_pairs": [
            f"map {left}-map {right}"
            for left, right in sorted(_mapped_bond_pairs(selected_product_mapped))
        ],
        "branch_id": branch_index + 1,
        "milestone_index": max(2, int(milestone_index)),
        "strategy_lens": strategy_mandate,
        "completed_milestones": [
            {
                "strategy_id": str(card.get("strategy_id") or ""),
                "key_bond_signature": list(card.get("key_bond_signature") or []),
                "topology_signature": str(card.get("topology_signature") or ""),
                "execution_domain": str(card.get("execution_domain") or ""),
            }
            for card in completed
        ],
        "accepted_target_rooted_prefix": _compact_route_rows(
            [dict(row) for row in route_steps if isinstance(row, Mapping)][-8:]
        ),
    }
    return "\n".join(
        [
            "Act only as the Strategy Generator for the next milestone inside an existing blind retrosynthesis branch.",
            "The target-rooted prefix and earlier StrategyCards are already host-replayed facts. Preserve them; do not restart or globally redesign the route.",
            "Select one new route-defining one-to-two-step construction for selected_upstream_leaf. It must create a further skeletal, ring-topology, stereochemical, or convergent-fragment simplification, not a cosmetic protection, redox, halogenation, methylation, or stock-driven edit.",
            "Do not build the route, propose precursor SMILES, write ReactionJSON, predict conditions, search literature, or use stock availability. Route Builder will execute this card one node at a time.",
            "Compare at least three leaf-local strategies internally and return only the strongest one that satisfies strategy_lens and is compatible with the accepted prefix.",
            "Write key_bond_changes only with map i-map j pairs that are actual bonds in selected_upstream_leaf_mapped. These inherited atom maps are authoritative; never renumber them from the canonical SMILES.",
            "Do not repeat a completed milestone. A biological or hybrid milestone is optional unless strategy_lens requests it; if exact substrate-product capability is not chemically credible, use the retained conventional chemical fallback instead of inventing an enzyme.",
            "Return one StrategyCardReport whose target_smiles is exactly selected_upstream_leaf. The card is a hypothesis and grants no route, reaction, evidence, stock, or solved authority.",
            "BlindUpstreamStrategyMilestoneInput:",
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
    paper_matched: bool = False,
    target_smiles: str = "",
) -> WorkerTask:
    return WorkerTask(
        task_id=(
            f"{spec.agent_id}:branch:{branch_index + 1}:strategy:{attempt_index}"
        ),
        case_id=_opaque_strategy_case_id(spec.run_id),
        task_type=(
            "paper_matched_strategy_generator"
            if paper_matched
            else "strategic_disconnection_mining"
        ),
        required_artifact_type="StrategyCardReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=(8_000 if paper_matched else 20_000),
            max_tool_calls=None,
            max_worker_runs=1,
            reasoning_effort=reasoning_effort,
        ),
        objective=prompt,
        allowed_workdir=str(spec.metadata.get("allowed_workdir") or "."),
        agent_mode="single",
        codex_auth_mode="ambient_codex_cli",
        model=model,
        host_context={"target_smiles": str(target_smiles or "")},
    )


def _strategy_portfolio_task(
    spec: AgentSpec,
    *,
    prompt: str,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
    target_smiles: str = "",
) -> WorkerTask:
    return WorkerTask(
        task_id=f"{spec.agent_id}:strategy-portfolio:1",
        case_id=_opaque_strategy_case_id(spec.run_id),
        task_type="paper_matched_strategy_generator",
        required_artifact_type="StrategyPortfolioReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=6_000,
            max_tool_calls=None,
            max_worker_runs=1,
            reasoning_effort=reasoning_effort,
        ),
        objective=prompt,
        allowed_workdir=str(spec.metadata.get("allowed_workdir") or "."),
        agent_mode="single",
        codex_auth_mode="ambient_codex_cli",
        model=model,
        host_context={"target_smiles": str(target_smiles or "")},
    )


def _strategy_portfolio_critic_task(
    spec: AgentSpec,
    *,
    prompt: str,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
    target_smiles: str = "",
) -> WorkerTask:
    return replace(
        _strategy_portfolio_task(
            spec,
            prompt=prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            target_smiles=target_smiles,
        ),
        task_id=f"{spec.agent_id}:strategy-critic:1",
        case_id=_opaque_strategy_case_id(spec.run_id + ":strategy-critic"),
        task_type="paper_matched_strategy_critic",
    )


def _strategy_card_from_record(
    record: WorkerRunRecord,
    *,
    expected_target: str,
    expected_mapped_target: str = "",
    paper_matched: bool = False,
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
    raw_card = dict(payload.get("strategy_card") or {})
    if paper_matched:
        raw_card = _paper_matched_strategy_card_payload(raw_card)
    card = normalize_strategy_card(raw_card)
    if not _valid_strategy_card(card):
        return None
    if not _strategy_card_bonds_match_target(
        card,
        target_smiles=expected_target,
        mapped_target_smiles=expected_mapped_target,
    ):
        return None
    return card


def _strategy_cards_from_portfolio_record(
    record: WorkerRunRecord,
    *,
    expected_target: str,
) -> list[dict[str, Any]] | None:
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    if (
        record.status != "accepted_draft"
        or artifact.get("artifact_type") != "StrategyPortfolioReport"
        or payload.get("schema_version") != "strategy_portfolio_report.v1"
        or _canonical_smiles(payload.get("target_smiles")) != expected_target
    ):
        return None
    raw_cards = payload.get("strategy_cards") or []
    if not isinstance(raw_cards, list) or len(raw_cards) != 3:
        return None
    cards: list[dict[str, Any]] = []
    for raw_card in raw_cards:
        if not isinstance(raw_card, Mapping):
            return None
        card = normalize_strategy_card(
            _paper_matched_strategy_card_payload(raw_card)
        )
        if not _valid_strategy_card(card) or not _strategy_card_bonds_match_target(
            card,
            target_smiles=expected_target,
        ):
            return None
        if _strategy_conflicts(card, cards):
            return None
        cards.append(card)
    return cards


def _paper_matched_strategy_card_payload(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expand a paper-style steering query into the legacy host contract.

    The additional fields are compatibility defaults, not extra model
    mandates. Paper-matched downstream prompts consume only the authored
    query and signature.
    """

    raw = dict(value or {})
    query = str(raw.get("strategy_query") or "").strip()
    critical_assumption = str(raw.get("critical_assumption") or "").strip()
    critic_checkpoint = str(raw.get("critic_checkpoint") or "").strip()
    return {
        **raw,
        # Identity is derived from the authored steering query.  The model no
        # longer fills a duplicate strategy_signature field.
        "strategy_signature": query,
        "critical_assumption": critical_assumption,
        "critic_checkpoint": critic_checkpoint,
        "scaffold_motif": query,
        "key_forward_transformation": query,
        "forward_transformation_class": query,
        "retrosynthetic_simplification": query,
        "key_bond_changes": [],
        "anchor_bond_changes": [],
        "precursor_only_bond_changes": [],
        "bond_order_changes": [],
        "conceptual_precursor_roles": [],
        "required_reactive_features": [],
        "atom_fragment_provenance": [],
        "functional_group_conflicts": [],
        "protection_policy": "encoded only in the authored steering query",
        "stereochemical_plan": "encoded only in the authored steering query",
        "stereochemical_control_basis": "encoded only in the authored steering query",
        "convergence_plan": query,
        "strategic_step_count": 1,
        "skeleton_change_class": query,
        "expected_complexity_drop": "high",
        "orthogonality_basis": query,
        "substrate_specific_failure_modes": [],
        "fallback_strategy": "",
        "strategy_basis": "paper-matched one-sentence steering query",
        "execution_domain": "chemical",
        "biocatalytic_intent": None,
    }


def _strategy_card_rejection_reason(
    record: WorkerRunRecord,
    *,
    expected_target: str,
    expected_mapped_target: str = "",
    paper_matched: bool = False,
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
    raw_card = dict(payload.get("strategy_card") or {})
    if paper_matched:
        raw_card = _paper_matched_strategy_card_payload(raw_card)
    card = normalize_strategy_card(raw_card)
    if not _valid_strategy_card(card):
        return "strategy_card_fields_invalid"
    if not _strategy_card_bonds_match_target(
        card,
        target_smiles=expected_target,
        mapped_target_smiles=expected_mapped_target,
    ):
        return "strategy_key_bond_not_in_target"
    return "strategy_card_output_invalid"


def _compact_builder_rejection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep one causal replay failure without duplicating its payload.

    Materialization records historically stored replay fields at the rejection
    root, while the paper Builder looked only for a nested replay_diagnostic.
    Normalize both shapes here so the next call sees the failed operation and
    host error instead of repeating the same edit under a new label.
    """

    row = dict(value)
    nested = dict(row.get("replay_diagnostic") or {})
    raw_chemical = dict(row.get("chemical_rejection") or {})
    is_chemical = bool(raw_chemical) or str(row.get("phase") or "") == (
        "key_event_critic"
    )
    diagnostic: dict[str, Any] = {}
    for key in (
        "reason",
        "replay_error",
        "operation_index",
        "failed_operation",
        "failure_stage",
        "failure_detail",
        "endpoint_aromaticity",
        "allowed_orders",
    ):
        candidate = nested.get(key)
        if candidate in (None, "", [], {}):
            candidate = row.get(key)
        if candidate not in (None, "", [], {}):
            diagnostic[key] = candidate
    compact: dict[str, Any] = {
        "reason": str(row.get("reason") or diagnostic.get("reason") or ""),
        "product_smiles": str(row.get("product_smiles") or ""),
    }
    if diagnostic and not is_chemical:
        compact["replay_diagnostic"] = diagnostic
    if is_chemical:
        reasons = [
            str(value)[:260]
            for value in raw_chemical.get("reasons") or row.get("reasons") or []
            if str(value).strip()
        ][:2]
        suggested_revision = str(
            raw_chemical.get("suggested_revision")
            or row.get("suggested_revision")
            or ""
        ).strip()[:420]
        compact["chemical_rejection"] = {
            "focus_step_id": str(
                raw_chemical.get("focus_step_id")
                or row.get("focus_step_id")
                or ""
            )[:160],
            "reasons": reasons,
            "suggested_revision": suggested_revision,
        }
    ancestor_smiles = [
        _canonical_smiles(value)
        for value in row.get("ancestor_smiles") or []
        if _canonical_smiles(value)
    ]
    if ancestor_smiles:
        compact["ancestor_smiles"] = list(dict.fromkeys(ancestor_smiles))
    attempted_net_edits = [
        dict(operation)
        for operation in normalize_reaction_operations(
            row.get("attempted_net_edits") or ()
        )
    ]
    if attempted_net_edits:
        compact["attempted_net_edits"] = attempted_net_edits
    mcts_state_fingerprint = str(row.get("mcts_state_fingerprint") or "")
    if mcts_state_fingerprint:
        compact["mcts_state_fingerprint"] = mcts_state_fingerprint
    return compact


def _step_claims_strategy_key_event(
    step: Mapping[str, Any], strategy_card: Mapping[str, Any] | None
) -> bool:
    """Schedule only a replayed Builder claim against Strategy's checkpoint.

    Reaction names and shared words have no authority.  The compact Builder
    relation requests the audit, while a real mapped skeletal edit is the
    Host-owned prerequisite.  The Critic still decides whether the edit truly
    instantiates the checkpoint and whether the chemistry is acceptable.
    """

    checkpoint = str(
        dict(strategy_card or {}).get("critic_checkpoint") or ""
    ).strip()
    row = dict(step)
    if (
        not checkpoint
        or _normalize_checkpoint_relation(row.get("checkpoint_relation"))
        != "executes_checkpoint"
    ):
        return False
    skeletal_edits = {
        (
            str(operation.get("op") or ""),
            *sorted(
                (
                    int(operation.get("map_a") or 0),
                    int(operation.get("map_b") or 0),
                )
            ),
        )
        for operation in normalize_reaction_operations(
            row.get("reaction_operations") or ()
        )
        if str(operation.get("op") or "") in {"break_bond", "add_bond"}
        and int(operation.get("map_a") or 0) > 0
        and int(operation.get("map_b") or 0) > 0
    }
    return bool(skeletal_edits)


def _key_event_fingerprint(step: Mapping[str, Any]) -> str:
    payload = {
        "mapped_product_smiles": str(step.get("mapped_product_smiles") or ""),
        "mapped_precursor_smiles": list(step.get("mapped_precursor_smiles") or []),
        "reaction_operations": [
            dict(value)
            for value in normalize_reaction_operations(
                step.get("reaction_operations") or ()
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]


def _key_event_focus_assessment(
    critique: Mapping[str, Any], focus_step_id: str
) -> dict[str, Any] | None:
    for value in critique.get("step_assessments") or []:
        if not isinstance(value, Mapping):
            continue
        assessment = dict(value)
        if str(assessment.get("step_id") or "") == str(focus_step_id):
            return assessment
    return None


def _connected_path_ancestor_smiles(
    steps: Iterable[Mapping[str, Any]],
    selected_product: str,
    selected_product_mapped: str = "",
) -> list[str]:
    """Return only products on the dependency chain leading to this leaf."""

    return [
        parent
        for row in reversed(
            _connected_path_step_rows(
                steps,
                selected_product,
                selected_product_mapped,
            )
        )
        if (parent := _canonical_smiles(row.get("product_smiles")))
    ]


def _compact_replayed_edit_summary(step: Mapping[str, Any]) -> str:
    """Describe one accepted host-replayed edit without copying structures."""

    labels: list[str] = []
    for operation in normalize_reaction_operations(
        step.get("reaction_operations") or ()
    ):
        op = str(operation.get("op") or "")
        if op in {"break_bond", "add_bond", "change_bond_order"}:
            pair = f"maps {operation.get('map_a')}-{operation.get('map_b')}"
            if op == "add_bond":
                labels.append(f"add bond {pair} order {operation.get('order')}")
            elif op == "change_bond_order":
                labels.append(
                    f"change bond {pair} by {operation.get('delta')}"
                )
            else:
                labels.append(f"break bond {pair}")
        elif op == "change_atom":
            field = "formal charge" if "formal_charge" in operation else "isotope"
            value = operation.get("formal_charge", operation.get("isotope"))
            labels.append(f"set {field} at map {operation.get('map_idx')} to {value}")
        elif op == "set_explicit_h":
            labels.append(
                f"set H count at map {operation.get('map_idx')} to {operation.get('count')}"
            )
        elif op == "add_group":
            labels.append(
                f"add {operation.get('fragment_smiles')} at map {operation.get('map_idx')}"
            )
        elif op == "remove_group":
            labels.append(
                "remove maps "
                + ",".join(str(value) for value in operation.get("map_indices") or [])
            )
        elif op in {"invert_stereocenter", "clear_stereocenter"}:
            labels.append(f"{op} at map {operation.get('map_idx')}")
        elif op == "set_bond_stereo":
            labels.append(
                f"set maps {operation.get('map_a')}-{operation.get('map_b')} stereo {operation.get('stereo')}"
            )
    return "; ".join(labels)[:480]


def _connected_path_step_rows(
    steps: Iterable[Mapping[str, Any]],
    selected_product: str,
    selected_product_mapped: str = "",
) -> list[dict[str, Any]]:
    """Return the Host-replayed target-to-leaf reaction spine.

    The mapped boundary is authoritative.  AiZ may serialize the same chiral
    precursor differently in its unmapped state; matching only that projection
    previously detached sibling leaves from their common coupling history.
    """

    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    current = _canonical_smiles(selected_product)
    current_mapped = _canonical_atom_mapped_smiles(selected_product_mapped)
    path: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if current or current_mapped:
        seen.add((current, current_mapped))
    while current or current_mapped:
        match = _parent_step_for_boundary(
            rows,
            product_smiles=current,
            mapped_product_smiles=current_mapped,
        )
        parent_row = match[0] if match is not None else None
        parent = (
            _canonical_smiles(parent_row.get("product_smiles"))
            if parent_row is not None
            else ""
        )
        parent_mapped = (
            _canonical_atom_mapped_smiles(
                parent_row.get("mapped_product_smiles")
            )
            if parent_row is not None
            else ""
        )
        parent_identity = (parent, parent_mapped)
        if parent_row is None or not parent or parent_identity in seen:
            break
        path.append(parent_row)
        seen.add(parent_identity)
        current = parent
        current_mapped = parent_mapped
    path.reverse()
    return path


def _parent_step_for_boundary(
    rows: Iterable[Mapping[str, Any]],
    *,
    product_smiles: str,
    mapped_product_smiles: str,
) -> tuple[dict[str, Any], int] | None:
    mapped_identity = _canonical_atom_mapped_smiles(mapped_product_smiles)
    canonical_identity = _canonical_smiles(product_smiles)
    constitution_identity = _canonical_smiles_nonisomeric(product_smiles)
    for raw in reversed(list(rows)):
        row = dict(raw)
        mapped_precursors = [
            _canonical_atom_mapped_smiles(value)
            for value in row.get("mapped_precursor_smiles") or []
        ]
        if mapped_identity:
            exact_mapped = [
                index
                for index, value in enumerate(mapped_precursors)
                if value and value == mapped_identity
            ]
            if len(exact_mapped) == 1:
                return row, exact_mapped[0]
        precursors = [
            _canonical_smiles(value)
            for value in row.get("precursor_smiles") or []
        ]
        exact = [
            index
            for index, value in enumerate(precursors)
            if canonical_identity and value == canonical_identity
        ]
        if len(exact) == 1:
            return row, exact[0]
        approximate = [
            index
            for index, value in enumerate(precursors)
            if constitution_identity
            and _canonical_smiles_nonisomeric(value) == constitution_identity
        ]
        if len(approximate) == 1:
            return row, approximate[0]
    return None


def _connected_path_reactions(
    steps: Iterable[Mapping[str, Any]],
    selected_product: str,
    selected_product_mapped: str = "",
) -> list[dict[str, str]]:
    """Return compact reaction history without promoting Builder claims to fact."""

    return [
        {
            "step_id": str(row.get("step_id") or ""),
            "reaction_family": str(
                row.get("reaction_family")
                or row.get("transformation_hypothesis")
                or ""
            )[:160],
            "checkpoint_relation": _normalize_checkpoint_relation(
                row.get("checkpoint_relation")
            ),
            "edit_summary": _compact_replayed_edit_summary(row),
        }
        for row in _connected_path_step_rows(
            steps,
            selected_product,
            selected_product_mapped,
        )
    ]


def _current_split_context(
    steps: Iterable[Mapping[str, Any]],
    *,
    selected_product: str,
    selected_product_mapped: str,
) -> dict[str, Any]:
    """Expose only the current parent split and its co-precursors."""

    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    match = _parent_step_for_boundary(
        rows,
        product_smiles=selected_product,
        mapped_product_smiles=selected_product_mapped,
    )
    if match is None:
        return {}
    parent, selected_index = match
    mapped_precursors = [
        str(value)
        for value in parent.get("mapped_precursor_smiles") or []
    ]
    canonical_precursors = [
        _canonical_smiles(value)
        for value in parent.get("precursor_smiles") or []
    ]
    later_rows = rows[rows.index(parent) + 1 :]
    siblings: list[dict[str, str]] = []
    for index, mapped in enumerate(mapped_precursors):
        if index == selected_index:
            continue
        canonical = (
            canonical_precursors[index]
            if index < len(canonical_precursors)
            else _canonical_smiles(mapped)
        )
        expanded = any(
            _canonical_atom_mapped_smiles(row.get("mapped_product_smiles"))
            == _canonical_atom_mapped_smiles(mapped)
            or _canonical_smiles(row.get("product_smiles")) == canonical
            for row in later_rows
        )
        siblings.append(
            {
                "mapped_smiles": mapped,
                "path_status": (
                    "expanded_on_current_path"
                    if expanded
                    else "not_expanded_on_current_path"
                ),
            }
        )
    if not siblings:
        return {}
    return {
        "parent_step_id": str(parent.get("step_id") or ""),
        "parent_reaction": str(
            parent.get("reaction_family")
            or parent.get("transformation_hypothesis")
            or ""
        )[:200],
        "co_precursors": siblings,
    }


def _node_prompt(
    *,
    target: str,
    branch_index: int,
    lens: str,
    selected_product: str,
    selected_product_mapped: str = "",
    steps: Iterable[Mapping[str, Any]],
    open_leaves: Iterable[str],
    prior_rejections: Iterable[Mapping[str, Any]],
    repair: bool,
    strategy_card: Mapping[str, Any],
    forbidden_strategy_cards: Iterable[Mapping[str, Any]],
    host_failure_feedback: Mapping[str, Any],
    complete_route_json: bool = False,
    editor_route_mutations: bool = False,
    compact_editor_context: bool = False,
    minimum_route_depth: int = 1,
    max_reactionjson_candidates: int = 1,
    paper_matched: bool = False,
) -> str:
    step_rows = [dict(step) for step in steps]
    strategy_anchor_fulfilled = _strategy_anchor_fulfilled_for_card(
        step_rows, strategy_card
    )
    accepted_path = (
        (
            _minimal_editor_prompt_route_rows(step_rows)
            if editor_route_mutations
            and (paper_matched or compact_editor_context)
            else _compact_route_rows(step_rows)
        )
        if complete_route_json or editor_route_mutations
        else [
            {
                "product_smiles": str(step.get("product_smiles") or ""),
                "precursor_smiles": list(step.get("precursor_smiles") or []),
                "reaction_family": str(
                    step.get("transformation_hypothesis") or ""
                ),
                "strategy_anchor": step.get("strategy_anchor") is True,
            }
            for step in step_rows
        ]
    )
    memory = {
        "schema_version": "compact_retrosynthesis_branch_context.v1",
        "phase": "local_repair" if repair else "route_builder",
        "campaign_target": target,
        "campaign_target_profile": _structure_profile(target),
        "branch_id": branch_index + 1,
        "strategy_lens": lens,
        "strategy_card": dict(strategy_card),
        "strategy_anchor_fulfilled": strategy_anchor_fulfilled,
        "forbidden_root_strategies": [dict(row) for row in forbidden_strategy_cards],
        "selected_open_leaf": selected_product,
        "selected_open_leaf_mapped": str(
            selected_product_mapped or _mapped_smiles(selected_product)
        ),
        "selected_open_leaf_profile": _structure_profile(selected_product),
        "accepted_path": accepted_path,
        "open_leaves": list(open_leaves),
        "prior_rejections": [dict(row) for row in prior_rejections],
        "host_failure_feedback": dict(host_failure_feedback),
        "route_json_contract": {
            "complete_linear_route_required": bool(complete_route_json),
            "minimum_route_depth": max(1, int(minimum_route_depth)),
            "editor_complete_route_json_default": bool(editor_route_mutations),
            "editor_may_reorder_insert_delete": bool(editor_route_mutations),
            "editor_may_change_conditions": bool(editor_route_mutations),
            "editor_may_change_functional_groups": bool(editor_route_mutations),
            "editor_may_change_route_length": bool(editor_route_mutations),
            "editor_may_change_terminal_leaves": bool(editor_route_mutations),
            "editor_may_return_dependency_closed_route_json": bool(
                editor_route_mutations
            ),
            "editor_context_compacted": bool(
                editor_route_mutations
                and (paper_matched or compact_editor_context)
            ),
            "maximum_steps": 25,
        },
    }
    if paper_matched and not repair:
        selected_canonical = _canonical_smiles(selected_product)
        current_mcts_state_fingerprint = _aiz_policy_state_fingerprint(
            selected_leaf_mapped=str(
                selected_product_mapped or _mapped_smiles(selected_product)
            ),
            route_steps=step_rows,
        )
        leaf_rejections = [
            _compact_builder_rejection(row)
            for row in prior_rejections
            if isinstance(row, Mapping)
            and _canonical_smiles(row.get("product_smiles"))
            == selected_canonical
            and (
                str(row.get("reason") or "")
                not in {
                    "candidate_did_not_advance_selected_mcts_path",
                    "candidate_repeats_same_mcts_state_edit",
                }
                or str(row.get("mcts_state_fingerprint") or "")
                == current_mcts_state_fingerprint
            )
        ]
        connected_path_reactions = _connected_path_reactions(
            step_rows,
            selected_product,
            selected_product_mapped,
        )
        ancestor_smiles = _connected_path_ancestor_smiles(
            step_rows,
            selected_product,
            selected_product_mapped,
        )
        current_split_context = _current_split_context(
            step_rows,
            selected_product=selected_product,
            selected_product_mapped=selected_product_mapped,
        )
        pending_checkpoint_feedback = dict(
            host_failure_feedback.get("pending_checkpoint_feedback") or {}
        )
        last_rejection_for_this_leaf = (
            leaf_rejections[-1] if leaf_rejections else {}
        )
        # The key-event Critic owns checkpoint feedback.  The same chemical
        # rejection used to be copied into the generic leaf rejection as
        # well, making the next Builder call read the full diagnosis twice.
        # Keep last_rejection_for_this_leaf for Host replay/cycle failures.
        if (
            pending_checkpoint_feedback
            and last_rejection_for_this_leaf.get("chemical_rejection")
        ):
            last_rejection_for_this_leaf = {}
        # Match the paper's next-step policy boundary: current node, concise
        # steering query, the accepted reaction spine, and only this leaf's
        # latest causal failure.  Full route structures belong to the host,
        # Critic, and Editor, not the next-step policy call.
        memory = {
            "schema_version": "paper_matched_route_builder_context.v7",
            "phase": "route_builder_node",
            "target_smiles": target,
            "strategy": {
                key: value
                for key, value in dict(strategy_card).items()
                if key in {
                    "strategy_query",
                    "critical_assumption",
                    "critic_checkpoint",
                }
            },
            "selected_leaf_mapped": str(
                selected_product_mapped or _mapped_smiles(selected_product)
            ),
        }
        for key, value in (
            ("connected_path_reactions", connected_path_reactions),
            ("ancestor_smiles", ancestor_smiles),
            ("current_split_context", current_split_context),
            ("last_rejection_for_this_leaf", last_rejection_for_this_leaf),
            ("pending_checkpoint_feedback", pending_checkpoint_feedback),
        ):
            if value:
                memory[key] = value
    if paper_matched and not repair:
        context_guidance: list[str] = []
        if memory.get("connected_path_reactions"):
            context_guidance.append(
                "connected_path_reactions is the complete compact Host-replayed reaction history on this target-to-leaf path. Use its reaction families and edit summaries to avoid undoing, repeating, or falsely claiming chemistry that has not occurred."
            )
        if memory.get("current_split_context"):
            context_guidance.append(
                "current_split_context contains only the parent reaction and mapped co-precursors from the current split. Use it for coupling-handle and functional-group compatibility; it is not the full search tree and its path_status is not a stock claim."
            )
        if memory.get("ancestor_smiles"):
            context_guidance.append(
                "ancestor_smiles is structural negative memory only. The replayed precursor set must not contain any listed ancestor; choose a different disconnection or functional-group move instead."
            )
        if memory.get("last_rejection_for_this_leaf"):
            context_guidance.append(
                "last_rejection_for_this_leaf is the latest Host replay, cycle, or AiZ same-state no-progress failure for this leaf. Repair that exact local cause and do not repeat any attempted_net_edits under a new reaction name."
            )
        if memory.get("pending_checkpoint_feedback"):
            context_guidance.append(
                "pending_checkpoint_feedback is the sole compact Key-event Critic memory and persists across preparatory moves, timeouts, and changes of selected leaf until the actual checkpoint passes. Repair its exact topology, handle, stereochemical, or sequence-dependency cause; do not erase it by relabeling a substitute as preparatory."
            )
        return "\n".join(
            [
                "Act as the Route Builder's next-step expansion policy for one selected MCTS node. strategy.strategy_query is the steering hypothesis and guides the whole pathway; strategy.critic_checkpoint names the one actual graph transformation reserved for the sparse key-event audit.",
                "Privately work out a complete chemically coherent pathway from selected_leaf_mapped through the Strategy's named construction toward accessible precursors, and compare plausible disconnections in that route context. Return only the single best current ReactionJSON move for selected_leaf_mapped. The one-object output boundary does not limit route-level reasoning; omit alternatives and the comparison process.",
                "One candidate must represent one executable reaction. A concerted or cascade key reaction may contain several graph edits, but its operations must encode the complete connected bond-change pattern; do not split one reaction into fictitious intermediate steps. Privately challenge the chosen move, then compile and mentally replay it against selected_leaf_mapped before answering; use only maps present there because the Host derives both endpoints.",
                "Check the Strategy against the actual net graph edit, not the reaction name. When the named construction consumes or creates specific reactive handles, those mapped atoms and bonds must participate in the defining operations. When stereochemical control is part of the named construction, the relevant stereochemistry or geometry must be represented or deliberately transformed in the replayable structures and operations. reaction_intent, catalysts, and conditions cannot substitute for missing topology or stereochemical information.",
                "Set checkpoint_relation=executes_checkpoint only when this candidate's ordered operations themselves realize strategy.critic_checkpoint. Set checkpoint_relation=preparatory for handle installation, unmasking, functional-group adjustment, or any other step that merely enables or mentions the checkpoint. This label is scheduling metadata, not proof or admission.",
                "ReactionJSON primitive syntax is exact: change_bond_order uses signed delta; change_atom changes formal_charge or isotope only; atom installation/removal uses add_group/remove_group. add_bond always creates a single bond and has no order field; to create a new double or triple bond, follow add_bond with change_bond_order delta 1 or 2. add_group fragment_smiles contains exactly one [*] attachment atom and encodes its attachment bond directly, for example [*]O, [*]=O, or [*]#N; do not output order. For set_bond_stereo provide only map_a, map_b, and E/Z/CIS/TRANS/NONE/ANY intent; the Host derives RDKit stereo reference neighbours.",
                "Conditions describe the forward reaction environment for the Host-replayed precursor set -> selected_leaf_mapped product. They must be chemically compatible with that forward transformation even though ReactionJSON operations are written retrosynthetically from product to precursors. Protection/deprotection, tether or reactive-handle installation/removal, activation, and every other covalent state change that defines a precursor must be encoded by ReactionJSON in its own executable step, never claimed only in conditions.",
                "Express the reaction family and purpose together as one concise reaction_intent sentence. Keep conditions concise and include any catalyst there; they are hypotheses, not proof or a Critic verdict.",
                "Check functional-group compatibility within the replayed precursor set, including incompatible protic/basic, organometallic, redox, or catalyst-sensitive handles. If compatibility requires a covalent change, make that change an explicit step rather than a condition note.",
                "Prefer a move that advances the steering hypothesis. Necessary enabling reactions may be performed one at a time when the current leaf lacks the required handles; once selected_leaf_mapped contains the needed reactive topology, prefer executing the named key construction instead of accumulating unrelated enabling or supporting transformations.",
                *context_guidance,
                "The Host/MCTS alone decides termination, budget exhaustion, stock and solved status, and short-tail stitching. The Builder has no handoff, fail, stop, or solved action; always return the best available ReactionJSON expansion.",
                "Return only checkpoint_relation, reaction_intent, ordered reaction_operations, and concise conditions. Return no complete RouteJSON, route skeleton, evidence, source, enzyme, validation, stock claim, or long explanation.",
                "PaperMatchedRouteBuilderContext:",
                json.dumps(memory, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ]
        )
    if paper_matched and editor_route_mutations:
        feedback = dict(host_failure_feedback)
        raw_materialization_failure = dict(
            feedback.get("editor_materialization_failure") or {}
        )
        materialization_failure = {
            key: raw_materialization_failure[key]
            for key in (
                "reason",
                "step_index",
                "failed_step_id",
                "product_smiles",
                "compiler_error",
                "detail",
                "operation_index",
                "failed_operation",
                "failure_stage",
                "failure_detail",
                "endpoint_aromaticity",
                "allowed_orders",
                "host_replayed_prefix_step_count",
                "host_open_precursors",
                "mapped_open_precursor_authority",
                "host_selected_open_precursor",
            )
            if key in raw_materialization_failure
        }
        repair_history: dict[str, Any] = {}
        if materialization_failure:
            repair_history["last_host_replay_failure"] = materialization_failure
        editor_context = {
            "schema_version": "paper_matched_route_editor_context.v4",
            "campaign_target": target,
            "strategy": {
                key: value
                for key, value in dict(strategy_card).items()
                if key in {"strategy_query", "strategy_signature"}
            },
            "route_json": accepted_path,
            "critic_annotations": {
                key: value
                for key, value in feedback.items()
                if key
                not in {
                    "editor_instruction",
                    "editor_materialization_failure",
                }
            },
            "route_replay": {
                "route_json_source": (
                    "host_checked_editor_working_draft"
                    if materialization_failure
                    else "current_host_replayed_route"
                ),
                "host_replayed_prefix_step_count": int(
                    materialization_failure.get(
                        "host_replayed_prefix_step_count"
                    )
                    or 0
                ),
                "mapped_open_precursor_authority": (
                    materialization_failure.get(
                        "mapped_open_precursor_authority"
                    )
                    or "host_routejson_dag_compiler"
                ),
            },
            "repair_history": repair_history,
        }
        return "\n".join(
            [
                "Act as the paper-style RouteJSON Editor. You receive the complete Host-replayed route plus Critic annotations, but return only the smallest dependency-closed replace_span that resolves every blocker. remove_step_ids names the old rows to replace; revised_steps contains the complete replacement chemistry. The Host preserves every unlisted row, merges the span, and replays the full route.",
                "Smallest means chemically sufficient, not fewest rows. Preserve unrelated viable chemistry, but enlarge the span through every affected dependency when a boundary changes; the span may cover one row, several rows, or the whole route.",
                "Choose the target-side steps you intend to preserve first and treat each exact Host-derived precursor they consume as a retained boundary. Revised upstream chemistry must directly generate every retained boundary it reconnects to. If no chemically coherent direct connection exists, include the incompatible retained step in remove_step_ids and revise a larger span.",
                "Do not invent an unsupported intermediate transformation merely to bridge independently designed endpoints. A net graph edit listed in rejected_net_edit_signatures remains rejected even if its reaction name, catalyst, or conditions change.",
                "Keep revised_steps in target-rooted retrosynthetic storage order. Its first product_smiles must be an exact open precursor at the retained target-side boundary (or the campaign target when replacing the root), and every later product_smiles must be emitted by an earlier row after the span is merged. Never put a newly exposed precursor into the replacing row's product_smiles.",
                "On a retry, repair_history contains only the Host's exact failure boundary. Use last_host_replay_failure.host_selected_open_precursor and host_open_precursors as map authority; do not reconstruct or renumber those structures.",
                "A retry must repair last_host_replay_failure.failed_operation at its operation_index, or revise the causal topology when no operation is identified; do not repeat the failed edit unchanged. If failed_step_id names an unlisted retained row, include that exact id in remove_step_ids and replace it rather than changing only earlier rows. Before returning, verify that every field change claimed in repair_summary is present in revised_steps.",
                "ReactionJSON primitive syntax is exact: change_bond_order uses signed delta; change_atom changes formal_charge or isotope only; atom installation/removal uses add_group/remove_group. add_bond always creates a single bond and has no order field; to create a new double or triple bond, follow add_bond with change_bond_order delta 1 or 2. add_group fragment_smiles contains exactly one [*] attachment atom and encodes its attachment bond directly, for example [*]O, [*]=O, or [*]#N; do not output order. For set_bond_stereo provide only map_a, map_b, and stereo intent; the Host derives RDKit reference neighbours.",
                "Atom maps are Host graph-replay identities. Use entry maps visible in route_json; introduce or remove atoms explicitly through add_group/remove_group, and give newly introduced atoms stable explicit maps when a later revised step must reference them. Do not output mapped_product_smiles or any precursor list; the Host derives both.",
                "Never invent stock availability, truncate an unresolved dependency, or promote an unavailable advanced intermediate to claim closure. New structures are allowed only when introduced through explicit, chemically meaningful, replayable ReactionJSON steps.",
                "Return one brief repair_summary and one replace_span. Each revised step needs only step_id, product_smiles, reaction_family, concise conditions/catalyst, and ordered reaction_operations. Do not output alternatives, tables, long explanations, evidence, validation, stock, or solved claims.",
                "PaperMatchedRouteEditorContext:",
                json.dumps(
                    editor_context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    strategy_domain = str(strategy_card.get("execution_domain") or "chemical")
    if strategy_anchor_fulfilled:
        phase_instructions = [
            "The immutable StrategyCard has already been executed by the strategy_anchor step in accepted_path. Preserve that target-level strategy, but do not repeat its key bond cleavage on this upstream leaf and do not require those already-cleaved bonds to remain present.",
            "Act as the Route Builder for the selected upstream leaf. Propose the best leaf-local precursor transformation that enables, prepares, or supplies the accepted route prefix; a functional-group or handle-installation step is allowed when it has a concrete forward role.",
            "Compare at least three local disconnections internally on route-prefix compatibility, skeletal simplification, chemoselectivity, stereochemical compatibility, and precursor accessibility, then return only the best candidate.",
            "Stock availability is an endpoint test, never a chemical justification for a disconnection.",
        ]
    else:
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
    if complete_route_json:
        phase_instructions.extend(
            [
                "Return candidate.route_json as the complete ordered linear retrosynthetic route beginning at selected_open_leaf and continuing to terminal starting-material leaves; do not stop after the key disconnection.",
                "Every route_json step must be contiguous: after the first step, each product_smiles must be one precursor generated by an earlier step, and every step must provide replayable reaction_operations.",
                "RouteJSON contains reaction transformations only. Do not emit terminal starting-material leaves as extra steps, do not emit a step with empty reaction_operations, and do not use a no-op step to pad the route. The last transformation's replayed precursor fragments are the terminal leaves.",
                "Preserve atom-map identities across the full route: later-step operations must use the maps present on the fragment emitted by the preceding replay, not freshly renumbered atoms.",
                f"This route suffix must contain at least {max(1, int(minimum_route_depth))} real replayable transformation step(s). One step is valid at the target root or any later leaf when its replayed precursors close in stock; never add no-op padding.",
                "Keep candidate.precursor_smiles and every route_json step precursor_smiles empty because the host derives all precursor structures by replay.",
            ]
        )
    if editor_route_mutations:
        phase_instructions = [
            "Act as the paper-style RouteJSON Editor. Return a complete revised candidate.route_json by default. Use candidate.route_patch only for a conditions-only edit or a genuinely isolated single-step mutation whose product boundary and every dependency remain unchanged. Preserve the campaign target and the overall Strategy intent; the frozen route rows are editable input, not immutable topology.",
            "The document must remain in target-rooted retrosynthetic storage order: the target-producing disconnection is first and each later product is an exact precursor emitted by an earlier step. Do not reorder it into laboratory forward-execution order; forward executability is checked by reversing the dependency traversal, not by reversing RouteJSON storage.",
            "Repair all Critic-identified blockers as one coordinated route document. You may reorder, insert, delete, or replace steps; alter conditions; add or remove functional groups and reaction handles through ReactionJSON; and change route length or terminal precursor identities when chemistry and dependency continuity require it. Preserve a non-blocking row only when it remains compatible with the coordinated repair.",
            "A rejected disconnection may be replaced when it is not chemically repairable as serialized. Retain the Strategy's overall synthetic intent and key construction when defensible, but do not preserve an impossible named mechanism or exact bond cut merely because it appeared in the initial StrategyCard.",
            "For an infeasible fragment union, install or use explicit complementary reaction handles, insert the required preparation/protection sequence, or replace the disconnection. Conditions alone cannot make an unfunctionalized graph-disconnected coupling executable.",
            "A revised route may be shorter or longer for a chemical reason, but never truncate an unresolved suffix or promote an unavailable advanced intermediate merely to lower the blocking fraction. It must begin at the campaign target, contain a complete connected dependency graph, and end at defensible terminal starting-material leaves; those terminal leaves need not equal the frozen route's leaves.",
            "If no chemically defensible complete repair can be encoded, return route_patch=[] and route_json=null so the host retains the original route and blocking critique rather than accepting a truncated or fabricated route.",
            "Every replacement step must be contiguous and independently replayable from its product through reaction_operations; precursor_smiles remain empty and are host-derived.",
            "For a local replace_step patch, step_id must identify the frozen row and its product boundary must stay unchanged. When a product boundary changes, return a complete chain-valid route_json with every affected dependency updated.",
            "RouteJSON is target-rooted: the first row has product_smiles=campaign target, and each later row's product_smiles is one precursor emitted by the previous row. Do not put a later precursor into an earlier row's product field.",
            "Use host_failure_feedback causally and return one edited route mutation, not several candidates. The host recompiles every edited route from the target and owns all mapped products and precursors.",
        ]
        if compact_editor_context:
            phase_instructions.insert(
                1,
                "accepted_path is compacted only by omitting non-structural prose and optional metadata; it still contains every frozen step, dependency boundary, and exact ReactionJSON program. Omitted text is not permission to delete or redesign a step.",
            )
    if strategy_domain in BIOLOGICAL_EXECUTION_DOMAINS:
        phase_instructions.extend(
            [
                "Classify each returned step independently. Set execution_domain=chemical for ordinary chemistry even inside a biological or hybrid branch; never copy the branch domain onto every step.",
                "For an enzymatic, whole_cell, or hybrid step, populate biocatalytic_step with enzyme/host identity at the strongest honest level, selectivity objective, substrate-scope basis, cofactor ledger assessment, precedent refs, and a falsifiable validation plan. For a chemical step set biocatalytic_step=null.",
                "Do not claim route-level step savings in ReactionJSON. The host binds the exact substrate-product boundary here; chemical-step equivalence and net savings are computed only later against an explicit retained chemical fallback span.",
            ]
        )
    candidate_limit = max(1, int(max_reactionjson_candidates))
    opening = (
        "Return exactly one candidate containing route_patch or one complete linear RouteJSON replacement."
        if editor_route_mutations
        else (
            "Return exactly one candidate containing one complete linear RouteJSON route."
            if complete_route_json
            else (
                "Return one JSON object that expands the selected node, hands off an eligible simple upstream precursor, or fails this Strategy branch."
                if paper_matched
                else (
                    "Expand exactly one retrosynthetic node and return one JSON object "
                    f"containing 1 to {candidate_limit} ranked, structurally distinct candidates."
                )
            )
        )
    )
    map_instruction = (
        "For every edited RouteJSON row, use only map indices present in that row's supplied mapped_product_smiles. Copy the exact mapped boundary for unchanged/replaced rows; for inserted rows, preserve the map namespace emitted by the dependency-producing ReactionJSON edit. Never renumber the whole route or reuse a map from another fragment namespace."
        if editor_route_mutations
        else (
            "For every RouteJSON row, use only map indices present in that row's mapped product boundary and preserve atom-map identities across dependencies."
            if complete_route_json
            else "Use only map indices present in selected_open_leaf_mapped."
        )
    )
    reactionjson_instruction = (
        "For route_patch, encode each structural mutation in the patched step's ordered reaction_operations; set_conditions may preserve the existing operations. For route_json, every row owns its ordered reaction_operations. precursor_smiles must be [] because the host deterministically derives all precursor structures by replay."
        if editor_route_mutations
        else (
            "When expanding, write the ordered candidate.reaction_operations first and set candidate.precursor_smiles=[]. The host applies ReactionJSON to selected_open_leaf_mapped and deterministically derives the canonical precursor structures. Handoff returns candidates=[] and no ReactionJSON candidate."
            if paper_matched
            else "Write the ordered candidate.reaction_operations first. candidate.precursor_smiles must be []; the host applies ReactionJSON to selected_open_leaf_mapped and deterministically derives the canonical precursor structures."
        )
    )
    return "\n".join(
        [
            opening,
            (
                "Route state is isolated from the other two independent branches; no cross-branch reaction-family mandate applies."
                if paper_matched
                else "Route state is isolated from other branches; compact prior StrategyCards are supplied only to enforce portfolio orthogonality."
            ),
            *phase_instructions,
            (
                "When a candidate is returned, its product_smiles must equal selected_open_leaf exactly after canonicalization."
                if paper_matched and not editor_route_mutations
                else "The candidate product_smiles must equal selected_open_leaf exactly after canonicalization."
            ),
            reactionjson_instruction,
            "The replayed output may contain at most four atom-contributing precursor molecules, must preserve the heavy-atom inventory required by the product, and must not repeat an ancestor.",
            map_instruction,
            "Do not use nullable schema filler fields on an operation; each primitive must contain only its semantically relevant fields.",
            "ReactionJSON field contract: break_bond/add_bond use map_a and map_b, and add_bond creates a single bond; change_bond_order uses map_a, map_b, and numeric delta; change_atom uses map_idx plus exactly one of formal_charge or isotope; set_explicit_h uses map_idx, count, and no_implicit; add_group uses map_idx and fragment_smiles, whose [*] attachment bond carries the bond order; remove_group uses map_indices; set_bond_stereo uses map_a, map_b, and stereo intent only because the Host derives RDKit reference neighbours. Do not output an order or stereo_atom_maps field.",
            "Never use change_atom to transmute an existing carbon/heteroatom into a new element; that is a host rejection. To attach new atoms, use add_group with exactly one dummy attachment, e.g. fragment_smiles='[*]Br' or '[*][Mg]Br'. The host deterministically assigns fresh maps to unmapped added atoms; explicit positive maps are also accepted when unique and collision-free. Bare 'Br' without the dummy attachment is invalid.",
            "If prior_rejections contains strategy_graph_edit_replay_failed, use its replay_diagnostic reason, attempted operations, and replayed fragments to repair the edit program. Do not rename the same idea or append unrelated hydrogen edits.",
            (
                (
                    "Return one complete edited route_json or one coordinated route_patch, not a prose-only sketch and not multiple output candidates."
                    if paper_matched
                    else "Return one route_patch or complete RouteJSON replacement, not a prose-only sketch and not multiple output candidates."
                )
                if editor_route_mutations
                else (
                    "Return one complete RouteJSON route, not a prose-only sketch and not multiple output candidates."
                    if complete_route_json
                    else (
                        "Return exactly one local ReactionJSON transformation candidate; the Builder has no terminal action and must not return a prose-only route."
                        if paper_matched
                        else (
                            f"Return 1 to {candidate_limit} local transformation candidates "
                            "inside the candidates array, ranked best first; do not return prose-only routes."
                        )
                    )
                )
            ),
            "The output is hypothesis-only and grants no validation, stock, evidence, condition authority, or solved claim.",
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
    paper_matched: bool = False,
    target_smiles: str = "",
    selected_product: str = "",
) -> WorkerTask:
    effective_task_type = task_type
    if paper_matched and task_type == "route_step_materialization":
        effective_task_type = "paper_matched_route_step"
    elif paper_matched and task_type == "route_chemistry_edit":
        effective_task_type = "paper_matched_route_editor"
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
        task_type=effective_task_type,
        required_artifact_type="RetrosynthesisProposalReport",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=(
                40_000
                if paper_matched and task_type == "route_chemistry_edit"
                else (16_000 if paper_matched else 32_000)
            ),
            max_tool_calls=None,
            max_worker_runs=1,
            reasoning_effort=reasoning_effort,
        ),
        objective=prompt,
        allowed_workdir=str(spec.metadata.get("allowed_workdir") or "."),
        agent_mode="single",
        codex_auth_mode="ambient_codex_cli",
        model=model,
        host_context={
            "target_smiles": str(target_smiles or ""),
            "selected_product": str(selected_product or ""),
        },
    )


def _opaque_strategy_case_id(run_id: str) -> str:
    return "strategy-case:" + hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:20]


def _critic_prompt(
    *,
    target: str,
    branch_index: int,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
    strategy_milestone_cards: Iterable[Mapping[str, Any]] = (),
    compact_level: int = 0,
    paper_matched: bool = False,
    audit_kind: str = "final_route",
    focus_step_id: str = "",
) -> str:
    level = max(0, int(compact_level))
    critic_strategy = _critic_strategy_card(
        strategy_card,
        compact_level=level,
    )
    if paper_matched:
        critic_strategy = {
            key: value
            for key, value in dict(strategy_card).items()
            if key in {
                "strategy_query",
                "critical_assumption",
                "critic_checkpoint",
            }
            and value not in (None, "", [], {})
        }
    critic_steps = [
        (
            _paper_critic_step_row(step, compact_level=level)
            if paper_matched
            else _critic_step_row(step, compact_level=level)
        )
        for step in steps
        if isinstance(step, Mapping)
    ]
    route = {
        "schema_version": "blind_route_critic_input.v1",
        "phase": (
            "key_event_candidate_audit"
            if audit_kind == "key_event"
            else "independent_chemical_critic"
        ),
        "campaign_target": target,
        "branch_id": branch_index + 1,
        "strategy_card": critic_strategy,
        "steps": critic_steps,
    }
    if audit_kind == "key_event":
        route["focus_step_id"] = str(focus_step_id)
    if not paper_matched:
        route["strategy_milestone_cards"] = [
            _critic_strategy_card(card, compact_level=level)
            for card in strategy_milestone_cards
            if isinstance(card, Mapping)
        ]
    if paper_matched:
        if audit_kind == "key_event":
            return "\n".join(
                [
                    "Act as the independent V9 key-event Critic. The Host has replayed one new Builder candidate marked executes_checkpoint; audit only focus_step_id against strategy_card.critic_checkpoint.",
                    "The preceding steps are immutable root-to-leaf context. Do not reject them, demand a complete route, require stock closure, or penalize a key event merely because later upstream synthesis is absent.",
                    "First decide checkpoint_match from the Host-derived mapped product, mapped precursors, and ordered graph edits. reaction_family and checkpoint_relation are scheduling claims, not evidence. checkpoint_match=true only when the actual edit instantiates critic_checkpoint and directly tests critical_assumption; exposing, preparing, unmasking, or executing a downstream event that leaves the critical assumption untested is false.",
                    "Return only checkpoint_match, verdict, blocking_type, at most two reasons, and one smallest local suggested_revision. When checkpoint_match=false because the action is a benign mislabeled preparatory move that preserves the topology required by the Strategy, use verdict=uncertain and blocking_type=none so the Host can retain it as preparatory. When checkpoint_match=false because the action substitutes for, consumes, or irreversibly cuts the required topology (for example, splitting a required intramolecular precursor into unrelated fragments), use verdict=reject and blocking_type=sequence_dependency so only that proposed action is rolled back. When checkpoint_match=true, pass means coherent execution, uncertain means plausible but unresolved, and reject requires a specific topology, handle, mechanism, atom-provenance, stereochemical, compatibility, or Strategy contradiction.",
                    "Missing literature, route incompleteness, later upstream synthesis, and stock metadata are never blockers. Return no route rewrite or long analysis.",
                    "V9KeyEventCriticInput:",
                    json.dumps(
                        route,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        return "\n".join(
            [
                "Act as an independent Route Critic. Forward-simulate every supplied reaction from the current frontier toward the target while leaving the target-rooted RouteJSON storage order unchanged.",
                "Use the exact host-derived mapped products, mapped precursors, ReactionJSON operations, and proposed conditions. Check mechanism, net structural/H/charge/redox plausibility, reactive handles, functional-group and stereochemical compatibility, selectivity, and sequence dependencies.",
                "Independently compare strategy_card.strategy_query, critical_assumption, and critic_checkpoint with the actual reaction families, structures, and bond edits. No Builder checkpoint_relation, role label, or host anchor claim is evidence. strategy_adherence=true only when at least one supplied step itself executes critic_checkpoint; a step that merely exposes or prepares a later event does not satisfy it.",
                "Atom maps are host graph-replay identities and preserve mapped element identity. Routine reagents and coproducts may be omitted, but atom installation/removal must be explicit through add_group/remove_group; change_atom may change formal charge or isotope only. Reject an element transmutation or unexplained mapped-atom provenance break, not the mere omission of non-route reagents.",
                "For each step: pass means executable as written; uncertain means plausible but unresolved; reject and blocking=true require a specific chemical contradiction or a specific contradiction of the supplied Strategy contract. Missing literature, stock metadata, or merely underspecified conditions are not blockers.",
                "Keep each assessment concise: at most two concrete reasons, a short condition assessment, and the smallest structure-local suggested revision. Do not output a long mechanistic analysis or repeat the route description.",
                "If the complete supplied route never performs the Strategy's named key construction, set strategy_adherence=false and reject the closest substituted or falsely key step with blocking=true and blocking_type=sequence_dependency. Ask the Editor to insert or replace the missing key event while preserving unrelated viable chemistry. This is a route-contract contradiction, not a score penalty. overall_assessment is reject if any step blocks, uncertain if none block and at least one is uncertain, otherwise viable.",
                "For any fragment union without complementary handles, require explicit handle installation/use or a chemically explicit replacement topology. A changed label, catalyst, or condition cannot repair a missing structural handle.",
                "Repair actions must preserve unrelated viable chemistry and the supplied target-to-current-frontier boundary; never improve the score by truncating a supplied suffix or claiming an advanced frontier intermediate is stock.",
                "Return only the compact critique defined by the schema.",
                "PaperMatchedRouteCriticInput:",
                json.dumps(route, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ]
        )
    audit_scope = "Audit atom provenance, plausible mechanism, functional-group compatibility, site selectivity, stereochemistry, sequence order, competing pathways, and enzyme identity/capability. For biological steps, also audit the exact host-bound substrate-product boundary, enzyme/host specificity, cofactor ledger, and whether the stated validation plan can falsify the capability claim."
    return "\n".join(
        [
            "Act as an independent senior synthetic chemist and forward-simulate every reaction in this frozen route.",
            "RouteJSON is stored in target-rooted retrosynthetic order: the first step consumes the final target, and every later step consumes a precursor emitted by an earlier step. Do not reject or reorder the document merely because this storage order is opposite to laboratory execution. Forward-simulate chemistry by traversing dependencies from terminal precursors back toward the target while preserving target-rooted RouteJSON order.",
            "This also applies to a route-local repair: audit the replacement neighborhood while preserving the frozen route strategy.",
            "You did not design the route. Do not preserve it out of politeness and do not replace its StrategyCard silently.",
            audit_scope,
            "A missing paper is not a chemical rejection. Do not browse or use target-name knowledge; judge only the supplied structures and route contract.",
            "Classify each supplied step independently: pass means the serialized transformation is chemically coherent as written; uncertain means conditions, precedent, substrate scope, or selectivity remain unresolved without a concrete contradiction; reject means a specific mechanistic, atom-provenance, functional-group, chemoselectivity, stereochemical, or dependency contradiction makes that step non-executable as written.",
            "Set overall_assessment=reject when any step is reject/blocking, uncertain when no step is reject but at least one is uncertain, and viable only when every step passes. Missing literature or stock metadata alone is never grounds for reject.",
            "For every rejected step, name the exact step_id and the smallest structure-local replacement boundary in repair_actions. Preserve every unrelated non-blocking step, the campaign-target root, and the complete target-to-terminal-leaf synthesis boundary. Never recommend improving the blocking fraction by truncating the route, deleting its unresolved suffix, or promoting an unavailable advanced intermediate to a terminal starting material.",
            "When a proposed fragment union lacks complementary reactive handles, reject that serialized step and propose installation or use of explicit compatible handles at the same advanced-intermediate boundary; conditions cannot rescue a graph-disconnected coupling.",
            "This critique grants no reaction proof, source authority, stock authority, or solved status.",
            "BlindRouteCriticInput:",
            json.dumps(route, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
    )


def _bounded_critic_prompt(
    *,
    target: str,
    branch_index: int,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
    strategy_milestone_cards: Iterable[Mapping[str, Any]] = (),
    maximum_bytes: int,
    paper_matched: bool = False,
    audit_kind: str = "final_route",
    focus_step_id: str = "",
) -> str | None:
    """Build a Critic prompt without dropping route topology.

    The former fallback kept only the last eight steps and could still exceed
    the byte cap when a biological contract was verbose.  Worse, the late
    ``ValueError`` discarded every durable branch.  Progressive compaction now
    retains every step's exact product, precursors, edit program, execution
    domain, and strategic anchor while removing prose and condition detail.
    """

    route_steps = [dict(step) for step in steps if isinstance(step, Mapping)]
    milestone_cards = [
        dict(card)
        for card in strategy_milestone_cards
        if isinstance(card, Mapping)
    ]
    for compact_level in range(3):
        prompt = _critic_prompt(
            target=target,
            branch_index=branch_index,
            strategy_card=strategy_card,
            strategy_milestone_cards=milestone_cards,
            steps=route_steps,
            compact_level=compact_level,
            paper_matched=paper_matched,
            audit_kind=audit_kind,
            focus_step_id=focus_step_id,
        )
        if len(prompt.encode("utf-8")) <= maximum_bytes:
            return prompt
    return None


def _critic_strategy_card(
    value: Mapping[str, Any],
    *,
    compact_level: int,
) -> dict[str, Any]:
    card = dict(value)
    if compact_level <= 0:
        return card
    keys = {
        "strategy_id",
        "strategy_milestone_index",
        "execution_domain",
        "strategy_query",
        "key_bond_changes",
        "key_bond_signature",
        "strategy_signature",
        "key_forward_transformation",
        "biocatalytic_intent",
    }
    compact = {key: card.get(key) for key in keys if card.get(key) not in (None, "", [], {})}
    if compact_level >= 2:
        compact.pop("strategy_query", None)
        compact.pop("key_forward_transformation", None)
        compact.pop("strategy_signature", None)
    intent = compact.get("biocatalytic_intent")
    if isinstance(intent, Mapping):
        keep = {
            "mode",
            "enzyme_classes",
            "ec_numbers",
            "candidate_ids",
            "whole_cell_hosts",
            "cofactor_assessment",
            "selectivity_objective",
        }
        compact["biocatalytic_intent"] = {
            key: intent.get(key)
            for key in keep
            if intent.get(key) not in (None, "", [], {})
        }
        if compact_level >= 2:
            compact["biocatalytic_intent"].pop("selectivity_objective", None)
    return compact


def _critic_step_row(
    step: Mapping[str, Any],
    *,
    compact_level: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step_id": str(step.get("step_id") or ""),
        "product_smiles": str(step.get("product_smiles") or ""),
        "precursor_smiles": list(step.get("precursor_smiles") or []),
        "mapped_product_smiles": str(step.get("mapped_product_smiles") or ""),
        "mapped_precursor_smiles": list(
            step.get("mapped_precursor_smiles") or []
        ),
        "reaction_operations": [
            dict(value)
            for value in step.get("reaction_operations") or []
            if isinstance(value, Mapping)
        ],
        "execution_domain": str(step.get("execution_domain") or "chemical"),
        "strategy_anchor": step.get("strategy_anchor") is True,
        "strategy_milestone_index": int(
            step.get("strategy_milestone_index") or 1
        ),
        "strategy_id": str(step.get("strategy_id") or ""),
    }
    if compact_level <= 1:
        row["transformation_hypothesis"] = str(
            step.get("transformation_hypothesis") or ""
        )[:768]
    predictions = [
        dict(value)
        for value in step.get("condition_predictions") or []
        if isinstance(value, Mapping)
    ]
    if compact_level == 0:
        row["condition_predictions"] = predictions
        row["biocatalytic_step"] = dict(step.get("biocatalytic_step") or {})
        row["biocatalytic_design_deficits"] = list(
            step.get("biocatalytic_design_deficits") or []
        )
    elif compact_level == 1:
        if predictions:
            prediction = predictions[0]
            allowed = {
                "conditions",
                "reagents",
                "catalyst",
                "solvent",
                "temperature",
                "time",
            }
            row["condition_prediction"] = {
                key: prediction.get(key)
                for key in allowed
                if prediction.get(key) not in (None, "", [], {})
            }
        bio = dict(step.get("biocatalytic_step") or {})
        if bio:
            allowed_bio = {
                "mode",
                "enzyme_classes",
                "ec_numbers",
                "candidate_ids",
                "whole_cell_hosts",
                "cofactor_assessment",
                "selectivity_objective",
            }
            row["biocatalytic_step"] = {
                key: bio.get(key)
                for key in allowed_bio
                if bio.get(key) not in (None, "", [], {})
            }
    elif predictions:
        # Structural maps and minimal conditions are both execution inputs;
        # even the strongest prompt compaction must not turn them into an
        # information-free uncertainty verdict.
        row["conditions"] = [
            str(reagent)[:240]
            for prediction in predictions[:1]
            for reagent in prediction.get("reagents") or prediction.get("conditions") or []
            if str(reagent)
        ][:8]
        row["catalyst"] = str(predictions[0].get("catalyst") or "")[:160]
    return row


def _paper_critic_step_row(
    step: Mapping[str, Any],
    *,
    compact_level: int,
) -> dict[str, Any]:
    """Project only replay and chemistry facts into the paper Critic."""

    predictions = [
        dict(value)
        for value in step.get("condition_predictions") or []
        if isinstance(value, Mapping)
    ]
    conditions = [
        str(value)[:240]
        for value in (
            step.get("conditions")
            or [
                reagent
                for prediction in predictions[:1]
                for reagent in prediction.get("reagents")
                or prediction.get("conditions")
                or []
            ]
        )
        if str(value)
    ][: (4 if compact_level == 0 else 2)]
    row = {
        "step_id": str(step.get("step_id") or ""),
        "mapped_product_smiles": str(step.get("mapped_product_smiles") or ""),
        "mapped_precursor_smiles": list(
            step.get("mapped_precursor_smiles") or []
        ),
        "reaction_operations": [
            dict(value)
            for value in step.get("reaction_operations") or []
            if isinstance(value, Mapping)
        ],
        "reaction_family": str(
            step.get("reaction_family")
            or step.get("transformation_hypothesis")
            or ""
        )[:160],
    }
    # checkpoint_relation is a Builder scheduling claim, not chemical
    # evidence.  The Host already selects focus_step_id for the sparse audit;
    # hiding the label prevents it from biasing either Critic.  Optional
    # condition fields are serialized only when they carry information.
    if conditions:
        row["conditions"] = conditions
    catalyst = str(
        step.get("catalyst")
        or next(
            (
                prediction.get("catalyst") or ""
                for prediction in predictions
            ),
            "",
        )
    )[:160]
    if catalyst:
        row["catalyst"] = catalyst
    return row


def _critic_task(
    spec: AgentSpec,
    *,
    prompt: str,
    branch_index: int,
    iteration: int,
    timeout_s: float,
    paper_matched: bool = False,
    target_smiles: str = "",
    audit_kind: str = "final_route",
    focus_step_id: str = "",
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
                    + ":"
                    + str(audit_kind)
                    + ":"
                    + str(focus_step_id)
                ).encode("utf-8")
            ).hexdigest()[:20]
        ),
        case_id=_opaque_strategy_case_id(spec.run_id + ":critic"),
        task_type=(
            "paper_matched_key_event_critic"
            if paper_matched and audit_kind == "key_event"
            else "paper_matched_route_critic"
            if paper_matched
            else "route_chemistry_critique"
        ),
        required_artifact_type="ChemicalStrategyCritique",
        input_refs=[],
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=(32_000 if paper_matched else 48_000),
            max_tool_calls=None,
            max_worker_runs=1,
            reasoning_effort=str(
                spec.metadata.get("critic_reasoning_effort")
                or spec.metadata.get("reasoning_effort")
                or "medium"
            ),
        ),
        objective=prompt,
        allowed_workdir=str(spec.metadata.get("allowed_workdir") or "."),
        agent_mode="single",
        codex_auth_mode="ambient_codex_cli",
        model=str(spec.metadata.get("model") or ""),
        host_context={
            "target_smiles": str(target_smiles or ""),
            "focus_step_id": str(focus_step_id or ""),
        },
    )


def _preflight_paper_matched_worker_schemas(
    spec: AgentSpec,
    *,
    target: str,
) -> None:
    """Validate every paper Worker schema before the first paid call.

    The representative tasks are built by the same constructors used during
    execution.  The preflight then compiles schemas through the Worker's
    canonical schema function, so this does not create a second schema
    contract.
    """

    model = str(spec.metadata.get("model") or "")
    default_effort = str(spec.metadata.get("reasoning_effort") or "medium")
    tasks = (
        _strategy_portfolio_task(
            spec,
            prompt="provider schema preflight",
            model=model,
            reasoning_effort=str(
                spec.metadata.get("strategy_reasoning_effort") or default_effort
            ),
            timeout_s=1.0,
            target_smiles=target,
        ),
        _strategy_portfolio_critic_task(
            spec,
            prompt="provider schema preflight",
            model=model,
            reasoning_effort=str(
                spec.metadata.get("critic_reasoning_effort")
                or spec.metadata.get("strategy_reasoning_effort")
                or default_effort
            ),
            timeout_s=1.0,
            target_smiles=target,
        ),
        _strategy_task(
            spec,
            prompt="provider schema preflight",
            branch_index=0,
            attempt_index=0,
            model=model,
            reasoning_effort=str(
                spec.metadata.get("strategy_reasoning_effort") or default_effort
            ),
            timeout_s=1.0,
            paper_matched=True,
            target_smiles=target,
        ),
        _node_task(
            spec,
            prompt="provider schema preflight",
            branch_index=0,
            node_index=0,
            model=model,
            reasoning_effort=default_effort,
            timeout_s=1.0,
            task_type="route_step_materialization",
            paper_matched=True,
            target_smiles=target,
            selected_product=target,
        ),
        _critic_task(
            spec,
            prompt="provider schema preflight",
            branch_index=0,
            iteration=0,
            timeout_s=1.0,
            paper_matched=True,
            target_smiles=target,
        ),
        _critic_task(
            spec,
            prompt="provider schema preflight",
            branch_index=0,
            iteration=0,
            timeout_s=1.0,
            paper_matched=True,
            target_smiles=target,
            audit_kind="key_event",
            focus_step_id="preflight-step",
        ),
        _node_task(
            spec,
            prompt="provider schema preflight",
            branch_index=0,
            node_index=0,
            model=model,
            reasoning_effort=default_effort,
            timeout_s=1.0,
            task_type="route_chemistry_edit",
            paper_matched=True,
            target_smiles=target,
            selected_product=target,
        ),
    )
    preflight_worker_response_schemas(tasks)


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


def _blocking_critic_steps(
    critique: Mapping[str, Any],
    steps: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return every concrete blocking route step for one Editor pass.

    The Critic is allowed to mark unresolved scope as ``uncertain``.  Only a
    concrete ``reject`` verdict triggers editing.  SynthEx's Editor receives
    the route-level critique, so withholding all but the first blocker creates
    an artificial local-repair bottleneck and prevents coordinated edits.
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
    blockers: list[dict[str, Any]] = []
    seen_step_ids: set[str] = set()
    for assessment_index, assessment in enumerate(assessments):
        if (
            str(assessment.get("verdict") or "") != "reject"
            and assessment.get("blocking") is not True
        ):
            continue
        step = by_id.get(str(assessment.get("step_id") or ""))
        # Codex may shorten an opaque host step id after route replay/editing.
        # The critic prompt preserves route order, so use an ordinal fallback
        # only when assessment and route cardinalities agree.
        if step is None and len(assessments) == len(steps):
            step = dict(steps[assessment_index])
        if step is None:
            continue
        step_id = str(step.get("step_id") or "")
        if step_id and step_id in seen_step_ids:
            continue
        if step_id:
            seen_step_ids.add(step_id)
        step["reasons"] = [
            str(value)
            for value in assessment.get("reasons") or []
            if str(value)
        ]
        step["critic_assessment"] = dict(assessment)
        blockers.append(step)
    return blockers


def _rejected_net_edit_signatures(
    blocking_steps: Iterable[Mapping[str, Any]],
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Keep short structural negative memory for the next Editor call.

    This is diagnostic context, not a second chemistry validator.  The Critic
    owns the rejection; sorting the normalized operations only makes a renamed
    copy of the same net edit recognizable to the model on the next pass.
    """

    signatures: list[dict[str, Any]] = []
    for step in blocking_steps:
        operations = normalize_reaction_operations(
            step.get("reaction_operations") or ()
        )
        if not operations:
            continue
        assessment = dict(step.get("critic_assessment") or {})
        signatures.append(
            {
                "step_id": str(step.get("step_id") or "")[:160],
                "blocking_type": str(
                    assessment.get("blocking_type") or "unspecified"
                )[:80],
                "net_edits": sorted(
                    json.dumps(
                        dict(operation),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for operation in operations
                ),
            }
        )
        if len(signatures) >= max(0, int(limit)):
            break
    return signatures


def _compact_critic_feedback(
    critique: Mapping[str, Any],
    blocking_steps: Iterable[Mapping[str, Any]],
    *,
    paper_matched: bool = False,
) -> dict[str, Any]:
    """Project a large Critic artifact into an Editor-sized repair brief."""

    blocking_rows = [
        dict(value) for value in blocking_steps if isinstance(value, Mapping)
    ]

    def clip(value: Any, limit: int) -> str:
        return str(value or "").strip()[: max(1, int(limit))]

    def clip_list(
        values: Iterable[Any],
        *,
        count: int,
        limit: int,
    ) -> list[str]:
        return [
            text
            for text in (clip(value, limit) for value in values)
            if text
        ][: max(0, int(count))]

    critique_assessments = {
        str(value.get("step_id") or ""): dict(value)
        for value in critique.get("step_assessments") or []
        if isinstance(value, Mapping) and str(value.get("step_id") or "")
    }
    compact_blockers: list[dict[str, Any]] = []
    all_failure_reasons: list[str] = []
    for blocking_step in blocking_rows:
        step_id = str(blocking_step.get("step_id") or "")
        assessment = critique_assessments.get(step_id) or dict(
            blocking_step.get("critic_assessment") or {}
        )
        keep_assessment: dict[str, Any] = {}
        for key in (
            "step_id",
            "verdict",
            "blocking",
            "blocking_type",
            "condition_assessment",
            "suggested_revision",
            "enzyme_assessment",
        ):
            if assessment.get(key) not in (None, "", [], {}):
                keep_assessment[key] = clip(assessment.get(key), 480)
        if assessment.get("reasons"):
            keep_assessment["reasons"] = clip_list(
                assessment.get("reasons") or (),
                count=2,
                limit=320,
            )
        compact_step = {
            key: blocking_step.get(key)
            for key in (
                "step_id",
                "product_smiles",
                "precursor_smiles",
                "mapped_product_smiles",
                "mapped_precursor_smiles",
                "transformation_hypothesis",
                "reaction_operations",
                "condition_predictions",
                "strategy_anchor",
            )
            if blocking_step.get(key) not in (None, "", [], {})
        }
        if compact_step.get("transformation_hypothesis"):
            compact_step["transformation_hypothesis"] = clip(
                compact_step["transformation_hypothesis"],
                240,
            )
        reasons = clip_list(
            blocking_step.get("reasons") or (),
            count=2,
            limit=320,
        )
        all_failure_reasons.extend(reasons)
        compact_blockers.append(
            {
                "route_step": compact_step,
                "assessment": keep_assessment,
            }
        )
    route_level_risks = list(critique.get("route_level_risks") or ())
    if paper_matched:
        blocking_ids = {
            str(dict(value.get("assessment") or {}).get("step_id") or "")
            or str(dict(value.get("route_step") or {}).get("step_id") or "")
            for value in compact_blockers
            if isinstance(value, Mapping)
        }
        step_annotations: list[dict[str, Any]] = []
        for raw_assessment in critique.get("step_assessments") or ():
            if not isinstance(raw_assessment, Mapping):
                continue
            assessment = dict(raw_assessment)
            step_id = clip(assessment.get("step_id"), 160)
            verdict = clip(assessment.get("verdict"), 40)
            # The complete RouteJSON already carries every route row, while
            # blocking_steps below carries each full blocking assessment.
            # Repeating those rows and assessments here consumed a large
            # fraction of Editor input without adding authority. Keep only
            # concise non-blocking uncertainty annotations.
            if step_id in blocking_ids or verdict == "pass":
                continue
            annotation: dict[str, Any] = {
                "step_id": step_id,
                "verdict": verdict,
                "blocking": assessment.get("blocking") is True,
            }
            reasons = clip_list(
                assessment.get("reasons") or (),
                count=2,
                limit=320,
            )
            if reasons:
                annotation["reasons"] = reasons
            for key, limit in (
                ("condition_assessment", 320),
                ("suggested_revision", 400),
            ):
                text = clip(assessment.get(key), limit)
                if text:
                    annotation[key] = text
            step_annotations.append(annotation)
        paper_blockers = [
            {
                "step_id": str(
                    dict(value.get("assessment") or {}).get("step_id")
                    or dict(value.get("route_step") or {}).get("step_id")
                    or ""
                ),
                "assessment": dict(value.get("assessment") or {}),
            }
            for value in compact_blockers
        ]
        return {
            "overall_assessment": clip(
                critique.get("overall_assessment"),
                320,
            ),
            "strategy_adherence": critique.get("strategy_adherence"),
            "step_annotations": step_annotations,
            "blocking_steps": paper_blockers,
            "rejected_net_edit_signatures": _rejected_net_edit_signatures(
                blocking_rows,
                limit=2,
            ),
            "route_level_risks": clip_list(
                route_level_risks,
                count=4,
                limit=400,
            ),
        }
    primary = compact_blockers[0] if compact_blockers else {}
    repair_actions = list(critique.get("repair_actions") or ())
    experimental_variables = list(critique.get("experimental_variables") or ())
    return {
        "overall_assessment": str(critique.get("overall_assessment") or ""),
        "strategy_adherence": critique.get("strategy_adherence"),
        # Singular aliases remain for old prompt consumers, while the plural
        # fields are the paper-matched Editor authority.
        "blocking_step": dict(primary.get("route_step") or {}),
        "step_assessment": dict(primary.get("assessment") or {}),
        "blocking_steps": compact_blockers,
        "failure_reasons": list(dict.fromkeys(all_failure_reasons)),
        "repair_actions": clip_list(
            repair_actions,
            count=4,
            limit=500,
        ),
        "route_level_risks": clip_list(
            route_level_risks,
            count=4,
            limit=400,
        ),
        "experimental_variables": clip_list(
            experimental_variables,
            count=4,
            limit=320,
        ),
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


def _expansions_from_record(
    record: WorkerRunRecord,
    *,
    expected_product: str,
    require_strategy_card: bool = False,
    mapped_product_smiles: str = "",
    require_reaction_operations: bool = False,
    require_complete_route_json: bool = False,
    minimum_route_depth: int = 1,
    single_step_only: bool = False,
    compiler: RouteJSONCompiler | None = None,
    reserved_atom_maps: Iterable[int] = (),
) -> list[NodeExpansion] | None:
    # A worker can produce a structurally safe RouteJSON draft while the
    # generic worker envelope is rejected for metadata/runtime reasons (for
    # example missing provenance handles or a non-zero CLI exit after a
    # completed turn).  In explicit whole-route compatibility mode that draft
    # may still reach the host compiler/Editor; the compiler remains the sole
    # authority and all safety checks are performed by
    # _route_json_candidate().  Non-RouteJSON and artifact-less runtime
    # failures remain fail-closed.
    if record.status != "accepted_draft" and _route_json_candidate(record) is None:
        return None
    payload = dict(dict(record.output_artifact or {}).get("payload") or {})
    candidates = [dict(row) for row in payload.get("candidates") or [] if isinstance(row, Mapping)]
    if len(candidates) != 1:
        return None
    row = candidates[0]
    raw_route = row.get("route_json")
    if raw_route is None:
        if require_complete_route_json:
            return None
        route_rows: list[Mapping[str, Any]] = [row]
    elif isinstance(raw_route, list):
        # Compiler-first Route Builder calls intentionally consume exactly one
        # model edit program.  A legacy/full-route payload remains supported
        # by this helper for Editor and compatibility tests, but it is never
        # allowed to become the structure authority for the sequential host.
        if single_step_only:
            route_rows = [row]
        else:
            route_rows = [value for value in raw_route if isinstance(value, Mapping)]
        if not route_rows or (
            require_complete_route_json
            and len(route_rows) < max(1, int(minimum_route_depth))
        ):
            return None
    else:
        return None

    expansions: list[NodeExpansion] = []
    compiler = compiler or RouteJSONCompiler()
    prior_products: set[str] = set()
    prior_precursors: tuple[str, ...] = ()
    prior_mapped_precursors: tuple[str, ...] = ()
    for index, raw_step in enumerate(route_rows):
        step = dict(raw_step)
        # A legacy one-step candidate keeps its fields at the candidate root;
        # a complete RouteJSON step owns its own fields.
        declared_product = _canonical_smiles(step.get("product_smiles"))
        product = declared_product
        if index == 0 and product != _canonical_smiles(expected_product):
            return None
        mapped_product_for_step = mapped_product_smiles if index == 0 else ""
        if index > 0:
            # A later RouteJSON step must consume the exact host fragment.  A
            # model may redraw only stereochemistry; constitutionally equal
            # declarations are normalized to the host-derived mapped product
            # and retained as an auditable declaration mismatch.
            match = _match_editor_precursor(
                product,
                prior_precursors,
                prior_mapped_precursors,
            )
            if match is None:
                return None
            product, mapped_product_for_step = match
        if product in prior_products:
            return None
        declared_precursors = tuple(
            dict.fromkeys(
                canonical
                for value in step.get("precursor_smiles") or []
                if (canonical := _canonical_smiles(value))
            )
        )
        operations = normalize_reaction_operations(step.get("reaction_operations") or ())
        if index == 0 and not operations and step is not row:
            operations = normalize_reaction_operations(row.get("reaction_operations") or ())
        reactionjson_audit: dict[str, Any] = {}
        if declared_product != product:
            reactionjson_audit.update(
                {
                    "declared_product_smiles": declared_product,
                    "declared_product_matches_host": False,
                    "declared_product_mismatch_type": "stereochemistry_only",
                }
            )
        if operations:
            mapped_product = mapped_product_for_step
            if not mapped_product:
                return None
            try:
                materialized = compiler.compile_step(
                    mapped_product_smiles=mapped_product,
                    operations=operations,
                    expected_product_smiles=product,
                    reserved_atom_maps=reserved_atom_maps,
                )
            except ReactionJsonReplayError:
                return None
            product = materialized.product_smiles
            reactionjson_audit = {
                **reactionjson_audit,
                **dict(materialized.audit),
            }
            precursors = materialized.precursor_smiles
            mapped_precursors = materialized.mapped_precursor_smiles
        else:
            precursors = declared_precursors
            mapped_precursors = tuple(_mapped_smiles(value) for value in precursors)
        atom_provenance_deficit = _has_atom_provenance_deficit(product, precursors)
        unexplained_atom_provenance_deficit = (
            atom_provenance_deficit
            and not (
                reactionjson_audit.get("external_atom_source_required") is True
                and reactionjson_audit.get(
                    "external_atom_source_grants_reaction_proof"
                )
                is False
            )
        )
        if (
            not product
            or not precursors
            or len(precursors) > 4
            or product in precursors
            or any(precursor in prior_products for precursor in precursors)
            or unexplained_atom_provenance_deficit
        ):
            return None
        if require_reaction_operations and not operations:
            return None
        strategy_card = normalize_strategy_card(
            row.get("strategy_card") or {},
            reaction_operations=operations if require_strategy_card else (),
        )
        if require_strategy_card and not _valid_strategy_card(strategy_card):
            return None
        enzyme_label = str(_route_field(step, row, "enzyme") or "")
        raw_biocatalytic_step = _route_field(step, row, "biocatalytic_step")
        execution_domain = normalize_step_execution_domain(
            _route_field(step, row, "execution_domain"),
            enzyme_label=enzyme_label,
            biocatalytic_step=(
                raw_biocatalytic_step
                if isinstance(raw_biocatalytic_step, Mapping)
                else None
            ),
        )
        biocatalytic_step, _biocatalytic_reasons = normalize_biocatalytic_step(
            raw_biocatalytic_step
            if isinstance(raw_biocatalytic_step, Mapping)
            else None,
            execution_domain=execution_domain,
            product_smiles=product,
            precursor_smiles=precursors,
            enzyme_label=enzyme_label,
            step_id=str(step.get("step_id") or ""),
        )
        expansions.append(
            NodeExpansion(
                product_smiles=product,
                precursor_smiles=precursors,
                reaction_family=str(_route_field(step, row, "reaction_family") or "retrosynthetic transformation"),
                rationale=str(_route_field(step, row, "transformation_rationale") or "model-proposed local disconnection"),
                step_role=_normalize_step_role(
                    _route_field(step, row, "step_role")
                ),
                checkpoint_relation=_normalize_checkpoint_relation(
                    _route_field(step, row, "checkpoint_relation")
                ),
                mapped_product_smiles=(
                    str(mapped_product_for_step or _mapped_smiles(product))
                ),
                mapped_precursor_smiles=mapped_precursors,
                conditions=tuple(
                    str(value)
                    for value in (_route_field(step, row, "conditions") or [])
                    if str(value)
                ),
                catalyst=str(_route_field(step, row, "catalyst") or ""),
                enzyme=enzyme_label,
                execution_domain=execution_domain,
                biocatalytic_step=biocatalytic_step,
                limitations=tuple(
                    str(value)
                    for value in (_route_field(step, row, "limitations") or [])
                    if str(value)
                ),
                product_retron_type=str(_route_field(step, row, "product_retron_type") or ""),
                strategy_card=strategy_card,
                reaction_operations=operations,
                reactionjson_audit=reactionjson_audit,
                step_id=str(step.get("step_id") or ""),
            )
        )
        prior_products.add(product)
        prior_precursors = precursors
        prior_mapped_precursors = mapped_precursors
    return expansions


def _reactionjson_candidates_from_record(
    record: WorkerRunRecord,
    *,
    expected_product: str,
    mapped_product_smiles: str,
    require_reaction_operations: bool,
    compiler: RouteJSONCompiler | None = None,
    max_candidates: int = 3,
    reserved_atom_maps: Iterable[int] = (),
) -> tuple[list[_CompiledReactionJsonCandidate], list[dict[str, Any]]]:
    """Compile local candidates independently so one invalid edit loses only itself."""

    if record.status != "accepted_draft":
        return [], [{"reason": "worker_output_not_accepted"}]
    artifact = dict(record.output_artifact or {})
    payload = dict(artifact.get("payload") or {})
    raw_candidates = [
        dict(value)
        for value in payload.get("candidates") or []
        if isinstance(value, Mapping)
    ]
    if not raw_candidates:
        return [], [{"reason": "candidate_count_invalid", "candidate_count": 0}]

    accepted: list[_CompiledReactionJsonCandidate] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    compiler = compiler or RouteJSONCompiler()
    for candidate_index, candidate in enumerate(
        raw_candidates[: max(1, int(max_candidates))]
    ):
        candidate_id = str(
            candidate.get("candidate_id")
            or f"{record.task_id}:candidate:{candidate_index + 1}"
        )
        single_payload = {**payload, "candidates": [candidate]}
        single_artifact = {**artifact, "payload": single_payload}
        single_record = replace(record, output_artifact=single_artifact)
        expansions = _expansions_from_record(
            single_record,
            expected_product=expected_product,
            mapped_product_smiles=mapped_product_smiles,
            require_reaction_operations=require_reaction_operations,
            require_complete_route_json=False,
            minimum_route_depth=1,
            single_step_only=True,
            compiler=compiler,
            reserved_atom_maps=reserved_atom_maps,
        )
        if expansions is None or len(expansions) != 1:
            rejected.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    **_expansion_rejection_diagnostic(
                        single_record,
                        expected_product=expected_product,
                        mapped_product_smiles=mapped_product_smiles,
                        require_reaction_operations=require_reaction_operations,
                        require_complete_route_json=False,
                        minimum_route_depth=1,
                        single_step_only=True,
                        reserved_atom_maps=reserved_atom_maps,
                    ),
                }
            )
            continue
        expansion = expansions[0]
        candidate_key = hashlib.sha256(
            json.dumps(
                {
                    "product_smiles": expansion.product_smiles,
                    "precursor_smiles": sorted(expansion.precursor_smiles),
                    "reaction_edit_digest": reaction_edit_digest(
                        expansion.reaction_operations
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if candidate_key in seen_keys:
            rejected.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "reason": "duplicate_reactionjson_candidate",
                    "candidate_key": candidate_key,
                }
            )
            continue
        seen_keys.add(candidate_key)
        explicit_score = candidate.get("prior_score")
        if explicit_score is None:
            explicit_score = candidate.get("score")
        score, cost = ranked_candidate_cost(candidate_index, explicit_score)
        accepted.append(
            _CompiledReactionJsonCandidate(
                candidate_index=candidate_index,
                candidate_id=candidate_id,
                expansion=expansion,
                score=score,
                cost=cost,
                candidate_key=candidate_key,
            )
        )
    if len(raw_candidates) > max(1, int(max_candidates)):
        rejected.append(
            {
                "reason": "candidate_limit_truncated",
                "candidate_count": len(raw_candidates),
                "accepted_prefix_limit": max(1, int(max_candidates)),
            }
        )
    return accepted, rejected


def _route_json_candidate(record: WorkerRunRecord) -> dict[str, Any] | None:
    """Extract one safe RouteJSON draft without granting it authority.

    The generic artifact validator may reject a blind RouteJSON draft for
    metadata-only reasons (notably empty precursor lists or absent provenance
    handles).  Such a draft must still reach the host compiler so the Editor
    receives the actual structural diagnostic.  Safety claims and raw reaction
    injection remain fail-closed here.
    """

    if record.status not in {"accepted_draft", "rejected_output"}:
        return None
    artifact = dict(record.output_artifact or {})
    if artifact.get("artifact_type") != "RetrosynthesisProposalReport":
        return None
    payload = dict(artifact.get("payload") or {})
    if payload.get("schema_version") != "retrosynthesis_proposal_report.v1":
        return None
    if payload.get("no_solved_claim") is not True or _route_draft_has_solved_claim(payload):
        return None
    if contains_raw_reaction_payload(payload):
        return None
    candidates = [
        dict(value)
        for value in payload.get("candidates") or []
        if isinstance(value, Mapping)
    ]
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    repair_status = str(candidate.get("repair_status") or "").strip().lower()
    if repair_status and repair_status != "revised":
        return None
    if candidate.get("no_solved_claim") is not True or candidate.get(
        "not_parent_route_proof"
    ) is not True:
        return None
    if contains_raw_reaction_payload(candidate):
        return None
    has_route = (
        isinstance(candidate.get("route_json"), list)
        and bool(candidate.get("route_json"))
    ) or (
        isinstance(candidate.get("route_patch"), list)
        and bool(candidate.get("route_patch"))
    ) or (
        isinstance(candidate.get("replace_span"), Mapping)
        and bool(dict(candidate["replace_span"]).get("remove_step_ids"))
        and bool(dict(candidate["replace_span"]).get("revised_steps"))
    )
    return candidate if has_route else None


def _route_draft_has_solved_claim(value: Any) -> bool:
    """Detect solved/status claims in a RouteJSON draft before host replay."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"solved", "is_solved"} and item is True:
                return True
            if key_text in {"verdict", "route_status", "status"} and str(
                item or ""
            ).strip().lower() == "solved":
                return True
            if _route_draft_has_solved_claim(item):
                return True
        return False
    if isinstance(value, list):
        return any(_route_draft_has_solved_claim(item) for item in value)
    return False


def _route_json_has_repairable_draft(record: WorkerRunRecord) -> bool:
    candidate = _route_json_candidate(record)
    if not candidate:
        return False
    route = candidate.get("route_json")
    patch = candidate.get("route_patch")
    span = candidate.get("replace_span")
    return (isinstance(route, list) and bool(route)) or (
        isinstance(patch, list) and bool(patch)
    ) or (
        isinstance(span, Mapping)
        and bool(dict(span).get("remove_step_ids"))
        and bool(dict(span).get("revised_steps"))
    )


def _route_json_failure_is_editor_repairable(diagnostic: Mapping[str, Any]) -> bool:
    return str(diagnostic.get("reason") or "") in {
        "route_json_invalid",
        "route_json_incomplete",
        "route_json_step_invalid",
        "route_json_chain_invalid",
        "route_json_step_reaction_operations_missing",
        "route_json_step_replay_failed",
    }


def _route_field(
    step: Mapping[str, Any],
    row: Mapping[str, Any],
    key: str,
) -> Any:
    """Read a RouteJSON field without turning an explicit empty value into a fallback."""

    return step[key] if key in step else row.get(key)


def _expansion_from_materialized(
    materialized: MaterializedReaction,
    metadata: Mapping[str, Any],
) -> NodeExpansion:
    row = dict(metadata)
    enzyme_label = str(
        row.get("enzyme")
        or next(
            (
                prediction.get("enzyme") or ""
                for prediction in row.get("condition_predictions") or []
                if isinstance(prediction, Mapping)
            ),
            "",
        )
    )
    raw_biocatalytic_step = row.get("biocatalytic_step")
    execution_domain = normalize_step_execution_domain(
        row.get("execution_domain"),
        enzyme_label=enzyme_label,
        biocatalytic_step=(
            raw_biocatalytic_step
            if isinstance(raw_biocatalytic_step, Mapping)
            else None
        ),
    )
    biocatalytic_step, _biocatalytic_reasons = normalize_biocatalytic_step(
        raw_biocatalytic_step
        if isinstance(raw_biocatalytic_step, Mapping)
        else None,
        execution_domain=execution_domain,
        product_smiles=materialized.product_smiles,
        precursor_smiles=materialized.precursor_smiles,
        enzyme_label=enzyme_label,
        step_id=str(row.get("step_id") or ""),
    )
    return NodeExpansion(
        product_smiles=materialized.product_smiles,
        precursor_smiles=materialized.precursor_smiles,
        reaction_family=str(
            row.get("reaction_family")
            or row.get("transformation_hypothesis")
            or "retrosynthetic transformation"
        ),
        rationale=str(
            row.get("transformation_rationale")
            or row.get("strategic_role")
            or "host-compiled model edit program"
        ),
        step_role=_normalize_step_role(row.get("step_role")),
        checkpoint_relation=_normalize_checkpoint_relation(
            row.get("checkpoint_relation")
        ),
        mapped_product_smiles=materialized.mapped_product_smiles,
        mapped_precursor_smiles=materialized.mapped_precursor_smiles,
        conditions=tuple(
            str(value)
            for value in (
                row.get("conditions")
                or next(
                    (
                        prediction.get("reagents") or []
                        for prediction in row.get("condition_predictions") or []
                        if isinstance(prediction, Mapping)
                    ),
                    [],
                )
            )
            if str(value)
        ),
        catalyst=str(
            row.get("catalyst")
            or next(
                (
                    prediction.get("catalyst") or ""
                    for prediction in row.get("condition_predictions") or []
                    if isinstance(prediction, Mapping)
                ),
                "",
            )
        ),
        enzyme=enzyme_label,
        execution_domain=execution_domain,
        biocatalytic_step=biocatalytic_step,
        limitations=tuple(
            str(value) for value in row.get("limitations") or [] if str(value)
        ),
        product_retron_type=str(row.get("product_retron_type") or ""),
        reaction_operations=materialized.reaction_operations,
        reactionjson_audit=materialized.audit,
        step_id=str(row.get("step_id") or ""),
    )


def _compile_editor_route_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    mapped_target_smiles: str,
    expected_target_smiles: str,
) -> list[NodeExpansion] | None:
    expansions, _ = _compile_editor_route_rows_with_diagnostic(
        rows,
        mapped_target_smiles=mapped_target_smiles,
        expected_target_smiles=expected_target_smiles,
    )
    return expansions


def _compile_editor_route_rows_with_diagnostic(
    rows: Iterable[Mapping[str, Any]],
    *,
    mapped_target_smiles: str,
    expected_target_smiles: str,
) -> tuple[list[NodeExpansion] | None, dict[str, Any]]:
    route_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not route_rows:
        return None, {
            "reason": "route_json_invalid",
            "detail": "route_json_must_be_a_non_empty_list",
        }
    first_product = _canonical_smiles(route_rows[0].get("product_smiles"))
    if first_product != _canonical_smiles(expected_target_smiles):
        return None, {
            "reason": "route_json_chain_invalid",
            "step_index": 0,
            "detail": "first_product_mismatch",
            "declared_product_smiles": first_product,
            "expected_product_smiles": _canonical_smiles(expected_target_smiles),
        }
    for index, row in enumerate(route_rows):
        if not _canonical_smiles(row.get("product_smiles")):
            return None, {
                "reason": "route_json_step_invalid",
                "step_index": index,
                "detail": "product_smiles_invalid",
            }
        if not normalize_reaction_operations(row.get("reaction_operations") or ()):
            return None, {
                "reason": "route_json_step_reaction_operations_missing",
                "step_index": index,
                "product_smiles": _canonical_smiles(row.get("product_smiles")),
            }
    try:
        compiled = RouteJSONCompiler().compile_route_graph(
            mapped_target_smiles=mapped_target_smiles,
            steps=route_rows,
            minimum_depth=1,
        )
    except ReactionJsonReplayError as exc:
        compiler_error = str(exc)
        failed_index = 0
        for prefix_size in range(1, len(route_rows) + 1):
            try:
                RouteJSONCompiler().compile_route_graph(
                    mapped_target_smiles=mapped_target_smiles,
                    steps=route_rows[:prefix_size],
                    minimum_depth=1,
                )
            except ReactionJsonReplayError:
                failed_index = prefix_size - 1
                break
        host_boundary = _editor_host_boundary_after_prefix(
            route_rows,
            failed_index=failed_index,
            mapped_target_smiles=mapped_target_smiles,
            expected_target_smiles=expected_target_smiles,
        )
        reason = (
            "route_json_chain_invalid"
            if any(
                marker in compiler_error
                for marker in (
                    "chain_product_mismatch",
                    "product_not_open_precursor",
                    "ambiguous_parent_map_namespace",
                    "product_cycle",
                )
            )
            else "route_json_step_replay_failed"
        )
        return None, {
            "reason": reason,
            "step_index": failed_index,
            "failed_step_id": str(route_rows[failed_index].get("step_id") or ""),
            "product_smiles": _canonical_smiles(
                route_rows[failed_index].get("product_smiles")
            ),
            "compiler_error": compiler_error,
            "compiler_mode": "target_rooted_route_dag",
            "detail": (
                "product_not_in_open_precursors"
                if "product_not_open_precursor" in compiler_error
                else compiler_error
            ),
            **reactionjson_failure_focus(exc),
            **host_boundary,
        }
    if not compiled or compiled[0].product_smiles != _canonical_smiles(
        expected_target_smiles
    ):
        return None, {
            "reason": "route_json_chain_invalid",
            "detail": "compiled_target_product_mismatch",
        }
    return (
        [
            _expansion_from_materialized(materialized, route_rows[index])
            for index, materialized in enumerate(compiled)
        ],
        {},
    )


def _editor_host_boundary_after_prefix(
    route_rows: Iterable[Mapping[str, Any]],
    *,
    failed_index: int,
    mapped_target_smiles: str,
    expected_target_smiles: str,
) -> dict[str, Any]:
    """Return the real mapped frontier at the first failed Editor row.

    A complete Editor draft is compiled from the target, not from its declared
    ``mapped_product_smiles`` fields.  Once a prefix replays successfully, the
    only valid map namespace for the next row is the frontier emitted by that
    replay.  This diagnostic is fed directly into the bounded Editor retry.
    """

    rows = [dict(row) for row in route_rows if isinstance(row, Mapping)]
    index = max(0, min(int(failed_index), len(rows)))
    if index == 0:
        target = _canonical_smiles(expected_target_smiles or mapped_target_smiles)
        frontier = [
            {
                "product_smiles": target,
                "mapped_product_smiles": str(mapped_target_smiles or ""),
            }
        ] if target and str(mapped_target_smiles or "") else []
    else:
        try:
            state = RouteJSONCompiler().compile_route_graph_state(
                mapped_target_smiles=mapped_target_smiles,
                steps=rows[:index],
                minimum_depth=1,
            )
        except ReactionJsonReplayError:
            return {
                "host_replayed_prefix_step_count": 0,
                "host_open_precursors": [],
                "mapped_open_precursor_authority": "host_routejson_dag_compiler",
            }
        frontier = [
            {
                "product_smiles": value.product_smiles,
                "mapped_product_smiles": value.mapped_product_smiles,
            }
            for value in state.open_precursors
        ]

    selected: tuple[str, str] | None = None
    if index < len(rows):
        failed_row = rows[index]
        selected = _match_editor_precursor(
            _canonical_smiles(failed_row.get("product_smiles")),
            (value["product_smiles"] for value in frontier),
            (value["mapped_product_smiles"] for value in frontier),
        )
        if selected is None:
            selected = _match_editor_precursor_by_operation_maps(
                (value["product_smiles"] for value in frontier),
                (value["mapped_product_smiles"] for value in frontier),
                normalize_reaction_operations(
                    failed_row.get("reaction_operations") or ()
                ),
            )

    result: dict[str, Any] = {
        "host_replayed_prefix_step_count": index,
        "host_open_precursors": frontier,
        "mapped_open_precursor_authority": "host_routejson_dag_compiler",
    }
    if selected is not None:
        result["host_selected_open_precursor"] = {
            "product_smiles": selected[0],
            "mapped_product_smiles": selected[1],
        }
    return result


def _editor_retry_route_rows(
    record: WorkerRunRecord,
    *,
    diagnostic: Mapping[str, Any],
    mapped_target_smiles: str,
) -> list[dict[str, Any]] | None:
    """Carry a failed Editor draft forward with a host-replayed prefix.

    The next Editor attempt repairs the previous draft instead of regenerating
    the original route.  Every successful prefix row is serialized from the
    compiler, and the failed row is rebound to the exact selected mapped open
    precursor when one can be identified.
    """

    candidate = _route_json_candidate(record)
    raw_route = candidate.get("route_json") if candidate else None
    if not isinstance(raw_route, list) or not raw_route:
        return None
    rows = [dict(row) for row in raw_route if isinstance(row, Mapping)]
    if len(rows) != len(raw_route):
        return None
    try:
        failed_index = int(diagnostic.get("step_index"))
    except (TypeError, ValueError):
        return rows
    failed_index = max(0, min(failed_index, len(rows) - 1))

    retry_rows = [dict(row) for row in rows]
    if failed_index > 0:
        try:
            state = RouteJSONCompiler().compile_route_graph_state(
                mapped_target_smiles=mapped_target_smiles,
                steps=rows[:failed_index],
                minimum_depth=1,
            )
        except ReactionJsonReplayError:
            state = None
        if state is not None:
            retry_rows[:failed_index] = RouteJSONCompiler.assemble_route(
                state.reactions,
                metadata=rows[:failed_index],
            )

    selected = diagnostic.get("host_selected_open_precursor")
    if isinstance(selected, Mapping):
        product = _canonical_smiles(selected.get("product_smiles"))
        mapped = str(selected.get("mapped_product_smiles") or "").strip()
        if product and mapped:
            retry_rows[failed_index]["product_smiles"] = product
            retry_rows[failed_index]["mapped_product_smiles"] = mapped
            # These are outputs of the still-failing row and therefore remain
            # model draft data until that row successfully replays.
            retry_rows[failed_index]["precursor_smiles"] = []
            retry_rows[failed_index]["mapped_precursor_smiles"] = []
    return retry_rows


def _bind_editor_failed_row_to_host_boundary(
    route_rows: Iterable[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind one failed working-draft row to the Host's exact mapped frontier."""

    rows = [dict(row) for row in route_rows if isinstance(row, Mapping)]
    failed_step_id = str(diagnostic.get("failed_step_id") or "")
    failed_index = next(
        (
            index
            for index, row in enumerate(rows)
            if failed_step_id
            and str(row.get("step_id") or "") == failed_step_id
        ),
        -1,
    )
    if failed_index < 0:
        try:
            failed_index = int(diagnostic.get("step_index"))
        except (TypeError, ValueError):
            return rows
    if failed_index < 0 or failed_index >= len(rows):
        return rows
    selected = diagnostic.get("host_selected_open_precursor")
    if not isinstance(selected, Mapping):
        return rows
    product = _canonical_smiles(selected.get("product_smiles"))
    mapped = str(selected.get("mapped_product_smiles") or "").strip()
    if not product or not mapped:
        return rows
    rows[failed_index]["product_smiles"] = product
    rows[failed_index]["mapped_product_smiles"] = mapped
    rows[failed_index]["precursor_smiles"] = []
    rows[failed_index]["mapped_precursor_smiles"] = []
    return rows


def _apply_replace_span(
    current_steps: Iterable[Mapping[str, Any]],
    replace_span: Mapping[str, Any],
) -> tuple[list[dict[str, Any]] | None, str]:
    """Merge one Editor-authored replacement span into the Host route.

    The model selects old rows by stable id and authors only their replacement.
    Entry/exit boundaries are deliberately not model fields: the existing
    target-rooted DAG compiler derives and verifies them after this mechanical
    merge.  This keeps one topology authority while allowing a repair to cover
    one row, multiple dependent rows, or the whole route.
    """

    rows = [
        _compact_route_spec(row)
        for row in current_steps
        if isinstance(row, Mapping)
    ]
    if not rows:
        return None, "editor_replace_span_current_route_empty"

    current_ids = [str(row.get("step_id") or "") for row in rows]
    if any(not value for value in current_ids) or len(set(current_ids)) != len(
        current_ids
    ):
        return None, "editor_replace_span_current_step_ids_invalid"

    remove_step_ids = [
        str(value)
        for value in replace_span.get("remove_step_ids") or []
        if str(value)
    ]
    if not remove_step_ids:
        return None, "editor_replace_span_remove_step_ids_missing"
    if len(set(remove_step_ids)) != len(remove_step_ids):
        return None, "editor_replace_span_remove_step_ids_duplicate"
    unknown = [value for value in remove_step_ids if value not in set(current_ids)]
    if unknown:
        return None, "editor_replace_span_step_not_found"

    raw_revised = replace_span.get("revised_steps")
    if not isinstance(raw_revised, list) or not raw_revised:
        return None, "editor_replace_span_revised_steps_missing"
    if any(not isinstance(value, Mapping) for value in raw_revised):
        return None, "editor_replace_span_revised_step_invalid"
    revised = [_compact_route_spec(value) for value in raw_revised]
    revised_ids = [str(row.get("step_id") or "") for row in revised]
    if any(not value for value in revised_ids) or len(set(revised_ids)) != len(
        revised_ids
    ):
        return None, "editor_replace_span_revised_step_ids_invalid"

    removed = set(remove_step_ids)
    retained_ids = {value for value in current_ids if value not in removed}
    if retained_ids.intersection(revised_ids):
        return None, "editor_replace_span_step_id_collision"

    first_removed_index = min(current_ids.index(value) for value in remove_step_ids)
    before = [
        row
        for index, row in enumerate(rows)
        if index < first_removed_index
        and str(row.get("step_id") or "") not in removed
    ]
    after = [
        row
        for index, row in enumerate(rows)
        if index > first_removed_index
        and str(row.get("step_id") or "") not in removed
    ]
    merged = [*before, *revised, *after]
    return (merged, "") if merged else (
        None,
        "editor_replace_span_route_empty",
    )


def _apply_route_patch(
    current_steps: Iterable[Mapping[str, Any]],
    patch_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str]:
    """Apply an Editor patch to route specifications, never to compiled graphs."""

    rows = [_compact_route_spec(row) for row in current_steps if isinstance(row, Mapping)]
    if not rows:
        return None, "editor_patch_current_route_empty"
    for raw in patch_rows:
        patch = dict(raw)
        op = str(patch.get("op") or "").strip().lower()
        step_id = str(patch.get("step_id") or "")
        after_step_id = str(patch.get("after_step_id") or "")
        step = patch.get("step")
        if op == "replace_step":
            if not isinstance(step, Mapping):
                return None, "editor_patch_replacement_missing"
            index = next(
                (i for i, row in enumerate(rows) if str(row.get("step_id") or "") == step_id),
                -1,
            )
            if index < 0:
                return None, "editor_patch_step_not_found"
            base = dict(rows[index])
            replacement = _compact_route_spec(step)
            # A patch is a step mutation, not an implicit deletion of the
            # ReactionJSON program.  Codex frequently edits product identity
            # or conditions while omitting the unchanged operations (and some
            # provider serializers encode the omission as an empty string).
            # Preserve the host/model route operation in that case; an
            # explicit non-empty operation list still replaces it.
            if not replacement.get("reaction_operations") and base.get(
                "reaction_operations"
            ):
                replacement["reaction_operations"] = [
                    dict(operation)
                    for operation in base.get("reaction_operations") or []
                    if isinstance(operation, Mapping)
                ]
            for field in (
                "product_smiles",
                "mapped_product_smiles",
                "precursor_smiles",
                "mapped_precursor_smiles",
                "reaction_family",
                "product_retron_type",
                "transformation_rationale",
                "conditions",
                "catalyst",
                "enzyme",
                "limitations",
            ):
                if field not in step:
                    replacement[field] = base.get(field)
            replacement["step_id"] = str(replacement.get("step_id") or step_id)
            rows[index] = {**base, **replacement}
        elif op == "insert_after":
            if not isinstance(step, Mapping):
                return None, "editor_patch_insertion_missing"
            index = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if str(row.get("step_id") or "") == after_step_id
                ),
                -1,
            )
            if index < 0:
                return None, "editor_patch_anchor_not_found"
            rows.insert(index + 1, _compact_route_spec(step))
        elif op == "delete_step":
            index = next(
                (i for i, row in enumerate(rows) if str(row.get("step_id") or "") == step_id),
                -1,
            )
            if index < 0:
                return None, "editor_patch_step_not_found"
            rows.pop(index)
        elif op == "reorder":
            order = [str(value) for value in patch.get("step_ids") or [] if str(value)]
            by_id = {str(row.get("step_id") or ""): row for row in rows}
            if len(order) != len(rows) or set(order) != set(by_id):
                return None, "editor_patch_reorder_invalid"
            rows = [by_id[value] for value in order]
        elif op == "set_conditions":
            index = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if str(row.get("step_id") or "") == step_id
                ),
                -1,
            )
            if index < 0:
                return None, "editor_patch_step_not_found"
            conditions = patch.get("conditions")
            catalyst = patch.get("catalyst")
            if conditions is None and catalyst is None:
                return None, "editor_patch_conditions_missing"
            if conditions is not None:
                if not isinstance(conditions, list):
                    return None, "editor_patch_conditions_invalid"
                rows[index]["conditions"] = [
                    str(value) for value in conditions if str(value).strip()
                ]
            if catalyst is not None:
                rows[index]["catalyst"] = str(catalyst)
        else:
            return None, "editor_patch_operation_invalid"
    return (rows, "") if rows else (None, "editor_patch_route_empty")


def _compact_route_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Strip host-derived fields before recompiling an Editor route."""

    row = dict(value)
    predictions = [
        dict(item)
        for item in row.get("condition_predictions") or []
        if isinstance(item, Mapping)
    ]
    return {
        "step_id": str(row.get("step_id") or ""),
        "product_smiles": str(row.get("product_smiles") or ""),
        "mapped_product_smiles": str(row.get("mapped_product_smiles") or ""),
        "precursor_smiles": list(row.get("precursor_smiles") or []),
        "mapped_precursor_smiles": list(
            row.get("mapped_precursor_smiles") or []
        ),
        "reaction_family": str(
            row.get("reaction_family") or row.get("transformation_hypothesis") or ""
        ),
        "step_role": _normalize_step_role(row.get("step_role")),
        "checkpoint_relation": _normalize_checkpoint_relation(
            row.get("checkpoint_relation")
        ),
        "product_retron_type": str(row.get("product_retron_type") or ""),
        "transformation_rationale": str(
            row.get("transformation_rationale") or row.get("strategic_role") or ""
        ),
        "conditions": list(
            row.get("conditions")
            or [
                reagent
                for prediction in predictions
                for reagent in prediction.get("reagents") or []
            ]
        ),
        "catalyst": str(
            row.get("catalyst")
            or next((item.get("catalyst") or "" for item in predictions), "")
        ),
        "enzyme": str(
            row.get("enzyme")
            or next((item.get("enzyme") or "" for item in predictions), "")
        ),
        "execution_domain": str(row.get("execution_domain") or "chemical"),
        "biocatalytic_step": (
            dict(row.get("biocatalytic_step") or {})
            if isinstance(row.get("biocatalytic_step"), Mapping)
            else None
        ),
        "limitations": list(row.get("limitations") or []),
        "reaction_operations": [
            dict(operation)
            for operation in row.get("reaction_operations") or []
            if isinstance(operation, Mapping)
        ],
    }


def _editor_route_expansions_from_record(
    record: WorkerRunRecord,
    *,
    current_steps: Iterable[Mapping[str, Any]],
    mapped_target_smiles: str,
    expected_target_smiles: str,
) -> tuple[list[NodeExpansion] | None, dict[str, Any], str]:
    candidate = _route_json_candidate(record)
    if candidate is None:
        return None, {"reason": "worker_output_not_accepted"}, ""
    replace_span = candidate.get("replace_span")
    if isinstance(replace_span, Mapping):
        merged, reason = _apply_replace_span(current_steps, replace_span)
        if merged is None:
            return None, {
                "reason": reason,
                "editor_mutation_mode": "replace_span",
                "replace_span": dict(replace_span),
            }, "replace_span"
        expansions, diagnostic = _compile_editor_route_rows_with_diagnostic(
            merged,
            mapped_target_smiles=mapped_target_smiles,
            expected_target_smiles=expected_target_smiles,
        )
        if expansions is None:
            return None, {
                **diagnostic,
                "editor_mutation_mode": "replace_span",
                "replace_span": dict(replace_span),
            }, "replace_span"
        return expansions, {}, "replace_span"
    patch_rows = candidate.get("route_patch")
    if isinstance(patch_rows, list) and patch_rows:
        patched, reason = _apply_route_patch(current_steps, patch_rows)
        if patched is None:
            return None, {"reason": reason}, "route_patch"
        # Keep every host-provided mapped boundary and compile the edited DAG
        # directly. A linear scaffold loses sibling frontiers and can bind an
        # edit to an isomorphic fragment from the wrong atom-map namespace.
        expansions, diagnostic = _compile_editor_route_rows_with_diagnostic(
            patched,
            mapped_target_smiles=mapped_target_smiles,
            expected_target_smiles=expected_target_smiles,
        )
        if expansions is None:
            # A RouteJSON Editor sometimes puts the newly exposed precursor
            # in ``replace_step.step.product_smiles``.  That is a boundary
            # error in target-rooted storage order: the product boundary of
            # the replaced row is immutable unless the whole downstream
            # chain is edited as well.  If the ReactionJSON operations are
            # nevertheless replayable against the frozen row, retain the
            # host-compiled prefix as a diagnostic working route.  This is
            # deliberately not promoted to the authoritative route because
            # its suffix still needs an Editor/search continuation.
            working = _editor_replayable_working_prefix_from_patch(
                current_steps=current_steps,
                patch_rows=patch_rows,
                mapped_target_smiles=mapped_target_smiles,
                expected_target_smiles=expected_target_smiles,
            )
            if working is not None:
                working_expansions, working_diagnostic = working
                return (
                    working_expansions,
                    {
                        **diagnostic,
                        **working_diagnostic,
                        "editor_working_route": True,
                    },
                    "route_patch_working_prefix",
                )
            return None, {
                **diagnostic,
                "editor_mutation_mode": "route_patch",
                "route_patch": [
                    dict(value) for value in patch_rows if isinstance(value, Mapping)
                ],
            }, "route_patch"
        return expansions, {}, "route_patch"
    raw_route = candidate.get("route_json")
    if isinstance(raw_route, list) and raw_route:
        expansions, diagnostic = _compile_editor_route_rows_with_diagnostic(
            raw_route,
            mapped_target_smiles=mapped_target_smiles,
            expected_target_smiles=expected_target_smiles,
        )
        if expansions is not None:
            return expansions, {}, "full_route_json"
        return None, {
            **diagnostic,
            "editor_mutation_mode": "full_route_json",
        }, "full_route_json"
    return None, {"reason": "editor_route_mutation_missing"}, ""


def _editor_replayable_working_prefix_from_patch(
    *,
    current_steps: Iterable[Mapping[str, Any]],
    patch_rows: Iterable[Mapping[str, Any]],
    mapped_target_smiles: str,
    expected_target_smiles: str,
) -> tuple[list[NodeExpansion], dict[str, Any]] | None:
    """Recover a host-replayable Editor prefix without granting full-route authority.

    In a retrosynthetic RouteJSON document ``product_smiles`` is the advanced
    intermediate at that row.  A model may return the newly exposed precursor
    there instead, which makes the complete patch fail at the first downstream
    dependency even though its ReactionJSON edit is valid for the frozen
    product.  We only recover a prefix when all of the following hold:

    * the patch is a single ``replace_step`` against an existing row;
    * the Editor operations replay against that row's host-mapped product; and
    * the target-rooted prefix through that row compiles completely.

    The caller serializes this as an auditable working candidate.  It is never
    treated as a closed or authoritative route.
    """

    frozen_steps = [
        dict(row) for row in current_steps if isinstance(row, Mapping)
    ]
    patches = [dict(row) for row in patch_rows if isinstance(row, Mapping)]
    if not frozen_steps or len(patches) != 1:
        return None
    patch = patches[0]
    if str(patch.get("op") or "").strip().lower() != "replace_step":
        return None
    step_id = str(patch.get("step_id") or "")
    index = next(
        (
            position
            for position, row in enumerate(frozen_steps)
            if str(row.get("step_id") or "") == step_id
        ),
        -1,
    )
    replacement = patch.get("step")
    if index < 0 or not isinstance(replacement, Mapping):
        return None
    base = frozen_steps[index]
    operations = normalize_reaction_operations(
        replacement.get("reaction_operations") or ()
    )
    if not operations:
        return None
    mapped_product = str(base.get("mapped_product_smiles") or "").strip()
    base_product = _canonical_smiles(base.get("product_smiles"))
    if not mapped_product or not base_product:
        return None
    try:
        # This is the decisive boundary check.  It proves only that the local
        # edit is replayable on the frozen product; it does not prove the
        # resulting reaction or any later route step.
        RouteJSONCompiler().compile_step(
            mapped_product_smiles=mapped_product,
            operations=operations,
            expected_product_smiles=base_product,
        )
    except ReactionJsonReplayError:
        return None

    patched, reason = _apply_route_patch(frozen_steps, patches)
    if patched is None:
        return None
    patched[index] = {
        **dict(patched[index]),
        # Keep the frozen target-rooted boundary.  The host derives the
        # precursor list from the Editor's replayable operations.
        "product_smiles": base_product,
        "precursor_smiles": [],
        "reaction_operations": [dict(value) for value in operations],
    }
    prefix = _editor_route_scaffold(
        patched[: index + 1],
        mapped_target_smiles=mapped_target_smiles,
    )
    expansions, diagnostic = _compile_editor_route_rows_with_diagnostic(
        prefix,
        mapped_target_smiles=mapped_target_smiles,
        expected_target_smiles=expected_target_smiles,
    )
    if expansions is None:
        return None
    return expansions, {
        "editor_mutation_mode": "route_patch_working_prefix",
        "working_route_depth": len(expansions),
        "working_route_step_id": step_id,
        "working_route_boundary_repaired": True,
        "working_route_compile_diagnostic": dict(diagnostic),
        "patch_apply_reason": reason,
    }


def _route_terminal_precursors(expansions: Iterable[NodeExpansion]) -> tuple[str, ...]:
    """Return only leaves not consumed as the product of a later route step."""

    rows = tuple(expansions)
    consumed_products = {row.product_smiles for row in rows[1:]}
    return tuple(
        dict.fromkeys(
            precursor
            for row in rows
            for precursor in row.precursor_smiles
            if precursor not in consumed_products
        )
    )


def _route_terminal_precursor_pairs(
    expansions: Iterable[NodeExpansion],
) -> tuple[tuple[str, str], ...]:
    """Return canonical leaves paired with the host-preserved atom maps."""

    rows = tuple(expansions)
    consumed_products = {row.product_smiles for row in rows[1:]}
    pairs: list[tuple[str, str]] = []
    for row in rows:
        mapped = tuple(row.mapped_precursor_smiles or ())
        for index, precursor in enumerate(row.precursor_smiles):
            if precursor in consumed_products:
                continue
            mapped_precursor = mapped[index] if index < len(mapped) else _mapped_smiles(precursor)
            pair = (_canonical_smiles(precursor), str(mapped_precursor or ""))
            if pair[0] and pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def _refresh_branch_from_reactionjson_or_search(
    branch: dict[str, Any],
    search: ChemEnzyReactionJsonOrSearch,
) -> None:
    """Project the current best route while keeping all OR siblings in ChemEnzy."""

    projection = search.project()
    branch["steps"] = [dict(row) for row in projection.steps]
    branch["open_leaf_states"] = deque(
        dict(row) for row in projection.open_leaf_states
    )
    branch["deferred_tail_leaf_states"] = deque(
        dict(row) for row in projection.deferred_tail_leaf_states
    )
    branch["blocked_materializations"] = list(
        dict.fromkeys(
            _canonical_smiles(row.get("smiles"))
            for row in projection.deferred_tail_leaf_states
            if _canonical_smiles(row.get("smiles"))
        )
    )
    branch["expanded_products"] = {
        _canonical_smiles(row.get("product_smiles"))
        for row in projection.steps
        if _canonical_smiles(row.get("product_smiles"))
    }
    branch["reactionjson_or_search"] = dict(projection.summary)
    branch["complete_in_bound_stock"] = bool(projection.complete)
    _sync_open_leaf_projection(branch)


def _sync_open_leaf_projection(branch: dict[str, Any]) -> None:
    """Derive unresolved leaves from active and template-tail-only states.

    ``open_leaf_states`` is the Route Builder work queue.  A leaf whose
    ReactionJSON failed the bounded materialization retries must leave that
    queue, but it is still an unresolved route leaf and must remain visible to
    the downstream short-tail search.  Keeping those meanings separate avoids
    both an infinite Route Builder retry and a false stock-closed route.
    """

    states = [
        dict(row)
        for row in branch.get("open_leaf_states") or []
        if isinstance(row, Mapping) and _canonical_smiles(row.get("smiles"))
    ]
    deferred = [
        dict(row)
        for row in branch.get("deferred_tail_leaf_states") or []
        if isinstance(row, Mapping) and _canonical_smiles(row.get("smiles"))
    ]
    branch["open_leaf_states"] = deque(states)
    branch["deferred_tail_leaf_states"] = deque(deferred)
    branch["open_leaves"] = deque(
        dict.fromkeys(
            _canonical_smiles(row.get("smiles"))
            for row in [*states, *deferred]
        )
    )


def _branch_has_expandable_leaf(branch: Mapping[str, Any]) -> bool:
    search = branch.get("_reactionjson_or_search")
    if isinstance(search, ChemEnzyReactionJsonOrSearch):
        return search.select_open_node() is not None
    return any(
        isinstance(row, Mapping)
        and bool(_canonical_smiles(row.get("smiles")))
        and bool(str(row.get("mapped_smiles") or "").strip())
        for row in branch.get("open_leaf_states") or []
    )


def _branch_stock_closed(branch: Mapping[str, Any]) -> bool:
    """Return true only for a materialized route with no unresolved leaf."""

    search_summary = dict(branch.get("reactionjson_or_search") or {})
    if search_summary:
        return (
            bool(search_summary.get("root_solved"))
            and bool(branch.get("steps"))
            and not bool(branch.get("open_leaves"))
            and not bool(branch.get("blocked_materializations"))
        )
    return (
        bool(branch.get("steps"))
        and not bool(branch.get("open_leaves"))
        and not bool(branch.get("blocked_materializations"))
    )


def _pop_open_leaf_state(
    branch: dict[str, Any],
) -> tuple[str, str] | None:
    """Pop one leaf without ever re-numbering its mapped graph."""

    states = branch.setdefault("open_leaf_states", deque())
    while states:
        raw = states.popleft()
        if not isinstance(raw, Mapping):
            continue
        product = _canonical_smiles(raw.get("smiles"))
        mapped = str(raw.get("mapped_smiles") or "").strip()
        if product and mapped and _canonical_smiles(mapped) == product:
            _sync_open_leaf_projection(branch)
            return product, mapped
    _sync_open_leaf_projection(branch)
    return None


def _route_dependency_links(
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, list[str]]]:
    """Describe the target-rooted DAG without inventing another authority."""

    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    step_ids = [
        str(row.get("step_id") or f"step:{index}")
        for index, row in enumerate(rows, start=1)
    ]
    products = [_canonical_smiles(row.get("product_smiles")) for row in rows]
    links: list[dict[str, list[str]]] = []
    for index, row in enumerate(rows):
        product = products[index]
        precursors = {
            canonical
            for value in row.get("precursor_smiles") or []
            if (canonical := _canonical_smiles(value))
        }
        links.append(
            {
                "parent_step_ids": [
                    step_ids[parent_index]
                    for parent_index, parent in enumerate(rows)
                    if product
                    and product
                    in {
                        _canonical_smiles(value)
                        for value in parent.get("precursor_smiles") or []
                        if _canonical_smiles(value)
                    }
                ],
                "expanded_precursor_step_ids": [
                    step_ids[child_index]
                    for child_index, child_product in enumerate(products)
                    if child_index != index
                    and child_product
                    and child_product in precursors
                ],
            }
        )
    return links


def _compact_route_rows(steps: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the entire route in Editor prompts while dropping bulky derived fields."""

    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    dependency_links = _route_dependency_links(rows)
    compact: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        route_row = _compact_route_spec(row)
        route_row["step_id"] = str(route_row.get("step_id") or f"step:{index}")
        route_row["strategy_anchor"] = bool(row.get("strategy_anchor"))
        route_row.update(dependency_links[index - 1])
        compact.append(route_row)
    return compact


def _minimal_editor_prompt_route_rows(
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep topology and replay authority while omitting non-structural prose.

    Paper-matched routes can contain 25 fully materialized rows. These rows
    retain every dependency identity, exact mapped boundary, concise condition
    hypothesis and ReactionJSON program; only non-executable prose and bulky
    derived audit metadata are compacted away.
    """

    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    dependency_links = _route_dependency_links(rows)
    compact: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item: dict[str, Any] = {
            "step_id": str(row.get("step_id") or f"step:{index}"),
            "product_smiles": str(row.get("product_smiles") or ""),
            "mapped_product_smiles": str(
                row.get("mapped_product_smiles") or ""
            ),
            "precursor_smiles": list(row.get("precursor_smiles") or []),
            "mapped_precursor_smiles": list(
                row.get("mapped_precursor_smiles") or []
            ),
            "reaction_operations": [
                dict(operation)
                for operation in row.get("reaction_operations") or []
                if isinstance(operation, Mapping)
            ],
        }
        reaction_family = str(
            row.get("reaction_family")
            or row.get("transformation_hypothesis")
            or ""
        ).strip()
        if reaction_family:
            item["reaction_family"] = reaction_family[:160]
        predictions = [
            dict(value)
            for value in row.get("condition_predictions") or []
            if isinstance(value, Mapping)
        ]
        item["conditions"] = [
            str(reagent)[:240]
            for reagent in (
                row.get("conditions")
                or [
                    value
                    for prediction in predictions[:1]
                    for value in prediction.get("reagents") or []
                ]
            )
            if str(reagent)
        ][:4]
        item["catalyst"] = str(
            row.get("catalyst")
            or next(
                (
                    prediction.get("catalyst") or ""
                    for prediction in predictions
                ),
                "",
            )
        )[:160]
        item.update(dependency_links[index - 1])
        compact.append(item)
    return compact


def _editor_route_scaffold(
    steps: Iterable[Mapping[str, Any]],
    *,
    mapped_target_smiles: str,
) -> list[dict[str, Any]]:
    """Expose host-derived downstream product identities to the Editor.

    Models often redraw an isomeric SMILES for a fragment that the preceding
    ReactionJSON replay already emitted.  The two strings may be constitutionally
    identical while differing in an unassigned or newly invented stereocenter.
    Such a redraw is not a new chemistry proposal; it is a broken RouteJSON
    edge.  This helper normalizes only the editor's working copy to the exact
    host precursor and leaves all ReactionJSON operations untouched.
    """

    rows = [dict(value) for value in steps if isinstance(value, Mapping)]
    if not rows:
        return []
    scaffold = [dict(row) for row in rows]
    current_mapped = str(mapped_target_smiles or "").strip()
    current_product = _canonical_smiles(current_mapped)
    if not current_mapped or not current_product:
        return scaffold
    previous: MaterializedReaction | None = None
    for index, row in enumerate(scaffold):
        operations = normalize_reaction_operations(row.get("reaction_operations") or ())
        if index == 0:
            row["product_smiles"] = current_product
        elif previous is not None:
            declared = _canonical_smiles(row.get("product_smiles"))
            match = _match_editor_precursor(
                declared,
                previous.precursor_smiles,
                previous.mapped_precursor_smiles,
            )
            if match is None:
                match = _match_editor_precursor_by_operation_maps(
                    previous.precursor_smiles,
                    previous.mapped_precursor_smiles,
                    operations,
                )
            if match is None:
                break
            row["product_smiles"], current_mapped = match
            current_product = row["product_smiles"]
        if not operations:
            break
        try:
            previous = RouteJSONCompiler().compile_step(
                mapped_product_smiles=current_mapped,
                operations=operations,
                expected_product_smiles=current_product,
            )
        except ReactionJsonReplayError:
            break
        if index + 1 < len(scaffold) and previous is not None:
            next_declared = _canonical_smiles(
                scaffold[index + 1].get("product_smiles")
            )
            next_match = _match_editor_precursor(
                next_declared,
                previous.precursor_smiles,
                previous.mapped_precursor_smiles,
            )
            if next_match is None:
                next_match = _match_editor_precursor_by_operation_maps(
                    previous.precursor_smiles,
                    previous.mapped_precursor_smiles,
                    normalize_reaction_operations(
                        scaffold[index + 1].get("reaction_operations") or ()
                    ),
                )
            if next_match is None:
                break
            # The next loop iteration repeats this assignment so that the
            # current mapped product always remains paired with its product.
    return scaffold


def _match_editor_precursor(
    declared_product: str,
    precursors: Iterable[str],
    mapped_precursors: Iterable[str] = (),
) -> tuple[str, str] | None:
    candidates = [
        (_canonical_smiles(value), str(mapped))
        for value, mapped in zip(precursors, mapped_precursors)
        if _canonical_smiles(value) and str(mapped)
    ]
    exact = [pair for pair in candidates if pair[0] == declared_product]
    if len(exact) == 1:
        return exact[0]
    nonisomeric_declared = _canonical_smiles_nonisomeric(declared_product)
    approximate = [pair for pair in candidates if _canonical_smiles_nonisomeric(pair[0]) == nonisomeric_declared]
    if len(approximate) == 1:
        return approximate[0]
    return None


def _match_editor_precursor_by_operation_maps(
    precursors: Iterable[str],
    mapped_precursors: Iterable[str],
    operations: Iterable[Mapping[str, Any]],
) -> tuple[str, str] | None:
    """Select a host precursor by the atom maps used by the next edit.

    Editor patches sometimes redraw a fragment with a constitutionally
    different SMILES, but retain the correct map IDs in the proposed
    ReactionJSON operation.  When exactly one prior precursor contains every
    referenced map, that mapped host fragment is the only safe chain identity.
    """

    required_maps: set[int] = set()
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        for key in ("map_a", "map_b", "map_idx"):
            try:
                value = int(operation.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                required_maps.add(value)
        for key in ("map_indices", "stereo_atom_maps"):
            values = operation.get(key)
            if not isinstance(values, (list, tuple, set)):
                continue
            for value in values:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    required_maps.add(parsed)
    if not required_maps:
        return None
    candidates: list[tuple[str, str]] = []
    for value, mapped in zip(precursors, mapped_precursors):
        canonical = _canonical_smiles(value)
        mapped_text = str(mapped or "")
        if not canonical or not mapped_text:
            continue
        observed_maps = {
            int(raw)
            for raw in re.findall(r":(\d+)", mapped_text)
            if raw.isdigit()
        }
        if required_maps.issubset(observed_maps):
            candidates.append((canonical, mapped_text))
    return candidates[0] if len(candidates) == 1 else None


def _expansion_from_record(
    record: WorkerRunRecord,
    *,
    expected_product: str,
    require_strategy_card: bool = False,
    mapped_product_smiles: str = "",
    require_reaction_operations: bool = False,
    minimum_route_depth: int = 1,
    single_step_only: bool = False,
) -> NodeExpansion | None:
    expansions = _expansions_from_record(
        record,
        expected_product=expected_product,
        require_strategy_card=require_strategy_card,
        mapped_product_smiles=mapped_product_smiles,
        require_reaction_operations=require_reaction_operations,
        minimum_route_depth=minimum_route_depth,
        single_step_only=single_step_only,
    )
    return expansions[0] if expansions else None


def _expansion_rejection_diagnostic(
    record: WorkerRunRecord,
    *,
    expected_product: str,
    mapped_product_smiles: str,
    require_reaction_operations: bool,
    require_complete_route_json: bool = False,
    minimum_route_depth: int = 1,
    single_step_only: bool = False,
    reserved_atom_maps: Iterable[int] = (),
) -> dict[str, Any]:
    """Return causal Route Builder feedback without weakening replay."""

    route_draft = _route_json_candidate(record)
    if route_draft is not None:
        row = route_draft
    elif record.status == "accepted_draft":
        payload = dict(dict(record.output_artifact or {}).get("payload") or {})
        candidates = [
            dict(value)
            for value in payload.get("candidates") or []
            if isinstance(value, Mapping)
        ]
        if len(candidates) != 1:
            return {"reason": "candidate_count_invalid"}
        row = candidates[0]
    else:
        return {"reason": "worker_output_not_accepted"}
    raw_route = None if single_step_only else row.get("route_json")
    if raw_route is None and require_complete_route_json:
        return {
            "reason": "route_json_missing",
            "declared_product_smiles": _canonical_smiles(row.get("product_smiles")),
        }
    if require_complete_route_json and not isinstance(raw_route, list):
        return {"reason": "route_json_invalid"}
    if require_complete_route_json:
        route_diagnostic = _route_json_diagnostic(
            raw_route,
            expected_product=expected_product,
            mapped_product_smiles=mapped_product_smiles,
            minimum_route_depth=minimum_route_depth,
        )
        if route_diagnostic is not None:
            return route_diagnostic
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
            reserved_atom_maps=reserved_atom_maps,
        )
        if replay.get("replay_succeeded") is not True:
            return {
                "reason": "strategy_graph_edit_replay_failed",
                "replay_error": str(replay.get("reason") or ""),
                **{
                    key: replay[key]
                    for key in (
                        "operation_index",
                        "failed_operation",
                        "failure_stage",
                        "failure_detail",
                        "endpoint_aromaticity",
                        "allowed_orders",
                    )
                    if key in replay
                },
                "declared_precursor_smiles": list(
                    replay.get("declared_precursor_smiles") or []
                ),
                "replayed_precursor_smiles": [],
            }
        return {
            "reason": "invalid_expansion_contract",
            "replay_error": str(replay.get("reason") or ""),
            "declared_precursor_smiles": list(
                replay.get("declared_precursor_smiles") or []
            ),
            "replayed_precursor_smiles": list(
                replay.get("replayed_precursor_smiles") or []
            ),
        }
    return {"reason": "invalid_expansion_contract"}


def _route_json_diagnostic(
    raw_route: Any,
    *,
    expected_product: str,
    mapped_product_smiles: str,
    minimum_route_depth: int = 1,
) -> dict[str, Any] | None:
    """Explain the first failed complete-route contract without changing authority."""

    if not isinstance(raw_route, list) or not raw_route:
        return {
            "reason": "route_json_invalid",
            "detail": "route_json_must_be_a_non_empty_list",
        }
    required_depth = max(1, int(minimum_route_depth))
    if len(raw_route) < required_depth:
        return {
            "reason": "route_json_incomplete",
            "detail": "route_depth_below_required_minimum",
            "route_depth": len(raw_route),
            "minimum_route_depth": required_depth,
        }
    prior_precursors: tuple[str, ...] = ()
    prior_mapped_precursors: tuple[str, ...] = ()
    prior_products: set[str] = set()
    for index, value in enumerate(raw_route):
        if not isinstance(value, Mapping):
            return {
                "reason": "route_json_step_invalid",
                "step_index": index,
                "detail": "route_json_step_must_be_an_object",
            }
        step = dict(value)
        declared_product = _canonical_smiles(step.get("product_smiles"))
        product = declared_product
        if not product:
            return {
                "reason": "route_json_step_invalid",
                "step_index": index,
                "detail": "product_smiles_invalid",
            }
        if index == 0 and product != _canonical_smiles(expected_product):
            return {
                "reason": "route_json_chain_invalid",
                "step_index": index,
                "detail": "first_product_mismatch",
                "declared_product_smiles": product,
                "expected_product_smiles": _canonical_smiles(expected_product),
            }
        mapped = mapped_product_smiles if index == 0 else ""
        if index > 0:
            match = _match_editor_precursor(
                product,
                prior_precursors,
                prior_mapped_precursors,
            )
            if match is None:
                return {
                    "reason": "route_json_chain_invalid",
                    "step_index": index,
                    "detail": (
                        "product_not_in_previous_mapped_precursors"
                        if product in prior_precursors
                        else "product_not_in_previous_precursors"
                    ),
                    "product_smiles": product,
                    "previous_precursors": list(prior_precursors),
                    "previous_mapped_precursors": list(prior_mapped_precursors),
                }
            product, mapped = match
        if index == 0 and not mapped:
            return {
                "reason": "route_json_step_invalid",
                "step_index": index,
                "detail": "mapped_product_smiles_invalid",
            }
        if product in prior_products:
            return {
                "reason": "route_json_chain_invalid",
                "step_index": index,
                "detail": "product_repeated",
                "product_smiles": product,
            }
        operations = normalize_reaction_operations(step.get("reaction_operations") or ())
        if not operations:
            return {
                "reason": "route_json_step_reaction_operations_missing",
                "step_index": index,
                "product_smiles": product,
            }
        # Use the same RouteJSONCompiler that grants structure authority to
        # the expansion path.  The lightweight diagnostic helper may return
        # mapped and unmapped precursor arrays in different fragment orders;
        # feeding those parallel arrays into the chain validator creates a
        # false `map_not_found` rejection for otherwise replayable routes.
        try:
            materialized = RouteJSONCompiler().compile_step(
                mapped_product_smiles=mapped,
                operations=operations,
                expected_product_smiles=product,
            )
        except ReactionJsonReplayError:
            replay = diagnose_reactionjson(
                mapped_product_smiles=mapped,
                operations=operations,
                declared_precursor_smiles=step.get("precursor_smiles") or [],
            )
            return {
                "reason": "route_json_step_replay_failed",
                "step_index": index,
                "product_smiles": product,
                "replay_diagnostic": replay,
            }
        precursors = tuple(materialized.precursor_smiles)
        mapped_precursors = tuple(materialized.mapped_precursor_smiles)
        if not precursors or any(value in prior_products for value in precursors):
            return {
                "reason": "route_json_chain_invalid",
                "step_index": index,
                "detail": "ancestor_cycle_or_empty_precursors",
                "replayed_precursor_smiles": list(precursors),
            }
        prior_products.add(product)
        prior_precursors = precursors
        prior_mapped_precursors = mapped_precursors
    return None


def _strategy_key_bond_pairs(
    strategy_card: Mapping[str, Any] | None,
) -> frozenset[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    card = dict(strategy_card or {})
    values = card.get("anchor_bond_signature") or card.get("key_bond_signature")
    if not values and card.get("key_bond_changes"):
        normalized = normalize_strategy_policy_card(card)
        values = (
            normalized.get("anchor_bond_signature")
            or normalized.get("key_bond_signature")
        )
    for value in values or ():
        match = re.fullmatch(r"map_pair:(\d+):(\d+)", str(value or "").strip())
        if match:
            pairs.add(tuple(sorted((int(match.group(1)), int(match.group(2))))))
    return frozenset(pairs)


def _strategy_card_digest(strategy_card: Mapping[str, Any] | None) -> str:
    card = dict(strategy_card or {})
    return str(
        card.get("strategy_digest")
        or card.get("content_sha256")
        or card.get("strategy_id")
        or ""
    )


def _mapped_bond_pairs(mapped_smiles: str) -> frozenset[tuple[int, int]]:
    molecule = Chem.MolFromSmiles(str(mapped_smiles or ""))
    if molecule is None:
        return frozenset()
    pairs: set[tuple[int, int]] = set()
    for bond in molecule.GetBonds():
        left = int(bond.GetBeginAtom().GetAtomMapNum() or 0)
        right = int(bond.GetEndAtom().GetAtomMapNum() or 0)
        if left and right:
            pairs.add(tuple(sorted((left, right))))
    return frozenset(pairs)


def _strategy_anchor_fulfilled_for_card(
    steps: Iterable[Mapping[str, Any]],
    strategy_card: Mapping[str, Any] | None,
) -> bool:
    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    required = _strategy_key_bond_pairs(strategy_card)
    if required:
        return required.issubset(
            _realized_strategy_key_bond_pairs(rows, strategy_card)
        )
    # A prose Strategy is steering context. Builder-authored role/anchor
    # labels are descriptive and cannot prove execution; the independent
    # full-route Critic owns that judgment.
    return False


def _realized_strategy_key_bond_pairs(
    steps: Iterable[Mapping[str, Any]],
    strategy_card: Mapping[str, Any] | None,
) -> frozenset[tuple[int, int]]:
    """Union replayed key edits belonging to one frozen StrategyCard."""

    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    digest = _strategy_card_digest(strategy_card)
    has_bound_cards = any(
        isinstance(row.get("strategy_card"), Mapping) for row in rows
    )
    realized: set[tuple[int, int]] = set()
    for row in rows:
        bound_card = row.get("strategy_card")
        if digest and has_bound_cards:
            if not isinstance(bound_card, Mapping):
                continue
            if _strategy_card_digest(bound_card) != digest:
                continue
        signature = reaction_edit_signature(row.get("reaction_operations") or ())
        realized.update(
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair in signature.get("changed_map_pairs") or ()
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        )
    return frozenset(realized)


def _strategy_anchor_progress(
    steps: Iterable[Mapping[str, Any]],
    strategy_card: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    required = _strategy_key_bond_pairs(strategy_card)
    realized = _realized_strategy_key_bond_pairs(rows, strategy_card)
    relevant_realized = required.intersection(realized)

    def labels(values: Iterable[tuple[int, int]]) -> list[str]:
        return [f"map_pair:{left}:{right}" for left, right in sorted(values)]

    return {
        "required_map_pairs": labels(required),
        "realized_map_pairs": labels(relevant_realized),
        "remaining_map_pairs": labels(required - relevant_realized),
        "fulfilled": _strategy_anchor_fulfilled_for_card(rows, strategy_card),
        "authority": "report_only_diagnostic",
        "grants_strategy_adherence": False,
        "grants_strategy_completion": False,
        "completion_semantics": "mapped_edit_overlap_only",
    }


def _ordered_strategy_cards_from_steps(
    *,
    root_strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in [root_strategy_card, *[
        dict(step).get("strategy_card")
        for step in steps
        if isinstance(step, Mapping)
    ]]:
        if not isinstance(raw, Mapping):
            continue
        card = dict(raw)
        digest = _strategy_card_digest(card)
        identity = digest or json.dumps(
            card, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if not card or identity in seen:
            continue
        seen.add(identity)
        cards.append(card)
    return cards


def _active_strategy_card_for_leaf(
    *,
    root_strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
    selected_product_mapped: str,
) -> Mapping[str, Any]:
    """Return the unfulfilled StrategyCard whose mapped bond lives on the leaf."""

    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    if not _strategy_anchor_fulfilled_for_card(rows, root_strategy_card):
        return root_strategy_card
    leaf_pairs = _mapped_bond_pairs(selected_product_mapped)
    cards = _ordered_strategy_cards_from_steps(
        root_strategy_card=root_strategy_card,
        steps=rows,
    )
    for card in reversed(cards[1:]):
        if _strategy_anchor_fulfilled_for_card(rows, card):
            continue
        required = _strategy_key_bond_pairs(card)
        if required and required.issubset(leaf_pairs):
            return card
    return root_strategy_card


def _strategy_milestone_index(
    branch: Mapping[str, Any], strategy_card: Mapping[str, Any]
) -> int:
    digest = _strategy_card_digest(strategy_card)
    cards = [
        dict(row)
        for row in branch.get("strategy_milestone_cards") or []
        if isinstance(row, Mapping)
    ]
    for index, card in enumerate(cards, start=1):
        if digest and _strategy_card_digest(card) == digest:
            return index
    return 1


def _expansion_executes_strategy_anchor(
    expansion: NodeExpansion,
    strategy_card: Mapping[str, Any] | None,
    *,
    fallback: bool = False,
) -> bool:
    """Report only explicit mapped overlap; model role labels are non-authority."""

    required_pairs = _strategy_key_bond_pairs(strategy_card)
    if not required_pairs:
        return False
    signature = reaction_edit_signature(expansion.reaction_operations)
    changed_pairs = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in signature.get("changed_map_pairs") or ()
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    }
    return bool(required_pairs.intersection(changed_pairs))


def _step_row(
    expansion: NodeExpansion,
    *,
    step_id: str,
    strategy_anchor: bool = False,
    strategy_milestone_index: int = 1,
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
    execution_domain = normalize_step_execution_domain(
        expansion.execution_domain,
        enzyme_label=enzyme,
        biocatalytic_step=expansion.biocatalytic_step,
    )
    biocatalytic_step, biocatalytic_reasons = normalize_biocatalytic_step(
        expansion.biocatalytic_step,
        execution_domain=execution_domain,
        product_smiles=expansion.product_smiles,
        precursor_smiles=expansion.precursor_smiles,
        enzyme_label=enzyme,
        step_id=step_id,
    )
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
    required_validation = ["structure", "reaction_feasibility"]
    if execution_domain in BIOLOGICAL_EXECUTION_DOMAINS:
        required_validation.append("exact_substrate_biocatalysis")
    return {
        "step_id": step_id,
        "product_smiles": expansion.product_smiles,
        "precursor_smiles": list(expansion.precursor_smiles),
        "mapped_product_smiles": expansion.mapped_product_smiles,
        "mapped_precursor_smiles": list(expansion.mapped_precursor_smiles),
        "transformation_hypothesis": expansion.reaction_family,
        "strategic_role": expansion.rationale,
        "step_role": _normalize_step_role(expansion.step_role),
        "checkpoint_relation": _normalize_checkpoint_relation(
            expansion.checkpoint_relation
        ),
        "source_hints": [],
        "required_validation": required_validation,
        "hypothesis_only": True,
        "condition_predictions": condition_predictions,
        "limitations": list(expansion.limitations),
        "strategy_card": strategy_card,
        "reaction_operations": [dict(row) for row in expansion.reaction_operations],
        "reaction_edit_digest": edit_digest,
        "reactionjson_audit": dict(expansion.reactionjson_audit or {}),
        "strategy_id": str(strategy_card.get("strategy_id") or ""),
        "strategy_digest": str(strategy_card.get("strategy_digest") or ""),
        "step_kind": (
            "biocatalytic_reaction"
            if execution_domain in BIOLOGICAL_EXECUTION_DOMAINS
            else "chemical_reaction"
        ),
        "execution_domain": execution_domain,
        "biocatalytic_step": biocatalytic_step,
        "biocatalytic_design_deficits": biocatalytic_reasons,
        "strategy_anchor": bool(strategy_anchor),
        "strategy_milestone_index": max(1, int(strategy_milestone_index)),
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


def _host_route_json_from_steps(
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize a complete RouteJSON projection from host-derived step rows."""

    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    if not rows:
        return []
    mapped_target = str(rows[0].get("mapped_product_smiles") or "")
    if all(_step_has_bound_replay_audit(row) for row in rows):
        materialized = [
            _materialized_reaction_from_bound_step(value) for value in rows
        ]
    else:
        # Compatibility-only projection for ordinary/legacy callers that do
        # not require ReactionJSON graph edits. A bound replay is immutable and
        # was handled above; unbound rows may be replayed once here.
        try:
            materialized = list(
                RouteJSONCompiler().compile_route_graph(
                    mapped_target_smiles=mapped_target,
                    steps=rows,
                )
            )
        except ReactionJsonReplayError:
            materialized = [
                MaterializedReaction(
                    product_smiles=_canonical_smiles(value.get("product_smiles")),
                    mapped_product_smiles=str(
                        value.get("mapped_product_smiles") or ""
                    ),
                    precursor_smiles=tuple(
                        _canonical_smiles(item)
                        for item in value.get("precursor_smiles") or []
                        if _canonical_smiles(item)
                    ),
                    mapped_precursor_smiles=tuple(
                        str(item)
                        for item in value.get("mapped_precursor_smiles") or []
                        if str(item)
                    ),
                    reaction_operations=tuple(
                        dict(item)
                        for item in value.get("reaction_operations") or []
                        if isinstance(item, Mapping)
                    ),
                    audit=dict(value.get("reactionjson_audit") or {}),
                )
                for value in rows
            ]
    metadata: list[dict[str, Any]] = []
    for value in rows:
        metadata.append(
            {
                "step_id": str(value.get("step_id") or ""),
                "reaction_family": str(value.get("transformation_hypothesis") or ""),
                "transformation_rationale": str(value.get("strategic_role") or ""),
                "step_role": _normalize_step_role(value.get("step_role")),
                "checkpoint_relation": _normalize_checkpoint_relation(
                    value.get("checkpoint_relation")
                ),
                "conditions": [
                    reagent
                    for prediction in value.get("condition_predictions") or []
                    if isinstance(prediction, Mapping)
                    for reagent in prediction.get("reagents") or []
                ],
                "catalyst": str(
                    next(
                        (
                            prediction.get("catalyst") or ""
                            for prediction in value.get("condition_predictions") or []
                            if isinstance(prediction, Mapping)
                        ),
                        "",
                    )
                ),
                "enzyme": str(
                    next(
                        (
                            prediction.get("enzyme") or ""
                            for prediction in value.get("condition_predictions") or []
                            if isinstance(prediction, Mapping)
                        ),
                        "",
                    )
                ),
                "step_kind": str(value.get("step_kind") or "chemical_reaction"),
                "execution_domain": str(
                    value.get("execution_domain") or "chemical"
                ),
                "biocatalytic_step": dict(value.get("biocatalytic_step") or {}),
                "biocatalytic_design_deficits": list(
                    value.get("biocatalytic_design_deficits") or []
                ),
                "required_validation": list(value.get("required_validation") or []),
                "limitations": list(value.get("limitations") or []),
            }
        )
    return RouteJSONCompiler.assemble_route(materialized, metadata=metadata)


def _step_has_bound_replay_audit(step: Mapping[str, Any]) -> bool:
    """Return true when a row is an immutable output of host replay."""

    row = dict(step)
    audit = dict(row.get("reactionjson_audit") or {})
    if (
        audit.get("schema_version") != "reactionjson_replay_audit.v1"
        or audit.get("accepted") is not True
        or not str(audit.get("content_sha256") or "")
    ):
        return False
    operations = normalize_reaction_operations(row.get("reaction_operations") or ())
    try:
        operation_count = int(audit.get("operation_count") or 0)
    except (TypeError, ValueError):
        return False
    if not operations or operation_count != len(operations):
        return False
    mapped_product = str(row.get("mapped_product_smiles") or "").strip()
    audited_mapped_product = str(audit.get("mapped_product_smiles") or "").strip()
    if (
        not mapped_product
        or _canonical_atom_mapped_smiles(mapped_product)
        != _canonical_atom_mapped_smiles(audited_mapped_product)
    ):
        return False
    precursors = sorted(
        _canonical_smiles(value)
        for value in row.get("precursor_smiles") or []
        if _canonical_smiles(value)
    )
    audited_precursors = sorted(
        _canonical_smiles(value)
        for value in audit.get("precursor_smiles") or []
        if _canonical_smiles(value)
    )
    mapped_precursors = sorted(
        _canonical_atom_mapped_smiles(value)
        for value in row.get("mapped_precursor_smiles") or []
        if _canonical_atom_mapped_smiles(value)
    )
    audited_mapped_precursors = sorted(
        _canonical_atom_mapped_smiles(value)
        for value in audit.get("mapped_precursor_smiles") or []
        if _canonical_atom_mapped_smiles(value)
    )
    return bool(
        precursors
        and len(precursors) == len(mapped_precursors)
        and precursors == audited_precursors
        and mapped_precursors == audited_mapped_precursors
    )


def _materialized_reaction_from_bound_step(
    step: Mapping[str, Any],
) -> MaterializedReaction:
    """Rehydrate a previously host-replayed row without executing it again."""

    row = dict(step)
    if not _step_has_bound_replay_audit(row):
        raise ReactionJsonReplayError("routejson_bound_replay_audit_invalid")
    audit = dict(row.get("reactionjson_audit") or {})
    return MaterializedReaction(
        product_smiles=_canonical_smiles(audit.get("mapped_product_smiles")),
        mapped_product_smiles=str(audit.get("mapped_product_smiles") or ""),
        precursor_smiles=tuple(
            _canonical_smiles(value)
            for value in row.get("precursor_smiles") or []
            if _canonical_smiles(value)
        ),
        mapped_precursor_smiles=tuple(
            str(value)
            for value in row.get("mapped_precursor_smiles") or []
            if str(value)
        ),
        reaction_operations=tuple(
            dict(value)
            for value in normalize_reaction_operations(
                row.get("reaction_operations") or ()
            )
        ),
        audit=audit,
    )


def _canonical_atom_mapped_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _route_steps_host_replay_validation(
    steps: Iterable[Mapping[str, Any]],
    *,
    mapped_target_smiles: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    if not rows:
        return {"complete": False, "reason": "routejson_route_empty"}
    target_mapped = str(mapped_target_smiles or "").strip()
    if not target_mapped:
        return {"complete": False, "reason": "routejson_target_map_missing"}
    try:
        compiled = RouteJSONCompiler().compile_route_graph(
            mapped_target_smiles=target_mapped,
            steps=rows,
            minimum_depth=1,
        )
    except ReactionJsonReplayError as exc:
        failed_index = 0
        for prefix_size in range(1, len(rows) + 1):
            try:
                RouteJSONCompiler().compile_route_graph(
                    mapped_target_smiles=target_mapped,
                    steps=rows[:prefix_size],
                    minimum_depth=1,
                )
            except ReactionJsonReplayError:
                failed_index = prefix_size - 1
                break
        return {
            "complete": False,
            "reason": "routejson_target_rooted_dag_replay_failed",
            "compiler_error": str(exc),
            "step_index": failed_index,
            "compiler_mode": "target_rooted_route_dag",
            **reactionjson_failure_focus(exc),
        }
    return {
        "complete": True,
        "compiled_step_count": len(compiled),
        "compiler_mode": "target_rooted_route_dag",
    }


def _route_steps_are_host_replayable(
    steps: Iterable[Mapping[str, Any]],
    *,
    mapped_target_smiles: str,
) -> bool:
    return _route_steps_host_replay_validation(
        steps,
        mapped_target_smiles=mapped_target_smiles,
    ).get("complete") is True


def _route_execution_profile(
    steps: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report actual step domains instead of inferring fusion from a branch label."""

    step_rows = [row for row in steps if isinstance(row, Mapping)]
    domains = sorted(
        {
            normalize_step_execution_domain(
                row.get("execution_domain"),
                enzyme_label=str(row.get("enzyme") or ""),
                biocatalytic_step=(
                    row.get("biocatalytic_step")
                    if isinstance(row.get("biocatalytic_step"), Mapping)
                    else None
                ),
            )
            for row in step_rows
        }
    )
    has_chemical = "chemical" in domains
    biological_domains = sorted(
        domain for domain in domains if domain in BIOLOGICAL_EXECUTION_DOMAINS
    )
    chemical_strategy_anchor_present = any(
        isinstance(row, Mapping)
        and normalize_step_execution_domain(
            row.get("execution_domain"),
            enzyme_label=str(row.get("enzyme") or ""),
            biocatalytic_step=(
                row.get("biocatalytic_step")
                if isinstance(row.get("biocatalytic_step"), Mapping)
                else None
            ),
        )
        == "chemical"
        and row.get("strategy_anchor") is True
        for row in step_rows
    )
    genuine_fusion = bool(has_chemical and biological_domains)
    return {
        "step_execution_domains": domains,
        "chemical_step_present": has_chemical,
        "biological_step_present": bool(biological_domains),
        "biological_execution_domains": biological_domains,
        "genuine_chemoenzymatic_fusion": genuine_fusion,
        "chemical_strategy_anchor_present": chemical_strategy_anchor_present,
        "strategic_chemoenzymatic_fusion": bool(
            genuine_fusion and chemical_strategy_anchor_present
        ),
        "semantics": {
            "strategy_domain_alone_never_grants_fusion": True,
            "fusion_requires_materialized_steps_in_both_domains": True,
            "strategic_fusion_requires_chemical_strategy_anchor": True,
        },
    }


def _paper_policy_budget_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project policy-budget diagnostics without plan-authority field names."""

    row = dict(value or {})
    # AiZ selected_solved already records the search-local observation.  Do
    # not copy the same boolean into a second plan field that can be mistaken
    # for canonical stock authority.
    row.pop("stock_closed", None)
    if "hard_failures" in row:
        row["hard_failures"] = [
            dict(item) for item in row.get("hard_failures") or []
            if isinstance(item, Mapping)
        ]
    return row


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
    for fallback_ordinal, branch in enumerate(branches, start=1):
        raw_branch_index = branch.get("branch_index")
        ordinal = (
            int(raw_branch_index) + 1
            if isinstance(raw_branch_index, int) and not isinstance(raw_branch_index, bool)
            else fallback_ordinal
        )
        family_id = f"codex:sequential:family:{ordinal}"
        lens = str(branch.get("lens") or f"strategy branch {ordinal}")
        strategy_card = dict(branch.get("strategy_card") or {})
        steps = [dict(row) for row in branch.get("steps") or []]
        execution_profile = _route_execution_profile(steps)
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
                "root_strategy_card": dict(
                    branch.get("root_strategy_card") or strategy_card
                ),
                "strategy_milestone_cards": [
                    dict(row)
                    for row in branch.get("strategy_milestone_cards") or []
                    if isinstance(row, Mapping)
                ],
                "strategy_milestone_attempts": [
                    dict(row)
                    for row in branch.get("strategy_milestone_attempts") or []
                    if isinstance(row, Mapping)
                ],
                "strategic_milestone_count": int(
                    branch.get("strategic_milestone_count") or 0
                ),
                "strategy_id": str(strategy_card.get("strategy_id") or ""),
                "strategy_digest": str(strategy_card.get("strategy_digest") or ""),
                "execution_domain": str(
                    strategy_card.get("execution_domain") or "chemical"
                ),
                "route_execution_profile": execution_profile,
                "chemical_critic": dict(branch.get("chemical_critic") or {}),
                "strategy_tree_engine": str(
                    branch.get("strategy_tree_engine")
                    or "chemenzy_best_first"
                ),
                "aizynthfinder_strategy_search": dict(
                    branch.get("aizynthfinder_strategy_search") or {}
                ),
                "strategy_anchor_diagnostics": [
                    dict(row)
                    for row in branch.get("strategy_anchor_diagnostics") or []
                    if isinstance(row, Mapping)
                ],
                "route_status": (
                    "host_compiled"
                    if steps
                    else (
                        "hypothesis_only_routejson_replay_rejected"
                        if int(branch.get("routejson_rejected_step_count") or 0) > 0
                        else "hypothesis_only_materialization_pending"
                    )
                ),
                "hypothesis_only": True,
                "routejson_rejected_step_count": int(
                    branch.get("routejson_rejected_step_count") or 0
                ),
                "routejson_replay_validation": dict(
                    branch.get("routejson_replay_validation") or {}
                ),
                "strategy_call_count": int(
                    branch.get("strategy_call_count") or 0
                ),
                "route_call_count": int(branch.get("route_call_count") or 0),
                # Keep observed stock closure in the diagnostic namespace;
                # ``GlobalCampaignPlan`` rejects authority-looking keys such
                # as ``stock_closed=true`` because plans themselves cannot
                # grant scientific status.  The paper metric is computed
                # later from the materialized route/stock ledger.
                "paper_policy_call_budget": _paper_policy_budget_projection(
                    branch.get("paper_policy_call_budget") or {}
                ),
                "paper_policy_budget_failure": dict(
                    branch.get("paper_policy_budget_failure") or {}
                ),
                "sidecar_recovered_prefix": bool(
                    branch.get("sidecar_recovered_prefix")
                ),
                "editor_attempt_count": int(
                    branch.get("editor_attempt_count") or 0
                ),
                "editor_call_count": int(
                    branch.get("editor_call_count") or 0
                ),
                "critic_call_count": int(
                    branch.get("critic_call_count") or 0
                ),
                "key_event_critic_call_count": int(
                    branch.get("key_event_critic_call_count") or 0
                ),
                "key_event_critic_completed": bool(
                    branch.get("key_event_critic_completed")
                ),
                "key_event_critic_history": [
                    dict(row)
                    for row in branch.get("key_event_critic_history") or []
                    if isinstance(row, Mapping)
                ],
                "pending_key_event_feedback": dict(
                    branch.get("pending_key_event_feedback") or {}
                ),
                "critic_editor_skipped_incomplete_route_json": bool(
                    branch.get("critic_editor_skipped_incomplete_route_json")
                ),
                "editor_applied_count": int(
                    branch.get("editor_call_count") or 0
                ),
                "materialization_failures": dict(
                    branch.get("materialization_failures") or {}
                ),
                "blocked_materializations": list(
                    branch.get("blocked_materializations") or []
                ),
                "materialization_diagnostics": [
                    dict(row)
                    for row in branch.get("materialization_diagnostics") or []
                    if isinstance(row, Mapping)
                ],
                "materialization_editor_history": [
                    dict(row)
                    for row in branch.get("materialization_editor_history") or []
                    if isinstance(row, Mapping)
                ],
                "reactionjson_or_search": dict(
                    branch.get("reactionjson_or_search") or {}
                ),
                "reactionjson_or_search_resets": [
                    dict(row)
                    for row in branch.get("reactionjson_or_search_resets") or []
                    if isinstance(row, Mapping)
                ],
                "reactionjson_candidate_batches": [
                    dict(row)
                    for row in branch.get("reactionjson_candidate_batches") or []
                    if isinstance(row, Mapping)
                ],
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
                "editor_working_route": dict(
                    branch.get("editor_working_route") or {}
                ),
                "editor_rejection_diagnostics": [
                    dict(row)
                    for row in branch.get("editor_rejection_diagnostics") or []
                    if isinstance(row, Mapping)
                ],
            }
        )
        if steps:
            route_json = _host_route_json_from_steps(steps)
            routejson_validation = _route_steps_host_replay_validation(
                steps,
                mapped_target_smiles=str(
                    branch.get("target_mapped_smiles") or _mapped_smiles(target)
                ),
            )
            host_replay_complete = routejson_validation.get("complete") is True
            skeletons.append(
                {
                    "skeleton_id": f"codex:sequential:skeleton:{ordinal}",
                    "route_family_id": family_id,
                    "summary": (
                        f"{len(steps)} host-compiled node expansions from "
                        f"{int(branch.get('call_count') or len(steps))} compact calls"
                    ),
                    "steps": steps,
                    "root_strategy_card": dict(
                        branch.get("root_strategy_card") or strategy_card
                    ),
                    "strategy_milestone_cards": [
                        dict(row)
                        for row in branch.get("strategy_milestone_cards") or []
                        if isinstance(row, Mapping)
                    ],
                    "strategic_milestone_count": int(
                        branch.get("strategic_milestone_count") or 0
                    ),
                    "strategy_anchor_diagnostics": [
                        dict(row)
                        for row in branch.get("strategy_anchor_diagnostics") or []
                        if isinstance(row, Mapping)
                    ],
                    "route_json": route_json,
                    "routejson_authority": (
                        "host_routejson_dag_compiler"
                        if host_replay_complete
                        else "legacy_declared_route_projection"
                    ),
                    "routejson_replay_complete": host_replay_complete,
                    "routejson_validation_scope": "target_rooted_route_dag",
                    "routejson_replay_validation": routejson_validation,
                    "canonical_admission_status": "pending_at_director_output",
                    "routejson_canonical_admission_complete": False,
                    "route_execution_profile": execution_profile,
                    "chemical_critic": dict(branch.get("chemical_critic") or {}),
                    "critic_call_count": int(
                        branch.get("critic_call_count") or 0
                    ),
                    "editor_attempt_count": int(
                        branch.get("editor_attempt_count") or 0
                    ),
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
                    "editor_working_route": dict(
                        branch.get("editor_working_route") or {}
                    ),
                    "editor_rejection_diagnostics": [
                        dict(row)
                        for row in branch.get("editor_rejection_diagnostics") or []
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
        if steps:
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
            f"{len(families)}/{requested_branch_count} Strategy hypotheses retained; "
            f"{len(skeletons)} have host-replayable target-rooted RouteJSON. "
            "Open leaves from replayable routes are delegated individually to the standard short-tail search."
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


def _model_output_validation_status(record: WorkerRunRecord) -> str:
    """Name worker/schema acceptance without implying route admission."""

    validation = dict(record.output_validation or {})
    if record.status == "accepted_draft" and validation.get("accepted") is not False:
        return "schema_accepted"
    if record.status == "rejected_output" or validation.get("accepted") is False:
        return "schema_rejected"
    return str(record.status or "worker_status_unknown")


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


def _canonical_mapped_smiles(value: Any) -> str:
    """Canonicalize a mapped boundary without changing its map namespace."""

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None or any(atom.GetAtomMapNum() <= 0 for atom in molecule.GetAtoms()):
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_smiles_nonisomeric(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _worker_task_contract_sha256(task: WorkerTask) -> str:
    """Digest scientific task identity while excluding a shrinking timeout."""

    row = task.to_dict()
    budget = dict(row.get("budget") or {})
    # On resume the same call has a smaller operational timeout remaining.
    # Requiring byte-identical timeout values would defeat replay even though
    # the prompt, model and structured-output contract are unchanged.
    budget.pop("timeout_s", None)
    row["budget"] = budget
    return _digest(row)


def _portable_model_input_sha256(
    value: WorkerTask | Mapping[str, Any],
) -> str:
    """Digest exactly what can affect one no-tool model response.

    Runtime task ids, timestamps, work directories, and shrinking timeouts are
    intentionally absent.  Every participant-visible input and the requested
    output family remains bound.
    """

    if isinstance(value, WorkerTask):
        row = {
            "case_id": value.case_id,
            "task_type": value.task_type,
            "artifact_type": value.required_artifact_type,
            "model": value.model,
            "reasoning_effort": value.budget.reasoning_effort,
            "prompt": value.objective,
            "input_refs": list(value.input_refs),
        }
    else:
        row = {
            "case_id": str(value.get("case_id") or ""),
            "task_type": str(value.get("task_type") or ""),
            "artifact_type": str(value.get("artifact_type") or ""),
            "model": str(value.get("model") or ""),
            "reasoning_effort": str(value.get("reasoning_effort") or ""),
            "prompt": str(value.get("prompt") or ""),
            "input_refs": list(value.get("input_refs") or []),
        }
    return _digest(row)


def _seed_record_matches_task(
    record: WorkerRunRecord,
    task: WorkerTask,
) -> bool:
    if task.allowed_tools:
        return False
    artifact = dict(record.output_artifact or {})
    metadata = dict(record.metadata or {})
    validation = dict(record.output_validation or {})
    current_validation = validate_worker_output(task, artifact)
    return bool(
        record.status == "accepted_draft"
        and record.case_id == task.case_id
        and artifact.get("artifact_type") == task.required_artifact_type
        and validation.get("accepted") is True
        and current_validation.get("accepted") is True
        and len(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= int(task.budget.max_output_bytes)
        and str(metadata.get("model") or "") == str(task.model or "")
        and str(metadata.get("model_reasoning_effort") or "")
        == str(task.budget.reasoning_effort or "")
    )


def _aiz_policy_state_fingerprint(
    *,
    selected_leaf_mapped: str,
    route_steps: Sequence[Mapping[str, Any]],
) -> str:
    """Identify one selected AiZ node and its accepted Host-replayed path."""

    return _digest(
        {
            "selected_leaf_mapped": str(selected_leaf_mapped or ""),
            "route_steps": [
                {
                    "mapped_product_smiles": str(
                        row.get("mapped_product_smiles") or ""
                    ),
                    "mapped_precursor_smiles": list(
                        row.get("mapped_precursor_smiles") or []
                    ),
                    "reaction_operations": [
                        dict(operation)
                        for operation in normalize_reaction_operations(
                            row.get("reaction_operations") or ()
                        )
                    ],
                }
                for row in route_steps
            ],
        }
    )


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
