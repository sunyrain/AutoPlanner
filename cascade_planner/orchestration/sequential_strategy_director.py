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
import copy
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
from cascade_planner.application.chemistry_inspection import (
    inspect_mapped_smiles,
)
from cascade_planner.application.reaction_proof_versions import (
    active_reaction_proofs,
)
from cascade_planner.application.routejson_compiler import (
    MaterializedReaction,
    RouteJSONCompiler,
)
from cascade_planner.application.route_edge_scope import (
    route_family_scoped_edge_ids,
)
from cascade_planner.application.biocatalytic_step_contract import (
    BIOLOGICAL_EXECUTION_DOMAINS,
    normalize_biocatalytic_step,
    normalize_step_execution_domain,
)
from cascade_planner.application.strategy_contract import (
    KEY_EVENT_REPAIR_SCOPES,
    normalize_reaction_operations,
    normalize_key_event_repair_scope,
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
    worker_provider_failure_reason,
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
_BUILDER_INPUT_TOKEN_RESERVE = 24_000
_BUILDER_OUTPUT_TOKEN_RESERVE = 16_000
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

_PATH_REPAIR_ROUTE_STATE_KEYS = (
    "steps",
    "open_leaves",
    "open_leaf_states",
    "deferred_builder_leaf_states",
    "expanded_products",
    "complete_in_bound_stock",
    "aizynthfinder_strategy_search",
    "paper_policy_call_budget",
    "sidecar_durable_prefix_step_count",
    "sidecar_recovered_prefix",
    "strategy_milestone_cards",
    "strategy_milestone_attempts",
    "strategic_milestone_count",
    "strategy_anchor_diagnostics",
    "key_event_critic_completed",
    "routejson_replay_validation",
)


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
class FrontierBuilderContext:
    """One canonical target-to-leaf path for the next Builder call."""

    target_smiles: str
    route_family_id: str
    branch_index: int
    selected_product_smiles: str
    selected_product_mapped: str
    connected_steps: tuple[Mapping[str, Any], ...]
    strategy_card: Mapping[str, Any]
    reserved_atom_maps: tuple[int, ...]
    prior_rejections: tuple[Mapping[str, Any], ...] = ()
    attempt_index: int = 1
    pending_checkpoint_feedback: Mapping[str, Any] | None = None
    path_repair: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RevisionBoundRouteCriticContext:
    """One final, target-rooted route revision owned by the Route Critic."""

    target_smiles: str
    route_family_id: str
    route_sha256: str
    graph_revision: int
    branch_index: int
    edge_ids: tuple[str, ...]
    steps: tuple[Mapping[str, Any], ...]
    strategy_card: Mapping[str, Any]
    strategy_milestone_cards: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class _RouteLineageContext:
    """Host-owned projection of one selected mapped leaf's route lineage.

    Strategy and Builder calls reason about one molecule occurrence, not the
    append order of the whole route DAG.  The complete DAG remains the
    authority for route-level Critic and Editor calls; this projection carries
    only the selected leaf's exact ancestor spine plus its current split.
    """

    selected_product_smiles: str
    selected_product_mapped: str
    connected_steps: tuple[Mapping[str, Any], ...]
    reaction_spine: tuple[Mapping[str, Any], ...]
    ancestor_smiles: tuple[str, ...]
    current_split_context: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _NodeCallBudget:
    model_invocations: int
    input_tokens: int
    output_tokens: int
    wall_time_s: float


@dataclass(frozen=True, slots=True)
class _KeyEventReviewDisposition:
    """Host decision produced by one selected-path follow-up Critic call."""

    status: str = "not_applicable"
    rejected_path_step_ids: tuple[str, ...] = ()
    rejection_reason: str = ""

    @property
    def rejected(self) -> bool:
        return self.status == "rejected" and bool(self.rejected_path_step_ids)


@dataclass(frozen=True, slots=True)
class _ModelCallReservation:
    input_tokens: int
    output_tokens: int


class _SharedModelCallLedger:
    """Atomically share one Director model budget across branch workers.

    Only calls that are actually about to run are reserved.  The ledger keeps
    a small protected balance for mandatory final Critic calls, but it does
    not pre-allocate hypothetical Editor rounds or private token pools to
    branches.  Actual provider usage is settled immediately, so unused
    capacity is visible to every branch.
    """

    def __init__(
        self,
        quota: _NodeCallBudget,
        records: Iterable[WorkerRunRecord],
        *,
        protected_model_invocations: int = 0,
        protected_input_tokens: int = 0,
        protected_output_tokens: int = 0,
    ) -> None:
        usage = _aggregate_usage(records, elapsed_s=0.0)
        self._quota = quota
        self._protected_model_invocations = max(0, int(protected_model_invocations))
        self._protected_input_tokens = max(0, int(protected_input_tokens))
        self._protected_output_tokens = max(0, int(protected_output_tokens))
        self._committed_model_invocations = int(usage["model_invocations"])
        self._committed_input_tokens = int(usage["input_tokens"])
        self._committed_output_tokens = int(usage["output_tokens"])
        self._inflight_model_invocations = 0
        self._inflight_input_tokens = 0
        self._inflight_output_tokens = 0
        self._lock = threading.Lock()

    def reserve(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[_ModelCallReservation | None, str]:
        requested_input = max(0, int(input_tokens))
        requested_output = max(0, int(output_tokens))
        with self._lock:
            if (
                self._committed_model_invocations
                + self._inflight_model_invocations
                + self._protected_model_invocations
                >= self._quota.model_invocations
            ):
                return None, "model_invocation_allocation_exhausted"
            if (
                self._committed_input_tokens
                + self._inflight_input_tokens
                + self._protected_input_tokens
                + requested_input
                > self._quota.input_tokens
            ):
                return None, "input_token_allocation_exhausted"
            if (
                self._committed_output_tokens
                + self._inflight_output_tokens
                + self._protected_output_tokens
                + requested_output
                > self._quota.output_tokens
            ):
                return None, "output_token_allocation_exhausted"
            reservation = _ModelCallReservation(
                input_tokens=requested_input,
                output_tokens=requested_output,
            )
            self._inflight_model_invocations += 1
            self._inflight_input_tokens += requested_input
            self._inflight_output_tokens += requested_output
            return reservation, ""

    def settle(
        self,
        reservation: _ModelCallReservation,
        record: WorkerRunRecord | None,
    ) -> None:
        completed_model_turn = bool(
            record is not None and not worker_provider_failure_reason(record)
        )
        usage = dict(record.usage or {}) if record is not None else {}
        actual_input = max(
            0,
            int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        )
        actual_output = max(
            0,
            int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        )
        with self._lock:
            self._inflight_model_invocations -= 1
            self._inflight_input_tokens -= reservation.input_tokens
            self._inflight_output_tokens -= reservation.output_tokens
            if completed_model_turn:
                self._committed_model_invocations += 1
                self._committed_input_tokens += actual_input
                self._committed_output_tokens += actual_output

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "quota": {
                    "model_invocations": int(self._quota.model_invocations),
                    "input_tokens": int(self._quota.input_tokens),
                    "output_tokens": int(self._quota.output_tokens),
                },
                "committed": {
                    "model_invocations": self._committed_model_invocations,
                    "input_tokens": self._committed_input_tokens,
                    "output_tokens": self._committed_output_tokens,
                },
                "inflight": {
                    "model_invocations": self._inflight_model_invocations,
                    "input_tokens": self._inflight_input_tokens,
                    "output_tokens": self._inflight_output_tokens,
                },
                "protected_final_critics": {
                    "model_invocations": self._protected_model_invocations,
                    "input_tokens": self._protected_input_tokens,
                    "output_tokens": self._protected_output_tokens,
                },
                "semantics": {
                    "shared_across_branches": True,
                    "settled_from_actual_usage": True,
                    "hypothetical_editor_rounds_reserved": False,
                },
            }


@dataclass(frozen=True, slots=True)
class _CompiledReactionJsonCandidate:
    candidate_index: int
    candidate_id: str
    expansion: NodeExpansion
    score: float
    cost: float
    candidate_key: str


@dataclass(frozen=True, slots=True)
class _PathRepairSpan:
    rollback_start_step_id: str
    rebuild_through_step_id: str
    repair_goal: str
    active_constraints: tuple[str, ...]
    original_steps: tuple[Mapping[str, Any], ...]
    durable_steps: tuple[Mapping[str, Any], ...]
    removed_steps: tuple[Mapping[str, Any], ...]
    preserved_suffix_steps: tuple[Mapping[str, Any], ...]
    reconnect_boundaries: tuple[Mapping[str, Any], ...]
    suffix_reconnect_boundaries: tuple[Mapping[str, Any], ...]
    completion_boundaries: tuple[Mapping[str, Any], ...]
    final_open_boundaries: tuple[Mapping[str, Any], ...]
    repair_frontier_product_smiles: str
    repair_frontier_mapped_product_smiles: str
    open_leaf_states: tuple[Mapping[str, str], ...]
    reserved_atom_maps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PathRepairBlockerScope:
    """One topology- or chemistry-coupled component for a repair transaction."""

    selected_step_ids: tuple[str, ...]
    deferred_step_ids: tuple[str, ...]
    component_step_ids: tuple[tuple[str, ...], ...]


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
        self.aizynthfinder_strategy_stock_index = str(aizynthfinder_strategy_stock_index or "")
        # Test-only deterministic stock. Production paper runs bind the
        # content-addressed ZINC+eMolecules SQLite index above.
        self.aizynthfinder_strategy_inline_stock_smiles = tuple(
            str(value) for value in aizynthfinder_strategy_inline_stock_smiles if str(value)
        )
        self.worker_record_seed_path = str(worker_record_seed_path or "").strip()
        self.worker_record_seed_recovery_mode = str(worker_record_seed_recovery_mode or "").strip()
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
        self._provider_failure_lock = threading.Lock()
        self._provider_runtime_failure: dict[str, Any] = {}

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

    def frontier_prompt_for(
        self,
        context: FrontierBuilderContext,
        config: DirectorConfig,
    ) -> str:
        """Build the ordinary one-step policy prompt for one canonical leaf."""

        path_repair = dict(context.path_repair or {})
        return _node_prompt(
            target=context.target_smiles,
            branch_index=context.branch_index,
            lens=str(
                context.strategy_card.get("strategy_query")
                or "continue the retained target-rooted route"
            ),
            selected_product=context.selected_product_smiles,
            selected_product_mapped=context.selected_product_mapped,
            steps=context.connected_steps,
            open_leaves=(context.selected_product_smiles,),
            prior_rejections=context.prior_rejections,
            repair=bool(path_repair),
            strategy_card=context.strategy_card,
            forbidden_strategy_cards=(),
            host_failure_feedback={
                "pending_checkpoint_feedback": dict(context.pending_checkpoint_feedback or {}),
                "path_repair": path_repair,
            },
            max_reactionjson_candidates=1,
            paper_matched=config.paper_matched_reach_profile,
        )

    def run_frontier_builder_once(
        self,
        spec: AgentSpec,
        *,
        context: FrontierBuilderContext,
        config: DirectorConfig,
        prompt: str | None = None,
    ) -> tuple[dict[str, Any], WorkerRunRecord]:
        """Continue one canonical leaf without creating a new agent role.

        The caller owns the campaign action and shared model ledger.  This
        method owns only the same prompt, worker schema, ReactionJSON compiler,
        atom-map namespace, and cycle rules used by the ordinary Builder.
        """

        self._prepare_worker_record_journal(spec)
        objective = prompt or self.frontier_prompt_for(context, config)
        _assert_node_prompt_size(objective, config.max_node_prompt_bytes)
        task = _node_task(
            spec,
            prompt=objective,
            branch_index=context.branch_index,
            node_index=max(0, int(context.attempt_index) - 1),
            model=str(spec.metadata.get("model") or config.model or ""),
            reasoning_effort=str(
                spec.metadata.get("reasoning_effort") or config.reasoning_effort or "medium"
            ),
            timeout_s=config.max_node_call_timeout_s,
            paper_matched=config.paper_matched_reach_profile,
            target_smiles=context.target_smiles,
            selected_product=context.selected_product_smiles,
        )
        record = self._run_journaled_worker(self.node_executor, task)
        provider_failure_reason = worker_provider_failure_reason(record)
        if provider_failure_reason:
            return (
                {
                    "status": "runtime_unavailable",
                    "runtime_unavailable": True,
                    "runtime_pause": True,
                    "reason": provider_failure_reason,
                    "candidate_count": 0,
                    "model_call_consumed": False,
                    "model_output_validation": "provider_error",
                },
                record,
            )
        compiled, rejected = _reactionjson_candidates_from_record(
            record,
            expected_product=context.selected_product_smiles,
            mapped_product_smiles=context.selected_product_mapped,
            require_reaction_operations=config.require_strategy_graph_edits,
            compiler=self.routejson_compiler,
            max_candidates=1,
            reserved_atom_maps=context.reserved_atom_maps,
        )
        if not compiled:
            diagnostic = (
                dict(rejected[0]) if rejected else {"reason": "frontier_builder_candidate_missing"}
            )
            diagnostic.setdefault("product_smiles", context.selected_product_smiles)
            return (
                {
                    "status": "rejected",
                    "diagnostic": diagnostic,
                    "candidate_count": 0,
                    "model_output_validation": _model_output_validation_status(record),
                },
                record,
            )

        candidate = compiled[0]
        ancestors = {
            canonical
            for step in context.connected_steps
            if (canonical := _canonical_smiles(step.get("product_smiles")))
        }
        ancestors.add(context.selected_product_smiles)
        repeated = sorted(set(candidate.expansion.precursor_smiles) & ancestors)
        if repeated:
            return (
                {
                    "status": "rejected",
                    "diagnostic": {
                        "reason": "candidate_repeats_target_rooted_ancestor",
                        "product_smiles": context.selected_product_smiles,
                        "repeated_ancestor_smiles": repeated,
                        "attempted_net_edits": [
                            dict(row) for row in candidate.expansion.reaction_operations
                        ],
                    },
                    "candidate_count": 0,
                    "model_output_validation": _model_output_validation_status(record),
                },
                record,
            )

        step_identity = hashlib.sha256(
            (
                f"{spec.agent_id}\0{context.route_family_id}\0"
                f"{context.attempt_index}\0{candidate.candidate_key}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        step = _step_row(
            candidate.expansion,
            step_id=f"codex:frontier:{step_identity}",
        )
        return (
            {
                "status": "compiled",
                "step": step,
                "candidate_id": candidate.candidate_id,
                "candidate_count": 1,
                "rejected_candidates": [dict(row) for row in rejected],
                "model_output_validation": _model_output_validation_status(record),
            },
            record,
        )

    def final_route_critic_prompt_for(
        self,
        context: RevisionBoundRouteCriticContext,
        config: DirectorConfig,
    ) -> str | None:
        """Compile the ordinary Route Critic prompt for one final graph revision."""

        return _bounded_critic_prompt(
            target=context.target_smiles,
            branch_index=context.branch_index,
            strategy_card=context.strategy_card,
            strategy_milestone_cards=context.strategy_milestone_cards,
            steps=context.steps,
            maximum_bytes=config.max_node_prompt_bytes,
            paper_matched=config.paper_matched_reach_profile,
            audit_kind="final_route",
        )

    def run_final_route_critic_once(
        self,
        spec: AgentSpec,
        *,
        context: RevisionBoundRouteCriticContext,
        config: DirectorConfig,
        prompt: str | None = None,
    ) -> tuple[dict[str, Any], WorkerRunRecord]:
        """Run one Critic-only audit after a later Host route mutation.

        This is not another online checkpoint and cannot edit topology.  It
        closes the review ownership gap created when the unified recovery
        loop materializes a Builder edge after the Director's normal final
        Critic/Editor state machine has already settled.
        """

        resolved_prompt = prompt or self.final_route_critic_prompt_for(
            context,
            config,
        )
        if not resolved_prompt:
            raise ValueError("final_route_critic_prompt_unavailable")
        # Finalization may run after the Director's main call has settled or
        # in a Critic-only resume process with a fresh runner instance. Bind
        # the same durable journal explicitly so model I/O, exact replay, and
        # the canonical verdict cannot drift into separate histories.
        self._prepare_worker_record_journal(spec)
        task = _critic_task(
            spec,
            prompt=resolved_prompt,
            branch_index=context.branch_index,
            iteration=0,
            timeout_s=config.critic_call_timeout_s,
            paper_matched=config.paper_matched_reach_profile,
            target_smiles=context.target_smiles,
            audit_kind="final_route",
            task_id_override=spec.agent_id,
            route_steps=context.steps,
        )
        record = self._run_journaled_worker(self.critic_executor, task)
        return (
            _critique_from_record(
                record,
                route_steps=(context.steps if config.paper_matched_reach_profile else ()),
            ),
            record,
        )

    def run_final_route_repair_once(
        self,
        spec: AgentSpec,
        *,
        campaign_context: CampaignContext,
        route_context: RevisionBoundRouteCriticContext,
        critique: Mapping[str, Any],
        config: DirectorConfig,
        route_family_alias: str,
    ) -> dict[str, Any]:
        """Repair one revision-bound blocking route through the ordinary loop.

        The final Critic is an evaluator, not a topology writer. A concrete
        reject therefore enters the same Editor -> Builder -> Host replay ->
        re-Critic transaction used during route construction. This method
        returns a host-assembled plan only after that transaction commits; the
        caller remains responsible for canonical ingestion and for replacing
        the rejected route-family revision atomically.
        """

        self._prepare_worker_record_journal(spec)
        started = time.monotonic()
        records: list[WorkerRunRecord] = []
        blocking_steps = _blocking_critic_steps(critique, route_context.steps)
        if not blocking_steps:
            return {
                "status": "not_needed",
                "reason": "final_route_critic_has_no_blocking_step",
                "plan": None,
                "usage": _aggregate_usage(records, elapsed_s=0.0),
            }

        target = _canonical_smiles(route_context.target_smiles)
        mapped_target = str(
            next(
                (
                    row.get("mapped_product_smiles")
                    for row in route_context.steps
                    if _canonical_smiles(row.get("product_smiles")) == target
                    and str(row.get("mapped_product_smiles") or "")
                ),
                "",
            )
            or _mapped_smiles(target)
        )
        steps = [dict(row) for row in route_context.steps if isinstance(row, Mapping)]
        try:
            route_state = self.routejson_compiler.compile_route_graph_state(
                mapped_target_smiles=mapped_target,
                steps=steps,
                minimum_depth=1,
                rebase_materialized_local_maps=True,
            )
        except ReactionJsonReplayError as exc:
            return {
                "status": "repair_unresolved",
                "reason": f"final_route_repair_input_not_replayable:{exc}",
                "plan": None,
                "usage": _aggregate_usage(
                    records,
                    elapsed_s=time.monotonic() - started,
                ),
            }
        # Final-route repair starts from rows that were already accepted by
        # per-reaction Host replay.  Re-serialize the globally rebased replay
        # here so every later rollback, Editor span, and full-route re-Critic
        # consumes one coherent route-level atom-map namespace.
        steps = self.routejson_compiler.assemble_route(
            route_state.reactions,
            metadata=steps,
        )
        terminal_states = [
            {
                "smiles": value.product_smiles,
                "mapped_smiles": value.mapped_product_smiles,
            }
            for value in route_state.open_precursors
        ]
        membership = self._stock_membership(tuple(str(row["smiles"]) for row in terminal_states))
        strategy_card = dict(route_context.strategy_card or {})
        strategy_milestone_cards = [
            dict(row) for row in route_context.strategy_milestone_cards if isinstance(row, Mapping)
        ]
        unresolved_terminal_states = [
            row for row in terminal_states if membership.get(str(row["smiles"])) is not True
        ]
        branch: dict[str, Any] = {
            "branch_index": int(route_context.branch_index),
            "lens": str(
                strategy_card.get("strategy_query") or "repair the rejected final route revision"
            ),
            "strategy_seed": "",
            "strategy_seed_source": "final_route_critic",
            "strategy_seed_sha256": "",
            "steps": steps,
            "open_leaves": deque(row["smiles"] for row in unresolved_terminal_states),
            "open_leaf_states": deque(dict(row) for row in unresolved_terminal_states),
            "deferred_builder_leaf_states": deque(),
            "target_mapped_smiles": mapped_target,
            "expanded_products": {
                _canonical_smiles(row.get("product_smiles"))
                for row in steps
                if _canonical_smiles(row.get("product_smiles"))
            },
            "call_count": 0,
            "strategy_call_count": 0,
            "route_call_count": 0,
            "path_repair_builder_call_count": 0,
            "editor_attempt_count": 0,
            "editor_call_count": 0,
            "critic_call_count": 0,
            "rejections": [],
            "materialization_failures": {},
            "materialization_diagnostics": [],
            "materialization_editor_history": [],
            "complete_in_bound_stock": not bool(unresolved_terminal_states),
            "strategy_tree_engine": "aizynthfinder_mcts",
            "strategy_card": strategy_card,
            "root_strategy_card": (
                dict(strategy_milestone_cards[0]) if strategy_milestone_cards else strategy_card
            ),
            "strategy_milestone_cards": strategy_milestone_cards,
            "strategy_milestone_attempts": [],
            "strategy_milestone_generation_count": 0,
            "strategy_critic_call_count": 0,
            "strategy_critic": {},
            "key_event_critic_call_count": 0,
            "key_event_critic_completed": True,
            "key_event_critic_history": [],
            "pending_key_event_feedback": {},
            "chemical_critic": dict(critique),
            "critic_editor_history": [
                {
                    "iteration": -1,
                    "round": 0,
                    "source": "revision_bound_final_route_critic",
                    "critic_task_ids": [str(critique.get("critic_task_id") or "")],
                    "actual_critic_call_count": 0,
                    "editor_task_ids": [],
                    "actual_editor_call_count": 0,
                    "critic": dict(critique),
                }
            ],
            # A repair branch retains reconnectable steps from the original
            # route.  Its newly generated steps therefore need a distinct
            # identity namespace even when the local Builder counter starts
            # again at one.
            "generated_step_id_prefix": (
                f"codex:repair:{route_context.route_sha256[:12]}:"
                f"attempt:{max(1, int(spec.attempt))}:"
                f"branch:{int(route_context.branch_index) + 1}"
            ),
            "route_family_alias_override": str(route_family_alias),
            "supersedes_route_family_id": str(route_context.route_family_id),
            "repair_origin_route_sha256": str(route_context.route_sha256),
        }
        quota = _node_call_budget(spec, mode="final_route_repair", config=config)
        repaired = self._repair_branch_transactionally(
            spec,
            target=target,
            branch=branch,
            blocking_steps=blocking_steps,
            critique=critique,
            iteration=-1,
            records=records,
            max_prompt_bytes=config.max_node_prompt_bytes,
            max_node_call_timeout_s=config.max_node_call_timeout_s,
            quota=quota,
            started=started,
            reserve_model_invocations=1,
            reserve_input_tokens=_CRITIC_INPUT_TOKEN_RESERVE,
            reserve_output_tokens=_CRITIC_OUTPUT_TOKEN_RESERVE,
            reserve_wall_time_s=config.critic_call_timeout_s,
            config=config,
        )
        if self._provider_runtime_failure_snapshot():
            return {
                "status": "runtime_unavailable",
                "reason": "model_provider_unavailable_during_final_route_repair",
                "plan": None,
                "branch": branch,
                "usage": _aggregate_usage(
                    records,
                    elapsed_s=time.monotonic() - started,
                ),
            }
        if not repaired:
            return {
                "status": "repair_unresolved",
                "reason": "transactional_final_route_repair_not_rebuilt",
                "plan": None,
                "branch": branch,
                "usage": _aggregate_usage(
                    records,
                    elapsed_s=time.monotonic() - started,
                ),
            }

        repaired_branches = self._run_codex_critics(
            spec,
            campaign_context,
            [branch],
            records,
            quota=quota,
            started=started,
            config=config,
        )
        repaired_branch = repaired_branches[0]
        repaired_critique = dict(repaired_branch.get("chemical_critic") or {})
        remaining_blockers = _blocking_critic_steps(
            repaired_critique,
            repaired_branch.get("steps") or (),
        )
        committed = any(
            str(row.get("status") or "") == "committed_after_recritic"
            for row in repaired_branch.get("path_repair_transactions") or ()
            if isinstance(row, Mapping)
        )
        usage = _aggregate_usage(records, elapsed_s=time.monotonic() - started)
        if (
            self._provider_runtime_failure_snapshot()
            or str(repaired_critique.get("status") or "") == "unavailable"
        ):
            return {
                "status": "runtime_unavailable",
                "reason": "final_route_recritic_unavailable",
                "plan": None,
                "branch": repaired_branch,
                "usage": usage,
            }
        if remaining_blockers or not committed:
            return {
                "status": "repair_unresolved",
                "reason": "final_route_repair_not_accepted_by_recritic",
                "plan": None,
                "branch": repaired_branch,
                "usage": usage,
            }

        plan = _compile_plan(
            campaign_context,
            mode="event_replan",
            branches=[repaired_branch],
            requested_branch_count=1,
        )
        return {
            "status": "repaired",
            "reason": "transactional_repair_committed_after_recritic",
            "plan": plan,
            "branch": repaired_branch,
            "usage": usage,
        }

    def __call__(
        self,
        spec: AgentSpec,
        context: CampaignContext,
        mode: str,
        config: DirectorConfig,
    ) -> AgentResult:
        started = time.monotonic()
        with self._provider_failure_lock:
            self._provider_runtime_failure = {}
        target = _canonical_smiles(context.target.get("canonical_smiles"))
        if config.paper_matched_reach_profile:
            _preflight_paper_matched_worker_schemas(
                spec,
                target=target,
                config=config,
            )
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
        provider_runtime_failure = self._provider_runtime_failure_snapshot()
        if not provider_runtime_failure:
            branches = self._run_codex_critics(
                spec,
                context,
                branches,
                records,
                quota=quota,
                started=started,
                config=config,
            )
            provider_runtime_failure = self._provider_runtime_failure_snapshot()
        usage = _aggregate_usage(records, elapsed_s=time.monotonic() - started)
        usage["durable_worker_record_journal"] = bool(self._worker_record_journal_path is not None)
        usage["replayed_worker_record_count"] = int(self._replayed_worker_record_count)
        usage["seeded_worker_record_count"] = int(self._seeded_worker_record_count)
        usage["worker_record_seed_used"] = bool(self._seeded_worker_record_count)
        usage["worker_record_seed_recovery_mode"] = (
            self.worker_record_seed_recovery_mode if self._seeded_worker_record_count else ""
        )
        usage["exact_seed_replay_count"] = int(self._exact_seed_replay_count)
        usage["critic_unavailable_branch_count"] = sum(
            bool(branch.get("steps"))
            and str(dict(branch.get("chemical_critic") or {}).get("status") or "") == "unavailable"
            for branch in branches
        )
        usage["critic_rejected_branch_count"] = sum(
            str(dict(branch.get("chemical_critic") or {}).get("status") or "") == "reject"
            for branch in branches
        )
        usage["accepted_expansions"] = (
            len(branches)
            if mode == "event_replan"
            else sum(len(branch.get("steps") or []) for branch in branches)
        )
        initial_builder_calls = sum(int(branch.get("route_call_count") or 0) for branch in branches)
        repair_builder_calls = sum(
            int(branch.get("path_repair_builder_call_count") or 0) for branch in branches
        )
        usage["actual_route_builder_policy_calls"] = initial_builder_calls + repair_builder_calls
        usage["actual_initial_route_builder_policy_calls"] = initial_builder_calls
        usage["actual_path_repair_builder_calls"] = repair_builder_calls
        usage["actual_critic_calls"] = sum(
            int(branch.get("critic_call_count") or 0) for branch in branches
        )
        usage["actual_strategy_critic_calls"] = max(
            (int(branch.get("strategy_critic_call_count") or 0) for branch in branches),
            default=0,
        )
        usage["actual_key_event_critic_calls"] = sum(
            int(branch.get("key_event_critic_call_count") or 0) for branch in branches
        )
        usage["actual_editor_calls"] = sum(
            int(branch.get("editor_attempt_count") or 0) for branch in branches
        )
        usage["strategy_branch_workers"] = int(config.strategy_branch_workers)
        usage["strategic_milestone_limit_per_branch"] = int(
            config.max_strategic_milestones_per_branch
        )
        usage["upstream_strategy_milestone_calls"] = sum(
            max(0, int(branch.get("strategy_call_count") or 0) - 1) for branch in branches
        )
        usage["realized_strategic_milestones"] = sum(
            int(branch.get("strategic_milestone_count") or 0) for branch in branches
        )
        usage["stop_on_first_stock_closed_branch"] = bool(config.stop_on_first_stock_closed_branch)
        usage["stock_closed_branch_count"] = sum(
            _branch_stock_closed(branch) for branch in branches
        )
        usage["stock_closed_early_stop_triggered"] = any(
            bool(branch.get("portfolio_early_stop_triggered")) for branch in branches
        )
        # SynthEx reports 25 Route Builder steps as a search ceiling.  Only a
        # Host/AiZ search termination may end a branch earlier; stock closure
        # remains a Host observation.
        paper_policy_branches = [
            branch
            for branch in branches
            if str(branch.get("strategy_tree_engine") or "") == "aizynthfinder_mcts"
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
            elif dict(branch.get("aizynthfinder_strategy_search") or {}).get("failed"):
                sidecar = dict(branch.get("aizynthfinder_strategy_search") or {})
                branch["paper_policy_budget_failure"] = {
                    "reason": "paper_strategy_sidecar_failed",
                    "required_calls": int(config.max_node_expansions_per_branch),
                    "actual_calls": int(branch.get("route_call_count") or 0),
                    "stock_closed": False,
                    "detail": str(sidecar.get("error") or "")[:800],
                }
        usage["paper_policy_call_budget"] = {
            "maximum_per_branch": int(config.max_node_expansions_per_branch),
            "branch_count": len(paper_policy_branches),
            "actual_calls": [
                int(branch.get("route_call_count") or 0) for branch in paper_policy_branches
            ],
            "branch_summaries": [
                {
                    "branch_index": int(branch.get("branch_index") or 0) + 1,
                    "actual_policy_calls": int(branch.get("route_call_count") or 0),
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
                bool(_branch_stock_closed(branch)) for branch in paper_policy_branches
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
        if provider_runtime_failure:
            usage["provider_runtime_failure"] = provider_runtime_failure
            usage["provider_runtime_failure_did_not_consume_semantic_budget"] = True
            usage["resume_required_task_ids"] = list(
                provider_runtime_failure.get("resume_required_task_ids")
                or [provider_runtime_failure.get("task_id")]
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
        strict_policy_failure = any(
            branch.get("paper_policy_budget_failure") for branch in paper_policy_branches
        )
        usable: list[dict[str, Any]] = []
        plan_branches: list[dict[str, Any]] = []
        branch_route_retention: list[dict[str, Any]] = []
        for branch in branches:
            steps = [dict(row) for row in branch.get("steps") or [] if isinstance(row, Mapping)]
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
                        dict(branch.get("paper_policy_budget_failure") or {}).get("reason")
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
        if not usable and not provider_runtime_failure:
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
        if provider_runtime_failure:
            # A transient provider outage is an operational pause, not a
            # scientific rollback. Every completed worker record is already
            # fsync'd by stable task id; returning the deterministic Host
            # projection lets the outer Kernel checkpoint the recoverable
            # prefix while keeping the Director reservation in flight. A
            # resume rebuilds this state from the journal and calls only tasks
            # whose successful record is absent.
            return _agent_result(
                spec,
                state=AgentState.FAILED,
                output=plan,
                usage=usage,
                error=(
                    "model_provider_unavailable:"
                    + str(provider_runtime_failure.get("reason") or "provider_unavailable")
                ),
                mode=mode,
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
                    row.get("schema_version") != "sequential_director_worker_record.v1"
                    or not all(key)
                    or not record_row
                ):
                    continue
                record = WorkerRunRecord(**record_row)
                # Cancellation is an interrupted execution boundary, not a
                # completed model result.  Keep its journal row as incident
                # history, but never replay the empty/cancelled record on a
                # resume; the smallest interrupted worker call must run again.
                if record.status == "cancelled" or worker_provider_failure_reason(record):
                    continue
                self._worker_record_cache[key] = record
                if seeded:
                    portable_digest = str(row.get("portable_model_input_sha256") or "")
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
                            dict(record.metadata or {}).get("event_log_path") or ""
                        ).strip()
                        if event_log_path and str(record.task_id or ""):
                            provenance_model_io = (
                                Path(event_log_path).expanduser().parent.parent / "model-io.jsonl"
                            )
                            self._load_seed_model_input_journal(provenance_model_io)
                            model_input = self._seed_model_inputs_by_task_id.get(
                                str(record.task_id or "")
                            )
                    if portable_digest:
                        self._seed_worker_records_by_model_input[portable_digest] = record
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
                and self.worker_record_seed_recovery_mode == "exact_model_io_v1"
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
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": "model_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        provider_failure_reason = worker_provider_failure_reason(record)
        if provider_failure_reason:
            self._record_provider_runtime_failure(
                reason=provider_failure_reason,
                task_id=task.task_id,
                task_type=task.task_type,
            )
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

    def _record_provider_runtime_failure(
        self,
        *,
        reason: str,
        task_id: str,
        task_type: str,
    ) -> dict[str, Any]:
        failure = {
            "reason": str(reason or "provider_unavailable"),
            "task_id": str(task_id or ""),
            "task_type": str(task_type or ""),
            "retryable_after_external_recovery": True,
            "semantic_budget_consumed": False,
        }
        with self._provider_failure_lock:
            if not self._provider_runtime_failure:
                self._provider_runtime_failure = dict(failure)
            failures = [
                dict(row)
                for row in self._provider_runtime_failure.get("failures") or []
                if isinstance(row, Mapping)
            ]
            failure_key = (failure["task_id"], failure["reason"])
            if failure_key not in {
                (str(row.get("task_id") or ""), str(row.get("reason") or "")) for row in failures
            }:
                failures.append(failure)
            self._provider_runtime_failure["failures"] = failures
            self._provider_runtime_failure["resume_required_task_ids"] = sorted(
                {str(row.get("task_id") or "") for row in failures if str(row.get("task_id") or "")}
            )
            return dict(self._provider_runtime_failure)

    def _provider_runtime_failure_snapshot(self) -> dict[str, Any]:
        with self._provider_failure_lock:
            return dict(self._provider_runtime_failure)

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

    @staticmethod
    def _path_repair_route_snapshot(
        branch: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            key: copy.deepcopy(branch[key])
            for key in _PATH_REPAIR_ROUTE_STATE_KEYS
            if key in branch
        }

    @staticmethod
    def _restore_path_repair_route_snapshot(
        branch: dict[str, Any],
        snapshot: Mapping[str, Any],
    ) -> None:
        for key in _PATH_REPAIR_ROUTE_STATE_KEYS:
            if key in snapshot:
                branch[key] = copy.deepcopy(snapshot[key])
            else:
                branch.pop(key, None)

    @staticmethod
    def _path_repair_recritic_summary(
        critique: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": str(critique.get("status") or ""),
            "overall_assessment": str(critique.get("overall_assessment") or ""),
            "strategy_adherence": critique.get("strategy_adherence") is True,
            "critic_task_id": str(critique.get("critic_task_id") or ""),
            "blocking_step_ids": [
                str(row.get("step_id") or "")
                for row in critique.get("step_assessments") or ()
                if isinstance(row, Mapping) and row.get("blocking") is True
            ],
        }

    def _finalize_pending_path_repair(
        self,
        branch: dict[str, Any],
        critique: Mapping[str, Any],
    ) -> bool:
        pending = branch.get("_pending_path_repair_transaction")
        if not isinstance(pending, Mapping):
            return False
        original_snapshot = dict(pending.get("route_snapshot") or {})
        if _branch_stock_closed(original_snapshot) and not _branch_stock_closed(branch):
            self._rollback_pending_path_repair(
                branch,
                reason="path_repair_stock_closure_regressed",
                candidate_critique=critique,
            )
            return False
        branch.pop("_pending_path_repair_transaction", None)
        transactions = branch.setdefault("path_repair_transactions", [])
        recritic = self._path_repair_recritic_summary(critique)
        for raw_index in pending.get("transaction_indices") or ():
            index = int(raw_index)
            if 0 <= index < len(transactions):
                transactions[index]["status"] = "committed_after_recritic"
                transactions[index]["recritic"] = dict(recritic)
        editor_task_ids = {
            str(value) for value in pending.get("editor_task_ids") or () if str(value)
        }
        for repair in branch.get("editor_repairs") or ():
            if (
                isinstance(repair, dict)
                and str(repair.get("editor_task_id") or "") in editor_task_ids
            ):
                repair["status"] = "committed_after_recritic"
        original_steps = [
            dict(row) for row in original_snapshot.get("steps") or () if isinstance(row, Mapping)
        ]
        if original_steps:
            branch.setdefault("route_alternatives", []).append(
                {
                    "reason": "pre_path_repair_authoritative_route",
                    "editor_task_ids": [
                        str(value) for value in pending.get("editor_task_ids") or () if str(value)
                    ],
                    "steps": original_steps,
                }
            )
        return True

    def _rollback_pending_path_repair(
        self,
        branch: dict[str, Any],
        *,
        reason: str,
        candidate_critique: Mapping[str, Any] | None = None,
    ) -> bool:
        pending = branch.pop("_pending_path_repair_transaction", None)
        if not isinstance(pending, Mapping):
            return False
        critique = dict(candidate_critique or branch.get("chemical_critic") or {})
        transactions = branch.setdefault("path_repair_transactions", [])
        recritic = self._path_repair_recritic_summary(critique)
        for raw_index in pending.get("transaction_indices") or ():
            index = int(raw_index)
            if 0 <= index < len(transactions):
                transactions[index]["status"] = "rolled_back_after_recritic"
                transactions[index]["reason"] = reason
                transactions[index]["recritic"] = dict(recritic)
        editor_task_ids = {
            str(value) for value in pending.get("editor_task_ids") or () if str(value)
        }
        for repair in branch.get("editor_repairs") or ():
            if (
                isinstance(repair, dict)
                and str(repair.get("editor_task_id") or "") in editor_task_ids
            ):
                repair["status"] = "rolled_back_after_recritic"
        self._restore_path_repair_route_snapshot(
            branch,
            dict(pending.get("route_snapshot") or {}),
        )
        original_critique = dict(pending.get("original_critique") or {})
        if original_critique:
            branch["chemical_critic"] = {
                **original_critique,
                "path_repair_outcome": "rolled_back_after_recritic",
                "path_repair_failure_reason": reason,
                "candidate_recritic": recritic,
            }
        branch.setdefault("editor_rejection_diagnostics", []).append(
            {
                "reason": reason,
                "transaction_status": "rolled_back_after_recritic",
                "candidate_recritic": recritic,
            }
        )
        return True

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
            if self._provider_runtime_failure_snapshot():
                return branches
            pending_online_repair = branch.pop("_pending_online_path_repair", None)
            if isinstance(pending_online_repair, Mapping):
                remaining_final_critics = sum(
                    bool(row.get("steps"))
                    or isinstance(row.get("_pending_online_path_repair"), Mapping)
                    for row in branches[index:]
                )
                self._repair_branch_transactionally(
                    spec,
                    target=target,
                    branch=branch,
                    blocking_steps=[
                        dict(row)
                        for row in pending_online_repair.get("blocking_steps") or []
                        if isinstance(row, Mapping)
                    ],
                    critique=dict(pending_online_repair.get("critique") or {}),
                    # Keep this task id distinct from the ordinary final
                    # Critic/Editor round zero while using the same transaction.
                    iteration=-1,
                    records=records,
                    max_prompt_bytes=config.max_node_prompt_bytes,
                    max_node_call_timeout_s=config.max_node_call_timeout_s,
                    quota=quota,
                    started=started,
                    reserve_model_invocations=remaining_final_critics,
                    reserve_input_tokens=(remaining_final_critics * _CRITIC_INPUT_TOKEN_RESERVE),
                    reserve_output_tokens=(remaining_final_critics * _CRITIC_OUTPUT_TOKEN_RESERVE),
                    reserve_wall_time_s=(remaining_final_critics * config.critic_call_timeout_s),
                    config=config,
                    repair_context_steps=[
                        dict(row)
                        for row in pending_online_repair.get("repair_context_steps") or []
                        if isinstance(row, Mapping)
                    ],
                    checkpoint_feedback=dict(
                        pending_online_repair.get("checkpoint_feedback") or {}
                    ),
                    repair_strategy_card=dict(pending_online_repair.get("strategy_card") or {}),
                )
                if self._provider_runtime_failure_snapshot():
                    return branches
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
                if self._provider_runtime_failure_snapshot():
                    return branches
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
                    reserve_calls_after_this_critic = reserve_critic_calls + reserve_editor_calls
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
                        reserve_calls_after_this_critic * maximum_repair_call_wall
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
                        * (_CRITIC_INPUT_TOKEN_RESERVE + _EDITOR_INPUT_TOKEN_RESERVE)
                    )
                    reserve_output_after_this_critic = (
                        _CRITIC_OUTPUT_TOKEN_RESERVE
                        + _EDITOR_OUTPUT_TOKEN_RESERVE
                        + future_families
                        * (_CRITIC_OUTPUT_TOKEN_RESERVE + _EDITOR_OUTPUT_TOKEN_RESERVE)
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
                    if not self._rollback_pending_path_repair(
                        branch,
                        reason=(f"path_repair_recritic_budget_exhausted:{budget_block_reason}"),
                    ):
                        branch["chemical_critic"] = _unavailable_critique(
                            f"critic_budget_exhausted:{budget_block_reason}"
                        )
                    break
                pending_repair = branch.get("_pending_path_repair_transaction")
                prompt = _bounded_critic_prompt(
                    target=target,
                    branch_index=int(branch.get("branch_index") or 0),
                    strategy_card=_final_route_strategy_card(branch),
                    strategy_milestone_cards=list(branch.get("strategy_milestone_cards") or []),
                    steps=list(branch.get("steps") or []),
                    maximum_bytes=config.max_node_prompt_bytes,
                    paper_matched=config.paper_matched_reach_profile,
                    repair_completion=(
                        pending_repair if isinstance(pending_repair, Mapping) else None
                    ),
                )
                if prompt is None:
                    # Prompt size is a runtime resource control, not a reason
                    # to erase every already-materialized route family.  If
                    # even the structure-only projection cannot fit, fail this
                    # Critic closed and allow the other independent families
                    # and the durable Director result to survive.
                    if not self._rollback_pending_path_repair(
                        branch,
                        reason="path_repair_recritic_prompt_byte_budget_exhausted",
                    ):
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
                            "branch_index": int(branch.get("branch_index") or 0) + 1,
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
                    route_steps=list(branch.get("steps") or []),
                )
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
                    if not self._rollback_pending_path_repair(
                        branch,
                        reason=(f"path_repair_recritic_execution_failed:{type(exc).__name__}"),
                    ):
                        branch["chemical_critic"] = _unavailable_critique(
                            f"critic_execution_failed:{type(exc).__name__}"
                        )
                    break
                records.append(record)
                if worker_provider_failure_reason(record):
                    return branches
                branch["critic_call_count"] = int(branch.get("critic_call_count") or 0) + 1
                critique = _critique_from_record(
                    record,
                    route_steps=(
                        list(branch.get("steps") or [])
                        if config.paper_matched_reach_profile
                        else ()
                    ),
                    required_step_ids=(
                        (str(pending_repair.get("required_checkpoint_step_id") or ""),)
                        if isinstance(pending_repair, Mapping)
                        and str(pending_repair.get("completion_mode") or "")
                        == "strategy_checkpoint"
                        else ()
                    ),
                )
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
                    self._rollback_pending_path_repair(
                        branch,
                        reason="path_repair_recritic_unavailable",
                        candidate_critique=critique,
                    )
                    break
                blocking_steps = _blocking_critic_steps(
                    critique,
                    list(branch.get("steps") or []),
                )
                if not blocking_steps:
                    completion_failure = _path_repair_recritic_completion_failure(
                        pending_repair,
                        critique,
                    )
                    if completion_failure:
                        self._rollback_pending_path_repair(
                            branch,
                            reason=completion_failure,
                            candidate_critique=critique,
                        )
                        break
                    self._finalize_pending_path_repair(branch, critique)
                    break
                pending_repair = branch.get("_pending_path_repair_transaction")
                if isinstance(pending_repair, Mapping):
                    component_resolved, component_diagnostic = (
                        _path_repair_component_recritic_result(
                            pending_repair,
                            blocking_steps,
                        )
                    )
                    if not component_resolved:
                        # A blocker in the rebuilt component, or any new
                        # blocker outside the explicitly deferred components,
                        # invalidates this candidate transaction.
                        self._rollback_pending_path_repair(
                            branch,
                            reason="path_repair_recritic_rejected",
                            candidate_critique={
                                **critique,
                                "path_repair_component_diagnostic": (component_diagnostic),
                            },
                        )
                        break
                    # Independent sibling blockers do not invalidate a
                    # successfully rebuilt component. Commit this atomic
                    # change, then repair the deferred component in a later
                    # round of the same Critic/Editor state machine.
                    self._finalize_pending_path_repair(branch, critique)
                if iteration >= max_rounds:
                    exhausted_critique = {
                        **critique,
                        "status": "reject",
                        "reason": "critic_editor_iteration_limit_reached",
                    }
                    branch["chemical_critic"] = exhausted_critique
                    self._rollback_pending_path_repair(
                        branch,
                        reason="path_repair_recritic_iteration_limit_reached",
                        candidate_critique=exhausted_critique,
                    )
                    break
                records_before_editor = len(records)
                # A concrete rejection makes one Editor call eligible. Keep
                # only the re-Critic that would validate an applied repair,
                # plus the first Critic for each untouched later route.
                repair_critic_reserve_calls = 1 + future_families
                repair_critic_reserve_input = (
                    repair_critic_reserve_calls * _CRITIC_INPUT_TOKEN_RESERVE
                )
                repair_critic_reserve_output = (
                    repair_critic_reserve_calls * _CRITIC_OUTPUT_TOKEN_RESERVE
                )
                remaining_repair_wall = _remaining_node_wall_time(started, quota)
                repair_critic_reserve_wall = (
                    remaining_repair_wall
                    * repair_critic_reserve_calls
                    / (repair_critic_reserve_calls + 1)
                )
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
                    reserve_model_invocations=repair_critic_reserve_calls,
                    reserve_input_tokens=repair_critic_reserve_input,
                    reserve_output_tokens=repair_critic_reserve_output,
                    reserve_wall_time_s=repair_critic_reserve_wall,
                    config=config,
                )
                if self._provider_runtime_failure_snapshot():
                    return branches
                editor_task_ids = [
                    str(row.task_id)
                    for row in records[records_before_editor:]
                    if ":editor:" in str(row.task_id or "")
                ]
                branch["critic_editor_history"][-1]["editor_task_ids"] = editor_task_ids
                branch["critic_editor_history"][-1]["editor_call_count"] = len(editor_task_ids)
                branch["critic_editor_history"][-1]["actual_editor_call_count"] = len(
                    editor_task_ids
                )
                if not edited:
                    repair_transactions = [
                        dict(row)
                        for row in branch.get("path_repair_transactions") or []
                        if isinstance(row, Mapping)
                    ]
                    latest_repair = repair_transactions[-1] if repair_transactions else {}
                    if latest_repair.get("status") == ("retained_uncommitted_prefix"):
                        branch["chemical_critic"] = {
                            **critique,
                            "path_repair_outcome": ("retained_uncommitted_prefix"),
                            "path_repair_retention_reason": str(latest_repair.get("reason") or ""),
                        }
                        break
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
                        branch["chemical_critic"] = _unavailable_critique("editor_execution_failed")
                    self._rollback_pending_path_repair(
                        branch,
                        reason="path_repair_followup_editor_failed",
                        candidate_critique=dict(branch.get("chemical_critic") or {}),
                    )
                    break
            self._rollback_pending_path_repair(
                branch,
                reason="path_repair_loop_exited_without_accepted_recritic",
            )
        for branch in branches:
            cards = [
                dict(row)
                for row in branch.get("strategy_milestone_cards") or []
                if isinstance(row, Mapping)
            ]
            if not cards and isinstance(branch.get("strategy_card"), Mapping):
                cards = [dict(branch["strategy_card"])]
            _refresh_strategy_milestone_projection(
                branch,
                strategy_cards=cards,
                use_key_event_critic=config.enable_key_event_critic,
            )
        return branches

    def _repair_branch_transactionally(
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
        quota: _NodeCallBudget,
        started: float,
        reserve_model_invocations: int,
        reserve_input_tokens: int,
        reserve_output_tokens: int,
        reserve_wall_time_s: float,
        config: DirectorConfig,
        repair_context_steps: Iterable[Mapping[str, Any]] | None = None,
        checkpoint_feedback: Mapping[str, Any] | None = None,
        repair_strategy_card: Mapping[str, Any] | None = None,
    ) -> bool:
        """Execute one Editor -> Host rollback -> Builder transaction."""

        authoritative_steps = [
            dict(row) for row in branch.get("steps") or [] if isinstance(row, Mapping)
        ]
        steps = (
            [dict(row) for row in repair_context_steps if isinstance(row, Mapping)]
            if repair_context_steps is not None
            else [dict(row) for row in authoritative_steps]
        )
        if repair_context_steps is not None:
            authoritative_identities = [
                (
                    str(row.get("step_id") or ""),
                    _key_event_fingerprint(row),
                )
                for row in authoritative_steps
            ]
            context_prefix_identities = [
                (
                    str(row.get("step_id") or ""),
                    _key_event_fingerprint(row),
                )
                for row in steps[: len(authoritative_steps)]
            ]
            if (
                len(steps) <= len(authoritative_steps)
                or context_prefix_identities != authoritative_identities
            ):
                branch.setdefault("editor_rejection_diagnostics", []).append(
                    {
                        "reason": ("online_path_repair_context_not_authoritative_extension"),
                        "authoritative_step_ids": [value[0] for value in authoritative_identities],
                        "context_step_ids": [str(row.get("step_id") or "") for row in steps],
                    }
                )
                return False
        concrete_blockers = [dict(row) for row in blocking_steps if isinstance(row, Mapping)]
        builder_calls_before = int(branch.get("path_repair_builder_call_count") or 0)
        builder_call_ceiling = int(config.max_node_expansions_per_branch)
        remaining_builder_calls = max(
            0,
            builder_call_ceiling - builder_calls_before,
        )
        if remaining_builder_calls <= 0:
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "reason": "path_repair_builder_budget_exhausted_before_boundary",
                    "repair_builder_calls_before": builder_calls_before,
                    "builder_phase_call_ceiling": builder_call_ceiling,
                }
            )
            return False
        blocker_scope, scope_diagnostic = _select_path_repair_blocker_scope(
            current_steps=steps,
            mapped_target_smiles=str(branch.get("target_mapped_smiles") or _mapped_smiles(target)),
            blocking_steps=concrete_blockers,
        )
        if blocker_scope is None:
            branch.setdefault("editor_rejection_diagnostics", []).append(scope_diagnostic)
            return False
        selected_blocker_ids = set(blocker_scope.selected_step_ids)
        feedback = _compact_critic_feedback(
            critique,
            concrete_blockers,
            paper_matched=True,
        )
        checkpoint_feedback = dict(checkpoint_feedback or {})
        active_checkpoint_constraints = [
            dict(row)
            for row in checkpoint_feedback.get("active_constraints") or []
            if isinstance(row, Mapping)
        ]
        if active_checkpoint_constraints:
            feedback["active_checkpoint_constraints"] = active_checkpoint_constraints
        failure_basin = dict(checkpoint_feedback.get("failure_basin") or {})
        if failure_basin:
            feedback["failure_basin"] = failure_basin
        transaction_strategy_card = dict(repair_strategy_card or branch.get("strategy_card") or {})
        feedback["repair_transaction_scope"] = {
            "selected_blocker_step_ids": list(blocker_scope.selected_step_ids),
            "deferred_blocker_step_ids": list(blocker_scope.deferred_step_ids),
            "component_count": len(blocker_scope.component_step_ids),
        }
        prompt = _path_repair_editor_prompt(
            target=target,
            strategy_card=transaction_strategy_card,
            steps=steps,
            critic_feedback=feedback,
            provisional_rejected_step_ids=(
                [
                    str(row.get("step_id") or "")
                    for row in steps[len(authoritative_steps) :]
                    if str(row.get("step_id") or "")
                ]
                if repair_context_steps is not None
                else ()
            ),
        )
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "reason": "path_repair_editor_prompt_byte_budget_exhausted",
                    "route_step_count": len(steps),
                }
            )
            return False
        if not _node_budget_allows(
            records,
            started=started,
            quota=quota,
            reserve_model_invocations=reserve_model_invocations,
            reserve_input_tokens=reserve_input_tokens + _EDITOR_INPUT_TOKEN_RESERVE,
            reserve_output_tokens=reserve_output_tokens + _EDITOR_OUTPUT_TOKEN_RESERVE,
            reserve_wall_time_s=reserve_wall_time_s,
        ):
            return False
        task = _node_task(
            spec,
            prompt=prompt,
            branch_index=int(branch.get("branch_index") or 0),
            node_index=(iteration + 1) * _MATERIALIZATION_RETRY_LIMIT,
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(
                spec.metadata.get("editor_reasoning_effort")
                or spec.metadata.get("reasoning_effort")
                or "medium"
            ),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=min(
                    max_node_call_timeout_s,
                    max(
                        0.001,
                        _remaining_node_wall_time(started, quota)
                        - max(0.0, reserve_wall_time_s)
                        - _deadline_settlement_reserve_s(quota),
                    ),
                ),
            ),
            task_type="route_path_repair_directive",
            paper_matched=True,
            target_smiles=target,
            selected_product=str(
                next(
                    (
                        row.get("product_smiles")
                        for row in concrete_blockers
                        if str(row.get("step_id") or "") in selected_blocker_ids
                    ),
                    "",
                )
                or ""
            ),
        )
        try:
            record = self._run_journaled_worker(self.editor_executor, task)
        except Exception as exc:
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "task_id": task.task_id,
                    "reason": "path_repair_editor_worker_exception",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            return False
        records.append(record)
        if worker_provider_failure_reason(record):
            return False
        branch["editor_attempt_count"] = int(branch.get("editor_attempt_count") or 0) + 1
        directive, diagnostic = _path_repair_directive_from_record(record)
        if directive is None:
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {"task_id": task.task_id, **diagnostic}
            )
            return False
        checkpoint_constraint_summary = _path_repair_checkpoint_constraint_summary(
            checkpoint_feedback
        )
        if checkpoint_constraint_summary:
            directive["active_constraints"] = list(
                dict.fromkeys(
                    [
                        checkpoint_constraint_summary,
                        *[
                            str(value).strip()
                            for value in directive.get("active_constraints") or []
                            if str(value).strip()
                        ],
                    ]
                )
            )[:5]
        additional_coupled_ids = tuple(
            str(value)
            for value in directive.get("additional_coupled_blocker_step_ids") or ()
            if str(value)
        )
        unknown_coupled_ids = sorted(
            set(additional_coupled_ids) - set(blocker_scope.deferred_step_ids)
        )
        if unknown_coupled_ids:
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "task_id": task.task_id,
                    "reason": "path_repair_additional_coupled_blocker_invalid",
                    "step_ids": unknown_coupled_ids,
                }
            )
            return False
        effective_selected_step_ids = tuple(
            dict.fromkeys(
                [
                    *blocker_scope.selected_step_ids,
                    *additional_coupled_ids,
                ]
            )
        )
        effective_deferred_step_ids = tuple(
            value
            for value in blocker_scope.deferred_step_ids
            if value not in set(additional_coupled_ids)
        )
        rollback, diagnostic = _prepare_path_repair_span(
            current_steps=steps,
            mapped_target_smiles=str(branch.get("target_mapped_smiles") or _mapped_smiles(target)),
            directive=directive,
            blocking_step_ids=effective_selected_step_ids,
            deferred_blocking_step_ids=effective_deferred_step_ids,
        )
        if rollback is None:
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "task_id": task.task_id,
                    "repair_directive": dict(directive),
                    **diagnostic,
                }
            )
            return False
        boundary_preflight = _path_repair_boundary_preflight(
            rollback,
            preserved_suffix_compatible=(directive.get("preserved_suffix_compatible") is True),
        )
        if boundary_preflight:
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "task_id": task.task_id,
                    "repair_directive": dict(directive),
                    **boundary_preflight,
                }
            )
            return False

        route_snapshot = self._path_repair_route_snapshot(branch)

        def restore_route_snapshot() -> None:
            self._restore_path_repair_route_snapshot(branch, route_snapshot)

        branch["steps"] = [dict(row) for row in rollback.durable_steps]
        completion_mode = (
            "strategy_checkpoint" if repair_context_steps is not None else "cut_frontier"
        )
        membership = self._stock_membership(
            (
                *(
                    str(row.get("smiles") or "")
                    for row in rollback.open_leaf_states
                ),
                *(
                    str(row.get("product_smiles") or "")
                    for row in rollback.completion_boundaries
                ),
            )
        )
        branch["open_leaf_states"] = deque(
            dict(row)
            for row in rollback.open_leaf_states
            if membership.get(str(row.get("smiles") or "")) is not True
        )
        branch["deferred_builder_leaf_states"] = deque()
        branch["expanded_products"] = {
            _canonical_smiles(row.get("product_smiles"))
            for row in rollback.durable_steps
            if _canonical_smiles(row.get("product_smiles"))
        }
        branch["complete_in_bound_stock"] = False
        branch["sidecar_durable_prefix_step_count"] = 0
        active_reconnect_boundaries = (
            rollback.reconnect_boundaries
            if completion_mode == "cut_frontier"
            else rollback.suffix_reconnect_boundaries
        )
        search_completion_boundaries = tuple(
            dict(row)
            for row in rollback.completion_boundaries
            if membership.get(_canonical_smiles(row.get("product_smiles"))) is not True
        )
        branch["_path_repair_resume"] = {
            "rollback_start_step_id": rollback.rollback_start_step_id,
            "rebuild_through_step_id": rollback.rebuild_through_step_id,
            "repair_frontier_mapped_product_smiles": (
                rollback.repair_frontier_mapped_product_smiles
            ),
            "repair_goal": rollback.repair_goal,
            "active_constraints": list(rollback.active_constraints),
            "durable_steps": [dict(row) for row in rollback.durable_steps],
            "reconnect_boundaries": [
                dict(row) for row in active_reconnect_boundaries
            ],
            "search_completion_boundaries": [
                dict(row) for row in search_completion_boundaries
            ],
            # The accepted reaction spine ends at the rollback frontier, but
            # deleting the mutable span from the Builder context also deletes
            # the exact mapped graph program that the Editor chose to repair.
            # Preserve only that local Host-replayed span as non-authoritative
            # reference material.  The Builder can retain sound provenance and
            # operations while changing the Critic-identified defect, without
            # mistaking rolled-back rows for accepted history.
            "repair_reference_span": _path_repair_reference_rows(
                rollback.removed_steps,
                key_event_critic_history=branch.get("key_event_critic_history") or (),
            ),
            "reserved_atom_maps": list(rollback.reserved_atom_maps),
            # Completion is the invariant that triggered this transaction.
            # Final-route repairs must restore the complete Host-derived cut
            # frontier; online Key-Critic repairs continue through enabling
            # moves until the scheduled checkpoint candidate.
            "completion_mode": completion_mode,
            # A route-span repair mutates the implementation of one rejected
            # checkpoint, not the strategic question being tested.  Freeze the
            # exact Strategy used by that Critic until the rebuilt checkpoint
            # earns a fresh pass; rollback topology must not trigger a new
            # receding-horizon Strategy inside the transaction.
            "strategy_card": transaction_strategy_card,
        }
        if completion_mode == "strategy_checkpoint":
            # The rejected checkpoint is no longer selected in the provisional
            # transaction.  A replacement must earn a fresh Key-Critic pass;
            # the historical rows remain the only memory authority.
            branch["key_event_critic_completed"] = False
        _sync_open_leaf_projection(branch)
        # Repair calls use the normal per-branch expansion ceiling and remain
        # cumulative across Editor transactions.  The global ledger is the
        # only additional resource authority.
        try:
            if str(branch.get("strategy_tree_engine") or "") != "aizynthfinder_mcts":
                raise RuntimeError("transactional_path_repair_requires_aizynthfinder_mcts")
            repair_records = self._expand_seeded_branches_aizynthfinder(
                spec,
                target=target,
                seeded=[branch],
                existing_records=records,
                route_quota=quota,
                critic_editor_call_reserve=reserve_model_invocations,
                critic_input_reserve=reserve_input_tokens,
                critic_output_reserve=reserve_output_tokens,
                config=config,
                started=started,
            )
            records.extend(repair_records)
        except Exception as exc:
            branch.pop("_path_repair_resume", None)
            restore_route_snapshot()
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "task_id": task.task_id,
                    "reason": "path_repair_builder_execution_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            return False

        branch.pop("_path_repair_resume", None)
        builder_steps = [dict(row) for row in branch.get("steps") or [] if isinstance(row, Mapping)]
        durable_count = len(rollback.durable_steps)
        durable_ids = [str(row.get("step_id") or "") for row in rollback.durable_steps]
        rebuilt_prefix_ids = [
            str(row.get("step_id") or "") for row in builder_steps[:durable_count]
        ]
        added_steps = builder_steps[durable_count:]
        builder_calls_after = int(branch.get("path_repair_builder_call_count") or 0)
        builder_phase_budget_respected = builder_calls_after <= builder_call_ceiling
        required_checkpoint_step_id = next(
            (
                str(row.get("step_id") or "")
                for row in reversed(added_steps)
                if str(row.get("checkpoint_relation") or "") == "executes_checkpoint"
            ),
            "",
        )
        boundary_rebuilt = bool(
            added_steps
            and _canonical_mapped_smiles(added_steps[0].get("mapped_product_smiles"))
            == _canonical_mapped_smiles(rollback.repair_frontier_mapped_product_smiles)
        )
        stitch_diagnostic: dict[str, Any] = {
            "suffix_stitched": False,
            "boundary_count": 0,
        }
        rebuilt_steps = builder_steps
        try:
            pre_stitch_state = self.routejson_compiler.compile_route_graph_state(
                mapped_target_smiles=str(
                    branch.get("target_mapped_smiles") or _mapped_smiles(target)
                ),
                steps=builder_steps,
                minimum_depth=1,
            )
        except ReactionJsonReplayError:
            pre_stitch_state = None
        if completion_mode == "cut_frontier":
            completion_boundary_reached = bool(
                pre_stitch_state is not None
                and _path_repair_frontier_reaches_boundaries(
                    product_smiles=(
                        row.product_smiles for row in pre_stitch_state.open_precursors
                    ),
                    mapped_product_smiles=(
                        row.mapped_product_smiles
                        for row in pre_stitch_state.open_precursors
                    ),
                    reconnect_boundaries=rollback.completion_boundaries,
                )
            )
        else:
            completion_boundary_reached = _path_repair_completion_reached(
                added_steps,
                completion_mode=completion_mode,
            )
        if rollback.preserved_suffix_steps and completion_boundary_reached:
            stitched_steps, stitch_diagnostic = _stitch_path_repair_suffix(
                mapped_target_smiles=str(
                    branch.get("target_mapped_smiles") or _mapped_smiles(target)
                ),
                rebuilt_steps=builder_steps,
                preserved_suffix_steps=rollback.preserved_suffix_steps,
                reconnect_boundaries=rollback.suffix_reconnect_boundaries,
            )
            if stitched_steps is not None:
                rebuilt_steps = stitched_steps
                branch["steps"] = [dict(row) for row in rebuilt_steps]
                stitched_state = self.routejson_compiler.compile_route_graph_state(
                    mapped_target_smiles=str(
                        branch.get("target_mapped_smiles") or _mapped_smiles(target)
                    ),
                    steps=rebuilt_steps,
                    minimum_depth=1,
                )
                membership = self._stock_membership(
                    tuple(row.product_smiles for row in stitched_state.open_precursors)
                )
                branch["open_leaf_states"] = deque(
                    {
                        "smiles": row.product_smiles,
                        "mapped_smiles": row.mapped_product_smiles,
                    }
                    for row in stitched_state.open_precursors
                    if membership.get(row.product_smiles) is not True
                )
                branch["expanded_products"] = {
                    _canonical_smiles(row.get("product_smiles"))
                    for row in rebuilt_steps
                    if _canonical_smiles(row.get("product_smiles"))
                }
                branch["complete_in_bound_stock"] = not bool(branch["open_leaf_states"])
                _sync_open_leaf_projection(branch)
        elif rollback.preserved_suffix_steps:
            stitch_diagnostic = {
                "suffix_stitched": False,
                "boundary_count": len(rollback.suffix_reconnect_boundaries),
                "reason": "path_repair_cut_frontier_not_reached",
            }
        replay_validation = _route_steps_host_replay_validation(
            rebuilt_steps,
            mapped_target_smiles=str(branch.get("target_mapped_smiles") or _mapped_smiles(target)),
        )
        structural_rebuild_complete = bool(
            rebuilt_prefix_ids == durable_ids
            and boundary_rebuilt
            and builder_phase_budget_respected
            and replay_validation.get("complete") is True
        )
        final_frontier_restored = completion_mode != "cut_frontier"
        if structural_rebuild_complete and completion_mode == "cut_frontier":
            try:
                final_state = self.routejson_compiler.compile_route_graph_state(
                    mapped_target_smiles=str(
                        branch.get("target_mapped_smiles") or _mapped_smiles(target)
                    ),
                    steps=rebuilt_steps,
                    minimum_depth=1,
                )
            except ReactionJsonReplayError:
                final_state = None
            final_frontier_restored = bool(
                final_state is not None
                and _path_repair_frontier_reaches_boundaries(
                    product_smiles=(
                        row.product_smiles for row in final_state.open_precursors
                    ),
                    mapped_product_smiles=(
                        row.mapped_product_smiles for row in final_state.open_precursors
                    ),
                    reconnect_boundaries=rollback.final_open_boundaries,
                )
            )
            if final_state is not None:
                final_membership = self._stock_membership(
                    row.product_smiles for row in final_state.open_precursors
                )
                branch["open_leaf_states"] = deque(
                    {
                        "smiles": row.product_smiles,
                        "mapped_smiles": row.mapped_product_smiles,
                    }
                    for row in final_state.open_precursors
                    if final_membership.get(row.product_smiles) is not True
                )
                branch["complete_in_bound_stock"] = not bool(
                    branch["open_leaf_states"]
                )
                _sync_open_leaf_projection(branch)
        ready_for_recritic = bool(
            structural_rebuild_complete
            and completion_boundary_reached
            and final_frontier_restored
            and (
                stitch_diagnostic.get("suffix_stitched") is True
                if rollback.preserved_suffix_steps
                else True
            )
        )
        transaction = {
            "transaction_index": len(branch.get("path_repair_transactions") or []) + 1,
            "iteration": iteration,
            "editor_task_id": task.task_id,
            "rollback_start_step_id": rollback.rollback_start_step_id,
            "rebuild_through_step_id": rollback.rebuild_through_step_id,
            "removed_step_ids": [str(row.get("step_id") or "") for row in rollback.removed_steps],
            "durable_step_ids": durable_ids,
            "repair_goal": rollback.repair_goal,
            "active_constraints": list(rollback.active_constraints),
            "repair_frontier_product_smiles": rollback.repair_frontier_product_smiles,
            "repair_frontier_mapped_product_smiles": (
                rollback.repair_frontier_mapped_product_smiles
            ),
            "reconnect_boundaries": [dict(row) for row in rollback.reconnect_boundaries],
            "builder_calls": max(
                0,
                builder_calls_after - builder_calls_before,
            ),
            "repair_builder_calls_before": builder_calls_before,
            "repair_builder_calls_after": builder_calls_after,
            "builder_phase_call_ceiling": builder_call_ceiling,
            "builder_phase_budget_respected": builder_phase_budget_respected,
            "selected_blocker_step_ids": list(effective_selected_step_ids),
            "deferred_blocker_step_ids": list(effective_deferred_step_ids),
            "added_step_count": len(added_steps),
            "boundary_rebuilt": boundary_rebuilt,
            "completion_boundary_reached": completion_boundary_reached,
            "final_frontier_restored": final_frontier_restored,
            "completion_mode": completion_mode,
            "required_checkpoint_step_id": required_checkpoint_step_id,
            "preserved_suffix_step_ids": [
                str(row.get("step_id") or "") for row in rollback.preserved_suffix_steps
            ],
            "suffix_stitch": dict(stitch_diagnostic),
            "routejson_replay_validation": dict(replay_validation),
        }
        if not structural_rebuild_complete:
            transaction["status"] = "rolled_back_uncommitted"
            transaction["reason"] = "path_repair_structural_rebuild_invalid"
            restore_route_snapshot()
            branch.setdefault("path_repair_transactions", []).append(transaction)
            branch.setdefault("editor_rejection_diagnostics", []).append(
                {
                    "task_id": task.task_id,
                    "reason": "path_repair_structural_rebuild_invalid",
                    "boundary_rebuilt": boundary_rebuilt,
                    "builder_phase_budget_respected": (builder_phase_budget_respected),
                    "repair_builder_calls_after": builder_calls_after,
                    "builder_phase_call_ceiling": builder_call_ceiling,
                    "added_step_count": len(added_steps),
                    "required_replacement_depth": len(rollback.removed_steps),
                    "suffix_stitch": dict(stitch_diagnostic),
                    "routejson_replay_validation": dict(replay_validation),
                }
            )
            return False

        if not ready_for_recritic:
            failure_reason = (
                "path_repair_cut_frontier_not_reached"
                if completion_mode == "cut_frontier"
                and not completion_boundary_reached
                else "path_repair_strategy_checkpoint_not_reached"
                if completion_mode == "strategy_checkpoint"
                and not completion_boundary_reached
                else "path_repair_final_frontier_not_restored"
                if not final_frontier_restored
                else str(stitch_diagnostic.get("reason") or "")
            )
            boundary_not_reached = bool(
                structural_rebuild_complete
                and failure_reason
                in {
                    "path_repair_cut_frontier_not_reached",
                    "path_repair_reconnect_boundary_not_reached",
                    "path_repair_strategy_checkpoint_not_reached",
                }
            )
            transaction["status"] = (
                "retained_uncommitted_prefix" if boundary_not_reached else "rolled_back_uncommitted"
            )
            transaction["reason"] = failure_reason or "path_repair_completion_not_reached"
            if boundary_not_reached:
                # A replayable but incomplete prefix remains diagnostic-only;
                # the old route is still the single authority.
                transaction["provisional_steps"] = [dict(row) for row in rebuilt_steps]
                transaction["provisional_open_leaf_states"] = [
                    dict(row)
                    for row in branch.get("open_leaf_states") or []
                    if isinstance(row, Mapping)
                ]
            else:
                branch.setdefault("editor_rejection_diagnostics", []).append(
                    {
                        "task_id": task.task_id,
                        "reason": transaction["reason"],
                        "suffix_stitch": dict(stitch_diagnostic),
                    }
                )
            restore_route_snapshot()
            branch.setdefault("path_repair_transactions", []).append(transaction)
            return False

        transaction["status"] = "rebuilt_pending_recritic"
        transactions = branch.setdefault("path_repair_transactions", [])
        transactions.append(transaction)
        pending = branch.get("_pending_path_repair_transaction")
        if not isinstance(pending, dict):
            pending = {
                "route_snapshot": copy.deepcopy(route_snapshot),
                "original_critique": copy.deepcopy(dict(critique)),
                "transaction_indices": [],
                "editor_task_ids": [],
                "selected_blocker_step_ids": list(effective_selected_step_ids),
                "deferred_blocker_step_ids": list(effective_deferred_step_ids),
                "completion_mode": completion_mode,
                "required_checkpoint_step_id": required_checkpoint_step_id,
                "active_constraints": list(rollback.active_constraints),
            }
            branch["_pending_path_repair_transaction"] = pending
        pending.setdefault("transaction_indices", []).append(len(transactions) - 1)
        pending.setdefault("editor_task_ids", []).append(task.task_id)
        branch["call_count"] = int(branch.get("call_count") or 0) + 1
        branch["editor_call_count"] = int(branch.get("editor_call_count") or 0) + 1
        branch.setdefault("editor_repairs", []).append(
            {
                "iteration": iteration,
                "editor_task_id": task.task_id,
                "mutation_mode": "transactional_path_repair",
                "status": "rebuilt_pending_recritic",
                "rollback_start_step_id": rollback.rollback_start_step_id,
                "rebuild_through_step_id": rollback.rebuild_through_step_id,
                "old_route_depth": len(rollback.original_steps),
                "durable_route_depth": len(rollback.durable_steps),
                "new_route_depth": len(rebuilt_steps),
                "removed_step_count": len(rollback.removed_steps),
                "builder_rebuild_step_count": len(added_steps),
            }
        )
        branch["complete_in_bound_stock"] = _branch_stock_closed(branch)
        return True

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
        reserve_model_invocations: int,
        reserve_input_tokens: int,
        reserve_output_tokens: int,
        reserve_wall_time_s: float,
        config: DirectorConfig,
    ) -> bool:
        steps = [dict(row) for row in branch.get("steps") or []]
        concrete_blockers = [dict(row) for row in blocking_steps if isinstance(row, Mapping)]
        if not concrete_blockers:
            return False
        if config.enable_transactional_path_repair:
            return self._repair_branch_transactionally(
                spec,
                target=target,
                branch=branch,
                blocking_steps=concrete_blockers,
                critique=critique,
                iteration=iteration,
                records=records,
                max_prompt_bytes=max_prompt_bytes,
                max_node_call_timeout_s=max_node_call_timeout_s,
                quota=quota,
                started=started,
                reserve_model_invocations=reserve_model_invocations,
                reserve_input_tokens=reserve_input_tokens,
                reserve_output_tokens=reserve_output_tokens,
                reserve_wall_time_s=reserve_wall_time_s,
                config=config,
            )
        blocking_step = concrete_blockers[0]
        # Surgical single-step replacement cannot preserve an AiZ dependency
        # suffix.  Do not silently grant a second full-route writer when a
        # caller selected no route-mutation mode: frozen paper profiles opt in
        # to direct document editing, while the self-correcting profile has
        # already returned through the transactional directive path above.
        if (
            not allow_editor_route_mutations
            and str(branch.get("strategy_tree_engine") or "") == "aizynthfinder_mcts"
        ):
            branch.setdefault("editor_execution_notes", []).append(
                {
                    "reason": "mcts_editor_mode_not_configured",
                    "requested_mode": "surgical",
                    "effective_mode": "none",
                    "semantics": {
                        "suffix_is_never_discarded": True,
                        "no_implicit_route_mutation_authority": True,
                    },
                }
            )
            return False
        step_id = str(blocking_step.get("step_id") or "")
        try:
            step_index = next(
                index for index, row in enumerate(steps) if str(row.get("step_id") or "") == step_id
            )
        except StopIteration:
            return False
        selected_product = _canonical_smiles(blocking_step.get("product_smiles"))
        if not selected_product:
            return False
        selected_product_mapped = str(
            blocking_step.get("mapped_product_smiles") or _mapped_smiles(selected_product)
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
            if str(dict(value.get("assessment") or {}).get("suggested_revision") or "")
        ]
        rejection = {
            "phase": "critic_editor",
            "step_id": step_id,
            "blocking_step_ids": [str(row.get("step_id") or "") for row in concrete_blockers],
            "product_smiles": selected_product,
            "failure_reasons": list(feedback.get("failure_reasons") or feedback_reasons),
            "repair_actions": list(feedback.get("repair_actions") or feedback_revisions),
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
        remaining_editor_budget = max(0, configured_editor_budget - editor_attempts_used)
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
                    (
                        (editor_prompt_steps, target, False),
                        "Codex Editor: dependency-closed RouteJSON repair preserving the StrategyCard",
                    ),
                    (
                        (editor_prompt_steps, target, True),
                        "Codex Editor: compact dependency-closed repair",
                    ),
                )
                if allow_editor_route_mutations
                else (
                    (
                        (prefix, selected_product, False),
                        "Codex Editor: surgical repair preserving the StrategyCard",
                    ),
                    ((prefix[-3:], selected_product, False), "Codex Editor: surgical repair"),
                    (
                        (prefix[-1:], selected_product, False),
                        "Codex Editor: compact surgical repair",
                    ),
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
                    prior_rejections=(() if allow_editor_route_mutations else [attempt_rejection]),
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
                records,
                started=started,
                quota=quota,
                reserve_model_invocations=reserve_model_invocations,
                reserve_input_tokens=(reserve_input_tokens + _EDITOR_INPUT_TOKEN_RESERVE),
                reserve_output_tokens=(reserve_output_tokens + _EDITOR_OUTPUT_TOKEN_RESERVE),
                reserve_wall_time_s=reserve_wall_time_s,
            ):
                return False
            # Keep the attempted-call count separate from the applied-edit
            # count.  ``editor_call_count`` is retained for compatibility and
            # means an edit that actually changed the working route; it must
            # not be used to infer that no Editor worker was invoked.
            task = _node_task(
                spec,
                prompt=prompt,
                branch_index=int(branch.get("branch_index") or 0),
                node_index=(iteration + 1) * _MATERIALIZATION_RETRY_LIMIT + editor_attempt,
                model=str(spec.metadata.get("model") or ""),
                reasoning_effort=str(
                    spec.metadata.get("editor_reasoning_effort")
                    or spec.metadata.get("reasoning_effort")
                    or "medium"
                ),
                timeout_s=_node_call_timeout_s(
                    started,
                    quota,
                    maximum=min(
                        max_node_call_timeout_s,
                        max(
                            0.001,
                            _remaining_node_wall_time(started, quota)
                            - max(0.0, reserve_wall_time_s)
                            - _deadline_settlement_reserve_s(quota),
                        ),
                    ),
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
            if worker_provider_failure_reason(record):
                return False
            branch["editor_attempt_count"] = int(branch.get("editor_attempt_count") or 0) + 1
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
                branch.setdefault("rejections", []).append(dict(attempt_rejection))
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
            route_expansions, diagnostic, mutation_mode = _editor_route_expansions_from_record(
                record,
                current_steps=editor_base_steps,
                mapped_target_smiles=str(
                    branch.get("target_mapped_smiles") or _mapped_smiles(target)
                ),
                expected_target_smiles=target,
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
                        branch.get("target_mapped_smiles") or _mapped_smiles(target)
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
                            expansion.step_id or f"codex:editor:{iteration + 1}:{route_index + 1}"
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
                        membership.get(value) is True for value in terminal_precursors
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
            branch["deferred_builder_leaf_states"] = deque()
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
            branch["editor_call_count"] = int(branch.get("editor_call_count") or 0) + 1
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
        branch["deferred_builder_leaf_states"] = deque()
        branch["blocked_materializations"] = []
        _sync_open_leaf_projection(branch)
        self._rebuild_branch_or_search_after_editor(
            branch,
            target=target,
            max_depth=max_node_expansions_per_branch,
        )
        branch["call_count"] = int(branch.get("call_count") or 0) + 1
        branch["editor_call_count"] = int(branch.get("editor_call_count") or 0) + 1
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
        steps = [dict(row) for row in branch.get("steps") or [] if isinstance(row, Mapping)]
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
                    _canonical_smiles(value) for value in row.get("precursor_smiles") or []
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
            stock_smiles=(value for value in terminal_precursors if membership.get(value) is True),
        )
        branch["_reactionjson_or_search"] = rebuilt
        _refresh_branch_from_reactionjson_or_search(branch, rebuilt)
        branch.setdefault("reactionjson_or_search_resets", []).append(
            {
                "reason": "critic_editor_route_mutation",
                "previous_summary": previous_summary,
                "rebuilt_summary": dict(branch.get("reactionjson_or_search") or {}),
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
        branch_mandates = _branch_mandates_for_profile(config.strategy_portfolio_mode)
        branches: list[dict[str, Any]] = [
            {
                "branch_index": branch_index,
                "lens": branch_mandates[branch_index % len(branch_mandates)],
                "strategy_mandate": branch_mandates[branch_index % len(branch_mandates)],
                "strategy_seed": "",
                "strategy_seed_source": "generated",
                "strategy_seed_sha256": "",
                "steps": [],
                "open_leaves": deque([target]),
                "open_leaf_states": deque([{"smiles": target, "mapped_smiles": mapped_target}]),
                "deferred_builder_leaf_states": deque(),
                "target_mapped_smiles": mapped_target,
                "expanded_products": set(),
                "call_count": 0,
                "strategy_call_count": 0,
                "route_call_count": 0,
                "path_repair_builder_call_count": 0,
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

        # A Strategy-only screen may explicitly promote its reviewed
        # three-card portfolio into route construction.  The file is loaded,
        # target-bound and content-addressed by the host CLI; the model sees
        # only the same compact StrategyCard fields it would have authored in
        # this run.  This path skips duplicate Strategy generation/review but
        # retains every Builder, key-event Critic, Editor and Host gate.
        promoted_portfolio = bool(config.reviewed_strategy_portfolio)
        if promoted_portfolio:
            accepted_cards: list[dict[str, Any]] = []
            for branch, raw_card in zip(
                branches,
                config.reviewed_strategy_portfolio,
                strict=True,
            ):
                card = normalize_strategy_card(_paper_matched_strategy_card_payload(raw_card))
                if (
                    not _valid_strategy_card(card)
                    or not _strategy_card_bonds_match_target(
                        card,
                        target_smiles=target,
                        mapped_target_smiles=mapped_target,
                    )
                    or _strategy_conflicts(card, accepted_cards)
                ):
                    raise ValueError("reviewed strategy portfolio promotion is invalid")
                accepted_cards.append(card)
                branch["strategy_card"] = card
                branch["root_strategy_card"] = dict(card)
                branch["strategy_seed"] = _strategy_title_from_card(card)
                branch["strategy_seed_source"] = "reviewed_strategy_screen"
                branch["strategy_seed_sha256"] = str(config.reviewed_strategy_portfolio_sha256)
                branch["lens"] = "Reviewed strategy - " + str(branch["strategy_seed"])

        # Preserve only one mandatory final Critic for each seeded paper
        # branch. Key-event Critics run for every replayed candidate that
        # claims the active checkpoint; they do not own a fixed per-Strategy
        # quota. Optional online Critic and Editor calls share the remaining
        # call/token balance dynamically below.
        # A tight Strategy -> Builder canary intentionally ends when its model
        # ceiling is exhausted. Reserving the later Critic/Editor phase in
        # that envelope would starve every Builder branch (1 portfolio call +
        # one node call per branch already consumes the whole ceiling).
        builder_only_model_ceiling = 1 + (
            int(config.strategy_branch_count) * int(config.max_node_expansions_per_branch)
        )
        builder_only_canary = bool(
            config.paper_matched_reach_profile
            and int(quota.model_invocations) <= builder_only_model_ceiling
        )
        critic_reserve_slots = (
            0 if builder_only_canary else max(0, int(config.strategy_branch_count))
        )
        critic_editor_call_reserve = critic_reserve_slots
        critic_editor_wall_reserve = (
            quota.wall_time_s
            * (
                _PAPER_CRITIC_EDITOR_WALL_FRACTION
                if config.paper_matched_reach_profile
                else _CRITIC_EDITOR_WALL_FRACTION
            )
            if critic_reserve_slots
            else 0.0
        )
        critic_input_reserve = critic_reserve_slots * _CRITIC_INPUT_TOKEN_RESERVE
        critic_output_reserve = critic_reserve_slots * _CRITIC_OUTPUT_TOKEN_RESERVE
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
        paper_portfolio_attempted = bool(config.paper_matched_reach_profile and len(branches) == 3)
        if (
            paper_portfolio_attempted
            and not promoted_portfolio
            and _node_budget_allows(
                records,
                started=started,
                quota=route_quota,
            )
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
                enhanced_strategy=config.enable_strategy_portfolio_critic,
            )
            if self._provider_runtime_failure_snapshot():
                return branches, records
            if (
                not promoted_portfolio
                and config.enable_strategy_portfolio_critic
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
                if self._provider_runtime_failure_snapshot():
                    return branches, records
        while (
            not self._cancelled()
            and any(not branch["strategy_card"] for branch in branches)
            and not paper_portfolio_attempted
        ):
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
                if int(branch["strategy_call_count"]) >= _STRATEGY_SEED_RETRY_LIMIT:
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
                if self._provider_runtime_failure_snapshot():
                    return branches, records
                progressed = True
            if not progressed or not _node_budget_allows(
                records,
                started=started,
                quota=route_quota,
            ):
                break

        seeded = [branch for branch in branches if branch["strategy_card"]]
        if self._provider_runtime_failure_snapshot():
            return branches, records

        # Phase 2 expands the already committed strategies round-robin.  Route
        # state remains isolated.  The paper profile co-generates all three
        # cards in one portfolio call; compatibility profiles enforce the same
        # orthogonality while selecting their cards serially.
        critic_slots = 0 if builder_only_canary else len(seeded)
        # Recompute the protected final-Critic balance from branches that were
        # actually seeded. Failed Strategy hypotheses do not strand quota.
        critic_editor_call_reserve = critic_slots
        critic_input_reserve = critic_slots * _CRITIC_INPUT_TOKEN_RESERVE
        critic_output_reserve = critic_slots * _CRITIC_OUTPUT_TOKEN_RESERVE
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
                        int(branch["route_call_count"]) >= config.max_node_expansions_per_branch
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
                    if self._provider_runtime_failure_snapshot():
                        return branches, records
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
                    _branch_stock_closed(branch) or not _branch_has_expandable_leaf(branch)
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

    def _review_selected_uncertain_key_event(
        self,
        spec: AgentSpec,
        *,
        target: str,
        branch: dict[str, Any],
        strategy_card: Mapping[str, Any],
        route_steps: Iterable[Mapping[str, Any]],
        records: list[WorkerRunRecord],
        shared_ledger: _SharedModelCallLedger,
        route_quota: _NodeCallBudget,
        config: DirectorConfig,
        started: float,
    ) -> _KeyEventReviewDisposition:
        """Revisit one uncertain checkpoint only after selected new evidence."""

        steps = [dict(row) for row in route_steps if isinstance(row, Mapping)]
        review = _pending_uncertain_key_event_evidence_review(
            branch,
            strategy_card=strategy_card,
            steps=steps,
        )
        if not review:
            return _KeyEventReviewDisposition()
        focus_step_id = str(review.get("focus_step_id") or "")
        evidence_step_id = str(review.get("evidence_step_id") or "")
        by_id = {
            str(row.get("step_id") or ""): row for row in steps if str(row.get("step_id") or "")
        }
        focus_step = by_id.get(focus_step_id)
        evidence_step = by_id.get(evidence_step_id)
        if focus_step is None or evidence_step is None:
            return _KeyEventReviewDisposition(status="evidence_unavailable")
        evidence_mapped = str(evidence_step.get("mapped_product_smiles") or "")
        checkpoint_feedback = _pending_key_event_feedback_for_leaf(
            branch,
            strategy_card=strategy_card,
            steps=steps,
            selected_product_mapped=evidence_mapped,
            include_uncertain=True,
        )
        history_row: dict[str, Any] = {
            "focus_step_id": focus_step_id,
            "product_smiles": str(focus_step.get("product_smiles") or ""),
            "strategy_id": str(strategy_card.get("strategy_id") or ""),
            "strategy_digest": _strategy_card_digest(strategy_card),
            "strategy_milestone_index": _strategy_milestone_index(branch, strategy_card),
            "lineage_root_mapped_smiles": str(review.get("lineage_root_mapped_smiles") or ""),
            "obligation_id": str(review.get("obligation_id") or ""),
            "review_of_obligation_id": str(review.get("obligation_id") or ""),
            "review_evidence_step_id": evidence_step_id,
            "required_selected_step_ids": [focus_step_id, evidence_step_id],
            "review_kind": "selected_direct_precursor_evidence",
        }
        prompt = _bounded_critic_prompt(
            target=target,
            branch_index=int(branch.get("branch_index") or 0),
            strategy_card=strategy_card,
            steps=steps,
            maximum_bytes=config.max_node_prompt_bytes,
            paper_matched=True,
            audit_kind="key_event_followup",
            focus_step_id=focus_step_id,
            checkpoint_feedback=checkpoint_feedback,
        )
        if prompt is None:
            history_row["status"] = "prompt_unavailable"
            branch.setdefault("key_event_critic_history", []).append(history_row)
            return _KeyEventReviewDisposition(status="prompt_unavailable")
        reservation, budget_reason = shared_ledger.reserve(
            input_tokens=_CRITIC_INPUT_TOKEN_RESERVE,
            output_tokens=_CRITIC_OUTPUT_TOKEN_RESERVE,
        )
        if reservation is None:
            history_row.update(
                {
                    "status": "budget_unavailable",
                    "reason": budget_reason,
                }
            )
            branch.setdefault("key_event_critic_history", []).append(history_row)
            return _KeyEventReviewDisposition(status="budget_unavailable")
        review_index = (
            int(branch.get("route_call_count") or 0)
            + int(branch.get("path_repair_builder_call_count") or 0)
            + int(branch.get("key_event_critic_call_count") or 0)
            + 1
        )
        critic_task = _critic_task(
            spec,
            prompt=prompt,
            branch_index=int(branch.get("branch_index") or 0),
            iteration=review_index,
            timeout_s=_node_call_timeout_s(
                started,
                route_quota,
                maximum=config.critic_call_timeout_s,
            ),
            paper_matched=True,
            target_smiles=target,
            audit_kind="key_event_followup",
            focus_step_id=focus_step_id,
        )
        try:
            critic_record = self._run_journaled_worker(self.critic_executor, critic_task)
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
                    "reasons": ["key_event_followup_critic_execution_failed"],
                },
            )
        shared_ledger.settle(reservation, critic_record)
        records.append(critic_record)
        if worker_provider_failure_reason(critic_record):
            return _KeyEventReviewDisposition(status="runtime_unavailable")
        branch["critic_call_count"] = int(branch.get("critic_call_count") or 0) + 1
        branch["key_event_critic_call_count"] = (
            int(branch.get("key_event_critic_call_count") or 0) + 1
        )
        critique = _bind_key_event_focus_assessment(
            _critique_from_record(critic_record),
            focus_step_id,
        )
        focus_assessment = _key_event_focus_assessment(critique, focus_step_id)
        checkpoint_match = critique.get("checkpoint_match") is True and focus_assessment is not None
        checkpoint_verdict = str(dict(focus_assessment or {}).get("verdict") or "")
        checkpoint_rejected = bool(
            focus_assessment is not None
            and (checkpoint_verdict == "reject" or focus_assessment.get("blocking") is True)
        )
        history_row.update(
            {
                "task_id": critic_task.task_id,
                "status": (
                    "rejected"
                    if checkpoint_rejected
                    else (
                        "completed"
                        if checkpoint_match and checkpoint_verdict == "pass"
                        else ("uncertain" if checkpoint_match else "not_checkpoint")
                    )
                ),
                "critic_status": str(critique.get("status") or "unavailable"),
                "checkpoint_match": checkpoint_match,
                "assessment": dict(focus_assessment or {}),
            }
        )
        branch.setdefault("key_event_critic_history", []).append(history_row)
        if checkpoint_rejected:
            assessment = dict(focus_assessment or {})
            reasons = [
                str(value).strip()
                for value in assessment.get("reasons") or ()
                if str(value).strip()
            ]
            rejection_reason = (
                "; ".join(reasons[:2]) or str(assessment.get("suggested_revision") or "").strip()
            )
            return _KeyEventReviewDisposition(
                status="rejected",
                rejected_path_step_ids=(focus_step_id, evidence_step_id),
                rejection_reason=(rejection_reason or "key_event_followup_critic_reject"),
            )
        return _KeyEventReviewDisposition(status=str(history_row["status"]))

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
        branch_count = len(seeded)
        shared_ledger = _SharedModelCallLedger(
            route_quota,
            existing_records,
            protected_model_invocations=critic_editor_call_reserve,
            protected_input_tokens=critic_input_reserve,
            protected_output_tokens=critic_output_reserve,
        )

        def advance(branch: dict[str, Any]) -> list[WorkerRunRecord]:
            local_records: list[WorkerRunRecord] = []
            route_records: list[WorkerRunRecord] = []
            branch_index = int(branch["branch_index"])
            path_repair_resume = dict(branch.get("_path_repair_resume") or {})
            repair_phase = bool(path_repair_resume)
            builder_counter_key = (
                "path_repair_builder_call_count" if repair_phase else "route_call_count"
            )
            builder_calls_before_phase = int(branch.get(builder_counter_key) or 0)
            builder_call_ceiling = int(config.max_node_expansions_per_branch)
            phase_builder_call_ceiling = max(
                0,
                builder_call_ceiling - builder_calls_before_phase,
            )
            search_diagnostic_key = (
                "path_repair_aizynthfinder_search"
                if repair_phase
                else "aizynthfinder_strategy_search"
            )
            durable_seed_steps = [
                dict(row)
                for row in path_repair_resume.get("durable_steps") or []
                if isinstance(row, Mapping)
            ]
            durable_seed_step_ids = [str(row.get("step_id") or "") for row in durable_seed_steps]
            root_strategy_card = dict(
                path_repair_resume.get("strategy_card")
                or branch.get("root_strategy_card")
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
            repair_frontier_selections: dict[str, set[str]] = {}
            path_rejection_pending: dict[str, Any] = {}

            def remember_path_repair_replay_failure(
                diagnostic: Mapping[str, Any],
            ) -> None:
                """Keep one transaction-owned copy of deterministic replay debt."""

                if not path_repair_resume:
                    return
                failures = _merge_path_repair_replay_failure(
                    path_repair_resume.get("replay_failures") or (),
                    diagnostic,
                )
                if not failures:
                    return
                path_repair_resume["replay_failures"] = failures
                branch_resume = branch.get("_path_repair_resume")
                if isinstance(branch_resume, dict):
                    branch_resume["replay_failures"] = copy.deepcopy(failures)

            def reject_selected_path(
                disposition: _KeyEventReviewDisposition,
                *,
                prompt_steps: Sequence[Mapping[str, Any]],
                model_call_consumed: bool,
            ) -> Mapping[str, Any]:
                """Rollback the durable Host view and ask AiZ to prune one edge."""

                rejected_ids = tuple(
                    dict.fromkeys(
                        str(value).strip()
                        for value in disposition.rejected_path_step_ids
                        if str(value).strip()
                    )
                )
                rejected = set(rejected_ids)
                rollback_index = next(
                    (
                        index
                        for index, row in enumerate(prompt_steps)
                        if str(row.get("step_id") or "") in rejected
                    ),
                    None,
                )
                if rollback_index is None:
                    raise RuntimeError("key_event_rejected_step_absent_from_selected_path")
                retained_steps = [dict(row) for row in prompt_steps[:rollback_index]]
                retained_projection = _materialize_aizynthfinder_projection(
                    steps=retained_steps,
                    mapped_target_smiles=str(
                        branch.get("target_mapped_smiles") or _mapped_smiles(target)
                    ),
                    search_diagnostics={},
                    stock_membership=self._stock_membership,
                )
                branch["steps"] = [dict(row) for row in retained_projection["steps"]]
                branch["open_leaf_states"] = deque(
                    dict(row) for row in retained_projection["open_leaf_states"]
                )
                branch["deferred_builder_leaf_states"] = deque()
                branch["expanded_products"] = {
                    product
                    for row in branch["steps"]
                    if (product := _canonical_smiles(row.get("product_smiles")))
                }
                branch["sidecar_durable_prefix_step_count"] = len(retained_steps)
                branch["complete_in_bound_stock"] = False
                _sync_open_leaf_projection(branch)
                rejection_row = {
                    "phase": "key_event_followup_critic",
                    "reason": "selected_path_key_event_rejected",
                    "rejected_path_step_ids": list(rejected_ids),
                    "pruned_step_id": str(prompt_steps[rollback_index].get("step_id") or ""),
                    "rejection_reason": disposition.rejection_reason,
                    "retained_prefix_step_ids": [
                        str(row.get("step_id") or "") for row in retained_steps
                    ],
                    "authority": "selected_path_key_event_critic",
                }
                branch.setdefault("rejections", []).append(rejection_row)
                branch.setdefault("aiz_path_rejections", []).append(rejection_row)
                path_rejection_pending.clear()
                path_rejection_pending.update(
                    {
                        "rejected_path_step_ids": list(rejected_ids),
                        "retained_prefix_step_ids": list(rejection_row["retained_prefix_step_ids"]),
                    }
                )
                return {
                    "candidates": [],
                    "model_call_consumed": bool(model_call_consumed),
                    "rejected_path_step_ids": list(rejected_ids),
                    "rejection_reason": disposition.rejection_reason,
                }

            def handle_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
                provider_failure = self._provider_runtime_failure_snapshot()
                if provider_failure:
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "runtime_unavailable": True,
                        "runtime_pause": True,
                        "stop_search": True,
                        "stop_reason": str(
                            provider_failure.get("reason") or "provider_unavailable"
                        ),
                    }
                if self._cancelled():
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": "host_cancelled",
                    }
                repair_completion_mode = str(
                    path_repair_resume.get("completion_mode") or ""
                )
                if (
                    path_repair_resume
                    and repair_completion_mode == "strategy_checkpoint"
                    and not list(path_repair_resume.get("reconnect_boundaries") or [])
                ):
                    repair_prompt_steps = [
                        dict(row)
                        for row in request.get("route_steps") or []
                        if isinstance(row, Mapping)
                    ]
                    repair_added_steps = repair_prompt_steps[len(durable_seed_steps) :]
                    selected_critic_pass_step_ids = {
                        str(row.get("focus_step_id") or "")
                        for row in branch.get("key_event_critic_history") or []
                        if isinstance(row, Mapping)
                        and str(row.get("status") or "") == "completed"
                        and str(row.get("focus_step_id") or "")
                    }
                    if _path_repair_completion_reached(
                        repair_added_steps,
                        completion_mode=repair_completion_mode,
                        selected_critic_pass_step_ids=selected_critic_pass_step_ids,
                    ):
                        return {
                            "candidates": [],
                            "model_call_consumed": False,
                            "stop_search": True,
                            "stop_reason": "path_repair_strategy_checkpoint_reached",
                        }
                if int(branch.get(builder_counter_key) or 0) >= builder_call_ceiling:
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": (
                            "path_repair_builder_budget_exhausted_before_boundary"
                            if repair_phase
                            else "route_builder_branch_call_ceiling_reached"
                        ),
                    }
                expandable = [
                    _canonical_smiles(value) for value in request.get("expandable_smiles") or []
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
                if path_rejection_pending:
                    prompt_step_ids = [str(row.get("step_id") or "") for row in prompt_steps]
                    rejected_ids = set(path_rejection_pending.get("rejected_path_step_ids") or [])
                    retained_prefix_ids = list(
                        path_rejection_pending.get("retained_prefix_step_ids") or []
                    )
                    if rejected_ids.intersection(prompt_step_ids):
                        raise RuntimeError("aiz_path_rejection_did_not_prune_selected_subtree")
                    if prompt_step_ids[: len(retained_prefix_ids)] != (retained_prefix_ids):
                        raise RuntimeError("aiz_path_rejection_changed_retained_prefix")
                    path_rejection_pending.clear()
                # The request path is already selected by AiZ and every row
                # has already crossed the Host replay boundary.  Persist the
                # deepest such prefix before making another paid call so a
                # later sidecar/provider failure cannot erase completed work.
                durable_depth = int(branch.get("sidecar_durable_prefix_step_count") or 0)
                if len(prompt_steps) > durable_depth:
                    branch["steps"] = [dict(row) for row in prompt_steps]
                    branch["open_leaf_states"] = deque(
                        {
                            "smiles": smiles,
                            "mapped_smiles": (
                                mapped_values[index]
                                if index < len(mapped_values) and mapped_values[index]
                                else _mapped_smiles(smiles)
                            ),
                        }
                        for index, smiles in enumerate(expandable)
                        if smiles
                    )
                    branch["expanded_products"] = {
                        product
                        for row in prompt_steps
                        if (product := _canonical_smiles(row.get("product_smiles")))
                    }
                    branch["sidecar_durable_prefix_step_count"] = len(prompt_steps)
                    _sync_open_leaf_projection(branch)
                if len(prompt_steps) < len(durable_seed_steps):
                    prompt_step_ids = [str(row.get("step_id") or "") for row in prompt_steps]
                    if prompt_step_ids != durable_seed_step_ids[: len(prompt_steps)]:
                        return {
                            "candidates": [],
                            "model_call_consumed": False,
                            "stop_search": True,
                            "stop_reason": "path_repair_durable_seed_prefix_mismatch",
                        }
                    seed_step = dict(durable_seed_steps[len(prompt_steps)])
                    seed_product_mapped = str(seed_step.get("mapped_product_smiles") or "")
                    expandable_mapped = {
                        _canonical_mapped_smiles(value)
                        for value in mapped_values
                        if _canonical_mapped_smiles(value)
                    }
                    if (
                        not seed_product_mapped
                        or _canonical_mapped_smiles(seed_product_mapped) not in expandable_mapped
                    ):
                        return {
                            "candidates": [],
                            "model_call_consumed": False,
                            "stop_search": True,
                            "stop_reason": "path_repair_durable_seed_boundary_mismatch",
                        }
                    return {
                        "candidates": [
                            {
                                "candidate_id": (
                                    "host-durable-seed:"
                                    + str(seed_step.get("step_id") or len(prompt_steps) + 1)
                                ),
                                "product_smiles": str(seed_step.get("product_smiles") or ""),
                                "mapped_product_smiles": seed_product_mapped,
                                "precursor_smiles": list(seed_step.get("precursor_smiles") or []),
                                "mapped_precursor_smiles": list(
                                    seed_step.get("mapped_precursor_smiles") or []
                                ),
                                "route_step": seed_step,
                                "prior": 1.0,
                                "candidate_key": _key_event_fingerprint(seed_step),
                            }
                        ],
                        "model_call_consumed": False,
                        "host_replay_seed": True,
                    }
                reconnect_boundaries = [
                    dict(row)
                    for row in path_repair_resume.get("reconnect_boundaries") or []
                    if isinstance(row, Mapping)
                ]
                completion_boundaries = [
                    dict(row)
                    for row in (
                        path_repair_resume.get("search_completion_boundaries")
                        or reconnect_boundaries
                    )
                    if isinstance(row, Mapping)
                ]
                if len(prompt_steps) > len(
                    durable_seed_steps
                ) and repair_completion_mode == "cut_frontier" and (
                    _path_repair_frontier_reaches_boundaries(
                    product_smiles=expandable,
                    mapped_product_smiles=mapped_values,
                    reconnect_boundaries=completion_boundaries,
                    )
                ):
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": "path_repair_reconnect_boundary_reached",
                    }
                selectable_indices = [index for index, value in enumerate(expandable) if value]
                if path_repair_resume and len(prompt_steps) > len(durable_seed_steps):
                    reached_boundary_indices = _path_repair_boundary_leaf_indices(
                        product_smiles=expandable,
                        mapped_product_smiles=mapped_values,
                        reconnect_boundaries=(
                            completion_boundaries
                            if repair_completion_mode == "cut_frontier"
                            else reconnect_boundaries
                        ),
                    )
                    unfinished = [
                        index
                        for index in selectable_indices
                        if index not in reached_boundary_indices
                    ]
                    if unfinished:
                        selectable_indices = unfinished
                selected_index = selectable_indices[0] if selectable_indices else None
                if path_repair_resume and len(prompt_steps) == len(durable_seed_steps):
                    repair_frontier_mapped = _canonical_mapped_smiles(
                        path_repair_resume.get("repair_frontier_mapped_product_smiles")
                    )
                    preferred_index = next(
                        (
                            index
                            for index, value in enumerate(mapped_values)
                            if repair_frontier_mapped
                            and _canonical_mapped_smiles(value) == repair_frontier_mapped
                        ),
                        None,
                    )
                    if preferred_index is None:
                        return {
                            "candidates": [],
                            "model_call_consumed": False,
                            "stop_search": True,
                            "stop_reason": "path_repair_mapped_frontier_not_expandable",
                        }
                    selected_index = preferred_index
                elif path_repair_resume and len(selectable_indices) > 1:
                    # A repair step may split the frontier into the molecular
                    # component that still carries the rejected graph edit
                    # and independent siblings (a cleaved tether, leaving
                    # fragment, coupling partner, and so on).  The repair
                    # transaction must follow the former.  Cycling over the
                    # raw AiZ array order can instead send the next Builder to
                    # a spectator and sever the Critic/Editor repair lineage.
                    # Derive the focus directly from the Host-replayed
                    # reference span; keep tied focus-bearing components
                    # eligible without inventing another identity authority.
                    selectable_indices = list(
                        _path_repair_focus_leaf_indices(
                            selectable_indices=selectable_indices,
                            mapped_product_smiles=mapped_values,
                            path_repair=path_repair_resume,
                        )
                    )
                    frontier_fingerprint = hashlib.sha256(
                        json.dumps(
                            {
                                "route_step_ids": [
                                    str(row.get("step_id") or "") for row in prompt_steps
                                ],
                                "open_mapped_precursors": sorted(
                                    _canonical_mapped_smiles(value)
                                    for value in mapped_values
                                    if _canonical_mapped_smiles(value)
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    prior_selections = repair_frontier_selections.setdefault(
                        frontier_fingerprint, set()
                    )
                    untried = [
                        index
                        for index in selectable_indices
                        if _canonical_mapped_smiles(
                            mapped_values[index]
                            if index < len(mapped_values)
                            else _mapped_smiles(expandable[index])
                        )
                        not in prior_selections
                    ]
                    if untried:
                        selected_index = untried[0]
                    else:
                        prior_selections.clear()
                        selected_index = selectable_indices[0]
                    prior_selections.add(
                        _canonical_mapped_smiles(
                            mapped_values[selected_index]
                            if selected_index < len(mapped_values)
                            else _mapped_smiles(expandable[selected_index])
                        )
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
                    mapped_values[selected_index] if selected_index < len(mapped_values) else ""
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
                    branch.setdefault("rejections", []).append(dict(prior_policy_feedback))
                rejected = list(branch.get("rejections") or [])
                active_strategy_card, refresh_strategy = _strategy_horizon_for_leaf(
                    config=config,
                    branch=branch,
                    root_strategy_card=root_strategy_card,
                    steps=prompt_steps,
                    selected_product_mapped=selected_mapped,
                )
                rejected_strategy_horizon = _rejected_strategy_horizon_for_leaf(
                    branch,
                    strategy_card=active_strategy_card,
                    steps=prompt_steps,
                    selected_product_mapped=selected_mapped,
                )
                if path_repair_resume:
                    # The Editor transaction owns a rejected checkpoint under
                    # its original Strategy.  Do not replace that Strategy or
                    # pre-complete its re-Critic obligation merely because the
                    # rollback frontier resembles a normal horizon boundary.
                    active_strategy_card = root_strategy_card
                    refresh_strategy = False
                    rejected_strategy_horizon = {}
                else:
                    branch["key_event_critic_completed"] = bool(
                        refresh_strategy and not rejected_strategy_horizon
                    )
                if (
                    int(config.max_strategic_milestones_per_branch) > 1
                    and refresh_strategy
                    and int(branch.get("strategy_milestone_generation_count") or 0)
                    < int(config.max_strategic_milestones_per_branch) - 1
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
                        quota=route_quota,
                        started=started,
                        budget_ledger=shared_ledger,
                        paper_matched=config.paper_matched_reach_profile,
                        retired_strategy_feedback={
                            "strategy_card": {
                                key: str(active_strategy_card.get(key) or "")
                                for key in (
                                    "strategy_query",
                                    "critical_assumption",
                                    "critic_checkpoint",
                                )
                            },
                            "assessment": dict(rejected_strategy_horizon.get("assessment") or {}),
                        }
                        if rejected_strategy_horizon
                        else None,
                    )
                    provider_failure = self._provider_runtime_failure_snapshot()
                    if provider_failure:
                        return {
                            "candidates": [],
                            "model_call_consumed": False,
                            "runtime_unavailable": True,
                            "runtime_pause": True,
                            "stop_search": True,
                            "stop_reason": str(
                                provider_failure.get("reason") or "provider_unavailable"
                            ),
                        }
                    if generated is not None:
                        active_strategy_card = generated
                    elif rejected_strategy_horizon:
                        return {
                            "candidates": [],
                            "model_call_consumed": True,
                            "stop_search": True,
                            "stop_reason": "strategy_horizon_replacement_unavailable",
                        }
                elif rejected_strategy_horizon:
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": "strategy_horizon_budget_exhausted",
                    }
                if config.enable_key_event_critic and not path_repair_resume:
                    review_disposition = self._review_selected_uncertain_key_event(
                        spec,
                        target=target,
                        branch=branch,
                        strategy_card=active_strategy_card,
                        route_steps=prompt_steps,
                        records=local_records,
                        shared_ledger=shared_ledger,
                        route_quota=route_quota,
                        config=config,
                        started=started,
                    )
                    if review_disposition.status == "runtime_unavailable":
                        provider_failure = self._provider_runtime_failure_snapshot()
                        return {
                            "candidates": [],
                            "model_call_consumed": False,
                            "runtime_unavailable": True,
                            "runtime_pause": True,
                            "stop_search": True,
                            "stop_reason": str(
                                provider_failure.get("reason") or "provider_unavailable"
                            ),
                        }
                    if review_disposition.rejected:
                        return reject_selected_path(
                            review_disposition,
                            prompt_steps=prompt_steps,
                            model_call_consumed=False,
                        )
                lineage_checkpoint_feedback = _pending_key_event_feedback_for_leaf(
                    branch,
                    strategy_card=active_strategy_card,
                    steps=prompt_steps,
                    selected_product_mapped=selected_mapped,
                )
                pending_checkpoint_feedback = (
                    _merge_key_event_feedback(
                        _path_repair_checkpoint_feedback(
                            path_repair_resume,
                            strategy_card=active_strategy_card,
                        ),
                        lineage_checkpoint_feedback,
                    )
                    if path_repair_resume
                    else lineage_checkpoint_feedback
                )
                # Diagnostic projection only. The append-only Critic history
                # is the authority and the next Builder context is derived
                # from it for the selected Strategy/leaf lineage.
                if not path_repair_resume:
                    branch["pending_key_event_feedback"] = dict(pending_checkpoint_feedback)
                axis_call_index = int(branch.get(builder_counter_key) or 0) + 1
                call_index = (
                    int(branch.get("route_call_count") or 0)
                    + int(branch.get("path_repair_builder_call_count") or 0)
                    + 1
                )
                prompt = _node_prompt(
                    target=target,
                    branch_index=branch_index,
                    lens=str(branch.get("lens") or ""),
                    selected_product=selected,
                    selected_product_mapped=selected_mapped,
                    steps=prompt_steps,
                    open_leaves=expandable,
                    prior_rejections=rejected,
                    repair=bool(path_repair_resume),
                    strategy_card=active_strategy_card,
                    forbidden_strategy_cards=(),
                    host_failure_feedback={
                        "pending_checkpoint_feedback": (pending_checkpoint_feedback),
                        "path_repair": {
                            key: value
                            for key, value in path_repair_resume.items()
                            if key
                            in {
                                "rollback_start_step_id",
                                "rebuild_through_step_id",
                                "repair_goal",
                                "active_constraints",
                                "reconnect_boundaries",
                                "repair_reference_span",
                                "replay_failures",
                            }
                        },
                    },
                    complete_route_json=False,
                    minimum_route_depth=1,
                    max_reactionjson_candidates=(config.max_reactionjson_candidates_per_node),
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
                        repair=bool(path_repair_resume),
                        strategy_card=active_strategy_card,
                        forbidden_strategy_cards=(),
                        host_failure_feedback={
                            "pending_checkpoint_feedback": (pending_checkpoint_feedback),
                            "path_repair": {
                                key: value
                                for key, value in path_repair_resume.items()
                                if key
                                in {
                                    "rollback_start_step_id",
                                    "rebuild_through_step_id",
                                    "repair_goal",
                                    "active_constraints",
                                    "reconnect_boundaries",
                                    "repair_reference_span",
                                    "replay_failures",
                                }
                            },
                        },
                        complete_route_json=False,
                        minimum_route_depth=1,
                        max_reactionjson_candidates=(config.max_reactionjson_candidates_per_node),
                        paper_matched=config.paper_matched_reach_profile,
                    )
                _assert_node_prompt_size(prompt, config.max_node_prompt_bytes)
                task = _node_task(
                    spec,
                    prompt=prompt,
                    branch_index=branch_index,
                    # _node_task accepts a zero-based ordinal and renders the
                    # human-facing task id as ordinal + 1. route_call_count is
                    # already one-based, so normalize it at this boundary.
                    node_index=call_index - 1,
                    model=str(spec.metadata.get("model") or ""),
                    reasoning_effort=str(spec.metadata.get("reasoning_effort") or "medium"),
                    timeout_s=_node_call_timeout_s(
                        started,
                        route_quota,
                        maximum=config.max_node_call_timeout_s,
                    ),
                    paper_matched=config.paper_matched_reach_profile,
                    target_smiles=target,
                    selected_product=selected,
                )
                reservation, budget_reason = shared_ledger.reserve(
                    input_tokens=_BUILDER_INPUT_TOKEN_RESERVE,
                    output_tokens=_BUILDER_OUTPUT_TOKEN_RESERVE,
                )
                if reservation is None:
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "stop_search": True,
                        "stop_reason": f"route_builder_{budget_reason}",
                    }
                try:
                    record = self._run_journaled_worker(self.node_executor, task)
                except Exception:
                    shared_ledger.settle(reservation, None)
                    raise
                shared_ledger.settle(reservation, record)
                provider_failure_reason = worker_provider_failure_reason(record)
                if provider_failure_reason:
                    return {
                        "candidates": [],
                        "model_call_consumed": False,
                        "runtime_unavailable": True,
                        "runtime_pause": True,
                        "stop_search": True,
                        "stop_reason": provider_failure_reason,
                    }
                branch[builder_counter_key] = axis_call_index
                branch["call_count"] = int(branch.get("call_count") or 0) + 1
                local_records.append(record)
                route_records.append(record)
                compiled, candidate_rejections = _reactionjson_candidates_from_record(
                    record,
                    expected_product=selected,
                    mapped_product_smiles=selected_mapped,
                    require_reaction_operations=True,
                    compiler=self.routejson_compiler,
                    max_candidates=(config.max_reactionjson_candidates_per_node),
                    reserved_atom_maps=_route_atom_map_namespace(
                        prompt_steps,
                        selected_mapped,
                    ).union(
                        {
                            int(value)
                            for value in path_repair_resume.get("reserved_atom_maps") or []
                            if int(value) > 0
                        }
                    ),
                )
                branch.setdefault("reactionjson_candidate_batches", []).append(
                    {
                        "node": call_index,
                        "product_smiles": selected,
                        "reported_candidates": (len(compiled) + len(candidate_rejections)),
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
                    branch.setdefault("materialization_diagnostics", []).append(row)
                    remember_path_repair_replay_failure(row)
                candidates: list[dict[str, Any]] = []

                def admit_candidate(
                    item: _CompiledReactionJsonCandidate,
                    expansion: NodeExpansion,
                    step: Mapping[str, Any],
                    *,
                    prior_moves: dict[str, dict[str, Any]],
                    candidate_identity: str,
                ) -> None:
                    candidates.append(
                        {
                            "candidate_id": item.candidate_id,
                            "product_smiles": expansion.product_smiles,
                            "mapped_product_smiles": (expansion.mapped_product_smiles),
                            "precursor_smiles": list(expansion.precursor_smiles),
                            "mapped_precursor_smiles": list(expansion.mapped_precursor_smiles),
                            "route_step": dict(step),
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
                    prior_moves[candidate_identity] = {
                        "candidate_id": item.candidate_id,
                        "attempted_net_edits": attempted_net_edits,
                    }
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
                                "mapped_product_smiles": (expansion.mapped_product_smiles),
                                "mapped_precursor_smiles": list(expansion.mapped_precursor_smiles),
                                "reaction_operations": list(expansion.reaction_operations),
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
                            "attempted_net_edits": list(repeated.get("attempted_net_edits") or []),
                            "authority": "aizynthfinder_selected_path_state",
                        }
                        branch.setdefault("rejections", []).append(row)
                        branch.setdefault("materialization_diagnostics", []).append(row)
                        continue
                    returned_ancestors = sorted(
                        {
                            precursor
                            for precursor in (
                                _canonical_smiles(value) for value in expansion.precursor_smiles
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
                            or _generated_builder_step_id(
                                branch,
                                branch_index=branch_index,
                                call_index=call_index,
                                candidate_index=item.candidate_index,
                            )
                        ),
                        strategy_anchor=_expansion_executes_strategy_anchor(
                            expansion,
                            active_strategy_card,
                            fallback=(
                                not config.paper_matched_reach_profile and not bool(prompt_steps)
                            ),
                        ),
                        strategy_milestone_index=_strategy_milestone_index(
                            branch, active_strategy_card
                        ),
                    )
                    extension_validation = _route_steps_host_replay_validation(
                        [*prompt_steps, step],
                        mapped_target_smiles=str(
                            branch.get("target_mapped_smiles") or _mapped_smiles(target)
                        ),
                    )
                    if extension_validation.get("complete") is not True:
                        attempted_net_edits = [
                            dict(operation)
                            for operation in normalize_reaction_operations(
                                expansion.reaction_operations
                            )
                        ]
                        rejection = {
                            "phase": "route_builder_extension_replay",
                            "node": call_index,
                            "product_smiles": selected,
                            "candidate_id": item.candidate_id,
                            "reason": "candidate_does_not_extend_target_rooted_route",
                            "routejson_replay_validation": dict(extension_validation),
                            "attempted_net_edits": attempted_net_edits,
                            "authority": "host_routejson_compiler",
                        }
                        branch.setdefault("rejections", []).append(rejection)
                        branch.setdefault("materialization_diagnostics", []).append(rejection)
                        prior_moves[candidate_identity] = {
                            "candidate_id": item.candidate_id,
                            "attempted_net_edits": attempted_net_edits,
                        }
                        pending_policy_feedback[policy_context_id] = rejection
                        remember_path_repair_replay_failure(rejection)
                        continue
                    if path_repair_resume:
                        boundary_stereo_conflict = _path_repair_boundary_stereo_conflict(
                            mapped_precursor_smiles=(expansion.mapped_precursor_smiles),
                            reconnect_boundaries=(
                                path_repair_resume.get("reconnect_boundaries") or ()
                            ),
                        )
                        if boundary_stereo_conflict is not None:
                            attempted_net_edits = [
                                dict(operation)
                                for operation in normalize_reaction_operations(
                                    expansion.reaction_operations
                                )
                            ]
                            rejection = {
                                "phase": "path_repair_boundary",
                                "node": call_index,
                                "product_smiles": selected,
                                "candidate_id": item.candidate_id,
                                **boundary_stereo_conflict,
                                "attempted_net_edits": attempted_net_edits,
                                "authority": "host_suffix_boundary",
                            }
                            branch.setdefault("rejections", []).append(rejection)
                            branch.setdefault("materialization_diagnostics", []).append(rejection)
                            prior_moves[candidate_identity] = {
                                "candidate_id": item.candidate_id,
                                "attempted_net_edits": attempted_net_edits,
                            }
                            pending_policy_feedback[policy_context_id] = rejection
                            # This candidate can never reach the preserved
                            # suffix: every upstream child would still synthesize
                            # the same wrong-stereo leaf.  Leave the current AiZ
                            # action set empty so it revisits this parent for a
                            # corrected Builder sibling.
                            continue
                        boundary_progress_failure = _path_repair_boundary_progress_failure(
                            selected_leaf_mapped=selected_mapped,
                            mapped_precursor_smiles=(expansion.mapped_precursor_smiles),
                            reconnect_boundaries=(
                                path_repair_resume.get("reconnect_boundaries") or ()
                            ),
                        )
                        if boundary_progress_failure is not None:
                            attempted_net_edits = [
                                dict(operation)
                                for operation in normalize_reaction_operations(
                                    expansion.reaction_operations
                                )
                            ]
                            rejection = {
                                "phase": "path_repair_boundary",
                                "node": call_index,
                                "product_smiles": selected,
                                "candidate_id": item.candidate_id,
                                **boundary_progress_failure,
                                "attempted_net_edits": attempted_net_edits,
                                "authority": "host_suffix_boundary",
                            }
                            branch.setdefault("rejections", []).append(rejection)
                            branch.setdefault("materialization_diagnostics", []).append(rejection)
                            prior_moves[candidate_identity] = {
                                "candidate_id": item.candidate_id,
                                "attempted_net_edits": attempted_net_edits,
                            }
                            pending_policy_feedback[policy_context_id] = rejection
                            continue
                    if (
                        config.enable_key_event_critic
                        and (
                            not path_repair_resume
                            or str(path_repair_resume.get("completion_mode") or "")
                            == "strategy_checkpoint"
                        )
                        and not bool(branch.get("key_event_critic_completed"))
                        and _step_claims_strategy_key_event(step, active_strategy_card)
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
                            checkpoint_feedback=pending_checkpoint_feedback,
                        )
                        history_row: dict[str, Any] = {
                            "focus_step_id": focus_step_id,
                            "product_smiles": selected,
                            "candidate_id": item.candidate_id,
                            "fingerprint": fingerprint,
                            "strategy_id": str(active_strategy_card.get("strategy_id") or ""),
                            "strategy_digest": _strategy_card_digest(active_strategy_card),
                            "strategy_milestone_index": (
                                _strategy_milestone_index(branch, active_strategy_card)
                            ),
                            "lineage_root_mapped_smiles": selected_mapped,
                            "required_selected_step_ids": [focus_step_id],
                        }
                        history_row["obligation_id"] = _key_event_obligation_id(history_row)
                        if critic_prompt is None:
                            history_row["status"] = "prompt_unavailable"
                            branch.setdefault("key_event_critic_history", []).append(history_row)
                        else:
                            critic_task = _critic_task(
                                spec,
                                prompt=critic_prompt,
                                branch_index=branch_index,
                                iteration=call_index,
                                timeout_s=_node_call_timeout_s(
                                    started,
                                    route_quota,
                                    maximum=config.critic_call_timeout_s,
                                ),
                                paper_matched=True,
                                target_smiles=target,
                                audit_kind="key_event",
                                focus_step_id=focus_step_id,
                            )
                            critic_reservation, critic_budget_reason = shared_ledger.reserve(
                                input_tokens=_CRITIC_INPUT_TOKEN_RESERVE,
                                output_tokens=_CRITIC_OUTPUT_TOKEN_RESERVE,
                            )
                            if critic_reservation is None:
                                history_row.update(
                                    {
                                        "status": "budget_unavailable",
                                        "reason": critic_budget_reason,
                                    }
                                )
                                branch.setdefault("key_event_critic_history", []).append(
                                    history_row
                                )
                                admit_candidate(
                                    item,
                                    expansion,
                                    step,
                                    prior_moves=prior_moves,
                                    candidate_identity=candidate_identity,
                                )
                                continue
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
                                        "reasons": ["key_event_critic_execution_failed"],
                                    },
                                )
                            shared_ledger.settle(critic_reservation, critic_record)
                            if worker_provider_failure_reason(critic_record):
                                return {
                                    "candidates": [],
                                    "model_call_consumed": True,
                                    "runtime_unavailable": True,
                                    "runtime_pause": True,
                                    "stop_search": True,
                                    "stop_reason": worker_provider_failure_reason(critic_record),
                                }
                            local_records.append(critic_record)
                            branch["critic_call_count"] = (
                                int(branch.get("critic_call_count") or 0) + 1
                            )
                            branch["key_event_critic_call_count"] = (
                                int(branch.get("key_event_critic_call_count") or 0) + 1
                            )
                            critique = _bind_key_event_focus_assessment(
                                _critique_from_record(critic_record),
                                focus_step_id,
                            )
                            focus_assessment = _key_event_focus_assessment(critique, focus_step_id)
                            checkpoint_match = (
                                critique.get("checkpoint_match") is True
                                and focus_assessment is not None
                            )
                            checkpoint_verdict = str(
                                dict(focus_assessment or {}).get("verdict") or ""
                            )
                            checkpoint_rejected = focus_assessment is not None and (
                                checkpoint_verdict == "reject"
                                or focus_assessment.get("blocking") is True
                            )
                            history_row.update(
                                {
                                    "task_id": critic_task.task_id,
                                    "status": (
                                        "rejected"
                                        if checkpoint_rejected
                                        else (
                                            "completed"
                                            if checkpoint_match and checkpoint_verdict == "pass"
                                            else (
                                                "uncertain"
                                                if checkpoint_match
                                                else "not_checkpoint"
                                            )
                                        )
                                    ),
                                    "critic_status": str(critique.get("status") or "unavailable"),
                                    "checkpoint_match": checkpoint_match,
                                    "assessment": dict(focus_assessment or {}),
                                }
                            )
                            branch.setdefault("key_event_critic_history", []).append(history_row)
                            if not path_repair_resume:
                                branch["pending_key_event_feedback"] = (
                                    _pending_key_event_feedback_for_leaf(
                                        branch,
                                        strategy_card=active_strategy_card,
                                        steps=prompt_steps,
                                        selected_product_mapped=selected_mapped,
                                    )
                                )
                            if not checkpoint_match:
                                # A benign scheduling mismatch remains a
                                # preparatory action.  A false substitute may
                                # still be locally rejected below by the
                                # Critic's explicit blocking verdict.
                                step["checkpoint_relation"] = "preparatory"
                            elif checkpoint_verdict == "pass":
                                # A pass retires the active horizon only if AiZ
                                # subsequently selects this candidate into the
                                # target-rooted path.  The append-only history
                                # records the passed proposal; the next request
                                # derives completion from selected route_steps.
                                pass
                            if checkpoint_rejected:
                                repair_scope = normalize_key_event_repair_scope(
                                    focus_assessment.get("repair_scope"),
                                    verdict=checkpoint_verdict,
                                )
                                chemical_rejection = {
                                    "focus_step_id": focus_step_id,
                                    "reasons": [
                                        str(value)
                                        for value in focus_assessment.get("reasons") or []
                                        if str(value)
                                    ][:2],
                                    "suggested_revision": str(
                                        focus_assessment.get("suggested_revision") or ""
                                    ),
                                    "repair_scope": repair_scope,
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
                                if path_repair_resume:
                                    # The current transaction already owns the
                                    # mutable route span.  Reject this proposed
                                    # checkpoint edge and let AiZ request a new
                                    # sibling; never create a nested Editor
                                    # transaction or clear its constraints.
                                    continue
                                if repair_scope == "strategy_horizon":
                                    # The append-only Critic row is the next
                                    # request's replacement authority.  Clear
                                    # only the diagnostic projection; the
                                    # rejected focus edge remains unadmitted.
                                    branch["pending_key_event_feedback"] = {}
                                    continue
                                if repair_scope == "route_span":
                                    repair_blockers = _blocking_critic_steps(
                                        critique,
                                        audit_steps,
                                    )
                                    if (
                                        prompt_steps
                                        and repair_blockers
                                        and config.enable_transactional_path_repair
                                    ):
                                        branch["_pending_online_path_repair"] = {
                                            "source": "key_event_critic",
                                            "focus_step_id": focus_step_id,
                                            "authoritative_steps": [
                                                dict(row) for row in prompt_steps
                                            ],
                                            "repair_context_steps": [
                                                dict(row) for row in audit_steps
                                            ],
                                            "blocking_steps": [
                                                dict(row) for row in repair_blockers
                                            ],
                                            "critique": copy.deepcopy(critique),
                                            # The append-only Key Critic
                                            # history remains authoritative;
                                            # carry its derived same-lineage
                                            # projection across the Editor
                                            # transaction so route-span repair
                                            # cannot reintroduce an earlier
                                            # rejected mechanism or control
                                            # defect.
                                            "checkpoint_feedback": copy.deepcopy(
                                                branch.get("pending_key_event_feedback") or {}
                                            ),
                                            "strategy_card": copy.deepcopy(active_strategy_card),
                                        }
                                        branch["pending_key_event_feedback"] = {}
                                        stop_reason = "key_event_route_span_repair_required"
                                    else:
                                        # Never send an immutable-product
                                        # defect back to the same-parent
                                        # Builder.  Profiles without the
                                        # transactional Editor retain the
                                        # accepted prefix as a terminal
                                        # diagnostic instead of looping.
                                        stop_reason = "key_event_route_span_repair_unavailable"
                                    return {
                                        "candidates": [],
                                        "model_call_consumed": True,
                                        "stop_search": True,
                                        "stop_reason": stop_reason,
                                    }
                                # focus_edge is the only rejection scope a
                                # same-parent Builder can actually change.
                                continue
                    if config.enable_key_event_critic and not path_repair_resume:
                        review_disposition = self._review_selected_uncertain_key_event(
                            spec,
                            target=target,
                            branch=branch,
                            strategy_card=active_strategy_card,
                            route_steps=[*prompt_steps, step],
                            records=local_records,
                            shared_ledger=shared_ledger,
                            route_quota=route_quota,
                            config=config,
                            started=started,
                        )
                        if review_disposition.rejected:
                            return reject_selected_path(
                                review_disposition,
                                prompt_steps=[*prompt_steps, step],
                                model_call_consumed=True,
                            )
                    admit_candidate(
                        item,
                        expansion,
                        step,
                        prior_moves=prior_moves,
                        candidate_identity=candidate_identity,
                    )
                return {
                    "candidates": candidates,
                    "model_call_consumed": True,
                }

            try:
                result = run_aizynthfinder_strategy_branch_sidecar(
                    target_smiles=target,
                    mapped_target_smiles=str(
                        branch.get("target_mapped_smiles") or _mapped_smiles(target)
                    ),
                    strategy_id=strategy_id,
                    strategy_text=strategy_text,
                    request_handler=handle_request,
                    stock_index_path=self.aizynthfinder_strategy_stock_index,
                    inline_stock_smiles=(self.aizynthfinder_strategy_inline_stock_smiles),
                    python_executable=(self.aizynthfinder_strategy_python_executable),
                    max_policy_calls=max(1, phase_builder_call_ceiling),
                    max_candidates_per_call=(config.max_reactionjson_candidates_per_node),
                    max_transforms=(
                        max(
                            config.max_node_expansions_per_branch,
                            len(durable_seed_steps) + phase_builder_call_ceiling,
                        )
                        if repair_phase
                        else config.max_node_expansions_per_branch
                    ),
                    timeout_s=max(
                        1.0,
                        _remaining_node_wall_time(started, route_quota),
                    ),
                    cancel_event=self.cancel_event,
                )
            except Exception as exc:
                recovered_depth = int(branch.get("sidecar_durable_prefix_step_count") or 0)
                branch.setdefault("rejections", []).append(
                    {
                        "phase": "aizynthfinder_strategy_search",
                        "reason": "aizynthfinder_strategy_sidecar_failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "fallback_used": False,
                    }
                )
                branch[search_diagnostic_key] = {
                    "engine": "AiZynthFinder.MctsSearchTree",
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_used": False,
                    "selected_depth": recovered_depth,
                    "selected_open_leaves": len(branch.get("open_leaf_states") or []),
                }
                if recovered_depth > 0 and branch.get("steps"):
                    branch["sidecar_recovered_prefix"] = True
                    branch["complete_in_bound_stock"] = False
                    _sync_open_leaf_projection(branch)
                return local_records

            projected_steps = [
                dict(row) for row in result.get("route_steps") or [] if isinstance(row, Mapping)
            ]
            search_diagnostics = dict(result.get("diagnostics") or {})
            pending_online_repair = branch.get("_pending_online_path_repair")
            if isinstance(pending_online_repair, Mapping):
                # The stop was requested before the rejected checkpoint edge
                # entered AiZ.  The request's Host-replayed prompt prefix is
                # therefore the durable authority, even if the sidecar's
                # terminal projection selected an empty sibling after seeing
                # the stop signal.
                projected_steps = [
                    dict(row)
                    for row in pending_online_repair.get("authoritative_steps") or []
                    if isinstance(row, Mapping)
                ]
                search_diagnostics["online_path_repair_retained_host_prefix"] = True
            materialized_projection = _materialize_aizynthfinder_projection(
                steps=projected_steps,
                mapped_target_smiles=str(
                    branch.get("target_mapped_smiles") or _mapped_smiles(target)
                ),
                search_diagnostics=search_diagnostics,
                stock_membership=self._stock_membership,
            )
            branch["steps"] = [dict(row) for row in materialized_projection["steps"]]
            selected_cards = _ordered_strategy_cards_from_steps(
                root_strategy_card=root_strategy_card,
                steps=branch["steps"],
            )
            branch["strategy_milestone_cards"] = selected_cards
            if config.enable_key_event_critic and selected_cards and not repair_phase:
                final_review_disposition = self._review_selected_uncertain_key_event(
                    spec,
                    target=target,
                    branch=branch,
                    strategy_card=selected_cards[-1],
                    route_steps=branch["steps"],
                    records=local_records,
                    shared_ledger=shared_ledger,
                    route_quota=route_quota,
                    config=config,
                    started=started,
                )
                if final_review_disposition.rejected:
                    reject_selected_path(
                        final_review_disposition,
                        prompt_steps=branch["steps"],
                        model_call_consumed=False,
                    )
                    projected_steps = [dict(row) for row in branch.get("steps") or []]
                    materialized_projection = _materialize_aizynthfinder_projection(
                        steps=projected_steps,
                        mapped_target_smiles=str(
                            branch.get("target_mapped_smiles") or _mapped_smiles(target)
                        ),
                        search_diagnostics={},
                        stock_membership=self._stock_membership,
                    )
                    selected_cards = _ordered_strategy_cards_from_steps(
                        root_strategy_card=root_strategy_card,
                        steps=branch["steps"],
                    )
                    branch["strategy_milestone_cards"] = selected_cards
                    search_diagnostics.update(
                        {
                            "final_selected_path_rejected": True,
                            "final_rejected_path_step_ids": list(
                                final_review_disposition.rejected_path_step_ids
                            ),
                            "final_rejection_reason": (final_review_disposition.rejection_reason),
                        }
                    )
            if config.enable_key_event_critic and selected_cards:
                branch["key_event_critic_completed"] = _selected_path_passed_strategy_checkpoint(
                    branch,
                    strategy_card=selected_cards[-1],
                    steps=branch["steps"],
                )
                if branch["key_event_critic_completed"]:
                    branch["pending_key_event_feedback"] = {}
            _refresh_strategy_milestone_projection(
                branch,
                strategy_cards=selected_cards,
                use_key_event_critic=config.enable_key_event_critic,
            )
            branch["open_leaf_states"] = deque(
                dict(row) for row in materialized_projection["open_leaf_states"]
            )
            branch["deferred_builder_leaf_states"] = deque()
            branch["expanded_products"] = {
                _canonical_smiles(row.get("product_smiles"))
                for row in branch["steps"]
                if _canonical_smiles(row.get("product_smiles"))
            }
            branch[search_diagnostic_key] = {
                **search_diagnostics,
                "canonical_route_projection_complete": bool(
                    materialized_projection["route_projection_complete"]
                ),
                "canonical_leaf_closure_complete": bool(
                    materialized_projection["leaf_closure_complete"]
                ),
                "canonical_terminal_leaf_count": int(
                    materialized_projection["terminal_leaf_count"]
                ),
                "canonical_unresolved_leaf_count": len(materialized_projection["open_leaf_states"]),
                "canonical_routejson_replay_validation": dict(
                    materialized_projection["routejson_replay_validation"]
                ),
                "semantics": {
                    "search_selected_solved_is_diagnostic_only": True,
                    "canonical_leaf_closure_is_host_owned": True,
                },
            }
            if (
                result.get("solved") is True
                and not materialized_projection["leaf_closure_complete"]
            ):
                branch.setdefault("materialization_diagnostics", []).append(
                    {
                        "phase": "aizynthfinder_strategy_projection",
                        "reason": "search_selected_solved_not_canonically_closed",
                        "route_projection_complete": bool(
                            materialized_projection["route_projection_complete"]
                        ),
                        "routejson_replay_validation": dict(
                            materialized_projection["routejson_replay_validation"]
                        ),
                        "canonical_unresolved_leaf_count": len(
                            materialized_projection["open_leaf_states"]
                        ),
                    }
                )
            provider_callback_count = int(search_diagnostics.get("provider_callback_count") or 0)
            sidecar_model_call_count = int(result.get("policy_calls") or 0)
            actual_policy_calls = len(route_records)
            branch[search_diagnostic_key]["provider_callback_count"] = provider_callback_count
            branch[search_diagnostic_key]["actual_policy_calls"] = actual_policy_calls
            branch[search_diagnostic_key]["sidecar_model_call_count"] = sidecar_model_call_count
            branch[search_diagnostic_key]["model_call_ledger_matches"] = (
                sidecar_model_call_count == actual_policy_calls
            )
            branch[search_diagnostic_key]["reported_mcts_iterations"] = int(
                result.get("mcts_iterations") or 0
            )
            policy_calls = actual_policy_calls
            cumulative_policy_calls = int(branch.get(builder_counter_key) or 0)
            maximum_calls = builder_call_ceiling
            budget_projection = {
                "maximum_calls": maximum_calls,
                "actual_calls": cumulative_policy_calls,
                "calls_before_phase": builder_calls_before_phase,
                "phase_call_ceiling": phase_builder_call_ceiling,
                "phase_calls": policy_calls,
                "stock_closed": bool(
                    materialized_projection["route_projection_complete"]
                    and materialized_projection["leaf_closure_complete"]
                ),
                "calls_exhausted": bool(
                    search_diagnostics.get("calls_exhausted")
                    or cumulative_policy_calls >= maximum_calls
                ),
                "host_stop_requested": bool(search_diagnostics.get("host_stop_requested")),
                "host_stop_reason": str(search_diagnostics.get("host_stop_reason") or ""),
                "within_cap": cumulative_policy_calls <= maximum_calls,
                "stopped_before_cap": cumulative_policy_calls < maximum_calls,
                "semantics": {
                    "host_mcts_or_stock_may_stop_before_call_ceiling": True,
                    "call_ceiling_is_not_a_required_minimum": True,
                    "builder_has_no_terminal_authority": True,
                    "stock_and_solved_are_host_owned": True,
                    "actual_calls_come_from_worker_records": True,
                    "cumulative_within_this_call_axis": True,
                    "provider_callbacks_are_not_model_calls": True,
                },
            }
            if not repair_phase:
                budget_projection["semantics"].update(
                    {
                        "initial_policy_axis": True,
                    }
                )
                branch["paper_policy_call_budget"] = budget_projection
            _sync_open_leaf_projection(branch)
            branch["complete_in_bound_stock"] = _branch_stock_closed(branch)
            return local_records

        results: dict[int, list[WorkerRunRecord]] = {}
        maximum_workers = min(max(1, int(config.strategy_branch_workers)), branch_count)
        with ThreadPoolExecutor(
            max_workers=maximum_workers,
            thread_name_prefix="autoplanner-aizynth-strategy",
        ) as executor:
            futures = {
                executor.submit(advance, branch): int(branch["branch_index"]) for branch in seeded
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        ledger_snapshot = shared_ledger.snapshot()
        for branch in seeded:
            branch["shared_model_budget_ledger"] = ledger_snapshot
        return [record for branch_index in sorted(results) for record in results[branch_index]]

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
                and not (config.stop_on_first_stock_closed_branch and early_stop.is_set())
                and int(branch["route_call_count"]) < config.max_node_expansions_per_branch
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
        maximum_workers = min(max(1, int(config.strategy_branch_workers)), branch_count)
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
        return [record for branch_index in sorted(results) for record in results[branch_index]]

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
            if worker_provider_failure_reason(record):
                break
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
                    "deferred_builder_leaf_states": deque(),
                    "target_mapped_smiles": _mapped_smiles(target),
                    "call_count": 1,
                    "complete_in_bound_stock": False,
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
        attempt_index = int(branch["strategy_call_count"]) + 1
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
            attempt_index=attempt_index,
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
        if worker_provider_failure_reason(record):
            return
        branch["strategy_call_count"] = attempt_index
        branch["call_count"] = int(branch["call_count"]) + 1
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
        branch["lens"] = "Codex-authored strategy - " + str(branch["strategy_seed"])

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
        enhanced_strategy: bool = True,
    ) -> None:
        """Generate the paper's three competing strategies in one model call."""

        prompt = _paper_strategy_portfolio_prompt(
            target=target,
            enhanced=enhanced_strategy,
        )
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
        if worker_provider_failure_reason(record):
            return
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
            branch["lens"] = "Codex-authored strategy - " + str(branch["strategy_seed"])

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
        """Review the three self-correcting hypotheses once before paid search.

        The reviewer returns the same compact portfolio contract, so this is
        one refinement boundary rather than a second strategy authority.  An
        unavailable or invalid review leaves the generator portfolio intact.
        """

        original_cards = [dict(branch.get("strategy_card") or {}) for branch in branches]
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
        if worker_provider_failure_reason(record):
            return
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
            branch["lens"] = "Critic-reviewed strategy - " + str(branch["strategy_seed"])

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
        budget_ledger: _SharedModelCallLedger,
        paper_matched: bool = False,
        retired_strategy_feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Plan the next strategy against one exact upstream mapped leaf.

        This is intentionally receding-horizon.  Future precursor maps are not
        guessed at the target: a new StrategyCard is requested after AiZ has
        selected a non-stock leaf with no applicable active horizon.  That can
        happen after a passed checkpoint or when selection moves to a sibling.
        Only checkpoints passed by the Key Critic on this leaf's exact
        target-to-leaf spine are reported as completed facts.  An unfinished
        sibling horizon remains branch state and is restored when AiZ returns
        to that lineage.  Strategy calls are accounted separately from Route
        Builder policy calls.
        """

        branch_index = int(branch.get("branch_index") or 0)
        route_step_rows = [dict(row) for row in route_steps if isinstance(row, Mapping)]
        next_strategy_call = int(branch.get("strategy_call_count") or 0) + 1
        next_generation_count = int(branch.get("strategy_milestone_generation_count") or 0) + 1
        milestone_index = next_generation_count + 1
        selected_path_steps = _connected_path_step_rows(
            route_step_rows,
            selected_product,
            selected_product_mapped,
        )
        path_cards = _ordered_strategy_cards_from_steps(
            root_strategy_card=dict(
                branch.get("root_strategy_card") or branch.get("strategy_card") or {}
            ),
            steps=selected_path_steps,
        )
        completed_cards = [
            card
            for card in path_cards
            if _selected_path_passed_strategy_checkpoint(
                branch,
                strategy_card=card,
                steps=selected_path_steps,
            )
        ]
        prompt = _milestone_strategy_prompt(
            campaign_target=campaign_target,
            selected_product=selected_product,
            selected_product_mapped=selected_product_mapped,
            branch_index=branch_index,
            milestone_index=milestone_index,
            strategy_mandate=str(branch.get("strategy_mandate") or branch.get("lens") or ""),
            completed_strategy_cards=completed_cards,
            route_steps=route_step_rows,
            retired_strategy_feedback=retired_strategy_feedback,
        )
        _assert_node_prompt_size(prompt, max_prompt_bytes)
        task = _strategy_task(
            spec,
            prompt=prompt,
            branch_index=branch_index,
            attempt_index=next_strategy_call,
            model=str(spec.metadata.get("model") or ""),
            reasoning_effort=str(spec.metadata.get("reasoning_effort") or "medium"),
            timeout_s=_node_call_timeout_s(
                started,
                quota,
                maximum=max_node_call_timeout_s,
            ),
            paper_matched=paper_matched,
            target_smiles=selected_product,
        )
        reservation, budget_reason = budget_ledger.reserve(
            input_tokens=_CRITIC_INPUT_TOKEN_RESERVE,
            output_tokens=_CRITIC_OUTPUT_TOKEN_RESERVE,
        )
        if reservation is None:
            branch.setdefault("strategy_milestone_attempts", []).append(
                {
                    "milestone_index": milestone_index,
                    "selected_product_smiles": selected_product,
                    "selected_product_mapped_smiles": selected_product_mapped,
                    "accepted": False,
                    "reason": f"shared_model_budget:{budget_reason}",
                }
            )
            return None
        try:
            record = self._run_journaled_worker(self.node_executor, task)
        except Exception:
            budget_ledger.settle(reservation, None)
            raise
        budget_ledger.settle(reservation, record)
        records.append(record)
        if worker_provider_failure_reason(record):
            return None
        branch["strategy_call_count"] = next_strategy_call
        branch["strategy_milestone_generation_count"] = next_generation_count
        branch["call_count"] = int(branch.get("call_count") or 0) + 1
        card = _strategy_card_from_record(
            record,
            expected_target=selected_product,
            expected_mapped_target=selected_product_mapped,
            paper_matched=paper_matched,
        )
        attempt = {
            "milestone_index": milestone_index,
            "selected_product_smiles": selected_product,
            "selected_product_mapped_smiles": selected_product_mapped,
            "task_id": task.task_id,
            "accepted": False,
        }
        if card is None:
            attempt["reason"] = _strategy_card_rejection_reason(
                record,
                expected_target=selected_product,
                expected_mapped_target=selected_product_mapped,
                paper_matched=paper_matched,
            )
            branch.setdefault("strategy_milestone_attempts", []).append(attempt)
            return None

        if not paper_matched:
            card = dict(card)
            card["host_lineage"] = {
                "root_mapped_smiles": selected_product_mapped,
                "milestone_index": milestone_index,
            }
            attempt["accepted"] = True
            attempt["strategy_id"] = str(card.get("strategy_id") or "")
            attempt["strategy_digest"] = str(card.get("strategy_digest") or "")
            branch.setdefault("strategy_milestone_attempts", []).append(attempt)
            branch.setdefault("strategy_milestone_cards", []).append(dict(card))
            branch["key_event_critic_completed"] = False
            branch["pending_key_event_feedback"] = {}
            return card

        critic_prompt = _upstream_strategy_critic_prompt(
            campaign_target=campaign_target,
            selected_product=selected_product,
            selected_product_mapped=selected_product_mapped,
            branch_index=branch_index,
            milestone_index=milestone_index,
            generated_card=card,
            completed_strategy_cards=completed_cards,
            accepted_route_steps=route_step_rows,
            retired_strategy_feedback=retired_strategy_feedback,
        )
        _assert_node_prompt_size(critic_prompt, max_prompt_bytes)
        critic_task = _upstream_strategy_critic_task(
            spec,
            prompt=critic_prompt,
            branch_index=branch_index,
            milestone_index=milestone_index,
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
            target_smiles=selected_product,
        )
        critic_reservation, critic_budget_reason = budget_ledger.reserve(
            input_tokens=_CRITIC_INPUT_TOKEN_RESERVE,
            output_tokens=_CRITIC_OUTPUT_TOKEN_RESERVE,
        )
        if critic_reservation is None:
            attempt["reason"] = f"strategy_critic_shared_model_budget:{critic_budget_reason}"
            branch.setdefault("strategy_milestone_attempts", []).append(attempt)
            return None
        try:
            critic_record = self._run_journaled_worker(self.critic_executor, critic_task)
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
                    "reasons": ["strategy_milestone_critic_execution_failed"],
                },
            )
        budget_ledger.settle(critic_reservation, critic_record)
        records.append(critic_record)
        if worker_provider_failure_reason(critic_record):
            return None
        branch["strategy_critic_call_count"] = (
            int(branch.get("strategy_critic_call_count") or 0) + 1
        )
        branch["call_count"] = int(branch.get("call_count") or 0) + 1
        reviewed_card = _strategy_card_from_record(
            critic_record,
            expected_target=selected_product,
            expected_mapped_target=selected_product_mapped,
            paper_matched=True,
        )
        critic_history = {
            "milestone_index": milestone_index,
            "task_id": critic_task.task_id,
            "accepted": reviewed_card is not None,
        }
        branch.setdefault("strategy_milestone_critic_history", []).append(critic_history)
        if reviewed_card is None:
            attempt["critic_task_id"] = critic_task.task_id
            attempt["reason"] = _strategy_card_rejection_reason(
                critic_record,
                expected_target=selected_product,
                expected_mapped_target=selected_product_mapped,
                paper_matched=True,
            )
            branch.setdefault("strategy_milestone_attempts", []).append(attempt)
            return None

        card = dict(reviewed_card)
        card["host_lineage"] = {
            "root_mapped_smiles": selected_product_mapped,
            "milestone_index": milestone_index,
        }
        attempt["accepted"] = True
        attempt["critic_task_id"] = critic_task.task_id
        attempt["strategy_id"] = str(card.get("strategy_id") or "")
        attempt["strategy_digest"] = str(card.get("strategy_digest") or "")
        branch.setdefault("strategy_milestone_attempts", []).append(attempt)
        branch.setdefault("strategy_milestone_cards", []).append(dict(card))
        # These two fields describe only the currently active milestone.  The
        # append-only key_event_critic_history retains prior checkpoint facts,
        # so resetting them does not erase or duplicate branch history.
        branch["key_event_critic_completed"] = False
        branch["pending_key_event_feedback"] = {}
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
        compiler_first = bool(require_strategy_graph_edits and not require_complete_route_json)
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
                or_search.node_state(selected_or_node) if selected_or_node is not None else None
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
        call_index = int(branch["route_call_count"]) + 1
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
        if worker_provider_failure_reason(record):
            return
        branch["route_call_count"] = call_index
        branch["call_count"] = int(branch["call_count"]) + 1
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
                        branch.pop("_last_materialization_editor_failure", {}) or diagnostic
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
                    1 if materialization_editor_attempted else _MATERIALIZATION_RETRY_LIMIT
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
                    deferred = branch.setdefault("deferred_builder_leaf_states", deque())
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
                        expansion.step_id or f"codex:branch:{branch_index + 1}:{len(steps) + 1}"
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
                    or _generated_builder_step_id(
                        branch,
                        branch_index=branch_index,
                        call_index=call_index,
                        candidate_index=item.candidate_index,
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
                    mapped_precursor_smiles=tuple(expansion.mapped_precursor_smiles),
                    route_step=step,
                    score=item.score,
                    cost=item.cost,
                    candidate_key=item.candidate_key,
                )
            )

        terminal_precursors = tuple(
            dict.fromkeys(
                precursor for candidate in or_candidates for precursor in candidate.precursor_smiles
            )
        )
        membership = self._stock_membership(terminal_precursors)
        inserted = search.expand(
            selected_node,
            or_candidates,
            stock_smiles=(
                precursor for precursor in terminal_precursors if membership.get(precursor) is True
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
            search.defer_failed_node(selected_node)
            branch.setdefault("rejections", []).append(
                {
                    "phase": "route_builder",
                    "node": call_index,
                    "product_smiles": selected_product,
                    "reason": "materialization_retry_limit_reached",
                    "strategy_retained": True,
                    "or_backtrack_enabled": True,
                    "normal_builder_continuation_required": True,
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
            dict(value) for value in payload.get("candidates") or [] if isinstance(value, Mapping)
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
            editor_feedback["host_replayed_route_scaffold"] = _compact_route_rows(current_route)
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
                            editor_feedback.get("route_builder_materialization_failure", {}).get(
                                "reason"
                            )
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
            if worker_provider_failure_reason(editor_record):
                return None
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
                    editor_feedback.get("route_builder_materialization_failure") or {}
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
                        dict(value) for value in replacement if isinstance(value, Mapping)
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
    """Protect only untouched routes' first Critic after the current call.

    Editor rounds and their follow-up Critics become eligible only after an
    actual blocking verdict. Reserving all configured repair rounds here was
    the static-budget defect that starved otherwise healthy Builder paths.
    The unused parameters remain explicit because the caller's state-machine
    position is useful diagnostic context and keeps this compatibility helper
    stable for saved-result replay tests.
    """

    del iteration, max_rounds
    future_critics = 0
    for value in branches[current_index + 1 :]:
        branch = dict(value)
        if not branch.get("steps") or dict(branch.get("chemical_critic") or {}).get("status"):
            continue
        # Preserve one mandatory initial Critic for every untouched future
        # route. Its optional Editor loop is budget-checked when that route
        # becomes current; reserving every hypothetical repair here can block
        # the current route's required post-Editor Critic.
        future_critics += 1
    return future_critics, 0


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
            + (
                config.max_node_expansions_per_branch
                if config.enable_transactional_path_repair
                else 0
            )
            + 2
            + 2 * config.max_route_local_repair_rounds
        )
        + (
            config.strategy_branch_count
            * config.max_node_expansions_per_branch
            * config.max_reactionjson_candidates_per_node
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
    completed_rows = [row for row in rows if not worker_provider_failure_reason(row)]
    if len(completed_rows) + max(0, int(reserve_model_invocations)) >= quota.model_invocations:
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
        for row in completed_rows
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
        for row in completed_rows
    )
    if input_tokens + max(0, int(reserve_input_tokens)) >= quota.input_tokens:
        return "input_token_allocation_exhausted"
    if output_tokens + max(0, int(reserve_output_tokens)) >= quota.output_tokens:
        return "output_token_allocation_exhausted"
    if _remaining_node_wall_time(started, quota) <= (
        max(0.0, float(reserve_wall_time_s)) + _deadline_settlement_reserve_s(quota)
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
    usable = _remaining_node_wall_time(started, quota) - _deadline_settlement_reserve_s(quota)
    return max(0.001, min(float(maximum), usable))


def _public_branch(branch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_index": int(branch.get("branch_index") or 0),
        "lens": str(branch.get("lens") or ""),
        "strategy_seed": str(branch.get("strategy_seed") or ""),
        "strategy_seed_source": str(branch.get("strategy_seed_source") or "generated"),
        "strategy_seed_sha256": str(branch.get("strategy_seed_sha256") or ""),
        "strategy_tree_engine": str(branch.get("strategy_tree_engine") or "chemenzy_best_first"),
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
        "strategic_milestone_count": int(branch.get("strategic_milestone_count") or 0),
        "steps": [dict(row) for row in branch.get("steps") or []],
        "open_leaves": list(branch.get("open_leaves") or []),
        "open_leaf_states": [
            dict(row) for row in branch.get("open_leaf_states") or [] if isinstance(row, Mapping)
        ],
        "deferred_builder_leaf_states": [
            dict(row)
            for row in branch.get("deferred_builder_leaf_states") or []
            if isinstance(row, Mapping)
        ],
        "call_count": int(branch.get("call_count") or 0),
        "strategy_call_count": int(branch.get("strategy_call_count") or 0),
        "route_call_count": int(branch.get("route_call_count") or 0),
        "editor_attempt_count": int(branch.get("editor_attempt_count") or 0),
        "editor_call_count": int(branch.get("editor_call_count") or 0),
        "editor_applied_count": int(branch.get("editor_call_count") or 0),
        "complete_in_bound_stock": bool(branch.get("complete_in_bound_stock")),
        "reactionjson_or_search": dict(branch.get("reactionjson_or_search") or {}),
        "aizynthfinder_strategy_search": dict(branch.get("aizynthfinder_strategy_search") or {}),
        "portfolio_early_stop_triggered": bool(branch.get("portfolio_early_stop_triggered")),
        "rejections": [dict(row) for row in branch.get("rejections") or []],
        "blocked_materializations": list(branch.get("blocked_materializations") or []),
        "materialization_failures": dict(branch.get("materialization_failures") or {}),
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
            dict(row) for row in branch.get("editor_repairs") or [] if isinstance(row, Mapping)
        ],
        "path_repair_transactions": [
            dict(row)
            for row in branch.get("path_repair_transactions") or []
            if isinstance(row, Mapping)
        ],
        "route_alternatives": [
            dict(row) for row in branch.get("route_alternatives") or [] if isinstance(row, Mapping)
        ],
        "editor_execution_notes": [
            dict(row)
            for row in branch.get("editor_execution_notes") or []
            if isinstance(row, Mapping)
        ],
        "editor_working_route": dict(branch.get("editor_working_route") or {}),
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
    if str(card.get("strategy_basis") or "") == ("paper-matched one-sentence steering query"):
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

    if str(card.get("strategy_basis") or "") == ("paper-matched one-sentence steering query"):
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
    precursor_pairs = [_parse_strategy_bond_pair(value) for value in precursor_values]
    precursor_pairs = [pair for pair in precursor_pairs if pair is not None]
    mapped_atoms = {atom for pair in target_pairs for atom in pair}
    if precursor_pairs and not all(
        left in mapped_atoms and right in mapped_atoms for left, right in precursor_pairs
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
            and _strategy_bucket(normalized_candidate) == _strategy_bucket(normalized_prior)
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
    """Return graph-derived topology aids without inventing scaffold names.

    RDKit exposes a cycle basis, not the ordered ring-system notation used by
    synthetic chemists.  Keep the basis explicitly unordered and report only
    actual ring junctions so Strategy workers cannot turn a sorted size list
    into a false ``A/B/C/D`` scaffold assignment.
    """

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {}
    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
    junction_counts: Counter[tuple[tuple[int, int], int]] = Counter()
    for left_index, left in enumerate(rings):
        for right in rings[left_index + 1 :]:
            shared_atom_count = len(left & right)
            if shared_atom_count:
                cycle_basis_sizes = tuple(sorted((len(left), len(right))))
                junction_counts[(cycle_basis_sizes, shared_atom_count)] += 1
    ring_neighbors: dict[int, set[int]] = {index: set() for index in range(len(rings))}
    for left_index, left in enumerate(rings):
        for right_index in range(left_index + 1, len(rings)):
            if left & rings[right_index]:
                ring_neighbors[left_index].add(right_index)
                ring_neighbors[right_index].add(left_index)
    ring_systems: list[dict[str, Any]] = []
    unseen = set(ring_neighbors)
    while unseen:
        pending = [min(unseen)]
        component: set[int] = set()
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(ring_neighbors[index] - component)
        unseen -= component
        system_rings = [rings[index] for index in sorted(component)]
        overlaps = [
            len(system_rings[left_index] & system_rings[right_index])
            for left_index in range(len(system_rings))
            for right_index in range(left_index + 1, len(system_rings))
            if system_rings[left_index] & system_rings[right_index]
        ]
        ring_systems.append(
            {
                "cycle_rank": len(system_rings),
                "cycle_basis_sizes_unordered": sorted(len(ring) for ring in system_rings),
                "atom_count": len(set().union(*system_rings)),
                "fused_pair_count": sum(value == 2 for value in overlaps),
                "spiro_pair_count": sum(value == 1 for value in overlaps),
                "bridged_overlap_pair_count": sum(value > 2 for value in overlaps),
            }
        )
    ring_systems.sort(
        key=lambda row: (
            -int(row["cycle_rank"]),
            -int(row["atom_count"]),
            tuple(row["cycle_basis_sizes_unordered"]),
        )
    )
    return {
        "cycle_rank": len(rings),
        "cycle_basis_sizes_unordered": sorted(len(ring) for ring in rings),
        "ring_systems": ring_systems,
        "ring_junction_topology": [
            {
                "cycle_basis_sizes_unordered": list(cycle_basis_sizes),
                "shared_atom_count": shared_atom_count,
                "pair_count": pair_count,
            }
            for (cycle_basis_sizes, shared_atom_count), pair_count in sorted(
                junction_counts.items()
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
        values.extend(str(value) for value in step.get("mapped_precursor_smiles") or [])
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


def _reaction_operation_atom_maps(
    operations: Iterable[Mapping[str, Any]],
) -> frozenset[int]:
    """Return the existing atom identities touched by a graph program."""

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
    return frozenset(required_maps)


def _path_repair_focus_atom_maps(path_repair: Mapping[str, Any]) -> frozenset[int]:
    """Derive the mutable repair center from its Host-replayed reference."""

    reference_rows = [
        dict(row)
        for row in path_repair.get("repair_reference_span") or ()
        if isinstance(row, Mapping)
    ]
    rejected_rows = [
        row
        for row in reference_rows
        if str(dict(row.get("prior_key_critic") or {}).get("status") or "") == "rejected"
        or str(dict(row.get("prior_key_critic") or {}).get("verdict") or "") == "reject"
    ]
    focus_rows = rejected_rows or reference_rows
    return frozenset(
        atom_map
        for row in focus_rows
        for atom_map in _reaction_operation_atom_maps(
            normalize_reaction_operations(row.get("reaction_operations") or ())
        )
    )


def _path_repair_focus_leaf_indices(
    *,
    selectable_indices: Iterable[int],
    mapped_product_smiles: Sequence[str],
    path_repair: Mapping[str, Any],
) -> tuple[int, ...]:
    """Keep repair continuation on the component carrying its reaction center.

    If the reference has no usable operation maps, or all current components
    have lost them, retain the existing AiZ choices.  This makes the identity
    correction structural where it is provable without adding a new blocker
    for older or generic repair records.
    """

    candidates = tuple(int(value) for value in selectable_indices)
    focus_maps = _path_repair_focus_atom_maps(path_repair)
    if len(candidates) < 2 or not focus_maps:
        return candidates
    scores = {
        index: len(
            focus_maps
            & _route_atom_map_namespace(
                (),
                mapped_product_smiles[index] if 0 <= index < len(mapped_product_smiles) else "",
            )
        )
        for index in candidates
    }
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return candidates
    return tuple(index for index in candidates if scores[index] == best_score)


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


def _paper_strategy_portfolio_prompt(*, target: str, enhanced: bool = True) -> str:
    topology_profile = _target_topology_profile(target)
    if not enhanced:
        topology_profile = {
            key: value for key, value in topology_profile.items() if key != "ring_systems"
        }
    context = {
        "schema_version": "paper_matched_strategy_portfolio_input.v1",
        "phase": "strategy_generator",
        "campaign_target": target,
        "target_topology_profile": topology_profile,
        "strategy_count": 3,
    }
    if not enhanced:
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
    return "\n".join(
        [
            "Act as the paper-matched Strategy Generator and create exactly three independent high-level strategies in this single call.",
            "Internally generate more than three possibilities, compare and attack their weakest chemical assumptions across the paper's four dimensions: scaffold/backbone, one or two key forward reactions, functional-group/protection compatibility, and stereochemical construction or control. Return only the three survivors; do not expose the internal debate.",
            "Use target_topology_profile.ring_systems only as a graph-derived aid for identifying the principal connected ring system. cycle_basis_sizes_unordered is an unordered, non-unique cycle basis, not a chemist's ordered A/B/C/D ring assignment; never rewrite that sorted list as an ordered x/y/z scaffold name. Infer the actual scaffold from campaign_target. For a fused, bridged, spiro, or otherwise complex polycyclic target, every surviving card must say how that principal scaffold/backbone is constructed, reorganized, or inherited from a specifically simpler scaffold. Installing only a side chain or peripheral ring while silently assuming the same complex core is already available is a late-stage tactic, not a complete Strategy card.",
            "For each card, output one strategy_query sentence, one critical_assumption sentence, and one critic_checkpoint sentence. strategy_query identifies the current route horizon: the high-level principal-scaffold construction logic, the reactive-handle motif that enables its first decisive event, and the main stereochemical or functional-group control. It need not enumerate the complete route or every ring closure. critical_assumption names the make-or-break chemical claim. critic_checkpoint is the earliest non-substitutable graph transformation that directly tests that assumption; a downstream event that could succeed while the assumption remains false, or a preparatory handle installation/unmasking, is not a valid checkpoint.",
            "Keep each horizon operational: any proposed multi-bond construction must name a consumable reactive-handle motif and a credible source of regio-, termination-, and stereochemical control. Do not hide several unsupported C-H bond formations or independent reactions inside one named cascade.",
            "The three strategy_query values must differ materially in the principal scaffold's skeletal construction or reorganization and key transformation logic, not merely in a peripheral appendage, reagents, or labels. Do not let all three cards inherit the same unexplained complex core.",
            "A chemical, chiral-pool, or chemoenzymatic horizon is eligible. Use a biological transformation only when the exact substrate-to-product change and its selectivity advantage are chemically credible; natural biosynthetic origin alone is not evidence that one callable enzyme can build the target core.",
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
        "schema_version": "strategy_portfolio_critic_input.v1",
        "phase": "strategy_portfolio_review",
        "campaign_target": target,
        "target_topology_profile": _target_topology_profile(target),
        "strategy_cards": [
            {
                key: value
                for key, value in dict(card).items()
                if key
                in {
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
            "Act as the independent Strategy Critic for one three-card portfolio before Route Builder search begins.",
            "Use target_topology_profile only as a graph-derived topology aid. ring_junction_topology reports actual shared-atom junctions and ring_systems identifies connected ring systems. cycle_basis_sizes_unordered is an unordered, non-unique cycle basis, not a chemist's ordered A/B/C/D ring assignment; never rewrite the sorted values as an ordered x/y/z scaffold name or infer fused/spiro relationships from them alone. Infer the actual scaffold from campaign_target. Challenge whether each named key construction can plausibly account for the target's backbone and stereochemical burden, whether the stated reactive-handle motif is sufficient at a high level, and whether critical_assumption identifies the real make-or-break claim. Require critic_checkpoint to be the earliest non-substitutable graph transformation that directly tests that claim; reject a downstream event that could occur even if the critical assumption or an earlier required key construction never occurred.",
            "Reject or minimally revise a horizon whose claimed multi-bond event lacks consumable reactive handles, whose regio-, termination-, or stereochemical control is only an adjective, or which hides several unsupported C-H bond formations or independent reactions inside one cascade label.",
            "Review the three cards as a portfolio. For a complex polycyclic target, three cards that install different peripheral groups but all assume the same unexplained principal core are one redundant strategy family and must be replaced with materially different core-construction or core-reorganization logics. Keep useful directions, but do not turn a strategy into a complete route or a required-map checklist.",
            "Copy every acceptable card verbatim when it is chemically and portfolio-level acceptable. A specific chemical contradiction, a non-testing checkpoint, peripheral-only scope, or a shared unexplained complex core is sufficient reason to revise or replace a card; preserve every unchallenged reactive-handle identity, protection or masking requirement, tether or precursor geometry clause, stereochemical-control clause, and sequencing constraint; never paraphrase merely for brevity or style.",
            "Do not make an acceptable card more specific by adding a named downstream reaction, reactive pair, catalyst, ligand, reagent, or mechanism that the Strategy Generator did not propose. Added detail is not criticism. When one concrete defect requires revision, change only the contradicted clause; replace the whole card only when its principal scaffold logic is itself unusable, and keep any replacement at the same high-level Strategy granularity.",
            "Return exactly three reviewed cards. Each card contains only strategy_query, critical_assumption, and critic_checkpoint, one concise sentence each. Do not expose the critique, score cards, write ReactionJSON, propose precursor structures or conditions, browse, inspect stock, or claim admission, validation, or solved status.",
            "StrategyPortfolioCriticInput:",
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
                "Keep the event operational: name its consumable reactive-handle motif and the actual source of regio-, termination-, and stereochemical control; do not hide unsupported C-H bond formations or independent reactions inside one cascade label.",
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
        "strategy_generation_version": ("v2" if "strategy_v2_slot=" in lens else "v1"),
        "phase": "strategy_generator",
        "campaign_target": target,
        "campaign_target_mapped": _mapped_smiles(target),
        "campaign_target_profile": _structure_profile(target),
        "campaign_target_bond_pairs": [
            f"map {left}-map {right}" for left, right in sorted(_target_bond_pairs(target))
        ],
        "branch_id": branch_index + 1,
        "strategy_lens": lens,
        "forbidden_root_strategies": [
            {
                "key_bond_signature": list(dict(card).get("key_bond_signature") or []),
                "topology_signature": str(dict(card).get("topology_signature") or ""),
                "execution_domain": str(dict(card).get("execution_domain") or ""),
            }
            for card in forbidden_strategy_cards
        ],
        "prior_strategy_rejections": [
            dict(row) for row in prior_rejections if row.get("phase") == "strategy_generator"
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


def _compact_retired_strategy_feedback(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(value or {})
    card = dict(raw.get("strategy_card") or {})
    assessment = dict(raw.get("assessment") or {})
    if not card and not assessment:
        return {}
    return {
        "strategy_query": str(card.get("strategy_query") or "")[:500],
        "critical_assumption": str(card.get("critical_assumption") or "")[:420],
        "critic_checkpoint": str(card.get("critic_checkpoint") or "")[:420],
        "blocking_type": str(assessment.get("blocking_type") or "none"),
        "reasons": [
            str(reason)[:260] for reason in assessment.get("reasons") or [] if str(reason).strip()
        ][:2],
        "suggested_revision": str(assessment.get("suggested_revision") or "")[:420],
    }


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
    retired_strategy_feedback: Mapping[str, Any] | None = None,
) -> str:
    context = _strategy_horizon_context(
        campaign_target=campaign_target,
        selected_product=selected_product,
        selected_product_mapped=selected_product_mapped,
        branch_index=branch_index,
        milestone_index=milestone_index,
        completed_strategy_cards=completed_strategy_cards,
        route_steps=route_steps,
        phase="strategy_horizon_generation",
    )
    context["strategy_lens"] = strategy_mandate
    retired_strategy = _compact_retired_strategy_feedback(retired_strategy_feedback)
    if retired_strategy:
        context["retired_strategy"] = retired_strategy
    return "\n".join(
        [
            "Act only as the Strategy Generator for the next route horizon inside an existing retrosynthesis branch.",
            "The exact selected_upstream_leaf_mapped, connected_path_reactions, completed_milestones, and current_split_context are one Host-derived leaf-lineage projection. Preserve that target-rooted reaction spine, but plan only for the selected molecular occurrence; a co-precursor marked expanded belongs to a sibling lineage and is context, not this leaf's reaction history.",
            "When retired_strategy is present, the Key Critic has rejected that horizon at this exact leaf because its checkpoint or critical assumption is not locally repairable. Replace its route-defining graph transformation; do not relabel the same checkpoint or merely swap reagents.",
            "Internally compare plausible leaf-local directions and choose the strongest next route-defining construction, scaffold reorganization, stereochemical relay, or convergent simplification. If the leaf still contains a complex principal ring system, explain its next meaningful decomplexification; a peripheral FGI or appendage edit is not the new Strategy unless the principal scaffold is already simple.",
            "The chosen horizon must be operational at Strategy granularity: name the consumable reactive-handle motif and the source of regio-, termination-, and stereochemical control, without hiding unsupported C-H bond formations or independent reactions inside one cascade label.",
            "Return only one concise strategy_query, one critical_assumption, and one critic_checkpoint. The query states the new horizon and enabling motif, not a complete route. The checkpoint is the earliest actual graph transformation that tests the assumption, not preparatory handle installation.",
            "Do not repeat a completed milestone, propose precursor SMILES, write ReactionJSON, predict conditions, search literature, or use stock availability. Route Builder will execute the selected horizon one reaction at a time.",
            "A biological step is optional and must name a credible substrate-to-product transformation and selectivity advantage in the same compact query; otherwise retain a chemical or chiral-pool direction.",
            "Return one compact StrategyCardReport whose target_smiles is exactly selected_upstream_leaf. The card is a hypothesis and grants no route, reaction, evidence, stock, or solved authority.",
            "BlindUpstreamStrategyMilestoneInput:",
            json.dumps(
                context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )


def _upstream_strategy_critic_prompt(
    *,
    campaign_target: str,
    selected_product: str,
    selected_product_mapped: str,
    branch_index: int,
    milestone_index: int,
    generated_card: Mapping[str, Any],
    completed_strategy_cards: Iterable[Mapping[str, Any]],
    accepted_route_steps: Iterable[Mapping[str, Any]],
    retired_strategy_feedback: Mapping[str, Any] | None = None,
) -> str:
    context = _strategy_horizon_context(
        campaign_target=campaign_target,
        selected_product=selected_product,
        selected_product_mapped=selected_product_mapped,
        branch_index=branch_index,
        milestone_index=milestone_index,
        completed_strategy_cards=completed_strategy_cards,
        route_steps=accepted_route_steps,
        phase="strategy_horizon_review",
    )
    context["generated_card"] = {
        "strategy_query": str(generated_card.get("strategy_query") or ""),
        "critical_assumption": str(generated_card.get("critical_assumption") or ""),
        "critic_checkpoint": str(generated_card.get("critic_checkpoint") or ""),
    }
    retired_strategy = _compact_retired_strategy_feedback(retired_strategy_feedback)
    if retired_strategy:
        context["retired_strategy"] = retired_strategy
    return "\n".join(
        [
            "Act as the Strategy Critic for one newly proposed upstream horizon.",
            "If retired_strategy is present, reject any generated_card that repeats or paraphrases its route-defining checkpoint or preserves the same disproven critical assumption; a reagent rename is not a new Strategy.",
            "The selected leaf, connected reaction spine, completed milestones, and current split are one Host-derived molecular-occurrence lineage. Audit whether the generated horizon can synthesize the exact selected_upstream_leaf while remaining chemically and sequentially compatible with that downstream spine. A co-precursor marked expanded belongs to a sibling lineage; use it for split compatibility but never treat its upstream reactions as this leaf's history.",
            "selected_upstream_leaf_stereo, when present, is the Host's compact RDKit observation of stereochemistry already encoded in the selected leaf. Use it to distinguish a center or alkene geometry that the proposed checkpoint can actually create or alter from one that already exists and is untouched by that event; it is not selectivity evidence.",
            "Copy the three generated sentences verbatim unless a concrete chemical contradiction, conflict with the accepted prefix, repeated milestone, or non-atomic checkpoint requires correction. Do not invent a more specific named reaction merely to make the card sound detailed, and preserve every unchallenged handle, protection, geometry, stereochemical-control, and sequencing clause.",
            "A Strategy horizon is not required to be the next Builder reaction. The Builder may first perform separate protection, redox, unmasking, or reactive-handle installation steps; audit their compatibility and ordering without replacing the route-defining horizon or its checkpoint with one of those preparatory reactions.",
            "While the selected leaf still has a complex principal scaffold, every revision or replacement must retain route-defining scaffold construction, reorganization, stereochemical relay, or convergent-simplification granularity. A peripheral functional-group adjustment is not a Strategy checkpoint unless the principal scaffold is already simple and no route-defining scaffold problem remains.",
            "The checkpoint must name the earliest fact observable immediately after one reaction. Retain every make-or-break structural or stereochemical outcome created in that same event when critical_assumption depends on it; do not weaken such a checkpoint to bond formation alone. Conversely, do not require a stereocenter, bond, or oxidation state created only by a genuinely later reaction.",
            "Reject or minimally revise a horizon whose multi-bond construction lacks consumable reactive handles, whose regio-, termination-, or stereochemical control is only an adjective, or which hides unsupported C-H bond formations or independent reactions inside one cascade label.",
            "Return only one compact StrategyCardReport for the exact selected_upstream_leaf, containing strategy_query, critical_assumption, and critic_checkpoint. Do not expose critique, alternatives, scores, precursor structures, ReactionJSON, conditions, evidence, or a route.",
            "UpstreamStrategyCheckpointReviewInput:",
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
        task_id=(f"{spec.agent_id}:branch:{branch_index + 1}:strategy:{attempt_index}"),
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


def _upstream_strategy_critic_task(
    spec: AgentSpec,
    *,
    prompt: str,
    branch_index: int,
    milestone_index: int,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
    target_smiles: str,
) -> WorkerTask:
    return replace(
        _strategy_task(
            spec,
            prompt=prompt,
            branch_index=branch_index,
            attempt_index=milestone_index,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            paper_matched=True,
            target_smiles=target_smiles,
        ),
        task_id=(
            f"{spec.agent_id}:branch:{branch_index + 1}:strategy-milestone:{milestone_index}:critic"
        ),
        case_id=_opaque_strategy_case_id(
            spec.run_id + f":branch:{branch_index + 1}:milestone:{milestone_index}:critic"
        ),
        task_type="paper_matched_strategy_critic",
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
        card = normalize_strategy_card(_paper_matched_strategy_card_payload(raw_card))
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
    route_validation = dict(row.get("routejson_replay_validation") or {})
    raw_chemical = dict(row.get("chemical_rejection") or {})
    is_chemical = bool(raw_chemical) or str(row.get("phase") or "") == ("key_event_critic")
    diagnostic: dict[str, Any] = {}
    for key in (
        "reason",
        "replay_error",
        "compiler_error",
        "step_index",
        "operation_index",
        "failed_operation",
        "failure_stage",
        "failure_detail",
        "detail",
        "endpoint_aromaticity",
        "allowed_orders",
        "invalidated_bond_stereo",
        "required_repair",
        "boundary_step_id",
        "boundary_product_smiles",
        "precursor_index",
        "actual_mapped_precursor_smiles",
        "selected_boundary_distance",
        "candidate_boundary_distance",
        "stereo_mismatch_atom_maps",
        "stereo_mismatch_bond_maps",
    ):
        candidate = nested.get(key)
        if candidate in (None, "", [], {}):
            candidate = route_validation.get(key)
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
            raw_chemical.get("suggested_revision") or row.get("suggested_revision") or ""
        ).strip()[:420]
        compact["chemical_rejection"] = {
            "focus_step_id": str(
                raw_chemical.get("focus_step_id") or row.get("focus_step_id") or ""
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
        for operation in normalize_reaction_operations(row.get("attempted_net_edits") or ())
    ]
    if attempted_net_edits:
        compact["attempted_net_edits"] = attempted_net_edits
    mcts_state_fingerprint = str(row.get("mcts_state_fingerprint") or "")
    if mcts_state_fingerprint:
        compact["mcts_state_fingerprint"] = mcts_state_fingerprint
    return compact


def _merge_path_repair_replay_failure(
    existing: Iterable[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Persist one deterministic Host replay failure for the repair transaction.

    A local repair may move to a different descendant leaf after every accepted
    edge. Leaf-local rejection memory therefore cannot prevent the same invalid
    ReactionJSON operation from recurring later in the same transaction. Keep
    only the failed operation and its causal Host error, deduplicated across
    leaves; product structures and the surrounding route remain authoritative
    elsewhere and are deliberately not copied here.
    """

    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}

    def add(row: Mapping[str, Any]) -> None:
        failed_operation = row.get("failed_operation")
        replay_error = str(row.get("replay_error") or row.get("compiler_error") or "").strip()
        if not replay_error or not isinstance(failed_operation, Mapping):
            return
        operation = dict(failed_operation)
        identity = json.dumps(
            {"replay_error": replay_error, "failed_operation": operation},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        occurrence_count = max(1, int(row.get("occurrence_count") or 1))
        if identity in positions:
            prior = merged[positions[identity]]
            prior["occurrence_count"] = int(prior["occurrence_count"]) + occurrence_count
            return
        compact = {
            "replay_error": replay_error,
            "failed_operation": operation,
            "occurrence_count": occurrence_count,
        }
        for key in (
            "failure_stage",
            "failure_detail",
            "endpoint_aromaticity",
            "allowed_orders",
            "invalidated_bond_stereo",
            "required_repair",
        ):
            value = row.get(key)
            if value not in (None, "", [], {}):
                compact[key] = copy.deepcopy(value)
        positions[identity] = len(merged)
        merged.append(compact)

    for value in existing:
        if not isinstance(value, Mapping):
            continue
        nested = dict(value.get("replay_diagnostic") or {})
        add(nested or value)

    compact_diagnostic = _compact_builder_rejection(diagnostic)
    replay_diagnostic = dict(compact_diagnostic.get("replay_diagnostic") or {})
    add(replay_diagnostic)
    return merged


def _step_claims_strategy_key_event(
    step: Mapping[str, Any], strategy_card: Mapping[str, Any] | None
) -> bool:
    """Schedule only a replayed Builder claim against Strategy's checkpoint.

    Reaction names and shared words have no authority.  The compact Builder
    relation requests the audit, while a real mapped skeletal edit is the
    Host-owned prerequisite.  The Critic still decides whether the edit truly
    instantiates the checkpoint and whether the chemistry is acceptable.
    """

    checkpoint = str(dict(strategy_card or {}).get("critic_checkpoint") or "").strip()
    row = dict(step)
    if (
        not checkpoint
        or _normalize_checkpoint_relation(row.get("checkpoint_relation")) != "executes_checkpoint"
    ):
        return False
    # The step has already passed Host ReactionJSON replay at this point.  Do
    # not make the audit trigger depend on one encoding style: a valid
    # annulation or fragmentation may be expressed through group operations,
    # bond-order changes, or an explicit stereochemical edit rather than only
    # add_bond/break_bond.
    graph_edit_operations = {
        "break_bond",
        "add_bond",
        "change_bond_order",
        "add_group",
        "remove_group",
        "invert_stereocenter",
        "clear_stereocenter",
        "set_bond_stereo",
        "set_tetrahedral_stereo",
    }
    return any(
        str(operation.get("op") or "") in graph_edit_operations
        for operation in normalize_reaction_operations(row.get("reaction_operations") or ())
    )


def _key_event_fingerprint(step: Mapping[str, Any]) -> str:
    payload = {
        "mapped_product_smiles": str(step.get("mapped_product_smiles") or ""),
        "mapped_precursor_smiles": list(step.get("mapped_precursor_smiles") or []),
        "reaction_operations": [
            dict(value)
            for value in normalize_reaction_operations(step.get("reaction_operations") or ())
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


def _bind_key_event_focus_assessment(
    critique: Mapping[str, Any], focus_step_id: str
) -> dict[str, Any]:
    """Bind the one key-event assessment to the Host-owned focus identity.

    The compact provider wire does not author opaque route step identifiers;
    the Host already owns that identity when it dispatches the single-focus
    audit.  Bind only one otherwise-unidentified assessment.  A conflicting
    identity or multiple assessments is an invalid/ambiguous response and is
    deliberately left unchanged for the normal unavailable path.
    """

    bound = dict(critique)
    raw_assessments = list(critique.get("step_assessments") or [])
    assessments = [
        dict(value) if isinstance(value, Mapping) else value for value in raw_assessments
    ]
    bound["step_assessments"] = assessments
    focus_id = str(focus_step_id or "")
    if not focus_id:
        return bound
    rows = [value for value in assessments if isinstance(value, Mapping)]
    if any(str(row.get("step_id") or "") == focus_id for row in rows):
        return bound
    if len(rows) != 1 or len(assessments) != 1:
        return bound
    if str(rows[0].get("step_id") or ""):
        return bound
    rows[0]["step_id"] = focus_id
    return bound


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
    for operation in normalize_reaction_operations(step.get("reaction_operations") or ()):
        op = str(operation.get("op") or "")
        if op in {"break_bond", "add_bond", "change_bond_order"}:
            pair = f"maps {operation.get('map_a')}-{operation.get('map_b')}"
            if op == "add_bond":
                labels.append(f"add bond {pair} order {operation.get('order')}")
            elif op == "change_bond_order":
                labels.append(f"change bond {pair} by {operation.get('delta')}")
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
        elif op == "set_tetrahedral_stereo":
            labels.append(
                f"set map {operation.get('map_idx')} configuration {operation.get('configuration')}"
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
            _canonical_smiles(parent_row.get("product_smiles")) if parent_row is not None else ""
        )
        parent_mapped = (
            _canonical_atom_mapped_smiles(parent_row.get("mapped_product_smiles"))
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
        precursors = [_canonical_smiles(value) for value in row.get("precursor_smiles") or []]
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


def _compact_path_reaction_rows(
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Project replayed reaction facts without copying molecular structures."""

    return [
        {
            "step_id": str(row.get("step_id") or ""),
            "reaction_family": str(
                row.get("reaction_family") or row.get("transformation_hypothesis") or ""
            )[:160],
            "checkpoint_relation": _normalize_checkpoint_relation(row.get("checkpoint_relation")),
            "edit_summary": _compact_replayed_edit_summary(row),
        }
        for row in steps
        if isinstance(row, Mapping)
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
    mapped_precursors = [str(value) for value in parent.get("mapped_precursor_smiles") or []]
    canonical_precursors = [
        _canonical_smiles(value) for value in parent.get("precursor_smiles") or []
    ]
    canonical_multiplicity = Counter(canonical_precursors)
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
        mapped_identity = _canonical_atom_mapped_smiles(mapped)
        expanded = any(
            mapped_identity
            and _canonical_atom_mapped_smiles(row.get("mapped_product_smiles")) == mapped_identity
            for row in later_rows
        )
        if not expanded and canonical and canonical_multiplicity[canonical] == 1:
            # Canonical-SMILES fallback is safe only for a unique precursor.
            # Symmetric/convergent splits may contain two isomorphic molecules
            # with different Host map namespaces; collapsing them here lies to
            # the next Builder about which occurrence has already expanded.
            expanded = any(
                _canonical_smiles(row.get("product_smiles")) == canonical for row in later_rows
            )
        siblings.append(
            {
                "mapped_smiles": mapped,
                "path_status": (
                    "expanded_on_current_path" if expanded else "not_expanded_on_current_path"
                ),
            }
        )
    if not siblings:
        return {}
    return {
        "parent_step_id": str(parent.get("step_id") or ""),
        "parent_reaction": str(
            parent.get("reaction_family") or parent.get("transformation_hypothesis") or ""
        )[:200],
        "co_precursors": siblings,
    }


def _route_lineage_context(
    steps: Iterable[Mapping[str, Any]],
    *,
    selected_product: str,
    selected_product_mapped: str,
) -> _RouteLineageContext:
    """Build the one leaf-local projection shared by Strategy and Builder."""

    rows = tuple(dict(row) for row in steps if isinstance(row, Mapping))
    connected = tuple(
        _connected_path_step_rows(
            rows,
            selected_product,
            selected_product_mapped,
        )
    )
    return _RouteLineageContext(
        selected_product_smiles=_canonical_smiles(selected_product),
        selected_product_mapped=str(selected_product_mapped or ""),
        connected_steps=connected,
        reaction_spine=tuple(_compact_path_reaction_rows(connected)),
        ancestor_smiles=tuple(
            _connected_path_ancestor_smiles(
                rows,
                selected_product,
                selected_product_mapped,
            )
        ),
        current_split_context=_current_split_context(
            rows,
            selected_product=selected_product,
            selected_product_mapped=selected_product_mapped,
        ),
    )


def _strategy_horizon_context(
    *,
    campaign_target: str,
    selected_product: str,
    selected_product_mapped: str,
    branch_index: int,
    milestone_index: int,
    completed_strategy_cards: Iterable[Mapping[str, Any]],
    route_steps: Iterable[Mapping[str, Any]],
    phase: str,
) -> dict[str, Any]:
    """Return the canonical leaf-local context for both Strategy roles."""

    lineage = _route_lineage_context(
        route_steps,
        selected_product=selected_product,
        selected_product_mapped=selected_product_mapped,
    )
    context: dict[str, Any] = {
        "schema_version": "strategy_horizon_context.v1",
        "phase": str(phase),
        "campaign_target": campaign_target,
        "selected_upstream_leaf": selected_product,
        "selected_upstream_leaf_mapped": selected_product_mapped,
        "selected_upstream_leaf_profile": _structure_profile(selected_product),
        "selected_upstream_leaf_topology_profile": _target_topology_profile(selected_product),
        "branch_id": branch_index + 1,
        "milestone_index": max(2, int(milestone_index)),
        "completed_milestones": [
            {
                "strategy_query": str(card.get("strategy_query") or "")[:500],
                "critical_assumption": str(card.get("critical_assumption") or "")[:300],
                "critic_checkpoint": str(card.get("critic_checkpoint") or "")[:300],
            }
            for card in completed_strategy_cards
            if isinstance(card, Mapping)
        ],
        "connected_path_reactions": [dict(row) for row in lineage.reaction_spine],
    }
    stereo = _compact_mapped_stereo_context(selected_product_mapped)
    if stereo:
        context["selected_upstream_leaf_stereo"] = stereo
    if lineage.current_split_context:
        context["current_split_context"] = dict(lineage.current_split_context)
    return context


def _path_repair_editor_prompt(
    *,
    target: str,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
    critic_feedback: Mapping[str, Any],
    provisional_rejected_step_ids: Iterable[str] = (),
) -> str:
    """Give the Editor full route semantics but no graph-migration burden."""

    provisional_ids = [
        str(value).strip() for value in provisional_rejected_step_ids if str(value).strip()
    ]
    context = {
        "schema_version": "path_repair_editor_context.v2",
        "campaign_target": target,
        "strategy": {
            key: value
            for key, value in dict(strategy_card).items()
            if key
            in {
                "strategy_query",
                "strategy_signature",
                "critical_assumption",
                "critic_checkpoint",
            }
        },
        "route_json": _minimal_editor_prompt_route_rows(steps),
        "critic_annotations": dict(critic_feedback),
    }
    if provisional_ids:
        context["provisional_rejected_step_ids"] = provisional_ids
    prompt_rows = [
        "Act as the route-level chemistry Editor. Read the complete current RouteJSON and the Critic's concrete blockers. RouteJSON is target-rooted: earlier rows are target-side and later rows are farther upstream.",
        "repair_transaction_scope already joins blockers connected by Host topology or the Critic's explicit chemical dependency. Deferred blockers are independent under the current evidence. If a deferred blocker nevertheless shares an inseparable protecting-group, reactive-state, or sequence dependency that makes separate repair chemically impossible, list only that blocker in additional_coupled_blocker_step_ids; otherwise return an empty list.",
        "Return only rollback_start_step_id, rebuild_through_step_id, additional_coupled_blocker_step_ids, preserved_suffix_compatible, one concise chemical repair_goal, and at most five active_constraints. rollback_start_step_id is the earliest target-side row that must change; rebuild_through_step_id is the last upstream row that must be regenerated. Include every selected or additionally coupled blocker affected by reaction reordering. Do not write revised steps, ReactionJSON operations, atom maps, precursor structures, stock claims, alternatives, or an explanation.",
        "The Host removes the dependency-closed start-through region, preserves the target-side durable prefix and unrelated branches, and treats later dependent rows as an exact old suffix. Before setting preserved_suffix_compatible=true, verify that the repair goal and every active constraint can still produce each retained suffix product with the same functional-group, protection, and stereochemical state. If not, extend rebuild_through_step_id through the incompatible row; return false only when no valid exact boundary can be retained, so the Host can stop before spending Builder calls.",
        "Use repair_goal to state the structural or mechanistic correction the rebuilt local pathway must achieve. Do not prescribe database edits or restate the whole route. active_constraints should contain only route-level chemistry that cannot be inferred from the molecular frontier, such as a Strategy-defining construction or an essential sequence/compatibility requirement.",
        "critic_annotations.active_checkpoint_constraints, when present, are unresolved Host-derived findings from earlier checkpoint candidates on this exact Strategy and mapped lineage. They remain binding across this route-span transaction. Do not reinterpret or omit them; use the directive's active_constraints only for additional span-level requirements, because the Host will carry a compact checkpoint summary forward automatically.",
        "A repair directive is not a deletion request and grants no admission: ordinary Builder calls must add a Host-replayable local path, reconnect the preserved suffix when one exists, and then survive complete-route replay and re-Critic. The old route remains authoritative until that transaction commits.",
        "PathRepairEditorContext:",
        json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    if provisional_ids:
        prompt_rows.insert(
            1,
            "Rows listed in provisional_rejected_step_ids were Host-replayed only to expose the failed checkpoint and were never admitted. A local route-span rebuild may start at one of those rows when the accepted prefix can remain unchanged, or at an earlier accepted row when that prefix must change. In either case the Host keeps the old accepted route authoritative until the complete rebuilt span passes replay and re-Critic.",
        )
    return "\n".join(prompt_rows)


def _path_repair_checkpoint_constraint_summary(
    checkpoint_feedback: Mapping[str, Any] | None,
) -> str:
    """Compact existing Key-Critic memory into one repair constraint.

    The append-only history remains the authority.  One bounded derived string
    lets the existing ``path_repair.active_constraints`` channel preserve the
    latest distinct chemical facts without adding another state field or
    allowing the Editor to silently retire prior checkpoint findings.
    """

    rows = [
        dict(row)
        for row in dict(checkpoint_feedback or {}).get("active_constraints") or []
        if isinstance(row, Mapping)
    ]
    if not rows:
        return ""
    compact: list[str] = []
    for row in rows[-6:]:
        blocking_type = str(row.get("blocking_type") or "chemical").strip()
        reasons = [
            str(value).strip()[:260] for value in row.get("reasons") or [] if str(value).strip()
        ]
        suggested_revision = str(row.get("suggested_revision") or "").strip()[:420]
        detail = reasons[0] if reasons else "unresolved checkpoint contradiction"
        if suggested_revision:
            detail += f" Required correction: {suggested_revision}"
        rendered = f"{blocking_type}: {detail}"
        if rendered not in compact:
            compact.append(rendered)
    return "Preserve unresolved Key-event Critic findings across this repair: " + " | ".join(
        compact
    )


def _path_repair_checkpoint_feedback(
    path_repair: Mapping[str, Any] | None,
    *,
    strategy_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt one unresolved transaction to the existing Key-Critic channel."""

    repair = dict(path_repair or {})
    constraints = [
        str(value).strip()[:520]
        for value in repair.get("active_constraints") or []
        if str(value).strip()
    ][:5]
    if not constraints:
        return {}
    repair_goal = str(repair.get("repair_goal") or "").strip()[:420]
    return {
        "strategy_digest": _strategy_card_digest(strategy_card),
        "active_constraints": [
            {
                "obligation_id": f"path-repair:{index + 1}",
                "severity": "blocking",
                "checkpoint_match": False,
                "blocking_type": "route_span_repair",
                "reasons": [constraint],
                "suggested_revision": repair_goal,
            }
            for index, constraint in enumerate(constraints)
        ],
    }


def _merge_key_event_feedback(
    transaction_feedback: Mapping[str, Any] | None,
    lineage_feedback: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge durable repair constraints with live same-lineage Critic facts.

    A route-span transaction starts with the Critic findings that caused its
    rollback, but later replacement checkpoint candidates can expose a new
    concrete defect.  ``key_event_critic_history`` remains the sole authority;
    this function only builds the bounded next-request projection.  Keeping
    the transaction rows and the six latest distinct live rows prevents both
    amnesia and an ever-growing prompt after sibling retries.
    """

    transaction = dict(transaction_feedback or {})
    lineage = dict(lineage_feedback or {})
    transaction_rows = [
        dict(row) for row in transaction.get("active_constraints") or [] if isinstance(row, Mapping)
    ][:5]
    lineage_rows = [
        dict(row) for row in lineage.get("active_constraints") or [] if isinstance(row, Mapping)
    ]
    if not transaction_rows and not lineage_rows:
        return {}

    merged_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append_distinct(row: Mapping[str, Any]) -> None:
        candidate = dict(row)
        signature = json.dumps(
            {
                "blocking_type": str(candidate.get("blocking_type") or ""),
                "reasons": [str(value) for value in candidate.get("reasons") or []],
                "suggested_revision": str(candidate.get("suggested_revision") or ""),
                "source_focus_step_id": str(candidate.get("source_focus_step_id") or ""),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen:
            return
        seen.add(signature)
        merged_rows.append(candidate)

    for row in transaction_rows:
        append_distinct(row)
    for row in lineage_rows[-6:]:
        append_distinct(row)

    feedback: dict[str, Any] = {
        "strategy_digest": str(
            lineage.get("strategy_digest") or transaction.get("strategy_digest") or ""
        ),
        "active_constraints": merged_rows,
    }
    failure_basin = dict(lineage.get("failure_basin") or {})
    if failure_basin:
        feedback["failure_basin"] = failure_basin
    return feedback


def _compact_mapped_stereo_context(
    mapped_smiles: Any,
    *,
    inspection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only Host-observed CIP and bond-stereo facts for one molecule."""

    result = (
        dict(inspection)
        if inspection is not None
        else inspect_mapped_smiles(str(mapped_smiles or ""))
    )
    if result.get("ok") is not True:
        return {}
    return {
        key: result[key]
        for key in ("centers", "unassigned_center_maps", "stereo_bonds")
        if result.get(key)
    }


def _compact_mapped_ring_topology(
    mapped_smiles: Any,
    *,
    inspection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the one Host/RDKit ring-path projection used by LLM workers."""

    result = (
        dict(inspection)
        if inspection is not None
        else inspect_mapped_smiles(str(mapped_smiles or ""))
    )
    if result.get("ok") is not True:
        return {}
    ring_paths = [
        [int(map_idx) for map_idx in ring if int(map_idx) > 0] for ring in result.get("rings") or []
    ]
    ring_paths = [ring for ring in ring_paths if ring]
    return {
        "ring_sizes": sorted(len(ring) for ring in ring_paths),
        "ring_paths": ring_paths,
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
    strategy_anchor_fulfilled = _strategy_anchor_fulfilled_for_card(step_rows, strategy_card)
    accepted_path = (
        (
            _minimal_editor_prompt_route_rows(step_rows)
            if editor_route_mutations and (paper_matched or compact_editor_context)
            else _compact_route_rows(step_rows)
        )
        if complete_route_json or editor_route_mutations
        else [
            {
                "product_smiles": str(step.get("product_smiles") or ""),
                "precursor_smiles": list(step.get("precursor_smiles") or []),
                "reaction_family": str(step.get("transformation_hypothesis") or ""),
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
            "editor_may_return_dependency_closed_route_json": bool(editor_route_mutations),
            "editor_context_compacted": bool(
                editor_route_mutations and (paper_matched or compact_editor_context)
            ),
            "maximum_steps": 25,
        },
    }
    if paper_matched and not editor_route_mutations and not complete_route_json:
        selected_canonical = _canonical_smiles(selected_product)
        current_mcts_state_fingerprint = _aiz_policy_state_fingerprint(
            selected_leaf_mapped=str(selected_product_mapped or _mapped_smiles(selected_product)),
            route_steps=step_rows,
        )
        leaf_rejections = [
            _compact_builder_rejection(row)
            for row in prior_rejections
            if isinstance(row, Mapping)
            and _canonical_smiles(row.get("product_smiles")) == selected_canonical
            and (
                str(row.get("reason") or "")
                not in {
                    "candidate_did_not_advance_selected_mcts_path",
                    "candidate_repeats_same_mcts_state_edit",
                }
                or str(row.get("mcts_state_fingerprint") or "") == current_mcts_state_fingerprint
            )
        ]
        lineage_context = _route_lineage_context(
            step_rows,
            selected_product=selected_product,
            selected_product_mapped=selected_product_mapped,
        )
        pending_checkpoint_feedback = dict(
            host_failure_feedback.get("pending_checkpoint_feedback") or {}
        )
        last_rejection_for_this_leaf = leaf_rejections[-1] if leaf_rejections else {}
        # The key-event Critic owns checkpoint feedback.  The same chemical
        # rejection used to be copied into the generic leaf rejection as
        # well, making the next Builder call read the full diagnosis twice.
        # Keep last_rejection_for_this_leaf for Host replay/cycle failures.
        if pending_checkpoint_feedback and last_rejection_for_this_leaf.get("chemical_rejection"):
            last_rejection_for_this_leaf = {}
        # Match the paper's next-step policy boundary: current node, concise
        # steering query, the accepted reaction spine, and only this leaf's
        # latest causal failure.  Full route structures belong to the host,
        # Critic, and Editor, not the next-step policy call.
        path_repair_context = dict(host_failure_feedback.get("path_repair") or {})
        memory = {
            "schema_version": "sequential_route_builder_context.v1",
            "phase": "route_local_repair" if repair else "route_builder_node",
            "target_smiles": target,
            "strategy": {
                key: value
                for key, value in dict(strategy_card).items()
                if key
                in {
                    "strategy_query",
                    "critical_assumption",
                    "critic_checkpoint",
                }
            },
            "selected_leaf_mapped": str(
                selected_product_mapped or _mapped_smiles(selected_product)
            ),
        }
        stereo_inspection = inspect_mapped_smiles(memory["selected_leaf_mapped"])
        stereo_context = _compact_mapped_stereo_context(
            memory["selected_leaf_mapped"],
            inspection=stereo_inspection,
        )
        if stereo_context:
            memory["selected_leaf_stereo"] = stereo_context
        selected_leaf_topology = _compact_mapped_ring_topology(
            memory["selected_leaf_mapped"],
            inspection=stereo_inspection,
        )
        if selected_leaf_topology.get("ring_paths"):
            memory["selected_leaf_topology"] = selected_leaf_topology
        for key, value in (
            (
                "connected_path_reactions",
                [dict(row) for row in lineage_context.reaction_spine],
            ),
            ("ancestor_smiles", list(lineage_context.ancestor_smiles)),
            (
                "current_split_context",
                dict(lineage_context.current_split_context),
            ),
            ("last_rejection_for_this_leaf", last_rejection_for_this_leaf),
            ("pending_checkpoint_feedback", pending_checkpoint_feedback),
            ("path_repair", path_repair_context if repair else {}),
        ):
            if value:
                memory[key] = value
    if paper_matched and not editor_route_mutations and not complete_route_json:
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
        if memory.get("selected_leaf_stereo"):
            context_guidance.append(
                "selected_leaf_stereo is the Host's compact RDKit CIP/bond-stereo inspection, not evidence of reaction selectivity."
            )
            context_guidance.append(
                "A map in selected_leaf_stereo.unassigned_center_maps means the immutable Host product does not demand one R/S assignment there. Do not add a stereo operation merely to fill that product omission; configure a generated precursor only when its actual geometry or stereochemistry controls the proposed reaction."
            )
        if memory.get("selected_leaf_topology"):
            context_guidance.append(
                "selected_leaf_topology is the Host's compact RDKit ring-path inspection for the current product. Use the mapped ring paths to verify that a named skeletal construction or fragmentation matches the actual graph; it is not evidence of feasibility or selectivity."
            )
        if memory.get("pending_checkpoint_feedback"):
            context_guidance.append(
                "pending_checkpoint_feedback.active_constraints is the complete compact set of blocking Key-event Critic findings that require a corrected candidate at this Strategy and mapped leaf lineage. Preserve and repair every listed topology, handle, stereochemical, compatibility, or sequence-dependency constraint across preparatory moves; a newer finding does not replace an older one, and only a later selected Critic pass retires the set. Uncertain evidence debt is reviewed by the Critic and is never assigned to a later Builder as a request to rewrite an old edge."
            )
            if memory["pending_checkpoint_feedback"].get("failure_basin"):
                context_guidance.append(
                    "pending_checkpoint_feedback.failure_basin is a diagnostic summary of distinct rejected checkpoint candidates on this same Strategy and mapped leaf lineage. When it shows a recurrent failure across structurally different candidates, do not merely rename reagents or draw another cosmetic checkpoint variant: choose a preparatory move that materially changes the reactive topology, required handle, or control element, or a genuinely different executable graph construction. It does not itself reject the Strategy; the independent Critic decides whether the horizon must change."
                )
        if memory.get("path_repair"):
            context_guidance.append(
                "path_repair gives the Critic-derived local repair goal and any exact old suffix boundary that the Host can reattach. Do not reproduce the preserved suffix or invent atom maps solely to force a match."
            )
            if memory["path_repair"].get("reconnect_boundaries"):
                context_guidance.append(
                    "When path_repair.reconnect_boundaries is present, this is a bounded replacement transaction, not a new stock search: choose the shortest local sequence to an exact mapped boundary, and make every returned precursor strictly closer to at least one boundary. Do not continue upstream toward stock or rebuild the preserved suffix; the Host detects and attaches the boundary."
                )
            if memory["path_repair"].get("replay_failures"):
                context_guidance.append(
                    "path_repair.replay_failures is transaction-wide negative memory for Host-proven invalid operations. Do not repeat the same failed_operation with the same replay_error on another descendant leaf."
                )
            if memory["path_repair"].get("repair_reference_span"):
                context_guidance.append(
                    "path_repair.repair_reference_span is the compact Host-replayed mutable span removed by this transaction. It is reference, not accepted history or an instruction to copy a rejected edge. Reuse its exact mapped atoms, coherent endpoints, and sound operations when useful, but correct every active Critic constraint; prior_key_critic distinguishes previously passed anchors from rejected attempts."
                )
        checkpoint_instruction = (
            "During route-local repair, path_repair.repair_goal guides the replacement chemistry. checkpoint_relation retains its normal Strategy meaning and never signals repair completion. The Host replays the provisional route and the route Critic alone decides whether the repair goal is resolved."
            if repair
            else "Set checkpoint_relation=executes_checkpoint only when this candidate's ordered operations themselves realize strategy.critic_checkpoint. Set checkpoint_relation=preparatory for handle installation, unmasking, functional-group adjustment, or any other step that merely enables or mentions the checkpoint. This label is scheduling metadata, not proof or admission."
        )
        private_path_instruction = (
            "Privately plan the shortest chemically coherent local replacement from selected_leaf_mapped to an exact path_repair.reconnect_boundaries molecular occurrence. Return only the single best next ReactionJSON move; omit alternatives and the comparison process."
            if repair and path_repair_context.get("reconnect_boundaries")
            else "Privately work out a complete chemically coherent pathway from selected_leaf_mapped through the Strategy's named construction toward accessible precursors, and compare plausible disconnections in that route context. Return only the single best current ReactionJSON move for selected_leaf_mapped. The one-object output boundary does not limit route-level reasoning; omit alternatives and the comparison process."
        )
        return "\n".join(
            [
                (
                    "Act as the Route Builder's next-step expansion policy for one selected MCTS node under a route-local repair. Keep the accepted target-side path and Strategy, and directly address path_repair; return one ordinary executable reaction at a time."
                    if repair
                    else "Act as the Route Builder's next-step expansion policy for one selected MCTS node. strategy.strategy_query is the steering hypothesis and guides the whole pathway; strategy.critic_checkpoint names the one actual graph transformation reserved for the sparse key-event audit."
                ),
                private_path_instruction,
                "One candidate must represent one executable reaction. A concerted or genuinely inseparable cascade may contain several graph edits, and its operations must encode the complete connected bond-change pattern; but an independent protection/deprotection, activation, redox change, workup transformation, or second reagent stage is a separate reaction edge even when it could be performed without isolating the intermediate. Do not telescope independent events to satisfy Critic feedback, and do not split one mechanistic event into fictitious intermediates. Privately challenge the chosen move, then compile and mentally replay it against selected_leaf_mapped before answering; use only maps present there because the Host derives both endpoints.",
                "Check the Strategy against the actual net graph edit, not the reaction name. When the named construction consumes or creates specific reactive handles, those mapped atoms and bonds must participate in the defining operations. When stereochemical control is part of the named construction, the relevant stereochemistry or geometry must be represented or deliberately transformed in the replayable structures and operations. reaction_intent, catalysts, and conditions cannot substitute for missing topology or stereochemical information.",
                checkpoint_instruction,
                "ReactionJSON primitive syntax is exact: change_bond_order uses signed delta; change_atom changes formal_charge or isotope only; atom installation/removal uses add_group/remove_group. add_bond always creates a single bond and has no order field; to create a new double or triple bond, follow add_bond with change_bond_order delta 1 or 2. add_group fragment_smiles contains exactly one [*] attachment atom and encodes its attachment bond directly, for example [*]O, [*]=O, or [*]#N; do not output order. For set_bond_stereo provide only map_a, map_b, and E/Z/CIS/TRANS/NONE/ANY intent; the Host derives RDKit stereo reference neighbours. To assign a newly created or unspecified tetrahedral center, use set_tetrahedral_stereo with map_idx and configuration R/S; the Host verifies actual CIP.",
                "Conditions describe the forward reaction environment for the Host-replayed precursor set -> selected_leaf_mapped product. They must be chemically compatible with that forward transformation even though ReactionJSON operations are written retrosynthetically from product to precursors. Protection/deprotection, tether or reactive-handle installation/removal, activation, and every other covalent state change that defines a precursor must be encoded by ReactionJSON in its own executable step, never claimed only in conditions.",
                "Express the reaction family and purpose together as one concise reaction_intent sentence. Keep conditions concise and include any catalyst there; they are hypotheses, not proof or a Critic verdict.",
                "Check functional-group compatibility within the replayed precursor set, including incompatible protic/basic, organometallic, redox, or catalyst-sensitive handles. If compatibility requires a covalent change, make that change an explicit step rather than a condition note.",
                "Prefer a move that advances the steering hypothesis. Necessary enabling reactions may be performed one at a time when the current leaf lacks the required handles; once selected_leaf_mapped contains the needed reactive topology, prefer executing the named key construction instead of accumulating unrelated enabling or supporting transformations.",
                *context_guidance,
                "The Host/MCTS alone decides termination, budget exhaustion, stock and solved status. The Builder has no handoff, fail, stop, or solved action; always return the best available ReactionJSON expansion.",
                "Return only checkpoint_relation, reaction_intent, ordered reaction_operations, and concise conditions. Return no complete RouteJSON, route skeleton, evidence, source, enzyme, validation, stock claim, or long explanation.",
                "PaperMatchedRouteBuilderContext:",
                json.dumps(memory, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ]
        )
    if paper_matched and editor_route_mutations:
        feedback = dict(host_failure_feedback)
        raw_materialization_failure = dict(feedback.get("editor_materialization_failure") or {})
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
                "invalidated_bond_stereo",
                "required_repair",
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
                    materialization_failure.get("host_replayed_prefix_step_count") or 0
                ),
                "mapped_open_precursor_authority": (
                    materialization_failure.get("mapped_open_precursor_authority")
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
                "ReactionJSON primitive syntax is exact: change_bond_order uses signed delta; change_atom changes formal_charge or isotope only; atom installation/removal uses add_group/remove_group. add_bond always creates a single bond and has no order field; to create a new double or triple bond, follow add_bond with change_bond_order delta 1 or 2. add_group fragment_smiles contains exactly one [*] attachment atom and encodes its attachment bond directly, for example [*]O, [*]=O, or [*]#N; do not output order. For set_bond_stereo provide only map_a, map_b, and stereo intent; the Host derives RDKit reference neighbours. To assign a newly created or unspecified tetrahedral center, use set_tetrahedral_stereo with map_idx and configuration R/S; the Host verifies actual CIP.",
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
                "Return one JSON object containing exactly one local ReactionJSON expansion for the selected node."
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
            "Write the ordered candidate.reaction_operations first and set candidate.precursor_smiles=[]. The host applies ReactionJSON to selected_open_leaf_mapped and deterministically derives the canonical precursor structures."
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
            "ReactionJSON field contract: break_bond/add_bond use map_a and map_b, and add_bond creates a single bond; change_bond_order uses map_a, map_b, and numeric delta; change_atom uses map_idx plus exactly one of formal_charge or isotope; set_explicit_h uses map_idx, count, and no_implicit; add_group uses map_idx and fragment_smiles, whose [*] attachment bond carries the bond order; remove_group uses map_indices; set_bond_stereo uses map_a, map_b, and stereo intent only because the Host derives RDKit reference neighbours; set_tetrahedral_stereo uses map_idx and configuration R/S for a new or unspecified center, which the Host verifies by CIP. Do not output an order or stereo_atom_maps field.",
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
    elif paper_matched and task_type == "route_path_repair_directive":
        effective_task_type = "path_repair_editor"
    return WorkerTask(
        task_id=(
            f"{spec.agent_id}:branch:{branch_index + 1}:"
            f"{('editor' if task_type in {'route_chemistry_edit', 'route_path_repair_directive'} else 'node')}"
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
                if paper_matched
                and task_type
                in {
                    "route_chemistry_edit",
                    "route_path_repair_directive",
                }
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


def _critic_focus_step_topology(
    critic_steps: Iterable[Mapping[str, Any]], focus_step_id: str
) -> dict[str, Any]:
    """Project one Host-owned Critic focus without duplicating route context."""

    focus_step = next(
        (
            row
            for row in critic_steps
            if str(row.get("step_id") or row.get("review_slot") or "") == str(focus_step_id)
        ),
        None,
    )
    if focus_step is None:
        return {}
    product_topology = _compact_mapped_ring_topology(focus_step.get("mapped_product_smiles"))
    precursor_topologies = [
        {"precursor_index": index, **topology}
        for index, mapped_precursor in enumerate(focus_step.get("mapped_precursor_smiles") or [])
        if (topology := _compact_mapped_ring_topology(mapped_precursor))
    ]
    if not product_topology and not precursor_topologies:
        return {}
    identity_key = "review_slot" if focus_step.get("review_slot") else "step_id"
    return {
        identity_key: str(focus_step_id),
        "product": product_topology,
        "precursors": precursor_topologies,
    }


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
    checkpoint_feedback: Mapping[str, Any] | None = None,
    repair_completion: Mapping[str, Any] | None = None,
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
            if key
            in {
                "strategy_query",
                "critical_assumption",
                "critic_checkpoint",
            }
            and value not in (None, "", [], {})
        }
    source_steps = [dict(step) for step in steps if isinstance(step, Mapping)]
    route_review_bindings = (
        _route_critic_review_bindings(source_steps)
        if paper_matched and audit_kind == "final_route"
        else []
    )
    critic_steps = [
        (
            _paper_critic_step_row(
                step,
                compact_level=level,
                review_binding=(route_review_bindings[index] if route_review_bindings else None),
                include_step_id=audit_kind != "final_route",
            )
            if paper_matched
            else _critic_step_row(step, compact_level=level)
        )
        for index, step in enumerate(source_steps)
    ]
    route = {
        "schema_version": "blind_route_critic_input.v1",
        "phase": (
            (
                "key_event_selected_evidence_review"
                if audit_kind == "key_event_followup"
                else "key_event_candidate_audit"
            )
            if audit_kind in {"key_event", "key_event_followup"}
            else "independent_chemical_critic"
        ),
        "campaign_target": target,
        "branch_id": branch_index + 1,
        "strategy_card": critic_strategy,
        "steps": critic_steps,
    }
    if audit_kind in {"key_event", "key_event_followup"}:
        route["focus_step_id"] = str(focus_step_id)
        focus_topology = _critic_focus_step_topology(critic_steps, str(focus_step_id))
        if focus_topology:
            route["focus_step_topology"] = focus_topology
        active_constraints = [
            dict(row)
            for row in dict(checkpoint_feedback or {}).get("active_constraints") or []
            if isinstance(row, Mapping)
        ]
        if active_constraints:
            route["active_checkpoint_constraints"] = active_constraints
        failure_basin = dict(dict(checkpoint_feedback or {}).get("failure_basin") or {})
        if failure_basin:
            route["failure_basin"] = failure_basin
    repair_completion = dict(repair_completion or {})
    if (
        audit_kind == "final_route"
        and str(repair_completion.get("completion_mode") or "") == "strategy_checkpoint"
    ):
        repair_focus_id = str(repair_completion.get("required_checkpoint_step_id") or "")
        repair_binding = next(
            (binding for binding in route_review_bindings if binding["step_id"] == repair_focus_id),
            {},
        )
        repair_focus_slot = str(repair_binding.get("review_slot") or "")
        repair_focus: dict[str, Any] = {
            "review_slot": repair_focus_slot,
        }
        focus_topology = _critic_focus_step_topology(
            critic_steps,
            repair_focus_slot,
        )
        if focus_topology:
            repair_focus["topology"] = focus_topology
        active_constraints = [
            str(value) for value in repair_completion.get("active_constraints") or [] if str(value)
        ]
        if active_constraints:
            repair_focus["active_constraints"] = active_constraints
        route["repair_checkpoint_focus"] = repair_focus
    if not paper_matched:
        route["strategy_milestone_cards"] = [
            _critic_strategy_card(card, compact_level=level)
            for card in strategy_milestone_cards
            if isinstance(card, Mapping)
        ]
    if paper_matched:
        if audit_kind in {"key_event", "key_event_followup"}:
            opening = (
                "Act as the independent key-event Critic. The Host has now selected a new immediate upstream step after an earlier uncertain audit. Re-audit only the unchanged focus_step_id against strategy_card.critic_checkpoint and the newly available local sequence evidence."
                if audit_kind == "key_event_followup"
                else "Act as the independent key-event Critic. The Host has replayed one new Builder candidate marked executes_checkpoint; audit only focus_step_id against strategy_card.critic_checkpoint."
            )
            return "\n".join(
                [
                    opening,
                    "The preceding steps are immutable root-to-leaf context. Do not reject them, demand a complete route, require stock closure, or penalize a key event merely because later upstream synthesis is absent.",
                    "Before passing the focus step, inspect every chemically compatible reactive handle and site in its actual mapped substrate; compare plausible intramolecular pairings, ring sizes, and competing chemo- or regioselective outcomes. Use uncertain only when no concrete contradiction is established.",
                    "focus_step_topology is the Host's compact RDKit ring-path projection for the mapped focus product and precursors. Use it to count the actual rings retained, created, or removed by the proposed event; it is deterministic graph context, not feasibility or selectivity evidence.",
                    "active_checkpoint_constraints, when present, are unresolved findings from earlier checkpoint attempts on this same Strategy and mapped leaf lineage. Re-evaluate every one against the new Host-replayed candidate. Return pass only if all are resolved; a new defect does not erase an older unresolved constraint.",
                    "failure_basin, when present, is a diagnostic projection of distinct earlier rejected checkpoint candidates on this same Strategy and mapped leaf lineage. One failed implementation may still warrant repair_scope=focus_edge. But when structurally different candidates repeatedly expose the same mechanistic, reactive-handle, selectivity, or control contradiction in the critical assumption itself, do not mechanically request another focus edge: use the existing repair_scope=strategy_horizon. Make that decision from the chemical evidence, not from a fixed attempt count; a genuinely new locally repairable defect remains focus_edge or route_span as appropriate.",
                    "The focus step's mapped product and every preceding row are immutable in a same-parent retry. Set repair_scope=focus_edge only when one replacement reaction edge can correct the unadmitted focus edge while keeping that mapped product unchanged. Set repair_scope=route_span when the correction requires inserting, reordering, or rebuilding multiple adjacent reactions, or changing the focus mapped product or any preceding row; the Host and Editor will rebuild that local span transactionally. Set repair_scope=strategy_horizon only when the checkpoint or critical assumption itself must be replaced because no credible edge or local-span repair can preserve it. Do not abandon a Strategy for one failed implementation. Use repair_scope=none for pass/uncertain; missing evidence that can arise only from extending a chemically coherent precursor farther upstream is uncertain, not a rejected rewrite.",
                    "First decide checkpoint_match from the Host-derived mapped product, mapped precursors, and ordered graph edits. reaction_family and checkpoint_relation are scheduling claims, not evidence. checkpoint_match=true only when the actual edit instantiates critic_checkpoint and directly tests critical_assumption; exposing, preparing, unmasking, or executing a downstream event that leaves the critical assumption untested is false.",
                    "Forward-simulate the focus edge from its exact mapped precursors to product. Check mechanism, net structural/H/charge/redox plausibility, mapped-atom provenance, and whether the stated conditions supply every required hydrogen transfer, redox, or workup event.",
                    "Serialized product stereochemistry states the intended outcome; it is not evidence that the substrate, catalyst, or conditions select that outcome. Judge stereochemical control from the actual precursor geometry, directing elements, catalyst, and conditions.",
                    "Do not invent a hidden required stereoisomer at a center or bond that the immutable Host product and campaign target leave unspecified. Still judge any claimed stereochemical control from the actual chemistry, but the missing product assignment alone is never a focus_edge or route_span Builder obligation; use uncertain when it only limits what can be proved, or strategy_horizon when the Strategy's specific stereochemical claim itself must be replaced.",
                    "When critical_assumption itself is a stereochemical, chemo-, regio-, or site-selectivity claim, a focus step that supplies no credible substrate, catalyst, ligand, auxiliary, or mechanistic control for that intended outcome is not executable as written: reject it. More generally, if suggested_revision would change the focus step's operations, covalent precursor state, catalyst, ligand, conditions, or order, return reject so the same parent leaf can produce a corrected edge. Reserve uncertain for missing evidence that can be supplied upstream without modifying the focus edge.",
                    "Reject a focus edge that telescopes independent reactions into one graph program. A concerted or genuinely inseparable cascade is one event; a separate protection/deprotection, activation, redox, workup transformation, or changed reagent stage with its own covalent change must be an adjacent edge. The smallest suggested revision should request that split rather than hiding both events in one operation.",
                    f"Return only checkpoint_match, verdict, blocking_type, repair_scope, at most two reasons, and one smallest suggested_revision. repair_scope must be one of {', '.join(KEY_EVENT_REPAIR_SCOPES)} and identifies the mutation owner independently of blocking_type. When checkpoint_match=false because the action is a benign mislabeled preparatory move that preserves the Strategy topology, use verdict=uncertain, blocking_type=none, repair_scope=none so the Host can retain it. When it substitutes for, consumes, or irreversibly cuts required topology, reject it and choose repair_scope from the actual mutable boundary. When checkpoint_match=true, pass means coherent execution, uncertain means plausible but unresolved, and reject requires a specific topology, handle, mechanism, atom-provenance, stereochemical, compatibility, or Strategy contradiction.",
                    "Missing literature, route incompleteness, later upstream synthesis, and stock metadata are never blockers. Return no route rewrite or long analysis.",
                    "KeyEventCriticInput:",
                    json.dumps(
                        route,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        strategy_audit = "Independently compare strategy_card.strategy_query, critical_assumption, and critic_checkpoint with the actual reaction families, structures, and bond edits. No Builder checkpoint_relation, role label, or host anchor claim is evidence. strategy_adherence=true only when at least one supplied step itself executes critic_checkpoint; a step that merely exposes or prepares a later event does not satisfy it."
        if route.get("repair_checkpoint_focus"):
            strategy_audit += " repair_checkpoint_focus binds the one Host-replayed step whose pending transaction must restore that checkpoint. Audit its mapped graph and compact ring topology against every active constraint; for this re-Critic, strategy_adherence=true only when this exact step executes critic_checkpoint. Return exactly one step_assessment with this review_slot."
        return "\n".join(
            [
                "Act as an independent Route Critic. Forward-simulate every supplied reaction from the current frontier toward the target while leaving the target-rooted RouteJSON storage order unchanged.",
                "Use the exact host-derived mapped products, mapped precursors, ReactionJSON operations, and proposed conditions. Check mechanism, net structural/H/charge/redox plausibility, reactive handles, functional-group and stereochemical compatibility, selectivity, and sequence dependencies.",
                strategy_audit,
                "Atom maps are host graph-replay identities and preserve mapped element identity. Routine reagents and coproducts may be omitted, but atom installation/removal must be explicit through add_group/remove_group; change_atom may change formal charge or isotope only. Reject an element transmutation or unexplained mapped-atom provenance break, not the mere omission of non-route reagents.",
                "For each step: pass means executable as written; uncertain means plausible but unresolved; reject and blocking=true require a specific chemical contradiction in the serialized route. Missing literature and stock metadata are not blockers; merely underspecified conditions are not blockers; Strategy non-adherence alone is not a blocker.",
                "Serialized product stereochemistry states the intended outcome; it is not evidence that the substrate, catalyst, or conditions select that outcome. Judge stereochemical control from the serialized precursor state and reaction environment.",
                "Do not invent a hidden required stereoisomer at a center or bond that the immutable Host product and campaign target leave unspecified. Still judge claimed selectivity from the chemistry, but product omission alone is not a chemical blocker or an Editor repair obligation.",
                "Each input row has a Host-issued review_slot. Return every review_slot exactly once; do not invent step IDs or machine digests. The Host restores the canonical step identity and authoritative reaction-edit digest after schema validation. Keep each assessment concise: at most two concrete reasons, a short condition assessment, and the smallest structure-local suggested revision. Do not output a long mechanistic analysis or repeat the route description.",
                "Write route_overall_evaluation as one concise 2-4 sentence whole-route judgment: state the route's strategic coherence, strongest feature, decisive unresolved risk or blocker, and experimental maturity. Synthesize rather than enumerate the step assessments, and do not use a table.",
                "coupled_blocker_groups is a compact route-level list of review-slot groups. Include a group only when two or more rejected steps require one coordinated replacement because they share an inseparable reactive-state, protecting-group, stereochemical, or sequence dependency; otherwise return an empty list. It groups repair scope only and does not admit chemistry.",
                "If the complete supplied route never performs the Strategy's named key construction, set strategy_adherence=false as observation metadata only. Assess every serialized step on its own chemistry; do not reject a chemically coherent route, create a blocking step, or invoke Editor merely to force the steering Strategy into an opportunistic route such as a stock-closed short path. A falsely named reaction may still be rejected when its actual graph edit or chemistry is contradictory. overall_assessment reports chemical route validity only: reject if any step has a concrete chemical blocker, uncertain if none block and at least one is uncertain, otherwise viable.",
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
            "Do not invent a hidden required stereoisomer at a center or bond that the immutable Host product and campaign target leave unspecified. Still judge claimed selectivity from the chemistry, but product omission alone is not a chemical blocker or a repair obligation.",
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
    checkpoint_feedback: Mapping[str, Any] | None = None,
    repair_completion: Mapping[str, Any] | None = None,
) -> str | None:
    """Build a Critic prompt without dropping route topology.

    The former fallback kept only the last eight steps and could still exceed
    the byte cap when a biological contract was verbose.  Worse, the late
    ``ValueError`` discarded every durable branch.  Progressive compaction now
    retains every step's exact product, precursors, edit program, execution
    domain, and strategic anchor while removing prose and condition detail.
    """

    route_steps = [dict(step) for step in steps if isinstance(step, Mapping)]
    milestone_cards = [dict(card) for card in strategy_milestone_cards if isinstance(card, Mapping)]
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
            checkpoint_feedback=checkpoint_feedback,
            repair_completion=repair_completion,
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
            key: intent.get(key) for key in keep if intent.get(key) not in (None, "", [], {})
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
        "mapped_precursor_smiles": list(step.get("mapped_precursor_smiles") or []),
        "reaction_operations": [
            dict(value)
            for value in step.get("reaction_operations") or []
            if isinstance(value, Mapping)
        ],
        "execution_domain": str(step.get("execution_domain") or "chemical"),
        "strategy_anchor": step.get("strategy_anchor") is True,
        "strategy_milestone_index": int(step.get("strategy_milestone_index") or 1),
        "strategy_id": str(step.get("strategy_id") or ""),
    }
    if compact_level <= 1:
        row["transformation_hypothesis"] = str(step.get("transformation_hypothesis") or "")[:768]
    predictions = [
        dict(value)
        for value in step.get("condition_predictions") or []
        if isinstance(value, Mapping)
    ]
    if compact_level == 0:
        row["condition_predictions"] = predictions
        row["biocatalytic_step"] = dict(step.get("biocatalytic_step") or {})
        row["biocatalytic_design_deficits"] = list(step.get("biocatalytic_design_deficits") or [])
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
                key: bio.get(key) for key in allowed_bio if bio.get(key) not in (None, "", [], {})
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
    review_binding: Mapping[str, Any] | None = None,
    include_step_id: bool = True,
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
                for reagent in prediction.get("reagents") or prediction.get("conditions") or []
            ]
        )
        if str(value)
    ][: (4 if compact_level == 0 else 2)]
    row = {
        "mapped_product_smiles": str(step.get("mapped_product_smiles") or ""),
        "mapped_precursor_smiles": list(step.get("mapped_precursor_smiles") or []),
        "reaction_operations": [
            dict(value)
            for value in step.get("reaction_operations") or []
            if isinstance(value, Mapping)
        ],
        "reaction_family": str(
            step.get("reaction_family") or step.get("transformation_hypothesis") or ""
        )[:160],
    }
    if include_step_id:
        row["step_id"] = str(step.get("step_id") or "")
    else:
        binding = dict(review_binding or {})
        row["review_slot"] = str(binding.get("review_slot") or "")
    # checkpoint_relation is a Builder scheduling claim, not chemical
    # evidence.  The Host already selects focus_step_id for the sparse audit;
    # hiding the label prevents it from biasing either Critic.  Optional
    # condition fields are serialized only when they carry information.
    if conditions:
        row["conditions"] = conditions
    catalyst = str(
        step.get("catalyst")
        or next(
            (prediction.get("catalyst") or "" for prediction in predictions),
            "",
        )
    )[:160]
    if catalyst:
        row["catalyst"] = catalyst
    return row


def _route_critic_review_bindings(
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Bind Critic rows to Host identities without asking the model to copy IDs."""

    bindings: list[dict[str, str]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            continue
        bindings.append(
            {
                "review_slot": f"review-{index:03d}",
                "reaction_edit_digest": reaction_edit_digest(step.get("reaction_operations") or ()),
                "step_id": str(step.get("step_id") or ""),
            }
        )
    return bindings


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
    task_id_override: str = "",
    route_steps: Iterable[Mapping[str, Any]] = (),
) -> WorkerTask:
    return WorkerTask(
        task_id=(
            str(task_id_override)
            if str(task_id_override)
            else (
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
            )
        ),
        case_id=_opaque_strategy_case_id(spec.run_id + ":critic"),
        task_type=(
            "paper_matched_key_event_critic"
            if paper_matched and audit_kind in {"key_event", "key_event_followup"}
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
            "route_review_bindings": _route_critic_review_bindings(route_steps),
        },
    )


def _preflight_paper_matched_worker_schemas(
    spec: AgentSpec,
    *,
    target: str,
    config: DirectorConfig,
) -> None:
    """Validate only schemas reachable in the selected Editor/review mode.

    The representative tasks are built by the same constructors used during
    execution.  The preflight then compiles schemas through the Worker's
    canonical schema function, so this does not create a second schema
    contract.
    """

    model = str(spec.metadata.get("model") or "")
    default_effort = str(spec.metadata.get("reasoning_effort") or "medium")
    tasks = [
        _strategy_portfolio_task(
            spec,
            prompt="provider schema preflight",
            model=model,
            reasoning_effort=str(spec.metadata.get("strategy_reasoning_effort") or default_effort),
            timeout_s=1.0,
            target_smiles=target,
        ),
        _strategy_task(
            spec,
            prompt="provider schema preflight",
            branch_index=0,
            attempt_index=0,
            model=model,
            reasoning_effort=str(spec.metadata.get("strategy_reasoning_effort") or default_effort),
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
    ]
    if config.enable_strategy_portfolio_critic:
        tasks.append(
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
            )
        )
    if config.enable_key_event_critic:
        tasks.append(
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
            )
        )
    if config.allow_editor_route_mutations:
        tasks.append(
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
            )
        )
    if config.enable_transactional_path_repair:
        tasks.append(
            _node_task(
                spec,
                prompt="provider schema preflight",
                branch_index=0,
                node_index=0,
                model=model,
                reasoning_effort=default_effort,
                timeout_s=1.0,
                task_type="route_path_repair_directive",
                paper_matched=True,
                target_smiles=target,
                selected_product=target,
            )
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
    rejected_assessment_ids = {
        str(value.get("step_id") or "")
        for value in assessments
        if (str(value.get("verdict") or "") == "reject" or value.get("blocking") is True)
        and str(value.get("step_id") or "")
    }
    coupled_by_step_id: dict[str, set[str]] = {
        step_id: set() for step_id in rejected_assessment_ids
    }
    for raw_group in critique.get("coupled_blocker_groups") or ():
        if not isinstance(raw_group, (list, tuple)):
            continue
        group = {
            str(value).strip()
            for value in raw_group
            if str(value).strip() in rejected_assessment_ids
        }
        if len(group) < 2:
            continue
        for step_id in group:
            coupled_by_step_id[step_id].update(group - {step_id})
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
        coupled_step_ids = sorted(coupled_by_step_id.get(step_id) or ())
        if coupled_step_ids:
            assessment["coupled_step_ids"] = coupled_step_ids
        step["reasons"] = [str(value) for value in assessment.get("reasons") or [] if str(value)]
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
        operations = normalize_reaction_operations(step.get("reaction_operations") or ())
        if not operations:
            continue
        assessment = dict(step.get("critic_assessment") or {})
        signatures.append(
            {
                "step_id": str(step.get("step_id") or "")[:160],
                "blocking_type": str(assessment.get("blocking_type") or "unspecified")[:80],
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

    blocking_rows = [dict(value) for value in blocking_steps if isinstance(value, Mapping)]

    def clip(value: Any, limit: int) -> str:
        return str(value or "").strip()[: max(1, int(limit))]

    def clip_list(
        values: Iterable[Any],
        *,
        count: int,
        limit: int,
    ) -> list[str]:
        return [text for text in (clip(value, limit) for value in values) if text][
            : max(0, int(count))
        ]

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
            "repair_scope",
            "condition_assessment",
            "suggested_revision",
            "enzyme_assessment",
        ):
            if assessment.get(key) not in (None, "", [], {}):
                keep_assessment[key] = clip(assessment.get(key), 480)
        coupled_step_ids = [
            clip(value, 160)
            for value in assessment.get("coupled_step_ids") or ()
            if clip(value, 160)
        ][:6]
        if coupled_step_ids:
            keep_assessment["coupled_step_ids"] = coupled_step_ids
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


def _critique_from_record(
    record: WorkerRunRecord,
    *,
    route_steps: Iterable[Mapping[str, Any]] = (),
    required_step_ids: Iterable[str] = (),
) -> dict[str, Any]:
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
    bindings = _route_critic_review_bindings(route_steps)
    required_ids = {str(value) for value in required_step_ids if str(value)}
    if required_ids:
        bindings = [binding for binding in bindings if binding["step_id"] in required_ids]
        if {binding["step_id"] for binding in bindings} != required_ids:
            return {
                "schema_version": "chemical_strategy_critique.v1",
                "status": "unavailable",
                "reason": "critic_step_binding_invalid",
                "binding_reasons": ["critic_required_step_binding_missing"],
                "semantics": {
                    "critic_required_before_evidence": True,
                    "host_step_identity_binding_failed": True,
                },
            }
    if bindings:
        expected_by_slot = {binding["review_slot"]: binding for binding in bindings}
        raw_assessments = payload.get("step_assessments")
        supplied = (
            [dict(row) for row in raw_assessments if isinstance(row, Mapping)]
            if isinstance(raw_assessments, list)
            else []
        )
        supplied_slots = [str(row.get("review_slot") or "") for row in supplied]
        binding_reasons: list[str] = []
        if len(supplied) != len(bindings):
            binding_reasons.append("critic_review_slot_count_mismatch")
        if len(set(supplied_slots)) != len(supplied_slots):
            binding_reasons.append("critic_review_slot_duplicate")
        if set(supplied_slots) != set(expected_by_slot):
            binding_reasons.append("critic_review_slot_set_mismatch")
        if binding_reasons:
            return {
                "schema_version": "chemical_strategy_critique.v1",
                "status": "unavailable",
                "reason": "critic_step_binding_invalid",
                "binding_reasons": sorted(set(binding_reasons)),
                "semantics": {
                    "critic_required_before_evidence": True,
                    "host_step_identity_binding_failed": True,
                },
            }
        bound_assessments = []
        for row in supplied:
            binding = expected_by_slot[str(row.get("review_slot") or "")]
            bound_assessments.append(
                {
                    **row,
                    "step_id": binding["step_id"],
                    "reaction_edit_digest": binding["reaction_edit_digest"],
                }
            )
        payload["step_assessments"] = bound_assessments
        raw_groups = payload.get("coupled_blocker_groups")
        bound_groups: list[list[str]] = []
        if isinstance(raw_groups, list):
            for raw_group in raw_groups:
                if not isinstance(raw_group, list):
                    continue
                slots = [str(value) for value in raw_group if str(value)]
                if any(slot not in expected_by_slot for slot in slots):
                    return {
                        "schema_version": "chemical_strategy_critique.v1",
                        "status": "unavailable",
                        "reason": "critic_step_binding_invalid",
                        "binding_reasons": ["critic_coupled_blocker_review_slot_unknown"],
                        "semantics": {
                            "critic_required_before_evidence": True,
                            "host_step_identity_binding_failed": True,
                        },
                    }
                bound_groups.append([expected_by_slot[slot]["step_id"] for slot in slots])
        payload["coupled_blocker_review_slot_groups"] = (
            list(raw_groups) if isinstance(raw_groups, list) else []
        )
        payload["coupled_blocker_groups"] = bound_groups
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
            require_complete_route_json and len(route_rows) < max(1, int(minimum_route_depth))
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
            # Persist the Host-resolved edit program, including atom maps
            # assigned to atoms introduced by add_group.  Keeping the raw
            # model program here makes a later whole-route replay allocate a
            # different namespace and invalidates the next upstream step.
            operations = materialized.reaction_operations
        else:
            precursors = declared_precursors
            mapped_precursors = tuple(_mapped_smiles(value) for value in precursors)
        atom_provenance_deficit = _has_atom_provenance_deficit(product, precursors)
        unexplained_atom_provenance_deficit = atom_provenance_deficit and not (
            reactionjson_audit.get("external_atom_source_required") is True
            and reactionjson_audit.get("external_atom_source_grants_reaction_proof") is False
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
                raw_biocatalytic_step if isinstance(raw_biocatalytic_step, Mapping) else None
            ),
        )
        biocatalytic_step, _biocatalytic_reasons = normalize_biocatalytic_step(
            raw_biocatalytic_step if isinstance(raw_biocatalytic_step, Mapping) else None,
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
                reaction_family=str(
                    _route_field(step, row, "reaction_family") or "retrosynthetic transformation"
                ),
                rationale=str(
                    _route_field(step, row, "transformation_rationale")
                    or "model-proposed local disconnection"
                ),
                step_role=_normalize_step_role(_route_field(step, row, "step_role")),
                checkpoint_relation=_normalize_checkpoint_relation(
                    _route_field(step, row, "checkpoint_relation")
                ),
                mapped_product_smiles=(str(mapped_product_for_step or _mapped_smiles(product))),
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
        dict(value) for value in payload.get("candidates") or [] if isinstance(value, Mapping)
    ]
    if not raw_candidates:
        return [], [{"reason": "candidate_count_invalid", "candidate_count": 0}]

    accepted: list[_CompiledReactionJsonCandidate] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    compiler = compiler or RouteJSONCompiler()
    for candidate_index, candidate in enumerate(raw_candidates[: max(1, int(max_candidates))]):
        candidate_id = str(
            candidate.get("candidate_id") or f"{record.task_id}:candidate:{candidate_index + 1}"
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
                    "reaction_edit_digest": reaction_edit_digest(expansion.reaction_operations),
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
        dict(value) for value in payload.get("candidates") or [] if isinstance(value, Mapping)
    ]
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    repair_status = str(candidate.get("repair_status") or "").strip().lower()
    if repair_status and repair_status != "revised":
        return None
    if (
        candidate.get("no_solved_claim") is not True
        or candidate.get("not_parent_route_proof") is not True
    ):
        return None
    if contains_raw_reaction_payload(candidate):
        return None
    has_route = (
        (isinstance(candidate.get("route_json"), list) and bool(candidate.get("route_json")))
        or (isinstance(candidate.get("route_patch"), list) and bool(candidate.get("route_patch")))
        or (
            isinstance(candidate.get("replace_span"), Mapping)
            and bool(dict(candidate["replace_span"]).get("remove_step_ids"))
            and bool(dict(candidate["replace_span"]).get("revised_steps"))
        )
        or (
            isinstance(candidate.get("repair_directive"), Mapping)
            and bool(dict(candidate["repair_directive"]).get("rollback_start_step_id"))
            and bool(dict(candidate["repair_directive"]).get("rebuild_through_step_id"))
            and bool(dict(candidate["repair_directive"]).get("repair_goal"))
        )
    )
    return candidate if has_route else None


def _route_draft_has_solved_claim(value: Any) -> bool:
    """Detect solved/status claims in a RouteJSON draft before host replay."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"solved", "is_solved"} and item is True:
                return True
            if (
                key_text in {"verdict", "route_status", "status"}
                and str(item or "").strip().lower() == "solved"
            ):
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
    return (
        (isinstance(route, list) and bool(route))
        or (isinstance(patch, list) and bool(patch))
        or (
            isinstance(span, Mapping)
            and bool(dict(span).get("remove_step_ids"))
            and bool(dict(span).get("revised_steps"))
        )
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
            raw_biocatalytic_step if isinstance(raw_biocatalytic_step, Mapping) else None
        ),
    )
    biocatalytic_step, _biocatalytic_reasons = normalize_biocatalytic_step(
        raw_biocatalytic_step if isinstance(raw_biocatalytic_step, Mapping) else None,
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
        checkpoint_relation=_normalize_checkpoint_relation(row.get("checkpoint_relation")),
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
        limitations=tuple(str(value) for value in row.get("limitations") or [] if str(value)),
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
            "product_smiles": _canonical_smiles(route_rows[failed_index].get("product_smiles")),
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
    if not compiled or compiled[0].product_smiles != _canonical_smiles(expected_target_smiles):
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
        frontier = (
            [
                {
                    "product_smiles": target,
                    "mapped_product_smiles": str(mapped_target_smiles or ""),
                }
            ]
            if target and str(mapped_target_smiles or "")
            else []
        )
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
                normalize_reaction_operations(failed_row.get("reaction_operations") or ()),
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
            if failed_step_id and str(row.get("step_id") or "") == failed_step_id
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


def _path_repair_directive_from_record(
    record: WorkerRunRecord,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    candidate = _route_json_candidate(record)
    if candidate is None:
        return None, {"reason": "path_repair_directive_missing"}
    raw = candidate.get("repair_directive")
    if not isinstance(raw, Mapping):
        return None, {"reason": "path_repair_directive_missing"}
    directive = {
        "rollback_start_step_id": str(raw.get("rollback_start_step_id") or "").strip(),
        "rebuild_through_step_id": str(raw.get("rebuild_through_step_id") or "").strip(),
        "additional_coupled_blocker_step_ids": list(
            dict.fromkeys(
                str(value).strip()
                for value in raw.get("additional_coupled_blocker_step_ids") or []
                if str(value).strip()
            )
        ),
        "preserved_suffix_compatible": (raw.get("preserved_suffix_compatible") is True),
        "repair_goal": str(raw.get("repair_goal") or "").strip(),
        "active_constraints": [
            str(value).strip()
            for value in raw.get("active_constraints") or []
            if str(value).strip()
        ][:5],
    }
    if not directive["rollback_start_step_id"]:
        return None, {"reason": "path_repair_rollback_start_step_id_missing"}
    if not directive["rebuild_through_step_id"]:
        return None, {"reason": "path_repair_rebuild_through_step_id_missing"}
    if not directive["repair_goal"]:
        return None, {"reason": "path_repair_goal_missing"}
    return directive, {}


def _select_path_repair_blocker_scope(
    *,
    current_steps: Iterable[Mapping[str, Any]],
    mapped_target_smiles: str,
    blocking_steps: Iterable[Mapping[str, Any]],
) -> tuple[_PathRepairBlockerScope | None, dict[str, Any]]:
    """Select one chemically coherent blocker component for one transaction.

    Host dependency ancestry and the Critic's explicit ``coupled_step_ids``
    are the only grouping authorities.  The latter joins topological siblings
    only when their remedies share an inseparable functional-state or sequence
    dependency; prose keywords never grant repair scope.
    """

    rows = [dict(row) for row in current_steps if isinstance(row, Mapping)]
    if not rows:
        return None, {"reason": "path_repair_current_route_empty"}
    compiler = RouteJSONCompiler()
    try:
        state = compiler.compile_route_graph_state(
            mapped_target_smiles=str(mapped_target_smiles or ""),
            steps=rows,
            minimum_depth=1,
        )
    except ReactionJsonReplayError as exc:
        return None, {
            "reason": "path_repair_current_route_not_replayable",
            "compiler_error": str(exc),
        }
    host_rows = compiler.assemble_route(state.reactions, metadata=rows)
    step_ids = [str(row.get("step_id") or "") for row in host_rows]
    if any(not value for value in step_ids) or len(set(step_ids)) != len(step_ids):
        return None, {"reason": "path_repair_current_step_ids_invalid"}
    requested_ids = list(
        dict.fromkeys(
            str(row.get("step_id") or "").strip()
            for row in blocking_steps
            if isinstance(row, Mapping) and str(row.get("step_id") or "").strip()
        )
    )
    if not requested_ids:
        return None, {"reason": "path_repair_blocking_step_ids_missing"}
    missing_ids = sorted(set(requested_ids) - set(step_ids))
    if missing_ids:
        return None, {
            "reason": "path_repair_blocking_step_not_found",
            "blocking_step_ids": missing_ids,
        }

    parent_indices = tuple(state.parent_step_indices)

    def descends_from(index: int, ancestor: int) -> bool:
        cursor: int | None = index
        while cursor is not None:
            if cursor == ancestor:
                return True
            cursor = parent_indices[cursor]
        return False

    blocker_indices = {step_ids.index(step_id) for step_id in requested_ids}
    adjacency = {index: set() for index in blocker_indices}
    for left in blocker_indices:
        for right in blocker_indices:
            if left >= right:
                continue
            if descends_from(left, right) or descends_from(right, left):
                adjacency[left].add(right)
                adjacency[right].add(left)
    requested_set = set(requested_ids)
    for raw in blocking_steps:
        if not isinstance(raw, Mapping):
            continue
        source_id = str(raw.get("step_id") or "").strip()
        if source_id not in requested_set:
            continue
        assessment = dict(raw.get("critic_assessment") or {})
        coupled_ids = {
            str(value).strip()
            for value in assessment.get("coupled_step_ids") or raw.get("coupled_step_ids") or ()
            if str(value).strip() in requested_set
        }
        source_index = step_ids.index(source_id)
        for coupled_id in coupled_ids:
            coupled_index = step_ids.index(coupled_id)
            if coupled_index == source_index:
                continue
            adjacency[source_index].add(coupled_index)
            adjacency[coupled_index].add(source_index)

    unassigned = set(blocker_indices)
    components: list[tuple[int, ...]] = []
    while unassigned:
        component: set[int] = set()
        frontier = [min(unassigned)]
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(adjacency[current] - component)
        unassigned -= component
        components.append(tuple(sorted(component)))
    components.sort(key=lambda value: value[0])
    component_step_ids = tuple(
        tuple(step_ids[index] for index in component) for component in components
    )
    selected_step_ids = component_step_ids[0]
    selected_set = set(selected_step_ids)
    deferred_step_ids = tuple(step_id for step_id in requested_ids if step_id not in selected_set)
    return (
        _PathRepairBlockerScope(
            selected_step_ids=selected_step_ids,
            deferred_step_ids=deferred_step_ids,
            component_step_ids=component_step_ids,
        ),
        {},
    )


def _path_repair_component_recritic_result(
    pending: Mapping[str, Any],
    blocking_steps: Iterable[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Accept a rebuilt component only when all remaining blockers were deferred."""

    current_ids = {
        str(row.get("step_id") or "").strip() for row in blocking_steps if isinstance(row, Mapping)
    }
    deferred_ids = {
        str(value).strip()
        for value in pending.get("deferred_blocker_step_ids") or ()
        if str(value).strip()
    }
    unexpected_ids = sorted(current_ids - deferred_ids)
    diagnostic = {
        "selected_blocker_step_ids": [
            str(value) for value in pending.get("selected_blocker_step_ids") or () if str(value)
        ],
        "deferred_blocker_step_ids": sorted(deferred_ids),
        "current_blocker_step_ids": sorted(current_ids),
        "unexpected_blocker_step_ids": unexpected_ids,
    }
    return not unexpected_ids, diagnostic


def _path_repair_boundary_preflight(
    span: _PathRepairSpan,
    *,
    preserved_suffix_compatible: bool,
) -> dict[str, Any]:
    """Stop a self-contradictory repair before any Builder call.

    Exact suffix replay remains the structural admission authority.  This
    preflight only enforces the Editor's own same-call boundary decision: a
    transaction may not retain a dependent suffix after declaring that its
    required molecular state is incompatible with the repair goal.
    """

    if not span.preserved_suffix_steps or preserved_suffix_compatible:
        return {}
    return {
        "reason": "path_repair_preserved_suffix_declared_incompatible",
        "reconnect_boundaries": [
            {
                "step_id": str(row.get("step_id") or ""),
                "product_smiles": str(row.get("product_smiles") or ""),
            }
            for row in span.reconnect_boundaries
        ],
        "builder_calls_avoided": True,
    }


def _prepare_path_repair_span(
    *,
    current_steps: Iterable[Mapping[str, Any]],
    mapped_target_smiles: str,
    directive: Mapping[str, Any],
    blocking_step_ids: Iterable[str],
    deferred_blocking_step_ids: Iterable[str] = (),
) -> tuple[_PathRepairSpan | None, dict[str, Any]]:
    """Compute one Host-owned, target-rooted repair span.

    The Editor names the inclusive ordered span whose chemistry must be
    revisited; it never chooses atom maps, open precursors, or rows to
    preserve.  Those facts come exclusively from a successful replay of the
    current target-rooted RouteJSON DAG.
    """

    rows = [dict(row) for row in current_steps if isinstance(row, Mapping)]
    if not rows:
        return None, {"reason": "path_repair_current_route_empty"}
    compiler = RouteJSONCompiler()
    try:
        state = compiler.compile_route_graph_state(
            mapped_target_smiles=str(mapped_target_smiles or ""),
            steps=rows,
            minimum_depth=1,
        )
    except ReactionJsonReplayError as exc:
        return None, {
            "reason": "path_repair_current_route_not_replayable",
            "compiler_error": str(exc),
        }
    host_rows = compiler.assemble_route(state.reactions, metadata=rows)
    step_ids = [str(row.get("step_id") or "") for row in host_rows]
    if any(not value for value in step_ids) or len(set(step_ids)) != len(step_ids):
        return None, {"reason": "path_repair_current_step_ids_invalid"}
    rollback_start_step_id = str(directive.get("rollback_start_step_id") or "").strip()
    rebuild_through_step_id = str(directive.get("rebuild_through_step_id") or "").strip()
    if rollback_start_step_id not in step_ids:
        return None, {
            "reason": "path_repair_rollback_start_step_not_found",
            "rollback_start_step_id": rollback_start_step_id,
        }
    if rebuild_through_step_id not in step_ids:
        return None, {
            "reason": "path_repair_rebuild_through_step_not_found",
            "rebuild_through_step_id": rebuild_through_step_id,
        }
    rollback_start_index = step_ids.index(rollback_start_step_id)
    rebuild_through_index = step_ids.index(rebuild_through_step_id)
    parent_indices = tuple(state.parent_step_indices)
    requested_blocker_ids = {str(item).strip() for item in blocking_step_ids if str(item).strip()}
    deferred_blocker_ids = {
        str(item).strip() for item in deferred_blocking_step_ids if str(item).strip()
    }
    if not requested_blocker_ids:
        return None, {"reason": "path_repair_blocking_step_ids_missing"}
    if requested_blocker_ids & deferred_blocker_ids:
        return None, {
            "reason": "path_repair_blocker_scope_overlap",
            "blocking_step_ids": sorted(requested_blocker_ids & deferred_blocker_ids),
        }
    missing_blocker_ids = sorted((requested_blocker_ids | deferred_blocker_ids) - set(step_ids))
    if missing_blocker_ids:
        return None, {
            "reason": "path_repair_blocking_step_not_found",
            "blocking_step_ids": missing_blocker_ids,
        }
    blocker_indices = {step_ids.index(value) for value in requested_blocker_ids}

    def descends_from(index: int, ancestor: int) -> bool:
        cursor: int | None = index
        while cursor is not None:
            if cursor == ancestor:
                return True
            cursor = parent_indices[cursor]
        return False

    if rollback_start_index > rebuild_through_index or not descends_from(
        rebuild_through_index, rollback_start_index
    ):
        return None, {
            "reason": "path_repair_span_direction_or_dependency_invalid",
            "rollback_start_step_id": rollback_start_step_id,
            "rebuild_through_step_id": rebuild_through_step_id,
        }
    # RouteJSON rows are topologically ordered, but independent sibling
    # branches may be interleaved in that order.  Array slicing would either
    # delete the sibling or reject an otherwise valid repair directive.  The
    # Host therefore owns the exact dependency closure: remove only rows that
    # descend from the declared start and have been reached by the declared
    # inclusive end boundary.
    removed_indices = {
        index
        for index in range(rebuild_through_index + 1)
        if descends_from(index, rollback_start_index)
    }
    uncovered_blockers = sorted(blocker_indices - removed_indices)
    if uncovered_blockers:
        return None, {
            "reason": "path_repair_span_does_not_cover_critic_blocker",
            "rollback_start_step_id": rollback_start_step_id,
            "rebuild_through_step_id": rebuild_through_step_id,
            "blocking_step_ids": [step_ids[index] for index in uncovered_blockers],
        }
    covered_deferred_blockers = sorted(
        step_ids.index(value)
        for value in deferred_blocker_ids
        if step_ids.index(value) in removed_indices
    )
    if covered_deferred_blockers:
        return None, {
            "reason": "path_repair_span_crosses_deferred_blocker_component",
            "rollback_start_step_id": rollback_start_step_id,
            "rebuild_through_step_id": rebuild_through_step_id,
            "blocking_step_ids": [step_ids[index] for index in covered_deferred_blockers],
        }
    preserved_suffix_indices = {
        index
        for index in range(len(host_rows))
        if index not in removed_indices
        and any(descends_from(index, removed_index) for removed_index in removed_indices)
    }
    durable_indices = set(range(len(host_rows))) - (removed_indices | preserved_suffix_indices)
    durable_rows = [dict(row) for index, row in enumerate(host_rows) if index in durable_indices]
    preserved_suffix_rows = [
        dict(row) for index, row in enumerate(host_rows) if index in preserved_suffix_indices
    ]
    reconnect_indices = [
        index
        for index in sorted(preserved_suffix_indices)
        if parent_indices[index] in removed_indices
    ]
    suffix_reconnect_boundaries = tuple(
        {
            "boundary_id": f"suffix-entry:{step_ids[index]}",
            "boundary_kind": "preserved_suffix_entry",
            "step_id": step_ids[index],
            "product_smiles": state.reactions[index].product_smiles,
            "mapped_product_smiles": state.reactions[index].mapped_product_smiles,
        }
        for index in reconnect_indices
    )
    removed_terminal_boundaries = tuple(
        {
            "boundary_id": (
                f"removed-terminal:{step_ids[producer_index]}:{occurrence_index}"
            ),
            "boundary_kind": "removed_terminal_open_precursor",
            "producer_step_id": step_ids[producer_index],
            "product_smiles": occurrence.product_smiles,
            "mapped_product_smiles": occurrence.mapped_product_smiles,
        }
        for occurrence_index, (occurrence, producer_index) in enumerate(
            zip(
                state.open_precursors,
                state.open_precursor_producer_step_indices,
                strict=True,
            )
        )
        if producer_index in removed_indices
    )
    reconnect_boundaries = (
        *suffix_reconnect_boundaries,
        *removed_terminal_boundaries,
    )
    final_open_boundaries = tuple(
        {
            "boundary_id": f"final-open:{step_ids[producer_index]}:{occurrence_index}",
            "boundary_kind": "final_open_precursor",
            "producer_step_id": step_ids[producer_index],
            "product_smiles": occurrence.product_smiles,
            "mapped_product_smiles": occurrence.mapped_product_smiles,
        }
        for occurrence_index, (occurrence, producer_index) in enumerate(
            zip(
                state.open_precursors,
                state.open_precursor_producer_step_indices,
                strict=True,
            )
        )
    )
    repair_frontier_reaction = state.reactions[rollback_start_index]
    if durable_rows:
        try:
            durable_state = compiler.compile_route_graph_state(
                mapped_target_smiles=str(mapped_target_smiles or ""),
                steps=durable_rows,
                minimum_depth=1,
            )
        except ReactionJsonReplayError as exc:
            return None, {
                "reason": "path_repair_durable_route_not_replayable",
                "compiler_error": str(exc),
            }
        durable_rows = compiler.assemble_route(
            durable_state.reactions,
            metadata=durable_rows,
        )
        open_leaf_states = tuple(
            {
                "smiles": row.product_smiles,
                "mapped_smiles": row.mapped_product_smiles,
            }
            for row in durable_state.open_precursors
        )
        durable_open_boundaries = [
            {
                "boundary_id": (
                    f"durable-open:{str(durable_rows[producer_index].get('step_id') or '')}:"
                    f"{occurrence_index}"
                ),
                "boundary_kind": "preserved_durable_open_precursor",
                "producer_step_id": str(
                    durable_rows[producer_index].get("step_id") or ""
                ),
                "product_smiles": occurrence.product_smiles,
                "mapped_product_smiles": occurrence.mapped_product_smiles,
            }
            for occurrence_index, (occurrence, producer_index) in enumerate(
                zip(
                    durable_state.open_precursors,
                    durable_state.open_precursor_producer_step_indices,
                    strict=True,
                )
            )
        ]
    else:
        open_leaf_states = (
            {
                "smiles": repair_frontier_reaction.product_smiles,
                "mapped_smiles": repair_frontier_reaction.mapped_product_smiles,
            },
        )
        durable_open_boundaries = [
            {
                "boundary_id": "repair-frontier:target",
                "boundary_kind": "repair_frontier",
                "producer_step_id": "",
                "product_smiles": repair_frontier_reaction.product_smiles,
                "mapped_product_smiles": (
                    repair_frontier_reaction.mapped_product_smiles
                ),
            }
        ]
    repair_frontier_mapped_identity = _canonical_mapped_smiles(
        repair_frontier_reaction.mapped_product_smiles
    )
    if not any(
        _canonical_mapped_smiles(row.get("mapped_smiles")) == repair_frontier_mapped_identity
        for row in open_leaf_states
    ):
        return None, {
            "reason": "path_repair_start_boundary_not_open",
            "rollback_start_step_id": rollback_start_step_id,
        }
    repair_frontier_position = next(
        (
            index
            for index, row in enumerate(durable_open_boundaries)
            if _canonical_mapped_smiles(row.get("mapped_product_smiles"))
            == repair_frontier_mapped_identity
        ),
        None,
    )
    if repair_frontier_position is None:
        return None, {
            "reason": "path_repair_start_boundary_not_open",
            "rollback_start_step_id": rollback_start_step_id,
        }
    preserved_open_boundaries = tuple(
        row
        for index, row in enumerate(durable_open_boundaries)
        if index != repair_frontier_position
    )
    completion_boundaries = (
        *preserved_open_boundaries,
        *reconnect_boundaries,
    )
    repair_goal = str(directive.get("repair_goal") or "").strip()
    if not repair_goal:
        return None, {"reason": "path_repair_goal_missing"}
    return (
        _PathRepairSpan(
            rollback_start_step_id=rollback_start_step_id,
            rebuild_through_step_id=rebuild_through_step_id,
            repair_goal=repair_goal,
            active_constraints=tuple(
                str(value).strip()
                for value in directive.get("active_constraints") or []
                if str(value).strip()
            )[:5],
            original_steps=tuple(dict(row) for row in host_rows),
            durable_steps=tuple(dict(row) for row in durable_rows),
            removed_steps=tuple(
                dict(row) for index, row in enumerate(host_rows) if index in removed_indices
            ),
            preserved_suffix_steps=tuple(dict(row) for row in preserved_suffix_rows),
            reconnect_boundaries=reconnect_boundaries,
            suffix_reconnect_boundaries=suffix_reconnect_boundaries,
            completion_boundaries=completion_boundaries,
            final_open_boundaries=final_open_boundaries,
            repair_frontier_product_smiles=(repair_frontier_reaction.product_smiles),
            repair_frontier_mapped_product_smiles=(repair_frontier_reaction.mapped_product_smiles),
            open_leaf_states=open_leaf_states,
            reserved_atom_maps=tuple(
                sorted(
                    _route_atom_map_namespace(
                        durable_rows,
                        str(mapped_target_smiles or ""),
                    )
                    | (
                        _route_atom_map_namespace(preserved_suffix_rows)
                        - _route_atom_map_namespace(
                            (),
                            *(
                                str(row.get("mapped_product_smiles") or "")
                                for row in suffix_reconnect_boundaries
                            ),
                        )
                    )
                )
            ),
        ),
        {},
    )


def _path_repair_boundary_leaf_indices(
    *,
    product_smiles: Iterable[str],
    mapped_product_smiles: Iterable[str],
    reconnect_boundaries: Iterable[Mapping[str, Any]],
) -> frozenset[int]:
    """Bind the largest stereo-aware multiset of boundary occurrences."""

    products = [_canonical_smiles(value) for value in product_smiles]
    mapped = [str(value or "").strip() for value in mapped_product_smiles]
    boundaries = [dict(row) for row in reconnect_boundaries if isinstance(row, Mapping)]
    candidates_by_boundary: dict[int, list[int]] = {}
    for boundary_index, boundary in enumerate(boundaries):
        expected_product = _canonical_smiles(boundary.get("product_smiles"))
        expected_mapped = str(boundary.get("mapped_product_smiles") or "").strip()
        candidates: list[tuple[int, int]] = []
        for index, product in enumerate(products):
            if (
                not product
                or product != expected_product
                or index >= len(mapped)
                or not mapped[index]
            ):
                continue
            exact = (
                _canonical_mapped_smiles(mapped[index])
                == _canonical_mapped_smiles(expected_mapped)
            )
            if exact:
                candidates.append((0, index))
                continue
            if (
                _deterministic_boundary_atom_map_translation(
                    expected_mapped,
                    mapped[index],
                )
                is not None
            ):
                candidates.append((1, index))
        candidates_by_boundary[boundary_index] = [
            index
            for _distance, index in sorted(candidates)
        ]

    matched_boundary_by_leaf: dict[int, int] = {}

    def assign(boundary_index: int, visited_leaves: set[int]) -> bool:
        for leaf_index in candidates_by_boundary.get(boundary_index, ()):
            if leaf_index in visited_leaves:
                continue
            visited_leaves.add(leaf_index)
            prior_boundary = matched_boundary_by_leaf.get(leaf_index)
            if prior_boundary is None or assign(prior_boundary, visited_leaves):
                matched_boundary_by_leaf[leaf_index] = boundary_index
                return True
        return False

    for boundary_index in sorted(
        candidates_by_boundary,
        key=lambda value: (len(candidates_by_boundary[value]), value),
    ):
        assign(boundary_index, set())
    return frozenset(matched_boundary_by_leaf)


def _path_repair_frontier_reaches_boundaries(
    *,
    product_smiles: Iterable[str],
    mapped_product_smiles: Iterable[str],
    reconnect_boundaries: Iterable[Mapping[str, Any]],
) -> bool:
    """Return true when the frontier is exactly the required occurrence multiset."""

    products = [_canonical_smiles(value) for value in product_smiles]
    mapped = [str(value or "").strip() for value in mapped_product_smiles]
    boundaries = [dict(row) for row in reconnect_boundaries if isinstance(row, Mapping)]
    if len(products) != len(mapped) or len(products) != len(boundaries):
        return False
    if any(not value for value in products) or any(not value for value in mapped):
        return False
    return len(
        _path_repair_boundary_leaf_indices(
            product_smiles=products,
            mapped_product_smiles=mapped,
            reconnect_boundaries=boundaries,
        )
    ) == len(boundaries)


def _counter_distance(left: Counter[Any], right: Counter[Any]) -> int:
    return sum(abs(int(left[key]) - int(right[key])) for key in left.keys() | right.keys())


def _bond_distance_signature(bond: Chem.Bond) -> tuple[int, bool, str]:
    return (
        int(round(10 * float(bond.GetBondTypeAsDouble()))),
        bool(bond.GetIsAromatic()),
        str(bond.GetStereo()),
    )


def _external_branch_signatures(
    molecule: Chem.Mol,
    *,
    anchor_index: int,
    shared_indices: frozenset[int],
) -> Counter[Any]:
    """Describe remappable substituent branches around one durable atom map."""

    signatures: Counter[Any] = Counter()
    anchor = molecule.GetAtomWithIdx(anchor_index)
    for neighbor in anchor.GetNeighbors():
        neighbor_index = int(neighbor.GetIdx())
        if neighbor_index in shared_indices:
            continue
        pending = [neighbor_index]
        component: set[int] = set()
        while pending:
            atom_index = pending.pop()
            if atom_index in component or atom_index in shared_indices:
                continue
            component.add(atom_index)
            atom = molecule.GetAtomWithIdx(atom_index)
            pending.extend(int(item.GetIdx()) for item in atom.GetNeighbors())
        element_counts = Counter(
            (
                int(molecule.GetAtomWithIdx(index).GetAtomicNum()),
                int(molecule.GetAtomWithIdx(index).GetFormalCharge()),
                bool(molecule.GetAtomWithIdx(index).GetIsAromatic()),
            )
            for index in component
        )
        internal_bonds = Counter(
            _bond_distance_signature(bond)[:2]
            for bond in molecule.GetBonds()
            if int(bond.GetBeginAtomIdx()) in component and int(bond.GetEndAtomIdx()) in component
        )
        attachment = molecule.GetBondBetweenAtoms(anchor_index, neighbor_index)
        if attachment is None:
            continue
        signatures[
            (
                _bond_distance_signature(attachment),
                int(neighbor.GetAtomicNum()),
                len(component),
                tuple(sorted(element_counts.items())),
                tuple(sorted(internal_bonds.items())),
            )
        ] += 1
    return signatures


def _mapped_boundary_distance(
    mapped_smiles: Any,
    boundary_mapped_smiles: Any,
) -> int | None:
    """Return a deterministic, provenance-aware suffix-boundary mismatch.

    Zero means stereo-aware molecular identity even when fresh atom-map numbers
    differ. Nonzero values compare durable mapped atoms and bonds, substituent
    components attached to that durable core, and global graph composition.
    This is a repair-transaction progress invariant, not a chemistry score.
    """

    actual_text = str(mapped_smiles or "").strip()
    boundary_text = str(boundary_mapped_smiles or "").strip()
    actual = Chem.MolFromSmiles(actual_text)
    boundary = Chem.MolFromSmiles(boundary_text)
    if actual is None or boundary is None:
        return None
    actual_maps = {
        int(atom.GetAtomMapNum()): int(atom.GetIdx())
        for atom in actual.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    boundary_maps = {
        int(atom.GetAtomMapNum()): int(atom.GetIdx())
        for atom in boundary.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    if (
        not actual_maps
        or not boundary_maps
        or len(actual_maps) != sum(int(atom.GetAtomMapNum()) > 0 for atom in actual.GetAtoms())
        or len(boundary_maps) != sum(int(atom.GetAtomMapNum()) > 0 for atom in boundary.GetAtoms())
    ):
        return None
    if (
        _deterministic_boundary_atom_map_translation(
            boundary_text,
            actual_text,
            use_chirality=True,
        )
        is not None
    ):
        return 0
    shared_maps = frozenset(actual_maps.keys() & boundary_maps.keys())
    if not shared_maps:
        return None

    Chem.AssignStereochemistry(actual, cleanIt=True, force=True)
    Chem.AssignStereochemistry(boundary, cleanIt=True, force=True)
    score = 0
    actual_shared_indices = frozenset(actual_maps[value] for value in shared_maps)
    boundary_shared_indices = frozenset(boundary_maps[value] for value in shared_maps)
    for map_number in shared_maps:
        actual_atom = actual.GetAtomWithIdx(actual_maps[map_number])
        boundary_atom = boundary.GetAtomWithIdx(boundary_maps[map_number])
        actual_atom_signature = (
            int(actual_atom.GetAtomicNum()),
            int(actual_atom.GetFormalCharge()),
            int(actual_atom.GetIsotope()),
            bool(actual_atom.GetIsAromatic()),
            int(actual_atom.GetTotalNumHs()),
            str(actual_atom.GetProp("_CIPCode")) if actual_atom.HasProp("_CIPCode") else "",
        )
        boundary_atom_signature = (
            int(boundary_atom.GetAtomicNum()),
            int(boundary_atom.GetFormalCharge()),
            int(boundary_atom.GetIsotope()),
            bool(boundary_atom.GetIsAromatic()),
            int(boundary_atom.GetTotalNumHs()),
            (str(boundary_atom.GetProp("_CIPCode")) if boundary_atom.HasProp("_CIPCode") else ""),
        )
        if actual_atom_signature != boundary_atom_signature:
            score += 20
        score += 8 * _counter_distance(
            _external_branch_signatures(
                actual,
                anchor_index=actual_maps[map_number],
                shared_indices=actual_shared_indices,
            ),
            _external_branch_signatures(
                boundary,
                anchor_index=boundary_maps[map_number],
                shared_indices=boundary_shared_indices,
            ),
        )

    for left in shared_maps:
        for right in shared_maps:
            if left >= right:
                continue
            actual_bond = actual.GetBondBetweenAtoms(actual_maps[left], actual_maps[right])
            boundary_bond = boundary.GetBondBetweenAtoms(boundary_maps[left], boundary_maps[right])
            actual_signature = (
                _bond_distance_signature(actual_bond) if actual_bond is not None else None
            )
            boundary_signature = (
                _bond_distance_signature(boundary_bond) if boundary_bond is not None else None
            )
            if actual_signature != boundary_signature:
                score += 16

    actual_elements = Counter(
        (int(atom.GetAtomicNum()), int(atom.GetFormalCharge())) for atom in actual.GetAtoms()
    )
    boundary_elements = Counter(
        (int(atom.GetAtomicNum()), int(atom.GetFormalCharge())) for atom in boundary.GetAtoms()
    )
    actual_bonds = Counter(_bond_distance_signature(bond)[:2] for bond in actual.GetBonds())
    boundary_bonds = Counter(_bond_distance_signature(bond)[:2] for bond in boundary.GetBonds())
    score += 4 * _counter_distance(actual_elements, boundary_elements)
    score += 2 * abs(int(actual.GetNumHeavyAtoms()) - int(boundary.GetNumHeavyAtoms()))
    score += 2 * _counter_distance(actual_bonds, boundary_bonds)
    score += 6 * abs(int(actual.GetRingInfo().NumRings()) - int(boundary.GetRingInfo().NumRings()))
    score += 6 * abs(len(Chem.GetMolFrags(actual)) - len(Chem.GetMolFrags(boundary)))

    connectivity_translation = _deterministic_boundary_atom_map_translation(
        boundary_text,
        actual_text,
        use_chirality=False,
    )
    if connectivity_translation is not None:
        score += 20 * len(
            _boundary_stereo_mismatch_atom_maps(
                boundary_text,
                actual_text,
                connectivity_translation,
            )
        )
        score += 20 * len(
            _boundary_stereo_mismatch_bond_maps(
                boundary_text,
                actual_text,
                connectivity_translation,
            )
        )
    return max(1, score)


def _path_repair_boundary_progress_failure(
    *,
    selected_leaf_mapped: str,
    mapped_precursor_smiles: Iterable[str],
    reconnect_boundaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Require one focus-bearing precursor to approach an exact suffix cut."""

    boundaries = [
        dict(row)
        for row in reconnect_boundaries
        if isinstance(row, Mapping) and str(row.get("mapped_product_smiles") or "").strip()
    ]
    if not boundaries:
        return None
    selected = Chem.MolFromSmiles(str(selected_leaf_mapped or "").strip())
    precursors = [str(value or "").strip() for value in mapped_precursor_smiles]
    if selected is None or not precursors:
        return {
            "reason": "path_repair_candidate_not_toward_reconnect_boundary",
            "detail": "mapped_boundary_distance_unavailable",
        }
    selected_maps = {
        int(atom.GetAtomMapNum()) for atom in selected.GetAtoms() if int(atom.GetAtomMapNum()) > 0
    }
    focus_precursors: list[tuple[int, str]] = []
    for index, value in enumerate(precursors):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        precursor_maps = {
            int(atom.GetAtomMapNum())
            for atom in molecule.GetAtoms()
            if int(atom.GetAtomMapNum()) > 0
        }
        if selected_maps & precursor_maps:
            focus_precursors.append((index, value))
    if not focus_precursors:
        return {
            "reason": "path_repair_candidate_not_toward_reconnect_boundary",
            "detail": "focus_bearing_mapped_precursor_missing",
        }

    best_observation: dict[str, Any] | None = None
    for boundary in boundaries:
        boundary_mapped = str(boundary.get("mapped_product_smiles") or "")
        selected_distance = _mapped_boundary_distance(
            selected_leaf_mapped,
            boundary_mapped,
        )
        for precursor_index, precursor in focus_precursors:
            candidate_distance = _mapped_boundary_distance(precursor, boundary_mapped)
            if candidate_distance is None:
                continue
            if candidate_distance == 0 and selected_distance != 0:
                return None
            if selected_distance is None:
                continue
            observation = {
                "boundary_step_id": str(boundary.get("step_id") or ""),
                "boundary_product_smiles": _canonical_smiles(boundary.get("product_smiles")),
                "precursor_index": precursor_index,
                "selected_boundary_distance": selected_distance,
                "candidate_boundary_distance": candidate_distance,
            }
            if candidate_distance < selected_distance:
                return None
            if best_observation is None or (
                candidate_distance - selected_distance,
                candidate_distance,
            ) < (
                int(best_observation["candidate_boundary_distance"])
                - int(best_observation["selected_boundary_distance"]),
                int(best_observation["candidate_boundary_distance"]),
            ):
                best_observation = observation
    return {
        "reason": "path_repair_candidate_not_toward_reconnect_boundary",
        **(best_observation or {"detail": "mapped_boundary_distance_unavailable"}),
    }


def _path_repair_completion_reached(
    added_steps: Iterable[Mapping[str, Any]],
    *,
    completion_mode: str,
    selected_critic_pass_step_ids: Iterable[str] | None = None,
) -> bool:
    """Decide when a no-suffix repair has rebuilt its declared invariant.

    This helper owns only the semantic boundary used by an online Key-Critic
    rewrite. Final-route repair completion is derived from the Host-replayed
    cut frontier and never from the presence or label of an arbitrary row.
    """

    rows = [dict(row) for row in added_steps if isinstance(row, Mapping)]
    if not rows:
        return False
    if completion_mode == "strategy_checkpoint":
        passed = (
            {str(value) for value in selected_critic_pass_step_ids if str(value)}
            if selected_critic_pass_step_ids is not None
            else None
        )
        return any(
            str(row.get("checkpoint_relation") or "") == "executes_checkpoint"
            and (passed is None or str(row.get("step_id") or "") in passed)
            for row in rows
        )
    return False


def _path_repair_recritic_completion_failure(
    pending_repair: Any,
    critique: Mapping[str, Any],
) -> str:
    """Return the unmet repair invariant reported by the existing re-Critic."""

    if not isinstance(pending_repair, Mapping):
        return ""
    if str(pending_repair.get("completion_mode") or "") != "strategy_checkpoint":
        return ""
    if critique.get("strategy_adherence") is not True:
        return "path_repair_recritic_strategy_checkpoint_missing"
    focus_step_id = str(pending_repair.get("required_checkpoint_step_id") or "")
    focus_assessments = [
        dict(row)
        for row in critique.get("step_assessments") or []
        if isinstance(row, Mapping) and str(row.get("step_id") or "") == focus_step_id
    ]
    if not focus_step_id or len(focus_assessments) != 1:
        return "path_repair_recritic_checkpoint_assessment_missing"
    return ""


def _deterministic_boundary_atom_map_translation(
    old_mapped_smiles: str,
    new_mapped_smiles: str,
    *,
    use_chirality: bool = True,
) -> dict[int, int] | None:
    """Choose one stable provenance-preserving boundary isomorphism.

    Atom-map numbers already present on both boundaries are durable Host
    identities and must remain fixed.  Anchoring those atoms before graph
    matching distinguishes real provenance from harmless automorphisms.  When
    several translations differ only by symmetry in phenyl, tert-butyl, silyl,
    or similar groups, choose the lexicographically smallest translation.  The
    stitched route still has to pass the full RouteJSON compiler, so this
    deterministic tie break cannot admit a connectivity or stereo mismatch.
    """

    old = Chem.MolFromSmiles(str(old_mapped_smiles or "").strip())
    new = Chem.MolFromSmiles(str(new_mapped_smiles or "").strip())
    if old is None or new is None or old.GetNumAtoms() != new.GetNumAtoms():
        return None
    old_maps = [int(atom.GetAtomMapNum()) for atom in old.GetAtoms()]
    new_maps = [int(atom.GetAtomMapNum()) for atom in new.GetAtoms()]
    if (
        any(value <= 0 for value in old_maps)
        or any(value <= 0 for value in new_maps)
        or len(old_maps) != len(set(old_maps))
        or len(new_maps) != len(set(new_maps))
    ):
        return None
    old_query = Chem.Mol(old)
    new_query = Chem.Mol(new)
    old_atoms_by_map = {int(atom.GetAtomMapNum()): atom for atom in old_query.GetAtoms()}
    new_atoms_by_map = {int(atom.GetAtomMapNum()): atom for atom in new_query.GetAtoms()}
    shared_maps = sorted(set(old_maps) & set(new_maps))
    used_isotopes = {
        int(atom.GetIsotope())
        for molecule in (old_query, new_query)
        for atom in molecule.GetAtoms()
        if int(atom.GetIsotope()) > 0
    }
    next_anchor_isotope = 65_535
    for map_number in shared_maps:
        old_atom = old_atoms_by_map[map_number]
        new_atom = new_atoms_by_map[map_number]
        if int(old_atom.GetIsotope()) != int(new_atom.GetIsotope()):
            return None
        while next_anchor_isotope in used_isotopes:
            next_anchor_isotope -= 1
        if next_anchor_isotope <= 0:
            return None
        old_atom.SetIsotope(next_anchor_isotope)
        new_atom.SetIsotope(next_anchor_isotope)
        used_isotopes.add(next_anchor_isotope)
        next_anchor_isotope -= 1
    for atom in old_query.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom in new_query.GetAtoms():
        atom.SetAtomMapNum(0)
    matches = old_query.GetSubstructMatches(
        new_query,
        uniquify=False,
        useChirality=bool(use_chirality),
        maxMatches=100_000,
    )
    translations = {
        tuple(
            sorted(
                (old_maps[old_index], new_maps[new_index])
                for new_index, old_index in enumerate(match)
            )
        )
        for match in matches
        if len(match) == old.GetNumAtoms()
    }
    if not translations:
        return None
    return dict(min(translations))


def _boundary_stereo_mismatch_atom_maps(
    old_mapped_smiles: str,
    new_mapped_smiles: str,
    translation: Mapping[int, int],
) -> tuple[int, ...]:
    """Report old-boundary atom maps whose assigned tetrahedral stereo differs."""

    old = Chem.MolFromSmiles(str(old_mapped_smiles or "").strip())
    new = Chem.MolFromSmiles(str(new_mapped_smiles or "").strip())
    if old is None or new is None:
        return ()
    Chem.AssignStereochemistry(old, cleanIt=True, force=True)
    Chem.AssignStereochemistry(new, cleanIt=True, force=True)
    old_centers = {
        int(old.GetAtomWithIdx(index).GetAtomMapNum()): str(label)
        for index, label in Chem.FindMolChiralCenters(
            old,
            includeUnassigned=True,
            includeCIP=True,
        )
        if int(old.GetAtomWithIdx(index).GetAtomMapNum()) > 0
    }
    new_centers = {
        int(new.GetAtomWithIdx(index).GetAtomMapNum()): str(label)
        for index, label in Chem.FindMolChiralCenters(
            new,
            includeUnassigned=True,
            includeCIP=True,
        )
        if int(new.GetAtomWithIdx(index).GetAtomMapNum()) > 0
    }
    return tuple(
        sorted(
            old_map
            for old_map, new_map in translation.items()
            if old_centers.get(old_map) != new_centers.get(new_map)
            and (old_map in old_centers or new_map in new_centers)
        )
    )


def _boundary_stereo_mismatch_bond_maps(
    old_mapped_smiles: str,
    new_mapped_smiles: str,
    translation: Mapping[int, int],
) -> tuple[tuple[int, int], ...]:
    """Report old-boundary mapped bonds whose E/Z assignment differs."""

    old = Chem.MolFromSmiles(str(old_mapped_smiles or "").strip())
    new = Chem.MolFromSmiles(str(new_mapped_smiles or "").strip())
    if old is None or new is None:
        return ()
    Chem.AssignStereochemistry(old, cleanIt=True, force=True)
    Chem.AssignStereochemistry(new, cleanIt=True, force=True)
    new_atoms = {
        int(atom.GetAtomMapNum()): int(atom.GetIdx())
        for atom in new.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    mismatches: list[tuple[int, int]] = []
    for bond in old.GetBonds():
        old_a = int(bond.GetBeginAtom().GetAtomMapNum())
        old_b = int(bond.GetEndAtom().GetAtomMapNum())
        new_a = int(translation.get(old_a, 0))
        new_b = int(translation.get(old_b, 0))
        if new_a not in new_atoms or new_b not in new_atoms:
            continue
        new_bond = new.GetBondBetweenAtoms(new_atoms[new_a], new_atoms[new_b])
        if new_bond is None:
            continue
        old_stereo = str(bond.GetStereo())
        new_stereo = str(new_bond.GetStereo())
        if old_stereo != new_stereo and (old_stereo != "STEREONONE" or new_stereo != "STEREONONE"):
            mismatches.append(tuple(sorted((old_a, old_b))))
    return tuple(sorted(set(mismatches)))


def _path_repair_boundary_stereo_conflict(
    *,
    mapped_precursor_smiles: Iterable[str],
    reconnect_boundaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Find a precursor that has the required graph but the wrong stereo.

    A later retrosynthetic expansion cannot change the product identity of the
    step that created this leaf.  Rejecting that step here lets AiZ revisit its
    parent and ask Builder for a corrected sibling instead of spending the
    remaining repair budget upstream of a suffix that can never reconnect.
    """

    boundaries = [dict(row) for row in reconnect_boundaries if isinstance(row, Mapping)]
    for precursor_index, raw_precursor in enumerate(mapped_precursor_smiles):
        precursor = str(raw_precursor or "").strip()
        if not precursor:
            continue
        connectivity_matches: list[tuple[dict[str, Any], dict[int, int]]] = []
        stereo_match = False
        for boundary in boundaries:
            expected = str(boundary.get("mapped_product_smiles") or "").strip()
            if not expected:
                continue
            connectivity = _deterministic_boundary_atom_map_translation(
                expected,
                precursor,
                use_chirality=False,
            )
            if connectivity is None:
                continue
            connectivity_matches.append((boundary, connectivity))
            if (
                _deterministic_boundary_atom_map_translation(
                    expected,
                    precursor,
                    use_chirality=True,
                )
                is not None
            ):
                stereo_match = True
                break
        if stereo_match or not connectivity_matches:
            continue
        boundary, connectivity = connectivity_matches[0]
        expected = str(boundary.get("mapped_product_smiles") or "").strip()
        return {
            "reason": "path_repair_reconnect_boundary_stereo_mismatch",
            "boundary_step_id": str(boundary.get("step_id") or ""),
            "boundary_product_smiles": _canonical_smiles(boundary.get("product_smiles")),
            "precursor_index": precursor_index,
            "actual_mapped_precursor_smiles": precursor,
            "stereo_mismatch_atom_maps": list(
                _boundary_stereo_mismatch_atom_maps(
                    expected,
                    precursor,
                    connectivity,
                )
            ),
            "stereo_mismatch_bond_maps": [
                list(pair)
                for pair in _boundary_stereo_mismatch_bond_maps(
                    expected,
                    precursor,
                    connectivity,
                )
            ],
        }
    return None


def _remap_mapped_smiles(
    value: Any,
    translation: Mapping[int, int],
) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        raise ValueError("path_repair_suffix_mapped_smiles_invalid")
    for atom in molecule.GetAtoms():
        old_map = int(atom.GetAtomMapNum())
        if old_map in translation:
            atom.SetAtomMapNum(int(translation[old_map]))
    mapped = [
        int(atom.GetAtomMapNum()) for atom in molecule.GetAtoms() if int(atom.GetAtomMapNum()) > 0
    ]
    if len(mapped) != len(set(mapped)):
        raise ValueError("path_repair_suffix_map_collision")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _remap_reaction_operation(
    operation: Mapping[str, Any],
    translation: Mapping[int, int],
) -> dict[str, Any]:
    row = dict(operation)
    for key in ("map_a", "map_b", "map_idx"):
        if key in row:
            old_map = int(row[key])
            row[key] = int(translation.get(old_map, old_map))
    for key in ("map_indices", "stereo_atom_maps"):
        if isinstance(row.get(key), list):
            row[key] = [int(translation.get(int(value), int(value))) for value in row[key]]
    if row.get("op") == "add_group" and row.get("fragment_smiles"):
        row["fragment_smiles"] = _remap_mapped_smiles(
            row["fragment_smiles"],
            translation,
        )
    return row


def _remap_path_repair_suffix_rows(
    rows: Iterable[Mapping[str, Any]],
    translation: Mapping[int, int],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for value in rows:
        row = dict(value)
        if row.get("mapped_product_smiles"):
            row["mapped_product_smiles"] = _remap_mapped_smiles(
                row["mapped_product_smiles"],
                translation,
            )
        if isinstance(row.get("mapped_precursor_smiles"), list):
            row["mapped_precursor_smiles"] = [
                _remap_mapped_smiles(item, translation) for item in row["mapped_precursor_smiles"]
            ]
        row["reaction_operations"] = [
            _remap_reaction_operation(operation, translation)
            for operation in row.get("reaction_operations") or []
            if isinstance(operation, Mapping)
        ]
        row.pop("reactionjson_audit", None)
        remapped.append(row)
    return remapped


def _stitch_path_repair_suffix(
    *,
    mapped_target_smiles: str,
    rebuilt_steps: Iterable[Mapping[str, Any]],
    preserved_suffix_steps: Iterable[Mapping[str, Any]],
    reconnect_boundaries: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Attach an untouched suffix across unambiguous molecular occurrences."""

    rebuilt = [dict(row) for row in rebuilt_steps if isinstance(row, Mapping)]
    suffix = [dict(row) for row in preserved_suffix_steps if isinstance(row, Mapping)]
    boundaries = [dict(row) for row in reconnect_boundaries if isinstance(row, Mapping)]
    if not suffix:
        return rebuilt, {"suffix_stitched": False, "boundary_count": 0}
    compiler = RouteJSONCompiler()
    try:
        # ``rebuilt`` is already a Host-materialized program. Reservations
        # guard only the admission of each new Builder edit; replaying the
        # resolved program against the same reservations would make its own
        # explicit add_group maps collide with themselves.
        rebuilt_state = compiler.compile_route_graph_state(
            mapped_target_smiles=str(mapped_target_smiles or ""),
            steps=rebuilt,
            minimum_depth=1,
        )
    except ReactionJsonReplayError as exc:
        return None, {
            "reason": "path_repair_rebuilt_prefix_not_replayable",
            "compiler_error": str(exc),
        }
    resolved_rebuilt = compiler.assemble_route(
        rebuilt_state.reactions,
        metadata=rebuilt,
    )
    available = list(rebuilt_state.open_precursors)
    translation: dict[int, int] = {}
    for boundary in boundaries:
        old_product = _canonical_smiles(boundary.get("product_smiles"))
        old_mapped = str(boundary.get("mapped_product_smiles") or "")
        candidates = [
            (index, occurrence)
            for index, occurrence in enumerate(available)
            if occurrence.product_smiles == old_product
        ]
        exact = [
            (index, occurrence)
            for index, occurrence in candidates
            if _canonical_mapped_smiles(occurrence.mapped_product_smiles)
            == _canonical_mapped_smiles(old_mapped)
        ]
        matched: tuple[int, Any, dict[int, int]] | None = None
        if len(exact) == 1:
            index, occurrence = exact[0]
            old_maps = _route_atom_map_namespace((), old_mapped)
            matched = (
                index,
                occurrence,
                {value: value for value in old_maps},
            )
        elif not exact:
            isomorphic = []
            for index, occurrence in candidates:
                mapping = _deterministic_boundary_atom_map_translation(
                    old_mapped,
                    occurrence.mapped_product_smiles,
                )
                if mapping is not None:
                    isomorphic.append((index, occurrence, mapping))
            if len(isomorphic) == 1:
                matched = isomorphic[0]
        stereo_conflicts: list[tuple[int, Any, dict[str, Any]]] = []
        if matched is None:
            for index, occurrence in enumerate(available):
                conflict = _path_repair_boundary_stereo_conflict(
                    mapped_precursor_smiles=(occurrence.mapped_product_smiles,),
                    reconnect_boundaries=(boundary,),
                )
                if conflict is not None:
                    stereo_conflicts.append((index, occurrence, conflict))
        molecular_occurrence_indices = {index for index, _occurrence in candidates} | {
            index for index, _occurrence, _conflict in stereo_conflicts
        }
        if matched is None and len(molecular_occurrence_indices) > 1:
            return None, {
                "reason": "path_repair_reconnect_boundary_ambiguous",
                "boundary_step_id": str(boundary.get("step_id") or ""),
                "boundary_product_smiles": old_product,
                "candidate_count": len(molecular_occurrence_indices),
            }
        if matched is None and len(stereo_conflicts) == 1:
            _index, _occurrence, conflict = stereo_conflicts[0]
            return None, {
                "reason": str(conflict.get("reason") or ""),
                "boundary_step_id": str(boundary.get("step_id") or ""),
                "boundary_product_smiles": old_product,
                "candidate_count": 1,
                "stereo_mismatch_atom_maps": list(conflict.get("stereo_mismatch_atom_maps") or []),
                "stereo_mismatch_bond_maps": list(conflict.get("stereo_mismatch_bond_maps") or []),
            }
        if matched is None and not candidates and not stereo_conflicts:
            return None, {
                "reason": "path_repair_reconnect_boundary_not_reached",
                "boundary_step_id": str(boundary.get("step_id") or ""),
                "boundary_product_smiles": old_product,
                "candidate_count": 0,
            }
        if matched is None:
            return None, {
                "reason": "path_repair_reconnect_boundary_ambiguous",
                "boundary_step_id": str(boundary.get("step_id") or ""),
                "boundary_product_smiles": old_product,
                "candidate_count": len(molecular_occurrence_indices),
            }
        available_index, _occurrence, mapping = matched
        for old_map, new_map in mapping.items():
            prior = translation.get(old_map)
            if prior is not None and prior != new_map:
                return None, {"reason": "path_repair_boundary_map_conflict"}
            if new_map in translation.values() and old_map not in translation:
                return None, {"reason": "path_repair_boundary_map_collision"}
            translation[old_map] = new_map
        available.pop(available_index)
    try:
        remapped_suffix = _remap_path_repair_suffix_rows(suffix, translation)
        # The repair prefix is now a Host-resolved program, so replay no
        # longer needs the reservation set that intentionally includes atoms
        # owned by the preserved suffix.  Passing that set through the suffix
        # itself would falsely classify its explicit atom maps as collisions.
        candidate_rows = [*resolved_rebuilt, *remapped_suffix]
        full_state = compiler.compile_route_graph_state(
            mapped_target_smiles=str(mapped_target_smiles or ""),
            steps=candidate_rows,
            minimum_depth=1,
        )
    except (ReactionJsonReplayError, ValueError) as exc:
        return None, {
            "reason": "path_repair_stitched_route_not_replayable",
            "compiler_error": str(exc),
        }
    return (
        compiler.assemble_route(full_state.reactions, metadata=candidate_rows),
        {
            "suffix_stitched": True,
            "boundary_count": len(boundaries),
            "preserved_suffix_step_count": len(suffix),
            "remapped_boundary_atom_count": sum(
                old_map != new_map for old_map, new_map in translation.items()
            ),
        },
    )


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

    rows = [_compact_route_spec(row) for row in current_steps if isinstance(row, Mapping)]
    if not rows:
        return None, "editor_replace_span_current_route_empty"

    current_ids = [str(row.get("step_id") or "") for row in rows]
    if any(not value for value in current_ids) or len(set(current_ids)) != len(current_ids):
        return None, "editor_replace_span_current_step_ids_invalid"

    remove_step_ids = [
        str(value) for value in replace_span.get("remove_step_ids") or [] if str(value)
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
    if any(not value for value in revised_ids) or len(set(revised_ids)) != len(revised_ids):
        return None, "editor_replace_span_revised_step_ids_invalid"

    removed = set(remove_step_ids)
    retained_ids = {value for value in current_ids if value not in removed}
    if retained_ids.intersection(revised_ids):
        return None, "editor_replace_span_step_id_collision"

    first_removed_index = min(current_ids.index(value) for value in remove_step_ids)
    before = [
        row
        for index, row in enumerate(rows)
        if index < first_removed_index and str(row.get("step_id") or "") not in removed
    ]
    after = [
        row
        for index, row in enumerate(rows)
        if index > first_removed_index and str(row.get("step_id") or "") not in removed
    ]
    merged = [*before, *revised, *after]
    return (
        (merged, "")
        if merged
        else (
            None,
            "editor_replace_span_route_empty",
        )
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
            if not replacement.get("reaction_operations") and base.get("reaction_operations"):
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
                (i for i, row in enumerate(rows) if str(row.get("step_id") or "") == after_step_id),
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
                (i for i, row in enumerate(rows) if str(row.get("step_id") or "") == step_id),
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
        dict(item) for item in row.get("condition_predictions") or [] if isinstance(item, Mapping)
    ]
    return {
        "step_id": str(row.get("step_id") or ""),
        "product_smiles": str(row.get("product_smiles") or ""),
        "mapped_product_smiles": str(row.get("mapped_product_smiles") or ""),
        "precursor_smiles": list(row.get("precursor_smiles") or []),
        "mapped_precursor_smiles": list(row.get("mapped_precursor_smiles") or []),
        "reaction_family": str(
            row.get("reaction_family") or row.get("transformation_hypothesis") or ""
        ),
        "step_role": _normalize_step_role(row.get("step_role")),
        "checkpoint_relation": _normalize_checkpoint_relation(row.get("checkpoint_relation")),
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
            row.get("catalyst") or next((item.get("catalyst") or "" for item in predictions), "")
        ),
        "enzyme": str(
            row.get("enzyme") or next((item.get("enzyme") or "" for item in predictions), "")
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
            return (
                None,
                {
                    "reason": reason,
                    "editor_mutation_mode": "replace_span",
                    "replace_span": dict(replace_span),
                },
                "replace_span",
            )
        expansions, diagnostic = _compile_editor_route_rows_with_diagnostic(
            merged,
            mapped_target_smiles=mapped_target_smiles,
            expected_target_smiles=expected_target_smiles,
        )
        if expansions is None:
            return (
                None,
                {
                    **diagnostic,
                    "editor_mutation_mode": "replace_span",
                    "replace_span": dict(replace_span),
                },
                "replace_span",
            )
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
            return (
                None,
                {
                    **diagnostic,
                    "editor_mutation_mode": "route_patch",
                    "route_patch": [
                        dict(value) for value in patch_rows if isinstance(value, Mapping)
                    ],
                },
                "route_patch",
            )
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
        return (
            None,
            {
                **diagnostic,
                "editor_mutation_mode": "full_route_json",
            },
            "full_route_json",
        )
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

    frozen_steps = [dict(row) for row in current_steps if isinstance(row, Mapping)]
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
    operations = normalize_reaction_operations(replacement.get("reaction_operations") or ())
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
    branch["open_leaf_states"] = deque(dict(row) for row in projection.open_leaf_states)
    branch["deferred_builder_leaf_states"] = deque(
        dict(row) for row in projection.deferred_builder_leaf_states
    )
    branch["blocked_materializations"] = list(
        dict.fromkeys(
            _canonical_smiles(row.get("smiles"))
            for row in projection.deferred_builder_leaf_states
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
    """Derive unresolved leaves from active and locally deferred states.

    ``open_leaf_states`` is the Route Builder work queue.  A leaf whose
    ReactionJSON failed the bounded materialization retries must leave that
    queue, but it is still an unresolved route leaf and must remain visible to
    the same Builder on the canonical frontier. Keeping those meanings
    separate avoids both an immediate identical retry and false closure.
    """

    states = [
        dict(row)
        for row in branch.get("open_leaf_states") or []
        if isinstance(row, Mapping) and _canonical_smiles(row.get("smiles"))
    ]
    deferred = [
        dict(row)
        for row in branch.get("deferred_builder_leaf_states") or []
        if isinstance(row, Mapping) and _canonical_smiles(row.get("smiles"))
    ]
    branch["open_leaf_states"] = deque(states)
    branch["deferred_builder_leaf_states"] = deque(deferred)
    branch["open_leaves"] = deque(
        dict.fromkeys(_canonical_smiles(row.get("smiles")) for row in [*states, *deferred])
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
    aizynthfinder_search = dict(branch.get("aizynthfinder_strategy_search") or {})
    if aizynthfinder_search:
        return (
            aizynthfinder_search.get("canonical_route_projection_complete") is True
            and aizynthfinder_search.get("canonical_leaf_closure_complete") is True
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
        str(row.get("step_id") or f"step:{index}") for index, row in enumerate(rows, start=1)
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
                    if child_index != index and child_product and child_product in precursors
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


def _path_repair_reference_rows(
    steps: Iterable[Mapping[str, Any]],
    *,
    key_event_critic_history: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Expose only the rolled-back mutable span to a repair Builder.

    ``connected_path_reactions`` remains the accepted-history authority.  These
    rows preserve the exact Host-replayed structures and graph programs that an
    Editor selected for revision, so a local repair does not have to rediscover
    valid atom provenance from an abstract directive.  Latest Key-Critic status
    is attached without copying its prose; active feedback carries the reasons.
    """

    source_rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    compact = _minimal_editor_prompt_route_rows(source_rows)
    latest_critic_by_step: dict[str, Mapping[str, Any]] = {}
    for raw in key_event_critic_history:
        if not isinstance(raw, Mapping):
            continue
        focus_step_id = str(raw.get("focus_step_id") or "")
        if focus_step_id:
            latest_critic_by_step[focus_step_id] = raw
    for source, row in zip(source_rows, compact):
        checkpoint_relation = str(source.get("checkpoint_relation") or "")
        if checkpoint_relation:
            row["checkpoint_relation"] = checkpoint_relation
        critic_row = latest_critic_by_step.get(str(row.get("step_id") or ""))
        if not isinstance(critic_row, Mapping):
            continue
        assessment = dict(critic_row.get("assessment") or {})
        critic_summary: dict[str, Any] = {
            "status": str(critic_row.get("status") or ""),
            "checkpoint_match": critic_row.get("checkpoint_match") is True,
        }
        verdict = str(assessment.get("verdict") or "")
        if verdict:
            critic_summary["verdict"] = verdict
        blocking_type = str(assessment.get("blocking_type") or "")
        if blocking_type:
            critic_summary["blocking_type"] = blocking_type
        row["prior_key_critic"] = critic_summary
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
            "mapped_product_smiles": str(row.get("mapped_product_smiles") or ""),
            "precursor_smiles": list(row.get("precursor_smiles") or []),
            "mapped_precursor_smiles": list(row.get("mapped_precursor_smiles") or []),
            "reaction_operations": [
                dict(operation)
                for operation in row.get("reaction_operations") or []
                if isinstance(operation, Mapping)
            ],
        }
        reaction_family = str(
            row.get("reaction_family") or row.get("transformation_hypothesis") or ""
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
                (prediction.get("catalyst") or "" for prediction in predictions),
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
            next_declared = _canonical_smiles(scaffold[index + 1].get("product_smiles"))
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
    approximate = [
        pair
        for pair in candidates
        if _canonical_smiles_nonisomeric(pair[0]) == nonisomeric_declared
    ]
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

    required_maps = set(_reaction_operation_atom_maps(operations))
    if not required_maps:
        return None
    candidates: list[tuple[str, str]] = []
    for value, mapped in zip(precursors, mapped_precursors):
        canonical = _canonical_smiles(value)
        mapped_text = str(mapped or "")
        if not canonical or not mapped_text:
            continue
        observed_maps = {int(raw) for raw in re.findall(r":(\d+)", mapped_text) if raw.isdigit()}
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
            dict(value) for value in payload.get("candidates") or [] if isinstance(value, Mapping)
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
                        "invalidated_bond_stereo",
                        "required_repair",
                    )
                    if key in replay
                },
                "declared_precursor_smiles": list(replay.get("declared_precursor_smiles") or []),
                "replayed_precursor_smiles": [],
            }
        return {
            "reason": "invalid_expansion_contract",
            "replay_error": str(replay.get("reason") or ""),
            "declared_precursor_smiles": list(replay.get("declared_precursor_smiles") or []),
            "replayed_precursor_smiles": list(replay.get("replayed_precursor_smiles") or []),
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
        values = normalized.get("anchor_bond_signature") or normalized.get("key_bond_signature")
    for value in values or ():
        match = re.fullmatch(r"map_pair:(\d+):(\d+)", str(value or "").strip())
        if match:
            pairs.add(tuple(sorted((int(match.group(1)), int(match.group(2))))))
    return frozenset(pairs)


def _strategy_card_digest(strategy_card: Mapping[str, Any] | None) -> str:
    card = dict(strategy_card or {})
    return str(
        card.get("strategy_digest") or card.get("content_sha256") or card.get("strategy_id") or ""
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


def _strategy_lineage_root(
    strategy_card: Mapping[str, Any] | None,
) -> str:
    lineage = dict(dict(strategy_card or {}).get("host_lineage") or {})
    return str(lineage.get("root_mapped_smiles") or "").strip()


def _selected_leaf_descends_from_mapped_root(
    *,
    steps: Iterable[Mapping[str, Any]],
    root_mapped_smiles: str,
    selected_product_mapped: str,
) -> bool:
    """Follow only Host-carried mapped precursor identities across the DAG."""

    root = _canonical_mapped_smiles(root_mapped_smiles)
    selected = _canonical_mapped_smiles(selected_product_mapped)
    if not root or not selected:
        return False
    if root == selected:
        return True
    children: dict[str, set[str]] = {}
    for raw in steps:
        if not isinstance(raw, Mapping):
            continue
        product = _canonical_mapped_smiles(raw.get("mapped_product_smiles"))
        if not product:
            continue
        children.setdefault(product, set()).update(
            precursor
            for value in raw.get("mapped_precursor_smiles") or []
            if (precursor := _canonical_mapped_smiles(value))
        )
    frontier = [root]
    visited = {root}
    while frontier:
        current = frontier.pop()
        for precursor in children.get(current, ()):
            if precursor == selected:
                return True
            if precursor not in visited:
                visited.add(precursor)
                frontier.append(precursor)
    return False


def _strategy_card_applies_to_leaf(
    strategy_card: Mapping[str, Any],
    *,
    steps: Iterable[Mapping[str, Any]],
    selected_product_mapped: str,
) -> bool:
    root = _strategy_lineage_root(strategy_card)
    return bool(
        root
        and _selected_leaf_descends_from_mapped_root(
            steps=steps,
            root_mapped_smiles=root,
            selected_product_mapped=selected_product_mapped,
        )
    )


def _key_event_obligation_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("obligation_id") or "").strip()
    if explicit:
        return explicit
    payload = {
        "strategy_digest": str(row.get("strategy_digest") or ""),
        "strategy_milestone_index": int(row.get("strategy_milestone_index") or 1),
        "focus_step_id": str(row.get("focus_step_id") or ""),
        "lineage_root_mapped_smiles": str(row.get("lineage_root_mapped_smiles") or ""),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]


def _pending_uncertain_key_event_evidence_review(
    branch: Mapping[str, Any],
    *,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one newly materialized direct-precursor review, if any.

    An uncertain key event is reviewed only after AiZ has selected a new step
    that synthesizes one of that event's direct mapped precursors.  Each such
    evidence step is reviewed once, avoiding both the old never-revisit gap
    and a per-Builder-call Critic loop.
    """

    active_digest = _strategy_card_digest(strategy_card)
    active_milestone = _strategy_milestone_index(branch, strategy_card)
    step_rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    by_id = {
        str(row.get("step_id") or ""): row for row in step_rows if str(row.get("step_id") or "")
    }
    reviewed_pairs = {
        (
            str(row.get("review_of_obligation_id") or ""),
            str(row.get("review_evidence_step_id") or ""),
        )
        for row in branch.get("key_event_critic_history") or ()
        if isinstance(row, Mapping)
        and str(row.get("review_of_obligation_id") or "")
        and str(row.get("review_evidence_step_id") or "")
    }
    for raw in branch.get("key_event_critic_history") or ():
        if not isinstance(raw, Mapping) or str(raw.get("status") or "") != ("uncertain"):
            continue
        row = dict(raw)
        if row.get("review_of_obligation_id"):
            continue
        row_digest = str(row.get("strategy_digest") or "")
        if row_digest:
            if not active_digest or row_digest != active_digest:
                continue
        elif int(row.get("strategy_milestone_index") or 1) != active_milestone:
            continue
        focus_step_id = str(row.get("focus_step_id") or "")
        focus_step = by_id.get(focus_step_id)
        if focus_step is None:
            continue
        direct_precursors = {
            _canonical_mapped_smiles(value)
            for value in focus_step.get("mapped_precursor_smiles") or ()
            if _canonical_mapped_smiles(value)
        }
        if not direct_precursors:
            continue
        obligation_id = _key_event_obligation_id(row)
        for evidence_step in step_rows:
            evidence_step_id = str(evidence_step.get("step_id") or "")
            if not evidence_step_id or evidence_step_id == focus_step_id:
                continue
            if (
                _canonical_mapped_smiles(evidence_step.get("mapped_product_smiles"))
                not in direct_precursors
            ):
                continue
            if (obligation_id, evidence_step_id) in reviewed_pairs:
                continue
            return {
                "obligation_id": obligation_id,
                "focus_step_id": focus_step_id,
                "evidence_step_id": evidence_step_id,
                "lineage_root_mapped_smiles": str(row.get("lineage_root_mapped_smiles") or ""),
                "source_history_row": row,
            }
    return {}


def _pending_key_event_feedback_for_leaf(
    branch: Mapping[str, Any],
    *,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
    selected_product_mapped: str,
    include_uncertain: bool = False,
) -> dict[str, Any]:
    """Derive actionable Critic constraints for one Strategy/leaf lineage.

    ``key_event_critic_history`` is the sole authority. Rejected attempts are
    actionable at the same Builder leaf and therefore enter its retry prompt.
    An uncertain attempt is evidence debt, not a request for a later Builder
    to rewrite an already materialized edge; it is included only for the
    dedicated follow-up Critic. A later selected pass retires the horizon.
    Sibling leaves and later Strategy cards cannot inherit this feedback.
    """

    active_digest = _strategy_card_digest(strategy_card)
    active_milestone = _strategy_milestone_index(branch, strategy_card)
    step_rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    selected_step_ids = {
        str(row.get("step_id") or "") for row in step_rows if str(row.get("step_id") or "")
    }
    constraints: dict[str, dict[str, Any]] = {}
    rejected_attempts: dict[str, dict[str, Any]] = {}
    for raw in branch.get("key_event_critic_history") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row_digest = str(row.get("strategy_digest") or "")
        if row_digest:
            if not active_digest or row_digest != active_digest:
                continue
        elif int(row.get("strategy_milestone_index") or 1) != active_milestone:
            continue
        lineage_root = str(row.get("lineage_root_mapped_smiles") or "").strip()
        if not lineage_root or not _selected_leaf_descends_from_mapped_root(
            steps=step_rows,
            root_mapped_smiles=lineage_root,
            selected_product_mapped=selected_product_mapped,
        ):
            continue
        status = str(row.get("status") or "")
        if status == "completed":
            required_selected_step_ids = {
                str(value)
                for value in row.get("required_selected_step_ids")
                or (str(row.get("focus_step_id") or ""),)
                if str(value)
            }
            if required_selected_step_ids and required_selected_step_ids.issubset(
                selected_step_ids
            ):
                constraints.clear()
                rejected_attempts.clear()
            continue
        if status == "uncertain" and include_uncertain:
            pass
        elif status != "rejected":
            continue
        assessment = dict(row.get("assessment") or {})
        reasons = [
            str(value)[:260] for value in assessment.get("reasons") or [] if str(value).strip()
        ][:2]
        suggested_revision = str(assessment.get("suggested_revision") or "")[:420]
        blocking_type = str(assessment.get("blocking_type") or "none")
        if not reasons and not suggested_revision:
            continue
        review_evidence_step_id = str(row.get("review_evidence_step_id") or "")
        # A rejected follow-up is the latest authority for the unresolved
        # obligation even after the rejected evidence edge has been pruned.
        # Requiring that removed edge to remain selected hid the blocking
        # verdict and exposed the retry path to the older ``uncertain`` row.
        # Non-rejected reviews remain path-local: they must not affect a
        # sibling path that did not select their evidence step.
        if (
            status != "rejected"
            and review_evidence_step_id
            and review_evidence_step_id not in selected_step_ids
        ):
            continue
        obligation_id = str(row.get("review_of_obligation_id") or _key_event_obligation_id(row))
        fingerprint = str(row.get("fingerprint") or obligation_id)
        rejected_attempts[fingerprint] = {
            "blocking_type": blocking_type,
            "checkpoint_match": row.get("checkpoint_match") is True,
        }
        constraints[obligation_id] = {
            "obligation_id": obligation_id,
            "severity": (
                "blocking"
                if status == "rejected" or assessment.get("blocking") is True
                else "warning"
            ),
            "checkpoint_match": row.get("checkpoint_match") is True,
            "blocking_type": blocking_type,
            "reasons": reasons,
            "suggested_revision": suggested_revision,
            "source_focus_step_id": str(row.get("focus_step_id") or ""),
        }
    if not constraints:
        return {}
    feedback: dict[str, Any] = {
        "strategy_digest": active_digest,
        "active_constraints": list(constraints.values()),
    }
    # This is a compact derived projection, not a second rejection authority.
    # Keep candidate-level Critic rows append-only, but expose when distinct
    # graph proposals are repeatedly falling into the same chemical basin so
    # Builder and Critic can stop treating every new focus_step_id as amnesia.
    if len(rejected_attempts) >= 2:
        blocking_counts = Counter(
            str(row.get("blocking_type") or "none") for row in rejected_attempts.values()
        )
        recurring = sorted(
            key for key, count in blocking_counts.items() if key != "none" and count >= 2
        )
        feedback["failure_basin"] = {
            "distinct_rejected_attempt_count": len(rejected_attempts),
            "blocking_type_counts": dict(sorted(blocking_counts.items())),
            "recurring_blocking_types": recurring,
            "distinct_candidate_fingerprints": sorted(rejected_attempts),
            "checkpoint_match_count": sum(
                1 for row in rejected_attempts.values() if row.get("checkpoint_match") is True
            ),
            "recurrent_across_distinct_candidates": bool(recurring),
            "authority": "derived_diagnostic_only",
        }
    return feedback


def _strategy_anchor_fulfilled_for_card(
    steps: Iterable[Mapping[str, Any]],
    strategy_card: Mapping[str, Any] | None,
) -> bool:
    rows = [dict(step) for step in steps if isinstance(step, Mapping)]
    required = _strategy_key_bond_pairs(strategy_card)
    if required:
        return required.issubset(_realized_strategy_key_bond_pairs(rows, strategy_card))
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
    has_bound_cards = any(isinstance(row.get("strategy_card"), Mapping) for row in rows)
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
    for raw in [
        root_strategy_card,
        *[dict(step).get("strategy_card") for step in steps if isinstance(step, Mapping)],
    ]:
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


def _final_route_strategy_card(branch: Mapping[str, Any]) -> dict[str, Any]:
    """Return the last Strategy horizon actually bound to the selected route."""

    root = dict(branch.get("root_strategy_card") or branch.get("strategy_card") or {})
    selected = _ordered_strategy_cards_from_steps(
        root_strategy_card=root,
        steps=(dict(row) for row in branch.get("steps") or [] if isinstance(row, Mapping)),
    )
    if selected:
        return dict(selected[-1])
    milestones = [
        dict(row)
        for row in branch.get("strategy_milestone_cards") or []
        if isinstance(row, Mapping)
    ]
    return dict(milestones[-1] if milestones else root)


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


def _rejected_strategy_horizon_for_leaf(
    branch: Mapping[str, Any],
    *,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
    selected_product_mapped: str,
) -> dict[str, Any]:
    """Return the selected-lineage Critic decision that retires a Strategy.

    The append-only Key Critic history remains the sole authority.  No retry
    counter or copied branch flag can independently abandon a horizon.
    """

    digest = _strategy_card_digest(strategy_card)
    milestone_index = _strategy_milestone_index(branch, strategy_card)
    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    for raw in reversed(list(branch.get("key_event_critic_history") or [])):
        if not isinstance(raw, Mapping) or str(raw.get("status") or "") != "rejected":
            continue
        row = dict(raw)
        assessment = dict(row.get("assessment") or {})
        if str(assessment.get("repair_scope") or "") != "strategy_horizon":
            continue
        row_digest = str(row.get("strategy_digest") or "")
        if row_digest:
            if not digest or row_digest != digest:
                continue
        elif int(row.get("strategy_milestone_index") or 1) != milestone_index:
            continue
        lineage_root = str(row.get("lineage_root_mapped_smiles") or "").strip()
        if not lineage_root or not _selected_leaf_descends_from_mapped_root(
            steps=rows,
            root_mapped_smiles=lineage_root,
            selected_product_mapped=selected_product_mapped,
        ):
            continue
        return row
    return {}


def _strategy_horizon_for_leaf(
    *,
    config: DirectorConfig,
    branch: Mapping[str, Any],
    root_strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
    selected_product_mapped: str,
) -> tuple[Mapping[str, Any], bool]:
    """Resolve the active Strategy and whether it needs a new horizon."""

    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    receding_horizon = bool(
        config.enable_key_event_critic and int(config.max_strategic_milestones_per_branch) > 1
    )
    if receding_horizon:
        cards = [
            dict(row)
            for row in branch.get("strategy_milestone_cards") or []
            if isinstance(row, Mapping)
        ]
        applicable = [
            card
            for card in cards[1:]
            if _strategy_card_applies_to_leaf(
                card,
                steps=rows,
                selected_product_mapped=selected_product_mapped,
            )
        ]
        if applicable:
            active = applicable[-1]
            return (
                active,
                bool(
                    _selected_path_passed_strategy_checkpoint(
                        branch,
                        strategy_card=active,
                        steps=rows,
                    )
                    or _rejected_strategy_horizon_for_leaf(
                        branch,
                        strategy_card=active,
                        steps=rows,
                        selected_product_mapped=selected_product_mapped,
                    )
                ),
            )
        root_retired = _selected_path_passed_strategy_checkpoint(
            branch,
            strategy_card=root_strategy_card,
            steps=rows,
        )
        if not root_retired:
            root_retired = bool(
                _rejected_strategy_horizon_for_leaf(
                    branch,
                    strategy_card=root_strategy_card,
                    steps=rows,
                    selected_product_mapped=selected_product_mapped,
                )
            )
        return root_strategy_card, root_retired
    if config.paper_matched_reach_profile:
        return root_strategy_card, False
    active = _active_strategy_card_for_leaf(
        root_strategy_card=root_strategy_card,
        steps=rows,
        selected_product_mapped=selected_product_mapped,
    )
    return (
        active,
        bool(
            _strategy_anchor_fulfilled_for_card(rows, root_strategy_card)
            and active is root_strategy_card
        ),
    )


def _strategy_milestone_index(branch: Mapping[str, Any], strategy_card: Mapping[str, Any]) -> int:
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


def _selected_path_passed_strategy_checkpoint(
    branch: Mapping[str, Any],
    *,
    strategy_card: Mapping[str, Any],
    steps: Iterable[Mapping[str, Any]],
) -> bool:
    """Return whether AiZ selected a Critic-passed checkpoint proposal."""

    digest = _strategy_card_digest(strategy_card)
    milestone_index = _strategy_milestone_index(branch, strategy_card)
    selected_step_ids = {
        str(row.get("step_id") or "")
        for row in steps
        if isinstance(row, Mapping) and str(row.get("step_id") or "")
    }
    if not selected_step_ids:
        return False
    return any(
        isinstance(row, Mapping)
        and str(row.get("status") or "") == "completed"
        and str(row.get("focus_step_id") or "") in selected_step_ids
        and (
            (digest and str(row.get("strategy_digest") or "") == digest)
            or (
                not str(row.get("strategy_digest") or "")
                and int(row.get("strategy_milestone_index") or 1) == milestone_index
            )
        )
        for row in branch.get("key_event_critic_history") or []
    )


def _strategy_milestone_progress(
    branch: Mapping[str, Any],
    *,
    steps: Iterable[Mapping[str, Any]],
    strategy_card: Mapping[str, Any],
    use_key_event_critic: bool,
) -> dict[str, Any]:
    """Derive one displayed milestone from the execution authority in use."""

    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    progress = _strategy_anchor_progress(rows, strategy_card)
    if not use_key_event_critic:
        return progress
    mapped_edit_overlap = progress.get("fulfilled") is True
    critic_confirmed = _selected_path_passed_strategy_checkpoint(
        branch,
        strategy_card=strategy_card,
        steps=rows,
    )
    return {
        **progress,
        "fulfilled": critic_confirmed,
        "mapped_edit_overlap": mapped_edit_overlap,
        "checkpoint_critic_confirmed": critic_confirmed,
        "authority": "selected_path_key_event_critic",
        "grants_strategy_completion": critic_confirmed,
        "grants_route_admission": False,
        "completion_semantics": ("host_replayed_selected_step_with_key_event_critic_pass"),
    }


def _refresh_strategy_milestone_projection(
    branch: dict[str, Any],
    *,
    strategy_cards: Iterable[Mapping[str, Any]],
    use_key_event_critic: bool,
) -> None:
    """Refresh the report projection from final steps and append-only audits."""

    cards = [dict(card) for card in strategy_cards if isinstance(card, Mapping)]
    steps = [dict(row) for row in branch.get("steps") or [] if isinstance(row, Mapping)]
    diagnostics = [
        {
            "strategy_id": str(card.get("strategy_id") or ""),
            "strategy_digest": str(card.get("strategy_digest") or ""),
            **_strategy_milestone_progress(
                branch,
                steps=steps,
                strategy_card=card,
                use_key_event_critic=use_key_event_critic,
            ),
        }
        for card in cards
    ]
    branch["strategic_milestone_count"] = sum(row.get("fulfilled") is True for row in diagnostics)
    branch["strategy_anchor_diagnostics"] = diagnostics


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


def _generated_builder_step_id(
    branch: Mapping[str, Any],
    *,
    branch_index: int,
    call_index: int,
    candidate_index: int,
) -> str:
    prefix = str(
        branch.get("generated_step_id_prefix") or f"codex:branch:{branch_index + 1}"
    )
    return f"{prefix}:node:{call_index}:candidate:{candidate_index + 1}"


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
    raw_strategy_card = dict(expansion.strategy_card or {})
    strategy_card = normalize_strategy_policy_card(raw_strategy_card)
    host_lineage = raw_strategy_card.get("host_lineage")
    if isinstance(host_lineage, Mapping):
        lineage_root = str(host_lineage.get("root_mapped_smiles") or "").strip()
        if lineage_root:
            # Lineage is Host scheduling metadata, not part of the chemical
            # Strategy identity or digest.
            strategy_card["host_lineage"] = {
                "root_mapped_smiles": lineage_root,
                "milestone_index": max(1, int(host_lineage.get("milestone_index") or 1)),
            }
    edit_digest = reaction_edit_digest(expansion.reaction_operations)
    conditions = tuple(
        value for raw in expansion.conditions if (value := _clean_condition_text(raw))
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
        "checkpoint_relation": _normalize_checkpoint_relation(expansion.checkpoint_relation),
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
        materialized = [_materialized_reaction_from_bound_step(value) for value in rows]
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
                    mapped_product_smiles=str(value.get("mapped_product_smiles") or ""),
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
                "execution_domain": str(value.get("execution_domain") or "chemical"),
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
    if not mapped_product or _canonical_atom_mapped_smiles(
        mapped_product
    ) != _canonical_atom_mapped_smiles(audited_mapped_product):
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
            str(value) for value in row.get("mapped_precursor_smiles") or [] if str(value)
        ),
        reaction_operations=tuple(
            dict(value)
            for value in normalize_reaction_operations(row.get("reaction_operations") or ())
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
    reserved_atom_maps: Iterable[int] = (),
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
            reserved_atom_maps=reserved_atom_maps,
        )
    except ReactionJsonReplayError as exc:
        failed_index = 0
        for prefix_size in range(1, len(rows) + 1):
            try:
                RouteJSONCompiler().compile_route_graph(
                    mapped_target_smiles=target_mapped,
                    steps=rows[:prefix_size],
                    minimum_depth=1,
                    reserved_atom_maps=reserved_atom_maps,
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


def _materialize_aizynthfinder_projection(
    *,
    steps: Iterable[Mapping[str, Any]],
    mapped_target_smiles: str,
    search_diagnostics: Mapping[str, Any],
    stock_membership: StockMembership,
) -> dict[str, Any]:
    """Compile one selected AiZ path and derive stock closure from its leaves.

    AiZ owns tree search and records whether its selected node was solved. The
    selected path becomes an AutoPlanner route only when every path action has
    Host metadata, the complete target-rooted RouteJSON replays, and the leaves
    produced by that replay pass the bound exact-identity stock oracle.
    """

    rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    diagnostics = dict(search_diagnostics or {})
    path_action_count = int(diagnostics.get("path_action_count") or 0)
    path_route_step_count = int(diagnostics.get("path_route_step_count") or 0)
    route_projection_complete = bool(
        diagnostics.get("path_route_projection_complete") is True
        and path_action_count == path_route_step_count == len(rows)
    )
    replay_validation = _route_steps_host_replay_validation(
        rows,
        mapped_target_smiles=mapped_target_smiles,
    )
    if replay_validation.get("complete") is not True:
        return {
            "steps": rows,
            "open_leaf_states": [],
            "route_projection_complete": False,
            "leaf_closure_complete": False,
            "terminal_leaf_count": 0,
            "routejson_replay_validation": replay_validation,
        }

    compiler = RouteJSONCompiler()
    state = compiler.compile_route_graph_state(
        mapped_target_smiles=str(mapped_target_smiles or ""),
        steps=rows,
        minimum_depth=1,
    )
    materialized_steps = compiler.assemble_route(state.reactions, metadata=rows)
    terminal_states = [
        {
            "smiles": occurrence.product_smiles,
            "mapped_smiles": occurrence.mapped_product_smiles,
        }
        for occurrence in state.open_precursors
    ]
    membership = stock_membership(tuple(str(row["smiles"]) for row in terminal_states))
    unresolved = [row for row in terminal_states if membership.get(str(row["smiles"])) is not True]
    leaf_closure_complete = bool(route_projection_complete and terminal_states and not unresolved)
    return {
        "steps": materialized_steps,
        "open_leaf_states": unresolved,
        "route_projection_complete": route_projection_complete,
        "leaf_closure_complete": leaf_closure_complete,
        "terminal_leaf_count": len(terminal_states),
        "routejson_replay_validation": replay_validation,
    }


def _route_steps_are_host_replayable(
    steps: Iterable[Mapping[str, Any]],
    *,
    mapped_target_smiles: str,
) -> bool:
    return (
        _route_steps_host_replay_validation(
            steps,
            mapped_target_smiles=mapped_target_smiles,
        ).get("complete")
        is True
    )


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
            dict(item) for item in row.get("hard_failures") or [] if isinstance(item, Mapping)
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
        family_id = str(
            branch.get("route_family_alias_override") or f"codex:sequential:family:{ordinal}"
        )
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
                "root_strategy_card": dict(branch.get("root_strategy_card") or strategy_card),
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
                "strategic_milestone_count": int(branch.get("strategic_milestone_count") or 0),
                "strategy_id": str(strategy_card.get("strategy_id") or ""),
                "strategy_digest": str(strategy_card.get("strategy_digest") or ""),
                "execution_domain": str(strategy_card.get("execution_domain") or "chemical"),
                "route_execution_profile": execution_profile,
                "chemical_critic": dict(branch.get("chemical_critic") or {}),
                "strategy_tree_engine": str(
                    branch.get("strategy_tree_engine") or "chemenzy_best_first"
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
                "strategy_call_count": int(branch.get("strategy_call_count") or 0),
                "route_call_count": int(branch.get("route_call_count") or 0),
                "path_repair_builder_call_count": int(
                    branch.get("path_repair_builder_call_count") or 0
                ),
                # Keep observed stock closure in the diagnostic namespace;
                # ``GlobalCampaignPlan`` rejects authority-looking keys such
                # as ``stock_closed=true`` because plans themselves cannot
                # grant scientific status.  The paper metric is computed
                # later from the materialized route/stock ledger.
                "paper_policy_call_budget": _paper_policy_budget_projection(
                    branch.get("paper_policy_call_budget") or {}
                ),
                "path_repair_aizynthfinder_search": dict(
                    branch.get("path_repair_aizynthfinder_search") or {}
                ),
                "paper_policy_budget_failure": dict(
                    branch.get("paper_policy_budget_failure") or {}
                ),
                "shared_model_budget_ledger": dict(branch.get("shared_model_budget_ledger") or {}),
                "sidecar_recovered_prefix": bool(branch.get("sidecar_recovered_prefix")),
                "editor_attempt_count": int(branch.get("editor_attempt_count") or 0),
                "editor_call_count": int(branch.get("editor_call_count") or 0),
                "critic_call_count": int(branch.get("critic_call_count") or 0),
                "key_event_critic_call_count": int(branch.get("key_event_critic_call_count") or 0),
                "key_event_critic_completed": bool(branch.get("key_event_critic_completed")),
                "key_event_critic_history": [
                    dict(row)
                    for row in branch.get("key_event_critic_history") or []
                    if isinstance(row, Mapping)
                ],
                "pending_key_event_feedback": dict(branch.get("pending_key_event_feedback") or {}),
                "critic_editor_skipped_incomplete_route_json": bool(
                    branch.get("critic_editor_skipped_incomplete_route_json")
                ),
                "editor_applied_count": int(branch.get("editor_call_count") or 0),
                "materialization_failures": dict(branch.get("materialization_failures") or {}),
                "blocked_materializations": list(branch.get("blocked_materializations") or []),
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
                "reactionjson_or_search": dict(branch.get("reactionjson_or_search") or {}),
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
                "path_repair_transactions": [
                    dict(row)
                    for row in branch.get("path_repair_transactions") or []
                    if isinstance(row, Mapping)
                ],
                "route_alternatives": [
                    dict(row)
                    for row in branch.get("route_alternatives") or []
                    if isinstance(row, Mapping)
                ],
                "editor_working_route": dict(branch.get("editor_working_route") or {}),
                "editor_rejection_diagnostics": [
                    dict(row)
                    for row in branch.get("editor_rejection_diagnostics") or []
                    if isinstance(row, Mapping)
                ],
                "supersedes_route_family_id": str(branch.get("supersedes_route_family_id") or ""),
                "repair_origin_route_sha256": str(branch.get("repair_origin_route_sha256") or ""),
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
                    "root_strategy_card": dict(branch.get("root_strategy_card") or strategy_card),
                    "strategy_milestone_cards": [
                        dict(row)
                        for row in branch.get("strategy_milestone_cards") or []
                        if isinstance(row, Mapping)
                    ],
                    "strategic_milestone_count": int(branch.get("strategic_milestone_count") or 0),
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
                    "critic_call_count": int(branch.get("critic_call_count") or 0),
                    "editor_attempt_count": int(branch.get("editor_attempt_count") or 0),
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
                    "editor_working_route": dict(branch.get("editor_working_route") or {}),
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
                "strategic_role": "distinct open leaf requiring normal Builder continuation",
            }
        )
        priorities.append(
            {
                "priority_id": f"codex:sequential:priority:{index}",
                "proposal_id": intermediate_id,
                "target_smiles": leaf,
                "route_family_ids": sorted(family_ids),
                "provider_preferences": ["codex_frontier_builder"],
                "retron_hints": [],
                "priority": max(1.0, 100.0 - index),
                "rationale": "continue the same branch with the standard one-step Builder contract",
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
            "Every open leaf from a replayable route continues through the same standard Builder contract."
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
    completed_rows = [row for row in rows if not worker_provider_failure_reason(row)]
    provider_failure_count = len(rows) - len(completed_rows)
    result: dict[str, int | float] = {
        "model_invocations": len(completed_rows),
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "accepted_expansions": sum(row.status == "accepted_draft" for row in completed_rows),
        "attempt_runs": len(rows),
        "provider_failure_count": provider_failure_count,
        "wall_time_s": max(0.0, float(elapsed_s)),
    }
    for record in completed_rows:
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

    if worker_provider_failure_reason(record):
        return "provider_error"
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


def _canonical_mapped_reactant_smiles(value: Any) -> str:
    """Canonicalize a mapped reactant while allowing an unmapped leaving atom.

    The Host reaction verifier deliberately permits bounded, unmapped departing
    atoms (for example chloride in a silylation).  A Route Critic may inspect
    that verified mapping without treating it as reaction proof.  Builder
    replay still uses :func:`_canonical_mapped_smiles`, which requires every
    atom in its mutable product boundary to be mapped.
    """

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None or not any(atom.GetAtomMapNum() > 0 for atom in molecule.GetAtoms()):
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _validated_edge_mapped_boundaries(
    edge: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Read mapped boundaries from the current Host reaction proof.

    Ordinary route hypotheses are atom-mapped by the Host validator after
    materialization, so they do not necessarily carry a ReactionJSON replay
    audit. The proof already binds one complete mapping to the exact canonical
    product/reactant multiset; consume that authority directly instead of
    copying a second mapped representation onto the edge.
    """

    expected_product = _canonical_smiles(edge.get("product_smiles"))
    expected_precursors = sorted(
        _canonical_smiles(value) for value in edge.get("precursor_smiles") or ()
    )
    required_checks = (
        "mapped_reaction_present",
        "mapped_product_matches",
        "mapped_reactants_match",
        "atom_maps_complete",
        "product_atom_maps_complete",
        "atom_maps_unique",
    )
    for proof in reversed(active_reaction_proofs(edge.get("reaction_proofs") or ())):
        checks = dict(proof.get("checks") or {})
        if proof.get("accepted") is not True or any(
            checks.get(name) is not True for name in required_checks
        ):
            continue
        parts = str(proof.get("mapped_reaction") or "").split(">")
        if len(parts) != 3:
            continue
        mapped_precursors = [
            value for item in parts[0].split(".") if (value := _canonical_mapped_smiles(item))
        ]
        mapped_products = [
            value for item in parts[2].split(".") if (value := _canonical_mapped_smiles(item))
        ]
        if (
            len(mapped_products) != 1
            or _canonical_smiles(mapped_products[0]) != expected_product
            or len(mapped_precursors) != len(expected_precursors)
            or sorted(_canonical_smiles(value) for value in mapped_precursors)
            != expected_precursors
        ):
            continue
        return mapped_products[0], mapped_precursors
    return "", []


def _route_critic_edge_mapped_boundaries(
    edge: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Read a Host-bound mapping for Critic inspection, not route admission.

    ``ReactionStepProof.accepted`` means that the reaction itself reached a
    validated-transform or precedent tier.  It is intentionally false for a
    structurally consistent ``L2_mapping_consistent`` edge.  Final route review
    needs the latter mapping as input, while the proof tier and its rejection
    reasons remain unchanged.  Product identity must be fully mapped; bounded
    unmapped leaving atoms are allowed only on precursor components.
    """

    expected_product = _canonical_smiles(edge.get("product_smiles"))
    expected_precursors = sorted(
        _canonical_smiles(value) for value in edge.get("precursor_smiles") or ()
    )
    # These checks establish that the serialized mapping names the canonical
    # edge and that every product atom has an unambiguous provenance identity.
    # Other verifier checks (departure budget, transform registry, precedent,
    # conditions) remain chemical-proof axes and must not gate Critic input.
    required_checks = (
        "structures_materialized",
        "mapped_reaction_present",
        "mapped_product_matches",
        "mapped_reactants_match",
        "product_atom_maps_complete",
        "atom_maps_unique",
        "mapped_elements_preserved",
        "stereochemical_product_matches",
    )

    for proof in reversed(active_reaction_proofs(edge.get("reaction_proofs") or ())):
        checks = dict(proof.get("checks") or {})
        if any(checks.get(name) is not True for name in required_checks) or not (
            checks.get("product_atoms_have_reactant_provenance") is True
            or checks.get("external_atom_source_replayed") is True
        ):
            continue
        parts = str(proof.get("mapped_reaction") or "").split(">")
        if len(parts) != 3:
            continue
        # Host proofs are normally forward (precursors >> product), but bind
        # direction from canonical identities so imported mapped reactions
        # cannot silently invert the route.
        for precursor_text, product_text in ((parts[0], parts[2]), (parts[2], parts[0])):
            mapped_precursors = [
                value
                for item in precursor_text.split(".")
                if (value := _canonical_mapped_reactant_smiles(item))
            ]
            mapped_products = [
                value
                for item in product_text.split(".")
                if (value := _canonical_mapped_smiles(item))
            ]
            if (
                len(mapped_products) == 1
                and _canonical_smiles(mapped_products[0]) == expected_product
                and len(mapped_precursors) == len(expected_precursors)
                and sorted(_canonical_smiles(value) for value in mapped_precursors)
                == expected_precursors
            ):
                return mapped_products[0], mapped_precursors
    return "", []


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
                    "mapped_product_smiles": str(row.get("mapped_product_smiles") or ""),
                    "mapped_precursor_smiles": list(row.get("mapped_precursor_smiles") or []),
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


def _frontier_strategy_card(
    route: Mapping[str, Any],
    *,
    route_family_id: str,
    connected_steps: Sequence[Mapping[str, Any]],
    selected_leaf_mapped: str,
) -> dict[str, Any]:
    """Resolve the newest declared Strategy horizon on this exact leaf lineage."""

    root = dict(route.get("strategy_card") or {})
    milestones = [
        dict(row) for row in route.get("strategy_milestone_cards") or [] if isinstance(row, Mapping)
    ]
    applicable = [
        row
        for row in milestones
        if _strategy_card_applies_to_leaf(
            row,
            steps=connected_steps,
            selected_product_mapped=selected_leaf_mapped,
        )
    ]
    if applicable:
        card = max(
            applicable,
            key=lambda row: int(
                dict(row.get("host_lineage") or {}).get("milestone_index")
                or row.get("strategy_milestone_index")
                or 1
            ),
        )
    else:
        card = root
    return normalize_strategy_policy_card(card, route_family_id=route_family_id)


def _frontier_unresolved_path_repair(
    route: Mapping[str, Any],
    *,
    connected_steps: Sequence[Mapping[str, Any]],
    selected_leaf_mapped: str,
) -> dict[str, Any]:
    """Derive one still-binding repair obligation for the selected lineage."""

    unresolved_statuses = {
        "retained_uncommitted_prefix",
        "rolled_back_uncommitted",
        "rolled_back_after_recritic",
    }
    transactions = [
        dict(row) for row in route.get("path_repair_transactions") or [] if isinstance(row, Mapping)
    ]
    transactions.sort(
        key=lambda row: (
            int(row.get("transaction_index") or 0),
            int(row.get("iteration") or 0),
        )
    )
    for transaction in reversed(transactions):
        if str(transaction.get("status") or "") not in unresolved_statuses:
            continue
        repair_root = str(transaction.get("repair_frontier_mapped_product_smiles") or "")
        if not repair_root or not _selected_leaf_descends_from_mapped_root(
            steps=connected_steps,
            root_mapped_smiles=repair_root,
            selected_product_mapped=selected_leaf_mapped,
        ):
            continue
        return {
            key: value
            for key, value in transaction.items()
            if key
            in {
                "transaction_index",
                "status",
                "reason",
                "rollback_start_step_id",
                "rebuild_through_step_id",
                "repair_goal",
                "active_constraints",
                "repair_frontier_mapped_product_smiles",
                "reconnect_boundaries",
            }
            and value not in (None, "", [], {})
        }
    return {}


def compile_frontier_builder_context(
    graph: Mapping[str, Any],
    *,
    frontier_molecule_id: str,
    route_family_ids: Iterable[str],
    attempt_index: int = 1,
    prior_rejections: Iterable[Mapping[str, Any]] = (),
) -> tuple[FrontierBuilderContext | None, dict[str, Any]]:
    """Resolve one unique canonical target-to-leaf path for Builder continuation.

    No model or projection may invent the selected mapped leaf.  The mapping,
    route history, strategy, and atom namespace all come from the one
    materialized parent route in the canonical hypergraph.
    """

    requested_routes = tuple(sorted({str(value) for value in route_family_ids if str(value)}))
    if len(requested_routes) != 1:
        return None, {
            "reason": "frontier_builder_route_family_binding_ambiguous",
            "route_family_ids": list(requested_routes),
        }
    route_id = requested_routes[0]
    routes = dict(graph.get("route_families") or {})
    route = dict(routes.get(route_id) or {})
    if not route:
        return None, {
            "reason": "frontier_builder_route_family_missing",
            "route_family_id": route_id,
        }
    molecules = dict(graph.get("molecules") or {})
    target_id = str(graph.get("target_molecule_id") or "")
    leaf_id = str(frontier_molecule_id or "")
    target = dict(molecules.get(target_id) or {})
    leaf = dict(molecules.get(leaf_id) or {})
    target_smiles = _canonical_smiles(target.get("canonical_smiles"))
    leaf_smiles = _canonical_smiles(leaf.get("canonical_smiles"))
    if not target_id or not target_smiles or not leaf_id or not leaf_smiles:
        return None, {"reason": "frontier_builder_molecule_identity_missing"}

    edges = dict(graph.get("edges") or {})
    scoped_edges = [
        dict(edges.get(str(edge_id)) or {})
        for edge_id in route_family_scoped_edge_ids(graph, family=route)
        if isinstance(edges.get(str(edge_id)), Mapping)
    ]
    by_product: dict[str, list[dict[str, Any]]] = {}
    for edge in scoped_edges:
        product_id = str(edge.get("product_molecule_id") or "")
        if product_id:
            by_product.setdefault(product_id, []).append(edge)

    paths: list[list[dict[str, Any]]] = []

    def visit(
        molecule_id: str,
        path: list[dict[str, Any]],
        seen: set[str],
    ) -> None:
        if len(paths) > 1:
            return
        if molecule_id == leaf_id:
            paths.append([dict(row) for row in path])
            return
        if molecule_id in seen:
            return
        next_seen = {*seen, molecule_id}
        for edge in sorted(
            by_product.get(molecule_id, ()),
            key=lambda row: str(row.get("edge_id") or ""),
        ):
            for precursor_id in edge.get("precursor_molecule_ids") or ():
                value = str(precursor_id or "")
                if value:
                    visit(value, [*path, edge], next_seen)

    visit(target_id, [], set())
    if not paths:
        return None, {
            "reason": "frontier_builder_target_rooted_path_missing",
            "frontier_molecule_id": leaf_id,
            "route_family_id": route_id,
        }
    if len(paths) != 1:
        return None, {
            "reason": "frontier_builder_target_rooted_path_ambiguous",
            "frontier_molecule_id": leaf_id,
            "route_family_id": route_id,
            "path_count": len(paths),
        }
    path = paths[0]
    if not path:
        return None, {"reason": "frontier_builder_target_is_not_expandable_leaf"}

    connected_steps: list[dict[str, Any]] = []
    selected_leaf_mapped = ""
    for edge in path:
        audit = dict(edge.get("reactionjson_audit") or {})
        mapped_product = str(
            audit.get("mapped_product_smiles") or edge.get("mapped_product_smiles") or ""
        )
        mapped_precursors = [
            str(value)
            for value in (
                audit.get("mapped_precursor_smiles") or edge.get("mapped_precursor_smiles") or ()
            )
        ]
        if not mapped_product or not mapped_precursors:
            proof_product, proof_precursors = _validated_edge_mapped_boundaries(edge)
            if proof_product and proof_precursors:
                mapped_product = proof_product
                mapped_precursors = proof_precursors
        precursor_ids = [str(value) for value in edge.get("precursor_molecule_ids") or ()]
        edge_precursor_smiles = [
            _canonical_smiles(value) for value in edge.get("precursor_smiles") or ()
        ]
        precursor_smiles = [
            _canonical_smiles(dict(molecules.get(value) or {}).get("canonical_smiles"))
            for value in precursor_ids
        ]
        if (
            not _canonical_mapped_smiles(mapped_product)
            or _canonical_smiles(mapped_product) != _canonical_smiles(edge.get("product_smiles"))
            or not precursor_ids
            or any(not value for value in precursor_smiles)
            or sorted(edge_precursor_smiles) != sorted(precursor_smiles)
            or len(mapped_precursors) != len(precursor_ids)
            or any(not _canonical_mapped_smiles(value) for value in mapped_precursors)
        ):
            validation_pending = not bool(active_reaction_proofs(edge.get("reaction_proofs") or ()))
            return None, {
                "reason": "frontier_builder_mapped_path_incomplete",
                "edge_id": str(edge.get("edge_id") or ""),
                "retryable_after_reaction_validation": validation_pending,
                "prerequisite_kind": ("reaction_validation" if validation_pending else ""),
            }
        mapped_by_identity: dict[str, list[str]] = {}
        for value in mapped_precursors:
            mapped_by_identity.setdefault(_canonical_smiles(value), []).append(value)
        aligned_mapped_precursors: list[str] = []
        for value in precursor_smiles:
            matches = mapped_by_identity.get(value) or []
            if not matches:
                return None, {
                    "reason": "frontier_builder_mapped_precursor_identity_unbound",
                    "edge_id": str(edge.get("edge_id") or ""),
                    "precursor_smiles": value,
                }
            aligned_mapped_precursors.append(matches.pop(0))
        if any(values for values in mapped_by_identity.values()):
            return None, {
                "reason": "frontier_builder_mapped_precursor_identity_ambiguous",
                "edge_id": str(edge.get("edge_id") or ""),
            }
        origins = [
            dict(value) for value in edge.get("origin_records") or () if isinstance(value, Mapping)
        ]
        origin = next(
            (
                value
                for value in origins
                if route_id in {str(item) for item in value.get("canonical_route_family_ids") or ()}
            ),
            origins[0] if origins else {},
        )
        connected_steps.append(
            {
                "step_id": str(origin.get("proposal_id") or edge.get("edge_id") or ""),
                "product_smiles": _canonical_smiles(edge.get("product_smiles")),
                "mapped_product_smiles": mapped_product,
                "precursor_smiles": precursor_smiles,
                "mapped_precursor_smiles": aligned_mapped_precursors,
                "transformation_hypothesis": str(
                    origin.get("transformation_hypothesis")
                    or edge.get("transformation_hypothesis")
                    or ""
                ),
                "reaction_operations": [
                    dict(value)
                    for value in edge.get("reaction_operations") or ()
                    if isinstance(value, Mapping)
                ],
            }
        )
        matching_leaf_indices = [
            index for index, precursor_id in enumerate(precursor_ids) if precursor_id == leaf_id
        ]
        if matching_leaf_indices:
            if len(matching_leaf_indices) != 1:
                return None, {
                    "reason": "frontier_builder_mapped_leaf_occurrence_ambiguous",
                    "edge_id": str(edge.get("edge_id") or ""),
                }
            selected_leaf_mapped = aligned_mapped_precursors[matching_leaf_indices[0]]

    if (
        not _canonical_mapped_smiles(selected_leaf_mapped)
        or _canonical_smiles(selected_leaf_mapped) != leaf_smiles
    ):
        return None, {
            "reason": "frontier_builder_mapped_leaf_identity_missing",
            "frontier_molecule_id": leaf_id,
            "route_family_id": route_id,
        }
    strategy_card = _frontier_strategy_card(
        route,
        route_family_id=route_id,
        connected_steps=connected_steps,
        selected_leaf_mapped=selected_leaf_mapped,
    )
    pending_checkpoint_feedback = _pending_key_event_feedback_for_leaf(
        route,
        strategy_card=strategy_card,
        steps=connected_steps,
        selected_product_mapped=selected_leaf_mapped,
    )
    path_repair = _frontier_unresolved_path_repair(
        route,
        connected_steps=connected_steps,
        selected_leaf_mapped=selected_leaf_mapped,
    )
    return (
        FrontierBuilderContext(
            target_smiles=target_smiles,
            route_family_id=route_id,
            branch_index=_route_branch_index(route),
            selected_product_smiles=leaf_smiles,
            selected_product_mapped=selected_leaf_mapped,
            connected_steps=tuple(connected_steps),
            strategy_card=strategy_card,
            reserved_atom_maps=tuple(sorted(_route_atom_map_namespace(connected_steps))),
            prior_rejections=tuple(
                dict(value) for value in prior_rejections if isinstance(value, Mapping)
            )[-2:],
            attempt_index=max(1, int(attempt_index)),
            pending_checkpoint_feedback=pending_checkpoint_feedback,
            path_repair=path_repair,
        ),
        {},
    )


def compile_revision_bound_route_critic_context(
    graph: Mapping[str, Any],
    *,
    route_family_id: str,
    include_unselected: bool = False,
) -> tuple[RevisionBoundRouteCriticContext | None, dict[str, Any]]:
    """Compile the exact target-rooted route revision for final Critic review.

    The digest covers only chemistry visible to the Critic.  Unrelated graph
    revisions therefore do not trigger another model call, while any new
    materialized edge, edit program, mapped boundary, condition, or Strategy
    binding does.
    """

    route_id = str(route_family_id or "")
    route = dict(dict(graph.get("route_families") or {}).get(route_id) or {})
    if not route or (route.get("selected") is False and not include_unselected):
        return None, {
            "reason": "final_route_critic_route_unavailable",
            "route_family_id": route_id,
        }
    molecules = dict(graph.get("molecules") or {})
    edges = dict(graph.get("edges") or {})
    target_id = str(graph.get("target_molecule_id") or "")
    target_smiles = _canonical_smiles(dict(molecules.get(target_id) or {}).get("canonical_smiles"))
    if not target_id or not target_smiles:
        return None, {
            "reason": "final_route_critic_target_identity_missing",
            "route_family_id": route_id,
        }

    scoped_edges = {
        str(edge_id): dict(edges.get(str(edge_id)) or {})
        for edge_id in route_family_scoped_edge_ids(graph, family=route)
        if isinstance(edges.get(str(edge_id)), Mapping)
    }
    by_product: dict[str, list[dict[str, Any]]] = {}
    for edge in scoped_edges.values():
        product_id = str(edge.get("product_molecule_id") or "")
        if product_id:
            by_product.setdefault(product_id, []).append(edge)

    ordered_edges: list[dict[str, Any]] = []
    pending = deque([target_id])
    visited_molecules: set[str] = set()
    visited_edges: set[str] = set()
    while pending:
        molecule_id = pending.popleft()
        if molecule_id in visited_molecules:
            continue
        visited_molecules.add(molecule_id)
        for edge in sorted(
            by_product.get(molecule_id, ()),
            key=lambda row: str(row.get("edge_id") or ""),
        ):
            edge_id = str(edge.get("edge_id") or "")
            if not edge_id or edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)
            ordered_edges.append(edge)
            pending.extend(
                str(value) for value in edge.get("precursor_molecule_ids") or () if str(value)
            )
    if not ordered_edges:
        return None, {
            "reason": "final_route_critic_target_rooted_route_missing",
            "route_family_id": route_id,
        }

    steps: list[dict[str, Any]] = []
    for edge in ordered_edges:
        edge_id = str(edge.get("edge_id") or "")
        audit = dict(edge.get("reactionjson_audit") or {})
        mapped_product = str(
            audit.get("mapped_product_smiles") or edge.get("mapped_product_smiles") or ""
        )
        mapped_precursors = [
            str(value)
            for value in (
                audit.get("mapped_precursor_smiles") or edge.get("mapped_precursor_smiles") or ()
            )
        ]
        if not mapped_product or not mapped_precursors:
            proof_product, proof_precursors = _route_critic_edge_mapped_boundaries(edge)
            if proof_product and proof_precursors:
                mapped_product = proof_product
                mapped_precursors = proof_precursors
        precursor_ids = [str(value) for value in edge.get("precursor_molecule_ids") or ()]
        precursor_smiles = [
            _canonical_smiles(dict(molecules.get(precursor_id) or {}).get("canonical_smiles"))
            for precursor_id in precursor_ids
        ]
        mapped_by_identity: dict[str, list[str]] = {}
        for value in mapped_precursors:
            mapped_by_identity.setdefault(_canonical_smiles(value), []).append(value)
        aligned_mapped_precursors: list[str] = []
        for value in precursor_smiles:
            matches = mapped_by_identity.get(value) or []
            if matches:
                aligned_mapped_precursors.append(matches.pop(0))
        if (
            not _canonical_mapped_smiles(mapped_product)
            or _canonical_smiles(mapped_product) != _canonical_smiles(edge.get("product_smiles"))
            or not precursor_ids
            or any(not value for value in precursor_smiles)
            or len(aligned_mapped_precursors) != len(precursor_ids)
            or any(
                not _canonical_mapped_reactant_smiles(value) for value in aligned_mapped_precursors
            )
            or any(values for values in mapped_by_identity.values())
        ):
            return None, {
                "reason": "final_route_critic_mapped_boundary_incomplete",
                "route_family_id": route_id,
                "edge_id": edge_id,
                "edge_ids": [str(value.get("edge_id") or "") for value in ordered_edges],
            }
        origins = [
            dict(value) for value in edge.get("origin_records") or () if isinstance(value, Mapping)
        ]
        origin = next(
            (
                value
                for value in origins
                if route_id in {str(item) for item in value.get("canonical_route_family_ids") or ()}
            ),
            origins[0] if origins else {},
        )
        biocatalytic_steps = [
            dict(value)
            for value in edge.get("biocatalytic_steps") or ()
            if isinstance(value, Mapping)
        ]
        steps.append(
            {
                "step_id": str(origin.get("proposal_id") or edge_id),
                "product_smiles": _canonical_smiles(edge.get("product_smiles")),
                "precursor_smiles": precursor_smiles,
                "mapped_product_smiles": mapped_product,
                "mapped_precursor_smiles": aligned_mapped_precursors,
                "reaction_operations": [
                    dict(value)
                    for value in edge.get("reaction_operations") or ()
                    if isinstance(value, Mapping)
                ],
                "reaction_family": str(
                    origin.get("reaction_family")
                    or origin.get("transformation_hypothesis")
                    or edge.get("transformation_hypothesis")
                    or ""
                ),
                "transformation_hypothesis": str(
                    origin.get("transformation_hypothesis")
                    or edge.get("transformation_hypothesis")
                    or ""
                ),
                "condition_predictions": [
                    dict(value)
                    for value in edge.get("condition_predictions") or ()
                    if isinstance(value, Mapping)
                ],
                "execution_domain": str(
                    origin.get("execution_domain") or edge.get("execution_domain") or "chemical"
                ),
                "strategy_anchor": origin.get("strategy_anchor") is True,
                "strategy_milestone_index": int(origin.get("strategy_milestone_index") or 1),
                "strategy_id": str(origin.get("strategy_id") or route.get("strategy_id") or ""),
                "strategy_digest": str(
                    origin.get("strategy_digest") or route.get("strategy_digest") or ""
                ),
                "biocatalytic_step": (biocatalytic_steps[0] if biocatalytic_steps else {}),
            }
        )

    # Canonical edges own reaction-local mappings.  A final Route Critic,
    # however, receives several connected edges at once and therefore needs
    # one route-level namespace.  Replaying here is the authority boundary:
    # it carries parent maps into child products and deterministically moves
    # fresh atoms introduced on sibling branches when their local numbers
    # collide.  Without this projection a harmless Cl:37 / Br:37 reuse on
    # separate materialized edges looks like an element-transmutation defect
    # to the Critic, and a child row can retain an operation map that no
    # longer matches its parent-produced intermediate.
    if all(row.get("reaction_operations") for row in steps):
        try:
            route_state = RouteJSONCompiler().compile_route_graph_state(
                mapped_target_smiles=str(steps[0].get("mapped_product_smiles") or ""),
                steps=steps,
                minimum_depth=1,
                rebase_materialized_local_maps=True,
            )
        except ReactionJsonReplayError as exc:
            return None, {
                "reason": "final_route_critic_route_namespace_not_replayable",
                "route_family_id": route_id,
                "edge_ids": [str(value.get("edge_id") or "") for value in ordered_edges],
                "compiler_error": str(exc),
            }
        steps = RouteJSONCompiler.assemble_route(
            route_state.reactions,
            metadata=steps,
        )

    strategy_card = dict(route.get("strategy_card") or {})
    milestone_cards = tuple(
        dict(value)
        for value in route.get("strategy_milestone_cards") or ()
        if isinstance(value, Mapping)
    )
    chemistry_input = {
        "schema_version": "revision_bound_route_critic_input.v1",
        "target_smiles": target_smiles,
        "route_family_id": route_id,
        "strategy_card": strategy_card,
        "strategy_milestone_cards": list(milestone_cards),
        "steps": steps,
    }
    route_sha256 = _digest(chemistry_input)
    return (
        RevisionBoundRouteCriticContext(
            target_smiles=target_smiles,
            route_family_id=route_id,
            route_sha256=route_sha256,
            graph_revision=int(graph.get("revision") or 0),
            branch_index=_route_branch_index(route),
            edge_ids=tuple(str(edge.get("edge_id") or "") for edge in ordered_edges),
            steps=tuple(steps),
            strategy_card=strategy_card,
            strategy_milestone_cards=milestone_cards,
        ),
        {},
    )


def _route_branch_index(route: Mapping[str, Any]) -> int:
    """Recover the one-based Strategy alias as a zero-based audit index."""

    aliases = [
        str(value)
        for value in (
            route.get("route_family_id"),
            route.get("route_family_alias_override"),
            *(route.get("aliases") or ()),
        )
        if str(value)
    ]
    branch_number = next(
        (
            int(match.group(1))
            for alias in aliases
            if (match := re.search(r"codex:sequential:family:(\d+)$", alias))
        ),
        1,
    )
    return max(0, branch_number - 1)


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
    "FrontierBuilderContext",
    "NodeExpansion",
    "RevisionBoundRouteCriticContext",
    "SEQUENTIAL_STRATEGY_SEARCH_SCHEMA",
    "SequentialStrategyDirectorRunner",
    "compile_frontier_builder_context",
    "compile_revision_bound_route_critic_context",
]
