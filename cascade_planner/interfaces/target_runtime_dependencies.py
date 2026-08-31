"""Small dependency helpers for the V4 target runtime."""
from __future__ import annotations

from typing import Any, Mapping


TARGET_PROFILE_DEFAULTS = {
    "fast": {
        "steps": 6,
        "iterations": 100,
        "topk": 50,
        "timeout": 300.0,
        "workers": 1,
        "max_input_tokens": 90_000,
        "max_output_tokens": 22_000,
        "max_model_wall_time_s": 900.0,
        "max_director_wall_time_s": 600.0,
    },
    "standard": {
        "steps": 14,
        "iterations": 500,
        "topk": 100,
        "timeout": 1_200.0,
        "workers": 1,
        "max_input_tokens": 90_000,
        "max_output_tokens": 22_000,
        "max_model_wall_time_s": 900.0,
        "max_director_wall_time_s": 600.0,
    },
    "proof": {
        "steps": 20,
        "iterations": 1_500,
        "topk": 120,
        "timeout": 1_800.0,
        "workers": 2,
        "max_input_tokens": 1_200_000,
        "max_output_tokens": 200_000,
        "max_model_wall_time_s": 1_800.0,
        "max_director_wall_time_s": 1_800.0,
    },
    # Paper protocol is an explicit execution surface.  It is deliberately
    # separate from the ordinary ``standard``/``proof`` operational profiles
    # so a low-cost canary cannot silently masquerade as a SynthEx run.
    "paper_synthex": {
        "steps": 6,
        "iterations": 500,
        "topk": 120,
        "timeout": 1_200.0,
        "workers": 1,
        "max_input_tokens": 6_000_000,
        "max_output_tokens": 2_000_000,
        "max_model_wall_time_s": 70_200.0,
        "max_director_wall_time_s": 70_200.0,
    },
    # ``paper_matched_reach`` is the isolated reach-only contract.  Keep the
    # historical ``paper_synthex`` name available for old manifests, but do
    # not make new experiments depend on that compatibility alias.
    "paper_matched_reach": {
        "steps": 6,
        "iterations": 500,
        "topk": 120,
        "timeout": 1_200.0,
        "workers": 1,
        # SynthEx does not report an aggregate token ceiling.  The former
        # 1.2M guard is lower than 75 real Codex policy calls plus the
        # Strategy/Critic/Editor phases and therefore changed the algorithm by
        # stopping non-stock-closed branches at 20/21 calls.  Keep a generous
        # operational emergency ceiling; per-branch policy calls and per-call
        # timeouts are the paper-facing scientific limits. The output ceiling
        # also covers complete 25-step Editor documents and detailed per-step
        # Critic assessments across all six improvement rounds.
        "max_input_tokens": 6_000_000,
        "max_output_tokens": 2_000_000,
        "max_model_wall_time_s": 70_200.0,
        "max_director_wall_time_s": 70_200.0,
    },
    # Self-correcting sequential canary: reuse the same target-only
    # stock/search runtime without claiming the frozen paper protocol.
    "self_correcting_sequential": {
        "steps": 6,
        "iterations": 500,
        "topk": 120,
        "timeout": 1_200.0,
        "workers": 1,
        "max_input_tokens": 900_000,
        "max_output_tokens": 300_000,
        "max_model_wall_time_s": 18_000.0,
        "max_director_wall_time_s": 18_000.0,
    },
}


# One canonical authority for the paper-matched policy/search envelope.  CLI,
# HTTP request compilation and panel subprocess construction all consume this
# object so a benchmark cannot advertise 3x25 while executing a smaller path.
SYNTHEX_MATCHED_PROFILE_DEFAULTS = {
    "strategy_search_profile": "synthex_matched",
    "strategy_tree_engine": "aizynthfinder_mcts",
    # The matched arm gives all three Strategy Generators the same neutral
    # prior.  AutoPlanner's enzyme-biased proposal is measured separately.
    "strategy_portfolio_mode": "paper_independent",
    "target_chemenzy_baseline": False,
    "model": "gpt-5.6-sol",
    "reasoning_effort": "medium",
    "strategy_branches": 3,
    # Strategy cards are still selected serially so each later card can be
    # checked against the already accepted portfolio.  Once frozen, the three
    # private Route Builder states may advance concurrently.
    "strategy_branch_workers": 3,
    # The paper gives each independent Route Builder a maximum 25-step search
    # allowance. Its released prompt also permits an earlier stop when the
    # route is complete, the frontier is simple enough for explorative search,
    # or continuing would violate the synthesis constraints.
    "stop_on_first_stock_closed_branch": False,
    "node_expansions_per_branch": 25,
    # The released SynthEx/SyntheLite search configuration uses beam width 1
    # and ``max_actions: 1`` for each LLM expansion.  Keep that exact matched
    # arm separate from AutoPlanner's wider multi-candidate OR ablation below.
    # The LLM policy is executed inside AiZynthFinder's MCTS/UCB tree. The
    # paper-matched arm keeps top-1 actions; a wider sibling-action arm remains
    # a separately labelled enhanced ablation.
    "reactionjson_candidates_per_node": 1,
    "enhanced_or_reactionjson_candidates_per_node": 3,
    "route_local_repair_rounds": 6,
    # A 25-step Critic/Editor document now carries the exact host atom-map
    # namespace, DAG dependencies and conditions required by the paper's
    # Improvement stage. 24 kB forced those execution fields out of the model
    # input; 96 kB is the existing solver ceiling and preserves the full route.
    "max_node_prompt_bytes": 96_000,
    "node_call_timeout_s": 600.0,
    "critic_call_timeout_s": 600.0,
    # The shared envelope must cover 75 policy calls, up to one online
    # key-event Critic for every replayed policy candidate, transactional
    # Builder repair phases, receding-horizon Strategy review, and the final
    # Critic/Editor loop. Repair phases reuse the configured per-branch node
    # ceiling and settle against this same ledger; there is no private repair
    # quota. It is deliberately
    # not a private Critic quota: every actual call still settles against the
    # one global ledger, whose protected final-Critic balance is unchanged.
    # This aggregate is operational and is not a SynthEx-reported parameter.
    "max_model_invocations": 240,
    "max_input_tokens": 6_000_000,
    "max_output_tokens": 2_000_000,
    # SynthEx reports per-call and per-search limits, not one 1,800 s cap for
    # the complete multi-agent run.  This is a conservative host envelope for
    # Per-call timeout remains the scientific limit; this aggregate is merely
    # an operational ceiling and must not be presented as paper-reported.
    "max_model_wall_time_s": 70_200.0,
    # Allow native short-tail searches and host materialization to finish after
    # model work.  The paper does not publish an aggregate target wall clock,
    # so this is an explicit emergency cutoff rather than a matched metric.
    "max_run_wall_time_s": 86_400.0,
    "max_prompt_context_bytes": 96_000,
    "max_accepted_expansions": 96,
    "max_attempt_runs": 256,
    "max_total_tasks": 1_024,
    "max_atom_mapping_reactions": 81,
    "max_stock_molecules": 256,
    "short_tail_steps": 6,
    "short_tail_iterations": 500,
    "short_tail_timeout_s": 1_200.0,
    "short_tail_engine": "AiZynthFinder 4.4.1",
    "route_builder_max_steps": 25,
    # SynthEx is sequential at the Route Builder boundary: one open leaf, one
    # ReactionJSON edit, then deterministic host replay.  The Critic/Improvement
    # stage is different: it receives the accumulated RouteJSON as an editable
    # document and must not truncate an AiZ path by replacing it with one local
    # step.
    # This flag is an admission contract, not a request to make the LLM emit a
    # 25-step document in one response.  The Route Builder remains node-wise:
    # every policy call returns one host-replayed ReactionJSON edit and the
    # host compiles those edits into the complete RouteJSON document.
    "route_builder_complete_linear_route": True,
    "editor_route_mutations": True,
    # Reference facts from SynthEx.  The launch preflight binds the concrete
    # local index, validates these identity/count semantics and checks its
    # final content hash before any paid experiment starts.
    "paper_reference_stock_catalog_name": "ZINC+eMolecules",
    # The paper prints 39,684,411 combined input entries.  Exact audit of the
    # released source snapshots shows 39,478,827 unique full-InChIKeys; the
    # 205,584 difference is 205,490 duplicate valid eMolecules keys plus 94
    # invalid eMolecules rows.  Membership must bind the unique cardinality.
    "paper_reference_stock_member_count": 39_478_827,
    "paper_reference_stock_unique_member_count": 39_478_827,
    "paper_reference_stock_declared_entry_count": 39_684_411,
    "paper_reference_stock_zinc_unique_count": 17_422_831,
    "paper_reference_stock_emolecules_input_rows": 23_081_629,
    "paper_reference_stock_emolecules_valid_rows": 23_081_535,
    "paper_reference_stock_emolecules_unique_count": 22_876_045,
    "paper_reference_stock_cross_source_overlap_count": 820_049,
    "paper_reference_stock_redundant_or_invalid_rows": 205_584,
    "paper_reference_stock_identity_key": "full_inchikey",
}

# Backward-compatible name for callers that only consume ChemEnzy controls.
CHEMENZY_PROFILE_DEFAULTS = TARGET_PROFILE_DEFAULTS


def inventory_snapshot_builder(payload: Mapping[str, Any]) -> Any:
    path = str(payload.get("inventory_snapshot_path") or "").strip()
    if not path:
        return None
    from cascade_planner.interfaces.live_stock import FrozenInventorySnapshotBuilder

    return FrozenInventorySnapshotBuilder(path)


__all__ = [
    "CHEMENZY_PROFILE_DEFAULTS",
    "SYNTHEX_MATCHED_PROFILE_DEFAULTS",
    "TARGET_PROFILE_DEFAULTS",
    "inventory_snapshot_builder",
]
