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
}


# One canonical authority for the paper-matched policy/search envelope.  CLI,
# HTTP request compilation and panel subprocess construction all consume this
# object so a benchmark cannot advertise 3x25 while executing a smaller path.
SYNTHEX_MATCHED_PROFILE_DEFAULTS = {
    "strategy_search_profile": "synthex_matched",
    "target_chemenzy_baseline": False,
    "model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "strategy_branches": 3,
    "node_expansions_per_branch": 25,
    "route_local_repair_rounds": 6,
    "max_node_prompt_bytes": 24_000,
    "node_call_timeout_s": 600.0,
    "critic_call_timeout_s": 600.0,
    "max_model_invocations": 120,
    "max_input_tokens": 1_200_000,
    "max_output_tokens": 200_000,
    # Keep the matched fixed-cutoff contract at 30 minutes.  Critic/Editor
    # calls are reserved inside this envelope; increasing the cutoff here
    # would silently change the benchmark being compared with SynthEx.
    "max_model_wall_time_s": 1_800.0,
    "max_prompt_context_bytes": 64_000,
    "max_accepted_expansions": 96,
    "max_attempt_runs": 256,
    "max_total_tasks": 1_024,
    "max_atom_mapping_reactions": 81,
    "max_stock_molecules": 256,
    "short_tail_steps": 6,
    "short_tail_iterations": 500,
    "short_tail_timeout_s": 1_200.0,
    "stock_catalog_name": "ZINC+eMolecules",
    "stock_member_count": 39_684_411,
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
