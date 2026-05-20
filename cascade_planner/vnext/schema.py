"""Stable vNext data contract for route-level CASP learning."""
from __future__ import annotations

from typing import Any


VNEXT_SCHEMA_VERSION = "autoplanner.vnext.2026-05-07"

VNEXT_FILES = {
    "step_pairs": "step_pairs.jsonl",
    "candidate_pools": "candidate_pools.jsonl",
    "route_states": "route_states.jsonl",
    "search_transitions": "search_transitions.jsonl",
}

SOURCE_VALUES = [
    "retrochimera",
    "enzyformer",
    "v3_retrieval",
    "enzexpand",
    "fake",
    "candidate",
    "benchmark_gt",
    "planner_route",
    "curated_cascade_step",
    "unknown",
]

SOURCE_BUDGET_GROUPS = [
    "chemical",
    "enzymatic",
    "rhea_retrorules",
    "fallback",
]

ROUTE_FEATURE_NAMES = [
    "filled_route",
    "progressive_route",
    "route_solved",
    "strict_stock_solve",
    "main_chain_reduction",
    "leaf_reduction",
    "naturalness",
    "condition_success",
    "compatibility_success",
    "enzyme_evidence",
    "issue_count",
]

CANDIDATE_NUMERIC_FEATURES = [
    "candidate_score",
    "stock_fraction",
    "main_reduction",
    "has_ec",
    "has_evidence",
    "large_aux_penalty",
    "self_loop",
]

CANDIDATE_METADATA_FEATURES = [
    "rank_inverse",
    "rank_log",
    "has_gt",
    "has_ec",
    "has_type",
    "has_doi",
    "has_uniprot",
    "has_T",
    "has_pH",
    "has_T_and_pH",
    "T_scaled",
    "pH_scaled",
    "has_solvent",
    "has_catalyst",
    "has_enzyme_uid",
    "has_cofactor",
    "has_condition_match",
]

BOTTLENECK_LABELS = [
    "candidate_generator_reactant_miss",
    "candidate_generator_reaction_detail_miss",
    "selector_missed_exact_candidate",
    "selector_missed_gt_reactant_candidate",
    "route_composition_or_order_miss",
    "stock_dead_end",
    "condition_failure",
    "compatibility_failure",
    "depth_mismatch",
    "skeleton_type_mismatch",
    "no_professional_solved_route",
    "no_route_returned",
]

OPERATION_MODE_VALUES = [
    "one_pot_simultaneous",
    "one_pot_sequential_addition",
    "sequential_isolated",
    "continuous_flow",
    "telescoped_no_isolation",
    "compartmentalized",
    "other",
    "unknown",
]


def schema_manifest(*, source_pack: str, counts: dict[str, int], files: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "source_pack": source_pack,
        "files": files,
        "counts": counts,
        "contracts": {
            "step_pairs": {
                "purpose": "single-step product/reactant supervision for StepEncoder pretraining",
                "label": "continuous utility label from exact GT, selected route, or weak planner label",
            },
            "candidate_pools": {
                "purpose": "listwise candidate-pool calibration under route context",
                "candidate_labels": "per-candidate utility labels aligned to step_pairs",
            },
            "route_states": {
                "purpose": "global route state/value/compatibility/bottleneck supervision",
                "labels": ["solved", "stock_closed", "progressive", "compatibility", "bottleneck_labels"],
            },
            "search_transitions": {
                "purpose": "candidate-pool search-distillation transitions for SearchPolicyNetwork and future offline RL",
            },
        },
    }
