"""Declarative legacy harness tool registry without tool implementations."""
from __future__ import annotations

from typing import Any, Callable, Mapping


LegacyToolHandler = Callable[[Any, dict[str, Any]], dict[str, Any]]

LEGACY_LOCAL_TOOL_NAMES = (
    "apply_source_text_condition_repairs",
    "audit_route_and_extract_frontier",
    "build_analogical_retrosynthesis_hypotheses",
    "build_source_detail_curator_records",
    "compile_hybrid_route_set",
    "compile_source_detail_chain_route",
    "emit_final_verdict",
    "extract_pdf_literature_structures",
    "extract_visual_literature_chain",
    "resolve_literature_structure_task",
    "run_chemenzy",
    "run_guided_chemenzy_rerun",
    "run_open_structure_research_agent",
    "run_route_expansion_subgoal_search",
    "run_self_evo_replay_gate",
    "run_smiles_first_literature_workflow",
    "stitch_literature_chain_with_subgoal_route",
    "validate_artifact_bundle",
    "validate_literature_intermediate_chain",
)


def bind_legacy_tool_registry(
    handlers: Mapping[str, LegacyToolHandler],
) -> dict[str, LegacyToolHandler]:
    """Bind named implementations and fail if registry ownership drifts."""
    registry = {str(name): handler for name, handler in handlers.items()}
    missing = sorted(set(LEGACY_LOCAL_TOOL_NAMES) - set(registry))
    unexpected = sorted(set(registry) - set(LEGACY_LOCAL_TOOL_NAMES))
    if missing or unexpected:
        raise ValueError(
            "legacy_tool_registry_mismatch:"
            f"missing={','.join(missing)};unexpected={','.join(unexpected)}"
        )
    if any(not callable(handler) for handler in registry.values()):
        raise TypeError("legacy_tool_registry_handler_not_callable")
    return {name: registry[name] for name in LEGACY_LOCAL_TOOL_NAMES}


__all__ = [
    "LEGACY_LOCAL_TOOL_NAMES",
    "LegacyToolHandler",
    "bind_legacy_tool_registry",
]
