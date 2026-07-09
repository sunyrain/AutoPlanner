"""vNext route-level learning components for AutoPlanner.

The vNext layer is intentionally optional at runtime. Frozen one-step engines
remain the chemistry source of truth; these models calibrate candidate pools
and route search decisions behind feature flags.
"""

from cascade_planner.vnext.schema import VNEXT_SCHEMA_VERSION

_MODEL_EXPORTS = {
    "CandidatePoolCrossAttentionRanker",
    "RouteStateTransformer",
    "SearchPolicyNetwork",
    "StepEncoder",
}
_RUNTIME_EXPORTS = {"default_vnext_runtime", "vnext_candidate_weight"}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from cascade_planner.vnext import models

        value = getattr(models, name)
        globals()[name] = value
        return value
    if name in _RUNTIME_EXPORTS:
        from cascade_planner.vnext import runtime

        value = getattr(runtime, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "VNEXT_SCHEMA_VERSION",
    "StepEncoder",
    "CandidatePoolCrossAttentionRanker",
    "RouteStateTransformer",
    "SearchPolicyNetwork",
    "default_vnext_runtime",
    "vnext_candidate_weight",
]
