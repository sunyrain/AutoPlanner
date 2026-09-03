"""Research-only DeepSeek-mediated route-tree control for AutoPlanner.

The canonical V4 runtime must not import this package. Route-tree experiments
load it through the explicit research live-benchmark worker; environment
variables configure only code already running inside this namespace.
"""

from cascade_planner.research.autoplannrellm.controller import (
    DeepSeekSelectionController,
)
from cascade_planner.research.autoplannrellm.proposals import append_llm_candidate

__all__ = ["DeepSeekSelectionController", "append_llm_candidate"]
