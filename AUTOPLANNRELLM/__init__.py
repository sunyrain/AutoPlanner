"""AUTOPLANNRELLM: AutoPlanner with DeepSeek-mediated route-tree control.

The package is intentionally separate from ``cascade_planner``. The base
AutoPlanner runtime imports it only behind explicit environment gates.
"""

from AUTOPLANNRELLM.controller import DeepSeekSelectionController
from AUTOPLANNRELLM.proposals import append_llm_candidate

__all__ = ["DeepSeekSelectionController", "append_llm_candidate"]
