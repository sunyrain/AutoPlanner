"""LLM-prior agent layer for AutoPlanner.

Agents in this package are deliberately outside the chemistry source-of-truth
path. They may propose priors or critiques, but validators and route exports own
the factual claims.
"""

from cascade_planner.agent.failure_policy import predict_failure_risk
from cascade_planner.agent.prior_generator import generate_strategic_prior
from cascade_planner.agent.route_critic import critique_route_payload

__all__ = ["generate_strategic_prior", "critique_route_payload", "predict_failure_risk"]
