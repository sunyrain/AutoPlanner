"""Compile canonical-frontier delegation into ChemEnzy's executable policy."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Protocol, Sequence

from cascade_planner.agent.chem_enzy_policy import ChemEnzySearchPolicy, RerunBudget


class GuidedChemEnzyRequest(Protocol):
    target_smiles: str
    target_name: str
    route_family_ids: Sequence[str]
    retron_hints: Sequence[str]
    forbidden_smiles: Sequence[str]

    def to_dict(self) -> dict[str, Any]: ...


def guided_native_search_policy(
    request: GuidedChemEnzyRequest,
    *,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate one Codex-selected frontier without injecting a reaction."""

    digest = sha256(
        json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    policy = ChemEnzySearchPolicy(
        policy_id=f"v4-guided-{digest}",
        operator_id=f"v4-frontier-{digest}",
        case_id=request.target_name or f"frontier-{digest}",
        evidence_refs=[
            *(f"route-family:{value}" for value in request.route_family_ids),
            *(f"retron:{value}" for value in request.retron_hints),
        ]
        or [f"canonical-frontier:{digest}"],
        terminal_blacklist=list(request.forbidden_smiles),
        preferred_subgoal={
            "target_smiles": request.target_smiles,
            "preferred_retrons": list(request.retron_hints),
        },
        source_budget={
            "preferred_retrons": list(request.retron_hints),
            "reaction_and_retron_priors_are_advisory_only": True,
        },
        rerun_reason="Codex-selected canonical subtarget expansion",
        budget=RerunBudget(
            max_reruns=1,
            max_iterations=max(1, int(limits.get("max_iterations") or 1)),
            max_depth=max(1, int(limits.get("max_steps") or 1)),
            expansion_topk=max(1, int(limits.get("expansion_topk") or 1)),
        ),
        mode="guided",
        compiler_metadata={
            "source": "v4_canonical_frontier",
            "not_raw_reaction_injection": True,
        },
    )
    return policy.to_dict()


__all__ = ["GuidedChemEnzyRequest", "guided_native_search_policy"]
