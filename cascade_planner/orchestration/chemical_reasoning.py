"""Compact chemical reasoning guidance and binding of model-proposed causal links.

Critic artifacts own the links. They describe chemical requirements, not new
Host facts or admission gates; per-step verdicts remain the only review status.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


CATALYSIS_PLANNING_GUIDANCE = (
    "Apply the same planning-level standard to chemical and biological catalysis. "
    "During route design, do not query enzyme databases or search for enzyme identities; "
    "propose chemically credible catalytic hypotheses from the supplied context and chemical knowledge. "
    "Identify an existing-catalyst proposal, screening hypothesis or catalyst-development "
    "dependency once in conditions or risks. State the required selectivity and "
    "preserved functionality; include compatible cofactor/regeneration needs where relevant, "
    "without assuming every member of an enzyme class uses the same cofactor. "
    "Do not invent variants, measured outcomes or precise operating windows. Unsupported "
    "activity/selectivity remains uncertain/evidence_missing; a catalyst name does not resolve it. "
    "Use proposal_underspecified only for an absent design choice whose plausible alternatives "
    "change the feasibility or compatibility judgment; name the consequence and smallest choice "
    "needed. Do not request SOP precision, new measurements or successful engineered sequences "
    "as clarification. A proposed choice is not evidence that it works. "
    "For a claimed non-isolative handoff, choose a plausible vessel/feed sequence and state "
    "whether the preceding catalyst remains, is removed or is inactivated; account for residual "
    "substrate entering the next catalyst system. Reconcile this design with the supplied consumer. "
    "Compatible turnover rates, intermediate lifetime and measured performance remain experimental "
    "dependencies. A missing handoff design is proposal_underspecified even when catalyst "
    "performance is also evidence_missing. Report the specific dependency without repeating "
    "generic evidence disclaimers or unrelated alternative-route comparisons."
)


STRATEGY_DIVERSITY_GUIDANCE = (
    "Judge strategic differences against the supplied task's decisive bottleneck. For de novo "
    "scaffold synthesis, distinguish backbone construction or reorganization. For process or "
    "selectivity development, distinct stereochemical origins, fragment-union order, or ways of "
    "avoiding an unstable intermediate can justify separate strategies using the same core, "
    "provided its construction or credible supply burden is explicit. Reagent renaming alone "
    "is not a distinct strategy. Compare the strongest alternative on the same standard of "
    "precursor supply and unverified catalytic requirements; retain it separately when it offers "
    "a promising different solution, without filling a quota."
)


STRATEGY_PRIORS = (
    "Reason backward from the hardest structural or selectivity commitment, then test it forward. "
    "Consider retrons, latent symmetry, convergent fragment union, polarity matching or umpolung, "
    "and whether a temporary functional group or stereochemical relay makes the key event simpler. "
    "Treat these as alternatives supported by the actual scaffold, never as mandatory named reactions. "
    "Use critical_assumption for the weakest substrate-specific prerequisite and critic_checkpoint "
    "for the first event that can falsify it: one graph transformation, not a downstream handoff "
    "or a whole-route process checklist. Keep those requirements in conditions and task-level "
    "evaluation. Distinguish a useful strategic intermediate from an "
    "easy starting material; reducing displayed step count by hiding its synthesis is not progress. "
    + CATALYSIS_PLANNING_GUIDANCE
)

DEPENDENCY_CRITIC_GUIDANCE = (
    "Think in chemical prerequisites as well as reactions. For the route-defining or problematic "
    "events, trace which earlier forward steps establish or preserve the required reactive handle, "
    "polarity, protection state, stereochemical relationship, or cyclization geometry. A locally "
    "correct preparation can supply the wrong state for its consumer. Compare the intended event "
    "with its strongest substrate-specific competitor. Report at most six consequential links in "
    "chemical_dependencies, each with consumer_review_slot, prerequisite_review_slots, and a concise "
    "requirement. Prerequisite slots may pass locally; do not relabel them reject merely to include "
    "them in a coordinated repair. Include only supplied steps and distinguish physical support "
    "from an intended product annotation. The links do not establish evidence or add a second verdict."
)

CHEMICAL_REVIEW_SCOPE = (
    "Evaluate chemistry only. Stock membership, stock closure and search termination are "
    "computed separately by the Host from the bound catalog and actual route leaves. "
    "Do not infer availability from molecular complexity, missing metadata or earlier Critic prose, "
    "and do not state stock-closed, not stock-closed, commercially available or unavailable in "
    "the chemical evaluation. You may identify the synthetic burden of a supplied advanced "
    "starting material without asserting its inventory status. Stock uncertainty alone must not "
    "change a step verdict, create a chemical blocker or trigger Editor repair. This also "
    "applies to repair_actions: do not call a supplied precursor an unsolved preparative "
    "boundary solely because its preparation is outside the displayed route."
)

EDITOR_INTENT_GUIDANCE = (
    "Edit the chemical intention: choose the steps whose transformation or delivered molecular state "
    "must be reconsidered, and state the property the replacement must achieve. Use "
    "chemical_dependencies to trace a failing consumer back to the preparations that determine its "
    "input. A locally passing preparation may need to change. Either retain its exact product and "
    "choose a compatible consumer, or include the relevant preparations in change_step_ids. "
    "When a replacement no longer uses a reagent-supply branch, include every reaction in that "
    "obsolete branch in change_step_ids, even if it passes locally. Omitted branches remain "
    "mandatory reconnections. Terminal inputs emitted only by removed reactions are optional "
    "old starting points, not obligations to synthesize obsolete reagents. Any new terminal "
    "starting point needs Host-confirmed stock membership or explicit upstream synthesis. "
    "Check the replacement forward through its retained target-side consumer, not only to the first "
    "reconnected molecule. Prefer the smallest causal intervention; do not fix stereochemical "
    "incompatibility by renaming a reaction or changing a catalyst without a chemical rationale. "
    "Preserve atoms through chemically atom-retaining steps; the Host consistently translates "
    "isomorphic suffix boundaries and their retained reactions. Do not add oxygen exchange or "
    "other chemistry solely to reproduce old atom-map labels. "
    "When previous_repair is supplied, use its concrete failure and recovery_request to reconsider "
    "the repair boundary. Include retained preparations only when their delivered state must change; "
    "do not repeat the same scope and goal without addressing why it failed."
)


def repair_requirements_for_review(pending: Mapping[str, Any]) -> list[str]:
    """Project the original concern without carrying stale step IDs or verdicts.

    Rebuilt reactions have new identities. An old ID disappearing cannot tell
    the next Critic whether the chemical contradiction disappeared with it.
    These are prior hypotheses to reassess, never a second acceptance rule.
    """
    original = dict(pending.get("original_critique") or {})
    selected = set(pending.get("selected_blocker_step_ids") or ())
    requirements = [str(pending.get("repair_goal") or "").strip()]
    requirements.extend(str(value).strip() for value in pending.get("active_constraints") or ())
    for dependency in original.get("chemical_dependencies") or ():
        if isinstance(dependency, Mapping) and dependency.get("consumer_step_id") in selected:
            requirements.append(str(dependency.get("requirement") or "").strip())
    for assessment in original.get("step_assessments") or ():
        if isinstance(assessment, Mapping) and assessment.get("step_id") in selected:
            requirements.extend(str(reason).strip() for reason in assessment.get("reasons") or ())
    return list(dict.fromkeys(value for value in requirements if value))


def bind_chemical_dependencies(
    values: Any, bindings: Iterable[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Translate opaque review slots once, retaining valid step judgments on bad links.

    Unknown/empty/self-referential links are discarded with diagnostics. This
    deliberately does not promote a semantic model hypothesis to graph truth.
    The repair compiler later owns the exact connected region and boundaries.
    """
    by_slot = {row["review_slot"]: row["step_id"] for row in bindings}
    if values is None:
        return [], []  # Historical artifacts predate causal-link output.
    if not isinstance(values, list):
        return [], [{"reason": "chemical_dependencies_not_a_list"}]
    bound, diagnostics = [], []
    seen = set()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            diagnostics.append({"index": index, "reason": "chemical_dependency_not_an_object"})
            continue
        consumer = str(raw.get("consumer_review_slot") or "")
        prerequisites = raw.get("prerequisite_review_slots")
        requirement = str(raw.get("requirement") or "").strip()
        if (consumer not in by_slot or not isinstance(prerequisites, list)
                or not prerequisites or not requirement
                or any(not isinstance(slot, str) or slot not in by_slot or slot == consumer
                       for slot in prerequisites)):
            diagnostics.append({"index": index, "reason": "chemical_dependency_binding_invalid"})
            continue
        producer_ids = tuple(dict.fromkeys(by_slot[slot] for slot in prerequisites))
        key = (by_slot[consumer], tuple(sorted(producer_ids)), requirement)
        if key in seen:
            continue
        seen.add(key)
        bound.append({"consumer_step_id": by_slot[consumer],
                      "prerequisite_step_ids": list(producer_ids), "requirement": requirement})
    return bound, diagnostics
